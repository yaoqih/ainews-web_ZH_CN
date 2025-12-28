---
companies:
- cohere
- lightblue
- openai
- mistral-ai
- nvidia
- amd
- hugging-face
- ollama
date: '2024-04-02T21:04:12.327421Z'
description: '以下是为您翻译的中文内容：


  **RAGFlow** 已开源，这是一款具备深度文档理解能力的 RAG（检索增强生成）引擎，支持 **16.3k 上下文长度**及自然语言指令。Lightblue
  发布了拥有 **520 亿参数**的 MoE（混合专家）模型 **Jamba v0.1**，但用户反馈褒贬不一。**Cohere** 开发的 **Command-R**
  模型现已在 Ollama 库中上线。


  对 **GPT-3.5-Turbo** 架构的分析显示，其参数量约为 **70 亿**，嵌入维度为 **4096**，级别与 OpenChat-3.5-0106
  和 Mixtral-8x7B 相当。研究表明，包括 **GPT-4** 在内的 AI 聊天机器人在说服力辩论中表现优于人类。此外，**Mistral-7B**
  在处理数学谜题时犯了一些有趣的错误。


  硬件方面，有人以 5.8 万美元的折扣价购入了一台配备 8 块 H100 GPU 的 **HGX H100 640GB** 服务器；另外还有关于 **Epyc
  9374F** 与 **Threadripper 1950X** 在大语言模型（LLM）推理性能上的对比。针对本地大模型的 GPU 推荐主要集中在显存（VRAM）和推理速度上，用户正在测试
  **4090 GPU** 与 **Midnight-miqu-70b-v1.0.q5_k_s** 模型的组合。最后，**Stable Diffusion** 正在影响玩家的游戏习惯，而一项
  AI 艺术评估显示，评价结果存在偏向“人类创作”标签的偏见。'
id: 673a1f3a-267d-4d1b-a03d-73462dbe536b
models:
- jamba-v0.1
- command-r
- gpt-3.5-turbo
- openchat-3.5-0106
- mixtral-8x7b
- mistral-7b
- midnight-miqu-70b-v1.0.q5_k_s
original_slug: ainews-not-much-happened-today-8015
people: []
title: 今天没什么事。
topics:
- rag
- mixture-of-experts
- model-architecture
- model-analysis
- debate-persuasion
- hardware-performance
- gpu-inference
- cpu-comparison
- local-llm
- stable-diffusion
- ai-art-bias
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月1日至4月2日的 AI 新闻。我们为您检查了 5 个 subreddit、[364 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 26 个 Discord（382 个频道和 4481 条消息）。预计节省阅读时间（以 200wpm 计算）：**463 分钟**。


所以你有时间：

- [观看来自 3B1B 的 30 分钟 GPT 入门介绍](https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s)
- [观看来自 Soheil Feizi 的 4 小时完整 LLM 入门介绍](https://twitter.com/FeiziSoheil/status/1774833586736189911)
- [尝试一个在 SWE-bench 上得分 12.29% 的开源 Devin 竞争对手](https://twitter.com/jyangballin/status/1775114444370051582?t=90xQ8sGy63D2OtiaoGJuww)

祝贺 [Logan 加入 Google](https://twitter.com/officiallogank/status/1775222819439149424?t=6FDPaNxZcbSsELal6Sv7Ug)。

---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取尚未实现，但即将推出。


**开源模型与库**

- **RAGFlow 开源**：RAGFlow 是一个基于深度文档理解的 RAG 引擎，利用了 pip-library-etl-1.3b，现已开源。主要特性包括 16.3k 上下文长度、自动化库解析、示例微调、静态和动态函数分析，以及自然语言指令支持。([链接](https://www.reddit.com/r/MachineLearning/comments/1bt0vg9/n_open_source_13b_multicapabilities_model_and/), [链接](https://www.reddit.com/r/MachineLearning/comments/1bt1ky8/p_ragflow_the_deep_document_understanding_based/))
- **Jamba v0.1 发布**：Lightblue 发布了 Jamba v0.1，这是一个拥有 52B 参数、采用 Apache 许可证的 Mixture-of-Experts (MoE) 架构模型。然而，一些用户发现其输出具有重复性，且表现低于预期。([链接](https://huggingface.co/lightblue/Jamba-v0.1-chat-multilingual), [链接](https://www.reddit.com/r/LocalLLaMA/comments/1btg38m/thoughts_on_jamba/)) 
- **来自 Cohere 的 Command-R**：Command-R 是 Cohere 推出的一款模型，可在 ollama 库中使用，据报道效果相当不错，但讨论度不高。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1bt9i4o/why_is_no_one_talking_about_commandr_from_cohere/))

**模型性能与能力**

- **GPT-3.5-Turbo 架构细节**：对 GPT-3.5-Turbo 的 logits 分析估计其 embedding 大小约为 4096，参数量约为 70 亿，与最近的开源模型如 OpenChat-3.5-0106 和 Mixtral-8x7B 的规模一致。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1btpk4h/logits_of_apiprotected_llms_leak_proprietary/))
- **AI 聊天机器人在辩论中击败人类**：在一项研究中，AI 聊天机器人在有争议话题的辩论中比人类更具说服力。与人类辩论者相比，当受到 GPT-4 的挑战时，人们更有可能改变主意。([链接](https://www.newscientist.com/article/2424856-ai-chatbots-beat-humans-at-persuading-their-opponents-in-debates/))
- **Mistral-7B 犯了有趣的错误**：Mistral-7B 在回答一个关于数水果的简单数学谜题时犯了有趣的错误，未能理解鞋子不是水果，并给出了不一致的答案。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1btc98u/i_have_4_oranges_1_apple_and_2_pairs_of_shoes_i/))

**硬件与性能**

- **折扣 HGX H100 机器**：有人在 eBay 上以仅 5.8 万美元的价格抢购了一台配备 8 张 H100 的全新 HGX H100 640GB 机器，远低于 27 万美元的零售价。([链接](https://i.redd.it/i8z79zdn7wrc1.png))
- **用于 LLM 推理的 Epyc 与 Threadripper**：发布了 Epyc 9374F 和 Threadripper 1950X CPU 在 LLM 推理任务上的性能对比。([链接](https://www.reddit.com/gallery/1bt8kc9))
- **本地 LLM 的 GPU 推荐**：针对在 Windows 上本地运行 7-13B 参数 LLM 的最佳 GPU 选择（GTX 1070, RTX 3050 8GB, RTX 2060S）征求了建议，关键考虑因素是 VRAM 和推理速度。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1btbmmv/which_gpu_can_run_llm_locally_faster_gtx_1070_or/))
- **使用 4090 GPU 优化本地 LLM**：一位拥有 4090 GPU 和 64GB RAM 的用户询问了用于无审查角色扮演/聊天机器人的最佳本地 LLM 推荐，发现 Midnight-miqu-70b-v1.0.q5_k_s 的速度太慢，仅为 0.37 it/s。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1bta2un/4090_64gb_ram_best_local_llm_for_uncensored_rpchat/))

**Stable Diffusion 与图像生成**

- **SD 减少了游戏时间**：Stable Diffusion 减少了一些用户的游戏时间，他们现在更倾向于利用生成式 AI 探索自己的创造力。([链接](https://www.reddit.com/r/StableDiffusion/comments/1bt03v1/has_sd_cut_down_your_gaming_time/))
- **评估 AI 艺术时的偏见**：在一项研究中，当人们认为艺术作品是由人类创作时，他们更倾向于 AI 生成的艺术，但他们很难分辨出其中的区别，这表明在评估 AI 艺术时存在偏见。([链接](https://www.sciencenorway.no/art-artificial-intelligence/people-liked-ai-art-when-they-thought-it-was-made-by-humans/2337417))
- **区域提示词实验**：使用 A8R8、Forge 以及分叉的 Forge Couple 扩展进行的区域提示词（Regional prompting）实验允许对图像生成进行更细粒度的控制，新界面支持动态注意力区域、蒙版绘制和提示词权重，以最大限度地减少泄露。([链接](https://www.reddit.com/r/StableDiffusion/comments/1btrf4p/part_2_experimenting_with_regional_prompting_with/))
- **为 LoRA 选择最佳 Epoch**：关于在训练 LoRA 模型时如何选择最佳 Epoch 的问题被提出，因为有时 30% 训练进度的结果看起来比 100% 更好。寻找合适的设置来重现 LoRA 捕捉到的美学也需要不断的尝试。([链接](https://www.reddit.com/r/StableDiffusion/comments/1btld1v/no_one_ever_discusses_the_postlora_settings_why/))
- **在 SD 中重现 MidJourney 风格**：一位用户正在学习在 Stable Diffusion 中重现 MidJourney 风格，以获得更多的控制力和一致性，并寻求关于对一张复古暹罗猫图像进行逆向工程的建议。([链接](https://i.redd.it/lxkfcqjuywrc1.jpeg))

**杂项**

- **Google AI 负责人谈论 AI 炒作**：Google 的 AI 负责人表示，投入到 AI 的数十亿美元意味着“大量的炒作，可能还有一些欺诈”，但我们可能正处于一场新科学复兴的开端。他认为 AGI 在未来十年内出现的概率为 50%。([链接](https://www.reddit.com/r/LocalLLaMA/comments/1bt7bjf/googles_ai_chief_says_the_billions_going_into_ai/))
- **对 OpenAI 统治地位的沮丧**：OpenAI 在工作场所的无处不在让一些人感到沮丧，他们反对其封闭模型、利用网页抓取数据获利以及背离开源原则。然而，为了维持生计，人们面临着必须使用它的压力。([链接](https://www.reddit.com/r/MachineLearning/comments/1bt9y8p/d_cant_escape_openai_in_my_workplace_anyone_else/))
- **一键式图像打标应用**：一款免费的 Windows 应用被开发出来，可以使用 GPT vision 一键重命名图像/GIF 并添加相关的元数据。([链接](https://v.redd.it/0x2h6v29tzrc1.gif))
- **DagsHub Storage Buckets 发布**：DagsHub 发布了 DagsHub Storage Buckets，这是一个与 Google Colab 集成的 S3 兼容的 Google Drive 替代方案，旨在为 ML 工作流提供可扩展的存储。([链接](https://www.reddit.com/r/MachineLearning/comments/1bt5hxw/p_scalable_and_mloriented_google_drive/))

**梗与幽默**

- **Stale Diffusion 论文**：一篇幽默的 "Stale Diffusion" 论文提议使用传统方法生成超现实的 5D 电影。作者们感叹该论文甚至被一些不严肃的场合拒绝了。([链接](https://www.reddit.com/r/MachineLearning/comments/1bt9u0o/p_stale_diffusion_hyperrealistic_5d_movie/))
- **OpenAI 移除 Sam Altman 的梗图**：一张梗图嘲讽了 OpenAI 移除了 Sam Altman 对其创业基金的所有权。([链接](https://i.redd.it/5yv7ah8pxzrc1.png))
- **“我能行”梗图**：一张梗图描绘了某人面对未知挑战时自信地声称自己“能行（take it）”。([链接](https://i.redd.it/lrakavx6cvrc1.png))


# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**AI 模型与架构**

- **DBRX**: [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1774916729149513956) 指出 DBRX 是 **HuggingFace 上最新 WildBench 排行榜中排名第一的开源模型**，它在 12T tokens 的高质量数据上训练而成，并采用了细粒度 MoE、GPT-4 tokenizer 以及架构修改等效率提升手段。
- **Jamba**: [@AI21Labs](https://twitter.com/AI21Labs/status/1774824070053331093) 发布了 Jamba 白皮书，详细介绍了 **交织了 Mamba、Transformer 和 MoE 的新型混合 SSM-Transformer 架构**。[@amanrsanger](https://twitter.com/amanrsanger/status/1774928438039810475) 指出 Jamba 的 KV cache 比纯 Transformer 小 8 倍，但在长上下文生成时需要更多内存。
- **Gecko**: [@kelvin_guu](https://twitter.com/kelvin_guu/status/1774855490687918561) 分享了 Gecko 是 **Massive Text Embedding Benchmark (MTEB) 上 768 维以下最强的模型**，可在 Google Cloud 上用于 RAG、检索、向量数据库等。
- **DenseFormer**: [@fchollet](https://twitter.com/fchollet/status/1774843303420420382) 强调了 DenseFormer，它在 **每个 Transformer 块中对所有先前块的输出进行加权平均**，通过稀疏化策略提高性能并避免 IO 瓶颈。

**检索增强生成 (RAG)**

- **使用 LlamaParse 和本地模型的 RAG**: [@llama_index](https://twitter.com/llama_index/status/1774832426000515100) 分享了关于 **使用 LlamaParse 和本地模型构建高级 PDF RAG** 的教程，使用了 @GroqInc、@qdrant_engine 的 FastEmbed 以及 flag-embedding-reranker 等工具来实现高效的 RAG 设置。
- **使用 Chroma 的 RAG**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1774919695373603040) 指出如果检索到的信息具有相关性，RAG 就会非常有效，而 "Advanced Retrieval for AI with @trychroma" 课程教授了 **提高检索相关性的技术**。
- **RAFT**: [@llama_index](https://twitter.com/llama_index/status/1774814982322172077) 正在举办一场关于 **检索增强微调 (RAFT)** 的网络研讨会，邀请了 RAFT 的主要共同作者 @tianjun_zhang 和 @shishirpatil_ 讨论微调和 RAG。
- **RAGFlow**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1774890566179774733) 分享了 RAGFlow，这是一个 **基于深度文档理解的 RAG 引擎**，目前已开源并支持本地部署的 LLM。

**工具与基础设施**

- **Keras 3 + JAX**: [@fchollet](https://twitter.com/fchollet/status/1774843979869343788) 对 Keras 3 + JAX 与流行的 HuggingFace 模型进行了基准测试，结果显示有 **1.5-3 倍的加速**，并指出 Keras 利用编译器来提升性能，因此用户可以编写简单、易读的代码。
- **Instructor**: [@jxnlco](https://twitter.com/jxnlco/status/1774813661900558442) 发布了 instructor 1.0.0，具有 **完善的自动补全、针对 partials、iterables 和原始响应的辅助方法**，同时保留了简单的 instructor.from_openai 和 instructor.patch API。
- **LangChain 金融助手**: [@virattt](https://twitter.com/virattt/status/1774909569723932850) 在 **LangChain 金融助手中添加了受巴菲特启发的工具**，如所有者盈余、ROE、ROIC 计算，并计划下一步添加基于价格的工具。
- **FastEmbed**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1774723490567860634) 宣布 FastEmbed 现在允许使用 SPLADE++ 模型 **生成高效、可解释的稀疏向量嵌入 (sparse vector embeddings)**。

**研究与技术**

- **文本转图像模型中的空间一致性**: [@RisingSayak](https://twitter.com/RisingSayak/status/1775018332191617436) 分享了一篇研究 **T2I 扩散模型中空间一致性** 的论文，通过系统的重标注（re-captioning）方法和 SPRIGHT 数据集对其进行了改进。
- **Sequoia**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) 分享了关于 Sequoia 的论文，这是一种 **硬件感知的投机解码 (speculative decoding) 算法，可将 LLM 推理速度提高 10 倍**，它根据可用硬件优化了投机 token 的存储。
- **从 LLM 中窃取信息**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) 指出一篇论文显示，可以 **利用 LLM API 的 logprobs 来提取模型信息**，如隐藏层维度或 token 嵌入。
- **时间对齐 (Temporal Alignment)**: [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1774858806817906971) 分享了一篇探索 **将 LLM 的知识对齐到特定时间点** 以创建时间锚定 (temporal grounding) 的论文。
- **预训练数据**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1774817920704512013) 强调了一篇综述论文，该论文汇总了 **创建高质量 LLM 预训练数据集的最佳实践**，因为越来越多的研究人员开始分享预训练数据构建的细节。

**梗与幽默**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1774709480116060240) 调侃道，**研究是“由宇宙本身设计的永恒且辉煌的数独或填字游戏流”**，而解决方案则累积在“人类知识的堆垛”之上。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1774703886558781763) 嘲讽了**美国人通过将事物与眼前视觉范围内的东西进行比较，从而发明出“离奇的非标准计量单位”的能力**。
- [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1774821522932117613) 分享了一个**关于 AI 解决问题的梗图**，图片展示了一个训练不佳的 GAN。
- [@suno_ai_](https://twitter.com/suno_ai_/status/1774857974260855071) 玩笑地介绍了一种**由 AI 生成的新频率 SuNoise™**，这种频率“以前被认为超出了人类听力范围”，只有 2.5% 的人能听到。
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1774885274809721140) 在**愚人节引用了一条史蒂夫·乔布斯的假名言**：“不要相信你在互联网上读到的一切”。

---

# AI Discord 回顾

> 摘要之摘要的摘要

- **Claude 3 Haiku 作为高性价比的 Opus 替代方案令人印象深刻**：体量更小、价格更低的 **Claude 3 Haiku** 模型因其有效的推理能力和对脑筋急转弯问题的处理能力而引发热议，在 Perplexity AI 中被视为 Opus 的高性价比替代方案。讨论还集中在 Perplexity 引入广告的潜在计划，以及相比 All 模式，用户更倾向于使用 Writing 专注模式以获得更简洁的 LLM 交互。([Perplexity AI Discord](https://discord.com/channels/1047197230748151888))

- **Gecko 和 Aurora-M 推动文本嵌入和多语言 LLM 的边界**：新款 **Gecko** 模型在 Massive Text Embedding Benchmark (MTEB) 上表现强劲，并可能加速 diffusion model 的训练，详情见其 [Hugging Face 论文](https://huggingface.co/papers/2403.20327)和 [arXiv 摘要](https://arxiv.org/abs/2403.20327)。同时，拥有 15.5B 参数的 **Aurora-M** 模型专注于多语言任务，已处理超过 2 万亿个训练 token，正如 [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) 和 [arXiv](https://arxiv.org/abs/2404.00399) 上所强调的那样。([LAION Discord](https://discord.com/channels/823813159592001537))

- **高效微调技术引发辩论**：Unsloth AI 社区的对话围绕数据集拆分策略、稀疏微调 (SFT) 与 QLora 等量化方法的有效性，以及模型 pre-training 的高昂成本展开。成员们还强调了建立强大检测系统以打击 AI 滥用，并保护 Discord 服务器免受恶意机器人和诈骗侵害的必要性。([Unsloth AI Discord](https://discord.com/channels/1179035537009545276))

- **Stable Diffusion 社区期待 SD3 并应对模型挑战**：Stable Diffusion 社区正热切期待 **Stable Diffusion 3 (SD3)** 在 4-6 周内的发布，同时也正致力于解决使用 Adetailer 和各种 embedding 工具渲染面部及手部细节的挑战。讨论涉及 AI 发展的飞速步伐、伦理考量（如使用专业艺术作品进行训练），以及未来 SD 版本潜在的显存需求。([Stability.ai Discord](https://discord.com/channels/1002292111942635562))

- **Mojo 24.2 引入 Python 友好特性，Tensor 讨论升温**：**Mojo Programming Language** 社区因 **Mojo 24.2** 的发布而沸腾，该版本带来了一系列 Python 友好特性和增强功能。讨论深入探讨了 Mojo 对并行性、值类型（value types）的处理以及 tensor 性能优化。Mojo 中 **MAX Engine 和 C/C++ interop** 的发布也因其简化 **Reinforcement Learning (RL) Python 训练**的潜力而引发关注。([Modular Discord](https://discord.com/channels/1087530497313357884))

- **Tinygrad 努力应对 AMD GPU 的不稳定性及文化阻力**：tinygrad 社区对使用 **AMD GPU** 时出现的严重系统不稳定性表示沮丧，强调了内存泄漏和不可恢复错误等问题。人们对 AMD 解决底层问题的承诺表示怀疑，并呼吁提供开源文档和现代软件实践。讨论还涉及了变通策略，以及 AMD 在软件和固件方法上进行根本性文化转变的必要性。([tinygrad Discord](https://discord.com/channels/1068976834382925865))

- **LLM Serving 平台竞争激烈，Triton 替代方案涌现**：LM Studio 和 MAX Serving 社区的讨论集中在不同 **LLM Serving 平台**的能力上，MAX Serving 正作为 Triton 的潜在替代方案被探索。用户寻求关于迁移现有设置的指导，并询问了对 GPU 托管模型等功能的支持。LM Studio 社区还在努力解决各种模型和硬件配置中的错误信息及兼容性问题。([LM Studio Discord](https://discord.com/channels/1110598183144399058), [Modular Discord](https://discord.com/channels/1087530497313357884))

- **检索增强微调 (RAFT) 成为焦点**：LlamaIndex 举办了一场网络研讨会，邀请了检索增强微调 (RAFT) 的主要共同作者 Tianjun Zhang 和 Shishir Patil，深入探讨了 RAFT 如何结合检索增强生成 (RAG) 和微调的优势，以提升语言模型在特定领域设置下的性能。该研讨会旨在为有兴趣在自己的项目中实施 RAFT 的人员提供见解和资源。([LlamaIndex Discord](https://discord.com/channels/1059199217496772688))

- **Axolotl 随着 Lisa 合并取得进展，面临 DeepSpeed 挑战**：Axolotl AI Collective 庆祝了 `lisa` 最新 PR 的通过以及用于测试的 YAML 示例的添加。然而，开发者在尝试使用 DeepSpeed 或 FairScale Single-Process Single-GPU (FSDP) 训练模型时遇到了显存溢出 (OOM) 错误。该集体在数据集统一工作方面也取得了进展，并表示有兴趣为超大规模语言模型 (VLLM) 探索 **runpod serverless**。([OpenAccess AI Collective Discord](https://discord.com/channels/1104757954588196865))

- **FastLLM 和 RankLLM 突破检索与重排序边界**：Qdrant 推出了 **FastLLM**，这是一款拥有 10 亿 token 上下文窗口的语言模型，旨在增强 AI 驱动的内容生成，详见其[发布公告](https://qdrant.tech/blog/fastllm-announcement/)。同时，由 @rpradeep42 等人开发的 **RankLLM**（一个经过微调用于重排序的开源 LLM 集合）被推荐给构建高级 RAG 系统的人员，并强调了选择合适重排序器 (reranker) 的重要性。([HuggingFace Discord](https://discord.com/channels/879548962464493619), [LlamaIndex Discord](https://discord.com/channels/1059199217496772688))

---

# PART 1: High level Discord summaries

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 3 Haiku 加入战局**：更小、更便宜的 **Claude 3 Haiku** 因其有效的推理和对陷阱问题的处理能力而引发热议，被视为 Opus 的高性价比替代方案。

- **Perplexity 用户思考广告前景**：关于 **Perplexity AI 策略**可能转向引入广告的讨论非常热烈，引发了对近期公告真实性的辩论，这可能与 **愚人节玩笑**有关。

- **选择更优的搜索方式**：参与者主张在 Perplexity 中使用 *Writing* 模式而非 *All* 模式，以获得更精简且问题更少的 **大语言模型 (LLM) 交互**。

- **Prompt 防御协议受到质疑**：针对 Perplexity AI 模型的 **Prompt 攻击**引发了安全担忧。讨论转向了建立强大防御机制以对抗恶意注入和数据投毒的必要性。

- **Gemini 1.5 Pro API 价格令人震惊**：一场活跃的对话抨击了 **Gemini 1.5 Pro API** 的高昂定价，引发了关于基于 **Token 消耗**的更具预算意识的分层定价结构的讨论。

- **拥抱 Cascade**：热心成员交流了对 **Stable Cascades** 的见解，并引用 Perplexity AI 进行深入理解。

- **窥探 Perplexity 的 Neo**：发起询问以揭示 **Neo** 的独特之处，并关注其独特的属性。

- **巧妙管理 Perplexity 订阅**：出现了一个小插曲，API 额度卡在 "Pending" 状态，且 Perplexity API 缺乏团队注册选项也引起了关注。

- **比较 Token 经济**：分享了对比 Perplexity 和 ChatGPT 之间 Token 开销的资源，帮助用户做出明智决定。

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Gecko 在 Text Embedding 领域达到新高度**：新型 *Gecko* 模型在 Massive Text Embedding Benchmark (MTEB) 上表现出强劲性能，并可能加速 Diffusion Model 的训练，详见其 [Hugging Face 论文](https://huggingface.co/papers/2403.20327)和 [arXiv 摘要](https://arxiv.org/abs/2403.20327)。对 Gecko 实际应用的关注体现在对其权重可用性的查询中。

**Aurora-M 照亮多语言 LLM 空间**：拥有 15.5B 参数的 Aurora-M 模型专注于多语言任务，同时遵循白宫 EO 设定的指南。该模型因处理了超过 2 万亿个训练 Token 而受到赞誉，详见 [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) 和 [arXiv](https://arxiv.org/abs/2404.00399) 的报道。

**Hugging Face 的 Diffusers 备受关注**：对 Hugging Face *Diffusers* 库的贡献引发了关于效率的辩论，重点集中在关于 Diffusers 中 CUDA autocast 的 PR 以及 Pipeline 中不完全统一的问题，见 [discussion #551](https://github.com/huggingface/diffusers/issues/551) 和 [PR #7530](https://github.com/huggingface/diffusers/pull/7530)。

**PyTorch 2.6 蓄势待发引发好奇**：关于 PyTorch 版本更新的讨论引发了兴趣，特别是 PyTorch 2.3 中静默添加的 bfloat16 支持，以及对即将发布的 PyTorch 2.6 新特性的期待。值得注意的贡献包括对 autocast 性能的批评，详情见 [GitHub 线程](https://github.com/pytorch/pytorch/issues/120930)。

**LangChain 活动携手 Harrison Chase 吸引 AI 工程师**：LangChain CEO Harrison Chase 准备在 **4 月 17 日下午 6:30** 的在线活动中发表讲话，主题关于如何利用 LangSmith 实现从原型到生产的跨越，注册链接见[此处](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/)。他的公司专注于使用 LLM 构建上下文感知推理应用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**预算有限下的模型实力**：公会成员积极辩论 **AI 建模中的成本与质量**，讨论范围从预训练不同数据集规模所需的 **5 万美元到数百万美元**不等。讨论重点放在如何在**资源效率**与保持**高质量输出**之间寻找平衡。

**诈骗防护升级**：针对恶意 Bot 和诈骗的增加，工程师社区强调需要强大的**检测系统**来阻止 AI 滥用并保护 Discord 服务器。

**节省空间的精准方案**：分享了在 **Google Colab** 等平台上保存微调模型（finetuned models）时节省空间的技巧，一位用户建议的方法可节省 8GB 空间，但警告称精度会有轻微损失。

**训练策略之争**：**数据集划分**的最优方法以及**稀疏微调 (SFT) 与量化方法 (Quantization methods)** 的应用是热门话题，关于性能与成本效益之间权衡的见解备受追捧。

**DeepSeek 集成热潮**：一名用户提议将 **DeepSeek 模型**集成到 **Unsloth 4bit** 中，展示了社区对模型多样性和效率提升的推动，并附带了 [Hugging Face 仓库](https://huggingface.co/deepseek-ai)和准备实施的 [Google Colab notebook](https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing)。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Cyberrealistic vs. EpicRealism XL**: 关于两个 Stable Diffusion 模型性能的辩论正在进行：虽然 **Cyberrealistic** 需要精确的 Prompt，但 **EpicRealism XL** 在写实图像的 Prompt 容忍度上表现更优。

**SD3 Is Coming**: 社区正热议 **Stable Diffusion 3 (SD3)** 预计在 4-6 周内发布的消息，尽管对时间表存疑，但对其改进功能（尤其是修复的文本生成功能）表现出明显的兴奋。

**Fixing Faces and Hands**: Stable Diffusion 爱好者们正在解决面部和手部细节渲染的挑战，推荐使用 **Adetailer** 和各种 Embedding 工具，在不牺牲处理速度的情况下提升图像质量。

**CHKPT Model Confusion**: 在众多的 **CHKPT models** 中，用户正在寻求最佳用例指导，并推荐将 **ponyxl**、**dreamshaperxl**、**juggernautxl** 和 **zavychroma** 等模型作为 Stable Diffusion 的 Checkpoint “入门套装”。

**Ethics and Performance in Model Development**: 讨论涉及 AI 发展的飞速步伐、使用专业艺术作品进行 AI 训练的伦理问题，以及对未来 Stable Diffusion 版本内存需求的推测，期间穿插着轻松的社区闲聊。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**DBRX Revealed**: 一个名为 **DBRX** 的新型开源语言模型引起了轰动，声称在既定基准测试中表现顶尖。[观看 DBRX 介绍](https://www.youtube.com/watch?v=dqFvOqC43rQ)。

**Whisper Models Under the Microscope**: 鉴于 **BetterTransformer** 错误率较高，**WhisperX** 可能会取代它。社区正在研读 [Transforming the Web](https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos) 和 [Apple 关于引用消解的最新论文](https://arxiv.org/pdf/2403.20329.pdf)。

**Speed Meets Precision in LLM Operations**: **LlamaFile** 在特定任务的 CPU 运行速度上比 llama.cpp 提升了 1.3x - 5x，这可能会改变未来的本地操作。用于 Hercules 微调的配置文件导致准确率下降，引发了关于 `lora_r` 和 `lora_alpha` 等设置的辩论。

**Hugging Face Misstep Halts Upload**: 由 Hugging Face 的 `safetensors.sharded` 元数据引起的 **ModelInfo** 加载问题正阻止上传到链上，驱动了关于修复方案的讨论。

**Brainstorming for WorldSim**: **WorldSim** 爱好者提议建立一个带有竞争性基准测试的 **"LLM Coliseum"**，支持上传预写脚本的文件，并推测未来发展，如竞争性排行榜和 AI 对战。

**Traffic Signal Dataset Signals Opportunity**: 一个 [交通信号图像数据集](https://huggingface.co/datasets/Sayali9141/traffic_signal_images) 浮出水面，尽管 Hugging Face 存在查看器兼容性问题，但有望助力视觉模型。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Trouble in GPU Paradise**: AMD GPU 给 tinygrad 用户带来了巨大困扰，出现了系统崩溃和类似 `"amdgpu: failed to allocate BO for amdkfd"` 的内存泄漏错误。用户分享了涉及 PCIe 电源循环的变通方法，但对 AMD 在解决这些 Bug 方面表现出的缺乏承诺感到不满。

**A Virtual Side-Eye to AMD's Program**: 加入 AMD Vanguard 计划的邀请遭到了 George Hotz 等人的怀疑，引发了关于此类计划有效性以及 AMD 对开源解决方案和更好软件实践需求的辩论。

**Learning Curve for Linear uOps**: #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224265332848328765) 频道分享了一份解释 linear uops 的详细 [文档](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md)，旨在揭开 tinygrad 中中间表示的神秘面纱，并辅以一份关于重大合并后新 [Command Queue 的教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md)。

**Tinygrad Pull Requests Under Scrutiny**: Pull Request [#4034](https://github.com/tinygrad/tinygrad/pull/4034) 解决了关于单元测试代码和后端检查的混淆。强调了为 CLANG 和 OpenCL 等各种后端维护适当测试环境的重要性。

**Jigsaw Puzzle of Jitted Functions**: 关于为什么 Jitted 函数不出现在 Command Queue 日志中的知识鸿沟，引发了关于 tinygrad 架构内 Jitted 操作与 Scheduled 操作执行情况的讨论。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio 陷入模型故障困扰**：请谨慎操作，LM Studio 在 RTX 3060 GPU 上运行 *estopian maid 13B q4* 模型时会出现未知异常，且有用户报告在长时间推理过程中发生崩溃。用户对 **Text-to-Speech 和 Speech-to-Text 功能** 的需求日益增长，但目前必须挂载像 *whisper.cpp* 这样的工具来实现语音功能。

**追求本地化隐私**：在本地 LLM 隐私化的探索中，一个建议是将 LM Studio 与 AnythingLLM 搭配使用以实现机密配置，尽管 LM Studio 本身并不内置文档支持。与此同时，**Autogen** 每次仅生成 2 个 tokens，让用户对其最佳配置产生疑问。

**GPU 讨论升温**：多 GPU 设置并不一定需要 SLI；然而，起作用的是 VRAM 而非组合 VRAM——这是运行模型的一个重要规格。双 Tesla P40 设置在运行 70B 模型时可以达到 3-4 tokens/sec，而预算有限的用户在赞赏 P40 的 VRAM 之余，也在权衡其与 4090 GPU 性能之间的差距。

**匿名需求的顶级模型**：对于有特定需求的工程师，推荐使用 **Nous-Hermes 2 Mistral DPO** 和 **Nous-Hermes-2-SOLAR-10.7B** 模型，特别是对于需要处理 NSFW 内容的用户。模型下载和执行过程中的技术故障让部分用户感到不满，怀疑缺失代理支持（proxy support）是罪魁祸首。

**渴望前代功能**：用户非常怀念在每次新生成时分割文本的便利性，因为目前的 **LM Studio 更新会覆盖现有输出**，这引发了恢复之前模式的请求。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Google 将互联网装进 RAM**：工程师们注意到 Google 强大的搜索性能可能归功于使用分布式版本的 FAISS 和改进的索引策略（如 **inverted indexes**）将网页嵌入到 RAM 中。讨论深入探讨了 Google 的基础设施选择，暗示了处理复杂且精确搜索查询的方法。

**探究 Google 的编程范式**：参与者剖析了 Google 对编程策略的使用，其中包括一些通常被排斥的结构，如全局变量和 `goto`，这展示了他们在系统中解决问题和提高效率的务实方法。

**Sparse Autoencoders 揭秘**：一个新的用于 Sparse Autoencoder (SAE) 的可视化库已经发布，揭示了它们的特征结构。对于 AI 模型中 SAE 特征分类的反应不一，既反映了细节的复杂性，也反映了 AI 可解释性（AI interpretability）中的抽象挑战。

**音乐 AI 的新地平线**：讨论了一篇探讨 **GANs** 和 **transformers** 在音乐创作中应用的论文，暗示了音乐 AI 未来的潜在方向，包括文本转音乐（text-to-music）的转换指标。同时，**lm-eval-harness** 基准测试在 **Anthropic Claude** 模型上的缺失，表明人们对全面模型评估框架的兴趣日益浓厚。

**GPT-NeoX 中的 Batch Size 权衡**：针对不均匀的 Batch Size 调整 **GPT-NeoX** 可能会因为负载不平衡而引入计算瓶颈，因为较大的 Batch 会拖慢处理速度。

**AI 竞技精神奖励**：有人建议 EleutherAI 社区参与 [Kaggle AI Mathematical Olympiad competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview)。算力资助（Compute grants）可以支持这些针对“科学领域 AI”计划的倾向。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 取消限制**：OpenAI 推出了一种[立即使用 ChatGPT](https://openai.com/blog/start-using-chatgpt-instantly) 的新方式，无需注册即可访问，旨在扩大 AI 的普及度，并确认用户交互有助于增强模型性能，同时提供可选的数据贡献选项。
  
- **提示词思考与管理模型**：工程师们讨论了在将 PDF 转换为 JSON 时，使用 Schema 与开放式提示词的效果，并对潜在的违反服务条款（Terms of Service）表示担忧，同时寻求关于自动化管理任务的提示词建议，包括指令划分和绩效规划。

- **AI 创作极限与原创性探究**：对不同 AI 在歌曲识别挑战中的响应对比揭示了 AI 创作能力的边界，一项[研究](https://arxiv.org/abs/2310.17567)指出 AI 表现出涌现行为（emergent behaviors），可能提供其训练集中未体现的原创输出。

- **期待 GPT-5 与探索 GPT-4**：社区对话反思了 Large Language Models (LLMs) 的反思能力，调侃了愚人节的技术链接，讨论了 GPT-4 相对于 Opus 的进步以及服务器稳定性问题，并分享了 **DALL-E 3** 图像编辑功能的使用心得，同时表达了对期待已久的 GPT-5 潜力的关注。

- **AI 功能的多样化应用**：工程师们正在探索各种 AI 工具（如 **Claude 3 Sonnet** 和 **Midjourney**）用于图像描述，并讨论了在三星 Galaxy Note 9 等设备上运行 AI 应用的兼容性挑战，解决方案包括检查系统版本或使用移动端浏览器作为替代方案。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **学习 RAFT**：一场关于 **检索增强微调 (RAFT)** 的网络研讨会（由 Tianjun Zhang 和 Shishir Patil 主讲）定于 **太平洋时间周四上午 9 点** 举行，届时将深入探讨 RAFT 相比传统语言模型微调的优势。准备材料包括 RAFT [博客文章](https://gorilla.cs.berkeley.edu/blogs/) 和完整的 RAFT [论文](https://arxiv.org/pdf/2403.10131.pdf)，注册链接在 [这里](https://lu.ma/v1bdat63)。

- **LlamaIndex 邀请参与网络研讨会**：LlamaIndex 正在举办一场关于 **RAFT** 的网络研讨会，将其比作参加“开卷考试”，并分享了一张使用不同工具构建 RAG 框架的示意图，详细的分步指南可以在 [这里](https://twitter.com/llama_index/status/1774950945488834684) 找到。

- **LlamaIndex 故障排除**：有报告称 **LlamaIndex** 文档过时、`OpenAI.organization` 设置存在困难，以及 `text-davinci-003` 等模型已被弃用。此外还讨论了在 RAG 中使用 **WeatherReader** 进行天气相关查询，以及使用 **LlamaParse** 手动处理 PDF 中图像的方法。

- **Agent 系统中的问题过度简化**：在构建多文档 RAG 系统领域，一位用户指出 *top_agent* 过度简化输入问题导致搜索结果不足。他们分享了关于错误缩小查询范围的细节，例如将“巧克力的过期日期”简化为仅剩“过期日期”。

- **值得观看的教程**：用户推荐了一个关于使用 **LlamaIndex** 构建 RAG 应用的 YouTube 教程，重点介绍了与 Pinecone 和 Gemini Pro 的集成，用于内容抓取、Embedding 转换和查询，访问地址为 [这里](https://youtu.be/B9mRMw0Jhfo)。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **JSON 处理的困扰**：工程师们正在讨论在 **LangChain** 中解析 JSON 的挑战，目前每一行都会创建一个单独的 Document，而不是一个带有完整元数据的 Document。该问题在 [JSON loader documentation](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json) 中有详细说明，但尚未发布解决方案。

- **使用 Tool 时 Token 消耗增加**：有人注意到当 LangChain agents 使用 tools 时，token 使用量增加了 50%，这归因于 tools 的数据检索和分词过程。虽然系统提示词（system prompt）会为推理假设执行一次，但并非所有 tools 都需要这样做。

- **LangGraph 迷宫**：分享了关于在 **LangGraph** 中利用基础模型作为状态的见解，并附带了一个 [GitHub notebook example](https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb)。此外，LangChain 中的 StructuredTool 字段可以使用 Pydantic 的 `BaseModel` 和 `Field` 类进行验证，详见 [GitHub issues](https://github.com/langchain-ai/langchain/issues/8066)。

- **微调是阻碍还是助力？**：关于从 chain 中实现结构化输出的对话建议采用两个 agents，以平衡微调后的专业知识和通用智能。然而，目前还没有达成明确的共识或策略来解决这一挑战。

- **PDF 和 PersonaFinders 激增**：讨论包括尝试使用向量嵌入（vector embeddings）在 PDF 之间映射内容，以便在语义上匹配段落。同时，一个名为 **PersonaFinder GPT** 的新版本发布，承诺根据识别出的个人属性提供对话式 AI 能力，并邀请在 [PersonaFinder Pro](https://chat.openai.com/g/g-xm4VgOF5E-personafinder-pro) 上进行测试。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LinkedIn 徽章：地位象征还是昙花一现？**：一位 LinkedIn 用户炫耀拥有超过 30 个 Top Voice 徽章，引发了对这类荣誉价值的质疑；[受质疑的 LinkedIn 徽章](https://www.linkedin.com/posts/jillanisofttech_datascience-artificialintelligence-analyticalskills-activity-7180469402079789056-oPrY?utm_source=share&utm_medium=member_desktop)。

- **AI 产生幻觉，开发者需警惕**：AI 虚构的软件包正被创建并被阿里巴巴等大公司误用，展示了一个潜在的恶意软件攻击向量；更多信息见 [The Register 的报道](https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/)。

- **十亿级 Token 模型即将来临**：Qdrant 推出了 **FastLLM**，能够支持 10 亿 token 的上下文窗口，旨在增强 AI 驱动的内容生成；深入了解其 [发布公告](https://qdrant.tech/blog/fastllm-announcement/)。

- **Diffuser 频道的深度讨论**：讨论集中在 **diffusers** 结合 **LoRA** 的复杂性上，涉及了没有明确解决方案的模型查询，并探讨了在 PDF 文件上微调语言模型的挑战，但尚未提供结论性的建议。

- **Gradio 4.25 首次亮相，带来增强的 UX**：**Gradio 4.25.0** 推出了诸如自动删除 `gr.State` 变量、`cache_examples="lazy"`、修复流式音频输出以及更直观的 `gr.ChatInterface` 等功能，以简化用户交互。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 通过 MAX Engine 变得强大**：Mojo 即将引入的 **MAX Engine 和 C/C++ 互操作性 (interop)** 旨在简化 **RL Python 训练**，可能允许在 Mojo 中快速重新实现 Python 环境，详情见 [Mojo Roadmap](https://docs.modular.com/mojo/roadmap#cc-interop)。同时，**Mojo 24.2** 专注于 Python 友好特性，令开发者感到兴奋，其深度探讨见 [MAX 24.2 公告](https://www.modular.com/blog/max-24-2-is-here-whats-new) 和关于 [Mojo 开源](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) 的博客文章。

**关注 Modular 的动态**：Modular 活跃的 Twitter 活动似乎是外联或发布系列的一部分，感兴趣其更新或活动的读者可以在 [Modular 的 Twitter](https://twitter.com/Modular) 上追踪他们的最新想法。

**关于 Tensor、测试和顶级代码的讨论**：关于 **Mojo** 特性和怪癖的公开对话仍在继续，例如通过减少拷贝初始化低效来提升 **Tensor 性能**。工程师们还提出了关于 **顶级代码 (top-level code)** 和 SIMD 实现的问题，强调了诸如 **Swift 风格并发** 和内联函数 (intrinsic function) 转换等挑战，部分指导可参考 [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782)。

**使用 Prism 解锁 CLI**：`Prism` CLI 库的重构带来了简写标志和嵌套命令结构等新功能，与 Mojo 24.2 更新保持一致。增强功能包括特定命令的参数验证器，其中 References 的开发历程和可用性是关注焦点，详见 [thatstoasty 在 GitHub 上的 Prism 项目](https://github.com/thatstoasty/prism)。

**在期待 GPU 支持的同时使用 MAX 进行部署**：关于将 **MAX 作为 Triton 后端替代方案** 的提问指向了 MAX Serving 的实用性，尽管目前尚不支持 GPU；文档可以指导通过本地 Docker 进行尝试，详见 [MAX Serving 文档](https://docs.modular.com/serving/get-started)。针对潜在 MAX 采用者的持续支持和澄清正在讨论中，强调 ONNX 模型可以平滑地融入 MAX 框架。

**Nightly Mojo 动态与文档**：资深 Mojo 用户收到了关于 **nightly build 更新** 的提醒，并被引导使用 `modular update` 命令，变更列在 [nightly build 变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中。此外，还记录了本地 **Mojo stdlib 开发** 的宝贵指南和最佳测试实践，建议使用 `testing` 模块而非 FileCheck，并指向 [stdlib 开发指南](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Miniconda 缩小了技术栈**：对于需要轻量级安装且不愿牺牲功能的场景，**Miniconda** 被验证为 Anaconda 的有效且更小的替代品。

- **征集 OhanaPal 合作伙伴**：*OhanaPal* 应用是一款利用 **OpenAI GPT API** 辅助神经多样性人群的创新工具，开发者正在寻求贡献者进行进一步的头脑风暴和原型设计。感兴趣的各方可以通过其 [网站](https://www.ohanapal.app/) 参与。

- **硬件设备的 3D 打印尺寸调整**：在 3D 打印 **O1 Light** 时，将模型放大 119.67% 以适配 M5 Atom Echo，GitHub 上的 [pull request #214](https://github.com/OpenInterpreter/01/pull/214) 为 M5Atom 增强了自动重连功能。

- **Windows 包管理增强**：给 Windows 用户的一个建议：除了传统的 Microsoft 产品外，可以考虑将 **winget** 和 **scoop** 作为可行的软件包管理工具。

- **开源 AI 促进独立性**：GitHub 上的 **fabric** 仓库提供了一个开源 AI 增强框架，通过众包 AI 提示词来解决特定问题；**Microsoft 的 UFO** ([GitHub - microsoft/UFO](https://github.com/microsoft/UFO)) 则探索了用于 Windows 交互的 UI-Focused Agents。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 中的聊天机器人前缀异常**：**undi95/remm-slerp-l2-13b:extended** 模型在 OpenRouter 的角色扮演聊天中意外地在回复前加上了 `{bot_name}:` 前缀；然而，最近的提示词模板（prompt templating）更改已被排除在原因之外。目前正在调查该场景下 `name` 字段的使用情况。

**SSL 连接之谜**：一次尝试连接 OpenRouter 的操作因 **SSL 错误**（描述为 *EOF occurred in violation of protocol*）而受阻，但社区尚未就解决方案达成共识。

**新书预告：Architecting with AI in Mind**：**Obie Fernandez** 发布了他的新书《Patterns of AI-Driven Application Architecture》的早期版本，书中重点介绍了 OpenRouter 的应用。该书可通过 [此处](https://leanpub.com/patterns-of-application-development-using-ai) 获取。

**Nitro 模型讨论升温**：尽管存在对 Nitro 模型可用性的担忧，但已确认 **Nitro 模型仍然可以访问且即将推出更多**。关于不同 AI 模型性能的困惑表明，用户对优化速度和效率有着浓厚的兴趣。

**模型故障排除与 Logit Bias**：用户在遇到 **NOUS-HERMES-2-MIXTRAL-8X7B-DPO** 等模型的问题时，讨论了针对特定任务的替代方案，如 **Nous Capybara 34B**，并指出其 30k 的上下文窗口（context window）可提升性能。此外，官方澄清了 OpenRouter 的 Logit Bias 应用目前仅限于 OpenAI 的模型。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **NumPy 出人意料的线程行为**：一位成员惊讶地发现 **NumPy** 没有充分利用线程，基准测试代码显示自定义的 `matmul` 函数性能更好。这凸显了 **NumPy** 在多线程能力方面的不足。

- **llamafile 文档提示词**：即将发布的 **llamafile 0.7** 引发了关于 **openchat 3.5** 内提示词模板（prompt templating）的讨论，揭示了需要更好的文档来消除用户的困惑。社区热切期待关于集成细节的更清晰指导。

- **TinyBLAS 提供无需 CUDA 的替代方案**：讨论中提到 **TinyBLAS** 可作为 GPU 加速的替代方案，但指出其性能取决于具体的显卡。该选项支持 GPU 而无需安装 CUDA 或 ROCm SDK，这能显著简化某些用户的环境搭建。

- **Windows ARM64 与 llamafile 的兼容性障碍**：询问 **llamafile** 对 **Windows ARM64** 支持情况的用户发现，虽然支持 ARM64X 二进制格式，但在 **AVX/AVX2** 模拟方面存在问题，这对于在 Windows ARM64 生态系统中工作的开发者至关重要。

- **本地部署故障**：参与者在本地部署 **llamafile** 时遇到了 **"exec format error"**，引发了故障排除讨论，包括建议从 zsh 切换到 bash，以及根据硬件配置正确执行 **Mixtral** 模型的细节。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**考虑 AMD 的 150 亿个理由**：MDEL 在 **AMD GPU** 上成功训练了一个 **15B 模型**，这表明 AMD 可能是大规模 AI 模型硬件领域的一个可行选择。

**训练冻结之谜**：有报告称在没有明显使用 `val_set_size` 或 `eval_table` 的情况下，训练在 Epoch 结束后挂起，暗示原因可能是存储空间不足，或者是某些模型或配置中尚未识别的 Bug。

**Axolotl 开发在恶作剧中继续**：Axolotl 开发团队批准了 **`lisa` 的 PR 合并**，添加了一个用于测试的 **YAML 示例**，并幽默地提议在愚人节与 OpenAI 建立合作伙伴关系。然而，目前仍存在文档缺失以及可能与 **DeepSpeed 或 FSDP** 训练尝试相关的显存溢出（OOM）错误。

**统一数据的困境**：目前正努力将 15 个数据集组合成统一格式，成员们正在克服从数据量到翻译不匹配等各种障碍。

**征集 Runpod 严谨评测**：社区对使用 **Runpod Serverless** 服务运行超大型语言模型表现出兴趣，并寻求社区经验的分享。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**FastLLM 强势进入 AI 领域**：[Qdrant 宣布了 FastLLM (FLLM)](https://qdrant.tech/blog/fastllm-announcement/)，这是一个号称拥有 10 亿 token 上下文窗口的语言模型，专为 Retrieval Augmented Generation (RAG) 设计。不过，怀疑论者认为其在 4 月 1 日发布的时间点可能暗示这只是个玩笑。

**理解 GPT 的可视化教程**：知名 YouTube 频道 3Blue1Brown 发布了一个关于 [Transformers 和 GPT 的视觉化介绍](https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s)，吸引了希望更清晰地理解这些架构概念的 AI 专业人士的关注。

**工程师构建开源 LLM 问答引擎**：在 [GitHub](https://github.com/developersdigest/llm-answer-engine) 上公开的一个开源项目 "llm-answer-engine" 引起了社区的兴趣。该项目利用 Next.js, Groq, Mixtral, Langchain 和 OpenAI 构建了一个受 Perplexity 启发的问答引擎。

**LLM 的结构化输出变得更简单**：工程界关注到 instructor 1.0.0 的发布，该工具旨在确保 Large Language Models (LLMs) 产生符合用户定义 Pydantic 模型的结构化输出，从而辅助其无缝集成到更广泛的系统中。

**Google 加强 AI 部门力量**：为了增强其 AI 产品，Google 聘请了 Logan Kilpatrick 来领导 AI Studio 并推进 [Gemini API](https://x.com/officiallogank/status/1775222819439149424?s)，这标志着这家科技巨头正加大投入，致力于成为 AI 开发者的核心枢纽。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **轻松上手 GPT**：成员们推荐了一个教学 [视频](https://www.youtube.com/watch?v=wjZofJX0v4M)，该视频以易于理解的方式解释了 Transformers 和 GPT 的基础知识，适合机器学习领域的新手。
- **极客梦想成真**：社区分享了一个雄心勃勃的项目——创建一个能够运行《雷神之锤》(Quake) 的自制 GPU。该项目展示了一位拥有游戏行业背景的开发者成功的 FPGA 设计；更多细节可以在其 [博客](https://www.furygpu.com/blog/hello) 上找到。
- **CPU 等于 GPU？并不尽然**：[Justine Tunney 的博客文章](https://justine.lol/matmul/) 在社区流传，讨论了 CPU 矩阵乘法的优化策略，并指出了其与 GPU 方法（如 warptiling）的区别。
- **Triton 性能分析成为焦点**：使用 Nsight Compute 对 Triton 代码进行性能分析（profiling）是一个主要话题。社区分享了关于优化性能的见解以及具体的命令，如 `ncu --target-processes all --set detailed --import-source yes -o output_file python your_script.py`，以改善开发工作流。文中强调了关键的性能提升，并参考了 [Accelerating Triton](https://pytorch.org/blog/accelerating-triton/) 的资源。
- **基准测试之争**：PyTorch 团队对近期与 JAX 和 TensorFlow 的基准测试对比表示担忧，并给出了官方回应。同时，Jeff Dean 发布的一条推文展示了 JAX 在多项测试中是 GPU 性能最快的，这引发了社区讨论；相关的基准测试表格可以在 [这里](https://x.com/JeffDean/status/1774274156944859455) 查看。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **RLAIF 可能提升 Opus**：据推测，应用 **Reinforcement Learning with Augmented Intermediate Features (RLAIF)** 可以通过优化其决策准确性来进一步增强 **Opus**。

- **Google 宏大的 AI 愿景**：**Google** 的新任 AI 产品负责人宣布，他们致力于在 **AI Studio** 和 **Gemini API** 的支持下，使 Google 成为 AI 开发者的首选之地。

- **DPO 的进展与讨论**：最近的一篇 [预印本论文](https://x.com/rm_rafailov/status/1774653027712139657?s=46) 探讨了大模型在 Direct Preference Optimization (DPO) 中的冗余（verbosity）问题。讨论中还提到了一项关于 Reinforcement Learning from Human Feedback (RLHF) 中冗余利用研究的反驳，该研究可在 [arXiv](https://arxiv.org/abs/2403.19159) 上查阅。

- **GPT-4 之后的 AI 迷雾**：在 GPT-4 发布之后，AI 社区注意到各公司倾向于增加保密性，减少分享模型细节，这偏离了之前的透明化准则。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Jamba 的速度洞察**：工程师们详细研究了在解码过程中，**Jamba** 的端到端吞吐效率如何随 token 数量的增加而提升。鉴于解码是顺序进行的，一些成员对这种增长表示质疑，但共识指出，即使上下文窗口增大，吞吐量增益依然存在，并会影响解码速度。

**解码效率之谜**：围绕一张显示 **Jamba** 解码步骤随 token 数量增加而变得更高效的图表，展开了一场关键讨论。困惑得到了解答，讨论阐明了每个 token 更高的吞吐量会影响解码阶段的效率，反驳了最初的误解。

---

# PART 2: 分频道详细摘要与链接

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1224278402006122586)** (888 messages🔥🔥🔥): 

- **Claude 3 Haiku 在与 Opus 的对比中表现稳健**：用户讨论了 **Claude 3 Haiku** 在 Perplexity AI 中的有效性，分析了它与 Opus 相比在处理推理和陷阱问题方面的表现，以及作为一款更小、更便宜模型的性价比。

- **对 Perplexity 引入广告的担忧**：用户对 **Perplexity AI 引入广告的潜在计划**表示猜测和担忧，特别是这与 **Pro 订阅** 的关系。该消息的真实性（可能是一个**愚人节玩笑**）正在被讨论，并引用了 AdWeek 和 Gadgets360 讨论**广告策略**的相关文章。

- **写作模式（Writing Mode）关注度的盛行**：讨论集中在 Perplexity 中的 *Writing* 专注模式是否更优，用户认为它比包含网络搜索的 *All* 专注模式提供了**更好的用户体验和更少的问题结果**。用户明显更倾向于写作模式，因为它能提供**更简洁的 LLM 交互**。

- **Prompt 攻击安全担忧**：一位用户询问了 Perplexity AI 如何保护其模型（如 Sonar）免受 **prompt 攻击**和其他安全漏洞的影响。对话转向了更广泛的问题，即如何保护 LLM 免受由于数据投毒或 **prompt injections** 导致的政策违规。

- **Gemini 1.5 Pro API 定价评论**：用户讨论了 **Gemini 1.5 Pro** 的预览定价，指出其 100 万 token 上下文能力的定价为 **每百万 token 7 美元**，价格昂贵。对话指出，希望未来能进行价格调整，并可能根据上下文窗口的使用情况实行分层定价。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://community.spiceworks.com/">未找到标题</a>：未找到描述</li>
<li><a href="https://x.com/LinusEkenstam/status/1774847013752070457?t=7tzw85sz9QgE_TN7zRA82Q&s=09">Linus ●ᴗ● Ekenstam (@LinusEkenstam) 的推文</a>：🚨 重磅 🚨 Apple 正在洽谈收购 Perplexity。这可能是一个非常令人兴奋的开始。</li>
<li><a href="https://fxtwitter.com/AravSrinivas/status/1775229252973334902?t=p2-h_dWeQhz6swoCVL66SA&s=19">Aravind Srinivas (@AravSrinivas) 的推文</a>：良好的氛围至关重要</li>
<li><a href="https://tenor.com/view/tiny-text-cant-see-ken-jeong-gif-5957945">微型文字 GIF - Tiny Text Cant - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/working-on-it-under-construction-gif-23162421">正在处理 GIF - Working On It - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/whine-give-up-pout-frustrated-gif-5313242348381105216">抱怨放弃 GIF - Whine Give Up - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/ouch-gif-12136515515962044163">哎哟 GIF - Ouch - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/yes-no-gif-16236377">是或否 GIF - Yes No - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/degout%C3%A9-chanceux-chance-chance-d%C3%A9butant-mr-miyaki-gif-22039391">Degouté Chanceux GIF - Degouté Chanceux Chance - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://www.instagram.com/perplexity.ai">登录 • Instagram</a>：未找到描述</li>
<li><a href="https://www.adweek.com/media/gen-ai-search-engine-perplexity-has-a-plan-to-sell-ads/">生成式 AI 搜索引擎 Perplexity 计划出售广告</a>：未找到描述</li>
<li><a href="https://www.reddit.com/r/singularity/comments/1bp885i/claude_3_haiku_is_the_new_budget_king/">Reddit - 深入探索一切</a>：未找到描述</li>
<li><a href="https://www.instagram.com/perplexityai?igsh=ZDg4MmZseWoweDJh">登录 • Instagram</a>：未找到描述</li>
<li><a href="https://www.youtube.com/watch?v=t-Nz6us7DUA">Morpheus 解释什么是真实的</a>：Morpheus 说：“你现在的外表就是我们所说的残余自我形象。它是你数字自我的心理投影。” 手沿着机翼滑动...</li>
<li><a href="https://www.youtube.com/watch?v=rie-9AEhYdY">我们必须为 Deep Learning 增加结构，因为...</a>：Paul Lessard 博士及其合作者发表了一篇关于 "Categorical Deep Learning and Algebraic Theory of Architectures" 的论文。他们旨在使神经网络...</li>
<li><a href="https://www.youtube.com/watch?v=poncZ1K9Tio">Scrubs - 两枚硬币，30 美分</a>：来自 Scrubs 第 3 季第 4 集的有趣片段。</li>
<li><a href="https://www.cbsnews.com/news/att-data-breach-2024-cbs-news-explains/">关于 AT&T 大规模数据泄露，客户应该了解的信息</a>：AT&T 周六表示，在暗网上发现的一个数据集包含约 7300 万当前和前任客户的社会安全号码和密码等信息。</li>
<li><a href="https://www.gadgets360.com/ai/news/perplexity-ai-powered-search-engine-could-soon-show-ads-report-5357479">报告称：AI 搜索引擎 Perplexity 可能很快向用户展示广告</a>：根据报告，Perplexity 将在其相关问题部分展示广告。</li>
<li><a href="https://slashdot.org/story/24/04/01/1653221/perplexity-an-ai-startup-attempting-to-challenge-google-plans-to-sell-ads">Perplexity，一家试图挑战 Google 的 AI 初创公司，计划出售广告 - Slashdot</a>：一位匿名读者分享了一份报告：声称是 Google 竞争对手的生成式 AI 搜索引擎 Perplexity，最近刚从 Jeff Bezos 等投资者那里获得了 7360 万美元的 B 轮融资...</li>
<li><a href="https://youtu.be/wjZofJX0v4M?feature=shared">GPT 到底是什么？Transformer 的视觉入门 | Deep Learning，第 5 章</a>：Transformer 及其先决条件的介绍。赞助者可提前观看下一章：https://3b1b.co/early-attention 特别感谢这些支持...</li>
<li><a href="https://tenor.com/view/spongebob-gif-7921357">海绵宝宝 GIF - Spongebob - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/the-bachelorette-you-see-what-im-saying-understand-get-it-you-feel-me-gif-16430678">The Bachelorette 你明白我在说什么吗 GIF - The Bachelorette You See What Im Saying Understand - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/i-feel-your-pain-all-the-feels-keezia-leigh-comfort-i-feel-you-gif-16152618">我能感受到你的痛苦 GIF - I Feel Your Pain All The Feels Keezia Leigh - 发现并分享 GIF</a>：点击查看 GIF</li>
<li><a href="https://tenor.com/view/protect-attack-punch-me-fight-me-prepare-gif-17784080">保护攻击 GIF - Protect Attack Punch Me - 发现并分享 GIF</a>

s</a>: 点击查看 GIF</li><li><a href="https://downforeveryoneorjustme.com/perplexity">Perplexity 宕机了？当前问题与状态 - DownFor</a>: Perplexity 无法加载？或者在使用 Perplexity 时遇到问题？在这里检查状态并报告任何问题！</li><li><a href="https://docs.perplexity.ai/page/application-status">Application Status</a>: 未找到描述</li><li><a href="https://www.quora.com">Quora - 分享知识、更好地了解世界的平台</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/AskReddit">Reddit - 尽情探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/NoStupidQuestions">Reddit - 尽情探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/explainlikeimfive">Reddit - 尽情探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/subreddits/search">search results</a>: 未找到描述</li><li><a href="https://community.spiceworks.com">no title found</a>: 未找到描述</li><li><a href="https://discuss.codecademy.com">Codecademy Forums</a>: Codecademy 的社区讨论论坛。</li><li><a href="https://hashnode.com">开始编写开发者博客：Hashnode - 自定义域名、子路径、托管/无头 CMS。</a>: 具有自定义域名、托管/无头 CMS 选项的开发者博客。我们全新的无头 CMS 为开发工具公司简化了内容管理。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1224304009352056885)** (19 messages🔥): 

- **探测 Stable Cascades 的细节**：一位成员分享了 Perplexity AI 的链接，涉及 **Stable Cascades** 的细节和见解。
- **探索 Neo 的独特性**：对 **Neo** 脱颖而出的原因感到好奇，从而在 [Perplexity AI](https://www.perplexity.ai/search/why-is-Neo-26.mxjH_TEmzTnxhxetBDQ) 上进行了搜索查询。
- **关于如何继续使用 Perplexity 的说明**：一位成员的查询被重定向到 [Perplexity AI](https://www.perplexity.ai/search/How-can-I-VBSI6VXuQOiKraNH.DMbrw)，同时提醒确保帖子可以通过 **Discord** [链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) 共享以供参考。
- **愚人节玩笑还是技术创新？**：通过 [Perplexity AI 搜索关于愚人节技术的内容](https://www.perplexity.ai/search/April-fool-tech-Au6YyiG1TCCPZdIBAawqHw)，探讨了幽默与进步之间的界限。
- **GPTDevil 集合引起关注**：一位成员指向了关于 **GPTDevil** 的 [Perplexity AI 集合](https://www.perplexity.ai/collections/GPTDevil-A.GvNcQZS0yjMGDDJtDPUQ)，表明了对该话题的兴趣。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1224277874119544852)** (16 messages🔥): 

- **添加 API 额度的问题**：一位成员报告了添加 API 额度的问题，称该过程卡在“Pending”状态，且发票显示为“void”。
- **在 API 中检索来源的请求**：一位成员询问是否可以通过 Perplexity API 获取提示词的来源，类似于浏览器提示词中可用的功能。
- **Token 成本对比资源**：一位成员为希望对比 Perplexity 模型和 ChatGPT 的 Token 成本的人提供了链接，可查看 [Perplexity 的定价](https://docs.perplexity.ai/docs/pricing) 和 OpenAI 的定价。
- **Perplexity API 不支持团队注册**：一位成员询问了为团队注册 Perplexity API 的流程，并被告知目前不支持团队注册。
- **对速率限制（Rate Limits）的困惑**：一位成员对使用 `sonar-medium-online` 模型时出现的不一致速率限制行为表示担忧，尽管遵守了规定的每分钟 20 次请求限制，但仍遇到了 429 错误。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/pricing">定价</a>: 未找到描述</li><li><a href="https://openai.com/pricing">定价</a>: 简单灵活。按需付费。
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1224372448158289980)** (525 messages🔥🔥🔥): 

- **Hugging Face Diffusers PR 讨论**：社区成员讨论了一个关于在 [Hugging Face Diffusers GitHub](https://github.com/huggingface/diffusers/pull/7530) 上禁用 CUDA 设备 autocast 的 PR。对话转向对 Hugging Face 的批评，指责其在不同的 pipeline 和 trainer 之间没有统一的代码，并指出了效率和荒谬之处。

- **关于合并不同 SD Pipelines 的持续性问题**：社区成员强调了 [GitHub discussion #551](https://github.com/huggingface/diffusers/issues/551) 中记录的一个持续问题，即合并不同的 Stable Diffusion pipelines，并指出由于决定保持独立的 pipelines，该复杂性依然存在。

- **对 Hugging Face 工程优先级的批评**：出现了一场批评 Hugging Face 在 Diffusers 上工程工作的讨论，涉及工程师不足、过多的“AI 代码思想领袖”，以及工程师采用微服务框架等冲突方法。

- **特定 PyTorch 版本的讨论**：社区成员对 PyTorch 版本进行了广泛的技术讨论，提到了 PyTorch 2.3 中静默添加的 bfloat16 支持以及 nightly builds 的复杂性。有人对 autocast 性能问题及可能的修复发表了评论，详情已添加到 [GitHub thread](https://github.com/pytorch/pytorch/issues/120930)，并对 PyTorch 2.6 版本的发布表示期待。

- **AI 生成图像与采样设置**：对各种 Diffusion 模型版本和配置生成的图像质量进行了评价，特别关注了锤子的图像。采样器及其配置的差异引发了关于这些参数的有效性和正确使用的交流。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa_recent_pytorch_nightlies_support_enough/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://tenor.com/view/old-gregg-easy-calm-down-relax-man-peach-gif-26522310">Old Gregg Easy GIF - Old Gregg Easy Calm Down - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/JrNL.gif">Tom And Jerry Mouse GIF - Tom And Jerry Mouse Bumped - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/blKOX.gif">Pelicula Western GIF - Pelicula Western Meme - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/pytorch/pytorch/issues/120930>">Issues · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - Issues · pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/issues/71631>">Issues · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - Issues · pytorch/pytorch</li><li><a href="https://github.com/huggingface/diffusers/issues/7563">[mps] training / inference dtype issues · Issue #7563 · huggingface/diffusers</a>：在没有 attention slicing 的情况下在 Diffusers 上进行训练时，我们看到：/AppleInternal/Library/BuildRoots/ce725a5f-c761-11ee-a4ec-b6ef2fd8d87b/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPS...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530/files">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>：此 PR 做了什么？修复了 #7529。在提交之前，此 PR 修复了一个拼写错误或改进了文档（如果是这种情况，您可以忽略其他检查）。您阅读了贡献者指南吗...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530#discussion_r1547822696">7529 do not disable autocast for cuda devices by bghira · Pull Request #7530 · huggingface/diffusers</a>：此 PR 做了什么？修复了 #7529。在提交之前，此 PR 修复了一个拼写错误或改进了文档（如果是这种情况，您可以忽略其他检查）。您阅读了贡献者指南吗？D...</li><li><a href="https://github.com/huggingface/diffusers/issues/551">Merging Stable diffusion pipelines just makes sense · Issue #551 · huggingface/diffusers</a>：遵循该理念，已决定为 Stable Diffusion 的 txt-to-img、img-to-img 和 inpainting 保留不同的 pipelines。结果如下：PR #549：代码重复了 4 次 (onnx...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1224354219436281906)** (9 条消息🔥): 

- **介绍用于高效文本嵌入的 Gecko**：Gecko 是一种紧凑型文本嵌入模型，通过将大型语言模型（LLM）的知识蒸馏到检索器中，展示了强大的检索性能。它在 Massive Text Embedding Benchmark (MTEB) 上优于现有模型，详情可见 [Hugging Face paper](https://huggingface.co/papers/2403.20327) 和 [arXiv abstract](https://arxiv.org/abs/2403.20327)。

- **Gecko 在 Diffusion 模型中的潜在应用**：对话建议探索使用 Gecko 来潜在地加速 Diffusion 模型训练，以取代 T5 的使用。讨论对模型性能的影响（特别是在嵌入方面）持推测态度。

- **Gecko 权重查询**：一名成员询问上述 Gecko 的权重是否可用，表明了对其在实际应用中的兴趣。

- **评估 Large Vision-Language Models**：[MMStar Benchmark](https://mmstar-benchmark.github.io/) 考察了评估 Large Vision-Language Models 的有效性，指出了诸如在文本足以解决问题时使用不必要的视觉内容等问题。

- **发布 Aurora-M，一个多语言 LLM**：介绍了 Aurora-M 的新预印本，这是一个拥有 15.5B 参数、经过 red-teamed、开源且持续预训练的多语言大语言模型。它已处理超过 2T 的训练 tokens，并符合 White House EO 的指南，更多详情见 [Twitter](https://twitter.com/__z__9/status/1774965364301971849?s=20) 和 [arXiv](https://arxiv.org/abs/2404.00399)。

- **提高 t2i 转换中的空间一致性**：在微调期间在 captions 中加入更好的空间描述，可以增强 text-to-image 模型生成的图像的空间一致性。该研究结果详见 [arXiv 预印本](https://arxiv.org/pdf/2404.01197.pdf)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mmstar-benchmark.github.io/">MMStar</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2403.20327">论文页面 - Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>：未找到描述</li><li><a href="https://x.com/__z__9/status/1774965364301971849?s=20">来自 ً ‎ (@__z__9) 的推文</a>：新的预印本！第一个多语言、经过 red-teamed、开源、持续预训练的 LLM - **Aurora-M**，符合关于安全、可靠和值得信赖的开发的 #WhiteHouse 行政命令...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order</a>：预训练语言模型支撑着多种 AI 应用，但其高昂的训练计算成本限制了可访问性。BLOOM 和 StarCoder 等计划旨在民主化对 p... 的访问。
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1224793707606184117)** (1 条消息): 

- **与 Harrison Chase 一起参加 LangChain 活动**：**LangChain** 的 CEO 兼联合创始人 Harrison Chase 将在即将举行的活动中发表讲话，探讨公司在使用 LangSmith 从原型转向生产时面临的挑战。在线会议定于 **4 月 17 日下午 6:30** 举行，感兴趣的人员可以在[此处](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/)注册。
- **从原型到生产，Harrison Chase 进行讲解**：演讲将涵盖从易于上手的 **GenAI apps** 到全面生产的过渡，重点介绍由此带来的新挑战。Chase 的公司 LangChain 专注于简化 LLMs 在开发上下文感知推理应用中的使用。

**提及的链接**：<a href="https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/">Meetup #3 LangChain and LLM: Using LangSmith to go from prototype to production, 2024 年 4 月 17 日星期三 18:30 | Meetup</a>：我们很高兴邀请到 LangChain 的联合创始人兼 CEO Harrison Chase 参加我们的第三次 LangChain and LLM France Meetup！不要错过这个难得的机会。

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1224281655204642826)** (212 条消息🔥🔥):

- **关于高效模型训练的讨论**：成员们讨论了构建对话式 AI 模型的策略，思考是根据响应时间拆分数据集，还是使用 LLM 获取更高质量的样本。共识倾向于在进行更复杂的拆分之前先从简单的开始。
- **量化与微调实践的辩论**：对话围绕 QLora 与 Sparse Fine Tuning (SFT) 等量化方法的有效性和性能损失展开，其中针对区块的微调（LoRA/QLoRA）可能无法达到 SFT 那样对单个权重的关注度。用户分享了关于资源效率与模型质量之间平衡的见解。
- **成本讨论**：社区交流了关于模型预训练的高昂成本，推测价格从小型尝试的 5 万美元到大型数据集和更多资源的数百万美元不等，强调了对成本效益的关注以及对更实惠训练机制的潜在需求。
- **呼吁防范诈骗和恶意机器人**：成员们注意到针对 Discord 服务器的机器人和诈骗活动有所增加，指出了 AI 技术被诈骗者利用的风险，以及增加警惕性和建立更好的欺诈内容检测系统的必要性。
- **微调模型提示词澄清与教程**：针对微调提示词和模型格式给出了具体建议，并讨论了对更清晰的教程、微调过程的视频内容以及社区协作式 Notebook 和 GitHub 资源的利用需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/muzeke-gif-27066384">Muzeke GIF - Muzeke - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b">jondurbin/airoboros-gpt-3.5-turbo-100k-7b · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1br8ry8/finetuning_a_llm_for_longform_creative_writing/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/am-ia-joke-to-you-am-ia-joke-is-this-a-joke-do-you-think-this-is-funny-do-you-think-this-is-a-joke-gif-14191111">Am Ia Joke To You Is This A Joke GIF - 你在开玩笑吗 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/i-aint-no-fool-wiz-khalifa-still-wiz-song-im-not-a-fool-im-not-an-idiot-gif-21822363">I Aint No Fool Wiz Khalifa GIF - 我不是傻瓜 Wiz Khalifa Still Wiz 歌曲 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=g68qlo9Izf0&t=850s">在单 GPU 上对 Llama-v2-7b 进行高效微调</a>：微调 LLM 时可能遇到的第一个问题是“host out of memory”错误。微调 7B 参数模型更加困难...</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">主页</a>：快 2-5 倍，内存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/intel-analytics/ipex-llm">GitHub - intel-analytics/ipex-llm: 在 Intel CPU 和 GPU（例如带有 iGPU 的本地 PC，以及 Arc、Flex 和 Max 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma 等）。一个与 llama.cpp, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, ModelScope 等无缝集成的 PyTorch LLM 库。</a>：加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma 等）在 Intel CPU 和 GPU 上...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1224319749425725440)** (311 条消息🔥🔥): 

- **Unsloth 更新导致推理中断**：用户报告在更新后使用 Unsloth AI 进行推理时遇到尺寸不匹配错误。官方已发布修复程序，回滚了更改，为用户解决了该问题。

- **Colab 上的模型保存挑战**：一位用户在 Google Colab 有限的存储空间内保存微调后的 13B 模型时遇到困难，被建议尝试使用 Kaggle，因为它提供免费的 2x Tesla T4。另一位用户建议使用 `model.save_pretrained_gguf("model", tokenizer, quantization_method = "q5_k_m", first_conversion = "q8_0")` 方法来节省 8GB 空间，但指出可能会有 0.1% 的精度损失。

- **Jamba 模型支持推测**：关于在 Unsloth 中添加 Jamba 模型支持的复杂性的讨论，承认由于其作为 Mamba 和 MoE 模型的特性，实现难度较大。

- **微调评估说明**：针对 Unsloth SFT Trainer 的评估过程提供了详细解答，解释了如何通过显式传递评估数据集和策略来获取评估指标。

- **加载数据集缓慢及潜在的 IPv6 问题**：用户讨论了在本地 Jupyter notebooks 中使用 `load_dataset` 时的显著延迟，怀疑这可能与 Ubuntu 系统上的 IPv6 设置有关。据报告，相同的命令在 Windows 配合 WSL 和 IPv4 环境下运行正常。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>：使用 Unsloth Notebook 训练 Tinyllama，数据集 https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://docs.wandb.ai/guides/track/jupyter">在 Jupyter Notebooks 中进行追踪 | Weights &amp; Biases 文档</a>：在 Jupyter 中使用 W&amp;B，无需离开 notebook 即可获得交互式可视化。</li><li><a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit/tree/main">unsloth/mistral-7b-instruct-v0.2-bnb-4bit at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">主页</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/295">推理无法工作 · Issue #295 · unslothai/unsloth</a>：嘿，我正在 Mistral 上运行文档中的 colab，训练正常，但在进行推理时出现尺寸不匹配错误：将 `pad_token_id` 设置为 `eos_token_id`:2 以进行开放式生成...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1224402368553291997)** (4 条消息): 

- **针对 Unsloth 4bit 的 DeepSeek 模型建议**：一名成员提议将最小的 **DeepSeek 模型** 添加到 **Unsloth 4bit** 中，称其为一个优秀的基础模型，并提供了 DeepSeek 的官方 [Hugging Face 仓库](https://huggingface.co/deepseek-ai)。另一名成员表示赞同并确认将进行实现。
- **待考虑的 Colab Notebook**：一个共享的 [Google Colab notebook](https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing) 被标记待审，表明其中的内容应当被**实现**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1NLqxHHCv3kFyw45t8k_CUfNlcepMdeDW?usp=sharing#scrollTo=Rdsd82ngpHCG">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/">deepseek-ai (DeepSeek)</a>：未找到描述</li><li><a href="https://chat.deepseek.com/">DeepSeek</a>：与 DeepSeek AI 聊天。
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1224258874895503440)** (377 条消息🔥🔥): 

- **Cyberrealistic 与 EpicRealism XL**：参与者讨论了 Cyberrealistic 和 EpicRealism XL 模型在写实图像生成方面的表现。他们发现 Cyberrealistic 需要详细的提示词，而 EpicRealism XL 在提示词较宽松的情况下能产生更好的效果。

- **对 SD3 的期待**：社区对 SD3 的发布充满期待，提到距离之前的公告已有 4-6 周的时间。用户对发布时机表示怀疑，同时表达了对新版本功能和改进的渴望，特别是修复后的文本生成功能。

- **面部和肢体模型挑战**：用户描述了使用 Stable Diffusion 进行面部和手部渲染时遇到的问题，并建议使用 Adetailer 和 embeddings 等各种修复方法。目前正努力寻找快速可靠的解决方案，以尽量减少批量生成图像时的额外处理时间。

- **CHKPT 模型指南**：由于可用模型数量庞大，用户分享了关于 CHKPT 模型指南的查询，寻求有关哪些模型最适合特定用途的信息。一些特定模型如 ponyxl、dreamshaperxl、juggernautxl 和 zavychroma 被推荐作为 Stable Diffusion Checkpoint “入门包”的一部分。

- **模型性能讨论**：对话涵盖了各种话题，包括 AI 发展的速度、围绕使用专业艺术作品进行 AI 训练的伦理考量，以及未来 Stable Diffusion 版本潜在的显存需求。此外还有一些玩笑和幽默，展现了社区互动的轻松一面。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/smirk-teehee-pokemon-laugh-psyduck-gif-10282192216865852036">Smirk Teehee GIF - Smirk Teehee Pokemon - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/anime-help-tears-cry-sad-gif-17104681">Anime Help GIF - Anime Help Tears - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://civitai.com/models/367412">Geeky Ghost Vid2Vid Organized v1 - v4.0 | Stable Diffusion Workflows | Civitai</a>：此工作流专为高级视频处理设计，融合了风格迁移、运动分析、深度估计等多种技术...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1brcntc/so_openai_never_had_any_magic_sauce/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://feedback.civitai.com/p/pass-or-fail-a-simple-and-controlled-model-ranking-feature>)">Feedback - Civitai</a>：向 Civitai 提供关于如何改进产品的反馈。</li><li><a href="https://huggingface.co/ostris/ip-composition-adapter/tree/main">ostris/ip-composition-adapter at main</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=0D6opXdC7ew">epicRealism - THIS is the Model you WANT!!!!</a>：epicRealism 是具有完美现实感的 1.5 模型。高度细腻的皮肤。真实的图像。超写实的面部。自然光线。也非常适合较少衣物的...</li><li><a href="https://civitai.com/models/229002">ICBINP XL - v4 | Stable Diffusion Checkpoint | Civitai</a>：如果你喜欢这个作品，请考虑请我喝杯咖啡 :) 在 Stable Horde 上免费使用此模型。这是期待已久的 ICBINP 后续作品，该模型是...</li><li><a href="https://youtu.be/B-Wd-Q3F8KM">The Count Censored</a>：我的音乐 Facebook 页面 https://www.facebook.com/pencilfacemusic</li><li><a href="https://civitai.com/models/277058/epicrealism-xl">epiCRealism XL - V5-Ultimate | Stable Diffusion Checkpoint | Civitai</a>：更新2：重回正轨，我从 V1 进行了精炼——可能是 SD3 发布前 SDXL 的最后一个版本。保守这个秘密，SDXL 的魔力 🧙‍♂️ 发生在：30 步...</li><li><a href="https://youtube.com/shorts/C5cIib7hiK8?si=z8FW2_UFwgZEn0LK">1万年かけて成長するフリーレン(上半身)/Frieren growing over 10,000 years(upper body) #葬送のフリーレン #frieren #アニメ</a>：未找到描述</li><li><a href="https://github.com/jhc13/taggui/releases">Releases · jhc13/taggui</a>：用于图像数据集的标签管理器和标注器。通过在 GitHub 上创建账户为 jhc13/taggui 的开发做出贡献。</li><li><a href="https://civitai.com/models/4201/realistic-vision-v60-b1">Realistic Vision V6.0 B1 - V5.1 (VAE) | Stable Diffusion Checkpoint | Civitai</a>：我建议在 Hugging Face 上查看有关 Realistic Vision V6.0 B1 的信息。该模型可在 Mage.Space（主要赞助商）和 S...</li><li><a href="https://asia.nikkei.com/Business/Technology/Japan-panel-pushes-to-shield-copyrighted-work-from-AI-training">Japan panel pushes to shield copyrighted work from AI training</a>：草案文件称，未经授权使用受保护材料可能构成侵权。</li><li><a href="https://civitai.com/models/15003/cyberrealistic">CyberRealistic - v4.2 | Stable Diffusion Checkpoint | Civitai</a>：想请我喝咖啡吗？（买一杯）示例中使用的可选 CyberRealistic 负面提示词，请查看 Hugging Face 上的 SDXL 版本 CyberRealistic Int...</li><li><a href="https://civitai.com/models/312530/cyberrealistic-xl">CyberRealistic XL - v1.1 (VAE) | Stable Diffusion Checkpoint | Civitai</a>：想请我喝咖啡吗？（买一杯）CyberRealistic XL 花了一些时间，但这是 CyberRealistic 的 SDXL 版本。该模型的标准...
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1224287410691768350)** (4 条消息): 

- **愚人节热情**：一位成员表示想创建一个不断重复“April fools”的 GGUF，并将其作为一个名为 q0_k_xl 的模拟 GPT4 进行分享。
- **DBRX 新 AI 亮相**：分享了一段名为“DBRX: A New State-of-the-Art Open LLM”的视频，介绍了 DBRX 这一开源通用语言模型，声称其在基准测试中创下了新纪录。点击[此处](https://www.youtube.com/watch?v=dqFvOqC43rQ)观看视频。

**提到的链接**：<a href="https://www.youtube.com/watch?v=dqFvOqC43rQ">DBRX: A New State-of-the-Art Open LLM</a>：介绍 DBRX，由 Databricks 创建的开源通用 LLM。在一系列标准基准测试中，DBRX 为已有的... 树立了新的 SOTA。

  

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224265625526861915)** (18 messages🔥): 

- **HF Transformers 令人惊讶的错误率**：成员们对 HF Transformers 的高错误率表示惊讶，并提到在 Distil whisper v3 large 中使用 **BetterTransformer**。表达了在未来项目中考虑使用 **WhisperX** 的意向。
  
- **预见 Web Silos 的终结**：分享了一个讨论 Web 即将发生变革的链接，包括搜索引擎从信息检索向更具预测性和个性化的方式转变，由于互联的数字生态系统而需要彻底改革收入策略，以及专用 Web 用户界面 (UIs) 可能过时。阅读关于即将到来的数字变革：[Transforming the Web](https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos)。

- **Apple 关于 Reference Resolution 的新论文**：讨论了 Apple 的一篇新论文，该论文表明一个 80M 参数的模型在大多数 Reference Resolution 基准测试中优于 GPT-3.5，而一个 250M 参数的模型则击败了 GPT-4。对话指出 Reference Resolution 对 AI Agent 的有效性至关重要。在此阅读 Apple 的论文 [here](https://arxiv.org/pdf/2403.20329.pdf)。

- **Reference Resolution 在 AI 准确性中的作用**：继续讨论 Reference Resolution 话题，强调这可能是 AI Agent 在执行任务期间出错的一个重要因素。进一步的对话涉及一种解释，即 80M 参数模型在未见过的任务上表现出奇地好，这可能是由于各模型之间的误差幅度较大或准确性相似。

- **来自 Twitter 的最新参考**：建议将一篇强调新发布信息的 Twitter 帖子用于后续讨论中的对比。它可以作为模型比较的进一步参考点。查看该 Twitter 帖子上的新信息 [on this Twitter post](https://vxtwitter.com/maxaljadery/status/1775196809893478797?s=46&t=stOPrwZiN_fxSK0RuC8Flg)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.20329">ReALM: Reference Resolution As Language Modeling</a>：Reference Resolution 是一个重要的问题，对于理解和成功处理各种类型的上下文至关重要。这些上下文包括之前的对话轮次以及相关的上下文...</li><li><a href="https://www.f5.com/company/blog/transforming-the-web-the-end-of-silos">Transforming the Web: The End of Silos</a>：我们使用互联网的方式即将发生重大变化。向更统一、更高效的 Web 导航方式转变，是对比我们已经习惯的传统孤岛式 Web 浏览的一次巨大飞跃...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1224275749834326017)** (104 messages🔥🔥): 

- **LlamaFile 的速度飞跃**：[Justine Tunney 宣布](https://x.com/justinetunney/status/1774621341473489024) LlamaFile 在 CPU 上的许多 Prompt/图像评估用例中现在比 llama.cpp 快 1.3x - 5x，详情见其矩阵乘法博客文章 [here](https://justine.lol/matmul/)。
- **探索 PII 脱敏**：一位成员分享了 Hugging Face 上一个热门的 PII（个人身份信息）脱敏数据集，可以通过 [here](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) 访问，同时讨论了处理个人敏感数据的挑战。
- **关于 Hermes 项目方向的辩论**：关于 Hermes 项目未来方向的讨论正在进行中，一些成员强调了模型良好处理 PII 的重要性，并提到了在 open-llama-3b 等较小模型中类似的各种能力。
- **对 Anthropic 立即封号的担忧**：一位成员对在快速登录和退出 Claude AI 网站后被立即封号表示困惑，推测这可能在系统看来显得可疑。
- **关于语言模型程序的讨论**：随后引发了一场关于使用带有 DSPy 的求解器来创建针对“逻辑连贯性”优化的极度压缩基础模型的广泛辩论，其中包括提到斯坦福大学研究人员对“LLM 效率的下一步”感兴趣，这一切都由 Omar Khattab 关于 DSPy 的视频引发，可以在 [here](https://youtu.be/Y94tw4eDHW0?t=549) 观看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19928">DiJiang: Efficient Large Language Models through Compact Kernelization</a>: 为了减轻 Transformers 的计算负载，线性注意力 (linear attention) 的研究获得了显著动力。然而，注意力机制的改进策略通常需要...</li><li><a href="https://arxiv.org/abs/2009.03393">Generative Language Modeling for Automated Theorem Proving</a>: 我们探索了基于 Transformer 的语言模型在自动定理证明中的应用。这项工作的动机在于，与...相比，自动定理证明器的一个主要局限性可能是...</li><li><a href="https://x.com/p00ssh/status/1775185708887539864?s=20">来自 poosh (e/λcc) (@p00ssh) 的推文</a>: attention is what you need, anon</li><li><a href="https://arxiv.org/abs/1806.00608">GamePad: A Learning Environment for Theorem Proving</a>: 在本文中，我们介绍了一个名为 GamePad 的系统，可用于探索机器学习方法在 Coq 证明助手中的定理证明应用。交互式定理证明器...</li><li><a href="https://huggingface.co/datasets?sort=trending&search=pii">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://x.com/justinetunney/status/1774621341473489024">来自 Justine Tunney (@JustineTunney) 的推文</a>: 我刚刚让 llamafile 在许多 prompt / 图像评估用例和硬件上的 CPU 速度比 llama.cpp 快了 1.3 倍到 5 倍。https://justine.lol/matmul/</li><li><a href="https://arxiv.org/abs/2012.14474">Paraconsistent Foundations for Probabilistic Reasoning, Programming and Concept Formation</a>: 文章认为 4 值超相容真值（此处称为 "p-bits"）可以作为高度 AI 相关的概率逻辑、编程和概念形成形式的概念、数学和实践基础...</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">microsoft/unilm 仓库中的 unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://www.youtube.com/watch?v=ZYf9V2fSFwU">AI 先驱展示 AI AGENTS 的力量 - “未来是 Agentic 的”</a>: Andrew Ng，Google Brain 和 Coursera 创始人，讨论了 Agent 的力量以及如何使用它们。加入我的通讯以获取定期 AI 更新 👇🏼https://www.matthewb...</li><li><a href="https://github.com/YuchuanTian/DiJiang">GitHub - YuchuanTian/DiJiang: “DiJiang: Efficient Large Language Models through Compact Kernelization”的官方实现，这是一种新型的基于 DCT 的线性注意力机制。</a>: “DiJiang: Efficient Large Language Models through Compact Kernelization”的官方实现，这是一种新型的基于 DCT 的线性注意力机制。 - YuchuanTian/DiJiang</li><li><a href="https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228">shisa-ai/shisa-jamba-v1-checkpoint-4228 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/unchained-foxx-silent-django-gif-4956511">Unchained Foxx GIF - Unchained Foxx Silent - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/Y94tw4eDHW0?si=cbH5-LV2dkXkkb0_&t=549">使用 DSPy 编程基础模型 (Foundation Models) / 使用 ColBERT 进行多向量语义搜索 - Omar Khattab</a>: Omar Khattab 是斯坦福大学的博士候选人，也是 AI/ML 领域的 Apple Scholar。在这次对话中，Omar 解释了如何对基础模型流水线进行编程...</li><li><a href="https://app.wordware.ai/r/5cad80d6-e0bf-4f37-b147-5b44b2273038">Wordware - prompt 提取分析</a>: text -> prompt
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1224368184103272539)** (37 messages🔥):

- **LLM Implementation Inquiry**：一个新手询问从哪里开始学习将开源 LLM（如 `llama 2`）作为微调后的聊天机器人集成到网站中。回复中未提供具体的资源或解决方案。
- **Quantized LLM Training Quandary**：在讨论使用 Hugging Face 的 Llama 架构训练 110m 参数模型时，一位成员对 GPU 显存访问效率低下表示担忧，指出 GPU 约 97% 的时间花在访问显存上。分享了一个复杂的 *BitLinear implementation*，并建议该问题可能是由模型大小或 next-token 预测需求引起的。
- **Fine-Tuning Frustration**：用户分享了一个用于在自定义领域基准上微调 `NousResearch/Hermes-2-Pro-Mistral-7B` 的 *configuration file*，但发现微调后准确率下降。列出了配置中的项，如 `lora_r`、`lora_alpha` 和 `sequence_len`，但未对观察到的准确率下降给出诊断。
- **Agent Inclusivity in Sample Time Reward Models**：发起了一场关于在采样时使用奖励模型的对话，指出有时通过 *best-of-n sampling* 可以获得更好的结果，在这种情况下，经过奖励微调的模型并不总是根据其自身标准生成最优答案。
- **Supervised Fine-Tuning Scripts and Tokenizer Configurations Shared**：提供了微调 LLM `OLMo-Bitnet-1B` 和 `NousResearch/Hermes-2-Pro-Mistral-7B` 的资源链接，包括关于处理特殊 token 和 Tokenizer 配置的讨论。一种特定技术确保了 Tokenizer 配置包含主要功能所需的 token，并且这些配置的 PR 已被合并。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B/tree/main">NousResearch/OLMo-Bitnet-1B at main</a>: 未找到描述</li><li><a href="https://wandb.ai/emozilla/olmo/runs/wwx8x2o3)">emozilla</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://hastebin.com/share/avuredixuq.yaml">Hastebin</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/discussions/12">NousResearch/Hermes-2-Pro-Mistral-7B · Add Padding Tokens</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1224260834235256874)** (4 messages): 

- **Traffic Signal Images for Vision Models**：分享了一个包含 [交通信号图像](https://huggingface.co/datasets/Sayali9141/traffic_signal_images) 的数据集，被认为对视觉模型的结构化输出和 tool-use 具有重要作用。注意，由于执行任意 Python 代码，Hugging Face 数据集查看器不支持此数据集。
- **Dataset Development Interest**：成员们对基于提供的交通信号图像链接构建数据集表示了兴趣。然而，未分享关于数据集构建的具体进展或细节。
- **Acknowledging Dataset Utility**：简要确认了提议的交通信号图像数据集的潜在用途。

**Link mentioned**: <a href="https://huggingface.co/datasets/Sayali9141/traffic_signal_images">Sayali9141/traffic_signal_images · Datasets at Hugging Face</a>: 未找到描述

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1224713092542365716)** (2 messages): 

- **Upload to Chain Fails Due to Hugging Face Metadata**：一位成员报告了上传到链时的问题，**Hugging Face** 意外地在模型元数据中添加了 `safetensors.sharded = true/false` 键。该键不被 Hugging Face Python 库的 `hf_api.py` 方法接受，导致无法加载推送和验证模型所需的 **ModelInfo**。
- **Seeking Solutions in Discord**：遇到上传问题的成员询问其他人是否也面临同样的问题，并寻求可能的解决方法。另一位成员提供了一个 Discord 消息链接，但此摘要中没有更多上下文。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1224583539249385523)** (5 messages):

- **探索 Scratchpad 的效用**：分享了一个示例，展示了如何使用 `<scratchpad>` 从 Claude 提示工程（prompt-engineering）指南中收集 RAG（检索增强生成）的证据。
- **关于 Scratchpad 与全上下文的辩论**：一位成员质疑使用 scratchpad 是否优于在提示词中包含全上下文，另一位成员给出了否定回答，建议拥有正确的上下文仍然是首选方法。
- **Scratchpad 在工作流中的角色**：另一位成员说明了 scratchpad 如何提供帮助，并提到了他们的工作流，其中包括类似于 scratchpad 的 `notes`，旨在用于用户交互。
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1224296713305591879)** (110 messages🔥🔥): 

- **展望文件上传与本地缓存**：成员们讨论了 **file uploading** 作为 **WorldSim** 功能的价值，认为它可以通过运行预写脚本来提高效率。另一个建议是维护 **local cache** 以模拟文件系统导航，以及提出“生成式虚拟机 (GVM)”的概念用于转储文件以保持一致性。

- **WorldSim 彩蛋被发现**：用户在 WorldSim 中发现了一个**愚人节彩蛋**。该彩蛋在讨论道德问题时触发，为交互增添了趣味性。

- **竞争性 WorldSim 挑战**：一位成员提出了类似于 “LLM Coliseum” 的概念，其中涉及 **WorldSim** 任务的类似竞争性基准测试可以在竞争环境中测试 LLM 之间的对抗，甚至可能由 LLM 担任比赛的裁判。

- **WorldSim 未来功能与路线图推测**：讨论了未来 **WorldSim** 功能的潜力，例如**竞争排行榜**、**text to video 集成**以及输入/输出能力。用户表达了对公开 **roadmap 或更新**的渴望，以提高未来开发的透明度。

- **WorldSim 作为 AI 对战平台**：分享了关于在 WorldSim 中进行 **AI battles** 的想法，其中“BBS 星群”可以运行各个时代的博弈，并出现统一对立面和哲学维度的新兴主题，包括寻宝和炼金术传说。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>：未找到描述</li><li><a href="https://worldsim-web.vercel.app/">world_sim</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.19459">Anomalous contribution to galactic rotation curves due to stochastic spacetime</a>：我们考虑了一种提出的量子引力替代方案，其中时空度量被视为经典的，即使物质场保持量子态。该理论的一致性必然要求...</li><li><a href="https://en.wikipedia.org/wiki/Core_War">Core War - Wikipedia</a>：未找到描述</li><li><a href="https://lostpedia.fandom.com/wiki/Hanso_Foundation">Hanso Foundation</a>：汉索基金会 (Hanso Foundation) 是由 Alvar Hanso 创立的组织，其目标是通过研究保护人类生命和促进福祉的方法来“伸向更美好的明天”。它最初...</li><li><a href="https://pastebin.com/DucD5J7N">div {  animation: none;  animation-play-state: paused;  line-height: 1.25; - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://tenor.com/view/her-theodore-joaquin-phoenix-scarlett-johannson-samantha-gif-5203383">Her Theodore GIF - Her Theodore Joaquin Phoenix - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://youtu.be/rSKMYc1CQHE?si=I6TPYRIMX9k6DUVE">Coding Adventure: Simulating Fluids</a>：让我们尝试说服一堆粒子表现得（至少在某种程度上）像水。使用 C# 和 HLSL 编写，在 Unity 引擎内运行。源代码：h...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1224342691660304476)** (244 messages🔥🔥): 

- **GPU 稳定性噩梦**：用户分享了使用 AMD GPU 时严重的系统不稳定体验，强调了在运行基准测试和对 GPU 进行压力测试后出现的内存泄漏和不可恢复错误等问题。报告了诸如 `"amdgpu: failed to allocate BO for amdkfd"` 和 `"amdgpu: Failed to create process device data"` 之类的错误，表明存在硬件/固件级别的问题。

- **AMD 的邀请遭到质疑**：一位用户在 AMD 的 subreddit 上引起对 GPU 重置问题的关注后，收到了 AMD Vanguard 计划的邀请。然而，George Hotz 等人对 AMD 解决底层问题的承诺表示怀疑，强调行动胜于言语，并强调开源文档和源码对于实质性进展的重要性。

- **tinygrad 社区对 AMD 的看法**：tinygrad 社区对 AMD 的软件和驱动程序方法感到明显的挫败。Hotz 预测，由于测试不力以及对现代软件实践的文化抵制，未来对 AMD MI300X 的大规模投资将会令人后悔。

- **处理 GPU 重置的方法**：讨论围绕应对崩溃后无法重置 AMD 显卡的变通策略，如 PCIe 电源循环和冗余。幽默地提出了各种轶事和潜在解决方案，如“PCIe 热插拔”或“GPU 组 RAID 1”，但这最终反映了问题的严重性。

- **对软件和固件实践的反思**：关于 AMD 需要在软件和固件实践方面进行根本性文化转变的对话正在进行。George Hotz 推测，通过正确的管理和测试协议（如 CI 和 fuzzing），有可能用更简单、更健壮的东西取代当前复杂的固件。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://looking-glass.io)">无标题</a>：未找到描述</li><li><a href="https://www.phoronix.com/news/AMD-Bridgman-Retires">AMD 长期开源 Linux 图形驱动倡导者退休 - Phoronix</a>：未找到描述</li><li><a href="https://forum.level1techs.com/t/radeon-7900xt-reset-bug/196270/4">Radeon 7900XT，重置 bug</a>：对于 AMD，本质是你要么得到一张会重置的显卡，要么得到一张不会重置的。由于设计是私有的，且 AMD 仍未设法修复重置问题，因此无法确定...</li><li><a href="https://www.reddit.com/r/Amd/comments/1bsjm5a/letter_to_amd_ongoing_amd/">Reddit - 深入了解</a>：未找到描述</li><li><a href="https://www.reddit.com/r/Amd/comments/1bsjm5">Reddit - 深入了解</a>：未找到描述</li><li><a href="https://mastodon.gamedev.place/@NOTimothyLottes/112190982123087000">NOTimothyLottes (@NOTimothyLottes@mastodon.gamedev.place)</a>：总之，看来我又遇到了一个无法规避且严重到无法忽视的编译器性能 bug。Wave-coherent（本应很快）的动态描述符选择表现得像批次中断...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/bec2aaf404aa3bdc569330a8d66c6678f8bfc459/examples/beautiful_mnist_multigpu.py">tinygrad/examples/beautiful_mnist_multigpu.py (GitHub)</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/geohot/7900xtx/blob/master/docs/MEC.md">7900xtx/docs/MEC.md (GitHub)</a>：通过在 GitHub 上创建账户来为 geohot/7900xtx 的开发做出贡献。
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224265332848328765)** (31 条消息🔥): 

- **线性 uOps 揭秘**：一位成员分享了一篇关于 [线性 uops 的文章](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md)，以帮助他人理解 tinygrad 中使用的中间表示。他们指出，虽然内容基于个人学习笔记，但欢迎反馈和建议。

- **命令队列（Command Queue）澄清**：围绕 tinygrad 中新的命令队列实现进行了讨论。分享了一个 [教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md)，解释了最近合并后的变化，指出命令队列是 "run_schedule" 函数的替代品。

- **测试代码谜题**：一个发往 tinygrad 的 Pull Request 提出了关于注释掉的 unittest 代码和后端检查的问题。在 [PR #4034](https://github.com/tinygrad/tinygrad/pull/4034) 中得到了澄清和修复，确保测试可以在 Intel GPU 上的 CLANG 和 OpenCL 等不同后端运行。

- **ShapeTracker 规范审查**：简要讨论了高级 ShapeTracker 的规范，涉及可合并性以及用数学符号 (Z^n) 表示形状的步长（strides），这些步长可以是负数。

- **关于 Jitted 函数的发现**：成员们试图理解为什么 Jitted 函数没有出现在命令队列日志中，讨论了 tinygrad 命令队列的操作层面以及它如何影响调度项的执行。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/backends.md">tinygrad-notes/backends.md 分支 main · mesozoic-egg/tinygrad-notes</a>: 通过在 GitHub 上创建账号，为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md">tinygrad-notes/commandqueue.md 分支 main · mesozoic-egg/tinygrad-notes</a>: 通过在 GitHub 上创建账号，为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops.md">tinygrad-notes/uops.md 分支 main · mesozoic-egg/tinygrad-notes</a>: 通过在 GitHub 上创建账号，为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4034">由 thedanhoffman 提交的重新启用 linearizer 测试的 has_local 检查 · Pull Request #4034 · tinygrad/tinygrad</a>: 在 learning tinygrad 频道中讨论过，结论是这可能是之前 PR 的一个疏忽。在运行 `CLANG=1 python3 -m pytest test/test_linearizer_overflows.py` 时触发。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3623/files">由 geohot 提交的恢复 ptx · Pull Request #3623 · tinygrad/tinygrad</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1224264931126415390)** (89 messages🔥🔥): 

- **LM Studio 错误故障**: 成员们在使用 LM Studio 时遇到错误，例如推理过程中的未知异常，以及在使用量化模型（Quantized models）发送几条消息后崩溃，特别提到了在 RTX 3060 GPU 上使用 *estopian maid 13B q4* 的问题。

- **在 LM Studio 中寻求语音理解模型**: 一位成员询问了能直接理解语音的模型，对此回复澄清说 **LM Studio 需要一个独立的工具** 来进行语音转文本，因为其内部没有内置 TTS (Text-to-Speech) 或 STT (Speech-to-Text) 功能，并提到了 *whisper.cpp* 作为例子。

- **为开发优化上下文长度和模型选择**: 出现了关于如何在 LM Studio 中管理上下文长度（Context Length）以及哪些模型最适合软件开发的讨论，并重申 **最佳实践和模型选择会根据用户的硬件而有所不同**。

- **LM Studio 设置和性能查询**: 用户正在寻求提高性能的技巧，建议通过设置中的 "GPU Offload" 选项让模型使用更多 GPU，并确认 **LM Studio 可以加载本地存储的 GGUF 文件**。

- **更新、降级和使用帮助**: 个人用户正在处理模型加载问题，寻求降级到 LM Studio 之前稳定版本的方法，并寻找特定且可能不存在的功能，例如对 PKL 模型或运行嵌入模型（embedding models）的支持——这两者目前都不受支持。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/welcome">欢迎 | LM Studio</a>: LM Studio 是一款用于在计算机上运行本地 LLM 的桌面应用程序。</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>: 欢迎来到非官方 LMStudio FAQ。在这里，你可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: 类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。</a>: 类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。 - moritztng/fltr</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp">llamafile/llamafile/sgemm.cpp 分支 main · Mozilla-Ocho/llamafile</a>: 通过单个文件分发和运行 LLM。通过在 GitHub 上创建账号，为 Mozilla-Ocho/llamafile 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1224270193015132161)** (44 messages🔥):

- **LM Studio 缺少文件输入功能**：成员们正在寻找一种向 LM Studio 中的模型提供文件的方法，类似于 OpenAI 的 API，但 **LM Studio 目前缺少此功能**。据指出，该功能已列入他们的 ToDo list。
- **不支持本地 Qwen MoE**：根据分享的 GitHub 拉取请求，[llama.cpp 对 Qwen MoE 的支持](https://github.com/ggerganov/llama.cpp/pull/6074) 正在开发中。
- **提出注重隐私的本地 LLM 解决方案**：建议将 LM Studio 与 AnythingLLM 结合使用，可以提供私密的本地 LLM + RAG 聊天机器人体验，尽管 LM Studio 本身缺乏内置的文档支持。
- **讨论用于 NSFW 内容的 7B 模型**：成员们正在讨论性能最好的无审查 7B 模型，如 **Nous-Hermes 2 Mistral DPO** 和 **Nous-Hermes-2-SOLAR-10.7B**，重点关注如何处理 NSFW 内容。
- **模型下载与执行的技术问题**：一些用户报告在本地下载和运行模型时遇到困难，出现静默失败，可能是由于 **缺乏代理支持** 或尝试在没有 GPU 的情况下加载大型模型。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.marktechpost.com/2024/03/31/mistral-ai-releases-mistral-7b-v0-2-a-groundbreaking-open-source-language-model/">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=-Rs8-M-xBFI&ab_channel=TimCarambat">使用这两个工具停止为 ChatGPT 付费 | LMStudio x AnythingLLM</a>：在本视频中，我们将安装两个用户友好的工具，用于下载、运行和管理强大的本地 LLM 以替代 ChatGPT。说真的...</li><li><a href="https://myanimelist.net/">MyAnimeList.net - 动漫数据库与社区 </a>：欢迎来到 MyAnimeList，全球最活跃的在线动漫社区和数据库。加入在线社区，创建你的动漫列表，阅读评论，探索论坛...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">由 simonJJJ 添加 qwen2moe · 拉取请求 #6074 · ggerganov/llama.cpp</a>：此 PR 为即将到来的 Qwen2 MoE 模型 hf 添加了代码支持。我更改了几个宏值以支持 60 个专家的设置。@ggerganov
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1224632728834539615)** (1 条消息): 

- **重新生成会覆盖之前的输出**：一位成员对 **继续生成** 会覆盖现有的助手输出表示担忧，而不是像以前那样创建一个新条目。他们希望回滚到以前的功能，即在每次新生成时拆分文本。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1224357313356365925)** (80 条消息🔥🔥): 

- **澄清 GPU 支持的困惑**：成员们澄清了运行多个 GPU **不需要 SLI**。[一位用户的设置表明](https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306) 系统会检测到合并的 VRAM，但实际的 VRAM 利用率受限于主 GPU。
- **NVIDIA Tesla P40 性能见解**：一篇分享的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) 详细介绍了双 **Tesla P40** 配置可以运行 70B 参数模型，性能约为 3-4 tokens/秒。
- **资源丰富的双 GPU 设置**：一位贡献者描述了一个使用 **三路 P40 GPU** 的预算型配置，透露 70B 模型在 8192 上下文长度下运行速度约为 4 tokens/秒。DIY 构建建议来自 [Mikubox Triple-P40 指南](https://rentry.org/Mikubox-Triple-P40)。
- **关于 VRAM 优于算力的辩论**：讨论中出现了比较 VRAM 价值与算力价值的观点，一位用户考虑用 **P40** 替换 **4090 GPU**，因为前者拥有更高的 VRAM，尽管 4090 的性能更优越。
- **平衡预算与性能**：成员们调侃了廉价硬件与理想性能之间的权衡，将其比作选择低端汽车改装，虽然能完成任务但在舒适度和质量上有所妥协。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rentry.org/Mikubox-Triple-P40">Mikubox Triple-P40 组装</a>：从 ebay 购买的 Dell T7910 “准系统（barebones）”，包含散热器。推荐卖家 “digitalmind2000”，因为他们使用现场发泡包装，确保工作站送达时完好无损。你可以自行选择 Xe...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306#:~:text=The%20card%20also%20has%2072,MHz%20(14%20Gbps%20effective).">NVIDIA Quadro RTX 8000 规格</a>：NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1224718735068364962)** (1 条消息): 

- **Autogen 的 Token 限制问题**：一位成员遇到了 *Autogen* 每次仅生成约 2 个 token 并错误地认为 Agent 已完成的问题。他们正在寻求建议，询问是否需要特殊配置才能让 **LM Studio** 与 *Autogen* 高效协作。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1224273178365394965)** (54 条消息🔥): 

- **模型蒸馏与 Claude3 Haiku 性能**：用户讨论了将 Claude3 Opus 等大型模型蒸馏（distillation）为 Claude3 Haiku 等更小、更高效模型的话题。一些人对 Haiku 的性能印象深刻，认为它足以应对许多原以为需要 GPT-4 的用例。
  
- **残差块讨论引发技术辩论**：围绕神经架构中的残差块（residual blocks）为何通常使用两个线性层展开了技术对话。用户解释说，带有非线性的两个层增加了表达能力，并允许灵活的参数化。

- **AI 数学奥林匹克竞赛参与**：提到 [Kaggle AI Mathematical Olympiad 竞赛](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview) 时，有人建议 EleutherAI 社区可以组队参赛。针对 “AI in science” 的计算赠款（Compute grants）可能会支持此类倡议。

- **资源共享与项目加入**：新成员介绍了自己，分享了他们在对齐（alignment）、隐私、大语言模型（LLM）微调以及自动形式化（autoformalisation）等领域的研究兴趣。他们希望为项目做出贡献并向社区学习。

- **构建可信的 Benchmarking 数据集**：一位用户询问在为新语言创建 Benchmarking 数据集时，是否有必要发表同行评审论文，寻求关于如何为数据集建立公信力的建议。

**提到的链接**：<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>：未找到描述

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1224272095111024660)** (111 条消息🔥🔥): 

- **破译 Google 的搜索基础设施**：讨论围绕 Google 将整个 Web 嵌入 RAM 以实现快速索引和检索的能力展开。参与者讨论了潜在的基础设施，评论指出 **Google 可能使用了分布式版本的 FAISS，并且主要在 RAM 中存储数据**以确保极快的响应时间，这对其业务至关重要。

- **沉思 Google 的编程方法**：在关于 Google 技术策略的进一步对话中，有人提到 Google 并不害怕使用像全局变量或 **`goto`** 这样“糟糕”的编程结构，只要它们能发挥作用。还提到了利用线程局部存储（thread local storage）来简化远程过程调用（RPC）中的上下文处理。

- **讨论文本索引的极限**：关于 Google 如何处理需要精确匹配的冷门文本搜索查询的问题引发了对 Google 使用 **倒排索引（inverted indexes）** 的解释。讨论了不同的索引策略，如全文搜索和倒排索引，以高效处理广泛且精确匹配的查询。

- **对新研究论文的期待**：大家对分享的新论文充满期待，特别是对 **LLM 安全过滤器的鲁棒性** 感兴趣。提供了一个近期研究的链接，并肯定了在该领域持续探索的必要性，包括防止对语言模型的逆向工程或滥用。

- **随性揭晓流畅的 OSS Agent**：分享了一个来自 Princeton NLP 的开源软件 Agent 链接，名为 **SWE-Agent**，该 Agent 声称在软件工程任务中的表现与 Devin 等私有 Agent 相当。作为 NLP 领域前沿开源贡献的一个案例，这引起了广泛关注。
<div class="linksMentioned">

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://swe-agent.com/">SWE-Agent</a>: 未找到描述</li><li><a href="http://arxiv.org/abs/2403.20327">Gecko: Versatile Text Embeddings Distilled from Large Language Models</a>: 我们介绍了 Gecko，一个紧凑且多功能的文本嵌入模型。Gecko 通过利用一个关键思想实现了强大的检索性能：将知识从大语言模型 (LLMs) 蒸馏到检索...</li><li><a href="https://en.wikipedia.org/wiki/Inverted_index?wprov=sfla1">Inverted index - Wikipedia</a>: 未找到描述</li><li><a href="https://x.com/anthropicai/status/1775211248239464837?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Anthropic (@AnthropicAI)</a>: 新的 Anthropic 研究论文：Many-shot jailbreaking。我们研究了一种在大多数大语言模型上都有效的长上下文越狱技术，包括由 Anthropic 开发的模型以及许多其他...</li><li><a href="https://x.com/blancheminerva/status/1774901289773584531?s=46">Tweet from Stella Biderman (@BlancheMinerva)</a>: 众所周知，微调可能会顺带移除 RLHF 防护 https://arxiv.org/abs/2310.03693。你能通过在数据中混合拒绝回答的示例来解决这个问题吗？这些是否重要...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1224340286151458858)** (4 messages): 

- **可视化 Sparse Autoencoder 特征**：发布了一个用于 Sparse Autoencoder (SAE) 特征的新 [可视化库](https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sae-vis-announcement-post-1)，因其在 AI alignment 背景下阐明 SAE 特征结构的作用而受到称赞。
- **聚光灯下的 SAE 特征**：分享了一篇 [论坛帖子](https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1)，探讨了 Sparse Autoencoder 特征是在揭示模型的属性还是仅仅反映数据分布，承认了这一问题的复杂性及其与 AI alignment 的相关性。
- **抽象动物架构？**：一位成员对将房子概念化为介于“具体长颈鹿”和“抽象长颈鹿”之间的东西表示困惑，展示了在 AI 模型中对特征进行分类的复杂性以及有时带有的奇思妙想。
- **表情符号胜过千言万语**：一位成员用耸肩表情符号 🤷‍♂️ 回应了对特征分类的困惑，表示对 AI interpretability 中固有歧义的困惑或接受。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/neelnanda5/status/1774463606656282806">Tweet from Neel Nanda (@NeelNanda5)</a>: 来自 @calsmcdougall 的出色 Sparse Autoencoder 特征可视化库！我的团队已经发现它非常有用，快去看看吧：https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sa...</li><li><a href="https://www.lesswrong.com/posts/BK8AMsNHqFcdG8dvt/a-selection-of-randomly-selected-sae-features-1">A Selection of Randomly Selected SAE Features — LessWrong</a>: 在这篇文章中，我们解释了一小部分 Sparse Autoencoder 特征，这些特征揭示了模型中明显的、有意义的计算结构……
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1224362016030986413)** (18 messages🔥): 

- **音乐创作 AI 综述**：一个 [arXiv 论文链接](https://arxiv.org/abs/2402.15294) 讨论了当前使用 **GANs** 和 **Transformers** 进行音乐创作的研究，重点关注风格复制和迁移，并建议未来的工作可以包括排行榜上的 text-to-music 指标，暗示了音乐 AI 评估的未来方向。
- **使用 lm-eval-harness 对 Claude 进行基准测试？**：一位用户询问是否有 **Anthropic Claude** 模型的 **lm-eval-harness** 结果库，表明在 API 支持和任务适配方面存在空白和潜在的分享发现的领域。
- **lm-eval-harness 中的 ValueError 故障排除**：一位用户报告了尽管使用了 **DEBUG verbosity** 仍持续出现的 `ValueError` 问题，并被建议分享 YAML 配置文件内容以获得进一步的故障排除帮助。
- **多语言能力凸显语系细微差别**：一位成员将 **arc_challenge** 机器翻译成多种语言，发现即使是没有经过特定语言训练的模型，如果该语言属于相关语系，也能表现良好，建议使用更多生成式评估来测试语言能力。
- **已编译 LM 评估任务列表**：经过讨论和调试工作，一位用户通过搜索 GitHub 成功编译了一份需要 *generate until* 功能的 **lm-evaluation-harness** 任务列表，促进了合适评估基准的发现。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.15294">A Survey of Music Generation in the Context of Interaction</a>：近年来，机器学习，特别是生成对抗网络 (GANs) 和基于注意力机制的神经网络 (Transformers)，已成功用于作曲和生成...</li><li><a href="https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+output_type%3A+generate_until&type=code">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1224452465437442178)** (3 条消息): 

- **寻求多模态视觉研究小组**：一位成员为其博士提案咨询专注于计算机视觉任务（如偏见检测和情绪识别）的研究小组或 Discord 频道。他们请求获取有关关键论文的信息，以及关于计算机视觉或多模态应用中偏见的广泛观点。
- **建议将 LAION 用于计算机视觉**：一位参与者建议将 **LAION** 作为一个社区，但指出它可能没有计算机视觉中偏见检测和情绪识别的特定子研究方向。
- **发现 Prisma 的多模态小组**：另一位成员提到 **Prisma multimodal group** 相当活跃，尽管它并不专门研究上述子方向。他们附上了一个 [推文链接](https://twitter.com/soniajoseph_/status/1767963443699790256) 供参考。
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1224409336051273768)** (2 条消息): 

- **不均匀的 Batch Size 可能会导致瓶颈**：一位用户指出，虽然可以对 **GPT-NeoX** 进行修改以使用不均匀的 Batch Size，但这样做可能会导致负载不均衡。最大的 Batch 将成为过程中的瓶颈，因为系统需要等待处理这些 Batch 的 GPU。
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1224408081409118298)** (1 条消息): 

- **无需等待即可使用 ChatGPT**：OpenAI 推出了[无需注册即可立即使用 ChatGPT](https://openai.com/blog/start-using-chatgpt-instantly) 的选项，旨在让更广泛的受众能够接触到 AI。该工具目前每周已有 185 个国家的超过 1 亿人使用。

- **无需账号，没问题**：公司正在逐步推出此功能，致力于实现让 ChatGPT 等 AI 工具广泛可用的使命。

- **您的输入有助于改进 AI**：用户与 ChatGPT 的交互可能会被用于改进模型性能，但即使不创建账号，也可以在设置中选择退出。有关数据使用的更多详细信息可以在 [帮助中心](https://help.openai.com/en/articles/5722486-how-your-data-is-used-to-improve-model-performance) 找到。

**提到的链接**：<a href="https://openai.com/blog/start-using-chatgpt-instantly">立即开始使用 ChatGPT</a>：我们正在让人们更轻松地体验 AI 的好处，而无需注册。

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1224279806548508672)** (95 条消息🔥🔥): 

- **Fine-Tuning 中的细微差别**：成员们讨论了使用 "\n" 或换行符如何*影响语言模型的 Fine-Tuning*，并指出 **LLMs 会学习它们被赋予的内容**，且类人格式化可能是有益的。
- **AI 歌曲识别失误**：一项 AI 对比显示 **ChatGPT、Gemini 和 Claude** 均未能正确列出 Laurent Voulzy 的歌曲 "Rockollection" 中提到的歌曲，这促使一位用户强调了在提供正确列表时 AI 的不同反应和**创意限制**。
- **探索 AI 生成内容的原创性**：有一场关于语言模型是仅仅在机械重复内容还是能够创造原创内容的对话。一项 **[研究](https://arxiv.org/abs/2310.17567)** 表明，AI 可以以训练数据中未出现的方式组合技能，强调 **AI 可能会表现出超出简单复制的涌现行为**。
- **AI 图像描述寻求者**：用户讨论了他们对用于描述图像的 AI 工具的寻找，提到了 **Claude 3 Sonnet** 和 **Midjourney** 的 `/describe` 命令，同时也观察到了此类工具的有效性及其可用性的局限性。
- **AI 应用兼容性问题**：一位用户强调了在三星 Galaxy Note 9 上访问 AI 应用时遇到的麻烦，讨论了潜在的兼容性或支持问题，并建议验证 **Android 和 Play Store 版本**，以及使用移动浏览器作为变通方法。

**提及链接**：<a href="https://arxiv.org/abs/2310.17567">Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models</a>：随着 LLM 的角色从语言统计建模转向作为通用 AI Agent，LLM 的评估方式应该如何改变？可以说，AI Agent 的一项关键能力是灵活地协作...

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1224262417563910235)** (38 条消息🔥): 

- **探索 LLM 的反思能力**：成员们讨论了提示 LLM 进行内部反思的可能性，其中一位分享了见解：虽然 LLM 作为一个文本预测器运行，但有效地构建 Prompt 可以避免逻辑跳跃，并引用了 OpenAI 官方关于 Prompt Engineering 的指南，详见[此处](https://platform.openai.com/docs/guides/prompt-engineering/give-the-model-time-to-think)。

- **愚人节技术玩笑？**：一位成员标记了一个链接，推测其为愚人节玩笑，而另一位成员则确认所讨论的功能是可以运行的，尽管由于权限限制无法提供截图。

- **GPT 模型对比及对 GPT-5 的期待**：对话包括了关于 **GPT-4** 尽管上下文窗口较小，但在感知上优于 **Opus** 的评论，并表达了对 GPT-5 的期待，认为一旦它包含更好的推理、代码解释和互联网访问功能，将成为强有力的竞争者。

- **服务器稳定性问题引起用户关注**：几位成员遇到了服务器稳定性问题，影响了他们登录和使用服务，其中一人报告了长时间持续的“Error in input stream”错误，并征求已知的解决方案。

- **多样化的 AI 利用与开发**：用户分享了他们在不同 AI 服务上的开发和经验，包括一个用于查找角色详情的自定义 GPT，以及关于 **DALL-E 3** 中新图像编辑功能可用性的讨论，并链接到了官方说明，详见[此处](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e)。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1224447127976153118)** (7 条消息): 

- **PDF 转 JSON 的难题**：一位成员询问了使用 GPT 将 PDF 转换为 JSON 对象的最佳方法，并思考使用 Schema 还是开放式 Prompt 会更有效。另一位用户建议始终发送 Schema，尽管他们注意到该过程的效果可能相当随机。

- **PDF 转换的 TOS 警告**：有人指出，将 PDF 转换为 JSON 可能违反了服务条款（TOS）。

- **征集研究参与者**：在亚美尼亚美国大学进行研究的计算机科学毕业生 Anna 邀请 ML engineers、内容创作者、Prompt engineers 和其他语言模型用户参加一个 20 分钟的访谈，讨论与大语言模型相关的挑战。

- **寻求替代管理工作的 Prompt**：一位成员请求关于替代管理任务的有效 Prompt 建议，重点是针对中层到 C-suite 管理职位的指令划分和绩效规划。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1224447127976153118)** (7 条消息): 

- **选择最佳 JSON 处理方法**：成员们正在讨论使用 GPT 从 PDF 中提取 JSON 的最佳方式。虽然一位成员尝试过指定 JSON Schema，但另一位成员正在尝试一种更开放的方法，让 GPT 尽可能多地捕获数据。

- **强制执行 Schema 时的随机结果**：在将文档转换为 JSON 的过程中，讨论了向 GPT 提供 Schema 的问题，经验表明成功程度各异，且结果中存在不可预测的因素。

- **了解 LLM 使用案例**：刚毕业并处于研究阶段的 Anna 正在寻求与 ML engineers 和其他专业人士讨论他们在应用大语言模型方面的经验和挑战，请感兴趣的人员私信或回复以进行可能的会面。

- **探索替代管理工作的 Prompt**：一位成员正在寻求关于中层和 C-suite 管理任务（如划分指令和绩效计划）的优秀管理替代 Prompt 建议，暗示了自动化管理功能的潜在进展。

---

**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1224391599367258227)** (1 条消息):

- **深入了解 LlamaIndex 的 RAFT 网络研讨会**：参加 LlamaIndex 关于 **检索增强微调 (Retrieval-Augmented Fine-Tuning, RAFT)** 的特别网络研讨会，届时将由首席共同作者 Tianjun Zhang 和 Shishir Patil 带来深入讲解。点击[此处](https://lu.ma/v1bdat63)注册定于本**周四上午 9 点（太平洋时间）**举行的活动。

- **通过即将举行的网络研讨会理解 RAFT**：本次网络研讨会将探讨 RAFT 如何结合检索增强生成 (RAG) 和微调的优势，以提升语言模型在特定领域设置中的性能。欢迎在**周四上午 9 点（太平洋时间）**参与，向该技术背后的专家学习。

- **为 RAFT 爱好者准备的补充资源**：如需了解 RAFT 方法论的更多背景信息，请查看专门的 RAFT [博客文章](https://gorilla.cs.berkeley.edu/blogs/)，并查阅完整的 RAFT [论文](https://arxiv.org/pdf/2403.10131.pdf)，为网络研讨会做好准备。

- **生成你自己的 RAFT 数据集**：感谢 @ravithejads，你现在可以使用 LlamaIndex 提供的 **RAFTDatasetPack** 创建 RAFT 数据集。点击[此处](https://llamahub.ai/l/llama-packs/llama-index-packs-raft-dataset?from=)获取该工具包，并在 [GitHub](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raft-dataset/examples/raft_dataset.ipynb) 上查看相应的 notebook。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/v1bdat63">LlamaIndex 网络研讨会：检索增强微调 (RAFT) · Zoom · Luma</a>：RAFT - 检索增强微调 🔥 Zhang 等人提出的检索增强微调 (RAFT) 是一种针对特定领域 RAG 微调预训练 LLM 的新技术...</li><li><a href="https://x.com/llama_index/status/1774814982322172077?s=20">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：新的 LlamaIndex 网络研讨会 🚨 - 快来学习如何进行检索增强微调 (RAFT)！做 RAG 就像是不复习就参加开卷考试。这仅比闭卷考试好一点点...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1224374342494847108)** (4 条消息): 

- **RAFT 网络研讨会预告**：*LlamaIndex* 宣布了一个聚焦于 **检索增强微调 (RAFT)** 的[新网络研讨会](https://twitter.com/llama_index/status/1774814982322172077)，将其比作参加“开卷考试”，并宣传了其相比传统语言模型微调的优势。
- **Mesudarshan 编写的高级 PDF RAG 教程**：推荐了 @mesudarshan 的一个教程，演示了如何使用 LlamaParse 和本地模型构建高级 **PDF 检索增强生成 (RAG)**，强调了提取是至关重要的一步。该过程使用了来自 *GroqInc* 的模型和来自 @qdrant_engine 的 *FastEmbed*，详见此[推文](https://twitter.com/llama_index/status/1774832426000515100)。
- **逐步构建 RAG 指南**：分享了 @lambdaEranga 构建包含 @llama_index、@ollama 和 @huggingface 工具的本地模型 RAG 的详细**示意图**，展示了为了开发目的将其封装在 *Flask server* 中的实用性。这是 Twitter 上的[指南](https://twitter.com/llama_index/status/1774950945488834684)。
- **用于高级 RAG 重排序的 RankLLM**：为构建高级 RAG 系统的人员推荐了由 @rpradeep42 等人开发的 **RankLLM**，这是一个经过**微调用于重排序 (reranking)** 的开源 **LLM** 集合。推文中强调了选择合适重排序器的重要性，并特别提到了 RankZephyr 模型，详见[推文](https://twitter.com/llama_index/status/1775166279911186930)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1224262634547707964)** (118 条消息 🔥🔥):

- **使用 OpenAI LLM 设置 LlamaIndex**：一位用户在指定 OpenAI LLM 的 organization id 时遇到问题。正确的方法是在代码中设置 `openai.organization`，或者在初始化 `OpenAI` 时传递 organization 参数，例如 `Settings.llm = OpenAI(organization="orgID",...)`。
- **应对过时的 LlamaIndex 文档**：用户反映对过时的教程和指向 GitHub 文件的失效链接感到沮丧。他们正在寻求结构化数据和自然语言 SQL 查询的指导，以及从已弃用的类（如 `NLSQLTableQueryEngine`）向新类（如 `SQLTableRetriever`）的过渡。
- **自定义 Agent 与错误排查**：讨论围绕使用 `OpenAIAgent.from_tools` 创建自定义 Agent、处理 ollama 等服务器端点的 `404 Not Found` 错误，以及在 RAG 中实现天气查询（使用 **WeatherReader**）等功能展开。
- **处理已弃用的 OpenAI 模型**：由于 `text-davinci-003` 模型已被弃用，一位用户遇到了错误。建议在他们的 `GPTVectorStoreIndex` 设置中将该模型替换为 `gpt-3.5-turbo-instruct`。
- **使用 LlamaParse 处理 PDF 中的图像**：关于处理 PDF 文件中图像的咨询被引导至 **LlamaParse**（一种能够读取图像的专有解析工具），以及涉及 `unstructured` 提取和 Vector Store 摘要的替代手动方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/phase1-collect-underpants-gnome-south-park-phase2-gif-22089237">Phase1 Collect Underpants GIF - Phase1 Collect Underpants Gnome - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">Introducing LlamaCloud and LlamaParse — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/prompts">llama_index/llama-index-core/llama_index/core/prompts at main · run-llama/llama_index</a>: LlamaIndex 是为您 LLM 应用准备的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/readers/weather/?h=weather">Weather - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_camel/?h=huggingface">HuggingFace LLM - Camel-5b - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/?h=custom#example-using-a-custom-llm-model-advanced">Customizing LLMs - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/">Building RAG from Scratch (Open-source only!) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/#accessing-prompts">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/agronholm/sqlacodegen?tab=readme-ov-file">GitHub - agronholm/sqlacodegen: Automatic model code generator for SQLAlchemy</a>: SQLAlchemy 的自动模型代码生成器。通过在 GitHub 上创建账号来为 agronholm/sqlacodegen 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama?tab=readme-ov-file#rest-api">GitHub - ollama/ollama: Get up and running with Llama 2, Mistral, Gemma, and other large language models.</a>: 快速上手 Llama 2, Mistral, Gemma 和其他大语言模型。 - ollama/ollama</li><li><a href="https://github.com/run-llama/create-llama?tab=readme-ov-file#customizing-the-llm">GitHub - run-llama/create-llama: The easiest way to get started with LlamaIndex</a>: 开始使用 LlamaIndex 的最简单方法。通过在 GitHub 上创建账号来为 run-llama/create-llama 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data/">Structured Data - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/#part-2-query-time-retrieval-of-tables-for-text-to-sql">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_personality/">Chat Engine with a Personality ✨ - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/8a8324008764a7fefb6f25b0e3aac81089590322/llama-index-legacy/llama_index/legacy/prompts/system.py#L4">llama_index/llama-index-legacy/llama_index/legacy/prompts/system.py at 8a8324008764a7fefb6f25b0e3aac81089590322 · run-llama/llama_index</a>: LlamaIndex 是为您 LLM 应用准备的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1224493601124388954)** (4 条消息): 

- **Top Agent 简化问题**：一位成员报告了在使用 Agents 构建 *multi-document rag system* 时遇到的问题，其中 *top_agent* 过度简化了问题。例如，关于巧克力保质期的查询被简化为仅“保质期”，导致搜索结果不理想。

- **具体查询简化示例**：该成员进一步举例说明了此问题，用户询问灭火器的保质期，但 Agent 仅使用“保质期”一词查询 *retrieval engine*。

- **IPEX-LLM 和 LlamaIndex 可能彻底改变聊天和文本生成**：分享了一篇标题为“使用 IPEX-LLM 和 LlamaIndex 开启文本生成和聊天的未来”的 Medium 文章链接，讨论了这些工具对未来文本生成和聊天应用的潜在影响。[点击此处阅读文章](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2)。

- **教程提醒：使用 LlamaIndex 创建 RAG 应用**：一位成员分享了一个 YouTube 视频教程，提供了使用 LlamaIndex、Pinecone 和 Gemini Pro 构建简单 RAG 应用的逐步指南。涵盖了抓取内容、转换为 vector embeddings、存储在 Pinecone 索引以及使用 LlamaIndex 查询 Gemini Pro 等关键流程。[在此观看教程](https://youtu.be/B9mRMw0Jhfo)。

**提到的链接**：<a href="https://youtu.be/B9mRMw0Jhfo">How to build a RAG app using Gemini Pro, LlamaIndex (v0.10+), and Pinecone</a>：让我们来讨论如何使用 LlamaIndex (v0.10+)、Pinecone 和 Google 的 Gemini Pro 模型构建一个简单的 RAG 应用。这是一个为刚接触该领域的开发者准备的逐步教程...

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1224332768759255173)** (109 条消息🔥🔥): 

- **使用 LangChain 处理复杂的 JSON**：一位用户在处理 JSON 时遇到困难，每行 JSON 都会创建一个 Document，而不是为整个 JSON 创建一个带有 metadata 的 Document。他们询问了解决方案，但目前没有后续跟进，原始问题在 [JSON loader 文档](https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json)中有描述。
  
- **使用 Tool 的 Agent Token 使用量增加**：一位用户注意到使用 Tool 的 Agent Token 使用量增加了 50%。对此解释是，Tool 会检索并进行 tokenize 数据，因此会消耗更多 Token；system prompt 会在推理假设时运行一次，但并非每个 Tool 都需要它。

- **关于 LangGraph 和 Structured Tool 验证的讨论**：用户讨论了在 LangGraph 中使用 base model 作为 state 的可能性，并提供了一个 GitHub [notebook 示例](https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb)。此外，还分享了来自 [GitHub issues](https://github.com/langchain-ai/langchain/issues/8066) 和 LangChain 文档的说明，关于如何使用 Pydantic 的 `BaseModel` 和 `Field` 类来验证 LangChain 中 StructuredTool 的字段。

- **结构化输出与 Fine-tuning 的问题**：用户讨论了从 chain 获取结构化输出以及在对模型进行 Fine-tuning 后保留基础知识相关的问题。一位用户建议使用两个 Agent，一个经过 Fine-tuning 的 Agent 和一个常规的 GPT 模型，以同时保持专业知识和通用知识。对话中没有发布关于结构化输出问题的最终解决方案。

- **使用 LangChain 在 PDF 之间映射内容**：一位用户尝试使用带有 RetrievalQA chain 的 RAG 在 PDF 之间映射相关内容，并被建议尝试使用 vector embeddings 根据语义内容匹配段落。他们还询问了在 LangHub 中处理图像时遇到的反序列化错误，但对话中同样没有提供解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:8000.>">未找到标题</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/templates/openai-functions-agent#usage>).">openai-functions-agent | 🦜️🔗 Langchain</a>: 此模板创建了一个使用 OpenAI function calling 来传达其采取何种行动决策的 Agent。</li><li><a href="https://python.langchain.com/docs/guides/structured_output">[beta] Structured Output | 🦜️🔗 Langchain</a>: 让 LLM 返回结构化输出通常至关重要。这是</li><li><a href="https://js.langchain.com/docs/integrations/document_loaders/file_loaders/json">JSON files | 🦜️🔗 Langchain</a>: JSON 加载器使用 JSON pointer 来定位 JSON 文件中你想要的目标键。</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html#langchain.chains.structured_output.base.create_structured_output_runnable.">langchain.chains.structured_output.base.create_structured_output_runnable &mdash; 🦜🔗 LangChain 0.1.14</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34f/examples/state-model.ipynb">langgraph/examples/state-model.ipynb at 961ddd49ed498df7ffaa6f6d688f7214b883b34f · langchain-ai/langgraph</a>: 通过在 GitHub 上创建账号，为 langchain-ai/langgraph 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1358>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8066>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/tool_error_handling#tryexcept-tool-call>).">Tool error handling | 🦜️🔗 Langchain</a>: 使用模型调用工具时，存在一些明显的潜在失败模式。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13662>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langgraph/blob/961ddd49ed498df7ffaa6f6d688f7214b883b34">GitHub - langchain-ai/langgraph at 961ddd49ed498df7ffaa6f6d688f7214b883b34f</a>: 通过在 GitHub 上创建账号，为 langchain-ai/langgraph 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1224453985591759021)** (5 条消息): 

- **推崇将 Langgraph 用于对话机器人**：一位成员赞扬了 **langgraph**，因为它使得实现循环（cycles）变得非常容易，并强调了其在创建高级对话式任务机器人中的重要性。这一特性使其区别于其他 LLM 应用框架，并将通过社区贡献（如博客文章）进行进一步记录。

- **使用 OpenGPTs 实现自定义食物订购**：一位用户通过集成自定义食物订购 API 展示了 **OpenGPTs** 的可扩展性，证明了该平台对自定义 AI 应用的适应能力。他们正在为其名为 "[Hack OpenGPT to Automate Anything](https://youtu.be/V1SKJfE35D8)" 的 YouTube 演示寻求反馈。

- **PersonaFinder GPT 已发布**：一位成员开发了 **PersonaFinder GPT**，这是一款对话式 AI，可以根据姓名、国家和职业提供个人信息。该工具可在 [PersonaFinder Pro](https://chat.openai.com/g/g-xm4VgOF5E-personafinder-pro) 进行测试。

- **征集资深 Prompters 测试新工具**：有一项请求，希望资深提示词工程师（prompters）测试并反馈一款新工具，该工具旨在实现自动化代码转换，以维护生产部署的代码标准和质量。工具访问地址见[此处](https://tinyurl.com/gitgud-langchain)。

- **分享了 Kleinanzeigen 广告**：分享了一个 **Kleinanzeigen** 上的图片广告，尽管它似乎与 LangChain AI Discord 上的 AI 或项目无关。可以在[此处](https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios)查看 **Mona Bild**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tinyurl.com/gitgud-langchain">GitGud</a>: 未找到描述</li><li><a href="https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios">Mona Bild repost</a>: 来自 tiktok 的知名图片 -，Wuppertal - Elberfeld-West 的 Mona Bild repost</li><li><a href="https://youtu.be/V1SKJfE35D8">Hack OpenGPT to Automate Anything</a>: 欢迎来到定制 AI 应用的未来！本演示展示了 OpenGPTs 的惊人灵活性和强大功能，这是一个由 LangChain 发起的开源项目。W...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1224266681153749002)** (79 messages🔥🔥): 

- **LinkedIn 的虚拟徽章引发热议**：LinkedIn 用户炫耀获得了 30 多个 Top Voice 徽章并分享了[帖子链接](https://www.linkedin.com/posts/jillanisofttech_datascience-artificialintelligence-analyticalskills-activity-7180469402079789056-oPrY?utm_source=share&utm_medium=member_desktop)，而其他人则质疑收集此类徽章的实际益处。
- **AI 虚构软件包的危害**：[The Register 报道](https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/)了一个令人震惊的发现，一个此前由生成式 AI 幻觉产生的软件包已被真实创建，导致包括阿里巴巴在内的几家大企业错误地引入了它。若非该测试包是良性的，可能会开启恶意软件威胁的大门。
- **PPO 算法性能挑战**：一名成员分享了一条曲线，显示 Proximal Policy Optimization (PPO) 算法存在问题，其他人建议在 Agent 性能下降时回滚到之前的 checkpoint。
- **Stable Diffusion 在 AI 图像生成领域的演进**：成员们讨论了 Stable Diffusion 的进展，包括 *stable cascade* 和 OOT diffusion 等新变体，以及引入了 ControlNet 等新工具以实现更多输入控制；同时，其他人强调了与 VRAM 需求相关的持续性能问题。
- **集成语言模型并解决误区**：一位用户寻求将语言模型与其代码库集成的帮助，讨论了 SDK 的利用，并表达了对更换供应商和在不重新编码的情况下修改 prompt 的担忧；而其他人则在探讨 Chatbot 开发，并就如何避免 AI 在回复中披露其身份寻求建议。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wccftech.com/chinese-ai-firm-unveils-deepeye-ai-box-featuring-up-to-48-tops-affordable-designs/">Chinese AI Firm Unveils &quot;DeepEye&quot; AI Box, Featuring Up To 48 TOPS &amp; Affordable Designs</a>: 中国 AI 公司发布 &quot;DeepEye&quot; AI 盒子，具备高达 48 TOPS 的算力及亲民的设计，承诺提供具有广泛升级路线图的经济型 AI 解决方案。</li><li><a href="https://www.theregister.com/2024/03/28/ai_bots_hallucinate_software_packages/">AI bots hallucinate software packages and devs download them</a>: 只需留意 ML 想象出的库并将其变为现实，并加入实际的恶意代码。不，等等，别那么做。</li><li><a href="https://www.reddit.com/r/photoshop/comments/r7c2bh/evenout_lighting_for_a_tileable_texture/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: 像 grep 一样，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。- moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

docphaedrus: https://youtu.be/7na-VCB8gxw?si=azqUL6dGSMCYbgdg
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1224350511785050323)** (5 messages): 

- **FastLLM 突破十亿 Token 障碍**：**FastLLM (FLLM)** 是 Qdrant 为 **Retrieval Augmented Generation** (RAG) 开发的轻量级语言模型，现已进入早期访问阶段，拥有令人印象深刻的 **10 亿 Token** 上下文窗口。它专门设计用于与 Qdrant 集成，预示着 AI 驱动的内容生成和检索能力的革命。在他们的[公告帖子](https://qdrant.tech/blog/fastllm-announcement/)中阅读更多信息。

- **考虑熵的强化学习**：分享了一篇关于 *Soft Actor-Critic* 的学术论文，这是一种 **Off-Policy Maximum Entropy Deep Reinforcement Learning** 方法，提供了关于强化学习中 **stochastic actor** 方法的见解。全文可在 [arXiv](https://arxiv.org/pdf/1801.01290v2.pdf) 上找到。

- **寻找合适的开源状态页**：Medium 上的一篇博文介绍了 **2024 年 6 个最佳开源状态页替代方案**，为寻求高效监控和传达系统状态的开发者及团队提供了见解。全文可在 [Medium](https://medium.com/statuspal/6-best-open-source-status-page-alternatives-for-2024-b68e5a967cc1) 阅读。

- **IPEX-LLM 和 LlamaIndex 引领潮流**：一篇新的 Medium 文章讨论了 **IPEX-LLM 和 LlamaIndex** 作为文本生成和聊天功能领域潜在的游戏规则改变者。关于这些先进工具的详细文章可在此处 [访问](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2)。

**提到的链接**：<a href="https://qdrant.tech/blog/fastllm-announcement/">Introducing FastLLM: Qdrant’s Revolutionary LLM - Qdrant</a>：轻量级且开源。专为 RAG 定制，并与 Qdrant 完全集成。

  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1224297516200230972)** (12 条消息🔥): 

```html
<ul>
  <li><strong>机器人意识流：</strong> 介绍 <strong>LLMinator</strong>，一个上下文感知的流式 Chatbot，支持通过 Langchain 和 Gradio 在本地运行 LLM，兼容 HuggingFace 的 CPU 和 CUDA。在 <a href="https://github.com/Aesthisia/LLMinator">GitHub</a> 上查看。</li>
  <li><strong>数据管理变得更简单：</strong> DagsHub 为 Colab 推出了与 DagsHub Storage Buckets 的新集成，承诺提供更好的数据管理体验，类似于面向 ML 的可扩展 Google Drive。示例 Notebook 可在 <a href="https://colab.research.google.com/#fileId=https%3a%2f%2fdagshub.com%2fDagsHub%2fDagsHubxColab%2fraw%2fmain%2fDagsHub_x_Colab-DagsHub_Storage.ipynb">Google Colab</a> 上获取。</li>
  <li><strong>Python 的新对手 Mojo：</strong> 关于 Mojo 编程语言在性能上超越 Python 的推测不断涌现，正如标题为 "Mojo Programming Language killed Python" 的 YouTube 视频中所讨论的那样。在此处观看完整解释 <a href="https://youtu.be/vDyonow9iLo">here</a>。</li>
  <li><strong>机器人展示：</strong> 一位成员构建了一个带有颜色传感器的高级巡线和避障机器人，并在 SUST_BlackAnt 的 YouTube 视频中进行了演示。在此处查看完整演示 <a href="https://www.youtube.com/watch?v=9YmcekQUJPs">here</a>。</li>
  <li><strong>使用 OneMix 启动 SaaS：</strong> 新的 SaaS 样板项目 OneMix 声称通过提供着陆页、支付和身份验证设置等必需品来加速项目启动。更多详情请访问 <a href="https://saask.ing">saask.ing</a>，演示视频请见 <a href="https://www.youtube.com/watch?v=NUfAtIY85GU&t=8s&ab_channel=AdityaKumarSaroj">YouTube</a>。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/#fileId=https%3a%2f%2fdagshub.com%2fDagsHub%2fDagsHubxColab%2fraw%2fmain%2fDagsHub_x_Colab-DagsHub_Storage.ipynb">Google Colaboratory</a>: 未找到描述</li><li><a href="https://x.com/__z__9/status/1774965364301971849?s=20">ً ‎ (@__z__9) 的推文</a>: 新预印本！首个经过红队测试的多语言开源持续预训练 LLM —— **Aurora-M**，符合 #WhiteHouse 关于安全、可靠和值得信赖的开发的行政命令...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: 首个根据美国行政命令进行红队测试的开源多语言语言模型</a>: 预训练语言模型是多种 AI 应用的基础，但其高昂的训练计算成本限制了可访问性。BLOOM 和 StarCoder 等倡议旨在使 p... 的访问民主化</li><li><a href="https://youtu.be/7na-VCB8gxw?si=4u6MDWEfCT3e0b0S">关于虚拟容器：保护你的 Python 应用</a>: 一个简短的演示，介绍如何使用 Podman (https://podman.io/) 或 Docker (http://docker.io/) 等平台将你的软件项目“容器化”以用于生产环境....</li><li><a href="https://youtu.be/vDyonow9iLo">Mojo 编程语言杀死了 Python</a>: 我将与你分享为什么 Mojo 很快就会变得非常流行。它在性能方面正在击败 Python，使其极具竞争力，关键在于：在保持 th... 的同时</li><li><a href="https://www.youtube.com/watch?v=9YmcekQUJPs">带有颜色传感器的高级循迹和避障机器人。由 SUST_BlackAnt 展示</a>: 这是一个高级循迹轨道。欢迎点赞、评论和分享。让我知道你是否喜欢它。如果你想联系我，请随时发送 em...</li><li><a href="https://github.com/Aesthisia/LLMinator">GitHub - Aesthisia/LLMinator: 基于 Gradio 的工具，可直接从 Huggingface 运行开源 LLM 模型</a>: 基于 Gradio 的工具，可直接从 Huggingface 运行开源 LLM 模型 - Aesthisia/LLMinator</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph 数据集 - SauravMaheshkar 收藏集</a>: 未找到描述</li><li><a href="https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20">Saurav Maheshkar ☕️ (@MaheshkarSaurav) 的推文</a>: 我目前正在研究 HyperGraph 表示学习，并在过去几天创建了一个 @huggingface 收藏集，包含：👉 处理后的数据集 👉 论文 👉 @Gradio space...</li><li><a href="https://saask.ing">SaaS King | 最佳 SaaS 模板</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=NUfAtIY85GU&t=8s&ab_channel=AdityaKumarSaroj">SaaS King 的 One Mix | 模板演示</a>: SaaS King 的 OneMix 快速介绍。OneMix 使用 Remix (Vite)、Tailwind、Supabase、Prisma、Stripe 和 Resend 构建。SaaS King 的 OneMix 如何帮助...</li><li><a href="https://www.youtube.com/watch?v=p77U2eyJFPU">制作了一个音乐老虎机，然后用它创作了一首歌 - captains chair 21</a>: 00:00 - 开始 01:35 - 构建音轨 08:28 - 音轨 我们的第一个 @HuggingFace space。这非常荒谬。https://huggingface.co/spaces/thepatch/the-slot-...</li><li><a href="https://huggingface.co/spaces/thepatch/the-slot-machine">The Slot Machine - thepatch 的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

grimsqueaker: 耶！谢谢！
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1224278505546711100)** (7 条消息): 

- **Batch Size 等效性查询**：一位成员询问，在训练不同规模的架构（如 ConvNeXt）时，使用 **batch size 32 且梯度累积为 2** 是否与 **batch size 64** 相当。

- **量子神经网络研究推广**：一位成员分享了他们正在进行的关于**量子神经网络模型**在传统图像数据集上性能的研究，并遇到了一个小问题。

- **特征提取与量子 SVM 咨询**：该成员详细阐述了他们的研究，提到他们使用 **Transformer 模型**提取了特征，并寻求在**量子 SVM (QSVM)** 中使用这些特征进行多类别分类的建议。

- **寻求量子核与超参数指导**：寻求关于为 QSVM 选择合适的**量子核**和**超参数**的建议，特别是在 **Qiskit 1.0.2** 环境下。

- **开放合作邀请**：另一位成员对 QSVM 研究表示了兴趣，随后发出了私信交流和潜在合作的公开邀请。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1224302630470287440)** (8 条消息🔥):

- **在 Diffusers 中触发 LoRA**：一位成员询问了**如何使用 diffusers 触发 LoRA**，对此回复提供了关于使用 [PEFT 进行推理](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference) 的指导，包括使用 *DiffusionPipeline* 加载和管理适配器（尤其是 LoRA）。

- **模型使用确认**：同一位成员随后提出了另一个关于如何确认**模型是否正在被使用**的问题，但在提供的消息中未收到直接回复。

- **寻求 PDF 方面的帮助**：一位社区成员请求帮助**在 PDF 文件上微调开源语言模型**，并表达了在这方面的挑战，但在提供的聊天记录中没有提供具体的建议。

- **询问 Mistral 的更新**：有人询问了关于 **Mistral** 的更新情况，但该查询后没有新的信息或回复。

- **技术讨论中的实时视频抖动**：一位社区成员分享了对实时视频中**抖动和漂移现象的观察**，询问这是否可能是由于舍入误差或过程中的 Bug 导致的，并希望能获得控制此问题以优化输出的见解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://spright-t2i.github.io/">SPRIGHT</a>：社交媒体描述标签</li><li><a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference">加载 LoRA 进行推理</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1224778620971515994)** (1 条消息): 

- **Gradio 4.25.0 发布，性能更流畅**：Gradio 的最新更新引入了 *自动删除 gr.State 变量* 功能，以增强高流量 Demo 的性能，并包含了一个用于浏览器标签页关闭时执行操作的 *unload 事件*。
- **延迟示例缓存（Lazy Example Caching）现已可用**：Gradio 4.25.0 添加了 **延迟示例缓存** 功能，通过 `cache_examples="lazy"` 实现，这尤其利好 **ZeroGPU** 用户，因为它在第一次请求时才缓存示例，而不是在服务器启动时。
- **修复流式音频 Bug**：更新修复了新版本 Gradio 中与 **流式音频输出** 相关的 Bug。
- **更直观的 gr.ChatInterface**：**gr.ChatInterface** 获得了升级，特别是支持直接从剪贴板粘贴图像。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1224585697344753674)** (8 条消息🔥): 

- **Mojo 中的 RL 集成挑战**：一位成员询问了在 Mojo 中运行 **强化学习（RL）Python 训练** 的挑战，特别是如何在 Mojo 中使用 PyTorch 环境。他们获悉了 Mojo 即将推出的 **MAX Engine** 和 **C/C++ interop**（互操作性），详见 [Mojo 路线图](https://docs.modular.com/mojo/roadmap#cc-interop)，这将允许重新实现 PyTorch 接口并加速 RL 环境的开发与执行。

- **Mojo 文档支持者**：针对关于文档的讨论，一位成员称赞了 Mojo 的**新文档**，称其非常全面。

- **Mojo 变量名中的数学符号**：有人提问 Mojo 是否支持类似 Julia 的数学名称。官方澄清 Mojo 目前仅支持变量名的 **ASCII 字符**，并遵循 Python 的变量命名约定。

- **Mojo 中的字符串处理疑问**：一位成员好奇为什么除法运算符 "/" 在 Mojo 中不是 "Stringable"，质疑是否所有字符串实体都应天生具备 Stringable 特性。

- **表情符号变量命名的替代方案**：另一位成员指出，在 Mojo 中，可以通过将符号（包括表情符号）包裹在**反引号**中来将其用作变量名，并提供了一个使用表情符号作为变量的示例。

**提到的链接**：<a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 路线图与注意事项 | Modular 文档</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1224461746820091905)** (10 条消息🔥): 

- **Modular 的推文连发**：Modular 分享了一系列推文，可能是活动或宣传的一部分。未提供推文的具体内容。
- **查看 Modular 的 Twitter**：获取更新和信息，请通过提供的 [链接](https://twitter.com/Modular) 关注 Modular 在其官方 Twitter 上发布的一系列推文。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1224783263936024597)** (1 条消息):

- **MAXimum Mojo 势头揭晓**：最新版本 MAX 24.2 已开放下载，为采用 Mojo 的 Python 开发者带来了一系列新特性。欲了解这些特性的更多深度内容，请参阅 [MAX 24.2 发布公告](https://www.modular.com/blog/max-24-2-is-here-whats-new)和 [Mojo 开源](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)博客中的专门文章。

**提及链接**：<a href="https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more">Modular: Mojo 24.2 更新内容：Mojo Nightly、增强的 Python 互操作性、开源标准库等</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 24.2 更新内容：Mojo Nightly、增强的 Python 互操作性、开源标准库等。

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1224259296297095188)** (47 条消息🔥): 

- **Mojo 中的并行性挑战**：Mojo 对非易并行（non-embarrassingly parallel）问题的处理仍处于开发阶段，一些人认为它类似于 [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782) 中概述的 Swift 并发模型。
- **值类型与标识 (Identity)**：一场讨论澄清了 Mojo 的值类型与 Python 不同，它本身不具有标识，这意味着 `is` 运算符可能具有不同的语义，侧重于值相等性而非对象标识。
- **Tensor 性能探究**：对比使用 Mojo 的 Tensor 结构体与直接使用 `DTypePointer` 的 Tensor 操作显示出显著的性能差异，这归因于低效的拷贝初始化，并通过改进实现得到了纠正 [详见 gist](https://gist.github.com/modularbot/88f71a13c2d3f546b9f4ee8a144ddd8e)。
- **顶级代码与 Escaping 运算符之谜**：有人对 Mojo 中顶级代码（top-level code）的实现以及似乎缺乏关于 "escaping" 运算符的文档提出了疑问，成员在官方文档中无法找到实质性信息。
- **SIMD Naïve Search 探索**：一位成员表示希望实现某[学术论文](https://arxiv.org/pdf/1612.01506.pdf)中概述的 SIMD Naïve Search，但在如何将 SSE2 内置函数（intrinsic functions）转换为 Mojo 结构方面面临不确定性。
<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/search?q=escaping+">Modular Docs</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/tensor/tensor#__mul__).">tensor | Modular Docs</a>：实现 Tensor 类型。</li><li><a href="https://gist.github.com/modularbot/88f71a13c2d3f546b9f4ee8a144ddd8e">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782">Swift Concurrency Manifesto</a>：Swift 并发宣言。GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1224463118445314068)** (2 条消息): 

- **Prism CLI 库重构成功**：模仿 Cobra 的 `Prism` CLI 库在 24.2 更新中进行了重大重构，带来了一系列新功能，如短格式标志（shorthand flag）支持和增强的命令结构（现在可以在结构体字段中管理父子关系）。该更新还确保命令可以使用自定义的位置参数验证函数；不过该库也自带了几个内置验证器。查看 [GitHub](https://github.com/thatstoasty/prism) 上的详细信息和示例。

- **缓解引用处理 (Reference Wrangle) 的困扰**：`Prism` 的作者表达了对 References 演进的强烈关注，称其为开发过程中的主要挑战。开发者们正热切期待未来更新中关于 References 更好的易用性。

**提及链接**：<a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: 模仿 Cobra 的 Mojo CLI 库。</a>：模仿 Cobra 的 Mojo CLI 库。通过在 GitHub 上创建账号来为 thatstoasty/prism 的开发做出贡献。

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1224291651669725284)** (4 条消息):

- **矩阵乘法之谜**：一位成员在运行 `matmul.mojo` 时遇到了与 `test_matrix_equal[matmul_vectorized](C, A, B)` 相关的错误；通过调整容差（tolerance）修复了该问题，这表明不同实现之间的结果一致性存在问题。
- **怀疑是舍入问题**：通过将 `matmul.mojo` 文件顶部的 `DType.float32` 更改为 `DType.float64`，该成员能够消除部分矩阵元素的错误，但并非全部，这表明错误可能与舍入有关。
  

---


**Modular (Mojo 🔥) ▷ #[⚡serving](https://discord.com/channels/1087530497313357884/1212827597323509870/1224265805278085180)** (7 messages): 

- **探索 Triton 之外的 MAX**：一位成员询问了 MAX 除了作为 Triton 后端之外的潜在优势。MAX Serving 被描述为 MAX Engine 的封装，可以使用本地 Docker 容器进行尝试，详情请参阅 [MAX Serving 文档](https://docs.modular.com/serving/get-started)。

- **寻求迁移说明**：同一位成员询问了如何将当前使用 Triton 推理服务器（包含两个模型：一个 tokenizer 和一个 ONNX/TensorRT 模型）的设置迁移到 MAX，并询问迁移是否像在配置中更新后端一样简单。

- **提供迁移至 MAX 的协助**：一位代表向考虑迁移到 MAX 的成员提供了帮助，表示渴望进一步了解其使用场景，并支持其流水线的性能升级。

- **迁移中的细节至关重要**：该代表询问了成员设置的具体细节，包括 tokenizer 模型是如何实现的，以及这两个模型是如何连接的，特别是是否使用了 Ensemble Models 或 Business Logic Scripting 功能。

- **无缝支持 ONNX，GPU 支持待定**：虽然确认了 ONNX 模型可以通过简单的配置更改后端来无缝工作，但该代表指出 MAX 目前不支持 GPU 托管模型，并表示该功能正在积极开发中。

**提到的链接**：<a href="https://docs.modular.com/serving/get-started">Get started with MAX Serving | Modular Docs</a>：展示如何在本地系统尝试 MAX Serving 的演练。

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1224435153498013726)** (11 messages🔥): 

- **发布新的 Mojo Nightly 版本**：Mojo nightly 版本已更新，您可以使用 `modular update nightly/mojo` 进行升级。有关 stable 版本与新 nightly 版本之间差异的详细 changelog 可以在 [GitHub](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 上找到。
- **检查 Mojo 版本间的差异**：为了对比 Mojo 版本之间的差异，提供了一个 diff 链接：[Comparing releases on GitHub](https://github.com/modularml/mojo/compare/4feb92e..1a8f912)。
- **Mojo Stdlib 的本地开发**：如果您想测试对 Mojo 中 stdlib 的本地修改，这里有关于如何开发和测试它的[文档](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies)，以及如何使用 `MODULAR_MOJO_NIGHTLY_IMPORT_PATH` 环境变量进行配置。
- **Mojo 测试最佳实践**：Mojo 中的新测试应优先使用 `testing` 模块中的方法而非 FileCheck，以遵循更好的实践。
- **贡献者协作频道**：建议使用当前频道进行 Mojo 贡献者之间的常规讨论，而具体问题应移至 GitHub 仓库。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/development.md#a-change-with-dependencies">mojo/stdlib/docs/development.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/compare/4feb92e..1a8f912">Comparing 4feb92e..1a8f912 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1224253236182122526)** (17 messages🔥):

- **Miniconda 作为轻量级替代方案**：一位成员询问 **Miniconda** 是否可以因其体积更小而成为 Anaconda 的有效替代品，这被证实是一个可行的选择。
- **OhanaPal 寻求合作者**：*OhanaPal* 是一款旨在帮助神经多样性人士处理日常任务和学习的应用，目前正在寻求社区在头脑风暴和原型设计方面的帮助，并提到他们目前使用 **OpenAI GPT APIs**。原型访问及更多信息可以在其[网站](https://www.ohanapal.app/)上找到。
- **社区参与号召**：参与者们收到了即将举行的 **April House Party** 的提醒，并受邀在一个新的指定频道中就 **Open Interpreter** 如何改善全人类的生存状况贡献想法。
- **对 Open Interpreter 移动端应用的期待**：Jordan Singer 在 Twitter 上发布的一条推文引发了关于 **Open Interpreter iPhone app** 潜力的讨论。成员们表达了极大的热情，该项目被鼓励在社区内进行开源开发。
- **React Native 应用的开端**：一位社区成员透露他们已经开始为 **Open Interpreter** 开发一个 **React Native** 应用，尽管尚未完全完成。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA">jordan singer (@jsngr) 的推文</a>：✨ 通过手机远程与你的电脑对话，我称之为 Teleport</li><li><a href="https://discord.gg/fjPmtRk8?event=1221828294811586572">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://www.ohanapal.app/">OhanaPal | 为超能者打造的超级应用</a>：欢迎来到 OhanaPal——赋能与包容在此交汇，为超能者让每一天都变得非凡。</li><li><a href="https://github.com/FiveTechSoft/tinyMedical">GitHub - FiveTechSoft/tinyMedical: 使用医疗数据集训练并保存为 GGUF 文件的 TinyLLama</a>：使用医疗数据集训练并保存为 GGUF 文件的 TinyLLama - FiveTechSoft/tinyMedical
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1224272736092950658)** (45 messages🔥): 

- **为完美适配而缩放**：在 3D 打印 **O1 Light** 时，建议将**模型缩放至 119.67%**，以便将 M5 Atom Echo 妥善放入其插槽。一位用户澄清说，尺寸问题是因为在安装前必须将 Atom 从外壳中取出。
- **M5Atom 上的无缝重连**：GitHub 上一个新的 **pull request** ([#214](https://github.com/OpenInterpreter/01/pull/214)) 解决了 M5Atom 的更新问题，使其在重启后能自动重新连接到上次成功的 WiFi 和 Server URL。
- **移动端控制的创意征集**：OpenInterpreter 团队暗示了未来实现手机控制的可能性，并分享了一个 GitHub 仓库 ([open-interpreter-termux](https://github.com/MikeBirdTech/open-interpreter-termux))，用于通过 **Termux** 在 **Android** 上运行 **OI**。
- **翻新 Windows 安装文档**：分享了在 **Windows** 上安装 **OpenInterpreter** 的更新说明，详细列出了必要的工具，如 **Git**、**virtualenv/MiniConda**、**Chocolatey** 和 **Microsoft C++ Build Tools**。对于 **Linux** 用户，记录了关于安装过程中各种错误的讨论，并呼吁分享这些错误以便协作排查。
- **替代 Windows 包管理器**：除了传统的 **Windows** 包管理工具外，建议用户考虑将 **winget** ([Microsoft 官方包管理器](https://learn.microsoft.com/en-us/windows/package-manager/winget/)) 和 **scoop** 作为 **Windows** 包管理需求的替代方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://01.openinterpreter.com/services/language-model">Language Model - 01</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/package-manager/winget/">使用 winget 工具安装和管理应用程序</a>: winget 命令行工具使开发人员能够在 Windows 计算机上发现、安装、升级、移除和配置应用程序。</li><li><a href="https://scoop.sh/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/01/pull/214/">自动重新连接到上次成功的 WiFi 和 Server URL（如果可用）由 aramsdale 提交 · Pull Request #214 · OpenInterpreter/01</a>: 自动重新连接到上次成功的 WiFi 和 Server URL（如果可用）。利用 Preferences 检测成功的 WiFi 连接，存储到 ssid 偏好设置，并在重启时调用。Server 同样...</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-termux">GitHub - MikeBirdTech/open-interpreter-termux: 在 Android 设备上安装 Open Interpreter 的说明。</a>: 在 Android 设备上安装 Open Interpreter 的说明。 - MikeBirdTech/open-interpreter-termux</li><li><a href="https://git-scm.com/download/win">Git - 下载安装包</a>: 未找到描述</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1224257328644100116)** (7 messages): 

- **探索 Open Interpreter**: 一位成员分享了他们的 YouTube 视频，题为 "Open Interpreter Advanced Experimentation - Part 2"，其中可能包含关于 **Open Interpreter** 的新实验。视频可在 [YouTube](https://www.youtube.com/watch?v=v9uXdRwAQ0c) 观看。

- **Fabric，AI 增强框架**: 介绍了一个名为 **fabric** 的 GitHub 仓库；这是一个用于通过 AI 增强人类能力的开源框架。它利用众包的 AI prompts 集来解决特定问题，访问地址为 [GitHub - danielmiessler/fabric](https://github.com/danielmiessler/fabric)。

- **微软用于 Windows OS 交互的 UFO**: 一位成员发现了 **Microsoft's UFO**，这是一个 GitHub 项目，被描述为用于 Windows OS 交互的以 UI 为中心的 Agent。有人提问这是否是微软在 Windows 上实现 **Open Interpreter (OI)** 的试验场，仓库地址为 [GitHub - microsoft/UFO](https://github.com/microsoft/UFO)。

- **YouTube 上的 Transformers 视觉介绍**: 分享了一个名为 "But what is a GPT? Visual intro to Transformers | Deep learning, chapter 5" 的视频，提供了对 Transformers 的介绍，这是 **LLMs (Large Language Models)** 背后的技术。视频可在 [YouTube](https://www.youtube.com/watch?v=wjZofJX0v4M) 观看。

- **社区对 GPT 教育内容的兴奋**: 成员们对关于 Transformers 和 **GPTs** 的教育内容表示兴奋。他们通过 "已书签！" 和 "太棒了 🚀" 等评论分享了他们的期待和认可。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=v9uXdRwAQ0c">Open Interpreter Advanced Experimentation - Part 2</a>: ➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren➤ Discord  - https://discord.com/invite/z5VVSGssCw➤ TikTok - https://www....</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: Transformers 及其先决条件的介绍。为赞助者提供的下一章预览：https://3b1b.co/early-attention 特别感谢这些支持...</li><li><a href="https://github.com/microsoft/UFO">GitHub - microsoft/UFO: 一个用于 Windows OS 交互的以 UI 为中心的 Agent。</a>: 一个用于 Windows OS 交互的以 UI 为中心的 Agent。通过在 GitHub 上创建账户为 microsoft/UFO 的开发做出贡献。</li><li><a href="https://github.com/danielmiessler/fabric">GitHub - danielmiessler/fabric: fabric 是一个通过 AI 增强人类能力的开源框架。它提供了一个模块化框架，使用众包的 AI prompts 集来解决特定问题，可随处使用。</a>: fabric 是一个通过 AI 增强人类能力的开源框架。它提供了一个模块化框架，使用众包的 AI prompts 集来解决特定问题，可随处使用。 - ...
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1224327044431614022)** (66 messages🔥🔥):

- **聊天机器人回复中的机器人名称前缀**：一名用户在使用 OpenRouter 进行角色扮演聊天（使用 `messages` 键）时，发现 **undi95/remm-slerp-l2-13b:extended model** 的回复以 `{bot_name}:` 开头，并询问这是由于 Prompt 错误还是需要进行文本替换。相关讨论澄清了最近的 Prompt 模板更新不应导致此问题，并进一步探讨了是否使用了 `name` 字段。

- **连接 OpenRouter 报错**：一名用户报告在尝试连接 OpenRouter 时出现 **SSL 错误**（*EOF occurred in violation of protocol*），但聊天中未直接给出解决方案。
  
- **《Patterns of Application Development Using AI》发布公告**：**Obie Fernandez** 宣布了他的新书《Patterns of AI-Driven Application Architecture》的早期版本发布，书中重点介绍了 OpenRouter 的使用。

- **模型性能与可用性咨询**：用户讨论了各种模型的性能以及 **nitro 和 non-nitro 模型** 的可用性，其中一名用户在 nitro 模型不可用后寻求目前最快的选项。会议确认 **nitro 模型仍然可用，并且更多模型正在路上**。

- **常规故障排除与模型建议**：用户分享了模型故障的经历（如 **NOUS-HERMES-2-MIXTRAL-8X7B-DPO**），并针对角色扮演等特定任务提供了替代模型建议，包括配备 30k 上下文窗口的 **Nous Capybara 34B**。针对 **OpenRouter logit bias** 在某些模型上不起作用的问题，解释称该功能仅在 OpenAI 的模型上受支持。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/abhishek/autotrain-mixtral-dgx-cloud-local">使用 AutoTrain 微调 Mixtral 8x7B</a>：未找到描述</li><li><a href="https://leanpub.com/patterns-of-application-development-using-ai">Patterns of Application Development Using AI</a>：探索构建智能、自适应且以用户为中心的软件系统的实用模式和原则，充分发挥 AI 的力量。
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1224268340306903050)** (41 messages🔥): 

- **基准测试与线程利用率揭秘**：一位成员对相比 **NumPy** 取得的显著改进表示惊讶，原以为 NumPy 已经经过深度优化，并请求查看基准测试代码。分享的代码同时使用了 NumPy 和自定义的 `matmul` 函数来展示性能差异，揭示了 **NumPy** 并不使用线程。

- **对 AI 新更新的热切期待**：讨论围绕 **llamafile 0.7** 的发布以及尝试将其与 **openchat 3.5** 结合使用展开。成员们寻求关于 Prompt 模板化以及在 UI 中使用变量的澄清，并指出由于缺乏文档而导致的困惑。

- **TinyBLAS 对比专有库**：在讨论 **llamafile** 在 CPU 与 GPU 上的性能时，提到可以使用 **`--tinyblas`** 标志在不安装 CUDA 或 ROCm SDK 的情况下获得 GPU 支持，但性能可能会因显卡而异。

- **Windows ARM64 兼容性查询**：关于 **Windows ARM64** 与 **llamafile** 兼容性的讨论引发了对支持情况和二进制格式的疑问，揭示了 Windows on ARM 支持带有 ARM64X 二进制文件的 PE 格式，但在 **AVX/AVX2** 模拟方面存在问题。

- **本地部署中的练习与故障排除**：用户在尝试**本地运行 llamafile** 时遇到了 **"exec format error"**，建议使用 bash 而非 zsh，并为在特定硬件配置上运行 **Mixtral** 模型提供了说明。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://json-schema.org/learn/getting-started-step-by-step">JSON Schema - 创建你的第一个 schema</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main">jartine/Mixtral-8x7B-Instruct-v0.1-llamafile at main</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/bagel-8x7b-v0.2-GGUF">TheBloke/bagel-8x7b-v0.2-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF">TheBloke/dolphin-2.7-mixtral-8x7b-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Septillihedron/SuperheroesPlusSchema/releases">Releases · Septillihedron/SuperheroesPlusSchema</a>：SkillsLibrary 插件的文档。通过在 GitHub 上创建账户来为 Septillihedron/SuperheroesPlusSchema 的开发做出贡献。</li><li><a href="https://learn.microsoft.com/en-us/windows/arm/arm64x-pe">Arm64X PE 文件</a>：Arm64X 是 Windows 11 SDK 中的一种 PE 文件类型，用于 Arm64 上的 x64 兼容性。Arm64X 对于中间件或插件开发者来说可能是一个很好的解决方案，因为代码可能会被加载到 x64 或 A...</li><li><a href="http://www.emulators.com/docs/abc_arm64ec_explained.htm">ARM64 Boot Camp: ARM64EC and ARM64X 详解</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1224562518127808553)** (6 messages): 

- **MDEL 携手 AMD 稳步前行**：MDEL 已成功使用 **AMD GPU** 训练了一个 **15B 模型**，这标志着在大规模模型硬件利用方面的一个潜在有趣进展。

- **Mistral 敞开大门**：Mistral 团队邀请社区成员参加 **office hour 环节**进行提问，信号显示了一个开放的对话和支持渠道。 

- **对新版本的怀疑**：一位成员开玩笑地询问 **v0.2 版本**是否是**愚人节玩笑**，反映了社区对该更新的惊讶或怀疑。

- **数据集统一的挑战**：一位贡献者正致力于将大约 **15 个不同的数据集**统一为 TSV 和 pickle 格式的索引文件，面临着翻译错位和海量数据等挑战。他们正在考虑创建一个不带权重的单一、巨大的语言对 JSON。

- **寻求 Runpod 使用经验**：一位用户询问了使用 **runpod serverless** 运行超大规模语言模型 (**VLLM**) 的经验，表明对该服务的社区知识感兴趣。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1224259266928574474)** (18 messages🔥): 

- **Pull Request 准予合并**：针对 `lisa` 的最新 PR 已获准合并，表明所做的更改被认为是可靠且有益的。
- **添加 YAML 示例并启动运行**：代码库中添加了一个示例 YAML 文件，并随后启动了一个测试运行以监控新配置的性能。
- **Axolotl 的恶作剧提议**：有一个开玩笑的提议，建议发布一个愚人节公告，声称 Axolotl 已与 OpenAI 合作，幽默地暗示 Axolotl 现在可以微调 GPT-4 及未来的模型。
- **DeepSpeed 显存溢出 (OOM) 问题**：开发者在使用 DeepSpeed 或 FairScale Single-Process Single-GPU (FSDP) 尝试训练模型时遇到了显存溢出错误，特别是在涉及 Lisa 更改时。
- **文档更新查询**：一位成员观察到 Axolotl 文档的更新，但指出目录似乎丢失了。另一位成员承认了该问题并计划很快解决。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/">Axolotl</a>：未找到描述</li><li><a href="https://github.com/OptimalScale/LMFlow/issues/726#issuecomment-2029701152">[BUG] LISA: same loss regardless of lisa_activated_layers · Issue #726 · OptimalScale/LMFlow</a>：描述该 bug，我认为目前的 LISA 实现可能存在问题。无论激活多少层，训练损失都没有区别。没有使用 LMFlow 而是使用了 HF ...</li><li><a href="https://github.com/kyegomez/BitNet/tree/main">GitHub - kyegomez/BitNet: "BitNet: Scaling 1-bit Transformers for Large Language Models" 的 pytorch 实现 - kyegomez/BitNet</a>：BitNet: Scaling 1-bit Transformers for Large Language Models 的 pytorch 实现 - kyegomez/BitNet
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1224331766807138334)** (16 messages🔥):

- **训练神秘挂起**：一位成员遇到了训练问题，进程在第一个 epoch 后卡住。他们确认该问题与评估无关，因为已设置 `val_set_size: 0`，这意味着并未执行评估。
- **寻找训练冻结的元凶**：通过讨论，有人建议该问题可能与存储不足或潜在的有 bug 的功能（如 `eval_table`）有关，该功能以在评估期间生成预测并上传到 `wandb` 而闻名。
- **Eval Table：是否有 Bug？**：一位成员澄清说，由于他们在训练期间不进行评估，因此没有启用 `eval_table`。有提示指出该功能可能存在 bug。
- **配置一致性不保证性能**：同一位成员提到，尽管使用了相同的配置（仅更改了模型名称），但在某些模型上遇到了训练问题，而在其他模型上则没有，这增加了根因的神秘感。
- **推理 Prompt 实践讨论**：在另一个线程中，建议在模型推理期间使用与训练时相同的 prompt 格式，特别是在用户输入中的指令背景下，以及微调后对 few-shot prompting 的影响。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1224327676286865451)** (29 条消息🔥): 

- **FastLLM 带着宏大声明发布**：Qdrant 宣布了他们的新语言模型 **FastLLM (FLLM)**，专为 Retrieval Augmented Generation 设计，具有惊人的 10 亿 token 上下文窗口。AI 社区指出这可能是有效的愚人节玩笑，因为它是在 4 月 1 日宣布的。
  
- **Transformer 的新教学精品**：3Blue1Brown 制作的一段名为 "But what is a GPT? Visual intro to Transformers | Deep learning, chapter 5" 的视频因提供了 Transformer 和 GPT 的视觉化介绍而受到关注。

- **LLM Answer Engine GitHub 项目揭晓**：GitHub 上一个名为 "llm-answer-engine" 的开源项目引起了兴趣，该项目使用包括 Next.js、Groq, Mixtral, Langchain 和 OpenAI 在内的强大技术栈构建了一个受 Perplexity 启发的回答引擎。

- **用于结构化 LLM 输出的 Instructor 抽象**：instructor 1.0.0 发布，这是一个确保 LLM 的结构化输出与用户定义的 Pydantic 模型保持一致的工具，简化了与其他系统模块的交互和集成。
  
- **Google 凭借新领导层加速 AI 发展**：Logan Kilpatrick 宣布加入 Google，负责 **AI Studio** 的产品领导并支持 **Gemini API**，这表明 Google 正致力于成为 AI 开发者的首选之地。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/officiallogank/status/1775222819439149424?s">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，负责 AI Studio 的产品领导工作并支持 Gemini API。前方有很多艰苦的工作，但我们将使 Google 成为 AI 开发者构建应用的最佳家园。...</li><li><a href="https://qdrant.tech/blog/fastllm-announcement/">介绍 FastLLM：Qdrant 革命性的 LLM - Qdrant</a>：轻量级且开源。专为 RAG 定制，并与 Qdrant 完全集成。</li><li><a href="https://9to5mac.com/2024/04/01/apple-ai-gpt-4/">Apple AI 研究人员夸赞其实用的端侧模型“显著优于” GPT-4 - 9to5Mac</a>：Siri 最近一直在尝试描述在 CarPlay 或通知播报功能中使用 Messages 接收到的图像。在...</li><li><a href="https://jack-clark.net/2024/03/28/what-does-1025-versus-1026-mean/">10^25 与 10^26 意味着什么？</a>：简要探讨基于 FLOPs 的监管意味着什么。最近的 AI 法规根据用于训练的浮点运算量定义了监管触发点……</li><li><a href="https://x.com/jyangballin/status/1775114444370051582?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 John Yang (@jyangballin) 的推文</a>：SWE-agent 是我们用于自主解决 GitHub 仓库问题的新系统。它在 SWE-bench 上的准确率与 Devin 相当，平均耗时 93 秒，且已开源！我们设计了一个新的 Agent-co...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">来自 Gustavo Cid (@_cgustavo) 的推文</a>：我曾经恳求 LLM 提供结构化输出。大多数时候，它们能理解任务并返回有效的 JSON。然而，在大约 5% 的时间里，它们做不到，我不得不编写胶水代码来避免...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya">来自 Gustavo Cid (@_cgustavo) 的推文</a>：我曾经恳求 LLM 提供结构化输出。大多数时候，它们能理解任务并返回有效的 JSON。然而，在大约 5% 的时间里，它们做不到，我不得不编写胶水代码来避免...</li><li><a href="https://x.com/sucralose__/status/1774782583731020200?s=46&t=JE84TqLviekDnEt8MAT-Eg">来自 Will Anthonio Zeppeli (@sucralose__) 的推文</a>：我对 ChatGPT 的后端进行了更多调查，发现了名为 "GPT Alpha" 模型的可靠证据，我认为它是 GPT-4 的继任者。可以提前启用它，但它...</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，负责 AI Studio 的产品领导工作并支持 Gemini API。前方有很多艰苦的工作，但我们将使 Google 成为 AI 开发者构建应用的最佳家园。...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s">但什么是 GPT？Transformer 的视觉入门 | 深度学习，第 5 章</a>：Transformer 及其先决条件的介绍。赞助者可提前观看下一章：https://3b1b.co/early-attention 特别感谢这些支持...</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: 使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper 构建一个受 Perplexity 启发的问答引擎</a>：使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper 构建一个受 Perplexity 启发的问答引擎 - developersdigest/llm-answer-engine
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1224477813810270208)** (4 条消息): 

- **"But what is a GPT?" 视频分享**：频道中重点推荐了一个提供 Transformer 和 GPT 视觉入门的 [YouTube 视频](https://www.youtube.com/watch?v=wjZofJX0v4M)。该视频面向机器学习初学者，为深入了解该主题提供了一个易于理解的阶梯。
- **自制 GPU 达成新里程碑**：一位成员讨论了他们的[个人项目](https://www.furygpu.com/blog/hello)，即在游戏行业工作多年后，开发一个能够运行 Quake 的定制全栈 GPU。这个 FPGA 设计代表了长期致力于渲染硬件方面的成果。
- **Justine 的 CPU Matmul 优化见解**：分享了指向 [Justine Tunney 博客](https://justine.lol/matmul/) 的链接，该博客描述了 CPU 矩阵乘法（Matmul）优化及其与 GPU 优化的相似之处，尽管没有使用某些 GPU 特有的技术，如 warptiling 或显式缓存。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.furygpu.com/blog/hello">Hello! &mdash; FuryGpu</a>: 在业余时间开发定制全栈 GPU 近四年后，我觉得是时候开始整理一些材料来展示这个项目、其技术细节以及...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT?  Visual intro to Transformers | Deep learning, chapter 5</a>: Transformer 及其先决条件的介绍。赞助者可提前观看下一章：https://3b1b.co/early-attention 特别感谢这些支持...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1224381448191086662)** (6 messages): 

- **使用 Nsight Compute 进行 Triton 代码分析**: 一位成员分享了他们看到的使用 Nsight Compute 对 Triton 代码进行 Profiling 的案例，这允许同时查看 PTX 和 Python 代码，正如 [Accelerating Triton](https://pytorch.org/blog/accelerating-triton/) 的 "Vectorized Load" 部分所示。他们询问如何为自己的工作设置这一极具洞察力的功能。
- **Triton Kernels 基准测试**: 讨论中提到的博文提供了一个加速 Triton Kernel 的详尽模板，提到对于典型的 Llama 风格推理输入，性能从 275us 显著提升至 47us。
- **Nsight Compute Trace 下载与远程启动的对比**: 有人询问使用 Nsight Compute 的最佳实践，质疑是生成 trace 文件并下载，还是使用远程启动（Remote Launch），并暗示设置远程启动可能比较繁琐。
- **使用 Nsight 分析 Triton 的命令**: 分享了一个用于使用 Nsight Compute 生成详细 Profile 的有用命令：`ncu --target-processes all --set detailed --import-source yes -o output_file python your_script.py`。这允许对 Triton 代码进行分析和后续分析。
- **成功设置分析的确认**: 一位成员确认成功设置了 Triton 代码分析并表示感谢，指出了所获得的分析信息的实用性和价值。

**相关链接**: <a href="https://pytorch.org/blog/accelerating-triton/">Accelerating Triton Dequantization Kernels for GPTQ</a>: TL;DR  

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1224411486214951132)** (3 messages): 

- **在 Ubuntu 上安装 Nsight DL Design 的麻烦**: 一位成员详细说明了他们在 Ubuntu 上通过运行 `.run` 文件安装 **Nsight DL Design** 的尝试，确认了已使用 `chmod +x` 赋予执行权限并使用了 `sudo`，但安装后无法找到 DL Design 应用程序。他们寻求关于安装后如何打开 **DL Design app** 的建议。
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1224392530720850010)** (2 messages): 

- **PyTorch 团队正在准备回应**: PyTorch 代表对比较 JAX、TensorFlow 和 PyTorch 的基准测试问题表示担忧，称 Keras 团队在处理这些问题时并未咨询他们。目前正在制定回应。
- **基准测试对决**: 分享了 Jeff Dean 的一条推文，强调了给定链接中的关键基准测试，显示 **JAX** 在大多数测试中在 GPU 上是最快的，TensorFlow 表现也很强劲，而 **PyTorch** 在速度上落后。推文引用了最近的基准测试表，可在[此处](https://x.com/JeffDean/status/1774274156944859455)查看。
<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://x.com/JeffDean/status/17742">derek dukes (@ddukes) 的推文</a>: 海滩篝火，准备看日落。你能赢过加州吗？</li><li><a href="https://x.com/JeffDean/status/1774274156944859455">Jeff Dean (@🏡) (@JeffDean) 的推文</a>: 这是链接中的关键基准测试表。GPU 上的 JAX 后端在 12 个基准测试中有 7 个是最快的，TensorFlow 后端在另外 5 个测试中是最快的。PyTorch 后端则没有...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

c_cholesky: 谢谢 😊
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1224755324100149390)** (1 messages): 

- **Opus 增强的潜力**: 讨论暗示，在准确的 *Opus 判断*假设下，**Opus** 可能会从进一步的 **RLAIF**（Reinforcement Learning with Augmented Intermediate Features）应用中获得额外收益。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1224783900036042772)** (2 messages):

- **Google AI 船队的新统帅**：一位人士宣布加入 **Google**，负责 AI Studio 的产品领导工作并支持 Gemini API。他表达了将 Google 打造为 AI 开发者首选目的地的决心，并表示：“我绝不会满足于现状。”
- **意想不到的变动**：这一举动让频道成员感到惊讶，引发了诸如“完全没料到这一招”之类的反应。

**提到的链接**：<a href="https://fxtwitter.com/OfficialLoganK/status/1775222819439149424">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，领导 AI Studio 的产品工作并支持 Gemini API。前方有很多艰苦的工作，但我们将使 Google 成为 AI 开发者构建应用的最佳家园。...

  

---


**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1224375780545396889)** (5 条消息): 

- **关于 DPO 和冗长性（Verbosity）的新预印本**：分享了一个[新预印本](https://x.com/rm_rafailov/status/1774653027712139657?s=46)的链接，强调了 Direct Preference Optimization (DPO) 与冗长性之间的相互作用，初步反馈指出了在大规模训练时面临的问题。
- **人类反馈强化学习研究**：一篇 [arXiv 预印本](https://arxiv.org/abs/2403.19159)讨论了在 Reinforcement Learning from Human Feedback (RLHF) 中对人类偏好偏差（特别是冗长性）的利用，并在 Direct Preference Optimization (DPO) 的背景下探索了这一研究不足的领域。
- **对 Rafael 工作的认可**：一位成员提到他们“大概应该读读”这篇预印本，并承认了 Rafael 在该领域的专业知识，指出他们之间有很深入的交流。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19159">Disentangling Length from Quality in Direct Preference Optimization</a>：Reinforcement Learning from Human Feedback (RLHF) 一直是近期 Large Language Models 取得成功的关键组件。然而，已知 RLHF 会利用人类偏好中的偏差，例如冗长性...</li><li><a href="https://x.com/rm_rafailov/status/1774653027712139657?s=46">来自 Rafael Rafailov (@rm_rafailov) 的推文</a>：关于 DPO 和冗长性之间相互作用的新预印本已发布。我们收到的关于 DPO 的第一批反馈之一是，在大规模训练时，模型会变得越来越冗长，直到发散。冗长性...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1224755945343811624)** (1 条消息): 

- **GPT-4 之后转向保密**：讨论强调了在 GPT-4 技术报告发布后，各公司开始转向保密，该报告显著省略了模型细节，标志着从开放数据共享向保守科学的转变。
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1224364644559360194)** (8 条消息🔥): 

- **Jamba 的速度之谜**：一位成员质疑 **Jamba** 如何随着 token 数量的增加而变得更快，特别是在序列化的解码（decoding）步骤中。引用的图表显示了*每个 token* 的端到端吞吐效率，从而引发了困惑。
- **解码速度的误解**：讨论随后澄清了图中描绘的加速在解码阶段也存在，而不仅仅是编码（encoding）阶段。尽管解码是一个序列化过程，但即使上下文规模增长，吞吐量也会增加。