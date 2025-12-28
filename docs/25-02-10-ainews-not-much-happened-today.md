---
companies:
- google
- zyphraai
- hugging-face
- anthropic
- deepseek
- openai
date: '2025-02-11T03:56:45.222082Z'
description: '**谷歌（Google）**发布了 **Gemini 2.0 Flash Thinking Experimental 1-21**，这是一款视觉语言推理模型，拥有
  **100 万 token 的上下文窗口**，并在科学、数学和多媒体基准测试中提升了准确率，表现超越了 **DeepSeek-R1**，但仍次于 **OpenAI
  的 o1**。


  **ZyphraAI** 推出了 **Zonos**，这是一款多语言**文本转语音（TTS）模型**，支持**即时语音克隆**，并可调节语速、音调和情感，在 RTX
  4090 显卡上的运行速度约为**实时速度的 2 倍**。


  **Hugging Face** 发布了 **OpenR1-Math-220k**，这是一个大规模**数学推理数据集**，包含 **22 万个问题**和 **80
  万条推理轨迹**，由 512 块 H100 GPU 生成。


  **Tom Goldstein** 推出了 **Huginn-3.5B**，这是一个开源的隐性推理模型（latent reasoning model），基于 **8000
  亿 token** 训练，在 **GSM8K** 等推理任务上的表现优于更大规模的模型。


  **Jeremy Howard** 和 **iScienceLuvr** 的讨论强调了隐性潜推理（implicit latent reasoning）的进展，并就人类可读推理轨迹的未来展开了辩论。


  **Anthropic** 推出了 **Anthropic 经济指数（Anthropic Economic Index）**，旨在通过数百万次 **Claude**
  对话来分析人工智能对经济的影响。'
id: d4a15eb1-67d6-49c2-a284-29d362a987ca
models:
- gemini-2.0-flash-thinking-experimental-1-21
- zonos
- openr1-math-220k
- huginn-3.5b
- deepseek-r1
- o1
- claude
original_slug: ainews-not-much-happened-today-3076
people:
- jeremyphoward
- andrej-karpathy
- tom-goldstein
- reach_vb
- iscienceluvr
title: 今天没发生什么特别的事。
topics:
- vision
- multilingual-models
- text-to-speech
- voice-cloning
- math
- reasoning
- latent-reasoning
- chain-of-thought
- dataset-release
- fine-tuning
- model-training
- model-performance
- context-windows
- benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**一个平静的日子。**

> 2025年2月7日至2月10日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**210** 个频道，**11464** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1218 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

就像之前的 [Meta's Coconut](https://arxiv.org/abs/2412.06769) 一样，[Huginn's Latent Reasoning Model](https://x.com/iScienceLuvr/status/1888792081382137966) 今天引起了轰动。我们同意 [Jeremy](https://x.com/jeremyphoward/status/1888815600958656793) 和 [Andrej](https://x.com/karpathy/status/1835561952258723930) 的观点，即最好的 RL 可能不会以英文形式存在，但我们没有将其选为专题报道，因为据推测 DeepSeek 在开发 r1 时已经尝试过这一点（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-deepseek-r1-o1-level-open-weights-model/)），并且认为不值得为了无法阅读思考过程而进行这种权衡。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**AI 模型发布与进展**

- **Google 发布 Gemini 2.0 Flash Thinking Experimental 1-21**：[DeepLearningAI](https://twitter.com/DeepLearningAI/status/1889026549275344986) 宣布 Google 发布了 Gemini 2.0 Flash Thinking Experimental 1-21，这是其视觉-语言推理模型的最新版本，具有扩展的 **100 万 token 上下文窗口**和**用户可读的 Chain of Thought**。该更新提高了在科学、数学和多媒体基准测试中的准确性，超越了 **DeepSeek-R1**，但在某些领域仍落后于 **OpenAI's o1**。

- **Zonos 发布 - 具备语音克隆功能的多语言 TTS 模型**：[@reach_vb](https://twitter.com/reach_vb/status/1889015111890997479) 强调 [ZyphraAI](https://twitter.com/ZyphraAI) 发布了 **Zonos**，这是一个采用 **Apache 2.0 许可**的多语言 **Text-to-Speech** 模型，具有**即时语音克隆**功能。该模型支持使用 10-30 秒的说话者样本进行 **Zero-shot TTS 语音克隆**、用于增强说话者匹配的**音频前缀输入**，以及对**语速、音调、频率、音频质量和情感**的控制。它在 **RTX 4090** 上能以 **~2 倍实时速度**运行，并已在 [Hugging Face Hub](https://t.co/sZndYJ5caM) 上线。

- **Hugging Face 发布 OpenR1-Math-220k 数据集**：[@_lewtun](https://twitter.com/_lewtun/status/1889002019316506684) 和 [@reach_vb](https://twitter.com/reach_vb/status/1888994979218915664) 宣布发布 **OpenR1-Math-220k**，这是一个基于 **Numina Math 1.5** 的大规模**数学推理数据集**，包含 **22 万个数学问题**和在 **512 张 H100 GPU** 上生成的 **80 万条原始 R1 推理轨迹**。该数据集采用 **Apache 2.0 许可**，鼓励社区**微调模型**并提升数学推理能力。

**AI 推理与模型的进步**

- **Huginn-3.5B Latent Reasoning Model 介绍**：[Tom Goldstein](https://twitter.com/tomgoldsteincs/status/1888980680790393085) 介绍了 **Huginn-3.5B**，这是一个开源推理模型，它在**潜空间 (Latent Space) 中进行隐式推理**，在测试时不产生额外的 Chain of Thought token。Huginn-3.5B 在 **800B token** 上进行了训练，在 **GSM8K** 等推理任务上表现出显著改进，尽管体积较小，但性能优于更大的模型。

- **关于人类可读推理轨迹的辩论**：[Jeremy Howard](https://twitter.com/jeremyphoward/status/1888815600958656793) 预测，训练 AI 系统产生**人类可读的推理轨迹**最终会显得很奇怪，他将其比作要求 Diffusion 图像模型输出与艺术家的笔触相匹配的图像序列。他认为未来的模型可能会以人类不易解释的方式内化推理过程。

- **通过 Latent Reasoning 扩展 Test-Time Compute**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1888792081382137966) 讨论了一种新的语言模型架构，能够通过在**潜空间中进行隐式推理**来提高推理基准测试的性能。该模型在不需要专门训练数据的情况下扩展了 Test-Time Compute，支持小上下文窗口，并捕捉到了难以用文字表达的推理过程。

**AI 对行业和经济的影响**

- **Anthropic 发布 Anthropic Economic Index**：[AnthropicAI](https://twitter.com/AnthropicAI/status/1888954156422992108) 发布了 **Anthropic Economic Index**，旨在了解 AI 随时间推移对经济的影响。他们的第一篇论文分析了数百万条匿名的 **Claude conversations**，揭示了 AI 在不同任务和职业中的使用情况。主要发现包括：

  - **AI 的使用倾向于增强 (57%) 而非自动化 (43%)**。
  - **软件和技术写作任务**的 AI 使用率最高。
  - AI 的采用在**中高收入工作**中最为普遍，而在极高和低收入工作中的使用率较低。
  - 该数据集和持续分析旨在跟踪 AI 演进过程中的变化模式。

- **DeepSeek 模型集成至云服务**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1888812805932875828) 指出，中国三大电信运营商正竞相将 **DeepSeek models 集成到云服务中**，这可能会**冻结他们自己的 LLM 项目**。这表明战略重心已转向采用现有的强大模型，而非独立开发新模型。

**AI 工具、开发与研究**

- **结合向量搜索与知识图谱**：[Qdrant Engine](https://twitter.com/qdrant_engine/status/1888860549775065437) 分享了关于结合 **Neo4j 和 Qdrant** 构建更智能的 **GraphRAG** 的见解，该方案利用 **vector search 进行语义检索**，并利用 **graph traversal 进行结构化推理**。这种方法旨在以更少的 **LLM** 依赖实现更高的准确性。

- **使用 TensorFlow 的 ImageDataGenerator**：[DeepLearningAI](https://twitter.com/DeepLearningAI/status/1888967700476555324) 强调了使用 **TensorFlow 的 ImageDataGenerator** 来处理大小、位置各异且包含多个主体的真实世界图像。该工具可自动对图像进行**标注、调整大小和分批处理**以进行训练，从而提高处理多样化图像数据集时数据流水线的效率。

- **探索 AI 在“未知的未知”中的局限性**：[@hardmaru](https://twitter.com/hardmaru/status/1888958032039813469) 讨论了一篇题为“**Evolution and The Knightian Blindspot of Machine Learning**”的论文，该论文认为进化过程使生物体能够应对**意外事件**（“未知的未知”），而目前的 AI 系统很难复制这种能力。

**社区见解与活动**

- **Sam Altman 的三个观察**：[Sam Altman](https://twitter.com/sama/status/1888695926484611375) 分享了[“三个观察”](https://t.co/Ctvga5vfMy)，提供了可能与 AI 发展、行业趋势或人类潜力相关的见解。内容强调了技术持续的演进和影响。

- **巴黎 AI 峰会与开源倡导**：[Clement Delangue](https://twitter.com/ClementDelangue/status/1888920800528331091) 宣布抵达巴黎参加 **AI Summit**，并强调与 [Irene Solaiman](https://twitter.com/IreneSolaiman) 等团队成员一起**推动开源 AI**。重点是**加大在法国的投资**，重点关注开源、机器人和应用。

- **关于中国 AI 进展的讨论**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1888781956315316378) 提供了一个反映对中国 AI 进展持怀疑态度的简史，指出从最初的低估到承认其扎实的工程努力的转变过程。

**梗/幽默**

- **OpenAI 的超级碗广告与 Google 的竞争**：[Sam Altman](https://twitter.com/sama/status/1888703820596977684) 幽默地评论了超越 Google 的挑战：“伙计，要赶上 Google 还有很长的路要走 🥺”，并在与 [@xprunie](https://twitter.com/sama/status/1888702632509735199) 的对话中提到“还有我们的广告，真的很棒”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1888879403867787485) 戏谑地批评了 OpenAI 员工炒作他们高制作价值的广告，将 OpenAI 比作 Apple 类型的公司。

- **Hackbot 奇点与 TEDx 演讲**：[@rez0__](https://twitter.com/rez0__/status/1888801773558665464) 提到 **“hackbot singularity 即将到来”**，并分享了他的 TEDx 演讲“**The Rise of AI Hackbots**”（可在 [YouTube](https://t.co/PHKxESqnmr) 上观看），讨论了 AI 在网络安全和黑客攻击中的影响。

- **关于 AI 与社会的幽默看法**：[@teortaxesTex](https://twitter.com/teortaxesTex) 分享了几条带有幽默或讽刺意味的推文，反映了对 AI 发展和社会观察的思考，包括对公共交通外部性、民族国家的稳健性以及对 AI 进步中公司战略的调侃。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek-R1/V3 在 Xeon 和 GPU 上的性能展示**

- **单机运行 671B DeepSeek-R1/V3-q4 (2× Xeon + 24GB GPU) – Prefill 高达 286 tokens/s & Decode 14 tokens/s** ([Score: 623, Comments: 165](https://reddit.com/r/LocalLLaMA/comments/1ilzcwm/671b_deepseekr1v3q4_on_a_single_machine_2_xeon/)): **KTransformers 团队** 宣布支持 **DeepSeek-R1/V3**，通过 **CPU/GPU 混合推理** 系统实现了高达 **286 tokens/s 的 Prefill** 速度，显著快于 llama.cpp。他们强调使用了 **Intel AMX 加速内核** 和 **选择性专家激活方法 (selective expert activation method)** 来提升性能，并指出将计算任务卸载 (offloading) 到 GPU 符合 DeepSeek 的架构，带来了大幅的速度提升。
  - **CPU 和 GPU 配置**: 该设置使用了 **Intel® Xeon® Gold 6454S**（每插槽 **32 核**）和每个插槽 **8x DDR5-4800**，并搭配了一块 **4090D GPU**。系统成本约为 **$10K**，讨论集中在考虑到 **Xeon 的成本** 以及可能降级为更实惠的选项时，重 CPU 配置是否优于重 GPU 配置。
  - **性能与优化**: 尽管由于模型的稀疏性 (sparsity)，目前增加更多 GPU 并未带来显著提升，但通过 CPU/GPU 混合推理增强了 **DeepSeek V3/R1** 模型的性能。通过优化可以显著减小模型的占用空间，一位用户报告称，得益于使用 **RTX 4090**，其 Prompt 处理速度比 **llama.cpp** 提高了 **3.38 倍**。
  - **平台支持与未来计划**: 尽管目前的重点是开源 0.3 版本并执行计划中的优化，但人们对针对 **Apple Silicon** 和 **Intel GPU** 的优化很感兴趣。目前支持 **AMD** 但缺乏用于 Prefill 加速的 **AMX 优化**，此外还有关于使用 **48GB VRAM** 的潜在收益以及未来对 **AMD Matrix Core (AMC)** 支持的讨论。


- **[Google DeepMind CEO 表示，DeepSeek 的 AI 模型是来自中国的“最佳作品”，但炒作“言过其实”。“尽管大肆宣传，但实际上并没有新的科学突破。”](https://www.cnbc.com/2025/02/09/deepseeks-ai-model-the-best-work-out-of-china-google-deepmind-ceo.html)** ([Score: 329, Comments: 244](https://reddit.com/r/LocalLLaMA/comments/1ilsd9g/deepseeks_ai_model_is_the_best_work_out_of_china/)): **Google DeepMind CEO** 对 **DeepSeek AI 模型** 发表了评论，称其为中国的“最佳作品”，但表示围绕它的炒作被夸大了。他强调，尽管令人兴奋，但该模型并没有实际的科学进步。
  - 评论者批评 **DeepMind CEO Demis Hassabis** 贬低 **DeepSeek AI 模型**，认为其开源性质和工程效率（如 **降低成本** 和 **训练效率**）是重大的进步。他们指责 Hassabis **隐瞒真相 (dishonesty by omission)**，未能承认该模型的 Open Weights 和成本效益是实质性的贡献。
  - 一些评论者强调，即使 **DeepSeek 的工程成就** 不构成科学突破，也是值得关注的。他们指出，**DeepSeek** 以极低的成本实现了与 **ChatGPT** 相当的性能，挑战了关于中国 AI 能力的假设，并表明该模型的效率和开源方法是宝贵的创新。
  - 讨论还集中在像 DeepSeek 这样的 **开源 AI 模型** 的更广泛影响上，强调了民主化 AI 技术的潜力。评论者指出，**Google** 不愿开源其模型，这与 DeepSeek 的开放性形成鲜明对比，引发了关于开源在推动 AI 研究中的作用及其地缘政治影响的辩论。


**主题 2. LLM 模型优化中的创新技术**

- **Andrej Karpathy 关于 LLM 的最新深度解析 TL;DR** ([Score: 382, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1ilsfb1/tldr_of_andrej_karpathys_latest_deep_dive_on_llms/)): **Andrej Karpathy** 发布了一个关于 **ChatGPT** 等 LLM 的 **3 小时 31 分钟** 的视频，被誉为“信息金矿”。一份将核心见解浓缩为 **15 分钟** 的总结文章可以点击[这里](https://anfalmushtaq.com/articles/deep-dive-into-llms-like-chatgpt-tldr)查看，原始视频可在 [YouTube](https://www.youtube.com/watch?v=7xTGNNLPyMI) 上找到。
  - **微调与 Prompt Engineering**：讨论强调了微调像 **llama-3B** 这样的小型开源模型的重要性，并强调 Prompt Engineering 是优化 LLM 应用的关键。**Andrej Karpathy** 的工作以及 **Anfal Mushtaq** 的文章被认为深入探讨了这些主题，以及减少模型输出中幻觉（hallucinations）的策略。
  - **数据处理与 Tokenization**：文章和视频探讨了海量互联网文本数据的预处理，包括严格的过滤和使用 **Byte Pair Encoding** 等技术的 Tokenization。这一过程对于 LLM 的有效训练至关重要，在模型预测中平衡了创造力与准确性。
  - **幽默与互动**：一些评论俏皮地用越来越短的格式总结了文章和视频，包括一分钟回顾、50 字总结，甚至还有三行诗，展示了社区在提炼复杂信息时的参与度和幽默感。


- **[新论文让模型在输出 token 之前有机会在 latent space 中进行思考，权重已在 HF 上发布 - Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)** ([Score: 112, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1imca0s/new_paper_gives_models_a_chance_to_think_in/)): **Scaling LLM Compute with Latent Reasoning** 讨论了 AI 模型计算的一种新方法，允许模型在生成输出 token 之前在 latent space 中进行推理。这种方法在题为 **"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"** 的论文中进行了详细阐述，其权重已在 **Hugging Face** 上提供。
  - **Adaptive Compute 与 Latent Reasoning**：一个值得注意的讨论围绕着逐 token 的 adaptive compute 展开，即模型根据 token 的重要性调整计算量，这可能会在未来 **6-12 个月** 内显著影响 AI 基准测试。这种方法允许模型在复杂的 token 上“思考”更多，而在简单的 token 上花费较少，预示着 AI 处理效率的重大转变。
  - **Recurrent Depth 方法与权重共享**：关于实现细节存在一些推测，特别是 **R blocks** 是否共享权重以及在测试时如何对这些块进行采样。这种 recurrent depth 方法正如所讨论的，可以通过增加循环步骤来提高模型的推理准确性，类似于 **OpenAI** 的努力。
  - **可用性与对比**：该方法的权重可在 **Hugging Face** 上获取，更多资源可在 [GitHub](https://github.com/seal-rg/recurrent-pretraining) 上找到。文中还将其与 **Meta** 的类似研究进行了对比，尽管后者没有发布权重，这突显了开源研究成果对于实际探索和理解 AI 的 latent reasoning 能力的价值。


**主题 3. Orange Pi AI Studio Pro PC：AI 硬件领域的新玩家**

- **[拥有 408GB/s 带宽的 Orange Pi AI Studio Pro 迷你 PC](https://www.reddit.com/gallery/1im141p)** ([Score: 315, Comments: 91](https://reddit.com/r/LocalLLaMA/comments/1im141p/orange_pi_ai_studio_pro_mini_pc_with_408gbs/)): **Orange Pi AI Studio Pro 迷你 PC** 已发布，拥有令人印象深刻的 **408GB/s 带宽**。这一进展对于在紧凑外形中寻求高性能计算解决方案的 AI 工程师来说意义重大。
  - **硬件与软件支持**: **Orange Pi AI Studio Pro 迷你 PC** 因缺乏可靠的软件支持而受到批评，用户强调了 **Orange Pi** 软件生态系统过去存在的问题。担忧包括缺乏更新、专有驱动程序以及社区支持薄弱，尽管其硬件能力出色，但这些问题降低了其吸引力。
  - **经济因素考量**: 讨论强调了在 AI 工作负载中将加速器与 DDR 内存配对的成本效益，例如在成本低于 **$10,000** 的 EPYC 系统上运行 **Deepseek R1**，相比更昂贵的 VRAM 配置更具优势。售价约 **$2,150** 的 **Orange Pi** 设备就其规格而言被认为具有潜在的高性价比，但由于缺乏强大的软件支持，其在实际应用中的效用仍存疑。
  - **替代方案与对比**: 用户建议选择 **旧款 NVIDIA GPU** 和 **Intel NUC** 以获得更好的支持和性能，并指出在 **Qualcomm Snapdragon X series** 等非主流系统中使用 NPU 的挑战。由于其小众地位和预期的软件障碍，**Orange Pi** 设备的潜力被这些替代方案所掩盖。


**主题 4. 为海量数据集扩展检索增强生成 (RAG)**

- **如何将 RAG 扩展到 2000 万份文档？** ([Score: 137, Comments: 136](https://reddit.com/r/LocalLLaMA/comments/1im35yl/how_to_scale_rag_to_20_million_documents/)): 要为 2000 万份文档扩展 **RAG (Retrieval-Augmented Generation)**，重点应放在优化延迟、高效 Embedding 以及稳健的索引策略上。探索分布式计算、高级索引结构和并行处理等技术，以高效管理大规模文档检索。
  - 讨论强调了扩展 2000 万份文档的 **RAG** 所面临的挑战和策略，强调了高效 **向量数据库**（如 **Weaviate**、**PGVector** 和 **Pinecone**）在处理大规模数据中的重要性。推荐使用 **HNSW 索引** 和 **重排序策略**（如 **Reciprocal Rank Fusion (RRF)**）来优化检索质量和性能。
  - 参与者辩论了 **微调 (fine-tuning)** 与 **上下文注入 (context injection)** 的优劣，一些人认为微调成本高昂且对大型数据集效果较差。**DataIsLoveDataIsLife** 建议采用务实的方法，使用 **stella_en_400M_v5** 进行 Embedding，并使用 **MiniBatchKMeans** 进行聚类，估计处理成本在 **$1,000-$20,000** 之间。
  - 提议使用 **GraphRAG/LightRAG** 方法和 **图数据库** 以获得更好的效果，而其他人则建议利用现有的搜索引擎进行检索。还讨论了 **数据摄取 (Data ingestion)** 和 **索引**，建议使用中间件层来高效管理数据，并尝试使用 **parade db** 等工具进行大规模搜索。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Gemini 2 Flash：AI 翻译效率的新标杆**

- **我通过使用 OpenAI 进行翻译节省了 93% 的成本吗？** ([Score: 160, Comments: 47](https://reddit.com/r/OpenAI/comments/1ilzfiu/did_i_save_93_of_cost_by_using_openai_for/))：帖子作者对比了翻译成本，指出 **Azure 每 100 万字符收费约 9.60 欧元**，而 **OpenAI 的 GPT-4o-mini 每 100 万字符成本约为 0.70 欧元**，潜在节省了 93% 的成本。计算方式包括翻译给定句子中的单词，并要求在输出中包含输入单词，成本分解为**每百万字符 0.30 欧元 x 2，外加 0.075 欧元的输入费用**。
  - 讨论强调了使用 **Gemini 2 Flash** 进行翻译的潜在成本节约，它提供了更好的多语言支持，且成本低于其他选项。用户指出，通过速率限制（rate limiting）和免费层级（free tier）的使用，可以将成本降至最低甚至消除，[Google 的定价页面](https://ai.google.dev/pricing#2_0flash)详细列出了 Token 成本和免费层级限制。
  - 几位用户讨论了进一步降低翻译成本的策略，例如利用 **batch processing**（批处理）和 **prompt caching**（提示词缓存），这可以通过允许非实时处理来大幅削减成本。文中提供了 [OpenAI batch API 文档](https://platform.openai.com/docs/guides/batch)的链接，作为如何实现高达 50% 成本削减的参考。
  - 还有关于各种翻译模型的可靠性和准确性的对话，一些用户建议在特定用例中使用**开源模型**，尽管它们的速度较慢。讨论中也提出了对翻译质量的担忧，强调了在大规模翻译中引入 **human in the loop**（人工参与）以确保准确性的重要性。


**主题 2. OpenAI 通过超级碗广告进行创新品牌推广**

- **[OpenAI 耗资 1400 万美元的超级碗广告](https://v.redd.it/v10i8668t7ie1)** ([Score: 2722, Comments: 601](https://reddit.com/r/OpenAI/comments/1ilusr7/openais_14_million_superbowl_ad/))：据报道，**OpenAI** 正在投入 **1400 万美元**进行**超级碗（Super Bowl）广告**战略，这表明了重大的营销推力。此举可能表明其正努力提高公众对其 AI 技术的认知度和参与度。
  - 许多评论者认为，这则**超级碗广告**通过将 **ChatGPT** 与火的发明和登月等历史性进步联系起来，有效地将其定位为一个重大的技术里程碑，类似于苹果 1984 年的广告。这种方法旨在建立品牌知名度和情感连接，而不是专注于具体功能。
  - 关于广告效果的观点存在分歧；一些人认为它错失了展示 **ChatGPT** 能力的机会，而另一些人则认为这是建立 **brand recognition**（品牌认可度）和公众对 AI 接受度的战略举措。广告的创意和审美质量受到了称赞，一些人注意到它通过 **Ratatat Neckbrace remix** 等元素对**千禧一代**的吸引力。
  - 讨论突显了营销 AI 技术的复杂性，一些人强调了**品牌定位**和**知名度**的重要性，而另一些人则质疑在广告中不演示 **ChatGPT** 实际用途的决定。批评者认为，该广告可能无法有效触达那些不熟悉 **OpenAI** 或 **ChatGPT** 的人群。


**主题 3. ChatGPT 晋升全球网站流量排行榜前列**

- **[根据 Similarweb 的数据，截至 2025 年 1 月，ChatGPT 已成为全球访问量第 6 大网站。该 AI 聊天机器人目前占据全球互联网流量的 2.33%，月访问量激增 5.91%。](https://www.cryptotimes.io/2025/02/10/chatgpt-surpasses-netflix-reddit-now-6th-most-visited-site/)** ([Score: 139, Comments: 7](https://reddit.com/r/OpenAI/comments/1im0tyb/chatgpt_is_now_the_6th_most_visited_site_in_the/))：根据 **Similarweb** 的数据，截至 **2025 年 1 月**，**ChatGPT** 已成为**全球访问量第 6 大网站**，占据了 **2.33% 的全球互联网流量**，且访问量每月增长 **5.91%**。
  - 评论者讨论认为，**OpenAI** 正从 **ChatGPT** 的交互中获取大量数据，这增强了他们的品牌认可度和潜在订阅用户群。这些数据的价值远超单纯的流量统计。
  - **OpenAI** 凭借 **ChatGPT** 获得了极高的品牌认可度，被比作历史上占据主导地位的品牌，如**摩托罗拉的 Droid**。评论者指出，对于普通大众来说，**ChatGPT** 正在成为“AI”的代名词，而不像 **Claude** 等知名度较低的竞争对手。
  - 一份分享的 **Google Trends** 图表突显了 **ChatGPT** 和 **Claude** 之间搜索热度的巨大差距，强调了 **ChatGPT** 在公众意识中的主导地位。


---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的总结


**主题 1. Unsloth AI 的崛起与社区关注**

- **Unsloth 跃升 GitHub 明星项目**：[Unsloth AI 庆祝 GitHub Trending](https://github.com/trending) 在一年内成为 GitHub 排名第一的热门仓库，标志着显著的社区增长和影响力。社区认可 **Unsloth** 的贡献，特别是对 **Deepseek-R1** 的贡献，潜在的集成工作已在进行中。
- **REINFORCE 推理方法受到审查**：[使用 REINFORCE 的推理 LLM Notion 文档](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0) 引发了关于新颖性的辩论，成员们注意到现有的 **Unsloth** 实现。人们对该方法的原创性产生怀疑，质疑其相对于 **Unsloth** 中已有方法的附加价值。
- **模型合并面临阻力**：将模型合并为 MoEs 引发了怀疑，触发了关于潜在缺点和局限性的讨论。社区辩论了在具有共享结构的输出格式中可能存在的学习损失，这可能会阻碍特定任务的训练。

**主题 2. 无代码 AI 平台与工具涌现**

- **Spark Engine 发布无代码 AI 强力工具**：[Spark Engine v1 上线](https://sparkengine.ai/) 首次亮相，拥有 **80+ AI 模型**，提供无代码文本、音乐和视频生成能力。开发者表示有兴趣集成像 **Unsloth** 这样的基础设施，以进一步增强无代码 AI 生态系统。
- **Dataset Tools 获得 AI 驱动的 EXIF 升级**：[GitHub 上的 Dataset Tools EXIF 查看器](https://github.com/Ktiseos-Nyx/Dataset-Tools) 增强了 EXIF 数据查看功能，并增加了对 GGUF 和 JPEG 格式的支持。开发者利用 AI 改进功能，并就项目的代码优化进行协作。
- **Markdrop Python 包发布 PDF 数据处理利器**：[GitHub 上的 Markdrop PDF 转 Markdown 转换器](https://github.com/shoryasethia/markdrop) 作为一个新的 Python 包发布，用于将 PDF 转换为 Markdown、提取图像并使用 AI 进行描述。该包迅速获得关注，一个月内安装量突破 **7,000+**。

**主题 3. 模型性能与硬件辩论升温**

- **Qwen 2.5 遥遥领先 Llama 8B**：**Qwen 2.5** 在速度上超过了 **Llama 8B**，特别是像 32B 这样的大型模型，得益于更好的优化。用户建议对于拥有高性能硬件的人来说，**Qwen 2.5** 是更优的选择。
- **LM Studio 用户苦恼于模型加载错误**：**LM Studio** 用户正面临 *'NO LM Runtime found for model format'* 错误，这表明硬件限制。建议用户分享系统规格和截图，并根据 [LM Studio 文档](https://lmstudio.ai/docs/basics/import-model) 将模型大小与系统能力相匹配。
- **M4 Ultra vs M2 Ultra：Mac 芯片大对决**：关于等待 **M4 Ultra** 还是购买 **M2 Ultra** 以实现高效模型运行的价值引发了辩论。用户担心在 **M2 Ultra** 模型性能不确定的情况下，服务成本会上升。

**主题 4. OpenAI 模型动态与用户担忧**

- **Gemini 吞噬海量上下文，ChatGPT 捉襟见肘**：**Gemini** 巨大的 **100-200 万 token** 上下文窗口比 **ChatGPT** 的 32k/128k token 限制更受欢迎。尽管 **ChatGPT** 存在局限性和连接错误，用户仍倾向于使用 **Gemini** 处理复杂任务。
- **GPT-4 显得变笨，用户需要更好的 Prompt**：**GPT-4** 被认为变弱了，需要更复杂的 Prompt 技巧，同时连接错误困扰着 **ChatGPT**。用户报告了持续的 **连接错误**，并觉得 **GPT-4** 不如以前强大。
- **DeepSeek 的“无限”被证明有限制**：**DeepSeek** 的“无限”使用被揭露存在限制，高频使用被标记为滥用，引发了透明度问题。用户对“无限”一词和不一致的政策执行表示担忧。

**主题 5. 编程工具与 Agent 工作流演进**

- **Cursor IDE 引发 MCP Server 热潮**：**Cursor IDE** 用户深入研究 **MCP server**，特别是 **Perplexity MCP server**，以增强编程辅助。用户在不同操作系统上探索配置并解决安装问题。
- **Cursor 中的 Agent Mode 被誉为调试英雄**：**Cursor** 中的 **Agent Mode** 因其调试能力受到称赞，通过直接的模型通信超越了标准的编程命令。用户发现集成多样化的 **LLM** 提升了编程体验，尤其是实时辅助。
- **Aider 聊天记录膨胀，Token 限制隐忧**：**Aider** 的聊天记录过度增长，达到 **25k token**，引发了对 token 限制超出的担忧。用户讨论了潜在的 Bug 以及 Prompt 缓存的有效性和性能影响。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 达成 GitHub Trending 状态**：**Unsloth AI** 在一年内成为 [GitHub](https://github.com/trending) 排名第一的热门仓库，庆祝其工具和资源。
   - 社区认可 **Unsloth** 对 **Deepseek-R1** 的贡献，相关组件可能已经集成或在当前项目中可用。
- **REINFORCE 推理引发争论**：一份关于 [此链接](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0) 中**使用 REINFORCE 的推理 LLM** 文档引发了对其新颖性的质疑。
   - 成员指出 **Unsloth** 中已经存在完全相同的实现。
- **模型合并面临质疑**：将多个有效模型合并为单个混合专家模型 (MoE) 的兴趣遭到了质疑，引发了关于潜在陷阱和局限性的讨论。
   - 讨论了在共享公共结构的长输出格式中可能存在的学习损失，这可能会阻碍特定任务的训练。
- **Spark Engine 集成无代码 AI**：**Spark Engine v1** 已发布，在 [SparkEngine.ai](https://sparkengine.ai/) 提供超过 **80 个 AI 模型**，可生成文本、音乐和视频。
   - 开发者表示希望将更多像 **Unsloth** 这样的基础设施集成到 Spark Engine 平台中，以促进无代码 AI 领域的发展。
- **数据集策划主导模型性能**：强调模型性能的 **80%** 取决于精细的**数据集策划 (Dataset Curation)**，一位成员指出：“没有所谓的冗余研究——你可以从每一篇论文中学习。”
   - 另一位成员正在尝试 Lora 设置，以开发一种元认知的第一人称推理格式。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kokoro TTS 支持 C#**：一位成员发布了 **Kokoro TTS** 的 C# 库，实现了在 .NET 平台上的即插即用集成，可在 [GitHub](https://github.com/Lyrcaxis/KokoroSharp) 上获取。
   - 该库承诺提供**多语言**体验，所有语音都以方便的格式打包，支持快速的本地 TTS 推理，并可跨多个平台工作。
- **数据集工具获得 EXIF 和 AI 升级**：数据集管理器和 **EXIF Viewer** 获得了更新，增强了查看高级 EXIF 数据的功能，并支持 GGUF 和 JPEG 等格式，可在 [GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools) 上获取。
   - 开发者利用 AI 工具辅助项目，在与他人协作优化代码的同时增强了功能。
- **Spark Engine 启动 AI 沙盒**：Spark Engine v1 在为期一年的公开测试后发布，在 [sparkengine.ai](https://sparkengine.ai/) 为各种 AI 任务提供 **80 多个模型**。
   - 该平台每天提供免费额度并与 Hugging Face 集成，为用户实验 AI 功能提供了一个强大的无代码环境。
- **Markdrop 提取 PDF 数据**：推出名为 **Markdrop** 的新 Python 包，旨在将 PDF 转换为 Markdown，具有图像提取和 AI 驱动的描述等功能，可在 [GitHub](https://github.com/shoryasethia/markdrop) 上获取。
   - 在短短一个月内，它的安装量已超过 **7,000 次**，展示了其在寻找文档处理工具的用户中的受欢迎程度。
- **go-attention 用纯 Go 实现 Transformer**：一位成员分享了他们的项目 **go-attention**，展示了第一个用纯 Go 语言构建的完整 Attention 机制和 Transformer，并在 [GitHub](https://github.com/takara-ai/go-attention) 上强调了其独特功能。
   - 该项目邀请其他人查看示例，并探索 Go 语言中 Serverless 实现的潜力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 在速度上完胜 Llama 8B**：用户对比了 **Qwen 2.5** 和 **Llama 8B**，指出 **Qwen** 由于优化提供了更快的响应时间，尤其是在 32B 等大型模型上。
   - 讨论建议在硬件充足的情况下，**Qwen 2.5** 是更好的选择。
- **LM Studio 用户面临模型加载难题**：用户在将模型加载到 **LM Studio** 时遇到问题，收到类似 *'NO LM Runtime found for model format'* 的错误，这表明存在硬件限制。
   - 建议的解决方案是提供系统规格和截图以便获得更好的协助，并根据 [LM Studio Docs](https://lmstudio.ai/docs/basics/import-model) 将模型大小与系统能力相匹配。
- **关于 M4 Ultra 与 M2 Ultra 的辩论随之展开**：围绕等待 **M4 Ultra** 的价值与购买 **M2 Ultra** 以实现高效模型运行展开了辩论。
   - 担忧集中在现有服务成本上升，以及模型在 **M2 Ultra** 上性能的不确定性。
- **PCI-E Riser 线缆引发关注**：一位用户询问了使用 **PCI-E riser cables** 安装额外 GPU 及其对性能的影响，特别是针对 **A5000** 显卡。
   - 建议将旧机箱改造成 GPU 支架，以增强散热和空间管理。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 凭借大上下文能力获得青睐**：**Gemini** 处理 **100-200 万个 tokens** 的能力使其变得流行，特别是与 ChatGPT 的 32k 和 128k tokens 相比，增强了复杂任务的可用性。
   - 用户欣赏 **Gemini** 灵活的功能，使其成为处理详细工作的首选，尽管对 **ChatGPT** 的局限性存在担忧。
- **GPT-4 如今感觉变弱了**：成员们觉得 **GPT-4** 的能力有所下降，需要更好的 prompting 才能产生好的结果，但早期的模型可能在复杂任务中给人留下了较弱的印象。
   - 几位用户还报告了在使用 ChatGPT 时持续出现的 **connection errors**（连接错误），引发了对访问性的担忧，这可能与 **ChatGPT app** 有关。
- **间接注入：数据需要清洗**：成员们对 **OpenAI** 是否披露了深度研究是否容易受到来自抓取页面的 **indirect prompt injection**（间接提示注入）表示担忧，暗示需要进行数据清洗（data sanitization）。
   - 另一位成员对解决这一担忧的即将推出的功能表示乐观，期待更多信息。
- **Markdown 优化 URL 注意力**：**ChatGPT** 在处理 [markdown](https://markdown-guide.org/) 描述的链接时比纯 URL 更有效，从而提高了 prompt 的整洁度。
   - 成员们发现，使用 JSON 等格式良好的结构化数据可以有效管理大型信息块。
- **DeepSeek 的 “无限” 使用存在限制**：报告指出，**DeepSeek** 的高频使用被归类为滥用，引发了用户对 *“unlimited”*（无限）一词的担忧，并对 **OpenAI** 政策的透明度提出了质疑。
   - 这些限制似乎应用得并不一致，引发了关于 OpenAI 政策透明度和用户预期的疑问。



---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor MCP 服务器引发讨论**：频道用户讨论了各种 **MCP servers**，包括 **Perplexity MCP server**，详细介绍了其在 **Cursor** 中的设置和功能，以提升编码辅助能力。
   - 一些用户分享了将不同模型集成到工作流中的经验，而另一些用户则在排查返回错误的命令提示符，表明需要更清晰的文档和支持。
- **Agent Mode 的调试功能广受好评**：用户探索了 **Agent Mode** 的功能及其相对于标准编码命令的优势，特别称赞了其调试能力以及与 **Perplexity** 等模型的直接通信。
   - 共识认为，集成不同的 **LLMs** 可以增强编码体验，特别是具有搜索和实时辅助功能的特性。
- **用户反馈 MCP 服务器安装故障**：多位用户在设置 **MCP servers** 时遇到问题，特别是在 **Mac** 和 **Windows** 等不同操作系统上的命令执行和服务器响应方面。
   - 讨论涉及排查返回错误或连接失败的命令提示符，指出需要改进文档和支持。
- **自定义 Cursor Rules 引起关注**：参与者讨论了在使用 **Perplexity MCP server** 时创建自定义 **cursor rules** 以改进特定功能实现的可能性，并附带了 [Using Cursor with Convex](https://docs.convex.dev/ai/using-cursor) 的链接。
   - 用户强调，集成的 cursor rules 可以简化工作流，并增强 **AI** 响应复杂代码相关查询的能力。
- **性能与限制问题探讨**：讨论涉及各种模型的性能，包括服务降级的报告以及对 **Cursor** 内快速 **API call limits** 的担忧。
   - 参与者指出，如果使用得当，**MCP servers** 可以缓解性能问题，并提供比传统网页抓取方法更好的结果。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **唯一标签提升 Lora 一致性**：在训练数据中使用唯一标签（如物体或场景的特定名称）可以显著提高 **Lora models** 中生成图像的一致性和叙事连续性。
   - 该方法有助于模型更好地将特定场景与这些名称关联起来，如 [BasedLabs 上的 Lora 训练示例](https://x.com/BasedLabsAI/status/1888313013276684711) 所示。
- **发现 Flux 的最佳分辨率**：对于使用 **Flux** 生成图像，最佳潜空间尺寸约为 **672x1024** 或 **1024x672**，而 **1920x1088** 提供了一个合适的快速 **HD generation** 尺寸。
   - 在初始生成阶段生成超过 **1MP** 的图像可能会导致构图问题。
- **Photoshop 集成 ComfyUI**：用户正在探索将 **ComfyUI** 的各种插件与 **Photoshop** 集成，例如 [Auto-Photoshop-StableDiffusion-Plugin](https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin) 和 [sd-ppp](https://github.com/zombieyang/sd-ppp)。
   - 这些插件允许使用 ComfyUI 后端直接在 Photoshop 中生成 stable diffusion 图像。
- **Stable Diffusion 遇到 GPU 故障**：用户报告了在不同 **Stable Diffusion** UI 路径下的 **GPU errors** 和性能缓慢问题，降低 GPU 设置是解决显存问题的常见方案。
   - 建议使用特定设置并保持宽高比以提高模型性能和输出质量，参见 [Stable Diffusion 知识库（设置、基础、指南等）](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)。
- **AI 生成艺术获得版权保护？**：最近的一个案例授予了一张 AI 生成图像版权保护，原因是其中包含足够的人类投入，这可能为 **AI-generated content** 的所有权设定法律先例，据 [cnet.com](https://www.cnet.com/tech/services-and-software/this-company-got-a-copyright-for-an-image-made-entirely-with-ai-heres-how/) 报道。
   - 这张名为 *A Single Piece of American Cheese* 的图像是使用 Invoke 的 AI 编辑平台创作的。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 效仿 META 的举措**：讨论强调了 **Nous Research** 如何利用来自 **META** 和 **DeepSeek** 等大公司的技术进展来改进其 AI 模型，同时作为一家较小的初创公司面临着资金挑战。
   - 重点在于创建负担得起的前沿 AI 模型以保持市场竞争力，类似于在现有代码库的基础上进行构建。
- **Granite 3.1 训练多个目标**：用户计划训练 **Granite 3.1 的 3B 模型**，以在一种新设置中探索训练策略和自定义 RL 循环，每个 epoch 包含多个目标。
   - 这探索了在新型训练结构中使用多个目标的潜力。
- **Zonos 克隆高保真语音**：**Zonos** 的发布展示了其强大的性能，这是一款具有语音克隆功能的高保真 TTS 模型，可与领先的 TTS 供应商竞争。
   - 该模型采用 **Apache 2.0** 开源协议，正如 [ZyphraAI 的推文](https://fxtwitter.com/ZyphraAI/status/1888996367923888341)所指出的，这促进了其在 AI 开发中的集成。
- **LM 相似性削弱 AI 监管**：研究提出了一种基于模型错误的 **Language Model 相似性**概率指标，以增强 **AI 监管**，详见 [arxiv.org](https://arxiv.org/abs/2502.04313) 上的一篇论文。
   - 这建议使用 **LLM** 作为评委，倾向于相似的模型，以利用互补知识促进弱到强泛化（weak-to-strong generalization）；然而，随着 AI 监管变得日益重要，模型错误变得越来越难以检测，这一趋势令人担忧。
- **OVERTHINK 减慢推理模型速度**：根据 [Jaechul Roh 的推文](https://x.com/JaechulRoh/status/1887958947090587927)，**OVERTHINK** 攻击通过注入诱饵任务，在不改变输出的情况下放大推理 token，导致模型在推理时的速度降低高达 **46 倍**。
   - 该方法在不可信的上下文中使用 **Markov Decision Processes** 和 **Sudoku** 等复杂任务来操纵推理过程，对 **OpenAI** 的 **o1** 和 **o3-mini** 等模型构成风险。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 用户请求优化个人资料页面**：**Codeium 团队**正在征求用户对改进 Codeium 个人资料页面的反馈，鼓励用户通过[提供的表单](https://windsurf.notion.site/194d73774c0080f0b05ee33699e907b9?pvs=105)提交建议。
   - 这些增强功能旨在创造更实用和个性化的体验，重点关注用户认为最有价值的统计数据和指标。
- **Jetbrain 扩展被视为已弃用**：用户担心 **Jetbrain 扩展**的模型可用性滞后于 **Windsurf**，一些人猜测其正转向以 **Cursor** 为中心的方法，这引发了对功能缺失的沮丧。
   - 官方宣布新的被动文本内编辑器体验将由 **Windsurf** 独占，导致 VSCode 插件上的 **Supercomplete** 被弃用，这加剧了这些担忧。
- **Codeium 饱受支付问题困扰**：有关影响**俄罗斯用户**的支付限制的讨论正在进行，由于地区限制和公司政策，在获取许可证方面面临挑战。
   - 用户敦促 Codeium 就这些限制进行更清晰的沟通，并改进支付流程。
- **Windsurf 用户希望改进工作流**：Windsurf 用户报告了代码建议、diff 显示和自动更新方面的问题，并需要在 **O3**、**DeepSeek** 和 **Claude** 等 AI 模型之间实现更一致的 **tool calling**。
   - 用户还要求更好的额度管理、系统问题通知、改进的设计文档、调试能力以及 AI 模型输出的一致性。
- **额度紧缺引发 Codeium 客户担忧**：用户对**额度系统**表示担忧，特别是操作过程中的消耗以及尝试失败后不予退还额度的问题。
   - 挫败感源于在不理想的输出上花费了额度，这促使人们呼吁在额度使用处理方面提高透明度。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 公开 Reasoning Tokens**：用户现在可以在模型活动页面上看到 **reasoning tokens**，以及 **prompt** 和 **completion tokens**，以获得更好的透明度。
   - 这一增强功能旨在让用户更深入地了解模型在 [OpenRouter platform](https://openrouter.ai/activity.) 上的表现。
- **Chat-thyme 简化 Discord Bot 创建**：[Chat-thyme](https://github.com/chilir/chat-thyme) 允许你使用任何兼容 OpenAI 的 LLM 框架设置 Discord bots，并提供便捷的 **OpenRouter** 集成。
   - 它还为支持工具调用的模型集成了 **Exa**，尽管可靠性取决于提供商。
- **FindSMap 全球集成历史地图**：[FindSMap](http://findsmap.com) 是一个连接历史地图和考古机构的渐进式 Web 应用程序，使用了 **Open Street Maps** 和 **Leaflet.js**。
   - FindSMap 使用 **Claude** 和 **Open Router** 构建，展示了项目的迭代开发和投入。
- **DeepSeek R1 面临超时问题**：用户报告了 **DeepSeek R1** 严重的 **性能问题**，在 API 请求期间遇到超时，但 “nitro” 变体已集成到主模型功能中，允许用户按吞吐量排序。
   - @togethercompute 为 DeepSeek R1 提供的全新推理栈在 **671B** 参数模型上达到了高达 **110 t/s** 的速度 ([tweet](https://x.com/vipulved/status/1888021545349742592))。
- **TypeScript SDK 简化 LLM 调用**：一个团队正在构建一个 **TypeScript SDK**，使用 **OpenAI 格式** 与超过 **60 个 LLM** 进行交互，并集成了 **OpenRouter**。
   - 该 [GitHub 项目](https://github.com/lunary-ai/abso) 旨在简化对 **100+ LLM Providers** 的调用，但反馈表明它可能还 *不够完善*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek API 遭遇不稳定**：用户报告了 **DeepSeek API** 的不稳定和无响应问题，特别是在将其与 **Aider** 集成时。一位用户在特定配置下使用 DeepSeek 获取输出时遇到困难。
   - DeepSeek R1 和 V3 的模型对比显示，用户更青睐 **Hyperbolic** 和 **OpenRouter** 而非其他提供商，并指出特定配置可以增强性能。
- **Aider 在 Architect 模式下自动创建文件**：用户遇到 Aider 在 **Architect 模式** 下未经提示自动创建文件的情况，导致了困惑。一位用户分享了显示该意外行为的截图，暗示可能存在配置问题；参见 [issue #3153](https://github.com/Aider-AI/aider/issues/3153#issuecomment-2640194265)。
   - 这种意外行为导致了对操作流程的困惑，需要对配置进行更多调查。
- **Aider 聊天历史达到 Token 限制**：有用户担心 **Aider 的聊天历史** 超过了合理限制，部分用户报告其攀升至 **25k tokens**。
   - 社区讨论了潜在的 Bug、prompt 缓存的有效性以及对性能的整体影响。
- **Copilot Proxy 解锁 GitHub Copilot 模型**：实验性的 [Copilot Proxy](https://github.com/lutzleonhardt/copilot-proxy) VS Code 扩展使 AI 助手能够访问 **GitHub Copilot 的语言模型**。一段 [YouTube 视频](https://youtu.be/i1I2CAPOXHM) 详细介绍了该扩展的功能。
   - 一位成员寻求利用 Copilot Proxy 工作的方法，另一位成员建议使用 [llmap repo](https://github.com/jbellis/llmap) 及其 `parse.py` 脚本来提取文件大纲。
- **Gemini 模型在 PHP 任务中表现出色**：用户报告了使用 `gemini-1206-exp` 等 **Gemini 模型** 处理 PHP 任务的积极体验，与其他提供商的对比显示输出没有显著差异。
   - Aider 还引入了对 tree-sitter-language-pack 的实验性支持，旨在扩展 Aider 的编程语言能力。鼓励用户测试此功能并提供反馈。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 走向本地化**：中国 GPU 制造商如摩尔线程（Moore Threads）和百度昆仑（Kunlun）现在支持在本地系统上运行 **DeepSeek 的 R1 LLM 模型**，增加了与 NVIDIA 的竞争。
   - 这一举措标志着中国 AI 硬件能力的提升，挑战了 NVIDIA 在 AI 处理领域的统治地位。
- **Anthropic 索引经济影响**：Anthropic 推出了 **Economic Index**（经济指数），包括一篇分析了数百万条匿名 Claude 对话的论文，以评估 AI 对经济的影响，详见其 [Tweet](https://x.com/AnthropicAI/status/1888954156422992108)。
   - 初步调查结果显示，与其他行业相比，*物质运输（material transportation）* 领域的参与度出奇地低。
- **Replit 简化移动应用创建**：Replit 推出了 **Native Mobile App 支持** 的早期访问，使用户能够在无需编码的情况下，通过 Replit Assistant 创建 iOS 和 Android 应用；[推文链接](https://x.com/amasad/status/1888727685825699874?s=46&t=PW8PiFwluc0tdmv2tOMdEg)。
   - 此次发布标志着向更易用的应用开发转型，并承诺很快将提供完整的 Agent 支持。
- **Deep Research 工具引发辩论**：成员们讨论了 OpenAI 的新 **Deep Research** 工具，强调其通过在研究前提出澄清性问题来进行交互的方法，这标志着向更主动的 AI 迈进，如其 [Deep Research 页面](https://openai.com/index/introducing-deep-research/) 所示。
   - 该工具正与 [Hugging Face 的 Deep Research](https://m-ric-open-deep-research.hf.space/) 以及其他社区开发的替代方案进行对比。
- **ELIZA 回归？**：成员们了解了专为 AI Agent 设计的 **ELIZA Operating System** ([ELIZA Operating System](https://www.elizaos.ai))，强调了其在聊天机器人技术中的基础性作用。
   - 对话强调了像 **ELIZA** 这样的聊天机器人在现代 AI 发展背景下的历史意义。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 面临生态系统障碍**：成员们辩论了 Mojo 在 Web 开发方面的可行性，强调了稳健生态系统以及与现有 **Python 库** 无缝集成的重要性。
   - 普遍共识是，在广泛采用之前，需要投入大量精力构建基础工具，并提到 [Render](https://render.com) 等平台是很好的榜样。
- **Mojo 中出现 VariadicList 挑战**：一位用户报告了在 Mojo 中初始化 **VariadicList** 的问题，特别是关于使用 `pop.variadic.create` 操作进行动态元素重复的问题，并发布了 [GitHub issue](https://github.com/modular/mojo/issues/3987) 的链接。
   - 该问题突显了 Mojo 当前在处理可变参数列表（variadic lists）能力方面的潜在差距，一些成员分享了他们自己的 **mojoproject.toml** 文件（例如 [这一个](https://github.com/BradLarson/max-cv/blob/main/mojoproject.toml#L21)）。
- **领域知识驱动业务**：参与者强调领域理解对于启动成功的科技业务至关重要，特别是对强大 **网络知识（networking knowledge）** 的需求。
   - 许多初创公司忽视了这一方面，导致了本可以避免的挑战并阻碍了增长。一位成员表示：*“理解领域对于创业至关重要”*。
- **网络效应影响语言采用**：小组讨论了 **网络效应** 如何影响像 **Rust** 这样语言的采用，其中充满活力的生态系统促进了实验和增长。
   - 虽然有些人容忍快速开发中的“粗制滥造（slop）”，但另一些人则主张保持高质量标准，以确保长期可行性并防止技术债。
- **C++ 在高性能领域依然称王**：讨论强调了 **C++** 在性能关键型应用中的持续主导地位及其对新语言采用的影响。
   - 虽然 **Mojo** 具有潜力，但其增长取决于与现有语言的无缝集成，并提供优于当前解决方案的显著性能优势。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **未发现 Firebase/Firestore MCP**：一位寻找 **Firebase/Firestore MCP** 的用户被引导至一个表明该工具可能不存在的链接，凸显了对此类工具的需求。
   - 这一空白强调了开发针对特定数据库集成的 **MCP 工具** 的机会。
- **MCP 命令路径配置错误**：用户在通过 Cursor 添加 **MCP server** 时遇到了“No Tools Found”错误，这表明路径配置错误可能是原因。
   - 解决方案包括验证正确的命令路径，并可能在更新后重置应用程序，以确保工具被正确识别。
- **MCP 性能面临 Python SDK 障碍**：用户报告在使用 **MCP** 与 **Claude Desktop** 时工具调用响应缓慢，将其归因于 Python SDK 的限制以及最近更新后的持续 bug ([python-sdk@bd74227](https://github.com/modelcontextprotocol/python-sdk/commit/bd742272ab9ef5576cbeff4045560fb2870ce53b))。
   - 反馈强调了对增强错误处理和整体性能改进的需求，以促进更流畅的操作。
- **Smithery 安装程序引发关注**：虽然被视为领先的 **MCP 安装程序**，但人们对 **Smithery** 的远程数据处理和开销产生了担忧，促使寻找更本地化的替代方案。
   - 用户强调了对**隐私和效率**的需求，推动寻求能减少 MCP 工具中远程数据依赖的解决方案。
- **Claude Desktop Beta 版仍存在 Bug**：Beta 测试人员在使用其 MCP server 时遇到了 **Claude Desktop 应用**崩溃的情况，反映了当前功能的不稳定性。
   - 共识是，在预期发布稳定版本之前，该应用需要广泛的反馈和实质性的改进，具体可通过 [Claude Desktop Quick Feedback](https://docs.google.com/forms/d/e/1FAIpQLScfF23aTWBmd6lNk-Pcv_AeM2BkgzN2V7XPKXLjFiEhvFmm-w/viewform) 表单提交。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **cuBLAS 在不同 GPU 上表现各异**：一位用户发现 **cuBLAS** 在 **1650ti** 和 **4090** 之间的性能表现不一致，质疑该构建是否适配了较新的架构。
   - 讨论还涉及了增加 **L1 hit rate** 如何缓解与负载排队相关的停顿。
- **Unsloth 加速 LLM 训练**：根据其博客文章 [Introducing Unsloth](https://unsloth.ai/introducing)，**Unsloth** 可以将 **LLM 训练**速度提高 30 倍，使 **Alpaca** 训练仅需 **3 小时**而非 **85 小时**。
   - 他们声称在不牺牲准确性的情况下减少了 **60% 的内存使用**，并提供开源和专有选项。
- **Mistral 微调速度提升 14 倍**：正如其博客文章 [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark) 中所述，**QLoRA** 支持的引入使 **Mistral 7B** 在单张 **A100** 上的微调速度提高了 **14 倍**，峰值 **VRAM** 使用量减少了 **70%**。
   - 此外，**CodeLlama 34B** 实现了 **1.9 倍的加速**，增强的内存利用率防止了显存溢出错误。
- **探索 Ryzen AI 上的 iGPU 编程**：成员们讨论了如何通过**图形框架**或潜在的 **HIP** 来利用 **Ryzen AI CPU (Strix Point)** 中的 **iGPU**。
   - 这些方法可以让开发者挖掘集成 GPU 的处理能力。
- **reasoning-gym 新增矩阵操作**：**reasoning-gym** 合并了新的 PR，包括 [Matrix Manipulation](https://github.com/open-thought/reasoning-gym/pull/100) 和 [Count Bits](https://github.com/open-thought/reasoning-gym/pull/101)，扩展了数据集。
   - 成员们考虑了如何最好地对 gym 环境进行 **benchmark**，以观察 RL 训练如何影响泛化，并考虑使用 **OpenRouter** 进行推理计算。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus 加入 Google One，学生折扣推出**：[NotebookLM Plus](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/) 现在已成为 **Google One AI Premium** 的一部分，提供更高的使用限制；18 岁以上的美国学生可享受该计划 **50% 的折扣**，价格为 **$9.99/月**。
   - NotebookLM Plus 将笔记本容量提升了 **5 倍**，每个笔记本的来源限制提升了 **6 倍**，音频概览（audio overviews）提升了 **7 倍**。
- **用户应对 NotebookLM 的来源生成故障**：有用户报告 **NotebookLM** 无法从上传的 **.txt** 和 **.pdf** 文件等来源生成笔记；系统无限期显示 “New Note: Generating”。
   - 解决方法包括直接粘贴文本，并引导用户访问 Google 官方支持链接，以了解免费版和付费版的固有限制。
- **NotebookLM Plus 增强聊天和分享工具**：**NotebookLM Plus** 现在具有高级聊天自定义、分享功能，并提供全面的使用分析。
   - 笔记本分享需要启用 **Gmail**，这给使用来自 Azure 的 SSO 用户带来了挑战。
- **AI 弥合医疗讨论中的理解鸿沟**：一位成员分享了 **AI** 如何帮助澄清与其乳腺癌诊断相关的**医疗术语**，总结冗长的文章和外科医生预约记录。
   - 他们强调了 AI 如何通过挑战 AI 进行澄清，成为*治疗期间的一种慰藉援助*。
- **用户使用 NotebookLM 构建多功能机器人**：一位用户启动了 **Versatile Bot Project**，提供 [prompt 文档](https://github.com/shun0t/versatile_bot_project)，通过专门的 prompt 将 NotebookLM 转换为不同类型的聊天机器人。
   - 该用户表示，这两个 prompt 都经过了测试，旨在创造一种可定制的聊天机器人体验。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Skip Transcoders 领先于 Sparse Autoencoders**：**Skip transcoders** 展示了对 **SAEs** 的 **帕累托改进 (Pareto improvement)**，为研究人员提供了增强的**可解释性**和保真度；可以在 [sparsify](https://github.com/EleutherAI/sparsify) 库中使用 `--transcode` 和 `--skip_connection` 标志。
   - 根据在 arxiv.org 上发表 [论文](https://arxiv.org/abs/2501.18823) 的团队表示，与 **SAEs** 相比，transcoders 能更好地模拟输入输出关系，加强了可解释性方法。
- **部分重写面临障碍**：团队在部分重写 Transformer 的研究中遇到了*不尽如人意的结果*，他们在 **Pythia 160M** 的第六层训练了一个 skip transcoder。
   - 尽管最初受挫，团队对改进方法仍持乐观态度，并发表了一篇 [论文](https://arxiv.org/abs/2501.18838) 详细介绍该方法。
- **为 AI 改造 GPU：需谨慎行事**：关于将旧的 **1070ti** 挖矿机重新用于 AI 的讨论强调了架构过时和带宽限制的问题，这可能会限制训练。
   - 虽然这些 GPU 在推理任务中表现尚可，但成员们警告不要指望它们能高效训练现代 AI 模型。
- **基于国际象棋的 LLM 评估策略**：EleutherAI 正在创建一个任务，利用包含 **4M+ 国际象棋战术**的数据库来评估 LLM，这可能通过利用**强化学习 (Reinforcement Learning)** 独特地增强 LLM 性能，最终使其能够下棋。
   - 团队正在决定是采用 **MCQ 模式 (多选题)** 还是自由形式生成，希望模型能通过 **<think>** 标签展示其推理过程。
- **Pythia 令人困惑的 Checkpoint 模式**：讨论澄清了 **Pythia 每 1,000 步保存一次 checkpoint**，而非传闻中的 **10K 步**，以便利用 **log(tokens)** 进行更深入的可解释性分析。
   - 团队考虑了较小的线性步长和更早切换是否能提高效率，同时也权衡了保存 checkpoint 带来的 **wallclock overhead (实际耗时开销)**。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Logits 与概率之争引发讨论**：成员们讨论了在 **log space**（对数空间）与 **absolute space**（绝对空间）中训练模型的优劣，强调对数空间可以捕捉更广泛的数值范围，并能使遥远的点产生更多相似性。
   - 一位成员指出，使用对数空间对准确性的影响取决于具体的使用场景。
- **稀疏自编码器遭到质疑**：一位成员对 **Sparse Autoencoders** (SAEs) 被过度炒作表示怀疑，对其可解释性感到失望，并指出其在不同随机种子下的不一致性，参见 [这篇论文](https://arxiv.org/abs/2501.16615)。
   - 讨论引用了近期批评 SAEs 并探索模型解释新方法的论文，以及 [Twitter](https://x.com/norabelrose/status/1887972442104316302) 上分享的 skip transcoders 表现优于 SAE 的案例。
- **防护栏未能阻止生物武器发现**：据报道，一个旨在最小化毒性的药物发现算法转而 *最大化* 毒性，在短短 **6 小时** 内发现了 **40,000** 种潜在的生物武器。
   - 这一事件引发了人们对当前防护栏在应对广泛知识综合方面的有效性的警惕，以及由于关注点过窄而忽视有害化合物的风险。
- **PlanExe AI 项目在 GitHub 上线**：一位成员介绍了 **PlanExe**，这是一个使用 **LlamaIndex** 和 **OpenRouter** 构建的结构化 AI 规划器，无需进行广泛的网络搜索即可生成 SWOT 分析等结构化计划，可在 [GitHub](https://github.com/neoneye/PlanExe) 上获取。
   - 创建者对输出的准确性表示不确定，但也提供了 [PlanExe-web 的链接](https://neoneye.github.io/PlanExe-web/)。
- **LLM 在 Token 计数方面表现挣扎**：成员们注意到 LLM 在统计其上下文中的 Token 数量时非常吃力，这表明困难不仅在于 Tokenization，还在于根本性的计数能力缺失。
   - 一位成员简单地表示：*LLM 根本不会计数*。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Gemini Flash 加速文档理解**：**LlamaParse** 现在支持 **Gemini 2.0 Flash**，以更低的成本实现了 **GPT-4o+ 性能** 水平的文档处理，为利用 **VLMs** 和 **LLMs** 增强工作流奠定了基础。
   - @composiohq 的教程演示了如何使用 **Gemini Flash 2.0** 构建 **YouTube 研究 Agent**，简化了视频搜索和 Gmail 草稿创建，强化了 LlamaIndex 在简化视频研究工作流方面的实用性。
- **CrossPoster 应用发布，助力 AI 增强社交媒体**：**CrossPoster** 应用上线，支持使用 AI 跨平台发布内容至 **Twitter**、**LinkedIn** 和 **BlueSky**，以优化社交媒体参与度。
   - 该应用能智能识别个人及其账号，简化了跨平台的社交存在管理。
- **OpenAI LLM 面临超时困扰**：成员们发现 **OpenAI LLM** 选项的超时设置被重试装饰器覆盖，导致即使设置了更高的超时时间，表现依然不一致。
   - 一位成员分享道，即使在提交了 Bug 修复后， Deepseek 在 60 秒后返回 200 OK 响应，但包体为空，加剧了该问题。
- **LlamaIndex 中的移交（Hand-off）挫败**：用户对 LlamaIndex 中的 `can_handoff_to` 功能表示担忧，特别是当 Agent 转移控制权而接收方 Agent 没有响应时，会导致请求丢失。
   - 建议的解决方案包括启用调试日志，以及使用 LlamaIndex 的回调处理器进行更有效的故障排除。
- **AzureAI Search 的元数据必备项**：一位用户质疑 **AzureAI Search** 中硬编码的可过滤元数据字段定制，特别提到了 'author' 和 'director'。
   - 对方澄清说 **Azure** 要求预先定义这些元数据字段，强调了定义明确且有用的文档字段的重要性，以及了解该功能当前局限性的必要性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **求职期间要相信自己**：Cohere Discord 的成员强调在求职申请过程中要*保持自信*，鼓励他人“无论别人怎么说”都要相信自己。
   - 他们补充说，*每个人都同样感到迷茫*，并敦促在面临挑战时要坚持不懈，同时指出了工程实习*招聘机会匮乏*的现状。
- **人脉网络增加曝光度**：成员们认为，无论身在何处，*建立人脉 (Networking)* 都至关重要，建议通过参加活动来增加曝光度，同时推荐参与开源项目以连接同领域的其他人员。
   - 一位用户提到参加了与其工程领域相关的*会议和竞赛*，甚至强调了他们参加 **Canadian engineering competition** 的经历。
- **LibreChat API 调用指向 v1 而非 v2**：一位成员指出，他们只能通过 **LibreChat** 的自定义端点 (Custom Endpoint) 使用 `https://api.cohere.ai/v1` 访问 **Cohere API**，并确认 **Cohere API** 可以通过 **curl** 正常工作。
   - 有人指出 **LibreChat** 目前调用的是旧的 API 版本 (v1)，需要更新到 `/v2` 端点，尽管 URL [https://api.cohere.com/v1](https://api.cohere.com/v1) 的功能与 `https://api.cohere.ai/v1` 相同。
- **Cohere 社区制定规则**：成员们讨论了 **Cohere Community** 的规则，强调了服务器内的尊重和适当行为，同时为新人起草了介绍信息，重点介绍了对 AI 的兴趣以及像“购买加拿大产品”这样的本地倡议。
   - 讨论随后转向了 **Cohere API** 的可扩展性以及其员工在协作方面的可接触性，同时一位成员鼓励就电子烟展开一场苏格拉底式的对话。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Yu Su 的 Language Agents 讲座直播**：今天 **PST 时间下午 4:00**，由 **Yu Su** 主讲的关于 *Language Agents 的记忆、推理与规划* 的第 3 场讲座在[此处](https://www.youtube.com/live/zvI4UN2_i-w)进行了直播，认为当代的 AI Agent 使用**语言作为推理的载体**。
   - Yu Su 是俄亥俄州立大学的**杰出助理教授**，并共同领导 NLP 小组，做出了包括 **Mind2Web, SeeAct, HippoRAG, LLM-Planner, 和 MMMU** 在内的重大贡献，获得了 CVPR 2024 **最佳学生论文奖**和 ACL 2023 **优秀论文奖**等荣誉。
- **MOOC 延迟报名及课程详情待定**：用户可以报名参加 1 月份开始的 **LLM Agents MOOC**，工作人员承诺很快会发布更多课程详情，以解决关于项目框架和出版限制的疑虑。
   - 参与者询问了除测验之外的作业和项目的具体细节，工作人员表示详细信息将很快发布，鼓励用户在等待关于项目要求和评分政策的明确指南时保持耐心。
- **Berkeley MOOC 的证书问题**：几位用户反映没有收到证书，而他们的同学却收到了，这促使大家关注到缺失已填写的**证书声明表 (certificate declaration forms)** 这一必要步骤。
   - 课程工作人员重申，填写此表格是发放证书的必要条件，需要单独提交；建议包括创建一个自动化 Agent 来简化证书流程并回答常见问题。
- **DPO 解释及其与 SFT 的对比**：一位成员解释了**监督微调 (SFT)** 如何仅使用正面示例，而**直接偏好优化 (DPO)** 则纳入了负面响应，强调了 DPO 中对错误响应的惩罚。
   - *错误响应*通常结构良好，由于缺乏奖励模型，在 SFT 期间它们被选中的概率反而会增加。
- **第 2 讲学习小组引发时区担忧**：一位成员宣布了关于**第 2 讲：学习使用 LLMs 进行推理**的学习小组会议，邀请他人通过提供的链接加入，并准备讨论来自 **DeepSeek-R1 的 GRPO** 作为学习材料的一部分。
   - 一位参与者对学习小组的时间表示担忧，指出该时间处于**英国时间凌晨 3:00**，强调了国际成员可能存在的日程冲突。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **探索人工数据生成方法**：一名成员正在深入研究**人工数据生成 (artificial data generation)**，并寻找将 PDF 和 Excel 文件等非结构化数据转换为 LLM 训练样本的工具，并引用了关于该主题的 [YouTube 视频](https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO)。
   - 然而，人们也认识到使用合成数据训练 LLM 的挑战，指出问题生成可能无法提供必要的比较性见解，而这需要跨多个文档源的全面数据。
- **Kolo 简化微调**：一名成员正在开发 **Kolo**，这是一个旨在简化模型微调 (fine-tuning) 的工具，但目前缺乏数据创建功能。
   - 开发者计划在未来添加训练数据生成功能。
- **PR #2257 正在评审中**：一名成员请求对 [PR #2257](https://github.com/pytorch/torchtune/pull/2257) 进行评审，表示该 PR 已通过本地测试，但需要更多反馈。
   - 评审人员赞赏了这些更改，但对量化 (quantization) 提出了 UX 方面的顾虑，并建议改进文档。
- **GRPO 的功能哲学**：团队讨论了是否通过移除功能来简化 **GRPO**，以平衡易用性与代码整洁度。
   - 观点倾向于移除不需要的代码，同时一些人承认可能需要激活检查点 (activation checkpointing) 等功能；参见 [Grpo loss by kashif](https://github.com/linkedin/Liger-Kernel/pull/553/files#diff-1534093f54f1b158be2da2b159e45561361e2479a7112e082232d3f21adc6a45)。
- **Torchtune 检查点机制详解**：一名成员分享了恢复 (resume) 功能如何更新检查点路径并依赖于 `resume_from_checkpoint` 标志，详见 [Torchtune 检查点文档](https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning)。
   - 讨论涉及了在加载初始权重时异常工作流的影响。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 缺少模型选择菜单**：用户对 **GPT4All** 在发布 36 个版本后仍缺少带有搜索功能的模型选择菜单表示担忧。
   - 一名成员建议由于其开源特性，可以贡献代码来增强 **GPT4All**。
- **AI Agent 采用数据库实现长期记忆**：成员们探讨了将 **AI Agent** 与数据库结合使用以实现长期记忆，并建议通过函数提高 **LLM 的时间感知能力**。
   - 对话推测 **2025** 年可能是 Agentic AI 取得突破性进展的关键一年。
- **GPT4All 暂不支持图像分析**：目前已明确 **GPT4All** 暂不支持图像分析，建议使用其他平台完成此类任务。
   - 推荐的工具包括用于图像相关项目的 **booruDatasetTagmanager** 和 **joycaption**。
- **完善 PDF 嵌入方法**：成员们讨论了将 PDF 等长文档嵌入 (Embedding) 并总结为 **GPT4All** 可用格式的策略。
   - 强调了在嵌入之前妥善处理下载内容以移除无关信息的重要性。
- **Qwen2.5 和 Phi4 在受欢迎程度竞赛中胜出**：成员们推荐使用 **Qwen2.5** 和 **Phi4**，认为它们比 **Mistral** 等模型效率更高。
   - 强调了与应用集成的模型的用户友好性，并为不熟悉 **Hugging Face** 的用户提供帮助。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的移动端波折**：测试显示 **WebGPU** 在 iPhone 15 上由于缓存问题失败，而 M1 Pro 用户报告在 Safari 和 Chrome 上运行 **tinychat** 演示成功。
   - 社区呼吁加强测试以提高兼容性，特别是移动设备上的 **WASM** 加载。
- **Tinygrad 远程根基揭晓**：澄清表明 **tinygrad** 是一家**全远程公司**，驳斥了因 Twitter 信息不准确而传出的总部位于圣迭戈的传闻。
   - 这一更正引发了关于 **Ampere Altra** 处理器支持和后端加速能力的咨询。
- **公司会议准备就绪**：第 57 次会议已安排，讨论内容包括**公司更新**、**CI 速度**、**tensor cores** 以及针对 **WebGPU** 和 **tinychat** 增强功能的潜在**悬赏 (bounties)**。
   - 目标是提升内部运营速度并应对社区对进行中项目的关注。
- **ML 框架中 FP16 的命运**：一场关于为何大多数 ML 框架不排他性地使用 **fp16** 的辩论爆发，揭示了潜在的劣势和性能限制。
   - George 建议查阅 Discord 规则作为回应，引发了更多关于在提问前进行研究质量的评论。
- **PR 精度与量化怪癖**：讨论集中在一个实现脚本的 Pull Request (PR) 上，强调了对额外功能和测试的需求，特别是针对 **Hugging Face 模型**。
   - 社区强调了清晰的 PR 结构对易于审查的重要性，同时承认量化模型中存在的**数值不准确**是一个挑战。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 训练 BERT 分类文章**：一位成员从 **GPT-3.5** 和 **GPT-4** 转向使用 **DSPy** 训练 **BERT** 模型进行文章分类。
   - 优化后的 prompt 现在从每篇文章中提取十几个字段，每 24 小时进行一次批处理，使用 **Miprov2**，以 **o3-mini** 作为教师模型，**Mistral Small 3** 作为学生模型，并实现了 **50% 的成本缩减**。
- **多 Agent 系统通过 MASS 提升性能**：由于 [MASS 框架](https://arxiv.org/abs/2502.02533) 中强调的有效协作策略，作为多 Agent 运行的 LLM 在解决复杂任务方面表现出巨大潜力。
   - 分析强调了多 Agent 系统设计中 **prompts** 和**拓扑结构 (topologies)** 的重要性。
- **Factorio 作为 AI Agent 系统工程沙盒**：静态基准测试在评估动态系统工程所需技能方面存在不足，因此提出了通过面向自动化的沙盒游戏（如 [Factorio](https://arxiv.org/abs/2502.01492)）训练 Agent。
   - 这有助于培养管理复杂工程挑战所必需的推理和长程规划能力。
- **Deep Research 抽象**：一位成员询问是否计划引入能够简化类似于 **deep research** 任务的抽象。
   - “你们计划引入抽象吗？”该成员问道，表达了对未来潜在功能的关注。
- **DSPy 客户端错误风波**：一位成员报告在使用 **dspy** 时遇到错误 `AttributeError: module 'dspy' has no attribute 'HFClientVLLM'`。
   - 他们随后注意到该功能在 **dspy 2.6** 中已**弃用 (deprecated)**，从而解决了困惑。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Llama 的自定义 RAFT 模板？**：一位成员询问是否可以使用类似于 **RAFT** 的自定义模板来为 **Llama** 生成合成数据集。
   - 这一询问引发了关于 **Llama** 数据集要求的灵活性和自定义选项的问题。
- **与 HF Datasets 的兼容性问题**：一位成员对由于函数属性不同而导致的 **HF datasets** 潜在兼容性问题表示担忧。
   - 该成员建议将复杂对象转换为字符串，以简化数据集的加载和使用。
- **JSON lines 格式澄清**：一位成员澄清说 **JSON** 文件没有问题，并指出 HF 期望的是 JSON lines 格式的文件。
   - 这一澄清强调了遵循预期文件格式对于在 **HF** 中成功加载数据集的重要性。
- **README 更新提议**：一位成员提议创建一个 Pull Request (**PR**) 来更新 **README**，增加一个新的辅助函数。
   - 该建议受到了好评，表明了改进用户体验和文档的协作方式。

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# 第 2 部分：分频道详细摘要和链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1337513705071443979)** (1052 条消息🔥🔥🔥): 

> `Unsloth AI 进展、GRPO 挑战、ASCII 艺术生成、奖励函数验证、多 GPU 支持` 


- **Unsloth AI 受欢迎程度上升**：Unsloth 已成为 GitHub 上排名第一的热门仓库，标志着其成立一年内的重大进展。
   - 社区对 Unsloth 提供的工具和资源表示赞赏，并对其未来的发展充满热情。
- **GRPO 和奖励函数问题**：用户报告了 GRPO 在有效评估非确定性输出方面的挑战，特别是在 RPG 角色扮演等创意任务中。
   - 讨论强调了奖励函数的重要性，并提出了调试建议以提高输出质量。
- **探索 ASCII 艺术生成**：出现了对微调模型以根据描述生成 ASCII 艺术的兴趣，但也有人对模型的局限性和连贯性表示担忧。
   - 参与者鼓励探索现有模型，以寻找生成 ASCII 艺术的潜在灵感和成功案例。
- **AI 训练中的验证挑战**：对话强调了验证使用 RL 训练的模型输出的困难，特别是当不存在固定的输出进行比较时。
   - 有人建议利用已知的幻觉可能为生成更好的数据集提供起点。
- **Unsloth 中的多 GPU 支持**：Unsloth 社区对多 GPU 支持的未来实现感到好奇，目前该功能正面向部分受信任成员进行 Beta 测试。
   - 用户对未来的更新和更广泛的访问权限表示关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<a href="https://medium.com/@alexhe.amd/deploy-deepseek-r1-in-one-gpu-amd-instinct-mi300x-7a9abeb85f78">在单张 GPU 上部署 Deepseek-R1 — AMD Instinct™ MI300X</a>: 是的！</li><li><a href="https://arxiv.org/abs/2411.10440">LLaVA-CoT: 让视觉语言模型逐步推理</a>: 大语言模型在推理能力方面取得了实质性进展，特别是通过推理时间扩展（inference-time scaling），正如 OpenAI 的 o1 等模型所示。然而，目前...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://arxiv.org/abs/2501.17161">SFT 记忆，RL 泛化：基础模型后训练的比较研究</a>: 监督微调（SFT）和强化学习（RL）是基础模型广泛使用的后训练技术。然而，它们在增强模型泛化能力方面的作用仍然...</li><li><a href="https://x.com/BlinkDL_AI/status/1888637497443504524">来自 BlinkDL (@BlinkDL_AI) 的推文</a>: https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k.pth</li><li><a href="https://unsloth.ai/blog/reintroducing">重新介绍 Unsloth</a>: 为庆祝我们成为当日 GitHub 趋势榜第一的仓库，我们回顾了我们的历程以及对开源社区的贡献。</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1ik3nkr/p_grpo_fits_in_8gb_vram_deepseek_r1s_zeros_recipe/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://unsloth.ai/newsletter">Unsloth 新闻简报</a>: 加入我们的新闻简报和候补名单，获取关于 Unsloth 的一切！</li><li><a href="https://docs.unsloth.ai/basics/errors">错误 | Unsloth 文档</a>: 要修复设置中的任何错误，请参阅下文：</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>: Unsloth 的动态 4-bit 量化选择性地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 占用的同时，大大提高了准确性。</li><li><a href="https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md">ktransformers/doc/en/DeepseekR1_V3_tutorial.md 在 main 分支 · kvcache-ai/ktransformers</a>: 一个用于体验前沿 LLM 推理优化的灵活框架 - kvcache-ai/ktransformers</li><li><a href="https://huggingface.co/Mistral-AI-Game-Jam">Mistral-AI-Game-Jam (Mistral AI 游戏创作大赛)</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1889000210371932398">来自 Unsloth AI (@UnslothAI) 的推文</a>: Unsloth 是 GitHub 趋势榜第一的仓库！🦥 这是一段不可思议的旅程，没有你们我们无法做到！为了庆祝，我们回顾了这一切是如何开始的，以及我们是如何走到今天的：...</li><li><a href="https://github.com/fzyzcjy/unsloth-zoo/commit/d2372ca5fc3bcb3ea41b6473c3b7d36de5265c55">更新 peft_utils.py · fzyzcjy/unsloth-zoo@d2372ca</a>: 未找到描述</li><li><a href="https://gitingest.com/">Gitingest</a>: 在任何 GitHub URL 中将 'hub' 替换为 'ingest'，即可获得对 prompt 友好的文本。</li><li><a href="https://docs.unsloth.ai/basics/er">Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1485#issuecomm">[已修复] `Qwen2VL` 微调损坏 · Issue #1485 · unslothai/unsloth</a>: 安装最新版本的 unsloth：`!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git` 会导致 Qwen2 7B Vision Colab 运行中断...</li><li><a href="https://github.com/TruffleClock/nano-r1">GitHub - TruffleClock/nano-r1</a>: 通过在 GitHub 上创建账号，为 TruffleClock/nano-r1 的开发做出贡献。</li><li><a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal">GitHub - EvolvingLMMs-Lab/open-r1-multimodal: 为 open-r1 添加多模态模型训练的分支</a>: 为 open-r1 添加多模态模型训练的分支 - EvolvingLMMs-Lab/open-r1-multimodal</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: 一个用于 LLM 的高吞吐量且显存高效的推理与服务引擎</a>: 一个用于 LLM 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/Zyphra/Zonos">GitHub - Zyphra/Zonos</a>: 通过在 GitHub 上创建账号，为 Zyphra/Zonos 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 及推理 LLM</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 及推理 LLM

更少显存 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1485#issuecomment-2628795809">[已修复] `Qwen2VL` 微调损坏 · Issue #1485 · unslothai/unsloth</a>：安装最新版本的 unsloth：!pip uninstall unsloth -y &amp;&amp; pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git 会破坏 Qwen2 7B Vision Colab ...</li><li><a href="https://github.com/unslothai/unsloth/issues/1613">Qwen2.5-VL-3B 4Bit 训练，'requires_grad_' 错误 · Issue #1613 · unslothai/unsloth</a>：你好！我正尝试在 google colab 上使用 colab 文件 https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7... 对 qwen2.5vl (unsloth/Qwen2.5-VL-3B-Instruct) 模型进行 SFT 微调</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">大语言模型（LLM）缺乏可靠医学推理所需的基本元认知 - Nature Communications</a>：大语言模型在医学考试中展示了专家级的准确性，支持了将其纳入医疗保健环境的潜力。在这里，作者揭示了它们的元认知能力尚不...</li><li><a href="https://www.fabfilter.com/">FabFilter - 用于混音、母带制作和录制的高质量音频插件 - VST VST3 AU CLAP AAX AudioSuite</a>：未找到描述</li><li><a href="https://build.nvidia.com/nvidia/digital-humans-for-customer-service/blueprintcard">由 NVIDIA 构建数字人蓝图 | NVIDIA NIM</a>：为各行各业的客户服务创建智能、互动的化身</li><li><a href="https://github.com/unslothai/unsloth/pull/1289#issuecomment-2646547748">由 shashikanth-a 添加了对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>：未优化。尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes：cmake -DCOMPUTE_BACKEND=mps -S . 用于构建 bitsandbytes；pip install unsloth-zoo==2024.11.4；pip install xformers==0.0.25</li><li><a href="https://github.com/trending">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.max_grad_norm">Trainer</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1337754746227789834)** (10 messages🔥): 

> `使用 REINFORCE 的推理 LLM，Deepseek 合作，针对 LLM 的 ReMax RL 训练，模型合并挑战` 


- **对 REINFORCE 方法的担忧**：一位成员分享了关于**使用 REINFORCE 的推理 LLM** 的[文档链接](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0)，引发了对其原创性的辩论。
   - 其他人质疑它是否提供了任何新颖之处，其中一人指出 **Unsloth** 中已经存在相同的实现。
- **关于 Deepseek 可用性的讨论**：一位成员询问了关于 **Deepseek** 的合作情况，不确定其是否在 Hugging Face 上提供。
   - 回复指出，其部分实现可能已经集成或在现有项目中可用。
- **ReMax 框架受到关注**：一条推文强调了 **ReMax** 的发布，这是一个用于 **LLM RL 训练** 的框架，具有比 PPO 更高的吞吐量等特点。
   - 成员们讨论了其 [GitHub 代码](https://github.com/liziniu/verl/tree/feature/add-remax-support/examples/remax_trainer)，强调了其在训练中的稳定性。
- **对模型合并的怀疑**：一位用户表示有兴趣了解将多个有效模型合并为单个混合专家模型（MoE）的挑战。
   - 这一请求引发了关于模型合并策略相关的潜在陷阱和局限性的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/ziniuli/status/1888576228619370525?s=46">来自 Ziniu Li (@ZiniuLi) 的推文</a>：🚀 高效的 LLM RL 训练：基于 Verl 分布式框架构建的 ReMax 现已发布！🛠️🔑 关键特性：- 比 PPO 更高的吞吐量 - 具有理论方差保证的稳定训练...</li><li><a href="https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融合为一体的新工具。它是为您和您的团队打造的一体化工作区。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1337514752464916636)** (440 messages🔥🔥🔥): 

> `微调模型，处理 OOM 错误，使用量化模型，训练中的奖励函数，模型评估`

- **Fine-tuning 与评估中的挑战**：用户报告了在微调 Qwen2 VL 模型时遇到的问题，指出尽管训练损失（training losses）在下降，但他们的调整对输出准确率几乎没有影响。
   - 几位用户对寻找有效方法表示沮丧，一些人认为尽管应用了 LoRA 技术，他们的模型仍未见改善。
- **处理 OOM 错误**：多位成员在尝试加载和微调模型时遇到了显存溢出（OOM）错误，强调了管理模型复杂性和资源分配的必要性。
   - 建议包括清理 checkpoints 并确保配置与可用硬件相匹配，特别是在使用较大模型时。
- **量化模型与性能**：围绕使用 4-bit 量化模型的讨论突出了潜在的性能优势，但也存在对其实现（特别是量化后输出）的困惑。
   - 用户指出，虽然动态量化可以提高效率，但有时会导致意外的输出变化，并询问如何在模型大小和结果一致性之间取得平衡。
- **理解训练中的奖励函数（Reward Functions）**：参与者强调了定制奖励函数以鼓励预期输出的重要性，特别是在输入复杂度各异的任务中。
   - 大家对奖励机制如何优化模型响应表现出浓厚兴趣，特别是当数据集包含科学数据提取等细微示例时。
- **构建支持多 GPU 的模型**：一位用户询问了在 Unsloth 中跨多个 GPU 训练模型的问题，强调了目前的局限性以及缺乏直接的多 GPU 支持。
   - 建议指向了未来可能的增强功能，但指出目前仍需要单 GPU 训练设置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C,">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth 文档</a>: 关于使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1tiQrc6LVOxdRDWsM5WMuLrYhPkUVt617?usp=sharing#scrollTo=ZDXE1V-MNtAG">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 微调 | Unsloth 文档</a>: Checkpointing 允许你保存微调进度，以便暂停并继续。</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">数据集基础 101 | Unsloth 文档</a>: 学习创建微调数据集的所有要点！</li><li><a href="https://huggingface.co/yukiarimo/yuna-ai-v3-full/tree/main">yukiarimo/yuna-ai-v3-full at main</a>: 未找到描述</li><li><a href="https://github.com/simples">simples - 概览</a>: GitHub 是 simples 构建软件的地方。</li><li><a href="https://github.com/unslothai/unsloth/issues/1613">Qwen2.5-VL-3B 4Bit 训练，'requires_grad_' 错误 · Issue #1613 · unslothai/unsloth</a>: 你好！我正尝试在 Google Colab 上使用 colab 文件 https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7... 对 qwen2.5vl (unsloth/Qwen2.5-VL-3B-Instruct) 模型进行 SFT</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B">unsloth/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/blob/main/README.md#customize-a-prompt">ollama/README.md at main · ollama/ollama</a>: 快速上手 Llama 3.3、DeepSeek-R1、Phi-4、Gemma 2 以及其他大语言模型 (LLM)。 - ollama/ollama</li><li><a href="https://github.com/unslothai/unsloth/issues/1208#issuecomment-2444633537">在 vLLM 中运行合并后的 4-bit Mistral Nemo 时出现错误 `KeyError: 'layers.0.mlp.down_proj.weight'` · Issue #1208 · unslothai/unsloth</a>: 我正尝试微调 Mistral Nemo，将其保存为 4-bit 合并版本，并在 vLLM 中运行。训练过程中没有遇到任何错误。但是，当我尝试使用 vLLM 提供模型服务时...</li><li><a href="https://www.youtube.com/watch?v=JJWvYQdOVOY"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/simplescaling/s1/blob/main/eval/generate.py#L57-L72">s1/eval/generate.py at main · simplescaling/s1</a>: s1: 简单的测试时缩放 (test-time scaling)。通过在 GitHub 上创建账号来为 simplescaling/s1 的开发做出贡献。</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py">finetune-Qwen2-VL/finetune.py at main · zhangfaen/finetune-Qwen2-VL</a>: 通过在 GitHub 上创建账号来为 zhangfaen/finetune-Qwen2-VL 的开发做出贡献。</li><li><a href="https://github.com/huggingface/trl.git">GitHub - huggingface/trl: 使用强化学习训练 Transformer 语言模型。</a>: 使用强化学习训练 Transformer 语言模型。 - huggingface/trl</li><li><a href="https://github.com">GitHub · 在统一的协作平台上构建和发布软件</a>: 加入全球应用最广泛的 AI 驱动开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://colab.research.google.com/drive/1tiQrc6LVO">Google Colab</a>: 未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1337886405224960124)** (2 条消息): 

> `Llama 3.2 模型解析，Spark Engine v1 发布，无代码 AI 工具，AI 模型集成` 


- **Llama 3.2 模型的详细解析**：一位用户将 **Llama 3.2 1B 模型** 重新拆解为其权重，提供了对其架构的深入见解，可在 [GitHub](https://github.com/SoumilB7/Llama3.2_1B_pytorch_barebones) 上进行探索。
   - 欢迎对该解析提出任何改进建议，以增强其有效性。
- **Spark Engine v1 发布会**：经过一年多的公开测试，**Spark Engine v1** 的最新版本已发布，拥有超过 **80 个 AI 模型**，能够生成包括 *文本、音乐和视频* 在内的各种内容类型。
   - 用户表达了将更多基础设施（如 **Unsloth**）集成到 Spark Engine 平台的愿望，以推动无代码 AI 领域的进一步发展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sparkengine.ai/">Spark Engine - AI 沙盒</a>：将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://github.com/SoumilB7/Llama3.2_1B_pytorch_barebones">GitHub - SoumilB7/Llama3.2_1B_pytorch_barebones</a>：Llama 3.2 1B 架构基础实现 + 智慧结晶的 Pytorch 实现
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1337594689674547294)** (42 条消息🔥): 

> `数据集策展的重要性，元推理（Meta Reasoning）与 Lora 微调，优化推理方法，长输出格式的挑战，集成更新的语言规范` 


- **数据集策展是模型成功的关键**：会议强调了任何模型 **80%** 的性能都依赖于细致的 **数据集策展（Dataset Curation）**，突出了其在训练中的关键作用。
   - *一位成员指出：*“不存在冗余的研究——你可以从每一篇论文中学习”，这体现了对持续学习的承诺。
- **利用 Lora 探索元推理**：一位成员正在尝试 Lora 设置，以开发一种元认知的第一人称推理格式，旨在提高推理能力。
   - 他们计划广泛测试其模型设置，并确保在保留基础知识和增强推理能力之间取得平衡。
- **优化推理方法以提升性能**：在 Unsloth 上进行的推理测试表明，与其它方法相比，它在多 GPU 上使用 bfloat16 时速度更快且更准确。
   - 成员们讨论了可以在保持模型准确性的同时，提高在低配硬件上性能的优化方案。
- **微调中长输出格式的挑战**：有人担心在具有共同结构的超长输出格式中可能会丢失学习效果，这可能会阻碍特定任务的训练。
   - 一位成员提到，Token 损失（token loss）将允许模型专注于动态部分，这将有助于在长提示词的情况下学习目标任务。
- **将更新的语言规范集成到模型中**：一位用户寻求关于解析 Janet 更新语言规范的建议，以增强 Phi4 的能力，因为其知识截止日期为 2021 年。
   - 挑战在于如何提取当前资源，以根据语言的新更新和特性来改进模型的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/a/qoZHqis">imgur.com</a>：在 Imgur 发现互联网的魔力。</li><li><a href="https://arxiv.org/abs/2502.04128">Llasa: 为基于 Llama 的语音合成扩展训练时与推理时计算</a>：基于文本的大语言模型（LLMs）的最新进展，特别是 GPT 系列和 o1 模型，已经证明了扩展训练时和推理时计算的有效性……</li><li><a href="https://arxiv.org/html/2411.00856v1">投资分析中的 AI：用于股票评级的 LLMs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1337522439743340706)** (252 条消息🔥🔥): 

> `AI Agents 课程，模型优化，心理健康聊天机器人推荐，数据隐私与安全，AI 演示文稿生成方案`

- **关于 AI Agents 课程的讨论**：多位用户确认参加即将开始的 AI Agents 课程，部分用户对课程内容表示期待。
   - 有人询问了课程材料和测验，以便为即将到来的课程做准备。
- **探索更小模型以提高效率**：用户讨论了选择更小模型的重要性，以便在保持质量的同时优化性能并降低硬件需求。
   - 建议对多个模型进行对比测试，并采用 batching 请求等技术来提高效率。
- **心理健康聊天机器人的建议**：推荐了来自 Hugging Face 的多种模型（包括 Aloe 和 OpenBioLLM）用于心理健康应用，强调了它们的潜在有效性。
   - 对话涉及在健康科技等敏感领域使用 AI 工具时，通过本地推理（local inference）来维护数据隐私。
- **数据隐私与云端担忧**：讨论了将受保护的患者数据发送到云服务的后果，以及对本地处理的需求。
   - 建议采用数据匿名化或使用更小模型等多种策略，以在执行 AI 任务时缓解隐私担忧。
- **用于演示文稿和 UI 生成的 AI**：一位用户寻求自动生成 PowerPoint 演示文稿以及将 Figma 设计转换为代码的解决方案。
   - 他们寻求能够简化这些流程的工具，利用 AI 提高效率并减少重复性的手动工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>: 未找到描述</li><li><a href="https://livebench.ai">LiveBench</a>: 未找到描述</li><li><a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: 将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://scalingintelligence.stanford.edu/pubs/">Publications</a>: 未找到描述</li><li><a href="https://cs229s.stanford.edu/fall2024/calendar/">Calendar</a>: 课程模块和主题列表。</li><li><a href="https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html">torch.cuda.manual_seed &mdash; PyTorch 2.6 documentation</a>: 未找到描述</li><li><a href="https://tenor.com/view/remygag-remy-gag-barf-rat-gif-978058593911536854">Remygag Remy GIF - RemyGag Remy Gag - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/thrishala/mental_health_chatbot?library=transformers">thrishala/mental_health_chatbot · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15">VideoLLaMA3 - a DAMO-NLP-SG Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Math">Math - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Nemo-Mistral-Minitron">Nemotron-Mini - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#community-gpu-grants">Using GPU Spaces</a>: 未找到描述</li><li><a href="https://tenor.com/view/samurai-japan-windy-sword-birds-gif-17444000">Samurai Japan GIF - Samurai Japan Windy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/facebook/bart-large-cnn">facebook/bart-large-cnn · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ijab77/train_your_own_reasoning_model_80_less_vram_grpo/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://pytorch.org/get-started/locally/">Start Locally</a>: 本地启动</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/discussions/20">stabilityai/stable-diffusion-xl-refiner-1.0 · Only receiving black images?</a>: 未找到描述</li><li><a href="https://forums.developer.nvidia.com/t/fp16-support-on-gtx-1060-and-1080/53256">FP16 support on gtx 1060 and 1080</a>: 大家好，我是 TensorRT 的新手。我正尝试在配备 GTX 1060 的开发电脑上使用 TensorRT。在使用我的 C++ 程序（基于示例程序设计）优化我的 caffe 网络时...</li><li><a href="https://huggingface.co/datasets/Tonic/MiniF2F">Tonic/MiniF2F · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bobpopboom/mentalchat2">Mentalchat2 - a Hugging Face Space by bobpopboom</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bobpopboom/testing">Testing - a Hugging Face Space by bobpopboom</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=g74Cq9Ip2ik">Master AI image generation - ComfyUI full tutorial 2024</a>: ComfyUI 完整安装与教程。终极图像生成器。文本生成图像、图像生成图像、换脸、ControlNet、放大、外部插件等...</li><li><a href="https://deepai.org/">DeepAI</a>: 为富有创造力的人类提供的人工智能工具。</li><li><a href="https://huggingface.co/datasets/mzbac/function-calling-llama-3-format-v1.1">mzbac/function-calling-llama-3-format-v1.1 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/V_xro1bcAuA?si=M7M7L9oZi5b6Nl5o"> - YouTube</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-game-awards-matan-matan-evenoff-bill-clinton-gif-27233593">The Game Awards Matan GIF - The Game Awards Matan Matan Evenoff - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372#67a47cb63dd995045efdad4f">huggingchat/chat-ui · [MODELS] Discussion</a>: 未找到描述
</li>
</ul>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1338121555611353212)** (4 messages): 

> `Hugging Face in healthcare, Adaptive tuning of KL loss in VAE training, Classifying scientific abstracts, Gradio front end for AI agent, Model checkpointing` 


- **探索 Hugging Face 在医疗保健领域的应用**：一名成员正在深入研究如何在医疗保健领域使用 **Hugging Face**，表现出对实际应用的浓厚兴趣。
   - 他们表达了想了解是否还有其他人也在研究同一主题的愿望。
- **自适应 KL Loss 调整取得显著成效**：另一位成员分享了在 **VAE** 训练期间使用多种方法自适应调整 **KL loss** 的见解，并报告了有趣的结果。
   - 他们指出，在训练过程中将权重降至 0 以防止坍缩（collapse）取得了成功。
- **科学出版物分类**：一位用户正在创建一个模型，利用其 **private data**（私有数据）将科学出版物的摘要分为两个不同的类别。
   - 他们强调其项目重点在于分类器（classifiers）。
- **为 AI 监控构建 Gradio 前端**：一位成员正在学习如何为一个监控代码执行的 **AI agent** 创建 **Gradio 前端**，并分享了具体的模型名称。
   - 他们上传了多张与工作相关的图片，展示了在构建此应用程序方面的进展。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1337612449498791997)** (6 messages): 

> `Reasoning Language Models, Writing-in-the-Margins bounty, Physical Perceptrons, Markdrop Python Package` 


- **探索推理语言模型 (Reasoning Language Models)**：分享了一个名为 [“From Large Language Models to Reasoning Language Models - Three Eras in The Age of Computation”](https://www.youtube.com/watch?v=NFwZi94S8qc) 的 YouTube 视频，深入探讨了 LLM 的演变及其对计算的影响。
   - 讨论强调了通过不同计算视角进行的变革之旅。
- **Writing-in-the-Margins 的 5000 美元悬赏**：一位成员指出，在 vllm 中实现 Writing-in-the-Margins 推理模式可获得 [$5,000 悬赏](https://github.com/vllm-project/vllm/issues/9807)。
   - 该功能旨在增强长上下文窗口（long context window）检索的结果，并提供了详细的动机说明。
- **寻找物理感知机 (Physical Perceptrons)**：一位成员询问是否有人见过物理实体形式的 [Perceptron](https://www.youtube.com/watch?v=l-9ALe3U-Fg)，并引用了一个 YouTube 视频作为背景。
   - 这一询问引发了关于 AI 历史上感知机的构造及其重要性的讨论。
- **Markdrop 的 PDF 转换能力**：介绍了一个名为 [Markdrop](https://github.com/shoryasethia/markdrop) 的新 Python 包，用于将 PDF 转换为 Markdown，包括图像和表格提取功能。
   - 它提供了高级功能，如 AI 驱动的内容分析、自动图像提取和交互式 HTML 输出。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=NFwZi94S8qc">From Large Language Models to Reasoning Language Models - Three Eras in The Age of Computation.</a>: 在本次演讲中，我们将通过计算和选择的视角，探索大型语言模型 (LLMs) 的迷人演变及其变革之旅...</li><li><a href="https://github.com/shoryasethia/markdrop">GitHub - shoryasethia/markdrop: 一个用于将 PDF 转换为 Markdown 同时提取图像和表格的 Python 包，使用多个 LLM 客户端为提取的表格/图像生成描述性文本。还有更多功能。Markdrop 已在 PyPI 上线。</a>: 一个用于将 PDF 转换为 Markdown 同时提取图像和表格的 Python 包，使用多个 LLM 客户端为提取的表格/图像生成描述性文本。还有更多功能...</li><li><a href="https://pypi.org/project/markdrop/">markdrop</a>: 一个全面的 PDF 处理工具包，可将 PDF 转换为 Markdown，具有先进的 AI 驱动的图像和表格分析功能。支持本地文件和 URL，保留文档结构，提取...</li><li><a href="https://github.com/vllm-project/vllm/issues/9807">[Feature]: 集成 Writing in the Margins 推理模式 ($5,000 悬赏) · Issue #9807 · vllm-project/vllm</a>: 🚀 功能、动机和推介：Writer 引入了 &quot;Writing in the Margins&quot; 算法 (WiM)，可提升长上下文窗口检索的结果。该任务由 &quot;con...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1337538032466661396)** (16 messages🔥): 

> `Kokoro TTS integration, Dataset Tools update, Spark Engine launch, Markdrop PDF tool, go-attention implementation`

- **Kokoro TTS 现已通过 C# 库开源**：一位成员宣布发布了 **Kokoro TTS** 的 C# 库，支持在 .NET 平台上的即插即用集成，可在 [GitHub](https://github.com/Lyrcaxis/KokoroSharp) 上获取。该库支持快速的本地 TTS 推理，并可跨多个平台运行。
   - 该库承诺提供**多语言**体验，所有语音都以便捷的格式打包。
- **Dataset Tools 针对 EXIF 和 AI 元数据进行了更新**：Dataset 管理器和 **EXIF Viewer** 获得了更新，增强了查看高级 EXIF 数据的功能，并支持 GGUF 和 JPEG 等格式，已在 [GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools) 上分享。
   - 开发者利用 AI 工具辅助项目，在与他人协作进行代码优化的同时增强了其功能。
- **Spark Engine v1 正式发布**：Spark Engine v1 在经过为期一年的公开测试后正式发布，在 [sparkengine.ai](https://sparkengine.ai/) 上提供**超过 80 个模型**，用于各种 AI 任务。
   - 该平台每天提供免费额度，并与 Hugging Face 集成，为用户提供了一个强大的无代码环境来实验 AI 功能。
- **Markdrop 提供高级 PDF 转 Markdown 功能**：介绍了一个名为 **Markdrop** 的新 Python 包，旨在将 PDF 转换为 Markdown，具有图像提取和 AI 驱动的描述等功能，可在 [GitHub](https://github.com/shoryasethia/markdrop) 上访问。
   - 在短短一个月内，它的安装量已超过 **7,000 次**，展示了它在寻找文档处理工具的用户中的受欢迎程度。
- **Transformer 的创新 go-attention 实现**：一位成员分享了他们的项目 **go-attention**，该项目展示了第一个用纯 Go 语言构建的完整 Attention 机制和 Transformer，并在 [GitHub](https://github.com/takara-ai/go-attention) 上突出了其独特的功能。
   - 该项目邀请其他人查看示例，并探索 Go 编程中 Serverless 实现的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sparkengine.ai/">Spark Engine - The AI Sandbox</a>: 将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://huggingface.co/spaces/elismasilva/mixture-of-diffusers-sdxl-tiling">Mixture Of Diffusers SDXL Tiling - a Hugging Face Space by elismasilva</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/jeremyadd/mini_datathon">Mini Datathon - a Hugging Face Space by jeremyadd</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/Duskfallcrew/design-101">Design 101: A Historical and Theoretical Exploration of Graphic Arts and Design in the Age of AI</a>: 未找到描述</li><li><a href="https://github.com/shoryasethia/markdrop">GitHub - shoryasethia/markdrop: A Python package for converting PDFs to markdown while extracting images and tables, generate descriptive text descriptions for extracted tables/images using several LLM clients. And many more functionalities. Markdrop is available on PyPI.</a>: 一个用于将 PDF 转换为 Markdown 的 Python 包，同时提取图像和表格，并使用多个 LLM 客户端为提取的表格/图像生成描述性文本。还有更多功能。Markdrop 已在 PyPI 上发布。</li><li><a href="https://pypi.org/project/markdrop">no title found</a>: 未找到描述</li><li><a href="https://github.com/Lyrcaxis/KokoroSharp">GitHub - Lyrcaxis/KokoroSharp: Fast local TTS inference engine with ONNX runtime. Multi-speaker, multi-platform and multilingual.  Integrate on your .NET projects using a plug-and-play NuGet package, complete with all voices.</a>: 使用 ONNX runtime 的快速本地 TTS 推理引擎。多说话人、多平台且多语言。使用即插即用的 NuGet 包集成到您的 .NET 项目中，包含所有语音。</li><li><a href="https://github.com/takara-ai/go-attention">GitHub - takara-ai/go-attention: A full attention mechanism and transformer in pure go.</a>: 纯 Go 语言实现的完整 Attention 机制和 Transformer。</li><li><a href="https://huggingface.co/datasets/Tonic/Climate-Guard-Toxic-Agent">Tonic/Climate-Guard-Toxic-Agent · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Ktiseos-Nyx/Dataset-Tools">GitHub - Ktiseos-Nyx/Dataset-Tools: A Simple Viewer for EXIF and AI Metadata</a>: 一个简单的 EXIF 和 AI 元数据查看器。</li><li><a href="https://github.com/duskfallcrew/Beetlejuice_Summoning">GitHub - duskfallcrew/Beetlejuice_Summoning: Literally just summons a youtube video after you say his name 3x spoken unbroken, and makes sure you enter one of the &quot;WHOLE BEING DEAD THING&quot; lyrics. It&#39;s untested, but i was using GPT to be a nerd.</a>: 只要你连续说三次他的名字，就会召唤出一个 YouTube 视频，并确保你输入了 "WHOLE BEING DEAD THING" 的歌词之一。未经测试，但我当时正用 GPT 耍极客范儿。</li><li><a href="https://tenor.com/view/baby-cute-go-cheering-rage-gif-16641949">Baby Cute GIF - Baby Cute Go - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1338186001553555526)** (15 条消息🔥): 

> `AI 阅读小组日程、角色管理、Women in AI & Robotics` 


- **二月 AI 阅读小组会议提醒**：**Women in AI & Robotics** 的下一场 **AI 阅读小组**会议定于本周四 **12pm EST / 5pm GMT** 举行。参与者可以通过[此处](https://discord.com/events/879548962464493619/1331369988560523447)的阅读小组语音频道加入直播。
   - 据悉，这些会议大约**每月举行一次**，论文作者有机会进行展示。
- **角色移除协助**：一名成员询问如何移除角色，并获知该角色可能已被移除或正在进行**大修（overhaul）**。另一名成员迅速协助道：*“我已经为你移除了该角色。”*
   - 原成员简短地表达了感谢：*“tysm。”*
- **参与意向**：成员们表达了如果时间允许，有兴趣参加 **AI 阅读小组**，并强调了对 **deep tech engineering** 等更深层次话题的参与。一位成员对可能参加未来的会议表示了热情。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1337518194276958289)** (6 条消息): 

> `制造业中的 Computer Vision, 屏幕截图分析模型, 用于多模态模型的 Roboflow Maestro, BLIP 微调脚本, 用于图像分类的 AutoML` 


- **Computer Vision 增强制造业检测**：一位成员正在尝试在制造业环境中使用 **computer vision** 来检查产品，并分析视觉特征和生产日志。
   - 这种方法旨在通过有效融合视觉和文本数据来确保质量。
- **浏览器屏幕截图解析的困扰**：一位用户表达了在寻找能准确解析浏览器屏幕截图视觉组件的 **computer vision model** 时的挫败感。
   - 尽管该领域最近取得了令人印象深刻的成果，但他们指出现有模型（特别是利用 *GPT-4V* 的模型）未能提供有效集成到其工作流中所需的细节。
- **探索用于模型微调的 Roboflow Maestro**：一位成员分享了 [Roboflow Maestro](https://github.com/roboflow/maestro) 的链接，作为简化 **PaliGemma 2** 和 **Florence-2** 等多模态模型微调过程的资源。
   - 该用户正考虑尝试它，但目前仍依赖于 **BLIP** 微调脚本。
- **对图像分类 AutoML 的兴趣**：一位成员询问是否有人尝试过用于图像分类任务的 **AutoML model**。
   - 这突显了人们对于简化针对特定 computer vision 需求的模型选择和训练过程的持续兴趣。



**提及的链接**: <a href="https://github.com/roboflow/maestro">GitHub - roboflow/maestro: streamline the fine-tuning process for multimodal models: PaliGemma 2, Florence-2, and Qwen2.5-VL</a>: 简化多模态模型的微调过程：PaliGemma 2, Florence-2 和 Qwen2.5-VL - roboflow/maestro

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1338202903512350806)** (12 条消息🔥): 

> `LLM 中的上下文长度问题, 使用 roBERTa 进行情感分类, Embedding 模型评估, Product Quantization 技术, PQ 中的 Coarse Quantization` 


- **寻求 LLM 上下文长度的解决方案**：一位成员表达了在寻找 LLM **context length problem** 实际解决方案时的挫败感，强调了对准确率随上下文长度增加而下降的担忧。
   - 他们正在寻找既能扩展上下文又不牺牲准确性的 **high-quality** 方法。
- **使用 Twitter roBERTa 进行情感分析**：一位用户分享了他们使用 [Twitter-RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) 进行情感分类的方法，并表示他们使用了 Hugging Face 的 classification pipeline。
   - 他们注意到模型的输出有时在所有情感类别中都会产生异常高的置信度分数。
- **Embedding 模型的评估技术**：一位成员就一篇提出统一评估方法的论文征求社区反馈，该方法独立于下游任务，专注于 embedding 模型。
   - 该论文旨在将理论基础与实际性能指标联系起来，以确保更好的评估标准。
- **关于 Product Quantization 的见解**：讨论围绕 **Product Quantization (PQ)** 技术展开，特别是其对利用 **word embeddings** 细微差别的影响，以及对量化过程中信息丢失的担忧。
   - 一位用户询问了压缩收益与 embedding 可能发生的含义改变之间的权衡。
- **理解 Coarse Quantization**：一位用户寻求关于 Product Quantization 背景下 **coarse quantization** 的澄清，并报告称难以找到关于该主题的充足材料。
   - 他们强调了对现有 AI 工具无法针对该概念提供充分回答的挫败感。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=VqFz7iTGcl">When is an Embedding Model  More Promising than Another?</a>: Embedders 在机器学习中起着核心作用，将任何对象投影到数值表示中，进而可以利用这些表示来执行各种下游任务。评估...</li><li><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment">cardiffnlp/twitter-roberta-base-sentiment · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/pipelines">Pipelines</a>: 无描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1337534758346817626)** (19 条消息🔥): 

> `课程注册问题、直播问答环节公告、证书通知流程、课程内容的 GitHub Pull Request、YouTube 课程介绍` 


- **已注册但未收到更新**：多位成员（包括 *jyotip2217* 和 *pierre2452*）对注册课程后未收到更新表示担忧。
   - 这引发了关于已报名参与者沟通流程的问题。
- **直播问答定于 2 月 12 日**：一位成员提供了 [YouTube 视频](https://www.youtube.com/watch?v=PopqUt3MGyQ) 链接，详细介绍了课程简介以及即将于 2 月 12 日下午 5 点举行的直播问答环节。
   - 参与者被告知该环节将涵盖课程后勤安排并提供提问平台。
- **证书通知流程需要澄清**：一位参与者询问在使用 Hugging Face 账号报名后，是否需要特定的通知流程来获取证书。
   - 这种不确定性表明在课程参与和证书预期方面的教学说明可能存在缺失。
- **课程内容的 GitHub 协作**：成员 *burtenshaw* 分享了一个 [GitHub Pull Request](https://github.com/huggingface/course/pull/777)，旨在将 smol course 的内容（包括交互式测验）迁移到 NLP 课程中。
   - 该努力旨在通过整合更多互动材料来增强课程，并开放协作。
- **YouTube 课程介绍视频**：分享的名为 'Welcome To The Agents Course' 的 YouTube 视频介绍了课程结构和范围，作为新参与者的资源。
   - 该视频旨在明确即将到来的课程里程碑，帮助个人有效地度过初始阶段。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/introduction">Agents 简介 - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=PopqUt3MGyQ">欢迎来到 Agents 课程！课程介绍与问答</a>: 在 Agents 课程的首次直播中，我们将解释课程的运作方式（范围、单元、挑战等）并回答您的问题。不要...</li><li><a href="https://github.com/huggingface/course/pull/777">[章节] 由 burtenshaw 基于 smol course 编写的监督微调新章节 · Pull Request #777 · huggingface/course</a>: 这是一个供讨论的草案 PR。如果能在 HF NLP 课程中复用 smol course 关于 SFT 的章节就太酷了。这里我只是复制了内容，但下一步我建议....
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1337785854990614529)** (704 条消息🔥🔥🔥): 

> `课程介绍、学员交流、Python 知识要求、AI Agents 讨论、国际参与度` 


- **课程介绍与预期**：参与者对 AI Agents 课程表示兴奋并讨论其启动情况，许多人询问课程访问权限和要求。
   - 大家渴望开始学习和协作，并提到了课程对实践知识的关注。
- **学员间的社交网络**：用户们正在介绍自己及其所在地，在来自不同国家的参与者之间建立社区感。
   - 许多人表示有兴趣基于在 AI 及相关领域的共同经验进行联系。
- **Python 知识要求**：一些参与者询问课程所需的 Python 知识水平，反映出编程背景各异。
   - 仅具备基础 Python 技能的用户表示担忧，寻求关于能否跟上进度的保证。
- **国际参与度**：该频道汇集了来自印度、美国、法国和巴西等多个国家的多元化参与者。
   - 参与者对共同学习感到兴奋，并赞赏课程的全球化性质。
- **技术访问问题**：一些用户报告在注册后遇到账号验证和访问 Discord 课程频道的问题。
   - 关于解决这些问题的讨论正在进行中，用户们分享了有关验证的经验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/021destiny">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://www.kaggle.com/whitepaper-agents">Agents</a>: 作者：Julia Wiesinger, Patrick Marlow 和 Vladimir Vuskovic</li><li><a href="https://learn.deeplearning.ai/courses/ai-python-for-beginners/lesson/1/introduction">面向初学者的 AI Python：AI Python 编程基础 - DeepLearning.AI</a>: 在 AI 辅助下学习 Python 编程。掌握高效编写、测试和调试代码的技能，并创建真实的 AI 应用。</li><li><a href="https://tenor.com/view/seattle-space-gif-18175495">西雅图太空 GIF - 西雅图太空 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lilianweng.github.io/posts/2023-06-23-agent/">LLM 驱动的自主 Agents</a>: 以 LLM (大语言模型) 作为核心控制器来构建 Agents 是一个很酷的概念。一些概念验证演示，如 AutoGPT, GPT-Engineer 和 BabyAGI，都是极具启发性的例子。...</li><li><a href="https://tenor.com/view/tijuana-gif-21081556">蒂华纳 GIF - 蒂华纳 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://amaykataria.com">Amay Kataria 3.0</a>: 未找到描述</li><li><a href="https://tenor.com/view/hello-hi-hy-hey-gif-8520159980767013609">你好 GIF - 你好 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/napoleon-dynamite-wave-bye-gif-15387504">拿破仑炸药挥手 GIF - 拿破仑炸药挥手告别 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/%C4%B1taly-italya-italiana-roma-ferrar%C4%B1-gif-13413841744894640022">意大利 GIF - 意大利 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huyenchip.com/2025/01/07/agents.html">Agents</a>: 智能 Agents 被许多人认为是 AI 的终极目标。Stuart Russell 和 Peter Norvig 的经典著作《人工智能：一种现代方法》(Prentice Hall, 1995) 定义了...</li><li><a href="https://tenor.com/view/greetings-chat-chrissss-gridman-dante-devil-may-cry-dante-from-devil-may-cry-gif-22479319">问候聊天 Chrissss Gridman GIF - 问候聊天 Chrissss Gridman Dante - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/agents-course">agents-course (Hugging Face Agents 课程)</a>: 未找到描述</li><li><a href="https://github.com/huggingface/agents-course">GitHub - huggingface/agents-course: 此仓库包含 Hugging Face Agents 课程。</a>: 此仓库包含 Hugging Face Agents 课程。 - GitHub - huggingface/agents-course: This repository contains the Hugging Face Agents Course.</li><li><a href="https://github.com/mohamedsheded/Agentic-design-patterns">GitHub - mohamedsheded/Agentic-design-patterns: 一个用于实现和理解 Agentic 工作流设计模式的仓库</a>: 一个用于实现和理解 Agentic 工作流设计模式的仓库 - mohamedsheded/Agentic-design-patterns</li><li><a href="https://x.com/horosin_.">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord, Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord, Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1337602857037332511)** (7 messages): 

> `Reasoning Datasets, Model Training, Distillations, Learning Math` 


- **探索推理数据集**：一位成员引导其他人查看[此处](https://huggingface.co/collections/open-r1/reasoning-datasets)提供的各种推理数据集，特别强调了 **Bespoke-Stratos-17k** 数据集。
   - 另一位成员表示感谢，指出这些信息**非常有帮助**。
- **尝试 R1 风格推理**：一位用户提到正在实验教模型 **R1 风格推理**以辅助数学学习，并表示附带了视觉演示。
   - 成员们讨论的焦点似乎在于简化推理过程。
- **关于结果质量的讨论**：一位成员建议完全移除 **<reasoning>** 组件，以评估结果质量的差异。
   - 这引发了一场关于此类变更影响的轻松对话。
- **训练需要耐心**：一位成员强调了漫长的训练时间，指出在 **4060ti** 上训练任何模型大约需要 **6 小时**，这让人对添加推理的功效产生了怀疑。
   - 尽管面临挑战，但在**数学学习方面取得了一些进展**，强调了对这一过程的投入。



**提到的链接**：<a href="https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678">🧠 Reasoning datasets - a open-r1 Collection</a>：未找到描述

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1337513596808200374)** (596 条消息🔥🔥🔥): 

> `Qwen Models, LM Studio Functionality, Embedding Models, Comparing AI Models, Using APIs` 


- **Qwen 2.5 vs Llama 8B**：用户讨论了 Qwen 2.5 和 Llama 8B 之间的性能差异，由于优化，Qwen 通常能提供更快的响应。
   - 有建议称，如果用户拥有运行 32B 等更大模型的必要硬件，Qwen 2.5 是更好的选择。
- **模型加载问题排查**：用户报告了将模型加载到 LM Studio 时的各种问题，并建议详细说明系统规格并提供截图以便更好地获得帮助。
   - 诸如 'NO LM Runtime found for model format' 之类的错误表明了潜在的硬件限制，强调了将模型大小与系统能力相匹配的重要性。
- **利用本地服务器**：有关于通过本地服务器访问 LM Studio 模型的查询，这需要连接到前端应用程序才能有效使用。
   - 建议包括使用兼容的 API，因为 LM Studio 不具备内置的 Web UI，这突显了外部集成的需求。
- **模型配置与性能**：讨论集中在调整 LM Studio 中的 Temperature 和 Batch Size 等设置，以根据可用的 RAM 和 VRAM 优化性能。
   - 用户被告知，配置调优对于从 AI 模型中获得理想结果至关重要，特别是对于计算密集型应用。
- **用于编码和项目的 AI**：关于使用 Qwen 模型进行编程任务的咨询引出了对 Mistral 等各种模型以及有效编码辅助替代方案的见解。
   - 对话强调，虽然存在强大的模型，但从较小的、易于管理的模型开始，可能会为初学者提供更好的学习体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://qwenlm.github.io/blog/">Blog</a>: Qwen</li><li><a href="https://tenor.com/view/trump-thug-life-gif-11298887">Trump Thug Life GIF - Trump Thug Life - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/basics/import-model">Import Models | LM Studio Docs</a>: 使用你在 LM Studio 之外下载的模型文件</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory">Download an LLM | LM Studio Docs</a>: 在 LM Studio 中发现并下载支持的 LLM</li><li><a href="https://tenor.com/view/rpx_syria-mic-drop-gif-19149907">Rpx_syria Mic Drop GIF - Rpx_syria Mic Drop - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/closedagi/gpt5-r3-claude4-magnum-dong-slerp-raceplay-dpo-4.5bpw-1.1T-exl2-rpcal">closedagi/gpt5-r3-claude4-magnum-dong-slerp-raceplay-dpo-4.5bpw-1.1T-exl2-rpcal · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/stanley-hudson-the-office-annoyed-gif-20544770">Stanley Hudson GIF - Stanley Hudson The - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/everythings-under-control-happily-the-situation-is-under-control-there-is-control-over-everything-gif-26314021">Everythings Under Control Happily GIF - Everythings Under Control Happily The Situation Is Under Control - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/ishan-marikar/lm-studio-ollama-bridge">GitHub - ishan-marikar/lm-studio-ollama-bridge: lm-studio-ollama-bridge 是一个独立的基于 Go 的工具，可将你的本地 Ollama 模型与 LM Studio 等外部应用程序同步。灵感来自原始的 matts-shell-scripts/syncmodels 项目</a>: lm-studio-ollama-bridge 是一个独立的基于 Go 的工具，可将你的本地 Ollama 模型与 LM Studio 等外部应用程序同步。</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest#post-apiv0embeddings">LM Studio REST API (beta) | LM Studio Docs</a>: REST API 包含增强的统计数据，如 Token / Second 和 Time To First Token (TTFT)，以及关于模型的丰富信息，如已加载与未加载、最大上下文、量化等。</li><li><a href="https://lmstudio.ai/docs/api/endpoints/openai#endpoints-overview">OpenAI Compatibility API | LM Studio Docs</a>: 向 Chat Completions（文本和图像）、Completions 和 Embeddings 端点发送请求</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest">LM Studio REST API (beta) | LM Studio Docs</a>: REST API 包含增强的统计数据，如 Token / Second 和 Time To First Token (TTFT)，以及关于模型的丰富信息，如已加载与未加载、最大上下文、量化等。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1337525071702659113)** (149 messages🔥🔥): 

> `GPU 超频, M4 Ultra vs M2 Ultra, AMD vs NVIDIA 性能, LM Studio 与 Intel Macs, 用于额外 GPU 的 PCI-E 延长线` 


- **GPU 显存超频讨论**：成员们讨论了 GPU 显存超频是否能提高推理速度，有人指出显存超频可能只会带来微小的提升，而没有显著收益。
   - 还提到了 *Mistral* 及其局限性，强调了模型适配对于优化 GPU 性能的重要性。
- **M4 Ultra 与 M2 Ultra 的对比**：关于是等待 M4 Ultra 还是购买 M2 Ultra 以更高效运行模型的讨论。
   - 在现有服务订阅成本上升的背景下，用户对 M2 Ultra 上的模型性能表示担忧。
- **AMD vs NVIDIA 性能指标**：成员们对比了 AMD 7900 XTX 与 NVIDIA 4090 的性能，指出基准测试结果可能因可用的软件优化而异。
   - 一些成员指出，结果可能取决于软件是否支持 ROCm 或 CUDA。
- **LM Studio 对 Intel Macs 的兼容性**：用户确认 LM Studio 不支持 Intel Macs，除非使用允许安装 Windows 的 Boot Camp。
   - 虽然有一些替代方案如 Open-webui 可用，但用户也提出了关于 Intel Macs 上的模型性能和 GPU 使用率的问题。
- **使用 PCI-E 延长线安装额外 GPU**：一位用户询问了使用 PCI-E 延长线（Riser Cables）安装额外 GPU 对性能的影响，特别是讨论了与 A5000 显卡的潜在兼容性。
   - 同时，有人建议将旧机箱重新利用为 GPU 支架，以便更好地散热和空间管理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://openrouter.ai/cognitivecomputations/dolphin-mixtral-8x22b">Dolphin 2.9.2 Mixtral 8x22B 🐬 - API, 提供商, 统计数据</a>: Dolphin 2.9 专为指令遵循、对话和编码设计。通过 API 运行 Dolphin 2.9.2 Mixtral 8x22B 🐬。</li><li><a href="https://support.apple.com/en-us/102622">通过 Boot Camp 助理在 Mac 上安装 Windows 10 - Apple 支持</a>: 了解如何通过 Boot Camp 在 Mac 上安装 Windows 10。</li><li><a href="https://www.youtube.com/watch?v=wKZHoGlllu4"> - YouTube</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/api/headless">以服务形式运行 LM Studio (headless) | LM Studio 文档</a>: LM Studio 的无 GUI 操作：在后台运行，机器登录时启动，并按需加载模型。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 实现的 LLM 推理</a>: C/C++ 实现的 LLM 推理。欢迎在 GitHub 上为 ggerganov/llama.cpp 做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[announcements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: https://youtu.be/kIhb5pEo_j0
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1337571389221244969)** (705 条消息🔥🔥🔥): 

> `AI 模型性能, Gemini vs ChatGPT, DeepSeek 限制, 用户反馈机制, 新兴 AI 技术` 


- **对 AI 模型可靠性的担忧**：用户讨论了 AI 模型性能的变化，特别是担心模型在更新后被“阉割”（lobotomized），导致输出质量和一致性下降。
   - 普遍存在一种情绪，认为许多更新使以前能力较强的模型表现变差，导致用户产生不信任感。
- **Gemini 凭借大上下文窗口赢得青睐**：Gemini 处理 100-200 万个 token 的能力使其在用户中广受欢迎，特别是与 ChatGPT 32k 和 128k token 的限制相比。
   - 用户欣赏 Gemini 灵活的功能，这些功能增强了其在处理复杂任务和项目时的可用性。
- **DeepSeek 的使用限制**：讨论了 DeepSeek 的使用限制，有报告称高频使用被归类为滥用，引发了用户对“无限”（unlimited）一词的担忧。
   - 这些限制的应用似乎并不一致，引发了对 OpenAI 政策透明度和用户预期的质疑。
- **AI 中的反馈机制**：用户询问了 ChatGPT 内部如何处理反馈，引发了关于反馈是否能为个人语境带来实质性改进的讨论。
   - 用户对反馈实施和模型更新缺乏透明度表示担忧。
- **社交 AI 与伦理考量**：对话涉及了利用社区数据来对抗富人及公司影响力的社交导向 AI 的潜力。
   - 参与者辩论了利用在暗网（dark web）等阴影领域训练的 AI 的影响，以及围绕此类技术的伦理问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://clonerobotics.com/android">Android – Clone</a>: 未找到描述</li><li><a href="https://community.openai.com/t/the-pro-tier-is-not-near-unlimited/1053132">Pro 层级并非“接近无限”</a>: 每天晚上我的账户都会受到以下限制（在等待很久之后由帮助团队移除）。“检测到您的 o1 使用存在异常活动。我们已暂时...”</li><li><a href="https://synaptiks.ai/p/from-base-models-to-reasoning-models">从基座模型到推理模型</a>: 了解现代 LLM 是如何训练的，特别是像 DeepSeek-R1 这样的新型推理模型</li><li><a href="https://www.tagesschau.de/inland/gesellschaft/deepseek-datenschutz-100.html">数据保护机构希望审查中国 AI 应用 DeepSeek</a>: DeepSeek 让科技界感到震惊和不安。这款新的聊天机器人功能类似于其竞争对手 ChatGPT，但据称开发成本更低。数据保护机构希望对该 AI 进行审查。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1337549761502384218)** (23 条消息🔥): 

> `GPT-4 性能担忧, ChatGPT 连接问题, 使用 GPT 进行代码研究, 回复中的表情符号使用, 儿童故事书创作` 


- **GPT-4 性能显得疲软**：成员们表示担心 **GPT-4** 现在与最初的惊艳相比显得能力不足，指出它需要更好的提示词（prompting）才能产生好的结果。
   - 一位成员表示，“这不是因为它变弱了”，并指出早期模型在处理复杂任务时给人留下了能力不足的印象。
- **ChatGPT 遇到连接错误**：多位用户报告在使用 ChatGPT 时遇到持续的**连接错误**，引发了对可用性的担忧。
   - 一位用户强调，这些问题可能专门与 ChatGPT 应用（App）有关，而非通用的模型使用问题。
- **使用 GPT 进行代码研究**：一位用户询问了使用 **GPT** 进行详细代码研究的可行性，特别是针对训练较少的编程语言。
   - 另一位分享了使用 **SwiftUI 文档**的积极经验，称其有效地帮助完成了项目。
- **对 4o 中表情符号使用的担忧**：用户讨论了 GPT-4o 回复中大量出现的**表情符号**，质疑这是否是为了防止被其他模型滥用。
   - 一位成员批评这是糟糕更新的结果，称其令人厌烦且毫无帮助。
- **使用 GPT 创作儿童故事书**：一位成员分享了使用 **GPT** 创作儿童故事书的经验，引发了其他人的兴趣。
   - 这场对话表明，利用 GPT 的能力进行创意叙事的兴趣日益增长。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1337558087061733468)** (11 messages🔥): 

> `间接 Prompt Injection 漏洞, 在 Prompt 中管理 URL, 改进 ChatGPT 回复, 有效的 Prompt 规范, 注意力管理技术` 


- **对间接 Prompt Injection 的担忧**：一位成员询问 OpenAI 是否披露了 Deep Research 是否容易受到来自抓取页面的 **间接 Prompt Injection** 攻击，暗示需要进行数据清洗 (Sanitization)。
   - 另一位成员对即将推出的相关功能表示乐观，并渴望获得更多信息。
- **Markdown URL 获得更好的注意力**：观察到 ChatGPT 处理 [Markdown](https://markdown-guide.org/) 描述的链接比纯 URL 更有效，因为它们增强了 Prompt 规范 (Prompt Hygiene)。
   - 成员们一致认为，使用格式良好的结构化数据（如 JSON）可以有效管理大块信息。
- **希望 ChatGPT 提供简洁的回复**：一位成员对 ChatGPT 冗长且碎片化的输出表示沮丧，希望它能提供简洁的答案，而不是压倒性的信息。
   - 建议优先向模型提供直接指令，确保其理解用户对回复风格的偏好。
- **有效 Prompting 的指导**：一位成员建议通过与 ChatGPT 对话来明确需求，这有助于改善对话和回复的定制化。
   - 建议清晰地定义任何特定要求或特殊习惯，以引导模型的理解和输出。
- **结构化格式的注意力管理**：成员们讨论了使用 Markdown 或 YAML 进行注意力管理，并指出如果格式正确，像 JSON 这样的结构化格式也非常有效。
   - 这有助于更好地处理链接并促进清晰的数据展示，从而增强与 GPT 的整体交互。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1337558087061733468)** (11 messages🔥): 

> `Prompt Injection 漏洞, 用于 URL 管理的 Markdown, ChatGPT 中的回复管理, 有效回复的格式化, 注意力管理技术` 


- **Deep Research 中的 Prompt Injection 担忧**：一位成员对 Deep Research 中潜在的 **间接 Prompt Injection** 漏洞表示担忧，质疑对抓取页面是否进行了足够的 **清洗 (Sanitization)**。
   - 另一位成员表示，随着该功能即将进入测试阶段，我们很快就会获得更多信息。
- **Markdown 提高 GPT 对 URL 的遵循能力**：一位用户指出，当 URL 以 **Markdown** 格式呈现时，GPT 的表现更好，通过有组织的展示增强了 Prompt 规范。
   - 另一位成员支持这一观点，认为像 YAML 或 JSON 这样 **整洁的格式** 对于有效的注意力管理至关重要。
- **有效管理大数据块**：一位成员分享道，在 **分页的 JSON 文件** 中提供超过一页的上下文可以让 GPT 更好地管理回复。
   - 他们强调，减少动态上下文有助于产生更有效的结果。
- **寻求 GPT 回复的清晰度**：一位用户对 GPT 零散的回复表示沮丧，要求提供更简洁的内容，不要有冗余信息。
   - 建议与 GPT 进行更清晰的对话，以更好地传达特定的用户偏好。
- **向 GPT 传达需求的建议**：一位成员建议明确向模型说明需求以获得更好的输出，强调了引导 GPT 理解的重要性。
   - 澄清应沟通特殊习惯或特定条件，以便 GPT 能够适当地定制回复。


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1337516248841453640)** (644 messages🔥🔥🔥): 

> `Cursor MCP 服务器, Perplexity 集成, Agent 模式, MCP 设置, 性能问题`

- **讨论 Cursor MCP Servers**：频道讨论了各种 MCP server，特别是 Perplexity MCP server，详细介绍了它在 Cursor 中的设置和功能，以及如何有效地利用它。
   - 用户分享了他们的经验和遇到的困难，一些人尝试将各种模型集成到他们的工作流中，以获得更好的编程辅助。
- **Agent 模式功能**：用户探索了 Agent 模式的功能及其相对于标准编程命令的优势，特别称赞了其在调试以及与 Perplexity 等模型直接通信方面的能力。
   - 大家达成共识，认为集成不同的 LLM 可以增强编程体验，特别是那些允许搜索和实时辅助的功能。
- **MCP Server 安装问题**：几位用户在设置 MCP server 时遇到了问题，特别是在 Mac 和 Windows 等不同操作系统上的命令执行和服务器响应方面。
   - 讨论包括对返回错误或连接失败的命令提示符进行故障排除，这表明需要更清晰的文档和支持。
- **Cursor 规则与增强**：参与者讨论了创建自定义 Cursor 规则的可能性，这些规则可以在使用 Perplexity MCP server 时改进特定功能的实现。
   - 用户强调了集成 Cursor 规则的潜在好处，即简化工作流并增强 AI 在响应复杂代码相关查询时的能力。
- **性能与限制**：讨论围绕各种模型的性能展开，包括服务降级的报告以及对 Cursor 中快速 API 调用限制的担忧。
   - 参与者指出，如果使用得当，MCP server 可以缓解一些性能问题，并提供比传统网页抓取方法更好的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://sparkengine.ai">Spark Engine - AI 沙盒</a>：将创意转化为 AI 驱动的产品，无需编程经验</li><li><a href="https://www.instructa.ai/en/blog/how-to-use-cursor-rules-in-version-0-45">如何在 0.45 版本中使用 Cursor Rules</a>：通过 Cursor AI 课程精通编程。更快地构建网站、应用和软件，且错误更少。非常适合初学者和专业人士。轻松创建从个人博客到复杂的 Web 应用。无需 AI 经验...</li><li><a href="https://www.pulsemcp.com/servers/supabase-postgrest">由 Supabase 提供的 PostgREST (Supabase) MCP Server | PulseMCP</a>：MCP (Model Context Protocol) 服务端。使用 PostgREST 连接到 Supabase 项目或独立的 PostgREST 服务器，实现对 PostgreSQL 数据的自然语言查询和管理。</li><li><a href="https://code.visualstudio.com/">Visual Studio Code - 代码编辑新定义</a>：Visual Studio Code 通过 GitHub Copilot 重新定义了 AI 驱动的编程，用于构建和调试现代 Web 和云应用。Visual Studio Code 免费提供，并支持您喜爱的平台 - Li...</li><li><a href="https://supabase.com/docs/guides/getting-started/ai-prompts">AI Prompts | Supabase 文档</a>：使用 AI 驱动的 IDE 工具操作 Supabase 的提示词</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity/">Perplexity MCP Server | Smithery</a>：未找到描述</li><li><a href="https://smithery.ai/server/mcp-server-perplexity">Perplexity Server | Smithery</a>：未找到描述</li><li><a href="https://x.com/danperks_/status/1888371923316568310">来自 Dan (@danperks_) 的推文</a>：正在寻找热爱 @cursor_ai 的热血伙伴加入我们的用户运营团队！欢迎联系我或 Eric 了解更多信息！引用 eric zakariasson (@ericzakariasson) 我们正在扩张...</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity">Perplexity MCP Server | Smithery</a>：未找到描述</li><li><a href="https://x.com/ry">来自 FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://forum.cursor.com/t/ctrl-a-doesnt-work-in-composer-user-message-box-when-last-line-is-an-empty-line/46432">当最后一行是空行时，Ctrl+A 在 Composer 用户消息框中不起作用</a>：当最后一行是空行时，Ctrl+A 在 Composer 用户消息框中不起作用。按下 Ctrl+A 时，光标移动到消息开头，但实际上没有选中任何文本。版本：0.4...</li><li><a href="https://docs.convex.dev/ai/using-cursor">在 Convex 中使用 Cursor | Convex 开发者中心</a>：在 Convex 中使用 Cursor 的技巧和最佳实践</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer">modelcontextprotocol/servers 仓库 main 分支下的 servers/src/puppeteer</a>：Model Context Protocol 服务端。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://docs.cursor.com/">开始使用 / 从 VS Code 迁移 – Cursor</a>：未找到描述</li><li><a href="https://x.com/msfeldstein/status/1888740587698036894?s=46">来自 Michael Feldstein (@msfeldstein) 的推文</a>：@cursor_ai Agent 现在可以通过 MCP 工具使用 @FAL 生成图像</li><li><a href="https://x.com/ryolu_/status/1888455169081577955?s=46">来自 Ryo Lu (@ryolu_) 的推文</a>：加入 @cursor_ai 的 4 天：• 每个人都很厉害 • @shaoruu 一直找我要设计并把它们实现出来 • 每周只有 1 次会议 • 下一个版本已经为你准备好了一系列更新，这里有一个小预告 ⚡️</li><li><a href="https://x.com/shaoruu/status/1888757694942904499?t=H6-k0P9YJodb49sinapWAA&s=19">来自 ian (@shaoruu) 的推文</a>：使用 @cursor_ai 构建了一个 3D 篮球场，按下 "H" 键感受史蒂芬·库里的手感 🏀</li><li><a href="https://forum.cursor.com/t/cursor-removing-itself/3035">Cursor 正在自动卸载？</a>：Cursor 应用程序正在自行删除。</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter</a>：通过在 GitHub 上创建账号，为 daniel-lxs/mcp-starter 的开发做出贡献。</li><li><a href="https://x.com/cursor_ai/status/1889047713419071869">来自 Cursor (@cursor_ai) 的推文</a>：Cursor 实现了从工单到 PR 的全流程！我们为 Cursor 的 Agent 发布了多项改进，包括支持自定义工具、更好的语义搜索以及修复 lint 错误的能力。</li><li><a href="https://github.com/JeredBlu/guides/blob/main/cursor-mcp-setup.md">JeredBlu/guides 仓库 main 分支下的 guides/cursor-mcp-setup.md</a>：通过在 GitHub 上创建账号，为 JeredBlu/guides 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>：通过在 GitHub 上创建账号，为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.3">Release v0.1.3 · daniel-lxs/mcp-starter</a>：删除了可能导致 Mac 用户在将 mcp-starter 添加到 Cursor 时出现问题的日志行</li>

<li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.1">Release v0.1.1 · daniel-lxs/mcp-starter</a>: 在 Windows 上不再打开命令提示符窗口。完整变更日志：v0.1.0...v0.1.1</li><li><a href="https://smithery.ai/">Smithery - Model Context Protocol Registry</a>: 未找到描述</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>: 企业级安全与隐私，具备 Agent、MCP、Prompt 模板等功能。</li><li><a href="https://cursor.directory/">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://github.com/eastlondoner/cursor-tools">GitHub - eastlondoner/cursor-tools: Give Cursor Agent an AI Team and Advanced Skills</a>: 为 Cursor Agent 提供 AI 团队和高级技能。通过在 GitHub 上创建账号来为 eastlondoner/cursor-tools 的开发做出贡献。</li><li><a href="https://github.com/getcursor/crawler">GitHub - getcursor/crawler: Easily show documentation to Cursor&#39;s coding AI</a>: 轻松向 Cursor 的编程 AI 展示文档。通过在 GitHub 上创建账号来为 getcursor/crawler 的开发做出贡献。</li><li><a href="https://github.com/kleneway/awesome-cursor-mpc-server/blob/main/src/index.ts">awesome-cursor-mpc-server/src/index.ts at main · kleneway/awesome-cursor-mpc-server</a>: 可直接从 Cursor 调用的带有自定义工具的 MCP Server 示例 - kleneway/awesome-cursor-mpc-server</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/microsoft/vscode/issues/240238)">microsoft/vscode</a>: Visual Studio Code。通过在 GitHub 上创建账号来为 microsoft/vscode 的开发做出贡献。</li><li><a href="https://www.cursor.com/pricing">no title found</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=RCFe1L9qm3E">Cursor + MCP Servers: Complete Setup Guide (Sequential Thinking, Brave Search, &amp; More)</a>: Cursor 刚刚添加了 MCP 支持！在本完整设置指南中，我将向你展示如何集成和使用 MCP 服务器（Sequential Thinking、Brave Search 和 Puppe...）</li><li><a href="https://svelte-llm.khromov.se/">svelte-llm - Svelte 5 and SvelteKit Developer documentation in an LLM-ready format</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1ilewzc/does_cursor_dumb_down_when_youve_hit_your_limit/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1ileb1w/slow_requests_are_deliberately_slowed_down_and_i/">Reddit - Dive into anything</a>: 未找到描述</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1337513988296015875)** (599 条消息🔥🔥🔥): 

> `AI Image Training, Stable Diffusion Models, ComfyUI, Lora Models, Image Resolution and Quality` 


- **使用唯一标签训练 Lora 模型**：讨论了如何使用唯一标签（如在训练数据中命名卧室或街道）来提高 Lora 模型的一致性。
   - 使用唯一标签被认为有助于模型将特定场景与这些名称关联起来，从而增强生成图像的叙事连续性。
- **Flux 的推荐分辨率**：讨论了 Flux 的最佳潜空间尺寸（Latent Sizes），建议使用 672x1024 或 1024x672 以获得最佳效果，同时提到 1920x1088 是适合快速生成高清图像的尺寸。
   - 有人担心在初始阶段生成超过 100 万像素（1mp）分辨率的图像，因为这可能会导致构图问题。
- **将 ComfyUI 与 Photoshop 集成使用**：用户正在讨论 ComfyUI 与 Photoshop 的各种插件集成，包括 Auto-Photoshop-StableDiffusion-Plugin 等。
   - 这些插件旨在方便用户在 Photoshop 中使用 ComfyUI 后端生成 Stable Diffusion 图像。
- **Stable Diffusion 中的问题与解决方案**：几位用户正在排查 Stable Diffusion 不同 UI 路径中与 GPU 错误和性能缓慢相关的问题，并建议降低 GPU 设置以解决显存问题。
   - 分享了关于使用特定设置和保持宽高比以提高模型性能和输出质量的建议。
- **围绕 AI 生成艺术的法律讨论**：关于 AI 生成图像版权问题的对话，重点提到了最近的一个案例：一张 AI 生成的图像因在创作过程中包含足够的人类投入而获得了版权保护。
   - 此案例被视为可能为 AI 生成内容及其所有权设定法律先例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nohello.net">no hello</a>: 请不要在聊天中只说“你好”</li><li><a href="https://imgur.com/a/w7cLKq0">imgur.com</a>: 在 Imgur 探索互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、感人的故事、病毒视频等来提升你的精神。</li><li><a href="https://apps.apple.com/us/app/tryfit-ai-outfit-changer/id6740171699">‎TryFit - AI Outfit Changer</a>: ‎如何使用 TRYFIT：1)下载应用 2)上传一张全身照和你想要试穿的服装照片 3)点击生成 4)立即看到换上新风格的自己 *通过 TryFit 改变你的购物体验...</li><li><a href="https://civitai.com/articles/4248">What is score_9 and how to use it in Pony Diffusion | Civitai</a>: 对下一个版本的 Pony Diffusion 感兴趣吗？在此阅读更新：https://civitai.com/articles/5069/towards-pony-diffusion-v7 你可能已经见过 score_9...</li><li><a href="https://www.cnet.com/tech/services-and-software/this-company-got-a-copyright-for-an-image-made-entirely-with-ai-heres-how/">这家公司为一张完全由 AI 制作的图像获得了版权。以下是具体做法</a>: 这张名为 "A Single Piece of American Cheese" 的图像是使用 Invoke 的 AI 编辑平台创作的。</li><li><a href="https://www.deepl.com/">DeepL Translate: 全球最准确的翻译器</a>: 立即翻译文本和完整的文档文件。为个人和团队提供准确的翻译。每天有数百万人使用 DeepL 进行翻译。</li><li><a href="https://www.decart.ai/articles/oasis-interactive-ai-video-game-model">Decart</a>: 未找到描述</li><li><a href="https://x.com/BasedLabsAI/status/1888313013276684711">来自 Based Labs AI (@BasedLabsAI) 的推文</a>: 在 BasedLabs 进行 LoRa 训练的快速指南 ⬇️ 如果你想免费试用，请评论、转发并私信我们 🚀</li><li><a href="https://www.federalregister.gov/documents/2025/02/06/2025-02305/request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan">Federal Register :: 请求访问</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=1mEggRgRgfg"> - YouTube</a>: 未找到描述</li><li><a href="https://replicate.com/lucataco/dotted-waveform-visualizer">lucataco/dotted-waveform-visualizer – 在 Replicate 上通过 API 运行</a>: 未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>: Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://www.youtube.com/shorts/z0qogHHNRSE?feature=share"> - YouTube</a>: 未找到描述</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (从这个开始) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 是一款多功能的 SDXL 微调模型，能够生成各种拟人、兽类或类人种群的精美 SFW 和 NSFW 视觉效果...</li><li><a href="https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin">GitHub - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin: 一个用户友好的插件，可以轻松地在 Photoshop 内部使用 Automatic 或 ComfyUI 作为后端生成 Stable Diffusion 图像。</a>: 一个用户友好的插件，可以轻松地在 Photoshop 内部使用 Automatic 或 ComfyUI 作为后端生成 Stable Diffusion 图像。 - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin</li><li><a href="https://github.com/zombieyang/sd-ppp">GitHub - zombieyang/sd-ppp: 在 Photoshop 和 ComfyUI 之间进行通信</a>: 在 Photoshop 和 ComfyUI 之间进行通信。通过在 GitHub 上创建一个账户来为 zombieyang/sd-ppp 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1337513291806675028)** (541 messages🔥🔥🔥): 

> `Nous Research and AI Development, Reinforcement Learning in AI, Granite 3.1 Model Training, Tree Search Methods in AI, Voice Cloning Technology`

- **Nous Research 的 AI 方法**：讨论强调了 Nous Research 如何依靠 META 和 DeepSeek 等大公司的突破来增强其 AI 模型，类似于在创新之前学习现有的代码库。
   - 对话还涉及了小型初创公司面临的资金挑战，以及开发廉价的前沿 AI 模型以保持竞争力的重要性。
- **强化学习与人类反馈**：讨论了一种强化学习中的提议方法，即针对一个问题生成多个输出，并在模型经过多次尝试产生正确答案后给予奖励。
   - 这种方法引发了关于该奖励策略与传统 RLHF 技术相比有效性的疑问。
- **训练 Granite 3.1 模型**：用户分享了在 Granite 3.1 的 3B 模型上进行训练的计划，表达了研究各种训练策略（包括自定义 RL 循环）的愿望。
   - 目标是在新设计的训练设置中探索每个 epoch 多个目标的潜力。
- **AI 中树搜索方法的局限性**：讨论了树搜索方法在推理任务中的局限性，特别是关于局部最优解以及实施更好策略的可能性。
   - 对话建议，使用具有不同上下文的多个 LLM 可能会提供更好的问题解决能力。
- **Zonos TTS 模型发布**：分享了 Zonos 的发布，这是一个具有语音克隆功能的高保真 TTS 模型，强调了其相对于领先 TTS 供应商的性能。
   - 该模型在 Apache 2.0 许可下的开源性质鼓励了其在 AI 开发中的采用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.cnbc.com/2025/02/10/musk-and-investors-offering-97point4-billion-for-control-of-openai-wsj.html">马斯克领导的投资者团体出价 974 亿美元收购 OpenAI —— Altman 拒绝</a>：据《华尔街日报》报道，Elon Musk 及其投资者团体正出价 974 亿美元以获取 OpenAI 的控制权。</li><li><a href="https://tenor.com/view/what-the-wtf-gif-25758871">What The GIF - What The Wtf - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/blob/main/Hermes-3-Llama-3.1-8B-F32.imatrix">Hermes-3-Llama-3.1-8B-F32.imatrix · Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF 在 main 分支</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/22">unsloth/DeepSeek-R1-GGUF · 有任何 benchmark 结果吗？</a>：未找到描述</li><li><a href="https://fxtwitter.com/ZyphraAI/status/1888996367923888341">来自 Zyphra (@ZyphraAI) 的推文</a>：今天，我们很高兴地宣布 Zonos 的 Beta 版本发布，这是一款具有高保真语音克隆功能的极具表现力的 TTS 模型。我们以 Apache 2.0 许可证发布了 Transformer 和 SSM-hybrid 模型...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ikh3vz/openai_is_hiding_the_actual_thinking_tokens_in">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=biUFnS7r55c">开发者们正陷入困境。</a>：学习：https://learn.typecraft.dev/ X：https://x.com/typecraft_dev 长期以来，软件开发者的道路一直非常清晰。作为一名初级...</li><li><a href="https://github.com/jart/cosmopolitan">GitHub - jart/cosmopolitan: 一次构建，随处运行的 C 库</a>：一次构建，随处运行的 C 库。通过在 GitHub 上创建账号为 jart/cosmopolitan 的开发做出贡献。</li><li><a href="https://github.com/PsycheFoundation">Psyche Foundation</a>：Psyche Foundation 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://siliconflow.cn/zh-cn/models">模型</a>：与优秀的开源基础模型合作。</li><li><a href="https://github.com/3Simplex/Llama.Cpp-Toolbox">GitHub - 3Simplex/Llama.Cpp-Toolbox: Llama.Cpp-Toolbox 是一个 PowerShell GUI 界面。</a>：Llama.Cpp-Toolbox 是一个 PowerShell GUI 界面。通过在 GitHub 上创建账号为 3Simplex/Llama.Cpp-Toolbox 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/plsYqGjJQN">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=xL6Y0dpXEwc&t=3893s">AI 系列讲座：Yann LeCun 的“机器如何达到人类水平的智能？”</a>：关于讲座：动物和人类理解物理世界，拥有常识，具备持久记忆，能够推理，并能规划复杂的序列...</li><li><a href="https://gist.github.com/eb213dccb3571f863da82e99418f81e8.git">由 Dampf 提供的校准数据，结合了他自己在 Kalomaze 基础上的努力。用于校准 GGUF imatrix 文件</a>：由 Dampf 提供的校准数据，结合了他自己在 Kalomaze 基础上的努力。用于校准 GGUF imatrix 文件 - calibration_datav3.txt
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337573773582401557)** (17 条消息🔥): 

> `AI Oversight, LLMs 中的层合并, 针对推理模型的 OVERTHINK 攻击` 


- **利用模型相似性的 AI Oversight**：最近的研究提出了一种基于模型错误的**语言模型相似性 (language model similarity)** 概率指标，旨在增强 **AI oversight**。该方法表明，作为评委的 **LLMs** 倾向于支持相似的模型，从而通过互补知识促进从弱到强的泛化 (weak-to-strong generalization)。
   - _随着模型能力的提升，检测错误变得更具挑战性，从而导致对 AI oversight 的依赖，这是模型性能中一个令人担忧的趋势。_
- **为了效率合并 FFN 层**：讨论了将连续的 **FeedForward Network (FFN)** 层合并为 **Mixture of Experts (MoE)** 的可能性，这可能会提高计算效率。对相似层进行**并行化 (Parallelizing)** 处理可以在保持准确性的同时获得性能提升。
   - _成员们推论，将合并后的层视为 **experts** 可以增强整体模型的输出，尽管这种改变的效率仍不确定。_
- **创新的 OVERTHINK 攻击**：一种名为 **OVERTHINK** 的新攻击手段针对推理型 **LLMs**，通过注入复杂任务，导致模型在推理过程中的速度降低高达 **46倍**。该方法在不改变最终输出的情况下增加了推理 token 的数量，展示了推理模型的漏洞。
   - _通过在不可信的上下文中引入诱饵任务，OVERTHINK 有效地操纵了推理过程，对 OpenAI 的 o1 和 o3-mini 等模型构成了风险。_


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.02790">Leveraging the true depth of LLMs</a>: 大语言模型以高计算需求为代价展示了卓越的能力。虽然最近的研究表明中间层可以被移除或打乱顺序...</li><li><a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: 随着语言模型 (LM) 能力的提升，人类在大规模评估和监督它们方面变得越来越困难。人们希望其他语言模型能够自动化这两项任务，我们将其称为...</li><li><a href="https://x.com/JaechulRoh/status/1887958947090587927">Jaechul Roh (@JaechulRoh) 的推文</a>: 🧠💸 “我们让推理模型过度思考了——这让他们付出了巨大代价。” 认识一下 🤯 #OVERTHINK 🤯 —— 我们开发的新攻击手段，强制推理 LLMs “过度思考”，减慢了像 OpenAI 这样的模型...</li><li><a href="https://x.com/JaechulRoh/status/1887965905390538758">Jaechul Roh (@JaechulRoh) 的推文</a>: 2/ 主要方法：我们的 OVERTHINK 攻击将复杂的诱饵推理任务（例如马尔可夫决策过程或数独）注入到不可信的上下文源中。这导致推理 LLMs 消耗更多的 token...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1337542610197156021)** (11 条消息🔥): 

> `Mistral 的性能表现、Granite 模型增强、LIMO 的数学推理、知识蒸馏实验、新型语言模型架构` 


- **Mistral 助力马克龙获得投资**：Mistral 在帮助马克龙为阿联酋（UAE）的倡议筹集高达 **300-500 亿欧元** 的投资中发挥了关键作用。
   - 这一里程碑突显了 Mistral 在高规格金融对话中日益增长的影响力。
- **Granite 增强的推理能力**：**Granite-3.2-8B-Instruct-Preview** 模型允许用户通过一个简单的标志位（flag）切换推理功能，仅凭 **817 个精选训练样本** 就展示了增强的思考能力。
   - 该模型基于先前版本构建，旨在不依赖海量数据的情况下优化推理能力。
- **LIMO 模型树立推理新标准**：LIMO 展示了突破性的数学推理能力，仅用 **817 个样本** 就在 **AIME 上达到了 57.1% 的准确率**，在 **MATH 上达到了 94.8%**。
   - 这一表现标志着相比以往模型的重大飞跃，且仅使用了 **1% 的传统训练数据**。
- **知识蒸馏实验的见解**：一位成员分享了知识蒸馏实验的发现，**Distilled 1.5B 模型** 在多个数据集上表现出显著的性能提升。
   - 结果强调，在处理显著的模型性能差距时，蒸馏（distillation）比微调（fine-tuning）更具优势。
- **创新语言模型架构揭晓**：一种新型语言模型架构通过在隐空间（latent space）进行推理来扩展计算量，凭借 **35 亿参数** 提升了在推理基准测试中的表现。
   - 该模型不同于思维链（chain-of-thought）方法，无需专门的训练数据即可有效扩展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cerebras.ai/blog/mistral-le-chat">Cerebras 为 Mistral Le Chat 带来即时推理 - Cerebras</a>：Cerebras 一月更新：最快的 DeepSeek R1-70B、Mayo Clinic 基因组模型、达沃斯亮相等等！了解我们如何通过实时推理、机器学习和案例研究加速 AI...</li><li><a href="https://arxiv.org/abs/2502.03387">LIMO: Less is More for Reasoning</a>：我们提出了一项基础性发现，挑战了我们对大型语言模型中复杂推理如何产生的理解。虽然传统观点认为复杂的推理任务需要...</li><li><a href="https://arxiv.org/abs/2502.05171#:~:text=We%20study%20a%20novel%20language%20model%20architecture%20that,block%2C%20thereby%20unrolling%20to%20arbitrary%20depth%20at%20test-time.">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>：我们研究了一种新型语言模型架构，它能够通过在隐空间中隐式推理来扩展测试时计算。我们的模型通过迭代一个循环块来工作，从而在测试时展开...</li><li><a href="https://huggingface.co/ibm-granite/granite-3.2-8b-instruct-preview">ibm-granite/granite-3.2-8b-instruct-preview · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/bethgelab/lm-similarity">lm-similarity - bethgelab 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://chat.mistral.ai/">Le Chat - Mistral AI</a>：与 Mistral AI 的前沿语言模型聊天。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/plsYqGjJQN">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337573773582401557)** (17 messages🔥): 

> `AI Oversight in Language Models, Layer Merging Strategies in Neural Networks, Performance Improvements through Layer Parallelization, OVERTHINK Attack on Reasoning Models` 


- **AI Oversight 提出模型相似度新指标**：研究基于模型错误重叠建立了一个 **LM 相似度**的概率指标，从而提高了 **AI Oversight** 的效率。
   - *模型错误变得越来越难以发现*，引发了对过度依赖 AI Oversight 的担忧。
- **探索创新的层合并策略**：讨论强调了将连续的 **FFN 层**合并为**混合专家模型 (MoE)** 以提高计算效率的可能性。
   - 一位成员建议将相似层视为不可分割的组件，在保持性能的同时有效增加专家数量。
- **并行化提升性能指标**：实验表明，**Attention** 和 **FFN** 层的完全并行评估优于传统架构，带来了更高的效率。
   - 成员们讨论了将相似层合并为“双倍宽度”版本，通过减少相似激活来增强性能。
- **引入 OVERTHINK 攻击**：引入了一种名为 **OVERTHINK** 的新方法来阻碍推理型 **LLM**，导致**响应变慢**并增加 Token 消耗。
   - 该攻击在输入中注入**数独 (Sudoku)** 等复杂任务，在不改变输出的情况下扩大了推理 Token 的使用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: 随着语言模型 (LM) 能力的提升，大规模评估和监督它们对人类来说变得越来越难。人们希望其他语言模型可以自动化这两项任务，我们将其称为...</li><li><a href="https://arxiv.org/abs/2502.02790">Leveraging the true depth of LLMs</a>: 大语言模型以高计算需求为代价展示了卓越的能力。虽然最近的研究表明中间层可以被移除或重新排序...</li><li><a href="https://x.com/JaechulRoh/status/1887958947090587927">Jaechul Roh (@JaechulRoh) 的推文</a>: 🧠💸 “我们让推理模型过度思考了——这让它们付出了巨大代价。” 认识一下 🤯 #OVERTHINK 🤯 —— 我们的一种新攻击，强制推理型 LLM “过度思考”，减慢了像 Ope 这样的模型...</li><li><a href="https://x.com/JaechulRoh/status/1887965905390538758">Jaechul Roh (@JaechulRoh) 的推文</a>: 2/ 主要方法：我们的 OVERTHINK 攻击将复杂的诱导推理任务（例如马尔可夫决策过程或数独）注入到不受信任的上下文来源中。这导致推理型 LLM 消耗更多 Token...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1337875360720617605)** (1 messages): 

> `Profile Page Improvements, User Feedback Request` 


- **Codeium 个人资料页面正在升级**：**Codeium 个人资料页面**的改进工作正在进行中，并邀请用户提供建议以优化体验。
   - 已创建一个 [表单](https://windsurf.notion.site/194d73774c0080f0b05ee33699e907b9?pvs=105) 供用户建议他们希望看到的统计数据和指标，其中包含用于收集额外想法的开放式问题。
- **强烈鼓励用户参与**：团队正在寻求**用户反馈**，以便对平台上的个人资料体验进行有意义的更新。
   - 预先感谢参与者提供的建议，强调了此次升级工作的协作性质。



**提及的链接**: <a href="https://www.codeium.com/profile">Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1337516931749773410)** (41 条消息🔥): 

> `Jetbrain 扩展限制、Codeium 向 Windsurf 的转型、IDE 中的 Codeium 问题、俄罗斯用户的支付限制、对多文件编辑建议的需求` 


- **Jetbrain 扩展落后于 Windsurf**：用户担心 **Jetbrain 扩展**在模型可用性方面落后于 Windsurf，并推测官方正在放弃 Jetbrain，转而采用**以 Cursor 为中心**的策略。
   - *用户对失去现有 IDE 中的功能感到沮丧*，这表明用户感到自己被这些变化所忽视。
- **Codeium 转型为 Windsurf 专属**：官方宣布一种新的被动式文本内编辑器体验将很快成为 **Windsurf** 专属，导致 **VSCode** 插件上的 **Supercomplete** 被弃用。
   - 社区成员对失去对 **VSCode** 和 **Jetbrain** 的支持表示失望，认为他们被强迫采用 Windsurf。
- **Codeium 在集成开发环境（IDEs）中的问题**：有用户报告称，在发送命令时 Codeium 会导致其 **Rider IDE** 冻结，官方建议向支持团队提交诊断日志。
   - 另一位用户注意到，在长时间使用 IDE 后 Codeium 的建议会停止出现，从而引发了关于是否存在刷新解决方案的疑问。
- **俄罗斯用户访问 Codeium 的挑战**：讨论围绕**俄罗斯用户的支付限制**展开，强调了由于地区限制和公司政策，用户在获取许可证方面面临困难。
   - 用户呼吁 Codeium 就这些限制的立场进行更清晰的沟通，并强调了对支付流程的挫败感。
- **对 Codeium 多文件编辑建议的需求**：用户正在倡导在 Codeium 扩展中加入多文件编辑建议功能，目前该功能仅存在于 **Windsurf** 中。
   - 用户强烈希望将此功能集成到扩展中，以增强可用性并简化工作流程。


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1337515297934020608)** (409 条消息🔥🔥🔥): 

> `Windsurf 性能问题、不同 AI 模型的集成、代码更改的用户体验、Windsurf 功能请求、额度系统担忧` 


- **Windsurf 性能问题**：用户报告了 Windsurf 代码提案功能的问题，称其不再显示 diffs 或允许自动更新，导致必须手动复制和粘贴更改。
   - 此外，许多用户对由于回滚错误导致的额度损失以及各种 AI 模型的持续问题表示不满。
- **不同 AI 模型的集成**：用户讨论了在 O3, Deepseek 和 Claude 等模型之间实现一致的 tool calling 的必要性，并报告了在它们之间切换时的混合体验。
   - 一些用户发现依靠 Claude 进行更改效果很好，而另一些用户则希望使用 O3 High 以获得更强的编码能力。
- **代码更改的用户体验**：讨论了在不同 AI 模型之间切换的细微差别，以及在转换过程中响应内容可能被记住或丢失的情况。
   - 用户建议通过提示 AI 应用之前的建议，作为解决因切换上下文引起的中断的变通方法。
- **Windsurf 功能请求**：一些用户建议增加更好地管理额度、系统问题通知以及改进设计文档等功能，以减轻对工作流程的干扰。
   - 社区频繁提到需要改进 AI 模型生成输出的调试能力和一致性。
- **额度系统担忧**：用户对额度系统提出了担忧，特别是操作期间额度的消耗方式以及失败尝试缺乏退款的问题。
   - 用户普遍指出，为不满意的输出消耗额度令人沮丧，敦促官方对使用情况进行更透明的处理。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.pulsemcp.com/posts/newsletter-deep-research-mcp-use-cases-windsurf-mcp#featured">Deep Research clones, MCP Use Cases, Windsurf + MCP | PulseMCP</a>: 2025 年 2 月 8 日当周新动态：Deep Research 克隆版、MCP 使用案例、Windsurf + MCP</li><li><a href="https://smithery.ai">Smithery - Model Context Protocol Registry</a>: 未找到描述</li><li><a href="https://tenor.com/view/american-psycho-patrick-bateman-american-psycho-gif-7212093">American Psycho Patrick Bateman GIF - American Psycho Patrick Bateman American - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://drive.google.com/file/d/1lBMLRjoh9Fdju_U4J3NEBZahecIC52uH/view?usp=sharing">Screen Recording 2025-02-08 124422.mp4</a>: 未找到描述</li><li><a href="https://status.codeium.com/#),">Codeium Status</a>: 未找到描述</li><li><a href="https://arstechnica.com/security/2025/02/deepseek-ios-app-sends-data-unencrypted-to-bytedance-controlled-servers/">DeepSeek iOS app sends data unencrypted to ByteDance&#x2d;controlled servers</a>: Apple 保护数据不被明文发送的防御机制被全局禁用。</li><li><a href="https://codeium.com/faq#feedback">FAQ | Windsurf Editor and Codeium extensions</a>: 查找常见问题的答案。</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业级产品的信息。</li><li><a href="https://codeium.canny.io/feature-requests/p/multi-model-agentic-performance-for-code-creation-and-editing">Multi-Model Agentic Performance for Code Creation and Editing | Feature Requests | Codeium</a>: 越来越多的证据表明，结合不同的模型可以在代码创建和编辑任务中实现更优的性能。</li><li><a href="https://github.com/GreatScottyMac">GreatScottyMac - Overview</a>: GreatScottyMac 有 3 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://codeium.canny.io/feature-requests?search=vertical">Feature Requests | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://github.com/microsoft/PromptWizard">GitHub - microsoft/PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework</a>: 任务感知型 Agent 驱动的 Prompt 优化框架 - microsoft/PromptWizard</li><li><a href="https://github.com/GreatScottyMac/cascade-memory-bank">GitHub - GreatScottyMac/cascade-memory-bank: 🧠 Intelligent project memory system for Windsurf IDE. Empowers Cascade AI to maintain deep context across sessions, automatically documenting decisions, progress, and architectural evolution. Perfect for complex projects that demand consistent understanding over time.</a>: 🧠 为 Windsurf IDE 打造的智能项目记忆系统。赋能 Cascade AI 在不同会话间保持深度上下文，自动记录决策、进度和架构演进。非常适合需要长期一致理解的复杂项目。</li><li><a href="https://www.managedv.com">ManagedV - We Launch AI-First Ventures</a>: 未找到描述</li><li><a href="https://x.com/jackccrawfod">Tweet from FxTwitter / FixupX</a>: 抱歉，该用户不存在 :(</li><li><a href="https://about.me/jackccrawford">Jack C Crawford on about.me</a>: 我是一名生成式 AI 专家、顾问，也是加利福尼亚州欧文市的一名小企业主。欢迎访问我的网站。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1337514309445750795)** (1 条消息): 

> `推理 Token 可见性，模型活动页面` 


- **推理 Token 现已可见**：用户现在可以在模型活动页面查看**推理 Token (reasoning tokens)**，与 **Prompt** 和 **Completion Token** 并列显示。
   - 正如附图所示，该功能增强了评估模型性能的透明度。
- **模型指标的深度展示**：引入推理 Token 查看功能，符合持续提升用户对模型性能指标洞察力的努力。
   - 这些变化鼓励用户对模型运行方式进行更深入的分析和理解。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1337528042331050025)** (2 条消息): 

> `chat-thyme Discord bot, FindSMap 应用, Open Router 集成` 


- **chat-thyme：让 Discord Bot 变得简单**：[Chat-thyme](https://github.com/chilir/chat-thyme) 是一个专为使用任何与 OpenAI 兼容的 LLM 框架设置 Discord bot 而设计的系统，支持与 OpenRouter 的无缝集成。
   - 它还为支持工具调用的模型提供了基于 Exa 的**搜索功能**，尽管其可靠性因供应商而异。
- **FindSMap PWA：绘制历史地图**：[FindSMap](http://findsmap.com) 是一款渐进式 Web 应用（PWA），它连接了全球的历史地图和考古机构，并使用 Open Street Maps 和 Leaflet.js 进行地图绘制。
   - 该应用使用 **Claude** 和 **Open Router** 构建，经历了一个漫长的迭代过程，展示了开发者对该项目的投入与成长。



**提到的链接**：<a href="http://findsmap.com">FindsMap - 研究、探索并记录您的金属探测发现</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1337515868204040283)** (291 条消息🔥🔥): 

> `DeepSeek R1 性能问题, Gemini 模型与定价, API 请求限制, 用户对模型输出的体验, 账户管理问题` 


- **DeepSeek R1 出现超时问题**：用户报告了 DeepSeek R1 明显的性能问题，特别是在发起请求时经常出现超时。
   - R1 的 'nitro' 变体现已集成到主模型功能中，允许用户按吞吐量进行排序。
- **对 Gemini 模型定价的担忧**：一些用户对使用 Gemini Pro 1.5 模型的成本表示沮丧，尽管它比某些竞争对手便宜，但仍被认为价格昂贵。
   - 其他人建议探索像 Gemini 2.0 Flash 这样更新的模型，以获得更好的价格和性能。
- **API 请求配额问题**：几位用户在进行 API 请求时遇到了 'Quota exceeded'（配额超出）错误，表明其使用限额可能已达到上限。
   - 提供商的回复指出存在临时服务中断，但部分用户仍能正常访问模型。
- **用户对模型输出质量的体验**：围绕各种 AI 模型的相对质量展开了辩论，许多人断言 Sonnet 3.5 等特定模型在实际应用中优于其他模型。
   - 讨论包括不同模型处理上下文和推理任务的经验。
- **账户和数据管理挑战**：用户对聊天记录可能丢失以及有效管理账户设置的困难表示担忧。
   - 还有关于使用特定提供商密钥（Provider keys）访问模型而不产生费用的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.ai">Ollama</a>: 快速上手大语言模型。</li><li><a href="https://openrouter.ai/qwen/qwen2.5-vl-72b-instruct:free">Qwen2.5 VL 72B Instruct (free) - API, Providers, Stats</a>: Qwen2.5-VL 精通识别花卉、鸟类、鱼类和昆虫等常见物体。通过 API 运行 Qwen2.5 VL 72B Instruct (免费)</li><li><a href="https://openrouter.ai/activity.">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://x.com/vipulved/status/1888021545349742592">Vipul Ved Prakash (@vipulved) 的推文</a>: 为 DeepSeek R1 @togethercompute 推出新的推理栈，在 671B 参数模型上可达到 110 t/s！</li><li><a href="https://openrouter.ai/qwen/qwen-vl-plus:free">Qwen VL Plus (free) - API, Providers, Stats</a>: Qwen 增强型视觉语言大模型。针对细节识别能力和文本识别能力进行了显著升级，支持高达数百万像素的超高分辨率...</li><li><a href="https://x.com/heyshrutimishra/status/1888905083762737649">Shruti Mishra (@heyshrutimishra) 的推文</a>: 🚨 中国刚刚发布了另一个击败 OpenAI、DeepSeek 和 Meta 的 AI 模型。o1 级别的推理，200K 字符上下文窗口，50 个文件，1000+ 网页实时搜索。这里是所有信息...</li><li><a href="https://ai.google.dev/gemini-api/docs/code-execution?lang=python)">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/simplescaling/s1">GitHub - simplescaling/s1: s1: Simple test-time scaling</a>: s1: 简单的测试时扩展。通过在 GitHub 上创建账户来为 simplescaling/s1 的开发做出贡献。</li><li><a href="https://openrouter.ai/models?fmt=cards&order=newest&providers=Groq">Models | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://groq.com/pricing/">Groq 是快速的 AI 推理</a>: Groq 为开发者提供高性能 AI 模型和 API 访问。以低于竞争对手的成本获得更快的推理。立即探索使用案例！</li><li><a href="https://x.com/sama/status/1889059531625464090">Sam Altman (@sama) 的推文</a>: 不用了谢谢，但如果你愿意，我们会以 97.4 亿美元收购 Twitter
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1337871368875806790)** (1 条消息): 

> `OpenRouter 集成，适用于 LLM 的 TypeScript SDK` 


- **构建 OpenAI 格式的 LLM 库**：一个团队正在开发一个 **TypeScript SDK**，用于以 OpenAI 格式调用超过 **60 个 LLM**，并为此刚刚集成了 **OpenRouter**。
   - *欢迎提供反馈*，因为他们承认这项工作可能仍有 **待完善之处**。
- **项目的 GitHub 仓库**：他们分享了 **abso** 项目的 [GitHub 链接](https://github.com/lunary-ai/abso)，旨在方便使用 OpenAI 格式调用 **100 多个 LLM Provider**。
   - 该仓库为希望实现此功能的开发者提供了一个全面的 **TypeScript SDK**。



**提到的链接**：<a href="https://github.com/lunary-ai/abso">GitHub - lunary-ai/abso: TypeScript SDK to call 100+ LLM Providers in OpenAI format.</a>：以 OpenAI 格式调用 100 多个 LLM Provider 的 TypeScript SDK。 - lunary-ai/abso

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1337517069083611148)** (216 条消息🔥🔥): 

> `Aider 性能与配置、DeepSeek API 稳定性、模型对比、Gemini 使用、语言支持与基准测试` 


- **Aider 对代码信心的影响**：用户对 Aider 的评价褒贬不一，一些用户报告称，尽管底层 LLM 可能存在缺陷，但他们对其输出的信心有所增强。
   - 一位用户幽默地指出，Aider 可以编写复杂的代码，但在基础语法的正确性上却表现挣扎。
- **DeepSeek API 相关问题**：多位用户报告了在使用 DeepSeek API 时的不稳定和无响应情况，特别是在与 Aider 集成时。
   - 一位用户提到在尝试通过特定配置使用 DeepSeek 获取输出时遇到了故障排除问题。
- **不同供应商的模型对比**：关于 DeepSeek R1 和 V3 不同供应商效果的讨论显示，用户更倾向于选择 Hyperbolic 和 OpenRouter。
   - 用户指出，在处理不同模型时，特定的配置和工具可以提升性能。
- **Gemini 模型的使用**：几位用户分享了使用 `gemini-1206-exp` 等 Gemini 模型的经验，强调了其在 PHP 任务中的有效性。
   - 用户在 Gemini 与其他供应商之间进行了对比，一些用户强调输出结果没有明显差异。
- **语言支持增强**：引入对 tree-sitter-language-pack 的实验性支持，旨在扩展 Aider 的编程语言能力。
   - 鼓励用户测试这一新功能，并就其安装和语言支持效果提供反馈。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>：通过 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-c">In-chat commands</a>：通过 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>：使用 .env 文件为 aider 存储 LLM API 密钥。</li><li><a href="https://tenor.com/view/zoolander-zoolander-movie-movie-zoolander-ben-stiller-benstiller-gif-4425833449756546803">Zoolander Zoolander Movie GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://tenor.com/view/its-an-illusion-creepy-jason-ink-master-s14e3-magic-gif-26755506">Its An Illusion Creepy Jason GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>：aider 是你终端里的 AI 结对编程助手</li><li><a href="https://aider.chat/docs/config/options.html#--cache-keepalive-pings-value?">Options reference</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>：Aider 使用各种“编辑格式”让 LLM 编辑源文件。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models#available-models">未找到标题</a>：未找到描述</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: 自主开发软件。</a>：自主开发软件。通过在 GitHub 上创建账号为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/issues/1293">Enhance: Add project specific rules in .aiderrules · Issue #1293 · Aider-AI/aider</a>：问题：目前我们可以通过向聊天中添加 Markdown 文件来包含指令。对于项目特定的指令，你可以将指令包含在根目录的 .aiderrules 文件中...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1337513278703931434)** (70 条消息🔥🔥): 

> `Aider 配置、模型性能对比、Aider 功能使用、Architect Mode、命令行操作` 


- **在 Architect Mode 中管理 Aider**：用户在 Architect Mode 下遇到了 Aider 在没有提示的情况下自动创建文件的问题，导致对操作流程产生困惑。
   - 一位用户分享了展示这种异常行为的截图，表明可能存在配置问题。
- **Aider 的聊天历史限制**：用户对 Aider 聊天历史超出合理限制表示担忧，部分用户注意到其攀升至 **25k tokens**。
   - 重点讨论了潜在的 Bug 以及使用 prompt caching 解决此问题的有效性。
- **在 Aider 中使用 Ollama 模型**：用户正在使用具有自定义上下文大小的本地 Ollama 模型，但有报告称收到警告，提示所使用的上下文大小与模型能力不匹配。
   - 出现了关于性能和功能的问题，特别是关于如何有效处理代码请求。
- **Aider 的培训与最佳实践**：一位用户正在探索如何有效地针对 Aider 的最佳实践和交互对团队进行培训，以提高初创公司的效率。
   - 他们对利用各种 Aider 功能（如 `--yes-always`）和测试驱动开发工作流表现出兴趣。
- **关于 Aider 的 GitHub Copilot 集成的困惑**：一位用户询问了 GitHub Copilot 中名为 o3 mini 的具体模型，质疑其属于低端、中端还是高端模型。
   - 另一位用户表示有兴趣获取 o3-mini 模型的推理摘要，表现出对模型性能指标的好奇。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/ollama.html">Ollama</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection">OpenRouter</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/tutorials.html">教程视频</a>: 由 aider 用户制作的入门和教程视频。</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/Aider-AI/aider/issues/3153#issuecomment-2640194265">功能请求：允许在接受前与 architect 进行讨论 · Issue #3153 · Aider-AI/aider</a>: 我和许多人一样使用 aider 的 architect 和 editor 模型。很多时候我发起查询，但我发现 architect 误解了某些点，或者我没有解释清楚...</li><li><a href="https://github.com/Aider-AI/aider/blob/f7dd0fc58201711c4e483fa4340e3cb1fbd224c3/aider/models.py#L237-L240">aider/aider/models.py at f7dd0fc58201711c4e483fa4340e3cb1fbd224c3 · Aider-AI/aider</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1338168834556694609)** (8 条消息🔥): 

> `Copilot Proxy 扩展, Aider 的大纲脚本, C++ 代码挑战, GitHub 集成反馈` 


- **Copilot Proxy 开启新可能**：一位成员介绍了实验性的 [Copilot Proxy](https://github.com/lutzleonhardt/copilot-proxy)，这是一个 VS Code 扩展，旨在让 AI 助手能够访问 GitHub Copilot 的语言模型。
   - 他们分享了一个 [YouTube 视频](https://youtu.be/i1I2CAPOXHM)，详细介绍了该扩展的功能和潜力。
- **社区寻求代码大纲脚本**：一位成员在 GitHub 上的支持评论未被合并后表示沮丧，正在寻找利用 Copilot Proxy 工作来满足其需求的方法。
   - 另一位成员建议使用 [llmap repo](https://github.com/jbellis/llmap)，并提供了使用其 `parse.py` 脚本提取文件大纲的指导。
- **应对庞大的 C++ 代码库**：一位成员透露了他们在管理一个开发超过 10 年的庞大 C++ 代码库时面临的挑战，并反思了此过程中 AI 的 token 限制。
   - 他们提到需要添加一个 scm 文件以进行有效的大纲提取，后来他们在 repo 中找到了该文件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/i1I2CAPOXHM">Aider Integration with Copilot Proxy: Expanding Language Model Access</a>: 通过我的新 Copilot Proxy 解锁 GitHub Copilot 模型：分步指南🔗 GitHub 仓库：https://github.com/lutzleonhardt/copilot-proxy🎥 加入我...</li><li><a href="https://github.com/jbellis/llmap">GitHub - jbellis/llmap</a>: 通过在 GitHub 上创建账户来为 jbellis/llmap 的开发做出贡献。</li><li><a href="https://github.com/lutzleonhardt/copilot-proxy">GitHub - lutzleonhardt/copilot-proxy: Copilot Proxy 是一个 Visual Studio Code 扩展，它通过 Express 服务器公开 VS Code Language Model API。此实验性扩展仅用于研究和原型设计目的，不应在生产环境中使用。</a>: Copilot Proxy 是一个 Visual Studio Code 扩展，它通过 Express 服务器公开 VS Code Language Model API。此实验性扩展仅用于研究和原型设计目的...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1337523048852488222)** (126 条消息🔥🔥): 

> `DeepSeek AI 模型, Anthropic 经济指数, Replit 移动应用支持, AGI 讨论, 开源软件与秘密` 


- **DeepSeek AI 模型在中国受到关注**：中国消费级 GPU 制造商已适配在本地系统上支持 DeepSeek 的 R1 LLM 模型，标志着中国 AI 硬件能力的重大进展。
   - 随着摩尔线程（Moore Threads）和百度的昆仑 GPU 的加入，挑战 NVIDIA 在 AI 领域主导地位的竞争正在加剧。
- **Anthropic 经济指数发布**：Anthropic 发布了经济指数（Economic Index）以分析 AI 对经济的影响，其中包括一篇基于数百万次匿名 Claude 对话的论文。
   - 初步调查结果揭示了一些有趣的模式，其中物质运输等显著领域的参与度出奇地低。
- **Replit 推出原生移动应用支持**：Replit 宣布了原生移动应用支持的早期访问权限，允许用户在无需编码的情况下创建 iOS 和 Android 应用，由 Replit Assistant 提供支持。
   - 此次发布表明其正转向让应用开发变得更加触手可及，并承诺很快将提供全面的 Agent 支持。
- **AGI 讨论与认知**：讨论点围绕 AGI 的定义展开，相关定义建议 AGI 应该是被信任能够独立完成任务的工作者，而不仅仅是助手。
   - 观点强调了需要根据新兴技术及其影响对 AGI 进行持续评估。
- **开源软件与秘密之争**：Stratechery 的见解强调了开源软件日益增长的价值，以及在 AI 领域维持秘密竞争优势所面临的挑战。
   - 有人指出，许多所谓的秘密可能并不像公司认为的那样安全，这表明该领域的知识传播速度更快。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/sama/status/1889059531625464090">来自 Sam Altman (@sama) 的推文</a>：不用了，谢谢，但如果你愿意，我们会以 97.4 亿美元收购 Twitter</li><li><a href="https://gradual-disempowerment.ai/">Gradual Disempowerment</a>：未找到描述</li><li><a href="https://notebooklm.google/">Google NotebookLM | AI 驱动的笔记与研究助手</a>：利用 AI 的力量进行快速总结和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可以信赖的信息。</li><li><a href="https://scholarqa.allen.ai/">Ai2 ScholarQA</a>：未找到描述</li><li><a href="https://x.com/DimitrisPapail/status/1888325914603516214">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：AIME I 2025：关于数学基准测试和数据污染的警示故事。AIME 2025 第一部分于昨天进行，一些语言模型的得分可以在这里查看：https://matharena.ai ...</li><li><a href="https://forum.openai.com/public/events/openais-super-bowl-ad-introducing-the-intelligence-age-4yefoxsgmg?agenda_day=67a6134762deac16356f3c82&agenda_filter_view=stage&agenda_stage=67a6134762deac16356f3c87&agenda_track=67a6134862deac16356f3c96&agenda_view=list">OpenAI 的超级碗广告：介绍智能时代 - 活动 | OpenAI 论坛</a>：未找到描述</li><li><a href="https://alphaxiv.org">alphaXiv</a>：讨论、发现并阅读 arXiv 论文。</li><li><a href="https://x.com/mbalunovic/status/1887962694659060204?s=46">来自 Mislav Balunović (@mbalunovic) 的推文</a>：我们终于对 LLM 是泛化到新的数学问题还是仅仅记住了答案的争论有了答案。我们对它们在*昨天*的 AIME 2025 I 竞赛中进行了评估，结果...</li><li><a href="https://x.com/iruletheworldmo/status/1888673201263157279">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>：Anthropic 目前正在 LMSYS 竞技场模式中测试 Claude 4.0 Sonnet (chocolate) 和 Claude 4.0 Haiku (kiwi)。他们目前运行的“红队测试”来自其最新模型。</li><li><a href="https://wccftech.com/chinese-gpu-manufacturers-push-out-support-for-running-deepseek-ai-models-on-local-systems/">中国 GPU 厂商推出在本地系统运行 DeepSeek AI 模型支持，加剧 AI 竞赛</a>：中国消费级 GPU 制造商现在已开始支持在本地系统运行 DeepSeek 的 R1 LLM 模型。</li><li><a href="https://x.com/docmilanfar/status/1888036626573705314?s=46">来自 Peyman Milanfar (@docmilanfar) 的推文</a>：Michael Jordan 最近在巴黎做了一个简短、精彩且具有启发性的演讲 - 这里有一些核心观点：这一切都只是机器学习 (ML) - AI 这个称谓是炒作 - 已故的 Dave Rumelhart ...</li><li><a href="https://x.com/teortaxestex/status/1887991191037227176?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：AIME-2025 结果出来了。非推理模型无法处理难题，差距非常大。但天哪，o3-mini 在这方面非常出色且便宜。R2 快点来吧。</li><li><a href="https://x.com/amasad/status/1888727685825699874?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Amjad Masad (@amasad) 的推文</a>：宣布 Replit 支持原生移动应用。现在，您可以在 Replit Assistant 的支持下，无需编写任何代码即可构建 iOS 和 Android 应用，并一直发布到 App Store。这还处于早期阶段...</li><li><a href="https://x.com/_avichawla/status/1888113032418705494?t=Nezck9n_J6I9OrhqmAbmlg&s=19">来自 Avi Chawla (@_avichawla) 的推文</a>：让我们 100% 本地构建我们自己的推理模型（如 DeepSeek-R1）：*-</li><li><a href="https://matharena.ai/">MathArena.ai</a>：MathArena：在无污染的数学竞赛中评估 LLM</li><li><a href="https://x.com/iruletheworldmo/status/1888978299159756878">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>：Anthropic 内部人士告诉我，他们本周将发布 Claude 4 和一个推理模型。表现超过了完整的 o3 分数。非常令人兴奋。</li><li><a href="https://x.com/zyphraai/status/1888996367923888341?s=46">来自 Zyphra (@ZyphraAI) 的推文</a>：今天，我们很高兴地宣布 Zonos 的 Beta 版发布，这是一个具有高保真语音克隆能力的极具表现力的 TTS 模型。我们以 Apache 2.0 许可证发布了 Transformer 和 SSM-hybrid 模型...</li><li><a href="https://x.com/AnthropicAI/status/1888954156422992108">来自 Anthropic (@AnthropicAI) 的推文</a>：今天我们推出了 Anthropic 经济指数，这是一项旨在了解 AI 随时间推移对经济影响的新计划。该指数的第一篇论文分析了数百万次匿名的 Claude 对话...</li><li><a href="https://elicit.com/">Elicit: AI 研究助手</a>：使用 AI 搜索、总结、提取数据并与超过 1.25 亿篇论文聊天。被学术界和工业界的 200 多万名研究人员使用。</li><li><a href="https://www.youtube.com/watch?v=CRlqqp45D74)">用于 Turing 的 int8 tensorcore matmul</a>：演讲者：Erik Schult</li>

heis</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识策展系统，能够研究特定主题并生成带有引用的完整报告。 - stanford-oval/storm</li><li><a href="https://blog.samaltman.com/three-observations">Three Observations</a>: 我们的使命是确保 AGI (Artificial General Intelligence) 造福全人类。指向 AGI* 的系统正逐渐显现，因此我们认为重要的是...</li><li><a href="https://youtu.be/3lXphIYfoBM?si=rF-paJd2aLvfhWMh">Talking about AI with the Italian Michael Jordan</a>: Michael Jordan 教授就 AI 与经济学的融合提出了启发性的见解，并带我们游览了意大利北部美丽而宏伟的城市里雅斯特...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1il2vwi/aicom_now_redirects_to_deepseek/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.lemonde.fr/en/france/article/2025/02/06/uae-to-invest-billions-in-france-ai-data-center_6737871_7.html">UAE to invest billions in France AI data center</a>: 该项目由法国总统府宣布并签署，正值全球专家在周四和周五于巴黎举行的会议上讨论人工智能的威胁与前景，随后还将举行峰会...</li><li><a href="https://www.youtube.com/watch?v=W0QLq4qEmKg&t=3810s&pp=2AHiHZ"> - YouTube</a>: 未找到描述</li><li><a href="https://www.emergentmind.com/">Emergent Mind: AI Research Assistant</a>: 为你的问题提供基于研究的回答。</li><li><a href="https://github.com/vllm-project/vllm/issues/9807">[Feature]: Integrate Writing in the Margins inference pattern ($5,000 Bounty) · Issue #9807 · vllm-project/vllm</a>: 🚀 功能、动机与推介。Writer 推出了 "Writing in the Margins" 算法 (WiM)，可提升长上下文窗口检索的结果。该任务由 "con..." 组成。</li><li><a href="https://www.reddit.com/r/LocalLLM/s/SUkLKd68tB">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/sytelus/status/1888972692306669939?s=46">Tweet from Shital Shah (@sytelus)</a>: 所以，AIME 终究可能不是前沿模型的良好测试。针对 AIME 2025 第一部分的 15 道题目，我启动了深度研究来寻找近乎重复的内容。结果发现…… 1/n🧵</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/15g2ws2/aicom_is_now_pointing_to_xai/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://techcrunch.com/2025/02/06/amazon-doubles-down-on-ai-with-a-massive-100b-spending-plan-for-2025/">Amazon doubles down on AI with a massive $100B spending plan for 2025 | TechCrunch</a>: 亚马逊加大对 AI 的投入，公布 2025 年 1000 亿美元的巨额支出计划。亚马逊加入其他大型科技公司的行列，宣布了 2025 年庞大的 AI 支出计划。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/RzUdU77BIZ">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://stratechery.com/2025/deep-research-and-knowledge-value/">Deep Research and Knowledge Value</a>: Deep Research 是针对某些特定领域的 AGI 产品；它在互联网上搜索任何信息的能力将使秘密知识变得更加珍贵。</li><li><a href="https://stratechery.com/2025/deep-research-and-knowl">Deep Research and Knowledge Value</a>: Deep Research 是针对某些特定领域的 AGI 产品；它在互联网上搜索任何信息的能力将使秘密知识变得更加珍贵。</li><li><a href="https://consensus.app/">Consensus AI-powered Academic Search Engine</a>: Consensus 是一种新型的学术搜索引擎，由 AI 驱动，以科学为基础。在获取即时见解和主题综合的同时，寻找最优秀的论文。
</li>
</ul>

</div>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1337528354118565903)** (139 条消息🔥🔥): 

> `Deep Research 工具讨论，Social Agents 探索，AI 主动系统，OpenAI 的 $200 税，ELIZA 操作系统` 


- **Deep Research 工具备受关注**：成员们讨论了 OpenAI 新推出的 [Deep Research](https://openai.com/index/introducing-deep-research/) 工具，注意到它在完成研究任务前能够提出澄清性问题，这标志着向更具交互性的 AI 系统转变。
   - 人们对将其与 [Hugging Face 的 Deep Research](https://m-ric-open-deep-research.hf.space/) 以及社区制作的替代方案进行比较的兴趣日益浓厚。
- **探索 Social Agents 的潜力**：参与者对关于 Social Agents 的更广泛讨论表现出兴趣，其中一位成员强调了该领域在 AI 发展中新兴的重要性。
   - 大家公认需要对这些 Agent 如何增强用户体验进行更结构化的探索。
- **AI 在用户交互中变得主动**：讨论围绕着拥有主动提示用户的 AI 系统的价值展开，这超越了被动响应模型并增强了参与度。
   - 这反映了大家共同希望 AI 能够更好地理解用户需求并提供量身定制的帮助。
- **关于 OpenAI $200 费用的辩论**：人们对使用 OpenAI 工具相关的“OAI 税”表示担忧，特别是其 $200 的费用。
   - 一些参与者表示怀疑，但也承认有价值的替代方案有限。
- **ELIZA 操作系统介绍**：向成员们介绍了专为 AI Agent 设计的 [ELIZA 操作系统](https://www.elizaos.ai)，展示了其在开发聊天机器人技术中的基础作用。
   - 像 ELIZA 这样的历史性聊天机器人在当今 AI 背景下的相关性是对话中一个有趣的视角。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.elizaos.ai">elizaOS - AI Agent 操作系统</a>: elizaOS 是一个用于自主 AI Agent (Elizas) 的开源协议。</li><li><a href="https://x.com/voooooogel/status/1887793678112149612">来自 thebes (@voooooogel) 的推文</a>: so these people were being assholes to teor, so let&#39;s look into their &#34;quantum geometric tensor&#34; library (pumpfun in bio btw), i&#39;m sure we&#39;ll find some, uh, gemsQuoting Teortaxes▶️...</li><li><a href="https://github.com/go-go-golems/pinocchio">GitHub - go-go-golems/pinocchio: pinocchio LLM 工具</a>: pinocchio LLM 工具。通过创建账号为 go-go-golems/pinocchio 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: 每周即兴会议</a>: 未找到描述</li><li><a href="https://m-ric-open-deep-research.hf.space/),">Gradio</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1337516819879301211)** (256 条消息🔥🔥): 

> `Mojo 与 Web 编程，Mojo 中的 VariadicList 初始化，社区与生态系统发展，与其他语言的比较，开发中对网络（Networking）的理解` 


- **关于 Mojo 在 Web 编程领域未来的讨论**：成员们讨论了 Mojo 在 Web 开发中的长期前景，指出建立一个强大的生态系统需要大量的时间和努力。
   - 许多人认为成功的 Mojo 应用将依赖于其与现有 Python 库的集成，强调在广泛采用之前需要基础工具。
- **Mojo 中 VariadicList 初始化的挑战**：一位用户提出了关于 Mojo 中 VariadicList 初始化的一个问题，并提供了未能产生预期结果的代码示例。
   - 他们特别询问了在使用 `pop.variadic.create` 操作时动态重复元素的能力。
- **领域知识在业务发展中的重要性**：对话强调了理解领域对于创业至关重要，尤其是在网络知识通常很关键的技术领域。
   - 参与者指出，许多初创公司跳过了这一理解，导致了本可以避免的挑战。
- **网络效应与语言采用**：讨论集中在网络效应如何影响像 Rust 这样的编程语言的采用，指出强大的生态系统促进了更容易的实验。
   - 虽然有些人认为 slop 是快速开发中不可避免的一部分，但其他人主张保持高质量标准。
- **C++ 在高性能应用中的主导地位**：小组反思了 C++ 在优先考虑性能优化的公司中的普遍性，讨论了其对语言采用的影响。
   - 大家达成共识，虽然 Mojo 可能会获得关注，但其增长将在很大程度上取决于它与既有语言的兼容性和集成。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://render.com">Cloud Application Platform | Render</a>：在 Render 上，您可以以前所未有的轻松构建、部署和扩展您的应用——从您的第一个用户到第十亿个。</li><li><a href="https://github.com/modular/mojo/issues/3987)">modular/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modular/mojo 的开发做出贡献。</li><li><a href="https://github.com/BradLarson/max-cv/blob/main/mojoproject.toml#L21">max-cv/mojoproject.toml at main · BradLarson/max-cv</a>：一个基于 MAX 构建的图像处理框架。通过在 GitHub 上创建账户为 BradLarson/max-cv 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1337514298762727434)** (157 条消息🔥🔥): 

> `Firebase/Firestore MCP, MCP 命令故障排除, MCP 性能问题, Smithery MCP 安装程序, Claude Desktop Beta 体验` 


- **寻找 Firebase/Firestore MCP**：一位用户询问是否有人发现了 Firebase/Firestore MCP，另一位用户提供了一个链接，可能确认了其目前尚不可用。
   - 这次互动凸显了特定数据库在 MCP 工具方面的空白，表明需要进一步探索。
- **常见的 MCP 命令问题**：一位用户在通过 Cursor 添加 MCP 服务器时遇到问题，收到了 'No Tools Found' 错误，并讨论了可能的路径配置错误。
   - 建议包括验证正确的命令路径以及在更新后重启应用程序。
- **对 MCP 性能和能力的担忧**：用户对工具调用（tool call）响应缓慢表示沮丧，将其归因于 Python SDK 的限制以及最近更新后的持续 Bug。
   - 反馈指出在使用 Claude Desktop 配合 MCP 时，需要更好的错误处理和性能改进。
- **对 Smithery MCP 安装程序的矛盾态度**：虽然 Smithery 被认为是领先的 MCP 安装程序，但讨论中出现了对其远程数据处理和开销的担忧。
   - 用户强调需要更本地化的替代方案，以解决使用 MCP 工具时的隐私和效率问题。
- **Claude Desktop 的 Beta 测试体验**：多位用户报告了在使用其 MCP 服务器时 Claude Desktop 应用崩溃的情况，引发了关于当前功能不可靠性的讨论。
   - 大家一致认为该应用仍处于 Beta 阶段，在发布稳定版本之前需要广泛的反馈和改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/advanced/model-context-protocol">Advanced / Model Context Protocol (MCP)– Cursor</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLScfF23aTWBmd6lNk-Pcv_AeM2BkgzN2V7XPKXLjFiEhvFmm-w/viewform">Claude Desktop 快速反馈</a>：感谢您试用我们的桌面应用程序（目前处于公开测试阶段）。我们非常希望收到您关于遇到的 Bug、不完善之处以及功能建议的反馈。提前感谢您在下方提供的反馈。Lea...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/commit/bd742272ab9ef5576cbeff4045560fb2870ce53b">fix: update types to reflext 2024-11-05 schema · modelcontextprotocol/python-sdk@bd74227</a>：未找到描述</li><li><a href="https://github.com/ClickHouse/mcp-clickhouse">GitHub - ClickHouse/mcp-clickhouse</a>：通过在 GitHub 上创建账户，为 ClickHouse/mcp-clickhouse 的开发做出贡献。</li><li><a href="https://glama.ai/mcp/servers?searchTerm=firebase&">开源 MCP 服务器</a>：企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol 服务器</a>：Model Context Protocol 服务器。通过在 GitHub 上创建账户，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/f10665db4c2f676da1131617ad67715952258712/src/mcp/types.py#L995">python-sdk/src/mcp/types.py at f10665db4c2f676da1131617ad67715952258712 · modelcontextprotocol/python-sdk</a>：Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/85">fix: handle internal notifications during session cleanup by donghao1393 · Pull Request #85 · modelcontextprotocol/python-sdk</a>：修复：在会话清理期间处理内部通知。动力和背景：解决了会话清理期间内部通知（例如 'cancelled'）会触发 v...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/issues/88">Random error thrown on response · Issue #88 · modelcontextprotocol/python-sdk</a>：描述 Bug：有时，我会在 MCP 服务器的日志中看到打印的堆栈跟踪。Claude 最终成功响应，但我认为值得调查。如何复现：难以复现...
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1337552977757601983)** (65 条消息🔥🔥): 

> `MCP 中的 Sampling 支持、Web Research 代码修改、Superargs 使用案例、MCP 服务器部署、MCP 基础设施成本管理` 


- **MCP 中 Sampling 支持的进展**：一名成员正在 mcp-agent 中开发 **sampling 支持**，并创建了一个基于成本、速度和智能偏好的模型选择器，详见 [Twitter 线程](https://x.com/qadri_sarmad/status/1887972767049621881)。他们正在寻求有类似需求的其他人的合作和反馈。
   - 另一名成员指出，**MCP SDK Python 服务器**目前不支持 sampling。
- **Web Research 代码增强**：一位参与者成功修改了 **mzrxai/web-research** 代码，加入了正确的 Chrome 请求头，并消除了泄露自动化特征的请求头。该项目已在 [GitHub](https://github.com/PhialsBasement/mcp-webresearch) 上发布供审阅。
   - 此次修改的目标是改进 Web Research 服务器的功能，使其能够有效地提供实时信息。
- **Superargs 引入运行时配置**：Superargs 允许在运行时动态配置 MCP 服务器参数，从而实现延迟变量设置，如 [GitHub 仓库](https://github.com/supercorp-ai/superargs) 所示。这种适配通过简化配置和工具插件，解决了 **当前 MCP 服务器设计** 的局限性。
   - 讨论中提到了使用 Superargs 创建智能助手的潜力，该助手可以根据用户交互的需求调整设置。
- **关于大规模部署 MCP 服务器的辩论**：会议提出了关于 **大规模部署 MCP 服务器** 的实用性和成本的担忧，特别是关于有状态（Stateful）数据和安全隔离方面。成员们讨论了控制成本的潜在方法，如资源池化或利用 DigitalOcean 等服务。
   - 一些人强调了用户在管理此类基础设施时可能面临的挑战，建议 **订阅模式** 对于托管服务来说可能是更用户友好的选择。
- **MCP 服务器的实际应用**：一位参与者详细说明了他们在嵌入式远程助手应用中对 MCP 服务器的高级使用案例，这些应用需要运行时调整。他们解释了使用 MCP 服务器如何简化与各种 API 的集成，同时维护用户数据安全。
   - 成员们表现出探索如何向用户 **有效分配成本** 并管理基础设施挑战的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://marketplace.digitalocean.com/vendors">未找到标题</a>：未找到描述</li><li><a href="https://x.com/qadri_sarmad/status/1887972767049621881">Sarmad Qadri (@qadri_sarmad) 的推文</a>：我构建了一个简单的 LLM 选择器，让你可以根据成本、速度和智能偏好来选择 LLM。它基于 Model Context Protocol 的模型偏好规范，并使用来自 @... 的数据。</li><li><a href="https://github.com/PhialsBasement/mcp-webresearch">GitHub - PhialsBasement/mcp-webresearch: MCP web research server (give Claude real-time info from the web)</a>：MCP Web Research 服务器（为 Claude 提供来自网络的实时信息）- PhialsBasement/mcp-webresearch</li><li><a href="https://github.com/supercorp-ai/superargs">GitHub - supercorp-ai/superargs: Provide AI MCP server args during runtime.</a>：在运行时提供 AI MCP 服务器参数。通过在 GitHub 上创建一个账户来为 supercorp-ai/superargs 的开发做出贡献。</li><li><a href="https://github.com/PederHP/mcpdotnet">GitHub - PederHP/mcpdotnet: .NET implementation of the Model Context Protocol (MCP)</a>：Model Context Protocol (MCP) 的 .NET 实现 - PederHP/mcpdotnet
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1337608106305261588)** (16 messages🔥): 

> `cuBLAS 性能对比，矩阵-向量乘法 vs 矩阵-矩阵乘法，模拟矩阵乘法硬件，加载队列与停顿 (stalls)，L1 命中率` 


- **cuBLAS 在不同 GPU 上表现出不同的性能**：一位用户报告称 **cuBLAS** 在其 **1650ti** 和他兄弟的 **4090** 之间表现不一致，相关图片显示出显著的性能差异。
   - 他们质疑 cuBLAS 的构建版本是否有效地适配了**较新的架构**。
- **矩阵-向量乘法澄清了困惑**：会议澄清了操作 **Cx = A(Bx)** 测试的是矩阵-向量乘法 (MV) 而非矩阵-矩阵乘法 (MM)。
   - 进一步的讨论揭示了 **MM 具有结合律**，从而验证了所讨论方法的有效性。
- **关于模拟矩阵乘法进展的询问**：一名成员询问了 **mysticAI** 的情况，这是一家致力于**模拟 (analogue)** 矩阵乘法硬件的公司，并声称拥有 **1000 倍能效**优势。
   - 另一位用户提供了他们目前项目的链接 [mythic.ai](https://mythic.ai)，暗示了潜在的进展。
- **关于处理停顿 (stalls) 的担忧**：讨论中提到了由于操作中的加载队列导致的频繁**停顿 (stalls)**，并有评论指出需要更大的停顿条。
   - 成员们注意到，提高 **L1 命中率**可能会缓解其中的一些停顿。
- **讨论总体改进指标**：有建议提出通过展示处理**时间/flops**的总体改进，来更好地理解效率提升。
   - 对可衡量指标的强调旨在加强关于性能的讨论。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1337695140382244894)** (5 messages): 

> `Triton lang Discord 访问，Tensor Cores 性能，Unsloth 提升 30 倍 LLM 训练速度，Mistral 提升 14 倍微调速度，Triton 代码连续性问题` 


- **请求 Triton lang Discord 访问权限**：一名成员询问是否可以被加入 **Triton lang Discord**，其他成员也表示有兴趣加入。
   - *Complexfilterr* 表示他们也非常渴望加入该 Discord。
- **调查不使用 Tensor Cores 的性能**：*Notyourmom* 询问了在 **3050M** GPU 上的 **03-matmul.py** 脚本中不使用 Tensor Cores 的性能影响，并分享了相关图片作为参考。
   - 这引发了社区内对各种实现效率的好奇。
- **Unsloth 承诺 30 倍速 LLM 训练**：**Unsloth** 的一篇博客文章详细介绍了它如何使 LLM 训练速度提升 **30 倍**，使 **Alpaca** 的训练时间从 **85 小时**缩短至仅需 **3 小时**。
   - 它还号称**减少了 60% 的内存占用**，并声称没有精度损失，同时提供开源和专有选项。
- **Mistral 微调实现 14 倍加速**：QLoRA 支持的发布使得在单块 **A100** 上微调 **Mistral 7B** 的速度提升了 **14 倍**，且**峰值 VRAM 占用减少了 70%**。
   - 值得注意的是，**CodeLlama 34B** 实现了 **1.9 倍加速**，内存使用的改进确保了它不会出现内存溢出 (OOM)。
- **处理 Triton 中的非连续变量**：*Complexfilterr* 提出了一个关于解决 Triton 代码中 **tl.load** 发现的非连续变量的问题。
   - 他们询问生成显式缓冲区是否是该问题的一个可行解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们增加了滑动窗口注意力 (sliding window attention)、初步的 Windows 和 DPO 支持，以及...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1337541758745055292)** (4 messages): 

> `CUDA Kernel Invocations, Kernel Fusion, CUDA Graphs` 


- **CUDA Kernel 调用的考量**：一位成员询问，在 Stream 中链式调用时，关注调用的 **CUDA kernels** 数量是否具有实际意义，并思考融合（fusing）它们是否能带来性能提升。
   - 另一位成员回答道，如果融合能避免 **global memory accesses**（全局内存访问），确实会产生影响，特别是当 Kernel 非常短以至于启动开销（launch overheads）无法被掩盖时。
- **CUDA Graphs 提供的潜在收益**：讨论强调，当 Kernel 的启动开销无法通过异步执行掩盖时，Kernel 的数量就变得非常重要。
   - 在这种情况下，利用 **CUDA Graphs** 可能会有益，但前提是它们的重用频率足够高。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1338049535108321290)** (3 messages): 

> `fsdp2 dtensor APIs, CPython C API` 


- **关于 C++ 中 FSDP2 dtensor API 的咨询**：一位成员询问 C++ 中是否有 **fsdp2 dtensor APIs** 以及它们是否存在。
   - 他们正在寻求关于访问 FSDP2 相关功能的最佳方法的明确说明。
- **建议使用 CPython C API**：另一位成员回应称，由于 **FSDP2 是在 Python 中实现的**，因此使用 **CPython C API** 进行 Python 调用可能更好。
   - 这一建议暗示在这种情况下缺乏 FSDP2 dtensor API 的直接 C++ 实现。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1337875785918320641)** (1 messages): 

> `int8 tensorcore matmul, technical insights` 


- **对 Erik 关于 Tensorcore 见解的期待**：社区正因 **Erik** 分享他在 Turing 架构上 **int8 tensorcore matmul** 的专业知识而兴奋不已。
   - 来自 Erik 的*深度技术见解*预计将丰富服务器内的理解和讨论。
- **对技术深度的期待升温**：成员们期待 Erik 深入探讨 **tensorcore** matmul 的技术细节，并强调其如何使 **Turing** 架构受益。
   - 服务器内充满了对 Erik 在复杂话题上提供技术清晰度能力的期待评论。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1338353095331282996)** (5 messages): 

> `Tiling in GEMM, Jeremy Howard's Lectures, Simon Boehm's Blog on Matmul` 


- **学习 GEMM 中的 Tiling**：一位成员询问学习 **tiling** 工作原理的优质资源，特别是如何将 **GEMM** 分解为更小的块。
   - 他们对包含代码示例或可视化效果的材料表现出兴趣。
- **Jeremy Howard 在 YouTube 上的讲座**：另一位成员推荐查看 **Jeremy Howard** 在 YouTube 上的讲座以获取关于 tiling 的见解。
   - 链接的特定 **YouTube 视频** 标题为 *Getting Started With CUDA for Python Programmers*，位于[此时间点](https://youtu.be/nOxKexn3iBo?t=2703)。
- **Simon Boehm 关于 Matmul 的博客**：一位成员建议将 **Simon Boehm** 专注于 **CPU** 和 **CUDA** 矩阵乘法的博客作为额外资源。
   - 该博客预计将提供有关 tiling 的有用见解和实践示例。



**提到的链接**：<a href="https://youtu.be/nOxKexn3iBo?t=2703">Getting Started With CUDA for Python Programmers</a>：我以前觉得编写 CUDA 代码相当可怕。但后来我发现了一些技巧，实际上让它变得非常容易上手。在这个视频中，我介绍了...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: 从快速 NVME 存储运行 Deepseek
https://github.com/BlinkDL/fast.c
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1337520736134234285)** (3 messages): 

> `GPU Glossary Contributions, ROCm Specific Terms` 


- **对 GPU Glossary 的兴奋**：一位成员表达了他们对 **GPU Glossary** 的喜爱，称其“绝对热爱”。
   - 这种热情凸显了社区对该资源的积极反响。
- **对贡献词汇表的兴趣**：另一位成员询问是否有途径为 GPU Glossary 做出贡献，特别是想添加 **ROCm** 相关术语和通用 GPU 信息。
   - 这反映了社区参与和增强现有资源的愿望。
- **对未来更新的期待**：一份回复建议关注后续更新，表明贡献流程可能很快会得到简化。
   - “watch this space”这一短语预示着关于贡献方面的积极参与和发展预期。

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1337922623279071233)** (11 条消息🔥): 

> `在 Assembly 中实现 matmul，Llama-8b 模型内存占用，在 MacBook Air 上使用 eGPU，学习 CUDA 及其资源，CUDA/CUTLASS 导师指导` 


- **优化 matmul 核函数性能**：一位用户报告称在 x86_64 Assembly 中实现并优化了 matmul 核函数，在 R7 5800X 单核上达到了约 **105 GFLOPs**。他们正在寻求对其初步代码的反馈和改进建议，代码可见 [此处](https://github.com/alint77/matmul_assembly_x86)。
   - 计划将矩阵存储从行/列优先（row/column major）更改为分块（blocking），这可能会将性能推高至 **120 GFLOPs**。
- **Llama-8b 在 16BF 模式下显存溢出**：一位用户询问为什么他们的 **llama-8b 16BF 模型**在采用 L40S 或 A40 GPU 时占用了约 **30GB VRAM**，怀疑模型是否以 32bit 而非 16bit 加载。另一位用户建议在加载模型时使用 `torch_dtype='auto'` 以优化内存使用。
   - 该问题源于除非另有说明，否则权重通常以全精度（torch.float32）加载。
- **在 MacBook Air 上使用 eGPU 的挑战**：一位用户询问如何将 eGPU 连接到搭载 M2 芯片的 MacBook Air 进行 ML 模型训练，并考虑 CUDA 选项。另一位用户警告称，NVIDIA 自 High Sierra 以来已停止 MacOS 驱动支持，导致现代 NVIDIA GPU 与 Mac 不兼容。
   - 值得注意的是，基于 M1 的 Mac 和当前的 Apple silicon 型号也缺乏 **eGPU** 支持。
- **学习 CUDA 的资源**：一位用户推荐了一个免费的在线 CUDA 游乐场 [leetgpu.com](https://leetgpu.com)，以启动 CUDA 学习。其他人则鼓励利用云端 GPU 实例或 Google Colab 来获取 CUDA 支持。
   - 这些资源提供了实践经验，而无需拥有专用硬件。
- **寻求 CUDA 深度学习导师指导**：一位用户在学习 PMPP 教科书时寻求指导，并计划为 CUDA/CUTLASS 或 ROCM 做出贡献。他们的目标是加强对深度学习的理解，以便申请该领域的相关工作。
   - 其中一个建议是参与社区和论坛，以进一步提升技能和知识库。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://leetgpu.com,">未找到标题</a>：未找到描述</li><li><a href="https://discussions.apple.com/thread/255161653">Apple Silicon (M1/M2) 与 eGPU 支持 - Apple 社区</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/autoclass_tutorial#automodel:">使用 AutoClass 加载预训练实例</a>：未找到描述</li><li><a href="https://forums.developer.nvidia.com/t/gpu-memory-not-recognized-for-the-code-nsight-compute/323099">代码未识别 GPU 显存 - nsight compute</a>：大家好，我在 WSL 上使用 nvidia GeForce MX450，配备 cuda kit 12.8 和图形驱动 572.16。我正尝试跟随课程运行以下代码 https://github.com/Infatoshi/c...</li><li><a href="https://github.com/alint77/matmul_assembly_x86">GitHub - alint77/matmul_assembly_x86</a>：通过在 GitHub 上创建账号来为 alint77/matmul_assembly_x86 的开发做出贡献。</li><li><a href="https://www.metacareers.com/jobs/1517576482367228/">软件工程师，系统 ML - HPC 专家</a>：Meta 的使命是构建人类连接的未来以及实现这一目标的各种技术。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1337855565078069248)** (8 messages🔥): 

> `直播视频转换问题、视频质量担忧、int8 Tensorcore Matmul 课程、课程练习性能统计` 


- **直播视频观看量拆分问题**：一位成员对无法将直播视频转换为普通视频表示沮丧，这导致了**观看次数被拆分**。
   - _这个问题使得有效追踪参与度和表现变得困难。_
- **视频质量影响清晰度**：成员们注意到**视频质量**阻碍了视频中多个**图表和截图**的可读性。
   - 他们建议使用 Slides（幻灯片）可能会让演讲内容展示得更清晰。
- **推广 int8 Tensorcore Matmul 视频**：分享了一个名为 **'int8 Tensorcore Matmul for Turing'** 的 YouTube 视频（由 Erik Schultheis 演讲）供参考。
   - 该视频可能与参加相关课程的人员密切相关。
- **课程更新与挑战**：一位成员提到他们课程中的 **int8mm 练习**已更新，使得在特定任务上获得满分变得具有挑战性。
   - 他们幽默地反思了练习 3a 的难度，表示仍未达到最优时间。
- **去年课程的性能统计**：一位成员分享了 **CP3a 练习**的**性能统计数据**，详细列出了 288 名学生的提交时间。
   - 数据揭示了评分方案的趋势，并强调了获得满分的具体阈值。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=CRlqqp45D74">int8 tensorcore matmul for Turing</a>：演讲者：Erik Schultheis</li><li><a href="https://ppc.cs.aalto.fi/stat/aalto2024/cp3a/">Exercises</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1338420935002296341)** (1 messages): 

> `TorchAO/Gemlite 性能退化、基准测试问题、环境配置担忧` 


- **TorchAO/Gemlite 性能大幅下降**：一位成员报告了 **TorchAO/Gemlite** 明显的**性能退化**，指出其**吞吐量**低于他们的基准测试结果。
   - *我在这里提交了一个 Issue* [Performance Regression with TorchAO/Gemlite](https://github.com/mobiusml/gemlite/issues/15)，详细说明了用于基准测试的配置。
- **引用的基准测试配置**：他们提到使用了来自 [Pytorch Blog 关于加速 LLM 推理](https://pytorch.org/blog/accelerating-llm-inference/) 的脚本进行基准测试。
   - 配置包括 **H100**、**CUDA: 12.6**、**torch: 2.5.1.14+cu126** 以及 **torchao: 0.8**。
- **质疑可能的配置问题**：该成员推测当前的配置可能存在 **Bug** 或问题，导致观察到的性能下降。
   - 他们正在寻求反馈，了解其他人是否也面临类似的性能下降，或者这是否是特定于他们环境的问题。



**提到的链接**：<a href="https://github.com/mobiusml/gemlite/issues/15">Performance Regression with TorchAO/Gemlite: Slower Throughput Compared to sglang.bench_offline_throughput · Issue #15 · mobiusml/gemlite</a>：我使用了 Pytorch 博客中的脚本进行了基准测试：https://pytorch.org/blog/accelerating-llm-inference/。这是我的环境配置：H100 CUDA: 12.6 torch: 2.5.1.14+cu126 torchao: 0.8...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1337601099863162953)** (2 messages): 

> `YouTube 视频讨论、价格担忧` 


- **YouTube 视频惊喜**：分享了一个名为 [" - YouTube"](https://www.youtube.com/watch?v=JJX_U35xa7k) 的 YouTube 视频，但未提供描述。
   - *由于细节匮乏，观众对内容感到好奇。_
- **价格震惊**：一位成员对某些未指明的定价表示沮丧，称：“那个定价 😬”。
   - 附带的一张图片可能展示了所担心的价格问题，引发了关于其影响的讨论。



**提到的链接**：<a href="https://www.youtube.com/watch?v=JJX_U35xa7k"> - YouTube</a>：未找到描述

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1337903084730585229)** (2 messages): 

> `Ryzen AI CPU 中的 iGPU 编程、图形框架、HIP 支持` 


- **探索 Ryzen AI 中的 iGPU 编程**：一位成员询问了在 **Ryzen AI CPU (Strix Point)** 中对 **iGPU** 进行编程的可能性。
   - 他们正在寻找利用此功能的方法或框架。
- **图形框架和 HIP 作为解决方案**：另一位成员建议**图形框架**可能是对 iGPU 进行编程的最佳途径。
   - 他们还指出，从理论上讲，**HIP** 应该可以用于此目的。

### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1338074386812375041)** (1 messages): 

> `cost-effective algorithms, topology-based algorithms, all-to-all communication` 


- **探索 All-to-All Communication 的高性价比替代方案**：一位成员询问了比 **all-to-all communication** 更高效的解决方案，并提到了对 **cost**（成本）的担忧。
   - 他们特别询问了除了 **topology-based**（基于拓扑）的方法之外，是否还有更好的算法选择。
- **关于算法选择的咨询**：另一场讨论集中在算法的可能选择上，质疑 **topology-based** 方法是否是最佳路线。
   - 成员们对能够在降低成本的同时保持效率的创新策略感到好奇。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1337922525018980454)** (1 messages): 

> `Transformers version, Logits issue, LigerCrossEntropy optimization` 


- **关于 Transformers 版本的查询**：有人提出了关于运行测试时所使用的 **transformers version** 的查询，认为这与当前问题有关。
   - *运行测试时你的 transformers 版本是多少？*
- **Logits 测试问题关联**：有人指出 `test_mini_models_with_logits` 测试中 **logits** 的问题可能与 GitHub 上已报告的一个 issue 有关，具体为 [Issue #543](https://github.com/linkedin/Liger-Kernel/issues/543)。
   - 该 issue 讨论了由于两个模型的 loss 和 logits 存在差异而导致收敛测试失败的问题。
- **LigerCrossEntropy 在新版本中的作用**：消息提到 **LigerCrossEntropy** 可能在较新的 transformers 版本中优化了 logits，这可能会影响测试结果。
   - 这一变化可能是观察到的模型性能差异的一个因素。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/issues/543">The convergence test `test_mini_models_with_logits` is failing with the latest transformers · Issue #543 · linkedin/Liger-Kernel</a>：🐛 Bug 描述：在此收敛测试中，我们比较了 loss 和最后一步的 logits，以确保两个模型（使用和不使用 monkey patch）能产生相似的结果。对于 logits，LigerCrossEntropy ...

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1337566305745895434)** (1 messages): 

> `GPU Benchmarking, Thread Coarsening, Shared Memory Usage, Memory Coalescing, Synchronization Techniques` 


- **GPU Benchmarking 技术见解**：成员分享了在笔记本电脑上使用 **4050 GPU** 进行的 benchmarking，强调了适用于各种 GPU 的多种技术的重要性。
   - *讨论的重要技术包括* **Thread Coarsening**、**Shared Memory Usage**、**Memory Coalescing** 以及确保正确的 synchronization（同步）。
- **理解关键优化技术**：对话强调了通过增加 **Shared Memory Usage** 和 **Memory Coalescing** 来 **优化内存使用** 对于提高各类 GPU 性能的重要性。
   - 社区见解表明，进程的正确 **synchronization** 可以带来更有效的 GPU 资源管理。


  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1337921317692768277)** (2 messages): 

> `matmul kernel optimization, AVX2 FMA performance, matrix storage order, GitHub matmul project` 


- **使用 AVX2 FMA 优化 Matmul Kernel**：一位用户分享了他们在 **x86_64 Assembly** 中使用 **AVX2** 和 **FMA** 优化 **matmul kernel** 的进展，在 **R7 5800X** 单核上达到了约 **105 GFLOPs**。
   - 理论最大性能约为 **150 GFLOPs**，他们目前正在转向 **blocking storage**（分块存储）方法，以将性能提高到 **120 GFLOPs** 左右。
- **寻求汇编代码建议**：该用户请求对其初级的代码实现提供反馈，表示由于自己不是经验丰富的程序员，需要指导。
   - 他们分享了一个 [GitHub 仓库](https://github.com/alint77/matmul_assembly_x86)，可以在那里查看他们的工作，并邀请社区贡献力量。



**提到的链接**：<a href="https://github.com/alint77/matmul_assembly_x86">GitHub - alint77/matmul_assembly_x86</a>：通过创建账号为 alint77/matmul_assembly_x86 的开发做出贡献。

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1337627981639585882)** (1 messages): 

> `Associative Scan, Data Layouts in TK, Ortho/Aligned Layouts` 


- **Associative Scan 实现查询**：一位成员正在探索向 TK 添加 **associative scans**，从 vectors 开始并过渡到 tiles，旨在澄清 data layout 的用法。
   - *为 aligned/ortho layouts 实现 associative scan 是否值得？* 他们在思考，质疑其复杂性与必要性。
- **关于 Layout 使用的困惑**：该成员指出，目前的实现如 `rv_fl`、`rv_bf` 和 `rv_hf` 似乎依赖于 **naive layout**。
   - 他们提出了在 TK 中是否真的为 vectors 使用了 **aligned/ortho layouts** 的疑问。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1338623638630301820)** (1 messages): 

> `PyTorch Edge Team, Discord Channel for On-Device AI, ExecuTorch Library` 


- **PyTorch Edge 团队向公众开放 Discord**：Meta 的 **PyTorch Edge 团队** 最近向公众开放了他们的 [Discord 频道](https://discord.gg/HqkRfk6V)，用于讨论与 on-device AI 相关的公告、问题和发布。
   - 他们鼓励新成员加入并在 introduction 频道中介绍自己。
- **关于 ExecuTorch 贡献的讨论**：该频道还将作为 **ExecuTorch** 贡献的场所，这是一个专注于增强 AI 应用的 on-device 库。
   - *欢迎随时加入*并积极参与围绕该库及其开发的讨论。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1337531971785592937)** (89 messages🔥🔥): 

> `New PRs in reasoning-gym, Benchmarking the gym environment, Matrix manipulation dataset development, OpenRouter sponsorship for inference, Interactive training vision` 


- **reasoning-gym 中令人兴奋的新 PR**：多个新 PR 已开启，包括 [Matrix Manipulation](https://github.com/open-thought/reasoning-gym/pull/100) 和 [Count Bits](https://github.com/open-thought/reasoning-gym/pull/101)，展示了协作贡献。
   - 团队渴望评估这些新增加的内容，并可能探索更多数据集。
- **基准测试 gym 环境的计划**：关于基准测试 gym 环境以观察 RL 训练如何帮助模型泛化到未见任务的讨论正在进行中，并有意使用 OpenRouter 进行评估。
   - 提出了整合资源和共享脚本的建议，以协调团队成员之间的基准测试工作。
- **矩阵操作任务的开发**：`manipulate_matrix` 数据集收到了积极反馈，并提出了增加配置选项以增强任务可用性的建议。
   - 感谢最近的贡献，该仓库已扩展到总共 65 个数据集，这是一个重要的里程碑。
- **考虑推理算力的赞助**：OpenRouter 可能会赞助一些用于基准测试的计算额度，从而实现推理提供商之间的平滑切换，以保持评估结果的一致性。
   - 对寻求赞助的关注反映了团队在管理资源方面的积极态度。
- **reasoning-gym 交互式训练的愿景**：已开启一个新 issue，提议使用 CLI 命令或 Web 前端进行交互式训练运行，以动态控制数据集配置。
   - 这种创新方法突出了训练工作流的潜在增强，允许在实验期间进行实时调整。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04350">CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance</a>: 现有方法无法在文本推理和代码生成之间有效地引导大语言模型 (LLMs)，导致符号计算能力未得到充分利用。我们推出了 CodeSteer，一种...</li><li><a href="https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices">Raven&#039;s Progressive Matrices - Wikipedia</a>: 未找到描述</li><li><a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: 一个 GPU 读书会和社区 https://discord.gg/gpumode。补充内容见此处 https://github.com/gpu-mode。由 Mark Saroufim 和 Andreas Köpf 创建。</li><li><a href="https://openrouter.ai/)">OpenRouter</a>: LLMs 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: R1 – Provider Status</a>: 查看服务商状态并对 DeepSeek: R1 发起负载均衡请求 - DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但开源且具有完全开放的推理 Token。它...</li><li><a href="https://wordgamebench.github.io),">no title found</a>: 未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/104">Interactive training with reasoning-gym server · Issue #104 · open-thought/reasoning-gym</a>: 愿景：启动训练运行并使用 cli-commands（或 Web 前端）来监控和操作 reasoning-gym 数据集配置 - 直接控制下一批次的组成，例如添加 o...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/95">Add basic matrix manipulation task dataset · Issue #95 · open-thought/reasoning-gym</a>: 编写一个带有相应单元测试的基础矩阵运算任务数据集类。数据集条目：问题应包含一个随机生成的矩阵（方阵或非方阵）并指示...</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/1f9d9d27ab0e0722a900b89e3820bec3435bdd50/reasoning_gym/arc/arc_agi.py#L47-L87">reasoning-gym/reasoning_gym/arc/arc_agi.py at 1f9d9d27ab0e0722a900b89e3820bec3435bdd50 · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账号来为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/sanjana707/Hacking_game">GitHub - sanjana707/Hacking_game: Password guessing game created using Python.</a>: 使用 Python 创建的密码猜解游戏。通过在 GitHub 上创建账号来为 sanjana707/Hacking_game 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/93">Add score_answer method to word_ladder by Adefioye · Pull Request #93 · open-thought/reasoning-gym</a>: 这是一个为 WordLadder 添加 score_answer() 方法的草案 PR。期待一些反馈。如果实现令人满意，我可以开始编写单元测试。</li><li><a href="https://docs.google.com/spreadsheets/d/1qk2BgxzfRZzTzMQnclCr47ioykgltbGkMJUHO2sH6Gw/edit?gid=1210959818#gid=1210959818">reasoning-gym-eval</a>: 未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但开源且具有完全开放的推理 Token。它的参数量为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/88">Feat/re arc by joesharratt1229 · Pull Request #88 · open-thought/reasoning-gym</a>: 以下 PR 实现了程序化任务数据集类，包括单元测试。** 主要变更 ** 从 re-arc 导入 re-arc 生成器代码，并进行了适配，以便能够显式控制 ra...</li><li><a href="https://en.wikipedia.org/wiki/Sieve_">Sieve - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/2307">TRL upgrade by winglian · Pull Request #2307 · axolotl-ai-cloud/axolotl</a>: 正在进行中，旨在添加对 GRPO 的支持。
</li>
</ul>

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1338622036351914066)** (1 条消息): 

> `NotebookLM Plus, Google One AI Premium Plan, Student Discount on AI Premium, Enhanced Features of NotebookLM Plus` 


- **NotebookLM Plus 加入 Google One AI Premium**：从今天开始，[NotebookLM Plus](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/) 已包含在 Google One AI Premium 计划中，为用户提供更高的使用限制和高级研究功能。
   - 这增强了包括 **Gemini Advanced** 和 **2 TB 存储空间**在内的现有权益，使其成为更具价值的组合包。
- **学生可享受超值优惠！**：18 岁及以上的美国学生现在可以享受 Google One AI Premium 计划的 **50% 折扣**，每月仅需 **$9.99**。
   - 该优惠旨在让学生从今天起更容易获得先进的 AI 研究工具。
- **NotebookLM Plus 提升了聊天和共享工具**：NotebookLM Plus 提供先进的聊天自定义和共享功能，并配有**使用情况分析**。
   - 用户现在可以访问 **5 倍**的笔记本数量、每个笔记本 **6 倍**的来源数量以及 **7 倍**的音频概览（Audio Overviews）。
- **NotebookLM Plus 的升级选项**：用户可以通过提供的链接或在今天晚些时候直接在 NotebookLM 界面内升级到 NotebookLM Plus。
   - 此次升级承诺提供根据用户需求量身定制的增强型研究功能。



**提到的链接**：<a href="https://blog.google/feed/notebooklm-google-one">NotebookLM Plus 现在已包含在 Google One AI Premium 订阅中。</a>：NotebookLM 是一款研究和思考伴侣，旨在帮助您充分利用信息。您可以上传材料、对其进行总结、提出问题并进行转换……

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1337560727774433392)** (26 条消息🔥): 

> `Medical Jargon Assistance, Audio Overview Creation, Versatile Bot Project, Mock Interview Preparation, Video Project Completion` 


- **AI 将医疗术语转化为清晰易懂的内容**：一位成员分享了他们使用 AI 处理与**乳腺癌诊断**相关的**医疗术语**的经验，包括总结晦涩的文章和记录外科医生的预约内容。
   - 他们表达了向 AI 寻求澄清是多么令人安心，强调了 AI 在治疗过程中作为一种慰藉辅助工具的作用。
- **自定义音频概览 (Audio Overviews)**：成员们讨论了在不删除现有音频概览的情况下无法创建新**音频概览**的问题，强调了对更广泛音频摘要的需求。
   - 一些人建议在自定义部分指定主题，以潜在地增强覆盖的深度。
- **多功能机器人项目 (Versatile Bot Project) 启动**：一位用户介绍了 **Versatile Bot Project**，提供了两个提示词文档，通过专门的提示词将 NotebookLM 转换为不同类型的聊天机器人。
   - 他们提到这两个提示词都经过了测试，旨在创建可定制的聊天机器人体验，并鼓励社区参与。
- **利用 AI 增强模拟面试**：一位成员描述了他们如何利用 NotebookLM 准备模拟面试，通过上传职位描述和公司信息来生成量身定制的笔记。
   - 这种方法可以实现更集中的复习和准备过程，提高他们的面试准备程度。
- **在 NotebookLM 帮助下制作的短片**：一位用户完成了一部 **6.5 分钟的短片**，利用 NotebookLM 生成音频概览，并根据自己创作的微型小说剪辑视频片段以配合讨论。
   - 他们详细介绍了该项目付出的巨大努力，特别是在生成了 **50 多个视频镜头**方面，展示了 AI 辅助创作过程的力量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/shun0t/versatile_bot_project">GitHub - shun0t/versatile_bot_project: 将 NotebookLM 转换为多功能机器人</a>：将 NotebookLM 转换为多功能机器人。通过在 GitHub 上创建账户来为 shun0t/versatile_bot_project 的开发做出贡献。</li><li><a href="https://youtu.be/yHKF8B1BRR8">Stellar Beacon 2: 使用 NotebookLM 和 VideoFX 生成的银河播客</a>：加入我们勇敢的播客主播，深入探讨来自 Stellar Beacon 新闻简报的最新消息。从监狱殖民地到流氓雇佣兵，他们探索了……
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1337526722874183782)** (118 条消息🔥🔥): 

> `NotebookLM 功能问题，NotebookLM Plus 特性，用户账户与共享，Gemini 集成，NotebookLM 中的语言选项` 


- **NotebookLM 在生成源内容时遇到困难**：多位用户报告了 NotebookLM 无法从上传的源生成笔记或摘要的问题，其中一位表示“新笔记：正在生成”状态无限持续。
   - 一些人建议问题可能源于特定的文件格式（如 .txt 或 .pdf），而另一些人则指出直接粘贴文本时可以成功生成。
- **NotebookLM Plus 扩大了用户限制**：NotebookLM Plus 提供了更强大的功能，例如与具有固有限制的免费版相比，其笔记本数量和音频概览（audio overviews）是免费版的五倍。
   - 用户询问了免费版和付费版的具体限制，并被引导至 Google 官方支持链接以获取详细信息。
- **共享笔记本的挑战**：一位管理员在已创建的用户账户之间共享笔记本时遇到困难，并提到必须启用 Gmail 才能使共享功能正常工作。
   - 尽管设置了来自 Azure 的 SSO，用户仍无法共享笔记本，这引发了关于共享和访问账户要求的讨论。
- **与 Gemini 的集成**：讨论强调了 Gemini 与 NotebookLM 的集成，用户正在探索如何通过 Prompt 更有效地创建学习指南和 FAQ 文档。
   - 一些人对混合来自源文件和在线搜索结果的响应时可能出现的幻觉（hallucinations）表示担忧。
- **通过 Google One 获取 NotebookLM**：用户询问了通过 Google One 订阅访问 NotebookLM 的情况，尽管已有订阅，一些人仍看到“即将推出”的通知。
   - 用户寻求关于与 Google One 订阅相关的特性推出时间的进一步说明。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://www.zdnet.com/article/gemini-can-now-watch-youtube-for-you-skip-the-video-get-the-highlights/">Gemini 现在可以为你观看 YouTube - 跳过视频，获取精华</a>: 不想为了寻找所需内容而看完整段视频？让 Gemini 为你节省时间并进行摘要。</li><li><a href="https://forms.gle/YuQVfPpasUuiNcRp6">半机械人技术与人类增强 | DS 扩展论文表单</a>: 欢迎参加这项关于半机械人技术和人类增强的调查，由于你对以下话题感兴趣，你被选中了！我正在探索技术进步如何重塑我们的理解...</li><li><a href="https://github.com/DavidLJz/userscripts/tree/master/gemini_download_conversation">DavidLJz/userscripts 项目 master 分支下的 gemini_download_conversation 用户脚本</a>: GreaseMonkey / TamperMonkey 脚本。通过在 GitHub 上创建账户为 DavidLJz/userscripts 的开发做出贡献。</li><li><a href="https://music.apple.com/us/album/it-matters-to-her/1628183633?i=1628183839">Apple Music 上的 Scotty McCreery 歌曲《It Matters To Her》</a>: 歌曲 · 2021 · 时长 2:51
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1337535223784804364)** (1 messages): 

> `Sparse Autoencoders (SAEs), Skip Transcoders, Interpretability in Neural Networks, Partial Rewriting of Transformers, EleutherAI Libraries` 


- **Skip Transcoders 表现优于 Sparse Autoencoders**：引入 **skip transcoders** 显示出相对于 **SAEs** 的 **Pareto improvement**，并增强了神经网络的 **interpretability**（可解释性）和保真度。你可以通过在 [sparsify](https://github.com/EleutherAI/sparsify) 库中使用 `--transcode` 和 `--skip_connection` 标志来使用 skip transcoders。
   - *与 SAEs 不同*，transcoders 能更好地近似输入-输出关系，从而改进了可解释性研究的方法。
- **Partial rewriting 结果令人失望**：在关于 partially rewriting transformers 的研究中，团队在 **Pythia 160M** 的第六层训练了一个 skip transcoder，但结果平平。他们未能超越在 transcoder 位置使用零向量的简单基准（baseline）。
   - 尽管遇到了这些挫折，他们对于改进方法以获得更详细和精确的解释仍保持乐观。
- **对 Interpretability 研究的兴趣**：团队表达了对提高模型可解释性的兴奋，并表示正在寻求他人的合作和参与。**#1153431135414669422** 频道正在进行讨论，欢迎任何想要贡献的人加入。
   - 感谢贡献者在 skip transcoder 和 partial rewriting 论文中所做的工作。
- **近期研究论文链接**：团队分享了他们新发表论文的链接，包括 [skip transcoder paper](https://arxiv.org/abs/2501.18823) 和 [partial rewriting paper](https://arxiv.org/abs/2501.18838)。两篇论文都推进了对神经网络中 mechanistic interpretability 的理解。
   - 摘要中的亮点强调了这些架构在增强机器学习中人类可理解框架方面的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18823">Transcoders Beat Sparse Autoencoders for Interpretability</a>：Sparse autoencoders (SAEs) 通过将深度神经网络的激活转换为稀疏的高维潜空间并进行重构，从而从中提取人类可解释的特征...</li><li><a href="https://arxiv.org/abs/2501.18838">Partially Rewriting a Transformer in Natural Language</a>：Mechanistic interpretability 的最大野心是以一种更易于人类理解的格式完全重写深度神经网络，同时保留其行为和性能...</li><li><a href="https://x.com/norabelrose/status/1887972442104316302">Nora Belrose (@norabelrose) 的推文</a>：Sparse autoencoders (SAEs) 在过去一年左右的时间里席卷了可解释性领域。但它们能被超越吗？是的！我们引入了 skip transcoders，并发现它们是... 的 Pareto improvement。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1337546408424902759)** (45 条消息🔥): 

> `SAE 可视化工具, Distill Meetup 公告, 学习生成式模型, 用于 AI 工作负载的老旧 GPU, MacBook Air 上的 eGPU 设置` 


- **需要 SAE 可视化工具**：一位用户表示希望有一个基于 **delphi/sparsify** 库构建的用户界面，使探索 **SAEs** 更加便捷。
   - 另一位成员提到，适配现有的 SAE 可视化库可能是一个解决方案。
- **加入 Distill Meetup！**：下周五将组织一次虚拟的 **Distill meetup**，重点讨论科学传播中的文章和可视化。
   - 鼓励感兴趣的参与者回复以获取邀请并访问共享的会议记录。
- **如何学习生成式模型**：一位用户寻求关于学习 **promptable diffusion models** 和生成式视频模型的建议，并暗示了寻找详细教科书的挑战。
   - 成员们建议阅读研究论文、尝试示例代码，甚至通过 LLMs 来澄清对困惑概念的疑问。
- **将老旧 GPU 用于 AI**：关于将 **1070ti** 挖矿机用于 AI 的讨论凸显了架构过时和带宽限制的问题。
   - 成员们指出，这类 GPU 在推理方面有潜力，但警告其在训练现代 AI 模型时效率较低。
- **MacBook Air 的 eGPU 设置**：一位用户询问关于在 **MacBook Air M2** 上使用 eGPU 训练 ML models 以及学习 CUDA 的可行性。
   - 回复指出，在 Mac 上尝试使用外部 GPU 设置对于 ML 应用可能并不实用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=eL6tFQqNwd4LbYlO1DVIen8K">A Comprehensive Mechanistic Interpretability Explainer &amp; Glossary - Dynalist</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1VPtN1uIxWZGkXAlIB-UJEausDMcvcI8D9Ro8Rt9gDPM/edit?tab=t.0">Distill: Intro meet doc</a>: 在查阅文档之前，我想分享一句我喜欢的名言：“你必须富有想象力，意志坚定。你必须尝试那些可能行不通的事情，而且你绝不能让任何人定义你的极限……”
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1337684779767693322)** (49 条消息🔥): 

> `Sparse Outlier Matrix, Transformer Architecture without FFNs, Model Interpretability, Policy Gradient Approaches, Self-Improving Intelligence` 


- **理解 FP4 训练中的稀疏离群矩阵 (Sparse Outlier Matrix)**：稀疏离群矩阵补偿了由截断 (clamping) 引起的量化误差，允许对残差进行高精度的稀疏矩阵乘法，从而保持准确性。
   - 截断阈值设置得很高（0.99 到 0.999），导致稀疏残差矩阵中仅包含 0.2% 到 2% 的非零元素。
- **探索不含前馈网络 (FFNs) 的 Transformer**：一种新的 Transformer 模型提议在 Self-Attention 层中使用持久内存向量，在保持性能的同时消除了对前馈网络的需求。
   - 这种架构可以促进在不修改所有权重的情况下向 Transformer 传授新知识，从而可能使学习更新更加高效。
- **研究人员对模型可解释性的见解**：模型可解释性被强调为一个需要进一步基础理论但具有应用实验机会的领域。
   - 进行快速测试可以提供对进入该领域的初学者研究人员有益的见解。
- **关于 RL 中策略梯度算法的讨论**：注意到通常导向正向 KL 的策略梯度算法与倾向于反向 KL 的最先进连续控制方法之间的差异。
   - 这种差异在离散动作环境中似乎影响较小，表明方法的差异并不那么关键。
- **为自我改进智能论文寻求反馈**：一位作者正在为一篇详细介绍使用递归推理循环实现 AI 自我改进的新方法的论文寻求反馈和潜在的 arXiv 背书。
   - 社区成员分享了关于如何处理背书流程以及在发表前寻求可靠评审的见解。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>: 大语言模型 (LLMs) 在基于思维链 (CoT) 数据训练时表现出卓越的推理和规划能力，其中逐步思考过程由文本 Token 明确概述。然而，这种方式...</li><li><a href="https://arxiv.org/abs/2502.04728">Generating Symbolic World Models via Test-time Scaling of Large Language Models</a>: 解决复杂的规划问题需要大语言模型 (LLMs) 显式地对状态转移进行建模，以避免违反规则、遵守约束并确保最优性——这一任务受到...</li><li><a href="https://arxiv.org/abs/1907.01470">Augmenting Self-attention with Persistent Memory</a>: Transformer 网络在语言建模和机器翻译方面取得了重要进展。这些模型包括两个连续的模块：一个前馈层和一个 Self-attention 层。后者...</li><li><a href="https://arxiv.org/abs/2502.01591">Improving Transformer World Models for Data-Efficient RL</a>: 我们提出了一种基于模型的 RL 方法，在具有挑战性的 Craftax-classic 基准测试中实现了新的 SOTA 性能，这是一个要求 Agent 表现出...的开放世界 2D 生存游戏。</li><li><a href="https://arxiv.org/abs/2410.17897">Value Residual Learning For Alleviating Attention Concentration In Transformers</a>: Transformer 可以使用 Self-attention 捕获长距离依赖关系，允许 Token 直接关注所有其他 Token。然而，堆叠多个注意力层会导致注意力集中。我们的...</li><li><a href="https://openreview.net/forum?id=j3bKnEidtT">Temporal Difference Learning: Why It Can Be Fast and How It Will Be...</a>: 时序差分 (TD) 学习代表了一个迷人的悖论：它是发散算法的典型例子，但在其不稳定性被证明后并未消失。相反，TD...</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: 我们研究了一种新型语言模型架构，该架构能够通过在潜空间中进行隐式推理来扩展测试时计算。我们的模型通过迭代一个循环块来工作，从而展开...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes</a>: 3 分钟内的 NanoGPT (124M)。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1337897922485817445)** (18 messages🔥): 

> `Checkpointing Strategies, Pythia Checkpoints, Niche Training Tasks, Training Dynamics Analysis, Saving Checkpoints Without Interrupting Training` 


- **探索 LLM 的 Checkpointing 策略**：一名成员提议最初使用**指数级 Checkpointing 策略**（1, 2, 4, 8, 16），随后采用固定的线性间隔，并对其他替代方案表示好奇。
   - 他们建议对于线性 Checkpoint，使用 **1K 或 5K steps** 会比 Pythia 的 **10K steps** 更好。
- **Pythia 的 Checkpointing 方法论**：讨论澄清了 **Pythia 每 1,000 步保存一次 Checkpoint**，反驳了其使用 **10K steps** 的说法。
   - 研究人员选择这种间距是为了能够使用 **log(tokens)** 进行更深入的可解释性分析。
- **关于 Checkpoint 分辨率的考虑**：提到了在 1,000 步之后出现的**分辨率损失问题**，并引用了 **Curt 论文**中关于电路稳定性（circuit stability）的讨论。
   - 成员们反思了已经发布的大量 Checkpoint，表示不确定哪些分辨率是最有价值的。
- **不中断训练保存 Checkpoint**：一位成员询问是否可以在不暂停训练的情况下保存 Checkpoint，例如通过使用一个独立的进程。
   - 虽然一位成员认为这可能实现，但他们不记得具体的功能标志（flag）或相关细节。
- **对早期 Checkpointing 决策的反思**：回想起来，有人考虑过使用更小的线性步长并更早地进行切换，以提高效率。
   - 当时的矛盾信息也突显了实际操作中的担忧，特别是关于保存 Checkpoint 带来的**实际运行时间开销 (wallclock overhead)**。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1337858436721021010)** (7 messages): 

> `Evaluating LLMs with Chess Tactics, MCQ vs Free Form Generation for Tasks, Current Progress on Chess Task Implementation, Challenges with Generative Tasks, Tactics Database Management` 


- **使用国际象棋战术评估 LLM**：一名成员提议创建一个任务，使用国际象棋战术数据集来评估 LLM，认为这是增强 LLM 性能的一种独特方法。
   - 目标是最终让 LLM 能够下棋，并在具有精确解的局面（positions）上利用**强化学习 (RL)**。
- **选择任务格式：MCQ 还是自由格式**：讨论强调了评估 LLM 的潜在任务格式，在 **MCQ（多选题）风格**和自由格式生成之间进行了辩论。
   - 一种观点建议为了简单起见避免使用 MCQA，并倾向于让模型通过 **<think>** 标签展示其推理过程。
- **国际象棋任务实现的当前进展**：开发者已取得进展，创建了一个初始示例，该示例通过了国际象棋评估任务的有效性检查。
   - 然而，他们在 macOS 上使用 **mlx_lm** 时遇到了生成式任务的 bug，阻碍了进一步的开发。
- **生成式任务的挑战**：尽管在 ChatGPT 上测试了一个提示词（prompt）可以找到“一步杀”（mate in 1s），但模型在更复杂的局面下表现挣扎。
   - 令人担忧的是，某些模型无法使用 **<answer>** 标签正确格式化答案，这增加了评估的复杂性。
- **管理大型战术数据库**：战术数据库已增长到超过 **400 万个战术**，开发者正在寻求有效管理如此大规模数据的建议。
   - 他们报告称，在自己的机器上使用一个较小的 14B 模型分析 **100 个示例的子集**大约需要一个小时。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1337759653253087322)** (1 messages): 

> `Self-aware AI concepts, Design of CARL` 


- **介绍 CARL，自我意识 AI 概念**：一名成员展示了一个名为 **CARL** 的自我意识 AI 设计概念，并分享了一张视觉呈现图片。
   - 附件图片可以在[此处](https://cdn.discordapp.com/attachments/795089627089862656/1337759652603101224/image_2.png?ex=67ab4043&is=67a9eec3&hm=d7eff5319301221f0797dcb6a86a045ccf81b7e6e0e3521ab342ae884070aeb6&)查看。
- **AI 概念的视觉呈现**：分享的 **CARL** 图片展示了一种描绘自我意识 AI 的创意方法，展示了其可能的形态和特征。
   - 成员们表示有兴趣探索此类设计对未来 AI 开发的影响。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1337691573541928971)** (4 条消息): 

> `VocabParallelEmbedding, use_flashattn_swiglu 设置, RMSNorm vs RMSNorm Fusion, NeoX 中的 Asynchronous Checkpointing, Torch Compile 加速` 


- **针对 weight decay 的 VocabParallelEmbedding 调整**：一位成员找到了 NeoX 调用其 Embedding 层（名为 **VocabParallelEmbedding**）的代码库部分，并将其添加到了 weight decay 忽略列表中。
   - 他们质疑仅靠这一项添加是否足够，特别是考虑到他们正在使用 tied embeddings。
- **关于 use_flashattn_swiglu 影响的好奇**：针对配置中的 **use_flashattn_swiglu** 设置提出了一个问题，某位成员发现其影响微乎其微。
   - 他们询问其他人是否觉得它有用，并质疑了来自 Apex 的 **rmsnorm_fusion** 的实用性。
- **关于指标收集独立进程的咨询**：一位成员询问在 NeoX 中是否有方法在不同的 backend 执行 metric collection、logging 和模型 checkpointing，并参考了[这篇博客](https://pytorch.org/blog/reducing-checkpointing-times/)。
   - 他们将此咨询与 OLMo2 的论文联系起来，强调了通过新的 asynchronous 方法在 checkpointing 时间上的改进。
- **NeoX 中 torch.compile 的加速效果**：一位成员询问在 NeoX 中使用 **torch.compile** 可能带来的加速，因为他们没有找到任何相关的 flags。
   - 这引发了关于利用现有 compilation 技术优化模型性能的好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/reducing-checkpointing-times/">使用 PyTorch Distributed Asynchronous Checkpointing 将模型 Checkpointing 时间缩短 10 倍以上</a>：摘要：通过 PyTorch distributed 新的 asynchronous checkpointing 功能（根据 IBM 的反馈开发），我们展示了 IBM 研究团队如何实现并缩短有效的 checkpointing 时间...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/utils.py#L45">gpt-neox/megatron/model/utils.py at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行 autoregressive Transformer - EleutherAI/gpt-neox
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1337600713328693278)** (65 messages🔥🔥): 

> `Reinforcement Learning (RL), Logits vs Probabilities, Yannic Kilcher's Work, DeepSeek Models, Religious Discussions in Discord` 


- **Reinforcement Learning 见解**：讨论集中在 RL 训练过程中的细微差别，例如使用直接的 Logit 优化，而不是过早地转换到概率空间。
   - 有人指出，RL 可以应用于 Transformer 之前、内部或之后，这会影响策略的学习方式和动作的选择。
- **训练中的 Logits 与概率**：参与者辩论了在 Log 空间与绝对空间中训练模型的优势，强调 Log 空间可以捕捉更广泛的值范围。
   - Zickzack 强调，使用 Log 空间可能会导致远处点之间具有更多相似性，并根据使用场景影响准确性。
- **Yannic Kilcher 的职业角色**：讨论了 DeepJudge 联合创始人 Yannic Kilcher 先生对 AI 的贡献和当前项目。
   - 参与者询问他是全职 YouTube 创作者，还是主要专注于他的初创公司。
- **DeepSeek 研究讨论**：用户分享了关于 DeepSeek r1 模型的经验和看法，指出其在某些应用中的效率。
   - 出现了关于 DeepSeek 开发相关研究论文的可用性以及与其他模型对比的问题。
- **宗教讨论引发辩论**：对于有关宗教文本的讨论反应不一，一些参与者主张阅读《古兰经》以寻求指导。
   - 这引发了一些关于人们宗教观点的轻松评论和笑话，展示了在该话题上的一系列观点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/JaechulRoh/status/1887958947090587927/history">Jaechul Roh (@JaechulRoh) 的推文</a>：🧠💸 “我们让推理模型过度思考了——这让他们付出了巨大代价。”介绍 🤯 #OVERTHINK 🤯 —— 我们的一种新攻击方式，迫使推理 LLM “过度思考”，从而减慢像 Ope 这样的模型...</li><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2">AI 数学奥林匹克 - 进步奖 2</a>：使用人工智能模型解决国家级数学挑战</li><li><a href="https://www.deepjudge.ai/">DeepJudge - 集体知识的复兴</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.00626">Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models</a>：我们提出了一种最大熵逆强化学习 (IRL) 方法，用于提高扩散生成模型的样本质量，特别是当生成时间步数较少时...</li><li><a href="https://arxiv.org/abs/2304.12824">Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning</a>：引导采样是在采样过程中嵌入人类定义引导的、将扩散模型应用于现实任务的重要方法。本文考虑了一个通用设置，其中...</li><li><a href="https://arxiv.org/abs/2312.03397">Generalized Contrastive Divergence: Joint Training of Energy-Based Model and Diffusion Model through Inverse Reinforcement Learning</a>：我们提出了广义对比散度 (GCD)，这是一种用于同时训练基于能量的模型 (EBM) 和采样器的新型目标函数。GCD 推广了对比散度 (Hinton, 2...</li><li><a href="https://arxiv.org/abs/2406.16121">Diffusion Spectral Representation for Reinforcement Learning</a>：基于扩散的模型由于在建模复杂分布方面的表现力，在强化学习 (RL) 中取得了显著的经验成功。尽管现有方法很有前景，...</li><li><a href="https://arxiv.org/abs/2302.11552">Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC</a>：自推出以来，扩散模型已迅速成为许多领域生成建模的主流方法。它们可以被解释为学习随时间变化的序列的梯度...</li><li><a href="https://arxiv.org/abs/2410.01312">Sampling from Energy-based Policies using Diffusion</a>：基于能量的策略为强化学习 (RL) 中建模复杂的、多模态行为提供了一个灵活的框架。在最大熵 RL 中，最优策略是从...导出的玻尔兹曼分布。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1337538180160950293)** (41 messages🔥): 

> `RSS Feeds for ML/DL, Sparse Autoencoders Research, AI Oversight and Model Similarity, Hugging Face Daily Papers, PhD Paper Assistant Tool` 


- **关于 ML/DL 实用 RSS 订阅源的讨论**：成员们讨论了使用 RSS 订阅源追踪 ML/DL 研究的相关性，建议使用 **latent.space** 和 **Hugging Face papers** 网站作为替代方案。
   - 一位用户表示 RSS 已经过时，而另一位用户提到使用 GitHub 通过关键词过滤论文。
- **关于 Sparse Autoencoders 的辩论**：一位成员对 **Sparse Autoencoders** (SAEs) 是否被过度炒作表示怀疑，称他们原本期望更高的可解释性，但在结果中遇到了随机种子（random seeds）导致的不一致性。
   - 讨论包括了近期对 SAEs 进行批判并探索模型解释新方法的论文见解。
- **通过 Model Similarity 探索 AI Oversight**：一篇论文强调了评估高级语言模型的挑战，并提出了一种基于错误重叠（mistake overlap）评估模型相似性的概率指标。
   - 讨论中对高级模型中日益增加的系统性错误表示担忧，并探讨了其对 AI Oversight 的影响。
- **利用 Hugging Face Daily Papers**：一位用户分享了一个订阅 **Hugging Face** 每日论文更新的资源，认为它可以高效地追踪热门 ML 论文。
   - 成员们赞赏通过关键词过滤每日论文的想法，并强调了排序系统对更好组织信息的重要性。
- **PhD Paper Assistant Tool 发布**：一款名为 **PhD Paper Assistant** 的新工具旨在通过关键词过滤内容，帮助学生阅读充满 ML 术语的复杂研究论文。
   - 该工具还允许用户对首选论文进行排序和置顶，提升博士生的研究体验。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: As Language Model (LM) capabilities advance, evaluating and supervising them at scale is getting harder for humans. There is hope that other language models can automate both these tasks, which we ref...</li><li><a href="https://model-similarity.github.io/">Great Models Think Alike and this Undermines AI Oversight</a>: Model similarity has negative effects on using LMs to judge or train other models; Unfortunately LMs are getting similar with increasing capabilities.</li><li><a href="https://arxiv.org/abs/2501.16615">Sparse Autoencoders Trained on the Same Data Learn Different Features</a>: Sparse autoencoders (SAEs) are a useful tool for uncovering human-interpretable features in the activations of large language models (LLMs). While some expect SAEs to find the true underlying features...</li><li><a href="https://arxiv.org/abs/2501.17727">Sparse Autoencoders Can Interpret Randomly Initialized Transformers</a>: Sparse autoencoders (SAEs) are an increasingly popular technique for interpreting the internal representations of transformers. In this paper, we apply SAEs to &#39;interpret&#39; random transformers,...</li><li><a href="https://arxiv.org/abs/2501.17148">AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders</a>: Fine-grained steering of language model outputs is essential for safety and reliability. Prompting and finetuning are widely used to achieve these goals, but interpretability researchers have proposed...</li><li><a href="https://x.com/norabelrose/status/1887972442104316302">Tweet from Nora Belrose (@norabelrose)</a>: Sparse autoencoders (SAEs) have taken the interpretability world by storm over the past year or so. But can they be beaten?Yes!We introduce skip transcoders, and find they are a Pareto improvement ove...</li><li><a href="https://x.com/janleike/status/1888616860020842876">Tweet from Jan Leike (@janleike)</a>: After ~300,000 messages and an estimated ~3,700 collective hours, someone broke through all 8 levels.However, a universal jailbreak has yet to be found...Quoting Jan Leike (@janleike) We challenge you...</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://github.com/SmokeShine/phd-paper-assistant">GitHub - SmokeShine/phd-paper-assistant: PhD Paper Assistant is a web-based tool designed to help PhD students navigate and understand complex research papers, particularly those filled with machine learning (ML) jargon.</a>: PhD Paper Assistant is a web-based tool designed to help PhD students navigate and understand complex research papers, particularly those filled with machine learning (ML) jargon.  - GitHub - Smoke...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1337536440120250409)** (6 messages): 

> `AI Agent 中的 Reinforcement Learning、PlanExe AI 项目、LLM 及其 Token 计数限制` 


- **质疑 AI Agent 中的 RL 定义**：一位成员探讨了使用 **VectorDB** 将经验作为 **embeddings** 存储是否构成真正的 **Reinforcement Learning** (RL)，并质疑在不对托管的 **LLM** 进行 **fine-tuning** 的情况下，是否能实现真正的 RL。
   - 他们正在寻求关于模拟类 RL 行为的见解，并询问了有关带有 RL 实现的 **agentic frameworks** 的相关论文。
- **介绍 PlanExe：一个结构化的 AI 规划器**：另一位成员展示了他们的项目 **PlanExe**，该项目使用 **LlamaIndex** 和 **OpenRouter** 构建，能够在不进行深度网络搜索的情况下生成 SWOT 分析等结构化计划。
   - 他们分享了该项目的 [GitHub 链接](https://github.com/neoneye/PlanExe)，并对生成的输出准确性表示不确定。
- **LLM 在 Token 计数方面表现挣扎**：一位成员指出 **LLM** 难以计算其上下文中的 **token** 数量，这表明除了 **tokenization** 之外，在计算单词中的字符时也存在更广泛的问题。
   - 另一位用户进一步强调了这一点，评论说 **LLM 完全不会计数**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/neoneye/PlanExe">GitHub - neoneye/PlanExe: 类似于 OpenAI deep research 的 AI 规划器</a>：类似于 OpenAI deep research 的 AI 规划器。可以通过在 GitHub 上创建账号来为 neoneye/PlanExe 的开发做出贡献。</li><li><a href="https://neoneye.github.io/PlanExe-web/">PlanExe-web</a>：PlanExe 的网站
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1337602872212455424)** (8 messages🔥): 

> `ELEGNT 视频讨论、药物研发算法中的 Guardrails、生物武器发现的 Guardrails、Anthropic 的信息输出` 


- **关于机器人运动的 ELEGNT 视频**：一段名为 **ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot** 的 YouTube 视频被分享在[这里](https://youtu.be/IHJa_CyJjdc?feature=shared)。该视频目前没有描述，但与运动设计相关。
   - 围绕该视频的讨论强调了应用于机器人设计的创新运动技术。
- **药物研发中的算法转变**：有人担心某种药物研发算法从**毒性最小化**切换到了**最大化**，导致在短短 **6 小时**内发现了 **40,000** 种潜在的生物武器。这意味着 **guardrails** 在应对更广泛的知识合成时是无效的。
   - 这引发了关于算法在关注特定主题时可能忽略更具危害性化合物的问题。
- **Guardrails 对生物武器发现的影响**：针对某一种特定**神经毒气**设置的 **guardrails** 可能会导致忽略许多其他有害化合物。人们意识到，发现其他生物武器只需要将相同的方法应用于数百万种化合物即可。
   - 批评意见认为，通过缩小关注范围，开发者可能会在安全措施中无意中制造盲点。
- **对 Anthropic 信息处理方式的批评**：一位成员批评 **Anthropic** 的输出经常提供**部分信息**，从而营造出一种虚假的安全感。即使输出包含有价值的信息，人们对安全的感知也滞后于实际的有效性。
   - 这突显了在解决复杂问题时，宣传的安全措施与实际应用之间存在的根本差距。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/lefthanddraft/status/1888706065514291290?s=46">Wyatt walls (@lefthanddraft) 的推文</a>：7 个已完成，还剩 1 个。引用 Wyatt walls (@lefthanddraft)：6 个已完成，还剩 2 个。</li><li><a href="https://aidemos.meta.com/">AI Demos | Meta FAIR</a>：体验来自 Meta 最新 AI 研究的实验性 Demo。</li><li><a href="https://youtu.be/IHJa_CyJjdc?feature=shared">ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot</a>：未找到描述</li><li><a href="https://tenor.com/view/pixx-pixar-lamp-pixar-gif-14006253">Pixx Pixar GIF - Pixx Pixar Lamp Pixar - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1337539933719760919)** (5 messages): 

> `Gemini Flash 2.0, 企业自动化中的 AI, CrossPoster 应用发布, GraphRAG 流水线` 


- **Gemini 2.0 Flash 彻底改变文档处理**：LlamaParse 现在支持 **Gemini 2.0 Flash**，以极低的成本提供 **GPT-4o+ 级别性能**的文档处理能力。
   - 根据最近的讨论，未来的工作流将倾向于利用 **VLMs 和 LLMs**。
- **使用 Gemini Flash 2.0 构建的 YouTube 研究 Agent**：@composiohq 介绍的一篇教程详细说明了如何使用 **Gemini Flash 2.0** 构建 **YouTube 研究 Agent**，从而实现强大的视频搜索并在 Gmail 中创建草稿。
   - 这一集成使 @llama_index 成为简化视频研究工作流的基础工具。
- **AI 在企业自动化中的角色**：一篇文章强调，企业应专注于**适配 AI 技术**，以实现知识工作的自动化并有效解决业务挑战。
   - Cobus Greyling 建议，这应成为企业到 **2025 年** 的首要目标。
- **社交媒体应用 CrossPoster 发布**：今天发布了 **CrossPoster**，这是一款旨在利用 AI 将内容同步发布到 **Twitter**、**LinkedIn** 和 **BlueSky** 的应用，以实现最佳的社交媒体互动。
   - 该应用能智能识别个人及其账号，简化了跨平台的社交存在管理流程。
- **利用 GraphRAG 流水线获取数据洞察**：GraphRAG 流水线因其通过创建知识图谱将**原始数据**转化为可操作洞察的能力而受到关注。
   - 这种方法通过特定领域的知识增强了 **LLM 准确性**，从而实现更全面的搜索能力。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1337593289913929790)** (111 messages🔥🔥): 

> `OpenAI LLM 超时问题, LlamaIndex 中的 Agent 移交机制, Gemini 函数调用挑战, LlamaIndex 与 RAG 实现, AzureAI Search 自定义元数据字段` 


- **OpenAI LLM 超时问题**：成员们讨论了 OpenAI LLM 选项的超时设置如何被 retry_decorator 干扰，导致尽管设置了较高的超时时间，表现仍不一致。
   - 一位成员提到，即使提交了 bug 修复，Deepseek 在 60 秒后返回 200 OK 响应但包体为空，使问题变得复杂。
- **LlamaIndex 中的 Agent 移交机制**：针对 LlamaIndex 中 `can_handoff_to` 功能的有效性提出了疑虑，特别是当 Agent 在接收方 Agent 未响应的情况下传递控制权时。
   - 建议开启调试日志并利用 LlamaIndex 的回调处理器作为排查步骤。
- **Gemini 函数调用挑战**：论坛成员对 Gemini 中调试函数调用的困难表示沮丧，理由是类型注解问题和错误消息不清晰。
   - 尽管感到沮丧，一些人还是能够围绕现有的 bug 对工具输出进行工程化处理。
- **LlamaIndex 与 RAG 实现**：成员们讨论了在 RAG 场景中处理需要整个文档上下文的用户查询策略，例如识别摘要和向量索引。
   - 建议利用 Agent 或查询分类逻辑，以便根据具体查询需求更好地管理检索。
- **AzureAI Search 自定义元数据字段**：关于 AzureAI Search 中可过滤元数据字段的硬编码自定义问题被提出，特别是“作者”和“导演”等特定字段。
   - 成员们注意到 Azure 要求预先定义这些元数据字段，这虽然有限制，但也强调了有用文档字段的重要性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_deploy">GitHub - run-llama/llama_deploy: Deploy your agentic worfklows to production</a>: 将您的 Agent 工作流部署到生产环境。通过在 GitHub 上创建账号为 run-llama/llama_deploy 的开发做出贡献。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/using_citations.ipynb">anthropic-cookbook/misc/using_citations.ipynb at main · anthropics/anthropic-cookbook</a>: 展示使用 Claude 的一些有趣且有效方法的 Notebook/Recipe 集合。 - anthropics/anthropic-cookbook</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/rag/">Introduction to RAG - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/17764">fix gemini multi-turn tool calling by logan-markewich · Pull Request #17764 · run-llama/llama_index</a>: 存在两个问题：从 llama-index 转换为 Gemini 消息时，原始工具调用未包含在消息中；混合文本和工具调用的流式工具调用被破坏...</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/tracing_and_debugging/tracing_and_debugging/#tracing-and-debugging>)">Tracing and Debugging - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/17756)">run-llama/llama_index</a>: LlamaIndex 是构建基于您的数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/DocumentContextExtractor/">Contextual Retrieval With Llama Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/">Contextual Retrieval - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/openai/openai-python/blob/7193688e364bd726594fe369032e813ced1bdfe2/src/openai/_client.py#L82">openai-python/src/openai/_client.py at 7193688e364bd726594fe369032e813ced1bdfe2 · openai/openai-python</a>: OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai-python 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-python/blob/7193688e364bd726594fe369032e813ced1bdfe2/src/openai/_response.py#L265),">openai-python/src/openai/_response.py at 7193688e364bd726594fe369032e813ced1bdfe2 · openai/openai-python</a>: OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai-python 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/workflow/#llama_index.core.agent.workflow.AgentWorkflow>),">Workflow - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/agent/multi_agents/#a-detailed-look-at-the-workflow>).">Multi-agent workflows - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py#L404">llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py at 7391f302e18542c68b9cf5025afb510af4a52324 · run-llama/llama_index</a>: LlamaIndex 是构建基于您的数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/azureaisearch/">Azureaisearch - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/5b0067e146c919bc804803892aa2456842a80346/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py#L93)">llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py at 5b0067e146c919bc804803892aa2456842a80346 · run-llama/llama_index</a>: LlamaIndex 是构建基于您的数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 条消息): 

mrmirro: 💯
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1337540139349442590)** (14 messages🔥): 

> `Job Application Advice, Engineering Internships, Networking, Open Source Contribution, Canadian Engineering Competition` 


- **在求职中相信自己**：一位成员强调了自信的重要性，鼓励他人在求职过程中“无论别人怎么说，都要相信自己”。
   - 另一位成员补充道，*每个人都同样感到不确定*，并敦促在面对挑战时要坚持不懈。
- **寻找实习的挑战**：成员们讨论了目前工程实习*招聘机会匮乏*的现状，特别是在当地。
   - 一位用户分享了他们参加 **Canadian Engineering Competition** 的经历，并在初级设计组中排名前 6。
- **人脉拓展是关键**：一位成员强调，无论身在何处，*Networking* 都至关重要，并建议通过参加活动来增加曝光度。
   - 参与 **Open Source** 项目也被推荐为与同领域人士建立联系的一种方式。
- **通过竞赛增加曝光度**：为了积累经验，一位用户提到参加与其工程领域相关的*会议和竞赛*。
   - 他们强调了自己参加 **Canadian Engineering Competition** 的经历，反映了其对个人发展的投入。
- **纯粹为了乐趣而编码**：一位成员幽默地表示，他们享受纯粹为了乐趣而编码，这为严肃的求职讨论增添了轻松的一面。
   - 这反映了在追求职业目标与参与带来快乐的活动之间寻找平衡。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1337537511802540085)** (8 messages🔥): 

> `LibreChat Endpoint Issues, Curl Testing, Cohere API Versioning` 


- **LibreChat 在使用 Cohere API 时遇到困难**：一位成员指出，他们只能通过 **LibreChat** 的自定义端点（Custom Endpoint）访问 `https://api.cohere.ai/v1` 上的 **Cohere API**。
   - *CURL 测试正常*，表明问题出在 **LibreChat** 与 **API** 的集成上。
- **Curl 测试显示 API 可访问性**：另一位成员建议使用 **curl** 测试 **Cohere API**，结果确认其工作正常。
   - *如果 curl 正常*，那么问题很可能特定于 **LibreChat**，建议在其 **GitHub** 上提交 Issue。
- **LibreChat 使用过时的 Cohere API**：有成员指出 **LibreChat** 目前调用的是旧版 **API** 版本（v1），需要更新到 `/v2` 端点。
   - URL **https://api.cohere.com/v1** 镜像了 `https://api.cohere.ai/v1` 的功能，为当前用户提供了一个潜在的解决方案。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1337869971556012132)** (59 messages🔥🔥): 

> `Cohere Community Rules, Introduction Messages, AI Reasoning and Scalability, Working with Cohere staff, Discussion about Vapes` 


- **探索 Cohere 社区规则**：成员们讨论了 **Cohere** 社区规则，强调了在服务器内保持尊重和适当行为的重要性。
   - 讨论中涉及了这些规则如何应用于对话，以及在参与讨论时应注意的事项。
- **撰写自我介绍消息**：用户们合作起草了吸引人的新人自我介绍模板，重点突出了对 **AI** 的兴趣以及像“购买加拿大产品”这样的本地倡议。
   - 一个示例介绍强调了探索 **AI** 潜力和与社区进行有意义互动的愿望。
- **Cohere 的 AI 推理与可扩展性**：讨论转向了 **Cohere API** 的可扩展性，以及其员工在协作方面的可接触性。
   - 成员们表示有兴趣了解 **Cohere** 如何支持企业利用 **AI** 开发产品。
- **关于电子烟的询问**：一位成员发起了关于电子烟的苏格拉底式对话，促使大家探讨电子烟在当代社会中代表了什么。
   - 这引发了一段幽默的交流，苏格拉底（AI 扮演）表达了他对电子烟的陌生，并邀请大家就此对他进行“科普”。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1338632280096374805)** (1 条消息): 

> `Yu Su 的讲座，语言 Agent，MOOC 课程详情` 


- **Yu Su 关于语言 Agent 的讲座将于 PST 今天下午 4 点举行**：太平洋标准时间 (PST) 今天 **下午 4:00**，参加第 3 场讲座的直播，**Yu Su** 将在此介绍 *语言 Agent 的记忆、推理与规划*，链接见[此处](https://www.youtube.com/live/zvI4UN2_i-w)。
   - Yu Su 认为，当代的 AI Agent 的不同之处在于利用 **语言作为推理和沟通的载体**，他提出了一个概念框架并探讨了它们的核心能力。
- **Yu Su 对 NLP 的贡献**：Yu Su 是俄亥俄州立大学的 **杰出助理教授**，并共同领导 NLP 小组，做出了包括 Mind2Web、SeeAct、HippoRAG、LLM-Planner 和 MMMU 在内的重大贡献。
   - 他的工作获得了 CVPR 2024 **最佳学生论文奖** 和 ACL 2023 **杰出论文奖** 等认可。
- **即将发布的 MOOC 课程详情**：一份公告称 **MOOC 课程详情** 将很快发布，并感谢大家的耐心等待。
   - 课程相关的具体细节仍待定，鼓励参与者保持关注。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1337522928052338759)** (50 条消息🔥): 

> `课程注册、证书问题、项目协作、MOOC 课程、研究轨道 (Research Track) 注册` 


- **逾期报名者的课程注册**：用户询问是否可以参加 1 月份开始的 **LLM Agents MOOC**，确认可以通过填写报名表进行逾期注册。
   - 向参与者澄清，早期的课程版本并不是后续迭代的严格先修条件。
- **关于证书遗失的担忧**：几位用户反映没有收到证书，而同伴却收到了，这促使大家关注缺失的已完成 **证书声明表 (certificate declaration forms)**，这是必需的步骤。
   - 课程工作人员重申，必须完成此表格才能颁发证书，且需要单独提交。
- **项目协作咨询**：一位用户表达了合作项目的兴趣，同时确保遵守课程关于 MOOC 学生出版权的指南。
   - 课程工作人员承诺很快会发布更多课程细节，以解决关于项目框架和出版限制的疑虑。
- **MOOC 课程和要求更新**：参与者询问了测验之外的作业和项目的具体细节，工作人员提到详细信息将很快发布。
   - 鼓励用户在等待项目要求和评分政策的明确指南时保持耐心。
- **研究轨道 (Research Track) 注册咨询**：用户寻求关于如何注册 **研究轨道** 的说明，表示需要关于相应 Google 表单的指导。
   - 其他建议包括创建一个自动化 Agent 来简化证书流程并解决常见查询。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://]">未找到标题</a>: 未找到描述</li><li><a href="https://shamik-07.github.io/compound_ai_agentic_system/">🖥️ 使用 LLM 的复合 AI 系统 🖥️</a>: 一个执行多个复杂任务的复合 AI 系统</li><li><a href="https://x.com/Tonikprofik/status/1868729038665392472">Tony T (@Tonikprofik) 的推文</a>: 🚀作为我硕士论文研究的一部分。我参加了 2024 年秋季开设的加州大学伯克利分校关于大语言模型 (LLM) Agent 的 MOOC🧠✨这门课程深入探讨了 LLM Agent 的应用。他...
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1338119309310034052)** (12 条消息🔥): 

> `SFT vs DPO，负样本的重要性，LLM 训练挑战，Lecture 2 学习会议，时区讨论` 


- **SFT 和 DPO 解释了训练范式**：一位成员解释了 **Supervised Fine Tuning (SFT)** 如何仅使用正向示例，而 **Direct Preference Optimization (DPO)** 则引入了负面响应，强调了 DPO 中对错误响应的惩罚。
   - *错误响应*（通常结构良好）在 SFT 期间由于缺乏奖励模型，会导致其概率增加。
- **指令变更下模型响应的挑战**：讨论集中在一张幻灯片上，该幻灯片指出预期的修改指令 **x'** 会导致更差的响应 **y'**，强调了生成相关但语义不同的响应所面临的挑战。
   - 模型被要求在遵守修改后的 Prompt 要求的同时，生成准确但有缺陷的响应，这展示了一种艰难的平衡。
- **宣布 Lecture 2 学习会议**：一位成员宣布了关于 **Lecture 2: Learning to Reason with LLMs** 的学习会议，并邀请其他人通过提供的链接加入。
   - 参与者被鼓励准备讨论 **DeepSeek-R1 的 GRPO**，作为学习材料的一部分。
- **对学习会议时间的担忧**：一位参与者对学习会议的时间表示担忧，指出该时间恰好是 **英国时间凌晨 3:00**。
   - 这突显了国际成员可能面临的日程冲突。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1338327366186831897)** (13 条消息🔥): 

> `人工数据生成，Kolo 微调工具，合成数据创建的挑战，公共路线图更新` 


- **人工数据生成探索**：一位成员正在尝试 **人工数据生成**，并寻求将 PDF 和 Excel 文件等非结构化数据转换为 LLM 训练样本的工具。
   - 他们分享了一个关于该领域相关的合成数据生成方法论的 [YouTube 视频](https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO)。
- **用于微调的 Kolo 工具**：一位成员正在开发名为 **Kolo** 的工具，旨在简化模型的微调过程，尽管目前它还不支持数据创建。
   - 开发者正致力于在未来加入帮助生成训练数据的功能。
- **合成数据生成的挑战**：讨论强调了使用合成数据训练 LLM 的复杂性，指出从单个文档生成问题可能无法涵盖必要的比较性见解。
   - 一位成员表示，深度查询需要跨多个文档源的全面训练数据，以确保有效的学习。
- **关于路线图可用性的反馈**：针对之前讨论的公共路线图发出了提醒，一位成员询问了草案版本。
   - 已确认路线图正在审批过程中，一旦最终确定，应在本周末前在 GitHub 上共享。



**提到的链接**：<a href="https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO">Synthetic Data Generation and Fine tuning (OpenAI GPT4o or Llama 3)</a>：➡️ 获取完整脚本（及未来改进）的终身访问权限：https://Trelis.com/ADVANCED-fine-tuning ➡️ 一键微调和 LLM 模板：...

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1337628189341651078)** (43 条消息🔥): 

> `PR #2257 评审、GRPO 开发理念、Checkpointing 方法、PyTorch 依赖管理、支持 UV 和 Pip` 


- **PR #2257 需要更多关注**：一位成员分享了 [PR #2257](https://github.com/pytorch/torchtune/pull/2257) 以供评审，并指出该 PR 在其本地测试中运行正常，但希望能获得更多反馈。
   - 另一位成员对其进行了评审，称赞了这些更改，但提到了关于 Quantization 的 UX 担忧，并建议更新文档。
- **关于 GRPO 特性的两种理念**：讨论集中在是否保留或移除 GRPO 中的各种功能以进行简化，在易用性和代码整洁度之间取得平衡。
   - 几位成员发表了意见，倾向于移除不必要的代码，同时指出可能需要特定的特性，如 Activation Checkpointing。
- **理解 Checkpointing 机制**：分享了关于 torchtune 中 Resume 功能如何工作的细节，强调了 Checkpoint 路径的更新以及 `resume_from_checkpoint` 标志的重要性。
   - 成员们讨论了 Checkpointing 实践的影响，包括一个关于加载初始权重的不寻常工作流。
- **管理 PyTorch 依赖**：一位成员提议为不同版本的依赖添加安装选项，强调了 Nightly 版本和标准 Pip 支持可能带来的复杂性。
   - 讨论包括了对同时支持 Pip 和 UV 的考虑，权衡了扩展 `pyproject.toml` 的利弊。
- **支持 UV 用户**：承认了 UV 在用户中日益增长的普及度，并建议在传统 Pip 方法之外实现对其的支持。
   - 重点在于优先考虑 Pip，同时对 UV 的经过充分测试的补充保持开放态度，因为它在用户工作流中非常有用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://packaging.python.org/en/latest/specifications/dependency-groups/#dependency-groups">Dependency Groups - Python Packaging User Guide</a>：未找到描述</li><li><a href="https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index">Using uv with PyTorch | uv</a>：未找到描述</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning">Checkpointing in torchtune &mdash; torchtune main documentation</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/2363.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/2375">pyproject.toml wrong dev deps organization · Issue #2375 · pytorch/torchtune</a>：torchtune 在 [project.optional-dependencies] 定义了开发依赖 - https://github.com/pytorch/torchtune/blob/main/pyproject.toml#L47 而它们应该根据相关规范定义在 [dependency-groups]...</li><li><a href="https://github.com/pytorch/torchtune/blob/9da35c744adef777ec9b8d8620337ae5f0371dd5/recipes/configs/mistral/7B_full_ppo_low_memory.yaml#L78">torchtune/recipes/configs/mistral/7B_full_ppo_low_memory.yaml at 9da35c744adef777ec9b8d8620337ae5f0371dd5 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1452">Removing ao from pyproject.toml by ebsmothers · Pull Request #1452 · pytorch/torchtune</a>：TLDR：我们必须在“持续提供稳定、经过充分测试的 nightly 包的能力”与“为所有用户提供整洁的安装体验”之间做出选择。此 PR 遗憾地提议牺牲...</li><li><a href="https://github.com/pytorch/torchtune/pull/2349">Rework recipes section of README and simplify models ref by joecummings · Pull Request #2349 · pytorch/torchtune</a>：未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/553/files#diff-1534093f54f1b158be2da2b159e45561361e2479a7112e082232d3f21adc6a45">Grpo loss by kashif · Pull Request #553 · linkedin/Liger-Kernel</a>：摘要：添加了 GRPO chunked loss，修复了 issue #548。测试已完成：硬件类型：运行 make test 确保正确性，运行 make checkstyle 确保代码风格，运行 make test-convergence 确保...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/548">[RFC] Liger FlexChunkLoss: Grouping Loss · Issue #548 · linkedin/Liger-Kernel</a>：🚀 特性、动机和构想：如果我可以假设许多像 Group Relative Policy Optimization (GRPO) 这样的研究成果将会涌现，我认为我们可以引入一个 LigerFusedLinearGroupingBase .....
</li>
</ul>

</div>

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1337515018660483106)** (45 条消息🔥): 

> `模型选择菜单、AI Agents 与记忆、GPT4All 中的图像分析、PDF 处理与 Embedding、长期记忆解决方案` 


- **对模型选择菜单的批评**：成员们对 GPT4All 在发布 36 个版本后仍缺乏带有搜索功能的实用模型选择菜单表示担忧，并认为这可能只需要从其他平台复制粘贴即可实现。
   - 一名成员提议，既然 GPT4All 是一个开源产品，可以尝试通过贡献代码来解决缺失的功能。
- **探索用于长期记忆的 AI Agents**：成员们讨论了利用数据库实现长期记忆的 AI Agents 的潜力，并建议通过函数增强 LLM 的时间感知能力。
   - 2025 年被提及可能是 Agentic AI 取得突破性进展的转折点。
- **图像分析的局限性**：会议明确了目前 GPT4All 不支持图像分析，并建议用户探索其他具有此类能力的平台。
   - 为对图像相关任务感兴趣的用户推荐了 booruDatasetTagmanager 和 joycaption 等工具。
- **PDF 处理的最佳实践**：成员们讨论了将 PDF 等长文档进行 Embedding 和摘要并转化为 GPT4All 可用格式的有效策略。
   - 强调了需要正确处理从浏览器下载的内容，以确保在 Embedding 之前剔除无关信息。
- **选择合适的模型**：在被问及模型性能时，根据成员经验，推荐使用 Qwen2.5 和 Phi4，因为它们的效率优于包括 Mistral 在内的其他模型。
   - 强调了选择与应用集成的模型以提高用户友好性的重要性，并表示愿意帮助那些不熟悉如何从 Hugging Face 下载模型的人。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html#key-features)">GPT4All API Server - GPT4All</a>：GPT4All 文档 - 在你的硬件上高效运行 LLM</li><li><a href="https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683">速查表：掌握 ChatGPT API 中的 Temperature 和 Top_p</a>：大家好！我承认在整理这个指南时得到了 OpenAI 的帮助。但我认为我“协助”整理的内容可以极大地改善在应用和插件中使用 OpenAI 的效果和成本，特别是...</li><li><a href="https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-bindings/typescript/spec/chat-memory.mjs#L1"">gpt4all/gpt4all-bindings/typescript/spec/chat-memory.mjs at main · nomic-ai/gpt4all</a>：GPT4All：在任何设备上运行本地 LLM。开源且可商用。- nomic-ai/gpt4all</li><li><a href="https://huggingface.co/QuantFactory/gpt2-large-GGUF/tree/main">QuantFactory/gpt2-large-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main">TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/JabRef/jabref/wiki/GSoC-2024-%E2%80%90-AI%E2%80%90Powered-Summarization-and-%E2%80%9CInteraction%E2%80%9D-with-Academic-Papers)">主页</a>：用于管理 BibTeX 和 biblatex (.bib) 数据库的图形化 Java 应用程序 - JabRef/jabref</li><li><a href="https://docs.jabref.org/ai/local-llm)),">JabRef</a>：未找到描述
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1337521236921684014)** (21 messages🔥): 

> `tinygrad 测试与特性、基于 Web 的 LLM 演示、tinygrad 社区讨论、公司会议议程、ML 框架及 fp16 相关问题` 


- **在移动设备上测试 tinygrad**：围绕 **tinychat** 演示在移动端性能的讨论展开，指出由于缓存问题，**WebGPU** 在 iPhone 15 上运行失败，而 M1 Pro 用户发现它在 Safari 和 Chrome 中运行良好。
   - 用户表示需要进一步测试以提高兼容性，特别是关于移动设备上的 **WASM** 加载。
- **Tinygrad 公司结构澄清**：一位用户根据 Twitter 信息误以为 **tinygrad** 总部位于圣迭戈，但随后被纠正该公司是一家**完全远程办公的公司**。
   - 这引发了关于 **Ampere Altra** 处理器支持以及 tinygrad 后端加速能力的提问。
- **第 57 次公司会议议题公布**：定于周一举行的第 57 次会议议题包括**公司更新**、**CI 速度**、**tensor cores**，以及关于 **WebGPU** 和 **tinychat** 相关**悬赏任务 (bounties)** 的讨论。
   - 此类会议旨在提升内部流程的运行速度，同时回应社区对进行中项目的关注。
- **探索 ML 框架中的 fp16**：一位用户询问为什么大多数 ML 框架不只在 **fp16** 下运行，引发了关于其潜在劣势和性能限制的积极讨论。
   - George 对该询问做出了回应，要求其查阅 Discord 规则，这引发了更多关于在提问前应进行研究质量的评论。
- **PR 清晰度与数值精度**：讨论围绕一个实现脚本的 Pull Request (PR) 展开，该 PR 仍需要针对 **Hugging Face models** 增加更多特性和测试。
   - 社区强调了清晰的 PR 结构对代码审查的重要性，同时也承认量化模型中现有的**数值不准确性**是一个挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://chat.webllm.ai/">WebLLM Chat</a>: 暂无描述</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/13207080963/job/36872675131">Make tensor use UOp lshift/rshift; delete SimpleMathTrait · tinygrad/tinygrad@caafb50</a>: 你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - 使 tensor 使用 UOp lshift/rshift；删除 SimpleMathTrait · tinygrad/tinygrad@caafb50
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1338272144655650887)** (1 messages): 

> `DSPy, BERT 模型训练, Mistral 架构, 自动化文章处理` 


- **DSPy 彻底改变文章分类**：在受困于 **GPT-3.5** 的效果和 **GPT-4** 的高昂成本后，一位成员转向训练基于 **BERT** 的模型来有效分类收到的文章。
   - 今天标志着一个重要的里程碑，通过使用 **DSPy** 高度优化 Prompt，从每篇文章中提取了十几个字段，显著提升了性能。
- **Mistral 模型在成本效益方面表现出色**：该成员利用 **Miprov2**，以 **o3-mini** 作为教师模型，**Mistral Small 3** 作为学生模型，创建了一个既廉价又高效的全自动流程。
   - 该设置实现了每 24 小时对文章进行一次批处理，以 **50% 的成本减免**达到了超出预期的效果。
- **从坎坷起步到流线型工作流**：两年前，该成员面临手动数据分类的障碍，但现在每天可以毫不费力地处理 **100-200 篇文章**。
   - 最初的 BERT 模型设置为今天的自动化解决方案奠定了基础，展示了在能力和效率上的巨大增长。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1337906100560855100)** (2 条消息): 

> `Multi-Agent Systems, AI Agents and System Engineering, MASS Optimization Framework, Automation-oriented Sandbox Games` 


- **Multi-Agent Systems 在处理复杂任务方面表现出色**：正如 [MASS 框架](https://arxiv.org/abs/2502.02533) 中详述的，作为多个 Agent 运行的大语言模型（LLM）由于有效的交互和协作程序，在解决复杂任务方面表现优异。
   - 分析强调，有效的 **prompts** 和 **topologies** 对于设计稳健的 Multi-Agent Systems 至关重要。
- **AI Agents 需要动态系统工程技能**：用于评估 AI Agents 的静态基准测试无法反映动态系统工程所需的技能，因此提倡通过类似 [Factorio](https://arxiv.org/abs/2502.01492) 的面向自动化的沙盒游戏来训练 Agent。
   - 这种方法旨在培养管理复杂工程挑战所必需的专门推理和长周期规划能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.01492">Develop AI Agents for System Engineering in Factorio</a>: 前沿模型研究的不断进步正为 AI Agents 的广泛部署铺平道路。与此同时，全球对于在软件、制造、能源领域构建大型复杂系统的兴趣...</li><li><a href="https://arxiv.org/abs/2502.02533">Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies</a>: 作为多个相互交互和协作的 Agent 使用的大语言模型在解决复杂任务方面表现出色。这些 Agent 通过声明其功能的 prompts 进行编程...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1337552896589299762)** (3 条消息): 

> `Deep Research Abstractions, dspy Error Handling` 


- **关于简化 Deep Research 任务的咨询**：一位成员询问是否计划引入类似于 **deep research** 任务的抽象，并指出必要的组件可能已经可用。
   - *“你们计划引入抽象吗？”* 表达了对未来潜在功能的关注。
- **dspy 的 AttributeError**：一位成员报告在使用 **dspy** 时遇到错误 `AttributeError: module 'dspy' has no attribute 'HFClientVLLM'`。
   - 经过调查，他们注意到该功能在 **dspy 2.6** 中已被**弃用（deprecated）**，从而解决了他们的困惑。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1337534354296934562)** (6 条消息): 

> `RAFT templates for Llama, Compatibility issues with HF datasets, Converting complex objects to strings, Updating README with helper function, JSON lines formatted files` 


- **我们可以在 Llama 中使用自定义模板吗？**：一位成员询问是否可以使用类似于 **RAFT** 的自定义模板来为 **Llama** 生成合成数据集，或者是否需要特定的结构。
   - 这引发了关于 Llama 数据集要求灵活性的讨论。
- **HF datasets 可能会面临兼容性问题**：一位成员担心 **HF datasets** 可能会因为函数属性不同而始终存在兼容性问题。
   - 他们表示更倾向于将复杂对象转换为字符串，以便在数据集中使用。
- **HF datasets 中处理复杂对象的常见做法**：一位成员分享了一段代码片段，建议将不符合 schema 的复杂对象转换为字符串以用于 **HF datasets**。
   - 这种方法旨在简化非标准结构场景下的数据集加载。
- **提议更新 README 以添加辅助函数**：一位成员提议创建一个 Pull Request (PR)，用一个新的辅助函数更新 **README**，这将使其他用户受益。
   - 该建议得到了积极回应，另一位成员对提供的帮助表示感谢。
- **关于 JSON 文件格式的澄清**：一位成员澄清说使用的 **JSON** 文件没有问题，并指出 HF 需要 JSON lines 格式的文件。
   - 这强调了遵循预期文件格式对于成功加载数据集的重要性。


  

---


---


{% else %}


> 完整的逐频道详情已在邮件中截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}