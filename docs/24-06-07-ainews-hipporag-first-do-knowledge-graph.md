---
companies:
- alibaba
- openai
date: '2024-06-07T23:55:52.482883Z'
description: '**阿里巴巴**发布了全新的开源 **Qwen2** 模型，参数范围涵盖 **0.5B 到 72B**，在 MMLU 和 HumanEval
  等基准测试中取得了最先进（SOTA）的结果。研究人员引入了**稀疏自编码器（Sparse Autoencoders）**来解释 **GPT-4** 的神经活动，从而改进了特征表示。**HippoRAG**
  论文提出了一种受海马体启发的检索增强方法，利用知识图谱和个性化 PageRank 实现高效的多跳推理。**逐步内化（Stepwise Internalization）**等新技术使大语言模型能够进行隐式思维链推理，提升了准确性和速度。**思想缓冲区（Buffer
  of Thoughts, BoT）**方法在显著降低成本的同时提高了推理效率。此外，研究人员还展示了一种新型的可扩展、**无矩阵乘法（MatMul-free）**大语言模型架构，其在十亿参数规模上可与最先进的
  Transformer 架构相媲美。“**单步多跳检索**”被强调为检索速度和成本方面的关键进展。'
id: b349be5b-3e3c-4042-9eb9-989b14a9cbac
models:
- qwen-2
- gpt-4
- hipporag
original_slug: ainews-hipporag-first-do-knowledge-graph
people:
- rohanpaul_ai
- omarsar0
- nabla_theta
- huybery
title: 'HippoRAG：首先，构建知识图谱。


  *(注：这句话模仿了医学界的希波克拉底誓言中的名言 "First, do no harm"——“首先，不要伤害”，这里将 "no harm" 巧妙地替换成了 "know(ledge)
  Graph"——“知识图谱”。)*'
topics:
- knowledge-graphs
- personalized-pagerank
- multi-hop-retrieval
- chain-of-thought
- implicit-reasoning
- sparse-autoencoders
- model-interpretability
- model-efficiency
- model-architecture
- fine-tuning
- reinforcement-learning
---

<!-- buttondown-editor-mode: plaintext -->**记忆是 LLM 所需的一切。**

> 2024年6月6日至6月7日的 AI 新闻。
我们为您检查了 7 个 Reddit 子版块、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务端（**409** 个频道和 **3133** 条消息）。
预计节省阅读时间（以 200wpm 计）：**343 分钟**。

> 热烈欢迎加入 [TorchTune Discord](https://discord.com/channels/1216353675241590815)。提醒一下，我们确实会考虑在 Reddit/Discord 追踪列表中增加内容的请求（我们会拒绝 Twitter 的增加请求——可定制的 Twitter 通讯即将推出！我们知道这让大家久等了）。

随着关于[记忆初创公司](https://x.com/swyx/status/1776698202147996050)以及长期运行的 Agent/[个人 AI](https://twitter.com/swyx/status/1776448691123241288) 领域融资增加的传闻，我们看到人们对高精度/高召回率记忆实现的兴趣日益浓厚。

今天的论文虽然不如 [MemGPT](https://arxiv.org/abs/2310.08560) 那么出色，但它预示了人们正在探索的方向。尽管我们不太推崇用自然智能模型来解释人工智能，但 [HippoRAG 论文](https://arxiv.org/abs/2405.14831)借鉴了“海马体记忆索引理论（hippocampal memory indexing theory）”，从而实现了一种有用的 Knowledge Graphs 和 “Personalized PageRank” 方案，这可能具有更坚实的实证基础。

 
![image.png](https://assets.buttondown.email/images/9bc6e9dc-3a08-4368-92a0-0c6e2a861ac3.png?w=960&fit=max)
 

讽刺的是，对该方法论最好的解释来自 [Rohan Paul 的一个推文串](https://x.com/rohanpaul_ai/status/1798664784130535789?utm_source=ainews&utm_medium=email)（我们不确定他是如何每天产出这么多内容的）：

 
![image.png](https://assets.buttondown.email/images/d3d8ddb6-856f-4cac-9821-9d302d748f36.png?w=960&fit=max)
 

与慢 10 倍以上且更昂贵的同类方法相比，**Single-Step, Multi-Hop retrieval** 似乎是关键的制胜点：

 
![image.png](https://assets.buttondown.email/images/1db63532-6ef7-42aa-ae40-4f667d1d3c93.png?w=960&fit=max)
 

第 6 节对当前在 LLM 系统中模拟记忆的技术进行了有用且简洁的文献综述。

 
![image.png](https://assets.buttondown.email/images/5617a9dc-52a6-4ff4-9a28-d4053a2c8165.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果。我们正在尝试使用 Haiku 进行聚类和 Flow Engineering。

**新的 AI 模型与架构**

- **来自阿里巴巴的新 SOTA 开源模型**：[@huybery](https://twitter.com/huybery/status/1798747031185559921) 宣布发布阿里巴巴的 Qwen2 模型，参数规模从 0.5B 到 72B 不等。这些模型在 29 种语言上进行了训练，并在 MMLU（72B 为 84.32）和 HumanEval（72B 为 86.0）等基准测试中达到了 SOTA 性能。除 72B 外，所有模型均采用 Apache 2.0 许可证。
- **用于解释 GPT-4 的 Sparse Autoencoders**：[@nabla_theta](https://twitter.com/nabla_theta/status/1798763600741585066) 介绍了一种新的 Sparse Autoencoders (SAEs) 训练栈，用于解释 GPT-4 的神经活动。该方法显示出前景，但目前仅能捕捉到一小部分行为。它消除了特征收缩（feature shrinking），直接设置 L0，并在 MSE/L0 前沿表现良好。
- **受海马体启发的检索增强**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798664784130535789) 概述了 HippoRAG 论文，该论文**模拟新皮层和海马体以实现高效的检索增强**。它从语料库中构建知识图谱，并使用个性化 PageRank 在单步内进行多跳推理，性能优于 SOTA RAG 方法。
- **隐式思维链推理**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1798834917465596414) 描述了关于**教导 LLM 隐式进行思维链（chain-of-thought）推理，而无需显式中间步骤**的研究。所提出的 Stepwise Internalization 方法在微调过程中逐渐移除 CoT token，使模型能够以高准确度和速度进行隐式推理。
- **通过 Buffer of Thoughts 增强 LLM 推理**：[@omarsar0](https://twitter.com/omarsar0/status/1799113545696567416) 分享了一篇提出 Buffer of Thoughts (BoT) 的论文，旨在提高 LLM 的推理准确性和效率。BoT 存储了从问题解决中提取的高级思维模板，并进行动态更新。它在多项任务上达到了 SOTA 性能，而成本仅为多查询提示（multi-query prompting）的 12%。
- **可扩展的无矩阵乘法 LLM，性能媲美 SOTA Transformer**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1799122826114330866) 分享了一篇论文，声称创建了首个在十亿参数规模下可与 SOTA Transformer 竞争的可扩展 MatMul-free LLM。该模型用三值操作（ternary ops）取代了 MatMul，并使用了 Gated Recurrent/Linear Units。作者构建了一个定制的 FPGA 加速器，在 13W 功耗下处理模型的吞吐量超过了人类阅读速度。
- **通过正交低秩自适应加速 LoRA 收敛**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1799127301185040729) 分享了一篇关于 OLoRA 的论文，该方法在保持效率的同时加速了 LoRA 的收敛。OLoRA 通过 QR 分解对自适应矩阵进行正交初始化，在多种 LLM 和 NLP 任务上表现优于标准 LoRA。

**多模态 AI 与机器人技术进展**

- **用于细粒度视觉理解的 Dragonfly 视觉语言模型**：[@togethercompute](https://twitter.com/togethercompute/status/1798789579622977732) 推出了利用多分辨率编码和放大补丁选择（zoom-in patch selection）的 Dragonfly 模型。Llama-3-8b-Dragonfly-Med-v1 在医学影像方面的表现优于 Med-Gemini。
- **用于视频理解与生成的 ShareGPT4Video**：[@_akhaliq](https://twitter.com/_akhaliq/status/1798923975285977416) 分享了 ShareGPT4Video 系列，以促进 LVLM 中的视频理解和 T2VM 中的视频生成。它包括一个包含 4 万条 GPT-4 标注的视频数据集、一个卓越的任意视频标注器，以及一个在 3 个视频基准测试中达到 SOTA 的 LVLM。
- **使用 Nvidia Jetson Orin Nano 的开源机器人演示**：[@hardmaru](https://twitter.com/hardmaru/status/1799039759429615761) 强调了开源机器人的潜力，分享了一个使用 Nvidia Jetson Orin Nano 8GB 板卡、Intel RealSense D455 摄像头和麦克风以及 Luxonis OAK-D-Lite AI 摄像头的机器人视频演示。

**AI 工具与平台更新**

- **用于高吞吐量 Embedding 服务的 Infinity**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1799075742783091163) 发现 Infinity 在通过 REST API 提供向量 Embedding 服务方面表现出色，支持多种模型/框架、快速推理后端、动态批处理，并能轻松与 FastAPI/Swagger 集成。
- **Amazon SageMaker 上的 Hugging Face Embedding Container**：[@_philschmid](https://twitter.com/_philschmid/status/1799093702679228664) 宣布 HF Embedding Container 在 SageMaker 上正式可用，改进了 RAG 应用的 Embedding 创建，支持流行架构，使用 TEI 实现快速推理，并允许部署开源模型。
- **Qdrant 与 Neo4j APOC 过程的集成**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1798988390471442757) 宣布 Qdrant 与 Neo4j 的 APOC 过程全面集成，为图数据库应用带来了先进的向量搜索功能。

**AI 模型的基准测试与评估**

- **MixEval 基准测试与 Chatbot Arena 的相关性达 96%**：[@_philschmid](https://twitter.com/_philschmid/status/1799007110715543690) 介绍了 MixEval，这是一个结合了现有基准测试与真实世界查询的开放基准。MixEval-Hard 是一个具有挑战性的子集。其运行成本为 0.6 美元，与 Arena 的相关性达 96%，并使用 GPT-3.5 作为解析器/裁判。阿里巴巴的 Qwen2 72B 在开源模型中位居榜首。
- **MMLU-Redux：重新标注的 MMLU 问题子集**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1798904604375265724) 创建了 MMLU-Redux，这是 MMLU 中跨 30 个学科的 3,000 个问题的子集，旨在解决诸如病毒学中 57% 的问题包含错误等问题。该数据集已公开。
- **质疑 MMLU 的持续相关性**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1798904604375265724) 质疑鉴于 SOTA 开源模型的饱和，我们是否已经完成了使用 MMLU 评估 LLM 的阶段，并提议将 MMLU-Redux 作为替代方案。
- **发现开源 LLM 的缺陷**：[@JJitsev](https://twitter.com/JJitsev/status/1799025453522649259) 从他们的 AIW 研究中得出结论，**当前的 SOTA 开源 LLM（如 Llama 3、Mistral 和 Qwen）在基础推理方面存在严重缺陷**，尽管它们声称拥有强大的基准测试表现。


**关于 AI 的讨论与观点**

- **谷歌关于人工超智能（ASI）开放性的论文**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1798899833010327577) 分享了一篇谷歌论文，认为实现 AI 开放性的要素已经具备，这对于 ASI 至关重要。该论文提供了开放性的定义、通过基础模型实现的路径，并探讨了安全影响。
- **关于微调可行性的辩论**：[@HamelHusain](https://twitter.com/HamelHusain/status/1799081797768360359) 分享了 Emmanuel Kahembwe 关于“为什么微调已死”的演讲，引发了讨论。虽然不像演讲者那样悲观，但 @HamelHusain 认为这个演讲很有趣。
- **Yann LeCun 谈 AI 监管**：在一系列推文中（[1](https://twitter.com/ylecun/status/1798839294930379209), [2](https://twitter.com/ylecun/status/1798861767906570602), [3](https://twitter.com/ylecun/status/1798896955705487457)），@ylecun 主张监管 AI 应用而非技术，并警告说监管基础技术并让开发者为滥用负责将扼杀创新、停止开源，且这些做法是基于不可信的科幻场景。
- **关于 AI 时间线和进展的辩论**：Leopold Aschenbrenner 在 @dwarkesh_sp 播客中讨论他关于 AI 进展和时间线的论文（[由一位用户总结](https://twitter.com/AlphaSignalAI/status/1798744310621597896)），引发了大量辩论，观点从称其为 AI 能力爆炸的重要案例到批评其依赖于持续指数增长的假设不等。

**其他**

- **Perplexity AI 在 NBA 总决赛期间投放广告**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1798892187545149521) 指出 **首个 Perplexity AI 广告在 NBA 总决赛第一场期间播出**。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1798897613070290945) 分享了该视频剪辑。
- **Yann LeCun 谈指数趋势与 S 型曲线（Sigmoids）**：[@ylecun](https://twitter.com/ylecun/status/1799064075487572133) 认为 **每一个指数趋势最终都会经过一个拐点并饱和成 S 型曲线（Sigmoid）**，因为动力学方程中的摩擦项会变得占主导地位。持续指数增长需要范式转移，正如在摩尔定律中所见。
- **John Carmack 谈 Quest Pro**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1799083581974749264) 分享说他 **曾努力尝试彻底砍掉 Quest Pro**，因为他认为这在商业上会失败，并会分散团队在更有价值的大众市场产品上的精力。
- **FastEmbed 库新增嵌入类型**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1798721749103915419) 宣布 **FastEmbed 0.3.0 增加了对图像嵌入 (ResNet50)、多模态嵌入 (CLIP)、延迟交互嵌入 (ColBERT) 和稀疏嵌入的支持**。
- **笑话与梗**：分享了各种笑话和梗，包括一个在没有 Flash Attention 的情况下输出胡言乱语的 GPT-4 GGUF（[链接](https://twitter.com/rohanpaul_ai/status/1799078458552967246)），@karpathy 针对 DeepMind 的 SAE 论文发布的 llama.cpp 更新（[链接](https://twitter.com/karpathy/status/1798920127779660129)），以及对 LLM 炒作周期和夸大言论的评论（[例子](https://twitter.com/corbtt/status/1798816753981788619)）。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！


**中国 AI 模型**

- **可灵 (KLING) 模型生成视频**：中国的可灵 (KLING) AI 模型生成了几段人们吃面条或汉堡的视频，定位为 OpenAI SORA 的竞争对手。用户讨论了该模型的可访问性及其潜在影响。 ([视频 1](https://v.redd.it/yjd6w6dg0z4d1), [视频 2](https://v.redd.it/8fwsasj08z4d1), [视频 3](https://v.redd.it/6p7hf3h02z4d1), [视频 4](https://v.redd.it/72ei2rjh235d1))
- **Qwen2-72B 语言模型发布**：阿里巴巴在 Hugging Face 上发布了 Qwen2-72B 中文语言模型。根据对比图，它在[多项基准测试中超越了 Llama 3](https://www.reddit.com/gallery/1d9mi13)。[官方发布博客](https://qwenlm.github.io/blog/qwen2/)链接也已提供。

**AI 能力与局限性**

- **开源 vs 闭源模型**：[截图展示了 Bing AI 和 CoPilot 等闭源模型如何限制某些话题的信息](https://www.reddit.com/gallery/1d9mi8t)，强调了开源替代方案的重要性。Andrew Ng [认为 AI 监管应侧重于应用](https://x.com/AndrewYNg/status/1788648531873628607)，而非限制开源模型的开发。
- **AI 作为“外星智能”**：Steven Pinker [认为 AI 模型是一种“外星智能”](https://v.redd.it/0kskj1mou15d1)，我们正在对其进行实验，且人类大脑可能与大语言模型相似。

**AI 研究与进展**

- **从 GPT-4 中提取概念**：OpenAI [关于使用稀疏自编码器识别 GPT-4 神经网络中可解释模式的研究](https://openai.com/index/extracting-concepts-from-gpt-4/)，旨在使模型更可靠、更具可控性。
- **针对 AI 的反垄断调查**：[微软和 Nvidia 因其 AI 相关业务举措正面临美国反垄断调查](https://news.bloomberglaw.com/antitrust/microsoft-nvidia-to-face-us-antitrust-probes-over-moves-in-ai)。
- **极端权重量化**：[研究通过极端权重量化实现了比原始版本性能更好的 Stable Diffusion v1.5 模型，体积缩小了 7.9 倍](https://snap-research.github.io/BitsFusion/)。

**AI 伦理与监管**

- **AI 审查担忧**：[Bing AI 拒绝提供某些信息的截图](https://www.reddit.com/gallery/1d9meap)引发了关于 AI 审查和信息开放获取重要性的讨论。 
- **测试 AI 的选举风险**：Anthropic [讨论了在其 AI 系统中测试和缓解潜在选举相关风险的努力](https://www.anthropic.com/news/testing-and-mitigating-elections-related-risks)。
- **使用社交媒体数据的批评**：[使用 Facebook 和 Instagram 帖子训练 AI 模型的计划面临批评](https://www.yahoo.com/tech/plans-facebook-instagram-posts-train-132129112.html)。

**AI 工具与框架**

- **用于角色扮演的 Higgs-Llama-3-70B**：针对角色扮演优化的 Llama-3 微调版本 [已在 Hugging Face 上发布](https://huggingface.co/bosonai/Higgs-Llama-3-70B)。
- **移除 LLM 审查**：Hugging Face 博客文章介绍了一种[用于移除语言模型审查的“abliteration”方法](https://huggingface.co/blog/mlabonne/abliteration)。
- **Atomic Agents 库**：新的[开源库，用于构建支持本地模型的模块化 AI Agent](https://www.reddit.com/r/LocalLLaMA/comments/1d9dw4s/atomic_agents_new_opensource_library_to_build_ai/)。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**1. LLM 进展与优化挑战**：

- **Meta 的 [Vision-Language Modeling 指南](https://arxiv.org/abs/2405.17247)** 提供了 VLM 的全面概述，包括训练过程和评估方法，帮助工程师更好地理解视觉到语言的映射。
- **DecoupleQ** 来自字节跳动，旨在利用新的量化方法大幅提升 LLM 性能，承诺 **7 倍压缩率**，尽管进一步的速度基准测试仍有待观察 ([GitHub](https://github.com/bytedance/decoupleQ))。
- **GPT-4o 即将推出的功能** 包括为 ChatGPT Plus 用户提供的新语音和视觉功能，以及为 Alpha 用户提供的实时聊天。 [在 OpenAI 的推文中阅读相关内容](https://x.com/OpenAI/status/1790130708612088054)。
- **高效推理和训练技术** 如 `torch.compile` 加速了 SetFit 模型，证实了在 PyTorch 中尝试优化参数以获得性能提升的重要性。
- **FluentlyXL Final** 来自 HuggingFace，在美学和光影方面引入了实质性改进，提升了 AI 模型的输出质量 ([FluentlyXL](https://huggingface.co/fluently/Fluently-XL-Final))。

**2. 开源 AI 项目与资源**：

- **TorchTune** 使用 PyTorch 简化了 LLM 的微调，并在 [GitHub](https://github.com/pytorch/torchtune) 上提供了一个详细的仓库。欢迎通过单元测试为 mqa/gqa 配置 `n_kv_heads` 等贡献。
- **Unsloth AI 的 Llama3 和 Qwen2 训练指南** 提供了实用的 Colab 笔记本和高效的预训练技术，以优化 VRAM 使用 ([Unsloth AI 博客](https://unsloth.ai/blog/contpretraining))。
- **LlamaIndex 中的动态数据更新** 通过定期索引刷新和元数据过滤器，帮助保持检索增强生成 (RAG) 系统的实时性，详见 [LlamaIndex 指南](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=document+managem)。
- **AI 视频生成和 Vision-LSTM** 技术探索了动态序列生成和图像读取能力 ([Twitter 讨论](https://x.com/rowancheung/status/1798738564735554047))。
- **TopK Sparse Autoencoders** 在 GPT-2 Small 和 Pythia 160M 上进行了有效训练，无需在磁盘上缓存激活值，有助于特征提取 ([OpenAI 发布](https://x.com/norabelrose/status/1798985066565259538))。

**3. AI 模型实施中的实际问题**：

- **LangChain 中的 Prompt Engineering** 面临重复步骤和提前停止的问题，敦促用户寻找修复方案 ([GitHub issue](https://github.com/langchain-ai/langchain/issues/16263))。
- **Automatic1111 在图像生成任务中的高 VRAM 消耗** 导致了显著的延迟，凸显了对内存管理解决方案的需求 ([Stability.ai 聊天](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/))。
- **Qwen2 模型故障排除** 揭示了乱码输出的问题，通过启用 flash attention 或使用正确的预设已修复 ([LM Studio 讨论](https://discord.com/channels/1110598183144399058/1111649100518133842))。
- **Mixtral 8x7B 模型误区纠正**：Stanford CS25 澄清其包含 256 个专家，而不仅仅是 8 个 ([YouTube](https://youtu.be/RcJ1YXHLv5o?si=gjcu--95HafBwEaT))。

**4. AI 监管、安全与伦理讨论**：

- **吴恩达 (Andrew Ng) 对 AI 监管的担忧** 呼应了全球关于 AI 创新受阻的辩论；与俄罗斯 AI 政策讨论的对比揭示了在开源和伦理 AI 方面的不同立场 ([YouTube](https://www.youtube.com/watch?v=5L2YAIk0vSc&ab_channel=TheTelegraph))。
- **Leopold Aschenbrenner 从 OAI 离职** 引发了关于 AI 安全措施重要性的激烈辩论，反映了在 AI 保护方面的分歧意见 ([OpenRouter 讨论](https://discord.com/channels/1091220969173028894/1094454198688546826))。
- **艺术软件中的 AI 安全**：Adobe 要求访问所有作品（包括签署 NDA 的项目），这促使注重隐私的用户建议使用 Krita 或 Gimp 等替代软件 ([Twitter 线程](https://x.com/SamSantala/status/1798292952219091042))。

**5. 社区工具、技巧与协作项目**：

- **Predibase 工具与热烈反馈**：LoRAX 在高性价比 LLM 部署方面表现出色，尽管存在邮件注册的小插曲 ([Predibase 工具](https://docs.predibase.com/user-guide/examples/rag))。
- **用于递归分析的 WebSim.AI**：AI 工程师分享了使用 WebSim.AI 进行递归模拟的经验，并对源自幻觉的有价值指标进行了头脑风暴 ([Google 表格](https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl3u06IrKWoy3tlrCWlMRXDxvoDCY))。
- **Modular 的 MAX 24.4 更新** 引入了新的 Quantization API 并支持 macOS，通过显著降低延迟和内存占用增强了 Generative AI 流水线 ([博客文章](https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos))。
- **GPU 冷却与电源解决方案** 讨论了设置 Tesla P40 及类似硬件的创新方法，并提供了实用指南 ([GitHub 指南](https://github.com/JingShing/How-to-use-tesla-p40))。
- **实验与学习资源**：由 tcapelle 提供，包括用于微调和提高效率的实用笔记本及 GitHub 资源 ([Colab 笔记本](https://wandb.me/connections))。

---

# 第 1 部分：Discord 高层摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **RAG 改进与表格转换尝试**：由于对 [Marker](https://github.com/VikParuchuri/marker/tree/master) 格式化 Markdown 表格的效果不满，引发了关于通过 Fine-tuning 该工具以优化输出的讨论。同时，也在探索如 [img2table](https://github.com/xavctn/img2table) 等替代的表格提取工具。

- **Predibase 为 Python 专家提供预测文本**：用户对 Predibase 的额度与工具表现出浓厚兴趣，其中 [LoRAX](https://predibase.github.io/lorax/) 因其高性价比、高质量的 LLM 部署能力而脱颖而出。确认函和邮箱注册环节普遍存在小故障，相关求助已定向至注册邮箱。

- **关于深度 LLM 理解的公开讨论**：来自 tcapelle 的帖子深入探讨了 LLM Fine-tuning，并分享了 [slides](http://wandb.me/llm-finetune-slides) 和 [notebooks](https://wandb.me/connections) 等资源。此外，一份在 [NVIDIA GTC 演讲](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62348/) 中分享的研究强调了通过 Pruning 策略来精简 LLM 的方法。

- **Cursor 代码编辑器吸引工程师关注**：AI 代码编辑器 [Cursor](https://www.cursor.com/) 利用来自 OpenAI 及其他 AI 服务的 API keys，因其代码库索引（codebase indexing）和代码补全（code completion）方面的改进而获得认可，甚至吸引用户从 GitHub Copilot 迁移。

- **Modal GPU 使用与大量 Gist 分享**：评估了 Modal 的 VRAM 使用情况和 A100 GPUs，同时分享了 [Pokemon 卡片描述的 Gist](https://gist.github.com/sroecker/5c3a9eb1fd0c898e4119b89ff1095038) 以及 [Whisper 适配技巧](https://gist.github.com/aksh-at/fb14599c28a3bc0f907ea45398a7651d)。会上提到了 GPU 可用性的不稳定性，并注意到缺乏显示队列状态的 Dashboard。

- **通过 Vector 和 OpenPipe 学习**：讨论内容包括使用 VectorHub 的 RAG 相关内容构建 Vector 系统，[OpenPipe 博客](https://openpipe.ai/blog) 上的文章也因其对讨论的贡献而受到关注。

- **Fine-tuning 工具与数据的难题**：用户正在解决课程会议录像的下载问题；同时，受合成数据集（synthetic datasets）涌入的启发，Bulk 工具的开发正在加速。有人在 Replicate 演示期间寻求关于本地 Docker 空间困境的帮助，但在聊天记录中尚未找到解决方案。

- **LLM Fine-tuning 修复进行中**：围绕 Fine-tuning 复杂性展开了热烈讨论，解决了合并 Lora 模型分片异常的问题，并提出了如使用 Mistral Instruct 模板进行 DPO Fine-tuning 等偏好。有趣的是，Axolotl 中 Token 空间组装的输出差异引起了关注，对话倾向于调试和潜在的解决方案。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Starship 成功翱翔并溅落**：SpaceX 的 **Starship** 测试飞行取得成功，在两大洋着陆，引起了工程界的关注；根据 [官方更新](https://www.perplexity.ai/page/Starship-Test-4-QCcbPm1tQay1u.pc9bAXVg)，在墨西哥湾和印度洋的成功溅落标志着该项目的显著进展。

- **精彩的咖喱对比**：对国际美食感兴趣的工程师分析了日本、印度和泰国咖喱的区别，指出了独特的香料、草药和成分；一份 [详细的细分报告](https://www.perplexity.ai/page/Comparing-Asian-Curry-bpXXIu9gTiKcWtcFzxKizw) 被传阅，提供了关于每种咖喱的历史起源和典型食谱的见解。

- **Perplexity 促销活动令参与者困惑**：期待 Perplexity AI “The Know-It-Alls” 广告带来重大更新的用户感到失望；相反，这只是一个宣传视频，让许多人觉得这更像是一个噱头而非实质性的揭晓，正如在 [General 频道](https://discord.com/channels/1047197230748151888/1047649527299055688/1248359965891100763) 中讨论的那样。

- **AI 社区交流 Claude 3 与 Pro Search**：关于 **Pro Search** 和 **Claude 3** 等不同 AI 模型的讨论非常热烈；模型偏好、搜索能力和用户体验是热门话题，同时还讨论了从 Perplexity Labs 中移除 **Claude 3 Haiku** 的事宜。

- **API 频道中的 llava 哀叹与 Beta 忧郁**：API 用户询问了 **llava** 模型的集成情况，并对新来源 **Beta 测试** 似乎不公开的性质表示不满，表现出对 Perplexity 团队提高透明度和加强沟通的强烈渴望。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Electrifying Enhancement with FluentlyXL**:
  备受期待的 **FluentlyXL Final** 版本现已发布，承诺在美学和光影方面带来实质性增强，详情见其 [官方页面](https://huggingface.co/fluently/Fluently-XL-Final)。此外，关注环保的技术爱好者可以探索新的 **Carbon Footprint Predictor**，以衡量其项目的环境影响 ([Carbon Footprint Predictor](https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor))。

- **Innovations Afoot in AI Model Development**:
  初出茅庐的 AI 工程师正在探索模型开发不同领域中快速演进的可能性，从 **SimpleTuner** 0.9.6.2 版本中新增的 MoE 支持 ([SimpleTuner on GitHub](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2))，到一个基于 TensorFlow 的 **ML Library**，其源代码和文档可在 [GitHub](https://github.com/NoteDance/Note) 上进行同行评审。

- **AI's Ascendancy in Medical and Modeling Musings**:
  最近的一段 YouTube 视频深入探讨了 **genAI** 在医学教育中日益增长的作用，强调了 **Anki** 和 **genAI 驱动的搜索** 等工具的优势 ([AI in Medical Education](https://youtu.be/kZMcNCV_RXk))。在开源领域，**TorchTune** 项目因促进 LLM 的 fine-tuning 而引起关注，相关探索在 [GitHub](https://github.com/pytorch/torchtune) 上有详细叙述。

- **Collider of Ideas in Computer Vision**:
  爱好者们正在汇集知识，为 Vision Language Models (VLMs) 创建有价值的应用，社区成员分享了新的 Hugging Face Spaces 应用 **Model Explorer** 和 **HF Extractor**，这些工具对 VLM 应用开发至关重要 ([Model Explorer](https://huggingface.co/spaces/dwb2023/model_explorer2), [HF Extractor](https://huggingface.co/spaces/dwb2023/hf_extractor)，以及相关的 [YouTube 视频](https://www.youtube.com/watch?v=w67fQ_-8hq0))。

- **Engaging Discussions and Demonstrations**:
  LLM 的多节点 fine-tuning 是一个讨论热点，引发了对一篇 [关于 Vision-Language Modeling 的 arXiv 论文](https://arxiv.org/abs/2405.17247) 的分享；同时，**Diffusers GitHub 仓库** 因其 text-to-image 生成脚本而被重点关注，这些脚本也可用于模型 fine-tuning ([Diffusers GitHub](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/))。此外，还传阅了一篇提供原生 PyTorch 优化见解的博客文章，以及为渴望从零开始训练模型的用户准备的 [训练示例 notebook](https://github.com/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AI Newbies Drowning in Options**: 一位社区成员表达了对海量待探索 AI 模型的*兴奋与不知所措*，这捕捉到了许多该领域新入局者的共同心声。
- **ControlNet's Speed Bump**: 用户 _arti0m_ 报告了 ControlNet 出现的意外延迟，导致图像生成时间长达 **20 分钟**，这与预期的速度提升背道而驰。
- **CosXL's Broad Spectrum Capture**: 来自 Stability.ai 的新 CosXL 模型拥有更广阔的色调范围，能够生成从“漆黑”到“纯白”具有更好对比度的图像。点击 [此处](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/) 了解更多。
- **VRAM Vanishing Act**: 关于 Automatic1111 web UI 内存管理挑战的讨论浮出水面，该界面似乎过度占用 VRAM，影响了图像生成任务的性能。
- **Waterfall Scandal Makes Waves**: 一场关于中国病毒式传播的假瀑布丑闻引发了激烈辩论，进而引发了对其环境和政治影响的更广泛讨论。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Adapter 重载引发关注**：成员们在尝试使用模型 Adapter 继续训练时遇到问题，特别是在使用 `model.push_to_hub_merged("hf_path")` 时，Loss 指标出现意外飙升，这指向了保存或加载过程中可能存在的处理不当。

- **通过特殊技术增强 LLM 预训练**：[Unsloth AI 的博客](https://unsloth.ai/blog/contpretraining) 概述了使用 Llama3 等 LLM 对韩语等语言进行持续预训练（Continued Pretraining）的效率，承诺减少 VRAM 占用并加速训练，并提供了一个实用的 [Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 用于实际应用。

- **Qwen2 模型开启扩展语言支持**：宣布支持 **Qwen2 模型**，该模型拥有高达 128K 的上下文长度并覆盖 27 种语言，[Daniel Han 在 Twitter](https://x.com/danielhanchen/status/1798792569507418231) 上分享了微调资源。

- **探索 Grokking**：讨论深入探讨了一种被称为 "Grokking" 的新识别出的 LLM 性能阶段，社区成员引用了一场 [YouTube 辩论](https://www.youtube.com/watch?v=QgOeWbW0jeA) 并提供了支持性研究链接以供进一步探索。

- **纠正 NVLink VRAM 误区**：对 NVIDIA NVLink 技术进行了澄清，成员解释说 NVLink 不会将 VRAM 合并为一个单一池，打破了关于其能够扩展计算可用 VRAM 容量的误解。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 简化 CUDA**：Triton 语言因其使用 Grid 语法 (`out = kernel[grid](...)`) 启动 CUDA Kernel 的简便性，以及在启动后轻松访问 PTX 代码 (`out.asm["ptx"]`) 的能力而受到认可，为 CUDA 开发者提供了更精简的工作流。

- **TorchScript 中的 Tensor 问题与 PyTorch Profiling**：在 **TorchScript** 中无法使用 `view(dtype)` 进行 Tensor 转换，这让寻求 **bfloat16s** 位操作能力的工程师感到沮丧。同时，PyTorch Profiler 因其提供性能洞察的实用性而受到关注，详见 [PyTorch profiling 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

- **提示更好的 LoRA 初始化**：分享了一篇博客 [Know your LoRA](https://datta0.github.io/blogs/know-your-lora/)，建议 **LoRA** 中的 A 和 B 矩阵可以从非默认初始化中受益，从而可能改善微调结果。

- **Note 库展示 ML 效率**：引用了 [Note 库的 GitHub 仓库](https://github.com/NoteDance/Note)，该库提供了一个与 TensorFlow 兼容的 ML 库，承诺在包括 Llama2、Llama3 等模型上实现并行和分布式训练。

- **LLM 量化领域的飞跃**：频道深入讨论了 [字节跳动的 2-bit 量化算法 DecoupleQ](https://github.com/bytedance/decoupleQ)，并提供了一篇关于改进 Straight-Through Estimator 量化方法的 [NeurIPS 2022 论文](https://arxiv.org/pdf/2206.06501) 链接，指出了量化过程中对内存和计算的考量。

- **AI 框架讨论升温**：LLVM.c 社区深入探讨了支持 Triton 和 AMD、解决 BF16 梯度范数确定性（Determinism）以及未来对 Llama 3 等模型的支持。话题还涉及确保训练中的 100% 确定性，并考虑使用 FineWeb 作为数据集，同时权衡了扩展和数据类型多样化的因素。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **抄袭席卷研究论文**：五篇研究论文因内容中无意包含 AI prompts 而被撤回；成员们对这一疏忽反应不一，既有幽默调侃也有失望。
  
- **Haiku 模型：高性价比**：关于 **Haiku AI model** 的讨论非常热烈，其成本效益和出色的性能受到赞赏，甚至被比作 "gpt 3.5ish" 的质量。

- **AI 审核：一把双刃剑？**：社区内对在 Reddit 和 Discord 等平台使用 AI 进行内容 **moderation**（审核）的优缺点展开了热烈讨论，权衡了自动化操作与人工监督之间的平衡。

- **通过 YouTube 掌握 LLMs**：成员们分享了有助于更好理解 LLMs 的 YouTube 资源，特别提到了 Kyle Hill 的 *ChatGPT Explained Completely* 和 3blue1brown，称赞其引人入胜的数学解释。

- **GPT 不断变化的能力**：GPT-4o 正在向所有用户推出，**新的语音和视觉功能**专为 ChatGPT Plus 预留。同时，社区正在应对自定义 GPTs 频繁的修改通知以及在使用带有 CSV 附件的 GPT 时遇到的挑战。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 内的庆祝与协作**：LM Studio 迎来一周年里程碑，讨论涉及利用多 GPU——推荐使用 **Tesla K80** 和 **3090** 以保持一致性——以及运行多个实例进行模型间通信。强调了在 **LLMs** 方面 GPU 优于 CPU，并指出了在 PlayStation 5 APUs 等强大硬件上使用 LM Studio 的实际问题。

- **Higgs 强势登场**：对 **LMStudio update** 充满期待，该更新将整合令人印象深刻的 **Higgs LLAMA**，这是一个拥有 700 亿参数的大型模型，可能为 AI 工程师提供前所未有的能力和效率。

- **硬件方面的挑战与变通**：针对 **Tesla P40** 等小众硬件的 GPU 散热和电源问题引发了创意讨论，从简陋的 Mac GPU 风扇到复杂的纸板风道。建议包括参考 [GitHub guide](https://github.com/JingShing/How-to-use-tesla-p40) 来处理专有连接的绑定问题。

- **模型兼容性故障排除**：Qwen2 乱码的修复方法包括切换 *flash attention*，同时承认了 Qwen2 在 CUDA offloading 方面的风险，并期待 llama.cpp 的更新。一位成员分享了通过 API 使用 llava-phi-3-mini-f16.gguf 结果不一的经历，引发了进一步的模型诊断讨论。

- **微调要点**：关于微调的细致观点强调了通过 LoRA 进行风格调整与 SFT 基于知识的调优之间的区别；LM Studio 在没有训练的情况下对系统 prompt 名称的限制；以及应对“懒惰” LLM 行为的策略，如硬件升级或 prompt 优化。

- **AMD 技术的 ROCm 历程**：用户交流了在各种 AMD GPUs（如 6800m 和 7900xtx）上启用 **ROCm** 的技巧和经验，并提出了在 Arch Linux 上使用的建议以及在 Windows 环境下的变通方案，以优化其 LLM 设置的性能。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 工程师，准备好建立联系**：对 [AI Engineer event](https://www.ai.engineer/worldsfair/2024/schedule) 感兴趣的工程师可以使用 Expo Explorer 门票参加会议，不过演讲者阵容尚未最终确定。
- **KLING 等同于 Sora**：快手（KWAI）名为 KLING 的类 Sora 新模型因其逼真演示而引发关注，正如 Angry Tom 的 [tweet thread](https://x.com/angrytomtweets/status/1798777783952527818?s=46&t=90xQ8sGy63D2OtiaoGJuww) 所展示的那样。
- **剖析 GPT-4o 的图像处理**：Oran Looney 在一篇深度文章中剖析了 OpenAI 决定使用 170 个 tokens 处理 GPT-4o 图像的决定，讨论了编程中“[magic numbers](https://en.wikipedia.org/wiki/Magic_number_(%programming%))”（魔数）的重要性及其对 AI 的最新影响。
- **“幻觉工程师”发表见解**：辩论了 GPT 的“有用幻觉范式”概念，强调了其构思有益指标的潜力，并将其与 “superprompts” 和社区开发的工具（如 [Websim AI](https://websim.ai/)）进行了类比。
- **递归现实与资源库**：AI 爱好者们尝试了 websim.ai 的自引用模拟，同时分享了一个 [Google spreadsheet](https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl3u06IrKWoy3tlrCWlMRXDxvoDCY) 和一个 [GitHub Gist](https://gist.github.com/SawyerHood/5d82679953ced7142df42eb7810e8a7a)，用于未来会议的协作和广泛讨论。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Mixtral 的专家数量揭晓**：一场富有启发性的 [Stanford CS25 讲座](https://youtu.be/RcJ1YXHLv5o?si=gjcu--95HafBwEaT) 澄清了关于 **Mixtral 8x7B** 的误解，透露其包含 32x8 个专家，而非仅仅 8 个。这一细节凸显了其 MoE 架构背后的复杂性。
  
- **DeepSeek Coder 在代码任务中胜出**：根据 [Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) 上分享的介绍，**DeepSeek Coder 6.7B** 在项目级代码补全方面处于领先地位，展示了在高达 2 万亿代码 token 上训练出的卓越性能。

- **Meta AI 详解视觉语言建模**：Meta AI 通过 ["An Introduction to Vision-Language Modeling"](https://arxiv.org/abs/2405.17247) 提供了一份关于 Vision-Language Models (VLMs) 的全面指南，为那些对视觉与语言融合感兴趣的人详细介绍了其工作原理、训练和评估。

- **RAG 格式化技巧**：围绕 **RAG 数据集** 创建的讨论强调了简单性和针对性的必要性，拒绝千篇一律的框架，并重点介绍了利用 Ollama 和 emo 向量搜索进行数据集生成的工具，如 [Prophetissa](https://github.com/EveryOneIsGross/Prophetissa)。

- **WorldSim 控制台的移动端优化**：最新的 **WorldSim 控制台** 更新解决了移动端用户界面问题，通过修复文本输入 bug、增强 `!list` 命令以及新增禁用视觉效果的设置来提升体验，同时集成了多功能的 **Claude 模型**。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 迈向 Agentic 检索**：最近在旧金山总部的一次演讲强调了 Retrieval-Augmented Generation (RAG) 向 **全 Agentic 知识检索** 的演进。此举旨在克服 top-k 检索的局限性，可通过 [视频指南](https://t.co/fCK8L9O2sx) 获取增强实践的相关资源。

- **LlamaIndex 增强记忆能力**：LlamaIndex 引入了 **Vector Memory Module**，通过向量搜索存储和检索用户消息，从而加强 RAG 框架。感兴趣的工程师可以通过分享的 [demo notebook](https://t.co/Z1n8YC4grM) 探索此功能。

- **Create-llama 中增强的 Python 执行**：Create-llama 与 e2b_dev 沙箱的集成现在允许在 Agent 内部执行 Python 代码，这一进步使得返回复杂数据（如统计图表图片）成为可能。正如[此处](https://t.co/PRcuwJeVxf)详述，这一新功能拓宽了 Agent 的应用范围。

- **同步 RAG 与动态数据**：在 RAG 中实现动态数据更新涉及重新加载索引以反映最新更改，这一挑战通过使用定期索引刷新来解决。通过多个索引或元数据过滤器，可以优化对销售或支持文档等数据集的管理，相关实践已在 [Document Management - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=document+managem) 中列出。

- **LlamaIndex 中使用 Embedding 进行优化和实体解析**：直接使用 LlamaIndex 框架创建 property graphs 并应用 Embedding，可以通过调整 `chunk_size` 参数来增强实体解析。通过 "Optimization by Prompting" 等 RAG 指南和 [LlamaIndex 指南](https://github.com/run-llama/llama_index/blob/main/docs/docs/module_guides/indexing/lpg_index_guide.md?plain=1#L430) 可以更好地理解这些功能的管理。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Andrew Ng 对 AI 监管敲响警钟**：[Andrew Ng](https://x.com/AndrewYNg/status/1798753608974139779) 对加州的 SB-1047 法案表示担忧，担心其可能阻碍 AI 的进步。公会的工程师们对比了全球监管环境，指出即使没有美国的限制，像俄罗斯这样的国家也缺乏全面的 AI 政策，正如一段关于普京及其 deepfake 的[视频](https://www.youtube.com/watch?v=5L2YAIk0vSc&ab_channel=TheTelegraph)所示。

- **Mojo 变得更聪明而非更复杂**：确认了 `isdigit()` 函数为了性能而依赖 `ord()`，并在出现问题时提交了[问题报告](https://github.com/modularml/mojo/issues/2975)。**Mojo** 的 Async 功能仍有待进一步开发，建议使用 `__type_of` 进行变量类型检查，VSCode 扩展可辅助进行编译前后的识别。

- **MAX 首次登陆 macOS**：Modular 的 MAX 24.4 版本现已支持 macOS，并推出了 [Quantization API](https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos)，社区贡献者已突破 200 人。该更新有望显著降低 AI pipeline 的延迟和内存占用。

- **动态 Python 备受关注**：最新的 nightly 版本支持动态选择 `libpython`，有助于简化 Mojo 的环境搭建。然而，VS Code 的集成仍存在痛点，需要手动激活 `.venv`，这在 nightly 变更说明以及新引入的 [microbenchmarks](https://github.com/modularml/mojo/tree/nightly/stdlib/benchmarks) 中有详细说明。

- **对 Windows 原生版 Mojo 的高度期待**：工程师们调侃并渴望即将发布的 **Windows 原生版 Mojo**，其发布时间表仍充满神秘感。对该版本的渴望凸显了其对社区的重要性，表明 Windows 开发者对此有浓厚兴趣。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **工程师们开始上手新的 Sparse Autoencoder 库**：Nora Belrose 最近发布的 [推文](https://x.com/norabelrose/status/1798985066565259538) 介绍了一个用于训练 **TopK Sparse Autoencoders** 的库，该库在 **GPT-2 Small** 和 **Pythia 160M** 上进行了优化，可以同时训练所有层的 SAE，而无需在磁盘上缓存 activations。
  
- **Sparse Autoencoder 研究进展**：一篇新 [论文](https://arxiv.org/abs/2406.04093) 揭示了 k-sparse autoencoders 的开发，它增强了重构质量与稀疏性之间的平衡，这可能会显著影响语言模型特征的可解释性。

- **LLM 的下一次飞跃，源自新皮质 (Neocortex)**：成员们讨论了 Jeff Hawkins 和 Numenta 的 **Thousand Brains Project**，该项目旨在将新皮质原理应用于 AI，专注于开放协作——向大自然复杂的系统致敬。

- **评估错误文件路径的混乱**：针对一个已知问题，成员们确认文件处理（特别是错误的测试结果文件放置——**应位于 tmp 文件夹中**）已列入修复清单，正如 [KonradSzafer 的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1926/files) 所示。


- **揭示 Data Shapley 的不可预测性**：关于一篇 [arXiv 预印本](https://arxiv.org/abs/2405.03875) 的讨论展开，评估了 Data Shapley 在不同设置下数据选择性能的不一致性，建议工程师关注所提出的假设检验框架，以预测 Data Shapley 的有效性。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI 视频合成战争升温**：一款**中国 AI 视频生成器**的表现超越了 Sora，通过快手 (KWAI) iOS 应用提供令人惊叹的 2 分钟、1080p、30fps 视频，引起了社区的显著关注。同时，Johannes Brandstetter 宣布了 **Vision-LSTM**，它结合了 xLSTM 处理图像的能力，并提供了代码和 [arxiv 预印本](https://arxiv.org/abs/2406.04303) 以供进一步探索。

- **Anthropic 的 Claude API 访问权限扩大**：**Anthropic** 正在为对齐研究提供 API 访问权限，申请访问需要提供机构隶属关系、职位、LinkedIn、GitHub 和 Google Scholar 个人资料，以促进对 AI 对齐挑战的深入探索。

- **Daylight 电脑引发关注**：新型 **Daylight 电脑** 因其减少蓝光发射和增强阳光直射下可见性的承诺而吸引了大量关注，引发了关于其相对于 iPad mini 等现有设备潜在优势的讨论。

- **模型调试与理论的新前沿**：围绕类似于“自我调试模型”的新方法展开了引人入胜的对话，这些方法利用错误来改进输出，同时还讨论了对 DPO 等复杂理论密集型论文中解析解的渴望。

- **深化机器人讨论的挑战**：成员们敦促对机器人内容的变现策略和具体数据进行更深入的洞察，特别要求对“40,000 个高质量机器人年数据”进行更细致的分解，并对该领域的商业模式进行更仔细的审查。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **使用 Aya 进行免费翻译研究，但商业用途需付费**：虽然 **Aya** 对学术研究免费，但商业应用需要付费以维持业务。面临 **Vercel AI SDK** 与 **Cohere** 集成挑战的用户可以找到指导，并已采取措施联系 SDK 维护者寻求支持。

- **聪明的 Command-R-Plus 胜过 Llama3**：用户表示 **Command-R-Plus** 在某些场景下优于 **Llama3**，并引用了对其在语言规范之外表现的主观体验。

- **探索 Cohere 使用的数据隐私选项**：对于在使用 Cohere 模型时担心数据隐私的用户，社区分享了关于如何在本地或 [AWS](https://docs.cohere.com/docs/cohere-on-aws) 和 [Azure](https://docs.cohere.com/docs/cohere-on-microsoft-azure) 等云服务上将这些模型用于个人项目的详细信息和链接。

- **开发者展示全栈专业知识**：一份全栈开发者作品集已上线，展示了在 **UI/UX, Javascript, React, Next.js, 和 Python/Django** 方面的技能。作品集可以在该开发者的[个人网站](https://www.aozora-developer.com/)上查看。

- **关注 GenAI 安全和新型搜索解决方案**：Rafael 正在开发一款防止 GenAI 应用出现幻觉的产品并邀请合作；而 Hamed 推出了 **Complexity**，这是一个令人印象深刻的生成式搜索引擎，邀请用户在 [cplx.ai](https://cplx.ai/) 进行探索。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Qwen 2 也支持韩语**：Voidnewbie 提到，最近添加到 OpenRouter 产品中的 [Qwen 2 72B Instruct](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) 模型也支持韩语。

- **OpenRouter 与网关故障作斗争**：多名用户在 Llama 3 70B 模型上遇到了 504 网关超时错误；数据库压力被确定为罪魁祸首，促使将任务迁移到只读副本以提高稳定性。

- **路由困扰引发技术对话**：成员报告 WizardLM-2 8X22 通过 DeepInfra 产生乱码响应，导致建议调整请求路由中的 `order` 字段，并暗示正在部署内部端点以帮助解决服务提供商问题。

- **AI 安全引发激烈争论**：Leopold Aschenbrenner 被 OAI 解雇一事引发了成员之间关于 AI 安全重要性的激烈辩论，反映了在 AI 保护措施的必要性和影响方面的观点分歧。

- **ChatGPT 的性能波动**：分享了关于 ChatGPT 在高流量期间可能出现性能下降的观察，引发了关于重负载对服务质量和一致用户体验影响的推测。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Flash-attn 安装需要高 RAM**：成员们强调了在 slurm 上构建 Flash-attn 时的困难；解决方案包括加载必要的模块以提供充足的 RAM。

**Finetuning 问题已修复**：报告了 Qwen2 72b 在 Finetuning 时的配置问题，建议进行另一轮调整，特别是由于 `max_window_layers` 的错误设置。

**多节点 Finetuning 指南**：分享了一个使用 Axolotl 和 Deepspeed 进行[分布式 Finetuning 的 Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1688)，标志着社区协作开发努力的增加。

**数据难题解决**：一名成员在配置 JSONL 格式的 `test_datasets` 时遇到的困难，通过采用为 `axolotl.cli.preprocess` 指定的结构得到了解决。

**针对工程化推理，API 优于 YAML**：澄清了 Axolotl 在 API 使用与 YAML 设置方面的配置困惑，重点在于扩展脚本化、持续模型评估的能力。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **从左到右序列生成的竞争者**：与 SkysoftATM 合作开发的创新型 **σ-GPT** 挑战了传统的 GPTs，通过动态生成序列，可能将步骤减少一个数量级，详见其 [arXiv 论文](https://arxiv.org/abs/2404.09562)。
- **讨论 σ-GPT 的高效学习**：尽管 **σ-GPT** 采用了创新方法，但对其实用性存在质疑，因为高性能的课程学习（curriculum）可能会限制其使用，这与 XLNET 有限的影响力有相似之处。
- **探索填充任务的替代方案**：对于某些操作，像 [GLMs](https://arxiv.org/abs/2103.10360) 这样的模型可能被证明更有效，而使用独特的 Positional Embeddings 进行 Finetuning 可能会增强基于 RL 的非文本序列建模。
- **AI 视频生成竞争升温**：快手 (KWAI) iOS 应用上的一款新型中国 AI 视频生成器可以制作 **1080p、30fps 的 2 分钟视频**，引起了轰动；而另一款生成器 Kling 凭借其逼真的能力，其真实性也受到了质疑。
- **社区对 Schelling AI 公告的反应**：Emad Mostaque 关于 Schelling AI 的推文引发了质疑和幽默的混杂，该项目旨在使 AI 和 AI 算力挖掘民主化，但因使用流行语和宏大主张而受到关注。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 中的 Early Stopping 障碍**：讨论强调了 LangChain 中的 `early_stopping_method="generate"` 选项在较新版本中无法按预期工作的问题，促使一位用户链接了一个[活跃的 GitHub Issue](https://github.com/langchain-ai/langchain/issues/16263)。社区正在探索权宜之计并等待官方修复。

- **RAG 和 ChromaDB 隐私担忧**：出现了关于在使用 LangChain 和 ChromaDB 时增强数据隐私的查询，建议在 vectorstores 中利用*基于元数据的过滤 (metadata-based filtering)*，正如 [GitHub 讨论](https://github.com/langchain-ai/langchain/discussions/9645)中所述，尽管承认该话题具有复杂性。

- **LLaMA3-70B 的 Prompt Engineering**：工程师们集思广益，探讨了让 LLaMA3-70B 执行任务而无需冗余开场白（prefatory phrases）的有效 Prompting 技术。尽管进行了多次尝试，但共享的对话中尚未出现确定的解决方案。

- **苹果推出生成式 AI 指南**：一位工程师分享了苹果新制定的生成式 AI [指导原则](https://drive.google.com/file/d/1s0imJ0zidk5-hraT46y8u4jnUby_oukk/view)，旨在优化苹果硬件上的 AI 运行，这对 AI 应用开发者可能非常有用。

- **B-Bot 应用的 Alpha 测试**：宣布了 B-Bot 应用程序的封闭 Alpha 测试阶段，这是一个专家知识交流平台，邀请函已发送至[此处](https://discord.gg/V737s4vW)，寻求测试人员提供开发反馈。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Phi-3 模型导出困惑已解决**：用户解决了将自定义 **phi-3 model** 导出到 Hugging Face 以及从其导入的问题，并从 [GitHub discussion](https://github.com/pytorch/torchtune/blob/16b7c8e16ade9ab0dc362f1ee2dd7f0e04fc227c/torchtune/_cli/download.py#L90) 中指出了潜在的配置错误。会议指出，通过使用 FullModelHFCheckpointer，Torchtune 在 checkpoint 过程中可以处理其自身格式与 HF format 之间的转换。
  
- **关于 PR 的澄清与欢迎**：针对通过 **n_kv_heads** 增强 **Torchtune** 以支持 mqa/gqa 的咨询得到了澄清，并鼓励提交 pull requests，前提是需要为任何拟议的更改提供 unit tests。

- **开发讨论中的依赖项风波**：工程师们强调了安装 Torchtune 时 **versioning of dependencies**（依赖版本控制）的准确性，并指出了版本不匹配引起的问题，引用了如 [Issue #1071](https://github.com/pytorch/torchtune/issues/1071)、[Issue #1038](https://github.com/pytorch/torchtune/issues/1038) 和 [Issue #1034](https://github.com/pytorch/torchtune/issues/1034) 等情况。

- **Nightly Builds 获得认可**：大家达成共识，有必要明确使用 Torchtune 完整功能集需要 **PyTorch nightly builds**，因为某些功能是这些版本独有的。

- **为更清晰的安装流程准备 PR**：一位社区成员宣布他们正在准备一个专门更新 Torchtune 安装文档的 PR，以解决围绕依赖版本控制和使用 **PyTorch nightly builds** 的难题。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **觉醒文化的终结**：**AI Stack Devs** 频道的讨论围绕着像 **Stellar Blade**（剑星）背后的游戏工作室展开；成员们赞赏这些工作室专注于游戏质量，而非 **Western SJW** 主题和 **DEI** 措施。
  
- **回归游戏本质**：成员们对 **Shift Up** 等 **Chinese** 和 **South Korean developers** 表示钦佩，因为他们专注于游戏开发，而没有卷入女权主义等社会政治运动，尽管韩国社会在这些问题上面临挑战。

- **Among AI Town - 开发中的 Mod**：一个 **AI-Powered "Among Us"** mod 引起了关注，游戏开发者注意到 AI Town 在早期阶段的有效性，尽管存在一些局限性和性能问题。

- **使用 Godot 升级**：**ai-town-discuss** 频道提到了从 AI Town 转向使用 [Godot](https://zaranova.xyz) 的过程，作为为 "Among Us" mod 添加高级功能的步骤，标志着在初始能力基础上的改进和扩展。

- **AI Town 的持续增强**：AI Town 的开发仍在继续，贡献者们正在推动项目前进，正如最近关于持续进展和更新的对话所表明的那样。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **是否有 Open Interpreter 的桌面版？**：一位公会成员询问是否有 **Open Interpreter** 的桌面 UI，但对话中未提供回复。

- **Open Interpreter 连接的波折**：一名用户正苦于 **Posthog connection errors**，特别是与 `us-api.i.posthog.com` 的连接，这表明其设置或外部服务可用性存在更广泛的问题。

- **使用 OpenAPI 配置 Open Interpreter**：讨论围绕 **Open Interpreter** 是否可以利用现有的 **OpenAPI specs** 进行 function calling 展开，建议通过某些配置中的 true/false 开关作为潜在解决方案。

- **在 Gorilla 2 中使用工具**：分享了在 **LM Studio** 中使用工具的挑战，以及在自定义 JSON 输出和 OpenAI toolcalling 方面取得的成功。建议查看 GitHub 上的 **OI Streamlit** 仓库以寻找可能的解决方案。

- **寻找技巧？查看 OI 网站**：在对一个请求的简短回复中，**ashthescholar.** 引导成员访问 **Open Interpreter** 的网站以获取指导。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mixtral 的专家系统揭秘**：一段澄清性的 [YouTube 视频](https://www.youtube.com/watch?v=RcJ1YXHLv5o) 破除了关于 Mixtral 的传言，确认其在各层中包含 **256 个专家**，并拥有惊人的 **467 亿参数**，其中 **129 亿激活参数** 用于 token 交互。

- **哪个 DiscoLM 占据统治地位？**：关于领先的 DiscoLM 模型的讨论充满了困惑，多个模型在争夺关注，建议对于仅有 3GB VRAM 的系统使用 **8b llama**。

- **在极小 VRAM 上实现内存最大化**：一位用户使用 Q2-k 量化在 3GB VRAM 配置上成功以每秒 6-7 个 token 的速度运行 **Mixtral 8x7b**，强调了模型选择中内存效率的重要性。

- **重新评估 Vagosolutions 的能力**：最近的基准测试引发了对 **Vagosolutions 模型** 的新兴趣，导致了关于微调 Mixtral 8x7b 是否能胜过微调后的 Mistral 7b 的辩论。

- **RKWV vs Transformers - 解码优势**：该频道尚未回应关于 RKWV 相对于 Transformers 的直观优势见解的请求，这表明可能存在疏忽或需要更多调查。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM 瞄准新的数字领域**：成员们分享了暗示 **LLM 正在被集成到 Web 平台** 的进展，Google 正在考虑为 Chrome 引入 LLM（[Chrome AI 集成](https://developer.chrome.com/docs/ai/built-in)），而 Mozilla 正在 Firefox Nightly 中实验使用 transformers.js 进行本地替代文本生成（[Mozilla 实验](https://hacks.mozilla.org/2024/05/experimenting-with-local-alt-text-generation-in-firefox-nightly/)）。用户推测最终目标是在 **操作系统级别** 实现更深层次的 AI 集成。

- **Prompt Injection 的技巧与行业**：通过一名成员的 LinkedIn 经历强调了一个使用 **Prompt Injection** 操纵电子邮件地址的有趣用例，并讨论了一个展示该概念的链接（[Prompt Injection 见解](https://infosec.town/notes/9u788f3ojs6gyz9b)）。

- **跨维度文本分析**：一位频道成员深入研究了测量文本中“概念速度”的概念，灵感来自一篇博客文章（[概念速度见解](https://interconnected.org/home/2024/05/31/camera)），并表现出将这些概念应用于天文新闻数据的兴趣。

- **维度：Embedding 的视觉前沿**：成员们赞赏了一篇关于使用 PCA、t-SNE 和 UMAP 进行降维解释的 Medium 文章（[3D 可视化技术](https://medium.com/@madhugraj/explainability-for-text-data-3d-visualization-of-token-embeddings-using-pca-t-sne-and-umap-8da33602615b)），这有助于可视化 200 篇天文新闻文章。

- **UMAP 在恒星聚类方面优于 PCA**：研究发现，当使用 GPT-3.5 进行标注时，对于分类的新闻主题（如嫦娥六号月球着陆器和 Starliner），UMAP 提供的聚类效果显著优于 PCA。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz 挑战 Taylor Series 悬赏假设**：George Hotz 对一个关于 Taylor Series 悬赏要求的问题给出了一个疑惑的评论，促使人们重新考虑假设的要求。

- **证明逻辑受到审视**：一名成员对某个未识别证明的有效性感到困惑，引发了一场辩论，质疑该证明的逻辑或结果。

- **符号化 Shape 维度归零**：出现了一场关于符号化 Shape 维度是否可以为零的讨论，表明了对 Tensor 操作中符号化表示极限的兴趣。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1248366290184441886)** (12 messages🔥): 

- **RAG 代表了一些搞笑的意思？**: 成员们开玩笑地讨论了 RAG 的含义，建议它代表 "random-ass guess"（随机瞎猜）。
- **开源模型训练的误区**: 一位成员分享了关于开源模型训练步数（steps）和轮数（epochs）误区的见解，指出 "Num epochs = max_steps / file_of_training_jsonls_in_MB" 的公式导致一个 31 MB 的文件运行了 30 个 epochs。
- **分享 Mistral 微调仓库**: 一位贡献者分享了一个 [GitHub 仓库](https://github.com/andresckamilo/mistral-finetune-modal)，该仓库旨在 Modal 中使用 Mistral 的微调 Notebook；另一位成员分享了 [modal-labs 的指南](https://github.com/modal-labs/llm-finetuning)，并指出过时的 deepspeed 文档存在问题。
- **混合搜索咨询**: 提出了关于归一化 BM25 分数以及在与稠密向量搜索（dense vector search）结合时如何加权的问题。一位成员推荐了一篇 [博客文章](https://aetperf.github.io/2024/05/30/A-Hybrid-information-retriever-with-DuckDB.html#fused_score) 以供进一步阅读。
- **访问旧的 Zoom 录制**: 对于访问旧 Zoom 录制的请求，得到的回复是去 Maven 网站查看。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aetperf.github.io/2024/05/30/A-Hybrid-information-retriever-with-DuckDB.html#fused_score">A Hybrid information retriever with DuckDB</a>: 数据库、数据可视化、数据科学</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: 微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/andresckamilo/mistral-finetune-modal.git">GitHub - andresckamilo/mistral-finetune-modal</a>: 通过在 GitHub 上创建账户来为 andresckamilo/mistral-finetune-modal 做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/)** (1 messages): 

_ribhu: 嘿，我可以帮忙。你能私信我详细信息吗？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1248542530790883369)** (17 messages🔥): 

- **享受 Modal 的 VRAM 和 GPU 算力**: 一位成员称赞了在 Modal 上使用 10 个 A100 GPU 的体验，并询问了 VRAM 使用情况的可见性。他们还分享了一个 [使用 Moondream VLM 描述宝可梦卡片的脚本](https://gist.github.com/sroecker/5c3a9eb1fd0c898e4119b89ff1095038)。

- **简化 flash-attn 的安装**: 他们寻求 pip 安装 `flash-attn` 的技巧，随后发现了一个非常有用的 Whisper Gist 并根据自己的需求进行了调整。该 Gist 可以在 [这里](https://gist.github.com/aksh-at/fb14599c28a3bc0f907ea45398a7651d) 找到。

- **Modal 的成本效益令用户印象深刻**: 使用 10x A100 (40G) 在 10 分钟内处理 1.3 万张图像仅花费 7 美元，这一高性价比受到了关注，并得到了其他成员的积极回应。

- **GPU 可用性问题**: 一位成员报告称等待 A10 需要 17 分钟，另一位成员认为这很不寻常。随后他们确认系统运行正常，任何问题都应标记给平台工程师进行审查。

- **缺乏队列状态可见性**: 另一位试图获取 H100 节点的成员询问是否有 Dashboard 可以查看队列状态。得到的澄清是，虽然有用于查看运行中/已部署应用的 Dashboard，但目前没有可以查看队列状态或可用性预估的界面。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/aksh-at/fb14599c28a3bc0f907ea45398a7651d">Insanely fast whisper on Modal</a>: Modal 上极速运行的 Whisper。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/sroecker/5c3a9eb1fd0c898e4119b89ff1095038">Modal: Batch eval Moondream with Pokemon dataset</a>: Modal：使用宝可梦数据集批量评估 Moondream。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://modal.com/docs/guide/webhook-urls">Web endpoint URLs</a>: 用户对 Web 端点的 URL 拥有部分控制权。对于 web_endpoint、asgi_app 和 wsgi_app 装饰器，可以提供一个可选的 label 关键字参数，用于分配 URL https://&...
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1248414079715119218)** (2 messages): 

- **通过 VectorHub 探索向量驱动系统**：一位成员分享了 Superlinked 提供的 VectorHub 链接，该平台提供了关于 [RAG, text embeddings, and vectors](https://superlinked.com/vectorhub/) 的资源。该平台是一个免费的教育和开源资源，旨在帮助数据科学家和软件工程师构建向量驱动的系统。

- **OpenPipe 上的相关文章**：另一位成员发布了 [OpenPipe blog](https://openpipe.ai/blog) 的链接以获取相关文章。该博客包含了与本频道讨论话题相关的见解和信息。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openpipe.ai/blog">OpenPipe</a>: 将昂贵的 LLM prompts 转换为快速、廉价的 fine-tuned 模型</li><li><a href="https://superlinked.com/vectorhub/">VectorHub by Superlinked</a>: Superlinked 提供的 VectorHub；学习构建可扩展的向量系统
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1248381668067250308)** (2 messages): 

- **对云端 GPU 平台的致谢**：一位成员感谢 Vishnu 创建了一个出色的云端 GPU 平台，并分享了一篇关于将其与 Axolotl 配合使用的博客文章。他们分享了一条 [tweet](https://x.com/cleavey1985/status/1798822521359511657) 和一篇[详细的博客文章](https://drchrislevy.github.io/posts/fine_tune_jarvis/fine_tune_jarvis.html)，讨论了从 LLM 课程和会议中获得的经验。

- **Huggingface 数据集处理崩溃报告**：一位用户报告了在具有 32 GB VRAM 的 A6000 上处理 Huggingface 数据集时发生的崩溃。他们提供了一个 [gist link](https://gist.github.com/alexis779/7cd7d6b2d43991c11cbebe43afff0347) 指向错误详情，并提到该过程在他们 32 GB 的 Dell 笔记本电脑上运行正常。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/cleavey1985/status/1798822521359511657).">Chris Levy (@cleavey1985) 的推文</a>: 准备根据我在由 @HamelHusain 和 @dan_s_becker 组织的精彩 LLM 课程/会议中学到的内容写一些博客。这一篇是关于在 JarvisLabs 上使用 @axolotl_ai 的。感谢 @vi...</li><li><a href="https://gist.github.com/alexis779/7cd7d6b2d43991c11cbebe43afff0347">在标称 32 GB 的 A6000 上数据集处理崩溃</a>: 在标称 32 GB 的 A6000 上数据集处理崩溃 - preprocess.py
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1248552887827824650)** (3 messages): 

- **额度申请新表单于 10 月 10 日开放**：一位成员宣布，用于获取额度的新表单将于 10 月 10 日星期一正式重新开放。该表单将持续开放至 18 日，允许在 15 日之前注册的新学生获得额度。
- **额度缺失问题**：另一位成员对账户中此前存在的额度丢失表示担忧。他们被建议联系 billing@huggingface.co 以解决此问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1248470069093335120)** (1 messages): 

- **Docker 本地磁盘空间问题**：一位成员报告了在尝试 Replicate 演示期间运行 `cog run` 和 `cog push` 命令时遇到的问题。他们怀疑本地计算机缺乏足够的磁盘空间，并询问是否可以远程进行构建过程。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1248392611824271482)** (8 messages🔥): 

- **额度获取说明**：多位用户讨论了关于额度可用性和获取的问题。会议中多次强调：*"无论你是否设置了账单，额度都已存入。你只需要在文件中记录一种支付方式即可访问这些额度。"*

- **表单和 Org ID 详情**：一位用户请求了关于其 Org ID (b9e3d34d-3c3c-4528-8e2f-2b31075b47fd) 的具体细节以便处理账单。随后有提示确认他们之前是否填写过表单，并表示可以通过邮件 jess@langchain.dev 进一步协调协助。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1248688490502426654)** (1 messages): 

- **Predibase Office Hours 时间重排**：Predibase 的 Office Hours 现在改在 **太平洋时间 6 月 12 日星期三上午 10 点** 进行。主题包括 LoRAX、multi-LoRA 推理，以及通过 speculative decoding 进行提速的 fine-tuning。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1248350814976868443)** (64 messages🔥🔥): 

```html
<ul>
  <li>
    <strong>Marker 在处理格式错误的 Markdown 表格时表现不佳：</strong> 一位用户对用于将 PDF 转换为 Markdown 的 <a href="https://github.com/VikParuchuri/marker/tree/master">Marker 工具</a>表示失望，解释说生成的 Markdown 表格通常不符合他们的要求。这引发了关于可能通过 Fine-tuning 该工具来改进表格格式的讨论。
  </li>
  <li>
    <strong>探索 Embedding Quantization：</strong> 讨论了 <a href="https://huggingface.co/blog/embedding-quantization">Quantized Embeddings</a> 的实用性，重点展示了一个涉及 4100 万条 Wikipedia 文本的真实检索场景 Demo。该博客文章涵盖了 Embedding Quantization 对检索速度、内存占用、磁盘空间和成本的影响。
  </li>
  <li>
    <strong>针对 RAG 复杂性的 GitHub 仓库：</strong> 一位成员分享了 <a href="https://github.com/jxnl/n-levels-of-rag">n-levels-of-rag</a> GitHub 仓库的链接及相关的 <a href="https://jxnl.github.io/blog/writing/2024/02/28/levels-of-complexity-rag-applications/">博客文章</a>，为理解和实现不同复杂程度的 RAG 应用提供了全面指南。
  </li>
  <li>
    <strong>应对表格提取挑战：</strong> 讨论了一个替代的表格提取工具，一位用户推荐了 <a href="https://github.com/xavctn/img2table">img2table</a>，这是一个基于 OpenCV 的库，用于从 PDF 和图像中识别并提取表格。用户分享了他们的经验以及对现有表格提取和转换工具的潜在改进建议。
  </li>
  <li>
    <strong>多语言内容 Embedding 模型咨询：</strong> 一位用户询问了适用于多语言内容的 Embedding 模型，这引发了关于各种推荐方案和 Fine-tuning 方法论的讨论，旨在更好地处理多语言环境下的特定需求。 
  </li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://useinstructor.com/blog/2024/06/06/enhancing-rag-with-time-filters-using-instructor/">Enhancing RAG with Time Filters Using Instructor - Instructor</a>: 未找到描述</li><li><a href="https://lancedb.github.io/lancedb/fts/">Full-text search - LanceDB</a>: 未找到描述</li><li><a href="https://github.com/castorini/pyserini">GitHub - castorini/pyserini: Pyserini is a Python toolkit for reproducible information retrieval research with sparse and dense representations.</a>: Pyserini 是一个用于稀疏和密集表示的可复现信息检索研究的 Python 工具包。 - castorini/pyserini</li><li><a href="https://github.com/xavctn/img2table">GitHub - xavctn/img2table: img2table is a table identification and extraction Python Library for PDF and images, based on OpenCV image processing</a>: img2table 是一个基于 OpenCV 图像处理的用于 PDF 和图像的表格识别与提取 Python 库 - xavctn/img2table</li><li><a href="https://github.com/jxnl/n-levels-of-rag">GitHub - jxnl/n-levels-of-rag</a>: 通过在 GitHub 上创建一个账号来为 jxnl/n-levels-of-rag 的开发做出贡献。</li><li><a href="https://jxnl.github.io/blog/writing/2024/02/28/levels-of-complexity-rag-applications/">Levels of Complexity: RAG Applications - jxnl.co</a>: 未找到描述</li><li><a href="https://python.useinstructor.com/blog/">Welcome to the Instructor Blog - Instructor</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: 未找到描述</li><li><a href="https://github.com/VikParuchuri/marker/tree/master">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: 快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为最优 RAG 解析文件。通过在 GitHub 上创建一个账号来为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://x.com/jxnlco">Tweet from undefined</a>: 未找到描述</li><li><a href="https://dub.sh/jxnl-rag">RAG - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/">Systematically Improving Your RAG - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/05/11/low-hanging-fruit-for-rag-search/">Low-Hanging Fruit for RAG Search - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/">Levels of Complexity: RAG Applications - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.co/writing/2024/02/05/when-to-lgtm-at-k/">Stop using LGTM@Few as a metric (Better RAG) - jxnl.co</a>: 未找到描述</li><li><a href="https://jxnl.github.io/blog/writing/2024/01/07/inverted-thinking-rag/">How to build a terrible RAG system - jxnl.co</a>: 未找到描述</li><li><a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG is more than just embedding search - Instructor</a>: 未找到描述</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: 未找到描述</li><li><a href="https://lancedb.com/">LanceDB - The Database for Multimodal AI</a>: 多模态 AI 数据库</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: 自主 Agent 的长期记忆。通过在 GitHub 上创建一个账号来为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://x.com/_philschmid/status/1798388387822317933">Tweet from Philipp Schmid (@_philschmid)</a>: 创建一个用于生成合成数据以微调自定义嵌入模型的流水线。👀 第一步 创建知识库：从准备特定领域的知识库开始，例如 PDF 或...</li><li><a href="https://www.youtube.com/watch?v=R0VJIW0IYPo">Toran Billups, Adventures with Synthetic Data</a>: Toran Billups - https://twitter.com/toranb?lang=en 网站 - https://toranbillups.com/ 加入远程 Meetup - https://www.meetup.com/denver-erlang-elixir/ New Yo...</li><li><a href="https://blog.dottxt.co/coalescence.html">Coalescence: making LLM inference 5x faster</a>: 未找到描述</li><li><a href="https://manisnesan.github.io/chrestotes/posts/2023-07-07-doc-expansion-by-query-pred.html">chrestotes - Document Expansion by Query Prediction to Improve Retrieval Effectiveness</a>: 未找到描述</li>

<li><a href="https://modal.com/blog/fine-tuning-embeddings">通过快速微调超越专有模型</a>：仅需几百个示例进行微调，即可开启你自己的数据飞轮。</li><li><a href="https://www.timescale.com/">面向时间序列和事件的 PostgreSQL ++</a>：专为处理高要求工作负载而设计，如时间序列、向量、事件和分析数据。基于 PostgreSQL 构建，提供专家支持且不收取额外费用。</li><li><a href="https://www.limitless.ai/">Limitless</a>：超越大脑的局限：由你所见、所说、所闻驱动的个性化 AI。</li><li><a href="https://www.raycast.com/">Raycast - 你的万能快捷方式</a>：集成在可扩展启动器中的强大生产力工具集。</li><li><a href="https://www.tensorlake.ai/">Tensorlake</a>：未找到描述</li><li><a href="https://dunbar.app/">Home</a>：你的个人缘分引擎。智能连接新员工入职、同伴学习、虚拟咖啡等场景。免费试用 dunbar，无需信用卡，激发有意义的连接...</li><li><a href="https://www.bytebot.ai/">Bytebot - 在网页抓取、自动化、测试和监控中利用 AI 的力量</a>：使用我们的 AI 赋能 SDK 增强并简化你的浏览器自动化。有了 Bytebot，创建网页任务就像编写 Prompt 一样简单。</li><li><a href="https://www.narohq.com/">Naro - AI 驱动的销售知识</a>：未找到描述</li><li><a href="https://modal.com/">Modal：面向开发者的高性能云</a>：自带代码，大规模运行 CPU、GPU 和数据密集型计算。面向 AI 和数据团队的 Serverless 平台。</li><li><a href="https://docs.pydantic.dev/latest/">欢迎来到 Pydantic - Pydantic</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM 微调 (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1248393672391659642)** (221 条消息🔥🔥): 

- **讨论 FastHTML 特性与组件**：成员们对 **FastHTML** 表现出兴奋和好奇，将其与 **FastAPI** 和 **Django** 进行了正面对比。对话包括关于创建应用、连接多个数据库以及使用 **picolink** 和 **daisyUI** 等各种库的详细说明。
  
- **Jeremy 及其团队的贡献与未来计划**：Jeremy 和 John 频繁参与回答问题，承诺未来将支持 Markdown，并讨论了对社区构建组件库的需求。Jeremy 邀请成员通过为 **Bootstrap** 或 **Material Tailwind** 等流行框架创建易于使用的 **FastHTML** 库来做出贡献。

- **使用 FastHTML 进行 Markdown 渲染**：Jeremy 和 John 讨论了在 **FastHTML** 中渲染 Markdown 的方法，包括使用 **scripts** 和 **NotStr** 类。John 分享了一个代码片段，演示了如何使用 JavaScript 和 UUIDs 来渲染 Markdown。

- **HTMX 集成与用例**：强调了 HTMX 在 **FastHTML** 中的作用，并提供了处理 **键盘快捷键** 和 **数据库交互** 等各种事件的示例。成员们还分享了 HTMX 的使用技巧和经验，突出了其有效性，并将其交互模式与 JavaScript 进行了对比。

- **编程工具与环境讨论**：讨论了 **Cursor** 和 **Railway** 等其他工具和平台，成员们分享了他们的经验和最佳实践。还分享了与 FastHTML 相关的资源和教程，例如 [WIP 教程](https://answerdotai.github.io/fasthtml/by_example.html) 和几个 GitHub 仓库。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://tailwindcss.com/docs/utility-first">Utility-First Fundamentals - Tailwind CSS</a>: 从一组受限的基础 Utility 构建复杂的组件。</li><li><a href="https://discord.gg/fbCU6btg">加入 fast.ai Discord 服务器！</a>: 查看 Discord 上的 fast.ai 社区 —— 与 10887 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://cursor.sh/">Cursor</a>: AI 代码编辑器</li><li><a href="https://tenor.com/view/hhgf-gif-25031041">Hhgf GIF - Hhgf - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/pydantic/FastUI">GitHub - pydantic/FastUI: 更快地构建更好的 UI。</a>: 更快地构建更好的 UI。通过在 GitHub 上创建账号，为 pydantic/FastUI 的开发做出贡献。</li><li><a href="https://github.com/AnswerDotAI/fasthtml-tut">GitHub - AnswerDotAI/fasthtml-tut: 初学者 FastHTML 教程配套代码</a>: 初学者 FastHTML 教程配套代码。通过在 GitHub 上创建账号，为 AnswerDotAI/fasthtml-tut 的开发做出贡献。</li><li><a href="https://github.com/AnswerDotAI/fastlite">GitHub - AnswerDotAI/fastlite: 为 sqlite 增加一些易用性</a>: 为 sqlite 增加一些易用性。通过在 GitHub 上创建账号，为 AnswerDotAI/fastlite 的开发做出贡献。</li><li><a href="https://railway.app/">Railway</a>: Railway 是一个基础设施平台，你可以在这里配置基础设施，在本地使用该基础设施进行开发，然后部署到云端。</li><li><a href="https://www.google.com)")">未找到标题</a>: 未找到描述</li><li><a href="https://htmx.org/examples/keyboard-shortcuts/">&lt;/&gt; htmx ~ 示例 ~ 键盘快捷键</a>: 未找到描述</li><li><a href="https://discord.gg/vZPypuvw">加入 fast.ai Discord 服务器！</a>: 查看 Discord 上的 fast.ai 社区 —— 与 10887 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://answerdotai.github.io/fasthtml/by_example.html">fasthtml - 通过示例学习 FastHTML</a>: 另一种入门介绍</li><li><a href="https://mui.com/material-ui/">Material UI: 实现 Material Design 的 React 组件</a>: Material UI 是一个开源的 React 组件库，实现了 Google 的 Material Design。它功能全面，可以开箱即用地用于生产环境。</li><li><a href="https://pixi.sh/latest/">快速入门</a>: 让包管理变得简单</li><li><a href="https://pixi.sh/latest/tutorials/python/">Python - Pixi by prefix.dev</a>: 无</li><li><a href="https://github.com/AnswerDotAI/fasthtml-example">GitHub - AnswerDotAI/fasthtml-example: FastHTML 示例应用</a>: FastHTML 示例应用。通过在 GitHub 上创建账号，为 AnswerDotAI/fasthtml-example 的开发做出贡献。</li><li><a href="https://image-gen-public-credit-pool.replit.app/">
图像生成演示

</a>: 未找到描述</li><li><a href="https://daisyui.com/components/">Components — Tailwind CSS Components ( version 4 update is here )</a>: 由 daisyUI 提供的 Tailwind CSS 组件示例</li><li><a href="https://pocketbase.io/">PocketBase - Open Source backend in 1 file</a>: 包含实时数据库、身份验证、文件存储和管理后台的单文件开源后端</li><li><a href="https://mdbootstrap.com/docs/standard/getting-started/installation/">MDBootstrap 5 download &amp; installation guide</a>: Material Design for Bootstrap 可免费下载。通过 npm、CDN、MDB CLI、GitHub 安装或作为 .zip 压缩包下载。</li><li><a href="https://sqlite-utils.datasette.io/">sqlite-utils</a>: 未找到描述</li><li><a href="https://mdbootstrap.com/docs/standard/">Bootstrap 5 &amp; Vanilla JavaScript - Free Material Design UI KIT</a>: 700+ 组件、精美模板、1 分钟安装、详尽教程和庞大社区。MIT 许可证 - 个人及商业用途免费。</li><li><a href="https://x.com/jeremyphoward/status/1796692221720490044">Tweet from Jeremy Howard (@jeremyphoward)</a>: 一个真实世界超媒体驱动应用程序的集合。 https://hypermedia.gallery/</li><li><a href="https://hypermedia.systems/book/contents/">Hypermedia Systems</a>: 未找到描述</li><li><a href="https://sqlite-utils.datasette.io/en/stable/">sqlite-utils</a>: 未找到描述</li><li><a href="https://answerdotai.github.io/fasthtml/">fasthtml</a>: 创建 HTML 应用的最快方式</li><li><a href="https://github.com/pydantic/FastUI/tree/main">GitHub - pydantic/FastUI: Build better UIs faster.</a>: 更快地构建更好的 UI。通过在 GitHub 上创建账号为 pydantic/FastUI 的开发做出贡献。</li><li><a href="https://pyviz.org/tools.html">All Tools — PyViz 0.0.1 documentation</a>: 未找到描述</li><li><a href="https://picocss.com,">no title found</a>: 未找到描述</li><li><a href="https://nbdev.fast.ai/">nbdev – Create delightful software with Jupyter Notebooks</a>: 编写、测试、记录并分发软件包和技术文章 —— 全都在你的 notebook 中一站式完成。</li><li><a href="https://www.cursor.com/">Cursor</a>: AI 代码编辑器</li><li><a href="https://picocss.com/">Pico CSS • Minimal CSS Framework for semantic HTML</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1248394293148651531)** (23 messages🔥): 

- **在 Modal 上寻找用于 FFT 的 Python 脚本：** 一位用户正在寻找在 Modal 上为 llama-3 运行 FFT 的 Python 脚本，但目前只找到了一个 Lora git 项目。
- **Mistral Instruct 的聊天模板：** 一位成员询问支持 DPO 微调系统提示词（system prompt）的可用聊天模板，寻求关于格式和模板使用的明确说明。
- **不建议混合使用 Axolotl 和 HF 模板：** 一位用户与 **hamelh** 讨论了如何将 Axolotl 微调模板与自定义推理代码匹配，以避免不匹配，因为 Axolotl 不使用 Hugging Face (HF) 模板。
- **对 Token 空间组装的困惑：** 一位新手用户询问为什么 Axolotl 在 Token 空间组装时会添加空格，觉得模板令人困惑。
- **7B Lora 合并导致额外分片的问题：** 一位用户遇到了一个奇怪的现象：合并 7B Lora 导致产生了 6 个分片而不是 3 个，另一位用户建议上传 LoRA 以进行调试。他们还引用了一个相关的 [GitHub issue](https://github.com/bigcode-project/starcoder/issues/137) 并提出了涉及 `torch.float16` 的可能修复方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/bigcode-project/starcoder/issues/137">Model size doubles after .merge_and_unload() and .save_pretrained() · Issue #137 · bigcode-project/starcoder</a>: 我的系统信息 peft==0.4.0 accelerate==0.18.0 transformers==4.28.0 py310 复现：训练后，我使用以下代码将 peft 权重与基础模型合并：model_ft = PeftModel.from_pretrained( AutoModel...</li><li><a href="https://github.com/georgian-io/LLM-Finetuning-Toolkit/blob/7c0413ebedba7ee96d0c17c02f2158c7d3c4c142/inference/text_generation/merge_script.py#L42C29-L42C29">LLM-Finetuning-Toolkit/inference/text_generation/merge_script.py at 7c0413ebedba7ee96d0c17c02f2158c7d3c4c142 · georgian-io/LLM-Finetuning-Toolkit</a>: 用于微调、消融实验和单元测试开源 LLM 的工具包。- georgian-io/LLM-Finetuning-Toolkit
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1248352287932551168)** (5 条消息): 

- **模型加载期间确认 Quantization**：一名成员询问 Quantization 是否在模型加载期间发生，以及 CPU 是否负责该过程。另一名成员确认，表示“它发生在加载模型权重时”，并提供了 [Hugging Face 文档](https://huggingface.co/docs/accelerate/en/usage_guides/quantization) 和相应的 [GitHub](https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44) 代码链接。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>: 未找到描述</li><li><a href="https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44">accelerate/src/accelerate/utils/bnb.py at v0.30.1 · huggingface/accelerate</a>: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持.....
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1248679311184105502)** (1 条消息): 

- **OAuth 集成请求说明**：一名成员感谢另一名成员提供关于 HF 选项的信息，并请求关于实现简单访问控制机制或集成 OAuth 的更多细节。他们特别寻求关于这些安全措施在其实际场景中如何运作的进一步说明。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1248428669513240606)** (2 条消息): 

- **Modal 模块错误已解决**：一名成员遇到了“No module named modal”的错误。他们通过使用命令 `pip install modal` 解决了此问题。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1248521774228832308)** (6 条消息): 

- **Langsmith 积分发放停滞：** 几位成员报告说，尽管填写了必要的表格，但仍未收到 **langsmith credits**。有人明确问道：*“langsmith 积分的状态如何，有人收到了吗？”*。

- **Fireworks 积分表格混淆：** 多位用户提到在提交包含账号 ID 的表格后未收到 **fireworks credits**。一名成员回忆说，当时填表用的是 *“当时的 user ID”* 而不是账号 ID。

- **请求分享账号 ID：** 针对 fireworks 积分缺失的问题，有人请求用户在特定频道 <#1245126291276038278> 中分享他们的账号 ID。

- **6 月 2 日课程积分兑换问题：** 一名成员询问，即使延迟了兑换，是否仍可以兑换 6 月 2 日之前购买的 **course credits**。该问题似乎已通过电子邮件解决，Dan Becker 随后回复道：*“看来我们已经通过电子邮件同步了”*。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[strien_handlingdata](https://discord.com/channels/1238365980128706560/1243773476301443073/1248538319155888129)** (3 条消息): 

- **下载课程录像遇到困难**：由于课程网站现在采用嵌入视频而非跳转至 Zoom 的方式，一名成员在下载课程录像时遇到困难。他们请求获取 Zoom 链接以获得下载权限。

- **Bulk 工具进行翻新**：受课程启发，该成员开始开发其名为“Bulk”工具的下一个版本。由于合成数据集（synthetic datasets）的大量涌现，他们看到了在该领域构建更多工具的价值，并征求反馈。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1248363113246363679)** (11 条消息🔥): 

- **用户大量反馈积分问题**：多位用户对未收到积分表示担忧。他们列出了自己的账号 ID 和电子邮件地址以供参考。
- **AI Engineer World's Fair 邀请**：一名用户邀请另一名用户在 [AI Engineer World's Fair](https://www.ai.engineer/worldsfair) 见面。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1248622455413805129)** (2 条消息): 

- **LoRA Land 论文引发 Finetuning 讨论**：一名用户分享了 [LoRA Land 论文](https://arxiv.org/abs/2405.00732) 并询问这是否会改变另一名用户对 Finetuning 的看法。另一名成员对此表示感兴趣，并用点赞表情符号对分享表示感谢。

### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1248659073382219808)** (1 messages): 

- **Demis Hassabis 在 Tribeca**：一位成员指出 **Demis Hassabis** 今晚将出席 *The Thinking Game* 的首映式。门票售价为 35 美元，活动将包括与 **Darren Aronofsky** 就 AI 和未来进行的对话。[链接](https://x.com/tribeca/status/1798095708777566602?s=46&t=E5D9ecTUOxGQ91MTF5a__g)

**提到的链接**：<a href="https://x.com/tribeca/status/1798095708777566602?s=46&t=E5D9ecTUOxGQ91MTF5a__g">来自 Tribeca (@Tribeca) 的推文</a>：快来听听 @GoogleDeepMind CEO 兼 AI 先驱 @demishassabis 与导演 @DarrenAronofsky 在 #Tribeca2024 上关于 AI、@thinkgamefilm 和未来的对话：https://tribecafilm.com/films/thin...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1248356431225294918)** (7 messages): 

- **Predibase 额度让用户感到兴奋**：一位用户感谢团队成员提供额度，并参考了 Predibase 上的一个[示例](https://docs.predibase.com/user-guide/examples/rag)。另一位用户询问额度何时过期，并注意到升级到 Developer Tier 需要添加信用卡。

- **邮件注册问题**：两名用户报告了在 Predibase 注册后接收确认邮件的问题，导致他们无法登录。

- **LoRAX 给工作坊参与者留下深刻印象**：一位用户称赞 Predibase 工作坊简化了微调和部署 LLM 的流程。他们提到 [LoRAX](https://predibase.github.io/lorax/) 表现尤为突出，它具有极高的成本效益，并有效地集成了将高质量 LLM 部署到 Web 应用程序中的工具。

**提到的链接**：<a href="https://docs.predibase.com/user-guide/examples/rag.">快速入门 | Predibase</a>：Predibase 提供了微调和提供开源 LLM 服务的最快方式。它构建在开源 LoRAX 之上。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1248519021872283648)** (19 messages🔥): 

- **Cursor：基于 VS-Code 的 AI 代码编辑器令人印象深刻**：成员们讨论了 **Cursor** 的功能，这是一款基于 **VS-Code** 的 AI 驱动代码编辑器。一位成员指出了它的核心优势：“*它会索引你的整个代码库，以便进行彻底的多文件和多位置更改*”。
  
- **Pro 订阅版因自动代码补全受到称赞**：一位用户指出使用 **Cursor 付费版** 带来了显著收益，特别强调了在使用 **GPT-4** 时*自动代码补全*方面的改进。
  
- **自定义 API key 让 Cursor 更加灵活**：Cursor 的集成允许用户输入自己的 API key，用于 **OpenAI, Anthropic, Google, 和 Azure** 等服务。这一功能受到了好评，因为它允许用户以自己的成本进行广泛的 AI 交互。
  
- **Cursor 相比 GitHub Copilot 更受青睐**：从 **GitHub Copilot** 切换到 **Cursor** 的用户分享了积极的体验，强调了生产力的提升和满意度。一位用户提到：“在使用 Copilot 大约一两年后，我完全切换了过来，并且再也没有回头。”

- **VS-Code 爱好者欢迎 Cursor**：讨论了长期 VS-Code 用户对 Cursor 的采用，指出由于 Cursor 是基于**开源 VS-Code** 构建的，它在保留熟悉功能的同时增加了 AI 驱动的增强功能。一位成员的背书：“*它感觉更好，你只需享受这些改进*”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.sh/miscellaneous/api-keys">Cursor - 更快地构建软件</a>：未找到描述</li><li><a href="https://www.cursor.com/">Cursor</a>：AI 代码编辑器
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1248578981343793163)** (2 messages): 

鉴于提供的聊天记录，没有足够的信息进行实质性总结。消息中没有提供重要的主题、讨论点、链接或感兴趣的博客文章。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1248357426823041177)** (29 messages🔥): 

- **新的并行函数调用功能**：用户现在可以通过设置 `"parallel_tool_calls: false"` 来禁用并行函数调用，这是 OpenAI 最近发布的一项功能。

- **额度有效期澄清**：OpenAI 的额度自发放之日起 **三个月** 后过期。这是针对用户关于额度使用截止日期查询的明确回复。

- **GPT-4 访问问题**：多位用户报告了访问 GPT-4 模型的问题以及 Rate Limits 的差异。建议受影响的用户发送邮件至 **shyamal@openai.com** 并抄送 **support@openai.com** 以寻求解决。

- **Cursor 和 iTerm2 使用额度**：第三方工具如 [Cursor](https://docs.cursor.sh/miscellaneous/api-keys) 和 [iTerm2](https://iterm2.com/ai-plugin.html) 允许用户使用他们的 OpenAI 额度。这些集成在使用各种 AI 模型和自带 API Key 方面提供了灵活性。

- **额度使用创意**：一位用户分享了一个 [Twitter 链接](https://twitter.com/m_chirculescu/status/1799174718286684245?t=gA7oEwPtbq9SuFC-tl6hSA&s=19)，列出了使用额度的创意方法，并邀请其他人分享更多点子。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cursor.sh/miscellaneous/api-keys">Cursor - Built Software Faster</a>: 无描述</li><li><a href="https://iterm2.com/ai-plugin.html">iTerm2 - macOS Terminal Replacement</a>: 无描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[capelle_experimentation](https://discord.com/channels/1238365980128706560/1248386323035721810/1248512841468874854)** (82 messages🔥🔥): 

- **tcapelle 的精彩演讲和资源**：tcapelle 分享了一个内容丰富的演讲，并附带了 [Slides](http://wandb.me/llm-finetune-slides) 和 Colab notebook ([connections](https://wandb.me/connections)) 链接，解释了 LLM Finetuning 中的各种概念。提供的 GitHub 仓库 ([connections](https://github.com/wandb/connections)) 包含了示例和代码。

- **Finetuning 技巧与社区互动**：tcapelle 建议对 Llama3 进行几个 Epoch 的训练，并使用大型 LLM 进行评估，强调了社区在 Mistral 和 Llama 系列模型上的丰富经验。他推荐使用 Alpaca 风格的评估以获得准确的性能指标。

- **快速学习与剪枝（Pruning）见解**：一篇 [Fast.ai 文章](https://www.fast.ai/posts/2023-09-04-learning-jumps/) 讨论了神经网络仅凭极少示例进行记忆的能力。tcapelle 在 GitHub 上分享了剪枝脚本 ([create_small_model.py](https://github.com/tcapelle/shear/blob/main/create_small_model.py))，并推荐观看 [GTC 2024 演讲](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62348/) 以获取优化 LLM 的见解。

- **Weave 工具包集成与功能**：Scottire 提到了 Weave 中即将推出的样本策展功能，并提供了一个用于向数据集添加行的代码片段。分享了 [Weave 的 GitHub](https://github.com/wandb/weave) 链接及其 [文档](https://wandb.github.io/weave/)，以帮助用户将 Weave 集成到他们的工作流中。

- **引人入胜的社区创意**：成员们讨论了组织每周见面会、论文俱乐部和工作组，以便进行协作学习和想法分享。关于“BRIGHT CLUB”的幽默建议以及各种社区联谊活动，突显了该小组充满活力且互相支持的氛围。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wandb.me/weave">简介 | W&amp;B Weave</a>：Weave 是一个由 Weights &amp; Biases 构建的轻量级工具包，用于跟踪和评估 LLM 应用。</li><li><a href="https://www.fast.ai/posts/2023-09-04-learning-jumps/">fast.ai - LLM 能从单个示例中学习吗？</a>：我们在微调 LLM 时注意到一种不寻常的训练模式。起初我们以为这是一个 bug，但现在我们认为这表明 LLM 可以有效地从单个示例中学习。</li><li><a href="https://wandb.ai/muellerzr/llama-3-8b-self-align-axolotl">muellerzr</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://m.youtube.com/watch?v=VWdRQL0CsAk">欢迎来到模型 CI/CD 课程！</a>：首先，让我们深入探讨为什么模型管理是一个如此重要的课题，为了激发兴趣，让我向你们展示我最近一个项目中的真实案例。在...</li><li><a href="https://wandb.me/gtc2024">GTC 2024：优化 LLM，一种剪枝和微调 7B 模型的实验性方法</a>：我们如何从大模型中制造出小模型！由 Thomas Capelle 使用 Weights &amp; Biases 制作</li><li><a href="https://github.com/tcapelle/shear/blob/main/create_small_model.py">tcapelle/shear 项目 main 分支下的 shear/create_small_model.py</a>：LLM 的剪切与剪枝。通过在 GitHub 上创建账号来为 tcapelle/shear 的开发做出贡献。</li><li><a href="https://github.com/wandb/weave">GitHub - wandb/weave: Weave 是一个由 Weights &amp; Biases 构建的用于开发 AI 驱动应用的工具包。</a>：Weave 是一个由 Weights &amp; Biases 构建的用于开发 AI 驱动应用的工具包。 - wandb/weave</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62348/">优化大语言模型：一种剪枝和微调 Llama2 7B 的实验性方法 | NVIDIA On-Demand</a>：面对大语言模型（LLM）的高计算需求，我们提出了一种模型剪枝和微调的实验性方法，以克服...</li><li><a href="https://github.com/t">t - 概览</a>：t 有 14 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/">Inspect</a>：用于大语言模型评估的开源框架</li><li><a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect：一个用于大语言模型评估的框架</a>：Inspect：一个用于大语言模型评估的框架 - UKGovernmentBEIS/inspect_ai</li><li><a href="https://wandb.me/connections">Google Colab</a>：未找到描述</li><li><a href="http://wandb.me/llm-finetune-slides">Hamel 的 LLM 课程</a>：充分利用你的 LLM 实验。Weights &amp; Biases 的 ML Engineer Thomas Capelle。嘿，我是 Thomas，Weights &amp; Biases 的 ML Engineer。让我们来谈谈在生产环境中迭代你的 LLM 应用...</li><li><a href="https://github.com/wandb/connections">GitHub - wandb/connections: 解决《纽约时报》Connections 拼图</a>：解决《纽约时报》Connections 拼图。通过在 GitHub 上创建账号来为 wandb/connections 的开发做出贡献。</li><li><a href="https://wandb.ai/augmxnt/shisa-v2?nw=nwuserrandomfoo">augmxnt</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.github.io/weave/">简介 | W&amp;B Weave</a>：Weave 是一个由 Weights &amp; Biases 构建的轻量级工具包，用于跟踪和评估 LLM 应用。</li><li><a href="https://www.youtube.com/watch?app=desktop&v=VWdRQL0CsAk">欢迎来到模型 CI/CD 课程！</a>：首先，让我们深入探讨为什么模型管理是一个如此重要的课题，为了激发兴趣，让我向你们展示我最近一个项目中的真实案例。在...</li><li><a href="https://www.youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6">模型 CI/CD 课程</a>：克服模型混乱，自动化关键工作流，确保治理，并简化端到端的模型生命周期。本课程将为你提供相关概念...</li><li><a href="https://colab.research.google.com/github/wandb/connections/blob/main/00_getting_started.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://wandb.ai/llm_surgery/shearllama/reports/GTC-2024-Optimizing-LLMs-An-experimental-approach-to-pruning-and-fine-tuning-7B-models--Vmlldzo3MjM0Mjc4">GTC 2024：优化 LLM，一种剪枝和微调 7B 模型的实验性方法</a>：我们如何从大模型中制造出小模型！由 Thomas Capelle 使用 Weights &amp; Biases 制作
</li>
</ul>

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1248401923938320526)** (1 条消息): 

- **Perplexity 发布《The Know-It-Alls》官方预告片**：分享了一个指向 [YouTube 视频，标题为 "The Know-It-Alls" by Perplexity](https://www.youtube.com/watch?v=QfoulVr6UU8) 的链接。视频描述提出了一个引人深思的问题：*“如果世界上所有的知识都触手可及，我们能否突破可能的边界？我们即将揭晓答案。”*

**提到的链接**：<a href="https://www.youtube.com/watch?v=QfoulVr6UU8">&quot;The Know-It-Alls&quot; by Perplexity | Official Trailer HD</a>：如果世界上所有的知识都触手可及，我们能否突破可能的边界？我们即将揭晓答案。加入搜索，寻找答案...

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1248359965891100763)** (493 条消息 🔥🔥🔥): 

- **Perplexity 的时区与服务器之谜**：成员们讨论了 Perplexity 遵循的时区，有人指出这可能取决于服务器位置，另一位猜测可能是 **+2 或类似时区**。一位用户询问如何查找服务器的确切位置信息。
  
- **持久附件（Persistent Attachments）问题**：一位用户对 Perplexity AI 在多次查询中仍纠结于一个无关的临时上下文文件表示沮丧。另一位解释说，开启新线程（Thread）可以解决这种上下文持久性问题。

- **《The Know-It-Alls》广告令观众失望**：成员们原本期待《The Know-It-Alls》的首映会带来重大更新或新功能，但结果证明这只是一个**纯宣传视频**，让许多人感到被耍了且大失所望。随后的评论将其比作超级碗广告。

- **Pro Search 与 Claude 3 的讨论**：讨论了关于 **Pro Search** 和 **Claude 3** 等 AI 模型的各种问题和偏好。用户分享了使用经验，一些人注意到最近从 Perplexity Labs 中移除了 **Claude 3 Haiku**。

- **赛马查询测试 PPLX**：用户使用赛马结果查询测试了 PPLX，根据搜索时间的不同，结果和准确性也各异。讨论显示，使用像 **<scratchpad-think>** 这样的结构化 Prompt 有助于使 AI 的推理过程更清晰、更准确。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.youtube.com/watch?v=QfoulVr6UU8">Perplexity 的 "The Know-It-Alls" | 官方高清预告片</a>：如果全世界的知识都触手可及，我们能否突破可能的界限？我们即将揭晓。加入搜索，寻找答案...</li><li><a href="https://tenor.com/view/xzibit-meme-inception-gif-13033570">Xzibit 表情包 GIF - Xzibit 表情包 Inception - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity 模型选择</a>：使用 jQuery 为 Perplexity AI 添加模型选择按钮</li><li><a href="https://www.perplexity.ai/search/tell-me-who-DzuCADD0TF6wCFjRjyvCsw#0">告诉我昨天 Hamilton 4:30 比赛的获胜者。这与赛马有关...</a>：2024年6月6日 Hamilton 4:30 比赛的获胜者是 Hey Lyla。这匹马由骑师骑乘，练马师训练，获得了第一名...</li><li><a href="https://docs.google.com/document/d/1ZxWtnHl5QAZxHtNu6-_2QC19pFjnNBIf4CbEgMOxcHs/edit?usp=sharing">18:05 Goodwood 2024年6月7日</a>：18:05 Goodwood 2024年6月7日 William Hill 每日助力见习骑师让赛。根据提供的每匹马的表现信息和其他因素，以下是比赛分析：Out...</li><li><a href="https://labs.perplexity.ai">Perplexity Labs</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/Yemens-Houthi-rebels-Y7pSGgxTQc2rUaTIwb_AGA">也门胡塞武装在突然行动中扣押了至少 9 名联合国工作人员及其他人...</a>：&lt;scratchpad&gt; [关键信息] 标题：官员称，也门胡塞武装在突然镇压行动中扣押了至少 9 名联合国工作人员及其他人。作者：Jon Gambrell...</li><li><a href="https://docs.google.com/document/d/1rj-BAeTmAc02hSATc_wuRwuO5o5ID5gMi8G_ugdAPcU/edit?usp=sharing">20:10 Goodwood 2024年6月7日</a>：20:10 Goodwood 2024年6月7日 综合稳健系统分析 1. Skysail (279) 状态：22/1，在 Sandown 让赛中获得 14 名中的第 10 名（10 浪，好地至软地）。休赛 9 个月。记录：赛道与距离 (CD)：...</li><li><a href="https://docs.google.com/document/d/1gJGTZxmstXAg5JdcGbdSdfDfbN7sKO8rEeEJyBS-HiM/edit?usp=sharing">18:45 Bath 2024年6月7日</a>：18:45 Bath 2024年6月7日 Betting.bet 实时赛马结果让赛。为了对 Bath 18:45 的比赛应用综合稳健系统，我们将结合 Pace Figure Patterns 和 dosage ratings。让我们...</li><li><a href="https://docs.google.com/document/d/14gcrycsKEHY3uMNkeEYttCMW3u7nm1_HkaaLTOSrR6Y/edit?usp=sharing"> 21:00 Bath 2024年6月7日</a>：21:00 Bath 2024年6月7日 Mitchell &amp; Co 让赛 综合稳健系统分析。让我们对比赛应用综合稳健系统，结合 Pace Figure Patterns 和 dosage...</li><li><a href="https://www.sportinglife.com/racing/results/2024-06-06/hamilton/801296/sodexo-live-handicap">16:30 Hamilton - 2024年6月6日 - 结果 - 赛马 - Sporting Life</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/Horse-Race-results-cU1gbCanT9iVXZXim3YF7A#0">2024年6月6日 Hamilton 4:30 / 16:30 赛马结果。报告...</a>：&lt;scratchpad&gt; [记录从提示词中提取的任何关键信息，例如假设、证据或任务指令] 用户想要以下比赛的结果...</li><li><a href="https://monica.im/home">Monica - 您的 ChatGPT AI 助手 Chrome 扩展程序</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/The-user-below-4WdPVuwYQFiVc1un.nXF3w">下方的用户正在投诉在使用时无法获得更新的信息...</a>：&lt;scratchpad&gt; 提示词中的关键信息：用户投诉 Perplexity AI 提供过时或错误的信息。请求一个关于...的示例</li><li><a href="https://www.perplexity.ai/search/scratchpad-9OmgWxf5QvaTxNDHUCUMQw">&lt;scratchpad&gt;</a>：根据提供的搜索结果，以下是 Anthropic 的 Claude AI 模型如何利用思维链 (CoT) 推理和 scratchpad 功能：Anthropic 的...</li><li><a href="https://www.perplexity.ai/search/What-is-scratchpad-vjKtl.d9QdqFaBjwAkv4ig">在 LLM / GPT 语境下，什么是 scratchpad 思维？</a>：在 GPT 等 LLM 的语境下，Scratchpad 思维是指一种设计模式，旨在帮助这些模型处理大型且复杂的数据...</li><li><a href="https://www.perplexity.ai/collections/scratchpadthink-wBPEohuUQH6tz5qMlH4F7g">&lt;scratchpad-think&gt;</a>：paradroid 在 Perplexity AI 上的一个合集 — </li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w">Complexity：Perplexity 的新扩展</a>：Perplexity AI 的 Complexity 扩展引入了一系列强大的功能，旨在增强用户体验并简化与...的交互</li><li><a href="https://www.attheraces.com/form/horse/A-Girl-Named-Ivy/IRE/3530199">A Girl Named Ivy |</a></li>

Horse Profile &amp; Next Race Odds | At The Races</a>: 从 At The Races 获取关于 A Girl Named Ivy 的最新信息，包括下一场比赛、最新的赛马赔率、以往胜绩、马主信息以及更多内容。
 </li><li><a href="https://www.perplexity.ai/search/I-am-a-.9Wxh.lpTBWJtw2vJrYzVQ">Perplexity</a>: 未找到描述</li><li><a href="https://monica.im">Monica - Your ChatGPT AI Assistant for Anywhere</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/fully-review-the-wW9gClrfRAWhD17DkDLsAQ#0">Review alternate version of Scratchpad</a>: &lt;scratchpad&gt; [记录从提示词中提取的关键信息，例如假设、证据或任务指令] 提供的框架概述了...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1248419712157749388)** (16 条消息🔥): 

- **SpaceX 的 Starship 取得里程碑式进展**：SpaceX 在 2024 年 6 月 6 日进行的 Starship 发射系统第四次试飞标志着重大进展。[此次飞行包括在墨西哥湾和印度洋的成功溅落](https://www.perplexity.ai/page/Starship-Test-4-QCcbPm1tQay1u.pc9bAXVg)。

- **亚洲咖喱品种对比**：对日本、印度和泰国咖喱的全面对比突显了它们各自的独特特征。[详细对比](https://www.perplexity.ai/page/Comparing-Asian-Curry-bpXXIu9gTiKcWtcFzxKizw)包括历史起源、香料、草药和典型食材。

- **对生成内容的担忧**：生成内容存在不准确的问题，特别是关于 playground.com 和 playground.ai 的对比。[对 Playground.com 的描述不准确](https://www.perplexity.ai/search/Difference-between-the-cDv9wJcqQvuWHnG95ZMfog)。

- **对加州参议院第 1047 号法案的反应**：各利益相关方对专注于 AI 安全的加州参议院第 1047 号法案（California Senate Bill 1047）做出了反应。[AI Safety 倡导者赞扬该法案为 AI 公司建立了明确的法律标准](https://www.perplexity.ai/search/california-bill-1047-TtVwNj_4TTWt73hgf7QTzg)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/Ygpti3sGizY">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/List-each-of-Y3RyPx4ySgCIUtTlWk60Mg">列出投资者在交易中关注的每个常见数据点，以及...</a>: 投资者使用各种数据点来做出明智的交易决策。以下是一些最常见的：1. 价格波动：...的变化</li><li><a href="https://www.perplexity.ai/page/Starship-Test-4-QCcbPm1tQay1u.pc9bAXVg">Starship 第四次测试：成功！</a>: SpaceX 在 2024 年 6 月 6 日成功的第四次试飞中，实现了 Starship 发射系统开发的重大里程碑。此次飞行见证了...</li><li><a href="https://www.perplexity.ai/search/define-what-indexing-NQ4.QH0SS5iBh_xPNAeYvg">Web Indexing：与 LLM 搜索工具的关系</a>: &lt;scratchpad&gt; 关键信息：在线发布新文章背景下的 Indexing。Indexing 与 Google、带有搜索功能的 LLM 等系统相关...</li><li><a href="https://www.perplexity.ai/search/wie-kann-ich-ZEW4okwWSIW7xuES32eTSQ#0">如何使用 Perplexity 生成图片？</a>: 要使用 Perplexity AI 创建图像，请按照以下步骤操作：1. 需要 Pro 账户：确保您拥有 Perplexity...</li><li><a href="https://www.perplexity.ai/search/Yemens-Houthi-rebels-Y7pSGgxTQc2rUaTIwb_AGA">也门胡塞武装在突袭行动中拘留了至少 9 名联合国工作人员及其他人...</a>: &lt;scratchpad&gt; [关键信息] 标题：据官员称，也门胡塞武装在突然镇压行动中拘留了至少 9 名联合国工作人员及其他人 作者：Jon Gambrell...</li><li><a href="https://www.perplexity.ai/search/Difference-between-the-cDv9wJcqQvuWHnG95ZMfog">playground.com 和 playground.ai 两个网站的区别</a>: 这两个网站 playground.com 和 playground.ai 用途不同，面向的受众也不同。以下是基于...的详细对比</li><li><a href="https://www.perplexity.ai/search/comment-installler-odoo-tozrdj7ARk6xRzEqvrUMLQ">如何安装 Odoo 进行收银管理</a>: 要安装用于收银管理的 Odoo，请遵循以下步骤：1. 安装 Odoo Community：- 安装 Odoo Community v9。- 创建...</li><li><a href="https://www.perplexity.ai/search/california-bill-1047-TtVwNj_4TTWt73hgf7QTzg">加州 1047 号法案</a>: 由参议员 Scott Wiener 提出的加州参议院第 1047 号法案 (SB 1047) 旨在监管先进人工智能的开发和部署...</li><li><a href="https://www.perplexity.ai/search/Starship-Test-4-l5qs.WoNSkOmJIXJU85gMA">Starship 第四次测试：成功！</a>: &lt;scratchpad&gt; 提示词中的关键信息：SpaceX 于 2024 年 6 月 6 日进行了 Starship 发射系统的第四次试飞。飞行器于...起飞</li><li><a href="https://www.perplexity.ai/page/Comparing-Asian-Curry-bpXXIu9gTiKcWtcFzxKizw">亚洲咖喱品种对比</a>: 咖喱是许多国家喜爱的菜肴，但不同菜系之间的口味和食材差异很大。日本、印度和泰国...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1248405652829638786)** (4 条消息): 

- **关于 API 中 llava 模型的疑问**：一位成员询问是否有计划允许在 API 中使用 **llava** 模型，并提到它已从 labs 中移除。提供的消息中没有记录任何回复。
- **对 sources beta 测试的挫败感**：成员们对 **sources beta testing** 表示沮丧，其中一人提到：*"我发誓我已经填了 5 次这个表单了。"* 他们质疑是否允许新人加入该 beta 计划。
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1248372767435526217)** (1 条消息): 

- **FluentlyXL 最终版发布**：**FluentlyXL Final** 版本现已发布，在美学和光影方面有所增强。在 [Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final) 页面查看更多详情。
- **碳足迹预测器发布**：一款用于预测碳足迹的新工具现已上线。在 [Carbon Footprint Predictor](https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor) 页面了解更多信息。
- **SimpleTuner 更新并支持 MoE**：**SimpleTuner** 0.9.6.2 版本包含 MoE 分层时间戳训练支持和简短教程。在 [GitHub](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2) 上开始使用。
- **LLM 资源指南汇编**：一份整理好的 LLM 讲解指南现已发布，涵盖了 vLLM, SSMs, DPO 和 QLoRA。在 [resource guide](https://x.com/willccbb/status/1798423849870270671) 中查看详情。
- **使用 TensorFlow 的 ML 库**：一个基于 TensorFlow 的新 **ML Library** 已发布。在 [GitHub](https://github.com/NoteDance/Note) 上查找源代码和文档。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://isamu-website.medium.com/understanding-triton-tutorials-part-2-f6839ce50ae7)">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/tcfd_counselor">Tcfd Counselor - 一个由 as-cle-bert 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/willccbb/status/1798423849870270671)">来自 will brown (@willccbb) 的推文</a>：过去一年里一直在学习 LLMs 等知识，将一些我最喜欢的讲解整理成了一份“教科书式”的资源指南，希望我刚开始学习时就能拥有它，也许它对其他人也有用...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1248352212573229277)** (248 messages🔥🔥): 

- **虚拟环境与包管理器的讨论**：成员们就使用 conda 还是 pyenv 来管理 Python 环境展开了辩论。一位用户表达了对 conda 的不满，更倾向于使用 pyenv，而另一位用户则承认自己一直在使用全局 pip 安装且没有遇到重大问题。
  
- **GPT 与 PyTorch 版本兼容性**：一位用户指出 Python 3.12 目前尚不支持 PyTorch。这引发了更多关于在不同项目中维护不同 Python 版本兼容性挑战的讨论。

- **HuggingFace 与学术研究查询**：一位成员询问了在不自行托管模型的情况下，使用 HuggingFace AutoTrain 进行学术项目的可行性。回复褒贬不一，指出可用的免费服务可能不支持较大的模型，并且可能需要支付 API 费用。

- **点击率与时尚度**：用户讨论了利用 AI 通过分析点击率和时尚评分来预测并生成高点击率的 YouTube 缩略图或时尚服装的想法。建议使用强化学习 (PPO) 和其他方法来针对人类偏好进行优化。

- **Gradio 隐私担忧**：提到了在保持访问权限的同时将 Gradio 应用设为私有的相关问题。此外，还讨论了关于潜在漏洞以及在不同仓库中更新 Gradio 版本的担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/ai-competition/MMFMChallenge">MMFMChallenge - ai-competition 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/v0.8.2/en/developer_guides/model_merging">Model merging</a>：未找到描述</li><li><a href="https://huggingface.co/intone/unaligned-llama3-8b-v0.1-16bit">intone/unaligned-llama3-8b-v0.1-16bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/intone/AmminoLoRA">intone/AmminoLoRA · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/discord-gif-27442765">Discord GIF - Discord - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/docs/peft/v0.8.2/en/developer_guides/lora#merge-adapters">LoRA</a>：未找到描述</li><li><a href="https://tenor.com/view/microsoft-windows-edge-microsoft-edge-gif-26202666">Microsoft Windows GIF - Microsoft Windows Edge - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/intone/Ammino-1.1B">intone/Ammino-1.1B · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/sbG4.gif">Lung Test Prank GIF - Pranks - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1248462488425594893)** (1 messages): 

- **医学教育中的 AI 概述**：一段新的 YouTube 视频讨论了 **genAI** 在医学教育中的现状及其未来轨迹。主题包括学习格局、使用 **Anki** 进行主动回忆、通过 AddOns 进行自定义，以及通过 Perplexity 集成 **genAI 驱动的搜索**。[AI In MedEd YouTube 视频](https://youtu.be/kZMcNCV_RXk)。

**提到的链接**：<a href="https://youtu.be/kZMcNCV_RXk">AI In MedEd: In 5* minutes</a>：在这一新系列的第一集中，我们将探讨 genAI 目前在医学教育中的地位以及它可能的发展方向。1: MedEd 的学习...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1248358634568548486)** (2 messages): 

- **成员寻求 AI 合作**：一位高级 Web 开发人员表示有兴趣在 AI 和 LLM 项目上进行合作。他们请求有兴趣组织合作的人员联系他们。
- **TorchTune 介绍**：一位成员分享了 [TorchTune 的链接](https://github.com/pytorch/torchtune)，这是一个专为 LLM 微调设计的原生 PyTorch 库。描述强调了其在利用 PyTorch 微调大语言模型中的作用。

**提到的链接**：<a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1248363044208119920)** (13 条消息🔥): 

- **Fluently XL 通过脚本生成出色的人像**：一位用户分享了使用 Fluently XL 进行文本到图像生成的经验，结合了 ControlNet 和图像提示词。他们提供了所使用的两个 GitHub 仓库链接：[第一个脚本](https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py) 和 [第二个脚本](https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py)。

- **关于阅读小组机会的讨论**：成员们讨论了建立一个专注于编程教程的阅读小组的可能性。他们同意在下周的通常时间开始第一次活动。

- **多 Agent BabyAGI 系统演示**：一位用户分享了 [Loom](https://www.loom.com/share/5b84b9284e2849f8bd2ca730c97c3f40?sid=f0a9a781-7bba-4903-aa16-0a20c0c76e7c) 上的演示视频链接，展示了 BabyAGI 作为一个真正的多 Agent 系统，其 Agent 运行在不同的节点上。

- **用于运行时检测的新 Rust crate**：一位用户宣布了一个用于在运行时检测操作系统的全新 Rust crate，并分享了他们的学习过程。他们提供了 [项目链接](https://dev.to/dhanushnehru/announcing-runtime-environment-a-rust-crate-for-detecting-operating-systems-at-runtime-3fc2) 并寻求支持。

- **在喜剧场景中使用搭载 dolphin-Llama3 模型的 Droid**：一位用户幽默地分享了一个运行 dolphin-Llama3 模型的 Droid 对脱口秀和吐槽大会演出的反应。另一位用户对这些视频表示了喜爱和赞赏。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dev.to/dhanushnehru/announcing-runtime-environment-a-rust-crate-for-detecting-operating-systems-at-runtime-3fc2">未找到标题</a>：未找到描述</li><li><a href="https://www.loom.com/share/5b84b9284e2849f8bd2ca730c97c3f40?sid=f0a9a781-7bba-4903-aa16-0a20c0c76e7c">NAPTHA 工具和基础设施更新 🛠️</a>：嗨，我是 Richard，在这个视频中，我将带你了解 NAPTHA 的最新更新，重点关注 Microsoft AutoGen 和 Crew AI 等多 Agent 框架。我们正致力于构建真正的多...</li><li><a href="https://www.instagram.com/p/C6wP_q-rwIS/?igsh=MWQ1ZGUxMzBkMA==">Mansion X 在 Instagram："准备出发 #ootd #ootdfashion Maude Mongeau 为 @the_mansion_x 代言"</a>：3 个赞，1 条评论 - the_mansion_x 于 2024 年 5 月 9 日发布："准备出发 #ootd #ootdfashion Maude Mongeau 为 @the_mansion_x 代言"。</li><li><a href="https://github.com/Saganaki22/StableAudioWebUI">GitHub - Saganaki22/StableAudioWebUI</a>：通过在 GitHub 上创建账号来为 Saganaki22/StableAudioWebUI 的开发做出贡献。</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>：深度学习的 Monorepo（单一或 "mono" 仓库）。- InServiceOfX/InServiceOfX</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>：深度学习的 Monorepo（单一或 "mono" 仓库）。- InServiceOfX/InServiceOfX
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1248355550333501524)** (13 条消息🔥): 

- **往期演讲汇编的 GitHub 仓库**：一位成员分享了一个 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group)，将其描述为 HuggingFace 阅读小组所有往期演讲的汇编。 

- **即将举行的会议提醒**：发布了关于即将召开会议的提醒，讨论的论文见 [此处](https://arxiv.org/abs/2102.06794)。该论文专注于在用于物理模拟的神经网络中引入归纳偏置（inductive bias），特别是解决接触力学（contact mechanics）问题。

- **可微接触模型论文概览**：该论文的高层级概览可通过 [YouTube 视频](https://www.youtube.com/watch?v=DdJ7RLmG0kg) 查看，文中为神经网络引入了一个可微接触模型（differentiable contact model）。该论文的配套代码可以在 [GitHub](https://github.com/Physics-aware-AI/DiffCoSim) 上找到。

- **Discord Human Feedback 邀请问题**：一位用户报告在尝试加入 Human Feedback 时遇到问题，收到了无效邀请消息。另一位成员做出了回应，并提议直接发送一个有效的链接，但由于该用户的隐私设置而遇到障碍。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2102.06794">Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models</a>：引入适当的归纳偏置在从数据中学习动力学方面起着至关重要的作用。越来越多的研究工作一直在探索在学习到的动力学中强制执行能量守恒的方法...</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group</a>：该仓库的目标是预编译 HuggingFace 阅读小组的所有往期演讲 - isamu-isozaki/huggingface-reading-group</li><li><a href="https://www.youtube.com/watch?v=DdJ7RLmG0kg">Extending Lagrangian &amp; Hamiltonian Neural Networks with Differentiable Contact Models | NeurIPS 2021</a>：论文：https://arxiv.org/abs/2102.06794 (arXiv) ********** 摘要 ********** 引入适当的归纳偏置在学习中起着至关重要的作用...</li><li><a href="https://github.com/Physics-aware-AI/DiffCoSim">GitHub - Physics-aware-AI/DiffCoSim</a>：通过引入可微接触模型，DiffCoSim 扩展了受 Lagrangian/Hamiltonian 启发的神经网络的适用性，从而能够学习混合动力学。 - Physics-aware-AI/DiffC...
</li>
</ul>

</div>

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1248408351369465947)** (10 条消息🔥): 

- **尽管存在技术问题，Hangout 仍获得积极反馈**：一位成员对周末的 Hangout 活动表示感谢，尽管遇到了一些技术问题。他们期待下一次活动。

- **为 VLM 创建实用应用**：一位成员分享了他们创建的两个 Hugging Face Spaces 应用，分别是 **Model Explorer** [Model Explorer](https://huggingface.co/spaces/dwb2023/model_explorer2) 和 **HF Extractor** [HF Extractor](https://huggingface.co/spaces/dwb2023/hf_extractor)，它们对 “VLM 应用特别有帮助”。他们还提供了一个 YouTube 视频来解释这些应用的创作动机：[YouTube 视频](https://www.youtube.com/watch?v=w67fQ_-8hq0)。

- **开源 ML 新贡献者的路径**：一位开源 ML 的新人寻求关于如何贡献和成长的指导。一位经验丰富的成员建议寻找 “good first issues”，阅读 CONTRIBUTING.md 文件，并尝试通过遵循标准和从现有的 PR 中学习来进行贡献。

- **transformers 库的 Good first issues**：建议将一些 GitHub issues 作为新贡献者的良好起点，例如 [Move weight initialization for DeformableDetr](https://github.com/huggingface/transformers/issues/29818) 和 [Adding Flash Attention 2 support for more architectures](https://github.com/huggingface/transformers/issues/26350)。这些建议旨在帮助新贡献者参与实际任务。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=w67fQ_-8hq0">揭开开源 VLM 的神秘面纱：使用 Model Explorer 加速你的原型设计</a>：在这段精彩的视频中，我深入探讨了视觉语言模型 (VLM) 的世界，并展示了两个旨在加速你初始阶段的创新应用...</li><li><a href="https://huggingface.co/spaces/dwb2023/model_explorer2">Model Explorer - dwb2023 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/dwb2023/hf_extractor">Hf Extractor - dwb2023 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/issues/29818">为 DeformableDetr 移动权重初始化 · Issue #29818 · huggingface/transformers</a>：系统信息不相关。复现请参见 Deformable Detr Modeling。预期行为：所有权重初始化都应在 xxxPretrainedModel 类的 _init_weights 中完成。</li><li><a href="https://github.com/huggingface/transformers/issues/26350">社区贡献：为更多架构添加 Flash Attention 2 支持 · Issue #26350 · huggingface/transformers</a>：功能请求。Flash Attention 2 是一个提供注意力操作内核的库，用于更快、更节省内存的推理和训练：https://github.com/Dao-AILab/flash-attention...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1248556261440749608)** (3 条消息): 

- **多节点微调咨询引起关注**：一位成员询问是否有人尝试过对 LLM 进行多节点微调。这引出了一篇由多位作者共同发表的 [arXiv 论文](https://arxiv.org/abs/2405.17247)，讨论了相关研究。
- **寻求实践指导**：针对提供的学术资源，另一位成员询问是否有实操资源或实践指导。在提供的消息中，对话没有进一步展开。

**提到的链接**：<a href="https://arxiv.org/abs/2405.17247">视觉语言建模简介</a>：随着近期大语言模型 (LLM) 的普及，人们进行了多次尝试将其扩展到视觉领域。从拥有一个可以引导我们穿过陌生环境的视觉助手...

  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1248453857533759538)** (8 条消息🔥): 

- **通过 Text-to-Image 示例探索 Diffusers**：一位成员分享了用于 Text-to-Image 生成脚本的 [Diffusers GitHub 仓库链接](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/)。另一位成员指出，该脚本可能用于微调模型。
- **使用原生 PyTorch 进行优化推理**：针对使用原生 PyTorch 进行优化推理，推荐了[这篇博文](https://pytorch.org/blog/accelerating-generative-ai-3/)。它涵盖了多种优化技术，如 bfloat16 精度、scaled dot-product attention、torch.compile 以及动态 int8 量化，可将 Text-to-Image Diffusion 模型的加速提升高达 3 倍。
- **从零开始训练模型**：一位成员表示有兴趣通过从零开始训练模型来生成类似 MNIST 的数据集样本。他们被引导至另一个资源，即一个[训练示例 Notebook](https://github.com/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)，其中提供了无条件生成的示例。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-generative-ai-3/">Accelerating Generative AI Part III: Diffusion, Fast</a>: 这篇文章是一个系列博客的第三部分，重点介绍如何使用纯原生 PyTorch 加速生成式 AI 模型。我们很高兴能分享一系列新发布的 PyTorch 性能 ...</li><li><a href="https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/">diffusers/examples/text_to_image at main · huggingface/diffusers</a>: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的先进 Diffusion 模型。 - huggingface/diffusers</li><li><a href="https://github.com/huggingface/notebooks/blob/main/diffusers/training_example.ipynb">notebooks/diffusers/training_example.ipynb at main · huggingface/notebooks</a>: 使用 Hugging Face 库的 Notebooks 🤗。通过在 GitHub 上创建账号来为 huggingface/notebooks 的开发做出贡献。
</li>
</ul>

</div>

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1248388908526604369)** (236 messages🔥🔥): 

- **新手面对 AI 模型感到不知所措**：一位新用户表达了他们最初对创建自己模型的兴奋和困惑，原因是基础模型数量庞大，并表示“*……这太疯狂了！！！*”。
- **ControlNet 减慢图像生成速度**：一位名为 _arti0m_ 的用户报告称，在使用 ControlNet 时图像生成时间显著变长，并表示困惑，因为他预期输出会更快，但却经历了长达 20 分钟的等待。
- **CosXL 模型生成更好的色调范围**：用户讨论了 Stability.ai 的新 CosXL 模型，该模型具有改进的色调范围，从“漆黑”到“纯白”。[模型链接](https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/)。
- **图像生成中的 VRAM 和内存问题**：一段对话强调了 Web UI Automatic1111 的内存管理问题，认为它可能导致过度的 VRAM 占用，从而显著减慢图像生成速度。
- **关于中国瀑布丑闻的有趣辩论**：用户对涉及中国某人造瀑布的病毒式丑闻进行了热烈讨论，强调了该问题如何引发了关于国内环境和政治影响的辩论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sandner.art/cosine-continuous-stable-diffusion-xl-cosxl-on-stableswarmui/">Cosine-Continuous Stable Diffusion XL (CosXL) on StableSwarmUI</a>：如何在 StableSwarmUI 上运行 Cosine-Continuous Stable Diffusion XL：来自 Stability.ai 的 CosXL Stable Diffusion 模型的设置教程、测试和质量预览</li><li><a href="https://x.com/arankomatsuzaki/status/1798899233246101701">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：BitsFusion: Diffusion 模型的 1.99 bits 权重量化。将 Stable Diffusion v1.5 的 UNet（1.72 GB, FP16）压缩至 1.99 bits（219 MB），实现了 7.9 倍的压缩率，甚至更好的……</li><li><a href="https://drive.google.com/file/d/1IBgfLqReWwhhWNXvnSCJH1gtQscgWPTV/view?usp=sharing">stable difusion web ui in sanoma three archives .zip</a>：未找到描述</li><li><a href="https://tenor.com/view/rhino-shit-rbxzoo-adurite-pooing-gif-26514280">Rhino Shit GIF - Rhino Shit Rbxzoo - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openart.ai/workflows/congdc/material-transfer-for-room/5NUUyIbVeqF6dQJIM4ft">AI Art Generator: Free Image Generator from OpenArt</a>：免费 AI 图像生成器。免费 AI 艺术生成器。免费 AI 视频生成器。100 多种模型和风格可供选择。训练您的个性化模型。最受欢迎的 AI 应用：草图转图像、图像转视频……</li><li><a href="https://tenor.com/view/uffa-gif-27431770">Uffa GIF - Uffa - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://imgur.com/a/4v0I4UO">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐 GIF、励志故事、病毒视频等来振奋精神……</li><li><a href="https://imgur.com/a/ABh3YmB">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐 GIF、励志故事、病毒视频等来振奋精神……
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1248358117498945670)** (132 条消息🔥🔥): 

- **训练续接的问题**：成员们讨论了使用 `model.push_to_hub_merged("hf_path")` 保存的模型 Adapter 在续接训练时遇到的问题，指出重新加载 Adapter 后性能会下降，一位用户报告其 Loss 从 0.4 增加到了 2.0。他们正在寻求继续训练时加载 Adapter 的正确方法。
  
- **持续预训练资源（Llama3 与韩语）**：[Unsloth AI 的博客文章](https://unsloth.ai/blog/contpretraining) 讨论了针对新语言对 Llama3 等 LLM 进行持续预训练（Continued Pretraining），强调了使用专门技术可以减少 VRAM 占用并加快训练速度。配套的 [Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) 可帮助用户应用这些方法。
  
- **Qwen2 模型发布**：社区宣布支持 Qwen2 模型，并指出其进步之处，如 128K 上下文长度和对 27 种语言的支持。[Daniel Han 的推文](https://x.com/danielhanchen/status/1798792569507418231) 分享了用于微调 Qwen2 7B 的 Colab 资源。

- **探索 Grokking**：成员们讨论了 LLM 性能中一个名为 "Grokking" 的新阶段。参考资料包括一段 [YouTube 视频](https://www.youtube.com/watch?v=QgOeWbW0jeA) 和几篇相关的论文（如[这一篇](https://arxiv.org/pdf/2201.02177)），提供了更深入的见解。

- **使用 NVIDIA NVLink 扩展 VRAM 的问题**：用户澄清了 NVLink 并不会将两块 GPU 的 VRAM 合并为一个共享池。每块 GPU 仍将显示其原始 VRAM 容量，这与关于扩展 VRAM 的常见误解相反。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行 LLM 持续预训练</a>：通过 Unsloth 对 Llama 3, Phi-3 和 Mistral 进行持续预训练，让模型学习新语言。</li><li><a href="https://datta0.github.io/blogs/know-your-lora/">了解你的 LoRA</a>：重新思考 LoRA 初始化。什么是 LoRA？LoRA 一直是微调领域（尤其是参数高效微调）的强大工具。它是一种非常简单的方法来微调你的模型...</li><li><a href="https://www.youtube.com/watch?v=QgOeWbW0jeA">新发现：LLM 具有性能阶段</a>：Grokking 是 LLM 性能的一个新阶段。从算术运算开始，我们分析了 Transformer 嵌入空间中的模式。Grokk...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | 模型卡片与 Prompt 格式</a>：Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含一条 system 消息，可以包含多条交替的 user 和 assistant 消息，并且始终以最后一条 user 消息结尾...</li><li><a href="https://x.com/danielhanchen/status/1798792569507418231">Daniel Han (@danielhanchen) 的推文</a>：已将 Qwen2 0.5+1.5+7 & 72b 的 4bit BnB 量化版本上传至 http://huggingface.co/unsloth。@UnslothAI 的 Qwen2 QLoRA 微调速度快 2 倍，节省 70% 的 VRAM，且上下文长度比 FA2 长 4 倍！72b 可适配...</li><li><a href="https://github.com/AlexBuz/llama-zip">GitHub - AlexBuz/llama-zip: 基于 LLM 的无损压缩工具</a>：基于 LLM 的无损压缩工具。通过在 GitHub 上创建账号来为 AlexBuz/llama-zip 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/577">在 LLaMA-3 上使用 rsLoRA 时出现奇怪的 grad_norm 峰值 · Issue #577 · unslothai/unsloth</a>：在使用 Unsloth 和 rsLoRA 训练 LLaMA-3 模型时，我总是看到非预期的 grad_norm 峰值：{'loss': 1.9848, 'grad_norm': 4.210731506347656, 'learning_rate': 1e-05, 'e...</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://openreview.net/forum?id=OZbn8ULouY">弹弓效应：自适应优化中的后期优化异常...</a>：自适应梯度方法，特别是 Adam ~\citep{kingma2014adam, loshchilov2017decoupled}，已成为优化神经网络（尤其是结合 Transformer）不可或缺的方法...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1248363421876097025)** (33 条消息🔥): 

- **Daniel Han YouTube 名人**：几位成员讨论了在 YouTube 上看到 **Daniel Han** 的经历，一位成员提到：“才 26 岁！太聪明了！！！” 另一位成员幽默地评论道：“他没那么聪明... <:heehee:1238495823734640712>”。

- **训练代码的艰辛**：一位成员分享了他的进展，说：“在写了 50 亿行代码后，我终于到了可以开始训练的地步了”，另一位成员鼓励道：“有进展就是好事！<:bale:1238496073228750912>”。随后的讨论涉及了量化代码的复杂性。

- **DeepSeek Coder 推荐**：一位成员询问了用于编程的小模型，并分享了 [DeepSeek Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) 的链接，强调了其从 1B 到 33B 不同版本的性能。

- **模型合并问题**：讨论围绕模型合并的问题展开，特别提到了将 4bit 和 LoRA 权重合并到 16bit 时遇到的问题。一位成员指出，“model.push_to_hub_merged 现在似乎只推送 adapter”，对该问题表示困惑，而其他人则尝试进行排查。

**提到的链接**：<a href="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct">deepseek-ai/deepseek-coder-6.7b-instruct · Hugging Face</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1248371549996712077)** (61 条消息🔥🔥): 

```html
<ul>
  <li><strong>Unsloth Llama3 缺少默认 LoRA adaptors</strong>：与某些假设相反，Unsloth 的 Llama3 模型不带默认的 LoRA adaptors。成员需要使用 <code>get_peft_model</code> 来进行设置。</li>
  <li><strong>Unsloth 即将支持 Ollama</strong>：即将发布的 Unsloth 版本将增加对 Ollama 的支持，引发了社区的热烈响应（一位用户分享道：“太棒了 ❤️”）。</li>
  <li><strong>出现 "GIL must be held" 错误消息</strong>：一位用户遇到了一个令人困惑的错误消息：*"GIL must be held before you call parseIValuesToPyArgsKwargs"*。排查建议是检查 Python 版本。</li>
  <li><strong>SFTTrainer vs UnslothTrainer 之争</strong>：用户询问在 <code>trl.SFTTrainer</code> 和 <code>unsloth.UnslothTrainer</code> 之间该使用哪一个。回复是两者都运行良好，选择取决于个人偏好。</li>
  <li><strong>禁用 Wandb 的说明</strong>：对于想要禁用 Wandb 追踪的用户，在训练参数中将环境变量 <code>"WANDB_DISABLED"</code> 设置为 *"true"* 并配合 <code>report_to = "none"</code> 即可实现。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rasbt/dora-from-scratch/blob/main/Using-LinearDoRAMerged.ipynb">dora-from-scratch/Using-LinearDoRAMerged.ipynb at main · rasbt/dora-from-scratch</a>：从零开始实现 LoRA 和 DoRA。通过在 GitHub 上创建账号为 rasbt/dora-from-scratch 的开发做出贡献。</li><li><a href="https://github.com/huggingface/trl/issues/1073">DPOTrainer Problem: trl/trainer/utils.py:456 · Issue #1073 · huggingface/trl</a>：问题发生在 trl/trl/trainer/utils.py 的第 456 行 else: # adapted from https://stackoverflow.com/questions/73256206 if &quot;prompt&quot; in k: to_pad = [torch.LongTensor(ex[k][::-1]) for ...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1248444127784337479)** (6 条消息): 

```html
- **友好的邀请**：“你最好邀请他们来这里！”以及“我们更友好”的保证，展示了社区的热情本质。
- **社区赞誉**：一位新成员表达了满意之情：“啊哈，我刚加入 Discord 服务器，这里非常棒”。另一位补充道“感谢分享！”，反映了群组内充满感激和积极的互动。
- **成员认可**：通过声明“没有人能打败 <@1179680593613684819> 或 <@160322114274983936>”来突出关键成员，认可他们对社区的宝贵贡献。
```
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1248547832961433621)** (10 messages🔥): 

- **Apple 对与 Nvidia 的合作仍不感兴趣**：成员们对为什么 **Apple** 尽管有机会却不与 **Nvidia** 合作表示好奇。一位成员质疑道：“如果过去没有合作，为什么现在要合作？”

- **1-bit LLM 工作组**：一位参与者提到在最近的一次优化研讨会上听说了 **1-bit LLM 工作组**。另一位提供了相关讨论频道 `#1240586843292958790` 的链接以获取更多细节。

- **事件调度时区已澄清**：关于预定事件的时区出现了疑问。一位成员澄清说，提到的下午 3 点的活动是 **PST 中午**。

- **用于粒子物理博士研究的 Alpaka**：一位用户分享了一个有趣的工具 [Alpaka](https://github.com/alpaka-group/alpaka)，并指出它在粒子物理研究的 CMSSW 中有所应用。他们提到：“不过我认为它在工业界并不流行。”

- **训练期间检查 CUDA core 利用率**：发布了关于在训练期间监控 **CUDA cores 利用率** 的查询。一位用户询问是否可以追踪其 5000 个核心中当前正在使用的数量。

**提到的链接**：<a href="https://github.com/alpaka-group/alpaka">GitHub - alpaka-group/alpaka: Abstraction Library for Parallel Kernel Acceleration :llama:</a>：并行 Kernel 加速抽象库 :llama: - alpaka-group/alpaka

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1248403200235667457)** (4 messages): 

- **使用 grid 语法简化 CUDA kernel 启动**：一位成员展示了一种使用 grid 语法启动 CUDA kernel 的更简单方法：`out = kernel[grid](...)`。这种方法为 CUDA 开发者简化了流程。
- **轻松访问 PTX 代码**：启动 kernel 后，可以使用 `out.asm["ptx"]` 轻松访问 PTX 代码。这种方法提供了对中间表示的快速访问，以便进一步检查。
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1248506694049206324)** (5 messages): 

- **TorchScript 中的 Tensor 类型转换困扰**：一位成员表达了在不改变底层数据或不使用 `view(dtype)` 的情况下，将 dtype 为 float32 的 Tensor 转换为 int32 的挫败感，因为这在 **TorchScript** 中“不受支持 :((”。他们提到需要对 bfloat16 进行位操作。

- **Philox unpack 困惑**：一位成员质疑在 PyTorch 中处理随机数时，为什么需要从 kernel 中调用 `at::cuda::philox::unpack`。他们想知道它是否应该在 host 端返回相同的结果。

- **寻找 Profiler 信息**：一位成员询问在某篇[专注于加速生成式 AI 模型的 PyTorch 博客文章](https://pytorch.org/blog/accelerating-generative-ai-2/)中使用了哪种 Profiler。另一位成员分享了一个 [PyTorch Profiler 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)作为回应，指出该教程解释了如何测量模型算子的时间和内存消耗。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-generative-ai-2/?">Accelerating Generative AI with PyTorch II: GPT, Fast</a>：这篇文章是一个系列博客的第二部分，专注于如何使用纯原生 PyTorch 加速生成式 AI 模型。我们很高兴能分享广泛的新发布的 PyTorch 性能特性...</li><li><a href="https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html">PyTorch Profiler &mdash; PyTorch Tutorials 2.3.0+cu121 documentation</a>：暂无描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1248597631492427857)** (1 messages): 

- **重新思考 LoRA 初始化**：一位成员分享了他们在 **LoRA** 中以不同方式初始化 A 和 B 矩阵的实验，并声称“我们可以比默认初始化做得更好”。他们鼓励其他人阅读其 [博客](https://datta0.github.io/blogs/know-your-lora/) 上的发现。

**提到的链接**：<a href="https://datta0.github.io/blogs/know-your-lora/">Know your LoRA</a>：重新思考 LoRA 初始化。什么是 LoRA？LoRA 在微调领域，特别是参数高效微调（PEFT）中一直是一个巨大的工具。这是一种非常简单的方法来微调你的模型...

  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1248643348260978731)** (2 条消息): 

- **Note 库通过 TensorFlow 革新机器学习**：一位成员分享了 [Note 库的 GitHub 仓库](https://github.com/NoteDance/Note)，该库利用 TensorFlow 简化了并行和分布式训练。该库包含 Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer 和 Segformer 等模型。

- **PyTorch 的 int4 解码创新**：另一位成员分享了一篇关于使用 Grouped-Query Attention 和低精度 KV cache 进行高效解码的 [博客文章](https://pytorch.org/blog/int4-decoding/)。文章讨论了在 Meta 的 Llama 和 OpenAI 的 ChatGPT 等 LLM 中支持长上下文长度的挑战。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/int4-decoding/">INT4 Decoding GQA CUDA Optimizations for LLM Inference</a>：一种使用低精度 KV cache 的高效解码 Grouped-Query Attention 技术  </li><li><a href="https://github.com/NoteDance/Note">GitHub - NoteDance/Note: Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer, etc, these models built with Note are compatible with TensorFlow and can be trained with TensorFlow.</a>：轻松实现并行训练和分布式训练。机器学习库。Note.neuralnetwork.tf 包包含 Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer 等，这些使用 Note 构建的模型与 TensorFlow 兼容，并可以使用 TensorFlow 进行训练。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1248457221386403861)** (2 条消息): 

- **Torch.compile 加速 SetFit 模型**：一位成员报告称，他们在 SetFit 模型上使用了 **torch.compile** 并获得了一定的加速。他们还提到，“你可以使用不同的参数调用 torch.compile！”
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1248487933510418473)** (20 条消息🔥): 

- **字节跳动的 2-bit 量化引发频道热议**：成员们讨论了 [字节跳动的 LLM 2-bit 量化算法 DecoupleQ](https://github.com/bytedance/decoupleQ)，指出该算法报告了困惑度（perplexity）但没有速度基准测试。该算法包含自定义 CUDA kernel，并需要一个针对 7B 模型耗时 3-4 小时的校准过程。
  
- **量化方法对比引发辩论**：一位用户寻求相对于 FP16 的基准数据，促使另一位用户澄清 DecoupleQ 侧重于困惑度而非速度基准，并建议它在 I/O 密集型操作中可能会更快。

- **量化技术对比**：详细讨论强调了带校准的量化与量化感知微调（quantization-aware fine-tuning）之间的区别。辩论强调了校准在内存和速度上的限制，以及量化感知微调的 VRAM 问题，而 LoRA 权重提供了一种高效的解决方法。

- **分享基于 Hessian 的量化见解**：成员指出 DecoupleQ 融合了 GPTQ 和 HQQ 的元素，在校准过程中使用了交替极小化（alternate minimization）和 Hessian。他们提到了基于 Hessian 的方法及其具体挑战，包括校准时巨大的内存和计算需求。

- **分享量化研究论文**：一位社区成员提供了一篇 [NeurIPS 2022 论文](https://arxiv.org/pdf/2206.06501) 的链接，该论文提出了一种改进方法来替代直通估计器（Straight-Through Estimator, STE），后者在训练期间经常导致梯度估计不佳和损失爆炸。

**提及的链接**: <a href="https://github.com/bytedance/decoupleQ">GitHub - bytedance/decoupleQ: A quantization algorithm for LLM</a>：一种用于 LLM 的量化算法。可以通过在 GitHub 上创建账户来为 bytedance/decoupleQ 的开发做出贡献。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1248394606262095943)** (184 条消息🔥🔥): 

- **Aleksagordic 思考 Triton 和 AMD 支持**：Aleksagordic 考虑了支持 Triton 以提高易用性和教育价值的好处，但承认这需要从 CUDA 和 HIP 的具体细节中抽象出来。Arund42 建议支持多种 GPU 类型，但警告说每个 kernel 都必须适配 Triton，这可能会导致分支（fork）。

- **BF16 的梯度范数（Gradient norms）和稳定性担忧**：Eriks.0595 强调了使全局梯度范数具有确定性（deterministic）的必要性，并对 BF16 代码在长时训练运行中的稳定性表示担忧。在考虑 FP8 集成之前，测试并解决当前的 BF16 稳定性被列为优先级。

- **支持 Llama 3 的计划及新功能集成**：Akakak1337 及其团队讨论了 RoPE、RMSNorm 以及可能集成的 YaRN 等功能。此外，他们还有意为即将推出的模型支持建立高层级路线图（roadmap），并改进测试框架，例如启用 multi-node 支持和各种 FP8 计算。

- **Checkpoints 和确定性挑战**：确保训练过程 100% 的确定性（determinism）包括解决 norm kernels 中的 atomics 问题，以及协调 master weights 的保存和恢复。Akakak1337 建议维护一个整洁的日志系统，以高效管理 checkpoint 数据。

- **探索 FineWeb 作为数据集**：Akakak1337 和 Arund42 讨论了使用 FineWeb 的优缺点，特别是它集中于英文文本，缺乏代码/数学等其他类型的数据，这可能会影响模型训练和评估结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.graphcore.ai/posts/simple-fp16-and-fp8-training-with-unit-scaling">Simple FP16 and FP8 training with unit scaling</a>：Unit Scaling 是一种新型低精度机器学习方法，能够在不进行损失缩放（loss scaling）的情况下以 FP16 和 FP8 训练语言模型。</li><li><a href="https://drive.google.com/file/d/1RdOmeGXgnQAsOreW9S7MU7H9A2rOGmFe/view?usp=sharing">last_ckpt_124M_400B.bin</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/issues/400">Broader vendor support for hardware acceleration · Issue #400 · karpathy/llm.c</a>：我理解 llm.c 项目的目标是保持极简和教育性，但其极简的方法使其成为各种平台上有吸引力的、可移植且高性能的目标。为了……</li><li><a href="https://github.com/karpathy/llm.c/pull/522/">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>：我们目前没有将 master weights 作为状态的一部分保存 -> 这会导致精度损失，因为否则当我们恢复训练时，必须通过从低精度上采样来重建 master weights……</li><li><a href="https://github.com/karpathy/llm.c/pull/432/files#diff-9b9e22c7c7c957363d4756f521df586b3bb857cd4f73b167256f2cfe0df717c2R106-R117)">only save missing bits to reconstruct fp32 master weights by ngc92 · Pull Request #432 · karpathy/llm.c</a>：我想我成功搞定了位操作（bit-fiddling），这仅需额外 16 位（而不是目前的 32 位）成本就能有效地为我们提供 fp31 master 参数。在合并之前，代码……</li><li><a href="https://arxiv.org/abs/2401.02954">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</a>：开源大语言模型（LLM）的快速发展确实令人瞩目。然而，以往文献中描述的扩展定律（scaling law）呈现出不同的结论，这给……蒙上了阴影。</li><li><a href="https://arxiv.org/abs/2309.00071">YaRN: Efficient Context Window Extension of Large Language Models</a>：旋转位置嵌入（RoPE）已被证明能有效编码 Transformer 语言模型中的位置信息。然而，这些模型无法泛化到超过其训练序列长度的范围……
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1248359567679815741)** (153 messages🔥🔥): 

- **5 篇研究论文因剽窃被撤稿**："5 篇研究论文因剽窃被撤稿，因为作者忘记删除他们使用的 Prompt。" 一位成员觉得这既幽默又不幸。
- **Haiku 模型因成本和质量受到赞赏**：几位成员讨论了 **Haiku AI 模型** 的**性价比**和质量。一位成员指出它的质量“类似于 GPT-3.5”，但价格非常实惠。
- **用于审核的 AI 引发辩论**：成员们辩论了在 Reddit 和 Discord 等平台上使用 AI 进行**审核 (Moderation)** 的问题。对于 AI 直接采取行动还是仅标记内容供人工审核，大家看法不一。
- **学习 LLM 的 YouTube 和在线资源**：推荐了一些学习 LLM 的 YouTube 视频和频道，包括 Kyle Hill 的 *ChatGPT Explained Completely* 以及 3blue1brown 深入讲解数学原理的视频。
- **OpenAI 语音能力的 LinkedIn 演示**：分享并赞赏了一个展示 OpenAI 新语音能力的 **LinkedIn 帖子**。你可以在[这里](https://www.linkedin.com/posts/alexhalliday_openai-showcased-their-new-voice-capabilities-activity-7204841736882286592-q9De?utm_source=share&utm_medium=member_ios)观看演示。
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1248359658922573865)** (33 messages🔥): 

- **实时聊天和新功能即将推出**：根据 [OpenAI 最近的更新](https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19)，GPT-4o 正在向所有用户推出，新的语音和视觉功能将在几周内面向 ChatGPT Plus 用户发布。实时聊天将进入 Alpha 阶段，供少数 Plus 用户使用，并在未来几个月内扩展到所有人。
- **频繁的 Custom GPT 更新引发挫败感**：用户对 Custom GPT 的频繁更新表示沮丧，因为这会导致收到“开始新聊天以获取最新指令”的通知。一位成员澄清说，当 GPT 的指令或能力发生任何变化时，就会发生这些更新，而不是底层模型本身发生了变化。
- **更新 Custom GPT 时出错**：用户报告了更新其 Custom GPT 时的问题，收到了“保存草稿出错”的消息。这表明平台上的 Custom GPT 功能可能存在潜在问题。
- **GPT 处理 CSV 文件的问题**：一位成员在让 GPT 从附加的 CSV 文件中选择单词时遇到困难，尽管这些文件仅包含简单的文本列表。他们最终不得不将列表直接复制到 GPT 的指令中，并寻求更好的解决方案。
- **订阅支持问题**：一位用户在续订订阅后仍无法看到 GPT-4 访问权限。建议通过 [OpenAI 帮助页面](https://help.openai.com)和实时聊天联系支持人员以解决此类问题。

**提到的链接**：<a href="https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19">来自 OpenAI (@OpenAI) 的推文</a>：所有用户今天将开始获得 GPT-4o 的访问权限。在接下来的几周内，我们将开始向 ChatGPT Plus 推出今天演示的新语音和视觉功能。

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1248506008222044170)** (6 messages): 

- **对 DALL-E 速度的挫败感**：一位用户对 DALL-E 的性能表示沮丧，指出它在快速尝试多个示例后会停止，然后需要等待一段时间。他们报告说由于字母生成错误，生成了大约 30 张图像都未成功。
- **GPT-4 处理技术问题的方法**：另一位用户指出，GPT-4 倾向于先提供一般性答案，然后详细分解步骤，他们认为这种方法旨在提高准确性。另一位成员表示同意，指出这种方法类似于“System 2 thinking”（系统 2 思维），尽管偶尔会误判何时切换方法，但仍然是有益的。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1248506008222044170)** (6 messages): 

- **DALL-E 遇到速率限制问题**：一位用户尝试使用 DALL-E 快速生成多张图像，但在被提示等待 2 分钟后停止。尽管尝试了大约 30 次操作，结果仍然不正确。
- **GPT-4 的迭代方法可能会让用户感到沮丧**：一位成员指出，GPT-4 在回答技术问题时倾向于从宏观开始并分解为详细步骤，这可能令人烦恼，但可能会提高准确性。另一位成员表示同意，将其比作 System 2 thinking，并提到即使 GPT-4 有时会误判使用时机，看到这种方法也是件好事。
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1248357019891663029)** (29 messages🔥): 

- **LM Studio 庆祝成立一周年**：一位用户提到 **LM Studio** 已经推出一年了，并使用表情符号庆祝这一里程碑。
- **混合 GPU 处理重负载**：一名成员询问是否可以将工作负载分配到多个 GPU 上，包括 **Tesla K80** 和 **3090**。其他成员建议，由于 **driver compatibility issues**（驱动兼容性问题），坚持使用 **3090** 或增加第二块 3090 将是最佳选择。
- **运行多个 LM Studio 实例**：一名成员询问是否可以在同一台机器上同时运行两个 LM Studio 实例以进行通信。回复澄清说，可以在一个实例的 Multi Model Session 中运行多个模型，并建议使用 **Autogen** 等框架或自定义 Python 脚本来实现模型交互。
- **在 PS5 和 VPS 上使用 LM Studio**：有人对在 **PlayStation 5 APU 上运行 LM Studio** 感到好奇，并询问是否可以在 VPS 上使用。评论指出 **high hardware requirements**（高硬件要求）使其主要适用于性能强大的 PC。
- **在 LM Studio 中从图像生成 embeddings**：一名成员询问关于使用 LM Studio 从图像生成 embeddings 的问题，并引用了 **daanelson/imagebind**。他们表示如果有兴趣，希望通过 LM Studio 在本地运行它。

**Sources**:
- "This is the way" [Mandalorian GIF](https://tenor.com/view/this-is-the-way-this-is-the-way-mandalorian-mandalorian-i-have-spoken-baby-yoda-gif-24159898)

**Link mentioned**: <a href="https://tenor.com/view/this-is-the-way-this-is-the-way-mandalorian-mandalorian-i-have-spoken-baby-yoda-gif-24159898">This Is The Way This Is The Way Mandalorian GIF - This Is The Way This Is The Way Mandalorian Mandalorian - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1248370514645483602)** (43 messages🔥): 

- **Qwen2 故障排除与修复**：一位用户遇到了 Qwen2 输出乱码的问题并询问预设情况，随后发现启用 flash attention 有时可以解决问题。另一位用户确认 ChatML 预设适用于 Qwen2。
- **通过 API 使用 Vision models 尝试失败**：一位用户报告了使用 llava-phi-3-mini-f16.gguf 的混合结果，尽管进行了各种调整，但在通过 API 使用时输出不一致。他们寻求关于 Phi 3 和其他 vision models 的建议，但未得到明确答复。
- **LM Studio 硬件问答**：一位新手询问为了在 LM 应用中获得最佳性能，应该升级 CPU 还是 GPU。回复明确指出，**GPUs** 对 **LLMs** 的性能提升远超 CPUs。
- **GPU 和 RAM 需求讨论**：有人提到运行大型模型（如 4_K_M 模型）需要大量的 RAM（超过 36GB）。将 context length 从 8K 减少到 4K 可以让模型运行顺畅。
- **Qwen2 模型的问题**：用户讨论了 Qwen2 模型的问题，特别是 cuda offloading 方面。这是一个已知问题，修复方案包括等待 llama.cpp 的更新以及随后的 GGUF 更新。

**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1d8kcc6/psa_multi_gpu_tensor_parallel_require_at_least/">Reddit - Dive into anything</a>: 未找到描述

  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1248446096603353139)** (11 messages🔥): 

- **为风格而非知识进行 Fine-tuning**：一位成员澄清说“fine-tuning 更多是关于风格而非通用知识”，这取决于所使用的 fine-tuning 方法类型，例如用于增加知识的 SFT 和用于风格的 LoRA。
- **Golden Gate Claude 与 GPU 误读**：关于 LLM 将 4060 误读为 GTX 460 有一段奇特的讨论，引发了关于“qualia”（感质）和意图性的问题，并将其与“golden gate claude”现象联系起来。
- **自定义 LM Studio 但不支持训练**：一位成员指出，在 LM Studio 中，你可以自定义 system prompt 来设置名称，但“无法进行训练或数据上传”。只有模型的原始 safetensors 文件可以进行 finetuned，GGUF 文件则不行。
- **克服“懒惰”的模型行为**：针对模型表现“懒惰”的问题，提出了多个建议，包括使用更强大的模型、尝试不同的模型，或调整预设模板以更好地适配 llama 3 模型。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1248388613801119805)** (19 条消息🔥): 

- **基准测试分数困惑**：讨论围绕“分数”的定义展开，提到“极有可能”是指 Cinebench 或其他基准测试软件。一位用户将 3900x 的分数与 Geekbench 6 数据进行了匹配，但发现与 5950x 的分数存在差异。
- **Tesla P40 散热挑战**：一名成员的 **Tesla P40 GPU** 到货了，但他们在安装散热方案时遇到了困难，不得不求助于纸板漏斗等临时方法。他们寻求关于反向气流运行的建议，并了解到旧款 Mac GPU 的风扇（特别是 27 英寸 2011 款 iMac GPU 风扇）可能是一个潜在的解决方案。
- **“马盖先式” GPU 散热改装**：由于 PC 机箱空间有限，一位用户讨论了昂贵散热方案的不切实际性，并考虑使用纸板风道。他们幽默地承认了这些虽然不寻常但很有必要的改装。
- **电源供应担忧与解决方案**：讨论了将 Tesla P40 GPU 连接到电源的挑战，因为该设置需要多个连接器。分享了一个有用的 [GitHub 指南](https://github.com/JingShing/How-to-use-tesla-p40)，帮助澄清了如何有效分配电源线。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/%D1%80%D0%B5%D0%B7%D0%B0%D1%82%D1%8C%D0%B6%D0%B5%D0%BB%D0%B5%D0%B7%D0%BE-%D1%80%D0%B0%D1%81%D0%BF%D0%B8%D0%BB%D0%B8%D1%82%D1%8C%D0%B6%D0%B5%D0%BB%D0%B5%D0%B7%D0%BE-cut-iron-sharp-spark-gif-15258290">резатьжелезо распилитьжелезо GIF - Резатьжелезо Распилитьжелезо Cut Iron - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/JingShing/How-to-use-tesla-p40">GitHub - JingShing/How-to-use-tesla-p40: A manual for helping using tesla p40 gpu</a>: 帮助使用 Tesla P40 GPU 的手册。可以通过在 GitHub 上创建账号来为 JingShing/How-to-use-tesla-p40 的开发做出贡献。</li><li><a href="https://rentry.org/Mikubox-Triple-P40">Mikubox Triple-P40 build</a>: 从 ebay 购买的包含散热器的 Dell T7910 “准系统”。我推荐 “digitalmind2000” 卖家，因为他们使用现场发泡包装，确保工作站完好无损地送达。你可以选择 Xe...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1248355433865941114)** (1 条消息): 

- **Higgs LLAMA 模型以 70B 参数量令人印象深刻**：新的 **Higgs LLAMA** 模型因其 700 亿（70B）参数规模所展现出的复杂性而备受关注。成员们正热切期待 **LMStudio 更新**，并指出该模型似乎使用了 *llamacpp 调整*。
  

---


### **LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1248481446348328990)** (2 条消息): 

- **ChatPromptTemplate 无法识别变量的问题**：一名成员询问 `ChatPromptTemplate.from_messages` 和 `ChatPromptTemplate.from_template` 是否支持在 `system_template` 或 `human_template` 中使用 `{tools}` 和 `{tools_names}`。他们注意到在调试 Prompt 时，尽管使用了 `create_react_agent`，但系统无法识别这些变量并返回了空白响应。
- **在 LangChain 中使用 LM Studio 代替 GPT**：另一名成员分享了他们在 LangChain 中尝试使用 LM Studio 的经验，提到他们有三个相互连接的 Chain。他们不确定在 LangChain 自身处理相同任务时，是否有必要在 `messages=[]` 中指定角色和 Prompt，或者使用系统和用户消息的前缀（Prefix）与后缀（Suffix）。
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1248482454667268106)** (11 条消息🔥): 

- **关于在 LM Studio 中为 6800m 开启 ROCm 的查询**：一名成员询问是否有人在 LM Studio 中成功为 6800m 启用了 ROCm。更多回复建议尝试 Arch Linux，并提到了在 Windows 上运行 ROCm 的各种问题。

- **7900xtx 在 Windows 上运行良好**：同一名成员分享说他们的 7900xtx 在 Windows 上运行没有问题，但他们的 6800m 笔记本电脑却不行。

- **Windows 上 ROCm 的变通方法**：另一名成员建议了一些变通方法，例如设置 `HSA_OVERRIDE_GFX_VERSION=10.3.0` 以及为 6800XT 系列以下的显卡使用额外的 ROCm 库。他们提到自己目前还没必要使用这些变通方法。

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1248395378554834964)** (9 条消息🔥): 

- **AI Engineer 日程的展厅说明**：一位成员询问了持有 Expo Explorer 门票参加 [AI Engineer 活动](https://www.ai.engineer/worldsfair/2024/schedule) 的分论坛参与情况。他们确认虽然展会分论坛是开放的，但更多演讲者的主题仍待确定。
- **快手的 KLING 对标 OpenAI 的 Sora**：[X](https://x.com/angrytomtweets/status/1798777783952527818?s=46&t=90xQ8sGy63D2OtiaoGJuww) 上的一篇帖子讨论了快手新推出的类 Sora 模型 KLING，并展示了 10 个令人印象深刻的案例。另一位成员注意到一段中国男子吃面条的演示视频具有极高的真实感。
- **GPT-4o Vision 的见解**：一位成员分享了 [Oran Looney](https://www.oranlooney.com/post/gpt-cnn/) 的一篇深度文章，探讨了 GPT-4o 处理高分辨率图像时的 Token 计费机制。该文章质疑了为什么 OpenAI 使用特定的 Token 数量 (170)，并探讨了编程中的“[魔数 (magic numbers)](https://en.wikipedia.org/wiki/Magic_number_(programming))”概念。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/angrytomtweets/status/1798777783952527818?s=46&t=90xQ8sGy63D2OtiaoGJuww">Angry Tom (@AngryTomtweets) 的推文</a>：OpenAI 的 Sora 很疯狂。但快手刚刚发布了一个名为 KLING 的类 Sora 模型，人们都为之疯狂。这里有 10 个不容错过的狂野案例：1. 一个中国男子坐在...</li><li><a href="https://www.oranlooney.com/post/gpt-cnn/">一张图片价值 170 个 Token：GPT-4o 如何编码图像？ - OranLooney.com</a>：事实是：GPT-4o 对高分辨率模式下使用的每个 512x512 切片收取 170 个 Token。按照约 0.75 Token/词计算，这意味着一张图片大约相当于 227 个词——仅为...的四倍。</li><li><a href="https://x.com/angrytomtweets/status/1798777783952527818?s=46&t=90xQ8">Angry Tom (@AngryTomtweets) 的推文</a>：OpenAI 的 Sora 很疯狂。但快手刚刚发布了一个名为 KLING 的类 Sora 模型，人们都为之疯狂。这里有 10 个不容错过的狂野案例：1. 一个中国男子坐在...</li><li><a href="https://www.ai.engineer/worldsfair/2024/schedule">AI Engineer 世界博览会</a>：加入 2,000 名由 AI 赋能并构建 AI 的软件工程师。2024 年 6 月 25 日至 27 日，旧金山。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1248727976544178370)** (98 条消息🔥🔥): 

- **幻觉工程师角色扮演**：成员们幽默地讨论了他们的头衔，其中一人自称是“高级幻觉工程师 (senior hallucinating engineer)”，另一人回应道：“不，Staff 是我用来召唤今日软件的工具。”
- **Websim.ai 探索**：成员们分享了使用 websim.ai 的经验，这是一个实时流式面部识别网站。一位用户指出，“在 websim.ai 的 websim.ai 版本中进入 websim.ai 可以让你不断递归……在第 4 层深度时，我的页面变得无响应了。”
- **GPT 的有用幻觉**：小组讨论了“有用幻觉范式 (useful-hallucination paradigm)”，即 GPT 在监控其响应时会产生有用的指标幻觉。这一概念受到了积极评价，评论包括“来自 quicksilver 和 stunspot 的超级提示词 (superprompts)”。
- **分享的资源和链接**：分享了几个链接，包括一个包含精选资源的 [Google 表格](https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl9u06IrKWoy3tlrCWlMRXDxvoDCY)，以及一个关于 Websim 系统提示词的 [GitHub Gist](https://gist.github.com/SawyerHood/5d82679953ced7142df42eb7810e8a7a)。
- **管理与未来会议**：成员们讨论了未来的管理和即将举行的会议计划。建议包括“演示设置 websim 概念的演练”，并确认了下周主持人职责的角色分配。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/SawyerHood/5d82679953ced7142df42eb7810e8a7a">websim.txt</a>: GitHub Gist: 立即分享代码、笔记和摘要。</li><li><a href="https://worldsim.nousresearch.com/">worldsim</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题, 日期, 协调人, 资源, @dropdown, @ GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://docs.google.com/spreadsheets/d/1F-9MgOsDlevbiokl9u06IrKWoy3tlrCWlMRXDxvoDCY/edit#gid=2061123208">Latent Space Friday AI In Action: Websim</a>: 资源 名称, 链接, 备注 Websim, &lt;a href=&quot;https://websim.ai/&quot;&gt;https://websim.ai/&lt;/a&gt; Podcast Ep, &lt;a href=&quot;https://www.latent.space/p/sim-ai&quot;&gt;https://www.latent.sp...</li><li><a href="https://t.co/evC8wiHkYz">Cyberpunk Chat Room</a>: 未找到描述</li><li><a href="https://websim.ai/c/2PLjreKO66U6TOhES">WebSim.AI - Self-Referential Simulated Web</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1248488718822543411)** (2 条消息): 

- **Mixtral 8x7B MoE 详解**：一位成员了解到 Mixtral 8x7B 并非拥有 8 个独立的专家，而是 32x8 个专家。他们分享了一个名为“Stanford CS25: V4 I Demystifying Mixtral of Experts”的 [YouTube 视频](https://youtu.be/RcJ1YXHLv5o?si=gjcu--95HafBwEaT)，来自 Mistral AI 和剑桥大学的 Albert Jiang 在视频中详细讨论了这一点。

- **可灵快手链接**：一位成员分享了 [可灵快手 (Kling Kuaishou)](https://kling.kuaishou.com/) 的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/RcJ1YXHLv5o?si=gjcu--95HafBwEaT">Stanford CS25: V4 I Demystifying Mixtral of Experts</a>: 2024年4月25日 演讲者：Albert Jiang, Mistral AI / 剑桥大学。揭秘 Mixtral of Experts。在本次演讲中，我将介绍 Mixtral 8x7B，一个稀疏的...</li><li><a href="https://kling.kuaishou.com/">可灵大模型</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1248368216804163655)** (60 messages🔥🔥): 

- **Meta AI 发布 Vision-Language Modeling 指南**：Meta AI 发布了一份名为 ["An Introduction to Vision-Language Modeling"](https://arxiv.org/abs/2405.17247) 的新指南，涵盖了 VLM 的工作原理、训练过程和评估方法。该指南旨在帮助对该领域感兴趣的人理解将视觉映射到语言背后的机制。

- **关于 Qwen-2 推理能力的辩论**：成员们讨论了对 Qwen-2 7B 的“实际体验评估 (vibe check)”，批评意见提到它与 Yi 1.5 34B 等更大的模型相比表现不佳。尽管它在大上下文下具有连贯性，但人们对其有效性持怀疑态度。

- **对新基准测试的兴奋**：57B-A14 MoE 被发现在许多编程相关的基准测试中击败了 Yi 1.5 34B。该模型采用 Apache 2.0 许可，对 VRAM 有较高要求。

- **GPU 匮乏影响模型测试**：许多成员表达了由于缺乏 GPU（“GPU poor”）而面临的挑战，这限制了他们测试通过 Together API 和 Fireworks 提供的模型的能力。

- **Outpainting 工具推荐**：对于图像扩展，成员们建议使用 Krita 插件（[GitHub 链接](https://github.com/Acly/krita-ai-diffusion)）和 Interstice Cloud。他们建议逐步增加图像分辨率以达到所需的宽高比。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.17247">An Introduction to Vision-Language Modeling</a>：随着近期大语言模型 (LLM) 的普及，人们尝试将其扩展到视觉领域。从拥有一个可以引导我们穿越陌生环境的视觉助手到...</li><li><a href="https://www.interstice.cloud/service">Interstice</a>：未找到描述</li><li><a href="https://www.abc4.com/news/wasatch-front/utah-h-mart-store-opening/">犹他州首家 H Mart 门店将于 6 月 7 日开业</a>：韩国美食爱好者无需等待太久，即可在新店买到所需的所有食材。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1248615969237241876)** (1 messages): 

- **DeepSeek Coder 6.7B 在编程任务中表现出色**：一位成员询问了代码模型，并分享了 [DeepSeek Coder 6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) 的详细介绍。该模型在 2 万亿 tokens 上进行训练（主要为代码），在从 1B 到 33B 的各种模型尺寸中，展示了项目级代码补全和填充 (infilling) 的 SOTA 性能。

**提到的链接**：<a href="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct">deepseek-ai/deepseek-coder-6.7b-instruct · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1248394175821512806)** (12 messages🔥): 

- **Mistral 保持简洁**：“很高兴看到 Mistral 理解保持连接性提示词 (connective prompts) 短小精悍的重要性。” 强调了连接性提示词的简洁性对于与 Mistral 进行有效沟通的重要性。

- **格式中过多的连字符**：一位成员对“过度使用 `--------------------`”表示担忧，并指出用户很难手动匹配“准确数量的 `-`”。

- **非通用模板框架**：提到他们格式中基于连字符的分隔符是专门为 RAG 生成数据集而设计的。他们澄清说，这种格式不属于 Markdown 或 XML 等标准化框架。

- **UltraChat 格式的差异**：一位成员在探索 Mistral 文档时，询问了 UltraChat 格式与他们使用的格式有何不同。

- **用于 RAG 数据集的 Prophetissa**：分享了 [Prophetissa](https://github.com/EveryOneIsGross/Prophetissa)，这是一个使用 Ollama 和 emo vector search 的 RAG 数据集生成器，并提到了潜在的改进方向，以及探索像 LangChain 这样有趣的替代方案。

**提到的链接**：<a href="https://github.com/EveryOneIsGross/Prophetissa">GitHub - EveryOneIsGross/Prophetissa: 使用 ollama 和 emo vector search 的 RAG 数据集生成器。</a>：使用 ollama 和 emo vector search 的 RAG 数据集生成器。 - EveryOneIsGross/Prophetissa

  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1248352039138754590)** (28 messages🔥): 

- **WorldSim 控制台更新使移动端可用**：最近对 **WorldSim 控制台**的更新修复了移动设备上的大量文本输入 bug。它还改进了复制/粘贴功能，更新了 `!list` 命令，并包含了一个禁用视觉效果的新选项。
- **文本重复及其他故障已修复**：解决了诸如文本重复和文本跳动等特定问题。“您的文本重复故障应该已修复”以及“输入时文本向上跳行的问题应该已修复”。
- **标记和保存聊天**：可以通过使用 `!save` 命令为聊天命名。未来可能会添加更多功能，如聊天大小/长度标签。
- **自定义命令和服务的趣味玩法**：成员们分享了有趣的自定义命令，如 `systemctl status machine_sentience.service`。还强调了创建有趣的虚构“服务”安装的能力。
- **WorldSim 使用 Claude 模型**：用户可以在 WorldSim 设置中从各种 **Claude 模型**中进行选择。然而，尝试使用自定义模型的效果被认为不如前者。

**提到的链接**：<a href="https://tenor.com/view/terminator2-ill-be-back-arnold-schwarzenegger-i-will-be-back-brb-gif-27347908">Terminator2 Ill Be Back GIF - Terminator2 Ill Be Back Arnold Schwarzenegger - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1248402567566725152)** (3 messages): 

- **RAG 的未来：拥抱 Agentic 知识检索**：@seldo 在旧金山总部的演讲探讨了 RAG 的未来，即从朴素的 top-k 检索转向完全的 Agentic 知识检索。[视频指南](https://t.co/fCK8L9O2sx)包含真实世界的代码示例，以帮助提升您的实践。

- **LlamaIndex 推出两个内存模块**：LlamaIndex 现在为 Agent 提供了两个内存模块，增强了 RAG 框架。**Vector Memory Module** 将用户消息存储在向量存储中，并使用向量搜索来检索相关消息；请参阅 [演示 notebook](https://t.co/Z1n8YC4grM)。

- **Create-llama 与 e2b_dev 的 Sandbox 集成**：Create-llama 现在与 e2b_dev 的 Sandbox 集成，允许执行 Python 代码进行数据分析，并返回诸如图表图像之类的完整文件。正如[此处](https://t.co/PRcuwJeVxf)所宣布的，此功能为 Agent 应用引入了许多新的可能性。
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1248377064776798322)** (95 messages🔥🔥): 

```html
<ul>
  <li><strong>简化 RAG 的动态数据更新</strong>：尽管查询引擎无法立即反映 VectorStoreIndex 中的更改，但一种解决方案是定期重新加载索引以确保其使用最新数据（如代码片段所示）。这确保了 RAG 应用能够动态地回答包含新数据的查询。</li>
  <li><strong>索引管理建议</strong>：在讨论管理不同数据集（如销售数据、人工成本、技术支持文档）的最佳方法时，建议使用独立的索引或应用元数据过滤器，让 LLM 根据从查询中推断出的主题决定查询哪个索引。</li>
  <li><strong>利用知识图谱增强 Embedding</strong>：用户讨论了如何直接使用 LlamaIndex 创建带有 Embedding 的属性图，以及将实体及其同义词的文本 Embedding 直接附加到知识图谱中实体节点的好处。</li>
  <li><strong>调整分块大小 (chunk sizes)</strong>：为了针对大文本优化 LlamaIndex，用户可以调整 `Settings` 类中的 `chunk_size` 参数，从而根据使用场景实现更好的分块管理和更精确的 Embedding。</li>
  <li><strong>图中的实体解析</strong>：执行实体解析可能涉及定义自定义检索器以定位和合并节点，利用手动删除和 upsert 等方法，如提供的 `delete` 方法示例所示。</li>
</ul>
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=document+managem">Document Management - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>).">Basic Strategies - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/prompts/prompt_optimization/#setup-vector-index-over-this-data>)">"Optimization by Prompting" for RAG - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/docs/docs/module_guides/indexing/lpg_index_guide.md?plain=1#L430">llama_index/docs/docs/module_guides/indexing/lpg_index_guide.md at main · run-llama/llama_index</a>: LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph/base.py#L191,">llama_index/llama-index-core/llama_index/core/indices/property_graph/base.py at main · run-llama/llama_index</a>: LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1248461480937000991)** (8 messages🔥): 

- **Async 功能仍在开发中**：针对有关线程的查询，有人指出虽然 **Mojo** 支持 `async fn` 和 `async def`，但尚未支持 `async for` and `async with`。还提到了 `@parallelize`，表明该领域正在持续开发中。
- **对 Windows 版本的强烈需求**：成员们表达了对 max/mojo 的 **Windows 原生版本** 的渴望，提出了多次请求，并对延迟进行了一些幽默的调侃。该版本的发布时间表仍不明确，目前是用户关注的热点话题。
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1799109375258484909>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1248728366169980980)** (1 messages): 

- **Mojo 社区会议 #2 发布**：**Modular** 宣布发布了名为 "Mojo Community Meeting #2" 的新 [YouTube 视频](https://www.youtube.com/watch?v=3FKSlhZNdL0)。视频内容包括 **Benny Notson** 演示的 **Basalt ML Framework**、**Maxim Zaks** 演示的 **Compact Dict** 以及 **Samay Kapad** 演示的 **Pandas for Mojo**。

**提及的链接**：<a href="https://www.youtube.com/watch?v=3FKSlhZNdL0">Mojo Community Meeting #2</a>：Mojo 社区会议 #2 演讲录像：🌋 Basalt ML Framework (Benny Notson) 📔 Compact Dict (Maxim Zaks) 🐼 Pandas for Mojo (Samay Kapad)...

  

---

### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1248664986096636075)** (1 messages): 

- **MAX 24.4 带来 Quantization API 和 macOS 兼容性**：Modular 宣布发布 [MAX 24.4](https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos)，引入了全新的 **Quantization API** 并将 MAX 的支持范围扩展到 macOS。该 API 可将 Generative AI 流水线的延迟和内存成本在**桌面端降低高达 8 倍**，在**云端 CPU 上降低高达 7 倍**，且无需重写模型或更新应用程序。

**提到的链接**：<a href="https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos">Modular: MAX 24.4 - Introducing Quantization APIs and MAX on macOS</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：MAX 24.4 - Introducing Quantization APIs and MAX on macOS

  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1248670420727496784)** (1 messages): 

- **Mojo 和 MAX 24.4 版本发布，支持 macOS**：Mojo 和 MAX 的最新版本 24.4 现已包含 quantization 和 macOS 支持。查看 [MAX 安装指南](https://modul.ar/install-max)、[发布博客](https://modul.ar/24-4)以及[使用 MAX 和 quantization 运行 llama3 的示例](https://modul.ar/llama3)。
- **完整更新日志已上线**：本次发布的完整更新日志可在 [Mojo 更新日志](https://modul.ar/mojo-changelog)和 [MAX 更新日志](https://modul.ar/max-changelog)中找到。 
- **庆祝社区贡献**：本次发布包含了社区对标准库的 200 多项贡献。
- **Python 安装修复**：当未设置 `MOJO_PYTHON_LIBRARY` 时，Mojo 现在会链接到 `PATH` 中的第一个 Python 版本。提供了一个[脚本](https://modul.ar/fix-python)用于在您的系统中查找兼容的 Python 环境。
- **安装遇到问题？** 在 [Discord 频道](https://discord.com/channels/1087530497313357884/1248684060134342696/1248684060134342696)中发布任何关于 MAX on macOS 或运行 llama3 quantization 示例的问题。
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1248508029478109266)** (7 messages): 

- **Andrew Ng 警告 AI 监管可能会扼杀创新**：[Andrew Ng 的推文](https://x.com/AndrewYNg/status/1798753608974139779)引起了人们对加州拟议法律 SB-1047 的关注，表达了对该法律可能扼杀 AI 创新的担忧。他认为，安全性应该根据应用场景而非技术本身来评估，并使用了类比：*“搅拌机是否安全，不能通过检查其电动机来确定。”*

- **关于监管可行性的讨论**：成员们讨论了监管基于 open-source 技术构建的 AI 应用的可行性。一位成员指出：*“如果技术可以从互联网下载，那么监管基于其构建的应用还有机会吗？我会说机会肯定不大。”*

- **与俄罗斯 AI 监管的对比**：一位成员评论了俄罗斯缺乏 open-source AI 监管讨论的情况，并引用了一段 [弗拉基米尔·普京与 AI 生成的自己对质](https://www.youtube.com/watch?v=5L2YAIk0vSc&ab_channel=TheTelegraph) 的视频。

- **Deepfake 和数据集解决方案**：强调了针对 “deepfakes” 和不当内容生成等主要 AI 问题的潜在解决方案。其中一个建议包括清除数据集中有害内容，*“好吧，这将是一个漫长的斗争故事。”*

- **AI 滥用的全球视角**：对话扩展到全球范围内 AI 滥用的必然性，并开玩笑说：*“即使美国禁止，瓦坎达（Wakanda）仍然会制作灭霸（Thanos）跳电臀舞的 deepfakes。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=5L2YAIk0vSc&ab_channel=TheTelegraph">震惊的普京与 AI 生成的自己对质</a>：弗拉基米尔·普京在俄罗斯电视台播出的马拉松式电话新闻发布会上否认他有替身。普京告诉观众，“只有一个……”</li><li><a href="https://x.com/AndrewYNg/status/1798753608974139779">来自 Andrew Ng (@AndrewYNg) 的推文</a>：保护创新和开源的努力仍在继续。我相信，如果任何人都能进行基础 AI 研究并分享他们的创新，我们的处境都会更好。现在，我深切关注……
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1248373946412437618)** (53 messages🔥): 

- **Mojo `isdigit()` 效率与问题**：讨论揭示了 `isdigit()` 函数内部为了效率使用了 `ord()` 和别名，并展示了 [源代码](https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/builtin/string.mojo#L236)。一位成员表示在使用该函数时遇到困难，最终提交了 [Issue 报告](https://github.com/modularml/mojo/issues/2975)。
- **Mojo 中 `async` 编程的现状**：成员们讨论了异步编程的状态，强调 `async` 仍处于开发中，`Coroutine` 的更新已出现在 nightly 构建版本中。
- **Mojo `type` 检查的替代方案**：一位新用户询问如何检查变量类型，得到的建议是使用 `__type_of` 函数，并了解了 VSCode 扩展和 REPL 如何帮助在编译前后识别类型。
- **Mojo 24.4 版本的变化与 Bug**：多位成员讨论了升级到 24.4 版本后的问题，包括 `islower` 函数导致的段错误 (seg faults)，以及 `Tensor` 缺失 `data()` 方法的问题（根据 [更新后的文档](https://github.com/modularml/mojo/blob/nightly/docs/changelog-released.md)，现在改为 `unsafe_ptr()`）。
- **Mojo 在各平台上的可用性**：一位成员询问关于 Google Colab 上的 Mojo，回复指出虽然目前不在 Colab 上，但用户目前可以使用 [Mojo Playground](https://docs.modular.com/mojo/playground)。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/string/">string | Modular Docs</a>: 实现用于处理字符串的基础对象方法。</li><li><a href="https://docs.modular.com/mojo/playground">Modular Docs</a>: 无描述</li><li><a href="https://tenor.com/view/homer-smart-simpsons-flames-gif-12447941">Homer Smart GIF - Homer Smart Simpsons - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/modularml/mojo/issues/2975>">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/manual/traits#implicit-trait-conformance">Traits | Modular Docs</a>: 为类型定义共享行为。</li><li><a href="https://docs.modular.com/mojo/roadmap#limited-polymorphism">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/builtin/string.mojo#L236>">mojo/stdlib/src/builtin/string.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - 第 36 期
https://www.modular.com/newsletters/modverse-weekly-36
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1248357696445612062)** (22 messages🔥): 

- **最新 nightly 版本增加动态 libpython 选择**：最新的 nightly 现在实现了动态 libpython 选择，不再需要设置 `MOJO_PYTHON_LIBRARY`。然而，尽管有说法称只要正确激活虚拟环境就不再需要，一些用户仍然需要使用 `Python.add_to_path`。

- **虚拟环境的 VS Code 集成问题**：用户报告称，通过 VS Code 运行 Mojo 文件无法保留虚拟环境的激活状态，需要使用 `source .venv/bin/activate` 进行手动干预。尽管集成终端显示虚拟环境已激活，但此问题依然存在。

- **新版 nightly Mojo 编译器发布**：新的 nightly Mojo 编译器版本 `2024.6.714` 已发布，变更日志和原始差异可在 [此处](https://github.com/modularml/mojo/compare/19a5981220b869d2a72fff6546e9104b519edf88...ceaf063df575f3707029d48751b99886131c61ba) 查看，[当前变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 见此。

- **微基准测试 (Microbenchmarks) 现已可用**：最新的 `nightly` 版本包含了 [Mojo GitHub 仓库](https://github.com/modularml/mojo/tree/nightly/stdlib/benchmarks) 中的微基准测试。鼓励用户尝试、添加或修改基准测试，并提供反馈以改进 benchmark 包。

**Link mentioned**: <a href="https://github.com/modula">modula - Overview</a>: GitHub 是 modula 构建软件的地方。

  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1248683953192177674)** (1 messages): 

- **TopK Sparse Autoencoder 训练代码的 Beta 版本发布**：基于 [OpenAI 最近的论文](https://x.com/norabelrose/status/1798985066565259538)，**TopK Sparse Autoencoder** 训练代码的 Beta 版本已发布。它为每个网络层训练一个 SAE，且无需在磁盘上缓存激活值（activations），未来的更新将包括多 GPU 支持和 AuxK 辅助损失。
    
- **该库的独特功能**：与其他库不同，该库可以同时为所有层训练 SAE，并已在 **GPT-2 Small** 和 **Pythia 160M** 上进行了测试。有关更多详细信息和贡献方式，请用户前往特定频道和 [GitHub 仓库](https://github.com/EleutherAI/sae)。

**提到的链接**：<a href="https://x.com/norabelrose/status/1798985066565259538">来自 Nora Belrose (@norabelrose) 的推文</a>：这是我们针对 TopK Sparse Autoencoders 的训练库，该模型由 OpenAI 在今天早上提出。我已经在 GPT-2 Small 和 Pythia 160M 上进行了测试。与其他库不同，它为...训练 SAE。

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1248351567279554624)** (45 messages🔥): 

- **关于 Data Shapley 的新见解**：分享了一篇 [arXiv 预印本](https://arxiv.org/abs/2405.03875)，讨论了 Data Shapley 在不同设置下进行数据选择时的一致性问题，并提出了一个假设检验框架来预测其有效性。讨论强调，高质量和低质量数据的混合仍然可以产生有用的见解。

- **澄清 LM Evaluation Harness 的用法**：一位用户询问如何使用 **lm harness** 进行评估，另一位用户指向了 [lm-evaluation-harness GitHub 仓库](https://github.com/EleutherAI/lm-evaluation-harness)。对于它是用于训练还是评估存在一些困惑，通过解释其在评估模型中的用途以及关于准确率计算细节的说明进行了澄清。

- **TorchScript 中张量类型转换的困难**：一位成员寻求关于在 TorchScript 中将张量从 `float32` 转换为 `int32` 而不改变底层数据的建议。提供了几项建议，包括使用 C++ 扩展以及考虑浮点到整数的量化（quantization）技术。

- **OpenLLM-Europe 社区介绍**：Jean-Pierre Lorré 介绍了 **OpenLLM-Europe**，旨在聚集欧洲开源 GenAI 的利益相关者，重点关注多模态模型及其评估与对齐（alignment）。

- **扩图 (Outpainting) 工具推荐**：对于将 4:3 图像转换为桌面壁纸，推荐使用 [stable-diffusion-ps-pea](https://github.com/huchenlei/stable-diffusion-ps-pea) 作为更现代且活跃的工具。对于没有 GPU 的用户，建议使用 SD Turbo 或基于 CPU 的模型等替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.03875">Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits</a>：Data Shapley 为数据估值提供了一种原则性方法，并在以数据为中心的机器学习 (ML) 研究中发挥着至关重要。数据选择被认为是 Data Shapl 的标准应用...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1248393598148542464)** (40 条消息🔥): 

- **Thousand Brains Project 旨在实现新型 AI 设计**：Jeff Hawkins 和 Numenta 正在启动一项[多年期项目](https://www.numenta.com/thousand-brains-project/)，旨在将 Thousand Brains Theory 实现为软件，并将新皮层（neocortex）原理应用于 AI。该项目承诺进行广泛的协作和开放研究实践，引起了社区的热烈反响。

- **讨论 GPT 自回归（Autoregressive）顺序调制**：一篇关于自回归模型的新论文通过添加允许动态 Token 采样的位置编码，[挑战了固定顺序](https://arxiv.org/abs/2404.09562)的假设。成员们将其与离散扩散模型（discrete diffusion models）进行了比较，但承认在概念严谨性和训练收益方面存在差异。

- **ReST-MCTS 提升 LLM 训练质量**：[ReST-MCTS](https://arxiv.org/abs/2406.03816) 方法利用过程奖励指导（process reward guidance）和蒙特卡洛树搜索（Monte Carlo Tree Search）来收集高质量的推理轨迹，以实现更好的 LLM 训练。该方法通过从 Oracle 正确答案中推断过程奖励，避免了手动标注的需求。

- **关于开放式 AI 危险性的辩论**：来自 Google DeepMind 的一篇[立场论文](https://arxiv.org/abs/2406.04268)认为，开放式 AI 系统是实现人工超智能（ASI）的可行路径。然而，一些成员批评这种方法可能具有危险性，因为它缺乏终止条件。

- **用于可解释特征的稀疏自编码器（Sparse autoencoders）**：另一篇论文介绍了 [k-sparse autoencoders](https://arxiv.org/abs/2406.04093)，以更好地平衡重构和稀疏性目标。该方法简化了调优并改善了重构-稀疏性边界，在从语言模型中提取可解释特征方面展现了潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.09562">σ-GPTs: A New Approach to Autoregressive Models</a>: 自回归模型（如 GPT 系列）使用固定的顺序（通常是从左到右）来生成序列。然而，这并非必须。在本文中，我们挑战了这一假设并展示了...</li><li><a href="https://www.numenta.com/thousand-brains-project/">Thousand Brains Project | Numenta</a>: Thousand Brains Project 是一个开源倡议，致力于根据 Thousand Brains Theory 创建一种新型的人工智能。</li><li><a href="https://arxiv.org/abs/2406.03816">ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</a>: 最近的 LLM 自我训练方法主要依赖于 LLM 生成响应，并过滤那些具有正确输出答案的响应作为训练数据。这种方法通常会导致低质量的微调...</li><li><a href="https://arxiv.org/abs/2406.04268">Open-Endedness is Essential for Artificial Superhuman Intelligence</a>: 近年来，AI 系统的通用能力出现了巨大飞跃，这主要得益于在互联网规模数据上训练的基础模型。尽管如此，创造开放式...</li><li><a href="https://arxiv.org/abs/2406.04093">Scaling and evaluating sparse autoencoders</a>: 稀疏自编码器提供了一种极具前景的无监督方法，通过从稀疏瓶颈层重构激活值，从语言模型中提取可解释的特征。由于语言模型...</li><li><a href="https://arxiv.org/abs/2110.02037">Autoregressive Diffusion Models</a>: 我们引入了自回归扩散模型（ARDMs），这是一类包含并推广了顺序无关自回归模型（Uria et al., 2014）和吸收离散扩散（Austin et al...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1248535164917645355)** (4 messages): 

- **关于 Sparse Autoencoders 的新论文**：一篇[关于 sparse autoencoders 的论文](https://arxiv.org/abs/2406.04093)讨论了使用 k-sparse autoencoders 来更有效地控制稀疏度，并改善重构与稀疏度之间的平衡。该论文还引入了用于评估特征质量的新指标，并展示了清晰的 Scaling Laws。
- **TopK Sparse Autoencoders 库**：[Norabelrose 分享了](https://x.com/norabelrose/status/1798985066565259538)一个针对 TopK sparse autoencoders 的训练库，已在 GPT-2 Small 和 Pythia 160M 上进行了测试。与其他库不同，它同时为所有层训练 SAE，且不需要在磁盘上缓存激活值。
- **TopK 激活值的复兴**：一名成员对 TopK 激活值重新引起关注表示兴奋，指出该领域之前的创新超前于其时代。这种重新燃起的兴趣被视为重新审视和实验这些方法的机会。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.04093">Scaling and evaluating sparse autoencoders</a>: Sparse autoencoders 提供了一种极具前景的无监督方法，通过从稀疏瓶颈层重构激活值，从语言模型中提取可解释的特征。由于语言模型...</li><li><a href="https://x.com/norabelrose/status/1798985066565259538">Nora Belrose (@norabelrose) 的推文</a>: 这是我们的 TopK sparse autoencoders 训练库，该方法由 OpenAI 今天早上提出。我已经在 GPT-2 Small 和 Pythia 160M 上进行了测试。与其他库不同，它为...训练 SAE
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1248628713399128179)** (4 messages): 

- **文件名处理问题的修复待处理**：一名成员承认了结果文件放置的问题，并保证已将其列入待办事项。他们提到**结果应该在 tmp 文件夹中**，文件可以在那里打开和读取。
- **PR 旨在解决文件名处理问题**：有人建议 Pull Request [Results filenames handling fix by KonradSzafer](https://github.com/EleutherAI/lm-evaluation-harness/pull/1926/files) 可能会解决之前提到的问题。该 PR 专注于通过重构和移动函数来解决 Bug，以便在整个代码库中更好地利用。

**提到的链接**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1926/files">Results filenames handling fix by KonradSzafer · Pull Request #1926 · EleutherAI/lm-evaluation-harness</a>: 该 PR 专注于解决：#1918 - 通过将处理结果文件名的函数移动到 utils，以便它们可以在代码库的其他部分使用；#1842 - 通过重构 Zeno 脚本以运行...

  

---



### **Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1248526518406353007)** (15 messages🔥): 

- **关于机器人文章深度的辩论**：讨论了机器人文章是否缺乏细节和见解，特别是“边走边变现”和业务动态没有充分阐述。一名成员建议，拆解具体细节和假设，特别是“40000 个高质量机器人年份数据”背后的计算，将增加更多深度。
- **定期回顾环节的建议**：有人建议 Nathan Lambert 考虑像 Peter Attia 那样做“季度回顾环节”，以反思收获并综合读者的反馈。这有助于在趋同严重的领域中区分内容。
- **将机器人内容建立在数据基础上的挑战**：另一名成员呼吁在讨论算法进展、数据生成成本和制造成本时，提供更多数字和明确的假设，特别是当文章批评 Covariant 的商业模式和 VC 预期时。推荐参考 Eric Jang 的 YouTube 视频。
- **关于 Physically Intelligent 机器人的反馈**：询问了 Physically Intelligent 拥有的机器人数量及类型。Nathan Lambert 提到是“aloha / 廉价设备到较好版本再到极好版本的混合”，但记不清所有品牌名称。
- **对机器人文章的不同反馈**：Nathan Lambert 承认收到了对该文章的褒贬不一的反馈，一些机器人领域的朋友喜欢它，而另一些则不然。尽管有批评，Lambert 仍然“看好 Covariant 作为一个企业的成功”。

**提到的链接**: <a href="https://archive.is/HAsy4">Robots are suddenly getting cleverer. What&#x2019;s changed?</a>: 未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1248405612048416799)** (16 messages🔥): 

- **中国 AI 视频生成器表现超越 Sora**：一款 [中国 AI 视频生成器](https://x.com/rowancheung/status/1798738564735554047) 发布，能够通过 KWAI iOS 应用生成 2 分钟、30fps、1080p 的视频，但需要中国手机号码。成员们讨论了其相对于 Sora 的优越性，并对质量和公开可用性发表了评论。
- **Vision-LSTM 将 xLSTM 与图像读取融合**：[Jo Brandstetter 分享了](https://x.com/jo_brandstetter/status/1798952614568116285?s=46) Vision-LSTM 的推出，它使 xLSTM 能够有效地读取图像。该帖子包含了 [arXiv 预印本](https://arxiv.org/abs/2406.04303) 的链接和 [项目详情](https://nx-ai.github.io/vision-lstm/)。
- **Kling AI 视频生成器接受测试**：[一位用户](https://x.com/op7418/status/1799047146089619589?s=46&t=_jodDCDeIUnWb_Td0294bw) 分享了他们对 KWAI 视频生成模型 Kling（可灵）的测试资格，并发布了一些生成的视频。成员们分析了这些视频，重点关注物理特性和光影/摄像机角度，并指出了质量的提升。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/op7418/status/1799047146089619589?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 歸藏(guizang.ai) (@op7418) 的推文</a>：我获得了 KWAI 视频生成模型 Kling 的测试资格。#KWAI #Kling #Sora #aivideo 以下是一些生成的视频 👇：</li><li><a href="https://x.com/jo_brandstetter/status/1798952614568116285?s=46">来自 Johannes Brandstetter (@jo_brandstetter) 的推文</a>：介绍 Vision-LSTM - 让 xLSTM 能够读取图像 🧠 它运行得……非常、非常好 🚀🚀 但请自行评判 :) 我们很高兴已经分享了代码！📜: https://arxiv.org/abs/2406.04303 🖥️: https...</li><li><a href="https://x.com/rowancheung/status/1798738564735554047">来自 Rowan Cheung (@rowancheung) 的推文</a>：在我们获得 Sora 访问权限之前，一款中国 AI 视频生成器刚刚发布。可以生成 30fps、1080p 质量的 2 分钟视频，可在 KWAI iOS 应用上使用，需绑定中国手机号。一些生成示例...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1248534701904236595)** (8 messages🔥): 

- **Anthropic 为 AI 对齐研究提供 API 访问权限**：一位成员分享了 Anthropic [支持文章](https://support.anthropic.com/en/articles/9125743-how-can-i-access-the-claude-api-for-alignment-research-purposes)的链接，详细说明了对齐（Alignment）研究人员如何访问 Claude API。感兴趣的个人需要提供其所属机构、职位、LinkedIn、GitHub 和 Google Scholar 个人资料以及简短的研究计划。

- **新的匿名模型 "anon-leopard" (Yolo AI) 发布**：分享的一条推文宣布在 LMSYS 上发布了一个名为 "anon-leopard" 的新匿名模型，由 "Yolo AI" 开发。据指出，该模型没有使用常见的 OpenAI Tokenizations。

- **对 Daylight 电脑的兴趣**：多位用户讨论了他们对 Daylight 电脑（[产品链接](https://daylightcomputer.com)）的兴趣，这是一款旨在通过避免蓝光发射并在直射阳光下可见，从而更健康、更人性化的电脑。一位已经拥有 iPad mini 的用户在讨论是否要预订它。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://daylightcomputer.com">Daylight | 一台更贴心的电脑</a>：Daylight Computer (DC1) 是一种新型的宁静电脑，专为深度工作和健康而设计。</li><li><a href="https://x.com/stevelizcano/status/1798713330414321805?s=46">来自 stephen 🌿 (@stevelizcano) 的推文</a>：LMSYS 上发布了新的匿名模型，名为 "anon-leopard"，自称为 "Yolo AI" 的 "Yolo"，但似乎没有使用任何常见的 OpenAI Tokenizations：</li><li><a href="https://support.anthropic.com/en/articles/9125743-how-can-i-access-the-claude-api-for-alignment-research-purposes">如何为了对齐研究目的访问 Claude API？| Anthropic 帮助中心</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1248532258982330378)** (1 messages): 

- **推文回复回避策略**：一位成员分享了他们处理某些推文的方法，称：*“我总是默认‘我不是这条推文的目标受众’，然后继续浏览。”* 他们认为这些推文通常看起来像溢出到时间线上的群聊消息。

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1248404013209354355)** (4 条消息): 

- **类比于自我调试模型 (self-debugging models)**：成员们将讨论的方法与自我调试模型进行比较，指出它以错误输出为条件来生成正确输出，并带有一个最大化改进难度的外层循环。*"他们在外部循环中尝试寻找一种直接生成策略（无自我改进步骤），该策略对于自我改进模型来说是最难进行优化的。"*
- **对解析解的兴趣**：一位成员表示需要更好地理解像 DPO 这样偏重理论的论文中的解析解。他们发现“在最小化改进的同时又将其最大化”这一点非常有趣。
- **对 DPO 风格理论论文的评论**：Nathan Lambert 指出，所讨论的论文大多是理论性的，实际评估极少，类似于原始的 DPO。他提到曾尝试为此创作内容，但由于其复杂性而放弃了。
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1248703645348790414)** (3 条消息): 

- **SnailBot 向用户发出关于秘密帖子的警报**：消息标记了角色 `<@&1216534966205284433>` 并宣布：*"Snailbot 找到了我关于技术内容的秘密帖子。"* 然而，这被证明是一个非正式的玩笑警报。
- **关于虚假帖子的澄清**：明确表示 *"不是真实的帖子"*，表明之前的消息不应被认真对待。
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1248372939494264974)** (38 条消息🔥): 

- **Aya 的翻译成本引发好奇**：一位成员询问了使用 **Aya** 进行翻译的成本。回复指出 Aya 对研究用途免费，但对商业用途不免费，因为业务需要维持运行。
  
- **Command-R-Plus 与 Llama3 的辩论**：讨论强调，一些用户发现 **Command-R-Plus** 比 **Llama3** 更聪明，尤其是在非特定语言方面。一位用户评论道：“不只是你一个人这么觉得”，强化了这一观点。

- **Vercel AI SDK 与 Cohere 集成问题**：一位用户报告了 **Vercel AI SDK** 对 Cohere 支持不完整的问题，并分享了指向 SDK 指南的[链接](https://sdk.vercel.ai/providers/legacy-providers/cohere)。另一位成员迅速采取行动，通过联系 SDK 的维护者来解决这些集成问题。

- **免费个人使用的隐私政策说明**：针对 Cohere 模型免费个人使用的**隐私政策**查询，通过解释可以在本地或在 AWS 和 Azure 等云服务上使用 Cohere 模型的选项进行了回答，并提供了相应的 [AWS](https://docs.cohere.com/docs/cohere-on-aws) 和 [Azure](https://docs.cohere.com/docs/cohere-on-microsoft-azure) 文档链接以获取更多详情。

- **开发者作品集分享**：一位正在寻找全栈开发职位的用户分享了他们的专业作品集，展示了 **UI/UX, Javascript, React, Next.js, 和 Python/Django** 等技能，可以在[这里](https://www.aozora-developer.com/)查看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.aozora-developer.com/">Welcome Journey</a>：未找到描述</li><li><a href="https://cohere.com/data-usage-policy">Data Usage Policy</a>：未找到描述</li><li><a href="https://sdk.vercel.ai/providers/legacy-providers/cohere">Legacy Providers: Cohere</a>：了解如何将 Cohere 与 Vercel AI SDK 结合使用。</li><li><a href="https://docs.cohere.com/docs/cohere-on-aws">Cohere on AWS - Cohere Docs</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/cohere-on-microsoft-azure">Cohere on Azure</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1248436380325904414)** (7 条消息): 

- **Rafael 为 GenAI 幻觉提供解决方案**：前 Apple AI/ML 工程师 Rafael 正在验证一款旨在通过实时检测和纠正幻觉来确保 GenAI 应用安全的产品。他正在寻找在产品或业务流程中面临类似 GenAI 问题的用户进行合作。

- **Complexity 的发布给社区留下深刻印象**：Hamed 宣布发布了 Complexity，这是一个基于 Cohere 构建的生成式搜索引擎，并立即收到了积极反馈。其他成员称赞了它的性能和设计，强调了尽管名字叫 Complexity（复杂），但其方法却简单且极简。

- **分享了 Complexity 链接**：Hamed 提供了 Complexity 搜索引擎的链接（[Complexity](https://cplx.ai/)），引发了社区的进一步测试和赞赏。

**提到的链接**：<a href="https://cplx.ai/">Complexity</a>：触手可及的世界知识

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1248718975190171660)** (1 messages): 

- **Qwen 2 72B Instruct 模型发布**：[Qwen 2 72B Instruct](https://openrouter.ai/models/qwen/qwen-2-72b-instruct) 模型现已上线。此次发布标志着 OpenRouter 在 2023-2024 年度的产品线中增加了一个重要成员。

**提及的链接**：<a href="https://openrouter.ai/models/qwen/qwen-2-72b-instruct)">Qwen 2 72B Instruct by qwen</a>：Qwen2 72B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它采用了 SwiGLU 激活函数、Attention QKV 偏置和 gro...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1248437902615248967)** (36 messages🔥): 

- **大范围 504 Gateway Timeout 错误**：许多用户在尝试使用服务器时遇到了 504 Gateway Timeout 错误，特别是使用 Llama 3 70B 模型时。Alex Atallah 确认并指出数据库压力与 504 错误同时发生，目前正将任务迁移到 read replica 以缓解此问题。

- **WizardLM-2 8X22 响应问题**：用户报告称，当通过 DeepInfra 路由时，WizardLM-2 8X22 会生成难以理解的响应。Alex Atallah 建议使用请求路由中的 `order` 字段排除有问题的供应商，进一步讨论显示，除 DeepInfra 外，其他供应商也可能存在问题。

- **关于路由控制和模型供应商的讨论**：Asperfdd 对在使用 Chub Venus 等服务时无法控制路由选项表示担忧，并寻求解决这些供应商问题的更新。讨论还暗示了为了故障排除而部署的内部端点。

- **关于 AI Security 的辩论**：针对 AI Security 的价值以及最近 OpenAI (OAI) 解雇 Leopold Aschenbrenner 一事展开了激烈讨论。观点差异显著，一些人对 AI Security 的担忧不以为然，而另一些人则批评 Aschenbrenner 在该问题上的立场。

- **高峰负载期间的 ChatGPT 性能**：用户 pilotgfx 推测 ChatGPT 在高峰负载时段表现较差，怀疑是由于用户量大而进行了某种形式的性能 Quantization。另一位用户表示同意，但概括认为“现在的 3.5 普遍非常愚蠢”。

**提及的链接**：<a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>：跨多个供应商进行请求路由

  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

voidnewbie: Qwen2 也支持韩语！
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1248399413760491643)** (10 messages🔥): 

- **Torch Tune 可用性询问**：一位用户询问：“我们有 Torch Tune 了吗？”在提取的消息中没有提供后续回复。

- **对日语专用模型的兴趣**：一位成员询问：“有专门针对日语优化的模型吗？”这条消息表明了对针对特定语言定制的模型的兴趣。

- **Qwen2 72b Finetuning 问题凸显**：一位成员指出了一个经过 Finetuning 的 Qwen2 72b 模型的配置问题，称其 `max_window_layers` 被设置为 28，与 7b 变体类似，这导致了明显的性能下降。他们建议由于这一“疏忽”，可能需要进行新一轮的 Finetuning。

- **关于模型性能的讨论**：对话中提到一位用户确认该模型“对我有效”，而另一位成员指出 GGUF 格式的性能特别差。有迹象表明，Commit 历史中已进行更新以修正该疏忽。

- **分享分布式 Finetuning 指南的 Pull Request**：一位成员分享了一个关于使用 Axolotl 和 Deepspeed 进行分布式 Finetuning 的[指南 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1688)。该 PR 是应要求创建的，表明协作项目正在持续增强。

**提及的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1688">Update multi-node.qmd by shahdivax · Pull Request #1688 · OpenAccess-AI-Collective/axolotl</a>：标题：使用 Axolotl 和 Deepspeed 进行多节点分布式 Finetuning；描述：此 PR 引入了使用 Axolotl 和 A... 设置分布式 Finetuning 环境的全面指南。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages): 

josharian: 我刚才也遇到了完全相同的情况。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1248604112539357276)** (8 条消息🔥): 

- **在 slurm 上构建 flashattention 存在问题**：一位成员对构建 flashattention 表示沮丧，称其“荒谬”。他们详细说明了解决方案，包括加载特定模块并确保 slurm 上有足够的 RAM。
- **需要集群特定的 CUDA 加载模块**：一位成员分享说，由于系统找不到 `cuda_home`，集群支持人员必须创建一个 CUDA 加载模块，从而解决了问题。
- **在 slurm 上安装 Nvidia Apex 的问题**：一位成员提到在 slurm 上安装 Nvidia Apex 时遇到困难，表明问题依然存在。
- **关于在 slurm 上安装 Megatron 的潜在指南**：一位用户宣布他们可能会创建一个在 slurm 上安装 Megatron 的指南，因为 Docker 容器在集群环境中无法通用。
- **Axolotl 安装问题**：一位成员报告称，在按照 [Axolotl 快速入门指南](https://github.com/xyz/axolotl) 使用 Python 3.10 和 Axolotl 0.4.0 安装带有 `flash-attn` 和 `deepspeed` 依赖项时，遇到了内存问题（使用了 64GB RAM）。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1248656568652660766)** (2 条消息): 

- **JSONL 测试集拆分问题已解决**：一位成员表示在以 `context_qa.load_v2` JSONL 格式添加 `test_datasets` 时遇到困难，收到错误“未发现数据集的测试拆分（no test split found for dataset）”。他们随后分享了一个可用的配置，使用的是按照 `axolotl.cli.preprocess` 文档指定的常规 JSONL 文件格式。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1248613006297006091)** (13 条消息🔥): 

- **Axolotl API 使用讨论**：一位成员询问如何使用 Axolotl API 进行推理的示例。他们特别希望通过编写脚本来执行连续的模型评估，通过提问多个问题并保存答案，而无需重复使用命令行。

- **关于 API 与 YAML 配置的澄清**：另一位成员将 API 使用请求与 YAML 配置混淆了。原提问者澄清了他们感兴趣的是基于 API 的推理，而不是基于 YAML 的配置。

- **Flash-attn RAM 需求**：一位成员询问安装 `flash-attn` 的 RAM 需求。解释称，虽然该库在安装时不需要太多 RAM，但在训练大型模型时需要大量资源，包括高显存（VRAM）和匹配的系统 RAM。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=427ff160-9bce-4211-9634-40334159abd3)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fccec5e3-585a-444f-9bcc-ebcb7e53f766)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1248388036643782717)** (29 条消息🔥): 

- **中国 AI 视频生成器在 Sora 之前发布**：一款可在快手（KWAI）iOS 应用上使用的中国 AI 视频生成器，可生成 30fps、1080p 分辨率的 2 分钟视频 ([source](https://x.com/rowancheung/status/1798738564735554047?t=7jS8zcHFXppvCqJk29g9kQ&s=19))。相关链接视频展示了该生成器的能力。
- **Stable Audio Open 多人模式更新**：Stable Audio Open 的一项更新引入了多人协作功能，旨在共同改进 Prompt 策略 ([source](https://x.com/multimodalart/status/1798846199098921306))。该更新旨在将个人探索过程转变为集体学习体验。
- **艺术软件中的 AI 安全讨论**：Adobe Photoshop 要求获得对任何创作作品（包括受保密协议 NDA 保护的作品）的完全访问权限，这引起了担忧 ([source](https://x.com/SamSantala/status/1798292952219091042))。一些成员建议使用 Krita、Affinity Designer/Photo 和 Gimp 等替代工具。
- **新款 AI 视频生成器 Kling 令人印象深刻**：一段名为 "New AI Video Generator is Sora-level" 的 YouTube 视频揭晓了 Kling，这是一款新型写实视频生成器 ([source](https://www.youtube.com/watch?v=BTfLq-XkO0w))。它的强大能力备受关注，但也有人对其真实性表示怀疑。
- **Schelling AI 的发布引发褒贬不一的反应**：@EMostaque 宣布成立 Schelling AI，旨在利用数字资产和 AI 算力挖矿实现 AI 民主化 ([source](https://x.com/EMostaque/status/1799044420282826856?t=VHHAmaKGFWPuHbWcd_zrJA&s=19))。其宏大的主张和大量流行术语的使用引发了社区的怀疑和幽默回应。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/multimodalart/status/1798846199098921306">来自 apolinario (multimodal.art) (@multimodalart) 的推文</a>: Stable Audio Open 多人模式上线了！🤼‍♂️ 模型非常强大 🔊 但对其进行 Prompt 编写非常新颖 ✏️ 感谢 @Ameerazam18 将我的多人模式 PR 合并到你的 Space - 将这种探索转化为...</li><li><a href="https://www.youtube.com/watch?v=BTfLq-XkO0w">新款 AI 视频生成器达到 Sora 级别</a>: 新型写实视频生成器 Kling 揭晓。#ainews #aivideo #openai #sora #ai #agi #singularity 感谢我们的赞助商 Brilliant。免费试用 Brilliant...</li><li><a href="https://x.com/rowancheung/status/1798738564735554047?t=7jS8zcHFXppvCqJk29g9kQ&s=19">来自 Rowan Cheung (@rowancheung) 的推文</a>: 一款中国 AI 视频生成器在我们获得 Sora 访问权限之前发布了。可以生成 30fps、1080p 质量的 2 分钟视频，通过中国手机号在快手（KWAI）iOS 应用上可用。一些生成效果...</li><li><a href="https://x.com/EMostaque/status/1799044420282826856?t=VHHAmaKGFWPuHbWcd_zrJA&s=19">来自 Emad (@EMostaque) 的推文</a>: 很高兴宣布 @SchellingAI 👋 我们将构建并支持由 AI 货币驱动且为 AI 货币提供动力的开源代码、模型和数据集 🚀 我们的重点是创新研究和审慎构建...</li><li><a href="https://tenor.com/view/okex-bitcoin-btc-bitmex-kraken-gif-17797219">Okex Bitcoin GIF - Okex Bitcoin Btc - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/SamSantala/status/1798292952219091042">来自 Sam Santala (@SamSantala) 的推文</a>: 我没读错吧？@Adobe @Photoshop 除非我同意你们完全访问我用它创作的任何内容，包括受保密协议 NDA 保护的作品，否则我就不能使用 Photoshop？</li><li><a href="https://kling.kuaishou.com/">可灵大模型</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1248714867561730109)** (5 条消息): 

- **σ-GPT 动态生成序列**：与 SkysoftATM 合作开发的新模型 **σ-GPT** 在推理时可以**以任何顺序动态生成序列**，挑战了 GPT 传统的从左到右的方法。根据其 [arXiv 论文](https://arxiv.org/abs/2404.09562)，这可能会将**生成所需的步骤减少一个数量级**。

- **对 σ-GPT 实用性的怀疑**：一位成员表示怀疑，指出为了达到高性能而需要课程学习（curriculum）**可能会在实践中阻碍这种方法**，且收益相对较小，并将其与 XLNET 不尽如人意的采用情况进行了比较（*“XLNET 基本上做了同样的事情作为预训练任务，但从未流行起来”*）。

- **替代方案和潜在应用**：一位成员建议，对于填充（infilling）任务，像 [GLMs](https://arxiv.org/abs/2103.10360) 这样的替代方案可能更实用。此外，为非文本序列建模（如在 RL 中）使用额外的一组 positional embeddings 来微调现有模型可能会很有趣。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.09562">σ-GPTs: A New Approach to Autoregressive Models</a>: 自回归模型（如 GPT 系列）使用固定的顺序（通常是从左到右）来生成序列。然而，这并非必要。在本文中，我们挑战了这一假设并展示了...</li><li><a href="https://x.com/ArnaudPannatier/status/1799055129829839166">来自 Arnaud Pannatier (@ArnaudPannatier) 的推文</a>: GPT 以从左到右的顺序生成序列。还有其他方法吗？与 @francoisfleuret 和 @evanncourdier 以及 @SkysoftATM 合作，我们开发了 σ-GPT，能够生成序列...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1248374913899827341)** (21 条消息🔥): 

- **LangChain 中的早停方法问题**：一位成员提到 LangChain 中的 `early_stopping_method="generate"` 选项在较新版本中不起作用，并链接了一个[公开的 GitHub issue](https://github.com/langchain-ai/langchain/issues/16263)。他们询问是否有修复计划或可用的变通方法。

- **使用 ChromaDB 的 RAG 中的数据隐私**：一位成员询问了在 LangChain 配合 ChromaDB 时如何管理特定用户的文档访问。另一位成员建议在 vectorstores 中[通过 metadata 进行过滤](https://github.com/langchain-ai/langchain/discussions/9645)，同时强调了其复杂性和最佳实践。

- **处理 Agent 中的重复步骤**：一位用户寻求帮助解决 Agent 因执行重复步骤而陷入死循环的问题。消息中未提供具体的解决方案。

- **LLaMA3-70B 的有效 Prompt 技巧**：一位用户征求 Prompt 建议，以使 LLaMA3-70B 在执行摘要等任务时，结果开头不会出现类似 “here is the thing you asked” 的短语。尝试了几个 Prompt，但未找到解决方案。

- **关于 Vector Store 使用的澄清**：另一位成员询问 Chroma 是否支持在相似性搜索期间返回 ID，并分享了使用 SupabaseVectorStore 和其他 vector stores 的细节。此消息快照中未记录对 Chroma 查询的直接回答。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:6333')">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/bagel_network/status/1799143240769081731">来自 Bagel 🥯 (@bagel_network) 的推文</a>: .@LangChainAI 弥合了语言模型与外部数据源之间的差距，能够轻松开发强大的应用程序。现在，凭借 Bagel 的微调能力和 LangChain 的框架...</li><li><a href="https://github.com/langchain-ai/langchain/discussions/9645">关于 Langchain 中基于 Metadata 过滤进行向量搜索的查询 · langchain-ai/langchain · Discussion #9645</a>: 我有大量的 PDF 文档，我想使用 Langchain 与之交互。我的具体要求是，我不希望在每次搜索时都对所有文档进行向量搜索...</li><li><a href="https://github.com/langchain-ai/langchain/issues/16263).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1248551426859335692)** (3 messages): 

- **Apple 的 Generative AI 原则**：分享了在 iPhone Pro、iPad Pro 和 MacBook Pro 等 Apple 设备上实现本地 Generative AI 的[指导原则](https://drive.google.com/file/d/1s0imJ0zidk5-hraT46y8u4jnUby_oukk/view)。这些原则旨在优化 AI 在设备硬件上的性能。

- **用于虚拟面试的 Streamlit 应用**：一位用户介绍了他们的第一个 Streamlit 应用 [Baby Interview AGI](https://baby-interview-agi.streamlit.app/)，这是一个使用 LangChain 和 OpenAI 构建的虚拟求职面试聊天应用。他们对在 Streamlit 的 community cloud 上发布感到非常兴奋。

- **B-Bot 应用的封闭 Alpha 测试**：一项公告详细说明了 **B-Bot 应用**开始进入封闭 Alpha 测试阶段，这是一个用于专家与用户之间知识交流的高级虚拟平台。他们正通过[邀请](https://discord.gg/V737s4vW)寻找 10-20 名测试人员，以完善应用并提供宝贵反馈。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/V737s4vW">加入 B-Bot Discord 服务器！</a>：查看 Discord 上的 B-Bot 社区 – 与其他 7 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://drive.google.com/file/d/1s0imJ0zidk5-hraT46y8u4jnUby_oukk/view">Results Apple On-Device GenAI MacOS, iOS New Utility for Lead Optimization Final 6-6-24.pdf</a>：未找到描述</li><li><a href="https://baby-interview-agi.streamlit.app/.">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1248454346484613161)** (10 messages🔥): 

- **导出自定义 phi-3 模型的问题**：一位用户询问了关于导出自定义 phi-3 模型的问题，并在尝试从 Hugging Face 下载模型文件时遇到了困难。他们通过一个 [GitHub 链接](https://github.com/pytorch/torchtune/blob/16b7c8e16ade9ab0dc362f1ee2dd7f0e04fc227c/torchtune/_cli/download.py#L90) 意识到某些配置可能导致了该问题。
- **关于模型文件 HF 格式的澄清**：另一位用户澄清说，从 Hugging Face 下载的模型是 HF 格式，但在 Torchtune 中使用 FullModelHFCheckpointer 时，它会在内部转换为 Torchtune 格式，并在保存 Checkpoint 时转回 HF 格式。
- **对 MQA/GQA 的 n_kv_heads 的兴趣**：一位用户询问是否有兴趣让 KV Cache 为 MQA/GQA 使用 n_kv_heads，以及是否欢迎提交 PR。他们得到的回复是欢迎提交 PR，但必须包含相应的单元测试。

**提及的链接**：<a href="https://github.com/pytorch/torchtune/blob/16b7c8e16ade9ab0dc362f1ee2dd7f0e04fc227c/torchtune/_cli/download.py#L90">pytorch/torchtune 中的 torchtune/_cli/download.py</a>：一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1248573771577688065)** (6 条消息): 

- **Torchtune 上的依赖版本管理**：成员们讨论了在 **Torchtune** 的安装说明中明确依赖版本的重要性。建议包括提醒用户确保已 *安装最新稳定版的 PyTorch*，并链接到相关 Issue，如 [Issue #1071](https://github.com/pytorch/torchtune/issues/1071)、[Issue #1038](https://github.com/pytorch/torchtune/issues/1038) 和 [Issue #1034](https://github.com/pytorch/torchtune/issues/1034)。
- **支持 PyTorch Nightlies**：会议指出，**Torchtune** 中的许多功能仅在 **PyTorch nightlies** 版本中受支持，应在某处明确说明，以便用户获得最完整的功能集。
- **即将发布的安装说明 PR**：一名成员提到，他们打算创建一个 PR 来更新安装说明，解决依赖版本管理以及 **PyTorch nightly** 构建版本的使用问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1071">运行简单的 `$ tune` 时缺失 torch._higher_order_ops 模块 · Issue #1071 · pytorch/torchtune</a>：我正尝试在 Llama3-7B 上进行多节点微调，刚刚安装了 torchtune。尝试运行 `$ tune ls` 命令时出现了 Traceback 错误...</li><li><a href="https://github.com/pytorch/torchtune/issues/1038">ModuleNotFoundError: No module named 'torch._higher_order_ops' · Issue #1038 · pytorch/torchtune</a>：系统信息 torch==2.0.0 torchtune==0.1.1 transformers==4.41.1 safetensors==0.4.3。复现步骤：从 torchtune.utils 导入 FullModelHFCheckpointer...</li><li><a href="https://github.com/pytorch/torchtune/issues/1034">codellama-2 微调失败 · Issue #1034 · pytorch/torchtune</a>：我正在使用 recipe 中提供的相同配置文件来微调 CodeLlama-2 模型，但遇到了一个奇怪的错误...
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1248396115548704820)** (5 条消息): 

```html
- **对 Stellar Blade 及工作室做法的兴趣**：一名成员询问另一名成员对 **Stellar Blade** 的看法，特别是该工作室反对西方 **SJW**（社会正义战士）的立场。另一名成员回复表示支持任何专注于制作好游戏而非“觉醒文化（wokeness）”的开发者。

- **中国开发者对 DEI 的态度**：一位成员指出，**中国开发者**通常不关心女性主义和 **DEI**（多元、平等与包容）。他们对这种态度表示赞同，强调应专注于游戏本身。

- **韩国开发者与女性主义**：讨论转向了开发 **Stellar Blade** 的韩国工作室 **Shift Up**。另一名成员评论了韩国在女性主义和低出生率方面的问题，称该工作室的做法“非常令人耳目一新”。
```
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1248464705815838720)** (8 条消息🔥): 

- **构建 AI 驱动的 "Among Us" 模组**：一名成员对另一个使用 AI Town 构建交互式 "Among Us" 模组的项目表示兴趣，称其“非常有趣”。他们询问了进度以及遇到的任何性能问题或限制。

- **简陋但有效**：另一名成员提到该项目仍处于初步阶段，指出：“它显然非常简陋，但 AI Town 对于这个项目来说真的很棒”。他们面临一些限制，但刻意缩小了开发范围。

- **切换到 Godot 以实现高级功能**：他们分享说，由于需要游戏编辑器提供的工具，他们已转向 [Godot](https://zaranova.xyz) 来实现新功能。他们强调，尽管最初面临限制，但该项目在使用 AI Town 时取得了基础性的成功。

- **AI Town 获得进一步开发**：他们承认另一名成员推动了 AI Town 的发展，表明该项目正在持续进步。
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1248412039320309831)** (11 条消息🔥): 

- **关于 Open Interpreter 桌面 UI 的好奇**：一位成员询问是否有适用于 Open Interpreter 的桌面 UI。该讨论串中未提供答案。

- **处理 Posthog 连接错误**：一位用户指出：*"有人知道如何忽略这些针对 (requests.exceptions.ConnectionError) 的 Backing off send_request(...) 吗？"* 这突显了连接到 `us-api.i.posthog.com` 时的问题。

- **询问 OpenAPI 规范和 Function Calling**：成员们讨论了 Open Interpreter 是否可以接收现有的 OpenAPI 规范进行 Function Calling。其中包括一个建议，即可能通过在配置中设置 true/false 值来实现。

- **在 LM Studio 中使用 Gorilla 2 LLM 和工具调用的困扰**：一位成员分享了 *"在 LM Studio 中使用工具调用（tool use）难以取得任何成功"* 的经历，特别提到了自定义 JSON 输出和 OpenAI toolcalling。有人鼓励查看 GitHub 上的一个 OI Streamlit 仓库以寻求潜在解决方案。
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 条消息): 

ashthescholar.: 是的，请查看 OI 的网站。
  

---



### **DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1248543569157947435)** (2 条消息): 

- **澄清 Mixtral 专家迷思**：分享的一段 YouTube 视频（[链接](https://www.youtube.com/watch?v=RcJ1YXHLv5o)）澄清了一个常见的误解。视频指出 *"每层有 8 个专家，共 32 层，总计 256 个专家"*，这反驳了 Mixtral 只有 8 个专家的传闻。

- **Mixtral 令人印象深刻的参数量**：视频进一步透露 Mixtral 拥有 **467 亿参数**。每个 token 可以与 **129 亿个激活参数**进行交互。
  

---


### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 条消息): 

sinan2: 与 RWKV 和 Transformers 相比，直观的优势是什么？
  

---


### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1248553005939425321)** (8 条消息🔥): 

- **对当前 DiscoLM 模型的困惑**：一位成员对哪款 DiscoLM 模型是当前的旗舰型号表示困惑，并指出现在模型非常多。另一位成员建议在 3GB VRAM 的配置下使用 **8b llama**。
- **在有限 VRAM 上运行 Mixtral 8x7b**：一位用户提到，他们可以在 3GB VRAM 的配置上以每秒 6-7 个 token 的速度运行 **Mixtral 8x7b** 的 Q2-k 量化版本。这种配置限制了模型选择，只能选择那些内存效率更高的模型。
- **探索 Vagosolutions 模型**：另一位成员考虑重新审视 **Vagosolutions 的模型**，并指出虽然之前的体验显示性能较差，但最近的基准测试表明可能有改进。他们讨论了 Mixtral 8x7b 的微调版（finetune）是否会优于 Mistral 7b 的微调版。
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1248484586317680761)** (5 条消息): 

- **LLM 可能会集成到 Web 平台中**：一位成员分享了一个[链接](https://developer.chrome.com/docs/ai/built-in)，暗示 **LLM 可能很快就会被集成到 Web 平台中**。另一位成员提到 **Firefox 正在集成 transformers.js**，并引用了一篇关于 Firefox Nightly 中本地替代文本（alt-text）生成的[文章](https://hacks.mozilla.org/2024/05/experimenting-with-local-alt-text-generation-in-firefox-nightly/)。

- **OS 级别的 AI 集成是最终目标**：一位用户认为，最终目标可能是**在 OS 级别集成 AI**，这暗示了比 Web 平台更深层次的集成。

- **用于电子邮件操纵的提示词注入**：另一位用户分享了一个有效使用 **Prompt injection（提示词注入）** 示例的[链接](https://infosec.town/notes/9u788f3ojs6gyz9b)。他们还分享了自己在 LinkedIn 个人资料中使用类似 Prompt injection 来操纵电子邮件地址的经历，尽管结果大多是垃圾邮件。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.chrome.com/docs/ai/built-in">未找到标题</a>：未找到描述</li><li><a href="https://infosec.town/notes/9u788f3ojs6gyz9b">Tilde Lowengrimm (@tilde)</a>：天哪，我快笑疯了。我为 @redqueen@infosec.town 写了一份职位发布，在末尾加入了这段代码：&quot;我们和所有人一样深受机器编写的垃圾邮件之苦，我们更倾向于 r...
</li>
</ul>

</div>
  

---

### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1248388745376436355)** (4 messages): 

- **探索文本中的概念速度 (Concept Velocity)**：一位成员分享了来自 [Interconnected.org](https://interconnected.org/home/2024/05/31/camera) 的博客文章，讨论了可视化 Embeddings 以及测量文本中概念速度的有趣想法。他们表示有兴趣将这些想法应用于他们的天文新闻数据，并指出这个过程是“曲折”且“技术性强”的。

- **Embeddings 的降维技术**：他们发现了一篇非常有用的 [Medium 博客文章](https://medium.com/@madhugraj/explainability-for-text-data-3d-visualization-of-token-embeddings-using-pca-t-sne-and-umap-8da33602615b)，解释了如何使用 PCA、t-SNE 和 UMAP 等降维技术来可视化高维 Token Embeddings。他们利用这些技术成功地可视化了 200 篇天文新闻文章。

- **UMAP 提升聚类效果**：最初使用 PCA 时，他们对 GPT-3.5 生成的主要类别进行了标注，并注意到诸如嫦娥六号月球着陆器和 Starliner 等新闻话题的聚类。他们发现，与 PCA 相比，UMAP 提供了更好的聚类结果。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://interconnected.org/home/2024/05/31/camera">Here comes the Muybridge camera moment but for text. Photoshop too</a>：发布于 2024 年 5 月 31 日星期五。2,785 字，28 个链接。作者：Matt Webb。</li><li><a href="https://medium.com/@madhugraj/explainability-for-text-data-3d-visualization-of-token-embeddings-using-pca-t-sne-and-umap-8da33602615b">Explainability for Text Data: 3D Visualization of Token Embeddings using PCA, t-SNE, and UMAP</a>：Token Embeddings 在自然语言处理 (NLP) 任务中起着至关重要的作用，因为它们编码了单词的上下文信息……
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1248379439377551463)** (2 messages): 

- **关于 Taylor Series 悬赏要求的疑问**：一位用户询问了 Taylor Series 悬赏的要求。George Hotz 似乎感到困惑，反问道：*"为什么你觉得会有要求？"*
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1248408135165677590)** (2 messages): 

```html
<ul>
    <li><strong>质疑证明的有效性</strong>：一位成员询问：“这个证明是如何证明任何东西的？”，表明在理解证明的逻辑或结果方面存在挑战。</li>
    <li><strong>符号形状维度 (Symbolic Shape Dim) 可以为零吗？</strong>：另一位成员询问符号形状维度是否可以为 0，以此探究符号表示的约束条件。</li>
</ul>
```
  

---



### **LLM Perf Enthusiasts AI ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/)** (1 messages): 

potrock: 这个太棒了。谢谢！
  

---



---



---



---



{% else %}


> 完整的各频道详细分析已为邮件格式截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}