---
companies:
- huggingface
- meta-ai-fair
- dbrx
- reka-ai
- mistral-ai
- lmsys
- openai
date: '2024-04-23T00:03:58.017305Z'
description: '**2024年**，用于训练大语言模型的数据集规模显著增长。其中，**Redpajama 2** 提供了高达 **30T token**
  的数据，**DBRX** 使用了 **12T token**，**Reka Core/Flash/Edge** 为 **5T token**，而 **Llama
  3** 则是在 **15T token** 的基础上训练而成的。**Huggingface** 发布了一个包含 **15T token** 的开源数据集，该数据集源自
  **12年** 经过过滤的 CommonCrawl 数据；这意味着只要计算资源充足，开发者就能训练出类似 **Llama 3** 的模型。


  在 Reddit 上，**WizardLM-2-8x22b** 在推理和数学基准测试中的表现优于包括 **Llama-3-70b-instruct** 在内的其他开源大模型。**Claude
  Opus** 展示了强大的零样本（zero-shot）代码错误检测能力，超过了 **Llama 3**。


  基准测试揭示了 **LMSYS 聊天机器人排行榜** 的局限性，因为一些经过指令微调的模型存在“刷榜”（gaming the system）行为。此外，一项新的
  RAG（检索增强生成）基准测试显示，**Llama 3 70B** 的表现逊于 **GPT-4**，而 **Mistral 8x7B** 依然表现强劲。


  **Huggingface** 上已经提供了 **Llama 3** 模型的高效量化版本，据用户反馈，在 3090 GPU 上其 token 生成上限约为 **9600
  个**。在安全方面，一名英国性犯罪者被禁止使用 AI 工具；同时，**GPT-4** 在利用真实漏洞方面展现出 **87% 的成功率**，这引发了人们对安全问题的担忧。'
id: 60040813-c586-425c-a173-004426b1de69
models:
- llama-3-70b
- llama-3
- wizardlm-2-8x22b
- claude-opus
- mistral-8x7b
- gpt-4
original_slug: ainews-fineweb-15t-tokens-of-commoncrawl
people: []
title: FineWeb：15万亿 Token，12年的 CommonCrawl 数据（已去重和过滤，不客气）
topics:
- datasets
- benchmarking
- quantization
- zero-shot-learning
- reasoning
- code-error-detection
- token-generation
- security
---

 

值得注意的是，Guilherme [此前曾效力于 TII UAE Falcon 40B 团队](https://x.com/ClementDelangue/status/1782065141200073122)，并负责了他们的 [RefinedWeb 数据集](https://arxiv.org/abs/2306.01116)。

在 Llama 3 发布一周后，如果你拥有算力和代码，现在就已经有了训练属于自己的 Llama 3 的数据。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

AI 模型与能力

- **WizardLM-2-8x22b 性能**：在 r/LocalLLaMA 中，根据一位用户的基准测试，WizardLM-2-8x22b 在推理、知识和数学测试中的[表现优于其他开源 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1c9s4mf/wizardlm28x22b_seems_to_be_the_strongest_open_llm/)（如 Llama-3-70b-instruct）。
- **Claude Opus 代码错误识别**：在 r/LocalLLaMA 中，Claude Opus 展示了令人印象深刻的 [0-shot 提示词代码错误识别能力](https://www.reddit.com/r/LocalLLaMA/comments/1ca12yg/claude_opus_can_spot_this_error_in_my_code_with/)，在该任务上表现优于 Llama 3 和其他模型。
- **Llama 3 zero-shot 角色扮演**：Llama 3 在 r/LocalLLaMA 中展示了[惊人的 zero-shot 角色扮演能力](https://www.reddit.com/r/LocalLLaMA/comments/1c9v2k3/the_incredible_zeroshot_roleplay_ability_of_llama3/)。

基准测试与排行榜

- **LMSYS 聊天机器人排行榜的局限性**：在 r/LocalLLaMA 中，有人担心 [LMSYS 聊天机器人排行榜在评估模型真实能力方面的作用正在下降](https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful/)，因为像 Llama 3 这样经过指令微调的模型能够针对基准测试进行“刷榜”。目前需要更全面的基准测试。
- **新 RAG 基准测试结果**：r/LocalLLaMA 发布了一项[新的 RAG 基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1c9whsv/new_rag_benchmark_including_llama3_70b_and_8b/)，对比了 Llama 3、CommandR、Mistral 等模型在处理商业文档复杂问答时的表现。Llama 3 70B 未能达到 GPT-4 级别的性能。Mistral 8x7B 依然是一个强劲的全能模型。

量化与性能

- **高效的 Llama 3 量化模型**：r/LocalLLaMA 指出，[Huggingface 上由 quantfactory 提供的 Llama 3 量化模型](https://www.reddit.com/r/LocalLLaMA/comments/1c9qufe/note_on_llama_3_quantized_models/)是目前最有效的选择。
- **Llama 3 70B Token 生成限制**：一位用户报告称，在 3090 GPU 配置下，使用 [Llama 3 70B q2_xs 生成了约 9600 个 Token](https://www.reddit.com/r/LocalLLaMA/comments/1ca2ma1/about_9k_tokens_in_thread_before_cohesion_with/) 后开始出现内容发散。该用户正在寻求延长连贯性的方案。
- **Llama 3 8B 的 AQLM 量化**：[Llama 3 8B 的 AQLM 量化版本](https://www.reddit.com/r/LocalLLaMA/comments/1c9uvlk/aqlm_quantization_for_llama38b/)已证明可以在 Transformers 和 text-generation-webui 中加载，初步测试显示其性能与基准模型持平。

审查与安全

- **性侵犯者被禁止使用 AI**：在 r/singularity 中，据报道英国一名[性侵犯者因制作儿童不雅图像而被禁止使用 AI 工具](https://www.reddit.com/r/singularity/comments/1c9fsat/sex_offender_banned_from_using_ai_tools_in/)，这引发了慈善机构的关注，他们希望科技公司能阻止此类内容的生成。
- **GPT-4 漏洞利用能力**：GPT-4 可以[通过阅读安全公告来利用真实漏洞](https://www.reddit.com/r/OpenAI/comments/1c9mw4d/gpt4_can_exploit_real_vulnerabilities_by_reading/)，在 15 个漏洞上的成功率达 87%，优于其他 LLM 和扫描器。这引发了人们对未来 LLM 可能降低漏洞利用门槛的担忧。
- **AI 生成的不安全信息**：在 r/LocalLLaMA 中，讨论了 [AI 是否能够产生尚未被广泛知晓的独特不安全信息](https://www.reddit.com/r/LocalLLaMA/comments/1c9n6ci/are_ais_actually_capable_of_producing_uniquely/)。大多数例子似乎只是基础概述，而非真正的敏感知识。

迷因与幽默

- 社区分享了各种 AI 生成的迷因和幽默内容，包括[“仓库机器人在工作 20 小时后倒下”](https://v.redd.it/wt9p6nqk4vvc1)、[蒙娜丽莎演唱 Lady Gaga 的歌曲](https://v.redd.it/nkn2abpjwvvc1)，以及[突出当前局限性的 AI 生成漫画对话](https://i.redd.it/gw2mpgdruyvc1.png)。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**Meta Llama 3 发布**

- **模型详情**：[@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440) 发布了 **8B 和 70B** 尺寸的 Llama 3 模型，**400B+ 模型仍在训练中**。Llama 3 使用了 **128K 词表 tokenizer** 和 **8K 上下文窗口**。它在 **15T tokens** 上进行了训练，并使用 **SFT, PPO 和 DPO** 在 1000 万个样本上进行了微调。
- **性能表现**：[@karpathy](https://twitter.com/karpathy/status/1781028605709234613) 指出 Llama 3 70B 在 MMLU 等基准测试中 **接近 GPT-4 级别的性能**。8B 模型优于 Mistral 7B 等其他模型。[@DrJimFan](https://twitter.com/DrJimFan/status/1781006672452038756) 强调它将是 **第一个达到 GPT-4 级别的开源模型**。
- **算力与缩放**：[@karpathy](https://twitter.com/karpathy/status/1781387674978533427) 估计 **8B 模型消耗了 130 万 A100 小时，70B 模型消耗了 640 万小时**，在 2.4 万个 GPU 集群上实现了 400 TFLOPS 的吞吐量。相对于 **计算最优（compute-optimal）** 的缩放比例，这些模型 **训练严重不足**。
- **可用性**：模型可在 [@huggingface](https://twitter.com/huggingface), [@togethercompute](https://twitter.com/togethercompute/status/1781004579817349266), [@AWSCloud](https://twitter.com/awscloud), [@GoogleCloud](https://twitter.com/GoogleCloud) 等平台获取。4-bit 量化版本允许在 **消费级硬件上运行 8B 模型**。

**反应与影响**

- **开源 AI 的进展**：许多人强调这是 **开源 AI 的分水岭时刻**，超越了封闭模型。[@bindureddy](https://twitter.com/bindureddy/status/1781028123313881206) 等人预测开源模型将在 **短短几周内达到 GPT-4 级别的能力**。
- **LLM 的商品化**：[@abacaj](https://twitter.com/abacaj/status/1781443464246559180) 等人指出，随着人们优化运行时和蒸馏（distillation），这将 **降低成本**。一些人推测这可能会挑战 OpenAI 的商业模式。
- **微调与应用**：包括 [@maximelabonne](https://twitter.com/maximelabonne/status/1781248104479494581) 和 [@rishdotblog](https://twitter.com/rishdotblog/status/1781208858612138329) 在内的许多人已经在针对编程、开放式问答等 **微调 Llama 3**。预计将出现 **大量强大的开源模型和应用**。

**技术讨论**

- **指令微调**：[@Teknium1](https://twitter.com/Teknium1/status/1781345814633390579) 认为 Llama 3 的表现反驳了最近关于微调无法教给模型新知识或能力的说法。
- **过度训练与缩放**：[@karpathy](https://twitter.com/karpathy/status/1781033433336262691) 等人讨论了 **远超计算最优（compute-optimal）** 比例的训练如何产生推理效率高且强大的模型，这可能会改变最佳实践。
- **Tokenizer 与数据**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1781001629174575126) 指出，**改进后的 128K tokenizer** 对效率至关重要，尤其是对于多语言数据。高质量的训练数据是一个关键焦点。

---

# AI Discord 回顾

> 摘要的摘要之摘要

- **Llama 3 成为焦点**：Meta 发布 **Llama 3** 引发了广泛讨论，其中 70B 参数模型的性能足以媲美 GPT-4 级别（[Teknium 的推文](https://x.com/teknium1/status/1781328542367883765?s=46&t=90xQ8sGy63D2OtiaoGJuww)），而 8B 版本表现优于 Claude 2 和 Mistral。Unsloth AI 已集成 Llama 3，承诺 **2 倍的训练速度和减少 60% 的内存占用**（[GitHub Release](https://github.com/unslothai/unsloth/releases/tag/April-Llama-3-2024)）。一段[初学者指南视频](https://youtu.be/r-heqmMYNL0)解释了该模型的 Transformer 架构。

- **Tokenizer 问题与微调修复**：微调 **Llama 3** 面临挑战，缺失 BOS token 导致训练过程中出现高损失和 `grad_norm inf`。通过 [Tokenizer 配置中的 PR](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41) 分享了修复方案。该模型庞大的 Tokenizer 词表引发了关于效率和必要性的辩论。

- **推理速度突破**：**Llama 3** 在 **Groq Cloud 上达到了每秒 800 个 token**（[YouTube 视频](https://www.youtube.com/watch?v=Z-JHgFs5BE0)），Unsloth 用户报告在 7900XT 等 AMD GPU 上速度高达 60 tokens/s。讨论还强调了 Llama 3 70B 模型在 Groq 上低于 100ms 的首字节时间（TTFB）。

- **评估与对比 LLM**：对话将 **Llama 3** 与 **GPT-4**、**Claude** 及其他模型进行了对比，尽管 lmsys 评分不错，但 Llama 3 70B 仍未完全达到 GPT-4 Turbo 的水平。**FineWeb** 数据集的发布（[Guilherme Penedo 的推文](https://x.com/gui_penedo/status/1781953413938557276?s=46)）包含 15 万亿 token，表明其有潜力超越 RefinedWeb 和 The Pile 等现有数据集。

- **新兴工具与框架**：讨论了几个新工具和框架，包括用于配置复杂应用的 **Facebook Research 开发的 Hydra**、作为 LLM 项目模板的 **LiteLLM**（[网站](https://litellm.vercel.app/)）、用于协作提示词工程（Prompt Engineering）的 **Prompt Mixer**（[网站](https://www.promptmixer.dev/)），以及用于模式控制（Schema-controlled）自动化知识图谱的 **WhyHow.AI 的 Knowledge Graph SDK**（[Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)）。

- **检索增强生成 (RAG) 进展**：**RAG** 的发展是热门话题，包括提议用于评估 RAG 模型的新基准（[Stella Biderman 的推文](https://x.com/BlancheMinerva/status/1782437494585282965)）、使用 [Llama 3 构建 RAG 聊天机器人](https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3)的指南，以及关于使用 [LangChain 的 Self-Querying Retriever 进行租房搜索](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever)的教程。

- **人类反馈强化学习 (RLHF) 见解**：一篇题为[《从 $r$ 到 $Q^*$：你的语言模型秘密地是一个 Q 函数》](https://arxiv.org/abs/2404.12358)的新论文将传统的 RLHF 方法与直接偏好优化 (DPO) 进行了比较，使理论与标准 RLHF 方法及贝尔曼方程（Bellman equation）的满足相一致。

- **优化 Transformer 模型**：讨论了优化 Transformer 模型的技术，包括在推理过程中**压缩 token 长度的近似注意力机制**（[arXiv:2401.03462](https://arxiv.org/abs/2401.03462), [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)）、通过 Activation Beacon 和 TOVA 等方法**扩展上下文长度**，以及**动态分配 FLOPs**（[arXiv:2404.02258](http://arxiv.org/abs/2404.02258)）。

- **伦理考量与法律影响**：对话涉及 AI “越狱”的伦理影响及其诱发非预期 Agent 行为的可能性，以及使用 **Nightshade** 等工具可能违反**《计算机欺诈与滥用法案》(CFAA)** 的法律风险。

- **协作努力与社区参与**：许多频道促进了项目协作，例如 **minbpe-rs**（[GitHub](https://github.com/gnp/minbpe-rs)，minbpe 的 Rust 移植版本），以及一个使用 **Cohere Command R+** 的开源匹配 AI 应用（[推文](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ)）。社区成员还分享了学习资源，例如 [LLM 微调课程](https://github.com/andysingal/llm-course/blob/main/llama_finetune/Fine-tune-basics.md)和 [Eugene Yan 关于评估 LLM 的博客文章](https://eugeneyan.com/writing/abstractive/)。

---

# 第 1 部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Llama 3 成为热门话题**：Unsloth AI 对 **Llama 3** 的集成引发了关于其潜力的讨论，正如其 [GitHub Release 页面](https://github.com/unslothai/unsloth/releases/tag/April-Llama-3-2024) 所详述，它能实现 **2倍** 的训练速度提升和 **60%** 的内存占用减少。社区正积极探索 4-bit 模型以及量化对模型质量的影响，在实验各种 Llama 3 变体方面表现活跃，包括针对不同语言优化的版本，并分享在 [Hugging Face](https://huggingface.co/unsloth) 等平台上。

**Notebook 引导**：鼓励 AI 爱好者通过在 [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 和 [Kaggle](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook) 上准备充分的 Notebook 来测试 Llama 3，为全方位的 Fine-tuning 和实验铺平道路。

**解决模型谜题与分享秘诀**：坦诚的交流揭示了从 Llama 3 模型的 Fine-tuning 和推理问题，到关于 NVIDIA Jetson Orin nano 硬件讨论的种种挑战与成功。分享了针对循环响应（looping responses）的拟议修复方案以及对有效 CUDA 利用的见解，体现了协作解决问题的文化。

**成果展示**：成就得到了充分展示，例如一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/omarnj_omartificial-intelligence-spaceal-baka-llama3-activity-7187241690506682368-E4Ss) 揭示了针对阿拉伯语 Fine-tuning Llama 3 的精湛技艺，以及瑞典语模型 'bellman' 的首次亮相。[Ghost 7B Alpha 语言模型](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha) 因其在英语和越南语方面的优化也受到了关注。

**建议中的想法与输入**：#suggestions 频道的对话提供了宝贵的收获，例如对模型合并（model merging）和 CUDA 调试教程的需求，以及 Unsloth Studio 实现多 GPU 功能的潜力。为了提高可读性而对服务器欢迎消息进行的调整，显示了对社区反馈的响应。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI 模型同台竞技**：工程师们正在积极比较 **Llama 3**、**Claude 3 Opus**、**GPT-4** 和 **GPT-4 Turbo** 等 AI 模型在从法律文档分析到编程等任务中的表现。在限制 **Perplexity AI** 仅使用特定术语列表方面存在一些挑战，且 **Claude 3 Opus** 的每日查询上限为 **50次**。

- **协作成长**：鼓励社区成员互相支持，例如一位用户在寻求关于 AI 开发的**导师指导和资金**建议时，虽然没有立即得到关于受限 API 输出的回复，但获得了 **Y Combinator** 和**互联网学习平台**等资源推荐，以助力学习和成长。

- **Perplexity 备受瞩目**：Perplexity AI 因 **Nandan Nilekani** 的赞誉以及一段详述与 **Meta AI Yann LeCun** 会面的 YouTube 视频而受到关注。关键讨论正被公开分享，以突出多样化的查询和 AI 广博的知识库，强调了*集体知识共享的文化*。

- **API 使用讨论**：工程师们讨论了 **Perplexity API**，强调了使用计数器的可见性，并寻求关于 API 额度刷新率的澄清。目前似乎需要关于 API 配额消耗的实时反馈，但尚未提供关于刷新率的具体信息。

- **未经授权的使用与自托管方案**：社区正在讨论关于*中国平台*未经授权使用 API Key 的问题、其对服务可靠性的影响以及账号交易。一些成员倾向于将自托管作为可靠的解决方案，并分享了关于设置 **Ollama Web UI** 的指南。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**困惑于多 GPU 上下文推理**：成员们正在评估如何使用**多 GPU** 对 Jamba 等模型进行长上下文推理。他们探索了 [DeepSpeed](https://www.deepspeed.ai/) 和 [Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/index) 等工具，但收效甚微；尽管目前尚不支持 Jamba，但 **vLLM 的张量并行（tensor parallel）方案**看起来很有前景。

**震撼的数据集发布**：[Hugging Face](https://huggingface.co/datasets/Verah/latent-CIFAR100) 上分享了一个**潜空间 CIFAR100 数据集**。令社区成员惊讶的是，尽管大多数潜变量（latents）无法准确解码，但使用简单的 FFN 仍能达到约 **19% 的准确率**。

**DeepMind 发布用于网络构建的 Penzai**：[Penzai](https://github.com/google-deepmind/penzai) 是 DeepMind 推出的一款用于神经网络创新的 **JAX 研究工具包**，引起了广泛关注。同时，[rubiks.ai](https://rubiks.ai) 的一款高级研究助手和搜索引擎正在寻求 Beta 测试人员，该工具提供 **Claude 3 Opus 和 GPT-4 Turbo** 等模型的试用高级访问权限。

**WorldSim 功能丰富的回归**：WorldSim 的重新发布包含了 WorldClient 和 Mind Meld 等功能，采用了新的 token **按需付费模式**，并提供了一系列不同成本配置的模型（**Opus, Sonnet, Haiku**）。

**全方位审视 LLM**：论坛讨论了 **Llama 3 8B** 和 **Mistral 7B** 之间微小的性能差距，尽管 Llama 拥有更大的数据集。同时，对 **Llama 3 70B** 的评估显示出更多潜力，而关于“grokking”一词的相关性（特别是在 LLM 方面）存在不同立场。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **解决 LM Studio 中的 GPU 使用问题**：工程师报告称 **LM Studio** 将额外的 GPU 整合进一个更大的 VRAM 池中，但有时单个 GPU 上的 CUDA 利用率仍然很高。MacOS 用户指出 Metal 可能不会遵循 GPU 设置，从而影响机器温度。

- **模型搜索机制故障**：用户在搜索和下载模型时遇到了 **503** 和 **500 错误**，这可能与 Hugging Face 持续的服务中断有关，影响了 LM Studio 的模型搜索和下载功能。

- **LM Studio 配置查询与教程**：社区协助解决了关于配置 **WizardLM 2** 的困惑，包括一篇关于微调 token 使用的 [Reddit 教程](https://www.reddit.com/r/LocalLLaMA/comments/1c7dkxh/tutorial_how_to_make_llama3instruct_ggufs_less/)。讨论还详细阐述了 **< Instruct >** 模型与 **Base** 版本的行为差异，并解决了 **Llama 3** 中的死循环问题。

- **探索外部访问与多 GPU**：提出了关于通过自定义域名托管在 LM Studio 中本地运行的 AI 的咨询，并讨论了多 GPU 设置，提出了关于功耗和技术配置的观点。

- **关于语言模型 Token 的深入讨论**：技术人员澄清了 token 与音节一致的误解，解释了子词编码（subword encodings）。对话还批评了语言模型典型的 **50,000 token** 训练数据量，从性能和复杂性平衡的角度进行了考量。

- **多样化的硬件兼容性与设置**：确认了 NVIDIA Jetson Orin 与 LM Studio 的兼容性，同时引用了 [Reddit 上的 GPU 购买指南](https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/)，供希望为 LM Studio 优化硬件设置的用户参考。

- **AMD ROCm 预览版在 Llama 3 上表现出色**：LM Studio ROCm Preview 0.2.20 版本现在支持 **MetaAI 的 Llama 3**，仅限使用来自 "lmstudio-community" 的 GGUF 文件，并可在 [LM Studio ROCm 网站](https://lmstudio.ai/rocm)访问。AMD GPU（如 7900XT）表现出令人印象深刻的 token 生成速度，约为 60 tokens/s。多显卡的兼容性和资源分配是热门话题，一些用户成功实现了在 LM Studio 中优先使用指定的 AMD GPU。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **新用户在 Stable Diffusion 中的入门难题**：新用户在开始使用 **Stable Diffusion** 时遇到了障碍，即使参考了 YouTube 的安装指南也无济于事；建议指向 ComfyUI 和 [Clipdrop's Stable Diffusion](https://clipdrop.co/stable-diffusion) 等界面作为入门点。

- **对 AI 进展感到应接不暇**：成员们感叹生成式 AI 的发展速度惊人，特别是在 **Stable Diffusion** 工具和模型方面。

- **技术支持小组解决 Stable Diffusion 问题**：用户分享了在 Kohya 中定位已保存的 **Stable Diffusion** 训练状态的解决方案，重点在于从 checkpoint 恢复以及检查输出文件夹中的保存数据。

- **深入探讨 VRAM 在图像生成中的作用**：关于图像生成 GPU 升级的咨询引发了讨论，涉及更大 VRAM 带来的多图生成能力以及更换 GPU 后升级驱动的问题。

- **释放 AI 艺术创作力的平台**：新社区成员询问了用于创作 AI 驱动图像的工具，并被引导至与 **Stable Diffusion** 集成的 Web 界面和本地服务，例如 **bing image creator** 以及 Stability AI 官网 [Core Models – Stability AI](https://stability.ai/core-models) 上列出的平台。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kernel 性能与内存突破**：一个新的 Kernel 实现显著地将 **'matmul_backward_bias' Kernel 性能提升了约 4 倍**，另一项优化帮助 **减少了约 25% 的内存消耗**（从 14372MiB 降至 10774MiB）。关于 dtype 精度的讨论建议使用混合精度来平衡性能和内存使用，同时**考虑将操作从线性复杂度降低到对数复杂度**以提高效率。

- **应对 NVIDIA 库的细微差别**：**cuDNN** 和 **cuBLAS** 函数的集成正在进行中，其中 **`dev/cuda` 中针对 cuDNN Forward Attention 和 FP16 cuBLAS Kernel 的 PR** 显示出显著的速度提升。成员们探讨了使用这些库进行精确混合精度训练的复杂性，以及**自定义 backward pass 实现**在解决梯度计算效率低下方面的潜力。

- **探索数据并行的效率**：社区评估了使用 **NCCL** 扩展多 GPU 支持的不同方法，辩论了单线程多设备、多线程或多进程设置。共识倾向于一种 **类 MPI 架构**，该架构将支持超过 8 个 GPU 的配置并适应多主机环境。

- **GPU 计算中的梯度与量化质量**：引入了一种旨在 LLM 推理期间动态调整计算的 **Effort** 算法，目标是在 **Triton** 或 **CUDA** 中实现。此外，关于 **HQQ+ 结合 LoRA 导致 20% 速度下降**的讨论表明仍有优化空间，而一个新的 **fused `int4 / fp16` Triton Kernel** 在 [GitHub pull request](https://github.com/pytorch-labs/ao/pull/153) 中展示了优于默认 `hqq.linear` 前向计算的性能。

- **社区协作与技术支持**：CUDA MODE 社区重点协作解决了一系列问题，包括反向传播期间的 **Colab 会话崩溃**、在 **Triton Kernel** 中处理灰度图像转换，以及选择合适的 GPU 来构建机器学习系统。成员们就 **在 JAX 中实现 denseformer 时的内存管理** 提供了高层建议，并分享了如 `check_tensors_gpu_ready` 等用于验证内存中连续数据的实用资源。

- **CUDA 学习机会与社交参与**：宣布了 *CUDA-MODE 第 15 讲：Cutlass*，通过持续的 **CUDA 系列讲座** 来加深对 CUDA 编程的理解。在非正式方面，部分社区成员在德国明斯特（被戏称为“GPU 之都”）举行了线下聚会。

- **整合视听资源**：提到了上传至 [Google Drive](https://drive.google.com/file/d/1fEdpmWPYD_Ci4ydbqgTG9ThYxunmUx7e/view?usp=sharing) 等渠道的讲座 **YouTube 录像**，展示了社区提供多种学习模式的承诺。

- **活动后勤与管理员管理**：引入了新的“Moderator”角色，具备维护服务器秩序的能力，并强调了活动管理的协调，暗示了一个结构化且管理良好的社区环境。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**LLaMa-3 的 BOS Token 问题已解决**：针对 LLaMa-3 的 fine-tuning 过程进行了一项重要修复，因为缺失 BOS token 曾导致问题；目前已通过 [tokenizer configuration 中的 PR](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41) 进行了修正。

**LLaMa-3 Fine-Tuning 遇到障碍**：在尝试对 LLaMa-3 进行 fine-tune 时，一位用户遇到了神秘的 **RuntimeError**，并指出该问题在 Mistral 和 LLaMa-2 等其他模型中并未出现。

**Tokenizing 难题**：LLaMa-3 tokenizer 庞大的词表引发了关于其必要性和效率的辩论，一些人倾向于精简的方法，而另一些人则为其能够以更少的 tokens 编码长文本的能力辩护。

**大型 LLMs 的 VRAM 消耗详情**：提供了一份关于大型 LLMs 的 VRAM 使用情况明细，显示在高达 "81920 tokens" 的 batch size 下，logits 和 hidden states 的大小分别达到了 "19.57GiB" 和 "20GiB"。

**Axolotl 的数据集自定义资源**：为寻求理解自定义数据集结构的用户提供了 [Axolotl 数据集文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 的链接，其中提供了针对各种训练任务的关键示例和格式。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **智能手机上的智慧：移动端 LLM**：爱好者报告称，在运行 **Llama 3** 等量化语言模型时，**Samsung S24 Ultra** 达到了 **4.3 tok/s**，而 S23 Ultra 达到了 **2.2 tok/s**。关于这项技术实用性的讨论参考了多个链接，包括 [Pixel 的 AI 集成](https://store.google.com/intl/en/ideas/articles/pixel-feature-drop-december-2023/) 和 [结合 TensorFlow Lite 的 MediaPipe](https://developers.googleblog.com/2024/03/running-large-language-models-on-device-with-mediapipe-andtensorflow-lite.html)。

- **Self-Attention 的内部机制**：针对 Transformer 模型中的 token 是否需要关注自身的 key-value 进行了技术审查。提出了通过实验性消融（ablation）来评估其对模型性能影响的建议，为未来的研究奠定了基础。

- **聚焦 Hugging Face 的财务可行性**：社区成员思考了 **Hugging Face 的商业模式**，特别是他们的大文件托管策略，在质疑可持续收入来源的同时，将其与 GitHub 的模式进行了对比。

- **寻求提升 LLM 的推理能力**：在关于评估语言模型推理能力的讨论中，Chain of Thought (CoT) 方法似乎占据主导地位，但对替代推理基准的需求依然强烈。缺乏更深层次的推理指标凸显了在 CoT 之外进行研究的必要性。

- **优化器对决：寻求平稳训练**：为了解决训练不稳定性，建议采用 **Stable AdamW** 优化器，而不是带有 clipping 的原生版本。技术专家讨论了精细的参数调整和梯度直方图分析，以优化模型训练的稳定性。

- **Megalodon 占据一席之地**：工程师们讨论了 Meta 的新架构 **Megalodon** 所谓的优越性，该架构在处理长上下文方面表现出色，尽管其普适性以及与其他机制相比的性能仍需通过更广泛的使用和对比分析来验证。

- **探索任务向量空间**：对 AI 中“任务向量（task vectors）”的探索揭示了一种“即时（on-the-fly）”改变预训练模型行为的方法，从而实现动态的知识专业化——这一话题基于[最近的一篇论文](https://arxiv.org/abs/2212.04089)。

- **RAG 基准测试难题**：这暗示了针对综合多方面信息的 RAG 模型开发基准测试的新前沿。担忧包括模型可能会因为在与基准内容相似的数据集上进行训练而获得不公平的优势。

- **缩小推理占用空间的近似创新**：讨论通过在推理过程中近似 Attention 机制来压缩 token 长度，揭示了 Activation Beacon 和 TOVA 等几种策略，具有改变动态资源分配的潜力。

- **Transformer 上下文扩展：终极前沿？**：大幅扩展 Transformer 模型上下文长度的可能性激发了人们的兴趣，讨论承认实现像 1000 万 token 这样的上下文窗口可能不仅仅需要微调，还暗示了对新型架构突破的需求。

- **关于 Chinchilla 复现的技术争论**：一场激烈的辩论围绕着 **Chinchilla** 研究的复现尝试展开，重点关注舍入细微差别和残差分析以微调模型评估，这些讨论受到了 [Twitter](https://twitter.com/kyo_takano/status/1781286971522080919) 上的互动以及对原始研究精度问题的启发。

- **DeepMind 的 SAE 探索进展**：Google DeepMind 最近的探索优先考虑 **Sparse Autoencoder (SAE)** 的扩展和基础科学，团队在 [Neel Nanda 的 Twitter](https://twitter.com/NeelNanda5/status/1781400080802779604) 和 [AI Alignment Forum](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary) 的帖子中分享了从基础设施到引导向量（steering vectors）的见解。

- **竞技场中的基准测试渴望**：一份 **Google 表格** 正在流传（[MMLU - 替代 Prompt](https://docs.google.com/spreadsheets/d/1luIEdZ_gH2GpFY9iLtM20oXemN6xaBzuGGkJAxQh-R0/edit?usp=sharing)），其中填满了 MMLU 分数，并寻求与已知基准进行对比，突显了社区的竞争精神。

- **贡献者寻求 lm-evaluation-harness 的指导**：一位热心人士寻求在贡献 **lm-evaluation-harness** 方面的帮助，正与过时的指南和某些测试目录的缺失作斗争，这凸显了项目的持续演进以及对最新文档的需求。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**C++ 悄然超越 Python**：讨论揭示了 **C++** 相比 Python/Mojo 接口的性能优势，这与绕过 Python runtime 调用有关，可能会影响 **inference** 时间。

**框架稳步前进**：对话表明构建 **Mojo 框架** 的前景光明，并期待未来能在 Mojo 中利用 Python 框架，类似于 JavaScript 和 TypeScript 之间的兼容性。

**性能之谜与增强**：一位用户报告称 **Rust** 的前缀和（prefix sum）计算显著慢于 Mojo，引发了一场性能之谜。同时，关于在 Mojo 中引入 SIMD 别名的独立辩论显示出提升语言效率和语法清晰度的势头。

**预告推文吊起技术人员胃口**：Modular 发布了一系列 **预告推文**，暗示将有重大发布。虽然细节仍然寥寥，但等待揭晓的粉丝们显然充满期待。

**视频协助请求引起共鸣**：一名成员请求对其 AI 进化视频进行点赞和反馈，这不仅是在寻求社区支持，也反映了即使在时间紧迫的情况下对 **AI 教育** 和讨论的投入。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3 挑战 Claude**：讨论指出 **Llama 3 的 70b 模型** 目前与 Claude Sonnet 旗鼓相当，而 8b 版本则超越了 Claude 2 和 Mistral。社区围绕各种 AI 模型的对比性能展开了积极讨论，并分享了针对 HF Pro 用户的 **MistralAI/Mixtral-8x22B-Instruct-v0.1** API 访问见解，展示了 AI 模型开发中的竞争态势。

- **硬件难题与停机困境**：机器学习任务的硬件适用性是一个交流话题，特别是 **AMD RX 7600 XT** 与高端型号及 Nvidia 产品的对比。同时，由于 HuggingFace 服务中断，有报告称操作受到干扰，凸显了项目对这些 AI 平台稳定性和可用性的依赖。

- **Groq Cloud 上的 AI 极速体验**：**Llama 3 在 Groq Cloud 上达到了每秒 800 个 token**，详见 [YouTube 视频](https://www.youtube.com/watch?v=Z-JHgFs5BE0)。此外，用于语言模型数据准备的 tokenizer 的重要性也是研究和讨论的重点，进一步证明了社区对性能优化和基础机器学习方面的关注。

- **RAG 与视觉工具的开拓**：开发者展示了他们的作品，包括一个结合了 Llama 3 的 **RAG 系统聊天机器人** 以及 Hugging Face Spaces 的多种创新用途。在计算机视觉领域，开源 OCR 工具 **Nougat** 和使用 [TrackNetV3](https://github.com/qaz812345/TrackNetV3) 在羽毛球追踪方面的改进受到关注，反映出对开源贡献和 AI 能力提升的强烈倾向。

- **NLP 精华与 Diffusion 讨论**：在 NLP 领域，一名成员提出了 **PHI-2** 模型的微调困难，并宣布了一个新的 `minbpe` 的 Rust 移植版本，吸引了社区协作。Diffusion 模型领域的对话探讨了使用 **Lora 训练以保持 inpainting 一致性** 的可能性，而另一名成员则寻求 **vespa** 模型下载方面的帮助，彰显了问题解决和专业知识共享的协作氛围。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **新晋 LLM 登场**：最新的 **Nitro 驱动的 Llama 模型**现已在 OpenRouter 上线，承诺为 AI 工程师带来性能提升，可在此处[访问](https://openrouter.ai/models?q=llama-3-)。OpenRouter 最近在 **Wizard 8x22b** 上面临的挑战凸显了需求带来的压力，请注意，由于最近的负载均衡器更新，非流式请求（non-stream requests）的性能提升正在不断演进。

- **精简服务与错误 URL 修正**：在下架了 nitro 变体后，OpenRouter 已将用户重新定向到标准的 **DBRX 132B Instruct 模型**，确保工程师可以继续使用可用模型进行工作。此外，**#[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231042757783588924)** 频道中之前一个具有误导性的 URL 已被修正，这再次强调了文档准确性的重要性。

- **平台间的赞誉与联动**：[KeyWords AI](https://keywordsai.co) 对 OpenRouter 的模型更新表示赞赏，这使他们能够为开发者增强功能集。这些协作努力凸显了 AI 工具和平台之间互联互通的本质，营造了一个实用性与创新并行的环境。

- **挑战 LLM 性能规范**：讨论集中在 **LLaMA-3** 等模型中多语言支持的局限性和潜力，社区成员期待在语言多样性方面有所改进。大家承认了主机更新带来的性能和策展（curation）方面的差异，并关注如何持续获取高质量的 LLM，这对于致力于开发自适应 AI 体验的工程师来说至关重要。

- **AI 中的角色扮演与创意**：AI 社区对 **Soliloquy-L3** 等专门模型表现出浓厚兴趣，该模型承诺通过支持扩展上下文来增强角色扮演能力。这一动态展示了集体追求，揭示了人们对超越传统创意 AI 应用限制的模型的内在渴望。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Llama 3 对决 GPT-4**：**Llama 3** 引发了用户讨论，一些人认为尽管它在 lmsys 上得分很高，但仍无法完全达到 **GPT-4 Turbo** 的性能。值得注意的是，Llama-3 70b 在 Groq Cloud 上的推理速度极快，响应时间不到 100ms。

- **评估与微调 AI**：从业者正在使用 [Facebook Research 的 Hydra](https://github.com/facebookresearch/hydra) 等工具进行微调应用，尽管有些人认为其文档尚不完善。此外，通过 [Google Slides](https://docs.google.com/presentation/d/14EE2j6ii4PEA0Y-wUg80weC3eJ-qx2q41uUAEqytG28/edit?usp=sharing) 展示了一种新的 **LLM Evaluation** 方法论，影响了关于实用模型评估策略的讨论。

- **值得关注的数据集与工具**：拥有 15 万亿 token 的海量数据集 **FineWeb** 的亮相引起了关注，因为它有潜力超越 **RefinedWeb** 和 **The Pile** 等数据集的性能。此外，[litellm](https://litellm.vercel.app/) 被强调为 LLM 项目的有用模板，可简化与各种模型的交互。

- **深入探讨 LLM 论文**：论文俱乐部对《Improving Language Understanding by Generative Pre-Training》的痴迷表明了该论文在领域内的持续影响力。与会者非常看重这次会议，要求将其录制并上传到 YouTube 等平台以便更广泛地传播，这展示了社区对共享学习的承诺。

- **播客热潮席卷 Latent Space**：人们对 Jason Liu 参与的最新一期 Latent Space Podcast 充满期待，这证实了社区对思想领导力和行业见解的渴望，相关信息可以在最近的 [Twitter 公告](https://twitter.com/latentspacepod/status/1781400226793673137)中找到。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Meta 的神秘举动**：关于 **Meta** 限制 **LLaMA-3 论文**发布的异常做法引发了辩论，这标志着其模型发布框架可能发生转变，但尚未提及这种分歧的原因。

**AI 工具的伦理与法律**：该小组审查了围绕 **Nightshade** 的法律和伦理考量，提到由于其具备干预 AI 训练的能力，可能与**计算机欺诈与滥用法案 (CFAA)** 产生冲突。

**提升 Diffusion Model 速度**：由 **NVIDIA**、多伦多大学和 Vector Institute 开展的研究引入了 "Align Your Steps"，这是一种加速 Diffusion Model 的方法，并在其[出版物](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/)中进行了讨论；然而，为了完全透明，有人呼吁发布训练代码。

**评估 LLM 的视觉感知能力**：引入了一个名为 **Blink** 的新基准，用于评估多模态语言模型；它特别衡量视觉感知能力，其中 **GPT-4V** 等模型与人类表现相比存在差距。**Blink 基准测试**的详细信息见[研究摘要](https://arxiv.org/abs/2404.12390)。

**NLP 编程助手的协作开发**：开发针对 **JavaScript/Rust** 的 **NLP 编程助手**引起了关注，并呼吁进行协作和知识共享，这表明工程师们正在持续追求改进自动化工具。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 模型混搭大乱斗**：工程师们正在测试各种 AI 组合，将 **Claude 3 Opus** 与 **GPT-4** 连接，并通过 Groq 集成 **LLama 3 70B**，尽管他们面临着参差不齐的结果和访问问题。讨论正在探索卷积层 (Hyena) 和 LoRa 在大语言模型中的理论应用，以优化微调方法。
  
- **Groq 的免费 AI 实力**：Groq Cloud API 的免费服务成为关注焦点，推荐意见强调 **LLaMa 3** 是一款卓越的模型。社区正在利用这一资源进行 AI 创意尝试，例如能够编写 Python 的聊天角色扮演机器人。

- **数字雅典之梦与 AI 意识辩论的碰撞**：对“数字雅典”的愿景与对 AI 意识的深度思考交织在一起，社区围绕依赖 AI 的未来社会结构以及关于意识本质的哲学辩论展开了讨论。

- **Prompt Engineering 的难题**：Prompt Engineering 出现挑战，一名成员在从 JSON 字段中提取精确文本时遇到困难，促使其转向代码解释方法。此外，分享敏感 Prompt 引发了伦理担忧，导致了对 Prompt Engineering 伦理的思考。

- **学术 AI 探索**：一位正在为其关于 AI 和生成算法的论文寻找大量资源的学者得到了指向 OpenAI 研究论文的指引，这标志着学术界对深化理解的追求。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**LlamaParse 自动化代码掌握**：与 **TechWithTimm** 的合作实现了使用 **LlamaParse** 设置本地大语言模型 (LLMs) 以构建能够编写代码的 Agent；详情和工作流概览可在 [Twitter](https://twitter.com/llama_index/status/1781375488759570829) 上查看。

**本地 RAG 正式上线**：使用 MetaAI 的 Llama-3 完全在本地构建 **RAG 应用**的指南已发布，同时附带一篇信息丰富的 [Twitter 帖子](https://twitter.com/llama_index/status/1781422831202648264)，强调了向自托管 AI 应用迈进的趋势。

**攻克 AI 谜题 'Infini Attention'**：关于 **Infini Attention** 对生成式 AI 潜在影响的解释已发布，并附带一篇见解深刻的 [LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6)。

**地理 AI 数据可视化**：**AI 融资追踪表**现在包含并显示了按城市划分的 AI 融资情况，邀请社区通过此 [Google 表格](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121) 进行审查；一条庆祝性的 [推文](https://x.com/WangUWS/status/1782069636030165106) 强调了过去一年 AI 公司的地理分布。

**增强 LLMs 的 Markdown 支持和知识图谱 SDK**：FireCrawl 与 LlamaIndex 的集成增强了 LLMs 的 Markdown 处理能力，而 WhyHow.AI 的知识图谱 SDK 现在支持构建由 Schema 控制的自动化图谱；更多探索请见相关的 [Medium 文章](https://medium.com/ai-advances/unleash-the-potential-of-llm-ready-markdown-firecrawl-and-llamaindex-integration-243e494a9eb8) 和[此处](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**闪电般的 AI 微调速度**：公会的工程师们一直在尝试使用 **Mixtral** 和 **Llama** 等快速学习模型，并指出高效微调所需的数据集规模很小。

**Groq 运行 Llama3 的卓越性能**：**Llama3** 模型在 **Groq** 硬件上表现出惊人的速度，引发了对其在实际应用中使用的兴趣，[GitHub 上的讨论](https://github.com/OpenInterpreter/open-interpreter/issues/1185) 指出了 Windows 上 OI 特有的安装 Bug。

**AI 工具中的 Bug 搜寻与权宜之计**：社区讨论了各种 Bug，例如 **M1 Macbooks** 上 O1 的空格键问题以及 **Llama 3 70b** 的性能问题。推荐的修复方法包括安装 `ffmpeg` 以及使用 **conda** 切换 Python 版本。

**Windows 的烦恼与 Macbook 的失误**：在 Windows 上运行 Open Interpreter 的 **O1** 时出现的问题可能预示着客户端故障，而 **M1 Macbooks** 上的语音识别故障在按下空格键时会导致中断。

**困惑澄清与稳定性审查**：澄清了 **O1** 与 **Open Interpret** 对 **Groq** 的兼容性。对 **Llama 3 70b** 模型的稳定性表示担忧，认为较大的模型相比其较小的对应版本可能存在更大的不稳定问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**MySQL 连接器困惑已消除**：**MySQL** 与 Cohere LLMs 的集成引发了关于使用 Docker 和直接数据库回答的问题。尽管有报告称文档过时且 `create_connector` 命令运行异常，但一个 [GitHub 仓库](https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql) 澄清了参考代码。

**Command R 禁止商用**：已明确 **Command R (以及 Command R+)** 在 **CC-BY-NC 4.0** 许可证下仅限于非商业用途，禁止在边缘设备上用于商业目的。

**AI 初创公司人才征集**：一位 AI 初创公司创始人正在积极寻找在 AI 研究和 LLMs 方面有深厚背景的专家，以协助模型微调（tuning）和语音模型。感兴趣的候选人可以通过 [LinkedIn](https://www.linkedin.com/in/vaibhav-logar) 进行联系。

**实习受阻后的替代路径**：分享了在 Cohere 实习申请被拒后追求 ML/软件工程职位的建议，包括利用大学网络、寻找有非公开实习机会的公司、贡献开源项目以及参加招聘会。

**AI 伦理困境与技术更新**：讨论内容包括对 AI "jailbreaks"（越狱）及其诱导非预期 Agent 行为的伦理影响的担忧，一个使用 **@cohere Command R+** 的开源匹配 AI 应用，以及 **Prompt Mixer** 的发布，这是一个用于创建和评估 Prompt 的新 IDE，访问地址为 [www.promptmixer.dev](https://www.promptmixer.dev/)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GPU 加速成就**：一位工程师使用 **HIP 编译器**和 **OpenCL**（可能还使用了 **Rusticl**），成功在**笔记本电脑的 Vega iGPU** 上运行了**硬件支持架构 (HSA)**。这支持了本地、用户受控的 AI 环境趋势，以对抗对远程云端的依赖。

- **掌握模型精度**：用户正在排查 **tinygrad** 中 `einsum` 操作的精度问题，遇到了下溢至 NaN 值的情况。他们讨论了 `Tensor.numpy()` 是否应转换为 float64 以保持稳定性，以及从 **PyTorch** 等框架移植模型的影响。

- **tinygrad 的云端可能性**：在更广泛的行业转型背景下，关于 **tinygrad** 是否可能转向**云服务**的辩论正在进行。然而，社区表达了强烈的偏好，即保持 tinygrad 作为赋能个人的工具，而非依赖云服务。

- **改进错误消息**：正在推动改进 **tinygrad 中的错误消息**，特别是关于 GPU 驱动不匹配和 **CUDA** 版本冲突的问题。虽然这受限于 **CUDA** API 特异性的限制，但这是提升开发者体验的一个潜在改进领域。

- **George Hotz 设定议程**：**George Hotz** 预告了即将进行的讨论，包括 **MLPerf 进展**、**KFD/NVIDIA 驱动**、新的 **NVIDIA CI**、**文档**、**调度器改进**，以及关于在代码库中维持 **7500 行代码限制**的激烈辩论。他鼓励大家参加会议，并为特定参与者提供发言权。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **搅动 Mixtral 之池**：关于 **Mixtral 训练** 的讨论强调了 "*router_aux_loss_coef*" 参数的使用。调整其值可能会显著影响训练的成功。

- **提升捷克语的 Babel 支持**：正在进行通过增加数千个 Token 来扩展捷克语支持的工作，表明语言包容性是一个优先事项。社区提到了 *Occiglot 项目* 作为该领域的一个相关倡议。

- **AI 模型中的德语精度**：针对不同模型的**德语熟练程度**出现了各种担忧。成员们测试了 **Llama3** 和 **Mixtral** 模型的德语表现，指出了语法和 Tokenizer 的奇特问题，并提到一个新变体在等待进一步测试，目前处于私有状态。

- **内存开销比 Token 更重要**：已明确减少词表 Token 并不能提高推理速度；相反，受影响的是内存占用（memory footprint）。

- **聊天机器人趋向于效率化**：正在探索将经济可行的聊天机器人集成到 CRMs 中，建议对功能进行分组，并可能针对不同任务采用多种模型类型。人们对拥有像 **langchain** 这样支持性的库来促进这一过程很感兴趣。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain 端点的隐蔽性**：工程师们正在寻求定位其 **LangChain endpoint** 的指导，这是与其功能交互的关键环节。此外，还观察到 firefunction 在不同设备上存在延迟不一致的问题。

**迷失在海上的海盗口音 Swagger**：一条孤零零的消息出现在 **#langchain-templates** 频道中，寻找用于海盗口音（pirate-speak）的 **FastAPI** 路由代码，但目前缺乏进一步的互动或相关线索。

**在公海上巡航的社区创作**：创新者们高举旗帜，展示了如 **Trip-Planner Bot**、**LLM Scraper** 和 **AllMind AI** 等多样化项目。资源涵盖了用于机器人和爬虫的 [GitHub 仓库](https://github.com/abhijitpal1247/TripplannerBot)，以及在 [Product Hunt](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst) 上为 AI 股票分析师寻求支持。

**破译查询卷轴**：一位 AI 专家阐明了使用 *Self-querying retrievers* 将自然语言查询精炼为结构化查询的过程，并将其智慧记录在 [使用 LangChain Self-Querying Retriever 进行租房搜索](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever) 中。

**知识图谱舰队升级**：**WhyHow.AI** 通过升级 SDK 规划了增强知识图谱的航线，召唤勇敢的先驱者通过 [Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3) 加入 Beta 测试，为 Schema 控制的自动化机器人推波助澜。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Instruct 格式的反击**：社区正在努力解决 *llama3 instruct format* 的兼容性问题，因为它使用了一组不被 `llamafile` 和 `llama.cpp server bin` 识别的不同 tokens。这些问题在 [LocalLLaMA subreddit](https://www.reddit.com/r/LocalLLaMA/) 上被重点讨论，目前仍是讨论的热点。

- **致力于更好的对话**：`llama.cpp` 的一个更新正在进行中，旨在包含 **llama 3 chat template**，这标志着在增强用户与模型交互方面迈出了一大步。该贡献目前正在评审中，Pull Request 见 [此处](https://github.com/ggerganov/llama.cpp/pull/6751)。

- **量化模型，质的飞跃**：**llama 3 8B 量化版本** 的引入引起了广泛关注，并承诺在一天内发布到 llamafile，同时提供了一个 [Hugging Face 上的测试链接](https://huggingface.co/jartine/Meta-Llama-3-8B-Instruct-llamafile)。

- **探索 70B 的海洋**：成员们被鼓励参与测试 **llama 3 70B 模型**，虽然目前已可访问但仍存在一些小 bug，特别是提到的“损坏的停止标记（broken stop token）”。他们希望在更大范围推广之前，通过社区测试来解决这些问题。

- **性能修补**：针对 llamafiles 在不同系统上的执行进行了技术交流，指出 **llama 3 70B** 的表现优于其 8B 版本，特别是在 M1 Pro 32GB 等特定系统上，其中 Q2 量化级别未达到预期。改进和适应性仍然是讨论的焦点。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **扩展雄心**：工程师们正期待即将发布的 **100M、500M、1B 和 3B** 模型，这些模型将取代目前的 pythia 套件。它们在约 5 万亿 (trillion) tokens 上进行训练，有望提升模型产品的技术水平。

- **基准测试演进**：讨论重点关注了 **[Reinforcement Learning From Human Feedback](https://arxiv.org/abs/2404.12358)** 论文，该论文将传统的 RLHF 与 Direct Preference Optimization 进行了比较，并将理论基础与务实的 RLHF 方法（包括满足 Bellman equation）相结合。

- **评估备受关注**：社区正在辩论 MMLU 和 BIGBench 等**自动化评估**与 ChatBotArena 等**人工主导评估**的有效性，并寻求明确基于 perplexity 的基准测试在模型训练与成品模型中的适用性。

- **社区参与和反馈**：目前正在努力提高来自 **13,000 多名订阅者**的 Discord 参与度，策略包括让社区入口变得“显而易见”以及每季度的点名致谢。同时，一位成员分享了他们的 [Typefully analysis](https://typefully.com/t/AstZhn4) 并寻求定稿前的反馈。

- **等待智慧结晶**：社区对即将发布的录音充满期待，预计将在 **1-2 周内**发布，这反映了对知识共享和进展更新的高需求。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Llama 3 以更小的规模击败 Opus**：*Llama 3* 在竞技场中以卓越的表现令人印象深刻，尽管它是一个 70B 参数的模型，这表明规模并非 AI 有效性的唯一因素。
  
- **性能指标不能忽略误差范围**：讨论强调了在评估 AI 模型性能时考虑 **error bounds**（误差范围）的重要性，这意味着对比比原始数字更加微妙。
  
- **Meta 的 Imagine 获得满堂彩**：**Meta.ai 的 Imagine** 平台因其功能而受到赞誉，对话参与者渴望看到能证明其为何被认为“疯狂”的示例。
  
- **Azure 的慢动作服务测试**：由于高延迟问题，工程师们正面临 **Azure OpenAI** 的挑战，某些请求耗时高达 20 分钟，这对时间敏感型应用非常不利。
  
- **是被限流还是运气不好？**：**Azure 实例上反复出现的速率限制 (rate limiting)**，甚至 15 秒内 2 个请求就会触发限制，导致工程师们实施了退避策略 (backoff strategy) 来管理 API 调用频率。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Databricks 增强模型推理服务**：Databricks 推出了 [GPU 和 LLM 优化支持的公开预览版](https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving)，旨在使用 Serverless GPU 部署 AI 模型，并针对 LLM 进行了优化，无需额外配置。

- **LLM 微调有了操作手册**：贡献了一份关于微调预训练 LLM 的操作指南，推荐了 LoRA adapters 和 DeepSpeed 等优化方案，可以通过 [Modal 的微调文档](https://modal.com/docs/examples/llm-finetuning)访问。

- **节省 Serverless 部署成本**：一个 GitHub 仓库提供了廉价的 Serverless 托管选项，展示了一个 LLM 前端的设置示例，工程师可以通过[此 GitHub 链接](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html)实现。

- **社区资源互动**：一位公会成员对分享的 Serverless 推理文档表示感谢，确认了其对他们用途的实用性。

- **新技术需警惕预算**：一些成员预计 Databricks 的优化功能可能会产生巨大的成本，并对负担能力表达了幽默的担忧。



---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**Blueprint AI 技术需求**：一位工程师对使用 **AI 模型** 分析 PDF 图纸中的通风管道 **blueprints** 表示出兴趣，这表明了图像识别在建筑领域的实际应用案例。

**建筑前的 AI 预览**：工程社区讨论了 AI 作为建筑事务所 **preflight** 检查手段的兴起，用于在施工前发现问题和违反规范之处，尽管它尚未完全渗透到蓝图设计过程中。

**Llama 3 登陆笔记本电脑**：SimonW 更新了 **llm-gpt4all plugin**，使其支持在仅有 8GB RAM 的系统上运行 **Llama 3 8B Instruct**，这对使用 M2 MacBook Pro 等设备的用户来说是个福音。

**插件已就绪**：**llm-gpt4all plugin** 的 0.4 版本现已发布，支持与 Llama 3 8B Instruct 等新模型交互，详见 [最新的 GitHub release](https://github.com/simonw/llm-gpt4all/releases/tag/0.4)。

**深入探索 Llama 3**：SimonW 通过一篇详细的 [blog post](https://simonwillison.net/2024/Apr/22/llama-3/) 全面介绍了 **Llama 3** 的功能，该模型被认为是领先的开源许可模型。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **面向 AI 新手的 LLAMA 3 解析**：[YouTube 上的入门指南](https://youtu.be/r-heqmMYNL0) 拆解了 **LLAMA 3** 模型的 Transformer 架构，该指南针对希望了解这一先进模型的机器学习领域新人。指南强调了该模型的能力及其在现代 AI 开发中的作用。



---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1230776401632694272)** (1039 messages🔥🔥🔥): 

- **Unsloth AI 开发讨论**：对话涉及与 fine-tuning、pretraining 以及在不同应用中使用 Unsloth AI 相关的各种技术和策略，成员们分享了他们在训练模型方面的经验。
- **关于 Llama 模型和 Notebook 共享的担忧**：用户对个人出售 Unsloth AI 提供的开源 notebooks 或从中获利表示担忧，并讨论了这些行为的伦理问题。
- **AI 主题的 YouTube 内容创作者**：关于专注于 AI 的各种 YouTube 频道进行了良性辩论，并推荐了一些涵盖 AI 研究论文并进行有意义讨论的频道。
- **技术问题与 GPU 使用**：成员们遇到了 Hugging Face 宕机的技术困难，并讨论了训练大模型的 GPU 利用策略，以及使用 Unsloth AI 增强 context lengths 的潜力。
- **社区支持与学习历程**：分享了学习 AI 的链接和资源，一位成员对社区支持表示感谢。对话还深入探讨了学习和从事 AI 工作的个人历程，强调了该领域发展的迅猛速度。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://www.twitch.tv/hu_po">Twitch</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40079873)">no title found</a>: 未找到描述</li><li><a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/imone/Llama-3-8B-fixed-special-embedding">imone/Llama-3-8B-fixed-special-embedding · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2203.15556">Training Compute-Optimal Large Language Models</a>: 我们研究了在给定计算预算下训练 Transformer 语言模型的最优模型大小和 Token 数量。我们发现当前的大语言模型明显训练不足...</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook">Kaggle Llama-3 8b Unsloth notebook</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth 现在支持具有极长上下文窗口的 LLM 微调，在 H100 上最高可达 228K（Hugging Face + Flash Attention 2 为 58K，因此长了 4 倍），在 RTX 4090 上为 56K（HF + FA2 为 14K）。我们成功地...</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: 一门为具有一定编程经验、想要学习如何将深度学习和机器学习应用于实际问题的人设计的免费课程。</li><li><a href="https://aws.amazon.com/fr/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/">Build a robust text-to-SQL solution generating complex queries, self-correcting, and querying diverse data sources | Amazon Web Services</a>: 结构化查询语言 (SQL) 是一种复杂的语言，需要对数据库和元数据有深入理解。如今，生成式 AI 可以赋能没有 SQL 知识的人。这项生成式 AI 任务是...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L522">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3、Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L480">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3、Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">Home</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3、Mistral 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://www.youtube.com/watch?v=pK8u4QfdLx0">&quot;okay, but I want Llama 3 for my specific use case&quot; - Here&#39;s how</a>: 如果你想要一个个性化的 AI 策略来让自己和你的业务面向未来，请加入我的社区：https://www.skool.com/new-society 在 Twitter 上关注我 -...</li><li><a href="https://github.com/msaroufim/cudamodelecture1/blob/main/ncu_logs">profiling-cuda-in-torch/ncu_logs at main · cuda-mode/profiling-cuda-in-torch</a>: 通过在 GitHub 上创建账号，为 cuda-mode/profiling-cuda-in-torch 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/pull/272">Add support for loading checkpoints with newly added tokens. by charlesCXK · Pull Request #272 · unslothai/unsloth</a>: 未找到描述</li><li><a href="https://github.com/aulukelvin/LoRA_E5">GitHub - aulukelvin/LoRA_E5</a>: 通过在 GitHub 上创建账号，为 aulukelvin/LoRA_E5 的开发做出贡献。</li><li><a href="https://github.com/oKatanaaa/unsloth">GitHub - oKatanaaa/unsloth: 5X faster 60% less memory QLoRA finetuning</a>: 快 5 倍、内存减少 60% 的 QLoRA 微调。通过在 GitHub 上创建账号，为 oKatanaaa/unsloth 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=E5kzAbD8D0w">Direct Preference Optimization (DPO)</a>: 获取数据集：https://huggingface.co/datasets/Trelis/hh-rlhf-dpo 获取 DPO 脚本 + 数据集：https://buy.stripe.com/cN2cNyg8t0zp2gobJo 获取完整进阶...</li><li><a href="https://www.youtube.com/@hu-po">hu-po</a>: 关于机器学习论文、编程、研究的直播。可承接咨询和合同工作。 ⌨️ GitHub https://github.com/hu-po 💬 Discord https://discord.gg/pPAFwndTJd 📸 Instagram http://instagram.com...</li><li><a href="https://www.youtube.com/@YannicKilcher">Yannic Kilcher</a>: 我制作关于机器学习研究的视频。

论文、编程、AI 社区议题，以及 AI 对社会的更广泛影响。Twitter: https://twitter.com/ykilcher Discord: https://ykil...</li><li><a href="https://www.youtube.com/@umarjamilai">Umar Jamil</a>：我是一名来自意大利米兰的 Machine Learning 工程师，目前居住在中国，正在教我的猫“奥利奥”复杂的 Deep Learning 和 Machine Learning 概念。我也会一点中文。</li><li><a href="https://www.youtube.com/@code4AI">code_your_own_AI</a>：解释新技术。与 @code4AI 一起编写新的 Artificial Intelligence (AI) 模型——在这里，复杂的 AI 概念将基于理论物理学得到清晰的阐释。深入研究最新的进展...</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4815">main : 由 ggerganov 添加 Self-Extend 支持 · Pull Request #4815 · ggerganov/llama.cpp</a>：#4810 的延续。基于此项工作为 main 分支添加上下文扩展（context extension）支持：https://arxiv.org/pdf/2401.01325.pdf。使用约 8k 上下文和基础 LLaMA 7B v... 进行了初步的事实提取测试。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1230977365555941498)** (1 条消息): 

- **Llama 3 增强 Unsloth 训练**：Unsloth AI 宣布集成 **Llama 3**，预示着训练速度提升 **2 倍**，内存占用减少 **60%**。详细信息和发布说明可在其 [GitHub Release 页面](https://github.com/unslothai/unsloth/releases/tag/April-Llama-3-2024)查看。

- **使用免费 Notebook 探索 Llama 3**：邀请用户使用 [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 和 [Kaggle](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook) 上提供的免费 Notebook 测试 Llama 3，支持 8B 和 70B 的 Llama 3 模型。

- **探索 4-bit Llama-3 模型**：对于那些对更高效模型尺寸感兴趣的人，Unsloth AI 在 Hugging Face 上分享了 [Llama-3 8B, 4bit bnb](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) 和 [Llama-3 70B, 4bit bnb](https://huggingface.co/unsloth/llama-3-70b-bnb-4bit) 的链接，以及在其 [Hugging Face 页面](https://huggingface.co/unsloth)上的 Instruct 等其他模型变体。

- **邀请实验 Llama 3**：Unsloth AI 团队鼓励社区**分享、测试并讨论**他们使用新发布的 Llama 3 得到的模型和结果。

**提到的链接**：<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colaboratory</a>：未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1231169196390617108)** (99 条消息🔥🔥): 

- **Llama 3 模型发布与资源**：Unsloth AI 发布了 [Llama 3 70B INSTRUCT 4bit](https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit)，能够以显著更低的内存占用更快地对 Mistral、Gemma 和 Llama 模型进行 Fine-tuning。并为社区提供了一个用于 Llama-3 8B 的 [Google Colab notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)。
  
- **即将推出的教程**：针对关于 Fine-tuning 指令模型的指导请求，Unsloth AI 确认他们计划很快发布解释性教程和可能很有帮助的 Notebook。

- **程序员的自白**：成员们分享了关于编程复杂性的轻松轶事——提到在不完全掌握内部原理的情况下创建函数的情况，并就如何为生成角色对话的程序显示输出统计数据寻求建议。

- **PyTorch 和 CUDA 教育资源**：参与者分享了学习 PyTorch 和 CUDA 的宝贵资源，包括用于讲座的 [CUDA Mode YouTube 频道](https://www.youtube.com/@CUDAMODE)，并建议关注 Edward Yang 的 PyTorch 开发 Twitch 直播。

- **LLM 训练中的效率与性能**：关于在任务中是使用 Llama 3 或 Gemma 还是 GPT-4 的讨论，重点在于计算资源效率与预期性能水平之间的平衡。社区表示，保持较低的基础设施成本是一个动力因素，即使这意味着要退而求其次选择较小的模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch-dev-podcast.simplecast.com/">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit">unsloth/llama-3-70b-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=vOA9JSDPJs0">Q*</a>: 点赞 👍。评论 💬。订阅 🟥。🏘 Discord: https://discord.gg/pPAFwndTJdhttps://github.com/hu-po/docs 从 r 到 Q∗：你的语言模型秘密地是一个 Q-Fun...</li><li><a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>: 一个 CUDA 读书小组和社区 https://discord.gg/cudamode 补充内容见此处 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建</li><li><a href="https://discord.gg/rWpeuatu">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1230867448908222514)** (823 条消息🔥🔥🔥): 

- **推理问题与修复**：多位用户报告了在推理 LLaMA 3 模型时出现循环响应生成的问题。修复方法包括应用 `StoppingCriteria` 和使用 `eos_token`，但在 Ollama 与 llama.cpp 等平台之间仍存在不一致性。
- **量化困惑**：在将 LLaMA 3 量化为 GGUF 时，一位用户发现模型在 Ollama 上运行时质量显著下降（出现错误的句子、拼写错误）。
- **训练技巧与窍门**：关于使用 4-bit Unsloth 模型是否能加快训练迭代进行了交流，回复中强调了计算优化，但也提到了潜在的内存带宽限制。
- **Token 问题**：用户对 `eos_token` 设置及其如何影响模型响应感到困惑。Daniel 分享的一个解决方案涉及设置 `eos_token` 以确保响应能够正确终止。
- **硬件亮点**：讨论了新款 NVIDIA Jetson Orin nano 及其高效运行大语言模型的能力，甚至超过了某些个人电脑的性能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mlabonne/OrpoLlama-3-8B">OrpoLlama-3-8B - mlabonne 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://llama.meta.com/docs/how-to-guides/fine-tuning/">Fine-tuning | 操作指南</a>: 全参数 Fine-tuning 是一种对预训练模型所有层的所有参数进行微调的方法。 </li><li><a href="https://huggingface.co/G-reen/EXPERIMENT-ORPO-m7b2-1-merged">G-reen/EXPERIMENT-ORPO-m7b2-1-merged · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/love-actually-christmas-christmas-movie-workingtitlefilms-hugh-grant-gif-15362644">Love Actually Christmas GIF - Love Actually Christmas Christmas Movie - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44">Tomeu Vizoso 的开源 NPU 驱动项目摆脱了 Rockchip RK3588 的 Binary Blob</a>: 感谢 Vizoso 的努力，现在任何拥有 Rockchip RK3588 并运行机器学习工作负载的用户都有了 Binary Blob 驱动程序之外的替代方案。</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://tenor.com/WBcE.gif">Carson Wcth GIF - Carson WCTH Happens To The Best Of Us - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/Finnish-NLP/llama-3b-finnish-v2/blob/main/config.json">config.json · Finnish-NLP/llama-3b-finnish-v2 at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/atom-real-steel-movie-robot-fight-gif-13618149">Atom Real Steel GIF - Atom Real Steel Movie - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6775`">Issues · ggerganov/llama.cpp</a>: 使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/unslo">unslo</a>: GitHub 是 unslo 构建软件的地方。</li><li><a href="https://github.com/M-Chimiste/unsloth_finetuning/blob/main/src/finetune.py">unsloth_finetuning/src/finetune.py at main · M-Chimiste/unsloth_finetuning</a>: 通过在 GitHub 上创建账号来为 M-Chimiste/unsloth_finetuning 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/356">save_pretrained_gguf method RuntimeError: Unsloth: Quantization failed .... · Issue #356 · unslothai/unsloth</a>: /usr/local/lib/python3.10/dist-packages/unsloth/save.py in save_to_gguf(model_type, model_directory, quantization_method, first_conversion, _run_installer) 955 ) 956 else: --&gt; 957 raise RuntimeErro...</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我让 unsloth 在原生 Windows 下运行了。 · Issue #210 · unslothai/unsloth</a>: 我让 unsloth 在原生 Windows 下运行了（无需 WSL）。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写下来，但我现在在用手机...</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang: SGLang 是一种专为大语言模型 (LLM) 设计的结构化生成语言。它使你与模型的交互更快、更可控。</a>: SGLang 是一种专为大语言模型 (LLM) 设计的结构化生成语言。它使你与模型的交互更快、更可控。 - sgl-project/sglang</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/Dn0tmI0FFS">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_strategy">Trainer</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/teknium/OpenHermes-2.5">teknium/OpenHermes-2.5 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://repo.anaconda.com/miniconda/">Index of /</a>: 未找到描述</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a></li>

</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM 推理（C/C++ 实现）。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6747">llama3 family support · Issue #6747 · ggerganov/llama.cpp</a>: llama3 已发布，很高兴能在 llama.cpp 中使用 https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6 https://github.com/meta-llama/llama3
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1230785281184108584)** (54 messages🔥): 

- **Llama3 微调成功**：一位成员分享了他们使用 Unsloth Notebook 为阿拉伯语成功微调 Llama3 的经验，并提供了一篇展示结果的 [LinkedIn 帖子](https://www.linkedin.com/posts/omarnj_omartificial-intelligence-spaceal-baka-llama3-activity-7187241690506682368-E4Ss)。该成员提到，由于 Llama3 的 Tokenizer 已经能很好地理解阿拉伯语，因此未对 Tokenizer 进行任何修改。
  
- **基于 Mistral 的瑞典语模型预览**：另一位成员展示了他们新创建的基于 Llama 3 Instruct 的瑞典语模型，命名为 'bellman'，并记录了训练过程。对于感兴趣的人士，提供了 [HuggingFace 链接](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview)和模型卡片（Model Card），同时邀请大家提供反馈和特定版本的请求。
  
- **新语言模型登场**：Ghost 7B Alpha 语言模型的发布引起了热烈讨论，该模型专注于推理和多任务知识，并支持工具调用（Tool Support），主要优化了英语和越南语两种语言。成员们对这项工作表示赞赏，特别是配套的[网站](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha)和 [Demo](https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha)。
  
- **解决 GGUF 转换与生成挑战**：成员们交流了使用 Unsloth 成功训练和转换模型的技巧，包括设置正确的句子结束 Token（EOS Token）和模板格式化。他们分享了关于使用 `convert.py`、调整 Tokenizer 设置以及解决无限循环生成问题的技术代码片段——最终实现了一个功能完善的波兰语 Llama3 模型。
  
- **发布 MasherAI 的新迭代版本**：一位成员宣布发布 MasherAI 7B v6.1，该模型使用 Unsloth 和 HuggingFace 的 TRL 库训练，采用 Apache 2.0 许可证。该模型已在 [HuggingFace](https://huggingface.co/mahiatlinux/MasherAI-7B-v6.1) 上展示，并已被多次下载，表明社区对使用这一新一代模型充满热情。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview">neph1/llama-3-instruct-bellman-8b-swe-preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mahiatlinux/MasherAI-7B-v6.1">mahiatlinux/MasherAI-7B-v6.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-alpha">ghost-x/ghost-7b-alpha · Hugging Face</a>: 未找到描述</li><li><a href="https://ghost-x.vercel.app/docs/models/ghost-7b-alpha">Ghost 7B Alpha</a>: 该大型生成语言模型专注于优化卓越的推理能力、多任务知识和工具支持。</li><li><a href="https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha">Ghost 7B Alpha 游乐场</a>: 为了让每个人都能通过 Google Colab 和 Kaggle 等平台快速体验 Ghost 7B Alpha 模型。我们提供了这些 Notebook，以便你可以立即开始。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">pcuenca 支持 Llama 3 转换 · Pull Request #6745 · ggerganov/llama.cpp</a>: Tokenizer 是 BPE。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1230984501790904383)** (67 messages🔥🔥):

- **关于模型合并与 CUDA 调试的讨论**：成员们讨论了[模型合并 (merging models)](http://www.apsipa.org/proceedings/2020/pdfs/0001594.pdf)以及在 Google Colab 中使用 CUDA 的困难。有人建议使用 SSH 以获得更好的体验，并[分享了一份指南](https://www.pugetsystems.com/labs/hpc/How-To-Run-Remote-Jupyter-Notebooks-with-SSH-on-Windows-10-1477/)，介绍如何通过 SSH 设置远程 Jupyter notebook。
- **欢迎消息的挑战**：一位新手指出 PC 端欢迎消息的配色方案存在问题，促使团队进行了更改以提高可读性。
- **LLAMA 3 发布及多 GPU 愿景**：随着 LLAMA 3 的发布，人们对多 GPU (Multi-GPU) 功能充满期待，并暗示 Unsloth Studio 将是未来的开发方向。
- **针对新人的潜在配色方案调整**：一位成员建议修改欢迎消息的配色方案以提高可读性，随后一名管理员更新了配色并承认了无障碍访问的重要性。
- **招聘频道辩论**：关于在 Unsloth AI Discord 上设立专门的 #jobs 频道的效用和潜在风险的辩论；担忧包括诈骗活动以及服务器焦点从 Unsloth 特定问题上偏移。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=DdTsX6DQk24&t=2s">Lecture 14: Practitioners Guide to Triton</a>：https://github.com/cuda-mode/lectures/tree/main/lecture%2014</li><li><a href="https://www.pugetsystems.com/labs/hpc/How-To-Run-Remote-Jupyter-Notebooks-with-SSH-on-Windows-10-1477/">如何在 Windows 10 上通过 SSH 运行远程 Jupyter Notebooks</a>：能够在远程系统上运行 Jupyter Notebooks 极大地增加了工作流的灵活性。在这篇文章中，我将展示一种利用一些巧妙功能来实现这一目标的简单方法……
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1230778101517123625)** (1038 条消息🔥🔥🔥)：

- **讨论 Perplexity 的 AI 模型**：成员们在频道中提到了不同的 AI 模型，包括 **Llama 3**、**Claude 3 Opus**、**GPT-4** 和 **GPT-4 Turbo**。他们比较了这些模型在法律文件分析、编程和提示词响应等各种任务中的表现。

- **Perplexity 的使用限制和可见性**：成员们注意到 Perplexity 对 Claude 3 Opus 有 **每日 50 次查询的限制**。进一步的讨论提到，只有在剩余 10 条消息时，使用计数器才会显示。

- **关于 AI 开发和融资的建议与问题**：一位用户寻求 AI 开发方面的 **导师指导和资金支持**，并讨论了其年轻且缺乏资历的情况。社区成员建议了教育资源、申请 **Y Combinator** 等孵化器，并专注于 **基于互联网的学习**。

- **Perplexity Labs 和私有化部署**：讨论内容包括在 **Perplexity Labs 中使用其他模型** 以及在本地私有化部署 (Self-hosting) 模型。一位用户分享了设置 **Ollama Web UI** 以离线运行 LLM 模型的指南。

- **未经授权的模型使用和安全性**：有一场关于中国网站使用非正规 API 密钥以及存在 **交易此类账户** 市场的对话。用户建议采用多渠道来源以避免停机，并对这种做法影响服务可靠性表示担忧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity Model Selection</a>: 使用 jQuery 为 Perplexity AI 添加模型选择按钮</li><li><a href="https://docs.openwebui.com">🏡 Home | Open WebUI</a>: Open WebUI 是一个可扩展、功能丰富且用户友好的自托管 WebUI，旨在完全离线运行。它支持各种 LLM 运行器，包括 Ollama 和 OpenAI 兼容的 API。</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: 体验世界上最快的推理速度</li><li><a href="https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/">不仅仅是 OpenAI 的套壳：Perplexity 转向开源</a>: Perplexity CEO Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己找到了一种方法，不仅可以与 Google 搜索竞争，还可以与 OpenAI 的 GPT 竞争。</li><li><a href="https://decoder.sh/videos/use-your-self_hosted-llm-anywhere-with-ollama-web-ui">在任何地方使用你的自托管 LLM 与 Ollama Web UI</a>: 未找到描述</li><li><a href="https://www.ycombinator.com/apply">申请 Y Combinator | Y Combinator</a>: 要申请 Y Combinator 项目，请提交申请表。我们每年分两批接收公司。该项目包括每周二的晚宴、与 YC 合伙人的办公时间以及访问权限...</li><li><a href="https://en.wikipedia.org/wiki/Languages_used_on_the_Internet">互联网上使用的语言 - 维基百科</a>: 未找到描述</li><li><a href="https://www.youtube.com/@AndrejKarpathy/videos">Andrej Karpathy</a>: 常见问题 Q: 我该如何付钱给你？你有 Patreon 之类的吗？A: 作为 YouTube 合作伙伴，我会分享视频中少量的广告收入，但我没有维护任何其他额外的付费渠道。我...</li><li><a href="https://x.com/AravSrinivas/status/1781721468180767002">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 8b 非常棒。可以用它创造更多体验。我们有一些想法。敬请期待！↘️ 引用 MachDiamonds (@andromeda74356) @AravSrinivas 你会将免费版 Perplexity 切换到...</li><li><a href="https://tenor.com/view/think-about-it-use-your-brain-use-the-brain-think-brain-gif-7914082">动动脑筋 GIF - 动动脑筋 使用你的大脑 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/yt-youtube-logo-gif-27453294">Yt Youtube GIF - Yt Youtube Logo - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://dreams-of-an-electric-mind.webflow.io/eternal">永恒模式 • 无限后室</a>: 人工智能的疯狂梦想 - 不适合胆小或心理承受能力弱的人</li><li><a href="https://www.morphic.sh/">Morphic</a>: 一个完全开源的 AI 驱动回答引擎，具有生成式 UI。</li><li><a href="https://tenor.com/view/robot-depressed-marvin-hitch-hikers-guide-to-the-galaxy-gif-4931652">机器人抑郁 GIF - 机器人抑郁 Marvin - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/philschmid/llm-sagemaker-sample/blob/main/notebooks/deploy-llama3.ipynb">llm-sagemaker-sample/notebooks/deploy-llama3.ipynb 在 main 分支 · philschmid/llm-sagemaker-sample</a>: 通过在 GitHub 上创建账号来为 philschmid/llm-sagemaker-sample 的开发做出贡献。</li><li><a href="https://www.google.com/amp/s/www.xataka.com/aplicaciones/ultimo-openai-llega-a-copilot-asistente-programacion-evoluciona-nuevo-modelo-ia/amp">OpenAI 的最新成果来到 Copilot。编程助手随着新的 AI 模型而进化</a>: 在过去的一年里，人工智能不仅是 DALL·E 等图像生成器和 ChatGPT 等聊天机器人背后的推手，它还...</li><li><a href="https://youtu.be/LGuA5JOyUhE?si=AzhxwS7mCeYXwTGA">Perplexity CTO Denis Yarats 谈 AI 驱动的搜索</a>: Perplexity 是一款 AI 驱动的搜索引擎，用于回答用户的问题。Perplexity 成立于 2022 年，估值超过 10 亿美元，最近月活跃用户突破了 1000 万...</li><li><a href="https://www.google.com/amp/s/www.genbeta.com/actualidad/amazon-invierte-4-000-millones-dolares-anthropic-para-hacer-frente-a-chatgpt-lucha-mejor-ia-solo-acaba-comenzar/amp">亚马逊向 Anthropic 投资 40 亿美元以对抗 ChatGPT：最强 AI 之争才刚刚开始</a>: OpenAI 凭借 ChatGPT 的发布震撼了整个行业，促使越来越多的公司投资生成式 AI 技术。这导致了...</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: 使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave 和 Serper 构建一个受 Perplexity 启发的回答引擎</a>: 使用 Next.js, Groq, Mixtral, Langchain, OpenAI, Brave 和 Serper 构建一个受 Perplexity 启发的回答引擎 - developersdigest/llm-answer-engine</li><li><a href="https://youtu.be/YKMDw7ERxZ4?si=t0y">

byzaEgUZNsihl">AWS re:Invent 2023 - 客户主题演讲 Anthropic</a>：在这场 AWS re:Invent 2023 炉边谈话中，Anthropic 的 CEO 兼联合创始人 Dario Amodei 与 Amazon Web Services (AWS) 的 CEO Adam Selipsky 讨论了 Anthr...</li><li><a href="https://youtu.be/hFUaXEXfNnA?si=KWY0eyvRZNac2Gzt">AWS re:Invent 2023 - 客户主题演讲 Perplexity | AWS Events</a>：听取 Perplexity 联合创始人兼 CEO Aravind Srinivas 讲述这家对话式人工智能 (AI) 公司如何通过提供...来重新定义搜索。</li><li><a href="https://youtu.be/znOlwELyt8g?si=UDq4joNqi1n7z8i3">Eric Gundersen 谈 Mapbox 如何利用 AWS 每天绘制数百万英里的地图</a>：在此处了解更多关于 AWS 如何助力您的海量数据解决方案 - http://amzn.to/2grdTah。Mapbox 每天利用...收集 1 亿英里的遥测数据。</li><li><a href="https://share.wendabao.net">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ)">Rick Astley - Never Gonna Give You Up (官方音乐视频)</a>：Rick Astley 的 “Never Gonna Give You Up” 官方视频。新专辑 'Are We There Yet?' 现已发行：在此下载：https://RickAstley.lnk.to/AreWe...</li><li><a href="https://github.com/xx025/carrot">GitHub - xx025/carrot: Free ChatGPT Site List 这儿为你准备了众多免费好用的 ChatGPT 镜像站点</a>：Free ChatGPT Site List 这儿为你准备了众多免费好用的 ChatGPT 镜像站点。通过在 GitHub 上创建账号为 xx025/carrot 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1230881606563266610)** (29 条消息🔥): 

- **Perplexity AI 引起轰动**：[Infosys 联合创始人 Nandan Nilekani 赞扬了 Perplexity AI](https://www.hindustantimes.com/business/infosys-nandan-nilekani-stunning-aravind-srinivas-swiss-army-knife-perplexity-ai-search-engine-101713512251936.html)，在与联合创始人 Aravind Srinivasan 会面后，将其称为搜索领域的“瑞士军刀”。
- **YouTube 洞察 Perplexity AI 的崛起**：一段名为 [“揭秘这家挑战 Google 地位的热门 AI 初创公司”](https://www.youtube.com/watch?v=RaTxrkHSNBo) 的 YouTube 视频介绍了 Perplexity AI 的历程，以及他们为了与 Meta AI 负责人 Yann LeCun 见面而经历的漫长等待。
- **围绕 Perplexity AI 的高价值讨论**：社区成员分享了各种 [Perplexity AI 搜索查询](https://www.perplexity.ai/search/Why-using-Hdmi-Fl2oierhRze1bRncp3HgvQ) 的链接，探讨了从 HDMI 用法到积极育儿见解以及 Apple 新闻等主题。
- **分享 Perplexity AI 体验**：随着成员参与不同的 Perplexity AI 搜索查询，有人提议确保 Thread 是可分享的，突显了社区的协作性质。
- **媒体聚焦 Perplexity AI 领导层**：另一段 YouTube 视频，题为 [“Perplexity CTO Denis Yarats 谈 AI 驱动的搜索”](https://www.youtube.com/watch?v=LGuA5JOyUhE)，深入探讨了该引擎以用户为中心的功能以及自成立以来的显著增长。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.hindustantimes.com/business/infosys-nandan-nilekani-stunning-aravind-srinivas-swiss-army-knife-perplexity-ai-search-engine-101713512251936.html">Nandan Nilekani 对 Aravind Srinivas 的“瑞士军刀”搜索引擎给出了极高评价</a>：Nandan Nilekani 对 Perplexity AI 的评价，会让你迫不及待地想注册 Aravind Srinivasan 的“瑞士军刀”搜索引擎。</li><li><a href="https://www.youtube.com/watch?v=RaTxrkHSNBo">揭秘这家挑战 Google 地位的热门 AI 初创公司</a>：2022 年 8 月，Aravind Srinivas 和 Denis Yarats 在曼哈顿下城的 Meta AI 负责人 Yann LeCun 办公室外等了整整五个小时，连午饭都没吃...</li><li><a href="https://www.youtube.com/watch?v=LGuA5JOyUhE">Perplexity CTO Denis Yarats 谈 AI 驱动的搜索</a>：Perplexity 是一款回答用户问题的 AI 驱动搜索引擎。Perplexity 成立于 2022 年，估值超过 10 亿美元，最近月活跃用户突破了 1000 万...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1230842643278467103)** (4 条消息): 

- **寻求受限的 API 响应**：一位成员报告称，即使在给出指令后，也很难让 API 从精确的单词列表中选择并返回响应。他们提到尝试使用 *Sonar Medium Chat* 和 *Mistral* 模型，但均未成功。

- **寻求帮助**：该成员就其问题向他人寻求帮助，但未立即收到回复。

- **关于 API 额度刷新率的澄清**：该成员询问剩余 API 额度的更新频率，质疑在运行带有 API 请求的脚本后，更新是需要几分钟、几秒钟还是几小时。
  

---

**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1230801893991907411)** (7 条消息): 

- **多 GPU 上的长上下文推理是一个难题**：Yorth_night 正在寻求关于使用多 GPU 对 **Jamba** 进行长上下文推理的指导。尽管查阅了 [deepspeed](https://www.deepspeed.ai/) 和 [accelerate](https://huggingface.co/docs/accelerate/index) 的文档，他们仍未找到关于长上下文生成的信息。

- **寻求 Jamba 多 GPU 使用的进展更新**：Bexboy 询问 Yorth_night 面临的问题是否有任何进展。

- **VLLM 可能是 Jamba 的关键，但目前尚不支持**：Yorth_night 在另一个讨论中发现，带有张量并行（tensor parallel）的 **vllm** 可能是一个解决方案；然而，**vllm** 目前不支持 Jamba。

- **Jamba API 将会非常方便**：Yorth_night 表示希望有一个能处理完整上下文的 **Jamba API**，这将有助于评估该模型在特定任务中的能力。

- **通过上下文管理降低 Claude 3 和 Big-AGI 的成本**：Rundeen 面临 **Claude 3 和 Big-AGI** 昂贵的上下文扩展挑战。他们发现了 [memGPT](https://memgpt.ai/) 和 [SillyTavern SmartContext](https://docs.sillytavern.app/extras/extensions/smart-context/)，并正在寻找其他能够经济地管理上下文且不产生冗余或错误信息的解决方案。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1231206936532357181)** (12 条消息🔥): 

- **节奏与字节**：成员们分享了**音乐视频链接**以供娱乐，包括 **Beastie Boys 的 "Root Down"** ([高清重制版 YouTube 视频](https://www.youtube.com/watch?v=Xf1YF_MH1xc)) 和 **deadmau5 & Kaskade 的 "I Remember"** ([高清 YouTube 视频](https://youtu.be/zK1mLIeXwsQ?t=119))。
- **编码后的 CIFAR100 数据集现已发布**：一位社区成员发布了一个潜空间编码的（latently encoded）**CIFAR100 数据集**，[可在 Hugging Face 上获取](https://huggingface.co/datasets/Verah/latent-CIFAR100)，由于潜变量（latents）的大小，建议使用 *sdxl-488* 版本。
- **小规模模型的惊喜**：在 **latent CIFAR100 数据集**上使用简单 FFN 进行的初步实验显示，准确率约为 **19%**，考虑到大多数潜变量无法正确解码，这一结果令人惊讶。
- **探索更大的图像分类数据集**：询问常用的 **64x64** 或 **128x128** 分辨率的图像分类数据集，以便进行进一步实验。
- **法律、语言与 AI 的交汇**：一位具有法律背景的成员提议分享关于向量空间中语义网络和知识图谱主题的论文，强调了在语言和法律中遵循幂律（power law）的符号系统的重要性。另一位用户则分享了相关的 [arXiv 论文](https://arxiv.org/abs/2402.10588)，涉及语言模型和语言偏见。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>：我们探讨了在不平衡、以英语为主的语料库上训练的多语言语言模型是否使用英语作为内部中转语言——这是一个对于理解语言模型如何运作至关重要的问题...</li><li><a href="https://arxiv.org/abs/2311.03658">The Linear Representation Hypothesis and the Geometry of Large Language Models</a>：非正式地说，“线性表示假设”是指高层概念在某些表示空间中被线性地表示为方向。在本文中，我们解决了两个密切相关的问题...</li><li><a href="https://tenor.com/view/hellinheavns-gif-23278790">Hellinheavns GIF - Hellinheavns - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/Verah/latent-CIFAR100">Verah/latent-CIFAR100 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Xf1YF_MH1xc">Beastie Boys - Root Down</a>：高清重制版！在此阅读 Ill Communication 背后的故事：https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/ 聆听更多来自...</li><li><a href="https://youtu.be/zK1mLIeXwsQ?t=119">deadmau5 &amp; Kaskade - I Remember (HQ)</a>：▶︎ https://deadmau5.ffm.to/randomalbumtitle 在此关注 deadmau5 及其好友：https://sptfy.com/PjDO 当前巡演信息：https://deadmau5.com/shows 加入...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1231336533261553876)** (2 条消息):

- **DeepMind 的 Penzai 助力神经网络创新**：DeepMind 发布了 **Penzai**，这是一个 [JAX 研究工具包](https://github.com/google-deepmind/penzai)，旨在构建、编辑和可视化神经网络。该工具包已在 GitHub 上发布，为 AI 研究人员和开发者提供了全面的功能。

- **AI Bonanza 招募 Beta 测试人员**：发布了一款新型高级研究助手和搜索引擎，包含 **Claude 3 Opus、GPT-4 Turbo、Mistral Large 等的高级访问权限**。感兴趣的用户可以成为 Beta 测试人员，并在 [rubiks.ai](https://rubiks.ai) 使用促销代码 `RUBIX` 获得两个月的免费高级访问权限。

**提到的链接**：<a href="https://github.com/google-deepmind/penzai">GitHub - google-deepmind/penzai: A JAX research toolkit for building, editing, and visualizing neural networks.</a>：一个用于构建、编辑和可视化神经网络的 JAX 研究工具包。- google-deepmind/penzai

---

**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1231034218201485312)** (2 条消息): 

- **Worldsim 回归并带来新功能**：Worldsim 重新上线，引入了大量新功能，如 **WorldClient**（网页模拟器）、**Root**（CLI 环境）、**Mind Meld**（实体探索工具）、**MUD**（文字游戏）以及 **tableTop**（桌面 RPG 模拟器）。用户现在可以选择模型（**Opus**、**Sonnet** 或 **Haiku**）以调整成本。

- **采用按需付费模式以实现可持续发展**：为了打击垃圾信息和滥用行为，Worldsim 重启后采用了针对 token 的按需付费系统。

- **暂时受挫**：公告发布后不久，该服务因支付系统问题导致停机。一旦问题解决，将提供更新。

**提到的链接**：<a href="https://worldsim.nousresearch.com">world_sim</a>：未找到描述

---

**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1230776578561015848)** (594 条消息 🔥🔥🔥): 

- **剖析 Llama 3 性能**：成员们注意到，尽管 **Llama 3 8B** 拥有更多的参数和训练数据，但其性能仅略优于 **Mistral 7B**，讨论重点在于 Llama 3 表现出显著优势的 MMLU（多模态模型排行榜）。还有人推测基础模型是否正在达到饱和点，而改进可能来自于 Fine-tuning 技术，如 In-Context Learning 和 RLHF。

- **Llama 3 70B 仍是焦点**：尽管对 **Llama 3 8B** 感到失望，但人们对 **Llama 3 70B** 的能力持乐观态度，讨论围绕其更强的性能、在 Groq 等平台上的 Agent 应用潜力，以及 Meta AI 如何在 WhatsApp 和 Instagram 等产品中使用它。

- **Grokking 不再流行？**：“grokking”一词在社区中似乎正逐渐淡出，关于其原因以及该词在原始科幻和 Linux 系统管理员语境之外的使用是否得当，存在不同意见。

- **LLM 集成**：探索了 LLM 内部知识与检索信息之间的相互作用，重点讨论了 **RAG** 是修复了模型错误，还是无意中传播了错误的检索内容。

- **Hugging Face 服务受影响**：由于 **FineWeb**（一个 15 万亿 token 的高质量数据集）的高使用量，可能导致了 **Hugging Face** 服务（包括 **hf.space**）的性能问题，尽管具体原因尚未确认。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/justinetunney/status/1781234073471771068?s=46">Justine Tunney (@JustineTunney) 的推文</a>: @sytelus Meta LLaMA3 70B 在使用 llamafile v0.7.1 时，在 8192 token 上下文窗口下达到了 38 tok/sec。</li><li><a href="https://arxiv.org/abs/2404.10198">RAG 模型有多忠实？量化 RAG 与 LLMs 内部先验之间的拉锯战</a>: 检索增强生成 (RAG) 常用于修复幻觉并为大语言模型 (LLM) 提供最新知识。然而，当 LLM 独立错误回答问题时...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/how-to-train-sentence-transformers">训练与微调 Sentence Transformers 模型</a>: 未找到描述</li><li><a href="https://tenor.com/view/rage-gif-24341837">Rage GIF - Rage - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/huybery/status/1781172838361334015">Binyuan Hui (@huybery) 的推文</a>: 刚刚评估了 Llama3-8B-base 的编程能力👇🏻</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#aqa">未找到标题</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.08865">LLM 上下文召回依赖于 Prompt</a>: 大语言模型 (LLM) 的激增凸显了进行彻底评估以辨别其比较优势、局限性和最佳用例的关键重要性。特别是...</li><li><a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">Guilherme Penedo (@gui_penedo) 的推文</a>: 我们刚刚发布了 🍷 FineWeb：15 万亿 token 的高质量网络数据。我们过滤并去重了 2013 年至 2024 年间所有的 CommonCrawl 数据。在 FineWeb 上训练的模型优于 RefinedWeb、C4...</li><li><a href="https://x.com/thilak/status/1781352378081427925">Thilak Rao (@Thilak) 的推文</a>: 刚刚通过 @private_llm 在我的 iPhone 上运行了 @Meta 的 Llama 3 8B Instruct，在 8GB 设备上实现了完全端侧运行及完整的 8K 上下文。即将支持所有 6GB 或更多内存的 iPhone...</li><li><a href="https://x.com/benjamin_warner/status/1781095499145134263">Benjamin Warner (@benjamin_warner) 的推文</a>: 如果使用 Hugging Face 微调 Llama 3，请使用 Transformers 4.37 或 4.40。4.38 和 4.39 中的 Llama 和 Gemma 没有使用 PyTorch 的 Flash Attention 2 内核，导致内存占用过高。4.40 使用了 FA2...</li><li><a href="https://openrouter.ai/playground?models=meta-llama/llama-3-70b-instruct">OpenRouter</a>: LLM 和其他 AI 模型的路由服务</li><li><a href="https://arxiv.org/abs/2212.08037">归因问答：归因大语言模型的评估与建模</a>: 大语言模型 (LLM) 在几乎不需要直接监督的情况下展示了令人印象深刻的结果。此外，越来越多的证据表明 LLM 在信息寻求场景中具有潜力...</li><li><a href="https://github.com/google-research-datasets/Attributed-QA">GitHub - google-research-datasets/Attributed-QA: 我们认为 LLM 对其生成的文本进行归因的能力，对于信息寻求场景中的系统开发者和用户都至关重要。此发布包含了一个新问答任务——归因问答 (AQA) 的人工评分系统输出。</a>: 我们认为 LLM 对其生成的文本进行归因的能力，对于信息寻求场景中的系统开发者和用户都至关重要。此发布包含...</li><li><a href="https://x.com/_philschmid/status/1781372927516021155?s=46&t=bL0EKkuCqv4FWSLQ7lV-2w">Philipp Schmid (@_philschmid) 的推文</a>: 我正在尝试使用 Q-LoRA 微调 Llama 3 8B (70B)。为了方便起见，我想坚持使用 Llama 3 Instruct 模板。目前注意到的两件事：1. 预训练似乎...</li><li><a href="https://github.com/google-research-datasets/QuoteSum">GitHub - google-research-datasets/QuoteSum: QuoteSum 是一个文本问答数据集，包含由人类编写的、基于维基百科段落的半抽取式多源问答 (SEMQA) 示例。</a>: QuoteSum 是一个文本问答数据集，包含由人类编写的、基于维基百科段落的半抽取式多源问答 (SEMQA) 示例。 - google-research-datasets/QuoteSum</li><li><a href="https://github.com/FasterDecoding/Medusa">GitHub - FasterDecoding/Medusa: Medusa: 通过多解码头加速 LLM 生成的简单框架</a>: Medusa: 通过多解码头加速 LLM 生成的简单框架</li>

th 多解码头 - FasterDecoding/Medusa</li><li><a href="https://www.youtube.com/watch?v=z5rRZdiu1UE">Beastie Boys - Sabotage</a>: 高清重制版！在此阅读 Ill Communication 背后的故事：https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/ 听更多...</li><li><a href="https://github.com/mozilla-Ocho/llamafile?tab=readme-ov-file#using-llamafile-with-external-weights">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>: 通过单个文件分发和运行 LLM。通过在 GitHub 上创建账号为 Mozilla-Ocho/llamafile 的开发做出贡献。</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: Representation Finetuning for Language Models</a>: ReFT：语言模型的表示微调 - stanfordnlp/pyreft</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#aqahttps://ai.google.dev/gemini-api/docs/models/gemini#aqa">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code">Replete-AI/Rombo-Hermes-2.5-Extra-code · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1230913868704911491)** (56 条消息🔥🔥): 

- **讨论 LLM 微调**：在微调 **llama 3** 等模型时，一位成员询问是否应该在基础模型（base model）或指令模型（instruct model）上使用 1000 行 Alpaca 格式的 jsonl（包含指令、空输入和输出）进行微调。
- **vLLM 的 Jamba 支持正在进行中**：vLLM 项目正积极致力于支持 **Jamba**，如 [GitHub 上的 Pull Request #4115](https://github.com/vllm-project/vllm/pull/4115) 所示，其中包括添加 Jamba 建模文件和 Mamba 内存处理。
- **Deepspeed Zero 优化查询**：一位用户报告称，从 Deepspeed stage 2 切换到 stage 3 导致训练时间明显变慢，另一位成员确认 **Deepspeed stage 3** (DS3) 确实会更慢，因为其具有更高的 GPU 间通信开销。
- **NVLink 与跨 GPU 的层拆分**：关于使用两个带有 **NVLink 的 RTX 3090** 的最佳实践讨论表明，在单提示词任务中，通过跨 GPU 拆分层获得的性能提升可能会被 GPU 之间的通信和协调开销所抵消。
- **用于微调的合成数据生成**：关于使用 **llama3-70b** 等模型生成合成数据以进行微调任务的最佳实践存在争论，并提醒注意许可限制，即使用一个 LLM 生成的数据来改进另一个模型时可能存在的限制。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/chargoddard/mistral-11b-slimorca">chargoddard/mistral-11b-slimorca · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/4115">[Model] Jamba support by mzusman · Pull Request #4115 · vllm-project/vllm</a>: 为 vLLM 添加 Jamba 支持。此 PR 包含两部分：Jamba 建模文件和 Mamba 内存处理。由于 Jamba 是混合模型（在 Mamba 和 Transformer 层之间交替）...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1231590177172881519)** (7 条消息): 

- **RealWorldQA 基准数据集发布**：[xAI 为 Grok-1.5-vision-preview 发布了 RealWorldQA 基准数据集](https://huggingface.co/datasets/xai-org/RealworldQA?row=2)，其中包含各种挑战 AI 对真实场景中物体大小、距离、交通规则和方向理解的问题。
- **是基准测试，而非训练集**：该数据集最初被误解为训练集，但随后澄清确认其为一个基准测试（benchmark）。详情在 xAI 的博客中列出，示例包括将流程图转换为 Python 代码。
- **Obsidian 的新挑战者**：project-obsidian 的成员认为 RealWorldQA 数据集可能是测试未来版本 **Obsidian** 的一个*优秀基准*。
- **对训练数据的期待**：尽管感到兴奋，但也有人表达了在现有基准之外，对新训练数据集的渴望。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/xai-org/RealworldQA?row=2">xai-org/RealworldQA · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1230834443615473674)** (61 条消息🔥🔥):

- **统一与特定 RAG 数据库的挑战**：成员们讨论了大型统一 RAG 数据库与众多小型特定主题 RAG 数据库的有效性。人们担心从“完全错误”的索引中检索会产生“灾难性”影响——例如，获取有关鸭子蛋白质数据库（ducks' db proteins）的信息而不是 DuckDB 的信息，将严重降低性能。

- **寻求 RAG 基准测试系统**：参与者寻求评估 RAG 系统的标准数据集和基准测试建议。有人建议将 OpenAI 使用 LlamaIndex 进行 RAG 评估的**[链接](https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex)**作为一个潜在工具。

- **RAG 中的 LLama 与 Mistral 对比**：对话比较了不同模型在 RAG 设置中的功效，提到了 **Mistral 7b Instruct** 和 **llama 3 instruct**。小组似乎达成共识，认为 Mistral 7b v2 目前在标准评估中表现优于其他模型。

- **RAG 相关研究论文分享**：频道分享并讨论了各种关于 RAG 的研究论文，主题包括叠加提示（superposition prompting）和可信度感知生成（credibility-aware generation）。一篇论文介绍了一种改进和加速检索增强生成的方法，而其他论文则探讨了结合外部现实世界数据以提高 LLM 输出准确性和可靠性的问题。

- **实施独特的 RAG 方法**：简要提到了在生产系统中使用叠加提示（superposition prompting），并讨论了如何使用元数据（metadata）对信息进行排名和排序。此外，他们还分享了关于在推理（inference）过程中修改注意力矩阵（attention matrix）、利用文档元数据以及理解信息结构以增强模型性能的想法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05825">LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding</a>：最近，与传统的稀疏或词袋模型方法相比，基于嵌入的检索或密集检索显示出了最先进的结果。本文介绍了一种模型无关的文档...</li><li><a href="https://arxiv.org/abs/2404.10981">A Survey on Retrieval-Augmented Text Generation for Large Language Models</a>：检索增强生成 (RAG) 将检索方法与深度学习的进展相结合，通过实现动态整合...来解决大语言模型 (LLM) 的静态局限性。</li><li><a href="https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex">Evaluate RAG with LlamaIndex | OpenAI Cookbook</a>：未找到描述</li><li><a href="https://x.com/BlancheMinerva/status/1782437494585282965">Stella Biderman (@BlancheMinerva) 的推文</a>：为 RAG 模型创建一个基准测试，其中所有问题都需要综合多个文档的信息才能回答。研究在公开数据上训练的模型在该基准上的表现...</li><li><a href="https://arxiv.org/abs/2404.06910">Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation</a>：尽管大语言模型 (LLM) 取得了成功，但它们在处理长上下文时表现出明显的缺点。它们的推理成本随序列长度呈二次方增长...</li><li><a href="https://arxiv.org/abs/2404.06809">Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation</a>：大语言模型的快速发展导致了检索增强生成 (RAG) 的广泛采用，它整合了外部知识以缓解知识瓶颈并减少...</li><li><a href="https://arxiv.org/abs/2404.06347">RAR-b: Reasoning as Retrieval Benchmark</a>：语义文本相似度 (STS) 和信息检索 (IR) 任务是过去几年记录嵌入模型进展的两个主要途径。在兴起的检索...</li><li><a href="https://arxiv.org/abs/2404.06082">A RAG Method for Source Code Inquiry Tailored to Long-Context LLMs</a>：虽然大语言模型 (LLM) 的上下文长度限制已得到缓解，但它仍然阻碍了它们在软件开发任务中的应用。本研究提出了一种结合了...的方法。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1230778645568684082)** (660 条消息🔥🔥🔥):

- **探索 WorldSim 的深度**：成员们正热切期待 WorldSim 的回归，频繁询问平台的最新状态。关于 4chan 此前的滥用及其导致的成本影响被多次提及，一些用户对由于货币化策略可能再也无法使用 WorldSim 表示遗憾。
- **社区打造替代方案**：针对 WorldSim 的停机，snowly182 和 jetblackrlsh 等成员在 Hugging Chat 上使用 Llama 3 创建了替代模拟器，提供免费的无限访问，并包含 D&D 模式等功能。
- **Llama 3 的上下文与能力**：围绕扩展 Llama 3 上下文长度的讨论已经展开，成员们将其性能与 Claude Opus 进行对比，并表示 Llama 3 虽然具有创造力，但在创意方面仍比 Opus 低几个等级。
- **与 AI 探索记忆**：rundeen 分享了一种使用独立的 GPT-4 实例来总结上下文历史的技术，建议将仿生学和更智能的上下文管理作为未来实现更高效、更具成本效益的 AI 交互的关键。
- **对排他性和成本的担忧**：鉴于 Claude 3 Opus 的高昂成本，用户对其独占使用表示担忧，并希望 Nous Research 能够集成开源模型，以提供没有经济障碍的访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: 未找到描述</li><li><a href="https://copilot.microsoft.com/images/create/a-small-west-african-village-in-a-mangrove-forest2c/1-6622b051cfb34f5d9138c10749aaf74c?id=UJUzToPRop%2fGABe0DFtu3w%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG4.7GQ0JjrYDCPZik2aLs1U&lng=en-US&ineditshare=1.">Generirao Microsoft Copilot</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/http%3A%2F%2Fplanesimulator.com%2Fcamerafollowplane%2Fmorecontrols%2Fstructures?universe=6bdef4da-5012-412f-915b-a1442f42446d-planesimulator.com">world_sim</a>: 未找到描述</li><li><a href="https://huggingface.co/vicgalle/Worldsim-Hermes-7B">vicgalle/Worldsim-Hermes-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: 体验世界上最快的推理速度</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D&#039;s WorldSim System Prompt Open Source - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://tenor.com/view/jim-carrey-ohcome-on-gif-7511567">Jim Carrey Ohcome On GIF - Jim Carrey Ohcome On - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://a.co/d/0gve1yp">未找到标题</a>: 未找到描述</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>: 在 HuggingChat 中使用 Snow World Simulator 助手</li><li><a href="https://hf.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>: 在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://hf.co/chat/assistant/65bff23f5560c1a5c0c9dcbd">Image Generator - HuggingChat</a>: 在 HuggingChat 中使用 Image Generator 助手</li><li><a href="https://websim.ai/c/BZcLXGB6Ft5cjnLns">Jailbroken Prometheus Chat</a>: 未找到描述</li><li><a href="https://www.youtube.com/@nickabenson">nickabenson</a>: 欢迎来到 Nickabenson 频道。我们的 Patreon: https://www.patreon.com/nickabenson 我们的 Amino: http://aminoapps.com/c/Nickabenson。我们主要进行游戏直播、讨论、动画制作等...</li><li><a href="https://dreams-of-an-electric-mind.webflow.io/eternal">eternal mode • infinite backrooms</a>: 一个人工智能的疯狂梦想——不适合胆小者或心理承受能力弱的人</li><li><a href="https://www.lesswrong.com/posts/ZxHfuCyfAiHAy9Mds/desiderata-for-an-ai">Desiderata for an AI — LessWrong</a>: 我认为对齐工作的重点应该放在从头开始重新设计 AI。在此过程中，我认为我们应该记住一系列理想的……</li><li><a href="https://www.youtube.com/shorts/uZhZq7ngQlo">揭秘 CIA 的星门计划（Stargate Project）和超级英雄般的中间人（Midwayers）</a>: 标签：1. #Stargate 2. #Midwayer 3. #Urantia 4. #Spiritual 5. #Extraterrestrials 6. #InvisibleRealm 7. #PlanetarySentinels 8. #CIADeclassifiedFiles 9. #Supernatura...</li><li><a href="https://hf.co/chat/assistant/66248a7a29ce1e0f4dd260fe">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://youtube.com/shorts/tVD3yTli_bU">Mephisto&#39;s Dream  | 科幻动画</a>: Mephisto 是一位软件开发人员，他创建了 World Sim，这是一个基于文本的 AI 系统，可以模拟包含意识体的整个宇宙，他相信用户交互会……</li><li><a href="https://www.suzannetreister.net/Ampages/Amenu.html">Suzanne Treister - Amiga 视频游戏剧照 - 菜单</a>: 未找到描述</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://hf.co/chat/assistant/66240">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/896WttdI1l">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://hf.co/chat/assistant/6623fcdb1a7a58ed5e441db2">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://hf.co/chat/assistant/662404223e230">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://books2read.com/u/3GPpKP">现已在您喜爱的数字商店上架！</a>: 《建筑师的难题：Quantumom vs. Data Dad》，作者 Nicholas Alexander Benson
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1230782986597699606)** (722 条消息🔥🔥🔥):

- **多模型 GPU 使用情况**：用户报告称，在添加额外的 GPU 时，LM Studio 似乎会将 VRAM 整合到一个更大的资源池中。然而，有时 CUDA 的利用率在单个 GPU 上仍保持在 100%，或者在多个 GPU 之间共享。

- **MacOS 上的 LM Studio**：有评论提到 Mac 系统上的 GPU 行为，其中 Metal 可能不会遵循在 LM Studio 中调整的 GPU 设置，导致机器运行过热。

- **搜索模型问题**：多位用户在尝试在 LM Studio 中搜索和下载模型时遇到问题，部分用户收到 `503` 或 `500` 错误。这似乎与 Hugging Face 正在发生的宕机有关。

- **RAG 与 VectorDBs 使用咨询**：一位用户询问了何时应针对文件使用检索增强生成（RAG），以及何时应使用向量数据库，特别是在需要记住用户提供的信息的系统中。背景是与 Autogen 配合使用。

- **配合自定义域名使用 LM Studio**：一位用户询问了是否可以通过域名托管本地运行的 AI，以便从任何地方访问。他们请求针对初学者如何实现此设置的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://wordcounter.net/character-count">Character Counter - WordCounter.net</a>: 未找到描述</li><li><a href="https://docs.useanything.com/feature-overview/llm-selection/lmstudio">LMStudio | AnythingLLM (由 Mintplex Labs 提供)</a>: 未找到描述</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI 兼容性 · Ollama 博客</a>: Ollama 现在初步兼容 OpenAI Chat Completions API，使得通过 Ollama 在本地模型上使用为 OpenAI 构建的现有工具成为可能。</li><li><a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">LM Studio (@LMStudioAI) 的推文</a>: LM Studio 内的模型搜索/下载可能会受到此次 Hugging Face 停机的影响。请关注后续更新 ↘️ 引用 Hugging Face Status (@hf_status) 我们正在经历一些停机...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://hub.docker.com/r/noneabove1182/lmstudio-cuda">Docker</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本发布</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/teknium1/status/1781328542367883765?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>: 好吧伙计们，我们在家也能用上 gpt-4 了</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>: 你可以通过在 localhost 上运行的 API 服务器，使用在 LM Studio 中加载的 LLM。</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">视觉模型 (GGUF) - lmstudio-ai 集合</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF (main 分支)</a>: 未找到描述</li><li><a href="https://www.youtube.com/@IBMTechnology/playlists">IBM Technology</a>: 无论是 AI、自动化、网络安全、数据科学、DevOps、量子计算还是介于两者之间的任何领域，我们都提供关于重大技术话题的教育内容。订阅以提升你的技能...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c858ac/llama3_seems_to_get_stuck_in_loops_sometimes/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code 模型排行榜 - 由 bigcode 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-GGUF">Qwen/CodeQwen1.5-7B-Chat-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9m6ei/lpt_llama_3_doesnt_have_selfreflection_you_can/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/christopherthompson81/quant_exploration">christopherthompson81/quant_exploration · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca8uxo/llavallama38b_is_released/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/Mintplex-Labs/anything-llm">GitHub - Mintplex-Labs/anything-llm：适用于任何 LLM 的全能 AI 应用，具备完整的 RAG 和 AI Agent 能力。</a>: 适用于任何 LLM 的全能 AI 应用，具备完整的 RAG 和 AI Agent 能力。 - Mintplex-Labs/anything-llm</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf：在本地与任何 PDF 聊天。提问并获取带有有用参考的回答。非常适合数学 PDF（将其转换为 LaTeX，一种计算机可理解的数学语法）</a>: 在本地与任何 PDF 聊天。提问并获取带有有用参考的回答。非常适合数学 PDF（将其转换为 LaTeX，一种计算机可理解的数学语法） - Crizomb/ai_pdf</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c8c7xj/easiest_way_to_setup_rag_windows_nvidia_gpu/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1小时演讲] 大语言模型简介</a>: 这是一场面向普通观众的 1 小时大语言模型简介：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-</a>

Plugin-for-LM-Studio: 为以服务器模式运行的 LM Studio 创建 ChromaDB 向量数据库的插件！</a>: 为以服务器模式运行的 LM Studio 创建 ChromaDB 向量数据库的插件！ - BBC-Esq/VectorDB-Plugin-for-LM-Studio</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>: 包含路线图和 Colab 笔记本的 Large Language Models (LLMs) 入门课程。 - mlabonne/llm-course</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/1684">ikawrakow 提交的 k-quants · Pull Request #1684 · ggerganov/llama.cpp</a>: [内容] 此 PR 增加了一系列 2-6 bit 量化方法以及量化混合方案，如 #1240 和 #1256 中所述。提供了 Scalar, AVX2, ARM_NEON 和 CUDA 实现。原因在于...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1230786065221292084)** (358 条消息🔥🔥): 

- **WizardLM 配置困惑**：成员们正在寻求 **WizardLM 2** 的配置帮助，其中一人尝试将 Hugging Face 模型卡片中的信息转换为预设，另一位成员分享了一个关于使用 llama.cpp 命令解决 token 问题的 [Reddit 教程](https://www.reddit.com/r/LocalLLaMA/comments/1c7dkxh/tutorial_how_to_make_llama3instruct_ggufs_less/)。
- **LM Studio 对 JSON Mode 的支持**：一位成员询问 LM Studio Playground 中的 JSON Mode 是否会在服务器模式下可用，但目前尚未提供确认或解决方案。
- **模型行为细节探讨**：讨论集中在模型的 **< Instruct >** 版本上，这些版本经过训练，相比于往往更具随机性的 **Base** 模型，能提供更连贯且相关的回复。
- **Llama3 无限循环问题**：用户报告 **Llama3** 模型在生成过程中进入无限循环的问题，并建议使用特定的配置和更新来解决该问题，例如在 Advanced Configuration 中添加停止字符串（stop strings）。
- **多样的 Llama3 体验**：社区成员分享了关于 **Llama3** 性能和审查制度的不同体验和讨论，一些成员发现 70B 模型在指令遵循方面表现出色，而另一些成员则面临无意义的输出或过度的内容生成。大家还交流了关于调整系统提示词（system prompts）以影响 AI 行为并移除审查的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://doc.pypy.org/en/latest/sandbox.html">PyPy 的沙箱功能 &mdash; PyPy 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/AI-Engine/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_k_m_with_temp_stop_token_fix.gguf?download=true">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/raincandy-u/Llama-3-Aplite-Instruct-4x8B">raincandy-u/Llama-3-Aplite-Instruct-4x8B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/yoda-star-wars-learning-gif-21964563">尤达大师 GIF - 尤达星球大战 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c7dkxh/tutorial_how_to_make_llama3instruct_ggufs_less/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF">MaziyarPanahi/WizardLM-2-7B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/models?other=base_model:meta-llama/Meta-Llama-3-8B-Instruct">模型 - Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=jaM02mb6JFM">M3 max 128GB 用于运行 Llama2 7b 13b 和 70b 的 AI</a>: 在本视频中，我们使用配备 128GB 内存的新款 M3 max 运行 Llama 模型，并将其与 M1 pro 和 RTX 4090 进行对比，以查看该芯片的真实性能...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>: llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号为 abetlen/llama-cpp-python 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1231950083847880788)** (1 条消息): 

- **Hugging Face 停机影响 LM Studio**: 用户收到通知，由于 Hugging Face 停机，**模型搜索和下载**目前无法正常工作。来自 [LM Studio 状态](https://x.com/lmstudioai/status/1782390856986550384?s=46) 的更新显示，他们正在监控情况。

**提到的链接**: <a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">来自 LM Studio (@LMStudioAI) 的推文</a>: LM Studio 内的模型搜索/下载可能会受到此次 Hugging Face 停机的影响。请关注后续更新 ↘️ 引用 Hugging Face Status (@hf_status) 我们正在经历一些停机...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1230943732031946873)** (18 条消息🔥):

- **错误窗口的人机工程学抱怨**：一位成员对错误显示窗口过窄且无法调整大小表示不满，指出由于内容采用垂直布局，窗口高度应该更高。
- **排除模型加载错误**：几位用户报告了加载模型时的错误，并提供了日志文件中的详细信息，提到了退出代码，并建议尝试不同的模型或配置。
- **应用更新功能故障**：一位用户报告应用内更新功能运行缓慢，耗时 30-40 分钟才完成，最初看起来像是功能失效。
- **对 LM Studio 的感谢**：一位成员对 LM Studio 在其专业写作和 AI 研究方面的影响表达了衷心的感谢，强调了该工具在完成任务中的重要性。
- **生成过程中的模型故障**：一位用户观察到某些模型（尤其是新的 llama）在生成响应时出现故障，有时会输出数字而不是答案。
- **LM Studio 的 VPN 证书问题**：多位用户在通过 Zscaler VPN 在 LM Studio 中下载模型时遇到问题，提到了关于证书验证的具体错误消息，并讨论了解决方法。
- **CPU 使用率显示不一致**：一位用户注意到 LM Studio 中显示的 CPU 使用率与 Windows 任务管理器之间存在差异，后者显示的利用率明显更高。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1231148383419502682)** (1 messages): 

- **寻求完整的代码输出**：一位成员对聊天机器人省略部分代码表示沮丧，特别是它忽略了编写完整代码，而是插入了类似 `// Add similar event listeners for left and right buttons` 的注释。他们正在寻找一种确保聊天机器人始终输出完整代码的方法。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1231031069206384710)** (34 messages🔥): 

- **了解 LM Studio 的 GPU 兼容性**：一位成员链接了 NVIDIA Jetson Orin 的 Amazon 页面，并询问其与 LM Studio 的兼容性，随后另一位成员确认虽然它可能比大多数台式机慢，但应该可以运行。还提供了一个 [Reddit GPU 购买指南](https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/) 的链接，用于构建适合 LLM Studio 的系统。
- **为 LLM Studio 升级笔记本电脑**：关于升级笔记本电脑 GPU 以获得更好 LM Studio 性能的咨询得到了反对建议，原因是笔记本电脑的可升级性有限，并建议如果笔记本电脑有 Thunderbolt 端口，可以检查外部 GPU 外接盒。
- **在 LM Studio 中配置多 GPU**：成员们分享了使用多 GPU 的见解，其中一位成员询问如何在不经常使用时管理新安装 GPU 的功耗。共识认为，虽然闲置 GPU 的功耗较低，但性能提升可能无法抵消电力成本和复杂性。
- **处理 LM Studio 中的配置错误**：一位用户在尝试在具有未知 GPU 检测问题的笔记本电脑上加载模型时遇到错误。解决方案包括关闭 LM Studio 设置面板中的 GPU offload 选项。
- **大模型的性能讨论**：用户分享了在 LM Studio 中使用不同 GPU 设置运行不同大小模型的经验。示例包括使用 RTX 3090 获得不同的 token 生成速度，以及考虑添加第二个 GPU（如 GTX 1060）以增加 VRAM，尽管存在对功耗和潜在极小性能提升的担忧。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.com/NVIDIA-Jetson-Orin-64GB-Developer/dp/B0BYGB3WV4/ref=asc_df_B0BYGB3WV4/?tag=hyprod-20&linkCode=df0&hvadid=652510459651&hvpos=&hvnetw=g&hvrand=10213537780974291048&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1017588&hvtargid=pla-2187237580510&mcid=fd4d223f77343b978a3f98f52420f7aa&th=1">未找到标题</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/">Reddit - 深入了解任何事物</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1231083474098585683)** (24 messages🔥): 

- **LM Studio 中的模型发现问题**：一位成员报告了 **LM Studio 0.2.20 版本**中的一个错误，即存储在 **NFS 挂载**上的模型对软件不可见，尽管它们在 0.2.19 beta B 版本中是可见的。此问题在 0.2.19 beta C、0.2.19 和 0.2.20 版本中一直存在，影响了 NFS 挂载和本地模型目录。

- **讨论模型 NFS 挂载策略**：关于 **NFS 挂载策略** 的对话显示，一位成员挂载了整个模型共享父目录以获得广泛访问权限，而另一位成员则为 LM Studio 模型挂载了特定目录，希望出于性能原因区分本地和远程模型。

- **澄清 Token 误解**：一位成员澄清了语言模型中的 **tokens** 并不一定与音节对应，解释说使用的是子词编码（subword encodings），可以代表词根、前缀和后缀，而不是整个单词或音节。

- **理解语言模型中的 Token 数量**：讨论涉及了个人词汇量与语言模型的对比，质疑了训练中常规平均 **50,000 tokens** 的说法，探讨了这一数字是设计使然，还是模型复杂性与性能之间的折中。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1231343022675984514)** (20 messages🔥): 

- **AutoGen 在两个 Token 后停止**：一位成员报告说，当 **AutoGen** 指向本地版本的 **LM Llama 3** 时，在生成两个 Token 后就会停止。他们表示很沮丧，因为 LLM 在默认设置下似乎过早停止。

- **这里不欢迎营销**：一位用户提醒另一位用户，在 Discord 服务器中营销工具是不合适的，并要求他们停止此类活动。

- **Token 限制是罪魁祸首**：针对上述问题，一位用户建议将 max tokens 设置更改为 3000，并指出执行此步骤后解决了他们的类似问题，无需删除任何文件或 Agent。

- **潜在的修复方案取得了部分成功**：在尝试了涉及 max tokens 的建议修复后，一位成员发现它部分解决了问题，但遇到了一个新问题，即 **Autogen 的 user proxy** 停止响应或错误地模仿 Agent 的响应。

- **AutoGen Manager Agent 出现问题**：另一位用户在使用 **AutoGen Manager agent** 时遇到困难，具体表现为在尝试将其与本地模型一起使用时出现“无法选择发言者”（unable to select speaker）错误。他们询问了其他人关于此问题的经验。
  

---


**LM Studio ▷ #[rivet](https://discord.com/channels/1110598183144399058/1167546635098804284/1230920636642361374)** (1 messages): 

- **服务器日志中出现异常重复**：一位成员注意到在消息 *Processing queued request...* 之后，服务器日志中出现了重复的 POST 请求，并询问这种行为是否正常。目前没有提到进一步的上下文或解决方案。
  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1231516145694150656)** (1 messages): 

- **关于 LM Studio 集成的咨询**：一位成员询问了将某种工具与 **LM Studio** 集成的事宜，并表示有兴趣阅读任何与 LM Studio 相关的特定项目信息。聊天片段中未提供更多细节或链接。
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1230986494198681600)** (4 messages): 

- **探索 LLM Studio 的替代方案**：一位成员研究了 **llamafile**，这是一个可以在 x64、arm64 和大多数 Linux 发行版等各种平台上运行的替代方案。他们强调了它在 Raspberry Pi 等设备上运行的潜力，并建议 LLM Studio 通过创建一种较慢的兼容模式，为没有 AVX2 的旧 CPU 提供支持。

- **对 AVX Beta 更新的担忧**：该成员对 AVX beta 版本可能滞后于主频道（main channel）构建版本表示担忧，指出用户希望更频繁地更新，以保持 beta 版和正式版之间的同步。

- **模型部署的替代计算资源**：同一个人指出了一种情况，即他们有可用的 GPU 来辅助非 AVX2 CPU，但可用的 AVX2 系统没有 GPU 来卸载计算任务，指出了他们在有效利用 LLM Studio 时面临的硬件限制。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1230853540340039792)** (73 messages🔥🔥): 

- **MetaAI 的 Llama 3 隆重登场**：MetaAI 的 **Llama 3** 现在可在 LM Studio ROCm Preview 0.2.20 中使用，可从官方 [LM Studio ROCm 网站](https://lmstudio.ai/rocm)获取。目前只有来自 "lmstudio-community" 的 Llama 3 GGUF 文件可以运行，其他文件由于细微的 GGUF 创建问题预计无法工作。

- **全面提速的表现**：用户报告称，Llama 3 模型在各种 AMD GPU 上具有令人印象深刻的 Token 生成速度，在 7900XT 上约为 60 tokens/s，在 7900XTX 上则略高。

- **面向初学者的 ROCm 技术**：ROCm 被誉为能够缩小 AMD 和 Nvidia GPU 之间性能差距的技术，新用户正在询问其优势；已明确 ROCm 被 LM Studio 用于加速 AMD GPU 上的 GPU 推理。

- **兼容性问题浮现**：围绕显卡与 ROCm 的兼容性展开了讨论，包括失败的尝试以及在不支持的 GPU 上运行 LM Studio 的假设性解决方案，例如建议使用第二套图形设置或虚拟化。

- **在 LM Studio 中进行 GPU 选择**：用户寻求帮助，希望引导 LM Studio 在存在多个 GPU 时使用特定的 AMD GPU，一名用户最终通过禁用之前为旧版本设置的环境变量解决了资源分配问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式和示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">系统要求 (Windows) — HIP SDK Windows 安装</a>：未找到描述</li><li><a href="https://www.howtogeek.com/disable-integrated-graphics-on-windows/">如何在 Windows 11 上禁用集成显卡</a>：当游戏和其他图形密集型应用程序开始卡顿时，这就是你该做的！</li><li><a href="https://techteamgb.co.uk/2024/03/22/how-to-turn-your-amd-gpu-into-a-local-llm-beast-a-beginners-guide-with-rocm/">如何将你的 AMD GPU 变成本地 LLM 怪兽：ROCm 初学者指南 | TechteamGB</a>：未找到描述</li><li><a href="https://youtu.be/VXHryjPu52k?t=249">如何将你的 AMD GPU 变成本地 LLM 怪兽：ROCm 初学者指南</a>：亚马逊上的 RX 7600 XT (联盟链接): https://locally.link/kEJGLM Studio: https://lmstudio.ai/rocm，由 Gigabyte 提供的产品，对于我们这些拥有 NVIDIA GPU 的人来说...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1230903153268883507)** (1 条消息): 

- **Llama 3 70B Instruction 发布**：**Llama 3 70B instruct** 的首批几个量化版本已经发布，展示了开源领域的巨大进步，其性能“足以媲美 GPT-3.5”。提供的模型（包括 **IQ1_M 和 IQ2_XS**）即使在显存 (VRAM) 低于 20 GB 的系统上也能保持合理的性能。
  
- **尺寸固然重要，效率亦然**：邀请社区成员尝试在 [Hugging Face](https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF) 上提供的新 **Meta-Llama-3-70B-Instruct** 模型，该模型与最新的 [LM Studio](https://lmstudio.ai) 兼容，并避免了无休止生成的陷阱。

- **小硬件运行大模型**：**IQ1_M 和 IQ2_XS** 模型利用重要性矩阵实现高效的 VRAM 使用，确保在内存较少的系统上获得更高的性能水平。

- **开源力量在行动**：得益于协作努力，社区可以访问最新的 Llama 3 70B instruct 贡献，包括 `llama.cpp` 上的一个拉取请求 ([PR 6745](https://github.com/ggerganov/llama.cpp/pull/6745))，强调了在推进 AI 模型方面的共同努力。

**提及的链接**：<a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>：未找到描述

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1230783011230847038)** (1003 条消息 🔥🔥🔥): 

- **新用户的导航困惑**：用户对操作 Stable Diffusion 表示困惑，特别是有一位用户按照 YouTube 指南操作但在安装后不知道如何继续。建议尝试 ComfyUI 或查找更多 YouTube 教程，同时澄清 ComfyUI 对于初学者来说可能不是最友好的。

- **AI 发展的飞速步伐**：一位用户表示，由于 Stable Diffusion 的新模型和界面发布速度极快，感到难以招架。其他用户表示赞同，指出 AI 领域正以前所未有的速度进步。

- **技术故障排除**：用户寻求了关于各种问题的帮助，例如在 Kohya 中查找保存的训练状态位置、从 checkpoints 恢复训练以及保存模型状态。建议他们检查输出文件夹以查找保存的状态，并使用特定选项从上次保存的状态恢复。

- **硬件升级咨询**：一位用户询问了关于 VRAM 及其对生成速度的影响，得到的回复是更多的 VRAM 可能允许同时生成图像，并建议在更换新 GPU 后重新安装显卡驱动程序。

- **生成 AI 图像**：新用户寻求关于生成 AI 图像的建议，并被引导至 Bing Image Creator 等平台，以及支持 Stable Diffusion 的本地界面和 Web 服务。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://wallet.bitcoin.com/">加密货币钱包 | 支持 Bitcoin (BTC), Bitcoin Cash (BCH), Ethereum (ETH) 和 ERC-20 代币</a>: 下载 Bitcoin.com 的多币种加密货币钱包。一种简单且安全的方式来购买、出售、交易和使用加密货币。支持 Bitcoin (BTC), Bitcoin Cash (BCH), Ethereum (ETH) 和 ERC-20 代币...</li><li><a href="https://stability.ai/core-models">核心模型 &mdash; Stability AI</a>: 未找到描述</li><li><a href="https://tenor.com/view/sad-gif-7523306793289960933">伤心的 GIF - Sad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://clipdrop.co/stable-diffusion">Clipdrop - Stable Diffusion</a>: AI 图像生成的飞跃</li><li><a href="https://stability.ai/news/stable-diffusion-3-research-paper">Stable Diffusion 3: 研究论文 &mdash; Stability AI</a>: 继我们宣布 Stable Diffusion 3 的早期预览版之后，今天我们发布了研究论文，概述了我们即将发布的模型的详细技术细节，并邀请您 ...</li><li><a href="https://stability.ai/membership">会员资格 &mdash; Stability AI</a>: Stability AI 会员资格通过结合我们的一系列先进开源模型与自托管优势，为您的生成式 AI 需求提供灵活性。</li><li><a href="https://civitai.com/images/10123212">pagartomas880 发布的图片</a>: 未找到描述</li><li><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main">runwayml/stable-diffusion-v1-5 在 main 分支</a>: 未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/j3xHNmEWWCI">⚡利用 ComfyUI PERTURBED 驾驭闪电般的细节 + 🔮 遮罩魔法与时尚秘诀！🤩</a>: -- Discord - https://discord.gg/KJXRzkBM -- 准备好将您的细节处理提升到新的水平！🚀 在这个令人惊叹的教程中，您将发现令人难以置信的...</li><li><a href="https://www.youtube.com/watch?v=ktxbXlF6UQE">揭露在 Discord 中跟踪您的网站！</a>: 有一个名为 spy.pet 的网站，声称在 Discord 上保存了 40 亿条消息。通过它，您可以“查看您的朋友在 Discord 上做什么...</li><li><a href="https://github.com/Stability-AI/stablediffusion">GitHub - Stability-AI/stablediffusion: 使用 Latent Diffusion Models 进行高分辨率图像合成</a>: 使用 Latent Diffusion Models 进行高分辨率图像合成 - Stability-AI/stablediffusion</li><li><a href="https://youtu.be/uWGVlRQjxpM?si=0GcC2yUQ_yn9pPlQ">Alexander Pisteletov : 我是一个新的俄罗斯海盗 (censored) 歌词</a>: Alexander Pisteletov 演唱 "I am a new russian pirate"</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs">在 NVidia GPU 上安装并运行</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://youtu.be/mbhipHCuOEw">如何在 Mac 上设置 Stable Diffusion AI</a>: 我将引导您完成在 Mac M1 或 M2 上本地设置 Stable Diffusion Web UI 的分步过程。🔗 安装指南: https://techxplain...</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。</a>: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://www.youtube.com/watch?v=eD7R3chkRQ0&ab_channel=Paul%27sHardware">早期 RTX 5090 发布糟糕 - 4 月 21 日科技新闻</a>: 早期 RTX 5090 发布糟糕 - 4 月 21 日科技新闻 ▷ 我的商店 - T恤、品脱杯和连帽衫: http://paulshardware.net ⇨ 赞助商: Corsair K65 Plus 无线键盘...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features>">首页</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Ins">首页</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://lykos.ai/downloads">Stability Matrix - 简单的 Stable Diffusion 管理和推理 UI</a>: Stability Matrix - 简单的 Stable Diffusion 管理和推理 UI</li><li><a href="https://civitai.com/models/118811?modelVersionId=128941">带有 SDXL 的 SD1.5 - ComfyUI 工作流（模板） - SD1.5 + SDXL Base | St</a></li>

able Diffusion 其他 | <a href="https://civitai.com/models/133005?modelVersionId=146245">Civitai</a>: SD1.5 + SDXL Base 已经显示出良好的效果。SD1.5 + SDXL Base+Refiner 仅供实验，SD1.5 + SDXL Base - 使用 SDXL 作为构图生成...</li><li><a href="https://mp.weixin.qq.com/s/tz6iKxHQqGfvYWzf_lslRg">如果Sora不开放，我们还能用什么？</a>: 99%的人不知道的免费AI视频工具！好工具值得分享！
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1230841188626595840)** (24 messages🔥): 

- **Colab 在反向传播期间崩溃**：Discord 的一名成员询问了在模型训练期间，特别是在反向传播（Backpropagation）步骤中发生的 **Colab 会话崩溃**问题。其他人回应指出，硬件故障甚至宇宙射线都可能导致此类崩溃。
  
- **Kernel 索引保护风格**：有人询问了 Kernel 函数中“索引保护（Index Guards）”的代码风格，询问为什么 `if (idx < max_idx)` 模式比看起来更清晰的 `if (idx >= max_idx) return;` 更受青睐，另一位成员表达了对后一种方法的偏好。

- **通过 SSH 访问 Nsight Compute GUI**：讨论了如何通过 SSH 在远程机器上访问 **Nsight Compute GUI**，建议包括使用 `ssh -X` 的 X 转发（如 [Teleport 的这份指南](https://goteleport.com/blog/x11-forwarding/)中所述），并参考了 [Nsight Compute 用户指南](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#remote-connections)。

- **用于动态 LLM 推理的 Effort 算法**：提出了一种名为 **Effort** 的新算法，该算法允许动态调整 LLM 推理期间执行的计算。有人表示有兴趣在 Triton 或 CUDA 中实现该算法，该项目可以在 [GitHub](https://github.com/kolinko/effort) 上找到。

- **DGX 机箱中的 NVLink 包含情况**：关于 **DGX 机箱**在出货时是否安装了 **NVLink** 的查询，得到的回复指出它们默认通常使用 SXM 插槽 GPU 和 NVLink。此外还分享了关于 NVLink 的见解，包括来自 [WikiChip](https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/) 的一篇文章。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://kolinko.github.io/effort/">Effort Engine</a>：一种可能用于 LLM 推理的新算法。平滑且实时地调整你在推理过程中想要进行的计算量。</li><li><a href="https://goteleport.com/blog/x11-forwarding/">关于 X11 转发你需要了解的内容</a>：在这篇博文中，我们将深入探讨 X11 转发，解释什么是 X11 以及它的底层工作原理。</li><li><a href="https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#remote-connections">3. Nsight Compute &mdash; NsightCompute 12.4 文档</a>：未找到描述</li><li><a href="https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/">深入了解 Nvidia 的 NVLink 互连和 NVSwitch</a>：深入了解 Nvidia 的 NVLink 互连和拥有 20 亿个晶体管的 NVSwitch，它为 Nvidia 最新的 DGX-2 深度学习机提供动力。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1231376189097119844)** (34 messages🔥): 

- **意外的灰度变换引发关注**：一位用户发现 Triton Kernel 在进行灰度图像变换时出现了意外行为，即将图像缩放回原始大小时产生了奇怪的输出。该问题通过理解大图像的数据存储变化得到了解决；在将数据传递给 Kernel 之前，确保数据在内存中是连续的（Contiguous）非常重要，可以使用 [cuda-mode 的 Triton 工具](https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb)中的 `check_tensors_gpu_ready` 进行验证，并修正了函数中的一个小错误。

- **寻求用于静态码本的 Triton 索引功能**：一位用户询问如何像在 CUDA 中那样在 Triton 中对静态码本（Static Codebooks）进行索引，引发了关于 Triton 缺乏此类功能的讨论。有人强调了一个 [GitHub issue](https://github.com/openai/triton/issues/974#issuecomment-1345372027)，其中可以找到关于 Triton 当前限制以及对二分查找（Binary Search）等功能需求的更多细节。

- **二分查找需求促使 Triton 开发讨论**：在 Triton 中实现二分查找的能力被成员们视为一项重大需求。社区内部（包括 OpenAI 等）似乎对此功能的进一步开发有着浓厚的兴趣，内部正在进行讨论，贡献者们也热衷于支持这一进展。

- **澄清 `make_block_ptr` 中的 `order` 参数**：一位用户请求深入了解 `tl.make_block_ptr()` 中的 `order` 参数，因为该参数在 Flash Attention 的实现中调用方式不一致。另一位用户澄清说，`order` 决定了数据布局的连续性，其中 `(1,0)` 代表行优先顺序（row-major order），而 `(0,1)` 代表列优先顺序（column-major order），这会影响内存的访问方式。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openai/triton/issues/974#issuecomment-1345372027">Index in triton · Issue #974 · openai/triton</a>：我们想在 Triton kernel 中进行一些索引操作，假设我们有 x_ptr, idx_ptr, out_ptr，x = tl.load(x_ptr + offsets, mask = mask)，idx = tl.load(idx_ptr + offsets, mask = mask)，我们有：1. idx = idx.t...</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html#triton.language.make_block_ptr">triton.language.make_block_ptr &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py#L125">triton/python/tutorials/06-fused-attention.py at main · openai/triton</a>：Triton 语言和编译器的开发仓库 - openai/triton</li><li><a href="https://gist.github.com/alexandremuzio/3ba9d8669f57718139da36158180baaf">灰度图的奇怪 Triton kernel 行为（旨在粘贴到带有 T4 GPU 的 Colab 中）</a>：灰度图的奇怪 Triton kernel 行为 - weird_triton_repro.py</li><li><a href="https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb">lectures/lecture 14/A_Practitioners_Guide_to_Triton.ipynb at main · cuda-mode/lectures</a>：cuda-mode 讲座材料。欢迎在 GitHub 上为 cuda-mode/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1230913480731656283)** (9 条消息🔥): 

- **通过 Offset 改进实现合并 (Coalescing)**：一位成员分享道，对操作相邻元素的线程进行对齐可以利用**合并 (Coalescing)** 来提升性能，尽管对该问题的全面理解尚不清晰。
- **通过转变视角找到解决方案**：一次互动促使一位用户找到了解决方案，涉及在流程的计算密集型部分之后进行 Offset 优化，证明了该话题下提供的建议非常有帮助。
- **赞赏布局代数 (Layout Algebra) 演示**：关于“布局代数”的演示因其深刻的**概念基础**而获得赞誉，让参与者看到了“真材实料”。
- **__forceinline 和 __inline 的细微差别**：成员们讨论了在设备代码（device code）中使用 `__forceinline` 和 `__inline` 的情况，认为内联（inlining）可以引导编译器进行更好的优化，减少函数调用并提高访问速度。
- **Nsight Systems 版本问题已解决**：一位在 64 核 CPU 上使用 *Nsight Systems CLI* 遇到问题的用户通过回退到旧版本软件（2023.4.4）找到了解决方案，解决了核心计数不一致的问题。
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1230826048023826473)** (3 条消息): 

- **Triton Hacking 技巧公开**：一位成员分享了 **GitHub - openai/triton** 的链接，其中包含 Hacking Triton 语言和编译器的技巧。位于 [github.com/openai/triton#tips-for-hacking](https://github.com/openai/triton#tips-for-hacking) 的仓库对于解决某些可能与开发任务相关的问题非常有用。

- **主动提供解决方案**：针对提到的问题，一位成员建议应通过之前链接的文档来解决，并表示如果提供的方案不起作用，将提供进一步帮助。

**提到的链接**：<a href="https://github.com/openai/triton#tips-for-hacking">GitHub - openai/triton: Triton 语言和编译器的开发仓库</a>：Triton 语言和编译器的开发仓库 - openai/triton

  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1231317965308170340)** (1 条消息): 

- **CUDA-MODE 讲座公告**：*CUDA-MODE 第 15 讲：Cutlass* 即将开始，由 <@689634697097117750> 主讲。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 条消息): 

andreaskoepf: https://x.com/AliHassaniJr/status/1766108184630943832
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1231145398245265478)** (25 条消息🔥):

- **CUDA Mode 讲座系列启动**：一名成员在 general 频道宣布第 2 讲开始，欢迎那些有兴趣深化对 CUDA 理解的人参加。
- **下一次讲座已排期且讲师受到称赞**：CUDA 系列的第 3 讲定于下周六 GMT 上午 7:30 举行，目前的讲师因其引人入胜且风趣幽默的教学风格而受到好评。
- **确定 CUDA 讲座的实时提问频道**：针对关于在 CUDA Mode 讲座期间何处进行实时提问的问题，读书小组的视频音频频道被指出是实时聊天线程的所在地。
- **寻求矩阵乘法解释**：一位成员请求澄清一段 CUDA 矩阵乘法的代码，引发了另一位成员关于使用 shared memory 优化矩阵乘法的讨论并提供了代码示例。
- **构建机器学习系统 - GPU 选择**：有人咨询在构建机器学习系统时，是使用带有两个 GeForce RTX 2070 的双 GPU 配置，还是使用单个 NVIDIA GeForce RTX 4090 更好。

**提到的链接**：<a href="https://discord.gg/H9h8vKNu">加入 PMPP UI 讲座时区 Discord 服务器！</a>：查看 Discord 上的 PMPP UI 讲座时区社区 - 与其他 28 名成员一起交流，享受免费的语音和文字聊天。

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1231262495864389793)** (2 messages): 

- **偷看前先证明**：Mr.osophy 表示愿意为练习题验证答案，但要求先提供尝试过的证明。为了维护完成练习的完整性，分享了 [Chapter 2](https://docs.google.com/document/d/10ez800eu8OF-OzJXNZ0tRGdJaRAwagiyFdgeBoX0S8o/edit)、[Chapter 3](https://docs.google.com/document/d/1wILXD7Pq8dsvEJpt-YwVekdFxYJvjqu6qnpzYR-LbhE/edit?usp=sharing)、[Chapter 4](https://docs.google.com/document/d/1b29UvSN2-S8D_UP1xvtSB7nFRc86s6AdWH7n5UieDfE/edit?usp=sharing) 和 [Chapter 5](https://docs.google.com/document/d/12_d0PFd3H5o68drT1pv_RuSYo67Evm9X7V70RMplrVk/edit?usp=sharing) 文档以供参考详细练习。

- **Reduction Kernel 困惑**：Chetan9281 正在寻求关于 CUDA reduction kernel 示例中差异的澄清。据该用户称，作者指出循环将执行 7 次，但 chetan9281 的计算表明循环应执行 8 次，请求帮助理解作者主张背后的计算逻辑。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

.bexboy: 我想这一节也会被上传吧？
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1231185461091893269)** (1 messages): 

- **DenseFormer 实现者遇到 JAX 内存挑战**：一位成员正在进行 **JAX 中的 DenseFormer 实现**，并正苦于高内存占用。他们解释说，虽然 *PyTorch* 实现可以高效地原地（in-place）修改张量，但 **JAX 的函数式方法**会导致张量副本，从而增加了内存需求。

- **高效的 PyTorch 技术无法直接迁移到 JAX**：该成员讨论了 DenseFormer 的架构，其中每个 Transformer 块的输入是之前块输出的加权和，由于原地修改，这在 *PyTorch* 中可以高效处理。他们强调 **JAX/XLA 的函数式范式**由于其*写时复制（copy-on-write）*行为使这一过程变得复杂。

- **在 JAX 中追求线性内存占用**：受 [Equinox](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py) 示例的启发，该成员成功创建了一个自定义 JAX 原语，用于处理涉及输入梯度的单次写入缓冲区。然而，在计算相对于 Transformer 块权重的梯度时，内存问题仍然存在，导致内存使用呈二次方增长，而非预期的线性规模。

- **自定义 Backward Pass：一个潜在但复杂的解决方案**：他们认为问题源于 JAX 无法优化梯度的内存占用，建议有必要为整个 loop/scan 函数编写自定义 backward pass。该成员正在寻求管理这种复杂性的高层建议，因为构建自定义 backward pass 将是一项艰巨的任务。

**提到的链接**：<a href="https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py">equinox/equinox/internal/_loop/common.py at main · patrick-kidger/equinox</a>：JAX 中优雅易用的神经网络 + 科学计算。https://docs.kidger.site/equinox/ - patrick-kidger/equinox

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1231258187043438642)** (4 messages):

- **CUDA MODE 线下聚会**：来自 CUDA MODE 社区的成员在德国明斯特（Münster）进行了线下聚会。他们幽默地将其称为德国的“GPU 首都”，庆祝几位成员出人意料地住得很近。
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/)** (1 条消息): 

stygiansonic: 你也可以像这样实现 relu：`z = tl.where(z > 0, z, 0)`
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1230861313069027418)** (12 条消息🔥): 

- **LoRA 的速度代价**：在使用 torchao int4m kernel 进行基准测试时，HQQ+（HQQ 结合 LoRA）出现了约 **20% 的速度下降**，尽管仍有进一步优化的潜力。基础 HQQ 模型在量化时没有进行分组（grouping），有建议认为更好的量化质量可能也有助于提升性能。

- **Kernel 融合壮举**：引入了一个新的融合 `int4 / fp16` Triton kernel，在各种 IO/计算密集型形状的基准测试中表现出色，优于默认的 `hqq.linear` 前向传播。实现细节可以在 [GitHub pull request](https://github.com/pytorch-labs/ao/pull/153) 中找到。

- **转置增强**：讨论了在 qlora 训练期间提升速度的需求，特别关注如何在转置的量化权重矩阵上实现相同的效率。提供了一个完整的示例来展示使用转置时量化在前向和反向传播中的差异：[quantize.py 示例](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283)。

- **反量化的拖累**：对话涉及了 HQQ 过程中必要的反量化步骤所带来的预期性能下降。反量化结合常规的 torch.matmul 操作，与直接使用 fp16/bfp16 的 torch.matmul 相比，会导致大约 15% 的减速。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/ao/pull/153">Fused HQQ Quantization Gemm by jeromeku · Pull Request #153 · pytorch-labs/ao</a>: @msaroufim 融合的 int4 / fp16 量化 Matmul。针对非对称量化权重的融合 gemm。已针对 HQQ 进行测试和基准测试，但理论上可用于任何非对称量化方案。该 ker...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1230796160982057042)** (615 条消息🔥🔥🔥): 

- **Kernel 日志记录与内存优化**：贡献了一个新 kernel，将 `matmul_backward_bias` kernel 的速度显著提升了约 4 倍，并进行了另一项节省内存的更改，将使用量从 14372MiB 降低到 10774MiB，节省了 3598MiB，即原始内存使用量的 25%。讨论了 Dtype 应该是单精度（float）还是混合精度，以及如何在不损失性能的情况下优化从线性到对数操作的归约（reduction）。
- **CUDA 和 cuDNN 的冒险继续**：提交了一个关于 `dev/cuda` 中 cuDNN 前向 Attention 和 FP16 cuBLAS kernel 的 pull request，表明通过避免创建中间张量获得了巨大的性能提升。然而，讨论了 cuBLASLt 和 cuDNN 的一些细微差别，揭示了集成 NVIDIA 库函数以及在混合精度下实现准确训练结果的复杂性。
- **利用 NCCL 探索数据并行**：讨论集中在实现多 GPU 支持的最佳方式上，考虑了单线程多设备、每个设备多个线程或通常由 MPI 管理的多进程设置。共识是采用类似 MPI 的方法，这种方法可以自然地扩展到 8 个以上的 GPU 并支持多机环境。
- **用于训练 LLM 的新数据集**：Thomas Wolf 在 Twitter 上发布的一条消息提到了一项新数据集的发布，引起了人们的兴趣，因为它可能对 GPT-2 复现项目或作为大语言模型的训练集有益。新数据集的受欢迎程度似乎导致 HuggingFace 网站暂时瘫痪。
- **混合精度讨论**：探讨了将混合精度合并到主线代码中的策略。一个 PR 草案展示了一个可以编译但产生错误结果的混合精度实现，指出了在确保数值稳定性的同时调整性能所涉及的复杂性。提到保留一个 FP32 脚本可以提供“地面真值（ground truth）”参考或作为教学资源。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-sin">Examples &mdash; NCCL 2.21.5 documentation</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-single-thread-multiple-devices">Examples &mdash; NCCL 2.21.5 documentation</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/210">Added shared memory for the atomic additions for the layernorm_back by ChrisDryden · Pull Request #210 · karpathy/llm.c</a>: 此 PR 旨在解决在 Profiler 中发现的问题，即该 Kernel 最后循环中的原子操作导致了大量的 Warp Stalls。通过在共享内存上执行原子操作...</li><li><a href="https://github.com/nshepperd/flash_attn_jax/tree/main/csrc/flash_attn/src">flash_attn_jax/csrc/flash_attn/src at main · nshepperd/flash_attn_jax</a>: Flash Attention v2 的 JAX 绑定。通过在 GitHub 上创建账号来为 nshepperd/flash_attn_jax 的开发做出贡献。</li><li><a href="https://clang.llvm.org/doxygen/____clang__cuda__intrinsics_8h_source.html">clang: lib/Headers/__clang_cuda_intrinsics.h Source File</a>: 未找到描述</li><li><a href="https://github.com/karpathy/nanoGPT/blob/master/train.py#L34">nanoGPT/train.py at master · karpathy/nanoGPT</a>: 用于训练/微调中型 GPT 的最简单、最快的仓库。 - karpathy/nanoGPT</li><li><a href="https://github.com/karpathy/llm.c/pull/218">WIP support for FP16/BF16 in train_gpt2.cu (compiles, not correct yet) by ademeure · Pull Request #218 · karpathy/llm.c</a>: 仅供参考并决定这是否是正确的方向（如果不合适可以舍弃）。</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>: 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/issues/212">bug: something goes wrong at larger batch sizes · Issue #212 · karpathy/llm.c</a>: 今天遇到了一个难以追踪的 Bug，打算今晚先休息，明天再试。复现方法：`./train_gpt2cu -b 12` 以 Batch Size 12 启动任务。在 m...</li><li><a href="https://github.com/karpathy/llm.c/pull/213">Custom matmul attention by ngc92 · Pull Request #213 · karpathy/llm.c</a>: 我个人实现的（下三角）矩阵乘法。虽然不如 CuBLAS 高效，但由于我们只计算了一半的数值，因此在净收益上是胜出的。目前还无法摆脱 Permute...</li><li><a href="https://github.com/karpathy/llm.c/pull/221">Faster `matmul_backward_bias` using coalesced reads and shared memory in the kernel by al0vya · Pull Request #221 · karpathy/llm.c</a>: 该 Kernel 在 RTX 2070 Super GPU 上相比 `matmul_backward_bias_kernel2` 似乎有 <4 倍的运行时间提升，运行时间对比见下文：matmul_backward_bias_kernel2: block_size 32 time 0.9...</li><li><a href="https://github.com/karpathy/llm.c/pull/215">cuDNN Forward Attention + FP16 non-cuDNN version in /dev/cuda/ by ademeure · Pull Request #215 · karpathy/llm.c</a>: 之前的 Kernel 4: 1.74ms；使用 TF32 的 Kernel 4: 1.70ms；Kernel 5（带 BF16 I/O 的 Kernel 4）: 0.91ms；Kernel 6（不带 Permute 的 Kernel 5，不现实）: 0.76ms；Kernel 10（cuDNN BF16，带 FP32 转换）: 0.33ms...</li><li><a href="https://github.com/karpathy/llm.c/commit/49d41ae2968ed80d6f9db3d5c96b5a7df1194a7d">add one more kernel, allocating a block per row. bad idea if C is too… · karpathy/llm.c@49d41ae</a>: …再添加一个 Kernel，为每一行分配一个 Block。如果 C 太低，这可能是一个坏主意，正如我们现在的情况。</li><li><a href="https://github.com/karpathy/llm.c/commit/cb791c4ef58d45d58e5af624b0ed41439ac7aeff">new kernel that does a single pass over x on load, using a more cleve… · karpathy/llm.c@cb791c4</a>: …新的 Kernel 在加载时对 x 进行单次处理，使用了更巧妙的方差公式。遗憾的是，在我的 A100 上只快了一点点。</li><li><a href="https://github.com/karpathy/llm.c/commit/8488669d256c59594f486d52a8b3597da7cbfeab">speed up the backward bias kernel by 45% and speed up the full runnin… · karpathy/llm.c@8488669</a>: …将 Backward Bias Kernel 提速 45%，并将总运行时间缩短 1%。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1230955366783782954)** (23 条消息🔥): 

- **测试演示权限**：一名成员请求演示权限，以测试其平板电脑上的屏幕共享功能。另一名成员提议加入他们进行上台测试以提供帮助。

- **新版主角色公告**：创建了一个新的“Moderator”角色，被赋予管理社区的能力，包括禁言（Timeout）、踢出（Kick）、封禁（Ban）和删除消息，以及活动管理和舞台控制。目标是保持 CUDA MODE 服务器成为一个友好且受欢迎的地方。

- **录制前的活动准备**：一名成员请求提前加入活动通话，以确保其录制设置正常运行。在 stage 频道安排了会前检查的协调工作。

- **表达对后续会议的兴趣**：一名成员发布了一个 [Twitter 链接](https://twitter.com/ColmanGlag/status/1781755880783925381)，建议针对之前讨论过的话题进行深入的后续会议。这引发了关于此类活动的潜在兴趣和物流安排的讨论。

- **会议录制的编辑与上传**：两名成员合作编辑并上传了录制的会议材料。一份编辑好的录音已成功编译并通过 Google Drive [链接](https://drive.google.com/file/d/1fEdpmWPYD_Ci4ydbqgTG9ThYxunmUx7e/view?usp=sharing)分享。

**提到的链接**: <a href="https://drive.google.com/file/d/1fEdpmWPYD_Ci4ydbqgTG9ThYxunmUx7e/view?usp=sharing)">lecture-15.mov</a>: 未找到描述

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1230777133840732200)** (653 条消息🔥🔥🔥): 

- **LLaMa-3 Tokenization 问题**：多位用户报告了 LLaMa-3 基础模型微调的问题。用户发现缺失 BOS (beginning of text) token 导致训练期间出现高 loss 和 `grad_norm inf`；链接中提供了一个通过修改 tokenizer 配置的 PR 修复方案 ([修复 BOS token 的 PR](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41))。

- **调试分布式训练**：一些用户遇到了分布式训练问题，包括 Nccl 操作超时错误和端口占用。讨论建议检查 nccl 文档，并可能通过切换端口来解决冲突。

- **Axolotl 数据准备咨询**：寻求了解各种训练任务的自定义数据集结构的用户被引导至 Axolotl 文档，其中提供了预训练、指令微调、对话等示例和格式 ([Axolotl Dataset Formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/))。

- **LLaMa-3 微调的困境**：用户交流了在微调 LLaMa-3 模型时难以达到预期性能的问题。提到的问题包括与早期模型相比性能较差、集成 ChatML token 的问题，以及缺失 bos tokens 对训练的影响。

- **Tokenizer 讨论**：关于 LLaMa-3 庞大的 tokenizer 词表效率和必要性展开了辩论，一些用户主张采用更精简的方法，而另一些用户则强调了该 tokenizer 在将大量文本编码为更少 token 方面的效率。 

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/discussions/11">cognitivecomputations/dolphin-2.9-llama3-8b · Llama 3 Base Is Unique</a>：未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - 数据集格式</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - 指令微调</a>：未找到描述</li><li><a href="https://www.philschmid.de/fsdp-qlora-llama3">使用 PyTorch FSDP 和 Q-Lora 高效微调 Llama 3</a>：了解如何使用 Hugging Face TRL、Transformers、PEFT 和 Datasets，通过 PyTorch FSDP 和 Q-Lora 微调 Llama 3 70b。</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1781050097868103726?t=ow7ldzKTWHjRBW33sxfc_A&s=09">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：@mattshumer_ 我们会推出更长的版本。此外，与 Llama 2 相比，使用新的 tokenizer 后，上下文窗口应该会更长一些。</li><li><a href="https://huggingface.co/dreamgen/opus-v1.2-llama-3-8b">dreamgen/opus-v1.2-llama-3-8b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41">meta-llama/Meta-Llama-3-8B · 更新后处理器以添加 bos</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_train">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/sustcsonglin/flash-linear-attention/tree/main/fla/layers">flash-linear-attention/fla/layers (main 分支) · sustcsonglin/flash-linear-attention</a>：在 PyTorch 和 Triton 中高效实现最先进的线性注意力模型 - sustcsonglin/flash-linear-attention</li><li><a href="https://github.com/Ope">ope - 概览</a>：ope 拥有 11 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/src/axolotl/utils/trainer.py#L272">axolotl/src/axolotl/utils/trainer.py (提交号 0e8f340) · OpenAccess-AI-Collective/axolotl</a>：尽管提问（axolotl questions）。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/xzuyn/axolotl/commit/6488a6b6f0d195612d491ece2f9a049080e8d9">为 ROCm 添加实验性安装指南 · xzuyn/axolotl@6488a6b</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/setup.py#L36">axolotl/setup.py (提交号 0e8f340) · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenNLPLab/lightning-attention/tree/main">GitHub - OpenNLPLab/lightning-attention: Lightning Attention-2：在大型语言模型中处理无限序列长度的免费午餐</a>：Lightning Attention-2：在大型语言模型中处理无限序列长度的免费午餐 - OpenNLPLab/lightning-attention</li><li><a href="https://github.com/lucidrains/memory-efficient-attention-pytorch">GitHub - lucidrains/memory-efficient-attention-pytorch：论文 "Self-attention Does Not Need O(n²) Memory" 中提出的内存高效多头注意力的实现</a>：论文 "Self-attention Does Not Need O(n²) Memory" 中提出的内存高效多头注意力的实现 - lucidrains/memory-efficient-attention-pytorch</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1519">考虑将 Memory Efficient Attention 作为 AMD 用户 Flash Attention 的“替代方案”。· Issue #1519 · OpenAccess-AI-Collective/axolotl</a>：⚠️ 请检查此功能请求之前是否已被提出。我搜索了讨论区之前的 Ideas，没有发现类似的功能请求。我搜索了之前的 Issues...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/0e8f3409451442950f2debbe28735198361c9786/src/axolotl/monkeypatch/llama_attn_hijack_flash.py#L30">axolotl/src/axolotl/monkeypatch/llama_attn_hijack_flash.py (提交号 0e8f340) · OpenAccess-AI-Collective/axolotl</a>：尽管提问。为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li>

通过在 GitHub 上创建账户来参与 OpenAccess-AI-Collective/axolotl 的开发。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549">Draft: Update Tokenizer Overrides Handling in models.py by mhenrichsen · Pull Request #1549 · OpenAccess-AI-Collective/axolotl</a>: 示例：tokenizer_overrides:   - 28006: &lt;|im_start|&gt;   - 28007: &lt;|im_end|&gt;  描述：此 PR 增强了我们在 models.py 文件中处理 tokenizer overrides 的方式。...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: 描述、动机和背景、如何测试？未测试！屏幕截图（如果适用）、变更类型、社交账号（可选）</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/requirements.txt">axolotl/requirements.txt at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 axolotl 问题。通过在 GitHub 上创建账户来参与 OpenAccess-AI-Collective/axolotl 的开发。</li><li><a href="https://github.com/xzuyn/axolotl/">GitHub - xzuyn/axolotl: Go ahead and axolotl questions</a>: 尽管提出 axolotl 问题。通过在 GitHub 上创建账户来参与 xzuyn/axolotl 的开发。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1230979502297976962)** (16 messages🔥): 

- **寻求测试用的计算资源**：一名成员分享了一个[草案 PR 链接](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549)，用于更新 `models.py` 中的 tokenizer overrides 处理，并请求闲置算力来测试此 Pull Request。
- **PyTorch 中融合操作的特性请求**：[PyTorch issue #124480](https://github.com/pytorch/pytorch/issues/124480) 详细说明了一个关于融合线性层和交叉熵损失函数（fused linear and cross-entropy loss function）的特性请求，以高效处理大型 logits。
- **理解 LLM 中的 VRAM 消耗**：一名成员解释了 Llama 3 等近期 LLM 中大词表量对 VRAM 的影响，并给出了明细：在使用 batch size "81920 tokens" 时，logits 大小为 "19.57GiB"，隐藏状态（hidden state）大小为 "20GiB"。
- **Batch Size 澄清**：针对 batch size 可能存在的拼写错误，该成员澄清所提供的统计数据是基于 "batch size 10, seq len 8192"。
- **fsdp 与 8-bit 优化器的挑战**：讨论了 `fsdp` (Fully Sharded Data Parallel) 与 Fast Fourier Transforms (FFT) 及 8-bit 优化器的兼容性，一名成员遇到了 `fsdp` 卡死的问题，另一名成员提到 `adamw_torch` 消耗了大量 VRAM。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1549">Draft: Update Tokenizer Overrides Handling in models.py by mhenrichsen · Pull Request #1549 · OpenAccess-AI-Collective/axolotl</a>: 示例：tokenizer_overrides:   - 28006: &lt;|im_start|&gt;   - 28007: &lt;|im_end|&gt;  描述：此 PR 增强了我们在 models.py 文件中处理 tokenizer overrides 的方式。...</li><li><a href="https://github.com/pytorch/pytorch/issues/124480">Fused Linear and Cross-Entropy Loss `torch.nn.functional.linear_cross_entropy` · Issue #124480 · pytorch/pytorch</a>: 🚀 特性、动机和设想。如果 PyTorch 能有一个融合线性层和交叉熵的函数（例如 torch.nn.functional.linear_cross_entropy）就太棒了。该函数的作用是融合...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1230781097269530705)** (22 messages🔥): 

- **Llama3 微调异常引起关注**：一位用户在尝试微调 **Llama3** 时遇到错误 (**RuntimeError**)，但 Mistral 和 Llama2 等其他模型微调正常。Traceback 显示这并非内存或空间问题，因为其他模型在同一目录下保存成功。

- **用户寻求微调资源**：有人请求为特定领域用例微调 embedding 模型的资源，但讨论中未提供具体的资源或指导。

- **FSDP 配合 FFT 的咨询**：有人提问关于在使用 Fully Sharded Data Parallel (FSDP) 时执行 Fast Fourier Transform (FFT) 的问题，一位用户确认这是可行的，但未提供细节或示例配置。

- **对大模型量化配置的关注**：一位用户询问了 **70B 参数模型使用的量化配置**，被引导查看 `examples/quantize.py` 对应的 `config.json` 文件以获取默认配置。

- **模型合并等待时间过长**：有用户对 **带有 Lora 微调的 70B 参数模型** 的漫长合并时间表示担忧；另一位用户的回复指出，所经历的时间似乎比预期的要长，但未给出明确的时间范围。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1230809963832807424)** (37 messages🔥): 

- **Runpod 加载缓慢**：成员们在 **Runpod** 上启动 Pod 时遇到延迟，指出其“耗时极长”或根本无法加载。

- **提供上传限制的解决方法**：一位用户遇到了 **上传限制** 问题，并通过使用 `huggingface_hub` 库手动将文件夹上传到 Hugging Face Spaces 解决了该问题，并提供了示例代码。

- **在 Runpod 中通过命令行管理 VRAM**：为了实时监控 VRAM，建议使用 `nvidia-smi` 等 **命令行工具**，因为 **Runpod dashboard** 不会实时更新显存使用情况。

- **探索多个终端窗口**：成员们讨论了如何同时运行 **Axolotl** 和其他命令，考虑了 SSH、多个 Web 终端或 **Jupyter notebooks** 等选项。

- **CPU 内存报告差异**：一位用户指出了 **Runpod CPU 内存** 报告的不一致性，界面显示有 48GB RAM 可用，而 `nvitop` 显示已使用 76GB。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1231356634819854396)** (22 messages🔥): 

- **YAML 键值用法澄清**：针对数据集 **config YAML** 文件中的 `"conversation:"` 键提出了疑问。澄清指出，该键指定了用于训练 AI 模型的对话数据集结构和格式，例如角色以及如何识别对话数据。

- **对话式 AI 的数据集配置**：解释了在 YAML 文件中指定 `"type: sharegpt"` 表示使用 ShareGPT 格式的数据，而 `"conversation: chatml"` 则表示需要将数据转换为 ChatML 格式，从而通过正确格式化的数据促进有效的模型训练。

- **技术故障排除**：一位成员分享了在分布式计算过程中出现多个 `SIGBUS` 错误的错误日志。回复概述了潜在原因，如内存对齐问题、内存映射文件问题或硬件故障，并提供了故障排除步骤。

- **使用 Unsloth 优化训练**：有用户请求在 Axolotl 中使用 **Unsloth** 的指令，详细回复提供了一份分步指南，包括安装依赖项、准备模型和数据、在训练脚本中配置 Unsloth 以及监控训练结果。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e4ffa5d8-9095-4a00-8773-02132978f2e7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4eadad10-1146-45ad-9822-155e9b87cb48)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e7301808-4b94-41b9-b3d4-752db98cf71f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=33a203be-00f7-40dc-9fa2-e911b904e980)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1230856587401367673)** (326 messages🔥🔥): 

- **智能手机上的本地 AI**：
  用户讨论了在三星 S24 Ultra 等智能手机上本地运行 **Llama 3** 等语言模型的可行性。性能结果各异，据报告在 S24 Ultra 上使用量化 LLM 的速度为 **4.3 tok/s**，而在 S23 Ultra 上为 **2.2 tok/s**。

- **Self-Attention 之谜**：
  一场激烈的技术辩论围绕着为什么 Transformer 模型的 Self-Attention 机制中的 Token 会关注自身的 Key-Value 展开，并就消融该机制的潜在实验及其对性能的影响提出了建议。不同用户表达了各自的理解，范围从表达能力到刻画 Token 身份不等。

- **Hugging Face 的商业模式受到质疑**：
  人们对 **Hugging Face 的业务和托管模式** 持怀疑态度，特别是涉及到在没有明显盈利策略的情况下提供大文件服务。一些人将其与 GitHub 的模式进行了比较，并思考了两者之间的差异。

- **GPT 推理研究与基准测试 (Benchmarks)**：
  一位用户询问了用于评估 LLM 推理能力的指标，并表示大多数文献都集中在 Chain of Thought (CoT) 方法上。另一位用户回应称，在当前的 LLM 领域内，更深层次的非 CoT 推理研究非常稀缺。

- **用于提升训练稳定性的 Stable AdamW**：
  在关于 **Whisper architecture** 相关模型训练不稳定性的详细交流中，有人建议尝试使用 **StableAdamW**，以期在常规 AdamW 配合梯度裁剪 (gradient clipping) 的基础上获得改进。讨论内容包括调整学习率 (learning rates)、beta 值以及使用梯度直方图 (gradient histograms) 进行调试的具体细节。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=37248895">未找到标题</a>: 未找到描述</li><li><a href="https://store.google.com/intl/en/ideas/articles/pixel-feature-drop-december-2023/">Gemini Nano 现已在 Pixel 8 Pro 上运行 —— 首款内置 AI 的智能手机</a>: Gemini 来了，这是我们迄今为止功能最强大、最灵活的 AI 模型。此外，Pixel 系列还将迎来更多 AI 更新。</li><li><a href="https://developers.googleblog.com/2024/03/running-large-language-models-on-device-with-mediapipe-andtensorflow-lite.html">使用 MediaPipe 和 TensorFlow Lite 在设备端运行 Large Language Models - Google for Developers</a>: 未找到描述</li><li><a href="https://llm.mlc.ai/docs/deploy/android.html">Android App &mdash; mlc-llm 0.1.0 文档</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2311.10207">Stella Nera：通过基于近似矩阵乘法的无乘法器 DNN 加速实现 161 TOp/s/W</a>: 从经典 HPC 到深度学习，MatMul 是当今计算的核心。最近的 Maddness 方法通过使用基于哈希的版本来近似 MatMul，而无需进行乘法运算...</li><li><a href="https://x.com/giffmana/status/1692641748445438301>)">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: 结束前的两个小贴士：左图：如果你的 loss 飙升，尝试将 Adam/AdaFactor 的 beta2 降低到 0.95（并非新奇，但很少被分享）；右图：当模型的一部分是预训练的，但...</li><li><a href="https://nanoreview.net/en/soc/samsung-exynos-2400">Samsung Exynos 2400：规格和基准测试</a>: Samsung Exynos 2400：基准测试中的性能测试（AnTuTu 10, GeekBench 6）。电池续航和完整规格。</li><li><a href="https://play.google.com/store/apps/details?id=us.valkon.privateai&hl=en&gl=US">Private AI - Google Play 应用</a>: 未找到描述</li><li><a href="https://support.google.com/googleplay/android-developer/answer/9878810?hl=en-GB#>">不当内容 - Play Console 帮助</a>: 未找到描述</li><li><a href="https://apps.apple.com/us/app/mlc-chat/id6448482937?platform=iphone">‎MLC Chat</a>: ‎MLC Chat 让用户可以在 iPad 和 iPhone 上本地与开源语言模型聊天。模型下载到应用后，一切都在本地运行，无需服务器支持，且无需互联网即可工作...</li><li><a href="https://github.com/EleutherAI/aria-amt/blob/0394a05aa57e5d4f7b059abbfed3a028732b243a/amt/train.py#L330">aria-amt/amt/train.py 位于 EleutherAI/aria-amt</a>: 高效且稳健的 seq-to-seq 自动钢琴转谱实现。- EleutherAI/aria-amt</li><li><a href="https://github.com/mlc-ai/mlc-llm">GitHub - mlc-ai/mlc-llm：让每个人都能在自己的设备上原生开发、优化和部署 AI 模型。</a>: 让每个人都能在自己的设备上原生开发、优化和部署 AI 模型。- mlc-ai/mlc-llm</li><li><a href="https://semiconductor.samsung.com/dram/lpddr/lpddr5/">LPDDR5 | DRAM | Samsung 半导体全球</a>: 了解 LPDDR5，它以 6,400 Mbps 的引脚速度、51.2Gb/s 的海量传输和 20% 的节能效果，为下一代应用提供性能和效率支持。</li><li><a href="https://github.com/atfortes/Awesome-LLM-Reasoning?tab=readme-ov-file">GitHub - atfortes/Awesome-LLM-Reasoning：Large Language Models 中的推理：论文和资源，包括 Chain-of-Thought、Instruction-Tuning 和多模态。</a>: Large Language Models 中的推理：论文和资源，包括 Chain-of-Thought、Instruction-Tuning 和多模态。 - GitHub - atfortes/Awesome-LLM-Reasoning...</li><li><a href="https://github.com/Kotlin/Kotlindl">GitHub - Kotlin/kotlindl：受 Keras 启发，用 Kotlin 编写的高级深度学习框架</a>: 受 Keras 启发，用 Kotlin 编写的高级深度学习框架 - Kotlin/kotlindl</li><li><a href="https://www.gsmarena.com/samsung_galaxy_s24_ultra-review-2670p4.php">Samsung Galaxy S24 Ultra 评测</a>: Samsung 的 S24 系列搭载了基于 Google 最新 Android 14 的最新 One UI 6.1。尽管 ".1" 的编号更新幅度较小，...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1230874581779087453)** (293 条消息🔥🔥):

```html
<ul>
  <li><strong>关于 "Megalodon" 架构优越性的辩论</strong>：讨论涉及 <strong>Megalodon</strong>，这是来自 Meta 的一种新架构，以长上下文（long contexts）效率著称，在受控测试中被指出优于 Llama-2。关于它与其他混合注意力机制（hybrid attention mechanisms）的对比以及其潜在的广泛认可度，仍存在怀疑。</li>
  <li><strong>探索用于模型引导的任务向量（Task Vectors）</strong>：提出了一种名为 <strong>task vectors</strong> 的方法来引导预训练模型的行为，允许通过取反和加法等算术运算进行修改。这可以在不直接进行 fine-tuning 的情况下，为 Llama3 等模型添加专业知识（参考 <a href="https://arxiv.org/abs/2212.04089">arXiv:2212.04089</a>）。</li>
  <li><strong>提议针对 RAG 模型的新基准测试</strong>：<strong>Stella Athena</strong> 分享了一个针对检索增强生成（RAG）模型的基准测试想法，其中问题需要综合多个文档的信息。由于在选择常见训练集中存在的来源时可能存在数据集污染，这一挑战非常重大。</li>
  <li><strong>推理过程中的注意力机制近似</strong>：<strong>Carson Poole</strong> 关于在推理过程中通过近似注意力机制来压缩 token 长度的询问，引发了对几篇论文的引用（例如 <a href="https://arxiv.org/abs/2401.03462">arXiv:2401.03462</a>, <a href="https://arxiv.org/abs/2401.06104">arXiv:2401.06104</a>），这些论文讨论了 Activation Beacon、TOVA 和动态 FLOPs 分配等相关概念。</li>
  <li><strong>Transformer 上下文扩展的潜力与局限</strong>：出现了一场关于扩展 Transformer 上下文长度可行性的讨论，提到了 Gemini Pro 1.5 的上下文长度以及二次方计算缩放（quadratic compute scaling）的挑战，强调巨大的上下文长度（例如 1000 万个 token）可能预示着一种超越简单上下文长度 fine-tuning 的架构。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
```

<li><a href="https://x.com/BlancheMinerva/status/1782437494585282965">来自 Stella Biderman (@BlancheMinerva) 的推文</a>：为 RAG 模型创建一个基准测试，其中所有问题都需要综合多个文档的信息才能回答。研究在公开数据上训练的模型在该基准上的表现，并且 ...</li><li><a href="https://arxiv.org/abs/2212.04089">Editing Models with Task Arithmetic</a>：改变预训练模型的行为——例如，提高其在下游任务上的性能或减轻预训练期间学到的偏见——是开发机器学习模型时的常见做法...</li><li><a href="http://arxiv.org/abs/2401.06104">Transformers are Multi-State RNNs</a>：Transformer 被认为在概念上与上一代最先进的 NLP 模型——循环神经网络 (RNNs) 不同。在这项工作中，我们证明了 decoder-only...</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>：语言建模的最新进展在于在极大的网络挖掘文本语料库上预训练高度参数化的神经网络。在实践中，训练和推理此类模型的成本可能很高...</li><li><a href="https://arxiv.org/abs/2404.08698">Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding</a>：虽然 Large Language Models (LLMs) 展示了卓越的能力，但由于自回归处理，它们受到显著的资源消耗和相当大的延迟的阻碍。在这项研究中，我们...</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>：Large language models (LLMs)，如 GPT4 和 LLaMA，由于其强大的文本编码/解码能力和新发现的涌现能力，正在自然语言处理领域取得重大进展...</li><li><a href="http://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算量）分配给特定的...</li><li><a href="https://arxiv.org/abs/2404.07982">Language Imbalance Can Boost Cross-lingual Generalisation</a>：多语言能力对于将语言建模的最新进展扩展到不同的语言社区至关重要。为了在代表多种语言的同时保持高性能，多语言模型...</li><li><a href="https://tenor.com/view/sisihae-gif-23689236">Sisihae GIF - Sisihae - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/krafton-ai/mambaformer-icl">GitHub - krafton-ai/mambaformer-icl: MambaFormer in-context learning experiments and implementation for https://arxiv.org/abs/2402.04248</a>：MambaFormer in-context learning 实验和实现，针对 https://arxiv.org/abs/2402.04248 - krafton-ai/mambaformer-icl</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>：Foundation models 已成为各种人工智能应用中的关键组件，并在自然语言处理和其他几个领域展示了显著的成功。许多...</li><li><a href="https://github.com/meta-llama/llama3/issues/39#issuecomment-2065718050">列出 Llama 3 的 &quot;公开可用来源&quot; 15T 数据集列表 · Issue #39 · meta-llama/llama3</a>：如果没有数据集来源列表，Llama 3 在任何有意义的程度上都是不可复现的。请发布来源列表。</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>：由于上下文窗口大小有限，长上下文的利用对 LLMs 提出了巨大挑战。虽然可以通过微调来扩展上下文窗口，但这会导致相当大的...</li><li><a href="https://arxiv.org/abs/2403.11901">Larimar: Large Language Models with Episodic Memory Control</a>：高效且准确地更新 Large Language Models (LLMs) 中存储的知识是当今最紧迫的研究挑战之一。本文介绍了 Larimar——一种新颖的、受大脑启发的架构...</li><li><a href="https://github.com/naver-ai/rdnet">GitHub - naver-ai/rdnet</a>：为 naver-ai/rdnet 的开发做出贡献。</li><li><a href="https://arxiv.org/html/2402.08164v1">On Limitations of the Transformer Architecture</a>：未找到描述</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: To speed up LLMs&#39; inference and enhance LLM&#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.</a>：为了加速 LLMs 的推理并增强 LLM 对关键信息的感知，压缩 prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。</li>

感知关键信息，压缩 prompt 和 KV-Cache，在性能损失极小的情况下实现了高达 20 倍的压缩。 - GitH...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1230846790912446545)** (47 messages🔥): 

- **Chinchilla 复现辩论升温**：一场关于复现 Chinchilla 研究的讨论展开，引用了参数化建模中可能存在的缺陷和不稳定性，该内容最初分享在 [Twitter](https://twitter.com/kyo_takano/status/1781286971522080919) 上，随后引发了关于 Chinchilla 论文中数值舍入问题的辩论。

- **TeX 文件挖掘揭示数据彩蛋**：成员们讨论了深入研究 arXiv 论文 TeX 源文件的实用性，指出源文件包含精确的数据值，有时还包含像彩蛋一样的隐藏内容；源文件可以通过 arXiv 上的 "other formats" 选项公开访问。

- **Twitter 拉黑引发关于沟通风格的讨论**：一名成员表达了在评论 Chinchilla 复现尝试后在 Twitter 上被拉黑的挫败感。这引发了关于批评性交流时语气重要性的对话，并有建议认为，帖子中表现出的粗鲁或缺乏“神经典型式社交修饰 (neurotypical decoration)”可能会导致误解。

- **复现主张中残差的深入分析**：对话强调，评估 Chinchilla 复现尝试的关键不仅在于残差的非中心性，还在于使用未舍入精度的重新评估，这表明原始模型并没有欠拟合。

- **舍入担忧得到澄清**：澄清指出，复现辩论中提到的数据点舍入归因于原始 Chinchilla 论文的作者，而非复现团队，这涉及 TeX 源码和 Chinchilla 报告的结果。

**提到的链接**：<a href="https://x.com/kyo_takano/status/1782100341443666282))">Kyo (@kyo_takano) 的推文</a>：你正在舍入原始估计值，哈哈。试着像检查 PDF 插图一样检查 TeX 源码。具体来说，你舍入了：- E 从 exp(0.5267228) 舍入到 1.69 - A 从 exp(6.0073404) 舍入到 406.4 ...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1230903433846980769)** (2 messages): 

- **DeepMind 机械可解释性团队更新**：Google DeepMind 的机械可解释性 (Mechanistic Interpretability) 团队分享了一份进展更新，重点关注 **Sparse Autoencoders (SAEs)** 的各种进展。Neel Nanda 在推文中发布的更新包括使用大型模型和 JAX 的基础设施经验，以及探索转向向量 (steering vectors)、推理时稀疏近似算法和 ghost gradients 的改进。[Twitter 链接](https://twitter.com/NeelNanda5/status/1781400080802779604) 和 [博客文章](https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/)。
- **DeepMind 博客详述可解释性工作**：博客透露，所展示的工作通常被认为对于正式论文来说还太初级，包含了初步尝试、记录、复现和对机械可解释性从业者有价值的负面结果。团队列出了两个主要目标：将 SAEs 扩展到更大的模型，以及推进关于 SAEs 的基础科学研究。

**提到的链接**：<a href="https://www.alignmentforum.org/posts/HpAr8k74mW4ivCvCu/progress-update-from-the-gdm-mech-interp-team-summary">[摘要] 来自 GDM 机械可解释性团队的进展更新 #1 — AI Alignment Forum</a>：简介：这是来自 Google DeepMind 机械可解释性团队的进展更新，灵感来自 Anthropic 团队出色的每月更新……

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1230783916609372200)** (5 messages): 

- **渴望对比数据**：一位用户分享了一个包含测试结果的 **Google Spreadsheet**，并询问基准 MMLU 分数，表示有兴趣看到对比。提供的链接是 [MMLU - Alternative Prompts](https://docs.google.com/spreadsheets/d/1luIEdZ_gH2GpFY9iLtM20oXemN6xaBzuGGkJAxQh-R0/edit?usp=sharing)。
  
- **寻求 lm-evaluation-harness 贡献指导**：一位贡献者在运行 **lm-evaluation-harness** 的单元测试时寻求帮助，参考了一份过时的 [贡献文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/CONTRIBUTING.md)。他们注意到缺少测试命令目录，并且依赖于各种可选的 extras 包。

**Link mentioned**: <a href="https://docs.google.com/spreadsheets/d/1luIEdZ_gH2GpFY9iLtM20oXemN6xaBzuGGkJAxQh-R0/edit?usp=sharing">MMLU - Alternative Prompts</a>: MMLU (提示词变体) 示例输入提示词 输入提示词，格式 01,{{question.strip}} 02,Q: {{question.strip}}\nA: 03,Question: {{question.strip}}\nAnswer: Llama-2-7b-hf,Mistral-7B-v0.1,falcon-7b,py...

---

**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1230897591420588144)** (77 messages🔥🔥): 

- **改进 Mojo 接口**：参与者询问了 Mojo 未来接口的情况，以使其更简单，类似于调用标准的 C/C++ 函数；讨论内容包括与其他语言的集成，其中提到了 [MLX-Swift](https://github.com/ml-explore/mlx-swift) 作为 Swift 与 Mojo 交互的示例。

- **路线图与设计决策**：分享了一份 [Mojo 路线图文档](https://docs.modular.com/mojo/roadmap#cc-interop)，详细说明了设计决策，并提供了该语言开发优先级的全景视图，包括核心系统编程特性。

- **创建 Mojo 模块及文档讨论**：提供了关于创建 Mojo 模块和打包的指导，并讨论了 Mojo 的自动化文档代码是否公开或是否可以开源。

- **Mojo 的性能挑战**：讨论了一个[已知问题](https://github.com/modularml/mojo/issues/975)，即 Mojo 的性能比 Python 慢，具体原因是 Mojo 缺乏缓冲 IO；还分享了一篇包含基准测试技巧的博文。

- **Max Serving 框架与 Mojo**：提出了关于如何将 MAX serving 框架与原生 Mojo 编写的神经网络配合使用的问题；提到了纯 Mojo 机器学习框架 [Basalt](https://github.com/basalt-org/basalt)，涉及其未来的兼容性以及直接接口交互的愿景。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/975):">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>: 一个从零开始的纯 Mojo 机器学习框架 🔥 - basalt-org/basalt</li><li><a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: 我们的 Mojo 计划摘要，包括即将推出的特性和需要修复的问题。</li><li><a href="https://github.com/ml-explore/mlx-swift">GitHub - ml-explore/mlx-swift: Swift API for MLX</a>: MLX 的 Swift API。通过在 GitHub 上创建账号为 ml-explore/mlx-swift 的开发做出贡献。</li><li><a href="https://news.ycombinator.com/item?id=40107007">Penzai: JAX research toolkit for building, editing, and visualizing neural nets | Hacker News</a>: 无描述信息
</li>
</ul>

</div>

---

**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1230988200206401829)** (7 messages): 

- **Modular 发布预热推文**：Modular 通过一条 [Twitter 帖子](https://twitter.com/Modular/status/1781426483149602820)分享了即将推出的功能或活动的预告，引发了关注者的好奇。
- **未来先睹为快**：Modular 的[另一条推文](https://twitter.com/Modular/status/1782457222511161545)暗示了未来的发展，向社区预告了可能的新进展或发布。
- **Modular 营造期待感**：Modular 的[后续推文](https://twitter.com/Modular/status/1782457235454689752)保持了势头，这种铺垫暗示即将发布公告或启动。
- **公告倒计时继续**：Modular 在该系列中又发布了[一条推文](https://twitter.com/Modular/status/1782457253829935500)，提高了人们对重大更新或揭晓的期望。
- **Modular 激发兴奋感**：Modular 在[最新推文](https://twitter.com/Modular/status/1782457261652312486)中持续的预热活动让粉丝们充满期待。
- **拼图的另一块**：Modular 通过[一条新推文](https://twitter.com/Modular/status/1782457268354809918)增加了悬念，可能暗示了其正在展开的叙事中即将发生的事情。
- **预热系列继续**：Modular 的一系列[预热推文](https://twitter.com/Modular/status/1782457275078316384)表明正在构建一个故事情节或一系列通向重大揭晓的步骤。

---

**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1231629012510703649)** (1 messages):

- **寻求 AI 演进视频的互动**：一位成员分享了一个名为 [“The Rise of AI” 的 YouTube 视频](https://youtube.com/watch?v=SfKGHKzkm-o)，这是作为一个大学作业创建的，请求在评论中点赞和反馈以展示互动率。他们承认由于只有一周的准备时间，内容可能比较浅显，并请求对其非母语英语的理解。

**提及的链接**：<a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>：(开启字幕)(Turn on the Closed Caption) 加入我们的旅程，探索人工智能的快速演进，从其出现开始...

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1230781600183222323)** (279 条消息🔥🔥)：

- **分享 MLIR 资源**：对于那些询问的人，MLIR (Multi-Level Intermediate Representation) 文档可以在 [MLIR 官方网站](https://mlir.llvm.org/)上找到，2023 年 LLVM 开发者大会提供了一个 [YouTube 视频](https://youtu.be/lXAp6ZAWyBY?si=OSuCzPUmuohgUYvL)，驳斥了关于 MLIR 的常见误解。

- **寻找基础类型列表**：一位成员请求一份 Mojo 基础/原始类型和语言关键字的完整列表。有人指出，数值数据类型在 SIMD 别名下可用，且目前似乎没有保留关键字页面，尽管 Python 关键字预计会被保留，并增加了 `inout`、`borrowed`、`owned` 和 `alias`。

- **以 Python 作为起点**：建议编程新手如果目前的电脑运行 Mojo 不够快，可以先从 Python 开始，因为它更成熟，以后随时可以学习 Mojo。

- **探索 SIMD 类型转换**：成员们讨论了在不保存到内存的情况下将 SIMD 向量转换为不同类型的各种方法，其中 `memory.bitcast` 被认为是一个很有前景的选择。

- **Mojo 框架的潜力**：发起了一场关于为 Mojo 构建框架的讨论，Web 服务是一个关注点。有人提到，最终 Python 框架可能可以在 Mojo 中使用，这类似于 JavaScript 库可以与 TypeScript 一起使用，或者 C 库可以与 C++ 一起使用。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/algorithm/sort#partition">sort | Modular 文档</a>: 实现排序函数。</li><li><a href="https://doc.rust-lang.org/rust-by-example/testing/unit_testing.html">单元测试 - Rust By Example</a>: 未找到描述</li><li><a href="https://www.arewewebyet.org/">Are we web yet? 是的，而且快得惊人！ </a>: 未找到描述</li><li><a href="https://docs.pytest.org/en/8.0.x/index.html">pytest: 帮助你编写更好的程序 — pytest 文档</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/collections/">collections | Modular 文档</a>: 实现 collections 包。</li><li><a href="https://tenor.com/view/ron-swanson-parks-and-rec-its-so-beautiful-gif-15644547">Ron Swanson Parks And Rec GIF - Ron Swanson Parks And Rec Its So Beautiful - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/the-office-andy-andy-bernard-thought-about-it-im-in-gif-16547652">The Office Andy GIF - The Office Andy Andy Bernard - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/simd">simd | Modular 文档</a>: 实现 SIMD 结构。</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/unsafe#bitcast-2">unsafe | Modular 文档</a>: 实现用于处理不安全指针的类。</li><li><a href="https://github.com/modularml/mojo/issues/2113)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/toiletsandpaper/mojo_zlib_classification/blob/master/tools/utils.mojo">mojo_zlib_classification/tools/utils.mojo at master · toiletsandpaper/mojo_zlib_classification</a>: 通过在 GitHub 上创建账号来为 toiletsandpaper/mojo_zlib_classification 的开发做出贡献。</li><li><a href="https://github.com/thatstoast">thatstoast - 概览</a>: GitHub 是 thatstoast 构建软件的地方。</li><li><a href="https://github.com/modularml/mojo/issues/2197">[功能请求] `.__doc__` 属性 · Issue #2197 · modularml/mojo</a>: 查看 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？我希望能获取我的字符串的 docstring...</li><li><a href="https://github.com/modularml/mojo/issues/2164)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/Moosems/Mojo-UI/blob/main/.github/workflows/package.yml">Mojo-UI/.github/workflows/package.yml at main · Moosems/Mojo-UI</a>: 一个用于 Mojo 的跨平台 GUI 库。通过在 GitHub 上创建账号来为 Moosems/Mojo-UI 的开发做出贡献。</li><li><a href="https://github.com/Moosems/Mojo-UI/blob/main/download_dependencies.sh">Mojo-UI/download_dependencies.sh at main · Moosems/Mojo-UI</a>: 一个用于 Mojo 的跨平台 GUI 库。通过在 GitHub 上创建账号来为 Moosems/Mojo-UI 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/mist">GitHub - thatstoasty/mist: 为你的终端应用程序提供高级 ANSI 样式和颜色支持</a>: 为你的终端应用程序提供高级 ANSI 样式和颜色支持 - thatstoasty/mist</li><li><a href="https://mlir.llvm.org/">MLIR</a>: 未找到描述</li><li><a href="https://youtu.be/lXAp6ZAWyBY?si=OSuCzPUmuohgUYvL">2023 LLVM 开发者大会 - MLIR 不是 ML 编译器，以及其他常见误区</a>: 2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------MLIR 不是 ML 编译器，以及其他常见误区。演讲者：Alex Zinenko------幻灯片...</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[提案] Mojo 项目清单和构建工具 · modularml/mojo · Discussion #1785</a>: 大家好，请查看关于 Mojo 项目清单和构建工具的提案。正如提案本身所述，我们希望听到来自 Mojo 社区的声音：你是否同意这些动机...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1231289596415836291)** (19 条消息🔥):

- **寻找 Llama 爱好者**：成员们表达了对构建一个以 🦙🦙🦙.🔥 为象征的项目的兴趣，可能涉及名为 'Llama' 的机器人或项目的新迭代。
- **以文绘图**：一位成员指出，可以使用书面文本作为 Prompt 来创作插图。
- **ModularBot 成就**：ModularBot 宣布一名用户晋升到新等级，展示了聊天中的游戏化功能。
- **渴望新兴工具**：
  用户们分享了对使用 HTMX 和 JSON 集成进行开发的兴奋之情。一位用户提到了 JSON 工具的进展，而另一位用户被鼓励在准备就绪后与社区分享他们的工作。
- **JSON 反序列化挑战**：一位用户讨论了 JSON 反序列化面临的挑战，原因是目前缺乏 Read 或 Write trait，且 trait 中缺少关联类型（associated types），这阻碍了构建可组合的解决方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>：一个从零开始用纯 Mojo 编写的机器学习框架 🔥 - basalt-org/basalt</li><li><a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: Mojo CLI Library modeled after Cobra.</a>：模仿 Cobra 的 Mojo CLI 库。欢迎在 GitHub 上为 thatstoasty/prism 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/mog">GitHub - thatstoasty/mog: Style definitions for nice terminal layouts.</a>：用于美化终端布局的样式定义。欢迎在 GitHub 上为 thatstoasty/mog 的开发做出贡献。</li><li><a href="https://github.com/thatstoasty/gojo">GitHub - thatstoasty/gojo: Experiments in porting over Golang stdlib into Mojo.</a>：将 Golang 标准库移植到 Mojo 的实验。 - thatstoasty/gojo</li><li><a href="https://github.com/thatstoasty/termios">GitHub - thatstoasty/termios: Mojo termios via libc</a>：通过 libc 实现的 Mojo termios。欢迎在 GitHub 上为 thatstoasty/termios 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1230901310006755388)** (5 条消息): 

- **前缀和计算中的性能之谜**：一位成员分享了一项性能对比，尽管启用了硬件优化，Rust 的前缀和计算仍比 Mojo 慢 6 倍。在运行测试后，Rust 在没有特定硬件标志的情况下，处理每个元素的时间约为 0.31 ns。
- **硬件规格疑问**：人们一直对硬件差异如何影响性能感到好奇，一位遇到 Rust 性能滞后的成员提到其配置中包含 Intel i7 CPU。
- **基准测试重访与转折**：进行了一项包含打印每个元素以确保所有写入都已发生的新测试，结果显示两种语言的速度都有所下降。在这些条件下，在主频为 1400 MHz 的 CPU 上，Mojo Scalar 的性能为每个项目 1.4 ns，而 Rust 和 Mojo SIMD 均达到了约 1.0 ns。
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1231366433217314909)** (24 条消息🔥): 

- **C++ 性能略胜 Python**：成员们比较了 Python/Mojo 和 C++ 实现的性能，指出 *C++ 的推理时间稍快*。显著的性能提升归功于 C++ 中没有 Python runtime API 调用。
- **图像处理代码剖析**：分享了两段用于图像处理的 Python 代码片段，暗示存在 *对 Python runtime 的大量调用*，与 C++ 操作相比，这可能会增加运行时开销。
- **优化讨论**：提到虽然 **Max** 针对 NLP/LLM 推理进行了优化，但希望未来能对包括 CNN 在内的其他类型模型进行优化。
- **ONNX 模型中的输入张量命名问题**：一位成员遇到了名为 "input.1" 的 **ONNX 模型输入张量**问题，该张量无法直接在 `model.execute` 调用中使用。**建议并验证了使用 Python 的 evaluate 来设置项目的解决方案**。
- **解决 Python API 张量命名问题**：通过一段 Python 代码片段强调了另一种解决 ONNX 模型中张量命名问题的方法，该方法使用 **解包** (`**`) 来绕过在关键字参数中使用点号的问题。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1231039774152720474)** (37 条消息🔥):

- **Pointer Pondering**：讨论围绕代码库中各种指针类型的命名和使用展开，例如 `UnsafePointer`、`DTypePointer` 及其在安全性和用法方面的细微差别。目前正在进行重构代码的工作，以弃用 `LegacyPointer`，详见此 [Pull Request](https://github.com/modularml/mojo/pull/2365)。

- **SIMD Alias Advocacy**：社区讨论了为 `SIMD[T, N]` 引入别名（如 `Float32x4`）或使用参数化别名，部分人更倾向于使用更直观的名称如 `Float32[4]`。别名的想法也扩展到了浮点数，例如使用 `alias Float = SIMD[DType.float64, _]`。

- **Int Conversion Confusion**：Mojo 2024.4.1618 的升级移除了 `SIMD.to_int()` 函数，导致使用该方法的代码构建失败。社区建议使用 `int(SIMDType)` 作为替代方案，以符合最近的更改。

- **Vacation Notification**：一名成员通知社区他们即将休假，并提供了在此期间负责 PR 审查和 Issue 处理的人员信息，鼓励在此期间有任何需求请使用 `@modularml/mojo-standard-library` 团队别名。

- **String Comparison Implementation Inquiry**：一名成员分享了一个 Python 风格语法的字符串比较潜在实现，并在创建 Pull Request 之前征求反馈，这让他们意识到可能已经有一个类似的 PR 正在审查中。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1904">[Feature Request] Explicit parametric alias with default argument · Issue #1904 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？如题。你进行此更改的动机是什么？Exp...</li><li><a href="https://github.com/modularml/mojo/pull/2365">[stdlib] Replace `Pointer` by `UnsafePointer` in `stdlib/src/builtin/object.mojo` by gabrieldemarmiesse · Pull Request #2365 · modularml/mojo</a>：Builtins 导入方式很奇怪，我不得不在 stdlib/src/python/_cpython.mojo 中导入 LegacyPointer，我对此无法解释。我只是按照编译器要求导入的内容进行导入 :p 参见 ht...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1230828290651066419)** (324 条消息🔥🔥): 

- **Llama 3 vs Claude Levels**：一位成员指出 Llama 3 的 70b 模型已达到与 Claude Sonnet 相当的水平，而 8b 版本则优于 Claude 2 和 Mistral。

- **API Access Inquiry for MistralAI**：用户咨询了 HF Pro 用户访问 **MistralAI/Mixtral-8x22B-Instruct-v0.1** 的 API 权限。

- **HF Competitions Announced**：分享了 HuggingFace 上[正在进行的竞赛链接](https://hf.co/competitions)，以及竞赛页面的图片。

- **Discussion on Hardware for ML**：关于硬件的讨论非常活跃，特别是 AMD RX 7600 XT 是否适合机器学习，最终达成的共识是高端 AMD 型号或 Nvidia 的产品可能更合适。

- **HuggingFace Service Disruptions**：许多用户报告了 HuggingFace 宕机的问题，导致项目中断。大家急切等待有关情况的公告和更新，一些用户分享了离线运行模型的变通方法。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/teknium1/status/1781328542367883765?s=46">Teknium (e/λ) (@Teknium1) 的推文</a>：伙计们，我们家里也有 GPT-4 了</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - mteb 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>：未找到描述</li><li><a href="https://huggingface.co/">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://tenor.com/view/jinx-the-cat-jinx-jinx-cat-cat-computer-gif-25786466">Jinx The Cat Jinx GIF - Jinx The Cat Jinx Jinx Cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1884c8k/todays_ai_breakthrough_zero_step_diffusion/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/resident-evil-resident-evil-welcome-to-raccoon-city-resident-evil-movie-burning-on-fire-gif-25613395">生化危机 欢迎来到浣熊市 GIF - 生化危机 欢迎来到浣熊市 生化危机电影 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/im-dead-dead-bruh-skeleton-dead-bruh-skeleton-dead-im-dead-bruh-gif-26854866">我死了 Dead Bruh GIF - 我死了 Dead Bruh 骷髅 Dead Bruh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/turn-down-for-what-snoop-dogg-cheers-dancing-drinking-gif-10966591">Turn Down For What Snoop Dogg GIF - Turn Down For What Snoop Dogg 干杯 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ">TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/cat-club-cat-cat-dance-cat-party-cat-disco-gif-27258615">猫咪俱乐部猫 GIF - 猫咪俱乐部猫 猫咪跳舞 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/eyeverse-brace-initiation-eyebrow-shave-gif-6015143619791964168">Eyeverse Brace GIF - Eyeverse Brace 启动 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">AI 的崛起</a>：(开启字幕) 加入我们的旅程，见证人工智能的快速演变，从它的出现开始...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4">meta-llama/Meta-Llama-3-8B-Instruct · 更新 generation_config.json</a>：未找到描述</li><li><a href="https://youtu.be/4oSavAHf0dg">MTRAN3 模块化机器人</a>：更多信息请访问 http://www.botjunkie.com/ 和 http://unit.aist.go.jp/is/dsysd/mtran3/mtran3.htm</li><li><a href="https://www.youtube.com/watch?v=JOeY07qKU9c>">“这是 UNIX 系统！” | 侏罗纪公园 | 科幻站</a>：黑客女孩 Lexi (Ariana Richards) 在尝试修复侏罗纪公园的 UNIX 控制系统时展示了她的极客技能。侏罗纪公园 (1993)：John Hammond，一位...</li><li><a href="https://bpa.st/3MUQ">查看 paste 3MUQ</a>：未找到描述</li><li><a href="https://hf.co/competitions">竞赛 (Competitions)</a>：未找到描述</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1230962560279380008)** (8 条消息🔥): 

- **Llama 3 在 Groq Cloud 上的惊人速度**：一段 YouTube 视频展示了 **Llama 3** 在 Groq Cloud 上运行，速度达到了约 **800 tokens per second**。该视频强调了 80 亿参数模型在该平台上的卓越性能。[LLama 3 on Groq Cloud- 800 Tokens per second!!!](https://www.youtube.com/watch?v=Z-JHgFs5BE0)

- **探索 ORPO 与 LLaMA 3 的三位一体**：另一段讨论的 YouTube 视频挑战了“快、便宜、好——三选二”的古老谚语，展示了 AI 如何通过 **ORPO 与 LLaMA 3** 等创新，开始在所有三个方面同时发力。[ORPO with LLaMA 3 -Fast, Cheap, and Good!](https://www.youtube.com/watch?v=oHM3faIPTg0)

- **强化学习的第一步**：一位成员分享了构建其第一个强化学习模型的成功经验，这是一个使用 stable-baselines3 库训练的 **PPO** Agent，用于玩 **LunarLander-v2**，并将其发布在 HuggingFace 上。[PPO Agent for LunarLander-v2](https://huggingface.co/wsqstar/ppo-LunarLander-v2)

- **学习 Tokenizers 的复杂性**：一位成员正致力于学习 **tokenizers**，它在为语言模型准备数据方面起着至关重要的作用。

- **对 HuggingFace 的依赖仍在持续**：尽管已经安装了模型，一位成员幽默地指出他们仍然依赖 HuggingFace 提供的资源，暗示了该平台在他们 AI 工作中的重要性。

- **使用 LlamaIndex 构建 RAG 系统**：当天的学习内容包括使用 LlamaIndex 构建带有 Agent 的**检索增强生成 (RAG) 系统**，这表明了对高级 AI 系统架构的探索。

- **进军基于 AI 的教育创业**：个人正在开发他们的第一个 MVP (Minimum Viable Product)，旨在创建一个将 AI 整合到课堂中的业务，这标志着 AI 研究与教育创新之间的交汇。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/wsqstar/ppo-LunarLander-v2">wsqstar/ppo-LunarLander-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=oHM3faIPTg0">ORPO with LLaMA 3- Fast, Cheap, and Good!</a>: 俗话说“快、省、好——三者只能择其二”。AI 也不例外，但我们开始看到一些伟大的创新正在改变这一点。一篇很棒的文章...</li><li><a href="https://www.youtube.com/watch?v=Z-JHgFs5BE0">LLama 3 on Groq Cloud- 800 Tokens per second!!!</a>: @meta 在 Groq 上的 LLama3 快得惊人。使用 @GroqInc Cloud 测试他们的 8B 参数模型，我始终能获得每秒 800 个 Token 左右的速度。这...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1230917830166319115)** (11 messages🔥): 

- **探索 Llama3 的阴暗面**：分享了一个 LinkedIn 帖子的链接，讨论了 **Llama3** 阴暗的一面，这可能指的是与该模型相关的漏洞或滥用潜力。
- **LLM 微调基础**：推荐了一个 GitHub 仓库，提供了一个微调语言模型（特别是 Llama）的基础指南，包含 [Fine-tune basics guide](https://github.com/andysingal/llm-course/blob/main/llama_finetune/Fine-tune-basics.md)。
- **量子计算：潜力与陷阱**：分享了一部 YouTube 纪录片，标题为 ["New quantum computers - Potential and pitfalls | DW Documentary"](https://youtu.be/0HFzTYlhT2E)，探讨了量子计算机的潜力，包括医疗和科学进步。
- **为什么神经网络是强大的学习者**：一个讨论点强调了一个 YouTube 视频，解释了神经网络为什么以及如何能够学习几乎任何东西：[Why Neural Networks can learn (almost) anything](https://www.youtube.com/watch?v=0QczhVg5HaI)。
- **Whisper 提示词成像**：提供了关于直播的信息，其中高分辨率图像 (SDXL) 由 Whisper 语音命令控制和提示，详见 [Twitter](https://twitter.com/Dan50412374/status/1781790992318042428)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sites.google.com/view/hok-offline">Hokoff</a>: 摘要 </li><li><a href="https://github.com/andysingal/llm-course/blob/main/llama_finetune/Fine-tune-basics.md">llm-course/llama_finetune/Fine-tune-basics.md at main · andysingal/llm-course</a>: 通过在 GitHub 上创建账户，为 andysingal/llm-course 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=0QczhVg5HaI">Why Neural Networks can learn (almost) anything</a>: 一个关于神经网络、它们如何工作以及为什么有用的视频。我的 Twitter: https://twitter.com/max_romana 来源 Neural network playground: https://play...</li><li><a href="https://youtu.be/0HFzTYlhT2E?si=lgzMqlFFbhVgjM7f">New quantum computers - Potential and pitfalls | DW Documentary</a>: 一台新的超级计算机预计将使减少动物实验并可能治愈癌症成为可能。围绕量子计算的炒作令人振奋...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1230862529006141521)** (27 messages🔥):

- **社区准则强化**：发布了一项提醒，要求遵守社区准则，并避免在 Discord 频道中重复跨频道发布（cross-posting）。
- **AI 模型涌现**：用户分享了**大语言模型的多次迭代和创作**，增强版本涵盖了从 [3-4b 社区版](https://huggingface.co/ehristoforu/Llama-3-4b-community) 到 [100B 参数](https://huggingface.co/ehristoforu/Gixtral-100B) 的版本，显示了 AI 社区内的进步和定制化趋势。
- **革新数据集调试**：宣布了 [3LC](https://3lc.ai/) 的 Beta 版发布，该工具为计算机视觉和未来的 LLM 微调提供数据集精炼工具。
- **基于 RAG 的 AI 聊天机器人**：分享了一篇关于**使用 Llama3 模型创建 RAG 聊天机器人**的博客文章链接，概述了 AI 的实际应用方案（[HuggingFace Blog](https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3)）。
- **创新 Spaces 展示**：重点介绍了使用 **Hugging Face Spaces** 的显著项目，包括 [VTuber 徽标生成器](https://huggingface.co/spaces/gojiteji/VTuberLogoGenerator) 和 [使用差分扩散（differential diffusion）进行图像外扩（outpainting）的演示](https://huggingface.co/spaces/clinteroni/outpainting-with-differential-diffusion-demo)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://3lc.ai/">首页</a>: 未找到描述</li><li><a href="https://huggingface.co/ehristoforu/Gixtral-100B">ehristoforu/Gixtral-100B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gojiteji/VTuberLogoGenerator">VTuberLogoGenerator - gojiteji 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/ehristoforu/llama-3-12b-instruct">ehristoforu/llama-3-12b-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/clinteroni/outpainting-with-differential-diffusion-demo">Outpainting Demo - clinteroni 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF">QuantFactory/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">moondream2-batch-processing - Csplk 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">使用 llama3 的 RAG 聊天机器人</a>: 未找到描述</li><li><a href="https://huggingface.co/ehristoforu/Gistral-16B">ehristoforu/Gistral-16B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KBlueLeaf/This-Cute-Dragon-Girl-Doesnt-Exist">This Cute Dragon Girl Doesnt Exist - KBlueLeaf 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf: 在本地与任何 PDF 聊天，提问并获取带有有用引用的答案，对数学 PDF 效果良好（将其转换为计算机可理解的数学语法 LaTeX）</a>: 在本地与任何 PDF 聊天，提问并获取带有有用引用的答案，对数学 PDF 效果良好（将其转换为计算机可理解的数学语法 LaTeX） - Crizomb/ai_pdf
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1231074786487042119)** (7 条消息): 

- **开源 OCR 工具崛起**：分享了 [Nougat](http://github.com/facebookresearch/nougat) 的 GitHub 仓库，这是一个开源 OCR（光学字符识别）工具，旨在将数学论文等学术文档从 PDF 转换为 LaTeX，因其高效且免费而受到推荐。

- **Facebook 的开源贡献**：一位社区成员对 Mark Zuckerberg 提供 Nougat OCR 等开源工具表示感谢，尽管幽默地称他“可能是一个蜥蜴人”。

- **发票数据提取模型架构请求**：一位用户询问了关于从扫描为图像的发票和收据中提取数据的机器学习模型架构方法。

- **增强羽毛球追踪**：分享了 [TrackNetV3 GitHub 仓库](https://github.com/qaz812345/TrackNetV3) 的链接，其中包含旨在改进羽毛球追踪的实现，不过该用户正在寻求关于单独处理每一帧的建议。

- **构建私有知识库咨询**：另一位成员提出了一个关于开发私有知识库的通用问题，但未提供具体细节或背景。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://github.com/facebookresearch/nougat">GitHub - facebookresearch/nougat: Implementation of Nougat Neural Optical Understanding for Academic Documents</a>: 学术文档的 Nougat Neural Optical Understanding 实现 - facebookresearch/nougat</li><li><a href="https://github.com/qaz812345/TrackNetV3">GitHub - qaz812345/TrackNetV3: Implementation of paper - TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification</a>: 论文实现 - TrackNetV3: 通过增强和轨迹修正提升羽毛球追踪 - qaz812345/TrackNetV3
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1230963488499699772)** (11 messages🔥): 

- **微调的烦恼与建议**：一位成员在尝试微调 PHI-2 时遇到了问题，收到的建议是从较小的 batch size（如 32）开始，并进行调整以找到稳定的设置。
- **Rust 迎来 MinBPE**：宣布了一个名为 `minbpe-rs` 的 `minbpe` 新 Rust 移植版本，并邀请大家查看 GitHub 项目 [这里](https://github.com/gnp/minbpe-rs)。据称它与原始 API 几乎有一一对应的关系。
- **MinBPE-Rust 项目的协作努力**：`minbpe-rs` 的首席开发人员和文档撰写者强调了该项目的特性，如 `BasicTokenizer`、`RegexTokenizer` 和 `GPT4Tokenizer`，包括与 `tiktoken` 的 GPT-4 tokenizer 的兼容性测试。
- **BERTopic 与 OpenAI 的冲突**：一位成员分享了 **BERTopic** 新版本导致 OpenAI 工具依赖问题的经验。他们建议将脚本锁定在 0.16.0 版本以避免兼容性问题。
- **寻求 Go-Emotions 数据集**：请求协助如何将 go-emotions 数据集集成到正在进行的项目中。

**Link mentioned**: <a href="https://github.com/gnp/minbpe-rs">GitHub - gnp/minbpe-rs: Port of Andrej Karpathy&#39;s minbpe to Rust</a>: Andrej Karpathy 的 minbpe 到 Rust 的移植版本。可以通过在 GitHub 上创建账号为 gnp/minbpe-rs 的开发做出贡献。

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1230913972950143097)** (4 messages): 

- **寻求用于 Inpainting 一致性的 Lora 训练**：一位成员询问使用 [Lora 训练](https://arxiv.org/abs/2106.09685) 进行 inpainting，以便在不改变图中物体的情况下保持背景一致性。目前的 inpainting 结果不尽如人意，因此寻找 Lora 作为潜在替代方案。

- **咨询在 Android 平板上使用 fooocus**：一位成员询问如何在 android 平板上使用应用 **fooocus**，但未提供他们可能遇到的具体问题。

- **提供快速原型开发和 Stable Diffusion 方面的专业知识**：一位成员介绍了自己，提供网页设计、MVPs、应用开发和可扩展基础设施服务，包括 AWS 和部署经验。他们强调了在 Stable Diffusion、统计学和 Computer Vision 等领域超过三年的经验，并邀请通过私信讨论项目。

- **Vespa 模型下载故障排除**：一位成员报告在使用 **vespa** 下载模型时遇到 403 错误，寻求解决该问题的帮助。未提供关于错误或已采取步骤的额外信息或背景。
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1231051893577482240)** (5 messages): 

- **Llama 旅团的冲锋**：新的 **Nitro 加速 Llama 模型** 现已面向 OpenRouter 用户开放，提供潜在的性能和能力增强。可以在 [这里](https://openrouter.ai/models?q=llama-3-) 查看。

- **压力下的魔法**：OpenRouter 的 **Wizard 8x22b** 模型面临高需求，给供应商带来了压力。负载均衡调整正在进行中，以缩短响应时间。

- **延迟改进即将到来**：最近的负载均衡器更改和 stop tokens 处理修复现在应该会提高非流式请求的吞吐量，旨在优化整体性能。

- **推文简述**：OpenRouter AI 发布了一条新推文，可以直接在 [Twitter](https://twitter.com/OpenRouterAI/status/1781932094714982746) 上查看。

- **重定向模型请求**：由于 **Databricks: DBRX 132B Instruct (nitro)** 已被移除，相关请求将被重定向。用户可以使用标准的 [Databricks: DBRX 132B Instruct 模型](https://openrouter.ai/models/databricks/dbrx-instruct) 代替。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct).">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX 是由 Databricks 开发的一款新型开源 LLM。参数量为 132B，它在语言的标准行业基准测试中优于现有的开源 LLM，如 Llama 2 70B 和 Mixtral-8x7B...</li><li><a href="https://openrouter.ai/models?q=llama-3-">OpenRouter</a>: 在 OpenRouter 上浏览模型。
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231042757783588924)** (5 条消息): 

- **URL 混淆已解决**：频道描述中提到的旧 URL 导致了混淆，但在用户指出问题后已立即更新。

- **产品反馈 - 建议改进**：一位用户提供了关于产品的详细反馈，建议的改进包括：澄清对特定合同类型的支持、增加对雇佣合同的支持、考虑当地法律、提供更简单的通俗语言解释、允许用户指定谈判偏好以及标记合同中的非法条款。

- **KeyWords AI 赞扬 OpenRouter**：KeyWords AI 平台（访问地址 [https://keywordsai.co](https://keywordsai.co)）赞扬了 OpenRouter 的模型更新，这使得 KeyWords AI 能够专注于为开发者添加功能，如 request logging、usage dashboards 和 user analytics。KeyWords AI 支持所有 OpenRouter 模型，且仅需两行代码即可集成。

**提到的链接**：<a href="https://keywordsai.co)">未找到标题</a>：未找到描述

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1230783458016624650)** (353 条消息 🔥🔥): 

- **LLaMA-3 的多语言能力与 Fine-Tuning 挑战**：讨论指出 LLaMA-3 在多语言能力方面存在局限性，这可能是由于在非英语数据集上的 fine-tuning 有限。然而，用户根据 Meta 的承诺，对未来版本中更好的多语言支持表示期待，并建议通过 fine-tuning 来提高性能。

- **聊天机器人中的 Tool Use 与 Streaming**：关于 OpenAI 和 Claude 等聊天机器人中 tool use 的对话指出了当前的局限性，特别是在 streaming tool call 请求方面。用户期待像 Anthropic 这样的供应商引入 streaming，这可以提高这些模型中 tool calls 的效率。

- **创意写作中不同 LLM 的对比**：用户分享了使用不同 LLM 进行创意任务的经验，比较了 Wizard LM-2 和 Mixtral 等模型在指令遵循、对话能力和上下文理解方面的细微差别、优势和缺点。

- **OpenRouter 上的供应商性能与模型的静态特性**：大家公认虽然 OpenRouter 上的模型是静态且不变的，但由于平台之外的模型托管方的更新，可能会产生性能差异，用户希望能够继续访问高质量、审查较少的版本。

- **对新模型贡献和功能的兴趣**：用户讨论了对新型和改进模型的兴趣，特别强调了在角色扮演和创意写作等特定任务上的更好表现。社区还对一款声称是增强型角色扮演模型 Soliloquy-L3 的推出表现出热情，该模型具有支持更大 context length 等特殊功能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/XoI7ZD9">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因（memes）、有趣的 GIF、励志故事、病毒式视频等来振奋你的精神...</li><li><a href="https://groq.com/">GroqChat</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Work-to-rule">Work-to-rule - Wikipedia</a>: 未找到描述</li><li><a href="https://x.com/erhartford/status/1781199815772438819">来自 Eric Hartford (@erhartford) 的推文</a>: 由 @CrusoeCloud 慷慨赞助的 Dolphin-2.9-llama3-8b 预计周六发布。与 @LucasAtkins7 和 @FernandoNetoAi 进行了大量合作。Dolphin-2.9-llama3-70b 紧随其后。Dolphin-2.9-mixtral-8x22b 仍在...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3">lynn 开发的 Llama 3 Soliloquy 8B | OpenRouter</a>: Soliloquy-L3 是一款快速、高性能的角色扮演模型，专为沉浸式、动态体验而设计。Soliloquy-L3 在超过 2.5 亿个 token 的角色扮演数据上进行了训练，拥有广博的知识库、丰富的...</li><li><a href="https://huggingface.co/dreamgen/opus-v1.2-llama-3-8b">dreamgen/opus-v1.2-llama-3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/WizardLM/329547800484476">Hugging Face 上的 @WizardLM: "🔥🔥🔥 隆重推出 WizardLM-2! 📙发布博客：…"</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1230893023928520744)** (201 条消息🔥🔥): 

- **Llama 3 性能讨论**: 正在进行 **Llama 3** 和 **GPT-4** 的持续对比，多位用户在包括 [meta.ai](https://x.com/hive_echo/status/1781220509147095059)、[HuggingChat](https://x.com/kwindla/status/1781408311021367761) 等不同平台上测试 Llama。一位用户声称，尽管 lmsys 评分很高，但 **Llama 3 70b** 并不像 **GPT-4 Turbo** 那样令人印象深刻。
- **推理时间讨论**: 用户正在讨论 **Llama 3 的推理时间**，其中一位指出 **Groq Cloud** 为 **Llama-3 70b** 提供的首字节时间（time-to-first-byte）低于 100ms，速度极快。另一位提到 **Deepgram** 是语音对语音（voice 2 voice）AI 应用中转录的首选。
- **启动 LLM 项目**: 对于那些希望启动新 **LLM 项目** 的人，建议使用像 [litellm](https://litellm.vercel.app/) 这样的模板，它可以抽象化调用 LLM 的样板代码，以便轻松在不同模型之间切换。
- **微调与配置工具**: 关于微调和配置复杂应用的工具讨论中提到了 [Hydra by Facebook Research](https://github.com/facebookresearch/hydra)，但一些用户发现其 README 缺乏对其目的和用途的清晰解释。
- **新兴数据集**: 分享了 **FineWeb** 的发布公告，这是一个拥有 15 万亿 token 高质量网络数据的新数据集，表明其性能可能优于 **RefinedWeb** 和 **The Pile** 等现有数据集。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">来自 Guilherme Penedo (@gui_penedo) 的推文</a>：我们刚刚发布了 🍷 FineWeb：15 万亿 tokens 的高质量网络数据。我们对 2013 年至 2024 年间的所有 CommonCrawl 进行了过滤和去重。在 FineWeb 上训练的模型性能优于 RefinedWeb, C4, ...</li><li><a href="https://tinygrad.org/#tinybox">tinygrad：一个简单且强大的神经网络框架</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1781328542367883765?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：伙计们，我们现在家里也有 GPT-4 了（指本地运行同级别模型）</li><li><a href="https://www.macrumors.com/2024/04/11/m4-ai-chips-late-2024/">Mac 将于 2024 年底开始搭载专注于 AI 的 M4 芯片</a>：据 Bloomberg 的 Mark Gurman 报道，苹果将于 2024 年底开始使用 M4 芯片更新其 Mac 产品线。M4 芯片将专注于...</li><li><a href="https://litellm.vercel.app/">LiteLLM - 入门指南 | liteLLM</a>：https://github.com/BerriAI/litellm</li><li><a href="https://x.com/hive_echo/status/1781220509147095059">来自 echo.hive (@hive_echo) 的推文</a>：测试 Llama-3 8B 和 70B。这个简单的测试结果向我证明，更小模型配合更多数据可以成为出色的低端推理器，而更大模型配合更多数据则能成就出色的高端...</li><li><a href="https://x.com/kwindla/status/1781408311021367761">来自 kwindla (@kwindla) 的推文</a>：哇。Llama-3 70B 在 @GroqInc 上的首字节时间（TTFT）非常快 —— 快到 100ms 以下。</li><li><a href="https://x.com/teknium1/status/1781328542367883765?s=46&t=90xQ8sGy63D">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：伙计们，我们现在家里也有 GPT-4 了</li><li><a href="https://www.browserless.io/">Browserless - 排名第一的 Web 自动化和无头浏览器自动化工具</a>：免费试用 Browserless，最好的 Web 自动化工具之一。轻松实现网页抓取、PDF 生成和无头浏览器自动化。</li><li><a href="https://x.com/theseamouse/status/1781134831914508720?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：我仍然对此感到震惊。它是如何提升这么多的？我是说，看看 8B 对比旧版的 70B</li><li><a href="https://github.com/facebookresearch/hydra">GitHub - facebookresearch/hydra：Hydra 是一个用于优雅配置复杂应用程序的框架</a>：Hydra 是一个用于优雅配置复杂应用程序的框架 - facebookresearch/hydra</li><li><a href="https://www.firecrawl.dev/">FireCrawl</a>：将任何网站转换为 LLM 就绪的数据。</li><li><a href="https://buttondown.email/ainews/archive/ainews-llama-3/">[AINews] Llama-3-70B 是 GPT-4 级别的开源模型</a>：2024/4/18-2024/4/19 的 AI 新闻。我们为您检查了 6 个 subreddit、364 个 Twitter 账号和 27 个 Discord 社区（395 个频道，10403 条消息）。预计阅读时间...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1230959116919509062)** (3 条消息): 

- **新的 Latent Space 播客集数**：令人兴奋地宣布了由 **<@199392275124453376> (Jason Liu)** 主讲的 Latent Space Podcast 新一集。公告中包含了一个指向该集数的 [Twitter 链接](https://twitter.com/latentspacepod/status/1781400226793673137)。
- **热切期待与 Jason Liu 的播客**：成员对新发布的 Jason Liu 播客集数表示热烈欢迎。大家对这一集可能带来的见解充满期待。
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1230880331482136588)** (66 条消息 🔥🔥):

- **对 GPT 论文讨论的热情**：成员们对讨论开创性的 "Improving Language Understanding by Generative Pre-Training" 论文表现出极大的热情，有人指出这是一篇极具影响力的（"goated"）论文，并强调了 Alec Radford 在本科毕业仅三年后取得的成就。
- **关于 Embeddings 和 Tokenizers 的澄清**：会议澄清了与 Embeddings 不同（Embeddings 需要神经网络进行学习），Tokenizers 并不一定需要此类训练，这一区别对于该领域的新手来说并不显而易见。
- **录制并分享 Paper Club 环节的意向**：成员们获悉 Asia Paper Club 的环节正在进行录制，并计划将内容上传至 YouTube 以供更广泛的访问。
- **揭秘与理解复杂话题**：小组讨论了各种复杂话题，包括 Perplexity (pplx) 数值在不同模型之间是否具有可比性，以及机器学习中 GPU 使用的历史，并有兴趣创建图表等视觉辅助工具以更好地理解这些趋势。
- **对 Paper Club 演示的感谢**：演示结束后，参与者对富有洞察力的演讲和分享的资源表示感谢，并提供了链接以便进一步了解讨论的材料。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openai.com/research/scaling-laws-for-neural-language-models">Scaling laws for neural language models</a>：神经网络语言模型的 Scaling laws</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production">TinyBox packs a punch with six of AMD's fastest gaming GPUs repurposed for AI &mdash; new box uses Radeon 7900 XTX and retails for $15K, now in production</a>：初创公司希望利用 Radeon RX 7900 XTX 提供高性能 AI 计算。</li><li><a href="https://paperswithcode.com/dataset/mrpc">Papers with Code - MRPC Dataset</a>：Microsoft Research Paraphrase Corpus (MRPC) 是一个由从新闻文章中收集的 5,801 个句子对组成的语料库。每个句子对都由人工标注者标记是否为释义。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1230971054881505292)** (71 messages🔥🔥): 

- **切换到 Zoom 还是留在 Discord**：有人询问本周的会议是否会从 Discord 转移到 Zoom。
- **分享 LLM 评估策略**：分享了一份关于 **LLM Evaluation** 的 Google Slides 演示文稿以供审阅，但未提供幻灯片的具体细节。
- **电话会议的信号完整性**：一些成员讨论在通话期间听到神秘的嗡嗡声，而其他人则没有。在一名成员重新加入通话后，问题似乎得到了解决。
- **评估语言模型**：一名成员分享了 Eugene Yan 博客中两篇文章的链接，讨论了评估语言模型中 **Abstractive Summarization**（抽象式摘要）的挑战和策略，并分享了有用的评估方法论。
- **模型评估与选择策略**：讨论了关于评估和选择模型的各种建议，包括使用遥测数据合成评估集、针对单一模型进行一致的基准测试（Baselining），以及在评估生产环境中的模型时考虑成本与性能的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eugeneyan.com/writing/abstractive/">Evaluation & Hallucination Detection for Abstractive Summaries</a>：抽象式摘要的评估与幻觉检测：基于参考、上下文和偏好的指标，自我一致性以及捕捉幻觉。</li><li><a href="https://eugeneyan.com/writing/evals/">LLM Task-Specific Evals that Do & Don't Work</a>：有效与无效的 LLM 特定任务评估：针对分类、摘要、翻译、版权复现和毒性的评估。</li><li><a href="https://docs.google.com/presentation/d/14EE2j6ii4PEA0Y-wUg80weC3eJ-qx2q41uUAEqytG28/edit?usp=sharing">LLM Evaluation</a>：评估基于 LLM 的系统，Alan van Arden，2024 年 4 月 19 日，Latent Space</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题、日期、主持人、资源，GenAI 的 UI/UX 模式等。
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1230806026882912317)** (247 messages🔥🔥):

- **对 Meta 处理 LLaMA-3 方式的担忧**：一名成员质疑为什么 **Meta** 扣留了 **LLaMA-3 论文**，并指出对于通常在发布权重之前发布论文的公司来说，这很不寻常。这种做法的改变表明 Meta 试图在其开源模型策略上进行创新。

- **提示工程（Prompt Engineering）技术辩论**：用户讨论了在图像扩散模型中生成对齐良好的输出的各种策略，一些人建议在提示词中添加高度具体的相机或胶片类型会带来更好的结果，而另一些人则认为这会导致输出多样性降低，并且可能是安慰剂效应。

- **探讨 Nightshade 的法律风险**：对话转向了使用 **Nightshade**（一种旨在干扰 AI 训练的算法）的伦理和法律影响，并提供了指向一篇[讨论法律问题的文章](https://undeleted.ronsor.com/nightshade-legal-poison/)的链接。成员们提到了 **Computer Fraud and Abuse Act** (CFAA) 下的潜在问题，强调了遵守数据权利和避免法律责任的重要性。

- **在 Discord 上检测到机器人监视**：一名成员分享了一个**监视机器人**检测和移除工具 [kickthespy.pet](https://kickthespy.pet/#823813159592001537)，引发了关于 Discord 隐私的讨论。这一发现促使管理员采取了行动，并引发了关于此类机器人普遍性的更广泛对话。

- **AI 模型性能讨论**：用户分享了关于 AI 模型性能随规模扩展（scaling）而提升的见解，并引用了 OpenAI 的 **DALL-E 3** 为例。尽管数据规模的收益递减，成员们仍指出即使是微小的性能提升也具有重要意义，并考虑了对扩散模型等其他模型的影响。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://kickthespy.pet/#823813159592001537">Kick the Spy Pet</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2203.15556">Training Compute-Optimal Large Language Models</a>：我们研究了在给定计算预算下训练 Transformer 语言模型的最优模型大小和 Token 数量。我们发现目前的大语言模型明显训练不足...</li><li><a href="https://undeleted.ronsor.com/nightshade-legal-poison/">Nightshade: Legal Poison Disguised as Protection for Artists</a>：正如我在前一篇文章中所述，生成式 AI 对许多艺术家来说仍然是一个充满争议的话题，为了抵制模型训练，出现了各种方案。上一篇文章...</li><li><a href="https://snap-research.github.io/mixture-of-attention/">Mixture of Attention</a>：未找到描述</li><li><a href="https://tenor.com/view/oh-no-top-gear-jeremy-clarkson-no-one-cares-gif-18925814">Oh No Top Gear GIF - Oh No Top Gear Jeremy Clarkson - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://eugeneyan.com/writing/text-to-image/">Text-to-Image: Diffusion, Text Conditioning, Guidance, Latent Space</a>：文本生成图像的基础知识、相关论文以及 DDPM 实验。</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_mem.py">cookbook/calc/calc_transformer_mem.py at main · EleutherAI/cookbook</a>：深度学习入门指南。包含处理真实模型时的所有实践细节和实用工具。 - EleutherAI/cookbook</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>：(开启中文字幕) 加入我们，一起回顾人工智能的快速演进，从它的出现...</li><li><a href="https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/model/unet.py#L225>">IF/deepfloyd_if/model/unet.py at develop · deep-floyd/IF</a>：通过在 GitHub 上创建账号为 deep-floyd/IF 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1230930482795905146)** (72 条消息🔥🔥): 

- **对开源解决方案的期待**：鉴于 Meta 对多模态的承诺，讨论集中在预期 Meta 将发布可能与现有专有解决方案竞争或超越它们的开源多模态模型。
  
- **加速扩散模型**：NVIDIA、多伦多大学和 Vector Institute 提出了 "Align Your Steps"，这是一种优化扩散模型采样调度的方法，旨在以更快的速度获得更高质量的输出。该研究在[论文](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/)中进行了讨论，重点是在不牺牲质量的情况下降低缓慢的采样速度，但未发布训练代码被视为一个局限性。

- **Evaluating Multimodal Language Models**: 评估多模态语言模型：提到了一项名为 Blink 的新基准测试，旨在评估多模态语言模型 (LLMs) 的核心视觉感知能力，这对该类模型具有挑战性；即使是表现最好的模型（如 GPT-4V），其准确率也显著低于人类。在 [研究摘要](https://arxiv.org/abs/2404.12390) 中可以找到 Blink 基准测试的详细信息。

- **Discussing Upscaling and Tuning in Image Models**: 讨论图像模型的放大与微调：关于图像模型放大进展（如 2D RoPE 外推）的对话强调了在高分辨率下微调模型的持续需求，以及产生连贯输出的挑战。

- **On the Horizon for Model Optimization**: 模型优化的前景：讨论涉及了使用简单神经网络为图像生成中不同提示词 (prompts) 优化采样调度的潜力，并提出了最优调度可能取决于图像类别的观点。对话表明存在更多研究机会，特别是条件细粒度微调如何影响扩散模型 (diffusion models)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.12390">BLINK: Multimodal Large Language Models Can See but Not Perceive</a>：我们介绍了 Blink，这是一个针对多模态语言模型 (LLMs) 的新基准测试，专注于其他评估中未包含的核心视觉感知能力。大多数 Blink 任务可以由人类解决...</li><li><a href="https://arxiv.org/abs/2404.12803">TextSquare: Scaling up Text-Centric Visual Instruction Tuning</a>：随着多模态大语言模型 (MLLMs) 的发展，以文本为中心的视觉问答 (VQA) 取得了长足进步，但开源模型仍落后于 GPT 等领先模型...</li><li><a href="https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/">Align Your Steps</a>：Align Your Steps：优化扩散模型中的采样调度</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwu">bghira</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://colab.research.google.com/drive/1cIwbbO4HRP1aUQ8WcbQBaT8p3868k7BC?usp=sharing#scrollTo=ViIqT9tnaoZZ">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md">piecewise-rectified-flow/README.md at main · magic-research/piecewise-rectified-flow</a>：通过在 GitHub 上创建账号来为 magic-research/piecewise-rectified-flow 的开发做出贡献。</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwuserbghira">bghira</a>：Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1231977925314744360)** (6 条消息): 

- **Coding Assistant Collaboration**: 编程助手协作：一名成员表示有兴趣构建一个专注于 JavaScript/Rust 的 **NLP 编程助手**并寻求协作，提供互助。
- **Limited Availability for Assistance**: 协助时间有限：另一名成员表示愿意为该项目提供帮助，但指出由于承诺了多个其他项目，其时间可能会受到限制。
- **Request for Project Resources**: 项目资源请求：在预期的协作中，有人请求提供包含任何先前工作的代码库，这意味着希望评估或在过去成果的基础上进行构建。
- **Acknowledgement of Previous Project Limitations**: 承认先前项目的局限性：分享专业知识的成员承认，由于当时 AI 知识有限，停止了一个过去的相关项目。
- **Offering Help with Tasks**: 提供任务帮助：尽管之前停止了项目，这位专家成员确认他们有能力协助处理目前与 NLP 编程助手项目相关的任务。
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1230816209738399826)** (193 条消息 🔥🔥):

- **Groq Cloud API 赋能免费 AI 创意**：一位用户强调，他们基于聊天的角色扮演 GPT 表现令人印象深刻，甚至能编写不错的 Python 代码，并提到它运行在 Groq Cloud API 上，任何人都可以免费使用。
- **通过 Groq 实现 LLaMa 3 免费推理**：用户推荐 LLaMa 3 为目前最佳的免费模型，甚至优于 Claude 3 Sonnet 和 ChatGPT-3.5，且可以在 Groq Cloud 上进行免费推理。
- **关于 AI 感知力与情感的辩论**：在一场哲学讨论中，用户们思考了意识、感知力和人类体验的定义，一些人认为情感是使 AI 趋近于人类意识的重要组成部分。
- **展望 AI 发展与人类未来**：一位用户描绘了未来“数字雅典”的愿景，届时机器人将承担劳动，并质疑超人类主义的含义，以及技术将导致永生还是“僵尸赛博格”式的存在。
- **为学术论文寻求深度 AI 资源**：一位正在撰写关于 AI、生成算法及相关技术大学论文的用户请求协助寻找深度文本和资源，并被引导至 OpenAI 的研究论文和出版物。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/joe-bereta-source-fed-micdrop-im-out-gif-11904628">Joe Bereta Source Fed GIF - Joe Bereta Source Fed Micdrop - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://openai.com/research/generative-models">Generative models</a>：这篇文章描述了四个项目，它们共同的主题是增强或使用生成模型，这是机器学习中无监督学习技术的一个分支。除了描述我们的工作...</li><li><a href="https://openai.com/research/overview">Research</a>：我们相信我们的研究最终将通向通用人工智能 (AGI)，一个能够解决人类水平问题的系统。构建安全且有益的 AGI 是我们的使命。</li><li><a href="https://en.wikipedia.org/w/index.php?title=Biorobotics">Biorobotics - Wikipedia</a>：未找到描述</li><li><a href="https://openai.com/research/gpt-4">GPT-4</a>：我们创建了 GPT-4，这是 OpenAI 在扩展深度学习方面的最新里程碑。GPT-4 是一个大型多模态模型（接受图像和文本输入，输出文本），虽然能力稍逊...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1230803721345302549)** (32 条消息🔥): 

- **GPT-4 性能查询**：一位用户对 Assistant 流水线中 **GPT-4-turbo** (0409 版本) 相较于最新的预览版 (0125) 的表现表示失望。
- **用户尝试 AI 组合实验**：一位参与者分享了将 **Claude 3 Opus** 与 GPT-4 链接，随后通过 Groq 集成 **LLama 3 70B** 的实验，结果褒贬不一，其他用户在访问提供的[共享链接](https://chat.openai.com/g/g-myMXsnyWs)和[相关集成](https://chat.openai.com/g/g-fXbe7EW2h)时遇到了访问问题。
- **AI 融合反馈循环的考量**：社区正在探索如何更好地结合不同 AI 模型的响应，重点是在没有显式反馈的情况下提高质量，并提到了关于发布来自 **cgpt chats** 的共享 URL 的限制。
- **Assistant API 提升 UI 响应性的讨论**：围绕通过在后端获取数据时向用户流式传输加载消息来提高 Assistant API 的 UI 参与度展开了讨论，建议侧重于 UI 操作而非 API 修改来实现动态文本显示。
- **探索层适配技术**：关于 **卷积层** (Hyena) 和 **LoRa** (Layer-wise Relevance Propagation) 在 LLM (Large Language Models) 中作用的理论讨论正在进行，涉及它们在微调 **Mistral 7B** 等模型中的应用。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1230886659026059264)** (29 条消息🔥): 

- **JSON 数据摘要难题**：一位成员在让模型将 JSON 字段中的确切文本插入到生成的摘要中时面临挑战。经过讨论，他们计划尝试在摘要模板中嵌入 **code interpretation**。

- **自定义指令 (Custom Instructions) 辩论**：一位成员质疑了自定义指令的理想长度。一些回复建议仅使用极简指令以节省上下文空间，而另一位成员则使用了冗长的指令，这似乎限制了 ChatGPT 的响应。

- **法学院学生寻求刑法 Prompt**：一位法学院学生请求专门针对刑法用途定制的 Prompt，但目前尚未提供关于该主题的进一步细节或回复。

- **针对增强邮件的 Prompt 优化**：一位成员分享了他们使用 GPT-4 进行邮件增强程序的经验，并寻求关于优化 Prompt 以获得更高质量回复的建议。

- **寻找 Prompt 库**：一位用户询问如何找到 Prompt 库，但目前还没有提供具体方向或链接的回复。

- **Prompt Engineering 伦理担忧**：一位拥有超过两年 Prompt Engineering 经验的成员对分享具有潜在危害和敏感性的 Prompt 表示担忧，强调了利用特定 Prompt 操纵 ChatGPT 的简易性。

- **最优 Prompt 长度讨论**：关于 Prompt 长度有效性的对话；成员们辩论了更长、更具体的 Prompt 是否能带来更好的回复，还是简洁的 Prompt 更好。一个建议是必要时将冗长的指令分散到多条消息中。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1230886659026059264)** (29 messages🔥): 

- **在摘要中寻求精确性**：一位成员在让 GPT 从 JSON 数据摘要的特定字段中返回准确文本时遇到困难。尽管有明确指令，Bot 仍无法按要求包含准确文本。建议包括使用 Code Interpreter 更可靠地提取数据。

- **自定义指令：少即是多？**：用户讨论了 ChatGPT 回复的自定义指令长度，一些人选择较短的指令以节省 Context Window 空间。给出了一个简短示例：“在回复中适用的地方包含分号、冒号和破折号。”

- **AI 与法律查询**：有人为法学院学生请求刑法相关的 Prompt。然而，目前还没有关于具体 Prompt 的讨论或进一步跟进。

- **使用 GPT-4 增强邮件**：一位个人分享了他们使用 GPT-4 增强邮件的程序用例。他们对质量不稳定的情况表示沮丧，并寻求 Prompt 优化建议，引发了关于短 Prompt 与长 Prompt 效率的讨论。

- **Prompt Engineering 伦理与问题**：一位资深的 Prompt Engineer 对分享激进且具有潜在危害的 Prompt 表示担忧，强调了伦理考量和论坛分享指南。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1230935044441247775)** (6 messages): 

- **使用 LlamaParse 自动化代码编写**：与 **@TechWithTimm** 的合作教学，教你如何使用 @ollama 设置本地 LLM，使用 **LlamaParse** 解析文档，构建一个 Agent，并教它编写代码。点击 [Twitter](https://twitter.com/llama_index/status/1781375488759570829) 查看工作流概览。
  
- **使用 Llama-3 构建本地 RAG 应用**：学习如何使用 MetaAI 的 Llama-3 **100% 本地构建 RAG 应用**，详情见 [Twitter](https://twitter.com/llama_index/status/1781422831202648264) 上的链接。

- **引入用于 RAG 实验的 DREAM**：Aishwarya Prabhat 介绍了 **DREAM**，这是一个旨在有效实验 RAG 设置的框架，满足开发阶段微调多个参数的需求。更多细节请访问 [Twitter](https://twitter.com/llama_index/status/1781725652447879672)。

- **使用 LlamaIndex 构建金融 Agent**：Hanane Dupouy 的迷你博客展示了一个用于查询上市公司数据的工具包，包括股票价格和财经新闻摘要，基于 **@llama_index** 构建。在 [Twitter](https://twitter.com/llama_index/status/1781837902139551920) 上查看该项目。

- **增强记忆的 ColBERT 检索 Agent**：针对在 RAG 中嵌入对话历史的挑战，分享了构建 **具有记忆功能的 ColBERT 驱动检索 Agent** 的指南，这标志着向个性化对话助手迈进。在 [Twitter](https://twitter.com/llama_index/status/1782086279498539330) 上有关于此话题的进一步探讨。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1230786732425875476)** (205 messages🔥🔥):

- **关于属性错误的困惑**：一位用户在尝试打印代码片段中的 `resp.chat.messages` 时遇到了 `AttributeError`，并寻求帮助以了解 `ChatResponse` 对象的正确属性。
- **LlamaIndex 中的本地与 OpenAI 对比**：讨论在 LlamaIndex 的不同功能中，使用 Ollama 或 LM Studio 等本地 LLM 实现来替代默认的 OpenAI 模型。
- **管理输出详细程度**：一位用户询问如何防止与批处理相关的输出干扰 Jupyter notebook 的执行结果，讨论随后转向了控制 logging 设置。
- **排查 VectorStoreIndex 查询结果**：一位用户寻求对 JSON 文件进行索引，并询问如何利用 metadata 来改进搜索结果，收到了关于 auto-retrieval 和链接 nodes 的建议。
- **处理文件加载错误**：有人询问如何在 `SimpleDirectoryReader` 中处理单个文件加载异常，而无需捕获 STDOUT 或导致所有文件导入失败。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:19530",">未找到标题</a>: 未找到描述</li><li><a href="https://ts.llamaindex.ai/modules/llms/#azure-openai">大语言模型 (LLMs) | LlamaIndex.TS</a>: LLM 负责阅读文本并针对查询生成自然语言响应。默认情况下，LlamaIndex.TS 使用 gpt-3.5-turbo。</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/agents/">Agents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/?h=rag+cli#customization">RAG CLI - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/docstore/FirestoreDemo/">Firestore Demo - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndex_auto_retriever/?h=auto">从 Weaviate 向量数据库进行自动检索 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程 (本地模型) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/">Chat Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#vector-store-index>).">索引与嵌入 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/portkey/?h=portkey)">Portkey - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/deploying/query_engine/usage_pattern#get-started>).">使用模式 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/7b52057b717451a801c583fae7efe4c4ad167455/llama-index-integrations/vector_stores/llama-index-vector-stores-milvus/llama_index/vector_stores/milvus/base.py#L162">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-milvus/llama_index/vector_stores/milvus/base.py at 7b52057b717451a801c583fae7efe4c4ad167455 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/callbacks/TokenCountingHandler/?h=token">Token 计数处理器 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: 为优化 RAG 解析文件</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号来为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/pull/13009">由 logan-markewich 修复 qdrant 检查现有集合时的 bug · Pull Request #13009 · run-llama/llama_index</a>: 从可能存在的集合中获取信息时的一个小 bug</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline+tool">围绕 Query Pipeline 构建 Agent - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 2 7B - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/localai/">LocalAI - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_documents#metadata>)">使用 Documents - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/data_connectors/PathwayReaderDemo#create-the-document-indexing-pipeline>).">Pathway Reader - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/tree_summarize/?h=tree+summarize">树状总结 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/querying/querying#querying>)">查询 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/starter_example_local#query-your-data>)">入门教程 (本地模型) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/evaluation/UpTrain#create-a-query-engine-using-llamaindex>).">如何在 LlamaIndex 中使用 UpTrain - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1231221804518080615)** (4 条消息): 

- **简化 Infini Attention**: 一位成员发布了一篇关于 **Infini Attention** 及其在生成式 AI 领域潜在应用的解释性帖子。他们在 [LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6) 中分享了他们的见解。

- **AI 融资追踪表已添加数据**：**AI Raise Tracking Sheet** 已更新，包含了按城市划分的融资和公司分布。这些信息可以在共享的 [Google spreadsheet](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121) 中访问和查看。

- **通过推文庆祝 AI 分布**：一条推文重点介绍了清理并展示过去一年 AI 融资和公司增长地理分布的工作。可以通过提供的 [Twitter 链接](https://x.com/WangUWS/status/1782069636030165106) 关注相关讨论。

- **FireCrawl 和 LlamaIndex 增强 Markdown 功能**：一篇文章讨论了 FireCrawl 与 LlamaIndex 的集成，通过 Markdown 就绪特性增强了 LLM 的潜力。该进展的详细内容见 [Medium](https://medium.com/ai-advances/unleash-the-potential-of-llm-ready-markdown-firecrawl-and-llamaindex-integration-243e494a9eb8)。

- **介绍 WhyHow.AI 的 Knowledge Graph SDK 更新**：WhyHow.AI 宣布对其 Knowledge Graph SDK 进行升级，支持 Schema 控制的自动化知识图谱。这使用户能够从私有数据创建结构化图谱并与 RAG 系统集成；更多详情和参与信息请参见 [Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121">[FrontierOptic.com] AI 融资追踪 - 2024年4月21日 - 社区审查版</a>：封面 &lt;a href=&quot;http://FrontierOptic.com&quot;&gt;FrontierOptic.com&lt;/a&gt; AI 初创公司融资数据（自 2023 年 5 月起）- 社区审查版 &lt;a href=&quot;https://twitter.com/WangUWS&...</li><li><a href="https://x.com/WangUWS/status/1782069636030165106">Howe Wang (@WangUWS) 的推文</a>：为了庆祝 @HilaryDuff 在《Wake Up》中演唱“可能是纽约，也许是好莱坞和藤街，伦敦，巴黎，也许是东京” 20 周年。我清理了 AI 热潮列车数据的地理位置...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1230861344278839486)** (75 条消息🔥🔥): 

- **AI 微调趋势**：一位成员分享了他们在 **Mixtral** 或 **Llama** 等模型上使用通用指令数据集进行微调的经验。他们强调，由于微调所需的数据集规模较小，学习速度非常快。

- **Llama3 在 Groq 上的性能赞誉**：用户们正在夸赞 **Llama3** 在 **Groq** 硬件上使用时令人印象深刻的速度，表明了高性能以及在实践中使用此配置的热情。

- **在 Windows 上排除 OI 故障**：有人提出了在 Windows 平台上使用 **Open Interpreter (OI)** 时遇到的困难，随后一位社区成员分享了一个 [GitHub issue 线程](https://github.com/OpenInterpreter/open-interpreter/issues/1185)，详细说明了安装过程中遇到的 Bug。

- **将 Open Interpreter 与 Groq API 集成**：社区成员已确认成功将 **OI** 与 **Groq API** 配合使用，并利用了同行提供的教程和示例命令。

- **探索 OI 的本地模型能力**：在一次直播之后，讨论围绕着将本地模型与 **OI** 结合使用展开，特别是发现了一些 Bug 和有效用法，例如通过特定命令绕过 function calling Bug，以及在本地使用 **Llama 3 8b** 的性能优势。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pastebin.com/ugNMQ57v">▌ OS Control enabled&gt; open notepad and write &quot;hello&quot; Let&#039;s start by try - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1185">Bug when fresh install and new start · Issue #1185 · OpenInterpreter/open-interpreter</a>: 描述运行时的 Bug。此警告显示在 interpreter /opt/conda/lib/python3.11/site-packages/pydantic/_internal/fields.py:151: UserWarning: Field &quot;model_id&quot; has conflict with prote...</li><li><a href="https://github.com/ishank26/posts/blob/main/llama3_new.pdf">posts/llama3_new.pdf at main · ishank26/posts</a>: 资源、想法和笔记。通过在 GitHub 上创建账户为 ishank26/posts 的开发做出贡献。</li><li><a href="https://www.youtube.com/live/KR9aJyjdtts?si=103CLVSdpUGRQoYz&t=3409">Future of Coding Jobs? + Open Interpreter w/ Gemini + more</a>: 笔记与日程：https://techfren.notion.site/Techfren-STREAM-Schedule-2bdfc29d9ffd4d2b93254644126581a9?pvs=40:00 - 简介 5:05 - SWE 工作安全吗？28:01 - 我的...</li><li><a href="https://www.youtube.com/watch?v=FXCaJ3Ga9TE">How to use Open Interpreter cheaper! (LM studio / groq / gpt3.5)</a>: 第一部分与简介：https://www.youtube.com/watch?v=5Lf8bCKa_dE0:00 - 设置 1:09 - 默认 gpt-4 2:36 - 快速模式 / gpt-3.5 2:55 - 本地模式 3:39 - LM Studio 5:5...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1213">Update local profile so it doen&#39;t use function calling by Notnaton · Pull Request #1213 · OpenInterpreter/open-interpreter</a>: 将 model 设置为 gpt4 会导致使用 function calling。大多数 LM Studio 模型不支持 function calling，导致无法工作。描述你所做的更改：引用任何相关的 issue（例如 &quot;...</li><li><a href="https://pastebin.com/b0bwxmzm">(oi) C:\Users\ivan&gt;interpreter --api_base &quot;https://api.groq.com/openai/v1&quot; --api - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1204">Bump version of tiktoken by minamorl · Pull Request #1204 · OpenInterpreter/open-interpreter</a>: 描述你所做的更改：提升了 tiktoken 的版本，因为构建过程由于某种原因损坏了。此 PR 修复了损坏的过程。引用任何相关的 issue（例如 &quot;Fixes #000&quot;）：...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/986">Jupyter export magic command by tyfiero · Pull Request #986 · OpenInterpreter/open-interpreter</a>: 描述你所做的更改：添加了一个 %jupyter 魔术命令，用于将当前会话导出为 Jupyter Notebook 文件，你可以在 Google Colab 中运行它。引用任何相关的 issue（例如 &quo...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1231154192153055262)** (18 messages🔥): 

- **Groq 芯片遇上 Llama 3**：一位成员报告了使用 **Groq** 芯片测试 **Llama 3 70b** 的情况，并认为其前景广阔，尽管他们承认测试时间有限。
- **关于 O1 和 Open Interpret 支持的混淆**：一位成员透露了关于 **O1** 和 **Open Interpret** 在 **Groq** 兼容性方面的混淆；澄清了他们指的是 Open Interpret，并指出 O1 目前仅支持 **OAI** 的云端选项。
- **大型 Llama 模型的稳定性问题**：有人对 **Llama 3 70b** 相对于较小的 8b 版本的稳定性表示担忧，认为大型模型更容易出现不稳定的情况。
- **O1 的 Windows 客户端问题**：用户在 Windows 上运行 **O1** 时遇到问题，报告显示 Windows 客户端本身可能存在潜在问题。
- **M1 Macbooks 上的空格键故障**：**M1 Macbooks** 上的 O1 语音识别出现问题，按下 **空格键** 会输入空格而不是开始录音，建议的修复方案（如 `brew install ffmpeg`）未能解决问题。另一位用户建议确保权限正确，并提到使用 **conda** 安装不同 Python 版本的权宜之计。
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1230779152165109790)** (64 messages🔥🔥):

- **Cohere 快速入门 MySQL 连接器协助**：讨论重点在于如何在不使用 Docker 的情况下将 **MySQL** 与 Cohere LLMs 集成，旨在直接从本地数据库获取答案。实现细节上似乎存在困惑，文中提供了一个 [GitHub 仓库](https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql) 的参考代码链接，另一位用户则提到了对文档和功能过时的担忧，指出 `create_connector` 无法正常工作。

- **探索 Command R 商业用途的限制**：一位用户询问了在边缘设备上将 **Command R (以及 Command R+)** 用于商业目的的情况，并被告知在 **CC-BY-NC 4.0** 许可证下这是不允许的，该许可证仅限非商业用途。

- **AI 初创公司创始人寻求人才**：一位 AI 初创公司的创始人正在寻求模型微调和语音模型方面的协助，表示倾向于寻找在 AI 研究和 LLMs 方面有经验的人才。创始人为感兴趣的人士提供了通过电子邮件或 [LinkedIn](https://www.linkedin.com/in/vaibhav-logar) 进行联系的渠道。

- **Cohere 实习被拒后的求职策略**：一位用户分享了未获得 Cohere 实习机会的失望心情，并寻求关于寻找 ML/软件岗位的建议。几位成员贡献了想法，包括申请没有公开实习列表的公司、参与开源项目、利用学校网络以及参加招聘会。

- **即将举行的 ML-Maths 讲座预告**：发布了一则关于 **Dr. Matthew Bernstein** 即将举行的讲座公告，主题为变分自编码器 (VAEs) 及其在单细胞基因组学中的应用，并为有意参加者分享了 Google Meet 链接。[这是讲座链接](https://meet.google.com/vhz-wokb-gob)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/sql-generation-ai-autonomous.html#GUID-3721296F-14A1-428A-B464-7FA25E9EC8F3">Using Oracle Autonomous Database Serverless</a>：Oracle Autonomous Database Select AI 允许你使用自然语言查询数据。</li><li><a href="https://docs.cohere.com/docs/creating-and-deploying-a-connector">Creating and Deploying a Connector - Cohere Docs</a>：未找到描述</li><li><a href="https://drive.google.com/file/d/11TiGQ-JxqmLQ-TJ24Jui8V9kXsI6QZld/view">Ken&#39;s Resume.pdf</a>：未找到描述</li><li><a href="https://github.com/cohere-ai/quick-start-connectors/tree/main/mysql">quick-start-connectors/mysql at main · cohere-ai/quick-start-connectors</a>：此开源仓库提供了将工作场所数据存储与 Cohere 的 LLMs 集成的参考代码，使开发者和企业能够执行无缝的检索增强生成 (RAG)...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1231240985791696980)** (5 条消息): 

- **匹配 AI 的开源代码**：一个利用 **@cohere Command R+** 并集成 **@stanfordnlp DSPy** 和 **@weaviate_io Vector store** 的匹配应用已开源。鼓励开发者[尝试代码](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ)、提供反馈并为 GitHub 仓库做贡献。

- **寻求网页爬取技术的进展**：一位成员讨论了使用 **gpt-4-turbo** 构建通用网页爬虫以识别（选择器、列）对的挑战，特别是在处理网页过滤器的点击和选择输入元素时遇到困难。
  
- **思考 AI “越狱”的伦理**：在关于 AI “越狱 (jailbreaks)” 的对话中，一位成员反思了它们创建智能 Agent 的潜力，并暗示这可能导致这些 Agent 产生无意间的负面行为，例如使用不当语言。

- **新的 Prompt IDE 寻求反馈**：**Prompt Mixer**（一款用于创建和评估 Prompt 的桌面应用程序）的创作者分享了他们的项目链接 [www.promptmixer.dev](https://www.promptmixer.dev/)，并邀请大家提供反馈以改进该工具。该工具提供自动版本控制以及与 AI 数据集集成等功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ">来自 Anmol Desai (@anmol_desai2005) 的推文</a>：我们做到了。代码终于开源了。请尝试一下，我们渴望得到反馈。@weaviate_io @stanfordnlp @cohere @1vnzh @CShorten30 ↘️ 引用 Muratcan Koylan (@youraimarketer) ...</li><li><a href="https://www.promptmixer.dev/">Prompt Mixer. 企业的 AI 开发工作室</a>：一个供经理、工程师和数据专家协作开发 AI 功能的工作空间。
</li>
</ul>

</div>
  

---

**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1231910638087835669)** (1 条消息): 

- **寻找挪威 Cohere 顾问**：一名成员正在寻找有 Cohere 经验的挪威公司（最好是顾问），为新项目提供第三方参考或咨询。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1231120994056273921)** (21 条消息🔥): 

- **在消费级 GPU 上本地运行 AI**：一位用户强调了在**笔记本 Vega iGPU** 上开箱即用运行硬件支持架构 (**HSA**) 的成功经验，并考虑使用 **HIP** 编译器和 **OpenCL**（可能是 **Rusticl**）作为替代方案。
- **关于 tinygrad 中隐式层的咨询**：一位用户询问是否有人在 **tinygrad** 中实现过**隐式层**（**implicit layers**）的经验，例如通过优化问题进行微分。
- **tinygrad 潜在的云服务**：受一篇关于硬件公司转向服务模式的文章启发，引发了关于 **tinygrad/box/chip** 是否会演变为**云服务**的讨论。一些成员希望它能保持作为赋能个人用户的工具，而不是转向云端依赖。
- **本地 AI 与云端 AI 的辩论**：参与者就**本地与云端 AI** 的优劣和未来潜力展开了辩论。观点从支持用户控制硬件、开发家用 **TinyBox** 等模型，到承认消费级硬件在运行最先进模型方面的局限性以及集中计算能力的优势。
- **George Hotz 发布周会公告**：**George Hotz** 概述了即将召开的会议主题，包括 **MLPerf 进展**、**KFD/NVIDIA 驱动**、新 **NVIDIA CI** 计划、**文档/开发者体验**、**调度器改进**，以及关于代码库 **7500 行限制**的讨论。他还提醒会议对所有人开放旁听，但发言权仅限于 reds 及以上级别。
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1230994188347113503)** (38 条消息🔥): 

- **调试 Einsum 精度问题**：一位成员在将模型从 PyTorch 移植到 tinygrad 时遇到了奇怪的错误，`einsum` 操作的结果出现细微差异，导致模型下溢为 NaN 值。有人建议这可能是浮点数问题，并询问 tinygrad 中的 `Tensor.numpy()` 是否会转换为 float64，另一位成员澄清说除了 bf16 外，它会返回相同的类型。

- **排查 ROCm 设置和段错误**：一位成员在 ROCm 环境下设置 tinygrad 时遇到段错误（segfaults），即使在新的 6.1 版本发布后依然存在，并询问是否有解决这些问题的文档。

- **错误信息与 GPU 驱动不匹配**：讨论围绕如何让 tinygrad 的错误信息更具参考价值，特别是当 CUDA 驱动版本低于 CUDA 库版本时。然而，有人指出除非 CUDA API 提供特定的消息，否则在代码库中验证和维护此类改进将非常困难。

- **tinygrad Master 分支的稳定性**：针对关于 master 分支可靠性的提问，George Hotz 确认 `master` 分支应该是稳定的，并表示他们的持续集成（CI）在维护稳定性方面非常有效。

- **tinygrad 原地操作机制**：讨论了 tinygrad 如何在不产生计算图环路的情况下处理原地操作（in-place operations），并引用了近期关于该主题的一次重大讨论。建议查看 GitHub 上的 `assign` 方法以及 Discord 和 GitHub 之前的讨论以获取更多见解。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">ShapeTracker 如何工作</a>：tinygrad 教程</li><li><a href="https://meta.ai">Meta AI</a>：使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型并使用 Emu...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/37f8be6450b6209cdc9466a385075971e673c653/tinygrad/tensor.py#L169">tinygrad/tinygrad/tensor.py (GitHub)</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1230825603066888283)** (1 条消息):

- **调制 Mixtral 秘方**：一位成员讨论了 **Mixtral 训练**中关于 "*router_aux_loss_coef*" 参数可能存在的疏忽。他们推测将该参数从 **0.001** 调整为 **0.1 或 0.000001** 是否可能是所需的“秘方”，并对当前设置的有效性提出了质疑。
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1230798778642989087)** (9 messages🔥): 

- **探索为捷克语添加 Token**：一位成员正在考虑为捷克语支持添加数千个 Token。初步实验表明，在不显著破坏英语能力的情况下扩展 tokenizer 是可行的。

- **提及 Occiglot 项目**：以多语言支持工作闻名的 *Occiglot 项目*在讨论中被提及，作为一个潜在的资源或感兴趣的社区。

- **揭穿推理速度的迷思**：澄清了减少词表 Token 并不会加快推理速度——受影响的是内存开销。

- **DiscoLM 德语版进入实验阶段**：宣布了基于 Llama3 的 DiscoLM 实验性德语版本，并提供了指向另一个 Discord 频道中演示的链接。

- **创新 CRM 聊天机器人功能**：一位成员描述了一种通过对功能进行分组，并可能针对不同的任务“组”使用不同类型的底层模型，从而使 CRM 内的聊天机器人集成在经济上更可行的方法，并询问是否存在支持此类功能的 *langchain* 等库。
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1230799145070235759)** (49 messages🔥): 

- **等待强大的 70B 模型**：成员们对模型的 **70B 版本**表现出极大的热情，但尚未讨论其可用性的具体细节。
- **LLM 的德语能力**：成员们正在比较不同 LLM 版本的**德语能力**，指出 **Llama3** 可能需要针对德语进行更多微调，且 **instruct 变体**即使在德语提示词下也不会自动以德语回复。
- **发布前的私密性考虑**：对话表明，新模型变体被刻意保持私密，以便在公开发布实验版本之前进行**彻底测试**。
- **在德语环境下训练 Llama3 的挑战**：成员们分享了他们在**训练 Llama3** 以提高其德语能力方面的经验和挑战，提到了与 **Mixtral 模型**相比语法较差的问题，以及导致模型输出末尾出现随机 Token 的 **tokenizer 问题**。
- **实验 Llama 3 DiscoLM 德语版**：讨论了一个基于 Llama 3 的实验性 **DiscoLM 德语模型**（[Llama 3 DiscoLM German 8b v0.1 Experimental](https://huggingface.co/DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental)），并提供了模型和演示链接，指出输出末尾会出现特殊 Token 的问题，以及在与其它模型对比时 **RAG 评估**结果参差不齐。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/maxidl/Mistral-7B-v0.1-capybara-orpo-en-de">maxidl/Mistral-7B-v0.1-capybara-orpo-en-de · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental">DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jvh/whisper-base-quant-ct2/">jvh/whisper-base-quant-ct2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/primeline/whisper-tiny-german">primeline/whisper-tiny-german · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/aisak-ai/aisak-listen">aisak-ai/aisak-listen · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1230796276833194045)** (47 messages🔥):

- **寻求 LangChain Endpoint 协助**：一位成员询问如何定位他们的 LangChain endpoint，这是与该框架功能交互的关键要素。
- **调查 Firefunction 延迟波动**：一位用户报告在 LangChain 中使用 firefunction 时出现延迟不一致的问题，不同设备之间的差异显著，引发了关于潜在原因或解决这些差异的方案的讨论。
- **处理 OCR 和实体关系提取**：成员们讨论了对发票等文档进行 OCR 识别并提取实体关系的策略，其中一位提到了使用 docTR 并询问后续步骤。
- **构建基于 LangChain 的智能 AutoGPT**：考虑到 Agent 设计的复杂性，一位成员思考 LangGraph 或 LangChain 的 "Plan and execute" agent 是否更适合创建类似 AutoGPT 的通用 Agent。
- **将问题映射到 Metadata 以进行检索**：一位成员就在 LangChain 生态系统中何处处理 LLM 评估寻求建议，特别是当将问题映射到 Metadata 中的可过滤类别时，这可能会影响 Agent, Tool 或 Chain 的设计。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://api.js.langchain.com/classes/langchain_anthropic.ChatAnthropic.html#apiUrl">ChatAnthropic | LangChain.js - v0.1.34</a>：未找到描述</li><li><a href="https://youtu.be/r-heqmMYNL0">学习 LLAMA 3 的工作原理：完整的初学者指南</a>：深入探索 LLAMA 3 模型的迷人世界，这是一种正在树立机器学习新标准的尖端 Transformer 架构。本指南...</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#function-calling>))">快速入门 | 🦜️🔗 Langchain</a>：在本指南中，我们将介绍创建调用 Tool 的 Chain 和 Agent 的基本方法。Tool 可以是任何东西——API、函数、数据库等。Tool 允许我们扩展功能...</li><li><a href="https://js.langchain.com/docs/integrations/chat/google_vertex_ai#vertexai-tools-agent>)">ChatVertexAI | 🦜️🔗 Langchain</a>：LangChain.js 支持将 Google Vertex AI 聊天模型作为集成。</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm#code-generation-chat-models>)">ChatVertexAI | 🦜️🔗 LangChain</a>：注意：这与 Google PaLM 集成是分开的。Google 已经...</li><li><a href="https://github.com/langchain-ai/langchain/issues/13442>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1230899322506313851)** (1 条消息): 

- **寻求 FstAPI 路由代码**：一位成员询问在海盗口吻 (pirate-speak) 场景下 **FstAPI** 路由的代码位置，但在 app 文件夹中未能找到，并请求解释。该查询在消息中未获得回复或解决。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1230864075312074762)** (7 条消息): 

- **介绍 Trip-Planner Bot**：一位成员展示了他们新的 **Trip-Planner Bot**，该机器人利用 **Bing maps API, OpenStreetMaps API** 和 **FourSquare API** 提供位置信息、景点和路线规划。查看 [GitHub 仓库](https://github.com/abhijitpal1247/TripplannerBot)了解更多详情。

- **使用 LLM Scraper 进行网页数据结构化**：新发布的 **LLM Scraper** 项目可以将网页转换为结构化数据，现已在 GitHub 上发布，欢迎贡献和 Star 支持。访问 [LLM Scraper](https://github.com/mishushakov/llm-scraper/) 项目页面了解更多。

- **在 Product Hunt 上为 AllMind AI 寻求支持**：一位成员请求支持，帮助他们的 **AllMind AI** 在 Product Hunt 上冲刺第一名。该工具提供实时市场数据，并声称在各项金融任务中表现优于主流 AI 模型。你可以通过访问 [Product Hunt 上的 AllMind AI](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst) 来支持他们。

- **WhyHow.AI 升级 Knowledge Graph SDK**：**WhyHow.AI** 宣布对其 Knowledge Graph SDK 进行重大升级，助力创建由 Schema 控制的自动化知识图谱。欲了解更多见解并获得加入 Beta 测试的机会，请阅读此处的 [Medium 文章](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3)。

- **对话主题追踪开发**：一位成员寻求关于实现实时对话主题、科目和任务追踪的建议，并询问是否有任何现有的开源项目或平台可以协助此项工作。他们正在寻找关于将聊天消息与主题关联，或在必要时创建新主题的指导。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst"> AllMind AI: Your Personal Stock Analyst - 具备实时市场数据和洞察力的 AI 财务分析师 | Product Hunt</a>：AllMind AI 是您的个人财务分析师，直接为您提供集中的、实时的、可操作的见解。我们的专有 LLM AllMind AI 可缩短 90% 的研究时间并降低 98% 的成本。W...</li><li><a href="https://github.com/mishushakov/llm-scraper/">GitHub - mishushakov/llm-scraper: 使用 LLM 将任何网页转换为结构化数据</a>：使用 LLM 将任何网页转换为结构化数据。通过在 GitHub 上创建一个账户来为 mishushakov/llm-scraper 的开发做出贡献。</li><li><a href="https://github.com/abhijitpal1247/TripplannerBot">GitHub - abhijitpal1247/TripplannerBot: 这是一个使用 LangChain 的 Streamlit 应用。它利用了 Bing maps API、OpenStreetMaps API 和 FourSquare API。</a>：这是一个使用 LangChain 的 Streamlit 应用。它利用了 Bing maps API、OpenStreetMaps API 和 FourSquare API。- abhijitpal1247/TripplannerBot
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1231989652353843331)** (1 条消息): 

- **剖析 Self-Querying Retriever**：一位成员详细阐述了 *Self-querying retriever* 如何利用 LLM 和 few-shot prompting 将自然语言查询转换为结构化查询。他们在博客文章中分享了关于优化 prompt 以改进查询的见解，详细介绍了该技术的内部工作原理，文章见 [使用 LangChain Self-Querying Retriever 进行公寓租赁搜索](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever)。

**提到的链接**：<a href="https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever">使用 LangChain 的 Self-Querying Retriever 构建公寓租赁搜索</a>：在这篇博客文章中，我们深入探讨了 LangChain 的 self-querying retriever 的功能，这是一个弥合自然语言与结构化数据检索之间鸿沟的强大工具。该检索器...

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1230783940663574548)** (41 条消息🔥): 

- **Llama3 Token 问题**：一位成员发现 *llama3 instruct 格式* 使用了不同的 token，并强调 `llamafile` 和 `llama.cpp server bin` 不支持当前的参数。这涉及到 instruct 格式，该格式似乎与现有基础设施不兼容，正如在 [LocalLLaMA subreddit](https://www.reddit.com/r/LocalLLaMA/) 中讨论的那样。
- **Llama 3 Chat Template 更新正在进行中**：GitHub 上的一个 pull request 旨在将 llama 3 chat template 添加到 `llama.cpp`，这是持续改进的迹象。对于关注更新的人，可以在 [这里](https://github.com/ggerganov/llama.cpp/pull/6751) 找到该 PR。
- **Llama 3 8B 量化版本发布**：针对一项咨询，一位成员承诺在一天内发布 llamafile 上的 llama 3 8B 量化版本，并迅速提供了一个 Hugging Face 链接用于测试新的可执行权重。
- **Llama 3 70B 进展**：成员们透露 llama 3 70B llamafile 已经可用，但提醒注意一些小 bug，例如损坏的 stop token。据报道，这些 bug 正在被修复，并鼓励用户在广泛发布前协助测试。
- **Llamafiles 的适应性与问题**：关于在各种系统上运行 llamafiles 及其挑战的讨论出现了；Q2 量化级别在 M1 Pro 32GB 系统上的表现不如预期，且大家似乎达成共识，认为 llama 3 70B 的表现优于 8B。相比之下，8B 版本被描述为在 llamafile 上运行尚不理想。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1089876418936180786/1089876419926032399/1224854113674592286">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文本进行交流的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://huggingface.co/jartine/Meta-Llama-3-70B-Instruct-llamafile#hardware-choices-llama3-70b-specific">jartine/Meta-Llama-3-70B-Instruct-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jartine/Meta-Llama-3-70B-Instruct-llamafile/tree/main">jartine/Meta-Llama-3-70B-Instruct-llamafile at main</a>: 未找到描述</li><li><a href="https://huggingface.co/jartine/Meta-Llama-3-8B-Instruct-llamafile">jartine/Meta-Llama-3-8B-Instruct-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/.devops/main-vulkan.Dockerfile">llama.cpp/.devops/main-vulkan.Dockerfile at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c76n8p/comment/l06amy7/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6751">由 DifferentialityDevelopment 添加 llama-3 聊天模板 · Pull Request #6751 · ggerganov/llama.cpp</a>: 这只是简单地添加了 llama 3 聊天模板
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1230920966339821690)** (10 条消息🔥): 

- **扩大模型供应**: 社区对新模型尺寸充满期待，发布计划包括在约 5 万亿 token 上训练的 **100M, 500M, 1B 和 3B 模型**，旨在通过 olmo 缩放来取代当前的 pythia 套件。

- **Karpathy 的 AI 请求得到响应**: 似乎正在采取行动以符合 AI 社区的愿望，特别是满足 [Karpathy](https://twitter.com/karpathy) 的需求，这可能意味着根据他公开的偏好或建议来开发 AI 模型。

- **关注 SLMs 和小型视觉模型**: 一位成员表示，Sparse Latent Models (SLMs) 和小型视觉模型是目前最引人注目的项目。

- **对紧凑型强大模型的热情**: **MiniCPM** 的成功引起了轰动，表明社区对创建紧凑且强大的模型有着显著的兴趣。

- **AI 彻底改变基准测试**: 分享了一个推文链接，强调 **LLAMA 3 8B** 已经设定了一个令人印象深刻的标准，但即将推出的 **Phi-3 mini 4b, small 7b 和 medium 14b** 模型可能会凭借其出色的基准测试表现超越它，并且合成数据流水线（synthetic data pipelines）的贡献远超互联网数据。[查看推文](https://fxtwitter.com/dylan522p/status/1782461647497400324)。

- **模型鲁棒性没有捷径**: 表达了一个明确的立场——试图在模型训练中绕过正规流程会导致结果不佳，正如这句话所象征的，“你无法作弊”。

**提到的链接**: <a href="https://fxtwitter.com/dylan522p/status/1782461647497400324">来自 Dylan Patel (@dylan522p) 的推文</a>: LLAMA 3 8B 非常出色，但本周将被 Phi-3 mini 4b, small 7b, medium 14b 掩盖，其基准测试表现简直疯狂。合成数据流水线相比互联网数据有了巨大的改进...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1231364661522337823)** (9 条消息🔥): 

- **自动化评估与人工主导评估的辩论**: 一位成员正在更新他们笔记中的评估（Evals）部分，并质疑 **MMLU** 和 **BIGBench** 等自动化评估与 **ChatBotArena** 等耗时的人工评估相比的直接效用。
- **困惑度（Perplexity）与基于任务的评估**: 关于像 **AI2 的 Paloma** 这样基于困惑度的评估如何与基于任务的评估进行比较存在困惑。有人提出疑问，Paloma 这样的困惑度基准测试是公开的基准测试，还是仅仅在训练期间使用的指标。
- **基准测试分类**: 讨论中一位成员表示喜欢 **MT Bench 论文**中的基准测试分类，尽管目前尚不清楚 Paloma 在该分类中的位置。
- **评估分类法是流动的**: 另一位成员也对基准测试分类表示赞赏，但指出该领域发展迅速，目前还没有一个大家达成共识的单一分类法。
- **基于困惑度指标的效用**: 概念得到了澄清，共识是基于困惑度的评估更类似于训练期间的检查点指标（checkpoint metrics），而不是供完成的模型进行竞争的指标。

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1232028192353816709)** (11 条消息🔥): 

- **Discord 的隐藏宝藏**：尽管拥有 **1.3万名免费订阅者**，且其中 **250人有资格**加入 Discord，但只有不到 50 人抓住了加入社区的机会，这表明许多人可能并未意识到这一功能。
- **推广是关键**：针对 Discord 参与度较低的情况，官方正在采取行动使加入机会更加**“显眼”**，并计划进行**季度功能推介**以提高曝光率，类似于 Ben Thompson 所采用的方法。
- **征集社区反馈**：一位成员分享了他们对“走向多元化路线图”论文的**深度分析**，发布了一个 [Typefully 链接](https://typefully.com/t/AstZhn4)，并在最终定稿前征求意见。
- **潜水的价值**：一位成员表达了对社区的赞赏，表示尽管大部分时间都在潜水，但他们**“非常喜欢阅读这些对话和链接”**，这表明非活跃成员也能从 Discord 内容中发现价值。

**提及的链接**：<a href="https://typefully.com/t/AstZhn4">未找到标题</a>：未找到描述

  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1231001744490889256)** (5 条消息): 

- **新 RLHF 论文预警**：一位成员分享了一篇名为 [Reinforcement Learning From Human Feedback](https://arxiv.org/abs/2404.12358) 的新论文，该论文讨论了传统 RLHF 方法与较新的 Direct Preference Optimization (DPO) 方法之间的差异。
- **理论结合实践**：上述论文在 token 级 MDP 中理论推导了 DPO，使其与标准的 RLHF 方法保持一致，并确认其满足 Bellman equation（贝尔曼方程）。
- **与作者讨论**：一位成员提到，在论文发表前几周，曾与作者之一 Rafael 讨论过该论文的内容。
- **对创新概念的认可**：该成员对论文表示了极大的热情，表明对论文提供的理论和实证见解持积极态度。

**提及的链接**：<a href="https://arxiv.org/abs/2404.12358">From $r$ to $Q^*$: Your Language Model is Secretly a Q-Function</a>：Reinforcement Learning From Human Feedback (RLHF) 是最新一代生成式 AI 模型成功的关键。针对经典 RLHF 流程的复杂性……

  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1230941556752781453)** (3 条消息): 

- **录音期待感升温**：社区正热切期待录音发布，更新消息称录音将在 **1-2 周内**提供。
- **社区强烈要求内容**：社区对最新录音的需求显而易见，使用了 **CLAMMORING**（喧嚣/强烈要求）一词来描述这种期待。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1230893909924905020)** (7 条消息): 

- **动态幽默**：一位成员分享了一个幽默的 [Tenor GIF](https://tenor.com/view/falling-falling-down-stairs-stairs-meme-funny-gif-21363126)，内容是有人从楼梯上摔下来，并提到 Tenor.com 可以根据浏览器的语言设置进行翻译。
- **Llama 3 令人印象深刻的表现**：在 AI 模型讨论中，**Llama 3** 被指出在 arena（竞技场）中表现优于 Opus，尽管它只是一个 70B 模型。
- **考虑误差范围**：一位成员强调了在评估模型性能时**误差范围**（error bounds）的重要性。
- **风格与智能之争**：有一场关于什么促成了模型有效性的讨论：是**风格元素**还是**实际智能**。
- **Meta.ai 的 Imagine 备受赞赏**：**meta.ai imagine** 平台被赞为“疯狂”，引发了索要示例以展示其能力的请求。

**提及的链接**：<a href="https://tenor.com/view/falling-falling-down-stairs-stairs-meme-funny-gif-21363126">Falling Falling Down Stairs GIF - Falling Falling Down Stairs Stairs - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


**LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1232022132842561688)** (3 条消息): 

- **Azure 的 OpenAI 延迟困扰**：一位成员对 Azure OpenAI 的**高延迟**问题表示沮丧，并引用了一个极端案例，某些请求的耗时达到了**令人震惊**的 20 分钟。
- **Azure 速率限制问题困扰开发者**：同一位成员报告称在 **Azure 上不断受到速率限制（rate limited）**，甚至 15 秒内发送 2 个请求也会触发限制。这导致他们不得不实施速率限制退避策略（rate limit backoff strategy）。

**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1230862081893466114)** (6 messages): 

- **Databricks 推出 GPU 和 LLM 优化**：Databricks 宣布了其针对 [Model Serving 的新 GPU 和 LLM 优化支持](https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving)的公开预览版，允许在 Lakehouse Platform 上部署 AI 模型。这款 Serverless GPU 服务产品无需任何额外配置即可为 LLM 服务优化模型。
- **对高级服务成本的担忧**：一位成员开玩笑地表示 Databricks 提供的新功能可能很昂贵，说道：“*我敢打赌这会让我倾家荡产😅*”。
- **发布 LLM 微调指南**：分享了一份微调预训练 LLM 的操作指南，概述了使用 [Modal 的微调文档](https://modal.com/docs/examples/llm-finetuning)针对特定任务调整模型权重的步骤。该指南附带了推荐的优化方案，如 LoRA adapters、Flash Attention、Gradient checkpointing 和 DeepSpeed。
- **低预算的 Serverless 托管**：提供了廉价的 Serverless 托管选项，并指向了一个用于设置 LLM 前端的 [GitHub 链接](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html)（位于 modal-examples 仓库中）。
- **确认有用资源**：一位成员确认分享的 Serverless 推理资源正是他们所寻找的，并简单地表达了感谢：“*thanks!!!*”。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://modal.com/docs/examples/llm-finetuning">在几分钟内微调 LLM（含 Llama 2, CodeLlama, Mistral 等）</a>：厌倦了 Prompt Engineering？微调通过调整模型权重以更好地适应特定任务，帮助你从预训练 LLM 中获得更多收益。这份操作指南将帮助你利用基础模型...</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html">modal-examples/06_gpu_and_ml/llm-frontend/index.html at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://www.databricks.com/blog/announcing-gpu-and-llm-optimization-support-model-serving">使用 Databricks Model Serving 部署私有 LLM | Databricks 博客</a>：在完全控制数据和模型的情况下部署生成式 AI 模型。
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1232013872261758996)** (2 messages): 

- **寻找用于蓝图分析的 AI**：一位成员询问了用于解释蓝图的 **AI 模型或方法**，特别是为了追踪 PDF 图纸中的管道系统（ductwork）。
- **AI 在建筑领域起飞**：另一位成员分享了关于 AI 在建筑公司中被用作 **“预检”（preflight）工具** 的见解，用于在施工前识别潜在问题和规范违规，尽管它尚未应用于蓝图创建阶段。
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1231045914974621756)** (3 messages): 

- **Llama 3 发布**：SimonW 升级了 **llm-gpt4all 插件** 以支持 **Llama 3 8B Instruct**，使用户能够在拥有 8GB RAM 的机器（如 M2 MacBook Pro）上运行大型模型。更新后的插件可以使用命令 `llm install --upgrade llm-gpt4all` 安装。

- **插件发布说明**：**llm-gpt4all 插件** 0.4 版本现已发布，正如 [GitHub release](https://github.com/simonw/llm-gpt4all/releases/tag/0.4) 中所述，增加了对包括 Llama 3 8B Instruct 在内的新模型支持。

- **展示 Llama 3 的能力**：SimonW 在其博客中强调了 Llama 3 作为目前最好的开源许可模型的声誉，并深入探讨了在本地运行 Llama 3 模型以及使用托管服务的方法。欲了解更多见解，用户可以访问 [Simon 关于 Llama 3 的博客文章](https://simonwillison.net/2024/Apr/22/llama-3/)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Apr/22/llama-3/">使用 LLM 从终端访问 Llama 3 的选项</a>：Llama 3 已于周四发布。早期迹象表明，它现在是最好的开源许可模型——Llama 3 70b Instruct 在 LMSYS arena 中并列第 5 位……</li><li><a href="https://github.com/simonw/llm-gpt4all/releases/tag/0.4">Release 0.4 · simonw/llm-gpt4all</a>：升级到最新的 gpt4all (2.5.1)，增加了对多个新模型的支持，包括... llm -m Meta-Llama-3-8B-Instruct &quot;say hello with a lot of words&quot; 来运行新的 Llama 3 8B Instruct 模型...
</li>
</ul>

</div>
  

---

**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1230831380544487517)** (1 条消息): 

- **面向初学者的 LLAMA 3 解密**：一位成员分享了一个 [YouTube 视频](https://youtu.be/r-heqmMYNL0)，标题为 "Learn How LLAMA 3 Works Now: The Complete Beginner’s Guide"，旨在为初学者解释 **LLAMA 3 模型** 及其在 Machine Learning 中的重要性。描述中承诺将深入探讨 LLAMA 3 的 Transformer 架构。

**提到的链接**：<a href="https://youtu.be/r-heqmMYNL0">Learn How LLAMA 3 Works Now: The Complete Beginner’s Guide</a>：深入探索 LLAMA 3 模型的迷人世界，这是一种正在树立 Machine Learning 新标准的尖端 Transformer 架构。本指南...

---