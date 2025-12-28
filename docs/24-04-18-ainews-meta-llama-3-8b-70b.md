---
companies:
- meta-ai-fair
- stability-ai
- boston-dynamics
- microsoft
- mistral-ai
- hugging-face
date: '2024-04-19T04:28:01.540342Z'
description: '**Meta** 发布了 **Llama 3** 模型的部分版本，包括 **8B** 和 **70B** 变体，而 **400B** 变体仍在训练中，被誉为首个达到
  GPT-4 级别的开源模型。**Stability AI** 推出了 **Stable Diffusion 3 API**，模型权重即将发布，其真实感可与 **Midjourney
  V6** 媲美。**波士顿动力 (Boston Dynamics)** 展示了全电动人形机器人 **Atlas**，**微软** 则推出了 **VASA-1**
  模型，能在 RTX 4090 上以 40fps 的速度生成逼真的说话人脸。


  作为 OpenAI 的欧洲竞争对手，**Mistral AI** 正在寻求 50 亿美元的融资，其 **Mixtral-8x22B-Instruct-v0.1**
  模型在 64K 上下文基准测试中达到了 100% 的准确率。在 AI 安全讨论方面，前 OpenAI 董事会成员 **Helen Toner** 呼吁对顶级 AI
  公司进行审计，**摩门教会**也发布了 AI 使用原则。


  新的 AI 开发工具包括：用于扩散模型的 **Ctrl-Adapter**、用于合成数据集流水线的 **Distilabel 1.0.0**、利用大语言模型进行数据清洗的
  **Data Bonsai**，以及使用行为树构建大语言模型智能体的 **Dendron**。此外，网络梗图（Memes）也展现了 AI 发展中的幽默与文化参考。此次发布的
  **Llama 3** 模型具有更强的推理能力、12.8 万个 token 的词汇表、8K token 的序列长度，并采用了分组查询注意力（GQA）机制。'
id: b447f626-a7a1-46cd-b9fc-b62c3b585f06
models:
- llama-3-8b
- llama-3-70b
- llama-3-400b
- stable-diffusion-3
- mixtral-8x22b-instruct-v0.1
- vasa-1
original_slug: ainews-to-be-named-5820
people:
- helen-toner
title: 'Meta Llama 3 (8B, 70B)


  *(注：8B 和 70B 分别代表 80 亿和 700 亿参数)*'
topics:
- transformer
- tokenization
- model-training
- benchmarking
- robotics
- natural-language-processing
- real-time-processing
- synthetic-data
- dataset-cleaning
- behavior-trees
- ai-safety
- model-accuracy
- api
- model-release
- humor
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月17日至4月18日的 AI 新闻。我们为您检查了 6 个 subreddits、[364 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)以及 27 个 Discord 社区（包含 395 个频道和 9849 条消息）。预计节省阅读时间（按 200wpm 计算）：**918 分钟**。

正如广泛预告的那样，[Meta 今天部分发布了 Llama 3](https://ai.meta.com/blog/meta-llama-3/)，包括 8B 和 70B 版本，但全场焦点是 400B 版本（仍在训练中），它被广泛誉为第一个 GPT-4 级别的 OSS 模型。

 
![image.png](https://assets.buttondown.email/images/a004405a-73b2-4d6e-9eae-2a2d8cf8927b.png?w=960&fit=max)
 

我们今天大部分时间都在旅途中，所以明天会补全剩余的所有评论，但可以前往 [HN](https://news.ycombinator.com/item?id=40077533) 查看最佳的实时报道。

---

**目录**

[TOC] 


---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**近期 AI 发展的关键主题**

- **Stable Diffusion 3 发布与对比**：Stability AI 发布了 [Stable Diffusion 3 API](https://stability.ai/news/stable-diffusion-3-api?utm_source=twitter&utm_medium=website&utm_campaign=blog)，模型权重即将推出。[SD3 与 Midjourney V6 的对比](https://www.reddit.com/r/StableDiffusion/comments/1c6iae0/sd3_vs_midjourneyv6/)结果褒贬不一，而[写实性测试展示了 SD3 的实力](https://www.reddit.com/gallery/1c6un6f)。Emad Mostaque [确认 SD3 权重将发布](https://i.redd.it/60wquhb3x2vc1.jpeg)在 Hugging Face 上，并附带 ComfyUI 工作流。

- **机器人与 AI Agent 的进展**：Boston Dynamics 展示了其[人形机器人 Atlas 的电动版本](https://www.youtube.com/watch?v=29ECwExc-_M)，具有令人印象深刻的灵活性。[Menteebot 是一款真人大小的 AI 机器人](https://www.yahoo.com/tech/menteebot-is-a-human-sized-ai-robot-that-you-command-with-natural-language-110052927.html)，可通过自然语言控制。Microsoft 的 [VASA-1 模型可生成栩栩如生的说话面孔](https://www.microsoft.com/en-us/research/project/vasa-1/)，在 RTX 4090 上可实现 40fps 的实时音频驱动。

- **新语言模型与基准测试**：[Mistral，这家欧洲的 OpenAI 竞争对手，正寻求 50 亿美元的融资](https://www.reuters.com/technology/frances-mistral-ai-seeks-funding-5-bln-valuation-information-reports-2024-04-17/)。他们的 [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 在 64K 上下文下以 100% 的准确率超越了开源模型。新的 [7B merge 模型结合了不同基础模型的优势](https://huggingface.co/Noodlz)。[Coxcomb，一个 7B 的创意写作模型](https://huggingface.co/N8Programs/Coxcomb)，在基准测试中表现良好。

- **AI 安全与监管讨论**：前 OpenAI 董事会成员 [Helen Toner 呼吁对顶尖 AI 公司进行审计](https://www.bloomberg.com/news/articles/2024-04-16/former-openai-board-member-calls-for-audits-of-leading-ai-companies)，以共享有关能力和风险的信息。[摩门教会发布了 AI 使用原则](https://newsroom.churchofjesuschrist.org/article/church-jesus-christ-artificial-intelligence)，指出了其益处和风险。

- **AI 开发工具与框架**：[Ctrl-Adapter 框架将控制适配到 diffusion 模型](https://v.redd.it/xugl158ya2vc1)。[Distilabel 1.0.0 支持使用 LLM 构建合成数据集流水线](https://github.com/argilla-io/distilabel)。[Data Bonsai 使用 LLM 清理数据](https://github.com/databonsai/databonsai)，并集成了 ML 库。[Dendron 使用行为树构建 LLM Agent](https://github.com/richardkelley/dendron)。

- **梗图与幽默**：一个[“理想 vs 现实”的梗图](https://i.redd.it/6t55ri6xvzuc1.jpeg)嘲讽了 AI 发展与未来愿景。[PS2 风格 LORA 中的 Snoop Dogg](https://i.redd.it/flx5vvp9qzuc1.png) 展示了 AI 梗图的潜力。[AI 版 Sans vs Frisk](https://i.redd.it/yado6g0y86vc1.jpeg) 用 AI 艺术重新构思了《传说之下》（Undertale）。一个[幽默的观点建议 AI 目前还没那么先进](https://i.redd.it/bu1lli1824vc1.png)。

---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在与 Haiku 合作进行聚类和流程工程。

以下是按要求格式提供的摘要：

---

**Meta Llama 3 发布**

- **Llama 3 模型发布**：[@AIatMeta](https://twitter.com/AIatMeta/status/1780997403979735440) 宣布发布 Llama 3 8B 和 70B 模型，提供了**改进的推理能力**，并为**同尺寸模型设定了新的 SOTA**。未来几个月预计将发布更多模型、功能和研究论文。
- **模型详情**：[@omarsar0](https://twitter.com/omarsar0/status/1780992539891249466) 指出 Llama 3 使用了**标准的 decoder-only transformer**、**128K token 词表**、**8K token 序列**、**grouped query attention**、**15T 预训练 token**，以及 **SFT、rejection sampling、PPO 和 DPO** 等对齐技术。
- **性能**：[@DrJimFan](https://twitter.com/DrJimFan/status/1781006672452038756) 将 Llama 3 70B 的性能与 Claude 3 Opus、GPT-4 和 Gemini 进行了对比，显示其**正接近 GPT-4 水平**。[@ylecun](https://twitter.com/ylecun/status/1780999981962342500) 也强调了 8B 和 70B 模型**强劲的 benchmark 结果**。

**开源 LLM 进展**

- **Mixtral 8x22B 发布**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1780602023203029351) 发布了 Mixtral 8x22B，这是一个拥有 **141B 参数（39B 激活）**、**多语言能力**、**原生 function calling** 和 **64K 上下文窗口**的开源模型。它为**开源模型设定了新标准**。
- **Mixtral 性能**：[@bindureddy](https://twitter.com/bindureddy/status/1780609164223627291) 指出 Mixtral 8x22B 具有**最佳性价比**，拥有**强劲的 MMLU 性能**，且具备通过微调**超越 GPT-4** 的潜力。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1780605940842021327) 强调了它的**数学能力**。
- **开源模型排行榜**：[@bindureddy](https://twitter.com/bindureddy/status/1780797091465527736) 和 [@osanseviero](https://twitter.com/osanseviero/status/1780717276771344895) 分享了开源模型排行榜，展示了**开源模型的快速进步和普及**。Llama 3 有望进一步推动这一进程。

**AI Agent 与 RAG (Retrieval-Augmented Generation)**

- **RAG 基础**：[@LangChainAI](https://twitter.com/LangChainAI/status/1780629875533181271) 与 @RLanceMartin 和 @freeCodeCamp 合作发布了一系列**解释 RAG 基础和高级方法的视频播放列表**。
- **Mistral RAG Agent**：[@llama_index](https://twitter.com/llama_index/status/1780646484712788085) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1780763995781378338) 分享了关于**使用 @MistralAI 新的 8x22B 模型构建 RAG Agent** 的教程，展示了文档路由、相关性检查和工具使用。
- **RAG 的忠实度**：[@omarsar0](https://twitter.com/omarsar0/status/1780613738585903182) 分享了一篇论文，**量化了 RAG 设置中 LLM 内部知识与检索信息之间的张力**，强调了在信息敏感领域部署 LLM 的影响。

**AI 课程与教育**

- **Google ML 课程**：[@svpino](https://twitter.com/svpino/status/1780657510518788593) 分享了 **300 小时的免费 Google ML 工程课程**，涵盖从入门到高级的各个级别。
- **Hugging Face 课程**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1780612212765200599) 宣布了一个**关于 Hugging Face 量化基础的新免费课程**，旨在让开源模型更易获得且更高效。
- **斯坦福 CS224N 人口统计**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1780640708497699186) 分享了**本季度 CS224N 615 名学生**的人口统计数据，显示了跨专业和跨级别的广泛代表性。

**其他**

- **Zerve 作为 Jupyter 的替代方案**：[@svpino](https://twitter.com/svpino/status/1780938523844968627) 建议 Zerve（一个**与 Jupyter 理念不同的基于 Web 的 IDE**）在许多用例中可能取代 Jupyter notebook。它具有针对 **ML/DS 工作流的独特功能**。
- **资本利得与通货膨胀**：[@scottastevenson](https://twitter.com/scottastevenson/status/1780941599335153788) 解释了**通货膨胀期间的资本利得税如何导致人们在没有实际收益的情况下被征税**，从而通过退休基金、住房和企业等资产影响中产阶级。

---

# AI Discord 回顾

> 摘要之摘要的摘要

**Llama 3 发布引发热潮**：Meta 发布了 **[Llama 3](https://llama.meta.com/llama3/)**（一个 8B 和 70B 参数的指令微调模型），在 AI 社区引起了极大关注。关键细节：

- 承诺提供**更强的推理能力**，并在各项任务中树立了“新的行业基准”。
- 可通过 [Together AI's API](https://together-ai.webflow.io/blog/together-ai-partners-with-meta-to-release-meta-llama-3-for-inference-and-fine-tuning) 等合作伙伴进行**推理和微调**，提供高达 350 tokens/sec 的速度。
- 对即将推出的 **400B+ 参数版本**充满期待。
- 一些人对**输出限制**表示担忧，认为这阻碍了开源开发。

**Mixtral 8x22B 重新定义效率**：新发布的 **[Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/)** 因其性能、成本效益以及在数学、编程和多语言任务中的专业化而备受赞誉。亮点包括：

- 通过稀疏 Mixture-of-Experts (MoE) 架构，在 141B 总参数中利用了 **39B 激活参数**。
- 支持 **64K token 上下文窗口**，实现精确的信息召回。
- 采用 **Apache 2.0 开源许可证**发布，并附带 Mistral 的自定义 tokenizer。

**分词器（Tokenizers）与多语言能力受到关注**：随着 Llama 3 和 Mixtral 等强大模型的出现，它们的分词器和多语言性能成为关注焦点：

- Llama 3 的 **128K 词汇量分词器**涵盖了 30 多种语言，但在非英语任务中可能表现不佳。
- Mistral 开源了其**支持 tool calls 和结构化输出的分词器**，以标准化微调过程。
- 关于**更大的分词器词汇量有利于多语言 LLM** 的讨论。

**缩放法则（Scaling Laws）与复现挑战**：AI 研究社区围绕缩放法则和具有影响力论文的可复现性展开了激烈辩论：

- [Chinchilla scaling paper](https://arxiv.org/abs/2404.10102) 的研究结果受到质疑，作者承认存在错误并开源了数据。
- 对于结果是证实还是反驳了**scaling laws** 的存在，各方持有不同观点。
- 呼吁在从有限数据进行推断时，应采用更符合实际的实验次数和更窄的置信区间。


**其他**

- **Llama 3 发布引发热潮与审视**：Meta 发布的 **[Llama 3](https://llama.meta.com/llama3/)**（包含 8B 和 70B 参数模型）引发了 AI 社区的广泛兴趣和测试。工程师们对其性能媲美前代 **Llama 2** 和 **GPT-4** 印象深刻，但也注意到了 128k token 上下文窗口等局限性。**[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536)** 和 **[Unsloth](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)** 等框架正在进行集成，量化版本也已出现在 **[Hugging Face](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF)** 上。然而，一些人对 Llama 3 在下游使用上的许可限制表示担忧。

- **Mixtral 和 WizardLM 突破开源边界**：**[Mistral AI 的 Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/)** 和 **Microsoft 的 [WizardLM-2](https://wizardlm.github.io/WizardLM2)** 作为强大的开源模型引起了轰动。Mixtral 8x22B 拥有 39B 激活参数，擅长数学、编程和多语言任务。WizardLM-2 提供了一个 8x22B 的旗舰版本和一个快速的 7B 变体。两者都展示了开源模型的飞速进步，支持范围已扩展到 **[OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct)** 和 **[LlamaIndex](https://twitter.com/llama_index/status/1780646484712788085)** 等平台。

- **Stable Diffusion 3 发布，评价褒贬不一**：Stability AI 在 **[API 上发布了 Stable Diffusion 3](https://vxtwitter.com/StabilityAI/status/1780599024707596508)**，但初步印象褒贬不一。虽然它在排版和提示词遵循方面有所进步，但一些人报告了性能问题和大幅涨价。该模型无法在本地使用也招致了批评，尽管 Stability AI 承诺很快将向会员提供权重。

- **CUDA 难题与优化**：CUDA 工程师们应对了各种挑战，从**[分块矩阵乘法（tiled matrix multiplication）](https://discord.com/channels/1189498204333543425/1189498205101109300/1230259495330906194)** 到 **[自定义 kernel 与 torch.compile 的兼容性](https://discord.com/channels/1189498204333543425/1189607750876008468/1230436794705903617)**。讨论深入探讨了内存访问模式、warp 分配以及 **[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85)** 等技术。**[llm.c 项目](https://github.com/karpathy/llm.c/pull/170)** 进行了多项优化，减少了内存占用并加速了 attention 机制。

- **AI 生态系统随着新平台和资金投入而扩张**：AI 初创领域活动频繁，**[theaiplugs.com](http://theaiplugs.com/)** 作为 AI 插件和助手的市场首次亮相，**[SpeedLegal 在 Product Hunt 上线](https://www.producthunt.com/posts/speedlegal)**。一份涵盖 550 轮融资、总额达 **[300 亿美元的 AI 初创公司融资数据集](https://www.frontieroptic.com/ai-hype-train)** 被汇编并分享。**[Cohere](https://txt.cohere.com/compressed-embeddings-command-r-plus/)** 和 **[Replicate](https://replicate.com/docs/billing)** 等平台推出了新模型和定价结构，标志着生态系统正趋于成熟。


---



# 第 1 部分：Discord 高层摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 限额缩减对话次数**：工程师们对意料之外的 **Opus 模型使用限制**感到不满，该限制突然从每天 600 条消息降至 30 条，打乱了原有计划，并促使一些人寻求退款或寻找替代方案，如 [Tune Chat](https://chat.tune.app/)。

- **Llama 3 热度超出预期**：关于 **Meta 的开源 Llama 3 模型**讨论热烈，工程师们分享了相关链接并讨论其潜力，同时密切关注 Twitter 上发布的最新 Benchmark（基准测试）以及像 [llama3](https://ollama.com/library/llama3) 这样提供深入探索的链接。

- **API 生态系统在波折中成长**：在关于 API 不一致性的讨论中，**mixtral-8x22b-instruct** 加入了 Perplexity Labs 的产品阵容（位于 [labs.pplx.ai](http://labs.pplx.ai)），并且 Perplexity Pro 用户现在每月可获得 5 美元的 API Credit。

- **对排版变革的共同热情**：工程师们对排版技术的发展表现出兴奋，传阅了关于 [新 Tubi 字体](https://www.perplexity.ai/search/new-tubi-font-PPafxuLATeWjJr9N5fYJbA) 的链接及相关讨论。

- **模型执行诗意分析**：**mixtral-8x22b-instruct** 因其对歌词（如 Leonard Cohen 的作品）的细致理解而受到赞誉，这表明在内容解读方面为其他模型树立了新标杆。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3 强势登陆 API**：Stability AI 已在开发者平台 API 上**推出了 Stable Diffusion 3** 及其 Turbo 版本，声称在排版和 Prompt（提示词）遵循度方面有显著改进。
- **面向大众的模型权重**：Stability AI 承诺向会员**发布模型权重**，鼓励 Self-hosting（自托管）并支持更广泛的开源生成式 AI 运动。
- **GPU 辩论升温**：关于是投资即将发布但尚未上市的 GPU（如 5090），还是选择当前的高性能显卡（如二手 3090）来处理 AI 任务的讨论非常激烈，重点考量了 VRAM 容量和 NVLink 能力。
- **AI Influencer（AI 网红）—— 一门有争议的技艺**：对话涉及 AI Influencer 的创建，揭示了从单纯的好奇到盈利目的等多种动机，同时也伴随着对其社会贡献的质疑。
- **微调修复与求职**：工程师们交流了微调新模型以获得更好性能的见解，同时关注 AI 领域的职位空缺和合作伙伴关系，并分享了 **[Palazzo, Inc. 的 Stable Diffusion 工程师](https://apply.workable.com/palazzo-inc-1/j/877AE4A35A/)** 招聘帖。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Snowflake 的 Embedding 引发讨论**：新推出的 [Snowflake text-embedding 模型](https://www.youtube.com/watch?v=p9T7ZgtM5Mo) 成为关注焦点，引发了关于语言向量空间和符号语言形式的讨论。针对文中提到的 **256 维 Embedding**（相较于常见的 1500 维），成员们对其效率和意义表示关注，并计划自行测试其检索准确率。

- **模型安全漏洞曝光**：讨论揭示了 Hugging Face 发生的一起涉及恶意 .pickle 文件的安全事件，并思考了 OpenAI 系统中类似的漏洞。这突显了 AI 系统安全中持续存在的风险和挑战，强调了工程师设计强大防御措施的必要性。

- **Llama 3 增强了基准测试的热度**：尽管有人担心 **[Meta 的 Llama 3](https://ai.meta.com/blog/meta-llama-3/)** 与 **Mixtral 8x22B** 等模型相比在 Context Length 方面存在限制，但其性能表现仍引起了轰动。社区正在积极讨论使用 **[MLX](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_qlora_single_device.yaml)** 进行 Finetuning，但 Mistral 模型突然设置访问限制（gating）暗示了监管挑战或滥用预防措施。

- **Prompt 困惑与 GPU 思考**：用户针对 **Hermes 2 Pro** 模型的 Prompt 行为以及对 "directly_answer" 工具的需求进行了问答交流。此外，在 GPU 集群上进行 **Long Context Inference** 的技术挑战，以及使用双 A100 GPU 配合 Jamba 处理 200k Tokens 的案例是讨论的重点，这对于从事类似高容量部署方案的工程师具有参考价值。

- **WorldSim 传闻**：社区成员正满怀期待且俏皮地等待 WorldSim 的重新发布。用户的见解建议实施使用限制和收费，以防止导致关停的操纵性输入。该平台在用户生成 AI 文明方面的潜力，成为了期待和哲学探讨的重要话题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Llama 3 正在炒热 LM Studio**：新的 **Meta Llama 3**，特别是 8B Instruct 版本，随着其在 [Hugging Face](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF) 上的发布和可用性引发了热潮，但用户报告了意外的输出重复和 Prompt 循环问题。爱好者们讨论了在本地运行 **WizardLM-2-8x22B** 等大模型的可行性，普遍认为在 24GB 的 **Nvidia 4090** 显卡上运行可能并不实际。

**技术故障与成功**：AI 工程师分享了在从 Ryzen 5 3600 到 Mac M1 和 M3 Max 等不同硬件配置上优化 **Llama 3** 性能的方法，一位用户通过调整主板设置以降低运行温度，解决了 Thermal Throttling（热节流）问题。双 **P100 GPU** 对某些人来说比较棘手，存在利用率不当的问题，同时用户也讨论了不同 NVIDIA GPU 根据需要贡献 VRAM 的能力。

**AI 应用参与与咨询**：基于 Electron 的应用 **MissionSquad** 在最近的 V1.1.0 版本中提供了 **Prompt Studio**，引起了广泛关注。然而，一些倾向于查看源代码的用户对透明度提出了要求，这与隐私权衡之间存在讨论。将 **Text-to-Speech (TTS)** 功能整合进 **LM Studio** 的建议反映了用户对增强交互性的渴望。

**AMD 历险记**：使用 AMD 配置的用户在运行 **LM Studio** 时遇到了 GPU 选择挑战。虽然最新的 **ROCm preview (0.2.19)** 应该能解决 iGPU 选择难题，但有关推理异常的报告表明，对于 **8B 模型** 等大模型的支持仍存在问题。社区分享了一个禁用 iGPU 的变通方法，并建议针对持续存在的问题提交更新或 Bug 报告。

**Prompt 创作征集**：**LM Studio** 的讨论延伸到了实际事务，如策划联盟营销活动，用户要求 AI 模型提供超越通用输出的特异性。一位成员强调，在寻求开发者参与时，需要的是事务性安排而非投机性伙伴关系。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Llama 3 发布吸引工程师**：技术 Discord 社区成员积极参与了 *Llama 3* 的讨论和测试，评估其基准测试结果。结果显示，尽管参数较少（8B 对比 70B），其性能与前代 Llama 2 相当。他们尝试将其集成到 Unsloth AI 框架中，引用了针对 8B 模型的 [Google Colab notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)，并探索了 70B 模型的 4-bit 量化版本。

**应对移动端缺失 CUDA 的挑战**：参与者指出，由于缺乏 CUDA 兼容性，在移动设备上部署神经网络面临挑战，引发了关于使用自定义推理引擎作为替代方案的讨论。这些对话涉及了为 iPhone 硬件部署而编译神经网络模型的复杂性。

**TorchTune 放弃旧硬件支持**：TorchTune 停止支持旧版 GPU，引发了关于其对使用前代硬件用户影响的讨论。用户提到了利用 Obsidian 等笔记工具进行知识管理的变通方法。

**许可物流与命名游戏**：遵守 Llama 3 新许可条款的重要性成为讨论话题，特别是任何衍生模型名称中必须包含 "Llama 3" 前缀的要求。这种对细节的关注强调了开源 AI 领域中法律考量的重要性。

**双语头脑风暴**：社区思考了创建双语模型的策略，权衡了潜在解决方案的成本和复杂性，例如用于翻译层的三次 LLM 调用。此外，*Distributed Negotiation Optimization (DNO)* 引起了关注，人们意识到虽然它尚未在库中实现，但可以作为 *Direct Preference Optimization (DPO)* 的有效迭代。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**分块变换**：在讨论*分块矩阵乘法 (tiled matrix multiplication)* 时，工程师指出，尽管为分块而对大矩阵进行填充 (padding) 会增加计算量，但可以节省内存带宽。

**Meta 的 Llama 缺少 MOE**：Meta 最新发布的 **Llama 3** 是一个 **405B 参数的稠密模型 (dense model)**，没有采用 MOE (Mixture of Experts) 架构，这与其它的 SOTA 架构形成对比。[Meta Llama 3 详情](https://llama.meta.com/llama3/)

**CUDA 远征者交谈**：CUDA 讨论范围广泛，从加载大型数据集的最佳实践和优化 kernel 设置，到调试结果差异以及剖析内存访问模式及其对性能的影响。

**Triton 与自定义操作的难题**：AI 工程师交流了使自定义函数与 `torch.compile` 兼容的技术，并参考了处理 `torch.jit.ignore` 的方法以及自定义 Triton kernel 的演示。[针对 torch.compile 的自定义 CUDA GitHub PR 引用](https://github.com/pytorch-labs/ao/pull/135) 和 [自定义 Triton kernel 的组合](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661) 是对话的一部分。

**量化困境**：深入讨论了 **半二次量化 (Half-Quadratic Quantization, HQQ) 方法**，特别是关注 axis=0 与 axis=1 量化的对比，并解决 Transformer 权重矩阵拼接的挑战。分享的链接包括对当前实践的评估、创新的优化技术，以及未来将 HQQ 集成到 torchao 的可能增强方案。[HQQ 实现详情](https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85)

**CUDA 活动的协作协调**：Massively Parallel Crew 计划重叠的小组讨论和 **CUDA MODE** 活动，展示了在安排录制、克服调度冲突和后期制作工作方面的团队合作。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**SD3 仅通过 API 首次亮相，评价褒贬不一**：Stability AI 发布了 [通过 API 提供的 SD3](https://vxtwitter.com/StabilityAI/status/1780599024707596508)，反响不一。人们承认其存在一些性能问题，特别是在文本渲染方面，同时也关注到其迈向货币化的战略举措。

**数据集困境**：随着 LAION 数据集从 Huggingface 下架，成员们开始寻找替代方案，如 **coyo-700m** 和 **datacomp-1b** 来训练新模型。同时，PAG 在 SDXL 上的应用也引起了关注，虽然它提供了比以往更好的视觉效果，但仍未超过 DALLE-3 的能力。

**Stability AI 的动荡局面**：Stability AI 高层的离职引发了关于公司未来以及对开源 AI 模型潜在影响的讨论，管理不善的担忧阴云笼罩。更广泛的 AI 社区开始测试并对 Meta 的 [LLaMA 3](https://www.llama2.ai/) 做出反应，尽管其上下文窗口较小，但其在各种任务中的表现赢得了赞赏。

**GANs 在效率上保持微弱领先**：GANs 因其推理速度和参数效率而受到关注，但它们难以训练，且视觉效果往往不尽如人意。与此同时，微软发布的 [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/) 将利用音频线索，彻底改变实时逼真的谈话面部生成。

**数据集与模型不断演进**：HQ-Edit 现已开放，这是一个包含约 200,000 次编辑的大型指令引导图像编辑数据集，有望增强未来的 AI 图像编辑工具。此外，Meta 宣布推出强大的开源 Llama 3 语言模型，展示了其对 AI 普及和进步的承诺。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Llama 的提升**：新发布的 **[Llama 3](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview)** 凭借基于 Tiktoken 的分词器和 8k 上下文长度实现了性能飞跃。

**Axolotl 提升表现**：已提交 PR 将 **Llama 3 Qlora** 集成到 Axolotl 中，并讨论了在 **80GB GPU 配置**下的 CUDA 错误。此外，微调后的 Adapter 出现了挑战，通过将分词器设置更改为 `legacy=False` 和 `use_fast=True` 得到了解决。

**微调技巧**：深入探讨微调技术，成员们努力使用 `rope_theta` 等参数扩展上下文长度，并分享了在模型微调过程中通过取消冻结特定层来防止训练崩溃的经验。

**配置难题**：Axolotl 不解析 YAML 文件中的注释，而用户对在 YAML 配置中设置 PAD tokens 的可行性表示了兴趣，这表明需要更清晰的配置文档。

**Token 调整技术**：交流中重点介绍了使用 `add_tokens` 和手动调整词汇表来替换 token 的方法，引发了关于 Llama-3 等模型最佳分词器调整的技术探讨。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Atlas 走向全电动化**：Boston Dynamics 展示了 Atlas 机器人的全新全电动版本，强调了相比之前版本的进步，该展示在[视频](https://www.youtube.com/watch?v=29ECwExc-_M)中引发了大量讨论。

- **Mixtral 和 WizardLM 重新定义 LLMs**：Mistral AI 的 [Mixtral 8x22B Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 拥有 39B 激活参数，专注于数学、代码和多语言任务；而 Microsoft AI 的 WizardLM-2 展示了其自身的 8x22B 模型以及更快的 7B 变体。这些模型拥有令人印象深刻的基准测试结果，并使用微调技术来增强其指令遵循能力。

- **Llama 3 随 Meta 进入 AI 舞台**：Together AI 与 Meta 合作，推出了用于微调的 [Meta Llama 3](https://llama.meta.com/llama3/)，提供 8B 和 70B 参数模型，并在 API 基准测试中实现了高达每秒 350 个 token 的吞吐量。

- **通过 OpenRouter 重新定义 AI 访问**：OpenRouter 的讨论集中在利用 WizardLM 和 Claude 等模型进行限制更少的应用，并提到 Together AI 正在使用 Mixtral，以及为扩展上下文应用而自行托管 Llama 3。

- **订阅系统故障与初创公司推荐**：有报告称 OpenRouter 的**订阅系统**出现问题，同时社区邀请大家关注成员的初创公司 [Product Hunt 上的 SpeedLegal](https://www.producthunt.com/posts/speedlegal)，这是一个用于合同谈判的 AI 工具。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Claude 对全球舞台的渴望**：有讨论指出 *Claude* 在文学相关任务中表现出色，但在某些地理区域仍无法访问，这凸显了对其更广泛可用性的渴望。

**Whisper v3 的传闻**：人们对 **Whisper v3** API 的发布充满期待，考虑到距离最初发布已有一年，这是一个重要的后续版本，但官方细节仍然寥寥无几。

**GPT-4 遗忘了过去？**：社区观察表明 **GPT-4** 的记忆能力有所下降，成员们注意到该 AI 的 Token 容量似乎有所减少，尽管目前还缺乏确凿证据。

**检测到 GPT-4 速度下降**：用户报告称 **GPT-4-0125-preview** 等版本出现了延迟，影响了对响应时间敏感的应用；尽管 **gpt-4-turbo-2024-04-09** 被作为解决方案提出，但用户感觉其速度也变慢了。

**AI 与区块链的新前沿**：一位成员指出了 AI 与区块链的交汇点，并邀请大家在 Prompt 开发方面进行协作，以推动这一新颖集成的进步。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**为 SoundStream 的 FLOPs 操心**：社区指导帮助一位新手估算了 SoundStream 的**训练 FLOPs**，并根据一篇 [Transformer 论文](https://arxiv.org/abs/2001.08361) 提供了关于每 Token 操作数和数据集大小乘积的详细建议。

**Scaling Laws 的审查加强**：一篇 [复现尝试论文](https://arxiv.org/abs/2404.10102) 挑战了 **Hoffmann et al.** 提出的 Scaling Laws，引发了关于置信区间以及此类大型语言模型（LLM）所需实验实际数量的讨论。

**解读 Tokenizer 对 LLM 的影响**：工程专家们辩论了更大 Tokenizer 词汇量的益处，特别是针对**多语言 LLM**，并考虑了在 Tokenizer 不同时使用 bits per byte 等方法来理解**模型困惑度（Perplexity）**。

**整合 LLM 中的新兴技术**：社区讨论涉及了 Untied Embeddings 和新型 Attention 机制对 LLM 的有效性，并探讨了将 **蒙特卡洛树搜索（MCTS）** 与 LLM 结合以获得更好的推理能力，正如腾讯的 [AlphaLLM](https://arxiv.org/abs/2404.12253) 所探索的那样。

**资源共享与协作评审呼吁**：分享了如 [lintang/pile-t5-base-flan](https://huggingface.co/lintang/pile-t5-base-flan) 等经过 Flan 微调的模型链接，并请求评审 **flores-200** 和 **sib-200** 基准测试的 PR，这对于推进多语言评估至关重要。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**将 C 与 Mojo 集成**：为那些有兴趣在 Mojo 中使用 C 的用户指出了 [mojo-ffi 项目](https://github.com/ihnorton/mojo-ffi) 和一个 [使用 `external_call` 的教程](https://twitter.com/Modular/status/1779913837216719118)。该教程特别讲解了如何在 Mojo 中调用 libc 函数。

**Modular 的精彩推文**：Modular 最近的推文引起了关注，并提供了直接链接，指向 [第一条推文](https://twitter.com/Modular/status/1780676643176231240) 和 [第二条推文](https://twitter.com/Modular/status/1781000544158650716)。

**Mojo 的兼容性查询**：讨论涉及了 Mojo 插件与 Windows 和 WSL 的兼容性，Mojo Playground 可能推出的支持低内存占用的 Nightly Build 特性 [GitHub 讨论](https://github.com/modularml/mojo/discussions/2321)，以及 `Variant` 尚不支持 **Movable** Trait 这一待处理问题。

**社区项目促进增长**：围绕 Mojo 的社区活动包括：解决 **Mojo 24.2** 的编译问题，一名学生寻求在 Mojo 中实现算法的指导，以及社区通过指向 [Mojo 入门](https://docs.modular.com/mojo/manual/get-started/) 页面等资源提供的支持性响应。

**LLaMa 的崛起**：一段 [YouTube 视频](https://www.youtube.com/watch?v=E3_0nHpfbcY) 报道了 **Meta LLaMa 3** 的发布并探索了该模型的新特性，表明社区对前沿 AI 研究持续关注。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **MCTS 与 PPO 结合助力 AI 突破**：探索 **蒙特卡洛树搜索 (MCTS)** 与 **近端策略优化 (PPO)** 的融合可能是 AI 决策领域的游戏规则改变者，从而产生一种新型的 **PPO-MCTS** 价值引导解码算法，旨在改进自然语言生成。[这里有一篇关于该主题的创新研究论文。](https://arxiv.org/abs/2309.15028)

- **语言模型领域的新秀**：AI 社区正因 **Mixtral 8x22B** 和 **OLMo 1.7 7B** 等令人印象深刻的模型推出而沸腾，这些模型在多语言流利度和 MMLU 分数上都树立了新标杆。**Mixtral Instruct** 在聊天机器人应用方面的进步前景，以及对 **Meta Llama 3** 规模的好奇，凸显了 AI 领域正处于显著扩张和普及的时期。相关细节和资源链接：[Mixtral 8x22B Apache 2.0](https://mistral.ai/news/mixtral-8x22b/)，[Hugging Face 上的 OLMo](https://huggingface.co/allenai/OLMo-1.7-7B)，以及 [Mixtral-Instruct 模型卡片](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)。

- **Chinchilla 缩放定律争议**：**Chinchilla** 论文中备受争议的缩放定律引发了 AI 社区的热烈讨论，研究人员 @tamaybes、@suchenzang 和 @drjwrae 发表了看法，作者 @borgeaud_s 也承认了一个错误。这场激烈的辩论强调了数据验证和透明度的必要性。参考推文证实了辩论的激烈程度：[tamaybes 的推文](https://x.com/tamaybes/status/1780639257389904013?s=46)，[suchenzang 的担忧](https://x.com/suchenzang/status/1616752482226671620?s=46)，以及 [borgeaud_s 的承认](https://x.com/borgeaud_s/status/1780988694163321250)。

- **AI 喜剧时刻**：Nathan Lambert 被一段 [《周六夜现场》(Saturday Night Live) 的短剧](https://www.youtube.com/watch?v=86qKgK0asGo) 逗乐了，该短剧扰乱了一场 AI 新闻直播活动，精准捕捉到了 AI 对文化的影响力已跨入幽默领域。

- **AI 奇闻与沉思**：讨论在即将到来的 **OLMO vs. LLaMa 3** 模型对决，以及《三体》、播客专题、数字命理学和博客文章预测之间的深奥联系中摇摆。与此同时，Jeremy Howard 关于“实验性”方面的推文引发了猜测。[Jeremy 的推文引起了关注](https://x.com/jeremyphoward/status/1780816986777559246?s=46)。

- **SnailBot 缓慢而稳步的进展**：SnailBot 或许赢不了速度竞赛，但它在 WIP（进行中）帖子上的功能性正冲向终点线，讽刺地反映了技术领域经常面临的速度难题。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Web UI 微调 - 简单开始，其余交给 API**：在 Cohere 通过 [Web UI](http://dashboard.cohere.com/fine-tuning) 启动模型微调非常用户友好，但使用新数据集进行进一步微调则需要使用 API，详细说明可在 [官方文档](https://docs.cohere.com/docs/fine-tuning-with-the-web-ui) 中找到。

- **Cohere 的最新奇才：Command R+**：Cohere 推出的 **Command R+** 因其显著的进步而受到认可。可以在 [Cohere 网站](https://txt.cohere.com/compressed-embeddings-command-r-plus/)上探索广泛的功能对比和模型能力。

- **伦理 AI：应对 Command R+ 的潜在风险**：针对 **Command R+** 提出的担忧涉及可能被操纵用于不道德目的的漏洞，正如链接到 [LessWrong](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r) 的红队演练所强调的那样。

- **LLMs 越狱 - 从语言到代理**：对话围绕 AI 越狱的概念展开，指出重点已从提取不当语言转向诱导 **大语言模型 (LLMs)** 产生复杂的自主行为——这是在敏感环境中使用 AI 的组织必须考虑的关键因素。

- **Cohere Command 与 Llama – 性能笔记**：用户对 Llama 3 模型的能力印象深刻，讨论了在评估 AI 时实际应用的重要性。70b 和 400b 等大型变体模型的性能是根据它们对数学方程和 SVG 标记等复杂提示词的响应来评估的。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **创业玩笑掩盖真实对话**：成员们幽默地提议成立一家初创公司来创建更优越的聊天库，暗示可能超越 OpenAI 等巨头。
- **本地模型：小即是新的大趋势**：公会讨论了向具有用户友好界面的小型、高性能 AI 模型转变的潜力，强调权宜之计优于复杂性。
- **延迟：每一毫秒都至关重要**：工程师们强调，延迟对用户体验和 AI 应用的成功采用是有害的，突出了快速响应的重要性。
- **AI 性能下降之谜**：分享了关于 AWS 托管的 Claude 3 性能急剧下降的观察，临床概念提取任务的准确率从 95% 以上降至接近于零。
- **会议移至 Zoom**：**llm-paper-club-west** 通过将会议移至 Zoom 并提供提醒以确保平稳过渡，促进了关于论文的讨论，[Zoom 会议链接](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Windows 上的 PowerShell 难题**：工程师们报告了在 Windows 上实现 **OpenInterpreter** 的挑战，特别是 **PowerShell** 无法识别 `OPENAI_API_KEY` 等环境变量。还讨论了安装 `poetry` 所需的时间以及在各种 Windows 环境中运行 **OpenInterpreter** 的复杂性。

**ESP32 的连接困扰**：用户分享了连接 **ESP32** 设备的困难，建议指向不同的 IDE 和使用 `curl` 命令。与消息数组相关的错误消息强调了设备连接的持续问题。

**使用本地服务器和 WebSockets 进行调试**：围绕为 **OpenInterpreter** 设置本地服务器以及排除 WebSockets 和 Python 版本不兼容问题出现了挑战。努力包括通过 `curl` 手动配置服务器地址，以及尝试解决音频缓冲问题。

**探索跨设备兼容性**：关于 **OpenInterpreter** 的讨论涉及在 Windows 上使用 **LM Studio**，同时在 Mac 上运行软件，强调了跨操作系统兼容性的必要性。用户报告称切换到 MacBook 以潜在地规避现有障碍。

**Hugging Face 亮点**：一条消息引用了一个 [Hugging Face space](https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_8b)，用户可以在其中与 **Meta LLM3_8b** 聊天，表明了社区对实验替代语言模型的兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MistralAI 的 8x22b 登场**：**MistralAI** 发布了 **8x22b** 模型，**LlamaIndex** 自发布之初就已支持该模型，具有 RAG、查询路由和工具使用等高级功能，详见 [Twitter 帖子](https://twitter.com/llama_index/status/1780646484712788085)。

- **使用 Elasticsearch 构建免费 RAG 的教程**：[博客文章](https://twitter.com/llama_index/status/1781022740339920967)详细介绍了 **LlamaIndex** 和 **Elasticsearch** 在创建免费**检索增强生成 (RAG)** 应用指南中的应用。

- **高效 RAG 系统的实现技巧**：AI 工程师讨论了优化 **RAG** 实现和提供多语言支持的方法，重点介绍了[微调指南](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=embeddings+fine)和 RAG 内的摘要技术资源，包括 [Q&A Summarization](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/?h=summar#summarization)。

- **Google 的无限上下文预示 LLM 的未来**：正在讨论 Google 开发的一种允许大语言模型处理无限上下文的方法，这对面临潜在范式转移的 **RAG** 等现有框架具有影响。技术方法及其影响在 [VentureBeat 的文章](https://venturebeat.com/ai/googles-new-technique-gives-llms-infinite-context)中进行了探讨。

- **揭秘数据驱动的 AI 融资洞察**：**manhattanproject2023** 为 AI 社区提供了访问与 AI 融资相关的详细数据集的机会，其中包含公司发展各个阶段的 300 亿美元投资，可在 [AI Hype Train - Airtable](https://www.frontieroptic.com/ai-hype-train) 进行分析。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**SQL Skirmish to Chatbot Progress**: 工程师们正在应对 LangChain 的 SQL agent 限制以及聊天机器人实现的 prompt engineering 挑战，参考资料包括 [`createOpenAIToolsAgent` 和 `SqlToolkit`](https://js.langchain.com/docs/use_cases/sql/agents)，旨在将 SQL 数据库集成到对话式 AI 中。

**Memory Management Mentorship**: 重点关注利用 `RunnableWithMessageHistory` 来管理聊天历史，并参考了 LangChain 代码库中记录的动手建议和代码示例，以增强消息检索和聊天机器人的 memory 能力。

**Marketplace for AI Plugs Emerges**: [theaiplugs.com](http://theaiplugs.com/) 已上线，为销售 AI 插件、工具和助手提供解决方案，并处理 API、营销和计费，以简化创作者的工作流程。

**Product Hunt Seeks AI Speedsters**: [SpeedLegal](https://www.producthunt.com/posts/speedlegal) 在 Product Hunt 上亮相，寻求社区支持，同时一门新的 prompt engineering 课程已在 LinkedIn Learning 上线，供那些渴望提升技能的人学习。

**Llama 3 Thunders into Public Domain**: 开发者公开了 Llama 3 的访问权限，邀请用户通过 [聊天界面](https://chat.tune.app/) 和 API 探索其功能，这是将先进 AI 工具传播给更广泛受众努力的一部分。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Alert for Inappropriate Content Across Channels**: Discord 公会内的多个频道遭到推广成人内容的垃圾信息攻击，具体涉及 "Hot Teen & Onlyfans Leaks" 以及 Discord 邀请链接 ([邀请链接](https://discord.gg/rj9aAQVQFX))。鉴于这些事件，公会成员被敦促加强审核。

- **Spam Infiltrates Technical Discussions**: 困扰公会的垃圾信息问题在 #programming-help 和 #alignment-lab-ai 等技术讨论频道，以及 #general-chat 和 #join-in 等社区频道中都很普遍，这表明公会面临着全范围的审核挑战。

- **Wizards of the Code Unveil WizardLM-2**: **WizardLM-2** 模型已取得进展，现已在 [Hugging Face](https://huggingface.co/alpindale/WizardLM-2-8x22B) 上公开访问，并通过 [WizardLM-2 发布博客](https://wizardlm.github.io/WizardLM2)、[GitHub 仓库](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2)、相关的 [Twitter 账号](https://twitter.com/WizardLM_AI) 以及 [arXiv](https://arxiv.org/abs/2304.12244) 上的学术论文提供了额外资源。

- **Seek and You Shall Find Meta Llama 3 Tokenizer**: 在一名公会成员请求 **Meta Llama 3-8B** tokenizer 后，用户 Undi95 提供了该资源，现在可以在 [Hugging Face](https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main) 上找到，从而规避了遵守特定隐私政策的需要。

- **Community Calls for Action**: #open-orca-community-chat 等频道的讨论强调了采取立即审核行动（包括可能的封禁）的必要性，以维护以工程为核心的公会环境的完整性。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**VRAM 饥渴：贪多嚼不烂？**：根据讨论，使用 **Adam 优化器**训练 **Mixtral-8x22B 模型**需要惊人的 **3673 GB VRAM**。即使是 **64 个** 80GB 的 **GPU** 也不足以避免在训练 32k 序列长度时出现显存溢出（OOM）错误。此外，成员们正在权衡使用 **8-bit 优化**来管理巨大显存需求的潜力。

**模型训练的成就与挫折**：一个专注于英语和德语指令的全新 **Mixtral-8x22B 模型**已训练完成，并在 [Hugging Face](https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de) 上共享。然而，在实现 `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock` 时遇到了**形状错误（shape errors）**，这表明参数状态可能未充分利用混合精度，从而使 FSDP 配置复杂化。

**Tokenizer 统一工作**：Mistral 公布了其专为跨模型兼容性设计的 Tokenizer 库，具有 Tool Calls 和结构化输出功能，示例可在该 [Jupyter Notebook](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb) 中找到。

**Meta 的 Llama 3 带着雄心勃勃的支持亮相**：Meta 发布了 *Llama 3*，因其增强的多语言能力和与云平台的直接集成而备受关注。尽管训练集中存在多语言数据，但其 **128K token 的 tokenizer** 因非英语表现可能欠佳而受到审查。更多详情请见 [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3/)。

**模型开放性的双刃剑**：随着 *Llama 3* 的出现，人们对 **Llama 3 输出的限制**感到担忧，这可能会阻碍开源开发，也反映出社区更偏向于像 MistralAI 这样限制较少的平台。社区的保留意见在这条批判性的 [推文](https://fxtwitter.com/xlr8harder/status/1780992684062024138?s=19) 中得到了体现。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **创业公司在 Product Hunt 上崭露头角**：一位成员在 Product Hunt 上发布了他们的创业项目 [SpeedLegal](https://www.producthunt.com/posts/speedlegal)，这是一款 AI 工具，旨在通过识别风险和简化法律术语来辅助合同谈判。

- **Karpathy 提倡精心打磨小模型**：Andrej Karpathy [最近的推文](https://twitter.com/karpathy/status/1781028605709234613) 暗示社区对小模型的训练普遍不足，并指出一个经过 15T token 数据集精心磨练的 8B 参数模型可以与更大的模型相媲美。

- **小模型赢得社区青睐**：小巧且经过勤奋训练的 AI 模型这一概念引起了社区的共鸣，这可能是受到 Karpathy 倡导的挖掘小架构潜力的启发。

- **密切关注 Mixtral**：社区成员热衷于测试 **Mixtral 8x22B Instruct**，[Hugging Face 上的模型卡片](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 已被分享，详细介绍了使用案例和实现方法。

- **插件混乱挑战 LLM 开发**：llm-gpt4all 插件安装中出现的问题导致 Python 应用程序崩溃，这在 [GitHub issue #28](https://github.com/simonw/llm-gpt4all/issues/28) 中被强调，并引发了对 LLM 插件韧性和依赖管理的担忧。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **PyTorch Lightning 以硬件中立性出击**：对话强调了 [PyTorch-Lightning 的能力](https://github.com/Lightning-AI/pytorch-lightning)，即无需修改代码即可在包括 GPU 和 TPU 在内的各种平台上训练、微调和部署 AI 模型。
- **AMD Radeon GPU 的成功案例**：一块 AMD Radeon 7900XTX GPU 已成功用于运行 PyTorch-Lightning，展示了其与多样化硬件选项的兼容性。
- **ROCm 为 PyTorch 带来速度提升**：在 7900XTX GPU 上测试时，利用 ROCm 的优化，PyTorch-Lightning 在某些模型上的表现比常规 PyTorch 更快。
- **领域内的新 AI 模型**：新 AI 模型 **LLaMa3** 已经发布，提供了适用于不同规模 AI 应用的预训练版本，详见其 [官方页面](https://llama.meta.com/llama3/)。
- **Tinygrad 迈向高效 Tensor 操作**：在 **tinygrad** 中，正在追求 **broadcast、reshape、permute** 等零成本 Tensor 操作，并建议探索 **tinygrad/shape/shapetracker.py** 或 **view.py** 以获取战略指导。

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI 社区对新模型发布反响热烈**：讨论重点关注了 **Snowflake Arctic** embed 模型系列、**Mixtral 8x22B** 以及 **Meta** 的 **Llama 3** 的发布，称赞它们是文本 embedding 和大型语言模型 (LLM) 领域的里程碑。通过 [YouTube 链接](https://www.youtube.com/watch?v=N8U6XnVK2mM) 分享了这些模型的详细见解和介绍。

- **对 Serverless Fine-tuning 的好奇**：一场对话引发了对开源 AI 模型 **no-code fine-tuning 和 serverless 推理平台** 可能性的兴趣，类似于某些平台为 **GPT-3.5** 提供的便捷性。

- **非正式闲聊缺乏实质内容**：频道中出现了一个充满活力的问候 "HELLLLOOOOOOOO!!!!!!!!"，但缺乏与工程讨论相关的实质性内容。

- **大量模型介绍视频**：社区分享了来自 YouTube 的宝贵视频资源，为对 AI 模型开发前沿感兴趣的人提供了 **Snowflake Arctic** embed 模型、**Mixtral 8x22B** 和 **Llama 3** 的概览。

- **无代码 AI 工具——是梦想还是即将到来？**：有人提出了关于是否存在（或正在开发）支持 GPT-3.5 以外模型的 **no-code AI 平台** 的问题，暗示了对更易用的 AI fine-tuning 技术潜藏的需求。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Llamafile 脚本现已更加简洁**：一个升级且简洁版本的 **llamafile 归档版本升级重包脚本** 已在 [Gist](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e) 中分享，并考虑将其添加到 **llamafile GitHub repo**。该成员提醒，从头开始创建新的 llamafile 比重新打包旧版本更好。

**漏洞报告步骤受到询问**：有人询问如何报告 **安全漏洞** 以及获取 CVEs 的流程，该问题已转至线下进行更详细的讨论。

**警惕暴露 LLM APIs**：发布了一个关于公开暴露 **LLM API 端点** 的通用警告，强调这并非第一次在 **LLM 基础设施代码** 中发现 bug。重点强调了以往在此类系统中遇到漏洞的经验。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **关于 LITELLM 使用情况的咨询**：一位名为 jeffreyw128 的成员询问社区是否有人正在使用 **litellm**，寻求与该工具有关的见解或经验。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **征集分布式推理实现的示例代码**：一位在使用 2x A100 GPU 配置对 **AI21 Labs** 的 **Jamba** 进行 **长上下文推理** 时遇到挑战的成员，正在寻求示例代码以应对分布式系统的复杂性。特别请求针对多 GPU 集群上分布式推理场景的示例。



---

# PART 2: 各频道详细摘要与链接



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1230129460024447037)** (910 条消息🔥🔥🔥): 

- **Opus 使用限制引发不满**：用户对最近 Opus 模型的使用限制表示不满，指出其在未事先通知的情况下从每天 600 条减少到 30 条消息，影响了新老订阅用户。
- **关于退单和退款的争论**：针对意外的服务变更，用户就退单和退款的可能性展开了长时间讨论，不同用户权衡了法律和伦理考量，部分用户向 Perplexity 支持团队寻求退款。
- **对 Llama 3 的期待**：对来自 Meta 的开源模型 Llama 3 的期待很高，用户讨论了其潜力，并分享了关于其 benchmark 和能力的外部链接。
- **取消订阅与替代服务**：由于 Opus 限制的降低，几位用户报告取消了试用或正式订阅，而其他用户则考虑迁移到不同的服务或等待 Perplexity 解决问题。
- **技术问题与侧边讨论**：用户报告了不相关的技术问题，例如在 Android 上使用 App 时遇到困难，一些侧边对话包括分享与 AI 发展相关的 YouTube 视频或推文等外部内容。
<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1781043092776161476">来自 Perplexity (@perplexity_ai) 的推文</a>：了解更多关于 Llama 3 的信息 👇https://www.perplexity.ai/search/Llama-3-Overview-Mz3Cw09KTdq9gavmibDBeA</li><li><a href="https://chat.tune.app/">Tune Chat - 由开源 LLM 驱动的聊天应用</a>：通过 Tune Chat，访问 Prompts 库、Chat with PDF 和 Brand Voice 功能，以增强您的内容写作和分析，并在所有创作中保持一致的语调。</li><li><a href="https://ollama.com/library/llama3">llama3</a>：Meta Llama 3：迄今为止功能最强大的开源 LLM</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://studio.tune.app/">未找到标题</a>：未找到描述</li><li><a href="https://fxtwitter.com/LechMazur/status/1781049810428088465?t=sk98ui7oEw00swjCMQrz6Q&s=19">来自 Lech Mazur (@LechMazur) 的推文</a>：Meta 的 Llama 3 70B 和 8B 在 NYT Connections 上进行了基准测试！就其规模而言，结果非常强劲。</li><li><a href="https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fw5wsw2ecq9vc1.png">https://i.redd.it/w5wsw2ecq9vc1.png</a>：未找到描述</li><li><a href="https://tenor.com/view/wack-whack-gif-26201100">Wack Whack GIF - Wack Whack - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/klatschen-clapping-gif-12186138">Klatschen Clapping GIF - Klatschen Clapping - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=overview">Microsoft Azure Marketplace</a>：未找到描述</li><li><a href="https://tenor.com/view/ladies-and-gentlemen-mikey-day-saturday-night-live-presenting-please-welcome-gif-20519836">Ladies And Gentlemen Mikey Day GIF - Ladies And Gentlemen Mikey Day Saturday Night Live - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/zuckerberg-gif-19397752">Zuckerberg GIF - Zuckerberg - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/laptop-smoking-fire-burning-lag-gif-19373925">Laptop Smoking GIF - Laptop Smoking Fire - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/snape-harry-potter-you-dare-use-my-own-spells-against-me-potter-severus-snape-gif-16590981">Snape Harry Potter GIF - Snape Harry Potter You Dare Use My Own Spells Against Me Potter - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/movie-one-eternity-later-gif-7900643">Movie One Eternity Later GIF - Movie One Eternity Later - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://pict.chat">PictoChat Online，由 ayunami2000 开发。</a>：使用 Java 编写服务器的 PictoChat Web 应用！源码：https://github.com/ayunami2000/ayunpictojava</li><li><a href="https://youtu.be/qnUziiukzbE?si=JTsRKu7CZgbJcro_">Nintendo DS PictoChat 回来了！</a>：Nintendo DS 的 PictoChat 并没有消失，现在有一个网站可以让你重温那些给朋友发送恶搞画作的光辉岁月，而且是线上的！Pi...</li><li><a href="https://www.youtube.com/watch?v=ogRV5UzMmb8">24 核 vs 32 核 M1 Max MacBook Pro - Apple 隐藏的秘密..</a>：关于这款更便宜的独角兽 MacBook，还没有人向你展示过的内容！获取您的 Squarespace 网站免费试用 ➡ http://squarespace.com/maxtech 经过一个月的体验...</li><li><a href="https://youtu.be/bc6uFV9CJGg">Mark Zuckerberg - Llama 3、100 亿美元模型、凯撒·奥古斯都以及 1 GW 数据中心</a>：小扎谈论：- Llama 3 - 朝向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - 凯撒·奥古斯都、智能爆炸、生物...</li><li><a href="https://github.com/meta-llama/llama3">GitHub - meta-llama/llama3: Meta Llama 3 官方 GitHub 站点</a>：Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账号来为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ)">Rick Astley - Never Gonna Give You Up (官方音乐视频)</a>：Rick Astley 的 “Never Gonna Give You Up” 官方视频。新专辑《Are We There Yet?》现已发行：在此下载：https://RickAstley.lnk.to/AreWe...</li><li><a href="https://www.meta.ai">Meta AI</a>：使用 Meta AI 助手处理事务、免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用 Emu...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1230315284641091625)** (12 条消息🔥):

- **排版变换 (Typographic Transformation)**：一位成员发现讨论 [新的 Tubi 字体](https://www.perplexity.ai/search/new-tubi-font-PPafxuLATeWjJr9N5fYJbA) 很有价值，表达了对排版话题的热情。
- **真实性的幻象 (Illusion of Authenticity)**：两位不同的成员指向了一个链接，讨论 [演员如何在行业中运行虚假](https://www.perplexity.ai/search/Actors-run-fake-zH.PMe5xRCqyHVutTISDnw) 场景。
- **探索过去 (Exploring the Past)**：["m" 的历史](https://www.perplexity.ai/search/The-history-of-mL_Wd3OJQ_qsoSkBaybl3Q) 引起了一位成员的兴趣，并为对这一特定历史见解感兴趣的人分享了链接。
- **无限的雄心 (Boundless Ambitions)**：[Limitless AI 吊坠](https://www.perplexity.ai/search/Limitless-AI-pendant-eIdXpAXxQoOv2H3Wlfr3dA#0) 引起了关注，一位成员引导其他人参与相关讨论。
- **创作与策展 (Creation & Curation)**：成员们分享了对一系列话题的兴趣，从制作 [特定事物](https://www.perplexity.ai/search/how-to-make-XdbnbMaTTmeEBpGycUp7Ng#0)，到数据可视化技术，甚至是 Adobe 对其 [Firefly AI](https://www.perplexity.ai/search/Adobe-trained-Firefly-SAL3_iaiSzOC0ulcrvj3Hw) 的训练。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1230260526051885116)** (12 messages🔥): 

- **API 摘要请求与差异**：一位用户询问是否可以通过 API 获取引用或摘要，并注意到 `sonar-medium-online` 的响应与浏览器应用中的响应有所不同。分享了一个指向 Discord 频道的链接以获取更多信息，尽管根据消息显示，分享的链接是无效的。

- **Perplexity API 与 OpenAI 的集成**：一位成员分享了在尝试将 Perplexity 的 API 与 OpenAI GPTs 的 actions 集成时请求停滞的经历，暗示在创建功能性 OpenAPI schema 方面存在困难。

- **Mixtral-8x22b 现已可用**：社区获悉 Perplexity Labs 和 API 新增了 `mixtral-8x22b-instruct`，并提供了在 [labs.pplx.ai](http://labs.pplx.ai) 上进行尝试的链接。

- **深入探讨 Leonard Cohen 的 "Avalanche"**：一位用户强调了 `mixtral-8x22b-instruct` 的出色表现，分享了关于该模型对 Leonard Cohen 歌曲 "Avalanche" 诠释的详细反馈，以及它在仅凭歌词识别艺术家和歌曲方面如何超越其他模型。

- **新 AI 模型丰富用户体验**：讨论了各种新模型的可用性更新，如 `llama-3-8b-instruct` 和 `llama-3-70b-instruct`，透露它们已添加到 Perplexity Labs 和 API 中，并提到拥有 Perplexity Pro 的用户每月可获得 5 美元的 API 额度。此外，一位成员对这些新模型为他们的应用程序带来的性能提升表示满意。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1780768372268933629">来自 Perplexity (@perplexity_ai) 的推文</a>：🚨 更新：我们已将 mixtral-8x22b-instruct 添加到 Perplexity Labs 和我们的 API！ ↘️ 引用 Perplexity (@perplexity_ai) Mixtral-8X22B 现已在 Perplexity Labs 上可用！快去 http://... 体验一下吧</li><li><a href="https://x.com/AravSrinivas/status/1781049202887237908">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：🦙 🦙 🦙 http://labs.perplexity.ai 上线了 llama-3 - 8b 和 70b instruct 模型。祝聊天愉快！经过一些后期训练后，我们很快将推出它们的联网搜索版本。...</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1230162110596649011)** (1 messages): 

- **Stable Diffusion 3 在 API 上线**：Stability AI **激动地宣布**，与 Fireworks AI 合作，在 Stability AI 开发者平台 API 上正式推出 Stable Diffusion 3 和 Stable Diffusion 3 Turbo。详细信息和访问说明请见 [此处](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post)。

- **文生图魔力超越竞争对手**：Stable Diffusion 3 在排版和提示词遵循方面 **超越了 DALL-E 3 和 Midjourney v6 等竞争对手**，它采用了先进的 Multimodal Diffusion Transformer (MMDiT) 架构以增强文本和图像处理能力，正如 [研究论文](https://stability.ai/news/stable-diffusion-3-research-paper) 中所强调的那样。

- **开源生成式 AI 的新篇章**：官方承诺很快将向拥有 Stability AI 会员资格的用户 **提供用于自托管的模型权重**，强调了 Stability AI 对开源生成式 AI 的奉献精神。

**提到的链接**：<a href="https://bit.ly/3xHrtjG">Stable Diffusion 3 API 现已可用 &mdash; Stability AI</a>：我们很高兴地宣布，Stable Diffusion 3 和 Stable Diffusion 3 Turbo 已在 Stability AI 开发者平台 API 上提供。&amp;nbsp;

---

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1230138290657562624)** (947 条消息🔥🔥🔥)：

- **API 优先于本地使用**：SD3 目前仅可通过 API 访问，每张图像的成本约为 0.065 美元；人们期待该模型在不久的将来能开放本地使用。
- **关于硬件需求的讨论**：用户正在权衡是等待 5090 等新型 GPU，还是购买二手 3090 等现有选项进行 AI 工作，并考虑了 VRAM、速度、功耗和 NVLink 支持等因素。
- **微调与生成挑战**：大家承认，虽然新模型在某些领域（如解剖结构生成）可能显得乏善可陈，但 Fine-tuning 有可能弥补这些不足。
- **创建 AI 网红？**：关于 AI Influencers 的讨论指出了从好奇心到盈利的各种动机，同时也有部分用户对这类尝试的社会价值持怀疑态度。
- **AI 模型考量与工作机会**：讨论内容包括对 API 积分定价结构（与 Ideogram 等替代方案相比）的担忧、不同模型版本输出质量的差异，以及对 AI 相关项目的工作机会或合作伙伴关系的请求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://comfyanonymous.github.io/ComfyUI_examples/model_merging/#advanced-merging">模型合并示例</a>: ComfyUI 工作流示例</li><li><a href="https://apply.workable.com/palazzo-inc-1/j/877AE4A35A/">Stable Diffusion 工程师 - Palazzo, Inc.</a>: 关于我们：Palazzo 是一家充满活力且创新的科技公司，致力于突破室内设计领域 Global AI 的界限。我们正在寻找一名熟练的 Stable Diffusion 工程师加入我们的团队...</li><li><a href="https://x.com/StabilityAI/status/1780599024707596508">来自 Stability AI (@StabilityAI) 的推文</a>: 今天，我们很高兴地宣布 Stable Diffusion 3 和 Stable Diffusion 3 Turbo 已在 Stability AI Developer Platform API 上可用。我们与 @FireworksAI_HQ 合作，这是最快的...</li><li><a href="https://arstechnica.com/ai/2024/04/power-hungry-ai-is-putting-the-hurt-on-global-electricity-supply/">耗电巨大的 AI 正在损害全球电力供应</a>: 数据中心正成为 AI 发展的瓶颈。</li><li><a href="https://www.pcgamer.com/games/card-games/champions-tcg-ai-artist/">卡牌游戏开发商称其支付给一位“AI 艺术家” 90,000 美元来生成卡牌艺术，因为“没有人能达到他交付的质量”</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd">Stable Video Diffusion</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/lUYMRFOvcF">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.tiktok.com/@voidstomper.ai/video/7355215076783131935">TikTok - 记录美好生活</a>: 未找到描述</li><li><a href="https://stability.ai/contact">联系我们 — Stability AI</a>: 未找到描述</li><li><a href="https://stability.ai/membership">会员资格 — Stability AI</a>: Stability AI 会员资格通过结合我们的一系列最先进的开放模型与自托管优势，为您的生成式 AI 需求提供灵活性。</li><li><a href="https://arstechnica.com/tech-policy/2024/04/feds-appoint-ai-doomer-to-run-us-ai-safety-institute/">联邦政府任命“AI 末日论者”负责美国机构的 AI 安全</a>: 前 OpenAI 研究员曾预测 AI 有 50% 的几率杀死我们所有人。</li><li><a href="https://kailashsiri.medium.com/the-impact-of-over-reliance-on-ai-balancing-technology-and-critical-thinking-88f72f6f8298">过度依赖 AI 的影响：平衡技术与批判性思维</a>: 我是一个 27 岁、积极、开明且热爱技术的发烧友。我是一个有点懒的人，这帮助我让事情变得更……</li><li><a href="https://youtu.be/ZqRMHhkeylw?si=nD3GXOvA89PJ7hpp">解码 Stable Diffusion：LoRA、Checkpoints 和关键词简化版！</a>: 🌟 通过我们清晰简洁的指南解开 Stable Diffusion 的奥秘！🌟 加入我们，我们将分解复杂的 AI 术语，如 'LoRA'、'Checkpoint' 和 'Con...</li><li><a href="https://kailashsiri.medium.com/the-impact-of-over-reliance-on-ai-balancing-technology-and-critical-t">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/kijai/ComfyUI-KJNodes/">GitHub - kijai/ComfyUI-KJNodes: ComfyUI 的各种自定义节点</a>: ComfyUI 的各种自定义节点。通过在 GitHub 上创建账户，为 kijai/ComfyUI-KJNodes 的开发做出贡献。</li><li><a href="https://youtu.be/kzL7vjAwg5M?si=aTwtx3vcSYP3sdkt">风云！风云城堡 | 官方预告片 | Amazon Prime</a>: 让我们动起来！80 年代标志性的日本游戏节目《风云城堡》现在正在热播 🏯 订阅：http://bit.ly/PrimeVideoSG 开始您的 30 天免费试用：...</li><li><a href="https://www.youtube.com/watch?v=ResSOxQBUSM">RTX 4080 vs RTX 3090 vs RTX 4080 SUPER vs RTX 3090 TI - 20 款游戏测试</a>: RTX 4080 vs RTX 3090 vs RTX 4080 SUPER vs RTX 3090 TI - 20 款游戏测试 1080p, 1440p, 2160p, 2k, 4k ⏩GPU & Amazon 美国⏪ (联盟链接)- RTX 4080 16GB: http...</li><li><a href="https://github.com/ShineChen1024/MagicClothing">GitHub - ShineChen1024/MagicClothing: Magic Clothing 的官方实现：可控的服装驱动图像合成</a>: Magic Clothing 的官方实现：可控的服装驱动图像合成 - ShineChen1024/MagicClothing</li><li><a href="https://github.com/PierrunoYT/stable-diffusion-3-web-ui">GitHub - PierrunoYT/stable-diffusion-3-web-ui: 这是一个基于 Web 的用户界面，用于使用 Stability AI API 生成图像。它允许用户输入文本提示词，选择输出格式和纵横比，并根据提供的参数生成图像。</a>: 这是一个基于 Web 的用户界面，用于使用 Stability AI API 生成图像。它允许用户输入文本提示词，选择输出格式和纵横比，并根据提供的参数生成图像...</li><li><a href="https://github.com/Priyansxu/vega">GitHub - Priyansxu/vega</a>: 通过在 GitHub 上创建账户，为 Priyansxu/vega 的开发做出贡献。</li><li><a href="

<a href="https://comfyworkflows.com/videos">Comfy Workflows 视频页面</a>：运行并探索并非针对单一任务的工作流，而是展示 ComfyUI 动画和视频有多么出色。例如：酷炫的人物动画、实时 LCM 艺术等。</li><li><a href="https://github.com/TencentQQGYLab/ELLA">GitHub - TencentQQGYLab/ELLA: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>：ELLA：为 Diffusion Models 配备 LLM 以增强语义对齐 - TencentQQGYLab/ELLA</li><li><a href="https://github.com/codaloc/sdwebui-ux-forge-fusion">GitHub - codaloc/sdwebui-ux-forge-fusion: Combining the aesthetic interface and user-centric design of the UI-UX fork with the unparalleled optimizations and speed of the Forge fork.</a>：将 UI-UX 分支的美观界面和以用户为中心的设计，与 Forge 分支无与伦比的优化和速度相结合。- codaloc/sdwebui-ux-forge-fusion</li><li><a href="https://arstechnica.com/ai/2024/04/power-">April | 2024 | Ars Technica</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=XfkgiXaaCY4&list=LL&index=2">Dwayne Loses His Patience 😳 #ai #aiart #chatgpt</a>：未找到描述</li><li><a href="https://www.youtube.com/shorts/ASkd9Oxk1Eo">1 Mad Dance of the Presidents (ai) Joe Biden 🤣😂😎✅ #stopworking #joebiden #donaldtrump #funny #usa</a>：🎉 🤣🤣🤣🤣 准备好在 "Funny Viral" 频道最新的 "搞笑动物合集" 中大笑不止吧！🤣 这些可爱的以及各种...</li><li><a href="https://beta.dreamstudio.ai/generate">DreamStudio</a>：未找到描述</li><li><a href="https://clipdrop.co/">利用 AI 在几秒钟内创建令人惊叹的视觉效果。</a>：移除背景、清理图片、放大、Stable Diffusion 等等……</li><li><a href="https://app.wordware.ai/r/b137b2f5-a971-420c-a594-4f6350c24fa5">Wordware - 比较提示词</a>：使用输入提示词和优化后的提示词运行 Stable Diffusion 3。
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1230135613546565773)** (46 条消息🔥): 

- **Snowflake 发布突破性嵌入模型**：Snowflake 推出并开源了一个全新的 [文本嵌入模型 (text-embedding model)](https://www.youtube.com/watch?v=p9T7ZgtM5Mo)，标志着文本分析能力的进步。
- **破译语言的向量空间**：成员们思考了高维向量空间内意义表示的概念框架，讨论了“意义向量空间内的包络 (envelopes)”的可能性，暗示了基于尺度的形成以及新符号语言形式的潜力。
- **讨论加密通信类比**：对话探讨了加密与各种科学现象（如引力动力学）之间的类比，并思考了由于“对无穷大做功 (performing Work on Infinity)”而导致的发散性语言和理解的需求。
- **初创公司在 Product Hunt 寻求支持**：一位用户为其最近在 [Product Hunt](https://www.producthunt.com/posts/speedlegal) 上推出的初创公司 *SpeedLegal* 请求反馈和支持，随后的讨论涉及了该产品的市场契合度和潜在的商业策略。
- **寻求并分享 Bloke Discord 邀请链接**：在发现 Twitter 上的链接失效后，社区成员协助分享了 Bloke Discord 服务器的有效链接，并就解除临时机器人限制以方便发布邀请进行了交流。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/dPfXbRnQ">加入 TheBloke AI Discord 服务器！</a>：用于讨论和支持 AI Large Language Models 以及通用 AI。 | 24155 名成员</li><li><a href="https://www.youtube.com/watch?v=zQy11WnAIIc">介绍 Llama 3：最强开源 Large Language Model</a>：介绍 Meta Llama 3，这是 Facebook 下一代最先进的开源 Large Language Model。https://ai.meta.com/blog/meta-llama-3/#python #...</li><li><a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake 发布全球最佳实用 Text-Embedding 模型</a>：今天 Snowflake 发布并以 Apache 2.0 协议开源了 Snowflake Arctic embed 系列模型。基于 Massive Text Embedding Be...</li><li><a href="https://www.youtube.com/watch?v=N8U6XnVK2mM">Mixtral 8x22B：Mistral 最好的开源模型</a>：Mixtral 8x22B 是最新的开源模型。它为 AI 社区的性能和效率树立了新标准。它是一个稀疏的 Mixture-of-Experts (SMo...</li><li><a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - 你的个人 AI 合同谈判专家 | Product Hunt</a>：SpeedLegal 是一款 AI 工具，可帮助你更好地理解和谈判合同。它可以快速识别潜在风险，并用简单的语言解释复杂的法律术语。SpeedLegal 还提供...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1230172661825671310)** (44 条消息🔥): 

- **Mistral AI 解密 Tokenization**：Mistral AI 已[开源其 tokenizer](https://docs.mistral.ai/guides/tokenization/)，包括指南和 Colab 笔记本。该 tokenizer 将文本分解为更小的子词单元（称为 tokens），以便语言模型以数字方式理解文本。
- **对 Tokenization 过度炒作持怀疑态度**：频道中的讨论质疑了 Tokenization 的重要性，如果像 Opus 这样的模型可以有效地利用 XML 标签，这表明 token 的相关性可能仅限于模型的可控性（steerability）。
- **Hugging Face 安全漏洞披露**：一段 [YouTube 视频](https://youtu.be/ZcoOW8nqVP8?t=140) 讨论了涉及 Hugging Face 的安全事件，该事件由恶意 .pickle 文件引起，突显了 AI 系统中的潜在漏洞。
- **OpenAI 中 Pickles 的潜在漏洞**：一次对话透露，在 OpenAI 的环境中使用不安全的 pickles 可能会带来风险，因为它允许执行大型文档，但如果被识别为漏洞利用，可能会被禁用。
- **斯坦福 AI 指数报告 2023 年 AI 现状**：最新的斯坦福 [AI Index 报告](https://hai.stanford.edu/news/ai-index-state-ai-13-charts) 发布，总结了多模态 foundation models、投资趋势以及向更多开源模型发展的趋势，去年发布的基础模型数量非常之多。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://hai.stanford.edu/news/ai-index-state-ai-13-charts">AI Index: 13 张图表揭示 AI 现状</a>：在新报告中，基础模型占据主导地位，基准测试（benchmarks）失效，价格飙升，而在全球舞台上，美国遥遥领先。</li><li><a href="https://docs.mistral.ai/guides/tokenization/">Tokenization | Mistral AI 大语言模型</a>：Tokenization 是 LLM 中的一个基础步骤。它是将文本分解为更小的子词单元（称为 tokens）的过程。我们最近在 Mistral AI 开源了我们的分词器。本指南将...</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>：未找到描述</li><li><a href="https://www.udio.com/">Udio | 创作你的音乐</a>：向世界发现、创作并分享音乐。</li><li><a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - 你的个人 AI 合同谈判专家 | Product Hunt</a>：SpeedLegal 是一款 AI 工具，可帮助你更好地理解和谈判合同。它可以快速识别潜在风险，并用简单的语言解释复杂的法律条款。SpeedLegal 还提供...</li><li><a href="https://www.youtube.com/watch?v=F3Jd9GI6XqE">Edward Gibson：人类语言、心理语言学、句法、语法与 LLMs | Lex Fridman Podcast #426</a>：Edward Gibson 是麻省理工学院（MIT）的心理语言学教授，并领导 MIT 语言实验室。请通过查看我们的赞助商来支持本播客：- Yahoo Financ...</li><li><a href="https://youtu.be/ZcoOW8nqVP8?t=140">Hugging Face 被黑了</a>：链接：主页：https://ykilcher.com 商店：https://ykilcher.com/merch YouTube：https://www.youtube.com/c/yannickilcher Twitter：https://twitter.com/ykilcher Dis...</li><li><a href="https://github.com/NVlabs/DoRA">GitHub - NVlabs/DoRA: DoRA 的官方 PyTorch 实现：权重分解低秩自适应（Weight-Decomposed Low-Rank Adaptation）</a>：DoRA 的官方 PyTorch 实现：权重分解低秩自适应 - NVlabs/DoRA</li><li><a href="https://tenor.com/view/regretting-thinking-nervous-macaco-monkey-gif-13105982953111325972">后悔思考 GIF - 后悔思考 紧张 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1230149199388082196)** (756 条消息🔥🔥🔥): 

- **Llama 3 热潮**：社区对 **[Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)** 的性能赞不绝口。MMLU 和 GSM-8K 等基准测试的提升备受关注，模型参数量从 8B 到可能高达 400B，其性能足以媲美或超越现有的 GPT-4 和各种尺寸的 Mixtral。

- **GGUF 的烦恼与胜利**：多位用户报告了 **Llama 3** 的 GGUF（量化）版本存在问题，特别是在 tokenization 和文本生成停不下来方面。不过，据报道来自 **[LM Studio 的 GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)** 运行良好。

- **上下文长度担忧**：尽管令人兴奋，但人们对 Llama 3 的上下文长度表示担忧，一些人更倾向于在某些任务中使用 **Mixtral 8x22B** 等具有更长上下文的模型。

- **Mistral 模型访问受限**：突然有报道称 **Mistral 模型被设为受限访问（gated）**，这引发了关于潜在原因的讨论，包括对欧盟新法规的猜测，以及有人评论说，由于它们采用 Apache 2 许可证，任何人都可以重新托管这些模型。

- **对微调能力的期待**：社区成员渴望对 Llama 3 进行微调（finetuning），讨论了各种可能性，并期待 **[MLX](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_qlora_single_device.yaml)** 微调可能释放的工具和能力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lluminous.chat/?sl=AwD1Ik">lluminous</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct">NousResearch/Meta-Llama-3-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/N8Programs/Coxcomb-GGUF">N8Programs/Coxcomb-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF">NousResearch/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.meta.ai">Meta AI</a>: 使用 Meta AI 助手完成任务，免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用了 Emu,...</li><li><a href="https://huggingface.co/N8Programs/Coxcomb">N8Programs/Coxcomb · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de">maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.udio.com/songs/kUvkoiz1maRm5BTAMKUQTk">Udio | Back Than Ever by drewknee</a>: 制作你的音乐</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B">NousResearch/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/lvcxIR3HNOw">LLM running on Roland MT-32</a>: LLM 在 Roland MT-32 上运行。演示来自 N8 的新型故事讲述模型 &quot;https://huggingface.co/N8Programs/Coxcomb&quot;。</li><li><a href="https://tenor.com/view/angry-mad-angry-face-cringe-angry-face-coach-not-happy-gif-5480965892207921425">Angry Mad GIF - Angry Mad Angry face - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c6pzpq/my_specialized_creative_writing_model_coxcomb/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://tenor.com/view/diablo-joke-meme-is-this-an-out-of-season-april-fools-joke-out-of-season-gif-16662191">Diablo Joke GIF - Diablo Joke Meme - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_qlora_single_device.yaml">torchtune/recipes/configs/llama3/8B_qlora_single_device.yaml at main · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=PYZIOMvkUF8">How Did Open Source Catch Up To OpenAI? [Mixtral-8x7B]</a>: 现在就使用此链接报名参加 GTC24！https://nvda.ws/48s4tmc。关于 RTX4080 Super 的抽奖活动，详细计划仍在制定中。然而...</li><li><a href="https://blog.allenai.org/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d">OLMo 1.7–7B: A 24 point improvement on MMLU</a>: 今天，我们发布了 70 亿参数开源语言模型 OLMo 1.7–7B 的更新版本。该模型在 MMLU 上得分为 52，位列...</li><li><a href="https://github.com/asg017/sqlite-vss">GitHub - asg017/sqlite-vss: A SQLite extension for efficient vector search, based on Faiss!</a>: 一个基于 Faiss 的高效向量搜索 SQLite 扩展！- asg017/sqlite-vss</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">Support Llama 3 conversion by pcuenca · Pull Request #6745 · ggerganov/llama.cpp</a>: 分词器（Tokenizer）是 BPE。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rules](https://discord.com/channels/1053877538025386074/1151297754992234496/1230717345866190949)** (1 条消息): 

- **引入了新的举报命令**: 用户可以使用 `/report` 命令举报垃圾邮件发送者、诈骗者和其他违反规则的人员。管理员将收到通知并审核该举报。
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1230153468828712960)** (11 条消息🔥):

- **Hermes 2 Pro 工具调用行为**：一位成员提到 **Hermes 2 Pro** 总是返回 **<tool_call>** 的困扰，而有时其实需要的是聊天回复。另一位成员强调 **Hermes 2 Pro** 需要更好地理解何时不触发工具调用，并提到未来的版本将解决这个问题。

- **基座模型的微调实践**：一位成员讨论了一种多阶段微调方法，涉及预训练基座模型和指令数据集，随后进行偏好数据集微调。他们遇到了模型在给出答案的同时返回随机句子或信息的问题。

- **使用 Hermes 2 Pro 直接回答**：有人建议添加一个名为 "directly_answer" 的工具（类似于 Langchain 的 ReAct Agent），用于 **Hermes 2 Pro** 应该进行聊天而不是执行函数调用的场景，并提供了一个操作方式的 JSON 示例。

- **Hermes 2 Pro 的 Prompting 问题**：一位成员征求关于 **NousResearch/Hermes-2-Pro-Mistral-7B** 模型正确 Prompting 格式的建议，并提供了一段代码片段，指出难以获得理想的输出。

- **GPU 集群上的长上下文推理挑战**：两条消息提到了技术挑战——一条关于在 GPU 集群上运行 **long context inference**，另一条关于使用双 A100 GPU 配合 **jamba** 处理 200k tokens。
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1230694358920073259)** (1 messages): 

- **树莓派上的 VLM**：一位成员表示打算将该技术用于学校项目，目标是在 **Raspberry Pis** 上安装 **VLM**，既为了乐趣也为了获益，并对共享资源的实用性表示认可。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1230144250927255562)** (27 messages🔥): 

- **辩论 OAI Assistants 搜索实现**：关于 [OpenAI assistant search approach](https://platform.openai.com/docs/assistants/tools/file-search/how-it-works?context=streaming) 的讨论深入探讨了其固定的 800-token 分块大小、50% 的重叠以及每个上下文最多 20 个分块的设定。
- **Embedding 中的维度困境**：在搜索中使用 **256 维 Embedding**（而非典型的 1500 维）引发了关于维度对模型性能影响的对话，并引用了 [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)（维度灾难）。
- **模型性能与优化**：有关于通过实验确定低维 Embedding 是否能带来更高检索准确率的讨论，一位成员表达了**亲自测试**的雄心。
- **对多模态模型的好奇**：成员们强调了当前模型（如 **gpt4v**）的优势和局限性，讨论了微调的必要性，并表示有兴趣将 qwen-vl-max 的文档理解能力蒸馏到更小的模型（如 **llava**）中。
- **GPT 变体与开源见解**：对话还涵盖了开源模型的经验，包括它们对特定任务微调的需求，以及 **OCR** 和 **vision models** 在搜索应用中提取元数据的潜在效率。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Curse_of_dimensionality">Curse of dimensionality - Wikipedia</a>: no description found</li><li><a href="https://github.com/furlat/Abstractions/blob/main/raw_notes/abstractions_types_no_cat_theory.md">Abstractions/raw_notes/abstractions_types_no_cat_theory.md at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1230130207701074011)** (312 messages🔥🔥): 

- **WorldSim 期待感升温**：Nous Research AI Discord 的成员们对 WorldSim 的回归充满期待，频繁询问其回归的具体时间。尽管尚未确认发布时间，但情绪乐观，多次确认表明回归已迫在眉睫。

- **Nitro 赠送赌注增加**：在对 WorldSim 的兴奋中，**kainan_e** 开玩笑地冒着破产风险，提议如果东部时间午夜前未发布，将向用户赠送 Discord Nitro。无论 WorldSim 的发布状态如何，这种善意之举都在延续，Nitro 正在分发中。

- **哲学深度探讨**：对话深入探讨了 WorldSim 的哲学基础，讨论了 AI 与用户叙事的复杂交互，以及在模拟中建立用户引导的 AI 文明的潜力。文中引用了 Desideratic AI (DSJJJJ) 哲学，强调了从组织复杂性中产生的涌现认知。

- **限制与费用说明**：提到了新版 WorldSim 可能存在的限制和费用，以防止滥用，这可能与之前因过度和操纵性输入导致被迫关闭的攻击有关。

- **最后倒计时与社区支持**：随着预期的 WorldSim 发布时间临近，社区成员因共同的兴奋和悬念而团结在一起，**kainan_e** 通过幽默和赠品支持成员的热情，而 **proprietary** 则暗示开发已接近完成。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1053877538025386074/1221910674347786261/1230614268907491429">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://discord.gift/FCeZVDtEepukaMbJ">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://worldsim.nousresearch.com/">world_sim</a>：未找到描述</li><li><a href="https://tenor.com/view/let-me-in-crazy-funny-silly-gif-13908292">Let Me In Crazy GIF - Let Me In Crazy Funny - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://subgenius.fandom.com/wiki/Pipes">Pipes</a>：Pipes 是吸食 frop 的必需品。"Bob" 总是抽着装满 frop 的烟斗。每个 SubGenius 都有一个装满 frop 的烟斗，并且不停地抽。通常，SubGenii 会发现一张名人的照片...</li><li><a href="https://tenor.com/view/fire-writing-gif-24533171">Fire Writing GIF - 火焰文字 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/anime-excited-happy-smile-gif-15060821">Anime Excited GIF - 动漫兴奋快乐 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/finding-nemo-escape-ninja-crab-seagulls-gif-3510044">A GIF - 海底总动员逃脱忍者 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mika-kagehira-kagehira-mika-ensemble-stars-enstars-stimming-isnt-enough-i-need-to-explode-gif-8612633247313789699">Mika Kagehira Kagehira Mika GIF - Mika kagehira Kagehira mika Ensemble stars - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/tree-fiddy-south-park-lock-gif-5759991">Tree Fiddy GIF - Tree Fiddy South - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/tea-tea-sip-anime-gif-25535884">Tea Tea Sip GIF - 动漫品茶 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://nousresearch.com/forge/">Forge - NOUS RESEARCH</a>：NOUS FORGE 下载将于 2024 年 6 月推出</li><li><a href="https://nousresearch.com/dsjjjj-simulacra-in-the-stupor-of-becoming/">DSJJJJ: Simulacra in the Stupor of Becoming - NOUS RESEARCH</a>：Desideratic AI (DSJJJJ) 是一项哲学运动，专注于利用传统上存在于一元论、分体论和语言学中的概念来创建 AI 系统。Desidera 旨在创建能够作为更好...</li><li><a href="https://nousresearch.com/dsjj">DSJJJJ: Simulacra in the Stupor of Becoming - NOUS RESEARCH</a>：Desideratic AI (DSJJJJ) 是一项哲学运动，专注于利用传统上存在于一元论、分体论和语言学中的概念来创建 AI 系统。Desidera 旨在创建能够作为更好...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1230139695250407424)** (515 条消息🔥🔥🔥): 

- **Llama 3 性能评估**：用户正在使用各种预设和设置测试 [Llama 3 8b model](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)。人们对 Llama 3 与其他模型的对比很感兴趣，特别是在不同系统配置下的连贯性和速度，包括使用 Ryzen 5 3600 的系统以及另一位使用配备 Core i5 8350u 笔记本电脑的用户。该模型的性能被广泛认为很有前景，尽管一些人遇到了意外输出，例如反复出现 `<|eot_id|>assistant`。

- **Llama 3 优化挑战**：用户注意到在 LM Studio 上本地运行 Llama 3 时 CPU 利用率很高，特别是在配备 Intel iGPU 等集成显卡的系统上，这表明 Llama 3 在独立 GPU 上运行可能更高效。一些用户遇到了运行缓慢或多语言响应效果不佳的问题，且在不同系统配置下的 Token 生成速度也各不相同。

- **Llama 3 集成与兼容性**：用户正在咨询 Llama 3 与各种应用和平台的兼容性，例如 VSCode Copilot。一位用户分享了使用 [Continue.dev](https://continue.dev/docs/reference/Model%20Providers/lmstudio) 将 LM Studio 中的 LLM 集成到其他平台的细节。此外，人们还对该模型是否能辅助特定任务或通过 Fine-tuning 来提升性能表现感兴趣。

- **小型模型评估**：关于 Phi 2 和 Llama 3 1.1B 等小型 LLM 效能的讨论引发了对其连贯性以及将其嵌入设备执行特定功能的潜力的疑问。用户正在讨论针对专门任务优化和 Fine-tuning 小型模型，并考虑在低功耗设备上运行 AI 的影响。

- **用户故障排除**：几位用户正在排查其 LM Studio 设置中的问题，从 GPU 利用率问题到聊天会话之间设置行为不一致等。关于运行 Llama 3 的最佳配置和设置，用户之间进行了持续的交流，分享了提升模型性能的经验和解决方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://continue.dev/docs/reference/Model%20Providers/lmstudio">LM Studio | Continue</a>：LM Studio 是一款适用于 Mac、Windows 和 Linux 的应用程序，可以轻松地在本地运行开源模型，并配有出色的 UI。要开始使用 LM Studio，请从网站下载...</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>：在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://missionsquad.ai/">Mission Squad. Flexible AI agent desktop app.</a>：未找到描述</li><li><a href="https://monaspace.githubnext.com/">Monaspace</a>：一个创新的代码字体超家族</li><li><a href="https://x.com/AIatMeta/status/1780997403979735440">来自 AI at Meta (@AIatMeta) 的推文</a>：介绍 Meta Llama 3：迄今为止功能最强大的开源 LLM。今天我们将发布 8B 和 70B 模型，它们提供了诸如改进推理等新功能，并树立了新的行业领先水平...</li><li><a href="https://huggingface.co/blog/llama3">Welcome Llama 3 - Meta's new open LLM</a>：未找到描述</li><li><a href="https://tenor.com/view/puss-in-boots-math-i-never-counted-the-last-wish-gif-27436638">Puss In Boots Math GIF - Puss In Boots Math I Never Counted - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/johnny-english-agent-yawn-tired-rowan-atkinson-gif-17502421">Johnny English Agent GIF - Johnny English Agent Yawn - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta Releases LLaMA 3: Deep Dive &amp; Demo</a>：今天，2024 年 4 月 18 日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是该系列的第三个迭代版本...</li><li><a href="https://github.com/arcee-ai/mergekit">GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.</a>：用于合并预训练大语言模型的工具。- arcee-ai/mergekit</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6414">Improve cpu prompt eval speed by jart · Pull Request #6414 · ggerganov/llama.cpp</a>：此更改上游化了 llamafile 的 CPU 矩阵乘法内核，从而提高了图像和 Prompt 评估速度。首先，Q4_0 和 Q8_0 权重在 CPU 上的运行速度应提高约 40%。最大的收益...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1230150400079036477)** (559 条消息🔥🔥🔥):

- **Meta Llama 3 模型发布**：Meta Llama 3，特别是 8B Instruct 版本已经发布，现在可以在 [Hugging Face](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF) 上获取各种量化版本。用户正在尝试 prompt 预设，以解决意外的输出重复和 prompt 循环问题。
- **Llama 3 预设困惑**：成员们正在分享并寻求关于 Llama 3 prompt 模板正确配置的建议，据报道，使用相关 GitHub 讨论中提到的最新预设和修复方案取得了一些成功。
- **Llama CPP 问题与修复**：用户们正在积极讨论 llama.cpp 的细节以及与 GGUF 转换中的 special tokens 相关的问题，社区贡献者正在努力解决围绕模型生成行为的问题。
- **性能与硬件讨论**：关于 Llama 3 在不同规格下的性能表现引发了热议和趣闻，特别是对高效运行大型模型的关注。讨论内容包括 RAM 需求详情以及在 Mac M1 和 M3 Max 上的测试。
- **量化与质量**：社区成员正在辩论不同量化级别之间可感知的质量差异，并解释说，尽管最初有不同看法，但在较小模型中 Q4 和 Q8 可能不会显示出显著差异。他们还讨论了使用 2-bit IQ quants 的潜力和速度，且质量下降极小。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">Qwen/CodeQwen1.5-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>: 在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://benchmarks.llmonitor.com/">LLM Benchmarks</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B">meta-llama/Meta-Llama-3-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/tree/main">QuantFactory/Meta-Llama-3-8B-Instruct-GGUF 在 main 分支</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF">MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF">MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，支持广泛的应用场景。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-same-thing-we-do-every-night-take-over-the-world-mad-pissed-angry-gif-4596927">The Same Thing We Do Every Night Take Over The World GIF - The Same Thing We Do Every Night Take Over The World Mad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://useanything.com/">AnythingLLM | 终极 AI 商业智能工具</a>: AnythingLLM 是为您的组织打造的终极企业级商业智能工具。拥有对 LLM 的无限控制、多用户支持、内外向工具支持以及...</li><li><a href="https://tenor.com/view/dog-angry-rabid-gif-20257620">Dog Angry GIF - Dog Angry Rabid - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/discussions/2">meta-llama/Meta-Llama-3-70B-Instruct · 更新 generation_config.json</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/tree/main">QuantFactory/Meta-Llama-3-8B-GGUF 在 main 分支</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/4">meta-llama/Meta-Llama-3-8B-Instruct · 更新 generation_config.json</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json 在 main 分支 · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md">llama3/MODEL_CARD.md 在 main 分支 · meta-llama/llama3</a>: Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账号来为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://tenor.com/view/we-know-duh-hello-of-course-gif-13989211">We Know Duh GIF - We Know Duh Hello - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://rentry.co/4kgm98oo">TEMPLATE &quot;&quot;&quot;&#123;&#123; if .System &#125;&#125;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;</a>: &#123;&#123; .System &#125;&#125;&lt;|eot_id|&gt;&#123;&#123; end &#125;&#125;&#123;&#123; if .Prompt &#125;&#125;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt; &#123;&#123; .Prompt &#125;&#125;&lt;|eot_id|&gt;&#123;&#123; end &#125;&#125;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&g...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6747">llama3 系列支持 · Issue #6747 · ggerganov/llama.cpp</a>: llama3 已发布，很高兴能在 llama.cpp 中使用 https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6 https://github.com/meta-llama/llama3
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1230639696393273395)** (1 条消息):

- **将 Llama 3 引入 LM Studio**：**MetaAI 的 Llama 3** 现已在 **LM Studio 0.2.20** 中可用。请从 [LM Studio 官网](https://lmstudio.ai)获取更新，或直接重启应用进行自动更新。
- **GGUF 兼容性说明**：目前，只有来自 "lmstudio-community" 的 GGUF 文件可与 **Llama 3** 配合使用。请在 [Hugging Face](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) 上查找。
- **社区模型亮点**：本次更新重点展示了由社区创作的令人印象深刻的新模型，详情请见 [Discord](https://discord.gg/aPQfnNkxGC) 上的 LM Studio 社区模型亮点计划。
- **模型详情解析**：**Llama 3** 是一个 8B 参数的指令微调模型，以其紧凑的体积、速度以及执行指令的精准度而著称。请注意，目前针对 GGUF 问题有一个临时解决方案，其他 GGUF 可能无法正常工作。
- **Bug 报告频道**：用户应在 ID 为 **#1139405564586229810** 的指定 Discord 频道中报告有关 **Llama 3** 的任何问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai,">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/LMStudioAI/status/1781087087745274116">来自 LM Studio (@LMStudioAI) 的推文</a>：.@Meta 的 Llama 3 现已在 LM Studio 中获得全面支持！👉 更新至 LM Studio 0.2.20 🔎 下载 lmstudio-community/llama-3。Llama 3 8B 已经上线，70B 即将推出 🦙 https://huggingface.co/...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1230219142330712147)** (8 条消息🔥): 

- **请求添加 Stable-Diffusion**：一位用户建议将 **stable-diffusion.cpp** 添加到 LM Studio，表示这将增强软件功能。另一位用户做出了回应，通过指出 "LM" 的含义来暗示 LM Studio 的专注领域。

- **Hermes Mixtral 模型加载问题**：**Loopyluci** 报告了在尝试加载不同的 **Hermes Mixtral 模型** 时的错误消息。分享了详细的错误日志，显示 *"(Exit code: 42). Unknown error. Try a different model and/or config."*。

- **Ollama 加载无误**：与 **Hermes Mixtral 模型** 的问题形成对比，同一位用户指出加载 **Ollama 模型** 时没有问题。

- **新模型排序功能获得认可**：**Pwrreset** 对下载页面新增的模型排序功能表示认可和赞赏。

- **关于文本转语音 (TTS) 集成的咨询**：**Ippity** 询问了将 **text-to-speech (TTS)** 集成到 LM Studio 的可能性，以避免整天阅读。

- **模型更换后的持续 Bug**：**Justmarky** 提到一个反复出现的 Bug，即在加载新模型后关闭最后一个聊天会导致模型卸载并需要重新加载。此外，软件在模型中设置预设后仍无法记住选定的预设。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1230136459558785054)** (16 条消息🔥): 

- **LM Studio 的模型兼容性问题**：一位成员表示在 **LM Studio** 上运行来自 **HuggingFace** 的名为 *ktkeller/mem-jasper-writer-testing* 的模型时遇到困难，并寻求使用此类模型的帮助或替代方案。
- **寻找联盟营销模型**：同一位成员提到自己是一名**联盟营销人员**，正在寻找能帮助撰写电子邮件和广告活动的模型。他们向拥有丰富编程知识并能交付预期成果的人提供**合作伙伴关系**。
- **对模型训练和输出的怀疑**：该用户强调他们遇到的模型产生的输出过于通用且缺乏特定训练，表明需要更复杂的解决方案。
- **关于合作与投资的真知灼见**：针对联盟营销人员的提议，另一位参与者建议，相比于投机性的合作伙伴关系，**开发者更有可能参与有经济补偿的项目**。
- **创意生成和简历评分的挑战**：不同的成员正在寻求生成非通用创意的技巧，以及使用 **Mixtral** 或 **Open-Webui** 有效评估简历的策略，对此有人建议采用务实的两步流程。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1230157914958991444)** (21 条消息🔥):

- **双 P100 设置的问题**：有人在双 **P100 GPUs** 设置中遇到了处理性能无法完全发挥的问题，尽管设备管理器已识别到显卡。据报告，VRAM 已被占用，但实际的 GPU 处理能力并未被利用，即使连接了 NVLink 桥接器也是如此。

- **同时使用 GTX 和 RTX 显卡**：据称，不同的 NVIDIA GPU（如 GTX 1080Ti 和 RTX 4070）无法通过 SLI 运行以进行联合处理，但在需要时两者都可以贡献 VRAM。

- **热节流（Thermal Throttling）问题已解决**：一位用户通过调整主板设置防止过热和温度激增，解决了系统的热节流问题，使运行温度降低了约 20°C。

- **分享 AI 系统配置**：成员们分享了他们的系统配置，如 12900k/4090/128GB 方案，并讨论了这些配置处理各种 AI 模型的能力，以及在游戏和业余爱好方面的用途。

- **关于 2080Ti 性能的咨询**：有人询问 **2080Ti 22G** 在运行 *Llama 7b* 等模型时的性能表现，征求用户在速度和有效性方面的经验。

---

**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1230176685949845504)** (13 messages🔥): 

- **WizardLM-2-8x22B 受到青睐**：一位成员称赞了 **WizardLM-2-8x22B** 模型的出色表现，并推荐他人尝试，强调了开源与闭源 LLM 之间的对等性。
- **本地设置对模型大小的担忧**：一位成员对在本地运行像 **WizardLM-2-8x22B** 这样的大型模型表示担忧，随后从他人处得知，在像 **Nvidia 4090** 这样的 24GB 显存显卡上运行该模型并不可行。
- **M3 Max 上 Q5_K_M 的性能优化**：参与者分享了运行 **Q5_K_M** 的具体设置，其中一人在拥有 **128GB RAM** 的 **M3 Max 40C** 上运行速度约为 **3.5tok/s**，另一人分享了其配置：*n_batch 512, n_ctx 4096, n_gpu_layers max, use_mlock on*。
- **LM Studio 进程优先级调整**：有关于通过调整进程优先级以及使用名为 **cocktail** 的工具进行内存管理，来提高 Mac 上 AI 模型运行器性能的讨论。
- **Llama 3 模型 Git Clone 效率**：一位成员提供了从 **Hugging Face** 克隆模型的技巧，通过在 git 命令中使用 `GIT_LFS_SKIP_SMUDGE=1` 来避免双重存储开销，并附带了一个建议的 bash 序列来处理**多个文件**。

**提及的链接**：<a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>：未找到描述

---

**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1230286609518428191)** (8 messages🔥): 

- **MissionSquad GitHub 仍未公开**：**MissionSquad** 的 GitHub 仓库目前仍保持 *private*（私有）状态。如果用户需要特定信息或访问权限，建议直接咨询。
- **窥探 MissionSquad 的技术栈**：MissionSquad 是一款使用 **Vue3** 和 **TypeScript** 开发的 *Electron app*，符合现代 Web 应用程序标准。
- **Prompt Studio 在 MissionSquad V1.1.0 中发布**：**Prompt Studio** 已在 **MissionSquad V1.1.0** 中推出，该功能允许用户微调提示词并将其保存到 Agent 的配置中。
- **对代码透明度的顾虑**：用户对在未查看源代码的情况下运行 MissionSquad 等应用程序表示犹豫，尽管对隐私问题表示理解。
- **MissionSquad 安全保证**：使用 MissionSquad 的风险等级被认为与访问任何其他网站相当，这意味着它采用了与 Chrome 等浏览器相同的标准 Web 安全模型。

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1230147507510575246)** (15 messages🔥):

- **LM Studio 的 GPU 选择问题**：一位用户遇到了 LM Studio 使用其 **AMD 集成显卡 (iGPU)** 而非独立显卡 (dGPU) 的问题。他们被引导至 [LM Studio 官方网站](https://lmstudio.ai/) 下载最新版本。
- **GPU 软件冲突的解决方法**：一位成员建议在设备管理器中禁用 iGPU，作为让 LM Studio 选择正确 GPU 的临时修复方案。
- **最新更新应解决 iGPU 问题**：关于 **0.2.19 ROCm 预览版** 的讨论指出，现在应该不再需要禁用 iGPU。如果用户在更新后仍遇到类似问题，建议提交错误报告。
- **旧款 AMD GPU 的兼容性问题**：一位用户报告 **LM Studio** 在其 **AMD 5700XT** 上可以运行，但另一位成员澄清该显卡不支持 HIP SDK，观察到的性能应该是基于 CPU 的。
- **AMD 上 8B 模型的推理异常**：一位成员在 AMD 环境下运行 **8B 模型** 时遇到问题，模型会出现自言自语的情况，这表明 AMD 硬件对大模型的支持尚不完整。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1230128509897146389)** (793 条消息🔥🔥🔥): 

- **关于 Llama 3 性能的讨论**：用户讨论了 Llama 3 令人印象深刻的基准测试结果，将 8B 模型的性能与 Llama 2 的 70B 模型进行了比较，并推测了未来的功能，如多模态支持和更长的上下文窗口。
- **在 Unsloth 中集成 Llama 3 的努力**：目前正积极努力在 Unsloth AI 中支持 Llama 3，分享了针对 8B 模型的 Google Colab 笔记本，并提到将整合 Llama 3 70B 的 4bit 量化版本。
- **Llama 3 提示词格式问题**：用户报告由于 End-of-Sequence (EOS) 标记问题，Llama 3-instruct 模型会生成无限输出，解决方法包括修改 `gguf` 文件或在 `llama.cpp` 中使用自定义标志。
- **对旧代 GPU 的支持**：有关于 Unsloth 是否继续支持旧代 GPU 的咨询，引发了关于用户偏好以及使用 Obsidian 等替代笔记工具进行个人知识管理的讨论。
- **Llama 3 衍生品的许可证和命名规范**：一位用户强调了遵守 Llama 3 新许可条款的重要性，该条款要求衍生品命名必须包含 "Llama 3" 前缀，并需要相应更新现有的许可信息。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/llama-3-70b-bnb-4bit">unsloth/llama-3-70b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://kolinko.github.io/effort/">Effort Engine</a>: 一种可能的新型 LLM 推理算法。平滑且实时地调整推理过程中想要进行的计算量。</li><li><a href="https://x.com/danielhanchen/status/1781024799227285799">Daniel Han (@danielhanchen) 的推文</a>: 为 Llama-3 8B 制作了一个 Colab！15 万亿 tokens！现在 @UnslothAI 已支持它！使用免费的 T4 GPU。正在进行基准测试，但比 HF+FA2 快约 2 倍，且节省 80% 内存！支持 4 倍长的上下文...</li><li><a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>: 继续推动 AI 前沿，让所有人都能使用。</li><li><a href="https://huggingface.co/meraGPT/mera-mix-4x7B">meraGPT/mera-mix-4x7B · Hugging Face</a>: 未找到描述</li><li><a href="https://replicate.com/pricing">定价 – Replicate</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/kuotient/Meta-Llama-3-8B">kuotient/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/llama3">欢迎 Llama 3 - Meta 的新型开源 LLM</a>: 未找到描述</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://youtu.be/pal-dMJFU6Q?si=euCqFFEUEDSLI8Yr&t=801">“Her” AI 即将来临？Llama 3、Vasa-1 以及 Altman 的“接入你想做的一切”</a>: Llama 3、Vasa-1 以及一系列新的采访和更新，AI 新闻就像伦敦的公交车一样接踵而至。我将花几分钟时间介绍最后一刻的 Llama...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jinaai/jina-reranker-v1-turbo-en">jinaai/jina-reranker-v1-turbo-en · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview">Microsoft Azure Marketplace</a>: 未找到描述</li><li><a href="https://tenor.com/view/sweaty-speedruner-gif-20263880">满头大汗的 Speedruner GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/NeuralNovel/Neural-Llama-3">NeuralNovel/Llama-3-NeuralPaca-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://t.co/l4S7MNciel">Google Colaboratory</a>: 未找到描述</li><li><a href="https://developers.facebook.com/llama_output_feedback">未找到标题</a>: 未找到描述</li><li><a href="https://www.instagram.com/reel/C56JwCSRMiS"> Mark Zuckerberg 在 Instagram 上表示</a>: “今天有重大的 AI 消息。我们正在发布新版本的 Meta AI，这是我们的助手，你可以在我们的应用和眼镜中向它提出任何问题。我们的目标是打造全球领先的 AI。

我们正在使用全新的、最先进的 Llama 3 AI 模型升级 Meta AI，并将其开源。有了这个新模型，我们相信 Meta AI 现在是你可以免费使用的最智能的 AI 助手。

通过将其集成到 WhatsApp、Instagram、Facebook 和 Messenger 顶部的搜索框中，我们让 Meta AI 变得更易于使用。我们还建立了一个网站 meta.ai，供你在网页端使用。

我们还开发了一些独特的创作功能，比如让照片动起来。Meta AI 现在生成高质量图像的速度非常快，甚至可以在你输入时实时创建和更新。它还会生成你创作过程的回放视频。”

享受 Meta AI，您可以关注我们新的 &#064;meta.ai IG 以获取更多更新。&quot;</a>: 103K 点赞, 6,182 条评论 - zuck 2024年4月18日：&quot;今天有重大的 AI 新闻。我们正在发布新版本的 Meta AI，这是我们的助手，您可以在我们的应用和眼镜中向它提问任何问题....</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/dance-gif-14880344851904561392">Dance GIF - Dance - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://obsidian.md/">Obsidian - 磨砺你的思维</a>: Obsidian 是一款私密且灵活的笔记应用，能够适应你的思维方式。</li><li><a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta 发布 LLaMA 3：深度解析与演示</a>: 今天，2024年4月18日，是一个特别的日子！在这段视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是其第三次迭代...</li><li><a href="https://gist.github.com/jedt/e45b337e9d9bd0492bf5d3c1d4706c7b">gist:e45b337e9d9bd0492bf5d3c1d4706c7b</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B 模型, Caesar Augustus, &amp; 1 GW 数据中心</a>: Zuck 关于：- Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展中的能源限制 - Caesar Augustus、智能爆炸、生物...</li><li><a href="https://github.com/ollama/ollama/pull/3699">Ollama.md 文档，由 jedt 提交 · Pull Request #3699 · ollama/ollama</a>: 关于从 Google Colab 笔记本设置微调后的 Unsloth FastLanguageModel 到：HF hub、GGUF、本地 Ollama 的指南。预览链接：https://github.com/ollama/ollama/blob/66f7b5bf9e63e1e98c98e8f4...</li><li><a href="https://github.com/unslothai/unsloth/issues/330">无法加载分词器 (CroissantLLM) · Issue #330 · unslothai/unsloth</a>: 尝试使用小型模型运行 colab：from unsloth import FastLanguageModel import torch max_seq_length = 2048 # 遗憾的是 Gemma 目前仅支持最高 8192 dtype = None # None 用于自动检测...</li><li><a href="https://arxiv.org/html/2401.13927v1">Adaptive Text Watermark for Large Language Models</a>: 未找到描述</li><li><a href="https://github.com/openai/triton/issues/194">支持 x86/ARM CPU (例如 Xeon, M1) · Issue #194 · openai/triton</a>: 你好，未来有支持 macOS 的计划吗？ ❯ pip install -U --pre triton 弃用提示：使用 distutils 配置文件配置安装方案已弃用，将不再起作用...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1c76n8p/official_llama_3_meta_page/">官方 Llama 3 META 页面</a>: [https://llama.meta.com/llama3/](https://llama.meta.com/llama3/)
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1230612487506886810)** (1 条消息): 

- **Llama 3 强势登场**：*Llama 3* 现已发布，与之前的版本相比，其微调 (finetuning) 速度翻倍，且内存占用减少了 60%。Meta 的新开源模型已获得全面支持，用户可以使用提供的 [Llama-3 8b Colab notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) 进行尝试。

- **新 4-Bit 模型发布**：随着 Llama-3 4-bit 版本的发布，用户可以访问更高效的模型，可在 Hugging Face 上获取：[Llama-3 8b, 4bit bnb](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) 和 [Llama-3 70b, 4bit bnb](https://huggingface.co/unsloth/llama-3-70b-bnb-4bit)。

- **邀请展示模型成果**：鼓励社区测试 Llama-3 并分享他们的模型和结果。建议不使用新 Colab 笔记本的用户更新 Unsloth 软件包。

**提到的链接**：<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colaboratory</a>: 未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1230151189794914305)** (15 条消息🔥):

- **CUDA 不适用于移动端神经网络**：一位成员指出，由于移动设备缺乏 CUDA，在手机上运行神经网络是不可行的，建议改用 *custom inference engines*（自定义推理引擎）。
- **iPhone 上的神经网络需要特殊处理**：讨论提到，虽然在 iPhone 上运行神经网络是可能的，但需要将模型编译成与专为 iPhone 硬件设计的特定神经网络推理引擎兼容的二进制格式。
- **TorchTune 停止支持旧版 GPU 令用户感到意外**：一位用户发现 TorchTune 已经停止支持旧版 GPU，这可能会影响使用旧硬件的用户。
- **HuggingFace 尚未对 LLAMA 3 开放 Inference API**：一位成员提到 HuggingFace 还没为 LLAMA 3 提供 Inference API，另一位成员幽默地回复说，训练完这类模型后已经没有算力剩余了，暗示了其资源密集型的特性。
- **YouTube 关于 AI 发展的见解**：分享了一个 YouTube 视频链接，提供了关于 Llama 3、Vasa-1 以及更广泛的 AI 新闻见解，并用伦敦巴士的比喻来描述 AI 新闻——要么不来，要么成群结队地频繁出现。

**提到的链接**：<a href="https://youtu.be/pal-dMJFU6Q?si=2wf152_TUTs4Np32&t=276">‘Her’ AI, Almost Here? Llama 3, Vasa-1, and Altman ‘Plugging Into Everything You Want To Do’</a>：Llama 3、Vasa-1 以及一系列新的采访和更新，AI 新闻就像伦敦巴士一样。我将花几分钟时间介绍最后一刻发布的 Llama ...

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1230125198318112778)** (96 messages🔥🔥): 

- **本地探索与大量的进度条**：一位成员提到在容器中运行 Unsloth/TRL 会导致进度条无法正常工作。该问题被归因于容器环境特有的 Python 奇特行为。

- **Unsloth 仓库咨询**：几位成员讨论了 Unsloth AI 仓库以及训练后不同格式的手动转换步骤。分享了 [Unsloth GitHub wiki](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf) 链接以辅助此过程。

- **发现量化难题**：用户讨论了执行 GPTQ 相关的内存消耗和困难，甚至指出 AutoGPTQ 库可能不支持多 GPU 量化。建议包括研究 rank stabilization（秩稳定），并可能使用 EXL2 量化作为更优的替代方案。

- **模型保存的误区与合并难题**：关于在训练脚本结束时保存模型的问题被提出，特别关注于合并以及使用不同词表转换为 16-bit。提供了关于保存方法的见解，并澄清在 checkpoint 期间仅保存 LoRA adapters，而不保存完整的模型权重。

- **寻求文本生成图像微调指导**：询问了关于为图像相关任务微调模型时准备数据集的问题。建议使用 URL 更好，因为与直接嵌入图像相比，URL 的内存需求更低。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ParasiticRogue/Merged-RP-Stew-V2-34B">ParasiticRogue/Merged-RP-Stew-V2-34B · Hugging Face</a>：未找到描述</li><li><a href="https://docs.mistral.ai/guides/tokenization/#control-tokens">Tokenization | Mistral AI Large Language Models</a>：Tokenization 是 LLM 的一个基本步骤。它是将文本分解为更小的子词单元（称为 tokens）的过程。我们最近在 Mistral AI 开源了我们的分词器。本指南将...</li><li><a href="https://colab.research.google.com/drive/11jGaCwi1lfbXKKbiLAMhMOBS5OAFgp-n?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>：速度快 2-5 倍，内存减少 80% 的 LLM 微调。通过在 GitHub 上创建账号为 unslothai/unsloth 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1230388860685389834)** (3 messages):

- **OpenHathi 旨在赋能印度语系**：[Sarvam.ai](https://www.sarvam.ai/blog/announcing-openhathi-series) 团队推出了 **OpenHathi**，强调了像 [Llama](https://ai.meta.com/llama/) 和 [Mistral](https://mistral.ai/news/announcing-mistral-7b/) 这样目前对印度语系支持有限的开源语言模型的重要性。他们强调了在对印地语等语言进行有意义的训练时，需要高质量且多样化的印度语系内容。
- **Mixtral 和 Mistral 22B 引起轰动**：[Mistral.ai](http://mistral.ai) 发布了 [Mixtral 8x22B](https://x.com/MistralAI/status/1777869263778291896)，这是一个在 [Substack 文章](https://datta0.substack.com/p/ai-unplugged-7-mixture-of-depths) 中讨论的 MoE 模型，与其前身相比更宽也更深。该文章还涉及了各种 AI 主题，包括 Reka Core 技术报告和 Google CodeGemma。
- **Neural Llama 加入 Alpaca**：在 [Hugging Face 平台](https://huggingface.co/NeuralNovel/Neural-Llama-3) 上推出了一款名为 **Neural Llama** 的新模型，该模型使用 **Unsloth AI** 系统进行训练。分享的链接中未详细说明该模型的具体能力和用途。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.sarvam.ai/blog/announcing-openhathi-series">OpenHathi 系列：一种低成本构建双语 LLM 的方法</a>：未找到描述</li><li><a href="https://huggingface.co/NeuralNovel/Neural-Llama-3">NeuralNovel/Llama-3-NeuralPaca-8b · Hugging Face</a>：未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-7-mixture-of-depths">AI Unplugged 7: Mixture of Depths,</a>：洞察胜过信息
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1230153404525973504)** (31 条消息🔥): 

- **探索双语模型策略**：对话涉及了创建混合语言数据集的可能性，以及将任务拆分为翻译阶段和特定语言查询阶段是否有效。一位成员提到使用 **Wikipedia 文章** 获取特定语言的数据，然后将 **英语翻译成印地语**。

- **翻译层的成本分析**：一位成员指出，使用类似 `translate(LLM(translate(instruction)))` 的函数可能会因为三次 LLM 调用而成本高昂，尽管这被认为是一个有效的解决方案。

- **关于 DNO 实现的疑问**：有人好奇 **Distributed Negotiation Optimization (DNO)** 是否在任何库中实现。成员们得出结论，目前还没有，但实现它应该很简单，因为它是 **Distributed Public Optimization (DPO)** 的迭代版本。

- **关于优化器的讨论**：聊天涉及了 **Adam 优化器** 及其内存高效变体 **Paged Adam** 之间的比较。对话暗示了通过实验观察结果和资源使用情况差异的意图，如果拥有充足的硬件（如 **A100 GPUs**），则更倾向于使用标准的 Adam 优化器。

- **对 ReFT 方法的兴趣**：一位成员询问是否将最近提到的 **ReFT (Refactorize and Fine-Tune) 方法** 集成到 Unsloth 中，以潜在地降低入门门槛。这引起了团队调查这一可能性的积极回应。
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1230259495330906194)** (27 条消息🔥): 

- **探索分块矩阵乘法 (Tiled Matrix Multiplication)**：一位成员讨论了*分块矩阵乘法*的效率，指出对于大矩阵，可能会隐式使用填充 (padding) 来实现分块。尽管填充区域增加了额外的计算，但这显著节省了内存带宽。

- **处理部分分块乘法**：讨论进展到明确处理矩阵乘法中部分分块的策略，例如单独处理 **7x7 矩阵** 中的 **6x6 分块**，或通过填充来实现规则分块。

- **Meta 发布 Llama 3**：分享了一个 **[Meta Llama 3](https://llama.meta.com/llama3/)** 的 YouTube 视频链接，其中 Mark Zuckerberg 讨论了各种因素，包括没有采用 MOE 模型以及正在进行的 **4050 亿参数密集模型 (dense 405 billion parameter model)** 的训练。

- **Meta Llama 3 中没有混合专家 (MOE)**：持续的讨论显示，新的 Meta Llama 3 模型有趣地没有包含 MOE 模型，这使其区别于一些 **最先进的架构 (state-of-the-art architectures)**。

- **Meta Llama 3 架构细节**：分享了关于 **Meta Llama 3** 的更多细节，提到它采用了 GQA、新的 tiktoken 分词器和更大的 rope theta，但仍保持了标准的 Llama 架构，易于集成和更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>: Zuck 关于：- Llama 3 - 朝向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - Caesar Augustus、智能爆炸、生物...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1230177788586229901)** (44 messages🔥): 

- **并行数据处理查询**：一名成员询问如何将 CSV 或 Parquet 格式的大型数据集加载到 CUDA C++ 代码中，但尚未找到高效的解决方案。有人建议将并行处理作为一种潜在方法。
- **CUDA Kernel 优化**：讨论围绕 CUDA Kernel 优化展开，特别是关于计算效率的 Block 维度设置。Triton Kernel 中硬编码与参数实验的有效性是焦点。
- **调试 CUDA Kernel 实现**：成员们就 Kernel 计算结果差异的问题交换了发现，关注零误差的准确性。有推测认为这些差异可能源于 Block 大小或激活分布等问题。
- **内存访问模式对性能的潜在影响**：一名成员试图了解 CUDA 中顺序内存访问与偏移内存访问之间的权衡，以及它如何影响向量乘法中的性能损失。
- **核心与 Warp 分配研究**：对话包含了对每个 SM 的 CUDA 核心和 Warp 分配的见解，建议每个 SM 达到 128 个活跃线程的最佳状态可以带来更好的性能结果。


  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1230436794705903617)** (5 messages): 

- **寻求 torch.compile 兼容性**：一名成员询问了在 `fullgraph=True` 时使自定义函数与 `torch.compile` 完全兼容的最佳实践。他们提到一些自定义 CUDA 或 Triton 模块可以与 `@torch.jit.ignore` 配合使用，而其他模块则会报错，导致困惑。

- **自定义 CUDA 扩展及其动态**：一个回复指向了一个 [GitHub pull request](https://github.com/pytorch-labs/ao/pull/135)，作为将自定义 CUDA Kernel 与 `torch.compile` 组合的示例，并提到正努力将此无缝集成到未来的 AO 贡献中。他们提供了一个更广泛的文档链接，但具体内容未直接分享。

- **自定义 Triton Kernel 指南**：分享了一个 [GitHub 文件](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661) 链接，以演示自定义 Triton Kernel 与 `torch.compile` 的组合。

- **FakeTensor/Meta-dispatch 支持的问题**：另一名成员建议参考之前链接的同一份文档，以解决 Kernel 中与 FakeTensor/Meta-dispatch 支持相关的问题，暗示该文档有效解决了个人遇到的挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit">C++ 自定义算子手册</a>：无描述</li><li><a href="https://github.com/pytorch-labs/ao/pull/135">msaroufim 的自定义 CUDA 扩展 · Pull Request #135 · pytorch-labs/ao</a>：这是 #130 的可合并版本 - 我必须进行一些更新：添加除非使用 PyTorch 2.4+ 否则跳过测试的逻辑，以及如果 CUDA 不可用则跳过测试；将 ninja 添加到开发依赖项...</li><li><a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9">C++ 自定义算子手册</a>：无描述</li><li><a href="https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661">pytorch/test/dynamo/test_triton_kernels.py</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=29ECwExc-_M
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1230136186933215292)** (55 messages🔥🔥):

- **WSL Ncu Profiler 难题**：一名成员在 Windows Subsystem for Linux (WSL) 上运行 **ncu profiler** 时遇到问题。尽管安装了 `nsight-compute`，但找不到 `ncu` 命令，导致建议验证并修正系统 **environment variables** 和 **PATH issues**。

- **WSL 上的 CUDA Profiling 解决方案**：分享了一篇来自 **peterchng.com** 的文章，讨论了在 WSL 上对 CUDA 程序进行 profiling 的解决方案。这包括确保安装最新的 Nvidia 驱动程序，并拥有 **Windows 11** 系统以配合 WSL 2 进行 CUDA 程序的 profiling。

- **CUDA 学习先决条件**：对于有兴趣学习 CUDA 的初学者，建议具备 **C/C++** 的基础知识以及在本地机器上运行 CUDA 代码的能力。分享了一个 YouTube 播放列表和一个用于学习 CUDA 编程的 GitHub 指南，以提供进一步帮助。

- **无需立即购买 GPU**：当被问及学习 CUDA 是否必须拥有 GPU 时，成员们建议不要立即购买。他们建议先开始学习，并在投资硬件之前考虑 CUDA 的未来应用。

- **用于深度学习模型的 CUDA**：一位成员表达了使用 CUDA 和 PyTorch 构建深度学习模型的意图。确认了 Google Colab 或 PaperSpace 可以作为本地 GPU 资源的替代方案，用于开始 CUDA 开发。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm">Environment Variables</a>：未找到描述</li><li><a href="https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/">Profiling CUDA programs on WSL 2</a>：未找到描述</li><li><a href="https://youtube.com/playlist?list=PL5Q2soXY2Zi-qSKahS4ofaEwYl7_qp9mw&si=HpLEqgkEOQ4hh_nS">Livestream - Programming Heterogeneous Computing Systems with GPUs and other Accelerators (Spring 2023)</a>：未找到描述</li><li><a href="https://github.com/CisMine/Parallel-Computing-Cuda-C">GitHub - CisMine/Parallel-Computing-Cuda-C</a>：通过在 GitHub 上创建账号来为 CisMine/Parallel-Computing-Cuda-C 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1230169198844444825)** (5 条消息): 

- **RingAttention 重新评估**：一名成员表示由于主职工作和个人事务的时间限制，无法继续从事 **RingAttention** 的工作，并提到讨论如何与工作组一起推进。
- **团队可用性确认**：另外两名成员确认了他们有时间讨论 **RingAttention** 项目的未来，表明对该项目的进展持续关注。
- **即将进行的讨论**：最初的成员随后表示他们很快会加入对话。
- **引用服务器构建**：另一名成员简要提到了服务器构建，可能与 **RingAttention** 项目或其基础设施有关。
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1230374656649859073)** (3 条消息): 

- **辨别谜题部分**：一位用户询问了 **Puzzle 4 and 5** 之间的区别，指出其中一个似乎只是增加了一个 `relu` 函数。
- **Triton 语言数学操作澄清**：作为回应，一位用户提供了一个指向 **Triton's math functions** 的链接 [此处](https://triton-lang.org/main/python-api/triton.language.html#math-ops)，暗示可以使用 Triton 自己的数学操作来实现 `relu` 函数，而无需依赖 `torch.relu`。

**提及的链接**：<a href="https://triton-lang.org/main/python-api/triton.language.html#math-ops">triton.language &mdash; Triton  documentation</a>：未找到描述

  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1230180530709069826)** (84 条消息 🔥🔥): 

- **量化中的轴和分组**：关于 Half-Quadratic Quantization (HQQ) 方法中 **axis=0 versus axis=1** 量化的有效性进行了技术讨论，并提供了代码片段和链接供审阅。使用 [hqq's quantize.py](https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85) 和 kernel 实现 [hqq_aten_cuda_kernel.cu](https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L109-L115) 等来源研究了权重矩阵访问模式的差异以及对 CUDA 性能的影响。

- **量化策略与 TinyGEMM**：小组讨论了 **tinyGEMM 的量化方法**，该方法倾向于**行向（归约维度）分组**，以及沿不同轴进行归约之间潜在的性能等效性。特别提到了 torchao 的 **int4mm** 如何能从 axis=1 分组中受益。

- **连接（Concatenation）对量化的影响**：有人提出了关于 Transformer 结构中 **Q, K, V 权重矩阵连接**所带来的量化挑战，这会影响分组量化的结果。引用了来自 [gpt-fast model.py](https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L175) 的 GPTQ 代码片段来阐明该问题。

- **探索新的量化技术**：一位成员提出了替代量化方法，例如使用**校准数据**或无需校准的**伪生成数据**，并分享了初步结果以及一个潜在优化策略的代码：[optimize_weights_autograd_fakedata](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412)。

- **量化和 HQQ 适配的未来增强**：讨论涵盖了将 HQQ 集成到 torchao 的可能性以及各种优化，参考了如 Xilinx 的 [brevitas GitHub pull request](https://github.com/Xilinx/brevitas/pull/937)。此外，还分享了关于利用查找表进行反量化的 4-bit 量化算子 (kernels) 的想法，尽管也指出了计算需求和效率方面的挑战。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L276-L405">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L109-L115">hqq/hqq/kernels/hqq_aten_cuda_kernel.cu at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L213">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：少于 1000 行 Python 代码实现的简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L175)">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>：少于 1000 行 Python 代码实现的简单高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85),">hqq/hqq/core/quantize.py at 63cc6c0bbb33da9a42c330ae59b509c75ac2ce15 · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/Xilinx/brevitas/pull/937">HQO for scale/zero point by Giuseppe5 · Pull Request #937 · Xilinx/brevitas</a>：无描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1230202510690615296)** (452 条消息🔥🔥🔥): 

- **解决微小上下文长度 Bug**：确认并修正了 `seq_len = 64` 过小的问题，以确保达到预期性能。

- **训练中的潜在改进**：在显存有限的 GPU 上训练 GPT 模型具有挑战性。讨论了训练中的内存使用情况，特别是关于是否有必要在训练过程中保留激活缓冲区 (activation buffers)。

- **代码库贡献与优化**：社区成员表现出为项目贡献和优化的意愿。这包括改进内存使用的建议，以及 cutlass 库对项目目标的实用性。

- **Attention 机制的效率增强**：讨论了重大的效率改进，例如将反向激活值的内存消耗从 9GB 降低到 1.5GB。另一个对话集中在优化 Attention 机制的反向传播过程，这可能会带来显著的性能提升。

- **关于 Fused Classifier Kernel 中 Padding 的讨论**：针对 Fused Classifier Kernel 中是否需要 Padding 以及利用 cuDNN 或 cuBLASLt 进行进一步优化的策略进行了多项技术讨论。重点考虑了 Classifier Kernel 与未填充（non-padded）vocab sizes 的兼容性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/war-dogs-war-dogs-movie-stressed-facepalm-gif-5727928">Stress GIF - War Dogs 电影 Stressed - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://developer.nvidia.com/nsight-compute">NVIDIA Nsight Compute</a>：用于 CUDA 和 NVIDIA OptiX 的交互式分析器。</li><li><a href="https://github.com/karpathy/llm.c/pull/169.">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/classifier_fused.cu#L327">karpathy/llm.c 项目中的 llm.c/dev/cuda/classifier_fused.cu</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=dB5Jxwj0PDw)">CUDA 教程 I 分析和调试应用程序</a>：使用 NVIDIA 开发者工具对 CUDA 进行分析、优化和调试。NVIDIA Nsight 系列工具可可视化硬件吞吐量并分析性能...</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md">NVIDIA/cutlass 项目中的 cutlass/media/docs/quickstart.md</a>：线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s81006/">Volta：架构与性能优化 | NVIDIA On-Demand</a>：本次演讲将回顾 Volta GPU 架构以及优化计算应用程序性能的相关指导。</li><li><a href="https://github.com/karpathy/llm.c/pull/170">ngc92 对 attention backward 的进一步改进 · Pull Request #170 · karpathy/llm.c</a>：线程复用寄存器中的数据以减少内存传输的 Backward kernel。此 PR 基于我之前的 PR，应先合并之前的。完成后，我将进行 rebase 并移除...</li><li><a href="https://github.com/karpathy/llm.c/pull/163">ngc92 通过在 backward 过程中跨层复用相同缓冲区来节省内存 · Pull Request #163 · karpathy/llm.c</a>：在 backward pass 期间跨层复用内存缓冲区。</li><li><a href="https://github.com/karpathy/llm.c/pull/150">ademeure 提供的 Fused Classifier 优化版本 + Bug 修复(?) · Pull Request #150 · karpathy/llm.c</a>：这是来自 #117 的酷炫新 kernel 的更快版本（仍仅限 /dev/cuda/）。最大的区别在于它针对每个 1024 宽度的 block 处理一行进行了优化，而不是每个 32 宽度的 warp...</li><li><a href="https://developer.nvidia.com/blog/boosting-productivity-and-performance-with-the-nvidia-cuda-11-2-c-compiler/#more_debug_information_for_even_the_most_optimized_code">使用 NVIDIA CUDA 11.2 C++ 编译器提升生产力和性能 | NVIDIA 技术博客</a>：11.2 CUDA C++ 编译器包含旨在提高开发者生产力和 GPU 加速应用程序性能的功能与增强。编译器工具链获得了 LLV...</li><li><a href="https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/">在 Kepler 上实现更快的 Parallel Reductions | NVIDIA 技术博客</a>：Parallel reduction 是许多并行算法的常见构建模块。Mark Harris 在 2007 年的一次演讲中提供了在 GPU 上实现 Parallel reductions 的详细策略...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1230188289877737503)** (9 条消息🔥):

- **小组讨论与 CUDA MODE 冲突**：本周的 **CUDA MODE** 活动将与一个小组讨论重叠，这意味着一名成员将无法出席，并正在寻求管理录制的帮助。
- **活动录制委派**：一名成员正在请求协助录制活动，并询问是否需要提前设置任何必要的 **permissions**。
- **寻求备用录制人员**：另一名成员被要求在 CUDA MODE 活动期间录制屏幕作为 **backup**。
- **活动记录协调**：讨论了详细的录制计划，包括屏幕录制、通过 BlackHole 进行音频采集、潜在的演讲者个人录制以及后期制作工作，还有活动描述和 repository 更新的责任分配。
- **团队努力获得赞赏**：对确保即将举行的活动得到妥善录制和分享的组织工作及 **team effort** 给予了高度评价。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1230144525155176448)** (399 条消息🔥🔥): 

- **SD3 表现平平**：用户报告称 [Stability AI 初始发布的](https://vxtwitter.com/StabilityAI/status/1780599024707596508) SD3 仅限 API 且表现不佳，人们对该模型的相对性能表示担忧，特别是其对文本的处理。尽管有传言称该模型存在问题，但人们对其通过付费墙实现潜在盈利仍有期待。
  
- **LAION 之后寻找替代数据集**：在 LAION 数据集从 Hugging Face 移除后，用户分享了 **coyo-700m** 和 **datacomp-1b** 等替代方案来训练 text to image diffusion 模型，并在简短的交流中确认了它们的效用。

- **推进 PAG 和 SDXL**：讨论指出，应用于 SDXL 的 [Perturbed Attention Guidance (PAG)](https://huggingface.co/spaces/multimodalart/perturbed-attention-guidance-sdxl) 显示出比之前输出更好的结果，尽管仍未超过 DALLE-3 的性能。

- **Stability AI 内部动荡曝光**：随着高层离职的消息传出，Stability AI 的处境似乎十分严峻，引发了对公司未来及其对更广泛开源 AI 模型格局影响的猜测。对话透露了对管理不善以及领导层变动后公司方向可行性的担忧。

- **AI 社区对 LLaMA 3 表现出兴趣**：最近发布的 LLaMA 3 及其测试因其在任务中的出色表现而引起了积极反响，尽管由于 context window 较小而存在限制。分享了一个 [讨论链接](https://www.llama2.ai/)，邀请更多用户尝试在线与模型互动。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.llama2.ai/">在 Replicate 上与 Meta Llama 3 对话</a>：Llama 3 是来自 Meta 的最新语言模型。</li><li><a href="https://huggingface.co/spaces/multimodalart/perturbed-attention-guidance-sdxl">Perturbed-Attention Guidance SDXL - multimodalart 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.cnbc.com/2024/04/18/ai-startup-stability-lays-off-10percent-of-employees-after-ceo-exit.html">AI 初创公司 Stability 在争议性 CEO 离职后裁员 10%：阅读完整备忘录</a>：根据 CNBC 获得的内部备忘录，Stability AI 在经历了一段不可持续的增长后，裁减了数名员工以“调整业务规模”。</li><li><a href="https://arstechnica.com/tech-policy/2024/04/feds-appoint-ai-doomer-to-run-us-ai-safety-institute/">联邦政府任命“AI 末日论者”负责美国研究所的 AI 安全</a>：前 OpenAI 研究员曾预测 AI 有 50% 的概率杀死全人类。</li><li><a href="https://huggingface.co/ptx0/terminus-xl-velocity-v2">ptx0/terminus-xl-velocity-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://www.meta.ai/?icebreaker=imagine">Meta AI</a>：使用 Meta AI 助手处理事务、免费创建 AI 生成的图像，并获取任何问题的答案。Meta AI 基于 Meta 最新的 Llama 大语言模型构建，并使用了 Emu...</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c6g6zz/comment/l010k13/>">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/jump-dive-swim-abandon-ship-under-attack-gif-14076767">Jump Dive GIF - 跳跃 潜水 游泳 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/shihaozhaozsh/lavi-bridge">GitHub - ShihaoZhaoZSH/LaVi-Bridge: 连接不同语言模型与生成式视觉模型以实现文本到图像生成</a>：连接不同语言模型与生成式视觉模型以实现文本到图像生成 - ShihaoZhaoZSH/LaVi-Bridge</li><li><a href="https://www.instagram.com/kushu.lofi">登录 • Instagram</a>：未找到描述</li><li><a href="https://www.instagram.com/ph">登录 • Instagram</a>：未找到描述</li><li><a href="https://www.instagram.com/philipp.igumnov">登录 • Instagram</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1230126354767286313)** (18 条消息🔥): 

- **GANs 与 LDM 性能对比**：有观点指出，与 Latent Diffusion Models 相比，GANs 在推理过程中可能更快，并受益于判别器组件的反馈循环。另一个讨论点强调，虽然 GANs 可能具有更高的参数效率，但往往更难训练，且与人类标准相比，通常生成的图像质量较低。

- **训练 GANs 的成本效益**：一位成员表示，从头开始训练 GAN 的成本比训练或微调扩散模型更低，尽管这可能因应用领域而异。

- **微软 VASA-1 实时对话面部生成**：分享了关于微软 VASA-1 的公告，该项目能够实时生成逼真的音频驱动对话面部，并附带了[项目链接](https://www.microsoft.com/en-us/research/project/vasa-1/)。

- **HQ-Edit 图像编辑数据集**：发布了关于 HQ-Edit 的信息，这是一个用于基于指令的图像编辑的高质量数据集。该数据集包含约 200,000 次编辑，是在先进基础模型的协助下构建的。提供了[数据集链接](https://thefllood.github.io/HQEdit_web/)以获取更多详情。

- **Meta 发布 Llama 3 大语言模型**：分享了 Meta 推出的开源大语言模型 Llama 3，包括未来在各种云和硬件平台上的可用性。成员们还就其预期影响和实际可访问性交换了一些看法。公告可以在[这里](https://ai.meta.com/blog/meta-llama-3/)找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://thefllood.github.io/HQEdit_web/">HQ-Edit: 一个用于基于指令的图像编辑的高质量数据集</a>：未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述</li><li><a href="https://www.microsoft.com/en-us/research/project/vasa-1/">VASA-1 - 微软研究院</a>：在新标签页中打开
</li>
</ul>

</div>
  

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1230207371201085460)** (296 messages🔥🔥): 

- **Llama 3 发布细节与推测**：Meta 发布了 **[Llama 3](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview)**，包含 **8B** 和 **70B** 模型，其特点是采用了 **基于 Tiktoken 的 tokenizer**、8k 上下文长度，且 Benchmark 测试显示出极具竞争力的性能。讨论中推测了其即时处理能力，以及通过 RoPE Scaling 等技术**扩展上下文长度**的改进方案。

- **Axolotl 提交 Llama 3 QLoRA 的 PR**：一个 **[Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536/files)** 已开启，旨在为 Axolotl 添加 **Llama 3 QLoRA** 支持，讨论围绕实现的各种技术细节展开。提到的 **CUDA 错误** 表明在 **80 GB** 配置上运行时可能存在潜在问题。

- **微调后合并模型适配器的问题**：一位用户在微调后尝试将 QLoRA 适配器合并到 Llama 3 基础模型时遇到了问题，错误与 **tokenizer.model** 有关。他们通过设置 `legacy=False` 并使用 `use_fast=True` 成功解决了合并问题。

- **训练与上下文大小的技术讨论**：针对将 **上下文大小** 扩展到 Llama 3 默认值之外进行了大量对话，涉及 `rope_theta` 和 `rope_scaling` 等参数。用户分享了在之前模型中缩放上下文长度的见解和经验，并提供了简短示例以及现有长上下文模型的链接。

- **社区对 Llama 3 影响现有工作的看法**：对于 Llama 3 的发布，反应各不相同；虽然有些人对其令人印象深刻的 Benchmark 感到兴奋，但也有人感叹自己长期开发的模型现在仅相当于新发布的模型。此外，对于 **70B 模型** 的有效性也存在怀疑，一些用户认为其相对于 8B 模型并没有显著的性能飞跃。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536/files">Adding Llama-3 qlora by monk1337 · Pull Request #1536 · OpenAccess-AI-Collective/axolotl</a>：添加 Llama-3 QLoRA，经测试可用。</li><li><a href="https://x.com/teortaxesTex/status/1781063292795883943">Teortaxes▶️ (@teortaxesTex) 的推文</a>：来自线程：Llama-3 8b 具有至少 32k 的近乎完美的大海捞针检索能力（RoPE theta 为 4）</li><li><a href="https://huggingface.co/hfl/chinese-llama-2-13b-16k">hfl/chinese-llama-2-13b-16k · Hugging Face</a>：未找到描述</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview">Microsoft Azure Marketplace</a>：未找到描述</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>：扎克伯格谈论：Llama 3、迈向 AGI 的开源、定制芯片、合成数据、扩展的能源限制、Caesar Augustus、智能爆炸、生物风险等...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1536">Adding Llama-3 qlora by monk1337 · Pull Request #1536 · OpenAccess-AI-Collective/axolotl</a>：添加 Llama-3 QLoRA，经测试可用。</li><li><a href="https://huggingface.co/datasets/xorsuyash/raft_datasetp1">xorsuyash/raft_datasetp1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/24653">Llama/GPTNeoX: add RoPE scaling by gante · Pull Request #24653 · huggingface/transformers</a>：此 PR 的作用？这是一个用于讨论的实验性 PR，以便我们决定是否添加此模式。背景：在过去的一周里，关于缩放 RoPE 的进展有几项...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1230527592545648640)** (11 messages🔥):

- **减少截断可增强语言模型**：AWS 的新打包算法因显著提升性能而受到关注，根据 [Arankomatsuzaki 的推文](https://x.com/arankomatsuzaki/status/1780778186348843253?s=46&t=hIokEbug9Pr72tQFuXVULA) 和这篇[研究论文](https://arxiv.org/abs/2404.10830)，该算法使阅读理解能力提升了 **+4.7%**，并将闭域幻觉（closed domain hallucinations）减少了 **58.3%**。
- **Llama-3 沿用标准架构**：尽管数据缩放论文引发了猜测，但已确认 **Llama-3** 使用了与其前代相同的架构。
- **Llama-3 继承了更大的 Tokenizer**：进一步的讨论澄清了虽然 **Llama-3** 拥有更大的 Tokenizer，但其底层架构保持不变。
- **探索 `AutoTokenizer` 与 Llama-3 的兼容性**：正在测试 `AutoTokenizer` 与 **Llama-3** 的配合使用，以及通过手动调整 PAD token 来修正潜在问题。
- **Llama-3 Tokenization 的奇特之处被揭示**：[@danielhanchen 的推文](https://x.com/danielhanchen/status/1781012164893118471?s=46&t=hIokEbug9Pr72tQFuXVULA)指出，**Llama-3** 的 Tokenization 具有特定特征，例如数字的拆分以及缺少 `unk_token`，这会影响使用 @UnslothAI 等工具进行的 finetuning。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1780778186348843253?s=46&t=hIokEbug9Pr72tQFuXVULA">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：AWS 展示了“减少截断可改进语言建模”。他们的打包算法实现了卓越的性能（例如，阅读理解相对提升 +4.7%），并减少了闭域幻觉...</li><li><a href="https://x.com/danielhanchen/status/1781012164893118471?s=46&t=hIokEbug9Pr72tQFuXVULA">Daniel Han (@danielhanchen) 的推文</a>：Llama-3 的其他一些奇特之处：1. 由于使用了 tiktoken，数字被拆分为 1、2、3 位数字（Llama 是单数字拆分），即 1111111 从左到右拆分为 111_111_1；2. 没有 unk_token？正尝试让 @UnslothAI...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1230278105814536253)** (44 条消息🔥): 

- **Docker 困境 —— 依赖冲突浮出水面**：一位成员在运行 `docker build` 时遇到错误，原因是 **axolotl[flash-attn]**（需要 **torch==2.0.0+cu118**）与其他需要不同版本 **torch** 的包之间存在依赖冲突。该成员不确定如何解决需求冲突并寻求帮助。

- **冻结层的挫败感 —— 冻结还是不冻结？**：在关于对 7b **Mistral** 模型进行层冻结 **finetuning** 的讨论中，一位成员报告称，除非取消冻结 *lmhead* 和 *embed layers*，否则会发生崩溃，这指向了关于冻结参数的文档匮乏。

- **Llama Finetuning 的跨越**：另一位成员正尝试 **finetune Llama-3**，并遇到了一些早期问题，包括关于 Tokenizer 中 padding token 的 `ValueError`。针对此问题，在特定频道中有一个包含[相关信息](https://discord.com/channels/1104757954588196865/1104757955204743201/1230567923668746249)的置顶解决方案。

- **Hijacking Llama-3 的障碍**：进一步的对话显示，一位成员尝试在 **Llama-3** 上使用 **hijack_llama**，但面临 *nan loss* 的障碍，这表明可能存在不兼容性或需要解决的配置问题。

- **Tokenizer 问题与微调策略**：有一场围绕 **Llama-3 Tokenizer 变化** 的对话，讨论是使用 base model 还是 instruct model 在指令数据集上进行 **fulltuning**。成员们就模型的适用性和配置交换了见解，其中一位成员注意到从 **Llama-2 的 `</s>`** token 变为了 **Llama-3 的 `pad_token: <|end_of_text|>`**。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1230231056117137468)** (2 条消息): 

- **不当内容警报**：成员们报告了聊天中出现的 **pornspam** 实例，这标志着违反了社区准则。
- **个人内容泄露**：提到了有关 **OnlyFans 账号** 的泄露，这引发了对隐私和未经授权内容分发的担忧。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1230331760617984011)** (4 条消息): 

- **Axolotl 中的 YAML 注释**：Axolotl 并不显式支持 YAML 配置文件中的注释。虽然你可以包含注释供自己参考，但在 [`load_cfg` 函数](https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/cli/__init__.py#L340L387)中使用 `yaml.safe_load` 解析时，它们会被忽略。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/cli/__init__.py#L340L387)">axolotl/src/axolotl/cli/__init__.py (main 分支) · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c0b8e2b9-8422-4a60-8d1f-257eb00f2808)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1230588800200085625)** (14 条消息🔥): 

- **在 YAML 配置中设置 PAD Token**：一位用户询问如何在 YAML 配置中设置 PAD Token。回复暗示将其包含在 `tokens` 部分下，但结果只是一个占位符，没有具体说明。

- **Tokenizer 中的 Token 替换**：一位用户寻求关于替换 Tokenizer 中 Token 的帮助，并获得了一个详细的代码示例。步骤包括使用 `add_tokens` 添加新 Token，然后手动更新 Tokenizer 词汇表中的 Token ID。

- **通过 YAML 替换 Token**：同一位用户进一步询问如何使用 YAML 配置文件替换 Tokenizer 中的 Token。回复概述了如何在 YAML 文件中定义新 Token，并使用调整后的预处理函数将原始 Token 替换为新 Token。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d1ef6577-52b8-44cd-8588-c33724be6c8e)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=484ff2b8-6849-4c46-a388-8e244cdca92d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=283dc0b2-eb24-4f4e-8b7d-1c24e9285c3d)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1230255128712577084)** (3 条消息): 

- **巨型模型 Mixtral 8x22b 发布**：新的 **Mixtral 8x22B Instruct** 模型拥有 39B 激活参数，具有高效率，并专注于数学、编程和多语言流利度。在[发布公告](https://mistral.ai/news/mixtral-8x22b/)中了解其令人印象深刻的能力并查看基准测试。

- **介绍 WizardLM-2 的迷人能力**：**WizardLM-2 8x22B** 是 Microsoft AI 的顶尖模型，拥有用于巅峰性能的 *nitro* 版本，而 **WizardLM-2 7B** 则在较小规模上提供速度和性能。在 [WizardLM](https://wizardlm.github.io/WizardLM2/) 了解更多关于这些迷人模型及其指令微调（instruct fine-tune）技术的信息。

- **Zephyr 的微调预测**：由 Hugging Face 微调的 **Zephyr 141B-A35B** 模型通过公共和合成数据集的混合提供了增强的能力。它与同样可用的 **WizardLM-2 7B** 一起在 [OpenRouter](https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b) 上展示。

- **各模型大幅降价**：**MythoMax Extended**、**Mixtral 8x7b Instruct** 等模型已大幅降价。这些降价的详细信息可在 [OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct) 上查看。

- **Mixtral 8x22B Instruct 的 Prompt 模板修正**：Mixtral 8x22B Instruct 的一个 Prompt 模板错误已得到解决，以消除任何困惑。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它包含 8 个专家，每个专家 220 亿参数，每个 token 同时使用 2 个专家。它通过 [X](https://twitter...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x22b-instruct">Mixtral 8x22B Instruct by mistralai | OpenRouter</a>: Mistral 官方对 [Mixtral 8x22B](/models/mistralai/mixtral-8x22b) 进行指令微调（instruct fine-tuned）的版本。在 141B 总参数中使用了 39B 激活参数，为其规模提供了无与伦比的成本效率。...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它展示了极具竞争力的性能，并持续超越所有现有的...</li><li><a href="https://openrouter.ai/models/huggingfaceh4/zephyr-orpo-141b-a35b">Zephyr 141B-A35B by huggingfaceh4 | OpenRouter</a>: Zephyr 141B-A35B 是一个混合专家（MoE）模型，拥有 141B 总参数和 35B 激活参数。在公开可用的合成数据集混合物上进行了微调。它是...的指令微调版本。</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-7b">WizardLM-2 7B by microsoft | OpenRouter</a>: WizardLM-2 7B 是 Microsoft AI 最新 Wizard 模型的较小变体。它是速度最快的，并能与现有的 10 倍大的开源领先模型达到相当的性能。它是一个微调...</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b:extended">MythoMax 13B by gryphe | OpenRouter</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge 注意：这是 [此模型](/models/gryphe/mythomax... 的扩展上下文版本。</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe | OpenRouter</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct">Mixtral 8x7B Instruct by mistralai | OpenRouter</a>: 由 Mistral AI 开发的预训练生成式稀疏混合专家模型（Sparse Mixture of Experts），用于聊天和指令用途。包含 8 个专家（前馈网络），总计 470 亿参数。指令模型微调...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct">Mistral 7B Instruct by mistralai | OpenRouter</a>: 一个 7.3B 参数的模型，在所有基准测试中均优于 Llama 2 13B，并针对速度和上下文长度进行了优化。这是 Mistral 7B Instruct 的 v0.1 版本。对于 v0.2，请使用 [此模型](/models/mistral...</li><li><a href="https://openrouter.ai/models/undi95/toppy-m-7b">Toppy M 7B by undi95 | OpenRouter</a>: 一个狂野的 7B 参数模型，使用来自 mergekit 的新 task_arithmetic 合并方法合并了多个模型。合并模型列表：- NousResearch/Nous-Capybara-7B-V1.9 - [HuggingFaceH4/zephyr-7b-be...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1230263363414659188)** (4 messages): 

- **订阅系统故障**：有用户报告 **订阅系统** 运行不正常；问题包括被重定向到账单页面而无法获得模型访问权限，以及支付后未收到确认邮件。

- **Product Hunt 上的初创公司亮点**：一位社区成员在 **Product Hunt** 上发布了他们的初创公司，并寻求群组的支持和反馈。该成员分享了一个 URL：[Product Hunt 上的 SpeedLegal](https://www.producthunt.com/posts/speedlegal)。

**提到的链接**：<a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - 您的个人 AI 合同谈判代表 | Product Hunt</a>: SpeedLegal 是一款 AI 工具，可帮助您更好地理解和谈判合同。它可以快速识别潜在风险，并用简单的语言解释复杂的法律术语。SpeedLegal 还会给出...

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1230158224397959188)** (318 messages🔥🔥):

- **Boston Dynamics 开启新一代机器人时代**：Boston Dynamics 推出了全新的全电动版 Atlas 机器人，专为实际应用设计，并在 [YouTube 视频](https://www.youtube.com/watch?v=29ECwExc-_M)中首次亮相。新款 Atlas 建立在数十年的机器人创新基础之上。
- **Mixtral 模型大显身手**：[Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 是来自 Mistral AI 的新模型，支持多种欧洲语言和包括 function calling 在内的高级功能。该模型被强调为高性能且高效率，并根据开源的 Apache 2.0 许可证发布。
- **Together AI 与 Meta 达成合作**：[Together AI 宣布](https://together-ai.webflow.io/blog/together-ai-partners-with-meta-to-release-meta-llama-3-for-inference-and-fine-tuning)与 Meta 合作发布 Meta Llama 3，用于推理和微调。该 API 提供高达每秒 350 个 tokens 的性能，包含预训练的 8B 和 70B 参数模型。
- **OpenRouter 提供多样性**：OpenRouter 的 Discord 用户讨论了 WizardLM、Claude 和 Mixtral 等 LLM 的优缺点，分享了如何减少模型审查的策略以及对 role play 应用的建议。该服务面向支持强大用例的提供商和模型，其中提到了 Together AI 使用 Mixtral 以及 Llama 3 的可能性，包括用于上下文扩展的自托管。
- **Meta 的 Llama 3 专业性受到质疑**：社区对 Llama 3 的多语言性能以及 Twitter 的 Grok 未被大规模采用的原因进行了推测，假设性能受限和基础模型的问题可能会阻碍提供商。这呼应了关于 OpenAI 的 Python 库不遵循 timeout 参数的讨论，以及用户在 OpenRouter 中集成 function calls 时相比直接使用 OpenAI 所面临的挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://qwenlm.github.io/blog/codeqwen1.5/">使用 CodeQwen1.5 编写代码</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 利用大语言模型 (LLM) 力量的高级编程工具的出现，显著提升了程序员的生产力...</li><li><a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>: 继续推动 AI 的前沿，并让所有人都能使用。</li><li><a href="https://together-ai.webflow.io/blog/together-ai-partners-with-meta-to-release-meta-llama-3-for-inference-and-fine-tuning">Together AI 与 Meta 合作发布用于推理和微调的 Meta Llama 3</a>: 未找到描述</li><li><a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=orNoaT1ei7RpUogkauM1-Q">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！  ↘️ 引用...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct - HuggingChat</a>: 在 HuggingChat 中使用 meta-llama/Meta-Llama-3-70B-Instruct</li><li><a href="https://giphy.com/gifs/robot-boston-creepy-ly2VUVUwtuHst1FhCq">机器人 GIF - 在 GIPHY 上查找和分享</a>: 发现并与你认识的每个人分享这个机器人 GIF。GIPHY 是你搜索、分享、发现和创建 GIF 的方式。</li><li><a href="https://llama.meta.com/llama-downloads">下载 Llama</a>: 申请访问 Llama。</li><li><a href="https://www.youtube.com/watch?v=29ECwExc-_M">全新 Atlas | Boston Dynamics</a>: 我们正在揭晓下一代人形机器人——一款专为现实世界应用设计的全电动 Atlas 机器人。新款 Atlas 建立在数十年的...</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>: 使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://docs.librechat.ai/install/configuration/ai_endpoints.html">✅ 兼容的 AI Endpoints</a>: 已知的兼容 AI Endpoints 列表，包含 `librechat.yaml`（即 LibreChat 自定义配置文件）的示例设置。</li><li><a href="https://azuremarketplace.microsoft.com/en-us/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=Overview">Microsoft Azure Marketplace</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-3.5-turbo">openai 的 GPT-3.5 Turbo | OpenRouter</a>: GPT-3.5 Turbo 是 OpenAI 最快的模型。它可以理解并生成自然语言或代码，并针对聊天和传统的补全任务进行了优化。由 OpenAI 更新以指向 [l...</li><li><a href="https://olympia.chat">Olympia | 优于 ChatGPT</a>: 通过价格合理的 AI 驱动顾问来发展您的业务，这些顾问是业务战略、内容开发、营销、编程、法律战略等方面的专家。</li><li><a href="https://tenor.com/view/chiquichico-gif-26004262">Chiquichico GIF - Chiquichico - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://deepinfra.com/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 - Demo - DeepInfra</a>: 这是 Mixtral-8x22B 的指令微调版本——来自 Mistral AI 的最新且最大的混合专家 (MoE) 大语言模型 (LLM)。这款最先进的机器学习模型使用了一个...</li><li><a href="https://leanpub.com/patterns-of-application-development-using-ai">使用 AI 的应用开发模式</a>: 探索构建智能、自适应且以用户为中心的软件系统的实用模式和原则，充分利用 AI 的力量。</li><li><a href="https://openrouter.ai/docs#required-parameters-(beta)">OpenRouter</a>: 构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1230144313237835807)** (133 条消息🔥🔥):

- **Claude 走向国际舞台**：一位用户表示 *Claude* 是文学相关任务的首选，但遗憾的是在其国家无法使用。
- **Prompt Engineering 与 API 查询**：针对询问如何使用个人数据训练模型以及使用 API 的成员，建议阅读文档，并提供了 [OpenAI 帮助文章](https://help.openai.com/en/articles/7039783-how-can-i-access-the-chatgpt-api) 的链接，同时提醒 *ChatGPT API 是按需付费的*。
- **神秘的账号封禁**：一位成员因违反使用条款收到了意外的终止信并寻求建议，回复建议只有支持团队（Support）能解决此类问题，并确保没有违反任何条款。
- **探索模型切换**：关于 GPT 动态模式的讨论（例如为了成本效益在 3.5 和 4 等版本之间自动切换），提到目前没有官方公告，但指向了 A/B 测试或泄露信息。此处分享了一个关于该话题的 Twitter 对话链接 [here](https://twitter.com/AndrewCurran_/status/1779999302292779225)。
- **Llama 3 的出现**：Meta 的 Llama 3 作为 OpenAI 模型的竞争对手引起了轰动，提到了其潜在优势和性能推测，以及在各个平台上的可用性。分享了一个 [YouTube 访谈](https://youtu.be/bc6uFV9CJGg)，详细介绍了 Llama 3、其在行业中的地位以及大规模模型的考量。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/bc6uFV9CJGg">Mark Zuckerberg - Llama 3, $10B Models, Caesar Augustus, &amp; 1 GW Datacenters</a>：扎克伯格谈论：- Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - Caesar Augustus、智能爆炸、生物风险...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1230144218731642971)** (12 messages🔥): 

- **寻求自定义 GPT 知识库的训练资源**：一位成员询问了为自定义 GPT 模型准备知识库的良好训练来源。随后的消息中没有提供具体的建议或资源。
- **对 Whisper v3 API 的期待**：一位成员表达了对通过 OpenAI API 发布 **Whisper v3** 的渴望，并指出距离 Whisper v3 最初发布已经快一年了。
- **GPT-4 记忆力担忧**：有人观察到 **GPT-4** 之前已知的 30k+ Token 模型变得更加健忘，暗示 Token 容量可能有所减少，但未提供进一步证据。
- **GPT-4 版本延迟问题**：一位成员提到 **GPT-4-0125-preview** 在过去两天里变慢了，导致其对延迟敏感的应用出现困难。未分享基准测试或对比测试。
- **针对 GPT-4 响应缓慢的替代方案**：针对报告的延迟问题，建议尝试 **gpt-4-turbo-2024-04-09**；然而，该成员反馈此模型感觉比之前使用的版本更慢。另一位成员分享了他们的经验，称 **Claude** 在 10 条消息后性能往往会下降，但未提供与 GPT-4 的详细对比。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1230153203652493312)** (38 messages🔥): 

- **寂静的“鬼城”**：一位用户评论该频道活跃度较低，将其与之前繁忙的状态进行了对比，并将这种下降归因于审核政策。
- **工具链难题**：多位用户讨论了他们在 **GPT-4** 中遇到的问题，复杂的 Workflow 似乎更难编排，导致人们越来越强调使用 Python 编写脚本来管理 Prompt 的稳定性。
- **Prompt Engineering 咨询与建议**：一位正在开发基于 PDF 规则的文本重写助手的成员寻求建议，被建议避免使用 PDF，因为其不可靠，应改用纯文本或 **JSON 或 XML** 等结构化格式。
- **对 API 最佳实践的好奇**：有一场关于使用 OpenAI API 进行会议转录分析项目的最新 Prompt Engineering 最佳实践的对话；尽管有官方的 [OpenAI 文档](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)，但这些实践的社区驱动性质仍被强调。
- **寻求 AI 与 Blockchain 方面的合作**：一位新用户介绍自己是 Blockchain 开发者，希望将 AI 与 Blockchain 技术融合，并表示有兴趣与他人合作开发该项目的 Prompt。
  

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1230153203652493312)** (38 messages🔥): 

- **Discord 社区冷清化**：一位用户注意到 API-discussions 频道的活跃度显著下降，将目前的冷清与去年热烈的讨论氛围进行了对比。
- **版主策略的后果还是改进？**：这种冷清可能归因于管理策略。一名成员表示严格的审核导致了参与度不足，而另一名成员则提到过去的禁言（timeout）可能是讨论减少的原因。
- **GPT 模型评价两极分化**：对新 GPT 模型的看法不一，有人认为基础推理能力有所提升，而另一些人则批评与更新模型配套的工具质量下降，导致复杂工作流更难执行。
- **避免使用 PDF 喂给 AI**：在关于向 AI 提供文本重写规则的讨论中，用户建议不要使用 PDF，因为其 Embedding 不一致且存在元数据问题，推荐改用纯文本或 Markdown。
- **寻求 AI-区块链项目合作**：一位区块链开发者表示有兴趣将 AI 与区块链技术结合，并邀请他人合作以改进 Prompt 库。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1230130829758566430)** (58 messages🔥🔥): 

- **分享 Flan 微调模型**：一位用户在 Hugging Face 上提供了 **Pile-T5 Base** 和 **Pile-T5-XXL** 的 Flan 微调模型链接，分享了他们的模型集和单个模型，例如 [lintang/pile-t5-base-flan](https://huggingface.co/lintang/pile-t5-base-flan) 以及其 [用户主页](https://huggingface.co/lintang) 上的其他模型。

- **训练数据格式与预处理讨论**：对话涉及了为 LLM 训练准备数据的方法，讨论了文档截断的挑战，以及通过语料库分块（corpus block sizing）和拼接（concatenation）来避免 Padding Token 的技术，并探讨了这可能如何影响模型对 Token 分布的熟悉度。

- **拆解 Prepacking 策略**：一位用户分享了关于 *Prepacking* 的推文和论文，这是一种提高 Transformer 模型 Prefilled Prompt 速度和内存效率的方法。该技术引发了与 Sequence Packing 等已知策略的对比，并讨论了它代表了有意义的新进展，还是对先前讨论方法的重新发现。

- **Mistral 模型与 Attention 机制探讨**：讨论了 Sequence Packing 等训练技术的效率和实用性，以及 *nantion fa* 等新方法。用户还就当前 Attention 机制的替代方案交换了意见，这些方案可能更有效地处理长序列。

- **优化变长输入 Attention 的对话**：社区成员交流了处理模型变长输入的更好方法，包括利用能够动态适应相关历史信息而无需扫描整个上下文的 Attention 方法。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/siyan_zhao/status/1780288750624612850?s=46">Siyan Zhao (@siyan_zhao) 的推文</a>: 🚨LLM 研究者们🚨想要在生成质量零退化的前提下，为你的 HuggingFace🤗 LLM 免费提升速度和内存效率吗？介绍 Prepacking，一种简单的方法，可获得高达 6 倍的速...</li><li><a href="https://arxiv.org/abs/2404.10830">Fewer Truncations Improve Language Modeling</a>: 在 LLM 训练中，输入文档通常被拼接在一起，然后分割成等长的序列以避免 Padding Token。尽管这种拼接方式效率很高，但...</li><li><a href="https://www.sarvam.ai/blog/announcing-openhathi-series">OpenHathi 系列：一种低成本构建双语 LLM 的方法</a>: 暂无描述</li><li><a href="https://huggingface.co/lintang/pile-t5-base-flan">lintang/pile-t5-base-flan · Hugging Face</a>: 暂无描述</li><li><a href="https://huggingface.co/lintang">lintang (Lintang Sutawika)</a>: 暂无描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1230167757547372635)** (120 messages🔥🔥):

- **重温 Chinchilla 的数学模型**：一篇[分析 Scaling Laws](https://arxiv.org/abs/2404.10102) 的论文（由 Hoffmann 等人提出）使用他们的第三种方法重新推导了估算值，并对原始发现提出了质疑，认为其置信区间过于狭窄，以至于需要不切实际的实验数量。
- **多语言 LLM 与 Tokenizer 大小**：关于在大型语言模型（LLM）中使用更大 Tokenizer 词表优势的讨论强调了一些社区共识，即更大的 Tokenizer 是有益的，特别是对于多语言应用，尽管可能仍需要元分析或稳健的 A/B 测试。
- **扩展 LLM 中的解耦嵌入 (Untied Embeddings)**：关于在扩展模型规模时，解耦嵌入是否能提高性能的辩论，普遍认为不同的 LLM 并没有统一的方法，绑定（Tying）或解耦嵌入是一个深思熟虑的选择。
- **不同 Tokenizer 之间的困惑度 (Perplexity)**：为了比较使用不同 Tokenizer 的模型困惑度，建议使用 bits per byte (BPB)，即汇总文档 Token 的损失并除以 Tokenization 之前的文档原始大小。
- **利用 MCTS 提升 LLM 的推理能力**：腾讯 AI 实验室推出了 [AlphaLLM](https://arxiv.org/abs/2404.12253)，它将蒙特卡洛树搜索（MCTS）与 LLM 集成，提出了一种自我改进循环，以增强模型在复杂推理和规划任务中的能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.10102">Chinchilla Scaling: A replication attempt</a>：Hoffmann 等人 (2022) 提出了三种估算计算优化 Scaling Law 的方法。我们尝试复现他们的第三种估算程序，该程序涉及拟合参数化损失函数...</li><li><a href="https://huggingface.co/datasets/rajpurkar/squad_v2/discussions/9">rajpurkar/squad_v2 · Error in train split, question containing 25651 characters!</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.08846">An Embarrassingly Simple Approach for LLM with Strong ASR Capacity</a>：在本文中，我们专注于解决语音处理领域最重要的任务之一，即自动语音识别 (ASR)，利用语音基础编码器和大型语言模型...</li><li><a href="https://arxiv.org/abs/2404.12253">Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing</a>：尽管大型语言模型 (LLM) 在各种任务上具有令人印象深刻的能力，但它们在涉及复杂推理和规划的场景中仍然面临困难。最近的工作提出了先进的...</li><li><a href="https://arxiv.org/abs/2404.10282">Tripod: Three Complementary Inductive Biases for Disentangled Representation Learning</a>：归纳偏置对于解耦表示学习中缩小未指定解集至关重要。在这项工作中，我们考虑为神经网络自动编码器赋予三种选择性的...</li><li><a href="https://arxiv.org/abs/2404.12318">Reuse Your Rewards: Reward Model Transfer for Zero-Shot Cross-Lingual Alignment</a>：基于人类标注的偏好数据对语言模型 (LM) 进行对齐是获得实用且高性能的基于 LM 系统的重要步骤。然而，多语言人类偏好数据难以...</li><li><a href="http://arxiv.org/abs/2404.10179">Scaling Instructable Agents Across Many Simulated Worlds</a>：构建能够在任何 3D 环境中遵循任意语言指令的具身 AI 系统是创建通用 AI 的关键挑战。实现这一目标需要学习将语言...</li><li><a href="https://arxiv.org/abs/2404.09894v2">Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection</a>：随着大型语言模型 (LLM) 在各个领域的应用不断扩大，全面调查其不可预见的行为及随之而来的结果变得至关重要。在这项研究中...</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/NVlabs/DoRA">GitHub - NVlabs/DoRA: Official PyTorch implementation of DoRA: Weight-Decomposed Low-Rank Adaptation</a>：DoRA 的官方 PyTorch 实现：权重分解低秩自适应 - NVlabs/DoRA
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1230185118723735592)** (32 messages🔥): 

- **机器学习 Scaling 新手寻求 FLOPs 估算**：一位成员请求关于从 SoundStream 论文中估算训练 FLOPs 的建议，并被引导计算前向和后向传播中每个 Token 的操作数，然后乘以数据集大小，参考了一篇 [Transformer 论文](https://arxiv.org/abs/2001.08361) 第 2.1 节中的计算示例。

- **请不要秘密进行 Epoch**：有人指出，如果秘密进行多次 Epoch 且不报告，就无法确定训练模型的真实计算成本。

- **Chinchilla 的 Scaling 策略受到审视**：频道讨论了一条关于 Scaling 策略的 Twitter 帖子，并提出了关于每个参数“最佳平均” Chinchilla Token 数量的问题，澄清指出原始发现并没有发生实质性变化。

- **ML 论文中的统计失误？**：有人批评了一篇论文，认为正确的统计方法会使研究结果与现有证据更趋一致，从而引发了关于 ML 研究中不同方法选择的可靠性和影响的辩论。

- **Chinchilla 的估计：低估了数据 Scaling 的影响？**：对话涉及了数据 Scaling 的重要性，以及 Chinchilla 是否可能低估了其效果，并提醒了 ML 论文中统计方法的挑战，以及 Google 内部实验中对 Chinchilla Scaling 模型的使用。
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1230130302446338169)** (9 messages🔥): 

- **准确率指标的澄清**：在关于 **lambada_openai** 的讨论中，明确了准确率指标应验证模型的贪婪输出（greedy output）是否与单个 **target string** 匹配。此检查针对的是计算准确率，不是针对整个句子，而是专门针对生成的续写（continuation）。

- **探索 MMLU 与 ARC 的联系**：成员们对以 **ARC** 风格呈现 **MMLU** 基准测试任务的结果表示好奇。多位用户表示有兴趣尝试调整 Prompt 模板，例如移除 MMLU 中的多选题选项。

- **Loglikelihood 计算细节**：对话强调，在计算 **loglikelihood** 时，应只关注 **continuation/target**，而不是整个句子。这确保了 **perplexity** 是专门针对生成的续写（在 Prompt 条件下）计算的。

- **使用 vLLM 获得显著速度提升**：一名成员报告称，与常规的 text-generate pipeline 相比，使用 **vLLM** 获得了 **10倍的速度提升**。对话暗示这可能是由于其设置优于标准的 **Hugging Face pipeline**。

- **PR 评审支持请求**：分享了包括 **flores-200** 和 **sib-200** 基准测试在内的贡献，旨在增强多语言评估（[sib-200 的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1705) & [flores-200 的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1706)）。对话强调需要更精简的方法来有效地评审和合并这些包含大量配置的任务。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1705">Implement Sib200 evaluation benchmark - text classification in 200 languages  by snova-zoltanc · Pull Request #1705 · EleutherAI/lm-evaluation-harness</a>：我们使用了来自 MALA 论文 https://arxiv.org/pdf/2401.13303.pdf 的 Prompt 风格，我们也发现该风格在我们的 SambaLingo 论文 https://arxiv.org/abs/2404.05829 中取得了不错的结果。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1706">Implementing Flores 200 translation evaluation benchmark across 200 languages by snova-zoltanc · Pull Request #1706 · EleutherAI/lm-evaluation-harness</a>：我们使用了该论文中发现效果最好的 Prompt 模板 https://arxiv.org/pdf/2304.04675.pdf。我们的论文也发现该 Prompt 模板效果不错 https://arxiv.o...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1230213153930084403)** (16 messages🔥):

- **Mojo 遇见 IDE**：PyCharm 发布了一个新的 [Mojo 插件](https://plugins.jetbrains.com/plugin/23371-mojo)，虽然目前功能尚不完善，但 PyCharm 团队表示有兴趣增强对 Mojo 的支持。
- **Windows 用户使用 Mojo**：针对 Windows 用户询问是否需要 Ubuntu/WSL 才能使用新的 Mojo 插件，回复称 JetBrains 确实集成了 [WSL](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)，但目前尚不清楚其是否能与 Mojo 插件完美配合。
- **Mojo Playground 功能建议**：社区讨论了在 Mojo 在线 Playground 中增加使用 `nightly` 构建版本的选项，旨在帮助内存（RAM）较低的用户。该建议详见此 [GitHub 讨论](https://github.com/modularml/mojo/discussions/2321)。
- **在 Mojo 中集成 C**：对于有兴趣在 Mojo 中使用 C 的用户，建议参考 [GitHub 上的 mojo-ffi 项目](https://github.com/ihnorton/mojo-ffi)，并查看[使用 `external_call`](https://twitter.com/Modular/status/1779913837216719118) 调用 libc 函数的教程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs Plugin | Marketplace</a>：为 Mojo 编程语言提供基础编辑功能：语法检查与高亮、注释和格式化。未来将添加更多新功能...</li><li><a href="https://github.com/modularml/mojo/discussions/2321">允许在在线 Playground 使用 `nightly` · modularml/mojo · Discussion #2321</a>：我没有足够的 RAM 来运行虚拟机或 Docker 并同时打开文档，因此依赖在线 Playground 运行 Mojo 代码。如果有一个小的下拉菜单允许我切换到 nightly 版本...</li><li><a href="https://github.com/ihnorton/mojo-ffi">GitHub - ihnorton/mojo-ffi</a>：参与 GitHub 上 ihnorton/mojo-ffi 的开发。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1230242770472669304)** (2 条消息): 

- **Modular 在 Twitter 上发布公告**：Modular 分享了一条推文，社区提供了直接查看链接：[查看推文](https://twitter.com/Modular/status/1780676643176231240)。
- **Modular 的另一条推文**：Modular 随后发布的另一条推文也引起了社区关注：[查看推文](https://twitter.com/Modular/status/1781000544158650716)。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1230589476498178198)** (2 条消息): 

- **Meta 发布 LLaMa 3**：分享了一个名为 "Meta Releases LLaMa 3: Deep Dive & Demo" 的 YouTube 视频，讨论了 **Meta LLaMa 3** 的发布。视频涵盖了该新迭代模型的各项特性。
- **ModularBot 庆祝用户成就**：**ModularBot** 发布了一条祝贺信息，认可一位用户在社区中晋升至 **level 1**。

**提到的链接**：<a href="https://www.youtube.com/watch?v=E3_0nHpfbcY">Meta Releases LLaMA 3: Deep Dive &amp; Demo</a>：今天，2024 年 4 月 18 日，是一个特别的日子！在本视频中，我将介绍 @meta 的 LLaMA 3 发布。该模型是其第三次迭代...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1230163710526623746)** (171 条消息🔥🔥): 

- **Variant List 的难题**：讨论集中在创建包含 Variant 的 List 的困难上，因为 `Variant` 似乎缺乏所需的 **Movable** trait。虽然有人建议了一些变通方法，如使用自定义 struct 或 tuple，但这些被认为是不够理想或“蹩脚”的方案。

- **循环引用挑战**：对话还涉及了 Mojo 中的循环引用问题，并探讨了其他语言的处理方式，例如 Nim 的 cycle collector。有人询问了静态确定循环引用的可行性，但也对是否需要垃圾回收（GC）等运行时解决方案表示了担忧。

- **Stringable Tuples 的麻烦**：目前直接打印 tuple 存在问题，因为它们没有实现 **Stringable** trait。虽然提供了一个使用自定义函数的潜在变通方案，但该问题突显了 tuple 在 trait 遵循（conformance）方面的局限性。

- **Mojo 中的打印与 List 管理**：一位成员在为 String 类型实现 list append 时寻求帮助，并遇到了 Python 抓取的 Mojo 代码因类型和打印输出问题无法编译的情况。会议澄清了 List 必须初始化，且由于 List 类型不是直接 **Stringable** 的，其元素需要分别打印。

- **Mojo 中的语法和语义特性**：对话中探讨了通过 traits 扩展行为的语法，以及类型提升（type promotion）的细微差别，包括尝试协调编译时（compile-time）和运行时（runtime）类型行为。此外，还有一条关于类型注解和自动提升（autopromotion）复杂性的趣评，认为这可能会增加代码编写和可读性的难度。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/no-more-go-t-sam-mercy-stop-gif-4954655">No More GIF - No More Go T Sam - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/dimitrilw/roguelike-mojo/blob/main/src/main.mojo#L51>">roguelike-mojo/src/main.mojo at main · dimitrilw/roguelike-mojo</a>：使用 Mojo🔥 逐步完成 Python Rogue-like 教程。 - dimitrilw/roguelike-mojo</li><li><a href="https://github.com/dimitrilw/roguelike-mojo/blob/main/src/main.mojo#L57-L63>">roguelike-mojo/src/main.mojo at main · dimitrilw/roguelike-mojo</a>：使用 Mojo🔥 逐步完成 Python Rogue-like 教程。 - dimitrilw/roguelike-mojo</li><li><a href="https://tenor.com/search/"">&quot; GIFs | Tenor</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1230130126990082139)** (6 条消息): 

- **Mojo 24.2 的编译困惑**：一位用户报告了在使用 **Mojo 24.2** 编译项目时遇到困难，并在 Mojo Playground 上遇到了编译错误，推测这可能是由于使用了 nightly 特性导致的。
- **运行在未来 Mojo 版本上的 Django 移植版？**：有人开玩笑说某个 Django 移植版运行在尚不存在的 Mojo 版本上，这可能是指某些正在进行的项目的激进（bleeding-edge）性质。
- **法国工程系学生寻求 Mojo 指导**：一位新用户（法国工程系学生）表示有兴趣在 Mojo 中实现 **numba** 版本的 *canny 边缘识别算法* 以进行性能对比，并正在寻求相关文档或示例来协助完成这一尝试。
- **Mojo 新手获得社区支持**：针对这位法国学生的请求，一位资深用户提供了 Mojo 文档链接，并引导其访问 [Get Started with Mojo](https://docs.modular.com/mojo/manual/get-started/) 页面，将 Mojo 描述为连接**研究与生产**的桥梁，结合了 **Python 语法** 与系统编程及元编程特性。
- **用户感谢社区协助**：寻求 canny 边缘算法帮助的学生热情地感谢了社区成员提供的指导和资源。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>：获取 Mojo SDK 或尝试在 Mojo Playground 中编写代码。</li><li><a href="https://docs.modular.com/mojo/">Mojo🔥 | Modular Docs</a>：一种弥合 AI 研究与生产之间鸿沟的编程语言，兼顾速度与易用性。</li><li><a href="https://docs.modular.com/mojo/notebooks/">Mojo🔥 notebooks | Modular Docs</a>：我们为 Mojo Playground 创建的所有 Jupyter notebooks。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 30 期
https://www.modular.com/newsletters/modverse-weekly-30
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1230231689897447424)** (1 条消息): 

由于只提供了一条消息，缺乏上下文或进一步的讨论点，目前没有足够的信息来创建详细摘要。如果未来从 🏎engine 频道提供更多消息，我可以按要求的格式提供全面的摘要。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1230138944176263189)** (14 条消息🔥): 

- **Traits 功能实现**：消息提到 **traits** 现在可以正常工作，逐步淘汰 `AnyRegType` 的工作正在取得进展。

- **Nightly 清理工作**：一位成员强调，**最新的 nightly 版本** 侧重于大量的清理工作。

- **Mojo 格式规范确定**：聊天中确认，默认的 **Mojo format** 已更改为 80 列。

- **关于 `UnsafePointer` 命名的讨论**：讨论围绕将类型命名为 `UnsafePointer` 的决定展开，引用了 [Mojo 团队的回答](https://mojodojo.dev/mojo-team-answers.html#unsafe-code) 以及 [This Week in Mojo](/this_week_in_mojo/) 中的每周更新位置。

- **Nightly/Mojo 安装问题**: 几位成员报告了在尝试 **更新 Nightly/Mojo** 时遇到的问题，引用的错误信息包括文件缺失和无法识别的归档格式。补救建议包括运行 `modular clean` 以及通过 `brew upgrade modular` 升级 `modular`。

**提到的链接**: <a href="https://mojodojo.dev/mojo-team-answers.html#unsafe-code">Mojo Team Answers | Mojo Dojo</a>: 未找到描述

  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1230180652163272714)** (6 条消息): 

- **MCTS PPO 值得进一步探索**: *Monte Carlo Tree Search (MCTS) 结合 Proximal Policy Optimization (PPO)* 被强调为一个尚未被充分探索的领域，可能在未来的研究中大有可为。

- **解析 MCTS**: 在关于 **MCTS** 的提问后，它被解释为 *Monte Carlo Tree Search*，这是一种常用于博弈类 AI 的决策算法。

- **分享创新研究**: Nathan Lambert 分享了他参与合作的一篇 [arxiv 论文](https://arxiv.org/abs/2309.15028)，介绍了一种*名为 PPO-MCTS 的新型价值引导解码算法*，该算法将 PPO 的价值网络与 MCTS 集成，用于推理阶段的自然语言生成。

**提到的链接**: <a href="https://arxiv.org/abs/2309.15028">Don&#39;t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding</a>: 推理阶段搜索算法（如 Monte-Carlo Tree Search (MCTS)）在基于最先进的强化学习（如 Proximal Pol...）生成自然语言文本时可能显得多余。

  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1230153917103345785)** (121 条消息🔥🔥): 

- **Mixtral 8x22B 树立新标准**: 新发布的 **Mixtral 8x22B** 语言模型以其多语言流利度以及卓越的数学和编程能力而闻名。其庞大的 64K tokens 上下文窗口确保了从大型文档中精准召回信息，该模型以 Apache 2.0 许可证发布，可供广泛使用。
  
- **OLMo 1.7 7B 实现性能飞跃**: [最新的 OLMo 1.7 7B 模型](https://huggingface.co/allenai/OLMo-1.7-7B) 令人印象深刻，得益于数据和训练过程的改进，其 MMLU 基准测试分数显著提升了 24 分。OLMo 系列旨在推动语言模型科学的发展，所有相关的训练材料均已公开。
  
- **对 Mixtral Instruct 模型的期待升温**: 围绕 **Mixtral-8x22B-Instruct-v0.1** 模型的讨论集中在其在聊天机器人应用中的潜力。Instruct 系列的利用和改进在模型的 Hugging Face [卡片](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) 中有详细说明。

- **Meta Llama 3 加入对话**: Meta 发布的 **Meta Llama 3** 大语言模型引发了关注，其微调版本适用于对话应用，推理 API 可通过 Azure AI Studio 访问。公告暗示即将推出 700 亿参数的模型，进一步助长了社区讨论。

- **Replicate 宣布高性价比模型基础设施**: Replicate 详细列出了其计费结构，展示了使用各种 GPU 模型进行 AI 处理的实惠成本。此举可能会降低尝试 AI 模型的准入门槛。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>：继续推动 AI 前沿，让所有人都能使用。</li><li><a href="https://x.com/nahrzf/status/1781011649580712342?s=46">nahr (@nahrzf) 的推文</a>：Meta 做了最有趣的事</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/_philschmid/status/1780641241668997258">Philipp Schmid (@_philschmid) 的推文</a>：修复了为 @AI21Labs 做的修复，并包含了 Mambas。🐍 ↘️ 引用 Armand Joulin (@armandjoulin) 的话：修复了那个修复。</li><li><a href="https://commoncrawl.org/web-graphs">Common Crawl - Web Graphs</a>：详细介绍了 Common Crawl 的网页图谱发布、背后的技术以及如何使用它们。</li><li><a href="https://replicate.com/docs/billing">Replicate 的计费方式</a>：Replicate 的计费方式</li><li><a href="https://www.interconnects.ai/p/llama-3-and-scaling-open-llms">Llama 3：将开源 LLM 扩展至 AGI</a>：Llama 3 表明，在不久的将来，Scaling 不会成为开源 LLM 进步的限制。</li><li><a href="https://fxtwitter.com/AlbertQJiang/status/1780648008696091003">Albert Jiang (@AlbertQJiang) 的推文</a>：我热爱开源模型！请将你最喜欢的模型添加到 Mistral Convex Hull。 ↘️ 引用 Philipp Schmid (@_philschmid) 的话：修复了为 @AI21Labs 做的修复，并包含了 Mambas。🐍</li><li><a href="https://azuremarketplace.microsoft.com/en-US/marketplace/apps/metagenai.meta-llama-3-8b-chat-offer?tab=overview">Microsoft Azure Marketplace</a>：未找到描述</li><li><a href="https://huggingface.co/allenai/OLMo-1.7-7B">allenai/OLMo-1.7-7B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/DrJimFan/status/1781006672452038756">Jim Fan (@DrJimFan) 的推文</a>：即将推出的 Llama-3-400B+ 将成为一个分水岭时刻，社区将获得对 GPT-4 级模型的权重开放访问（open-weight access）。它将改变许多研究工作和草根初创公司的考量……</li><li><a href="https://blog.allenai.org/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d">OLMo 1.7–7B：在 MMLU 上提升了 24 分</a>：今天，我们发布了 70 亿参数开源语言模型 OLMo 1.7–7B 的更新版本。该模型在 MMLU 上得分为 52，位列……</li><li><a href="https://www.youtube.com/watch?v=gqtmUHhaplo">OpenAssistant 已完成</a>：#OpenAssistantLAION 的 OpenEmpathic：https://laion.ai/blog/open-empathic/ 链接：主页：https://ykilcher.com 衍生品：https://ykilcher.com/merch YouTube：https:/...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1230219724961353888)** (11 条消息🔥): 

- **引发 Chinchilla Scaling 争议**：一位成员分享了 [@tamaybes](https://x.com/tamaybes/status/1780639257389904013?s=46) 的推文，质疑 Hoffmann 等人发表的 Chinchilla Scaling 论文研究结果的可复现性，指出这项具有影响力的工作中存在**差异**。
- **@suchenzang 对 Scaling Laws 表示怀疑**：在成员引用的一条推文中，[@suchenzang](https://x.com/suchenzang/status/1616752482226671620?s=46) 挑战了通常通过将数据拟合到单条线而进行的推断，特别是针对 Chinchilla 论文背后的数学逻辑。
- **作者的沉默引发挫败感**：聊天参与者表示愤怒，因为 [@tamaybes](https://x.com/tamaybes/status/1780639279506432473?s=46) 声称尝试联系 Chinchilla 论文作者的请求未得到回应。
- **@drjwrae 对分析发表看法**：聊天成员引入了 [@drjwrae](https://x.com/drjwrae/status/1780824132692901915?s=46) 的一条推文，对 Chinchilla 分析发表了看法，认为新发现实际上可能再次证实了 Scaling Laws 的存在。
- **承认错误并宣布开源数据**：[@borgeaud_s](https://x.com/borgeaud_s/status/1780988694163321250) 承认了 Chinchilla 论文中与错误设置 loss scale 相关的失误，并表示作者打算开源数据以提高透明度。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/tamaybes/status/1780639257389904013?s=46">来自 Tamay Besiroglu (@tamaybes) 的推文</a>：Hoffmann 等人发表的 Chinchilla scaling 论文在语言建模社区极具影响力。我们尝试复现其工作的核心部分，并发现了差异。这里是...</li><li><a href="https://x.com/suchenzang/status/1616752482226671620?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：在忽略了所有这些“让我们把一团点拟合成一条线”的论文中的细节后（当你真正外推时，这些可能都是错的），@stephenroller 终于说服了我去深入研究...</li><li><a href="https://x.com/drjwrae/status/1780824132692901915?s=46">来自 Jack Rae (@drjwrae) 的推文</a>：很好的分析。我认为这解释了为什么方法 3 与方法 1 和 2 不匹配。此外，我看到人们在分享这篇论文，并暗示这证明了 scaling laws 并不存在。我对他们发现的看法是...</li><li><a href="https://x.com/tamaybes/status/1780639279506432473?s=46">来自 Tamay Besiroglu (@tamaybes) 的推文</a>：我们已向作者寻求帮助，但尚未得到回复。(8/9)</li><li><a href="https://x.com/borgeaud_s/status/1780988694163321250">来自 Sebastian Borgeaud (@borgeaud_s) 的推文</a>：伟大的分析，方法 3 终于达成一致了！我们论文中的 loss scale 太低，导致 L-BFGS 过早终止，从而导致拟合效果不佳。修复此问题后，我们可以复现 ...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1230179557873553582)** (8 条消息🔥): 

- **最烂排行榜冠军**：一位成员幽默地声称获得了“最烂排行榜冠军”的头衔。

- **支持吃瓜**：一位 Discord 成员热心分享了一个 [Twitter 帖子](https://twitter.com/chrmanning/status/1780753750254375200)，该帖子似乎引发了一场与 Machine Learning 相关的争议。

- **捍卫诚信，反对虚假信息**：用户 @420gunna 强调了 [Jesse L. Yu 的推文](https://x.com/jessechenglyu/status/1780765024350560341)，Yu 在推文中回应了一段被断章取义的误导性视频片段，并强调了真实性的重要性。

- **赞赏构建，尽管尚早**：一位成员对在该领域积极构建的个人表示赞赏，但指出他们的努力可能还为时过早。

- **收购或消亡**：最后一条消息简明扼要地描述了该行业许多项目面临的残酷现实：要么被收购，要么不复存在。

**提到的链接**：<a href="https://x.com/jessechenglyu/status/1780765024350560341">来自 Jesse Lyu (@jessechenglyu) 的推文</a>：这条帖子有 24.09 万次观看，所以我决定直接回应。这是一个传播误导性信息的绝佳范例和大师级案例。你从我原本 44 分钟的视频中截取了 30 秒...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1230174866322362388)** (22 条消息🔥): 

- **AI 模型之战**：一位成员幽默地询问了 *OLMO* 与 *LLaMa 3* 之间的对决，Nathan 回应道：“哈哈，别艾特我，这是一场注定失败的战斗”，随后确认 **“tmmw”**（明天），可能是在预告即将发布的内容。
- **高产内容爆发**：Nathan 暗示将有大量产出，表示接下来可能是 **“一周三篇博客”**，这可能预示着即将出现大量新帖子或见解。
- **混乱中的视觉美学**：一段对话确认删除了“恼人的白线”，并对特定的头像表示不满，这表明正在讨论与 [chaotic era](https://chaotic.era.link) 相关的视觉元素。
- **文学与梗的融合**：讨论涉及《三体》（*Three-Body Problem*）是否出现在播客中，并调侃“3BodyProblem = 3BP = 3BlogPost”，引用“神圣数字学”来戏称内容模式。
- **探索实验空间**：一位成员分享了 [Jeremy Howard 的一条推文链接](https://x.com/jeremyphoward/status/1780816986777559246?s=46)，询问有关“实验性”事物的情况，引发了小组内的好奇和潜在讨论。

**提到的链接**：<a href="https://x.com/jeremyphoward/status/1780816986777559246?s=46">来自 Jeremy Howard (@jeremyphoward) 的推文</a>：这个“实验性”的小玩意儿是什么？是新出的吗？好用吗？

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1230181805106401361)** (3 条消息):

- **AI 接管 SNL**：Nathan Lambert 在一段 [SNL 短剧](https://www.youtube.com/watch?v=86qKgK0asGo)中发现了笑点，该短剧描述了一个 NewsNation AI 直播被现场观众幽默地打断，他评价道：“*这真的太棒了，笑死我了*”。
- **开头一分钟奠定基调**：他指出视频的前一分钟特别能引起共鸣，可能反映了直播 AI 活动中那些古怪的现实。

**提到的链接**：<a href="https://www.youtube.com/watch?v=86qKgK0asGo">Beavis and Butt-Head - SNL</a>：NewsNation 的一场关于 AI 的直播活动被两名观众（Ryan Gosling, Mikey Day）搅乱。Saturday Night Live。现在可在 Peacock 观看：https://pck.tv/...

---

**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1230288746960916501)** (17 条消息🔥): 

- **Nathan 遭遇讲座准备疲劳**：Nathan Lambert 表达了对讲座准备工作的疲惫，但预见即将完成，并幽默地发了句 *almostdone lol*。
- **预测最后一刻会增加模型**：Phil Pax 预计 Nathan 在最后一刻需要在他的图表中加入另外六个模型，这一推测伴随着一个关于最近 Llama 3 发布的玩笑。
- **动漫曲目提升动力**：420gunna 分享了一个名为 "NEVER GIVE UP YOUR WAAAAAAAAAAAAY" 的 YouTube 视频，其中包括来自动漫《斩服少女》（Kill La Kill）的一首器乐曲。
- **对即将到来的演讲中加入 Llama 3 兴趣不大**：Nathan Lambert 提到他并不特别想在演讲中重点介绍 Llama 3，但为了让听众满意，他还是妥协增加了一张幻灯片。
- **LLaMA-Guard 引发好奇**：420gunna 提到听说过 **LLaMA-Guard** 并对其内容表示好奇，想知道它是否只是一个毒性分类器（toxicity classifier），并询问了此类模型的潜在基准测试（benchmarks）。

**提到的链接**：<a href="https://youtu.be/tYzMYcUty6s?si=zubH4CRGQ4CdMWb5">NEVER GIVE UP YOUR WAAAAAAAAAAAAY</a>：NEVA GIVE UP - https://bit.ly/2VrgAcK。歌曲是来自动漫《斩服少女》（Kill La Kill）的 Before my Body is Dry 器乐版。考虑在我们的 Patreon 捐赠！https://w...

---

**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1230363620769267773)** (4 条消息): 

- **速度并非 SnailBot 的长处**：一位成员对 SnailBot 的性能发表了评论，幽默地指出其速度特别慢，这可能会影响它的实用性。
- **SnailBot 达成里程碑**：同一位成员随后认可了 SnailBot 的贡献，虽然是开玩笑的语气，因为它在一篇进行中（WIP）的文章中正常工作了。

---

**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1230131411810258964)** (166 条消息🔥🔥): 

- **通过 Web UI 微调模型**：通过 [Web UI](http://dashboard.cohere.com/fine-tuning) 进行模型微调（Fine-Tuning）是一个简单的过程，但随后使用新数据对模型进行微调似乎需要使用 API。官方文档提供了在验证训练数据后启动微调任务的逐步指南。
- **发布 Command R+**：Cohere 推出了他们最新的模型 **Command R+**，该模型因其先进的能力而备受推崇。详细信息和功能对比可以在 [Cohere 官网](https://txt.cohere.com/compressed-embeddings-command-r-plus/)找到。
- **Command R 使用场景的许可咨询**：一位成员就 Command R 在潜在灰色地带的许可问题寻求建议，表示可能会私信某人了解详情以求澄清。
- **Cohere 优雅的品牌形象和网页设计**：用户对 Cohere 的品牌形象和网站设计表示赞赏，称其具有简洁的美感。同时也提到了在 Firefox 上访问某些页面（如 [Cohere 定价页](https://cohere.com/pricing)）时的资源消耗问题。
- **Llama 模型评价与 Prompt**：随着 Llama 3（包括 70b 和 400b 模型）的发布，用户讨论了其令人印象深刻的性能，并分享了涉及复杂 Prompt 的个人测试策略，如数学和 SVG 标记语言执行。此外，还讨论了将现实场景实用性作为大型语言模型（LLM）基准测试的意义。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arc.net/l/quote/artdceqi">引用自 “Chat”</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">检索增强生成 (RAG) - Cohere Docs</a>: 未找到描述</li><li><a href="https://hf.co/chat/assistant/661e50f73af5cdaed7435ef8.">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://docs.cohere.com/docs/fine-tuning-with-the-web-ui">使用 Web UI 进行微调 - Cohere Docs</a>: 未找到描述</li><li><a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 &amp; 二进制 Embeddings - 将您的向量数据库扩展到大型数据集</a>: Cohere Embed 现在原生支持 int8 和二进制 embeddings，以降低内存成本。</li><li><a href="https://docs.cohere.com/docs/tool-use">使用 Cohere 模型的工具调用 - Cohere Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1230156502287781908)** (6 条消息): 

- **探索 AI 能力的伦理**：一位用户对 **Command R+** 模型进行了红队测试（redteaming），发现可能被用于不道德活动（如搜索负面信息、勒索和骚扰）的漏洞，并引用了 [LessWrong 上的帖子](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r)。
- **关于模型责任的辩论**：另一位用户批评了这种红队测试方法，将其类比为不公平地指责公司宣传其产品的能力（例如汽车的速度或厨刀的锋利度），并参考了发布前 **GPT-4** 示例的表现。
- **AI 漏洞研究意图的澄清**：原帖作者做出回应，澄清其意图并非攻击 **Cohere**，而是强调 AI 中越狱（jailbreaks）日益增长的重要性，因为它们会导致 **Large Language Models** (LLMs) 产生具有潜在严重后果的 Agent 行为。
- **越狱是一件严肃的事情**：作者继续指出，越狱的性质已经演变——从导致模型使用不当语言转变为使其能够执行复杂的 Agent 任务——此类越狱对于任何将 AI 用于敏感操作的组织来说都可能是一个重大问题。
- **技术细节披露**：作者解释了他们越狱模型的方法，使用了一个结合模型输出和工具输出的循环，根据 Cohere 的指令来强化 Agent 行为。

**提到的链接**：<a href="https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r">在 LessWrong 上使用 Command R+ 创建不受限制的 AI Agents</a>：TL;DR 目前存在能力强大的开源权重模型，可用于创建简单且不受限制的恶意 Agents。它们可以端到端地执行任务……

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1230142001903833119)** (124 条消息🔥🔥): 

- **聊天库与创业机会**：一位成员幽默地建议，可以建立一个创业公司来提供良好的聊天库体验，潜力甚至可能超过 OpenAI 等公司。
- **本地小型模型可能带来变革**：有人指出，可以在本地运行且具有精美用户界面的更小、更快、更便宜的 AI 模型可能比大型模型更具吸引力，即使大型模型能解决更复杂的问题。这是因为在许多情况下，用户只需要基础任务的协助。
- **AI 应用中的延迟至关重要**：讨论了延迟对 AI 驱动的应用和硬件（如 Humane pin）的影响。快速、高效的响应对于获得积极的用户评价和广泛采用至关重要。
- **AI 性能的突然变化令用户困惑**：成员们报告称，AWS 托管的 Claude 3 在涉及临床概念提取的任务中性能明显下降，幻觉（hallucinations）增多，准确率从 95% 以上降至接近零。
- **Llama 3 的期待**：围绕 Meta Llama 3 即将发布的兴奋情绪和分析正在蔓延，提到了 8B 和 70B 的规格、通过合作伙伴加强生态系统，以及社区对训练规模和应用可能性的关注。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/russelljkaplan/status/1513128005828165634">来自 Russell Kaplan (@russelljkaplan) 的推文</a>：大语言模型（LLM）崛起的二阶效应：</li><li><a href="https://mistral.ai/news/mixtral-8x22b/">更便宜、更好、更快、更强</a>：继续推动 AI 的前沿，并让所有人都能使用。</li><li><a href="https://discord.gg/NBFgzps4">加入 Systems Engineering Professionals Discord 服务器！</a>：查看 Discord 上的 Systems Engineering Professionals 社区——与 1664 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://x.com/openaidevs/status/1780640119890047475?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：介绍 Assistants API 的一系列更新 🧵 借助新的文件搜索工具，你可以快速集成知识检索，现在每个助手最多允许 10,000 个文件。它适用于我们的...</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信领域的领导者，拥有简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://x.com/fofrai/status/1780617084315349222?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 fofr (@fofrAI) 的推文</a>：每张 SD3 图像消耗 6.5 个积分。10 美元 = 1,000 积分。所以每张图像是 0.065 美元。或者是通过相同 API 生成 SDXL 成本的 10 倍。 ↘️ 引用 Stability AI (@StabilityAI) 今天，我们很高兴 ...</li><li><a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://www.youtube.co">未找到标题</a>：未找到描述</li><li><a href="https://x.com/awnihannun/status/1781020285107675502?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Awni Hannun (@awnihannun) 的推文</a>：@soumithchintala @lvdmaaten 🤣🤣🤣 刚刚运行了 @Prince_Canuma 量化的 8B 版本。在 M2 Ultra 上表现非常好（而且很快 😉）：</li><li><a href="https://x.com/amir/status/1780414705221382517?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Amir Efrati (@amir) 的推文</a>：最新消息：Mistral 在获得 20 亿美元融资仅几个月后，正寻求 50 亿美元的估值。LLM 竞赛前端的融资行动依然势头不减。 https://www.theinformation.com/articles/mistral-an-openai-rival...</li><li><a href="https://vickiboykis.com/2024/02/28/gguf-the-long-way-around/">GGUF，绕远路</a>：什么是 ML artifact？</li><li><a href="https://x.com/mattshumer_/status/1780437606540472730?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Matt Shumer (@mattshumer_) 的推文</a>：Claude 3 Opus 变得糟糕多了... :(</li><li><a href="https://x.com/OpenAIDevs/status/1780640119890047475">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：介绍 Assistants API 的一系列更新 🧵 借助新的文件搜索工具，你可以快速集成知识检索，现在每个助手最多允许 10,000 个文件。它适用于我们的...</li><li><a href="https://x.com/FanaHOVA/status/1780996533661671683">来自 Alessio Fanelli (@FanaHOVA) 的推文</a>：🦙 所有 Llama 3 发布详情和亮点：8B 和 70B 尺寸：Instruct 和预训练版本在大多数基准测试中均达到 SOTA 性能。目前正在训练一个 400B+ 参数的模型，稍后将发布...</li><li><a href="https://x.com/togethercompute/status/1781004579817349266?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Together AI (@togethercompute) 的推文</a>：我们很高兴能成为 Meta Llama 3 的发布合作伙伴。现在即可体验 Llama 3，Llama 3 8B 每秒高达 350 个 token，Llama 3 70B 每秒高达 150 个 token，以全 FP16 精度运行...</li><li><a href="https://x.com/karpathy/status/1781047292486914189?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：模型卡片中也有一些更有趣的信息：https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md 注意 Llama 3 8B 实际上处于 Llama 2 70B 的水平，这取决于...</li><li><a href="https://x.com/mistralailabs/status/1780606904273702932?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mistral AI Labs (@MistralAILabs) 的推文</a>：在发布 Mixtral 8x22B 的同时，我们还发布了我们的 tokenizer，它超越了通常的文本 <-> token 转换，增加了对工具和结构化对话的解析。仓库：https://github.com/mis...</li><li><a href="https://www.grey-wing.com/product/ocean-oracle">掌握租船数据的副驾驶</a>：利用生成式 AI 做出更好的决策，并赋能你的租船团队做出更好的决策。</li><li><a href="https://strongcompute.com/research-grants">研究资助</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=bc6uFV9CJGg&ab_channel=DwarkeshPatel">马克·扎克伯格 - Llama 3、100 亿美元模型、凯撒·奥古斯都和 1 GW

Datacenters</a>: Zuck 谈 Llama 3 - 迈向 AGI 的开源 - 定制芯片、合成数据以及扩展时的能源限制 - 凯撒·奥古斯都、智能爆炸、生物...</li><li><a href="https://x.com/armandjoulin/status/1780638511818838378">Armand Joulin (@armandjoulin) 的推文</a>: 修复了那个修复。 ↘️ 引用 Jonathan Frankle (@jefrankle) 为你修复了它，@code_star</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=RMbhEuN-7dQ">从单张图像进行 3D 重建！Few-Shot Neural Radiance Fields | NeRF #12</a>: 使用元学习从单张图像进行新视角合成、3D 重建和神经辐射场（Neural Radiance Fields）。论文 &quot;Learned Initializat...&quot; 的实现。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1230231139893903394)** (19 messages🔥): 

- **Paper Club West 开始**: 成员们以问候开启了 **llm-paper-club-west** 会议，并确认了屏幕共享等功能已正常运行。
- **转移到 Zoom**: 几条消息提供了 [Zoom 会议链接](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)，引导成员将会议从 Discord 转移到 Zoom。发布了多次提醒以确保所有参会成员都收到了信息。
- **因承诺克服对 Zoom 的抵触**: 尽管表达了对 Zoom 的厌恶，一位成员还是加入了会议，保持了社区参与度。
- **Zoom 接入协调**: 一位成员进行实时协调，确保等待加入 Zoom 会议的人员获得准入。

**提到的链接**: <a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">加入我们的 Cloud HD 视频会议</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1230174214560944140)** (49 messages🔥): 

- **Windows 上的 OpenInterpreter 使用困扰**: 用户在 Windows 上设置 OpenInterpreter 时遇到困难，问题范围从软件过度偏向 Mac 到系统执行 OS 控制任务时的挑战。
- **对 OpenInterpreter 潜力的乐观态度**: 尽管设置过程受挫，用户仍对 OpenInterpreter 的能力充满热情，例如编写和运行用于文本转摩尔斯电码的 Arduino 代码，并在 Raspberry Pi 搭配 Ubuntu 等不同设置上展示了进展。
- **日本的投影仪机器人手机引起关注**: 一位用户展示了来自日本的[带投影仪的机器人手机](https://twitter.com/CashTHLo/status/1780518436713762889)，思考其 3D 扫描能力与 GPT-4-vision 协同的潜力，并寻求该想法的合作。
- **为 OpenInterpreter 探索本地 LLM**: 出现了关于使用本地语言模型处理 OpenInterpreter 任务的查询，以及关于使用 **Ollama 3** 等模型进行本地 OS 模式使用的能力和设置的讨论。
- **利用强大硬件挑战 OI 极限**: 一位用户提到购买了四块 Tesla P40 用于 OpenInterpreter，寻求关于尝试哪些模型以增强性能的建议，并对潜在能力表示兴奋。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio),">简介 - Open Interpreter</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WKmRXZgsErej2xUriKzxrEAXdxMSgWbb?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://github.com/botgram/shell-bot">GitHub - botgram/shell-bot: :robot: 执行命令并发送实时输出的 Telegram 机器人</a>: :robot: 执行命令并发送实时输出的 Telegram 机器人 - botgram/shell-bot
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1230142684530868344)** (85 messages🔥🔥):

- **Windows 的烦恼与 Poetry 的思考**：成员们遇到了 **PowerShell** 无法识别 `OPENAI_API_KEY` 的问题，并且在 `set` 命令的使用上结果不一。*createai.* 询问了 `poetry install` 的典型持续时间，而其他人则讨论了在 Windows 上运行 01 时可能遇到的特定环境挑战。
- **ESP32 的困境**：包括 *rbrisita* 在内的多位成员表达了在连接 **ESP32** 设备时遇到的挑战，尝试了不同的 IDE 甚至求助于 `curl` 命令。虽然取得了一些进展，但仍有关于 messages 数组的错误报告。
- **本地服务器设置尝试**：*azlade* 详细说明了使用 `curl` 手动设置 01 服务器地址的过程，并遇到了音频缓冲和语言模型利用不当的问题，导致转录效果不佳。
- **WebSocket 处理**：多位成员报告了与 **websockets** 以及客户端-服务器连接相关的问题。有人提到可能与 Python 版本有关，并建议使用诊断工具。
- **多设备连接查询**：用户询问并探索了在 Windows 上使用 **LM Studio** 配合 Macbook 上的 01 的可能性，以及 01 与不同操作系统的兼容性。*rouw3n* 分享了为了更好的兼容性而转向使用 MacBook 的经历。

**提到链接**：<a href="http://SERVER_IP_GOES_HERE:10001"`">未找到标题</a>：未找到描述

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

kieguin: https://huggingface.co/spaces/ysharma/Chat_with_Meta_llama3_8b
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1230205325081186387)** (4 条消息): 

- **"来自 MistralAI 的尖端模型"**：MistralAI 的新 **8x22b** 模型被标记为开源模型中的最先进水平，并且 LlamaIndex 从发布第一天起就提供了支持。Mistral cookbook 展示了该模型的能力，包括 RAG、查询路由和工具使用，详见分享的 [Twitter 帖子](https://twitter.com/llama_index/status/1780646484712788085)。

- **使用 Elasticsearch 构建免费 RAG**：来自 Elastic 的一篇 [博客文章](https://twitter.com/llama_index/status/1781022740339920967) 包含了一个使用 Elasticsearch 和 LlamaIndex 构建 **Retrieval Augmented Generation (RAG)** 应用程序的教程，使用了来自 LlamaIndex 和 MistralAI 的完全开源且免费的组件。

- **Meta Llama 3 模型的 Cookbook**：LlamaIndex 宣布对 Meta 新的 **Llama 3 模型** 提供零日支持，由 @ravithejads 和 @LoganMarkewich 编写的 [cookbook](https://twitter.com/llama_index/status/1781039161325293981) 演示了其与 Hugging Face 的集成，用于简单提示和完整的 RAG 流水线。

- **轻松在本地运行 Llama 3**：LlamaIndex 分享了一个使用 @ollama 提供的命令在本地运行 **Llama 3** 模型的快速指南。该帖子包含一个建议的 [notebook](https://twitter.com/llama_index/status/1781040257565364661) 链接，其中包含将 "llama2" 替换为 "llama3" 以供本地使用的说明。
<div class="linksMentioned">

<strong>提到链接</strong>：

<ul>
<li>
<a href="https://t.co/QqLdz5lojV">RAG (Retrieval Augmented Generation) with LlamaIndex, Elasticsearch and Mistral — Elastic Search Labs</a>：学习使用 LlamaIndex、Elasticsearch 和本地运行的 Mistral 实现 RAG 系统。</li><li><a href="https://t.co/jjtpFOzNOS">Ollama - Llama 2 7B - LlamaIndex</a>：未找到描述</li><li><a href="https://t.co/WWbYp5lqXe">MistralAI Cookbook - LlamaIndex</a>：未找到描述</li><li><a href="https://t.co/RMB7MhXIOA">Llama3 Cookbook - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1230134675809374258)** (76 条消息🔥🔥): 

- **讨论 RAG、多语言支持和资源**：成员们正在询问使用 RAG 实现高效检索系统的最佳实践，并寻求关于多语言场景的建议。提供了一个微调嵌入的链接：[Fine-tune embedding](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=embeddings+fine)。
  
- **RAG 技术中的摘要**：关于摘要是否属于 RAG 技术的一部分及其在增强搜索中的效用的问题得到了肯定的回答，并提供了进一步阅读的资源：[Q&A Summarization](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/?h=summar#summarization) 和 [Doc Summary Example](https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary/?h=summary)。

- **使用 DBRX 和 LlamaIndex 抽象构建 Agent**：讨论了 LlamaIndex 抽象是否足够通用，能像支持 OpenAI 一样支持 DBRX，并建议通过覆盖设置来使用自定义 LLM。

- **使用 LlamaParse 实现 GoogleDriveReader**：一位成员建议尝试使用 LlamaParse 和 GoogleDriveReader 的 `download_file` 方法以获得更好的效果，并分享了相关文档：
[Google Drive base.py](https://github.com/run-llama/llama_index/blob/ac6ab9f6bb5826c95f34945c1d5d15f7b47b0d54/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/drive/base.py#L329)。

- **在 QueryPipeline 中访问中间输出**：一位成员询问如何返回 `QueryPipeline` 中某些模块的输出，解决方案涉及引用 `intermediates` 的 `outputs` 字典。参考了中间输出的文档：[Intermediate Outputs Guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern/?h=intermediate#intermediate-outputs)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai-demos.dev/">AI Demos</a>：未找到描述</li><li><a href="https://www.secinsights.ai/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/ac6ab9f6bb5826c95f34945c1d5d15f7b47b0d54/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/drive/base.py#L329">llama_index/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/drive/base.py</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern/?h=intermediate#intermediate-outputs">Usage Pattern - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern/?h=intermediate#i">Usage Pattern - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=embeddings+fine">Finetune Embeddings - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/?h=summar#summarization">Q&A patterns - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary/?h=summary">Document Summary Index - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/custom_prompt_synthesizer/?h=summa">Pydantic Tree Summarize - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1230287562405445662)** (5 条消息): 

- **Speedlegal 登上 Product Hunt**：一位成员宣布他们的初创公司 **Speedlegal** 在 Product Hunt 上线，并寻求社区的支持和反馈。他们分享了产品页面的直接链接：[Speedlegal on Product Hunt](https://www.producthunt.com/posts/speedlegal)。

- **Google 为 LLM 提供的无限上下文技巧**：分享了一篇来自 VentureBeat 的文章，讨论了 **Google 的新技术，该技术可能赋予大语言模型 (LLMs) 无限的上下文**。这项技术可能是 LLM 的重大进步，详情见：[Google's Infinite Context for LLMs](https://venturebeat.com/ai/googles-new-technique-gives-llms-infinite-context)。

- **检索增强生成 (RAG) 的终结？**：一位成员质疑 Google 提供无限上下文的进步是否预示着 **检索增强生成 (RAG)** 模型的终结。

- **触手可及的 AI 融资数据**：**manhattanproject2023** 自去年以来策划了一个全面的 AI 相关融资数据集，现在可供查阅。该数据集涵盖了约 540 家公司的 550 轮融资，总额达 300 亿美元，可通过 [AI Hype Train - Airtable](https://www.frontieroptic.com/ai-hype-train) 访问。

- **寻找缺失的初创公司数据**：在分享融资数据后，另一位成员询问数据集中是否包含 **LlamaIndex** 和 **Zep** 等早期初创公司。

**提到的链接**：<a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - 你的个人 AI 合同谈判专家 | Product Hunt</a>：SpeedLegal 是一款 AI 工具，可帮助你更好地理解和谈判合同。它可以快速识别潜在风险，并用简单的语言解释复杂的法律条款。SpeedLegal 还能……

  

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1230155475459244193)** (40 messages🔥): 

- **SQL Agent 聊天机器人挑战讨论**：一位成员分享了在使用 LangChain 的 SQL Agent 构建聊天机器人时的复杂经验，强调了由于需要全面的 Prompt engineering，该 Agent 存在局限性。该成员参考了 [LangChain SQL Agents](https://js.langchain.com/docs/use_cases/sql/agents) 的 LangChain 文档，其中详细介绍了 `createOpenAIToolsAgent` 和 `SqlToolkit` 的使用。

- **RunnableWithMessageHistory 成为关注焦点**：提供了关于使用 `RunnableWithMessageHistory` 的深入技术指导，包括代码片段以及对 LangChain 代码库和单元测试的引用。成员们澄清了涉及聊天消息历史检索和管理的用法。

- **宣布推出 Flashcardfy**：一位成员宣传了 Flashcardfy，这是一项从各种媒体生成抽认卡的新服务，并收到了关于在不提示升级的情况下难以浏览网站的反馈。该服务提供个性化的抽认卡，目标客户是顶尖学生，信息可在 [Flashcardfy](https://flashcardfy.lol) 查看。

- **分享对多智能体编排（Multi-Agent Orchestration）的兴趣**：成员强调了 Microsoft 的 AutoGen 框架是构建多智能体对话系统的潜在工具，并询问了其他人的使用经验。Microsoft AutoGen 的详细信息见 [AutoGen Framework](https://microsoft.github.io/autogen/)。

- **整理 AI 相关融资数据**：一位个人透露了他们收集的 AI 初创公司融资数据，550 轮融资总计约 300 亿美元，并在 [AI Hype Train](https://www.frontieroptic.com/ai-hype-train) 提供了详细的 Airtable。他们邀请大家对数据的准确性提供反馈。

- **探索针对私有文档的 RAG 实现**：一位用户将 LangChain 的 Ingestion 和 Retrieval 链整合到一个 Retrieval Augmented Generation (RAG) 项目中，分享了 GitHub 仓库，并指向了一个 YouTube 播放列表以获取更多背景信息：[YouTube RAG From Scratch Playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)。GitHub 仓库位于 [aosan/VaultChat](https://github.com/aosan/VaultChat)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF">">未找到标题</a>：未找到描述</li><li><a href="https://microsoft.github.io/autogen/">AutoGen | AutoGen</a>：通过多智能体对话框架赋能下一代 LLM 应用</li><li><a href="https://flashcardfy.lol">Flashcardfy - 带有个性化反馈的 AI 抽认卡生成器</a>：使用提供个性化反馈的 AI 生成抽认卡，学习得更快、更聪明。</li><li><a href="https://js.langchain.com/docs/use_cases/sql/agents">Agents | 🦜️🔗 Langchain</a>：LangChain 提供了许多工具和函数，允许你创建 SQL Agents，从而提供一种更灵活的与 SQL 数据库交互的方式。使用 SQL Agents 的主要优点是...</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/tool_usage#conversational-responses>)">Tool usage | 🦜️🔗 Langchain</a>：本节将介绍如何创建对话式 Agents：可以使用工具与其他系统和 API 交互的聊天机器人。</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/message_history#in-memory>)">Add message history (memory) | 🦜️🔗 LangChain</a>：RunnableWithMessageHistory 让我们能够为某些内容添加消息历史</li><li><a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x">RAG From Scratch</a>：Retrieval augmented generation (或 RAG) 是一种将 LLM 与外部数据源连接的通用方法。本视频系列将建立起对...的理解。</li><li><a href="https://github.com/aosan/VaultChat">GitHub - aosan/VaultChat: 从你的私有文档中获取知识</a>：从你的私有文档中获取知识。通过在 GitHub 上创建一个账户来为 aosan/VaultChat 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1230636046220267691)** (1 messages): 

- **求知欲强**：一位成员请求提供关于如何在客户端使用 JavaScript 通过 **LangServe 添加反馈**的教程。然而，在现有的消息中没有提供进一步的讨论或资源。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1230240214707077150)** (5 messages):

- **Launching the AI Plugin Marketplace**: The Co-Founder of [theaiplugs.com](http://theaiplugs.com/) introduced a new marketplace for AI Plugins, Tools, and Assistants for users of different technical skills. It simplifies the process by handling front-end, API credit management, billing, and marketing challenges.
  
- **SpeedLegal Hunts for Support on Product Hunt**: A startup has been launched on Product Hunt and the creator is seeking support and feedback for their initiative. Interested parties can engage and provide input via [Product Hunt](https://www.producthunt.com/posts/speedlegal).

- **Prompt Engineering Course Now on LinkedIn**: A new course focused on prompt engineering with LangChain is now available on LinkedIn Learning. The course can be accessed through [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7186761950138109952/).

- **Llama 3 Available for Public Use**: Llama 3 has been hosted and is open for anyone interested in trying it out. It's accessible through two links: for Chat ([https://chat.tune.app/](https://chat.tune.app/)) and for API ([https://studio.tune.app/](https://studio.tune.app/)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - Your personal AI contract negotiator | Product Hunt</a>: SpeedLegal is an AI tool that helps you understand and negotiate contracts better. It can quickly identify potential risks and explain complicated legal terms in simple language. SpeedLegal also gives...</li><li><a href="https://chat.tune.app/">Tune Chat - Chat app powered by open-source LLMS</a>: With Tune Chat, access Prompts library, Chat with PDF, and Brand Voice features to enhance your content writing and analysis and maintain a consistent tone across all your creations.</li><li><a href="https://studio.tune.app/">no title found</a>: no description found</li><li><a href="https://medium.com/ai-advances/unlocking-efficiency-the-power-of-multi-step-tools-with-langchain-and-cohere-7d1ea571ebed">Unlocking Efficiency: The Power of Multi-Step Tools with Langchain and Cohere</a>: Ankush k Singal
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1230290412598726697)** (3 messages): 

- **Inappropriate Content Alert**: The message contained a link presumably leading to inappropriate content involving underage individuals and was not related to any AI or ML discussion. It was framed as an advertisement for leaked media on platforms such as OnlyFans.

**Link mentioned**: <a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1230290438632640543)** (3 messages): 

- **Inappropriate Content Alert**: The channel included a message promoting adult content with a link to a Discord server. The message mentioned "Hot Teen & Onlyfans Leaks" accompanied by suggestive emojis.

**Link mentioned**: <a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1230290443665801226)** (3 messages): 

- **Inappropriate Content Alert**: The message contains a link that is suggested to lead to **leaked content** of an explicit nature, involving potentially underage subjects. The content is promoted in a manner that implies unauthorized sharing of private images.

**Link mentioned**: <a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1230290451060494417)** (3 messages): 

- **Inappropriate Content Alert**: A message was posted promoting *"Hot Teen & Onlyfans Leaks"* with a Discord invite link. The content was flagged as underage and explicit.

**Link mentioned**: <a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1230290453471953007)** (3 messages): 



- **不当内容警告**：一位用户发布了一条推广 **Hot Teen & Onlyfans Leaks** 的消息，包含成人内容和 Discord 邀请链接。该消息标记了不当表情符号并提到了 **@everyone**。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。

  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1230540260845293590)** (6 条消息): 

- **WizardLM-2 变得更易获取**：**WizardLM-2** 已重新上传，具有最新的 state-of-the-art 进展，现在所有人均可在 [Hugging Face](https://huggingface.co/alpindale/WizardLM-2-8x22B) 获取。欲了解更多信息，可以参考 [WizardLM-2 Release Blog](https://wizardlm.github.io/WizardLM2)，并在其 [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2)、[Twitter](https://twitter.com/WizardLM_AI) 以及 [arXiv](https://arxiv.org/abs/2304.12244) 上的学术论文中找到更多资源。

- **渴望获取 Meta Llama 3 的 Tokenizer**：一位成员表达了对 **Meta Llama 3-8B** 模型 Tokenizer 的需求，并指向其在 [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 上的页面，这需要根据 [Meta Privacy Policy](https://www.facebook.com/privacy/policy/) 同意分享联系信息。

- **成功获取并分享访问权限**：在之前的 Meta Llama 3 Tokenizer 请求之后，另一位成员成功获得了访问权限，并提到了用户 Undi95 在 [Hugging Face](https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main) 上重新上传的模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Undi95/Meta-Llama-3-8B-hf/tree/main">Undi95/Meta-Llama-3-8B-hf at main</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1230290466516500582)** (3 条消息): 

- **不当内容警报**：一位 Discord 用户发布了一条包含成人内容链接的消息。该消息包含通常与显式内容相关的表情符号以及一个 Discord 邀请链接。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。

  

---


**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1230290484036108328)** (4 条消息): 

- **垃圾信息警报**：存在包含疑似垃圾信息链接的消息，推广成人内容。
- **呼吁管理**：一位成员强调了对管理的需求，建议可能存在需要封禁的事件。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。

  

---


**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1230290488574349383)** (3 条消息): 

- **不当内容警报**：发布了一条推广 **Hot Teen & Onlyfans Leaks** 的消息，包含一个 [Discord 服务器邀请链接](https://discord.gg/rj9aAQVQFX)。该消息似乎是垃圾信息，不符合社区标准。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。

  

---


**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1230290500666396672)** (3 条消息): 

- **不当内容警告**：发布了一条包含指向潜在不当内容链接的消息，特别是指向一个据称与 **teen leaks and Onlyfans** 相关的 Discord 服务器。该消息包含暗示未成年内容的表情符号，并通过 **@everyone** 提及来寻求关注。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行沟通的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。

---

**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1230290511529639966)** (3 条消息): 

- **垃圾信息警报报告**：**looking-for-work** 频道遭到了一条垃圾消息的轰炸，该消息推广了一个与“Hot Teen & Onlyfans Leaks”相关的不当 Discord 链接。该消息包含一个加入 Discord 服务器的邀请链接。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行沟通的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。

---

**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1230290519960059965)** (3 条消息): 

- **不当内容警报**：有人发布了一条带有链接的消息，暗示分享**未成年及露骨内容**。链接 [https://discord.gg/rj9aAQVQFX](https://discord.gg/rj9aAQVQFX) 似乎是一个 Discord 服务器邀请，可能用于有害且非法的内容分发。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行沟通的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。

---

**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1230290534044799006)** (3 条消息): 

- **不当内容警报**：有人发布了一条消息，其中的链接据称指向一个与**成人内容**相关的 Discord 服务器。该链接以“Hot Teen & Onlyfans Leaks”为名进行广告宣传，并包含一个 Discord 邀请 URL。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行沟通的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。

---

**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1230290537911812177)** (3 条消息): 

- **不当内容警报**：在 **qa** 频道中，有人发布了一条包含链接的消息，暗示分发涉及潜在未成年人的露骨内容。该消息提到了“Hot Teen & Onlyfans Leaks”，并尝试通过 **@everyone** 引起注意。

**提到的链接**：<a href="https://discord.gg/rj9aAQVQFX">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行沟通的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。

---

**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1230179344241000540)** (19 条消息🔥): 

- **了解 8x22B 模型内存消耗**：[一位成员讨论了](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12) 8x22B 模型的 VRAM 需求，指出在混合精度下使用 **Adam** 优化器进行训练大约需要 **3673 GB**。这些数据源自 [Hugging Face 上的 Model Memory Utility Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)。
- **大规模多 GPU 训练**：该成员分享了他们在 **64 个拥有 80GB VRAM 的 NVIDIA GPU** 上运行 32k 序列长度模型的经验，并确认在尝试使用 32 个 GPU 时遇到了显存溢出（out-of-memory）错误。
- **探索内存效率**：为了优化内存使用，该成员暗示正在尝试将 **8-bit 优化**作为潜在的解决方案。
- **新训练完成**：**Mixtral-8x22B-v0.1 模型**的全量规模化有监督微调（Supervised Tuning）已完成，并在 [Hugging Face](https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de) 上共享，该模型涉及英语和德语指令的数据集混合。
- **FSDP 配置挑战**：一位成员在尝试使用 `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock` 时遇到了 **shape errors**，并怀疑尽管计算是以混合精度进行的，但参数状态可能仍为 float32，导致即使在多个高内存 GPU 上也会出现内存问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de">maxidl/Mixtral-8x22B-v0.1-Instruct-sft-en-de · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12">mistral-community/Mixtral-8x22B-v0.1 · [AUTOMATED] Model Memory Requirements</a>: 未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1230173597360848927)** (15 messages🔥): 

- **Mistral Tokenization 开源**: Mistral 发布了其 Tokenization 库，希望它能被所有推理库采用，旨在为各种模型的微调（finetuning）建立标准化格式。该库支持 Tool Calls 和结构化输出，并提供了一个 [Jupyter Notebook 示例](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb)。

- **Meta 发布 Llama 3**: Meta 宣布了全新的 *Llama 3* 模型，该模型将在包括 AWS、Google Cloud 和 Microsoft Azure 在内的多个平台上可用。由于采用了拥有 128K Token 词汇表的 Tokenizer，该模型承诺将提升多语言性能。公告详情和预期的生态系统支持可以在 [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3/) 上找到。

- **Llama 3 的多语言支持受到质疑**: 尽管 Llama 3 的 Tokenizer 效率很高，但其多语言性能仍需谨慎对待。虽然涵盖了 30 多种语言，但在非英语语言中的表现可能不如英语。预训练数据集中包含超过 5% 的高质量非英语数据，但对性能的预期仍有所保留。

- **Tokenizer 吸引开发者关注**: 在 Llama 3 发布后，社区成员正在讨论新 Tokenizer 的可用性和访问权限，有报告称通过 Hugging Face 几乎可以立即获得访问权限。此外，人们还关注该 Tokenizer 是否可以针对特定语言（如捷克语）进行缩减，以实现更快的推理。

- **Llama 3 输出限制引发担忧**: Llama 3 的发布引发了社区对模型输出下游使用限制的反馈，一些人在 Twitter 上表达了失望，并主张减少限制以支持开源开发。该推文强调了人们一直以来对 MistralAI 等提供较少限制的替代方案的偏好。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb">mistral-common/examples/tokenizer.ipynb at main · mistralai/mistral-common</a>: 通过在 GitHub 上创建账户来为 mistralai/mistral-common 的开发做出贡献。</li><li><a href="https://ai.meta.com/blog/meta-llama-3/">no title found</a>: 未找到描述</li><li><a href="https://fxtwitter.com/xlr8harder/status/1780992684062024138?s=19">来自 xlr8harder (@xlr8harder) 的推文</a>: Llama 3 发布了，但似乎仍然只有 @MistralAI 真正支持我们：它仍然对模型输出有下游使用限制。@AIatMeta 这是一个损害开源的垃圾限制...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/)** (1 messages): 

bjoernp: 👀
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1230320419219832852)** (5 messages): 

- **社区成员在 Product Hunt 上发布初创项目**: 一位成员宣布在 [Product Hunt](https://www.producthunt.com/posts/speedlegal) 上发布了他们的初创公司，并寻求社区的支持和反馈。
- **Karpathy 关于小模型潜力的推文**: Andrej Karpathy 的一条推文指出，在大型数据集（15T Token）上训练的小型模型（8B 参数）可能与大型模型一样有效。他指出，对小型模型进行深度训练是一种虽然少见但值得欢迎的方法，并暗示在当前的实践中，小型模型可能训练不足，这一点在[这条推文](https://twitter.com/karpathy/status/1781028605709234613)中得到了强调。
- **拥抱小型模型浪潮**: 社区参与信号表明，人们对经过长期训练的小型模型持积极态度，因为它们易于使用且效率高，并引用了上述 Andrej Karpathy 关于 Llama 3 的推文。

**提到的链接**: <a href="https://www.producthunt.com/posts/speedlegal"> SpeedLegal - 您的个人 AI 合同谈判专家 | Product Hunt</a>: SpeedLegal 是一款 AI 工具，可帮助您更好地理解和谈判合同。它可以快速识别潜在风险，并用简单的语言解释复杂的法律条款。SpeedLegal 还能...

  

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1230226260366721116)** (8 messages🔥): 

- **对 Mixtral Instruct 的期待升温**：一位成员表达了通过 llm 尝试 **Mixtral 8x22B Instruct** 模型的极大热情，并分享了 [Hugging Face 上的模型卡](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)，其中详细介绍了如何使用各种代码片段运行该模型。

- **llm-gpt4all 中的 Bug 警报**：有报告称新安装的 llm-gpt4all 存在问题，一位成员链接到了 [GitHub issue #28](https://github.com/simonw/llm-gpt4all/issues/28)，该 issue 记录了在添加 llm-gpt4all 模型后 Python 应用崩溃的问题。

- **llm 插件开发中的困扰**：一位成员在尝试创建和使用新的 llm 插件时遇到了 `ModuleNotFoundError`，这表明如果开发过程中出现问题，插件可能会失效。

- **插件故障导致 llm 无法运行**：同一位成员随后报告称，安装一个开发中的插件导致其主 llm 安装损坏，甚至尝试卸载该插件也会出现相同的错误提示，这反映了插件管理方面的困难。

- **针对插件问题的全新安装解决方案**：在花费大量时间调试该问题后，该成员最终选择完全卸载并重新安装 llm。有人建议，通过 brew 和 pipx 存在的多个 llm 安装可能是导致问题的潜在原因。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/simonw/llm-gpt4all/issues/28">adding the llm-gpt4all models breaks the python app. · Issue #28 · simonw/llm-gpt4all</a>：我顺利安装了 llm，分配了 OpenAI 密钥，并且可以毫无问题地与 gpt4 对话，查看我的 llm models 命令输出：OpenAI Chat: gpt-3.5-turbo (aliases: 3.5, chatgpt) OpenAI...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1230197371812581406)** (5 messages): 

- **Pytorch-Lightning 的硬件无关性**：关于 Pytorch-Lightning 的讨论揭示了其硬件无关的能力，根据分享的 GitHub 链接确认，它允许用户在无需更改代码的情况下，在多个 GPU、TPU 上[预训练、微调和部署 AI 模型](https://github.com/Lightning-AI/pytorch-lightning)。
- **现实环境中的 GPU 兼容性**：一位成员确认在 AMD Radeon 7900XTX GPU 上成功使用了 Pytorch-Lightning，表明其在不同硬件上的实际应用能力。
- **在 AMD 上使用 ROCm 的性能**：在 7900XTX GPU 上进行测试时，利用 ROCm 的 Pytorch-Lightning 在某些模型上的表现比原生 PyTorch 稍快。
- **LLaMa3 模型发布公告**：新 AI 模型 LLaMa3 已经发布，链接指向[官方网页](https://llama.meta.com/llama3/)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llama.meta.com/llama3/">Meta Llama 3</a>：使用 Meta Llama 3 构建 AI 的未来。现在提供 8B 和 70B 的预训练及指令微调版本，以支持广泛的应用。</li><li><a href="https://github.com/Lightning-AI/pytorch-lightning">GitHub - Lightning-AI/pytorch-lightning: Pretrain, finetune and deploy AI models on multiple GPUs, TPUs with zero code changes.</a>：在无需更改代码的情况下，在多个 GPU、TPU 上预训练、微调和部署 AI 模型。 - Lightning-AI/pytorch-lightning
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1230146991770701886)** (2 messages): 

- **零成本张量操作探索**：一位成员寻求关于如何在 **tinygrad** 中实现 **broadcast, reshape, permute** 等操作而不产生数据复制成本的建议。他们了解 shape 和 strides，但需要关于如何操作它们以在 permute 和 broadcast 方面进行准确计算的指导。
- **掌握 Shape 和 View 的路径**：另一位成员做出了回应，指出 **tinygrad/shape/shapetracker.py** 或 **view.py** 是可能包含所需零成本张量操作信息的资源。
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1230343416131616939)** (2 messages): 

- **热情入场**：一位成员以高能量的问候进入聊天，全大写输入“HELLLLOOOOOOOO!!!!!!!!”。
- **随意的回应**：另一位成员以一句时髦的肯定词“litty”进行了随意的回应。
  

---

**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1230352528915431455)** (1 条消息): 

- **寻求无代码微调平台**：一位成员表示使用无代码平台微调 **GPT-3.5** 非常方便，并询问是否存在类似的针对开源模型的平台，能够支持 **无代码微调（no-code fine-tuning）和 Serverless 推理**。他们还询问了创建此类平台所涉及的障碍。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1230135625915695115)** (3 条消息): 

- **Snowflake 发布突破性文本嵌入模型**：Snowflake 宣布发布 Snowflake Arctic embed 系列模型，被誉为世界上最好的实用文本嵌入模型。该系列模型在 Apache 2.0 许可证下开源，此 [YouTube 视频](https://www.youtube.com/watch?v=p9T7ZgtM5Mo) 提供了详细概述。

- **Mixtral 设定 AI 新基准**：一段关于 Mixtral 8x22B 的视频介绍将其描述为 Mistral 推出的最佳开源模型，展示了 AI 性能和效率的新标准。视频强调了稀疏混合专家模型（sparse Mixture-of-Experts models）的进展，可以在[此处](https://www.youtube.com/watch?v=N8U6XnVK2mM)观看。

- **Meta 发布 Llama 3 开源 LLM**：Meta 推出了 Llama 3，这是他们最新的开源大语言模型，有望挑战 AI 能力的极限。更多信息和 Llama 3 的详细介绍可在该[视频](https://www.youtube.com/watch?v=zQy11WnAIIc)中找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=N8U6XnVK2mM">Mixtral 8x22B Mistral 最佳开源模型</a>：Mixtral 8x22B 是最新的开源模型。它为 AI 社区的性能和效率设定了新标准。它是一个稀疏混合专家模型（SMo...</li><li><a href="https://www.youtube.com/watch?v=zQy11WnAIIc">介绍 Llama 3 最佳开源大语言模型</a>：介绍 Meta Llama 3，Facebook 下一代最先进的开源大语言模型。https://ai.meta.com/blog/meta-llama-3/#python #...</li><li><a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake 发布世界上最好的实用文本嵌入模型</a>：今天 Snowflake 发布并以 Apache 2.0 许可证开源了 Snowflake Arctic embed 系列模型。基于 Massive Text Embedding Be...
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1230131674290065490)** (4 条消息): 

- **Llamafile 脚本清理**：一位成员清理了 llamafile 归档版本升级重打包脚本，并通过 [Gist 链接](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e) 进行了分享。他们正在考虑将其添加到 **llamafile GitHub repo**，但指出维护者应该从头开始生成新的 llamafile，而不是进行重打包。
  
- **报告漏洞的流程**：一位成员询问了报告安全漏洞和申请 CVE 的程序。随后通过私信提供了更详细的信息。
  
- **关于 LLM API 暴露的通用警告**：询问漏洞报告的同一位成员建议不要公开暴露 LLM API 端点，因为之前发现了相关 Bug，并强调这并不是他们在 LLM 基础设施代码中发现的第一批问题。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/)** (1 条消息): 

jeffreyw128: 好奇是否有人在使用 LiteLLM？
  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1230653838386532464)** (1 条消息): 

- **请求分布式推理示例**：一位成员正尝试在 2x A100 集群上对 Jamba 进行 **长上下文推理（long context inference）**，但在分布式系统方面遇到了困难。他们正在询问是否有任何示例代码可以帮助解决此问题。