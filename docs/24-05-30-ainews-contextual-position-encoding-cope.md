---
companies:
- meta-ai-fair
- google-deepmind
- anthropic
- perplexity-ai
- langchain
- openai
date: '2024-05-31T03:11:48.061328Z'
description: '**Meta AI** 研究员 **Jason Weston** 介绍了 **CoPE**，这是一种针对 Transformer 的新型位置编码方法。该方法通过引入“上下文”来创建可学习的门控，从而提升了模型处理计数和复制任务的能力，并在语言建模和代码编写方面表现更佳。此外，该方法未来可能通过外部存储器进行扩展，以辅助门控计算。


  **Google DeepMind** 发布了经过快速推理优化的 **Gemini 1.5 Flash** 和 **Pro** 模型。**Anthropic**
  宣布 **Claude** 的工具使用（tool use）功能正式全面开放，增强了其为复杂任务调度工具的能力。**Alexandr Wang** 推出了 **SEAL
  排行榜**，用于对前沿模型进行私密的专家评估。


  **Karpathy** 对 **GPT-3** 发布四周年进行了回顾，强调了模型规模化和实际应用中的改进。**Perplexity AI** 推出了 **Perplexity
  Pages**，旨在将研究内容转化为视觉精美的文章，被 **Aravind Srinivas** 称为“AI 版维基百科”。'
id: 32c6f94e-a602-4966-a326-9f2dade3994b
models:
- cope
- gemini-1.5-flash
- gemini-1.5-pro
- claude
- gpt-3
original_slug: ainews-contextual-position-encoding-cope
people:
- jason-weston
- alexandr-wang
- karpathy
- arav-srinivas
title: '**上下文位置编码 (CoPE)**'
topics:
- positional-encoding
- transformers
- counting
- copying
- language-modeling
- coding
- external-memory
- tool-use
- model-evaluation
- inference-speed
- model-benchmarking
- scaling
- research-synthesis
---

<!-- buttondown-editor-mode: plaintext -->**兄弟，再来一个 RoPE 变体，就一个**

> 2024/5/29-2024/5/30 的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**391** 个频道，**4383** 条消息）。
预计节省阅读时间（按 200wpm 计算）：**478 分钟**。

虽然是平静的一天，但 CoPE 论文引起了一些关注：所以我们来聊聊它。

传统的 LLM 在[计数和复制等简单算法任务上存在已知问题](https://x.com/pfau/status/1796273583603302823)。这很可能是其位置编码（positional encoding）策略导致的缺陷。

Meta AI 的 Jason Weston 发布了关于 [CoPE](https://x.com/jaseweston/status/1795978611784089799) 的[论文](https://arxiv.org/abs/2405.18719)，这是一种针对 Transformer 的新型位置编码方法，它考虑了 *Context*（上下文），并创建了具有可学习索引的“门控（gates）”。

 
![image.png](https://assets.buttondown.email/images/ad0fb3fc-e851-46c7-8f03-6d9ae4f24043.png?w=960&fit=max)
 

通过使用该方法，CoPE LLM 可以：

- 根据需要按 Head “计数”距离，例如第 i 个句子或段落、单词、动词等。而不仅仅是 Token。
- 解决标准 Transformer 无法处理的[计数和复制任务](https://x.com/jaseweston/status/1795978614132920656/photo/1)。  
![image.png](https://assets.buttondown.email/images/d765bdf5-9ceb-4d1f-a7be-6eb4e2c214c9.png?w=960&fit=max)
 
- 在语言建模 + 代码任务上获得更好的 PPL。
 
![image.png](https://assets.buttondown.email/images/7c43c0a4-767d-4ea0-a687-f1ad52252cb0.png?w=960&fit=max)
 

**你甚至可以修改这个概念，使用 [external memory](https://x.com/krishnanrohit/status/1796061792201814466)（外部存储）而不仅仅是局部上下文来计算门控。**

正如 [Lucas Beyer 所指出的](https://x.com/giffmana/status/1796077219455869414)，今年涌现的大量位置编码变体或许是更丰富的研究源泉，因为“Linear attention 旨在移除模型容量，这从长期来看没有意义。而位置嵌入（Position embedding）则是为了**为模型增加缺失的能力**，这显然更有意义。”

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**新 AI 模型与基准测试**

- **Contextual Position Encoding (CoPE)**：[@jaseweston](https://twitter.com/jaseweston/status/1795978611784089799) 介绍了 CoPE，这是一种针对 Transformer 的新型位置编码方法，它将上下文纳入考虑，使其能够解决计数和复制任务，并提升了在语言建模和编程方面的性能。
- **SEAL Leaderboards**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1795857651592491281) 推出了 SEAL Leaderboards，用于对前沿模型进行私密的专家评估，这些评估具有不可利用性（unexploitable）且持续更新。
- **Gemini 1.5 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1796216673348833445) 在其 API 上发布了 Gemini 1.5 Flash 和 Pro 模型，其中 Flash 专为快速、高效的推理而设计，支持每分钟 1000 次请求。
- **Claude 支持工具使用**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) 宣布 Claude 的工具使用（tool use）功能正式开放（GA），使其能够为复杂任务智能地选择和编排工具。

**AI 应用与平台的进展**

- **ChatGPT 免费版升级**：[@gdb](https://twitter.com/gdb/status/1795970586050429005) 指出 ChatGPT Free 档位正在让尖端 AI 功能得到广泛普及。
- **Claude Tool Use GA**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) 使 Claude 的工具使用功能正式开放（GA），允许其智能地选择和编排工具，以端到端地解决复杂任务。
- **GPT-3 四周年**：[@karpathy](https://twitter.com/karpathy/status/1795980744436932871) 反思了 GPT-3 发布四周年，以及它如何证明只需通过训练更大的模型就能提升实际任务的表现，这使得更好的算法成为 AGI 进展的加分项而非必要条件。他提到，如果现在给他一台性能强 10 倍的计算机，他会非常清楚该用它做什么。
- **Perplexity Pages**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1796203494401040846) 推出了 Perplexity Pages，允许用户将研究成果转化为具有格式化图像和章节的精美文章。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1796220011448786949) 将 Perplexity 通过 Pages 满足世界好奇心的使命描述为“AI 维基百科”，只需“一键转换”即可完成分析来源和合成可读页面的工作。
- **Milvus Lite**：[@LangChainAI](https://twitter.com/LangChainAI/status/1796206411288039430) 与 Milvus 合作，通过结合双方的能力来简化强大的 GenAI 应用的创建。
- **Property Graph Index**：[@llama_index](https://twitter.com/llama_index/status/1795869279457546447) 推出了 Property Graph Index，提供了一个高级 API，用于使用 LLM 构建和查询知识图谱。
- **LangSmith 中的重复实验**：[@LangChainAI](https://twitter.com/LangChainAI/status/1796222825898074235) 在 LangSmith 中增加了对运行多次重复实验的支持，以平滑由于变异性带来的噪声。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**技术发展与合作伙伴关系**

- **OpenAI 合作伙伴关系**：OpenAI [宣布与](https://www.reddit.com/r/OpenAI/comments/1d3hf0b/vox_media_and_the_atlantic_sign_content_deals/) The Atlantic、Vox Media 和 WAN-IFRA 建立合作伙伴关系，以帮助新闻出版商探索 AI 集成。他们似乎还 [与 Apple 达成了一项交易](https://www.reddit.com/r/OpenAI/comments/1d3726q/what_do_you_actually_use_ai_for_on_a_regular_basis/)。/r/OpenAI 的讨论集中在 [人们如何日常使用 ChatGPT](https://www.reddit.com/r/OpenAI/comments/1d3gczj/anyone_else_talk_to_chatgpt_all_day/)。

- **Google Gemini 模型**：Google 将 Gemini 1.5 Flash 的 [输出价格翻了一番](https://www.reddit.com/gallery/1d3h2ev)。考虑到 API 成本，他们更新的 Gemini 1.5 0514 模型在 [Chatbot Arena 排行榜上评价良好](https://www.reddit.com/gallery/1d36gcn)。

- **Mistral AI 的 Codestral**：Mistral AI 推出了 [Codestral，一个 22B 参数的开源权重代码模型](https://mistral.ai/news/codestral/)，采用 Mistral AI Non-Production License 授权。The Verge [报道了 Codestral 的发布](https://www.theverge.com/2024/5/29/24166334/mistral-debuts-a-coding-assistant-called-codestral)。

- **Groq 加速 Llama 3**：Groq [宣布 Llama 3 在其系统上以每秒 1200+ tokens 的速度运行](https://x.com/GroqInc/status/1795919195076784340)。

**模型基准测试与评估**

- **AutoCoder 超越 GPT-4**：AutoCoder，一种新的代码生成模型，在 HumanEval 基准测试的 pass@1 指标上 [超越了 GPT-4 Turbo 和 GPT-4o](https://www.reddit.com/r/LocalLLaMA/comments/1d3qx5q/autocoder_a_new_model_designed_for_the_code/)。它还提供了一个功能更全的代码解释器。

- **Scale AI 的 SEAL 排行榜**：Scale AI 推出了 [带有私有数据集和付费标注者的 SEAL 排行榜](https://www.reddit.com/r/LocalLLaMA/comments/1d3idzn/scale_ai_are_introducing_high_quality_arenas_with/)，以便对前沿模型进行更公平、更高质量的专家评估。一份 [信息图解释了 SEAL 的方法](https://www.reddit.com/gallery/1d3n6ag)。

- **GPT-4 律师资格考试声明受到质疑**：一项 [MIT 的研究发现 GPT-4 在律师资格考试中的得分并未达到之前声称的第 90 百分位](https://link.springer.com/article/10.1007/s10506-024-09396-9)。

- **TimeGPT-1 在时间序列基准测试中夺冠**：在 30,000+ 时间序列基准测试中，TimeGPT-1 与 TimesFM、Chronos、Moirai 和 Lag-Llama 等其他基础时间序列模型相比，[在准确性和速度上排名第一](https://www.reddit.com/r/MachineLearning/comments/1d3h5fs/d_benchmarking_foundation_models_for_time_series/)。


**AI 硬件与性能**

- **AI 训练算力每年增长 4-5 倍**：[用于 AI 训练的算力规模每年增长 4-5 倍](https://www.reddit.com/r/singularity/comments/1d3xfhs/the_amount_of_compute_used_in_training_is/)，凸显了快速的进展。([1](https://i.redd.it/5fnoh21ubi3d1.jpeg))
- **Groq 将 LLama 3 性能更新至 1200+ tokens/秒**：[Groq 在其硬件上将 LLama 3 的性能更新至每秒 1200+ tokens](https://x.com/GroqInc/status/1795919195076784340)。
- **高通发布 Snapdragon X Plus/Elite 基准测试**：高通发布了 [Snapdragon X Plus 和 X Elite 的基准测试，显示 Hexagon NPU 具有 45 TOPS 的性能](https://www.notebookcheck.net/Qualcomm-releases-official-Snapdragon-X-Plus-and-Snapdragon-X-Elite-benchmarks-for-45-TOPS-Hexagon-NPU.841811.0.html)，从而实现高效的端侧 AI。
- **Sambanova 在 Llama 3 上创下 1000 tokens/秒的速度记录**：[Sambanova 系统在 Llama 3 8B 上达到每秒 1000 tokens](https://venturebeat.com/ai/sambanova-breaks-llama-3-speed-record-with-1000-tokens-per-second/)，创下了新的速度记录。


---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. 新型 AI 模型发布与基准测试**：

- 拥有 40B 参数的 **[Yuan2.0-M32 模型](https://x.com/osanseviero/status/1796082193044844590)** 在 Math/ARC 任务上超越了 **Llama 3 70B**，且在生成过程中仅使用了 3.7B 个激活参数。
- **Codestral 模型发布与集成**：**Mistral AI** 发布了 **Codestral-22B-v0.1**，这是一款支持 80 多种编程语言的代码生成模型。它在代码指令和 Fill in the Middle (FIM) 任务中表现出色，[其博客文章中提供了更多细节](https://mistral.ai/news/codestral/)。**LlamaIndex** 为 Codestral 提供了 [首日支持和教程 notebook](https://t.co/YxeyHhSjKU)，同时它也与 **Ollama** 兼容，支持通过 [LlamaIndex 直接支持](https://t.co/gsPHHF4c0K) 进行本地运行。
- **[K2](https://huggingface.co/LLM360/K2)** 是一款完全开源的模型，其计算量比 Llama 2 70B 少 35%，但性能却更胜一筹，展示了高效的 AI 工程能力。

**2. AI 系统的优化与进展**：

- **Whisper 模型优化实现 6.3 倍加速**：一位社区成员利用 **static cache**、**HQQ quantization**、**torchao 4-bit kernel** 以及 **torch.compile with fullgraph** 等技术成功优化了 **Whisper 模型**。这一组合带来了显著的 6.3 倍速度提升。一篇[详细的博客文章](https://github.com/karpathy/llm.c/pull/475)即将发布，分享此次优化过程的见解。
- 讨论涵盖了通过 **templating block sizes**（如 `blockDim.x`）来**显著提升 CUDA kernel 性能**，特别是在 fused classifiers 中。
- 建议使用 **Cloudflare R2** 替代 Python 依赖项和**内部 S3** 来共享大型数据集，从而优化成本并避免杂费。

**3. AI 模型微调与定制**：

- 成员们探讨了**微调 LLM**（如 Llava）以处理**图像和视频理解任务**的**理想策略**，并辩论了 **Direct Preference Optimization (DPO)** 相对于 **Supervised Fine-Tuning (SFT)** 的优势。
- 讨论了 **Anti-prompts** 技术，这是一种通过在预定义单词处停止生成来引导对话流的技术，允许用户在模型继续输出前进行干预。
- 分享了关于**微调 Llama3 base 模型**以及在指令模型上使用 **DPO** 来创建定制角色（如历史人物或虚构角色）的建议。

**4. 竞赛与开源倡议**：

- 宣布了 **NeurIPS 模型融合（Model Merging）竞赛**，提供 **[$8,000 奖金池](https://x.com/LChoshen/status/1796256513519989102)**，旨在彻底改变模型选择和融合技术，以创建最优的 LLM。
- **[LAION](https://laion.ai/notes/open-gpt-4-o/)** 呼吁社区贡献力量，共同构建具有大规模多模态能力的 **open GPT-4-Omni** 模型，并提供了数据集、教程和指导。
- 展示了来自近期黑客松的 **[Le Tigre 项目](https://devpost.com/software/le-tigre)**，这是一个基于 Mistral 7B 的多模态变体，灵感源自 GPT-4-V 的架构。

---

# 第 1 部分：Discord 高层摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity Pages 开创更精美的帖子**：**Perplexity AI** 推出了 **Perplexity Pages**，这是一款将研究转化为精选文章的工具，旨在打造 **AI 版维基百科**。该功能目前面向 Pro 用户开放，预计将向更多用户开放，详情见其 [博客文章](https://pplx.ai/pages)。

**Grok 的不足引发对卓越搜索的追求**：社区成员 sneakyf1shy 致力于构建一个优于 Grok 的模型，旨在增强 Perplexity Web 应用中的搜索功能。社区还讨论了现有模型、API 和索引数据的效能，指出了局限性并展望了改进方向。

**Pages 反馈：好、坏与丑陋**：试用 **Perplexity Pages** 的用户反馈不一；一些人称赞其实用性，而另一些人则遇到了内容板块缺失等问题。社区的情绪从对 Perplexity 索引能力的怀疑到对该功能的兴奋不等，一份 [操作指南](https://www.perplexity.ai/page/How-to-Use-FvLfzZ_ATyqE2n_tAGKk7A) 正在感兴趣的人群中传阅。

**API 焦虑与 Google vs. OpenAI 的宿怨之战**：技术讨论深入探讨了用户友好型 API 的可扩展性挑战以及多步推理的改进。与此同时，Google 与 OpenAI 的竞争备受关注，引发了对其 AI 战略举措的辩论，并伴随着对 AGI 进展和市场影响力的推测。

**好奇者探索 AI 伦理与物理**：**sharing** 频道重点展示了成员关于复杂话题在伦理和物理维度的贡献。关于 **意识**、**LLM 功能** 以及所谓的 **利弊分析** 的讨论链接表明，社区正致力于实质性且多样化的 AI 相关主题。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Google Gemini 的失误**：Google Vertex AI 定价的不一致引发了混乱，用户担心 Vertex AI 按字符计费与 Google 直接 AI 服务按 token 计费之间的差异，这引发了 [讨论帖](https://ai.google.dev)。

- **LLM 微调技巧**：社区分享了微调大语言模型 (LLMs) 的知识和经验，特别关注多智能体 (multi-agent) 系统中的确定性工具调用 (tool calling)，以及成功应用来自 [robocorp/llmstatemachine](https://github.com/robocorp/llmstatemachine) 等仓库的状态机逻辑。另一个焦点是使用 GGUF 格式通过自定义数据改进 LLM 的微调，这得到了一个活跃的 Hugging Face Pull Request 的支持，该 PR 提供了从 HF 到 GGUF 更简便的转换方法 ([来源](https://github.com/huggingface/transformers/pull/30928))。

- **拥抱 Modal 的多面机制**：关于 Modal 任务执行的辩论和故障排除非常激烈，重点讨论了数据集路径和 config 设置等问题。社区通过分享 WANDB 集成见解、共享配置文件以及引导用户查阅 [Modal 文档](https://modal.com/docs/guide/trigger-deployed-functions) 以进行深入学习来做出回应。

- **通过论文和资源包扩展学习**：一系列学习资源涌现，包括 Meta 关于 vLLM 的论文、[GitHub](https://github.com/marco-jeffrey/awesome-llm-resources) 上的 LLM 资源合集，以及 AI 编程 Humble Bundle 的详细信息。此外，一篇关于扩展 Llama3 上下文窗口的论文也引起了关注 ([来源](https://arxiv.org/abs/2404.19553))。

- **全球聚会与活动**：即将举行的 AI 活动备受关注，例如 6 月 7 日至 9 日在新加坡、悉尼和旧金山举行的 **Global AI Hackathon**，该活动由顶尖 AI 构建者支持，旨在探索“AI 让世界更美好”——感兴趣的参与者可以通过 [此链接](https://lu.ma/igqisb0e) 报名。同时，在 Discord 上，来自美国东西海岸和欧洲地区的成员表达了对当地见面会的积极热情并分享了场地信息。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 用户的福利**：ChatGPT 免费用户获得了重大升级，现在可以使用浏览 (browsing)、视觉 (vision)、数据分析 (data analysis)、文件上传和 GPTs，这为实验和开发开辟了新途径。
- **OpenAI 赋能非营利组织**：OpenAI 推出了 **OpenAI for Nonprofits**，为慈善组织提供更便捷的工具访问权限，这标志着通过先进的 AI 应用支持社会公益的战略举措。此外还讨论了更多细节，包括应对欺骗性 AI 用途的策略。
- **GPT-4 可用性与性能讨论**：社区围绕 **GPT-4 的可用性和性能**展开了热烈讨论，注意到免费用户可能会遇到自动模型切换，并对长文本 GPT-4 输出中出现的“胡言乱语 (word salad)”问题表示担忧。成员们还触及了 GPT 模型的定制化和潜在的内存增强。
- **编程辅助与 API 最佳实践**：AI 工程师对比了 **GPT-4o、Mistral 的 codestral 和 Copilot** 等代码辅助工具，强调了速度和准确性。他们还分享了使用代理后端服务器保护 API keys 的知识，以及在长时间会话中考虑 API 稳定性而非基于浏览器的交互的重要性。
- **AI 工具中的偏见与故障排除**：工程师们幽默地承认了在评估自己的 AI 工具时存在个人偏见，并交流了故障排除技巧，建议对低于 4 的版本拆分请求以保持兼容性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **PDF 与 AI 的 Everything-AI 集成**：[Everything-AI 项目](https://github.com/AstraBert/everything-ai)现在集成了 `llama.cpp` 和 `Qdrant`，允许与 PDF 进行交互式交流，社区贡献增强了 HuggingFace 的工具和模型库。

- **Yuan 2.0-M32 的竞争优势**：新发布的 **Yuan2.0-M32 模型**拥有 400 亿参数和创新架构，在 Math/ARC 任务中超越了 Llama 3 70B。该消息在 [Twitter](https://x.com/osanseviero/status/1796082193044844590) 上披露，并在 [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf) 上展示，同时附带了支持性的[研究论文](https://hf.co/papers/2405.17976)链接。

- **Nvidia Embed V1 让可视化变得触手可及**：一位用户分享了他们的 [Nvidia Embed V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) Space，用于展示 Nvidia 的嵌入模型，并邀请通过 PR 进行增强，以实现更精细的功能或令人兴奋的新示例。

- **Hugging Face 与 DuckDB 联手实现更流畅的数据集处理**：DuckDB 与 Hugging Face 数据集的融合（通过 `hf://` 路径实现）简化了数据集成流程，正如[教程博客文章](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c22db89519e4d)中所详述的那样，这标志着数据处理便利性的一大进步。

- **AI 社区为 NeurIPS 模型合并竞赛做好准备**：为 NeurIPS 宣布的一项专注于模型合并 (model merging) 的竞赛引起了 AI 社区的兴趣，承诺提供 8000 美元的奖励，并有机会突破模型选择技术的界限，正如[官方推文](https://x.com/LChoshen/status/1796256513519989102)所述。

- **Whisper 模型通过时间戳进行微调**：关于使用 Whisper 模型提取词级时间戳的讨论强调了该方法的文档，并将工作归功于诸如 *“Robust Speech Recognition via Large-Scale Weak Supervision”* 等研究，表明了音频处理及其应用的增强。

- **开源模型迎来 K2 的潜力**：两个全新的完全开源模型（包括 [K2](https://huggingface.co/LLM360/K2)）因其出色的表现而受到赞誉，特别是 K2，与 Llama 2 70B 模型相比，它在计算量减少 35% 的情况下表现优异，突显了高效 AI 模型工程取得的进展。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Codestral 加入代码模型之战**：**Mistral** 推出了 **Codestral-22B-v0.1**，能够处理 80 多种编程语言，在代码指令和 Fill in the Middle (FIM) 等任务中表现出色。对于有兴趣测试该模型的人，可以[在此下载并探索 Codestral-22B](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF)。

**永无止境的上下文长度挑战**：工程师们强调了 **llama 系列**等模型的局限性，其上限为 **4096** tokens，并指出 RoPE 扩展允许最大 **16k** tokens，同时对上下文大小的重要性进行了激烈的讨论。

**硬件讨论升温**：**RTX 5090** 因其传闻中的 **448-bit 总线**和 **28 GB GDDR7** 显存引发了推测。与此同时，关于 CPU 推理的务实比较以及 GPU 设置（如使用多张 **3090** 显卡）的优缺点占据了讨论的主导地位。

**Whisper 与 Amuse 成为焦点**：观察到 **Whisper 模型**与 **llama.cpp** 不兼容的技术问题，以及 **Amuse** 的 GitHub 链接失效。解决方案包括利用 **whisper.cpp** 以及通过可用的 [Hugging Face 链接](https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip)访问 **Amuse**。

**添加推理 GPU 的实用技巧**：一场讨论阐明了在 **LM Studio** 中添加额外 GPU 进行推理的现实情况，强调了对适当空间、电源和正确设置管理的需求，证明了摆弄硬件既是一门科学也是一门艺术。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama3 在 AI 对决中胜过 Phi3**：工程师们一致认为 **Llama3** 在测试中优于 Phi3，评论称赞了它的性能，并批评 Phi3 “*极其合成感*”。用户建议不要使用 Phi3 模型，强调了基础版 Llama3 的有效性。

- **完善角色扮演 AI**：建议从训练 Llama3 基础模型开始，然后进行指令遵循（instruction following）微调，以创建定制的角色扮演角色。然而，仅通过指令提示 Llama3 “*假设你是 [X]*” 可能比标准的微调过程产生更好的效果。

- **用于受控对话的 Anti-Prompts**：讨论了 anti-prompts 的效用，揭示了一种通过在预定义单词处停止生成来引导聊天模型对话流的策略。这种技术使用户能够在模型恢复输出之前进行干预。

- **模型训练与微调的陷阱**：讨论指出，通常不鼓励在指令模型之上进行微调，因为可能会导致价值损失。在基础模型上使用 Direct Preference Optimization (DPO) 可以更有效地为特定角色定制输出。

- **新兴模型与技术难题**：分享了对 Yuan 等新模型的热情，并提醒实际应用比基准测试结果更重要。一位用户遇到了 Apple M3 Pro GPU 与 CUDA 不兼容的问题，这引发了关于利用 Google Colab 等服务进行模型训练和微调的建议。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **节省 AI 训练成本**：成员们强调了经济高效地**训练 Stable Diffusion** 模型的方法，讨论了 **[Google Colab](https://github.com/hollowstrawberry/kohya-colab)** 等工具和 **RunDiffusion** 等服务，因为它们提供了预算友好的解决方案。

- **优化图像准确度**：讨论了增强**图像生成**的技巧，特别关注使用 **ControlNet** 和高级采样器。对于动态 LoRA 控制，社区分享了 **[sd-webui-loractl](https://github.com/cheald/sd-webui-loractl)** GitHub 仓库。

- **Ruby 加入 AI API 之战**：推出了一款用于 Stability AI API 的全新**开源 Ruby SDK**，旨在简化核心模型和 SD3 模型的图像生成任务。该 SDK 可以在 **[GitHub](https://github.com/OlympiaAI/stability)** 上找到并参与贡献。

- **对 SD3 的期待与焦虑**：社区交流了关于 **Stable Diffusion 3 可能发布**的想法，表达了对许可问题的担忧，并将财务支持与 **Midjourney** 等竞争对手进行了比较。

- **儿童友好型 AI**：发起了一场关于如何安全地**向儿童介绍 Stable Diffusion** 的讨论，重点是利用 **ControlNet** 负责任地将儿童的草图转化为精美的图像。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 与学习的未来**：一场关于 [GPT-4-OMNI](https://laion.ai/notes/open-gpt-4-o/) 作为教育助手效用的讨论正在蓬勃展开，社区对其多模态能力感到兴奋，这标志着个性化学习体验将迎来阶梯式的变革。

- **模型更新中的污染警报**：**Luxia 21.4b** 模型在 v1.0 到 v1.2 版本更新期间，数据污染率飙升了 29%，这在 [HuggingFace 上的 GSM8k 测试结果](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1)中得到了证实，尽管这一问题并未影响其他测试基准。

- **位置编码走向上下文感知**：对话中讨论了 **Contextual Position Encoding (CoPE)**，这是一种对传统位置编码的新颖改进。正如 [Jason Weston 的推文](https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw)所强调的，它提升了语言建模和编程任务的表现。

- **重量级对决：MLPs vs. Transformers**：社区对 **MLP-Mixer 在因果性和序列长度方面的限制**进行了批判性讨论，引发了对 MLP 作为静态权重与 Transformer 具备动态上下文相关权重能力的深入探讨。

- **解码模型性能**：贡献内容包括分享一篇关于[学习率和权重平均的 Arxiv 论文](https://arxiv.org/abs/2405.18392)，辩论梯度多样性在 mini-batch SGD 性能中的作用，并宣布了一项奖金高达 8,000 美元的 NeurIPS 竞赛，专注于模型合并（model merging），该消息由 [Leshem Choshen 发布在推文上](https://x.com/LChoshen/status/1796256513519989102)，并托管在[竞赛官方页面](https://llm-merging.github.io/)。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 难题解决**：工程师们发现了 **[Triton 代码中的一个 bug](https://github.com/triton-lang/triton/issues/2302)**，该 bug 会导致 int32 乘法溢出。这揭示了生产规模的数据如何暴露单元测试中不明显的限制，例如 CUDA 中 16 位的网格维度（grid dimension）限制。
  
- **性能调优揭秘**：有建议指出，对 `blockDim.x` 等块大小进行模板化（templating）可以显著提升 CUDA kernel 的性能。讨论还包括提议合并 **layernorm recomputations** 的分支，以便在重新调整以减少冗余之前，优先优化功能性改进。

- **Whisper 模型获得超级更新**：一位成员通过利用 **static cache**、**HQQ quantization**、**torchao 4-bit kernel** 以及带有 **fullgraph** 的 **torch.compile**，成功优化了 **Whisper 模型**，实现了 6.3 倍的加速，并承诺会发布一篇[详细的博客文章](https://github.com/karpathy/llm.c/pull/475)。

- **低精度乘法器的复杂细节**：查询范围从指定 **fp4 multiplication** 的精确操作到探索激活和梯度中的**混合精度层**。提到了一个用于 **FP6-LLM** 的 CUDA kernel，展示了针对 fp16 激活与 MX fp6_e3m2 权重的混合输入乘法，计算通过 tensor cores 执行。

- **利用 Cloudflare R2 和内部 S3 的巧妙规避方案**：工程师们讨论了使用 **Cloudflare R2** 来降低出站流量费用（egress fees）和 Python 依赖项，同时考虑使用预先上传资源的内部 S3 存储来共享大型数据集而不产生额外费用。这与关于 **[安装错误与兼容性](https://github.com/pytorch/ao/pull/296)** 的讨论相呼应，包括处理需要增强 CUDA 能力的构建以及避免隔离环境问题的技巧。

这些针对性的讨论反映了社区对实现性能提升、优化成本效率以及解决大规模部署机器学习模型时面临的实际问题的关注。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Codestral 问世并支持多语言**：**Mistral AI** 推出了 **Codestral**，这是一款支持 80 多种编程语言的新型本地代码生成模型。LlamaIndex 在发布首日即实现了集成，并提供了教程 [notebook](https://t.co/YxeyHhSjKU)。它还兼容 **Ollama**，通过直接 [支持](https://t.co/gsPHHF4c0K) 提升了本地执行能力。

- **在本地构建知识图谱**：工程师们讨论了结合 **Ollama** 模型与 **Neo4j** 数据库在本地构建知识图谱的方法，并提供了一份详尽的 [指南](https://t.co/5ee6LwM7RE) 和额外的 [操作细节](https://t.co/xhoIEi9egq)。

- **聚焦金融洞察的 NLP 见面会**：伦敦将举办一场由 **LlamaIndex**、**Weaviate_io** 和 **Weights & Biases** 参与的 NLP 见面会，重点讨论在金融服务中使用 LLM，以及向量数据库管理的相关议题，并提供 [报名](https://t.co/vli6DY8Xg7) 链接。

- **LlamaParse 扩展格式处理能力**：**LlamaParse** 改进了处理 Excel 和 Numbers 等电子表格的功能，使其更易于在 RAG 管道中使用；可通过提供的 [notebook](https://t.co/60MvR0h5DC) 和 [demo](https://t.co/IfF4UUqB0C) 了解更多信息。

- **探索 API 框架和数据存储领域**：社区交流了选择 **API 框架** 的见解，并对具有异步能力的 **FastAPI** 表示认可。此外还讨论了从 **SimpleStore** 迁移到 **RedisStore** 的数据存储转型，包括使用 `IngestionPipeline` 的策略。分享了相关文档和示例链接，包括一个 [Google Colab](https://colab.research.google.com/drive/1hiDkBbAJcO3RDrS7CD2ZeQEHGqNv07pq?usp=sharing) 和多个 [LlamaIndex 资源](https://docs.llamaindex.ai/en/latest/examples/vector_stores/pinecone_auto_retriever/)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Le Tigre 咆哮进入多模态领域**：工程师们一直在讨论 "Le Tigre"，这是一个基于 Mistral 7B 模型的多模态项目，受 GPT-4-V 架构启发，已在 [Devpost](https://devpost.com/software/le-tigre) 和 [GitHub](https://github.com/HugoLB0/Le-Tigre) 上展示。人们对 LAION 5B 数据集充满期待，但其发布时间仍不确定。

- **Sonic 声音非凡**：Cartesia AI 推出了 Sonic，这是一款 *state-of-the-art* 生成式语音模型，因其逼真的音质和惊人的 135ms 延迟而备受赞誉；详情可通过其 [博客](https://cartesia.ai/blog/sonic) 和 [Twitter](https://twitter.com/cartesia_ai/status/1795856778456084596) 公告查看。

- **模型合并热潮**：NeurIPS 模型合并竞赛（Model Merging Competition）引发了热议，该竞赛设有 8,000 美元的奖金池，旨在推进模型合并技术。同时，关于在 Transformer 中使用 FFT 替代 Self-attention 的议题激发了学术好奇心，灵感来自一篇建议该方法能以更低计算需求达到接近 BERT 准确度的 [论文](https://arxiv.org/pdf/2105.03824)。

- **ToonCrafter 助力卡通制作**：ToonCrafter 是一个专为草图引导动画设计的项目，引发了工程师们的怀疑与好奇。他们指出该项目有潜力颠覆传统的动漫制作成本，可能将数十万美元的成本大幅降低。

- **GPT-4-Omni 开放征集**：LAION 宣布征集开源 GPT-4-Omni 项目的贡献，旨在促进大规模多模态能力的协作开发，详见其 [博客文章](https://laion.ai/notes/open-gpt-4-o/)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **通过及时提示词教学 AI**：讨论强调了通过包含特定上下文的提示词和响应来提升模型性能，重点关注使用 100k 或更小上下文窗口的 [in-context learning](https://twitter.com/VictorTaelin/status/1776677635491344744) 策略；这可以简化高效的数据处理，其中像 **RWKV** 这样具有状态保存功能的模型可能具有节省时间的优势。

- **超越反向传播与模型合并**：放弃传统 backpropagation 的新型训练方法引起了关注，暗示了模型效率的潜在复杂性和变革性影响。NeurIPS **model merging competition**（模型合并竞赛）已经公布，奖金池为 8,000 美元；更多详情可通过[特定推文](https://x.com/LChoshen/status/1796256513519989102)获取。

- **以小博大，超越巨头**：最近发布的 **Yuan2-M32** 模型拥有 40B 参数，但在生成过程中仅有 3.7B 激活参数，在基准测试中以更低的资源消耗与 **Llama 3 70B** 旗鼓相当，引发了社区对其进行 [fine-tune](https://github.com/IEIT-Yuan/Yuan2.0-M32) 并利用其能力的呼声。

- **应对专业化 AI 工具时代**：目前的趋势是群体更倾向于具有通用能力的 Large Language Models (LLMs)，而非利基模型；社区成员兴奋地分享了诸如用于 LLM 应用的 [rust library](https://x.com/LChoshen/status/1796256513519989102) 以及 **MoRA**（一种用于微调期间高秩更新的工具，可在 [GitHub](https://github.com/kongds/MoRA) 上获取）等创新成果。

- **开启 RAG 数据集访问权限**：一个新的 **RAG dataset** 已在 [Hugging Face](https://huggingface.co/datasets/glaiveai/RAG_v1) 上线，用户需同意分享联系方式即可获取。同时，社区还讨论了衡量相关性的指标，如 **MRR** 和 **NDCG**，并基于 [Hamel et al](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) 的见解对其进行了评判。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Swift 拥抱 ABI 稳定性**：在关于 ABI 稳定性的讨论中，有人指出 **Swift 为 Apple 的操作系统保持了 ABI 稳定性**，而 **Rust** 则刻意避开了它。保持 ABI 稳定性可能会限制某些编程语言性能提升的潜力。

- **对 Mojo 潜力的怀疑**：关于 Mojo 成为广泛采用的低级协议的想法遭到了质疑，理由是其缺乏某些关键类型，且难以取代像 **C** 这样根深蒂固的语言。

- **Mojo 寻求更好的 C++ 互操作性**：**Modular** 社区强调了 **C++ interoperability** 对 Mojo 成功的重要性，并讨论了未来可能支持从 Mojo 代码生成 C++ 头文件的计划。

- **包管理与 Windows 支持**：Mojo 包管理器正在持续开发中，[GitHub 讨论](https://github.com/modularml/mojo/discussions/413)和[提案线程](https://github.com/modularml/mojo/discussions/1785)证明了这一点。然而，社区对 Mojo 仍不支持 **Windows** 表示了不满。

- **理顺 Nightly 版本**：一个重要的 Mojo nightly 构建版本 `2024.5.3005` 已经发布，其中包含重大更改，例如从 `String` 中移除了 `Stringable` 构造函数和几个 `math` 函数。此外，大约 25% 的 Mojo 安装来自 nightly 版本，以确保为新手提供简单的体验。这些更改引起的问题已得到解决，例如修正了 `String` 到 `str` 的转换，以及 CI [PR #2883](https://github.com/modularml/mojo/pull/2883) 中的修复。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **MixMyAI：新晋选手**：[mixmyai.com](https://mixmyai.com) 的发布引起了社区关注，该平台自称为全能 AI 工具箱，具有无月费和以隐私为中心等吸引人的特性。

- **免费层级的秘密仍未揭开**：关于如何获取某项未指明服务的免费层级的讨论引起了兴趣，但获取这种难以捉摸的福利的方法仍然是一个谜，目前尚无明确解决方案。

- **人才招聘**：一位拥有全栈、Blockchain 和 AI 技能的高级开发者宣布正在寻找新机会，这表明该社区是潜在招聘和协作的热点。

- **模型行为：受控 vs. 自我审查**：关于模型的澄清出现了，区分了自我审查的模型和使用外部审查器的模型；特别指出了 OpenRouter 上类似 Claude 模型的独特设置。

- **编程包发布**：OpenRouter 与 Laravel 和 Ruby 的集成包的创建和发布——包括 [laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) 和 [open_router](https://github.com/OlympiaAI/open_router)——展示了活跃的社区贡献和跨语言支持。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **通过 API 进行特定领域的网页搜索**：一位用户描述了如何使用 **API options object** 为特定域名设置网页搜索连接器；关于多域名限制的后续讨论正在进行中。
- **用于学术创新的 AI**：个人正在开发一个 Retrieval-Augmented Generation (RAG) 模型，以增强其大学的搜索能力，并详细说明了包含 **.edu 域名**和外部评论网站（如 **RateMyProfessors**）的意图。
- **Embedding 类型转换策略**：提出了将 **uint8 embeddings 转换为 float** 以进行数学运算的问题，该用户被引导至更专业的开发频道以获得深入帮助。
- **初创公司寻求用户留存见解**：一家初创公司提供 **10 美元奖励**，征求对其无代码 AI 工作流构建器的反馈，以分析用户注册后的流失情况，并注明讨论应在更相关的频道继续。
- **Cohere 的市场策略**：一名 Cohere 员工强调，公司并不追求 **Artificial General Intelligence (AGI)**，而是致力于为生产环境开发**可扩展模型 (scalable models)**。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**使用 `ChatMessageHistory` 记录对话**：Kapa.ai 展示了使用 **LangChain** 的 `ChatMessageHistory` 类来**持久化聊天对话**，提供了一个跨会话维护上下文的清晰示例，并参考了 [LangChain 文档](https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory)。

**应对 LLM 对话复杂性**：讨论集中在利用 Large Language Models (LLMs) 设计**非线性对话流**的困难，提到了提取和 JSON 处理方面的疑虑。链接了一个 **GitHub** 上的实验性方法来展示这些挑战。

**构建分析型 Copilot**：工程对话包括将 **LangChain** 与 **PostgreSQL 数据库**配对的策略，提供了通过 few-shot learning 处理模糊 SQL 查询结果的见解。

**增强交互性的混合 Agent**：揭示了 **LangChain** 中 `create_react_agent` 和 `create_sql_agent` 的集成，详细说明了避免常见初始化陷阱的步骤，以及正确命名工具对成功运行的重要性。

**进化的 AI 助手与知识图谱**：**Everything-ai v3.0.0** 等新版本的发布包含了集成 **llama.cpp** 和 **Qdrant 支持的向量数据库**等进展，而[在各频道分享](https://www.youtube.com/watch?v=Bxj4btI3TzY)的一段教程视频为学习者提供了使用 **Pinecone**、**LangChain** 和 **OpenAI** 创建 Bot 的实用指南。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **价格上涨引发性价比讨论**：社区成员讨论了某项未具名服务的剧烈价格变动，对其此前广受赞誉的性价比提出了质疑；人们怀疑之前的赞誉是否是基于涨价后的费率。
- **GPT-5 猜测升温**：一张来自 [X 的未证实表格](https://x.com/VachanReddyK/status/1795099977766551828/photo/1) 讨论了 GPT-5，引发了关于 OpenAI 可能会为了给新模型铺路而将 GPT-4o 免费化的猜测；文中还提到了 AI 专家 Alan D. Thompson 的见解 [关于 Alan](https://lifearchitect.ai/about-alan/)。
- **OpenAI 定价因拼写错误被指责**：OpenAI 最初的定价公告中出现了一个拼写错误，引发了混乱，随后在 24 小时内得到了解决和修正；修正后的定价现在反映了公司的真实意图 [LoganK 的官方帖子](https://x.com/officiallogank/status/1796044236594278544?s=46)。
- **OpenAI 的商业化转型引发不满**：OpenAI 的内部紧张局势在讨论中浮出水面，讨论提到了 Microsoft 据称向该公司施压，要求其优先考虑商业化而非研究，导致员工之间出现分歧 [《金融时报》文章](https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075)。
- **OpenAI 与 Apple 的合作引起轰动**：鉴于 Microsoft 对 OpenAI 的投资，社区反思了 Azure-Apple 合作伙伴关系中的战略影响和潜在冲突；商业动态与数据政策考虑的结合正受到密切关注。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 文档备受关注**：[OpenInterpreter 文档](https://docs.openinterpreter.com/settings/all-settings#auto-run) 获得了积极关注，其中列出的语言模型中，著名的 **LiteLLM** 支持 100 多种模型。此外，专门为 **RayNeo X2** 和 **Brilliant Labs** 镜框量身定制的 Android/iOS 客户端开发也引起了关注，社区渴望测试通过 [GitHub](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile) 分享的应用程序。

- **LLaMA 引发热烈讨论**：工程师们就本地使用 **LLaMA** 展开了激烈辩论，特别是对于运行温度较高的 NVLinked 3090 配置。有人提出了替代方案，包括利用 **Groq** 获取免费模型访问权限，将对话引向更可持续且高效的硬件解决方案。

- **TTS 热情中夹杂担忧**：关于使用个人声音定制 TTS 的查询引发了好奇，但目前没有提供直接的解决方案。与此同时，一名成员询问了 2024 年 4 月 30 日下的订单的发货情况，随后被引导至特定的置顶制造更新，这暗示了产品开发人员在沟通上的运营重点。

- **M5 Cardputer 激发期待**：关于 M5 cardputer 的更新引起了一些骚动，平衡了用户的兴奋与怀疑，在概述最新制造细节的置顶消息中可以找到相关保证。此外，还流传着一条关于仅出于教育目的使用 [Hugging Face 上的 ChatTTS 模型](https://huggingface.co/2Noise/ChatTTS) 的警告提醒，强调要遵守学术诚信。

- **对 Codestral 模型的好奇心达到顶峰**：对新 Codestral 模型的咨询激发了成员的兴趣，暗示了测试和审查的潜力。社区似乎愿意探索新的模型奇迹，突显了对最新模型开发的积极参与。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ChatGPT 免费版功能大幅增强**：OpenAI 增强了 *ChatGPT Free* 层的能力，新增了：浏览、视觉、数据分析和文件上传功能。用户正将“**rate limits**”（速率限制）视为潜在的制约因素，官方公告详见[此处](https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww)。

- **对话式语音 AI，A16Z 大举押注**：成员们在讨论 [a16z 对语音 AI 的投资](https://x.com/omooretweets/status/1795834644732285402)时，怀疑与兴趣交织，并推测 AI 将如何超越投资者的兴奋点，彻底改变电话通话。

- **Cartesia 凭借 Sonic 突破音障**：Cartesia 发布了 [Sonic](https://play.cartesia.ai/)，这是他们全新的低延迟生成式语音模型，引发了关于其在实时多模态场景中应用的讨论。欲了解更多见解，请查看他们的[博客文章](https://cartesia.ai/blog/sonic)。

- **YC 领导层变动解析**：Paul Graham 在 Twitter 上澄清了关于 Sam 离开 Y Combinator 的猜测，并在[他的推文](https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)中否认了被解雇的传闻。

- **增强检索的 Embedding Adapters**：工程界密切关注 TryChroma 关于 [embedding adapters](https://research.trychroma.com/embedding-adapters) 的技术报告，重点在于提高检索性能，这一概念与 Vespa 使用冻结 embeddings 的做法密切相关。

- **播客深度解析百万上下文 LLM**：由 @markatgradient 参与的新播客节目讨论了训练百万上下文 LLM 的挑战，引用了历史方法以及 RoPE、ALiBi 和 Ring Attention 等变体。该节目可在[此处](https://x.com/latentspacepod/status/1796247856891969709)收听。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**LLM360 启动社区 AMA**：Mozilla AI 的 LLM360 通过[关于其新 65B 模型和开源计划的 AMA](https://discord.com/events/1089876418936180786/1240722407594004561) 开启社区互动，促进与 AI 爱好者的知识共享和问答。

**湾区工程师，请关注**：一场[线下开源 Hack Lab 活动](https://discord.com/events/1089876418936180786/1243732435674337413)已安排在湾区举行，邀请当地成员协作并分享专业知识。

**Embeddings 洞察会议**：关于利用 [llamafiles 生成 embeddings](https://discord.com/events/1089876418936180786/1242590711778381914) 的社区会议，为寻求在机器学习项目中应用 embeddings 的工程师提供了实践学习体验。

**Mozilla AI 增强开发者支持**：在“Amplifying Devs”活动中，由主持人引导的讨论将集中于如何更好地支持 Mozilla AI 内部的开发社区，这是开发者成长和协作的重要平台。

**解决 LlamaFile 难题**：工程师们报告了在 M2 Studio 上运行 `granile-34b-code-instruct.Q5_0.llamafile` 并使用 Python 中的 VectorStoreIndex 时遇到的挑战，解决方案涉及正确的 IP 绑定和处理 WSL localhost 的特殊问题。对具有视觉/图像能力的 LlamaFiles 的兴趣正在增长，Mozilla 的 llava-v1.5-7b-llamafile [已在 Hugging Face 上架](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main)，可能为创意 AI 应用提供图像支持。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**针对多媒体任务微调 LLMs**：成员们正在探索微调 **large language models** (LLMs)（如 **Llava**）的理想策略，以处理涉及**图像和视频理解**的任务。关于使用 **Direct Preference Optimization** (DPO) 相对于 **Supervised Fine-Tuning** (SFT) 的益处和实用性引发了热烈讨论，特别是关于有效进行 DPO 所需的数据量。

**DPO 降低了 VRAM 占用**：一位工程师发现 DPO 期间 **VRAM usage** 意外减少，这引起了人们的兴趣，并引发了对近期可能导致这种效率提升的更新的猜测。

**寻找 Protobuf 高手**：社区内正在公开征集在 **Google's Protobuf** 方面有深厚背景的专家，特别是那些具备逆向工程、恶意软件分析或 Bug 赏金猎人技能的人才。

**SDXL 定制广告活动遇到障碍**：有人请求在**精炼 SDXL 模型**方面提供专业指导，目标是优化模型以生成定制的产品广告，但目前使用 **LoRA training** 或 **ControlNet** 尚未获得理想效果。

**小数据实现大对话**：大家对于仅有**几百个样本**的小型数据集是否足以成功进行 DPO 感到好奇，特别是针对像日常闲聊（chitchat）这样细微的领域。有人建议，手动编译这样一个数据集可能是一种可行的方法。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI 驱动的文学到游戏转型**：Rosebud AI 正在举办 **Game Jam: "Book to Game"**，参与者将使用 Phaser JS 在 AI Game Maker 平台上将书籍转化为游戏，角逐 **$500 奖金**，提交截止日期为 7 月 1 日。有关此次活动的消息通过 [Rosebud AI 的推文](https://x.com/Rosebud_AI/status/1796273820044595368)分享，感兴趣的开发者可以加入他们的 [Discord 社区](https://discord.gg/rosebud-ai)。

**Android 访问困扰**：Discord 社区的一位新成员描述 Android 体验“有点难以导航……不稳定且有 Bug”，但确认他们仍能参与内容互动。他们还询问了如何更改用户名，表达了一种感觉自己像个“外星人”的困惑。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GPU 未来推测引发好奇**：关于未来 **2 到 5 年** GPU 演进的讨论暗示了更大规模 **64x64 matrix multiplication arrays (MMA)** 的使用，并开玩笑地建议“做一个更大的脉动阵列（systolic array） 😌”。

- **Tinygrad 在整数梯度方面胜过 Torch**：Tinygrad 因其计算整数梯度的能力而受到关注，而这项任务在 Torch 中会导致 `RuntimeError`。Tinygrad 的处理方式是在反向传播（backpropagation）期间将整数视为 float，然后再转回整数。

- **辩论 AI 框架的主导地位**：一位成员断言 **Tinygrad** 优于 **TensorFlow 和 PyTorch**，引发了关于尽管个人偏好 Tinygrad，但为什么 AI 社区可能更倾向于 TensorFlow 而非 PyTorch 的讨论。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **提议特定语言的 Codestral**：一位成员发起了一项讨论，关于通过将 **Codestral** 拆分为单个编程语言来减小其体积的可能性，并假设并非所有语言对整体模型的贡献都是平等的。
- **对语言权重的关注**：人们对 **45GB Codestral 模型**中的权重分布感到好奇，推测大部分权重被分配给了英语，但每种编程语言可能仍会对模型的整体能力产生重大影响。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

遗憾的是，由于只提供了一条消息，且该消息缺乏与 AI 工程师相关的足够技术内容或细节，无法按照给定指南创建摘要。如果提供了具有适当细节的更多消息，可以生成摘要。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **加入 Open GPT-4-Omni 倡议**：[LAION](https://laion.ai/notes/open-gpt-4-o/) 呼吁社区贡献力量以开发开源版本的 GPT-4-Omni，并提供数据集、教程和指导性博客文章。他们还通过一篇 [Twitter 帖子](https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19)发布了这一消息，鼓励更广泛的参与。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


{% if medium == 'web' %}

# 第二部分：各频道详细摘要与链接



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1245766462090776747)** (1 条消息): 

- **Perplexity Pages 将研究转化为文章**：Perplexity 推出了 **Perplexity Pages**，帮助用户将他们的研究转化为视觉上吸引人的文章。用户可以在其 Library 中开始创建 Pages，更多信息可在 [Perplexity 的博客](https://pplx.ai/pages)上找到。
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1245451617420181536)** (686 条消息🔥🔥🔥): 

```html
- **Grok fails to impress; sneakyf1shy builds better search model**: Users discussed their disappointment with Grok and sneakyf1shy mentioned working on a similar project with intentions of enhancement. They aim to surpass Perplexity's web app by creating a comprehensive custom searching pipeline.
- **OpenAI and API enhancements**: Conversations highlighted the challenges of creating user-friendly APIs and scaling them effectively. Some users, such as sneakyf1shy, expressed interest in developing API solutions that could improve multi-step reasoning and integrating own indexing/cache layers.
- **Perplexity Pages gains traction; user experiences varied**: Many users explored Perplexity Pages, sharing their experiences and learnings. Some users encountered issues like missing sections in converted threads, while others found it a valuable addition for documentation and knowledge databases. One user shared a [Perplexity Pages guide](https://www.perplexity.ai/page/How-to-Use-FvLfzZ_ATyqE2n_tAGKk7A).
- **Skepticism and API limitations**: Users expressed skepticism about Perplexity's use of its own index, questioning the true capabilities of their web scraper. Some lamented the inactivity and limited availability of the API, while others discussed alternative models and their efficiencies.
- **Google and OpenAI comparisons stir debate**: Lively debates ensued about Google’s and OpenAI’s AI strategies, resource usage, and effectiveness in comparison to competitors like Nvidia. Users speculated on AGI developments and commercial impacts, especially regarding OpenAI's products and potential future releases.
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ollama.com">Ollama</a>：快速上手大语言模型。</li><li><a href="https://www.theverge.com/2024/5/29/24167511/openai-appears-to-have-closed-its-deal-with-apple">OpenAI 似乎已完成与苹果的交易。</a>：据彭博社报道，苹果曾与 Google 和 OpenAI 就将他们的聊天机器人集成到 iOS 18 中进行过谈判，但目前看来 OpenAI 胜出了。双方计划在苹果开发者大会上宣布这一消息……</li><li><a href="https://tenor.com/view/cute-adorable-sticker-stickers-gif-15884535962552606625">可爱迷人的 GIF - 可爱迷人的贴纸 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=-qGa0oTY120">Perplexity Pages 介绍</a>：您已经使用 Perplexity 搜索答案、探索新话题并扩展知识。现在，是时候分享您的所学了。向您介绍 Perplexity Pages……</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">怀疑按下 X GIF - Doubt Press X La Noire - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://start.solidjs.com/">SolidStart：细粒度响应式（Fine-Grained Reactivity）走向全栈</a>：SolidStart 是一个 JavaScript Framework，旨在构建 SolidJS 应用并将其部署到各种服务商。</li><li><a href="https://perplexity.typeform.com/pages-beta">Perplexity Pages - Beta 访问权限</a>：使用 Typeform 将数据收集变成一种体验。创建美观的在线表单、调查、测验等。免费试用。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1245456771984785460)** (15 messages🔥): 

- **Perplexity 关于优缺点的辩论**：一名成员分享了一个深入探讨某个话题**优缺点**的链接。点击[此处](https://www.perplexity.ai/search/Vor-und-Nachteile-jyWAvvwhT1qoWsdFiCP7mQ)查看完整讨论。
- **理解分歧性问题**：两名成员分享了同一个指向 Perplexity 搜索的链接，内容关于 **division**（分歧/除法）——可能是在详细讨论一个具有争议的话题或技术查询。点击[此处](https://www.perplexity.ai/search/what-is-div-44iy0Oo.SqSCDZjhN.IPiw)探索搜索结果。
- **深入探讨物理学与伦理学**：一名成员分享了深入探讨特定主题的**物理学**和**伦理学**的页面链接。阅读关于 [The Physics of](https://www.perplexity.ai/page/The-Physics-of-zJlhgwErRNiV5RndBBtOfA) 和 [The Ethics of](https://www.perplexity.ai/page/The-Ethics-of-gIYG3OV8TGm.neismMEbWQ) 的完整文章。
- **探索意识与 AI 功能**：一名成员重新分享了讨论**意识**和 **LLM 功能**的热门 Beta 页面，旨在获得更多观点和反馈。访问关于 [意识](https://www.perplexity.ai/page/Understanding-the-Conscious-Gtw786J5QQe4EpR4TfusXw) 和 [LLM 如何运作](https://www.perplexity.ai/page/How-LLMs-function-h515ZojmQFiTCxRHB3OEyw) 的讨论。
- **Perplexity 的新 AI Wikipedia 功能**：Arav Srivinas 详细介绍了 Perplexity 通过 Pages 创建 **AI Wikipedia** 的愿景，该功能目前已在 Pro 版上线，很快将面向所有人开放。在 Twitter [此处](https://x.com/AravSrinivas/status/1796220011448786949)查看完整公告和详情，并在[此处](https://www.perplexity.ai/hub/blog/perplexity-pages?utm_medium=social&utm_campaign=pages-launch)查看博客文章。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1796220481055654072">Aravind Srinivas (@AravSrinivas) 的推文</a>：并非每个人都需要通过提问和 Prompt Engineering 对话会话来获取 Perplexity 的日常知识。我们是第一个允许通过永久链接分享“线程（threads）”的。然而...</li><li><a href="https://x.com/AravSrinivas/status/1796221195542786522">Aravind Srinivas (@AravSrinivas) 的推文</a>：这只是更多惊人事情的开始：生成完整的调研报告、博客文章、关于某个主题的整本书，或者关于最新动态的简报或个人传记。...</li><li><a href="https://x.com/AravSrinivas/status/1796220757514842262">Aravind Srinivas (@AravSrinivas) 的推文</a>：你可以像写文档一样创建一个页面作为独立实体（具有完整的互联网访问权限），或者你可以像今天一样继续在 Perplexity 上提问并将其转换为...</li><li><a href="https://x.com/AravSrinivas/status/1796220011448786949">Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity 的使命是满足世界的好奇心。我们从带有引用的 Wikipedia 中汲取了灵感。我们很高兴能通过推出 Pages 进一步发展，它被描述为“AI Wikipedia”...
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1245462218842312927)** (50 messages🔥): 

- **Gemini 的 Google 定价困惑**：成员们对 Google Vertex AI 及其直接 AI 服务之间的定价差异感到困惑（[来源](https://ai.google.dev/)）。有人指出，“在 Vertex AI 上是按字符计费，而在 [ai.google.dev](https://ai.google.dev) 上是按 Token 计费”。
  
- **多 Agent 系统中的确定性 Tool Calling**：几位成员讨论了构建具有 Tool Calling 能力的 GPT 驱动的 Agent 策略。分享了一个基于状态机逻辑的资源：**GitHub - robocorp/llmstatemachine**（[来源](https://github.com/robocorp/llmstatemachine)）。
 
- **统一的课程资源仓库**：一名成员创建了一个 GitHub 仓库，用于整合课程期间分享的有用链接和幻灯片（[来源](https://github.com/bikash119/mastering_llm)）。另一个分享的资源是包含该课程所有往届演讲者的 Twitter 列表（[来源](https://x.com/i/lists/1796060854359580751)）。
 
- **关于额度表单的查询和问题**：多条消息讨论了额度表单未发送确认函的问题。**Danbecker** 确认该表单不发送确认函，并对用户的耐心表示感谢。
  
- **Finetuning LLM 与 GGUF 格式**：关于使用 GGUF 格式在自定义数据上 Finetuning LLM 的讨论非常普遍。大家对即将推出的简化 HF 到 GGUF 转换的改进感到兴奋，并分享了一个正在进行的 HuggingFace PR 作为相关更新（[来源](https://github.com/huggingface/transformers/pull/30928)）。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ankur-singh.github.io/blog/finetune-inference">使用 Ollama 运行微调后的 LLM</a>：演示如何在你的数据上微调 LLM 并使用 Ollama 运行的完整工作流。</li><li><a href="https://tenor.com/view/rug-pull-gif-21378865">Rug Pull GIF - Rug Pull - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://ai.google.dev/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/swyxio/ai-notes/blob/main/Resources/Good%20AI%20Podcasts%20and%20Newsletters.md">ai-notes/Resources/Good AI Podcasts and Newsletters.md at main · swyxio/ai-notes</a>：为软件工程师快速了解 AI 新进展提供的笔记。作为 https://latent.space 写作和产品头脑风暴的数据存储，并清理了规范引用...</li><li><a href="https://youtu.be/Wo95ob_s_NI?si=S-Kxzq01GKmGs6oa">John Schulman (OpenAI 联合创始人) - 推理、RLHF 及 2027 AGI 计划</a>：John Schulman 谈论 posttraining 如何驯服 shoggoth，以及未来进展的本质...时间戳：00:00:00 Pre-training, post-training, 以及未来的能力...</li><li><a href="https://github.com/robocorp/llmstatemachine">GitHub - robocorp/llmstatemachine：一个用于构建具有状态机逻辑和聊天历史记忆的 GPT 驱动 Agent 的 Python 库。</a>：一个用于构建具有状态机逻辑和聊天历史记忆的 GPT 驱动 Agent 的 Python 库。 - robocorp/llmstatemachine</li><li><a href="https://x.com/i/lists/1796060854359580751)">来自 GitHub 的推文 - FixTweet/FxTwitter：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/bikash119/mastering_llm">GitHub - bikash119/mastering_llm</a>：通过在 GitHub 上创建账号来为 bikash119/mastering_llm 的开发做出贡献。</li><li><a href="https://x.com/chroepke">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://calendly.com/christianroepke/introductory-call">30 分钟介绍性通话 - Christian Röpke</a>：预约与我的免费介绍性通话，分享你的挑战并了解我如何能为你提供帮助。</li><li><a href="https://github.com/huggingface/transformers/pull/30928">FEAT / Trainer：实验性功能 - 在将模型推送到 Hub 时添加 `GGUF` 转换，由 younesbelkada 提交 · Pull Request #30928 · huggingface/transformers</a>：此 PR 的作用？引入了一个新的 quantization_config，旨在仅用于 trainer.push_to_hub()，它在底层调用了一个 GGUF 转换 Space - (目前：https://huggingf...</li><li><a href="https://www.quora.com/Should-you-fine-tune-an-LLM-or-just-do-prompt-engineering/answer/Tong-Hui-Kang-1,">你应该微调 LLM，还是只做 Prompt Engineering？- Quora</a>：未找到描述</li><li><a href="https://youtu.be/Mn_9W1nCFLo?si=SWUPvbQ9ZCAxmAK_">LLaMA 详解：KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU</a>：全面解释来自 Meta 的 LLaMA 1 和 LLaMA 2 模型，包括 Rotary Positional Embeddings, RMS Normalization, Multi-Query Attention, KV-Cache, Grou...</li><li><a href="https://youtu.be/UiX8K-xBUpE?si=UgGM6oimKVhvub-b">Mistral / Mixtral 详解：Sliding Window Attention, Sparse Mixture of Experts, Rolling Buffer</a>：在本视频中，我将介绍 Mistral 7B 和 Mixtral 8x7B 模型中的所有创新：Sliding Window Attention, 带有 Rolling Buffer 的 KV-Cache, Pre...</li><li><a href="https://youtu.be/bCz4OMemCcA?si=X5lnwL_cmE16XFFS">Attention is all you need (Transformer) - 模型详解（包括数学）、推理和训练</a>：完整解释 Transformer 模型的所有层：Multi-Head Self-Attention, Positional Encoding，包括所有的矩阵乘法和...</li><li><a href="https://github.com/leloykun">leloykun - 概览</a>：Machine Learning (AI) 研究工程师 @expedock • 2x IOI &amp; 2x ICPC 世界总决赛入围者 • Math @ AdMU - leloykun</li><li><a href="https://leloykun.github.io/">Franz Louis Cesista</a>：数学家 | Machine Learning (AI) 研究科学家</li><li><a href="https://calendly.com/leloy/chat-with-franz">与 Franz 聊天 - Franz Louis Cesista</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1245600306302287913)** (5 条消息): 

- **研讨会幻灯片已分享**：Workshop 1 的幻灯片现在可以在 [Google presentation](https://docs.google.com/presentation/d/1hcfR4ZhAMmzFFiXJi_O3WE9fojOmHZbjSoG7DcJivso/edit#slide=id.g1ec9867125_0_0) 中查看。该文档设置为只读模式。

- **方言识别的微调**：讨论了让 LLM 理解并区分不同芬兰瑞典语（Fenni-Swedish）方言的最佳方法。建议使用了解瑞典语的模型进行微调，并为每种方言添加标签，可能将其视为独立的语言。

- **调试模型并使用 Axolotl 进行微调**：一位用户寻求关于调试模型的建议，并确认了 chat template 的功能。他们提到尝试微调 Qwen 模型，但遇到了训练开始后随即停止的问题，并分享了他们的 [notebook](#)。

- **自然语言转 SQL 的可扩展性**：一位用户讨论了针对大量表从自然语言生成准确 SQL 查询的挑战。他们正在寻求过滤相关表并向 LLM 添加知识的有效方法，分享了当前的方法以及在微调过程中遇到的过拟合（overfitting）和幻觉（hallucinations）问题。

- **使用知识图谱处理表关系**：建议构建关于表关系的知识图谱，并利用它来识别用于查询生成的各种相关表。这种方法可以帮助过滤表，并为 LLM 提供必要的上下文。

**提到的链接**：<a href="https://docs.google.com/presentation/d/1hcfR4ZhAMmzFFiXJi_O3WE9fojOmHZbjSoG7DcJivso/edit#slide=id.g1ec9867125_0_0">fine-tuning workshop 1 slides</a>：面向数据科学家和软件工程师的 LLM 微调

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1245512391899349103)** (2 条消息): 

```html
- **来自悉尼的新成员加入团队**：一位新成员介绍了自己，提到他们是常驻澳大利亚悉尼的高级分析（Advanced Analytics）部门的高级经理。他们表示有兴趣将微调应用于特定用例，并使用最少的 prompting 部署 LLM，同时学习在生产环境中托管和部署 LLM 的最佳实践。

- **全球 AI 黑客松预警**：宣布了即将于 6 月 7 日至 9 日举行的 **全球 AI 黑客松（Global AI Hackathon）**，活动将在新加坡、悉尼和旧金山等多个城市举行。鼓励参与者通过 [此链接](https://lu.ma/igqisb0e) 进行 RSVP，并指出该黑客松得到了顶尖 AI 构建者的支持，旨在解决“AI 让世界更美好”的问题。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/igqisb0e">Singapore | Build Together: AI Hackathon · Luma</a>：注意：此活动名额有限，请 RSVP 以确保您的席位。请仅在您可以全程参加活动的情况下进行 RSVP。快来与顶尖 AI 黑客一起构建和交流吧……</li><li><a href="https://www.buildclub.ai/events/build-together">Build Club</a>：Build Club 是全球 AI 创始人联系并发布初创项目的最佳场所。这是一个面向顶尖构建者的 100% 免费社区。快来加入我们！
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1245461312679706644)** (74 条消息🔥🔥): 

- **Modal 任务执行故障排除**：多位用户在 Modal 上运行训练任务时遇到问题，包括报错以及对数据集路径和 config 设置的困惑。共识是确保 config 文件中的数据集路径与预期位置匹配，并正确使用 Modal 的云存储。
- **WANDB 集成问题**：用户在使 WANDB 集成正常工作时遇到困难，建议在训练运行前重命名 secrets 并设置环境变量，如 `ALLOW_WANDB=true`。*“你的 secret 必须重命名。你必须删除它，并将其从 'my-wandb-secret' 更改为 'wandb'”*。
- **澄清配置和 Secrets**：用户分享了他们的配置文件，并讨论了路径和 secret 名称的正确设置，包括确保 `wandb_watch` 配置正确。 
- **有用的参考资料和示例**：引导用户参考 [Modal 文档](https://modal.com/docs/guide/trigger-deployed-functions) 和示例仓库，以更好地理解如何在 Modal 上部署和调用函数。**“我建议从 Modal 的 hello world 示例开始 (https://modal.com/docs/examples/hello_world)”**。
- **App 功能和后续步骤**：在成功部署模型后，用户讨论了如何继续，包括使用 Modal 平台进行进一步实验以及部署模型的实际应用。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/llm-finetuning/blob/main/README.md">llm-finetuning/README.md at main · modal-labs/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/5eb7c4b3034ce67e177e9924ef6642c2cad4bc17/config/llama-3.yml#L7C1-L10">llm-finetuning/config/llama-3.yml at 5eb7c4b3034ce67e177e9924ef6642c2cad4bc17 · modal-labs/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main">GitHub - modal-labs/llm-finetuning: Llama/Mistral/CodeLlama 等模型的微调指南</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://wandb.ai/hongnangao/golden-gate-bridge-repeng/runs/6fo15ch8/overview?nw=nwuserhongnangao">hongnangao</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/hongnangao/golden-gate-bridge-repeng/runs/5k4or992?nw=nwuserhongnangao">hongnangao</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/5eb7c4b3034ce67e177e9924ef6642c2cad4bc17/src/train.py#L150-L154">llm-finetuning/src/train.py at 5eb7c4b3034ce67e177e9924ef6642c2cad4bc17 · modal-labs/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的微调指南 - modal-labs/llm-finetuning</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">调用已部署的函数</a>: Modal 允许你获取由部署创建的函数并从其他上下文中调用它。</li><li><a href="https://modal.com/apps">登录</a>: 欢迎回到 Modal！通过在下方选择身份提供商登录你的 Modal 账户。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1245600288614649889)** (5 messages): 

- **Meta 发布关于 vLLM 的论文**：一名成员分享了 **AI at Meta** 发布的一篇与 vLLM 相关的论文。点击[此处](https://arxiv.org/abs/2405.17247)查看论文。

- **LLM 资源 GitHub 仓库**：一名成员开始建立一个收集 LLM 资源的 GitHub 仓库，对参加 "Mastering LLMs" 工作坊的人员非常有用。点击[此处](https://github.com/marco-jeffrey/awesome-llm-resources)访问并贡献该仓库。

- **扩展 LLama3 上下文窗口**：分享了一篇讨论如何将 **LLama3 的上下文窗口**从 8K 扩展到 80K 的论文。包括数据、模型、数据生成流水线和训练代码在内的完整资源集将在此处公开[发布](https://arxiv.org/abs/2404.19553)。

- **AI 编程 Humble Bundle**：目前有一个针对 AI 编程和 Prompt Engineering 的 **Humble Bundle** 优惠，社区成员可能会感兴趣。更多详情和购买选项请见[此处](https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.17247">An Introduction to Vision-Language Modeling</a>：随着近期大语言模型 (LLMs) 的普及，人们尝试将其扩展到视觉领域。从拥有一个可以引导我们穿越陌生环境的视觉助手到...</li><li><a href="https://arxiv.org/abs/2404.19553">Extending Llama-3&#39;s Context Ten-Fold Overnight</a>：我们通过 QLoRA 微调将 Llama-3-8B-Instruct 的上下文长度从 8K 扩展到 80K。整个训练周期非常高效，在单台 8xA800 (80G) GPU 机器上仅需 8 小时。结果...</li><li><a href="https://github.com/marco-jeffrey/awesome-llm-resources">GitHub - marco-jeffrey/awesome-llm-resources: a collection of resources around LLMs, aggregated for the workshop &quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&quot; by Dan Becker and Hamel Husain&quot;</a>：LLM 相关资源的集合，为 Dan Becker 和 Hamel Husain 的 &quot;Mastering LLMs: End-to-End Fine-Tuning and Deployment&quot; 工作坊汇总而成。</li><li><a href="https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email">The Complete ChatGPT, Anthropic, Gemini Prompt Engineering, API, and Programming Mega Bundle</a>：AI 正在崛起——通过这些在线课程与它一同成长！学习 Prompt Engineering、LangChain 等！您的购买将助力 Children’s Miracle Network。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1245722980898701354)** (2 messages): 

- **在官方 Axolotl Docker 上运行**：*我们正在官方 Axolotl Docker 上运行。它每天构建一次。* 如果你分享具体的配置和一些样本数据集，我们可以从我们这边尝试一下。

- **分享用于调试的配置和命令**：[这是配置](https://discord.com/channels/1238365980128706560/1242542198008975430/1245437637624467569) 以及 [这是发出的命令](https://discord.com/channels/1238365980128706560/1242542198008975430/1245437637624467569)。tokenizer 可能存在问题，但尚未进行调试。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1245481968758423582)** (2 messages): 

- **账单设置是获取 Replicate 额度的前提**：一名成员询问是否需要设置账单信息才有资格获得 Replicate 额度。另一名成员确认了这一要求，并建议设置较低的每月限额以避免产生意外费用。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245844758509912105)** (1 messages): 

- **Langsmith HIPAA 合规性咨询**：一位用户询问 Langsmith 是否提供允许在 **符合 HIPAA 标准的框架**上部署的付费方案。该用例涉及处理 PII/PHI，需要供应商作为 Business Associate 并签署 DPA。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1245453864476348497)** (2 messages): 

- **有趣的 Text2SQL 方法综述**：一位成员分享了一篇关于不同 Text2SQL 方法的[有趣综述](https://arxiv.org/pdf/2403.02951)。另一位用户表示感谢，并指出该资源非常有用。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1245463062274637915)** (16 messages🔥): 

- **Copilot Chat 通过 @workspace 成为领域专家**：一位用户分享了一个[链接](https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace)，解释了在 Copilot Chat 中引用 `@workspace` 如何使其能够智能地检索相关文件和符号。示例包括查找数据库字符串的配置位置或在代码库中验证日期。

- **终端中的 Copilot 内联聊天**：有人指出 Copilot 可以直接在终端中调用，这是一个大多数人尚未意识到的功能，增强了其在编辑器之外的可用性。

- **Copilot 与 Cursor 之争**：用户对比了 Copilot 和 Cursor，称赞 Copilot 拥有更好的结果和解决方案。然而，Cursor 注入自定义模型（如 GPT-4）的能力及其可定制环境被强调为显著优势。

- **用于 Function Calling 的 JSON Schema 和 Zod**：一位用户分享了 [JSON Schema 信息](https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f)以及来自 OpenAI [GitHub](https://github.com/openai/openai-node/blob/master/examples/tool-call-helpers-zod.ts) 的 Zod 使用示例，尽管指出某些示例已过时。他们提到使用 Deno 作为 Jupyter notebook 内核可以很好地运行这些示例，并承诺很快发布更多细节。

- **在 Twitter 上分享的 Document Mimicry 理解**：一位用户感谢另一位的见解，并分享了一篇解释 Document Mimicry 的 [Twitter 帖子](https://twitter.com/nehiljain/status/1795949311135502443)。该用户发现带着 Document Mimicry 的思路进行 Prompting 非常有益。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace">使用 @workspace 上下文引用进行聊天</a>：如何使用 Copilot 的 @workspace 聊天功能针对整个代码库提问。</li><li><a href="https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f?pvs=4,">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一体化工作区。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1245456482623815781)** (6 messages): 

- **通过重新安装包解决错误**：成员们讨论了通过重新安装包来解决错误。建议包括使用 `pip install -e` 和来自 [Jarvis Labs](https://jarvislabs.ai) 的模板。

- **使用 Jarvis 模板获得成功**：一位成员确认他们使用 Jarvis Labs 提供的模板成功解决了错误。这对其他遇到类似问题的成员很有帮助。

- **Workshop 2 幻灯片已发布**：Workshop 2 的幻灯片可在 Google Docs 上查看。你可以在[这里](https://docs.google.com/presentation/d/1otXeE6D5kJiDuxFYk3t9Nq9pKesN4-_6YhgLGRXmSU4/edit#slide=id.g1ec9867125_0_0)查看。

**提到的链接**：<a href="https://docs.google.com/presentation/d/1otXeE6D5kJiDuxFYk3t9Nq9pKesN4-_6YhgLGRXmSU4/edit#slide=id.g1ec9867125_0_0">Fine-tuning workshop 2 幻灯片</a>：掌握 LLM：开发者与数据科学家大会

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1245456201567961138)** (18 messages🔥): 

- **FIB 论文基准测试突显了 LLM 的事实准确性**：[FIB 论文](https://arxiv.org/abs/2211.08412)专注于衡量 LLM 在摘要任务中的事实一致性，结果显示像 BLOOM 这样的模型会对事实一致的摘要给出更高评分，但在逐字一致性（verbatim consistency）方面表现不佳。另一篇[在此链接](https://arxiv.org/abs/2305.11747)的论文则质疑了 LLM 识别不一致摘要的有效性。
  
- **使用干净数据进行微调可减少幻觉**：如这篇 [推文线程](https://x.com/stefanhgm/status/1765466556216053879) 所强调的，当使用干净的训练数据时，通过微调来减少幻觉是有效的。然而，[研究表明](https://arxiv.org/abs/2405.05904)使用新知识进行微调可能会增加幻觉，尤其是在闭卷问答（closed-book QA）等任务中。

- **简化 LLM 的评估流程**：讨论强调了增强 Text-to-SQL 等任务评估流程的重要性，建议使用 L1（针对语法的单元测试和断言）和 L2（针对相关性的人工反馈）。利用执行结果和基于 Schema 的模糊搜索可以验证正确性，而性能评估则需要更细致的检查。

- **用于 LLM 评分的评估库**：推荐的 LLM 评估评分工具包括 [Hugging Face 的 Evaluate 库](https://huggingface.co/docs/evaluate/en/index) 和 Braintrust 的 [Autoevals](https://github.com/braintrustdata/autoevals)，它们为 NLP、AI 模型等提供了多种评估方法。这些工具旨在通过最佳实践和可复现性来简化流程。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/stefanhgm/status/1765466556216053879">来自 Stefan Hegselmann (@stefanhgm) 的推文</a>：在训练或提示数据中删除不支持的事实是否能有效减少幻觉？我们针对 GPT-4 和 Llama 2 在生成患者摘要方面进行了测试。与 @shannonzshen, Florian Gie... 合作。</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">有效与无效的特定任务 LLM 评估</a>：涵盖分类、摘要、翻译、版权重复和毒性方面的评估。</li><li><a href="https://huggingface.co/docs/evaluate/en/index">🤗 Evaluate</a>：未找到描述</li><li><a href="https://github.com/braintrustdata/autoevals">GitHub - braintrustdata/autoevals: AutoEvals 是一个使用最佳实践快速轻松评估 AI 模型输出的工具。</a>：AutoEvals 是一个使用最佳实践快速轻松评估 AI 模型输出的工具。 - braintrustdata/autoevals</li><li><a href="https://arxiv.org/abs/2405.05904">在新知识上微调 LLM 是否会诱发幻觉？</a>：当大语言模型通过监督微调进行对齐时，它们可能会遇到预训练期间未获得的新事实信息。通常推测这可能会教会模型...</li><li><a href="https://arxiv.org/abs/2211.08412">通过新闻摘要评估大语言模型的事实一致性</a>：虽然大语言模型 (LLM) 已被证明在多种任务中非常有效，但它们也以产生幻觉信息而闻名。为了衡量 LLM 是否更倾向于事实一致的内容...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1245763983894515812)** (1 messages): 

- **微调 Gradio 文档需要明确性**：一位成员表示有兴趣协助 Gradio 文档微调项目，特别是专注于创建微调输入/输出数据集。他们建议从代码块中生成以用户为中心的问题，旨在产生细粒度、可操作的响应，并询问了有助于此转换过程的模板。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1245475686127370261)** (56 messages🔥🔥): 

- **Axolotl README 示例停滞**：一位用户报告运行 `axolotl` README 示例（`accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml`）时，因 GPU 显存占满而卡住。随后讨论了关于评估、磁盘读写利用率以及多 GPU 配置相关的潜在挑战。

- **OpenLLaMA-3B 示例在单 GPU 上成功运行**：在尝试多种配置后，用户发现 `openllama-3b` 示例在单 GPU 上可以运行，这表明多 GPU 设置可能存在问题。他们分享了配置，并指出更改为使用 `bf16` 并启用了 `tf32`。

- **WSL2 上的 NCCL 问题**：另一位用户寻求在 WSL2 上安装 NCCL 的建议，但面临多个错误。建议切换到 Linux 和 Docker 以获得更稳定的环境，一些用户分享了他们的经验并建议使用 `ddp_backend: gloo` 等替代配置。

- **Prompt 模板配置**：一位成员询问在特定训练任务的配置中使用标准模板还是自定义模板，特别是针对具有特定列的数据集的 function calling 任务。讨论鼓励分享在配置中使用模板的最佳实践。

- **GPU 性能结果**：分享了在不同 GPU 上运行 CodeLlama 7B 的结果，显示了每个 epoch 训练时间的显著差异。该成员指出这些发现是为了澄清与 README 中记录的时间差异。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1245524869127471115)** (35 messages🔥): 

- **FSDP 促进数据并行**：一位成员分享了一个 GitHub issue 链接，关于 [FSDP 实现 DDP, ZeRO-1, ZeRO-2 和 FSDP 模式之间的无缝切换](https://github.com/pytorch/pytorch/issues/102532)，并指出 DeepSpeed 较难使用，而 FSDP 对 LLM 来说更易用。
- **hf+accelerate 的推理问题**：一位成员报告在运行 meta-llama 示例时出现“混合字母”（乱码），怀疑是 `device_map="auto"` 的问题。他们提供了代码片段作为上下文，并收到了在标记 Accelerate 团队之前分享发现的建议。
- **Prompt：社区故障排除**：故障排除线程进行了反复的代码分享和建议，引导用户在问题持续存在时开启 GitHub issue。目的是帮助调试并加快未来类似问题的解决。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/openai-community/gpt2">openai-community/gpt2 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/102532">FSDP enable users to seamlessly switch between DDP, ZeRO-1, ZeRO-2 and FSDP flavors of data parallelism · Issue #102532 · pytorch/pytorch</a>: 🚀 功能、动机和宣传 https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/ DeepSpeed 难以使用，FSDP 对 LLM 来说易于使用，但 FSDP 不支持 ZeRO.....</li><li><a href="https://github.com/huggingface/accelerate/issues">Issues · huggingface/accelerate</a>: 🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持.....
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1245478967834710017)** (6 messages): 

- **QwenCode 模型问题已解决**：一位成员通过本地下载模型并将 tokenizer 修改为 **Qwen2Tokenizer** 解决了问题，并指出“*一切运行正常*”。然而，他们强调了 **QwenCode 模型上传** 的问题，目前仍在等待 Qwen 团队的回复。

- **Axolotl 配置的合理性检查**：分享了用于模型量化和设置 CUDA 选项的详细配置，指定了 **8-bit quantization** 和各种精度设置。另一位成员确认了关于模型权重、dtype 设置和 AMP 支持的解释是正确的。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1245465600734658610)** (7 条消息): 

- **Gradio 在直观性上胜过 Streamlit**：一位成员分享道，他们发现 **Gradio 比 Streamlit 直观得多**，尤其是在制作 Demo 时（*“我直接选择了 Gradio”*）。
- **OAuth 安全问题得到解决**：一位用户对 `gr.OAuthProfile` 的安全性提出了担忧，但得到的澄清是：“*OAuth 不存在该漏洞*”，且与仅在 Header 中添加用户数据相比，OAuth 是更安全的选择。详细的使用和共享技术记录在 [Gradio Guide](https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers) 中。
- **Gradio vs. Streamlit 详细对比**：据一位成员介绍，Gradio 能够精细地跟踪依赖关系，不会像 Streamlit 那样重新渲染所有内容。Gradio 还可以在各种 Python 环境中运行，包括 Jupyter notebooks，并提供诸如队列系统（queueing system）之类的后端功能。

**提到的链接**：<a href="https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers">Sharing Your App</a>：Gradio 分步教程

---

### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245775449376817222)** (86 条消息🔥🔥): 

- **Modal 重视多 GPU 训练**：Modal 团队一直在积极致力于加固多 GPU 设置，但由于安全 Hypervisor 比预期更严格，目前仍偶尔面临一些问题。鼓励用户通过[此处](https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1716911367749919)的专用 Slack 线程关注进度。

- **GPU 推理的冷启动瓶颈**：由于必须将权重从磁盘传输到 GPU VRAM，LLM 或 Stable Diffusion 推理的真正“冷”启动将耗时数秒。有关减轻这些延迟问题的详细解决方案和优化讨论，请参阅[此处](https://modal.com/docs/guide/cold-start#cold-start-performance)。

- **高效处理模型权重**：管理大型模型权重的最佳实践对于优化 ML 应用程序的启动时间至关重要。Modal 提供了多种策略，例如在构建时将权重存储在容器镜像中或使用分布式文件系统，详见[此处](https://modal.com/docs/guide/model-weights)。

- **与本地服务的无缝集成**：Modal 允许本地 Python 代码与运行在 localhost 上的任何服务进行交互。有关部署服务以及使用 Modal 凭据从其他应用程序连接这些服务的详细信息，请参阅[此处](https://modal.com/docs/guide/trigger-deployed-functions)。

- **用于数据管理的分布式对象**：Modal 提供分布式字典（dicts）和队列（queues），以实现分布式系统各组件之间的高效交互和数据传输。了解有关这些对象的工作原理及其最佳使用实践的更多信息，请参阅[此处](https://modal.com/docs/guide/dicts-and-queues)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/learning-learnding-funny-the-simpsons-dumb-gif-5270072">Education Is Key GIF - Learning Learnding Funny - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/modal-labs/awesome-modal">GitHub - modal-labs/awesome-modal: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php.</a>：一个精选的、令人惊叹的 Modal 应用程序、演示和闪光点列表。灵感来自 awesome-php。- modal-labs/awesome-modal</li><li><a href="https://modal.com/docs/guide/dicts-and-queues">Dicts and queues</a>：Modal 提供各种分布式对象，以实现分布式系统不同组件之间的无缝交互和数据传输。两个关键对象是字典和队列，两者都...</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">Invoking deployed functions</a>：Modal 允许你获取由部署创建的函数，并从其他上下文中调用它。</li><li><a href="https://modal.com/docs/examples">Featured examples</a>：如何在 Modal 上运行 LLM、Stable Diffusion、数据密集型处理、计算机视觉、音频转录和其他任务。</li><li><a href="https://modal.com/docs/guide/model-weights">Storing model weights on Modal</a>：高效管理大型模型的权重对于优化 ML 和 AI 应用程序的构建时间和启动延迟至关重要。本页讨论了处理模型权重的最佳实践...</li><li><a href="https://modallabscommunity.slack.com/archives/C069RAH7X4M/p1716911367749919">Slack</a>：未找到描述</li><li><a href="https://modal.com/docs/guide/cold-start#cold-start-performance">Cold start performance</a>：Modal 函数在容器中运行。</li><li><a href="https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/">The State of AI Infrastructure at Scale 2024</a>：财富 1000 强公司如何应对 AI 对其基础设施不断增长的需求？他们能否足够快地部署生成式 AI，同时对其进行严格控制以提供出色的...
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245828732384448623)** (59 条消息🔥🔥): 

- **LangChain 框架概述**：一位用户通过分享 [LangChain 介绍页面](https://python.langchain.com/v0.2/docs/introduction/)，阐明了 LangChain、LangSmith 和 LangServe 等各种工具之间的区别，该页面解释了工具包及其组件。LangChain 提供开发和部署工具，而 LangSmith 则提供检查和优化功能。
- **LangFlow 和 LangGraph 的混淆**：讨论中提到 LangFlow 未在 LangChain 图表中出现，并澄清了 LangFlow 虽然使用了 LangChain 框架，但在技术栈和用途上与之无关。
- **深度理解资源**：用户分享了几个链接以增强理解和实际应用，包括一篇关于 [使用 LLM 构建应用的一年心得](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) 的 O'Reilly 文章、一个 [来自 Next.js 的生成式 UI GitHub 仓库](https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md)，以及一个关于 [LangChain LangGraph 的 GitHub 系列视频](https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg)。
- **LangServe 和 LangGraph 评论**：用户对 LangServe 及其功能表示了特别的赞赏，而另一位用户则称赞了 LangGraph 工作流的概念，称其为“天才之作”。此外，还讨论了 LangSmith 为满足合规性可能与欧洲服务器集成的问题。
- **社区参与和项目经验**：用户分享了使用 LangChain 的个人经验，强调了它在构建内部应用方面的实用性，并认可了它在高级别和详细实现中的灵活性。一些幽默和轻松的评论提到了掌握这些工具所需的复杂性和知识广度。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>：LangChain 是一个用于开发由大语言模型 (LLM) 驱动的应用的框架。</li><li><a href="https://www.answer.website/">answers, how they should be displayed.</a>：由 developers digest 构建的问答引擎。</li><li><a href="https://reflex.dev/">Reflex · Web apps in Pure Python</a>：未找到描述。</li><li><a href="https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md">langchain-nextjs-template/app/generative_ui/README.md at main · langchain-ai/langchain-nextjs-template</a>：LangChain + Next.js 入门模板。通过在 GitHub 上创建账户为 langchain-ai/langchain-nextjs-template 的开发做出贡献。</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>：未找到描述。</li><li><a href="https://tenor.com/view/woodstock-happy50th-anniversary-happy-gif-26217300">Woodstock Happy50th GIF - Woodstock Happy50th Anniversary - Discover &amp; Share GIFs</a>：点击查看 GIF。</li><li><a href="https://github.com/wandb/openui">GitHub - wandb/openui: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live.</a>：OpenUI 让你用想象力描述 UI，然后实时查看渲染效果。 - wandb/openui
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/)** (1 条消息): 

imaurer: Simon 的新闻通讯是一个很好的资源：
https://simonwillison.net/about/#subscribe
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1245462893164494878)** (93 条消息🔥🔥): 

```html
- **用于 Inspect 网站的 Quarto**：成员们讨论了在 [Inspect AI 网站](https://ukgovernmentbeis.github.io/inspect_ai/) 中使用 **Quarto** 的情况，一些人表示强烈赞同，“Quarto 是最好的”。
- **日志作为可复现性的单位**：Inspect AI 将日志作为可复现性单位的做法受到了几位成员的称赞。有人表示：“这感觉领先于时代（以一种非常好的方式）👀。”
- **Inspect AI 的链接和资源**：分享了多个重要链接，包括 [Inspect 主页](https://ukgovernmentbeis.github.io/inspect_ai/)、[AI Safety Institute](https://www.aisi.gov.uk/) 以及 [Inspect LLM workshop 仓库](https://github.com/jjallaire/inspect-llm-workshop)。
- **对 Inspect AI 的关注和反馈**：与会者讨论了关于 Inspect AI 的各个方面和建议，包括在 UI 中比较运行结果的功能以及对未来增强功能的想法。一位成员评论道：“Solvers 非常棒”，强调了该工具的灵活性和可组合性。
- **录制问题已解决**：最初在访问会议视频录制时存在一些问题，但随后这些问题得到了解决。修复后，一位成员确认道：“JJ 的录制现在对我来说可以正常播放了”。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ianww.com/llm-tools">LLM eval tools spreadsheet</a>: 包含 50 多个用于测试模型和改进提示词的 LLM 评估工具的电子表格。</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/">Inspect</a>: 用于大语言模型评估的开源框架</li><li><a href="https://tenor.com/view/frustrated-waaaaaaaa-wwe-angry-mad-gif-13112986">Frustrated Waaaaaaaa GIF - Frustrated Waaaaaaaa WWE - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846">Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/lets-go-lets-go-marvel-let%27s-go-thor-let%27s-go-lets-go-thor-gif-6938549561677021369">Lets Go Lets Go Marvel GIF - Lets go Lets go marvel Let&#039;s go thor - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/jjallaire/inspect-llm-workshop">GitHub - jjallaire/inspect-llm-workshop</a>: 通过在 GitHub 上创建账号来为 jjallaire/inspect-llm-workshop 的开发做出贡献。</li><li><a href="https://tenor.com/view/yes-gif-22712908">Yes GIF - Yes - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai">Inspect</a>: 用于大语言模型评估的开源框架</li><li><a href="https://github.com/ukgovernmentbeis/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai</li><li><a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai</li><li><a href="https://www.aisi.gov.uk/">The AI Safety Institute (AISI)</a>: AI Safety Institute 是科学、创新和技术部的一个部门，旨在促进严谨的研究，以实现先进的 AI 治理。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1245452585813409914)** (42 messages🔥): 

- **关于额度有效期的提问**：一位成员询问额度是否会在课程结束后消失。这表明了对访问连续性的担忧。

- **表单提交问题**：包括 voberoi 在内的一些用户报告称，在问题更新后，他们的表单中出现了答案缺失或空白的情况。Danbecker 安慰说底层数据并未丢失，尽管存在这些问题，提交的内容应该仍然有效。

- **OpenAI 账号混淆**：一位用户澄清了课程活动所需的 OpenAI 账号是否与登录 chatgpt 的账号相同，建议使用 platform.openai.com 进行账号登录。

- **Predibase 注册错误**：出现了一个问题，用户无法使用 Hotmail 地址注册，因为 Predibase 错误地显示了一条信息，称该服务仅限 Gmail 账号。该平台通常会限制来自某些消费级域名的账号。

- **额度表单截止日期和处理时间**：再次重申额度表单提交的截止日期为午夜，HuggingFace 和 Modal 等不同平台有特定的额度发放审核时间。由于直播课程的安排，Modal 的额度处理出现了轻微延迟。

**提到的链接**：<a href="https://x.com/hamelhusain/status/1795871985265946934?s=12">Hamel Husain (@HamelHusain) 的推文</a>：3,500 美元的算力额度将于今天截止。在 2024 年 5 月 29 日晚上 11:59 PST 之后，我们将无法再发放这些额度。引用 Eugene Yan (@eugeneyan) 的话：公益广告：LLM-conf + finetuning workshop 的报名即将截止...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1245489299550507132)** (5 messages): 

- **阿尔伯克基（Albuquerque）聚会计划**：
    - 有人提到他们住在墨西哥西北部，但经常去凤凰城探望朋友。他们正在寻找该地区有趣的活动或聚会。

- **洛杉矶欢迎西海岸成员**：
    - 一位来自洛杉矶的成员发出了友好的问候。

- **圣路易斯奥比斯波（SLO）午餐邀请**：
    - 一位来自圣路易斯奥比斯波（SLO）的成员邀请从旧金山沿 101 公路开车前往洛杉矶的人在 SLO 停留吃午餐。他们强调了该地区优秀的餐厅、当地酿酒厂和品酒机会。

- **旧金山 LLM 爱好者聚会**：
    - 一位成员转发了关于在旧金山 Mission 区的合作社举行的约 50 人聚会的详情，讨论 LLM evals。有意参加者请通过非匿名的社交账号私信（DM）他们以获取邀请。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1245452373472706652)** (16 messages🔥): 

- **密集的东海岸讨论**：成员们正在分享他们在东海岸的位置，范围从马里兰州和弗吉尼亚州到新泽西州和加拿大。现场充满了潜在聚会的氛围，出现了诸如“我们在中间地点见面吧”和“我上周刚在那儿徒步旅行！”之类的评论。

- **对纽约 AI Tinkerers 活动的期待**：在纽约举办的 [AI Tinkerers 活动](https://nyc.aitinkerers.org/p/live-from-civic-hall-nyc-tech-week-24-meetup) 引起了轰动。一位成员说：“我周一会去参加 AI Tinkerers 活动……联系我（HMU）”，其他人也表达了兴趣并进行了注册。

- **华盛顿特区（DC）聚会的可能性**：多位来自 DC 地区的成员表达了对当地聚会的兴趣。诸如“看来我们以后肯定应该办一次 DMV 聚会”和“看来我们需要在 DC 办一次聚会”之类的评论表明计划正在考虑中。

**提到的链接**：<a href="https://nyc.aitinkerers.org/p/live-from-civic-hall-nyc-tech-week-24-meetup">
来自 Civic Hall 的直播！AI Tinkerers 聚会 | NY#TechWeek [AI Tinkerers - 纽约市]
</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1245452950361604158)** (27 messages🔥): 

- **来自欧洲的问候**：来自欧洲各地的成员，包括法国、德国、芬兰、西班牙、荷兰和奥地利，互相介绍了自己并交换了问候。值得注意的是，人们还评论了过去在彼此国家生活的经历。
- **伦敦聚会热潮**：组织伦敦聚会的热情显而易见，几位来自英国的成员（包括一些从布里斯托尔赶来的成员）表达了在 6 月 5 日和 6 日聚会的兴趣。关于时间安排和细节的协调正在进行中。
- **巴黎计划**：一位成员询问是否有其他人在巴黎，得到的回复表示未来几周可能会有空。
- **图尔库（Turku）的夏天**：一位芬兰成员讨论了在图尔库享受 +25-27°C 的夏季天气，尽管由于引人入胜的课程和 Discord 活动恰好与如此美好的天气重合而感到有些纠结。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1245461244379402281)** (4 messages): 

- **保持公告频道的通知开启**：建议成员保持新 **announcements** 频道的通知开启。正如管理员所分享的，该频道对于任何关键更新和提醒都至关重要。
  
- **填写表格以获取额度**：多次提醒在截止日期前填写供应商额度申请表。分享了来自 [OpenAI](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f)、[Hugging Face](https://docs.google.com/forms/d/...)、[Modal](https://docs.google.com/forms/d/...) 和 [Fireworks](https://docs.google.com/forms/d/...) 等供应商的额度申请特定链接。

- **活动详情位于 Events 类别中**：活动日程和 Zoom 直播链接将发布在 "Events" 类别中。此部分还会根据个人时区显示剩余时间，以避免对活动时间的混淆。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>: 未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSc7U01uRlMd2jeeeLZtaePTul-xBZXBwRx3x8qD2iIpuqE_mg/viewform">Hugging Face 额度申请</a>: 在我们为您申请 🤗 HF 额度以使用 https://huggingface.co 的付费服务之前，我们需要了解一些简要信息！如有任何疑问，请联系 website@huggingface.co。 ...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfoCoXNhUjka09mu8rmgB1YM9s3529-F2oJdP5HkHT1SGfV2Q/viewform">Modal 黑客松额度</a>: 要领取您的 Modal 额度，请先在 https://modal.com/ 注册账号。然后，通过此表格告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些示例可以帮助您开始...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform">Fireworks 额度 - 精通 LLMs：开发者与数据科学家大会</a>: 请填写下方表格以获取 $250 的 Fireworks 额度！如有疑问/寻求帮助或需要更多额度，请加入我们的 Discord ;) https://discord.gg/fireworks
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

abhay_m: 👋
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1245459520663060532)** (3 messages): 

- **ChatGPT 免费用户获得升级**：所有 ChatGPT 免费用户现在可以访问 **browsing, vision, data analysis, file uploads 和 GPTs**。这是一项重大的功能增强，旨在扩展用户能力。

- **OpenAI 启动非营利组织计划**：OpenAI 推出了一项新计划 **OpenAI for Nonprofits**，旨在让非营利组织更容易使用其工具。更多详情可以访问 [这里](https://openai.com/index/introducing-openai-for-nonprofits/)。

- **打击 AI 的欺骗性用途**：OpenAI 讨论了旨在瓦解利用 AI 进行欺骗性秘密影响力行动的努力。阅读更多关于正在采取的策略和行动的信息，请点击 [这里](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations/)。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1245459824368680960)** (375 messages🔥🔥): 

```html
<ul>
    <li><strong>关于 GPT-4o 可用性的澄清：</strong> 多名成员询问了面向免费用户的 GPT-4o 可用性。据解释，免费用户无法强制获取访问权限，系统会根据其判断在 GPT-3.5 和 GPT-4o 之间自动切换。</li>
    <li><strong>对订阅价值的担忧：</strong> 一位用户对继续支付 ChatGPT 订阅费用表示困惑。回复强调了订阅者的优势，如早期访问新功能、配额以及订阅者专属的额外功能。</li>
    <li><strong>关于 AI 分析能力的讨论：</strong> 用户辩论了不同 AI 模型处理逻辑推理任务的能力，例如“apples test”和“susan test”。有观点指出，AI 模型经常表现出基于训练数据的偏见。</li>
    <li><strong>代码和模型使用见解：</strong> 成员们讨论了使用各种 AI 模型辅助编程，比较了 GPT-4o、Mistral 的 Codestral 和 Copilot 等工具的性能。速度和准确性被认为是选择特定模型的关键因素。</li>
    <li><strong>新闻和媒体检测 AI 构想：</strong> 一位用户讨论了一个通过评估社交媒体帖子来检测虚假新闻和宣传的 AI 概念。另一位用户建议，这可能会遇到 AI 解释中常见的幻觉（hallucination）和偏见等问题。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/they-did-that-on-the-simpsons-professor-chaos-butters-south-park-it-was-on-th">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/they-did-that-on-the-simpsons-professor-chaos-butters-south-park-it-was-on-the-simpsons-gif-22242623">They Did That On The Simpsons Professor Chaos GIF - They Did That On The Simpsons Professor Chaos Butters - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codestral.mistral.ai`">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/quickstart.ipynb">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1245460314313457787)** (64 messages🔥🔥): 

- **GPT-4 输出“词汇乱炖” (Word Salad)**: 多名用户报告了 **GPT-4 和 GPT-4o** 的问题，即长回复会退化为充满流行词的“词汇乱炖”。一位用户提供了一个详细示例，涉及将文本转换为 Pinyin（拼音），在初始连贯的文本之后，输出变得毫无意义。
- **有限制地免费访问 GPT Store**: 免费用户目前可以访问和浏览 **GPT Store**，但除了使用 **GPT-3.5** 之外，无法运行 GPTs。一位成员引用横幅提示澄清道：“GPTs 将在未来几周内向免费用户开放。敬请期待！”
- **自定义 GPTs 的优势与局限**: 用户讨论了创建自定义 GPTs 的优势，例如定义特定角色和能力，并确认只有 **Plus 订阅者** 才能创建 GPTs。Memory 功能在自定义 GPTs 中尚不可用，但未来可能会推出。
- **API 和使用问题**: 讨论内容包括对 API 访问和模型使用差异的挫败感，特别提到一些用户混淆了 Chat 和 Completions API。成员们还分享了通过使用代理后端服务器来 **保护 API keys** 的技巧。
- **使用 GPT 编程及稳定性问题**: 一位用户在浏览器中使用 GPT 处理长上下文问题时遇到延迟和进度缓慢。他们考虑切换到 API 以获得更好的稳定性，正如另一位用户建议的那样，API 通常能更好地处理长时间会话。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1245460184651005982)** (2 messages): 

- **开发者偏向自己的创作**: 一位成员幽默地提到自己对自己构建的工具有偏见。他们承认在对该工具进行好评时可能存在偏见。
- **提供的故障排除建议**: 另一位成员询问如何修复特定问题，同时建议了应使用的正确版本。他们建议：“任何低于 4 的版本，我认为你必须拆分为 2 个独立的请求。”
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1245460184651005982)** (2 messages): 

- **关于在较低版本中拆分请求的讨论**: 一位成员询问特定问题是否已修复，并澄清对于任何低于 4 的版本，请求必须拆分为两个独立的请求。这表明了针对 API 版本兼容性的持续故障排除和支持。
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245830528599592970)** (5 条消息): 

- **Everything-AI 支持 llama.cpp 和 Qdrant**：现在你可以通过社区成员开发的 [everything-ai](https://github.com/AstraBert/everything-ai) *与你的 PDF 进行对话*。 
- **Mistral 模型已量化**：感谢社区贡献者，Mistral 模型的[量化版本](https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF) *Codestral-22B-v0.1-GGUF* 现已发布。
- **Nvidia 的 Embedding 模型 Demo 发布**：查看由另一位成员开发的全新 [Nvidia-Embed-V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) Demo。
- **新的 Image Gen Pro 工具**：一位社区成员在 HuggingFace 上发布了 [Image Gen Pro](https://huggingface.co/spaces/KingNish/Image-Gen-Pro)。
- **DuckDB 集成 HuggingFace Datasets**：DuckDB 为超过 150,000 个数据集添加了 `hf://` 路径，使集成变得[比以往任何时候都更容易](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c2db89519e4d)。

**提到的链接**：<a href="https://huggingface.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1245452119314792518)** (362 条消息🔥🔥): 

- **在不使用 Git 的情况下撤销 HuggingFace 上的 Commit**：一位用户不小心向 HuggingFace 的主仓库提交了代码，并询问如何撤销。讨论转向使用 HuggingFace CLI/库而非 Git，最终的解决方案是重新进行提交。
- **训练问题与 Learning Rate 讨论**：由于需要更改 Learning Rate，用户对重新训练模型进行了广泛讨论。建议包括使用 1e-3 或 1e-4 等数值以避免 catastrophic forgetting，并特别提到了 TinyLlama 1.1B 等模型。
- **音频处理与音高检测**：用户探讨了分析音频文件的音调、音高和语调的复杂性，并参考了数学解决方案以及 [CREPE Pitch Tracker](https://pypi.org/project/crepe/) 等工具。
- **Model Merging 竞赛公告**：发布了一项关于 NeurIPS 的 Model Merging 竞赛的公益公告，邀请参与者报名并角逐 8000 美元的奖金。竞赛详情可以在[这里](https://llm-merging.github.io/)找到。
- **Fine-Tuning Mistral 与 Tokenization**：用户讨论了 Fine-Tuning Mistral 和 TinyLlama 等模型时合适的 Tokenization 格式，并提供了将数据预处理为所需 Prompt 格式的脚本和示例。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/crepe/">crepe</a>: CREPE pitch tracker</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.4">TinyLlama/TinyLlama-1.1B-Chat-v0.4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/tokenizer_config.json">tokenizer_config.json · TinyLlama/TinyLlama-1.1B-Chat-v1.0 at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636">I Saw W Gus Fring GIF - I Saw W Gus Fring Gus - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/no-pixels-cant-see-ben-chang-community-ken-jeong-gif-17361588">No Pixels Cant See GIF - No Pixels Cant See Ben Chang - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=4KalMNIbRUM">VOXTA APARTMENT - DEMO GAMEPLAY</a>: 等待结束了！🎉 走进 Voxta Apartment，您的交互式 AI 伴侣 Anna 将欢迎您享受她的迷人陪伴和众多活动...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 NeurIPSConf 的 Model Merging 竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLM！🧠✨ 💻为了科学而来 💰为了 $8K 而留 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://github.com/pytorch/xla/">GitHub - pytorch/xla: Enabling PyTorch on XLA Devices (e.g. Google TPU)</a>: 在 XLA 设备（如 Google TPU）上启用 PyTorch。通过在 GitHub 上创建账号来为 pytorch/xla 的开发做出贡献。</li><li><a href="https://tenor.com/view/ok-ok-and-okay-buddy-dont-care-didnt-ask-gif-25239605">Ok Ok And GIF - Ok Ok And Okay Buddy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/openai/whisper">GitHub - openai/whisper: Robust Speech Recognition via Large-Scale Weak Supervision</a>: 通过大规模弱监督实现鲁棒的语音识别 - openai/whisper</li><li><a href="https://storage.googleapis.com/libtpu-releases/index.html">no title found</a>: 未找到描述</li><li><a href="https://pytorch.org/xla/release/2.3/index.html#quickstart>">PyTorch on XLA Devices &mdash; PyTorch/XLA master documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

venatic007: ✋🏻
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1245453587425792011)** (12 条消息🔥): 

- **GNNs 简化模拟器状态 embedding**: 一位成员解释了将 **GNNs** 用于图结构数据的方法，指出实体间的关系被编码为 **edge attributes**，而特征则作为每个实体的 **tensors** 存储。这种方法“以类似于简单 **2D convolution layers** 的方式合并所有这些内容，从而为你提供 **simulator state embedding**”。

- **Yuan2.0-M32 在 Math 任务中表现出色**: 拥有 40B 参数和全新 **router architecture** 的新模型 **Yuan2.0-M32**，在 **Math/ARC** 任务中超越了 **Llama 3 70B**。该模型已在 [X 上介绍](https://x.com/osanseviero/status/1796082193044844590)，并可在 [Hugging Face](https://huggingface.co/IEITYuan/Yuan2-M32-hf) 上获取，同时附带了 [研究论文](https://hf.co/papers/2405.17976)。

- **关于 Backpropagation 算法的视频**: 一段名为 [“**Machine Learning** 中最重要的算法”](https://www.youtube.com/watch?v=SmZmBKc7Lrs) 的 **YouTube** 视频解释了 **Backpropagation** 算法在推动 **Machine Learning** 领域发展中的重要性。

- **NeurIPS Model Merging 竞赛**: **NeurIPS** 的 **Model Merging** 竞赛公告发布，为有效的 **model selection** 和 **merging** 提供 8000 美元的奖金。详细信息和报名信息已在 [Twitter](https://x.com/LChoshen/status/1796256513519989102) 上分享。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1796082193044844590">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 介绍 Yuan2.0-M32 🔥 具有 3.7B active params（总计 40B）的 MoE 👀 全新 router architecture 🚀 在 2T tokens 上训练 🏆 考虑到 active params 的数量，metrics 令人印象深刻 🤯 在 Math 方面优于 Llama 3 70B...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>: 🚨 NeurIPS 上的 Model Merging 竞赛！🚀 你能彻底改变 model selection 和 merging 吗？让我们创建最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元留下 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://www.youtube.com/watch?v=SmZmBKc7Lrs">Machine Learning 中最重要的算法</a>: Shortform 链接: https://shortform.com/artem 在这段视频中，我们将讨论 Backpropagation —— 一种驱动整个 Machine Learning 领域的算法，并且 ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1245456344182554725)** (8 条消息🔥): 

```html
- **Demo Nvidia's embedding model**: A member shared a demo for Nvidia's new embedding model and requested PRs for cool examples or improved functions. *"You can test it out here: [Nvidia Embed V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1/)."*
- **Llama 3 SOLAR recreation attempt**: A user attempted to recreate Upstage's old Solar models using Llama 3. They used datasets like **`llm-wizard/alpaca-gpt4-data`** and [shared the model on HuggingFace](https://huggingface.co/cookinai/Llama-3-SOLAR-v0.2).
- **Codestral-22B quantized version**: Shared a quantized version of Codestral-22B-v0.1, created using llama.cpp, beneficial for code-related tasks. *"More details in the [Blogpost](https://mistral.ai/news/codestral/)."*
- **DuckDB supports Hugging Face datasets on WrenAI**: Announcement about DuckDB supporting the `hf://` path, enabling easy loading and querying of Hugging Face datasets in WrenAI. Learn more [here](https://blog.getwren.ai/how-to-load-huggingface-datasets-into-duckdb-and-query-with-gpt-4o-c2db89519e4d).
- **LLMinator v1.0.3 releases new features**: LLMinator now supports websocket interaction, context-aware chatbots, model conversion, and customized LLM inference parameters. Check out the project on [GitHub](https://github.com/Aesthisia/LLMinator).
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/cookinai/Llama-3-SOLAR-v0.2">cookinai/Llama-3-SOLAR-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF">QuantFactory/Codestral-22B-v0.1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1/">Tonic 的 NV-Embed - Tonic 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog">Hugging Face – 博客</a>: 未找到描述</li><li><a href="https://huggingface.co/blog-explorers">blog-explorers (Blog-explorers)</a>: 未找到描述</li><li><a href="https://github.com/Aesthisia/LLMinator">GitHub - Aesthisia/LLMinator: 基于 Gradio 的工具，可直接从 Hugging Face 运行开源 LLM 模型</a>: 基于 Gradio 的工具，可直接从 Hugging Face 运行开源 LLM 模型 - Aesthisia/LLMinator
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1245595801598689310)** (3 messages): 

- **关于无需参考模型的 PPO 查询引起了关注**：一位用户询问了在不使用参考模型的情况下使用 **PPO** (Proximal Policy Optimization) 的可能性，并提到他们的实习项目截止日期很紧。另一位成员建议查看一篇关于名为 **SimPO** 的替代方法的论文，该方法“消除了对参考模型的需求”，并提供了 [arXiv 链接以供进一步阅读](https://arxiv.org/abs/2405.14734)。
- **AI 模型中新的数学改进令成员们感到兴奋**：一篇链接到 [Hugging Face 上的论文](https://huggingface.co/papers/2405.17976) 因其在数学上的改进（特别是在强化学习领域）引起了热议。另一位成员分享了一篇题为“*Direct Preference Optimization*”的特定论文，并指出其中提到了 **SimPO**，这是一种比传统的基于参考模型的方法更高效的替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14734">SimPO: Simple Preference Optimization with a Reference-Free Reward</a>：Direct Preference Optimization (DPO) 是一种广泛使用的离线偏好优化算法，它通过对人类反馈强化学习 (RLHF) 中的奖励函数进行重新参数化，以增强...</li><li><a href="https://huggingface.co/papers/2405.17976">Paper page - Yuan 2.0-M32: Mixture of Experts with Attention Router</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1245465671241044091)** (17 条消息🔥): 

- **支持使用 ViT 进行图像回归**：用户讨论了可以通过 Hugging Face 的 `ViTForImageClassification` 并设置 `problem_type="regression"` 来处理图像回归任务。有关使用 Image 特性准备图像列数据集的说明可以在[这里](https://huggingface.co/docs/datasets/v2.3.2/en/image_process)找到。

- **提供微调演示 Notebook**：Niels Rogge 分享了用于微调任务的演示 Notebook，特别提到它适用于 Idefics2 和 PaliGemma 等模型。Notebook 可以在[这里](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma)获取，更多资源将很快通过 YouTube 视频分享。

- **使用 DINOv2 进行单目深度估计**：DINOv2（一种 ViT 模型）支持使用 DPT 头部进行单目深度估计任务。示例实现可以在[模型页面](https://huggingface.co/models?search=dpt%20dino)找到。

- **微调 Transformer 的最佳实践**：建议在微调 Transformer 模型时使用 AdamW 优化器和余弦学习率调度器（cosine learning rate scheduler）。建议包括使用 GPU 能容纳的最大 batch size，并利用 ConvNext、DINOv2 或 SigLIP 代替 ViT 以获得更好的性能。

- **NeurIPS 模型合并竞赛**：NeurIPS 宣布了一项由 Hugging Face 等赞助的模型合并竞赛。详情和报名信息可以在公告[推文](https://x.com/LChoshen/status/1796256513519989102)中找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/v2.3.2/en/image_process">处理图像数据</a>：未找到描述</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 NeurIPSConf 模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创建最好的 LLM！🧠✨ 💻 为科学而来 💰 为 8000 美元而留 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/do">Do (Tran)</a>：未找到描述</li><li><a href="https://huggingface.co/models?search=dpt%20dino">Models - Hugging Face</a>：未找到描述</li><li><a href="https://x.com/NielsRogge/status/1795106366752723094.">来自 Niels Rogge (@NielsRogge) 的推文</a>：事实证明我的 Idefics2 Notebook 同样适用于 PaliGemma 微调 :) 在这里找到它：https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma 对于 JSON 用例，一个微型 VLM ...</li><li><a href="https://github.com/google-research/tuning_playbook?tab=readme-ov-file#choosing-the-batch-size">GitHub - google-research/tuning_playbook: 系统地最大化深度学习模型性能的指南。</a>：系统地最大化深度学习模型性能的指南。 - google-research/tuning_playbook</li><li><a href="https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford">Papers with Code - Stanford Cars 基准测试（细粒度图像分类）</a>：目前 Stanford Cars 上的 SOTA 是 CMAL-Net。查看 73 篇带代码论文的完整比较。</li><li><a href="https://arxiv.org/abs/2211.12879">用于细粒度图像分类的数据增强 Vision Transformer</a>：最近，Vision Transformer (ViT) 在图像识别方面取得了突破。其自注意力机制 (MSA) 可以提取不同像素块的判别性标注信息，以提高...</li><li><a href="https://news.ycombinator.com/item?id=40505099">Llama 3-V：以 100 倍更小的模型和 500 美元达到 GPT4-V 的水平 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1245475582993633310)** (9 messages🔥): 

- **Whisper 中的词级时间戳**：一位用户询问如何使用 Whisper 模型获取词级时间戳，并分享了[文档链接](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.return_token_timestamps)。他们引用了论文 *“Robust Speech Recognition via Large-Scale Weak Supervision”*，并提到 Arthur Zucker 是贡献者。

- **CUDA device map 上的冲突**：一位用户在将 `device_map` 设置为 `'cuda'` 时遇到问题，并收到错误消息 "mode accelerated already used"。另一位用户分享了他们使用 Sentence Transformers 和 LLM 进行主题标注的成功经验，尽管他们并不理解那个技术问题。

- **自定义评估计划**：一位用户询问如何在特定步数（25k, 50k, 100k, 200k）设置自定义评估计划，这是基于训练性能随数据呈对数变化的认知模式。

- **开源模型发布**：一位用户兴奋地分享了两个完全开源的语言模型的发布，链接到了一个名为 [K2](https://huggingface.co/LLM360/K2) 的模型和另一个新的模型集合。K2 的亮点在于其性能超过了 Llama 2 70B 模型，且计算量减少了 35%。

- **NeurIPS 模型合并竞赛**：一位用户宣布了 NeurIPS 的模型合并竞赛，并附上了官方[公告推文](https://x.com/LChoshen/status/1796256513519989102)。该竞赛邀请参与者通过模型选择和合并带来革命性变化，奖金为 8000 美元。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/LLM360/K2">LLM360/K2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - a m-a-p Collection</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.return_token_timestamps">Whisper</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1245459761294606376)** (91 messages🔥🔥): 

```html
- **Codestral 模型发布及用途讨论**：发布了 **Codestral-22B-v0.1** 模型，支持包括 Python, Java 和 JavaScript 在内的 80 多种编程语言。该模型支持代码指令和 Fill in the Middle (FIM) 功能；[博客文章中有更多详情](https://mistral.ai/news/codestral/)。
- **对模型变体的担忧**：成员们讨论了不同量化变体的实用性，一些人指出 **_S 变体**通常太“笨（smoothbrained）”且没有用处。
- **代码模型和 Prompt 格式**：讨论了查询 Codestral-22B-v0.1-GGUF 的推荐格式，参考了[此 GitHub 链接](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF#prompt-format)。
- **有限硬件上的加载问题**：一位用户因系统配置较低在 **LM Studio** 上遇到了加载时间过长的问题，建议使用较小的模型可能会更好。
- **咨询业务联系方式**：一位成员询问项目的直接业务联系方式，并被引导发送邮件至 **team@lmstudio.ai** 进行进一步讨论。
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>: Empowering developers and democratising coding with Mistral AI.</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF#prompt-format">bartowski/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1">mistralai/Codestral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF">bartowski/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1245456795195932732)** (62 messages🔥🔥): 

- **模型上下文长度之苦**：成员们幽默地感叹了模型的上下文长度，指出像 **llama** 系列在不进行修改的情况下上限为 *4096*，但可以通过 **RoPE** 调整扩展到约 *16k*。一位成员开玩笑说：*"尺寸很重要 (Size matters)"*。

- **AlchemistCoder-DS-6.7B 微调讨论**：一位成员分享了 [HuggingFace 上的](https://huggingface.co/internlm/AlchemistCoder-DS-6.7B) **AlchemistCoder-DS-6.7B** 模型链接，其性能与 **Deepseek Coder 33B** 不相上下。文中还提供了将该模型转换为 GGUF 格式并配合 llama.cpp 使用以简化部署的说明。

- **Minerva-350M 兼容性困扰**：一位成员报告了自 **Minerva-350M** 发布以来在 LM Studio 中使用该模型的问题，面临生成困难和整体兼容性挑战。另一位成员建议先确保这些模型能在原生 llama.cpp 中运行，如果不行则提交功能请求 (feature request)。

- **高效角色扮演的挑战**：一位新用户发现使用 **Blue-Orchid-2x7b-Q4_K_M.gguf** 等模型难以实现有效的角色扮演，并寻求关于正确提示词和设置的指导，甚至分享了来自 Reddit 的详细角色扮演系统提示词。成员们建议测试不同的模型并调整设置。

- **GoLang 和 Kubernetes 的模型推荐**：一位新用户询问适用于 GoLang 和 Kubernetes 的模型，得到的推荐是 **Claude Haiku**，因其效率高且具备强大的上下文填充能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/internlm/AlchemistCoder-DS-6.7B">internlm/AlchemistCoder-DS-6.7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nakodanei/Blue-Orchid-2x7b_GGUF/tree/main">nakodanei/Blue-Orchid-2x7b_GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/failspy/Llama-3-8B-Instruct-MopeyMule">failspy/Llama-3-8B-Instruct-MopeyMule · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/models?sort=trending&search=uncensored">Models - Hugging Face</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1xrMwhrz4DIdwzY4gI3GIrxQ0phQjVNmu2RGKRnGnRAM/edit?usp=drivesdk">High Quality Story Writing Type First Person</a>：我的自定义 GPTs 的主 Google 文档：https://docs.google.com/document/d/1Cbwy3HuNTCzCaMXscU6FrgqvgjA2TFzOw1ucLqtbCyU/edit?usp=drivesdk。用于高质量故事的极度 NSFW 版系统提示词文本...</li><li><a href="https://huggingface.co/YorkieOH10/AlchemistCoder-DS-6.7B-Q8_0-GGUF">YorkieOH10/AlchemistCoder-DS-6.7B-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/YorkieOH10">YorkieOH10 (Yorkie)</a>：未找到描述</li><li><a href="https://huggingface.co/YorkieOH10/AlchemistCoder-DS-6.7B-Q4_K_M-GGUF">YorkieOH10/AlchemistCoder-DS-6.7B-Q4_K_M-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 messages): 

cancerous1: 感谢 ROCm/Windows 构建版本 🍻 你让我的模型运行空间翻倍了。
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/)** (1 messages): 

tiltspinner: 谢谢！
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1245501068448895086)** (2 messages): 

- **llama.cpp 不支持 Whisper 模型**：一位用户无法定位模型路径 *vonjack/whisper-large-v3-gguf/whisper-large-v3-q8_0.gguf*。另一位用户澄清说 **Whisper 模型** 在 **llama.cpp** 中不受支持，而是应该在 **whisper.cpp** 中使用。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1245698249772503071)** (207 messages🔥🔥): 

- **使用加密货币挖矿进行老化测试 (Burn-In Tests)**：一位用户建议，测试新 GPU VRAM 负载的一种方法是通过“显存密集型山寨币”进行加密货币挖矿。*“这些代币旨在抗 ASIC，可以在老化测试期间覆盖电力成本”*，尽管为了不推广加密货币而避免了具体细节。

- **NVIDIA 新 GPU 规格传闻**：传闻 NVIDIA GeForce RTX 5090 将采用缩减的 448-bit 总线接口和 28 GB GDDR7 显存。*“看起来他们正在使用缩减的核心，以便将完整核心留给专业级产品。”* [来源](https://wccftech.com/nvidia-geforce-rtx-5090-founders-edition-gpu-dual-slot-dual-fan-cooler/)。

- **关于高 RAM 和 GPU 推理的讨论**：成员们讨论了 CPU 与 GPU 推理的挑战，强调 **内存通道 (memory channels)** 比 CPU 核心数更重要。*“一旦超过了 VRAM，速度就会变慢，所以这可能是一个两害相权取其轻的问题，而不是一个理想化的问题。”*

- **购买用于推理的高端硬件**：成员们辩论了双路 EPYC 和 M 系列 Apple 等硬件的性价比。有人提到：*“在这个价位上，8x 3090 变得非常合理，这里二手只要 500 美元。”*

- **LLM 与幻觉 (Hallucinations)**：成员们表达了对 AI 生成信息的不信任，并讨论了减轻幻觉的技术。一位用户指出：*“LLM 非常擅长重写给定的文本，但在不出现错误的情况下从‘记忆’中提取信息却很困难。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wccftech.com/nvidia-geforce-rtx-5090-cut-down-gb202-gpu-448-bit-bus-28-gb-gddr7-memory/">NVIDIA GeForce RTX 5090 To Feature Cut-Down GB202 GPU With 448-Bit Bus &amp; 28 GB GDDR7 Memory</a>：传闻 NVIDIA GeForce RTX 5090 显卡将配备缩减的 448-bit 总线接口，显存高达 28 GB GDDR7。</li><li><a href="http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-5-plus.html">Orange Pi - Orangepi</a>：未找到描述</li><li><a href="https://sceniccitysummit.com/schedule/">Schedule - Scenic City Summit</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1245615615431934032)** (3 messages): 

- **添加第二个 GPU 进行推理加速**：用户正在讨论如何添加第二个 GPU 进行推理。一位用户幽默地描述了这一过程，解释说只要有物理空间和充足的电源，**LM Studio** 就会在两个 GPU 之间平衡负载，但强调需要管理 *tensor_split, CUDA_VISIBLE_DEVICES* 和 *main_gpu* 等设置。
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1245537914725732467)** (2 messages): 

- **Amuse 的 GitHub 链接失效**：一位成员指出 Amuse 的 GitHub 链接不再有效，且无法在提示词框中输入。
- **Hugging Face 托管 Amuse**：另一位成员提供了一个从 [Hugging Face](https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip) 下载 `Amuse_v1.3.0.zip` 的有效链接，并确认其运行正常。

**提到的链接**：<a href="https://huggingface.co/Stackyard-AI/Amuse/blob/main/Amuse_v1.3.0.zip">Amuse_v1.3.0.zip · Stackyard-AI/Amuse at main</a>：未找到描述

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1245469955491889192)** (1 messages): 

- **Codestral 22B 欢迎编程爱好者**：**Mistral 最新的编程模型** Codestral 现已开放下载。它具有 **22B 参数规模**，对于拥有高容量 GPU 并寻求强大模型的用户来说，这是一个极具吸引力的选择。[在此下载 Codestral-22B](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF)。

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1245456743295881356)** (233 messages🔥🔥): 

- **测试中 Llama3 压倒 Phi3**：成员们一致认为 Llama3 的表现优于 Phi3，其中一位表示“*Llama-3-8b 要好得多*”，另一位则称 Phi3 “*极其合成化（extremely synthetic）*”。共识是应避免使用 Phi3 模型（尤其是 mini 版本），转而选择 Llama3 基础模型。

- **创建角色扮演角色的微调建议**：建议的工作流包括在进行指令遵循（instruction following）微调之前，先对 Llama3 基础模型进行训练。一位用户分享了微调 Llama3 以模仿 Paul Graham 的经验，但发现其效果不如直接添加“*假设你是 Paul Graham*”之类的提示词。

- **关于在指令模型上进行微调的辩论**：通常建议不要在指令模型（instruction models）之上进行微调，因为这可能会产生损失，而不是为模型增加价值。对于耶稣、特朗普或其他特定角色的扮演，在基础模型（base models）上进行微调并使用 DPO 被认为是更好的选择。

- **关于使用 anti-prompts 的动态讨论**：成员们讨论了 anti-prompts 在更好地控制聊天模型对话流方面的效用。Anti-prompts 可以在用户定义的词汇处停止生成，允许用户在模型继续输出之前输入自己的文本。

- **新模型与优化查询**：对于新发布的模型（如 Yuan）存在兴奋情绪，但仍保持怀疑态度，强调实际应用比 Benchmark 更重要。用户分享了在各种工具和平台上进行模型微调和推理的经验与挑战，并表达了对 Unsloth 提供多 GPU 支持的渴望。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14734">SimPO: Simple Preference Optimization with a Reference-Free Reward</a>：Direct Preference Optimization (DPO) 是一种广泛使用的离线偏好优化算法，它重新参数化了人类反馈强化学习 (RLHF) 中的奖励函数，以增强...</li><li><a href="https://huggingface.co/REILX/Phi-3-medium-128k-code-instruct">REILX/Phi-3-medium-128k-code-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/gpuopenanalytics/pynvml/issues/53">nvmlDeviceGetName throws UnicodeDecodeError invalid start byte · Issue #53 · gpuopenanalytics/pynvml</a>：在 WSL2 上运行以下代码会抛出标题中提到的错误：from pynvml import * handle = nvmlDeviceGetHandleByIndex(0) print(nvmlDeviceGetName(handle)) Stacktrace: File &quot;&lt;stdi...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1245557277654777856)** (6 messages): 

- **关于 AI 训练偏见的网络研讨会**：加入 Tom Hosking 的网络研讨会，探讨 AI 训练中的人类反馈如何具有主观性和偏见，从而导致事实性等关键方面代表性不足。在此处观看 [研讨会](https://eu1.hubs.ly/H09npRg0)，并在 [Arxiv](https://arxiv.org/abs/2309.16349) 上阅读研究论文。

- **CoPE：上下文位置编码 (Contextual Positional Encoding)**：来自 FAIR 的一种针对 Transformer 的新位置编码方法——上下文位置编码 (CoPE)，它将上下文因素纳入其中以改进功能。该方法因其能够根据句子或段落等不同需求计算每个 Head 的距离而受到称赞。[查看推文](https://x.com/ylecun/status/1795985933998715217)。

- **代码共享中的个人边界**：一位成员表达了不愿公开分享代码的想法，表示希望保持私有。

- **狗狗欣赏**：一条轻松的消息强调了对成员们发布可爱狗狗照片的赞赏。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ylecun/status/1795985933998715217">Yann LeCun (@ylecun) 的推文</a>：CoPE：上下文位置编码。来自 FAIR 的一篇新论文，@elonmusk 可以用它来改进 Grok。引用 Jason Weston (@jaseweston)：🚨 上下文位置编码 (CoPE) 🚨 上下文至关重要！...</li><li><a href="https://eu1.hubs.ly/H09npRg0">Tom Hosking <> Prolific 网络研讨会 | 应对 AI 训练中人类反馈的偏见</a>：加入我们，参加这场富有洞察力的网络研讨会，探讨人类反馈在评估和训练大语言模型 (LLMs) 中的作用。了解偏好评分尽管是标准，但如何可能具有主观性...</li><li><a href="https://arxiv.org/abs/2309.16349">Human Feedback is not Gold Standard</a>：人类反馈已成为评估大语言模型性能的事实标准，并越来越多地被用作训练目标。然而，目前尚不清楚哪些属性...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1245451658369044572)** (130 条消息🔥🔥): 

- **Unsloth 支持原生微调并提供 Colab notebook**：一位成员提到 Unsloth 支持原生微调（native finetuning），并分享了一个指向 Colab notebook 的 [GitHub 链接](https://github.com/unslothai/unsloth#-finetune-for-free)，用于持续预训练（continuous pretraining）。
- **Command-R 模型偏好与 EOS Token 讨论**：一位成员表示 Command-R 是最好的模型，但另一位用户反驳称，训练数据中必须包含 EOS_TOKEN，模型才能知道文本补全何时结束。他们提供了一个 [YouTube 视频](https://www.youtube.com/watch?v=T1ps611iG1A) 作为示例。
- **Apple M3 GPU 与 CUDA 不兼容**：一位用户在搭载 Apple M3 Pro GPU 的 Mac 上遇到了 "Torch not compiled with CUDA enabled" 错误。解释指出 Apple 的 GPU 不支持 CUDA，建议改用 Google Colab。
- **Llama3-8b 模型性能讨论**：用户讨论了在不同硬件配置上运行 Llama3-8b 的情况，例如配备 16GB 内存的 Beelink Ser5 MAX 迷你电脑，并考虑升级到 32GB 或 64GB RAM。讨论强调，更大的 RAM 可以运行量化（quantization）更少的大型模型。
- **微调问题与数据集大小疑虑**：一位用户报告在微调 Llama3 时遇到了问题，导致出现“垃圾结果（garbage results）”，并询问是否需要更大的数据集。另一位成员回应称，使用特定数据进行训练可能会导致模型偏向该数据，而忽略之前的指令。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.eraser.io/diagramgpt">DiagramGPT – AI 图表生成器</a>：在几秒钟内从纯英语或代码片段提示生成技术图表。图表包括时序图、流程图、实体关系图、云架构图、数据流图...</li><li><a href="https://huggingface.co/chat/assistant/65e71408a6654bcc68624d8d">Diagrams Creator - HuggingChat</a>：在 HuggingChat 中使用 Diagrams Creator 助手</li><li><a href="https://www.youtube.com/watch?v=T1ps611iG1A">我如何为我的简报微调 Llama 3：完整指南</a>：在今天的视频中，我将分享如何利用我的简报来微调 Llama 3 模型，以便使用创新的开源工具更好地起草未来内容...</li><li><a href="https://diagrams.mingrammer.com/docs/getting-started/examples">示例 · Diagrams</a>：这里有更多示例。</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1245457683814027386)** (351 条消息🔥🔥): 

- **关于 Kohya SS 和预算受限下的模型训练讨论**：成员们讨论了在不产生高额成本的情况下**训练 Stable Diffusion** 模型的方案，重点介绍了 **Google Colab** 等工具和 **RunDiffusion** 等服务。一位成员分享了用于便捷训练的 Kohya Colab [GitHub 链接](https://github.com/hollowstrawberry/kohya-colab)。

- **ControlNet 与推理优化**：详细对话涉及了使用 ControlNet 和各种采样器来**提高图像生成准确性**。分享了一个**动态 LoRA 控制扩展**的 GitHub 链接：[sd-webui-loractl](https://github.com/cheald/sd-webui-loractl)。

- **适用于 Stability AI 的新 Ruby SDK**：一位成员宣布推出了用于图像生成的开源 **Stability AI API Ruby SDK**，支持核心模型和 SD3 模型。他们提供了 [GitHub 链接](https://github.com/OlympiaAI/stability) 供社区访问和贡献。

- **即将推出的模型与社区情绪**：社区推测了 **Stable Diffusion 3 (SD3)** 的发布日期和功能，表达了怀疑与希望并存的情绪。讨论内容包括潜在的许可挑战，以及与 Midjourney 等收入显著更高的竞争对手的财务支持对比。

- **向儿童教授 Stable Diffusion**：一位教师寻求关于在不生成显式内容的情况下**向儿童介绍 Stable Diffusion** 的建议。建议包括使用 **ControlNet** 将孩子的画作转换为写实图像，使技术更具吸引力和教育意义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/deepseek-ai/DeepSeek-VL-7B">与 DeepSeek VL 7B 聊天 - 由 deepseek-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://drive.google.com/file/d/1IBgfLqReWwhhWNXvnSCJH1gtQscgWPTV/view?usp=sharing">sanoma three 压缩包中的 stable diffusion web ui .zip</a>：未找到描述</li><li><a href="https://imgur.com/rYfd6lA">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 gif、鼓舞人心的故事、病毒式视频等来振奋精神...</li><li><a href="https://github.com/cheald/sd-webui-loractl/issues/30">功能请求：ComfyUI 实现？· Issue #30 · cheald/sd-webui-loractl</a>：嘿！你有没有可能将该逻辑实现为 ComfyUI 节点？</li><li><a href="https://imgur.com/aMqdy4z">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 gif、鼓舞人心的故事、病毒式视频等来振奋精神...</li><li><a href="https://github.com/cheald/sd-webui-loractl">GitHub - cheald/sd-webui-loractl: 一个用于在图像生成过程中动态控制 LoRA 权重的 Automatic1111 扩展</a>：一个用于在图像生成过程中动态控制 LoRA 权重的 Automatic1111 扩展 - cheald/sd-webui-loractl</li><li><a href="https://github.com/hollowstrawberry/kohya-colab">GitHub - hollowstrawberry/kohya-colab: 基于 kohya-ss 和 Linaqruf 工作的、易于使用的 Stable Diffusion LoRA 训练 Google Colab 笔记本</a>：基于 kohya-ss 和 Linaqruf 工作的、易于使用的 Stable Diffusion LoRA 训练 Google Colab 笔记本 - hollowstrawberry/kohya-colab</li><li><a href="https://github.com/PixArt-alpha/PixArt-sigma">GitHub - PixArt-alpha/PixArt-sigma: PixArt-Σ: 用于 4K 文本生成图像的 Diffusion Transformer 从弱到强训练</a>：PixArt-Σ：用于 4K 文本生成图像的 Diffusion Transformer 从弱到强训练 - PixArt-alpha/PixArt-sigma</li><li><a href="https://github.com/OlympiaAI/stability">GitHub - OlympiaAI/stability: 适用于 Stability AI API 的 Ruby SDK</a>：适用于 Stability AI API 的 Ruby SDK。通过创建账户为 OlympiaAI/stability 的开发做出贡献。</li><li><a href="https://civitai.com/models/35966/dpm-2m-alt-karras-sampler">DPM++ 2M alt Karras [ 采样器 ] - Automatic v1.6.0 | Stable Diffusion 其他 | Civitai</a>：这是 DPM++ 2M Karras 采样器的替代版本。我不声称这个采样器是终极或最好的，但我经常使用它，因为我...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d4cwi9/pcm_phased_consistency_model/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1245535144501317685)** (11 条消息🔥): 

- **GPT-4-OMNI 将变革教育领域**：一位成员分享了一篇关于 [GPT-4-OMNI](https://laion.ai/notes/open-gpt-4-o/) 及其对教育潜在影响的文章，将其想象为可以彻底改变我们学习方式的“个人学习助手”。讨论强调了由于 **multi-modal models** 的进步，这一愿景已触手可及。

- **新的 alignment 方法论文已提交**：另一位成员宣布他们已向 Arxiv 提交了关于新 alignment 方法的论文，并期待在获得批准后进行分享。此次提交引发了人们对论文内容及其潜在影响的好奇。

- **Luxia 21.4b 模型污染担忧**：在 GSM8k 测试中，**Luxia 21.4b** 模型不同版本之间的污染显著增加，根据 [HuggingFace](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1) 上分享的数据，从 v1.0 到 v1.2 增加了 29%。在 ARC 和 Wino 等其他评估指标中未观察到这种污染。

- **NeurIPS 模型合并竞赛**：NeurIPS 宣布举办一场模型合并（model merging）竞赛，承诺提供丰厚的奖励和科学贡献的机会。详细信息和报名信息已在一条 [推文](https://x.com/LChoshen/status/1796256513519989102) 中提供，同时还附带了竞赛 [官方页面](https://llm-merging.github.io/) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">Call to Build Open Multi-Modal Models for Personal Assistants | LAION</a>: &lt;p&gt;像 OpenAI 最近推出的 GPT-4-OMNI 这样的技术再次展示了强大的 multi-modal models 在积极变革方面的潜力……</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1">saltlux/luxia-21.4b-alignment-v1.2 · contamination results v1.0 vs v1.2 on GSM8K</a>: 未找到描述</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 @NeurIPSConf 模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创建最好的 LLMs！🧠✨ 💻为科学而来 💰为 $8K 奖金而留 💬Discord: https://discord.gg/dPBH...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1245465301026345000)** (50 条消息🔥): 

- **恒定学习率 vs. 余弦调度 (Cosine schedule)**：一篇 [arXiv 论文](https://arxiv.org/abs/2405.18392) 指出，带有冷却期 (cooldowns) 的恒定学习率在扩展性上具有可预测性和可靠性，与余弦调度类似。此外，随机权重平均 (stochastic weight averaging) 被证明可以在不增加额外训练成本的情况下提高性能。
- **引入上下文位置编码 (CoPE)**：[@jaseweston](https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw) 的一条推文讨论了 CoPE，这是一种针对 Transformer 的新型位置编码方法，它考虑了上下文信息。它能够处理计数和复制任务，并在语言建模和编程任务中表现出更好的性能。
- **Sonic：快速生成式语音模型发布**：[Cartesia AI](https://x.com/cartesia_ai/status/1795856778456084596?s=46) 宣布发布 Sonic，这是一款模型延迟仅为 135ms 的生成式语音模型，是其构建实时多模态智能愿景的一部分。
- **梯度多样性影响 mini-batch SGD 性能**：一篇 [arXiv 论文](https://arxiv.org/abs/1706.05699) 表明，梯度之间的高度相似性会降低 mini-batch SGD 的性能。梯度多样性对于在保持性能的同时实现加速至关重要。
- **NeurIPS 模型合并竞赛**：[NeurIPS 2023](https://x.com/LChoshen/status/1796256513519989102) 将举办一场模型合并 (Model Merging) 竞赛，奖金高达 8,000 美元。该竞赛由包括 Hugging Face 和 Sakana AI Labs 在内的机构赞助。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>: 规模已成为获得强大机器学习模型的主要因素。因此，理解模型的扩展特性是有效设计正确训练设置的关键...</li><li><a href="https://x.com/jaseweston/status/1795978611784089799?s=61&t=ryK3X96D_TkGJtvu2rm0uw">Jason Weston (@jaseweston) 的推文</a>: 🚨 上下文位置编码 (CoPE) 🚨 上下文至关重要！CoPE 是一种针对 Transformer 的新型位置编码方法，它考虑了 *上下文*。- 可以根据每个 Head 独立“计算”距离...</li><li><a href="https://arxiv.org/abs/2405.16684">gzip Predicts Data-dependent Scaling Laws</a>: 过去的工作已经建立了扩展定律，将神经语言模型 (LM) 的性能预测为参数量和训练 Token 数量的函数，从而实现最优...</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596?s=46">Cartesia (@cartesia_ai) 的推文</a>: 今天，我们很高兴发布了为每个设备构建实时多模态智能愿景的第一步：Sonic，一个极速（🚀 135ms 模型延迟）、栩栩如生的生成式语音模型...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>: 🚨 NeurIPS 上的模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLM！🧠✨ 💻 为科学而来 💰 为 8,000 美元而留 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://arxiv.org/abs/1706.05699">Gradient Diversity: a Key Ingredient for Scalable Distributed Learning</a>: 实验观察到，mini-batch 随机梯度下降 (SGD) 算法的分布式实现在超过一定程度后会出现加速饱和和泛化能力下降...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1245451696646262854)** (191 messages🔥🔥): 

- **MLP-Mixer 在因果性和序列长度方面的困境**：成员们讨论了使 MLP-Mixer 具备因果性并在各种序列长度下保持有效的挑战。一位成员指出：“似乎需要许多奇怪的技巧才能让 MLP 模型在任何序列长度下工作并具备因果性。”
  
- **Transformer 作为动态 MLP-Mixer**：对话强调了 Transformer 如何被视为上下文相关的 MLP-Mixer。一位成员认为：“Attention 基本上是一个 MLP-Mixer，其在时间维度上的权重是动态生成的，” 强调了上下文相关性的重要性。

- **对 MLP 与 Transformer 的批评及替代方案**：存在对 MLP 相对于 Transformer 的实用性和优越性的批评。一位用户表示：“不久前，MLP-Mixer 在许多领域本可以达到 SOTA，” 而其他人则指出需要上下文相关的操作来实现更好的可扩展性和适应性。

- **工业界对 Transformer 的偏好**：再次强调了 Transformer 在工业界的主导地位，并与过去的趋势进行了比较。一位成员评论道：“工业界一向如此。在 CNN 出现之前，他们也对 SVM 非常着迷，” 这表明了偏好的演变。

- **在 Diffusion 模型中探索替代方案与集成**：一些成员提到了 Diffusion 模型在机器人领域的应用，并对混合模型表示了兴趣。Gers101 提到：“Diffusion 目前在机器人领域非常火，他们使用 Diffusion 为模仿学习建模动作空间，” 反映了它们的多功能集成。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，将图像上的自回归学习重新定义为从粗到细的“次尺度预测”或“次分辨率预测”...</li><li><a href="http://arxiv.org/abs/2405.08553">Improving Transformers with Dynamically Composable Multi-Head Attention</a>：多头注意力 (MHA) 是 Transformer 的核心组件。在 MHA 中，注意力头独立工作，导致了诸如注意力评分矩阵的低秩瓶颈和头部冗余等问题。...</li><li><a href="https://x.com/arankomatsuzaki/status/1503543031923945475">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：使用稀疏全 MLP 进行高效语言建模。稀疏全 MLP 改善了语言模型的困惑度 (PPL)，与基于 Transformer 的 MoE 以及稠密 Transformer 相比，训练效率提升了高达 2 倍...</li><li><a href="https://arxiv.org/abs/1603.05691">Do Deep Convolutional Nets Really Need to be Deep and Convolutional?</a>：是的，它们需要。本文提供了第一个实证演示，证明深度卷积模型确实需要既深又是卷积的，即使是使用蒸馏等方法训练时也是如此...</li><li><a href="https://github.com/twistedcubic/attention-rank-collapse">GitHub - twistedcubic/attention-rank-collapse: [ICML 2021 Oral] 我们展示了纯注意力机制会遭受秩坍缩，以及不同机制如何应对它。</a>：[ICML 2021 Oral] 我们展示了纯注意力机制会遭受秩坍缩，以及不同机制如何应对它。 - twistedcubic/attention-rank-collapse</li><li><a href="https://arxiv.org/abs/2108.13002#microsoft">A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP</a>：卷积神经网络 (CNN) 是计算机视觉领域主导的深度神经网络 (DNN) 架构。最近，基于 Transformer 和多层感知器 (MLP) 的模型，如 Vision Transformer...</li><li><a href="https://arxiv.org/abs/2306.13575">Scaling MLPs: A Tale of Inductive Bias</a>：在这项工作中，我们重新审视了深度学习中最基础的构建块——多层感知器 (MLP)，并研究了其在视觉任务上的性能极限。对 MLP 的实证见解是...</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1245495179285102674)** (1 条消息): 

- **研究人员在语言模型中发现潜在推理**：分享了一个来自 [Jannik Brinkmann 的推文](https://x.com/BrinkmannJannik/status/1795827121585332459?t=UEx6PpAys4nmmLaEtyZSSQ&s=19)链接，他在其中讨论了在语言模型中发现潜在推理（latent reasoning）和搜索证据的研究。他们即将发表的 #acl2024 论文对一个在树搜索（tree search）上训练的 Transformer 进行了逆向工程，揭示了人类可理解的后向链接（backward chaining）电路。

**提到的链接**：<a href="https://x.com/BrinkmannJannik/status/1795827121585332459?t=UEx6PpAys4nmmLaEtyZSSQ&s=19">Jannik Brinkmann (@BrinkmannJannik) 的推文</a>：我们能在语言模型中找到潜在推理和搜索的证据吗？我们的 #acl2024 论文（与 @abhayesian 和 @VictorLevoso 合作）逆向工程了一个在树上训练的 Transformer 的内部机制...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1245453247129456772)** (19 条消息🔥): 

- **引用 Arxiv 论文**：有人分享了一篇 [Arxiv 论文](https://arxiv.org/pdf/2403.08295)以回应关于特定研究论文的查询。
- **讨论 Pull Requests**：成员们讨论了两个 Pull Request：[PR #2643](https://github.com/vllm-project/vllm/pull/2643) 用于向 API 服务器添加 `/get_tokenizer` 以简化集成；以及 [PR #1794](https://github.com/EleutherAI/lm-evaluation-harness/pull/1794) 在 EleutherAI 的仓库中实现类似功能。另一位成员提到了与 Logits 支持相关的 PR #1196，该请求已被拒绝。
- **机器翻译评估 PR**：一位成员分享了一个 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900)，内容是针对 11 种语言的机器翻译版 ARC challenge 评估，并寻求评审。
- **Token 评估异常**：随后讨论了各种数据集中出乎意料的快速 Token 评估，并提供了[额外解释](https://arxiv.org/abs/2405.14782)的链接。通过强调评估方法的差异，澄清了能耗测量和 Token 处理方面的问题。
- **LM_Eval 中的设备支持**：一位成员询问是否能将 LM_Eval 的设备支持扩展到 "CUDA" 之外。他们获悉 NPU 支持正在审查中，并建议为其他设备类型提交 issue 以征集社区贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14782">Lessons from the Trenches on Reproducible Evaluation of Language Models</a>：语言模型的有效评估仍然是 NLP 领域的一个开放挑战。研究人员和工程师面临着方法论问题，例如模型对评估设置的敏感性、属性获取的难度...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>：此 PR 为 11 种语言的机器翻译版 arc challenge 添加了任务。我们未来还将添加更多语言。</li><li><a href="https://github.com/vllm-project/vllm/pull/2643">Adding `/get_tokenizer` to api_server for lm-evaluation-harness ease integration.  by AguirreNicolas · Pull Request #2643 · vllm-project/vllm</a>：OpenAI 已经在大多数模型案例中支持聊天模型的 logprobs。与此同时，lm-evaluation-harness 正在开发中以支持此功能。为了运行端点的评估...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1794">Vllm get tokenizer by AguirreNicolas · Pull Request #1794 · EleutherAI/lm-evaluation-harness</a>：问题：vllm 服务提供的模型名称可能会更改，导致其无法响应任何 Huggingface 仓库，从而无法获取 tokenizer 并运行 lm-eval-har...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1245715839592955986)** (4 条消息): 

- **Pythia Tokenizer 频率说明**：一位成员询问 **Pythia 的 Hugging Face tokenizer** 中的 Token ID 是否根据其训练语料库的频率进行排序。另一位成员澄清道：“通常不是，”并指出为代码等特定上下文添加了额外的 Token。
- **将提供 Token 频率**：讨论以承诺在下午晚些时候提供来自 **the Pile** 的 Token 频率而结束。目前没有分享进一步的细节或链接。
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245689415075106866)** (3 messages): 

- **关于 ggml 库的问题**：一位成员向社区寻求帮助，询问是否有人在项目中使用过 **ggml 库**，并表示有一些相关问题。
- **NVIDIA 研究演讲见解**：NVIDIA 研究演讲的关键要点包括：一款 **4nm 芯片实现了 96 int4 TOPs/Watt** 的性能，以及对 **2:8 稀疏性** 的探索。成员分享了 [研究演讲](https://youtu.be/gofI47kfD28?si=41UIMkpMCyb_qWqA) 的链接以及相关的 [Physics of LLMs 论文](https://arxiv.org/abs/2404.05405)。
- **Meta 的 AI 硬件进展**：讨论重点介绍了 Meta 的下一代 **Meta Training and Inference Accelerator (MTIA)**，其特点是 **每个节点拥有 72 个加速器，在 90W 功耗下达到 354 TFLOPS/s (INT8)**。更多详情可以在 [Meta AI 博客](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware) 中找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1796253349932843214?s=46">来自 Daniel Han (@danielhanchen) 的推文</a>：我从 NVIDIA 研究演讲中记录的笔记：1) NVIDIA 拥有一款研究性质的推理 4nm 芯片，可实现 96 int4 TOPs/Watt，而 Blackwell 为 20T/W；2) B200 的 float4 是指数位=2 且尾数位=2？也许我听错了...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1245470054271946753)** (9 messages🔥): 

- **Triton 中的 Int32 乘法溢出**：一位成员发现 Triton 代码中存在一个 bug，即乘法在加到基础张量指针之前是以 **int32** 进行的，这导致大型张量出现溢出和 CUDA 内存错误。该问题在单元测试中未被发现，但在生产环境中由于数据量巨大而暴露。

- **CUDA 中的 Grid 维度限制**：另一位成员分享了他们在 **CUDA** 中的经验，代码在实际数据下崩溃，因为 **y 和 z 方向** 的 grid 维度是 16 位的，导致超过 65k 个 block 时出现问题。这一限制在单元测试中也未出现。

- **在 Triton Kernel 中传递元数据**：讨论了 Triton Kernel 缺乏对传递 **tuples** 或结构化值的支持。成员提到，如果能在一个对象中传递 shape 和 strides 等元数据，将能简化代码，特别是对于高维情况。

- **Triton.language.dot 与 Tensor Core 支持**：一位成员询问 `tl.dot` 是否支持 **bf16 Tensor Core**，并指出将 bf16 向上转型（upcasting）为 fp32 速度很慢。他们链接了一个 [GitHub issue](https://github.com/triton-lang/triton/issues/2302)，讨论了当输出类型为 bfloat16 时与 `out_type` 相关的 bug。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot.html#triton.language.dot">triton.language.dot &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/issues/2302">当 `out_type` 为 `bfloat16` 时，`tl.dot` 中的 `out_type` 存在 bug · Issue #2302 · triton-lang/triton</a>：当 dtype 为 bfloat16 时，tl.dot 中的 out_type 运行不符合预期。grad_query += tl.dot(grad_softmax, key, use_accelerator, dtype) 在这种情况下会出现编译错误。编译错误信息...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245580642574667877)** (6 messages): 

- **咨询 torch.profiler C++ API**：一位成员询问是否有 `torch.profiler` 的 C++ API，并提到虽然可以使用 `pybind` 从 Python 调用 C++ 函数，但他们正在寻求一种直接从 C++ 本身获取 trace 的方法。

- **Nightly 版本 torch.compile 变慢**：一位成员注意到 `torch.compile` 在最近的 nightly 版本中似乎变慢了。另一位成员链接了一个 [GitHub PR](https://github.com/pytorch/pytorch/pull/126320)，该 PR 应该能解决这个问题。

- **torch.compile 与反向传播 Kernel**：确认了 `torch.compile` 会为前向和反向传播生成 Kernel。用户可以通过设置 `TORCH_LOGS="output_code python your_code.py` 来识别这些 Kernel。

- **Profiling 中的 Triton Kernel**：一位成员询问在 torch profiling 中看到的 "triton kernel" 是代表所有 Triton Kernel 的聚合还是特定的某一个。他们注意到尽管进行了搜索，但在 `output_code` 中未找到相关条目。

**提及的链接**：<a href="https://github.com/pytorch/pytorch/pull/126320">由 Chillee 为 partitioner 添加了内存预算 · Pull Request #126320 · pytorch/pytorch</a>：来自 ghstack 的堆栈（最早的在底部）：#127520 -> #126320 #127446

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1245517354679533610)** (11 messages🔥): 

- **使用 torch.compile 没有速度提升**：一位用户尝试使用 `torch.compile` 来加速 Hugging Face 的 `AutoModelForCausalLM` 的推理和训练，但没有观察到任何改进。他们描述了在训练过程中模型损坏的问题，以及普遍缺乏速度提升的情况。
  
- **使用 torch.compile 的技巧**：另一位成员提供了建议，包括设置 `model.config.use_cache=True` 以及在 `model.forward` 上使用 `torch.compile` 以获得更好的效果。他们分享了一个辅助脚本，并指出最新的 `transformers` 版本可能存在问题，并链接到了他们的 [GitHub repository](https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py)。

- **推荐 vllm 及批处理咨询**：该用户接受了建议，并表示可能会尝试使用 `vllm` 进行推理。他们还询问了关于使用单个模型处理并发图像处理的高效 Batching 技术，并提到之前使用 `multiprocessing` 库的尝试并未成功。

**提到的链接**：<a href="https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py">hqq/hqq/utils/generation_hf.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1245648613519003658)** (4 messages): 

- **对 FP4 乘法规范的困惑**：一位成员对“两个 FP4 数值相乘”是否涉及精确计算后再舍入，还是一个有损过程表示不确定。他们推测将 P_1, P_2 转换为 FP32，然后进行乘法和舍入是否会得到相同的结果。
- **混合精度层探索**：有人询问是否考虑过用于激活值和梯度的“混合精度层”，建议在前向传递中使用 FP4/FP8，在反向传递中使用 FP8/FP16。他们指出，在基础的反向传播算法中，将梯度应用于 FP4 权重可能没有意义。
- **在特定问题上标记专家**：建议在 AO 的 Issue 追踪器上提出技术问题并标记 `vkuzo` 以获得更好的回复，因为他不经常查看 Discord。
- **点积累加的精度**：一位成员澄清说，根据规范，点积累加的精度是“由实现定义的”，并且“取决于支持 MX 的硬件”。他们提到 PyTorch 支持混合精度层进行模拟，尽管各种精度具体的硬件支持情况仍然未知。
- **FP6-LLM CUDA Kernel 细节**：FP6-LLM CUDA Kernel 被描述为一种针对 FP16 激活值和 MX fp6_e3m2 权重的“混合输入 matmul”，计算是在 Tensor Cores 中以 FP16 完成的。
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1245771308265111564)** (1 messages): 

- **Whisper 模型提速 6.3 倍**：一位成员宣布了通过使用 **static cache**、**HQQ quantization**、**torchao 4-bit kernel** 以及带有 fullgraph 的 **torch.compile** 优化 Whisper 模型的惊人结果。他们透露“明天将发布一篇博客文章”详细分享这些见解。
  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1245771200072781924)** (3 messages): 

- **活跃频道**：一位用户表示有兴趣在一段时间的沉寂后重新活跃该频道的讨论。
- **MatMul 演示 GIF**：分享了一个包含 **矩阵乘法 (matmul)** 动画的 **开发中 (work-in-progress)** 演示，可以在[此处](https://media.discordapp.net/attachments/1225431825929863320/1245745841143025826/full_tensor_multiplication.gif?ex=6659deb9&is=66588d39&hm=6f6bfb0b97e8a64415706dfe84c40cf3efe2a64c5e2e930e09ca478f0f1d15da&)查看。
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1245483580789100604)** (122 条消息🔥🔥): 

- **使用 Cloudflare R2 替代 Python 依赖**：一位成员建议使用 Cloudflare R2 存储 pretok 数据，以消除 Python 依赖，并指出由于没有类似 S3 的出站流量费（egress fees），其成本更低。他们确认已成功启用 R2，并将在进一步测试后反馈。

- **用于大规模生产的 DNDEBUG 宏**：引入了 DNDEBUG 宏，该宏在编译时移除 assert 检查，适用于大规模生产运行。这在 CUDA 的 Kernel 大小检查中对于性能调优非常有用。

- **通过模板化 Block 大小提升性能**：成员们讨论了通过模板化变量（如 blockDim.x）在 CUDA Kernel 中实现显著的速度提升，特别是在 fused classifier 中。这种方法已显示出可衡量的性能增益，并且与更复杂的模板化方法相比，可以简化代码。 

- **在内部 S3 存储数据以避免费用**：建议使用内部 S3 存储以避免杂费，预先上传 tokenizer 和数据集文件等资源。这种方法便于共享用于训练的大型数据集，而不会产生额外费用。

- **合并并优化 Kernel 重计算**：提议合并一个涉及 layernorm 重计算的代码分支，尽管它目前还不是最快的版本。目标是先集成功能性改进，随后通过减少冗余计算（如复用 mean 和 rstd 值）进行进一步优化。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/475">karpathy 尝试添加 llmc lib 目录的实验 · Pull Request #475 · karpathy/llm.c</a>: 未找到描述</li><li><a href="https://en.cppreference.com/w/c/error/assert">assert - cppreference.com</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1245616070069588008)** (3 条消息): 

- **观影会进度跟进**：一位新成员询问了讲座系列的当前进度和即将播放的视频。另一位成员回复称，下一场可能是 Lecture 7，但不确定具体细节。
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1245467775376560190)** (72 条消息🔥🔥): 

- **CI 测试与贡献者权限问题**：一位新贡献者由于初始权限问题在 PyTorch 的 CI 测试中遇到挑战，但随后获得了更宽松的权限。分享了 [GitHub Pull Request #296](https://github.com/pytorch/ao/pull/296) 用于测试。
  
- **排查安装错误**：几位用户排查了关于 PyTorch 的安装错误，重点讨论了构建 C++ 扩展的问题，并提出了各种修复方案，包括使用 `USE_CPP=0 pip install .` 以及升级 pip 和 setuptools。

- **Windows 兼容性问题**：成员们讨论了 PyTorch 和 triton 与 Windows 的兼容性，指出 triton 在 Windows 上没有官方支持。特别关注了解决需要 CUDA 算力（capability）大于 8 的构建问题。

- **打包与安装建议**：用户分享了符合 PEP 标准的正确打包实践见解，并分享了一个[相关的 PR 供参考](https://github.com/TimDettmers/bitsandbytes/pull/1078)。他们建议使用 `pip install --no-build-isolation .` 等命令，并确保安装了 `wheel`，以避免构建过程中的隔离环境问题。

- **CUDA 与 Dtype 讨论**：用户参与了关于 kernel 编译和正确 CUDA 设备算力需求的讨论。分享了关于优化代码的具体反馈，例如 *“对于 bit pack，容器大小可以根据 dtype 确定”*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/actions/runs/9305510545/job/25612612623#step:15:99">修复了 NF4Tensor 的 f-string 打印问题 (#297) · pytorch/ao@4c1d568</a>：用于量化和稀疏化的原生 PyTorch 库 - 修复了 NF4Tensor 的 f-string 打印问题 (#297) · pytorch/ao@4c1d568</li><li><a href="https://pastebin.com/CMhYTn20">正在处理 /home/swan/pytorch/ao 安装构建依赖项 ... 完成 Ge - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/pytorch/ao/pull/296">由 msaroufim 优雅地处理 cpp 扩展 · Pull Request #296 · pytorch/ao</a>：这修复了 #288 中的问题。更具体地说，我们引入了一个新的环境变量，以便我们可以在不构建 cpp 扩展的情况下本地安装 ao：USE_CPP=0 pip install .，这出于方便考虑非常有用...</li><li><a href="https://github.com/TimDettmers/bitsandbytes/pull/1078">由 matthewdouglas 将构建数据迁移到 pyproject.toml · Pull Request #1078 · TimDettmers/bitsandbytes</a>：将大部分元数据移动到 pyproject.toml，符合 PEP 517、PEP 518 和 PEP 621。删除了 requirements.txt 文件（但尚未删除 conda environment.yml 文件）。更新了文档以指导使用...</li><li><a href="https://download.pytorch.org/whl/cu121">未找到标题</a>：未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp311-cp311-win_amd64.whl">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245459641802952934)** (5 messages): 

- **Mistral AI 发布 Codestral**：Mistral AI 发布了 Codestral，这是一个全新的代码生成模型，支持本地运行，并在 80 多种编程语言上进行了训练。LlamaIndex 提供了 [Day 0 支持](https://t.co/k2nHDiMnwD)，并准备好了演示其用法的 [Notebook](https://t.co/YxeyHhSjKU)。
- **Codestral 获得 Ollama 支持**：除了独立运行能力外，Codestral 还得到了 Ollama 的支持，允许通过 LlamaIndex 对 [Ollama 的一流支持](https://t.co/gsPHHF4c0K)进行本地执行。
- **本地知识图谱指南**：一份新指南解释了如何遵循预定义模式，并使用本地模型 (@ollama, @huggingface) 以及 Neo4j 作为图存储来构建知识图谱。详情请见[此处](https://t.co/xhoIEi9egq)，完整指南请见[此处](https://t.co/5ee6LwM7RE)。
- **伦敦 NLP 聚会**：来自 LlamaIndex 的 @hexapode 将加入 @weaviate_io 和 @weights_biases，在 6 月 12 日的伦敦 NLP 聚会上发表关于在金融服务中使用 LLM 的演讲。在此处[报名](https://t.co/vli6DY8Xg7)，获取关于管理向量数据库和处理金融数据的见解。
- **LlamaParse 现在支持电子表格**：LlamaParse 现在可以处理各种电子表格，包括 Excel 和 Numbers，将它们转换为适用于 RAG 流水线的干净表格。查看详细的 [Notebook](https://t.co/60MvR0h5DC) 和演示[此处](https://t.co/IfF4UUqB0C)。

**提到的链接**：<a href="https://t.co/vli6DY8Xg7">Solving the challenges of using LLMs in production with financial services data, Wed, Jun 12, 2024, 6:00 PM | Meetup</a>：如果你正在构建用于处理金融服务数据的 NLP 流水线，你就会知道在生产环境中管理向量数据库、可靠地处理大型数据是多么困难。

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1245574510246232084)** (89 messages🔥🔥): 

- **iOS 浏览器导致网站崩溃**：一位用户提到该网站在 **iOS 浏览器**（使用 Chrome 和 Safari）上不断崩溃。另一位用户建议提供包含可复现步骤的更详细错误报告，以协助调试。

- **默认提示词模板变量列表**：用户询问 LlamaIndex 中的默认提示词模板变量。回复澄清了变量 `schema_str`、`info_str` 和 `query_str`，并提供了详细的代码示例和[文档链接](https://docs.llamaindex.ai/en/latest/examples/vector_stores/pinecone_auto_retriever/)。

- **文本分块策略**：关于在 LlamaIndex 中创建节点时的**默认分块策略**进行了讨论。澄清了默认分块设置为使用 `SentenceSplitter` 的 1024 个 Token，相关细节通过[文档链接](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=sentencesplitter#sentencesplitter)提供。

- **切换 API 框架**：用户询问选择 API 框架的建议，提到了 **Flask** 和 **FastAPI**。推荐使用 FastAPI，因为它支持异步编程，有利于处理多个用户请求。

- **将数据迁移至 RedisStores**：用户请求关于将数据从 **SimpleStore** 迁移到 **RedisStore** 的建议。回复指出这虽然困难但可行，建议采用将节点和 Embedding 添加到新向量存储的方法，并指出 `IngestionPipeline` 可以自动执行此过程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern/#accessing-prompts">Usage pattern - LlamaIndex</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1hiDkBbAJcO3RDrS7CD2ZeQEHGqNv07pq?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLAutoVectorQueryEngine/">SQL Auto Vector Query Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=sentencesplitter#sentencesplitter">Node Parser Modules - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/?h=sentencesplitter#llama_index.core.node_parser.SentenceSplitter">Sentence splitter - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1245673302715006987)** (1 messages): 

- **Property Graphs 在 LlamaIndex 中备受关注**：[Unveiling the Power of Property Graphs with LlamaIndex](https://medium.com/ai-advances/unveiling-the-power-of-property-graphs-with-llamaindex-233be48934f9) 是聊天中分享的一篇新博客文章，强调了 Property Graphs 在 AI 开发中的强大功能和潜力。该文章旨在深入探讨 LlamaIndex 如何利用这项技术。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1245485617731731516)** (52 messages🔥): 

- **Le Tigre 黑客松大获成功**：成员们讨论了在黑客松期间构建的 "Le Tigre" 项目，该项目被描述为“受 GPT-4-V 架构启发、基于 Mistral 7B 模型的多模态变体”。更多详情和项目链接可以在 [Devpost](https://devpost.com/software/le-tigre) 和 [GitHub](https://github.com/HugoLB0/Le-Tigre) 上找到。

- **即将发布的 LAION 数据集**：成员们询问了 LAION 5B 数据集的预计发布时间 (ETA)，表示该数据集已经逾期很久，并询问是否会重新发布。

- **Sonic：生成式语音模型**：Cartesia AI 推出了 Sonic，这是一款具有 135ms 延迟的 SOTA 级逼真生成式语音模型。更多信息和演示可在其 [博客](https://cartesia.ai/blog/sonic) 和 [Twitter](https://x.com/cartesia_ai/status/1795856778456084596) 上查看。

- **对 ToonCrafter 的质疑**：成员们对用于草图引导动画的新项目 ToonCrafter 持谨慎乐观态度，该项目可在 [GitHub](https://huggingface.co/Doubiiu/ToonCrafter/tree/main) 和 [Gradio](https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#2-local-gradio-demo) 上获得。关于其质量和实际用途存在一些讨论和怀疑。

- **动漫制作成本缩减**：一段对话强调了动漫制作经济效益的变化，指出先进的建模工具可以显著降低成本。一位成员提到，以前每集动漫的成本高达数十万美元。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Gradio/status/1796177536348561512">来自 Gradio (@Gradio) 的推文</a>：ToonCrafter 仓库中提供本地 Gradio 演示：https://github.com/ToonCrafter/ToonCrafter?tab=readme-ov-file#2-local-gradio-demo 模型：https://huggingface.co/Doubiiu/ToonCrafter 🚀 感到兴奋 ...</li><li><a href="https://huggingface.co/Doubiiu/ToonCrafter/tree/main">Doubiiu/ToonCrafter at main</a>：未找到描述</li><li><a href="https://doubiiu.github.io/projects/ToonCrafter/">ToonCrafter: Generative Cartoon Interpolation</a>：未找到描述</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596">来自 Cartesia (@cartesia_ai) 的推文</a>：今天，我们很高兴发布我们使命的第一步，即为每台设备构建实时多模态智能：Sonic，一款极速（🚀 135ms 模型延迟）、逼真的生成式语音模型...</li><li><a href="https://x.com/hugolb05/status/1795426269099606329">来自 Hugo Le Belzic (@hugolb05) 的推文</a>：在 @cerebral_valley 和 @MistralAILabs 黑客松期间，我们构建了 &#34;Le Tigre&#34;，这是一个基于 Mistral 7B 模型的多模态变体，灵感来自 GPT-4-V 的架构。更多详情...</li><li><a href="https://youtu.be/cvZ9thKolOA?si=yHgMyzqfpM8tVcxu&t=53">极主夫道 | 预告片 | Netflix 动漫</a>：这位世界级的家庭主夫曾是令人闻风丧胆的传奇黑帮成员！“极主夫道”是这部温馨作品期待已久的动漫改编...</li><li><a href="https://getwrightonit.com/animation-price-guide/>">未找到标题</a>：未找到描述</li><li><a href="https://devpost.com/software/le-tigre">Le Tigre</a>：Le Tigre 擅长跨音频、视觉和文本的实时推理。它是我们对 Mistral 开源模型进行广泛协作和微调的结果。</li><li><a href="https://laion.ai/notes/open-gpt-4-o/">呼吁为个人助手构建开源多模态模型 | LAION</a>：&lt;p&gt;像 OpenAI 最近推出的 GPT-4-OMNI 这样的技术再次展示了强大的多模态模型在积极转型方面的潜力...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1245492363246178314)** (1 条消息): 

- **LAION 呼吁为 GPT-4-Omni 贡献力量**：LAION 邀请社区参与构建 **开源 GPT-4-Omni**，其[博客文章](https://laion.ai/notes/open-gpt-4-o/)中提供了详细的方向指引。该倡议旨在创建一个具有类似于 GPT-4-OMNI 的大规模 multi-modal 能力的开源模型。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">呼吁为个人助手构建开源 Multi-Modal 模型 | LAION</a>: &lt;p&gt;像 OpenAI 最近推出的 GPT-4-OMNI 这样的技术再次展示了强大的 multi-modal 模型在积极变革方面可能具有的潜力...</li><li><a href="https://fxtwitter.com/laion_ai/status/1795910332008804428">来自 LAION (@laion_ai) 的推文</a>: 帮助我们构建开源 GPT-4-Omni！通过这篇博客文章，我们展示了极具前景的方向（包括数据集和教程） https://laion.ai/notes/open-gpt-4-o/
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245464253595521094)** (33 条消息🔥): 

- **Era3D 推出高分辨率多视图扩散模型**：一位成员分享了关于 [Era3D](https://penghtyx.github.io/Era3D) 的细节，称其是一种利用高效行注意力（row-wise attention）进行高分辨率多视图扩散的新方法。该论文由来自香港科技大学（HKUST）和北京大学（PKU）等多个机构的作者共同完成。
  
- **对激励性用户评分的担忧**：讨论强调了为用户评分提供激励往往会导致数据质量下降。一位成员指出，用户可能会通过提交随机评分来博弈系统，以增加获得奖励的机会，并引用了 Midjourney 中免费 GPU 时长被滥用的例子。

- **NeurIPS 模型合并竞赛发布**：一项专注于模型合并（model merging）的竞赛已发布，引起了广泛的参与兴趣。相关竞赛细节，包括 [推文链接](https://x.com/LChoshen/status/1796256513519989102) 和报名链接，已分享给那些有兴趣通过提升 LLM 性能来争夺 8,000 美元奖金池的人。

- **FFT 以极高效率替代 Self-Attention**：讨论涉及一篇 2021 年的论文，该论文将 Transformer 中的 Self-Attention 替换为 FFT，达到了 BERT 92% 的准确率，但计算成本显著降低。这种方法引发了人们对其是否得到进一步研究的好奇，并附上了 [论文链接](https://arxiv.org/pdf/2105.03824)。

- **用于生成式卡通插值的 ToonCrafter**：提到了 ToonCrafter，这是一个用于生成式卡通插值的研究项目，并附带了 [GitHub 链接](https://github.com/ToonCrafter/ToonCrafter)。在发现展示其能力的 [项目文档](https://doubiiu.github.io/projects/ToonCrafter/) 后，讨论了该项目的可行性和令人印象深刻的效果。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18428">DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention</a>: 具有大规模预训练的扩散模型在视觉内容生成领域取得了显著成功，特别是以 Diffusion Transformers (DiT) 为代表。然而，DiT 模型...</li><li><a href="https://arxiv.org/abs/2403.01643">You Need to Pay Better Attention</a>: 我们引入了三种新的注意力机制，在效率和学习能力方面优于标准多头注意力，从而提高了性能和更广泛的部署能力...</li><li><a href="https://penghtyx.github.io/Era3D/">Efficient High-Resolution Multiview Diffusion on Canonical Orthogonal Cameras</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.05219">Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers</a>: 大语言模型 (LLMs) 深刻改变了世界。它们的 Self-Attention 机制是 Transformer 在 LLM 中取得成功的关键。然而，平方级的计算成本 $O(n^2)$ 使得...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">来自 Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>: 🚨 NeurIPSConf 模型合并竞赛！🚀 你能彻底改变模型选择和合并吗？让我们创造最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元而留 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://doubiiu.github.io/projects/ToonCrafter/">ToonCrafter: Generative Cartoon Interpolation</a>: 未找到描述</li><li><a href="https://tenor.com/view/ha-ha-ha-ha-ha-happy-funny-%C5%9Bmiech-gif-22074544">Ha Ha Ha Happy GIF - Ha Ha Ha Ha Ha Happy - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: 生成式卡通插值研究论文</a>: 生成式卡通插值研究论文 - ToonCrafter/ToonCrafter</li><li><a href="https://syncedreview.com/2021/05/14/deepmind-podracer-tpu-based-rl-frameworks-deliver-exceptional-performance-at-low-cost-19/">谷歌用傅里叶变换替换 BERT 的 Self-Attention：准确率 92%，GPU 速度提升 7 倍 | Synced</a>: 自 2017 年推出以来，Transformer 架构已主导自然语言处理 (NLP) 领域。Transformer 应用的唯一限制之一是巨大的计算量...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1245571712494338078)** (11 messages🔥): 

- **分享 In-context Learning 技术**：成员们讨论了使用 Prompt 来“教导”模型的方法，涉及 100k Context Windows 或更短的窗口。他们强调在 System Prompt 中包含信息或 Prompt 与 Response 可以提升性能。

- **高效数据处理的担忧**：一位成员对每次请求都向模型高效输入大量数据的可行性表示担忧，提到这可能非常耗时。他们建议像 **RWKV** 这样可以保存 State 的模型可能会更有效地处理这一问题。

- **非 Backpropagation 训练构想**：提出了一种不使用 Backpropagation 训练模型的想法，以一种全新的方式利用 Pretraining 数据。该成员承认这可能是一个复杂的概念，并表示如果需要可以进一步解释。

- **扩展 Preprompt 成功的案例**：一位成员分享了一个成功使用 In-context Learning 解决问题的案例，引用了 [这条推文](https://twitter.com/VictorTaelin/status/1776677635491344744)。他们指出，该个人因其成就获得了 1 万美元的奖金。
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1245468046831915091)** (5 messages): 

- **对欧洲大学趣味学习环境的向往**：一位成员表达了希望在欧洲找到一所具有良好学习氛围和趣味文化的大学。他们提到：*"只是很难在欧洲等地找到具有那种文化的大学。"*
- **德国与美国大学的文化差异**：另一位成员观察到德国和美国大学之间存在显著的文化差异，指出：*"美国的教授在 1:1 交流时更加平易近人/亲切。"*
- **寻找 Web App 开发团队成员**：一位成员正在开发一个 Web Application 并寻找潜在的团队成员，询问：*"我可以就此发一个简短的帖子吗？"*
- **Codestral Mistral 代码模型介绍**：分享了一个名为 [Codestral Mistral AI's first-ever code model](https://www.youtube.com/watch?v=WRAbOHJJMF4) 的视频，将 Codestral 描述为 *"一个专门为代码生成任务设计的 Open-weight 生成式 AI 模型。"*

**提到的链接**：<a href="https://www.youtube.com/watch?v=WRAbOHJJMF4">Codestral Mistral AI's first-ever code model</a>：Codestral 是 Mistral 的首个代码模型。Codestral 是一个专门为代码生成任务设计的 Open-weight 生成式 AI 模型。它有助于开发...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1245452539760218278)** (3 messages): 

- **Scale 的私有评估数据集以透明度为傲**：分享了指向 [Scale's Leaderboard](https://scale.com/leaderboard) 的链接，强调了他们的**专有评估数据集**。这些数据集确保了“无偏见且未受污染的结果”，通过不断更新的数据集和模型促进持续进化的竞争环境。

- **对 Scale 提出的担忧**：一位成员对 Scale 的做法表示担忧。具体而言，该成员提到 Scale “为除了 Llama 3 之外的所有模型提供 SFT 以及可能的 RLHF 数据”，对这种做法的可靠性和透明度提出了质疑。

**提到的链接**：<a href="https://scale.com/leaderboard">SEAL leaderboards</a>：未找到描述内容

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1245457755167260784)** (41 条消息🔥): 

- **Llama 3 vs Code Llama**：一场关于 **Llama 3 70B** 在各项任务中表现优于 **Code Llama** 的讨论展开了，尽管后者是专门的代码模型。一位用户表达了对 "Code Llama 3" 的期待。

- **混合模型策略**：用户讨论了相比于专门化模型，拥有“全能型” **LLMs** 的好处。提到了“动态卸载到云端”是未来的一项关键创新。

- **Yuan2-M32 发布**：浪潮信息（IEIT-Yuan）新发布的 **Yuan2-M32** 模型拥有 40B 参数，但在生成过程中仅有 3.7B 激活参数，在大多数基准测试中以显著更少的资源达到了与 **Llama 3 70B** 相当的水平。用户受邀对其进行微调，并分享了 [代码和论文](https://github.com/IEIT-Yuan/Yuan2.0-M32)。

- **NeurIPS 模型合并竞赛**：分享了在 **NeurIPS 举办的模型合并竞赛**公告，奖金为 8000 美元。详情见 [公告推文](https://x.com/LChoshen/status/1796256513519989102)。

- **新进展与工具**：一位成员重点介绍了一个用于构建 **LLM 应用** 的新 [Rust 库](https://x.com/LChoshen/status/1796256513519989102)。另一位分享了 **MoRA** 的发布，这是一种用于微调的高秩更新（high-rank updating）技术，可在 [GitHub](https://github.com/kongds/MoRA) 上获取。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/_philschmid/status/1796191402935632043?s=46">Philipp Schmid (@_philschmid) 的推文</a>：仅凭 3.7B 激活参数就在 MMLU 上达到 72.2%，在 HumanEval 上达到 74.4%？🤔 一个使用了全新 Attention Router 机制的 40B Mixture of Experts 模型 👀 Yuan2-M32 由 IEIT-Yuan 发布。TL;DR: 🧠 40B ...</li><li><a href="https://x.com/janleike/status/1795497960509448617">Jan Leike (@janleike) 的推文</a>：我很高兴加入 @AnthropicAI 继续超级对齐（superalignment）使命！我的新团队将致力于可扩展监督、弱到强泛化以及自动化对齐研究。如果你...</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Leshem Choshen @LREC 🤖🤗 (@LChoshen) 的推文</a>：🚨 @NeurIPSConf 模型合并竞赛！🚀 你能彻底改变模型选择与合并吗？让我们创造最好的 LLMs！🧠✨ 💻 为科学而来 💰 为 8000 美元奖金而留 💬 Discord: https://discord.gg/dPBH...</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>：MoRA：用于参数高效微调的高秩更新 - kongds/MoRA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1245546994731061361)** (7 条消息): 

- **吴恩达（Andrew Ng）关于 Agent 的新免费课程**：成员们讨论了吴恩达新推出的关于 Agent 的免费短课，课程将 Agent 描述为“能够独立规划并执行各种任务以达成目标的实体”。
- **对 Agent 的幽默解读**：一位成员幽默地将 Agent 定义为“for 循环中的 LLM，哈哈”，为 Agent 的概念增添了轻松的视角。
- **创建 DPO 数据集的挑战**：一位成员在使用 GPT-4 和 Mistral7b 生成回复以创建 DPO 数据集时遇到困难，注意到在给定相同问题和上下文时，两个模型生成的输出同样出色。
- **探索用于数据集的弱模型**：为了解决数据集质量问题，一位成员考虑使用较弱的 7b 模型，但注意到 Falcon7b instruct 经常对许多查询给出 "<nooutput>"。
- **关于 Transformer XL 论文的疑问**：一位成员就 Transformer XL 论文中关于上下文如何编码进隐藏状态并用于获取 Logits 的概念寻求澄清，对论文中描述的过程提出了疑问。
  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1245674129072459807)** (14 条消息🔥): 

- **Hybrid search 至关重要且实现简单**：一名成员强调，引入 Hybrid search 非常关键且极易实现。他们表示：*"Hybrid search 是必须的，而且添加起来非常容易。"*

- **讨论了用于相关性评估的 MRR 和 NDCG 指标**：成员们根据顾问 [Hamel et al](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) 的建议，讨论了使用 **MRR** 和 **NDCG** 作为相关性指标。关于这些指标是否需要人工评估以及检索后如何确定排名，存在一些困惑。

- **新的 RAG 数据集已发布**：一个新的 **RAG 数据集**已分享，可以通过 [Hugging Face](https://huggingface.co/datasets/glaiveai/RAG_v1) 在特定条件下访问。用户需要同意分享其联系信息才能获取。

- **分享了幽默的 GIF 动图**：分享了一个猫咪要求进门的幽默 **GIF 动图**，来源自 [Tenor](https://tenor.com/view/cats-let-us-in-gif-13593927)。该 GIF 为讨论增添了轻松的氛围。

- **确认任务正在进行中（WIP）**：一名成员确认某项任务仍处于进行中状态，目前正在测试。他们提到：*"啊，那个还在 WIP（进行中），现在正在测试。"*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/glaiveai/RAG_v1">glaiveai/RAG_v1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">构建 LLMs 应用一年的经验教训（第一部分）</a>：未找到描述</li><li><a href="https://tenor.com/view/cats-let-us-in-gif-13593927">Cats Let Us In GIF - Cats Let Us In - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1245458164510363841)** (30 条消息🔥): 

- **ABI 稳定性的复杂性讨论**：社区成员讨论了不同语言之间 ABI 稳定性的复杂性，强调 **Rust 故意缺乏 ABI 稳定性**以避免向后兼容性问题，而 **Swift** 为 Apple 操作系统维护 ABI 稳定性。讨论指出，在 Go 等语言中，由于 API 兼容性的限制，维护 ABI 稳定性往往会限制性能优化。
- **Mojo 作为潜在 C "协议" 面临质疑**：一位成员幽默地建议 Mojo 可能成为类似 C 的新低级协议，但其他人表示怀疑。成员们提到了 Mojo 缺乏关键类型（如 `size_t` 和 `uint_fast_t`）以及 C 等既有语言提供稳定性的惯性。
- **C++ 互操作性对 Mojo 的重要性**：大家一致认为 Mojo 具有良好的 **C++ 互操作性**对于利用庞大的现有代码库非常有价值。**clattner** 提到未来计划探索从 Mojo 代码生成 C++ 头文件，这有助于简化过渡和采用。
- **Mojo 包管理正在开发中**：社区渴望 Mojo 包管理器，并提到了关于项目清单（manifest）格式的持续讨论。分享了 [GitHub 讨论](https://github.com/modularml/mojo/discussions/413) 和 [提案线程](https://github.com/modularml/mojo/discussions/1785) 的链接，表明该功能正在开发中，但并非迫在眉睫。
- **Mojo 尚未支持 Windows**：一些用户对 Mojo 缺乏 Windows 支持感到沮丧，并幽默地表示以后再来查看。尽管有请求，但答案仍然是 Mojo 目前在 Linux 环境之外不可用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rust-lang/rust/issues/111423)">Issues · rust-lang/rust</a>：赋能每个人构建可靠且高效的软件。- Issues · rust-lang/rust</li><li><a href="https://github.com/modularml/mojo/discussions">modularml/mojo · Discussions</a>：浏览 modularml mojo 的 GitHub 讨论论坛。讨论代码、提问并与开发者社区协作。</li><li><a href="https://github.com/modularml/mojo/discussions/413">[RFC] Allow Importing Modules via URLs · modularml/mojo · Discussion #413</a>：概述 Mojo 的主要优先级之一是解决“两种语言问题”，这意味着它必须既能用于应用开发用例，也能用于一次性脚本。依赖管理...</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[Proposal] Mojo project manifest and build tool · modularml/mojo · Discussion #1785</a>：大家好，请查看这个关于 Mojo 项目清单和构建工具的提案。正如提案本身所述，我们希望听到 Mojo 社区的声音：你是否同意这些动机...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1796232248678883347>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1245780625613521036)** (1 条消息): 

- **使用 Mojo🔥 加速 K-Means 聚类**：ModularBot 发布了一个新视频，介绍如何将 **K-Means 聚类从 Python+NumPy 移植到 Mojo** 以获得显著的性能提升。视频承诺了详细步骤，并声称实现了巨大的 **250 倍加速**。[在此观看视频](https://www.youtube.com/watch?v=3bg5YBCcuWA)。

**提到的链接**：<a href="https://www.youtube.com/watch?v=3bg5YBCcuWA">Speed up K-Means clustering by porting Python implementation to Mojo🔥</a>：在本视频中，我们将分享将 kmeans 聚类从 Python+NumPy 移植到纯 Mojo 的分步指南，以实现巨大的（250倍）加速！如何实现？Mojo 在...方面具有 Python 风格。

  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1245480100787388436)** (5 messages): 

- **Mojo 的 `^` 运算符和 `bit_not` 用法说明**：一位用户询问 Mojo 中的 `^` 运算符是否与 C 语言中的相同，以及 `bit_not` 的用法。另一位用户澄清说 `bit_not` 是 `~val`，XOR 操作是 `x ^ y`，而转移运算符（transfer operator）则跟在值后面，如 `val^`。

- **调试 C 和 Mojo 之间的 XOR 操作**：一位成员在比较涉及移位和 XOR 的 C 和 Mojo 代码时发现了不一致。检查后，他们发现问题出在 C 代码中打印了错误的变量，现在两份代码的结果已一致。

- **Mojo 中的 `for` 循环和迭代器**：一位用户询问 Mojo 中的 `for` 循环如何终止（对比 Python 的 `StopIteration`），并确认 `for` 循环会调用 `__iter__` 方法并生成一个迭代器。
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1245619388334477312)** (6 messages): 

- **Mojo 与 Python 的比较受到质疑**：一位成员对 Santiago 关于 Mojo 和 Python 在二分查找（binary search）上的性能比较表示怀疑，指出在循环中调用 Python 接口 100,000 次并不是公平的比较。该成员对 Modular 转发这条推文感到惊讶，尽管其基准测试方法存在疑问，如[这条推文](https://x.com/svpino/status/1795811741538099685)所示。

- **透明度与有缺陷的基准测试**：另一位成员承认，虽然基准测试可能不够严谨，但 Santiago 对其方法是透明的。他们用一个耸肩的表情符号暗示了一种随性的态度。

- **分享编译器基准测试仓库**：分享了一个很酷的 [GitHub 仓库](https://github.com/nordlow/compiler-benchmark)，该仓库比较了不同编程语言和编译器的编译速度，不过它没有考虑缓存因素。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/svpino/status/1795811741538099685">Santiago (@svpino) 的推文</a>：Mojo 🔥 在速度上碾压了 Python。我用 Mojo 重写了一个简单的 Python 二分查找函数。改动非常小。我在循环中调用该函数 100,000 次：• Python: 547 ms • Mojo 🔥: 44 ms。这...</li><li><a href="https://x.com/svpino)">来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1245458528437801073)** (22 messages🔥): 

- **Mojo Nightly 版本发布更新**：新的 Mojo nightly 构建版本 `2024.5.3005` 已发布。此次更新包含一些关键更改，例如从 `String` 中删除了 `Stringable` 构造函数，以及删除了几个 `math` 函数等。提供了 [Raw diff](https://github.com/modularml/mojo/compare/fadceb1d7612bd0499f7280554f8ea5d774fcdef...8ae83916ebc7b3134948f466c0f56ee3e5569062) 和 [当前的 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **Nightly 与已发布构建版本**：一位 Mojo 官方人员表示，目前约有 25% 的 Mojo 安装来自 nightly 构建。这一决定是为了保持简单性，以免经验不足的用户因进入错误的分支而感到困惑。

- **删除 Stringable 构造函数的影响**：用户报告了由于从 `String` 中删除 `Stringable` 构造函数而导致的错误。建议的解决方案是改用 `str`。

- **公共 Mojo 仓库的 CI 修复**：公共 Mojo 仓库的持续集成（CI）已修复。之前的回归问题是由 `String.strip()` 的更改引起的，已在 [GitHub Pull Request #2883](https://github.com/modularml/mojo/pull/2883) 中解决。

- **`__setitem__` 与 List 容量的行为**：用户讨论了对列表的 `capacity` 使用 `__setitem__` 不会更新列表的长度（length）。相反，建议使用 `append` 向列表添加元素。

**提到的链接**：<a href="https://github.com/modularml/mojo/pull/2883.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1245689971906449428)** (5 messages): 

- **免费层级谜团仍未解开**：一位成员对另一位用户如何获得服务的免费层级表示好奇。对话似乎尚未解决，没有提供更多细节。
  
- **MixMyAI 发布公告引发用户关注**：对 [mixmyai.com](https://mixmyai.com) 进行了全面介绍，称其为 *"满足所有 AI 需求的一站式解决方案"*。主要特点包括无月费、价格最低、注重隐私的操作、强大的 UI 以及对多种 AI 模型的支持。
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1245469815074983967)** (45 messages🔥): 

- **开发者寻求机会**：一位用户介绍自己是资深全栈、区块链和 AI 开发者，具有开发网站、dApps 和 AI 项目的经验，询问是否有人在寻找开发人员。

- **用户在使用免费模型时遇到困难**：一位名为 *best_codes* 的用户报告了免费模型无法工作的问题并寻求帮助。情况随后似乎得到了解决，因为他们确认模型现在可以正常工作了。

- **Gemini 1.5 Pro Ratelimit 已明确**：一位用户询问了 Gemini 1.5 Pro 的 Ratelimit，另一位用户澄清说，虽然文档中的默认值是 15 RPM，但他们最近成功协商到了更高的限制，这表明自定义账户限制是可能的。

- **Moderated 与 Self-Moderated 模型**：讨论明确了 Self-Moderated 模型没有外部审核，而 Moderated 模型在端点上使用外部审核模型，在处理之前过滤输入。这主要适用于 OpenRouter 上的 Claude。

- **Laravel 和 Ruby 软件包发布公告**：两位开发者分别宣布了将 OpenRouter 集成到 Laravel 和 Ruby 项目中的软件包，并寻求社区的支持和贡献，分享了 [laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) 和 [open_router](https://github.com/OlympiaAI/open_router) 的 GitHub 链接。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/moe-mizrak/laravel-openrouter">GitHub - moe-mizrak/laravel-openrouter: Laravel package for OpenRouter (A unified interface for LLMs)</a>: 用于 OpenRouter 的 Laravel 软件包（LLMs 的统一接口） - moe-mizrak/laravel-openrouter</li><li><a href="https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email">MPT-7B and The Beginning of Context=Infinity — with Jonathan Frankle and Abhinav Venigalla of MosaicML</a>: 第 13 集：在 9 天内以 20 万美元训练 Mosaic 的 &quot;llongboi&quot; MPT-7B，如何为训练准备优质数据，以及开源模型的未来</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: 用于 OpenRouter API 的 Ruby 库。通过创建账户为 OlympiaAI/open_router 的开发做出贡献。</li><li><a href="https://github.com/OlympiaAI/raix-rails">GitHub - OlympiaAI/raix-rails: Ruby AI eXtensions for Rails</a>: 用于 Rails 的 Ruby AI 扩展。通过创建账户为 OlympiaAI/raix-rails 的开发做出贡献。</li><li><a href="https://openrouter.ai/docs#limits>)">Docs | OpenRouter</a>: 构建模型无关的 AI 应用
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1245502154798530630)** (37 messages🔥): 

- **用于站点限制的 Web 搜索 API**：一位成员澄清说，API 允许使用 options 对象将 Web 搜索连接器设置为特定域名。另一位用户询问是否可以同时限制在多个站点，目前尚待确认。
- **为大学构建 RAG**：一位参与者提到正在为他们的大学构建一个 Retrieval-Augmented Generation (RAG) 模型，旨在包含 .edu 域名和像 RateMyProfessors 这样的外部评论网站。
- **Embedding 转换咨询**：一位成员就如何将 uint8 Embedding 转换为 float 并返回以进行计算寻求建议，他们被引导至更适合讨论技术问题的频道。
- **初创公司寻求反馈**：一家初创公司的代表提供 10 美元的奖励，征求对其无代码 AI 工作流构建器的反馈，以了解用户在注册后流失的原因。该消息被要求移至更合适的频道。
- **Cohere 的重点已明确**：当被问及 Cohere 在 AI 行业的地位时，一名员工澄清说，虽然他们不专注于 Artificial General Intelligence (AGI)，但他们优先考虑创建适合生产的可扩展模型。

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

sssandra: 嗨，让我给你一些 Cohere 积分！正在私信。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1245471790440906843)** (34 messages🔥): 

- **使用 `ChatMessageHistory` 自动化对话存储**：Kapa.ai 提供了关于使用 LangChain 中的 `ChatMessageHistory` 类实现**存储和重新加载先前对话**的详细步骤和代码。解释中包括了存储、加载和清除消息的方法，以及将此功能与 `RunnableWithMessageHistory` Agent 集成的过程，并附有 [官方文档](https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory) 支持。

- **自动化 LLM 对话流中的挑战**：一位成员讨论了一个用于构建**非线性 LLM 对话流**的库，以及他们在高效提取值和状态方面面临的挑战。他们强调了在推理 Prompt 中使用 JSON 输出的问题，并询问了替代方法，并链接了一个相关的 [GitHub 实验](https://github.com/TonySimonovsky/prompt_engineering_experiments/blob/main/experiments/OpenAIAttentionGrab/OpenAI%20Attention%20Grab%20(report).ipynb)。

- **使用 LangChain 构建分析型 Copilot**：成员们分享了创建一个**与 PostgreSQL 数据库交互的分析型 Copilot** 的技巧和解决方案。建议包括实现用于处理 SQL 查询结果的自定义工具，以及使用 Few-shot Prompting 来处理模糊的用户查询。

- **结合 React 和 SQL Agent**：Kapa.ai 协助回答了关于在 LangChain 中集成 `create_react_agent` 与 `create_sql_agent` 的疑问。解决方案涉及创建具有指定名称和描述的工具，并提供了一个修复工具初始化中常见错误的示例。

- **扩展知识图谱能力**：一位社区成员请求协助**增强 LLMGraphTransformer**，以在节点和关系中包含协变量（covariates）。他们分享了受 Graph RAG 启发的方法论，并就如何修改 Prompt 和有效处理协变量寻求指导。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/modules/agents/quick_start/#adding-in-memory>)">快速入门 | 🦜️🔗 LangChain</a>：为了更好地理解 Agent 框架，让我们构建一个拥有两个工具的 Agent：一个用于在线查找信息，另一个用于查找我们加载到索引中的特定数据。</li><li><a href="https://github.com/langchain-ai/langchain/issues/20380>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#in-memory>)">添加消息历史（记忆） | 🦜️🔗 LangChain</a>：`RunnableWithMessageHistory` 让我们能够为某些类型的 Chain 添加消息历史。它包装了另一个 Runnable 并为其管理聊天消息历史。</li><li><a href="https://github.com/langchain-ai/langchain/issues/19904>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1245516576976011345)** (2 messages): 

- **使用 Pinecone、LangChain 和 OpenAI 构建与数据对话的功能**：[这个 YouTube 教程](https://www.youtube.com/watch?v=Bxj4btI3TzY) 逐步展示了如何利用 **Pinecone**、**LangChain** 和 **OpenAI** 创建一个聊天机器人。它面向初学者，并以作者的博客内容作为示例数据集。
  
- **Everything-ai v3.0.0 集成了 llama.cpp 和 Qdrant**：[Everything-ai v3.0.0](https://astrabert.github.io/everything-ai/) AI 助手现在支持 **llama.cpp**，并结合了 **Qdrant 支持的向量数据库**用于存储和查询文档。详细的设置说明已在其 [GitHub 仓库](https://github.com/AstraBert/everything-ai) 中提供，包括一个**基于 LangChain 的文档预处理流水线**，以确保上下文感知的响应。

**提到的链接**：<a href="https://www.youtube.com/watch?v=Bxj4btI3TzY">如何使用 Pinecone、LangChain 和 OpenAI 构建与数据对话的功能</a>：在这个易于理解的初学者教程中，我逐步展示了如何使用 Pinecone、LangChain 和 OpenAI 构建聊天机器人。我导入了我的整个博客，其中包含...

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

zackproser: https://www.youtube.com/watch?v=Bxj4btI3TzY
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1245455125955215421)** (23 messages🔥): 

- **关于 Flash 和价格变动的辩论**：一位成员指出，某项服务的价格变动发生在用户已经称赞其性价比之后。他们推测，“大多数人称赞 Flash 的性价比，推测是基于调价后的价格。”
  
- **关于 GPT-5 和免费 GPT-4o 的传闻**：在 X 上分享的一张表格暗示 GPT-5 即将到来，并伴随推测称 OpenAI 可能会让 GPT-4o 免费，以为其发布做准备。该表格被标注为“有趣（未证实）”，讨论中还引用了一位名为 Alan D. Thompson 的 AI 专家 [lifearchitect.ai 上的简介](https://lifearchitect.ai/about-alan/)。

- **OpenAI 价格修正**：X 上的一篇帖子解释说，OpenAI 价格的初始发布版本存在拼写错误，并在 24 小时内得到了修正。修正后的价格现在是准确的预期价格 [LoganK 的帖子](https://x.com/officiallogank/status/1796044236594278544?s=46)。

- **微软压力下的 OpenAI 发展方向**：有讨论涉及 OpenAI 内部的紧张局势，提到其最大的支持者微软如何向公司施压，要求其更多地关注商业产品，从而与倾向于科学研究的人员产生冲突 [FT 文章](https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075)。

- **对 OpenAI-Apple 交易的反应**：社区对 OpenAI 与 Apple 合作的消息做出了反应，引发了关于 Azure 计算额度是否会支持此次部署，以及它将如何与 Apple 的用户数据政策衔接的推测。一位用户幽默地想知道“Satya 是否因为这笔交易不是跟微软谈的而感到恼火”，并讨论了更广泛的战略影响。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_arohan_/status/1796228607255396423">来自 rohan anil (@_arohan_) 的推文</a>：@giffmana 别担心，今天又有一个集群到货了，以防你想再训练一些。</li><li><a href="https://x.com/officiallogank/status/1796044236594278544?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@artificialguybr 嘿！现在的定价页面是准确的，最初发布时价格有一个拼写错误（为了 I/O 发布进行的最后冲刺），我们在大约 24 小时后修复了它，我们显示的价格是...</li><li><a href="https://x.com/VachanReddyK/status/1795099977766551828/photo/1">来自 VACHAN (@VachanReddyK) 的推文</a>：GPT-5 👀</li><li><a href="https://x.com/amir/status/1795959410340049036?s=46">来自 Amir Efrati (@amir) 的推文</a>：微软并不一定喜欢 OpenAI 与 Apple 的结盟。https://www.theinformation.com/articles/openai-ceo-cements-control-as-he-secures-apple-deal?utm_source=ti_app&rc=c48ukx</li><li><a href="https://www.ft.com/content/ccbdff7c-ede3-4d62-968d-189fb0205075">11 月未遂政变后 OpenAI 内部的分歧依然存在</a>：无描述</li><li><a href="https://lifearchitect.ai/about-alan/)">关于 Alan</a>：AI 顾问，前门萨国际（天才家庭）主席，前 Sir James Dyson 家族办公室、PwC、Glencore 顾问……前 Sir Andrew Lloyd Webber、Debbie R... 首席音响师。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245806878777282570)** (9 messages🔥): 

- **前 OpenAI 董事会成员批评监管及相关事件**：[Helen Toner 和 Tasha McCauley](https://archive.is/rwRju) 在《经济学人》的 *By Invitation* 栏目中发表了关于 AI 监管和 OpenAI 事件的评论，但未透露具体细节。他们审视了导致 CEO Sam Altman 辞职的过程，该过程经过了 WilmerHale 的外部审查。
- **Text Davinci-003 在 ChatGPT 之后发布**：关于 GPT-3 迭代发布时间线的讨论指出，**Text Davinci-003** 是在 ChatGPT 发布之后推出的，而 -002 被认为不足以支持聊天机器人功能。
- **GPT-3.5 的混淆与误导信息**：成员们认为，说“任何人都可以用现有的 GPT-3.5 构建 ChatGPT”是不准确的。他们还提到 GPT-3.5 模型的命名方案，尤其是围绕 -002 和 -003 的部分，一直令人困惑。

**提到的链接**：<a href="https://archive.is/rwRju">OpenAI 董事会成员回应前成员的警告</a>：无描述

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1245451815517163552)** (21 条消息🔥): 

- **OpenInterpreter 文档获得推荐**：*"指定要使用的语言模型。查看 [models](https://docs.openinterpreter.com/settings/all-settings#auto-run) 部分以获取可用模型列表。"* 分享了一个指向 [LiteLLM](https://github.com/BerriAI/litellm) 的链接：它支持超过 100 多个模型。
- **对 Interpreter 移动客户端的兴奋**：一位用户提到了用于该解释器的 Android/iOS 客户端，并分享了 [GitHub 链接](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile)。另一位用户表示期待将其与他们的 RayNeo X2 和 Brilliant Labs 镜框配合使用。
- **本地 LLM 的发热问题**：关于在强力配置上运行 **LLaMA** 等本地模型的讨论，一位用户指出他们堆叠的 NVLinked 3090 产生了巨大的热量。其他人则强调使用 **Groq** 等服务来获取免费模型访问权限。
- **语音作为 TTS 的咨询**：*"嗨，有什么办法可以把我的声音作为 TTS 吗？"* 一位用户询问关于集成自己的声音以实现文本转语音（text-to-speech）功能的问题。
- **发货咨询被重定向**：*"发货了吗？我是在 2024 年 4 月 30 日上午 8:06:23 下单的...嗯...只是想问问？"* 另一位用户被引导去查看特定频道中的置顶消息，以获取生产进度更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile">01/software/source/clients/mobile at main · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账户为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#auto-run">All Settings - Open Interpreter</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1245474458047746069)** (9 条消息🔥): 

- **对 M5 Cardputer 更新的热情**：成员们对 M5 Cardputer 表示兴奋，其中一人肯定道 *"等不及看这个的更新了！超级令人兴奋"*。随着用户期待消费级设备中比开发套件更高质量的组件，期待感不断增加。
  
- **ChatTTS 的学术免责声明**：分享了一个指向 [Hugging Face 上的 ChatTTS](https://huggingface.co/2Noise/ChatTTS) 的链接，并附带免责声明，指出该信息 *"仅用于学术目的"*。它仅供教育使用，不用于商业或法律目的。

- **M5 Cardputer 的置顶生产更新**：通过引用一条带有生产更新的置顶消息，缓解了关于 M5 Cardputer 可能是“圈钱项目”的担忧。成员们强调了开发者沟通的重要性。 

- **对 Codestral 模型的兴趣**：一位成员询问是否有人已经尝试过 Codestral，并表示它 *"看起来是个不错的模型"*。这激发了其他成员的好奇心和潜在测试。

**提到的链接**：<a href="https://huggingface.co/2Noise/ChatTTS">2Noise/ChatTTS · Hugging Face</a>：未找到描述

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1245461054864228372)** (17 条消息🔥): 

- **OpenAI 扩展 ChatGPT 免费版功能**：成员们讨论了 ChatGPT 免费版新增的 browse、vision、data analysis、文件上传和 GPTs 功能，并指出“rate limits”可能是一个限制因素。[OpenAI 的公告](https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww) 详细介绍了这些新功能。
- **A16Z 关于语音 AI 的投资逻辑**：一位成员分享了 [a16z 的新投资逻辑](https://x.com/omooretweets/status/1795834644732285402)，核心围绕对话式语音 Agent 以及 AI 彻底改变电话沟通的潜力。一些用户对区分真正的技术进步与投资炒作表示怀疑。
- **Cartesia 发布状态空间语音模型**：[Cartesia 发布了 Sonic](https://x.com/cartesia_ai/status/1795856778456084596)，这是一款低延迟生成式语音模型，旨在跨设备集成实时多模态智能。此次发布及其对 AI 的潜在影响引发了讨论。查看他们的 [博客文章](https://cartesia.ai/blog/sonic) 并在此处体验 [Sonic](https://play.cartesia.ai/)。
- **YC 澄清 Sam 的离职**：Paul Graham 在 [一条推文](https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 中澄清了 Sam 从 Y Combinator 离职的真相，回应了围绕他退出的误解。
- **用于检索的 Embedding Adapters**：关于 [TryChroma 技术报告](https://research.trychroma.com/embedding-adapters) 的讨论强调了 Embedding Adapters 在提高检索性能方面的潜力。另一位成员指出了这与 Vespa 处理 frozen embeddings 方法的相似之处，并链接了 [相关的 Vespa 博客文章](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/">Leveraging frozen embeddings in Vespa with SentenceTransformers</a>: 如何在 Vespa 中使用 SentenceTransformers 库实现 frozen embeddings 方法，并同时优化你的搜索应用。</li><li><a href="https://x.com/omooretweets/status/1795834644732285402">来自 Olivia Moore (@omooretweets) 的推文</a>: 🚨 新的 @a16z 投资逻辑！是时候让 AI 重塑电话通话了 —— 欢迎对话式语音 Agent 📱 我们感兴趣的投资方向 + 市场图谱（来自我和 @illscience）👇</li><li><a href="https://x.com/cartesia_ai/status/1795856778456084596">来自 Cartesia (@cartesia_ai) 的推文</a>: 今天，我们很高兴发布实现为每个设备构建实时多模态智能使命的第一步：Sonic，一个极速（🚀 135ms 模型延迟）、逼真的生成式语音模型...</li><li><a href="https://x.com/paulg/status/1796107666265108940?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Paul Graham (@paulg) 的推文</a>: 我听腻了 YC 开除 Sam 的传闻，所以这里是事实经过：</li><li><a href="https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 OpenAI (@OpenAI) 的推文</a>: 所有 ChatGPT 免费用户现在都可以使用 browse、vision、data analysis、文件上传和 GPTs。引用 OpenAI (@OpenAI)：我们正在开放对新旗舰模型 GPT-4o 的访问，以及 browse 等功能...</li><li><a href="https://x.com/cartesia_ai/">GitHub - FixTweet/FxTwitter 推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://research.trychroma.com/embedding-adapters">Embedding Adapters</a>: 未找到描述</li><li><a href="https://buttondown.email/ainews/archive/ainews-sonic-a-low-latency-voice-model-for/">[AINews] 1 万亿 token 上下文，实时，在设备上？</a>: SSMs is all you need。2024/5/28-5/29 的 AI 新闻。我们为你检查了 7 个 subreddits、384 个 Twitter 账号和 29 个 Discord（389 个频道，5432 条消息）....
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1245806790734774425)** (5 条消息): 

- **关于训练百万级上下文 LLM 的新播客**：[新一期播客节目](https://x.com/latentspacepod/status/1796247856891969709)已上线，标题为“如何训练百万级上下文的 LLM！”。本期邀请了 @markatgradient 讨论如何将 Llama-3 扩展到 1M+ 上下文，并实现近乎完美的 NIAH 评估结果。节目还涵盖了长上下文的历史、RoPE、ALiBi、Ring Attention 以及各种 NIAH 变体。

**提到的链接**：<a href="https://x.com/latentspacepod/status/1796247856891969709">来自 Latent Space Podcast (@latentspacepod) 的推文</a>：🆕 播客：如何训练百万级上下文的 LLM！@ylecun 说我们应该要么发表，要么出局。我们邀请了 @markatgradient 来透露他团队如何将 Llama-3 扩展到 1M+ 上下文并获得近乎完美的 @G... 的所有细节。

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1245452118362423409)** (2 条消息): 

```html
<ul>
    <li><strong>No messages to summarize</strong>: The channel "llm-paper-club-west" currently holds no substantial messages that can be summarized. Only placeholders are present without any actual content to analyze.</li>
</ul>
```
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1245805942046724270)** (1 条消息): 

- **LLM360 启动成员组织的活动**：LLM360 开启了首个由成员主导的活动，包括[关于其新 65B 模型和开源工作的 AMA](https://discord.com/events/1089876418936180786/1240722407594004561)。这一举措标志着 Mozilla AI 开始推行社区驱动的互动活动。
- **湾区即将举行的活动**：对于身处湾区的成员，已发布了一个[线下开源 Hack Lab](https://discord.com/events/1089876418936180786/1243732435674337413)活动。鼓励成员点击“感兴趣”进行预约。
- **使用 llamafiles 的 Embeddings 演示**：一位著名的社区成员将主持一场[关于使用 llamafiles 进行 embeddings 的演示](https://discord.com/events/1089876418936180786/1242590711778381914)。该活动承诺深入探讨 embeddings 在机器学习中的实际应用。
- **Amplifying Devs 活动**：[一场名为“Amplifying Devs”的会议](https://discord.com/events/1089876418936180786/1242653066512175157)将由开发者版主进行讨论。重点将放在支持 Mozilla AI 社区内的开发者。
- **关于 GenAI Bug 赏金的 AMA**：[由 0din 发起的新 AMA](https://discord.com/events/1089876418936180786/1245800040086245416)将探讨 GenAI Bug 赏金。参与者 Saoud Khalifah 和另一位社区成员将揭示这一新兴话题。
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1245516861966389402)** (19 条消息🔥): 

- **在 M2 Studio 上运行 LlamaFile 出错**：一位成员在 M2 Studio 上尝试运行 `granite-34b-code-instruct.Q5_0.llamafile` 时收到 `error: unknown argument: --temp` 错误，并寻求修复方案。
- **LlamaFile 连接被拒绝**：另一位成员在 Python 中构建 VectorStoreIndex 时遇到“连接被拒绝”错误，当时 llamafile.exe 运行在 8080 端口。建议尝试将 LlamaFile 绑定到 `0.0.0.0` 而不是 `127.0.0.1` 以排查 IP 地址问题。
- **WSL Localhost 问题已解决**：该成员发现 WSL 对“localhost”的定义映射不正确，通过指定 WSL 特定的以太网 IP 地址解决了连接问题。
- **寻求 LlamaFiles 的视觉/图像支持**：一位成员询问哪里可以找到支持视觉/图像的 LlamaFiles，并分享了 [Hugging Face 上 Mozilla 的 llava-v1.5-7b-llamafile](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main) 链接，该文件可能支持此类功能，由 jartine 提交。

**提到的链接**：<a href="https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/tree/main">Mozilla/llava-v1.5-7b-llamafile at main</a>：未找到描述内容。

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1245471263212961913)** (10 条消息🔥): 

- **寻求 LLM 微调建议**：成员们正在询问如何针对**图像和视频理解**微调 Llava 等大语言模型 (LLMs)。该问题仍处于征求建议阶段。
- **DPO 数据需求担忧**：关于 **DPO** (Direct Preference Optimization) 与 **SFT** (Supervised Fine-Tuning) 数据需求的讨论展开。一位成员表示担心，如果 DPO 需要与 SFT 同样多的数据，那么从一开始就确保 SFT 具有高质量数据可能会更直接。
- **小数据集用于 DPO 的可行性**：有人好奇**数百个样本**是否足以进行有效的 DPO，特别是在涉及通用闲聊的领域。一位成员表示，手工构建这样一个数据集可能是可行的。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1245460020993200149)** (4 条消息): 

- **寻求具有 Protobuf 经验的后端开发**：一位成员正在寻找从事后端工作且具有 **Google Protobuf** 经验的人员。他们提到愿意为专业知识付费，并明确表示对逆向工程师、恶意软件分析师或 Bug 猎人感兴趣。
- **DPO VRAM 占用之谜**：另一位成员观察到 **DPO VRAM 占用**显著减少。他们询问是否发生了更新，因为他们的配置没有改变，但 VRAM 占用突然减半。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1245515666899861545)** (1 条消息): 

- **SDXL 模型优化寻求帮助**：一位成员正在寻求帮助以**优化 SDXL 模型**，用于生成定制产品广告。他们提到尝试过 LoRA 训练但结果不尽如人意，并请求任何有 **SDXL 微调和 ControlNet** 经验的人提供帮助：*"如果你有这些方面的经验，或者认识相关人员，请私信。"*
  

---



### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1245835549294461041)** (1 条消息): 

- **Rosebud AI 举办的新 Game Jam**：Roberto 宣布了一个**新的 Game Jam：“Book to Game”**，参与者可以在 AI Game Maker 平台上使用 Phaser JS 将文学作品转化为互动游戏。该活动设有 **500 美元的奖金池**，提交截止日期为 7 月 1 日。 

- **加入 Rosebud AI 的第三届 Game Jam**：鼓励参与者将任何形式的文学作品（从小说到同人小说）改编成引人入胜的游戏。更多详情可以通过他们的 [Twitter 公告](https://x.com/Rosebud_AI/status/1796273820044595368) 了解，并加入他们的 [Discord 服务器](https://discord.gg/rosebud-ai)。

**提及的链接**：<a href="https://x.com/Rosebud_AI/status/1796273820044595368">来自 Rosebud AI 🌹 (@Rosebud_AI) 的 Rosie 的推文</a>：使用 AI 将你最喜欢的故事变成游戏！📚 👾 为我们的第三届 Game Jam 做好准备：“Book to Game”。使用 Rosebud Game Maker 将文学作品转化为互动游戏，并将故事带入...

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1245831648553996347)** (9 条消息🔥): 

- **新成员加入并遇到 Android 端问题**：一位新成员提到他们刚加入，发现 Android 手机上的导航*"有点难……有故障且多 Bug"*。他们澄清说，尽管有这些问题，他们仍然可以与世界互动。
- **关于更改用户名的困惑**：这位新成员询问小组*"如何更改用户名？"*，并承认在平台上感觉像个*"外星人"*。
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1245499010454786109)** (3 条消息): 

- **对近期 GPU 发展的思考**：一位成员对 GPU 的未来表示好奇，思考它们在 **5 年（甚至 2 年）**后会是什么样子。另一位成员开玩笑地建议：*“做一个更大的脉动阵列 (systolic array) 😌。”*
- **64x64 MMA 暗示**：在后续讨论中，一位成员暗示了未来 GPU 设计中 **64x64 矩阵乘法阵列 (MMA)** 的潜力。
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1245541195254599690)** (6 messages): 

- **Tinygrad 轻松处理整数梯度**：一位成员指出 **Torch** 不允许整数变量拥有梯度，会导致类似 `RuntimeError: Only Tensors of floating point and complex dtype can require gradients` 的错误。而在 **Tinygrad** 中，同样的代码可以毫无问题地计算整数的梯度。
- **与整数反向传播相关的行为**：另一位成员表示 **Tinygrad** 的计算方式就像 Tensor 是浮点数一样，然后将其转换为原始 dtype。这使得 Tinygrad 与其他框架有所区别。
- **Tinygrad 优越性主张**：一位成员热烈宣称 **Tinygrad** 优于 **TensorFlow** 和 **PyTorch**，并对其评价最高。然而，这引发了关于为什么 TensorFlow 被认为优于 PyTorch 的疑问。
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1245785057839415317)** (2 messages): 

- **按语言拆分 Codestral 引发好奇**：一位成员想知道如果将 **Codestral** 按单个编程语言拆分，体积是否会更小。他们质疑 **Python 训练是否增强了 JS 模型**，并考虑使用 Mixture of Experts 方法，其中每个 Expert 对应不同的语言。
- **权重主要以英语为主**：另一位成员表示赞同，假设 **模型的大部分权重是基于英语的**，而每种编程语言只贡献了较小的一部分。他们对这个 **45GB 模型** 的分布表示好奇。
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/)** (1 messages): 

_awesomewaffle: 明天将参加在 Netflix 举办的 PRS 活动。还有其他人参加这个活动吗？
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1245472460971970580)** (1 messages): 

- **LAION 呼吁构建开源 GPT-4-Omni**：[LAION](https://laion.ai/notes/open-gpt-4-o/) 发布了一篇博客文章，寻求帮助以构建开源的 GPT-4-Omni，并概述了包含数据集和教程的极具前景的方向。该倡议邀请社区参与以丰富该项目。

**提到的链接**：<a href="https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19">来自 LAION (@laion_ai) 的推文</a>：帮助我们构建开源 GPT-4-Omni！通过这篇博客文章，我们展示了极具前景的方向（包括数据集和教程）https://laion.ai/notes/open-gpt-4-o/

{% else %}

> 完整的频道分类明细已针对电子邮件进行了截断。
> 
> 如果您想查看完整明细，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！


{% endif %}

---

如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！