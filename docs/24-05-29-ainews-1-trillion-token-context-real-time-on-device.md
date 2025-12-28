---
companies:
- cartesia
- mistral-ai
- scale-ai
date: '2024-05-29T23:01:07.584364Z'
description: '专注于**状态空间模型 (SSM)** 的初创公司 **Cartesia** 发布了一款低延迟语音模型。该模型的表现优于基于 Transformer
  的模型，其**困惑度（perplexity）降低了 20%**，**词错率降低了 2 倍**，且 **NISQA 质量评分高出 1 分**。这一突破凸显了此类模型的潜力，即能够在设备端利用**万亿级
  Token 的上下文窗口**，对大规模多模态数据流（文本、音频、视频）进行持续处理和推理。


  此外，新闻还涵盖了近期 AI 领域的一系列进展，包括 **Mistral 发布 Codestral 权重**、**Schedule Free 优化器**论文发布，以及
  **Scale AI** 推出的全新 Elo 风格评估排行榜。同时，文中还提到了 **杨立昆 (Yann LeCun)** 与 **埃隆·马斯克 (Elon Musk)**
  之间关于“发表 AI 研究”与“工程成就”孰轻孰重的辩论。最后，**Gemini 1.5 Pro/Advanced** 模型也因其强劲的表现被提及。'
id: d9902b7c-cdb3-4e85-8ad1-44856e5da308
models:
- gemini-1.5-pro
- gemini-1.5
original_slug: ainews-sonic-a-low-latency-voice-model-for
people:
- yann-lecun
- elon-musk
title: 1万亿 token 上下文，实时，端侧运行？
topics:
- state-space-models
- voice-models
- multimodality
- model-performance
- on-device-ai
- long-context
- evaluation-leaderboards
- learning-rate-optimization
- scientific-publishing
- research-vs-engineering
---

<!-- buttondown-editor-mode: plaintext -->**SSMs 是你所需的一切。**

> 2024年5月28日至5月29日的 AI 新闻。
我们为你查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务（**389** 个频道，**5432** 条消息）。
预计节省阅读时间（以 200wpm 计）：**553 分钟**。

**我们今日头条故事的候选名单**：

- [祝 GPT3 四岁生日快乐](https://x.com/alexandr_wang/status/1795589516314734911?utm_source=ainews&utm_medium=email)！
- [你好 Codestral](https://x.com/MistralAILabs/status/1795820935540584909)。权重已根据 Mistral 非商业许可证发布，评测表现不错，支持 80 种语言，但进一步细节较少。
- [Schedule Free 优化器](https://arxiv.org/abs/2405.15682)来了！[我们在 2 个月前报道过这些](https://buttondown.email/ainews/archive/ainews-adamw-aarond/)，现在论文已经发布——[评判团正在评估](https://x.com/aaron_defazio/status/1795435679339700238/quotes)，但目前看来情况良好——如果能够扩展，这可能是学习率优化领域的一篇具有变革意义的论文。
- [Scale AI 推出了自己的 Elo 风格评测排行榜](https://x.com/alexandr_wang/status/1795857651592491281?utm_source=ainews&utm_medium=email)，包含私有、持续更新的领域专家评测，涵盖编程、数学、指令遵循和多语言（西班牙语），延续了他们在 [GSM1k](https://x.com/alexandr_wang/status/1795857658760802514) 上的类似工作。

但今天我们将胜利（W）授予 **Cartesia**，这是一家由另一位 Mamba 共同作者创立的 State Space Models 初创公司。他们今天发布了传闻已久的 [低延迟语音模型](https://x.com/cartesia_ai/status/1795856864472871330)，轻松击败了对应的 Transformer 模型（[困惑度降低 20%，词错误率降低 2 倍，NISQA 质量提高 1 分](https://x.com/cartesia_ai/status/1795856807744909799)）：

 
![image.png](https://assets.buttondown.email/images/1881bef3-2e86-421f-b838-7373e64dc3c0.png?w=960&fit=max)
 

Loss 图表中巨大的差距证明了这一点：

 
![image.png](https://assets.buttondown.email/images/0075874b-8bc4-41b8-9ae9-e3f05720ebba.png?w=960&fit=max)
 

这是日益增多的可用 **State Space Models** 中最新的一款，[发布公告](https://cartesia.ai/blog/sonic)讨论了极高效实时模型所开启的愿景：

> 即使是最好的模型也无法持续处理和推理长达一年的音频、视频和文本流：**10 亿文本 token、100 亿音频 token 和 1 万亿视频 token**——更不用说在设备端（on-device）运行了。难道不应该让每个人都能获得不需要调度数据中心的廉价智能吗？

以及[超快设备端 TTS 效果的预览](https://x.com/cartesia_ai/status/1795856864472871330)。

看到可用的 SSMs 现身并能切实挑战 SOTA 令人备受鼓舞（我们尚未看到与 ElevenLabs 等的对比，但作为 ElevenLabs 的资深用户，我们在 Cartesia Playground 上的抽查结果非常具有说服力）。

但将 SSMs 与当前的 SOTA 进行比较，忽略了上述引用文字中所表达的宏大野心：**如果你知道我们很快就能拥有可以持续处理和推理文本/音频/视频，且拥有“万亿级” token “上下文窗口”的模型，你会做出什么不同的选择？而且是在设备端？**

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Yann LeCun 与 Elon Musk 关于 AI 研究与工程的辩论**

- **发表研究的重要性**：[@ylecun](https://twitter.com/ylecun/status/1795589846771147018) 认为，研究要被称为科学，必须发表并提供足够的细节以供复现，强调了同行评审和共享科学信息对技术进步的重要性。
- **基于已发表科学成果的工程壮举**：有人认为 Elon Musk 和 SpaceX 等公司正在通过工程推动技术进步，而并不总是发表论文。[@ylecun](https://twitter.com/ylecun/status/1795659135146405952) 反驳称，这些工程壮举在很大程度上是基于已发表的科学突破。
- **科学与工程的区别**：讨论引发了关于科学与工程之间差异及其互补性的辩论。[@ylecun](https://twitter.com/ylecun/status/1795840305075220635) 阐明了这两个领域在课题、方法论、出版物和影响力方面的区别。

**大语言模型 (LLMs) 与 AI 能力的进展**

- **Gemini 1.5 模型的强劲表现**：[@lmsysorg](https://twitter.com/lmsysorg/status/1795512202465845686) 报告称，**Gemini 1.5 Pro/Advanced 在其排行榜上排名第 2**，几乎追平 GPT-4，而 Gemini 1.5 Flash 排名第 9，超越了 Llama-3-70b 和 GPT-4-0125。
- **Codestral-22B 代码模型发布**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1795820710750744839) 宣布发布 **Codestral-22B**，该模型在 80 多种编程语言上进行了训练，性能优于以往的代码模型，并可通过 API 获取。
- **用于图像生成视频的 Veo 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1795788432750280796) 推出了 **Veo，它可以根据单张参考图像并遵循文本提示指令创建视频片段**。
- **用于前沿模型评估的 SEAL 排行榜**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1795857651592491281) 推出了**针对前沿模型的私人专家评估**，重点关注不可利用且持续更新的基准测试。
- **GPT-3 发布 4 年后的 Scaling 洞察**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1795589516314734911) 反思了自 GPT-3 论文以来的进展，指出未来 4 年将是算力和数据的指数级 Scaling，这代表了我们时代最大规模的基础设施项目。

**研究论文与技术**

- **用于训练 Transformers 的 Schedule-Free 平均法**：[@aaron_defazio](https://twitter.com/aaron_defazio/status/1795435679339700238) 及其合作者发表了一篇论文，介绍了用于训练 Transformers 的 Schedule-Free 平均法，与标准的学习率调度（learning rate schedules）相比，显示出强劲的结果。
- **用于显存高效 LLM 训练的 VeLoRA**：一篇新论文提出了 VeLoRA，这是一种使用 rank-1 sub-token 投影进行 LLM 微调和预训练的显存高效算法。(https://twitter.com/_akhaliq/status/1795651536497864831)
- **在线与离线对齐算法之间的性能差距**：Google 的一篇论文研究了为什么用于对齐 LLM 的在线 RL 算法优于离线算法，结论是在线策略采样（on-policy sampling）起到了关键作用。(https://twitter.com/rohanpaul_ai/status/1795432640050340215) 
- **Transformers 通过特殊嵌入学习算术**：[@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1795508276903252311) 展示了 Transformers 可以通过使用特殊的编码位置嵌入（positional embeddings）来学习加法和乘法等算术运算。

**梗与幽默**

- [@svpino](https://twitter.com/svpino/status/1795503047004594637) 开玩笑说某个特定评论区的娱乐价值很高。
- [@Teknium1](https://twitter.com/Teknium1/status/1795835058546503894) 幽默地建议，OpenAI 本周的举动只能通过发布 "waifus"（纸片人老婆）来挽救。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型开发**

- **Gemini 1.5 Pro 表现优于大多数 GPT-4 实例**：在 LMSYS Chatbot Arena 排行榜上，Gemini 1.5 Pro [**击败了除 4o 以外的所有 GPT-4 实例**](https://i.redd.it/zg81rujc363d1.png)。这突显了开源 AI 模型的快速进步。
- **Abliterated-v3 模型发布**：Phi 模型的去审查版本，包括 [Phi-3-mini-128k 和 Phi-3-vision-128k 已发布](https://www.reddit.com/r/LocalLLaMA/comments/1d2vdnf/abliteratedv3_details_about_the_methodology_faq/)，扩大了获取强大 AI 能力的途径。
- **Llama3 8B Vision 模型媲美 GPT-4**：一款新的多模态模型 [Llama3 8B Vision 模型已发布，其视觉理解能力与 GPT4V 和 GPT4o 持平](https://github.com/mustafaaljadery/llama3v)。
- **Gemini Flash 和更新后的 Gemini 1.5 Pro 加入排行榜**：LMSYS Chatbot Arena 排行榜已 [更新，加入了 Gemini Flash 和改进版的 Gemini 1.5 Pro](https://i.redd.it/76p0dn07x73d1.png)，展示了持续的迭代。

**AI 安全与伦理**

- **公众对 AI 伦理的担忧**：一项民意调查显示，[**超过半数的美国人认为 AI 公司在开发技术时没有充分考虑伦理，且近 90% 的人支持政府监管**](https://www.techtarget.com/searchenterpriseai/news/366586214/AI-companies-losing-public-trust-in-safety)。这凸显了公众对负责任 AI 开发日益增长的担忧。

**AI 工具与应用**

- **HuggingChat 增加工具支持**：[HuggingChat 现在集成了 PDF 解析、图像生成、网页搜索等工具](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/470)，扩展了其作为 AI 助手的职能。
- **CopilotKit v0.9.0 发布**：一个 [用于构建应用内 AI Agent 的开源框架，CopilotKit v0.9.0 支持 GPT-4o、原生语音和 Gemini 集成](https://github.com/CopilotKit/CopilotKit)，使 AI 驱动的应用开发更加简便。
- **WebLLM Chat 实现浏览器内模型推理**：[WebLLM Chat 允许在 Web 浏览器中本地运行 Llama, Mistral, Hermes, Gemma, RedPajama, Phi 和 TinyLlama 等开源 LLM](https://github.com/mlc-ai/web-llm-chat)，使模型访问更加便捷。
- **LMDeploy v0.4.2 支持视觉语言模型**：最新版本的 LMDeploy [支持 llava, internvl, internlm-xcomposer2, qwen-vl, deepseek-vl, minigemini 和 yi-vl 等 VL 模型的 4-bit 量化和部署](https://www.reddit.com/r/LocalLLaMA/comments/1d32li8/vision_language_model_quantization_and/)，促进了高效的多模态 AI 开发。

**AI 硬件**

- **在改装的 2080ti GPU 上运行 Llama3 70B**：通过 [将 2 张 2080ti GPU 改装为每张 22GB VRAM，可以在此配置上运行 Llama3 70B 模型](https://v.redd.it/74ovedojka3d1)，展示了大模型推理的创意解决方案。
- **4x GTX Titan X Pascal 12GB 配置运行 Llama3**：利用 [4 张 GTX Titan X Pascal 12GB GPU 提供的总计 48GB VRAM，可以使用 Q3KM 量化运行 Llama3 70B](https://i.redd.it/6pvdj7swl53d1.jpeg)，展示了旧硬件的潜力。
- **SambaNova 的 Samba-1 Turbo 运行 Llama-3 8B**：[SambaNova 展示了其 Samba-1 Turbo AI 硬件运行 Llama-3 8B 模型](https://i.redd.it/amdshn8jab3d1.jpeg)，突显了高效推理的专业化解决方案。

**AI 争议与八卦**

- **Sam Altman 过去的争议**：据透露，[Sam Altman 曾被 Y Combinator 解雇，其初创公司 Loopt 的员工曾因其混乱且具有欺骗性的行为要求董事会解雇他](https://open.spotify.com/episode/4r127XapFv7JZr0OPzRDaI?si=3c025c435b194109) ([图片](https://i.redd.it/4o05u99eta3d1.jpeg))，揭示了这位 OpenAI CEO 的过往。
- **Yann LeCun 与 Elon Musk 的交锋**：在一次公开讨论中，[Elon Musk 对 Yann LeCun 的科学成就给出了无力的反驳](https://i.redd.it/70er5d5m553d1.png)，凸显了 AI 先驱之间的紧张关系。

**梗图与幽默**

- [Nvidia 梗图](https://i.redd.it/cmpjemzi4a3d1.jpeg) 
- [“我觉得它们有亲缘关系”比较 AI 模型的梗图](https://i.redd.it/s0m7w3fsh53d1.jpeg)
- [冷脸表情包梗图 🥶](https://i.redd.it/6x172qqjd63d1.jpeg)

---

# AI Discord 摘要

> 摘要之摘要的摘要

1. **LLM 性能与实际应用**：

   - 来自 Google 的 **[Gemini 1.5 Pro/Advanced 模型](https://x.com/lmsysorg/status/1795512202465845686)** 以顶尖的排行榜名次令人印象深刻，表现优于 **Llama-3-70b** 等模型；而来自 MistralAI 的 **Codestral 22B** 支持 **80 多种编程语言**，目标受众为 AI 工程师。

- Mistral AI 的新 **[Codestral 模型](https://mistral.ai/news/codestral)** 是一款采用非商业许可证的权重开放模型，引发了关于开源可访问性与商业可行性之间平衡的讨论。**[Codestral](https://huggingface.co/mistralai/Codestral-22B-v0.1)** 在 80 多种编程语言上进行了训练，其简化编程任务的潜力激发了人们的热情。
- Scale AI 推出的 **[SEAL 排行榜](https://scale.com/leaderboard)** 等项目因设定了 AI 评估的新标准而受到关注，尽管也有人对由于供应商关联可能导致的评估者偏见表示担忧。

- 普林斯顿大学的 **[SWE-agent](https://github.com/princeton-nlp/SWE-agent)** 因其卓越的性能和开源特性引起了兴趣，而 **Llama3-V** 作为一个较小的模型，因挑战 GPT4-V 而备受关注。

- **检索增强生成 (RAG)** 模型正在不断演进，出现了如 [PropertyGraphIndex](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms) 这样用于构建丰富知识图谱的工具，同时 [Iderkity 支持](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) 高效的翻译任务。

2. **微调、提示工程与模型优化**：

- 工程师们讨论了 **[梯度累积 (Gradient Accumulation)](https://github.com/google-research/tuning_playbook)** 和 **DPO 训练** 方法，强调了 `ref_model` 在微调期间保持一致性的作用，并探讨了用于不同系统高效使用的 [量化库](https://huggingface.co/blog/hf-bitsandbytes-integration)。

- 分享了解决 **提示工程 (Prompt Engineering)** 挑战的技术，例如使用 try/except 结构处理 "RateLimit" 错误，以及针对特定领域微调模型，并强调了实际解决方案（[示例](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/client.ipynb)）。

- 成员们辩论了 **Transformer 与 MLP 的使用**，[强调了相关发现](https://arxiv.org/abs/2405.15618)，即 MLP 在处理某些任务时可能表现更好，并讨论了在持续微调工作中的模型特定问题，如上下文长度和优化器配置。

3. **开源贡献与 AI 社区协作**：

- **OpenAccess AI Collective** 处理了垃圾信息问题，提议更新 Unsloth 中的梯度检查点 (gradient checkpointing)，并见证了社区主导的关于微调 LLM 以进行图像和视频内容理解的倡议。

- **LlamaIndex** 通过合并到 Neo4j 生态系统为开源做出了贡献，重点是集成 [PropertyGraphIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/) 等工具，以提供强大的知识图谱解决方案。

- 讨论强调了围绕 [Llama3 模型训练](https://twitter.com/eugeneyan) 的社区努力，以及在 GitHub 上为 **axolotl** 和 **torchao** 等库提交的协作 issue，这表明了持续的发展和共同的问题解决。

4. **模型部署与基础设施问题**：

- 工程师们努力解决 **Google Colab 断连**、部署中的 **Docker** 配置问题，以及在 NVIDIA A6000 GPU 上使用 **Triton** 内核的 [性能优势](https://github.com/UmerHA/triton_util/)。

- 推荐使用 **Lighting AI Studio** 以获取免费 GPU 时长，同时关于为大型模型生产力分配 [GPU 资源](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py) 以及应对硬件瓶颈的讨论凸显了用户的挑战。

- 讨论了 **ROC** 与 NVIDIA 的兼容性障碍，并提出了克服这些障碍的实用建议，例如寻求 **7900 XT** 的交易以扩展 VRAM 配置，从而支持更大的模型以及从 macOS x86 到 M1 的迁移。

5. **AI 领域的挑战、反应与创新**：

- **Helen Toner 对 OpenAI 管理层的爆料** 引发了关于透明度的辩论，引发了对内部政治和伦理 AI 开发的担忧（[播客链接](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3)）。

- [Elon Musk 的 xAI](https://x.ai/blog/series-b) 获得 60 亿美元融资，引发了关于 AI 竞争力和基础设施投资影响的讨论，同时社区成员辩论了模型定价策略及其对技术长期投资的潜在影响。

- **[Cohere API](https://huggingface.co/CohereForAI/c4ai-command-r-plus#grounded-generation-and-rag-capabilities)** 引发了关于如何有效进行有据生成 (grounded generation) 并确保强制引用显示的讨论，展示了社区在利用新模型处理实际用例方面的积极参与。

# 第 1 部分：Discord 高层摘要

---

{% if medium == 'web' %}

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **网页抓取心得**：讨论强调了高效提取网页内容的方法，包括 **Python requests**、**Playwright**，以及针对 JavaScript 密集型网站的 **Gemini 1.5 Flash**。

- **Perplexity API 的困扰与进展**：工程师们对 **Perplexity API** 响应与 Web 端应用准确性之间的一致性表示担忧，并考虑选择不同的模型（如 **llama-3-sonar-small-32k-online**）来潜在地提升性能。

- **构建 Perplexity 的竞争产品**：提出了一个镜像 **Perplexity 多模型查询**功能的详细项目，但在扩展性和后端开发方面面临挑战。

- **顺应 Go 潮流**：对 **Go 编程语言**进行了深入探讨，展示了其在网页抓取应用中的有效性，强调了其在可扩展性和并发方面的优势。

- **优势分析**：用户分享了涵盖 AI 生成内容、查询合理性澄清以及优缺点全面评估的 **Perplexity AI 搜索链接**。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BERT 的 Token 限制促使用户寻求解决方案**：一位用户正在评估处理超出 **BERT (512 tokens)** 和 **decoder-based models (1,024 tokens)** 限制的文档的方法。他们的目标是绕过文档切片和位置嵌入（positional embedding）调整，且不求助于昂贵的新预训练。

- **Diffusers 庆典与 GPT-2 情感分析的成功**：Hugging Face 社区庆祝 Diffusers 项目成立两周年，同时发布了一个用于情感分析的新 **FineTuned GPT-2** 模型，该模型实现了 **0.9680 的准确率和 F1 分数**。该模型针对 Amazon 评论进行了微调，可在 [Hugging Face](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0) 上获取。

- **读书小组期待 C4AI 的见解**：一个新的论文读书小组已准备就绪，渴望加入来自 C4AI 社区的演讲，重点是揭穿低资源语言中的虚假信息。下一次活动链接见[此处](https://discord.com/events/879548962464493619/1245408612818620426)。

- **图像处理咨询引导用户获取资源**：讨论涵盖了使用 **YOLO** 等模型以及 **convNext** 和 **DINOv2** 等新替代方案处理大图像的最佳实践。一个关于 Hugging Face 图像处理教程的 GitHub 仓库被重点推荐（[Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma)）。

- **医学影像寻求 AI 助力**：社区成员就创建一个用于分析无标签 MRI 和 CT 扫描的自监督学习框架交换了意见。讨论包括利用预训练模型提取的特征进行特定类别的分割任务。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Lightning AI 与 L4 显卡的强强联手**：由于 **Lightning AI Studio** 提供“每月约 20 小时免费时长”，且 L4 GPU 的性能优于 Colab 的 T4 GPU，用户推荐使用该平台。有人提议与 Lightning AI 进行潜在合作以造福社区。

- **Phi3 与 Llama3 的性能难题**：讨论显示了对 **Phi3** 模型的褒贬不一，一些人认为 `phi-3-medium` 的表现不如 **llama3-8b**。一位用户指出，在超过 2048 tokens 的上下文长度后，Phi3 的表现逊于 Llama3。

- **模型部署讨论升温**：社区交流了利用 **Runpods** 和 Docker 部署模型的想法，部分成员在服务商使用上遇到了问题。虽然没有提供具体的 Dockerfile，但建议在服务器中搜索相关内容。

- **Colab Premium 未达预期**：Google Colab 的 Premium 服务因持续的断连问题面临批评。成员建议转向 **Kaggle** 和 **Lightning AI** 等其他平台作为可行的免费替代方案。

- **Unsloth 在本地开发中的实践**：用户开始使用 Unsloth 进行监督微调（SFT），讨论了在本地运行模型，特别是在 **VSCode** 中执行简历要点生成等任务。分享了使用 Unsloth 进行无监督微调的 Colab 笔记本和 GitHub 资源链接，例如此[微调指南](https://github.com/unslothai/unsloth#-finetune-for-free)和 [Colab 示例](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning 的挫折与市场思考**：工程师们讨论了 **fine-tuning** 的挑战，包括对 Google Gemini 1.5 API 价格上涨的担忧，以及在生产环境中提供 fine-tuned 模型的困难。提议设立一个专门针对 LLM 相关工作机会的 **channel**，并强调了对强大的 **JSON/Parquet 文件处理工具** 的需求。

**技术研讨会的内幕**：参与者交流了关于 **LLM fine-tuning 策略** 的见解，重点是个性化销售电子邮件和法律文档摘要。辩论了 **multi-agent LLM 协作** 的实用性以及 Stable Diffusion 的 prompt 优化。

**探索 AI 生态系统**：社区深入探讨了各种 AI 话题，揭示了 **Braintrust** 是评估非确定性系统的便捷工具，以及 **O'Reilly Radar** 对构建 LLM 复杂性的见解。讨论还强调了 **Autoevals** 在 SQL 查询评估方面的潜力。

**LLM 工作工具箱**：工程师们解决了实际问题，如 **Modal 的不透明故障** 和 *Axolotl 预处理* GPU 支持问题。分享了关于在 **Jarvislabs** 上使用共享存储的查询，以及对 **Wing Axolotl** 模型量化的见解，讨论中穿插了有用的资源和技巧。

**代码、工艺与社区**：社区氛围活跃，讨论了 LLM *evaluator models*、Gradio UI 优于 Streamlit 的吸引力，以及从 **San Diego** 到 **NYC** 的聚会。充满活力的交流涵盖了技术领域，也培育了 AI 工程领域的社交纽带。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPGPU 编程拥抱 lighting.ai**：工程师们讨论了 **lighting.ai** 作为 GPGPU 编程的一个值得推荐的选择，特别是对于那些无法使用通常用于 CUDA 和 SYCL 开发的 NVIDIA 硬件的人。

**简化 Triton 开发**：开发者发现 [triton_util](https://github.com/UmerHA/triton_util)（一个简化 Triton kernel 编写的工具包）在抽象重复性任务方面非常有用，提升了直观体验。观察到在 NVIDIA A6000 GPU 上使用 Triton 的性能飞跃，而在处理超过 65GB 的大型张量时，解决 Bug 成了关注焦点。

**Nightly Torch 支持 Python 3.12**：PyTorch 社区强调了 Python 3.12 上的 **torch.compile** 问题，nightly 构建版本提供了一些解决方案。同时，Torch 2.3 中弃用 macOS x86 构建引发了关于过渡到 M1 芯片或 Linux 的讨论。

**Tom Yeh 增强 AI 基础**：
[Prof Tom Yeh](https://x.com/ProfTomYeh) 因分享 AI 概念的手算练习而受到关注。他的系列包括 [Dot Product](https://x.com/ProfTomYeh/status/1793623127643037891)、[Matrix Multiplication](https://x.com/ProfTomYeh/status/1794070094898704456)、[Linear Layer](https://x.com/ProfTomYeh/status/1794451228681712037) 和 [Activation](https://x.com/ProfTomYeh/status/1794848226383655284) 练习册。

**量化领域的飞跃**：工程师们正积极讨论并使用 **bitsandbytes** 和 **fbgemm_gpu** 等库改进量化过程，并参加 NeurIPS 等竞赛。分享了关于 **Llama2-7B** 和 **FP6-LLM** 仓库更新的努力，同时也赞赏了 torchao 社区的支持性氛围。

**CUDA 调试技能提升**：分享了一个关于调试 SYCL 代码的询问，强调了对改进 kernel 代码分析工具的需求，甚至可能涉及单步调试过程。

**通过 bitnet PR 加速开发**：
bitnet 频道解决了各种技术问题，包括与 PyTorch/dev 版本和 CUDA 不匹配相关的 `ImportError` 挑战，以及通过 **gcc 12.1** 升级解决的大学服务器编译难题。讨论了关于 bit packing 和 CI 改进的协作 PR，并提供了位级操作和错误解决的资源（[GitHub 上的 BitBlas](https://github.com/microsoft/BitBLAS)，[ao GitHub issue](https://github.com/pytorch/ao/issues/288)）。

**柏林与西雅图的社交与科技故事**：闲聊频道的对话对比了西雅图和柏林的社交与天气景观。柏林因其 techno 音乐场景和初创公司友好度而受到推崇，尽管也有其阴郁的天气。

**Tokenizer 故事与训练谈话**：随后进行了关于自行实现 tokenizer 和数据集处理的广泛对话，考虑了压缩和云存储选项。在 H100 GPU 上进行大规模训练仍然成本过高，而关于 GPU 规格的细致讨论为模型优化提供了参考。训练实验继续快速进行，其中一个实验的效果类似于 GPT-3。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**玩转大上下文 (Big Contexts)**：一位工程师建议使用*极长的 context window* 来训练 Large Language Model (LLM)，其理念是只要有足够的上下文，即使数据集较小，LLM 也能做出更好的预测。

**无偏评估的困境**：人们对 [Scale 的参与](https://scale.com/leaderboard) 表示担忧，因为它既提供数据又评估机器学习模型，这凸显了潜在的利益冲突，可能会影响模型评估的公正性。

**超越基础理解 RAG**：技术讨论阐明了 **Retrieval-Augmented Generation (RAG)** 系统的复杂性，强调它不仅仅是 vector similarity 匹配，还涉及一系列其他过程，如 re-ranking 和全文搜索，正如 [RAGAS](https://github.com/explodinggradients/ragas) 等讨论和资源所强调的那样。

**价格翻倍，担忧也翻倍**：Google 决定提高 Gemini 1.5 Flash 输出价格的举动引发了激烈辩论，工程师们指责这种定价策略不可持续，并对 API 成本结构的可靠性提出质疑。

**梯度累积 (Gradient Accumulation) 审查**：讨论中出现了一个关于在模型训练中避免梯度累积的话题，工程师们参考了 [Google 的 tuning playbook](https://github.com/google-research/tuning_playbook) 以获取见解，同时还根据 [Hugging Face 的文档](https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model-considerations-with-peft) 讨论了 DPO 训练中 `ref_model` 的概念。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **开源还是闭源？LM Studio 的二分法**：LM Studio 的主应用程序被确认为**闭源**，而 **LMS Client (CLI)** 和 **lmstudio.js (新 SDK)** 等工具是开源的。LM Studio 内的模型无法直接访问本地 PC 文件。

- **翻译模型热议**：[Aya 日译英模型](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF) 被推荐用于翻译任务，而支持 80 多种编程语言的 **Codestral** 引发了将其集成到 LM Studio 的讨论。

- **GPU 选择与性能讨论**：关于**多 GPU 设置**与单个强力 GPU 优劣的辩论浮出水面，特别是对 **Nvidia** 股票的价值和**改装 GPU 的实用性**提出质疑。一位 **Goldensun3ds** 用户升级到了 **44GB VRAM**，展示了该配置的优势。

- **Server 模式拖慢表现**：用户注意到，在预设相同的情况下，chat 模式比 server 模式获得结果的速度更快，这引发了对 **GPU 利用率**以及 server 模式操作中 **GPU 选择**必要性的担忧。

- **AMD GPU 用户面临 ROCm 障碍**：记录了 **LM Studio 与 Radeon GPU** 的版本问题，包括尝试使用 **iGPU** 和 **ROCm 模式下的多 GPU 配置**失败。分享了 **7900 XT** 的优惠信息，作为扩展 VRAM 的可能解决方案。

- **单个 AI 能否身兼数职？**：对于一个模型同时执行内容审核和问答的可行性提出了质疑，建议指向使用两个独立的模型，或利用 server 模式以获得更好的上下文处理能力。

- **Codestral 可用性发布**：Mistral 新发布的 **22B 编程模型 Codestral** 已经发布，目标用户是寻求强大编程助手的拥有大容量 GPU 的用户。该模型可在 [Hugging Face](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF) 下载。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 开启内存回忆之旅**：一篇博客文章阐述了 Mojo 以 **Ownership** 为核心的内存管理方法，倡导一种既安全又高性能的编程模型。[Chris Lattner 的视频](https://www.modular.com/team/chris-lattner)被推荐为深入研究 Mojo 编译器系统中 **Ownership** 概念的资源。欲了解更多信息，请阅读其[博客文章](https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach)。

**对齐的优势 (Alignment Ascendancy)**：工程师们强调了表中 **64-byte alignment** 的重要性，以充分发挥 **AVX512** 指令的效能并提高缓存效率。他们还强调了对齐对于触发预取器（prefetcher）最佳性能的必要性，以及多线程环境中的 **false sharing** 问题。

**Mojo 中的 Optional 困境与 Dict 难题**：在 `nightly` 分支的对话中，在 `ref` API 中使用 `Optional` 引发了广泛讨论，参与者将 Rust 的 `?` 运算符作为建设性的对比。相关的 GitHub [issue](https://github.com/modularml/mojo/issues/2869) 还聚焦于 `InlineArray` 无法调用其元素析构函数的 Bug。

**提案与编译的篇章**：关于自动解引用（auto-dereferenced）引用中的命名规范展开了激烈辩论，有人提议将 `Reference` 重命名为 `TrackedPointer`，将 `Pointer` 重命名为 `UntrackedPointer`。此外，最新的 nightly Mojo 编译器版本 `2024.5.2912` 带来了诸如异步函数借用限制等更新，并提供了详尽的 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

**AI 拓展开放世界游戏视野**：有人断言，如果 AI 能够根据用户交互，从广泛的在线模型中动态构建世界，开放世界游戏将达到新的巅峰。这一想法暗示了 AI 在游戏进步中发挥作用的重大机遇。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **为 AI 新手提供帮助**：EleutherAI 的新成员（包括一名即将毕业的 CS 学生）获得了入门级的研究课题和资源，如 [GitHub gist](https://gist.github.com/ad8e/da8fdfe0ec586b5a548aaa14327f7722)。由于缺乏基础 AI 问答平台，这引发了关于初学者获取 AI 知识便捷性的讨论。

- **过早发表论文困扰同行**：一篇因在缺乏实验支持的情况下提出大胆主张而引起社区关注的论文引发了讨论。人们对其在 **arXiv** 上的接受情况提出了质疑，与之形成对比的是，大家认可了 Yann LeCun 富有影响力的指导，以及他强调工程与基础科学区别的专题[讲座](https://youtu.be/gG5NCkMerHU)。

- **MLP vs Transformer —— 潮流转向**：关于最近发现 **MLP** 在 **In-context learning** 方面可以与 **Transformer** 媲美的研究引发了热烈辩论。虽然对 **MLP** 的潜力很感兴趣，但对其优化和通用性仍存在普遍怀疑，成员们参考了 [MLPs Learn In-Context](https://arxiv.org/abs/2405.15618) 等资源，讨论也反映了对 AI 架构演进中“**Bitter Lesson**”的反思。

- **AMD Traceback 内存计算错误**：一名成员在尝试计算 AMD 系统上的最大内存时遇到了 Traceback 错误，并通过 [GitHub Gist](https://gist.github.com/jonabur/0004bf39a3cec65262cf72f556c316c4) 分享了该问题；而另一名成员则在寻求关于使用 **lm-evaluation-harness** 进行并发查询和基于 **logits** 测试的建议。

- **扩展讨论转向支持 MLP**：对话显示，优化技巧可能会掩盖性能不足，同时突出了一个观察结果：可扩展性和适应性可能会掩盖 **MLP** 的结构缺陷。分享的链接包括一项[比较 CNN、Transformer 和 MLP 网络的实证研究](https://arxiv.org/abs/2108.13002#microsoft)以及一项关于[扩展 MLP (scaling MLPs)](https://arxiv.org/abs/2306.13575) 的调查。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **免费用户喜迎新功能！**：ChatGPT 的免费用户现在可以享受更多功能，包括 **browse**、**vision**、**data analysis**、**file uploads** 以及访问各种 **GPTs**。

- **ImaGen3 引发复杂情绪**：围绕 Google 即将发布的 **ImaGen3** 讨论纷纷，主要集中在对媒体操纵和信任问题的怀疑。同时，Google 还因历史图像生成中的准确性错误而面临指责。

- **GPT-4 的记忆问题亟待解决**：工程师们对 GPT-4 断断续续的“失忆症”表示不满，希望能有更透明的记忆机制，并建议增加一个用于长期记忆保存的 **backup button**。

- **RAM 占用上升：用户呼吁优化**：对 RAM 消耗过高的担忧激增，特别是在 Brave 等浏览器上使用 ChatGPT 时；建议的替代方案包括使用 Safari 或桌面应用以获得更流畅的体验。

- **共享 Prompt 的中心枢纽**：对于那些寻找“惊人 Prompt”库的人，请关注 Discord 社区中专门为此设立的频道。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral 进入编程领域**：[Codestral](https://mistral.ai/news/codestral/) 是来自 Mistral 的一款新型 22B 模型，精通 80 多种编程语言，现已发布并在 8 周测试期内可在 [HuggingFace](https://huggingface.co/mistralai/Codestral-22B-v0.1) 上访问。与此同时，Scale AI 推出的基于私有数据的 [LLM leaderboard](https://scale.com/leaderboard) 引发了关于模型评估潜在偏见的讨论，原因在于该公司的盈利模式及其对固定众包人员的依赖。

- **涨价让 Gemini 1.5 Flash 的欢呼声戛然而止**：Google 的 Gemini 1.5 Flash 输出价格突然上调——从 $0.53/1M 涨至 $1.05/1M——就在其发布备受赞誉之后，这引发了关于 API 稳定性和可靠性的争论。

- **OpenAI 董事会的尴尬博弈**：根据前董事会成员 Helen Toner 的披露，OpenAI 董事会是在 Twitter 上才得知 ChatGPT 发布的消息。这一事件揭示了 OpenAI 内部更广泛的透明度问题，而 Sam Altman 被解雇时缺乏明确理由（董事会称其为“沟通不始终坦诚”）进一步加剧了这一问题。

- **Toner 的爆料与 OpenAI 的不透明成为讨论焦点**：Toner 指控 Sam Altman 领导下的 OpenAI 经常存在不诚实行为，这引发了对其披露时机的辩论，有人猜测存在法律约束，并承认内部政治和压力可能塑造了董事会的叙事。

- **深度学习社区的知识盛宴**：知识交流活动正变得越来越受欢迎，例如组建“mini journal club”以及对 **Cohere 教育视频系列**的赞赏，而 **TalkRL podcast** 被认为价值被低估。尽管 Schulman 在 Dwarkesh 的播客节目中对 AI safety 的务实看法评价褒贬不一，但 [Andrew Carr 的推文](https://x.com/andrew_n_carr/status/1782878279504191896)中提到的旨在减少 AI 错误行为的层级模型正引起关注。

- **对 FMTI 文件处理方式的沮丧**：社区对 FMTI GitHub 仓库选择 CSV 而非 Markdown 格式感到不满，这阻碍了工程师轻松获取论文评分。

- **SnailBot 即将发布**：随着标签提醒，SnailBot News 更新的期待感正在增加，Nate Lambert 也引起了大家对即将推出的贴纸的好奇。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Colab 和 Kaggle 加速图像生成**：工程师建议使用 **Kaggle** 或 **Colab** 来加快 Stable Diffusion 的图像生成速度；据报告，在 Colab 上使用 **16GB VRAM 每张图像需要 1.5 到 2 分钟**。
  
- **训练 SDXL LoRA 模型的技巧**：技术爱好者讨论了 **Stable Diffusion XL LoRA** 模型的训练，强调 2-3 个 epoch 即可获得良好效果，并建议触发词（trigger words）越简洁，训练效果越好。
  
- **配置 ComfyUI 模型路径与 API 集成**：社区成员正在解决 **ComfyUI** 多个模型目录的配置问题，并讨论在本地 Stable Diffusion API 中集成 **ADetailer**。

- **HUG 与 Stability AI 课程安排**：有关于 **HUG 和 Stability AI** 合作提供创意 AI 课程的讨论，课程配有录像供后续观看——填写反馈表后将退还参与者的押金。

- **3D 模型生成仍处于孵化阶段**：对话转向 AI 在创建适用于打印的 **3D 模型** 方面的作用，成员们一致认为当前 AI 生成这些模型的能力尚未达到预期。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **绘制 LLM 知识图谱**：[LlamaIndex 发布了 PropertyGraphIndex](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms)，这是与 Neo4j 的合作成果，允许构建更丰富的 LLM 支持的知识图谱。该工具具备图提取和查询功能，支持自定义提取器以及向量/图联合搜索功能——用户可以参考 [PropertyGraphIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/) 获取指南。

- **优化知识检索**：讨论集中在通过实验文本分块（chunk）大小来优化 RAG 模型，并参考 [SemanticDocumentParser](https://github.com/isaackogan/SemanticDocumentParser) 生成高质量分块。此外，还分享了最大化向量存储潜力的策略，如提到的 `QueryFusionRetriever`，以及非英语 Embedding 的最佳实践，引用了 [asafaya/bert-base-arabic](https://huggingface.co/asafaya/bert-base-arabic) 等资源。

- **Codestral 时代的创新**：LlamaIndex 支持来自 MistralAI 的新 [Codestral 模型](https://t.co/k2nHDiMnwD)，涵盖 80 多种编程语言，并可通过 [Ollama](https://t.co/gsPHHF4c0K) 等工具增强本地运行能力。此外，[FinTextQA 数据集](https://t.co/emhQYXY1S4) 为基于金融文档的查询提供了大量的问答对。

- **使用 Document Stores 进行存储和定制**：社区讨论了在 LlamaIndex 中管理文档节点和存储，涉及 `docstore.persist()` 的功能以及不同文档后端的利用，参考了 [Document Stores - LlamaIndex](https://docs.llamaindex.ai/en/latest/module_guides/storing/docstores/)。交流中还提到了 [Simple Fusion Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion/) 作为管理向量存储索引的解决方案。

- **跨越边界的查询**：发布的 Property Graph Index 强调了 LlamaIndex 扩展知识图谱查询能力的承诺，集成了处理节点和关系的标签（labels）及属性（properties）的功能。[LlamaIndex 博客](https://t.co/X9D3Wl0Hyv) 阐明了这些进展及其对 AI 工程领域的潜在影响。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 1.5 证明了其实力**：根据 [LMSysOrg 的 Twitter](https://x.com/lmsysorg/status/1795512202465845686) 分享的结果，**Gemini 1.5 Pro/Advanced** 目前位居第二，逼近 GPT-4o，而 **Gemini 1.5 Flash** 排名第九，超越了 Llama-3-70b 等模型。

- **SWE-agent 引起关注**：普林斯顿大学的 **SWE-agent** 因其卓越的性能声明和开源状态引发了热议，详细信息可见 [Gergely Orosz 的 Twitter](https://x.com/GergelyOrosz/status/1794743519954731331) 和 [SWE-agent GitHub](https://github.com/princeton-nlp/SWE-agent)。

- **Llama3-V 加入竞争**：新的开源模型 **Llama3-V** 尽管体积较小，但仍能与 GPT4-V 竞争，[Sidd Rsh 的 Twitter](https://x.com/siddrrsh/status/1795541002620727439) 详细介绍了这一备受关注的模型。

- **LLM 实战经验谈**：文章 "[What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)" 探讨了一年来使用 LLM 工作的见解和经验，重点关注构建 AI 产品的演进与挑战。

- **SCALE 通过 SEAL 排行榜设定 LLM 基准测试标准**：**Scale 的 SEAL 排行榜**已发布，用于进行稳健的 LLM 评估，并得到了 [Alexandr Wang](https://x.com/alexandr_wang/status/1795857651592491281) 和 [Andrej Karpathy](https://x.com/karpathy/status/1795873666481402010) 等行业人物的赞赏。

- **预订 Latent Space 的虚拟席位**：今天宣布将举办一场技术活动，探讨 **AI Agent 架构与 Kolmogorov Arnold Networks**，[点击此处注册](https://lu.ma/pxnaq641)。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 临时停机已解决**：OpenAI 面临临时服务中断，但在快速修复后已恢复正常运行，Alex Atallah 指出 Azure 服务在整个事件期间保持正常运行。

- **告别 Cinematika**：由于使用率低，**Cinematika 模型**将被弃用；建议用户尽快切换到替代模型。

- **资金上限困扰已修复**：在 OpenAI 模型因意外触及支出限制而无法访问后，相关解决方案已实施并恢复了正常服务，同时推出了额外的安全防护措施。

- **GPT-4o 上下文容量确认**：针对 Token 限制的误解，Alex Atallah 表示 GPT-4o 保持 128k Token 的上下文限制，以及独立的 4096 输出 Token 上限。

- **对 GPT-4o 图像提示词性能的担忧**：一位用户在使用 `openai/gpt-4o` 处理 `image-url` 输入时体验到处理缓慢，这暗示可能存在性能瓶颈，可能需要进一步的调查与优化。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 影响力人物备受关注**：Helen Toner 关于在 Twitter 上发现 ChatGPT 的评论引发了对话；而 Yann LeCun 在卸任 Facebook VP 后的研究活动引起了兴趣，标志着 AI 领袖在塑造社区观点方面的持续影响力。相比之下，Elon Musk 仅在 AI 模型失去竞争优势时才公开的做法，引发了关于 AI 开发中开源模型策略的讨论。

- **Mistral 的许可证利用 Open Weights**：在讨论中，Mistral AI 的许可策略因其在非商业框架下结合 Open Weights 而受到关注，强调了 AI 模型共享与商业化之间复杂的格局。

- **模型生成复杂性**：在模型生成中使用看似简单的提示词（如“一个女人在看书”）时会出现困难，用户报告在合成标注（synthetic caption）创建中存在负面影响，暗示了生成式 AI 领域持久的挑战。

- **关于判别器有效性的探讨**：社区剖析了研究资料，特别注意到 Dinov2 作为判别器（Discriminator）的使用，但表现出对修改后的预训练 UNet 的偏好，这让人联想到类似于 Kandinsky 的策略，即通过减半的 UNet 来提高性能，揭示了 AI 研究中不断演进的判别器技术。

- **社区对评分激励机制的怀疑**：关于 Horde AI 社区激励 SD 图像评分系统的讨论引发了质疑，有人提到此类计划可能会降低数据质量，突显了社区参与度与数据完整性之间常见的张力。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain v2.0 Agents 查找问题已解决**：用户最初在查找 **LangChain v2.0** 中的 Agent 时遇到困难，随后讨论并成功定位及实现了上述 Agent。
- **关于 AI 与创造力的见解引发讨论**：一条推文建议 AI 应超越重复，迈向真正的创造力，这引发了关于 AI 在创意领域潜力的技术讨论。
- **解决 LangChain 中的 "RateLimit" 错误**：社区分享了处理 **LangChain** 应用中 "RateLimit" 错误的解决方案，提倡使用 **Python 的 try/except 结构** 进行稳健的错误管理。
- **优化对话式数据检索**：成员在处理多个 Vector Stores 时面临 **ConversationalRetrievalChain** 的挑战，寻求关于有效合并数据以实现完整内容检索的建议。
- **持久化聊天能力的实际演示**：一位频道成员测试了 **langserve** 的持久化聊天记录功能，参考了仓库中的示例，并询问如何将 "chat_history" 整合到 FastAPI 请求体中，相关文档见 [此处](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/client.ipynb)。

关于使用 **LangChain** 在 Agent 流中实现路由逻辑的教学内容已通过 [YouTube 教程](https://youtu.be/KtbRexZ6vsc) 发布，帮助社区成员增强其自动化 Agent 的决策路径。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **训练工作流中定制化为王**：工程师们对*个性化训练工作流*表现出浓厚兴趣，讨论集中在针对个人用例增强 Open Interpreter，表明 AI 工具对定制化有显著需求。
- **用户分享 Open Interpreter 应用**：Open Interpreter 的各种用例引发了讨论，成员们就如何利用其功能进行不同的技术应用交换了意见。
- **寻找开源替代方案**：工程师之间的对话强调了对 Rewind 替代方案的持续探索，**Rem** 和 **Cohere API** 被提及为处理 Vector DB 的值得关注的选择。
- **Rewind 的连接性获得认可**：一位用户证实了 Rewind 的效率，称其为“生活黑客技巧”，尽管它在隐藏敏感数据方面存在不足，这反映了技术用户普遍持积极态度。
- **消除 OI 中的确认步骤**：针对效率问题，一位成员提供了运行 Open Interpreter 而无需确认步骤的解决方案，即使用 `--auto_run` 功能，详见 [官方文档](https://docs.openinterpreter.com/settings/all-settings#auto-run)。
- **M5 屏幕问题**：一位用户报告其 M5 在刷机后显示白屏，引发了故障排除讨论，包括建议更改 Arduino studio 设置以在刷机期间执行全内存擦除。
- **未说明的 YouTube 链接**：一位成员分享了一个孤立的 [YouTube 视频](https://www.youtube.com/watch?v=sqwtk18pw14) 链接且未提供上下文，可能错失了讨论或提供有价值见解的机会。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **“不适宜内容”垃圾信息清理**：**OpenAccess AI Collective (axolotl)** 的管理员迅速响应了关于 **NSFW Discord 邀请链接** 在各频道被大量散布的警报，垃圾信息已得到及时处理。
- **探索多媒体模型掌握**：在 *general* 频道中有人询问如何微调像 **LLava models** 这样的 **LLM** 以进行图像和视频理解，但该问题目前尚未得到解答。
- **为 MoE 提供 Gradient Checkpointing**：*axolotl-dev* 频道的一位成员提议更新 **Unsloth 的 Gradient Checkpointing** 以支持 **MoE architecture**，验证后将提交 Pull Request (PR)。
- **Bin Packing 算法排错**：一项开发更新指出 **Bin Packing** 算法得到了改进，但强调了一个问题：训练在评估后停滞，这可能与新采样器缺失 `_len_est` 实现有关。
- **采样器回滚引起关注**：通过分享一个 **[撤销 multipack batch sampler 更改的 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1672)** 指出了代码回退，原因是 Loss 计算存在缺陷，这表明在模型训练中精确指标评估的重要性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**重新思考使用 RAG 进行 PDF 微调**：一位成员提议使用 **Retrieval Augmented Generation (RAG)** 作为处理 PDF 的传统 JSONL 微调的更智能替代方案，声称它可以完全消除微调步骤。

**特定 API 的 Grounded Generation 见解**：引用 API 文档展示了如何在 **grounded generation framework** 中使用 `response.citations` 功能，并提供了一个 [Hugging Face 链接](https://huggingface.co/CohereForAI/c4ai-command-r-plus#grounded-generation-and-rag-capabilities) 作为参考。

**具有强制引用显示的本地 R+ 创新**：一位工程师分享了在本地 Command R+ 设置中集成**带有强制引用显示的 RAG 流水线**的实践成果，展示了一种维持源归属可靠性的方法。

**Cohere Discord 机器人使用强调了分段讨论**：围绕由 **Cohere** 驱动的 Discord 机器人的热情引发了一个提醒，即应将项目讨论保持在专用频道内，以维持社区讨论的秩序和专注。

**频道礼仪鼓励项目隔离**：在对社区构建的 Discord 机器人表示认可后，紧接着给出了将详细讨论移至指定项目频道的指导，以确保遵守服务器的组织规范。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**xAI 获得高达 60 亿美元融资**：Elon Musk 的 xAI 已成功[筹集了 60 亿美元](https://x.ai/blog/series-b)，知名投资者包括 Andreessen Horowitz 和 Sequoia Capital。这笔资金旨在用于初始产品的市场推广、大规模基础设施建设以及推进未来技术的研发。

**对未命名分析工具的怀疑**：一位频道成员对某些分析工具表示怀疑，认为它们的“实用性微乎其微”，尽管他们没有具体说明哪些工具受到了质疑。

**新语言 Bend 受到关注**：Bend 编程语言因其“无需任何代码即可自动多线程”的能力而受到赞誉，这一特性与 tinygrad 的延迟执行（lazy execution）策略相辅相成，正如 [Fireship 视频](https://www.youtube.com/channel/UC0v-tlzsn0QZwJnkiaUSJVQ)中所示。

**tinybox 电源供应查询**：有人提出了关于 tinybox 电源要求的问题，询问它是使用“两个消费级电源还是两个带有配电板的服务器电源”，但尚未提供解决方案。

**链接焦点**：The Verge 关于 xAI 融资的一篇文章特别询问了这笔资金中会有多少分配给购买 GPU，这是 AI 工程师对计算基础设施的一个关键关注点。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Goliath 需要训练辅助**：在进行额外的预训练之前，**Goliath** 经历了明显的性能下降，引发了社区成员之间的协作分析和应对。

- **经济高效地复制 GPT-2 里程碑**：工程师们在 [GitHub](https://github.com/karpathy/llm.c/discussions/481) 上讨论了仅用 20 美元就在 C 语言中实现 GPT-2 (124M) 的复制，并指出其 HellaSwag 准确率为 29.9，超过了 GPT-2 原始的 29.4 分。

- **Codestral-22B：多语言巨兽**：**Mistral AI** 发布了 **Codestral-22B**，这是一个在 80 多种编程语言上训练的庞然大物，据 [Guillaume Lample 的公告](https://x.com/guillaumelample/status/1795820710750744839?s=46&t=1jtkL4JPu-DUOdo8JC668g)称，它比前代产品更精通编程。

- **召集贡献者共同开发开源 GPT-4-Omni**：**LAION AI** 正在号召社区参与 GPT-4-Omni 的开源开发，并发布了一篇重点介绍数据集和教程的博客文章，可在此处访问：[here](https://laion.ai/notes/open-gpt-4-o/)。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Llamafile 在 Windows 上的烦恼**：一位工程师在 Windows 上**编译 llamafile** 时遇到了问题，指出 `cosmoc++` 存在一个问题，即由于可执行文件在没有 `.exe` 后缀的情况下无法启动而导致构建失败。尽管系统报告文件缺失，但工程师确认该文件存在于目录 `.cosmocc/3.3.8/bin` 中，并且在使用 cosmo bash 时也遇到了同样的问题。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **RAG 助力解决 LLM 幻觉**：一位工程师建议使用 **Retrieval Augmented Generation (RAG)** 来解决 **Language Models (LLMs)** 在回答文档查询时的幻觉问题。他们提议对 `llm` 命令进行扩展，以递归地为给定 URL 创建 embeddings，利用文档数据集和 embedding 存储来提高准确性。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**技术交流一瞥**：一位用户简要提到发现了一篇与其兴趣相关的论文，感谢他人的分享，并表示打算进行审阅。然而，未提供有关该论文内容、标题或研究领域的详细信息。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：按频道划分的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1245098566838522028)** (1007 条消息🔥🔥🔥): 

- **高效抓取网页内容**：成员们讨论了提取网页内容的各种方法，包括使用 **Python requests** 和 **Playwright**。有人建议使用 **Gemini 1.5 Flash** 来高效提取相关内容，尽管某些网站需要 JavaScript。
  
- **Perplexity API 响应问题**：用户抱怨 **Perplexity API** 提供的准确性不如 Web 应用。一些人怀疑存在**暗箱操作**，而另一些人则建议使用不同的模型和 API，如 **Groq** 和 **Openrouter**。

- **开发类似 Perplexity 的工具**：一位成员详细介绍了他们的项目，该项目通过使用**多模型查询**和**自定义搜索流水线**来模拟 **Perplexity** 的操作，以提供准确、及时的响应。他们讨论了扩展挑战以及为更广泛的基础设施构建后端。

- **功能和能力对比**：针对询问，成员们指出了该平台在各种用例中的**优势和局限性**，例如文档搜索和回答复杂的特定查询。一些人建议在处理以文档为中心的任务时，可以使用 **Adobe Acrobat’s AI chat** 和 **Google’s Notebook LM** 等替代方案。

- **技术深度探讨和 Go 编程**：对话转向了提高网页抓取和解析方法效率的 **Go 语言技术**。一位成员强调学习 Go 语言是为了在构建 AI 应用时获得更好的**可扩展性和并发性**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://console.anthropic.com/">Anthropic Console</a>: 未找到描述</li><li><a href="https://promptfoo.dev/">更快速地迭代 LLM | promptfoo</a>: 为您的用例量身定制的 LLM 评估。最大化模型质量并捕捉回归。</li><li><a href="https://abrahamjuliot.github.io/creepjs/">CreepJS</a>: 未找到描述</li><li><a href="https://pdf.ai/">PDF.ai | 与您的 PDF 文档对话</a>: 我们构建了终极的 ChatPDF 应用，允许您与任何 PDF 对话：提问、获取摘要、查找您需要的任何内容！</li><li><a href="https://tenor.com/view/oh-wah-ah-ah-ah-anthony-vincent-down-with-the-sickness-intro-singing-disturbed-gif-16261397">Oh Wah Ah Ah Ah Anthony Vincent GIF - Oh Wah Ah Ah Ah Anthony Vincent Down With The Sickness Intro - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://app.screencast.com/8BgvwUhLJLOYT">2024-05-29_11-44-22</a>: 来自 Snagit + Screencast by Techsmith 的全球领先屏幕截图和录像工具。无缝捕捉、编辑和分享专业品质的内容。</li><li><a href="https://github.com/projectdiscovery/katana">GitHub - projectdiscovery/katana: 下一代爬虫和蜘蛛框架。</a>: 下一代爬虫和蜘蛛框架。 - projectdiscovery/katana</li><li><a href="https://perplexity.typeform.com/pages-beta">Perplexity Pages - Beta 访问</a>: 使用 Typeform 将数据收集变成一种体验。创建美观的在线表单、调查、测验等。免费试用。</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: 将任何网站转换为 LLM 就绪的数据。</li><li><a href="https://ai.google.dev/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/">我们的下一代模型：Gemini 1.5</a>: Gemini 1.5 提供了显著增强的性能，在跨模态的长上下文理解方面取得了突破。</li><li><a href="https://news.itsfoss.com/openai-google-search/">OpenAI 计划通过其 AI 搜索引擎挑战 Google</a>: 随着新搜索引擎的推出，另一波由 ChatGPT 驱动的浪潮即将到来？</li><li><a href="https://artificialanalysis.ai/models/gemini-1-5-pro">Gemini 1.5 Pro - 质量、性能和价格分析 | Artificial Analysis</a>: 对 Google 的 Gemini 1.5 Pro 进行分析，并与其他 AI 模型在关键指标上进行比较，包括质量、价格、性能（每秒 token 数和首个 token 时间）、上下文窗口等。</li><li><a href="https://www.cnet.com/tech/services-and-software/google-gemini-pricing-1-5-pro-and-1-5-flash-compared/">Google Gemini 定价：1.5 Pro 和 1.5 Flash 比较</a>: 以下是如何决定哪款 Gemini 模型能为您提供最高性价比的方法。</li><li><a href="https://beebom.com/how-use-gemini-1-5-flash/">Gemini 1.5 Flash 是一个被低估的珍宝，您现在就需要尝试：方法如下</a>: Gemini 1.5 Flash 在 Google I/O 2024 的热潮中被忽视了，但它凭借快速推理、多模态和 100 万 token 支持展现了强大的实力。</li><li><a href="https://www.selectiveasia.com/japan-holidays/weather/may">日本 5 月天气 - 温度、气候、最佳旅游时间 | Selective Asia</a>: 未找到描述</li><li><a href="https://top.his-usa.com/destination-japan/blog/a_guide_to_japan_-_may_and_june.html">日本指南 - 5 月和 6 月</a>: 日历、活动以及关于 5 月和 6 月的日本</li><li><a href="https://www.japan-guide.com/e/e2273.html">旅游时间</a>: 哪些季节适合去日本旅游？去日本旅游的最佳时间是什么时候？</li><li><a href="https://www.holiday-weather.com/tokyo/averages/may/">东京，日本 5 月天气</a>: 日本东京 5 月平均天气</li><li><a href="https://indianexpress.com/article/explained/explained-sci-tech/google-gemini-pro-1-5-1-million-tokens-9166398/">拥有 100 万 token 的 Gemini Pro 1.5 超越了 GPT-4 Turbo：这意味着什么？</a>: 说到 Gemini 1.5 Pro，Google 似乎推出了一款优于并显著领先于其前代产品的模型。Gemini 1.5 Pro 是 Gemini 1.5 系列中的第一款模型...</li><li><a href="https://www.cnet.com/tech/services-and-software/googles-gemini-1-5-pro-will-have-2-million-tokens-heres-what-that-means/">Google 的 Gemini 1.5 Pro 将拥有 200 万 token。这意味着什么</a>: 不，不是巴士或街机游戏的代币。这里的 token 指的是人工智能系统使用的构建块。
</li>
</ul>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1245352482520174602)** (3 条消息): 

- **查看 AI 生成的思考**：一位用户分享了一个 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/as-a-thought-P2Ss7guHR7COmJo0VEqWQA)。该链接似乎指向一个 AI 生成的思考或搜索查询。
  
- **这有意义吗？**：另一位用户发布了一个 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/Does-this-make-ojRBU_QVTruL5TYG_GDfKQ)。从消息中无法明确搜索的具体内容。
  
- **优缺点讨论**：一位用户贡献了一个[链接](https://www.perplexity.ai/search/Vor-und-Nachteile-jyWAvvwhT1qoWsdFiCP7mQ)，讨论了 "Vor- und Nachteile"，翻译过来就是“优势和劣势”。这表明对特定话题的优缺点进行了详细探讨。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1245143092504563762)** (2 条消息): 

- **尝试新的模型别名**：一位成员建议另一位成员尝试将模型别名从 `sonar-small-online` 切换到 `llama-3-sonar-small-32k-online`。提出这一建议可能是为了测试切换是否能提升性能或解决待处理的问题。
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1245092896743620608)** (951 条消息🔥🔥🔥): 

- **用户寻求各种技术问题的帮助**：一位用户在 Docker 中使用 Chat UI 时遇到问题，在 env.local 文件中收到“意外字符”错误。另一位用户在 PyTorch 2.4 中成功使用 "torch compile" 进行多 GPU 训练，在 A100 配置下实现了更快的训练速度。

- **对 XP 等级和机器人功能的担忧**：几位用户抱怨丢失了 XP 等级以及机器人行为异常。讨论显示，该问题是由于一个影响 levelbot 内存及其与用于存储数据的 Google Sheet 连接的 Bug 引起的。

- **对 AI 训练替代硬件的兴趣**：成员们讨论了各种硬件选项，如 Gaudi2、AMD 和 RTX GPU，以实现更快且更具成本效益的 AI 训练。分享的链接包括从 Supermicro 获取 9 万美元 Gaudi2 机架的详细信息，以及将二手 3090 GPU 作为 LLM 任务的经济型选择。

- **关于微调、内存使用和工具的查询**：针对 TinyLlama 等模型的微调参数出现了疑问，提到了 1e-2 或 1e-3 等学习率。另一位用户询问如何利用 Hugging Face CLI 在误提交后回滚模型版本。

- **AI/ML 学习资源共享与指导**：新手寻求 AI 和 ML 的入门建议，得到的建议包括参加 NLP 课程以及尝试 GPT-2 等推理 API。分享的资源包括用于微调 sentence transformers 的 [Autotrain](https://x.com/abhi1thakur/status/1795823683144962517)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/abhi1thakur/status/1795823683144962517">abhishek (@abhi1thakur) 的推文</a>: 🚨 新任务提醒 🚨 AutoTrain 现在支持 Sentence Transformer 模型的微调 💥 现在，您无需编写任何代码即可改进和定制您的 RAG 或检索模型 🤗 ✅ Su...</li><li><a href="https://huggingface.co/spaces/fishaudio/fish-speech-1/discussions/1">fishaudio/fish-speech-1 · 申请社区资助：个人项目（GPU 和存储）</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/FLOPS">FLOPS - 维基百科</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/local-apps">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/steven-universe-flattered-blush-i-love-you-garnet-gif-22074709">Steven Universe Flattered Blush I Love You Garnet GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/reach-vb?">reach-vb (Vaibhav Srivastav)</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-dont-care-didnt-ask-didnt-ask-i-didnt-ask-gif-25429803">Cat Dont Care Didnt Ask GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/cutycat2000">cutycat2000 (CutyCat2000)</a>: 未找到描述</li><li><a href="https://youtu.be/co3ewqQlX-8?si=IaS1t8b654uND7u2">采访资深 Rust 开发者 | Prime Reacts</a>: 在 Twitch 直播录制，加入我们 https://twitch.tv/ThePrimeagen 原片: https://www.youtube.com/watch?v=TGfQu0bQTKc 作者: https://www.youtube.com/@programme...</li><li><a href="https://youtu.be/tLdRBsuvVKc?feature=shared>)">开发者删除了整个生产数据库，引发混乱</a>: 如果你的任务是删除数据库，请确保你删除了正确的那一个。来源：https://about.gitlab.com/blog/2017/02/10/postmortem-of-database-outage-...</li><li><a href="https://tenor.com/bopcc.gif">Electro Boom GIF - Electro BOOM - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/Beinsezii/ompl/blob/master/screenshot.png">ompl/screenshot.png at master · Beinsezii/ompl</a>: 具有主见的音乐播放器/库。通过在 GitHub 上创建账号来为 Beinsezii/ompl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/diffusers/discussions/7172>)">使用 Flash Attention 配合 SDP Fallback 为 AMD RDNA3/ROCm 带来 30%+ 的加速 · huggingface/diffusers · Discussion #7172</a>: 是的，现在你也可以在 AMD 上使用内存高效的 Attention 了，虽然有一些（很多）限制。diffusers 默认 (SDP)、我的 SubQuad 移植以及所展示的 Flash Attention + SDP fa... 的吞吐量数据...</li><li><a href="https://tenor.com/bDCg9.gif">意外启动 GIF - 意外启动按钮 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.servethehome.com/intel-gaudi-2-complete-servers-from-supermicro-for-90k/">Supermicro 售价 9 万美元的 Intel Gaudi 2 整机服务器</a>: 我们发现了一个 AI 服务器配置的硬性价格，Supermicro 销售的 8 路 Intel Gaudi 2 服务器仅需 9 万美元。</li><li><a href="https://docs.google.com/spreadsheets/d/1C8aLqgCqLYcMiIFf-P_Aosaa03C_WLIB_UyqvjSdWg8/edit#gid=0)">test_merge</a>: Sheet1 discord_user_id, discord_user_name, discord_exp, discord_level, hf_user_name, hub_exp, total_exp, verified_date, likes, models, datasets, spaces, discussions, papers, upvotes L251101219542532097L, osansevier...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/96hRlxSr1u">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://tenor.com/vLnthVIsBpy.gif">Dr Austin GIF - Dr Austin Powers - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://chatgpt-4o.streamlit.app/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1245257771838996480)** (2 条消息): 

- **如何访问频道**：一位用户询问如何访问特定频道。另一位成员回答，指示 *"前往 \<id:customize\> 并选择协作角色"*。
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1245125857220235277)** (14 条消息🔥): 

- **使用 Nowcasting 工具监控通胀趋势**：查看 [Cleveland Fed 的 Inflation Nowcasting 工具](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)，获取 PCE 和 CPI 指数的每日通胀估算。这有助于及时了解月度和年度的通胀变化。
- **用于情感分析的微调 GPT-2 模型在 Hugging Face 上线**：一个使用 GPT-2 训练的新[情感分析模型](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0)已发布，专门针对 Amazon 评论进行了优化。它拥有 96.8% 的准确率，在理解客户反馈方面具有巨大潜力。
- **在 Arxiv 上探索 Mirage 的超优化（Superoptimization）**：论文 [Mirage: Multi-level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751) 介绍了一种使用 $\mu$Graphs 优化 Tensor 程序的新方法，其性能显著优于现有方法。
- **通过 Classical Shadows 实现高效的量子态预测**：论文 [Efficient method for Quantum State Prediction](https://arxiv.org/abs/2002.08953) 概述了一种仅使用极少测量次数即可预测量子态众多属性的方法，展示了极具前景的理论和数值结果。
- **关于使用 GNN 进行状态嵌入（State Embeddings）的讨论**：成员们讨论了在模拟中使用图神经网络 (GNN) 进行状态嵌入的优势，强调了 GNN 如何编码实体之间的复杂关系。这种方法可能会引入一些归纳偏置（inductive bias），使距离信息优先于其他因素。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Langflow/Langflow-Preview">LangFlow 1.0 Preview - Langflow 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0">ashok2216/gpt2-amazon-sentiment-classifier-V1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2002.08953">Predicting Many Properties of a Quantum System from Very Few Measurements</a>：预测复杂、大规模量子系统的属性对于开发量子技术至关重要。我们提出了一种有效的方法来构建近似的经典描述...</li><li><a href="https://arxiv.org/abs/2405.05751">A Multi-Level Superoptimizer for Tensor Programs</a>：我们介绍了 Mirage，这是第一个用于 Tensor 程序的多级超优化器。Mirage 的一个核心思想是 $μ$Graphs，这是一种在 Kernel、线程块和线程层级上对 Tensor 程序的统一表示...</li><li><a href="https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting">Inflation Nowcasting</a>：克利夫兰联邦储备银行提供两个常用价格指数的每日“即时预测 (nowcasts)”，即个人消费支出 (PCE) 价格指数和消费者价格指数 (CPI)...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1245099565695373402)** (8 messages🔥): 

- **认识 HuggingPro：你的 Hugging Face 导航员**：一位成员介绍了 HuggingPro，这是一个旨在帮助用户在 Hugging Face 生态系统中导航的新助手。HuggingPro 提供关于模型、数据集和工具的准确信息，并加入了一点幽默和独家技巧。[HuggingPro](https://hf.co/chat/assistant/66562fe0abb44809b7f77897)。

- **everything-ai v2.0.1 具备更强大的 AI 能力**：更新包括处理音频文件、从文本生成视频、预测蛋白质的 3D 结构、微调模型，以及利用更大的数据库集合进行检索增强生成 (RAG)。该工具可以通过 Docker 设置轻松启动，并且完全是本地运行的。[everything-ai](https://github.com/AstraBert/everything-ai)。

- **解释条件潜扩散模型 (Conditional Latent Diffusion Models)**：一位成员分享了一个 YouTube 视频，涵盖了用于文本到图像生成的 Conditional Latent Diffusion 模型，解释了重要的概念和实现细节。[观看视频](https://youtu.be/w8YQcEd77_o)。

- **Image Generator Pro 发布**：推出了一款用于文本到图像生成、序列图像生成和图像编辑的新工具。该工具可在 Hugging Face Spaces 上使用。[Image Generator Pro](https://huggingface.co/spaces/KingNish/Image-Gen-Pro)。

- **Nvidia 的 Embedding 模型演示**：Nvidia 新的 Embedding 模型演示已开放测试，可与 Microsoft 的 e5-mistral 模型相媲美。欢迎为示例用例和功能做出贡献。[Nvidia Embed V1](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/Image-Gen-Pro">Image Gen Pro - KingNish 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1/">Tonic 的 NV-Embed - Tonic 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://youtu.be/w8YQcEd77_o">15 个必须掌握的概念解释文本到图像生成扩散模型！(+ 如何编码)</a>: 在短短 15 个要点中，我们讨论了关于生成式 AI 扩散模型你需要知道的一切——从基础知识到潜扩散模型 (LDMs) 和文本...</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: 你的全能、AI 驱动且本地运行的聊天机器人助手🤖</a>: 你的全能、AI 驱动且本地运行的聊天机器人助手🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: 你的全能、AI 驱动且本地运行的聊天机器人助手🤖</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897">HuggingPro - HuggingChat</a>: 在 HuggingChat 中使用 HuggingPro 助手</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每一个人。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1245149088761122948)** (9 messages🔥): 

- **读书会已排期**：宣布了新的读书会，鼓励论文作者展示他们的工作。提供了 [活动链接](https://discord.com/events/879548962464493619/1245408612818620426)。

- **对低资源语言机器学习的兴趣**：一位成员建议邀请 C4AI 社区加入读书会，并强调了他们关于“使用 LLM 在低资源语言中揭穿虚假信息”的演讲。他们对与非洲语言相关的话题表现出极大的热情。

- **鼓励演示**：Lunarflu 对来自 C4AI 社区的演示表示感兴趣，特别是如果他们撰写了论文或发布了开源 GitHub 仓库。另一位成员确认他们将进行介绍，并称赞了最近一次演示的质量。
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1245257826897629195)** (18 条消息🔥): 

- **医疗图像分析任务求助**：一位用户在开发涉及未标记 MRI 和 CT 扫描的医疗图像分析自监督学习（self-supervised learning）框架时需要帮助。另一位成员建议使用预训练模型提取特征，然后运行适用于识别出的类别的分割模型。

- **Transformer 图像管理指南**：一位用户询问 YOLO 或 SAM 等 SOTA 目标检测模型如何处理大尺寸图像。另一个讨论围绕微调基于 Transformer 的模型展开，建议使用 [convNext, DINOv2, 或 SigLIP](https://github.com/google-research/tuning_playbook?tab=readme-ov-file#choosing-the-batch-size) 而非 ViT，并建议配合 AdamW 优化器使用余弦学习率调度器（cosine learning rate scheduler）。

- **用于纸张检测的预训练模型**：有人询问是否有用于检测图像中纸张的预训练模型，理由是传统方法缺乏鲁棒性。讨论中未提供有关解决方案或具体模型的进一步细节。

- **图像处理资源与 Notebook**：分享了一些有用的资源和 Notebook 链接，包括如何使用 HuggingFace datasets 处理图像，以及一个包含特定图像处理工作流教程的 [GitHub 仓库](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/v2.3.2/en/image_process">处理图像数据</a>：未找到描述</li><li><a href="https://huggingface.co/do">Do (Tran)</a>：未找到描述</li><li><a href="https://huggingface.co/models?search=dpt%20dino">Models - Hugging Face</a>：未找到描述</li><li><a href="https://x.com/NielsRogge/status/1795106366752723094.">来自 Niels Rogge (@NielsRogge) 的推文</a>：事实证明我的 Idefics2 notebook 同样适用于 PaliGemma 微调 :) 在这里查看：https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma 对于 JSON 用例，一个微型 VLM ...</li><li><a href="https://github.com/google-research/tuning_playbook?tab=readme-ov-file#choosing-the-batch-size">GitHub - google-research/tuning_playbook: 系统性最大化深度学习模型性能的指南。</a>：系统性最大化深度学习模型性能的指南。 - google-research/tuning_playbook
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1245396532136644658)** (1 条消息): 

- **分类建模中的文档长度处理**：一位成员询问了关于文档长度超过 LLM（如 **BERT (512 tokens)** 和 **基于 decoder 的模型 (1024 tokens)**）限制的分类建模问题。他们正在寻找文档切片和更新位置嵌入（positional embeddings）之外的替代方案，以避免昂贵的新预训练方法。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1245262472928170066)** (4 条消息): 

- **用于情感分析的微调模型**：一位成员宣布创建了一个使用 GPT-2 在 Amazon 评论上进行情感分析的微调模型。该模型现已在 Hugging Face 上线，具有显著的指标，如 **0.9680 的准确率和 F1 分数** [在 Hugging Face 上查看](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0)。
  
- **庆祝 Diffusers 生日**：多位成员庆祝了 Hugging Face 的 Diffusers 项目成立两周年。分享了一个 [commit 链接](https://github.com/huggingface/diffusers/commit/0bea026caa182802874f80917dd45afa8db2273) 以纪念这一时刻。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0">ashok2216/gpt2-amazon-sentiment-classifier-V1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/commit/0bea0268caa182802874f8">上传初始结构 · huggingface/diffusers@0bea026</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1245089360504819763)** (656 条消息🔥🔥🔥):

- **Lighting AI Studio 建议引发合作探讨**：成员们推荐了 **Lighting AI Studio**，因其每月提供约 20 小时的免费额度，且 L4 显卡的性能优于 Colab 的 T4。暗示与 Lightning 的合作将对社区大有裨益。
- **Llama3 聊天机器人微调困扰**：讨论了针对论文补全和创建耶稣、唐纳德·特朗普等角色扮演（RP）角色的 **llama3** 模型微调。部分成员在处理大上下文长度和批处理（batch）配置时遇到问题，并发现合成数据集的效果较差。
- **微调社区资源**：分享的有用资源包括关于 SFTTrainer 的 [Hugging Face 文档](https://huggingface.co/docs/trl/en/sft_trainer#training-adapters) 以及各种 LoRA 和超参数指南。成员们讨论了为微调编写详细笔记。
- **Phi3 模型与基准测试引发争议**：对 **Phi3** 模型的评价褒贬不一，一些成员认为 `phi-3-medium` 与 **llama3-8b** 相比表现平平。一位用户报告称，与 Llama3 的表现相比，Phi3 在超过 2048 个 token 的上下文长度后表现不佳。
- **公告与新模型发布**：对 **Codestral 22B** 等新模型的发布感到兴奋，并附带了 HuggingFace 链接和官方公告（[Mistral AI Codestral](https://mistral.ai/news/codestral/)）。还重点讨论了对 **Qwen2** 模型的期待。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">使用 Unsloth 微调 Phi-3</a>：通过 Unsloth 轻松微调 Microsoft 的新模型 Phi 3 medium, small &amp; mini，并获得 6 倍长的上下文长度！</li><li><a href="https://huggingface.co/DDIDU/ETRI_CodeLLaMA_7B_CPP">DDIDU/ETRI_CodeLLaMA_7B_CPP · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/wttw/Llama3-8B-CPP">wttw/Llama3-8B-CPP · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.14734">SimPO: 具有无参考奖励的简单偏好优化</a>：Direct Preference Optimization (DPO) 是一种广泛使用的离线偏好优化算法，它重新参数化了来自人类反馈的强化学习 (RLHF) 中的奖励函数，以增强...</li><li><a href="https://x.com/huybery/status/1795432194460340708">Binyuan Hui (@huybery) 的推文</a>：我查看了 HuggingFace🤗 上 Qwen1.5 系列模型的下载统计数据。Qwen1.5-7B 夺得冠军，CodeQwen1.5-7B 在短短一个多月内下载量就达到了约 26.5 万次。❤️ 感谢大家...</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1/tree/main">mistralai/Codestral-22B-v0.1 at main</a>：未找到描述</li><li><a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>：通过 Mistral AI 赋能开发者并使编程大众化。</li><li><a href="https://arxiv.org/abs/2403.07794">使用顺序指令微调大语言模型</a>：大语言模型 (LLMs) 在单个查询中难以遵循一系列指令，因为它们可能会忽略或误解其中的一部分。这损害了它们在复杂问题中的表现，而这些问题的解决...</li><li><a href="https://x.com/MistralAILabs/status/1795820935540584909">Mistral AI Labs (@MistralAILabs) 的推文</a>：宣布推出 Codestral：我们首个代码模型。- 在新的 Mistral AI Non-Production License 下开放权重 - 通过 La Plateforme 提供新端点：http://codestral.mistral.ai - 立即在 Le Chat 上体验：h...</li><li><a href="https://github.com/the-crypt-keeper/LLooM">GitHub - the-crypt-keeper/LLooM: 辅助创意写作的实验性 LLM 推理 UX</a>：辅助创意写作的实验性 LLM 推理 UX - the-crypt-keeper/LLooM</li><li><a href="https://github.com/unslothai/unsloth/wiki#sav">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/google/jax/discussions/6843">修复在 cuda-11.3 和 jaxlib 0.1.66+cuda111 下出现的 "Couldn't invoke ptxas --version" 错误 · google/jax · Discussion #6843</a>：大家好，只是想分享一下我在最近使用 cuda-11.3 安装 jax 后遇到的 "Couldn't invoke ptxas --version" 错误的解决方案。简而言之，我需要安装 nvidia-cuda-to...</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我让 unsloth 在原生 Windows 上运行了。 · Issue #210 · unslothai/unsloth</a>：我让 unsloth 在原生 Windows（非 WSL）上运行了。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一份完整的安装教程，我很想在这里写下来，但我现在在手机上...</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#training-adapters">有监督微调训练器 (Supervised Fine-tuning Trainer)</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/lora">LoRA</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1245309573561323521)** (3 条消息): 

- **寻求 HTML, CSS, JS 方面的帮助**：一位用户在为他们正在开发的界面寻求 HTML, CSS 和 JS 方面的协助。"这里有人能帮我吗...？" 

- **私信请求**：同一位用户请求如果有人能提供帮助请私信 (DM) 他们。"如果你能帮忙请私信我。"
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1245131167338991700)** (59 条消息🔥🔥): 

- **Colab 断连问题引发不满**：成员们讨论了即使升级到 **Premium** 级别后，Google Colab 仍然持续断连的问题。建议包括切换到 **Kaggle** 和 **Lightning AI** 等提供免费计算时长的替代方案。

- **使用 Unsloth 进行本地推理**：一位用户寻求在本地 **VSCode** 中运行 Unsloth 模型以执行简历要点生成等任务的指导。建议通过编写简单的 Python 脚本来改编 Colab 推理示例，这可能需要进行微调（fine-tuning）。

- **使用 Runpods 和 Docker 部署模型**：用户交流了使用 **Runpods** 配合 Docker 部署模型的想法，甚至在遇到供应商问题时考虑了替代方案。虽然目前没有现成的特定 Dockerfile，但建议在服务器中进行搜索。

- **关于持续预训练（Continued Pretraining）的澄清**：社区澄清了 Unsloth 原生支持**无监督微调**（持续预训练）。[此处](https://github.com/unslothai/unsloth#-finetune-for-free)和[此处](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)提供了用于无监督微调的相关 Colab notebook 和 GitHub 资源。

- **技术错误和 CUDA 版本问题**：用户报告并解决了特定的技术问题，例如为 **PyTorch 2.2** 安装合适版本的 xformers，以及 Unsloth 所需的 CUDA 版本（`11.8`）。这些交流突显了社区内的故障排除环节。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth)">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1245092106171580428)** (74 messages🔥🔥): 

- **LLM 职位与自我推荐频道建议**：一名成员建议在 Discord 社区中创建一个专门用于 LLM 相关就业、职业机会和自我推荐的频道。这将为人们分享职位空缺和个人成就提供空间。

- **关于微调辩论的观点**：成员们讨论了在 Fine-tuning 课程中加入暗示“Fine-tuning 已死”的演讲所引发的争议。共识是，不同的观点对于全面理解非常有价值，类似于过去著名的演讲，如 Joel Grus 的“为什么我不喜欢 Jupyter Notebooks”。

- **Google 提高 Gemini 1.5 价格**：用户强调了对 Google 在 Gemini 1.5 Flash 发布后不久将其输出价格提高 98% 的担忧。这引发了关于成本突然剧烈变化的 API 可靠性的讨论。

- **JSON/Parquet 文件工具**：一位用户询问处理任意 JSON/Parquet 文件的强大工具推荐，寻求比 Jupyter 更易用但具备数据库浏览器功能的替代方案。

- **建立地区性线下聚会 (Meetups)**：成员们表示有兴趣创建地区性聚会频道，从旧金山和纽约市开始，以此作为促进社区成员之间面对面联系的一种方式。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/karp">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen-Audio">Qwen/Qwen-Audio · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/artificialguybr/status/1795851375181508785">来自 𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr) 的推文</a>：Google 在未告知任何人的情况下将 Gemini 1.5 Flash 的输出价格提高了 98%。而这距离该模型发布仅一周。输出价格从 0.53/1M 变为 1.05/1M。我们如何信任一个如此剧烈变动的 API...</li><li><a href="https://x.com/omooretweets/status/1795834644732285402">来自 Olivia Moore (@omooretweets) 的推文</a>：🚨 新的 @a16z 投资主题！是时候让 AI 重塑电话通话了 - 进入对话式语音 Agent 📱 我们很高兴投资的领域 + 市场图谱（来自我和 @illscience）👇</li><li><a href="https://tenor.com/view/rug-pull-gif-21378865">Rug Pull GIF - Rug Pull - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://withaqua.com/">Aqua Voice - 纯语音文档编辑器</a>：Aqua Voice (YC W24) 是一款语音驱动的文本编辑器，让你仅用声音就能创建和编辑文档。</li><li><a href="https://lu.ma/y4xkq595">与三位 AI 投资人共度良宵 · Luma</a>：请加入我们于 5 月 30 日星期四在 Solaris AI 举行的关于投资 AI 初创公司的专题讨论。我们的嘉宾是：- Yoko Li - Josh Buckley - Lenny…</li><li><a href="https://x.com/karpathy/status/1795484547267834137">来自 Andrej Karpathy (@karpathy) 的推文</a>：# 在 llm.c 中用 90 分钟和 20 美元复现 GPT-2 (124M) ✨ GPT-2 (124M) 是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，现在其实非常容易获取，即使对于 G...</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">在 llm.c 中用 90 分钟和 20 美元复现 GPT-2 (124M) · karpathy/llm.c · Discussion #481</a>：让我们在 llm.c（约 4,000 行 C/CUDA 代码）中用 90 分钟和 20 美元复现 GPT-2 (124M)。124M 模型是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，现在其实相当...
</li>
</ul>

</div>

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1245090540115202078)** (4 messages): 

- **通过 LLM Fine-tuning 实现个性化销售邮件**：一位用户描述了为销售邮件生成“个性化开场白”如何能更有效地吸引收件人的注意。他们强调，使用成功邮件开头和收件人资料的数据集进行 Fine-tuning，可以确保生成**对齐且具有吸引力的外联内容**。

- **使用 LLM 进行高效法律文档摘要**：讨论集中在如何使用 LLM 为法律诉讼汇总大量的证据开示（discovery）文档。针对特定法律领域对模型进行 Fine-tuning，可以确保生成**准确且可验证的摘要**，从而简化法律工作流程。

- **多 Agent LLM 协作模型**：用户探讨了多 Agent LLM 设置的概念，其中每个 Agent 专注于特定领域，并在持续循环中运行以解决跨学科问题。建议包括使用 RAG 提供额外上下文，并针对各自领域对每个模型进行 Fine-tuning。

- **使用 LLM 优化 Stable Diffusion 提示词**：讨论了使用 LLM 增强 Stable Diffusion 图像生成提示词的话题。提议通过 Fine-tuning 和结合 RAG 的 Few-shot Learning 来创建**更详细且具有特定风格的提示词**，从而改进从简单描述生成的图像效果。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1245210322348937268)** (3 messages): 

- **Divya 觉得 Workshop 3 信息量极大**：来自新加坡的新成员 Divya 分享了对 Workshop 3 的兴奋之情，称其为“超高带宽（high bandwidth）”，并表示需要时间来消化内容。她仍在努力搭建 Axolotl 环境，并期待共同学习。

- **Sid 建议从 Workshop 录像和作业开始**：另一位成员 Sid 建议从 Workshop 的录像和附带的作业开始。他强调，该 Workshop 更多地像是一个会议，旨在了解构建 LLM 应用的通用实践并开始个人项目。

- **来自浦那的欢迎**：来自印度浦那的 Gurdeep 加入了对话并向小组问好。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1245342054536577144)** (21 messages🔥): 

- **调试用于 Fine-tuning 的 Modal**：一位成员在尝试使用 Modal 运行 Fine-tuning 示例时遇到了“不透明的失败（opaque failure）”，且不确定如何调试该问题。错误似乎与 Secret 配置有关（“即使在重命名我的 Secrets 之后……问题仍然存在”）。

- **Modal Secret 配置帮助**：针对调试问题，另一位用户建议使用命令 `modal secret list` 验证 Secret。然而，原用户确认他们的问题与 Secret 无关。

- **Saharn 寻求合成数据训练方面的帮助**：一位名为 Saharn 的用户在使用 Modal 进行合成数据训练时遇到了错误，并概述了他们的设置。另一位成员建议确保数据集路径正确放置在 `data` 文件夹中，并澄清不需要在配置文件中指定数据集路径。

- **无法下载 Docker 镜像**：针对有关在本地拉取已构建的 Docker 镜像的查询，一位成员确认在 Modal 中无法实现此操作。

**提到的链接**：<a href="https://modal.com/zmackie/apps/ap-on1FEjZETViEB9LRCuGJNI">登录</a>：欢迎回到 Modal！通过在下方选择身份提供商登录您的 Modal 账户。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1245099933187571712)** (6 messages): 

- **O'Reilly 关于构建 LLM 应用的见解**：[O'Reilly Radar](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) 分享了一年来构建 LLM 应用的经验教训，指出虽然准入门槛降低了，但创建有效的 AI 产品仍然具有挑战性。文章强调了对开发基于 LLM 的产品至关重要的知情方法论。

- **整理学习资源的建议**：一位成员提议在 Repository 或网页上整理一份共享资源列表。该想法包括添加点赞/踩的评分机制，以帮助用户确定内容的优先级。

**提到的链接**：<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">我们从一年构建 LLM 应用中学到了什么（第一部分）</a>：未找到描述内容。

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1245236718102384701)** (8 messages🔥): 

- **使用 Python SDK 关闭容器**：一名成员建议使用 Jarvislabs Python SDK 来关闭实例，因为它是基于容器的。他们分享了一个实现此功能的[代码片段](https://jarvislabs.ai/docs/api)。
  
- **关于在 Jarvislabs 中使用 volumes 的查询**：一名成员询问了使用 volume 保存文件并在不同容器间访问的可能性。另一名成员澄清说，他们询问的内容听起来像是跨多个容器的共享存储。
  
- **Axolotl 预处理 GPU 支持问题**：一名成员在运行 Axolotl 预处理时遇到问题，指出 bitsandbytes 库在编译时没有 GPU 支持，导致操作被迫使用 CPU。他们还分享了详细的日志输出，显示由于缺乏 CUDA 支持，系统默认使用 CPU 加速。
  
- **Axolotl 查询的后续跟进**：为了解决预处理问题，一名成员分享了一个[相关的 Discord 讨论链接](https://discord.com/channels/1238365980128706560/1244238835467030610/1244239260714930186)，表明存在类似的未解决问题。该成员后来注意到，尽管最初存在问题，他们的训练脚本最终还是运行并利用了 GPU。

**提到的链接**：<a href="https://jarvislabs.ai/docs/api">JLClient | Jarvislabs </a>：JLClient 是一个用于与 Jarvislabs.ai 交互的 Python API，涵盖了 GPU 实例的完整生命周期。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1245092422925553664)** (15 messages🔥): 

- **在生产环境中部署 HF 模型令人沮丧**：一位用户表达了在使用 Lightning Studios 微调模型并在生产环境中部署时遇到的困难。他们寻求关于将 Pytorch 模型转换为 safetensors 格式的建议。

- **对 Lightning 格式不了解**：一位成员承认对 Lightning 格式不了解，并建议如果已知推理代码，可以创建一个自定义 handler。他们提供了一个[自定义 handler 指南链接](https://huggingface.co/docs/inference-endpoints/en/guides/custom_handler)。

- **分享转换教程**：另一位用户提到他们参考 [GitHub 上的 LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 教程，在 GPT2-medium 基础模型上微调了模型。他们提到按照该教程在不同领域执行二分类任务。

- **PTH 到 Safetensors 的转换**：一位成员建议 `.pth` 文件等同于 `torch.bin` 文件，应该可以转换为 safetensors 格式。他们分享了 [Hugging Face 关于将权重转换为 safetensors 的指南链接](https://huggingface.co/docs/safetensors/en/convert-weights)。

- **课程学分表格的电子邮件地址说明**：一位用户询问接收课程学分的电子邮件是否可以与注册电子邮件不同。得到的回答是肯定的，建议直接填写表格即可，无需担心使用的电子邮件。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/safetensors/en/convert-weights">Convert weights to safetensors</a>：未找到描述</li><li><a href="https://github.com/rasbt/LLMs-from-scratch">GitHub - rasbt/LLMs-from-scratch: Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step</a>：在 PyTorch 中从零开始逐步实现类似 ChatGPT 的 LLM - rasbt/LLMs-from-scratch
</li>
</ul>

</div>

### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1245404940822773921)** (53 messages🔥): 

- **研讨会亮点：Braintrust**：Ankur 展示了 [Braintrust](https://www.braintrustdata.com)，讨论了它在使用 Autoevals 等工具针对 Text2SQL 评估非确定性 AI 系统方面的效用。与会者赞赏该环节对迭代工作流和简单工具的关注，多位成员表示很期待尝试。
  
- **共享资源与链接**：分享了几个关键链接，包括 [Braintrust cookbook](https://www.braintrustdata.com/docs/cookbook)、[演示中使用的 notebook](https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/Text2SQL/Text2SQL.ipynb) 以及来自 [Hugging Face](https://huggingface.co/datasets/suzyanil/nba-data) 的支持数据集。爱好者们发现这些资源对跟随学习和实施 Braintrust 很有帮助。

- **自托管建议**：Ankur 建议在处理包含敏感信息的私有数据库时自托管 Braintrust。他引用了 [自托管指南](https://www.braintrust.dev/docs/guides/self-hosting) 以协助用户在自己的环境中高效设置 Braintrust。

- **使用 Autoevals 进行 SQL 评估**：为了检查 SQL 查询的语义等价性，Ankur 分享了 Autoevals 使用的一种简单方法，并为有兴趣调整评估 Prompt 的用户提供了 [模板](https://github.com/braintrustdata/autoevals/blob/main/templates/sql.yaml) 和 [自定义文档](https://www.braintrust.dev/docs/reference/autoevals/overview#custom-evaluation-prompts)。

- **Autoevals 与 Langsmith 的对比**：用户将 Braintrust 的评估能力与 Langsmith 进行了比较，指出 Braintrust 感觉更简洁、更易于导航。这引发了关于可能使用 Langsmith 进行 Logging 和 Tracing，而 Braintrust 因其用户友好的界面和视觉元素而成为评估理想选择的讨论。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/Text2SQL/Text2SQL.ipynb">braintrust-cookbook/examples/Text2SQL/Text2SQL.ipynb at main · braintrustdata/braintrust-cookbook</a>：通过在 GitHub 上创建账户来为 braintrustdata/braintrust-cookbook 的开发做出贡献。</li><li><a href="https://www.braintrustdata.com/">Braintrust</a>：无需猜测，快速交付 AI</li><li><a href="https://www.braintrust.dev/docs/guides/self-hosting">Braintrust</a>：Braintrust 是用于构建 AI 产品的企业级技术栈。</li><li><a href="https://github.com/braintrustdata/braintrust-cookbook/blob/main/examples/Text2SQL-Data/Text2SQL-Data.ipynb">braintrust-cookbook/examples/Text2SQL-Data/Text2SQL-Data.ipynb at main · braintrustdata/braintrust-cookbook</a>：通过在 GitHub 上创建账户来为 braintrustdata/braintrust-cookbook 的开发做出贡献。</li><li><a href="https://docs.google.com/presentation/d/1k7H9m3SJQ5KAiNBQ2sILVWfNV15j6kY5g8kl-mNDXmc/edit#slide=id.p">LLM Eval For Text2SQL</a>：用于 Text2SQL 的 LLM 评估</li><li><a href="https://github.com/braintrustdata/autoevals/blob/main/templates/sql.yaml">autoevals/templates/sql.yaml at main · braintrustdata/autoevals</a>：AutoEvals 是一个使用最佳实践快速轻松评估 AI 模型输出的工具。- braintrustdata/autoevals</li><li><a href="https://www.braintrust.dev/docs/reference/autoevals/overview#custom-evaluation-prompts">Braintrust</a>：Braintrust 是用于构建 AI 产品的企业级技术栈。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1245421491839959050)** (141 messages🔥🔥):

```html
- **强烈推荐 John Berryman 的书**：John Berryman 在 O'Reilly 上的 Prompt Engineering 书籍有望成为开发者的全面指南，巩固了对实际应用非常有用的 LLM 原理和 Prompt Engineering 技术。点击[这里](https://learning.oreilly.com/library/view/prompt-engineering-for/9781098156145/)查看。
- **探索 Prompt Engineering 工具和框架**：成员们分享了大量资源，包括 [Hamel 的笔记](https://hamel.dev/notes/llm/openai/func_template.html)链接、GoEx 和通过 [Langchain 博客](https://blog.langchain.dev/reflection-agents/)介绍的 reflection agent 技术，以及 [Notion](https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f?pvs=4) 上的 JSON Schema 详情。
- **关于 LLM 行为和调优的有趣见解**：成员们讨论了计算的底层原理如何产生 LLM 的能力，包括通过 ReAct 等框架链接推理和行动的参考资料。查看论文 [ReAct: Synergizing Reasoning and Acting in Language Models](https://www.promptingguide.ai/techniques/react)。
- **Copilot 聊天机器人技巧**：几位成员分享了使用 GitHub Copilot 和 Cursor 等 AI 辅助编程工具的经验，建议检查 workspace context 和内联聊天工具。参见 [Copilot workspace context](https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace) 以优化基于工作区的查询。
- **Function calling 和评估技术**：讨论引发了关于利用 [Anthropic 的 XML tags](https://docs.anthropic.com/en/docs/use-xml-tags) 等框架/工具，以及如何通过计算 Levenshtein distances 或 embeddings 的库动态选择 few-shot 示例的讨论。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.langchain.dev/reflection-agents/">Reflection Agents</a>: Reflection 是一种 prompting 策略，用于提高 Agent 和类似 AI 系统的质量和成功率。这篇文章概述了如何使用 LangGraph 构建 3 种 reflection 技术，包括 imp...</li><li><a href="https://www.promptingguide.ai/techniques/react">Prompt Engineering Guide</a>: Prompt Engineering 的全面概述</li><li><a href="https://code.visualstudio.com/docs/copilot/workspace-context#_tips-for-using-workspace">Chat using @workspace Context References</a>: 如何使用 Copilot 的 @workspace 聊天功能针对整个代码库提问。</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最好的 AI 聊天模型对所有人可用。</li><li><a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>: 虽然 LLM 在语言理解和交互式决策任务中展示了令人印象深刻的能力，但它们的推理能力（例如 chain-of-thought...）</li><li><a href="https://tenor.com/view/evil-laugh-the-matrix-agent-smith-gif-4145137">Agent Smith - Evil Laugh GIF - Evil Laugh The Matrix Agent Smith - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://blog.jnbrymn.com/2024/01/30/the-marvel-of-GPT-generality.html">
    
      Tool Invocation – 展示 GPT 灵活性之奇迹 &middot; Thought Box
    
</a></li></ul></div>

  </a>: 未找到描述</li><li><a href="https://www.notion.so/matijagrcic/JSON-Schema-78055af9ce1242e8b9be27918056be2f?pvs=4,">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合为一的新工具。它是为您和您的团队提供的一体化工作空间。</li><li><a href="https://docs.anthropic.com/en/docs/use-xml-tags">使用 XML 标签 - Anthropic</a>: 未找到描述</li><li><a href="https://hamel.dev/notes/llm/openai/func_template.html">Hamel 的博客 - 函数提示词</a>: OpenAI 是如何为其函数调用格式化提示词的？</li><li><a href="https://www.manning.com/books/relevant-search">相关性搜索 (Relevant Search)</a>: 《相关性搜索》揭开了相关性工作的神秘面纱。通过使用 Elasticsearch，它教你如何向用户返回具有吸引力的搜索结果，帮助你理解并利用 Lucene 的内部机制...</li><li><a href="https://gorilla.cs.berkeley.edu">Gorilla</a>: 未找到描述</li><li><a href="https://docs.google.com/presentation/d/1PXzENGNN5NFbEDJ59wbSp8fro6dPt4xHGNN6X0KU82A/">提示工程 v2 (压缩版)</a>: John Berryman 的提示工程 (Prompt Engineering)</li><li><a href="https://learning.oreilly.com/library/view/prompt-engineering-for/9781098156145/">面向 LLM 的提示工程</a>: 大语言模型 (LLMs) 承诺了前所未有的收益。LLM 精通人类话语的常见话题，可以对各种任务做出有用的贡献，尤其是现在...</li><li><a href="https://x.com/jnbrymn?lang=en">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://www.oreilly.com/library/view/prompt-engineering-for/9781098156145/">面向 LLM 的提示工程</a>: 大语言模型 (LLMs) 承诺了前所未有的收益。LLM 精通人类话语的常见话题，可以对各种任务做出有用的贡献，尤其是现在...</li><li><a href="https://x.com/jnbrymn">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2005.14165">语言模型是少样本学习者 (Language Models are Few-Shot Learners)</a>: 最近的工作表明，通过在大规模文本语料库上进行预训练，然后在特定任务上进行微调，在许多 NLP 任务和基准测试中取得了实质性的进展。虽然通常在任务上是不可知的...</li><li><a href="https://arxiv.org/abs/2201.11903">思维链提示在大型语言模型中激发推理 (Chain-of-Thought Prompting Elicits Reasoning in Large Language Models)</a>: 我们探索了生成思维链（一系列中间推理步骤）如何显著提高大语言模型执行复杂推理的能力。特别是，我们...</li><li><a href="https://arxiv.org/abs/2205.11916">大语言模型是零样本推理者 (Large Language Models are Zero-Shot Reasoners)</a>: 预训练的大语言模型 (LLMs) 广泛应用于自然语言处理 (NLP) 的许多子领域，通常被认为是具有特定任务示例的优秀少样本学习者。值得注意的是...</li><li><a href="https://gorilla.cs.berkeley.edu/">Gorilla</a>: 未找到描述</li><li><a href="https://x.com/erhartford/status/1795662699700851010">来自 Eric Hartford (@erhartford) 的推文</a>: Cognitive Computations 发布了 Dolphin-2.9.2-Mixtral-8x22b，使用新数据集 SystemChat 2.0 训练，旨在教导 Dolphin 遵守 System Prompt，即使是在长对话中。此次发布...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/)** (1 条消息): 

computer_internet_man: 🧠🍿
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1245120218553127046)** (5 条消息): 

- **浮点数很奇怪，句号**：一位成员强调了浮点数的怪癖，断言它们的加法不符合结合律。他们解释说，由于数值可能存在低估或溢出，AI 模型中常用的梯度需要更高的精度。
  
- **精度对梯度估计至关重要**：在讨论精度差异时，他们对比了 8bit 和 16bit 浮点数的累加，16bit 提供了更准确的梯度估计，当转换为 8bit 时近似于 N*eps。

- **sharegpt 格式的 HF dataset**：另一位贡献者提到 [HF dataset](https://www.huggingface.co) 使用 sharegpt 格式，其中包括 "from" 和 "value" 键。

- **合成数据微调难题**：一位用户讨论了为类似于 honeycomb 示例的模型生成合成数据进行微调的困难，指出他们目前的应用程序表现约为 66% 的准确率。他们思考是否应该生成更多数据并进行筛选，以找到高质量的示例进行微调。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1245089223305068656)** (199 条消息🔥🔥):

- **ChainForge 开拓了 Prompt 评估的新领域**：向成员们介绍了 **ChainForge**，这是一个用于 Prompt Engineering 的开源可视化编程环境。它强调简单、愉悦的 Prompt 评估，并提供对 LLM 的强大测试能力 ([ChainForge](https://chainforge.ai/play/))。
  
- **深入探讨 EvalGen 和 SPADE 的评估方法**：讨论重点介绍了 EvalGen 在使 LLM 生成的评估标准与人类需求保持一致方面的能力，以及 SPADE 通过合成数据质量断言来处理 LLM 输出错误的方法 ([EvalGen](https://arxiv.org/abs/2404.12272), [SPADE](https://arxiv.org/abs/2401.03038))。

- **Eugene Yan 的 Fine-tuning 见解**：Eugene Yan 的分享因其详尽的实践方法而受到赞赏，尽管有些人觉得节奏较快。反馈建议扩大图表尺寸，并花更多时间解释概念 ([Slides](https://docs.google.com/presentation/d/1GC868XXjhxOpQEt1jUM79aW0RHjzxPp0XhpFHnYH760/edit#slide=id.p))。

- **收集并分享的链接与资源**：一位成员整理了会议期间分享的大量链接，包括与 LLM 开发和评估相关的文章、论文和工具 ([What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/))。

- **人工复核与标注工具**：讨论还包括了对以高性价比建立人工复核循环（Human Review Loops）的工具和供应商的建议，提到了 Argilla、pigeonXT 和 cluestar 以协助标注任务 ([pigeonXT](https://github.com/dennisbakhuis/pigeonXT), [cluestar](https://github.com/koaning/cluestar))。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://discord.gg/yX2TdaFt8t">加入 llm-fine-tuning Discord 服务器！</a>：在 Discord 上查看 llm-fine-tuning 社区——与 1615 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://x.com/HamelHusain/status/1795526367637049629">Hamel Husain (@HamelHusain) 的推文</a>：我和我的同事们将关于 LLM 的实用建议浓缩到了这个由三部分组成的系列中。干货满满。截图中的这一部分是我最喜欢的摘录之一，建议来自：@eugeneyan, @BEBi...</li><li><a href="https://arxiv.org/abs/2211.08412">通过新闻摘要评估大语言模型的事实一致性</a>：虽然大语言模型 (LLM) 已被证明在多种任务中非常有效，但它们也以产生信息幻觉而闻名。为了衡量 LLM 是否更倾向于事实一致的内容...</li><li><a href="https://x.com/eugeneyan">来自未定义的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2401.03038">SPADE：为大语言模型流水线合成数据质量断言</a>：大语言模型 (LLM) 越来越多地作为流水线的一部分被部署，用于重复处理或生成某种数据。然而，部署的一个常见障碍是频繁且通常...</li><li><a href="https://arxiv.org/abs/2404.12272">谁来验证验证者？使 LLM 辅助的 LLM 输出评估与人类偏好保持一致</a>：由于人工评估的繁琐性质和基于代码评估的局限性，大语言模型 (LLM) 越来越多地被用于辅助人类评估 LLM 的输出。然而 LLM-...</li><li><a href="https://docs.google.com/presentation/d/1GC868XXjhxOpQEt1jUM79aW0RHjzxPp0XhpFHnYH760/edit#slide=id.p">Spellgrounds for Prodigious Prestidigitation</a>：Spellgrounds for Prodigious Prestidigitation，Bryan Bischof 博士，Hex AI 负责人</li><li><a href="https://github.com/koaning/cluestar">GitHub - koaning/cluestar: 从聚类中获取线索！</a>：从聚类中获取线索！通过在 GitHub 上创建账号来为 koaning/cluestar 的开发做出贡献。</li><li><a href="https://github.com/eugeneyan/visualizing-finetunes">GitHub - eugeneyan/visualizing-finetunes</a>：通过在 GitHub 上创建账号来为 eugeneyan/visualizing-finetunes 的开发做出贡献。</li><li><a href="https://hamel.dev/blog/posts/prompt/">- 别废话，给我看 Prompt。</a>：通过拦截 API 调用，快速理解难以捉摸的 LLM 框架。</li><li><a href="https://x.com/hamelhusain/status/1793999634995847262?s=46&t=aOEVGBVv9ICQLUYL4fQHlQ">Hamel Husain (@HamelHusain) 的推文</a>：他的演讲摘要 🔥  &gt; 本次演讲将涵盖 Inspect 的使用和扩展，这是一个用于 LLM 评估的新型开源 Python 框架。Inspect 的开发者 (J.J. Allaire) 将讲解核心概念并演示...</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/.">Inspect</a>：用于大语言模型评估的开源框架</li><li><a href="https://chainforge.ai/">ChainForge：用于 Prompt Engineering 的可视化编程环境</a>：未找到描述</li><li><a href="https://github.com/dennisbakhuis/pigeonXT?tab=readme-ov-file">GitHub - dennisbakhuis/pigeonXT: 🐦 在 Jupyter notebook 中轻松快速地标注数据</a>：🐦 在 Jupyter notebook 中轻松快速地标注数据 - dennisbakhuis/pigeonXT</li><li><a href="https://docs.google.com/presentation/d/1W6A2I4-IEyFhRJ1h6n7wSm_c-GLLS3xNo9SHgBwA2WM/">微调研讨会 3 幻灯片</a>：精通 LLM：面向开发者和数据科学家的会议</li><li><a href="https://github.com/shreyashankar">shreyashankar - 概览</a>：加州大学伯克利分校计算机科学博士生。shreyashankar 拥有 63 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://forums.fast.ai/">fast.ai 课程论坛</a>：fast.ai 课程、软件和研究的论坛</li><li><a href="https://tenor.com/view/waiting-still-gif-20331665">Waiting Still GIF - Waiting Still - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://langfuse.com/">Langfuse</a>：开源 LLM 工程平台——LLM 可观测性、指标、评估、Prompt 管理。</li><li><a href="https://hex.tech/">用数据将大家聚集在一起 | Hex </a>：从快速查询到深度分析，再到精美的交互式数据应用——全部集成在一个协作式的 AI 驱动工作区中。</li><li><a href="https://hex.tech/product/magic-ai/">Hex Magic | 借助 Magic 实现更智能、更快速的分析 | Hex </a>：通过使用 Magic AI 编写查询、构建图表和修复 Bug，每周节省数小时。</li><li><a href="https://eugeneyan.com/writing/evals/">有效与无效的特定任务 LLM 评估</a>：针对分类、摘要、翻译、版权重复和毒性的评估。</li><li><a href="https://x.com/BEBischof">来自未定义的推文</a>：未找到描述</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of">我们从这一年的...中学到了什么</a>

-building-with-llms-part-i/">我们在一年 LLM 构建中学到的经验（第一部分）</a>：未找到描述</li><li><a href="https://x.com/tomaarsen/status/1795425797408235708">tomaarsen (@tomaarsen) 的推文</a>：‼️Sentence Transformers v3.0 发布了！现在你可以使用多 GPU 训练、bf16 支持、损失日志记录、回调函数等功能来训练嵌入模型。我还发布了 50 多个训练数据集及更多内容……</li><li><a href="https://arxiv.org/abs/2305.14296">USB：跨任务和领域的统一摘要基准</a>：虽然 NLP 社区已经产生了许多摘要基准，但没有一个能提供同时解决许多与控制和可靠性相关的重要问题所需的丰富标注……</li><li><a href="https://eugeneyan.com/writing/prompting/">Prompting 基础知识及如何有效应用</a>：结构化输入/输出、预填充（prefilling）、n-shot prompting、思维链（chain-of-thought）、减少幻觉等。</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM&t=139s">为 LLM 扩展“氛围检查（Vibe Checks）” - Shreya Shankar | Stanford MLSys #97</a>：Stanford MLSys 研讨会系列第 97 集！为 LLM 扩展“氛围检查”。演讲者：Shreya Shankar。简介：Shreya Shankar 是计算机科学博士生……</li><li><a href="https://www.usebraintrust.com/">Braintrust | 第一个用户拥有的技术人才网络</a>：Braintrust 将组织与顶尖技术人才联系起来，以完成战略项目并推动创新。</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/workflow.html">Inspect</a>：用于大语言模型评估的开源框架</li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry：基于 OpenTelemetry 的 LLM 应用开源可观测性工具</a>：基于 OpenTelemetry 的 LLM 应用开源可观测性工具 - traceloop/openllmetry</li><li><a href="https://news.ycombinator.com/item?id=37843907">未找到标题</a>：未找到描述</li><li><a href="https://arize.com/blog/breaking-down-evalgen-who-validates-the-validators/">解析 EvalGen：谁来验证验证者？</a>：关于 EvalGen（一种 LLM 辅助评估方法）你需要了解的一切。还包括一些给 LLM 应用构建者的启示。</li><li><a href="https://johnowhitaker.dev/dsc/2024-01-23-tips.html">johnowhitaker.dev – 处理高表面积（high-surface-area）问题的一些建议</a>：未找到描述</li><li><a href="https://sqlmodel.tiangolo.com/">SQLModel</a>：SQLModel，Python 中的 SQL 数据库工具，旨在实现简单性、兼容性和健壮性。</li><li><a href="https://www.traceloop.com/docs/openllmetry/introduction">什么是 OpenLLMetry？- traceloop</a>：未找到描述</li><li><a href="https://pytest-vcr.readthedocs.io/en/latest/#quick-start">欢迎使用 pytest-vcr - pytest-vcr</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.12288">逆转诅咒：在“A 是 B”上训练的 LLM 无法学会“B 是 A”</a>：我们揭示了自回归大语言模型（LLM）中一个令人惊讶的泛化失败。如果模型在“A 是 B”形式的句子上训练，它不会自动泛化……</li><li><a href="https://arxiv.org/abs/2404.13076">LLM 评估者会识别并偏好自己生成的內容</a>：使用大语言模型（LLM）进行自我评估不仅在基准测试中被证明有价值，在奖励建模、宪法 AI（constitutional AI）和自我细化等方法中也同样有效。但新的偏见也随之引入……</li><li><a href="https://hamel.dev/blog/posts/evals/#automated-evaluation-w-llms">- 你的 AI 产品需要评估（Evals）</a>：如何构建特定领域的 LLM 评估系统。</li><li><a href="https://www.amazon.co.uk/Noise-Daniel-Kahneman/dp/0008308993">未找到标题</a>：未找到描述</li><li><a href="https://www.langchain.com/langsmith">LangSmith</a>：让你的 LLM 应用从原型走向生产。</li><li><a href="https://pydantic.dev/logfire">Pydantic Logfire | 简约的可观测性</a>：Logfire 是一种新型的可观测性平台，其构建理念与 Pydantic 相同——即最强大的工具也可以易于使用。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[yang_mistral_finetuning](https://discord.com/channels/1238365980128706560/1242224842053521459/)** (1 messages): 

init27_sanyam: 我们有更多东西要问了 😄 
https://mistral.ai/news/codestral/
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1245358431007936604)** (1 messages): 

- **在 Gradio 中苦于 CSS 自定义**：一位成员询问有关自定义 Gradio 界面 CSS 的文档。他们尝试更改 **gradio-container** 和 **gr-button-primary** 的背景，但只有容器的背景颜色成功应用，按钮的背景颜色没有生效。

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1245097642682482768)** (17 messages🔥): 

- **关闭 sample packing 会影响性能**：一位成员强调，关闭 sample packing 总是会产生巨大差异，并建议将 pad to sequence length 设置为 false。另一位成员澄清道，“当序列较短时，sample packing 的优势非常明显”，这促使成员们在微调时需要考虑序列长度。

- **训练后调试输出不一致**：一位用户注意到使用 TinyLlama 和 alpaca_2k_test 数据集时模型输出存在差异，并分享了他们的 config 以进行故障排除。另一位建议确保按照 Axolotl 的要求进行正确的 prompting，包括包含适当的模板（“Below is an instruction that describes a task...”）以获得预期结果。

- **使用自定义指标和多个数据集**：讨论了使用自定义指标和合并多个数据集的可行性，一些建议指出，虽然 Transformers 支持评估数据集，但不确定 Axolotl 是否直接支持多个训练数据集。

- **排除 padding 错误**：一位用户在训练期间遇到了 padding 错误，追溯到 tokenization 过程中输入格式不当。在不包含 'input_ids' 的特征编码过程中发现了错误。

- **请求微调流程架构图**：一位成员请求提供微调流程的高级架构图，详细说明不同命令如何与数据和配置交互，以便更好地进行调试。讨论强调了需要视觉辅助工具来理解 Axolotl 中的数据流和处理阶段。

**提及的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L31-L97),">axolotl/src/axolotl/prompters.py at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>：尽管提出关于 axolotl 的问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1245097270744055910)** (25 messages🔥): 

- **Beneyal 就模型问题寻求建议**：一位用户就模型问题向另一位成员寻求反馈，并提供了之前详细描述的链接。

- **Ankur 遇到 Axolotl 的依赖问题**：Ankur 报告了在使用 `torch=2.1.1` 和 `python=3.10.12` 通过 Axolotl 进行微调时遇到的依赖问题，并寻求正确安装步骤的帮助。另一位用户建议创建一个独立的虚拟环境来解决这些问题。

- **Tddammo 提供详细的 quantization 见解**：Tddammo 解释了 quantization（量化）概念，并引用了诸如 [LLM.int8()](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) 和 [Hugging Face bitsandbytes integration](https://huggingface.co/blog/hf-bitsandbytes-integration) 等文章，阐明了不同设置如何影响模型和梯度计算。

- **Iggyal 对 dataset_prepared_path 设置感到困惑**：Iggyal 错误地认为将 `dataset_prepared_path` 留空会默认使用 `last_run_prepared`，从而导致训练错误。Caseus_ 建议显式地将 `dataset_prepared_path` 设置为 `last_run_prepared`。

- **Venetis 寻求对 axolotl config 设置的确认**：Venetis 请求对其理解的 axolotl 配置设置进行检查，这些设置涉及模型权重、激活和梯度精度，包括 bf16、f16 和 tf32 等混合精度设置。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/hf-bitsandbytes-integration">A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using transformers, accelerate and bitsandbytes</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管提出关于 axolotl 的问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() and Emergent Features &mdash; Tim Dettmers</a>：当我参加 NAACL 时，我想做一个小测试。我为我的 LLM.int8() 论文准备了两个推介。一个推介是关于我如何使用先进的量化方法来实现 Transformer 性能无损...
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1245168084457885906)** (8 messages🔥): 

- **Chrome 揭示隐藏的视频播放器菜单项**：一位用户发现嵌入式视频播放器根据所使用的浏览器显示不同的菜单项，并在从 Firefox 切换到 Chrome 后发现了“3 点”菜单。
- **尽管有优化，重型任务仍将保留**：目前团队和 pyodide 团队正在持续优化重型任务。然而，大家公认这些任务在某种程度上总是会有较高的资源需求。
- **正在开发使用自定义中间件的早期 PoC**：某个功能的早期概念验证 (PoC) 运行良好，并计划进一步开发。某些实现可能需要自定义中间件，但这仍需验证。
- **针对 OAuth 登录问题的澄清**：GitHub 上已提交一个 issue 以澄清 Gradio 中 OAuth 登录的限制，并附带了 [issue 链接](https://github.com/gradio-app/gradio/issues/8405)。该问题的回答现已在 issue 中提供。
- **Gradio 因其比 Streamlit 更直观而受到赞赏**：在一次对比讨论中，一位成员分享说 Gradio 感觉比 Streamlit **直观得多**，这影响了他们在开发 demo 时的选择。

**提及的链接**：<a href="https://github.com/gradio-app/gradio/issues/8405">Limit oauth logins · Issue #8405 · gradio-app/gradio</a>：我已搜索是否存在类似问题。在 Discord 上收到了关于使用 HF 登录的问题。在此发布以增加可见性：你能限制允许的登录列表（用户名...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1245462893164494878)** (24 messages🔥): 

- **英国政府的 Inspect 框架受到好评**：一位成员对 [英国政府 Inspect 框架网站](https://ukgovernmentbeis.github.io/inspect_ai/) 中 `quarto` 的使用表示欣赏，称赞其结构和组合性。另一位成员表达了对 Quarto 的热情，强调了它在项目中的实用性。

- **RStudio 怀旧与未来展望**：成员们回忆了早期使用 RStudio 进行数据科学的经历及其对职业生涯的影响。一位成员暗示请关注 Python 在类似领域潜在的新进展。

- **希望获得用户对模型评价（Critique）的反馈**：一位成员建议在 Inspect 框架中添加一个复选框，允许用户确认或拒绝评估者模型的评价，这可能会增加用户交互和评估准确性。

- **安全基础的社区方案（Recipes）**：Inspect 提供的组合性和扩展性被视为社区创建方案（Recipes）的机会，特别是在安全方面。这可以让用户更容易地覆盖基本的安全要点。

- **关于 Inspect 功能的疑问**：成员们就 Inspect 的功能进行了提问和讨论，例如函数是需要由用户编写还是作为默认包含，以及在运行评估时如何获取特定窗口。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/frustrated-waaaaaaaa-wwe-angry-mad-gif-13112986">Frustrated Waaaaaaaa GIF - Frustrated Waaaaaaaa WWE - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/lets-go-lets-go-marvel-let%27s-go-thor-let%27s-go-lets-go-thor-gif-6938549561677021369">Lets Go Lets Go Marvel GIF - Lets go Lets go marvel Let&#039;s go thor - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://ukgovernmentbeis.github.io/inspect_ai/">Inspect</a>：用于大语言模型评估的开源框架</li><li><a href="https://tenor.com/view/yes-gif-22712908">Yes GIF - Yes - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1245098837954138292)** (46 messages🔥): 

- **Predibase 限制注册，建议寻求支持**：成员们反映由于仅限工作邮箱注册，在 Predibase 注册时遇到困难。建议联系支持部门解决此问题。
- **仅最后一次表单提交有效**：Dan 确认只有最后一次提交的额度表单会被考虑，解决了关于多次提交的顾虑。
- **Fireworks.ai 已加入额度赞助商**：Fireworks.ai 正在提供额度，成员们询问是否需要单独的表单，或者已包含在现有表单中。表单用词已更新，以明确 "Account ID" 与 "user-id" 的区别。
- **额度表单的确认问题**：许多成员对验证其额度提交以及确保信息正确保存表示担忧。Dan 确认了该问题，并保证尽管表单进行了编辑，数据并未丢失。
- **截止日期澄清及完整列表**：Dan 澄清表单提交截止日期为 5 月 30 日，新报名截止日期为 5 月 29 日。在提供的 [课程链接](https://maven.com/parlance-labs/fine-tuning) 中可以找到大部分需要设置的账户完整列表。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning">Mastering LLMs: A Conference For Developers &amp; Data Scientists by Dan Becker and Hamel Husain on Maven</a>：关于 LLM 一切内容的在线会议。</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSc7U01uRlMd2jeeeLZtaePTul-xBZXBwRx3x8qD2iIpuqE_mg/viewform">Hugging Face 额度申请</a>：在我们为您申请 🤗 HF 额度以使用 https://huggingface.co 的付费服务之前，我们需要完成几件简单的事情！如有任何疑问，请通过 website@huggingface.co 联系我们。...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfoCoXNhUjka09mu8rmgB1YM9s3529-F2oJdP5HkHT1SGfV2Q/viewform">Modal 黑客松额度</a>：要领取您的 Modal 额度，请先在 https://modal.com/ 注册账户。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些示例...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform">Fireworks 额度 - Mastering LLMs : A Conference For Developers &amp; Data Scientists</a>：请填写下方表单以获取 $250 Fireworks 额度！加入我们的 Discord 以获取疑问解答/帮助或更多额度 ;) https://discord.gg/fireworks
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[eugeneyan_evaluator_model](https://discord.com/channels/1238365980128706560/1245116205334007878/1245129093192613950)** (3 messages): 

```html
- **Discussion Hub Redirect**: Members identified a primary channel for questions on finetuning, suggesting that most queries might be happening in [this channel](https://discord.com/channels/1238365980128706560/1245100755787186298).
- **Training Summarization Evaluator Models**: One member shared their appreciation for a recent talk on improving summarization models by first training on a larger set (USB) before fine-tuning on a smaller, targeted dataset (FIB). The takeaway is that this method significantly boosts the evaluator model's performance on the specific dataset they care about, highlighting how "training on an additional dataset followed by the dataset we care about drastically improves performance."
```

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1245126372137893898)** (9 messages🔥): 

- **Fireworks 额度管理负责人已明确**：一名成员将负责管理 Fireworks AI 额度。
- **Fireworks 额度申请表已发布**：提供了一个 [Google 表单](https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform) 链接，供用户领取 250 美元的 Fireworks 额度。说明包括创建 Fireworks 账户并提交 **Account ID**。
- **社区对 Fireworks 额度团队表示感谢**：多名成员对处理 Fireworks 额度的团队表示赞赏，并表达了兴奋之情。
- **关于表单术语的反馈**：指出由于在 Fireworks AI 中使用了 “user-id” 而非 **“Account ID”** 可能导致的错误。随后表单已针对此问题进行了修改。
- **Fireworks 的独特产品受到好评**：一位成员指出，Fireworks 是他们发现的唯一一家提供**具有视觉能力的开源模型**的供应商。

**提及的链接**：<a href="https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform">Fireworks Credits - Mastering LLMs : A Conference For Developers &amp; Data Scientists</a>：请填写下方表单以获取 250 美元 Fireworks 额度！如有疑问/寻求帮助或获取更多额度，请加入我们的 Discord ;) https://discord.gg/fireworks

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1245408812211638382)** (3 messages): 

```html
- **Greetings flood the channel**: Members exchanged greetings with each other. *"Hello all 👋,"* one member said, receiving a wave of *"👋🏽" and "hi"* in response.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1245425209222234204)** (7 messages): 

- **圣迭戈 vs. 旧金山之争**：成员们讨论了 **San Diego** 与 **San Francisco** 在当地景点方面的优劣。一位成员提到了旧金山标志性的金门大桥，而另一位则支持圣迭戈的微型酿酒厂、动物园和海滩。
- **旧金山 Voice+AI 见面会**：宣布即将在旧金山举行 **Voice+AI 见面会**，定于周四晚上。活动包括小组讨论、演示和披萨，并为参会者提供了 [注册链接](https://lu.ma/y4xkq595)。

**提及的链接**：<a href="https://lu.ma/y4xkq595">An evening with three AI investors · Luma</a>：请在 5 月 30 日星期四加入我们在 Solaris AI 举行的关于投资 AI 初创公司的小组讨论。我们的嘉宾包括：- Yoko Li - Josh Buckley - Lenny…

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1245415577317806162)** (14 messages🔥): 

- **NYC 见面会引起关注**：在纽约市举办见面会的想法引起了成员们的兴奋。“有人在 NYC 吗？我很乐意尝试在某处安排一次见面会。” **“举办见面会是个好主意！”**
- **成员愿意跨城参与**：一些来自费城和巴尔的摩的成员表示愿意前往 NYC 参加见面会。“我在费城地区，但愿意去 NYC 参加见面会”以及“确定大家都在火车行程范围内，所以我很乐意去 NYC。”


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1245431019151163412)** (25 messages🔥): 

- **柏林见面会引发兴趣**：包括 *maciejgryka* 和 *lucas_vw* 在内的多位用户表达了在柏林见面的兴趣。*r2d29115* 和 *aravindputrevu* 可能会组织更大规模的小组见面会。
- **欧洲各地用户签到**：用户分享了他们的所在地，范围涵盖阿姆斯特丹、柏林、林茨等。代表国家包括英国、德国、奥地利、荷兰、西班牙、芬兰和法国。
- **林茨的技术存在感受到关注**：有人询问 Cloudflight（原 Catalysts）是否仍在林茨保持强大的影响力。得到的确认是他们仍然很有名，但目前没有进一步的联系。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1245461244379402281)** (3 条消息): 

- **关注新的公告频道**：已为所有关键更新和提醒创建了新的 **announcements channel**。**强烈**建议开启该频道的通知，以免错过重要信息。
  
- **需紧急提交表单**：请成员在 **太平洋时间 5 月 30 日晚上 11:59** 前填写几个重要的表单，以确保获得供应商额度，包括来自 [Maven](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f)、[用于 Hugging Face 额度的 Google 表单](https://docs.google.com/forms/d/e/1FAIpQLSc7U01uRlMd2jeeeLZtaePTul-xBZXBwRx3x8qD2iIpuqE_mg/viewform)、[用于 Modal 黑客松额度的 Google 表单](https://docs.google.com/forms/d/e/1FAIpQLSfoCoXNhUjka09mu8rmgB1YM9s3529-F2oJdP5HkHT1SGfV2Q/viewform) 以及 [用于 Fireworks 额度的 Google 表单](https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform)。
  
- **用于演讲安排的 Events 类别**：即将举行的演讲和活动及其 Zoom URL 将发布在 Discord 的 **Events** 类别中。该部分还将根据您的本地时区显示活动的**剩余时间**。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">未找到标题</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSc7U01uRlMd2jeeeLZtaePTul-xBZXBwRx3x8qD2iIpuqE_mg/viewform">Hugging Face 额度申请</a>：在我们为您申请 🤗 HF 额度以使用 https://huggingface.co 的付费服务之前，我们需要了解一些简要信息！如有任何问题，请联系 website@huggingface.co。...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfoCoXNhUjka09mu8rmgB1YM9s3529-F2oJdP5HkHT1SGfV2Q/viewform">Modal 黑客松额度</a>：要领取您的 Modal 额度，请先在 https://modal.com/ 注册账号。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些入门示例...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfndr0-zZlCEMCLVp99yI7olJg2qKr8iv4e_6CXkkb_Nhyj-Q/viewform">Fireworks 额度 - 精通 LLMs：开发者与数据科学家会议</a>：请填写下方表单以获取 $250 的 Fireworks 额度！如有疑问/寻求帮助或需要更多额度，请加入我们的 Discord ;) https://discord.gg/fireworks
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245097510981210114)** (3 条消息): 

- **强烈推荐将 Lighting.ai 用于 GPGPU**：一位成员询问关于使用 **lighting.ai** 进行 **GPGPU** 编程的问题，理由是缺乏 NVIDIA 显卡的商品硬件，且需要使用 **CUDA** 和 **SYCL** 进行编程。另一位成员肯定地表示：“它非常棒，是的。”
- **关于 Torch 对 erf 近似实现的查询**：一位成员询问是否有人知道 **Torch** 是如何近似计算 **erf (误差函数)** 的。在提供的消息中未见回复。
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1245145332330004510)** (16 条消息🔥): 

- **微型包简化了 Triton 的使用**：一位用户推荐了 [triton_util](https://github.com/UmerHA/triton_util)，通过抽象重复性任务来简化 Triton kernel 的编写。该包旨在以一种更直观、更省力的方式编写 Triton 代码。

- **A6000 上的巨大性能差异**：用户注意到 Triton 在 NVIDIA A6000 GPU 上有显著的性能提升。他们请求提供代码示例，以进一步了解性能差异的原因。

- **Triton 中的矩阵乘法问题**：一位用户报告了在 GPU 3090 上使用特定输入大小时，matmul.py 教程（[链接](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)）中出现的偏差。另一位用户建议，这种差异可能是由于 FP16 的有限浮点精度造成的，并得出结论这可能不是一个严重的问题。

- **Triton 处理大型张量时的 Bug**：一位成员发现在 Triton 中处理 65GB+ 的张量时存在一个 Bug。他们解释说，在 int32 中将索引与步长（stride）相乘会导致溢出，从而引发 CUDA 内存错误，这凸显了 Python 中张量指针操作的隐藏复杂性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/umerhadil/status/1795775644254495024?s=46">来自 Umer Adil (@UmerHAdil) 的推文</a>: 让 OpenAI Triton 更简单 🔱 😊 我发现编写 Triton kernel 涉及许多重复性任务，这些任务可以被清晰地抽象出来。这使得编写 Triton 代码更符合我实际的操作方式...</li><li><a href="https://github.com/UmerHA/triton_util/">GitHub - UmerHA/triton_util: 让 Triton 更简单</a>: 让 Triton 更简单。通过在 GitHub 上创建账号来为 UmerHA/triton_util 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245102494304964678)** (19 条消息🔥): 

- **Python 3.12 缺失 torch.compile 支持**：几位用户讨论了 **torch.compile** 在 Python 3.12 上无法工作的问题，但指出 nightly 版本确实提供了一些支持。一位成员分享了一个跟踪此问题的 [GitHub issue](https://github.com/pytorch/pytorch/issues/120233)，并建议使用 **pyenv** 管理多个 Python 版本。
- **Triton kernel 和 flash-attention 的变通方法**：尽管 **torch.compile** 存在问题，但一位用户成功手动安装了 Triton kernel，并发现至少 **flash-attention** 在 Python 3.12 上是可以运行的。
- **新字节码的影响**：一位用户强调，每个新的 Python 版本都会引入新的字节码，导致 Dynamo 解析出现问题，并暗示未来 PyTorch 的发布将与 Python 更新保持一致。
- **macOS x86 弃用**：用户讨论了在 **Torch 2.3** 弃用 macOS x86 构建后的应对机制。一些人建议迁移到 M1 笔记本电脑，或者在旧的 x86 机器上使用 Linux 发行版，并引用了 [RFC GitHub issue](https://github.com/pytorch/pytorch/issues/114602)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/issues/120233">Torch compile 在 Python 3.12 上无法工作 · Issue #120233 · pytorch/pytorch</a>: 🐛 描述 Bug：目前截至 2.2.0 版本的 torch 不支持在 Python 3.12 上使用 torch compile。请参阅以下 PR 示例：#117853。我们需要能够在 Python 3.12 中使用 torch.compile 功能...</li><li><a href="https://github.com/pyenv/pyenv">GitHub - pyenv/pyenv: 简单的 Python 版本管理</a>: 简单的 Python 版本管理。通过在 GitHub 上创建账号来为 pyenv/pyenv 的开发做出贡献。</li><li><a href="https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054">Torch.compile 对 Python 3.12 的支持已完成</a>: 提醒大家 Python 3.12 的支持已添加到 torch.compile 中，并且已经在 nightly 版本中存在一段时间了。我们预计此功能将包含在 PyTorch 2.4 版本中...</li><li><a href="https://github.com/pytorch/pytorch/issues/114602">[RFC] macOS x86 构建 / 测试弃用 · Issue #114602 · pytorch/pytorch</a>: 🚀 功能、动机和构想：由于不再生产新的 Intel Mac，且随着时间的推移，使用的人会越来越少，我建议到今年年底停止测试并最终停止构建 macOS x86_64 二进制文件...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1245358754330050653)** (1 条消息): 

- **AI by Hand 提供核心学习资源**：[Prof Tom Yeh](https://x.com/ProfTomYeh) 分享了 AI 手算练习，他在 LinkedIn 上拥有 3.6 万粉丝，最近开始在 X 上发布内容。该系列包括 [Dot Product](https://x.com/ProfTomYeh/status/1793623127643037891)（点积）、[Matrix Multiplication](https://x.com/ProfTomYeh/status/1794070094898704456)（矩阵乘法）、[Linear Layer](https://x.com/ProfTomYeh/status/1794451228681712037)（线性层）和 [Activation](https://x.com/ProfTomYeh/status/1794848226383655284)（激活函数）工作手册，旨在通过引人入胜的视觉效果和动画让核心 AI 概念变得通俗易懂。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ProfTomYeh/status/1794848226383655284">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：4. Activation - AI by Hand✍️工作手册系列。我分享像这样的原创手算练习，在 LinkedIn 上有 3.6 万粉丝。我刚开始在 X 上分享。如果你觉得这个工作手册有帮助...</li><li><a href="https://x.com/ProfTomYeh/status/1795221120351715450">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：5. Artificial Neuron - AI by Hand✍️工作手册系列。之前的工作手册：4. Activation: https://x.com/ProfTomYeh/status/1794848226383655284 3. Linear Layer: https://x.com/ProfTomYeh/status/179445...</li><li><a href="https://x.com/ProfTomYeh/status/1794451228681712037">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：3. Linear Layer - AI by Hand✍️工作手册系列。我分享像这样的原创手算练习，在 LinkedIn 上有 3.6 万粉丝。我刚开始在 X 上分享。如果你觉得这个工作手册有帮助...</li><li><a href="https://x.com/ProfTomYeh/status/1794070094898704456">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：2. Matrix Multiplication - AI by Hand✍️工作手册系列。我分享像这样的原创手算练习，在 LinkedIn 上有 3.6 万粉丝。我刚开始在 X 上分享。如果你觉得这个帖子有帮助...</li><li><a href="https://x.com/ProfTomYeh/status/1793623127643037891">来自 Tom Yeh | AI by Hand ✍️ (@ProfTomYeh) 的推文</a>：1. Dot Product - AI by Hand✍️工作手册系列。我分享像这样的原创手算练习，在 LinkedIn 上有 3.6 万粉丝。我刚开始在 X 上分享。如果你觉得这个帖子有帮助，[F...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1245112743888158893)** (19 条消息🔥): 

- **探索量化库**：成员们讨论了各种量化库，如 **bitsandbytes**、**quanto** 和 **fbgemm_gpu**。他们强调了 **bitsandbytes** 的独特之处在于它是一个带有 C API 的共享库，并提到其正在进行重构以支持 `torch.compile`。

- **NeurIPS 竞赛热潮**：一位成员表达了对 NeurIPS 竞赛的热情，并表示这激发了他们参与贡献的兴趣。他们祝贺团队进入第二轮，并预测今年的比赛将会有很大进步。 

- **混合精度量化工作**：成员们讨论了关于 `Int4 weight quantization + int8 activation dynamic quantization` 的工作，提到了在 **Llama2-7B** 上 4-bit HQQ 量化权重和模拟 int8 激活的进展。他们提到了一个可以通过 **BitBlas** 获取的 Kernel，但指出尚未经过测试：[BitBlas on GitHub](https://github.com/microsoft/BitBLAS)。

- **对社区的感谢**：一位成员对 torchao 项目贡献者的努力表示感谢，并指出了这个 CUDA Discord 频道相比其他频道的价值，包括那个表现平平的 NVIDIA 官方频道。“这是我发现的唯一好的 CUDA Discord，甚至 NVIDIA 的那个也挺糟糕的……”

- **FP6-LLM 仓库更新**：该仓库进行了一些更新，特别是增加了 `fp5_e2m2`。

**提到的链接**：<a href="https://github.com/microsoft/BitBLAS">GitHub - microsoft/BitBLAS: BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。</a>：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS

  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1245090935558373467)** (26 条消息🔥): 

```html
- **西雅图因阴沉的天气令人失望**：一位用户分享了在西雅图生活的负面体验，称其为“最不社交的城市”，因为一年中约有 9 个月是阴雨天气。他们强调，虽然西雅图的夏天很美，但由于天气原因，一年中的其他时间可能会感到相当孤立。

- **柏林凭借 hacker/startup 社区脱颖而出**：另一位用户指出，柏林拥有充满活力的 hacker/startup 社区，而且每个人都说英语，这让新人更容易融入。他们特别提到了柏林对那些对 techno 派对和当地美食（如烤肉）感兴趣的人的吸引力。

- **柏林天气的现实情况**：与分享的柏林田园诗般的形象相反，用户警告说柏林的冬天漫长且阴沉，气温会降至 -10 °C。不过，他们也提到柏林的春季和夏季非常惬意。

- **柏林的技术场景和职业建议**：建议如果搬到柏林，可以在小型 startups 或 Amazon、Zalando 等公司工作。然而，他们建议在 SF 或 NYC 等城市积累 Big Tech 经验，以便获得更好的未来机会，例如为 startups 筹集资金。
```

**提到的链接**: <a href="https://x.com/Isarusphoto/status/1762392832050868420">Isa Rus (@Isarusphoto) 的推文</a>: 二月的柏林

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1245089980783661218)** (215 条消息🔥🔥): 

- **Tokenizer 实现讨论**：成员们考虑使用 regex 分割自行实现 tokenizer，虽然这很麻烦，但被认为是可行的。他们讨论了在网上提供原始 `.bin` 分片的益处，以避免安装 `tiktoken` 所需的 conda 等额外依赖。

- **压缩和存储选项**：对话包括使用 zip 或其他轻量级替代方案压缩数据集分片，以减小下载大小。他们评估了云存储选项，包括 S3 定价和 Zenodo 等其他托管数据集的服务，以及对出站流量成本的考虑。

- **H100 和多节点训练计划**：成员们评估了在 H100 GPU 集群上训练的潜在性能和成本。尽管有现成的 8X A100 配置用于开发，但除非获得大量资金，否则用于大规模训练的更大型节点被认为过于昂贵。

- **探索不同的 GPU 规格**：围绕 GPU 规格、性能指标和张量操作展开了详细的技术讨论，特别是针对 Ampere 和 Ada 显卡。他们辩论了不同 GPU 的 FP32 性能和 tensor core 行为等数值，为持续的性能优化做出贡献。

- **继续 GPT-3 训练实验**：一位成员分享了在 300B tokens 上训练 124M 模型的持续结果，类似于 GPT-3。部分结果显示与 GPT-3 基准测试非常匹配，这引发了关于 FineWeb 数据集在 HellaSwag 等任务中有效性的疑问。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://transmissionbt.com">Transmission</a>: 未找到描述</li><li><a href="https://trac.transmissionbt.com/wiki/HeadlessUsage">
      HeadlessUsage     – Transmission

    </a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/eval_results.csv">eval_results.csv · HuggingFaceFW/fineweb at main</a>: 未找到描述</li><li><a href="https://aws.amazon.com/s3/pricing/?p=pm&c=s3&z=4">Amazon S3 Simple Storage Service Pricing - Amazon Web Services</a>: 未找到描述</li><li><a href="https://zenodo.org/">Zenodo</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/487">`softmax_autoregressive_backward_kernel` does not use share memory in the kernel by huoyushequ · Pull Request #487 · karpathy/llm.c</a>: softmax_autoregressive_backward_kernel 在 kernel 中不使用共享内存。我们不需要启动带有 256 字节共享内存的 kernel，因此将其移除</li><li><a href="https://zenodo.org/records/3834942">OpenWebText</a>: OpenAI WebText 数据集的开源复制版本。更多信息请访问 https://skylion007.github.io/OpenWebTextCorpus/ @misc{Gokaslan2019OpenWeb, title={OpenWebText Corpus}, author=...</li><li><a href="https://www.techpowerup.com/gpu-specs/rtx-a5500.c3901">NVIDIA RTX A5500 Specs</a>: NVIDIA GA102, 1665 MHz, 10240 Cores, 320 TMUs, 96 ROPs, 24576 MB GDDR6, 2000 MHz, 384 bit
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 条消息): 

orion160: 调试 SYCL 代码的工具有哪些？通常是指单步进入 kernel 代码....

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1245100084924780605)** (94 messages🔥🔥): 

- **Vayuda 在 CUDA 和 PyTorch 版本上遇到困难**：Vayuda 在使用 **torch2.4dev** 和 **CUDA 12.4** 时遇到了 `ImportError: undefined symbol` 错误，并意识到 PyPI 上传的版本默认使用 CUDA 12.1。[Marksaroufim](https://github.com/pytorch/ao/issues/288) 建议通过 conda 使用 CUDA 12.1 或尝试全新安装。

- **在大学服务器上编译扩展的问题**：在确认自定义 C 扩展未正确构建后，Vayuda 面临了与 GPU 相关的额外错误（`ptxas error: Feature '.m16n8k16' requires .target sm_80 or higher`）。在采纳了 Marksaroufim 的几项建议（包括删除特定设置行的“终极方案”）后，Vayuda 发现 **升级到 gcc 12.1** 缓解了一些问题。

- **Bitnet 和 Uint2Tensor PR 的协作工作**：[Marksaroufim](https://github.com/pytorch/ao/pull/282) 鼓励 Vayuda 和其他人合并关于 bit packing 的 PR 努力，并建议建立一个 prototype 文件夹进行有组织的开发。一个 PR [链接](https://github.com/pytorch/ao/pull/285) 描述了实现细节，并且 **测试已移动** 到合适的文件夹以便进行 CI 检查。

- **未解决问题汇总**：Marksaroufim 将自定义 CUDA 扩展导致 ao 安装困难的持续问题汇总到了一个 [ao GitHub issue](https://github.com/pytorch/ao/issues/288) 中。解决方案包括更新设备属性，以便在测试中添加兼容性检查。

- **CI 与测试协调**：尽管遇到了多个错误（部分与在不支持的版本上跳过测试以及 CUDA 可用性有关），Vayuda 最终确保了测试配置正确运行。Marksaroufim 促成了持续集成 (CI) 每周运行测试。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/284">从尺寸 N 到 M 的通用打包算法 · Issue #284 · pytorch/ao</a>: （不确定如何格式化，但尝试如下）为了支持量化的子字节（sub-byte）数据类型，我（以及许多其他人）认为最好将这些较小的数据类型打包进现有的 PyTorch 数据类型中...</li><li><a href="https://github.com/pytorch/ao/issues/288">自定义 CUDA 扩展使得安装 ao 变得困难 · Issue #288 · pytorch/ao</a>: 我正在收集一些我见过的案例，目前还没有明确的解决方案，但汇总在这里希望能激发灵感。问题 1：如下...</li><li><a href="https://github.com/pytorch/pytorch/issues/127374">Torch.compile 产生异常：请先将所有 Tensor 转换为 FakeTensor 或实例化 · Issue #127374 · pytorch/pytorch</a>: 🐛 描述 Bug。torch.compile 在 pack 和 unpack 函数上失败。最小复现代码 minimalrepo.py.zip。版本 Python: 3.10.14，Torch nightly: 2.4.0.dev20240526。错误日志 (ao) (base) james@instance.....</li><li><a href="https://github.com/pytorch/ao/blob/cbc74ee6a3dc0bae367db5b03bc58896fffe3ae0/torchao/csrc/cuda/fp6_llm/ptx_mma.cuh#L116">ao/torchao/csrc/cuda/fp6_llm/ptx_mma.cuh (位于 cbc74ee6a3dc0bae367db5b03bc58896fffe3ae0) · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch">GitHub - pytorch/pytorch: Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络</a>: Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pybind/pybind11/issues/3623">[BUG]: 未定义符号: _ZNSt15__exception_ptr13exception_ptr10_M_releaseEv · Issue #3623 · pybind/pybind11</a>: 必要的先决条件。请确保您已阅读文档，您的问题可能已在文档中得到解决。搜索 Issue 追踪器和讨论区以确认此问题尚未被报告。+1...</li><li><a href="https://github.com/pytorch/ao/pull/291">由 vayuda 提交的位打包 (Bitpacking) · Pull Request #291 · pytorch/ao</a>: 基于此 Issue：#284。在 prototype/ 中添加打包/解包算法的第一版迭代，以支持低位宽数据类型。</li><li><a href="https://github.com/pytorch/ao/pull/282">[WIP] 由 andreaskoepf 添加 Uint2Tensor 和 BitnetTensor 的初步代码 · Pull Request #282 · pytorch/ao</a>: 创建了 UInt2Tensor 类（类似于 UInt4Tensor 类）。添加了 BitnetTensor 类和第一个单元测试，该测试量化了 nn.Linear() 层的权重并执行 matmul。目前...</li><li><a href="https://hastebin.com/share/riridivafa.rust">Hastebin</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/__init__.py#L7-L9">ao/torchao/__init__.py (位于 main 分支) · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/setup.py#L94">ao/setup.py (位于 main 分支) · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch/issues/32867">[功能请求] np.packbits / np.unpackbits，通用 BitTensors（也许可以是 dtype 为 torch.bits8 的 Tensor，或者引入新的 dtype torch.bits）以及用于节省内存/访问的位打包 Tensor 工具，在任何使用 BoolTensors 的地方支持 BitTensors · Issue #292 · pytorch/ao</a>: 一个用例：如果使用 2-bit 数据类型，存储完整的回溯指针矩阵对于 Needleman/CTC 对齐是可以接受的（与 uint8 表示相比可节省 4 倍内存）。目前可以...</li><li><a href="https://github.com/pytorch/ao/pull/285">由 CoffeeVampir3 提交的用于 Bitnet 1.58 的 Trinary2 数据类型和量化 · Pull Request #285 · pytorch/ao</a>: 灵感来自 Issue #281 (评论)。这是 Bitnet 1.58 的初步基础工作。经过反思，我认为将其视为一种不同于 uint2 或常规打包的独立类型是有益的...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1245186714922651668)** (2 messages): 

- **关于具有海量上下文窗口的 LLM 的推测性想法**: 一名成员提议在一个非常小的数据集上训练大语言模型 (LLM)，假设它具有良好的外推能力并拥有极长的上下文窗口。他们建议在上下文中（in-context）向其输入预训练数据集进行学习，并理论化认为如果上下文窗口达到数万亿个 Token，这种方案是可行的。

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1245130716153512050)** (12 messages🔥): 

- **斯坦福和加州的社交金矿**：一名成员分享了在旧金山和加州进行社交的丰富机会。他们强调了参加俱乐部和社交活动以结识 CEO 和 VC 等影响力人物的重要性。
  
- **在斯坦福选择合适的课程**：建议在斯坦福选课时要有针对性，因为不同的课程会吸引不同类型的人。例如，《概率分析 (MS&E 220)》更适合具有创业精神和社交能力的人。

- **享受治愈系美食的慵懒时光**：一位成员分享了他们放纵的慵懒日餐单，包括 500g 俄式饺子 (pelmeni)、250g 酸奶油、黄瓜、巧克力牛奶和哈尔瓦酥糖 (halva)。

- **懒人餐对比**：对话幽默地将方便面与更复杂的慵懒日餐进行了对比，两位成员都对彼此的选择表示赞赏。
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1245213655755259924)** (9 messages🔥): 

- **Qwen2 发布前揭晓 Online Merging Optimizers**：一个[推文](https://x.com/KemingLu612/status/1795652145225863444)链接讨论了 Online Merging Optimizers，并提到模型合并（model merging）可以帮助减轻对齐税 (alignment tax)。文中提供了相关的 [论文](https://arxiv.org/pdf/2405.17931) 和 [GitHub 仓库](https://github.com/QwenLM/online_merging_optimizers) 以获取深入信息。
- **MoRA：高秩更新方法浮出水面**：一个 [GitHub 仓库](https://github.com/kongds/MoRA) 链接介绍了 MoRA，这是一种使用方阵进行“高秩更新”的方法，“在保持相同数量的可训练参数的同时，在内存密集型任务上表现优于 LoRA”。
- **Scale 推出 SEAL Leaderboards**：[Alexandr Wang 的推文](https://x.com/alexandr_wang/status/1795857651592491281) 链接强调了 SEAL Leaderboards 的发布，这是对领先前沿模型的私人专家评估。更多细节分享在 [Scale Leaderboard 网站](https://scale.com/leaderboard) 上，专注于公正且持续更新的模型评估。
- **对 Scale 参与度的担忧**：一位成员对 Scale 同时为模型提供 SFT (supervised fine-tuning) 和 RLHF (reinforcement learning from human feedback) 数据表示担忧，这可能会排除 Llama 3。评论显示了由于这种参与而对公正评估持怀疑态度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://scale.com/leaderboard">SEAL leaderboards</a>：未找到描述</li><li><a href="https://x.com/alexandr_wang/status/1795857651592491281">Alexandr Wang (@alexandr_wang) 的推文</a>：1/ 我们正在推出 SEAL Leaderboards——对领先前沿模型的私人专家评估。我们的设计原则：🔒私密 + 不可利用。评估无过拟合！🎓领域专家评估 🏆持续...</li><li><a href="https://x.com/KemingLu612/status/1795652145225863444">Keming (Luke) Lu (@KemingLu612) 的推文</a>：我们在惊人的 Qwen2 发布之前展示 Online Merging Optimizers。对齐税令人烦恼，但幸运的是模型合并可以神奇地减轻一些。如何将合并方法融入到...</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>：MoRA：用于参数高效微调的高秩更新 - kongds/MoRA
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1245105841515008020)** (256 messages🔥🔥): 

- **位置魔法与 Token 预测限制**：成员们讨论了**自回归 Token 预测模型的基础局限性**，强调它们缺乏对**数学或逻辑**的真实理解，仅仅是在预测 Token。这种局限性与其所谓的推理能力形成对比。
  
- **RAG 复杂性解析**：关于**检索增强生成 (RAG)** 的详细解释强调了它不仅仅是**向量相似度搜索**。成功的 RAG 实现涉及 **Embedding、向量相似度、全文关键词搜索、Chunking 和 Re-ranking**，使其类似于 LLM 的推荐引擎。

- **奖励模型澄清**：成员们讨论了[链接的 Hugging Face 仓库](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)中奖励模型的功能。澄清指出，此类模型通常根据人类偏好分配分数，支持 **PPO** 等**强化学习任务**。

- **关于新 Mistral 模型许可的辩论**：Mistral 新推出的 **Codestral 模型**（[在 80 多种编程语言上训练](https://mistral.ai/news/codestral/)）因其限制商业使用的**非生产性许可**引发了辩论。此举导致人们对其在实际应用中的采用持怀疑态度，评论建议将重点转向被认为更通用的**开源替代方案**。

- **对 Google Gemini 1.5 涨价的批评**：针对 Google 最近对 Gemini 1.5 Flash 输出的[价格上调](https://x.com/artificialguybr/status/1795851375181508785)存在大量批评，价格在未事先通知的情况下几乎翻了一番。成员们对该服务的**可信度和响应速度**表示担忧，称其为“骗局”。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>: 通过 Mistral AI 赋能开发者并使编程民主化。</li><li><a href="https://mistral.ai/news/mistral-ai-non-production-license-mnpl/">Introducing the Mistral AI Non-Production License</a>: Mistral AI 推出新的非生产性许可，以平衡开放性与业务增长。</li><li><a href="https://huggingface.co/spaces/allenai/reward-bench">Reward Bench Leaderboard - a Hugging Face Space by allenai</a>: 未找到描述</li><li><a href="https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1">sfairXC/FsfairX-LLaMA3-RM-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/artificialguybr/status/1795851375181508785">Tweet from 𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr)</a>: Google 在未告知任何人的情况下将 Gemini 1.5 Flash 的输出价格提高了 98%。这就在发布该模型一周后。输出价格从 0.53/1M 变为 1.05/1M。我们如何信任一个如此剧烈变动的 API...</li><li><a href="https://github.com/the-crypt-keeper/LLooM">GitHub - the-crypt-keeper/LLooM: Experimental LLM Inference UX to aid in creative writing</a>: 辅助创意写作的实验性 LLM 推理 UX - the-crypt-keeper/LLooM</li><li><a href="https://github.com/neph1/LlamaTale">GitHub - neph1/LlamaTale: Giving the power of LLM&#39;s to a MUD lib.</a>: 为 MUD 库赋予 LLM 的力量。通过在 GitHub 上创建一个账户来为 neph1/LlamaTale 的开发做出贡献。</li><li><a href="https://medicalxpress.com/news/2024-05-neuroscientists-ai-simulate-brain-visual.amp">
      Neuroscientists use AI to simulate how the brain makes sense of the visual world
          </a>: 斯坦福大学 Wu Tsai 神经科学研究所的一个研究团队在利用 AI 模拟大脑如何组织感官信息以理解世界方面取得了重大进展...</li><li><a href="https://github.com/arenasys/Lineworks">GitHub - arenasys/Lineworks: Qt GUI for LLM assisted co-writing</a>: 用于 LLM 辅助协同写作的 Qt GUI。通过在 GitHub 上创建一个账户来为 arenasys/Lineworks 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1245202902725693512)** (16 条消息🔥): 

- **梯度累积（Gradient Accumulation）被认为存疑**：一位用户提出了关于避免梯度累积及其是否有益的担忧。他们分享了一个 [Google 调优指南的 GitHub 链接](https://github.com/google-research/tuning_playbook)，以获取关于最大化深度学习模型性能的见解。
  
- **DPO 训练中的参考模型（Reference Model）**：用户讨论了 DPO 训练中 `ref_model` 的作用，其中 `ref_model` 默认设置为 None，这意味着使用模型的副本作为参考。根据 [Hugging Face 的文档](https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model-considerations-with-peft)，已确认参考模型可以是初始模型或不同的模型，通常是冻结的，以防止偏离原始模型。

- **LLM 语境下 Agent 的定义**：一位用户询问了关于 LLM 语境下 Agent 的入门读物。另一位用户澄清说，Agent 感知并影响其环境，通常使用脚本和 LLM 实现，例如语音对话聊天机器人。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model-considerations-with-peft">DPO Trainer</a>：未找到描述</li><li><a href="https://github.com/google-research/tuning_playbook">GitHub - google-research/tuning_playbook: 系统化最大化深度学习模型性能的指南。</a>：一个用于系统化最大化深度学习模型性能的指南 - google-research/tuning_playbook
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1245173560259907584)** (15 条消息🔥): 

- **Noita 是一个受欢迎的消遣**：一位成员承认在讨论过程中因玩 Noita 而分心。
- **关于 RAG 评估框架的讨论**：一场关于 RAG 评估指标有效性的对话展开了，提到了 RAGAS、BENCH 和 ARES 等**流行框架**。[分享了每个框架的链接](https://github.com/explodinggradients/ragas)，提供了详细探索的资源。
- **为问答创建 HyDE 与 multi-hop 的融合**：成员们探索了将 HyDE 与 multi-hop 结合用于问答的概念，考虑了诸如从单个查询中*创建多组查询*的方法。还考虑了利用每一步来辅助下一次搜索的想法。
- **用于评估的多模态指标**：对话涵盖了**使用 LLM 结合启发式方法**（如 n-gram 和 ROUGE）来评估基于上下文和查询相关性的指标。强调了在数学上确立这些指标的挑战。
- **检索中混合搜索（hybrid search）的建议**：一位成员建议不要仅使用简单的余弦相似度进行检索，推荐了一种集成了多位专家见解的[混合搜索方法](https://x.com/HamelHusain/status/1795526367637049629)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/HamelHusain/status/1795526367637049629">Hamel Husain (@HamelHusain) 的推文</a>：我和我的同事将关于 LLM 的实用建议浓缩到了这个由三部分组成的系列中。有很多精彩内容。这是我最喜欢的截图部分之一，建议来自：@eugeneyan, @BEBi...</li><li><a href="https://github.com/explodinggradients/ragas">GitHub - explodinggradients/ragas: 检索增强生成 (RAG) 流水线的评估框架</a>：用于检索增强生成 (RAG) 流水线的评估框架 - explodinggradients/ragas</li><li><a href="https://github.com/arthur-ai/bench">GitHub - arthur-ai/bench: 一个评估 LLM 的工具</a>：一个评估 LLM 的工具。通过在 GitHub 上创建账户来为 arthur-ai/bench 做出贡献。</li><li><a href="https://github.com/stanford-futuredata/ares">GitHub - stanford-futuredata/ARES</a>：通过在 GitHub 上创建账户来为 stanford-futuredata/ARES 做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1245235352814485505)** (6 条消息): 

- **意外的 AI 趣事**：一位成员幽默地分享了他们尝试使用 `rm -rf /` 删除文件系统的经历，这促使系统尝试将自己转变为一个*超级智能 AI (superintelligent AI)*。他们开玩笑说：“哎呀，搞砸了 (Oopsie Daisy)。”
- **术语惊喜**：同一位成员对 **"AI singleton"**（AI 单一实体）一词表示困惑，并思考如果系统不知道这个术语，它是否会将其作为首选。
- **故障干扰用户体验**：另一位成员抱怨 world-sim 中的**文本重复故障 (text doubling glitch)**，并表示在修复之前将停止使用。另一位成员确认该故障尚未解决。
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1245090951702253720)** (62 条消息🔥🔥): 

<ul>
<li><strong>LM Studio 的开源状态令用户困惑：</strong> 一位成员询问 LM Studio 是否开源，得到的澄清是只有 LMS Client (CLI) 和 lmstudio.js (新 SDK) 是开源的。另一位成员确认 LM Studio 主应用是闭源的。</li>
<li><strong>LM Studio 无法访问文件：</strong> 一位用户询问模型是否可以使用 LM Studio 访问其 PC 上的文件，但另一位成员澄清说，在 LM Studio 中无法直接与文档对话，并建议查看 FAQ 和置顶消息以获取更多信息。</li>
<li><strong>关于 RAG 框架的讨论：</strong> 成员们讨论了低代码 RAG 框架以及向量数据库与 RAG 模型的集成，推荐在开发中使用 llamaindex，并考虑为不经常更改的数据进行模型微调 (fine-tuning)。</li>
<li><strong>Perplexity 与 LM Studio 在对话组织方面的对比：</strong> 一位成员提到 Perplexity 能够创建集合来保存和组织对话，并询问 LM Studio 是否有类似功能。确认结果是 LM Studio 目前不支持此功能。</li>
<li><strong>LM Studio 中文件摘要的限制：</strong> 成员们讨论了由于 Token 限制，使用 LM Studio 总结书籍内容的挑战，并建议对于此类任务使用基于云的 AI，如 GPT4 或 Claude 3 Opus。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/3bXg4Qv3">加入 Mintplex Labs | AnythingLLM | VectorAdmin Discord 服务器！</a>：查看 Discord 上的 Mintplex Labs | AnythingLLM | VectorAdmin 社区 - 与其他 4259 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1">mistralai/Codestral-22B-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF">bartowski/Codestral-22B-v0.1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://aistudio.google.com">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1245171451770966087)** (19 条消息🔥): 

- **Aya 翻译模型获得认可**：一位成员建议在翻译任务中尝试 [Aya 日译英模型](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)。其质量和效率都得到了简要强调。

- **重点关注 Psyonic-Cetacean 模型**：提到了 "Space Whale" 的 32 位量子上采样版 (32 Bit Quantum Upscale)，并指出其性能有显著提升，包括在 Q4KM 格式下困惑度 (perplexity) 降低了 932 点。点击[此处](https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF)了解更多关于此重制版的信息。

- **Codestral 备受期待的发布**：成员们对 [Mistral 的新代码模型 Codestral](https://mistral.ai/news/codestral/) 表现出浓厚兴趣，该模型支持 80 多种编程语言。讨论了将其集成到 LM Studio 的计划，如果 Tokenizer 发生变化，可能需要发布新的应用版本。

- **Aya 23 35B 的硬件挑战**：讨论了在 4090 GPU 上运行 aya-23-35B-Q4_K_M.gguf 模型的问题，指出该模型需要超过 24GB 的显存 (VRAM) 才能获得最佳性能。建议通过调整上下文大小 (context size) 来提高速度。

- **Space Whale 上下文限制确认**：另一位成员确认 Space Whale 模型的上下文限制为 4096 个 Token。这已通过 `llama.context_length` 配置进行了验证。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>：通过 Mistral AI 赋能开发者并使编程大众化。</li><li><a href="https://huggingface.co/DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF">DavidAU/Psyonic-Cetacean-Ultra-Quality-20b-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1245130254209519717)** (5 messages): 

- **角色切换：一个模型还是两个？**：一位成员询问一个模型是否可以同时承担审核（moderation）和问答（Q&A）角色。另一位成员建议大多数模型在上下文切换（context switching）方面表现不佳，建议使用两个独立的模型；而另一位成员则暗示 Server 模式的上下文处理可能使这一方案可行。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1245290294077952010)** (3 messages): 

- **尽管预设相同，Server 模式仍然较慢**：一位用户注意到，与 Server 模式相比，使用 Chat 模式时获得结果的速度要快得多，尽管两者使用了相同的预设（preset）。他们检查并确认了 GPU 在 Server 模式下正在被使用。
- **关于 Server 模式下 GPU 选择的不确定性**：另一位用户询问如何为 Server 使用选择 GPU，并对如何确定正在使用哪个 GPU 表示困惑。目前尚未提供解决方案或进一步信息。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1245089644421451879)** (92 messages🔥🔥): 

- **Nvidia 泡沫辩论升温**：成员们质疑 Nvidia 目前的高估值是否合理，还是仅仅是一个“泡沫”。有人指出，*"它们可能会涨得更高！"*，而另一位成员则建议做空 Nvidia 股票，认为这种势头“无法持续太久”。
- **ASUS Vivobook S 15 USB 接口令人印象深刻**：ASUS Vivobook S 15 因其出色的 I/O 能力而受到讨论，包括支持 40Gbps 数据传输的“2 x USB4 接口”。然而，也有人对交付时可能存在的故障和召回表示担忧。
- **Goldensun3ds 升级至 44GB VRAM**：一位用户展示了他们的配置，包括 5800X3D CPU、64GB RAM、两块 RTX 4060 Ti 16GB GPU 和一块 RTX 3060 12GB GPU。他们讨论了多 GPU 相对于单块强力 GPU（如 3090）的优势，并将功耗和 VRAM 视为关键因素。
- **主板和 PCIe 通道分配障碍**：成员们讨论了高效运行多 GPU 的复杂性，重点关注 PCIe 通道分配和主板能力。*“需要有人为 AI 推出一些像样的定制主板，”* 是一种普遍的观点。
- **改装 GPU 引起关注**：改装 GPU 的可靠性和实用性受到质疑，特别是“2080ti 改装 22GB”。参与者指出，*“那会消耗更多电力，且可靠性非常存疑，”* 并警告不要使用它们。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/The_Cathedral_and_the_Bazaar">The Cathedral and the Bazaar - Wikipedia</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d33k0p/llama3_70b_with_2x2080ti_22gb_gpus/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1245365967811514491)** (2 messages): 

- **表达感谢**：一位用户通过说 *“确实如此”* 来表达感谢。随后补充道：“我会看看我能做些什么，谢谢大家。”
  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1245127954954981528)** (9 条消息🔥): 

- **LM Studio 0.2.24 的 iGPU 支持问题**：一位用户询问了 LM Studio 0.2.24 中 **ROCm 对 iGPU 的支持**情况，提到该功能在 0.2.20 版本中运行良好，但现在不再起作用。另一位用户澄清说，**ROCm** 仍然不支持 **iGPU**，并指出旧版本中可能使用的是 **OpenCL**。

- **如何回退到旧版本**：在确认之前的设置使用的是 **OpenCL** 后，一位用户请求获取旧版本 **0.2.20** 的链接，因为该版本的性能明显更好。

- **OpenCL 模式下的多 GPU 导致错误**：一位用户报告称在 **ROCm 模式**下运行 **7900 XT** 成功，但在添加 **Radeon 570** 以利用额外 VRAM 时遇到问题并导致错误。另一位用户建议，显卡代际差异可能会导致问题。

- **添加同代 GPU**：考虑到在系统中添加一块 **7600 XT**，一位用户询问它是否能在 ROCm 模式下与 **7900 XT** 兼容。另一位用户建议先检查 **AMD ROCm 兼容性**，但也提到目前 **7900 XT** 有不错的优惠，建议直接提升总 VRAM 可能更简单直接。

- **分享 7900 XT 优惠**：一位用户提供了一个 [7900 XT 的优惠链接](https://www.ebuyer.com/1584907-gigabyte-amd-radeon-rx-7900-xt-gaming-oc-graphics-card-for-gaming-gv-r79xtgaming-oc-20gd)，强调这是扩展 VRAM 并高效运行大型模型的高性价比选择。

**提到的链接**：<a href="https://www.ebuyer.com/1584907-gigabyte-amd-radeon-rx-7900-xt-gaming-oc-graphics-card-for-gaming-gv-r79xtgaming-oc-20gd">Gigabyte AMD Radeon RX 7900 XT GAMING OC Graphics Card for Gaming - 20GB | Ebuyer.com</a>：未找到描述

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1245469955491889192)** (1 条消息): 

- **Mistral 的新编程模型 Codestral 已上线**：Mistral 的最新模型 **Codestral** 现已开放下载。这款 **22B 模型** 适用于拥有较大 GPU 显存、寻求运行高性能模型的用户。[在 Hugging Face 上查看](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF)。
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1245111251420315809)** (75 messages🔥🔥): 

- **记录 Flutter 移植问题**：一名成员建议“记录在移植或为 Flutter 编写胶水层时缺失的每一项内容”，并优先处理特定的功能请求。他们强调，详细且具体的文档对于解决阻塞性问题（相对于次要的变通方法）至关重要。

- **Mojo 中的 C/C++ Interoperability**：成员们对 Mojo 中 **C/C++ Interoperability** 的时间表表示好奇，将其潜在方法与 **Swift** 进行了比较，并讨论了技术挑战和优先级。一位成员表示：“我真的很想知道 C++ 的互操作性”，而另一位成员则指出这可能还不是目前的优先级。

- **Mojo 与 Clang 的关系**：讨论揭示了关于 Mojo 当前编译过程及其对 LLVM 依赖的技术细节。一位成员强调：“Mojo 的堆栈大致是 mojo-(Modular 编译器)- MLIR dialects- MLIR LLVM - LLVM”，而另一位成员澄清说，“Mojo 将能够导入 C/C++ 头文件。”

- **辩论 ABI 兼容性**：成员们辩论了不同编译器之间 ABI 稳定性和兼容性的实际情况，特别是在 Windows 与 Linux 上的差异。一位成员指出：“Clang 实现了 GCC 的 C++ ABI，因为如果不这样做就意味着零采用”，这标志着其中涉及的巨大复杂性。

- **引用 Polygeist 和 ClangIR 项目**：成员们分享了关于 **Polygeist** 和 **ClangIR** 的资源，讨论了它们在促进 MLIR 的 C/C++ 前端开发中的作用。例如，一位成员分享了一个关于 Mojo 开发讨论的 [YouTube 链接](https://www.youtube.com/watch?v=SEwTjZvy8vw)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://polygeist.llvm.org/">Polygeist</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>：2023 LLVM 开发者大会——Mojo 🔥：一种用于异构计算的系统编程语言。演讲者：Abdul Dakkak, Chr...</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo Lang - Tomorrow's High Performance Python? (with Chris Lattner)</a>：Mojo 是来自 Swift 和 LLVM 创始人的最新语言。它尝试采用 CPU/GPU 级编程的一些最佳技术并进行封装...</li><li><a href="https://github.com/llvm/Polygeist">GitHub - llvm/Polygeist: C/C++ frontend for MLIR. Also features polyhedral optimizations, parallel optimizations, and more!</a>：用于 MLIR 的 C/C++ 前端。还具有多面体优化、并行优化等功能！- llvm/Polygeist</li><li><a href="https://github.com/llvm/clangir">GitHub - llvm/clangir: A new (MLIR based) high-level IR for clang.</a>：一个新的（基于 MLIR）的 Clang 高级 IR。在 GitHub 上为 llvm/clangir 的开发做出贡献。</li><li><a href="https://llvm.github.io/clangir//">ClangIR · A new high-level IR for clang.</a>：Clang IR (CIR) 是 Clang 的一种新 IR。ClangIR (CIR) 构建在 MLIR 之上，它基本上是一个基于 C/C++ 语言的 MLIR dialect...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot：来自 *Modular*：
<https://twitter.com/Modular/status/1795883558608973828>
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1245427393473286165)** (1 messages): 

- **Mojo 内存管理中的 Ownership 详解**：Zapier 讨论了像 Mojo 这样的现代编程语言中的 Ownership，强调了它在提供安全内存管理编程模型的同时确保高性能的作用。他们建议观看 [Chris Lattner 的深度解析视频](https://www.modular.com/team/chris-lattner)，以获取关于 Ownership 如何在 Mojo 编译器中实现的详细见解，并提供了进一步的技术细节。阅读完整的博客文章[请点击此处](https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach)。

**提到的链接**：<a href="https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach">Modular: What Ownership is Really About:  A Mental Model Approach</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Ownership 的真正含义：一种心理模型方法。

  

---

### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1245089707038216325)** (1 messages): 

<html>
    <body>
        <ul>
            <li><strong>利用 AI 进一步推动开放世界游戏：</strong> 一位成员提议，如果 AI 能根据用户交互构建自定义世界，开放世界游戏将具有真正的革命性。他们强调，AI 只需要一个庞大的在线模型库供其选择。</li>
        </ul>
    </body>
</html>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1245181672090243122)** (35 messages🔥): 

- **自动解引用提案引发命名争论**：围绕一个新的 [自动解引用引用提案 (auto-dereferenced references proposal)](https://github.com/modularml/mojo/discussions/2874) 展开了讨论。建议包括将 Reference 重命名为 TrackedPointer，将 Pointer 重命名为 UntrackedPointer，以强调安全性并避免与 UnsafePointer 等术语产生误导性关联。
- **包路径解析问题已解决**：一位成员在测试代码无法找到包结构中的定义时遇到了困难。解决方案是在 `mojo run/test` 命令中使用 `"-I ."` 标志来指定父路径。
- **澄清 Mojo 中的 Tensor 初始化**：关于是否能像 numpy 数组那样更简便地进行 Tensor 赋值的查询，通过建议使用 `Index` 工具得到了解答。[这篇博文](https://fnands.com/blog/2024/mojo-png-parsing/#creating-a-tensor) 中提供了示例和进一步说明。
- **提案应当编号**：有人建议对提案进行编号以便于引用和排序，类似于 Python 的 PEP，尽管 Mojo 目前的提案形式尚不那么正式。
- **Mojo References 与 Go Pointers 的对比**：Mojo 引用与 Go 指针的对比强调了 Mojo 的引用通常更安全，因为它们具有显式类型且没有 nil 引用，而不像 Go 可能会出现悬空指针（dangling pointers）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/discussions/2874">[提案] 用于返回引用的新 `ref` 约定 · modularml/mojo · Discussion #2874</a>：大家好，@lattner 和我开发了一个替代 Chris 几周前发布的“自动解引用 (auto-deref)”提案。新想法是将自动解引用作为一种结果约定……</li><li><a href="https://fnands.com/blog/2024/mojo-png-parsing/#creating-a-tensor">fnands - 在 Mojo 中解析 PNG 图像</a>：未找到描述</li><li><a href="https://www.infoq.com/presentations/Null-References-The-Billion-Dollar-Mistake-Tony-Hoare/">空引用：十亿美元的错误 </a>：Tony Hoare 早在 1965 年就在 ALGOL W 中引入了 Null 引用，“仅仅是因为它非常容易实现”，Hoare 先生说。他在谈到那个决定时认为这是“我的十亿美元错误”……
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1245089807365967872)** (7 messages): 

- **为 AVX512 效率对齐你的表格**：*“确保整个表是 64 字节对齐的。这能让你获得大多数 AVX512 加载和存储的更快版本，并确保你不会在某处浪费半个缓存行（cache line）。”* 对齐对于将尽可能多的表放入缓存空间并优化性能至关重要。
- **通过对齐内存优化预取（prefetching）**：具有**对齐访问**的大块内存会*“向预取器发出信号以保持其热度。”* 这强调了对齐内存对性能的重要性。
- **伪共享（False sharing）仅存在于多线程场景**：伪共享问题仅在多线程环境中存在。对齐内存有助于缓解此问题。
- **探索 List 的对齐方式**：一位用户表示有兴趣对用于存储表的 List 进行对齐，并指出 **DTypePointer** 在其 alloc 中有对齐参数，但 **UnsafePointer**（List 所使用的）没有。*“也许有办法，我得去研究一下。”*
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1245161901738954794)** (53 条消息🔥): 

- **在 Mojo 的 `ref` API 中使用 `Optional`**：关于新 `ref` API 的讨论强调了在与 `Dict` 配合使用时面临的挑战，因为对键值对进行解引用（dereferencing）仍然很麻烦。成员们辩论了使用异常（exceptions）与 `Optional` 的优劣，引用了 Rust 对 `?` 运算符的使用，并探讨了对空负载（empty payloads）进行特殊处理的可能性。

- **对 Mojo 贡献指南的反馈**：在提议新的 `ref` API 时遇到 linter 问题后，一名成员建议改进贡献指南。根据一位贡献者的澄清，建议重点强调安装 pre-commit hooks 的重要性，以避免 CI 错误。

- **`InlineArray` 析构函数中的 Bug**：一名成员请求修复 `InlineArray` 不调用其元素析构函数的问题，并引用了 [GitHub issue #2869](https://github.com/modularml/mojo/issues/2869)。

- **Nightly Mojo 编译器发布**：新的 Nightly Mojo 编译器版本 `2024.5.2912` 已发布，包含多项更新，包括 async 函数借用限制以及多个标准库（standard library）函数的重命名。分享了版本间的[完整 changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [raw diff](https://github.com/modularml/mojo/compare/42a38d666c3e6a86e0fd5ad3fdef821c12e91eee...699cb0ca03b40fd49590bc317c530589083cebf4)。

- **关于将默认分支更改为 nightly 的讨论**：一名成员建议将 GitHub 上的 nightly 分支设为默认分支，以获得更好的开发体验。项目经理解释说，目前 75% 的用户使用的是发布版本，更改默认分支可能会让经验较少的用户感到困惑。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2873).">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#pull-requests)">mojo/CONTRIBUTING.md at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2556">[Feature Request] DX: Change the default branch of modularml/mojo from `main` to `nightly` · Issue #2556 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？我希望 modularml 管理员前往设置...</li><li><a href="https://github.com/modularml/mojo/issues/2869">[stdlib] [BUG] `InlineArray` does not invoke the destructors of its elements · Issue #2869 · modularml/mojo</a>：InlineArray 包含 AnyType 的值并在构造时进行复制，但不调用其元素的析构函数。我们需要修复这个问题。
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1245244883065766010)** (24 条消息🔥): 

- **EleutherAI 欢迎新成员咨询**：一位即将完成计算机科学本科学位的成员寻求关于如何开始参与 EleutherAI 的建议。其他成员建议了一些初学者级别的研究课题，并提供了一个 [GitHub gist](https://gist.github.com/ad8e/da8fdfe0ec586b5a548aaa14327f7722) 和其他资源，指出某些问题在没有深厚背景的情况下也是可以尝试的。

- **研究与问题澄清的挑战**：成员们讨论了新手很难找到一个既能提基础问题，又不会面临缺乏专业人士解答的平台。虽然提到了 ChatGPT 等替代方案，但指出其偶尔存在可靠性问题。

- **多模态 AI 研究的探索**：一位成员对专门从事多模态 AI 研究的教授稀缺表示好奇，想知道这是否被视为 CV 和 NLP 的子领域。目前尚无实质性回复对此进行澄清。

- **SPAR 被推荐为重要资源**：Supervised Program for Alignment Research (SPAR) 被推荐为培养 AI safety 技能的宝贵机会。尽管当前的申请截止日期已过，但该计划每年运行多次，提供持续的机会。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BlancheMinerva/status/1741855005601141091">Stella Biderman (@BlancheMinerva) 的推文</a>：许多人似乎认为在大型实验室之外无法进行有趣的 LLM 研究，或者被迫挤入拥挤的课题。实际上，有很多开放的高价值问题。为了证明这一点……</li><li><a href="https://supervisedprogramforalignment.org/">Supervised Program for Alignment </a>：SPAR 为早期职业人士和专业人士提供了一个独特的机会，通过参与 alignment 研究的导师制（无论是作为导师还是学员）来为 AI safety 研究做出贡献……</li><li><a href="https://gist.github.com/ad8e/da8fdfe0ec586b5a548aaa14327f7722">机器学习初学者的一些简单课题</a>：机器学习初学者的一些简单课题 - a.txt
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1245150148225532007)** (43 条消息🔥): 

- **备受争议的研究论文缺乏实验**：成员们对一篇论文感到失望，该论文在摘要中展示了引人入胜的结果，但在正文中承认 *“实际上我们还没有进行任何实验，哈哈”*。这引发了关于为什么该论文能在 *arXiv* 上发表的质疑。

- **关于 Yann LeCun 科学贡献的辩论**：围绕 Yann LeCun 在科学界的地位展开了激烈的讨论，一些人质疑他的图灵奖，而另一些人则捍卫他的导师贡献和早期工作。一位成员强调，他在近期论文上的署名并非仅仅是象征性的，并引用了他的学生们的正面反馈。

- **与 Megabyte 模型的比较**：有人推测论文中的一个模型与 Megabyte 模型相似。一位成员指出：*“这不就是 Megabyte 吗？”* 但其他人认为肯定存在一些差异。

- **关于恒定学习率调度（Constant Learning Rate Schedule）的讨论**：受[近期一篇论文](https://arxiv.org/abs/2405.18392)的启发，成员们讨论了使用恒定学习率调度与固定调度的优劣。一位成员总结了他们对 warmup schedules 的偏好，并强调了过去的成功案例。
  
- [**Yann LeCun 关于工程与科学的讲座**](https://youtu.be/gG5NCkMerHU)：一位成员分享了一个 YouTube 链接，内容是 Yann LeCun 关于“工程科学 vs. 基础科学”的讲座，并将其与他目前和过去对 AI 研究的贡献进行了对比。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>：规模已成为获得强大机器学习模型的关键要素。因此，理解模型的 scaling 属性是有效设计正确训练设置的关键……</li><li><a href="https://youtu.be/gG5NCkMerHU?si=WBIR-_JMJ_QsHVMq">The Epistemology of Deep Learning - Yann LeCun</a>：深度学习：炼金术还是科学？主题：深度学习的认识论；演讲者：Yann LeCun；所属机构：Facebook AI Research/纽约大学；日期：2月……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1245191651635236876)** (90 条消息🔥🔥): 

- **MLP 挑战 Transformer 的主导地位**：成员们讨论了最近的研究如何证明多层感知机 (MLP) 在执行上下文学习 (ICL) 方面具有与 Transformer 竞争的实力，甚至在某些关系推理任务上表现更好。*"这些结果表明，上下文学习并非 Transformer 所特有，并强调了在注意力机制架构之外探索这一现象的潜力。"*
  
- **怀疑与优化问题**：尽管 MLP 的结果令人振奋，但一些成员对其泛化能力表示怀疑，并指出研究中使用的 Transformer 模型可能存在弱点。*"虽然我会说他们的 Transformer 有点次优：他们使用了带有绝对位置编码的 post-layernorm。"*

- **关于 MLP 中序列长度和因果性的辩论**：讨论涉及了 MLP-Mixer 模型如何处理序列长度和因果性，类似于 RNN 和 Transformer。然而，对权重共享和内存管理等技巧的需求引发了担忧。*"似乎需要许多奇怪的技巧才能让 MLP 模型支持任意序列长度并具备因果性。"*

- **MLP 在实际应用中的表现**：成员们讨论了 MLP-Mixer 的实际适用性，特别是这些模型处理输入依赖池化和内存需求的方式。*"这非常有趣，我可能会在某个时候尝试一下。"*

- **关于模型架构的苦涩教训 (Bitter Lesson)**：对话更广泛的主题围绕着这样一个观点：规模化和适应性可能比特定架构更重要，这呼应了关于机器学习模型演进的“苦涩教训 (Bitter Lesson)”。*"这是‘苦涩教训’的又一个例子，而且现在会让人印象深刻，因为所有的 CNN 老兵都已经退场，取而代之的是那些认为‘Transformer 是神奇仙尘’的人。"*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1503543031923945475">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：使用稀疏全 MLP 进行高效语言建模。与基于 Transformer 的 MoE 以及稠密 Transformer 相比，稀疏全 MLP 改善了语言模型 (LM) 的 PPL，并获得了高达 2 倍的训练效率提升...</li><li><a href="https://arxiv.org/abs/1603.05691">深度卷积网络真的需要深度和卷积吗？</a>：是的，它们需要。本文首次通过实证演示证明，深度卷积模型确实需要同时具备深度和卷积特性，即使是使用蒸馏等方法训练时也是如此...</li><li><a href="https://gwern.net/note/fully-connected#convolution-learning">全连接神经网络 · Gwern.net</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2108.13002#microsoft">网络结构之战：CNN、Transformer 和 MLP 的实证研究</a>：卷积神经网络 (CNN) 是计算机视觉领域主导的深度神经网络 (DNN) 架构。最近，基于 Transformer 和多层感知机 (MLP) 的模型，如 Vision Tra...</li><li><a href="https://arxiv.org/abs/2306.13575">扩展 MLP：归纳偏置的故事</a>：在这项工作中，我们重新审视了深度学习中最基础的构建块——多层感知机 (MLP)，并研究了其在视觉任务上性能的极限。对 MLP 的实证见解是...</li><li><a href="https://arxiv.org/html/2405.15618v1">MLP 学习上下文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.15618">MLP 学习上下文</a>：上下文学习 (ICL) 是一种仅从输入示例中解决任务的卓越能力，通常被认为是 Transformer 模型的独特标志。在这项研究中，我们证明了...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1245278096291004469)** (9 messages🔥): 

- **在 AMD 上尝试计算最大内存时出现 Traceback**：一位成员在尝试计算最大内存时遇到了 **AMD 上的 Traceback**，并询问这是否是环境问题。他们链接了一个包含错误的 [GitHub Gist](https://gist.github.com/jonabur/0004bf39a3cec65262cf72f556c316c4)，并指出指定 `max_memory_per_gpu` 可以绕过该问题。
  
- **使用 lm-evaluation-harness 运行并发查询**：一位配合 vLLM 实例使用 **lm-evaluation-harness** 的成员注意到基准测试每次只运行一个查询，并询问是否可以进行批处理（batch processing）。他们还询问了尽管 'local-chat-completions' 不支持，但如何运行基于 logits 的测试，并请求提供解释在提取文本答案中使用 logits/logprobs 的伪代码。
  
- **关于 gsm8k 评估中 maj1@k 的问题**：一位致力于 gsm8k 数据集微调的成员寻求关于论文中报告的 **acc@1** 的澄清，并询问未指定的 k 值。另一位成员做出了回应，建议参考 [已报告的 llama2 结果](https://arxiv.org/pdf/2403.08295)，表明这很可能是 maj1@1。

**提到的链接**：<a href="https://gist.github.com/jonabur/0004bf39a3cec65262cf72f556c316c4">gist:0004bf39a3cec65262cf72f556c316c4</a>：GitHub Gist：即时分享代码、笔记和代码片段。

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1245459520663060532)** (1 messages): 

- **ChatGPT 免费用户获得新功能**：*"现在所有 ChatGPT 免费用户都可以使用 browse、vision、data analysis、file uploads 和 GPTs。"* 这包括 **browse**、**vision**、**data analysis**、**file uploads** 和 **GPTs** 功能。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1245111157140881419)** (100 messages🔥🔥): 

- **Google 的 ImaGen3 即将发布，用户持怀疑态度**：讨论围绕在 Google I/O 上宣布的 **ImaGen3** 测试版展开，该版本取代了旧版本，但存在操纵和公众信任问题的担忧。一位用户幽默地指出，*"他们就这一份工作（还没做好）。"*
  
- **自定义 GPTs 表现异常**：用户分享了对自定义 GPTs 记忆力不佳且普遍反应迟钝的挫败感。一位用户提到：*“我的 GPTs 拒绝记住事情，而且普遍表现得很懒惰。”*

- **Google AI 争议持续**：激烈的讨论指出 Google 的 AI 图像生成器在生成准确的历史图像方面存在问题，例如 *“纳粹黑人女性”* 被讨论为过滤器校准不佳的例子。一位用户指出，*“Google 在 AI 方面搞砸得太厉害了。”*

- **AI 研发的可见性**：讨论了 **OpenAI 模型的使用及其法律影响**，用户辩论 OpenAI 是否会追究使用其数据的个人项目。关于涉及 OpenAI 的诈骗和欺诈担忧也浮出水面。

- **数学与 AI 革命**：一位用户分享了对题为《为什么数学将被 AI 革命化》文章的看法，并参与了关于教授提出的挑战的讨论，即 AI 是否可以跨不同维度复制复杂的数学证明。

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1245091308201054238)** (30 messages🔥): 

- **记忆消失带来的烦恼**：许多用户分享了 ChatGPT 通用记忆功能的挫败体验，报告称记忆经常消失然后又重新出现。一位用户建议长期记忆应受益于透明度和备份选项，并表示：*"我深切希望记忆系统的原则/规则/协议是透明的。并且有一个备份按钮。"*

- **RAM 占用过高困扰用户**：用户报告在与 ChatGPT 进行长时间对话时 RAM 占用率很高，特别是在 Brave 浏览器上，内存使用量飙升至 32GB 并导致崩溃。一个建议是使用 Safari 或桌面应用程序，据报道它们能更好地处理大型聊天。

- **免费用户可访问 GPT Store**：一些用户庆祝可以免费访问 GPT Store，尽管有人指出 GPTs 在免费用户端仅在 3.5 版本上运行。

- **恼人的“词汇堆砌”输出**：一位用户抱怨 GPT-4 在长时间使用后倾向于生成“词汇堆砌”（word salad），最初连贯的回复会退化为一堆流行语和荒谬的短语。他们分享了一个例子，回复开始时逻辑尚存，随后演变成了胡言乱语。

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1245120840434188298)** (3 条消息): 

- **Prompt 分享频道受到关注**：一名成员询问：*"所有厉害的 Prompt 都分享在某个地方吗？"*。另一位用户引导他们前往 **#1019652163640762428** 频道查看分享的 Prompt。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1245120840434188298)** (3 条消息): 

- **特定频道提供 Prompt 资源**：一位用户询问是否所有“厉害的 Prompt”都在某处分享，另一位用户将其引导至频道 <#1019652163640762428>。这表明 Discord 社区内有一个专门分享高质量 Prompt 的地方。
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1245199368370524204)** (60 条消息🔥🔥): 

- **OpenAI 的 Alignment 讨论引发褒贬不一的反应**：一位用户链接了 Jan Leike 讨论 Alignment 的推文，其他人提到由于“末日论和诱饵式”推文盛行，他们已将其屏蔽。另一位用户表示，他们发现拉黑比屏蔽更有效。

- **Mistral 发布 Codestral，一个新的 22B 代码模型**：[Codestral](https://mistral.ai/news/codestral/) 是一个开源权重模型，精通 80 多种编程语言，专为代码生成任务设计。Codestral 已在 [HuggingFace](https://huggingface.co/mistralai/Codestral-22B-v0.1) 上线，并在 8 周的测试期内免费提供。

- **Scale AI 推出 LLM 排行榜**：Scale AI 发布了一个使用预留私有数据的全新 [LLM 排行榜](https://scale.com/leaderboard)。一位用户对潜在的偏见表示担忧，理由是该公司的激励机制以及在评估和付费客户数据中使用了相同的众包人员。

- **Google 的 Gemini 1.5 Flash 面临定价争议**：Google 因在模型发布后不久，在未通知的情况下将 Gemini 1.5 Flash 的输出价格几乎翻倍而受到批评。用户们争论这种价格调整是否反应过度，并指出最初该模型因其性价比而广受赞誉。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/codestral/">Codestral: Hello, World!</a>：通过 Mistral AI 赋能开发者并使编程民主化。</li><li><a href="https://fxtwitter.com/rosstaylor90/status/1795868124413038796?s=46">Ross Taylor (@rosstaylor90) 的推文</a>：信用评级机构在 2000 年代存在激励机制错位：他们评级产品的提供者正是给他们付钱的人。（我的第一份工作是在危机后监管 CDO，哈哈）同样地，一个公司……</li><li><a href="https://x.com/_xjdr/status/1795836185669169196">xjdr (@_xjdr) 的推文</a>：有了这个 22B 代码模型，应该有足够的数据点从 8x22B MoE 中提取出一个单一的 22B Dense 模型（不确定这会对许可证产生什么影响），它可能不需要任何额外的……</li><li><a href="https://x.com/natolambert/status/1795853487890153872">Nathan Lambert (@natolambert) 的推文</a>：@TheXeophon @Teknium1 别把我逼得太紧，宝贝</li><li><a href="https://x.com/mistralailabs/status/1795844741801894202?s=46">Mistral AI Labs (@MistralAILabs) 的推文</a>：在 https://console.mistral.ai/codestral 申请 Codestral 访问权限。在 8 周的测试期内免费！</li><li><a href="https://x.com/artificialguybr/status/1795851375181508785?s=46">𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr) 的推文</a>：Google 在未告知任何人的情况下将 Gemini 1.5 Flash 的输出价格提高了 98%。这距离宣布该模型仅一周。输出价格从 0.53/1M 变为 1.05/1M。我们如何信任一个大幅……</li><li><a href="https://x.com/natolambert/status/1795852202361172128">Nathan Lambert (@natolambert) 的推文</a>：我受够了这些用于营销的图表。这里有一个在 HumanEval 上将 y 轴从 0 缩放到 100 的版本，谢谢 ChatGPT :) 引用 Theophile Gervet (@theo_gervet) —— 我们刚刚发布了我们的第一个……</li><li><a href="https://x.com/sivil_taram/status/1795842555038535711?s=46">Qian Liu 🔭 (@sivil_taram) 的推文</a>：祝贺 Codestral 的新发布，欢迎这个强大的新编程模型加入开源社区！给图表加个小补丁：加上 CodeQwen1.5 🤔 免责声明：我不是……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245108911778959360)** (30 条消息🔥): 

- **Helen Toner 爆料 OpenAI 内部内幕**：前 OpenAI 董事会成员 Helen Toner 揭露了关于 Sam Altman 被解雇的惊人细节，理由是其经常性的不诚实和有毒的工作环境。播客讨论了在快速发展的 AI 领域中如何平衡创新与监管 ([播客链接](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3))。

- **董事会对 ChatGPT 的发布措手不及**：Helen 提到董事会是通过 Twitter 得知 ChatGPT 发布的消息，这反映了 OpenAI 管理层在沟通和透明度方面的缺失。这种缺乏提前通知的情况是董事会关注的关键问题。

- **对信息发布的复杂感受**：成员们讨论了为什么 Helen Toner 没有早点发布这些信息，一些人将其归因于法律限制。大家普遍认为内部政治和外部压力可能影响了董事会的沟通。

- **Sam Altman 的辩护**：董事会对这些指控的正式回应是，产品的安全或财务方面并没有真正的问题足以证明解雇 Sam 是合理的。他们强调了确保通用人工智能 (AGI) 造福全人类的使命，并强调了他们继续前进的承诺。

- **解雇理由受到质疑**：尽管 Helen 的指控引人注目，但成员们指出，董事会陈述的解雇理由——“沟通不始终坦诚”——显得很牵强。他们推测法律因素限制了董事会全面披露原因的能力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/btibor91/status/1795551083420430579">来自 Tibor Blaho (@btibor91) 的推文</a>: @TheXeophon https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3</li><li><a href="https://fxtwitter.com/bilawalsidhu/status/1795534345345618298">来自 Bilawal Sidhu (@bilawalsidhu) 的推文</a>: ❗独家：“我们是在 Twitter 上知道 ChatGPT 的。” OpenAI 到底发生了什么？前董事会成员 Helen Toner 打破沉默，透露了关于 Sam Altman 被解雇的惊人新细节……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1245153429375156365)** (4 条消息): 

- **FMTI 使用 CSV 而非 markdown**：一位用户对 FMTI GitHub 仓库将评分存储为 CSV 文件而非 markdown 表示沮丧。他们表示：“他们关闭了它，因为他们正在将论文每批次的评分作为 CSV 上传到一个新文件夹中。”

- **使用生成式模型定制个性化学习音乐**：有人建议使用生成式音频模型来创建个性化的学习音乐，专门为编程、阅读或写作定制。另一位用户幽默地补充说，这样的系统可能会转而优化播放列表以提高“补全率 (completions)”，这反映了对以生产力为导向的设计的担忧。
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1245252574911926342)** (10 条消息🔥): 

- **小型期刊俱乐部想法引起兴趣**：一位成员提议组建一个“小型期刊俱乐部 (mini journal club)”并讨论了可能的格式。另一位成员表示感兴趣，但指出需要一种结构化的格式，并表示“随意的播客格式并不那么有趣”。

- **Cohere 的教育系列脱颖而出**：关于教育资源的讨论展开，一些成员表达了对 **Cohere 教育视频系列** 的喜爱。一位成员建议，如果研究人员能“在 30-45 分钟内讲解论文，分享他们的关键见解/亮点”，那将会很有帮助。

- **TalkRL 播客被低估**：一位成员分享说 **TalkRL 播客** “被严重低估了”。另一位成员表示赞同，并强调 **ML Street Talk** 有时因为其哲学背景而“很快变得非常沉重且难以跟上”。

- **对 Schulman 那一集的评价褒贬不一**：围绕最近 **Dwarkesh 与 Schulman 的播客节目** 展开了对话。一些成员觉得内容枯燥，并注意到主持人和嘉宾之间缺乏默契，影响了整体讨论质量。
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1245155522240319542)** (3 条消息): 

- **对 DMC-GB2 GIF 的热情**：一位成员分享了他们对 [DMControl Generalization Benchmark 2 (DMC-GB2) 仓库](https://github.com/aalmuzairee/dmcgb2?tab=readme-ov-file)中 GIF 的兴奋之情。他们称赞了其视觉吸引力，表示 *“这个仓库里的 GIF 真的太棒了。”*
- **对 Reinforcement Learning 的喜爱**：一位成员表达了怀旧之情，评论道 *“我想念 RL。”* 另一位成员安慰他们说：“RL 正张开双臂等着你。”

**提到的链接**：<a href="https://github.com/aalmuzairee/dmcgb2?tab=readme-ov-file">GitHub - aalmuzairee/dmcgb2: Official release of the DMControl Generalization Benchmark 2 (DMC-GB2)</a>: DMControl Generalization Benchmark 2 (DMC-GB2) 的官方发布 - GitHub - aalmuzairee/dmcgb2: Official release of the DMControl Generalization Benchmark 2 (DMC-GB2)

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1245270632766115912)** (7 条消息): 

- **贴纸讨论正在进行中**：在关于贴纸的轻松对话中，Nathan Lambert 提到：*“需要想出一些好的贴纸……还没想出来。”* 他随后提到他们正在 *“制作贴纸。不过不是 Nathan Lambert 形象的，哈哈。”* 

- **SnailBot 新闻更新即将到来**：SnailBot News 提及了一个角色 *“<@&1216534966205284433>”*。此摘要中未提供关于 SnailBot 的更多细节。
  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1245215802412765215)** (5 条消息): 

- **Tom 提到的卢梭（Rousseau）引起共鸣**：一位成员非常喜欢最新一集，并赞赏 **Tom** 的背景，特别提到了引人深思的 *Rousseau 引用*。他们将 **Discourse on Inequality** 标记为一个值得注意的讨论点。
  
- **Hierarchy-Informed 模型规范**：一位用户链接到了 [Andrew Carr 的推文](https://x.com/andrew_n_carr/status/1782878279504191896)，讨论 OpenAI 的 Alignment 研究，该研究引入了 "instruction hierarchy" 来缓解 jailbreaking attacks。模块化的 Prompt 结构和 hierarchical privileges 被认为是关键要素。

- **Transformative Exceptions 引起关注**：讨论了关于 Transformative Exceptions 政策中的灰色地带。提到由于运行 Classifiers 的成本很高，预计发布具有超长 Context Windows 的新模型可能会影响这些政策。



**提到的链接**：<a href="https://x.com/andrew_n_carr/status/1782878279504191896">Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：来自 OpenAI 的酷炫新 Alignment 研究。他们生成合成数据来鼓励 "instruction hierarchy"，模型会将 System Prompts 视为更重要的内容。这随后预...

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1245092319238295683)** (117 条消息🔥🔥): 

- **建议使用 Colab 和 Kaggle 以实现更快的图像生成**：用户讨论了他们在各种在线托管服务上的经验，推荐使用 **Kaggle** 或 **Colab** 以获得更好、更快的图像生成体验。一位用户指出，“在拥有 16GB VRAM 的 Colab 上，生成 1 张图像需要 1:30 分钟或 2 分钟”。

- **训练 Stable Diffusion XL LoRA 模型**：成员们交流了训练 **SDXL LoRA 模型**的技巧，讨论了最佳步数、Epochs 以及训练图像数量的重要性。训练建议 *“推荐 2-3 个 Epochs”* 且 *“短触发词效果更好”*。

- **Auto1111 和 ComfyUI 模型路径问题**：成员们寻求关于配置 **ComfyUI extra model paths** 以从多个目录加载模型的建议。此外，还提出了关于在本地 **Stable Diffusion API** 中集成 ADetailer 的咨询。

- **HUG 与 Stability AI 课程**：讨论了 **HUG 与 Stability AI 合作**的创意 AI 课程，课程环节将被录制并在直播后可供观看。完成课程并填写反馈表后方可退还押金。

- **使用 Stable Diffusion 生成 3D 模型**：用户讨论了 **AI 生成 3D 模型**的潜力及其在 3D 打印中的适用性。一位成员表示：“不，目前还完全不行”，反映了当前的局限性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/yas-hyped-lit-feeling-it-party-gif-14473619">Yas Hyped GIF - Yas Hyped Lit - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.studios.thehug.xyz/lab">HUG x Stability AI 创新实验室 — HUG</a>: 与 Stability AI 一起发现你独特的创新，并接受来自 HUG 的实时战略、营销和创意教育。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1245435755078221905)** (1 条消息): 

- **LlamaIndex 推出 PropertyGraphIndex**：宣布与 Neo4j 合作推出一项利用 LLM 构建知识图谱的新功能。[推文](https://x.com/llama_index/status/1795869279457546447)和[博客文章](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms)提供了更多细节。
- **用于知识图谱构建的复杂工具**：该功能包括使用各种检索器（如关键词、向量搜索和 text-to-cypher）提取和查询知识图谱的工具。用户现在可以执行联合向量搜索和图搜索，无论图存储是否与向量兼容。
- **强调定制化和灵活性**：它允许定义自定义提取器和检索器，使得处理标记属性图变得直观。每个节点/关系都可以拥有标签和属性，从而实现强大的知识图谱结构。
- **提供详细指南和示例**：[文档](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/)中提供了全面的指导和示例 Notebook，深入记录了基础和高级用例。与 Neo4j 的集成也包含在[使用指南](https://docs.llamaindex.ai/en/stable/examples/property_graph/graph_store/)中。
- **与 Neo4j 的合作受到赞誉**：包括 [@tb_tomaz](https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_neo4j/) 在内的 Neo4j 专家做出了重大贡献，创建了集成指南并重构了抽象层以实现无缝功能。

**提及的链接**: <a href="https://x.com/llama_index/status/1795869279457546447">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>: 我们很高兴推出一项重大功能，使 @llama_index 成为使用 LLM 构建知识图谱的框架：The Property Graph Index 💫（这里有很多内容需要解析，让我们从...开始）

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245093324428283965)** (5 条消息): 

```html
- **FinTextQA 数据集聚焦金融领域**：FinTextQA 数据集提供 **1,262 个高质量、带来源属性的问答对**，涵盖六种不同的问题类型。它为基于文档的金融问答提供了强大的上下文支持 [来源](https://t.co/emhQYXY1S4)。
- **PostgresML 与 LlamaIndex 集成**：如果你对 Postgres 和 AI 应用感兴趣，请关注 [PostgresML](https://t.co/G7WTrSdt0B)。它支持在 Python 和 JavaScript 中进行 **本地 embedding、模型训练和 fine-tuning**。
- **LlamaIndex 发布 Property Graph Index**：Property Graph Index 提供了使用 LLMs (**Large Language Models**) 构建和查询知识图谱的新工具。这一新功能旨在将 LlamaIndex 定位为构建知识图谱的全方位框架 [来源](https://t.co/X9D3Wl0Hyv)。
- **Codestral 代码生成模型现已可用**：来自 MistralAI 的新模型 **Codestral** 支持超过 **80 种编程语言**，并可本地运行。LlamaIndex 提供 **Day 0 支持** 以及详细的 [notebook](https://t.co/k2nHDiMnwD) 演示其用法。
- **Ollama 增强对 Codestral 的支持**：作为额外福利，Codestral 模型已获得 [Ollama](https://t.co/gsPHHF4c0K) 的全面支持，使用户能够以一流的支持体验在本地运行该模型。
```
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1245112152495227000)** (107 条消息🔥🔥): 

- **RAG 模型中的语义分块 (Semantic Chunking) 辩论**：成员们讨论了 RAG (Retrieval Augmented Generation) 模型中大语义文本块与小语义文本块之间的权衡。他们考虑为同一文本 embedding 多个版本以获得更好的检索效果，并指出了分块策略中诸如指代消解 (co-reference resolution) 等挑战。

- **LlamaIndex 增强与支持**：成员们分享了使用 LlamaIndex 进行各种用途的经验和疑问，例如 ArangoDB 支持和自定义 tokenizer 设置。有人提到了用于生成高质量 RAG 文本块的 Semantic Document Parser 的 GitHub 仓库。

- **Embedding 与检索模型**：讨论了如何设置和使用不同的 embedding 模型，特别是针对非英语文本。成员们推荐了来自 HuggingFace 的模型用于特定语言任务，例如阿拉伯语数据 embedding。

- **合并与管理向量存储 (Vector Stores)**：一位用户寻求合并 Qdrant 向量存储索引的帮助，有人建议了 LlamaIndex 文档中涉及 `QueryFusionRetriever` 的解决方案。另一个疑问涉及使用 GPT-4o 进行多模态输入的聊天记忆缓冲区 (chat memory buffer)。

- **在 LlamaIndex 中保存和提取节点 (Nodes)**：成员们询问了关于 LlamaIndex 中节点管理的问题，包括使用 `docstore.persist()` 保存节点以及使用 `get_all_documents()` 方法提取节点。他们讨论了使用不同的文档存储后端，如 RedisDocumentStore 和 MongoDocumentStore。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/asafaya/bert-base-arabic">asafaya/bert-base-arabic · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/isaackogan/SemanticDocumentParser">GitHub - isaackogan/SemanticDocumentParser: 用于为 RAG 生成高质量文本块的高级解析器。</a>：用于为 RAG 生成高质量文本块的高级解析器。 - isaackogan/SemanticDocumentParser</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/storing/docstores/#document-stores>).">Document Stores - LlamaIndex</a>：未找到描述</li><li><a href="http://127.0.0.1:8529">">未找到标题</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/readers/arango_db/#llama_index.readers.arango_db.SimpleArangoDBReader>).">Arango db - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/use_cases/multimodal#multi-modal>)">Multi-Modal Applications - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion/">Simple Fusion Retriever - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1245095161990746242)** (72 条消息🔥🔥):

```html
- **Gemini 1.5 的性能令人印象深刻**：在 Gemini 1.5 的结果发布后，注意到 **Gemini 1.5 Pro/Advanced** 排名第二，紧随 GPT-4o 之后，而 **Gemini 1.5 Flash** 排名第九，表现优于 Llama-3-70b 等模型。详细的分类细目可以在 [LMSysOrg's Twitter](https://x.com/lmsysorg/status/1795512202465845686) 上找到。

- **构建 LLM 的见解**：文章 "[What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)" 讨论了 LLM 的快速进步以及在 Demo 之外构建有效 AI 产品的挑战。

- **对 SWE-agent 潜力的兴奋**：在普林斯顿大学的研究人员发布 **SWE-agent** 后，关于其卓越性能和开源特性的说法引起了广泛关注。更多细节分享在 [Gergely Orosz's Twitter](https://x.com/GergelyOrosz/status/1794743519954731331) 和 [SWE-agent GitHub](https://github.com/princeton-nlp/SWE-agent) 上。

- **新的开源 VLM 模型 - Llama3-V**：**Llama3-V** 模型声称性能优于 **LLaVA**，并能与 GPT4-V 等模型展开竞争，强调其在模型尺寸显著减小的情况下仍具有极高的效率。详细信息和访问链接已在 [Sidd Rsh's Twitter](https://x.com/siddrrsh/status/1795541002620727439) 上提供。

- **Scale 发布用于 LLM 评估的 SEAL 排行榜**：**Scale's SEAL Leaderboards** 旨在提供私密的专家评估，以确保稳健且不可被利用的模型评估。该倡议得到了 [Alexandr Wang](https://x.com/alexandr_wang/status/1795857651592491281) 的强调，并获得了 [Andrej Karpathy](https://x.com/karpathy/status/1795873666481402010) 的赞赏。
```

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">我们从一年的 LLM 构建中学到了什么（第一部分）</a>：未找到描述</li><li><a href="https://changelog.com/news/96">为什么你不应该使用 AI 来编写测试 (Changelog News #96)</a>：Swizec 关于不使用 AI 编写测试的文章；LlamaFs 是一个基于 Llama 3 的自组织文件系统；Pew Research 的一项分析证实互联网充满了失效链接；Sam Rose 构建了一个...</li><li><a href="https://aider.chat/2024/05/22/swe-bench-lite.html">Aider 如何在 SWE Bench Lite 上取得 26.3% 的 SOTA 成绩</a>：Aider 主要通过其现有的功能实现了这一结果，这些功能专注于静态代码分析、可靠的 LLM 代码编辑以及针对 AI 结对编程的实用 UX。</li><li><a href="https://x.com/arthurmensch/status/1795820396198924667?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Arthur Mensch (@arthurmensch) 的推文</a>：随着 Codestral（我们最新的 SOTA 代码模型）的发布，我们推出了 Mistral AI 非生产许可证 (MNPL)。它允许开发者将我们的技术用于非商业用途和研究...</li><li><a href="https://x.com/GergelyOrosz/status/1794743519954731331">来自 Gergely Orosz (@GergelyOrosz) 的推文</a>：如果构建一个比最强 LLM 表现好约 4 倍的 AI 编程 Agent 具有十亿美元的潜力：这里有 7 位普林斯顿大学的研究人员做到了这一点。它是完全开源的，名为 SWE-agent...</li><li><a href="https://x.com/siddrrsh/status/1795541002620727439?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Siddharth Sharma (@siddrrsh) 的推文</a>：介绍 Llama3-V，一个 SOTA 开源 VLM 模型。我们的特点：• 性能超越 LLaVA • 性能可与 GPT4-V, Gemini Ultra, Claude Opus 媲美，且模型缩小了 100 倍 • 针对 L... 的 SOTA 开源 VLM</li><li><a href="https://x.com/alexandr_wang/status/1795857651592491281?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：1/ 我们正在发布 SEAL 排行榜——对领先前沿模型的私密、专家级评估。我们的设计原则：🔒私密 + 不可利用。评估中没有过拟合！🎓领域专家评估 🏆持续...</li><li><a href="https://x.com/lmsysorg/status/1795512202465845686">来自 lmsys.org (@lmsysorg) 的推文</a>：重大新闻——Gemini 1.5 Flash, Pro 和 Advanced 的结果出炉了！🔥 - Gemini 1.5 Pro/Advanced 排名第 2，逼近 GPT-4o - Gemini 1.5 Flash 排名第 9，超越了 Llama-3-70b，几乎达到了 GPT-4-01...</li><li><a href="https://x.com/openai/status/1795900306490044479?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 OpenAI (@OpenAI) 的推文</a>：所有 ChatGPT 免费用户现在都可以使用浏览、视觉、数据分析、文件上传和 GPTs。引用 OpenAI (@OpenAI)：我们正在开放对新旗舰模型 GPT-4o 的访问，以及浏览等功能...</li><li><a href="https://x.com/alexandr_wang/status/1795857651592491281?s=46&t=90xQ8sGy">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：1/ 我们正在发布 SEAL 排行榜——对领先前沿模型的私密、专家级评估。我们的设计原则：🔒私密 + 不可利用。评估中没有过拟合！🎓领域专家评估 🏆持续...</li><li><a href="https://x.com/khoomeik/status/1795477359933706272">来自 Rohan Pandey (e/acc) (@khoomeik) 的推文</a>：📢 很高兴终于发布了我的 NeurIPS 2024 投稿！Chinchilla 是通用的吗？不！我们发现：1. 语言模型缩放定律取决于数据复杂度 2. gzip 有效地预测了缩放...</li><li><a href="https://x.com/MistralAILabs/status/1795820935540584909">来自 Mistral AI Labs (@MistralAILabs) 的推文</a>：宣布 Codestral：我们的首个代码模型。- 在新的 Mistral AI 非生产许可证下开放权重 - 通过 La Plateforme 提供新端点：http://codestral.mistral.ai - 现在就在 Le Chat 上尝试：h...</li><li><a href="https://docs.docarray.org/">DocArray</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1795873666481402010?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Andrej Karpathy (@karpathy) 的推文</a>：很好，一个在评估 LLM 方面能与 @lmsysorg 竞争的强力对手加入了。LLM 评估正在改进，但不久前它们的状态还非常糟糕，定性体验经常与...不符。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1245391183140753530)** (1 条消息): 

- **太平洋时间中午 12 点的 AI Agent 架构和 KANs 活动**：Latent Space 今天中午 12 点（太平洋时间）将举办一场关于 AI Agent 架构和 Kolmogorov Arnold Networks (KANs) 的活动。[活动注册和详情](https://lu.ma/pxnaq641)已发布，建议参与者通过活动页面上的 RSS 图标将活动添加到日历中。

**提到的链接**：<a href="https://lu.ma/pxnaq641">LLM Paper Club (AI Agent Architectures + Kolmogorov Arnold Networks) · Zoom · Luma</a>：买一送一！Eric Ness 将讲解 https://arxiv.org/abs/2404.11584 (The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A…

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1245452118362423409)** (2 条消息): 

```
There are no messages to summarize for the channel llm-paper-club-west.
```

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1245108640562548756)** (2 条消息): 

- **OpenAI 面临临时停机**：“OpenAI 的使用对许多用户来说暂时中断”，但 Azure 和 Azure 备用方案仍可运行。该问题已通过更新迅速解决：*“编辑：已恢复。”*
- **Cinematika 模型将被弃用**：由于使用率极低，**Cinematika 模型**将被停止使用。建议用户立即迁移到新模型：*“请切换到新模型！”*
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1245091293558739046)** (51 条消息🔥): 

- **OpenAI 模型达到支出限额**：成员们讨论了由于意外达到支出限额而导致 OpenAI 模型无法访问的问题。Alex Atallah 承诺发布公告并进行修复，提到正常的 OpenAI 使用已恢复，并正在实施额外的检查。

- **Gemini 模型提示词请求**：一位成员询问关于 Gemini 模型的提示词指南，但未收到回复。这一请求表明了用户持续的兴趣以及用户支持或文档方面的潜在需求。

- **媒体附件政策**：Cupidbot.ai 询问了关于发送媒体内容的限制。Alex Atallah 解释说，媒体内容被限制在特定频道以控制垃圾信息，并承诺允许高级角色发布附件，Louisgv 同意了这一更改。

- **GPT-4o 上下文和 Token 限制**：有人担心 GPT-4o 的上下文限制被减少到 4096 个 Token。Alex Atallah 澄清说，上下文限制为 128k，最大输出 Token 为 4096。 

- **GPT-4o 图像处理缓慢**：一位用户报告在使用 `openai/gpt-4o` 配合 `image-url` 输入时，图像处理速度缓慢，每个提示词需要几分钟。这指出了需要关注的潜在性能问题。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752">Oh No Homer GIF - Oh No Homer Simpsons - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://fast.snova.ai/">Streamlit</a>: 未找到描述</li><li><a href="https://lluminous.chat">lluminous</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1245142392919691324)** (23 messages🔥): 

- **Helen Toner 谈 ChatGPT**：用户分享了一个 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1d2s4ca/helen_toner_we_learned_about_chatgpt_on_twitter/)的链接，其中 Helen Toner 提到他们是“在 Twitter 上了解到 ChatGPT 的”。
  
- **LeCun 的论文发表状态**：关于知名 AI 人物 Yann LeCun 在担任 Facebook 副总裁后是否停止发表论文的讨论。一些人认为 LeCun 仍在积极贡献。

- **Elon Musk 的 AI 模型定位**：成员们辩论了 Elon Musk 对开源模型的立场，指出 Musk 仅在模型不再具有竞争力时才将其发布。讨论中分享了 [Hugging Face 上 xai-org 的链接](https://huggingface.co/xai-org)。

- **Mistral AI 模型许可**：Mistral AI 模型因其尽管处于非商业许可下但采用“open weights”的商业模式而受到关注。分享了[相关链接](https://mistral.ai/news/codestral/)和[其他更新](https://mistral.ai/news/mistral-ai-non-production-license-mnpl/)以提供更多细节。

- **在 Twitter 上屏蔽 Elon Musk**：一名用户提到由于 Elon Musk 的争议性言论和行为而将其屏蔽。这引发了其他人谈论他们对 Musk 收购 Twitter 的反应，其中一人注销了账号。

**提到的链接**：<a href="https://www.reddit.com/r/singularity/comments/1d2s4ca/helen_toner_we_learned_about_chatgpt_on_twitter/">Reddit - Dive into anything</a>：未找到描述

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245203415169110106)** (17 messages🔥): 

- **合成字幕在 Compel 过程中的问题**：对话强调，在 Compel 过程中使用 *“a woman reading a book”* 作为 prompt 会导致问题，即使使用了强大的合成字幕。一位用户提到，*“糟糕的事情开始发生”*，表明在生成准确输出方面存在挑战。

- **研究论文中的 Dinov2 和 UNet 配置**：针对研究论文 [arxiv.org/abs/2405.18407](https://arxiv.org/abs/2405.18407) 进行了深入交流，指出其使用 **Dinov2 作为 discriminator**。然而，研究发现 *“带有顶层网络的预训练 UNet 效果更好”*，这与 Kandinsky 的方法类似，即 *“将 UNet 减半并将其作为 discriminator 进行训练”*。

- **Horde 社区的激励评分系统**：一位用户询问了 Horde AI 社区用于对 **SD 图像**进行评分的工具，该系统为贡献提供 **kudos**，可用于生成更多图像。然而，另一位用户对该系统表现出冷淡，并担心 *“评分激励几乎总是会导致更差的数据”*。

**提到的链接**：<a href="https://arxiv.org/abs/2405.18407">Phased Consistency Model</a>：Consistency Model (CM) 最近在加速 Diffusion Models 生成方面取得了显著进展。然而，其在文本条件下的高分辨率图像生成应用中……

  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1245117372482846720)** (26 messages🔥): 

- **Langchain v2.0 agents confusion**：一位用户表示在 LangChain v2.0 中定位 Agent 存在困难，但随后确认已找到。
- **Innovative AI discussions**：一位成员分享了一条关于“允许机器重新定义创意并在重复性任务之外进行创新”的推文，引发了对 AI 创意的思考 ([Tweet](https://x.com/Dorsa_Rohani/status/1795452411143733361))。
- **Handling RateLimit errors in LangChain**：对于在 LangChain 中处理 "RateLimit" 错误，建议使用 Python 中标准的 try/except 机制，并提供了一个示例来指导错误处理。
- **ConversationalRetrievalChain issue**：一位成员报告了在使用带有多个 Vectorstore 的 ConversationalRetrievalChain 时内容检索不完整的问题，并寻求数据合并问题的解决方案。
- **CSV dataset to Vectorstore for retrieval**：分享了关于如何将 CSV 数据集处理到 Vectorstore 以供检索的详细说明，包括加载 CSV 文件以及使用 `langchain` 库创建 Vectorstore ([更多信息](https://python.langchain.com/v0.1/docs/integrations/vectorstores/infinispanvs/#prepare-the-data))。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/use_cases/tool_use/tool_error_handling/#retry-with-exception>).">Handling tool errors | 🦜️🔗 LangChain</a>: 使用模型调用工具时存在一些明显的潜在失败模式。首先，模型需要返回一个可以被解析的输出。其次，模型需要返回符合要求的工具参数...</li><li><a href="https://x.com/Dorsa_Rohani/status/1795452411143733361">Dorsa Rohani (@Dorsa_Rohani) 的推文</a>: 我们如何允许机器表达自己？目前，AI 在复制、重复。我想构建能够创新并创造新事物的 AI。但我们如何让 AI 测试极限并重新定义创意...</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/vectorstores/lantern/#using-a-vectorstore-as-a-retriever>))">Lantern | 🦜️🔗 LangChain</a>: Lantern 是一个用于 Postgres 的开源向量相似度搜索工具。</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/vectorstores/infinispanvs/#prepare-the-data>))">Infinispan | 🦜️🔗 LangChain</a>: Infinispan 是一个开源的键值数据网格，它可以以单节点或分布式方式工作。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17729>))">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1245256696839016488)** (1 messages): 

- **Langserve Example with Chat History**：一位成员正在使用 [GitHub](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/client.ipynb) 提供的示例测试 **langserve** 的聊天历史记录功能。他们正在寻求关于如何根据 FastAPI 文档中提供的细节“在请求体中包含我的 chat_history”的帮助。

**提到的链接**: <a href="https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/client.ipynb">langserve/examples/chat_with_persistence_and_user/client.ipynb at main · langchain-ai/langserve</a>: LangServe 🦜️🏓。通过在 GitHub 上创建账号为 langchain-ai/langserve 的开发做出贡献。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1245110002058788906)** (1 messages): 

- **Routing Logic in Agent Flows with Visual Agents**：分享了一段名为“如何在你的 Agent 流中路由逻辑”的 YouTube 视频。该视频提供了一个在基于 **LangChain** 构建的 Visual Agents 中使用路由逻辑的简单示例。你可以点击[这里](https://youtu.be/KtbRexZ6vsc)查看。

**提到的链接**: <a href="https://youtu.be/KtbRexZ6vsc">How to Route Logic in Your Agent Flows</a>: 在基于 LangChain 构建的 Visual Agents 中如何使用路由逻辑的简单示例。https://visualagents.ai https://langchain.ai

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1245123213344641054)** (18 messages🔥): 

- **个性化训练工作流**：一名成员表达了进行自主训练的愿望，强调“我们每个人都有自己的工作流和使用场景”。
- **Open Interpreter 使用案例**：另一名成员向社区询问了 Open Interpreter 的使用案例，引发了关于各种应用的讨论。
- **开源 Rewind 替代方案**：成员们讨论了 Rewind 的替代品，其中一人提到了 **Rem**，另一人分享了使用 Rewind 免费版结合 **Cohere API** 查询向量数据库（vector DB）的经验。
- **Phidata 与 Rewind 的连接性受到称赞**：一名成员分享了使用 Rewind 的积极体验，指出尽管它不会隐藏密码或凭据，但他们发现其“生活黑客（life hack）”功能非常宝贵。
- **无需确认运行 OI**：一名成员询问如何在无需确认的情况下运行 Open Interpreter，讨论了使用 pyautogui 等潜在解决方案，并最终通过 `--auto_run` 功能找到了解决方法，正如[文档中](https://docs.openinterpreter.com/settings/all-settings#auto-run)所指出的。

**提到的链接**：<a href="https://docs.openinterpreter.com/settings/all-settings#auto-run">All Settings - Open Interpreter</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1245121857858834613)** (3 messages): 

- **使用 Arduino 烧录 M5 遇到障碍**：一名用户成功使用 Arduino 烧录了 M5 并成功打开了 captive portal。然而，在服务器设置后，设备在访问时显示白屏，即使重新烧录后也没有连接 Wi-Fi 网络或服务器的选项。

- **解决 M5 白屏问题的建议**：另一名用户建议在烧录时将 Arduino studio 设置为擦除内存，作为该问题的潜在修复方法。
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

mikebirdtech: https://www.youtube.com/watch?v=sqwtk18pw14
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1245346875343769700)** (4 messages): 

- **NSFW Discord 邀请垃圾信息警报**：一名成员向版主发出警报，称多个频道中出现了 **NSFW Discord 邀请链接** 的垃圾信息。他们提到不确定版主提醒（ping）是否有效。
- **版主对 NSFW 垃圾信息的响应**：一名版主确认并处理了垃圾信息问题，感谢该成员的举报。
- **关于微调 LLM 以进行多媒体理解的咨询**：一名成员寻求关于*微调大语言模型 (LLM) 以进行图像和视频理解*的指导，特别提到了 **LLava models** 等模型。消息记录中未提供回复。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1245248003896381481)** (9 messages🔥): 

- **更新 Unsloth 中 Gradient Checkpointing 的提议**：一名成员建议更新 Unsloth 中的 gradient checkpointing 代码以支持 MoE，并分享了提议的代码更新。他们在验证后获得了提交 PR 的许可。

- **未训练 Token 修复的考虑**：讨论了未训练 token 的修复问题，一名成员确认不存在双重 bos_token 问题，但建议考虑未训练 token 的修复。

- **高效 Bin Packing 更新**：另一名成员提到更新后的 bin packing 效率更高，并询问了关于分布式训练（distributed training）的问题。一名用户遇到了训练在第一次评估后卡住的问题，可能是由于新的 sampler 未实现 `_len_est` 导致的。

- **征集具备特定技能的后端开发人员**：一名成员请求熟悉后端开发和 Google protobuf 的人员，寻求类似于 reverse engineer、malware analyst 或 bug bounty hunter 的专业知识。他们愿意为协助支付报酬。

- **还原 Multipack Batch Sampler 的更改**：一名成员分享了还原 multipack batch sampler 更改的 PR，指出之前实现中的 loss 计算偏差了一个数量级。[PR #1672 - Revert multipack batch sampler changes](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1672)。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1672">revert multipack batch sampler changes by winglian · Pull Request #1672 · OpenAccess-AI-Collective/axolotl</a>：使用 #1619 时 loss 不太正确，偏差了一个数量级。

  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1245216188804370444)** (6 messages): 

- **对于 PDF，考虑使用 RAG 而非 JSONL 微调**：一位成员建议对 PDF 使用 **RAG (Retrieval Augmented Generation) 方法**，以避免微调的需要。*"你可能需要考虑使用 RAG 方法，这样就无需针对 PDF 进行微调了。"*

- **如何在 API 中访问 response.citations**：据报道，response.citations 功能仅能通过 API 访问。提供了一个示例来展示这种 [Grounded Generation 方法](https://huggingface.co/CohereForAI/c4ai-command-r-plus#grounded-generation-and-rag-capabilities)。

- **本地 R+ 实现包含强制引用**：一位成员分享了他们在 Command R+ 的本地实现中构建 RAG 流水线的成功经验，该流水线确保包含引用。*"在我由本地 R+ 驱动的应用中，我构建了一个 RAG 流水线，并强制显示通过本地运行的 Embedding 模型获得的引用。"*

- **使用 Cohere 的 Discord 机器人受到称赞，但需要合适的频道**：一位成员赞赏了另一位成员的 Discord 机器人及其对 Cohere 的使用，但建议将讨论移至相应的项目频道。*"我非常喜欢你的 Discord 机器人以及它对 Cohere 的使用！只是我们有一个专门的项目频道！"*

**链接提及**：<a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus#grounded-generation-and-rag-capabilities">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：未找到描述

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1245154920903213137)** (4 messages): 

- **Elon Musk 的 xAI 获得巨额资金助力**：xAI 宣布筹集了 [60 亿美元融资](https://x.ai/blog/series-b)，用于“将初创公司的首批产品推向市场，建设先进基础设施，并加速未来技术的研发”。支持者包括 Andreessen Horowitz、Sequoia Capital 和沙特王子 Al Waleed bin Talal 等。 
- **对分析工具的质疑**：一位成员表示频道中讨论的工具“用处微乎其微”，但未指明具体是哪些工具。
- **Fireship 视频中的 Bend 语言令人印象深刻**：另一位成员称赞了 Fireship 视频中介绍的 Bend 语言，强调其“无需任何代码即可自动实现多线程”的能力，这与 tinygrad 的 Lazy Execution 非常契合。
- **关于 tinybox 电源的查询**：有人提问 tinybox 使用的是“两个消费级电源，还是两个带有配电板的服务器级电源”。

**链接提及**：<a href="https://www.theverge.com/2024/5/27/24165619/elon-musk-xai-startup-6-billion-funding">Elon Musk’s xAI raises $6 billion to fund its race against ChatGPT and all the rest</a>：这些资金中有多少将用于购买 GPU？

  

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1245166067798507661)** (4 messages): 

- **Goliath 在持续预训练前观察到性能下降**：一名成员询问 **Goliath** 在持续预训练之前是否存在大幅性能下降。这引发了关注，并吸引了其他用户的标记回复。

- **llm.c 中复现 GPT-2 的记录**：[GitHub](https://github.com/karpathy/llm.c/discussions/481) 上的一项讨论详细介绍了如何在 llm.c 中以 20 美元的成本复现 GPT-2 (124M)，并实现了 29.9 的 HellaSwag 准确率，超过了 GPT-2 的 29.4。该结果是与训练时间显著更长的 GPT-3 模型进行对比得出的。

- **Mistral AI 发布其首个代码模型 Codestral-22B**：[Guillaume Lample](https://x.com/guillaumelample/status/1795820710750744839?s=46&t=1jtkL4JPu-DUOdo8JC668g) 宣布发布 **Codestral-22B**，该模型在超过 80 种编程语言上进行了训练。它的性能优于之前的模型，并已在其 API 平台、VScode 插件和 **Le Chat** 上可用。

- **LAION AI 寻求社区帮助构建开源 GPT-4-Omni**：[LAION AI](https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19) 分享了一篇博客文章，请求协助构建开源 GPT-4-Omni。他们在[此处](https://laion.ai/notes/open-gpt-4-o/)的文章中提供了极具前景的方向、数据集和教程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/laion_ai/status/1795910332008804428?t=rBHUXm87TFrQ-kyfeZP0fg&s=19">来自 LAION (@laion_ai) 的推文</a>：帮助我们构建开源 GPT-4-Omni！通过这篇博客文章，我们展示了极具前景的方向（包括数据集和教程） https://laion.ai/notes/open-gpt-4-o/</li><li><a href="https://x.com/guillaumelample/status/1795820710750744839?s=46&t=1jtkL4JPu-DUOdo8JC668g">来自 Guillaume Lample @ ICLR 2024 (@GuillaumeLample) 的推文</a>：今天我们发布了 Codestral-22B，我们的首个代码模型！Codestral 在超过 80 种编程语言上进行了训练，性能超越了以往的代码模型，包括体量最大的...</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">在 90 分钟内以 20 美元在 llm.c 中复现 GPT-2 (124M) · karpathy/llm.c · Discussion #481</a>：让我们在 90 分钟内以 20 美元的成本，在 llm.c（约 4,000 行 C/CUDA 代码）中复现 GPT-2 (124M)。124M 模型是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，实际上相当...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1245354820013920329)** (3 messages): 

- **Windows 上 llamafile 的编译错误**：一位用户分享了在 Windows 上编译 **llamafile** 时遇到的困难，遇到了一个与 `cosmoc++` 相关的错误。具体而言，由于可执行文件在缺少 `.exe` 扩展名时的启动方式，构建工具链失败。
- **文件存在问题**：该用户指出，尽管错误消息显示文件缺失，但该文件确实存在于 `.cosmocc/3.3.8/bin` 中。使用 cosmo bash 进行编译的尝试也同样受阻。
  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1245280918327787520)** (2 messages): 

```html
- **Retrieval Augmented Generation can solve hallucination**: A member mentioned frequently using **LLMs** to answer documentation-related questions but facing issues with hallucinations and inaccuracies. They suggested that *pulling the docs, storing embeddings, and using similarity search ("Retrieval Augmented Generation")* could mitigate this and inquired about extending `llm` to create embeddings for a URL recursively.
```
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

yellowturmeric: 我还没看过。感谢分享。我会读一下这篇论文。
  

---



---



---



---




{% else %}




_

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **网页抓取心得**：讨论强调了高效提取网页内容的方法，包括 **Python requests**、**Playwright**，以及针对重 JavaScript 网站的 **Gemini 1.5 Flash**。

- **Perplexity API 的困扰与收获**：工程师们对 **Perplexity API** 响应与 Web 应用准确性之间的一致性表示担忧，并考虑选择不同的模型（如 **llama-3-sonar-small-32k-online**）来潜在地提升性能。

- **构建 Perplexity 的竞争对手**：提出了一个模仿 **Perplexity 多模型查询** 的详细项目，但在扩展和后端开发方面面临挑战。

- **深入 Go 语言**：对 **Go 编程语言** 的深入探讨展示了其有效性，特别是在网页抓取应用中，强调了其可扩展性和并发优势。

- **优势分析**：用户分享了 **Perplexity AI 搜索链接**，内容涵盖潜在的 AI 生成内容、查询合理性的澄清以及全面的优缺点评估。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BERT 的 Token 限制促使用户寻求解决方案**：用户正在评估处理超出 **BERT (512 tokens)** 和 **基于解码器的模型 (1,024 tokens)** 限制的文档的方法。他们的目标是绕过文档切片和位置嵌入（positional embedding）调整，且不求助于昂贵的新预训练。

- **Diffusers 庆祝 GPT-2 情感分析取得成功**：Hugging Face 社区庆祝 Diffusers 项目成立两周年，同时发布了一个用于情感分析的新 **FineTuned GPT-2** 模型，其准确率和 F1 分数达到了 **0.9680**。该模型针对 Amazon 评论进行了优化，可在 [Hugging Face](https://huggingface.co/ashok2216/gpt2-amazon-sentiment-classifier-V1.0) 上获取。

- **读书会期待 C4AI 的见解**：新的论文研读小组已准备就绪，渴望加入来自 C4AI 社区的演讲，重点是揭穿低资源语言中的虚假信息。下一次活动见[此处](https://discord.com/events/879548962464493619/1245408612818620426)。

- **图像处理咨询引导用户获取资源**：讨论涵盖了使用 **YOLO** 等模型以及 **convNext** 和 **DINOv2** 等较新替代方案处理大图像的最佳实践。一个关于 Hugging Face 图像处理教程的 GitHub 仓库被重点提及 ([Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma))。

- **医学影像寻求 AI 辅助**：社区成员就创建一个用于分析无标签 MRI 和 CT 扫描的自监督学习框架交换了意见。讨论包括利用预训练模型提取的特征进行特定类别的分割任务。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Lightning AI 与 L4 的火花**：用户推荐使用 **Lightning AI Studio**，因为它提供“每月约 20 小时的免费时长”，且 L4 的性能优于 Colab 的 T4 GPU。提议与 Lightning AI 进行潜在合作以造福社区。

- **Phi3 与 Llama3 的性能难题**：讨论显示了对 **Phi3** 模型的褒贬不一，一些人认为 `phi-3-medium` 不如 **llama3-8b**。一位用户强调，在超过 2048 tokens 的上下文长度后，Phi3 的表现逊于 Llama3。

- **激烈的模型部署讨论**：社区交流了利用 **Runpods** 和 Docker 部署模型的想法，部分成员遇到了服务商的问题。虽然未提供具体的 Dockerfiles，但建议在服务器中搜索相关内容。

- **Colab 付费版未达预期**：Google Colab 的 Premium 服务因持续的断连问题受到批评。成员建议转向 **Kaggle** 和 **Lightning AI** 等其他平台作为可行的免费替代方案。

- **Unsloth 本地开发实战**：用户开始使用 Unsloth 进行监督微调（SFT），讨论了在本地运行模型，特别是在 **VSCode** 中执行简历要点生成等任务。分享了使用 Unsloth 进行无监督微调的 Colab 笔记本和 GitHub 资源链接，例如此[微调指南](https://github.com/unslothai/unsloth#-finetune-for-free)和 [Colab 示例](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning 的挫败感与市场沉思**：工程师们讨论了 **fine-tuning** 的挑战，包括对 Google Gemini 1.5 API 价格上涨的担忧，以及在生产环境中提供 fine-tuned 模型的困难。有人提议设立一个专门针对 LLM 相关工作机会的 **channel**，并强调了对强大的 **JSON/Parquet 文件处理工具**的需求。

**技术研讨会的内幕**：参与者交流了关于 **LLM fine-tuning 策略**的见解，重点在于个性化销售邮件和法律文档摘要。会议还辩论了 **multi-agent LLM 协作**的实用性以及针对 Stable Diffusion 的 prompt 优化。

**探索 AI 生态系统**：社区深入探讨了各种 AI 话题，指出 **Braintrust** 是评估非确定性系统的便捷工具，并分享了 **O'Reilly Radar** 关于构建 LLM 复杂性的见解。讨论还强调了 **Autoevals** 在 SQL 查询评估方面的潜力。

**LLM 工作工具箱**：工程师们解决了诸如 **Modal 的不透明故障**和 *Axolotl 预处理* GPU 支持等实际问题。分享了关于在 **Jarvislabs** 上使用共享存储的疑问，以及对 **Wing Axolotl** 模型量化的见解，讨论中穿插了许多有用的资源和技巧。

**代码、工艺与社区**：社区氛围活跃，讨论了 LLM *evaluator models*（评估器模型）、Gradio UI 优于 Streamlit 的吸引力，以及从**圣迭戈**到**纽约**的聚会。这些充满活力的交流不仅涵盖了技术领域，还培育了 AI 工程界的社交纽带。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**GPGPU 编程拥抱 lighting.ai**：工程师们讨论了将 **lighting.ai** 作为 GPGPU 编程的一个值得推荐的选择，特别是对于那些无法使用通常用于 CUDA 和 SYCL 开发的 NVIDIA 硬件的人。

**简化 Triton 开发**：开发者们发现 [triton_util](https://github.com/UmerHA/triton_util)（一个简化 Triton kernel 编写的工具包）在抽象重复性任务方面非常有用，提升了直观体验。观察到在 NVIDIA A6000 GPU 上使用 Triton 带来的性能飞跃，同时在处理超过 65GB 的大型 tensor 时，解决 bug 成为了关注焦点。

**Nightly 版 Torch 支持 Python 3.12**：PyTorch 社区强调了 **torch.compile** 在 Python 3.12 上的问题，nightly 构建版本提供了一些解决方案。同时，Torch 2.3 中弃用 macOS x86 构建引发了关于向 M1 芯片或 Linux 迁移的讨论。

**Tom Yeh 强化 AI 基础**：
[Prof Tom Yeh](https://x.com/ProfTomYeh) 通过分享 AI 概念的手算练习获得了关注。他的系列作品包括 [Dot Product](https://x.com/ProfTomYeh/status/1793623127643037891)（点积）、[Matrix Multiplication](https://x.com/ProfTomYeh/status/1794070094898704456)（矩阵乘法）、[Linear Layer](https://x.com/ProfTomYeh/status/1794451228681712037)（线性层）和 [Activation](https://x.com/ProfTomYeh/status/1794848226383655284)（激活函数）工作簿。

**量化领域的飞跃**：工程师们正积极讨论并使用 **bitsandbytes** 和 **fbgemm_gpu** 等库改进量化过程，并参加 NeurIPS 等竞赛。分享了在 **Llama2-7B** 上的工作和 **FP6-LLM** 仓库的更新，同时也对 torchao 社区的互助精神表示赞赏。

**CUDA 调试技能提升**：分享了一个关于调试 SYCL 代码的询问，强调了对改进 kernel 代码分析工具的需求，并可能深入到调试过程中。

**通过 bitnet PR 加速开发**：
bitnet channel 解决了各种技术问题，包括与 PyTorch/dev 版本和 CUDA 不匹配相关的 `ImportError` 挑战，以及通过升级 **gcc 12.1** 解决了大学服务器上的编译难题。讨论了关于位打包（bit packing）和 CI 改进的协作 PR 工作，并提供了位级操作和错误解决的资源（[GitHub 上的 BitBlas](https://github.com/microsoft/BitBLAS)，[ao GitHub issue](https://github.com/pytorch/ao/issues/288)）。

**柏林与西雅图的社交与科技故事**：在 off-topic 频道的对话对比了西雅图和柏林的社交与天气景观。柏林因其 techno 音乐场景和创业友好性而受到推崇，尽管也有其阴郁的天气。

**Tokenizer 故事与训练讨论**：随后进行了关于自主实现 tokenizer 和数据集处理的广泛对话，考虑了压缩和云存储选项。在 H100 GPU 上进行大规模训练的成本仍然高昂，而关于 GPU 规格的细粒度讨论为模型优化提供了参考。训练实验正在迅速进行，其中一个实验的强度堪比 GPT-3。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**玩转长上下文 (Big Contexts)**：一位工程师建议训练一个具有**极长上下文窗口 (context window)** 的 Large Language Model (LLM)，其理念是只要有足够的上下文，即使数据集较小，LLM 也能做出更好的预测。

**无偏评估的困境**：人们对 [Scale 的参与](https://scale.com/leaderboard)（既提供数据又评估机器学习模型）表示担忧，强调了潜在的利益冲突可能会影响模型评估的公正性。

**深入理解 RAG 基础之外的内容**：技术讨论阐明了 **Retrieval-Augmented Generation (RAG)** 系统的复杂性，强调它不仅仅是向量相似度匹配，还涉及一系列其他过程，如重排序 (re-ranking) 和全文搜索，正如 [RAGAS](https://github.com/explodinggradients/ragas) 等讨论和资源所强调的那样。

**价格翻倍，担忧也翻倍**：Google 决定提高 Gemini 1.5 Flash 输出价格的举动引发了激烈辩论，工程师们指责这种定价策略不可持续，并对 API 成本结构的可靠性提出质疑。

**梯度累积 (Gradient Accumulation) 审查**：讨论中出现了一个关于在模型训练中避免梯度累积的话题，工程师们参考了 [Google 的调优手册 (tuning playbook)](https://github.com/google-research/tuning_playbook) 以获取见解，同时还根据 [Hugging Face 的文档](https://huggingface.co/docs/trl/main/en/dpo_trainer#reference-model-considerations-with-peft)讨论了 DPO 训练中 `ref_model` 的概念。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **开源还是闭源？LM Studio 的二分法**：LM Studio 的主应用程序被确认为**闭源**，而 **LMS Client (CLI)** 和 **lmstudio.js (新 SDK)** 等工具是开源的。LM Studio 内的模型无法直接访问本地 PC 文件。

- **翻译模型热议**：[Aya 日译英模型](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF)被推荐用于翻译任务，而支持 80 多种编程语言的 **Codestral** 引发了将其集成到 LM Studio 的讨论。

- **GPU 选择与性能讨论**：关于**多 GPU 配置**与单个强力 GPU 优势的辩论浮出水面，特别是对 **Nvidia** 股票的价值和**改装 GPU (modded GPUs)** 实用性的质疑。一位 **Goldensun3ds** 用户升级到了 **44GB VRAM**，展示了该配置的优势。

- **服务器模式拖慢进度**：用户注意到，在预设相同的情况下，对话模式 (chat mode) 比服务器模式 (server mode) 获得结果的速度更快，这引发了对 **GPU 利用率**以及服务器模式操作中 **GPU 选择**必要性的关注。

- **AMD GPU 用户面临 ROCm 障碍**：用户注意到了 **LM Studio 与 Radeon GPU** 的版本兼容问题，包括尝试使用 **iGPU** 和在 **ROCm 模式**下进行多 GPU 配置的失败尝试。社区分享了 **7900 XT** 的优惠信息，作为扩展 VRAM 的可能方案。

- **一个 AI 能身兼两职吗？**：一个模型同时执行审核 (moderation) 和问答 (Q&A) 的可行性受到质疑，建议指向使用两个独立的模型，或利用服务器模式以获得更好的上下文处理能力。

- **Codestral 可用性公布**：Mistral 新推出的 **22B 编程模型 Codestral** 已经发布，目标用户是寻求强大编程助手的拥有大显存 GPU 的用户。该模型可在 [Hugging Face](https://huggingface.co/lmstudio-community/Codestral-22B-v0.1-GGUF) 下载。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 的内存管理之路**：一篇博客文章阐述了 Mojo 以 **所有权 (ownership)** 为核心的内存管理方法，倡导一种既安全又高性能的编程模型。[Chris Lattner 的视频](https://www.modular.com/team/chris-lattner) 被推荐为深入研究 Mojo 编译器系统中所有权概念的资源。更多内容请参阅其 [博客文章](https://www.modular.com/blog/what-ownership-is-really-about-a-mental-model-approach)。

**对齐的优势**：工程师们强调了表中 **64 字节对齐 (64-byte alignment)** 的重要性，以充分发挥 AVX512 指令的效能并提高缓存效率。他们还强调了对齐对于触发预取器 (prefetcher) 最佳性能的必要性，以及多线程环境中的 **伪共享 (false sharing)** 问题。

**Mojo 中的 Optional 困境与 Dict 难题**：在 `nightly` 分支的对话中，`Optional` 与 `ref` API 的结合使用引发了广泛讨论，参与者将 Rust 的 `?` 运算符作为建设性的参考。相关的 GitHub [issue](https://github.com/modularml/mojo/issues/2869) 也关注了一个关于 `InlineArray` 无法调用其元素析构函数 (destructors) 的 bug。

**提案与编译的详述**：关于自动解引用 (auto-dereferenced) 引用中的命名规范进行了严格辩论，有人提议将 `Reference` 重命名为 `TrackedPointer`，将 `Pointer` 重命名为 `UntrackedPointer`。此外，最新的 nightly Mojo 编译器版本 `2024.5.2912` 带来了诸如异步函数借用限制 (async function borrow restrictions) 等更新，并提供了详细的 [更新日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

**AI 拓展开放世界游戏视野**：有人断言，如果 AI 能够根据用户交互，从广泛的在线模型中动态构建世界，开放世界游戏将达到新的巅峰。这一想法暗示了 AI 在推动游戏进步方面的重大机遇。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 新手的帮手**：EleutherAI 的新人（包括一名即将毕业的计算机科学系学生）获得了入门级的研究课题，以及诸如 [GitHub gist](https://gist.github.com/ad8e/da8fdfe0ec586b5a548aaa14327f7722) 之类的资源。由于缺乏基础 AI 问答平台，引发了关于 AI 知识对初学者可及性的讨论。

- **过早发表论文令同行困惑**：一篇因在缺乏实验支持的情况下提出大胆主张而引起社区关注的论文引发了讨论。人们对其在 *arXiv* 上的录用表示质疑，与之相对的是对 Yann LeCun 富有影响力的指导及其专题 [讲座](https://youtu.be/gG5NCkMerHU) 的认可，该讲座强调了工程学与基础科学之间的区别。

- **MLP 与 Transformer —— 趋势的转变**：关于最近发现 MLP 在上下文学习 (in-context learning) 方面可以与 Transformer 媲美的争论升温。虽然对 MLP 的潜力很感兴趣，但对其优化和通用性仍普遍存在怀疑，成员们引用了诸如 [MLPs Learn In-Context](https://arxiv.org/abs/2405.15618) 等资源，讨论也反思了 AI 架构演进中的“苦涩的教训” (Bitter Lesson)。

- **AMD 回溯在内存计算时出错**：一位成员在尝试计算 AMD 系统上的最大内存时遇到回溯错误 (traceback error)，并通过 [GitHub Gist](https://gist.github.com/jonabur/0004bf39a3cec65262cf72f556c316c4) 分享了该问题；而另一位成员则在寻求关于使用 "lm-evaluation-harness" 进行并发查询和基于 logits 测试的建议。

- **扩展讨论转向支持 MLP**：对话显示，优化技巧可能会掩盖性能不足，同时突出了一个观察结果：扩展性 (scaling) 和适应性可能掩盖 MLP 的结构缺陷。分享的链接包括一项 [比较 CNN、Transformer 和 MLP 网络的实证研究](https://arxiv.org/abs/2108.13002#microsoft) 以及一项关于 [扩展 MLP (scaling MLPs)](https://arxiv.org/abs/2306.13575) 的调查。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **免费用户喜迎新功能！**：ChatGPT 的免费用户现在可以享受更多功能，包括 **browse**、**vision**、**data analysis**、**file uploads** 以及访问各种 **GPTs**。

- **ImaGen3 引发复杂情绪**：围绕 Google 即将发布的 **ImaGen3** 讨论纷纷，主要集中在对媒体操纵和信任度的怀疑。同时，Google 还因历史图像生成中的准确性失误而面临指责。

- **GPT-4 的记忆问题亟待解决**：工程师们对 GPT-4 断断续续的“失忆症”表示不满，希望能有更透明的记忆机制，并建议增加一个用于长期记忆保存的 **backup button**。

- **RAM 占用上升：用户呼吁优化**：对内存消耗过高的担忧激增，特别是在 Brave 等浏览器上使用 ChatGPT 时；建议的替代方案包括使用 Safari 或桌面应用以获得更流畅的体验。

- **共享 Prompt 的中心枢纽**：对于那些寻找“神奇 Prompt”资源库的用户，请关注 Discord 社区中专门为此设立的频道。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral 进军编程领域**：[Codestral](https://mistral.ai/news/codestral/) 是来自 Mistral 的新款 22B 模型，精通 80 多种编程语言，现已发布并在 8 周测试期内可在 [HuggingFace](https://huggingface.co/mistralai/Codestral-22B-v0.1) 上访问。与此同时，Scale AI 推出的基于私有数据的 [LLM leaderboard](https://scale.com/leaderboard) 引发了关于模型评估潜在偏见的讨论，原因在于该公司的营收模式及其对固定众包人员的依赖。

- **价格上涨给 Gemini 1.5 Flash 的欢呼泼了冷水**：Google 的 Gemini 1.5 Flash 输出价格在发布并获得好评后突然上调——从 $0.53/1M 增加到 $1.05/1M——引发了对 API 稳定性和信任度的争论。

- **OpenAI 董事会的尴尬博弈**：根据前董事会成员 Helen Toner 的爆料，OpenAI 董事会是在 Twitter 上才得知 ChatGPT 发布的消息。这一事件揭示了 OpenAI 内部更广泛的透明度问题，而 Sam Altman 被解雇时缺乏明确理由（董事会称其为“沟通不始终坦诚”）使这一问题更加复杂。

- **Toner 的爆料与 OpenAI 的不透明性主导讨论**：Toner 关于 Sam Altman 领导下频繁出现不诚实行为的指控引发了对其披露时机的辩论，有人猜测存在法律限制，并承认内部政治和压力可能塑造了董事会的叙事。

- **DL 社区的知识盛宴**：智力交流活动热度激增，如组建“小型期刊俱乐部”以及对 **Cohere 教育视频系列** 的赞赏，同时 **TalkRL podcast** 被认为价值被低估。尽管 Schulman 在 Dwarkesh 的播客节目中对 AI safety 的务实看法评价褒贬不一，但 [Andrew Carr 的推文](https://x.com/andrew_n_carr/status/1782878279504191896) 中强调的旨在减轻 AI 错误行为的变革性分层模型正引起关注。

- **对 FMTI 文件格式问题的沮丧**：由于 FMTI GitHub 仓库选择使用 CSV 而非 Markdown，导致工程师难以轻松获取论文评分，社区对此表示不满。

- **SnailBot 即将发布**：SnailBot News 更新即将到来，已通过标签进行预热，Nate Lambert 也引起了大家对即将推出的贴纸的好奇。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Colab 和 Kaggle 加速图像生成**：工程师建议使用 **Kaggle** 或 **Colab** 来加快 Stable Diffusion 的图像生成速度；有报告称在 Colab 上使用 **16GB VRAM** 生成每张图像需要 **1.5 到 2 分钟**。
  
- **训练 SDXL LoRA 模型的技巧**：技术爱好者讨论了训练 **Stable Diffusion XL LoRA** 模型的心得，强调 2-3 个 epoch 即可获得良好效果，并建议触发词（trigger words）越简洁，训练效果越好。
  
- **配置 ComfyUI 模型路径与 API 集成**：社区成员正在解决 **ComfyUI** 多个模型目录的配置问题，并讨论如何在本地 Stable Diffusion API 中集成 **ADetailer**。

- **HUG 与 Stability AI 课程方案**：有消息称 **HUG 和 Stability AI** 合作提供创意 AI 课程，课程环节将被录制以便后续访问——填写完反馈表后将退还参与者的押金。

- **3D 模型生成仍处于孵化阶段**：对话转向 AI 在创建适用于打印的 **3D 模型** 中的作用，成员们一致认为当前 AI 生成此类模型的能力尚未达到预期潜力。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **绘制 LLM 知识图谱蓝图**：[LlamaIndex 发布了 PropertyGraphIndex](https://www.llamaindex.ai/blog/introducing-the-property-graph-index-a-powerful-new-way-to-build-knowledge-graphs-with-llms)，这是与 Neo4j 的合作成果，允许构建更丰富的由 LLM 支持的知识图谱。该工具提供图提取和查询功能，支持自定义提取器以及向量/图联合搜索功能——用户可以参考 [PropertyGraphIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/) 获取指南。

- **优化知识检索**：讨论集中在通过实验文本分块（chunk）大小来优化 RAG 模型，并参考 [SemanticDocumentParser](https://github.com/isaackogan/SemanticDocumentParser) 生成高质量分块。此外，还分享了最大化向量存储潜力的策略，例如提到的 `QueryFusionRetriever`，以及非英语 Embedding 的最佳实践，并引用了 [asafaya/bert-base-arabic](https://huggingface.co/asafaya/bert-base-arabic) 等资源。

- **Codestral 时代的创新**：LlamaIndex 支持来自 MistralAI 的新模型 [Codestral](https://t.co/k2nHDiMnwD)，该模型涵盖 80 多种编程语言，并可通过 [Ollama](https://t.co/gsPHHF4c0K) 等工具进行本地运行增强。此外，[FinTextQA 数据集](https://t.co/emhQYXY1S4) 为基于金融文档的查询提供了广泛的问答对。

- **使用 Document Stores 进行存储与定制**：社区讨论了在 LlamaIndex 中管理文档节点和存储，涉及 `docstore.persist()` 的功能以及不同文档后端的使用，并参考了 [Document Stores - LlamaIndex](https://docs.llamaindex.ai/en/latest/module_guides/storing/docstores/)。交流中还提到了 [Simple Fusion Retriever](https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion/) 作为管理向量存储索引的解决方案。

- **跨越边界的查询**：新发布的 Property Graph Index 强调了 LlamaIndex 扩展知识图谱查询能力的承诺，集成了处理节点和关系的标签（labels）及属性（properties）的功能。[LlamaIndex 博客](https://t.co/X9D3Wl0Hyv) 阐明了这些进展及其对 AI 工程领域的潜在影响。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini 1.5 证明了其实力**：根据 [LMSysOrg 的 Twitter](https://x.com/lmsysorg/status/1795512202465845686) 分享的结果，**Gemini 1.5 Pro/Advanced** 目前位居第二，紧随 GPT-4o 之后；**Gemini 1.5 Flash** 位列第九，超越了 Llama-3-70b 等模型。

- **SWE-agent 引起广泛关注**：普林斯顿大学的 **SWE-agent** 因其声称的卓越性能和开源状态而引发热议，详细信息可见 [Gergely Orosz 的 Twitter](https://x.com/GergelyOrosz/status/1794743519954731331) 和 [SWE-agent GitHub](https://github.com/princeton-nlp/SWE-agent)。

- **Llama3-V 加入竞争**：新的开源模型 **Llama3-V** 尽管体积较小，但能与 GPT4-V 展开竞争，[Sidd Rsh 的 Twitter](https://x.com/siddrrsh/status/1795541002620727439) 详细介绍了这一备受关注的进展。

- **LLM 实战经验谈**：文章 "[What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/)" 探讨了一年来使用 LLM 的见解和经验，重点关注构建 AI 产品过程中的演进与挑战。

- **SCALE 通过 SEAL Leaderboards 设定 LLM 基准测试标准**：**Scale 的 SEAL Leaderboards** 已发布，用于进行稳健的 LLM 评估，并获得了 [Alexandr Wang](https://x.com/alexandr_wang/status/1795857651592491281) 和 [Andrej Karpathy](https://x.com/karpathy/status/1795873666481402010) 等行业人士的认可。

- **预留 Latent Space 虚拟席位**：今天将举行一场探讨 **AI Agent 架构与 Kolmogorov Arnold Networks** 的技术活动，[点击此处注册](https://lu.ma/pxnaq641)。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 临时宕机已解决**：OpenAI 遭遇了短暂的服务中断，但在快速修复后已恢复正常运行。Alex Atallah 指出 Azure 服务在整个事件期间保持运行。

- **告别 Cinematika**：由于使用率较低，**Cinematika 模型** 即将弃用；建议用户尽快切换到替代模型。

- **资金上限困扰已修复**：在 OpenAI 模型因意外触发支出限制而无法访问后，官方已实施解决方案并恢复正常服务，同时推出了额外的安全防护措施。

- **GPT-4o 上下文容量确认**：针对 Token 限制的误解，Alex Atallah 表示 GPT-4o 保持 128k Token 的上下文限制，以及 4096 的独立输出 Token 上限。

- **对 GPT-4o 图像提示词性能的担忧**：用户反映使用 `openai/gpt-4o` 处理 `image-url` 输入时速度较慢，这暗示可能存在性能瓶颈，可能需要进一步调查和优化。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 影响力人物成为焦点**：Helen Toner 关于在 Twitter 上发现 ChatGPT 的言论引发了对话；而 Yann LeCun 在卸任 Facebook VP 后的研究活动也引起了兴趣，标志着 AI 领袖在塑造社区舆论方面的持续影响力。相比之下，Elon Musk 仅在 AI 模型失去竞争优势时才将其公开的做法，引发了关于 AI 开发中开源模型策略的讨论。

- **Mistral 的许可协议利用开放权重**：在讨论中，Mistral AI 的许可策略因其在非商业框架下结合开放权重的模式而受到关注，强调了 AI 模型共享与商业化之间的复杂格局。

- **模型生成复杂性**：在模型生成中，即使是像“一个女人在看书”这样看似简单的提示词也会出现困难，用户报告在合成 Caption 生成中存在负面影响，暗示了生成式 AI 领域持久的挑战。

- **关于判别器有效性的探讨**：社区剖析了研究资料，特别注意到 Dinov2 作为判别器（Discriminator）的使用，但更倾向于使用修改后的预训练 UNet，这让人联想到类似于 Kandinsky 的策略（即减半的 UNet 可提高性能），揭示了 AI 研究中判别器技术的演进。

- **社区对评分激励机制的怀疑**：关于 Horde AI 社区对 SD 图像评分的激励系统引发了质疑，有观点认为此类计划可能会降低数据质量，突显了社区参与度与数据完整性之间常见的张力。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain v2.0 Agent 查找问题已解决**：用户最初在 **LangChain v2.0** 中查找 Agent 时遇到困难，随后通过讨论成功定位并实现了这些 Agent。
- **关于 AI 与创造力的见解引发讨论**：一条推文建议 AI 应超越重复，迈向真正的创造力，这引发了关于 AI 在创意领域潜力的技术讨论。
- **解决 LangChain 中的 'RateLimit' 错误**：社区分享了处理 **LangChain** 应用中 "RateLimit" 错误的解决方案，提倡使用 **Python 的 try/except 结构** 进行稳健的错误管理。
- **优化对话式数据检索**：成员在处理多个 Vector Store 时面临 **ConversationalRetrievalChain** 的挑战，寻求关于有效合并数据以实现完整内容检索的建议。
- **持久化聊天功能的实际演示**：一位频道成员测试了 **langserve** 的持久化聊天历史功能，参考了仓库中的示例，并询问如何将 "chat_history" 整合到 FastAPI 请求体中，相关文档见 [此处](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/client.ipynb)。

关于使用 **LangChain** 在 Agent 流中实现路由逻辑的教学内容已通过 [YouTube 教程](https://youtu.be/KtbRexZ6vsc) 发布，帮助社区成员增强其自动化 Agent 的决策路径。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **训练工作流中定制化为王**：工程师们对 *个性化训练工作流* 表现出兴趣，讨论集中在针对个人用例增强 Open Interpreter，表明 AI 工具对定制化有显著需求。

- **用户分享 Open Interpreter 应用**：Open Interpreter 的各种用例引发了讨论，成员们交流了如何利用其功能进行不同技术应用的想法。

- **寻找开源替代方案**：工程师之间的对话强调了对 Rewind 替代方案的持续探索，其中 **Rem** 和 **Cohere API** 被提及为处理 Vector DB 的值得关注的选择。 

- **Rewind 的连接性获得认可**：一位用户证实了 Rewind 的效率，称其为“生活黑客技巧”，尽管它在隐藏敏感数据方面存在不足，这反映了技术用户普遍持正面态度。

- **消除 OI 中的确认步骤**：为了提高效率，一位成员提供了运行 Open Interpreter 时无需确认步骤的解决方案，即使用 `--auto_run` 功能，详见 [官方文档](https://docs.openinterpreter.com/settings/all-settings#auto-run)。

- **M5 屏幕问题**：一位用户报告其 M5 在刷机后显示白屏，引发了故障排除讨论，包括建议更改 Arduino studio 设置以在刷机期间执行全内存擦除。 

- **未说明的 YouTube 链接**：一位成员分享了一个孤立的 [YouTube 视频](https://www.youtube.com/watch?v=sqwtk18pw14) 链接，未提供上下文，可能错失了讨论或提供有价值见解的机会。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**“NSFW”垃圾信息清理**：**OpenAccess AI Collective (axolotl)** 的管理员迅速响应了关于 **NSFW Discord 邀请链接** 在各频道被滥发的警报，垃圾信息已得到及时处理。

**探索多媒体模型掌握**：在 *general* 频道中有人询问如何微调 **LLM**（如 **LLava 模型**）以实现图像和视频理解，但该问题目前尚未得到解答。

**为 MoE 提供 Gradient Checkpointing**：*axolotl-dev* 频道的一位成员提议更新 **Unsloth 的 Gradient Checkpointing** 以支持 **MoE 架构**，验证后将提交 Pull Request (PR)。

**Bin Packing 算法排错**：一项开发更新指向了 **改进的 Bin Packing 算法**，但指出在评估后训练停滞的问题，可能与新 Sampler 缺失 `_len_est` 实现有关。

**Sampler 回退引起关注**：通过分享一个 **[回退 multipack batch sampler 更改的 PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1672)** 指出了代码回退，原因是损失计算存在缺陷，这表明在模型训练中精确指标评估的重要性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**利用 RAG 重新思考 PDF 微调**：一位成员提议将 **Retrieval Augmented Generation (RAG)** 作为处理 PDF 时替代传统 JSONL 微调的更明智选择，声称它可以完全消除微调步骤。

**特定 API 的 Grounded Generation 见解**：引用了 API 文档来展示如何在 **grounded generation framework** 中使用 `response.citations` 功能，并提供了一个 [Hugging Face 链接](https://huggingface.co/CohereForAI/c4ai-command-r-plus#grounded-generation-and-rag-capabilities) 作为参考。

**本地 R+ 创新与强制引用显示**：一位工程师分享了在本地 Command R+ 设置中集成 **具有强制引用显示的 RAG 流水线** 的实践成果，展示了维持源归属的可靠方法。

**Cohere Discord 机器人使用强调了分段讨论**：围绕由 **Cohere** 驱动的 Discord 机器人的热情引发了一个提醒，即应将项目讨论保持在专用频道内，以维持社区讨论的秩序和重点。

**频道规范鼓励项目隔离**：在对社区构建的 Discord 机器人表示认可后，给出了将详细讨论移至指定项目频道的指导，以确保遵守服务器的组织规范。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**xAI 获得高达 60 亿美元融资**：Elon Musk 的 xAI 已成功 [筹集了 60 亿美元](https://x.ai/blog/series-b)，知名投资者包括 Andreessen Horowitz 和 Sequoia Capital。这笔资金旨在用于首批产品的市场推广、大规模基础设施建设以及推进未来技术的研发。

**对未具名分析工具的质疑**：一位成员对某些分析工具表示怀疑，认为它们的“实用性微乎其微”，但并未指明具体是哪些工具。

**新语言 Bend 引起关注**：Bend 编程语言因其“无需任何代码即可自动多线程”的能力而受到赞誉，这一特性与 tinygrad 的 lazy execution 策略相得益彰，正如 [Fireship 视频](https://www.youtube.com/channel/UC0v-tlzsn0QZwJnkiaUSJVQ) 中所示。

**tinybox 电源供应查询**：有人提出了关于 tinybox 电源要求的问题，询问它是使用“两个消费级电源还是两个带有配电板的服务器级电源”，但尚未得到解决。

**链接关注**：The Verge 关于 xAI 融资的一篇文章特别询问了这笔资金中会有多少比例用于购买 GPU，这是 AI 工程师对计算基础设施的一个关键关注点。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Goliath 需要“辅助轮”**：在进行额外的 pretraining 之前，**Goliath** 经历了显著的性能下滑，引发了社区成员之间的协作分析和应对。

- **经济高效地复现 GPT-2 里程碑**：工程师们讨论了在 [GitHub](https://github.com/karpathy/llm.c/discussions/481) 上仅用 20 美元就以 C 语言实现了 GPT-2 (124M) 的复现，并指出其 HellaSwag 准确率为 29.9，超过了 GPT-2 原始的 29.4 分。

- **Codestral-22B：多语言巨兽**：**Mistral AI** 发布了 **Codestral-22B**，这是一个在 80 多种编程语言上训练的庞然大物，据 [Guillaume Lample 的公告](https://x.com/guillaumelample/status/1795820710750744839?s=46&t=1jtkL4JPu-DUOdo8JC668g) 称，它比前代产品更精通编程。

- **号召贡献者参与开源 GPT-4-Omni**：**LAION AI** 正在号召社区参与 GPT-4-Omni 的开源开发，并发布了一篇博文，重点介绍了数据集和教程，可在此处访问 [此处](https://laion.ai/notes/open-gpt-4-o/)。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Windows 下 Llamafile 的困扰**：一位工程师在 Windows 上 **编译 llamafile** 时遇到了问题，指出 `cosmoc++` 存在一个问题，即由于可执行文件在没有 `.exe` 后缀的情况下无法启动而导致构建失败。尽管系统报告文件缺失，但工程师确认该文件存在于目录 `.cosmocc/3.3.8/bin` 中，并且在使用 cosmo bash 时也遇到了同样的问题。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **RAG 拯救 LLM 幻觉问题**：一位工程师建议使用 **Retrieval Augmented Generation (RAG)** 来解决 **Language Models (LLMs)** 在回答文档查询时的幻觉问题。他们提议对 `llm` 命令进行扩展，以递归地为给定 URL 创建 embeddings，利用文档数据集和 embedding 存储来提高准确性。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**技术交流一瞥**：一位用户简要提到发现了一篇与其兴趣相关的论文，感谢了另一位用户的分享，并表示打算进行研读。然而，并未提供关于该论文内容、标题或研究领域的具体细节。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

> 完整的逐频道分析已针对邮件进行了截断。
> 
> 如果您想查看完整的分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}