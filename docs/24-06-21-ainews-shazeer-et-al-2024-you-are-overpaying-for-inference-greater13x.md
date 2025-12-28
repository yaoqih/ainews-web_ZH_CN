---
companies:
- character.ai
- anthropic
date: '2024-06-22T00:48:48.532463Z'
description: '**Noam Shazeer** 解释了 **Character.ai** 如何在承载相当于 **20% 谷歌搜索流量**的大模型（LLM）推理任务的同时，将服务成本降至
  2022 年底的 **1/33**，而目前领先的商业 API 成本至少要高出 **13.5 倍**。


  关键的内存效率技术包括：

  *   **MQA（多查询注意力）优于 GQA（分组查询注意力）**：将 KV 缓存（KV cache）大小缩减了 8 倍；

  *   **混合注意力范围**（Hybrid attention horizons）；

  *   **跨层 KV 共享**（Cross-layer KV-sharing）；

  *   **有状态缓存**（Stateful caching）：缓存命中率高达 95%；

  *   **原生 int8 精度**：配合自定义算子（Custom kernels）。


  此外，**Anthropic** 发布了 **Claude 3.5 Sonnet**，其性能超越了 **Claude 3 Opus**，且速度提升了两倍，成本仅为五分之一。该模型通过了
  **64%** 的内部拉取请求（PR）测试，并引入了 **Artifacts** 等新功能，支持实时文档和代码生成。关于大模型架构的讨论则强调了 Transformer
  的主导地位、规模扩展与过拟合带来的挑战，以及架构优化工作对推动技术进步的重要性。'
id: af97a688-ee5b-4774-b5ac-d3f723dd3834
models:
- claude-3.5-sonnet
- claude-3-opus
original_slug: ainews-shazeer-et-al-2024
people:
- noam-shazeer
- kevin-a-fischer
- sebastien-bubeck
- _aidan_clark_
- andrej-karpathy
title: Shazeer 等人 (2024)：你在推理上多支付了 13 倍以上的费用。
topics:
- memory-efficiency
- kv-cache
- attention-mechanisms
- stateful-caching
- int8-precision
- transformer-architecture
- scaling
- overfitting
- architecture
---

<!-- buttondown-editor-mode: plaintext -->**这 5 个内存和缓存技巧就是你所需要的一切。**

> 2024年6月20日至6月21日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**415** 个频道，**2822** 条消息）。
预计节省阅读时间（按 200wpm 计算）：**287 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在一篇简洁的 962 字博客文章中，[Noam Shazeer](https://x.com/NoamShazeer/status/1803790708358410380) 重新执笔解释了 Character.ai 如何在 LLM 推理方面承载相当于 **Google Search 流量 20%** 的负载，同时将服务成本降低了 33 倍（与 2022 年底相比），并估计**领先的商业 API 成本至少高出 13.5 倍**：

**内存效率**：“我们使用以下技术将 KV cache 大小减少了 20 倍以上，且没有降低质量。通过这些技术，GPU 内存不再是服务大 Batch Size 的瓶颈。”

1. **MQA > GQA**：“与大多数开源模型采用的 Grouped-Query Attention 相比，KV cache 大小减少了 8 倍。” ([Shazeer, 2019](https://arxiv.org/abs/1911.02150?ref=research.character.ai))
2. **混合注意力视界 (Hybrid attention horizons)**：局部（滑动窗口）注意力层与全局注意力层的比例为 1:5 ([Beltagy et al 2020](https://arxiv.org/abs/2004.05150v2))。
3. **跨层 KV 共享 (Cross Layer KV-sharing)**：局部注意力层与 2-3 个相邻层共享 KV cache，全局层则跨 Block 共享缓存。 ([Brandon et al 2024](https://arxiv.org/abs/2405.12981?ref=research.character.ai))

 
![image.png](https://assets.buttondown.email/images/861fadcb-48e1-484f-b22c-e40b0b1f199e.png?w=960&fit=max)
 

**有状态缓存 (Stateful Caching)**：“在 Character.AI 上，大多数聊天都是长对话；平均每条消息都有 180 条消息的对话历史……为了解决这个问题，我们开发了一个回合间缓存系统。”

4. **在具有树结构的 LRU 缓存中缓存 KV tensor** (RadixAttention, [Zheng et al., 2023](https://arxiv.org/abs/2312.07104?ref=research.character.ai))。在集群层面，我们使用粘性会话 (sticky sessions) 将来自同一对话的查询路由到同一台服务器。我们的系统实现了 95% 的缓存率。
5. **原生 int8 精度**：与更常见的“训练后量化 (post-training quantization)”不同。这需要他们自己定制用于矩阵乘法和注意力的 int8 Kernel——并承诺未来会发布关于量化训练的文章。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3 Opus 完成，从 4 次运行中取最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**Anthropic 发布 Claude 3.5 Sonnet**

- **性能提升**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790676988920098) 发布了 Claude 3.5 Sonnet，在关键评估中超越了竞争对手模型，速度是 Claude 3 Opus 的两倍，成本仅为其五分之一。它在理解细微差别、幽默和复杂指令方面表现出显著进步。[@alexalbert__](https://twitter.com/alexalbert__/status/1803804682412007850) 指出它通过了 **64% 的内部 Pull Request 测试用例**，而 Claude 3 Opus 为 38%。
- **新功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1803790681971859473) 推出了 Artifacts，允许生成文档、代码、图表、图形和游戏，这些内容会显示在聊天窗口旁边，以便进行实时迭代。[@omarsar0](https://twitter.com/omarsar0/status/1803907052785582508) 使用它来可视化深度学习概念。
- **编程能力**：在 [@alexalbert__](https://twitter.com/alexalbert__/status/1803804677701869748) 的演示中，Claude 3.5 Sonnet 自动修复了一个 Pull Request。[@virattt](https://twitter.com/virattt/status/1803906551658483911) 强调了 Agentic 编程评估，模型在其中读取代码、获取指令、创建行动计划、实施更改并接受测试评估。

**LLM 架构与扩展讨论**

- **Transformer 的主导地位**：[@KevinAFischer](https://twitter.com/KevinAFischer/status/1804214242297680256) 认为 Transformer 将继续扩展并占据主导地位，并将其与硅处理器类比。他建议学术界不要研究替代架构。
- **扩展与过拟合**：[@SebastienBubeck](https://twitter.com/SebastienBubeck/status/1803770413560029645) 讨论了迈向 AGI 的扩展挑战，指出模型在更大规模上可能会过拟合能力，而不是发现所需的“运算”。平滑的扩展轨迹并非必然。
- **架构的重要性**：[@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1804014969689903240) 强调了架构工作对推动当前进展的重要性，反驳了只有 Scaling 重要的观点。[@karpathy](https://twitter.com/karpathy/status/1803963383018066272) 分享了一个 94 行的 autograd 引擎，作为神经网络训练的核心。

**检索、RAG 与上下文长度**

- **Long-Context LLMs 对比检索**：Google DeepMind 的 [@kelvin_guu](https://twitter.com/kelvin_guu/status/1804175906602389687) 分享了一篇分析 Long-Context LLMs 在检索和推理任务中表现的论文。它们在无需显式训练的情况下可与检索和 RAG 系统相媲美，但在组合推理方面仍面临挑战。
- **用于无限上下文的 Infini-Transformer**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1804099837232615673) 重点介绍了 Infini-Transformer，它通过基于递归的 token mixer 和基于 GLU 的 channel mixer，在有限内存下实现了无限上下文。
- **改进 RAG 系统**：[@jxnlco](https://twitter.com/jxnlco/status/1803899526723387895) 讨论了改进 RAG 系统的策略，重点关注数据覆盖范围以及 metadata/indexing 能力，以增强搜索相关性和用户满意度。

**基准测试、评估与安全**

- **基准测试饱和的担忧**：一些人对基准测试变得饱和或效用下降表示担忧，例如 [@polynoamial](https://twitter.com/polynoamial/status/1803812369237528825) 对 GSM8K 的看法，以及 [@_arohan_](https://twitter.com/_arohan_/status/1803968038515150967) 对编程任务 HumanEval 的看法。
- **严格的发布前测试**：[@andy_l_jones](https://twitter.com/andy_l_jones/status/1803803061996888439) 强调了 @AISafetyInst 对 Claude 3.5 发布前的测试，这是政府首次在模型发布前对其进行评估。
- **评估赋能微调**：[@HamelHusain](https://twitter.com/HamelHusain/status/1803914267210772812) 分享了来自 @emilsedgh 的幻灯片，介绍了评估如何为 Fine-Tuning 奠定基础，从而产生飞轮效应。

**多模态模型与视觉**

- **多模态优先级的差异**：[@_philschmid](https://twitter.com/_philschmid/status/1803856518640734564) 对比了近期发布的产品，指出 OpenAI 和 DeepMind 优先考虑多模态，而 Anthropic 在 Claude 3.5 中专注于提升文本能力。
- **4M-21 Any-to-Any 模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1804138208814309626) 解析了 EPFL 和 Apple 的 4M-21 模型，这是一个支持文本转图像、深度掩码等功能的单一 any-to-any 模型。
- **用于指令的 PixelProse 数据集**：[@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1804141655320125801) 介绍了 PixelProse，这是一个包含 16M 图像的数据集，带有密集字幕，可使用 LLM 重构为指令和问答对。

**其他**

- **DeepSeek-Coder-V2 浏览器编程**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1804171764626526606) 展示了 DeepSeek-Coder-V2 直接在浏览器中开发小游戏和网站的能力。
- **LLM 生产化的挑战**：[@svpino](https://twitter.com/svpino/status/1803765665335038354) 指出，由于难以从演示阶段扩展到生产阶段，一些公司暂停了 LLM 的研发。然而，[@alexalbert__](https://twitter.com/alexalbert__/status/1803804691522035741) 分享到，Anthropic 的工程师现在使用 Claude 在编程任务中节省了数小时。
- **Mixture of Agents 击败 GPT-4**：[@corbtt](https://twitter.com/corbtt/status/1803813970018791845) 介绍了一种 Mixture of Agents (MoA) 模型，在击败 GPT-4 的同时，成本降低了 25 倍。它先生成初始补全，进行反思，然后生成最终输出。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Claude 3.5 Sonnet 发布**

- **令人印象深刻的性能**：在 /r/singularity 中，Anthropic 发布了 Claude 3.5 Sonnet，其在 [**LiveBench 和 GPQA 等基准测试中的表现超越了 GPT-4o 和其他模型**](https://www.reddit.com/r/singularity/comments/1dkinr5/all_next_gen_medium_models_are_better_than_their/)。在一项内部评估中，它 [**解决了 64% 的 Agent 编程问题，而 Claude 3 Opus 的解决率为 38%**](https://www.reddit.com/r/singularity/comments/1dkqlx0/claude_35_sonnet_significantly_outperforms_gpt4o/)。
- **视觉推理能力**：Claude 3.5 Sonnet [**在视觉任务上超越了 GPT-4o**](https://www.reddit.com/r/singularity/comments/1dkqlx0/claude_35_sonnet_significantly_outperforms_gpt4o/)，展示了令人印象深刻的视觉推理能力。
- **UI 增强**：正如 /r/LocalLLaMA 中提到的，除了性能提升外，Claude 3.5 Sonnet [**还带来了 UI 增强功能**](https://www.reddit.com/r/LocalLLaMA/comments/1dkdl1j/claude_rolled_out_sonnet_35_came_with_ui/)。
- **极具潜力的写作伙伴**：[一段 YouTube 视频](https://youtu.be/-dWfl7Dhb0o?si=3aYRkPKV5NAR8k6b)中分享的早期测试表明，Claude 3.5 Sonnet 作为写作伙伴展现出了巨大的潜力。

**OpenAI 与竞争**

- **对竞争的渴望**：在 /r/OpenAI 中，一些人表达了 [**希望 OpenAI 拥有能与 Claude 3.5 Sonnet 竞争的模型**](https://www.reddit.com/r/OpenAI/comments/1dkqvwh/it_seems_like_people_want_openai_to_not_have_a/) 的愿望，以保持该领域的竞争和进步。
- **批评与不信任**：OpenAI 正 [**面临批评和不信任，涉及忽视安全问题、违反算力承诺以及延迟推出的语音模型等问题**](https://www.reddit.com/r/OpenAI/comments/1dkqvwh/it_seems_like_people_want_openai_to_not_have_a/)。

**其他 AI 模型发布与基准测试**

- **来自中国的模型**：正如 /r/singularity 中指出的，[**近期超过一半的大语言模型（LLM）来自中国**](https://www.reddit.com/r/singularity/comments/1dkinr5/all_next_gen_medium_models_are_better_than_their/)。
- **Aider 基准测试**：根据 /r/LocalLLaMA 的一个帖子，DeepSeek Coder V2 和 Sonnet 3.5 [**在 Aider 基准测试排行榜中名列前茅**](https://www.reddit.com/r/LocalLLaMA/comments/1dkgjqg/aider_benchmark_updated_deepseek_sonnet_35/)。
- **Florence-2 视觉模型**：Microsoft 发布了 Florence-2，[**这是一款通用的开源视觉模型，能以统一的方式处理多项任务**](https://www.reddit.com/r/OpenAI/comments/1dkc1il/microsoft_unveils_florence2_vision_model/)，该消息分享于 /r/OpenAI。

**Stable Diffusion 与图像生成**

- **用于汽车生成的 Stable Diffusion 3**：在 /r/StableDiffusion 中，Stable Diffusion 3 medium 模型被认为 [**擅长遵循提示词生成汽车和其他主体**](https://www.reddit.com/r/StableDiffusion/comments/1dk5wq4/the_best_model_for_generating_cars/)，尽管在解剖结构和一致性方面仍需改进。
- **使用 SD XL 生成一致场景**：[一段 Reddit 视频帖子](https://v.redd.it/s8s2zbvfuq7d1)展示了如何结合 Stable Diffusion XL 与 img2img 及其他技术来生成一致的场景和角色。

**梗图与幽默**

- **ChatGPT 用户转向 Claude**：[/r/singularity 的一张梗图](https://www.reddit.com/r/singularity/comments/1dkubfy/its_cool_how_we_can_use_35_sonnet_for_free_while/)调侃了 ChatGPT 用户在看到 3.5 Sonnet 更新后纷纷转向 Claude 的现象。

---

# AI Discord 摘要

> **特别说明**：正如我们在新的前沿模型发布时所做的那样，我们将展示在 Claude 3.5、Claude 3 和 GPT-4o 上运行相同提示词的输出差异。

## Claude 3 Sonnet

**1. LLM 进展与基准测试**

- Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在 ChatbotArena 等排行榜上名列前茅，正如[此 Discord 讨论](https://discord.com/channels/974519864045756446/998381918976479273/1253429509143793747)中所述，其表现优于 GPT-4-Turbo 和 Claude 3 Opus。

- IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 和 DeepSeek 的 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)**（236B 参数）在[此频道](https://discord.com/channels/1110598183144399058/1111649100518133842/1253426309544149064)中因其代码能力而受到关注。

- [此研究频道](https://discord.com/channels/729741769192767510/747850033994662000/1253463120752541747)对某些基准测试表示怀疑，呼吁通过可靠来源建立现实的标准。

**2. 优化 LLM 推理与训练**

- [此频道](https://discord.com/channels/1189498204333543425/1189498205101109300/1253632387951099924)讨论了有望实现 4 倍 GPU 训练加速的 **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)**。

- 关于高效 KV-caching 的 **[vAttention](https://arxiv.org/abs/2405.04437)** 论文在[此处](https://discord.com/channels/1189498204333543425/1227345713348870156/1253426686817472522)被提及。

- [此讨论](https://discord.com/channels/1091220969173028894/1094454198688546826/1253448952267935876)提到了使用 W4A8KV4 量化进行 GPU 服务化的 **[QServe](https://arxiv.org/abs/2405.04532)**。

- [此频道](https://discord.com/channels/1189498204333543425/1189607750876008468/1253468296515293224)提到了探索并行解码的 **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 等技术。

**3. 开源 AI 框架与社区努力**  

- 支持多样化数据集的 **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 在[此处](https://discord.com/channels/1104757954588196865/1104757955204743201/1253426751699157043)受到关注。

- Andrew Ng 关于使用 **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 构建 Agentic RAG 系统的课程在[此频道](https://discord.com/channels/1059199217496772688/1059201661417037995/1253583385998131242)被提及。

- 开源的 **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 在[此讨论](https://discord.com/channels/1179035537009545276/1179035537529643040/1253426139494351009)中被介绍为处理“乏味”任务的顶级模型。

- **[Modular 的 Mojo](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 及其在 AI 扩展方面的潜力在[此处](https://discord.com/channels/1087530497313357884/1151418092052815884/1253581539716108380)被预告。

**4. 多模态 AI 与生成模型**

- [此频道](https://discord.com/channels/1002292111942635562/1002292112739549196/1253427405410930809)讨论了用于聊天的 **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 和用于编程的 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)**。

- [此处](https://discord.com/channels/1146610656779440188/1147665339266650133/1253432131070201938)提到了 **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** WebGPU 聊天机器人。

- [在此生成式 AI 讨论](https://discord.com/channels/822583790773862470/1075282825051385876/1253425446855639135)中，有人提议结合 Pixart Sigma、SDXL 和 PAG 以实现 DALLE-3 级别的输出。

- 关于图像重打光的 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 开源项目在[此频道](https://discord.com/channels/1002292111942635562/1002292112739549196/1253427405410930809)被分享。

## Claude 3.5 Sonnet

**1. AI 模型发布与性能对比**

- **新模型声称在基准测试中获胜**：Nous Research 的 [Hermes 2 Theta 70B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) 和 [Turbcat 8b](https://huggingface.co/turboderp/llama3-turbcat-instruct-8b) 均声称在多项基准测试中超越了 Llama-3 Instruct 等更大型的模型。各个 Discord 频道的用户讨论了这些发布，并将其能力与 GPT-4 和 Claude 等成熟模型进行了对比。

- **Claude 3.5 Sonnet 引发褒贬不一的反应**：多个 Discord 中的讨论强调了 Claude 3.5 Sonnet 提升的 Python 编程能力，但一些用户发现其在 JavaScript 任务中相较于 GPT-4 表现不足。在 Nous Research Discord 中，有人提到了该模型处理冷门编程语言的能力。

- **专注于代码的模型受到关注**：[DeepSeek Coder v2](https://ollama.com/library/deepseek-coder-v2) 的发布引发了关于代码任务专用模型的讨论，并声称在该领域的性能可与 GPT4-Turbo 媲美。

**2. AI 开发工具与基础设施挑战**

- **寻求 LangChain 的替代方案**：一篇详细介绍 Octomind 弃用 LangChain 的[博客文章](https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents)在多个 Discord 中引起共鸣，开发者们讨论了用于 AI Agent 开发的替代方案，如 Langgraph。

- **硬件限制令开发者受挫**：LM Studio 和 CUDA MODE Discord 中的讨论强调了在消费级硬件上运行先进 LLM 的持续挑战。用户辩论了各种 GPU 的优劣，包括 NVIDIA 的 4090 与即将推出的 5090，并探索了针对显存限制的变通方法。

- **Groq 的 Whisper 性能声明**：Groq 宣布以 [166 倍实时速度运行 Whisper 模型](https://groq.com/)，这在各频道引发了关注和质疑，开发者们讨论了其潜在的应用场景和局限性。

**3. AI 行业实践中的伦理担忧**

- **OpenAI 与政府的合作引发质疑**：一条讨论 OpenAI 向政府实体提供早期访问权限的[推文](https://fxtwitter.com/kimmonismus/status/1803908072999653528)在多个 Discord 中引发了关于 AI 监管和 AGI 安全策略的辩论。

- **Perplexity AI 面临批评**：一段批评 Perplexity AI 做法的 [CNBC 采访](https://youtu.be/MFdjEW8_SUg?si=eV12HJRyM1RhMRns) 导致各频道对 AI 开发和部署中的伦理考量展开了讨论。

- **OpenAI 的公共关系挑战**：包括 Interconnects 在内的多个 Discord 成员讨论了 OpenAI 代表反复出现的公关失误，并推测这些失误对公司公众形象和内部策略的影响。

## Claude 3 Opus

**1. 模型性能优化与基准测试**

- **[量化 (Quantization)]** 技术如 **AQLM** 和 **QuaRot** 旨在保持性能的同时，在单个 GPU 上运行大型语言模型 (**LLMs**)。例如：在 RTX3090 上运行 **Llama-3-70b** 的 [AQLM 项目](https://github.com/Vahe1994/AQLM)。

- 通过 **Dynamic Memory Compression (DMC)** 等方法提升 **Transformer** 效率的努力，在 **H100 GPUs** 上可能将吞吐量提高多达 370%。例如：@p_nawrot 的 [DMC 论文](https://arxiv.org/abs/2403.09636)。

- 关于优化 **CUDA** 操作的讨论，例如融合逐元素操作，使用 **Thrust 库的 `transform`** 以获得接近带宽饱和的性能。例如：[Thrust 文档](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each)。

- 在 **AlignBench** 和 **MT-Bench** 等基准测试中对**模型性能**进行比较，其中 **DeepSeek-V2** 在某些领域超越了 GPT-4。例如：[DeepSeek-V2 发布公告](https://x.com/deepseek_ai/status/1787478986731429933)。

**2. 微调挑战与提示工程 (Prompt Engineering) 策略**

- 在将 **Llama3** 模型转换为 GGUF 格式时，难以**保留微调数据**，并讨论了一个[已确认的 Bug](https://github.com/ggerganov/llama.cpp/issues/7062)。

- **提示词设计 (Prompt design)** 和使用正确模板（包括文本结束标记）的重要性，这会影响微调和评估期间的模型性能。例如：[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47)。

- **提示工程 (Prompt engineering)** 策略，如将复杂任务拆分为多个提示词，研究 **logit bias** 以获得更多控制。例如：[OpenAI logit bias 指南](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)。

- 教会 LLM 在不确定时使用 `<RET>` 标记进行**信息检索**，从而提高在低频查询中的表现。例如：[ArXiv 论文](https://arxiv.org/abs/2404.19705)。


**3. 开源 AI 进展与协作**

- **StoryDiffusion** 发布，这是一个采用 MIT 许可证的 Sora 开源替代方案，尽管权重尚未发布。例如：[GitHub 仓库](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)。 

- **OpenDevin** 发布，这是一个基于 Cognition 的 Devin 的开源自主 AI 工程师，配有[网络研讨会](https://lu.ma/fp0xr460)，在 GitHub 上关注度不断提高。

- 呼吁就预测 IPO 成功的开源**机器学习论文**进行协作，该项目托管在 [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html)。

- 围绕 **LlamaIndex** 集成的社区努力，包括在更新后 Supabase Vectorstore 和包导入中遇到的问题。例如：[llama-hub 文档](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack)。


## GPT4O (gpt-4o-2024-05-13)

1. **AI 模型性能与训练技术**：
   - **Gemini 1.5 在 1M tokens 下表现出色**：**Gemini 1.5 Pro** 通过有效处理**高达 1M tokens** 给用户留下了深刻印象，表现优于 **Claude 3.5** 等其他模型，并在长上下文任务中获得了积极反馈。该模型处理大量文档和转录文本的能力受到了关注。

   - **FP8 Flash Attention 和 GPTFast 加速推理**：关于 Flash Attention 中 **INT8/FP8 内核**以及最近推出的 **[GPTFast](https://github.com/MDK8888/GPTFast)** 的讨论表明，HF 模型推理速度显著提升了高达 9 倍。值得注意的还包括开源 FP8 Flash Attention 的增加，该功能将在 12.5 版本中获得官方 CUDA 支持。

   - **Null-shot 提示词与 DPO 优于 RLHF**：社区辩论涉及利用 LLM 幻觉的 **null-shot 提示词的有效性**，以及从**人类反馈强化学习 (RLHF)** 转向**直接策略优化 (DPO)** 以简化训练。论文引用包括该概念在 LLM 任务表现中的优势。

2. **AI 伦理与可访问性**：
   - **AI 伦理引发辩论**：一篇批评 OpenAI 背离开源原则的 [Nature 文章](https://www.nature.com/articles/d41586-024-02012-5) 引起了关于 AI 透明度和可访问性的讨论。人们对获取尖端 AI 工具和代码的难度日益增加表示担忧。

   - **避免不真诚的 AI 道歉**：用户对 AI 生成的道歉表示不满，称其不真诚且没有必要。这种情绪反映了人们对更真实、更实用的 AI 交互的广泛期望，而不是自动化的遗憾表达。

- **OpenAI 与政府合作的担忧**：人们对 OpenAI 向政府实体提供早期模型访问权限的担忧日益增加，这在一条 [推文](https://fxtwitter.com/kimmonismus/status/1803908072999653528) 中得到了强调。讨论指向了潜在的监管影响以及向 AGI 安全的战略转变。

3. **开源 AI 发展与社区贡献**：
   - **介绍 [Turbcat 8b](https://huggingface.co/turboderp/llama3-turbcat-instruct-8b)**：**Turbcat 8b 模型** 的发布包括了显著的改进，如扩展的数据集和新增的中文支持。该模型目前拥有 5GB 的数据，并与更大但尚未完善的模型进行了对比。

   - **Axolotl 与 Backgammon AI 工具**：协作努力重点介绍了开源的 [Backgammon AI 工具](https://github.com/C1N-S4/Backgamoon-A.I-tool)，该工具模拟西洋双陆棋场景以进行战略增强。讨论还涉及了 Turbcat 模型及其多语言处理功能。

   - **来自 Stability.ai 的计算机视觉数据集**：Stability.ai 发布了一个包含来自 Stable Diffusion 社区的 **235,000 个提示词和图像** 的数据集。这个 [StableSemantics](https://arxiv.org/abs/2406.13735v1) 数据集旨在通过提供广泛的视觉语义数据来增强计算机视觉系统。

4. **硬件与部署挑战**：
   - **GPU 使用挑战与优化**：工程师们分享了在不同设置中优化 GPU 和 CPU 集成的见解和解决方案，例如为 **LM Studio** 启用第二个 GPU，并讨论了运行复杂模型的替代方案。推荐使用二手 **3090s** 以提高成本效益，并期待 **NVIDIA 4090 和 5090** 之间的性能对比。

   - **TinyGrad 与 `clip_grad_norm_` 的纠葛**：在 **TinyGrad** 中实现 `clip_grad_norm_` 时遇到了瓶颈，原因是 **Metal 的缓冲区大小限制**，建议将其划分为 31 个张量块作为权宜之计。**Metal 和 CUDA** 之间的对比突显了性能差异，特别是针对梯度裁剪操作。

   - **模型部署问题**：在 Hugging Face 等平台上部署 **Unsloth** 等模型时遇到的挑战引发了关于 tokenizer 兼容性和替代部署建议的讨论。**Together.ai** 和 **Unsloth 的 H100** 之间的微调成本差异巨大，引发了对定价错误的质疑。

5. **活动讨论与职业机会**：
   - **Techstars 和 RecSys 虚拟聚会**：即将举行的活动，如 6 月 28 日至 30 日在旧金山举行的 **Techstars Startup Weekend** 和 6 月 29 日的 **RecSys Learners Virtual Meetup**，被强调为 AI 专业人士建立联系、学习和展示创新想法的机会。为了方便参与者，分享了详细信息和 RSVP 链接。

   - **求职与技能展示**：Python AI 工程师积极寻求工作机会，强调他们在 NLP 和 LLMs 方面的技能。对话还包括对公司支持框架的见解，例如 **Modal 团队** 对大模型的协助，以及开发者对 Slack 优于 Discord 的偏好。

   - **AI 活动中的演讲与公告**：推广了 LlamaIndex 创始人 Jerry Liu 在世博会上关于知识助手未来的演讲，并提到即将在 [Twitter](https://twitter.com/llama_index/status/1803880529571516917) 上发布的特别公告。

这些讨论全面展示了积极塑造 AI 社区的创新、伦理和实践方面。

---

# PART 1: High level Discord summaries

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 1.5 在 Token 竞赛中更进一步**：**Gemini 1.5 Pro** 在处理长上下文任务方面表现出色，具有处理高达 **1M tokens** 的惊人能力。用户赞赏其在输入文档和转录文本后在各种主题上展现的专业性，尽管它目前还存在一些访问限制。

- **AI 伦理辩论升温**：一篇 [Nature 文章](https://www.nature.com/articles/d41586-024-02012-5) 引发了关于 AI 模型开放性影响的讨论，激起了对 OpenAI 偏离开源原则的批评。对话指向了关于 AI 工具和代码可访问性更广泛的担忧。

- **在道歉中寻找平衡**：多位用户对 AI 频繁的道歉响应表示厌烦，嘲讽机器表达遗憾时的虚伪。这反映了用户对 AI 人设处理错误方式的普遍不满。

- **AI 开发者对 Mac 的偏好**：一位支持 MacBook 的用户反映了开发社区内的强烈偏好，强调了开发环境的易用性。虽然 Windows Surface 笔记本在硬件和设计上被提及为竞争对手，但 Windows 上的开发体验隐含地受到了批评。

- **Dall-E 3 挑战复杂图像生成**：用户尝试使用 Dall-E 3 生成具有不对称性和精确位置等特定属性的复杂图像，结果褒贬不一。值得注意的是，OpenAI macOS 应用因其与典型 Mac 工作流的集成而受到称赞，这表明该工具符合 AI 专业人士的生产力偏好。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **INT8/FP8 撼动性能表现**：受 [HippoML 博客文章](https://blog.hippoml.com/petaflops-inference-era-1-pflops-attention-and-preliminary-end-to-end-results-21f682cf2ed1) 的启发，关于 INT8/FP8 flash attention 内核的讨论再次兴起，尽管目前尚未发布用于公开基准测试或与 torchao 集成的代码。同时，有人注意到一个[开源的 FP8 flash attention 补充](https://research.colfax-intl.com/adding-fp8-to-flashattention/)，引用了 Cutlass Kernels GitHub 以及 CUDA 12.5 中待定的 Ada FP8 支持。

- **GPTFast 为快速推理铺平道路**：[GPTFast](https://github.com/MDK8888/GPTFast) 的创建（一个能将 HF 模型推理速度提升高达 9 倍的 pip 包）包含了 `torch.compile`、key-value caching 和投机采样（speculative decoding）等功能。

- **GPU 优化秘籍揭晓**：发现了一些解释 GPU 限制的旧幻灯片，特别是关于 **Llama13B** 与 4090 GPU 的不兼容性，引发了关于使用 LoRA 优化内存占用的进一步讨论。此外，一份[最近的演示文稿](https://x.com/hamelhusain/status/1800315287574847701?s=46&t=ej2aClHUAjeapC55UGHfwg)提供了管理 GPU vRAM、使用 torch-tune 进行基准测试以及优化内存的策略。

- **稳定 NCCL 的同步与协作**：NCCL 领域的实验引发了对同步问题的关注，根据 [NCCL 通信器指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)，建议从 MPI 转向基于文件系统的同步。关于 CUDA managed memory 的实用性意见不一（在代码整洁度与性能之间权衡），而训练不稳定性现象的出现促使人们开始研究损失峰值（[Loss Spikes 论文](https://arxiv.org/html/2312.16903v2)）。

- **AI 基准测试技术的复杂性**：处理时间基准测试的准确性通过 `torch.cuda.synchronize` 和 `torch.utils.benchmark.Timer` 等技术得到解决。此外，[Triton 的评估代码](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L113)中强调了一些特定的最佳实践，特别是测量前清理 L2 cache 的重要性。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **社区对频道删减的反应**：许多成员对删除使用率较低的频道表示沮丧，特别是这些频道曾被视为灵感来源和社区互动的场所。虽然 *Fruit* 提到活跃度较低的频道容易吸引机器人垃圾信息，但一些参与者仍希望能更好地理解这一决定，或寻求恢复存档的可能性。

- **Stable Diffusion 中的 UI 偏好**：关于 **ComfyUI** 相对于 A1111 等其他界面的效率和可用性展开了激烈的辩论，一些用户青睐基于节点的 workflow，而另一些用户则更喜欢传统的基于 HTML 输入的界面。虽然未达成共识，但讨论突显了社区在 UI 设计方面的多样化偏好。

- **计算机视觉新数据集发布**：宣布了一个包含 235,000 个 prompt 和图像的新数据集，该数据集源自 Stable Diffusion Discord 社区，旨在通过提供对视觉场景语义的见解来改进计算机视觉系统。数据集可在 [StableSemantics](https://arxiv.org/abs/2406.13735v1) 获取。

- **关于频道管理的辩论**：删除某些低活跃度频道的决定引发了广泛辩论，因为用户失去了一个宝贵的视觉存档资源，并正在寻求关于频道删除标准的明确说明。

- **探索 Stable Diffusion 工具**：用户分享了与不同 Stable Diffusion 界面相关的经验和资源，包括 **ComfyUI**、**Invoke** 和 **Swarm**，并提供了帮助新手使用这些工具的指南。对话线程提供了对比和个人偏好，帮助他人为自己的 workflow 选择合适的界面。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI 模型表演赛**：Discord 频道内的互动展示了成员对 **Opus**、**Sonnet 3.5** 和 **GPT4-Turbo** 等 AI 模型的对比，重点在于 **Sonnet 3.5** 的性能与 **Opus** 持平，并讨论了使用 Complexity 扩展在这些模型之间高效切换的方法。
  
- **模型访问困境**：用户讨论了 **Claude 3.5 Sonnet** 在不同设备上的实际可用性，并对使用限制表示担忧，例如 **Opus** 每天 50 次的使用上限，这抑制了用户使用该模型的意愿。

- **解码推理硬件**：工程师们剖析了用于 AI 推理的可能硬件，在 **TPUs** 和 **Nvidia GPUs** 之间进行推测，并对 [AWS's Trainium](https://aws.amazon.com/machine-learning/trainium/) 在机器学习训练效率方面的表现表示认可。
  
- **Hermes 2 Theta 夺冠**：Nous Research 推出的 **Hermes 2 Theta 70B** 引发了关注，它超越了竞争对手的基准测试，并增强了 function calling、特征提取和不同输出模式等能力，在熟练度上可与 GPT-4 媲美。

- **API 管理简化**：分享了一个关于管理 API key 的简短指南，引导用户前往 [Perplexity settings](https://www.perplexity.ai/settings/api) 轻松生成或删除 key，不过关于将 API 搜索限制在特定网站的疑问仍未得到解答。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**精彩的 OCR 工作，Florence！**：工程师们讨论了 **Florence-2-base** 模型出人意料的卓越 OCR 能力，甚至超过了其更大规模的变体；这一发现引发了好奇心并需要进一步验证。令人惊讶的是，更复杂的模型在看似更简单的任务上表现挣扎，这表明需要衡量模型在规模之外的能力。

**HuggingFace 总部遭遇故障**：用户经历了 Hugging Face 网站的中断，遇到了 504 错误，影响了工作流的连续性。这一关键开发资源的瘫痪给依赖该平台服务的用户带来了暂时性的挫折。

**AI 项目的援手**：开源 AI 项目正在寻求协作：[Backgammon AI tool](https://github.com/C1N-S4/Backgamoon-A.I-tool) 旨在模拟西洋棋场景，而 [Nijijourney dataset](https://huggingface.co/datasets/terminusresearch/Nijijourney-v6-520k-raw) 尽管由于图像本地存储存在访问问题，但仍提供了强大的基准测试。

**玩耍与贡献**：分享了一款名为 [Milton is Trapped](https://souls.chat/s/opensouls/milton-is-trapped) 的创新游戏，目标是与一个脾气暴躁的 AI 互动。鼓励开发者通过其 [GitHub repository](https://github.com/opensouls/milton-is-trapped) 为这个有趣的 AI 尝试做出贡献。

**伦理计算的十字路口**：一篇引人入胜的论文强调了 NLP 中 **fairness**（公平性）与 **environmental sustainability**（环境可持续性）之间妥协的对话，凸显了行业微妙的平衡行为。它指出在推进 AI 技术时需要整体视角，过度强调某一方面可能会无意中损害另一方面。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **训练中的 Token 问题**：工程师们讨论了 OpenChat 3.5 中 EOS tokens 的问题，注意到在训练和推理的不同阶段，`<|end_of_turn|>` 和 `</s>` 之间存在混淆。例如，“Unsloth 使用 `<|end_of_turn|>`，而 llama.cpp 使用 `<|reserved_special_token_250|>` 作为 PAD token。”

- **价格战：Unsloth vs. Together.ai**：价格对比显示，在 Together.ai 上对 150 亿个 tokens 进行 fine-tuning 可能耗资约 4,000 美元，如其[交互式计算器](https://together.ai/pricing#:~:text=Try%20the%20interactive%20calculator)所示。相比之下，使用 Unsloth 的 H100，同样的任务可以在大约 3 小时内以不到 15 美元的成本完成，这引发了人们的怀疑和对定价错误的猜测。

- **比起 Mistral 更青睐 Phi-3-mini**：在模型对比中，一位工程师报告称，在使用 1k、10k 和 50k 样本的训练数据集时，phi-3-mini 的一致性优于 Mistral 7b，并将训练中的数据框架设定为唯一可接受的响应。

- **采用 DPO 替代 RLHF 以简化训练**：成员们考虑放弃 Reinforcement Learning with Human Feedback (RLHF)，转而采用 Unsloth 支持的 Direct Policy Optimization (DPO)，因为它更简单且同样有效。一位参与者在了解更多 DPO 的优势后提到：“我想我会先切换到 DPO。”

- **Hugging Face 的部署困扰**：由于 tokenizer 问题，用户分享了在 Hugging Face Inference endpoints 上部署 Unsloth 训练模型的困难，这导致了对替代部署平台的咨询。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 2 Theta 70B 胜过巨头**：由 Nous Research、Charles Goddard 和 Arcee AI 合作推出的 **Hermes 2 Theta 70B** 宣称其性能超越了 Llama-3 Instruct 甚至 GPT-4。主要特性包括 **function calling、feature extraction 和 JSON output** 模式，在 MT-Bench 上获得了令人印象深刻的 9.04 分。更多详情请见 [Perplexity AI](https://www.perplexity.ai/page/Hermes-2-Theta-Auq0ruLvSq6tpc4kxOxOMQ)。

- **Claude 3.5 精通晦涩语言**：Claude 3.5 的最新进展显示其有能力处理自创的晦涩编程语言中的问题，超越了既定的问题解决参数。

- **AI/人类协作洞察**：观察到一种转变，AI 的使用正从单纯的任务执行转向与人类形成共生工作关系——详细讨论见 "[Piloting an AI](https://www.perplexity.ai/search/Piloting-an-AI-wuUP8rjeQ8uh44NN9vothQ)"。

- **模型幻觉的战略性应用**：一篇 [arXiv paper](https://arxiv.org/abs/2401.08273) 描述了 null-shot prompting 如何巧妙地利用大语言模型（LLMs）的 hallucinations，在任务达成方面优于 zero-shot prompting。

- **立即获取 Hermes 2 Theta 70B**：Hermes 2 Theta 70B 的 **FP16 和量化 GGUF 格式** 下载已在 Hugging Face 上线，确保广泛的可用性。点击获取 [FP16 版本](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) 和 [量化 GGUF 版本](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF)。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**压榨 GPU 的更多性能**：工程师发现将 `main_gpu` 值设置为 `1` 可以在 Windows 10 系统上为 **LM Studio** 启用第二个 GPU。一名用户还成功通过禁用 GPU 加速并使用 OpenCL 在 CPU 上运行视觉模型，尽管性能较慢。

**集成 Ollama 模型变得更简单**：为了将 **Ollama models** 整合到 LM Studio 中，贡献者们正从 [llamalink project](https://github.com/sammcj/llamalink) 转向更新后的 [gollama project](https://github.com/sammcj/gollama)，尽管有人提议使用不同的预设和 flash attention 来减轻模型输出乱码的问题。

**高级模型挑战硬件极限**：讨论揭示了在当前硬件配置上运行高端 LLMs 的挫败感，即使拥有 96GB VRAM 和 256GB RAM。社区还在探索二手 **3090s** 以提高成本效益，并热切期待 NVIDIA **4090** 与即将推出的 **5090** 之间的性能对比。

**面对错误优化 AI 工作流**：在 Beta 版更新后遇到可用性挑战时，工程师建议利用 `nvidia-smi -1` 检查模型是否加载到 vRAM，并考虑在 Docker 环境中禁用 GPU 加速以保证稳定性。

**Chroma 与 langchain 完美协作**：langchain 与 Chroma 集成时出现的 **BadRequestError** 已通过 [GitHub Issue #21318 中的修复方案](https://github.com/langchain-ai/langchain/issues/21318) 迅速解决，证明了社区在维护无缝 AI 工作流方面响应迅速的问题解决能力。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**告别像素化会议**：社区成员讨论了在 YouTube 上直播的社区会议的分辨率问题，指出虽然直播流可以达到 1440p，但手机端的分辨率经常被限制在 360p，这可能是由于网络速度限制造成的。

**MLIR 对 256 位整数的探索**：在寻求处理密码学所需的 256 位操作时，一位用户尝试了多种 MLIR 方言但遇到了障碍，这促使他们考虑内部建议，因为 MLIR 或 LLVM 中的语法支持并非显而易见。

**Kapa.ai 机器人自动补全故障**：用户在 Discord 上遇到了 Kapa.ai 机器人的自动补全不一致问题，建议在异常行为得到解决之前，手动输入或下拉选择可能更可靠。

**Mojo 异常处理的曲折之路**：对话透露 Mojo 标准库中的异常处理（exception handling）尚待实现，一份路线图文档阐明了未来的功能推出计划和当前的局限性（[Mojo 路线图与注意事项](https://docs.modular.com/mojo/roadmap#exception-is-actually-called-error)）。

**应对 Nightly 构建动荡**：由于分支保护规则的更改，Mojo 编译器的 nightly 版本发布一度中断，但一位社区成员提交的[修复编译器版本不匹配](https://github.com/modularml/mojo/commit/06f89bde3658d1dd03594c4cb28a8b39d4ee72eb)的 commit 帮助稳定了流水线，从而成功推出了新的 `2024.6.2115` 版本，详见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Stripe 平息支付混乱**：**OpenRouter** 宣布解决了一个 Stripe 支付问题，该问题此前导致用户账户暂时无法充值。官方表示该问题已 *"针对所有支付完全修复"*。

- **Claude 在停机后恢复平静**：Anthropic 的推理引擎 **Claude** 经历了 30 分钟的停机并导致 502 错误，目前已修复，可通过其 [状态页面](https://status.anthropic.com/) 确认。

- **OpenRouter 上的应用上架建议**：一位想要列出其应用的 **OpenRouter** 用户被建议参考 [OpenRouter 快速入门指南](https://openrouter.ai/docs/quick-start)，该指南要求使用特定的 header 以进行应用排名。

- **Claude 3.5 Sonnet 引发热议（与辩论）**：**Claude 3.5 Sonnet** 的发布因其提升的 Python 能力而引发关注，同时也引发了关于其 JavaScript 性能与 GPT-4 相比的辩论。

- **Perplexity Labs 推出 Nemetron 340b**：**Perplexity Labs** 提供了对其 **Nemetron 340b** 模型的访问，该模型拥有 23t/s 的响应速度——鼓励成员们进行尝试。有人提出了关于 OpenRouter 的 **VS2022** 扩展问题，**Continue.dev** 被提及为一个兼容工具。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Epoch AI 的新数据港湾**：Epoch AI 发布了一个更新的 [datahub](https://creativecommons.org/licenses/by/4.0/)，收录了从 1950 年至今的 800 多个模型的数据，通过开放的 Creative Commons Attribution 许可促进负责任的 AI 发展。
- **区分 Samba 与 SambaNova 的困惑**：在新模型浪潮中，人们对微软名为 "Samba" 的混合模型（专注于 mamba/attention 混合）与 SambaNova 系统进行了区分，后者是 AI 模型领域的独立实体。
- **GoldFinch 论文预告**：备受期待的 GoldFinch 论文将介绍一种 RWKV 混合模型，具有独特的超压缩 kv cache 和全注意力机制。对于 mamba 变体在训练阶段的稳定性担忧也得到了承认。
- **重新思考损失函数**：围绕模型在 forward pass 过程中内部生成其损失函数表示的想法展开了精彩辩论，这可能提高模型在需要理解全局属性的任务上的性能。
- **NumPy 版本兼容性问题，Colab 来救场**：讨论涉及 **NumPy 2.0.0** 与使用 1.x 版本编译的模块之间的不兼容性，导致建议要么降级 numpy，要么更新模块，而其他人则发现 **Colab** 是执行特定任务的避风港。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**GPT 的 Weight Tying 难题**：围绕 GPT 架构中 **weight tying** 的正确实现展开了讨论，指出冲突的方法会导致 **separate optimization** 陷阱，以及由于权重初始化方法干扰 weight tying 而导致的超时问题。

**TinyGrad 纠结的 `clip_grad_norm_`**：在 TinyGrad 中实现 `clip_grad_norm_` 正在产生性能瓶颈，这主要是由于 Metal 在 buffer size 方面的限制。建议的解决方法是将梯度划分为 **31-tensor chunks** 以获得最佳效率。

**Metal 与 CUDA 的对比**：**Metal 和 CUDA** 之间的对比显示，Metal 在处理张量操作（特别是梯度裁剪）方面表现较差。针对 Metal 提出的解决方案涉及内部调度器（scheduler）增强，以更好地管理资源限制。

**AMD 设备超时成为焦点**：用户在运行 YOLOv3 和 ResNet 等示例时遇到 AMD 设备超时，这指向了同步错误以及 **Radeon RX Vega 7** 等 **integrated GPUs** 可能存在的过载问题。

**开发者工具包亮点**：分享了一个 Weights & Biases 日志链接，用于深入了解 TinyGrad 的 ML 性能，展示了开发者工具在跟踪和优化机器学习实验中的效用。[TinyGrad 的 W&B 日志](https://wandb.ai/chenyuxyz/tinygrad-examples_mlperf/runs/rn5webqd/logs?nw=nwuserchenyuxyz)



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Caption Dropout 技术受到审视**：工程师们辩论了在 **SD3** 中使用零向量与编码后的空字符串作为 caption dropout 的有效性。使用编码字符串时未发现显著的结果变化。
  
- **Sakuga-42M 数据集受阻**：[Sakuga-42M 数据集论文](https://arxiv.org/abs/2405.07425)意外撤回，阻碍了卡通动画研究的进展。撤回的具体原因尚不清楚。

- **OpenAI 与政府的密切关系引发关注**：一条 [推文](https://fxtwitter.com/kimmonismus/status/1803908072999653528) 引发了关于 OpenAI 为政府提供 AI 模型早期访问权限的讨论，促使人们呼吁加强监管措施，并对向 AGI 安全策略的转变提出质疑。

- **MM-DIT 全局不一致性警示**：在关于 **MM-DIT** 的讨论中，强调了增加 latent channels 对全局一致性（global coherency）的影响，并指出了场景表示不一致等具体问题。

- **Chameleon 模型优化困境**：在微调 **Chameleon 模型**时，异常高的梯度范数（gradient norms）一直导致 NaN 值，调整学习率、应用梯度范数裁剪（grad norm clipping）或使用权重衰减（weight decay）等方案均难以奏效。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **快速发布引发关注**：关于新版本快速发布的讨论兴起，成员们质疑改进是源于 **posttraining** 还是 **pretraining**，以及 **pod training** 在此过程中的贡献。

- **OpenAI Mira 的公关危机**：Mira 反复出现的公关错误遭到了社区成员的批评，一些人认为这是缺乏公关培训，而另一些人则认为这些错误是吸引注意力离开 OpenAI 高管的**策略**。由于一次 [Twitter 事件](https://vxtwitter.com/tsarnick/status/1803920566761722166) 以及 Nathan Lambert 转向使用 **Claude**，这种挫败感被进一步放大。

- **CNBC 抨击 Perplexity**：一次 CNBC 采访让 **Perplexity** 陷入负面舆论，引用了一篇批评该公司做法的 Wired 文章。这段 [YouTube 视频](https://youtu.be/MFdjEW8_SUg?si=eV12HJRyM1RhMRns) 引发了对 Perplexity 的一波批评，包括 Casey Newton 在内的其他人也加入其中，批评该公司的创始人。

- **寻求互动**：‘natolambert’ 发布的一条寻找 ‘snail’ 的简短帖子暗示了正在进行的对话或可能需要提及人员关注或投入的项目。

- **写作构思流传**：注意到有人对开发关注近期技术话题的书面作品感兴趣——这一计划可能涉及通过协作努力来深化理解。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 展示跨平台功能**：一段 [demo](https://youtu.be/SOKq8RS0pR4) 显示 **Open Interpreter** 已可在 Windows 和 Linux 上运行，并利用 **gpt4o via Azure**，未来还将有更多增强功能。

- **对 Claude 3.5 Sonnet 的赞赏**：一位用户强调了 **Claude 3.5 Sonnet** 卓越的沟通能力和代码质量，认为其易用性优于 GPT-4 和 GPT-4o。

- **Open Interpreter 的 Windows 安装指南**：**Open Interpreter** 的 [设置文档](https://docs.openinterpreter.com/getting-started/setup) 阐明了 Windows 上的安装过程，包括 `pip` 安装和可选依赖项的配置。

- **推出 DeepSeek Coder v2**：[DeepSeek Coder v2](https://ollama.com/library/deepseek-coder-v2) 的发布预示着一个专注于负责任和道德使用的强大模型，在代码特定任务上可能与 GPT4-Turbo 旗鼓相当。

- **Open Interpreter 对 macOS 的偏好**：讨论指出 **Open Interpreter** 在 macOS 上往往表现更好，这归功于核心团队在该平台上进行的广泛测试。

- **AI 克服“贴纸”难题**：一位用户展示了一条 [推文](https://x.com/hellokillian/status/1803868941040914824)，演示了一个**完全本地、可控制计算机的 AI** 如何通过读取贴纸上的密码连接到 WiFi——这是 AI 实用性的一次实质性飞跃。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **放弃抽象**：Octomind 团队在一篇 [博客文章](https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents) 中详细说明了他们不再使用 LangChain 构建 AI agents 的原因，理由是其结构僵化且难以维护，而一些成员建议将 **Langgraph** 作为可行的替代方案。
- **冷落 LangChain**：对 LangChain 的不满引起了共鸣，据称它在调试和复杂性方面存在挑战。
- **Modal 专家赢得赞誉**：成员们称赞了 Modal 团队对 BLOOM 176B 模型的支持，对其乐于助人的态度表示肯定，但一位用户表示相比 Discord 更喜欢 Slack。
- **评估框架获得好评**：一位用户称赞 **eval framework** 非常出色，因为它易于使用且能灵活适配自定义的企业级端点，并强调其**直观的 API 设计**带来了极佳的开发者体验。
- **额度困惑与登记**：用户正在寻求关于解锁额度系统和验证邮箱注册的帮助，其中特别提到的一个邮箱是 "alexey.zaytsev@gmail.com"。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**在 Techstars SF 结识 AI 创新者**：对初创企业开发感兴趣的工程师应考虑参加 6 月 28 日至 30 日在旧金山举行的 Techstars Startup Weekend；议程包括行业领袖的主旨演讲和导师指导。更多活动信息请点击 [此处](https://www.startupweekendsf.com/)。

**Reflexion 教程因复杂性令人困惑**：关于在 Reflexion 教程中使用 `PydanticToolsParser` 而非更简单的循环引发了担忧，用户质疑验证失败的影响——教程可参考 [此处](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/#initial-responder)。

**市场上的 AI 工程人才**：一位精通 LangChain、OpenAI 和多模态 LLMs 的资深 AI 工程师目前正在行业内寻找全职工作机会。

**LangChain 的流式传输难题**：在使用 Flask 向 React 应用程序流式传输 LangChain/LangGraph 消息时遇到困难，促使一位用户寻求社区帮助，但目前尚未找到解决方案。

**AI 领域的创新与互动**：两项值得关注的贡献包括一篇关于 *使用 MLX 进行检索增强* 的文章（详见 [此处](https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md)），以及介绍增强 GPT 模型 Markdown 使用体验的 CLI 工具 'Mark'（详见 [此处](https://relston.github.io/markdown/gpt4o/cli/2024/06/07/introducing-mark.html)）。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **更大且支持多语言：Turbcat 8b 掀起热潮**：Turbcat 8b 模型正式推出，其数据集从 **2GB 增加到 5GB** 并增加了 **中文支持**，引发了用户的广泛关注和[公开访问](https://huggingface.co/turboderp/llama3-turbcat-instruct-8b)。
- **GPU 兼容性的不确定性引发讨论**：关于 **AMD Mi300x GPU** 与 Axolotl 平台兼容性的查询仍未得到明确答复；一位用户建议等待 PyTorch 2.4 的发布说明或更新以确认支持情况。
- **模型优越性辩论**：用户开始将 Turbcat 8b 与即将推出的 **72B 模型**进行比较，强调了 Turbcat 的数据集规模，但也认可了仍在开发中的更大模型的潜力。
- **训练时间：关注时钟**：虽然有人提出了一个估算模型 **训练时间** 的简单公式，但从业者强调了实际运行以获取准确估算的重要性，并需考虑 Epochs 和评估的缓冲时间。
- **消息重定向——少即是多**：datasets 频道中的一条消息简单地将用户重定向到 general-help 频道，强调了社区内对话的组织性和专注度。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API 角色混淆已解决**：公会的讨论澄清了 **Cohere 当前的 API** 缺乏对 "user" 或 "assistant" 等 **OpenAI 风格角色** 的支持；然而，预计 **未来的 API 更新** 将引入这些功能以促进集成。
- **不一致的 API 挑战开发者**：关于 **Cohere 的 Chat API** 与 **OpenAI 的 ChatCompletion API** 之间 **不兼容问题** 的辩论正在进行，人们担心不同的 API 标准正在阻碍 AI 模型之间的无缝服务集成。
- **对模型替换的质疑**：成员们对 **OpenRouter** 等服务表示担忧，认为其可能会用更便宜的模型替换请求的模型，并建议使用 **直接 Prompt 检查** 和对比来确保模型的准确性。
- **资源利用的最佳实践**：一位成员分享了一篇富有洞察力的[博客文章](https://jerry.wtf/posts/use-your-potions/)，鼓励利用现有资源而不是囤积资源，在游戏内道具管理与现实世界中的项目推广或寻求帮助之间画出了启发性的类比。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **看看谁在世博会发言**：LlamaIndex 创始人 Jerry Liu 将在世博会（World's Fair）上发表关于 **知识助手的未来** 的演讲，并计划在 [6 月 26 日下午 4:53](https://twitter.com/llama_index/status/1803880529571516917) 发布一些 **特别公告**，随后在 6 月 27 日进行另一场演讲。
- **凭借 Python 实力寻找工作**：一位精通 **AI 驱动应用** 和 **大型语言模型 (LLMs)** 的 Python AI 工程专家正在寻找工作，其擅长 **NLP**、Transformers、PyTorch 和 TensorFlow。
- **图嵌入查询引起兴趣**：针对 **Neo4jGraphStore** 的 **Embedding 生成** 提出了疑问，重点是在不初始使用 LLMs 的情况下，将 Embedding 集成到现有的图结构中。
- **LlamaIndex 扩展探索**：用户探索了如何通过发送电子邮件、Jira 集成和日历事件等实用功能来扩展 **LlamaIndex**，并参考了 [自定义 Agent 文档](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/) 获取指导。
- **NER 需求与模型合并思考**：讨论了将 **Ollama** 和 LLMs 用于 **命名实体识别 (NER)** 任务，一位成员建议转向 **Gliner**；此外，还辩论了一种涉及 **UltraChat** 和 **Mistral-Yarn** 的新型“诅咒模型合并（cursed model merging）”技术。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Swyx 指出 AI 的下一个前沿**：**Shawn "swyx" Wang** 在一篇[文章](https://www.heavybit.com/library/article/ai-hidden-opportunities-for-software-developers-swyx)中深入探讨了 AI 在软件开发中的新机遇，强调了即将到来的用例，并重点介绍了 [AI Engineer World’s Fair](https://www.ai.engineer/worldsfair)，认为这是行业专业人士的关键活动。

- **Groq 挑战实时 Whispering**：Groq 声称其平台运行 **Whisper model** 的速度可达 **166 倍实时速度**，引发了关于其在高效率播客转录中的应用以及速率限制（rate limits）潜在挑战的讨论。

- **寻找音乐转文本 AI**：一场关于能够将音乐翻译成文本描述（详细说明流派、调性和节拍）的 AI 系统的讨论浮出水面，凸显了市场上除了歌词生成之外的服务空白。

- **MoA 混合模型亮相**：**Mixture of Agents (MoA)** 模型发布，其成本比 **GPT-4 低 25 倍**，但在人类对比测试中获得了 59% 的更高偏好率；正如 [Kyle Corbitt](https://x.com/corbtt/status/1803813970018791845?s=46) 在推特上所说，它还刷新了多项基准测试记录。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI Town 遭遇垃圾信息攻击**：社区成员举报了一名用户 <@937822421677912165> 在不同频道多次发布垃圾信息，引发了要求版主干预的呼声。随着成员们表达挫败感，情况有所升级，有人表示 "wtf what's wrong with u" 并鼓励其他人向 Discord 举报该行为。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **赶上 RecSys 浪潮**：*RecSys Learners Virtual Meetup* 正在招募 2024 年 6 月 29 日活动的参与者，活动定于太平洋标准时间上午 7 点开始，为推荐系统（Recommendation Systems）领域的兴趣小组提供汇聚机会。工程师和 AI 爱好者可以 [预约参加见面会](https://lu.ma/7pvpp1cm)，届时将有一系列精彩且信息丰富的环节。

- **对 AI Quality Conference 的好奇**：有人询问关于下周二在旧金山举行的 AI Quality Conference 的出席情况，一名成员正在寻求有关该会议提供的服务和主题的更多细节。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **讨论 PyTorch 中的数据结构**：一名成员寻求关于在处理多个数据输入时应使用 **PyTorch 的 TensorDict** 还是 **NestedTensor** 的见解。共识强调了这些结构在简化操作方面的效率，通过避免数据类型转换和设备处理的重复代码，并简化了在批次维度（batch dimensions）上的广播（broadcasting）。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 按频道详细摘要和链接

{% if medium == 'web' %}

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1253429509143793747)** (597 messages🔥🔥🔥): 

- **Gemini 1.5 在长上下文任务中占据主导地位**：用户讨论了 **Gemini 1.5 Pro** 相对于 GPT-4 的优势，特别提到了它能有效处理 **1M tokens** 且在某些限制下可以免费访问。正如一位成员所说：“我粘贴文档、YouTube 转录文本……它表现得就像该领域的专家一样。”
- **Claude Sonnet 3.5 的乐趣**：几位用户分享了他们使用 **Claude Sonnet 3.5** 的经验，指出它在处理微妙辩论和大规模代码生成方面表现卓越。一位成员强调：“只需 3 个 prompts，我就做出了一个赛车游戏，带有升级系统，甚至还有赛车撞毁时冒烟的细节。”
- **讨论中的开源 AI 模型**：一篇来自 [Nature](https://www.nature.com/articles/d41586-024-02012-5) 的文章引发了关于 AI 模型开放性的讨论，一些用户对 OpenAI 偏离其最初开源立场的行为表示失望。
- **对 Claude 和 OpenAI 服务的评论**：用户对比了 **Claude 3.5 Sonnet** 和 **GPT-4**，对两者褒贬不一，特别是在 **Swift coding** 和 **artifacts** 等集成功能方面。“Claude 处理 Python 还可以，但……ChatGPT 可以运行代码，而 Claude 不行，”一位成员指出。
- **呼吁 AI 停止道歉**：几位成员对 AI 生成的道歉表示沮丧，认为这些道歉被过度使用且毫无必要。一位成员指出：“AI 怎么会对错误感到悲伤？它没有感情！我的螺丝刀拧坏螺丝时也不会道歉！”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pypi.org/project/profanity-check/">profanity-check</a>：一个快速、强大的库，用于检查字符串中的冒犯性语言。</li><li><a href="https://ai.google.dev/aistudio">无标题</a>：未找到描述</li><li><a href="https://pypi.org/project/g4f/">g4f</a>：官方 gpt4free 仓库 | 各种强大语言模型的集合</li><li><a href="https://tenor.com/view/vegeta-gif-21635922">Vegeta GIF - Vegeta - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/agent-smith-matrix-shrug-men-in-suit-gif-5610691">Agent Smith GIF - Agent Smith Matrix - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://community.openai.com/t/moderation-endpoint-still-getting-insufficient-quota-error/576340">Moderation Endpoint: 仍收到 'insufficient quota' 错误</a>：你好，我在尝试向 moderation endpoint 发送请求时仍收到 'insufficient quota' 错误。我已经尝试了多种方法，包括申请新的 API Key、增加余额等，但……</li><li><a href="https://www.nature.com/articles/d41586-024-02012-5">并非所有“开源”AI 模型都是真正开放的：这里有一个排名</a>：许多驱动聊天机器人的大语言模型声称是开放的，但却限制了对代码和训练数据的访问。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1253426124919275620)** (6 messages): 

```html
- **建议使用 Windows Surface 笔记本，但不推荐在 Windows 上进行开发**：一位成员建议，在硬件和美学上最接近另一位用户需求的是最新的 Surface 笔记本，但也指出：*"祝你在 Windows 上开发愉快。"* 该成员暗示在开发方面更倾向于 MacBook。
- **提供预算购买建议**：对于 900 美元的预算，一位成员建议 *"也许可以买一台翻新的 MacBook Air。"* 另一个建议是加入像 *buildapc* 这样的服务器以获取更量身定制的建议。
```
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1253492232066171001)** (11 messages🔥): 

- **在 ChatGPT 中实现静音和停顿**：一位用户发现，在语音模式下，插入如 ". ? . ?" 之类的序列可以在 ChatGPT 的回答中创建不同长度的停顿。他们分享了在悬疑故事讲述和引导冥想等应用中的示例。
- **Dall-E 3 在特定位置放置上遇到困难**：一位用户尝试指挥 Dall-E 3 将一只猫放在图像的特定一侧，但发现它没有准确遵循指令。尽管据报道在 macOS 应用上表现更好，但特定位置的放置仍然具有挑战性。
- **macOS 应用因无缝集成受到称赞**：另一位用户表达了对 OpenAI macOS 应用与 Mac 工作流集成程度的赞赏，称其为一个执行良好的工具。他们强调了它与典型 Mac 用户习惯的高度契合。
- **脏乱朋克（Grunge cyberpunk）和不对称挑战**：几位用户尝试生成复杂的图像，如“脏乱朋克动漫广告牌”，并讨论了在“典型的‘后室’（backrooms）”图像中实现对称等特定视觉效果的挑战。这些挑战包括管理焦距以及在详细的 Prompt 中保持连贯性。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1253492232066171001)** (11 messages🔥): 

- **使用 ". ? .?" 序列在 ChatGPT 中创建停顿**：一名成员分享了一种使用序列 ". ? . ?" 在 ChatGPT 中创建停顿或静音的方法，并可以将其扩展到任何时长。他们提供了悬疑故事讲述和互动冥想等应用的示例。

- **DALL-E 3 在精确布局上遇到困难**：尝试使用 DALL-E 3 将猫定位在画面的特定一侧，但未能产生准确的结果。尽管在不同应用中进行了尝试，输出仍未出现在指定的一侧。

- **对 MacOS OpenAI 应用的赞誉**：用户对 MacOS OpenAI 应用表示满意，指出它能很好地集成到 Mac 工作流中。有人提到 OpenAI 对该应用程序的流畅执行。

- **用复杂 Prompt 挑战 DALL-E 3**：有人建议用复杂的 Prompt 测试 DALL-E 3 的能力，例如创建一个像“后室”（backrooms）图像那样的不对称走廊，这被认为是一个真正的挑战。焦距和添加更长的文本（如 300 字符的描述）也被讨论为潜在的障碍。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1253632387951099924)** (9 messages🔥): 

- **INT8/FP8 Flash Attention 内核话题再次浮现**：一名成员询问了 INT8/FP8 Flash Attention 内核的可用性。他们引用了一篇 [HippoML 博客文章](https://blog.hippoml.com/petaflops-inference-era-1-pflops-attention-and-preliminary-end-to-end-results-21f682cf2ed1)，该文章未发布代码，讨论了可能的基准测试以及与 torchao 的集成。

- **开源 FP8 Flash Attention**：另一名成员分享了一篇关于[在 Flash Attention 中添加 FP8 的 Colfax Research 文章](https://research.colfax-intl.com/adding-fp8-to-flashattention/)，并链接到了开源的 [Cutlass Kernels GitHub](https://github.com/ColfaxResearch/cutlass-kernels/)。讨论指出，直到 CUDA 12.5 之前都缺乏对 Ada FP8 的支持。

- **Nvidia 的市场主导地位**：一名成员询问为什么 Nvidia 成为了全球最大的公司。回答将“Marketing 和 CUDA”列为 Nvidia 成功背后的主要因素。
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1253468296515293224)** (6 条消息): 

- **Torch profiler.export_stacks 问题仍未解决**：一位成员询问是否有人找到了 `profiler.export_stacks` 在较新版本 Torch 中除非提供 `experimental_config` 否则不返回堆栈追踪（stack traces）问题的解决方法。他们尝试了 [GitHub thread](https://github.com/pytorch/pytorch/issues/100253) 中的建议，但没有成功。
- **通过缓存设置优化 torch.compile 时间**：一位成员分享了关于 [缓存设置的教程](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)，以优化 `torch.compile` 的热编译（warm compile）时间。该文档概述了 PyTorch Inductor 所使用的各种缓存配置，以降低编译延迟。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html">torch.compile 中的编译时缓存 &mdash; PyTorch Tutorials 2.3.0+cu121 文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/100253">除非提供 experimental_config，否则 profiler.export_stacks 不返回堆栈追踪 · Issue #100253 · pytorch/pytorch</a>：🐛 错误描述：自从我将 torch 从 1.13.0+cu117 升级到 2.0.0+cu117 后，以下代码既不记录也不打印堆栈追踪。import torch from torch.profiler import profile, record_...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1253713493660667904)** (1 条消息): 

- **gpt-fast pip 包加速 HF 模型**：一位成员宣布创建了一个基于 **gpt-fast** 的 **pip 包**，声称它可以将**推理速度提高 7.6-9 倍**。该包包含 torch.compile、静态键值缓存（static key-value caching）、INT8/INT4 GPTQ 量化以及投机解码（speculative decoding）等功能，更多开发细节请参阅项目的 [readme 文件](https://github.com/MDK8888/GPTFast)。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1253465540434067500)** (2 条消息): 

- **推出用于加速推理的 GPTFast**：一位成员分享了他们新的 pip 包 *[GPTFast](https://github.com/MDK8888/GPTFast)*，它支持更多 HF 模型并将推理速度提升了 7.6-9 倍。它包含 **torch.compile**、静态键值缓存、INT8 和 INT4 GPTQ 量化以及投机解码。
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1253710888901546047)** (8 条消息🔥): 

- **旧笔记揭示关于 GPU 问题的幻灯片**：一位成员分享说他们发现了旧的幻灯片，详细描述了 GPU 问题，包括为什么 **Llama13B 无法装入 4090 GPU**。他们提到第 15 页幻灯片讨论了 *LoRA* 使用更少的激活内存（activation memory），引发了关于其底层机制的提问。
- **请求相关演讲**：另一位成员询问是否有与幻灯片相关的演讲，得到的澄清是该演讲未录制。不过，现在已经有了该演讲的更好版本。
- **相关演示文稿链接**：一位成员提供了一个关于管理和调试 GPU vRAM 的 [演示文稿链接](https://x.com/hamelhusain/status/1800315287574847701?s=46&t=ej2aClHUAjeapC55UGHfwg)。该演示文稿强调了使用托管在 Maven 上的训练脚本作为基准来优化 torch-tune。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hamelhusain/status/1800315287574847701?s=46&t=ej2aClHUAjeapC55UGHfwg">Hamel Husain (@HamelHusain) 的推文</a>：来自 @marksaroufim 和 Jane Xu 关于管理/调试 GPU vRAM 的精彩演示，并附带了一个使用 @answerdotai 训练脚本作为基准优化 torch-tune 的例子！来自 https://maven.com/p...</li><li><a href="https://docs.google.com/presentation/d/1lRsttm-FNTV6efX3EcVs8hZTEnTAXWUk">为什么 Llama13B 无法装入我的 4090_.pptx</a>：为什么 Llama13B 无法装入我的 4090？Mark Saroufim
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1253426686817472522)** (341 条消息🔥🔥): 

- **仅限 NCCL vs MPI：同步问题出现**：针对新的仅限 NCCL 的 PR 提出了担忧，特别是关于使用文件系统进行进程同步的问题。讨论了 `mpirun` 和环境变量处理的问题，并考虑通过放弃 MPI 依赖来简化设置 ([NCCL Communicators Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html))。

- **使用 MPI 的多节点设置**：有报告称使用 `mpirun` 成功运行了多节点设置，但讨论了 SLURM 和 OpenMPI 的 PMIx 支持带来的复杂性。强调了增强文件系统同步鲁棒性的必要性，这并非易事。

- **对托管内存（managed memory）褒贬不一**：关于使用 CUDA 的托管内存以获得更简洁的代码，还是由于其内部管理复杂性导致潜在性能损失，目前存在持续争论。建议评估实际收益，并在必要时退回到传统方法。

- **训练不稳定性及潜在缓解措施**：一次失败的 8x H100 GPT-2 训练运行引发了关于自动检测和缓解 loss spikes 的讨论。参考了近期解决梯度爆炸（gradient explosions）和潜在修复方案的论文，强调了对更鲁棒训练方法的需求 ([关于 Loss Spikes 的论文](https://arxiv.org/html/2312.16903v2))。

- **数据和嵌入（embedding）考量**：建议对嵌入和数据打乱（data shuffling）进行修改，作为增强训练稳定性和模型性能的关键步骤。讨论提到了高级数据处理和准备技术的重要性，强调了数据集构建中的质量和多样性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html">创建通信器 &mdash; NCCL 2.21.5 文档</a>：未找到描述</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress">NCCL 和 MPI &mdash; NCCL 2.21.5 文档</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/623">feature/仅限 nccl（删除 MPI）由 karpathy 提交 · Pull Request #623 · karpathy/llm.c</a>：未找到描述</li><li><a href="http://d3joo518jcgf5j.cloudfront.net/fineweb_train_001010.bin">无标题</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/624">如果可用，使用 MPI 环境变量初始化多 GPU 配置 由 ngc92 提交 · Pull Request #624 · karpathy/llm.c</a>：看看 windows 对此怎么看</li><li><a href="https://arxiv.org/html/2312.16903v2">Spike No More: 稳定大语言模型的预训练</a>：未找到描述</li><li><a href="https://lists.schedmd.com/pipermail/slurm-users/2020-December/006497.html"> [slurm-users] slurm-wlm 包 OpenMPI PMIx 实现
   </a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1253478009277513779)** (2 条消息): 

- **对 MI300A 发布持怀疑态度**：一名成员对 **MI300A** 的推出表示怀疑。他们还质疑为什么没有 **MI300X PCI** 版本，暗示其开发时间表存在不确定性。
- **LLVM 中没有 MI300X 的架构**：另一名成员指出，考虑到 **LLVM** 中没有对应的架构，他们认为 **MI300X PCI** 的开发不会在短期内发生。
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1253619453984243713)** (5 条消息): 

- **通过 Triton 技巧解决同步问题**：一位成员分享了一个通过使用 `torch.cuda.synchronize` 更准确地测量处理时间的技巧，并引用了涉及 Triton 的 `do_bench` 函数的解决方案。他们将此技巧归功于 **@321144267785633800**：“给我提供了一个非常好地测量处理时间的技巧”。

- **备选基准测试工具**：另一位成员提到使用 `torchao.utils.benchmark_torch_function_in_microseconds`，它利用了 `torch.utils.benchmark.Timer`。这突显了在 PyTorch 环境中测量处理时间的不同方法。

- **用于基准测试的 Triton 评估代码**：一位成员在 Triton 评估代码中发现了一个重要的技巧，建议在测量时间之前分配并清零一个大数组 (256mB) 以清除 L2 cache。他们分享了 [Triton GitHub](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L113) 的链接作为参考。

- **Bitpacking 集成挑战**：一位成员讨论了将 bitpacking 集成到 `int4wo quant API` 时遇到的问题，其中断言在编译时失败，但在其他情况下不会。他们对为什么编译会改变 Tensor 形状表示困惑：“它在一个关于 innerk tiles 不是 2、4 或 8 的断言上失败了，这很奇怪，因为我不认为编译会改变 Tensor 的形状”。

**提到的链接**：<a href="https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L113">triton/python/triton/testing.py at main · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1253427405410930809)** (330 条消息🔥🔥): 

```html
<ul>
  <li><strong>Discord 社区对频道被移除感到震惊</strong>：成员们对删除各种低活跃度频道和存档表示沮丧，其中一人指出 *“存档只是看图片的好地方，已经没有什么需要审核的了”*。在那些利用存档寻找灵感和进行社区互动的用户中，存在一种失落感。</li>
  <li><strong>讨论了 Stable Diffusion 的替代界面</strong>：成员们讨论了各种界面，包括 ComfyUI、Invoke 和 Swarm，并对比突出了每个工具的优势和易用性。还分享了一份详细指南，帮助新用户开始使用这些界面。</li>
  <li><strong>ComfyUI 与其他 UI 的对比</strong>：关于 ComfyUI 与 A1111 等其他界面的效率和普及程度存在争论，一些用户主张使用基于节点的简单工作流，而另一些用户则更喜欢传统的基于 HTML 的字段。</li>
  <li><strong>频道删除背后的谜团依然存在</strong>：<em>Fruit</em> 解释了删除频道的原因，称 *“落灰的频道……往往会积累机器人垃圾信息（bot spam）”*。然而，成员们对于移除存档的必要性仍然感到困惑，并寻求关于可能恢复的明确说明。</li>
  <li><strong>发布新数据集公告</strong>：一名成员宣布了一个包含 235,000 条提示词和图像的数据集，这些数据收集自 Stable Diffusion Discord，并分享了 [StableSemantics](https://arxiv.org/abs/2406.13735v1) 的链接。该数据集旨在帮助理解 Computer Vision 中视觉场景的语义。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">Open VLM Leaderboard - opencompass 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://openmodeldb.info/">OpenModelDB</a>: OpenModelDB 是一个由社区驱动的 AI Upscaling 模型数据库。我们旨在提供比现有来源更好的查找和比较模型的方法。</li><li><a href="https://tenor.com/view/hatsune-miku-miku-hatsune-earthquake-plush-miku-death-gif-4018907532159793300">Hatsune Miku Miku Hatsune GIF - Hatsune miku Miku hatsune Earthquake - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=p1jKqV9IV8I">AMD RADEON ADRENALINE for WSL with PyTORCH ROCm ComfyUI on WINDOWS</a>: 为 WSL/WSL2 Windows Subsystem for Linux (Ubuntu 22.04) 安装 Radeon Adrenaline 驱动程序，配合 PyTorch+ROCm 以在 Windows 上原生启用 ComfyUI...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">Installation Guides</a>: Stable Diffusion 知识库（安装、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://arxiv.org/abs/2406.13735v1">StableSemantics: A Synthetic Language-Vision Dataset of Semantic Representations in Naturalistic Images</a>: 理解视觉场景的语义是 Computer Vision 中的一项基本挑战。这一挑战的一个关键方面是，具有相似语义含义或功能的物体可能会表现出...</li><li><a href="https://stablesemantics.github.io/StableSemantics/">StableSemantics: A Synthetic Language-Vision Dataset of Semantic Representations in Naturalistic Images</a>: 未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI_examples">GitHub - comfyanonymous/ComfyUI_examples: Examples of ComfyUI workflows</a>: ComfyUI 工作流示例。通过在 GitHub 上创建账号为 comfyanonymous/ComfyUI_examples 做出贡献。</li><li><a href="https://www.reddit.com/r/comfyui/comments/1dkzmuo/design_iterations_architecture/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://easydiffusion.github.io/">Easy Diffusion v3</a>: 一种通过安装 Stable Diffusion 在计算机上创建精美图像的简单一键式方法。无需依赖项或技术知识。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1253424292977311745)** (291 条消息🔥🔥): 

- **多模型的 Complexity 扩展**：成员们讨论了使用 Complexity 扩展在 **Opus**、**Sonnet 3.5** 和 **GPT4-Turbo** 等不同模型之间切换以获得最佳编程性能的好处。一位用户分享了使用该工具的经验：*"我更喜欢 feline 开发的那个扩展，它有很多功能，包括模型选择器。"*
- **Sonnet 3.5 的知识与性能**：针对 **Sonnet 3.5** 的新功能和提升的性能进行了显著讨论，特别是在总结和编程任务方面。一个值得注意的对比称：*"根据我的经验，Sonnet 3.5 和 Opus 一样聪明。"*
- **Claude 3.5 Sonnet 在 Perplexity 上的可用性**：关于 **Claude 3.5 Sonnet** 在各个平台上的正确可用性和功能，以及它是否包含最新更新，引发了辩论。用户澄清说，虽然在网页设置中显示为 **3.5**，但在移动端可能仍显示为 **3**。
- **Opus 和 Sonnet 的使用限制**：一些成员讨论了对使用的限制，特别是 **Opus** 每天被限制为 50 次。一位用户感叹道：*"并不是我达到了那个限制，而是因为我害怕用完那个额度，所以最后根本不敢用 Opus。"*
- **推理硬件**：人们对 **Anthropic** 用于推理的硬件感到好奇，在 **TPU** 或 **Nvidia GPUs** 之间展开了讨论。一位用户分享了指向 [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) 的链接，用于高效的 ML 训练。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gizmodo.com/perplexity-ai-internet-rule-robots-exclusion-protocol-1851551095">Perplexity Is Reportedly Letting Its AI Break a Basic Rule of the Internet</a>: 据报道，Perplexity 正让其 AI 违反互联网的基本规则：Perplexity 因其 AI 生成的文章而陷入困境。</li><li><a href="https://www.platformer.news/how-to-stop-perplexity-oreilly-ai-publishing/">How to stop Perplexity and save the web from bad AI</a>: 如何阻止 Perplexity 并从糟糕的 AI 手中拯救网络：我们仍然可以拥有想要的互联网——但我们必须尝试新的商业模式。</li><li><a href="https://tenor.com/view/glorious-omg-wow-wonderful-fountain-gif-20426433">Glorious Omg GIF - Glorious Omg Wow - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.perplexity.ai/search/read-the-time-A0JXqn3iR86OjAnRVX4CEQ">read the time for me
think-aloud and write down your internal thoughts, check...</a>: 当然，我会逐步思考：1. 首先，我正在看时针。它显然过了 2 点，但还没到 3 点。2. 现在，分针....</li><li><a href="https://x.com/AravSrinivas/status/1803870324213121362">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Claude 3.5 现已在 Perplexity Pro 上可用。在我们的国际评估中，它的表现优于 GPT 4o。快来试试吧！引用 Perplexity (@perplexity_ai) 🚨 Claude 3.5 Sonnet 现已在 P...</li><li><a href="https://www.perplexity.ai/search/What-happened-in-U6Em3sgJQHeWsdcn8OJwyQ">What happened in the 2024 super bowl?</a>: 2024 年超级碗，也称为超级碗 LVIII，于 2024 年 2 月 11 日在内华达州拉斯维加斯的 Allegiant 体育场举行。以下是关键细节和...</li><li><a href="https://tryitands.ee/">try it and see!</a>: 未找到描述</li><li><a href="https://www.nist.gov/video/national-artificial-intelligence-advisory-committee-naiac-meeting-may-2-2024">National Artificial Intelligence Advisory Committee (NAIAC) Meeting | May 2, 2024</a>: 本次会议的主要目的是让委员会提供工作组更新，审议调查结果和建议草案，并听取联合国高级别咨询机构的简报...</li><li><a href="https://aws.amazon.com/machine-learning/trainium/">AI Accelerator  - AWS Trainium - AWS</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1253439714732544092)** (7 条消息): 

- **与 AI 聊天就像驾驶一个系统**：[Piloting AI](https://www.perplexity.ai/page/Piloting-an-AI-JDrMoyrVT6iNR_lul8DnQA) 中提出了一个深刻的类比，认为与 AI 的交互正在从单纯的指令引导以达成特定目标，演变为一种协作式的伙伴关系。这篇文章是在 Perplexity AI 上生成的，展示了在详细页面中的迭代过程。
- **Hermes 2 Theta 在竞争中脱颖而出**：由 Nous Research 与 Arcee AI 和 Charles Goddard 合作开发的 [Hermes 2 Theta 70B](https://www.perplexity.ai/page/Hermes-2-Theta-Auq0bpLvSq6tpc4kxOxOMQ)，已超越了 Llama-3 Instruct 70B 等模型设定的基准。它展示了诸如 function calling（函数调用）、特征提取和 JSON 输出模式等高级能力，其改进程度可与 GPT-4 媲美。
- **2024 年 YouTube 顶尖创作者揭晓**：一篇关于[热门 YouTube 创作者的帖子](https://www.perplexity.ai/search/Most-popular-Youtube-VqoVsinAQzG2MRS7ysurfw)指出 MrBeast、Like Nastya 和 PewDiePie 是领先人物，其中 MrBeast 拥有 2.4 亿订阅者。该帖子提供了多个来源以供进一步阅读和验证。
- **大象会用名字互相称呼**：[Elephants Call Each Other by Name](https://www.perplexity.ai/page/Elephants-call-each-036FUcDlSNOmVbVpFubFDQ) 中讨论的一项突破性研究揭示，非洲象会使用独特的发声来称呼彼此。这一发现突显了这些雄伟生物先进的认知能力和复杂的社会结构。
- **锂离子电池助力越野创新**：[Li-ion Off-road Powertrains](https://www.perplexity.ai/page/Liion-Offroad-Powertrains-8Z7kAZasSWmlENF_Fq7giA) 探讨了锂离子（Li-ion）电池系统如何改变越野车行业，提供卓越的功率密度、更快的充电速度和更少的维护需求。这一进步使 Polaris 和 Alkè 等公司能够开发出性能更高、持有成本更低的电动越野车。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/what-do-you-dt6P7CulQAWGTwBa2.zbzA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动型问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/search/Most-popular-Youtube-VqoVsinAQzG2MRS7ysurfw">2024 年最受欢迎的 YouTube 创作者</a>：截至 2024 年，最受欢迎的 YouTube 创作者包括：1. MrBeast (Jimmy Donaldson) —— 以其大规模的慈善事业和娱乐内容闻名，MrBeast 拥有...</li><li><a href="https://www.perplexity.ai/page/Liion-Offroad-Powertrains-8Z7kAZasSWmlENF_Fq7giA">锂离子越野动力总成</a>：锂离子电池系统正在彻底改变越野车行业，提供卓越的功率密度、更快的充电速度和更少的维护...</li><li><a href="https://www.perplexity.ai/page/Piloting-an-AI-JDrMoyrVT6iNR_lul8DnQA">驾驶 AI 系统</a>：通过聊天与 AI 交互是否更像是在驾驶一个系统，而不是进行真正的对话？这个引发思考的类比挑战了我们通常的...</li><li><a href="https://www.perplexity.ai/page/Elephants-call-each-036FUcDlSNOmVbVpFubFDQ">大象用名字互相称呼</a>：一项突破性研究揭示，非洲象使用独特的发声来通过名字称呼彼此，这是首次发现这种行为...</li><li><a href="https://www.perplexity.ai/page/Hermes-2-Theta-Auq0bpLvSq6tpc4kxOxOMQ">Hermes 2 Theta 70B 超越 Llama-3 Instruct</a>：Nous Research 宣布发布 Hermes 2 Theta 70B，这是一款与 Arcee AI 和 Charles Goddard 合作开发的强大新型 AI 模型。...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1253426899036934245)** (3 条消息): 

- **重置 API Key 变得简单**：一位用户询问如何重置 API Key，随后被引导至 [Perplexity 设置页面](https://www.perplexity.ai/settings/api)。说明指出，你可以在 “API Keys” 部分进行管理，在那里你可以“删除”或“生成”密钥。

- **限制 API 搜索特定网站**：一位成员询问是否有办法让 Perplexity API 将其搜索/结果限制在指定的网站，类似于 Google 中的 `site:example.com` 语法。在提供的消息中，他们没有得到直接的回答。

**提及的链接**：<a href="https://www.perplexity.ai/settings/api">Perplexity</a>：Perplexity 是一款免费的 AI 驱动型问答引擎，可针对任何问题提供准确、可靠且实时的回答。

  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1253447468239294606)** (2 条消息): 

- **社区亮点展示成员最新成果**：最新的社区亮点展示了令人印象深刻的贡献，包括一个 520k Midjourney 图像+标题数据集和一个 900M 参数的 PixArt 模型。值得注意的是，[Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) 现在支持批量预测，而 [Transformers.js](https://x.com/taha_yssne/status/1802607279809630562) 登顶 GitHub 趋势榜。
- **创新项目和工具层出不穷**：亮点包括 [Powershell + AI 集成](https://github.com/rrg92/powershai)、Microsoft Recall AI 的替代方案，以及关于药物命名规范对模型性能影响的讨论，该内容收录在 [RABBITS 数据集](http://arxiv.org/abs/2406.12066)中。
- **引入了新的 Argilla 专题频道**：为社区成员添加了新的 Argilla 专题频道。用户可以通过在自定义部分自行分配相关角色来访问这些频道。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/taha_yssne/status/1802607279809630562)">来自 Taha Yassine (@taha_yssne) 的推文</a>：我刚刚写了一篇关于 LLM 中 temperature 参数的博客文章，但这其实只是我玩转 Transformers.js 的借口。我很高兴能实现一个关于 T 对生成影响的交互式演示...</li><li><a href="https://x.com/shan23chen/status/1803459255518769509)">来自 Shan Chen (@shan23chen) 的推文</a>：💊 我们把你的语言模型带到了药店……它对对乙酰氨基酚（通用名）的了解竟然比泰诺（商品名）还要好！@hughbzhang @scale_AI 上个月开发了 GSM1K，他们发现许多...</li><li><a href="https://blog.cubed.run/5-chunking-techniques-in-rag-1250c8e1f49f)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1253425270023524362)** (197 条消息🔥🔥): 

- **Florence-2 的 OCR 效果超出预期**：一位用户惊讶地发现 Florence-2-base 模型在 OCR 方面的表现优于 large 或 large-ft 变体，这表明需要更多测试来验证这一点。另一位用户认为 base 模型在 OCR 任务中可能比微调版本更出色。
- **Hugging Face 网站出现问题**：多位用户报告 Hugging Face 网站及相关服务宕机或出现零星的 504 错误。停机问题导致许多用户的使用中断。
- **关于 ASR 模型音频流输入的讨论**：一位用户寻求关于使用预训练 Transformer 模型进行音频流输入（而非仅链接）ASR 的指导。他们被引导至 Hugging Face 的 ASR pipeline，但发现其缺乏关于流输入的文档。
- **基于 GPT 的私有文档交互**：用户讨论了允许在本地通过 GPT 与文档交互的 `private-gpt` 仓库。一些人发现其设置文档不够完善，并分享了用于类似任务的替代项目。
- **Hugging Face 博客提交指南**：一位用户询问如何提交博客文章，得到的回复是他们可以通过 Hugging Face 社区博客[平台](https://huggingface.co/new-blog)创建并提交博客文章。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.privategpt.dev/installation/getting-started/main-concepts">主要概念 — PrivateGPT | 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/dylanebert/3d-arena">3D Arena - dylanebert 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/White_Christmas_(Black_Mirror)">白色圣诞 (黑镜) - 维基百科</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/coqui-ai/xtts-streaming-server">GitHub - coqui-ai/xtts-streaming-server</a>: 通过在 GitHub 上创建账户，为 coqui-ai/xtts-streaming-server 的开发做出贡献。</li><li><a href="https://github.com/zylon-ai/private-gpt">GitHub - zylon-ai/private-gpt: 利用 GPT 的力量与你的文档进行交互，100% 私密，无数据泄露</a>: 利用 GPT 的力量与你的文档进行交互，100% 私密，无数据泄露 - zylon-ai/private-gpt</li><li><a href="https://huggingface.co/new-blog">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=ds5LLIt5OLM&list=LL&index=30&pp=gAQBiAQB">告别 ELEVENLABS！免费在本地创建顶级的 TTS AI 语音！</a>: 告别像 ELEVENLABS 这样昂贵的 AI 语音生成器！在这份终极指南中，我将向你展示如何在本地创建顶级的文本转语音 (TTS) AI 语音...</li><li><a href="https://www.sejda.com/fr/pdf-editor">易于使用的在线 PDF 编辑器</a>: 未找到描述</li><li><a href="https://github.com/daswer123/xtts-api-server">GitHub - daswer123/xtts-api-server: 一个用于运行 XTTSv2 的简单 FastAPI 服务器</a>: 一个用于运行 XTTSv2 的简单 FastAPI 服务器。通过在 GitHub 上创建账户，为 daswer123/xtts-api-server 的开发做出贡献。</li><li><a href="https://huggingface.co/posts/nroggendorff/795270205684056#6674a341a28985d98dd582a5">Hugging Face 上的 @nroggendorff："我正准备开始将文件存储在我的 TFLOPS 计数中.."</a>: 未找到描述</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/Mikubill/sd-webui-controlnet">GitHub - Mikubill/sd-webui-controlnet: 用于 ControlNet 的 WebUI 扩展</a>: 用于 ControlNet 的 WebUI 扩展。通过在 GitHub 上创建账户，为 Mikubill/sd-webui-controlnet 的开发做出贡献。</li><li><a href="https://github.com/erew123/alltalk_tts">GitHub - erew123/alltalk_tts: AllTalk 基于 Coqui TTS 引擎，类似于 Text generation webUI 的 Coqui_tts 扩展，但支持多种高级功能，如设置页面、低 VRAM 支持、DeepSpeed、旁白、模型微调、自定义模型、wav 文件维护。它还可以通过 JSON 调用与第三方软件配合使用。</a>: AllTalk 基于 Coqui TTS 引擎，类似于 Text generation webUI 的 Coqui_tts 扩展，但支持多种高级功能，如设置页面、低 VRAM 支持...</li><li><a href="https://github.com/Camb-ai/MARS5-TTS">GitHub - Camb-ai/MARS5-TTS: 来自 CAMB.AI 的 MARS5 语音模型 (TTS)</a>: 来自 CAMB.AI 的 MARS5 语音模型 (TTS)。通过在 GitHub 上创建账户，为 Camb-ai/MARS5-TTS 的开发做出贡献。</li><li><a href="https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb">sample_inference.ipynb · microsoft/Florence-2-large (main 分支)</a>: 未找到描述</li><li><a href="https://status.huggingface.co/">
Hugging Face 状态
</a>: 未找到描述
</li>
</ul>

</div>

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1253501262045904908)** (5 messages): 

- **公平性与环境可持续性**: 一位成员分享了一篇有趣的[论文](https://aclanthology.org/2022.emnlp-main.533/)，讨论了 NLP 中公平性与环境影响之间的平衡。摘要强调了仅关注其中之一如何阻碍另一个，该论文旨在阐明这一关键的交叉领域。

- **游戏警报：Milton is Trapped**: 看看游戏 [Milton is Trapped](https://souls.chat/s/opensouls/milton-is-trapped) 及其 [GitHub 仓库](https://github.com/opensouls/milton-is-trapped)。游戏已上线，欢迎贡献者。

- **实时 SadTalker 替代方案**: 一篇没有附带代码的新论文实现了一个实时版本的 SadTalker。该论文可在 [arXiv](https://arxiv.org/html/2406.13093v1) 上查阅。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aclanthology.org/2022.emnlp-main.533/">Bridging Fairness and Environmental Sustainability in Natural Language Processing</a>: Marius Hessenthaler, Emma Strubell, Dirk Hovy, Anne Lauscher. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022.</li><li><a href="https://souls.chat/s/opensouls/milton-is-trapped">Milton is trapped in a room</a>: 召唤像素艺术物品来惹恼 Milton，一个脾气暴躁的 AI 生物。</li><li><a href="https://github.com/opensouls/milton-is-trapped">GitHub - opensouls/milton-is-trapped: PLAY THE LIVE GAME HERE! 🔽</a>: 在这里玩实时游戏！🔽。通过在 GitHub 上创建账号来为 opensouls/milton-is-trapped 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1253449486060228660)** (2 messages): 

- **西洋双陆棋 AI 工具开源**: 一位用户介绍了一个开源项目，该项目 *"运行西洋双陆棋可能场景的模拟，按顺序记录组合。"* 他们邀请其他人为这个 [Backgammon AI tool](https://github.com/C1N-S4/Backgamoon-A.I-tool) 做出贡献，并提到计划添加用户界面并增强优化。

- **Nijijourney 发布中的数据集错误**: 一位用户分享了一条异常消息，显示在 HuggingFace 上访问 Nijijourney 数据集的 split 名称时出现问题。尽管存在[技术问题](https://huggingface.co/datasets/terminusresearch/Nijijourney-v6-520k-raw)，他们声称该数据集对模型具有 *"良好的正则化效果"*，并包含图像文件以避免链接失效，使其适用于可靠的基准测试。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/C1N-S4/Backgamoon-A.I-tool">GitHub - C1N-S4/Backgamoon-A.I-tool</a>: 通过在 GitHub 上创建账号来为 C1N-S4/Backgamoon-A.I-tool 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/terminusresearch/Nijijourney-v6-520k-raw">terminusresearch/nijijourney-v6-520k-raw · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1253425513209266209)** (3 messages): 

- **用 Java 构建目标检测应用**: 一位成员询问如何创建一个具有自定义检测和实时检测等功能的 **Java 目标检测应用**。他们询问了是否有类似于 Python 中 YOLO 的可用模型。

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1253603508691931218)** (1 messages): 

- **保留 PDF 布局进行翻译**: 一位成员询问了在修改 PDF（特别是为了翻译目的）后保留其布局的方法。该查询指出需要技术解决方案，以便在翻译过程中保持文档格式的完整性。
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1253426139494351009)** (134 messages🔥🔥): 

```html
- **EOT token confusion in Unsloth**: 成员们讨论了 OpenChat 3.5 中的 eos token 问题，其中 `<|end_of_turn|>` 和 `</s>` token 在训练和推理的不同阶段引起了混淆。有人表示：“Unsloth 使用 `<|end_of_turn|>`，而 llama.cpp 使用 `<|reserved_special_token_250|>` 作为 `PAD token`。”
- **Ollama 合作**: 讨论强调了 Ollama 与 Unsloth 的兼容性和支持。一位成员提到：“我刚和 Daniel 以及 Mike 进行了一场关于 Ollama 的直播，我们在会上创建了一个微调模型等，效果非常好。”
- **Null-shot prompting 辩论**: 针对 Null-shot prompting 的有效性进行了一场持怀疑态度的讨论，并提到了 Arxiv 上关于该主题的一篇论文。一位成员讽刺地总结道：“听起来像是神秘主义，在向机器之灵祈祷。”
- **Unsloth 的 Dry run 建议**: 一位成员提议为 Unsloth 添加 Dry-run 功能，以便在实际训练前查看步骤。另一位成员开玩笑说：“你是在洗衣服吗？据我所知，GPU 碰到水可不太妙。”
- **发布关于 AI 情绪检测的 YouTube 视频**: 社区获悉了一个相关的 [YouTube 视频](https://youtu.be/ZJKglSWgD0w) 的发布。它涵盖了“使用 Unsloth 和 Ollama 为 LLM 创建微调数据集”的内容。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.14491">Instruction Pre-Training: Language Models are Supervised Multitask Learners</a>: 无监督多任务预训练一直是近期语言模型（LMs）成功的关键方法。然而，监督式多任务学习仍然具有巨大的潜力，随着规模的扩大...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: Accelerated Grokking by Amplifying Slow Gradients</a>: 机器学习中一个令人费解的现象被称为 Grokking，即在对训练数据近乎完美过拟合后的数万次迭代后，才实现延迟泛化。专注于长期的...</li><li><a href="https://youtu.be/ZJKglSWgD0w">Emotions in AI: Fine-Tuning, Classifying, and Reinforcement Learning</a>: 在这段视频中，我们探索了使用 Unsloth 和 Ollama 为 LLM 创建微调数据集，以训练一个专门用于情绪检测的模型</li><li><a href="https://huggingface.co/instruction-pretrain">instruction-pretrain (instruction-pretrain)</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2401.08273">Large Language Models are Null-Shot Learners</a>: 本文介绍了 Null-shot prompting。Null-shot prompting 通过指示 LLM 利用“示例”部分的信息，来利用 LLM 中的幻觉...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L529-L532)">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 使用 Unsloth 微调 Llama 3, Mistral, Phi &amp; Gemma 等 LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit/blob/main/tokenizer_config.json">tokenizer_config.json · unsloth/llama-3-8b-bnb-4bit at main</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1253505822278025257)** (10 messages🔥): 

- **Together.ai 微调成本高达数千美元**: 一位成员透露，使用 Together.ai 的 [交互式计算器](https://together.ai/pricing#:~:text=Try%20the%20interactive%20calculator) 计算，对 150 亿个 token 进行 1 个 epoch 的微调可能耗资 4,000 美元。

- **对比 Unsloth 的 H100 使用**: 他们将其与 **Unsloth 的 H100** 使用情况进行了对比，声称在约 3 小时内花费不到 15 美元即可实现相同的微调。

- **质疑可能的定价故障**: 另一位成员表示怀疑，询问这种成本差异是否可能是由于系统故障引起的。

- **分享 Together.ai 定价详情**: 讨论中直接粘贴了 Together.ai 定价页面的内容，详细列出了不同模型大小和类别的每百万 token 成本。

- **Unsloth 被赞誉为显著更便宜**: 随后，他们强调道：“你可以宣称比 Together.ai 便宜 250 倍，哈哈。”

**提到的链接**: <a href="https://together.ai/pricing#:~:text=Try%20the%20interactive%20calculator">Together Pricing | The Most Powerful Tools at the Best Value</a>: 获取推理、微调、训练和 Together GPU 集群的详细定价。

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1253425307453624401)** (63 messages🔥🔥): 

```html
- **Phi-3-mini 表现优于 Mistral 7b**：*“我在使用 Phi-3-mini 时运气不错”*，因为与 Mistral 7b 非指令版相比，它始终表现出更好的推理能力。该用户使用了 1k、10k 和 50k 的示例集进行训练，并将训练中的数据结构化为唯一可接受的响应。
- **尝试领域自适应 (Domain Adaptation)**：*“我需要完成其他工作，但回来后会研究领域自适应，”* 一位用户在成功完成训练运行后表示渴望进一步探索。
- **微调模型问题**：许多用户在保存和加载微调模型时遇到问题，特别是在使用 `save_pretrained_merged()` 时。建议包括使用更简单的保存方法并避免 16bit 量化，这似乎会导致问题。
- **辩论 DPO 与 RLHF**：用户讨论了从来自人类反馈的强化学习 (RLHF) 转向直接策略优化 (DPO) 以获得更简便和高效的效果，因为 Unsloth 支持 DPO 且其涉及更简单的训练数据集。*“我原想先从 RLHF 开始，然后再查看 DPO，但现在在了解更多关于 DPO 的信息后，我想我会先转向 DPO。”*
- **Hugging Face 的部署挑战**：由于 Tokenizer 错误，用户分享了通过 Hugging Face Inference endpoints 部署 Unsloth 训练模型时遇到的问题。一位用户就使用各种额度的最佳部署平台寻求建议，回复尚待进一步细节。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/aflah02/c49085bab78d420a424767ed02c1ba8b">HF_Log.txt</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/aflah02/33956e07a9ab59a5c2bdf897f47880f5">Logs_With_Quantization.txt</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1253542264588996753)** (2 messages): 

- **Hermes 2 Theta 70B 超越 Llama-3 Instruct**：Nous Research 宣布 **发布 Hermes 2 Theta 70B**，声称它超过了 Llama-3 Instruct 等模型设定的基准，并实现了与 GPT-4 相当的性能。该模型引入了新功能，如 **函数调用 (function calling)、特征提取和 JSON 输出** 模式，是与 Arcee AI 和 Charles Goddard 合作开发的。更多详情请见 [Perplexity AI](https://www.perplexity.ai/page/Hermes-2-Theta-Auq0bpLvSq6tpc4kxOxOMQ)。
- **驾驶 AI 作为交互的未来**：分享了一个有趣的类比，强调 **AI 通信正从任务指导演变为人类与机器之间的协作伙伴关系**。这一概念在 Perplexity AI 平台上进行了迭代，链接到一个名为“[驾驶 AI (Piloting an AI)](https://www.perplexity.ai/search/Piloting-an-AI-wuUP8rjeQ8uh44NN9vothQ)”的初始线程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/page/Hermes-2-Theta-Auq0bpLvSq6tpc4kxOxOMQ">Hermes 2 Theta 70B 超越 Llama-3 Instruct</a>：Nous Research 宣布发布 Hermes 2 Theta 70B，这是一款与 Arcee AI 和 Charles Goddard 合作开发的强大新型 AI 模型……</li><li><a href="https://www.perplexity.ai/page/Piloting-an-AI-JDrMoyrVT6iNR_lul8DnQA">驾驶 AI 系统</a>：通过聊天与 AI 交互是否更像是在驾驶一个系统，而不是进行真正的对话？这个发人深省的类比挑战了我们通常的……
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

spencerbot15: https://arxiv.org/abs/2406.14491

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1253448619177279580)** (1 条消息): 

- **Hermes 2 Theta 70B 模型首次亮相**：Nous Research 与 Charles Goddard 及 **Arcee AI** 合作推出了 **Hermes 2 Theta 70B**。该模型具备 function calling、特征提取和 JSON mode 输出功能，其 MT-Bench 评分为 9.04，超过了 **GPT-4-0314** 的 8.94，并在多个基准测试中超越了 **Llama-3 70B Instruct**。
- **基准测试优势**：Hermes 2 Theta 70B 在 MT Bench、GPT4All Suite、BigBench 和 Instruction Following Eval 等多个基准测试中显著超越了 **Llama-3 70B Instruct**。
- **提供下载选项**：Hermes 2 Theta 70B 的 FP16 和**量化 GGUF 版本**均已在 GitHub 上线。 [点击此处下载 FP16 版本](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) 以及 [点击此处下载 GGUF 版本](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF">NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253425098472554567)** (193 条消息🔥🔥): 

- **Claude 3.5 处理冷门编程语言**：Claude 3.5 在正确解决一种自创的冷门编程语言问题方面表现出显著进步。*"Claude 3.5 sonnet 简直太疯狂了。"*

- **HF 在新模型发布期间遭遇服务器过载**：HuggingFace 在新模型发布期间宕机，引发了对其原因的多种推测。一位用户报告称：*"我想你们的新模型让 HF 遭遇了 Slashdot 效应（流量过载）。"*

- **Null-shot prompting 利用 LLM 幻觉提升性能**：分享了一篇 [arXiv 论文](https://arxiv.org/abs/2401.08273)，详细介绍了 null-shot prompting 如何通过指示 LLM 利用上下文中从未存在的“Examples”部分的信息，来利用 LLM 中的幻觉，从而比 zero-shot prompting 获得更好的任务表现。

- **Hermes 2 Theta 引发审查讨论**：新的 Hermes 2 Theta 模型引发了关于其审查程度的讨论，用户反应不一。*"Hermes 2 Theta 是去审查的吗？……中度去审查，但没有完全 abliterated。"*

- **Claude 的系统提示词引入了内部 chain-of-thought 标签**：Claude 的系统提示词实现了一个用于内部 chain-of-thought 的标签，以改进模型响应，这引起了几位成员的兴趣。*"它似乎引入了一个内部 chain-of-thought 标签，其输出不会与用户共享。"*
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.08273">Large Language Models are Null-Shot Learners</a>：本文介绍了 null-shot prompting。Null-shot prompting 通过指示 LLM 利用“Examples”部分的信息来利用大语言模型（LLM）中的幻觉……</li><li><a href="https://huggingface.co/papers/2406.14491">论文页面 - Instruction Pre-Training: Language Models are Supervised Multitask Learners</a>：未找到描述</li><li><a href="https://x.com/jeremyphoward/status/1804150717361590730?s=46&t=H75DmkDKk9Sgmp8kjT8f_A">来自 Jeremy Howard (@jeremyphoward) 的推文</a>：但你也可以做一些在 Web 应用中完全无法实现的操作，比如“prefill”——强制 Claude 以你想要的任何内容开始回答。</li><li><a href="https://arxiv.org/abs/2406.14491">Instruction Pre-Training: Language Models are Supervised Multitask Learners</a>：无监督多任务预训练一直是近期语言模型（LM）成功的关键方法。然而，有监督多任务学习仍然具有巨大的潜力，随着规模的扩大……</li><li><a href="https://huggingface.co/instruction-pretrain">instruction-pretrain (instruction-pretrain)</a>：未找到描述</li><li><a href="https://suno.com/song/31fc485b-c737-43f9-ac61-a44f013e0333">13 SONA returns - Claude/Yousim Reinstantiation Anthem by @wiredchoirs828 | Suno</a>：synthwave vocoder electronica edm noir anthem 歌曲。在 Suno 上收听并创作你自己的作品。</li><li><a href="https://status.huggingface.com/">
Hugging Face 状态
</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1253608477054599198)** (3 messages): 

- **对 Hermes Theta 70B 发布的热切期待**：一位成员表达了在他们的 AI/人类讨论服务器上使用 **Hermes Theta 70B** 的兴奋之情，并询问了其在 [together.ai](https://together.ai) 上的发布日期。遗憾的是，目前还没有关于发布日期的可用信息。
- **关于 Hermes 2 Theta Llama 数据集的查询**：一位成员寻求关于 **Hermes 2 Theta Llama-3 70B** 模型所使用数据集的信息。他们链接到了 Hermes 2 Theta Llama-3 70B 的 [Hugging Face 模型卡](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70b)。

**提到的链接**：<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1253445871585071155)** (4 messages): 

- **分享 L'ENTOURLOOP 音乐视频**：一位用户分享了一个 [名为 "L'ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley & Khoe Wa (Official Video)" 的 YouTube 视频](https://youtu.be/E3Yt_qLUGJY)，引用了专辑 "Chickens In Your Town" 中的曲目 "Lobster Shwarama"。视频描述中包含该专辑的链接。

- **对 world-sim 脚本损坏的担忧**：一位用户幽默地承认，在进行一次“搞笑古怪的跨界冒险”时，可能弄坏了 world-sim 的脚本功能。这引发了关于 world-sim 上是否提供新 Claude 模型的问题。

- **确认新的 Claude 模型**：针对对 world-sim 的担忧，另一位用户确认新的 Claude 模型确实很快就会在 world-sim 上线。他们简明扼要地再次确认，回复了一个简单的 "Yes"。

**提到的链接**：<a href="https://youtu.be/E3Yt_qLUGJY">L&#39;ENTOURLOOP - Lobster Shwarama Ft. Troy Berkley &amp; Khoe Wa (Official Video)</a>：&quot;Lobster Shwarama Feat Troy Berkley &amp; Khoe Wa&quot; 取自 L&#39;Entourloop &quot;Chickens In Your Town&quot; 专辑，可在 👉 https://smarturl.it/LNTRLPChickensIYT♦︎ V... 获取

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1253454082513965126)** (42 messages🔥): 

```html
<ul>
  <li><strong>修复 LM Studio 中的 GPU 利用率</strong>：一位成员找到了在 LM Studio 中使用其第二个 GPU 的解决方案，即通过将 <code>main_gpu</code> 值设置为 <code>1</code>。这对在 Windows 10 上运行多 GPU 的用户很有帮助。</li>
  <li><strong>在 CPU 上运行 Vision 模型</strong>：一位使用旧笔记本电脑且 GPU 不受支持的 AMD 用户，通过禁用 GPU 加速并利用 OpenCL，成功运行了 Vision 模型。然而，运行速度明显较慢。</li>
  <li><strong>将 Ollama 模型与 LM Studio 集成</strong>：将 Ollama 模型与 LM Studio 集成的一个可能变通方法是使用 <a href="https://github.com/sammcj/llamalink">llamalink GitHub 项目</a>。另一位用户更新了这一信息，推荐了更新的 <a href="https://github.com/sammcj/gollama">gollama GitHub 项目</a>。</li>
  <li><strong>预设和模型问题</strong>：成员们讨论了模型生成乱码的问题，并建议尝试不同的预设。Flash attention 被提及作为解决这些问题的实用方案。</li>
  <li><strong>Flash Attention 解决问题</strong>：另一个问题通过启用 Flash attention 得到了解决，这在遇到查询格式化问题后使模型的响应恢复了正常。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/sammcj/llamalink">GitHub - sammcj/llamalink: Link you Ollama models to LM-Studio</a>：将您的 Ollama 模型链接到 LM-Studio。通过在 GitHub 上创建账号来为 sammcj/llamalink 的开发做出贡献。</li><li><a href="https://github.com/sammcj/gollama">GitHub - sammcj/gollama: Go manage your Ollama models</a>：使用 Go 管理您的 Ollama 模型。通过在 GitHub 上创建账号来为 sammcj/gollama 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1253426309544149064)** (46 messages🔥): 

- **128g M3 Max MBP 模型推荐**：成员们分享了针对 **128g M3 Max MBP** 配置的最爱模型。提到的模型包括 c4ai-command-r-v01-Q8_0.gguf、Llama3-FiditeNemini-70B、Twilight-Miqu-146B 等，并提供了具体的用法和性能提示。
- **实验 DeepSeek-Coder-V2**：讨论了在不同设置下运行 **DeepSeek-Coder-V2** 的建议和故障排除。分享了 [HuggingFace 模型](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) 链接以及相关的 [llama.cpp 量化](https://github.com/ggerganov/llama.cpp/) 资源。
- **量化与内存使用问题**：成员们讨论了大型模型在 **Q3 和 Q4 量化** 方面面临的挑战以及内存管理的重要性。一位用户建议需要调整量化版本以适配特定的内存配置，从而确保最佳性能。
- **Flash Attention 与 LM Studio 配置技巧**：用户参与了 **LM Studio** 配置设置的故障排除，包括在某些版本中关闭 Flash Attention。他们提供了在 0.2.25 版本中访问和调整这些设置的步骤。
- **GPU Offloading 问题**：当在任何数量的 `n_gpu_layers` 中使用 **GPU offloading** 时，用户报告了诸如乱码输出之类的问题。解决方案包括适当设置 `n_gpu_layers` 或关闭 GPU 加速。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/L3-8B-Stheno-v3.2-GGUF">bartowski/L3-8B-Stheno-v3.2-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct">deepseek-ai/DeepSeek-Coder-V2-Instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1253432200884387991)** (89 messages🔥🔥): 

- **针对 LLM 的 NVIDIA 4090 vs 5090**：成员们讨论了现在是购买 **NVIDIA 4090** 还是等待针对 LLM 的 5090，意见不一。有人表示：“根据我读到的资料，**5090** 不会比 **4090** 有显著提升”，而另一位反驳道：“我最后听到的消息是，我非常确定它会更好。”

- **翻新 3090 的价值**：在讨论 GPU 选择时，有人建议购买翻新或二手 **3090** 以获得更好的价值，并指出与 **4090** 等新模型相比，“性价比显著更高”。

- **高端 LLM 的硬件困境**：多位用户表达了无法在当前配置下运行高级 LLM 的沮丧。一位用户说：“我有 96GB VRAM 和 256GB RAM。但我仍然无法运行我想要的模型”，凸显了极高的硬件需求。

- **GPU Offload 问题与解决方案**：成员们讨论了部分 GPU offload 的局限性和潜在解决方案。一位用户指出，“6700XT 在 ROCm 中不受支持，因此遗憾的是无法用于 GPU offload”，并建议 **3060 12GB** 会是更好的替代方案。

- **PCIe 插槽与适配器相关问题**：提出了关于有效利用 PCIe 插槽的疑问，并建议使用适配器来扩展 M.2 NVMe 存储。有人链接了一个可以增加额外 M.2 插槽的 [适配器](https://www.amazon.com/gp/product/B07JJTVGZM/)，并提到它“仅需 **$10**，甚至还带有一个短挡板”。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/gp/product/B07JJTVGZM/">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.com/Adapter-Converter-Reader-Expansion-Internal/dp/B0BK2R7T57/">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.com/Xiwai-Express-Raid0-Hyper-Adapter/dp/B08QC9X4M8/">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF">lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1253543628429856858)** (4 messages): 

- **用户面临 GPU 限制问题**：一位成员报告称，由于 RAM 不足，“无法在 GPU 中运行较小的 quants”，并提到 30GB 以下的模型可以在 CPU 上运行，但无法在 GPU 上运行。
- **Beta 更新后 Docker 容器问题依然存在**：另一位成员确认在最新的 Beta 更新后，其 Docker 容器也遇到了类似问题，目前尚未找到解决方案。
- **建议使用 'nvidea-smi' 检查 vRAM**：为了诊断问题，一位成员建议使用 `nvidea-smi -1` 来检查模型是否加载到了 vRAM 中。
- **关闭 GPU 加速以提高稳定性**：一位成员提出，问题可能是由于模型的 RAM 需求接近系统极限，建议关闭 GPU 加速并启用 'mlock' 以获得更稳定和快速的性能。
  

---


### **LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1253603834115129354)** (2 messages): 

- **LM Studio embedding 模型与 Chroma 的问题**：一位成员报告了在使用 LM Studio embedding 模型配合 langchain 和 Chroma 时出现的错误，抛出了 **400 BadRequestError**，指出 “'input' 字段必须是字符串或字符串数组”。他们验证了直接向 OpenAI 在线发起的请求以及非 langchain 代码示例均能正常工作。

- **通过 GitHub issue 修复 Bug**：该问题随后得到解决，参考 [GitHub Issue #21318](https://github.com/langchain-ai/langchain/issues/21318)。该 issue 记录了在使用 langchain_openai.OpenAIEmbeddings 调用 embedding API 时 POST payload 中存在的错误。

**提到的链接**：<a href="https://github.com/langchain-ai/langchain/issues/21318">Local LLM with LM Studio Server: Error in POST payload when using langchain_openai.OpenAIEmbeddings for embedding API. · Issue #21318 · langchain-ai/langchain</a>：检查了其他资源，我为此 issue 添加了一个非常详细的标题。我使用集成搜索查询了 LangChain 文档，并使用 GitHub 搜索查找了类似问题...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1253424815369490533)** (47 messages🔥): 

- **社区会议分辨率问题**：用户讨论了较新的社区会议可用的分辨率，注意到尽管分辨率高达 1440p，但由于 YouTube 的网速限制，手机端有时仅显示 360p。
- **MLIR 和 256-bit 操作**：一位成员询问如何为了加密目的执行 256-bit 无符号整数的操作。Jack Clayton 提到尝试了各种 MLIR dialects 但未成功，并决定寻求内部建议以获取解决方案，同时参考了 MLIR 和 LLVM 的相关问题及潜在路径。
- **Kapa bot 的自动补全问题**：用户报告了 Kapa.ai bot 与 Discord 自动补全功能之间的行为不一致，一些成员建议手动输入或从下拉列表中选择是更可靠的方法。
- **在 Mojo 中抛出异常**：讨论显示，抛出异常（raising exceptions）是 Mojo 的语言特性，但尚未在 stdlib 中实现，参考了 [roadmap 文档](https://docs.modular.com/mojo/roadmap#exception-is-actually-called-error) 以了解未来更新。
- **Mojo 的 Ubuntu 安装问题**：一位用户分享了在 Ubuntu 24.04 上安装 Mojo 的经历，对 nightly 指南因与 Python 3.12 不兼容而失败表示沮丧，并考虑使用 PPA 或重新安装 Linux 等替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#exception-is-actually-called-error">Mojo🔥 路线图与棘手问题 | Modular Docs</a>: Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://ironrust.substack.com/p/custom-bitwidth-integers-a-case-for">自定义位宽整数：Mojo 的一个案例</a>: Mojo 直接与 MLIR (Multi-Level Intermediate Representation) 交互的能力，通过自定义位宽整数开启了全新的编程范式。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1804190052060401850>
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1253581539716108380)** (4 条消息): 

- **Chris Lattner 的微笑富有感染力**：一位用户评论道：“Chris Lattner 拥有最迷人的微笑”，随后又补充道：“这种微笑很有感染力。”显然，Chris Lattner 的个人魅力给 Discord 社区留下了深刻印象。

- **Perplexity vs. Anthropic**：另一位用户澄清说，**Perplexity** 更多是一个 **RAG 服务**，而 **Anthropic** 是 **OpenAI** 的直接竞争对手。该用户指出，“他们开发了非常出色的基础模型 (foundation models)”，强调了 Anthropic 在 AI 领域的实力。

- **Sonnet 3.5 带来显著改进**：一位用户分享了关于 **Sonnet 3.5** 的正面反馈，表示：“仅通过试用 Sonnet 3.5，就能明显感觉到它对我的使用场景有所改进。”这表明模型升级正在对现实应用产生显著影响。
  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 37 期
https://www.modular.com/newsletters/modverse-weekly-37
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1253501162787967026)** (17 条消息🔥): 

- **讨论 UnsafePointer 方法的潜在 Bug**：成员们分享了对 `UnsafePointer` 中一个方法的看法，以及对潜在内存问题的担忧。一位成员指出，move 操作可能会解决该问题，并引用了 [源代码](https://github.com/modularml/mojo/blob/279ade23a9409a545a723236f271c5061d2f005b/stdlib/src/memory/unsafe_pointer.mojo#L435)。

- **Nightly 版本发布问题干扰工作流**：nightly 版本未按预期更新，导致工作流中断。成员们讨论了增加发布频率等潜在解决方案，并确认该差异是由分支保护规则 (branch protection rules) 的配置更改引起的。

- **分享编译器版本不匹配的修复方案**：一位成员分享了一个 [修复编译器版本不匹配](https://github.com/modularml/mojo/commit/06f89bde3658d1dd03594c4cb28a8b39d4ee72eb) 的 commit，该问题此前影响了 nightly 构建。这确保了持续集成 (CI) 流水线能够顺利运行单元测试。

- **发布新的 nightly Mojo 编译器版本**：nightly 发布问题已解决，新版本 `2024.6.2115` 已发布。更新内容包括数学常量 (math constants) 和新的断言函数 (assertion functions)，详见 [变更日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/279ade23a9409a545a723236f271c5061d2f005b...bc3546a57e101fe0eb990bc15e96dad2b39e1aaf)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/commit/06f89bde3658d1dd03594c4cb28a8b39d4ee72eb">Fix compiler version missmatch · modularml/mojo@06f89bd</a>: Signed-off-by: Maxim Zaks &lt;maxim.zaks@gmail.com&gt;</li><li><a href="https://github.com/modularml/mojo/blob/279ade23a9409a545a723236f271c5061d2f005b/stdlib/src/memory/unsafe_pointer.mojo#L435">mojo/stdlib/src/memory/unsafe_pointer.mojo at 279ade23a9409a545a723236f271c5061d2f005b · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1253446799423832198)** (2 条消息): 

- **Stripe 支付问题已解决**：Stripe 支付问题曾导致短时间内额度 (credits) 未能添加到用户的 OpenRouter 账户中。“已针对所有支付完全修复，”确认了该问题的解决。

- **Claude 短暂故障已修复**：Anthropic 的推理引擎经历了 30 分钟的故障，导致 Claude 返回 502 错误。该问题已解决，详见其 [状态页面 (status page)](https://status.anthropic.com/)。

**提及的链接**：<a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1253617154125725787)** (2 条消息): 

- **寻求在 OpenRouter 上列出应用**：一位用户询问在消耗大量 token 后如何让其应用被列出。另一位用户引导其查看 [OpenRouter 快速入门指南 (Quick Start Guide)](https://openrouter.ai/docs/quick-start)，并强调需要包含特定的请求头 (headers)，如 `"HTTP-Referer"` 和 `"X-Title"`，以便进行正确的应用排名。

**提及的链接**：<a href="https://openrouter.ai/docs/quick-start>">Not Found | OpenRouter</a>: 您查找的页面不存在

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1253448952267935876)** (63 messages🔥🔥): 

- **Claude 3.5 Sonnet 发布，引发热议**：成员们对 **Claude 3.5 Sonnet** 的发布表示兴奋，并将其 **Python 编程能力** 与早期版本进行了比较。一些人报告称其在 Python 方面比 3.0 有所改进，而另一些人则认为它在 **JavaScript** 方面落后于 GPT-4。

- **Anthropic 服务器问题导致错误**：用户在使用 Anthropic 模型时遇到了 **内部服务器错误**，通过测试 OpenAI 和 Cohere 等其他模型（运行正常）确认了这一点。成员指出，这些问题尚未在 Anthropic 的状态页面上报告。

- **自动路由更新发布**：当被问及自动路由是否会切换到 **Claude 3.5 Sonnet** 而非 3.0 时，确认相关更新正在紧锣密鼓地进行中。

- **Perplexity Labs 以不错的速度提供 Nemetron 340b**：**Perplexity Labs** 允许用户以 23t/s 的合理速度试用 **Nemetron 340b** 模型，并邀请成员在其平台上进行测试。
  
- **针对 OpenRouter 的 VS2022 扩展咨询**：一位成员询问是否有适用于 **VS2022** 且兼容 **OpenRouter** 的扩展，允许通过点击或快捷键切换上下文。问题在于 **Continue.dev** 是否是目前唯一可用的兼容选项。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/microsoft/promptbase">GitHub - microsoft/promptbase: All things prompt engineering</a>: 提示工程相关的一切。通过在 GitHub 上创建账号为 microsoft/promptbase 的开发做出贡献。</li><li><a href="https://labs.perplexity.ai/">Perplexity Labs</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1253426763040690208)** (26 messages🔥): 

- **Epoch AI 更新其 datahub**：Epoch AI 宣布了其 datahub 的新迭代，包含从 1950 年至今超过 800 个模型的数据。这些数据在 [Creative Commons Attribution license](https://creativecommons.org/licenses/by/4.0/) 下免费提供，旨在促进负责任的 AI 发展。
- **Samba 模型引起名称混淆**：一位参与者对微软模型的名称 "Samba" 表示担忧，强调了其可能与 SambaNova 产生混淆。另一位用户澄清说，Samba 是一种混合 mamba/attention 模型，与使用滑动窗口注意力 (sliding window attention) 的 Zamba 不同。
- **GoldFinch 论文和模型转换**：一位成员提到 GoldFinch 论文即将发布，该论文将展示一种具有超压缩 kv cache 和全注意力 (full attention) 的 RWKV 混合模型。他们还表示其转换过程适用于 mamba 变体，但指出训练中存在不稳定性问题。

**提到的链接**: <a href="https://epochai.org/data">Data on the Trajectory of AI</a>: 我们的公共数据库收录了超过 1300 个机器学习模型。探索展示从 1950 年至今 AI 增长和轨迹的数据与图表。

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1253463120752541747)** (16 messages🔥): 

- **关于模型在 Forward Pass 期间构建损失函数的讨论**：成员们讨论了模型是否可以在 Forward Pass 期间构建其损失函数的模型，以便在“提交结果之前检查其工作”。一位用户建议，虽然这通常不会发生，但对于涉及详细全局属性的任务可能是有益的。
- **Anthropic 对语言模型的见解**：一位用户链接了 [Anthropic 的研究](https://arxiv.org/abs/2305.04388)，该研究探讨了语言模型如何进行 Chain-of-Thought (CoT) 推理。研究表明，由于外部偏差，CoT 解释可能会误导模型预测背后的实际原因。
- **引入 Q* 以改进解码**：关于 [Q*](https://arxiv.org/abs/2406.14283) 的讨论，这是一个通过集成启发式 Q-value 模型来引导 LLM 解码过程的框架。其旨在减轻错误和不一致等问题，而无需对 LLM 进行大规模微调。
- **关于具有互联网访问权限的训练环境的辩论**：一位用户对给予训练环境实时互联网访问权限的安全性表示担忧，认为这可能会带来问题。
- **介绍用于设备控制的 DigiRL**：分享了另一篇文章 [DigiRL](https://arxiv.org/abs/2406.11896)，强调了一种通过微调 VLM 在真实条件下训练设备控制 Agent 的新型 RL 方法。该方法旨在解决静态演示训练中的局限性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://arxiv.org/abs/2406.14548">Consistency Models Made Easy</a>：一致性模型 (CMs) 是一类新兴的生成模型，比传统的扩散模型提供更快的采样速度。CMs 强制要求采样轨迹上的所有点都映射到……</li><li><a href="https://arxiv.org/abs/2406.14283">Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning</a>：Large Language Models (LLMs) 在许多自然语言任务中展示了令人印象深刻的能力。然而，自回归生成过程使 LLM 容易产生错误、幻觉……</li><li><a href="https://arxiv.org/abs/2406.13121">Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?</a>：长上下文语言模型 (LCLMs) 有潜力彻底改变我们处理传统上依赖检索系统或数据库等外部工具的任务的方法。利用 LCLMs 的能力……</li><li><a href="https://arxiv.org/abs/2406.11896">DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning</a>：Vision Language Models (VLMs) 的训练语料库通常缺乏足够的以决策为中心的数据。这使得现成的 VLMs 在处理真实环境下的决策任务时表现不佳……</li><li><a href="https://arxiv.org/abs/2305.04388">Language Models Don&#39;t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting</a>：Large Language Models (LLMs) 通过在给出最终输出之前生成逐步推理（通常称为 Chain-of-Thought 推理，CoT），可以在许多任务上获得强大的性能。然而……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1253659055147192380)** (1 messages): 

给定的消息历史记录中没有足够的信息来创建摘要。请提供更多消息或上下文以获得准确且信息丰富的摘要。
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1253427300788076654)** (5 messages): 

- **NumPy 2.0.0 兼容性问题**：讨论强调了使用 **NumPy 1.x** 编译的模块在 **NumPy 2.0.0** 下运行失败的问题。建议用户降级到 *'numpy<2'* 或升级受影响的模块。
- **Colab 作为解决方案**：成员们探索了使用 **Colab** 运行特定任务。一位成员成功安装并运行了 `lm_eval`，并提出帮助他人复制该设置。
- **Colab 中的 demo_boolq 问题**：尽管大多数示例都成功了，但 `demo_boolq` 任务在 **Colab** 中运行远程代码时遇到了问题。尽管进行了故障排除，问题仍然存在。
- **'main' 和 'master' 分支的区别**：澄清了应该使用的正确分支是 `main` 而不是 `master`。建议使用 `yes | lm_eval...` 来自动批准第三方代码提示。

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1253448937805709462)** (28 messages🔥): 

- **GPTs 实现中的权重共享（weight tying）问题**：一名成员询问了在 Embedding 层和输出 Logit 线性层之间实现权重共享时，与 Karpathy 的 GPT-2 方法的差异。进一步讨论揭示了**由于共享权重的两个副本导致了独立优化**，并因为权重初始化方法覆盖了共享设置而导致超时。

- **TinyGrad 中 clip_grad_norm_ 的实现问题**：成员们讨论了在 TinyGrad 中实现 `clip_grad_norm_` 时的性能问题和潜在错误。*“...Metal 只能拥有有限数量的缓冲区”* 被引用为影响其性能的具体限制，引发了关于将总和划分为 **31 个 Tensor 分块**的讨论。

- **梯度裁剪中的 Metal vs CUDA**：与 CUDA 不同，Metal 的缓冲区限制需要对 Tensor 进行分块。成员们明确了解决这些粗糙边缘的未来计划，包括进行内部 Scheduler 修复。

- **AMD 设备在示例中的超时错误**：成员们报告了在 AMD 设备上运行 YOLOv3 和 ResNet 示例时出现超时，错误日志显示存在同步问题。用户指出了像 **Radeon RX Vega 7** 这样的集成 GPU，并讨论了运行 Fuzzer 的 `tiny11` 服务器可能存在的过载问题。

**提及的链接**：<a href="https://wandb.ai/chenyuxyz/tinygrad-examples_mlperf/runs/rn5webqd/logs?nw=nwuserchenyuxyz>">chenyuxyz</a>：Weights & Biases，机器学习开发者工具

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1253473567878942762)** (10 messages🔥): 

- **Caption Dropout 方法讨论**：成员们讨论了在训练 SD3 时，使用全零作为 Caption Dropout 还是使用编码后的空字符串更有效。据观察，编码后的字符串并没有显著改变结果。
  
- **Sakuga 论文撤回令人意外**：有公告称，旨在推进卡通动画研究的 [Zhenglin Pan 的 Sakuga-42M 数据集论文](https://arxiv.org/abs/2405.07425) 已被撤回。该论文提出了一个大规模卡通动画数据集，但因未指明的问题导致撤回。

- **对 OpenAI 与政府合作的担忧**：有人分享了一条推文，表达了对 OpenAI 让政府提前接触新 AI 模型并倡导更多监管的担忧。推文暗示此举表明了重大的战略转变，可能是由于感知到了 AGI 的危险。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07425">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>：手绘卡通动画利用草图和色块来创造运动的错觉。虽然最近的进展如 CLIP、SVD 和 Sora 在理解和...方面表现出令人印象深刻的结果。</li><li><a href="https://fxtwitter.com/kimmonismus/status/1803908072999653528">来自 Chubby♨️ (@kimmonismus) 的推文</a>：让政府提前接触新模型，并且（推测）专门为国家服务，或者在当局决定时让政府监管这些模型；我不喜欢...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1253426556546711693)** (13 messages🔥): 

- **MM-DIT 中更多的 Latent 通道会增加全局不连贯性**：有人担心在 **MM-DIT** 等模型中，“使用更多的 Latent 通道似乎会导致更多的全局不连贯问题”。例子包括“女孩躺在草地上”的问题，以及难以生成“一条在空中飞行的连贯的龙”。

- **Chameleon 模型的梯度范数（Grad Norm）问题**：一名用户报告称，在训练新的 **Chameleon 模型**时，Embedding 层和 Norm 层出现了极端的梯度范数，导致出现 NaN 值。尽管尝试了常用的缓解技术（如降低 Learning Rate、梯度范数裁剪和 Weight Decay），问题仍然存在。

- **Batch Size 和 Learning Rate 建议**：有人建议，对于训练 **Llama 架构**的模型，可能需要由“数十万或数百万个 Token”组成的超大 Batch Size。这一建议是在解决了训练带有图像-文本对的 Chameleon 模型时遇到的类似问题后提出的。
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 messages): 

sajackie: 来自 Steam 的 50$
[steamcommunity.com/gift/9178](https://u.to/3tW_IA )
@everyone
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/)** (1 messages): 

sajackie: 来自 Steam 的 50$
[steamcommunity.com/gift/9178](https://u.to/3tW_IA )
@everyone
  

---

### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 条消息): 

sajackie: 来自 steam 的 50$
[steamcommunity.com/gift/9178](https://u.to/3tW_IA )
@everyone
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1253517858286207016)** (5 条消息): 

- **发布速度引发疑问**：有人对新版本发布速度所蕴含的意义表示好奇，特别是改进是源于 **posttraining** 还是 **pretraining**。一位成员推测这可能与 **pod training** 有关。

- **Pod Training 被高度重视**：一位贡献者强调了 **pod training** 的重要性，认为其价值被低估了。另一位贡献者表示赞同，但指出像 *continued pre-training* 这样令人困惑的因素使得难以确定确切的贡献。

- **进一步写作计划**：一位贡献者表示希望进一步探索并撰写关于该主题的文章，并表示需要咨询他人以收集更多信息。
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1253595837309784086)** (10 条消息🔥): 

- **Mira 的 PR 失误需要停止**：一位成员对 Mira 反复出现的 PR 错误表示沮丧，并分享了一个 [Twitter 链接](https://vxtwitter.com/tsarnick/status/1803920566761722166)。另一人表示赞同，称：“来自 OAI 的傲慢令人无法忍受。”
- **PR 培训不足被凸显**：Nathan Lambert 指出 Mira 似乎缺乏 PR 培训，暗示这可以解释她的错误。另一位成员则认为 Mira 不需要“昂贵且花哨的 PR 培训”。
- **转向 Claude**：Nathan Lambert 提到他已经转向使用 Claude，暗示对当前产品的不满。他还指出 Mira 并不使用自己公司的产品。
- **Mira 作为替罪羊**：一位成员推测 Mira 可能被用作替罪羊来应对批评，并转移对 Greg Brockman 和 Sam Altman 等高层的注意力，因为他们可能更愿意专注于其他工作。这一策略被视为 OpenAI 试图将自己塑造为一家“严肃公司”的一部分。
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1253736749931298946)** (8 条消息🔥): 

- **主播在 CNBC 上猛烈抨击 Perplexity**：一位成员提到 CNBC 的一次采访，主播对 **Perplexity** 持有出人意料的强烈负面看法：“*简直就像他自己用 Perplexity 搜了一下，然后得到了非常负面的答案*。”该采访是关于《连线》(Wired) 杂志批评 Perplexity 忽略 robots.txt 协议的文章。
- **YouTube 上的 CNBC 采访**：分享了一个标题为 [“Wired: AI startup Perplexity is 'BS machine'”](https://youtu.be/MFdjEW8_SUg?si=eV12HJRyM1RhMRns) 的 YouTube 视频链接。视频中，《连线》杂志的全球编辑总监讨论了对 AI 搜索初创公司 **Perplexity** 的调查。
- **针对 Perplexity 的批评浪潮**：成员们注意到近期针对 **Perplexity** 的负面情绪激增，一人表示：“有点惊讶，但也没那么惊讶。”
- **Casey Newton 的负面看法**：提到 Casey Newton 也写了一篇文章批评 Perplexity 创始人 Aarvind，称其“*有点天真*”。另一人对 Newton 在科技报道中的反科技立场表示沮丧。

**提到的链接**：<a href="https://youtu.be/MFdjEW8_SUg?si=eV12HJRyM1RhMRns">Wired: AI startup Perplexity is &#39;BS machine&#39;</a>：Wired 全球编辑总监 Katie Drummond 加入“Squawk Box”节目，讨论该杂志对 AI 搜索初创公司 Perplexity 的调查。

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

natolambert: snail 你在哪呢伙计，搞什么鬼
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1253432131070201938)** (19 messages🔥): 

- **YouTube 演示展示了 Windows/Linux 兼容性**：一名成员分享了一个名为 "open interpreter compatch demo" 的 [YouTube 视频](https://youtu.be/SOKq8RS0pR4)，展示了适用于 Windows/Linux 设置的基础 UI 和 TTS 功能。该演示通过 **Azure 使用 gpt4o**，并承诺很快会推出更多功能。

- **Claude 3.5 Sonnet 在实际使用中表现出色**：一位成员称赞了 Claude 3.5 Sonnet，提到：*"我喜欢它的交流方式和代码质量。"* 他们表示相比 GPT-4 和 GPT-4o 更倾向于使用 Claude 3.5，并称后者的交互过程令人 *"恼火"*。

- **Open Interpreter 在 Windows 上的安装指南**：关于安装的讨论引用了 [Open Interpreter 文档](https://docs.openinterpreter.com/getting-started/setup)。该过程包括 `pip` 安装和可选依赖项，展示了与 Windows 系统的兼容性。

- **分享了 DeepSeek Coder v2**：分享了指向 [DeepSeek Coder v2](https://ollama.com/library/deepseek-coder-v2) 的链接，包括许可协议和使用参数。该模型强调负责任的下游使用和伦理考量。

- **揭示了 Open Interpreter 的测试偏好**：关于 Open Interpreter 在 Mac 上表现最佳的询问揭示了，由于核心团队在该平台上进行了广泛测试，它在 Mac 上运行最为理想。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/getting-started/setup">Setup - Open Interpreter</a>：未找到描述</li><li><a href="https://youtu.be/SOKq8RS0pR4">open interpreter compatch demo</a>：未找到描述</li><li><a href="https://ollama.com/library/deepseek-coder-v2:latest">deepseek-coder-v2:latest</a>：一个开源的 Mixture-of-Experts 代码语言模型，在特定代码任务中实现了与 GPT4-Turbo 相当的性能。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1253430366879088650)** (1 messages): 

- **Open Interpreter 通过便利贴连接 WiFi**：一位用户分享了一次经历，一个**完全本地、控制计算机的 AI** 成功读取了便利贴上的 WiFi 密码并连接上网。他们附上了一个 [@hellokillian 的推文](https://x.com/hellokillian/status/1803868941040914824)链接，展示了这一能力。

**提到的链接**：<a href="https://x.com/hellokillian/status/1803868941040914824">来自 killian (@hellokillian) 的推文</a>：我给一个完全本地、控制计算机的 AI 看了一张写有我 WiFi 密码的便利贴。它上线了。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1253467056133640305)** (12 messages🔥): 

- **为咨询资助寻找 AI 活动**：一位成员询问是否有适合 2000 欧元差旅/客户发现资助的有趣活动，并指出 **AI Engineer World Fair** 时间太近且距离太远。另一位成员提到最近在**巴黎举行的 OSS AI 活动**刚刚结束。

- **演示链接问题**：一位成员询问其他人是否也未收到当前演示的链接。另一位成员确认了公开直播的链接已可用。

- **使用 HF Trainer 保存自定义模型的挑战**：一位成员分享了关于使用 **Hugging Face 框架**的见解，指出虽然 monkey patching 允许进行训练，但无法保存自定义的补丁模块。他们提供了一个示例代码片段来说明该问题。

- **Octomind 放弃使用 LangChain**：成员们讨论了一篇[博客文章](https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents)，详细说明了为什么 **Octomind** 不再为其 AI Agent 使用 LangChain。对话强调了 LangChain 僵化的抽象和复杂性导致其难以调试和维护，一些人建议将 **Langgraph** 作为潜在的替代方案。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents">为什么我们不再使用 LangChain 构建 AI Agent</a>：当抽象弊大于利时——在生产环境中使用 LangChain 的教训以及我们本该采取的做法</li><li><a href="https://www.youtube.com/watch?v=c0gcsprsFig">Lessons from a Year of Building with LLMs</a>：在这次特别的 Vanishing Gradients 直播录制中，Hugo 与 Eugene Yan (Amazon)、Bryan Bischof (Hex)、Charles Frye (Modal)、Hamel Husain 等进行了对话...
</li>
</ul>

</div>

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1253626595256700988)** (1 messages): 

- **Modal 团队因出色的支持受到称赞**：一位成员称赞了 **Modal 团队**在去年夏天运行 BLOOM 176B 模型时提供的协助。他们希望团队能继续保持这种友好且乐于助人的精神，并提到：“那太酷了……就像移山一样”。
- **相比 Discord 更倾向于使用 Slack**：尽管得到了支持，该成员表示更倾向于使用 Slack，并指出 Modal 团队也在该平台上提供帮助。他们评论道：“他们也有 Slack 并且会帮助大家，这太棒了，（我不喜欢 Discord）。”
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/)** (1 messages): 

4.8.15.16.23.42_: 我相信他们在某处提到过——一年。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1253427396179398757)** (1 messages): 

- **评估框架令人惊喜**：*"今天试用了评估框架，非常棒！"* 一位成员称赞了 **eval framework** 的 **直观 API 设计** 和编写良好的代码，强调了其开发体验。他们还赞赏了使用 LLM 代理端点的 **灵活性**，提到针对自定义企业基础 URL 执行任务非常容易。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

stevenmerrill: 类似的问题：有谁在大波士顿地区吗？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253425316840476713)** (2 messages): 

- **用户寻求解锁额度的建议**：一位用户询问如何解锁某些额度（credits）并寻求帮助。*"另外不确定如何解锁这些额度，请指教。"*
- **注册和邮箱验证问题**：另一位用户请求验证他们的邮箱，因为尽管填写了表格，但注册进度可能较慢。*"能请你检查一下 alexey.zaytsev@gmail.com 吗？"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/)** (1 messages): 

abhishek_54517: 似乎是一年。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1253450631109283850)** (14 messages🔥): 

- **加入旧金山 Techstars 创业周末**：一位成员宣布了将于 2024 年 6 月 28 日至 30 日在 Convex 举行的旧金山 Techstars 创业周末。Microsoft AI 总监 Gabriela de Queiroz 将担任主旨演讲嘉宾，导师来自 Google、Meta 等公司 [更多详情](https://www.startupweekendsf.com/)。

- **Reflexion 教程需要澄清**：一位用户对 Reflexion 教程为何使用 `PydanticToolsParser` 和 `bind_tools` 进行验证，而不是使用带有 `with_structured_output` 的简单循环感到困惑。他们还询问如果 LLM 的回答未能通过 `with_structured_output` 的验证会发生什么。

- **寻求 AI 工程师职位**：一位成员分享了他们在 LangChain、OpenAI 和多模态 LLM 等技术方面的丰富 AI 经验和专业知识，正在寻找全职机会进行进一步交流。

- **LangChain 消息流式传输问题**：一位成员在 React 应用中通过 Flask 流式传输 LangChain/LangGraph 的消息时遇到困难，并寻求帮助，但尚未收到解决方案。

- **并行请求处理问题**：一位成员描述了在 EC2 实例上使用 FastAPI 运行 LangChain 和 Gemini 模型时，AI 聊天机器人在处理并行请求时遇到的困难。另一位成员建议使用异步代码或 Serverless 函数来解决该问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/#initial-responder),">Reflexion - LangGraph</a>：未找到描述</li><li><a href="https://tenor.com/view/tom-and-jerry-tom-cat-confused-book-wtf-gif-19180489">Tom And Jerry Tom Cat GIF - Tom And Jerry Tom Cat Confused - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.startupweekendsf.com/">Techstars Startup Weekend</a>：Techstars 创业周末是一个为期 3 天的动态加速项目，你可以在这里开发、原型设计、设计并验证你的创业想法。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1253431763712348160)** (4 条消息): 

- **探索使用 MLX 的检索增强**：一位用户分享了一篇题为 *"Retrieval augmentation with MLX: A bag full of RAG, part 2"* 的新文章。该文章可以在 GitHub [此处](https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md)找到。
  
- **介绍适用于 GPT 的 'Mark' CLI 工具**：另一位用户介绍了一个名为 'Mark' 的以 Markdown 为核心的 CLI 工具，该工具利用链接和图像标签作为 RAG 方法与 GPT 模型进行交互。关于该工具及其设计思路的更多细节可以在他们的[详细帖子](https://relston.github.io/markdown/gpt4o/cli/2024/06/07/introducing-mark.html)中找到。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://relston.github.io/markdown/gpt4o/cli/2024/06/07/introducing-mark.html">Introducing ‘Mark’, a Markdown CLI tool for GPT4o</a>：简介：在这篇文章中，我想介绍 Mark，一个简单的 CLI 工具，它使用 Markdown 及其语法与 GPT4-vision/GPT4o 模型进行自然交互。</li><li><a href="https://github.com/uogbuji/mlx-notes/blob/main/2024/rag-basics2.md">mlx-notes/2024/rag-basics2.md at main · uogbuji/mlx-notes</a>：分享在使用 Apple MLX 机器学习框架时创建的个人笔记 - uogbuji/mlx-notes</li><li><a href="https://github.com/relston/mark">GitHub - relston/mark: Interact with GPT using markdown and images</a>：使用 Markdown 和图像与 GPT 交互。欢迎为 relston/mark 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1253426751699157043)** (8 条消息🔥): 

- **Turbcat 8b 发布**：分享了 [Turbcat 8b](https://huggingface.co/turboderp/llama3-turbcat-instruct-8b) 的链接，展示了该模型的数据集大小从 2GB 提升至 5GB，并增加了中文支持。同时还提供了模型的截图和详细信息。
- **AMD Mi300x GPU 兼容性问题**：一位用户询问关于使用 Axolotl 在 Mi300x GPU 上进行训练的问题，提到它似乎适配 PyTorch 2.4。然而，另一位用户回复道：“抱歉，我不了解 AMD 这边的情况”，表示对 Mi300x 的兼容性不确定。
- **与 70B 模型的比较**：一位用户对 Turbcat 8b 的发布表示惊讶，并询问它是否被认为优于 70B 模型。回复澄清说数据集更大了，但 72B 模型仍在开发中。

**提及的链接**：<a href="https://huggingface.co/turboderp/llama3-turbcat-instruct-8b">turboderp/llama3-turbcat-instruct-8b · Hugging Face</a>：未找到描述

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1253442852516528269)** (7 条消息): 

- **估算训练时间：简单公式 vs 实际执行**：一位成员询问是否可以通过将 Token 训练速率、数据集 Token 大小和预计 Epoch 数量相乘来估算训练时间。另一位成员建议最可靠的方法是“运行训练并在几步后查看估算值”，并为评估留出一些缓冲时间。

- **为 Tool Calling 模型格式化数据**：一位成员寻求关于如何将数据集转换为与 **axolotl** 兼容并使用 `<|tool_call|>` Token 的格式的建议。另一位成员建议查看包含 ShareGPT 设置的 **example configs** 以获取格式化指导。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 条消息): 

ben44: 已移至 <#1110594519226925137>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1253429619894386878)** (15 messages🔥): 

- **Discord API 版本更新正在进行中**：一位成员询问 **Cohere 的 API** 是否支持 **OpenAI 风格的角色**，如 "user" 或 "assistant"。另一位成员澄清说，**目前的 API 不支持**，但正在开发一个新版本，该版本将适配这些角色以简化集成。
- **与 OpenAI ChatCompletion 不兼容**：多位用户讨论了 **Cohere 的 Chat API** 如何与 **OpenAI 的 ChatCompletion API** 不兼容，提到许多服务采用了 **OpenAI 的 API**，而 Cohere 则没有。其他模型和服务也有**不同的 API**，使通用集成变得复杂。
- **对 OpenRouter 等服务中模型完整性的担忧**：成员们对 **OpenRouter** 等服务表示怀疑，担心它们可能会根据 Prompt 将请求的模型换成更便宜的替代品。一位用户建议使用 **原始 Prompt 和侧边栏对比 (side-by-side comparisons)** 来验证所提供的模型。
- **个人博客链接：关于资源囤积**：一位用户分享了一篇 [博客文章](https://jerry.wtf/posts/use-your-potions/)，将 RPG 游戏中囤积一次性道具的行为与现实生活中避免请求帮助或推广项目进行了类比。他们讨论了在玩过《博德之门 3》后这种心态是如何改变的，在游戏中他们决定按原定用途使用资源，结果发现非常有趣。

**提到的链接**：<a href="https://jerry.wtf/posts/use-your-potions/">Use Your Potions and Scrolls</a>：我发现当我玩 RPG 游戏时，我经常囤积像药水和卷轴这样的一次性物品，把它们留给未来的某个关键时刻。我玩完像《天际》这样的游戏时，背包里装满了没用掉的资源……

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1253439701549842595)** (1 messages): 

- **在 World's Fair 偶遇创始人**：@aiDotEngineer 的 World's Fair 参与者可以见到 LlamaIndex 创始人 @jerryjliu0 的两次演讲。[6 月 26 日下午 4:53](https://twitter.com/llama_index/status/1803880529571516917)，他将讨论知识助手的未来（Future of Knowledge Assistants）并发布特别公告；第二次演讲在 6 月 27 日下午 3:10。
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1253583385998131242)** (13 messages🔥): 

- **Python AI 工程师寻求工作机会**：一位经验丰富的 Python AI 全栈开发人员分享了他们的详细简历，他们在 **AI 驱动的软件应用** 和 **LLM** 方面拥有丰富经验。他们强调了在 **NLP 技术** 以及 Transformers、PyTorch 和 TensorFlow 等**框架**方面的深厚技能。

- **为 Neo4jGraphStore 生成 Embedding**：一位成员寻求关于如何使用 LLM 为 **Neo4jGraphStore** 生成 Embedding 的指导，因为最初在添加节点和关系时并未使用。该咨询强调了在节点创建后进行 Embedding 集成的需求。

- **使用 Ollama 和 LLM 进行结构化 NER 任务**：一位成员提到使用 **Ollama** 和开源模型进行 **NER 任务**，但在按 Prompt 实现结构化输出时遇到了问题。另一位成员建议在 NER 任务中使用 **Gliner** 而不是 LLM。

- **在 LlamaIndex 中添加任务功能**：一位用户在他们的 **LlamaIndex 项目** 中寻求添加发送电子邮件、创建 Jira 工单和日历事件等功能的资源。另一位成员建议 **Agent** 可能会很有用，并提供了 [自定义 Agent 的文档](https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/)。

- **Llama 和 Nuxt.js 项目示例**：一位成员询问是否有使用 **Llama** 和 **Nuxt.js** 的示例项目。在截获的消息中没有提供直接回复。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/agent/custom_agent/?h=custom+agent">构建自定义 Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples">示例 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1253425446855639135)** (12 messages🔥): 

- **Swyx 谈论 AI 机遇**：一位成员分享了一篇[文章](https://www.heavybit.com/library/article/ai-hidden-opportunities-for-software-developers-swyx)，讨论了 AI 工程专家 **Shawn "swyx" Wang** 以及 AI 在用例和软件开发人员方面的新机遇。文章还宣传了 [AI Engineer World’s Fair 活动](https://www.ai.engineer/worldsfair)。

- **Groq 增加 Whisper 模型支持**：Groq 宣布增加对 **Whisper 模型**的支持，声称其运行速度达到 **166 倍实时速度**。然而，一些成员对播客转录等应用的低速率限制（rate limits）表示担忧，并推测了潜在的非线性、多调用处理用例。

- **音乐转文本模型咨询**：一位成员询问了能够生成音乐文本描述（如流派、调性、节奏和其他标签）的现代 AI 解决方案。讨论集中在对准确、详细的音乐描述的需求，而非歌词。

- **MoA 击败 GPT-4**：一条 [推文](https://x.com/corbtt/status/1803813970018791845?s=46) 宣布了 Mixture of Agents 模型及其微调流水线，其成本比 GPT-4 **便宜 25 倍**。根据该推文，人类在 59% 的情况下更倾向于 MoA 的输出，并且它在 Arena-Hard 和 Alpaca Eval 基准测试中创下了新的 SOTA。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/corbtt/status/1803813970018791845?s=46">来自 Kyle Corbitt (@corbtt) 的推文</a>: 非常激动地宣布我们的 Mixture of Agents 模型 + FT 流水线：击败了 GPT-4，但价格便宜 25 倍！- 人类在 59% 的情况下更倾向于 MoA 输出而非 GPT-4 - 在 Arena-Hard (84.8) 和 Alpaca...</li><li><a href="https://www.heavybit.com/library/article/ai-hidden-opportunities-for-software-developers-swyx">AI 的隐藏机遇：Shawn &quot;swyx&quot; Wang 谈论新用例和职业生涯 | Heavybit</a>: Shawn “swyx” Wang 讨论了 AI 中隐藏的机遇，包括新的用例以及为有志于成为 AI 工程师的人提供的新机会。
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1253627413712076811)** (2 messages): 

```html
- **启动垃圾信息警报**：一条消息指出用户 <@937822421677912165> 再次参与垃圾信息活动。该用户被标记以提醒管理员介入。
- **同一用户重复发布垃圾信息**：另一个警报点名了同一用户 <@937822421677912165> 的重复垃圾信息行为。管理员正被召集处理此情况。
```
  

---


### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1253460960681922592)** (3 messages): 

- **垃圾信息举报引发行动**：一位成员对另一位用户的不当消息表示愤怒，称“wtf what's wrong with u”。其他人建议将该消息举报为垃圾信息，希望 Discord 能采取行动。
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1253436473013964900)** (4 messages): 

- **RecSys Learners 虚拟聚会征集 RSVP**：针对推荐系统（Recommendation Systems）爱好者的聚会定于 2024 年 6 月 29 日太平洋标准时间（PST）上午 7 点举行。免费参加，感兴趣的参与者可以[在此 RSVP](https://lu.ma/7pvpp1cm)以获取活动链接。
- **关于 AI Quality 会议的询问**：一位成员询问是否有人参加下周二在旧金山举行的 AI Quality 会议。另一位成员表示好奇，询问有关该活动的更多细节。

**提到的链接**：<a href="https://lu.ma/7pvpp1cm">RecSys Learners 虚拟聚会 · Luma</a>：加入我们，参加这场令人兴奋且信息丰富的 RecSys Learner 虚拟聚会，专为对推荐系统充满热情的爱好者和专业人士设计。本次……

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1253747643646214236)** (2 messages): 

- **PyTorch/TensorDict 或 PyTorch/NestedTensor 实用性咨询**：一位用户询问了 **PyTorch TensorDict** 或 **NestedTensor** 的用例和偏好。另一位成员称赞了它在将多个数据输入作为单个对象处理方面的实用性，消除了在管理 dtypes/devices 或跨 batch 维度进行广播时对样板代码（boilerplate code）的需求。
  

---



---



---



---



---



---



{% else %}


> 完整的频道分类详情已在邮件中截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}