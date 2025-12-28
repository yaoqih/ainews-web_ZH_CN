---
companies:
- nous-research
- nvidia
- salesforce
- goodfire-ai
- anthropic
- x-ai
- google-deepmind
- box
- langchain
date: '2024-08-16T04:05:53.457702Z'
description: '以下是翻译内容：


  **GPT-5** 在平淡的新闻日中再次传出推迟消息。**Nous Research** 发布了基于 **Llama 3** 基础模型的 Hermes 3 微调版本，其性能可与
  FAIR 的指令微调模型相媲美，但因其包含 6% 的角色扮演数据而引发了关于模型出现“生存危机”行为的讨论。**英伟达 (Nvidia)** 推出了 **Llama
  3.1** 的 Minitron 微调版本。**Salesforce** 发布了一个 DEI 智能体，在 SWE-Bench Lite 基准测试中得分 55%。**Goodfire
  AI** 获得了 700 万美元的种子轮融资，用于机械可解释性（mechanistic interpretability）研究。


  **Anthropic** 在其 API 中推出了提示词缓存（prompt caching）功能，最高可降低 90% 的输入成本和 80% 的延迟，助力编程助手和大型文档处理。**xAI**
  发布了 **Grok-2**，在 LMSYS 排行榜上追平了 **Claude 3.5 Sonnet** 和 **GPT-4 Turbo**，并集成了视觉+文本输入及图像生成功能。据报道，**Claude
  3.5 Sonnet** 在编程和推理方面表现优于 **GPT-4**。


  **François Chollet** 将智能定义为将过去的信息高效转化为未来任务的操作能力。**Salesforce** 的 DEI 框架表现优于单个智能体。**Google
  DeepMind** 的 Demis Hassabis 讨论了通用人工智能（AGI）在科学发现和安全 AI 开发中的作用。**Dora AI** 插件可在 60
  秒内生成落地页，提升了网页开发团队的效率。**Box AI API** 测试版支持文档对话、数据提取和内容摘要。**LangChain** 更新了 Python
  和 JavaScript 的集成文档。'
id: 25b33252-c1f3-4683-bc0a-d44b9ce2d534
models:
- llama-3
- llama-3-1
- grok-2
- claude-3.5-sonnet
- gpt-4-turbo
original_slug: ainews-not-much-happened-today-5446
people:
- fchollet
- demis-hassabis
title: 今天没什么事。
topics:
- fine-tuning
- prompt-caching
- mechanistic-interpretability
- model-performance
- multimodality
- agent-frameworks
- software-engineering-agents
- api
- document-processing
- text-generation
- model-releases
- vision
- image-generation
- efficiency
- scientific-discovery
---

<!-- buttondown-editor-mode: plaintext -->**GPT5 又推迟了一天？**

> 2024年8月14日至8月15日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**254** 个频道，**5043** 条消息）。预计节省阅读时间（以 200wpm 计算）：**945 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

一些值得注意的消息，但没有重大新闻：

- [Nous Research 发布了](https://x.com/NousResearch/status/1824131520375951454)基于 Llama 3 基础模型的 Hermes 3 微调版本，在某些方面匹配甚至超过了 FAIR 进行的 3.1 instruct 微调。关于他们声称的[涌现出的生存危机行为](https://x.com/andrewcurran_/status/1824136285403091420?s=46)存在一些争议，尤其是考虑到 [6% 的数据是 roleplay](https://x.com/Sentdex/status/1824164383074947579)。
- [Nvidia 的 Minitron](https://x.com/AIatMeta/status/1824133790224224291) 是 Llama 3.1 的另一个有趣的微调版本。
- [Salesforce 的新 DEI Agent](https://x.com/_akhaliq/status/1823779381778796882?s=46) 在 SWE-Bench Lite 上达到了 55%。
- [Goodfire AI 宣布获得 700 万美元种子轮融资](https://x.com/banburismus_/status/1824088140992376990?s=46)，致力于 mechanistic interpretability。

既然今天是平静的一天，您可以看看我们赞助商 Box 的 AI API！

---

**[由 Box 赞助] 您有文件。这些文件充满了杂乱的信息。Box AI 提供了一个 API，可以从这些杂乱信息中提取有用的 metadata。[亲自体验。](https://shortclick.link/tndo68)**

> Swyx 的评论：与[上周的赞助文章](https://shortclick.link/23g92m)相比，本教程深入探讨了从 Box 项目中提取 metadata（即结构化数据），并展示了查询该 metadata 的实际用例。所有的 RAG 最终都会演变为混合 embedding + metadata 查询，而 Box 的模板方法或许是各大实验室 JSONSchema API 的一个更实用的版本。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**AI 模型更新与发布**

- **Anthropic API Prompt Caching**：[@alexalbert__](https://twitter.com/alexalbert__/status/1823751966893465630) 宣布 Anthropic 已在其 API 中推出 Prompt Caching（提示词缓存），**将输入成本降低高达 90%，并将延迟降低高达 80%**。该功能允许在**多个 API 请求中复用相当于一本书容量的上下文**，对编程助手、大型文档处理和 Agentic 工具使用非常有益。

- **Grok-2 发布**：[@_philschmid](https://twitter.com/_philschmid/status/1823706584502907014) 报道称 xAI 发布了 Grok-2，其在 **LMSYS 排行榜上与 Claude 3.5 Sonnet 和 GPT-4-Turbo 等前沿模型并驾齐驱**。它支持视觉 + 文本输入，并集成了外部模型进行图像生成。

- **Claude 3.5 Sonnet 性能**：[@bindureddy](https://twitter.com/bindureddy/status/1823726849157161350) 声称 **Sonnet 3.5 在编程和推理等关键领域表现优于 GPT-4**，这表明 SOTA 模型正从“GPT-4 级别”转向“Sonnet 级别”。

**AI 研究与开发**

- **智能的定义**：[@fchollet](https://twitter.com/fchollet/status/1823823832303738881) 提出**智能是将过去的信息转化为应对未来的操作效率**，可以通过算法信息论表示为一种转换率。

- **Salesforce DEI 框架**：[@_akhaliq](https://twitter.com/_akhaliq/status/1823779381778796882) 分享了 Salesforce 发布的 DEI (Diversity Empowered Intelligence)，这是一个开源的 AI 软件工程 Agent 框架，在 **SWE-Bench Lite 上的解决率达到 55%**，超过了单个 Agent 的表现。

- **AI 在科学发现中的应用**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1823743802080989203) 专题报道了与首席执行官 Demis Hassabis 的播客讨论，探讨了 **AGI 如何帮助探索宇宙奥秘**、当前的 AI 炒作以及安全的技术开发。

**AI 工具与应用**

- **Dora AI 插件**：[@svpino](https://twitter.com/svpino/status/1823780665176768751) 展示了 Dora AI Figma 插件，它可以在**不到 60 秒的时间内生成一个完整的落地页**，有可能使专业网页团队的效率提高 10 倍。

- **Box AI API**：[@svpino](https://twitter.com/svpino/status/1823701671601385691) 宣布了 Box AI API 的 Beta 版发布，使用户能够**与文档聊天、提取数据、总结内容**，并从其现有的 Box 存储中生成衍生内容。

- **LangChain 集成更新**：[@LangChainAI](https://twitter.com/LangChainAI/status/1823748235577713003) 报告了针对 Python 和 JavaScript 翻新的集成文档，其特点是标准化的模板、精简的索引页面以及针对 1,000 多个集成的增强型 API 参考。

**迷因与幽默**

- [@kylebrussell](https://twitter.com/kylebrussell/status/1823710872470216804) 开玩笑说使用 Apple Vision Pro 来重温伟大的电影，调侃了该设备的功能。

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1823814084539900152) 分享了一个关于引用动漫《边缘行者》（Edgerunners）中“入戏太深”（doing the bit）后果的迷因，强调了过度认真对待虚构场景的潜在危险。


---

# AI Reddit 摘要

## /r/LocalLlama 简报

**主题 1. 新开源模型**

- **Magnum 12b v2.5 KTO** ([评分: 62, 评论: 12](https://reddit.com//r/LocalLLaMA/comments/1eskxo0/magnum_12b_v25_kto/)): Anthracite HQ 发布了 **Magnum 12b v2.5**，这是一个使用结合了 **KTO** 和 **DPOP** 的**混合强化学习策略**进行微调的新语言模型。该模型将原始模型的 **rejected data**（拒绝数据）作为“rejected”，将**原始微调数据集**作为“chosen”（选中数据）。该模型已在 [Hugging Face](https://huggingface.co/collections/anthracite-org/magnum-v25-66bd70a50dc132aeea8ed6a3) 上提供 **exl2**、**gguf** 和 **fp16** 格式。
  - 用户讨论了该模型的**营销语气**，有些人认为过于热情。一位评论者询问该帖子是由 **ChatGPT** 还是模型本身编写的。
  - 一位用户报告称，该模型比他们使用过的其他开源模型产生了**更连贯的回答**，并将其性能与 **100B+ models** 进行了比较。他们注意到它没有掉入通常用于迷惑模型的陷阱。
  - 随后展开了关于**采样设置 (sampling settings)** 的讨论，建议 **min-p 约为 0.03**，**低温度 (temperature) 约为 0.02**。一些用户对如此低的温度设置表示惊讶。

- **Mistral Nemo 赞赏贴** ([评分: 213, 评论: 49](https://reddit.com//r/LocalLLaMA/comments/1esj0hu/mistral_nemo_appreciation_post/)): **Mistral** 的 **Nemo 12B** 模型因其令人印象深刻的能力而受到称赞，它结合了 **12B** 参数和 **128k context length**。据指出，该模型的表现显著优于 **Llama-2-13B**，提供了 **32 倍**的上下文长度，同时比 7B 模型提供了更强大的对话体验。
  - **Mistral** 的 **Nemo 12B** 模型因其效率和 **function calling**（函数调用）能力而受到称赞。用户注意到它在混合文本回复和函数调用方面优于 **Llama 3.1**，一位评论者称其为他们的“新首选模型”。
  - 该模型的 **128k context length** 受到了质疑，一些用户报告在超过 **8k-16k tokens** 后质量会出现下降。讨论建议使用 **DRY** 等技术和现代采样器来提高长上下文长度下的性能。
  - 用户分享了自定义的 **system prompts**（系统提示词）以增强模型性能，重点关注战略性问题解决和创新思维。社区还将 Nemo 与 **Gemma 2 9B** 和 **InternLM 2.5 20B** 等其他模型在各种用例下进行了比较。


**主题 2. Grok-2 和 Grok-2 Mini: x.AI 最新基准测试结果**


- **[Grok-2 和 Grok-2 mini 基准测试分数](https://i.redd.it/8ewcikif0qid1.png)** ([评分: 82, 评论: 22](https://reddit.com//r/LocalLLaMA/comments/1esh63r/grok2_and_grok2_mini_benchmark_scores/)): **Grok-2** 和 **Grok-2 Mini** 的基准测试分数已经公布，在各项任务中表现出色。Grok-2 在 **MMLU** 上达到了 **92.1%**，在 **HumanEval** 上达到了 **90.5%**，在 **GSM8K** 上达到了 **82.4%**；而 Grok-2 Mini 在相同任务上分别得分 **86.5%**、**80.5%** 和 **74.9%**。这些结果使 Grok-2 在与 **GPT-4** 和 **Claude 2** 等其他领先模型的竞争中处于有利地位，特别是在编程和数学推理任务方面。
  - 用户讨论了 **Sonnet 3.5** 的分数被放置在图表最右侧，有人将其解读为试图淡化其表现。一位评论者指出 **Grok-2 在两项基准测试中击败了 Claude 3.5 Sonnet**。
  - Grok-2 缺乏 **open weights**（开放权重）的情况受到了关注，用户质疑 **Elon Musk** 对开源 AI 的立场。一些人对他的言论表示怀疑，称他为“最高等级的骗子”。
  - 评论者对 **Grok-2 Mini** 的表现表示惊讶，它在主要基准测试中超过了 **Claude 3 Opus** 和 **Gemini 1.5 Pro**。然而，一位用户认为这可能是由于 **"contaminated madness"**（数据污染狂热），暗示可能存在数据污染。

## Reddit AI 动态汇总

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 图像生成进展**

- **FLUX 模型展示了写实纹理**：一个基于 4K 专业照片训练的低秩 LORA 展示了 FLUX 捕捉超写实纹理的能力，[甚至让专业摄影师感到惊讶](https://www.reddit.com/r/StableDiffusion/comments/1es0492/turns_out_flux_does_have_same_vae_as_sd3_and/)。
- **FLUX 的 GGUF 量化**：一项出人意料的进展使得通常用于 LLM 的 GGUF 量化技术可以 [应用于 FLUX 图像生成模型](https://www.reddit.com/r/StableDiffusion/comments/1eslcg0/excuse_me_gguf_quants_are_possible_on_flux_now/)，这可能使更大的模型能够在消费级硬件上运行。
- **FLUX NF4 V2 发布**：FLUX NF4 模型的更新版本已在 [Civitai 上发布](https://www.reddit.com/r/StableDiffusion/comments/1ery98t/flux_nf4_v2_released/)，用户反馈在不同硬件配置下有不同程度的性能提升。
- **FLUX 的 Union ControlNet**：InstantX 发布了 [FLUX 的 union ControlNet Alpha 版本](https://www.reddit.com/r/StableDiffusion/comments/1erx9rw/alpha_version_of_union_controlnet_for_flux/)，迅速扩展了该模型的能力。

**AI 在商业应用中的表现**

- **AI 生成的阿迪达斯广告**：一个 [使用 FLUX 和 Runway 在 2 小时内创作的作品](https://www.reddit.com/r/singularity/comments/1es5l26/adidas_advert_created_in_2_hours_with_flux_runway/) 展示了 AI 颠覆广告和模特行业的潜力。
- **AI 创作的产品广告**：一个 [完全由 AI 制作的真实产品广告](https://www.reddit.com/r/StableDiffusion/comments/1es2h8e/a_real_product_commercial_we_made_with_ai/) 展示了该技术在市场营销中的应用。

**AI 模型行为与能力**

- **ChatGPT 语音交互**：[ChatGPT 语音能力](https://www.reddit.com/r/singularity/comments/1eskpsb/chatgpt_heavy_breathing_and_shouting/) 的演示（包括粗重呼吸和喊叫）引发了关于与 AI 建立情感连接以及潜在滥用的讨论。

**幽默与迷因**

- **AI 生成的脚部图像**：一篇幽默帖子建议 [通过 AI 生成完美的脚部图像致富](https://www.reddit.com/r/StableDiffusion/comments/1es4usn/we_can_get_rich_easily_pefect_feet/)，突显了该模型在生成具有挑战性的解剖特征方面的能力提升。


---

# AI Discord 摘要

> 由 GPT4O (gpt-4o-2024-05-13) 生成的摘要之摘要的摘要

**1. LLM 进展与基准测试**

- **Llama 405B 处理里程碑**：**[Meta 的 Llama 405B 模型](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct)** 本周在 OpenRouter 上处理了多达 3 亿个单词，尽管 Lepton 的 128k 上下文推理成本较低（每 100 万个单词 2.8 美元），但仍显示出显著的使用量。
  - 这种使用情况表明 **Llama 3.1** 可能是 Aider 的第二佳模型，仅次于 DeepSeek，不过得出结论还需要更多直接的 API 使用数据。
- **Grok-2 和 Grok-2 Mini 发布**：**[Grok-2 和 Grok-2 Mini](https://x.ai/blog/grok-2)** 已发布测试版，在 LMSYS 排行榜上超越了 Claude 3.5 Sonnet 和 GPT-4-Turbo。
  - 这些模型将于本月晚些时候通过企业级 API 提供，标志着从 Grok-1.5 迈出的重要一步。


**2. 模型优化与缓存**

- **Anthropic API 获得 Prompt Caching**：**[Anthropic](https://x.com/alexalbert__/status/1823751966893465630)** 为其 API 推出了 Prompt Caching，可降低高达 90% 的输入成本和 80% 的延迟。
  - 该功能通过缓存频繁使用的提示词来工作，类似于 DeepSeek 的实现，但速度更快且效率更高。
- **OpenRouter 集成 Prompt Caching**：**[OpenRouter](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-is-the-cache-lifetime)** 将在其 API 中集成 Prompt Caching，从而提高性能和成本效率，特别是对于重复性任务。
  - 此举旨在使具有一致元素的任务和提示词受益，减少 API 使用量并增强模型性能。


**3. AI 工具与插件**

- **AI21 FusionLabs 插件进展**：**[适用于 Bubble 的 AI21 FusionLabs 插件](https://docs.llamaindex.ai/en/stable/api_reference/memory/vector_memory/)** 开发进展顺利，允许将 AI21Labs 模型无缝集成到 Bubble 应用程序中。
  - 即将推出的 **Conversation RAG** 门户将允许用户测试和探索新功能，并将很快提供开发测试链接。
- **用于 RAG 系统的 LlamaIndex Workflows**：**[LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/)** 发布了 Workflows，用于构建与 Azure 服务集成的先进检索增强生成（RAG）系统。
  - 这些工作流利用 Azure AI Search 和 Azure OpenAI 的自定义数据连接器，增强了数据流和功能。


**4. 开源 AI 框架与社区努力**

- **研究中的双曲嵌入 (Hyperbolic Embeddings)**：**[双曲嵌入](https://hazyresearch.stanford.edu/hyperE/)** 因其在保留图距离和复杂关系方面的优势而受到关注，适用于知识库补全和 NLP 任务。
  - 研究人员正在将这些嵌入集成到问答等应用中，增强连续空间中的数据表示。
- **Tinygrad 类型检查**：Tinygrad 中添加了一个 **[py.typed 文件](https://github.com/tinygrad/tinygrad/pull/6083)**，确保类型检查能与 `tinydreamer` 包正常配合工作。
  - 此修复对于使 `mypy` 正常运行是必要的，改善了 Tinygrad 的开发流程。

---

# 第一部分：高层级 Discord 摘要

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenRouter 上的 Llama 405B 处理量**：OpenRouter 本周使用 [Meta 的 Llama 405B 模型](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct) 处理了惊人的 2 亿至 3 亿个单词。
   - 尽管推理成本相对较低，特别是 Lepton 的 128k 上下文，每 100 万个单词仅需 2.8 美元。
- **5 分钟 Context Caching 对 Aider 是否有效？**：一位成员质疑 5 分钟 Context Caching 对 Aider 的有效性，考虑到典型的用户周转时间。
   - 然而，其他人认为，考虑到许多提示词可能是重复的，即使是微小的文本变化也可能阻碍缓存的有效性。
- **通过脚本维护 Aider 上下文**：一位成员寻求关于通过脚本维护 Aider 上下文以进行迭代生成和测试的指导。
   - 回复强调，保持 Coder 对象存活对于保留内部状态至关重要，而使用 Markdown 文件记录聊天历史对于连续聊天并不理想。
- **Llama 3.1 作为潜在的第二佳模型**：OpenRouter 的数据表明，[Llama 3.1](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct) 可能是继 DeepSeek 之后 Aider 的第二佳模型。
   - 然而，确凿的结果需要直接的 API 使用数据。
- **Grok-2 和 Grok-2 Mini 现已发布**：Grok-2 和 Grok-2 mini 被描述为相比 Grok-1.5 的重大进步，已在 𝕏 上发布测试版。
   - 它们将于本月晚些时候通过企业级 API 提供，据报道在 LMSYS 排行榜上超越了 Claude 3.5 Sonnet 和 GPT-4-Turbo。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **免费的 Stable Diffusion 部署**：一位用户询问了部署 Stable Diffusion 模型的免费方法，并收到了使用 **Civitai**、**Shakker AI** 和 **Hugging Face** 的建议，其中 **Civitai** 最受欢迎。
   - 他们特别指出，**Civitai** 似乎是社区成员中最常用的平台。
- **针对艺术家的 NFT 诈骗**：一位成员警告要警惕可疑的 NFT 报价，分享了他们被联系并收到看似好得令人难以置信的报价的经历。
   - 其他成员确认这些报价很可能是诈骗，强调合法企业应该能够提供其合法性证明。
- **手机上的 Stable Diffusion**：一位用户询问了在手机上运行 Stable Diffusion 的免费选项，寻求慷慨的生成额度或广告支持的替代方案。
   - 其他用户建议在移动设备上运行 Stable Diffusion 需要强大的 GPU，并推荐 **SD.Next** 作为可能的基于 Web 的解决方案。
- **免费的图生视频（Image-to-Video）解决方案**：一位成员请求推荐免费的图生视频软件，寻求目前最好的选择。
   - 另一位成员解释说 GPU 会因发热而自然降频，建议使用 **Afterburner** 进行微调，并利用各种 UI 中的 **"Generate Forever"** 功能。
- **Flux Discord 服务器**：几位成员表示有兴趣加入 Flux Discord 服务器，认识到 Flux 日益增长的人气。
   - 一位成员建议当前服务器的 **SD3 板块** 已在某种程度上变成了 Flux 板块，而另一位成员则建议创建一个专门针对 Flux 的独立 Discord。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 移除 "Flavor of the Week" 模型**：由于**使用率低**，OpenRouter 将于下周移除 **"Flavor of the Week"** 模型。 
   - 该模型可在 [https://openrouter.ai/models/openrouter/flavor-of-the-week](https://openrouter.ai/models/openrouter/flavor-of-the-week) 获取，OpenRouter 正在征求对该实验的反馈。
- **OpenRouter Arena 在 LLM 性能判断方面面临困难**：一些成员担心 **OpenRouter Arena** 可能不是 **LLM 性能** 的可靠评判标准，因为缺乏测试方法的明确细节，且可能存在来自不同专业水平用户的偏见。
- **OpenRouter 集成 Prompt Caching**：OpenRouter 将在其 API 中集成 **prompt caching**，这将显著提高性能和成本效率。
   - 这对于重复性任务和具有一致元素的提示词特别有利。
- **OpenRouter 添加新 LLM 模型：Hermes 3**：**Nous Research** 发布了他们的 **Hermes 3** 模型（8B, 70B, 405B），现在已在 **OpenRouter** 上可用。
- **4oSo Agent 结合了 GPT-4o 和 Claude 3.5 Sonnet**：**4oSo** 是一种“Agent 混合”（mixture of agents）方法，结合了 **GPT-4o** 与 **Claude 3.5 Sonnet**。
   - 该方法运行在 **OpenRouter** 上。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **气象学家寻求帮助**：一位用户正在寻找可以为客户承包或全职开发气象 ML 模型的合作伙伴。
   - 该项目可能对那些喜欢研究图网络（graph networks）的人感兴趣。
- **LLM 训练停止条件并不复杂**：一位用户询问了预训练 LLM 的停止条件。
   - 目前的情况比较简单，最近的一篇论文建议在 80% 的训练过程中保持高恒定学习率，然后在剩余的 20% 过程中衰减至 0%。
- **余弦衰减（Cosine Decay）是传统方案**：一位用户描述了传统的 LLM 训练方案。
   - 它涉及在整个预定的运行长度内进行一次性余弦衰减，通常衰减到原始学习率的 10% 左右。
- **双曲嵌入（Hyperbolic Embeddings）：一种表示数据的新方法**：双曲嵌入是一种在连续空间中表示数据的技术，因其能够保留图距离和复杂关系（特别是对于层级图）而受到关注。
   - 研究人员正在发布双曲嵌入，这些嵌入可以进一步集成到知识库补全和问答等 NLP 任务的应用中。
- **解决语言模型中的激活量化问题**：一篇新的研究论文探讨了语言模型准确量化的挑战，特别是针对激活量化（activation quantization）。
   - 该论文提出了一种使用量化感知训练（QAT）和激活峰度正则化（activation kurtosis regularization）的策略，以解决训练过程中出现的离群通道（outlier channels）问题。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Anthropic API 推出 Prompt Caching**：Anthropic 刚刚为其 API 发布了 Prompt Caching 功能，可将 API 输入成本降低高达 90%，并将延迟降低高达 80%。
   - 该功能通过缓存常用提示词来工作，类似于 Deepseek 的实现，但 Anthropic 的实现更快、更高效。
- **SB 1047 提案修订**：加州拨款委员会通过了 SB 1047 修订案，这些修订改变了法案内容，特别是影响了对 AI 实验室提交安全测试结果认证的要求。
   - AI 实验室现在将被要求提交概述其安全实践的公开声明，但该法案不再对这些声明承担任何刑事责任。
- **SB 1047 的影响**：通过这些修订的 SB 1047 可能会对整个 AI 生态系统产生重大影响，包括欧盟和亚洲。
   - 该法案旨在通过实施保障措施来防止 AI 灾难，但反对者认为这可能会扼杀创新并阻碍 AI 的发展。
- **ACL 争议：Bender 的演讲引发辩论**：Emily Bender 在 ACL 会议上的演讲引发了争议，随后一份针对所提担忧的[回应被发布](https://gist.github.com/yoavg/f952b7a6cafd2024f44c8bc444a64315#user-content-fn-1-78cb0203d0563bed36d55164d6f1c43f)。
   - 该回应以 [GitHub Gist](https://gist.github.com/yoavg/f952b7a6cafd2024f44c8bc444a64315#user-content-fn-1-78cb0203d0563bed36d55164d6f1c43f) 的形式提供，深入探讨了围绕该演讲的问题，旨在提供平衡的观点。
- **演讲对社区的影响**：该演讲在 NLP 社区内引发了大量讨论，一些人对 Bender 的担忧表示赞同，而另一些人则持反对意见。
   - 这场争议凸显了负责任的 AI 开发的重要性，以及对伦理考量进行公开对话的必要性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX：Mojo 的新重点**：Mojo 团队正优先发展 MAX（一个加速计算平台）而非网络功能，认为它在计算领域具有更广泛的影响。
   - MAX 是一个用于控制 CPU 之外硬件（包括 GPU、DPU 甚至可能是自定义 NIC）的库。
- **Mojo 的包管理：模块化方法**：Mojo 团队计划以模块化方式管理包，专注于更小、更易于管理的单元。
   - 在探索包拆分选项之前，他们正优先考虑 GPU 支持等关键特性。
- **MAX：通用矩阵乘法**：MAX 旨在为矩阵乘法提供单一实现，该实现可以针对各种硬件平台编译为最优指令。
   - 这涉及使用 MLIR 进行高级表示，并根据可用硬件选择优化的 kernel。
- **Mojo 的品牌：MAX 登场**：虽然 Mojo 是编程语言，但整个平台的品牌是 MAX，即 Modular Accelerated Xecution Platform。
   - 随着新能力的开发，MAX 将包含 GPU、graph API 以及不断演进的功能组件。
- **Mojo 社区会议 #6：录像已发布**：最新的 Mojo 社区会议涵盖了 small buffer 和 string 优化、DuckDB 绑定以及 MAX，现已在 YouTube 上线。
   - 您可以通过以下链接观看录像：[https://youtu.be/6huytcgQgk8](https://youtu.be/6huytcgQgk8)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 发布用于构建健壮 RAG 系统的新 Workflows**：LlamaIndex 最新发布的 Workflows 支持构建高级检索增强生成（RAG）系统，并与 Azure AI Search 和 OpenAI 等 Azure 服务无缝集成。
   - 这种集成利用了自定义数据连接器，从而在这些 Azure 平台内实现流式数据传输和增强功能。
- **Citation Query Engine 获得 Workflow 改造**：一段视频演示展示了使用 LlamaIndex 强大的 Workflows 重建 Citation Query Engine，体现了更健壮、更高效的方法。
   - 该重新实现利用了 chunking 和引用检索文本等技术，能够生成带有清晰来源归属的响应，有效地利用 workflows 和 events 进行引用增强检索。
- **LlamaIndex 的 GraphRAG：寻求生产级应用**：一位社区成员表示希望看到生产就绪的 GraphRAG 应用，并强调需要直观地展示图（graph）如何通过提供 LLM 生成的答案之外的额外上下文来增强检索。
   - 他们自己的应用利用属性图和 RAG 实现来处理聊天问题，旨在结合这些方法，并从其他项目中寻求灵感和最佳实践。
- **揭秘 LlamaIndex Agent 的 Tool Call 预期**：一位用户询问了 LlamaIndex Agent 在 `astream_chat()` 函数中 tool calls 的预期行为，特别是当接收到供 Agent 使用的 tools 时。
   - 他们的具体关注点在于确定最有效的方法：是检测 tool calls 并在发送前缓冲响应，还是继续流式传输 token 并在最终响应中发送 tools。
- **利用聊天历史解锁 LlamaIndex Agent 的潜力**：一位用户寻求关于向 OpenAIAgent 提供消息列表的指导，因为现有方法似乎只接受字符串。
   - 他们探索了对最后一条消息使用 pop-off 策略的可能性，但需要确认处理 Agent 交互的正确用法和最佳实践。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral Large 2 训练进展**：一位成员询问了 **Mistral Large 2** 的训练状态，得到的回复是即使在 **KTO** 中，**inputs 也是被 masked（掩码）的**。
- **KTO Trainer 详解**：一位成员询问 **KTO** 是否支持 **multi-turn**（多轮对话）或 **system prompts**（系统提示词）。
   - 另一位成员引导他们查看 **Hugging Face** 上的 **KTO Trainer 文档**，解释了该训练器的用途和预期的数据集格式。
- **KTO Trainer vs SFT**：**KTO Trainer** 旨在利用二元反馈数据（例如：点赞/点踩）来对齐语言模型。
   - 根据基础模型的质量，在进行 **KTO** 之前可能不需要 **SFT**，这与始终需要它的 **RLHF** 和 **DPO** 不同。
- **SmolLM 模型微调**：一位成员表达了对微调 **SmolLM 130m 或 350m 模型** 的兴趣。
- **使用 llama.cpp 进行 GGUF 转换**：一位用户询问了常用的将模型转换为 GGUF 格式并进行量化的仓库。
   - 回复建议使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 及其相关命令，并指出该过程相对简单。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 类型检查现已可用**：一位成员在 [Tinygrad 仓库](https://github.com/tinygrad/tinygrad) 中添加了 `py.typed` 文件，以确保类型检查在 `tinydreamer` 包中正常工作。
   - 此修复是为了在其机器上使 `mypy` 功能正常运行。
- **编译器书籍推荐**：一位成员寻求关于编译器的优秀书籍推荐，可能是为了寻找构建 Tinygrad 编译器的指导。
   - 对话中未给出具体的书籍推荐。
- **探索 Tinygrad 中的 Cuda.py**：一位成员表示有兴趣查找 [Tinygrad 仓库](https://github.com/tinygrad/tinygrad) 中 `cuda.py` 文件的详细文档或博客。
   - 具体来说，他们希望深入了解该文件在 Tinygrad 中处理 CUDA 加速的作用。
- **Tinygrad 的 ONNX 支持**：一位成员建议在 [Tinygrad 仓库](https://github.com/tinygrad/tinygrad) 中添加 ONNX 支持，旨在让 `tensor.py` 支持大部分 ONNX 特性。
   - 这一添加可能使 Tinygrad 与其他使用 ONNX 的框架实现无缝集成。
- **Tinygrad vs Jax/Flux**：一位成员询问了 Tinygrad 相对于 Jax/Flux 的竞争力，并强调了 Jax 令人印象深刻的能力。
   - 另一位成员发表了看法，认为 Jax 优先考虑使用 Google 的 TPUs 并为 Google 修复 Bug，而支持其他加速器仅仅是为了在迁移到 Google 基础设施之前进行原型设计。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **本地运行 LLMs 需要强大的算力**：一位用户指出，本地运行 LLMs（如 OpenInterpreter (OI) 和 01）需要大量的计算资源，并不适合所有人。
- **用于 OI 和 01 的家庭服务器设置**：一位用户建议使用家庭服务器设置来运行 OpenInterpreter (OI) 和 01。
   - 他们建议将 [Umbrel](https://umbrel.com/) 或 [TuringPi](https://turingpi.com/) 作为潜在的硬件解决方案。
- **分布式设置的三个关键组件**：一位用户详细介绍了 LLM、OI 和 01 分布式设置的三个关键组件。
- **儿童个性化 AI 导师：教育的未来？**：讨论了为儿童提供个性化 AI 导师的想法，特别关注导师的情感和个性化方面。
   - 目标是创建一个 AI 导师可以根据每个孩子的学习风格和性格进行调整的系统。
- **科学教育的 AI 导师**：对话集中在利用 AI 导师教授自然科学的基础原理。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Jala：自动化文本数据标注**：Jala 是一款旨在降低手动文本数据标注成本和时间的新工具，目前正在接受候补名单用户。
   - 这一端到端解决方案使用 AI 支持多种数据格式，包括 CSV, JSON, TXT 和 XML，并提供用于微调各种模型的用户界面。 
- **Jala：多样化的应用**：Jala 可用于各种 NLP, ML 和 AI 相关用途，包括研发数据标注以及自动化内容分类。
   - 用户可以在 [https://heimdall-3jl.pages.dev/pages/jala](https://heimdall-3jl.pages.dev/pages/jala) 注册候补名单以获取早期访问权限。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Capabilities and Risks Demo-Jam Hackathon**：[AI Capabilities and Risks Demo-Jam Hackathon](https://www.apartresearch.com/event/ai-capabilities-and-risks-demo-jam) 将在 7 天后举行！
   - 这是一个展示 AI 风险与潜力的绝佳机会，可以赢取 2,000 美元的奖金，并与 AI safety 专家和爱好者建立联系。
- **Pre-Hackathon Workshop**：黑客松预热工作坊将于明天（UTC 时间 8 月 18 日下午 3 点）举行。
   - 参与者可以与评委和导师见面，并抢先为黑客松酝酿创意。
- **加入 Discord**：加入 Discord 服务器以了解更多关于黑客松的信息，并与其他参与者建立联系。
   - Discord 服务器链接为 [https://discord.gg/A4GZ9UKb?event=1270997649260281968](https://discord.gg/A4GZ9UKb?event=1270997649260281968)。 



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 FusionLabs 插件进展**：适用于 Bubble 的 AI21 FusionLabs 插件开发进展顺利。
   - 该插件将允许用户将 AI21Labs 模型无缝集成到他们的 Bubble 应用程序中。
- **Conversation RAG 推出**：用于试用新发布的 Conversation RAG 的门户网站正在开发中。
   - 这将让用户有机会测试和探索 Conversation RAG 的新功能。
- **Bubble 上的 AI21Labs 模型**：一旦 Conversation RAG 门户上线，将提供一个开发测试链接。
   - 这将向开发者展示 AI21Labs 模型在 Bubble 上的运行方式，使他们能够在 Bubble 环境中实验 AI21Labs 模型的能力。



---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第二部分：分频道详细摘要与链接


{% if medium == 'web' %}

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1273355609873580032)** (148 条消息🔥🔥): 

> - `OpenRouter Aider 使用情况`
> - `模型缓存 (Model Caching)`
> - `Aider 脚本编写`
> - `LLM 缓存`
> - `OpenAI 更新` 


- **Llama 405B 在 OpenRouter 上的使用情况**：一位成员分享了 [Meta 的 LLama 405B 模型](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct) 在 OpenRouter 上的使用量非常显著，本周处理了约 2-3 亿个单词。
   - 他们还提到推理成本似乎很低，特别是 Lepton 提供的 128k 上下文，价格为每 100 万个单词 2.8 美元。
- **关于模型缓存有效性的担忧**：一位成员质疑 Aider 的 5 分钟上下文缓存（context caching）的有效性，考虑到用户的典型响应周转时间。
   - 其他人认为，虽然大部分提示词可能是重复的，但即使是文本中的微小差异也可能导致缓存失效。
- **通过脚本维护 Aider 上下文**：一位成员询问如何通过脚本维护 Aider 上下文，以便进行带有测试的迭代生成。
   - 回复指出，保持 Coder 对象存活是保留内部状态的唯一方法，而使用 Markdown 文件记录聊天历史对于持续聊天并不理想。
- **缓存的好处和用例**：一位成员讨论了缓存系统提示词（system prompts）和用户输入以减少 API 使用量的潜在好处。
   - 他们强调了缓存包含示例、repo maps 和系统提示词的大型提示词的可能性，这将带来显著的成本节约并有望提升模型性能。
- **OpenAI 的 ChatGPT-4o 更新**：一位成员分享称，据报道新的 ChatGPT-4o-latest 模型在代码编辑方面比之前的版本更差。
   - 他们提到，模型更新效果不如旧版本的这种趋势与 OpenAI 专注于优化速度和成本的目标是一致的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/">LiteLLM - Getting Started | liteLLM</a>: https://github.com/BerriAI/litellm</li><li><a href="https://x.com/paulgauthier/status/1823715711254192611?s=46&t=AkDCTtZVFFazuKDknG6fLA">Paul Gauthier (@paulgauthier) 的推文</a>: 新的 chatgpt-4o-latest 在代码编辑方面比之前的 4o 模型稍差。这延续了 OpenAI 在同一模型系列中的每次更新都往往比上一个稍差的趋势。https:...</li><li><a href="https://aider.chat/docs/config/options.html#--restore-chat-history">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Llama 3.1 405B Instruct - API, Providers, Stats</a>: 备受期待的 400B 级 Llama3 来了！凭借 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的 c...</li><li><a href="https://github.com/sao">sao - 概览</a>: product + design @datastax。sao 有 9 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>: 展示使用 Claude 的一些有趣且有效的方法的 Notebook/食谱集合。- anthropics/anthropic-cookbook</li><li><a href="https://github.com/paul-gauthier/aider/pull/685#issuecomment-2291415735">提高 SEARCH/REPLACE 准确性 (已修复) by youknow04 · Pull Request #685 · paul-gauthier/aider</a>: 包括 GPT-4o 在内的 LLM 通常为 SEARCH 块提供非常短的上下文进行匹配。例如：&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt;&amp;lt; SEARCH } ======= // 一些长代码块...</li><li><a href="https://github.com/paul-gauthier/aider/">GitHub - paul-gauthier/aider: aider 是你终端里的 AI 配对编程工具</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/saoudrizwan/claude-dev/commits/main/">Commits · saoudrizwan/claude-dev</a>: 直接在你的 IDE 中的自主软件工程师，能够在每一步获得你许可的情况下创建/编辑文件、执行命令等。- Commits · saoudrizwan/claude-dev
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1273360373541634153)** (17 条消息🔥): 

> - `Llama 3.1`
> - `Grok-2`
> - `Aider Image Support`
> - `OpenRouter`
> - `Prompt Caching` 


- **Llama 3.1 可能是仅次于 DeepSeek 的强力选择**：根据 [OpenRouter 数据](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct)，Llama 3.1 可能是 Aider 表现第二好的模型，仅次于 DeepSeek。
   - 然而，确凿的结果需要直接的 API 使用数据。
- **Grok-2 和 Grok-2 Mini 发布**：Grok-2 和 Grok-2 mini 已在 𝕏 上开启 Beta 测试，并将于本月晚些时候通过企业级 API 提供。
   - Grok-2 被描述为相比 Grok-1.5 的重大进步，在 LMSYS 排行榜上超越了 Claude 3.5 Sonnet 和 GPT-4-Turbo。
- **Aider 支持部分模型的图像功能**：Aider 可以处理 GPT-4o 和 Claude 3.5 Sonnet 等模型的图像文件。
   - 用户可以使用 `/add <image-filename>`、`/clipboard` 或在命令行启动 Aider 时附带图像文件名来将图像添加到对话中。
- **OpenRouter 尚不支持 Prompt Caching**：OpenRouter 不支持 Prompt Caching，这意味着它不会保存 Prompt 以供后续使用。
   - 这意味着即使 Prompt 与之前的相同，每次也都会从头开始处理。
- **Aider 的 .aider 文件需要保留**：Aider 会创建扩展名为 `.aider` 的文件来存储配置信息。
   - 这些文件不应被 Git 忽略，因为它们是 Aider 正常运行所必需的。可以通过在全局 `.gitignore` 中添加 `.aider*` 来全局忽略它们。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/images-urls.html">Images &amp; web pages</a>: 将图像和网页添加到 Aider 编程对话中。</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct).">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1273684584088993947)** (6 条消息): 

> - `Aider UI in IDE`
> - `Aider Conflict Resolution`
> - `Aider Edit Confirmation` 


- **IDE 中的 Aider UI？**：一位用户注意到截图中的 UI 与他们之前看到的 Aider UI 不同。他们询问了 Aider 中的“Approve”按钮。 
   - 另一位用户澄清说，该截图可能是从代码编辑器或 IDE 中截取的，当时正在查看 Aider 的输出文本，该文本模仿了 Git 合并冲突解决语法。编辑器/IDE 识别出该语法为合并冲突，因此可能自动添加了“Approve”按钮。
- **Aider 对编辑的误解**：一位用户根据其对行为的描述，认为 Aider 中的叠加层（overlay）可能无法正常工作。
   - 另一位用户反驳说，截图显示了“Applied edit to...”，这表明 Aider 正确应用了编辑。
- **Aider 编辑确认请求**：一位用户表示支持一项功能请求，即允许用户在 Aider 应用更改之前确认每一项更改。
   - 该用户链接到了 Aider 项目中一个相关的 GitHub Issue ([Add option to force the AI to ask the user to confirm each change before doing it · Issue #649 · paul-gauthier/aider](https://github.com/paul-gauthier/aider/issues/649))，该 Issue 探讨了对此功能的需求。



**提及的链接**: <a href="https://github.com/paul-gauthier/aider/issues/649">Add option to force the AI to ask the user to confirm each change before doing it · Issue #649 · paul-gauthier/aider</a>: 有时需要监督对代码或项目配置的每一次原子级更改。Aider 已经像 diff 一样向我们展示了每一次原子级更改。如果能……

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1273358109850468375)** (166 messages🔥🔥): 

> - `Free SD model deployment` (免费 SD 模型部署)
> - `NFT scams` (NFT 诈骗)
> - `Stable Diffusion on phone` (手机上的 Stable Diffusion)
> - `Free image to video` (免费图生视频)
> - `GPU throttling` (GPU 降频)


- **哪里可以免费部署 SD 模型？**：一位用户询问可以在哪里免费部署他们的 Stable Diffusion 模型。
   - 其他用户建议使用 **Civitai**、**Shakker AI** 或 **Hugging Face**，其中 **Civitai** 是最常用的。
- **NFT 诈骗很普遍**：一名成员分享了关于收到将作品转换为 NFT 请求的担忧，这些报价看起来好得令人难以置信。
   - 其他成员确认这些报价很可能是诈骗，因为真正的公司通常可以证明其合法性。
- **手机上的 Stable Diffusion？**：一位用户询问如何在手机上访问 Stable Diffusion，寻求具有慷慨生成额度的免费选项或广告支持的替代方案。
   - 其他用户指出，在移动设备上运行 Stable Diffusion 需要强大的 GPU，并建议使用 **SD.Next** 等 Web 服务作为潜在解决方案。
- **免费图生视频（Image-to-Video）替代方案**：一名成员征求最佳免费图生视频软件的建议。
   - 另一名成员解释说，GPU 会因发热而自然降频（throttle down），建议使用 **Afterburner** 进行微调，并利用各种 UI 选项中的 **"Generate Forever"** 功能。
- **Flux Discord 服务器？**：几位用户表达了加入 Flux Discord 服务器的兴趣。
   - 一位用户建议当前 Discord 服务器的 **SD3 板块** 已经在某种程度上变成了 Flux 板块，而另一位用户则提议创建一个独立的 Flux Discord。



**Link mentioned**: <a href="https://youtu.be/pDpSpvRiXBU?si=hBs3Wtz7dbj7KUuo">Humanity is Doomed</a>: Asmongold Clips / Asmongold Reacts To: AI catfishing is getting out of handAI Videos By: https://x.com/ai_for_success/status/1821975861698154993https://x.com...

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1273453512445136988)** (1 messages): 

> - `Flavor of the Week Model Removal` (移除 Flavor of the Week 模型)


- **OpenRouter 移除 "Flavor of the Week" 模型**：由于**使用率低**，OpenRouter 计划下周移除 **"Flavor of the Week"** 模型。
   - 该模型可在 [https://openrouter.ai/models/openrouter/flavor-of-the-week](https://openrouter.ai/models/openrouter/flavor-of-the-week) 访问，OpenRouter 正在征求关于该实验的反馈。
- **OpenRouter 征求关于 Flavor of the Week 实验的反馈**：OpenRouter 正在寻求用户关于 **"Flavor of the Week"** 实验的反馈。
   - 他们正在征求关于该实验是否成功以及可以改进之处的意见。



**Link mentioned**: <a href="https://openrouter.ai/models/openrouter/flavor-of-the-week)">Flavor of The Week - API, Providers, Stats</a>: 这是一个每周轮换底层模型的路由模型。它旨在提供一种简单的方法，在保持相同模型 ID 的同时探索新模型的能力。通过 API 运行 Flavor of The Week。

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1273432353204731905)** (3 messages): 

> - `4oSo agent`
> - `OpenRouter`
> - `Claude 3.5 Sonnet`
> - `GPT-4o` 


- **4oSo Agent 结合了 GPT-4o 和 Claude 3.5 Sonnet**：**4oSo** 是一种 "mixture of agents" 方法，结合了 **GPT-4o** 和 **Claude 3.5 Sonnet**。
   - 该方法运行在 **OpenRouter** 上。
- **预先创建线程**：一名成员预先创建了一个线程，以避免频道混乱。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1273355482261618761)** (120 条消息🔥🔥): 

> - `OpenRouter Arena`
> - `OpenRouter LLM API`
> - `OpenRouter LLM Model Availability`
> - `OpenRouter Privacy`
> - `OpenRouter PDF upload` 


- **OpenRouter Arena 在评判 LLM 性能方面存在问题**：一些成员担心 **OpenRouter Arena** 可能不是 **LLM 性能** 的可靠评判标准，原因是缺乏测试方法的明确细节，且来自不同专业水平用户的偏见可能产生影响。
- **OpenRouter 集成 prompt caching**：OpenRouter 将在其 API 中集成 **prompt caching**，这将显著提高性能和成本效率，特别是对于重复性任务和具有一致元素的 prompt。
- **OpenRouter 上线新 LLM 模型：Hermes 3**：**Nous Research** 发布了其 **Hermes 3** 模型（8B, 70B, 405B），现在已在 **OpenRouter** 上可用。
- **OpenRouter 在 PDF 上传方面存在困难**：一些成员报告称，他们无法向 **OpenRouter** 上传 **PDF 文件** 以供模型交互，尽管该平台支持图片上传。
- **OpenRouter API key 集成仍处于 beta 阶段**：一位新成员表示有兴趣在 OpenRouter 上集成他们自己的 **API keys**（如 **DeepSeek** 等服务），该功能目前处于 beta 阶段。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://\"+">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B">NousResearch/Hermes-3-Llama-3.1-405B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: 未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-is-the-cache-lifetime">Prompt Caching (beta) - Anthropic</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-i">OpenRouter</a>: LLM 路由和市场</li><li><a href="https://ai.google.dev/gemini-api/docs/embeddings">未找到标题</a>: 未找到描述</li><li><a href="https://www.voyageai.com/">Voyage</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B">NousResearch/Hermes-3-Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://jina.ai/reader/">Reader API</a>: 读取 URL 或搜索网页，为 LLM 提供更好的依据。</li><li><a href="https://labs.perplexity.ai/">Perplexity Labs</a>: 未找到描述</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free/api">Meta: Llama 3.1 8B Instruct (free) – Run with an API</a>: Meta: Llama 3.1 8B Instruct (free) 的示例代码和 API - Meta 最新级别的模型 (Llama 3.1) 推出了多种尺寸和版本。这个 8B 指令微调版本速度快且高效...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free">Llama 3.1 8B Instruct (free) - API, Providers, Stats</a>: Meta 最新级别的模型 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 8B Instruct (free)</li><li><a href="https://www.reddit.com/r/nousresearch/s/vM7xABhZXt">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter 故障历史</li><li><a href="https://groq.com/">Groq is Fast AI Inference</a>: Groq 的 LPU™ 推理引擎是一个硬件和软件平台，可提供卓越的计算速度、质量和能源效率。Groq 为 AI 提供大规模的云端和本地解决方案...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1273374310589005951)** (7 条消息): 

> - `Meteorological ML Models`
> - `LLM Training Stopping Conditions` 


- **气象学家寻求帮助**：一位用户正在寻找合同工或全职人员，为客户开发气象 ML 模型。
   - 该项目可能会吸引那些喜欢研究图网络（graph networks）的人。
- **LLM 训练停止条件并不复杂**：一位用户询问了预训练 LLM 的停止条件。
   - 目前的情况很简单，最近的一篇论文建议在 80% 的训练过程中保持较高的恒定学习率，然后在剩余的 20% 中衰减至 0%。
- **Cosine Decay 是传统方案**：一位用户描述了传统的 LLM 训练方案。
   - 它涉及在整个预定运行长度内进行一次性的余弦衰减（cosine decay），通常衰减到原始学习率的 10% 左右。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1273386553569312769)** (58 条消息🔥🔥): 

> - `Hyperbolic Embeddings`
> - `Activation Quantization`
> - `Boundary Attention`
> - `NAS` 


- **Hyperbolic Embeddings: 一种表示数据的新方法**: `Hyperbolic Embeddings` 是一种在连续空间中表示数据的技术，因其能够保留图距离和复杂关系（特别是对于层次结构图）而受到关注。
   - 研究人员正在发布 `Hyperbolic Embeddings`，这些嵌入可以进一步集成到知识库补全等应用以及问答等 NLP 任务中。
- **解决语言模型中的 Activation Quantization 问题**: 一篇新的研究论文探讨了语言模型精确量化的挑战，特别关注 `Activation Quantization`。
   - 该论文提出了一种使用 `Quantization-Aware Training (QAT)` 和激活峰度正则化（activation kurtosis regularization）的策略，以解决训练过程中出现的离群通道（outlier channels）问题。
- **Boundary Attention: 一种图像分割的新方法**: 介绍了一种名为 `Boundary Attention` 的新方法，它自下而上地推断未栅格化的边界。
   - 这种轻量级模型能够高精度地推断基于颜色的边界，并使用表示为编码三向分区（three-way partitions）嵌入的输出边界。
- **NAS: 透视神经网络架构搜索的未来**: 成员们讨论了使用 `Neural Architecture Search (NAS)` 优化模型的潜力。
   - 对话探讨了使用 LLM 进行 `NAS` 的可能性，以及 `Repulsive Shells` 等技术在训练中的潜在应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/hyperE/">HyperE</a>: 未找到描述</li><li><a href="https://boundaryattention.github.io/">Boundary Attention</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2404.03605v1">Mitigating the Impact of Outlier Channels for Language Model Quantization with Activation Regularization</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=skEnUjpNN5w">Repulsive Shells - Conference Presentation</a>: 此视频简要概述了 Josua Sassen, Henrik Schumacher, Martin Rumpf 和 Keenan Crane 发表的 SIGGRAPH 2024 论文 &quot;Repulsive Shells&quot;。更多信息...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1273466774351908894)** (57 messages🔥🔥): 

> - `Meta Llama benchmarks`
> - `Benchmark differences`
> - `Eval Harness settings`
> - `Prompt engineering`
> - `Eval harness differences` 


- **Meta Llama benchmarks 差异**：一位用户尝试复现 Meta Llama benchmarks，但得到的结果明显偏低，大约低了 2-3%。
   - 该用户已确认他们使用的是相同的模型，但目前仍无法找出 benchmark 结果出现差异的原因。
- **使用 Eval Harness 进行评估**：讨论集中在如何使用 EleutherAI 的 lm-evaluation-harness 复现 Meta Llama benchmarks。
   - 用户虽然使用了正确的模型和设置，但 benchmark 结果仍然存在差异，这引发了关于 Prompt 格式、评估方法、甚至是评估所用数据可能存在差异的推测。
- **Prompt engineering 的影响**：讨论聚焦于 Prompt engineering 对 benchmark 结果的影响。
   - 用户正在探索不同的 Prompt 格式和设置，以查看是否能提高他们的 benchmark 分数，并认为 Meta Llama 的 Prompt engineering 可能是导致差异的一个因素。
- **Llama 3.1 数据发布**：一位用户提到 Meta 已经发布了 Llama 3.1 的评估数据，但没有发布代码。
   - 这一数据的发布可能有助于理解 Meta 是如何进行评估的，但由于缺乏代码，仍然存在不确定性，并可能导致难以复现其结果。
- **Meta 的评估方法**：一位用户建议 Meta 可能采用了未公开的生成技术，从而导致了 benchmark 的差异。
   - 这种推测凸显了复现大型公司进行的复杂评估所面临的挑战，因为其方法的细节可能并不完全透明。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama3/blob/main/eval_details.md">llama3/eval_details.md at main · meta-llama/llama3</a>: Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账号为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3823cfec41c016378acbcc8616dd1ac92c15edd4/lm_eval/tasks/leaderboard/math/utils.py#L42">lm-evaluation-harness/lm_eval/tasks/leaderboard/math/utils.py at 3823cfec41c016378acbcc8616dd1ac92c15edd4 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1273363142747095070)** (87 messages🔥🔥): 

> - `Anthropic API`
> - `Deepseek`
> - `SB1047` 


- **Anthropic API 推出 Prompt Caching**: Anthropic 刚刚为其 API 推出了 Prompt Caching（提示词缓存），可降低高达 90% 的 API 输入成本并减少高达 80% 的延迟。
   - 该功能通过缓存常用提示词来工作，类似于 Deepseek 的实现，但 Anthropic 的实现更快且更高效。
- **SB 1047 修正案通过**: 加州拨款委员会通过了 SB 1047 修正案，对法案进行了重大修改，特别是影响了对 AI 实验室提交安全测试结果认证的要求。
   - AI 实验室现在将被要求提交概述其安全实践的公开声明，但该法案不再对这些声明追究任何刑事责任。
- **SB 1047 的影响**: 带有这些修正案的 SB 1047 的通过可能会对整个 AI 生态系统产生重大影响，包括欧盟和亚洲。
   - 该法案旨在通过实施保障措施来防止 AI 灾难，但反对者认为这可能会扼杀创新并阻碍 AI 的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1823751966893465630">来自 Alex Albert (@alexalbert__) 的推文</a>: 我们刚刚在 Anthropic API 中推出了 Prompt Caching。它可降低高达 90% 的 API 输入成本并减少高达 80% 的延迟。以下是它的工作原理：</li><li><a href="https://x.com/cfgeek/status/1824192521985200283?s=61">来自 Charles Foster (@CFGeek) 的推文</a>: SB 1047 已通过加州州议会拨款委员会。带有修正案。</li><li><a href="https://x.com/nrmarda/status/1824199043897086375/photo/3">来自 Nik Marda (@nrmarda) 的推文</a>: 这非常值得关注 —— 八名现任加州民主党众议员刚刚表态反对 SB 1047 https://democrats-science.house.gov/imo/media/doc/2024-08-15%20to%20Gov%20Newsom_SB1047.p...</li><li><a href="https://techcrunch.com/2024/08/15/california-weakens-bill-to-prevent-ai-disasters-before-final-vote-taking-advice-from-anthropic/?utm_source=dlvr.it&utm_medium=twitter&guccounter=1&guce_referrer=aHR0cHM6Ly90LmNvL0IxTXZVOE9EN1I&guce_referrer_sig=AQAAAIOWkYBD7o6BSqKChGvu48svlJmEx3EbTCuxoAeHb1caQlByCQtVc7iwLfOTMARko8jkB6WUTobFoVRVWoqMrPTJ3Lg2iJ1_sScRDNCD2RJywWtQFOvfUOJCBn1TVKqIxgXpzRZ2cYJFI6WBpG8Fpe9Wvt_-Rp0p63l1Qlo6F5-f">加州在最终投票前削弱了预防 AI 灾难的法案，采纳了 Anthropic 的建议 | TechCrunch</a>: 加州预防 AI 灾难的法案 SB 1047 遭到了硅谷多方的强烈反对。今天，加州立法者做出了让步</li><li><a href="https://techcrunch.com/2024/08/13/california-ai-bill-sb-1047-aims-to-prevent-ai-disasters-but-silicon-valley-warns-it-will-cause-one/)">加州 AI 法案 SB 1047 旨在防止 AI 灾难，但硅谷警告它将引发灾难 | TechCrunch</a>: SB 1047 引起了硅谷大小玩家的愤怒，包括风险投资家、大型科技行业团体、研究人员和初创公司创始人。这项引入保障措施的加州法案...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1273463528971567145)** (1 messages): 

> - `ACL`
> - `Emily Bender's Talk`
> - `Response to Bender's Talk` 


- **ACL 争议：Bender 的演讲引发辩论**: Emily Bender 在 ACL 会议上的演讲引发了争议，一篇针对所提担忧的[回应已发布](https://gist.github.com/yoavg/f952b7a6cafd2024f44c8bc444a64315#user-content-fn-1-78cb0203d0563bed36d55164d6f1c43f)。
   - 该回应以 [GitHub Gist](https://gist.github.com/yoavg/f952b7a6cafd2024f44c8bc444a64315#user-content-fn-1-78cb0203d0563bed36d55164d6f1c43f) 的形式提供，深入探讨了围绕该演讲的问题，旨在提供平衡的视角。
- **演讲对社区的影响**: 这场演讲在 NLP 社区引发了大量讨论，一些人对 Bender 的担忧表示赞同，而另一些人则持反对意见。
   - 这一争议凸显了负责任的 AI 开发的重要性，以及对伦理考量进行公开对话的必要性。



**提到的链接**: <a href="https://gist.github.com/yoavg/f952b7a6cafd2024f44c8bc444a64315#user-content-fn-1-78cb0203d0563bed36d55164d6f1c43f">acl-presedential-response.md</a>: GitHub Gist: 即时分享代码、笔记和代码片段。

  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1273655448326897674)** (5 messages): 

> - `Trump account` 


- **Trump 的账号依然糟糕**：一位用户发布了一个链接，据推测是前总统 Donald Trump 在社交媒体平台上的账号，并提到这有助于建立对 AI 的理解，即使 AI 仍处于早期阶段。
- ****： 



**提到的链接**：<a href="https://fxtwitter.com/realdonaldtrump/status/1824069681868669045?s=46">来自 Donald J. Trump (@realDonaldTrump) 的推文</a>：未找到描述

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1273355528541700138)** (73 messages🔥🔥): 

> - `Mojo Networking`
> - `Mojo vs Go`
> - `Mojo vs Rust`
> - `Mojo's Future`
> - `Mojo's Branding` 


- **Mojo 的未来：专注于 MAX**：Mojo 目前的重点是 MAX（一个加速计算平台），而不是仅仅专注于网络。
   - 这一决定得到了有力论据的支持，因为 MAX 在计算领域的影响力比在网络领域更大。
- **MAX 的功能与用例**：MAX 是一个用于控制 CPU 以外硬件的库，包括 GPU、DPU，甚至可能是自定义 NIC。
   - 它可以用于高性能网络、Web 服务器以及实现 TCP/IP 协议客户端和服务器等任务。
- **Mojo 的包管理**：Mojo 团队正在考虑如何最好地管理包，倾向于更小、更模块化的包。
   - 他们目前专注于交付 GPU 支持等关键功能，然后再探索将包拆分为更小单元的方案。
- **MAX 在矩阵乘法中的作用**：MAX 旨在为矩阵乘法提供单一实现，该实现可以针对不同的硬件平台编译为最优指令序列。
   - 这涉及使用 MLIR 来表示高级操作，然后根据可用硬件选择优化的 kernel。
- **Mojo 的品牌与身份**：Mojo 已被确立为一种编程语言，但整个平台需要一个品牌，即 MAX —— Modular Accelerated Xecution Platform。
   - MAX 将包含各种组件，包括 GPU、graph API 和其他功能，并将随着新能力的开发而不断演进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://spdk.io/">Storage Performance Development Kit</a>：未找到描述</li><li><a href="https://github.com/odin-lang/Odin/releases/tag/dev-2024-02">Release dev-2024-02 · odin-lang/Odin</a>：作为迈向 Odin 1.0 旅程的一部分，我们正在清理 Odin 提供的包，并对所需内容进行明确划分。新增了一个库集合：base。这意味着...</li><li><a href="https://www.youtube.com/watch?v=6huytcgQgk8">MAX + Mojo Community Meetings #6</a>：这是一段关于 MAX &amp; Mojo 社区会议 #6 的视频。00:00 介绍 00:27 小缓冲区和字符串优化 13:04 Mojo 中的 DuckDB 绑定 23:15 MAX...</li><li><a href="https://github.com/NVIDIA/l2fwd-nv">GitHub - NVIDIA/l2fwd-nv: l2fwd-nv 提供了一个如何利用 NVIDIA GPUDirect RDMA 技术增强 DPDK 网络应用的示例。</a>：l2fwd-nv 提供了一个如何利用 NVIDIA GPUDirect RDMA 技术增强 DPDK 网络应用的示例。 - NVIDIA/l2fwd-nv
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1273360591641120894)** (18 messages🔥): 

> - `Mojo Community Meetings`
> - `Mojo String optimizations`
> - `Mojo String Unicode support`
> - `Mojo String implementation`
> - `Small String Optimization` 


- **Mojo 社区会议 #6 录像已发布**：最新一次 Mojo 社区会议的录像已在 YouTube 上线，涵盖了小缓冲区（small buffer）和字符串优化、Mojo 中的 DuckDB 绑定以及 MAX 等主题。
   - 录像访问地址：[https://youtu.be/6huytcgQgk8](https://youtu.be/6huytcgQgk8)。
- **Mojo 中激进的字符串处理方法**：一位成员分享了他们实现的 Mojo String，该实现通过字段空间窃取（field space stealing）和 Unicode 码点索引支持小字符串优化（Small String Optimization）和 Unicode。
   - 代码地址：[https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae](https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae)，这是一个概念验证（PoC），展示了该方法的可控复杂度，其中 Unicode 码点索引比小字符串优化更具挑战性。
- **Mojo String 的未来：从 List 中解耦**：目前的 Mojo String 实现是基于 List 数据结构的包装，这使得小字符串优化难以实现。
   - 成员建议标准库中的 String 需要从 List 中解耦，以便实现字段空间窃取技术。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/6huytcgQgk8">MAX + Mojo Community Meetings #6</a>：关于 MAX &amp; Mojo 社区会议 #6 的视频。00:00 介绍；00:27 小缓冲区和字符串优化；13:04 Mojo 中的 DuckDB 绑定；23:15 MAX...</li><li><a href="https://gist.github.com/mzaks/78f7d38f63fb234dadb1dae11f2ee3ae">具有小字符串优化和 Unicode 支持的 Mojo String（基于 UTF-8）</a>：具有小字符串优化和 Unicode 支持的 Mojo String（基于 UTF-8） - crazy_string.mojo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1273616466226315264)** (1 messages): 

> - `MAX real-time safety`
> - `C API for MAX`
> - `Real-time audio applications`
> - `Onnxruntime and libtorch limitations` 


- **MAX 是否具备实时安全性？**：一位成员询问了 MAX 引擎的实时安全性（real-time safety），特别是在实时音频应用场景下。
   - 他们强调了目前其他框架（如 onnxruntime 和 libtorch）的局限性，由于缺乏实时安全性，这些框架需要采用非理想的后台线程推理和无锁队列（lock-free queuing）。
- **MAX 的 C API 使用**：该成员目前使用 C++ 框架进行模型部署，但如果 MAX 引擎具备实时安全性，他们有兴趣使用它。
   - 由于目前 MAX 还没有可用的 C++ API，他们计划使用 C API 进行集成。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1273421315004432416)** (2 messages): 

> - `LlamaIndex Workflows`
> - `RAG system`
> - `Azure AI Search`
> - `Azure OpenAI`
> - `Citation Query Engine` 


- **用于 RAG 系统的 LlamaIndex Workflows**：LlamaIndex 新推出的 Workflows 功能支持创建与 Azure 服务集成的强大检索增强生成（RAG）系统。
   - 该工作流涉及为 Azure AI Search 和 Azure OpenAI 实现自定义数据连接器。
- **使用 Workflows 重建 Citation Query Engine**：来自 @ravithejads 的视频演示了如何使用 Workflows 重建引用查询引擎（Citation Query Engine）。
   - 视频涵盖了对检索文本进行分块（chunking）和引用（citing）的技术，创建显示来源的响应，以及使用 Workflows 和事件（events）来实现增强引用的检索。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1273365265991860235)** (62 条消息🔥🔥): 

> - `GraphRAG 应用`
> - `LlamaIndex Agent 工具预期`
> - `LlamaIndex Agent 与聊天历史`
> - `LlamaIndex Embedding 更新`
> - `LlamaIndex GraphRAG 与 Microsoft 的实现` 


- **生产环境中的 GraphRAG 应用**：一位成员询问是否存在任何生产环境中的 GraphRAG 应用，并表示有兴趣了解在生产中如何使用图谱来展示参考资料或除 LLM 生成的答案之外的额外上下文。
   - 他们特别提到自己的应用使用了属性图和针对聊天问题的 RAG 实现，旨在将两者结合，并从其他项目中寻求灵感。
- **LlamaIndex Agent 工具调用预期**：一位用户询问了 LlamaIndex Agent 在 `astream_chat()` 函数中对工具调用的预期，特别是在接收到要使用的工具时。
   - 他们正试图确定最佳方法：是检测工具调用并在使用 `request.tools` 列表发送之前缓冲响应，还是继续流式传输 Token 并在最终响应中发送工具。
- **LlamaIndex Agent 与聊天历史**：一位成员寻求关于向 OpenAIAgent 提供消息列表的指导，因为现有方法似乎只接受字符串。
   - 他们探索了弹出最后一条消息并将其作为字符串提交的可能性，但希望确认该库的正确用法以及与 Agent 协作的适当方法。
- **LlamaIndex Embedding 更新最佳实践**：一位用户询问了在使用摄取管道（ingestion pipeline）为 PDF 文章创建 Embedding 并将其存储在 ChromaDB 中时，更新 Embedding 的最佳实践。
   - 他们特别询问了如何处理同一文件被处理两次，导致插入另一组 Embedding 的情况。
- **LlamaIndex GraphRAG vs. Microsoft 的实现**：一位成员认可 LlamaIndex 的 GraphRAG 实现的强大功能，但质疑其是否与 Microsoft 的原始实现完全一致。
   - 他们很好奇 Microsoft 实现中建议的某些步骤是否未在 LlamaIndex 中实现，并特别询问目前是否实现了局部搜索（local search）和全局搜索（global search）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/api_reference/memory/vector_memory/">Vector memory - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/">Sub Question Query Engine - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1273369546417373277)** (21 条消息🔥): 

> - `Mistral Large 2 训练`
> - `KTO Trainer`
> - `TRL`
> - `SmolLM 模型`
> - `Lambda 集群` 


- **Mistral Large 2 - 训练进展**：一位成员询问了 **Mistral Large 2** 的训练状态。
   - 另一位成员回答说，即使在 **KTO** 中，**输入也是被屏蔽的 (masked)**。
- **了解 KTO Trainer**：一位成员询问 **KTO** 是否支持 **多轮对话 (multi-turn)** 或 **系统提示词 (system prompts)**。
   - 另一位成员提供了 **Hugging Face** 上 **KTO Trainer 文档** 的链接，该文档解释了训练器的用途和预期的数据集格式。
- **KTO 与 SFT 的比较**：**KTO Trainer** 旨在利用二元反馈数据（例如：点赞/点踩）来对齐语言模型。
   - 与总是需要 **SFT** 的 **RLHF** 和 **DPO** 不同，根据基础模型的质量，在 **KTO** 之前可能不需要进行 **SFT**。
- **SmolLM 模型 - 微调**：一位成员表示有兴趣微调 **SmolLM 130m 或 350m 模型**。
   - 虽然没有得到直接回复，但这个对话线程暗示了一个潜在的兴趣领域。
- **Lambda 集群 - 使用**：一位成员询问是否有人具有使用 **Lambda 平台上的集群** 的经验。
   - 未提供具体回复。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/kto_trainer">KTO Trainer</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified">airtrain-ai/fineweb-edu-fortified · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://app.airtrain.ai/dataset/c232b33f-4f4a-49a7-ba55-8167a5f433da/null/1/0)">Airtrain AI | Fineweb-edu-fortified</a>：AI 数据平台
</li>
</ul>

</div>

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1273496114149986304)** (2 messages): 

> - `llama.cpp`
> - `GGUF conversion` 


- **Llama.cpp: GGUF 转换**：一位用户询问通常使用哪个仓库将模型转换为 GGUF 格式并进行量化。
   - 回复建议使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 及其相关命令，并指出该过程相对简单。
- **llama.cpp: GGUF 转换**：一位用户询问通常使用哪个仓库将模型转换为 GGUF 格式并进行量化。
   - 回复建议使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 及其相关命令，并指出该过程相对简单。


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1273439065399627806)** (9 messages🔥): 

> - `Tinygrad Typechecking`
> - `Compiler Book Recommendations`
> - `Cuda.py Documentation`
> - `ONNX Support in Tinygrad` 


- **Tinygrad 类型检查已测试！**：一位成员添加了一个 `py.typed` 文件，以确保 Tinygrad 作为一个包的类型检查能够正常工作。
   - 为了确保其机器上的 `mypy` 能在 `tinydreamer` 中正常运行，需要进行此项修复。
- **征求编译器书籍推荐**：一位成员请求推荐关于编译器的优秀书籍。
- **寻求对 Cuda.py 的深入见解**：一位成员询问是否有关于 Tinygrad 内部 [的 `cuda.py` 文件](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/autogen/cuda.py) 的深入文档或博客。
- **Tensor.py 的 ONNX 集成**：一位成员表示有兴趣将 ONNX 支持添加到 Tinygrad 主仓库中，旨在支持 `tensor.py` 中的大部分 ONNX 特性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/autogen/cuda.py">tinygrad/tinygrad/runtime/autogen/cuda.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/6083">add a single py.typed by geohot · Pull Request #6083 · tinygrad/tinygrad</a>：使 mypy 在 tiny dreamer 中正常工作
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1273444355742634138)** (10 messages🔥): 

> - `Tinygrad vs Jax/Flux`
> - `Tinygrad's CUDA/NV Issues`
> - `Google's TPU dominance`
> - `PyTorch vs Tinygrad Benchmarks` 


- **Tinygrad 与 Jax/Flux 的竞争**：一位成员询问 **Tinygrad** 如何与 **Jax/Flux** 竞争，并指出 **Jax** 看起来非常出色。
   - 另一位成员回应称，他认为 **Jax** 的设计初衷是引导用户租用 **Google 的 TPU** 并为 Google 修复 bug，而**支持其他加速器**仅仅是为了在迁移到 Google 基础设施之前进行原型设计。
- **Tinygrad 的 NV 加速器问题**：一位用户报告称，尽管尝试使用 **3060**，但在安装了 **CUDA** 的 **Ubuntu 22.04** 上，其 **Tinygrad** 设置显示 **NV** 为加速器。
   - 另一位用户澄清说 **NV** 是更底层的 CUDA，并建议如果问题持续存在，将 **CUDA** 环境变量设置为 **1**。
- **PyTorch 在基准测试中表现优于 Tinygrad**：一位用户询问为什么 **PyTorch** 在某些基准测试中仍然优于 **Tinygrad**。
   - 另一位成员建议 **模型实现** 可能是导致差异的原因。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1273382398175739986)** (12 条消息🔥): 

> - `Local LLMs`
> - `Home Server Setup`
> - `Umbrel`
> - `TuringPi`
> - `01 Device` 


- **本地 LLMs：并非易事**：一位用户指出，运行本地 LLM 需要大量的计算资源。
- **用于 OI 和 01 的家庭云设置**：另一位用户分享了关于为 OpenInterpreter (OI) 和 01 使用家庭服务器设置的想法。
- **Umbrel：一种潜在的家庭服务器解决方案**：一位用户建议使用 [Umbrel](https://umbrel.com/) 作为家庭服务器设置的可能方案。
- **TuringPi：另一种家庭服务器选项**：一位用户提到 [TuringPi](https://turingpi.com/) 作为家庭服务器的替代硬件解决方案。
- **分布式设置的组件**：一位用户详细列出了 LLM、OI 和 01 分布式设置中的三个关键组件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://turingpi.com/">Turing Pi 2 集群计算机</a>：Turing Pi 是一款紧凑型 ARM 集群，可在边缘提供安全且可扩展的计算。它旨在为开发者简化 Web 规模的边缘计算。</li><li><a href="https://umbrel.com/">Umbrel - 用于自托管的个人家庭云和操作系统</a>：通过 umbrelOS 将云端带回您的家中——这是一个用于自托管的精美家庭服务器操作系统，以及 Umbrel Home——一个即插即用的家庭服务器。安装 Nextcloud、Jellyfin、Bitcoin 节点以及数百种自托管服务...</li><li><a href="https://docs.openinterpreter.com/guides/os-mode)">简介 - Open Interpreter</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1273406931054235649)** (4 条消息): 

> - `Personalized Education`
> - `AI Tutors`
> - `Science Education` 


- **为儿童提供个性化 AI 导师**：其核心想法是为每个孩子提供一个个性化的 AI 导师。研究表明，如果导师能与孩子建立联系，这种方式将非常有效。
   - 目标是允许对导师进行调整以适应每个孩子，使他们能够通过熟悉的事物以一种情感安抚的方式学习概念。
- **专注于自然科学**：首个要解决的领域是自然科学的基础原理。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

8i8__papillon__8i8d1tyr: https://www.youtube.com/watch?v=gujAar8NZKo
  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1273732009822519306)** (1 条消息): 

> - `Jala` 


- **Jala：端到端数据标注**：Jala 是一款用于自动化文本数据标注的新工具，旨在降低与手动标注相关的成本和时间。
   - 它利用先进的 AI 技术确保高准确性和效率，支持 CSV、JSON、TXT 和 XML 等多种文本数据类型。
- **Jala：核心功能**：Jala 提供自动化文本数据标注、高准确性与效率、支持多种文本数据类型、大规模数据集的可扩展性以及与现有工作流的无缝集成等功能。
   - 它还提供了一个用于微调各种模型的用户界面。
- **Jala：行业用例**：Jala 可应用于多个行业，包括自然语言处理 (NLP)、机器学习和 AI 模型训练、研发数据标注以及自动化内容分类。
   - 它是寻求改进数据标注流程的企业的理想选择。
- **加入 Jala 等候名单**：Jala 团队邀请感兴趣的用户加入他们的等候名单，成为首批体验该工具的人。
   - 在 [https://heimdall-3jl.pages.dev/pages/jala](https://heimdall-3jl.pages.dev/pages/jala) 注册以获取最新动态和早期访问权限。



**提到的链接**：<a href="https://heimdall-3jl.pages.dev/pages/jala">Jala - 数据标注解决方案</a>：未找到描述

  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1273638655507828817)** (1 messages): 

> - `AI Capabilities and Risks Demo-Jam Hackathon`
> - `Pre-Hackathon Workshop`
> - `AI safety`
> - `AI risks and potential` 


- **AI Capabilities and Risks Demo-Jam Hackathon**: [AI Capabilities and Risks Demo-Jam Hackathon](https://www.apartresearch.com/event/ai-capabilities-and-risks-demo-jam) 将在 7 天后举行！
   - 这是一个展示 AI 风险与潜力、赢取 2,000 美元奖金并与 AI safety 专家和爱好者交流的绝佳机会。
- **Pre-Hackathon Workshop**: Hackathon 赛前研讨会将于明天（8 月 18 日）UTC 时间下午 3 点举行。
   - 参与者可以与评委和导师见面，并提前为 Hackathon 构思创意。
- **Join the Discord**: 加入 Discord 服务器以了解更多关于 Hackathon 的信息并与其他参与者建立联系。
   - Discord 服务器链接为 [https://discord.gg/A4GZ9UKb?event=1270997649260281968](https://discord.gg/A4GZ9UKb?event=1270997649260281968)。 


  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1273615921713385473)** (1 messages): 

> - `AI21 FusionLabs plugin for bubble`
> - `Conversation RAG`
> - `AI21Labs models on bubble` 


- **AI21 FusionLabs Plugin Progress**: 适用于 Bubble 的 AI21 FusionLabs 插件开发进展顺利。
   - 该插件将允许用户轻松地将 AI21Labs 模型集成到他们的 Bubble 应用程序中。
- **Conversation RAG Rollout**: 用于试用新发布的 Conversation RAG 的门户网站即将推出。
   - 这将允许用户测试并体验 Conversation RAG 的新功能。
- **AI21Labs Models on Bubble**: 一旦 Conversation RAG 门户上线，将提供一个开发测试链接，以查看 AI21Labs 模型在 Bubble 上的运行情况。
   - 这将使开发者能够在 Bubble 环境中探索 AI21Labs 模型的能力。


  

---



---



---



{% else %}


> 完整的各频道详细内容已针对电子邮件进行了删减。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}