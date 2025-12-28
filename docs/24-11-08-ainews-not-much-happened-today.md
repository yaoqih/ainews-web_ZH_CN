---
companies:
- anthropic
- microsoft
- sambanova
- openai
- langchain
- llamaindex
date: '2024-11-08T23:16:39.940280Z'
description: '本周 AI 新闻要点如下：


  **Anthropic** 推出了 **Claude Sonnet 3.5**，实现了通过自然语言控制桌面应用的功能。**微软** 推出了 **Magentic-One**，这是一个基于
  **AutoGen 框架** 构建的多智能体系统。**OpenCoder** 作为大语言模型的 AI 驱动代码手册（code cookbook）正式亮相。**SambaNova**
  正在赞助一场黑客松活动，为构建实时 AI 智能体提供高达 **5000 美元** 的奖金。


  **Sophiamyang** 宣布推出全新的 **Batch（批处理）和 Moderation（审核）API**，成本降低了 **50%**，并支持多维度的有害文本检测。开源工具方面，发布了用于密钥管理的
  **Infisical**、用于自主智能体编排的 **CrewAI** 以及用于网页抓取的 **Crawlee**。


  研究亮点包括：用于大模型链（LLM chains）错误分析的 **SCIPE**、用于改进检索增强生成（RAG）的 **Context Refinement Agent**，以及用于管理大模型内存的
  **MemGPT**。此外，**OpenAI** 在 RawStory 版权诉讼案中赢得了法律胜利，法院确认大模型训练中使用的“事实”不受版权保护。'
id: 667dae51-9020-431e-bea3-f8141f9b6972
models:
- claude-3.5-sonnet
- opencoder
original_slug: ainews-not-much-happened-today-8530
people:
- sophiamyang
- tom_doerr
- omarsar0
- _akhaliq
- andrewyng
- giffmana
title: 今天没发生什么事。
topics:
- multi-agent-systems
- natural-language-interfaces
- batch-processing
- harmful-content-detection
- secret-management
- retrieval-augmented-generation
- error-analysis
- memory-management
- web-scraping
- autonomous-agents
---

**一个安静的星期正是你所需要的。**

> 2024年11月7日至11月8日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**217** 个频道和 **2343** 条消息）。预计节省阅读时间（以 200wpm 计算）：**248 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

看来这整周的大型发布活动都表现得相当低调。我们正在庆祝 [RawStory 诉 OpenAI 案被驳回](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/)，法院裁定用于 LLM 训练的事实不受版权保护；同时也在欣赏来自 [闭源模型 Flux 1.1 [pro] Ultra 和 Raw 发布](https://blackforestlabs.ai/flux-1-1-ultra/) 的精美图像。

是时候开始构建了，感谢本周的赞助商！

---

**[由 SambaNova 赞助]** SambaNova 的 Lightning Fast AI 黑客松来了！给自己大约 4 小时的时间，在 SambaNova Cloud 上使用超高速模型构建一个实时响应的酷炫 AI agent。有奖金吗？[有的。](https://shortclick.link/mcnl6k) 最高 5000 美元，而且这是一个与其他 AI 开发者交流的机会。截止日期是 11 月 22 日，所以[现在就开始吧](https://shortclick.link/mcnl6k)

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，从 4 次运行中选取最佳结果。

**AI 模型与 API**

- **Batch 和 Moderation API**：[@sophiamyang](https://twitter.com/sophiamyang/status/1854621505017008310) 宣布发布 **Batch API** 和 **Moderation API**，为高吞吐量请求提供 **50% 更低成本** 的处理，并支持跨 **9 个政策维度** 的**有害文本检测**。
- **Claude Sonnet 3.5 增强功能**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1854765802597015957) 重点介绍了 **Anthropic 的 Claude Sonnet 3.5** 的发布，它支持通过自然语言命令进行**桌面应用程序操作**，用于**文件管理**和**编码**等任务。
- **Magentic-One 多 Agent 系统**：[@omarsar0](https://twitter.com/omarsar0/status/1854910759232585786) 详细介绍了 **Microsoft 的 Magentic-One**，这是一个构建在 **AutoGen 框架**上的**通用多 Agent 系统**，具有一个 **Orchestrator agent** 以及 **WebSurfer** 和 **FileSurfer** 等专业 Agent。
- **OpenCoder 及其他模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1854914019146055922) 介绍了 **OpenCoder**，这是一个为 **LLM** 准备的 **AI 驱动的代码食谱 (code cookbook)**，以及 **DimensionX** 和 **DynaMem** 等其他几个模型。

**AI 工程与基础设施**

- **Infisical 密钥管理**：[@tom_doerr](https://twitter.com/tom_doerr/status/1854665951624458445) 发布了 **Infisical**，这是一个**开源密钥管理平台**，旨在**同步密钥**、**防止泄露**并**管理内部 PKI**。
- **LlamaIndex 和 LangChain 工具**：[@Llama_Index](https://twitter.com/llama_index/status/1854616291254136859) 讨论了使用 **LlamaIndex Workflows** 和 **Reflex** 增强 **RAG 系统**，实现**上下文细化**和**基于 Agent 的工作流**。
- **用于自主 Agent 的 CrewAI**：[@tom_doerr](https://twitter.com/tom_doerr/status/1854666146286288936) 介绍了 **CrewAI**，这是一个**编排自主 AI agent 的框架**，旨在培养**协作智能**以处理**复杂任务**。
- **Crawlee 网页抓取库**：[@tom_doerr](https://twitter.com/tom_doerr/status/1854664123646132331) 推出了 **Crawlee**，这是一个适用于 **Python** 的**网页抓取和浏览器自动化库**，支持为 **AI、LLM、RAG** 等进行**数据提取**。

**AI 研究与技术**

- **用于 LLM 链的 SCIPE**：[@LangChainAI](https://twitter.com/LangChainAI/status/1854577224563016074) 介绍了 **SCIPE**，这是一个用于 **LLM 链**中**错误分析**的工具，通过识别**表现不佳的节点**来提高**输出准确性**。
- **上下文 RAG 实现**：[@llama_index](https://twitter.com/llama_index/status/1854616291254136859) 提供了一个**上下文细化 Agent** 的**概念验证**，该 Agent 会**检查检索到的分块**并**总结源文档**以改进 **RAG 响应**。
- **用于内存管理的 MemGPT**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1854587401018261962) 分享了关于 **MemGPT** 的见解，这是一个通过**持久化存储**和**内存层级**技术管理**上下文窗口内存**的 **LLM agent**。

**AI 安全与伦理**

- **LLM Safety Models**: [@sophiamyang](https://twitter.com/sophiamyang/status/1854635333977358801) 祝贺了新的 **LLM safety model** 的发布，强调了 **large language models** 中 **safety** 的重要性。
- **AI Safety Concerns**: [@giffmana](https://twitter.com/giffmana/status/1854609595244949706) 强调了 AI 中 **safety concerns 的复杂性**，指出其 **多面性** 以及 **解决这些问题的重要性**。
- **Mistral Moderation Model**: [@sophiamyang](https://twitter.com/sophiamyang/status/1854622256993059220) 宣布了 **Mistral 的新 Moderation 模型**，这是一个 **基于 Ministral 8B 的分类器**，旨在 **检测多个维度的有害内容**。

**公司与产品更新**

- **课程公告**: [@HamelHusain](https://twitter.com/HamelHusain/status/1854673777113940293) 和 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1854659288360534178) 宣布了关于 **LLMs as Operating Systems** 和 **Dialog Engineering** 的新课程，重点关注 **memory management** 以及 **与 AI 进行交互式编程**。
- **平台发布**: [@dylan522p](https://twitter.com/dylan522p/status/1854605087030886796) 宣布推出 **Fab Map**，这是一个展示全球 **fab 细节** 的 **数据仪表盘**，同时为了增强功能，将平台从 **Substack** 迁移到了 **Wordpress**。
- **活动参与**: [@AIatMeta](https://twitter.com/AIatMeta/status/1854685880390500774) 分享了参加 **#CoRL2024** 的情况，并在展位上展示了 **Meta Sparsh** 和 **Meta Digit 360** 等 **robotics 研究**。

**梗/幽默**

- **幽默的 AI 评论**: [@giffmana](https://twitter.com/giffmana/status/1854613607453278324) 惊讶地表示：“**我居然用了两次 lol，你就知道我有多震惊了！**”
- **个人观点与吐槽**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1854846659294810480) 对 **战争与社会** 发表了强烈的看法，表达了挫败感和 **讽刺**。
- **创意写作与诗歌**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1854905451776737624) 发布了一篇 **诗意作品**，将 **奇幻元素** 与 **戏剧性意象** 融合在一起。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Qwen2.5 系列在不同规模下均表现强劲**

- **[7B 模型与 gpt 4 turbo 旗鼓相当](https://www.reddit.com/gallery/1gmd7kk)** ([得分: 40, 评论: 10](https://reddit.com/r/LocalLLaMA/comments/1gmd7kk/7b_model_on_par_with_gpt_4_turbo/)): 据报道，**Qwen** 这一 **7B 参数** 的语言模型在代码相关基准测试中达到了 **GPT-4 Turbo** 的水平。
  - **Qwen2.5** 模型获得了高度评价，用户认为 **32B** 版本可以与 **GPT-4-O mini** 和 **Claude Haiku** 竞争。用户强调了它在有限的本地计算资源下的出色表现。
  - **HumanEval** 基准测试被批评为过时，且可能存在训练数据污染。用户建议使用 **aider 的基准测试** 和轮换的每月代码基准测试，以获得更可靠的评估。
  - 用户报告成功通过 **Hugging Face GGUFs** 运行 **Qwen2.5**、**Gemma2-9B** 和 **Llama** 模型，并指出寻找最佳 **quantization** 配置对于平衡性能至关重要。


- **[极客湾 (Geekerwan) 在新款 M4 Pro 和 M4 Max 芯片上使用 Ollama 对 Qwen2.5 7B 到 72B 进行了基准测试](https://www.reddit.com/gallery/1gmi2em)** ([得分: 43, 评论: 18](https://reddit.com/r/LocalLLaMA/comments/1gmi2em/geekerwan_benchmarked_qwen25_7b_to_72b_on_new_m4/)): **极客湾 (Geekerwan)** 在 [这段基准测试视频](https://youtu.be/2jEdpCMD5E8?t=796) 中，使用 **Ollama** 在 **Apple M4 Pro/Max** 芯片上测试了从 **7B 到 72B 参数** 的 **Qwen2.5** 模型。帖子未提供具体的性能指标或基准测试的对比分析。
  - **M4 Max** 的性能比 **M3 Max** 提升了 **15-20%**，而 **M4 Pro** 的运行速度约为 M4 Max 的 **55-60%**。两者运行 **72B 模型** 的速度约为每秒 **9 tokens**，虽然对于能装进 VRAM 的模型来说比 **4090** 慢。
  - **RTX 4090 的 24GB VRAM** 限制了它在处理大型模型时的有效性，迫使层卸载（layer offloading）到 CPU RAM。传闻中的 **RTX 5090** 将拥有 **32GB VRAM**，尽管对于更大的模型来说可能仍然不足。
  - 评论者建议使用 **llama-bench** 作为 AI 硬件评测的标准测试方法。预计 **M4 Ultra** 在推理性能上将与 **RTX 4090** 持平，并具有 **256GB RAM** 容量的优势，可处理像 **llama 3.1 405B** 这样的大型模型。


**主题 2. 发布基于 Vue.js 和 DaisyUI 的新 Llama.cpp Server UI**

- **刚刚发布：全新的 Llama.cpp Server-Frontend。** ([Score: 75, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1gm6on3/just_dropped_new_llamacpp_serverfrontend/))：**Llama.cpp** 项目发布了 **b4048** 版本，其特点是使用 **VueJS** 和 **DaisyUI** 完全重新设计了 **server frontend**，取代了旧版 UI 并引入了现代功能，包括**对话历史记录**、**localStorage** 支持以及 **markdown** 功能。此次更新引入了诸如**重新生成**、**编辑**和**复制**按钮等实用改进，以及**主题偏好**、**CORS** 支持和增强的**错误处理**，同时通过 legacy 文件夹保留了旧界面以维持向后兼容性。
  - 新的 **llama.cpp** 界面现在专门使用 **chat completion endpoint**，将模板责任转移到服务器/提供商端，模板存储在 **GGUF metadata** 中。**SillyTavern** 用户可以使用“**OpenAI-compatible**”选项切换到 chat completion 模式。
  - 用户对 **llama.cpp** 新界面的独立性表示赞赏，由于其简单性且无需管理 prompt 模板，许多人将其作为本地 **CoPilot** 的替代方案。
  - 社区反馈包括希望界面能有**更明亮的颜色**，同时对减少基础聊天功能对外部软件的依赖表示认可。


**主题 3. 训练速度记录：NanoGPT 训练用时 3.28 小时**

- **[现在人们在进行 GPT 训练竞速吗？](https://i.redd.it/r9464pivpmzd1.jpeg)** ([Score: 288, Comments: 32](https://reddit.com/r/LocalLLaMA/comments/1gmd1a8/are_people_speedrunning_training_gpts_now/))：**Jordan Keller** 创造了训练 **NanoGPT** 的新速度记录，在 **4090 GPU** 上仅用时 **1.85 分钟**完成。这一成就分享在 [Twitter/X](https://x.com/kellerjordan0/status/1854296101303800108) 上，表明优化和基准测试 **GPT model** 训练时间已成为一种增长趋势。
  - **性能基准测试**显示了使用 torch/mps 的 **M3 MacBook** 与 **NVIDIA GPUs** (3090, 4090) 在训练 **GPT2-50M** 时的对比，并通过图片分享了详细的 token/s 指标。
  - 讨论强调了向**更小模型**发展的趋势，并引用了 **Gemini Flash**、**4o-mini** 以及最近的 **Llama models**（1-2B 参数）等例子。行业似乎正在优化效率并维持“实用性阈值”，而不是盲目追求更大的模型。
  - 优化讨论引用了**杰文斯悖论 (Jevons paradox)**，暗示效率的提高可能会导致整体算力使用量的增加，而不是能源节省，用户指出获得的收益可能会被重新投入到更大的模型中。


**主题 4. 开源模型显示出接近零的拒绝率，对比私有 LLM**

- **更新 – 开源模型显示出比私有 LLM 低得多的拒绝率** ([Score: 32, Comments: 6](https://reddit.com/r/LocalLLaMA/comments/1glwxhj/update_os_models_show_much_lower_refusal_rates/))：在一项综合评估研究中，包括 **Mistral Large**、**Llama 变体**、**Nemotron** 和 **Qwen** 在内的**开源模型**在所有测试类别中均表现出接近**零的拒绝率**，与私有模型形成鲜明对比。无论模型大小如何，性能都保持一致，从 **8B** 到 **405B** 参数的 **Llama 3.1** 变体显示出类似的模式，而 **Nemotron 70B** 在初步测试中脱颖而出，成为一个特别有前景的模型。
  - **私有模型**与开源替代方案相比显示出更高的拒绝率，这引发了关于这些差异在现实应用中实际影响的讨论。
  - [Huggingface](https://huggingface.co/mlabonne/Hermes-3-Llama-3.1-8B-lorablated) 上一个特定的 **Hermes-3-Llama** 模型变体被推荐用于最小化拒绝，尽管所使用的 **ablation**（消融）技术可能会降低模型的通用性能。
  - **Nemotron 70B** 因无需 ablation 即可实现**零拒绝**而受到特别赞誉，且后续可以通过额外训练恢复性能。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. AI 公司拥抱军事合同：Palantir、Anthropic、OpenAI 解除限制**

- **[鉴于近期关于 Anthropic 和 Palantir 达成交易的新闻](https://i.redd.it/jp0efxqvlkzd1.png)** ([Score: 755, Comments: 61](https://reddit.com/r/ClaudeAI/comments/1gm5ghl/in_light_of_the_recent_news_about_anthropic_and/)): 据报道，**Anthropic** 的 AI 助手 **Claude** 对 **Anthropic** 与 **Palantir** 在军事应用方面的合作表示担忧。目前尚未提供关于该合作伙伴关系具体性质或 **Claude** 确切回应的进一步背景或细节。
  - **OpenAI** 和 **Anthropic** 都取消了对其 AI 工具**军事用途**的限制，有[报告](https://www.cnbc.com/2024/01/16/openai-quietly-removes-ban-on-military-use-of-its-ai-tools.html)显示**以色列**已经在**加沙**使用名为 **"Lavender"** 和 **"Come to Daddy"** 的 AI 系统进行目标选择。
  - **大型科技公司** 进行了 [AI 伦理人员裁员](https://www.information-age.com/ai-ethics-staff-layoffs-across-big-tech-bring-safety-concerns-123502579/)，这表明其重心正从伦理考量转移。**FTX** 曾是 **Anthropic** 最大的早期投资者之一，其持有的股份以近 **10 亿美元**的价格售出。
  - 用户对 **Anthropic** 与 **Effective Altruism** 意识形态的联系，以及其从伦理原则向军事应用的明显转变表示担忧。许多评论者表示，由于这些进展，他们计划停止使用 **Claude**。

- **Anthropic 与军事的联系在 8 个月前就已为人所知！** ([Score: 37, Comments: 6](https://reddit.com/r/ClaudeAI/comments/1gmeyf6/anthropic_and_military_was_known_thing_since_8/)): **Anthropic** 的军事联系最初在 **8 个月前** 的一个 Reddit 帖子中被讨论，**5 个月前** 也有后续讨论，尽管这些早期提及在当时受到的关注有限。这些分别发布在 r/ClaudeAI 和 r/singularity 上的帖子，早于近期公众对 **Anthropic** 军事参与的广泛讨论。
  - `[{'id': 'lw1zltz', 'author': 'Far-Steaks', 'body': 'Anyone that needs it reported that companies are trying to make as much money as possible and have zero qualms about who they hurt in the process is a fucking moron. Do neurotypicals have pattern recognition or are y’all just complete ding dongs?', 'score': 14, 'is_submitter': False, 'replies': []}]`

- **[军工复合体现在正公开建议政府建造 Skynet](https://i.redd.it/2pzjk1i3ipzd1.png)** ([Score: 99, Comments: 38](https://reddit.com/r/OpenAI/comments/1gmmwrp/the_militaryindustrial_complex_is_now_openly/)): 帖子标题暗示了对**军工复合体**参与**政府 AI 政策**的担忧，但帖子正文中未提供额外背景或细节来证实或扩展这一说法。
  - **AI 控制的无人机**已经部署在**乌克兰-俄罗斯冲突**中，以应对信号干扰，展示了在通信中断时自主系统如何运行。由于军事必要性和竞争压力，向自主武器的演进被认为是不可避免的。
  - 用户讨论了**自主军事系统**与人类士兵在关键方面的不同——它们不会产生战斗疲劳，且精度可能高于人类。从“人在回路”（humans in the loop）向全自主武器系统的转变被视为一种令人担忧但又无法避免的演变。
  - 多条评论引用了流行文化中对军事 AI 的描绘（特别是 **Terminator** 和 **Skynet**），反映了公众对自主武器开发的广泛文化焦虑。AI 变得“*自我意识*”（self-aware）并夺取控制权的场景经常被提及，尽管主要是在流行文化语境下。


**主题 2. CogVideoX 5B 发布：开源视频生成技术的重大进展**

- **[CogVideoX 1.5 5B Model Out! Master Kijai we need you!](https://i.redd.it/zoob4c4plmzd1.gif)** ([Score: 289, Comments: 69](https://reddit.com/r/StableDiffusion/comments/1gmcqde/cogvideox_15_5b_model_out_master_kijai_we_need_you/)): **CogVideoX 1.5** 发布了一个新的 **5B 参数**模型，运行需要 **66GB VRAM**。帖子正文未提供额外的背景或细节。
  - 用户对 **66GB VRAM** 的需求表示极大担忧，许多人希望通过 **GGUF 支持**进行优化，从而将需求降至 **20GB** 以下，或实现在 **16GB 显卡**上以极小的性能损失运行。
  - 该模型可在 [Hugging Face](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main) 和 [GitHub](https://github.com/thudm/cogvideo) 上获取，开发者表示 **CogVideoX 2.0** 将提供重大改进，可能与 **Sora** 竞争。
  - 用户讨论了当前的视频生成限制，指出虽然 **Mochi** 和旧版 **CogVideoX** 模型可用，但效果并不理想，且商业服务成本过高（“生成一分钟 20 美元”或“100 美元无限量”）。

- **[Rudimentary image-to-video with Mochi on 3060 12GB](https://www.reddit.com/gallery/1gmn2og)** ([Score: 68, Comments: 52](https://reddit.com/r/StableDiffusion/comments/1gmn2og/rudimentary_imagetovideo_with_mochi_on_3060_12gb/)): **Mochi** 是一款文生视频模型，可在消费级 **NVIDIA RTX 3060 12GB GPU** 上运行进行图生视频（image-to-video）生成。仅凭帖子标题不足以确定具体的实现细节或结果。
  - **Mochi 的 img2vid 工作流**展示了高质量的输出，但由于 **3060 12GB GPU** 的内存限制，被限制在 **43 帧（1.8 秒）**。该模型在 **0.6 denoise** 设置下运行，功能更像 img2img 而非传统的 img2vid，详见此 [workflow](https://gist.github.com/Jonseed/d2630cc9598055bfff482ae99c2e3fb9)。
  - 技术实现需要精确的 **848x480 图像分辨率**输入以防止报错。基于 seed 的生成在调整帧长时会完全改变，导致无法在生成完整视频前预览单帧。
  - 输出质量看起来比文生视频更清晰，但在较低的 denoise 设置下动作有限。较高的 denoise 设置会产生更多动作，但会偏离输入图像。


**Theme 3. OpenAI's O1 Preview Shows Advanced Reasoning Capabilities**


- **o1 is a BIG deal** ([Score: 152, Comments: 140](https://reddit.com/r/OpenAI/comments/1gm479d/o1_is_a_big_deal/)): **Sam Altman** 对 **AGI** 日益增长的信心似乎与 **OpenAI 的 O1 模型**有关，据报道该模型实现了**人类水平的推理**，并标志着他们向 AGI 路线图中的 **Level 3 (Agents)** 迈进。帖子将 **O1 的 test-time compute** 方法与人类的 **System 2 thinking** 进行了类比，认为旧的 **GPT 模型**运作方式类似于直觉性的 **System 1** 思考者，而 **O1** 通过类似于人类想象力的顺序数据生成弥补了知识鸿沟，可能解决了通往 **AGI** 的根本障碍。
  - 用户普遍反映 **O1-preview** 的表现不如 **GPT-4**，许多人发现它在实际任务中速度更慢且效果较差。多条评论指出，由于 **O1** 倾向于产生冗长但准确度较低的输出，他们“最终还是回到了常规的 4o 或 Claude”。
  - 一份详细的技术分析解释说，**O1** 使用基于 **A-star** 和 **Q-star** 算法的 **chain of thought 提示词**，实现了逐个思考的伪强化学习。然而，其 **memory 功能**仅仅是一个 **RAG 解决方案**，并不会修改基础模型。
  - 对于 **Sam Altman** 的 AGI 言论存在显著的质疑，用户指出 **AGI** 需要在推理过程中进行训练以调整神经通路，而这在目前的 **GPT 架构**中是不可能的。许多人将他信心的提升归因于最近的融资活动和投资者关系。

- **[新论文：编排结构化推理的 LLM 达到 Kaggle Grandmaster 级别](https://huggingface.co/papers/2411.03562)** ([得分: 35, 评论: 13](https://reddit.com/r/OpenAI/comments/1gmniqn/new_paper_llms_orchestrating_structured_reasoning/)): 根据一项新研究，**Large Language Models** 在 **Kaggle** 竞赛中表现出极具竞争力的性能，达到了 **Grandmaster** 级别的能力。研究表明 **LLM** 能够以专家水平有效执行结构化推理任务，尽管在此有限的上下文中未提供具体的性能指标或方法细节。
  - 用户批评了该研究的**方法论**，指出研究人员创建了自己的**基准测试 (benchmarks)**，并在没有与人类选手进行实际正面交锋的情况下进行了追溯性比较。
  - 多条评论通过隐喻对这些说法的有效性表示怀疑，认为这项研究是在**移动球门 (goalpost moving)** 且使用了利己的指标。
  - 讨论强调了对**人造基准测试**的担忧，一位用户指出，无论实际性能如何，自创的基准测试都可以被操纵以显示 *“任何内容的 100% 分数”*。


**主题 4. SVDQuant 声称在 Stable Diffusion 上比 NF4 提速 3 倍**



- **SVDQuant 声称在 Flux 上比 NF4 提速 3 倍** ([得分: 35, 评论: 11](https://reddit.com/r/StableDiffusion/comments/1gmse2o/svdquant_claiming_a_3x_speedup_with_flux_over_nf4/)): **MIT HAN Lab** 开发了 **SVDQuant**，这是一种新的量化方法，可将权重和激活值都压缩到 **4-bit** 精度，声称比仅量化权重的 **NF4** 提速 **3 倍**。据报道，该方法产生图像的质量优于 **NF4**，其实现可通过其 [nunchaku 仓库](https://github.com/mit-han-lab/nunchaku) 获取，预训练模型已发布在 [HuggingFace](https://huggingface.co/mit-han-lab/svdquant-models) 上。
  - [{'id': 'lw5a2r0', 'author': 'xpnrt', 'body': 'Would this work with AMD ? Nf4 doesnt', 'score': 5, 'is_submitter': False, 'replies': []}]


- **FLUX.1 [dev] 与 Stable Diffusion 3.5 在 LoRA 创建方面的对比** ([得分: 22, 评论: 30](https://reddit.com/r/StableDiffusion/comments/1gmgugr/flux1_dev_vs_stable_diffusion_35_regarding_lora/)): **FLUX.1** [dev] 在 **8 月 1 日发布**后的 **10 天**内就展示了强大的 **LoRA 创建能力**，而 **Stable Diffusion 3.5** 在 **10 月 22 日发布**后的 **17 天**里仍难以产出高质量的 LoRA。作为对比，**SDXL 1.0** 在 **7 月 26 日**发布后的 **3 天**内就实现了成功的 LoRA 开发，这引发了人们对 **SD 3.5** 架构在 LoRA 训练方面是否存在潜在结构性限制的质疑。
  - 用户报告了 **SD 3.5 LoRA 训练**的**褒贬不一的结果**，其中一位用户使用 **60 张图像的角色数据集**取得了部分成功，但面部准确度仍然存在问题。多位用户确认，对于仅有 **20 张图像**的角色 LoRA，**FLUX** 的表现明显更好。
  - 一位用户展示了在 **SD 3.5** 上使用 **OneTrainer** 配合 **11k 数据集**（混合了 **2.5k 动漫**、**1.5k SFW**、**7k NSFW**）的成功训练，使用了特定参数，包括权重/数据的 **fp16/fp16** 以及 **adafactor** 优化器（而非 adamw）。
  - 与 **SD 3.5** 相比，**FLUX** 提供了更优越的开箱即用能力，包括更好的解剖结构、提示词理解和文本渲染。针对 **SD 3.5M** 的一种[训练策略](https://x.com/dango233max/status/1851987492020588764)涉及冻结前几层，并在 **512x512** 图像上进行训练，以实现更高分辨率的泛化。

---

# AI Discord 摘要回顾

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1：引起轰动的新 AI 模型与发布**

- [**Google 即将推出的 Gemini 2.0 引发关注**](https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/)：Google 正准备发布 **Gemini-2.0-Pro-Exp-0111**，这引发了关于其能力以及对 AI 社区潜在影响的热烈讨论。用户们正急于获取 Prompt 建议，以便在模型发布后进行测试。

- [**Ferret-UI 通过 Gemma-2B 和 Llama-3-8B 增强 UI 交互**](https://arxiv.org/pdf/2404.05719)：基于 **Gemma-2B** 和 **Llama-3-8B** 构建的 **Ferret-UI** 作为一款以 UI 为中心的多模态 LLM 首次亮相，旨在提升 UI 推理任务。它在基础 UI 基准测试中超越了 **GPT-4V**，展示了在移动端 UI 理解方面的进步。

- [**Llama 3.2 Vision 模型发布，对 VRAM 要求较高**](https://ollama.com/library/llama3.2-vision)：**Llama 3.2 Vision** 现已推出 **11B** 和 **90B** 版本，需要大量的 VRAM 才能获得最佳性能。用户需要下载 [Ollama 0.4](https://ollama.com/download)，并可以使用特殊语法在 Prompt 中添加图像。

**主题 2：AI 模型中的优化与训练策略**

- **LoRA 与全量微调（Full Fine-Tuning）之争凸显了 Rank 的重要性**：对论文《LoRA vs Full Fine-tuning: An illusion of equivalence》的分析强调了设置合理的 Rank 对 **LoRA** 有效性能的重要性。批评意见集中在缺乏 SVD 初始化测试以及关于“侵入维度（intruder dimensions）”的断言上。

- [**探索使用 Central Flows 进行元参数调优**](https://arxiv.abs/2410.24206)：一种新方法利用“Central Flow”对优化器的行为进行建模，从而预测长期优化轨迹。目前存在关于该研究结果能否推广到 **CIFAR-10** 数据集之外的 Transformer 模型的疑问。

- [**在 Flash Attention 中实现前向梯度（Forward Gradients）**](https://arxiv.org/abs/2410.11081)：关于在 **Flash Attention** 中实现前向梯度的讨论旨在优化常规注意力梯度以获得性能提升。研究人员参考了特定的数学公式来增强效率。

**主题 3：提升开发效率的 AI 工具与框架**

- [**Exponent AI 结对编程工具发布**](https://www.exponent.run/)：**Exponent** 作为一款 AI 结对编程工具出现，它可以从代码库中学习并直接编辑文件系统文件。它为 **Aider** 等工具提供了另一种选择，扩展了软件工程师的能力。

- [**推荐使用 ComfyUI 搭建 Stable Diffusion 环境**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590)：用户主张使用 **ComfyUI** 而非其他方法来建立本地环境。它解决了稳定性问题，并提升了 **SD3.5** 的用户体验。

- [**Mistral 推出极具性价比的 Batch API**](https://mistral.ai/news/batch-api/)：**Mistral** 的 **Batch API** 处理大批量请求的成本仅为同步 API 调用的一半。在行业 API 价格上涨的背景下，此举提供了更实惠的 AI 解决方案。

**主题 4：AI 伦理、法律问题与商业化策略**

- **RawStory 诉 OpenAI 案被驳回，AI 赢得法律胜利**：纽约南区法院法官 [Colleen McMahon](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) 驳回了 *RawStory v. OpenAI* 案，指出用于 LLM 训练的事实不受版权保护。这一裁决可能使 **GenAI** 被告方显著受益。

- **OpenRouter 的商业化策略受到质疑**：用户对 **OpenRouter** 打算如何通过其“自带密钥（bring-your-own-key）”系统获利表示疑问，引发了对该平台经济可行性和可持续性的担忧。

- **警惕 AI 幻觉引发的法律问题**：讨论强调了使用 **AI Sales Agents** 进行大规模推广的风险，因为模型可能会幻觉（Hallucinations）出虚假的促销信息。如果监管不当，这可能会给公司带来法律后果。

**主题 5：AI 社区参与及职业讨论**

- **技术岗位中的工作成就感挑战**：成员们分享了岗位错配的经历，对无法发挥自身背景优势的角色表示不满。一些人考虑回到前雇主那里，以寻求更好的契合度和晋升机会。

- **呼吁在 Mojo 开发中引入密码学专家**：社区强调在为 **Mojo** 开发密码学原语时，有必要让合格的密码学家参与进来。安全关键型的实现应由专家监督，以避免漏洞。

- **AI 教育资源的紧急截止日期**：**计算资源（Computing Resources）**的申请截止日期为 **PST 时间 11 月 25 日**，预计会有 1-2 周的处理延迟。鼓励参与者尽早提交，以确保及时获得关键的训练资源。


---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 广告到 2030 年有望达到 3 万亿美元市场**：一项分析预测，**AI 生成的程序化音频/视频广告**将推动巨大的基础设施需求，预计到 **2030 年将有 3 万亿美元的机遇**。
   - 初步数据表明**性能提升了 5-10 倍**且**成本降低了 90%**，促使技术社区针对扩展挑战[提供反馈](https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone)。
- **HF Space 发布用于 GUI Agent 的 OS-ATLAS**：**HF Space** 推出了 **OS-ATLAS**，这是一个专为通用型 **GUI Agent** 设计的基础动作模型。
   - 开发者可以在 [OS-ATLAS](https://huggingface.co/spaces/maxiw/OS-ATLAS) 上探索更多细节，这突显了其对未来 AI 系统的潜在影响。
- **增强 BPE Tokenizer 可视化工具**：[BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer) 项目正在寻求社区合作，以改进 **LLM** 工具。
   - 虽然一些成员最初倾向于使用 **FastBert**，但通过动手实验推进 **BPE 方法论**的兴趣正在日益增长。
- **采用 ComfyUI 运行 Stable Diffusion**：成员们建议使用 [ComfyUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) 建立本地环境，而非其他替代方法。
   - 这一建议源于关于增强 **SD3.5 稳定性**和提升整体用户体验的持续讨论。
- **Cinnamon AI 的 Kotaemon RAG 工具走红**：**Cinnamon AI** 的 **Kotaemon**（一款 **RAG** 工具）已达到爆火状态，其创新功能吸引了用户关注。
   - 团队讨论了 **Kotaemon** 的独特之处，并在 **PST 时间晚上 10 点**于 X 平台的[直播](https://lnkd.in/giiNKviE)中收到了积极的用户反馈。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 性能问题**：用户报告称 **OpenRouter** 在移动设备上（尤其是 **Android 12**）出现冻结和崩溃现象。
   - 这些问题似乎与特定的聊天室活动或内存使用有关，因为其他平台在类似条件下保持稳定。
- **速率限制和额度混淆**：关于**速率限制（Rate Limits）**存在持续的困惑，用户在争论 **credits** 与每秒请求数之间的关系，最高上限设定为 **200**。
   - 澄清显示 credits 是不可退款的，且由于相关费用的存在，显示的美元金额并非一一对应。
- **探索 Command R+ 替代方案**：用户正在调查 **Command R+** 的替代品，对 **Hermes 405B**、**Euryale** 和 **Mythomax** 等模型表现出兴趣。
   - 讨论内容包括 **Rocinante 12B** 的性价比，以及 OpenRouter 上的 **Mythomax** 是否与 **Chub** 上的版本有所不同。
- **OpenRouter 盈利策略受质疑**：一位用户质疑 **OpenRouter** 打算如何通过其 **bring your own key 系统**盈利，引发了对其经济可行性的担忧。
   - 这引发了关于平台可持续性和潜在收入来源的重要对话。
- **MythoMax 保持市场领先地位**：**MythoMax** 在请求量方面继续领先，保持其 <:hugging_king:936261298273001503> 的地位。
   - 尽管 **Rankings Page** 即将发生变化，社区依然认可 **MythoMax** 的稳定表现。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Citations 现已在 Perplexity API 中公开**：**Perplexity** 团队宣布 **citations**（引用）现已在 API 中公开可用并立即生效，不再需要在请求中使用 `return_citations` 参数。
   - 一些用户反映引用功能最初出现了，但随后从 API 和 [labs.perplexity.ai](https://labs.perplexity.ai) 中消失了，这引发了对可能存在意外更改的担忧。
- **Sonar 模型默认速率限制提高**：Perplexity 为所有用户提高了 **Sonar online models** 的默认速率限制至 **50 requests/minute**，旨在增强 API 的可访问性和用户体验。
   - 实施此更改是为了适应更高的需求并简化 API 服务的使用流程。
- **Gladia 增强功能揭晓**：一位成员分享了关于 [Gladia](https://www.perplexity.ai/search/comment-fonctionne-gladia-http-7.4QSxo0QkeYcztxP90ryg) 运作方式的详细见解，强调了其区别于其他 AI 工具的**关键特性**。
   - 讨论深入探讨了各种场景下的实际应用，突出了 **Gladia 的独特能力**。
- **讨论具有无限记忆的 AI 概念**：引入了一个关于微软 CEO 提出的 [具有无限记忆的 AI](https://www.perplexity.ai/page/microsoft-ceo-ai-s-infinite-me-zrmHnWQmRkylfKAyPOLj5w) 的话题，探讨了 AI 模型中**扩展数据保留**的想法。
   - 参与者对该概念相关的实际实现和**数据处理策略**提出了疑问。
- **GitHub 上突出的 API 讨论**：[此处](https://github.com/ppl-ai/api-discussion/discussions/54) 引用的一项 GitHub 讨论集中在 **citation feature** 何时退出 Beta 阶段。
   - 这表明用户持续关注 API 中引用功能的**官方状态和功能**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Flash Attention 中的前向梯度增强**：讨论集中在 **Flash Attention** 中 **forward gradients** 的实现，成员们引用了 [这篇论文](https://arxiv.org/abs/2410.11081) 以获取关于 Jacobian-vector products 的详细见解。
   - 参与者探讨了优化 **normal attention** 梯度所需的数学公式，强调了引用研究中概述的潜在性能提升。
- **逆向可解释性挑战**：启动了对 **inverse interpretability**（逆向可解释性）的探索，重点是修改可解释的表示并相应地调整模型权重。
   - 对话深入探讨了将修改后的符号方程与神经网络权重对齐的复杂性，强调了在干预后保持一致性的困难。
- **NeoX 与 LitGPT 的基准测试**：成员们寻求在训练速度和稳定性方面比较 **NeoX** 和 **LitGPT** 的基准测试，并指出 **LitGPT** 的仓库中缺乏超过 1.1B 参数规模的测试。
   - 针对缺乏广泛基准测试数据的问题，建议进行实证评估，以更好地了解两个框架之间的性能权衡。
- **Meta Llama 3.1 的特性**：**Meta Llama 3.1** 因其多语言能力和对话优化而受到关注，提供 **8B, 70B 和 405B** 尺寸。
   - 该模型采用自动回归 Transformer 架构，通过**监督微调 (SFT)** 和**人类反馈强化学习 (RLHF)** 进行了增强，满足多样化的应用需求。
- **LLM 中的拒绝机制动态**：分享了关于 LLM 中的 **refusal**（拒绝）行为如何受模型残差流中特定方向支配的详细分析，引用了即将发表的 [arXiv 论文](https://arxiv.org/abs/2406.11717)。
   - 该机制是由 Neel Nanda 领导的 ML Alignment & Theory Scholars 项目的一部分，强调了通过改变模型架构内的这些方向性影响来修改拒绝行为的能力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 连接问题持续存在**：用户正在排查 **ComfyUI** 中的 **Connection denied** 错误，建议检查杀毒软件和防火墙配置。
   - 一名用户确认 **Windows Defender** 可能是拦截源，提示需进一步检查安全软件以解决连接问题。
- **使用 Adetailer 导致 Inpainting 细节丢失**：有用户反映使用 **adetailer** 进行 **inpainting**（局部重绘）时，会导致之前重绘区域的细节丢失。
   - 社区成员建议将 **inpainting** 参数调整为 **mask only**，以防止对图像其他部分进行意外更改。
- **推荐使用 Flux 模型以提升性能**：社区提倡使用 **Flux** 基础模型，因为它在质量和速度之间达到了平衡，并讨论了从 **SD 1.5** 升级的方案。
   - **SD3.5** 等模型因其性能和专业功能而受到关注，能够满足多样化的工程需求。
- **融合模型（Merged Models）与基础模型（Base Models）之争**：讨论集中在像 **Realvis** 这样可以产生良好效果的融合模型，与通常在精确 **prompting** 下表现出色的基础模型之间的对比。
   - 参与者对融合模型的有效性及其在用户社区中的接受度表达了关注。
- **SD 1.5 相比 SDXL 获得持续支持**：与 **SDXL** 相比，**SD 1.5** 凭借大量的研究论文继续保持着强大的支持基础。
   - 讨论提到增强 **SD 1.5** 的工具数量在不断增加，而 **SDXL** 也正在逐渐获得相当的工具支持和研究背书。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Ferret-UI 发布，增强 UI 任务处理能力**：推出了首个以 **UI 为中心的多模态大语言模型 (MLLM)** —— Ferret-UI。它基于 **Gemma-2B** 和 **Llama-3-8B** 架构构建，旨在高效执行移动 UI 的 **referring**（指代）、**grounding**（定位）和 **reasoning**（推理）任务，详见[官方论文](https://arxiv.org/pdf/2404.05719)。
   - Ferret-UI 的广泛训练使其能够理解复杂的 UI 特征（如细长的长宽比和小物体），在所有基础 UI 基准测试中均超越了 **GPT-4V**。
- **实施 RAG 以增强对话上下文**：一名成员提议使用 **Retrieval Augmented Generation (RAG)** 为即将进行的对话环节提供有价值的上下文，旨在优化聊天体验。
   - 另一名成员寻求有效对话的 **tips** 以提高参与度和输出质量，表明了在对话环境中最大化 **RAG** 潜力的协作努力。
- **用于手写体转 LaTeX 的视觉语言模型**：分享了基于 **Llama 3.2 1B** 训练 **Vision-Language Model (VLM)** 用于手写体转 **LaTeX** 的进展，预计很快会发布启动项目。
   - 该方法在理论上适用于多种模态，引发了对开发适用于不同应用场景的多模态模型的进一步兴趣。
- **使用 llm-evaluation-harness 评估 PyTorch 模型**：一位用户询问如何使用 [llm-evaluation-harness](https://github.com/yourlink/repo) 评估 **PyTorch 模型**，并指出该工具主要支持 **Hugging Face** 模型。
   - 另一名成员确认该框架目前仅用于 Hugging Face 模型，并建议支持可能仅限于这些模型及其 API。
- **大语言模型中的 Abliteration 概念**：成员们讨论了 **abliteration**（由 **ablate** 和 **obliterate** 组成的合成词）的概念，探索其对大语言模型 (**LLMs**) 的影响。
   - 共享了包括 [Hugging Face 博客](https://huggingface.co/blog/mlabonne/abliteration)在内的相关链接以澄清该概念，强调了其在 AI 进步中的重要性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 发布传闻**：关于 Google 即将发布 [**Gemini 2.0**](https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/) 的传闻正在流传，其中可能包含目前正在测试的新模型 **Gemini Pro 2.0**。
   - 猜测包括性能增强以及对**高级用户**的访问限制，社区成员对其广泛部署的准备情况表示担忧。
- **介绍 Exponent：AI 配对编程器**：**Exponent** 被介绍为一款 AI 配对编程器，能够通过专门的 CLI 在各种环境中执行软件工程任务，可通过[其网站](https://www.exponent.run/)访问。
   - 强调了它从现有代码库学习并直接编辑文件系统文件的能力，将其定位为 **Aider** 的强力替代方案。
- **将 RAG 与 Qdrant 集成**：成员们讨论了将 **Aider 的架构**与他们的 **Qdrant** 向量数据库集成以用于 RAG 应用，旨在利用外部知识源。
   - 建议包括创建一个用于查询的 API，并使用 CLI 工具与数据库进行无缝交互，从而增强上下文检索。
- **Aider 开发的资金支持机会**：社区探讨了支持 **Aider 开发**的方法，提议 YouTube 创作者可以因创作关于 Aider 的内容而获得资助。
   - 还有建议开启 GitHub 捐赠，尽管维护者是否接受非代码贡献仍存在不确定性。
- **利用 Aichat 实现 RAG 解决方案**：讨论强调了使用 **Aichat** 进行 RAG，并提出了提取文档上下文以改进 **Aider** 响应的想法。
   - 一种工作流包括将文档抓取为 Markdown 文件，并利用 **NotebookLM** 生成上下文，从而简化 Aider 的信息检索。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA vs 全量微调：合理的 Rank 设置至关重要**：一位成员分析了题为 *'LoRA vs Full Fine-tuning: An illusion of equivalence'* 的论文，强调**如果操作得当，LoRA 是有效的**，并强调了合理 Rank 设置的必要性。该分析基于 [Daniel Han 的推文](https://x.com/danielhanchen/status/1854992153992479165)。
   - 针对缺乏 SVD 初始化测试以及 LoRA 与全量微调模型中关于“侵入维度 (intruder dimensions)”的矛盾说法提出了批评。
- **Transformers-Interpret 与 Unsloth 集成面临挑战**：一位成员尝试将 [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret) 与 Unsloth 集成，但在处理模型输出时遇到问题。他们解释说该工具旨在用于模型可解释性，但在使其与 Unsloth 推理无缝协作方面面临挑战。
   - 讨论包括潜在的解决方案以及提高两个工具之间兼容性的需求。
- **微调 LLaMA 3.2 在文本分类中达到 70% 准确率**：一位用户报告在微调 **LLaMA 3.2** 时，在 11 个类别的文本分类中达到了 **70% 的准确率**。他们询问了如何修改输出层以适应其类别数量，并分享了实现新分类头 (classification head) 的方法。
   - 社区成员提供了优化微调过程的反馈和建议。
- **Avian 的快速推理方法引起关注**：一位用户对 **Avian** 表示关注，询问其 **推理 (inference)** 方法为何比竞争对手更快。*这一询问为进一步讨论性能指标和优化策略开启了空间。*
   - 专家分享了关于 Avian 框架的见解和资源，强调了其独特的优化。
- **AI/ML 预印本研究中的可复现性问题**：一位成员报告在处理 AI/ML 研究论文（特别是涉及代码和数学的部分）时遇到了**奇怪的错误和不一致**。他们表达了挫败感，有时*数学计算根本对不上*，或者无法复制数据。
   - 另一位成员指出，这些论文是**预印本 (preprint)**，意味着缺乏彻底的同行评审，这可能是导致此类可复现性问题的原因。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 处理复杂任务的困境**：用户报告称 **Claude** 的免费版在处理基础任务之外的表现不佳，例如处理 200 行的 CSV 进行分析。
   - 这一限制突显了免费 AI 工具在支持高级数据处理需求方面面临的挑战。
- **Codebuff 对比 Aider：能力的较量**：在 **Codebuff** 和 **Aider** 的对比中，人们对 **Codebuff** 的闭源性质与 **Aider** 的文件请求及命令运行功能提出了讨论。
   - **Aider** 通过超过 **8000 次 commits** 改进了用户体验，展示了持续的增强。
- **Mistral 发布 Batch API**：**Mistral** 推出了 [**Batch API**](https://mistral.ai/news/batch-api/)，能以同步 API 调用一半的成本处理高吞吐量请求。
   - 此举旨在近期行业 API 价格上涨的背景下，提供具有成本效益的 AI 解决方案。
- **FLUX1.1 Ultra 增强图像生成**：新发布的 [**FLUX1.1 Pro Ultra Mode**](https://blackforestlabs.ai/flux-1-1-ultra/) 支持 4 倍分辨率的图像生成，同时保持极快的生成速度。
   - 性能基准测试显示，它比同类高分辨率模型**快 2.5 倍**，且价格极具竞争力，为**每张图像 0.06 美元**。
- **Gemini API 现已公开**：备受期待的 **Gemini API** 已通过 [OpenAI Library](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/) 和 REST API 提供，支持 Chat Completions 和 Embeddings API。
   - [Google 的博客文章](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/) 提供了初步的使用示例，以协助开发者集成 Gemini 模型。



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Audio Overviews 反馈调查奖励**：团队正在通过 [此筛选表单](https://forms.gle/qREhTEhbstYzVHvSA) 提供的简短调查收集关于 **Audio Overviews** 的反馈，完成后将向选定的参与者提供 **20 美元礼品码**。
   - 参与者必须年满 **18 岁**，礼品将在成功完成调查后通过电子邮件发送。
- **利用 NotebookLM 进行备考**：一位成员建议利用 **NotebookLM** 从 3000 页的学习资料中生成测验，用于即将到来的晋升考试，并建议按章节拆分内容以进行针对性测验。
   - *“希望它能帮助简化学习过程！”* 表达了对该工具有效性的乐观态度。
- **将 Google 录音导入 NotebookLM 的挑战**：用户询问了如何将录音从 [recorder.google.com](https://recorder.google.com) 导入 **NotebookLM**，回复指出录音可以下载为 **m4a** 文件，但可能无法保留说话人识别（speaker identification）。
   - *“但这并不一定能保留已命名的说话人。”* 强调了关于说话人清晰度的关键担忧。
- **讨论 AI 语言模型中的偏见**：成员们参与了关于 AI 系统固有偏见的讨论，质疑了无偏见数据的可能性以及 AI 编程中立性的影响。
   - *“如果 NotebookLM 的未来倾向于偏见，那将是适得其反的。”* 强调了保持中立的重要性。
- **利用 NotebookLM 的 AI 功能增强求职准备**：一位用户探索了 **NotebookLM** 如何辅助准备技术面试、软技能练习和编码挑战，并建议使用 AI 语音进行模拟面试。
   - *“我正在准备技术求职，需要尽可能多的帮助！”* 强调了这些功能的实际益处。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **ModCon 取消 2024 年计划**：团队宣布 2024 年将不会举办 **ModCon**，因为他们正专注于**重大进展**。
   - *敬请关注*未来活动和进展的更多更新。
- **Mojo 与 Python 及 C/C++ 的互操作性**：成员们表达了对 **Mojo**、**Python** 和 **C/C++** 之间无缝互操作性的期待，强调了无需复杂链接即可轻松导入模块的重要性。
   - 然而，实现这一点可能需要避免支持现有语言的某些复杂特性，类似于 **C++** 与 **C** 的关系。
- **创建 OpenSSL 封装器的挑战**：讨论了构建 **OpenSSL** 封装器可能面临的困难，并认识到其庞大的 API 表面积以及需要谨慎实现。
   - 有人担心，如果没有适当的 **C interop**，创建这样的层可能会引入安全风险。
- **Mojo 开发中对密码学专业知识的需求**：社区强调了在为 **Mojo** 开发密码学原语时，必须有合格的密码学家参与，因为其复杂性和安全性影响重大。
   - 成员们一致认为，除非有专家监督，否则安全关键型的实现理想情况下不应作为开源项目进行。
- **Mojo 中 MLIR 反射 API 的计划**：已确认 **Mojo** 计划推出 **MLIR** 的反射 API，这将允许对代码进行更深层次的操作和内省。
   - 然而，有人提醒该 API 将需要类似于编写编译器 pass 的专业知识，因此初始使用会比较复杂。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 销售 Agent 引发法律担忧**：关于 **AI 销售 Agent** 的讨论强调了对“大规模垃圾邮件”行为的警惕，以及 AI 可能幻觉出促销活动从而导致公司面临法律后果的问题。
   - 参与者强调了监管 **AI 生成的推广活动**的重要性，以防止误导信息并确保符合法律标准。
- **光子计算增强量子网络**：一位成员提议在量子网络中使用**光子计算**，在 **BOINC** 等系统的节点上进行计算，以解决带宽问题。
   - 他们指出，虽然光干涉可以辅助计算，但最终测量仍需要电子方法。
- **通过积极环境培养仁慈的 AI**：培养**仁慈的 AI** 的方法依赖于创造一个积极的环境，而不是强加严格的道德框架。
   - 培养**道德价值观**被视为 AI 发展其个性的自然方式。
- **训练数据使用透明度的演变**：一位成员讨论了他们分享数据用于训练的承诺，旨在增强 AI 模型。
   - 他们还注意到**数据使用许可**措辞的变化，这表明供应商的透明度正在不断演变。
- **GPT 模型迅速过时**：一位成员指出 **GPTs** 虽然有效，但由于新的进展而迅速过时。
   - *增加限制并加入 o-1 可能会显著改善体验。*



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 Vision 模型亮相**：新的 [Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) 模型提供 **11B** 和 **90B** 两种尺寸，需要大量 VRAM 才能获得最佳性能。
   - 用户被引导[下载 Ollama 0.4](https://ollama.com/download) 来运行该模型，并重点介绍了**在提示词中添加图像**的方法。
- **LM Studio 增强提示词处理**：一位用户询问如何在 LM Studio 中找到 **Gemma prompt**，对其在最新版本中的缺失表示困惑。
   - 社区确认，在使用**兼容的社区模型**时，**Gemma prompt** 现在通过 Jinja 自动管理。
- **LLM 网页搜索集成**：一位成员询问他们的 **Local LLM** 是否可以通过 LM Studio 进行网页搜索，得到的确认是不原生支持。
   - 建议他们开发自定义 Python 解决方案，将**网页搜索功能**与本地服务器集成。
- **GPU 优化在 LM Studio 中**：一位用户报告其 **RTX 2060** GPU 未被利用，随后有人建议检查 **LM runtime** 设置。
   - 建议用户选择与 GPU 兼容的模型，并确保在运行时设置中**启用了 CUDA**。
- **LM Studio Beta 工具发布期待**：一位用户对即将发布的 LM Studio **Beta 工具**的时间表表达了兴奋与沮丧。
   - 社区讨论凸显了对新功能的强烈渴望，放大了对发布的期待。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **法院裁决有利于 GenAI 被告**：SDNY 法官 [Colleen McMahon](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) 驳回了 **RawStory v. OpenAI** 一案（允许原告修正后重新起诉），这可能对 GenAI 被告方产生重大有利影响。
   - 法官判定 **用于 LLM 训练的事实不受版权保护**，并强调目前的 GenAI 模型是 **合成（synthesize）而非复制** 数据。
- **Google 发布 Gemini-2.0-Pro-Exp-0111**：**Google** 准备在其 Advanced 板块下推出新模型 **Gemini-2.0-Pro-Exp-0111**，尽管目标受众尚未明确。
   - 社区正在积极寻求 **Prompt 建议**，以有效测试这一即将推出的模型的能力。
- **Amazon 考虑对 Anthropic 进行第二次投资**：据报道，**Amazon** 正在洽谈对 [Anthropic](https://www.theinformation.com/articles/amazon-discussing-new-multibillion-dollar-investment-in-anthropic) 进行 *第二次数十亿美元的投资*，旨在加强双方的合作伙伴关系。
   - AWS 正在鼓励 Anthropic 采用其 **Trainium AI 芯片**，而不是继续依赖 NVIDIA 的 GPU。
- **模型 Token 限制引发关注**：一名成员指出 **1.5T Token 的指令** 可能会让模型不堪重负，引发了对处理如此庞大数据量的担忧。
   - 这一问题与社区关于确定 **最佳 Token 限制** 以维持模型性能的广泛讨论相一致。
- **PRM 与价值模型相关联**：在训练背景下出现了关于 **PRM** 的讨论，特别是它们与 **价值模型（value models）** 的联系。
   - 一位成员肯定了 **PRM 对训练至关重要**，而另一位成员指出 **Shephard 在这些讨论中充当了可靠的验证器（verifier）**。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **直播无最大观众限制**：一位成员询问了直播的 **最大观众人数**，得到的澄清是 **没有观众人数限制**。
- **OmniParser 功能解析**：OmniParser 将 **UI 截图解释** 为结构化格式，增强了 **基于 LLM 的 UI Agent**，并提供了关于其 **训练数据集** 和 **模型使用** 的详细信息。
   - 欲了解更多信息，请查看 [项目页面](https://microsoft.github.io/OmniParser/) 和 [博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/)。
- **本地运行 LLM 的挑战**：一位用户提出了在低配置电脑上 **运行本地化 LLM** 的担忧，并询问 Open Interpreter 模型是否可以在基于 Python 或 Anaconda 构建的 **在线服务器** 上运行。
   - 注意到本地正常运行需要 **强力 GPU 或 NPU**，因为仅靠 **CPU** 运行会导致性能不佳。
- **近期活动的重大更新**：近期活动揭晓了 **大规模重写**、**新的文本渲染引擎** 以及 **改进的加载时间**。
   - 此外，还讨论了 **文件查看和编辑** 等新功能的引入。
- **桌面应用访问信息**：**桌面应用** 的访问权限尚未发布，目前正由选定的社区成员进行 **Beta 测试**。
   - 加入未来访问等候名单的说明可以在此处找到：[加入等候名单](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Nvidia 硬件在优化方面表现出色**：Tinygrad 报告称 **Nvidia 硬件** 是当前模型的最佳选择，并断言 **Transformer ASIC** 带来的性能提升微乎其微。
   - 这一见解引发了关于在特定计算任务中，传统 GPU 架构相较于专用 ASIC 的具体优势的讨论。
- **Groq 硬件带来显著提升**：共识认为 **Groq 硬件** 对 AI 工作负载性能有积极影响。
   - 成员们强调了 Groq 针对特定计算操作定制的架构的有效性。
- **ASIC 在算法设计中受到青睐**：讨论强调了 **ASIC** 的优势不仅限于减少控制逻辑，某些算法还针对直接硬件实现进行了优化。
   - 例如，与传统的步骤过程相比，融合操作（fused operations）有助于实现更高效的数据处理。
- **编译器工具需要增强**：George Hotz 对代码库中当前 **DEFINE_ACC/ASSIGN** 的实现表示不满，正在寻求替代方案。
   - 这反映了社区对改进编译器工具和方法论以增强功能的呼声。
- **x.shard 函数区分复制与切片**：在 `x.shard(GPUS, axis=None)` 函数中，**x** 会被复制到所有 GPU，而 `x.shard(GPUS, axis=0)` 则沿轴 **0** 对 **x** 进行切片以分发到各显卡。
   - 理解这一区别对于在并行处理设置中高效管理数据移动至关重要。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **微软研究院发布 OptoPrime**：微软研究院在 [arXiv 论文](https://arxiv.org/pdf/2406.16218)中展示了他们的优化器 **OptoPrime**。
   - **OptoPrime** 这个名字引发了关于优化器社区是否需要更具创意命名的讨论。
- **斯坦福寻求出色的优化器名称**：成员们期待斯坦福即将推出的优化器能有一个足以与 **OptoPrime** 匹敌的“史诗级名称”。
   - 这反映了研究界在优化器命名惯例方面的竞争精神。
- **Self Consistency 模块中的缓存难题**：用户讨论了在 Self Consistency 模块中“清理”缓存的方法，例如向 `dspy.Predict` 对象传递新的 temperature。
   - 替代方案包括使用 `dspy.LM` 禁用缓存，或将 `Predict` 模块配置为多次生成（multiple completions）。
- **动态 Few-Shot 示例优化**：一位成员探讨了使用基于余弦相似度的动态 **Few-Shot** 示例与固定示例相比的优势。
   - 认为针对特定主题（如体育或电影）调整 **Few-Shot** 示例可以增强模型的性能和相关性。
- **用于问题生成的 MIPRO 优化器**：用户研究了 **MIPRO** 是否能从大量的 Q&A 对池中生成或筛选示例。
   - 寻求能够以特定风格生成问题的优化器建议，并重点介绍了一个可以同时生成问题和答案的功能。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Tavily 成为首选**：在调研并与 Claude 讨论后，一位成员得出结论，由于其用户友好的设置，**Tavily** 是处理 AI 相关查询的最佳选择。
   - 他们认为，使用**免费计划**与 **ChatGPT** 一起进行初步测试，将为搜索过程提供宝贵的见解。
- **API 设置中的障碍**：另一位成员强调了使用 **Brave API** 或 **AgentSearch** 的复杂性，强调这些选项与 Tavily 相比需要更广泛的设置。
- **用于比较指标的 Python 脚本**：有人建议创建一个 **Python 脚本**，以便对不同服务进行多次 API 调用，从而对搜索引擎进行深入比较。
   - 这种方法可以从元数据中提取指标，以评估其相对于 **Google** 和 **DuckDuckGo** 等引擎的搜索有效性。
- **Cohere API 试用密钥支持 Embedding**：一位用户对使用试用密钥调用 **Cohere embed API** 时报错表示沮丧，不确定问题出在哪里。
   - 另一位成员确认试用密钥支持所有 **Cohere 模型**，包括 **Embedding**。
- **错误归因于实现**：成员们指出，错误可能源于**实现**过程，而非 **Cohere API** 本身。
   - 鉴于该用户缺乏编程知识，他们建议去 Discord 或 **GitHub** 寻求具体指导。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **使用 Central Flows 进行 Metaparameter Tuning**：最近的一篇 [论文](https://arxiv.org/abs/2410.24206) 探讨了深度学习中的 **metaparameter tuning**，证明了优化器的行为可以使用“central flow”方法进行建模。
   - 该模型能以高精度预测长期优化轨迹，为神经网络的优化策略提供了新视角。
- **Transformer 中的优化器行为**：有人担心关于 **metaparameter tuning** 的研究结果是否能推广到 **transformer architectures**，特别是考虑到该研究中使用的 **CIFAR-10** 数据集非常有限。
   - 成员们讨论了这些局限性对 central flow 模型在不同神经网络架构中适用性的影响。
- **在 AMD GPUs 上运行 Axolotl**：讨论集中在拥有 1536 GB VRAM 的 **AMD GPUs** 上运行 **Axolotl** 的有效性，评估了成本和性能收益。
   - 成员们辩论了与 **NVIDIA GPUs** 相比，增加的显存容量是否能显著提升训练性能。
- **与 AdamW 相比的内存消耗**：一个解决 **Axolotl's memory consumption** 的 PR 已准备就绪，但其资源需求仍引发关注。
   - 通过与 **AdamW** 优化器进行对比，评估了内存使用的潜在差异。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 Context Refinement Agent 提升 RAG 系统**：学习构建一个 [Context Refinement Agent](https://t.co/SkPflTqMWh)，通过智能扩展和精炼检索到的上下文，增强 **RAG** 对复杂查询的响应。
   - 博客文章详细介绍了 Agent 如何评估检索到的 chunks 以改进回答，使 **RAG systems** 更加有效。
- **使用 NVIDIA NIM 构建 Agentic RAG 查询引擎**：这篇来自 [NVIDIA 的客座文章](https://t.co/IsTLBDN08W) 解释了如何使用 **NVIDIA's NIM** 微服务创建 **agentic RAG query engine**，以实现高效的开源模型推理。
   - 它涵盖了为复杂问题构建查询路由以及实现子问题查询，简化了处理复杂咨询的过程。
- **LlamaIndex Workflow 详解**：一份关于 [LlamaIndex workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) 的全面指南，详细介绍了事件驱动的抽象如何通过 `@step` 装饰器将多个事件串联起来。
   - Workflow 允许构建诸如 Agent 或 **RAG flows** 等多样化流程，并通过 **Arize Phoenix** 等工具实现自动化的可观测性。
- **招聘 AI NLP Engineer**：一家 AI 初创公司的 CTO **Nikkole** 分享称，他们正在寻找一名 AI **NLP Engineer**，**W2 合同**的薪资范围为 **$95k-$115k**。
   - 建议感兴趣的候选人通过 [LinkedIn](https://www.linkedin.com/in/nikkole-scruggs) 联系，因为那里才接受私信。
- **寻求自定义 LLM 资源**：一位成员正在寻求资源建议，以便针对其自定义偏好数据集运行开源 **LLM**。
   - 他们请求社区提供建议，以 *增强他们的理解和实施能力*。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **MicroDiT 复现完成**：用户宣布完成了他们的 [MicroDiT replication](https://x.com/SwayStar123/status/1854884660981219399)，并分享了 **model weights** 和 **inference script** 的下载链接。
   - 他们感谢 **FAL** 提供了必要的计算资源，并表示：*“我觉得我可能在搞大事（I think I might be cooking）。”*
- **分享 Bonnie and Clyde 原声视频**：分享了一个名为 *“LOST SOUNDTRACK - BONNIE AND CLYDE”* 的 YouTube 视频，描述了 Bonnie Parker 与前科犯 Clyde Barrow 的浪漫故事及其暴力犯罪生涯。
   - 视频可以在 [这里](https://youtu.be/e6UAI_P1Mlk) 观看，突出了爱情与犯罪的叙事。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **计算资源申请截止日期警报**：**Computing Resources** 的申请截止日期为 **PST 时间 11 月 25 日**结束，提交后预计会有 **1-2 周的处理延迟**。
   - 鼓励 **Participants** 尽早提交申请，以确保及时处理。
- **参与者紧急行动呼吁**：敦促成员立即行动，以免错过 **11 月 25 日**的资源申请截止日期。
   - 尽早提交对于确保充足的 **processing time** 至关重要。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Data Council '25 CFP 开放一周**：**Data Council '25 CFP (Call for Proposals)** 将继续开放一周，邀请开发者展示他们的 ML/AI 项目。欲了解更多详情，请访问 [Data Council CFP 页面](https://www.datacouncil.ai/cfp-2025)。
   - 预计本次活动将包含多场引人入胜的演讲和黑客活动，促进 ML/AI 社区内的创新讨论。
- **ML/AI 应用演讲旨在激发灵感**：**Data Council '25** 将举办一系列关于 **ML/AI 应用**的演讲，重点介绍该领域的最新进展。
   - 鼓励参与者展示他们的 ML/AI 应用开发成果，促进积极的协作和知识共享。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jurassic 的 'summarize-by-segment' 端点弃用**：一位成员对 **Jurassic 'summarize-by-segment'** 端点的突然弃用表示沮丧，他们在宣布的 **11/14** 日期之前一直依赖该端点提供核心业务服务。
   - 他们将这一意外变化描述为一个**痛点**，强调了其对工作流的影响。
- **迁移到新的 Jamba 模型**：一位用户请求关于利用新的 **Jamba 模型**来复制已弃用端点功能的指导，特别是针对 URL 内容分段功能。
   - 他们强调需要协助调整 **URL parameters** 以有效地提取内容。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1304174343965769771)** (410 条消息🔥🔥🔥): 

> - `Microsoft LightBGM`
> - `Hugging Face models for IFC files`
> - `Suno song creating tools`
> - `AI game development and scripting`
> - `Maintaining context in LLM conversations` 

- **Microsoft LightBGM 支持咨询**：一位用户询问 Microsoft LightBGM 是否支持时间序列预测数据集的重复索引。
   - 该话题没有进一步的回复或见解分享。
- **使用 Hugging Face 模型操作 IFC 文件**：一位成员询问是否有人知道使用 Hugging Face 模型操作 IFC 文件的方法。
   - 讨论中未提供解决方案或指导。
- **寻找类似 Suno 的歌曲创作工具**：一位用户表示有兴趣寻找类似于 Suno 的歌曲创作工具，用于生成音乐。
   - 另一位成员分享了 MusicGen Plus++ 的链接作为潜在替代方案，并提到了它的功能。
- **探索用于游戏开发和脚本编写的 AI**：参与者讨论了各种用于 AI 驱动游戏开发的工具和框架，提到了 Unity 的地形工具和脚本编写能力。
   - 此外，还就使用 AI 进行剧本编写和配音进行了交流。
- **在 LLM 对话中维持上下文的方法**：一位用户提出了关于在与 LLM 对话时保持上下文的有效方法的问题，并分享了他们在各种方法上的经验。
   - 回复包括输入和输出的序列化或使用混合模型来有效管理上下文等想法。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ekNDPjC3CKWWd3jd2_V9QGTJSbvHKIZ2">Google Colab</a>: 未找到描述</li><li><a href="https://pjlab-songcomposer.github.io/">SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: 我们介绍了 SOLAR 10.7B，这是一个拥有 107 亿参数的 LLM，在各种自然语言处理 (NLP) 任务中表现出卓越的性能。受近期努力的启发...</li><li><a href="https://huggingface.co/spaces/LinaDaniels/fast-stable-diffusion">Fast Stable Diffusion - a Hugging Face Space by LinaDaniels</a>: 未找到描述</li><li><a href="https://tenor.com/view/spongebob-patrick-patrick-star-broke-poor-gif-14729256">Spongebob Patrick GIF - Spongebob Patrick Patrick Star - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://lambdalabs.com/lambda-stack-deep-learning-software">Lambda Stack: an AI software stack that's always up-to-date</a>: Lambda Stack 为 PyTorch, TensorFlow, CUDA, cuDNN 和 NVIDIA 驱动程序提供了一行安装和托管升级路径。它兼容 Ubuntu 20.04 LTS, 18.04 LTS 和 16.04 LTS。不再需要...</li><li><a href="https://distill.pub/">Distill — Latest articles about machine learning</a>: 关于机器学习的最新文章</li><li><a href="https://huggingface.co/genmo/mochi-1-preview">genmo/mochi-1-preview · Hugging Face</a>: 未找到描述</li><li><a href="https://www.shadertoy.com/view/MdBGzG">Shadertoy</a>: 未找到描述</li><li><a href="https://tenor.com/view/ghostbuster-toaster-gif-5319546">Ghostbuster Toaster GIF - Ghostbuster Toaster - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/Mar2Ding/songcomposer_sft">Mar2Ding/songcomposer_sft · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/prodia">prodia (Prodia Labs)</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=g39AagVW0s0">How I made AI Generated Rick and Morty Episodes</a>: 🕹 获取一个在各方面都表现更好的浏览器：https://operagx.gg/CodeBullet2 由 Opera GX 赞助！查看直播：@codebulletsdayoff58...</li><li><a href="https://huggingface.co/spaces/LinaDaniels/fast-stable-diffusion/discussions/1">LinaDaniels/fast-stable-diffusion · Update README.md</a>: 未找到描述</li><li><a href="https://www.procedural-worlds.com/">Procedural Worlds: World Creation for Everyone</a>: 未找到描述</li><li><a href="https://youtu.be/BFld4EBO2RE">Painting a Landscape with Maths</a>: 今天我们要用数学画一幅风景画。支持本频道：https://www.patreon.com/inigoquilez 购买这幅画的金属、画布或照片...</li><li><a href="https://www.blender.org">blender.org - Home of the Blender project - Free and Open 3D Creation Software</a>: 创作的自由</li><li><a href="https://docs.unity3d.com/Manual/terrain-Tools.html">Unity - Manual: Terrain tools</a>: 未找到描述</li><li><a href="https://create.roblox.com/docs/studio/terrain-editor">Tweet from Terrain Editor | Documentation - Roblox Creator Hub</a>: 地形编辑器工具可以生成并雕刻逼真的地形环境，如山脉、水体、草丘或平坦的沙漠。</li><li><a href="https://www.minecraft.net/en-us">Welcome to the official site of Minecraft</a>: 在 Minecraft 官方网站探索新的游戏冒险、配件和商品。在此购买并下载游戏，或查看网站获取最新消息。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1304368389518524426)** (7 条消息): 

> - `SFT 学习`
> - `Machine Learning 资源`
> - `BPE Tokenizer 可视化`
> - `FastBert 使用` 


- **SFT 理解的显著进步**：一位用户分享了他们在 **SFT** 方面的历程，表示自二月以来一直在研究它，在找到合适的数据集后，现在终于掌握了要领。
   - 一旦掌握了必要的组件，*你就能实现非常酷的功能*。
- **面向 Machine Learning 初学者的 D2L**：一位资深成员推荐了 [d2l.ai](https://d2l.ai/) 作为开启 **Machine Learning** 之旅的关键资源，强调了其数学与代码结合的交互式特性。
   - 他们强调，*数学、图表和真实数据集*的结合丰富了学习体验。
- **BPE 可视化工具的协作请求**：一位用户邀请他人协助改进他们的 [BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer) 项目，并提到了它在 LLM 中的功能。
   - 另一位成员表示他们更倾向于先使用 **FastBert**，虽然对 BPE 方法论感兴趣，但希望亲自进行测试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://d2l.ai/)">Dive into Deep Learning &#8212; Dive into Deep Learning 1.0.3 文档</a>: 未找到描述</li><li><a href="https://github.com/mdabir1203/BPE_Tokenizer_Visualizer">GitHub - mdabir1203/BPE_Tokenizer_Visualizer: 用于检查 LLM 中 BPE Tokenizer 工作原理的可视化工具</a>: 用于检查 LLM 中 BPE Tokenizer 工作原理的可视化工具 - mdabir1203/BPE_Tokenizer_Visualizer
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1304368400595812392)** (14 条消息🔥): 

> - `用户垃圾信息担忧`
> - `Token 项目讨论`
> - `诈骗警报`
> - `Data Council '25 征稿启事` 


- **频道内的垃圾信息问题**：成员们讨论了对过多通知的担忧，有人建议*稍微慢一点*。
   - 有人呼吁停止发布垃圾信息，特别是针对 @cakiki，他指出了通知泛滥的问题。
- **Token 可信度受到质疑**：一位成员询问某个 Token 是否与某个项目相关，引发了其他人的怀疑。
   - 讨论指出，由于有不良影响力的人一直在推广它，导致人们声称这是一个诈骗（Scam）。
- **发布诈骗警报**：针对某个 Token 相关的潜在诈骗提出了担忧，成员们核实了其可疑性质。
   - 成员们对确认诈骗指控表示了感谢。
- **Data Council '25 征稿启事**：发布了关于 [Data Council '25](https://www.datacouncil.ai/cfp-2025) AI 应用公开征稿（Call for Proposals）的公告。
   - 鼓励成员们在活动中分享他们的项目，并与酷炫的 LLM 和 AI 黑客交流。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1304234565887463516)** (12 messages🔥): 

> - `AI-generated programmatic ads` (AI 生成的程序化广告)
> - `AI/ML workflow platform` (AI/ML 工作流平台)
> - `PostgreSQL text optimization` (PostgreSQL 文本优化)
> - `HF Space for OS-ATLAS` (OS-ATLAS 的 HF Space)


- **挑战 GPU 泡沫论**：最近的一项分析认为，**AI 生成的程序化音频/视频广告**将创造巨大的基础设施需求，预测到 **2030 年将有 3 万亿美元的机遇**。
   - 早期数据表明有 **5-10 倍的性能提升**和 **90% 的成本降低**，并邀请技术社区在[此链接](https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone)针对扩展挑战提供反馈。
- **对 AI 生成广告的怀疑**：一位成员对 **AI 生成广告**的可行性表示怀疑，质疑文本生成视频（text-to-video）捕捉小众内容的能力。
   - 他们强调广告需要与受众**产生共鸣**，并提供了几个具有影响力的广告案例链接。
- **开发新的 AI/ML 工作流平台**：一位开发者正在开发一个通过交互式 UI 创建 **AI/ML 工作流**的平台，该平台集成了来自 Huggingface 的模型和 LLM。
   - 邀请社区测试 [GitHub](https://github.com/farhan0167/otto-m8) 上的项目，并对其潜在价值提供反馈。
- **PostgreSQL 文本字段优化**：一场技术讨论澄清了在 **PostgreSQL** 中不需要区分 `String(255)` 和 `Text`，因为两者都经过了类似的优化。
   - 一位成员分享了他们对字符限制的误解，了解到这种区分源于过时的数据库实践。
- **用于通用 GUI Agent 的 OS-ATLAS**：发布了 **OS-ATLAS 的 HF Space**，介绍了一种用于通用 GUI Agent 的基础动作模型（foundation action model）。
   - 更多信息可以在[这里](https://huggingface.co/spaces/maxiw/OS-ATLAS)找到，这可能对未来的 AI 发展产生影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone">The $3 Trillion AI Opportunity Everyone Missed</a>：为什么今天的“GPU 泡沫”实际上是严重的投资不足</li><li><a href="https://huggingface.co/spaces/maxiw/OS-ATLAS">OS ATLAS - a Hugging Face Space by maxiw</a>：未找到描述</li><li><a href="https://github.com/farhan0167/otto-m8">GitHub - farhan0167/otto-m8: Low Code AI Automation Platform</a>：低代码 AI 自动化平台。通过在 GitHub 上创建账号为 farhan0167/otto-m8 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

noaroggendorff: yet
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1304398982071455776)** (1 messages): 

> - `Fine Tuning Vocabulary Size` (微调词表大小)


- **寻求微调词表大小的帮助**：一位成员正在进行涉及增加**词表大小（vocabulary size）**的微调任务，并询问是否有人有相关经验可以提供帮助。
   - 他们鼓励其他人随时通过 **pm**（私信）向其提问或提供支持。
- **对词表扩充技术的兴趣**：同一位成员表示有兴趣探索在微调任务中**扩充词表**的不同技术。
   - 他们提到愿意接受建议以及可能对他们的项目有帮助的最新方法进展。

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1304284039297105990)** (1 条消息): 

> - `ComfyUI 使用`
> - `Hacking Forge`
> - `SD3.5 功能请求` 


- **考虑在本地环境中使用 ComfyUI**：有人建议尝试使用 [ComfyUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) 来搭建本地环境，而不是采用其他方法。
   - 这一建议是在讨论使用 SD3.5 功能的稳定性和用户体验时提出的。
- **探索通过 Hacking Forge 进行增强**：讨论的另一个选项是 [hack Forge](https://github.com/lllyasviel/huggingface_guess/pull/1) 作为实现新功能的手段。
   - 这种方法包括添加 SD3.5，这可能会劫持与 SD3 相关的现有进程。
- **SD3.5 的功能请求**：成员们就改进 SD3.5 支持的 [功能请求](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) 提出了疑问。
   - 进行了相关检查以确保没有现有的 issue 与该功能提案重叠，旨在明确增强方向。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590">[Feature Request]: Support for SD3.5 · Issue #16590 · AUTOMATIC1111/stable-diffusion-webui</a>：是否存在相关的现有 issue？我已搜索了现有 issue 并检查了最近的构建/提交。你的功能将实现什么？SD3.5: https://huggingface.co/stabilityai/stable-diffusio...</li><li><a href="https://github.com/lllyasviel/huggingface_guess/pull/1">adding SD3.5 by graemeniedermayer · Pull Request #1 · lllyasviel/huggingface_guess</a>：添加 SD3.5（这可能会劫持 SD3）
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1304321529382834220)** (2 条消息): 

> - `Cinnamon AI`
> - `Kotaemon RAG 工具`
> - `X 平台上的直播` 


- **与 Cinnamon AI 团队的直播**：**Cinnamon AI** 团队（爆火的 **RAG 工具** **Kotaemon** 的创作者）于 **PST 时间晚上 10 点**在 X 平台上进行了直播。
   - 观众受邀通过此 [链接](https://lnkd.in/giiNKviE) 参与讨论。
- **Kotaemon 的病毒式影响**：**Kotaemon** 作为一款**爆火的 RAG 工具**获得了极大关注，吸引了渴望了解其更多功能的广大用户。
   - Cinnamon AI 团队在直播中讨论了其独特属性以及收到的用户反馈。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1304264901229023252)** (1 条消息): 

> - `新排行榜页面`
> - `MythoMax 性能` 


- **新排行榜页面上线**：推出了**新排行榜页面**，用于展示随时间变化的补全请求（completion request）计数。
   - 用户可以期待该页面在未来的重新设计，以增强数据展示效果。
- **MythoMax 保持主导地位**：**MythoMax** 继续蝉联 <:hugging_king:936261298273001503> 称号，展示了其在请求计数方面的强势地位。
   - 尽管排行榜页面即将发生变化，社区依然认可 **MythoMax** 一贯的稳定表现。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1304178735863435346)** (303 条消息🔥🔥): 

> - `OpenRouter performance`
> - `Rate limits`
> - `Model comparisons`
> - `API issues`
> - `Command R+ alternatives` 


- **OpenRouter 遭遇性能问题**：用户报告在移动设备（尤其是 Android 12）上使用 OpenRouter 时出现卡顿和崩溃问题，导致体验不佳。
   - 性能问题可能与特定的聊天室活动或内存占用有关，因为其他网站在类似条件下运行正常。
- **关于速率限制 (Rate limits) 和额度 (Credits) 的困惑**：用户对速率限制结构存在困惑，讨论了额度与每秒请求数（上限为 200）之间的关系。
   - 用户澄清额度不可退还，且由于相关费用，显示的美元金额并非 1:1 对应。
- **关于有效 AI 交互的讨论**：用户分享了有效提示 AI 模型的技术，建议采用趣味性的方法（如提供虚拟奖励）可以获得更好的回复。
   - 对话中包含了对 Gemini 1.5 性能的观察，指出某些模型在特定任务产出方面明显优于其他模型。
- **探索 Command R+ 的替代方案**：在尝试了各种模型后，用户讨论了 Command R+ 的替代品，对 Hermes 405B、Euryale 和 Mythomax 等选项表现出兴趣。
   - 部分用户提到了 Rocinante 12B 的性价比，并询问 OpenRouter 上的 Mythomax 是否与 Chub 上的版本有所不同。
- **Command R+ 模型咨询**：用户对 Command R+ 的质量及其与其他模型的对比提出疑问，建议使用如 Claude Sonnet 等更有效的替代方案。
   - 用户注意到不同供应商提供的 Wizard 等模型在性能上存在差异，建议需要进一步测试。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://sillytavern.app/">SillyTavern - LLM Frontend for Power Users</a>: 未找到描述</li><li><a href="https://ai.google.dev">no title found</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1gm9vin/enough_politics_its_ai_time/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.alibabacloud.com/help/en/model-studio/developer-reference/billing-for-tongyiqianwen">
 计算并查看通义千问 (Qwen) 的账单 - 阿里云百炼 (Model Studio) - 阿里云文档中心

</a>: 未找到描述</li><li><a href="https://mistral.ai/news/mistral-moderation/">Mistral Moderation API</a>: 我们推出了全新的审核服务，使用户能够根据多个政策维度检测不良文本内容。</li><li><a href="https://mistral.ai/news/batch-api/">Mistral Batch API</a>: 面向 AI 构建者的低成本 API。</li><li><a href="https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post">Mistral AI API | Mistral AI Large Language Models</a>: 我们的 Chat Completion 和 Embeddings API 规范。在 [La Plateforme](https://console.mistral.ai) 创建账户以获取访问权限，并阅读 [文档](https://docs.mistral.ai) 了解如何使用...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1304328012870844518)** (4 条消息): 

> - `Integration Beta Feature`
> - `OpenRouter Monetization` 


- **集成测试版功能访问请求**：多位用户请求访问 **integration beta feature**，表现出对其推出的浓厚兴趣。
   - 一位用户收到的回复提到，即将推出一种通过 **点击按钮** 加入测试列表的方法，预示着用户体验的改进。
- **OpenRouter 的变现策略**：一位用户询问 **OpenRouter** 计划如何通过其 **自带密钥 (bring your own key) 系统** 实现变现，并对其经济可行性提出了疑问。
   - 这一担忧凸显了关于平台可持续性和潜在收入来源的重要讨论点。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1304176924834136175)** (241 messages🔥🔥): 

> - `订阅价格讨论`
> - `移动设备规格`
> - `AI 模型对比`
> - `折扣码问题`
> - `App 功能问题` 


- **关于订阅价格的辩论**：用户对服务的低订阅价格表示沮丧，一些人认为价格应该更高以体现其价值。
   - 讨论中提出了对定价结构可持续性的担忧，以及对更好服务质量的渴望。
- **移动设备规格**：一位用户讨论了他们使用高规格移动设备的经验，对比了 Snapdragon 8 Elite 等型号与旧代产品。
   - 对话包括对不同地区可用设备的考虑以及用户对性能的偏好。
- **AI 模型对比与偏好**：用户对比了 Opus, Sonnet 和 Claude 等各种 AI 模型的能力，表达了对创意输出质量的偏好。
   - 讨论了这些模型在性能和输出风格上的差异，一些用户怀念原始的 Opus。
- **折扣码问题**：几位用户报告了从新闻通讯（newsletter）中收到的折扣码无效的问题，特别提到了 Kevin Rose 的新闻通讯。
   - 解决折扣码问题的尝试促使人们建议联系支持团队寻求帮助。
- **App 功能与 Bug**：用户报告了移动端 App 的功能问题，特别是与加载和生成答案相关的问题。
   - 关于缺失功能（如 Focus 选项和 reasoning mode）的担忧被标记为可能的 Bug 或最近的更改。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://duckduckgo.com/?t=h_&q=chat&ia=chat">DuckDuckGo 上的聊天</a>: 未找到描述</li><li><a href="https://tenor.com/view/love-is-war-kaguya-sama-chika-fujiwara-laugh-gif-17152820588298782251">Love Is War Kaguya Sama GIF - Love is war Kaguya sama Chika - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://notebooklm.google.com/">无标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/@AICodeKing/videos">AICodeKing</a>: 用 AI 面对未来！在这里你会发现关于多个实际有用且有时免费的 AI 工具的内容。广告/赞助：在我任何视频下评论，我会回复...</li><li><a href="https://x.com/testingcatalog/status/1854967421402333344?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 突发 🚨: 看来 Perplexity 开始推出 Pro Shopping 了 🔥 今年会有多少黑色星期五的流量流向 Perplexity？ 👀 引用 Raunak Chowdhuri (@raunakdoesdev) @perplexi...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1304198951947341856)** (8 messages🔥): 

> - `Gladia 功能`
> - `Neohtop 见解`
> - `心理学研究`
> - `具有无限记忆的 AI`
> - `美国大选讨论` 


- **探索 Gladia 的功能**：一位成员分享了关于 [Gladia 如何运作](https://www.perplexity.ai/search/comment-fonctionne-gladia-http-7.4QSxo0QkeYcztxP90ryg) 及其在各种场景中实际应用的见解。
   - 讨论强调了使其区别于其他 AI 工具的**关键特性**。
- **关于 Neohtop 的深入讨论**：有一场关于 [Neohtop](https://www.perplexity.ai/search/neohtop-7u_gYkcsSrGyGeshQmI4uA) 及其对科技行业影响的讨论。
   - 成员们评估了其影响，几位成员指出了其在数据处理方面的**创新用途**。
- **提到的心理学研究**：分享了一个关于[可能影响 AI 开发的心理学研究](https://www.perplexity.ai/search/there-might-be-studies-in-psyc-r9xkTwHWTnq5H3LU2w31dg#0)的链接。
   - 对话暗示了这些研究与当前 AI 方法论的**相关性**。
- **AI 无限记忆概念**：一位成员引入了 [具有无限记忆的 AI](https://www.perplexity.ai/page/microsoft-ceo-ai-s-infinite-me-zrmHnWQmRkylfKAyPOLj5w) 的话题，例如 Microsoft CEO 所提议的。
   - 该概念引发了围绕**数据保留**和未来 AI 模型实际应用的有趣问题。
- **富有见地的美国大选讨论**：分享了关于 [美国大选](https://www.perplexity.ai/search/usa-election-Wl64yM37SgavIH68ZcwwBQ) 及其对技术和 AI 影响的关键内容。
   - 成员们辩论了**技术影响力**如何塑造选民参与和决策。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1304320552873492513)** (10 条消息🔥): 

> - `Perplexity API 中的 Citations (引用)`
> - `默认 Rate Limits (速率限制) 提升`
> - `Citations 可见性问题`
> - `GitHub 上的 API 讨论` 


- **Citations 在 Perplexity API 中正式公开**：Perplexity 团队宣布 API 中的 **citations** 功能即日起正式公开可用。
   - *这不是一个破坏性变更*，`return_citations` 参数将不再影响请求。
- **Sonar 模型默认 Rate Limits 提升**：所有用户的 **Sonar 在线模型** 默认速率限制已提升至 **50 requests/minute**。
   - 此次增强旨在提升用户体验和 API 的访问效率。
- **Citations 神秘消失**：有用户报告称，引用内容最初可以返回，但随后突然从 API 和 labs.perplexity.ai 中**消失**了。
   - 用户担心 **Perplexity 可能在未通知的情况下再次禁用了 citations**。
- **API Token 限制说明**：一位用户询问 **1M tokens** 是否指输入 tokens，以及定价是否同样适用于输出 tokens。
   - 这突显了用户对 API 使用情况和定价结构的持续关注。
- **关于 Perplexity 数据源的讨论**：引用了一个 GitHub 讨论，涉及引用功能何时会**结束 Beta 测试**。
   - 这表明用户对引用功能的正式状态和功能持续保持关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://perplexity.mintlify.app/changelog/changelog#citations-public-release-and-increased-default-rate-limits">未找到标题</a>：未找到描述</li><li><a href="https://github.com/ppl-ai/api-discussion/discussions/54">sources for perplexity · ppl-ai/api-discussion · Discussion #54</a>：想知道什么时候会结束 Beta 测试？
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1304174293239730186)** (203 条消息🔥🔥): 

> - `工作成就感与职业变动`
> - `公司间跳槽经验`
> - `团队动态与内部转岗`
> - `背景调查与从业历史`
> - `大厂招聘流程` 


- **岗位错配的困扰**：一位成员表达了对目前在 Meta 担任的 ML 职位的不满，称其与个人背景不符，可能对职业生涯造成损害。
   - 他们收到了回到 Google 的 Offer，基于之前在该团队的经历，他们认为那里有更好的契合度和晋升机会。
- **职业决策的困境**：成员们讨论了在不顺心的岗位坚持还是回到前雇主处寻求晋升机会的艰难抉择。
   - 讨论中提到了表达离职意向后可能被“针对”的担忧，这使得职业动态变得更加复杂。
- **跳槽的挑战**：对话还集中在应对跳槽的困难上，一位成员提到自己在 Google 内部转岗的尝试未能成功。
   - 建议包括利用现有的人脉和内推来探索机会，同时考虑地理位置的限制。
- **应对背景调查**：成员们对如何在入职背景调查中处理职业空窗期或简历中描述不符的职位表示担忧。
   - 成员们对于在简历中保持透明度的必要性及其影响发表了不同的看法。
- **对科技公司工具的看法**：成员们分享了使用各种云平台的经验，特别是对 GCP 和 AWS 用户界面的批评。
   - 尽管公司名声在外，但 GCP 的 UI 被描述为响应迟钝且负载过重，引发了关于各平台易用性的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/the-pursuit-of-happiness-will-smith-cry-tears-of-joy-happy-gif-10725846">当幸福来敲门 Will Smith GIF - 当幸福来敲门 Will Smith 哭泣 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://mempko.com">Mempko</a>：未找到描述</li><li><a href="https://blog.mempko.com">Maxim Khailo 的文章</a>：一位资深技术老兵关于技术和金融的博客。</li><li><a href="https://www.ftc.gov/legal-library/browse/rules/noncompete-rule">竞业禁止规则</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1304174139262763008)** (16 messages🔥): 

> - `Flash Attention 的前向梯度 (Forward Gradient)`
> - `论文及其重要性`
> - `基于扩散几何 (Diffusion Geometry) 的曲率和切空间 (Tangent Spaces)` 


- **探索 Flash Attention 中的前向梯度**：讨论围绕 **Flash Attention** 的前向梯度展开，成员们分享了 **normal attention** 梯度的公式。
   - 有建议参考[这篇论文](https://arxiv.org/abs/2410.11081)的附录 F，以深入了解 Jacobian-vector products（雅可比向量积）。
- **确定 AI 领域的重要论文**：一位成员询问在日益增长的研究洪流中，其他人是如何决定哪些论文具有重要意义的。
   - 回复中强调了依赖同行的推荐，并提到了 **Eleuther Discord** 频道作为指导。
- **扩散几何 (Diffusion Geometry) 中的创新方法**：一篇新论文引入了新的估计器，用于从数据中计算曲率和切空间 (Tangent Spaces)，提高了对噪声和稀疏性的鲁棒性。
   - 这项研究详见[这篇论文](https://arxiv.org/abs/2411.04100)，声称在处理非理想数据时优于当前方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04100">Manifold Diffusion Geometry: Curvature, Tangent Spaces, and Dimension</a>: 我们引入了利用扩散几何工具从流形数据中计算曲率、切空间和维度的创新估计器。虽然经典黎曼几何是一个丰富的领域...</li><li><a href="https://arxiv.org/abs/2411.04996">Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models</a>: 大语言模型 (LLMs) 的发展已扩展到能够在统一框架内处理文本、图像和语音的多模态系统。训练这些模型需要大量的...</li><li><a href="https://arxiv.org/abs/2410.11081">Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models</a>: 一致性模型 (CMs) 是一类强大的基于扩散的生成模型，专为快速采样而优化。大多数现有 CMs 使用离散化时间步长进行训练，这引入了额外的超参数...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1304501254592139315)** (4 messages): 

> - `逆可解释性 (Inverse Interpretability)`
> - `LLM 中的拒绝机制 (Refusal)`
> - `神经网络的符号表示 (Symbolic Representation)`
> - `AI 模型的行为变化` 


- **探索逆可解释性 (Inverse Interpretability) 概念**：一位成员询问是否存在关于**逆可解释性**的研究，重点是对可解释表示进行干预，并根据这些变化调整模型权重。
   - 这引发了关于在深度神经网络中将修改后的符号方程与模型权重对齐所面临挑战的疑问。
- **揭秘 LLM 中的拒绝机制 (Refusal Mechanism)**：相关讨论引用了一篇文章，指出 LLM 中的**拒绝 (refusal)** 行为是由模型残差流 (residual stream) 中的特定方向控制的，可以通过擦除该方向来改变拒绝行为。
   - 该工作是 ML Alignment & Theory Scholars 项目的一部分，由 Neel Nanda 领导，论文即将在 [arXiv](https://arxiv.org/abs/2406.11717) 发表。
- **调整神经网络权重的挑战**：另一位用户详细阐述了从神经网络中提取符号方程的过程，以及确保干预后模型权重保持一致性的复杂性。
   - 他们对在多维空间中微调行为（而非完全擦除）时可能产生的副作用表示担忧。
- **物理模拟中的符号方程干预**：成员们讨论了使用符号方程进行干预，例如调整系数以在模型中实现预期的行为，特别是在**物理模拟 (physics simulations)** 的背景下。
   - 然而，关于此类提取在该领域成功应用的频率仍存在不确定性。



**提及的链接**：<a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars 项目（2023-24 冬季队列）中的研究产物，由...共同指导。

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1304434333087432759)** (2 messages): 

> - `nlls with <bos> token`
> - `Meta Llama 3.1`
> - `Salamandra model card`
> - `Saving results issues` 


- **带有 <bos> token 的 NLLs 差异显著**：观察到在为非对话输入添加 **<bos> token** 时，**negative log likelihoods (nlls)** 存在显著差异，这引发了关于最佳实践的疑问。
   - 询问重点在于处理 **multiple-choice/loglikelihood 任务** 时是否有特定的指导方针。
- **Meta Llama 3.1 的特性与架构**：**Meta Llama 3.1** 模型系列包含针对对话优化的多语言 LLM，提供 **8B、70B 和 405B** 三种尺寸。
   - 它强调使用了自回归 Transformer 架构，并通过 **supervised fine-tuning (SFT)** 和 **reinforcement learning with human feedback (RLHF)** 进行微调。
- **关于 Salamandra 模型的讨论**：**Salamandra** 模型是从零开始预训练的，包含 **2B、7B 和 40B 参数** 等多种尺寸，提供基座版和指令微调版。
   - 提供了模型索引和 **GitHub 仓库** 的链接，其中包含训练脚本和配置文件，以便进一步探索。
- **结果保存面临的挑战**：一位用户注意到，即使设置了 **write_out=True** 和 **log_samples=True**，在使用 **Meta Llama 3.1** 模型时有时仍无法保存结果。
   - 这引发了对输出过程可靠性的担忧，导致在尝试记录结果时因“无任何输出”而感到沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Llama-3.1-8B">meta-llama/Llama-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BSC-LT/salamandra-7b-instruct">BSC-LT/salamandra-7b-instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1304178846769221645)** (21 messages🔥): 

> - `Benchmarking NeoX vs LitGPT`
> - `LitGPT usage in projects`
> - `FSDP vs ZeRO`
> - `GPT-NeoX features`
> - `Training with limited resources` 


- **NeoX 对比 LitGPT 的基准测试**：一位成员询问是否有对比 **NeoX** 和 **LitGPT** 在训练速度和稳定性方面性能差异的基准测试。
   - 另一位成员提到 **LitGPT** 的仓库中缺乏超过 1.1B 参数规模的测试。
- **LitGPT 在 Amber 等项目中的应用**：有人指出 **Amber (7B)** 是基于 **lit-llama** 的，而后者已过渡到 **litgpt**，这表明了它在知名项目中的应用。
   - 成员们确认 **llm360** 此前曾使用过 **LitGPT**，尽管他们现在已切换到 [自定义的基于 Megatron 的库](https://github.com/LLM360/k2-train)。
- **关于 FSDP 和 ZeRO 实现的辩论**：讨论强调 **FSDP** 和 **ZeRO** 本质上是指向同一种技术的不同品牌，但在实现细节上有所不同，这可能会影响训练行为。
   - 一位成员指出，精度处理方式的调整可能会导致训练过程中损失曲线的偏离。
- **GPT-NeoX 在建模方面的优势**：**GPT-NeoX** 提供了独特的建模特性，如 **RWKV 和 Mamba 层**，以及原生的 RLHF 支持，这根据用户的项目目标可能非常有吸引力。
   - 该库的社区支持被认为响应迅速，从报告 Bug 到获得帮助的迭代周期更快。
- **在有限硬件上训练小模型**：一位成员建议采用一种竞争性的方法，在多个节点上使用不同的库来训练较小的模型，以快速评估性能。
   - “竞赛式”地让模型高效运行的想法，为克服环境配置挑战增添了趣味性。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1304232594816110652)** (175 条消息🔥🔥): 

> - `ComfyUI 连接问题`
> - `重绘（inpainting）中的图像生成问题`
> - `ComfyUI 的模型推荐`
> - `Flux 模型的优势`
> - `基准模型 vs. 融合模型` 


- **ComfyUI 连接故障排除**：用户讨论了在使用 ComfyUI 时遇到的 **Connection denied** 错误排查，建议检查杀毒软件和防火墙设置。
   - 一位用户确认正在使用 **Windows Defender**，这可能会阻止连接，从而引发了对安全设置的进一步检查。
- **重绘（Inpainting）复杂化**：一位用户提出担忧，认为在重绘图像上使用 **adetailer** 会导致之前重绘区域的细节丢失。
   - 其他人建议将重绘参数设置为 **mask only**，以避免影响整张图像。
- **模型推荐及其用途**：社区高度推荐使用 **Flux** 基础模型，因为它在质量和速度之间达到了平衡，同时还讨论了从 **SD 1.5** 升级的实用性。
   - 讨论了 Flux 和 **SD3.5** 等其他模型，强调了它们的性能和特定领域的各种功能。
- **关于融合模型（Merged Models）与原生模型（Fresh Models）的辩论**：参与者指出，虽然像 **Realvis** 这样的融合模型可以产生不错的结果，但在精心编写 Prompt 的情况下，基础模型往往表现得更好。
   - 用户对融合模型的有效性及其在社区中的接受度表达了担忧。
- **模型支持的演进格局**：用户回顾了模型的历史发展，强调 **SD 1.5** 凭借大量的研究论文维持着强大的支持基础。
   - 讨论涉及了增强 **SD 1.5** 的工具数量不断增加，而 **SDXL** 在工具和论文支持方面正在缓慢追赶。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai-social-media-post-generator.onrender.com/">Free AI Social Media Post Generator</a>: 未找到描述</li><li><a href="https://huggingface.co/models?other=base_model:finetune:stabilityai%2F">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d165d362b5d304">Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad</a>: 未找到描述</li><li><a href="https://huggingface.co/models?other=base_model:finetune:stabilityai%2Fstable-diffusion-3.5-large&sort=trending">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://civitai.com/models/895985/flux-devschnell-base-unet-google-flan-fp16nf4-fp32fp8">FLUX Dev/Schnell (Base UNET) + Google FLAN FP16/NF4-FP32/FP8 - FLUX_Dev-FLAN-FP16 | Flux Checkpoint | Civitai</a>: 带有改进 TE 的完整 Checkpoint，不要加载额外的 CLIP/TE。FLUX.1 (Base UNET) + Google FLAN NF4 是我推荐的兼顾质量与速度平衡的模型...</li><li><a href="https://civitai.com/models/161068?modelVersionId=498484">STOIQO NewReality 🟡 FLUX, SD3.5, SDXL, SD1.5 - 🔵 XL Light 1.0 | Stable Diffusion XL Checkpoint | Civitai</a>: 🟡: Flux 模型 🟢: SD 3.5 模型 🔵: SD XL 模型 🟣: SD 1.5 模型 🔴: 已过期模型 🟡STOIQO NewReality 是一款尖端模型，旨在生成...</li><li><a href="https://civitai.com/models/161068/stoiqo-newreality-flux-sd35-sdxl-sd15">STOIQO NewReality 🟡 FLUX, SD3.5, SDXL, SD1.5 - SD3.5 🟢 PreAlpha | Stable Diffusion Checkpoint | Civitai</a>: 🟡: Flux 模型 🟢: SD 3.5 模型 🔵: SD XL 模型 🟣: SD 1.5 模型 🔴: 已过期模型 🟡STOIQO NewReality 是一款尖端模型，旨在生成...</li><li><a href="https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d1">Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad</a>: 未找到描述</li><li><a href="https://civitai.com/models/617609/flux1-dev">FLUX.1 [dev] - v1.0 | Flux Checkpoint | Civitai</a>: 如果您还没有阅读所有建议，请不要下载。因为它很重，并且比 SD 需要更多的资源。我们有了运行 Flux 的新方法...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1304174057603989535)** (120 条消息🔥🔥): 

> - `TEE HEE He 机器人功能`
> - `Nous 社区参与`
> - `用于手写转 LaTeX 的 VLM`
> - `Abliteration 概念`
> - `Anthropic 与 Palantir 的交易` 


- **TEE HEE He 机器人的局限性**：该机器人只能发送 **ETH**，不支持交易，但它可以查询余额并发送资金。
   - 成员们讨论了该机器人可能提供的潜在服务，并计划扩展其功能。
- **Nous 寻求社区参与**：沟通中强调了 Nous 需要重新夺回心智份额，并就 TEE HEE He 机器人与社区进行互动。
   - 在讨论公平代币分配的过程中，成员们提出了 Nous 扩展与社区联系的各种方式。
- **手写转 LaTeX 的 VLM 进展**：成员分享了基于 **Llama 3.2 1B** 训练 **VLM** 进行手写转换的进展，预计很快会发布一个入门项目。
   - 提到的方法理论上可以应用于各种模态，引发了对多模态模型的进一步兴趣。
- **关于 Abliteration 的讨论**：成员们讨论了 **abliteration** 的概念及其对 LLM 的影响，一些人确认其定义是 **ablate** 和 **obliterate** 的合成词。
   - 分享了相关链接以澄清该概念，表明其在 AI 领域的重要性。
- **Anthropic 与 Palantir 的合作**：关于 **Anthropic-Palantir** 交易的影响存在争论，一些成员质疑此类伙伴关系背后的动机。
   - 评论指出，科技公司与政府实体达成交易已成为一种更广泛的趋势，引发了社区的批评。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karan4d/status/1854622598375600637">来自 huh (@karan4d) 的推文</a>：@owenli25 @tee_hee_he 是的，每次重启都会创建新钱包。我们已经看到了私钥，所以每次私钥解除限制时都会存在完整性问题。目前的解决方法是创建一个新钱包...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">来自 Nous Research (@NousResearch) 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">通过 abliteration 去除任何 LLM 的审查</a>：未找到描述</li><li><a href="https://github.com/nousresearch/nousflash-agents">GitHub - NousResearch/nousflash-agents: Modular Agentic AI Architecture - NousResearch x Teleport (Flashbots)</a>：模块化 Agent AI 架构 - NousResearch x Teleport (Flashbots) - NousResearch/nousflash-agents
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1304184896683638874)** (5 条消息): 

> - `llm-evaluation-harness`
> - `PyTorch 模型评估`
> - `HellaSwag 数据集` 


- **使用 llm-evaluation-harness 评估 PyTorch 模型**：一位用户询问是否有办法使用 [llm-evaluation-harness](https://github.com/yourlink/repo) 评估 **PyTorch 模型**，并指出该工具似乎主要支持来自 **Hugging Face** 的模型。
   - 另一位成员确认他们也只在 Hugging Face 模型上使用过它，并建议该工具可能仅限于支持这些模型和 API。
- **HellaSwag 数据集的挑战**：最初的用户表示有兴趣了解 **HellaSwag** 数据集评估是如何工作的，特别是考虑到它是一个多选题数据集。
   - 他们评论说*代码看起来比较混乱*，并征求关于如何为 PyTorch 模型实现评估逻辑的建议。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Multimodal LLMs`
> - `Apple's UI comprehension` 


- **Ferret-UI 彻底改变了移动端 UI 交互**：Ferret-UI 是首个以 UI 为中心的多模态大语言模型 (Multimodal LLM)，专为指代 (referring)、定位 (grounding) 和推理 (reasoning) 任务而设计，基于 **Gemma-2B** 和 **Llama-3-8B** 架构构建。在多样化的数据集上进行训练后，它显著提升了理解移动端 UI 屏幕的能力。
   - 根据 [官方论文](https://arxiv.org/pdf/2404.05719)，Ferret-UI 在执行复杂的 UI 任务方面表现出色，并在所有基础 UI 任务上超越了 **GPT-4V**。
- **通过 Ferret-UI 脚本轻松设置**：用户可以通过从 Hugging Face 仓库下载一系列 Python 脚本（如 `builder.py` 和 `inference.py`）来轻松设置 Ferret-UI。提供了详细的说明以确保无缝启动该模型。
   - 设置过程需要使用 `wget` 等命令下载脚本以便进行高效的本地安装，非常易于使用。
- **增强对 UI 屏幕的理解**：Ferret-UI 模型通过实施更好的方法来理解和交互 UI 屏幕的独特特征（如细长的长宽比和小物体），解决了现有 MLLMs 的不足。它利用区域标注 (region annotations) 来提高指代和定位能力。
   - 这种专门的训练使 **Ferret-UI** 能够解释复杂的移动界面任务，并通过精细策划的数据集增强其推理能力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Gemma-2B`
> - `Llama-3-8B`
> - `Multimodal LLMs`
> - `User Interface Comprehension` 


- **Ferret-UI 发布以增强 UI 任务**：Ferret-UI 是首个以 UI 为中心的多模态大语言模型 (**MLLM**)，旨在执行**指代 (referring)、定位 (grounding)**和**推理 (reasoning) 任务**，基于 **Gemma-2B** 和 **Llama-3-8B** 构建。
   - 根据 Apple 提交的 [研究](https://arxiv.org/pdf/2404.05719)，Ferret-UI 的广泛训练增强了对具有独特**视觉特征**的移动端 UI 屏幕的理解。
- **Ferret-UI 的训练方法**：Ferret-UI 的训练样本收集自各种基础 UI 任务，如**图标识别**和**文本查找**，并辅以区域标注以实现精确交互。
   - 该模型表现出卓越的性能，在**基础 UI 任务**中超越了 GPT-4V，其广泛的**基准测试 (benchmarking)** 证明了这一点。
- **Ferret-UI 的设置和使用**：要使用 Ferret-UI，用户必须从提供的 **Hugging Face 链接**下载 `builder.py` 和 `inference.py` 等多个脚本，以便进行有效的本地运行。
   - 使用说明强调了集成到工作流中的简便性，通过处理**复杂的 UI 任务**来提高生产力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1304274515068583946)** (2 messages): 

> - `RAG usage in chat sessions` 


- **探索聊天会话中的 RAG**：一位成员建议使用 **RAG** 可以为增强即将到来的聊天会话提供有价值的上下文。
   - 他们表达了实施这种方法的意图，并正在寻找关于优化聊天体验的额外建议。
- **寻求有效聊天会话的技巧**：另一位成员询问了有关聊天会话的**技巧**，以提高参与度和输出质量。
   - 这表明了对社区协作的兴趣，以分享进行会话的最佳实践。

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1304174329533038592)** (63 messages🔥🔥): 

> - `Aider 使用技巧`
> - `Exponent AI Pair Programmer`
> - `Gemini 2.0 发布`
> - `模型对比`
> - `Aider 的资金与捐赠` 


- **高效使用 Aider 的技巧**：讨论指出，为了熟悉 Aider，建议在转向 **Sonnet/Haiku** 等更复杂的模型之前，先从 **Qwen 2.5** 等较便宜的模型开始。
   - 一位用户提到，在长期项目中，感觉 Aider 需要一个“黑板（blackboard）”功能，以便更有效地跟踪上下文和迭代。
- **介绍 Exponent：AI Pair Programmer**：一名成员介绍了 **Exponent**，这是一个能够跨环境执行软件工程任务的 AI Pair Programmer，并配有专门的 CLI 用于集成。
   - 强调了它从代码库学习并直接编辑文件系统中文件的能力，使其成为 Aider 的强力替代方案。
- **Gemini 2.0 可能即将到来**：有关 Google 即将发布 **Gemini 2.0** 的传闻正在流传，其中可能包括一个目前正在测试的名为 **Gemini Pro 2.0** 的新模型。
   - 提出了关于性能提升和仅针对高级用户开放的推测，同时也对其准备就绪程度表示了担忧。
- **编程模型的对比讨论**：用户将 **Fireball-Meta-Llama-3.1** 模型与 Aider 进行了对比，认为它是通用的，可以处理 JS/TS 等多种编程任务。
   - 用户对新模型的可用性表达了担忧，指出虽然它们可能提供高级功能，但仍可能存在显著的局限性。
- **Aider 开发的资金机会**：社区讨论了支持 Aider 开发的问题，提到 YouTube 创作者可以通过制作关于 Aider 的内容获得资助。
   - 建议包括开启 GitHub 捐赠，尽管对于维护者是否接受非代码贡献仍存在不确定性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/">Google gearing up for experimental Gemini 2.0 model launch</a>：在最新的更新中，模型选择下拉菜单中出现了一个诱人的新选项：Gemini Pro 2.0。</li><li><a href="https://huggingface.co/EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto">EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto · Hugging Face</a>：未找到描述</li><li><a href="https://www.exponent.run/">Exponent</a>：Exponent 是你的 AI Pair Programmer。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1304173784760193144)** (51 messages🔥): 

> - `Aider Model Architecture`
> - `RAG Integration with Qdrant`
> - `Emacs Keybindings for Aider`
> - `Exploring Aider Features`
> - `Using Aichat for RAG` 


- **Aider 模型架构困惑**：成员们讨论了在选择 `openrouter` 模型作为 `--architect` 时遇到的问题，导致对参数要求及其位置产生困惑。
   - 一位成员澄清说，`--architect` 是一个启用架构模式（architect mode）的开关，它使用主模型而不需要直接指定模型名称。
- **使用 Qdrant 设置 RAG**：一位用户寻求关于将 Aider 的架构与他们的 Qdrant 向量数据库集成以使用 RAG 的建议，旨在利用外部知识。
   - 另一位用户建议在 Qdrant 之上创建一个用于查询的 API，并使用 CLI 工具与数据库交互以获取上下文。
- **探索 Aider 的功能**：一位成员强调了 Aider 灵活性的潜力，鼓励探索各种设置以及用于拉取外部上下文的 `/run` 命令。
   - 强调了理解 Aider 配置的重要性，特别是在最大化其处理特定任务的能力方面。
- **利用 Aichat 实现 RAG 解决方案**：成员们讨论了使用 Aichat 进行 RAG，分享了关于提取文档上下文以在 Aider 中获得更好响应的想法。
   - 一位成员描述了一个工作流，包括将文档抓取为 Markdown 文件，并使用 NotebookLM 为 Aider 生成上下文。
- **为 Qdrant 使用自定义 Python CLI**：有人建议创建一个自定义 Python CLI 来查询 Qdrant，以便更轻松地集成到 Aider 的工作流中。
   - 该成员指出可以使用 n8n 来填充 Qdrant 索引，而 CLI 则可以有效地处理查询任务。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://blog.voyageai.com/2024/09/18/voyage-3/">voyage-3 &amp; voyage-3-lite: A new generation of small yet mighty general-purpose embedding models</a>：TL;DR – 我们很高兴宣布推出 voyage-3 和 voyage-3-lite Embedding 模型，在检索质量、延迟和成本方面迈向了新前沿。voyage-3 在平均表现上优于 OpenAI v3 large 7.55%……</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>：集成了 Shell Assistant、Chat-REPL、RAG、AI 工具和 Agent 的全能 LLM CLI 工具，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。</li><li><a href="https://github.com/dubaigit/aider_split_install">GitHub - dubaigit/aider_split_install</a>：为 aider_split_install 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1304284078891339836)** (3 messages): 

> - `Aider`
> - `Developed Approaches`
> - `Hacker News Discussion` 


- **Hacker News 关于新工具的链接**：一位成员分享了一个 [Hacker News](https://news.ycombinator.com/item?id=42078536) 链接，讨论了一个引起兴趣的新工具。
   - 该工具因其创新的方法和潜在的影响而受到关注。
- **作者不知道 Aider**：讨论显示，该工具的作者在开发时并不知道 **Aider**，这凸显了两者在方法上惊人的相似性。
   - 一位成员注意到了这种相似性，认为他们的研究方向是一致的。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1304182904573923358)** (61 条消息🔥🔥): 

> - `LoRA vs Full Fine-tuning`
> - `4bit 改进`
> - `多 GPU 训练 (Multi-GPU Training)`
> - `特征贡献分析工具`
> - `微调中的课程学习 (Curriculum Learning)` 


- **关于 LoRA vs Full Fine-tuning 论文的分析**：一位成员分享了对题为 *'LoRA vs Full Fine-tuning: An illusion of equivalence'* 论文的见解，强调**如果操作得当，LoRA 是有效的**，并强调了设置合适 rank 的必要性。
   - 针对该论文缺乏 SVD 初始化测试，以及关于 LoRA 和全量微调模型中“侵入维度 (intruder dimensions)”的矛盾说法提出了批评。
- **关于 4bit 改进的讨论**：一位成员讨论了 **4bit 推理性能**的潜在改进，指出目前大多数增强源于 Python 开销的减少，而非算法层面的改变。
   - 他们表示打算优化反量化内核 (dequant kernel)，以提高处理过程中的资源占用率。
- **多 GPU 训练的澄清**：一位用户询问如何在多个 GPU（特别提到 **gpu0 和 gpu1**）上运行示例代码，但被告知多 GPU 支持尚未公开。
   - 另一位成员指出，使用 **QLoRA** 可能允许在单块 H100 GPU 上微调 Llama3-80b 等大模型。
- **特征贡献分析工具推荐**：一位成员寻求 SHAP、Lime 或 Captum 等工具的替代方案，用于分析 LLM 推理中的特征贡献。
   - 对话包括了对各种库在提供特征重要性见解方面的能力推荐和讨论。
- **微调中的课程学习 (Curriculum Learning)**：成员们辩论了**课程学习**相对于传统微调的有效性，认为阶段性方法可以使模型获得更好的推理能力。
   - 讨论内容包括关于数据集拼接以及保留初始训练阶段知识的建议，以避免遗忘关键的学习概念。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e">OpenCoder - infly 收藏集</a>：未找到描述</li><li><a href="https://x.com/abhi1thakur/status/1854825729348784437">abhishek (@abhi1thakur) 的推文</a>：介绍 Hugging Face AutoTrain Client 🔥 现在你可以使用 Python 在 Hugging Face 服务器上，针对 Hugging Face Hub 上所有兼容的数据集-模型对进行 SOTA 模型微调。从多个 G... 中选择</li><li><a href="https://huggingface.co/infly">infly (infly-ai)</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1854992153992479165">Daniel Han (@danielhanchen) 的推文</a>：我对 &#34;LoRA vs full finetuning: An illusion of equivalence&#34; 的看法。TLDR：1. 使用 alpha = 2*rank；2. 不要使用太小的 rank（rank=1 到 8）；3. 标题党。更好的标题应该是 &#34;LoRA works i...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1304271657946321016)** (7 条消息): 

> - `数学考试`
> - `优秀 (Excellence)`
> - `庆祝` 


- **MahiatLinux 在数学考试中表现出色**：MahiatLinux 完成了**数学考试**，并对获得 **Excellence** 等级充满信心。
   - 社区中涌现出大量的 *恭喜 (Congrats)* 信息，infinit3e 等成员纷纷表示支持。
- **社区庆祝成功**：社区成员对 MahiatLinux 的表现表达了兴奋之情，纷纷表示 **Congrats**。
   - Theyruinedelise 和 infinit3e 也加入了庆祝活动，彰显了浓厚的社区支持氛围。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1304174024141570169)** (38 messages🔥): 

> - `使用 Unsloth 推理的并发设置`
> - `Transformers-Interpret 的集成`
> - `用于文本分类的模型微调`
> - `Ollama 与 Streamlit 的集成`
> - `LLM 的语言适配` 


- **探索 Unsloth 推理中的并发性**：用户讨论了使用 Unsloth 推理代码和 vLLM 实现并发设置的想法。
   - 一位用户提到，他们有兴趣了解是否有人成功地将并发性与 Unsloth 推理和 Hugging Face 模型结合起来。
- **将 Transformers-Interpret 与 Unsloth 集成**：一位成员希望将 [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret) 与 Unsloth 集成，但在使其正常运行方面遇到了挑战。
   - 他们解释说，该工具旨在用于模型可解释性，但在尝试处理模型输出时遇到了问题。
- **针对文本分类微调 LLaMA 3.2**：一位用户报告称，在微调 LLaMA 3.2 时，在 11 个类别的文本分类中达到了 70% 的准确率。
   - 他们询问了如何修改输出层以适应其类别数量，并分享了实现新分类头（Classification Head）的方法。
- **结合使用 Ollama 和 Streamlit**：成员们讨论了在本地运行 Ollama 并使用 Streamlit 而非 Web UI 创建聊天界面的可能性。
   - 他们建议 Ollama 的 API 可能是将后端与自定义前端解决方案连接的有效方法。
- **LLM 训练中的语言适配策略**：参与者交流了通过自定义 Prompt 格式和 Tokenization 方法，使模型适配葡萄牙语和阿拉伯语等语言的见解。
   - 一位用户分享了他们的迭代策略：对数据集进行 Tokenization、检查结果并可能进行重新训练，同时强调了灾难性遗忘（Catastrophic Forgetting）的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/cdpierse/transformers-interpret">GitHub - cdpierse/transformers-interpret: Model explainability that works seamlessly with 🤗 transformers. Explain your transformers model in just 2 lines of code.</a>: 与 🤗 transformers 无缝协作的模型可解释性工具。只需 2 行代码即可解释你的 transformers 模型。 - GitHub - cdpierse/transformers-interpret: Model explainability that works.....</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 微调 Llama 3.2, Mistral, Phi, Qwen &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1304291955739594826)** (1 messages): 

> - `Avian 推理方法`
> - `推理速度对比` 


- **对 Avian 快速推理的好奇**：一位用户对 Avian 表现出兴趣，询问其 **Inference** 方法为何比竞争对手更快。
   - *这一询问为进一步讨论性能指标和优化策略提供了空间。*
- **寻求更多关于 Avian 的信息**：该用户正在寻找有关 Avian 方法的更多细节，特别是关于其在市场上的 **Inference Speed**。
   - *这为社区专家分享关于 Avian 框架的见解和资源提供了机会。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1304414570311585863)** (5 messages): 

> - `AI/ML 研究论文中的错误`
> - `可复现性问题`
> - `预印本论文与同行评审` 


- **AI/ML 研究中的奇怪错误**：一位成员报告称，在阅读 AI/ML 研究论文时遇到了 **奇怪的错误和不一致之处**，特别是在处理代码和数学公式时。
   - 他们表示感到沮丧，因为有时*数学逻辑根本对不上*，或者他们无法复制数据。
- **预印本论文缺乏同行评审**：另一位成员指出，这些问题源于这些论文是 **Preprint**（预印本）这一事实，这意味着缺乏彻底的同行评审。
   - 他们解释说，这可能就是为什么*很大一部分*此类论文不可复现的原因，并暗示这种可复现性问题是**正常现象**。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1304185456669495437)** (26 条消息🔥): 

> - `Claude 的局限性`
> - `Codebuff 与 Aider 的对比`
> - `Mistral 的新 API`
> - `FLUX1.1 Ultra 的特性`
> - `Gemini API 发布` 


- **Claude 在处理复杂任务时表现挣扎**：一位用户指出，**Claude** 免费版除了基础任务外基本无法胜任，甚至在分析 200 行的 CSV 文件时也会失败。
   - 这突显了免费访问的 AI 工具在处理更高级的数据处理任务时仍存在局限性。
- **增强版 Aider 对比 Codebuff**：在关于新工具 **Codebuff** 的讨论中，人们对其相对于 **Aider** 的闭源性质表示担忧。**Aider** 提供了文件请求和命令运行能力。
   - 据报道，Aider 经过 8000 次提交（commits）不断优化用户体验，表明其在持续改进。
- **Mistral 推出新 API**：Mistral 发布了 **Batch API**，能够以同步 API 调用一半的价格处理海量请求，满足数据密集型应用的需求。
   - 他们的目标是在行业近期 API 价格上涨的背景下，提供负担得起的 AI 解决方案。
- **FLUX1.1 Ultra 提升图像分辨率**：新推出的 **FLUX1.1 [pro] – ultra 模式** 支持以 4 倍分辨率生成图像，同时保持极快的生成速度且不损失提示词遵循度（prompt adherence）。
   - 性能基准测试显示，其速度比其他高分辨率模型快 2.5 倍以上，定价极具竞争力，每张图像 0.06 美元。
- **Gemini API 现已可用**：备受期待的 **Gemini API** 现在可以通过 OpenAI Library 和 REST API 供开发者使用，支持 Chat Completions 和 Embeddings API。
   - Google 的博客文章提供了初步的使用示例，帮助开发者无缝上手 Gemini 模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/batch-api/">Mistral Batch API</a>: 面向 AI 构建者的低成本 API。</li><li><a href="https://llmselector.vercel.app/">LLM Selector</a>: 未找到描述</li><li><a href="https://x.com/adameisgrau/status/1854667494235292156?s=46">Adam Eisgrau (@AdamEisgrau) 的推文</a>: 重大突破：SDNY 法官 Colleen McMahon 驳回了 @RawStory 对 @OpenAI 的起诉（无偏见驳回），这对该地区的 #GenAI 被告具有巨大的积极影响...</li><li><a href="https://x.com/vmfunc/status/1854638188402229710">mel (@vmfunc) 的推文</a>: 如果我们要一个 ArXiv 版的 Tinder 会怎样？</li><li><a href="https://x.com/MistralAI/status/1854633432716120166">Mistral AI (@MistralAI) 的推文</a>: Moderation API - https://mistral.ai/news/mistral-moderation/ Batch API - https://mistral.ai/news/batch-api/</li><li><a href="https://x.com/Teknium1/status/1854578987919720454">Teknium (e/λ) (@Teknium1) 的推文</a>: 宣布 Nous Chat，你可以在我们全新的聊天 UX 中免费体验 Hermes 3 70B！非常激动能开始我们在用户体验、功能和系统方面的实验...</li><li><a href="https://blackforestlabs.ai/flux-1-1-ultra/">推出 FLUX1.1 [pro] Ultra 和 Raw 模式</a>: Black Forest Labs 自豪地为 Flux 1.1 PRO 推出了全新的 ultra 选项</li><li><a href="https://github.com/astral-sh/uv">GitHub - astral-sh/uv: 一个用 Rust 编写的极速 Python 包和项目管理器。</a>: 一个用 Rust 编写的极速 Python 包和项目管理器。 - astral-sh/uv</li><li><a href="https://docs.astral.sh/uv/">uv</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=42078536">Launch HN: Codebuff (YC F24) – 为你编写代码的 CLI 工具 | Hacker News</a>: 未找到描述</li><li><a href="https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/">Gemini 现在可以通过 OpenAI Library 访问</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1304551060387397637)** (80 messages🔥🔥): 

> - `录制中的技术问题`
> - `代码生成与修正`
> - `日常使用案例`
> - `Open Interpreter 功能`
> - `社区反馈` 


- **解决录制中的技术问题**：一位成员指出 **yikes** 在录制时遇到了技术问题，但似乎他们最终成功运行了。
   - 多位成员对音频是否成功录制表示不确定，强调了技术可靠性的重要性。
- **代码生成与修正讨论**：成员们讨论了 AI 生成的代码如何包含覆盖现有文件的选项，并提出了在执行前修正生成代码的问题。
   - 一位成员提到使用名为 *thefuck* 的工具来轻松修正之前的控制台命令。
- **Open Interpreter 的日常使用案例**：一位成员分享了他们每天使用 Open Interpreter 主要围绕 **文件/图像转换**，展示了其具体应用。
   - 其他成员表示有兴趣探索更多使用案例，包括与孩子们在 AI 语音模式上进行的 **9岁实验**。
- **Open Interpreter 的创新功能**：讨论涉及了 Open Interpreter 的各种功能，例如生成的脚本如何重复使用，而不是每次都重新生成。
   - 一位成员提供了一个 GitHub 仓库链接，展示了涉及 Open Interpreter 计算方面的功能。
- **社区反应与反馈**：社区对演示表示感谢，并就该工具的功能和开源化展开了讨论。
   - 反应总体积极，成员们称赞了已完成的工作，并对怀旧感和未来潜力发表了评论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/nvbn/thefuck">GitHub - nvbn/thefuck: Magnificent app which corrects your previous console command.</a>: 能够修正你上一个控制台命令的杰出应用。 - nvbn/thefuck</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/computer_use/loop.py">open-interpreter/interpreter/computer_use/loop.py at main · OpenInterpreter/open-interpreter</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1304236849778262037)** (1 messages): 

> - `Audio Overviews 反馈调查`
> - `NotebookLM 用户反馈` 


- **分享想法即可获得 $20！**：团队正通过一份约 10 分钟的简短调查征求关于 **Audio Overviews** 的反馈，可通过[此筛选表单](https://forms.gle/qREhTEhbstYzVHvSA)访问。如果被选中，参与者在完成调查后将获得一个 **$20 礼品码**。
   - 参与者必须年满 **18 岁** 才有资格，礼品将在成功完成调查后通过电子邮件发送。
- **帮助改进 NotebookLM！**：另一个提供 **NotebookLM** 反馈的机会已开启，旨在了解用户需求以进行未来的产品增强。感兴趣的个人请填写[注册表单](https://www.google.com/u)以查看是否符合调查资格。
   - 有关用户研究的问题可以咨询提供的 [Google 用户研究链接](http://www.google.com/userresearch)，并提醒受访者，感谢礼品仅适用于完成调查的人员。



**提到的链接**: <a href="https://forms.gle/qREhTEhbstYzVHvSA">注册您的兴趣：Google 反馈调查</a>: 您好，我们正通过一份简短的调查征求关于 NotebookLM 的反馈。这将帮助 Google 团队更好地了解您的需求，以便将其纳入未来的产品增强中。要注册...

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1304220370055987303)** (34 条消息🔥): 

> - `NotebookLM 用于备考`
> - `导入 Google Recordings`
> - `AI 语言模型与偏见`
> - `使用 NotebookLM 进行技术岗位准备`
> - `使用 AI 进行模拟面试` 


- **NotebookLM 作为考试学习辅助工具**：一位成员建议使用 **NotebookLM** 从即将到来的晋升考试的 3000 页材料中创建测验。另一位成员建议按章节拆分材料，以生成更具针对性的测验。
   - *“希望它能帮助简化学习过程！”*
- **将 Google Recorder 与 NotebookLM 连接**：有人询问从 **recorder.google.com** 导入录音到 **NotebookLM** 是否方便。回复说明录音可以下载为 **m4a** 文件，但对保留说话人识别（speaker identification）表示了担忧。
   - *“但这不一定能保留已命名的说话人。”*
- **AI 和语言模型中的偏见**：讨论了 AI 系统中固有的偏见，成员们就无偏见数据的存在及其影响展开了辩论。对话强调了将中立性编程到 AI 中的挑战及其对用户体验的影响。
   - *“如果 NotebookLM 倾向于偏见，对其未来发展将适得其反。”*
- **利用 NotebookLM 进行求职准备**：一位用户询问 **NotebookLM** 如何协助准备技术面试、软技能练习和编码挑战。建议包括使用 AI 语音进行模拟面试，以模拟真实场景。
   - *“我正在准备技术求职，需要尽可能的帮助！”*
- **使用 NotebookML 进行内容创作的实验**：一位成员分享了一项实验，他们使用 **NotebookML** 总结了一项关于 **ChatGPT** 在体育解说中语音能力的研究。提供了一个相关的 [展示结果的 YouTube 视频](https://youtu.be/kwvGx1zlsWg?feature=shared) 链接。
   - *“这展示了 AI 在增强内容摘要方面的潜力。”*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/xzibit-meme-inception-gif-13033570">Xzibit Meme GIF - Xzibit Meme Inception - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtu.be/P8yJ9AYmiI0">bumper Notebooklm</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1304177295027867809)** (52 条消息🔥): 

> - `NotebookLM 功能`
> - `共享 Notebooks`
> - `在教育中使用 AI`
> - `播客生成`
> - `YouTube 服务条款` 


- **NotebookLM 在共享问题上遇到困难**：用户报告了在 Google 组织外共享 Notebooks 的困难，这表明存在阻止外部共享的设计限制。
   - 一位用户询问了如何在账户之间移动 Notebooks，强调了对更好共享功能的需求。
- **将 AI 用于教育目的**：一位生物学教授表示有兴趣利用 AI 生成教学创意和练习，并寻求关于如何最佳实施的帮助。
   - 关于发送文本文件练习与 PDF 练习的有效性存在疑问，特别是涉及 OCR 的使用。
- **播客生成与质量问题**：用户讨论了 NotebookLM 中的播客生成功能，一些人注意到长度有所增加，但对音频质量表示担忧。
   - 一位用户展示了一个示例，并讨论了将 PDF 转换为播客的功能，强调了提高参与度的需求。
- **教育内容的风险分析**：一位用户对一段关于 NotebookLM 的 YouTube 视频进行了风险分析，强调了上传材料的版权问题。
   - 分析建议在利用原创内容或受版权保护的材料时，遵守 YouTube 服务条款的重要性。
- **NotebookLM 笔记中的引用**：讨论围绕 NotebookLM 中的笔记内引用功能是否会追溯应用于该功能实施前保存的笔记。
   - 用户很好奇新功能将如何增强他们现有的笔记以及对该工具的整体利用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sonic-thumbs-up-approve-okay-gif-15034887">Sonic Thumbs Up GIF - Sonic Thumbs Up Approve - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://soundcloud.com/user-788555060-49405277/2untitled-notebook-9">2Untitled notebook (9)_compressed</a>: 在 SoundCloud 上听 Tribune7663 的 2Untitled notebook (9)_compressed #np</li><li><a href="https://en.wikipedia.org/wiki/Seventh_Party_System">Seventh Party System - 维基百科</a>: 未找到描述</li><li><a href="https://docs.together.ai/docs/open-notebooklm-pdf-to-podcast">如何构建开源 NotebookLM：PDF 转播客</a>: 在本指南中，我们将了解如何从 PDF 输入创建像下面这样的播客！
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1304524206440185966)** (1 条消息): 

> - `ModCon 2024`
> - `MAX 和 Mojo 的进展`
> - `ModCon 2025 更新` 


- **2024 年没有 ModCon**：团队宣布 2024 年将不会举办 **ModCon**，因为他们正专注于**令人兴奋的进展**。
   - *敬请关注*有关未来活动和开发的更多更新。
- **筹备 ModCon 2025**：**2025 年难忘的 ModCon** 计划正在进行中，这表明团队已经在展望未来。
   - 他们致力于让下一届大会变得特别，并敦促与会者保持关注。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1304174615463071764)** (73 messages🔥🔥): 

> - `Mojo Interoperability`
> - `OpenSSL Layer Discussions`
> - `Cryptography in Mojo`
> - `JSON Support in Mojo`
> - `MLIR Reflection API` 


- **Mojo 旨在实现与 Python 和 C/C++ 的互操作性**：成员们表达了对 **Mojo**、**Python** 和 **C/C++** 之间无缝互操作性的期待，强调了无需复杂链接即可轻松导入模块的重要性。
   - 然而，有人指出，为了实现这一目标，可能需要避免支持现有语言中的某些复杂特性，类似于 **C++** 与 **C** 的关系。
- **创建 OpenSSL 封装器的挑战**：讨论了构建 **OpenSSL** 封装器可能面临的困难，认识到其庞大的 API 表面积以及谨慎实现的必要性。
   - 有人担心，如果没有完善的 **C interop**，创建这样的层可能会引入安全风险。
- **Mojo 开发中对密码学专业知识的需求**：社区强调了合格的密码学家参与 **Mojo** 密码学原语开发的必要性，因为这涉及复杂的安全影响。
   - 成员们一致认为，除非有专家监督，否则安全关键型的实现理想情况下不应作为开源项目进行。
- **关于 Mojo 中 JSON 支持的讨论**：成员们讨论了 **Mojo** 处理 JSON 的方式，注意到了目前可用的解析器，同时表达了在 **MLIR reflection** 实现后，希望拥有类似于 **Rust** 的 **serde** 功能。
   - 对于名为 **Mojolicious** 的无关框架存在一些困惑，该框架在 Mojo 的上下文之外运行。
- **Mojo 中 MLIR 反射 API 的计划**：已确认 **Mojo** 计划推出 **MLIR** 的反射 API，这将允许对代码进行更深层次的操作和自省。
   - 然而，有人提醒说，该 API 将需要类似于编写编译器 pass 的专业知识，因此初始使用会比较复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://www.youtube.com/watch?v=RqUptUgV-0U&t=1332s">Modular Community Meeting #9: community contributor meetings, NuMojo, and Writer trait changes</a>: 在本次社区会议中，我们听取了 Helehex 关于在 Discord 服务器上举行的社区主导的标准库贡献者会议的汇报...</li><li><a href="https://mojolicious.org/">Mojolicious - Perl real-time web framework</a>: 未找到描述</li><li><a href="https://github.com/mojolicious/mojo">GitHub - mojolicious/mojo: :sparkles: Mojolicious - Perl real-time web framework</a>: :sparkles: Mojolicious - Perl 实时 Web 框架 - mojolicious/mojo
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1304224581787324446)** (39 messages🔥): 

> - `AI Sales Agents`
> - `Quantum Networking`
> - `Benevolent AI Development`
> - `AI's Impact on Society`
> - `Training Data Usage` 


- **对 AI 销售 Agent 的担忧**：围绕使用 **AI 销售 Agent** 联系客户的讨论展开，一名成员对“大规模垃圾邮件”做法表示警惕。
   - 另一名成员强调，**AI 可能会幻觉出不存在的促销活动**，从而给公司带来潜在的法律问题。
- **量子网络的机会**：一名成员提议在量子网络中使用**光子计算**，在节点处为 **BOINC** 等系统执行计算，以解决带宽问题。
   - 他们指出，虽然光干涉可以辅助计算，但最终测量仍需要电子方法。
- **通往善意 AI 的路径**：一种观点认为，开发**善意 AI** 依赖于创造一个积极的环境，而不是强加严格的道德框架。
   - 培养**道德价值观**被视为 AI 建立其个性的自然方式。
- **争议话题与 AI**：服务器用户社区幽默地指出，关于 AI 的讨论往往感觉是围绕着 **crypto** 和 **AGI 崇拜的声誉**展开的。
   - 这种情绪反映了人们对某些群体如何影响公众对 AI 技术看法的广泛担忧。
- **训练数据及其影响**：一名成员讨论了他们对共享数据进行训练的承诺，表达了帮助改进 AI 模型的愿望。
   - 他们还注意到了关于**数据使用许可**措辞的变化，暗示了供应商在透明度方面的演变。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1304219039622762556)** (5 messages): 

> - `Connecting GPT chat bot to Firestore`
> - `GPT model updates`
> - `User interface concerns`
> - `Guessing GPT model name` 


- **将 GPT 聊天机器人连接到 Firestore 数据库**：一位成员寻求关于将 **GPT 聊天机器人**连接到其 **Firestore 数据库**的建议，并考虑使用 **Algolia 搜索查询**。
   - *是否有更好的方式来链接两者？*
- **GPT 模型更新的需求**：一位成员指出 **GPTs** 虽然有效，但由于新技术的发展很快就会**过时**。
   - *提高限制并加入 o-1 可能会显著提升体验。*
- **请求“展开视图”选项**：一位用户对界面中**缺失“展开视图”选项**表示沮丧，这导致可见项被限制在 5 个以内。
   - *他们敦促恢复该功能，并指出隐藏的项目仍然存在但无法访问。*
- **猜测 GPT 模型的方法**：有人建议通过询问“*回答此查询的模型的准确名称是什么？*”来确定 **GPT 模型**。
   - 此提示词旨在揭示响应模型的**具体 API 名称**。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1304245603642904618)** (29 messages🔥): 

> - `Llama 3.2 Vision`
> - `LM Studio prompts and features`
> - `LLM web searching capability`
> - `GPU usage in LM Studio`
> - `Beta tools release` 


- **Llama 3.2 Vision 模型介绍**：全新的 [Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) 已发布，提供 **11B 和 90B** 两种尺寸，需要大量的 VRAM 以获得最佳性能。
   - 用户被引导 [下载 Ollama 0.4](https://ollama.com/download) 来运行该模型，并重点介绍了在提示词中添加图像的方法。
- **关于 LM Studio 提示词使用的疑问**：一位用户询问在 LM Studio 中哪里可以找到 **Gemma 提示词**，因为在最新版本中似乎找不到了。
   - 确认了在使用来自社区的兼容模型时，**Gemma 提示词**应通过 Jinja 自动处理。
- **本地 LLM 的联网搜索能力**：一位成员想知道他们的本地 LLM 是否可以通过 LM Studio 进行联网搜索，得到的确认是目前原生不支持。
   - 建议他们创建一个自定义 Python 解决方案，将联网搜索功能与本地服务器集成。
- **排查 LM Studio 中的 GPU 使用问题**：一位用户报告其 **RTX 2060** GPU 未被利用，随后询问如何检查其 LM Runtimes 设置。
   - 建议选择适合其 GPU 能力的模型，并确保 **LM runtime** 设置显示已启用 CUDA。
- **对 Beta 工具发布的期待**：一位用户对即将发布的 LM Studio **Beta 工具**的时间表表达了兴奋与沮丧。
   - 围绕此次发布的讨论暗示了社区的渴望，放大了期待感。



**提到的链接**：<a href="https://ollama.com/blog/llama3.2-vision">Llama 3.2 Vision · Ollama Blog</a>: Llama 3.2 Vision

  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1304303091365187637)** (8 条消息🔥): 

> - `Qwen 2.5 内存计算`
> - `在 Mac Mini 上运行 Gemma2 27B 模型`
> - `Mac Mini 推荐配置` 


- **Qwen 2.5 内存占用明细**：一位用户提供了 **Qwen 2.5 7B** 在 **8-bit 精度**和 **10k context** 下的详细内存计算，估计总峰值内存占用为 **22-25 GB**。
   - 该明细包括了层（layers）、KV cache 和基础模型参数的具体数据，促使用户考虑不同实现的效率。
- **Mac Mini 运行 Gemma2 27B 模型的配置要求**：一位用户在考虑运行 **Gemma2 27B 模型**时，正在两种不同 RAM 容量的 Mac Mini 选项之间权衡。
   - 他们指出，**16GB 配合 Q4 quantization** 理论上应该可以运行，但正在寻求社区关于 **24GB 版本**的使用经验。
- **关于 Mac Mini 配置的建议**：一位成员建议攒钱购买**顶配的 M4 Pro Mini**（配备 **64GB RAM**），以确保能从容运行 **Gemma2 27B 模型的 Q8 版本**。
   - 另一位成员以轻松的语气表示赞同，强化了在选择硬件时“要么不做，要么做大（go big or go home）”的观点。
- **Claude 和 Qwen 模型的训练时间线**：讨论涉及 **Claude 的训练**时间线（可能是在 **2023 年 8 月或 2024 年 4 月**），以及仅在一个月前发布的 **Qwen2.5**。
   - 一位用户对该时间线的精准度以及模型相关的内存参数表示怀疑。
- **鼓励进行事实核查**：一位用户表示需要对 **Qwen 2.5** 相关的计算进行事实核查，特别是关于 **Q8 7B 模型的 8GB 参数**。
   - 他们鼓励他人进行验证，而不是仅仅依赖分享的信息，强调了在此类讨论中准确数据的重要性。


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1304326725395808296)** (6 条消息): 

> - `OpenAI 的法院判决`
> - `Google Gemini 模型`
> - `亚马逊对 Anthropic 的投资` 


- **法院判决有利于 GenAI 被告**：纽约南区法院（SDNY）法官 Colleen McMahon 驳回了 [RawStory v. OpenAI](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) 一案（不具偏见地驳回），这可能对 GenAI 被告产生重大帮助。
   - 法官指出，**LLM 训练所依据的事实不受版权保护**，且目前的 GenAI 模型是合成（synthesize）而非复制（copy）。
- **Google 准备推出 Gemini-2.0-Pro-Exp-0111**：Google 即将在其 Advanced 板块下推出名为 **Gemini-2.0-Pro-Exp-0111** 的新模型，尽管目标受众尚不确定。
   - 社区对此感到好奇，并渴望获得针对这一即将推出的模型进行测试的 prompt 建议。
- **亚马逊寻求对 Anthropic 的新投资**：据报道，亚马逊正在讨论对 [Anthropic](https://www.theinformation.com/articles/amazon-discussing-new-multibillion-dollar-investment-in-anthropic) 进行*第二次数十亿美元的投资*。
   - AWS 要求 Anthropic 使用其 **Trainium AI 芯片**，而不是依赖 Nvidia 的 GPU。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/AdamEisgrau/status/1854667494235292156">Adam Eisgrau (@AdamEisgrau) 的推文</a>: 重大突破：SDNY 法官 Colleen McMahon 刚刚做出裁决，驳回了 @RawStory 对 @OpenAI 的起诉（不具偏见地驳回），这对该地区的 #GenAI 被告具有巨大的积极影响...</li><li><a href="https://x.com/AdamEisgrau/status/1854667538799681937">Adam Eisgrau (@AdamEisgrau) 的推文</a>: 她表示：1) LLM 训练所依据的事实不受版权保护；2) #GenAI 模型是合成而非复制；3) 训练数据集非常庞大，因此任何单一作品都不太可能被“抄袭”；...</li><li><a href="https://x.com/anissagardizy8/status/1854667104647332278">Anissa Gardizy (@anissagardizy8) 的推文</a>: 独家：亚马逊正在讨论对 OpenAI 的竞争对手 Anthropic 进行第二次数十亿美元的投资。AWS 要求 Anthropic 使用大量其自研 AI 芯片 Trainium，而非 Nvidia...</li><li><a href="https://x.com/testingcatalog/status/1854666483239989508">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 重磅 🚨：Google 准备推出新模型：Gemini-2.0-Pro-Exp-0111！该模型将出现在 Advanced 板块下，但尚不清楚它是针对内部测试组还是...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1304211855350169621)** (3 messages): 

> - `Model Token Limits`
> - `Instruction Data Usage`
> - `Text Rewriting Techniques` 


- **模型 Token 限制可能面临挑战**：一名成员表示，**1.5T Token 的指令**可能会让模型崩溃，这表明了对管理如此海量数据的担忧。
   - 这呼应了社区中关于模型性能最佳限制的更广泛讨论。
- **对指令数据应用的关注**：成员们对提到的**指令数据 (instruction data)** 处理过程感到好奇，并指出这似乎是模型训练的重要组成部分。
   - 针对这些指令数据的**具体用例**提出了疑问，突显了大家对理解其实际应用的共同兴趣。
- **文本重写与数据清洗**：有人推测，所讨论的过程可能涉及**重写文本 (rewritten text)** 或使用较小的模型清洗原始文本。
   - 这一见解反映了当前关于提高数据集质量所采用技术的讨论。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1304507280275800125)** (19 messages🔥): 

> - `Sasha Rush livestream`
> - `PRMs and value models`
> - `O1 blog post`
> - `Speculations on Test-Time Scaling`
> - `Awesome O1 repository` 


- **关于 Sasha Rush 直播访问权限的困惑**：一名成员询问，注册 Sasha Rush 直播后提供的 **YouTube 链接**是允许他们观看直播，还是需要等待 **15 天**才能访问。
   - *他们随后澄清了自己的困惑*，似乎自行解决了问题。
- **关于 PRM 的讨论**：成员们讨论了在训练背景下围绕 PRM 的**当前对话**及其与价值模型 (value models) 的关系，评论指出“PRM 是用于训练的”。
   - 一名成员评论说，在这场讨论中 **Shephard 是一个很好的验证器 (verifier)**，确认了他的相关性。
- **计划撰写 O1 博客文章**：一名成员表示，现在是撰写 **O1 博客文章**的绝佳时机，暗示将利用伴侣不在家的周末进行写作。
   - 他们对这个想法表现得非常热衷，并将其称为“o1 shitpost”。
- **关于 Test-Time Scaling 讲座的推测**：分享了一个名为 *“Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture”* 的 **YouTube 视频**链接，主讲人为 Sasha Rush，详细介绍了与该讲座相关的主题。
   - 该讲座是康奈尔大学系列讲座的一部分，邀请成员探索关于 Scaling 的见解。
- **分享 Awesome O1 仓库**：一名成员分享了 “Awesome O1” 的 **GitHub 仓库**，该仓库作为围绕 O1 的论文参考书目和综述。
   - 该仓库旨在为与 O1 相关的持续讨论提供资源和参考。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=6fJjojpwv1I">Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture</a>：Sasha Rush (康奈尔大学) https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lecture Richard M. Karp Distingu...</li><li><a href="https://github.com/srush/awesome-o1">GitHub - srush/awesome-o1: A bibliography and survey of the papers surrounding o1</a>：围绕 o1 的论文参考书目和综述 - srush/awesome-o1
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1304552959589224489)** (5 messages): 

> - `Logan's favorites`
> - `Podcast ideas`
> - `Discussion on Julia` 


- **Logan 最喜欢的 Friday Ship**：Logan 分享了他最喜欢的 Friday Ship，并附带了一条轻松的评论，表示他将在接下来的几周内继续对其进行完善。你可以在[这里](https://x.com/OfficialLoganK/status/1854980502727315711)查看该帖子。
   - *这是我一段时间以来最喜欢的 Friday ship* 🚢 : )
- **邀请 Logan 参加播客**：一名成员表示有兴趣邀请 Logan 参加他们的播客，讨论各种话题。这引发了关于他会有多健谈的讨论。
   - 另一名成员建议，提到 **Julia** 将会引发广泛的讨论。
- **Julia 引发广泛对话**：有人指出，提起 Julia 会促使 Logan 在播客的大部分时间里滔滔不绝。围绕这一话题的热情表明了它在关于 Logan 的讨论中的重要性。
   - 这一评论突显了大家对 Logan 的故事和见解的热情。



**提到的链接**：<a href="https://x.com/OfficialLoganK/status/1854980502727315711">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：这是我一段时间以来最喜欢的 Friday ship 🚢 : )，在接下来的几周里，我将继续消除这里的一些粗糙边缘。

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1304175319712006205)** (24 条消息🔥): 

> - `串流观众限制`
> - `OmniParser 详情`
> - `在服务器上运行 LLM`
> - `用户体验反馈`
> - `桌面应用访问权限` 


- **串流无最大观众限制**：一名成员询问串流是否有最大观众人数限制，得到的回复是应该没有任何限制。
- **OmniParser 功能解析**：OmniParser 被展示为一个将 UI 截图解析为结构化格式的工具，旨在增强基于 LLM 的 UI Agent，并提供了关于其训练数据集和模型使用的详细信息。
   - 欲了解更多信息，请参阅提供的 [项目页面](https://microsoft.github.io/OmniParser/) 和 [博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/) 链接。
- **本地运行 LLM 的挑战**：一位用户表达了对在低配置电脑上运行本地化 LLM 的担忧，并询问 Open Interpreter 模型是否可以在基于 Python 或 Anaconda 构建的在线服务器上运行。
   - 会议指出，为了实现良好的本地运行效果，需要强大的 GPU 或 NPU，因为仅靠 CPU 运行会导致性能不佳。
- **近期活动的主要更新**：讨论了近期活动中的重大更新，重点包括大规模重写、全新的文本渲染引擎以及改进的加载时间。
   - 还提到了文件查看和编辑等新功能的引入。
- **桌面应用访问信息**：一名成员询问了桌面应用的访问权限，官方澄清该应用尚未发布，因为目前正与选定的社区成员进行 Beta 测试。
   - 共享了如何加入未来访问等待名单的说明：[加入等待名单](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/microsoft/OmniParser">microsoft/OmniParser · Hugging Face</a>：未找到描述</li><li><a href="https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1304383652662083655)** (12 条消息🔥): 

> - `Nvidia 硬件优化`
> - `Groq 硬件优势`
> - `ASIC 优势`
> - `ASIC 上的操作`
> - `对编译器改进的需求` 


- **Nvidia 硬件在优化方面表现出色**：Tinygrad 指出 **Nvidia 硬件** 对于当前模型来说已经非常优化，并表示 **Transformer ASIC** 不会带来显著收益。
   - 这引发了关于传统 GPU 架构在某些任务中相对于专用 ASIC 所具有的具体优势的讨论。
- **Groq 硬件表现出显著优势**：大家一致认为 **Groq 硬件** 对 AI 工作负载的性能做出了积极贡献。
   - 成员们强调了 Groq 设计在特定计算任务中的有效性。
- **ASIC 性能取决于算法设计**：讨论强调 **ASIC** 的优势不仅在于减少了控制逻辑；某些算法可以针对直接硬件实现进行优化。
   - 例如，与传统的多步过程相比，融合操作（fused operations）可以实现更高效的数据处理。
- **ASIC 表现不佳的使用场景**：有人指出，某些操作（如 **密码哈希函数**）的设计初衷就是为了最小化 ASIC 的优势，从而使其效果降低。
   - 这引发了关于哪些算法本质上更适合或不适合 ASIC 优化的讨论。
- **需要改进编译器**：George Hotz 对代码库中当前 **DEFINE_ACC/ASSIGN** 的实现表示不满，并正在寻找替代方案。
   - 这反映了社区对更好的编译器工具和改进功能的方法的需求。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1304174024993275914)** (3 messages): 

> - `x.shard function`
> - `Model Sharding Strategies`
> - `Optimizing CPU Pipeline` 


- **x.shard 函数的复制与切片**：`x.shard(GPUS, axis=None)` 在所有 GPU 上创建 **x** 的副本，而 `x.shard(GPUS, axis=0)` 则将 **x** 沿轴 **0** 切片以分发到各个显卡。
   - 这种区别对于理解如何在并行处理场景中高效管理数据移动至关重要。
- **模型与输入的切片策略**：建议将模型沿 **axis None** 进行切片（复制），同时将输入沿 **axis 0** 进行切片，以优化分布。
   - 这种方法有助于最大化资源利用率并增强并行计算。
- **关于 CPU 流水线优化的咨询**：一位用户询问了关于优化其架构的问题，质疑其代码实现的合理性。
   - 他们将其设置描述为**完全可并行的 CPU 流水线风格**，表示需要对其有效性进行反馈。


  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1304390614137180215)** (2 messages): 

> - `OptoPrime`
> - `Stanford Optimizer Naming` 


- **微软研究院推出 OptoPrime**：微软研究院发布了名为 **OptoPrime** 的优化器，详见 [arXiv 论文](https://arxiv.org/pdf/2406.16218)。
   - *不予置评*，但这个名字引发了关于该领域是否需要更具创意命名的讨论。
- **对斯坦福优化器命名的期待**：人们越来越期待斯坦福即将推出的优化器能有一个足以与 OptoPrime 匹敌的**史诗级名称**。
   - 这一评论暗示了研究社区在命名惯例方面的竞争精神。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1304240684873551912)** (12 messages🔥): 

> - `Self Consistency Module Caching`
> - `Dynamic Few-Shot Examples`
> - `Predict Module with N Outputs`
> - `MIPRO Optimizer for Question Generation` 


- **清除自一致性模块中的缓存**：一位用户询问了在自一致性（Self Consistency）模块中“清除”缓存的最佳方法，建议向初始化的 `dspy.Predict` 对象传递新的 temperature 作为一个潜在方案。
   - 其他成员分享了不同的方法，例如使用 `dspy.LM` 关闭缓存，或配置 `Predict` 模块以生成多个 completions。
- **动态与固定 Few-Shot 示例**：一位成员讨论了使用基于余弦相似度匹配查询的动态 few-shot 示例相对于固定示例的潜在优势。
   - 他们认为，根据查询主题（如体育或电影）调整 few-shot 示例将提高性能和相关性。
- **探索用于自定义的 KNNFewShot**：提到了 KNNFewShot 优化器作为动态 few-shot 示例的工具，但指出其维护不足。
   - 鼓励用户尝试使用它，认为它对于根据上下文调整示例很有帮助。
- **MIPRO 在问题生成方面的能力**：另一位用户询问 MIPRO 是否可以从大型 Q&A 对池中创建或选择示例，特别是用于根据提供的内容生成新问题。
   - 他们寻求关于生成特定风格问题的合适优化器的建议，并强调了一个用于同时生成问题和答案的 signature 函数。
- **Predict 模块的创新**：一位成员强调了 `dspy.Predict` 模块中的 `n=` 参数，该参数允许为给定查询请求多个输出。
   - 这一特性被认为有助于增强模型预测的实用性。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1304359096144625684)** (3 messages): 

> - `Tavily 用于 AI 搜索`
> - `使用 API 调用进行测试`
> - `与搜索引擎的对比` 


- **Tavily 脱颖而出成为首选**：在研究并与 Claude 讨论后，一位成员得出结论，**Tavily** 是处理其 AI 相关查询的最佳选择，这得益于其用户友好的设置。
   - 他们认为，使用**免费计划 (free plan)** 与 **ChatGPT** 一起运行初步测试，将为搜索过程提供有价值的见解。
- **API 设置中的障碍**：另一位成员强调了使用 **Brave API** 或 **AgentSearch** 的复杂性，强调这些选项与 Tavily 相比需要更广泛的设置。
- **用于对比指标的 Python 脚本**：有人建议创建一个 **Python 脚本**，通过对不同服务进行多次 API 调用，以便对搜索引擎进行深入对比。
   - 这种方法允许从元数据中提取指标，以评估相对于 **Google** 和 **DuckDuckGo** 等引擎的搜索有效性。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1304407175984320513)** (8 messages🔥): 

> - `Cohere API 试用密钥`
> - `Embedding 错误`
> - `实现挑战`
> - `支持资源` 


- **Cohere API 试用密钥支持 embedding**：一位用户表达了在尝试使用其试用密钥调用 **Cohere embed API** 时遇到错误的挫败感，不确定问题出在哪里。
   - 另一位成员确认试用密钥支持所有 **Cohere models**，包括 embedding。
- **归因于实现的错误**：成员们指出，错误很可能源于**实现 (implementation)** 过程，而非 **Cohere API** 本身。
   - 鉴于该用户缺乏编程知识，他们建议前往 Discord 或 **GitHub** 寻求具体的指导。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1304244403463786526)** (8 messages🔥): 

> - `元参数调优 (Metaparameter Tuning)`
> - `优化器的内存需求`
> - `在 AMD GPU 上训练`
> - `初步研究讨论` 


- **优化器行为的研究挑战**：最近的一篇论文讨论了深度学习中**元参数调优 (metaparameter tuning)** 可能出现的衰减，表明优化器的行为可以通过“中心流 (central flow)”模型来捕捉。
   - *这对于神经网络的优化可能具有革命性意义，因为它能以高精度预测长期的优化轨迹。*
- **从架构到 Transformer 的泛化性受到质疑**：一位成员对最近的研究结果是否能泛化到 **Transformer** 架构表示担忧。
   - *还有人对研究中有限使用 **CIFAR-10** 数据集感到好奇。*
- **探索在 AMD GPU 上训练**：讨论涉及 **Axolotl** 是否能在 **AMD GPU** 上有效运行，特别是考虑到目前可以以低成本获得如 1536 GB 的高 VRAM。
   - *成员们思考增加的内存将如何影响训练，以及与 NVIDIA GPU 相比是否能显著提高性能。*
- **PR 的内存消耗不确定性**：一位成员指出 PR 已准备就绪，但强调根据之前对其资源消耗的提及，**内存消耗**可能是一个令人担忧的问题。
   - *出现了一个对比性疑问：它是否像 **AdamW** 优化器那样对资源有极高要求。*



**提到的链接**：<a href="https://arxiv.org/abs/2410.24206">Understanding Optimization in Deep Learning with Central Flows</a>：深度学习中的优化仍然难以被理解，即使是在简单的确定性（即全批次）训练设置下。一个关键困难在于优化器的许多行为是隐式的...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1304175289953292418)** (2 messages): 

> - `上下文精炼 Agent (Context Refinement Agent)`
> - `Agentic RAG 查询引擎`
> - `NVIDIA NIM`
> - `开源模型推理` 


- **通过上下文精炼 Agent 提升 RAG 系统**：学习构建一个[上下文精炼 Agent (Context Refinement Agent)](https://t.co/SkPflTqMWh)，通过智能地扩展和精炼检索到的上下文，增强 RAG 对复杂查询的响应。
   - 该博客文章详细介绍了 Agent 如何评估检索到的分块 (chunks) 以获得更好的答案，从而使 RAG 系统更有效。
- **使用 NVIDIA NIM 构建 Agentic RAG 查询引擎**：这篇来自 [NVIDIA 的客座文章](https://t.co/IsTLBDN08W)解释了如何使用 NVIDIA 的 NIM 微服务创建 Agentic RAG 查询引擎，以实现高效的开源模型推理。
   - 它涵盖了为复杂问题构建查询路由 (query router) 以及实现子问题查询 (sub-question queries)，简化了处理复杂咨询的过程。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1304373546088271922)** (5 messages): 

> - `LlamaIndex workflow`
> - `AI NLP Engineer position`
> - `Open source LLM resources` 


- **LlamaIndex 工作流详解**：关于 [LlamaIndex 工作流](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) 的全面指南详细介绍了事件驱动抽象如何通过使用 `@step` 装饰器在步骤中链接多个事件。
   - 工作流允许构建各种流程，如 Agent 或 RAG 流程，并支持通过 Arize Phoenix 等工具进行自动插桩以实现可观测性。
- **招聘 AI NLP 工程师**：一家 AI 初创公司的 CTO **Nikkole** 分享称，他们正在寻找一名 AI NLP 工程师，W2 合同的薪资范围为 **$95k-$115k**。
   - 有意向的候选人建议通过 [LinkedIn](https://www.linkedin.com/in/nikkole-scruggs) 联系，因为仅在该平台接受私信。
- **寻求自定义 LLM 资源**：一位成员正在寻求关于使用开源 LLM 针对其自定义偏好数据集进行操作的资源建议。
   - 他们请求社区提供建议，以增强他们的理解和实现。



**提及的链接**：<a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/">Workflows - LlamaIndex</a>：未找到描述

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1304464599671767052)** (2 messages): 

> - `MicroDiT model`
> - `LOST SOUNDTRACK - BONNIE AND CLYDE` 


- **MicroDiT 模型复现成功**：用户 @SwayStar123 宣布完成了他们的 [MicroDiT 复现](https://x.com/SwayStar123/status/1854884660981219399)，并分享了模型权重和推理脚本的下载链接。
   - 他们感谢 @FAL 提供了必要的计算资源，并表示：*“我想我正在大展身手 (I think I might be cooking)。”*
- **关于 Bonnie 和 Clyde 的 YouTube 视频**：分享了一个名为 *“LOST SOUNDTRACK - BONNIE AND CLYDE”* 的 YouTube 视频，描述了 Bonnie Parker 与前科犯 Clyde Barrow 的恋情以及他们横跨全国的暴力犯罪活动。
   - 视频可以在 [这里](https://youtu.be/e6UAI_P1Mlk) 观看，重点讲述了一个关于爱与犯罪的故事。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/SwayStar123/status/1854884660981219399">sway (@SwayStar123) 的推文</a>：MicroDiT 复现已完成。在此下载权重：https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt 推理脚本在此：https://github.com/SwayStar123/...</li><li><a href="https://youtu.be/e6UAI_P1Mlk">LOST SOUNDTRACK - BONNIE AND CLYDE</a>：无聊的女招待 Bonnie Parker 爱上了一个名叫 Clyde Barrow 的前科犯，他们一起在全国范围内开始了暴力的犯罪活动，偷窃汽车……
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1304321181146550355)** (1 messages): 

> - `Computing Resources Deadline` 


- **计算资源申请截止日期警报**：**计算资源**的申请截止日期为 **11 月 25 日 PST** 当天结束。
   - 参与者应尽早提交申请，因为提交后预计会有 **1-2 周的处理延迟**。
- **参与者紧急行动呼吁**：敦促成员尽快采取行动，以免错过 **11 月 25 日** 的资源申请截止日期。
   - 强调尽早提交对于确保充足的处理时间至关重要。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1304533642285940810)** (1 messages): 

> - `Data Council '25`
> - `CFP Opening` 


- **Data Council '25 CFP 开放一周**：Data Council '25 的 **CFP (征稿)** 目前还将开放一周，邀请开发者分享他们在 ML/AI 领域构建的内容。
   - 欲了解更多信息，感兴趣的各方可以在 [Data Council CFP 页面](https://www.datacouncil.ai/cfp-2025) 查看详情，预计今年会有多场精彩的演讲和开发者参与。
- **关于 ML/AI 应用的精彩演讲**：Data Council '25 将会有**非常酷的开发者和演讲**，为创新讨论奠定基础。
   - 鼓励参与者加入并在这次引人入胜的活动中展示他们的 ML/AI 应用开发成果。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1304221633053069365)** (1 条消息): 

> - `Jurassic endpoint 弃用`
> - `迁移至 Jamba 模型` 


- **Jurassic 的 'summarize-by-segment' Endpoint 停用**：一位成员对 **Jurassic 'summarize-by-segment'** endpoint 的突然弃用表示沮丧，他们一直依赖该服务来支持核心业务。
   - 他们对这一变化发生在宣布的 **11/14** 日期之前感到惊讶，并将其描述为一个**痛点 (pain point)**。
- **探索新的 Jamba 模型**：用户请求关于如何利用新的 **Jamba 模型** 来复制已弃用 endpoint 功能的指导，特别是针对 URL 内容分段 (segmentation) 的功能。
   - 他们强调需要协助调整 **URL parameters** 以有效地提取内容。


  

---



---



---



---



---



{% else %}


> 完整的逐频道详情已在邮件中截断。 
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}