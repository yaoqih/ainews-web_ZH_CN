---
companies:
- deepseek
- openai
- nvidia
- langchain
date: '2025-01-28T05:28:32.064176Z'
description: '**DeepSeek** 在 2025 年出人意料地登上了主流新闻，产生了重大的文化影响。**DeepSeek-R1** 模型采用了庞大的
  **6710 亿参数 MoE（混合专家）架构**，并展现出与 **OpenAI o1** 相当的**思维链（CoT）**能力，且成本更低。**DeepSeek
  V3** 模型利用 **fp8 精度**，在训练 **2360 亿参数模型**时，速度比其前代产品快了 **42%**。**Qwen2.5** 多模态模型支持图像和视频，参数规模从
  **30 亿到 720 亿**不等，具备强大的视觉和智能体（agentic）能力。**LangChain** 与 **LangGraph** 的集成使 AI 聊天机器人具备了记忆和工具使用能力，包括
  **DeFi Agent**（去中心化金融智能体）等应用。相关讨论强调了**英伟达（NVIDIA）**在硬件加速方面的作用，同时也对因 **DeepSeek**
  的高效和市场恐慌导致的股价下跌表示担忧。尽管效率有所提升，但在推理扩展和 MoE 设计改进的推动下，计算需求预计仍将增长。'
id: bfd6ecb6-ea05-4120-af25-be51bdfc677c
models:
- deepseek-r1
- deepseek-v3
- qwen2.5-vl
- o1
original_slug: ainews-deepseek-1-on-us-app-store-nvidia-stock
people:
- sama
- mervenoyann
- omarasar0
- teortaxestex
- nptacek
- carpeetti
- finbarrtimbers
- cwolferesearch
- arthurrapier
- danhendrycks
- scaling01
- janusflow
title: DeepSeek 登顶美国 App Store，英伟达股价暴跌 17%。
topics:
- moe-architecture
- chain-of-thought
- fp8-precision
- multimodality
- vision
- agentic-ai
- inference-scaling
- gpu-optimization
- model-efficiency
- ai-chatbots
- memory-integration
- tool-use
- stock-market-reactions
---

<!-- buttondown-editor-mode: plaintext -->**DeepSeek is all you need.**

> 2025年1月24日至1月27日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 服务器（**225** 个频道和 **11316** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1229 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们一直努力保持新闻报道的技术性，但在极少数情况下，主流/非技术性新闻的影响力如此之大，以至于它也出现在了这里。

今天就是这样的日子。

[/r/LocalLlama](https://www.reddit.com/r/LocalLLaMA/comments/1iasyc3/deepseek_is_1_on_the_us_app_store/):


![image.png](https://assets.buttondown.email/images/a7439f78-2553-49dd-8169-5199b2f1c32c.png?w=960&fit=max)


以及 [sama](https://x.com/sama/status/1884066337103962416):


![image.png](https://assets.buttondown.email/images/f608fcb6-fa5f-4798-a202-4434d1872e47.png?w=960&fit=max)


最终，大部分讨论都非常无益，看起来就像这个版本的变体：


![image.png](https://assets.buttondown.email/images/4861bf2e-0401-4526-a859-1161a880876b.png?w=960&fit=max)


我们主要报道 DeepSeek 进入主流新闻的文化时刻，这从未出现在我们对 2025 年的预测中。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与增强**

- **DeepSeek-R1 和 V3 的效率**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1883976116949639568) 讨论了 **V3** 如何展示出比之前的 **67B 模型**快 **42%** 的速度来训练 **236B 模型**的能力，并利用 **fp8** 精度为更大型的模型保持速度。[@nptacek](https://twitter.com/nptacek/status/1883920168952422789) 强调 **DeepSeek-R1** 需要大量的 GPU，并重点介绍了其具有 **671B 参数**的 **MoE 架构**。[@carpeetti](https://twitter.com/casper_hansen_/status/1883974834025292047) 赞扬了 **DeepSeek-R1** 的**思维链 (CoT)** 能力，能以极低的成本媲美 **OpenAI 的 o1**。

- **Qwen2.5 模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1883954645602906249) 宣布发布 **Qwen2.5-VL**，这是一款能够处理**图像和视频**的**多模态模型**，版本包括 **3B、7B 和 72B 参数**。[@omarasar0](https://twitter.com/omarsar0/status/1883965524205359460) 详细介绍了 **Qwen2.5** 强大的**视觉能力**和 **Agent** 特性，支持**长视频理解**和**结构化数据输出**。

- **LangChain 和 LangGraph 集成**：[@LangChainAI](https://twitter.com/LangChainAI/status/1883666232789889259) 分享了使用 **LangGraph** 构建 **AI 聊天机器人**的教程，实现了**记忆和工具集成**。他们还展示了 **DeFi Agent** 等应用，可自动化 **Aave 协议操作**。

**计算与硬件**

- **NVIDIA 的影响**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1883979845266465076) 对在 **32K Ascend 910C 集群**上进行训练表示担忧，暗示可能会做空 **NVIDIA** 股票。[@samyj19](https://twitter.com/giffmana/status/1883662627920031857) 和 [@ykylee](https://twitter.com/giffmana/status/1883661880822284792) 讨论了 **DeepSeek-R1** 的**推理速度优化**，利用 **NVIDIA H800** GPU 来提升性能。

- **算力需求**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1883961408553116091) 认为，尽管 **DeepSeek** 效率很高，但由于**推理侧扩展 (inference scaling)**，**算力需求**仍将**增加**。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391) 分析了 **DeepSeek-v3** 的 **Mixture-of-Experts (MoE)** 设计，强调了其在**效率和性能**方面的改进。

**AI 竞争与市场反应**

- **股市反应**：[@MiddleOpenAI](https://twitter.com/nearcyan/status/1883944036811096517) 报道称，在 **DeepSeek** 取得进展后，由于**市场恐慌**，**NVIDIA** 的股价大幅下跌 **-17%**。[@arthurrapier](https://twitter.com/fchollet/status/1883973637075816555) 同样表达了对 **NVIDIA** **看跌信号**的担忧，而 [@DanHendrycks](https://twitter.com/DanHendrycks/status/1883660982641426727) 等人则强调了由于**芯片供应链依赖**而导致的**战略脆弱性**。

- **竞争格局**：[@scaling01](https://twitter.com/scaling01/status/1883912104182452629) 批评了**市场对 DeepSeek 的反应**，认为 **DeepSeek** 的效率挑战了**高利润模型背后的假设**。[@janusflow](https://twitter.com/janusflow/status/1883932760940888071) 指出 **DeepSeek** 的发布对**技术生态系统**具有**颠覆性**，引发了**市场波动**。

**AI 应用与用例**

- **Agent 能力**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1883954202733827584) 介绍了 **Grace, Kane 和 Flows**，这些 AI **Agent** 能够在**电脑和智能手机**上**执行命令**，展示了**实时交互**和**多步推理**能力。

- **历史研究与药物研发**：[@omarsar0](https://twitter.com/omarsar0/status/1883890211538776501) 探讨了 **LLMs** 在**历史研究**中的应用，例如**转录早期现代意大利语**和**生成历史解读**。此外，还讨论了通过**幻觉特性**与**药物研发**的结合。

- **视频与图像处理**：[@mervenoyann](https://twitter.com/mervenoyann/status/1883916608961479034) 展示了 **DeepSeek** 的 **Janus-Pro** 用于**多模态图像生成**，超越了 **DALL-E** 等模型。[@chethaan](https://twitter.com/chethaan/status/1883923932786655491) 强调了 **NVIDIA** 的 **Cosmos Tokenizer** 用于**物理 AI 训练**，增强了**图像和视频的 Token 化**。

**技术讨论与创新**

- **强化学习与训练效率**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1883968204650996137) 强调了**强化学习 (RL)** 在 **DeepSeek** 模型中的重要性，突出了 **DeepSeek Zero 范式**中独立的并行工作。[@lateinteraction](https://twitter.com/lateinteraction/status/1883939171926241324) 讨论了并没有所谓的秘密革命性技术，将成功归功于**工程精度**。

- **量化技术**：[@danielhanchen](https://twitter.com/danielhanchen/status/1883901952922448162) 详细介绍了 **DeepSeek R1** 量化至 **1.58bit** 的过程，通过**动态量化**在保持可用性的同时实现了 **80% 的体积缩减**。这一创新使模型能够在更**普及的硬件**上运行。

- **Mixture-of-Experts (MoE) 模型**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391) 解释了 **DeepSeek-v3** 的 **MoE** 架构，包含**共享专家**和**多 Token 预测**，提升了**训练效率**和**模型性能**。

**AI 业务与市场反应**

- **开源 vs. 闭源模型**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1883946119723708764) 倡导**开源 AI**，表示：“**AI 不是一场零和博弈。开源 AI 是惠及所有人的浪潮！**” [@cwolferesearch](https://twitter.com/cwolferesearch/status/1883885191661326391) 也表达了类似观点，赞扬了 **DeepSeek** 开源模型的**透明度**和**成本效益**。

- **投资策略**：[@swyx](https://twitter.com/swyx/status/1883961408553116091) 建议不要**做空 NVIDIA**，认为 **DeepSeek** 的进步**推动了算力需求**而非减少。相反，[@scaling01](https://twitter.com/scaling01/status/1883944036811096517) 基于 **DeepSeek** 对 **AI 算力经济学**的影响提出了**做空**策略。

- **招聘与人才获取**：[@AlexAlbert__](https://twitter.com/alexalbert__/status/1883907893294170610) 等人提到了 **Anthropic** 等 **AI 公司**的**招聘机会**，强调需要**多元化的技术背景**来驱动**未来的 AI 创新**。


---

# AI Reddit Recap

## /r/LocalLlama Recap

**主题 1. DeepSeek 登顶美国 App Store：市场影响**

- **[Deepseek 登顶美国 App Store](https://i.redd.it/sr4kvvnv3ffe1.jpeg)** ([Score: 1618, Comments: 341](https://reddit.com/r/LocalLLaMA/comments/1iasyc3/deepseek_is_1_on_the_us_app_store/)): **Deepseek** 已在 **U.S. App Store** 的“免费应用排行榜”中位列第一，超越了 **ChatGPT** 和 **Threads** 等知名应用。这一排名突显了其作为**智能 AI 助手**的竞争优势，并对其在面对成熟 AI 工具时的市场地位产生了影响。
  - 舆论对 **Deepseek** 的竞争优势持怀疑态度，并担心它会因为潜在的国家安全风险而面临与 **TikTok** 类似的命运。一些用户对高流量导致的服务器宕机表示沮丧，而另一些用户则质疑与 **ChatGPT** 和 **Perplexity** 等其他 AI 模型相比，它能为普通用户提供哪些独特价值。
  - 讨论强调了 **Deepseek** 的**开源特性**，用户指出其模型权重和训练方法可能会发布，从而使其更易于获取。一些用户讨论了本地运行 **Deepseek** 的可行性，并提到了可以在消费级硬件上运行的**蒸馏模型 (distilled models)**，尽管完整模型需要大量资源。
  - 评论反映了关于 AI 发展中**全球竞争**的更广泛对话，一些用户批评了“护城河”的概念，并强调多个国家都能开发出具有竞争力的软件。此外，还有关于**美国**技术竞争方式的看法，以及 **Deepseek** 的开源方法对国际动态影响的辩论。


- **[OpenAI 员工对 Deepseek 的反应](https://i.redd.it/ij7ubrn3mkfe1.jpeg)** ([Score: 1239, Comments: 256](https://reddit.com/r/LocalLLaMA/comments/1ibej82/openai_employees_reaction_to_deepseek/)): **OpenAI** 员工 Steven Heidel 批评了与 **DeepSeek** 相关的数据隐私问题，暗示美国人正在为了免费服务将数据交给 **CCP**。讨论强调，与 OpenAI 的模型不同，**DeepSeek** 可以在没有互联网连接的情况下本地运行。
  - **开源与本地运行**：许多评论者强调 **DeepSeek** 是开源的，可以在本地或云端硬件上运行，这解决了对数据隐私和对外国实体依赖的担忧。**TogetherAI** 被提及为一个托管该模型的服务，且不使用数据进行训练，为本地运行提供了替代方案。
  - **审查与模型透明度**：舆论对 AI 模型的透明度持怀疑态度，一些用户注意到 **DeepSeek** 在对齐 **CCP** 叙事方面表现出审查倾向，这强调了对像 **HuggingFace** 正在开发的这类真正开放模型的需求。
  - **硬件与可访问性**：关于运行 **DeepSeek** 等大型模型的硬件要求的讨论强调，虽然个人可能缺乏资源，但资金充足的初创公司可能负担得起必要的基础设施。一些用户提到了具体的硬件配置，例如使用 **30 块 3090/4090** 或 **9 块 80 GB 大显存 GPU** 来满足模型的需求。

- **1.58bit DeepSeek R1 - 131GB 动态 GGUF** ([Score: 552, Comments: 125](https://reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/)): 该帖子讨论了 DeepSeek R1 671B MoE 模型的 **dynamic quantization**（动态量化）至 GGUF 格式的 **1.58bits**，通过仅将 MoE 层量化为 **1.5bit**，同时保持 attention 和其他层为 **4 或 6bit**，有效地将磁盘占用减少到 **131GB**。这种方法防止了产生乱码和无限重复的问题，在 2x H100 80GB GPU 上实现了 **140 tokens/s** 的处理速度，并成功在特定条件下生成了一个 **Flappy Bird** 游戏。更多资源和细节可以在 [Hugging Face](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S) 和 [Unsloth 博客](https://unsloth.ai/blog/deepseekr1-dynamic)上找到。
  - **量化策略**：成功量化 **DeepSeek R1 671B MoE 模型** 的关键是仅将 MoE 层量化为 **1.5bit**，同时保持其他层具有更高的精度（4 或 6 bits），这符合 **BitNet 论文** 的原则，即建议保留某些层的高精度以优化性能。这种方法防止了计算成本过高的问题，并保持了模型执行复杂任务的能力，例如生成 **Flappy Bird** 游戏。
  - **兼容性与实现问题**：用户讨论了在不同设置下运行模型的挑战并寻求指导，例如 **Ollama**、**LM studio** 和 **llama.cpp**，强调了理解具体实现和兼容性问题的重要性。还有关于硬件要求的咨询，一位用户指出 **24GB GPU（如 RTX 4090）** 应该能达到 **1 到 3 tokens/s** 的速度。
  - **社区反馈与性能预期**：社区对该模型的性能给出了显著的正面反馈，用户对其能力表示惊讶，特别是生成无错的 **Flappy Bird** 游戏的能力。用户还讨论了潜在的性能指标，例如在不同硬件配置上的推理速度，并对基准测试以及与 **Q2KS** 等其他模型的比较表示了兴趣。


**主题 2. Deepseek 如何降低 95-97% 的成本**

- **Deepseek 究竟为何如此便宜？** ([Score: 386, Comments: 334](https://reddit.com/r/LocalLLaMA/comments/1ib4ksj/how_exactly_is_deepseek_so_cheap/)): Deepseek 通过采用避免 **RLHF (Reinforcement Learning from Human Feedback)**、利用 **quantization** 以及实施 **语义输入 HTTP 缓存** 等策略，实现了 **95-97% 的成本削减**。然而，关于 R1 是否经过量化存在困惑，引发了关于潜在补贴或 **OpenAI/Anthropic** 是否过度收费的问题。
  - 讨论强调了 **Deepseek** 的成本节约策略，重点是将 **MoE (Mixture of Experts)**、**FP8** 精度和 **multi-token prediction (MTP)** 作为关键因素。这些技术选择，加上**廉价的电力**和**较低的研发成本**，使其与 **OpenAI/Anthropic** 相比实现了显著的成本降低。一些用户怀疑存在**政府补贴**或**亏本经营**以获取市场份额。
  - 也有人对 Deepseek 运营的**真实成本**和**效率**持怀疑态度，一些评论者质疑其定价模型的**财务透明度**和**可持续性**。有人担心他们是否在使用更便宜的 **Nvidia H800** 芯片，以及 **OpenAI/Anthropic** 是否因为潜在不可持续的商业模式而过度收费。
  - Deepseek 模型的开源性质（可在 **Huggingface** 等平台获取）被视为一种竞争优势，允许广泛采用和**托管的灵活性**。然而，一些用户对这些模型的**运行质量**和**性能**表示怀疑，报告了翻译能力方面的问题，并质疑 Deepseek 声明的**可信度**。


**主题 3. 本地 LLM 兼容性新工具：“Can You Run It?”**

- **有人需要为开源 LLM 创建一个“你能运行它吗？”工具** ([Score: 298, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1iaubfm/someone_needs_to_create_a_can_you_run_it_tool_for/)): 一位非技术用户表达了对类似于 **System Requirements Lab** 的工具的需求，用于开源 **LLM**（如 **Deepseek, LLaMA,** 和 **Mistral**），以确定这些模型是否可以在其硬件上运行。他们提议建立一个系统，用户可以输入电脑配置，从而获得直观的性能评估和优化建议，例如使用量化版本以更好地兼容低端系统。
  - 提到了几种用于确定 **LLM** 是否可以在特定硬件上运行的工具和资源，包括 **[Vokturz's can-it-run-llm](https://huggingface.co/spaces/Vokturz/can-it-run-llm)** 和 **[NyxKrage's LLM-Model-VRAM-Calculator](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)**。这些工具帮助用户计算 VRAM 需求并评估模型与系统的兼容性。
  - 社区成员分享了估算硬件需求的经验法则，例如 **每 1B 参数 1GB** 和 **每 1K 上下文 1GB**，并推荐在低端系统上使用 **llama 3.2** 或 **Qwen 2.5** 以获得最佳性能。他们还讨论了量化和上下文长度对性能及内存占用的影响。
  - 用户（如 **Solid_Owl** 和 **Shark_Tooth1**）对易于使用、开源且能保护隐私并保持模型需求更新的工具有需求，他们表达了对隐私的担忧，并寻求针对本地 **LLM** 使用的可靠性能预期工具。


- **我为开源 LLM 创建了一个“你能运行它吗”工具** ([Score: 261, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1ib2uuz/i_created_a_can_you_run_it_tool_for_open_source/)): 我为开源 **LLM** 创建了一个 **“你能运行它吗”工具**，该工具提供 **tk/s 预估**，以及在 GPU 上运行模型（包含 **80% 层卸载** 和 **KV 卸载** 等选项）的说明。该工具已在 **Linux 单 Nvidia GPU** 环境下测试，现征求其他系统（包括多 GPU 设置）的反馈，以识别潜在问题。[GitHub 链接](https://github.com/Raskoll2/LLMcalc)。
  - **Mac 兼容性**：**Environmental-Metal9** 调整了针对 macOS 的计算，报告了 **M1 Max** 上性能预估的差异。他们提出通过 Pull Request 或 Pastebin 贡献 **Mac 支持** 的补丁。
  - **用户界面建议**：包括 **Catch_022** 和 **MixtureOfAmateurs** 在内的用户建议通过创建带有 **GUI 的便携式可执行文件** 或将其托管为 **网站** 来简化工具的可用性，从而消除安装 Python 的需求。
  - **Web 界面与变现**：**Whole-Mastodon6063** 为该工具开发了 Web 应用界面，**mxforest** 建议通过在线托管并投放广告来获取潜在收益，**Ok-Protection-6612** 和 **femio** 支持通过赞助进行变现的想法。


**主题 4. Qwen 3.0 MOE: 新兴推理模型**

- **[Qwen3.0 MOE? 新的推理模型?](https://i.redd.it/0vnua5vqxjfe1.png)** ([Score: 239, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1ibb8rr/qwen30_moe_new_reasoning_model/)): **Binyuan Hui** 的一条推文暗示了 **Qwen3.0 MOE** 和一个潜在的 **新推理模型**，预示着即将发布的公告或活动。该推文暗示了 AI 领域的重大进展，尽管未提供具体细节。
  - **Qwen2.5-VL** 已确认是即将发布的版本之一，并在 [Hugging Face](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) 上创建了合集。这表明除了视觉模型之外，可能很快会有包括 **Qwen MoE** 和 **Qwen 3.0** 在内的更新。
  - **DeepSeek** 被提及为处理巨大算力需求的合作伙伴，一些用户希望看到采用 **Apache/MIT 许可证** 的新推理模型。人们对各种模型尺寸和功能充满期待，包括音频模型和像 **Qwen 2.5 100B+** 这样的大规模模型。
  - 由于临近 **春节假期**，发布公告的时机受到质疑，人们对发布前的炒作持怀疑态度。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Nvidia 股票波动：DeepSeek 高效模型的影响**

- **[Nvidia 泡沫破裂](https://i.redd.it/xpen835nbkfe1.png)** ([评分: 627, 评论: 245](https://reddit.com/r/OpenAI/comments/1ibd2p8/nvidia_bubble_bursting/)): **Nvidia 的股价**经历了显著下跌，五天内下跌了 **$17.66 (12.68%)**，从 2023 年 1 月 24 日的峰值 **$142.02** 跌至 1 月 27 日的 **$121.56**。该公司的**市值 (market cap)** 据报道为 **$2.97 万亿**，**市盈率 (P/E ratio)** 为 **48.07**，52 周波动范围在 **$60.70** 至 **$153.13** 之间。
  - 许多评论者将股价下跌视为**买入机会**，多人表示相信 **Nvidia** 将会反弹。**itsreallyreallytrue** 和 **AGIwhen** 认为，尽管 **DeepSeek** 声称降低了 GPU 需求，但由于其在 AI 基础设施中的关键作用，对 Nvidia GPU 的需求依然强劲。
  - 讨论中表现出对 **DeepSeek** 的主张及其对 Nvidia 股价影响的怀疑，**TheorySudden5996** 和 **Agreeable_Service407** 指出，尽管效率可能提升，但对 GPU 的需求仍然巨大。**DerpDerper909** 认为，即使效率有所提高，Nvidia 也会从小型公司开发 AI 模型门槛降低中受益。
  - **Cramer4President** 等人批评了基于短期股票表现就认为“泡沫破裂”的观点，主张从更长的时间跨度来看待。**OptionsDonkey** 和 **Legitimate-Arm9438** 强调 Nvidia 的长期价值依然强劲，认为目前的下跌是暂时波动而非根本性问题。


- **[这是关于 DeepSeek 的吗？你认为他真的担心吗？](https://i.redd.it/v8oe8q5seife1.jpeg)** ([评分: 540, 评论: 203](https://reddit.com/r/OpenAI/comments/1ib4vq7/was_this_about_deepseek_do_you_think_he_is_really/)): **Sam Altman** 强调了与复制现有成功想法相比，创建创新且具有风险的项目的挑战，并强调了认可个人研究人员突破性工作的重要性。他最后总结道，这些努力代表了“世界上最酷的事情”。
  - **对 Sam Altman 言论的批评**：许多评论者批评了 **Sam Altman** 对个人研究人员的强调，认为突破往往是协作努力的结果。**Neofelis213** 强调了“孤独研究者”的迷思，并指出像 **Sam Altman** 和 **Elon Musk** 这样的人物往往掩盖了技术进步的实际贡献者。
  - **历史背景与贡献**：讨论集中在 **Transformer 架构**和 **LLM** 的起源上，用户指出 **Google** 发表了奠基性论文《Attention Is All You Need》，OpenAI 在此基础上进行了构建。**coloradical5280** 等人强调了这些发展的协作性质，以及 **Ilya Sutskever** 等关键人物在技术演进中的作用。
  - **伦理与版权担忧**：一些评论提到了在训练 AI 模型中使用受版权保护材料的伦理影响，**Riegel_Haribo** 提到了 AI 训练中涉及的大规模版权侵权。这引发了关于使用公共数据的合法性和公平性的辩论，并引用了 **Aaron Swartz** 被起诉等历史案例。

- **“每个模型都有审查”是一个无知的论点** ([Score: 179, Comments: 146](https://reddit.com/r/OpenAI/comments/1iazo74/every_model_has_censorship_is_an_ignorant_argument/)): 该帖子批评了西方对 **DeepSeek** 和 **ChatGPT** 审查制度的看法，认为虽然两者都有审查，但 **CCP** 的审查危害更大，因为它压制了对威权权力的批评。作者强调，与西方替代方案不同，**Chinese AI models** 普遍受到政府审查，并强调了 **CCP** 统治下中国公民受到的剥削，许多人年收入不足 **$4,000** 且缺乏免费医疗。该帖子谴责西方人为了廉价的中国产品而忽视这些问题。
  - 几位评论者认为，**审查制度和威权主义**并非中国独有，因为美国也从事类似的做法，包括在 **Gemini** 和 **ChatGPT** 等 **AI models** 中进行审查，以及对无证劳工的依赖。他们认为，西方 **AI models** 也受到审查以保护政治利益，且美国自身也存在贫富差距和剥削问题。
  - 讨论强调了 **AI** 技术的**剥削性质**，指出数据集通常是利用未支付报酬的知识产权汇编而成的，而创建和维护这些技术所涉及的劳动价值被低估。评论者批评了在谴责中国做法的同时忽视西方国家类似问题的虚伪行为，例如 **Lockheed Martin** 在政府支出中的角色，以及像 **Larry Ellison** 这样的亿万富翁在 **AI** 监控中的作用。
  - 一些评论者对**审查制度对 AI 发展的影响**表示怀疑，认为像 **HuggingFace** 上的开源项目可以绕过审查。他们指出，随着 **R1** 等模型被逆向工程，**AI** 的快速进步削弱了审查的力量，因为更多的模型是在本地开发的，且限制更少。


**Theme 2. DeepSeek R1's Coding Efficiency vs OpenAI O3**

- **[DeepSeek R1 比 o1 便宜 25 倍，且在相同成本下，其编码基准测试表现优于“未发布”的 o3。DeepSeek 正在让 OpenAI 感到压力。](https://i.redd.it/w6rngm2iyhfe1.png)** ([Score: 355, Comments: 111](https://reddit.com/r/OpenAI/comments/1ib3j3a/deepseek_r1_is_25x_cheaper_than_o1_and_better_in/)): **DeepSeek R1** 的定位是比 **OpenAI** 的 **o1** 模型便宜 **25x**，并在相同成本下展示了优于“未发布”的 **o3** 的编码性能。图形数据突显了 **DeepSeek R1** 在编码基准测试中 **15.8%** 的优异表现得分，强调了其成本效益以及相对于其他模型的竞争优势。
  - 几位评论者对 **DeepSeek R1** 性能声明的可信度表示质疑，指出数据中存在**问号**，且需要第三方验证。人们对论文的方法论以及缺乏关于训练硬件的可验证信息表示担忧。
  - 人们对 **DeepSeek** 推广的频率和性质表示怀疑，认为可能存在蓄意的宣传活动或“**astroturfing**（草根营销）”。评论者将其与 **Claude** 和 **Gemini** 等其他模型的推广进行了比较，指出存在类似的激进营销模式。
  - 一些用户对 **AI** 领域竞争的加剧表示支持，希望能有更多像 **Meta** 和 **Claude** 这样的参与者。然而，其他人对关于 **DeepSeek** 的大量宣传帖子感到沮丧，质疑其与 **OpenAI** 和 **Claude** 等成熟模型相比的实际效用和性能。


**Theme 3. Debates on DeepSeek vs ChatGPT: A Censorship Perspective**

- **[受 Octopus 启发的 Logarithmic spiral manipulator 可以操纵多种物体](https://v.redd.it/abwbgx3e1hfe1)** ([Score: 537, Comments: 41](https://reddit.com/r/OpenAI/comments/1ib0pow/octopusinspired_logarithmic_spiral_manipulator/)): 帖子标题讨论了一种受 **Octopus** 启发的 **Logarithmic spiral manipulator**，它能够处理各种物体。关于 AI 在政治审查中的伦理影响并未被直接提及，这表明主题与标题之间可能存在混淆。
  - **技术起源**: **Logarithmic spiral manipulator** 技术由 **University of Science and Technology of China** 开发，并在中国进行了测试。这澄清了关于起源的任何困惑，因为一些评论误将其归功于日本。
  - **设计与构造**: 该机械手似乎由 **3D printed pieces** 构造而成，并通过两侧的两条线进行操作，强调了软件在其功能中的重要作用。人们对软件是否会 **Open Source** 表现出浓厚兴趣，这可能会使其更易于使用。
  - **公众反应与幽默**: 讨论中包含了幽默和反乌托邦式的反应，提到了 **Robot tentacles** 及其在 **War and entertainment** 场景中的潜在用途。这突显了围绕先进机器人技术的复杂情感和想象力推测。

---

# AI Discord 摘要回顾

> 由 o1-preview-2024-09-12 生成的摘要之摘要的总结

**主题 1. DeepSeek R1 模型颠覆 AI 格局**

- **DeepSeek R1 缩减至 1.58-Bit，威力不减！**：社区对 [DeepSeek R1 在 1.58-bit 量化下运行](https://x.com/UnslothAI/status/1883899061893546254) 感到惊叹，其体积从 **720GB** 缩减至 **131GB**，却保留了完整的推理能力。
- **DeepSeek R1 正面挑战 OpenAI O1**：用户将 **DeepSeek-R1** 与 **OpenAI** 的 **O1** 进行对比，指出 R1 在 [aider's polyglot](https://aider.chat/2025/01/24/r1-sonnet.html) 等基准测试中的表现与 O1 持平甚至超越。
- **DeepSeek 的亮相震动科技市场**：据报道，DeepSeek 的 R1 导致美国科技股下跌 **6000 亿美元**，引发了关于中国 AI 实力崛起的讨论；详情见 [这段 Bloomberg 视频](https://www.youtube.com/watch?v=7GV_OdqzmIU)。

**主题 2. Qwen2.5 模型打破上下文壁垒**

- **Qwen2.5 发布 100 万 Token 上下文——越大越好吗？**：阿里巴巴发布了具有惊人的 **100 万 Token 上下文长度** 的 [Qwen2.5 模型](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M)，引发了关于如此长上下文实用性的辩论。
- **Qwen2.5-VL 擅长 OCR——手写识别不在话下！**：全新的 [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) 以其先进的 OCR 能力给用户留下深刻印象，包括手写分析和强大的视觉内容解析。
- **Qwen2.5-VL 对标 DALL-E 3：视觉语言模型之战**：用户将 Qwen2.5-VL 的视觉理解能力与 **DALL-E 3** 等模型进行对比，强调了其在金融和商业任务中处理结构化数据输出的能力。

**主题 3. AI 工具进阶，融入开发者工作流**

- **RAG 策略引发开发者讨论**：开发者深入研究检索增强生成（RAG）方法，讨论向量数据库和嵌入模型，如 [voyage-code-3](https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.)。
- **代码库索引：过于简陋的方法？**：一些用户批评代码库索引工具未能充分利用项目文件，并引用了 [Cursor 的文档](https://docs.cursor.com/context/codebase-indexing)，而另一些人则认为在正确配置下它们非常有用。
- **AI 结对编程随 Aider 和 CodeGate 起飞**：[CodeGate 集成](https://docs.codegate.ai/how-to/use-with-aider) 允许开发者直接在终端进行结对编程，通过 AI 辅助增强编码工作流。

**主题 4. OpenRouter 扩展新模型与供应商**

- **Liquid AI 在 OpenRouter 上引起轰动**：Liquid AI 将 [多语言模型 LFM 40B, 7B 和 3B](https://openrouter.ai/liquid/lfm-7b) 引入 OpenRouter，声称在主要语言中拥有顶尖性能。
- **DeepSeek Nitro：够快但不够强？**：[DeepSeek R1 的 Nitro 变体](https://openrouter.ai/deepseek/deepseek-r1:nitro) 承诺更快的响应速度，但用户反馈其在实际场景中的表现并未超越标准版 R1。
- **OpenRouter 用户自带密钥 (BYOK)**：讨论强调在 OpenRouter 中使用个人 API 密钥以缓解速率限制并控制支出，使用时会收取 **5% 的费用**。

**主题 5. 全球 AI 政策与投资加剧竞争**

- **中国 1 万亿元豪赌 AI 引发全球竞赛**：据 [此处](https://x.com/rwang07/status/1883210410763121073) 报道，中国宣布对 AI 投资 **1 万亿元**（约 1370 亿美元），引发了关于美国能否跟上步伐的疑问。
- **大国竞争背景下的美国 AI 政策辩论**：讨论强调美国正考虑在大国竞争的旗帜下资助 AI，这与 **CHIPS Act** 等历史性产业政策有异曲同工之妙。
- **DeepSeek 的崛起引起地缘政治关注**：DeepSeek 模型的成功加剧了对中国在 AI 领域影响力日益增长的担忧，促使人们分析 AI 在国家竞争力中的作用。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 升级**：社区成员展示了运行在 **1.58-bit 量化**下的 **DeepSeek R1**，具备完全正常的推理能力，引用了[这条推文](https://x.com/UnslothAI/status/1883899061893546254)，并强调模型体积从 **720GB** 缩减至 **131GB**。
   - 他们将 **DeepSeek-R1** 与 **OpenAI** 的 **o1** 模型进行了对比，指出用户对本地使用和开源协作处理高级推理任务的兴趣日益浓厚。
- **Qwen2.5 百万 Token 发布**：聊天集中在阿里巴巴的 **Qwen2.5**，[Qwen 推文](https://x.com/Alibaba_Qwen/status/1883557964759654608)中透露了 **100 万 Token 的上下文长度**，并提到了 **14B** 参数的指令微调版本。
   - 成员们讨论了扩展的上下文长度是否名副其实，并对充足 **VRAM** 支持下的大规模本地推理（local inference）表示乐观。
- **SmoLlm 微调受到关注**：多位用户测试了使用 **Unsloth** 进行 **SmoLlm** 微调，并成功通过 **ollama** 部署，默认温度（temperature）为 `0.8`，这在[讨论线程](https://discord.com/channels/1179035537009545276/1179039861576056922/1332482886506385429)中得到了澄清。
   - 他们强调了与个人工作流的平滑集成，称其 *“无需显式温度设置即可正常工作”*，并确认已准备好用于本地代码审查任务。
- **数据集格式化的细节**：用户分享了关于使用 **'instruction'、'input' 和 'output'** 字段构建训练数据的技巧，参考了 [Wikimedia 数据集](https://huggingface.co/datasets/wikimedia/wikipedia)和个人论坛收藏的示例。
   - 他们指出字段名称不匹配会导致 **Unsloth** 在微调期间报错，强调需要**一致的问答格式**以确保模型行为正确。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **速度困扰还是速度提升？关注 Cursor IDE**：用户报告了**请求时间缓慢**以及在 **Cursor** 上使用 **Claude** 时的一些摩擦，[Fast and Slow Requests – Cursor](https://docs.cursor.com/get-started/usage#fast-and-slow-requests) 中描述了非高峰时段的部分修复方案。
   - 他们还称赞了 **DeepSeek R1** 在某些任务中的表现，并提到可以使用 [Spark Engine](https://sparkengine.ai) 作为补充，以减少请求时间和成本。
- **Claude vs DeepSeek：代码洞察之战**：**DeepSeek R1** 擅长规划任务，而 **Claude** 则能产生更高级的响应，如 [DeepSeek R1 - API 文档](https://api-docs.deepseek.com)所示。
   - 社区讨论指出，在简单任务中使用 **DeepSeek** 可以节省费用，但用户在处理繁重的代码生成和调试时仍依赖 **Claude**。
- **代码库索引：过于简陋？**：一些用户认为 **Cursor** 的**代码库索引（codebase indexing）**没有利用所有项目文件，如 [Context / Codebase Indexing – Cursor](https://docs.cursor.com/context/codebase-indexing) 所示。
   - 其他人则为其辩护，认为索引可以改进代码建议，并建议调整设置以获得更好的效果。
- **RAG 策略引发开发者对话**：关于检索增强生成（**RAG**）方法的讨论包括向量库和嵌入（embedding）方法，例如[这篇博客文章](https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.)中描述的 **voyage-code-3**。
   - 参与者强调，结构良好的嵌入和检索可以减少错误，并提高代码密集型项目的输出质量。
- **告别 Claude：替代方案层出不穷**：一些用户权衡在简单场景下用 **DeepSeek** 替换 **Claude**，同时也探索了 [GitHub Copilot](https://github.com/) 和 [Spark Engine](https://sparkengine.ai)。
   - 观点各异，但对话指向结合各平台的优势以实现平衡的工作流。



---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.2.2 发布，带来更流畅的 Cascade**：团队推出了 **Windsurf 1.2.2**，其特点是增强了 **Cascade** 内存并修复了对话卡顿问题，详见 [Windsurf Editor Changelogs](https://www.codeium.com/changelog)。**Cascade** 现在支持网页查询或直接 URL 输入，使 Prompt 交互更加动态。
   - 尽管有了这些修复，一些用户仍然遇到 **internal error** 激增，干扰了日常编码任务，引发了对进一步稳定性更新的呼吁。
- **性能受损且免费额度大幅削减**：各频道频繁出现的错误和延迟导致了用户沮丧和额度损失，削弱了对 **Windsurf** 的信任。如 [Pricing](https://codeium.com/pricing) 所示，免费计划现在仅提供 **5** 个高级模型额度（从 50 个下调）。
   - 社区反馈强调，这些变化限制了新用户的上手，并阻碍了调试（debugging）环节。
- **DeepSeek 的首次亮相仍不确定**：讨论表明，**DeepSeek** 集成到 Codeium 不会很快到来，这引发了对其错过低成本运营优势的担忧。一些用户公开表示，如果近期不公布时间表，将放弃 Windsurf。
   - 他们还质疑 DeepSeek 在工具调用（tool calls）方面的可靠性，希望 Codeium 能尽快解决这些疑虑。
- **Git 来救场**：成员们建议使用 **Git** 进行版本控制，以防 **Cascade** 和 **Windsurf** 更新触发意外错误。他们引用了 [Learn Git Branching](https://learngitbranching.js.org/) 等资源来保持代码稳定并维护进度。
   - 标记里程碑和回滚到之前提交（commits）等最佳实践，有助于在 AI 驱动的更改失败时防止重大损失。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的 R1 推出引发骚动**：**R1** 在 Perplexity Pro 上取代了 **O1** 和 **Grok** 等旧宠，将每日查询限制为 **10** 次，引发了用户抗议。
   - 一些用户威胁要转向 **DeepSeek** 或 **ChatGPT**，理由是性能欠佳且使用条款不明，而另一些人则通过 [此参考资料](https://intercom.help/perplexity-ai/en/articles/10354288-refunds) 要求退款。
- **DeepSeek 的数据困境引发分歧**：**DeepSeek** 由于中国所有权引发了隐私担忧，用户担心数据从美国路由到中国的做法。
   - 社区反馈毁誉参半，一些人看到了 **DeepSeek** R1 模型的潜力，但对敏感查询的**数据主权（data sovereignty）**表示怀疑。
- **十亿级构想：5000 亿美元的 AI 转型**：据 [此来源](https://www.perplexity.ai/page/stargate-project-InQ5ZvKETX6c5I6he1zc_A) 称，一笔传闻中的 **5000 亿美元** 交易可能会重塑 AI，引发了对自动化和机器学习未来方向的推测。
   - 贡献者认为这种可能性是高级研究资金的一个重大转折点，与过去推动 **新 AI 框架（AI frameworks）** 的繁荣期相似。
- **初创公司的 2 亿美元之路与标普指数飙升**：一位成员展示了如何通过 [Wingify 方法](https://www.perplexity.ai/page/wingify-T9bxT5tHSY2sRduhPzHIXg) 实现 **2 亿美元** 的初创公司退出，重点在于资源化扩展和投资者关系。
   - 他们还注意到 **S&P 500** 指数创下收盘历史新高，他们认为这为渴望复制这一成功的雄心勃勃的创始人增添了动力。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Psyche Nudges Collaboration**：Nous 在 **Solana** 上推出了 **Nous Psyche**，这是一个利用**异构计算 (heterogeneous compute)** 的开源生成式 AI 协作训练网络，代码已在 [GitHub](https://github.com/PsycheFoundation/psyche) 上共享。
   - 他们计划于 30 日与 **Solana Foundation** 合作举办测试网活动，并在其[博客](https://nousresearch.com/nous-psyche/)中引用了神话灵感来团结开发者。
- **DeepSeek R1 蒸馏 (Distillation) 取得进展**：研究人员引用了 **Distilling System 2 into System 1** 论文 ([arXiv](https://arxiv.org/abs/2407.06023v3))，为 **R1** 蒸馏模型提出了新的改进方案。
   - 他们还指出与 [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) 的潜在协同作用，以优化数据集的覆盖范围和处理。
- **LLM Live2D 桌面助手亮相**：新款 [LLM Live2D Desktop Assistant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) 支持 Windows 和 Mac，具有语音触发功能，并可通过屏幕感知实现完整的计算机控制。
   - 其方法融合了系统剪贴板检索和交互式命令，为用户的日常任务提供了一个生动的角色界面。
- **Qwen2.5-VL 突破 OCR 障碍**：新款 [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) 模型展示了先进的 OCR 能力，包括手写分析和强大的视觉内容解析。
   - 社区成员称赞其在图像上的强大文本识别能力，标志着多模态 (multi-modal) 任务的一次重大飞跃。
- **类人 LLM 论文引发伦理辩论**：论文 [Enhancing Human-Like Responses in Large Language Models](https://arxiv.org/abs/2501.05032) 探讨了提升 AI **自然语言理解 (natural language understanding)** 和**情感智能 (emotional intelligence)** 的精细技术。
   - 该研究强调了在用户参与度方面的提升，同时也引发了对偏见的担忧，呼吁更深入地审视人机交互动态。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas 与 O1 联动**：OpenAI 发布了一项新更新，使 ChatGPT Canvas 能够与 **OpenAI o1** 配合使用，并在所有层级用户的 macOS 桌面应用上渲染 **HTML** 和 **React** 代码。
   - 他们预告了即将面向 **Enterprise** 和 **Edu** 用户发布的版本，预示着专业环境下的功能扩展。
- **DeepSeek 震撼科技市场**：DeepSeek R1 与 **O1** 和 **GPT-4o** 正面交锋，因其代码生成准确性和成本优势而广受赞誉。
   - 据称其首次亮相导致美国主要科技股蒸发了近 **6000 亿美元**，引发了关于 AI 竞赛中更大颠覆的猜测。
- **O3 Mini 发布在即**：社区传闻指出 **O3 Mini** 可能很快发布，尽管有人担心它可能只是 **O1 Mini** 的小幅升级，缺乏真正的多模态 (multimodal) 功能。
   - [Sam Altman 的一条推文](https://x.com/sama/status/1883294216329281627)承诺为 Plus 层级提供**每日 100 次查询**，暗示即将扩大 operator 访问权限。
- **Token 的 Tiktoken 问题**：用户报告了关于 **Tiktoken** 将 token 拆分为单个字符的不确定性，导致在某些输入中产生混淆。
   - 这引发了关于特殊 token 限制的讨论，并指向了可能解释 Tiktoken 不规则合并规则的研究。
- **LangChain 的 ChatPrompt 与向量存储 (Vector Stores)**：成员们探索了将**向量存储 (vector store)** 文档输入到 LangChain 的 **ChatPromptTemplate** 中，并指出官方指导有限。
   - 他们考虑将标准 prompt 路径作为备选方案，等待更稳健配置的成功案例。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 因攻击受阻**：恶意攻击冲击了 **DeepSeek API**，导致间歇性停机和响应时间变慢，这一点已由 [DeepSeek Service Status](https://status.deepseek.com/) 和用户报告确认。
   - 一些人注意到对基于 R1 的产品需求激增，而另一些人则在推测这些攻击的严重程度，并探索 [OpenRouter 替代方案](https://openrouter.ai/deepseek/deeps)。
- **LLM 推理提供商：盈利还是陷阱？**：**推理提供商的盈利能力**取决于高利用率以抵消固定成本，成员们权衡了不同服务之间的各种定价模型。
   - 一些人认为低使用率会导致利润微薄，从而引发了关于与 [DeepSeek-R1-Nitro](https://openrouter.ai/deepseek/deepseek-r1:nitro) 等高流量发布版本协同效应的讨论。
- **R1 与 O1 的竞争升级**：一项新的基准测试暗示，**DeepSeek 的 R1** 搭配 Sonnet 在某些场景下可能优于 **O1**，这一结论得到了 [R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA](https://aider.chat/2025/01/24/r1-sonnet.html) 数据支持。
   - 怀疑者坚持认为 **O1 Pro** 在编程任务中表现更佳，而一些人则将希望寄托在 [Unsloth AI 的推文](https://x.com/UnslothAI/status/1883899061893546254)中提到的 R1 1.58-bit 格式上。
- **Qwen2.5 与 Janus-Pro 强势登场**：**Alibaba Qwen2.5-1M** 和 **Janus-Pro** 因其 100 万 token 的上下文能力而备受关注，这在 [Qwen 的推文](https://x.com/Alibaba_Qwen/status/1883557964759654608)和 [Janus-Pro 的提及](https://x.com/_akhaliq/status/1883914398127083665)中得到了强调。
   - 评论者认为它们是 **O1** 的强力竞争对手，并引用了 [DeepInfra](https://deepinfra.com/deepseek-ai/DeepSeek-R1) 上与 **DeepSeek-R1** 的对比。
- **Aider 与 CodeGate（及 Rust）结盟**：[CodeGate 集成](https://docs.codegate.ai/how-to/use-with-aider) 授权 **Aider** 用户直接在终端进行结对编程，通过 API key 在 OpenAI 和 Ollama 之间切换。
   - 其他人在 Aider 中测试了新的 **Rust crates** 以增强上下文，并注意到 **architect mode** 会隐藏编辑器模型的响应，导致他们请求 [GitHub 上的 bug 修复](https://github.com/Aider-AI/aider/issues/2929)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek Distill 备受好评**：参与者测试了 **DeepSeek R1 Distill** 的 Q3 和 Q4 量化版本，赞扬其强大的知识保留能力以及在 [LLM Explorer 目录](https://llm.extractum.io/list/?query=deepseek%20r1)上的表现。
   - 他们观察到高参数模型可以提升编程任务，但需要仔细规划并发和 VRAM 以实现流畅推理。
- **Chatter UI 困惑**：成员报告了将 **Chatter UI** 连接到 LM Studio 时的问题，将故障追溯到 [ChatterUI GitHub 仓库](https://github.com/Vali-98/ChatterUI)中的错误 URL 和端口冲突。
   - 他们敦促验证本地主机地址并对齐 LM Studio 识别的端点以稳定请求。
- **Apple M3 Max 的 Token 速度**：一些用户估计 **DeepSeek-R1** 在配备 48GB RAM 的 Apple M3 Max 上达到了每秒 16–17 个 token，强调了硬件限制。
   - 讨论集中在芯片架构的效率限制上，一些人考虑在 llama.cpp 中使用负载均衡方法以增加速度。
- **MoE 策略**：爱好者们研究了针对特定任务的 **Mixture of Experts (MoE)** 解决方案，注意到在代码生成工作流中潜在的性能提升。
   - 他们强调了内存考量，并指向 [MoE 资源](https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe)以获取实践见解和部署策略。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Liquid AI 进驻 OpenRouter**：Liquid AI 通过 **OpenRouter** 推出了新模型 [LFM 40B](https://openrouter.ai/liquid/lfm-40b)、[LFM 3B](https://openrouter.ai/liquid/lfm-3b) 和 [LFM 7B](https://openrouter.ai/liquid/lfm-7b)，扩展了多语言覆盖范围。
   - 他们将 **LFM-7B** 视为企业级聊天的首选，强调了其在主要语言中极高的性能尺寸比（performance-to-size ratio）。
- **DeepSeek Nitro：极速捷径还是令人失望？**：DeepSeek R1 的 **Nitro** 变体已发布，声称响应速度更快，详见[公告](https://openrouter.ai/deepseek/deepseek-r1:nitro)。
   - 一些用户反映其在实际表现中未能超越标准版 R1，反馈暗示沉重的用户需求导致了系统压力。
- **Amazon Nova 突然崩溃**：**Amazon Nova** 模型目前处于宕机状态，因为 Amazon Bedrock 将激增的使用量误判为密钥泄露，导致了误导性的 400 状态码。
   - 团队正在加紧解决这一上游问题，预计服务稳定后将发布官方更新。
- **DeepSeek 的超载考验**：频繁的 503 错误和缓慢的响应时间困扰着 **DeepSeek R1**，这指向了高流量以及潜在的恶意活动。
   - DeepSeek 限制了新用户注册并面临可靠性担忧，凸显了应对极高用户负载的挑战。
- **BYOK 势头强劲**：OpenRouter 的讨论强调了使用 **BYOK** 来缓解速率限制（rate limits）并控制费用，使用费为 5%。
   - 社区成员一致认为，接入个人密钥可以帮助避开瓶颈，尽管一些人担心成本管理的复杂性。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Janus 领跑**：DeepSeek 推出了 [Janus Pro 模型](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)，号称拥有先进的推理性能且无需高端 GPU，引发了关于超越美国基准测试新前沿的猜测。
   - 参与者称赞了 Janus Pro 改进的多模态理解能力，引用了题为 **Janus-Series: Unified Multimodal Understanding and Generation Models** 的白皮书，并讨论了其改变科技市场情绪的潜力。
- **Qwen2.5-VL 的视觉探索**：阿里巴巴 Qwen 发布了 [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) 以庆祝农历新年，重点强调了**长视频理解**和先进的视觉识别。
   - 讨论强调了其处理金融和商业任务中**结构化数据输出**的能力，而一篇[博客文章](https://qwenlm.github.io/blog/qwen2.5-1m/)详细介绍了其上下文长度达到 1M tokens，适用于更广泛的企业用例。
- **GPRO 在 PPO 中崭露头角**：一场激烈的对话围绕 GPRO 消除 **Value Function** 和 **Generalised Advantage Estimation (GAE)** 展开，声称它可能解决 PPO 中的 **stuck loss** 和早期收敛问题。
   - 用户指出 GAE 对折扣和（discounted sum）的依赖阻碍了某些场景下的扩展，而 GPRO 的全局归一化奖励保持了训练稳定，引发了对其与开源 RL 库集成的兴趣。
- **DSL 梦想与 PydanticAI 现身**：一位成员探索了使用 [PydanticAI](https://ai.pydantic.dev/) 在生产级生成式应用中实现**结构化输出**，建议它可以与 **LlamaIndex+LangChain** 集成。
   - 他们还讨论了构建一个将*自然语言*转换为 DSL 的**健身记录应用**，参考了 **Microsoft ODSL 论文** 以寻求部分解决方案，并强调了语音到 DSL 流水线的挑战。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 的双重重击：R1 与 Janus Pro**：DeepSeek 全新的 R1 模型因其权重开放（open-weight）的方法引发了巨大轰动，在推理和性能基准测试中以极低的训练成本匹配或超越了一些成熟的 LLM。行业传闻指向 Janus Pro (7B) 在 Hugging Face 上的发布，强调了其透明度以及在文本和图像方面的先进能力。
   - 怀疑论者质疑 R1 的泛化和推理极限，而其他人则称赞其在数学和编程任务中的显著飞跃。因此，像 Meta 这样的大型企业已经建立了“作战室（war rooms）”，以分析 DeepSeek 的训练配方（training recipes）和成本效率。
- **Qwen2.5-VL 点亮视觉语言融合**：阿里巴巴的 Qwen2.5-VL 首次亮相，具备强大的多模态能力，支持长视频理解和精确重定位。观察者将其与过去的重大发布进行了比较，指出视觉语言模型在感知和竞争方面可能发生的转变。
   - 开发者强调了在精选任务上的显著性能提升，引发了对实际应用场景的推测。官方演示和提交（例如 Qwen2.5-VL GitHub）展示了先进的图像到文本协同作用和长上下文处理能力的融合。
- **Nous Psyche 遭黑客攻击与 Solana 设置**：Nous Research 推出了 Nous Psyche，这是一个基于 Solana 的协作训练网络，旨在推动开放超级智能计划。尽管这一概念令人兴奋，但黑客攻击的消息动摇了人们对其安全措施的信任。
   - 讨论还涉及了在推进复杂生成模型方面，开放实验室与资金充足的封闭实验室之间的广泛问题。这次黑客攻击强调了将区块链生态系统与 AI 训练结合时，严格安全保障的重要性。
- **Tulu 3 vs Tulu 4 与偏好微调的困扰**：爱好者们重新审视了 Tulu3，注意到尽管通常倾向于在策（on-policy）方法，但在偏好微调（preference tuning）中使用了离策（off-policy）数据。这标志着在完善基于偏好的训练流水线方面仍存在持续的复杂性。
   - 对 Tulu4 的期待与日俱增，用户希望它能解决 Tulu3 面临的挑战。讨论突显了将偏好微调扩展到更广泛应用中尚未解决的挑战。
- **中国数千亿 AI 政策及其全球影响**：中国宣布投入 1 万亿元人民币（1370 亿美元）用于 AI，引发了关于研发（R&D）快速扩张的激烈推测。参与者注意到这与美国的《芯片法案》（CHIPS Act）等工业政策有相似之处，但质疑美国是否准备好匹配如此大规模的 AI 资金。
   - 随着共和党人可能在大国竞争的理想下资助 AI，国防相关的视角也随之出现。对于工程师来说，这些政策可能会提供更多尖端硬件和激励措施，从而加剧全球 AI 竞赛。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 引发热议**：成员们注意到其项目报告中提到的 **$5M** 训练成本（参考 **DeepSeek V3**），并提到了[作为确认的图片附件](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)。**DeepSeek R1** 声称能以极低的成本与 **o1** 等老牌闭源模型竞争，这引发了人们对开源竞争力的讨论。
   - 社区讨论强调了 **R1+Sonnet** 在 [aider 的 polyglot 基准测试中取得的新 SOTA 声明](https://aider.chat/2025/01/24/r1-sonnet.html)，而[暗示强劲结果的推文](https://x.com/lmarena_ai/status/1882875989610594542)让许多人对其更深层次的推理能力感到好奇。
- **Qwen2.5-VL 与 DALL-E 3 的对决**：阿里巴巴发布了 **Qwen2.5-VL**，这是一款旨在在视觉理解和定位方面超越 **DALL-E 3** 的多模态模型，详见其[官方公告](https://x.com/Alibaba_Qwen/status/1883954247743725963)。该模型在特定基准测试中也与 **Stable Diffusion** 展开竞争，并在其 [Qwen 集合](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)中强调了先进的图像生成功能。
   - 用户对比了 [GenEval 和 DPG-Bench](https://x.com/LiangWenfeng_/status/1883918900741763293) 上的指标，指出 **DeepSeek** 的 **Janus-Pro-7B** 也在多模态领域参与竞争，这引发了关于这些新模型的成本效益和实际应用性的更广泛讨论。
- **Operator 与推理模型掀起波澜**：参与者称赞 **Operator** 能够快速生成初始代码库，但也对其处理复杂站点和视频采样率的能力表示担忧，如[此视频演示](https://x.com/klazuka/status/1883880742322888903)所示。与此同时，关于 **R1** 等推理模型的讨论表明，它们在编程任务及其他领域具有先进的 Agent 能力。
   - 一些人归功于 [function calling 基准测试](https://x.com/_philschmid/status/1883055262669349287)指出了多步约束，为 **DeepSeek R1** 等模型在集成到开发流水线时如何处理复杂工作流提供了视角。
- **Model Context Protocol (MCP) 势头强劲**：成员们对 **MCP** 作为跨工具集成 AI 功能的统一方法表现出热情，参考了用 **Go**、**Rust** 甚至 **assembly** 构建的服务器，详见 [MCP server 仓库](https://github.com/modelcontextprotocol/servers)。他们比较了如何通过 [mcp-obsidian](https://github.com/MarkusPfundstein/mcp-obsidian) 等插件将其与 **Obsidian** 互连，用于转录和文档记录。
   - **MCP party** 的计划鼓励社区反馈和协同，并呼吁查阅[最新规范](https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability)和教程，凸显了对一致的跨应用协议的浓厚兴趣。
- **Latent Space 发布新播客**：简要提到 **Latent Space** 播客发布了新的一集，[在此分享](https://x.com/latentspacepod/status/1883354909367787565)。
   - 社区对这一更新表示欢迎，期待能进一步深入探讨这些新兴 AI 技术和协作计划。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **层收敛与 Tokenization 策略**：在 [一篇 ICLR 2023 论文](https://openreview.net/forum?id=wlMDF1jQF86) 中，成员们观察到 **Layer Convergence Bias** 如何使浅层比深层学习得更快。
   - 另一组人对 [Armen Aghajanyan 论文](https://arxiv.org/pdf/2412.16326) 中提出的新 **Causally Regularized Tokenization** 表示赞赏，并指出其在 **LlamaGen-3B** 中提升了效率。
- **DeepSeek R1 与 GRPO 的差距**：参与者对 **DeepSeek** 声称 R1 使用廉价芯片的说法表示质疑，参考了约 **160 万美元** 的训练成本以及开源细节的匮乏。
   - 他们还发现 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 或 [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason) 中几乎没有真正的 **GRPO** 实现，暗示真实的 R1 运行主要依赖于 **PPO**。
- **AlphaZero 的演进与好奇心驱动的 AI**：采用者认可 **AlphaZero** 的精简设计，但指出实际设置中很少直接跳到其技术。
   - 一些人提到了 **empowerment** 概念（[维基百科条目](https://en.wikipedia.org/wiki/Empowerment_(artificial_intelligence))）和好奇心驱动的方法，认为它们是未来大规模训练的灵活方法。
- **Scaling Laws 与 20-Token 技巧**：一项 [Chinchilla 库分析](https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb) 表明，**20-tokens-per-parameter** 规则几乎与完全优化的 Chinchilla 设置相匹配。
   - 社区成员将此与 tokens-per-parameter 比率中的平坦极小值联系起来，表明微小的偏差可能不会大幅损害性能。
- **可解释性与多轮 Benchmark**：一些用户强调训练中的 **verified reasoning** 是可解释性的新优先级，关注 LLM 如何推理而不仅仅是输出。
   - 同时，**scbench**、**zeroSCROLLS** 和 **longbench** 等框架正在被集成，尽管它们的多轮特性可能需要不同的实现策略。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **System Prompt 的运用**：现在可以在 [Bolt](https://x.com/boltdotnew/status/1883949779572646008) 的项目或全局层级设置 **system prompt**，让你从一开始就注入首选的库和方法。
   - 社区成员正在交流塑造 **Bolt** 行为的最佳方式，并呼吁分享高级使用技巧以实现更顺畅的开发。
- **结构化与拆分策略**：成员们辩论了过于僵化的规划对创造力的影响，提到了重启循环以及在拆分复杂组件时对灵活方法的需求。
   - 一位用户推荐使用 **NEXTSTEPS.md** 大纲的系统化方法，并指出结构化迁移如何帮助保持清晰度而不扼杀新想法。
- **作为安全网的指南**：遵守 **GUIDELINES.md** 提高了稳定性，确保每个组件按顺序构建并与妥善管理的上下文窗口集成。
   - 参与者将避免混乱合并归功于这些护栏，稳定的文档实践为持续进展铺平了道路。
- **Bolt 的计费与错误困扰**：一些人抱怨大量的 Token 使用和频繁的 **rate limits**，并提到退款和成本差异问题。
   - 错误消息和网络故障让他们不得不寻求专业帮助，因为 **Bolt** 有时在没有交付结果的情况下消耗了大量 Token。
- **Supabase 角色权限的突破**：一位用户克服了复杂的 **Supabase** 策略，构建了包括超级管理员和管理员在内的多个登录角色，解决了递归陷阱。
   - 同时也探索了与 Netlify 和 GitHub 的集成，尽管私有仓库目前仍无法访问，这促使对 **Bolt** 的核心功能进行进一步修改。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Client 投诉与语音聊天抱怨**：开发者们正在努力解决无需重启即可动态更新 **MCP client** 工具的问题，并呼吁在 [multimodal-mcp-client](https://github.com/Ejb503/multimodal-mcp-client) 中提供更清晰的语音集成文档。
   - 许多关注点集中在 **server config** 的改进以及减少对私有 API 的依赖上，特别是针对 **Kubernetes** 部署。
- **Variance Log 工具捕捉异常**：**MCP Variance Log** 方案将低概率对话事件收集到 [SQLite database](https://github.com/truaxki/mcp-variance-log) 中，用于用户数据分析。
   - 采用者指出 **Titans Surprise mechanism** 是该方法的灵感来源，可以增强 Agent 工作流中的长期记忆。
- **KoboldCPP 与 Claude 建立新连接**：如 [GitHub](https://github.com/PhialsBasement/KoboldCPP-MCP-Server) 所示，一个新的 **KoboldCPP-MCP Server** 促进了 **Claude** 与其他 MCP 应用之间的 AI 协作。
   - 社区成员指出，这为更同步的任务和更深层次的 AI 到 AI 交互铺平了道路。
- **Inception Server 运行并行 LLM 任务**：**MCP Inception server** 允许使用各种参数进行并发查询，详见 [其仓库](https://github.com/tanevanwifferen/mcp-inception)。
   - 开发者计划通过爬虫扩展加密货币相关功能，暗示了更多的使用场景。
- **Shopify 商家与 Claude 对话**：一个 **MCP server for Shopify** 使用 Claude 进行店铺分析，如 [此仓库](https://github.com/amir-bengherbi/shopify-mcp-server) 所示。
   - 目前的端点集中在产品和订单上，为商家提供了一条直接获取 AI 驱动数据洞察的途径。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **HeyGen 场景与 ElevenLabs 音调**：一位用户展示了使用 **HeyGen** 和 **RunWayML’s Act-One** 制作逼真数字人视频的工作流，视频看起来像是在倾听，链接至 [UnrealMysteries.com](https://UnrealMysteries.com)。
   - 他们还展示了一个名为 "Thomas" 的 **ElevenLabs** 语音，带有一种 **HAL** 的氛围，增添了独特风格。
- **NotebookLM 用于播客摘要**：一位用户使用 **NotebookLM** 将每周新闻压缩成播客格式，并称赞其快速摘要的能力。
   - 其他人希望有更强大的 Prompt 来改进音频内容创作并提升工具的能力。
- **混合 HeyGen 与 MiniMax**：成员们尝试了混合内容，结合 **HeyGen** 静态图像和来自 **MiniMax** 的见解来制作长视频。
   - 他们观察到，相比单独使用其中一种技术，这种方式能产生更具吸引力的叙事，引发了进一步的创作尝试。
- **NotebookLM 的限制与困惑**：成员们在 UI 更改后遇到了 **NotebookLM** 中链接源丢失的问题，引发了对引用丢失的担忧。
   - 另一位用户发现了 **1000 条笔记** 的限制，呼吁为高级用法提供更清晰的文档。
- **语言切换与 PDF 页码争议**：一些人在默认语言设置上遇到困难，通过切换 [notebooklm.google/?hl=es](https://notebooklm.google/?hl=es) 等 URL 来获得更好的控制。
   - 其他人注意到部分 PDF 页面无法生成见解，指出 **NotebookLM** 中的页面引用存在不一致。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hunyuan-Video 在 12GB VRAM 上取得成功**：**hunyuan-video 模型**在低至 **12GB VRAM** 的环境下也能有效运行，提供本地图生视频（image-to-video）处理能力，吸引了众多开发者。
   - 社区成员赞扬了其在日常实验中的易用性，并参考 [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-forge-webui-with-zluda) 进行高级调整。
- **Kling AI 缺乏图生视频功能**：用户注意到 **Kling AI** 在图像质量上与 **hunyuan** 接近，但目前尚不支持视频转换。
   - 他们认为对于完整的流水线（pipeline）来说，缺失该功能令人遗憾，一些人希望后续更新能尽快解决这一差距。
- **面向新图像创作者的 Forge 与 Swarm**：对于寻求更简单本地 **AI image generation** 工具的新手来说，**Forge** 和 **Swarm** 成为了热门选择。
   - 资深用户推荐使用 **ComfyUI** 以获得更大的灵活性，但他们提醒初学者注意其额外的复杂性。
- **Stable Diffusion 倾向于 32GB RAM 或更高配置**：运行 **Stable Diffusion** 最好配备 **32GB RAM** 的系统，而 **64GB** 能确保更流畅的体验。
   - 使用 **RTX 4090** 或 **AMD 7900XTX** 的成员报告称，在升级内存后，硬件冲突明显减少。
- **Deepseek 需要高达 1.3TB 的 VRAM**：**Deepseek** 系列（包括 **V3** 和 **R1**）在全精度下需要超过 **1.3TB 的 VRAM**，这远超消费级设备的承载能力。
   - 只有拥有 **A100** 或 **H100** 等多 GPU 集群的用户才能处理这些模型，迫使其他用户只能寻找更小的替代方案。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeepSeek 通过 TinyZero 和 Open R1 实现双重突破**：[TinyZero](https://github.com/Jiayi-Pan/TinyZero) 项目以简洁、易获取的方式复现了 **DeepSeek R1 Zero**，并为贡献者提供了图像和详细信息。同时，Hugging Face 的 [Open R1](https://github.com/huggingface/open-r1) 提供了 **DeepSeek-R1** 的完全开源版本，鼓励协作开发。
   - 这两个仓库都邀请社区参与，展示了在 HPC 背景下对可复现研究的强力推动。
- **解决 HPC 中的 NCCL 超时问题**：多位成员报告了在多节点训练期间遇到 **NCCL timeouts**，并询问调试的最佳实践。他们对 GPU 任务进行了性能分析（profiling），并考虑使用高级策略来处理大规模设置中的超时问题。
   - 社区记录了包括 CUDA 版本不匹配在内的常见陷阱，强调了对强大 HPC 调试工具的需求。
- **Adam Paszke 的 Mosaic GPU DSL 魔法**：著名的 **Adam Paszke** 在 YouTube 直播中讨论了他的 **Mosaic GPU** DSL，重点关注底层 GPU 编程。社区成员可以在 [GitHub](https://github.com/gpu-mode) 上找到补充材料，并加入 [Discord](https://discord.gg/gpumode) 进行主动学习。
   - 该讲座承诺将深入探讨用于 GPU 优化的布局系统（layout systems）和分块（tiling）技术。
- **JAX 在旧款 GPU 上运行 FP8**：一项 [GitHub discussion](https://github.com/jax-ml/jax/discussions/26077) 透露，**JAX** 可以在 sm<89 的 Nvidia GPU 上使用 **fp8**，突破了典型的硬件限制。PyTorch 用户报告在旧款 GPU 上运行失败，引发了对 JAX 变通方案的好奇。
   - 这一差距激发了人们对 JAX 究竟如何绕过标准限制的兴趣，促使对库内部机制的进一步探索。
- **Arc-AGI 扩展迷宫任务与 FSDP**：**Arc-AGI** 环境在 reasoning-gym 中增加了多项式方程、迷宫任务和更多示例，并参考了来自 CLRS 的算法。同时，**Tiny-GRPO** 引入了 **FSDP support**，大幅降低了 VRAM 占用并提升了效率。
   - 成员们还提出了关于家庭关系数据和 **GSM8K** 模板的想法，计划推送到 HF hub 以方便用户下载。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 文档消失后重新上线**：由于 Cloudflare 托管问题，**Mojo 文档**突然离线，引发了用户的不满。
   - 开发团队表示歉意，并确认文档已**恢复上线**，并更新了来自 [Mojo GitHub changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md) 的参考内容。
- **新 GPU Package API 登陆 Nightly 版本**：用户确认 **GPU package API** 文档已在 nightly 版本中上线，为 **Mojo** 提供了高级 GPU 功能。
   - 他们对这一更新表示欢迎，认为这是一项重大改进，并指出了 [changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md) 中的最新更新。
- **CSS Struct Fluent API 引发警告**：一位开发者构建了一个 `struct`，使用**链式调用 API (fluent API)** 风格生成 CSS，但在 Zed Preview 中遇到了未使用值的警告。
   - 他们尝试使用 `_ = ` 来消除警告，但希望能有一种更优雅的解决方案来保持代码清晰。
- **List 与 Representable Trait 的纠缠**：一位用户在将 `List[Int]` 传入函数时遇到了困难，发现编译器无法将 **Int** 识别为 **Representable**。
   - 他们指出了 [int.mojo](https://github.com/modular/mojo/blob/nightly/stdlib/src/builtin/int.mojo#L1146) 和 [List module](https://github.com/modular/mojo/blob/nightly/stdlib/src/collections/list.mojo#L441) 中可能存在的条件一致性 (conditional conformance) 问题。
- **Unsafe Pointers 与函数指针 FFI 障碍**：在使用 **UnsafePointer** 时，发现值结构体 (value structs) 中的对象标识 (object identity) 会发生偏移，导致指针独立移动时产生困惑。
   - 他们还指出，**Mojo** 中的函数指针 FFI 仍然不够可靠，仅部分兼容 C ABI，且文档有限。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Presenter 亮相与多 Agent 魔法**：LlamaIndex 推出了 [Presenter](https://twitter.com/llama_index/status/1883307955782901926)，这是一个**多 Agent 工作流**，可以在一个流水线中创建包含 **Mermaid 图表**、脚本生成和**报告生成**的视觉丰富型幻灯片。
   - 社区成员赞扬了 *Presenter 易于访问的结构*，展示了这些参考案例如何演变成协调复杂步骤的高级**演示文稿构建 Agent**。
- **文档钻取与 Google 风格的收益**：MarcusSchiesser 发布了[一个完全开源的模板](https://twitter.com/llama_index/status/1883675662839636427)，用于**多步文档研究 Agent**，其灵感来自 Google 的深度研究 (deep research) 方法。
   - 用户提到该模板处理**复杂研究工作流**的能力，并指出它解决了高级项目中对集成**分析与引用**的普遍需求。
- **Scaleport 的快速理赔处理**：Scaleport AI 与一家旅游保险公司达成[合作伙伴关系](https://twitter.com/llama_index/status/1883929949205336509)，利用 **LlamaIndex** 自动从医疗报告中进行**理赔评估**，并使用 **OCR** 进行数据提取。
   - 社区成员强调了其*显著的时间节省*，并指出这些方法展示了如何通过 **AI 驱动的风险分析**来实现更高效的保险流程。
- **DeepSeek 与 LlamaIndex 的巧妙集成**：LlamaIndex 现在已集成 [DeepSeek-R1 API](https://twitter.com/llama_index/status/1883986763380842864)，支持在统一环境中使用 **deepseek-chat** 和 **deepseek-reasoner** 进行高级调用。
   - 开发者肯定了这种协同效应的提升，并参考 [DeepSeek 文档](https://api-docs.deepseek.com/) 启用了 **API-key** 导入和无缝模型使用。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **日本监管自由：Cohere 保持不受影响**：Cohere 发现，针对高级计算的**日本 AI 新规**在 2025 年 5 月之前不太可能对他们产生影响，因为他们的语言模型不在关键限制范围之内。
   - 这些规则专门针对强大的芯片和扩展，目前 **Cohere** 未受波及，而他们的法律团队正在密切关注相关修订情况。
- **仪表盘困境：Cohere 即将进行 UI 重构**：社区反馈指出 [Cohere dashboard](https://dashboard.cohere.com/) 上存在**令人困惑的界面元素**，特别是镜像按钮布局。
   - 建议的改进包括为 Discord 和邮件支持提供更大的行动号召（CTA）按钮，用户也敦促采用更**精简**的设计方案。
- **纯粹文本：Cohere 不提供 TTS 或 STT**：**Cohere** 正式确认其专注于大语言模型，不提供内置的文本转语音（TTS）或语音转文本（STT）功能。
   - 这一明确表态结束了关于音频功能的猜测，重申了 **LLM** 支持是该平台的核心优势。
- **ChatCohere 代码纪事：分步 LLM 设置**：开发者展示了如何定义 **ChatCohere**，使用 `bind_tools` 绑定工具，然后通过结构化消息调用 LLM 以执行高级任务。
   - 一些人提到将**逆向规划**作为最后一步检查，并强调 Cohere 坚持基于文本的解决方案，而非 TTS 或 STT 集成。
- **工具层级：Cohere 的多步方法**：Cohere 的文档详细介绍了从用户提示词检索到最终文本生成的阶段性**多步（multi-step）**流程。
   - 社区成员称赞了这种**系统化**的分解，强调了顺序推理如何细化复杂的输出，并确保在每个阶段提取相关数据。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **开源图像分析获得关注**：人们正在寻找处理图像提示词的开源模型，并转向 [Taggui](https://taggui.com) 等框架进行打标签。他们很难找到一个在打标签和响应生成方面都表现出色的确定选项。
   - 一些人主张采用不需要高级配置的更简单设置。另一些人指出市场缺乏明确的领先者，这促使了更多的实验。
- **DeepSeek R1 模型初期遇挫**：多位用户报告了 [DeepSeek R1](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) 存在推理不完整和聊天模板错误的问题。他们提到在没有补丁的情况下很难在本地稳定运行。
   - 基准测试暗示其结果与 LLAMA 相当，但尚未有人确认其输出完全可靠。一些人呼吁在实际场景中信任该模型之前进行更多测试。
- **本地文档分析引起好奇**：爱好者们希望通过探索 **PDFGear** 等工具进行本地文本索引来保持数据私密性。他们的目标是在不依赖云服务或上传的情况下查询个人文档。
   - 关于如何处理复杂的 PDF 和大量文本，意见不一。人们要求提供详细的示例和更简单的流水线来简化这些流程。
- **GPT4All 等待 DeepSeek R1 支持**：社区成员询问 **DeepSeek R1** 何时能以官方、易于安装的方式进入 GPT4All。贡献者表示集成仍在进行中，但未给出确切的发布时间表。
   - 他们希望实现一键设置，不需要大量的后期手动调整。有人暗示修复方案已接近完成，但尚未发布官方声明。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 周一狂欢**：**Advanced LLM Agents MOOC** 将于 **1 月 27 日** **4:00PM PST** 开始，持续至 **4 月 28 日**，通过 [课程网站](http://llmagents-learning.org/sp25) 提供每周 [直播](https://www.youtube.com/live/g0Dwtf3BH-0) 和资源。
   - 参与者可以 [在此注册](https://forms.gle/9u6HdVCWXgws16go9) 并在 **YouTube** 上观看回放，对于非伯克利学生，没有紧迫的截止日期，也不提供线下参加机会。
- **证书与困惑交织**：成员反映 **Fall'24 MOOC 证书** 仍未发放，工作人员宣布即将发布消息并请大家耐心等待。
   - 一些人还提到在报名后未收到 **确认邮件**，纷纷表示 *“在同一条船上...”*，而工作人员承诺很快会有官方更新。
- **黑客松与无线下聚会**：爱好者询问了 **黑客松机会**，工作人员指出虽然兴趣浓厚，但本学期尚无最终计划。
   - 其他人寻求线下访问权限，但获悉只有伯克利正式学生才允许进入现场，因此其他所有人只能依赖虚拟平台。
- **Substack 谜团浮现**：在 **#mooc-readings-discussion** 中出现了一个令人好奇的 [Substack 链接](https://substack.com/home/post/p-154577981)，但提供的上下文很少。
   - 社区对此保持观望，等待后续对该共享资源的进一步说明。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **梯度引导取得进展**：关于 `Tensor.gradient` 用法的困惑爆发，文档称 *'计算 targets 相对于 self 的梯度'*，但上下文暗示其含义是 *'计算 self 相对于 targets 的梯度'*，引发了讨论。
   - 参与者提议修改文档以确保准确性，并指出 **tensor.py** 可能需要为未来的参考提供进一步的澄清。
- **STRIDE 与 FLIP 之争**：建议将 **STRIDE** 更名为 **FLIP** 以避免名称过于通用，旨在提高代码库的清晰度。
   - 贡献者支持这一转变，理由是残留的旧引用会使更新复杂化并减慢功能集成。
- **周一疯狂：第 55 次会议**：计划于圣地亚哥时间周一早上 6 点举行，**Meeting #55** 计划讨论最近的多梯度设计、公司更新以及 **resnet** 和 **bert** 等项目。
   - 参与者期望解决新的项目悬赏（bounties），意在细化即将到来的任务和截止日期。
- **BobNet 命名困惑**：在 [GitHub 引用](https://github.com/qurAI-amsterdam/bobnet) 暗示使用了 bounding box 后，围绕 **BobNet** 产生了疑问，然而代码只是普通的 feed-forward。
   - 成员强调了命名清晰度的重要性，指出标题与功能之间的不匹配可能会误导新用户。
- **Tinygrad 中的格式化之争**：用户讨论了官方格式化工具，一些人引用了 **Black**，而另一些人则指向了 [pre-commit config](https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7) 中的 **Ruff**。
   - 达成的共识是 **Ruff** 有效地规范了格式，敦促贡献者遵循推荐的方法。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 中的联邦学习热潮**：贡献者提议为 **Torchtune** 联邦学习创建 **每个节点 N 个分片**，在每个分块训练完成后将权重与保存的 **optimizer state** 合并。
   - 一些人询问如何 *“在没有过度中断的情况下”* 简化训练，而另一些人则讨论了与 **torch distributed** 和 **raylib** 方法的潜在协同作用。
- **部分参数大乱斗**：社区成员辩论了应用 **opt-in backward hooks** 的 **性能提升**，以便某些参数在梯度准备就绪后立即更新。
   - 他们还权衡了一种仅使用独立更新器优化 **output projection** 的策略，并对并行运行多个优化器的复杂性表示担忧。
- **EBNF 胜过 Regex**：在一名成员声称 **regex** “看起来像格式错误的 tokenizer”后，社区转向使用 **EBNF grammars** 以获得更好的可读性。
   - 一些人发现 **EBNF** 虽然更冗长但更容易理解，直接引用称赞其 *“在保持鲁棒性的同时具有人类可读性。”*
- **Deepseek 惊艳的 Janus 系列**：一位用户批评 **Deepseek** 更新过于频繁，并引用了这份关于 **Janus-Series: Unified Multimodal Understanding and Generation Models** 的 [报告](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)。
   - 其他人则在调侃这些多模态功能的潜在影响范围，其中一人在与过时模型的持续对比中打趣道 *“他们需要冷静一下”*。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 保留 Python 解释器**：**OpenInterpreter** 项目在**昨天**提交了一个 commit，计划发布集成 Python 解释器的 **1.0** 版本，并根据其 [GitHub 仓库](https://github.com/OpenInterpreter/open-interpreter) 的确认，在重大发布前将网站保持在极简状态。
   - 他们承诺在正式发布后进行更大规模的更新，社区反馈主要集中在用户交互的扩展上。
- **DeepSeek R1 触发 400 错误**：一位用户报告了 **Deepseek_r1** 模型的 **400 错误**，该模型通过指向 [https://api.deepseek.com](https://api.deepseek.com) 的 `api_base` 进行配置，由于模型不存在而产生 **BadRequestError**。
   - 对话指出在 **OpenAIException** 下出现了 **invalid_request_error**，这给尝试在工作流中运行 `$ interpreter -y --profile deepseek.yaml` 的用户带来了困惑。
- **DeepSeek 在测试中媲美 OpenAI-o1**：社区成员注意到 **DeepSeek-R1** 和较小的 **Distill-Qwen-1.5B** 模型在数学和代码任务上的表现与 **OpenAI-o1** 持平，参考了 [deepseek-r1 库信息](https://www.ollama.com/library/deepseek-r1)。
   - 他们还强调了 **DeepSeek** 在 OS 模式下的 tool-calling 要求以及集成 vision model 可能存在的问题，旨在优化高级场景下的使用。
- **结合 Ollama 和 Llamafile 的本地使用**：通过使用 [Ollama](https://www.ollama.com/) 和 [Llamafile](https://github.com/Mozilla-Ocho/llamafile) 展示了完全在本地资源上运行 **Open Interpreter** 的努力，响应了官方 [本地运行指南](https://docs.openinterpreter.com/guides/running-locally) 中的 `interpreter --local` 命令。
   - 讨论集中在多模型设置中是否有必要启用 vision model，并呼吁明确在组合框架中的使用方法。
- **DSH - AI Terminal 邀请贡献者**：一个名为 **DSH - Ai terminal** 的项目正在寻求对其开源应用的改进，参考了 [其 GitHub 仓库](https://github.com/gokul6350/dsh-shell)。
   - 开发者被鼓励为该项目点亮 star 并分享用户反馈，以增强其未来的功能。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DeepSeek R1 受到审视**：初步的 benchmark 表明 **DeepSeek R1** 与 **o1-mini** 和 **Claude 3.5 Sonnet** 相当，这与它在挑战性 LLM 基准测试中媲美 **o1** 的说法相矛盾，详见 [此处](https://x.com/JJitsev/status/1883158738661691878)。
   - 参与者质疑其在奥数级 **AIW 问题**上的有效性，并引用了 [这篇论文](https://arxiv.org/abs/2406.02061) 来衡量其真实能力。
- **流水线获得音频升级**：有人建议使用 **audio widgets** 来比较增强效果，并集成了来自 **DeepSeq** 或 **O1** 等库的失真处理。
   - 贡献者强调了交互式功能对于检查和优化音频变化的便利性，目标是改进流水线。
- **测试流水线发布**：一位用户分享了一个 [开发阶段的流水线](https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt)，展示了在繁忙的旅途后的初步进展。
   - 他们邀请大家对流水线的功能特性提供反馈，重点是如何更好地探索音频增强能力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GitHub 垃圾内容增加**：一位成员指出随机垃圾信息淹没了 GitHub 仓库，掩盖了关于项目 bug 和功能的有价值报告。
   - 这引发了挫败感，其他人讨论了可能的过滤器和更警觉的分流（triaging）机制来遏制这些杂乱内容。
- **语言界限：自然语言 vs 代码**：一位用户询问是否有可靠的方法来检测文本是**自然语言**还是结构化代码（如 HTML 或 Python）。
   - 他们提出了使用专门的分类器来清晰分类文本格式的想法。
- **DSPy + DeepSeek 的困境**：一位参与者尝试为 70B COT 示例优化 **dspy + deepseek**，但无法明确简化流程的具体步骤。
   - 其他人就运行时间和内存限制提出了疑问，强调了大规模优化的复杂性。
- **BSR 六小时停滞**：一位用户运行 **BSR** 示例六小时未见收敛，引发了对该方法实用性的质疑。
   - 这引发了关于替代策略的辩论，或者这种超长运行时间是否值得其产出。
- **针对 FastAPI 的 PyPI 压力**：一位开发者需要 PyPI 上更新的 **RC** 版本以匹配现代 **FastAPI** 依赖项，因为过时的包导致安装失败。
   - 他们指出主分支在三周前已有一个修复补丁，敦促维护者发布新版本。

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Deepseek 亮相与 GRPO 猜想**：成员们询问了 **Deepseek 算法** 以及是否有人在进行复现，并提到了 **trl** 中与 **grpo** 可能存在的联系。
   - 一位参与者建议它*可能指的是 grpo*，这表明人们对高级 RL 方法重新产生了兴趣，尽管目前尚未得到官方确认。
- **H200 vs 5090 GPU 的博弈**：一位用户在权衡购买 **2x 5090s** 还是 **1x H200**，并指出 H200 拥有更多 RAM，但性能收益尚不确定。
   - 他们提到了成本和速度方面的担忧，希望能获得关于哪种配置能更好支持**重型 AI 工作负载**的真实反馈。
- **RL 框架支持停滞不前**：一位成员指出 **trl** 缺乏在线 RL 训练器（online RL trainers），表达了对更广泛 RL 库集成的渴望。
   - 然而，另一条回复坚持认为这*极有可能不会*发生，加剧了对扩展 RL 支持的疑虑。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 提示词窥探**：用户询问了 **Berkeley Function Call Leaderboard** 上不支持函数调用的模型的系统提示词，随后有人引用了 [Gorilla GitHub 代码 (第 3-18 行)](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18)。
   - 该仓库专注于针对**函数调用（function calls）**训练和评估 LLM，并为非函数版本提供了所需的系统消息。
- **Gorilla 排行榜资源现身**：[Gorilla 函数调用排行榜的代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) 被分享作为相关系统提示词的所在地。
   - 它包含了**受函数启发（function-inspired）**的提示词定义，可以为寻求非函数用法参考的用户提供指导。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **2025 水晶球与实时数据盛会**：在 **1 月 28 日**，一场名为 **2025 Crystal Ball: Real-Time Data & AI** 的活动将邀请 **Rayees Pasha (RisingWave Labs)**、**Sijie Guo (StreamNative)** 和 **Chang She (LanceDB)** 参加，重点讨论实时数据如何提升 AI，详见 [此 Meetup 链接](https://www.meetup.com/streaming-stories/events/305736950/)。
   - 他们强调，如果没有低延迟的数据流水线（data pipelines），AI 的潜力就无法得到充分利用，并指出 **Apache Iceberg** 是助力各行业新兴分析的关键方法。
- **行业领袖预测 2025 创新方向**：专家小组成员预测，到 2025 年，**实时数据流（real-time data streaming）**将塑造 AI 的新工作流，在运营效率和快速决策方面赋予显著优势。
   - 他们计划解决从消费者应用到企业用例中不断演变的数据基础设施障碍，强调流技术与 AI 日益增长的需求之间的协同作用。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **论文阅读俱乐部再次聚首**：**论文阅读俱乐部（Paper Reading Club）**本周回归并安排了会议，详见 [Discord 活动链接](https://discord.com/events/1089876418936180786/1329844319703662664)。与会者可以期待对 AI 研究的深入探讨，重点关注能引起工程受众共鸣的高级论文。
   - 组织者鼓励参与者加入并分享想法，在社区环境中进行活跃的**前沿讨论**。
- **Discord 活动激发社区参与**：除了论文阅读俱乐部，本周还重点推出了各种 **Discord 活动**，以保持成员对新活动的参与度。用户受邀加入正在进行的对话，获得交流技术见解的机会。
   - 负责人提醒大家查看公告频道以获取实时更新，强调了在这些协作聚会中**积极参与**的重要性。

---

**HuggingFace Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1332440841297461248)** (1123 messages🔥🔥🔥): 

> `模型微调, 动态量化, 硬件需求, Agentic AI, 训练数据集`

- **探索模型微调技术**：讨论围绕 Qwen2.5-VL 等模型的微调展开，以及使用特定配置进行训练的影响，包括使用 LoRA 适配器及其对 embeddings 的影响。
   - 参与者分享了关于使用现有数据集的见解，以及利用各种 notebook 资源进行有效模型训练的经验。
- **模型的动态量化**：DeepSeek-R1 和 Qwen2.5 等模型的动态量化是一个焦点，强调了其在减小模型大小的同时不损失输出连贯性的效率。
   - 用户对 1-bit quant 版本的潜在可用性表示关注，同时质疑在考虑到模型大小和性能下降的情况下，此类模型的必要性。
- **AI 模型的硬件考量**：参与者辩论了不同 GPU 配置对于有效运行 AI 模型的益处，讨论了 3060s 与 4090s 等选项及其运行大型模型的适用性。
   - 共识表明，虽然更多的 GPUs 可以提供更大的 VRAM，但功耗和性能需要仔细权衡。
- **机器学习基础的重要性**：多位用户强调了机器学习基础知识的必要性，提到理解 AI 模型背后的背景和底层原理的重要性。
   - 推荐了相关的在线课程和资源，以确保未来的开发者具备从事 AI 工作所需的核心技能。
- **现有 AI 工具和库的利用**：讨论涵盖了在运行 inference 时使用 VLLM 和 FastAPI 等流行库的情况，重点关注它们如何管理性能指标并简化模型部署。
   - 参与者强调，有效利用这些工具可以提高生产力，同时也警告不要在没有人工审查的情况下盲目信任生成的代码。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html">CPU — vLLM</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb">Google Colab</a>：未找到描述</li><li><a href="http://jasonwryan.com/blog/2012/03/17/vampires/">求助吸血鬼分类学 - jasonwryan.com</a>：未找到描述</li><li><a href="https://x.com/RussellBal/status/1883283659396104263">来自 russell@unturf. (@RussellBal) 的推文</a>：每一台旧的 2u 双路 Xeon 都要运行一个 Agent，而且大脑将是本地的。是的，它比 GPU 慢，但有了推理模型，我们只需让它们慢慢跑（let them cook），对吧？Xeon 运行蒸馏到 8B 的 R1...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S">unsloth/DeepSeek-R1-GGUF (main 分支)</a>：未找到描述</li><li><a href="https://x.com/Alibaba_Qwen/status/1883557964759654608">来自 Qwen (@Alibaba_Qwen) 的推文</a>：我们正在通过最新的开源模型 Qwen2.5-1M 提升竞争水平！💥 现在支持 100 万 TOKEN 上下文长度 🔥 以下是更新内容：1️⃣ 开源模型：迎接 Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - Unsloth 合集</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1883899061893546254">来自 Unsloth AI (@UnslothAI) 的推文</a>：隆重推出 1.58bit DeepSeek-R1 GGUF！🐋 DeepSeek-R1 现在可以在 1.58-bit 下运行，同时保持功能完整。我们将 671B 参数模型从 720GB 缩小到了仅 131GB —— 尺寸减少了 80%。原生量化...</li><li><a href="https://x.com/tom_doerr/status/1883517455445733580">来自 Tom Dörr (@tom_doerr) 的推文</a>：Unsloth：更快的 LLM 微调库</li><li><a href="https://gist.github.com/darkacorn/01b0db678d4d91b371e4eba274b911a6">gist:01b0db678d4d91b371e4eba274b911a6</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://huggingface.co/unsloth/Hermes-3-Llama-3.1-8B/tree/main">unsloth/Hermes-3-Llama-3.1-8B (main 分支)</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M">Qwen/Qwen2.5-14B-Instruct-1M · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen/Qwen2.5-VL-7B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://weechat.org/">WeeChat，可扩展的聊天客户端</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ib7mg4/i_spent_the_last_weekend_optimizing_the_deepseek/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course：一个关于对齐 smol 模型的课程。</a>：一个关于对齐 smol 模型的课程。通过在 GitHub 上创建账号为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">数据集入门 101 | Unsloth 文档</a>：学习创建用于微调的数据集的所有要点！</li><li><a href="https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md">trl/docs/source/grpo_trainer.md (main 分支) · huggingface/trl</a>：使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT">FreedomIntelligence/medical-o1-reasoning-SFT · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/prithivMLmods/Llama-Song-Stream-3B-Instruct-GGUF">prithivMLmods/Llama-Song-Stream-3B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://russell.ballestrini.net/">
    Russell Ballestrini
</a>

</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-">Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/1078">[Bug(CMake 3.17)] CUDA::cublasLt not found but can be specified absolutely · Issue #1078 · ggerganov/llama.cpp</a>: 前提条件 请在提交 issue 前自行回答以下问题。我正在运行最新的代码。目前开发非常迅速，因此还没有标记版本。我...</li><li><a href="https://github.com/ggerganov/llama.cpp.git">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai">Welcome | Unsloth Documentation</a>: 初识 Unsloth？</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: 又称持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: 在 Ollama 上本地运行自定义个人助手（类似 ChatGPT）的初学者指南</li><li><a href="https://www.coursera.org/specializations/machine-learning-introduction">Machine Learning</a>: 由斯坦福大学和 DeepLearning.AI 提供。通过机器学习专项课程 #BreakIntoAI。掌握基础 AI 概念并... 免费注册。</li><li><a href="https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics">sebastiandizon/genius-song-lyrics · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1332482886506385429)** (20 条消息🔥): 

> `NLP 课程完成, SmoLlm 微调, Ollama 温度设置, AI 生成文本检测, DeepSeek R1 对比 OpenAI O1` 


- **以优异成绩完成 NLP 课程**：一位成员宣布，“我终于以最高分完成了 NLP 课程，并开发了 AI 生成文本检测软件”，现在正兴奋地准备深入研究 **LLMs**。
   - 他们还完成了一门针对 **RAG 系统** 的**信息检索课程**，认为非常有帮助。
- **探索 SmoLlm 微调**：一位成员询问是否有人微调过 **SmoLlm 模型**，以及是否值得这样做。
   - 另一位成员分享了他们使用 **unsloth** 成功微调并在没有温度设置的情况下通过 **ollama** 运行的经验。
- **揭秘 Ollama 的默认温度**：针对关于 **ollama** 默认温度的问题，一位成员引用道：“根据 Ollama 的专家说法，它是 0.8（写在他们的文档中）”。
   - 这一澄清得到了认可，因为它为询问者节省了**时间**。
- **关于 CUDA 显存的幽默调侃**：一位成员幽默地评论说，他们的身体就像一台机器，能将水和薯片转化为 **RuntimeError: CUDA out of memory**。
   - 这引发了笑声，另一位成员开玩笑地建议“如何安装更多 VRAM”，并戏称可以去 **downloadram.com**。
- **DeepSeek R1 在 AI 模型中的崛起**：分享了一个标题为“DeepSeek R1 trimmed to 1.58bit 131 GB with unclothe #ai”的 YouTube 视频链接，展示了其开源能力。
   - 视频强调 **DeepSeek-R1** 正在与 **OpenAI 的 O1 推理模型** 展开竞争，引起了成员们的浓厚兴趣。



**提到的链接**：<a href="https://www.youtube.com/shorts/IzNQuD-FvIk">DeepSeek R1 trimmed to 1.58bit 131 GB with unclothe #ai</a>：DeepSeek-R1 最近因其在完全开源的情况下能与 OpenAI 的 O1 推理模型相媲美而引起轰动。我们探索了如何让更多本地用户能够使用它...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1332441133275418654)** (371 条消息🔥🔥): 

> `Unsloth 错误, 微调数据集格式化, 文本补全与聊天机器人数据集, DeepSeek R1 部署, 不同硬件上的模型部署` 


- **微调过程中的 Unsloth 错误**：用户报告了在运行训练 Notebook 时遇到的问题，特别是运行完所有单元格后出现的错误，例如 'RuntimeError: Unsloth: You must call FastLanguageModel.for_inference(model) before doing inference for Unsloth models.'。
   - 其他人也确认了类似问题，促使 Notebook 进行了修复，以解决这些错误并确保训练过程顺畅。
- **微调数据集格式化**：多位用户讨论了如何为各种模型训练格式化数据集，包括替换 Notebook 中的 'instruction'、'input' 和 'output' 字段。
   - 用户寻求了具体的示例和指导，关于如何调整数据集（如 Wikimedia 数据集）以符合成功微调所需的预期格式。
- **为 Unsloth 使用 Jupyter Notebook**：由于不喜欢使用 Google 服务，一位用户询问了是否可以使用 Jupyter Notebook 代替 Colab 来进行 Unsloth 微调项目。
   - 已确认确实可以使用 Jupyter Notebook 来运行 Unsloth 模型。
- **DeepSeek R1 部署与性能**：关于 DeepSeek R1 模型的 1-bit 量化版本的部署及其在 MI300X 等硬件上的性能，用户提出了疑问。
   - 用户讨论了对该模型性能的预期、使用经验以及在各种配置下的部署挑战。
- **从论坛创建聊天机器人数据集**：关于从论坛帖子创建聊天机器人数据集的讨论强调了保持源自帖子的问答格式。
   - 用户分享了将 Reddit 等平台的对话格式转换为可用于微调模型的可用数据集的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb#scrollTo=hvJcwnb9Qy8b">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1a8wP89019HKso87oE_b_P2wFUJzTPPYm?authuser=2#scrollTo=QmUBVEnvCDJv.">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Nemo_(12B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/deepseek-r1">运行 Deepseek-R1 / R1 Zero</a>: DeepSeek 最新的 R-1 模型是目前最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。了解如何运行和微调该模型。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main">unsloth/DeepSeek-R1-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/gus-gustavo/reddit_roastme?row=0">gus-gustavo/reddit_roastme · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Skorcht/finaldatasethopefully">Skorcht/finaldatasethopefully · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/wikimedia/wikipedia">wikimedia/wikipedia · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py#L383">unsloth/unsloth/models/_utils.py at main · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 70% 的显存微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 和 Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1332949351206752370)** (61 条消息🔥🔥): 

> `模型训练技术, AugmenToolKit 使用, 代码漏洞审查, Loss 曲线解读` 


- **以有趣风格训练 LLM**：一位成员分享了他们训练模型编写长篇散文的努力，利用 **AugmenToolKit** 进行数据集生成，旨在根据输入材料实现多样的写作风格。
   - 尽管发现 Loss 曲线过于线性（可能表明过拟合），他们仍表示乐观。
- **对 Loss 格式的担忧**：参与者讨论了训练过程中观察到的**线性 Loss**，一位参与者建议不要使用 instruct 模型以避免浪费计算资源。
   - 他们认为不当的数据集格式和结构可能会导致模型泛化方面的挑战。
- **针对代码审查的微调**：一位成员宣布针对代码审查微调了 **Qwen-2.5-Coder-7b** 模型，特别提到了代码中的漏洞。
   - 他们澄清说，其工作提供了 **16-bit 和 4-bit** 量化选项可供下载。
- **AI 开发中的同行支持**：几位成员表达了对彼此项目的志同道合和兴趣，分享见解并鼓励协作。
   - 一位成员强调了他们的经验，声称已经训练了超过 **100-200 个模型**，并为新手提供建议。
- **数据集格式讨论**：讨论强调了正确数据集格式的重要性，重点关注训练中摘要和片段是如何配对的。
   - 成员们分享了关于不均匀的数据长度如何影响训练结果的想法，并提出了改进建议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/doss1232/Offensive-Qwen-2.5-Coder-7B">doss1232/Offensive-Qwen-2.5-Coder-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO">CyberNative/Code_Vulnerability_Security_DPO · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1332646014120624221)** (49 条消息🔥): 

> `微调 LLM, LLaMA 4 预期, 强化学习增强, 向量数据库与量化, DeepSeek 与推理模型` 


- **用户探索微调评估**：一位用户建议创建一个自定义评分系统来评估微调后的 LLM，强调需要进行约 **100 个查询**的手动测试。
   - 他们指出，传统的指标如 **ROUGE**、**BLEU** 和 **F1** 测试可能不足以验证技术准确性。
- **对 LLaMA 4 寄予厚望**：成员们对 **LLaMA 4** 表达了很高的期望，预见其在 **test-time compute** 和源自 **DeepSeek** 学习过程的**速度提升**方面会有显著改进。
   - *一位成员推测，为了保持竞争力，可能需要一些令人印象深刻的功能*，并建议将资源投入到其他地方。
- **强化学习与 CoT 生成**：用户讨论了强化学习的有趣前景，其中一位考虑提交一个 **Colab notebook**，用于生成推理 CoT 数据集，以便使用 **Unsloth** 进行微调。
   - 对话强调了整合**计算机视觉**技术以增强上下文理解的潜力。
- **向量数据库与量化的奇特案例**：一位用户询问了用于存储各种量化位的向量数据库，以及针对这些数据的相似性搜索是如何运作的。
   - 另一位用户寻求澄清，对存储量化位的术语及其应用提出疑问。
- **DeepSeek 与 Sonnet 3.6 的性能对比**：在实际测试中，一位用户声称 **Sonnet 3.6** 在创新编码任务中优于所有推理模型，这表明有效性取决于任务类型。
   - 这引发了关于不同模型之间语言理解能力细微差别的讨论。



**提及的链接**: <a href="https://github.com/SalesforceAIResearch/perfcodegen">GitHub - SalesforceAIResearch/perfcodegen</a>: 通过创建账号为 SalesforceAIResearch/perfcodegen 的开发做出贡献。

  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1332440021411827857)** (762 条消息🔥🔥🔥): 

> `Cursor IDE 性能, DeepSeek 与 Claude 的对比, 代码库索引, RAG 实现, 模型的用户体验`

- **Cursor IDE 性能与用户体验**：用户对 Cursor 模型的请求响应慢表示担忧，特别是 Claude，强调了在项目开发时的挫败感。
   - 一些用户报告称在非高峰时段性能有所提升，并指出像 R1 和 DeepSeek 这样的模型是可用的替代方案，可以优化他们的工作流。
- **DeepSeek 与 Claude 的对比**：DeepSeek R1 在规划任务中更受青睐，而 Claude 由于其更高质量的响应，通常用于处理更复杂的输出。
   - 用户普遍认为这两个模型都优于其他替代方案，一些人讨论了如何利用 DeepSeek 处理较简单的任务，以节省高级请求的成本。
- **Cursor 中的 Codebase 索引**：Cursor 的索引功能在有效性方面存在争议，一些用户断言它没有充分利用代码库来进行推荐。
   - 尽管存在批评，但人们承认索引对于改进代码库问答至关重要，并且在配置得当时可以提升用户体验。
- **RAG 实现讨论**：用户分享了在代码库中实现检索增强生成（RAG）方法的经验和策略，重点关注向量存储和 embedding 技术。
   - 社区成员讨论了各种模型在 embedding 和检索方面的性能，断言正确的实现是减少错误和改善结果的关键。
- **切换到替代模型**：几位用户考虑在不同模型之间切换，特别是利用 DeepSeek 处理基础任务，而将 Claude 留给更复杂的需求。
   - 还提到了 GitHub Copilot 和 Spark Engine 等替代平台，用户满意度和 API 集成能力各不相同。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://sparkengine.ai">Spark Engine - 高级无代码 AI 构建器</a>: 无需编程即可构建和部署 AI 模型。开始使用最用户友好的 AI 平台进行创作。</li><li><a href="https://docs.cursor.com/get-started/usage#usage-based-pricing">入门 / 使用 – Cursor</a>: 未找到描述</li><li><a href="https://docs.cursor.com/get-started/usage">入门 / 使用 – Cursor</a>: 未找到描述</li><li><a href="https://docs.cursor.com/context/codebase-indexing">上下文 / 代码库索引 – Cursor</a>: 未找到描述</li><li><a href="https://download.todesktop.com/230313mzl4w4u92/Cursor%20Setup%200.45.3%20-%20Build%20250124b0rcj0qql-x64.exe">未找到标题</a>: 未找到描述</li><li><a href="https://aistudio.google.com/apikey">未找到标题</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/is-cursor-using-full-version-of-r1/44756/5">Cursor 是否在使用完整版的 R1？</a>: 团队确认使用的是拥有 671B 参数的模型，而非 Distill R1 版本。</li><li><a href="https://x.com/awnihannun/status/1883276535643455790">来自 Awni Hannun (@awnihannun) 的推文</a>: DeepSeek R1（完整的 680B 模型）在 3 台搭载 MLX 的 M2 Ultra 上以高质量 4-bit 模式运行良好。问了它一个编程问题，它思考了约 2k tokens，总共生成了 3500 tokens：</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>: R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://winstall.app/apps/Anysphere.Cursor">使用 winget 安装 Cursor - winstall</a>: AI 代码编辑器</li><li><a href="https://forum.cursor.com/t/cursor-does-not-send-files-to-claude/43948">Cursor 无法向 Claude 发送文件</a>: 一个 Cursor 连续三次无法向 Claude 发送文件数据的例子。这消耗了我宝贵的 Fast-Replies，而且这种情况经常发生。我正在使用的版本是：Version: ...</li><li><a href="https://forum.cursor.com/t/slow-pool-information/41812?u=danperks">慢速池信息</a>: 你好！想提供一些关于慢速池等待时间的细节，以及为什么它们最近有所增加…… Anthropic 容量：我们正在与 Anthropic 合作以扩大 Sonnet 的流量规模。目前我们正处于...</li><li><a href="https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.">voyage-code-3：通过低维量化嵌入实现更准确的代码检索</a>: TL;DR – 介绍 voyage-code-3，我们专为代码检索优化的下一代 Embedding 模型。在一系列测试中，它的表现平均优于 OpenAI-v3-large 13.80% 和 CodeSage-large 16.81% …</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API、供应商、统计数据</a>: DeepSeek R1 发布了：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并且具有完全公开的推理 tokens。它的参数量为 671B，在一次推理过程中有 37B 激活。运行...</li><li><a href="https://forum.cursor.com/">Cursor - 社区论坛</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://gofund.me/58f99126">捐助“求助呼吁：三口之家在火灾中失去一切”，由 Griffin 家族发起</a>: 帮助一个家庭在悲惨的火灾后重建……Griffin 家族需要您对“求助呼吁：三口之家在火灾中失去一切”的支持。</li><li><a href="https://docs.cursor.com/get-started/usage#fast-and-slow-requests">入门 / 使用 – Cursor</a>: 未找到描述</li><li><a href="https://github.com/cline/cline">GitHub - cline/cline: 直接在您的 IDE 中的自主编码 Agent，能够在每一步都获得您许可的情况下创建/编辑文件、执行命令、使用浏览器等。</a>: 直接在您的 IDE 中的自主编码 Agent，能够在每一步都获得您许可的情况下创建/编辑文件、执行命令、使用浏览器等。 - cline/cline</li><li><a href="https://github.com/danilofalcao/cursor-deepseek">GitHub - danilofalcao/cursor-deepseek: 一个高性能的、启用 HTTP/2 的代理服务器，专门设计用于让 Cursor IDE 的 Composer 使用 DeepSeek 和 OpenRouter 的语言模型。该代理将兼容 OpenAI 的 API 请求转换为 DeepSeek/OpenRouter 的 API 格式，使 Cursor 的 Composer 和其他兼容 OpenAI API 的工具能够无缝使用这些模型。</a>: 一个高性能的、启用 HTTP/2 的代理服务器，专门设计用于让 Cursor IDE 的 Composer 使用 DeepSeek 和 OpenRouter 的语言模型。该代理将...</li><li><a href="https://api-docs.deepseek.com">您的第一次 API 调用 | DeepSeek API 文档</a>: DeepSeek API 使用与 OpenAI 兼容的 API 格式。通过修改配置，您可以使用 OpenAI SDK 或兼容 OpenAI API 的软件。</li>

I 访问 DeepSeek API。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/11453">ggml : x2 speed for WASM by optimizing SIMD by ngxson · Pull Request #11453 · ggerganov/llama.cpp</a>: 动机：此 PR 通过为 qX_K_q8_K 和 qX_0_q8_0 点积函数利用 SIMD 指令，为 WASM 带来了巨大的速度提升。令人惊讶的是，此 PR 中 99% 的代码是由 De... 编写的</li><li><a href="https://fireworks.ai/blog/fireworks-quantization">How Fireworks evaluates quantization precisely and interpretably </a>: 深入探讨 Fireworks AI 如何思考量化，并使用散度指标确保质量并为用户创建自定义解决方案  
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1332439967309369345)** (1 messages): 

> `Windsurf 1.2.2 Release, Cascade's Memory Improvements, Web Search Capabilities` 


- **Windsurf 1.2.2 发布并带来关键增强**：团队宣布发布 **Windsurf 1.2.2**，其中包括对对话卡顿的修复以及对 **Cascade 内存系统**的改进。
   - 此次更新旨在通过多项关键增强功能，使整体 **Windsurf 体验更加流畅**且更可靠。
- **Cascade 将 Web 搜索提升到新高度**：Cascade 现在可以自动或通过 **URL 输入**搜索 Web，使用户能够进行实时查询或粘贴链接以获取上下文。
   - 用户还可以利用 `@web` 和 `@docs` 等命令进行搜索，增强了在流行文档网站上的可用性。



**提到的链接**：<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变化。

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1332440809986981920)** (252 条消息🔥🔥): 

> `Windsurf 性能问题、免费和 Pro 计划额度变更、DeepSeek 模型集成预期、Cascade 用户体验、扩展兼容性问题` 


- **Windsurf 性能问题困扰用户**：用户报告 Windsurf 存在持续的错误和卡顿问题，特别是 Cascade 无法正确执行命令。
   - 几位成员对在编码过程中尝试调试和修复错误时损失额度（credits）表示沮丧。
- **免费计划额度大幅削减 - 现仅剩 5 个 prompt**：Windsurf 免费版现在仅提供 5 个高级模型用户 prompt 额度，低于之前可通过创建新账号获取的 50 个额度。
   - 这一变化引起了依赖较高额度限制进行编码任务的用户的不满。
- **近期预计不会集成 DeepSeek**：用户急于了解 DeepSeek 何时会集成到 Codeium 中，并担心该平台会落后于竞争对手。
   - 一些成员表示，如果 DeepSeek 集成没有实现，他们打算转向其他工具。
- **使用 Cascade 的体验褒贬不一**：虽然一些用户发现 Cascade 很有帮助且高效，但另一些用户报告它开始忽略既定规则并错误地修改代码。
   - 参与者建议保持请求简洁并使用项目规则（project rules）来缓解这些问题。
- **Windsurf 的扩展兼容性问题**：用户遇到扩展与当前版本的 Windsurf 不兼容的问题，导致工作流中断。
   - 建议尝试安装旧版本的扩展，这些版本可能对 IDE 版本没有那么严格的要求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.apple.com/documentation/swift/">Swift | Apple 开发者文档</a>：使用强大的开源语言构建应用。</li><li><a href="https://docs.codeium.com/windsurf/getting-started#forgot-to-import-vs-code-configurations">Windsurf - 入门指南</a>：未找到描述</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/the-simpson-leech-leeches-gif-11029678">The Simpson Leech GIF - The Simpson Leech Leeches - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/windsurf_ai/status/1880354013922857384?s=46">来自 Windsurf (@windsurf_ai) 的推文</a>：Wave 2 已上线。本次更新包含：🌐网络搜索🧠自动生成的记忆💼企业级支持... 以及更多！</li><li><a href="https://codeium.com/pricing">定价 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 对个人用户永久免费。团队可以通过我们的企业版方案提升效率，获得增强的个性化和灵活部署支持。</li><li><a href="https://youtu.be/DKUe0ST_qi4?si=G5IOHXH6B9IG6qa8">通过 Cascade 自动生成的记忆节省时间和 Flow Action 额度 - Windsurf 编辑器</a>：了解 Cascade 的自动生成记忆如何帮助你快速衔接进度——无需额外的 prompt 或消耗额外的 flow action 额度...</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1332439820613455903)** (505 条消息🔥🔥🔥): 

> `Windsurf 更新与 Bug、Cascade 性能、DeepSeek 用户体验、支持与文档、开发中的 Git 使用` 


- **更新后 Windsurf 的性能问题**：用户反馈在最新的 Windsurf 更新后出现了明显的延迟和内部错误消息，使其与之前的版本相比可用性降低。
   - 许多人对更新影响其工作流表示沮丧，希望能回滚版本或获得紧急修复。
- **对 Cascade 集成的评价褒贬不一**：一些用户分享了使用 Cascade 的经验，强调在 Prompt 中提供详细上下文如何改善结果，而另一些用户则遇到了内部错误问题。
   - 用户注意到额度消耗不平衡的问题，即纠正 Cascade 犯下的错误会导致更高的额度消耗。
- **对 DeepSeek 集成的关注**：关于将 DeepSeek R1 集成到 Windsurf 的讨论持续不断，用户对其相比当前模型更低的运营成本非常感兴趣。
   - 同时也存在对 DeepSeek 在 tool calls 方面的表现以及在 Agent 工作流中整体可靠性的担忧。
- **Google 身份验证问题**：多名用户报告了通过 Google 身份验证登录 Windsurf 时遇到困难，特别是涉及 G Suite 账号时。
   - 建议的临时解决方案包括在问题解决前使用标准 Gmail 账号以快速访问。
- **使用 Git 进行版本控制的重要性**：许多用户强调了在使用 Windsurf 的同时配合 Git 进行版本控制的必要性，并指出这可以减轻 AI 错误带来的影响。
   - 分享了使用 Git 的最佳实践（包括标记里程碑），以帮助用户在开发过程中保持代码完整性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://learngitbranching.js.org/">Learn Git Branching</a>: 一个用于教育和挑战的交互式 Git 可视化工具！</li><li><a href="https://githowto.com/">GitHowTo: 关于 Git 的引导式教程</a>: 未找到描述</li><li><a href="https://graphite.dev/">Graphite - 端到端开发者平台</a>: Graphite 帮助 GitHub 上的团队更快地交付更高质量的软件。</li><li><a href="https://developer.apple.com/documentation/">精选 | Apple 开发者文档</a>: 浏览最新的示例代码、文章、教程和 API 参考。</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium 文档</a>: 未找到描述</li><li><a href="https://www.promptingguide.ai/">Prompt Engineering 指南</a>: Prompt Engineering 的全面概述</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://www.austinmann.com/trek/iphone-16-pro-camera-review-kenya">iPhone 16 Pro 相机评测：肯尼亚 — 旅行摄影师 - Austin Mann</a>: 来自肯尼亚的问候！上周在 Apple 发布会上，最令我印象深刻的 iPhone 相机功能是全新的相机控制按钮、升级的 4800 万像素超广角传感器...</li><li><a href="https://youtu.be/WYb2aMVnuYY">AI 代码编辑器的未来，访谈 Kevin Hou (Codeium, Windsurf)</a>: 本集由 Codeium 产品工程主管 Kevin Hou 主讲，涵盖了公司从 GPU 虚拟化到创建领先 AI 的历程...</li><li><a href="https://github.com/sweetpad-dev/sweetpad#rea">GitHub - sweetpad-dev/sweetpad: 使用 VSCode 开发 Swift/iOS 项目</a>: 使用 VSCode 开发 Swift/iOS 项目。通过在 GitHub 上创建账号为 sweetpad-dev/sweetpad 做出贡献。</li><li><a href="https://codeium.com/windsurf">Codeium 推出的 Windsurf 编辑器</a>: 未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，让开发者保持高效流转。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://youtu.be/DKUe0ST_qi4?si=G5IOHXH6B9IG6qa8">通过 Cascade 自动生成的记忆节省时间和 Flow Action 额度 - Windsurf 编辑器</a>: 了解 Cascade 的自动生成记忆如何帮助你从上次中断的地方继续——无需额外的 Prompt 或消耗额外的 Flow Action 额度...</li><li><a href="https://github.com/sweetpad-dev/sweetpad#readme">GitHub - sweetpad-dev/sweetpad: 使用 VSCode 开发 Swift/iOS 项目</a>: 使用 VSCode 开发 Swift/iOS 项目。通过在 GitHub 上创建账号为 sweetpad-dev/sweetpad 做出贡献。</li><li><a href="https://www.trae.ai/privacy-policy">TraeAI - 隐私政策</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1332440930271105187)** (755 条消息🔥🔥🔥): 

> `Perplexity Pro 变更，R1 模型引入，用户对 AI 模型的反馈，DeepSeek 的隐私担忧，与其他 AI 服务的对比` 


- **Perplexity Pro 因模型变更面临抵制**：用户对 Perplexity Pro 移除 O1 和 Grok 等热门模型表示沮丧，导致刚续订计划的订阅者感到不满。
   - 许多人正在考虑转向 DeepSeek 或 ChatGPT 等替代方案，因为 R1 等新模型的使用限制被认为无法满足需求。
- **R1 模型的引入引发质疑**：新推出的 R1 模型限制很大，与 O1 合计每天仅提供 10 次查询，引起了习惯于更高限制的用户批评。
   - 尽管 R1 价格更低且托管在美国，但其发布过程中遇到了性能和可靠性问题，导致用户将其与之前的模型相比时给出了负面评价。
- **用户反馈凸显质量问题**：许多用户报告 R1 的回答质量不足，特别是在编程和研究等专业任务中，促使一些人更倾向于使用其他平台。
   - 关于 Prompt 处理的投诉也随之出现，用户注意到与之前备受青睐的 O1 相比，R1 似乎会误解请求。
- **隐私和数据路由担忧**：用户担心他们的数据将如何被处理，特别是考虑到 DeepSeek 是一家中国公司，以及中美之间数据路由的影响。
   - 据报道，查询是在美国处理的，但数据处理实践缺乏透明度引起了一些用户的怀疑。
- **广告增加和用户界面挫败感**：一些用户批评 Perplexity 界面上的广告增加，认为该平台的广告投放开始效仿传统的搜索引擎。
   - 用户对用户体验和界面更改表示不满，许多用户对服务中增加的杂乱和噪音感到不知所措。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://labs.perplexity.ai)?">未找到标题</a>: 未找到描述</li><li><a href="https://app.chathub.gg/chat/cloud-doubao-1.5-pro">Doubao 1.5 Pro | ChatHub</a>: 在 ChatHub 上与 Doubao 1.5 Pro 及 20 多个 AI 模型聊天</li><li><a href="https://x.com/gmishra/status/1883951104607805615">来自 Gaurav Mishra (@gmishra) 的推文</a>: 为 @perplexity_ai Pro 付费的理由是可以轻松测试不同的模型！！正在测试 @deepseek_ai，与 @OpenAI 相比相当不错，我很惊讶 @perple...</li><li><a href="https://x.com/testingcatalog/status/1883775532086804953?s=61">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: DeepSeek R1 即将作为新的推理选项加入 Perplexity 👀* 尚未向公众开放 /t @denisyarats</li><li><a href="https://x.com/apostraphi/status/1883927593319293430?s=46">来自 Phi Hoang (@apostraphi) 的推文</a>: DeepSeek R1 现已在 Perplexity 上线。引用 Perplexity (@perplexity_ai)：DeepSeek R1 现已在 Perplexity 上线，以支持深度网络研究。新增了一个 Pro Search 推理模式选择...</li><li><a href="https://x.com/dee_bosa/status/1883921252102099439?s=46">来自 Deirdre Bosa (@dee_bosa) 的推文</a>: Perplexity CEO Aravind Srinivas 谈 DeepSeek 最新动态 https://x.com/i/spaces/1mnxeAgoDbvxX</li><li><a href="https://tenor.com/view/spider-man-we-one-gif-18212100">蜘蛛侠 GIF - Spider Man We - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://intercom.help/perplexity-ai/en/articles/10354288-refunds">退款 | Perplexity 帮助中心</a>: 了解更多关于 Perplexity Pro 退款的信息。</li><li><a href="https://www.reddit.com/r/singularity/comments/1hxykyr/deepseek_v3_is_hugely_chinese_biased/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/user/maximim12/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://developer.visa.com/capabilities/vau">
      Visa 账户更新程序概览
    </a>: 未找到描述</li><li><a href="https://developer.mastercard.com/product/automatic-billing-updater-abu/">Mastercard 开发者</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1332522184416952401)** (26 messages🔥): 

> `AI 进展, 金融趋势, 地震力学, 动作冒险电影, 初创公司见解` 


- **5000 亿美元交易将改变 AI 格局**：一名成员强调了一项潜在的 **5000 亿美元交易**，这可能会显著改变 AI 的版图，相关见解链接见[此处](https://www.perplexity.ai/page/stargate-project-InQ5ZvKETX6c5I6he1zc_A)。
   - 这一机遇被视为该领域创新的关键时刻。
- **如何打造价值 2 亿美元的初创公司**：分享了关于如何白手起家创办初创公司并在退出时达到 **2 亿美元** 估值的技巧，强调了[此处](https://www.perplexity.ai/page/wingify-T9bxT5tHSY2sRduhPzHIXg)提供的战略增长方法。
   - 该方法详细介绍了应对初创公司障碍和最大化估值的实用见解。
- **标普 500 指数创收盘历史新高**：标普 500 指数近期创下收盘历史新高，表明市场表现强劲且投资者信心充足，参考此[讨论](https://www.perplexity.ai/page/s-p-500-hits-record-closing-hi-yPKWo3jUQPOAfqvvoQ12Kg)。
   - 这一里程碑展示了市场在各种经济挑战中的韧性。
- **了解地震**：一名成员寻求关于**地震如何发生**的澄清，并提供了指向相关资源的科普链接，见[此处](https://www.perplexity.ai/search/how-do-earthquakes-happen-rlsZqPoKRS2jMv7t0PmrSw#0)。
   - 该讨论旨在消除关于地震活动和地壳运动的误解。
- **最佳动作冒险电影**：几位用户讨论了**动作冒险电影**，并指向了一份精选的最佳影片清单，见[此处](https://www.perplexity.ai/search/best-action-adventure-movies-c-lft_ADWwSLW6rj12clUyig)。
   - 这次对话重点介绍了在类型片爱好者中引起兴奋的热门选择。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1333070624041467986)** (4 messages): 

> `Sonar JSON 响应格式, LinkedIn URL API, 响应格式问题, Sonar vs Sonar-Pro` 


- **Sonar JSON 响应格式 Bug**：成员们对使用 `sonar` 模型时出现无效 JSON 响应表示沮丧，响应内容被 Markdown 包裹。
   - 有人指出切换到 `sonar-pro` 可以解决该问题，但也对成本影响表示担忧。
- **尝试通过 API 获取 LinkedIn URL**：一名成员正努力通过 API 仅提供用户名和工作单位来检索 LinkedIn URL，但经常收到不相关的结果。
   - 他们正在寻求改进 Prompt 或其他策略的建议，以获得更好的结果。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1333471715258531902)** (1 messages): 

> `Nous Psyche, 协作式 AI 训练, 开源模型, 异构计算` 


- **Nous Psyche 发布协作式 AI 训练网络**：今天我们宣布 **Nous Psyche**，这是一个在 **@Solana** 上运行的生成式 AI 协作训练网络，旨在利用**异构计算 (Heterogeneous Compute)** 创建开源模型。
   - 该倡议旨在挑战“只有封闭实验室才能推动**超级智能 (Superintelligence)** 前沿”的论调。
- **探索 Psyche 神话**：该项目从 Psyche 的神话中汲取灵感——这是一个凡人**在神圣的逆境中寻求救赎**的故事。
   - 关于这一迷人叙事的更多见解可以在我们的 [博客](https://nousresearch.com/nous-psyche/) 上找到。
- **Psyche 的 GitHub 仓库**：您可以在 [GitHub](https://github.com/PsycheFoundation/psyche) 上探索我们的开放基础设施，其目标是为人类实现**超级智能**开发的**民主化**和**去中心化**。
   - 该倡议鼓励更广泛地参与推进 AI 技术，并旨在重塑所有权和可访问性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1883912370696704011">来自 Nous Research (@NousResearch) 的推文</a>: 近期的 AI 突破挑战了现状论调，即只有封闭的大型实验室才有能力推动超级智能的前沿。今天我们宣布在 @Solana 上构建 Nous Psyche - 一个协作式...</li><li><a href="https://github.com/PsycheFoundation/psyche">GitHub - PsycheFoundation/psyche: 一个旨在为人类实现超级智能开发民主化和去中心化的开放基础设施。</a>: 一个旨在为人类实现超级智能开发民主化和去中心化的开放基础设施。 - PsycheFoundation/psyche
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1332441913919275149)** (681 messages🔥🔥🔥):

> `Nous Psyche 公告、Testnet 参与、分布式训练与声誉系统、诈骗代币、协作式开源开发` 


- **Nous Psyche 公告**：Nous 宣布推出 Psyche，这是一个建立在 Solana 之上的生成式 AI 协作训练网络，强调协作而非竞争。
   - 该项目旨在利用去中心化和无须信任的计算，许多人对其在 AI 发展方面的意义深感兴趣。
- **Testnet 参与**：参与者对即将到来的 Psyche Testnet 表示兴奋，尽管有一些技术要求，但大家仍期待其能提供用户友好的体验。
   - Testnet 预计很快就会上线，更多细节预计将在 30 日与 Solana Foundation 共同举办的活动中公布。
- **分布式训练与声誉系统**：讨论围绕验证协议展开，以防止恶意节点在系统中占据主导地位，其中包括声誉系统和概率检查。
   - 有人对声誉系统增加的复杂性表示担忧，这可能会使安全属性的分析变得复杂。
- **诈骗代币**：社区被警告要警惕冒充 Nous 品牌的欺诈性代币，并确认目前没有官方代币与 Psyche 项目相关联。
   - 鼓励用户举报任何诈骗或冒充者，以维护社区的诚信。
- **协作式开源开发**：参与者强调了开源社区内协作的重要性，将其比作一部体育动漫，每个人都旨在共同进步。
   - 大家一致认为，为该项目做出贡献可以推动生成式 AI 的重大进步。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.youtube.com/@ambiance461">Ambiance</a>: 该频道的主要目标是上传 YouTube 上缺失的氛围/灵魂风格的 DnB（我只上传 YouTube 上没有的内容）-----------------------------------------------------------...</li><li><a href="https://huggingface.co/DavidAU/L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B">DavidAU/L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/ifable/gemma-2-Ifable-9B">ifable/gemma-2-Ifable-9B · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2310.10837">Approximating Two-Layer Feedforward Networks for Efficient Transformers</a>: 如何在不牺牲性能的情况下减少神经网络 (NNs) 的计算和内存需求？许多最近的工作使用稀疏专家混合 (MoEs) 来构建资源高效的大语言...</li><li><a href="https://x.com/ryunuck/status/1883032334426873858">ryunuck (p≈np) (@ryunuck) 的推文</a>: 我已经完全解开了 Q* 的奥秘：它是 LLM 的一个新基础模块，一个以文本为条件的空间计算机模型。在这条推文的附件中，你可以看到一个为路径寻找而训练的模型...</li><li><a href="https://wiki.pygmalion.chat/bot-creation/trappu/introduction">Introduction</a>: PLists 和 Ali:Chat 的介绍。</li><li><a href="https://x.com/NousResearch/status/1883912370696704011">Nous Research (@NousResearch) 的推文</a>: 最近的 AI 突破挑战了现状，即只有封闭的大型实验室才有能力推动超智能的前沿。今天我们宣布构建在 @Solana 上的 Nous Psyche —— 一个酷炫的...</li><li><a href="https://publish.obsidian.md/hallerite/rl-for-deepseek-r1">the rl for deepseek-r1 - Entropic Musings - Obsidian Publish</a>: DeepSeek-R1 的 RL - Entropic Musings - 由 Obsidian Publish 提供支持。</li><li><a href="https://x.com/junxian_he/status/1883183099787571519">Junxian He (@junxian_he) 的推文</a>: 我们仅用 8K 示例就在 7B 模型上复现了 DeepSeek-R1-Zero 和 DeepSeek-R1 的训练，结果出奇地强。🚀 从 Qwen2.5-Math-7B（基础模型）开始，我们直接对其进行 RL...</li><li><a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - Lain Lain iwakura Serial experiments lain - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://fxtwitter.com/RLanceMartin/status/1883209736629448725">Lance Martin (@RLanceMartin) 的推文</a>: R1 Deep Researcher：基于 @deepseek_ai R1 + @ollama 的全本地研究助手。给 R1 一个主题，看它搜索网页、学习、反思、进一步搜索，只要你愿意就一直重复。为你提供一份带有...的报告。</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=header">危机沟通策略比较调查问卷</a>: 本调查探讨了文化价值观如何影响摩洛哥和美国环境灾难期间的危机沟通策略。它调查了媒体使用、公众信任和受众...</li><li><a href="https://x.com/cneuralnetwork/status/1883195767986569430">neural nets. (@cneuralnetwork) 的推文</a>: 发布 DeepSeek R1 博客，详细解释了整篇论文，没有遗漏任何数学内容，但任何具备基础高中数学知识的人都能理解（链接在回复中），请分享和转发...</li><li><a href="https://fxtwitter.com/Teknium1/status/1882893748742598669">Teknium (e/λ) (@Teknium1) 的推文</a>: 我们使用 5k 个 DeepSeek R1 蒸馏的 CoT 重新训练了 Hermes。我可以确认几件事：1. 你可以拥有通用 + 推理模式，我们使用静态系统提示词标记了来自 R1 的所有 longCoT 样本，这...</li><li><a href="https://x.com/disclosetv/status/1883675709954298338?t=UJRV7ZCFU0xIEYnwuO-xIg&s=19">Disclose.tv (@disclosetv) 的推文</a>: 新闻：中国的 DeepSeek AI 超越了 ChatGPT，目前在美国 Apple 免费应用下载排行榜中位列第一。</li><li><a href="https://docs.psyche.network">Intro to Psyche - Psyche</a>: 未找到描述</li><li><a href="https://huggingface.co/nbeerbower/mistral-nemo-gutenberg-12B-v2">nbeerbower/mistral-nemo-gutenberg-12B-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: DeepSeek-R1 的完全开源复现</a>: DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=ghltnvQmYKA">DeepSeek 可能会结束美股的特殊性：3 分钟 MLIV</a>: Anna Edwards, Guy Johnson, Kriti Gupta 和 Mark Cudmore 在 "Bloomberg: The Opening Trade" 中为分析师和投资者解析今日的关键主题。--------Mo...</li><li><a href="https://github.com/langchain-ai/ollama-deep-researcher">GitHub - langch</a>

ain-ai/ollama-deep-researcher: 完全本地的网络研究和报告撰写助手</a>: 完全本地的网络研究和报告撰写助手 - langchain-ai/ollama-deep-researcher</li><li><a href="https://www.youtube.com/watch?v=1xDVbu-WaFo">Hugging Face Journal Club - DeepSeek R1</a>: Hugging Face 的后训练团队讨论了 DeepSeek 开创性的 R1 模型背后的技术报告。- 报告：https://github.com/deepseek-ai/DeepSeek-...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/issues/29">在消费级显卡上进行速通？ · Issue #29 · KellerJordan/modded-nanogpt</a>: 你好，感谢这个出色的仓库！如果能在消费级显卡（如 RTX4090）上进行速通，我将不胜感激。由于它是 125M 参数，RTX4090 的 24GB 显存应该能以经典方式容纳，...</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1332713452774490133)** (14 messages🔥): 

> `R1 Distillation Models, Llama 3 performance issues, Image Captioning with DeepSeek, Building AI Assistants, Fine-tuning for performance` 


- **探讨 R1 蒸馏模型**：一位成员询问了 R1 蒸馏模型的使用，建议将其应用于类似 R1 的数据集，并引用了一篇描述蒸馏过程的[最新论文](https://arxiv.org/abs/2407.06023v3)。
   - 另一位成员建议复制像 [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) 这样的系统作为潜在的改进方案。
- **Llama 3 表现出异常行为**：一位用户对 **Llama 3** 模型在连续两天稳定运行后提供荒谬的回复表示沮丧。
   - 相比之下，另一位用户报告了使用 **Llama 3b instruct** 成功执行任务的情况，这表明性能差异可能取决于具体模型。
- **图像字幕生成中的快速收敛**：一位成员报告称，在针对图像字幕生成任务（通常被认为具有挑战性）微调蒸馏后的 DeepSeek R1 时，收敛速度很快。
   - 他们在 [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb) 中分享了他们的工作，展示了极具前景的结果。
- **启动本地 AI 助手项目**：一位用户寻求创建本地 AI 助手的指导，询问是否需要像 **Llama** 这样的模型，并寻求初学者资源。
   - 有人建议与 **Ask DeepSeek/Hermes/ChatGPT** 等工具协作以促进学习。
- **通过财务自由优化学习速度**：一位成员建议利用 LLM 资源、互联网工具和社区项目进行高效学习。
   - 他们强调要针对学习速度进行优化，尤其是在具备财务灵活性时。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.06023v3">Distilling System 2 into System 1</a>: 大语言模型 (LLMs) 可以在推理过程中消耗额外的计算量来生成中间思维，这有助于产生更好的最终响应。自 Chain-of-Thought (Wei et al., 2022) 以来，许多...</li><li><a href="https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k">bespokelabs/Bespoke-Stratos-17k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/githubpradeep/notebooks/blob/main/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb">notebooks/VLM_DeepSeek-R1-Distill-Qwen-1.5B.ipynb (main 分支) · githubpradeep/notebooks</a>: 通过在 GitHub 上创建账号来为 githubpradeep/notebooks 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1333413045912735775)** (2 条消息): 

> `类人 LLM 增强，危机沟通策略` 


- **LLM 类人响应研究**：一篇题为 *Enhancing Human-Like Responses in Large Language Models* 的论文探讨了通过增强自然语言理解和情感智能（利用微调和心理学原理等技术），使 LLM 更加**类人化**的进展。
   - 研究结果表明，这些增强功能**改善了用户交互**，并为 AI 应用开辟了新的可能性，同时也引发了对这些属性带来的**伦理影响**的进一步审视。
- **危机沟通策略问卷**：一份名为 *Comparative Crisis Communication Strategies* 的问卷旨在探讨**文化价值观**如何影响摩洛哥和美国在环境灾难期间的危机沟通。
   - 它专门调查了媒体使用情况和受众反应，以了解传统和数字通信在这些文化背景下塑造灾难响应的作用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.05032">Enhancing Human-Like Responses in Large Language Models</a>: 本论文探讨了使大语言模型 (LLM) 更加类人化的进展。我们专注于增强自然语言理解、对话连贯性和情感的技术...</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing">Questionnaire on Comparative Crisis Communication Strategies</a>: 这项调查探讨了文化价值观如何影响摩洛哥和美国在环境灾难期间的危机沟通策略。它调查了媒体使用、公众信任和受众...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1332843827794280488)** (2 条消息): 

> `LLM Live2D 助手，Qwen2.5-VL 模型，OCR 能力` 


- **遇见你的新助手：LLM Live2D！**：[LLM Live2D Desktop Assistant](https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant) 现已支持 Windows 和 MacOS，具有语音命令和与角色的独特交互功能。
   - 它结合了屏幕感知和剪贴板内容检索以增强用户体验，提供无缝的全电脑控制。
- **Qwen2.5-VL 在 OCR 方面表现出色！**：新推出的 [Qwen2.5-VL 模型](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) 以其出色的光学字符识别 (OCR) 能力脱颖而出，包括手写分析。
   - 它包含用于理解视觉内容的高级功能，同时可作为电脑和手机操作的动态工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ylxmf2005/LLM-Live2D-Desktop-Assitant">GitHub - ylxmf2005/LLM-Live2D-Desktop-Assitant: 由 LLM 驱动的 Live2D 桌面助手！支持 Windows 和 MacOS，它可以感知屏幕、检索剪贴板内容，并以独特的声音响应语音命令。具有语音唤醒、唱歌功能和全电脑控制，可与你喜爱的角色进行无缝交互。</a>: 你的 LLM 驱动 Live2D 桌面助手！支持 Windows 和 MacOS，它能感知屏幕，检索剪贴板内容，并以独特的声音响应语音命令。具有...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1333413045912735775)** (2 条消息): 

> `Human-Like Large Language Models, Crisis Communication Strategies` 


- **类人 LLM 的进展**：名为 [Enhancing Human-Like Responses in Large Language Models](https://arxiv.org/abs/2501.05032) 的论文探讨了提高 AI 系统中**自然语言理解**和**情感智能**的技术。
   - 该研究评估了使用多样化数据集进行微调（fine-tuning）等方法，并指出这些增强功能可以带来更好的用户交互，同时也引发了关于偏见的伦理担忧。
- **关于危机沟通的硕士项目问卷**：一名成员分享了一份[问卷](https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing)，重点关注摩洛哥和美国环境灾难期间**文化价值观**对危机沟通策略的影响。
   - 该调查研究了**媒体使用**、**公众信任**和受众反应等方面，强调了沟通方式如何影响灾难响应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfvGy-3sh-GzcwdWmxJ-1qttRlI8MOpYEQmk_kz9aCsstPnvw/viewform?usp=sharing">比较危机沟通策略问卷</a>：这项调查探讨了在摩洛哥和美国发生环境灾难期间，文化价值观如何影响危机沟通策略。它研究了媒体使用、公众信任和受众...</li><li><a href="https://arxiv.org/abs/2501.05032">Enhancing Human-Like Responses in Large Language Models</a>：本文探讨了使大型语言模型 (LLMs) 更加类人化的进展。我们专注于增强自然语言理解、对话连贯性和情感...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 条消息): 

voltamachine: neat
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1332446242294333632)** (1 条消息): 

> `ChatGPT Canvas Update, OpenAI o1 Integration, HTML & React Rendering, Desktop App Features` 


- **ChatGPT Canvas 现已支持 OpenAI o1**：OpenAI 今天宣布 **Canvas** 现已支持 **OpenAI o1**；用户可以从模型选择器中选择 o1 或使用 `/canvas` 命令。
   - 此更新适用于 **Pro**、**Plus** 和 **Team** 用户，扩展了 Canvas 工具的通用性。
- **Canvas 可以渲染 HTML 和 React 代码**：在最新更新中，Canvas 现在可以直接在 ChatGPT 中渲染 **HTML** 和 **React** 代码。
   - 此功能对 **Pro**、**Plus**、**Team** 甚至 **Free** 用户开放，增强了平台的能力。
- **Canvas 已在 macOS 桌面应用中全面推出**：Canvas 的更新已在 **macOS** 版 **ChatGPT 桌面应用**中全面推出，适用于所有层级的用户。
   - 这意味着所有用户现在都可以在桌面上无缝使用 Canvas。
- **Enterprise 和 Edu 更新即将推出**：Canvas 的这两项更新将在几周内向 **Enterprise** 和 **Edu** 用户推出。
   - 这确保了随着功能的逐步实施，更多用户能够访问最新特性。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1332449891028308130)** (537 条消息🔥🔥🔥): 

> `DeepSeek vs OpenAI models, Impact of DeepSeek on stock market, AI competition in tech industry, Performance comparisons of LLMs, User experiences with AI models`

- **DeepSeek 与 OpenAI 模型**：许多用户正在将 DeepSeek R1 与 OpenAI 的模型（特别是 O1 和 GPT-4o）进行比较，一些人指出 DeepSeek 在生成的代码中通常需要更少的修正。
   - 部分参与者表达了对 DeepSeek 的偏好，理由是其性能相当且成本更低，从而引发了关于哪种模型更优越的讨论。
- **DeepSeek 对股市的影响**：据报道，DeepSeek 能力的发布导致美国科技股大幅下跌，其中包括 Nvidia，其市值缩水了近 6000 亿美元。
   - 行业观察人士正在思考像 DeepSeek 这样具有竞争力的 AI 模型的出现，将如何颠覆老牌科技公司以及更广泛的市场。
- **科技行业的 AI 竞争**：随着 DeepSeek 的开源模型提供强有力的竞争，人们开始担心 OpenAI 等老牌玩家将如何应对 AI 格局的这一转变。
   - 讨论参与者强调了 AI 领域创新和竞争日益增长的重要性，尤其是随着更多价格亲民的模型出现。
- **LLM 的性能对比**：用户正在分享不同 AI 模型的基准测试结果和个人体验，表明性能会根据具体任务而有很大差异。
   - DeepSeek 和 Gemini 经常被提及，作为在某些应用中可以超越 O1 和 GPT-4o 等传统产品的模型。
- **AI 模型的使用体验**：贡献者们正在讨论他们使用 DeepSeek 和其他 AI 模型进行编码任务的第一手经验，并注意到成功率的差异。
   - 虽然一些用户对传统的 AI 模型表示沮丧，但其他人对新替代方案的性能和能力表示认可和满意。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.nextplatform.com/2024/10/25/cerebras-trains-llama-models-to-leap-over-gpus/">Cerebras 训练 Llama 模型以超越 GPU</a>：就在几个月前，晶圆级计算先驱 Cerebras Systems 还在吹嘘其几台连接在一起的 WSE-3 引擎可以运行……</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>：R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://www.cnbc.com/2025/01/27/chinese-ai-applications-are-looking-to-move-beyond-chatbots.html">中国 AI 应用现在有了更大的目标——它们正着眼于聊天机器人之外</a>：过去一周发布的一系列产品展示了中国公司如何迅速推出与 OpenAI 的 ChatGPT 竞争的 AI 模型。</li><li><a href="https://www.cnbc.com/2025/01/24/how-chinas-new-ai-model-deepseek-is-threatening-us-dominance.html">中国的新 AI 模型 DeepSeek 如何威胁美国的霸权</a>：一家来自中国的实验室在发布了比美国巨头更便宜、且使用性能较低芯片的令人印象深刻的 AI 模型后，引发了硅谷的恐慌。</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M">Qwen/Qwen2.5-14B-Instruct-1M · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/paul_cal/status/1882440978872865020">来自 Paul Calcraft (@paul_cal) 的推文</a>：Anthropic CEO @DarioAmodei 加入了新兴的 RL 合唱，“这不像推理或测试时计算 [..] 是一种完全全新的方法，它更像是一种涌现属性 [..] 训练的……”</li><li><a href="https://x.com/thinking_panda/status/1883849302939971783">来自 ShanghaiPanda (@thinking_panda) 的推文</a>：中国的 #DeepSeek 现已抹去美国股市 2 万亿美元的市值。😜 过去中国打破美国技术垄断（制造业）需要几十年。后来，是几年（互联网……）</li><li><a href="https://www.cnn.com/2025/01/27/tech/deepseek-stocks-ai-china/index.html">一项名为 DeepSeek 的震撼性中国 AI 进展正导致美股暴跌 | CNN 商业</a>：未找到描述</li><li><a href="https://x.com/CodeByPoonam/status/1883175938613207134?t=-uNTSIJlDYOx3QEMAE4A-w&s=19">来自 Poonam Soni (@CodeByPoonam) 的推文</a>：再见 ChatGPT。Deepseek R1 发布仅 5 天，世界就已经被它的潜力所震撼。13 个让你大吃一惊的例子（不要错过第 5 个）：</li><li><a href="https://youtu.be/7GV_OdqzmIU?si=IKCRD0tUOkHtOplS">Cerebras 联合创始人解析 Blackwell GPU 延迟</a>：Cerebras 首席系统架构师兼联合创始人 J.P. Fricker 解释了 Nvidia Blackwell 的技术挑战。00:12 中介层介绍 02:54...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1332908414527078470)** (34 条消息🔥): 

> `O3 Mini 发布日期, O3 Mini 特性, Tiktoken 中的 Tokenization, Gemini vs GPT, 从 Word 文件中抓取 URL` 


- **O3 Mini 发布日期推测**：备受期待的 **O3 Mini** 发布据传是在本周初，尽管可能会因不可预见事件而延迟。
   - 成员们渴望更新，一些人表示关于消息限制的承诺看起来很“疯狂”。
- **关于 O3 Mini 多模态能力的辩论**：有人担心 **O3 Mini** 可能只是 **O1 Mini** 的增强版，缺乏用户渴望的多模态功能。
   - 用户对限制表示失望，希望能有增强的功能来更有效地解决复杂问题。
- **Tiktoken 的 Tokenization 挑战**：一位用户询问为什么 **Tiktoken** 有时会将 token 处理为单个字符而不是合并它们，并指出特定输入存在不一致性。
   - 另一位用户建议特殊 token 限制可能会影响这种行为，并引用了研究论文中记录的潜在限制。
- **比较 Gemini 2.0 和 GPT 的性能**：讨论强调虽然 **Gemini 2.0** 很有竞争力，但它缺乏 **GPT** 中所见的高级功能和复杂的集成。
   - 用户注意到 **Gemini 2.0** 无法使用 LaTeX 格式化复杂的数学问题，这是 GPT 在教育场景中的显著优势。
- **从 Word 文件中抓取 URL**：一位用户分享了使用 GPTs **抓取 URL**（嵌入在 Word 文件的锚文本中）时的困难，称其经常无法检索到完整路径。
   - 尽管预料到会有缺陷，但他们指出，虽然使用 XML 标签通常会产生更好的结果，但目前的方法并不可靠。



**提到的链接**：<a href="https://x.com/sama/status/1883294216329281627">来自 Sam Altman (@sama) 的推文</a>：好的，我们听到大家的声音了。*Plus 层级每天将获得 100 次 o3-mini 查询 (!)* *我们将尽快把 operator 引入 Plus 层级* *我们的下一个 agent 将在 Plus 层级推出并可用* 尽情享受 😊 引用中...

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1332520880063909908)** (12 条消息🔥): 

> `LangChain ChatPromptTemplate, 用户关于格式的反馈, 复杂 Prompt 与 Vector Stores` 


- **探索 LangChain 的 ChatPromptTemplate**：一位成员询问是否有人尝试过将 LangChain 的 **ChatPromptTemplate** 用于从外部 **vector store** 提取上下文的复杂 Prompt。
   - 他们指出官方文档缺乏将文档从 **vector store retriever** 传递给 Prompt 模板的示例。
- **社区对实现方式的不确定性**：另一位成员回应称他们尚未尝试过，但建议使用**常规 Prompt 结构**作为变通方案。
   - 他们对结果表示感兴趣，并表示如果实现成功，渴望了解更多信息。
- **用户关于格式和清晰度的反馈**：一位用户对聊天中的清晰度和格式表示沮丧，要求**还他 10 分钟时间**。
   - 这一评论似乎强调了在未来的互动中需要更高效的讨论或更清晰的指示。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1332520880063909908)** (12 条消息🔥): 

> `Langchain ChatPromptTemplate, Vector Store 集成` 


- **关于 Langchain ChatPromptTemplate 使用的问题**：一位用户询问是否有人利用 Langchain 的 **ChatPromptTemplate** 处理复杂的 Prompt，这些 Prompt 在用户输入的同时引用外部 **vector store** 获取上下文。
   - 他们注意到缺乏将文档从 vector store 检索器传递到 Prompt 模板的文档示例，寻求社区见解。
- **对测试 ChatPromptTemplate 的兴趣**：另一位用户回答说他们还没有尝试过，但建议使用**常规 Prompt 结构**进行集成。
   - 他们对结果表示关注，并鼓励发帖者如果方案可行就分享他们的发现。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1332452988131868702)** (449 条消息🔥🔥🔥): 

> `DeepSeek API 问题, 推理提供商的盈利能力, R1 和 O1 模型对比, 新 AI 模型发布, Aider 用户体验`

- **DeepSeek 的 API 正在经历中断**：用户报告 DeepSeek API 间歇性宕机，导致 R1 模型的响应和输出出现问题。
   - DeepSeek 承认遭受了大规模恶意攻击，这可能是导致服务中断和对其服务高需求的原因。
- **推理提供商的盈利能力**：关于成为推理提供商盈利能力的讨论强调，在固定成本下，高利用率是盈利的关键。
   - 低利用率的推理可能产生微不足道的利润，特别是在评估竞争服务之间的定价策略时。
- **R1 与 O1 的性能对比**：一些用户声称，根据某些基准测试，DeepSeek 的 R1 模型结合 Sonnet 的表现优于 O1 结合 Sonnet。
   - 然而，其他人表示怀疑，指出 O1 Pro 在特定编程任务中可能仍然更胜一筹。
- **新 AI 模型发布**：Qwen2.5-1M 和 Janus-Pro 等新模型的推出使它们成为现有系统的有力竞争者，特别是由于它们具有 100 万 token 的上下文长度。
   - 随着推理框架和能力的进步，这些新模型在很大程度上被视为 AI 领域现有产品的增强。
- **Aider 的用户体验**：用户在使用 DeepSeek 时尝试在 Aider 中进行不同的配置，包括降低最大 token 限制以应对 API 问题。
   - 持续的服务中断促使用户探索替代模型和平台，同时分享改进性能的配置建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/Alibaba_Qwen/status/1883557964759654608">来自 Qwen (@Alibaba_Qwen) 的推文</a>：我们正在通过最新的开源模型 Qwen2.5-1M 提升竞争水平！💥 现在支持 100 万 TOKEN 上下文长度 🔥 以下是更新内容：1️⃣ 开源模型：迎接 Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://x.com/UnslothAI/status/1883899061893546254">来自 Unsloth AI (@UnslothAI) 的推文</a>：推出 1.58bit DeepSeek-R1 GGUFs！🐋 DeepSeek-R1 现在可以在 1.58-bit 下运行，同时保持功能完备。我们将 671B 参数模型从 720GB 缩小到仅 131GB —— 体积减少了 80%。朴素量化...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>：R1+Sonnet 在 aider 多语言基准测试中设定了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://openrouter.ai/deepseek/deeps">OpenRouter</a>：LLMs 的统一接口。为您的 Prompt 寻找最佳模型和价格。</li><li><a href="https://deepinfra.com/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 - Demo - DeepInfra</a>：DeepSeek-R1-Zero 是一个通过大规模强化学习 (RL) 训练的模型，没有经过监督微调 (SFT) 作为初步步骤，在推理方面表现出卓越的性能。尝试 A...</li><li><a href="https://plugins.jetbrains.com/plugin/25249-coding-aider">Coding Aider - IntelliJ IDEs 插件 | Marketplace</a>：将 Aider 的 AI 驱动编程辅助无缝集成到您的 IDE 中。这种集成通过提供快速访问精准代码来提高您的生产力...</li><li><a href="https://medium.com/@nimritakoul01/evaluating-the-ai-scientist-63e419e575b8">评估 AI Scientist</a>：在本文中，我将总结由 sakana.ai 开发的 AI 驱动的端到端 Agentic 流水线...</li><li><a href="https://aider.chat/docs/usage/watch.html">在您的 IDE 中使用 Aider</a>：Aider 可以监视您的文件，并对您在喜爱的 IDE 或文本编辑器中添加的 AI 注释做出响应。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">DeepSeek R1 (nitro) - API、供应商、统计数据</a>：DeepSeek R1 来了：性能与 [OpenAI o1](/openai/o1) 相当，但它是开源的，并且具有完全开放的推理 Token。它的规模为 671B 参数，在一次推理过程中有 37B 处于激活状态。运行...</li><li><a href="https://x.com/_akhaliq/status/1883914398127083665">来自 AK (@_akhaliq) 的推文</a>：DeepSeek 刚刚发布了一些新模型，人们还在适应 R1。Janus-Pro 是一个新颖的自回归框架，统一了多模态理解和生成。它解决了局限性...</li><li><a href="https://x.com/kimi_moonshot/status/1883532744225161369?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Kimi.ai (@Kimi_Moonshot) 的推文</a>：Kimi k1.5：多模态推理模型 - 现已在 http://Kimi.ai 上线 🦄💡 Kimi k1.5 能做什么？🔹 图像转代码：将图像转换为结构化代码和见解 🔹 GeoGuessr：识别并精准定位...</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://aider.chat/docs/install/docker.html">使用 Docker 运行 Aider</a>：aider 是您终端中的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/images-urls.html">图像与网页</a>：将图像和网页添加到 aider 编程聊天中。</li><li><a href="https://docs.docker.com/build/building/multi-stage/">多阶段构建</a>：了解多阶段构建以及如何使用它们来改进构建并获得更小的镜像</li><li><a href="https://aider.chat/docs/usage/commands.html?">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API、供应商、统计数据</a>：DeepSeek R1 Distill Llama 70B 是基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">来自 Qwen (@Alibaba_Qwen) 的推文</a>：🎉 恭喜发财🧧🐍 在迎接农历新年之际，我们激动地宣布推出 Qwen2.5-VL，这是我们最新的旗舰视觉语言模型！🚀💗 Qwen Chat: https://chat.qwenlm.ai 📖 Blog: http...</li><li><a href="https://github.com/deepseek-ai/awesome-deepseek-integration/blob/main/README.md">awesome-deepseek-integration/README.md at main · deepseek-ai/awesome-deepseek-integration</a>：通过在 GitHub 上创建账号，为 deepseek-ai/awesome-deepseek-integration 的开发做出贡献。</li><li><a href="https://status.deepseek.com/incidents/vx6w5ypzpgj7">【已恢复】DeepSeek 网页/API不可用（[Resolved]DeepSeek Web/API Service Not Available）</a>：未找到描述</li><li><a href="https://github.com/restatedev/sdk-python/">GitHub - restatedev/sdk-python: Restate SDK for Python</a>：Restate SDK for Python。通过在 GitHub 上创建账号，为 restatedev/sdk-python 的开发做出贡献。</li><li><

<a href="https://github.com/PierrunoYT/awesome-ai-dev-tools">GitHub - PierrunoYT/awesome-ai-dev-tools: A curated list of powerful and innovative AI-powered development tools, including code editors, plugins, and productivity enhancers.</a>: 一个精选的强大且创新的 AI 驱动开发工具列表，包括代码编辑器、插件和生产力增强工具。 - PierrunoYT/awesome-ai-dev-tools</li><li><a href="https://www.vxreddit.com/r/ChatGPT/comments/1i9bhuc/chatgpt_pro_me_and_my_wallet/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1332455338729013329)** (120 messages🔥🔥): 

> `Deepseek API Issues, Aider Functionality with Architect Mode, Model Pairing and Switching, Token Usage in Aider, Using Aider with Rust` 


- **Deepseek API 的问题**：用户报告了 **Deepseek API** 宕机或响应缓慢的问题，影响了 **Aider** 中的回复，即使状态页面显示其运行正常。
   - 几位成员尝试了不同的设置，并强调了检查 API 性能和关键配置的重要性。
- **Aider 中 Architect Mode 的问题**：一位成员指出，在 **architect mode** 中，**editor model** 的响应不可见，只显示 architect model 的响应。
   - 讨论围绕这究竟是一个 Bug 还是与浏览器功能的兼容性问题展开。
- **在 Aider 中切换模型的困难**：用户对在 Aider 中**临时切换模型**表示沮丧，指出更改主模型也会同时更改 editor model，从而导致工作流中断。
   - 参与者分享了变通策略，包括使用特定命令为单个 Prompt 切换模型。
- **Aider 中的 Token 使用监控**：关于在使用 Aider 时如何分别追踪 architect 和 editor model 的 **token usage** 提出了疑问。
   - 寻求关于 `/tokens --model sonnet` 等命令是否能按预期工作的澄清。
- **在 Aider 中集成新的 Rust Crates**：一位新用户询问如何将**新的 Rust crates** 引入 Aider，以获得更好的模型上下文理解和使用。
   - 探讨了将外部库添加到模型上下文的能力，突出了 Aider 与不同编程语言的集成。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://xx.x.x.xx:1234```">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: Architect model 描述如何解决编码问题，而 Editor model 将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。</li><li><a href="https://app.hyperbolic.xyz/models/deepseek-r1/api">Hyperbolic AI Dashboard</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: 使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://github.com/Aider-AI/aider/issues/2929">`/editor` for VSCode on Windows does not work · Issue #2929 · Aider-AI/aider</a>: 问题：在 Windows 上当编辑器设置为使用 VSCode 时，/editor 命令失败。$ aider --editor &quot;code --wait&quot; ────────────────────────────────────────────────────────────...</li><li><a href="https://github.com/Aider-AI/aider/issues/3020">specify the interval of lines where to read and/or modify · Issue #3020 · Aider-AI/aider</a>: 问题：我能否选择在文件的何处进行读取/修改，例如选择要放入上下文的行间隔？这在处理超大文件时非常有用。版本...</li><li><a href="https://www.aibase.com/news/14931">字节跳动发布豆包大模型 1.5 Pro，性能超越 GPT-4o 和 Claude3.5Sonnet</a>: 未找到描述</li><li><a href="https://www.aibase.com/tool/35837">Doubao-1.5-pro-Doubao-1.5-pro 是一款高性能的稀疏混合专家 (MoE) 大语言模型，专注于在推理性能和模型能力之间实现最佳平衡。</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1332686498834944000)** (3 messages): 

> `CodeGate integration with Aider, Comparative AI tools for web apps, Aider's functionality` 


- **CodeGate 现已集成 Aider**：[CodeGate](https://docs.codegate.ai/how-to/use-with-aider) 现在支持与 Aider 集成，使用户能够直接在终端中与 LLM 进行结对编程。
   - 此集成允许访问来自 **OpenAI** 和 **Ollama** 的模型，需要用户相应地配置其 API keys。
- **AI 对比工具更新请求**：一位用户创建了一个用于构建 Web 应用的 [AI 工具对比表](https://github.com/renatocaliari/comparative-ai-tools-for-building-web-apps)，其中包括 Aider，并寻求贡献以保持其更新。
   - 他们鼓励其他人如果知道任何需要添加或更新的相关工具，请在 GitHub 上提交 issue 或 pull request。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.codegate.ai/how-to/use-with-aider">在 Aider 中使用 CodeGate | CodeGate</a>：为 CodeGate 配置 Aider</li><li><a href="https://github.com/renatocaliari/comparative-ai-tools-for-building-web-apps">GitHub - renatocaliari/comparative-ai-tools-for-building-web-apps</a>：通过在 GitHub 上创建账号，为 renatocaliari/comparative-ai-tools-for-building-web-apps 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1332442697503477771)** (413 messages🔥🔥🔥): 

> `LM Studio Model Comparisons, DeepSeek R1 Distill Models, Using AI for Coding, Benchmarking AI Models, Chatter UI Setup Issues` 


- **对比 DeepSeek R1 Distill 模型**：用户正在讨论 DeepSeek R1 Distill Qwen 14b 与 7b 模型的性能对比，以及量化（quantization）等因素对输出质量的影响。
   - 高参数模型被认为拥有更多知识，尽管 Q3 和 Q4 等不同量化版本的有效性会根据模型的能力而有所不同。
- **使用 AI 工具进行编码**：个人表示有兴趣将 AI 集成到他们的编码工作流中以提高效率，一些人考虑将 R1 Distill 模型用于简单的编码任务。
   - 对模型性能的担忧以及参数规模与量化之间的权衡，引发了关于本地使用最佳配置的讨论。
- **为 AI 模型创建基准测试**：用户讨论了如何为各种 AI 模型创建基准测试（benchmarks），强调了基准数据集和自定义修改的重要性。
   - 建议利用 LiveCodeBench 等资源来有效地测试和对比模型输出。
- **Chatter UI 与 LM Studio 设置**：用户报告了在将 Chatter UI 连接到 LM Studio 时遇到的问题，特别是关于端口配置和运行模型方面。
   - 故障排除步骤包括确保正确的 URL 格式以及检查必要的设置以促进模型交互。
- **MoE 模型的性能**：讨论探讨了混合专家模型（MoE）激活模型特定部分的能力，提出了关于效率和知识保留的问题。
   - 强调虽然 MoE 允许先进的处理效率，但了解单个专家的能力对于有效的模型部署至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://llm.extractum.io/list/?query=deepseek%20r1">"deepseek r1" 搜索结果</a>: 在我们的 LLM Explorer 目录中，针对 'deepseek r1' 查询，在 3b、13b、30b 和 70b 的小型及大型开源语言模型中排名靠前的匹配项。</li><li><a href="https://llm.extractum.io/list/?query=deepseek">"deepseek" 搜索结果</a>: 在我们的 LLM Explorer 目录中，针对 'deepseek' 查询，在 3b、13b、30b 和 70b 的小型及大型开源语言模型中排名靠前的匹配项。</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - Qwen 系列</a>: 未找到描述</li><li><a href="https://tenor.com/view/correct-futurama-the-best-kind-of-correct-yes-yep-gif-5787390">Correct Futurama GIF - Correct Futurama The Best Kind Of Correct - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/NaniDAO/deepseek-r1-qwen-2.5-32B-ablated">NaniDAO/deepseek-r1-qwen-2.5-32B-ablated · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/api/openai-api">OpenAI 兼容性 API | LM Studio 文档</a>: 向 Chat Completions（文本和图像）、Completions 和 Embeddings 端点发送请求</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">导入模型 | LM Studio 文档</a>: 使用在 LM Studio 之外下载的模型文件</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio 社区)</a>: 未找到描述</li><li><a href="https://huggingface.co/livecodebench">livecodebench (Live Code Bench)</a>: 未找到描述</li><li><a href="https://finance.yahoo.com/news/ai-exposed-power-stocks-get-crushed-as-fears-about-deepseek-trigger-stock-market-sell-off-164007338.html">受 AI 影响的电力股惨遭重创，对 DeepSeek 的担忧引发科技股抛售</a>: 随着中国初创公司 DeepSeek 引发投资者对美国公司 AI 芯片支出的担忧，受 AI 影响的电力股随科技股抛售一同大跌。</li><li><a href="https://github.com/Vali-98/ChatterUI">GitHub - Vali-98/ChatterUI: 基于 react-native 构建的 LLM 简单前端。</a>: 基于 react-native 构建的 LLM 简单前端。通过在 GitHub 上创建账户为 Vali-98/ChatterUI 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe>">Mixture of Experts 详解</a>: 未找到描述</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLM</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://lmstudio.ai/docs/basics">LM Studio 入门 | LM Studio 文档</a>: 在 LM Studio 中本地下载并运行 Llama 3.1, Phi-3 和 Gemma 2 等大语言模型 (LLMs)</li><li><a href="https://lmstudio.ai/docs/advanced/tool-use">工具调用 | LM Studio 文档</a>: 使 LLM 能够与外部函数和 API 进行交互。</li><li><a href="https://github.com/ollama/ollama/issues/4643">Llama.cpp 现在支持跨多台机器的分布式推理。 · Issue #4643 · ollama/ollama</a>: Llama.cpp 现在支持跨多个设备进行分布式处理以提高速度，这将是 Ollama 的一个极佳补充 https://github.com/ggerganov/llama.cpp/tree/master/examples/rpc https://www.red...</li><li><a href="https://www.coursera.org/specializations/machine-learning-introduction">机器学习</a>: 由斯坦福大学和 DeepLearning.AI 提供。通过机器学习专项课程 #BreakIntoAI。掌握基础 AI 概念并... 免费注册。</li><li><a href="https://lmstudio.ai/docs/api/ttl-and-auto-evict">空闲 TTL 和自动驱逐 | LM Studio 文档</a>: 可选择在一定时间（TTL）后自动卸载空闲模型</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h5eyb8/lm_studio_running_on_npu_finally_qualcomm/?rdt=61321">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1332450586251231272)** (156 条消息🔥🔥): 

> `运行 DeepSeek 模型的硬件、在 LM Studio 中使用多 GPU、LLM 在 Apple M3 Max 上的性能、适合编程任务的理想 GPU、DDR5 内存与 AI 工作负载`

- **运行本地 LLM 的硬件指南**：用户讨论了运行本地 LLM 的硬件配置，建议重点关注具有高 VRAM 的 GPU，如 RTX 3090 或 A6000，以处理并发的补全请求。
   - 针对并发 Prompt 的担忧被提出，建议将负载均衡器（load balancer）与 llama.cpp 结合使用，以帮助高效管理多个请求。
- **Apple M3 Max 的性能预期**：一位成员询问了 DeepSeek-R1 模型在配备 48GB RAM 的 M3 Max 上的性能，另一位用户估计速度约为每秒 16-17 个 tokens。
   - 这反映了对 RAM 限制以及在 Apple 硬件上使用模型效率的考量。
- **为 RTX 3080 选择合适的模型**：针对适用于 RTX 3080 的编程模型给出了建议，其中 Qwen2.5 Coder 7b 被提及为一个可行的选择。
   - 用户承认了 RTX 3080 在处理大型模型时的局限性，促使他们测试更小的配置以获得更好的性能。
- **用于 LLM 的双 GPU 系统**：探讨了在单服务器设置中运行多个 GPU 的可行性，一些用户建议每个 GPU 需要独立加载模型，以便同时处理多个请求。
   - 讨论强调了使用负载均衡器和专门的软件配置（如 paddler）的必要性，以促进有效的多 GPU 设置。
- **DDR5 内存与 AI 处理能力**：用户对 DDR 内存的演进表示关注，讨论了未来的 DDR6 技术将如何提高 AI 工作负载的带宽。
   - 对话中强调了选择支持高内存带宽的正确服务器设置对于优化 LLM 性能的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.phoronix.com/review/supermicro-h13ssln-epyc-turin">来自 Supermicro H13SSL-N 针对 AMD EPYC 9005 "Turin" 1P 服务器的评测推文 - Phoronix</a>：未找到描述</li><li><a href="https://i.imgur.com/A2otU">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://www.phoronix.com/review/8-12-channel-epyc-9005">来自 AMD 第五代 EPYC (Turin) 8 通道与 12 通道 DDR5-6000 内存性能评测的推文 - Phoronix</a>：未找到描述</li><li><a href="https://www.pugetsystems.com/landing/Harrison-Kinsley---Intel-Xeon-W-3300-Workstation-156/">与 Harrison Kinsley 的合作</a>：Harrison Kinsley 在其所有商业网站上使用 Flask 进行 Web 开发，在 Ensmo.com 中使用 Scikit Learn 和 TensorFlow 进行机器学习和数据分析，并使用 Natural Language Toolkit 进行自然语言...</li><li><a href="https://tenor.com/view/kevin-hart-kevin-hart-damn-gif-22709278">Kevin Hart Kevin GIF - Kevin Hart Kevin Hart - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://x.com/Sentdex/status/1883596161778696247">来自 Harrison Kinsley (@Sentdex) 的推文</a>：我现在可以确认，是的，通过 DeepSeek R1，你在家就能拥有 AGI。经过多次尝试，使用 llama.cpp 在 CPU 和 RAM 上本地运行，速度约为 3.4 tokens/sec。</li><li><a href="https://www.reddit.com/r/radeon/s/LpQkNtcoNr">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/distantmagic/paddler">GitHub - distantmagic/paddler：为 llama.cpp 量身定制的有状态负载均衡器 🏓🦙</a>：为 llama.cpp 量身定制的有状态负载均衡器 🏓🦙 - distantmagic/paddler</li><li><a href="https://www.reddit.com/r/pcmasterrace/s/E1Gaw7Cspw">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference：多 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？</a>：多 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？ - XiongjieDai/GPU-Benchmarks-on-LLM-Inference</li><li><a href="https://www.aaronn.de/en/products/pocketai-accelerator/">Pocket AI - 便携式即插即用 AI 加速器 | Aaronn</a>：Pocket AI - 一款搭载 NVIDIA RTX GPU 的便携式 AI 加速器，为 AI 开发者和工业应用提供最大的灵活性和可靠性。</li><li><a href="https://lmstudio.ai/docs/system-requirements">系统要求 | LM Studio 文档</a>：LM Studio 在 Mac (M1/M2/M3/M4)、Windows (x64/ARM) 和 Linux (x64) 上支持的 CPU、GPU 类型</li><li><a href="https://lmstudio.ai/docs">LM Studio 文档 | LM Studio 文档</a>：了解如何使用 LM Studio 在本地运行 Llama、DeepSeek、Phi 和其他 LLM。</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ia4mx6/project_digits_memory_speed/">Project Digits 内存速度</a>：我最近看到了一张意外泄露的 NVIDIA 关于 Project Digits 内存速度的幻灯片。速度为 273 GB/s。此外，128 GB 是基础内存。只有...</li><li><a href="https://www.cybenetics.com/evaluations/psus/2570/#offcanvasExample">Cybenetics 测试 - SAMA GT650W</a>：未找到描述</li><li><a href="https://www.amazon.com/dp/B0DM2LC8HX?th=1">Amazon.com: SAMA 电源 850W，GT 850W 全模组 PSU 80 Plus 金牌效率，符合 ATX 3.1 和 PCIE 5.1 标准，支持 RTX 30 40 系列：电子产品</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1332448715771875419)** (4 条消息): 

> `Liquid AI 加入 OpenRouter，Nitro DeepSeek R1，Amazon Nova 模型问题` 


- **Liquid AI 发布多语言模型**：我们很高兴地宣布 [Liquid](https://liquid.ai) 已作为最新的提供商加入 OpenRouter，为平台带来了强大的专有模型，如 [LFM 40B](https://openrouter.ai/liquid/lfm-40b)、[LFM 3B](https://openrouter.ai/liquid/lfm-3b) 和 [LFM 7B](https://openrouter.ai/liquid/lfm-7b)。
   - LFM-7B 作为**同类最佳的多语言模型**脱颖而出，针对主要语言的性能进行了优化，拥有卓越的性能尺寸比。
- **Nitro DeepSeek R1 发布！**：DeepSeek R1 的全新 **Nitro 变体**现已上线，正如[公告](https://openrouter.ai/deepseek/deepseek-r1:nitro)中所述，它承诺提供更快、更可靠的性能。
   - 即将推出的功能包括动态 Nitro 变体，允许按速度对提供商进行排序，未来的更新将显示吞吐量的**中位数而非平均值**。
- **Amazon Nova 模型宕机**：目前，由于 Amazon Bedrock 的上游问题，**Amazon Nova 模型处于宕机状态**。该问题将使用量激增误判为密钥泄露，并返回误导性的**状态码 400**。
   - 我们正在积极寻求解决方案，并将在获得新信息时提供更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">DeepSeek R1 (nitro) - API, Providers, Stats</a>: DeepSeek R1 来了：性能与 [OpenAI o1](/openai/o1) 相当，但它是开源的，并具有完全开放的推理 Token。它的参数量为 671B，在一次推理过程中有 37B 处于激活状态。Ru...</li><li><a href="https://liquid.ai)">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/liquid/lfm-40b)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/liquid/lfm-3b)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/liquid/lfm-7b)">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: 全球同类最佳的英语、阿拉伯语和日语模型，原生支持法语、德语和西班牙语，经过优化，可作为私有企业聊天、代码、快速指令遵循等的基座...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1332442085231562862)** (535 条消息🔥🔥🔥): 

> `DeepSeek 模型性能，OpenRouter API 问题，模型建议与提交，BYOK 集成，DeepSeek 提供商当前状态`

- **DeepSeek 模型性能的不确定性**：用户报告 DeepSeek 模型的性能波动，特别是 R1，经历了响应缓慢和类似 '503 model is overloaded' 的错误。DeepSeek 的 Nitro 变体是通往 Fireworks 的更快快捷方式，但其表现并未达到预期。
   - 更新表明系统问题仍在持续，可能是由于巨大的用户需求导致的停机。
- **OpenRouter API 停机时间**：多名用户在通过 OpenRouter 使用 DeepSeek 时遇到了显著的延迟和错误，引发了关于是否迁移到直接使用 API 的讨论。建议使用来自 DeepSeek 的自带密钥 (BYOK) 以缓解速率限制 (rate limits)。
   - 用户对 API 的速度和可靠性表示沮丧，并将其与聊天室的性能进行了对比。
- **模型建议和提交流程**：一位用户询问了在建议被删除后，如何让模型获准在 OpenRouter 上使用的流程。针对模型需要有愿意接入 OpenRouter 的推理提供商 (inference providers) 这一要求提供了指导。
   - 该用户在尝试重新提交模型建议时遇到了速率限制问题。
- **在 OpenRouter 中集成 BYOK**：自带提供商 API 密钥允许用户通过其提供商账户直接控制速率限制和成本，OpenRouter 额度中会扣除 5% 的费用。讨论强调了使用 BYOK 对成本管理的潜在影响。
   - 建议用户将密钥接入 OpenRouter，以便更好地控制其 API 使用情况。
- **DeepSeek 提供商的现状**：DeepSeek 最近因恶意攻击面临问题，导致新注册受到服务限制。用户注意到 deepinfra 作为 R1 提供商的局限性，这可能是由于 OpenRouter 经历的可靠性问题所致。
   - 讨论强调了对 DeepSeek 服务的高需求，这导致了在维持稳定性能方面面临挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://flowith.io">flowith 2.0 - Your AI Creation Workspace, with Knowledge</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/crypto-api.">OpenRouter</a>: LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs">Quick Start | OpenRouter</a>: 开始使用 OpenRouter 构建</li><li><a href="https://operator.chatgpt.com/geo-blocked">Operator</a>: 一个可以使用自带浏览器为你执行任务的 Agent。</li><li><a href="https://openrouter.ai/api/v1`">OpenRouter</a>: LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/structured-outputs">Structured Outputs | OpenRouter</a>: 强制模型输出结构化内容</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/integrations#automatic-fallback)">Integrations | OpenRouter</a>: 在 OpenRouter 中使用你自己的供应商密钥</li><li><a href="https://arxiv.org/abs/2404.06654">RULER: What&#39;s the Real Context Size of Your Long-Context Language Models?</a>: 大海捞针 (NIAH) 测试旨在检查从长篇干扰文本（“干草堆”）中检索一条信息（“针”）的能力，已被广泛采用...</li><li><a href="https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys">Integrations | OpenRouter</a>: 使用你自己的供应商 API 密钥</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF">bartowski/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/parameters#include-reasoning">Parameters | OpenRouter</a>: 配置请求参数</li><li><a href="https://huggingface.co/Steelskull/L3.3-MS-Nevoria-70b">Steelskull/L3.3-MS-Nevoria-70b · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API, Providers, Stats</a>: DeepSeek R1 Distill Llama 70B 是基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i7o9xo/comment/m8n3rvk/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但已开源且具有完全开放的推理 Token。其参数量为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: 在多个供应商之间路由请求</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: DeepSeek 服务状态</li><li><a href="https://tenor.com/U8PF.gif">Snow White Parody GIF - Snow White Parody - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-Guard-3-8B">meta-llama/Llama-Guard-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/gomlx/gomlx">GitHub - gomlx/gomlx: GoMLX: An Accelerated Machine Learning Framework For Go</a>: GoMLX：一个为 Go 语言加速的机器学习框架 - gomlx/gomlx</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">no title found</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/faq">FAQ | DeepSeek API Docs</a>: 账户
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1332472633911934987)** (436 条消息🔥🔥🔥): 

> `AI 研发, 开源模型, Janus 系列发布, 互联网与技术历史, 联邦学习`

- **开源在 AI 发展中的作用**：讨论强调，随着格局的演变，美国政府限制开源 AI 模型的方法可能不会成功，并引用了 Meta 等公司持续使用此类模型的情况。
   - 参与者指出，虽然政府可能寻求控制开源计划，但创新驱动力和广泛采用可能会抵消这些努力。
- **Janus Pro 的发布**：Janus Pro 模型发布，标志着 DeepSeek 在 AI 研发方面的持续进展。
   - 对话表明，DeepSeek 的活动信号了在 AI 技术竞争格局中的不懈推进。
- **互联网发展的历史背景**：几位参与者讨论了互联网的起源及其最初由美国政府资助的情况，包括 ARPANET 的作用。
   - 对话强调了对互联网发展的各种贡献，并承认了塑造其早期基础设施的复杂国际格局。
- **关于研究方法的论述**：围绕各种 AI 方法论展开了辩论，参与者指出了研究中基础性进展与优化之间的差异。
   - 在运营效率的背景下，人们对新想法与成熟技术相比的有效性表示了担忧。
- **采用新技术的挑战**：一位参与者对调试新模型的复杂性提出了担忧，建议需要额外的层来辅助解释和理解。
   - 这引发了关于在创新方法与 AI 系统透明度和可理解性必要性之间取得平衡的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://fxtwitter.com/sama/status/1883294216329281627">Sam Altman (@sama) 的推文</a>：好的，我们听到大家的呼声了。*Plus 层级每天将获得 100 次 o3-mini 查询 (!)* 我们将尽快把 Operator 引入 Plus 层级 *我们的下一个 Agent 发布时也将向 Plus 层级开放，祝大家使用愉快 😊" 引述...</li><li><a href="https://www.theverge.com/2023/12/15/24003542/openai-suspends-bytedances-account-after-it-used-gpt-to-train-its-own-ai-model">OpenAI 在字节跳动使用 GPT 训练其自有 AI 模型后暂停了其账号。</a>：在今天的 Command Line 栏目中，我报道了字节跳动一直违反 Microsoft 和 OpenAI 的开发者许可，使用 GPT 生成的数据在中国训练其自有的竞争模型...</li><li><a href="https://en.wikipedia.org/wiki/List_of_largest_companies_by_revenue">按营收排名的最大公司列表 - Wikipedia</a>：未找到描述</li><li><a href="https://tenor.com/view/chess-gif-25810828">国际象棋 GIF - 国际象棋 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/michael-jackson-comendo-picoca-gif-9669437860846841235">迈克尔·杰克逊吃爆米花 GIF - Michael Jackson comendo picoca - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/adder-adderko-snake-ouroboros-overwerk-gif-21047022">Adder Adderko GIF - Adder Adderko 蛇 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/i-swear-to-god-nick-thorpe-fbi-international-i-promise-cross-my-heart-gif-26205749">我向上天发誓 Nick Thorpe GIF - I Swear To God Nick Thorpe Fbi International - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/capitulo0-capitulo-cero-ernesto-sevilla-david-lynch-chanante-gif-15470197">Capitulo0 Capitulo Cero GIF - Capitulo0 Capitulo Cero Ernesto Sevilla - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://d2l.ai/chapter_introduction/index.html">1. 简介 — Dive into Deep Learning 1.0.3 文档</a>：未找到描述</li><li><a href="https://fxtwitter.com/sama/status/1883185690508488934">Sam Altman (@sama) 的推文</a>：一场革命既无法被制造，也无法被阻止。唯一能做的，就是由它的几个孩子中的一个，凭借胜利来为它指明方向。——拿破仑</li><li><a href="https://fxtwitter.com/sama/status/1883305404089901269">Sam Altman (@sama) 的推文</a>：看大家对 Operator 的反应很有趣。让我想起了 ChatGPT 发布的时候！</li><li><a href="https://tenor.com/view/dahliabunni-popcorn-gif-11542556772657816665">Dahliabunni 爆米花 GIF - Dahliabunni Popcorn - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=iSmS6_j8Tnw">[IM] 功夫大师圆桌对决 - 叶问</a>：点赞并订阅以获取更多精彩片段...[IM] 功夫大师圆桌对决 - 叶问 #IPMan #KungFu 版权免责声明根据 1976 年版权法第 107 条...</li><li><a href="https://www.youtube.com/watch?v=i5Sdqf3jQkE">关于机器人技术的未来，他们都错了。</a>：机器人技术正处于历史性时刻的边缘，但该行业真正需要的是一个完全不同的时刻。</li><li><a href="https://www.youtube.com/watch?v=kPRA0W1kECg">6 分钟内演示 15 种排序算法</a>：在 6 分钟内对 15 种排序算法进行可视化和“听觉化”。对随机打乱的整数进行排序，速度和项目数量均经过调整...</li><li><a href="https://www.youtube.com/watch?v=mhKC3Avqy2E">训练大语言模型在连续潜空间中进行推理 – COCONUT 论文解析</a>：AI 不一定非要用文字思考。我们解析了 COCONUT (Chain of Continuous Thought) 🥥，这是一篇让 Chain-of-Thought 在向量而非...</li><li><a href="https://tenor.com/view/kuuchuu-buranko-ichiro-irabu-devi-word-of-the-day-irabu-ichiro-gif-26530271">空中秋千 伊良部一郎 GIF - Kuuchuu Buranko Ichiro Irabu Devi Word Of The Day - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reia.org/">首页</a>：未找到描述</li><li><a href="https://www.youtube.com/shorts/VNv-Cz-U6AY">“男子营救困在贝壳里的无助章鱼” #viralshort</a>：这段暖心的视频展示了一名男子在海滩上营救一只困在贝壳里的无助章鱼。看着他照顾它并与它玩耍，他们的纽带不断加深...</li><li><a href="https://reia.org/">首页</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/Janus">GitHub - deepseek-ai/Janus: Janus 系列：统一多模态理解与生成模型</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf (位于 main 分支) · deepseek-ai/Janus</a>：Janus 系列：统一多模态理解与生成模型 - deepseek</li>

-ai/Janus
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1332450182691815497)** (39 messages🔥): 

> `GPRO and PPO, Deepseek papers, Qwen2.5-VL model, Janus-Pro model, Friston podcast` 


- **GPRO 对 PPO 收敛的影响**：讨论了 GPRO 移除 **Value Function** 和 **Generalised Advantage Estimation (GAE)** 是否能缓解 **PPO** 中的 loss 停滞和早期收敛问题。
   - 有人指出，虽然 GAE 使用折扣和方法，但 GPRO 的方法利用了全局归一化的奖励模式。
- **Deepseek 论文持续发布**：成员们对 **Deepseek** 不断发布新论文感到兴奋，重点关注他们最新的 **Janus-Pro** 和 **Qwen2.5-VL** 模型。
   - 关于 **Qwen2.5-VL** 与之前模型差异的问题突出了其在视频理解方面的进步，展示了其快速的发展。
- **Qwen2.5-VL 的增强功能**：**Qwen2.5-VL** 模型理解复杂视觉并作为能够与工具交互的 **Agent** 的能力是讨论的一个重点。
   - 成员们注意到了分析图像和视频等显著特征，使 **Qwen2.5-VL** 成为视觉语言模型（vision-language models）领域的一个显著进步。
- **Friston 播客见解**：一位成员分享了 **Karl Friston** 的播客，讨论了将神经科学与智能联系起来的关键见解，包括自适应组织变革。
   - 核心主题包括 **active inference**（主动推理）的重要性以及可持续创新中生态系统和谐的必要性。
- **新论文活动公告**：即将举行关于 **Janus-Pro** 和 **Qwen2.5-VL** 的讨论，表明研究重点在于理解它们在 AI 发展中的影响。
   - 鼓励成员参加这些活动，特别是考虑到这些论文是最近发布的且热度正高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://open.spotify.com/episode/3ZAPncRTDzGGJSsFhlgtaB">Episode #45 | Karl Friston | Active intelligence, non-equillibrium steady states and enterprise jazz</a>：The Only Constant · Episode</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibhew9/qwen_just_launced_a_new_sota_multimodal_model/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/bytedance/UI-TARS-desktop">GitHub - bytedance/UI-TARS-desktop: A GUI Agent application based on UI-TARS(Vision-Lanuage Model) that allows you to control your computer using natural language.</a>：一个基于 UI-TARS（视觉语言模型）的 GUI Agent 应用程序，允许你使用自然语言控制电脑。- bytedance/UI-TARS-desktop</li><li><a href="https://arxiv.org/abs/2501.12326">UI-TARS: Pioneering Automated GUI Interaction with Native Agents</a>：本文介绍了 UI-TARS，这是一种原生 GUI Agent 模型，它仅感知屏幕截图作为输入，并执行类似人类的交互（例如键盘和鼠标操作）。与现有的 Agent 不同...</li><li><a href="https://github.com/bytedance/UI-TARS">GitHub - bytedance/UI-TARS</a>：为 bytedance/UI-TARS 的开发做出贡献。</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>：Janus 系列：统一的多模态理解与生成模型 - deepseek-ai/Janus
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1332931480959320144)** (17 条消息🔥): 

> `自然语言到 DSL 代码资源，PydanticAI 框架，生成式 AI 的结构化输出，健身记录 App 使用案例，DSL 与 JSON 的讨论` 


- **寻求 NL 到 DSL 代码的资源**：一位成员请求将 **Natural Language** 转换为 **Domain Specific Language** 代码的资源，并表示 **LLMs** 在没有大量微调的情况下很难完成这项任务。
   - 他们提到 **Microsoft ODSL 论文** 是一个很好的起点，同时也欢迎进一步的建议。
- **PydanticAI 框架探索**：讨论转向了 **PydanticAI** 框架，该框架旨在简化使用生成式 AI 构建生产级应用的过程，并分享了一个参考链接。
   - 一位成员对其 beta 状态表示不确定，而其他人则建议使用 **LlamaIndex+LangChain** 等替代方案来实现结构化输出。
- **定义健身记录 App 的使用案例**：一位成员概述了他们对 **健身记录 App** 的探索，旨在实现高效的用户交互以及在 **DSL** 中进行可组合交互的潜力。
   - 他们强调了实现 **voice to DSL** 转换的目标，并承认这是一个需要逐步探索的艰巨挑战。
- **DSL 中可执行性的重要性**：对话强调了 **DSLs** 与 **JSON** 之间的区别，一位成员断言 DSL 是可执行的，且需要比 JSON 更复杂。
   - 一种相反的观点认为，类 JSON 实例可以作为 DSL 运行，能够支持可执行步骤。
- **学习 LLM 数据提取的方法论**：成员们讨论了使用结构化输出创建基础的 **'hello world'** 示例，以了解 **LLMs** 可以从提示词中提取哪些类型的数据。
   - 这种实践方法被推荐用于阐明结构化输入在健身记录场景下如何影响 DSL 的行为。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://table-agent.com/">Table Agent</a>: 用于表格研究的 AI 驱动数据助手</li><li><a href="https://ai.pydantic.dev/">Introduction</a>: 用于在 LLMs 中使用 Pydantic 的 Agent 框架 / 适配层
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1332443465497182230)** (45 条消息🔥): 

> `DeepSeek AI 模型发布，Qwen2.5-VL 模型公告，Mistral 的 IPO 计划，AI 与经济影响，公众对 AI 治理的看法` 


- **DeepSeek AI 挑战美国竞争对手**：DeepSeek 开发了一款推理模型，据报道在没有使用先进 Nvidia 芯片的情况下性能超越了美国同行，这引发了人们对美国科技公司真实竞争优势的质疑。
   - 这导致了对其对股市潜在影响的推测，因为像 Satya Nadella 这样的行业领袖也开始认真对待 DeepSeek。
- **Qwen2.5-VL 在农历新年发布**：阿里巴巴 Qwen 宣布推出其旗舰视觉语言模型 **Qwen2.5-VL**，该模型具有先进的视觉理解和长视频理解能力。
   - 亮点包括其精确的定位能力和结构化数据输出，可增强金融和商业领域的任务。
- **Mistral 令人质疑的 IPO 声明**：围绕 Mistral 状态的矛盾信息引发了讨论，该公司声称“不予出售”，但据报道正在筹备 IPO。
   - 这引发了关于科技行业企业“双重思想”的揶揄评论。
- **公众对 AI 治理的情绪**：成员们对 AI 治理的想法表达了复杂的观点，指出如果 AI 监管能改善他们的生活，他们愿意接受。
   - 辩论强调了人类和 AI 主导系统中腐败和激励结构的挑战。
- **AI 领导地位的感知风险**：对话触及了这样一个观点：如果 AI 能够提供更好的生活质量，许多人会接受 AI 的领导，尽管存在潜在的负面影响。
   - 批评者对这种心态的影响表示担忧，将其与历史上治理失败的案例进行了比较。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/TFTC21/status/1882571514891080030">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1883954247743725963">来自 Qwen (@Alibaba_Qwen) 的推文</a>：🎉 恭喜发财🧧🐍 在我们迎接农历新年之际，我们很高兴地宣布推出 Qwen2.5-VL，我们最新的旗舰级视觉语言模型！🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://www.msn.com/en-gb/money/other/is-deepseek-about-to-cause-a-stock-market-crash/ar-AA1xV6nG">MSN</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>：未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-1m/">Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens</a>：技术报告 HuggingFace ModelScope Qwen Chat HuggingFace Demo ModelScope Demo DISCORD 简介 在升级 Qwen2.5-Turbo 以支持高达一百万 Tokens 的上下文长度两个月后，我们回来了...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek-R1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://tenor.com/view/rodney-king-get-along-gif-22105666">Rodney King Get Along GIF - Rodney King Get Along - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/ag2oss/status/1882878967713259705">来自 AG2 (@ag2oss) 的推文</a>：宣布我们对社区驱动 Agent 开发的愿景。阅读关于 AG2 的：- 治理模型 - 社区结构 - 开源承诺 - 前进道路 https://medium.com/@ag2ai/ag2s-vision-for...</li><li><a href="https://medium.com/@ag2ai/ag2s-vision-for-community-driven-agent-development-f9f3ca2b0dc8">AG2 对社区驱动 Agent 开发的愿景</a>：两年前我们在开发 FLAML 时首次提出了 AutoGen 背后的概念，我们的目标很简单：让它变得更容易……</li><li><a href="https://www.podchaser.com/podcasts/chinatalk-725507?">ChinaTalk</a>：由 Jordan Schneider 主持，426 集，2 条评分与评论。探讨中国、技术和美中关系的对话。嘉宾包括广泛的分析师、政策制定者和学者。H...</li><li><a href="https://uk.finance.yahoo.com/news/deepseek-cause-stock-market-crash-065127717.html">DeepSeek 是否即将引发股市崩盘？</a>：随着股市被专注于 AI 的美国科技公司主导，OpenAI 的竞争对手 DeepSeek 是否即将让一切崩塌？文章：DeepSeek 是否即将引发股市崩盘？a...</li><li><a href="https://www.youtube.com/watch?v=V-Fla5hxMRg">中国的 DeepSeek 引发全球科技股抛售</a>：CNBC 的 Andrew Ross Sorkin 和 Becky Quick 讨论当日新闻。如需观看来自 CNBC 的独家直播视频，请订阅 CNBC PRO：https://cnb.cx...</li><li><a href="https://tenor.com/view/bogdanoff-dump-it-stocks-crypto-gif-20477588">Bogdanoff Dump It GIF - Bogdanoff Dump It Stocks - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: DeepSeek-R1 的完全开源复现</a>：DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1332446763919085649)** (206 messages🔥🔥): 

> `DeepSeek 模型更新, Qwen 2.5-VL 发布, AI 公司策略, NVIDIA 市场地位, Edge AI 讨论`

- **DeepSeek 模型更新备受关注**：近期 DeepSeek 模型的发布引发了关于其颠覆当前 AI 范式的讨论，许多观察者注意到其高效性和开源权重（open-weight）的特性。
   - 尽管企业级大规模集成的确定性尚不明确，但人们预期 DeepSeek 将推动推理模型（reasoning models）的更广泛采用。
- **Qwen 2.5-VL 发布引发关注**：阿里巴巴推出了 Qwen 2.5-VL 作为其旗舰级视觉语言模型（vision-language model），具备长视频理解和精确重定位等功能。
   - 此次发布展示了 Qwen 致力于提升 AI 有效整合视觉和语言处理能力的决心。
- **AI 公司策略出现分歧**：讨论揭示了 OpenAI 与 Anthropic 之间的差距，前者似乎正在向多个领域多元化发展，而后者则专注于 LLM 的开发。
   - 外界对 OpenAI 以消费者为中心的策略持怀疑态度，认为这可能分散了其在维持前沿研究方面的精力。
- **AI 演进中 NVIDIA 面临市场挑战**：随着 AI 模型向高效化和商品化（commoditization）转变，人们对 NVIDIA 的市场估值日益担忧，这也引发了对其未来需求的质疑。
   - 随着竞争加剧和技术进步，投资者正在重新评估 NVIDIA 在 AI 硬件领域的地位。
- **关于 Edge AI 可行性的辩论**：Edge AI 的可行性受到质疑，因为大多数复杂的推理模型在不牺牲性能的情况下可能不适合本地部署。
   - 讨论表明，虽然 Edge AI 可能存在应用场景，但云端解决方案仍然是处理重型工作负载的首选。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/junxian_he/status/1883183099787571519">Junxian He (@junxian_he) 的推文</a>：我们仅用 8K 个样本就在 7B 模型上复现了 DeepSeek-R1-Zero 和 DeepSeek-R1 的训练，结果出奇地强。🚀 从 Qwen2.5-Math-7B（基座模型）开始，我们对其进行 RL...</li><li><a href="https://blog.vllm.ai/2025/01/27/v1-alpha-release.html">vLLM V1：vLLM 核心架构的一次重大升级</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Trudy/gemini-image-to-code">Gemini Image to Code - Trudy 在 Hugging Face Space 上的空间</a>：未找到描述</li><li><a href="https://x.com/bfspector/status/1883051606369001873">Benjamin F Spector (@bfspector) 的推文</a>：我们获得了首批 Nvidia B200 的早期访问权限。我们分享了初步的基准测试结果，并编写了最快的（公开）Attention Kernel，达到 925+ BF16 TFLOPs：自 PTX 指令集发布以来...</li><li><a href="https://x.com/Kimi_Moonshot/status/1883164161506738232">Kimi.ai (@Kimi_Moonshot) 的推文</a>：🚀 推出 Kimi k1.5 – 现已上线 Web 端 http://Kimi.ai！我们很高兴宣布在 Web 端发布 Kimi 1.5！我们还推出了英文支持（仍在微调中）。查看简易模式...</li><li><a href="https://x.com/DanHendrycks/status/1883660982641426727">Dan Hendrycks (@DanHendrycks) 的推文</a>：我同意。从美国竞争力的角度来看，你需要国内的 AI 芯片制造，而不仅仅是通过出口管制来限制 AI 芯片。影响 AI 能力的维度有很多：算法...</li><li><a href="https://x.com/_lewtun/status/1883142636820676965">Lewis Tunstall (@_lewtun) 的推文</a>：我们正在复现完整的 DeepSeek R1 数据和训练流水线，以便每个人都能使用他们的配方。与其秘密进行，不如让我们公开合作！🧪 第一步：复现 R1-Distil...</li><li><a href="https://x.com/lmarena_ai/status/1882875989610594542">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：❤️‍🔥 WebDev Arena 更新：令人兴奋的新条目！- #2: @deepseek_ai DeepSeek-R1 - #4: 新的 Gemini-2.0-Flash-Thinking。DeepSeek-R1 跃升至第 2 位，与 Claude 3.5 Sonnet 的差距不到 40 分，展现了强大的能力...</li><li><a href="https://x.com/TheStalwart/status/1883902565064352233">Joe Weisenthal (@TheStalwart) 的推文</a>：*DEEPSEEK：限制仅限中国手机号码注册</li><li><a href="https://x.com/alibaba_qwen/status/1883954247743725963?s=46">Qwen (@Alibaba_Qwen) 的推文</a>：🎉 恭喜发财🧧🐍 在迎接农历新年之际，我们激动地宣布推出 Qwen2.5-VL，我们最新的旗舰级 Vision-Language Model！🚀💗 Qwen Chat: https://chat.qwenlm.ai 📖 Blog: http...</li><li><a href="https://x.com/nrehiew_/status/1882853607307100162">wh (@nrehiew_) 的推文</a>：Anthropic，我已经为你做好了推介。忽略 Stargate，这就是一个价值 1T 的推介。右上角 = 优秀</li><li><a href="https://x.com/LiJunnan0409/status/1882620700567195976">Li Junnan (@LiJunnan0409) 的推文</a>：很高兴分享我的 http://Rhymes.ai 研究团队将加入 Salesforce Research @SFResearch！我将担任研究总监一职，向 @CaimingXiong 汇报。期待...</li><li><a href="https://x.com/Mobius_Labs/status/1882841665427390858">Mobius Labs (@Mobius_Labs) 的推文</a>：我们重新蒸馏的 @deepseek_ai R1 (1.5B) 表现优于原始蒸馏模型！请访问 https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0 获取。我们正在蒸馏更多模型...
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1332838823302004867)** (16 messages🔥): 

> `Self-Play Paradigm, Scaling Synthetic Data Pipelines, Role of Self-Determination Theory, Claims about Deepseek, Critique of Media Reporting` 


- **Self-Play 范式获得支持**：*无需人工参与的 Self-play* 正在被讨论为 AI 未来进步的主要框架，并得到了社区内爱好者的支持。
   - 一位社区成员引用道，*AI 能力的下一次飞跃将源于优化的框架*，如 MuZero。
- **合成数据流水线是关键**：一位成员强调了扩展 *合成数据流水线 (Synthetic Data Pipelines)* 的重要性，而不是仅仅依赖创新的训练方法。
   - 他们对 AI 模型缺乏基础数据的说法表示怀疑。
- **AI 讨论中的自我决定理论**：在围绕 AI 类人推理的讨论中，出现了一个关于引入 *自我决定理论 (Self-Determination Theory)* 的疑问，重点关注心理学框架。
   - 社区成员对围绕治疗导向的聊天机器人和积极心理学的文献表现出兴趣。
- **关于 Deepseek 数据源的说法**：关于 *Deepseek* 是否披露了使用 *Llama* 和 *Qwen* 进行数据生成的问题浮出水面，一些人将其归因于沟通误解。
   - 这一说法似乎源于对 *Deepseek-LLM* 的引用，而该模型是基于 Llama 架构的。
- **对媒体报道实践的批评**：人们对记者处理 AI 话题的方式感到不满，特别是关于发明权的声明，例如声称 *MoE 是在 Deepseek 发明的*。
   - 社区成员建议建立一个聊天群组，让记者可以从精通 AI 文献的人士那里获得反馈，以避免误报。



**提到的链接**：<a href="https://x.com/finbarrtimbers/status/1883243939031056813">finbarr (@finbarrtimbers) 的推文</a>：co-sign 引用 doomslide (@doomslide) 在再次阅读论文并试用 R1 后，我得出的结论是，极其可预测的下一次能力飞跃将是（巧妙地优化...

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1332990612559106099)** (38 条消息🔥): 

> `Gary Marcus 的观点, Nous Psyche 发布, 关于民族国家的政治视角, AI 突破叙事, 关于 AI 的学术视角` 


- **Gary Marcus 引发争议**：成员们对 Gary Marcus 的立场表示困惑和幽默，有人指出他为了博取关注而转向“对华鹰派”立场。
   - *许多人认为他在缺乏深度 AI 知识的学术界和新闻界受众中很吃得开*，而这些人往往忽略了他自相矛盾的言论。
- **Nous Psyche 的雄心勃勃的发布**：Nous Research 宣布了 **Nous Psyche**，这是一个基于 **Solana** 构建的生成式 AI 协作训练网络，旨在挑战只有封闭实验室才能推进超级智能的观点。
   - 尽管令人兴奋，但在提到 ***Nous 被黑客攻击*** 后，人们对安全性提出了担忧。
- **对民族国家的挫败感**：一位成员质疑了民族国家的相关性，称国家优越性的想法已经过时，并批评了统治精英对市场的控制。
   - 这种情绪反映了对政治结构变革需求的更广泛感受，引起了讨论中其他人的共鸣。
- **AI 讨论中的黑色幽默**：在技术讨论中，一位成员发现了一个与 AI 开发公告相关的编辑后评论非常幽默，称其为喜剧天才之作。
   - 这种轻松的氛围与对 AI 影响和平台安全性的更深层担忧形成了鲜明对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1883912370696704011">来自 Nous Research (@NousResearch) 的推文</a>: 最近的 AI 突破挑战了现状叙事，即只有封闭的大型实验室才有能力推动超级智能的前沿。今天我们宣布了构建在 @Solana 上的 Nous Psyche - 一个协作...</li><li><a href="https://x.com/rm_rafailov/status/1883419883150713023">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>: @teortaxesTex RLVR 根本不是个事。多年来我们至少在 1000 篇论文中训练过这个。在过去的最后三天之后，我正在认真考虑放弃公开研究。</li><li><a href="https://bsky.app/profile/mathiasgehrig.bsky.social/post/3lgqwb3rwtk2k">Mathias Gehrig (@mathiasgehrig.bsky.social)</a>: 最后一段陈述肯定是我今天读到的最错误的东西。我明白你为身为美国人或其他什么感到自豪，但即便如此。</li><li><a href="https://x.com/LiangWenfeng_/status/1883978669900824681">来自 Liang Wenfeng 梁文锋 (@LiangWenfeng_) 的推文</a>: 未找到描述</li><li><a href="https://x.com/steph_palazzolo/status/1883620099862773842">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>: 新消息：在中国对冲基金发布了一款令人印象深刻且极其廉价的 AI 模型后，美国 AI 公司正陷入混乱。Meta 已经设立了 4 个“作战室”来剖析 DeepSeek 模型，看看有什么见解...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1332447389839261796)** (177 条消息🔥🔥): 

> `DeepSeek 的崛起, 市场对 AI 模型的反应, Qwen 模型发布, 投资者情绪, 行业颠覆`

- **DeepSeek 对 AI 市场的影响**：DeepSeek 的 R1 模型发布引起了巨大轰动，引发了关于其与 OpenAI 和 Meta 等成熟模型竞争表现的热烈讨论。
   - 随着 DeepSeek 表现超出预期，许多行业专业人士甚至普通用户（如家人）现在都在询问它。
- **Qwen2.5-VL 的发布**：备受期待的 Qwen2.5-VL 模型发布正引发热潮，人们提到了它的多模态能力以及对竞争对手的潜在影响。
   - 观察家们注意到，这次模型发布可能会重塑 AI 领域的认知，类似于过去一些著名的发布。
- **发布后的投资者担忧**：金融市场正在对 AI 的发展做出反应；一些人认为 AI 模型发布的激增正导致股价大幅波动，尤其是像 Nvidia 这样的公司。
   - 投资者正在讨论其中的风险，理由是对市场饱和及 AI 投资未来的担忧。
- **公众兴趣与 AI 意识**：公众对 AI 的兴趣显著增加，越来越多的人开始询问创新模型和技术，包括 DeepSeek。
   - 讨论参与者报告称，朋友和家人都在向他们打听这些进展，这表明在传统科技圈之外，人们的意识也在不断增强。
- **AI 与股市动态的比较**：分析师和用户正在辩论这种快节奏的 AI 发展对股市的影响，并注意到像 Nvidia 这样成熟公司的意外下跌。
   - 讨论包括对股价重大后果的预测，这些预测受到 AI 相关技术和能力变动的推动。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://qwenlm.github.io/blog/qwen2.5-1m/">Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens</a>: 技术报告 HuggingFace ModelScope Qwen Chat HuggingFace Demo ModelScope Demo DISCORD 简介：在升级 Qwen2.5-Turbo 以支持高达一百万 tokens 的上下文长度两个月后，我们又回到了...</li><li><a href="https://x.com/garryt">Tweet from undefined</a>: 未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=42825193">For context: R1 is a reasoning model based on V3. DeepSeek has claimed that GPU ... | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/splitbycomma/status/1883588991813042605">Tweet from caspian (@splitbycomma)</a>: 普通人觉得 DEEPSEEK 很可爱，因为它分享了它的思考过程</li><li><a href="https://x.com/DavidSHolz/status/1883222685741879722">Tweet from David (@DavidSHolz)</a>: 在我的测试中，DeepSeek 在中国古代哲学和文学方面碾压了西方模型，同时对英语的掌握也比我的第一手中文资料强得多。感觉就像...</li><li><a href="https://fxtwitter.com/sethbannon/status/1883301772053332349?s=46">Tweet from Seth Bannon (@sethbannon)</a>: Youtube 和 Reddit 都屏蔽了 Operator。这是未来趋势的征兆吗？</li><li><a href="https://x.com/willccbb/status/1883414339518148960?s=61">Tweet from will brown (@willccbb)</a>: Llama-1B 上的 GRPO 自我修正 :')</li><li><a href="https://x.com/steph_palazzolo/status/1883620099862773842?s=46">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: 新闻：在中国一家对冲基金发布了一款令人印象深刻且极其廉价的 AI 模型后，美国 AI 公司正陷入混乱。Meta 已经成立了 4 个“作战室”来剖析 DeepSeek 模型，看看能得到什么启发...</li><li><a href="https://x.com/TheXeophon/status/1883048355875672457">Tweet from Xeophon (@TheXeophon)</a>: R1 CoT 的前四个词分析</li><li><a href="https://x.com/johnschulman2/status/1883221980931142113">Tweet from John Schulman (@johnschulman2)</a>: R1 的思维链与论文和博客文章中分享的 o1-preview CoT 之间存在一些有趣的相似之处（例如 https://openai.com/index/learning-to-reason-with-llms）。特别是...</li><li><a href="https://x.com/huybery/status/1883775353950519479">Tweet from Binyuan Hui (@huybery)</a>: 今晚有一些惊喜</li><li><a href="https://x.com/hamelhusain/status/1883707463251472448?s=46">Tweet from Hamel Husain (@HamelHusain)</a>: 蒸馏版的 70b-R1 现在已经在 Groq 上线了。在文档里藏得挺深，但确实在那儿。</li><li><a href="https://x.com/garrytan/status/1883655771067744441">Tweet from Garry Tan (@garrytan)</a>: 即使只用了几次，DeepSeek 搜索也让人感觉更有粘性，因为看到推理过程（甚至是它对已知和未知内容的诚实态度）大大增加了用户的信任感</li><li><a href="https://x.com/dylan522p/status/1883930768533332340">Tweet from Dylan Patel (@dylan522p)</a>: 所以这就像是用 600 万美元的训练成本（忽略研究成本、消融实验、来自 GPT 的蒸馏数据、各种集群的资本支出等）造成了 2 万亿美元的市值损失。想象一下，如果中国投入 3 亿美元在...</li><li><a href="https://x.com/reach_vb/status/1883911714158305719">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: 等一下，DeepSeek 刚刚发布了 Janus 7B（MIT 许可证）——多模态 LLM（也能生成图像）🔥</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - a Qwen Collection</a>: 未找到描述</li><li><a href="https://x.com/georgejrjrjr/status/1883629241742635313?s=61">Tweet from George (@georgejrjrjr)</a>: 他们早就知道！我怎么知道的？Meta 一年前就因为我的一篇深度帖联系过我，想雇佣我！引用 Amir Efrati (@amir) 的新闻：DeepSeek 引发的恐慌是真实的。Meta Platforms 担心 DS 是...</li><li><a href="https://x.com/btibor91/status/1883627800831365567?s=61">Tweet from Tibor Blaho (@btibor91)</a>: The Information 报道称，幻方量化（High-Flyer Capital）旗下的 DeepSeek AI 在性能和成本效益上已经超越了 Meta 的 Llama，并与 OpenAI 的模型不相上下，这引发了 Meta 的担忧和快速反应...</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>: Janus 系列：统一的多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://www.thetimes.com/world/europe/article/french-ai-lucie-looks-tres-chic-but-keeps-getting-answers-wrong-7vk2szmdg">French AI ‘Lucie’ looks très chic, but keeps getting answers wrong</a>: 这款由马克龙和公共资金支持的聊天机器人因提供错误信息而面临批评。甚至它的 Logo 也受到了质疑</li><li><a href="https://git">

hub.com/QwenLM/Qwen2-VL/commits/main/">Commits · QwenLM/Qwen2.5-VL</a>: Qwen2-VL 是由阿里巴巴云 Qwen 团队开发的多模态大语言模型系列。 - Commits · QwenLM/Qwen2.5-VL
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1332512167026626613)** (23 条消息🔥): 

> `Deepseek 性能、Scale.ai 担忧、中国科技评论、社交媒体虚假账号、文革对科技创始人的影响` 


- **Deepseek 性能对比**：一位成员表示，**o1 与 Sonnet 搭配**并没有比单独使用 **o1** 产生更好的结果，凸显了对模型性能的担忧。
   - 另一位成员指出，与单模型表现相比，使用各种模型作为编辑器并没有提高 **o1** 或 **R1** 的分数。
- **对 Scale.ai CEO 言论的担忧**：一位成员指出，提到拥有 **50k GPU** 的 **Scale.ai** CEO，如果对标注数据的需求下降，将面临巨大的财务损失风险。
   - 这引发了在不断变化的市场需求中，**Scale.ai** 模式可持续性的疑问。
- **关于中国科技视角的讨论**：针对 **Alex Wang** 对中国科技的看法出现了评论，指出由于其**移民背景**，他的家庭背景可能会使他的观点产生偏差。
   - 对话引用了**文革**这一影响科技行业人士观点的重大时期。
- **虚假账号与虚假信息**：有人对一个可能传播错误信息的**虚假账号**表示担忧，导致对信息源可信度的怀疑。
   - 随后讨论了发布的信息应该归功于**真实身份还是虚假身份**。
- **澄清身份混淆**：一位成员澄清了身份混淆，指出提到的人可能只是与 **Deepseek** 团队中的另一个人同名。
   - 这进一步强调了追踪科技项目相关人员讨论的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1882837818101359001?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>：o1 与 Sonnet 搭配并没有产生比单独使用 o1 更好的结果。使用其他各种模型作为编辑器似乎也没有比 o1 或 R1 的独立评分有所提高。</li><li><a href="https://x.com/dylan522p/status/1883569162100080875?s=46">来自 Dylan Patel (@dylan522p) 的推文</a>：关于 Deepseek V3 和 R1 的讨论归结为：移动曲线意味着你构建了更多并扩展了更多。</li><li><a href="https://x.com/gzilgalvis/status/1883107575010619649?s=46">来自 Gustavs Zilgalvis (@GZilgalvis) 的推文</a>：这并非无害行为 deepseek-r1</li><li><a href="https://fxtwitter.com/wordgrammer/status/1883448109814206892">来自 wordgrammer (@wordgrammer) 的推文</a>：有一次，我遇到了一位 YC 支持的初创公司的创始人。他说他的主要项目是一个待办事项列表应用。但他注意到他的 GPU 集群在非工作时间（当所有任务都完成后）没有被使用……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1333110343966392421)** (4 条消息): 

> `REINFORCE 首字母缩写、编写 RLHF 书籍、Open-Instruct 与 vLLM 的集成、OpenRLHF 框架维护` 


- **REINFORCE 代表了某些含义！**：术语 **REINFORCE** 是 'REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility' 的首字母缩写。
   - 这为其在强化学习中的复杂功能提供了一些清晰度。
- **编写 RLHF 书籍带来的见解**：一位成员幽默地提到，他们在“编写 RLHFbook 过程中真的学到了很多”，强调了写作的过程。
   - *通过写作学习*似乎是最近几位成员的座右铭！
- **关于 Open-Instruct 的 vLLM 集成的疑问**：讨论了 **Open-Instruct** 的潜在内部使用，特别关注其与 **vLLM** 的集成。
   - 如果 **OpenRLHF** 框架无法持续，人们对这种集成的**维护**表示担忧。
- **维护 OpenRLHF 和 vLLM 的集成**：一位成员寻求澄清 **AllenAI** 是否计划为新版本的 **vLLM** 维护 **OpenRLHF** 集成。
   - 他们的担忧源于不想依赖一个可能面临未来**维护**挑战的 OSS 项目。


  

---

### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1333469136327675980)** (4 条消息): 

> `Tulu3 论文分析，Pref Tuning 挑战，对 Tulu4 的期待` 


- **重新审视 Tulu3 的 Off-Policy 数据**：一位成员注意到，尽管有证据表明在数据量相同的情况下 **on-policy** 效果更好，但 **Tulu3** 的偏好数据中仍包含了一些 **off-policy 数据**。
   - *为什么做出这种选择？*
- **偏好微调（Preference Tuning）中持续存在的挑战**：另一位成员表示，**pref tuning** 领域仍有**许多难关需要攻克**，表明该领域仍存在持续的疑虑和复杂性。
   - *仍有巨大的改进空间。*
- **对 Tulu4 发布的热切期待**：展望未来，一位成员表达了对 **Tulu 4** 的渴望，暗示了对技术进步和改进的期待。
   - *这种期待预示着该系列有望延续成功。*


  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 条消息): 

the_real_jrb: 它来了！Qwen2.5-VL。https://qwenlm.github.io/blog/qwen2.5-vl/
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1332743114246459484)** (15 条消息🔥): 

> `DeepSeek R1 发布，市场对 DeepSeek 的反应，AIW 问题变体，John Schulman 的评论，Jay Alammar 的分析` 


- **DeepSeek R1 难以令人完全信服**：根据 [JJitsev](https://x.com/jjitsev/status/1883158738661691878?s=46) 的评论，虽然宣称 DeepSeek R1 在奥数级别的数学和编程问题上能与 o1/o1-preview 媲美，但人们对其在 **泛化能力（generalization）** 以及 SOTA LLM 中存在的 **推理缺陷** 方面的有效性持怀疑态度。
   - *R1 真的好吗？* 是一个反复出现的问题，暗示了对其性能的担忧。
- **市场对 DeepSeek 应用发布的恐慌**：[Zvi 的文章](https://thezvi.substack.com/p/deepseek-panic-at-the-app-store)指出，虽然 DeepSeek 发布了多个版本，但市场反应仅在应用发布后才出现，这描绘了 **市场效率的偏差**。
   - 标普指数和 Nvidia 股票大幅下跌，表明市场反应往往不可预测，且不一定与实际事件直接挂钩。
- **讨论 AIW 问题结构**：讨论强调 AIW 实例是由结构化模板生成的，允许自然的问题变体，且这些变体不应改变 **难度** 或 **可解性**。
   - 这类推理问题的实用性受到质疑，一位成员称其相关性仅相当于统计单词中的字母数量。
- **John Schulman 建立联系**：John Schulman 指出，**r1 思维链（chains of thought）** 与论文和博客中分享的 **o1-preview CoT** 之间存在有趣的相似之处，特别是频繁使用过渡短语进行错误修正，正如他在 [推文](https://x.com/johnschulman2/status/1883221980931142113?s=46) 中所述。
   - 这一评论引发了关于不同推理模型如何可能趋同于相似结果的讨论，突显了社区内的多样化解读。
- **Jay Alammar 强调 DeepSeek 的重要性**：Jay Alammar 的分析讨论了 DeepSeek-R1 如何通过其 **权重开放模型（open weights model）** 以及对类似于 OpenAI o1 的推理模型训练方法的见解，代表了 AI 领域的重大进展。
   - 他的 [初稿文章](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) 反思了模型训练中透明度和可复现性的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1">图解 DeepSeek-R1</a>：推理 LLM 的秘诀</li><li><a href="https://thezvi.substack.com/p/deepseek-panic-at-the-app-store">App Store 里的 DeepSeek 恐慌</a>：DeepSeek 发布了 v3。</li><li><a href="https://x.com/johnschulman2/status/1883221980931142113?s=46">John Schulman (@johnschulman2) 的推文</a>：r1 思维链与论文和博客文章中分享的 o1-preview CoT 之间存在一些有趣的相似之处（例如 https://openai.com/index/learning-to-reason-with-llms）。特别是...</li><li><a href="https://x.com/jjitsev/status/1883158738661691878?s=46">Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：(又一个) 崛起与衰落的故事：宣称 DeepSeek R1 在奥数级数学和编程问题上匹配 o1/o1-preview。它能处理揭示泛化和基础能力的 AIW 问题变体吗...</li><li><a href="https://x.com/JJitsev/status/1883158749785006533">Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：AIW 实例是从定义问题结构的模板生成的。重要的是，我们可以通过不改变结构或难度的修改来引入自然的问题变体...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1332736823709011999)** (4 条消息): 

> `招聘板块上线，频道不当行为` 


- **关于频道不当行为的讨论**：一名成员对频道内某篇帖子的适当性表示担忧，提到为一名患病作者提供模型的紧迫性。
   - *根据反馈，有人建议该内容可能过于具有自我推销性质*。
- **即将推出的招聘板块概念**：一名成员暗示未来可能会推出一个类似 **job board** 的平台。
   - 这引起了大家的兴趣，因为它可能解决社区的就业需求和机会。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1333425817375604798)** (6 条消息): 

> `理解 Deepseek，聊天机器人格式化，社区参与` 


- **社区对 Deepseek 的热议**：一名成员指出，一篇关于 **Deepseek** 的帖子正在一家金融公司内部流传，表明人们对其影响深感兴趣。
   - *这种分享凸显了人们对 Deepseek 功能和应用日益增长的好奇心。*
- **改进聊天机器人格式**：一名成员强调，正尝试使 **chatbot format** 对社区讨论更有用。
   - *这一举措旨在提高用户之间的清晰度和参与度。*
- **积极的社区反馈**：一名成员对一篇帖子表示赞赏，简单地评价道：“**好帖**”。
   - *友好的交流营造了支持性的社区氛围。*
- **欢迎新成员**：一名成员向 Florian 打招呼，强化了社区的欢迎氛围。
   - *这种举动有助于在群组内建立联系。*


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1332800307842908272)** (15 条消息🔥): 

> `中国 AI 新政策，美国产业政策，AI 领域的大国竞争，CHIPS Act，Jones Act 与国防制造` 


- **中国 AI 产业获得巨大助力**：中国宣布了一项新的 AI 政策，包括在未来五年内投入 **1 万亿元人民币**（1370 亿美元）支持其 AI 产业，正如 [@rwang07](https://x.com/rwang07/status/1883210410763121073) 所强调的。
   - 这一举措被描述为可能是 2025 年最重要的中国 AI 政策。
- **美国政府在产业政策上的挣扎**：讨论中提到了**美国政府实施有效产业政策的能力较低**，特别是在共和党领导下。
   - 人们对支持技术移民和新兴科技领域等基本政策的政治意愿持怀疑态度。
- **AI 竞赛引发军工行业关注**：评论者认为，**共和党可能会动员资源**用于 AI，将其纳入大国竞争的叙事中，类似于冷战时期的“导弹差距”。
   - 有观点认为，AI 的军事应用可能比工业进步更容易获得资金。
- **产业政策的历史背景**：**CHIPS Act** 被回顾为加强美国半导体生产和技术的重大两党合作成果。
   - 然而，GPU 出口管制仍然是一个有争议的问题，表明现任政府政策仍存在局限性。
- **Jones Act 带来的挑战**：**Jones Act** 被讨论为导致美国国防制造效率低下的长期监管障碍。
   - 该法律在美国国内航运业制造了一个缺乏竞争的泡沫，对造船商的国际竞争力产生了负面影响。



**提到的链接**：<a href="https://x.com/rwang07/status/1883210410763121073">Ray Wang (@rwang07) 的推文</a>：中国人工智能产业发展行动方案（中国银行支持人工智能产业链发展行动方案）将在未来五年提供 1 万亿元人民币（1370 亿美元）支持其 AI 产业 🇺🇸🇨🇳 这可能是最...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1332440515840446465)** (233 条消息🔥🔥): 

> `DeepSeek R1 进展，Qwen2.5-VL 发布，Operator 功能，Prompt 工程工具，推理模型应用`

- **DeepSeek R1 的训练成本参考**：一位成员澄清，500 万美元的训练成本参考是指 DeepSeek V3，可以在该项目的报告中找到。
   - 提供的一张图片附件被引用作为此信息的确认。
- **Qwen2.5-VL 模型发布**：阿里巴巴宣布推出 Qwen2.5-VL，这是一款能够生成图像并执行智能任务的 multimodal 模型。
   - 该模型声称在多个 benchmarks 上优于 DALL-E 3 和 Stable Diffusion，强调了其视觉理解和定位能力。
- **关于 Operator 功能的见解**：用户讨论了 Operator 在编程环境中的能力，重点强调了其在生成初始代码库方面的有效性。
   - 注意到了在处理复杂网站和视频采样率方面的挑战，突显了改进的必要性。
- **Prompt Engineering 工具和经验**：成员们分享了使用 Braintrust 和 Humanloop 等各种 Prompt Engineering 工具的经验，讨论了它们的可用性和功能。
   - 一位用户认为 Braintrust 的功能表现良好，同时也提到了对 Humanloop 的 UX 和价格透明度的担忧。
- **Reasoning Models 及其应用研究**：参与者探讨了像 R1 这样的 Reasoning Models 的潜在应用，包括改进的编程和 Agentic 能力。
   - 对话中表达了对 Reasoning Models 能够显著增强任务执行（超越传统聊天功能）这一观点的兴趣。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://]">未找到标题</a>：未找到描述</li><li><a href="https://x.com/NousResearch/status/1883912370696704011">来自 Nous Research (@NousResearch) 的推文</a>：最近的 AI 突破挑战了现状，即只有封闭的大型实验室才有能力推动超级智能的前沿。今天我们宣布在 @Solana 上构建的 Nous Psyche - 一个...</li><li><a href="https://x.com/huybery/status/1883775353950519479">来自 Binyuan Hui (@huybery) 的推文</a>：今晚有一些惊喜</li><li><a href="https://x.com/lmarena_ai/status/1882875989610594542">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：❤️‍🔥WebDev Arena 更新：令人兴奋的新条目！- #2: @deepseek_ai DeepSeek-R1 - #4: 新的 Gemini-2.0-Flash-Thinking。DeepSeek-R1 跃升至第 2 位，与 Claude 3.5 Sonnet 的差距仅不到 40 分，展现了强大的能力...</li><li><a href="https://x.com/LiangWenfeng_">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">来自 Qwen (@Alibaba_Qwen) 的推文</a>：🎉 恭喜发财🧧🐍 在迎接农历新年之际，我们激动地宣布发布 Qwen2.5-VL，这是我们最新的旗舰级 vision-language 模型！🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://app.discuna.com/invite/ai_engineer">Discuna</a>：未找到描述</li><li><a href="https://x.com/alecm3/status/1883147247485170072?t=55xwg97roj74RglY2Dil_g&s=19">来自 Alec (@alecm3) 的推文</a>：Deepseek 烂透了</li><li><a href="https://x.com/klazuka/status/1883880742322888903.">来自 Keith Lazuka (@klazuka) 的推文</a>：这是 Operator 在 GitHub 中创建新的 Python Web 应用的视频。效果出奇地好。https://operator.chatgpt.com/v/67978eebb89c81909ed9a584d7fce506 这是代码仓库和...</li><li><a href="https://stackoverflow.com/questions/77628629/is-it-possible-to-use-macos-accessibility-api-features-from-a-cli-or-library">是否可以从 CLI 或库中使用 macOS Accessibility API 功能？</a>：我正在开发一个需要利用 macOS Accessibility API 来读取任何应用程序中选定文本的应用程序。我将通过 FFI 从 Rust 调用 Swift 库。我已经能够获得一个 POC...</li><li><a href="https://x.com/thankscline/status/1882878536450814263?s=46">来自 Cline (@thankscline) 的推文</a>：当大家都在争论 DeepSeek R1 与 o1 的基准测试时，Cline 社区发生了一些迷人的事情：开发者们开始自发地使用：- DeepSeek R1 ($0.55/M) 用于规划阶段...</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>：未找到描述</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>：R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1?r=f2tys&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">图解 DeepSeek-R1</a>：推理型 LLM 的秘诀</li><li><a href="https://x.com/alecm3/status/1883147247485170072?t=55xwg97roj74RglY2">来自 Alec (@alecm3) 的推文</a>：Deepseek 烂透了</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - Qwen 集合</a>：未找到描述</li><li><a href="https://x._philschmid/status/1883055262669349287?s=46">来自 Philipp Schmid (@_philschmid) 的推文</a>：Function Calling 尚未解决 ‼️ 一项新的基准测试显示，LLM 在多步、受限的函数调用方面表现挣扎。ComplexFuncBench 旨在测试复杂的函数调用评估...</li><li><a href="https://x.com/LiangWenfeng_/status/1883953499068887189">来自 Liang Wenfeng 梁文锋 (@LiangWenfeng_) 的推文</a>：Deepseek 2025/25/02 ⏳🐋</li><li><a href="https://steve-yegge.medium.com/the-death-of-the-stubborn-developer-b5e8f78d326b">顽固开发者的死亡</a>：我在五月份写了一篇名为《初级开发者的死亡》的博文。它让人们感到愤怒。我的论点后来得到了...</li><li><a href="https://x.com/rajammanabrolu/status/1883583493290238106?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Prithviraj (Raj) Ammanabrolu (@rajammanabrolu) 的推文</a>：简单来说，不。我一直在回顾我从 2019 年的 GPT-1/2 到 2024 年的 Qwen，使用“可验证”奖励（数学谜题游戏、通过单元测试的 Python 代码）进行 RL 的旧结果...</li><li><a href="https://x.com/alibaba_qwen/status/1883557964759654608?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 Qwen (@Alibaba_Qwen) 的推文</a>：我们正在通过最新的开源模型 Qwen2.5-1M 提升水平！💥 现在支持 100 万 Token 的上下文长度 🔥 以下是新内容：1️⃣ 开源模型：迎接 Qwen2.5-7B-Instruct-1M ...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>：我们复现了 DeepSeek</li>

R1-Zero 在 CountDown 游戏中表现出色，通过 RL（强化学习），3B 基础 LM 自主发展出了自我验证和搜索能力。你可以体验到那种“顿悟时刻”...</li><li><a href="https://x.com/giffmana/status/1883432865293049954">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：你知道，在去年的 ICML（维也纳，24年7月）上，我听了一场关于欧盟在 AI 领域工作及制定计划尝试的演讲。当时在征集专家研究员的帮助/意向。我注册了...</li><li><a href="https://x.com/Yuchenj_UW/status/1883391135441371223">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：@pmarca</li><li><a href="https://x.com/LiangWenfeng_/status/1883918900741763293">来自 梁文锋 (@LiangWenfeng_) 的推文</a>：🚨DeepSeek 刚刚发布了另一个开源 AI 模型 Janus-Pro-7B。它是多模态的（可以生成图像），并在 GenEval 和 DPG-Bench 基准测试中击败了 OpenAI 的 DALL-E 3 和 Stable Diffusion...</li><li><a href="https://x.com/hamptonism/status/1883147826571706735">来自 ₕₐₘₚₜₒₙ — e/acc (@hamptonism) 的推文</a>：中国银行计划在 AI 产业投资 1 万亿元人民币。</li><li><a href="https://x.com/LiangWenfeng_/status/1883874025350508861">来自 梁文锋 (@LiangWenfeng_) 的推文</a>：DeepSeek 即将发布新品</li><li><a href="https://x.com/rwang07/status/1883210410763121073?s=46">来自 Ray Wang (@rwang07) 的推文</a>：中国新的 AI 产业发展行动计划（中国银行支持人工智能产业链发展行动方案）将在未来五年提供 1 万亿元人民币（1370 亿美元）支持其 AI 产业 🇺🇸🇨🇳 这可能是最...</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF">bartowski/Qwen2.5-7B-Instruct-1M-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/teortaxesTex/status/1883605616742351013">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：当一个 HFT（高频交易）大佬说 GRPO 是基于夏普比率（Sharpe ratio，用于衡量投资中的风险调整后收益）时，那种感觉……毕竟大鳄们都是量化交易员。如果你足够疯狂，一切都是 ROI 问题...</li><li><a href="https://x.com/vllm_project/status/1883966341557936514">来自 vLLM (@vllm_project) 的推文</a>：🚀 随着今天 v0.7.0 的发布，我们很高兴地宣布 vLLM V1 的 Alpha 版本：一个重大的架构升级，速度提升 1.7 倍！代码整洁、优化的执行循环、零开销的前缀缓存...</li><li><a href="https://x.com/teortaxesTex/status/1883926389306671376">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：这是假的，必须举报到它消失。伙计，美国人真的不太能接受竞争。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.promptlayer.com/">PromptLayer - 最整洁的 Prompt 工程方式。用于 Prompt 管理、Prompt 评估和 LLM 可观测性的平台</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1882893748742598669?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：我们使用 5k 条 DeepSeek R1 蒸馏的 CoT 数据重新训练了 Hermes。我可以确认几件事：1. 你可以拥有通用 + 推理模式，我们使用静态系统提示词标记了来自 R1 的所有长 CoT 样本...</li><li><a href="https://www.youtube.com/watch?v=X5adgxV0gBE">DeepSeek R1 - 震惊整个行业的中国 AI “侧边项目”！</a>：加入我的通讯以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅：https://www.youtube.com/@matthew_berman 👉🏻 Twitter：https:/...</li><li><a href="https://stratechery.com/2025/deepseek-faq/">DeepSeek 常见问题解答</a>：DeepSeek 彻底颠覆了人们对 AI 以及与中国竞争的预期。它是什么，为什么它很重要？</li><li><a href="https://www.youtube.com/watch?v=jrf76uNs77k&t=868s">推理蒸馏的非凡有效性：使用 DeepSeek R1 击败 OpenAI o1</a>：https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation 我们训练了 Bespoke-Stratos-32B，这是我们的推理模型...</li><li><a href="https://github.com/madsys-dev/deepseekv2-profile/blob/924174cb5dc11fad24bdaad3fd820ebf87506368/workspace/blog/optimizing-mla.md">deepseekv2-profile/workspace/blog/optimizing-mla.md 位于 924174cb5dc11fad24bdaad3fd820ebf87506368 · madsys-dev/deepseekv2-profile</a>：通过在 GitHub 上创建账户来为 madsys-dev/deepseekv2-profile 的开发做出贡献。</li><li><a href="https://youtu.be/bJzj5lTiqe0?si=n73aW2Zm8U3qIjKO">DeepSeek R1：中国的开源 AI 模型如何以 3% 的成本击败 OpenAI</a>：DeepSeek-R1：颠覆 OpenAI 领导地位的中国开源 AI。在本集中，Sam 和 Matt 讨论了 DeepSeek R1 模型的最新突破...</li><li><a href="https://youtu.be/HM92mmG6YTs?feature=shared">DeepSeek R1 vs o1：AI 解释专家自主性（一种更好的 MoE）</a>：OpenAI o1（旧的、专有的）与 Deep... 的性能对比

DeepSeek R1（新款，开源）。两个 LLM 的任务是解释关于 Autonomy of Expert 的新 AI 论文...</li><li><a href="https://x.com/pmarca/status/1882903903777558677">来自 Marc Andreessen 🇺🇸 (@pmarca) 的推文</a>：绝对不是。引用 TFTC (@TFTC21) Sam Altman 的话：推进 AI 可能需要“改变社会契约”。“整个社会结构都将面临辩论和重构。”</li><li><a href="https://www.bankofchina.com/aboutboc/bi1/202501/t20250123_25254674.html">1万亿元！提供专项综合金融支持 助力人工智能产业链发展</a>：未找到描述</li><li><a href="https://github.com/glut23/webvtt-py">GitHub - glut23/webvtt-py: Read, write, convert and segment WebVTT caption files in Python.</a>：在 Python 中读取、写入、转换和分割 WebVTT 字幕文件。- glut23/webvtt-py</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">Mixture-of-Experts (MoE) LLMs</a>：从底层开始理解 DeepSeek、Grok 和 Mixtral 等模型...</li><li><a href="https://youtubetranscriptoptimizer.com/blog/05_the_short_case_for_nvda">做空 Nvidia 股票的理由</a>：Nvidia 很难达到目前市场过高预期的所有原因。</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://www.bankofchina.com/aboutboc/bi1/202501/t20250123_25254674.h">中国银行全球门户网站-提示信息</a>：未找到描述</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_sou">Mixture-of-Experts (MoE) LLMs</a>：从底层开始理解 DeepSeek、Grok 和 Mixtral 等模型...</li><li><a href="https://www.stepfun.com">阶跃星辰</a>：未找到描述</li><li><a href="https://buttondown.com/ainews/archive/ainews-tinyzero-reproduce-deepseek-r1-zero-for-30/">[AINews] TinyZero: Reproduce DeepSeek R1-Zero for $30</a>：RL is all you need。2025/1/23-2025/1/24 的 AI 新闻。我们为您检查了 7 个 subreddits、433 个 Twitter 和 34 个 Discord（225 个频道，3926 条消息）....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 新播客！https://x.com/latentspacepod/status/1883354909367787565
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1332454948553887831)** (193 条消息🔥🔥): 

> `Model Context Protocol (MCP), MCP 工具集成, 转录与文档, Obsidian 集成, 服务器功能与实现` 


- **对 MCP 及其潜力的兴奋**：成员们对 **Model Context Protocol (MCP)** 表达了极大的热情，将其描述为跨应用和工具集成 AI 能力的关键点。
   - 讨论强调了 MCP 如何作为各种应用的中心枢纽，鼓励探索其在现实场景中的能力。
- **与现有工具和库的集成**：参与者讨论了 MCP 与 **Cursor** 和 **Cline** 等各种工具的集成，表现出对这些工具如何增强功能和简化工作流程的兴趣。
   - 还提到了将 MCP 与 **Obsidian** 连接用于文档和转录的潜力，强调了社区的协作性质。
- **探索 MCP 服务器的不同语言**：使用各种编程语言创建 MCP 服务器的灵活性是一个关键点，建议使用 **Go**、**Rust** 甚至 **assembly** 以获得最佳性能。
   - 成员们指出，这种语言独立性允许开发人员专注于其特定实现的性能和安全需求。
- **关于 MCP 的未来讨论与协作**：提到了举办 **MCP party** 以讨论经验教训和心得的计划，反映了知识共享的协作方式。
   - 鼓励成员参与并为正在进行的开发做出贡献，表明了在推进 MCP 工具和能力方面的积极参与。
- **文档与教程**：提到了为有兴趣实现 MCP 的用户提供的全面文档，包括关于最佳实践的详细章节。
   - 参与者表示有兴趣查看最新的 README 和教程，以更好地了解如何在项目中有效利用 MCP。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cs16.samke.me/">cs16.css</a>: 基于 Counter Strike 1.6 UI 的 CSS 库。</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability-negotiation">Architecture</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/architecture/#capability">Architecture</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/cancellation/">Cancellation</a>:           ℹ️                  协议修订版本：2024-11-05      Model Context Protocol (MCP) 支持通过通知消息可选地取消进行中的请求。任何一方都可以发送...</li><li><a href="https://github.com/tumf/mcp-shell-server">GitHub - tumf/mcp-shell-server</a>: 通过在 GitHub 上创建账户，为 tumf/mcp-shell-server 的开发做出贡献。</li><li><a href="https://github.com/rusiaaman/wcgw">GitHub - rusiaaman/wcgw: Shell and coding agent on claude desktop app</a>: Claude 桌面应用上的 Shell 和编码 Agent。通过在 GitHub 上创建账户，为 rusiaaman/wcgw 的开发做出贡献。</li><li><a href="https://github.com/MarkusPfundstein/mcp-obsidian">GitHub - MarkusPfundstein/mcp-obsidian: MCP server that interacts with Obsidian via the Obsidian rest API community plugin</a>: 通过 Obsidian REST API 社区插件与 Obsidian 交互的 MCP server - MarkusPfundstein/mcp-obsidian</li><li><a href="https://github.com/go-go-golems">GO GO GOLEMS!</a>: GO GO GOLEMS 构建 GO GO GADGETS。GO GO GOLEMS! 拥有 34 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: 未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers。通过在 GitHub 上创建账户，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/rusiaaman/wcgw/blob/fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7/src/wcgw/client/mcp_server/server.py#L129-L138">wcgw/src/wcgw/client/mcp_server/server.py at fbe8c5c3cca4f7a149f8c099c63696d9ede7f9e7 · rusiaaman/wcgw</a>: Claude 桌面应用上的 Shell 和编码 Agent。通过在 GitHub 上创建账户，为 rusiaaman/wcgw 的开发做出贡献。</li><li><a href="https://github.com/go-go-golems/go-go-mcp">GitHub - go-go-golems/go-go-mcp: Anthropic MCP go implementation</a>: Anthropic MCP 的 Go 语言实现。通过在 GitHub 上创建账户，为 go-go-golems/go-go-mcp 的开发做出贡献。</li><li><a href="https://github.com/calclavia/mcp-obsidian">GitHub - smithery-ai/mcp-obsidian: A connector for Claude Desktop to read and search an Obsidian vault.</a>: 一个用于 Claude Desktop 读取和搜索 Obsidian 库的连接器。 - smithery-ai/mcp-obsidian
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1332461827061321728)** (97 条消息🔥🔥): 

> `Layer Convergence Bias, Causally Regularized Tokenization, DeepSeek Model Discussion, R1 Training Costs, GRPO Implementation Challenges` 


- **层收敛偏差 (Layer Convergence Bias) 观察**：研究表明，**Deep Neural Networks** 中的浅层比深层收敛更快，这被称为 **Layer Convergence Bias**。这一现象归因于浅层中更平坦的局部极小值，从而导致更稳定的梯度。
   - 更多详情请查看 **2023 年 2 月 1 日**发表的 [ICLR 2023 论文](https://openreview.net/forum?id=wlMDF1jQF86)。
- **因果正则化分词 (Causally Regularized Tokenization) 见解**：**Armen Agha 及其团队** 的最新工作揭示，仅针对重构优化的图像 Tokenizers 会阻碍**下游自回归模型性能 (downstream autoregressive model performance)**。他们的新方法 **Causally Regularized Tokenization** 在效率和质量上都有显著提升。
   - 更多细节可以在[已发表的论文](https://arxiv.org/pdf/2412.16326)中找到，该论文对比了 **LlamaGen-3B** 的性能。
- **围绕 DeepSeek 模型的疑问**：讨论中对 **DeepSeek** 开发出比 **NVIDIA** 便宜得多且模型性能更有效的芯片的说法表示怀疑。参与者指出，在模型构建材料或算力方面缺乏透明度。
   - 讨论中对推测的价格估算以及缺乏必要的开源信息表示了担忧。
- **估算 R1 训练成本**：参与者建议，根据数据集大小和 Token 数量来估算 **R1 训练成本** 可以得到一个粗略的近似值，并询问 R1 论文中是否提到了 800k 的样本量。大家普遍认为，成本可能明显低于之前的迭代版本。
   - 计算表明推理成本可能约为 **$1.6M**，讨论暗示 **R1** 可能比 **V3** 更具成本效益。
- **GRPO 实现差距**：尽管有多个 GitHub 仓库声称实现了 **GRPO**，但似乎缺乏实现类似 **R1** 运行效果的实际应用。参与者对 **TinyZero** 和 **SimpleRL** 在复现运行中未能有效利用 GRPO 表示失望。
   - 目前看来主要还是在使用 **PPO**，这表明在充分探索 GRPO 潜力方面存在差距。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ArmenAgha/status/1882897021">mhase (@mloge) 的推文</a>：┣¨ｽﾄｴﾌｽｷｰ</li><li><a href="https://openreview.net/forum?id=wlMDF1jQF86">哪一层学习得更快？对神经网络的系统性探索...</a>：我们通过实验证明，神经网络中浅层比深层收敛更快，并为这一发现提供了理论依据和实际价值。</li><li><a href="https://x.com/ArmenAgha/status/1882897021667090797">Armen Aghajanyan (@ArmenAgha) 的推文</a>：正式宣布，这是我在 FAIR 任职期间参与的最后一篇论文！我们的论文揭示了仅针对重构优化的图像分词器会损害下游自回归模型的性能，挑战了...</li><li><a href="https://x.com/TheXeophon/status/1883933054366015545">Xeophon (@TheXeophon) 的推文</a>：搞什么鬼</li><li><a href="https://en.wikipedia.org/wiki/Taylor_Swift%E2%80%93Ticketmaster_controversy">Taylor Swift–Ticketmaster 争议 - 维基百科</a>：未找到描述</li><li><a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: DeepSeek R1-Zero 的简洁、易用的复现</a>：DeepSeek R1-Zero 的简洁、易用的复现 - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/hkust-nlp/simpleRL-reason">GitHub - hkust-nlp/simpleRL-reason: 这是 DeepSeek-R1-Zero 和 DeepSeek-R1 在有限数据的小模型上的训练复现</a>：这是 DeepSeek-R1-Zero 和 DeepSeek-R1 在有限数据的小模型上的训练复现 - hkust-nlp/simpleRL-reason</li><li><a href="https://en.m.wikipedia.org/wiki/List_of_common_misconceptions">常见误解列表 - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1332471378347167855)** (311 条消息🔥🔥): 

> `GRPO Implementation Details, AlphaZero Evolution, Empowerment in AI, Reinforcement Learning Challenges, Experience Replay in RL`

- **理解 GRPO Group Size 和 Batch Size**：讨论围绕 GRPO 的实现展开，澄清了 1024 的 batch size 可能包含多个样本，每个样本是一个 64 的 group。参与者指出，考虑 sequence length 如何影响训练过程和 gradient calculation 至关重要。
   - 大家达成共识，对于 GRPO 来说，使用 replay buffers 可能并非必要，因为该算法可以通过聚合来自 groups 的 gradients 来运行，而无需保留早期的样本。
- **AlphaZero 的开发过程**：参与者反思了 AlphaZero 的演进，指出它简化了之前的方法，并吸取了过去版本的经验。他们强调了所面临的重大工程挑战，以及在实践中不直接采用 AlphaZero 方法论的理由。
   - Bearcat9705 指出，随后的论文在之前迭代的基础上进行了增量改进，同时也关注了方法的简洁程度。
- **AI 中的好奇心与 Empowerment**：讨论了 AI 中 empowerment 的概念，重点关注其与内在动机（intrinsic motivation）以及最大化未来选择之间的关系。Synquid 分享了个人研究工作的见解，强调了其理论基础和潜在应用。
   - 参与者对 curiosity-driven models 与当前语言学习方法之间的关系表示感兴趣，并建议重新探索这些概念。
- **实施 Reinforcement Learning 的挑战**：对话涉及在 Reinforcement Learning 中奖励模糊结果与明确任务相比的难度，特别是在训练 LLM 的背景下。共识是，与传统方法相比，集中的 reward structure 可以产生更好的学习效果。
   - Fessus 强调，简单的成功或失败 token 就可以有效地引导学习，而无需传统 Reinforcement Learning 方法的复杂性。
- **RL 中的 Experience Replay 策略**：对话强调了传统 experience replay 在现代 Reinforcement Learning 实践中的关联性正在减弱。参与者一致认为，保留所有以前的样本已不再常见，并质疑在当前实现中维持 replay buffer 的必要性。
   - 他们辩论了在没有 replay buffer 的情况下收集 batches 的优势，承认即使是用于聚合 traces 的小规模收集也可能不会带来显著收益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1509.08731">Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning</a>: 互信息是一个核心统计量，在机器学习的所有领域都有应用，无论是训练多模态数据的密度模型，还是在最大化...</li><li><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: 由于数学推理的复杂性和结构化特性，它对语言模型构成了重大挑战。在本文中，我们介绍了 DeepSeekMath 7B，它继续对 DeepSeek-Co 进行预训练...</li><li><a href="https://philippe-eecs.github.io/vitok/">ViTok</a>: 扩展视觉分词器（Visual Tokenizers）用于重建与生成的经验学习</li><li><a href="https://arxiv.org/abs/2501.13926">Can We Generate Images with CoT? Let&#39;s Verify and Reinforce Image Generation Step by Step</a>: 思维链（CoT）推理已在大型模型中被广泛探索，以应对复杂的理解任务。然而，这类策略是否可以应用于... 仍然是一个开放性问题。</li><li><a href="https://arxiv.org/abs/2405.17399">Transformers Can Do Arithmetic with the Right Embeddings</a>: Transformer 在算术任务上的糟糕表现很大程度上似乎源于它们无法追踪大跨度数字中每个数字的准确位置。我们修复了这...</li><li><a href="https://arxiv.org/abs/2501.11651">Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling</a>: 大语言模型（LLMs）在复杂推理任务中展示了卓越的能力。然而，现有方法主要依赖模仿学习，难以实现有效的测试时...</li><li><a href="https://arxiv.org/abs/2410.14606">Streaming Deep Reinforcement Learning Finally Works</a>: 自然智能将经验处理为连续流，实时进行时刻不停的感知、行动和学习。流式学习是经典强化学习的运作模式...</li><li><a href="https://arxiv.org/abs/2501.11425">Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training</a>: 大语言模型（LLMs）Agent 在处理交互式环境中的复杂任务时变得越来越关键。现有工作主要侧重于通过行为克隆来增强性能...</li><li><a href="https://x.com/its_dibya/status/1883595705736163727">Tweet from Dibya Ghosh (@its_dibya)</a>: 随着 R1 的发布，很多人都在问“为什么我们两年前没发现这个？”好吧……2年前，我花了6个月时间专门研究这个（针对 math+gsm8k 的 PG / PPO），但我的结果...</li><li><a href="https://arxiv.org/abs/2301.07969">Fast Inference in Denoising Diffusion Models via MMD Finetuning</a>: 去噪扩散模型（DDMs）已成为从复杂数据分布中生成高质量样本的流行工具。这些模型能够捕捉复杂的模式和结构...</li><li><a href="https://x.com/RamanujanVivek/status/1882882551670555095">Tweet from Vivek Ramanujan (@RamanujanVivek)</a>: 很高兴（迟到地）分享我们最近的工作，介绍了因果正则化分词（Causally Regularized Tokenization）📺，以 0.5 倍的每张图像 Token 数量（256 vs 576）和 0.25 倍的... 匹配了 LlamaGen-3B 的生成性能。</li><li><a href="https://x.com/leloykun/status/1883561892926677029">Tweet from leloy! (@leloykun)</a>: （线性）注意力机制作为测试时回归。到目前为止，你可能已经听说过线性注意力、上下文学习、测试时缩放等……在这里，我将讨论：1. 统一的...</li><li><a href="https://en.wikipedia.org/wiki/Empowerment_(artificial_intelligence)">Empowerment (artificial intelligence) - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/TencentARC/SEED-Voken">GitHub - TencentARC/SEED-Voken: SEED-Voken: A Series of Powerful Visual Tokenizers</a>: SEED-Voken：一系列强大的视觉分词器 - TencentARC/SEED-Voken</li><li><a href="https://x.com/bycloudai/status/1880106360731496661">Tweet from bycloud (@bycloudai)</a>: 终于有人做到了：测试时计算 + 扩散模型。这确实是一个非常有趣的组合 🧵</li><li><a href="https://arxiv.org/abs/2501.09732">Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps</a>: 生成模型在各个领域产生了重大影响，很大程度上归功于它们通过增加数据、计算资源和模型规模在训练期间进行缩放的能力，这种现象...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1333412563219906612)** (2 messages): 

> `Chinchilla library, LLM scaling laws, 20-tokens-per-parameter heuristic` 


- **Chinchilla 库的新 LLM 分析**：一位成员在其 [Chinchilla library](https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb) 中添加了对 LLM scaling law 的分析，重点介绍了其用于 scaling law 研究的工具包。
   - 他们指出一个重要发现：在评估时，“**每个参数 20 个 token**” (**20-tokens-per-parameter**) 的启发式方法的效果几乎与完全优化的 Chinchilla 模型一样好。
- **关于 20-Token 启发式方法的有趣发现**：该成员建议，20-token 启发式方法的有效性并非仅仅因为数字本身，而是因为随着计算量 (compute) 的增加，该比例处于平坦极小值 (flat minima)。
   - 他们在更高计算量下观察到了这种平坦性的视觉确认，说明了 scaling law 中的一种基本行为。
- **质疑缩放效应的本质**：另一位成员推测观察到的现象是否仅仅是由于缩放引起的，并提出“*如果放大得足够大，每条平滑曲线都是平坦的*”。
   - 这一询问开启了关于 scaling laws 内在本质及其在分析中影响的讨论。



**Link mentioned**: <a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/llm/main.ipynb">chinchilla/examples/llm/main.ipynb at master · kyo-takano/chinchilla</a>: 一个用于 scaling law 研究的工具包 ⚖。通过在 GitHub 上创建一个账户来为 kyo-takano/chinchilla 的开发做出贡献。

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1332540720120070207)** (3 messages): 

> `Verified Reasoning in Training, Mechanisms of Model Learning, Interpretability in Fine-Tuning, Insights from Model Weights` 


- **经过验证的推理是新的趋势 (meta)**：*如果训练期间的经过验证的推理 (verified reasoning) 是新的趋势*，成员们讨论了机械可解释性研究者 (mechanistic interpreters) 在分析中应该优先考虑什么以进行适应。
   - 对话暗示了重点可能会转向理解模型的推理能力。
- **理解微调过程中的 LLMs**：一位成员强调了*理解 LLMs 在微调 (fine-tuning) 期间如何学习以及学习了什么*的重要性，特别是关于输入-输出对。
   - 这可能会阐明模型权重 (model weights) 和表示中捕捉到的与推理能力相关的因素。
- **学习机制缺乏可解释性**：成员们达成共识，即*对于模型在针对特定目标进行微调时如何学习，目前还缺乏可解释性 (interpretability)*。
   - 他们不仅关注整体性能，还关注影响微调成功和泛化能力的因素。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1333541169048850454)** (4 messages): 

> `scbench, zeroSCROLLS, longbench` 


- **评估 scbench 的集成挑战**：一位成员指出集成 **scbench** 会很棘手，因为它需要**多轮 (multi-turn)** 能力，建议需要进一步调查。
   - 这表明实现过程中的复杂性可能会影响未来的开发时间表。
- **对 zeroSCROLLS 的兴趣**：另一位成员表达了对探索 **zeroSCROLLS** 的兴奋，表明该选项势头良好。
   - 他们似乎对其潜在优势持乐观态度，尽管尚未讨论具体细节。
- **添加 longbench**：一位成员确认他们也添加了 **longbench**，表明功能扩展取得了进展。
   - 这一添加可以补充现有工具，并在未来提供更广泛的功能集。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1333276412966080542)** (2 messages): 

> `Multimodal Channel Guidelines, Community Project Collaboration` 


- **多模态频道误解**：一位成员指出某些帖子不属于多模态频道，强调其与频道目的无关。
   - 建议采用另一种方法：*如果你在寻找协作，请创建一个社区项目*。
- **细致的多模态模型讨论**：另一位成员承认了这种混淆，同意删除帖子，同时指出该模型具有细致的多模态能力。
   - 他们表示该话题*可能需要在这里以外的地方发布*，表明需要更清晰的社区指南。


  

---

### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1333509080513253427)** (1 条消息): 

> `System prompts 定制，优化 Bolt 的行为` 


- **定制你的 System Prompt**：你现在可以为每个项目或全局设置 **system prompt**，从而在 [Bolt](https://x.com/boltdotnew/status/1883949779572646008) 中获得量身定制的体验。
   - 这一功能是用户强烈要求的，它允许你包含你**最喜欢的库 (libraries)** 和技术，确保 Bolt 按照你的工作流偏好运行。
- **分享使用技巧**：鼓励用户分享关于如何有效使用新的 system prompt 定制功能的最佳技巧。
   - *你将如何优化你的 Bolt 体验？* 参与下方的讨论吧！



**提到的链接**：<a href="https://x.com/boltdotnew/status/1883949779572646008">来自 bolt.new (@boltdotnew) 的推文</a>：你现在可以按项目或全局设置 system prompt！💡 将你最喜欢的库和技术放在那里，让 Bolt 始终使用它们。这个备受期待的功能允许你优化 Bolt 的行为...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1332650967874670663)** (7 条消息): 

> `项目结构化挑战，组件拆分策略，利用指南提升稳定性，从过去的项目中学习，跟踪项目变更` 


- **项目结构化挑战**：成员们讨论了在使用 Bolt 的项目中，结构化的 prompt 和规划可能会抑制创造力，并强调了灵活性的必要性。
   - 一位成员指出，试图预先消除所有问题会导致在没有明确指南的情况下陷入不断重新开始的循环。
- **组件拆分策略**：一位成员分享了在拆分复杂组件时遇到的困难，Context 限制在拆分过程中导致了问题。
   - 他们建议采用系统化的方法，包括详细的代码审查和记录在 NEXTSTEPS.md 文件中的结构化迁移步骤。
- **利用指南提升稳定性**：严格遵守 GUIDELINES.md 有助于稳定项目开发，确保组件按顺序且系统地构建。
   - 通过 GUIDELINES 和 NEXT STEPS 文档，Context Window（上下文窗口）得到了有效管理，避免遗忘关键信息。
- **从过去的项目中学习**：成员们反思了他们的学习过程，识别了早期项目的陷阱，例如在没有明确指南的情况下过早引入 Supabase。
   - 他们强调了定义基础设计系统的重要性，以防止不一致并促进项目更顺畅地进行。
- **跟踪项目变更**：讨论了一个详细的跟踪系统，包括日志和 Changelog，用于监控项目进展和可能的版本回退。
   - 一位成员分享了一个项目结构的链接，该结构侧重于项目管理的关键目录，尽管部署版本存在一些局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://heroic-bombolone-70b361.netlify.app">Vite + React + TS</a>：未找到描述</li><li><a href="https://x.com/KevinNaughtonJr/status/1882833510957985819">来自 Kevin Naughton Jr. (@KevinNaughtonJr) 的推文</a>：软件工程可能是唯一一种你可以被困在一项任务中数天/数周/数月，而没有人会眨一下眼、质疑你的能力或对你感到不满的职业。
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1332440177074896978)** (304 条消息🔥🔥): 

> `Bolt 中的错误处理、计费与 Token 限制、实现用户角色、使用 Netlify 部署、将 GitHub 连接到 Bolt` 


- **Bolt 中的错误处理**：用户报告了 Bolt 频繁出现的错误和问题，包括速率限制（rate limits）和网络错误，在消耗大量 Token 后感到沮丧。
   - 许多人转而寻求迁移问题的帮助，或寻求专业协助来解决他们的问题。
- **计费与 Token 消耗**：用户对使用 Bolt 时的高 Token 消耗表示担忧，一些用户声称在 Prompt 上花费了数百万 Token 却进展甚微。
   - 用户讨论了遇到问题后退款的可能性，并强调了成本与达成成果之间的差距。
- **实现用户角色**：一位用户成功创建了一个具有多个登录角色的应用，包括超级管理员（super admin）和管理员（admin），克服了 Supabase 策略的复杂性。
   - 实施过程被认为具有挑战性，因为策略导致了递归问题，但最终实现了一个功能完备的系统。
- **使用 Netlify 部署**：用户询问如何通过 Netlify 将他们的 Bolt 项目连接到自定义域名，并明确了需要重新部署才能使更新生效。
   - 强调了在 Bolt 中所做的更改不会自动反映在 Netlify 上。
- **将 GitHub 连接到 Bolt**：一位用户寻求关于将现有 GitHub 仓库导入 Bolt 的帮助，但在访问私有仓库时遇到了权限问题。
   - 目前，用户无法在 Bolt 中访问私有仓库（private repos），这是一个正在解决的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/364486390102097930/1332441767861157969">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://diji.art">Diji.art - Digital Design Marketplace</a>: 在高品质服装上创建并销售独特的设计。加入我们的创作者和时尚爱好者社区。</li><li><a href="https://diji.art/designs">Diji.art - Digital Design Marketplace</a>: 在高品质服装上创建并销售独特的设计。加入我们的创作者和时尚爱好者社区。</li><li><a href="https://www.anthropic.com/pricing#anthropic-api">Pricing</a>: Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://discordscrapper.netlify.app/">Vite + React + TS</a>: 未找到描述</li><li><a href="https://repocloud.io/boltdiy">RepoCloud | Bolt.diy: Choose Your AI Model</a>: 探索 Bolt.diy，这是选择你喜欢的 AI 模型的终极分支。使用 OpenAI 和 Anthropic 等顶级 LLM 定制你的编程体验！</li><li><a href="https://thinktank.ottomator.ai/">oTTomator Community</a>: 创新者和专家聚集地，共同推动 AI 驱动自动化的未来</li><li><a href="https://www.youtube.com/watch?v=jkfVvWndbeE">Bolt.new Developer&#39;s Guide to Effortless API Integration and Stop All CORS Errors</a>: 正在为 API 集成而苦恼，还是在解决那些讨厌的 CORS 错误？观看我在这份终极 Bolt 开发者指南中分解这一切，实现轻松的 API 集成...</li><li><a href="https://docs.github.com/rest/git/blobs#create-a-blob">REST API endpoints for Git blobs - GitHub Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1332473138146967632)** (233 条消息🔥🔥): 

> `MCP 客户端问题、服务器配置、语音聊天集成、开源工具、Kubernetes 集成`

- **MCP 客户端的挑战**：用户讨论了 MCP 客户端的各种问题，特别是无法在不重启客户端的情况下动态更新工具。语音聊天功能的集成对开发者来说仍然是一个重大的痛点。
   - 许多用户表示需要更清晰的文档以及这些工具中更好的集成能力。
- **服务器配置关注点**：关于 MCP 服务器配置设置的讨论一直在进行，包括 `disabled` 和 `autoApprove` 选项。用户注意到不同设置带来的复杂性以及使用组织账户的影响。
   - 强调了对允许更好服务器管理的功能的需求，以及在不依赖专有 API 的情况下工作的能力。
- **多设备使用的工具集成**：交流了关于链式调用工具以创建一个主服务器来处理跨多个设备的功能的想法。参与者强调了中央控制器与单个客户端之间有效通信以管理服务器更新的需求。
   - 人们对如何利用 Kubernetes 高效运行 MCP 服务器表现出特别的兴趣。
- **开源与社区努力**：讨论强调了开源项目在推动社区驱动开发中的作用，特别是在 MCP 工具链方面。用户鼓励贡献并使工具准备好被更广泛地采用，尽管目前还存在一些不足。
   - 对话涉及了各种开源客户端，以及拥有公开源代码在确保透明度和协作方面的优势。
- **用户对 API 管理的看法**：参与者分享了在服务器 API 管理方面的经验，特别是他们在请求超时和处理各种 API 配置方面面临的限制。用户寻求明确 MCP 工具最终是否会与更多主流 API 集成以改进功能。
   - 还表达了对云服务相比替代方案额外成本的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://api.systemprompt.io`平衡">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/magic-gif-26166638">Magic GIF - Magic - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://glama.ai/mcp/servers/2au072rrbc">mcp-jdbc</a>: 用于访问任何可通过 JDBC 访问的数据库（如 Postgres, Oracle, mysql, mariadb, sqlite 等）的 MCP。</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client/blob/master/README.md">multimodal-mcp-client/README.md（位于 master 分支）· Ejb503/multimodal-mcp-client</a>: 一个用于语音驱动的 Agent 工作流的多模态 MCP 客户端 - Ejb503/multimodal-mcp-client</li><li><a href="https://boards.greenhouse.io/anthropic/jobs/4495047008">软件工程师，Model Context Protocol</a>: 英国伦敦</li><li><a href="https://github.com/cookiecad/mcp-runner">GitHub - cookiecad/mcp-runner: 一个用于运行具有进程复用能力的 MCP (Model Context Protocol) 服务器的 TypeScript SDK</a>: 一个用于运行具有进程复用能力的 MCP (Model Context Protocol) 服务器的 TypeScript SDK - cookiecad/mcp-runner</li><li><a href="https://youtu.be/hYCL8tA-8Nk?si=4B8Gd8NmJstLwV6V">MCP Gmail 扩展，通过语音 Agent 控制您的收件箱</a>: 使用 SystemPrompt MCP Gmail 体验电子邮件管理的未来——自然语言语音命令与智能邮件处理的结合。此演示展示了我们的...</li><li><a href="https://github.com/Mintplex-Labs/anything-llm/issues/2883">[FEAT]: Model Context Protocol (MCP) 集成 · Issue #2883 · Mintplex-Labs/anything-llm</a>: 您希望看到什么？描述：请求将 Model Context Protocol (MCP) 支持集成到 AnythingLLM 中，以增强不同平台间上下文处理的互操作性和标准化...</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client">GitHub - Ejb503/multimodal-mcp-client: 一个用于语音驱动的 Agent 工作流的多模态 MCP 客户端</a>: 一个用于语音驱动的 Agent 工作流的多模态 MCP 客户端 - Ejb503/multimodal-mcp-client</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/main/src/everything/sse.ts">servers/src/everything/sse.ts（位于 main 分支）· modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/typescript-sdk/tree/main?tab=readme-ov-file#http-with-sse">GitHub - modelcontextprotocol/typescript-sdk: 官方的 Model Context Protocol 服务器和客户端 Typescript SDK</a>: 官方的 Model Context Protocol 服务器和客户端 Typescript SDK - modelcontextprotocol/typescript-sdk</li><li><a href="https://github.com/Ejb503/multimodal-mcp-client/blob/master/proxy/src/handlers/mcpHandlers.ts#L234-L237">multimodal-mcp-client/proxy/src/handlers/mcpHandlers.ts（位于 master 分支）· Ejb503/multimodal-mcp-client</a>: 一个用于语音驱动的 Agent 工作流的多模态 MCP 客户端 - Ejb503/multimodal-mcp-client</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-gmail">systemprompt-mcp-gmail</a>: 一个专门的 Model Context Protocol (MCP) 服务器，使您能够从 Gmail 帐户搜索、阅读、删除和发送电子邮件，并利用 AI Agent 协助每项操作。最新版本：...</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-core">systemprompt-mcp-core</a>: 一个专门的 Model Context Protocol (MCP) 服务器，与 systemprompt.io 集成以提供强大的 Prompt 管理功能。该服务器支持无缝创建、管理和版本控制...</li><li><a href="https://www.npmjs.com/package/systemprompt-mcp-notion">systemprompt-mcp-notion</a>: 一个专门的 Model Context Protocol (MCP) 服务器，将 Notion 集成到您的 AI 工作流中。该服务器通过 MCP 实现对 Notion 的无缝访问，允许 AI Agent 与页面、数据库进行交互...
</li>
</ul>

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1332458145414381720)** (10 条消息🔥): 

> `MCP Variance Log Tool, KoboldCPP-MCP Server, Notmuch Email Integration, MCP Inception Server, Shopify MCP Server` 


- **MCP Variance Log Tool 发布**：一款受 **Titans Surprise 机制** 启发的工具，可将低概率交互记录到 [SQLite 数据库](https://github.com/truaxki/mcp-variance-log) 中，用于用户数据收集和个性化。
   - 该工具旨在通过捕捉异常对话事件来增强长期记忆能力。
- **用于 AI 通信的 KoboldCPP-MCP Server**：分享了一个专为 **AI 与 AI 通信** 设计的服务器，配合 KoboldCPP 使用，方便与 Claude 及 [GitHub](https://github.com/PhialsBasement/KoboldCPP-MCP-Server) 上其他兼容 MCP 的应用进行交互。
   - 此设置旨在增强跨多个应用的协作式 AI 操作。
- **使用 Notmuch 发送 HTML 邮件**：为 **Notmuch 邮件用户** 创建了一个名为 [mcp-notmuch-sendmail](https://github.com/runekaagaard/mcp-notmuch-sendmail) 的工具，利用 Notmuch 查询发送样式化的 HTML 邮件。
   - 该工具仍处于早期开发阶段，正在征求反馈。
- **用于并行查询的 MCP Inception Server**：**MCP Inception server** 允许向 LLM 发送针对各种参数的并发查询，目前正在开发中，详见 [GitHub](https://github.com/tanevanwifferen/mcp-inception)。
   - 未来的更新可能会实现抓取和分类加密货币的增强功能。
- **Shopify 商家与 Claude 的集成**：推出了一款面向 **Shopify 商家** 的 MCP server，方便与 Claude 进行自然交互，以执行分析店铺数据等任务（[GitHub 链接](https://github.com/amir-bengherbi/shopify-mcp-server)）。
   - 该项目正在进行中，初步提供了一些专注于产品、客户和订单的 Endpoint。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tanevanwifferen/mcp-inception">GitHub - tanevanwifferen/mcp-inception: 从你的 MCP 客户端调用另一个 MCP 客户端。卸载 Context Windows，委托任务，在模型之间进行拆分</a>: Call another MCP client from your MCP client. Offload context windows, delegate tasks, split between models - tanevanwifferen/mcp-inception</li><li><a href="https://github.com/amir-bengherbi/shopify-mcp-server">GitHub - amir-bengherbi/shopify-mcp-server: 用于 Shopify API 的 MCP Server</a>: MCP Server for Shopify API. Contribute to amir-bengherbi/shopify-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/PhialsBasement/KoboldCPP-MCP-Server">GitHub - PhialsBasement/KoboldCPP-MCP-Server: 通过 Claude 或其他兼容 MCP 的应用与 KoboldCPP 进行 AI 对 AI 通信</a>: AI to AI comms with koboldcpp from Claude/other MCP compatible apps - PhialsBasement/KoboldCPP-MCP-Server</li><li><a href="https://github.com/truaxki/mcp-variance-log">GitHub - truaxki/mcp-variance-log: 寻找对话结构中的统计变异并将异常事件记录到 SQLite 数据库的 Agentic 工具。</a>: Agentic tool that looks for statistical variations in conversation structure and logs unusual events to a SQLite database. - truaxki/mcp-variance-log</li><li><a href="https://github.com/giovannicocco/mcp-server-postman-tool-generation">GitHub - giovannicocco/mcp-server-postman-tool-generation</a>: Contribute to giovannicocco/mcp-server-postman-tool-generation development by creating an account on GitHub.</li><li><a href="https://github.com/runekaagaard/mcp-notmuch-sendmail">GitHub - runekaagaard/mcp-notmuch-sendmail: 一个使用 Notmuch 读取邮件并使用 Sendmail 发送邮件的 Model Context Protocol 服务器</a>: A model context protocol server that reads mails with notmuch and sends mail with sendmail - runekaagaard/mcp-notmuch-sendmail</li><li><a href="https://github.com/frgmt0/mcp-reasoner-nightly.git">GitHub - frgmt0/mcp-reasoner-nightly: 为 Claude Desktop 实现的系统化推理 MCP Server，具有 Beam Search 和思维评估功能。</a>: A systematic reasoning MCP server implementation for Claude Desktop with beam search and thought evaluation. - frgmt0/mcp-reasoner-nightly
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1332451319449125048)** (10 条消息🔥): 

> `HeyGen avatars, ElevenLabs 语音选项, 使用 NotebookLM 制作播客, 混合使用 HeyGen 和 MiniMax, NotebookLM 笔记限制` 


- **HeyGen Avatars 工作流详解**：一位用户详细介绍了使用 HeyGen 的工作流，涉及 Avatar 截图、**HailouAI/MiniMax** 以及 **RunWayML's Act-One**，以此创建看起来像 Avatar 正在“倾听”的引人入胜的视频。此外，他们还提供了视频链接，在 [UnrealMysteries.com](https://UnrealMysteries.com) 上展示了这一过程。
   - 与标准的 HeyGen 输出相比，这种方法在视频中实现了更好的 Avatar 交互，展示了一种新颖的视频创作技术。
- **来自 ElevenLabs 的类 HAL 语音**：一名成员询问了视频中特定位置使用的语音，另一名成员透露那是名为 **'Thomas' 的 ElevenLabs 语音**，让人联想到 HAL。这一选择是刻意为之，旨在唤起类似的语调。
   - 这段对话凸显了在创意内容（包括播客和视频制作）中使用语音技术的趋势。
- **NotebookLM 的播客集成**：一位用户分享了使用 NotebookLM 将每周新闻总结为播客格式的经验，尽管该功能尚未广泛普及，但已显示出其效用。他们强调希望通过更好的 Prompt 来增强音频内容的制作。
   - 这反映了人们对利用 AI 工具进行内容生成（特别是在播客领域）日益增长的兴趣。
- **HeyGen 与 MiniMax 的创新混合使用**：一位用户评论了将 **HeyGen** 静态图与 **MiniMax** 混合进行长视频创作的强大能力，称赞了两者的无缝集成。与独立使用任何一种工具相比，这种技术增强了视觉叙事效果。
   - 用户正在探索技术的创意组合，以提高内容制作水平和叙事效果。
- **关于 NotebookLM 笔记限制的查询**：一名成员引导用户查看 'Introduction to NotebookLM' 笔记本以获取有关笔记限制的信息，根据以往的了解，建议上限为 **1000 条笔记**。这一查询表明用户持续需要关于工具能力的清晰文档和反馈。
   - 明确限制可以提升用户体验，确保高效利用平台进行笔记记录和总结。


---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1332450588494925824)** (226 条消息🔥🔥): 

> `NotebookLM 可用性问题，Audio overview 生成延迟，PDF 源可见性问题，NotebookLM 中的语言设置，用户角色与权限` 


- **NotebookLM 面临可用性问题**：用户在 UI 更改后遇到链接源消失的问题，限制了功能并阻碍了用户体验。
   - 社区成员对更新过程中移除重要的 UX 元素表示沮丧，尽管他们曾请求增强功能。
- **Audio overview 生成延迟**：部分用户报告从上传的源生成音频概览时出现异常长的延迟，暗示可能存在 Bug。
   - 建议包括删除并重新上传源，以缓解生成时间过长的问题。
- **PDF 源可见性问题**：用户注意到上传的 PDF 中的某些页面似乎可见性较低，导致 AI 无法提供这些页面的信息。
   - 用户对 NotebookLM 引用页面的方式表示担忧，特别是当其计算页数的方法与印刷页码不一致时。
- **语言设置困惑**：用户对 NotebookLM 的默认语言设置表示困惑，报告输出结果与所需语言不匹配的问题。
   - 社区建议包括检查 Google 账号设置，并利用特定的 URL 来设置语言偏好。
- **了解用户角色**：关于个人资料中的 'user' 角色出现了疑问，特别是涉及 Discord 权限方面。
   - 澄清表明，这些角色可能与 Discord 内部出于组织目的设置的不同权限有关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google/?hl=es">Google NotebookLM | AI 驱动的笔记与研究助手</a>：利用 AI 的力量进行快速摘要和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可以信赖的信息。</li><li><a href="https://notebooklm.google.com/notebook/3499bd65-a247-4519-b1b9-0481e9154496/audio">未找到标题</a>：未找到描述</li><li><a href="http://cloud.google.com/text-to-speech/docs/basics">未找到标题</a>：未找到描述</li><li><a href="https://illuminate.google.com/">Illuminate | 以你的方式学习</a>：使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是您更快理解复杂内容的 Gen AI 工具。</li><li><a href="https://notebooklm.google.com/?hl=es-ES">登录：Google 账号</a>：未找到描述</li><li><a href="https://support.google.com/accounts?p=verify_age">账号设置：您的浏览器不受支持。</a>：未找到描述</li><li><a href="https://getgotak.com/products/gotak-server">GoTAK 现场 TAK Server</a>：GoTAK Server 是运行最新版本 TAK Server 的嵌入式服务器。获取预编程的开发板，跳过命令行和混乱设置的麻烦。这是最快的...</li><li><a href="https://cloud.google.com/generative-ai-app-builder/docs/connect-third-party-data-source">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/ua4rYsMdC4U">AI Software - SNL</a>：一位老师（Ego Nwodim 饰）向她的学生展示了一个由 AI（Timothée Chalamet, Bowen Yang 饰）主持的教育播客。Saturday Night Live。现在可在 Peacock 上流式传输：...</li><li><a href="https://cloud.google.com/distributed-cloud-air-gapped?hl=en#disconnected-sovereign-cloud-solution">Google Distributed Cloud 气隙隔离版 | 主权云</a>：GDC 气隙隔离版使公共部门组织和企业能够满足严格的数据驻留和安全要求。</li><li><a href="https://youtubetranscript.com/">YouTube Transcript - 阅读 YouTube 视频</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1332477109393686600)** (223 messages🔥🔥): 

> `Hunyuan-video model, Kling AI quality, AI image generation setups, Stable Diffusion RAM requirements, Deepseek model limitations` 


- **Hunyuan-video 模型取得成功**：许多用户确认 **hunyuan-video model** 运行高效，即使在 **12 GB VRAM** 的系统上也是如此。
   - 虽然并不完美，但据报告在 image to video 应用中非常有效。
- **Kling AI 与 Hunyuan 的对比**：一位用户提到，就本地使用的质量而言，**Kling AI** 是目前最接近 **hunyuan** 模型的选项之一。
   - 然而，它目前缺乏必要的 image-to-video 功能，而这是许多用户的核心需求。
- **AI 图像生成的最佳配置**：建议新用户使用 **Forge** 或 **Swarm**，因为它们为本地 AI 图像生成初学者提供了更好的支持和教程。
   - 虽然对于高级用户高度推荐 **ComfyUI**，但其复杂性对初学者来说可能是一个挑战。
- **Stable Diffusion 的 RAM 需求**：为了有效运行 **Stable Diffusion**，建议至少配备 **32GB RAM**，**64GB** 为最佳。
   - 鼓励使用 **RTX 4090** 或 **AMD 7900XTX** 系统的用户确保其 RAM 满足这些要求，以避免出现并发症。
- **Deepseek 模型硬件需求**：**Deepseek V3** 或 **R1** 模型在全精度下需要超过 **1.3TB 的 VRAM** 才能运行，这超出了典型消费级设备的范畴。
   - 运行此类超大模型需要多个像 **A100** 或 **H100** 这样的高端 GPU，这使得大多数用户只能寻求更小的替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://upscayl.org/">Upscayl - AI Image Upscaler</a>：暂无描述</li><li><a href="https://openmodeldb.info/">OpenModelDB</a>：OpenModelDB 是一个由社区驱动的 AI 放大模型数据库。我们的目标是提供一种比现有来源更好的方式来查找和比较模型。</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides#amd-forge-webui-with-zluda">Webui 安装指南</a>：Stable Diffusion 知识库（配置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1332451305720909854)** (20 messages🔥): 

> `Flash Infer Talk Questions, Support for Alternative Attention Methods, Deepseek Events, NCSA Hackathon Participation, Distributed Training Stacks` 


- **Flash Infer 讲座提问**：成员们讨论了如何在 YouTube 的 **Flash Infer Talk** 期间提问，并指出如果不先创建频道就无法提问。
   - *“如果你能在某个时候代为提问——我很好奇对替代注意力机制的支持……”* 是提出的一个寻求进一步澄清的关键问题。
- **替代注意力机制探索**：大家对 **differential attention** 的支持及其实现（包括 **flex attention** 的使用）感到好奇。
   - 一位成员分享了展示该概念的 **pseudocode**，并指出 **flex attention** 目前不支持 **differential attention**。
- **Deepseek 咨询**：一位成员询问了有关 **deepseek** 相关事件的摘要或 TLDR，表现出对近期进展的兴趣。
   - 讨论中没有提供关于 **deepseek** 的更多细节。
- **NCSA 黑客松参赛邀请**：一位成员正在寻找另外两人加入 **NCSA hackathon**，并鼓励感兴趣的人通过 DM 联系。
   - 分享了活动信息，但遇到了 CSS 错误，导致无法加载更多细节。
- **对分布式训练技术栈的好奇**：一位成员询问了团队在处理超过 100B 参数的大规模 **distributed training** 时使用的技术栈。
   - 他们分享了使用 **JAX** 的经验，但强调了新团队成员面临的陡峭学习曲线。



**提及的链接**：<a href="https://www.openhackathons.org/s/siteevent/a0CUP000013BcYw2AK/se000370">Open Hackathons</a>：暂无描述

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1332814793886797884)** (1 条消息): 

> `Triton 中的 Tensor 操作，Triton 中的内联汇编` 


- **在 Triton 中移动 Tensor 元素？**: 一位成员询问了在 Triton 中*将 Tensor 元素向左移动*的方法，并对可能的内联汇编解决方案表示好奇。
   - 未收到任何回复，这表明社区内可能存在不确定性或缺乏现有的解决方案。
- **讨论了内联汇编的潜力**: 讨论涉及了内联汇编是否可以促进 Triton 中的 Tensor 操作，但未详细说明具体细节。
   - 对该话题的兴趣表明需要更多的资源或示例。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1332465575028326452)** (76 条消息🔥🔥): 

> `PTX ASM 段错误问题，CUDA Kernel 加载错误，DeepSeek 讨论，CUDA 版本与兼容性，NCCL 超时调试` 


- **PTX ASM 段错误问题**: 一位用户报告了在尝试将其 PTX ASM 代码中从向量寄存器存储到共享内存时出现段错误（segfault），暗示可能存在内存地址问题。
   - 另一位用户建议检查地址是否在正确的内存空间中，并建议在地址上调用 `__cvta_generic_to_shared()`。
- **CUDA Kernel 加载错误**: 一位用户在 Jupyter Lab 中尝试加载 CUDA 模块时，由于 Ninja 构建系统不可用而遇到 ImportError，导致无法找到所需的共享对象文件。
   - 安装 Ninja 后，该用户仍面临 ImportError，表明内联扩展无法打开，可能是由于构建过程未成功完成。
- **DeepSeek 讨论**: 一位成员幽默地推测 DeepSeek 的情况是否会导致雇主雇佣更多的 CUDA 开发者，并强调了潜在的成本节约。
   - 其他成员也加入了关于经济影响的调侃，提到了使用适当计算资源带来的效率提升。
- **CUDA 版本与兼容性**: 用户讨论了各种 CUDA 版本的兼容性问题，特别指出旧款 GPU 在运行 CUDA 12 等较新版本时可能会比较吃力。
   - 一位用户分享了在尝试将过时硬件与现代软件需求配合使用时所面临的挫折。
- **NCCL 超时调试**: 一位用户询问了如何调试在多节点训练期间遇到的 NCCL 超时问题，提到了分析（profiling）方面的改进，但超时问题仍未解决。
   - 有人请求提供处理 NCCL 超时的最佳实践，表明在多节点训练设置中需要有效的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda">CUDA Toolkit Documentation 12.8</a>: 未找到描述</li><li><a href="https://x.com/__tensorcore__/status/1883060903282954282">来自 Vijay (@__tensorcore__) 的推文</a>: 🔥🚨 CUTLASS Blackwell 发布了 🚨🔥 3.8 版本加载了对 Blackwell 新特性的支持，甚至还有一个 Attention Kernel 👀 快去看看吧: https://github.com/nvidia/cutlass 等不及要...</li><li><a href="https://docs.nvidia.com">NVIDIA Documentation Hub - NVIDIA Docs</a>: 未找到描述</li><li><a href="https://stackoverflow.com/q/53422407/10107454)">nvcc 和 NVIDIA-smi 显示的 CUDA 版本不同</a>: 我对运行 which nvcc 和 nvidia-smi 显示的不同 CUDA 版本感到非常困惑。我在 ubuntu 16.04 上同时安装了 cuda9.2 和 cuda10。现在我设置了指向 cuda9.2 的 PATH。所以...</li><li><a href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-generations).">1. 介绍 — NVIDIA CUDA 编译器驱动程序 12.8 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1332448788513685554)** (13 条消息🔥): 

> `NCCL timeouts debugging, Linear Warmup in Learning Rates, Torch Inductor Internals, Vision-based model optimizations, Fused CUDA kernels example` 


- **多节点训练中的 NCCL 超时**：一名成员报告在 8 节点训练期间遇到 **NCCL timeouts**，并提到使用 PyTorch profiler 对代码进行性能分析以提升性能。
   - 他们寻求关于在多节点设置中调试 NCCL 超时的**最佳实践**指导。
- **带有线性预热的最佳学习率策略**：一名成员再次确认，在过渡到不同的 LR 策略之前，在前 N 步使用 **linear warmup** 是有效的，特别是对于视觉模型。
   - 他们建议以 **5e-4** 的高学习率开始，并在大量的迭代次数中逐渐降低到 **5e-5**。
- **关于 Torch Inductor 内部原理的咨询**：一名成员询问了关于 **Torch Inductor** 内部原理的文档，特别是涉及 **subgraphs、ComputedBuffer** 和 **IR nodes** 等概念。
   - 他们正在寻找详细解释这些概念之间交互作用的资源。
- **Linear + ReLU 融合算子的挑战**：一名成员寻求关于在 CUDA 中创建 **fused kernel** 作为 PyTorch 扩展的帮助，希望在保持权重加载功能的同时替换标准层。
   - 他们表示有兴趣手动实现这一点，而不依赖于像 **triton** 这样的现有解决方案。
- **非 Transformer 视觉模型与优化偏好**：一名成员分享了他们的经验，即**大多数非 Transformer 视觉模型**在使用 Adam 优化时表现不佳，更倾向于使用 **Triangular** 或 **WSD** 学习率调度等替代方案。
   - 这引发了关于视觉任务中各种优化策略有效性的更广泛讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts">CosineAnnealingWarmRestarts &mdash; PyTorch 2.5 documentation</a>: 未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR">LinearLR &mdash; PyTorch 2.5 documentation</a>: 未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR">CosineAnnealingLR &mdash; PyTorch 2.5 documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1332799975129616507)** (1 条消息): 

> `Adam Paszke, Mosaic GPU, GPU MODE community, GPU programming` 


- **Adam Paszke 谈 Mosaic GPU**：在短短 **10 分钟**内，传奇人物 **Adam Paszke** 将讨论他用于底层 GPU 编程的 DSL —— **Mosaic GPU**。
   - 在 [YouTube](https://www.youtube.com/@GPUMODE) 上观看直播演讲，并在这次富有洞察力的会议中扩展你的 GPU 编程知识。
- **探索 GPU MODE 社区资源**：对于感兴趣的人，可以在 [GitHub](https://github.com/gpu-mode) 上找到由 Mark Saroufim 和 Andreas Köpf 创建的关于 GPU 编程的补充内容。
   - 访问官方 [Discord](https://discord.gg/gpumode) 频道加入这个不断壮大的社区，进行动态讨论和协作学习。



**提到的链接**: <a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: 一个 GPU 阅读小组和社区 https://discord.gg/gpumode 补充内容见此处 https://github.com/gpu-mode 由 Mark Saroufim 和 Andreas Köpf 创建 

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1332490589081440256)** (2 messages): 

> `TinyZero, Open R1` 


- **TinyZero 复现了 DeepSeek R1 Zero**：[TinyZero](https://github.com/Jiayi-Pan/TinyZero) 项目是一个易于上手的 **DeepSeek R1 Zero** 复现版本，展示了简洁的实现方式。
   - 其 GitHub 仓库包含详细信息和图像，方便贡献者探索和参与。
- **Open R1 作为一个完全开源项目构建**：由 Hugging Face 发起的 [Open R1](https://github.com/huggingface/open-r1) 项目是 **DeepSeek-R1** 的完全开放复现版本，旨在促进协作开发。
   - 鼓励开发者在 GitHub 上做出贡献，确保为功能增强和修改提供包容性的环境。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: Clean, accessible reproduction of DeepSeek R1-Zero</a>: DeepSeek R1-Zero 的简洁、易上手的复现 - Jiayi-Pan/TinyZero</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: Fully open reproduction of DeepSeek-R1</a>: DeepSeek-R1 的完全开放复现。通过在 GitHub 上创建账号来为 huggingface/open-r1 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1332811365618028598)** (2 messages): 

> `Atomic Semi Careers, Hinge Health Job Opening` 


- **Atomic Semi 寻找实战型工程师**：Atomic Semi 正在组建一支卓越的实战型工程师团队，以在技术领域进行创新，并声称：**“我们将拥有从原子到架构的整个技术栈。”**
   - *“我们相信我们的团队和实验室可以利用包括 3D 打印机和 e-beam writers 在内的先进工具制造任何东西。”*
- **Hinge Health 为 AI 平台招聘 Staff Engineer**：Hinge Health 正在为其 AI 平台招聘 **Staff Engineer**，详情分享在 LinkedIn [此处](https://www.linkedin.com/jobs/view/4096940351)。
   - 该消息鼓励大家转发，并表示可以 **DM（私信）了解职位详情**。



**提及的链接**：<a href="https://atomicsemi.com/careers/">Careers</a>：未找到描述

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1332561635025354843)** (5 messages): 

> `High Performance Computing in ML, Basics of Parallel Computing, Understanding Neural Networks, Self Implementation of SVM, Learning Path for Practical Skills` 


- **学习高性能计算基础**：一位成员解释说，你将学习对于 **High Performance Computing**（高性能计算）和 **Machine Learning** 系统至关重要的不同概念。
   - 他们强调了理解 **GPU architecture** 的重要性，以便编写高效的算法并利用 **hw-aware** 技术优化 AI 工作流。
- **学习要素**：当被问及先修知识时，一位成员建议掌握一些 **Parallel Computing**（并行计算）的基础知识以及 **Neural Networks**（神经网络）运作方式的知识。
   - 需要重点关注的关键主题包括 **matmuls**、**attention** 和 **activations**。
- **寻求 SVM 实现方面的帮助**：一位成员询问了关于仅使用 **Numpy** 和 **Scipy** 自主实现 **SVM** 的疑问。
   - 这表明社区具有协作性，愿意帮助解决具体的代码挑战。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1332529746562514996)** (4 messages): 

> `Tiled Matrix Multiplication Issues, Floating Point Type Mismatch, Dummy Matrix Declaration, Result Comparison Code` 


- **分块矩阵乘法（Tiled Matrix Multiplication）显示不匹配**：一位用户报告说，当使用某种方法声明虚拟矩阵时，大矩阵（320x320）在 CPU 和 GPU 之间的结果不匹配，而小矩阵（4x4）则正常。
   - 他们最初怀疑矩阵声明方式的不同导致了内存泄漏。
- **怀疑浮点类型导致不匹配**：另一位成员建议问题可能源于矩阵声明中使用的 **floating point type**（浮点类型），并促使该用户检查在使用整数时是否会出现不匹配。
   - 该用户随后澄清说，他们将矩阵声明为 float 数组，但输入的是整数。
- **请求对比代码**：一位成员请求分享对比代码以进一步诊断不一致的原因，这表明需要更详细的分析。
   - 这一请求暗示了通过潜在的协作来识别并解决矩阵乘法问题。


  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1332941600459853846)** (1 条消息): 

> `YouTube Recordings` 


- **探索最新的 YouTube 录制视频**：对于有兴趣补课的观众，现在有两个新的录制视频可用：[录制视频一](https://www.youtube.com/watch?v=iOLBJwENuvA) 和 [录制视频二](https://www.youtube.com/watch?v=wKd90avC8Nc)。
   - 这些录制视频非常适合那些希望跟进最新讨论的人。
- **回顾近期会议的精彩内容**：查看录制视频中分享的近期亮点，其中涵盖了社区中重要的讨论和见解。
   - 对于想要深入了解正在进行的议题的人来说，这些视频是极佳的资源。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1332816042749329408)** (3 条消息): 

> `Emulation in Torch, JAX fp8 support on Nvidia GPUs` 


- **探索 Torch 中的仿真**：讨论中提到 **Torch** 具有一定的仿真支持，表明其具备仿真某些功能的能力。
   - 一位成员询问是否有**指南**详细说明仿真功能在代码中的具体实现位置。
- **JAX 独特的 fp8 能力**：[GitHub](https://github.com/jax-ml/jax/discussions/26077) 上的一篇链接讨论强调了 **JAX** 如何在 **sm < 89** 的 Nvidia GPU 上运行 **fp8**，而这通常仅限于 **sm >= 89** 的 GPU（如 RTX 4090 或 A100）。
   - 这段特定的 Discord 对话强调了关于在旧款 GPU 上**运行 fp8** 的困惑，同时指出 **JAX** 成功绕过了 **PyTorch** 中存在的这些问题。



**提到的链接**：<a href="https://github.com/jax-ml/jax/discussions/26077">Why can JAX run fp8 on Nvidia GPUs with sm &lt; 89? · jax-ml/jax · Discussion #26077</a>：fp8 仅在 sm &gt;= 89 的 GPU（如 RTX 4090 或 A100）上具有硬件支持。我看到有人尝试在旧款 GPU 上的 PyTorch 中运行它（例如这个脚本）并报错。但 JAX 可以实现……

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1332807899679953019)** (11 条消息🔥): 

> `Mosaic Layout System, TiledLayout Comments, SMEM to Registers Transfer, IR Generation Flags in Mosaic` 


- **Mosaic 布局系统统一**：讨论强调了使用 **XLA tiling notation** 统一 **Mosaic** 布局系统，告别了特殊情况的堆砌。
   - 值得注意的是，这专门适用于寄存器中的数组布局，而 **SMEM** 仅需要一级 tiling 和 swizzle。
- **TiledLayout 反馈**：一位成员对讨论表示感谢，称关于 **TiledLayout** 的注释“非常漂亮”，正是他们所寻找的。
   - 他们现在正在探索从寄存器到 **SMEM** 的映射，并表示需要进行实际操作。
- **SMEM 到寄存器的传输方法**：对话还涉及了合成 **SMEM** 到寄存器传输的方法，并详细说明了其在代码中的实现。
   - 这包括一个旨在最小化 **bank conflicts** 的规划器，以提升性能。
- **对中间表示（IR）的兴趣**：有人提出了关于检查 Mosaic 生成的**中间表示 (IR)** 的问题。
   - 询问是否有任何可用的标志（flags）来导出 (dump) IR，表明了对更深层次技术洞察的渴望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/jax-ml/jax/blob/95cb0eb1c969948f21e901317a083375ad13194a/jax/experimental/mosaic/gpu/fragmented_array.py#L144">jax/jax/experimental/mosaic/gpu/fragmented_array.py at 95cb0eb1c969948f21e901317a083375ad13194a · jax-ml/jax</a>：Python+NumPy 程序的可组合转换：微分、向量化、JIT 到 GPU/TPU 等 - jax-ml/jax</li><li><a href="https://github.com/jax-ml/jax/blob/95cb0eb1c969948f21e901317a083375ad13194a/jax/experimental/mosaic/gpu/fragmented_array.py#L1605-L1606">jax/jax/experimental/mosaic/gpu/fragmented_array.py at 95cb0eb1c969948f21e901317a083375ad13194a · jax-ml/jax</a>：Python+NumPy 程序的可组合转换：微分、向量化、JIT 到 GPU/TPU 等 - jax-ml/jax
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1333542150192758795)** (1 messages): 

> `Tile Lang, BitBLAS repo, Backward kernels` 


- **Tile Lang 终于发布**：一位成员对 **Tile Lang** 的发布表示兴奋，并提到早在 10 月份的 **BitBLAS** 仓库提交中就曾提及过它。
   - *希望我终于能编写那些 BitBLAS 目前缺失的高效 backward kernels*。
- **对 BitBLAS 改进的兴趣**：人们对于通过集成 **Tile Lang** 来增强 **BitBLAS** 的兴趣日益浓厚，旨在提高 backward kernels 的效率。
   - 这将解决一些已被指出的功能缺失问题。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1333474041151098890)** (1 messages): 

> `dpo loss, simpo loss, liger with trl, ligerdpo trainer` 


- **关于 dpo 和 simpo loss 使用的咨询**：一位成员询问是否有办法将 Liger 的 **dpo** 或 **simpo loss** 与 **trl** 或 **ligerdpo trainer** 结合使用。
   - 他们希望这个咨询适合在该频道讨论。
- **对 Liger 功能的兴趣**：该成员的咨询凸显了将 **Liger** 的特定功能与 **trl** 结合使用的兴趣，表明需要更好的集成或工具。
   - 这也反映了社区正在不断探索结合各种技术和工具以增强性能。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

mobicham: https://x.com/Mobius_Labs/status/1883951887965393301
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1333212622098661546)** (3 messages): 

> `WGMMA Instructions, Pointer Math in PTX ISA, Memory Handling Strategies` 


- **探索 WGMMA 指令实现**：一位成员正尝试让 **WGMMA 指令** 独立工作，重点关注 **pointer math**（指针计算），以便按照 **NVIDIA PTX ISA** 的概述将正确的元素加载到寄存器中。
   - 他们对内存管理表示困惑，询问 **dst object** 是否会自动处理内存计算。
- **TK 的内存管理方法**：同一位成员注意到，在查看 **TK 生成的 WGMMA 指令** 时，它们似乎通过传入连续的内存段来避免指针计算。
   - 这引发了一个疑问：这是否意味着对 **PTX ISA** 的理解有误，或者是否存在对此类计算的自动处理。
- **线程加载指定讨论**：该成员假设代码设计为每个线程将其特定元素加载到指定的寄存器中，并询问这是否是正确的方法。
   - 他们正在寻求关于当前设计中内存处理机制和意图的澄清。


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1332451795095785633)** (44 messages🔥): 

> `Polynomial Equations PR, Maze Task Proposal, FSDP Support in Tiny-GRPO, Family Relationships Dataset, GSM8K Templates` 


- **Reasoning Gym 新增多项式方程**：一位成员提交了一个 PR，在简单线性方程的基础上增加了对 **多项式方程** 的支持，增强了 reasoning-gym 的功能。
   - 另一位成员提到，由于 CLRS 采用 Apache 许可证，他们可以从中复制算法。
- **迷宫任务想法建议**：一位成员提议增加一个专注于寻找最短路径长度的 **迷宫任务**，并就此寻求反馈。
   - 其他人表示兴奋，认为迷宫谜题是对 reasoning-gym 的极佳补充。
- **FSDP 和 Tiny-GRPO 增强**：一位成员指出 **Tiny-GRPO** 已添加 **FSDP 支持**，这为减少 VRAM 占用打开了大门，同时也提出了进一步增强的请求。
   - 这被视为使 Tiny-GRPO 更加用户友好和高效的一步。
- **家庭关系数据集策略**：讨论了生成家庭关系数据集的策略，揭示了 LLM 在解决此类问题方法上的复杂性。
   - 提供了一个建议的实现方案，并指向了一个现有的家庭关系代码库以供参考。
- **GSM8K 模板数据集提案**：一位成员建议创建一个基于模板的 **GSM8K 数据集** 版本，类似于 Apple 的工作，并在 GitHub 上发布了相关代码。
   - 讨论了在 HF hub 上托管此模板版本的计划，以便根据用户需求进行动态下载。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://hkust-nlp.notion.site/simplerl-reason">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://github.com/google-deepmind/clrs/tree/master/clrs/_src/clrs_text">google-deepmind/clrs 项目中的 clrs/clrs/_src/clrs_text</a>：通过在 GitHub 上创建账号，为 google-deepmind/clrs 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/4">添加程序化《爱丽丝梦游仙境》（AIW 问题）数据集 · Issue #4 · open-thought/reasoning-gym</a>：《爱丽丝梦游仙境》问题具有以下基础模板（及多种变体）：“爱丽丝有 N 个兄弟和 M 个姐妹。请问爱丽丝的兄弟有多少个姐妹？” 参见论文：A...</li><li><a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: 一组旨在挑战大语言模型在存在误导信息时的推理能力的提示词集合</a>：一组旨在挑战大语言模型在存在误导信息时的推理能力的提示词集合 - cpldcpu/MisguidedAttention</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/6">数据集：单词乱序重组 · Issue #6 · open-thought/reasoning-gym</a>：我建议从 Level 0 开始：单词乱序重组：加载长度不超过 max_length 的自然语言文本片段（例如来自 data 中的儒勒·凡尔纳短篇小说），对每个单词随机交换字符...</li><li><a href="https://github.com/apple/ml-gsm-symbolic">GitHub - apple/ml-gsm-symbolic: GSM-Symbolic 模板和生成的数据</a>：GSM-Symbolic 模板和生成的数据。通过在 GitHub 上创建账号，为 apple/ml-gsm-symbolic 的开发做出贡献。</li><li><a href="https://github.com/open-thought/tiny-grpo/blob/eafedd78ff86dbb724a3dd21bb04ab6523ac8f3c/train.py#L122-L130">open-thought/tiny-grpo 项目中的 tiny-grpo/train.py</a>：极简且可定制的 GRPO 实现。通过在 GitHub 上创建账号，为 open-thought/tiny-grpo 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/graphs/family_relationships.py">open-thought/reasoning-gym 项目中的 reasoning-gym/reasoning_gym/graphs/family_relationships.py</a>：程序化推理数据集。通过在 GitHub 上创建账号，为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/11">由 panispani 添加对所有 CLRS 任务的支持 · Pull Request #11 · open-thought/reasoning-gym</a>：CLRS 是经典的算法教科书。Deepmind 推出了 CLRS 基准测试，其中还包括大多数经典算法的文本版本，称为 CLRS-text。在此 PR 中，我移植了所有...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/12">[乱序任务 Level 1] - 添加句子重排序及验证单元测试 · Pull Request #12 · open-thought/reasoning-gym</a>：为 issue 6 中的乱序任务 Level 1 添加数据集生成器。我将此任务视为句子重排序任务。生成数据示例：{'question': 'Correct the following sen...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/8">添加安装说明 · Pull Request #8 · open-thought/reasoning-gym</a>：在成功构建 reasoning_gym 仓库时遇到了一些问题。觉得其他人可能也需要它以避免浪费时间。</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/9">在 README.md 中添加安装说明 · Issue #9 · open-thought/reasoning-gym</a>：使用该库的主要方式应该是通过 pip install reasoning-gym，参见项目的 PyPI 页面。对于 git clone 后的本地源码安装，使用 pip install -e .（-e 表示可编辑模式）。除此之外...</li>
</ul>

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1333523117607227442)** (5 messages): 

> `Mojo Documentation, GPU Package API` 


- **Mojo 文档经历宕机**：**Mojo documentation** 经历了宕机，团队正在积极努力尽快解决该问题。
   - 在宕机期间，团队对用户的耐心等待表示了感谢，并致力于让用户及时了解进度。
- **文档恢复上线并包含新的 GPU API 信息**：宕机结束后，一名成员确认 **docs** 已恢复上线。
   - **GPU package API 文档** 现在也可以在 nightly 版本中访问，这对寻求更新信息的用户来说是个好消息。
- **代码引用中的拼写错误**：一位成员指出代码片段中可能存在拼写错误，指出应改为 `# val=6`。
   - 他们附带了一张 [图片](https://cdn.discordapp.com/attachments/1098713601386233997/1333523117112561706/image.png?ex=679933ae&is=6797e22e&hm=f4806c352f0e31d85e4082447280257cbc8624a7b39ab2b101bf42dc4174ded4&) 以清晰说明该错误。
- **分享了 GitHub Changelog 链接**：分享了 **Mojo GitHub changelog** 的链接，提供了对文档更新内容的深入了解。
   - 包含 [changelog](https://github.com/modular/mojo/blob/nightly/docs/changelog.md) 有助于用户了解 Mojo 编程语言的最新变化。



**提及的链接**：<a href="https://github.com/modular/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modular/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modular/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1332488778928881735)** (91 messages🔥🔥): 

> `Mojo CSS Struct, List and Representable Trait Issues, Unsafe Pointers and Object Identity, Function Pointer FFI in Mojo, Documentation Downtime` 


- **Mojo CSS Struct 开发**：一位用户正在创建一个 `struct` 以使用 `fluent API` 风格生成 CSS，但在 Zed Preview 中遇到了未使用值的警告问题。
   - 他们建议使用 `_ = ` 来消除警告，但表示更倾向于一种更整洁的解决方案。
- **List 和 Representable Trait 的困惑**：关于 `Representable` trait 的困惑引发了讨论，特别是关于在函数中使用 `List[Int]` 作为参数的问题。
   - 有人指出，虽然 `List` 具有表示函数，但编译器无法将 `Int` 识别为 `Representable`，这表明 conditional conformance 存在问题。
- **Unsafe Pointer 和对象标识 (Object Identity) 问题**：一位用户在使用 `UnsafePointer` 为其 value struct 进行反向操作时遇到了对象标识问题，发现他们的指针被独立影响。
   - 他们发现将类变量调整为指针有助于正确跟踪更改，从而维持对象标识。
- **Mojo 中 Function Pointer FFI 的限制**：对话强调了当前版本的 Mojo 无法可靠地支持将函数指针传递给 C 函数。
   - 虽然 Mojo 的子集可能符合 C ABI，但目前缺乏此类规则的文档。
- **文档宕机与解决**：用户报告了访问 Mojo 文档的问题，发现由于 Cloudflare 的托管问题导致文档下线。
   - 团队承认了该问题，并确认文档在初始报告后不久已恢复上线。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/mojo/issues/3968">modular/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modular/mojo 的开发做出贡献。</li><li><a href="https://github.com/modular/mojo/blob/nightly/stdlib/src/builtin/int.mojo#L1146">mojo/stdlib/src/builtin/int.mojo at nightly · modular/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modular/mojo 的开发做出贡献。</li><li><a href="https://github.com/modular/mojo/blob/nightly/stdlib/src/collections/list.mojo#L441">mojo/stdlib/src/collections/list.mojo at nightly · modular/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modular/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1332867869377302611)** (4 条消息): 

> `Multi-agent workflows, Document research agents, Automation in travel insurance claims, LlamaIndex integrations, DeepSeek API` 


- **Presenter: 你的全新演示文稿 Multi-Agent 工作流**：介绍 [Presenter](https://twitter.com/llama_index/status/1883307955782901926)，这是一个能够创建视觉效果丰富的演示文稿的 Multi-agent 工作流，具有 **Mermaid diagrams** 和脚本生成功能。
   - 该仓库是那些旨在构建具有强大功能的 **report generation agent** 的开发者的完美参考。
- **文档研究开源模板**：受 Google Deep Research 启发，@MarcusSchiesser 在[这里](https://twitter.com/llama_index/status/1883675662839636427)创建了一个完全开源的全栈模板，用于 **multi-step document research agents**。
   - 该工具解决了用户在高效处理复杂研究任务方面的重大需求。
- **Scaleport AI 自动化理赔评估**：了解 [Scaleport AI](https://twitter.com/llama_index/status/1883929949205336509) 如何与领先的旅游保险提供商合作，利用 LlamaIndex 从复杂的医疗报告中**自动进行理赔评估**。
   - 他们利用 **advanced OCR** 进行数据提取，展示了 AI 驱动分析在该领域的有效性。
- **LlamaIndex 对 DeepSeek-R1 API 的官方集成**：LlamaIndex 现在集成了 [DeepSeek-R1 API](https://twitter.com/llama_index/status/1883986763380842864)，允许使用 `deepseek-chat` 和 `deepseek-reasoner` 等模型。
   - 访问 [DeepSeek](https://api-docs.deepseek.com/) 了解 API keys 和模型支持的详情，为用户提供无缝集成体验。



**提到的链接**: <a href="https://t.co/jtfBvBig1y">DeepSeek - LlamaIndex</a>: 未找到描述

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1332628659026268191)** (74 条消息🔥🔥): 

> `LlamaIndex 访问权限, LLM.complete kwargs 用法, LlamaIndex 中的 Evaluators, 本地 RAG 实现, 文档问题` 


- **LlamaIndex 访问更新**：一位用户询问了目前在等待名单中获得 LlamaIndex 访问权限的时间表。
   - 讨论中未提供具体的时间范围。
- **在 LLM.complete 中使用 kwargs**：一位成员寻求关于在 `LLM.complete` 中使用 `**kwargs` 的文档，旨在消息方法中动态传递参数。
   - 社区回复指出 kwargs 会被发送到 LLM API，并建议为参数指定 `generation_config`。
- **与 Evaluators 相关的成本**：一位用户询问在使用 Anthropic 作为 LLM 时，`FaithfulnessEvaluator` 和 `RelevancyEvaluator` 是否会通过 API 请求产生额外费用。
   - 确认这两个评估器都会调用 LLM，从而影响成本。
- **本地 RAG 实现疑问**：一位参与者讨论了在使用 Ollama 和 LlamaIndex 实现本地 RAG 时面临的挑战，表示难以找到有用的文档。
   - 尽管最初存在问题，他们还是成功完成了入门示例，但认为文档缺乏清晰度。
- **文档与性能问题**：用户对模型文档和 LlamaIndex 的性能表示担忧，其中一位用户提到运行期间 CPU 占用率过高。
   - 社区承认了文档的局限性，但鼓励通过贡献来改进，并讨论了使用 Ollama 进行更简便配置的好处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/understanding/agent/rag_agent/">Adding RAG to an agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>：未找到描述</li><li><a href="https://christophergs.com/blog/running-open-source-llms-in-python#install">Running Open Source LLMs In Python - A Practical Guide</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/">LlamaCPP - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/">Building an LLM Application - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/vllm/">vLLM - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/17647">update llama-cpp integration + docs by logan-markewich · Pull Request #17647 · run-llama/llama_index</a>：文档和依赖项相当陈旧。已更新以使其更现代化。</li><li><a href="https://github.com/run-llama/llama_index/issues/7547">[Question]: GGUF model support? · Issue #7547 · run-llama/llama_index</a>：问题验证：我已在文档和 Discord 中搜索过答案。问题：新版本的 llama.cpp 抛出错误，因为现在仅支持基于 GGUF 的模型。这项工作...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1332534744990089311)** (34 messages🔥): 

> `Cohere 法律法规, Cohere UI 反馈, 社区参与, GitHub 作为协作工具` 


- **Cohere 的法律法规限制极少**：法律反馈确认，与日本的大多数 AI 贸易不受新规影响，新规主要针对先进计算芯片和大模型。Cohere 模型不属于这些类别，且相关规定要到 2025 年 5 月才会生效。
   - 法律团队正在积极监控局势，以便在法规正式实施前应对任何潜在变化。
- **用户对 Cohere 网站 UI 的反馈**：反馈强调了对 [Cohere dashboard](https://dashboard.cohere.com/) UI 的困惑，特别是页面两侧按钮的相似性。建议包括将 Discord 和电子邮件联系选项放在更显眼的位置。
   - 一位用户建议进行设计更改，例如像其他平台一样简化按钮布局。
- **寻求项目合作伙伴**：一位用户询问如何寻找合作伙伴进行协作项目，并建议利用 GitHub。有人指出该频道主要用于讨论，而非招聘。
   - 成员们表示，该领域需要多样性，而不是创建多家做同样事情的公司。
- **欢迎新社区成员**：新成员加入了社区，介绍自己是渴望做出贡献的软件开发人员和 AI 爱好者。用户之间充满了友爱和欢迎的氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://platform.deepseek.com/usage,">DeepSeek Platform</a>：加入 DeepSeek API 平台以访问我们的 AI 模型、开发者资源和 API 文档。</li><li><a href="https://dashboard.cohere.com/">Login | Cohere</a>：登录以通过易于使用的 API 访问先进的 Large Language Models 和 NLP 工具。</li><li><a href="https://cohere.com/">The World&#x27;s Leading AI Platform for Enterprise | Cohere</a>：Cohere 是领先的企业级 AI 平台。通过安全且可扩展的 AI 增强劳动力、自动化工作流并丰富客户体验。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1333025312203866212)** (34 messages🔥): 

> `Cohere 文档, 逆向规划, Cohere LLM 使用, Cohere 平台概览, TTS 和 STT 功能` 


- **Cohere 提供不具备 TTS 或 STT 能力的 LLM**：Cohere 平台专注于大语言模型 (LLM)，不提供文本转语音 (TTS) 或语音转文本 (STT) 功能。
   - 这一点在用户讨论平台功能时得到了确认，并明确了仅提供 LLM。
- **在 Cohere 中使用 LLM：分步指南**：要在 Cohere 中使用 LLM，需使用 `ChatCohere` 类定义 LLM，使用 `bind_tools` 绑定工具，并使用消息调用 LLM。
   - 提供了一个示例代码片段来引导用户完成此过程，清晰地展示了各个步骤。
- **理解逆向规划的概念**：一位用户提出了一种逆向规划方法，即检查最后几个步骤，以便从该点向后促进操作。
   - 机器人开始搜索有关逆向规划技术的信息，但未找到特定资源。
- **什么是 Cohere？简要概览**：Cohere 是一个允许开发由 LLM 驱动的应用程序的平台，强调安全和私有部署。
   - 它提供了一套用于构建自然语言任务（如分类、摘要和内容生成）的工具包，并具有自定义模型训练的灵活性。
- **Cohere 多步工具使用详解**：Cohere 文档描述了一种系统的工具使用方法，涉及从用户消息检索到响应生成的多个阶段。
   - 多步工具使用允许顺序推理，这对于需要在任务期间进行调整的 Agent 至关重要。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1332556881591533599)** (2 messages): 

> `应用查看, 私信` 


- **用户对应用表示兴趣**：一位用户表示他们会查看该应用，并提到将通过私信 (DM) 进行后续跟进。
   - 这表明社区内对该应用有着持续的参与和兴趣。
- **提醒查收私信**：一位成员艾特了另一位用户，请求其查收私信。
   - 这强调了直接沟通在协调活动或反馈中的重要性。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1332449455148109955)** (50 条消息🔥): 

> `开源图像分析模型, DeepSeek 模型问题, 本地运行模型, 文档分析工具, DeepSeek R1 可用性` 


- **寻求开源图像分析模型**：用户正在询问支持图像上传和查询响应的最佳开源图像分析模型，并提到了使用 [Taggui](https://taggui.com) 等框架进行图像打标。
   - 虽然建议中包含了各种模型，但关于哪些模型能有效分析并响应图像查询仍不明确。
- **排查 DeepSeek 模型错误**：多位用户报告了 DeepSeek R1 模型的问题，特别是聊天模板和推理任务中的错误，表明该模型目前还无法完全开箱即用。
   - 讨论点包括可在本地运行的特定模型，以及围绕 LLAMA 和 DeepSeek 等工具的各种性能基准测试。
- **对本地文档分析工具的兴趣**：用户表示需要本地工具来分析个人文档，而无需将其上传到云端服务，以寻求统计见解和主题出现频率。
   - 建议包括使用 PDFGear 等软件，但对数据隐私仍有顾虑。
- **DeepSeek R1 支持与可用性**：社区询问了 GPT4All 支持 DeepSeek R1 的进展，并提到正在进行实现平滑集成的工作。
   - 用户对 DeepSeek 何时能在无需额外设置的情况下完全运行的时间表感到好奇。
- **本地与远程模型查询**：关于本地运行模型与远程运行模型的讨论表明，在模板设置和本地文件功能方面存在挑战。
   - 一些用户尝试将模型与本地文档连接，但遇到了语法错误并需要调整配置。



**提到的链接**：<a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF · Hugging Face</a>：未找到描述内容

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1332820683201183825)** (1 条消息): 

> `高级 LLM Agents MOOC, 直播安排, 课程网站, 报名信息, 结业证书` 


- **高级 LLM Agents MOOC 即将开课**：**高级 LLM Agents MOOC** 将于 **1 月 27 日下午 4:00 (PST)** 开始，并持续到 **4 月 28 日**。
   - 你需要的一切，包括直播链接和家庭作业，都可以在 [课程网站](http://llmagents-learning.org/sp25) 上找到。
- **直播安排公布**：从 1 月 27 日开始，我们将在每个 **周一的下午 4:00 - 6:00 (PST)** 为每位客座讲师举办直播。
   - 第一个 [直播链接](https://www.youtube.com/live/g0Dwtf3BH-0) 将在课程网站的教学大纲中分享。
- **立即报名课程**：现在报名还不晚 —— 你可以 [在这里注册](https://forms.gle/9u6HdVCWXgws16go9)。
   - 问题或反馈应在 <#1280370030609170494> 中向课程工作人员提出。
- **结业证书详情即将公布**：关于 **结业证书要求** 的更多信息将很快发布。
   - 请注意，第一周的课程 **没有截止日期**。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1332446684688551958)** (47 条消息🔥): 

> `证书发放、MOOC 报名确认、课程时区参与、线下出席、Hackathon 参与` 


- **证书仍待处理**：成员们讨论了 **MOOC Fall'24 证书**的状态，目前**尚未发放**；相关公告即将发布。
   - *感谢您的耐心等待！*
- **MOOC 报名困惑**：几位成员对提交申请后未收到课程报名的**确认邮件**表示担忧。
   - 一位成员在谈到缺乏更新时表示：*我也一样...*
- **不同时区的参与**：参与者询问了在**不同时区**参加讲座的问题，并得到确认，所有课程在直播后都可以在 **YouTube** 上观看。
   - 一位成员特别询问：*我可以离线模式学习讲座吗？*
- **无线下出席机会**：已确认 MOOC 学生**没有机会**线下参加讲座，由于名额有限，将优先考虑 Berkeley 学生。
   - 一位成员请求会见讲师，但被告知受政策限制。
- **对 Hackathon 机会的兴趣**：有人就即将到来的课程中潜在的 **Hackathon 机会**提出了请求，讲师们对这一兴趣表示认可。
   - 成员们对未来的 Hackathon 充满期待，引发了热烈的讨论。



**提及的链接**：<a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/)** (1 条消息): 

interdimensionalbeing_: https://substack.com/home/post/p-154577981
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1333183689747595426)** (14 条消息🔥): 

> `梯度计算困惑、STRIDE 与 FLIP 的讨论、第 55 次会议议程、TinyChat 中的资产获取问题、RISC 架构咨询` 


- **梯度文档需要澄清**：关于 `Tensor.gradient` 的文档出现了混淆，文档称 *“计算 targets 相对于 self 的梯度”*，但在调用上下文中似乎不准确。
   - 为了清晰起见，建议修改为 *“计算 self 相对于 targets 的梯度”*。
- **从 STRIDE 转向 FLIP**：有人建议用 **FLIP** 替换 **STRIDE**，因为 STRIDE 这个术语被认为过于通用。
   - 这一更改旨在提高命名规范的特异性和清晰度。
- **第 55 次会议日程安排**：第 55 次会议定于圣地亚哥时间周一早上 6 点举行，讨论的主题包括**公司更新**、新的 multi 和 gradient 以及项目悬赏。
   - 重点关注领域将包括 **resnet** 和 **bert** 等模型，以及特定项目的讨论。
- **TinyChat 获取字体资产出现问题**：在 TinyChat 中获取字体资产时遇到问题，因为某些资产未包含在示例脚本 `fetch_assets.sh` 中。
   - 一位用户表示愿意寻找缺失的文件以解决问题。
- **关于现场部署 RISC 架构的咨询**：一位新人询问了运行**现场部署模型**的 RISC 架构选项，目标最大权重为 8G。
   - 目标是为超紧凑的物理部署寻找解决方案。

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1332570120685486141)** (18 条消息🔥): 

> `BobNet 澄清、Tinygrad 中的格式化工具、Tinygrad 价值化计划、Tinygrad 学习资源、Tensor UOp 操作` 


- **澄清 BobNet 的名称起源**：一位用户对模型 **BobNet** 的命名提出了疑问，指出它与“Bounding Box Network”的备注不符，因为它看起来更像是一个普通的前馈网络。他们引用了其 [GitHub 链接](https://github.com/qurAI-amsterdam/bobnet) 以获取更多上下文。
   - *它是一个 Bounding Box Network 吗？* 区分命名约定和实际功能至关重要。
- **讨论 Tinygrad 格式化工具**：一位用户询问了 Tinygrad 使用的官方格式化工具，并指出与 **Black** 和 **Ruff** 存在不一致。另一位成员确认使用的是 **ruff**，并提供了相关 [pre-commit 配置](https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7) 的链接。
   - 这为贡献者提供了首选工具的信息，有助于保持代码格式的一致性。
- **关于 Tinygrad 运行时支持的讨论**：一位用户提出了关于 **Tinygrad** 未来支持 **NV=1 和 CUDA=1** 运行时的疑问。这引发了关于探索 Tinygrad 与 CUDA 关系的价值讨论。
   - 确定 Tinygrad 的长期路线图可能会影响用户在贡献和学习方面的决策。
- **Tinygrad 最佳入门资源**：频道中推荐了一个特定的 [GitHub 仓库](https://github.com/mesozoic-egg/tinygrad-notes/tree/main)（Tinygrad 笔记），被认为是“目前发现的最好的入门教程”。这可以帮助新手更轻松地理解该框架。
   - 不断增长的资源列表可以提高那些渴望深入研究 Tinygrad 和机器学习的初学者的可访问性。
- **探索 Tensor UOp 功能**：针对 `tensor.py` 中不同方法的使用提出了询问，例如 **_broadcasted()**、**_apply_uop()** 和 **_apply_broadcasted_uop()**，特别是关于 **Tensor.lshift** 的部分。不过有成员指出，修改其定义会导致大量错误，需要进一步澄清。
   - 讨论强调了简化这些操作的潜力，但也指出了实现中的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/.pre-commit-config.yaml#L7">tinygrad/.pre-commit-config.yaml at master · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/qurAI-amsterdam/bobnet">GitHub - qurAI-amsterdam/bobnet: PyTorch implementation of the Bounding Box Network (BoBNet) from the ConvNet-Based Localization of Anatomical Structures in 3D Medical Images paper.</a>：来自 3D 医学图像中基于 ConvNet 的解剖结构定位论文的 Bounding Box Network (BoBNet) 的 PyTorch 实现。 - qurAI-amsterdam/bobnet</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>：Tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1332519377437986977)** (5 条消息): 

> `GPU 效率、WSL 虚拟化、Regex 格式错误、EBNF 语法` 


- **CPU 训练耗时极长**：一位成员分享了他们在 CPU 上运行训练过程的经历，指出尽管 Python 的 **CPU 利用率为 60%**，但训练在 **step 0/76** 停滞了数小时。
   - *直到你无法使用 GPU 时，你才会意识到 GPU 的效率有多高。*
- **寻找 WSL 虚拟化选项**：在重新审视 WSL 后，一位成员在主板设置中发现了 **虚拟化选项**，令人惊讶的是它被隐藏在“超频 (Overclocking)”选项下。
   - *既然我已经把它跑起来了，速度应该会有显著提升。*
- **Regex 失误**：围绕 Regex 展开了讨论，一位成员评论说 **Regex 看起来就像是格式错误的 Tokenizer**。
   - 这促使另一位成员分享了他们从 Regex 转向其他方案的经历。
- **转向 EBNF 语法**：一位成员指出，他们已经停止使用 Regex，转而使用 **EBNF 语法**，认为其可读性更好。
   - *它们虽然更冗长，但具有更好的人类可读性。*


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1332838739600216096)** (6 条消息): 

> `Federated Learning with Torchtune, Selective Application of Optimizer Hooks, Using torch distributed primitives, Managing Optimizer States` 


- **Fed Learning：有效管理分片**：有人建议在 Federated Learning 中为**每个节点创建 N 个分片**，每个实体在一个分片上进行训练，然后在开始下一个分片之前合并权重并使用保存的 Optimizer State。
   - 这种方法引发了关于效率以及是否可能避免 **Torchtune** 训练中断的疑问。
- **利用 Torch Distributed 提升效率**：提议使用 **torch distributed primitives** 作为优化 Federated Learning 的手段，允许在全局同步到 rank 0 之前在进程组内进行更新。
   - 此外，还提到使用 **raylib** 来编排外部并行性，以改进管理。
- **通过 Backward Hooks 释放增益**：有人提出了关于选择性应用 **opt-in backward hooks** 进行参数更新可能带来的**性能增益**问题。
   - 讨论了是否在梯度就绪时立即对某些参数执行 step，而其他参数则等待完整的 Optimizer Step。
- **Optimizer 和参数选择策略**：考虑智能地将 Optimizer 仅应用于某些参数（如 output projection），以便更快地清除梯度。
   - 然而，有人对支持 **main optimizer** 以及针对参数子集的独立 Optimizer 的复杂性表示担忧，同时还提到了梯度累积（grad accumulation）的典型挑战。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1333489666090991616)** (10 条消息🔥): 

> `Deepseek, Nvidia Stock Experiences, Market Sentiment, Investment Strategies, Comparison of AI Models` 


- **Deepseek 的更新频率需要缓一缓**：一位用户对 **Deepseek** 的快速更新节奏表示沮丧，并引用了[此处](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)的一份详细技术报告。
   - 该报告重点介绍了 **Janus-Series**，旨在实现统一的多模态理解与生成。
- **Nvidia 股票引发用户遗憾**：一位用户哀叹他们的 Nvidia 投资，担心一周内可能出现第二次失败，导致用户之间展开了一场轻松的“甩锅游戏”。
   - 另一位用户幽默地建议，他们会通知其他人，以便其他人可以做空他们未来的购买。
- **市场触发因素与情绪**：讨论围绕近期市场趋势与日本利率挂钩展开，而一些用户错误地将这些波动归咎于 **Deepseek**。
   - 聊天中指出反向操作流行市场情绪的重要性，并强调在接受财务建议时要谨慎。
- **投资方法建议**：一位用户建议始终与市场情绪反向操作作为一种潜在策略，而另一位用户则警告不要向开发者寻求财务建议。
   - 关于豆豆娃（beanie babies）等非传统投资的笑话凸显了讨论的幽默基调。
- **模型对比受到审查**：一位用户质疑用于对比的模型是否已经过时，这表明在关于 **Deepseek** 的讨论中，人们关注的是当前技术。
   - 与此询问相关的一张图片进一步引发了关于模型评估相关性和及时性的辩论。



**提到的链接**：<a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>：Janus-Series: Unified Multimodal Understanding and Generation Models - deepseek-ai/Janus

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1332634100736983040)** (12 条消息🔥): 

> `项目开发状态、网站更新、Python Interpreter 功能、最新开发版本、用户反馈与建议` 


- **项目开发状态确认**：一名成员确认项目的最后一次 commit 是在**昨天**，表明开发仍在持续进行。
   - 这向社区保证，尽管存在疑虑，进度仍在推进。
- **网站正处于最小化变更阶段**：针对询问，一名成员评论称，在当前开发阶段，网站维持在**最小化状态（minimal state）**。
   - 他们承诺在正式发布后会进行更新。
- **1.0 版本将包含 Python Interpreter**：一名成员分享称，即将发布的 **1.0** 版本将整合相同的 Python Interpreter，同时过渡到新的 bash 工具。
   - 他们指出，对于需要持续运行 Python Interpreter 的操作，这种方法在 **token 效率**上表现出色。
- **获取最新开发版本**：一名成员请求最新版本的下载链接，并获得了指向 **GitHub repository** 的链接。
   - 他们分享了使用 `pip install` 的安装命令以便于访问。
- **用户交互增强建议**：有反馈建议通过实用函数扩展 AI 的能力，以减少重复性任务。
   - 建议包括更好的**内部文档**以及 AI 编辑自身指令的能力。



**提到的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>：计算机的自然语言界面。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1332953693640986655)** (2 条消息): 

> `Deepseek_r1, API 错误` 


- **Deepseek_r1 模型遇到问题**：一名用户尝试使用 **Deepseek_r1** 模型，配置如下：`llm: model: "Deepseek_r1" temperature: 0 api_key: "sk-d....." api_base: "https://api.deepseek.com"`。
   - 然而，模型返回了 **400 错误**，提示该模型不存在，导致工作流出现困惑。
- **API 调用出现 BadRequestError**：在使用命令 `$ interpreter -y --profile deepseek.yaml` 运行 interpreter 时报告了一个错误，导致 `BadRequestError`。
   - 错误详情将其指定为 **OpenAIException**，指向由于模型缺失导致的 **invalid_request_error**。



**提到的链接**：<a href="https://api.deepseek.com"```">未找到标题</a>：未找到描述

  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1332492119574712382)** (4 条消息): 

> `DeepSeek 模型, Open Interpreter 本地设置, AI 终端应用开发, Vision 模型讨论` 


- **DeepSeek 模型展现出极具前景的性能**：DeepSeek 的第一代推理模型，特别是 **DeepSeek-R1**，在包括数学和代码在内的各种任务中，表现出与 OpenAI-o1 相当的性能。
   - 较小的蒸馏模型，如 **DeepSeek-R1-Distill-Qwen-1.5B**，也被证明取得了出色的 Benchmark 结果。
- **本地运行 Open Interpreter**：Open Interpreter 可以完全在本地运行，允许用户集成多个本地模型提供商，如 [Ollama](https://www.ollama.com/) 和 [Llamafile](https://github.com/Mozilla-Ocho/llamafile)。
   - 用户可以使用 Local Explorer 功能并通过命令 `interpreter --local` 来简化本地设置过程。
- **DeepSeek 在 OS 模式下的挑战**：一位成员分享道，OS 模式目前缺乏与 **DeepSeek** 的集成，因为它需要 tool calling 和 vision model 才能正常运行。
   - 目前似乎正在考虑寻找一种变通方法，以便在这种情况下有效地利用 DeepSeek。
- **关于 Vision Model 功能的讨论**：针对在多模型设置中启用 vision model 提出了疑虑，质疑其在系统中的必要性。
   - 对话表明，需要明确 vision model 在当前配置中是如何运行的。
- **征集 DSH - AI Terminal 的贡献**：一个名为 **DSH - Ai terminal** 的开源项目正在寻求贡献者，以增强应用、改进功能并获取反馈。
   - 鼓励参与者在 [GitHub](https://github.com/gokul6350/dsh-shell) 上为该项目点亮 Star 以支持其开发，并查看提供的截图以获取视觉概览。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ollama.com/library/deepseek-r1">deepseek-r1</a>：DeepSeek 的第一代推理模型，性能堪比 OpenAI-o1，包括六个基于 Llama 和 Qwen 从 DeepSeek-R1 蒸馏出的稠密模型。</li><li><a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1332626872441372682)** (12 条消息🔥): 

> `DeepSeek R1 性能, 音频增强工具, Pipeline 测试` 


- **DeepSeek R1 表现不及预期**：**DeepSeek R1** 的初步对比表明，它可能仅与 **o1-mini** 和 **Claude 3.5 Sonnet** 相当，这与它在艰难的 Benchmark 上媲美 **o1** 的说法相矛盾。这一评估让人对其处理揭示泛化缺陷的 AIW 问题（AIW problems）的能力产生怀疑，详见[此处](https://x.com/JJitsev/status/1883158738661691878)。
   - *又一个关于兴衰的故事*引发了人们对 DeepSeek R1 在奥数级挑战中表现的担忧，因此有必要[在这篇论文中](https://arxiv.org/abs/2406.02061)调查其实际能力。
- **请求在 Pipeline 中增强音频工具**：一位成员建议添加 **audio widgets**，以便更好地比较增强效果对声音变化的影响。他们还建议从 **DeepSeq** 或 **O1** 等库中获取更多样化的失真和噪声。
   - 该反馈旨在完善正在开发的 Pipeline，强调需要能让用户更具交互性地体验音频变化的功能。
- **Pipeline 测试和开发进度**：一位用户在结束一天的旅程后分享了他们正在开发阶段的测试 Pipeline 链接。链接可以在[此处](https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt?usp=sharing)找到。
   - 他们鼓励频道中的其他人对该 Pipeline 提供反馈，希望能进一步完善功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tc4YgdsZeEtsZCdnawYaEC7b12NBQfYt?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://x.com/JJitsev/status/1883158738661691878">Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：(又) 一个关于兴衰的故事：DeepSeek R1 声称在奥数级数学和编码问题上能媲美 o1/o1-preview。它能处理揭示泛化和基础能力的 AIW 问题变体吗...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1332475950696501258)** (2 messages): 

> `DeepSeek R1, AIW 版本对比, 基准测试性能` 


- **DeepSeek R1 的主张受到质疑**：使用 AIW 版本的初步对比结果表明，尽管此前有相关主张，**DeepSeek R1** 在严苛的基准测试中并未达到或超越 **o1**。
   - 目前，**DeepSeek R1** 似乎与 **o1-mini** 和 **Claude 3.5 Sonnet** 持平，这引发了对其处理挑战性问题效能的疑问。
- **DeepSeek R1 在复杂任务上的局限性**：有人担心 **DeepSeek R1** 是否能处理那些暴露了领先 **LLM** 在泛化和推理方面差距的 **AIW 问题**版本。
   - 根据一则帖子，**DeepSeek R1** 是否能有效应对奥数级别的数学和编程挑战仍不确定。
- **社交媒体对 DeepSeek R1 的热议**：X 上的一则帖子指出，围绕 **DeepSeek R1** 的叙事展示了其在针对既有模型的性能主张上的起伏。
   - 持续的讨论反映了对其应对严谨数学和编程任务能力的怀疑。



**提到的链接**：<a href="https://x.com/JJitsev/status/1883158738661691878">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：(又) 一个关于起落的故事：DeepSeek R1 被声称在奥数级数学和编程问题上可与 o1/o1-preview 媲美。它能处理那些揭示泛化和基础...的 AIW 问题版本吗？

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1333164853799555082)** (8 messages🔥): 

> `GitHub issue 垃圾信息, 自然语言 vs 编程, dspy + deepseek 优化, pypi 版本更新` 


- **GitHub 充斥着垃圾 issue**：一位成员注意到，由于随机的垃圾提交，**GitHub 的 issues 板块**目前看起来状况糟糕。
   - *似乎有人正在 GitHub 上发送大量随机 issue 垃圾信息*。
- **寻求语言区分模型**：一位用户询问是否存在一种模型（不限于 **LLM**），能够区分**自然语言**和编程、HTML 等代码格式。
   - 这表明用户正在寻找能更好分类文本内容类型的工具。
- **dspy 与 deepseek 优化的挑战**：一位成员询问是否有人成功为 **COT** 示例运行了 **dspy + deepseek** 70B 优化。
   - *“针对 COT 优化”是什么意思？* 这一提问引发了关于该功能的进一步讨论。
- **对收敛时间过长的担忧**：一位成员报告称运行 **BSR 示例**六小时仍未收敛，引发了对流程耗时的担忧。
   - 对话暗示了对优化过程的挫败感。
- **请求发布新的 pypi 版本**：一位用户表示沮丧，因为 **pypi** 上最新的 **RC** 版本由于依赖项过时，无法与现代 **FastAPI** 应用良好协作。
   - *当前问题已在 3 周前的 main 分支上修复*，因此需要向 pypi 推送新版本。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1333249619383029852)** (7 messages): 

> `deepseek 算法, H200 vs 5090s, RL 框架支持` 


- **关于 Deepseek 算法实现的讨论**：成员们正在询问 **deepseek 算法**的实现，其中一位成员询问是否有人正在复现它。
   - 另一位成员建议这可能指的是最近添加到 **trl** 中的 **grpo**。
- **GPU 选项对比：H200 vs 5090s**：一位成员正在考虑购买 **2x 5090s** 或 **1x H200**，并指出 H200 拥有更多显存但可能速度较慢。
   - 尽管 H200 具有优势，但该咨询凸显了对其相对于 5090s 性能的不确定性。
- **关于扩展 RL 支持的咨询**：一位成员注意到目前缺乏对 **trl 的在线 RL 训练器**的支持，但表示有兴趣扩展 **RL** 框架支持。
   - 他们邀请用户反馈希望集成哪些其他 **RL** 框架。
- **RL 支持未来扩展有限**：在回答关于稍后扩展 **RL** 支持的问题时，一位成员表示这**极有可能不会**发生。
   - 这表明了对目前缺乏特定算法支持的坚定立场。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1332928201902788740)** (3 messages): 

> `排行榜模型的 System Prompts，Gorilla GitHub 仓库` 


- **寻找非 Function Calling 模型的 System Prompts**：一位成员询问了排行榜中不支持 Function Calling 的模型所使用的 System Prompts 的位置。
   - 另一位成员迅速分享了一个 [GitHub 仓库链接](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18)，其中包含相关代码供参考。
- **Prompt 详情的资源链接**：分享的 GitHub 链接指向了 [Gorilla Function Call 排行榜的代码](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py)，其中定义了 System Prompts。
   - 该仓库专注于训练和评估 LLM 的 Function Calls 能力，为用户提供了丰富的资源。



**提及的链接**：<a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py#L3-L18">gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1333198192572895363)** (1 messages): 

> `2025 水晶球小组讨论，实时数据处理，AI 与数据流技术，行业领袖与见解` 


- **2025 水晶球：利用 AI 塑造未来**：**1 月 28 日**加入行业领袖的专题讨论 **2025 Crystal Ball: Real-Time Data and AI**，重点关注实时数据在增强 AI 能力方面的变革作用。
   - *如果没有实时数据处理，AI 的真正潜力将无法得到开发*，强调了对促进低延迟预测技术的需求。
- **来自 AI 领袖的专家见解**：小组专家包括 **Rayees Pasha (RisingWave Labs)**、**Sijie Guo (StreamNative)** 和 **Chang She (LanceDB)**，他们将讨论 AI 和数据流领域的关键进展。
   - 讨论将涵盖 AI 与实时数据之间**不断演变的关系**，以及对 2025 年即将到来的挑战和进步的预测。
- **深入探讨实时处理技术**：预计将**深入探讨关键技术**，如 Apache Iceberg，强调它们在彻底改变数据基础设施和 AI 效率方面的作用。
   - 专家们将解释尖端的流处理和实时分析如何为各行各业带来**突破性的业务影响**。
- **释放 AI 的真正潜力**：缺乏实时数据处理能力，AI 的真正潜力就无法实现，这强调了数据处理创新的必要性。
   - 本次活动旨在剖析 AI 系统如何利用实时数据解决现实世界的挑战并开启新机遇。



**提及的链接**：<a href="https://www.meetup.com/streaming-stories/events/305736950/">2025 Crystal Ball: Real-Time Data and AI, Tue, Jan 28, 2025, 9:00 AM   | Meetup</a>: **关于**在 2025 Crystal Ball: Real-Time Data and AI 专题讨论中展望数据流和 AI 的未来。毫无疑问，AI 将塑造我们未来的岁月。

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1333489291195846758)** (1 messages): 

> `论文研读会 (Paper Reading Club)，Discord 活动` 


- **本周论文研读会！**：**Paper Reading Club** 本周将再次聚会。查看 [Discord](https://discord.com/events/1089876418936180786/1329844319703662664) 上的活动详情！
- **Discord 活动提醒**：别忘了探索 **Discord** 上发生的精彩活动，包括 Paper Reading Club。加入对话并参与本周的社区活动！


  

---


---


{% else %}


> 完整的频道细分内容已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}