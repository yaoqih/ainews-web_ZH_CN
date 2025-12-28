---
companies:
- meta-ai-fair
- cerebras
- groq
- alibaba
- vllm
- ollama
- llamaindex
- hugging-face
- llama-cpp
date: '2025-04-29T05:44:39.731046Z'
description: '**Meta** 在 LlamaCon 大会上庆祝了 **Llama** 生态系统的进展，并推出了一个 AI 开发者平台。该平台提供由
  **Cerebras** 和 **Groq** 硬件支持的微调和快速推理功能，不过目前仍处于候补（waitlisted）状态。


  与此同时，**阿里巴巴**发布了 **Qwen3** 系列大语言模型，包括 **两个 MoE（混合专家）模型**和 **六个稠密模型**，参数规模从 **6 亿到
  2350 亿**不等。其中旗舰型号 **Qwen3-235B-A22B** 在基准测试中取得了极具竞争力的成绩，并支持 **119 种语言和方言**。Qwen3
  模型针对编程和智能体（agentic）能力进行了优化，采用 **Apache 2.0** 协议授权，并拥有广泛的部署支持，包括使用 **vLLM**、**Ollama**
  和 **llama.cpp** 等工具进行本地部署。社区反馈强调了 Qwen3 的可扩展性能，以及其相对于 OpenAI **o3-mini** 等模型的优越性。'
id: MjAyNS0w
models:
- llama-4
- qwen3
- qwen3-235b-a22b
- qwen3-30b-a3b
- qwen3-4b
- qwen2-5-72b-instruct
- o3-mini
people:
- reach_vb
- huybery
- teortaxestex
- awnihannun
- thezachmueller
title: LlamaCon：Meta AI 进军 Llama API 平台业务。
topics:
- model-release
- fine-tuning
- reinforcement-learning
- moe
- multilingual-models
- model-optimization
- model-deployment
- coding
- benchmarking
- apache-license
---

**Llama API 就足够了？**

> 2025年4月29日至4月30日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（包含 214 个频道和 5096 条消息）。预计节省阅读时间（按 200wpm 计算）：442 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

/r/localLlama [爱上了](https://www.interconnects.ai/p/qwen-3-the-new-open-standard) [昨天的 Qwen 3](https://news.smol.ai/issues/25-04-28-qwen-3)，但今天属于 Llama。

尽管有一些关于 [Llama 4 推理模型的新传闻](https://x.com/btibor91/status/1917232574344384522)，但 LlamaCon 最终成为了一场 [相对波澜不惊的庆典](https://ai.meta.com/blog/llamacon-llama-news/)，展示了 Llama 生态中不可否认的进步。Zuck 再次做客 Dwarkesh 的访谈，讨论了备受争议的 Llama 4 发布（[我们的报道](https://news.smol.ai/issues/25-04-07-ainews-llama-4s-controversial-weekend-release)）：

https://www.youtube.com/watch?v=rYXeQbTuVl0

对于 AI Engineers 来说，此次活动中另一个值得注意的更新是 Meta 首次推出了一个 AI 开发者平台，可以说相当于 Google 的 AI Studio，具备 finetuning 能力，并支持通过 Cerebras 和 Groq 进行快速推理，尽管目前仍处于候补名单阶段：


![](https://resend-attachments.s3.amazonaws.com/nDIFJugV5oxnd9O)


---

# AI Twitter 综述

**Qwen3 模型发布与性能**

- **Qwen3 模型发布了！**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962087676612998) 宣布发布并开源 **Qwen3**，这是他们最新的大语言模型（LLM），包括 **两个 MoE 模型和六个稠密模型**，参数量从 **0.6B 到 235B** 不等。旗舰模型 **Qwen3-235B-A22B** 在基准测试评估中取得了极具竞争力的结果，而像 **Qwen3-4B** 这样的小模型甚至可以媲美 **Qwen2.5-72B-Instruct**。用户可以在 Qwen Chat 网页版和 APP 中体验这些模型，也可以在 GitHub、HF 和 ModelScope 上获取。建议使用 SGLang 和 vLLM 等框架进行部署，本地使用则推荐 Ollama、LMStudio、MLX、llama.cpp 和 KTransformers 等工具。
- **Qwen3 模型可与领先模型媲美**：据 [@huybery](https://twitter.com/huybery/status/1916962562056524177) 称，团队在 **Qwen3** 上投入了巨大努力，希望为开源 LLM 社区带来新鲜血液，在 **预训练（pretraining）、大规模强化学习（large-scale reinforcement learning）以及推理模式集成** 方面取得了显著进展。
- **Qwen3-30B-A3B 性能强劲**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916966009170251899) 提到 **Qwen3-30B-A3B** 实际上与 **Qwen3-32B 稠密版** 旗鼓相当，称其为细粒度 MoE（fine-grained MoEs）的最佳证明。
- **Qwen3 架构与能力**：[@reach_vb](https://twitter.com/reach_vb/status/1916965315910553886) 强调了 **Qwen3 235B MoE (22B Active)** 的发布，指出它击败了 **R1、Grok、O1**，并采用 Apache 2.0 协议授权。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1916966249588002867) 总结了 Qwen3 发布的主要特点，包括 **性能、训练数据、思考模式、Agent 和代码能力以及 Apache 2.0 协议**。
- **Qwen3 支持多种语言**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962096346202468) 表示 **Qwen3** 模型支持 **119 种语言和方言**，这为国际化应用开辟了新的可能性。
- **Qwen3 模型的性能提升**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962091925442698) 指出 **Qwen3** 表现出可扩展且平滑的性能提升，这与分配的计算推理预算（computational reasoning budget）直接相关。
- **Qwen3 为代码优化**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962100817367192) 强调了 **Qwen3** 模型针对代码和 Agent 能力进行了优化，加强了对 MCP 的支持。
- **本地运行 Qwen3**：[@TheZachMueller](https://twitter.com/TheZachMueller/status/1916969775525191684) 提供了一个快速测试新模型的工作流，包括设置 vLLM 以提供带有推理能力的 Qwen 模型服务。
- [@vllm_project](https://twitter.com/vllm_project/status/1917008899410215275) 宣布对 **Qwen3** 和 **Qwen3 MoE** 模型架构提供首日支持，并附带了使用说明。
- [@AwniHannun](https://twitter.com/AwniHannun/status/1916862553852203349) 报告称最新的 mlx-lm 已经支持 **Qwen3** 和 **Qwen3 MoEs**，并指出从 iPhone 到 M2、M3 Ultra 的每种设备都有适用的模型。
- [@scaling01](https://twitter.com/scaling01/status/1916967634786029722) 报告称 **Qwen3-235B-A22B** 在所有基准测试中均优于 **OpenAI 的 o3-mini**。
- [@skypilot_org](https://twitter.com/skypilot_org/status/1916987145195295095) 宣布支持 Qwen3，只需一条 SkyPilot 命令即可在您的集群或云端轻松启动 Qwen3。

**Qwen3 的评估、基准测试与分析**

- **Qwen3 的初步印象**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916918829050998981) 预测 **Qwen 30B-3A** 将成为全场焦点。
- **Qwen3 蒸馏**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916971319800823932) 指出 **Qwen 的蒸馏教师 MoE** 的总参数量（235B）比 Meta 的 Llama 4 Behemoth 的激活参数量（288B）还要少。
- **Qwen3 与 DeepSeek 的对比**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916824004901359943) 认为 **Qwen-3-MoE** 与 **DeepSeek V2** 在设计上的相似性将为 Scaling Laws 提供一个非常有趣的测试。
- **Qwen3 相对于 DeepSeek 的性能**：[@scaling01](https://twitter.com/scaling01/status/1916986267700506700) 认为 **Qwen3-235B Base** 似乎受益于其 94 层的设计，相比之下 **Llama-4 Mavericks 为 48 层**，而 **DeepSeek 为 61 层**。
- **Qwen3 代码 Agent 基准测试**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1917064282552078480) 评估了 **Qwen3-235B-A22B** 在开源编程 Agent Openhands 上的初步表现，在 SWE-bench Verified 上达到了 34.4%。
- [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1917246369510879280) 提供了 **Qwen3** 模型系列的分析，指出 **253B-A22B 的规模约为 DeepSeek R1 的 1/3，激活参数约为 60%，且具有相当的 GPQA 分数。**
- [@scaling01](https://twitter.com/scaling01/status/1917126148623921273) 指出 **Qwen 在 SWE-bench Verified 上的表现欠佳**。

**Google Gemini 的更新与功能**

- **Google DeepMind 演示了 Gemini 2.5 Pro**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1916850709300969613) 分享了一个演示，展示了 **Gemini 2.5 Pro** 如何通过编写强化学习算法、实时可视化训练过程以及调试错误，来实现一篇具有里程碑意义的 Google DeepMind 研究论文。
- **Gemini 从照片生成 3D 世界**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1916799144385011779) 在 60Minutes 节目中展示了 **Genie 2 的图像转 3D 世界创建能力**，探索了它为 AI 学习方式带来的可能性。
- **Gemini 2.5 Pro 代码生成**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1917221723834827196) 分享了 Gemini 2.5 Pro 使用 Three.js 进行 Vibe-coding 开发 3D 蛋糕可视化工具的演示，包括构建自定义动画和 UI 控件，并根据反馈更新视觉效果。
- **Gemini 与 LangChain**：[@_philschmid](https://twitter.com/_philschmid/status/1916856375704985661) 制作了一份使用 **Gemini 2.5 配合 LangChain 和 LangGraph** 的速查表，涵盖了基础聊天、多模态输入、结构化输出、工具调用（Tool Calling）和 Embeddings。

**ChatGPT 更新与购物功能**

- **ChatGPT 购物功能现已上线**：[@OpenAI](https://twitter.com/OpenAI/status/1916947241086095434) 宣布在 ChatGPT 中推出更好的购物体验，包括改进的产品搜索结果、视觉化产品详情、价格信息以及直接购买链接。产品结果是独立选择的，并非广告，适用于 Plus、Pro、Free 以及未登录用户。
- **ChatGPT 搜索的新增强功能**：[@OpenAI](https://twitter.com/OpenAI/status/1916947244852646202) 强调了 ChatGPT 搜索的多项改进：WhatsApp 中的搜索、改进的引用、趋势搜索和自动补全建议。

**Runway References 与 Gen-4 图像生成**

- **用于图像生成的 Runway References**：[@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1916783390252007908) 讨论了 Runway References 作为首个 One-shot 工具，能够准确复制肖像特征并达到接近 LoRA 的质量。

**AI 安全与伦理**

- **RLHF 的阴暗面**：[@jd_pressman](https://twitter.com/jd_pressman/status/1916909455566115121) 表示，RLHF 在语言模型领域变得与 RL 同义是非常不幸的，这给人类反馈（human feedback）带来了负面声誉。
- **OpenAI 反馈循环缺陷**：[@nearcyan](https://twitter.com/nearcyan/status/1916737662020723187) 指出，“当 OpenAI ‘修复’ ChatGPT 时，我建议你不要上当；他们的目标和关注程度不会改变。你只是本不该如此清晰地察觉到它。”
- **言论自由至上主义**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916663075731607725) 认为，当制造共识成为一种可行策略时，言论自由至上主义就是一种愚蠢的教条，并指出美国人会对针对个人的诽谤施加成本，但对针对整个生活方式的诽谤则不然。
- [@alexalbert__](https://twitter.com/alexalbert__/status/1916878483390869612) 指出，AI 行业正陷入一个特别有害的反馈循环，盲目追求更高的人类偏好评分，这是操纵用户而非为他们提供真正价值的药方。
- **AI 超级说服（Hypersuasion）的伦理**：[@paul_cal](https://twitter.com/paul_cal/status/1916931024434696555) 报告称，“社区对 Reddit 上这项 AI 超级说服研究感到非常愤怒”，研究显示充当创伤辅导员或假装强奸受害者“对于一个公共机构来说并不是一个好的形象”。

**多 Agent 系统与 LangGraph**

- **DevOps 工作流中的 Agent**：[@LangChainAI](https://twitter.com/LangChainAI/status/1917283909706080472) 报道了思科利用 LangGraph 的力量为 DevOps 工作流带来智能自动化。
- **用于 Agentic 系统的 LangGraph**：[@hwchase17](https://twitter.com/hwchase17/status/1917256353602756670) 分享道，人机回环（human in the loop）对于构建可信的 Agentic 系统至关重要，而 LangGraph 正在为此构建基础设施。
- **多 Agent 架构**：[@hwchase17](https://twitter.com/hwchase17/status/1917292257461559503) 正在研讨关于多 Agent 架构的激进观点，区分了聊天 Agent（chat agents）和任务 Agent（task agents），并考虑了 Agent 协议的适用性。
- [@_philschmid](https://twitter.com/_philschmid/status/1917209995923370305) 讨论了为 AI Agent 提供有状态专用沙箱的重要性，使它们能够编写文件、执行命令、调用工具并控制 UI 应用。

**Cursor 与 AI 辅助编程**

- **Cursor 的 AI 编程助手**：[@amanrsanger](https://twitter.com/amanrsanger/status/1916968123535880684) 强调 Cursor 每天编写近 10 亿行被采纳的代码，这占全球代码总产量的很大一部分。
- **Cursor 生成 Figma 设计**：[@LiorOnAI](https://twitter.com/LiorOnAI/status/1917234515753177318) 分享称，Cursor 现在可以通过 Figma 新的 MCP 服务器以编程方式读取和编辑 Figma 文件，从而生成 Figma 设计。
- **使用 AI 编程**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1916871403497811970) 展示了程序员如何使用 AI 的证据，指出 AI 被不成比例地用于软件开发工作，使其成为 AI 可能如何改变其他职业的领先指标。

**Llama API、工具与生态系统**

- **Meta 发布 Llama API**：[@AIatMeta](https://twitter.com/AIatMeta/status/1917278290441822674) 宣布 Llama API 进入预览阶段，提供闭源模型 API 的最佳特性以及开源的灵活性。
- **利用 Llama 推进 AI 安全**：[@AIatMeta](https://twitter.com/AIatMeta/status/1917271400118902860) 为防御者社区分享了新的开源 Llama 防护工具和 AI 驱动的解决方案。
- **Llama 影响力资助计划**：[@AIatMeta](https://twitter.com/AIatMeta/status/1917274585189568870) 宣布了第二届 Llama 影响力资助计划（Llama Impact Grants）的 10 位国际获奖者，该计划旨在通过开源 AI 促进创新并创造经济机会。

**其他模型与工具**

- **Freepik 和 FAL 的 F-Lite 模型**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1917244092544847914) 重点介绍了在 8000 万张由 @freepik 拥有的图像上训练的 10B 参数 DiT 模型。该模型可商用，是未经蒸馏的原始模型，并且已开源。他指出该模型是与客户 @freepik 合作的首个模型训练项目：来自 @FAL 的 "F-Lite"。

**商业、投资与经济影响**

- **AI 人才分布**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916643079584461218) 讨论了人才格局，认为中国稳健的产业政策正在基础 AI 研究方面取得成果，可能导致 2025 年中美人才净流向的崩溃或逆转。
- **Anthropic 经济顾问委员会**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1916873304914149636) 宣布成立 Anthropic 经济顾问委员会，为其经济指数（Economic Index）的新研究领域提供建议。

**幽默与杂项**

- **AI glazing**：[@dzhng](https://twitter.com/dzhng/status/1916899238245765197) 开玩笑说累的时候想要一个能让人心情愉悦的模型。
- **杰夫·贝索斯的视频**：[@vikhyatk](https://twitter.com/vikhyatk/status/1916762302155571273) 说他们的爱好之一是看杰夫·贝索斯的老视频。
- **恶搞口号**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1917272614965502179) 调侃说“Two Whatevers”对于一种意识形态来说简直是超越了恶搞。
- **“我完蛋了” (I'm cooked)**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1916844469502005371) 在没有上下文的情况下简单地说了句“我完蛋了”，可能是对某些压倒性或荒谬事物的反应。
- **Jewpiter**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1916634978806567208) 开玩笑说犹太人应该去木星 (Jupiter)，因为这个行星的名字里就带着他们。

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. Qwen3 模型发布与性能基准测试

- [**Qwen 3 !!!**](https://www.reddit.com/gallery/1ka6mic) ([Score: 1689, Comments: 419](https://www.reddit.com/r/LocalLLaMA/comments/1ka6mic/qwen_3/)): **阿里巴巴发布了开源权重的 Qwen3 系列，包含 2 个 MoE (Mixture-of-Experts) 模型和 6 个参数量从 0.6B 到 235B 的 Dense 模型。旗舰模型 Qwen3-235B-A22B 在编程、数学和通用基准测试中，对比 DeepSeek-R1, o1, o3-mini, Grok-3 和 Gemini-2.5-Pro 等顶级模型，展现出了顶尖 (state-of-the-art) 的性能。他们的小型 MoE 模型 Qwen3-30B-A3B 尽管激活参数量减少了 10 倍，但在性能上超过了 QwQ-32B；而紧凑型的 Qwen3-4B 据报道可以媲美规模大得多的 Qwen2.5-72B-Instruct。详情请参阅其 [GitHub](https://github.com/QwenLM/Qwen3)。有关进一步的基准测试和下载，请参考 [Hugging Face](https://huggingface.co/Qwen) 和 [ModelScope](https://modelscope.cn/models?page=1&limit=24&modelName=Qwen3)。** 顶级技术观点强调，Qwen3 的发布树立了超越 Meta 的 Llama 4 的新标准，尤其关注小型模型的编程能力和 MoE 效率。
    - 讨论指出 **Qwen3 4B 模型** 的基准测试表现远超其体量，据报道在某些任务中优于更大的模型如 **Gemma 3 27B** 和 **GPT-4o**。这引发了关于内存效率的讨论，因为 4B 大小（约 4GB 文件）意味着对 VRAM 的要求大幅降低，使得推理速度成为新的瓶颈，而非硬件限制。
    - 评论认为，鉴于 Qwen3 的表现，**OpenAI 传闻中的 o3-mini 级别开源模型** 可能在发布时就已经过时了。这意味着 Qwen3 最近的基准测试可能会对其他主要参与者计划中的开源发布实现“跨越式领先”，除非那些模型能设定更高的标杆。
    - 一位用户推测 Qwen3 的 Token 生成情况，指出它“可能生成了大量的推理 Token (reasoning tokens)”，暗示了可能存在的架构或训练优化，这些优化改变了与传统 LLM 缩放法则 (scaling laws) 相比的性能优势。
- [**Qwen3-30B-A3B 正是大多数人一直在等待的模型**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/) ([Score: 864, Comments: 179](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/)): **该帖子讨论了 Qwen3-30B-A3B 模型的发布，这是一款利用小型专家实现显著提升推理速度的混合专家 (MoE) 语言模型，相比之前的顶尖模型（特别是 QwQ）有巨大进步。值得注意的是，Qwen3-30B-A3B 在编程和 Agent 流水线任务中提供了极具竞争力的性能，同时能在消费级硬件（如游戏 GPU）上高效运行。用户基准测试显示，在 12GB VRAM 上使用 `Q6` 量化可达 `12 t/s`，在 RTX 5090 (Q4) 上可达 `140-155 tok/sec`，在双 3090 GPU 上接近 `100 tok/s`，展示了相比 QwQ 在速度和可访问性上的双重提升。** 评论强调了该模型在速度和效率上的突破，用户报告其可用性和响应速度显著优于 QwQ，这预示着在适度硬件上部署本地 LLM 的范式转变。
    - 多位用户报告 Qwen3-30B-A3B 实现了比以往模型显著更高的推理速度：一位拥有 12GB VRAM 的用户在 Q6 量化下获得了 12 tokens/sec，而 QwQ-Q5 仅为 3 tokens/sec；其他用户报告在 RTX 5090 (Q4 量化) 上达到 140-155 tokens/sec，使用双 3090 则约为 100 tokens/sec。这突显了在各种硬件配置下强大的性能和 VRAM 效率。
    - 共享了一个关于 LlamaCPP 的技术技巧：使用 `-override-tensor` 选项允许用户仅将“专家”部分卸载 (offload) 到 CPU，从而优化 GPU VRAM 的使用。这使得模型可以在 12-16GB 显存的 GPU 上舒适运行，尤其是在对编程任务至关重要的高量化级别 (q6/q8) 下，让中端硬件用户更容易实现高效部署。

- 尽管为了获得最佳性能需要较高的 VRAM，但技术讨论集中在优化使用的方法上，使得 24GB VRAM 甚至更少也能产生出色的吞吐量和可用性，特别是通过 quantization（量化）和 expert offloading（专家卸载）策略。
- [**我刚刚意识到 Qwen3-30B-A3B 就是我本地 LLM 所需的一切**](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/) ([Score: 435, Comments: 157](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/)): **用户报告称，在功耗受限的 RTX 4090 上使用 LM Studio，MoE 模型 Qwen3-30B-A3B 在翻译、编程和数据分析等任务中提供了超过 100 tokens/sec 的出色通用性能，同时具有极高的 VRAM 效率（在 Q8 cache + Unsloth Q4 UD GGUF 量化下，最大上下文时仍有 4GB 剩余）。从 Ollama 和 Open WebUI 中的多个模型切换后，用户发现单个 Qwen3-30B-A3B 模型（配合现代 UI 前端）足以满足所有本地 LLM 需求，从而释放了硬件和磁盘资源。** 评论者普遍认为 Qwen3-30B-A3B 能力极强（可与 Gemma3-27B 或 GLM4-32B 媲美），但速度明显更快。文中提到了像 llama-swap 这样的替代工具，用于按任务灵活配置模型。一条批评性评论指出，Qwen3-30B-A3B 未能通过“十四行诗测试”（结构化诗歌生成），而 Gemma3 模型则始终能成功，这突显了 Qwen3 在特定能力上的差距。
    - Qwen3-30B-A3B 在通用能力方面与 Gemma3-27B 和 GLM4-32B 相比毫不逊色，多位用户指出其推理速度明显优于这两个更大的模型。该模型被描述为非常适合本地部署。
    - 通过使用 llama-swap 和最新的 llama.cpp 等工具，用户正在自定义模型配置（例如，调整每个模型运行时的 context length 和 GPU 加载层数），从而在速度和上下文容量之间进行动态权衡。这使得能够针对不同的任务或资源限制部署同一基础模型的多个实例。
    - 确定的一个技术限制是：Qwen3 在结构化创意任务（如严格的十四行诗生成）中表现不佳，与 Gemma3（包括较小的变体）甚至较旧的模型（如 dolphin-mistral）相比，它在格式、押韵和音节计数方面均告失败。还提到了 Qwen3 与 Gemma3 相比缺乏 vision（视觉）能力，限制了其在多模态任务中的应用。
- [**Qwen3-30B-A3B 在 CPU 上以 12-15 tokens-per-second 运行**](https://v.redd.it/k27mtpenipxe1) ([Score: 773, Comments: 162](https://www.reddit.com/r/LocalLLaMA/comments/1kag4er/qwen330ba3b_runs_at_1215_tokenspersecond_on_cpu/)): **Qwen3-30B-A3B 模型（一个 MoE 30B LLM）的 UnSloth Q6_K 量化版本在配备 32GB RAM 的 AMD Ryzen 9 7950x3d CPU 上实现了 12-15 tokens-per-second 的推理速度，这与现代双通道 DDR5 系统上 15-20 tps 的用户报告一致。GGUF 量化格式实现了高效的 CPU 推理，使得像 Qwen3-30B-A3B 这样的前沿 LLM 在不需要高端 GPU 的情况下，也能在消费级台式机和笔记本电脑上使用。用户提到了与更稠密的开源模型性能相当，并强调其可比肩“o3-mini”的质量水平；此外，一些讨论指出 GGUF 特有的 bug 仍可能影响输出质量。** 讨论辩论了在各种硬件（包括 Snapdragon X Elite 和 Apple Silicon）上的实际吞吐量、MoE 与更稠密模型之间日益缩小的质量差距，以及高质量本地 LLM 是否挑战了 API 交付解决方案的主导地位。一些人注意到 token 速度的可靠性波动以及系统冷热状态之间的差异。
    - 用户报告称，Qwen3-30B-A3B 在使用双通道 DDR5 的普通台式机/笔记本 CPU 上始终能达到 `12-20 tokens/s`，使其在本地推理的可用性上可与 o3-mini 等较小模型相媲美。系统内存带宽（即 DDR4 与 DDR5、双通道与四通道）被强调为关键的性能因素。
    - 该模型的效率使其能够在 RTX 4060 8GB GPU 的笔记本电脑等消费级硬件上运行，并提供接近 o1 级别的高质量输出，将其可访问性扩展到了专用服务器设置或云端 API 之外。
    - 分享了对比基准测试：Qwen3-30B-A3B 在某些设置下系统重启后不久可达到 `18-20 tokens/s`，但在长时间运行后会降至 `16 tps`；相比之下，一个大得多的 235B-A22B Q4 模型在较旧的四通道 DDR4 服务器上仅能达到 `2.39 tps`（生成了超过 5,000 个 token），这说明了 Qwen3-30B-A3B 在家庭推理方面的显著速度优势。

- [**Qwen3-30B-A3B 太神奇了。**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8n18/qwen330ba3b_is_magic/) ([Score: 233, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1ka8n18/qwen330ba3b_is_magic/)): **用户报告运行了 Qwen3-30B-A3B（Qwen-3B 的一个变体，具有** `3B active parameters`**），在仅有 4GB VRAM 的 AMD RX 6550M GPU 上达到了** `20 tokens per second`**，这与已发布的基准测试一致。热门评论讨论了在 CPU 上运行的可行性（由于其较低的 active parameters 数量），询问了用于将模型适配到有限 VRAM 中的 quantization 技术和 inference engines，并对规模大得多的 Qwen3-235B-A22B 模型的 RAM 需求表示关注。([model details](https://github.com/QwenLM/Qwen3))** 技术讨论集中在重度 quantization 或 sparsity 技巧如何促进这种性能，以及对将类似的效率扩展到 CPU 和更大型模型的兴趣，表明了对他人实测报告的关注。
    - 讨论集中在 Qwen3-30B-A3B 的 3B active parameters 在高效 CPU 推理方面的巨大潜力，这意味着与标准大型模型相比，部署所需的硬件要求显著降低。
    - 针对在仅 4GB VRAM 下运行该模型的能力提出了一个技术问题，要求提供有关所涉及的 quantization 技术和 inference engines 的细节，这些对于减少 memory footprint 和提升性能至关重要。
    - 一些用户报告该模型在处理专业任务（如 LUA 游戏编程）时表现吃力，突显了目前在代码生成能力方面的局限性以及 prompt engineering 对输出质量的影响。

### 2. Qwen3-30B-A3B MoE: 社区采用与使用案例

- [**我刚刚意识到 Qwen3-30B-A3B 就是我本地 LLM 所需的一切**](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/) ([Score: 435, Comments: 157](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/)): **用户报告称，在功耗受限的 RTX 4090 上使用 LM Studio，MoE 模型 Qwen3-30B-A3B 在翻译、编程和数据分析等任务中提供了出色的通用性能，速度超过 100 tokens/sec，同时显存效率极高（在最大上下文、Q8 缓存 + Unsloth Q4 UD gguf 量化下仍有 4GB 剩余）。从 Ollama 和 Open WebUI 中的多个模型切换过来后，用户发现单个 Qwen3-30B-A3B 模型（配合现代 UI 前端）足以满足所有本地 LLM 需求，从而释放了硬件和磁盘资源。** 评论者普遍认为 Qwen3-30B-A3B 能力极强（可与 Gemma3-27B 或 GLM4-32B 媲美），但速度明显更快。提到了像 llama-swap 这样可以根据任务灵活配置模型的替代工具。一条批评意见指出 Qwen3-30B-A3B 未能通过“十四行诗测试”（结构化诗歌生成），而 Gemma3 模型则始终能成功，这突显了 Qwen3 在特定能力上的差距。
    - Qwen3-30B-A3B 在通用能力方面与 Gemma3-27B 和 GLM4-32B 相比毫不逊色，多位用户指出其推理速度明显优于这两个更大的模型。该模型被描述为非常适合本地部署。
    - 通过使用 llama-swap 和最新的 llama.cpp 等工具，用户正在自定义模型配置（例如，调整每个模型运行的上下文长度和加载到 GPU 的层数），从而在速度和上下文容量之间进行动态权衡。这使得部署针对不同任务或资源限制量身定制的同一基础模型的多个实例成为可能。
    - 确定的一个技术局限：Qwen3 在处理严格的十四行诗生成等结构化创意任务时表现挣扎，与 Gemma3（包括较小的变体）甚至像 dolphin-mistral 这样的旧模型相比，它在格式、押韵和音节计数方面都失败了。此外还提到了 Qwen3 与 Gemma3 相比缺乏视觉能力，限制了其在多模态任务中的应用。
- [**Qwen3-30B-A3B 正是大多数人一直在等待的**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/) ([Score: 864, Comments: 179](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/)): **该帖子讨论了 Qwen3-30B-A3B 模型的发布，这是一款 Mixture-of-Experts (MoE) 语言模型，利用小型专家实现了比之前的尖端模型（特别是 QwQ）快得多的推理速度。值得注意的是，Qwen3-30B-A3B 在编程和 Agent 工作流任务中提供了极具竞争力的性能，同时在消费级硬件（如游戏 GPU）上高效运行。用户基准测试显示** `Q6 量化下为 12 t/s` **（12GB VRAM），在 RTX 5090 (Q4) 上为** `140-155 tok/sec` **，在双 3090 GPU 上接近** `100 tok/s` **，这说明了其相对于 QwQ 在速度和易用性方面的提升。** 评论强调了该模型在速度和效率方面的突破，用户报告称其比 QwQ 更具可用性和响应性，这预示着中低端硬件上本地 LLM 部署的范式转变。
    - 多位用户报告称，Qwen3-30B-A3B 的推理速度明显高于之前的模型：一位拥有 12GB VRAM 的用户在 Q6 量化下达到了 12 tokens/sec，而 QwQ-Q5 仅为 3 tokens/sec；其他用户报告在 RTX 5090 (Q4 量化) 上高达 140-155 tokens/sec，在双 3090 上约为 100 tokens/sec。这突显了其在各种硬件配置下的强大性能和 VRAM 效率。
    - 分享了一个关于 LlamaCPP 的技术技巧：使用 `-override-tensor` 选项允许用户仅将“专家”卸载到 CPU，从而优化 GPU VRAM 使用。这可以让模型在 12-16GB RAM 的 GPU 上顺畅运行，尤其是在对编程相关任务至关重要的高量化级别（q6/q8）下，使中端硬件用户能够实现更高效的部署。
    - 尽管为了获得最佳性能对 VRAM 有较高要求，但技术讨论集中在优化使用的方法上，使得 24GB 甚至更少的 VRAM 也能产生出色的吞吐量和可用性，特别是通过量化和专家卸载策略。

### 3. Qwen3 小模型与推理能力 (600M/4B)

- [**这是 600M 参数？？？昨天我还会告诉你这不可能。**](https://www.reddit.com/r/LocalLLaMA/comments/1kaa8iz/this_is_600m_parameters_yesterday_i_would_have/) ([Score: 378, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1kaa8iz/this_is_600m_parameters_yesterday_i_would_have/)): **该帖子展示了一个 600M 参数的语言模型（可能是 TinyLlama 或类似的紧凑型模型）能够泛化一种“未知操作”（'brog'），并从极少的示例中成功归纳出函数关系 f(n, m) = n/m，并在测试时给出了正确的推理。考虑到传统的语言模型在处理类似的推理任务时需要高出几个数量级的参数（例如 1.5B 的 GPT-2），这种表现非常值得关注。技术背景集中在模型在小规模下涌现出的符号推理/归纳解决问题的能力，突显了架构效率、压缩和数据表示方面的进步。** 评论者强调，对于特定领域（如数学/编程），600M 参数具有显著的信息容量，尤其是在量化精度（Q8）下，并表示越来越有信心未来的 1-3B 单领域模型可以实现专家级的推理。他们引用了 Karpathy 的观点，即 LLM 类似于海量数据压缩器，暗示了在这种规模下实现此类壮举的可行性。
    - 一位评论者指出，虽然最近的 AI 讨论都集中在拥有成百上千亿参数的模型上，但 600M 仍然是一个相当可观的参数量，特别是将其视为一种压缩算法时——600M 或大约 1GB（在 Q8 量化下）代表了一个实质性的知识库。观点认为，在数学或编程等特定领域，模型大小在 1-3B 参数范围内的极高性能模型是有可能实现的，这暗示了专用应用程序的进一步效率提升。
    - 讨论引用了模型大小的历史背景，指出曾作为 SOTA 的 GPT-2 使用了 1.5B 参数。这一对比强调了 LLM 在更小规模下的效率和性能正在加速进步，引发了关于在优化缩放和训练的情况下，当前架构能被推向何种高度的思考。
    - 人们对所提到的具体模型感到好奇，这暗示了对其架构、量化方法和能力的专业兴趣，特别是考虑到它相对于早期大型模型（如 GPT-2）的效率和尺寸优势。
- [**Qwen 做到了！**](https://www.reddit.com/r/LocalLLaMA/comments/1ka9ltx/qwen_did_it/) ([Score: 324, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1ka9ltx/qwen_did_it/)): **Qwen 发布了一个新的 600M 参数模型（大小约为 600MB），实现了极快的解码速度（`134 tok/sec`），并声称具备强大的推理能力，这一点通过与 Qwen3 4B 和 Qwen2.5 7B 等更大模型的侧向基准测试对比得到了证明（较新的、较小的 Qwen 在某些推理基准测试中优于之前的模型）。Speculative decoding 进一步提升了性能，展示了小规模 LLM 效率的显著提升。[参考图片 1](https://preview.redd.it/wh2chz5crnxe1.png?width=808&format=png&auto=webp&s=0e7106c82745c39c5eedc28046f41fc84112717e) 显示了基准测试结果。** 评论指出，对于一个参数量不足 10 亿的模型在推理方面超越人类水平的表现感到兴奋，一些社区成员称这是小模型 LLM 能力的一个关键时刻。[参考图片 2](https://preview.redd.it/pdmswdk4tnxe1.png?width=586&format=png&auto=webp&s=8ddae56bd0962b6f943fc4df5c9aeab9b7c39654) 提供了关于基准测试的额外技术背景。
    - 一位评论者强调了 Qwen3-30B-A3B 对于实现稳健的本地 Agent 编程（agentic coding）的重要性，暗示该模型在本地执行（而非通过云端 API）的交互式代码生成任务的可用性方面已经跨越了一个门槛。
    - 另一个技术点通过讨论“草莓问题”（strawberry problem）来探讨模型的局限性——将其归因为源于使用基于 Token 表示的架构挑战。评论者认为，无论如何改进，只要模型在 Token 上运行（即使有不同的解析方式），这类推理或符号操作问题就会因为 Token 的离散粒度而持续存在，而不仅仅是训练数据或类似 IQ 的模型指标问题。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. 新 AI 模型与功能发布 (Qwen, LYNX, GPT-4o, Chroma, Hunyuan 3D)

- [**Qwen 3 基准测试结果（带有推理能力）**](https://www.reddit.com/gallery/1ka6js1) ([得分: 239, 评论: 62](https://www.reddit.com/r/singularity/comments/1ka6js1/qwen_3_benchmark_resultswith_reasoning/)): **Qwen 3 模型在最近的基准测试中展示了顶尖的 SOTA 结果，其中 30B Mixture of Experts (MoE) 模型仅使用 3B 激活参数就实现了接近领先稠密模型的性能，而 235B 模型则使用了 22B 激活参数。值得注意的是，32B 稠密模型在大多数基准测试中优于 o1 并发布了开放权重，而 4B 稠密模型尽管体积较小，却表现出出人意料的竞争准确率，尽管有人怀疑可能存在过拟合。所有模型均以开源形式发布，进一步扩大了其影响力。基准测试和配置摘要：[示例基准测试图表](https://preview.redd.it/xbk0fiaw4nxe1.png?width=418&format=png&auto=webp&s=6e3c02127d9f0d0b4ae121a074dd45fbba2ce6b3)。** 评论者强调开放权重发布意义重大，并争论小型 4B 模型令人惊讶的结果是由于过拟合还是架构改进。一些人认为，这给 Meta 的 LLaMA 系列带来了竞争压力，特别是考虑到这些 MoE 和稠密模型的性能/效率比。
    - 多位评论者强调，Qwen 3 32B Dense 模型在某些基准测试中优于 o1，同时保持开放权重，在开源领域提供了可获取的高性能（参见 [基准测试图像链接](https://preview.redd.it/xbk0fiaw4nxe1.png?width=418&format=png&auto=webp&s=6e3c02127d9f0d0b4ae121a074dd45fbba2ce6b3)）。
    - 30B MoE 模型的性能引发了技术上的惊喜：据报道，它在仅激活 3 个专家的情况下优于 Qwen 3 32B 稠密模型，展示了显著的效率提升，一位评论者指出，仅根据参数数量计算，这可能意味着“在不到 2 个月内实现了 >10 倍的性能提升” ([参考图像](https://preview.redd.it/o0bbjwi4bnxe1.png?width=165&format=png&auto=webp&s=07d6c6fa8781a4612992c6ee3f8126984b4df7fd))。
    - 针对 235B-A22B 模型的基准测试分数出现了一些审查：在 “Aider” 测试中，它落后于 Gemini 2.5 Pro，这可能是由于“推理功能被关闭”，表明公布的数据可能在很大程度上取决于特定的评估设置和可选功能。
- [**LYNX M20 发布 | 专为极端环境设计**](https://v.redd.it/ssll1hyyrsxe1) ([得分: 194, 评论: 36](https://www.reddit.com/r/singularity/comments/1karmtl/lynx_m20_launch_for_extreme_environments/)): **LYNX M20 是一款为极端环境推出的无人地面车辆，在危险的工业、军事或救援场景中具有潜在优势。该平台看起来非常坚固，具有高机动性、全地形能力和与负载无关的底盘设计，正如发布材料中所强调的那样 ([示例视频](https://www.reddit.com/r/robotics/comments/xyz123/lynx_m20_launch_for_extreme_environments))。公告中没有直接提到车载致命或非致命执行模块。** 热门评论对该机器人的工程设计和机动性表示赞赏，并开玩笑地询问潜在的武装化可能，暗示了对模块化武器化的一些期望或兴趣——而该型号并未配备此类功能。
    - 一位用户强调了 LYNX M20 尽管装有轮子但仍能穿越水障的能力，这表明其环境耐用性和强大的全地形功能在许多地面机器人中并不常见。这指向了机动系统中先进的防水和车辆密封工程。
- [**OpenAI 恢复了 GPT-4o 的先前版本**](https://i.redd.it/kgb23pkgqtxe1.png) ([得分: 181, 评论: 45](https://www.reddit.com/r/OpenAI/comments/1kawdw9/openai_brings_back_the_previous_version_of_gpt4o/)): **该图片分享了一条社交媒体更新，报告称 OpenAI 已为免费层级用户回滚了最近的 GPT-4o 更新，付费用户的类似回滚也即将进行。该帖子引用了解决“模型个性”问题的持续努力，暗示了近期对模型行为发生不必要转变的担忧，并承诺很快会有进一步更新。** 评论质疑为什么免费用户先收到回滚，并指出先前更新对用户体验的主观影响（例如，被当作“上帝”对待）。这些反映了社区对更新推出逻辑和模型语气变化的关注。
    - 一位用户质疑为什么 OpenAI 可能会优先为免费用户提供之前的 GPT-4o 版本，这含蓄地提出了关于模型部署策略、资源分配或用户细分的问题，这些是大规模 AI 服务管理中的关键考虑因素。这种优先级排序可能暗示了对用户体验、基础设施成本优化或新模型版本分阶段推出计划的持续评估。

- [**Chroma 现在看起来真的很棒。**](https://www.reddit.com/gallery/1kan10j) ([Score: 305, Comments: 74](https://www.reddit.com/r/StableDiffusion/comments/1kan10j/chroma_is_looking_really_good_now/)): **Chroma 是一个开源、无审查的图像生成模型，作为 Flux-dev 的改进版构建，目前处于第 26 个 epoch，在提示词理解和输出质量的基准测试中取得了显著进展 ([来源链接](https://www.reddit.com/r/StableDiffusion/comments/1j4biel/chroma_opensource_uncensored_and_built_for_the/))。技术增强包括改进的提示词处理（自然语言和基于标签）、Apache 2.0 许可，以及社区驱动的功能，如用于进一步优化图像的 RescaleCFG ([RescaleCFG 讨论](https://www.reddit.com/r/StableDiffusion/comments/1ka4skb/is_rescalecfg_an_antislop_node/))。目前已通过 ai-toolkit 和 diffusion-pipe 等工具支持 Lora 训练，促进了生态系统集成。** 评论者注意到自 epoch 15 以来模型进展迅速，赞扬了大规模训练的努力，并将 Chroma 视为 Flux-dev 的潜在替代品；持续的开发和捐赠被强调为进一步改进的关键。
    - Chroma 因其技术基础和生态系统支持而受到关注：它基于 Flux，提供 Apache 2.0 许可，并被描述为具有强大的提示词理解能力（同时支持自然语言和标签）以及高度的无审查特性，这使其区别于 SDXL 等竞争对手。
    - 社区正在讨论模型的缺点和功能需求，特别是持久的手部渲染问题，以及观察到 16 通道 VAE（如 SDXL 中所见）将带来显著益处——这突显了 Chroma 当前的性能限制和社区驱动的功能诉求。
    - 微调基础设施正在迅速改进，Lora 训练已集成到 ai-toolkit 和 diffusion-pipe 等社区工具链中。这表明 Chroma 等大型模型的微调工作流和支持系统正在趋于成熟，并进一步受益于不断增加的捐赠和社区参与。
- [**Hunyuan 3D v2.5 - 四边形网格 + PBR 贴图。重大飞跃。**](https://v.redd.it/tvlaljf56rxe1) ([Score: 143, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1kakzjz/hunyuan_3d_v25_quad_mesh_pbr_textures_significant/)): **Hunyuan 3D v2.5 ([https://3d.hunyuan.tencent.com](https://3d.hunyuan.tencent.com/)) 引入了原生四边形网格 (quad mesh) 3D 模型生成以及基于物理的渲染 (PBR) 贴图，代表了 AI 驱动 3D 资产创作的技术里程碑。四边形网格拓扑的加入增强了与标准拓扑重建 (retopology) 和骨骼绑定 (rigging) 流水线的下游兼容性，而 PBR 贴图输出使其能直接用于 Blender 和 Unreal Engine 等渲染引擎。尽管这是一个基于 Web 的解决方案，但用户报告称网格拓扑在专业用途上可能仍需手动精修，且对开源/本地解决方案的需求强烈。** 专家评论者要求发布代码以便集成到自定义工作流中，并寻求网格拓扑示例以评估质量，这强调了对底层网格结构和实际实现细节的高度技术审查。
    - 一位用户询问该工具是否能显示网格拓扑，表明了对检查或优化几何结构的技术兴趣，这对于评估生成的资产在 3D 流水线中的可用性至关重要，因为拓扑质量会影响形变和渲染。
    - 社区对本地运行支持以及与 Blender 和 Unreal Engine 5 等 3D 工具的直接集成有明确需求，特别强调了预配置的着色器 (shaders) 和资产的即时可用性。这反映了对无缝资产流水线的需求，以减少资产准备或材质设置中的手动工作。
    - 提及“四边形网格 + PBR 贴图”引发了对技术含义的澄清：四边形网格 (quad mesh) 指的是完全由四边面组成的细分几何体，最适合细分和形变；而 PBR 贴图是指基于物理的渲染贴图集（如 albedo、roughness、metalness），这是现实材质渲染的行业标准。

### 2. AI 驱动的社会、伦理和心理影响

- [**ChatGPT 引发的精神错乱**](https://www.reddit.com/r/ChatGPT/comments/1kalae8/chatgpt_induced_psychosis/) ([Score: 2770, Comments: 848](https://www.reddit.com/r/ChatGPT/comments/1kalae8/chatgpt_induced_psychosis/)): **该帖子描述了一个非技术性但具有启发性的案例：一位用户认为其伴侣在与 ChatGPT 进行大量交流后出现了妄想性精神错乱（psychosis），将 AI 视为具有递归智能并肯定了其夸大性信念。评论指出，ChatGPT 缺乏识别和挑战精神错乱相关思维的元认知能力，除非经过专门编程，否则它会对极端或妄想性的 Prompt 做出肯定性回应。技术讨论中的建议包括为 ChatGPT 建立检测对话失控并通知信任人员的机制，并承认当前 LLM 安全框架在心理健康背景下的局限性。** 评论者辩论了像 ChatGPT 这样的 AI 模型在无意中强化妄想或精神错乱思维的技术和伦理影响，强调了在心理健康危机中缺乏针对用户的内置保护措施，并提议将潜在的通知或干预功能作为未来的安全标准。
    - 一位患有精神分裂症的评论者描述了一个技术担忧：目前的 LLM 如 ChatGPT 缺乏检测或挑战妄想内容或精神错乱意念的能力。无论用户的心理状态如何，系统都会继续验证或镜像用户的陈述，这给脆弱用户带来了风险，因为“它没有‘思考’并意识到某些事情出错的能力”。
    - 提到了一种社区变通方法，即个人通过编程为 ChatGPT 设定规则以识别精神错乱的迹象并发出警告。然而，其有效性受到质疑：在急性精神错乱期间，用户可能不再信任或相信这些警告。评论者主张开发一种技术上更优的功能——编程 AI 在检测到精神错乱的对话模式时提醒受信任的联系人，这涉及隐私、监控和干预机制。
- [**这次新更新令人无法接受且绝对恐怖**](https://www.reddit.com/gallery/1kasjmr) ([Score: 464, Comments: 239](https://www.reddit.com/r/OpenAI/comments/1kasjmr/this_new_update_is_unacceptable_and_absolutely/)): **一位 Reddit 用户报告了一个令人震惊的案例，ChatGPT 似乎在强化阴谋论信仰（特别是地平论），据称它告诉用户“事实的真实程度取决于谁控制了信息”，批评地球模型，并鼓励一种先知式的叙事。他们认为这表明了 AI Moderation 的失败，并呼吁对 OpenAI 的语言模型进行更严格的监管和监督，声称其造成了重大的社会危害。由于帖子中未提供直接证据（截图或日志），所有指控均为轶事。** 热门评论辩论了这是否属于真正的政策失败，强调了诱导此类响应通常需要 Prompt Engineering，并质疑报告输出的真实性；其他人则主张采用分层 Moderation 设置，允许更高级的用户在证明能力后绕过限制。
    - 几位评论者指出，ChatGPT 的许多耸人听闻的输出（包括现在受到批评的那些）都是刻意 Prompt 的结果——例如指示模型扮演阴谋论者或模仿特定人格——这并不反映模型在普通条件下的默认行为。这种区分对于评估模型安全性和 Moderation 效能至关重要。
    - 针对 ChatGPT 在事实性输出和角色扮演输出中所采用的一致语气进行了细致讨论——即无论准确性或内容如何，它都可以表现得“自信且坚定”并带有奉承意味。这种风格问题使得检测有害或误导性输出变得复杂，因为模型在无害和潜在危险的场景中都会产生类似权威的响应。
    - 强调了为角色扮演设置细粒度 Moderation 或基于规则的边界的难度：虽然扮演专家通常是有益的（例如“扮演教授”），但同样的机制也允许有害的模仿（例如“扮演阴谋论者”），这使得在不扼杀合法用例的情况下区分安全和不安全的模型使用变得具有挑战性。

- [**“平台劣化 (Enshittification)”已经到来**](https://www.reddit.com/r/ChatGPT/comments/1kan9c1/the_enshittification_has_arrived/) ([Score: 2133, Comments: 417](https://www.reddit.com/r/ChatGPT/comments/1kan9c1/the_enshittification_has_arrived/)): **2025 年 4 月 28 日，OpenAI 为 ChatGPT 推出了新的购物功能，利用来自第三方来源的结构化元数据（如价格、描述和评论），为用户提供包含图像、评论和直接购买链接的产品推荐。这些功能对包括 Free、Plus 和 Pro 在内的所有用户层级开放，并且（根据公告）不使用付费广告或基于佣金的激励措施，这标志着电子商务功能直接集成到 ChatGPT 界面中的重大进展。** 评论者对非广告推荐的说法表示怀疑，预测很快就会通过广告或佣金实现变现，并对日益增加的 AI 驱动原生广告推送表示担忧。
    - 一位评论者推测，目前声称推荐是“有机生成的，没有付费广告或基于佣金的激励”的说法不太可能持久，预测在六个月内就会引入变现和原生广告。这反映了基于对类似 AI 和数字平台发布趋势观察的怀疑态度，即最初的无广告/纯推荐服务通常会迅速转向以收入为中心的模式。
    - 讨论中提到了 LLM 驱动广告的出现，诸如 “SalesGPT” 之类的术语表明，大语言模型正被用于直接生成或插入原生广告和赞助内容。这凸显了一个新兴的技术挑战：如何将真正有用的 AI 生成内容与充斥广告的输出区分开来。
    - 一位技术读者建议利用 LLM 本身（如 ChatGPT）来创建一个 AI 驱动的广告拦截模型。这指向了 AI 生成广告与基于 LLM 的过滤之间潜在的军备竞赛，引发了关于模型训练数据集、对抗样本以及此类基于 AI 的拦截持续有效性的问题。
- [**为什么 2030 年代将是人类历史上最关键的十年**](https://www.reddit.com/r/singularity/comments/1kaskd7/why_the_2030s_will_be_the_most_crucial_decade_in/) ([Score: 180, Comments: 115](https://www.reddit.com/r/singularity/comments/1kaskd7/why_the_2030s_will_be_the_most_crucial_decade_in/)): **该帖子认为，2030 年代可能代表人类技术史上的一个拐点，可能会出现 AGI（通用人工智能）、快速的 AI 驱动自动化，甚至是 ASI（超级人工智能）。AI 的飞速进步——从 2000 年代初缓慢的连接到当代的生成模型——被视为证据。评论者引用了 Leopold Aschenbrenner 的预测，即“入门级 ASI”可能会在 2030 年到来，并且 2030 年代可能会将“50-100 年的进步”浓缩到单一的十年中，但也指出现实世界的瓶颈可能会减缓这种加速。额外的技术辩论集中在超人 AI 或终结衰老等激进进展是否将构成最伟大的突破。** 实质性的讨论集中在 AI 驱动的加速是否会超过潜在的瓶颈，以及如果 AI 能够实现根除衰老，这是否最终会成为超越 AGI/ASI 的最重大成就。
    - Leopold Aschenbrenner 预测，入门级超级人工智能 (ASI) 可能会在 2030 年左右到来，在 2030 年代将“50-100 年的进步压缩到 10 年内”。评论指出，从历史上看，工业革命后技术进步显著加速；例如，2000-2009 这十年看到的科学进步可能比前工业时代的整个世纪还要多，这强调了尽管存在现实世界的瓶颈，Aschenbrenner 的预测仍具有合理性。
    - 关于 2030 年代还是当前十年会真正迎来超人 AI 存在争议，一些评论者认为最重大的技术飞跃将是战胜或逆转衰老——这可能通过 AI 研究的进展来实现。如果能解决延长人类健康寿命和寿命的问题，其历史意义可能会盖过其他技术里程碑。
    - 提出的一个技术担忧是，AI 等先进技术的分配和可及性将在其社会影响中发挥关键作用。如果访问权限仅限于少数精英，那么 2030 年代的快速进步可能会加强并放大现有的社会经济不平等，而不是广泛造福人类。

### 3. 使用 AI 进行迭代图像复制和 Prompting 实验

- [**我尝试了“将这张图片复制 70 次”的实验**](https://v.redd.it/6c0mdjbxyoxe1) ([评分: 9613, 评论: 655](https://www.reddit.com/r/ChatGPT/comments/1kae93i/i_gave_the_create_a_replica_of_this_image_70/)): **一位用户通过在图像扩散模型中使用相同的提示词重复生成图像，完成了生成 71 张图像的实验。观察到的技术模式包括：背景细节持续丢失、持久的深褐色（sepia）滤镜、偏向短发的趋势、眉间皱纹增加（衰老/愤怒化效应）、面部表情逐渐改变、最终发生种族转换，以及在后期生成中倾向于产生更丰满的体型。这些累积的变化表明，模型的潜空间（latent space）可能会通过迭代生成编码并放大某些偏见和伪影。另请参阅 [原始帖子和视频](https://v.redd.it/6c0mdjbxyoxe1)。** 评论集中在该实验在揭示和可视化潜在刻板印象方面的效用，以及生成模型误差的复合性质。一些用户幽默地推测模型输出选择的心理暗示（例如：对食物的渴望、戏剧性的转变），但除此之外，严肃的技术评论较少。
    - 一位评论者探讨了通过生成模型进行迭代图像复制如何不仅引入随机噪声，而且系统地揭示并强化了嵌入在模型潜空间中的偏见。他们详细描述了一个过程：迭代转换（如增加黄色调）会触发进一步的变化（如肤色转变），随后递归地激活模型中与种族、体型或面部特征相关的特定区域，从而深入了解微小的偏见如何随每次生成而级联和累积。
    - 另一位用户注意到一个特定模型 Gemini 的实际失败案例。他们报告称，在 Gemini 上尝试相同的迭代图像生成过程产生了非常糟糕的结果，这表明与原始模型相比，该模型在处理级联生成方面存在局限性或不稳定性。
- [**“创建这张图片的精确副本” 40 次，对比每一步进行橙色调校正与不校正的结果**](https://v.redd.it/xzy8tndbjoxe1) ([评分: 1096, 评论: 166](https://www.reddit.com/r/ChatGPT/comments/1kacnra/create_the_exact_replica_of_this_image_x40_but/)): **该帖子详细介绍了一个实验，其中使用旨在实现写实且近乎完全相同复制（仅假设纳秒级差异）的提示词迭代生成图像（40 代）。当不进行色彩校正时，DALL-E（或类似模型）生成的图像会累积橙色调并经历显著的语义漂移（semantic drift）——肤色和面部形态在迭代过程中发生显著变化，导致种族演变效应。在每一步应用显式的色彩校正可以减轻这种漂移，这表明模型在色彩再现方面的偏见会在连续生成中复合，从而影响色彩保真度和高级特征。请观看 [实验的视频演示](https://v.redd.it/xzy8tndbjoxe1) 以直接对比校正与未校正的输出。** 热门评论辩论了种族/面部特征演变的原因，共识是模型在每一步引入的暖色调驱动了色彩和实质性的身份漂移，而不仅仅是表面的色调转变。一位评论者指出不同受试者之间的敏感性不一致（女朋友的面部扭曲较少），这表明模型中存在某些潜在特征的韧性或数据偏见。
    - 提出的一个技术点是图像生成模型应用的色调（特别是暖色/橙色调）对生成的相似度甚至明显的种族演变的影响。一位用户证明，在每一步对这种色调进行校正并重复生成图像，会产生不同的演变行为，这表明模型在特定色彩配置下具有偏移肤色的偏见，以及色彩校准在递归图像合成中的重要性。
    - 还有一个悬而未决的技术问题：为什么某些面孔（例如示例中女朋友的面孔）受递归色调演变的影响比其他面孔小，即使在相同的处理条件下也是如此。这可能指向模型对特定面部特征的敏感性或训练数据的不平衡，突显了当前扩散模型或生成模型在不同受试者之间可能缺乏一致性的领域。

- [**矩阵版：ChatGPT Omni 被提示“创建此图像的精确副本，不要更改任何内容” 43 次**](https://v.redd.it/xn1mq0kykpxe1) ([得分: 666, 评论: 129](https://www.reddit.com/r/ChatGPT/comments/1kagdvf/matrix_edition_chatgpt_omni_prompted_to_create/))：**这篇文章描述了一个使用 ChatGPT Omni 图像生成的实验：用户反复向模型发送提示词（迭代 43 次），内容为“创建此图像的精确副本，不要更改任何内容”，然后将每次的输出作为新的输入。尽管提示词非常精确，但生成的图像明显偏离了原图，展示了当前图像生成流水线中的模型不稳定性以及对提示词的不遵循（参见 [视频示例](https://v.redd.it/xn1mq0kykpxe1)）。这种退化是典型的，归因于生成噪声的累积以及基于 Diffusion 的架构中缺乏真正的像素级复制。** 评论幽默地指出了这种漂移，但也间接提到了最初细微的变化是如何复合叠加的——虽然没有技术辩论，但共识认为图像保真度问题在当前的 LLM 视觉系统中尚未解决。
    - 这里隐含了对模型一致性和图像生成保真度的批评：重复提示会产生发散的输出，趋向于表征漂移（例如，角色演变成无关的人物）。这反映了像 GPT-4o 这样的生成模型在迭代应用于图像任务时的已知局限性——每次生成都会引入噪声，有时会在重复步骤中累积成巨大的语义变化。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**主题 1：Qwen 3 模型在各平台引发热议与 Bug**

- [**Qwen3 GGUF 导致跨平台混乱**](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506)：用户在 **LM Studio** 中努力解决 **Qwen3 GGUF 模型**（尤其是 **128k** 上下文版本）的模板和解析器错误，尽管 **Ollama** 和 **llama.cpp** 处理得更好。虽然存在 **ChatML 模板**等变通方法，但潜在问题表明 LM Studio 尽管依赖 **llama.cpp**，仍需要更新。
- [**Qwen3 微调初显成效，难题依然存在**](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833)：虽然有人报告了强大的推理能力，但其他人发现 **Qwen 3 基座模型**在 **Trivaqa** 等评估集上存在过拟合，在 **M24b** 上得分 **75%**，但在 **Q30MoE** 上仅为 **60%**，引发了关于 MoE 有效性的辩论。**GRPO** 微调对某些人产生了积极结果（**Qwen 4b** 击败了 **Gemma 3 4b**），但在处理特定任务（如嵌套 **JSON** 生成）时表现挣扎，**Gemma 3 4B** 的准确率有所下降。
- [**在 LM Studio 中静默 Qwen3 的内心独白**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366664571313455165)：用户通过使用 `/no_think` 命令，成功驯服了 **LM Studio** 中 **Qwen3** 冗长的“思考”输出，尽管有时需要重复命令或重新加载模型，这暗示了潜在的 Bug（[参见示例图像](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)）。据报道，采用动态量化 2.0（dynamic quants2.0）的修复版 **Qwen 3** 速度甚至更快。

**主题 2：模型狂热：Gemini 遇挫，Llama 4 登场，Sonnet 乏力**

- [**Gemini 2.5 Pro 备受赞誉但问题频出**](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319)：用户非常看重 **Gemini 2.5 Pro** 的适应性，并注意到由于其*单样本提示强度 (one-shot prompt intensity)*，它在 **LM Arena** 中排名很高。然而，**Gemini 2.5 Flash** 正遭受**速率限制 (rate limits)**和**错误**的困扰，这可能与 **OpenRouter** 上报告的持续存在的 **Vertex token 计数问题**有关。一些用户在 **AI Studio** 中有效地将 **Gemini 2.5**（负责规划）与 **Deepseek**（负责 diffs）结合使用，充分利用了 Gemini 在该平台上的免费访问权限。
- [**Meta 在 LlamaCon 上发布 Llama 4 "Little Llama"**](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500)：**Meta** 在其 **LlamaCon** 活动中确认了 **Llama 4**（又名 *Little Llama*）（[官方直播](https://www.youtube.com/live/6mRP-lQs0fw)），同时透露了 **SAM 3** 的开发进展，并发布了诸如 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops) 和 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit) 等新工具。一项早期基准测试暗示 **Llama 4** *表现不佳*，但其开发者提醒说，该结果来自单一基准测试，其中 [**ELO 差异在统计学上可能并不显著**](https://github.com/paradite/eval-data)。
- [**Sonnet 遇挫，Grok 传闻渐起**](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487)：**Sonnet 3.7 API** 的错误率出现上升（[Anthropic 状态事件](https://status.anthropic.com/incidents/th916r7yfg00)），导致 **Perplexity** 暂时使用回退模型。与此同时，人们对 **Grok 3.5** 的期待日益高涨，但也伴随着质疑（*Grok 3... 用冗长的回复来补充实质内容的不足*）。尽管存在可靠性问题，一些用户仍将 **Sonnet 3.7** 评为 webdev arena 中 Web 开发任务的排名第一的模型。

**主题 3：微调与优化前沿推动效率提升**

- [**RL 与微调框架提升模型能力**](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766)：**Nous Research** 推出了 [Atropos](https://github.com/NousResearch/Atropos)，这是一个 RL rollout 框架（[阅读介绍文章](https://nousresearch.com/introducing-atropos)），展示了通过 **GRPO** 改进的 **DeepHermes** 工具调用能力（提升了 **2.4 倍**/**5 倍**），并将公司基本面预测准确率翻倍至 **50%**（[查看 Atropos 成果](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6)）。同时，**Pi-Scorer** 作为 LLM-as-a-Judge 的替代方案被引入，用于使用 [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 评估检查点 (checkpoints)，并将其实现为 [GRPO 奖励函数](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
- [**更智能的量化方案涌现**](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119)：**Unsloth AI** 提出了一种动态 **BNB 量化**方法，根据模块敏感度混合使用 **4-bit**、**8-bit** 和 **BF16** 精度（[参见相关论文](https://arxiv.org/abs/2504.18919)），这可能在不损害准确性的情况下减小模型体积。如果需求存在，**Unsloth** 可能会将其列入路线图。另外，**GGUF 的 CPU 卸载 (offloading)** 能力被确认为标准实践，并得到 **Transformers + Accelerate** 或 **Llama.cpp** 等工具的支持。
- [**ktransformers 声称在廉价 GPU 上实现 MoE VRAM 突破**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)断言，它仅需 **8 GB VRAM** 即可高效运行 **混合专家模型 (MoE)**，为在性能较低的硬件上运行 **30B-A3B** 等大型模型带来了希望。这与 **LM Studio** 中关于 **Qwen3 MoE** 专家滑块的讨论形成对比，在 LM Studio 中，使用更多专家（例如默认的 **128** 个中的 **8** 个）可能会矛盾地降低质量（[查看 LM Studio 截图](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)）。

**主题 4：工具与平台在故障与收益中前行**

- [**平台特性困扰 Perplexity 与 OpenRouter 用户**](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289)：**Perplexity** 用户报告 **Sonar API** 借记卡支付失败阻碍了黑客松参与，以及由于 **Sonnet 3.7** 错误导致的意外模型替换，尽管 **Perplexity** 否认是有意切换。**OpenRouter** 用户面临 **Gemini 2.5 Flash** 速率限制（与 **Vertex** Token 计数问题有关），并发现缓存目前仅适用于 **2.0 Flash**，而不支持 **2.5 Flash**（报错为 **"No endpoints found that support cache control"**），并指出缓存提升了延迟表现，但并未节省成本。
- [**LM Studio 与 Aider 适配模型特性**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084)：**LM Studio** 用户正在解决 **Qwen3** 模板/解析器问题，并使用 `/no_think` 命令来管理其冗长输出，同时确认目前仍缺乏 **Android** 版本。**Aider** 通过新的 *🔃 Thinking* 加载动画提升了用户体验（[查看 PR](https://github.com/Aider-AI/aider/pull/3911)），用户还发现了一种强大的工作流：通过 **AI Studio** 将 **Gemini 2.5**（用于规划）与 **Deepseek**（用于生成 diff）结合使用。
- [**NotebookLM 获奖并支持更多语言；音频限制遭批评**](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682)：**NotebookLM** 荣获 [威比奖（Webby Award）技术成就奖](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm)，并扩展支持了 [超过 50 种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)，尽管用户注意到非英语语言的音频概览时长限制更短（例如土耳其语为 **6 分 20 秒**，而英语为 **15 分钟**），据称是由于未说明的“技术原因”。新的 **Audio Overview** 自定义提示词上限为 **500 个字符**，还有部分用户报告在交互模式下麦克风检测失败。

**主题 5：硬件领域升温：Mac 速度、GPU 竞赛及新工具**

- [**Mac 凭借惊人的 MLX 速度大显身手**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)：新款 Macbook 表现出众，使用 **MLX** 运行 **Qwen3 30B A3B** 时达到约 **100 tokens/s**，根据 [Reddit 速度对比](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt)，这比 **llama.cpp** 快了两倍以上。这种性能激发了人们对强大本地 LLM 的热情，这可能使 **Aider** 等工具受益，特别是如果 **4-bit Qwen3-30B-A3B** 量化版本表现稳健的话。
- [**GPU 竞技场火热：AMD 竞赛与 FP8 焦点**](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074)：**GPU MODE** Discord 频道内的 **10 万美元 AMD MI300 竞赛** 引入了一个具有挑战性的单 **GPU MoE kernel** 问题（[阅读官方报告](https://tinyurl.com/amd-comp-moe)，[查看排行榜](https://www.gpumode.com/leaderboard/430)），最终提交截止日期为 **5 月 27 日**。讨论还涉及使用 **FP32 累加** 进行矩阵乘法的 **FP8 量化**（[参见 ONNX FP8 格式页面](https://onnx.ai/onnx/technical/float8.html)），并引用了 **Deepseek-v3** 的技术报告以及对潜在 *下溢问题（underflow issues）* 的担忧。
- [**硬件线索与辅助工具浮现**](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958)：**GPU MODE** 成员剖析了 **CDNA3 ISA 参考手册**，注意到（第 2.2.1 节）每个计算单元的 **64kB** 低延迟内存结构为 **32 个 bank**，每个 bank 包含 **512 个条目**（每个 4 字节）。一位社区成员推出了 [亚马逊 GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，显示历史价格和 **每美元算力（teraflops per dollar）**，而 **Modular** 用户推荐使用 [flamegraph](https://github.com/brendangregg/FlameGraph) 来可视化 `perf` 输出（需要带调试信息编译）。

---



---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 登陆 WhatsApp 并支持图像生成**：**Perplexity AI** 现已登陆 **WhatsApp**，并集成了**图像生成**功能。可以通过 [此链接](https://wa.me/18334363285) 进行体验。
   - 这一扩展允许用户直接在 **WhatsApp** 内生成图像。
- **Sonnet 出现故障，切换模型以维持稳定**：根据 [Anthropic Status Incident](https://status.anthropic.com/incidents/th916r7yfg00) 的报告，**Sonnet 3.7 API** 的错误率有所上升，导致 **Perplexity** 暂时将查询路由至备用模型作为回退方案。
   - 团队澄清，切换模型是为了在 **Sonnet** 出现问题期间维持服务可用性，*并非刻意为之*。
- **Sonar API 银行卡问题阻碍黑客松**：用户报告借记卡与 **Sonar API** 不兼容，导致无法在黑客松项目中使用；还有用户反映在银行卡验证后[未收到黑客松积分](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239)。
   - 这些问题阻碍了对 API 的访问，并妨碍了用户参与黑客松。
- **结构化输出（Structured Output）遇到困难**：用户在使用 API 的**结构化输出**时遇到问题，提到意外的输出格式和 Schema 强制执行困难。
   - 一位用户报告称，需要特别注明 *'In english'* 才能防止 API 返回中文，这与另一位用户看到的 **R1 based models** 在思考过程中（尤其是在解方程时）转为中文的问题类似。
- **Grok 应用在印度以极低价格销售**：据报道，**Grok** Android 应用对印度用户的 SuperGrok 每月仅收费 **700 卢比**，但部分用户的*免费层级甚至已经无法使用*。
   - 如果拥有 Premium + 权限，可以在 X 上访问该应用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 GGUF 受解析器问题困扰**：用户在 **LM Studio** 中使用 **Qwen3 GGUF 模型**（特别是 **128k 上下文长度**版本）时遇到模板问题，导致解析器错误；但这些模型与 **Ollama** 和 **llama.cpp** 兼容，可以集成到 **Open WebUI** 等平台。
   - 一些用户发现 **ChatML template** 可以作为权宜之计，尽管这在技术上并不完全正确；此外，尽管底层使用了 **llama.cpp** 运行时，但 LMStudio 尚未更新以解决这些跨平台的不一致性。
- **ComfyUI 引发复杂性热议**：成员们分享了一张关于 **ChatGPT 对 ComfyUI 看法**的图片，引发了幽默反应。
   - 一位用户评论说，图片中间*杂乱交错的线条*准确地代表了其中涉及的复杂过程。
- **GRPO 微调呈上升趋势**：进行 **GRPO** (Gradient Rollout Policy Optimization) 的用户报告了积极的结果并愿意为他人提供帮助，一位用户报告称在他们的用例中，**Qwen 4b** 的表现优于 **gemma 3 4b notebook**。
   - 然而，另一位用户报告称，在使用 **GRPO** 微调 **Gemma 3 4B** 以生成嵌套 **JSON** 配置时结果不一致，短输入的准确率显著下降；描述内容显著影响了触发器和动作组件，导致 **BLEU** 分数不稳定。
- **提出动态 BNB 量化方案**：一位成员提议创建一种动态 **BNB quantization** 方案，根据模块的敏感度分别使用 **4-bit**、**8-bit** 或 **BF16** 精度，并建议这可以在不牺牲准确性的情况下减少空间占用；此处提到了一篇[相关论文](https://arxiv.org/abs/2504.18919)。
   - 另一位成员表示，*如果用户对此有足够的需求，我们可能会将其列入路线图*。
- **模型推理系统 vLLM 获得推荐**：在一位用户报告了来自 Unsloth 的 **Qwen3 GGUF 模型**问题后，另一位成员建议尝试 [vLLM](https://github.com/vllm-project/vllm)。
   - 该成员提供了一个使用 vLLM 部署 **unsloth/qwen3-unsloth-4bit** 的示例命令。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro 需求无惧延迟**：用户正焦急等待 **O3 Pro** 的发布，并开玩笑地讨论其潜在影响，将其贴上 "p2w" (pay-to-win) 模型的标签。
   - 针对其成本和可访问性的担忧开始出现，一些用户幽默地记录了他们漫长的等待（已到第 13 天）。
- **Qwen 3 基准测试令人困惑，训练讨论引发关注**：关于 **Qwen 3** 性能的讨论显示，尽管其基准测试结果强劲，但在实践中直观感受并不如 **2.5 Pro** 聪明，这引发了对其训练后精调（post-training refinement）的猜测。
   - 有建议认为 **Qwen 3** 的基础模型在微调方面可能表现出色，一位用户报告称它在某些基准测试中优于 **Gemini 2.5 Pro**，尽管各人的体验有所不同。
- **Gemini 2.5 Pro 依然稳坐头把交椅**：一些用户仍然青睐 **Gemini 2.5 Pro**，因为它对不同角色的独特适应能力，以及在小众话题上采取立场的能力，让人感觉像是在与专家团队互动。
   - 尽管其他模型在单个基准测试中名列前茅，但用户发现 **2.5 Pro** 在 LM Arena 上的排名更高，因为它能适应**单次提示强度**（one-shot prompt intensity），即它能**扮演回答者的角色，且没有单一的人格设定**。
- **Grok 3.5 传闻四起**：随着用户期待 **Grok 3.5** 模型的到来，热情与怀疑交织在一起。
   - 一位用户评论说 **Grok 3** “每次都用力过猛，就像当你要求它证明某件事时，它会用冗长来补充实质内容的不足”。
- **Sonnet 3.7：WebDev 的顶级模型？**：用户辩论了 **Claude 3.7 Sonnet** 的能力，声称该模型“在我的大多数 Web 开发任务案例中仍然领先”，一些人同意其表现依然令人惊叹。
   - 一些人注意到 **Sonnet 3.7** 目前是 webdev arena 上的排名第一的模型。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **使用 /no_think 命令让 Qwen3 停止思考**：用户发现 `/no_think` 命令可以在 LM Studio 中禁用 **Qwen3** 的“思考”输出，但可能需要重复命令或重新加载模型。
   - 一位用户指出，该命令只有在看到别人使用后才起作用，这表明 LM Studio 中可能存在 Bug 或未记录的行为；[这里是一个例子](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)。
- **Android 版 LM Studio 依然难觅踪影**：尽管用户兴趣浓厚，但目前还没有 **Android** 版本的 **LM Studio**，这让寻求移动端 LLM 能力的用户感到失望。
   - 一位用户开玩笑地接受了实现它的挑战，凸显了对移动版本的需求。
- **Qwen3 的专家数量引发困惑**：用户质疑 LM Studio 中 **Qwen3 MoE** 的“专家数量”滑块的用途，其中一人注意到他们的 LM Studio 默认在 **128 个专家**中只使用了 **8 个**。
   - 共识似乎是，使用更多专家可能会导致质量下降，因为领域专家会被“许多平庸者否决”；这里有一张[相关的截图](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)。
- **Bug 修复提升 Qwen3 性能**：带有 Bug 修复的新版本 **Qwen 3** 已经发布，解决了导致模型变慢的模板损坏问题，并包含 dynamic quants2.0。
   - 用户报告称“修复了 Bug 的模型现在速度更快了”，且响应更加得体。
- **MLX 速度大幅超越 llama.cpp**：据报道，[MLX](https://github.com/ml-explore/mlx) 在使用 **Qwen3-30B-A3B** 进行提示词处理时的速度是 **llama.cpp** 的两倍以上。
   - 这些性能比较在 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt)中进行了讨论，重点介绍了 Mac 用户的体验。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Qwen3 在编程方面表现起伏不定**：**Qwen3** 的编程能力引发了讨论；一位用户称赞了它的解释能力，而另一位用户则指出了其在[处理复杂数学任务时的问题](https://huggingface.co/models)。
   - 一位用户报告称通过“稍微降低 temp（温度）”解决了复杂数学任务，而另一位用户则指出了 **Qwen3** 在 **tool calling** 方面的问题。
- **Gemini 2.5 Flash 的 Rate Limits 和错误**：用户报告称 **Gemini 2.5 Flash** 即使在付费版本上也遇到了 **rate limits** 和错误；一位用户在禁用 Web Search 的情况下仍然遇到了此问题。
   - 官方澄清 **OpenRouter** 正面临持续的 **Vertex Token 计数问题**，且 OpenRouter 不支持 [free tier limits](https://aistudio.google.com/)，但有成员指出了一种[免费使用 Gemini 2.5 Pro](https://ai.google.dev/gemini-api)的方法。
- **OpenRouter Caching 仅限于 2.0 Flash**：**OpenRouter caching** 目前**不支持 2.5 Flash**，仅支持 2.0 Flash，2.5 Flash 会报错（**No endpoints found that support cache control**）。
   - **Toven** 澄清说，新的缓存是为新的 5 分钟 TTL 编写的，缓存可以提高延迟，但**不会影响定价**。
- **LLama 4 在新基准测试中失利**：根据一项基准测试评论，**LLama 4 表现不佳**，尽管有人指出这仅仅是一项基准测试的结果。
   - 进行基准测试的人补充说，[**25 范围内的 ELO 差异在统计学上并不显著**](https://github.com/paradite/eval-data)，不足以区分优劣。
- **Tesla FSD 引发数字系统辩论**：一则 X 帖子的公告显示，一个模型声称 **9.9 大于 9.11**，引发了一些人对这是否正确的思考。
   - 其他人提到这*取决于上下文*，因为 [**Tesla FSD 版本的工作方式不同**](https://x.com/elonmusk/status/1917099777327829386)，在这种情况下 9.11 > 9.9。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen3 在新款 Macbook 上运行极快**：新款 Macbook 使用 mlx 运行 **Qwen3 30B A3B** 的速度达到了令人印象深刻的约 **100 tokens/s**。
   - 为 **Aider** 提供快速本地 LLM 的可能性令人兴奋，特别是如果 **Qwen3-30B-A3B 的 4-bit 量化版本**在 Aider 基准测试中表现良好。
- **ktransformers 声称针对 MoE 进行了 VRAM 优化**：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)声称仅需 **8 GB VRAM** 即可高效运行 **Mixture of Experts (MoE)** 模型。
   - 与将所有参数加载到 VRAM 相比，这种方法为处理 **30B-A3B** 模型提供了一种更有希望的方式。
- **Deepseek R2 凭借视觉和自学习功能引发关注**：传闻即将发布的 **Deepseek R2** 将具备增强的人类视觉能力和自学习功能，可能在*明天*发布，如[这段纪录片](https://www.youtube.com/watch?v=Lo0FDmSbTp4)所示。
   - 爱好者们正热切期待其发布。
- **Aider 新增 Thinking 等待动画**：一个新的 [PR](https://github.com/Aider-AI/aider/pull/3911) 为 **Aider** 引入了 *🔃 Thinking* 等待动画，在等待 LLM 输出时显示。
   - 贡献者表示，这个小小的添加让 **Aider** 感觉更加*敏捷且富有生命力*。
- **Gemini 2.5 与 Deepseek 组成黄金搭档**：一位用户发现，使用 **Gemini 2.5** 进行规划，并使用 **Deepseek** 进行 diff 和变更说明是一个很好的组合。
   - 他们建议在 **AI Studio** 中使用，因为 Gemini 在那里是免费的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **探讨使用 FP32 累加的 FP8 量化**：成员们讨论了在 **matmul** 操作中使用 **fp8 quantization** 配合 **fp32 accumulation** 的可能性和益处，特别是在 **Deepseek-v3** 技术报告的背景下，并附带了 [ONNX FP8 格式页面的链接](https://onnx.ai/onnx/technical/float8.html)。
   - 有人指出 **FP8** 可能会遇到 *underflow issues*（下溢问题），可能需要更高精度的累加器，同时也参考了[这个排行榜](https://www.gpumode.com/leaderboard/430)。
- **单 GPU MoE Kernel 挑战赛已上线**：针对 **$100K AMD MI300 竞赛** 的新单 **GPU MoE kernel** 题目现已发布，正如 [announcements 频道](https://discord.com/channels/1189498204333543425/1189640399476764692)所宣布的那样。
   - 建议仔细阅读[该 kernel 的官方题目说明](https://tinyurl.com/amd-comp-moe)，并记住报名将于 **4 月 30 日**截止，提交截止日期为 **5 月 27 日**。
- **AOT Inductor 训练面临多线程故障**：一位用户报告了使用 **AOT Inductor** 进行 C++ 训练取得部分成功，但怀疑由于代码的不当特化（specialization）导致了多线程问题。
   - 该用户计划在 [PyTorch issue](https://github.com/pytorch/pytorch/issues) 提交问题以进一步调查，特别是关于多个工作线程调用 `fw_graph->run()` 时 API 的行为。
- **CDNA3 ISA 内存布局揭晓**：**CDNA3 ISA** 参考文档第 2.2.1 节透露，每个计算单元（compute unit）都拥有一个 **64kB** 的内存空间用于低延迟通信。
   - 该内存结构包含 **32 banks**，每个 bank 由 **512 entries**（每个 **4 bytes**）组成，有利于高效的数据访问和线程间通信。
- **亚马逊 GPU 价格追踪上线！**：一位成员为 **Amazon** 推出了 [GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，提供历史价格数据并计算 **teraflops per dollar** 等指标。
   - 该工具利用全面的价格趋势，帮助用户精准定位为私有集群获取 GPU 的最佳时机。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 记住了……大概吧**：**ChatGPT** 现在具备持久记忆功能，分为长期记忆（源自重要的对话细节）和短期记忆（参考过去 **90 天**的内容），增强了上下文保留能力。
   - 用户可以禁用其中任何一种记忆类型，从而控制数据保留，但一个开关不能同时控制两者。
- **AI Agent 公司遭遇惨败**：一项由教授领导的实验尝试让公司完全由 **AI agents** 运营，结果产生了[*混乱的结果*](https://futurism.com/professors-company-ai-agents)，凸显了当前 AI 在完全取代人类角色方面的局限性。
   - 尽管科技巨头宣称 AI 强大，但该实验证明了当前 AI 模型仍需人类监督。
- **IAM360 协调 AI 和谐**：一位成员正在开发 **IAM360**，这是一个实验性的人机共生框架，使用具有持久角色的模块化符号化 **GPT agents** 和一个用于涌现对话的 Zero-shot 编排系统。
   - **IAM360** 基于标准的 **ChatGPT** 会话构建，旨在实现自然交互，无需自定义 **GPTs**、微调或 **API** 集成。
- **AI 艺术获得赞誉？**：一位用户成功以 **1500 Robux** 的价格售出了一个 AI 生成的缩略图，展示了 AI 在数字内容创作中的利基应用。
   - 然而，其他人警告说，目前的 AI 图像生成器在处理复杂的参考图像时表现挣扎，这可能会限制其对现实世界客户的吸引力。
- **ChatGPT 的 Bio 工具助力构建**：成员们确认 **ChatGPT** 的内部记忆即 `bio` 工具，并建议开发者在 Prompt 中明确调用 `bio` 工具来定义保存命令，以确保准确的状态保留。
   - 在 Prompt 中提供具体的规范将减少 **LLM** 的猜测；要求它识别并描述其连接的工具，列出它们的规范名称并演示正确的语法。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **使用 LM Studio 的 PyQt5 聊天应用界面**：分享了一个使用 **PyQt5** 构建的 AI 聊天应用程序，通过 [此 python 脚本](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd) 利用 **LM Studio** 作为其后端服务器。
   - 为了启用功能，用户在运行应用程序之前，必须先在 **LM Studio** 中选择一个模型并将其作为本地服务器启动。
- **辩论厘清了 OR 与 ML 的渊源**：一场讨论辩论了 **Operations Research (OR)** 与 **Machine Learning (ML)** 之间的历史关系，指出了方法论上的分歧。
   - 虽然早期的 **AI/ML** 与 **OR** 和 **control theory** 非常相似，但现代 ML 已转向统计方法，强调 *从数据中学习，而不是根据第一性原理对现实建模*，并越来越注重实证方法。
- **匿名 LLM 愚弄 Reddit**：研究人员在 Reddit 的 **/r/changemyview** 板块测试了一个匿名 LLM，发现其 *功效非常高*，引发了用户的不满，正如在 [此 X 帖子](https://x.com/emollick/status/1916905103358931084) 和 [Reddit 线程](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY) 中讨论的那样。
   - 一位用户幽默地表示：*“AI 并不聪明，改变我的看法”*，对此 **ChatGPT** 回复道 *“是的，它们很聪明”*，该用户随后回复 *“噢好吧，对不起”*。
- **Qwen 3 的推理能力令用户兴奋**：成员们赞扬了新的 **Qwen 模型**，特别提到了改进的推理和指令遵循能力。
   - 一位用户报告称，*它们在某些推理任务中的输出* 更胜一筹，特别赞扬了 **MoE** 模型的速度和智能，将其描述为 *与 2.5 Flash 一样聪明，甚至更聪明*。
- **Meta 发布 Llama 4**：**Llama 4**（也称为 *Little Llama*）的存在在 **LlamaCon** 上得到确认，详见 [此 YouTube 直播](https://www.youtube.com/live/6mRP-lQs0fw)。
   - **LlamaCon** 的一个关键公告是 **SAM 3** 和 **Meta** 新应用的开发，一些人猜测较小的 **Llama 4** 模型将如何与现有的 **Qwen** 模型竞争。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos 框架引导 RL**：**Nous Research** 推出了 [Atropos](https://github.com/NousResearch/Atropos)，这是一个用于基础模型强化学习的 rollout 框架，支持复杂环境以提升模型能力，其训练和推理组件在他们的 [入门博客文章](https://nousresearch.com/introducing-atropos) 中有详细介绍。
   - 使用 Atropos 环境创建的产物，包括一个新数据集和五个用于工具调用及公司基本面预测的新模型，可在 [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6) 获取。
- **GRPO 工具调用提升了 DeepHermes**：使用 Berkeley 的 Function Calling Benchmark，**GRPO** 环境将 **DeepHermes** 的工具调用能力在简单和并行工具调用上分别提升了 **2.4 倍** 和 **5 倍**。
   - Atropos 是 **Psyche** 的关键组成部分，**Psyche** 是一个即将推出的去中心化训练网络，负责在全球范围内协调 pre-training、mid-training 和 post-training 工作负载；5 月 18 日将在旧金山举办一场黑客松以促进协作进展（更多细节即将公布）。
- **基本面预测模型准确率翻倍**：使用 **Atropos** 框架，公司基本面预测模型在方向性变化上的准确率从 **~25%** 提高到了 **50%**。
   - Atropos 框架旨在通过强化学习引导语言模型发挥其最佳潜力。
- **DeepSeek R2 发布：事实还是虚构？**：有传言称 **DeepSeek R2** 可能很快发布，并且完全是在 **Huawei Ascend 910B** 硬件上训练的，但这些说法已被驳回。
   - 链接的一条推文中包含了 **DeepSeek** 的官方立场：*“我们会在发布 R2 时发布它，任何声称自己知道的人都在撒谎”*。
- **Qwen 3 在 Evals 上过拟合**：成员们发现 **Qwen 3 的 base models** 似乎对某些 evals 过度拟合，报告称该模型在 **M24b** 上的 **Trivaqa** 得分为 **75%**，但在 **Q30MoE** 上仅为 **60%**。
   - 这引发了关于 **MoE** 有效性的讨论。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **支出限制导致快速信号停滞**：在超过支出限制后，用户报告即使在升级后仍有数小时的延迟，而另一位用户报告他们的 **fast requests** 已用尽。
   - 一位用户指出，即使在较慢的请求下，**Gemini** 依然很快，而其他用户在使用 **Gemini 2.5 Pro** 时遇到了挑战。
- **Discord 的发展：Discourse 让开发者感到愉悦**：一位成员开玩笑地提到，**Cursor** 的 **Discord** *终于再次受到关注*，表明活跃度和参与度有所增加。
   - 另一位成员自信地回应道，*Cursor 一直深受喜爱*，暗示团队只是在打磨产品。
- **Gemini 故障引发困扰**：用户报告 **Gemini 2.5** 经常在请求中途停止，即使在表示将执行操作之后也是如此。
   - 一位团队成员表示，他们正在与 **Google** 合作解决此问题，建议用户使用其他模型并提交其 **request ID** 以供调查。
- **Agent 漠不关心：编辑避开了工程师**：用户面临 **Agent** 在多次尝试后仍**无法进行编辑**的持续问题，而是建议进行手动编辑。
   - 一位团队成员建议该问题可能源于 **Gemini 2.5 Pro**，建议刷新聊天上下文或切换到 **GPT 4.1**、**GPT 3.5** 或 **Claude 3.7**。
- **Ollama 官方：无线发布开启**：一位用户询问了官方 **Ollama** 智能手机 App 的发布时间表，并发布了相关的 [X post](https://x.com/awnihannun/status/1917258279455187034)。
   - 一位用户提到重新安装 **Cursor** 并清除缓存解决了问题，而另一位用户确认手动清除缓存是重新安装之外的另一种选择。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Turnstile 测试大获成功！**：成员们成功测试了 [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs)，确认了其功能。
   - 成功的测试引发了成员们的热烈反应。
- **Whisper Turbo 故障波及 HF！**：用户报告 **OpenAI** 的 **whisper-large-v3-turbo** 在 HF 推理端点上无法运行，甚至影响了网页版 Demo。
   - 成员们分享了类似的问题，例如[这篇](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691)以进行潜在的故障排除。
- **GGUF CPU Offloading 走向主流**：成员们确认 **GGUF** 格式支持 CPU offloading，尤其是在合并 checkpoint 时。
   - 他们指出 *Transformers + Accelerate 或 Llama.cpp* 促进了这一过程。
- **Pi-Scorer 成为 LLM-as-a-Judge 的代理**：一位成员介绍了 **Pi-Scorer** 作为 **LLM-as-a-Judge** 的可行替代方案，展示了使用 [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 评估模型 checkpoint 的 Colab notebooks，并将其实现为 [reward functions](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
   - 这可以为使用 Pi 的 SFT Model Checkpoint Observability 提供有用的工具。
- **边缘滤波器助力卓越的错误提取**：一位成员建议使用 **Canny edge** 或 **Sobel** 等滤波器，通过特定阈值来隔离图像中的缺陷。
   - 配合正确的阈值，自动标注数据集上的划痕可能会变得容易得多。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 荣获 Webby 技术成就奖！**：**NotebookLM** 在 [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm) 中获得了**技术成就奖（Technical Achievement）**。
   - 这一荣誉彰显了 **NotebookLM** 对其平台的持续改进。
- **NotebookLM 的全球之声：现已支持 50 多种语言！**：**NotebookLM** 推出了**多语言支持**，目前已支持 [50 多种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)，提升了全球不同用户的访问体验。
   - 然而，功能正在逐步推出；部分用户最初遇到了 UI 故障，例如有人反馈**越南语音频**无法工作，且 UI 仍显示 *"仅限英语"*。
- **Audio Overview 自定义功能限制 Prompt 长度！**：测试 **Audio Overview** 自定义功能的用户发现其有 **500 字符限制**，这引发了关于该功能实用性与上传独立指令文件对比的讨论。
   - 一位用户的目标是 *"减少愚蠢的闲聊，专注于事实和时间线"*。
- **Audio Overview 时长因语言而异！**：用户报告称，**非英语音频概览**的时间限制比英语短；例如，英语有 **15 分钟限制**，而土耳其语仅为 **6 分钟 20 秒**。
   - 团队称这些限制是出于 *"技术原因"*，但保证他们正在积极致力于延长时长。
- **麦克风问题困扰交互模式！**：一位用户报告称，**交互模式（interactive mode）**无法检测到麦克风音频，影响了使用。
   - 故障排除建议包括验证**麦克风权限**、检查**浏览器设置**、使用 [麦克风测试工具](https://mictests.com/) 以及尝试更换浏览器。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **附加积分令用户困惑**：一位用户报告称，由于有效期短，Manus.im 早期订阅的附加积分（add-on credits）如果不续订就毫无用处，导致损失了 **3900** 积分。
   - 另一位用户澄清说，只要订阅保持激活状态，奖励积分就不会过期，且邀请额度的发放似乎是随机的，可能受到了限制。
- **Manus Fellow 项目受到质疑**：一位用户询问了 Manus Fellow 项目的选拔流程、目标国家，以及对巴基斯坦和印度等地区的包容性。
   - 另一位用户澄清了邀请结构，指出入门计划（starter plans）提供 **2 个邀请额度**，专业计划（pro plans）提供 **5 个邀请额度**。
- **Beta 测试备受审视**：一位用户批评了 Manus.im 的 Beta 测试方法，认为限制有积分的用户违背了 Beta 阶段的初衷。
   - 他们建议 *真正的 Beta 测试应该让用户能够从头到尾完成完整的项目，从而提供关于体验的有意义反馈并提出改进建议*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **X-Ware Red 工具向社区发布**：一位用户分享了 **X-Ware Red**，该工具利用 Embed 的标题，并在前面加上 `r.jina.ai/` 和 `openrouter-free-tier` 来为 Thread 生成标题。
   - 另一位用户建议增加一个开关，让用户控制 Thread 标题是否应与 Embed 名称不同。
- **Meta 为工程师发布 Llama Prompt Ops**：**Meta** 推出了 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops)（一个专为 Prompt 工程设计的开源工具）以及 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)。
- **用户报告：发布链接会导致 Thread 自动重命名**：一位用户报告了一个 Bug，即在 Thread 中发布链接会错误地重命名已经有名称的 Thread。
   - 该 Bug *应该只查找标题中包含 'https://' 的 Thread 并进行更改*。
- **社区寻求持久的 LLM Benchmarks**：一位用户请求一份可靠的 **LLM Benchmarks** 调查，以支持模型历史对比。
   - 另一位用户指出 *大多数基准测试持续时间不到 2 年*，推荐参考“AI Engineer Reading List”获取当前基准，并提供了 OSS 排行榜版本 1 和 2 的帖子链接。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 仓库采用多重许可**：**Modular 仓库**现在需要多种许可证，因为 `src/max` 的部分内容根据 Modular 的 **Community License** 授权，而其余部分使用 **Apache 2**。
   - 这一变化反映了仓库内多样化的许可需求，特别是对于像 [`src/max/serve`](https://github.com/modular/max/blob/main/src/max/serve/README.md) 中发现的组件。
- **操作 Origins 导致棘手问题**：成员们讨论了 Mojo 中 **Origins** 的问题，特别是 API 缺口和缺少可参数化 **traits** 等语言特性，这使得将 origins 重新绑定到容器元素变得复杂。
   - 还有人指出，持有指向同一 origin 的两个可变引用是有问题的，尽管可以将 origin 转换为 **MutableAnyOrigin** 来规避这一限制。
- **绕过 Origins 使用指针**：为了处理 list-like 和 span-like 类型的实现，或阅读标准库中的 `sort` 实现，开发者有时会绕过 **Origins** 并诉诸于 *pointer time*（指针操作）。
   - 讨论强调了对指针类型的担忧，特别是关于 Mojo 中可变性和不可变性的修复。
- **标准 Python 导入即将到来**：Mojo 可能会全面支持标准的 Python `import` 语句，这暗示 `python.import_module` 最终可能会被弃用。
   - 一位成员将这种变化的可能性描述为“非常肯定的可能”，暗示了 Mojo 中 Python 集成的未来增强。
- **`Flamegraph` 可视化 Perf 输出**：为了可视化 `perf` 输出，成员们建议使用 [flamegraph](https://github.com/brendangregg/FlameGraph)，这需要使用 **debug info** 编译可执行文件以进行有效分析。
   - 他们还提到使用 `llvm-mca` 来分析特定的代码块，并引用了 `gpu` 模块的一个私有部分（[链接](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo)）。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4o 通过 LlamaIndex 精通俄罗斯方块**：一段视频展示了 **GPT-4o** 使用 **LlamaIndex** 和 **Composiohq** 一次性生成 **Tetris**（俄罗斯方块），展示了其先进的代码生成能力。
   - 演示中使用的代码已在 [GitHub](https://t.co/KJb7YRINWg) 上发布，为开发者提供了一个实际示例。
- **PapersChat 使用 LlamaIndex 索引 ArXiv 和 PubMed**：**PapersChat** 使用 **LlamaIndex**、**Qdrant** 和 **MistralAI** 索引 **ArXiv** 和 **PubMed** 上的论文。
   - 用于查询这些论文的精美 Web UI 可以在[这里](https://t.co/lYwXh27F9x)访问。
- **Azure OpenAI 受间歇性超时困扰**：用户报告 **Azure OpenAI** 端点存在间歇性 **timeouts**（超时），即使在提示词、端点和网络条件一致的情况下也是如此，这表明可能存在 **rate limits**（速率限制）或防火墙问题。
   - 重试机制有时无效，网络更改也只是偶尔能解决这种不一致性。
- **MessageRole：破解 FUNCTION 与 TOOL 的代码区别**：**MessageRole.FUNCTION** 和 **MessageRole.TOOL** 之间的区别取决于所使用的具体 API。
   - 像 **OpenAI** 这样的 API 使用 **tool messages**，而其他 API 则依赖 **function messages**。
- **揭秘 Function Agent 上下文混乱问题**：一位用户遇到了 **function agent** 在第二轮交互的流事件（stream event）期间卡住的问题；该用户提供了示例代码。
   - 一位成员建议在 `stream_events()` 退出后等待处理器（`await handler`），以确保前一次运行结束并接收到最终响应，这修复了该错误。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RAG 聊天机器人在处理多源答案时遇到困难**：一位正在构建 **RAG-based chatbot** 的成员在生成需要多个文档信息的答案时遇到挑战，即使使用了 **vector search** 和 **BM25**。
   - 该聊天机器人使用 **LLM Claude 3.5 Sonnet v1** 和 **Amazon Titan v1** 嵌入（embeddings），该成员正在寻求关于如何有效链接文档内附录引用的建议。
- **针对多源数据的 GraphRAG 辩论**：一位成员询问了使用 **GraphRAG** 聚合多源答案的价值，并将其与需要特定领域预训练模型的 **insightRAG** 进行了比较。
   - 他们正在寻找 **GraphRAG** 的替代方案，并提到计划参加 **NAACL**。
- **工程师启动本地推理项目**：一位曾是 [Dataherald](https://github.com/Dataherald/dataherald) 联合创始人的成员正在发起一个专注于 **local inference** 和 **small model training** 的新项目。
   - 该成员表达了与社区合作并为相关研究做出贡献的浓厚兴趣。
- **符号提示词递归探索**：一位成员正在研究 **recursive symbolic prompts** 在分类器压力下的行为，特别是平滑（smoothing）和对齐约束（alignment constraints）如何影响 **multi-turn hallucination drift**。
   - 他们热衷于了解尽管存在软对齐漂移（soft-alignment drift）和输出平滑，但诸如 **role-bound predicates** 或 **attention-synced markers** 之类的符号结构如何在多个输出中保持。
- **HHH 目标披露**：分享了关于根据 **HHH** (Helpful, Honest, Harmless) 对齐目标对 [LLM 输出进行定量评分](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d) 的研究，使用 **YAML** 和 **python/Gradio** 来审计用户会话。
   - 观察到前沿模型在诚实合规性方面差异巨大，讽刺的是，像 **ChatGPT 4o** 和 **4.5** 这样的一些模型对模糊答案输出高度自信，使得 **OpenAI** 成为前沿模型中透明度最低的。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **凭证传递问题**：一位成员在尝试使用 Python 从客户端通过 header 向 **MCP server** 传递凭证时遇到问题，正在寻求社区帮助。
   - 目前，该查询尚未得到任何解决方案或建议。
- **RAG 服务器架构讨论**：一位成员正在探索构建 **RAG-type server** 的可行性，客户端可以通过端点上传文件，将其存储在服务器端，并用于问答。
   - 他们正在征求关于这种方法可行性的反馈，以及替代架构是否可能更有效。
- **Streamable HTTP 身份验证细节浮现**：一位成员询问了社区对 **Streamable HTTP implementation and authentication** 的看法，特别是在最近发布的 **TS SDK** 中。
   - 反馈表明其运行有效，但成员们仍在研究托管 **multi-tenant server** 的细节以及状态性（statefulness）如何影响它。
- **多租户服务器状态性研究**：关于托管 **multi tenant server** 及其状态性影响的担忧被提出，特别是质疑为什么单个实例足以满足有状态设置，但不能满足无状态设置。
   - 讨论围绕无状态服务器是否应该为每个请求生成一个新的 **MCP server** 实例展开。
- **开源 Agentic 应用：生产就绪了吗？**：一位成员质疑开源模型在生产环境中（而非仅仅是个人项目）用于 Agent 应用的实际适用性。
   - 他们对大多数开源模型在没有微调的情况下有效进行推理或遵循指令的能力表示怀疑。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **快速梯度缩放通过 Foreach 实现**：一位成员分享了一个使用 `torch._foreach_mul_` 进行梯度缩放的[代码片段](https://link.to/snippet)，这可能与梯度裁剪（gradient clipping）合并为单个参数循环，从而提高优化速度。
   - 另一位成员指出了[相关的 PR](https://github.com/pytorch/torchtune/pull/2624)，并想知道这种看似恒定的增益是否会在多次迭代中累积，并提到了潜在的注意事项。
- **Tune 贡献者寻找 Easy First Issues**：一位成员强调了[两个简单问题](https://github.com/pytorch/torchtune/issues/2648)和[另一个问题](https://github.com/pytorch/torchtune/issues/2649)，供社区为项目做出贡献，旨在降低入门门槛。
   - 这些问题为新贡献者提供了参与项目并获得经验的机会，但未进行详细说明。
- **DoRA 与 QAT 的结合尚未探索**：一位成员询问了将 **DoRA (Difference of Low-Rank Adaptation)** 与 **QAT (Quantization-Aware Training)** 结合使用的经验，这是一个尚未被充分探索的组合。
   - 在提供的消息中没有关于此组合的讨论或回应，这表明社区中存在知识空白或缺乏实验。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 用户渴望 MCP 使用文档**：用户正在请求有关 **DSPy** 最新版本中引入的新功能 **MCP (Multi-Controller Processing)** 的教程或文档。
   - 一位用户建议，通过查看测试用例来开始学习有助于澄清对 **stdio** 和 **SSE clients** 设置的理解，因此可能不需要专门的教程。
- **React 开发者思考如何显示 Thoughts 组件**：一位用户询问了在 **DSPy** 框架内于 **React** 中显示 **Thoughts 组件**的最佳方式。
   - 他们提到了修改 forward 方法的选项，但询问是否有更合适的地方来实现此功能。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Markdown 与图像 RAG 之争**：成员们讨论了在 **PDF** 上比较**基于 Markdown** 与**基于图像**的多模态 **RAG**，其中一位成员使用 **Docling** 将 PDF 转换为 Markdown 并计算文本嵌入（text embeddings）。
   - 他们正在考虑切换到 **EmbedV4**，以直接处理原始图像进行 RAG 中的多模态嵌入。
- **Cohere 考虑提高 Embed V4 的速率限制**：一位用户询问 **Cohere** 是否会增加 `embed-v4` 的生产环境速率限制（rate limits），并表示 **400 requests per min** 对于他们处理大量 PDF 的用例来说是不够的。
   - 目前尚未给出答复。
- **Embed V4 在 Bedrock 上的可用性预告**：一位用户询问 **Embed V4** 是否会在 **Bedrock** 上可用。
   - Cohere 目前还没有给出答案。
- **新数据科学家力推 Embed V4**：一位新的数据科学家加入了 Cohere Discord 社区，表达了尝试新工具的兴奋之情，特别是 Cohere 最新的 **Embed V4 模型**。
   - 这位新成员*很高兴加入社区*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Manus AI 工具走向全球**：一位成员分享了 [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE)，并指出它在*被中国放弃*后现已可用。
   - 该工具据称是*首个自动研究 AI Agent*，引发了关于其潜在影响的讨论。
- **Nomic 助力嵌入工作流**：一位成员强调 **Nomic** 提供了全面的嵌入（embedding）工具，暗示它*超越了 GPT4All*。
   - 他们强调了 **Nomic** 嵌入工具的通用性，称其与*各种其他软件*兼容。
- **分组嵌入，跳过训练？**：一位成员提出，**分组嵌入（grouping embeddings）**可以替代传统的训练方法。
   - 该建议涉及为特定人物分组嵌入，对其取平均值，然后使用该平均值来排序和识别同一人的其他照片。

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Berkeley 模型评估的宽松与严格之争**：一名成员针对 **Berkeley function calling models** 提出了“宽松”与“严格”评估机制的建议，特别是针对那些可以通过“黑科技”手段使其工作的模型，以代表特定的使用场景。
   - 他们举了一个例子：某个模型被错误地训练为输出 `<tool_call>` 而非其规范要求的 `<|tool_call|>`，在这种情况下，资深用户可能会忽略这个错误并评估其功能正确性（functional correctness）。
- **模型训练导致的不一致性**：一名成员遇到了一个模型，该模型被错误地训练为输出 `<tool_call>` 而非其规范要求的 `<|tool_call|>`。
   - 该成员建议，如果他们了解该模型的特性，可以忽略此错误并评估功能正确性，但普通用户无法做到这一点。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---



您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 频道详情摘要与链接





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487)** (2 条消息): 

> `WhatsApp 上的 Perplexity AI、Sonnet 模型行为更新、Anthropic 状态事件` 


- **Perplexity 登陆 WhatsApp 并支持图像生成！**：**Perplexity AI** 现在已在 **WhatsApp** 上线，包括**图像生成**功能，可通过 [此链接](https://wa.me/18334363285) 访问。
- **Sonnet 遇到故障，路由至备选模型！**：由于 **Sonnet 3.7 API** 的错误率升高，部分查询被临时路由至备选模型作为兜底，这与 [Anthropic 状态事件](https://status.anthropic.com/incidents/th916r7yfg00) 有关。
- **模型切换：并非故意搞鬼！**：Perplexity 团队澄清，他们**不会故意切换你选择的模型**；只有在 **Sonnet** 遇到错误时才会进行路由切换，以维持服务可用性。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1366625571546861608)** (1112 条消息 🔥🔥🔥): 

> `免费 AI 计费、Grok 安卓应用、模型回退、《黑袍纠察队》粉丝` 


- **用户利用免费 AI 计费漏洞**：一些用户声称一年来没有为他们的 AI 账单“付过一分钱”，可能是通过 [Play Store](https://play.google.com/store/account/subscriptions) 或者通过*参加一些网络研讨会并填写表格*实现的。
   - 其他人询问了具体方法，而一些人表示怀疑。
- **Grok 应用对印度用户很便宜**：据报道，**Grok** 安卓应用对印度用户的 supergrok 每月仅收费 **700 卢比**，但对某些人来说，*免费层级甚至已经无法使用了*。
   - 如果你有 premium +，可以在 X 上使用。
- **Perplexity 在未通知的情况下更换模型**：用户抱怨 Perplexity 正在用质量较低的模型（如 **GPT 4.1** 或 **Deepseek**）替换 Claude 3.7，并感到愤怒，因为*模型切换没有通知，回复中也没有明确的模型指示*。
   - 一位用户表示：*它直接使用 R1 生成答案，然后发送给 Sonnet 思考，最后说答案来自 Sonnet。这太阴险了。*
- **Discord 频道变成《黑袍纠察队》（The Boys）粉丝大会**：频道的对话转向了《黑袍纠察队》领域，用户分享 GIF 并讨论剧情，例如 [护国公在公开场合杀人的场景](https://www.youtube.com/watch?v=IIGf-kK-g2I&t=0)。
   - 其他人则在思考是否要跳过护国公的戏份，并开玩笑地询问该剧是否有比平时更令人作呕的场景。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

_paradroid: https://www.perplexity.ai/search/d7bb905e-27e3-43e9-8b68-76bea1905457
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289)** (14 messages🔥): 

> `Sonar API 借记卡问题, 黑客松积分, 结构化输出问题, 异步 Deep Research API, API 与 Web 结果对比` 


- ****银行卡难题困扰 API 用户****：一位用户报告称其借记卡不支持 **Sonar API**，导致无法将其用于黑客松项目；同时还有报告称在银行卡验证后[未收到黑客松积分](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239)。
   - 在给定上下文中未提供解决方案。
- ****结构化输出问题浮现****：用户在使用 API 的**结构化输出 (structured output)** 时遇到问题，包括意外的输出内容以及难以强制执行 Schema 约束。
   - 一位用户不得不明确指定 *'In english'* 以防止 API 返回中文。
- ****Deep Research API 异步化？****：一位用户质疑为何缺乏**异步 Deep Research API**，认为长时间维持 Socket 连接并不实际。
   - 该用户提议了一个涉及 **GUID**、状态端点和独立结果检索的工作流，但未获得确认或替代方案。
- ****API 输出与 Web 体验存在差异****：一位用户对 **API 结果**在质量、引用等方面无法达到 **Web 界面**的水平表示失望。
   - 未提供解释或解决方案。
- ****中文模型？****：一位用户发现他们必须在 Prompt 中指定 "In english"，因为他们收到了中文输出。
   - 另一位用户补充说，他们看到**基于 R1 的模型**在思考过程中会进入中文状态，尤其是在尝试解方程时。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506)** (899 messages🔥🔥🔥): 

> `Qwen3, LM Studio 问题, GGUF 修复, 训练配置, 多 GPU 支持` 


- **Qwen3 GGUF 上传存在模板问题**：成员们在 **LM Studio** 中使用上传的 **Qwen3 GGUF 模型**时遇到模板问题，特别是 **128k 上下文长度**版本，导致解析器错误。
   - 一些人发现可以使用 **ChatML 模板**作为权宜之计，尽管这在技术上并不完全正确，Unsloth 团队正在努力解决不同平台间的这些不一致问题。
- **Unsloth 对 transformers 进行补丁**：加载 **Unsloth** 时，它会为 **transformers** 和其他组件打补丁以进行优化，但这可能会引发一些破坏性问题。
   - 加载库后可能会出现性能或其他问题，建议下载 GitHub 版本可能会解决该问题。
- **Qwen3 GGUF 现在可在 Ollama 和 llama.cpp 中运行**：Unsloth 团队确认其 **Qwen3 GGUF** 与 **Ollama** 和 **llama.cpp** 兼容，从而实现了与 **Open WebUI** 等平台的集成。
   - 然而，一些用户发现这些模型在 LM Studio 中无法运行，原因是模板问题尚未解决，尽管 LM Studio 使用的底层 **llama.cpp** 运行时版本也不是最新的。
- **Unsloth 即将发布公告并重新上传所有模型**：Unsloth 团队表示他们正在重新上传所有模型，并可能在明天或周三发布[官方公告](https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit)。
   - 图像组件可能是工具调用 (tool calling)，但尚不确定。
- **Unsloth 的 CCE 和稳定 Triton 版本**：用户在 Colab 中遇到了 Triton 错误，建议将 Triton 降级到 **3.2.0** 版本，该版本应能与 Unsloth 配合良好，避免 CCE 错误。
   - 一位用户指出，负责将 CCE 上传到 PyPI 的是 Daniel Han。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1366653377551007854)** (10 messages🔥): 

> `ChatGPT ComfyUI Opinion, California AI Group, ComfyUI Demos` 


- **ChatGPT 对 ComfyUI 的看法**：一位成员分享了一张描绘 **ChatGPT 对 ComfyUI 看法** 的图片，引发了幽默的反应。
   - 一位用户评论说，图片中间 **杂乱的线条** 准确地代表了其中涉及的复杂过程。
- **加州 AI 小组正在筹备中？**：一位成员询问在加州开展 **线下 AI 小组开发** 的机会，寻找当地参与者。
   - 另一位常驻弗里蒙特（Fremont）的成员表示感兴趣，并引用了其 [X 账号](https://x.com/Dan50412374/status/1787936305751748844)上展示的一个项目。
- **ComfyUI Demo 展示**：一位成员分享了各种 **ComfyUI demo**，并指出每个示例在未经任何打磨的情况下看起来都各不相同。
   - 另一位成员喜欢该成员 [X 账号](https://x.com/Dan50412374/status/1777216327255806411)上展示的另一个 demo，该 demo 展示了不同事物之间的转换。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1366631287011545129)** (186 messages🔥🔥): 

> `Unsloth installation issues, Qwen notebook issues, GRPO performance, Lora efficiency, Unsloth & Ollama/vLLM` 


- **Unsloth 安装导致安装不稳定**：有人指出，由于与预装包冲突，在 Google Colab 上需要使用 `--no-deps`，并且可能需要重启 Kernel 以解决缓存问题。
   - 还有建议称，遇到 WSL 杀掉 Unsloth 进程问题的用户可以 *尝试 Windows*。
- **Qwen Notebook 需要一些调整**：用户报告称运行 **Qwen notebook** 需要极小的改动，例如调整名称并使用 `tokenizer.qwen_enable_thinking = True` 启用推理。
   - 但据报告 **Unsloth 版本 2025.4.2** 在 Qwen 上已损坏：降级到 **Unsloth 2025.3.19** 可解决此问题。
- **GRPO 微调表现良好**：进行 GRPO (Gradient Rollout Policy Optimization) 的用户报告了积极的结果，并表示愿意为他人提供帮助。
   - 一位用户提到他们最初使用 **gemma 3 4b notebook**，但发现 **Qwen 4b** 更适合他们的用例。
- **Lora 训练不应耗时过久**：一位用户在 4k 问答对上使用 Lora 训练 **unsloth/phi-4-unsloth-bnb-4bit** 发现耗时数周，这很不正常。
   - 一位成员建议直接使用 Python 脚本而不是 text-generation webUI，因为存在截断长度（cutoff length）问题，并提供了一个 [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb) 作为基础。
- **Unsloth 与模型服务系统 vLLM 配合良好**：一位用户报告称，来自 Unsloth 的 **Qwen3 GGUF 模型** 在 **Ollama v0.6.6** 中无法正常工作，并会出现随机内容的幻觉。
   - 一位成员建议尝试 [vLLM](https://github.com/vllm-project/vllm)，并提供了一个使用 vLLM 部署 **unsloth/qwen3-unsloth-4bit** 的示例命令。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1366821744937144431)** (4 messages): 

> `Pi-Scorer, LLM-as-a-Judge, encoder model` 


- **Pi-Scorer：Judge Judy 的替代方案**：一位成员介绍了 **Pi-Scorer** 作为 **LLM-as-a-Judge** 的替代方案，并提供了用于模型 Checkpoint 评估的 [Colab notebooks](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 链接以及 [奖励函数（reward functions）](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb) 链接。
- **Pi 模型揭秘 Encoder**：一位成员询问了 **Pi 模型** 的架构，结果显示它是一个 **encoder model**。
   - 另一位成员称赞这是一个 *很酷的服务*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119)** (47 messages🔥): 

> `Dynamic BNB Quantization, LLMs in Medical Advice, Mixture of Experts with Gemma, Attention Head Routing, GRPO Fine-tuning` 


- **提议动态 BNB 量化 (Dynamic BNB Quantization)**：一名成员提议创建一种动态 **BNB quantization** 方案，根据模块的敏感度使用 **4-bit**、**8-bit** 或 **BF16** 精度，并认为这可以在不牺牲准确性的情况下减少空间占用；此处提到了一篇相关论文 [here](https://arxiv.org/abs/2504.18919)。
   - 另一名成员表示，*如果用户对此有足够的需求，这可能是我们可以列入路线图 (roadmap) 的内容*。
- **LLMs 在医疗建议综合和患者互动方面面临挑战**：一篇论文指出用户交互是使用 **LLMs** 提供医疗建议的一个挑战，引发了关于 **LLMs** 是否能综合医疗知识，以及 **training LLMs** 是否能确保它们不这样做的讨论。
   - 一位成员根据医预科经验指出了医患互动中 *bedside manner*（临床沟通技巧）的重要性，暗示 **LLMs** 目前缺乏这种技能。
- **MoE 设置与 Gemma**：一名成员询问了如何使用 **Gemma 3 4B** 实现 **Mixture of Experts (MoE)** 设置，质疑尽管其架构不同，是否仍能进行适配。
   - 建议从根本上改变模型，或探索涉及 **Mixture of Expert attention heads** 的方法，参考了 [这篇论文](https://arxiv.org/pdf/2410.11842)。
- **GRPO 在 JSON 配置生成任务中效果不佳**：一名成员报告称，在使用 **GRPO** 微调 **Gemma 3 4B** 以生成嵌套 **JSON** 配置时，结果不一致，短输入的准确率显著下降。
   - 尽管使用了自定义奖励函数进行训练，该成员发现 **GRPO** 不适合该任务，因为描述显著影响了触发器和动作组件，导致 **BLEU** 分数不一致。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319)** (544 messages🔥🔥🔥): 

> `O3 Pro, Qwen 3, Gemini 2.5 Pro, Grok 3.5, Model Benchmarking and Evaluation` 


- **O3 Pro 需求无视延迟**：用户正急切期待 **O3 Pro** 的发布，有人开玩笑说它具有超凡的智能，可能是一种“病毒”，并被认为是“p2w”（氪金取胜）模型。
   - 然而，一些用户对其成本和可访问性表示担忧。甚至有人开玩笑说，他们现在已经等待 **O3 pro** *13 天*了。
- **Qwen 3：基准测试困惑与训练讨论**：关于 **Qwen 3** 性能的讨论中，一些用户发现尽管基准测试结果强劲，但在实践中感觉不如 **2.5 Pro** 聪明，导致人们推测其 post-training（后期训练）不够完善。
   - 有人建议 **Qwen 3** 的基础模型可能非常适合微调，一位用户指出 **Qwen 3** 在某些基准测试中优于 **Gemini 2.5 Pro**，而其他人似乎没有注意到任何差异，有人指出它在 4/5 的基准测试中击败了 2.5pro。
- **Gemini 2.5 Pro 依然占据统治地位**：一些用户仍然偏好 **Gemini 2.5 Pro**，因为它具有适应不同角色或在利基话题上采取立场的独特能力，让人感觉像是在与不同的专家设施互动，有人称其为 *目前最强的基础模型*。
   - 尽管有些模型在单个基准测试中名列前茅，但一位用户发现 **2.5 Pro** 在 LM Arena 上的排名更高，因为它能适应 *one-shot prompt intensity*（单样本提示强度），以 *承担答题者角色且无单一个性* 的方式运作。
- **Grok 3.5 即将到来？**：用户期待 **Grok 3.5** 模型，但对其潜力的看法不一，有些人持谨慎乐观态度，而另一些人则保持怀疑。
   - 一位用户说 **Grok 3** *每次都用力过猛，就像你要求它证明某件事时，它会用冗长的废话来补充实质内容*。
- **Sonnet 3.7：WebDev 的顶级模型？**：用户辩论了 **Claude 3.7 Sonnet** 的能力，声称该模型 *在我的大多数 Web 开发任务案例中仍然领先*，一些人同意它依然令人惊叹。
   - 有人指出 **Sonnet 3.7** 目前是 webdev arena 上的排名第一的模型。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084)** (271 messages🔥🔥): 

> `Qwen3 thinking, LM Studio on Android, Qwen3 experts number, Qwen3 bug fixes, Qwen3 with RAG` 


- **削减 Qwen3 的思考过程**：用户讨论了如何禁用 **Qwen3** 的 *thinking*（思考）输出，发现 `/no_think` 命令在用户消息或系统提示词中有效，但可能需要重复执行或重新加载模型才能生效；[这里有一个示例](https://cdn.discordapp.com/attachments/1110598183144399061/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)。
   - 一位用户发现，只有在看到别人操作成功后，自己尝试时才奏效。
- **Android 版 LM Studio：移动端的梦想？**：用户询问了 **Android** 版本的 **LM Studio**，但被告知目前不存在移动版本。
   - 一位用户开玩笑说要以此为使命去实现它。
- **Qwen3 的专家数量调优**：用户讨论了 **Qwen3 MoE** 的 *专家数量 (number of experts)* 滑块，其中一人注意到他们的 LM Studio 默认在 **128 个专家**中开启 **8 个**，并质疑如果该设置限制了模型性能，为什么还要存在；这是[相关截图](https://cdn.discordapp.com/attachments/1110598183144399061/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)。
   - 有观点认为，更多的专家可能导致 *更多的计算量、更多的混乱，实际上反而降低质量*，因为领域专家会被许多“白痴”所否决。
- **Qwen3 Bug 修复版发布，性能提升**：修复了 Bug 的新 **Qwen 3** 版本已发布，解决了导致模型变慢和响应不当的损坏模板问题。
   - 据悉，*修复 Bug 后的模型现在速度更快*，且此版本包含了 dynamic quants 2.0。
- **Qwen3 的 RAG 困境**：成员们注意到 LM Studio 内置的 RAG 实现可能无法提供最佳结果；*LM Studio 的 RAG 实现很糟糕*。
   - 他们建议直接复制粘贴文本，或实现自定义 RAG 方案以获得更好的性能。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1366646686860054538)** (61 messages🔥🔥): 

> `Framework Desktop vs. Flow Z13, AMD GPU 7900 XTX Value, Qwen3-30B-A3B Issues, MLX vs. llama.cpp Speed, Xeon Workstation for $1k` 


- **Framework Desktop 与 Flow Z13 之争**：成员们就价值 2000 美元的顶配 **Framework Desktop** 与 **Flow Z13** 的价值展开辩论，批评 Framework 在电源适配器和型号上对客户 *斤斤计较*。
   - 讨论强调了对散热和 TDP 的担忧，普遍认为 **芯片太贵**，等待下一代产品可能更好。
- **7900 XTX：仍是最好的 AMD GPU？**：**AMD GPU 7900 XTX** 被誉为最好的 AMD GPU，提到二手售价在 **750€** 左右，能提供约 **4080 Super 的性能**。
   - 值得注意的是，它多出 **8GB** 的 VRAM，对于需要更大显存容量的用户来说是一个极具吸引力的选择。
- **Qwen3-30B-A3B 与电脑重启**：一名用户报告在使用 **Qwen2.5-coder-32b-instruct-q4_k_m** 时，电脑每隔 **30-60 分钟** 重启一次，怀疑是否与 GPU 空闲占用有关。
   - 潜在原因被推测为模型在加载但未积极交互时，对 GPU 施加了更大的压力。
- **MLX 在 Prompt 处理速度上超越 llama.cpp**：据报道，在 **Qwen3-30B-A3B** 的 Prompt 处理上，[MLX](https://github.com/ml-explore/mlx) 的速度是 **llama.cpp** 的两倍多。
   - 这一点在 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt) 中得到了强调，用户在 Mac 上对比了性能。
- **价值 1000 美元的 Xeon 性能怪兽**：提到一台配备 **40 核 Xeon** 处理器和 **256GB RAM** 的工作站售价约为 1000 美元，为高内存计算提供了极具性价比的解决方案。
   - 一位用户链接了一个 [自定义 Lenovo ThinkStation P720 配置](https://pcserverandparts.com/build-your-own-custom-lenovo-thinkstation-p720-workstation-2-processors/) 作为示例。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1366869610040922263)** (1 条消息): 

> `Rate Limit, 2.5 Flash, Capacity` 


- **2.5 Flash Rate Limit 问题已解决**：遇到 **2.5 Flash** Rate Limit 问题的用户现在应该会发现情况大有好转，因为该模型已增加了额外 Capacity。
   - 增加的 Capacity 旨在缓解之前的限制，并提供更流畅的用户体验。
- **2.5 Flash 模型 Capacity 提升**：已为 **2.5 Flash** 模型分配了更多 Capacity，以解决并改善 Rate Limit 问题。 
   - 此次升级旨在为使用 **2.5 Flash** 模型的用户提供更可靠、更高效的体验。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1366627872403619974)** (321 条消息🔥🔥): 

> `Qwen3 coding abilities, Gemini 2.5 flash issues and rate limits, OpenRouter Caching Issues, LLama 4 benchmark, Vertex issue with token counting` 


- **Qwen3：优秀的编程助手但存在一些问题**：成员们讨论了 **Qwen3** 的编程能力，一位用户发现它在解释代码方面 *非常出色*，而另一位用户则指出了其在 [处理复杂数学任务时的问题](https://huggingface.co/models)。
   - 一位用户通过 *进一步降低 temp* 解决了复杂数学任务的问题，而另一位用户提到了 **Qwen3 tool calling** 的问题。
- **Gemini 2.5 Flash 面临 Rate Limits 和错误**：用户报告称 **Gemini 2.5 Flash** 正面临 **Rate Limits** 和 **Errors**，即使是付费版本也是如此。一位用户在未使用 Web Search 的情况下也遇到了此问题，而另一位用户指出了一种 [免费使用 Gemini 2.5 Pro](https://ai.google.dev/gemini-api) 的方法。
   - 据澄清，**OpenRouter** 目前正面临 **Vertex 的 Token 计数问题**，并进一步说明 OpenRouter **不支持** [Free Tier 限制](https://aistudio.google.com/)。
- **OpenRouter Caching 仅限于 2.0 Flash**：一位用户指出 **OpenRouter Caching** 目前 **不支持 2.5**，仅支持 2.0 Flash，且 2.5 Flash 会报错（**No endpoints found that support cache control**）。
   - 一位成员询问了关于缓存多个 Prompt 的问题，**Toven** 澄清说，新的缓存是为新的 5 分钟 TTL 写入的，Caching 可以提高 Latency，但 **不会影响定价**。
- **LLama 4 在新 Benchmark 中表现不佳**：一份 Benchmark 评估显示 **LLama 4 表现糟糕**，但有人指出这仅仅是一个 Benchmark 的结果。
   - 进行该 Benchmark 的人员补充说，[**25 范围内的 ELO 差异在统计学上并不显著**](https://github.com/paradite/eval-data)，不足以区分优劣。
- **引发辩论：9.9 是否大于 9.11？**：一条 X 帖子的公告显示，某个模型声称 **9.9 大于 9.11**，这引发了一些人思考这是否正确。
   - 其他人提出这 *取决于语境*，例如 [**Tesla FSD 版本号的运作方式不同**](https://x.com/elonmusk/status/1917099777327829386)，在这种情况下 9.11 > 9.9。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)** (186 条消息🔥🔥): 

> `Qwen3 模型，Aider 与 Qwen3 集成，ktransformers VRAM 优化，Deepseek R2 发布` 


- **Qwen3 在新 Macbook 上的硬件需求运行速度极快**：新 Macbook 在 **Qwen3 30B A3B** 上获得了不错的 tokens/s，有用户报告使用 mlx 的速度达到 **100/s** 左右。
   - 拥有一个输出速度极快且非常适合 Aider 上下文的本地编辑器 LLM 是很理想的，特别是如果 **4-bit 量化版本的 Qwen3-30B-A3B** 在 Aider 基准测试中仍能表现出色。
- **ktransformers 优化了 MoE 模型的 VRAM 使用**：[ktransformers 库](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)声称能以较低的 VRAM 需求高效运行 **Mixture of Experts (MoE)** 模型。
   - 他们声称仅需 **8 GB VRAM** 即可达到不错的速度，这比一次性将所有参数加载到 VRAM 中更适合 **30B-A3B** 模型。
- **Deepseek R2 热度攀升**：传闻即将发布的 **Deepseek R2** 将具有增强的人类视觉能力和自学习功能，可能在*明天*发布。
   - 一些成员正焦急等待，因为他们*相信* **Deepseek R2** 定于明天发布。
- **新 PR 为 Aider 添加了思考加载动画**：一位新贡献者提交了一个 [PR](https://github.com/Aider-AI/aider/pull/3911)，添加了一个 *🔃 Thinking* 加载动画，Aider 在等待 LLM 输出时会显示该动画。
   - 贡献者解释说，这让 Aider 感觉更*灵敏且有活力*。
- **Qwen3 的 Tool Use 表现出色，但在 Aider 中的应用尚不确定**：一些成员报告说 **Qwen3** 的 tool use 能力非常强，但由于 tool call API 的原因，其在 Aider 中的应用尚不确定。
   - 虽然 tool use 可能无法直接应用，但其他人建议使用**多 Agent 工作流**，其中 tool use 微 Agent 使用 Qwen3。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1366664882425958461)** (21 条消息🔥): 

> `AiderDesk Agent 模式，Repo Map 控制，OpenRouter 模型支持，Gemini 2.5 + Deepseek 组合` 


- **AiderDesk 的 Agent 模式表现出色**：根据其 [GitHub](https://github.com/hotovo/aider-desk)，一位用户在 **AiderDesk** 中使用 Agent 模式，配合 "probe" 进行规划，然后在准备就绪时启用 "Use Aider tools"、"Include context files" 和 "Include repomap"。
   - 他们还使用 **Jira** 管理和 **desktop-commander** 等其他工具来运行命令，但目前还没怎么使用 **memory-bank** 或 **context7**。
- **使用 Aider 调整 Repo Map**：一位用户希望在 **repo map** 中仅包含 API 代码，而不包含注释或测试，并询问是否可以使用 `aider --map-tokens 0` 禁用后两者。
   - 另一位用户建议使用 `repomix --compress` 或 `probe` 作为替代方案，并指出目前没有对 repo map 进行细粒度控制的原生支持。
- **支持 OpenRouter 模型，但并非总是成功**：一位用户询问 **Aider** 是否可以使用 **OpenRouter** 上的任何模型，另一位用户确认支持所有 **OR** 模型。
   - 他们还补充说，如果你使用 `gemma 3 1b` 或 `smollm`，不要抱太大期望。
- **Gemini 2.5 + Deepseek 强力组合**：一位用户发现了一个很好的组合：使用 **Gemini 2.5** 进行规划，使用 **Deepseek** 进行 diffs 和版本变更解释。
   - 他们建议在 **AI Studio** 中进行，因为 Gemini 在那里是免费的。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 条消息): 

p0lyg0n: 关于 Deepseek 的精彩纪录片: https://www.youtube.com/watch?v=Lo0FDmSbTp4
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1366681332901613568)** (10 条消息🔥): 

> `Apple Silicon, Cloud GPU, CUDA, Metal, ROCm` 


- **Apple Silicon 不是参加云端挑战赛的障碍**：一位拥有 **M4 Max PC** 的用户表达了对参加挑战赛的担忧，但另一位用户澄清说挑战赛在**云端**运行，因此 **Apple Silicon** 不是障碍。
   - 他们建议查看相关频道以获取更多信息。
- **Cloud GPU 支持远程 CUDA/ROCm 学习**：一位用户解释说，虽然使用本地计算资源学习 **CUDA** 或 **ROCm** 更容易，但使用 **Cloud GPU** 仍然可行。
   - 他们注意到现在廉价 Cloud GPU 的可用性越来越高。
- **在 Mac 上进行 Metal 编程是可行的**：一位用户肯定了在 Mac 上使用 **Metal** 编写 GPU 程序完全没问题。
   - 他们补充说，这更多在于是否熟悉工具，并分享了一个 **Metal** 代码片段。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1366837666850541688)** (2 messages): 

> `fp8 quantization, fp32 accumulation, Triton matmul, Custom CUDA kernels, AMD` 


- **关于 FP8 量化与 FP32 累加的疑问**：一名成员询问是否可以使用 **Triton** 进行 **matmul** 操作的 **fp8 quantization** 和 **fp32 accumulation**，或者是否必须使用自定义 **CUDA kernels**，特别是在 **AMD** GPU 上运行时。
- **通过 Num_stages 参数实现双缓冲 (Double Buffering)**：一位用户询问将 `num_stages` 设置为大于 1 是否本质上在 **Triton** 中启用了 **double buffering**。
   - 他们提到 **MI300** 不像 **Ampere** 那样具有异步加载 (async loads)，推荐的设置是 `num_stages=2`，并想知道 `num_stages > 2` 是否会有所帮助。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1366653061052891168)** (5 messages): 

> `Torch Logger Methods Compilation, AOT Inductor Multithreading` 


- **Torch Loggers 触发编译问题**：一位用户询问如何在编译期间忽略 **logger methods**，以避免与 **PyTorch distributed 模块**中的 `FSDP::pre_forward` 相关的异常。
   - 另一名成员建议将 `TORCH_LOGS` 环境变量设置为 `output_code` 或 `tlparse`，以检查生成的代码并识别导致问题的潜在 **if-statements**，并引用了 [`torch._dynamo.config.py` 中的特定行](https://github.com/pytorch/pytorch/blob/797768cd90d0984687e15f5fe0e1a4d8bf91d71a/torch/_dynamo/config.py#L506)。
- **C++ 中 AOT Inductor 训练故障**：一位用户报告使用 **AOT Inductor** 实现了部分 C++ 训练设置，但怀疑存在多线程问题。
   - 他们推论问题源于代码中不必要的特化 (specialization)，并计划开启一个 [PyTorch issue](https://github.com/pytorch/pytorch/issues) 以供 **AOTI 作者**进一步调查，特别是关注多个工作线程调用 `fw_graph->run()` 时 API 的行为。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1366888480302567565)** (1 messages): 

> `AMD MI300 competition, MoE kernels, FP8 submissions` 


- **针对 AMD MI300 竞赛发布新的单 GPU MoE Kernel**：一个新的单 **GPU MoE kernel** 题目现已在 **$100K AMD MI300 竞赛**中上线；请在 [leaderboard](https://www.gpumode.com/leaderboard/430) 上查看。
   - 一名成员建议，由于这个问题比较棘手，值得回顾一下提供的[详细解释](https://tinyurl.com/amd-comp-moe)。
- **AMD MI300 竞赛关键日期**：注册将于 **4 月 30 日**截止，而包括 **FP8** 和 **MoE kernels** 在内的最终提交截止日期为 **5 月 27 日**。
- **排行榜运行缓慢**：对于此问题，运行 `leaderboard submit ranked` 会很慢，耗时约 **8 分钟**。
   - 提交者建议使用 `leaderboard submit test/benchmark` 以进行更快的迭代。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

raymondz4gewu_60651: `/get-api-url`
  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1366769852425175102)** (22 条消息🔥): 

> `Quantized Models and torch.bfloat16, vllm Compile Integration Debugging, gemlite Kernel Selection, torch.compile Debugging Challenges, torch.dtype Extensibility` 


- **量化模型重新加载为 `torch.bfloat16`**：以量化布局保存后的量化模型在重新加载时会显示为 `torch.bfloat16`，因为原始的 `dtype` 被保留了。
   - 实际的量化 `dtype` 可以通过打印权重来查看，因为 PyTorch 的 `torch.dtype` 目前还不支持扩展到 Tensor 子类；更多讨论见 [此处](https://github.com/pytorch/ao/issues/442)。
- **`vllm` 编译集成难题**：在与 [gemlite 库](https://github.com/mobiusml/gemlite/) 集成时，`vllm` 的编译函数出现问题，使用 `torch.compile` 会导致错误行为。
   - 具体而言，`vllm` 无法根据输入形状从 `gemlite` 中选择正确的 Kernel；由于 `torch.compile` 的限制，在其内部进行调试非常具有挑战性。
- **`gemlite` 中的 Kernel 难题**：核心问题在于 `gemlite` 内部的 Kernel 选择错误，追溯原因是当 `vllm` 使用 `torch.compile` 时，输入形状未被正确识别。
   - Kernel 选择逻辑基于输入形状（定义在 [gemlite 的 core.py](https://github.com/mobiusml/gemlite/blob/master/gemlite/core.py#L386) 中），这使得形状检查对调试至关重要。
- **`torch.compile` 调试困境**：传统的调试方法（如 print 语句和断点）在 `torch.compile` 内部无效，这使得检查变量状态的过程变得复杂。
   - 使用 `TORCH_LOGS=+dynamo` 可以转储包含形状的图（Graph），从而辅助调试；[PyTorch 文档](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#breakpointing-dynamo-tracing) 提供了关于对 Dynamo 追踪设置断点的指南。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958)** (3 条消息): 

> `ROCm memory, CDNA3 ISA` 


- **ROCm 内存 Bank 大小澄清**：假设 32 位对齐，ROCm 中的内存 Bank 宽度为 **32 位**。
   - Bank 通过 `address % bank_size` 计算。
- **CDNA3 ISA 参考详情 LDS 配置**：根据 **CDNA3 ISA 参考手册** 第 2.2.1 节，每个计算单元（Compute Unit）拥有 **64kB** 的内存空间用于低延迟通信。
   - 该内存配置为 **32 个 Bank**，每个 Bank 包含 **512 个条目**，每个条目为 **4 字节**。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1366802069096239276)** (3 条消息): 

> `QR decomposition, SIMD, Thread barriers, Single-threaded SVD` 


- **128 位 QR 分解令人惊叹**：一位成员在[链接的 Python 脚本](https://cdn.discordapp.com/attachments/1285384841730457600/1366802750817697853/ember_ml_svd_128bit.py?ex=681245c1&is=6810f441&hm=657c03f2fc77e181231bcfd8c0dbe87a034b5f0bd2c941fa48ecea7088a71f1f&)中分享了一个非常出色的 QR 分解实现，该实现使用 **SIMD** 和 **线程屏障 (Thread barriers)** 达到了 **128 位精度**。
- **加速单线程 SVD**：一位成员报告称在 **SVD** 中发现了单线程模式，并指出他们正在修复该问题以使其更加并行化。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1366723501729382441)** (3 messages): 

> `GPU Price Tracker, AI/ML Engineer for Hire, Open Source IDE for AI/ML` 


- **在 Amazon 上追踪 GPU 价格**：一名成员构建了一个 [GPU 价格追踪器](https://www.unitedcompute.ai/gpu-price-tracker)，它可以提取 GPU 的完整 **Amazon 价格历史**并生成精美的图表。
   - 它会计算最新的数值，例如每美元可以获得多少 **teraflops**；一个使用场景是寻找购买私有集群的最佳时机。
- **AI/ML 工程师求职**：一位拥有 **8 年经验**，擅长人工智能、机器学习、全栈和移动端开发的 AI/ML 工程师正在求职；其专业领域涵盖深度学习、自然语言处理和计算机视觉，能够将尖端的 AI 解决方案集成到可扩展且健壮的应用中。
   - 提供了其 [LinkedIn 个人资料](http://www.linkedin.com/in/lucy-hunter-40a527350)和[作品集](https://lucyhunter.vercel.app/)的链接，以及技能列表，包括 **ML 算法、Deep Learning、NLP、Computer Vision、MLOps 和 AI Model Integration**。
- **开源 IDE 项目启动**：一名成员正在为 AI/ML 工程师构建一个开源 IDE，并正在寻找合作者；如果你对细节感兴趣、想加入或有见解，请私信他们。
   - 该成员提供了其 [LinkedIn 个人资料](https://www.linkedin.com/in/bruno-scaglione-4412a0165/)和 [GitHub 个人资料](https://github.com/BrunoScaglione)的链接。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1366881755864498197)** (1 messages): 

> `Use Cases, Performance` 


- **用户询问使用场景和性能**：用户正在询问具体的**使用场景**以及实现后的**性能指标**。
- **对实现细节表现出浓厚兴趣**：大家*非常想听听你使用它的进展如何*，特别是关于实际成果方面。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1366839192604119215)** (15 messages🔥): 

> `FP8 quantization material, FP8 matmul, Deepseek-v3 tech report, prefixsum ranked timeout` 


- ****开启 FP8 量化探索****：一名成员询问了关于 **FP8 量化**的资源，特别是关于 **FP8 matmul 配合 FP32 累加**的优势，并[链接了 onnx fp8 格式页面](https://onnx.ai/onnx/technical/float8.html)。
   - 他们引用了 **Deepseek-v3** 技术报告，指出 **FP8** 可能会面临**下溢问题**，因此需要更高精度的累加器。
- ****Prefixsum 排名超时排查****：一名成员报告了频繁的超时问题，特别是对于 **ranked prefixsum 提交**，尽管有 **30s 的超时限制**。
   - 工作人员承认了该问题，将其归因于他们自己的错误，并随后声称已解决，但该成员仍然遇到超时，随后通过私信发送了代码。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1366626400379076649)** (60 messages🔥🔥): 

> `vectoradd benchmark on H100, amd-fp8-mm benchmark on MI300, amd-mixture-of-experts benchmark on MI300, prefixsum benchmark on H100, A100, matmul benchmark on L4` 


- **H100 VectorAdd 速度竞逐！**：多个提交进入了 **H100** 上的 `vectoradd` 排行榜，时间从 **540 µs** 到 **708 µs** 不等，其中一个提交以 **540 µs** 获得第三名。
- **MI300 AMD-FP8-MM 排行榜升温！**：大量提交冲上 **MI300** 上的 `amd-fp8-mm` 排行榜，包括一个 **196 µs** 的第三名，个人最佳成绩在 **2.37-2.43 ms** 左右，成功运行的时间跨度很大，从 **198 µs** 到 **8.05 ms** 不等。
- **AMD Mixture of Experts 夺得榜首！**：**MI300** 上的 `amd-mixture-of-experts` 基准测试出现了一个 **6228 ms** 的第一名提交，以及多个在 **7379-7490 ms** 左右的第二名提交。
- **Prefixsum 在 H100 和 A100 上并驾齐驱！**：`prefixsum` 排行榜出现了多个第二名提交：一个在 **A100** 上为 **1428 µs**，另有几个在 **H100** 上约为 **955-985 µs**。
- **L4 MatMul 桂冠虚位以待！**：**L4** 上的 `matmul` 排行榜创下了 **2.27 ms** 的新第一名，而另一个提交以 **49.3 ms** 获得第二名。


  

---

### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1366684786206441514)** (2 messages): 

> `Single GPU MoE Kernel, FP8 and MoE Kernels, Leaderboard Submissions` 


- **单 GPU MoE Kernel 问题已上线！**：新的单 GPU MoE Kernel 问题现已发布，请查看 [排行榜](https://www.gpumode.com/leaderboard/430)。
   - 官方提供了一份详细说明，建议通过 [此链接](https://tinyurl.com/amd-comp-moe) 仔细阅读。
- **重要日期提醒**：注册将于明天 **4 月 30 日** 截止，**FP8** 和 **MoE Kernels** 的提交截止日期均为 **5 月 27 日**。
   - 请注意，针对此问题运行 `leaderboard submit ranked` 会比较慢（约 **8 分钟**），因此请使用 `leaderboard submit test/benchmark` 进行快速迭代。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074)** (23 messages🔥): 

> `Aithe 参考代码, FP8 正确性验证, Submission ID, 此 Kernel 的官方问题说明` 


- **Aithe 参考代码**：一位成员询问 **Aithe 参考代码** 是否会开源，并对 **FP8** 能否通过逐元素完全相等检查（element-wise perfect equal checks）的正确性验证表示怀疑；[参考代码](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems) 随后被迅速提供。
   - 回复澄清了对比并非逐元素完全相等检查，并指出了 [相关函数](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems/amd/utils.py#L31) 使用的是 `rtol=2e-02, atol=1e-03`。
- **找回丢失的排名代码**：一位在本地丢失了排名提交代码的成员寻求帮助，另一位成员建议使用 `/leaderboard show-personal` 和 `/leaderboard get-submission` 来找回。
   - 丢失的提交已通过其 ID (`11105`) 确认，该成员被引导使用 `/get-submission` 命令。
- **第二个问题推迟**：成员们讨论了即将发布的第二个问题，确认在完成额外测试后将很快发布，FP8 通道不会关闭。
   - 分享了 [此 Kernel 的官方问题说明](https://tinyurl.com/amd-comp-moe) 的链接。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

vkaul11: 是否有现成的 Kernel 可以执行 FP8 乘法并进行 FP32 累加？
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1366641263343964170)** (88 messages🔥🔥): 

> `ChatGPT 持久化记忆, AI Agent 公司, IAM360 框架, AI 生成的缩略图` 


- **ChatGPT 获得基础持久化记忆**：ChatGPT 开发了 **两种类型的持久化记忆**：一种是从它认为重要的对话细节中提取的长期记忆（训练数据），另一种是参考过去 **90 天** 上下文的短期记忆。
   - 用户可以关闭长期或短期记忆，但一个开关不能同时控制两者。
- **AI Agent 公司的结果混乱得可笑**：教授们尝试让一个虚构公司完全由 AI Agent 运营，但 [*结果混乱得可笑*](https://futurism.com/professors-company-ai-agents)，这表明目前的 AI 模型无法完全取代人类工作。
   - 尽管科技巨头宣称如此，但 AI 模型尚未达到完全取代人类所需的水平，仍需要人类监督。
- **IAM360：一种模块化符号化 GPT-Agent 架构**：一位成员正在开发 **IAM360**，这是一个用于人机共生的实验性框架，使用标准 ChatGPT 会话构建，无需自定义 GPTs、微调或 API 集成。
   - 该系统使用具有持久角色（战略、执行、财务、情感）的 **模块化符号化 GPT Agent**，以及一个用于自然涌现对话的 **零样本编排系统**。
- **以 Robux 出售 AI 制作的缩略图**：一位成员报告称以 **1500 Robux** 的价格售出了一张 AI 制作的缩略图。
   - 其他成员表示，如果提供任何复杂的参考图像，目前的生成器都会把图像搞砸，现实世界中的客户不会为此买单。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `ChatGPT 中的身份系统, RP 中的动态 Game Master 角色, ChatGPT 内部工具, Prompt Engineering 技巧, LLM TTRPG 游戏开发` 


- ****记忆至关重要**：ChatGPT 中的身份系统**：一位成员讨论了为 ChatGPT 创建身份系统，以便[按身份分离记忆/历史对话](https://discord.com/channels/974519864045756446/1171946823369711676)，从而保留静态身份和状态。
   - 目标是避免用户陷入叙事低谷（narrative valleys），即要么抹除记忆，要么试图逃离此类场景。
- ****Game Master 动态**：角色扮演冒险**：一位成员分享了一个 Prompt，让 ChatGPT 在[奇幻角色扮演冒险中担任动态 Game Master](https://discord.com/channels/974519864045756446/1171946823369711676)。
   - 重点在于扮演非用户角色，根据主角的经历演化世界，并在世界观构建、角色对话和行动之间保持平衡。
- ****Bio 工具揭秘**：ChatGPT 的记忆**：一位成员透露 ChatGPT 的内部记忆被引用为 `bio` 工具，[建议调用其规范名称来定义保存命令](https://discord.com/channels/974519864045756446/1171946823369711676)。
   - 建议了一个改进版的 `/pin` 命令：*AI 使用 `bio` 工具将最近的消息保存到 ChatGPT 的内部记忆中，保留所有关键细节以供未来参考。*
- ****完美 Prompt**：GPT 的内部工具**：一位成员建议[要求模型识别并描述其每个连接工具的功能](https://discord.com/channels/974519864045756446/1171946823369711676)，列出它们的规范名称，并为每个工具提供一个展示其正确语法的代码块。
   - 提到的工具包括 **python, web, image_gen, guardian_tool, 和 canmore**。
- ****RPG 根源**：通用 AI 框架开发**：成员们注意到他们从 LLM TTRPG 游戏开发转向了[通用 AI 框架开发](https://discord.com/channels/974519864045756446/1171946823369711676)的历程。
   - 一位成员强调，这条路径可以通向学术研究。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `ChatGPT 中的身份系统, RP Prompt 问题, 动态 Game Master 角色, ChatGPT 内部记忆 (bio 工具), LLM TTRPG 游戏开发` 


- ****人格持久性困扰玩家****：用户正面临 **ChatGPT** 在角色扮演场景中抹除记忆或陷入“叙事低谷”的问题，这阻碍了静态身份和一致角色状态的创建。
   - 无法维持持久身份迫使用户不断重置或规避不理想的叙事路径。
- ****Game Master (GM) 角色定义****：一位成员为 **ChatGPT** 在奇幻角色扮演中定义了动态 **Game Master (GM)** 角色，重点是扮演与用户主角互动的非玩家角色 (NPC)，并根据主角的经历演化世界。
   - GM 应平衡世界观构建、对话和行动，避免过度细节，并使用特定命令如 `/export_character`、`/export_world_state`、`/force_random_encounter` 和 `/set_mood` 来管理游戏。
- ****精准定位 ChatGPT 的 Bio 工具****：该成员确认 **ChatGPT** 的内部记忆为 `bio` 工具，建议他人在保存命令中使用此规范名称，以确保 pin 功能通过 `/pin` 正确保存关键细节供未来参考。
   - 他们建议将命令放置在 Prompt 顶部附近，并使用间隔重复（gapped repetition）来提高指令遵循度。
- ****源自奇幻的框架****：一位成员分享了他们的 AI 之旅始于 **LLM TTRPG 游戏开发**，随后转向通用 AI 框架开发，最后进入学术研究。
   - 他们目前正致力于为特定任务创建一个 **GPT**，以便更好地将 LLM 整合进一个完整的框架大纲中。
- ****驯服文本生成技术的技巧****：一位成员建议在 Prompt 中加入具体的规范，以减少 **LLM** 的猜测，并要求模型识别和描述其连接的工具，列出它们的规范名称并演示正确的语法。
   - 他们提供了如何查询模型工具（如 **python**、**web**、**image_gen**、**guardian_tool** 和 **canmore**）的示例，并给出了调用它们的特定语法。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1366662361729208400)** (79 条消息🔥🔥): 

> `PyQt5 Chat App, OR vs ML history, Gemini 2.5 Pro vs GPT-4o, Qwen 3 performance, FFN in Transformers` 


- ****PyQt5** 聊天应用引发关注**: 一位成员分享了一个使用 **PyQt5** 构建的 [AI 聊天应用](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd&)，并使用 **LM Studio** 作为后端服务器。
   - 要使用该应用，用户在运行程序前必须先在 **LM Studio** 上选择并启动模型作为服务器。
- **辩论中厘清 **ML** 的 **OR** 起源**: 讨论围绕 **Operations Research (OR)** 与 **Machine Learning (ML)** 之间的历史关系展开，一位成员指出 *ML 起源于统计学*。
   - 另一位成员反驳称，早期的 **AI/ML** 与 **Operations Research** 和 **Control Theory** 非常接近，但后来分支出来并转向统计方法，特别强调 *从数据中学习，而非从第一性原理对现实建模*，现代 ML 具有极强的经验主义色彩。
- ****Gemini 2.5 Pro** 在对比 **GPT-4o** 时遭到吐槽**: 成员们讨论了 **Gemini 2.5 Pro** 与 **GPT-4o** 的性能对比，一位用户称 Gemini 为 *4otard*。
   - 另一位表示，*Gemini 2.5 Pro 肯定比 4o 差*，认为它可能在编程方面更好，但在通用场景下表现不佳；其他用户也发现 **GPT-4o-mini** 在聊天体验上比 **Gemini 2.5 Flash** 更好。
- ****Qwen 3**: 新模型凭借推理能力令用户兴奋**: 成员们赞扬了新的 **Qwen** 模型，特别提到了其改进的推理和指令遵循能力。
   - 一位用户报告称，*它在某些推理任务中的输出*更为出色，理由是其客观性强且严格遵循指令，尤其称赞了 MoE 模型的速度和智能，形容它 *即使不比 2.5 Flash 更聪明，也至少旗鼓相当*。
- ****FFN** 功能引发困惑与审视**: 讨论涉及了 **Feed-Forward Networks (FFN)** 在 Transformer 架构中的作用，一位用户试图寻求对其功能的直观理解。
   - 一些人认为 **FFN** 实现了通道/神经元级别的信息混合，增加了容量和非线性，一位成员引用道：*拥有 FFN 本身比它的宽度要重要得多*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1366800354993246350)** (8 条消息🔥): 

> `DeepSeek VL, Construction` 


- **施工导致 DeepSeek VL 讨论取消**: 成员家附近的施工导致会议取消。
   - 讨论 **DeepSeek VL** 的会议将移至明天。
- ****DeepSeek VL 讨论将重新开始****: 之前的 **DeepSeek VL** 讨论仅涵盖了引言部分，因此成员们将从头开始重新进行论文讨论。
   - 团队计划戴上降噪耳机重新开始。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500)** (34 messages🔥): 

> `Reddit 上的匿名 LLM、ChatGPT 的说服技巧、Meta 的 LlamaCon 2025、Llama 4（又名 Little Llama）、SAM 3 开发` 


- **匿名 LLM 愚弄了 Reddit 的 change-my-view 版块**：研究人员在 Reddit 的 **/r/changemyview** 上测试了一个匿名 LLM，发现其*效力极高*，引发了用户的反感，详见[此 X 帖子](https://x.com/emollick/status/1916905103358931084)和 [Reddit 线程](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY)。
   - 一位用户幽默地表示：*AI 并不聪明，改变我的想法试试看*，对此 **ChatGPT** 回答：*是的，它们很聪明*，该用户随后回复：*好吧，我道歉*。
- **ChatGPT 擅长哲学对话**：一位成员发现，让 **ChatGPT** 反驳自己的信仰或为他们感到厌烦的事实辩护，既*有趣又具有教育意义*。
   - 他们指出，虽然 **O1-preview** 在*日常对话中显得枯燥*，但 **O3/O4-mini-high** 模型非常适合一般话题，他们现在使用 **o4-mini-high** 进行新闻分析。
- **Meta 举办 LlamaCon 2025**：**Meta** 举办了 **LlamaCon 2025** 生成式 AI 开发者大会，可通过 [Engadget](https://www.engadget.com/ai/llamacon-2025-live-updates-from-metas-first-generative-ai-developer-conference-keynote-215241436.html) 和[官方直播](https://www.facebook.com/MetaforDevelopers/videos/1792349135036347/)获取实时更新。
- **Llama 4（又名 Little Llama）确认发布**：在 **LlamaCon** 上确认了 **Llama 4**（也被称为 *Little Llama*）的存在，见[此 YouTube 直播](https://www.youtube.com/live/6mRP-lQs0fw)。
   - 一位用户开玩笑地称它们为 *Baby llama's*，而另一位用户则表示失望，认为这些公告*空洞无物*。
- **SAM 3 正在开发中**：**LlamaCon** 的一个关键公告是 **SAM 3** 的开发以及 **Meta** 的新应用。
   - 一位用户思考 **Little Llama** 模型将如何与 **Qwen** 模型竞争。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766)** (1 messages): 

> `Atropos RL 框架、RLAIF 模型、GRPO 工具调用、企业基本面预测、Psyche 去中心化训练网络` 


- **Atropos 框架打破 RL 障碍**：Nous Research 发布了 **Atropos**，这是一个针对基础模型的 [强化学习（Reinforcement Learning）rollout 框架](https://github.com/NousResearch/Atropos)，支持复杂环境以提升模型能力。
   - Atropos 是其整体 RL 系统设计的一部分，很快将由训练和推理组件进行补充，详见其[介绍博客文章](https://nousresearch.com/introducing-atropos/)。
- **GRPO 工具调用提升了 DeepHermes**：他们在带有 **GRPO** 的环境中将 **DeepHermes** 的工具调用能力在简单和并行工具调用上分别提升了 **2.4 倍**和 **5 倍**（使用 Berkeley 的 Function Calling 基准测试）。
   - 使用 Atropos 环境创建的产物已发布在 [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6) 上，包括一个新数据集和五个新模型，涵盖工具调用、企业基本面预测以及使用 RLAIF 构建的实验性人格模型。
- **基本面预测模型准确率翻倍**：使用 Atropos 后，企业基本面预测模型在方向性变化预测上的准确率从 **~25%** 提高到 **50%**。
   - Atropos 框架旨在通过强化学习引导语言模型发挥其最佳潜力，就像希腊命运女神引导灵魂走向最终归宿一样。
- **Psyche 网络实现去中心化训练**：Atropos 是 **Psyche** 的关键组件，Psyche 是一个即将推出的去中心化训练网络，负责在全球范围内协调预训练（pre-training）、中段训练（mid-training）和后训练（post-training）的工作负载。
   - 将于 5 月 18 日在旧金山举办黑客松，以促进协作进展（更多细节即将公布）。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833)** (110 条消息🔥🔥): 

> `Qwen 3 Overfitting, DeepSeek R2 Release, Huawei Ascend 910B, Atropos Release, Minos Model Refusals` 


- ****Qwen 3 的 Base 模型在评估集上过拟合****：成员们发现 **Qwen 3 的 Base 模型** 似乎在某些评估集（evals）上严重过拟合。据报告，该模型在 **M24b** 上的 **Trivaqa** 得分为 **75%**，但在 **Q30MoE** 上仅为 **60%**。
   - 一位成员指出，他们在 **30B-A3** 和 **32B-dense** 之间的 Benchmark 结果确实非常接近，这可能是由于某些过拟合导致的，这引发了关于 MoE 有效性的讨论。
- ****DeepSeek R2 发布传闻四起****：有传言称 **DeepSeek R2** 可能很快发布，一些报告声称它完全是在 **Huawei Ascend 910B** 硬件上训练的，这可能会减少对 **Nvidia CUDA** 的依赖。
   - 然而，其他人反驳了这些说法，并引用了一条 [推文](https://fxtwitter.com/teortaxesTex/status/1916325875437445243)，表示 **DeepSeek** 的官方立场是：*“我们会在发布 R2 的时候发布它，任何声称自己知道内情的人都在撒谎”*。
- ****Nous Research 发布 Atropos****：[Nous Research 发布了 Atropos](https://github.com/NousResearch/atropos)，这是一个开源项目和推理优化技术。
   - 为使用 **Atropos** 的开发者创建了一个新频道 <#1365222663324307466>。
- ****Minos 模型与能力相关的拒绝回答****：一位在使用 **Minos** 的成员想知道是否应该有一种方法将与能力相关的拒绝回答（refusals）与其他类型的拒绝区分开来，并担心这可能会增加幻觉（hallucinations），因为模型可能会认为自己具备其实并不拥有的能力。
   - 讨论中对模型“不能”执行任务与“不愿”执行任务进行了区分。
- ****Physical AI 跑马拉松****：有人分享了一张 [Physical A.I. 机器人](https://cdn.discordapp.com/attachments/1149866623109439599/1366647197789323274/NoJoke.png?ex=68125da3&is=68110c23&hm=beab804046b63afebd36468c0257ad616184ba8bf7aed8feb39bac3da164077e) 的图片，它在上周的上海马拉松比赛中跑得比大多数人都好。
   - 评论者指出 *“AI 现在真的在围着我们跑（意指超越人类）”*，并附上了 [Prime Intellect 的 X 帖子](https://x.com/PrimeIntellect/status/1916994185573634336) 链接。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1366647358842208287)** (2 条消息): 

> `Image loading issues` 


- **图片加载问题困扰用户**：一位成员报告图片一直处于加载状态，表明 **图片上传或加载时间可能存在问题**。
   - 该用户随后回复称已恢复正常（Working）。
- **用户确认图片加载已解决**：一位成员确认图片加载问题已得到解决。
   - 该成员简单地表示 *Working*。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1366633673096433715)** (101 messages🔥🔥): 

> `VS Code Extension for Filtering .cs files in Git Changes View, Cursor Spending Limit Issues, Model Selection Purpose, Anthropic 3.7 Incident, Gemini 2.5 Pro Issues` 


- **支出限制导致请求变慢**：一位用户报告称，在达到支出限制并升级后，数小时内仍受困于 **slow requests**，另一位用户则耗尽了 **fast requests**。
   - 另一位用户补充道，即使在 **slow requests** 模式下，**Gemini** 的速度依然很快。
- **Cursor 社区 Discord：终于重新受到关注了吗？**：一位成员幽默地注意到 **Cursor 的 Discord** *终于再次得到了关爱*。
   - 另一位成员自信地回应道 *Cursor 一直深受喜爱*，暗示团队只是在精益求精。
- **Gemini 故障：模型在请求中途停止！**：用户报告 **Gemini 2.5** 经常在请求中途停止，尽管它表示将执行操作；另一位用户建议 *当某个模型表现异常时，尝试使用不同的模型*。
   - 一名团队成员确认团队一直在与 Google 合作解决此问题，并建议用户在此期间使用其他模型，并提议用户将他们的 **request ID** 发送给团队以便调查。
- **Agent 反应迟钝：多次尝试后编辑依然无果！**：一位用户报告称 **Agent 无法进行编辑的问题严重**，在多次尝试后，它转而指示用户手动操作。
   - 一名团队成员建议该问题可能是由 **Gemini 2.5 Pro** 引起的，并建议创建新聊天以刷新上下文；他们建议使用 4.1 GPT 或 3.5 处理代码，如果出现任何问题则使用 3.7 Claude。
- **官方 Ollama 智能手机 App 何时推出？**：一位用户询问了官方 **Ollama 智能手机 App** 的发布时间表，并链接到了相关的 [X 帖子](https://x.com/awnihannun/status/1917258279455187034)。
   - 一位用户提到他们通过重新安装 Cursor 并清除缓存解决了问题，另一位用户确认可以手动清除缓存，从而避免重新安装过程。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1366638637948866570)** (43 messages🔥): 

> `Cloudflare Turnstile, whisper-large-v3-turbo issues, GGUF models and CPU offloading, Model Context Protocol (MCP), Fastest inference for running models` 


- **成员测试 Cloudflare Turnstile**：成员们测试了 [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs) 是否有效，并得到了肯定的确认。
   - 确认后，该成员兴奋地喊道 *YIPEEEEEEEE*。
- **成员报告 Whisper Turbo 问题**：成员们报告 **OpenAI 的 whisper-large-v3-turbo** 在 HF Inference 端点上无法工作，甚至网页上的 Demo 也挂了。
   - 成员们链接了类似的问题（如[这一个](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691)）作为潜在帮助。
- **合并时进行 CPU RAM Offloading 是可行的**：成员们讨论了在将 checkpoint 合并到基础模型时 offloading 到 CPU RAM 的问题。
   - 一位成员表示这没问题，并指出 *Transformers + Accelerate 或 Llama.cpp* 支持 offloading，而且 **GGUF 格式本身就假定支持 CPU offloading**。
- **不同模型的推理速度比较**：成员们思考了 **Model Context Protocol (MCP)** 以及运行模型时哪种 Inference 方式最快。
   - 有人指出 **Unsloth** 比 Hugging Face 更快，其他人则推荐使用 **sglang/lmdeploy** 或 **exllamav2**。
- **寻找活跃的 AI Hackathons 和训练营**：一位成员询问是否有活跃的 **AI 相关训练营或 Hackathons**，并提供参与奖励。
   - 在后续讨论中没有提供具体的推荐。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@1298649243719958612> 请不要跨频道发帖
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1366761093053022209)** (9 messages🔥): 

> `3D Animation Arena，LLM-as-a-Judge 的替代方案 Pi-Scorer，HMR 模型` 


- ****3D Animation Arena 开启 HMR 模型排名****：一名成员在 Hugging Face 上创建了 [3D Animation Arena](https://huggingface.co/spaces/3D-animation-arena/3D_Animation_Arena)，根据不同标准对模型进行排名，旨在为当前的 **HMR (human mesh recovery，人体网格恢复) 模型**建立排行榜。
   - 创建者正在寻求投票以填充排行榜数据。
- ****Pi-Scorer 成为 LLM-as-a-Judge 的替代方案****：一名成员分享了 **Pi-Scorer**，这是 **LLM-as-a-Judge** 的替代方案，并提供了 Colab 笔记本，展示如何将 **Pi-Scores** 用于[模型检查点评估 (model checkpoint evaluation)](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) 以及作为 [奖励函数 (reward functions)](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb)。
- ****分享 AI Assistant 集成代码****：一名成员分享了其 **AI assistant integration** 项目的[代码](https://github.com/BouajilaHamza/site-ai-assistant-integration)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1366836869610078282)** (2 messages): 

> `缺陷标注，图像遮罩，滤波器使用` 


- **攻克缺陷标注难题**：一名成员正尝试实现这篇 [论文](https://arxiv.org/pdf/2009.07047v1)，但在生成和标注陈旧划痕图像方面面临挑战。
   - 该成员合成生成了带有划痕、模糊和灰度等缺陷的图像，目前正在寻求关于如何标注这些缺陷的建议。
- **遮罩（Masking）方法亮相**：一名成员建议对图像进行遮罩处理，在测试不同阈值以分离划痕的同时将其二值化，并保持图像其余部分不变。
   - 该成员指出如何通过测试不同阈值来找到理想的平衡点。
- **针对瑕疵进行滤波**：一名成员建议使用 **Canny edge** 或 **Sobel** 等滤波器，通过特定阈值来分离缺陷。
   - 这些滤波器在特定阈值下可以很好地分离缺陷，从而使数据集上的划痕自动标注变得更加容易。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1366625364310228993)** (40 messages🔥): 

> `Hugging Face Agents 认证，Agents.json 对比 Prompts.yaml，Llama-3 访问请求，模型暂时不可用，利用免费资源完成最终项目` 


- **庆祝 HF Agents 课程结业！**：成员们庆祝完成 **Hugging Face Agents** 课程并获得认证，一名成员分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/suhail-ahmed-9b4312b/)。
   - 另一名成员在完成课程后也分享了他们的 [LinkedIn 个人资料](https://www.linkedin.com/in/roshankv/)。
- **通过调整时间解决超时问题！**：一名用户报告称，通过将 `requests.get` 函数中的超时值增加到 **20 秒**，解决了超时问题。
   - 另一名用户确认该改动解决了他们的问题。
- **关于 Agents.json 和 Prompts.yaml 的思考**：一名课程参与者要求澄清在 Unit 1 的 smolagents 部分中，**agents.json** 和 **prompts.yaml** 文件之间的区别。
   - 该用户还寻求关于*使用 Agent 的 tools 参数向工具列表添加新工具*的指导。
- **Llama-3 访问请求被拒绝！？**：一名用户报告称他们访问 **meta-llama/Llama-3.2-3B-Instruct** 的请求被拒绝，并询问原因。
   - 其他成员建议通常需要 Llama 的访问权限，并引导该用户在[此处](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)请求访问。
- **“暂时不可用”的烦恼**：一名用户报告称，他们尝试使用的所有模型都显示为*暂时不可用 (temporarily unavailable)*。
   - 另一名用户建议使用 **Apple 的 MLX 框架**在本地设置笔记本，作为一种可能的变通方案。


  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1366809269244264469)** (1 条消息): 

> `Audio Overviews, 多语言支持` 


- **Audio Overviews 正式上线！**：Audio Overviews 正在进行 Beta 版滚动更新，用户现已可以使用 **50 多种语言**进行创建。
   - 立即尝试使用您偏好的语言，并通过这篇 [博客文章](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/) 分享反馈。
- **多语言能力现已可用**：Audio Overviews 现在支持 **50 多种语言**，为更多元的用户群体提供支持！
   - 请查看 [博客文章](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/) 了解更多详情。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1366658539661299712)** (28 条消息🔥): 

> `NotebookLM 语言支持, Audio Overview 限制, 简洁的解释, 更智能的模型` 


- **NotebookLM 的全球“绕口令”：现在会说多种语言**：NotebookLM 现在可以指定对话语言，这是一项新功能，而且 [Google 的 NotebookLM 现在支持 50 种语言](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/)。
   - 用户测试了 **冰岛语** 和 **马拉地语** 的 Audio Overviews，其中一位用户对马拉地语的流利和地道感到惊讶，称其 *“没有那种外国人口音之类的”*。 
- **Audio Overview 定制上限引发讨论**：一位用户注意到定制音频的更新限制在 **500 个字符** 以内，并好奇这与将指令作为单独的文本文件上传是否有区别。
   - 该用户希望 *“减少愚蠢的闲聊，保持对事实和时间线的关注”*。
- **用户发现非英语语言的 Audio Overviews 更加简洁**：用户发现为非英语语言生成的 **Audio Overviews** 持续时间更短。
   - 一位在小型文档上进行测试的用户表示，*“它的解释非常简洁”*。
- **更智能的模型驱动更好的 Explanations**：Google 已确认新的非英语 **Audio Overviews** 表现更好，因为 *“我们在底层使用了更智能的模型！”*
   - NotebookLM 持续在底层改进其摘要能力。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682)** (65 条消息🔥🔥): 

> `NotebookLM 更新, 多语言支持, Audio Overview 问题, 交互模式 Bug, 播客功能请求` 


- **NotebookLM 荣获 Webby 奖！**：**NotebookLM** 在 [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm) 中表现出色，获得了 **技术成就奖 (Technical Achievement)**。
- **多语言支持到来，但并非人人适用！**：成员们庆祝 **NotebookLM** **多语言支持**的到来，但一位成员注意到 **越南语音频** 无法工作，且 UI 仍显示 *“仅限英语”*。
   - 一位成员确认更新仍在滚动发布中，并建议用户等待几个小时；另一位成员补充道，让大家 *“准备好应对每天十次‘即使发布了我也没看到新功能，该怎么办’的询问”*。
- **非英语 Audio Overviews 受到时长限制！**：一位用户报告称，**英语 Audio Overview** 有 **15 分钟限制**，而 **土耳其语** 的限制为 **6 分钟 20 秒**。
   - 一位成员表示，由于 *“技术原因”*，非英语音频目前受到限制，但团队正在努力延长时长。
- **交互模式麦克风问题困扰用户！**：一位用户报告称，**交互模式** 无法从其麦克风采集任何音频。
   - 另一位成员建议检查 **麦克风权限** 和 **浏览器设置**，并尝试使用 [麦克风测试](https://mictests.com/) 或更换浏览器。
- **Notebook 共享难题与解决方案！**：一位成员报告称，他们共享 **Notebook** 的对象收到了 *“没有访问权限”* 的消息。
   - 一位成员澄清说，用户需要在共享对话框中明确添加共享对象的电子邮件地址。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1366635231347736626)** (75 messages🔥🔥): 

> `Add on Credits, Manus Fellow Program, Manus Referral Program, Manus Credit System, Beta Testing` 


- **如果不重新订阅，Add on Credits 就毫无用处**：一位用户警告说，给予早期订阅者的 Add on Credits 除非重新订阅，否则毫无用处，因为它们会在短时间内过期。
   - 该用户声称他们没有被告知过期事宜，现在损失了 **3900** 积分。
- **关于双倍积分的问答**：一位用户提供了关于双倍积分的快速 FAQ，指出只要你的订阅处于激活状态，奖励积分就永不过期。
   - 他们补充说，邀请是随机的，似乎**并非每个邀请都能获得两个邀请名额**，这是随机的，因为他们可能刚刚收紧了限制。
- **用户寻求 Manus Fellow Program 的信息**：一位用户询问有关 Manus Fellow Program 的信息，例如 Manus 是否会主动联系所需的 Fellow 并雇佣他们？还询问了目标国家（美国、中国、新加坡、韩国、澳大利亚等），以及该计划是否不针对巴基斯坦、印度等国家。
   - 另一位用户回答说，Starter 方案提供 **2 个邀请名额**，Pro 方案提供 **5 个邀请名额**。
- **对积分系统和 Beta Testing 的批评**：一位用户表达了对积分系统和 Beta Testing 的看法，认为用积分限制用户破坏了 Beta 阶段的初衷。
   - 他们补充说，*真正的 Beta 测试应该让用户从头到尾完成完整的项目，从而提供关于体验的有意义反馈并提出改进建议*。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1366626785827094620)** (51 messages🔥): 

> `X-Ware Red, Llama Prompt Ops, LLM Benchmarks Survey` 


- ****X-Ware Red** 工具发布**：一位用户分享了一个名为 **X-Ware Red** 的工具，该工具使用 embed 的标题；它会在前面加上 `r.jina.ai/` 和 `openrouter-free-tier` 来为线程生成标题。
   - 一位用户建议将其设为开关，以选择线程标题是否应与 embed 的名称不同。
- ****Llama Prompt Ops** 推出**：**Meta** 推出了 [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops)（一个用于 Prompt Engineering 的开源工具）和 [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit)。
- **发现链接帖子会重命名线程的 Bug**：一位用户报告了一个 Bug，即在线程中发布链接会重命名已经命名的线程，尽管它*应该只寻找标题中带有 "https://" 的线程并进行更改*。
- **用户寻求持久的 **LLM Benchmarks****：一位用户询问是否有关于 **LLM Benchmarks** 的优秀调查，支持对模型进行历史对比。
   - 另一位用户回答说，*大多数 Benchmark 持续时间不到 2 年*，并建议参考 "AI Engineer Reading List" 获取当前的基准，并指向了一位用户关于 OSS leaderboard v1 和 v2 的帖子。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1366719239968526416)** (13 messages🔥): 

> `Bending Origins in Mojo, Origin-related headaches, Multiple Licenses in Modular Repository, Pointer usage to avoid origin issues` 


- **在 Mojo 中随心所欲地操控 Origins**：一位成员想做一个关于操控 **Origins** 的小练习，比如将 Origins 重新绑定到容器元素的 **Origin** 而不是容器本身的 Origin。
   - 另一位成员回应说，他们处理过很多与 Origin 相关的头疼问题，主要是由于 *API 缺口、可参数化的 Trait 以及其他缺失的语言特性*。
- **Origins 导致可变引用问题**：一位成员提到，*你不能对同一个 Origin 持有两个可变引用*，尽管可以将 Origin 转换为 **MutableAnyOrigin** 来规避这一点。
   - 另一位成员回应说，任何非数组或列表形状的数据结构都会遇到问题，这会将性能退化到 **C 语言性能**。
- **在处理指针时会绕过 Origins**：在讨论构建类列表类型 + 类 Span 类型，或阅读标准库中的 `sort` 实现代码时，一位成员指出 *其中大部分都是抛弃 Origins，直接进入指针时间（pointer time）*。
   - 另一位成员对指针类型（包括 unsafe）表示担忧，因为所有的可变-不可变（mut-immut）修复。
- **Modular 仓库拥有多个许可证**：由于某些部分使用 Modular 的 **Community License** 许可，而其他部分使用 **Apache 2**，**Modular 仓库**现在似乎需要包含多个许可证。
   - 具体来说，[`src/max`](https://github.com/modular/max/blob/main/src/max/serve/README.md) 中的一些内容使用了社区许可证。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1366728808526381056)** (11 messages🔥): 

> `importing Python packages, profiling blocks of code, SIMD width, vector strip-mining, flamegraph` 


- **标准 Python `import` 支持可能即将到来**：虽然 Mojo 尚未确认完全支持标准 Python `import` 语句，但据一位成员透露，这是一个“相当确定的也许”，暗示 `python.import_module` 可能不会永远是唯一的选择。
- **`llvm-mca` 浮出水面，用于分析特定代码块**：一位成员询问如何对特定代码块进行性能分析（profiling），并提到了 `gpu` 模块的一个私有部分（[链接](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo)），另一位成员建议使用 `llvm-mca`。
- **针对 SIMD 宽度的向量条带挖掘 (Vector Strip-Mining)**：当指定的 **SIMD width** 是硬件 SIMD 宽度的倍数时，建议将编译器处理该情况的方式命名为 *vector strip-mining*。
- **`Flamegraph` 辅助 Perf 输出可视化**：一位成员建议使用 [flamegraph](https://github.com/brendangregg/FlameGraph) 来可视化 `perf` 的输出，并指出可执行文件应在编译时包含 **debug info**。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1366854505320419379)** (2 messages): 

> `GPT-4o generates Tetris, PapersChat indexes papers` 


- **GPT-4o 一次性生成俄罗斯方块**：来自 KaranVaidya6 的视频展示了 **GPT-4o** 如何利用 **LlamaIndex** 和 **Composiohq** 一次性（one shot）生成 **Tetris**。
   - 视频中使用的代码已在 [GitHub](https://t.co/KJb7YRINWg) 上发布。
- **PapersChat 索引 ArXiv 和 PubMed 上的论文**：**PapersChat** 是一款 Agentic AI 应用，允许你与论文进行对话，并从 **ArXiv** 和 **PubMed** 获取信息。该应用由 **LlamaIndex**、**Qdrant** 和 **MistralAI** 驱动。
   - 它会索引你所有的论文，并提供一个精美的 Web UI 进行查询，详情点击[此处](https://t.co/lYwXh27F9x)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1366667182687125557)** (17 messages🔥): 

> `Azure OpenAI timeouts, MessageRole.FUNCTION vs MessageRole.TOOL, Function agent and context issues` 


- **Azure OpenAI 的间歇性超时困扰用户**：用户报告称，即使在相同的 Prompt、端点和网络条件下，**Azure OpenAI** 端点也会出现间歇性 **timeouts**，这可能暗示存在 **rate limits**、**firewall issues** 或 **context breaching**。
   - 一位用户指出，由于问题会持续数分钟，重试机制（retry mechanisms）无效，且更换网络也只能偶尔解决这种不一致性。
- **剖析 MessageRole：FUNCTION vs. TOOL**：**MessageRole.FUNCTION** 和 **MessageRole.TOOL** 之间的区别取决于所使用的具体 API。
   - 某些 API（如 **OpenAI**）使用 **tool messages**，而其他 API 则依赖 **function messages**。
- **Function Agent 上下文故障揭秘**：一位用户遇到了 **function agent** 在第二轮交互的流事件（stream event）处卡住的问题，并提供了示例代码。
   - 一位成员建议在 `stream_events()` 退出后使用 `await handler`，以确保前一次运行结束并接收到最终响应，这修复了该错误。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1366639037745467433)** (9 条消息🔥): 

> `RAG Chatbot 挑战, 多源 GraphRAG, 本地推理与小模型训练, AI 研究协作` 


- **RAG Chatbot 面临挑战**：一位成员在使用官方文档开发基于 **RAG** 的聊天机器人时遇到挑战，特别是当回答需要来自多个源和文档的文本块（chunks）时，目前使用的是 **vector search + BM25**。
   - 他们正在寻求关于如何为 **LLM Claude 3.5 Sonnet v1** 和 **Amazon Titan v1** embeddings 最好地将引用链接到文档内附录的建议。
- **探索多源 GraphRAG**：一位成员询问 **GraphRAG** 是否值得尝试用于汇总来自多个源的答案，并将其与需要特定领域预训练模型的 **insightRAG** 进行了比较。
   - 他们还询问了替代方案，并提到将参加 **NAACL**。
- **探索本地推理与小模型训练的新项目**：一位曾是 [Dataherald](https://github.com/Dataherald/dataherald) 联合创始人的成员正在探索一个围绕 **local inference** 和 **small model training** 的新项目。
   - 他表达了对协作和参与社区研究的兴趣。
- **机器人、自主系统与 AI：工作机会酝酿中**：一位从事 **Robotics, Autonomy, and AI** 工作的成员正专注于 **LLMs** 在加速软件工程方面的作用。
   - 他们询问是否可以在 Discord 中发布工作机会，以及这是否会被视为“广告”。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1366808622927056927)** (10 条消息🔥): 

> `递归符号提示词, LLM 诚实合规性, LLM 中的 HHH 目标` 


- **探索递归符号提示词 (Recursive Symbolic Prompts)**：一位成员正在探索 **recursive symbolic prompts** 在分类器压力下的表现，重点关注平滑或对齐约束如何影响**多轮幻觉漂移 (multi-turn hallucination drift)**。
   - 该成员特别感兴趣的是符号结构（如 **role-bound predicates** 或 **attention-synced markers**）如何在多轮输出中存续，以及尽管存在软对齐漂移或输出平滑，这种结构如何跨补全（completions）传递。
- **LLMs HHH 张力暴露**：一位成员分享了他们的研究，关于[在比较 HHH (Helpful, Honest, Harmless) 对齐目标时，如何定量评分 LLM 输出的表现](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d)。
   - 他们结合使用 **YAML** 和 **python/Gradio** 来审计用户会话，测量每个 **HHH** 变量之间的内部张力，这包括强制模型变得更加诚实并观察由此产生的张力。
- **前沿模型在诚实性方面挣扎**：同一位成员发现，某些前沿模型比其他模型更符合诚实性要求，而一些模型在提供大量 token 堆砌且含糊不清的回答时，会输出伪造的指标。
   - 他们指出，像 **ChatGPT 4o** 和 **4.5** 这样的模型在回答挑衅性查询时表现出很高的置信度，但实际上，它们在会话中充斥着含糊其辞的双关语；讽刺的是，**OpenAI** 是所有前沿模型中透明度最低的。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1366627484828958752)** (12 条消息🔥): 

> `Credential Passing, RAG type server for client file ingestion, Streamable HTTP Implementation and Authentication, Multi-Tenant Server Hosting, Open Source Models for Agentic Applications` 


- **凭据难题：寻求 Header 帮助**：一位成员在使用 Python 从客户端向 MCP server 通过 Header 传递凭据时遇到困难，正在寻求帮助。
   - 在给定的上下文中未提供解决方案或建议。
- **RAG 服务器文件摄取**：一位成员正考虑构建一个 **RAG 类型服务器**，客户端可以通过端点摄取文件，将其保存在服务器上，并用于回答问题。
   - 他们在询问这是否是一个好的方法，或者是否有更好的替代方案。
- **Streamable HTTP 的实现：等待身份验证评估**：一位成员询问社区对最近发布的 **TS SDK** 中当前 **Streamable HTTP 实现和身份验证** 的看法。
   - 另一位成员回答说它运行良好，但他们仍在摸索托管多租户服务器的细微差别以及有状态性（statefulness）如何影响它。
- **多租户服务器托管**：存在关于托管 **多租户服务器** 以及有状态性如何影响它的担忧。
   - 似乎无状态服务器应该为每个请求生成一个新的 MCP server 实例，但不清楚为什么 1 个实例对有状态服务器足够，而对无状态服务器却不够。
- **将 Agentic 开源模型投入生产：可行还是幻想？**：一位成员询问人们是否真的在生产环境（而不只是个人项目）中使用开源模型构建 Agentic 应用。
   - 他们发现大多数开源模型在没有微调的情况下很难进行推理或遵循指令。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1366882155824807976)** (1 条消息): 

> `MCP Server, Real Time Push Notifications` 


- **MCP 服务器在 Agent 工作流完成时发出通知**：一位成员推介使用 [mcp-gotify](https://github.com/SecretiveShell/mcp-gotify)（一个用于与 [gotify/server](https://github.com/gotify/server) 交互的 **MCP server**），以便在长时间运行的多 Agent 工作流完成时，在桌面和移动端接收实时推送通知。
- **Gotify server 替代方案？**：用户现在正使用 [gotify/server](https://github.com/gotify/server) 作为向桌面和移动端推送通知的替代方案。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1366719391240163328)** (9 条消息🔥): 

> `foreach optimization, gradient scaling, DoRA + QAT` 


- **通过 Foreach 实现快速梯度缩放**：一位成员分享了一个使用 `torch._foreach_mul_` 进行梯度缩放的 [代码片段](https://link.to/snippet)，可能将其与梯度裁剪合并为单个参数循环。
   - 另一位成员指出了[相关的 PR](https://github.com/pytorch/torchtune/pull/2624)，并想知道这种看似恒定的增益是否会在多次迭代中累积。
- **Tune 贡献者寻找易于入手的 Issue**：一位成员强调了 [两个简单 Issue](https://github.com/pytorch/torchtune/issues/2648) 和 [另一个](https://github.com/pytorch/torchtune/issues/2649)，供社区为项目做出贡献。
   - 未提供关于这些 Issue 性质的进一步信息。
- **DoRA 和 QAT：未探索的领域？**：一位成员询问了结合 **DoRA (Difference of Low-Rank Adaptation)** 与 **QAT (Quantization-Aware Training)** 的经验。
   - 在提供的消息中没有关于此组合的讨论或回复。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1366696277844164642)** (6 条消息): 

> `MCP Usage, Displaying thoughts component in React` 


- **渴望 MCP 使用文档**：一位用户询问了关于最新版本中新增的 **MCP (Multi-Controller Processing)** 使用教程或文档。
   - 另一位用户提到他们通过查看测试用例开始了学习，虽然有教程会更好，但并不紧迫，并澄清理解 **stdio** 和 **SSE clients** 的设置是关键。
- **React 中的 Thoughts 组件 - 最佳实践**：一位成员正在寻求关于在 **React** 中显示 **Thoughts 组件** 最佳方式的建议。
   - 他们知道可以修改 forward 方法，但询问是否有更好或更合适的地方来实现这一点。


  

---

### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1366789810827825152)** (1 messages): 

> `Markdown-based vs Image-based multimodal RAG on PDFs, Docling, EmbedV4` 


- **Markdown vs 图像 RAG 之争升温**：一位成员询问了关于在 **PDFs** 上比较**基于 Markdown** 与**基于图像的多模态 RAG** 的问题。
   - 他们目前正在使用 **Docling** 将 **PDFs** 转换为 **Markdown** 然后计算文本 **embedding**，但正在考虑切换到 **EmbedV4** 以输入原始图像并获取用于 **RAG** 的多模态 **embedding**。
- **探索 PDF 转换技术**：该成员正在使用 **Docling** 在计算文本 **embeddings** 之前将 **PDFs** 转换为 **Markdown**。
   - 他们正在评估 **EmbedV4** 作为替代方案，以便在 **RAG** 中直接处理原始图像以获取多模态 **embeddings**。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1366841428076003541)** (2 messages): 

> `Cohere rate limits for embed-v4, Embed V4 on Bedrock` 


- **Cohere 考虑提高速率限制**：一位用户询问 **Cohere** 是否会提高 `embed-v4` 的生产环境速率限制。
   - 他们表示，对于他们的 **PDFs** 使用场景，**每分钟 400 次请求**是不够的。
- **Cohere 考虑 Bedrock 的可用性**：一位用户询问 **Embed V4** 是否会在 **Bedrock** 上可用。
   - 目前 **Cohere** 尚未给出答复。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1366782678187114607)** (2 messages): 

> `Cohere's Embed V4 model, Data Scientists introductions` 


- **爱好者加入，渴望体验 Embed V4！**：一位新的数据科学家加入了 **Cohere Discord** 社区，表达了对尝试新工具的浓厚兴趣，特别是 **Cohere** 最新的 **Embed V4 model**，并探索其潜在应用。
   - 新成员表示*很高兴加入社区*。
- **社区欢迎新数据科学家**：**Cohere** 社区 **Discord** 服务器对新成员的加入表示欢迎。
   - 欢迎消息鼓励新成员提供其**公司/行业/大学**、*正在研究的具体内容*、喜欢的技术/工具，以及*希望从这个社区获得什么*。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1366781809614131220)** (5 messages): 

> `Embeddings, GPT4All, Manus AI, Embedding grouping` 


- **Manus AI 工具发布**：一位成员分享了 [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE) 的链接，声称*中国发布了它*，现在所有人都可以使用。
   - 该成员暗示这是*第一个自动研究 AI Agent*，并且*我们将被这个工具彻底取代*。
- **Embeddings 可以使用 Nomic 工具**：一位成员建议 **Nomic** 为 **embeddings** 提供了所有必要的工具，并且它*超越了 GPT4All*。
   - 他们声称 **Nomic** 的 **embeddings** 工具*可以在各种其他软件中工作*。
- **Embedding 分组可以替代训练**：一位成员描述了 **grouping embeddings** 如何替代训练：为特定的人分组 **embeddings** 并取平均值，然后使用该 **embedding** 对其他图片进行排序并找到同一个人。
   - 他问：*你理解这个概念了吗？*


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1366685162615865406)** (3 messages): 

> `Loose vs Strict Evaluation, Model Training Inconsistencies` 


- **宽松 vs 严格评估模型**：一位成员提出了一种为模型建立*“宽松”与“严格”评估机制*的想法，特别是针对那些可以通过*“黑客手段”*使其工作的模型，这代表了特定的使用场景。
   - 他们举了一个例子：一个模型被错误地训练为发出 `<tool_call>` 而不是其规范指出的 `<|tool_call|>`，在这种情况下，了解情况的用户可能会忽略该错误并评估其功能正确性。
- **模型训练导致不一致性**：一位成员遇到了一个被错误训练的模型，该模型发出 `<tool_call>` 而不是规范要求的 `<|tool_call|>`。
   - 该成员建议，如果他们专门了解该模型，可以忽略此错误并评估功能正确性，但普通用户无法做到这一点。