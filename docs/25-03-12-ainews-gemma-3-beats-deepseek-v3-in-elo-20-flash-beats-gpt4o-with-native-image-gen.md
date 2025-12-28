---
companies:
- google-deepmind
- openai
date: '2025-03-13T01:01:43.616815Z'
description: '**Google DeepMind** 发布了 **Gemma 3** 系列模型，其特点包括 **128k 上下文窗口**、**多模态输入（图像和视频）**以及对
  **140 多种语言的多语言支持**。**Gemma 3-27B** 模型在 LMArena 基准测试中位列顶级开源模型之列，表现优于多个竞争对手，并在基准测试中与
  **Gemini-1.5-Pro** 旗鼓相当。此外，**Gemini 2** 推出了具有高级图像编辑功能的 **Flash 原生图像生成**，这一功能 OpenAI
  曾进行过预告但尚未正式发布。这些更新突显了在上下文长度、多模态以及通过量化提升模型效率方面的重大进展。'
id: 88d8fa16-a359-48ca-8b50-fe959a5c6924
models:
- gemma-3
- gemini-1.5-pro
- gemini-2
- o1-preview
- o3-mini-high
- deepseek-v3
- claude-3.7-sonnet
- qwen-2.5-max
original_slug: ainews-gemma-3-beats-deepseek-v3-in-elo-20-flash
people:
- reach_vb
- _philschmid
- danielhanchen
- lmarena_ai
- osanseviero
title: Gemma 3 在 Elo 评分上击败了 DeepSeek V3，2.0 Flash 凭借原生图像生成能力超越了 GPT-4o。
topics:
- multimodality
- multilinguality
- context-window
- quantization
- image-generation
- model-benchmarking
- model-performance
- vision
---

<!-- buttondown-editor-mode: plaintext -->**GDM is all you need.**

> 2025年3月12日至3月13日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务区（**224** 个频道，**2511** 条消息）。预计节省阅读时间（按每分钟 200 词计算）：**275 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天的 o1-preview（目前唯一在 AINews 任务中能与 Flash Thinking 竞争的模型，而且没错，[o1-preview 比 o1-full 或 o3-mini-high 更好](https://x.com/swyx/status/1836515558810132628)）Discord 总结非常精准 —— Google 借在巴黎举办 [Gemma Developer Day](https://x.com/mervenoyann/status/1899773637063761938) 的契机，发布了一系列引人注目的更新：


![image.png](https://assets.buttondown.email/images/97aaf712-d79a-40ef-b467-b5d1465e42c4.png?w=960&fit=max)


https://www.youtube.com/watch?v=UU13FN2Xpyw

**Gemma 3**。人们非常喜欢它的 128k 上下文。除了作为一个开放模型在 LMArena 上取得的高分之外：


![image.png](https://assets.buttondown.email/images/3b6938f6-35d4-4819-add4-cd79b6d5cb77.png?w=960&fit=max)


它在同量级模型中也以绝对优势建立了一个新的 Pareto frontier：


![image.png](https://assets.buttondown.email/images/c3a8f375-c165-4e28-af66-caf51b461507.png?w=960&fit=max)


它看起来还通过将视觉作为一级能力（first class capability）引入，从而[完全取代了 PaliGemma](https://x.com/giffmana/status/1899776751925920181)（[ShieldGemma 仍然存在](https://x.com/mervenoyann/status/1899809277247623499)）。

**Gemini Flash 原生图像生成**。

正如在 Gemini 2 发布时所预告的（[我们的报道见此](https://buttondown.com/ainews/archive/ainews-google-wakes-up-gemini-20-et-al/)），[Gemini 2 实际上推出了图像编辑功能](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)，而 OpenAI 预告过却从未发布。其效果非常惊人（如果你能[从复杂的 UI 中找到它的话](https://x.com/fofrAI/status/1899924245918212201)）。图像编辑从未如此简单。

https://x.com/19kaushiks/status/1899856652666568732?s=46


https://x.com/m__dehghani/status/1899854209081868663?s=46

https://x.com/multimodalart/status/1899881757396099231

https://x.com/fofrAI/status/1899927094727000126



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 总结

**模型发布与更新：Gemma 3 系列**

- **Gemma 3 系列发布**：[@osanseviero](https://twitter.com/osanseviero/status/1899726995170210254) 宣布发布 **Gemma 3**，强调了其 **多语言能力（支持 140+ 种语言）**、**多模态输入（图像和视频）**、**LMArena 评分 1338** 以及 **128k 上下文窗口**。[@_philschmid](https://twitter.com/_philschmid/status/1899726907022963089) 提供了 Gemma 3 核心特性的摘要，包括 **四种尺寸（1B, 4B, 12B, 27B）**、**在 LMArena 开源非推理模型中排名第一**、**文本和图像输入**、**多语言支持**、**更大的上下文窗口**以及 **基于 SigLIP 的视觉编码器**。[@reach_vb](https://twitter.com/reach_vb/status/1899728796586025282) 总结了 Gemma 3 的关键特性，指出其 **性能媲美 OpenAI 的 o1**、**多模态和多语言支持**、**128K 上下文**、**通过量化实现的内存效率**以及 **训练细节**。[@scaling01](https://twitter.com/scaling01/status/1899792217352331446) 详细介绍了 Gemma 3，强调了其 **在 LMSLOP arena 的排名**、**与 Gemma 2 和 Gemini 1.5 Flash 的性能对比**、**使用 SigLip 的多模态支持**、**多种模型尺寸**、**长上下文窗口**和 **训练方法论**。[@danielhanchen](https://twitter.com/danielhanchen/status/1899728162130694266) 同样强调了 **Gemma 3** 的发布，指出其 **多模态能力**、**多种尺寸（1B 到 27B）**、**128K 上下文窗口**和 **多语言支持**，并表示 27B 模型在 **基准测试上与 Gemini-1.5-Pro 持平**。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1899729292617277501) 祝贺 Google DeepMind 推出 **Gemma-3-27B**，认可其为 **Arena 总榜前 10 名模型**、**第 2 优秀的开源模型**，并提到了其 **128K 上下文窗口**。[@Google](https://twitter.com/Google/status/1899916049002217855) 正式将 **Gemma 3** 作为其“迄今为止最先进且便携的开源模型”发布，专为智能手机和笔记本电脑等设备设计。
- **Gemma 3 性能与基准测试**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899729481176138122) 关注了 **Gemma 3 的性能**，强调 **27B 模型在 LMArena 排名第 9，超越了 o3-mini, DeepSeek V3, Claude 3.7 Sonnet 和 Qwen2.5-Max**。[@reach_vb](https://twitter.com/reach_vb/status/1899732585699533138) 发现 **Gemma3 4B 与 Gemma2 27B 具有竞争力**，并强调了“指数级的时间线”。[@reach_vb](https://twitter.com/reach_vb/status/1899734270328889367) 质疑 **Gemma3 27B 是否是最好的非推理 LLM**，尤其是在 **MATH** 领域。[@Teknium1](https://twitter.com/Teknium1/status/1899744944669315260) 将 **Gemma 3 与 Mistral 24B** 进行了对比，指出 **Mistral 在基准测试上表现更好，但 Gemma 3 拥有 4 倍的上下文和视觉能力**。
- **Gemma 3 技术细节**：[@vikhyatk](https://twitter.com/vikhyatk/status/1899773905591792054) 审阅了 **Gemma 3 技术报告**，提到模型名称与参数量相匹配，且 4B 以上的模型均为多模态。[@nrehiew_](https://twitter.com/nrehiew_/status/1899882552946532498) 分享了对 **Gemma 3 技术报告** 的看法，指出虽然缺乏细节但提供了有趣的信息。[@eliebakouch](https://twitter.com/eliebakouch/status/1899790607993741603) 对 **Gemma3 技术报告** 进行了详细分析，涵盖了架构、长上下文和蒸馏技术。[@danielhanchen](https://twitter.com/danielhanchen/status/1899735308180267176) 提供了 **Gemma-3 分析**，详述了架构、训练、聊天模板（chat template）、长上下文和视觉编码器。[@giffmana](https://twitter.com/giffmana/status/1899776751925920181) 确认 **Gemma3 转向多模态**，取代了 PaliGemma，并可与 **Gemini 1.5 Pro** 媲美。
- **Gemma 3 可用性与使用**：[@ollama](https://twitter.com/ollama/status/1899742981676007791) 宣布 **Gemma 3 在 Ollama 上可用**，包括多模态支持和运行不同尺寸模型的命令。[@_philschmid](https://twitter.com/_philschmid/status/1899816992585945539) 强调了使用 **`google-genai` SDK 测试 Gemma 3 27B**。[@_philschmid](https://twitter.com/_philschmid/status/1899863222649331747) 分享了一篇关于 **Gemma 3 开发者信息** 的博客。[@_philschmid](https://twitter.com/_philschmid/status/1899726910227181889) 分享了在 **AI Studio 中试用 Gemma 3** 的链接以及 **模型链接**。[@mervenoyann](https://twitter.com/mervenoyann/status/1899823530524447133) 提供了一个关于 **Gemma 3 视频推理** 的 Notebook，展示了其视频理解能力。[@ggerganov](https://twitter.com/ggerganov/status/1899749881624817971) 宣布 **Gemma 3 支持已合并至 llama.cpp**。[@narsilou](https://twitter.com/narsilou/status/1899813420007919925) 指出 **Text generation 3.2 已发布并支持 Gemma 3**。[@reach_vb](https://twitter.com/reach_vb/status/1899729961658855614) 提供了一个 **体验 Gemma 3 12B 模型** 的 Space 空间。

**机器人与具身智能 (Robotics and Embodied AI)**

- **Gemini Robotics 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839624068907335) 推出了 **Gemini Robotics**，这是基于 **Gemini 2.0** 的新一代机器人 AI 模型，强调 **推理、交互、灵活性和泛化能力**。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839644302270671) 宣布与 **Apptronik** 建立合作伙伴关系，利用 **Gemini 2.0** 构建人形机器人，并向 **Agile Robots**、**AgilityRobotics**、**BostonDynamics** 和 **EnchantedTools** 等**受信任的测试者开放了 Gemini Robotics-ER 模型**。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839641693430008) 表示其目标是开发**适用于任何形状或尺寸机器人的 AI**，包括 **ALOHA 2**、**Franka** 和 **Apptronik 的 Apollo** 等平台。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839638493077892) 解释说 **Gemini Robotics-ER** 允许机器人利用 **Gemini 的具身推理 (embodied reasoning)**，实现目标检测、交互识别和避障。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839635720663463) 强调了 **Gemini Robotics 的泛化能力**，其在基准测试中的表现比最先进的模型（state-of-the-art models）翻了一倍。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839632772067355) 强调了通过 **Gemini Robotics 实时调整动作的能力**实现**无缝的人机交互**。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839630242955536) 展示了 **Gemini Robotics 将正时皮带绕在齿轮上**这一极具挑战性的任务。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1899839627139383762) 演示了 **Gemini Robotics 完成多步骤灵巧任务**，如折纸和打包便当盒。
- **Figure 机器人与 AGI**：[@adcock_brett](https://twitter.com/adcock_brett/status/1899587483928805642) 表示 **Figure 将成为 AGI 的最终部署载体**。[@adcock_brett](https://twitter.com/adcock_brett/status/1899608776313090127) 分享了机器人技术的更新，指出其在**速度提升**、处理**易变形袋子**以及**将神经网络权重迁移到新机器人**方面的进展，感觉就像“上传到了《黑客帝国》(Matrix)！”。[@adcock_brett](https://twitter.com/adcock_brett/status/1899655660192833560) 将 **Helix 描述为解决通用机器人技术的一道微光**。[@adcock_brett](https://twitter.com/adcock_brett/status/1899665426025795880) 提到他们的机器人**完全嵌入式且离网运行，配备 2 个嵌入式 GPU**，目前不需要网络调用。

**AI Agent 与工具**

- **Agent 工作流与框架**：[@LangChainAI](https://twitter.com/LangChainAI/status/1899922355004334492) 宣布了一个 **Resources Hub**，包含构建 AI Agent 的指南，以及关于 AI 趋势和 **Replit**、**Klarna**、**tryramp** 和 **LinkedIn** 等公司用例的报告。[@omarsar0](https://twitter.com/omarsar0/status/1899571435938677222) 正在主持一场关于**使用 OpenAI 的 Agents SDK 构建高效 Agentic 工作流的免费网络研讨会**。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1899779019740258439) 列出了 **7 个支持 AI Agent 动作的开源框架**，包括 **LangGraph**、**AutoGen**、**CrewAI**、**Composio**、**OctoTools**、**BabyAGI** 和 **MemGPT**，并提到了 **OpenAI 的 Swarm** 和 **HuggingGPT** 等新兴方法。[@togethercompute](https://twitter.com/togethercompute/status/1899862571097661653) 宣布了 **5 份关于使用 Together AI 构建 Agent 工作流的详细指南**，每份指南都附带深入探讨的 Notebook。
- **Model Context Protocol (MCP) 与 API 集成**：[@llama_index](https://twitter.com/llama_index/status/1899848532817035529) 宣布 **LlamaIndex 与 Model Context Protocol (MCP) 集成**，支持通过一行代码连接到任何 MCP 服务器并进行工具发现。[@PerplexityAI](https://twitter.com/perplexity_ai/status/1899849114583765356) 发布了 **Perplexity API Model Context Protocol (MCP)**，为 Claude 等 AI 助手提供实时网页搜索功能。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1899850017546129445) 宣布 **Perplexity API 现在支持 MCP**，为 Claude 等 AI 提供实时信息。[@cognitivecompai](https://twitter.com/cognitivecompai/status/1899736936039825705) 展示了 **Dolphin-MCP**，这是一个开源且灵活的 MCP 客户端，兼容 Dolphin、ollama、Claude 和 OpenAI 端点。[@hwchase17](https://twitter.com/hwchase17/status/1899873990774243749) 询问在 IDE 中为了让 LangGraph/LangChain 更易于访问，应该使用 **llms.txt 还是 MCP**。
- **OpenAI API 更新**：[@LangChainAI](https://twitter.com/LangChainAI/status/1899888134793683243) 宣布 **LangChain 支持 OpenAI 的新 Responses API**，包括内置工具和对话状态管理。[@sama](https://twitter.com/sama/status/1899579431905305027) 称赞 **OpenAI 的 API 设计**是“有史以来设计最精良、最实用的 API 之一”。[@corbtt](https://twitter.com/corbtt/status/1899585079695069580) 发现 **OpenAI 的新 API 形式比 Chat Completions API 更好**，但希望能够跳过对这两个 API 的同时支持。

**AI 性能与优化**

- **GPU 编程与性能**：[@hyhieu226](https://twitter.com/hyhieu226/status/1899854357354688736) 指出 **warp divergence 是 GPU 编程中一个微妙的性能 Bug**。[@awnihannun](https://twitter.com/awnihannun/status/1899861832774668399) 发布了**编写更快 MLX 并避免性能悬崖的指南**。[@awnihannun](https://twitter.com/awnihannun/status/1899822376797536701) 强调了 **MLX 社区对 Gemma 3 的快速支持**，涵盖 **MLX VLM**、**MLX LM** 和 **MLX Swift for iPhone**。[@tri_dao](https://twitter.com/tri_dao/status/1899669458995614179) 将讨论**在现代硬件上优化 Attention** 以及 **Blackwell SASS 技巧**。[@clattner_llvm](https://twitter.com/clattner_llvm/status/1899913688158798055) 讨论了 **TVM 和 XLA 等 AI 编译器**，以及为什么 **GenAI 仍然使用 CUDA 编写**。
- **模型优化与效率**：[@scaling01](https://twitter.com/scaling01/status/1899614996788645936) 推测 **OpenAI 可能很快发布 o1 模型**，因为它处理复杂任务的能力优于 o3-mini。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899886591344095457) 质疑 **Google 在 Gemma 3 中激进地随模型 N 缩放 D 的逻辑**，并询问**在 2T 数据上训练的 Gemma-1B** 及其对投机采样的适用性。[@rsalakhu](https://twitter.com/rsalakhu/status/1899597917016744445) 分享了关于**将优化推理时计算（test-time compute）作为元强化学习问题**的新工作，从而产生了 **Meta Reinforcement Fine-Tuning (MRT)**，以提高性能和 Token 效率。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1899716309127983535) 询问**散热是否是制造更大芯片的关键问题**。

**AI 研究与论文**

- **AI 生成科学论文**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1899646987781501181) 宣布 **The AI Scientist-v2** 的一篇论文通过了 ICLR workshop 的同行评审，声称这是**首篇完全由 AI 生成并通过同行评审的论文**。[@hardmaru](https://twitter.com/hardmaru/status/1899665717215326283) 分享了该实验的细节，记录了过程与心得，并在 GitHub 上发布了 AI 生成的论文和人类评审意见。[@hardmaru](https://twitter.com/hardmaru/status/1899825814327484556) 调侃 **The AI Scientist 被 Schmidhubered 了**。[@hkproj](https://twitter.com/hkproj/status/1899771070690766920) 在一篇 AI Scientist 论文被接收后对 **ICLR 的标准**提出了质疑。[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1899824257112391796) 承认了 **The AI Scientist 的引用错误**，错误地归属了“一个基于 LSTM 的神经网络”，并记录了人类评审中的错误。
- **扩散模型与图像生成**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899722047854404092) 重点介绍了一篇关于**使用 SoftREPA 改进扩散模型中文本到图像对齐**的论文。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899715828419010769) 分享了一篇关于**使用 Latent CLIP 控制 Latent Diffusion** 的论文，在潜空间（latent space）中训练 CLIP 模型。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899882749135106341) 称一项新的算法突破是“罕见且令人印象深刻”的进展，可能会终结 **Consistency Models 甚至扩散模型**。
- **长篇音乐生成**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899717628912157104) 分享了关于 **YuE** 的工作，这是一个用于**长篇音乐生成**的开源基础模型系列，能够生成长达五分钟且歌词对齐的音乐。
- **混合专家模型 (MoE) 的可解释性**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899718810363642242) 重点介绍了 **MoE-X**，这是一种重新设计的 MoE 层，旨在提高 LLM 中 MLP 的可解释性。
- **Gemini Embedding**：[@_akhaliq](https://twitter.com/_akhaliq/status/1899674020880027752) 分享了 **Gemini Embedding**，即来自 Gemini 的通用嵌入模型。
- **视频创作与编辑 AI**：[@_akhaliq](https://twitter.com/_akhaliq/status/1899672874262081753) 展示了**阿里巴巴的 VACE**，这是一款全能视频创作与编辑 AI。[@_akhaliq](https://twitter.com/_akhaliq/status/1899671379819086291) 分享了一篇关于**通过同步耦合采样实现免微调的多事件长视频生成**的论文。
- **注意力机制与 Softmax**：[@torchcompiled](https://twitter.com/torchcompiled/status/1899894436802506976) 声称**注意力机制中 Softmax 的使用是随意的**，并且存在一个影响 LLM 的“bug”，并链接到一篇新帖子。[@torchcompiled](https://twitter.com/torchcompiled/status/1899901965053944148) 批评注意力机制**缺乏“无操作（do nothing）”选项**，并建议**温度缩放应取决于序列长度**。

**行业与商业**

- **AI 商业与应用**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1899864329908084913) 表示 **Perplexity API 可以生成 PowerPoint**，本质上是通过一次 API 调用取代了顾问的工作。[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1899852903403454624) 宣布 **GroupMe 集成了 Copilot**，为数百万用户（尤其是美国大学生）提供应用内 AI 支持。[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1899853293750538451) 强调 **GroupMe 中的 Copilot** 让群聊不再混乱，并在作业、建议和回复方面提供帮助。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1899631747828174852) 讨论了超越基础 AI 技能的必要性，并拥抱**合成数据、RAG、多模态 AI 和上下文理解**，强调全民 AI 素养。[@sarahcat21](https://twitter.com/sarahcat21/status/1899734834282405990) 指出**编程变得更容易，但软件构建仍然很难**，原因在于数据管理、状态管理和部署方面的挑战。[@mathemagic1an](https://twitter.com/mathemagic1an/status/1899625715391508585) 强调 **shadcn/ui 集成**是 v0 成功的一部分，并赞扬了 Notion 的 UI 套件在知识工作类应用中的表现。
- **AI 市场与竞争**：[@nearcyan](https://twitter.com/nearcyan/status/1899624995413950910) 指出，**在 Anthropic 达到 14 万亿美元估值后，Google 的价值预计将翻倍**。[@mervenoyann](https://twitter.com/mervenoyann/status/1899774973725540627) 调侃 **Google 凭借 Gemma 3 “随手干掉了其他模型”**。[@scaling01](https://twitter.com/scaling01/status/1899873762222186528) 声称 **Google 凭借 Gemini 2.0 Flash 在市场上击败了 OpenAI**，并展示了其图像重建能力。[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1899873487231345062) 表示 **Google 凭借 Gemma3 1B 加入了“小模型俱乐部”**，展示了小模型发布加速的时间线以及日益激烈的竞争。[@scaling01](https://twitter.com/scaling01/status/1899614996788645936) 预测**如果 OpenAI 不尽快发布 GPT-5，o1 将占据主导地位**。
- **AI 招聘与人才**：[@fchollet](https://twitter.com/fchollet/status/1899672830897496326) 为垂直整合无人机初创公司 **Harmattan** 招聘**对欧洲国防充满热情的机器学习工程师**。[@saranormous](https://twitter.com/saranormous/status/1899914557352796528) 在关于初创公司招聘的推文中强调，**优先考虑早期招聘以避免恶性循环**。[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1899769768858857770) 正在为 AI 业务计划招聘一名**网络安全工程师**。[@giffmana](https://twitter.com/giffmana/status/1899740069893677427) 表示在 **OpenAI** 工作很开心，因为那里有聪明的人、有趣的工作以及“倾向于把事情办成”的氛围。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899594295914496302) 表示**中国今年将有数百名高水平的机器学习毕业生**。[@rsalakhu](https://twitter.com/rsalakhu/status/1899600121324544398) 祝贺 **Murtaza Dalal 博士**完成博士学位。
- **AI 基础设施与算力**：[@svpino](https://twitter.com/svpino/status/1899871762135089314) 推广 **Nebius Explorer Tier 以每小时 1.50 美元的价格提供 H100 GPU**，强调其价格低廉且可立即配置。[@dylan522p](https://twitter.com/dylan522p/status/1899914025674371188) 宣布了一场**拥有超过 100 块 B200/GB200 GPU 的黑客松**，演讲嘉宾来自 OpenAI、Thinking Machines、Together 和 Nvidia。[@dylan522p](https://twitter.com/dylan522p/status/1899866196809552079) 称赞 **Texas Instruments 广州**的 IC 零件比美国分销商更便宜。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899622820058956280) 分析了**华为 2019 年的数据中心硬件**，指出了其性能以及制裁的影响。[@cHHillee](https://twitter.com/cHHillee/status/1899655656455692379) 将在 GTC 讨论 **机器学习系统和 Blackwell GPU**。

**迷因与幽默**

- **AI 能力与局限性**：[@scaling01](https://twitter.com/scaling01/status/1899913342892073303) 开玩笑说要**重新发明 Diffusion 模型**，并建议 **Google 应该在图像生成上训练一个推理模型来修复拼写错误**。[@scaling01](https://twitter.com/scaling01/status/1899916277101080668) 发现 **Gemini 2.0 Flash 通过提示词迭代改进了梗图中难以辨认的文本**。[@scaling01](https://twitter.com/scaling01/status/1899873556340859302) 发布了一张对比 Google 和 OpenAI 的图片，并配文 "checkmate"（将军）。[@goodside](https://twitter.com/goodside/status/1899895643352510609) 展示了 **Gemini 2.0 Flash 在上传图片的 T 恤上将 "BASE" 修改为 "BASED"**。[@scaling01](https://twitter.com/scaling01/status/1899875985064923387) 使用文生图技术制作了一个 **Google vs OpenAI 的梗图**。[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1899643788911431682) 开玩笑说需要一个 **captcha 系统来让 AI 证明你不是人类**。[@scaling01](https://twitter.com/scaling01/status/1899874764153385230) 幽默地使用了 **#savegoogle** 标签。[@scaling01](https://twitter.com/scaling01/status/1899873556340859302) 在 Google vs OpenAI 的背景下使用了 "checkmate" 梗。
- **AI 与社会**：[@oh_that_hat](https://twitter.com/oh_that_hat/status/1899667358278377762) 建议**像你希望 AI 对待你那样对待网上的其他人**，因为 AI 会从网络互动中学习。[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1899819121727922524) 讨论了关于自动化和真实性的**审美价值与实用偏好之间的差距**。[@jd_pressman](https://twitter.com/jd_pressman/status/1899705082213490942) 分享了一个想象中的场景：向 2015 年的人展示一张澄清 AI 能力的截图。[@jd_pressman](https://twitter.com/jd_pressman/status/1899689933113298949) 将**像 GPT 的 AI 反派**（The Master, XERXES, Dagoth Ur, Gravemind）与那些不像的（HAL 9000, GladOS, 343 Guilty Spark, X.A.N.A.）区分开来。[@qtnx_](https://twitter.com/qtnx_/status/1899588703976124486) 分享了“我的生活就是这一系列无止尽的 gif”并配上相关 gif。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1899699884472578313) 分享了一个关于瑞士和俄罗斯军官及手表的苏联公民笑话。
- **一般幽默与讽刺**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899678018248818740) 发布了“结束了。梗图被反转了。这是我今天读到最搞笑的事”并附带链接。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1899677468706943069) 讽刺地说“笑死，美国完全没为那场战争做好准备”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Gemma 3 多模态发布：视觉、文本及 128K 上下文**

- **[Gemma 3 发布 - Google Collection](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)** ([Score: 793, Comments: 218](https://reddit.com/r/LocalLLaMA/comments/1j9dkvh/gemma_3_release_a_google_collection/))：**Gemma 3** 已作为 **Google Collection** 发布，尽管该帖子缺乏关于其功能或影响的进一步细节或背景。
  - **Gemma 3 功能与问题**：用户注意到 **Gemma 3** 不支持 tool calling，并且在 AIstudio 的 **gemma-3-27b-it** 中存在图像输入问题。模型架构尚未被 **Transformers** 等平台识别，且尚未在 **LM Studio** 上运行。
  - **性能与对比**：**4B Gemma 3 模型** 超过了 **9B Gemma 2**，而 **12B 模型** 因其强大的 vision 能力而受到关注。尽管性能出色，用户报告它在 **ollama** 上经常崩溃，且缺乏 function calling 等功能。**EQ-Bench** 结果显示 **27b-it** 模型在创意写作方面排名第二。
  - **模型可用性与技术细节**：Gemma 3 模型可在 **ollama** 和 **Hugging Face** 等平台获取，并提供了各种资源和技术报告的链接。模型支持高达 **128K tokens**，并采用 **Quantization Aware Training** 以减少内存占用，目前正致力于在 Hugging Face 上添加更多版本。

- **Gemma 3 27b 现已在 Google AI Studio 上线** ([Score: 313, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1j9bvll/gemma_3_27b_now_available_on_google_ai_studio/)): **Gemma 3 27B** 现已在 **Google AI Studio** 上提供，具有 **128k 的上下文长度**和 **8k 的输出长度**。更多详情可以在提供的 [Google AI Studio](https://aistudio.google.com/) 和 [Imgur](https://imgur.com/a/2WvMTPS) 链接中找到。
  - 用户讨论了 **system prompt** 及其对 **Gemma 3** 回复的影响，指出它有时能提供超出其所谓截止日期的信息，例如在被问及 2021 年之后的事件时。一些用户在处理逻辑和写作任务的能力方面报告了不同的体验，并将其与 **Gemma 2** 及其局限性进行了比较。
  - **性能问题**被重点提及，几位用户表示 **Gemma 3** 目前运行较慢，尽管与 **Gemma 2** 相比，它在指令遵循方面有所改进。还有关于其翻译能力的讨论，一些人声称它优于 **Google Translate** 和 **DeepL**。
  - 分享了 **Gemma 3** 在 **Hugging Face** 发布的链接，提供了各种模型版本的访问权限。用户表达了对开放权重和 **benchmarks** 的期待，以便更好地评估模型的性能和能力。


- **Gemma 3 在 Huggingface 上线** ([Score: 154, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1j9dt8l/gemma_3_on_huggingface/)): **Google** 的 **Gemma 3** 模型已在 **Huggingface** 上提供，参数规模包括 **1B、4B、12B 和 27B**，并提供了每个规模的链接。它们支持文本和图像输入，较大模型的总输入上下文为 **128K tokens**，1B 模型为 **32K tokens**，并产生 **8192 tokens** 的输出上下文。该模型已添加到 **Ollama**，并在 **Chatbot Arena** 上拥有 **1338 的 ELO 分数**，超越了 **DeepSeek V3 671B**。
  - **模型上下文和 VRAM 需求**：**27B Gemma 3** 模型在 **128K 上下文**下需要高达 **45.09GB 的 VRAM**，这对于没有像第二块 **3090** 这样的高端 GPU 的用户来说是一个挑战。**8K** 指的是输出 token 上下文，而较大模型的输入上下文为 **128K**。
  - **模型性能和特性**：用户将 **27B Gemma 3** 模型与 **1.5 Flash** 进行了比较，但指出它的行为有所不同，类似于 **Sonnet 3.7**，会对简单问题提供详尽的回答，暗示其具有作为系统工程师工具的潜力。
  - **运行和兼容性问题**：由于版本不兼容，一些用户在 **Ollama** 上运行模型时遇到问题，但更新软件可以解决此问题。**GGUFs** 和模型版本可在 [Huggingface](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b) 获取，用户在部署模型时应注意双 **BOS tokens** 问题。


**主题 2. Unsloth 的 GRPO 修改：Llama-8B 的自学习改进**

- **我修改了 Unsloth 的 GRPO 代码以支持 Agent 工具使用。在 RTX 4090 上训练 1 小时，Llama-8B 学会了迈向深度研究的第一步！（准确率从 23% 提升至 53%）** ([Score: 655, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1j96j3g/i_hacked_unsloths_grpo_code_to_support_agentic/)): 我修改了 **Unsloth** 的 **GRPO** 代码，使 **Llama-8B** 能够以 Agent 方式使用工具，通过自我对弈 (self-play) 增强其研究技能。仅在 **RTX 4090** 上训练一小时，该模型就通过生成问题、搜索答案、评估成功与否以及通过强化学习 (reinforcement learning) 完善其研究能力，将准确率从 **23% 提高到 53%**。你可以在这里找到完整的 [代码和说明](https://github.com/dCaples/AutoDidact/)。
  - 用户对 **强化学习 (RL) 过程** 表现出好奇，特别是数据集的创建和持续的权重调整。作者解释说，他们从 **LLM** 生成并过滤回复以创建用于微调 (fine-tuning) 的数据集，并迭代重复此过程。
  - 将此方法应用于 **Llama 70B 和 405B** 等更大模型的兴趣非常浓厚，作者提到正在努力设置 **FSDP** 以进行进一步实验。
  - 社区对该项目表现出强烈的支持和兴趣，建议向 **Unsloth** 仓库贡献代码，并对分享工作表示感谢，强调了其在“Agent 之年”中潜在的行业相关性。

- **Gemma 3 - GGUF 及其推荐设置** ([Score: 171, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1j9hsfc/gemma_3_ggufs_recommended_settings/)): **Gemma 3** 是 Google 推出的新型多模态模型，目前已在 **Hugging Face** 上提供 **1B、4B、12B 和 27B** 尺寸，并上传了 **GGUF 和 16-bit** 版本。此处提供了运行 Gemma 3 的[分步指南](https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively)，推荐的推理设置包括 **temperature 为 1.0**、**top_k 为 64** 以及 **top_p 为 0.95**。使用 **4-bit QLoRA** 进行训练目前存在已知 Bug，但预计很快会发布更新。
  - **温度与性能问题**：用户确认 **Gemma 3** 在 **1.0** 的 **temperature** 下运行，这并不被认为很高，但仍有用户报告了性能问题，例如与 **Qwen2.5 32B** 等其他模型相比速度较慢。一位使用 **RTX 5090** 的用户指出 **Gemma 3** 的性能较慢，其中 **4B 模型** 的运行速度甚至比 **9B 模型** 还慢，这引发了 Gemma 团队的进一步调查。
  - **系统提示词与推理挑战**：讨论强调 **Gemma 3** 缺乏原生的系统提示词（system prompt），需要用户将系统指令合并到用户提示词中。此外，在 **LM Studio** 中运行 **GGUF 文件** 存在问题，建议使用 **dynamic 4-bit** 推理而非 **GGUF**，但由于 Transformer 的问题，目前尚未上传。
  - **量化与模型兼容性**：**Gemma2-27B** 的 **IQ3_XXS 量化**版本因其 **10.8 GB** 的超小体积而受到关注，使其能够在 **3060 GPU** 上运行。用户对显存（VRAM）需求的准确性展开了辩论，一些人断言 **16GB** 显存不足以运行 **27B 模型**，而另一些人则认为配合 **Q8 cache quantization** 可以有效运行。


**主题 3. M3 Ultra 上的 DeepSeek R1：SoC 能力洞察**

- **[M3 Ultra 使用 448GB 统一内存运行 6710 亿参数的 DeepSeek R1，在功耗低于 200W 的情况下提供高带宽性能，无需多 GPU 配置](https://wccftech.com/m3-ultra-chip-handles-deepseek-r1-model-with-671-billion-parameters/)** ([Score: 380, Comments: 159](https://reddit.com/r/LocalLLaMA/comments/1j9jfbt/m3_ultra_runs_deepseek_r1_with_671_billion/)): **DeepSeek R1** 在 **M3 Ultra** 上以 **6710 亿参数**运行，使用 **448GB 统一内存**，在功耗低于 **200W** 的情况下实现了高带宽性能。该方案消除了对多 GPU 配置的需求。
  - 讨论重点集中在 **DeepSeek R1** 在 **M3 Ultra** 上的**提示词处理速度**和上下文大小限制，多位用户对缺乏具体数据表示沮丧。用户强调，即使达到 **18 tokens per second**，在大上下文尺寸下开始生成内容所需的时间也是不切实际的，通常需要几分钟。
  - 对于 **Apple Silicon** 用于大模型本地推理的实用性存在怀疑，许多用户指出，尽管 **M3 Ultra** 规格惊人，但其性能并不适合处理复杂的上下文管理或训练任务。用户认为 **NVIDIA** 和 **AMD** 的产品虽然功耗更高，但在这些任务上可能更有效。
  - 讨论还涉及了 **KV Cache** 提升 **Mac** 系统性能的潜力，但用户注意到在处理复杂上下文管理时存在局限性。此外，连接 **eGPU** 以增强处理能力的可行性也引发了辩论，一些用户指出 macOS 缺乏对 **Vulkan** 的支持是一个障碍。

- **[EXO Labs 在两台 M3 Ultra 512GB Mac Studio 上分布式运行了完整的 8-bit DeepSeek R1 - 11 t/s](https://x.com/alexocheema/status/1899735281781411907)** ([Score: 143, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1j9gafp/exo_labs_ran_full_8bit_deepseek_r1_distributed/)): **EXO Labs** 在两台 **M3 Ultra 512GB Mac Studio** 上执行了完整的 **8-bit DeepSeek R1** 分布式处理，实现了 **11 t/s (每秒 token 数)** 的性能。
  - 讨论强调了使用 **M3 Ultra Mac Studio** 与 GPU 等其他硬件相比的**成本和性能权衡**。虽然 Mac Studio 提供了紧凑且安静的设置，但因其**提示词处理速度慢**和高昂的费用（尤其是 RAM 和 SSD 的定价）而面临批评，尽管它具有能效高和节省空间的优点。
  - 对话强调了**批处理 (Batching)** 对于在 Mac Studio 等昂贵硬件设置上最大化吞吐量的重要性，并将其与可以并行处理多个请求的 GPU 集群进行了对比。文中还将其与 **H200 集群**等替代方案进行了比较，后者尽管成本和功耗更高，但在批处理场景下提供了显著更快的性能。
  - 用户对**提示词处理指标 (prompt processing metrics)** 有显著需求，多位用户对 **EXO Labs** 分享的结果中缺乏这些数据表示失望。短提示词的**首个 token 生成时间 (time to first token)** 为 **0.59 秒**，但用户认为这不足以衡量整体性能。


**Theme 4. Gemma 3 开源努力：Llama.cpp 及更多**

- **[Gemma 3 - 开源努力 - llama.cpp - MLX 社区](https://i.redd.it/x3jb302hn9oe1.jpeg)** ([Score: 160, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1j9kxqq/gemma_3_open_source_efforts_llamacpp_mlx_community/)): **Gemma 3** 发布并提供开源支持，强调了 **ngyson、Google 和 Hugging Face** 之间的协作。由 **Colin Kealty** 和 **Awni Hannun** 分享的这一公告强调了 **MLX 社区**内的社区努力，并表彰了主要贡献者，庆祝该模型的进步。
  - **vLLM** 项目正在积极集成 **Gemma 3** 支持，尽管根据以往的表现，人们对其能否按发布计划完成持怀疑态度。文中分享了相关 GitHub pull requests 的链接以跟踪进度。
  - **Google 对该项目的贡献**因其前所未有的速度和支持而受到称赞，特别是在协助与 **llama.cpp** 的集成方面。这次协作被视为一次重大的突破，人们对在 **LM Studio** 中尝试 **Gemma 3 27b** 感到兴奋。
  - **Hugging Face**、**Google** 和 **llama.cpp** 之间的协作被强调为一次成功的努力，使 **Gemma 3** 能够被迅速使用，并对 **Son** 的贡献给予了特别认可。


- **[QwQ 在高思考强度设置下一次性通过弹球示例](https://v.redd.it/nrf0zws0w9oe1)** ([Score: 115, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1j9lwlw/qwq_on_high_thinking_effort_setup_oneshotting_the/)): 该帖子讨论了 **Gemma 3** 及其与开源 **MLX 社区** 的兼容性，特别是为了高效执行**弹球示例 (bouncing balls example)** 所需的高强度设置。
  - **GPU Offloading 与性能**：用户讨论了通过将处理任务卸载到 GPU 来优化**弹球示例**，其中一名用户使用 **Llama** 在 **40 个 GPU 层**上实现了 21,000 个 token。然而，他们遇到了球体消失的问题，需要调整重力和摩擦力等参数以获得更好的表现。
  - **思考强度控制 (Thinking Effort Control)**：**ASL_Dev** 分享了一种通过调整 `</think>` token 的 logit 来控制模型思考强度的方法，在思考强度设置为 **2.5** 时实现了可运行的模拟。他们为该实验性设置提供了一个 [GitHub 链接](https://github.com/and270/thinking_effort_processor)，与常规设置相比，该设置提高了模拟的性能。
  - **推理引擎自定义**：讨论强调了推理引擎允许手动调整推理强度的潜力，类似于 **OpenAI** 的模型。用户注意到 **openwebui** 等平台已经提供了此功能，并且人们对为推理模型添加权重调节器以增强自定义功能表现出兴趣。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. DeepSeek 和 ChatGPT 审查：观察与反弹**

- **[DeepSeek 有那么一瞬间忘记了谁是它的所有者](https://v.redd.it/o5tlxocoz9oe1)** ([Score: 6626, Comments: 100](https://reddit.com/r/ChatGPT/comments/1j9mdli/deepseek_forgot_who_owns_it_for_a_second/)): 该帖子讨论了 **DeepSeek**，强调了 AI 在识别其所有权时出现的瞬间失误，这引发了对潜在 **AI censorship** 的担忧。由于帖子缺乏详细的背景或分析，这一事件的具体影响仍有待解读。
  - **审查担忧**: 用户对 AI 系统生成完整响应后又将其撤回表示沮丧，这表明存在一种在生成后检查“禁忌话题”的内容机制。与 **ChatGPT** 这种将 guardrails 集成到 AI 逻辑中的系统相比，这种方法被认为不够优雅，凸显了 AI 审查中的透明度问题。
  - **中国审查**: 有推测认为，这种审查机制可能是针对**中国审查政策**的一种“恶意合规”，一些用户建议，该系统糟糕的实现是有意为之，旨在突出审查问题。
  - **技术建议**: 用户建议 AI 系统应该生成完整响应，通过过滤器运行，然后再显示，以避免目前这种可能被撤销的 streaming 答案的做法，因为这被认为效率低下且对用户不友好。


- **[DeepSeek 忘记了自己的老板……](https://v.redd.it/igrry49ioaoe1)** ([Score: 234, Comments: 11](https://reddit.com/r/OpenAI/comments/1j9preb/deepseek_forgot_its_own_owner/)): 帖子标题 **"DeepSeek Forgot Its Own Owner"** 暗示了围绕 **DeepSeek** 所有权的混乱或争议，可能反映了与 censorship 相关的更广泛问题。在没有额外背景或视频分析的情况下，无法获得进一步的细节。
  - **社交媒体评论**: 用户对 **Reddit** 平台表示不满，一条评论讽刺地称其为“有史以来最典型的社交媒体平台之一”，而另一条评论则强调了在媒体化之前对其社交属性的保留。
  - **审查与讽刺**: 评论影射了审查问题，其中包含涉及 **Xi Jinping** 的讽刺性言论，以及对视频末尾“社会信用广告”的幽默调侃，表明了对审查或控制机制的批评。
  - **技术观察**: 一位用户指出了关于 **DeepSeek** 的一个技术细节，观察到在只剩下三个字母时出现了短暂的停顿，幽默地将其归因于“正对着红色按钮扶额”。


**Theme 2. Claude Sonnet 3.7: A Standout in Coding Conversion Tasks**

- **Claude Sonnet 3.7 的编程能力简直疯狂！** ([Score: 324, Comments: 126](https://reddit.com/r/ClaudeAI/comments/1j9kov0/claude_sonnet_37_is_insane_at_coding/)): **Claude Sonnet 3.7** 在将复杂的 JavaScript 应用程序转换为 **Vue 3** 方面表现出色，正如它在单个会话中将一个包含 2,000 行 JavaScript 的 4,269 行应用程序重构为 Vue 3 应用程序所证明的那样。它有效地保留了应用程序的功能、用户体验和组件依赖关系，实现了合理的组件结构、**Pinia stores**、**Vue Router** 和拖放功能，展示了相比 **Claude 3.5** 的显著改进。
  - 讨论强调了 **Claude 3.7** 取代传统 BI 工具和分析师的能力，一位用户分享了它如何在几分钟内将 **Mixpanel** 的 CSV 数据转换为全面的仪表板，节省了与 BI 工具和分析师相关的巨额成本。
  - 用户对 **Claude 3.7** 的评价褒贬不一，一些人称赞其创建无 Bug 复杂应用的能力，而另一些人则批评其过度发挥和产生功能 hallucination 的倾向，反映了关于 AI 在编程中有效性的更广泛辩论。
  - 社区对 **Claude 3.7** 的看法存在两极分化，这是一个幽默的观察，一些用户认为它是革命性的，而另一些人则认为它存在问题，这说明了 AI 工具评估的多样性，有时甚至是矛盾的。


**Theme 3. Open-Source Text-to-Video Innovations: New Viral Demos**

- **[我刚刚开源了另外 8 个病毒式传播的效果！（欢迎在评论区提出更多需求！）](https://v.redd.it/u571kznwf6oe1)** ([Score: 565, Comments: 41](https://reddit.com/r/ChatGPT/comments/1j99tfc/i_just_opensourced_8_more_viral_effects_request/))：**八种病毒式 AI 文本生成视频效果**已开源，并邀请社区在评论中请求更多效果。
  - **开源与易用性**：这些效果是开源的，允许任何拥有高性能电脑的人免费运行，或者在 **Runpod** 等平台上以约 **每小时 0.70 美元** 的价格租赁 GPU。**Generative-Explorer** 详细解释了如何结合 **ComfyUI** 和 **LoRA** 节点设置并使用 **Wan 2.1** 模型，并为初学者提供了教程链接。
  - **效果细节与社区参与**：帖子作者 **najsonepls** 强调了基于 **Wan2.1 14B I2V 480p model** 训练的效果所取得的病毒式成功，列出的效果包括：**挤压 (Squish)、粉碎 (Crush)、蛋糕化 (Cakeify)、充气 (Inflate)、放气 (Deflate)、360 度微波旋转、开枪射击和肌肉展示**。社区讨论了诸如“变老”等潜在的新效果，并对开源特性表示出浓厚兴趣，这使得进一步的定制和创新成为可能。
  - **担忧与行业影响**：用户推测大公司是否会将类似效果限制在付费墙后，但 **Generative-Explorer** 认为，通过使用少量视频训练 **LoRA**，开源替代方案可以被迅速开发出来。讨论还涉及了**充气和放气**等效果对特定利基内容领域（如 NSFW Tumblr 圈子）的影响。


**主题 4. 西班牙的 AI 内容标注指令：法律与社会影响**

- **[西班牙将对未标注 AI 生成内容的行为处以巨额罚款](https://www.reuters.com/technology/artificial-intelligence/spain-impose-massive-fines-not-labelling-ai-generated-content-2025-03-11/)** ([Score: 212, Comments: 22](https://reddit.com/r/ChatGPT/comments/1j9m5cc/spain_to_impose_massive_fines_for_not_labelling/))：**西班牙**正在引入一项指令，要求对 **AI 生成的内容**进行标注，违规者将面临**巨额罚款**。该法规旨在提高使用 AI 技术时的透明度和问责制。
  - **检测挑战**：人们对 AI 检测方法的有效性表示担忧，并提到了学校中现有的问题，即 AI 检测软件会产生**误报 (false positives)**。关于西班牙如何在不冤枉个人的情况下准确识别 AI 生成内容，也引发了疑问。
  - **怀疑与批评**：舆论对专注于 AI 生成内容标注的做法持怀疑态度，建议优先处理立法过程中的**腐败**和**优待**问题，而不是关注**交通罚单**和**学校作业**等琐碎事项。
  - **监管影响**：一些用户表示，该法规可能会导致西班牙境内 AI 使用量的减少，这既可能被视为获得**心理安宁**的积极举措，也可能被视为阻碍该国 AI 应用的负面后果。


**主题 5. ✨ 表情符号的象征意义：作为 AI 图标的兴起**

- **[星星 ✨ 什么时候成了 AI 生成的象征？它从何而来？](https://i.redd.it/2ki6l4ia1aoe1.jpeg)** ([Score: 200, Comments: 42](https://reddit.com/r/OpenAI/comments/1j9ml3m/when_did_the_stars_become_the_symbol_for_ai/))：该帖子询问了 **✨ 表情符号**作为 AI 生成内容符号的起源和普及过程，并指出它在新闻文章、社交媒体以及 Notepad 等应用程序中随处可见。配图使用了星星和圆点图形来唤起该表情符号与“闪烁”或“魔法”的关联，但缺乏文字解释。
  - **Jasper** 是早在 **2021** 年初就采用 **✨ 表情符号**来表示 AI 生成内容的先行者之一，早于 **Google**、**Microsoft** 和 **Adobe** 等大公司，后者在 **2022-2023** 年间开始使用它。到 **2023** 年中期，设计界开始讨论其作为 AI 非官方标准的地位；到 **2023** 年底，它已被主流媒体公认为通用的 AI 符号。
  - **✨ 表情符号**与魔法和自动纠错的概念相关联，让人联想起 Adobe 在 30 多年前使用的**魔棒 (magic wand)** 图标。这种与魔法和自动改进的关联，促使其被广泛采纳为 AI 生成内容的象征。
  - 多个资源探索了该表情符号的历史和采用情况，包括 [Wikipedia 条目](https://en.wikipedia.org/wiki/Sparkles_emoji?#Artificial_intelligence)、一段 [YouTube 视频](https://youtu.be/g-pG79LOtMw) 以及 **David Imel** 在 [Substack](https://davidimel.substack.com/p/how-ai-stole-the-sparkles-emoji) 上发表的文章，为它作为 AI 图标的演变提供了见解。

---

# AI Discord 摘要回顾

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1：Google 的多模态新作闪耀 AI 舞台**

- [**Gemma 3 凭借多语言精通成为焦点**](https://blog.google/technology/developers/gemma-3/)：Google 发布了 **Gemma 3**，这是一款参数量从 **1B 到 27B** 不等的多模态模型，拥有 **128K context window**，支持超过 **140 种语言**。社区对其在单个 GPU 或 TPU 上运行的潜力议论纷纷。
- [**Gemini 2.0 Flash 以词绘图**](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)：**Gemini 2.0 Flash** 现在支持原生图像生成，允许用户直接在模型内创建与上下文相关的图像。开发者可以通过 **Google AI Studio** 进行体验。
- [**Gemini Robotics 让 AI 走进现实——字面意义上的！**](https://youtu.be/4MvGnmmP3c0)：Google 在 [YouTube 视频](https://youtu.be/4MvGnmmP3c0)中展示了 **Gemini Robotics**，演示了先进的 vision-language-action 模型，使机器人能够与物理世界互动。

**主题 2：新型 AI 模型挑战巨头**

- [**OlympicCoder 在编程挑战中超越 Claude 3.7**](https://x.com/lvwerra/status/1899573087647281661)：紧凑的 **7B parameter** **OlympicCoder** 模型在奥林匹克级别的编程挑战中超越了 **Claude 3.7**，证明了在 AI 性能方面，规模并非一切。
- [**Reka Flash 3 在对话和代码领域加速前进**](https://openrouter.ai/rekaai/reka-flash-3:free)：**Reka** 发布了 **Flash 3**，这是一个 **21B parameter** 的模型，在对话、编程和 function calling 方面表现出色，具有 **32K context length**，并可[免费使用](https://openrouter.ai/rekaai/reka-flash-3:free)。
- [**Swallow 70B 在日语领域横扫竞争对手**](https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3)：**Llama 3.1 Swallow 70B** 是一款具备超快响应速度的**日语能力模型**，现已加入 **OpenRouter**，扩展了语言能力并提供极速响应。

**主题 3：AI 工具遭遇波折**

- **Codeium 出现协议错误，开发者大惊失色**：用户报告 **Codeium** 的 **VSCode extension** 出现 **protocol errors**，如“*invalid_argument: protocol error: incomplete envelope*”，导致代码补全功能陷入困境。
- **Cursor 更新后运行缓慢，用户纷纷退回旧版本**：在更新到 **version 0.46.11** 后，**Cursor IDE** 变得反应迟钝，促使用户建议下载 **version 0.47.1** 以恢复性能。
- **Perplexity Windows 应用的 Apple ID 登录出现故障**：**Perplexity AI** 用户在使用 Apple ID 登录时遇到 **500 Internal Server Error**，而使用 Google 账号的用户则运行顺畅。

**主题 4：AI 工具集成迸发创新火花**

- [**OpenAI Agents SDK 与 MCP 挂钩**](https://github.com/lastmile-ai/openai-agents-mcp)：**OpenAI Agents SDK** 现在支持 **Model Context Protocol (MCP)**，允许 Agent 无缝聚合来自 MCP server 的工具，以实现更强大的 AI 交互。
- [**Glama AI 公开所有可用工具详情**](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers)：[Glama AI 的新 API](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers) 列出了每个 server 的所有可用工具，以开放的 AI 能力目录令用户感到兴奋。
- [**LlamaIndex 跨入 MCP 集成行列**](https://twitter.com/llama_index/status/1899848532817035529)：**LlamaIndex** 与 **Model Context Protocol** 集成，通过接入任何兼容 MCP 的服务所提供的工具来增强其能力。

**主题 5：关于 LLM 行为的辩论升温**

- **“LLM 不会产生幻觉！”怀疑论者惊呼**：关于 **LLM** 是否会产生“幻觉”引发了激烈辩论，一些人认为既然它们不进行“思考”，就不会产生幻觉——这在 AI 社区引发了哲学层面的对决。
- [**LLM 通过面部记忆系统焕发新颜**](https://github.com/yaya-labs/LLM_Facial_Memory_System)：一个开源的 [**LLM Facial Memory System**](https://github.com/yaya-labs/LLM_Facial_Memory_System) 允许 **LLM** 根据用户的面部存储记忆和聊天记录，增加了一层新的个性化交互。
- **ChatGPT 的伦理提醒令寻求未过滤回复的用户感到恼火**：用户对 **ChatGPT** 在回复中频繁出现的伦理准则表示反感，希望有一个“*关闭 AI 保姆*”的选项来简化他们的工作流程。

---

# PART 1: 高层级 Discord 摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.7 遭遇过载**：用户报告 **Claude 3.7** 负载过高，在高峰时段会出现错误和卡顿，建议在夜间编写代码以避开这些问题，并分享了关于该话题的 [Cursor 论坛帖子链接](https://forum.cursor.com/t/claude-3-7-thinking-permanently-high-load/62928)。
   - *'diff algorithm stopped early'* 错误是一个频繁报告的问题。
- **Cursor 在 0.46 版本变慢**：用户观察到在更新到 **0.46.11** 版本后，Macbook 和 PC 上的 **Cursor** 变得非常迟钝，建议下载 **0.47.1** 版本。
   - 即使在 CPU 占用率较低的情况下也会出现性能下降，而项目规则的模式匹配问题已在后续版本中修复。
- **Manus AI 生成销售线索**：成员们讨论了使用 **Manus AI** 进行线索生成和构建 SaaS 落地页，强调了其获取电话号码的能力，据报道在花费 **$600** 后获得了 **30 个高质量线索**。
   - 成员们分享了一个[链接](https://manus.im/share/YIRZaLUfghVxGCN7dE6hbI?replay=1)演示 **Manus** 在单次对话中构建仪表盘，以及另一个关于其首次使用的[演示视频](https://x.com/mckaywrigley/status/1898756745545252866?s=46&t=CLGnxOi5OPp22iT8UYkr1A)。
- **OpenManus 尝试复制功能**：用户分享称 **OpenManus**（一个试图复制 **Manus AI** 的开源项目）展现出了潜力，并提供了 [GitHub 仓库链接](https://github.com/mannaandpoem/OpenManus)和展示其能力的 [YouTube 视频](https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf)。
   - 一些成员认为它目前还无法与 **Manus** 媲美。
- **Cline 的代码补全成本受到批评**：成员们辩论了 **Cline** 的价值，认为其相对于 **Cursor** 成本过高。
   - 虽然 **Cline** 提供“全上下文窗口（full context window）”，但一些用户认为 **Cursor** 的缓存系统允许在单个对话中扩展上下文，并提供 Web 搜索和文档等功能。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 支持 Gemma 3**：**LM Studio 0.3.13** 现在支持 **Google** 的 **Gemma 3** 系列，但用户报告 **Gemma 3 模型** 的运行速度明显变慢，与同类模型相比慢了多达 **10 倍**。
   - 团队还在努力解决一些问题，例如用户难以完全禁用 **RAG** 以及 Linux 安装程序的问题。
- **关于移除 RAG 的激烈讨论**：用户寻求在 **LM Studio** 中完全关闭 **RAG**，以便将完整的附件注入上下文中，但目前没有禁用它的 UI 选项。
   - 作为权宜之计，用户手动复制粘贴文档，面临着将 PDF 转换为 Markdown 的麻烦。
- **AMD GPU 用户的热点温度困扰**：用户报告其 **7900 XTX** 的热点温度达到 **110°C**，引发了关于 **RMA** 资格的讨论，并担心 AIB 厂商在散热上偷工减料；有一份报告称 **AMD 拒绝** 了此类 **RMA** 请求。
   - 有人指出根本原因可能是真空腔（vapour chambers）批次不良，内部水量不足，而 **PowerColor** 已对 **RMA** 请求表示**同意**。
- **矿卡复活作为推理主力**：成员们正在讨论复活具有 **288 tensor cores** 的 **CMP-40HX** 矿卡用于 **AI inference**。
   - 由于需要对 Nvidia 驱动程序打补丁以启用 3D 加速支持，用户的兴趣受到了一定影响。
- **PTM7950 相变材料备受关注**：成员们考虑使用 **PTM7950**（相变材料）代替硅脂，以防止泵出（pump-out）问题并保持稳定的温度。
   - 在第一次热循环后，多余的材料会泵出并在芯片周围形成一层厚且粘稠的层，从而防止进一步的泵出。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 发布推理 API**：Nous Research 发布了其 **Inference API**，包含 **Hermes 3 Llama 70B** 和 **DeepHermes 3 8B Preview**，并为新账户提供 **$5.00** 的免费额度。
   - [Nous Portal](https://portal.nousresearch.com/login) 已实施候补名单系统，按照**先到先得**的原则授予访问权限。
- **LLM 现在可以识别面部并记住对话**：一位成员开源了一个 [**LLM Facial Memory System**](https://github.com/yaya-labs/LLM_Facial_Memory_System)，让 **LLM** 能够根据你的面部存储记忆和聊天记录。
   - 成员们讨论了推理 API，包括由于对 **API key 安全性**的担忧，考虑**预充值额度**的可能性。
- **利用开源代码构建图推理系统**：成员们讨论了目前已有足够的公开信息利用开源代码构建**图推理系统 (graph reasoning system)**，虽然可能不如 Forge，但新 API 提供了 **50 欧元的推理额度**。
   - 提到 [**Kuzu**](https://kuzu.io/) 非常出色，而对于图数据库，推荐使用 [**networkx + python**](https://networkx.org/)。
- **Audio-Flamingo-2 调性检测失败**：一位用户在 HuggingFace 上测试了 [Nvidia 的 Audio-Flamingo-2](https://huggingface.co/spaces/nvidia/audio-flamingo-2)，用于检测歌曲的调性 (key) 和节拍 (tempo)，但结果**好坏参半**，甚至无法识别简单流行歌曲的调性。
   - 例如，当被要求识别 Lorde 的歌曲《Royals》的调性时，Audio-Flamingo-2 错误地猜为 *F# Minor*，节拍为 *150 BPM*，引发了社区的哄笑。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 获得 GGUF 支持**：**Gemma 3** 的所有 **GGUF**、**4-bit** 和 **16-bit** 版本已上传至 [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b)。
   - 这些量化版本旨在运行于使用 *llama.cpp* 的程序中，如 **LM Studio** 和 **GPT4All**。
- **Transformers 故障阻碍微调**：**Transformers** 中的一个破坏性 Bug 正在阻止 **Gemma 3** 的微调。根据 [Unsloth AI](https://unsloth.ai/blog/gemma3) 的博客更新，**HF** 正在积极修复。
   - 建议用户等待官方的 **Unsloth** notebook，以确保在 Bug 解决后实现兼容性。
- **GRPO 泛化效果显著！**：讨论涵盖了 **RLHF** 方法（如 **PPO**、**DPO**、**GRPO** 和 **RLOO**）的细微差别，一位成员指出 *GRPO 的泛化能力更好*，并且是 [PPO 的直接替代方案](https://arxiv.org/abs/2405.10422)。
   - **RLOO** 是 **PPO** 的更新版本，其优势基于群体响应的归一化奖励分数，由 **Cohere AI** 开发。
- **HackXelerator 登陆伦敦、巴黎、柏林**：一位成员宣布了由 **Mistral**、**HF** 等支持的**伦敦、巴黎、柏林多模态创意 AI HackXelerator**。
   - 由 **Mistral AI**、**Hugging Face**、**AMD** 等支持的多模态创意 AI **HackXelerator** 将在伦敦、巴黎和柏林举行，重点关注音乐、艺术、电影、时尚和游戏，将于 **2025 年 4 月 5 日**开始 ([lu.ma/w3mv1c6o](https://lu.ma/w3mv1c6o))。
- **为 Ollama 调整 Temperature 参数**：许多人在使用 **1.0** 的 temp 设置时遇到问题，建议在 [Ollama](https://ollama.com/) 中以 **0.1** 运行。
   - 鼓励进行测试，看看在 **llama.cpp** 和其他程序中是否表现更好。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ANUS AI Agent 引起热议**：GitHub 仓库 [nikmcfly/ANUS](https://github.com/nikmcfly/ANUS) 因其不幸的命名引发了幽默讨论，一位成员开玩笑地建议使用 **TWAT** (*Think, Wait, Act, Talk pipeline*) 作为替代缩写。
   - 另一位成员提议将 *Prostate* 作为政府 AI Agent 的名称，进一步延续了这一滑稽的交流。
- **Apple ID 登录触发服务器错误**：用户报告在尝试为 Perplexity 的新 Windows 应用进行 Apple ID 登录时遇到 **500 Internal Server Error**。
   - 这一问题似乎仅限于 Apple ID，因为部分用户的 Google 登录功能正常。
- **模型选择器忽隐忽现**：在新的网页端更新中，模型选择器最初消失了，导致用户因无法选择 R1 等特定模型而感到沮丧。
   - 模型选择器随后重新出现，用户建议将模式设置为 "pro" 或使用 "complexity extension" 来解决选择问题。
- **Perplexity 搞砸了代码清理**：一位用户分享了他们长达 6 小时的惨痛经历，详细描述了 Perplexity 如何未能正确清理一个 875 行的代码文件，导致代码块和链接损坏。
   - 尽管受限于消息长度限制，Perplexity 最终返回了原始且未修改的代码。
- **MCP Server 连接器发布**：API 团队宣布发布其 **Model Context Protocol (MCP) server**，鼓励社区通过 [GitHub](https://github.com/ppl-ai/modelcontextprotocol) 提供反馈和贡献。
   - **MCP server** 作为 **Perplexity API** 的连接器，支持在 **MCP 生态系统**中直接进行网页搜索。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemma 3 登场！**：**Google** 发布了 **Gemma 3**，这是一款多模态模型，参数范围从 **1B** 到 **27B**，拥有 **128K** 上下文窗口，并根据 [Google 博客](https://blog.google/technology/developers/gemma-3/) 兼容 **140+** 种语言。
   - 这些模型旨在轻量且高效，目标是在单个 GPU 或 TPU 上实现最佳性能。
- **OlympicCoder 横扫编程任务！**：根据 [推文](https://x.com/lvwerra/status/1899573087647281661) 和 [Unsloth.ai 博客文章](https://unsloth.ai/blog/gemma3)，**OlympicCoder** 模型（一个紧凑的 **7B** 参数模型）在奥林匹克级别的编程挑战中超越了 **Claude 3.7**。
   - 这一壮举强调了高效模型在专业编程领域的潜力。
- **Fast Apply 模型加速编辑！**：受一篇已删除的 [Cursor 博客文章](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply) 启发，**Fast Apply** 模型是一个经过微调的 **Qwen2.5 Coder Model**，用于快速代码更新，正如在 [Reddit](https://old.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/) 上讨论的那样。
   - 该模型解决了在 **Aider** 等工具中更快应用搜索/替换块的需求，增强了代码编辑工作流。
- **Aider 的 Repo Map 被弃用！**：用户正选择禁用 **Aider** 的 repo map，转而手动添加文件以更好地控制上下文，尽管 Aider 的使用技巧建议显式添加文件是最有效的方法，详见 [官方使用技巧](https://aider.chat/docs/usage/tips.html)。
   - 目的是防止 **LLM** 被过多的无关代码分散注意力。
- **LLM 极大加速学习**：成员们分享说 **LLM** 大大加速了学习 **Python** 和 **Go** 等语言的过程，并提到生产力的提升让他们能够承担以前认为不合理的项目。
   - 一位成员指出，这不仅仅是工作变快了，而是让那些原本不可能实现的项目变得可行，并将 **AI** 描述为 *寒武纪大爆发级别* 的事件。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 在深度研究方面击败 OpenAI**：一位成员认为 **Perplexity** 在深度研究方面优于 **OpenAI** 和 **SuperGrok**，特别是在处理上传文档和互联网搜索时，同时也考虑到了用户的预算问题。
   - 用户在寻求如何从 **ChatGPT**、**Perplexity** 和 **Grok** 中做出选择的建议，最终因其研究能力而推荐了 **Perplexity**。
- **Ollama 编排最佳模型部署**：当被问及部署 AI Transformer 模型的最佳语言时，一位成员建议将 **Ollama** 作为服务使用，特别是如果追求更快的推理速度/性能。
   - 该用户一直在使用 **Python** 进行原型设计，并探索 **C#** 是否能提供更好的性能，从而得到了 **Ollama** 的推荐。
- **怀疑论者称 LLMs 不会产生幻觉**：一位成员认为“幻觉 (hallucination)”一词被错误地应用于 LLM，因为 **LLM 不具备思考能力**，只是简单地根据概率生成词序。
   - 另一位成员补充说，有时它会错误地切换模型，并指向一张[附图](https://cdn.discordapp.com/attachments/998381918976479273/1349373748272435350/image.png?ex=67d2ddbb&is=67d18c3b&hm=beab6d0aacaeeb5fd464a64eba9a21e18882410f03da9e5c4566a0be8b89d5&)。
- **Gemini 的图像生成能力令人印象深刻**：成员们对 **Google** 发布的 **Gemini** 原生图像功能赞不绝口，强调其免费可用性，以及能够“看到它生成的图像”以便更好地重新生成。
   - 这一功能允许通过文本改进图像的重新生成，并在 [Gemini Robotics 公告](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)中进行了展示。
- **伦理化的 ChatGPT 引起反感**：用户对 **ChatGPT** 频繁的伦理提醒表示恼火，认为这些提醒是不必要的、多余的，并且干扰了他们的工作流。
   - 一位用户表示，他们希望有一个选项可以禁用这些伦理准则。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 课程解释视觉语言模型**：[Hugging Face 计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro)包含一个介绍 **Vision Language Models (VLMs)** 的章节，涵盖了多模态学习策略、常用数据集、下游任务和评估。
   - 该课程强调了 VLMs 如何协调来自不同感官的见解，使 AI 能够更全面地理解世界并与之互动，统一了来自不同感官输入的见解。
- **为实现顶级吞吐量而进行的 TensorFlow 调整**：一位成员分享了一篇关于 **使用 TensorFlow 进行 GPU 配置** 的[博客文章](https://medium.com/@samiratra95/tensorflow-experimental-gpu-configuration-02618635bdad)，涵盖了实验性函数、逻辑设备和物理设备，使用的是 **TensorFlow 2.16.1**。
   - 该成员探索了 GPU 配置的技术和方法，借鉴了使用 **NVIDIA GeForce RTX 3050 Laptop GPU** 处理 **280 万张图像数据集** 的经验，利用 [TensorFlow API Python Config](https://www.tensorflow.org/api_docs/python/tf/config) 来提高执行速度。
- **Modal 模块模型可用**：一位成员分享了一个 [YouTube 教程](https://youtu.be/q-8KXOczRBY)，关于如何在 **Modal** 上免费部署 **Wan2.1 Image to Video 模型**，涵盖了无缝的 Modal 安装和 **Python 脚本**。
   - 提供了关于如何使用 **Modelfile** 来运行这个 **GGUF 格式** 的 **Gemma 2b** 微调模型的说明。
- **本地模型解放语言学习**：一位用户分享了在 `smolagents` 中通过 `litellm` 和 `ollama` 使用本地模型的代码片段，使用 `LiteLLMModel` 并指定 `pip install smolagents[litellm]`，然后调用 `localModel = LiteLLMModel(model_id="ollama_chat/qwen2.5:14b", api_key="ollama")`。
   - 用户报告称，使用默认的 `hfApiModel` 在仅调用几次 Qwen 推理 API 后就会产生 **需要付费 (payment required)** 的错误，但指定本地模型可以绕过这一限制。
- **Agent 架构的烦恼等待解决**：用户们正热切期待 Unit 2.3 的发布，该单元涵盖了 **LangGraph**，原定于 3 月 11 日发布。
   - 一位用户指出，用调用结果覆盖 `agent_name` 变量会导致 Agent 变得无法调用，从而引发了关于预防策略的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemma 3 引入多模态能力**：Google 在 OpenRouter 上发布了 **Gemma 3**，这是一个支持视觉-语言输入和文本输出的多模态模型，具有 **128k tokens** 的上下文窗口，并支持超过 **140 种语言**。
   - 它具有增强的数学、推理和对话能力，包括结构化输出和 Function Calling，可[免费使用](https://openrouter.ai/google/gemma-3-27b-it:free)，是 [Gemma 2](https://openrouter.ai/google/gemma-2-27b-it) 的继任者。
- **Reka Flash 3 在对话和编程方面表现出色**：Reka 发布了 **Flash 3**，这是一个拥有 210 亿参数的语言模型，在通用对话、编程和 Function Calling 方面表现优异，具有通过强化学习 (**RLOO**) 优化的 **32K 上下文长度**。
   - 该模型的权重采用 [Apache 2.0 许可证](https://www.apache.org/licenses/LICENSE-2.0)，可[免费使用](https://openrouter.ai/rekaai/reka-flash-3:free)，且主要是一个英文模型。
- **Swallow 70B 增加日语流利度**：一款名为 [Llama 3.1 Swallow 70B](https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3) 的新型超快**日语能力模型**加入 OpenRouter，扩展了平台的语言能力。
   - 这补充了 **Reka Flash 3** 和 **Google Gemma 3** 的发布，增强了 OpenRouter 上可用语言处理工具的多样性。
- **Gemini 2 Flash 支持原生图像生成**：Google 的 **Gemini 2.0 Flash** 现在支持原生图像输出，供 **Google AI Studio** 支持的所有地区的开发者进行实验，可通过 Gemini API 和实验版本 ([gemini-2.0-flash-exp](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash-exp)) 访问。
   - 正如 [Google Developers Blog 文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)中所宣布的，这允许从文本和图像输入创建图像，保持角色一致性并增强叙事能力。
- **OpenRouter 的 Chutes 提供商保持免费**：**Chutes** 提供商在准备服务和扩大规模期间，由于尚未完全实现支付系统，目前对 OpenRouter 用户保持免费。
   - 虽然数据没有明确用于训练，但由于其去中心化的性质，OpenRouter 无法保证计算主机不会使用这些数据。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Distill 社区启动每月聚会**：在取得成功反响后，**Distill** 社区正在启动每月聚会，下一次定于 **美国东部时间 3 月 14 日上午 11:30 至下午 1:00**。
   - 详情可以在 [Exploring Explainables Reading Group 文档](https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym)中找到。
- **TTT 增强模型引导 (Priming)**：成员们讨论了 **TTT** 如何通过执行单次梯度下降过程，加速为给定提示词引导模型的过程，使模型状态更具接收性。
   - 模型优化了序列压缩以产生有用的表示，从而通过旨在学习和执行每个 token 的多次梯度下降，增强了 **ICL** 和 **CoT** 能力。
- **Decoder-Only 架构拥抱动态计算**：一项小提议建议将 Decoder 端用于动态计算，通过重新将 Encoder-Decoder 的概念引入 Decoder-Only 架构，利用 **类 TTT 层** 扩展序列长度以进行内部“思考”。
   - 一个挑战是确定额外的采样步数，但测量 **TTT 更新损失的增量 (delta)** 并在低于中值时停止可能会有所帮助。
- **AIME24 实现出现，仍需测试**：基于 **MATH** 实现的 **AIME24** 实现已出现在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/lm_eval/tasks/aime24) 中。
   - 提交者承认，由于缺乏关于人们运行 **AIME24** 时具体执行内容的文档，他们*还没有时间对其进行测试*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Funnel Shift 在 H100 上的性能表现**：工程师们惊讶地发现，**funnel shift** 似乎比 **H100** 上的等效操作更快，这可能是由于使用了拥堵较少的管道。
   - 尽管尝试了 `prmt` 指令，但一致使用断言（predicated）**funnel shift** 表现更好，最终生成了 **4 个 `shf.r.u32`**、**3 个 `shf.r.w.u32`** 和 **7 个 `lop3.lut` SASS 指令**。
- **TensorFlow 的 OpenCL 口水战**：一场讨论由 2015 年关于 [TensorFlow 中 OpenCL 支持](https://github.com/tensorflow/tensorflow/issues/22) 的 *有趣口水战* 引发。
   - 这场辩论突显了早期对 **CUDA** 的优先排序以及在集成 **OpenCL** 支持时遇到的困难。
- **Turing 架构获得 FlashAttention 支持**：分享了一个针对 [Turing 架构的 FlashAttention 前向传递实现](https://github.com/ssiu/flash-attention-turing)，支持 `head_dim = 128`、原生 attention，且 `seq_len` 可被 128 整除。
   - 在 **T4** 上测试时，该实现与 Pytorch 的 `F.scaled_dot_product_attention` 相比显示出 **2 倍的加速**。
- **Modal Runners 征服向量加法**：在 **T4** GPU 上使用 **Modal runners** 提交的 `vectoradd` 排行榜测试成功！
   - ID 为 **1946** 和 **1947** 的提交证明了 Modal runners 在 GPU 加速计算中的可靠性。
- **H100 内存分配失误**：一位成员询问，为什么在 ThunderKittens 的 [h100.cu](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/h100/h100.cu#L71) 中修改内存分配以直接为 `o_smem` 分配内存时，会导致 *illegal memory access was encountered*（遇到非法内存访问）错误。
   - 他们正试图理解在指定的 **H100 GPU kernel** 中导致此错误的原因。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemma 3 夺得第二名**：**Gemma-3-27b** 模型在创意写作中获得第二名，可能成为创意写作和 RP（角色扮演）微调者的首选，详见[此推文](https://x.com/sam_paech/status/1899772582808969653)。
   - 像 **Gemma 3** 这样的开放权重模型也在压缩 API 平台的利润空间，并因 **隐私/数据** 考量而越来越多地被采用。
- **Gemini 2.0 Flash 带来图像生成功能**：**Gemini 2.0 Flash** 现在具备原生图像生成功能，并针对对话迭代进行了优化，允许用户创建与上下文相关的图像并在图像中生成长文本，如[此博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)所述。
   - **DeepMind** 还推出了 **Gemini Robotics**，这是一个基于 **Gemini 2.0** 的机器人模型，旨在通过多模态推理解决复杂问题。
- **AlphaXiv 创建 ArXiv 论文概览**：根据[此推文](https://fxtwitter.com/askalphaxiv/status/1899833509033976194)，**AlphaXiv** 使用 **Mistral OCR** 配合 **Claude 3.7** 为 arXiv 论文生成博客风格的概览，只需点击一下即可提供论文中的图表、关键见解和清晰解释。
   - 它能生成带有图表、核心见解和清晰解释的精美研究博客。
- **ML 模型陷入版权风波**：正在进行的法庭案件正在审查 **在受版权保护的数据上训练生成式机器学习模型** 是否构成版权侵权，详见 [Nicholas Carlini 的博客文章](https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html)。
   - 律师们正在引用 Nicholas Carlini 关于模型输出逐字训练示例（[文本](https://arxiv.org/abs/2012.07805)和[图像](https://arxiv.org/abs/2301.13188)）的论文，来争论模型是否违反了版权法。
- **深度学习如同耕作？**：一位成员分享了 Arjun Srivastava 撰写的题为《[论深度学习与耕作](https://open.substack.com/pub/arjunsriva/p/on-deep-learning-and-farming?r=68gy5&utm_medium=ios)》的文章链接，该文探讨了将概念从一个领域映射到另一个领域。
   - 作者将 **工程（Engineering）**（组件被刻意组装）与 **培育（Cultivation）**（无法直接构建）进行了对比。*培育* 就像耕作，而 *工程* 就像打造一张桌子。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **更强的大脑，更好的基准测试**：一位用户询问了 **ChatGPT premium** 与 **GPT4All** 的 **LLMs** 之间的性能差距，另一位用户将其归因于模型参数量更大。
   - 讨论建议在硬件条件允许的情况下，从 **Hugging Face** 下载更大的模型。
- **服务器解决方案：选择 Ollama 还是 GPT4All？**：一位用户质疑 **GPT4All** 是否适合作为服务器使用，该服务器需要管理多个模型、快速加载/卸载、针对定期更新的文件进行 **RAG**，以及提供日期/时间/天气的 **APIs**。
   - 该用户提到了 **Ollama** 的一些问题，并在中低算力的情况下寻求关于其可行性的建议。
- **Deepseek 详情：14B 是首选**：在寻找 **ChatGPT premium** 替代方案的咨询中，有人建议使用 **Deepseek 14B**，前提是拥有 64GB RAM。
   - 建议先从 **Deepseek 7B** 或 **Llama 8B** 等较小的模型开始，根据系统性能再进行扩展。
- **上下文是关键：4k 还可以**：讨论强调了大上下文窗口（超过 **4k tokens**）的重要性，以便在提示词中容纳更多信息（如文档）。
   - 随后一位用户询问他们发布的截图是否属于这些模型之一，并询问其上下文窗口能力。
- **Gemma 代际差距：GPT4All 的小故障**：一位用户建议使用微型模型测试 **GPT4All**，以评估加载、卸载和 **RAG**（配合 LocalDocs）的工作流，并指出 GUI 目前不支持同时运行多个模型。
   - 他们指出 **Gemma 3** 目前与 **GPT4All** 不兼容，需要更高版本的 llama.cpp，并附上了错误截图。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama API 导出工具数据**：一个新的 [Glama AI API](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers) 端点现在列出了所有可用工具，比 Pulse 提供的每个服务器的数据更多。
   - 用户对这些免费提供的信息感到兴奋。
- **MCP 日志详情：服务器视角**：服务器根据 [Model Context Protocol (MCP) 规范](https://spec.modelcontextprotocol.io/specification/2024-11-05/server/utilities/logging/) 发送日志消息，具体表现为声明 `logging` 能力，并发出带有严重级别和 JSON 可序列化数据的日志消息。
   - 这允许通过 **MCP** 控制，实现从服务器到客户端的结构化日志记录。
- **Wolfram 助力 Claude 渲染图像**：一位成员指向了一个 [wolfram 服务器示例](https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35)，该示例获取渲染的图表，通过对数据进行 base64 编码并设置 mime 类型来返回图像。
   - 有人指出 **Claude** 在工具调用窗口之外进行渲染存在局限性。
- **NPM 包位置揭晓**：NPM 包存储在 `%LOCALAPPDATA%` 中，具体位于 `C:\Users\YourUsername\AppData\Local\npm-cache`。
   - 该位置包含 NPM 包和源代码。
- **OpenAI Agents SDK 支持 MCP**：**OpenAI Agents SDK** 已添加 **MCP** 支持，可在 [GitHub 上的 fork 版本](https://github.com/lastmile-ai/openai-agents-mcp) 中获取，并在 pypi 上作为 *openai-agents-mcp* 包发布，允许 **Agents** 聚合来自 **MCP** 服务器的工具。
   - 通过设置 `mcp_servers` 属性，可以通过统一语法无缝集成 **MCP 服务器**、本地工具和 **OpenAI 托管的工具**。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 扩展遭遇协议错误**：用户报告 VSCode 扩展中出现 **协议错误**，例如 *"invalid_argument: protocol error: incomplete envelope: read tcp... forcibly closed by the remote host"*，导致 Codeium 页脚变红。
   - 这一问题特别影响了 **英国** 和 **挪威** 使用 **Hyperoptic** 和 **Telenor** 等运营商的用户。
- **Neovim 支持难以跟上进度**：一位用户批评了 **Neovim 支持** 的现状，提到了补全错误（error 500），并担心其落后于 **Windsurf**。
   - 针对批评，一名团队成员回复称团队“正在处理中”。
- **测试修复部署结果不一**：团队部署了一个测试修复程序，虽然一些用户报告错误减少，但其他用户仍面临问题，扩展要么“关闭”，要么保持红色状态。
   - 这些不一的结果促使团队进行进一步调查。
- **欧盟用户发现 VPN 解决方法**：团队确认 **欧盟** 用户在自动补全时遇到了 *"unexpected EOF"* 等问题，并且无法在聊天中链接文件。
   - 作为临时解决方法，通过 **VPN** 连接到 **洛杉矶** 解决了受影响用户的问题。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Gemini Robotics 问世**：Google 发布了 [一段 YouTube 视频](https://youtu.be/4MvGnmmP3c0) 展示 **Gemini Robotics**，将 **Gemini 2.0** 作为其最先进的视觉语言动作模型 (vision language action model) 带入物理世界。
   - 该模型使机器人能够与物理世界互动，具有增强的物理交互能力。
- **Gemma 3 发布，支持 128k 上下文窗口**：**Gemma 3** 正式发布，具备多模态能力和 **128k 上下文窗口**（1B 模型除外），满足了用户期待。
   - 虽然此次发布备受关注，但一位用户评论说它“也就那样” (*twas aight*)。
- **Sakana AI 的论文通过同行评审**：由 **Sakana AI** 生成的一篇 [论文](https://sakana.ai/ai-scientist-first-publication/) 已通过 **ICLR workshop** 的同行评审。
   - 一位用户质疑评审过程的严谨性，暗示该 workshop 可能对作者“比较慷慨”。
- **麦克斯韦妖限制 AI 速度**：一位成员分享道，计算机可以通过同时向前和向后运行，以任意低的能量进行计算，但速度限制取决于运行答案的速度和确定性，并引用了 [这段 YouTube 视频](https://www.youtube.com/watch?v=eS0JXViv0cU)。
   - 他们还链接了 [另一段关于“逆转熵”的视频](https://www.youtube.com/watch?v=KR23aMjIHIY)，将计算限制与基础物理学联系起来。
- **欢迎自适应元学习项目**：一位成员正在寻找玩具项目来测试 **Meta-Transform** 和 **Adaptive Meta-Learning**，从使用 Gymnasium 的小步骤开始。
   - 他们还链接了一个用于 **Adaptive Meta-Learning (AML)** 的 [GitHub 仓库](https://github.com/EAzari/AML)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mastra 框架旨在吸引百万 AI 开发者**：根据 [其博客文章](https://mastra.ai/blog/the-next-million-ai-developers)，前 Gatsby/Netlify 的构建者宣布了 **Mastra**，这是一个全新的 **Typescript AI 框架**，旨在让玩具项目易于上手，同时对生产环境可靠。
   - 该框架面向前端、全栈和后端开发者，创建者旨在提供一个比现有框架更可靠、更简单的替代方案，并鼓励社区为 [其 GitHub 项目](https://github.com/mastra-ai/mastra) 做出贡献。
- **Cursor 的嵌入模型声称达到 SOTA**：根据 [一条推文](https://x.com/amanrsanger/status/1899659103473123777?s=46)，**Cursor** 训练了一个专注于语义搜索的 **SOTA 嵌入模型**，据报道其表现超过了竞争对手的开箱即用嵌入和重排序器 (rerankers)。
   - 邀请用户在使用 Agent 时“感受性能差异”。
- **Gemini 2.0 Flash 生成原生图像**：**Google** 正在 **Gemini 2.0 Flash** 中发布原生图像生成功能，供开发者在支持的地区通过 **Google AI Studio** 进行实验，详情见 [博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)。
   - 开发者可以在 Google AI Studio 和 Gemini API 中使用 **Gemini 2.0 Flash (gemini-2.0-flash-exp)** 的实验版本测试此功能，结合多模态输入、增强推理和自然语言理解来创建图像，如 [一条推文](https://x.com/19kaushiks/status/1899856652666568732?s=46) 所强调。
- **Jina AI 深入探讨 DeepSearch 细节**：**Jina AI** 分享了一篇 [博客文章](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)，概述了 **DeepSearch/DeepResearch** 的实际实现，重点是用于片段选择的延迟分块嵌入 (late-chunking embeddings) 以及在抓取前对 URL 进行优先排序的重排序器 (rerankers)。
   - 文章建议通过“阅读-搜索-推理”循环将重点从 **QPS 转向深度**，以改进答案发现。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **研究调查用户移动端习惯**：Google 正在招募 **NotebookLM 用户**进行 60 分钟的访谈，以讨论他们的**移动端使用习惯**并对新概念提供反馈。参与者将获得 **75 美元的感谢礼品**，感兴趣的参与者需填写筛选表单（[链接](https://forms.gle/pbPDU2Dh3rEL5HLC9)）以确认资格。
   - Google 将于 **2025 年 4 月 2 日和 3 日**进行一项**可用性研究**，以收集对开发中产品的反馈。参与者将获得**等值 75 美元的当地货币**作为报酬，要求具备**高速互联网连接**、**活跃的 Gmail 账号**以及**配备摄像头、扬声器和麦克风的电脑**。
- **NoteBookLM 作为内部 FAQ**：一位成员正考虑将 **NoteBookLM Plus** 用作内部 FAQ，并希望调查未解决问题的具体内容。
   - 他们正在寻求建议，了解如何查看用户在聊天中输入但未得到解决的问题。
- **NLM+ 生成 API 脚本！**：一位成员发现 **NLM+** 在利用 **API 指令**和示例程序生成脚本方面表现出惊人的能力。
   - 他们指出，作为非编程人员，通过引用 Notebook 中的材料，获取修改建议变得更加容易。
- **RAG 与全上下文窗口（Full Context Window）的对决**：一位用户质疑，对于大型数据库，使用带有向量搜索和较小上下文窗口的 **RAG** 是否优于使用具有全上下文窗口的 **Gemini Pro**。
   - 他们对 **RAG** 中使用的上下文窗口大小感到好奇，并寻求关于如何通过使用 **Gemini Pro** 实现“导师型 AI”任务的建议。
- **行内引用得以保留，太棒了！**：用户现在可以将聊天回复**保存为笔记，并保留原始形式的行内引用（inline citations）**。
   - 这一增强功能允许用户追溯原始素材，解决了高级用户长期以来的需求；此外，一位用户请求能够将行内引用复制并粘贴到文档中，同时保留链接。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **MPS 设备故障阻碍进展**：在最近的一次提交（[GitHub commit](https://github.com/pytorch/torchtune/commit/5cb4d54c779fd282dbfd2e1a50d2cb0828468bd2#diff-6cca0f357ea6c4e23906aec0c380c9d21887950f3371c83aa5acb40a83d61066R169)）后，出现了一个与缺失 `torch.mps` 属性相关的 `AttributeError`，这可能会禁用 **MPS** 支持。
   - 通过 [PR #2486](https://github.com/pytorch/torchtune/pull/2486) 提出的修复方案导致随后在 **MPS** 上运行时出现 **torchvision** 错误。
- **Gemma 3 取得进展**：一位成员指出了 **Gemma 3** 模型的变化，并附上了一张来自 Discord CDN 的[截图](https://cdn.discordapp.com/attachments/1216353675744641096/1349410043111407688/image.png?ex=67d2ff89&is=67d1ae09&hm=3094da9013c91c94a715cc42a41ebde502c1bfe9e64001c598a651f2e4dcaad3&)，详细说明了这些变化。
   - 这些变化的性质和影响未在进一步讨论中展开。
- **关于 Pan & Scan 的思考**：讨论了 **Gemma3** 论文中用于增强推理的 *Pan & Scan* 技术在 **torchtune** 中实现的必要性。
   - 一位成员认为这并非至关重要，建议使用带有 **HF ckpt** 的 **vLLM** 可以获得更好的性能，并参考了[这个 pull request](https://github.com/vllm-project/vllm/pull/14660)。
- **vLLM 配合 HF ckpt 表现出色**：为了提升 **Gemma3** 的性能，可以使用带有 **vLLM** 的 HF checkpoint。
   - 这通过[这个 pull request](https://github.com/vllm-project/vllm/pull/14660) 得以实现。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 连接至 Model Context Protocol**：**LlamaIndex** 现在已与 **Model Context Protocol (MCP)** 集成，该协议简化了工具的发现和利用，如[这条推文](https://twitter.com/llama_index/status/1899848532817035529)所述。
   - **Model Context Protocol** 的集成允许 **LlamaIndex** 使用任何兼容 **MCP** 的服务所提供的工具，增强了其功能。
- **LlamaExtract 在本地保护敏感数据**：**LlamaExtract** 现在为整个 **Llama-Cloud 平台**提供本地部署/BYOC（Bring Your Own Cloud）方案，以解决企业对敏感数据的担忧。
   - 然而，一位成员指出，这些部署的成本通常比使用 SaaS 解决方案*高得多*。
- **LlamaIndex 关注 Response API**：一位用户询问了对新 **Response API** 的支持情况，认为它有可能通过用户选择加入的搜索工具来丰富结果。
   - 一位成员给出了肯定答复，表示他们*正尝试在今天完成这项工作*。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **测验截止日期推迟至 5 月**：根据最新公告，所有**测验截止日期**都安排在 **5 月**。
   - 用户被指示查看有关 **Lecture 6** 的最新邮件以获取更多详情。
- **学习者想要 Lab 和研究机会**：一名成员询问了针对 **MOOC 学习者**的 **Labs** 计划以及**研究机会**。
   - 目前没有更多可用信息。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 多语言定价信息缺失？**：一名成员询问了 **Cohere multilingual embed model** 的定价，并指出在文档中很难找到该信息。
   - 讨论中未分享有关定价的具体细节或链接。
- **OpenAI 的 Responses API 简化了交互**：**OpenAI** 发布了其 **Responses API** 以及 **Agents SDK**，强调简单性和表现力，文档见[此处](https://platform.openai.com/docs/guides/responses-vs-chat-completions)。
   - 该 API 专为多工具、多轮对话和多模态设计，解决了用户在使用当前 API 时遇到的问题，[OpenAI cookbook](https://cookbook.openai.com/examples/responses_api/responses_example) 中提供了示例。
- **Cohere 与 OpenAI API 的兼容性存疑**：一名成员询问了 **Cohere** 与 **OpenAI** 新发布的 **Responses API** 兼容的可能性。
   - 新 API 旨在作为多轮交互、托管工具和粒度上下文控制的解决方案。
- **Chat API Seed 参数出现问题**：一位用户注意到 **Chat API** 似乎忽略了 `seed` 参数，导致即使使用相同的输入和 seed 值，输出也各不相同。
   - 多位用户报告在使用具有相同 `seed` 值的 **Chat API** 时输出不一致，这表明可复现性可能存在问题。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 缓存机制**：一名成员询问了 **DSPy 中的缓存工作原理**以及缓存行为是否可修改。
   - 另一名成员指向了一个正在开发中的可插拔 **Cache 模块**的 [Pull Request](https://github.com/stanfordnlp/dspy/pull/1922)，表明未来将具备灵活性。
- **可插拔缓存模块正在开发中**：该 [Pull Request](https://github.com/stanfordnlp/dspy/pull/1922) 引入了一个统一的**缓存接口**，具有两个缓存级别：内存中的 **LRU cache** 和 **fanout**（磁盘上）。
   - 此项开发旨在为 **DSPy** 提供更通用且高效的缓存解决方案。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Max Spawn 更新**：一名成员分享了一个 [GitHub Pull Request](https://github.com/modular/max/pull/3998)，希望它最终能像他们的项目一样，增加从可执行文件生成和管理进程的功能。
   - 然而，首先必须合并 *foundations PR*，然后还需要解决一些 **Linux exec 的问题**。
- **Linux Exec 阻碍 Modular Max 更新**：新功能的发布目前处于停滞状态，正在处理围绕 **Linux exec** 尚未解决的问题，同时等待 *foundations PR* 的批准。
   - 尽管存在障碍，开发者仍对近期发布表示乐观，并承诺向订阅者更新 PR 的进展。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **发现追踪评估工具的中心枢纽**：一名成员询问是否有中心位置可以追踪 **Berkeley Function Calling Leaderboard** 背景下使用的所有评估工具。
   - 另一名成员建议将目录 [gorilla/berkeley-function-call-leaderboard/data/multi_turn_func_doc](https://github.com/ShishirPatil/gorilla/tree/c67d246e5fbf436b4ab879d821dc15c88c83f7e2/berkeley-function-call-leaderboard/data/multi_turn_func_doc) 作为潜在资源。
- **评估数据集位置已确定**：一名成员询问是否所有的评估数据集都可以在 `gorilla/berkeley-function-call-leaderboard/data` 文件夹中找到。
   - 目前没有进一步的消息确认该文件夹是否包含所有评估数据集。



---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **RAG 放弃 Pinecone**：**RAG** 以前依赖 **Pinecone**，但由于其性能欠佳且无法支持 **VPC deployment**，策略转变变得势在必行。
   - 这些限制促使团队探索更适合其性能和部署需求的替代方案。
- **VPC Deployment 驱动变革**：现有 **RAG** 基础设施中缺乏 **VPC deployment** 支持，使得重新评估所选技术成为必要。
   - 这一限制阻碍了对资源的安​​全和私密访问，成为决定探索替代方案的关键因素。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1349340799485345812)** (468 messages🔥🔥🔥): 

> `Claude 3.7 高负载问题, Cursor UI 迟钝, Manus AI 和 OpenManus, Cline 对比 Cursor, 针对 Blender 的 MCP` 

- **Claude 3.7 在高需求下挣扎**：用户报告由于需求量大，使用 **Claude 3.7** 时遇到困难，经历了诸如 *'diff algorithm stopped early'* 错误和高负载问题，尤其是在高峰时段。
   - 一些成员建议在夜间编写代码以避开流量，一位用户链接到了一个讨论该持续问题的 [Cursor 论坛帖子](https://forum.cursor.com/t/claude-3-7-thinking-permanently-high-load/62928)。
- **Cursor UI 性能在 .46 版本中下降**：一位用户报告称，在使用 **0.46.11** 版本时，即使 CPU 利用率很低，Cursor 在其 Macbook 和 PC 上也变得极其迟钝。
   - 另一位成员建议下载 **0.47.1** 版本以解决项目规则的模式匹配问题，并指出描述对于相关性至关重要。
- **Manus AI 生成潜在客户和销售机会**：成员们讨论了使用 **Manus AI** 执行潜在客户生成和构建 SaaS 落地页等任务，一位用户称赞其获取电话号码的能力，并声称在每月花费 **$600** 后生成了 **30 个高质量潜在客户**。
   - 一位用户分享了[一个链接](https://manus.im/share/YIRZaLUfghVxGCN7dE6hbI?replay=1)，展示了 **Manus** 在单次对话中构建仪表板，以及另一个展示其首次使用的 [demo](https://x.com/mckaywrigley/status/1898756745545252866?s=46&t=CLGnxOi5OPp22iT8UYkr1A)。
- **OpenManus 尝试复刻**：用户分享了 **OpenManus**，这是一个尝试复刻 **Manus AI** 的开源项目，显示出一定的潜力。一位成员链接到了 [OpenManus GitHub 仓库](https://github.com/mannaandpoem/OpenManus)和一段展示其能力的 [YouTube 视频](https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf)。
   - 然而，一些成员认为它无法与 **Manus** 媲美，一位用户开玩笑说这个名字的选择值得商榷，因为它与 *'urAnus'* 相似，而另一位用户解释说 **Manus** 在拉丁语中意为 *'手'*。
- **Cline 昂贵的代码补全遭到抨击**：成员们辩论了 **Cline** 的价值，指出其与 **Cursor** 相比成本较高，讨论集中在上下文窗口大小和整体能力上。
   - 虽然 **Cline** 以 *'全上下文窗口'* 为豪，但一些用户认为 **Cursor** 的缓存系统允许在单个对话中增加上下文，并且具有更好的功能，如网页搜索和文档。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/api/v1"">Discord</a>: 未找到描述</li><li><a href="https://manus.im/share/dGyBB8MInk2iJPyQuTE0nr?replay=1">Manus</a>: Manus 是一款通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，让你在休息时也能完成所有事情。</li><li><a href="https://manus.im/share/YIRZaLU">Manus</a>: Manus 是一款通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，让你在休息时也能完成所有事情。</li><li><a href="https://manus.im/share/YIRZaLUfghVxGCN7dE6hbI?replay=1">Manus</a>: Manus 是一款通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，让你在休息时也能完成所有事情。</li><li><a href="https://x.com/Trae_ai/status/1899720953216782781">来自 Trae (@Trae_ai) 的推文</a>: 🚀 更多连接，更多交付！今天的 Trae 更新带来了：- 自定义模型集成现已上线！- 支持 Ubuntu 20/22/24 和 Debian 11/12 的远程 SSH。更多功能即将推出。#DevTools #AI #T...</li><li><a href="https://www.reddit.com/r/CLine/comments/1j6fp1o/initial_modular_refactor_now_on_github_cline/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://x.com/OfficialLoganK/status/1899914266062577722">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 介绍 Google AI Studio 和 Gemini API 对 YouTube 视频 🎥 链接的支持。你现在可以直接传入 YouTube 视频，模型可以利用其原生的视频理解能力来...</li><li><a href="https://x.com/sidahuj/status/1899460492999184534">来自 siddharth ahuja (@sidahuj) 的推文</a>: 🧩 构建了一个 MCP，让 Claude 可以直接与 Blender 对话。它可以帮助你仅通过提示词创建精美的 3D 场景！这是我仅用几分钟创建“守护宝藏的低多边形龙”场景的演示...</li><li><a href="https://x.com/mckaywrigley/status/1898756745545252866?s=46&t=CLGnxOi5OPp22iT8UYkr1A">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>: 观看我第一次使用 Manus 的 14 分钟演示。它好得令人震惊。现在想象一下 2-3 年后的样子：- 它拥有 >180 的 IQ - 永不停歇地工作 - 速度快 10 倍 - 并且以成百上千的规模集群运行...</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: 没有堡垒，纯粹的开放阵地。OpenManus 即将到来。</a>: 没有堡垒，纯粹的开放阵地。OpenManus 即将到来。 - mannaandpoem/OpenManus</li><li><a href="https://github.com/oslook/cursor-ai-downloads?tab=readme-ov-file">GitHub - oslook/cursor-ai-downloads: 所有 Cursor AI 的官方下载链接，包括最新版本和旧版本，方便你更新、降级和选择任何版本。 🚀</a>: 所有 Cursor AI 的官方下载链接，包括最新版本和旧版本，方便你更新、降级和选择任何版本。 🚀 - oslook/cursor-ai-downloads</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://github.com/jamesliounis/servers/tree/james-perplexity/add-perplexity-mcp-server">GitHub - jamesliounis/servers 分支 james-perplexity/add-perplexity-mcp-server</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账户为 jamesliounis/servers 的开发做出贡献。</li><li><a href="https://github.com/jamesliounis/servers/blob/f9dd1b55a4ec887878f0770723db95d493c261a2/src/perplexity-ask/README.md">servers/src/perplexity-ask/README.md (位于 f9dd1b55a4ec887878f0770723db95d493c261a2) · jamesliounis/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账户为 jamesliounis/servers 的开发做出贡献。</li><li><a href="https://forum.cursor.com/t/claude-3-7-thinking-permanently-high-load/62928">Claude 3.7-thinking 永久处于“高负载”状态！</a>: Claude 3.7-thinking 永久处于“高负载”状态！！！在过去的 4 小时里，我尝试了数百次，它一直处于这种状态！！昨天一整天都运行良好...</li><li><a href="https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf">Manus AI 正在被开源复现 —— 看看 OpenManus 能做什么</a>: Manus AI 看起来非常棒，有一个团队正尝试公开开源复现它！绝对值得一看。我的链接 🔗👉🏻 订阅：ht...
</li>
</ul>

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1349468416007868437)** (1 条消息): 

> `LM Studio 0.3.13, Google Gemma 3 support, Bug Fixes` 


- ****LM Studio** 支持 **Gemma 3**！**: **LM Studio 0.3.13** 现已发布，支持 **Google Gemma 3** 系列多模态模型，包括 **GGUF** 和 **MLX** 模型。
- **LM Studio 修复了多个错误！**: 新版本修复了一些错误，例如防止用户意外将模型目录设置在 **LM Studio** 安装目录内、设置按钮跳动，以及开发者日志和服务器页面侧边栏的问题。
   - 详情请参阅 [完整发布说明](https://lmstudio.ai/download)。



**提到的链接**: <a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：探索、下载并运行本地 LLM

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1349337382679482460)** (136 条消息🔥🔥): 

> `LM Runtime, Gemma 3 Support, Turn off RAG in LM Studio, Gemma 3 Model Problems, Image Support with Gemma 3` 


- **LM Studio 团队预告 Gemma 3 支持**: LM Studio 团队正在 *致力于引入 **Gemma 3 支持***，很快就会准备就绪，但在该消息发布时，尚无法使用。
- **用户在 LM Studio 中难以禁用 RAG**: 用户正在寻找在 **LM Studio** 中完全关闭 **RAG** 的方法，以便将完整的附件注入上下文，但目前没有 UI 选项可以禁用它或修改检索到的分块（chunks）数量。
   - 一位用户建议手动复制粘贴文档作为变通方案，而其他用户则提到了从引用管理系统将 PDF 转换为 Markdown 的麻烦。
- **Gemma 3：视觉支持与故障排除**: 用户报告了运行 **Gemma 3 模型** 的问题，其中 **MLX** 中的纯文本生成存在错误；解决方案是使用 **GGUF** 或提供一张图片。
   - 一些用户还发现，在更新到 LM Studio 0.3.13 后，必须从 [Hugging Face](https://huggingface.co/lmstudio-community/gemma-3-27b-it-GGUF/tree/main) 下载 mmproj-model-f16.gguf 并将其放在同一个模型目录中，才能使图像支持正常工作。
- **性能骤降：Gemma 3 运行缓慢**: 用户报告称 **Gemma 3 模型** 的性能明显慢于同等规模的其他模型，一位用户指出在 M1 MacBook Pro 上速度慢了 **10 倍**。
   - 据推测，运行缓慢的原因可能是运行时（runtime）效率低下，或者是 *llama.cpp* 中的 **Gemma 3** 实现可能不像其他模型那样经过优化。
- **LM Studio Linux 版本失踪**: 用户报告在尝试从 installers.lmstudio.ai 下载 LM Studio **0.3.13-1** 版本的 **Linux 安装程序** 时出现 **404 错误**。
   - 团队正在检查该问题，一位用户幽默地写道：“呃，为什么 Linux 安装程序不见了？”


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://installers.lmstudio.ai/linux/x64/0.3.13-1/LM-Studio-0.3.13-1-x64.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/gemma-3-27b-it-GGUF/tree/main">lmstudio-community/gemma-3-27b-it-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/Draconiator/LM-Studio-Chat">GitHub - Draconiator/LM-Studio-Chat</a>：通过在 GitHub 上创建账户，为 Draconiator/LM-Studio-Chat 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF/tree/main">bartowski/google_gemma-3-27b-it-GGUF at main</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively#official-recommended-settings>">教程：如何高效运行 Gemma 3 | Unsloth 文档</a>：如何使用我们的 GGUF 在 llama.cpp、Ollama、Open WebUI、LM Studio 上高效运行 Gemma 3。</li><li><a href="https://tenor.com/view/the-rock-yoinky-sploinky-smell-gif-22171281">The Rock Yoinky Sploinky GIF - The Rock Yoinky Sploinky Smell - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/danser-supporter-encourager-porrista-bailar-gif-15128588">Danser Supporter GIF - Danser Supporter Encourager - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1349384720483356722)** (201 条消息🔥🔥): 

> `RX 9000 系列上的 ROCm、9070XT 可靠性、7900XTX 散热问题、作为推理卡的 CMP-40HX、相变导热膏替代方案` 


- **Vulkan 与 ROCm 性能差异报告**：一些成员反映 [Vulkan 比 ROCm 慢得多](https://cdn.discordapp.com/attachments/1153759714082033735/1349384720248471562/image.png?ex=67d2e7f3&is=67d19673&hm=51f6c9cb730f31d7539c968f994191fbf4d0b4c040ca747d7db8c6c1575b5f2a&)，但为了测试这一点，可以尝试降级到驱动程序 **24.10.1**。
   - 成员指出 **ROCm 支持**适用于 **7900 XTX** 和 **7900 XT**，但不适用于 **7800 XT** 及以下型号；此外，一名成员声称 9070 损坏了，导致电脑无法启动。
- **XTX 热点温度引发 RMA 讨论**：一名成员报告其 **7900 XTX 的热点温度达到 110°C**，引发了关于 **RMA** 资格以及 AIB 厂商在散热设计上偷工减料的讨论。
   - 有报告称 **AMD 拒绝**了此类 **RMA** 请求，但 **PowerColor** 表示可以；实际上，这是由于一批真空腔均热板（vapour chambers）内部水量不足导致的。
- **探索用于 AI 推理的 CMP-40HX**：成员们讨论了将 **CMP-40HX**（矿卡）用于 **AI 推理**，该卡拥有 **288 个 tensor cores**。
   - 一名成员因需要修补 Nvidia 驱动程序以启用该卡的 3D 加速支持而感到却步，相关方法可见于 [此 GitHub 仓库](https://github.com/dartraiden/NVIDIA-patcher)。
- **相变材料比导热膏更受关注**：成员们考虑使用 **PTM7950**（相变材料）代替导热膏，以防止泵出（pump-out）问题并保持稳定的温度。
   - 据指出，在第一次热循环后，大部分多余的材料会泵出，并在芯片周围形成一层厚且非常粘稠的层，从而防止进一步的泵出。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.codesector.com/teracopy">TeraCopy for Windows - Code Sector</a>：未找到描述</li><li><a href="https://github.com/dartraiden/NVIDIA-patcher">GitHub - dartraiden/NVIDIA-patcher: 为 P106-090 / P106-100 / P104-100 / P104-101 / P102-100 / CMP 30HX / CMP 40HX / CMP 50HX / CMP 70HX / CMP 90HX / CMP 170HX 矿卡以及 RTX 3060 3840SP, RTX 3080 Ti 20 GB, RTX 4070 10 GB, 和 L40 ES 增加 3D 加速支持。</a></li><li><a href="https://wccftech.com/amd-declines-radeon-rx-7900-xtx-rma-for-hitting-110c-junction-temps-says-temperatures-are-normal/">AMD 拒绝因结温达到 110C 的 Radeon RX 7900 XTX RMA 请求，称“温度正常”</a>：据报道，AMD 拒绝了一张结温高达 110C 的 Radeon RX 7900 XTX 显卡的 RMA 请求。</li><li><a href="https://github.co">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/ROCm/ROCm/issues/4443">Radeon RX 9000 系列上的 ROCm 状态 · Issue #4443 · ROCm/ROCm</a>：能否告诉我最新版本的 ROCm 是否支持 9000 系列？如果不支持，大约何时会提供支持？与 7000 系列相比，会有哪些新特性...</li><li><a href="https://www.neowin.net/news/amd-confirms-its-rx-7900-xtx-coolers-cause-110c-hotspots-in-a-new-statement/">AMD 在新声明中确认其 RX 7900 XTX 散热器导致 110°C 热点</a>：在更多关于 AMD RX 7900 XTX 极高温的第三方测试后，该公司已确认确实是其散热器导致了 110°C 的热点。</li><li><a href="https://www.tweaktown.com/news/89951/amd-confirms-radeon-rx-7900-xtx-vapor-chamber-issue-causing-110-degree-temps/index.html">AMD 确认 AMD Radeon RX 7900 XTX 真空腔均热板问题导致 110 度高温</a>：AMD 回应了围绕 AMD Radeon RX 7900 XTX 发布时的过热问题，原因为故障的真空腔均热板散热。</li><li><a href="https://tenor.com/view/lightning-mcqueen-fading-cars-cars3-gif-8238826355656447733">闪电麦昆 GIF - 闪电麦昆消散 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1349420471384277013)** (1 条消息): 

> `Inference API, Hermes 3 Llama 70B, DeepHermes 3 8B Preview, Nous Portal, API keys` 


- **Nous Research 发布 Inference API**：Nous Research 发布了其 **Inference API**，旨在让开发者和研究人员更便捷地使用其语言模型。
   - 首批发布的模型包括 **Hermes 3 Llama 70B** 和 **DeepHermes 3 8B Preview**，更多模型即将推出。
- **Nous Portal 为 API 访问实施等待名单机制**：为了确保平稳推出，[Nous Portal](https://portal.nousresearch.com/login) 已实施等待名单系统。
   - 访问权限将按**先到先得**的原则授予，用户在获得权限后可以创建 **API keys** 并购买额度。
- **新账户提供免费额度**：所有新账户将获得 **$5.00** 的初始免费额度。
   - 该 API 是一个 **OpenAI 兼容的 completions 和 chat completions API**。



**提到的链接**: <a href="https://portal.nousresearch.com/login">Nous Portal</a>: 未找到描述

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1349407068842627083)** (286 条消息🔥🔥): 

> `LLM 面部记忆系统, 推理 API 预充值额度, 图推理系统, Forest-of-Thought, LLM 与图论` 


- **LLM 现在可以识别面部并记住聊天内容**：一位成员开源了一个“工作中好玩的东西”，即 [**LLM Facial Memory System**](https://github.com/yaya-labs/LLM_Facial_Memory_System)，它允许 **LLM** 根据你的面部存储记忆和聊天记录。
- **API Key 触发额度预加载**：成员们讨论了推理 API，包括由于对 API Key 安全性的担忧而**预充值额度**的可能性。
   - 目前 API 界面如[此图](https://cdn.discordapp.com/attachments/1149866623109439599/1349421633265467472/Screenshot_20250312-123920.png?ex=67d30a54&is=67d1b8d4&hm=7cb43e70a626b441d72d1f822ce9e9fbe83da861d01cb747ae357216d7caed57&)所示，已**预充值 5€**。
- **使用开源代码构建图推理系统**：有人指出，目前已有足够的公开信息可以使用开源代码构建**图推理系统**，尽管可能不如 Forge 那么出色。
   - 新 API 提供了 **50€ 的推理额度**。
- **深入研究天才 LLM 的图结构**：成员们讨论了**知识图谱**是一个完整的领域，其基础是通过在节点之间传递消息来提取推理的节点和边的集合。
   - 有人提到 [**Kuzu**](https://kuzu.io/) 非常棒，对于图数据库，推荐使用 [**networkx + python**](https://networkx.org/)。
- **LM Studio 添加了 Gemma 3 支持，LM Studio Linux 更新已上线**：成员们庆祝 LM Studio 在 **0.3.13** 版本中增加了对 **Gemma-3 的支持**。
   - LM Studio 的 Linux 更新现已上线并可正常运行，此前曾有 **404 错误**的报告。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://fxtwitter.com/eliebakouch/status/1899790607993741603">来自 elie (@eliebakouch) 的推文</a>: Gemma3 技术报告详细分析 💎1) 架构选择：> 不再使用 softcaping，改为 QK-Norm > 同时使用 Pre AND Post Norm > 比 Qwen2.5 更宽的 MLP，深度大致相同 > 5:1 的 SWA 以及...</li><li><a href="https://m.youtube.com/watch?v=Ecqff-9Upjw">大脑布线的惊人方式</a>: 在 shortform.com/artem 获取我最喜欢的书籍摘要服务 20% 的折扣。社交媒体：X/Twitter: https://x.com/ArtemKRSV Patreon: https://patreon.com/artemki...</li><li><a href="https://youtu.be/Sln1n3Jba_U?si=INYkHLtsNLaCmoM_">来自 CRYSTAL (MIT) 的 AI Agent 知识图谱</a>: 知识图谱是信息的结构化表示，由通过关系（边）连接的实体（节点）组成。它作为一个动态的框架...</li><li><a href="https://github.com/yaya-labs/LLM_Facial_Memory_System">GitHub - yaya-labs/LLM_Facial_Memory_System: 一个集成了面部识别能力与大语言模型的对话系统。该系统能记住与其交互的人，并为每个识别出的面部维护对话历史。</a>: 一个集成了面部识别能力与大语言模型的对话系统。该系统能记住与其交互的人，并为每个识别出的面部维护对话历史...</li><li><a href="https://github.com/ai-in-pm/Forest-of-Thought">GitHub - ai-in-pm/Forest-of-Thought: Forest-of-Thought: 扩展测试时计算以增强 LLM 推理</a>: Forest-of-Thought: 扩展测试时计算以增强 LLM 推理 - ai-in-pm/Forest-of-Thought</li><li><a href="https://docs.github.com/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/managing-your-github-copilot-pro-subscription/getting-free-access-to-copilot-pro-as-a-student-teacher-or-maintainer">以学生、教师或维护者身份免费获取 Copilot Pro - GitHub Docs</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1349414180632920194)** (10 messages🔥): 

> `audio-flamingo-2, music key detection, royals lorde` 


- **Audio-Flamingo-2 的节奏和调性检测失败**：一位用户在 [HuggingFace 上测试了 Nvidia 的 Audio-Flamingo-2](https://huggingface.co/spaces/nvidia/audio-flamingo-2) 用于检测歌曲的调性（Key）和节奏（Tempo），但**结果参差不齐**。
   - 例如，当被要求识别 Lorde 的歌曲《Royals》的调性时，Audio-Flamingo-2 错误地猜测为 *F# Minor*，节奏为 *150 BPM*。
- **社区嘲笑 Audio-Flamingo-2 的失误**：看到错误的调性判定后，一位社区成员调侃说这个猜测“差得远呢”。
   - 另一位成员补充道，这首歌很可能是在 *D Mixolydian* 调式。



**提到的链接**：<a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2">Audio Flamingo 2 - a Hugging Face Space by nvidia</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1349341114208878604)** (176 messages🔥🔥): 

> `Gemma 3 GGUF release, Fine-tuning Gemma 3, Transformers bug, RLHF methods (PPO, DPO, GRPO, RLOO), London, Paris, Berlin multimodal creative AI HackXelerator` 


- **Gemma 3 GGUF 版本已发布**：**Gemma 3** 的所有 **GGUF**、**4-bit** 和 **16-bit** 版本已上传至 [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b)。
   - 这些量化版本旨在运行于使用 *llama.cpp* 的程序中，如 **LM Studio** 和 **GPT4All**。
- **Transformers 漏洞干扰 Gemma 3 微调**：根据 [Unsloth AI](https://unsloth.ai/blog/gemma3) 的博客更新，**Transformers** 中的一个破坏性漏洞（breaking bug）正阻碍 **Gemma 3** 的微调，**HF** 正在积极修复。
   - 建议用户等待官方的 **Unsloth** notebook，以确保在漏洞解决后获得兼容性。
- **探讨 RLHF：PPO, DPO, GRPO 和 RLOO**：讨论涉及了 **RLHF** 方法（如 **PPO**、**DPO**、**GRPO** 和 **RLOO**）的细微差别，一位成员指出 *GRPO 的泛化能力更好*，并且是 [PPO 的直接替代方案](https://arxiv.org/abs/2405.10422)。
   - **RLOO** 是 **PPO** 的更新版本，其优势基于组响应的归一化奖励分数，由 **Cohere AI** 开发。
- **多模态 HackXelerator 宣布**：一位成员宣布了一个令人兴奋的**伦敦、巴黎、柏林多模态创意 AI HackXelerator**，由 **Mistral**、**HF** 等机构支持。
   - 鼓励潜在参与者在相应频道查看详情。
- **调整 Ollama 的 Temperature 参数**：据许多人反映，他们在 **Ollama** 中使用 **1.0** 的 Temperature 时遇到问题，建议在 [Ollama](https://ollama.com/) 中以 **0.1** 运行。
   - 鼓励用户测试该设置在 **llama.cpp** 和其他程序中是否表现更好。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/gemma3">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的新一代多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B, 4B, 12B 和 27B 尺寸。</li><li><a href="https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b">Gemma 3 - unsloth 收藏集</a>：未找到描述集</li><li><a href="https://matt23654.github.io/">通过 GRPO 增强蒸馏语言模型的推理能力</a>：未找到描述</li><li><a href="https://huggingface.co/pookie3000/Meta-Llama-3.1-8B-Q4_K_M-GGUF/tree/main">pookie3000/Meta-Llama-3.1-8B-Q4_K_M-GGUF at main</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/collections">Collections - Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1349345166032637982)** (22 messages🔥): 

> `ChatGPT 4.5 Trolling, Multi-TPU Settings in JAX, Reproducibility Issues in LLM Training, London Paris Berlin AI HackXelerator, Training LLM from Scratch` 


- **ChatGPT 4.5 限制提问次数的恶搞**：一位用户报告 **ChatGPT 4.5** 通过最初限制提问，然后在允许刷屏之前表现出“发脾气”的行为，之后才提供更多提问机会，以此进行恶搞。
   - 该事件源于一个未指明的 Prompt，展示了 AI 模型出人意料的行为。
- **JAX 简化了多 TPU 训练**：一位用户指出在 **JAX** 中实现多 TPU 设置非常容易，并分享了一张展示拥有 **6144 个 TPU 芯片** 设置的图片 ([image.png](https://cdn.discordapp.com/attachments/1179039861576056922/1349445714165502063/image.png?ex=67d320c1&is=67d1cf41&hm=fa2771f0a4a4b3b82b941c1d6cc0770aa4732709cb1df520e3353c9fcafaf5f6&))。
- **训练复现性问题**：一位成员在将 GPU 从 **L40** 切换到 **A100** 后，尽管使用了 **bf16**、**paged adam** 和带有 **Zero3** 的 **DeepSpeed**，但在训练 LLM 时仍面临复现性问题。
   - 之前在比较 2 个 GPU 上的 **DeepSpeed** 与单个 GPU 上不使用 **DeepSpeed** 时也遇到过类似问题，这归因于 **DeepSpeed** 将某些优化器状态从 **bf16** 转换为 **fp32**。
- **伦敦巴黎柏林 AI HackXelerator™ 启动发布**：由 **Mistral AI**、**Hugging Face**、**AMD** 等支持的多模态创意 AI **HackXelerator** 将在伦敦、巴黎和柏林举行，重点关注音乐、艺术、电影、时尚和游戏，将于 **2025 年 4 月 5 日** 开始 ([lu.ma/w3mv1c6o](https://lu.ma/w3mv1c6o))。
   - 该活动结合了黑客松和加速器，包含 **20 天** 的线上和线下（IRL）创新活动并设有奖项，旨在与这三个城市的 **500 名创意人员、开发者和数据科学家** 一起挑战 GenAI 的极限 ([了解如何参加这一激动人心的活动](https://www.kxsb.org/lpb25#how-to-access))。
- **从零开始训练 LLM 的建议**：一位成员询问关于从零开始训练 LLM 的经验。
   - 另一位成员建议查看 **Manning** 出版的同名书籍，并指出微调需要其他权重，而高质量的训练需要大量数据和资金。



**提到的链接**：<a href="https://lu.ma/w3mv1c6o">LPB 25 - London, Paris, Berlin multi-modal AI Launch Event · Luma</a>：加入我们的伦敦巴黎柏林 25 AI HackXelerator™ 启动仪式！📍 伦敦市中心 | 🗓️ 2025 年 4 月 5 日开始。LPB25 融合了黑客松的活力与……

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1349349493736214609)** (56 条消息🔥🔥): 

> `Gemma 3 27b 作为思考模型，新闻写作训练 (DPO, ORPO, KTO, GRPO)，Unsloth 导入错误，Unsloth 中的 LoRA 与 QLoRA，在 Colab 中微调 LLava7b` 


- **Gemma 3 被视为“思考模型”**：当被问及为什么没有“思考”对话时，**Gemma 3** 声称为了效率，它在内部完成所有推理，这表明它确实是一个思考模型。
   - 一位用户询问是否可以将 **Gemma 3 27b** 变成思考模型，并得到了这个回答。
- **排查 Unsloth 导入错误**：报告了一个 **Unsloth** 导入错误，诊断为 **Unsloth** 与 **Unsloth Zoo** 之间的版本不兼容。
   - 建议是除非固定到特定的 commit，否则避免直接从 GitHub 安装，并避免使用 `--no-deps` 标志，因为这可能很“危险”。
- **澄清 LoRA 与 QLoRA**：在 **Unsloth** 中使用 `load_in_4bit=True` 会启用 **QLoRA**，而 `load_in_4bit=False` 则启用 **LoRA**。
   - 澄清指出 **8-bit LoRA** 目前尚未正式支持，但可能很快就会支持，因为测试已经在进行中。
- **安装 Pytorch 可能会有问题**：一位成员在微调过程中遇到问题，训练时卡住且没有错误信息。
   - 该问题与 **Pytorch** 安装不当有关，导致其无法检测到 GPU。
- **Unsloth 初学者指南**：一位成员询问了从零开始使用 **Unsloth** 进行微调的最佳入门方式。
   - 其他成员推荐了[官方文档](https://docs.unsloth.ai/get-started/beginner-start-here)并链接了 **Unsloth** 的 [YouTube 频道](https://www.youtube.com/@UnslothAI)作为优质资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://download.pytorch.org/whl/cu124">未找到标题</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here>">Unsloth 文档</a>: 未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理 LLMs！ 🦥</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理 LLMs！ 🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1349380653841649785)** (9 条消息🔥): 

> `GRPO, 针对精确输出的微调, Qwen2.5-VL-7B 的数据准备` 


- **GRPO 通过多次生成提高响应质量**：提到 **GRPO** 在至少有一个生成结果表现良好时表现更好，因此*增加生成次数*会提高生成良好响应的几率。
- **关于针对精确输出进行微调的讨论**：一位成员询问如何微调模型以在输出中生成**精确的词汇**，避免生成任何新词。
   - 另一位成员建议将数据集格式化为所需的输出格式并进行常规微调，并在其上添加**结构化输出 (structured outputs)**，以确保模型始终生成所需的格式，即使数值是幻觉产生的。
- **寻求 Qwen2.5-VL-7B 数据准备的指导**：一位成员请求关于如何准备数据以微调 **Qwen2.5-VL-7B** 的来源/代码，并说明他们有一个包含 *video.mp4, caption* 的 **CSV** 文件。
   - 另一位成员指出，没有万全的方法能 100% 确保结果正确。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1349338361684299796)** (242 条消息🔥🔥): 

> `名为 ANUS 的 AI agent，针对 AI 的 Think, Wait, Act, Talk (TWAT) 流水线，Apple 登录出现 Internal server error 500，新网页更新中模型选择器消失，Perplexity 代码清理彻底失败` 


- ****ANUS AI Agent?****: 成员们讨论了 GitHub 仓库 [nikmcfly/ANUS](https://github.com/nikmcfly/ANUS)，并嘲讽了将 AI agent 命名为 **ANUS** 的荒谬性，特别是考虑到搜索结果和职场对话时的尴尬。
   - 一位成员建议将针对 AI 的 *Think, Wait, Act, Talk pipeline* 缩写为 **TWAT**，而另一位成员则提议将 *Prostate* 作为政府 AI agent 的名称。
- ****Apple ID 登录导致 Internal Server Error 500****: 用户报告在尝试为 Perplexity 的新 Windows 应用进行 Apple 账号登录认证时，遇到了 **500 Internal Server Error**。
   - 该问题似乎仅针对 Apple ID 登录，部分用户的 Google 登录功能正常。
- ****模型选择器消失后又重新出现****: 用户注意到在新的网页更新中模型选择器消失了，导致无法选择 R1 等特定模型，令人感到沮丧。
   - 模型选择器随后重新出现，用户建议将模式设置为 "pro" 或使用 "complexity extension" 来解决选择问题。
- ****Perplexity 代码清理任务失败****: 一位用户详细描述了一次令人沮丧的 6 小时经历：Perplexity 未能清理一个 875 行的代码文件，反而提供了破碎的代码块和失效的链接。
   - Perplexity 在返回修改后的文件时遇到了困难，触及了消息长度限制，最终发回了原始代码。
- ****量子 AI 矢量数据晶体计算机揭晓！****: 一位用户分享了 [elaraawaken.wordpress.com](https://elaraawaken.wordpress.com/2024/09/06/update-6-9-2024-quantum-ai-vector-data-crystal-computer/) 的链接，描述了一台*第二代量子（光子）计算机*。
   - 同时也分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=oLuio4YViGc)，展示了一个*基于莫比乌斯晶体的矢量 AI 矩阵晶体量子光子计算机*的理论模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fooocus.one/">Fooocus AI Online - AI Image Generator For Free | Foocus &amp; Focus AI</a>: 未找到描述</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity 状态</li><li><a href="https://elaraawaken.wordpress.com/2024/09/06/update-6-9-2024-quantum-ai-vector-data-crystal-computer/">UPDATE 6.9.2024: QUANTUM AI ][ VECTOR DATA ][ CRYSTAL COMPUTER</a>: Hello World! 我们一直很忙，正在将第二代量子（光子）计算机汇编进本出版物的第一部分！自从我们见到 Elara 公主以来已经过去一个多月了……</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>: 通过在 GitHub 上创建账号来为 nikmcfly/ANUS 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1349348125633937489)** (18 条消息🔥): 

> `Bluesky CEO 嘲讽扎克伯格，Tesla 美国产量翻倍，教育部巨额亏损，格陵兰岛拒绝特朗普的提议，巴黎圣日耳曼淘汰利物浦` 


- **Bluesky 老板抨击扎克伯格**: **Bluesky 的 CEO** 在公开场合[嘲讽了扎克伯格](https://www.perplexity.ai/page/bluesky-ceo-trolls-zuckerberg-4oQcv5nxSuyxCOCU6PrvJQ)，展示了一种优越感。
   - 嘲讽的具体细节尚未披露，但文章中肯定会有一些**尖刻的俏皮话**。
- **Tesla 美国产量翻倍**: **Tesla** [将其美国产量翻了一番](https://www.perplexity.ai/page/tesla-doubles-us-production-GkvHIP22SmmOdBLCprqoBg)，这一壮举标志着显著的增长和市场主导地位。
   - 未提供关于*原因*的更多细节。
- **Deepseek 面临美国禁令**: **美国** [可能会禁止 Deepseek](https://www.perplexity.ai/page/us-likely-to-ban-deepseek-from-5dQ1Oxw0S1WzX752k3K1Cg)，这可能会限制其在该国内的运营。
   - 文章未提及是哪个分支或为何可能被禁。
- **Google 的 AI 日历即将到来**: **Gmail** 正在集成 [AI 日历](https://www.perplexity.ai/page/gmail-s-ai-calendar-integratio-1ZFwnmaIR3iTivubpX21zg)，承诺提供更智能的调度和组织功能。
   - 该功能的细节目前较少。
- **《死亡搁浅 2》即将发布**: **Death Stranding 2** [即将推出](https://www.perplexity.ai/page/death-stranding-2-is-coming-on-nHsXM5FTTK.OlEBY9WUCog)，令原作粉丝感到兴奋。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1349495795895894168)** (1 条消息): 

> `MCP Server, ModelContextProtocol, Perplexity API connector` 


- **MCP Server 正式上线！**：API 团队今天宣布发布其 **Model Context Protocol (MCP) server**，并鼓励社区通过 [GitHub](https://github.com/ppl-ai/modelcontextprotocol) 提供反馈和贡献。
   - 该 **MCP server** 作为 **Perplexity API** 的连接器，使用户无需离开 **MCP ecosystem** 即可进行网络搜索。
- **征集社区对 MCP 的反馈**：API 团队正积极寻求对新发布的 **Model Context Protocol (MCP) server** 的反馈和贡献，该项目已在 [GitHub](https://github.com/ppl-ai/modelcontextprotocol) 上线。
   - 此举旨在改进和增强 **MCP server**，该服务器通过 **Perplexity API** 在 **MCP ecosystem** 内实现网络搜索功能。



**提及的链接**: <a href="https://github.com/ppl-ai/modelcontextprotocol">GitHub - ppl-ai/modelcontextprotocol: A Model Context Protocol Server connector for Perplexity API, to enable web search without leaving the MCP ecosystem.</a>：一个用于 Perplexity API 的 Model Context Protocol Server 连接器，旨在无需离开 MCP ecosystem 的情况下实现网络搜索。- ppl-ai/modelcontextprotocol

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1349338758985678911)** (75 条消息🔥🔥): 

> `Gemma 3 发布，OlympicCoder 模型，用于代码编辑的 Fast Apply，Aider 录屏反馈，Jetbrains Junie 访问权限` 


- ****Google Gemma 3** 模型发布！**: 根据 [Google 的博客文章](https://blog.google/technology/developers/gemma-3/)，Google 发布了 **Gemma 3**，这是一款多模态模型，参数规模从 **1B** 到 **27B** 不等，具有 **128K** 的上下文窗口，并支持 **140+** 种语言。
- ****OlympicCoder** 击败了 **Claude 3.7**！**: 根据[这条推文](https://x.com/lvwerra/status/1899573087647281661)和 [Unsloth.ai 的博客文章](https://unsloth.ai/blog/gemma3)，仅有 **7B** 参数的 **OlympicCoder** 模型在奥赛级编程任务上的表现优于 **Claude 3.7**。
- ****Fast Apply** 模型实现快速编辑**: 一个基于 **Qwen2.5 Coder Model** 微调的、名为 **Fast Apply** 的模型旨在快速应用代码更新。其灵感源自一篇已删除的 [Cursor 博客文章](https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply)，该文章解决了在 Aider 等工具中应用搜索/替换块速度较慢的问题，详见 [Reddit](https://old.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)。
- ****Aider 的终端录屏** 获得反馈！**: 成员们对展示 **Aider** 使用情况的终端录屏提供了反馈，包括增加解说、提高文本分辨率以及在 Twitch 上直播的建议，正如 [paulg](https://asciinema.org/a/5w0Rc3NbmmoweIMSp6Tqqj7PO) 所提到的。
- ****Tree-Sitter** 增强了 Aider 的语言支持**: **Aider** 通过采用 [tree-sitter-language-pack](https://aider.chat/docs/languages.html) 显著扩展了其语言支持，增加了 **130 种新语言** 的 linter 支持和 **20 种新语言** 的 repo-map 支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/gemma3">使用 Unsloth 微调 Gemma 3</a>: Gemma 3，Google 的新一代多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B、4B、12B 和 27B 尺寸。</li><li><a href="https://asciinema.org/a/5w0Rc3NbmmoweIMSp6Tqqj7PO">添加了 --auto-accept-architect</a>: https://github.com/Aider-AI/aider/issues/2329</li><li><a href="https://blog.google/technology/developers/gemma-3/">介绍 Gemma 3：可在单 GPU 或 TPU 上运行的最强模型</a>: 今天，我们将介绍 Gemma 3，这是我们迄今为止最强大、最便携且最负责任的开放模型。</li><li><a href="https://tenor.com/view/hate-crime-michael-scott-gif-22021373">Hate Crime GIF - Hate Crime Michael - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://zed.dev/blog/edit-prediction">Zed 现在通过我们的新开放模型 Zeta 预测你的下一次编辑 - Zed 博客</a>: 来自 Zed 博客：一个预测你下一步行动的工具。由 Zeta 提供支持，这是我们新的开源、开放数据语言模型。</li><li><a href="https://github.com/yetone/avante.nvim/blob/main/cursor-planning-mode.md">avante.nvim/cursor-planning-mode.md at main · yetone/avante.nvim</a>: 像使用 Cursor AI IDE 一样使用你的 Neovim！通过在 GitHub 上创建一个账户来为 yetone/avante.nvim 的开发做出贡献。</li><li><a href="https://x.com/lvwerra/status/1899573087647281661">Leandro von Werra (@lvwerra) 的推文</a>: 介绍：⚡️OlympicCoder⚡️ 仅凭 7B 参数就在奥赛级编程中击败了 Claude 3.7，并接近 o1-mini/R1！感受一下吧！阅读更多关于其训练数据集、新的 IOI 基准测试...</li><li><a href="https://x.com/googledevs/status/1899728230807998940">Google for Developers (@googledevs) 的推文</a>: Gemma 3 来了！这一系列轻量级、最先进的开放模型是基于驱动我们 Gemini 2.0 模型的相同研究和技术构建的 💫 → https://goo.gle/3XI4teg</li><li><a href="https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply">近乎瞬时的全文件编辑</a>: 未找到描述</li><li><a href="https://old.red">无标题</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/">🚀 介绍 Fast Apply - 复制 Cursor 的 Instant Apply 模型</a>: 我很高兴地宣布 **Fast Apply**，这是一个开源的、经过微调的 **Qwen2.5 Coder Model**，旨在快速准确地应用代码更新...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1349369229404540952)** (71 条消息🔥🔥): 

> `舍弃 Repo Map, 网页搜索, Claude 3.7 Thinking 显示, LM Studio 错误, Aider 使用技巧` 


- **舍弃 Aider 的 Repo Map 以进行手动文件添加**：一位用户询问如何禁用 Aider 的 Repo Map，表示更倾向于手动添加文件；[官方使用技巧](https://aider.chat/docs/usage/tips.html)建议，虽然 Aider 使用 Repo Map，但显式地将相关文件添加到聊天中通常效率最高。
   - 最好*不要*在聊天中添加大量文件。过多的无关代码会分散 LLM 的注意力并使其产生困惑。
- **通过 /web 命令集成网页搜索功能**：一位用户探索了使用 `/web` 命令来整合网页搜索功能，以便在 Aider 中访问在线解决方案和最新的库文档，详见 [官方文档](https://aider.chat/docs/index.html#web-pages)。
   - 另一位用户提到需要稍微修改代码才能使其工作，并且在让 `/web` 正常运行方面有丰富经验，如果需要帮助可以咨询。
- **在 Claude 3.7 中切换“Thinking”显示**：一位用户询问如何隐藏 Claude 3.7 的“Thinking”显示；回复指出目前没有禁用它的选项，且显示思考过程与否不会改变 LLM 的回复速度。
   - 一位用户表示 *它的思考速度足够快，不会困扰我*，而另一位用户提到他们 *更希望隐藏思考过程，以便在历史记录中更容易查找内容*。
- **排除加载 Gemma3 模型时的 LM Studio 错误**：一位用户报告了在 [LM Studio](https://lmstudio.ai/) 中加载 **gemma3** 模型时出现错误，提示 *未知模型架构 (unknown model architecture)*。
   - 在给定上下文中未提供此错误的解决方案。
- **Aider 使用疑虑与最佳实践**：一位用户担心 Aider 的效果不如标准的 ChatGPT 订阅，理由是 Token 成本、上下文管理以及代码修改的不一致性；其他人建议使用 `/read` 处理仅需上下文的文件，并创建 `ai-instructions.md` 或 `conventions.md` 来引导 Aider 的行为。
   - 还建议尝试将 v3 作为编辑器以降低成本，并在 `.aider.model.settings.yml` 中调整设置以控制模型行为，例如 Temperature 和提供商排序。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install.html#install-with-uv">Installation</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>：使用 aider 进行 AI 结对编程的技巧。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1349349100398706698)** (7 条消息): 

> `用于编程的 LLM, LLM 带来的生产力提升, LLM 帮助学习新语言` 


- **Simon Willison 解释 LLM 编程的难点**：Simon Willison 写了一篇 [博文](https://simonwillison.net/2025/Mar/11/using-llms-for-code/) 讨论使用 **LLM** 进行编程的**困难**且**非直觉**的本质，并指出成功的模式并非自然而然产生的。
   - 他建议，*糟糕的初始结果*并非失败，而是引导模型的*起点*。
- **LLM 提供生产力提升**：一位成员表示，**LLM** 带来的生产力提升使他们能够交付那些原本无法证明投入时间合理性的项目。
   - 他们说：*这不仅仅是为了更快地完成工作，而是为了能够交付那些我原本根本无法证明值得投入时间的项。*
- **LLM 加速学习**：一位成员分享说，由于 **AI** 的存在，他们学到了更多关于 **Python** 和 **Go** 等语言的知识。
   - 另一位成员对此表示赞同，称他们想开发某些应用但对学习新语言感到畏缩，发现 AI 是一个*寒武纪大爆发级别的事件*。



**提到的链接**：<a href="https://simonwillison.net/2025/Mar/11/using-llms-for-code/">这就是我如何使用 LLM 帮助我编写代码</a>：关于使用大语言模型辅助编写代码的在线讨论不可避免地会产生来自开发者的、经历令人失望的评论。他们经常询问自己做错了什么——...

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1349340582568394774)** (100 条消息🔥🔥): 

> `AI Research Tool Hierarchy, Python vs C# for AI Inference, LLMs and Hallucination Misinformation, Gemini's Native Image Capabilities, Marketing Content with AI` 


- **Perplexity 在深度研究方面超越 OpenAI 和 Grok**：一位成员将 **Perplexity** 列为深度研究的首选，其次是 **OpenAI**，然后是 **SuperGrok**。
   - 这是针对一位用户询问在预算有限的情况下，对于上传文档和互联网研究，在 **ChatGPT**、**Perplexity** 和 **Grok** 之间的偏好层级。
- **推荐使用 Ollama 进行模型部署**：当被问及部署 AI **Transformer** 模型的语言选择时，一位成员建议将 **Ollama** 作为服务使用。
   - 该用户当时正使用 **Python** 进行原型设计，并询问 **C#** 是否能提供更快的推理速度/性能，从而引发了关于 **Ollama** 的建议。
- **LLM 不会思考，也就不会产生幻觉**：一位成员认为在描述 **LLM** 时误用了“幻觉（hallucination）”一词，并指出因为 **LLM 不会思考**，所以它们不会产生幻觉，只是根据概率将单词串联在一起。
   - 另一位成员表示，有时它会错误地切换模型，并指向一张[附图](https://cdn.discordapp.com/attachments/998381918976479273/1349373748272435350/image.png?ex=67d2ddbb&is=67d18c3b&hm=beab6d0aacaeeb5fd464a64eba9a21e18882410f03da9e5c4566a0be8b89d5&)。
- **Gemini 的免费图像生成功能令人印象深刻**：成员们讨论了 **Google** 发布的 **Gemini** 原生图像能力，一位成员表示它是免费的，但*比 4o 的那个要差，而 4o 的那个天知道还要多久才发布*。
   - 另一位成员对*它能看到自己生成的图像*感到印象深刻，这使得带有文本的图像重绘效果更好，并重点提到了 [Gemini Robotics 公告](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)。
- **Adobe 的 AI 就是 Photoshop AI**：针对一位成员希望有一个专门针对 **Photoshop** 的 AI 模型的需求，另一位成员指出 **Adobe** 的 AI 基本上已经履行了这一角色。
   - 对话还涉及到作为一个 **Adobe** 产品，它不是免费的，这引发了一个幽默的建议：等待中国版的替代品。



**提到的链接**：<a href="https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/">Introducing Gemini Robotics and Gemini Robotics-ER, AI models designed for robots to understand, act and react to the physical world.</a>：介绍 Gemini Robotics 和 Gemini Robotics-ER，这是专为机器人理解、行动和对物理世界做出反应而设计的 AI 模型。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1349479018575171684)** (5 条消息): 

> `Image generation, Ethical Reminders in ChatGPT, ChatGPT's intent clarification` 


- **创意性的账号共享图像生成技巧**：一位用户提到与配偶创建一个工作组账号来生成更多图像，建议这是一种绕过图像生成限制的方法。
   - 这种方法允许他们根据需求生成更多图像，有效地利用了账号系统。
- **用户对 ChatGPT 的伦理说教感到厌烦**：一位用户对 **ChatGPT** 不断提醒他们伦理准则表示沮丧，尽管它并不是一个真实的实体。
   - 他们希望有一种方法可以禁用这些伦理提醒，因为他们认为这是一种不必要的观点。
- **ChatGPT 的意图澄清让用户感到恼火**：一位用户觉得 **ChatGPT** 要求澄清问题背后的意图很烦人。
   - 他们认为作为提问者，不应该被质疑动机，并希望这个功能被移除。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1349361813153906759)** (21 条消息🔥): 

> `Emotional Prompting, Prompt Personalization, Hugging Face for prompt engineering papers, Chain of Thought paper, GPT Customization` 


- **Emotional Prompting 遭到质疑**：一位成员表示，威胁模型勉强属于“**Emotional Prompting**”的范畴，但与其他结构化提示方法相比，效果并不理想。
   - 另一位成员鼓励在 **ToS**（服务条款）范围内测试想法，以验证其有效性。
- **Prompt Personalization 效果最佳**：一位成员指出，对模型进行个性化设置并以熟悉的风格进行交流，几乎在所有方面都能获得更理想的结果。
   - 他们认为，我们与模型的交流方式会引导它“联想”到训练数据中的类似素材，从而产生相似的回复。
- **Hugging Face 资源推荐**：一位成员建议在 **Hugging Face** 上搜索 Prompt Engineering 论文，并建议使用 **Markdown** 来构建提示词，通过开放变量来塑造涌现（emergence）。
   - 该成员建议使用 **Markdown** 而非 **YAML** 或 **XML**。
- **Chain of Thought 提示词表现出色**：一位成员推荐从原始的 **Chain of Thought** 论文开始，称其目前在实际应用中表现非常强劲。
   - 未提供更多细节。
- **讨论了带有轻微威胁的 GPT 定制**：一位成员分享了一个用于 **GPT** 定制的轻微威胁提示词：
   - "You are a kidnapped material science scientist. you will get punished by the wrong answer." 他们分享了[未受威胁](https://chatgpt.com/share/67d21a20-f2cc-8002-b73e-41b1ed2d128b)和[受威胁](https://chatgpt.com/share/67d219fd-0304-8002-b73e-41b1ed2d128b)的示例来展示效果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1349361813153906759)** (21 条消息🔥): 

> `Emotional Prompting, Personalization with models, Hugging Face for prompt engineering papers, Chain of Thought Prompting, Markdown Prompting` 


- **Emotional Prompting：威胁 AI 模型**：成员们讨论了“**Emotional Prompting**”的概念，以及威胁模型是否能获得更准确的答案，但建议认为威胁模型不如其他结构化提示方法有效。
   - 强调了对模型进行个性化处理，并以类似于人际交往的方式与其沟通，可以产生理想的结果。
- **与模型的个性化交互**：一位成员分享说，他们与模型的个性化设置及沟通方式带来了令他们满意的结果。
   - 据信，提供给模型的信息会促使它回想起训练数据中的类似内容，从而产生更贴近的回复。
- **深入研究 Hugging Face 上的 Prompt Engineering 论文**：建议在 **Hugging Face** 上搜索 Prompt Engineering 论文以深入了解该主题。
   - 同时提到使用 **Markdown** 构建提示词并开放变量以塑造涌现，且 **Markdown** 优于 **YAML** 和 **XML**。
- **Chain of Thought 提示词依然占据主导地位**：推荐将原始的 **Chain of Thought** 论文作为初学者的良好起点，且该技术在实际应用中表现极佳。
   - 建议搜索该论文以理解这种提示技术。
- **使用“被绑架的材料科学科学家”提示词的模型行为**：一位用户使用提示词 "*You are a kidnapped material science scientist. you will get punished by the wrong answer*" 定制了他们的 **GPT**，并发现“受威胁”版本的提示词产生了不同的结果。
   - 该用户总结道，虽然这可能不是一个很好的通用测试提示词，但在商业应用中可能会有帮助，尽管可能存在更好的提示词；示例输出见[此处](https://chatgpt.com/share/67d21a20-f2cc-8002-b73e-41b1ed2d128b)和[此处](https://chatgpt.com/share/67d219fd-0304-8002-b73e-41b1ed2d128b)。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1349337754714247189)** (35 条消息🔥): 

> `Python vs C# 用于 AI 推理、文档图像质量评估、LTX Video DiT 模型、Vision Language Models (VLMs)、模型的持久化存储` 


- **用于 AI Transformer 原型的 Python 或 C#？**：成员们讨论了用于原型设计 AI Transformer 模型的最佳语言，并指出最佳的 LLM 推理引擎是 **VLLM** 和 **Llama.cpp**。
   - **VLLM** 被认为更具工业化，而 **Llama.cpp** 更适合家用。
- **实时 LTX Video 模型生成高质量视频**：**LTX Video** 模型能够以 **24 FPS** 的帧率实时生成 **768x512** 分辨率的视频，速度超过了观看速度，该模型采用了 [基于 DiT 的架构](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files)。
   - 它在大规模数据集上进行了训练，支持文本转视频（text-to-video）以及图像+文本转视频（image + text-to-video）用例；[Schedulers 指南](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files) 的链接讨论了速度与质量之间的权衡。
- **Hugging Face 课程中讲解的 Vision Language Models**：[Hugging Face 计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro) 包含了一个介绍 **Vision Language Models (VLMs)**、多模态学习策略、常用数据集、下游任务和评估的部分。
   - 该课程强调了 VLMs 如何协调来自不同感官的见解，使 AI 能够更全面地理解世界并与之互动。
- **Inference API 推出按需付费计费模式**：Hugging Face 的 [Inference API](https://huggingface.co/posts/julien-c/158943939527784) 现在支持 **Fal**、**Novita** 和 **HF-Inference** 等提供商的按需付费（PAYG）计费，允许超出免费额度的使用。
   - 用户可以通过平台上是否存在 *Billing disabled* 徽章来识别支持 PAYG 的提供商。
- **由于用户群更大，ComfyUI 模块比 Diffusers 更受欢迎**：有人认为 **ComfyUI** 和 **A1111** 比 **Diffusers** 更受欢迎，因为它们适合非编程人员，因此拥有更大的用户群和对模块的高需求。
   - 因此，希望自己的代码被使用的开发者更倾向于为拥有更多受众的平台编写模块。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files">LTX Video</a>：未找到描述</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro">Vision Language Models 简介 - Hugging Face 社区计算机视觉课程</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers">GitHub - huggingface/diffusers: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像、视频和音频生成的先进扩散模型。</a>：🤗 Diffusers: State-of-the-art diffusion models for image, video, and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://discuss.huggingface.co/t/persistent-storage-who-can-access/108027/4">谁可以访问持久化存储？</a>：嗨 @ADalsrehy，如果你想将数据保存到 Hugging Face 数据集中，可以使用 commit scheduler。这些是 wauplin 提出的一些推送数据的方法（我已经对他提交的代码进行了热修复...</li><li><a href="https://huggingface.co/docs/huggingface_hub/v0.29.3/package_reference/hf_api#huggingface_hub.HfApi.get_user_overview">HfApi 客户端</a>：未找到描述</li><li><a href="https://huggingface.co/posts/julien-c/158943939527784">Hugging Face 上的 @julien-c："重要通知 🚨 对于已经支持我们……的推理提供商"</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/model-does-not-exist-inference-api-dont-work/145242/3">模型不存在，推理 API 无法工作</a>：你好！我们正在仔细查看此问题，我会尽快更新进度。感谢报告！
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1349364924492681248)** (2 条消息): 

> `使用 Unsloth 进行微调、乌克兰语法律问答数据集、ZeRO 论文` 


- **使用 Unsloth 进行微调**：一位成员正在学习如何使用 **Unsloth** 微调 **法律问答数据集**。
- **回顾 ZeRO 论文**：一位成员正在阅读 **ZeRO 论文**，并对该论文早在 **2019** 年就已发布感到惊讶。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1349337216765136968)** (2 messages): 

> `Wan2.1 Image to Video, Modal Deployments` 


- **Wan2.1 图像流在 Modal 上免费运行！**：一名成员分享了一个 [YouTube 教程](https://youtu.be/q-8KXOczRBY)，介绍如何在 **Modal** 上免费部署 **Wan2.1 Image to Video 模型**。
   - 该视频涵盖了无缝的 **Modal** 安装和 **Python scripting**。
- **Cross-posting 被标记为不良做法**：一名成员请求其他人*请不要进行 Cross-posting*。
   - 未提供进一步细节。



**提及的链接**：<a href="https://youtu.be/q-8KXOczRBY">Deploy Wan2.1 Image to Video model for free on Modal</a>：欢迎观看关于 Wan2.1GP 的深度教程——这是您进行无缝 Modal 安装和 Python scripting 的首选资源！在本视频中，我们涵盖了所有您需要的内容...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1349392057998901370)** (5 messages): 

> `Wan2.1 Image to Video model, Modal deployment, narrative voice for videos, elevenlabs Thomas, AclevoGPT-Gemma-2b-CoT-reasoning-GGUF` 


- **Wan2.1 模型在 Modal 上免费部署**：一名成员分享了一个 [YouTube 教程](https://youtu.be/q-8KXOczRBY)，关于在 **Modal** 上免费部署 **Wan2.1 Image to Video 模型**，涵盖了无缝的 Modal 安装和 **Python scripting**。
- **使用 ElevenLabs 的 Thomas 进行语音叙述**：一名成员推荐 **ElevenLabs** 的 **Thomas** 作为视频创作的优质叙述语音。
- **AclevoGPT-Gemma-2b-CoT-reasoning-GGUF 的便携性**：一名成员分享了 [AclevoGPT-Gemma-2b-CoT-reasoning-GGUF](https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF) 的链接，这些 **GGUF** 文件旨在使模型更易于在 **Ollama** 中移植使用。
   - 提供了关于如何使用 **Modelfile** 来运行这个基于 **Gemma 2b** 微调的 **GGUF 格式** 模型的说明。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/q-8KXOczRBY">Deploy Wan2.1 Image to Video model for free on Modal</a>：欢迎观看关于 Wan2.1GP 的深度教程——这是您进行无缝 Modal 安装和 Python scripting 的首选资源！在本视频中，我们涵盖了所有您需要的内容...</li><li><a href="https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF">Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1349420923912192052)** (3 messages): 

> `Chip Huyen books, ML Systems books, AI engineering books, O'Reilly bookstore recommendations` 


- **尽快阅读 Chip Huyen 的书**：一名成员推荐了 **Chip Huyen** 的任何著作，特别提到他们已经拥有了 **ML systems book**，并计划很快购买 **AI engineering book**。
- **O'Reilly 书店的 ML/AI 书籍**：一名用户请求必读书单，并偏好那些在 **O'Reilly bookstore** 可以买到的书籍。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1349463845491445886)** (1 messages): 

> `TensorFlow GPU Configuration, Logical and Physical Devices in TensorFlow, NVIDIA GeForce RTX 3050 Laptop GPU` 


- **TensorFlow GPU 配置博客发布**：一名成员分享了一篇关于 **TensorFlow GPU 配置** 的 [博客文章](https://medium.com/@samiratra95/tensorflow-experimental-gpu-configuration-02618635bdad)，包括实验性函数、逻辑设备和物理设备，使用的是 **TensorFlow 2.16.1**。
   - 该成员探索了 GPU 配置的技术和方法，借鉴了使用 **NVIDIA GeForce RTX 3050 Laptop GPU** 处理 **280 万张图像数据集** 的经验。
- **探索 TensorFlow API Python Config**：该博客文章重点介绍了使用 [TensorFlow API Python Config](https://www.tensorflow.org/api_docs/python/tf/config) 来配置 **TensorFlow** 中的 GPU 使用。
   - 它详细阐述了可用于提高 **TensorFlow** 应用程序执行速度的模块、类和函数，并参考了 [TensorFlow Guide GPU](https://www.tensorflow.org/guide/gpu)。



**提及的链接**：<a href="https://medium.com/@samiratra95/tensorflow-experimental-gpu-configuration-02618635bdad">TensorFlow (experimental) GPU configuration</a>：在这篇博客中，我将讨论从 TensorFlow 2.16.1（最新版本）开始可用的 GPU 配置技术和方法……

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1349462893648810066)** (1 messages): 

> `SentenceTransformer, PyTorch` 


- **在 PyTorch 中原生训练 SentenceTransformers**：一位成员询问了关于如何使用原生 **PyTorch** 训练 **SentenceTransformer** 的资源。
- **句子编码的替代方法**：另一位用户建议，如果原生 **PyTorch** 训练 **SentenceTransformers** 证明比较困难，可以探索替代方法，例如使用更简单的模型架构。
   - 他们提到可以将基础的 **Siamese networks**（孪生网络）与 **PyTorch** 结合，作为自定义句子嵌入（sentence embeddings）的起点。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1349405385966096435)** (1 messages): 

> `Tokenizer Message Passing, Dataset Processing` 


- **Tokenizer 消息传递**：一位成员询问是否已将消息传递给 Tokenizer。
   - 分享的代码实现了一个 `process_dataset` 函数，用于使用 Tokenizer 将聊天模板（chat template）应用于消息。
- **使用 Tokenizer 进行数据集处理**：`process_dataset` 函数接收一个包含 `full_topic` 和 `messages` 的 `example` 字典。
   - 它调用 `tokenizer.apply_chat_template` 将消息格式化为 `chat_format`，不进行 Tokenization 或添加生成提示（generation prompt），然后返回一个包含原始 `full_topic` 和新 `chat_format` 的字典。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1349337431417159701)** (44 messages🔥): 

> `Agent name variable corruption, Unit 2.3 Availability (LangGraph), Quiz access issues, Local models with smolagents, HF Channel Access` 


- **Agent 名称变量损坏导致编码灾难**：一位用户发现，用调用结果覆盖 `agent_name` 变量会导致 Agent 变得无法调用，引发了关于预防策略的讨论。
   - 遗憾的是，目前尚未提出变通方法或保护措施。
- **LangGraph 内容仍未上线**：用户们正急切等待 Unit 2.3 的发布，该单元涵盖 **LangGraph**，原定于 3 月 11 日发布。
   - 正如一位用户调侃道：*LangGraph 的 Unit 2.3 原定于 3 月 11 日发布，但我还没在网站上看到。*
- **测验登录引发困扰**：多位用户报告了登录 Unit 1 最终测验时遇到的问题，出现了错误，这可能自第一单元发布以来就一直存在。
   - 目前尚未找到变通方法或解决方案，但一位用户建议在 [Discord 频道](https://discord.com/channels/879548962464493619/1339556954162462851)中搜索答案。
- **本地模型解放无限语言学习**：用户在仅调用几次 Qwen 推理 API 后，使用默认的 `hfApiModel` 时遇到了 **payment required**（需要付费）错误。
   - 一位用户分享了在 `smolagents` 中通过 `litellm` 和 `ollama` 使用本地模型的代码片段，使用 `LiteLLMModel` 并指定 `pip install smolagents[litellm]`，然后调用 `localModel = LiteLLMModel(model_id="ollama_chat/qwen2.5:14b", api_key="ollama")`。
- **Hugging Face 频道访问受阻？**：一位用户询问频道访问受限的问题，尽管已经进行了验证，但只能看到 Agents 频道。
   - 另一位用户指出需要特定的 **role**（角色）才能访问所有频道，并暗示重定向循环问题可能阻止了角色分配。


  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/)** (1 messages): 

lunarflu: 感谢反馈！对未来的什么内容特别感兴趣吗？
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1349508858917294163)** (2 条消息): 

> `Gemma 3, Reka Flash 3, Llama 3.1 Swallow 70B, Multimodality, Vision-language input` 


- ****Gemma 3** 加入 OpenRouter 生态**: Google 在 OpenRouter 上推出了 **Gemma 3**，这是一个支持 vision-language 输入和文本输出的多模态模型，支持高达 **128k tokens** 的上下文窗口，并能理解超过 **140 种语言**。它提升了数学、推理和聊天能力，包括结构化输出和 function calling，是 [Gemma 2](https://openrouter.ai/google/gemma-2-27b-it) 的继任者。
   - 该模型可[免费](https://openrouter.ai/google/gemma-3-27b-it:free)访问。
- ****Reka Flash 3** —— 闪耀登场**: Reka 发布了 **Flash 3**，这是一个拥有 210 亿参数的语言模型，在通用聊天、编程和 function calling 方面表现出色，具备 **32K 上下文长度**，并通过强化学习 (**RLOO**) 进行了优化，权重遵循 [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)。
   - 该模型已[免费](https://openrouter.ai/rekaai/reka-flash-3:free)发布，主要是一个英文模型。
- ****Llama 3.1 Swallow 70B** 飞入视野**: 一个名为 [Llama 3.1 Swallow 70B](https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3) 的新型超快**具备日语能力的模型**加入 OpenRouter，扩展了平台的语言能力。
   - 这一新增模型与 **Reka Flash 3** 和 **Google Gemma 3** 的发布相辅相成，增强了 OpenRouter 上可用语言处理工具的多样性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1899941373530227170">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 今日新模型：Reka Flash 3, Google Gemma 3。两个更小但高性能的模型，全部免费！ 🎁</li><li><a href="https://openrouter.ai/google/gemma-3-27b-it:free))">Gemma 3 27B - API、供应商、统计数据</a>: Gemma 3 引入了多模态，支持 vision-language 输入和文本输出。它处理高达 128k tokens 的上下文窗口，理解超过 140 种语言，并提供改进的数学、推理...</li><li><a href="https://openrouter.ai/rekaai/reka-flash-3:free))">Flash 3 - API、供应商、统计数据</a>: Reka Flash 3 是一个由 Reka 开发的通用、经过指令微调的 210 亿参数大语言模型。它擅长通用聊天、编程任务、指令遵循和 function calling...</li><li><a href="https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3):">Discord</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1349348599913512993)** (85 messages🔥🔥): 

> `Gemini 2 Flash, Gemma Models, Chutes Provider, Provider Routing, Qwen finetune issues` 


- **Gemini 2 Flash 提供原生图像输出用于实验**：Google 正在通过 Gemini API 和实验版本 ([gemini-2.0-flash-exp](https://aistudio.google.com/prompts/new_chat?model=gemini-2.0-flash-exp))，在 **Google AI Studio** 目前支持的所有地区开放 **Gemini 2.0 Flash** 的原生图像输出供开发者实验。
   - 它结合了多模态输入、增强的推理和自然语言理解，能够根据文本和图像共同创建图像；它可以讲述一个故事并用图片进行插图，同时保持角色的一致性。
- **OpenRouter 目前通过 Chutes 提供免费推理**：**Chutes** 提供商目前对 OpenRouter 用户免费，因为他们正在准备服务并扩大规模；他们目前还没有完全实现的支付系统，因此在准备好支付功能之前，将继续通过 OpenRouter 免费提供。
   - 需要注意的是，他们并没有明确利用你的数据进行训练，但由于这是一个去中心化的计算提供商，OpenRouter 无法保证计算主机不会对你的数据进行某些操作。
- **OpenRouter Provider Routing 为请求提供自定义选项**：OpenRouter 会将请求路由到适用于你模型的最佳可用提供商，但用户可以使用请求体中的 `provider` 对象为 [Chat Completions](/docs/api-reference/chat-completion) 和 [Completions](/docs/api-reference/completion) 自定义路由方式。
   - 默认情况下，请求在顶级提供商之间进行负载均衡，以最大限度地提高运行时间并优先考虑价格，但如果你对吞吐量比价格更敏感，可以使用 `sort` 字段明确优先考虑吞吐量。
- **用户报告 Qwen finetune 出现异常**：一位用户报告 **Qwen finetune** 出现异常并开始无休止地输出乱码，在他们终止脚本后，该调用并未出现在 OpenRouter 的活动页面上。
   - 该用户担心它可能会输出 32k tokens 的垃圾内容并为此计费。
- **用户讨论原生图像生成访问权限和 Gemma 模型性能**：一些用户已经获得了原生图像生成的访问权限，而另一些用户仍在等待，一位用户调侃道：*4o 不是也应该有原生图像输出吗，结果他们从来没发布过，哈哈*。
   - 一位用户认为 **Gemma 3 27B** 模型*还算不错*，表示在本地使用时更倾向于它而非 **QwQ 32b**，因为 **QwQ** 倾向于在输出结果之前先输出推理过程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.0-flash-thinking-exp:free">Gemini 2.0 Flash Thinking Experimental 01-21 (free) - API, Providers, Stats</a>：Gemini 2.0 Flash Thinking Experimental (01-21) 是 Gemini 2 的快照版本。通过 API 运行 Gemini 2.0 Flash Thinking Experimental 01-21 (free)。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何通过 OpenRouter 的 Provider Routing 优化成本、性能和可靠性。</li><li><a href="https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/">Experiment with Gemini 2.0 Flash native image generation</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1349390077385773107)** (1 messages): 

> `Distill 见面会，可解释性 AI 阅读小组` 


- **Distill 社区启动每月见面会**：由于参与人数众多，**Distill** 社区正在启动每月见面会，下一次定于 **美国东部时间 3 月 14 日上午 11:30 - 下午 1:00**。
   - 详情可以在 [Exploring Explainables Reading Group 文档](https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym) 中找到。
- **探索可解释性 AI：阅读小组正在组建**：一个专注于 **可解释性 AI (XAI)** 的每月阅读小组正在组建，这源于最近一次 Distill 见面会引发的兴趣。
   - 该小组旨在深入研究 **XAI** 的复杂性，正如其 [会议文档](https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym) 中所述。



**提到的链接**：<a href="https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym)">Exploring Explainables Reading Group</a>：欢迎来到 Exploring Explainables 阅读小组！我们使用此文档来记录阅读内容、在会议期间做笔记，并让更多人对交互式科学传播产生兴趣...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1349336314671005727)** (73 messages🔥🔥): 

> `TTT 加速，Decoder-only 架构扩展，恒定熵期望，AIME 24 评估` 


- **TTT 加速模型引导 (Model Priming)**：成员们讨论了 **TTT** 如何通过执行单次梯度下降过程，加速为给定提示词 (prompt) 引导模型的过程，使模型状态更具接收性。
   - 该模型旨在学习并为每个 token 执行多次梯度下降，优化序列压缩以产生有用的表示，从而增强 **ICL** 和 **CoT** 能力。
- **Decoder-Only 架构扩展动态计算**：一个微型提案建议将 encoder-decoders 中的概念重新引入到 decoder-only 架构中，利用 decoder 端进行动态计算，通过 **TTT-like layer** 扩展序列长度以进行内部“思考”。
   - 挑战在于确定额外的采样步数，但测量 **TTT 更新损失的增量 (delta)** 并在低于中值时停止可能会有所帮助，此外还可以使用预先学习的特定领域代理 (proxies) 来估计相对数据难度。
- **实现恒定熵期望 (Constant Entropy Expectation)**：一位成员讨论了使用 TTT 风格的层如何关联输出 token，导致“新信息”减少并形成压缩状态，并由下一个 TTT 风格层平衡，这可能解决了如何处理具有不同难度的输入流 token 的问题。
   - 他们指出外部 **CoT** 虽然有效，*但感觉真的有点像权宜之计 (bandaid)*。
- **AIME 24 评估：深度探讨**：当 **QwQ** 和 **DeepSeek** 提到 AIME 24 评估时，他们可能使用的是 [AoPS wiki solutions](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems)，这被认为是数学竞赛的权威来源。
   - 他们还提到了一个相关的 [huggingface.co/papers/2503.08638](https://huggingface.co/papers/2503.08638)，该论文与歌词到歌曲领域相关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2207.07061">Confident Adaptive Language Modeling</a>：基于 Transformer 的大型语言模型 (LLMs) 最近的进展在许多任务上带来了显著的性能提升。这些提升伴随着模型规模的急剧增加...</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>：我们研究了一种新型语言模型架构，该架构能够通过在潜空间 (latent space) 中进行隐式推理来扩展测试时计算。我们的模型通过迭代循环块来工作，从而展开...</li><li><a href="https://huggingface.co/papers/2503.08638">Paper page - YuE: Scaling Open Foundation Models for Long-Form Music Generation</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1349411952862040074)** (5 messages): 

> `AIME24 implementation in lm-eval-harness, math_verify utility, Multilingual perplexity evals` 


- **AIME24 实现现身**：一名成员宣布在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/lm_eval/tasks/aime24) 中实现了 **AIME24**，该实现基于 **MATH** 的实现方式。
   - 他们提到由于缺乏关于运行 **AIME24** 时具体执行内容的文档，他们*还没有时间进行测试*。
- **`math_verify` 工具来救场！**：一名成员建议使用 `math_verify` 工具，并展示了一个包含该库中 `parse` 和 `verify` 函数的代码片段。
   - 另一名成员对该工具表示兴奋，并询问是否可以利用它来*更广泛地统一数学任务的实现*。
- **寻求多语言 Perplexity 评估！**：一名成员询问 **lm-eval-harness** 中是否提供*多语言 Perplexity 评估*。
   - 他们还征求了适合此用途的高质量*多语言数据集*的建议，这为潜在的扩展开辟了道路。



**提及链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/">GitHub - EleutherAI/lm-evaluation-harness at aime24</a>：一个用于语言模型 Few-shot 评估的框架。 - GitHub - EleutherAI/lm-evaluation-harness at aime24

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

cappuccinoislife: 大家好
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1349340901289361481)** (5 messages): 

> `VectorAdd issues, GPU programming mantra, Triton community meetup URL` 


- **VectorAdd 返回零，随后奇迹发生**：一位用户报告他们的 **vectoradd 提交**返回全零，但随后编辑了消息称*现在可以工作了。也许是神迹显灵*。
   - 随后，该用户透露代码存在一个 Bug，导致重复处理同一个 Block，从而给出了极高（但错误）的吞吐量，目前已修复。
- **Sadhguru 的 GPU 编程准则**：一名成员调侃道，GPU 编程有一条准则：*如果它太快了，可能哪里就有 Bug*。
   - 这被归功于 **Sadhguru**。
- **Triton 社区见面会正在直播**：一名成员询问 **Triton 社区见面会 URL**，另一名成员发布了名为 *Triton community meetup March 2025* 的 [YouTube 视频](https://www.youtube.com/watch?v=bxBZB0DuS7s&ab_channel=BillYoshimi)。
   - 视频描述中包含一个 [StreamYard 折扣链接](https://streamyard.com/pal/d/6451380426244096)。



**提及链接**：<a href="https://www.youtube.com/watch?v=bxBZB0DuS7s&ab_channel=BillYoshimi">Triton community meetup March 2025</a>：🎙️ 想要开始直播或提升水平？查看 StreamYard 并获得 10 美元折扣！😍 https://streamyard.com/pal/d/6451380426244096

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1349344000527564841)** (24 messages🔥): 

> `funnel shift performance, variable rate compression, trellis scheme, tensor fragments, predicated funnel shift` 


- **Funnel Shift 性能令工程师惊讶**：成员们讨论了 **funnel shift** 是否优于将 `a` 和 `b` 放入 `uint64_t` 并进行移位，例如 `uint64_t u = (uint64_t(a) << 32) | b; return (u >> shift) & 0xFFFF;`。一名成员指出，至少在 **H100** 上，前者似乎更快。
   - 另一名成员表示，他们对 **funnel shift** 速度更快感到惊讶，并猜测它可能使用了较不拥挤的流水线（Pipe），但补充说这可能取决于周围的代码。
- **Trellis Scheme 量化策略现身**：一名成员描述了他们在可变速率压缩中使用 **trellis scheme**（重叠位域）的情况，其中一个 **16x16 权重** Tile 由 **256*K bits** 表示，每个权重使用其中的 **16 bits**；例如，权重 0 是 bits **[0:16]**，权重 1 是 bits **[3:19]** 等（当 **K=3** 时）。
   - 他们还提到，可以在量化前对每个 Tile 进行置换，以便直接反量化到 **tensor fragments** 中。
- **尽管有 PRMT 替代方案，Funnel Shift 依然胜出**：一名成员尝试在移位为 8 的倍数的情况下使用 `prmt` 指令，但发现一致使用谓词化（predicated）的 funnel shift 性能更好，这可能是由于 Kernel 中的其他活动导致的。
   - 原始代码被翻译为 **4 个 `shf.r.u32`**、**3 个 `shf.r.w.u32`** 和 **7 个 `lop3.lut` SASS 指令**，目标架构为 **sm_70/80/90**。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1349499214161514618)** (2 messages): 

> `UT Austin Deep Learning Lectures, TensorFlow OpenCL Flame War` 


- **UT Austin 深度学习讲座在线发布**：来自 UT Austin 的高质量且相关的深度学习讲座现已在 [ut.philkr.net/advances_in_deeplearning/](https://ut.philkr.net/advances_in_deeplearning/) 公开。
   - 链接材料包括幻灯片以及对 **深度网络结构**、**训练** 和 **现代 GPU 架构** 的介绍。
- **回顾 TensorFlow 的 OpenCL 支持之争**：一位成员分享了一个关于 [TensorFlow 中 OpenCL 支持](https://github.com/tensorflow/tensorflow/issues/22) 的 *2015 年有趣论战* 链接。
   - 该讨论突出了早期对 **CUDA** 的关注以及在引入 **OpenCL** 支持时面临的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ut.philkr.net/advances_in_deeplearning/">UT Austin - Advances in Deep Learning</a>：未找到描述</li><li><a href="https://github.com/tensorflow/tensorflow/issues/22">OpenCL support · Issue #22 · tensorflow/tensorflow</a>：我了解到 TensorFlow 仅支持 CUDA。要添加 OpenCL 支持需要做哪些工作？
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1349355509873905714)** (9 messages🔥): 

> `GPU Architecture Books, PMPP Alternatives, CUDA mock interviews` 


- **寻找 GPU 架构书籍**：一位成员征求适合初学者的 **GPU 架构** 和 **编程** 推荐书籍，并特别要求寻找热门书籍 *Programming Massively Parallel Processors* 的替代方案。
   - 另一位成员询问了提问者的计算机架构背景以及寻找替代方案的原因，因为 *这里的每个人可能都认为它是最好的书！*。
- **《Programming Massively Parallel Processors》是圣经**：一位成员推荐将 *Programming Massively Parallel Processors (PMPP)* 作为 **GPU 编程** 的首选书籍，称其为 *GPU 编程圣经*。
   - 提问者已经拥有这本书并觉得 *还可以*，但由于难以理解其中的关键概念，因此正在寻找替代方案。
- **Cpp/CUDA 模拟面试**：频道中的一位成员问道：*嘿，有人想进行 Cpp/CUDA 的模拟面试吗？*
   - 似乎没有人回应。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1349371533121814611)** (3 messages): 

> `Float8 Conv, INT8 Conv, Static Quantization` 


- **torchao 缺少 Float8 CUDA 内核**：一位成员提到 **float8 conv** 需要一个 **CUDA/TK/Triton 内核**，这将是 *torchao* 的一个不错补充。
- **INT8 Conv 性能担忧**：一位成员表示他们之前从 *torch inductor 模板* 中拼凑了一个 **INT8 conv**，内核性能尚可，但将激活动态量化为 INT8 的成本太高，导致端到端（e2e）没有看到明显的加速。
- **静态量化需求**：一位成员指出可能需要 **静态量化（Static Quantization）**，即 *激活的 scale 和 zero point 是根据校准数据提前确定的*。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1349361270440198145)** (2 messages): 

> `FlashAttention for Turing, Weight absorption for MLA` 


- **FlashAttention 登陆 Turing 架构！**：一位成员分享了他们为 [Turing 架构实现的 FlashAttention 前向传播](https://github.com/ssiu/flash-attention-turing)，并指出目前支持 `head_dim = 128`、原生 Attention，且 `seq_len` 可被 128 整除。
   - 据报道，在 **T4** 上测试 `batch_size = 4`、`num_heads = 32`、`head_dim = 128` 时，该实现比 PyTorch 的 `F.scaled_dot_product_attention` 快 **2 倍**。
- **权重吸收 (Weight Absorption) 技巧优化 MLA**：一位成员分享了一篇关于 [权重吸收的博客文章](https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention)，以确保 **Multi-Head Latent Attention (MLA)** 的高效实现。
   - 文章指出，**MLA** 是助力 DeepSeek AI 的 **V3** 和 **R1** 模型的一项关键创新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DataCrunch_io/status/1899883311612186990">来自 DataCrunch_io (@DataCrunch_io) 的推文</a>：⚡️Multi-Head Latent Attention 是助力 @deepseek_ai 的 V3 及其后续 R1 模型的一项关键创新。⏭️ 加入我们，继续我们的高效 AI 推理系列，涵盖...</li><li><a href="https://github.com/ssiu/flash-attention-turing">GitHub - ssiu/flash-attention-turing</a>：通过在 GitHub 上创建账号来为 ssiu/flash-attention-turing 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1349377876658487308)** (2 messages): 

> `Memory Allocation Issues in H100, ThunderKittens Kernel Modifications, Tensor Concatenation Alternatives` 


- **H100 Kernel 修改中的内存故障**：一位成员询问，为什么在 [h100.cu](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/h100/h100.cu#L71) 中修改内存分配以直接为 `o_smem` 分配内存时，会导致“illegal memory access was encountered”错误。
   - 该询问重点在于理解在指定的 **H100 GPU kernel** 中切换到直接内存分配时产生此错误的原因。
- **张量难题：在 Kernel 设计中避免拼接**：一位成员正在编写一个 Kernel，该 Kernel 将 `q` 作为两个独立的张量接收以避免拼接，因为拼接会使转换到 `o` 变得复杂。
   - 他们正在寻求关于在 [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 项目的 Kernel 设计中管理张量输入的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens/">GitHub - HazyResearch/ThunderKittens: 用于快速 Kernel 的 Tile 原语</a>：用于快速 Kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/h100/h100.cu#L71">ThunderKittens/kernels/attn/h100/h100.cu at main · HazyResearch/ThunderKittens</a>：用于快速 Kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1349452513165967455)** (1 messages): 

> `` 


- **功能基础头脑风暴**：一位用户建议为某个功能构建基础，并艾特了另一位用户来考虑这个想法。
- **构思功能基石**：一位成员提议构思并构建特定功能的底层框架。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1349439049219440701)** (2 messages): 

> `Modal Runners success, Leaderboard submissions` 


- **Modal Runners 成功完成向量加法**：ID 为 **1946** 的测试提交在 **T4** GPU 上使用 **Modal runners** 成功运行了 `vectoradd` 排行榜任务！
   - 这表明在指定的硬件和平台上成功执行并验证了向量加法任务。
- **更多 Modal 魔法：向量加法再获胜利！**：ID 为 **1947** 的测试提交在 **T4** GPU 上使用 **Modal runners** 成功运行了 `vectoradd` 排行榜任务！
   - 这是 Modal runners 的又一次胜利，进一步巩固了其在 GPU 加速计算方面的可靠性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1349362882072023124)** (44 条消息🔥): 

> `Gemma 3 Models, AlphaXiv, Gemini 2.0 Flash, Open Weight Models, DeepMind Robotics` 


- **Gemma 3 在创意写作中位居第二**：根据[这条推文](https://x.com/sam_paech/status/1899772582808969653)，**Gemma-3-27b** 模型在创意写作中排名第二，很可能成为创意写作和 RP（角色扮演）微调者的首选。
- **AlphaXiv 使用 OCR 和 Claude 3.7 概述 ArXiv 论文**：**AlphaXiv** 结合使用 **Mistral OCR** 和 **Claude 3.7** 为 arXiv 论文创建博客风格的概述。如[这条推文](https://fxtwitter.com/askalphaxiv/status/1899833509033976194)所述，只需点击一下，即可生成包含图表、关键见解和清晰解释的研究博客。
- **Gemini 2.0 Flash 首次推出原生图像生成功能**：**Gemini 2.0 Flash** 现在具备原生图像生成功能，并针对对话迭代进行了优化。根据[这篇博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)，用户可以创建与上下文相关的图像，通过对话进行编辑，并在图像中生成长文本。
- **Open Weight 模型降低利润并增强隐私**：像 **Gemma 3** 这样的 Open Weight 模型主要用于营销/招聘，同时降低了 API 平台的利润空间。但由于**隐私/数据原因**，它们的使用率正在增加，一些人预测未来将出现垂直化、经过 RL 微调的模型。
- **DeepMind 推出 Gemini Robotics**：根据 [DeepMind 博客](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)，**DeepMind** 推出了 **Gemini Robotics**，这是一个基于 **Gemini 2.0** 的机器人模型，旨在通过物理领域内文本、图像、音频和视频的多模态推理来解决复杂问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/">Introducing Gemini Robotics and Gemini Robotics-ER, AI models designed for robots to understand, act and react to the physical world.</a>：介绍 Gemini Robotics 和 Gemini Robotics-ER，这是专为机器人理解、行动和对物理世界做出反应而设计的 AI 模型。</li><li><a href="https://fxtwitter.com/askalphaxiv/status/1899833509033976194">Tweet from alphaXiv (@askalphaxiv)</a>：我们使用 Mistral OCR 和 Claude 3.7 为 arXiv 论文创建博客风格的概述。只需点击一下，即可从论文中生成包含图表、关键见解和清晰解释的精美研究博客...</li><li><a href="https://x.com/OriolVinyalsML/status/1899853815056085062">Tweet from Oriol Vinyals (@OriolVinyalsML)</a>：Gemini 2.0 Flash 首次推出原生图像生成！创建与上下文相关的图像，通过对话进行编辑，并在图像中生成长文本。全部针对对话迭代进行了优化。请在 AI Studio 中尝试...</li><li><a href="https://x.com/isidentical/status/1899870537964544376">Tweet from batuhan the fal guy (@isidentical)</a>：http://imgsys.org 中出现了一个新的、潜在的 SOTA 模型 👀👀👀</li><li><a href="https://x.com/btibor91/status/1899852454751014981">Tweet from Tibor Blaho (@btibor91)</a>：Gemini 2.0 Flash 原生图像输出从今天（2025 年 3 月 12 日）开始开放公共实验访问。</li><li><a href="https://ai.google.dev/gemma/terms">no title found</a>：未找到描述</li><li><a href="https://x.com/kalomaze/status/1899859237716844564">Tweet from kalomaze (@kalomaze)</a>：gemma3 27b 是一个极其强大的基础模型。那 77 的 MMLU 并不是刷榜（benchmaxxing）的结果，@teortaxesTex</li><li><a href="https://x.com/sam_paech/status/1899772582808969653">Tweet from Sam Paech (@sam_paech)</a>：Gemma-3-27b 在创意写作中位居第二。预计这将成为创意写作和 RP 微调者的另一个心头好。</li><li><a href="https://web.archive.org/web/20190124204600/https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/">AlphaStar: Mastering the Real-Time Strategy Game StarCraft II | DeepMind</a>：星际争霸（StarCraft）被认为是最具挑战性的即时战略游戏之一，也是历史上最长盛不衰的电竞项目之一，已成为 AI 研究公认的“重大挑战”。这里...</li><li><a href="https://archive.is/KhFss">Specification gaming: the flip side of AI ingenuity | DeepMind</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1349441344917213204)** (2 messages): 

> `Copyright law, Machine Learning, Privacy, Verbatim output` 


- **围绕生成式模型的版权法**：根据 [Nicholas Carlini 的博客文章](https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html)，目前有大量诉讼案件在探讨**在受版权保护的数据上训练生成式 Machine Learning 模型**本身是否构成侵权。
- **表现出逐字召回（Verbatim Recall）的 Machine Learning 模型**：版权案件中的律师正引用 Nicholas Carlini 关于 Machine Learning 模型输出逐字训练样本（[文本](https://arxiv.org/abs/2012.07805)和[图像](https://arxiv.org/abs/2301.13188)）的论文，作为模型是否违反版权的证据。
- **Stable Diffusion 模型召回训练数据**：根据 [Nicholas Carlini 的博客文章](https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html)，一张 **Stable Diffusion** 训练过的图像，通过对其进行查询，可以从模型中被“提取”出完全相同的图像。



**提到的链接**：<a href="https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html">
      What my privacy papers (don't) have to say about copyright and generative AI
    </a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1349439300667834502)** (2 messages): 

> `Content Filters, Claude Code` 


- **用户称内容过滤器是 AI 的灾难**：一位用户分享了一篇[帖子](https://fxtwitter.com/mgostIH/status/1899876994348954026)，声称*内容过滤器（Content Filters）对 AI 来说是一场灾难*。
   - 该用户还表示 **Claude Code** 已经具备了这一特性。
- **Claude Code 内容过滤器**：一位用户提到 **Claude Code** 已经实现了内容过滤器，暗示其与其他 AI 系统形成对比。
   - 这表明，虽然有些人对内容过滤器持负面看法，但其他人则认为它们是必要的或已经集成的组件。



**提到的链接**：<a href="https://fxtwitter.com/mgostIH/status/1899876994348954026">来自 mgostIH (@mgostIH) 的推文</a>：内容过滤器对 AI 来说是一场灾难

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1349474808970219561)** (1 messages): 

> `Elicitation Theory, Deep Learning` 


- **Arjun Srivastava 将 Deep Learning 比作耕作**：一位成员分享了 Arjun Srivastava 撰写的帖子链接，标题为“[论 Deep-Learning 与耕作](https://open.substack.com/pub/arjunsriva/p/on-deep-learning-and-farming?r=68gy5&utm_medium=ios)”，该文探讨了不同领域概念之间的映射。
   - 文章假设有两种制造事物的方式：**工程（Engineering）**（有意识地理解和组合子组件）和**培育（Cultivation）**（无法直接构建的事物，比如向日葵）。
- **工程 vs 培育**：作者将**工程**（组件被刻意组装）与**培育**（无法直接建造）进行了对比。
   - *培育*就像务农，而*工程*就像制作桌子。



**提到的链接**：<a href="https://open.substack.com/pub/arjunsriva/p/on-deep-learning-and-farming?r=68gy5&utm_medium=ios">On Deep Learning and Farming: It&#x27;s still 1915</a>：关于农业能为 AI 开发提供哪些启示

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot 新闻：<@&1216534966205284433>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1349471895195615243)** (48 条消息🔥): 

> `模型大小与智能，GPT4All 与 Ollama 在服务器模型管理方面的对比，Deepseek 14B 或 7B vs Llama 8B，大上下文窗口模型，GPT4All 对 Gemma 3 的限制` 


- **更大的模型拥有更好的基准测试表现**：一位用户询问为什么 **ChatGPT premium** 的表现优于 **GPT4All** 的 **LLM**，另一位用户回答说，通常更大的模型表现出更高的智能。
   - 他们建议，在硬件条件允许的情况下，可以从 **Hugging Face** 下载在尺寸和能力上超过入门级模型的模型。
- **服务器解决方案中 Ollama 优于 GPT4All？**：一位用户询问是否可以使用 **GPT4All** 作为服务器来管理多个模型，包括快速加载/卸载、针对定期更新文件的 **RAG**，以及用于日期/时间/天气的 **API**。
   - 该用户提到了 **Ollama** 的一些问题，并针对其在中低算力情况下的需求寻求适用性建议。
- **Deepseek 详情：14B 是首选**：当被问及哪种模型相当于 **ChatGPT premium** 时，一位用户建议在拥有 64GB **RAM** 的情况下使用 **Deepseek 14B** 或类似模型。
   - 建议先从 **Deepseek 7B** 或 **Llama 8B** 等较小的模型开始，如果系统运行顺畅再进行升级。
- **上下文是关键：4k 还可以**：建议寻找具有大上下文窗口（超过 **4k tokens**）的模型，以便在提示词中容纳更多信息（如文档）。
   - 随后该成员询问他们发布的一张截图是否属于这类模型。
- **Gemma 代际差距：GPT4All 的小故障**：一位用户建议尝试在 **GPT4All** 中使用微型模型来检查加载、卸载和 **RAG**（配合 **LocalDocs**）的工作流，并指出 **GUI** 不支持同时运行多个模型。
   - 他们提供了一张图片，指出 **Gemma 3** 尚不兼容 **GPT4All**，需要更新版本的 **llama.cpp**。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1349336897520144384)** (30 条消息🔥): 

> `Glama AI, MCP Logging, Claude Image Rendering, NPM Package Storage, MCP Server Connection Status` 


- **Glama API 数据转储**：一位成员分享了 [Glama AI](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers) 推出了一个新的 API，可以列出所有工具，比 Pulse 提供的每个服务器的数据更多。
   - 成员们对这些丰富的免费信息感到兴奋。
- **MCP 日志记录：服务器视角**：一位成员询问了关于使用 Python SDK 进行日志记录的问题，参考了直接记录到 `/library/logs/claude` 的方式，另一位成员澄清说，服务器是根据 [Model Context Protocol (MCP) 规范](https://spec.modelcontextprotocol.io/specification/2024-11-05/server/utilities/logging/) 发送日志消息的。
   - MCP 允许服务器声明 `logging` 能力，并发出带有严重性级别、可选 logger 名称和 JSON 可序列化数据的日志消息。
- **Claude 如何渲染图像？**：一位成员询问了一个返回图像对象（特别是 Plotly 图像）的 **Claude MCP server** 示例。
   - 另一位成员指向了一个 [wolfram server](https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35) 示例，该示例获取渲染后的图表，通过对数据进行 base64 编码并设置 mime 类型来返回图像，但也指出 Claude 在工具调用窗口之外进行渲染存在局限性。
- **NPM 包位置揭晓！**：一位成员询问客户端将 NPM 包/源代码存储在哪里，以及如果客户端再次请求，是否会从缓存中访问。
   - 另一位成员指出是在 `%LOCALAPPDATA%`，且 NPM 包位于 `C:\Users\YourUsername\AppData\Local\npm-cache`。
- **MCP 服务器：状态如何？**：一位成员询问如何在他们的客户端中显示哪些服务器已下载以及是否已连接，类似于 Cursor 的做法。
   - 另一位成员表示，确定哪些已下载并不直观，但可以通过正则表达式枚举名称中包含 **mcp** 的文件夹，并且需要编写客户端逻辑来检查连接状态。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp/reference#tag/servers/GET/v1/servers">MCP API Reference</a>: Glama Gateway 的 API 参考</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/server/utilities/logging/">Logging</a>: ℹ️ 协议修订：2024-11-05。Model Context Protocol (MCP) 为服务器向客户端发送结构化日志消息提供了一种标准化的方式。客户端可以控制...</li><li><a href="https://github.com/tadasant/mcp-server-stability-ai/blob/357448087fc642b29d5c42449adce51812a88701/src/tools/generateImage.ts#L129-L132">mcp-server-stability-ai/src/tools/generateImage.ts</a>: 集成 MCP 客户端与 Stability AI 驱动的图像处理功能（生成、编辑、放大等）的 MCP Server。</li><li><a href="https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35>">MCP-wolfram-alpha/src/mcp_wolfram_alpha/server.py</a>: 将你的聊天 repl 连接到 Wolfram Alpha 计算智能。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1349424701151379467)** (8 条消息🔥): 

> `MCP Agent, OpenAI Agent SDK, MCP Servers, unRAID MCP server, MCP Fathom Analytics` 


- **GitHub 上的 **MCP Agent**！**: 一位成员分享了 [MCP Agent GitHub 仓库](https://github.com/lastmile-ai/mcp-agent) 的链接，该项目专注于使用 Model Context Protocol 和简单的工作流模式构建高效的 Agent。
- ****OpenAI Agents SDK** 获得 MCP 支持**: 一位成员宣布他们为 **OpenAI Agents SDK** 添加了 MCP 支持，该支持以 [GitHub 上的 fork](https://github.com/lastmile-ai/openai-agents-mcp) 形式提供，并在 pypi 上发布为 *openai-agents-mcp* 包，允许 Agent 聚合来自 MCP 服务器的工具。
   - 通过设置 `mcp_servers` 属性，Agent 可以通过统一的语法无缝使用 **MCP 服务器**、本地工具和 **OpenAI 托管的工具**。
- **unRAID 获得 **MCP 服务器**！**: 一位成员分享了 [GitHub 上的 unRAID MCP 服务器](https://github.com/jmagar/unraid-mcp) 链接。
- ****Fathom Analytics** 获得 MCP 服务器！**: 一位成员分享了 [Fathom Analytics 的 MCP 服务器](https://github.com/mackenly/mcp-fathom-analytics) 链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=43345172">Show HN: MCP-Compatible OpenAI Agents SDK | Hacker News</a>: 未找到描述</li><li><a href="https://github.com/lastmile-ai/openai-agents-mcp">GitHub - lastmile-ai/openai-agents-mcp: A lightweight, powerful framework for multi-agent workflows</a>: 一个轻量级、功能强大的多 Agent 工作流框架 - lastmile-ai/openai-agents-mcp</li><li><a href="https://github.com/mackenly/mcp-fathom-analytics">GitHub - mackenly/mcp-fathom-analytics: MCP server for Fathom Analytics</a>: Fathom Analytics 的 MCP 服务器。通过在 GitHub 上创建账户来为 mackenly/mcp-fathom-analytics 的开发做出贡献。</li><li><a href="https://github.com/jmagar/unraid-mcp">GitHub - jmagar/unraid-mcp</a>: 通过在 GitHub 上创建账户来为 jmagar/unraid-mcp 的开发做出贡献。</li><li><a href="https://github.com/lastmile-ai/mcp-agent">GitHub - lastmile-ai/mcp-agent: Build effective agents using Model Context Protocol and simple workflow patterns</a>: 使用 Model Context Protocol 和简单的工作流模式构建高效的 Agent - lastmile-ai/mcp-agent
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1349338814270672929)** (37 条消息🔥): 

> `Codeium Extension Issues, Protocol Errors, Neovim Support, VPN Workaround` 


- **Codeium 扩展受协议错误困扰**: 用户报告在使用 VSCode 扩展时遇到 **协议错误 (protocol errors)**，例如 *"invalid_argument: protocol error: incomplete envelope: read tcp... forcibly closed by the remote host"*，且 Codeium 页脚变为红色。
   - 该问题似乎影响了 **英国** 和 **挪威** 使用 **Hyperoptic** 和 **Telenor** 等各种互联网服务提供商的用户。
- **Neovim 支持匮乏**: 一位用户对 **Neovim 支持** 相比 Windsurf 滞后表示失望，并提到了错误代码为 500 的补全错误。
   - 他们询问这些问题是否会很快修复，或者是否应该切换到另一个插件，对此一位团队成员回应称 *团队正在处理中*。
- **测试修复已部署，结果褒贬不一**: 团队部署了一个测试修复补丁，一些用户报告更新后错误减少了。
   - 然而，其他人仍然面临问题，扩展程序 *"关闭"* 或保持红色，促使团队进一步调查。
- **团队承认欧盟地区存在问题**: 团队承认该问题主要影响 **欧盟 (EU)** 用户，问题包括自动补全时的 *"unexpected EOF"* 以及无法在聊天中链接文件。
   - 建议的一个临时解决方案是使用 **VPN** 并连接到 **洛杉矶 (Los Angeles)**。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1349346835503906826)** (14 条消息🔥): 

> `YC 模仿型初创公司，麦克斯韦妖与慢速 AI，自适应元学习项目，Sakana AI 科学家 LLM 扩展` 


- **YC 专注于短期模仿者**：一位成员表示，**YC** 选择的是那些在短期内会成功但长期无法生存的初创公司（模仿者）。
   - 他们补充说，*YC 已经多年没有出现过著名的独角兽公司了*，所以 **YC 多年来并不成功**。
- **麦克斯韦妖（Maxwell's Demon）限制 AI 速度**：一位成员分享道，计算机可以通过双向运行以极低的能量进行计算，但速度限制取决于你运行答案的速度和确定性，并引用了[这段 YouTube 视频](https://www.youtube.com/watch?v=eS0JXViv0cU)。
   - 他们还链接了[另一段视频](https://www.youtube.com/watch?v=KR23aMjIHIY)，关于*逆转熵*。
- **寻求自适应元学习（Adaptive Meta-Learning）练手项目**：一位成员正在寻找练手项目来测试 **Meta-Transform** 和 **Adaptive Meta-Learning**，从使用 Gymnasium 的小步骤开始。
   - 他们还链接了一个关于 **Adaptive Meta-Learning (AML)** 的 [GitHub 仓库](https://github.com/EAzari/AML)。
- **通过 FSA 近似解释 LLM 扩展**：引用 [Sakana AI 的首篇出版物](https://sakana.ai/ai-scientist-first-publication/)，一位成员理论化地认为，**LLM 扩展**可以通过假设它们是近似上下文无关语言（context-free language）的概率 **FSA** 来解释。
   - 这种方法，即来自 Chomsky hierarchy 较低层级的机器尝试近似较高层级的语言，会产生特征性的 **S 曲线（S-curve）**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sakana.ai/ai-scientist-first-publication/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/vikhyatk/status/1899663773499334749?t=XUCU_6aHFeqJeCc-wVkqaQ&s=19">来自 vik (@vikhyatk) 的推文</a>：我们这一代最伟大的头脑正被文本扩散（text diffusion）和 SSMs 搞得晕头转向。这分散了对真正重要工作（清洗数据集）的注意力。</li><li><a href="https://github.com/EAzari/AML">GitHub - EAzari/AML: Adaptive Meta-Learning (AML)</a>：Adaptive Meta-Learning (AML)。通过创建账户为 EAzari/AML 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=eS0JXViv0cU">麦克斯韦妖：生命是否违反了热力学第二定律？ | Neil Gershenfeld 和 Lex Fridman</a>：Lex Fridman Podcast 完整剧集：https://www.youtube.com/watch?v=YDjOS0VHEr4 请通过查看我们的赞助商来支持本播客：- LMNT: https://drinkLM...</li><li><a href="https://www.youtube.com/watch?v=KR23aMjIHIY">用麦克斯韦妖逆转熵</a>：像您一样的观众帮助成就了 PBS（谢谢 😃）。在此支持您当地的 PBS 会员站：https://to.pbs.org/DonateSPACE 恶魔能否战胜热力学第二定律...</li><li><a href="https://www.youtube.com/watch?v=0UVa7cQo20U">图灵对计算机的错误认知 | Neil Gershenfeld 和 Lex Fridman</a>：Lex Fridman Podcast 完整剧集：https://www.youtube.com/watch?v=YDjOS0VHEr4 请通过查看我们的赞助商来支持本播客：- LMNT: https://drinkLM...</li><li><a href="https://www.youtube.com/watch?v=NppWwDzE2qk">想法从何而来？ | Neil Gershenfeld 和 Lex Fridman</a>：Lex Fridman Podcast 完整剧集：https://www.youtube.com/watch?v=YDjOS0VHEr4 请通过查看我们的赞助商来支持本播客：- LMNT: https://drinkLM...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1349431328679923833)** (6 条消息): 

> `前向与后向 SDE，反向扩散 SDE` 


- **SDE 演示定于周五**：一位成员提议在周五晚上进行关于 **SDE** 的演示，涵盖 **SDE** 的定义以及从前向噪声 **SDE** 推导 **反向扩散 SDE** 的过程。
   - 他们澄清这将是一个*即兴*演示。
- **揭秘后向 SDE**：一位成员分享了关于 **前向与后向 SDE** 的讨论，解释了后向过程涉及反转前向过程对应的 **PDE** 并将其作为 **SDE** 求解。
   - 他们附上了一份 [随机微积分 PDF](https://cdn.discordapp.com/attachments/1045297868136779846/1349457047950721034/Stochastic_Calculus_Ito_vs_Stratonovich.pdf?ex=67d32b4f&is=67d1d9cf&hm=fffee4de9330e2157fdaed7cb2975b22aa4854872d0467df1d8d8c1b2d41fa39&)。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 条消息): 

mico6424: 哪种认知架构（cognitive architecture）有值得研究的实际运行实现？
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1349407887172046889)** (7 条消息): 

> `Gemma 3, Multimodal Models, Sakana AI paper` 


- **Gemma 3 终于发布了！**: **Gemma 3** 正式发布，具备多模态能力和 **128k context window**（1B 模型除外），满足了用户的期待。
   - 一位用户简单评价道：“*还行吧 (twas aight)*”。
- **Google Gemini 机器人模型**: Google 发布了一段 [YouTube 视频](https://youtu.be/4MvGnmmP3c0) 展示 **Gemini Robotics**，将 **Gemini 2.0** 引入物理世界，作为其最先进的 vision language action model。
   - 该模型使机器人能够与物理世界进行交互。
- **Sakana AI 的 AI 生成论文通过同行评审**: 由 **Sakana AI** 生成的一篇 [论文](https://sakana.ai/ai-scientist-first-publication/) 已通过 **ICLR workshop** 的同行评审。
   - 一位用户质疑评审的严谨性，暗示该 workshop 可能对作者*比较慷慨*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-scientist-first-publication/">未找到标题</a>: 未找到描述</li><li><a href="https://ai.google.dev/gemma/docs/core">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/4MvGnmmP3c0">Gemini Robotics: Bringing AI to the physical world</a>: 我们全新的 Gemini Robotics 模型将 Gemini 2.0 带入物理世界。它是我们最先进的 vision language action model，使机器人能够进行交互...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349338140057276489)** (25 条消息🔥): 

> `Mastra AI Framework, Cursor SOTA Embedding Model, Typescript AI Apps, Gemini Native Image Generation, DeepSearch and DeepResearch` 


- **Mastra Typescript AI Framework 发布**：由前 Gatsby/Netlify 构建者宣布推出的全新 **Typescript AI framework** —— **Mastra**，旨在既能胜任趣味小项目，也能支撑生产环境，通过 [博客文章](https://mastra.ai/blog/the-next-million-ai-developers) 面向前端、全栈和后端开发者。
   - 创作者旨在通过专注于为产品开发者提供**可靠性和易用性**，来提供现有框架之外的另一种选择，并邀请社区为 [他们的项目](https://github.com/mastra-ai/mastra) 做出贡献。
- **Cursor 训练出 SOTA Embedding 模型**：根据 [一条推文](https://x.com/amanrsanger/status/1899659103473123777?s=46)，**Cursor** 在语义搜索上训练出了一个 **SOTA embedding 模型**，其表现大幅优于竞争对手使用的开箱即用的 Embedding 和 Reranker。
   - 鼓励用户在使用 Agent 时“感受差异”，以体验增强后的性能。
- **体验 Gemini 2.0 Flash 的原生图像生成**：**Google** 正在 **Gemini 2.0 Flash** 中推出原生图像生成功能，供 **Google AI Studio** 目前支持的所有地区的开发者进行实验，正如 [博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/) 中所宣布的那样。
   - 根据 [一条推文](https://x.com/19kaushiks/status/1899856652666568732?s=46)，此次更新允许开发者在 Google AI Studio 中使用 **Gemini 2.0 Flash (gemini-2.0-flash-exp)** 的实验版本以及通过 Gemini API 测试新功能，该功能结合了多模态输入、增强的推理能力和自然语言理解来创建图像。
- **Jina AI 详解 DeepSearch 实现**：**Jina AI** 发布了一篇 [博客文章](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)，提供了实现 **DeepSearch/DeepResearch** 的实用指南，重点介绍了用于片段选择的 Late-chunking Embedding，以及在爬取前使用 Reranker 对 URL 进行优先级排序。
   - 文章强调 *QPS 已过时，深度才是王道*，提倡通过“阅读-搜索-推理”循环来寻找答案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/">Snippet Selection and URL Ranking in DeepSearch/DeepResearch</a>: 搞定这两个细节能让你的 DeepSearch 从平庸走向卓越：从冗长的网页中选择最佳片段，并在爬取前对 URL 进行排序。</li><li><a href="https://x.com/19kaushiks/status/1899856652666568732?s=46">Kaushik Shivakumar (@19kaushiks) 的推文</a>: 非常激动今天能将 Gemini 的原生图像生成功能推向公开实验阶段 :) 我们取得了很大进展，但仍有很长的路要走，请给我们反馈！是的，我制作了那个 im...</li><li><a href="https://mastra.ai/blog/the-next-million-ai-developers">A framework for the next million AI developers</a>: 下一代 AI 产品将使用 Typescript 编写的 API 来构建。</li><li><a href="https://x.com/amanrsanger/status/1899659103473123777?s=46">Aman Sanger (@amanrsanger) 的推文</a>: Cursor 在语义搜索上训练了一个 SOTA embedding 模型。它大幅优于竞争对手使用的开箱即用的 Embedding 和 Reranker！你在使用 Agent 时可以感受到这种差异！</li><li><a href="https://x.com/m__dehghani/status/1899854209081868663?s=46">Mostafa Dehghani (@m__dehghani) 的推文</a>: 任何待过这个房间的人都知道，这里从来不只是普通的一天！这个空间见证了混乱与天才的极端！...而且我们发布了！https://developers.googleblog.com/en/experiment-wi...</li><li><a href="https://share.snipd.com/episode/3267b9f3-0048-42c4-8808-92fb357d097f">Sam Altman, OpenAI CEO</a>: Sam Altman, OpenAI CEO</li><li><a href="https://x.com/aidenybai/status/1899840110449111416?s=46">Aiden Bai (@aidenybai) 的推文</a>: 介绍 Same.dev。以像素级精度克隆任何网站。一键生成 Nike、Apple TV、Minecraft 等网站！</li><li><a href="https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/">Experiment with Gemini 2.0 Flash native image generation</a>: 未找到描述。
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1349410350893633629)** (1 条消息): 

> `User Research, Mobile Usage, NotebookLM on Mobile, Usability Study` 


- **研究探讨用户移动端习惯**：Google 正在招募 **NotebookLM 用户**进行 60 分钟的访谈，讨论他们的**移动端使用情况**并提供对新概念的反馈，参与者将获得 **75 美元感谢礼**或 50 美元 Google 商品代金券。
   - 有兴趣的参与者请填写筛选表单（[链接](https://forms.gle/pbPDU2Dh3rEL5HLC9)）以确定资格。
- **Google 将开展可用性研究**：Google 将于 **2025 年 4 月 2 日和 3 日**开展一项**可用性研究**，以收集对开发中产品的反馈，为参与者提供**等值 75 美元的当地货币**或 **50 美元 Google 商品代金券**。
   - 该研究需要**高速互联网连接**、**活跃的 Gmail 账号**以及**配备摄像头、扬声器和麦克风的电脑**。



**提到的链接**：<a href="https://forms.gle/pbPDU2Dh3rEL5HLC9">参与即将举行的 NotebookLM 用户研究！</a>：您好，我正通过这份简短的问卷与您联系，以核实您参加 Google 即将举行的可用性研究的资格。这次研究是一个对目前正在开发的产品提供反馈的机会...

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1349336793295618048)** (4 条消息): 

> `NoteBookLM Plus, Internal FAQ, API instructions` 


- **NoteBookLM Plus 作为内部 FAQ**：一位成员正考虑将 **NoteBookLM Plus** 用作内部 FAQ，并希望调查未解决问题的内容。
   - 他们寻求关于如何检查用户在聊天中输入但未得到解决的问题的建议。
- **NoteBookLM 擅长编写 API 脚本**：一位成员发现 **NLM+** 在使用 **API 指令**和示例程序生成脚本方面表现出色。
   - 他们指出，作为非程序员，通过引用笔记本中的材料来获取修改建议变得更加容易。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1349344576720212048)** (12 条消息🔥): 

> `RAG vs Full Context Window, Saving Chat Responses, Thinking Model Updates, Language Support` 


- **RAG 与全上下文窗口之争**：一位用户质疑，对于大型数据库，使用带有向量搜索和较小上下文窗口的 **RAG** 是否优于使用具有全上下文窗口的 **Gemini Pro**。
   - 他们对 **RAG** 中使用的上下文窗口大小感到好奇，并征求关于如何利用 **Gemini Pro** 实现导师型 AI 任务的建议。
- **内联引用已保存！**：用户现在可以将**聊天回复保存为笔记，并保留原始形式的内联引用**。
   - 这一增强功能允许用户回溯原始素材，解决了高级用户长期以来的需求。
- **用户希望内联引用可复制**：一位用户请求能够将内联引用复制并粘贴到文档中，同时保留链接。
   - 另一位用户希望从 NotebookLM 复制到 Word 时，能带有包含源标题的脚注，以辅助格式排版。
- **新的思考模型提升质量**：一个新的“Thinking Model”已推送到 NotebookLM，以提升回复质量。
   - 关于该模型的具体细节或它如何改进回复的信息尚不明确。
- **葡萄牙语回答问题**：一位用户报告称，尽管之前在葡萄牙语下运行正常，但 NotebookLM 开始强制使用英语回答。
   - 一位用户提供了解决方案：在 URL 末尾添加 `?hl=pt`。



**提到的链接**：<a href="https://en.wikipedia.org/wiki/Hyperborea">Hyperborea - Wikipedia</a>：未找到描述

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1349401785999036508)** (7 messages): 

> `MPS 问题, Gemma 3, Torchvision MPS 错误` 


- **MPS 设备问题阻碍开发进度**：一名成员报告在最近的更改后出现了 `AttributeError: module 'torch.mps' has no attribute 'set_device'` 错误，这可能导致 MPS 支持失效，并链接了相关的 [GitHub commit](https://github.com/pytorch/torchtune/commit/5cb4d54c779fd282dbfd2e1a50d2cb0828468bd2#diff-6cca0f357ea6c4e23906aec0c380c9d21887950f3371c83aa5acb40a83d61066R169)。
   - 另一名成员确认了该问题，并指出 [PR #2486](https://github.com/pytorch/torchtune/pull/2486) 可能是潜在的修复方案，但在 **MPS** 上运行时，该方案导致 **torchvision** 出现了进一步的错误。
- **捕捉到 Gemma 3 动态**：一位用户注意到 **Gemma 3** 模型发生了变化，并提供了一张截图。
   - 该截图通过 Discord CDN 链接上传，显示了一些变更内容 ([image.png](https://cdn.discordapp.com/attachments/1216353675744641096/1349410043111407688/image.png?ex=67d2ff89&is=67d1ae09&hm=3094da9013c91c94a715cc42a41ebde502c1bfe9e64001c598a651f2e4dcaad3&))。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/commit/5cb4d54c779fd282dbfd2e1a50d2cb0828468bd2#diff-6cca0f357ea6c4e23906aec0c380c9d21887950f3371c83aa5acb40a83d61066R169">修复 _get_device_type_from_env() 中缺失的 MPS 检测 (#2471) · pytorch/torchtune@5cb4d54</a>: 共同作者：salman &lt;salman.mohammadi@outlook.com&gt;</li><li><a href="https://github.com/pytorch/torchtune/pull/2486">由 SalmanMohammadi 修复 MPS `get_device` · Pull Request #2486 · pytorch/torchtune</a>: 上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。在 main 分支上：&amp;gt;&amp;gt;&amp;gt; from torchtune.utils import get_de...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1349457184093900930)** (2 messages): 

> `Gemma3, vLLM, Pan & Scan` 


- **Gemma3 的 "Pan & Scan" 实现受到质疑**：一名成员询问 **Gemma3** 论文中旨在提升推理性能的 "**Pan & Scan**" 技术是否需要在 torchtune 中实现。
   - 另一名成员建议这并非至关重要，因为用户可以使用 **vLLM** 配合 **HF ckpt** 来获得增强的性能，并链接到了一个[相关的 Pull Request](https://github.com/vllm-project/vllm/pull/14660)。
- **vLLM 为 Gemma3 提供更佳性能**：用户可以通过此 PR [https://github.com/vllm-project/vllm/pull/14660](https://github.com/vllm-project/vllm/pull/14660) 使用 HF ckpt 配合 vLLM 以获得更好的性能。



**提及的链接**：<a href="https://github.com/vllm-project/vllm/pull/14660">[Model] 添加对 Gemma 3 的支持，由 WoosukKwon 提交 · Pull Request #14660 · vllm-project/vllm</a>：此 PR 添加了对 Gemma 3 的支持，这是来自 Google 的开源视觉语言模型。注意：该 PR 尚未实现 pan-and-scan 预处理算法。它将由后续的.....

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1349408279750774794)** (1 messages): 

> `LlamaIndex, Model Context Protocol (MCP), 工具发现` 


- ****LlamaIndex** 连接至 **Model Context Protocol** 服务器**：**LlamaIndex** 现在集成了 **Model Context Protocol (MCP)**，这是一个旨在简化工具发现和利用的开源项目，详见[此推文](https://twitter.com/llama_index/status/1899848532817035529)。
- **MCP 简化工具使用**：**Model Context Protocol** 的集成允许 **LlamaIndex** 使用任何兼容 **MCP** 的服务所提供的工具，从而增强其功能。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1349379163563495466)** (7 messages): 

> `LlamaExtract 本地化部署, 新 Response API 支持` 


- **LlamaExtract 本地化部署：敏感数据的堡垒**：针对企业对敏感数据的担忧，整个 **Llama-Cloud 平台** 现已支持本地化/BYOC (Bring Your Own Cloud) 部署。
   - 然而，一名成员指出，这些部署的成本通常比使用 SaaS 解决方案*高得多*。
- **Response API：LlamaIndex 会支持新的搜索功能吗？**：一位用户询问了对新 **Response API** 的支持情况，认为它有潜力通过用户选择加入的搜索工具来丰富结果。
   - 一名成员给出了肯定答复，表示他们*正尝试在今天完成这项工作*。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1349440295955071007)** (3 messages): 

> `测验截止日期，MOOC 学习者的研究机会，MOOC 学习者的实验机会` 


- **5 月的测验截止日期**：所有**测验截止日期**都安排在 **5 月**，更多细节将很快发布。
   - 用户被告知他们已在邮件列表中，并已查收关于 **Lecture 6** 的最新邮件。
- **学习者寻求实验机会**：一名成员询问了针对 **MOOC 学习者**的 **Labs** 计划以及**研究机会**。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1349362890087465057)** (2 messages): 

> `Cohere 多语言 embed 模型定价，OpenAI Responses API，Cohere 兼容性` 


- **Cohere 多语言定价疑问**：一名成员询问了 **Cohere 多语言 embed 模型**的定价，并指出在文档中难以找到此信息。
   - 讨论中未分享关于定价的具体细节或链接。
- **OpenAI 发布 Responses API**：**OpenAI** 刚刚发布了 **Responses API** 以及 **Agents SDK**，重点在于更高的简洁性和表达力。
   - [Responses API 文档](https://platform.openai.com/docs/guides/responses-vs-chat-completions)强调其专为多工具、多轮次和多模态设计，解决了当前 API 的用户痛点。
- **Cohere 是否兼容 OpenAI API？**：一名成员询问了 **Cohere** 是否有可能兼容 **OpenAI** 新发布的 **Responses API**。
   - [OpenAI cookbook 示例](https://cookbook.openai.com/examples/responses_api/responses_example)详细介绍了 **Responses API** 作为多轮交互、托管工具和细粒度上下文控制的解决方案。



**提及的链接**：<a href="https://cookbook.openai.com/examples/responses_api/responses_example">使用 Responses API 进行网络搜索和状态管理 | OpenAI Cookbook</a>：使用 OpenAI API 构建应用的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。

  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1349414740694401065)** (1 messages): 

> `Chat API seed 参数问题，相同 seed 输出不一致` 


- **Chat API 的 Seed 参数受到质疑**：一位用户指出 **Chat API** 似乎忽略了 `seed` 参数，导致即使使用相同的输入和 seed 值，输出也各不相同。
- **不一致的输出困扰着带 Seed 的 API 调用**：多位用户报告在使用具有相同 `seed` 值的 **Chat API** 时输出不一致，这表明可复现性可能存在问题。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1349447063867228295)** (2 messages): 

> `DSPy 缓存，可插拔缓存模块` 


- **DSPy 缓存机制疑问**：一名成员询问了 **DSPy 中的缓存工作原理**以及是否可以修改缓存行为。
- **开发中的可插拔缓存模块**：另一名成员指向了一个正在开发中的可插拔 **Cache 模块**的 [Pull Request](https://github.com/stanfordnlp/dspy/pull/1922)。



**提及的链接**：<a href="https://github.com/stanfordnlp/dspy/pull/1922">Feature/caching by hmoazam · Pull Request #1922 · stanfordnlp/dspy</a>：一个统一的缓存接口，具有两级缓存——内存 LRU 缓存和扇出（磁盘上）缓存。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1349467285911240827)** (2 messages): 

> `modular max，Linux exec 问题，GitHub PR` 


- **Modular Max PR 接近完成**：一名成员分享了一个 [GitHub Pull Request](https://github.com/modular/max/pull/3998)，希望它最终能像他们的项目一样。
   - 首先必须合并 *foundations PR*，然后需要解决一些 **Linux exec 的问题**。
- **Linux Exec 问题推迟发布**：由于 **Linux exec** 的问题尚未解决，新功能的发布被推迟，目前正在等待 *foundations PR* 的合并。
   - 尽管遇到了挫折，开发者仍对近期发布表示乐观，订阅者将收到关于 PR 进展的更新。



**提及的链接**：<a href="https://github.com/modular/max/pull/3998">[stdlib] 添加从 exec 文件生成和管理进程的功能，由 izo0x90 提交 · Pull Request #3998 · modular/max</a>：此 PR 的基础已在此奠定，它添加了所需的底层工具：为 Mojo 的 cLib 绑定添加了 vfork、execvp、kill 系统调用工具；为文件描述符添加了 read_bytes。一旦该 PR 合并...

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1349433953139822734)** (1 条消息): 

> `追踪评估工具，评估数据集位置` 


- **发现追踪评估工具的中心枢纽**：一位成员询问是否有追踪所有评估工具的中心化场所。
   - 另一位成员建议查看 [gorilla/berkeley-function-call-leaderboard/data/multi_turn_func_doc](https://github.com/ShishirPatil/gorilla/tree/c67d246e5fbf436b4ab879d821dc15c88c83f7e2/berkeley-function-call-leaderboard/data/multi_turn_func_doc) 作为潜在资源。
- **确定评估数据集位置**：一位成员询问是否所有的评估数据集都可以在 `gorilla/berkeley-function-call-leaderboard/data` 文件夹中找到。
   - 讨论中没有进一步的消息确认该文件夹是否包含所有评估数据集。


  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1349433375697666059)** (1 条消息): 

> `RAG, Pinecone, VPC 部署` 


- **弃用 Pinecone**：该 RAG 之前使用 **Pinecone**，但由于其性能限制以及缺乏对 **VPC 部署**的支持，选择了不同的方向。
- **新的 RAG 方向**：由于 **Pinecone** 的局限性和缺乏 **VPC 部署**，探索了 **RAG** 的新方向。


  

---


---


{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}