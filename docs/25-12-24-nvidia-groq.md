---
companies:
- nvidia
- groq
- openai
- tesla
- epoch-ai
- gemini
date: '2025-12-24T05:44:39.731046Z'
description: '**Groq** 的领导团队将根据一项价值 **200 亿美元现金** 的“非独家许可协议”加入 **英伟达 (Nvidia)**。尽管英伟达声明并非收购
  Groq 公司本身，但这仍标志着 AI 芯片领域的一次重大并购。黄仁勋计划将 Groq 的低延迟处理器整合到英伟达的 AI 工厂架构中，以增强 AI 推理和实时工作负载的性能。


  Twitter 上的热点包括：**Gemini** 被用作追踪卡路里的消费级工具；OpenAI 讨论了“部署差距 (deployment gap)”，重点关注模型在医疗和商业领域的应用；特斯拉的
  FSD v14 被描述为消费级 AI 的“物理图灵测试”。


  **Epoch AI** 指出了基准测试面临的挑战，强调供应商差异和集成问题正影响着模型质量的评估。此外，AI 社区关于编程智能体 (coding agents)
  与开发者体验融合的讨论仍在继续。'
id: MjAyNS0x
models:
- gemini
- fsd-v14
people:
- jensen_huang
- xeophon
- js_denain
- jim_fan
title: 英伟达以 200 亿美元现金收购 Groq（大部分业务）；系史上规模最大的“高管雇佣式收购”（execuhire）。
topics:
- benchmarking
- inference
- model-evaluation
- ai-integration
- agent-patterns
- real-time-processing
- low-latency
- developer-experience
- healthcare
- business-workflows
- consumer-ai
---

**高管雇佣（Execuhires）回归了！**

> 2025年12月24日至12月25日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 24 个 Discord 服务（208 个频道，5086 条消息）。预计节省阅读时间（以每分钟 200 字计）：346 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

高管雇佣（Execuhires）最早始于 [2024 年 8 月](https://news.smol.ai/issues/24-08-02-ainews-execuhires-tempting-the-wrath-of-khan)，并在 [2025 年 6 月](https://news.smol.ai/issues/25-06-11-execuhires-2) 再次出现，但看来 2025 年的平安夜上演“帽子戏法”也不算太晚。在一篇仅有 5 句话的帖子中，Groq 确认了一项“非独家许可协议”，Groq 的大部分领导团队将加入 Nvidia，留下 GroqCloud，而现任 CFO 将出任旧 Groq 的 CEO，据报道[总对价为 200 亿美元现金。](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html)

这是一场名义之外的全面收购，而且有几个事实让它变得很有趣：Groq 在 9 月份的最后估值为 69 亿美元，并表示是 Nvidia 主动找上门来的。Nvidia 此前最大的收购案是 2019 年以 70 亿美元收购 Mellanox，而这次收购仅占 Nvidia 现金储备的三分之一。

Jensen 的引言是我们目前掌握的关于未来计划的最详尽细节：

> “我们计划将 Groq 的低延迟处理器集成到 NVIDIA AI 工厂架构中，扩展平台以服务更广泛的 AI 推理（inference）和实时工作负载，”黄仁勋（Huang）写道。
> 
> 黄补充道，“虽然我们正在为我们的队伍增加才华横溢的员工并获得 Groq 的 IP 许可，但我们并不是在收购 Groq 这家公司。”

这就是我们所知道的全部，但在半导体（semis）领域，这非常非常令人震惊，尤其是对于那些心怀希望的 Nvidia 竞争对手而言。

---

# AI Twitter 综述

**热门推文（按互动量排序）**

- **Gemini 作为消费者工具**：一个关于“AI 作为成瘾性助手”的病毒式案例，使用 Gemini 进行卡路里追踪 ([推文](https://twitter.com/skooookum/status/2003608923371389157))。
- **“部署差距” / 能力悬置**：OpenAI 将 2026 年的进展描述为既关乎前沿能力，也关乎*如何让模型得到良好应用*——特别是在**医疗保健**、商业和日常生活工作流中 ([推文](https://twitter.com/OpenAI/status/2003594025098785145))。
- **FSD 作为“物理图灵测试”**：Jim Fan 将 Tesla FSD v14 描述为第一个在日常使用中让人感觉与人类驾驶员无异的消费级 AI，强调了“超现实 → 常规 → 依赖”转变的速度之快 ([推文](https://twitter.com/DrJimFan/status/2003593613918531891))。
- **算力/基础设施现实主义**：莫迪（Modi）关于 ISRO 发射的帖子在整体互动中占据主导地位，但很大程度上超出了 AI 工程的范畴 ([推文](https://twitter.com/narendramodi/status/2003677923820335183), [推文](https://twitter.com/narendramodi/status/2003681952323502169))。

---

**基准测试与评估：供应商差异、测试框架 Bug 以及“分数到底意味着什么？”**

- **基准测试很脆弱，因为流水线（pipeline）很脆弱**：Epoch AI 指出，报告的分数通常是供应商行为（超时、速率限制、分词（tokenization）怪癖、缺少参数、瞬时错误）的下游产物，而*较新的模型/供应商*受到的影响尤为严重 ([Epoch 综述](https://twitter.com/EpochAIResearch/status/2003592566772822516), [供应商错误说明](https://twitter.com/EpochAIResearch/status/2003592610569683089))。[@xeophon](https://twitter.com/xeophon/status/2003592720741466478)（与 [@js_denain](https://twitter.com/EpochAIResearch/status/2003592622724776201) 合作）的客座文章将其转化为一份极具操作性的清单：如果你的测试框架（harness）没有控制采样参数、重试、截断、工具调用（tool-calling）差异和 API 边缘情况，那么你测量的就不是模型质量——而是供应商的可靠性和集成债务。
- **“同一模型，不同供应商，不同输出质量”成为首要问题**：多位工程师呼应，开放模型生态系统现在对推理（inference）供应商的依赖程度不亚于对权重（weights）的依赖；基准测试供应商需要“Agent 测试框架”的纪律（提示词工程（prompting）、部署配置、采样、工具行为），而不是简单的 one-shot 评估脚本 ([总结](https://twitter.com/eliebakouch/status/2003604370534072445), [博客链接](https://twitter.com/dejavucoder/status/2003594248973930929))。这也引发了关于“开放”在实践中意味着什么的更广泛讨论——仅有权重并不等同于可复现性 ([LMArena 关于“开放”的灰色地带](https://twitter.com/arena/status/2003620051078074593))。

---

**编程 Agent、Agent 封装与开发者体验（DX）的融合**

- **从“Agent 模式”转向 Prompt + Tool**：多位开发者报告称，利用当前的前沿/编程模型，许多经典模式（计划/反思循环、手工编写的工具策略）正变得可选——优秀的 Prompting + 工具定义通常就足够了，这使得工作重心转向了**上下文工程（Context Engineering）**和良好的默认设置（[diptanu](https://twitter.com/diptanu/status/2003674481144004667)，[Weaviate 的定义 + 上下文工程视角](https://twitter.com/weaviate_io/status/2003824281231220902)）。
- **封装 Agent 是缺失的原语**：[@hwchase17](https://twitter.com/hwchase17/status/2003599022871777467) 认为 [**agent.md**](http://agent.md/) **+ Skill**（作为开放标准）可以定义一个 Agent，但我们仍然缺乏一个便携式单元来捆绑：规则、Skill、MCP 服务器/工具以及子 Agent——“一个整洁的 zip 小文件，能生成一整个 Agent 小队”（[后续讨论](https://twitter.com/hwchase17/status/2003715230120173737)）。他指出 **OpenCode 的 Agent 规范**是一个更好的基准，因为它允许一个 Agent 被用作“主 Agent”或“子 Agent”，从而实现完全专业化的“将整个环境转变为一个编写 LangGraph 的 Agent”工作流（[推文](https://twitter.com/hwchase17/status/2003922408240304245)）。
- **工具链围绕作为可重用策略模块的“Skill”发布**：Mistral 的 Vibe CLI 将“Skill”作为可重用的规则包发布，此外还支持推理模型和终端主题化——明确推动可共享的、项目级的 Agent 策略产物（[推文](https://twitter.com/MistralAI/status/2003843358054068327)）。
- **使用限制经济学塑造行为**：Anthropic/Claude 在元旦期间将 Pro/Max 限制翻倍，明确鼓励开发者加大对 Agent 工作流的尝试（[Claude](https://twitter.com/claudeai/status/2003918730833608902), [Alex Albert](https://twitter.com/alexalbert__/status/2003923042100273389)）。另一方面，用户报告称“配额消耗（Quota Burn）”是迭代式 Agent 循环中的一个现实约束（[推文](https://twitter.com/vikhyatk/status/2003647290507227396)）。
- **新兴的 UX 模式**：Windsurf “Wave 13” 强调了**真正的并行 Agent** + 专用 Agent 终端，反映了在指挥者风格的编排 UX（工作树 + 级联）上的趋同（[Cognition](https://twitter.com/cognition/status/2003926592406671472)，以及 [swyx 的元评论](https://twitter.com/swyx/status/2003941412572934361)）。Base44 展示了一个类似 IDE 的方向：在查看实时预览的同时编辑代码；点击 UI 即可跳转到定义代码——将 UI 视为代码的导航索引（[推文](https://twitter.com/MS_BASE44/status/2003868520359317749)）。

---

**开源模型与“推理分发层”：MiniMax M2.1, GLM-4.7, Qwen Image Edit**

- **MiniMax M2.1 的分发攻势**：M2.1 出现在多个“开发者聚集”的平台——**LMArena Code Arena** ([Arena](https://twitter.com/arena/status/2003585316029104383))、**Cline** ([MiniMax](https://twitter.com/MiniMax__AI/status/2003599117503852680))、**Kilo** ([推文](https://twitter.com/MiniMax__AI/status/2003606223191703708))、**Roo Code** ([推文](https://twitter.com/MiniMax__AI/status/2003611728320561528))、**Ollama** ([推文](https://twitter.com/MiniMax__AI/status/2003715959719362584))、**BlackboxAI** ([推文](https://twitter.com/MiniMax__AI/status/2003926396335460447)) 等。基准测试/排行榜进一步加强了其采用率：在 SWE-bench 变体和 SciCode 上表现强劲 ([Ofir Press](https://twitter.com/OfirPress/status/2003625671042732329))，并在 Vals Index 的权重开放（open-weight）模型中排名第 2，仅次于 GLM 4.7，但具有更低的延迟和成本 ([ValsAI](https://twitter.com/ValsAI/status/2003646964664287667))。MiniMax 还声称其长程代码编写（long-horizon coding）的价格约为 Opus 的 1/10 ([推文](https://twitter.com/MiniMax__AI/status/2003673337671602378))。
- **智谱 GLM-4.7 的势头 + devpack/MCP 集成**：智谱强调了持续的开源努力，并在 Hugging Face 趋势榜排名第 1 ([推文](https://twitter.com/Zai_org/status/2003828175089098943))。Roo Code 宣布支持 GLM-4.7 ([推文](https://twitter.com/roocode/status/2003652972555997560))。智谱还推出了 MCP 风格的开发者工具，如用于对话内仓库探索的 **Zread MCP**（无需离开 Agent 工作流即可搜索/读取文件）([推文](https://twitter.com/Zai_org/status/2003872419791229285))。此外，工程师们展示了通过 MLX 分布式 + 批处理生成在 Apple Silicon 集群上实现 GLM 4.7 的高吞吐量本地推理（例如，在 **4× M3 Ultra** 上实现 **63 tok/s** 的吞吐量，6-bit，batch size 4）([awnihannun](https://twitter.com/awnihannun/status/2003854411848904937))。
- **Qwen Image Edit 2511 作为“产品化的开源图像编辑器”**：Qwen-Image-Edit-2511 已在 Replicate 和其他 UI 上线 ([Replicate 发布](https://twitter.com/Alibaba_Qwen/status/2003751934013100458), [TostUI](https://twitter.com/Alibaba_Qwen/status/2003753784527507781)，以及 [@_akhaliq](https://twitter.com/_akhaliq/status/2003601664675316051) 等社区 HF spaces)。微调的可访问性得到提升：AI Toolkit 增加了对 LoRA 的支持，并提供了一个 **3-bit 精度恢复适配器**，支持在小于 24GB VRAM 的环境下进行微调 ([ostrisai](https://twitter.com/ostrisai/status/2003808898189611491))。

---

**训练与研究笔记：Agent 的 RL、预训练技巧以及表示/注意力机制修复**

- **工具使用型 Agent 的端到端 RL (Agent-R1)**：一条长技术推文将 Agent 训练定义为本质上的 RL，因为存在**随机的工具/环境反馈**，并提出了用于信用分配（credit assignment）的显式掩码和 ToolEnv 交互循环。据报告，在多跳问答（multi-hop QA）上，该方法相比原生 RAG 提升巨大（例如，GRPO 为 **0.3877 EM**，而 RAG 为 **0.1328 EM**）([推文](https://twitter.com/omarsar0/status/2003862504490086596))。
- [**Character.AI](http://character.ai/) 的预训练 “Squinch” 及相关技巧**：[@simon_mo_](https://twitter.com/simon_mo_/status/2003608325624406482) 总结了 CAI 的一篇博客文章，描述了他们如何在网络较弱的情况下，通过使用 Noam Shazeer 的梯度压缩算法 “**Squinch**”（以及其他预训练技巧），在 **GCP H100-TCPX** 上保持强大的 MFU。后续推文强调了他们的蒸馏方法非常值得关注 ([eliebakouch](https://twitter.com/eliebakouch/status/2003632344159424562))。
- **无需海量配对数据的多模态 (SEMI)**：DeepLearningAI 总结了 SEMI：通过投影器（projector）和从少量配对示例生成的 LoRA 适配器，将任何预训练编码器接入 LLM；在数据丰富的领域进行训练，即可少样本适配（few-shot adapts）到新领域 ([推文](https://twitter.com/DeepLearningAI/status/2003593131132916204))。
- **值得关注的架构/表示相关论文**：
    - **PoPE vs RoPE 纠缠**：声称 RoPE 纠缠了内容和位置；提议用 PoPE 作为修复方案 ([推文](https://twitter.com/agopal42/status/2003900815560659303))。
    - **循环层 ViT 压缩**：建议使用 K≪N 层并结合循环机制重写 N 层 ViT，在仅用约 2-3 层的情况下达到 DINOv2 的性能 ([推文](https://twitter.com/f14bertolotti/status/2003760506214158693))。
    - **注意力缩放原理解析**：一篇关于“为什么要除以 √d_k”的清晰文章，旨在防止注意力机制中的 Softmax 饱和/梯度消失 ([推文](https://twitter.com/viplismism/status/2003807608571076782))，并附带一个细致的反向观点，即 L2 归一化注意力仅在关于 Value 相关性的严格假设下才能保持方差 ([推文](https://twitter.com/ArmenAgha/status/2003918120881475832))。

---

**机器人、自主性与“物理图灵测试”框架**

- **NVIDIA 机器人技术栈进展**：Jim Fan 将机器人技术定位为“最后的重大挑战”，并列举了 NVIDIA 最近发布的内容：**GR00T VLA** 开源 Checkpoints (N1, N1.5, N1.6)、**GR00T Dreams** 世界模型、**SONIC** 全身控制基础模型，以及 RL 后训练配方——涵盖从仿真到 sim2real 的全过程 ([thread](https://twitter.com/DrJimFan/status/2003879965369290797), [sim2real note](https://twitter.com/DrJimFan/status/2003879976173818298))。
- **人形机器人自主交互**：Brett Adcock 发布了机器人与人交互并在无需 teleop（远程操作）的情况下响应指令的演示，强调了语音到操作的耦合（意图 → 像素 → 动作）([swag demo](https://twitter.com/adcock_brett/status/2003598494838431874), [autonomy claim](https://twitter.com/adcock_brett/status/2003598719971995709), [voice+manipulation framing](https://twitter.com/adcock_brett/status/2003909157897015585))。
- **Waymo 的“人类模块无法扩展”**：一则尖锐的批评称，旧金山的一起事故反映了远程“确认检查”的积压，暗示了一个依赖陷阱，即人类仍然是自主技术栈中的吞吐量瓶颈 ([tweet](https://twitter.com/Yuchenj_UW/status/2003708815934640536))。

---

**宏观主题：人才、产品周期与“部署差距”**

- **人才争夺战关乎使命 + 同伴**：Sarah Hooker 的观点：顶尖人才有很多选择；获胜的关键是与志同道合、不断突破边界的人共事，而不仅仅是薪酬 ([tweet](https://twitter.com/sarahookr/status/2003581788850127276))。
- **3 个月模型周期下的产品策略**：一份被广泛分享的关于“Lovable”增长经验的总结指出，PMF（产品市场匹配）在每个模型周期都会“过期”；MVP 让位于“MLP”（最受喜爱产品），护城河变成了发布速度 + 品牌，而非技术 ([tweet](https://twitter.com/crystalsssup/status/2003704941962285463))。
- **OpenAI 的“能力过剩” (capability overhang)**：这一组中最明确的元观点：模型能力正领先于实际的用户部署；2026 年的进展取决于通过更好的 UX/工作流和行业集成（医疗、商业）来缩小这一采用差距 ([tweet](https://twitter.com/OpenAI/status/2003594025098785145))。
- **工程劳动力转向编排 (orchestration)**：一种管理层观点认为，IC（独立贡献者）正在成为“编排者”——频繁的上下文切换 + 判断力/品味比纯粹的实现速度更重要（该推文互动量为 0，但捕捉到了多个线程中出现的新兴主题） ([tweet](https://twitter.com/brivael/status/2003871914104688867))。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

没有内容达到我们的标准

## 较低技术门槛的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 生成艺术与动画实验

- [**Former 3D Animator trying out AI, Is the consistency getting there?**](https://www.reddit.com/r/StableDiffusion/comments/1puszuc/former_3d_animator_trying_out_ai_is_the/) (Activity: 1280): **一位前 3D 动画师正在尝试整合 AI 以增强 3D 动画的真实感。通过使用在他们自己的 3D 渲染图上训练的自定义 LoRA，他们旨在保持角色的原始本质，同时增加 AI 驱动的真实感。该项目涉及除 ComfyUI 之外的一系列复杂工具，重点关注 AI 增强的角色动作是否看起来像人类，或者这种幻觉是否会失败。** 评论反映了幽默与怀疑的交织，一些用户开玩笑说 3D 动画师会失业，而另一些人则提供了极少的反馈。技术辩论很少，更多关注于项目的新颖性和执行力。
- [**not all cosplayers are real**](https://www.reddit.com/r/ChatGPT/comments/1pua0aj/not_all_cosplayers_are_real/) (Activity: 1002): **该帖子讨论了使用 AI 工具 Nano Banana Pro 在几分钟内生成逼真的 Cosplay 图像，强调了在社交媒体中被滥用的可能性，个人可能会将 AI 生成的图像作为真实的 Cosplay 展示以获取点击量或打赏。作者指出这些图像并非精挑细选，承认存在一些不准确之处，但仍认为结果具有说服力。这引发了对在线 Cosplay 社区真实性的担忧。** 评论中一个显著的观点表达了对 Cosplay 社区中 AI 生成内容盛行的沮丧，表明真正的 Cosplayer 越来越担心此类技术对其手艺的影响。
- [**The illusion painter, part 2**](https://www.reddit.com/r/aivideo/comments/1puqnba/the_illusion_painter_part_2/) (Activity: 1288): **标题为“幻觉画家，第二部分”的帖子似乎是一个系列的延续，涉及一位作品能创造幻觉的画家。评论中的技术讨论对叙事结构进行了批评，一位评论者建议该系列本可以以一个反转结束，即展示一幅真实的画作而非幻觉。这暗示了在视觉艺术语境下对叙事预期和颠覆的关注。** 评论反映了幽默与批评的交织，一位用户幽默地对角色在虚构中的死亡表示满意，而另一位则询问“反派”何时会被阻止，表明了对内容的叙事驱动型参与。
- [**The illusion painter, part 2**](https://www.reddit.com/r/aivideo/comments/1puqnba/the_illusion_painter_part_2/) (Activity: 1289): **标题为“幻觉画家，第二部分”的帖子似乎是一个关于“幻觉画家”系列或叙事的延续。帖子的技术内容从标题或评论中并不明确，但它暗示了一个欺骗或意外结果的主题，可能涉及视觉艺术或叙事反转。评论反映了幽默与批评的交织，其中一条建议通过让最后一幅作品成为真实的画作来进行叙事反转，表明了颠覆预期的可能主题。** 评论反映了幽默与批评的交织，其中一条建议通过让最后一幅作品成为真实的画作来进行叙事反转，表明了颠覆预期的可能主题。

### 2. AI 角色与 Meme 创作

- [**我让 CGPT 将自己与其他 AI Chatbots 一起生成为角色**](https://www.reddit.com/r/ChatGPT/comments/1pukx34/i_asked_cgpt_to_generate_itself_as_a_character/) (热度: 2441): **该图像是将各种 AI Chatbots 创意地表现为动漫风格的角色，每个角色都有独特的配色方案和风格。这种艺术化的描绘包括了 ChatGPT、Gemini、Grok 和 Claude，每个角色都通过独特的视觉元素来体现其被感知的个性或功能。该图像是非技术性的，更多是作为一种视觉隐喻，而非对这些 Chatbots 的功能或架构的技术说明。[查看图片](https://i.redd.it/zv353p0mv49g1.png)**。评论区对这张图片进行了幽默的互动，有人指出 Grok 的形象似乎误解了任务要求，暗示对其角色设计的戏谑批评。另一条评论则幽默地建议 ChatGPT 的角色现在在设定上拥有了绿发，反映了社区对这种视觉呈现的参与感。
- [**过去 VS 现在**](https://www.reddit.com/r/ChatGPT/comments/1pu7rsu/back_then_vs_now/) (热度: 1340): **该图像是一个迷因（meme），幽默地对比了学生对信息源的依赖从 Wikipedia 转向 ChatGPT 等 AI 工具的转变。它突出了学生现在更倾向于使用 AI 获取信息，而曾经作为主要来源的 Wikipedia 则被描绘得不再那么重要。该迷因反映了教育背景下的一个更广泛趋势，即 AI 越来越多地被用于研究和学习，使 Wikipedia 等传统来源显得黯然失色。** 一条评论幽默地指出了学生使用 AI 而非可引用来源的讽刺性，而另一条评论则反思了 Wikipedia 受欢迎程度的文化转变——从最初被不建议作为来源，到现在被 AI 抢了风头。

### 3. AI 驱动的音乐与视频创作

- [**指环王迪斯科：一曲 Funk 统领众戒 | Wicked AI 制作的音乐视频**](https://www.reddit.com/r/aivideo/comments/1pu8smq/lord_of_the_rings_disco_one_funk_to_rule_them_all/) (热度: 1433): **Wicked AI 发布了一段名为《指环王迪斯科：一曲 Funk 统领众戒》的音乐视频，创意地将《指环王》系列的元素与迪斯科主题相结合。该视频是使用 AI 生成的，展示了 AI 在制作复杂多媒体内容方面的能力。然而，一些观众注意到音乐风格并未严格遵循传统的迪斯科或 Funk 流派，表明在流派分类上可能存在偏差。** 一位评论者表达了对 AI 生成内容的惊叹与担忧交织的情绪，强调了 AI 进步的双重性质：既令人印象深刻，又可能令人不安。
- [**指环王迪斯科：一曲 Funk 统领众戒 | Wicked AI 制作的音乐视频**](https://www.reddit.com/r/aivideo/comments/1pu8smq/lord_of_the_rings_disco_one_funk_to_rule_them_all/) (热度: 1436): **Wicked AI 发布了一段名为《指环王迪斯科：一曲 Funk 统领众戒》的音乐视频，创意地将《指环王》系列的元素与迪斯科主题相结合。该视频是 AI 生成内容的产物，展示了现代 AI 将文化主题与音乐融合的能力。然而，一些观众注意到音乐并未严格遵循传统的迪斯科或 Funk 流派，表明在流派分类上可能存在偏差。** 一位评论者表达了对 AI 生成内容的惊叹与担忧交织的情绪，强调了 AI 进步的双重性质：既令人印象深刻，又可能令人不安。

---

# AI Discord 简报

> 由 gpt-5.1 生成的摘要之摘要的摘要
> 

**1. Wave 13 Coding Agents 与 AI IDE 工具链**

- **Windsurf Waves 推出免费的准前沿 SWE-1.5**: Windsurf 发布了 **Wave 13: Shipmas Edition**，增加了 **并行多 Agent Cascade 工作流**、**专用 zsh 终端**（macOS 上需手动开启）、**Git worktree 支持**、**多 Cascade 面板与标签页**以及 **上下文窗口指示器**，同时将其准前沿编码模型 **SWE-1.5** 向所有用户免费开放 3 个月（正常吞吐量）。团队在 [Wave 13 公告](https://discord.com/channels/1027685395649015980/1027688115592237117/1453488772837671157)中将 SWE-1.5 定位为接近 **SWE-Bench-Pro** 的性能，并将这些功能打包在名为 “Merry Shipmas!” 的季节性发布中。
    - 工程师强调，**Git worktree 支持**加上**多面板 Cascade** 允许在*同一个仓库*中并发进行分支开发和实验，而不会陷入合并地狱，这直接针对了常见的 Agentic 编码工作流。**专用终端**在用户自己的 `.zshrc` 下运行，社区成员认为，与临时的沙盒化 shell 相比，这对于稳健的工具链、路径设置和长时间运行的命令至关重要。
- **OpenRouter 直接接入 Open-WebUI**: 一位社区开发者发布了一个针对 **OpenRouter Responses API** 的 **Open-WebUI 集成管道**，项目名为 `Open-WebUI-OpenRouter-pipe`，发布在 GitHub 上的 [Open-WebUI-OpenRouter-pipe](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe)。作者邀请用户在实际工作负载中进行测试并提交 Bug，以便在广泛采用之前完善该集成。
    - 与此同时，位于 [llumen demo](https://llumen-demo.easonabc.eu.org/) 的 **llumen** 聊天 UI 演示发布了 [v0.4.2](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2) 补丁，专门修复了标题生成故障和细微的聊天 Bug，显示出前端 Agentic 体验的快速迭代。OpenRouter 用户还讨论了编码流中的**响应缓存**，一位工程师声称 **80–90% 的缓存命中率**是现实的，而另一位工程师则警告说，在 Router 层面简单的缓存层很难安全地转化为面向用户的成本节省。
- **DSPy 关注 TextGrad 和 Agentic Context Engineering**: 在 **DSPy** 社区中，成员询问来自 **“Agentic Context Engineering”** ([*Agentic Context Engineering*](https://arxiv.org/pdf/2510.04618)) 和 **“LLM Autodiff / TextGrad”** ([*LLM AutoDiff: TextGrad*](https://arxiv.org/abs/2501.16673)) 的想法是否会落地到 DSPy，并链接了参考实现 [textgrad](https://github.com/zou-group/textgrad)。工程师们辩论了基于文本的类梯度更新是否会成为 “DSPy 杀手”，但其他人指出“它似乎没有被积极维护”，并将其更多地视为概念前身而非生产工具。
    - 这引发了更广泛的 **Prompt 优化**讨论，用户观察到“每种方法至少有 10 个细微变体，比如 textgrad 版本”，表达了对论文的疲劳，但对 DSPy 中的**可组合优化原语**表现出兴趣。一位新的**高级全栈/区块链工程师**加入，列出了现代技术栈（React/Next.js/Nuxt, Node/Nest/Laravel/FastAPI, Solidity, Docker/AWS, PostgreSQL/Redis/MongoDB），强调了 DSPy 的受众正日益转向系统和基础设施工程师，而不仅仅是 ML 研究员。

**2. 视频、音频与多模态模型工具**

- **ElevenLabs 转型为一站式 AI 视频商城**：在 OpenAI 社区中，用户报告称将 [**ElevenLabs**](https://elevenlabs.io/) 作为使用 **Sora 2**、**Google Veo 3.1** 和 **Kling 2.6** 生成视频的中心，并称赞其 *“所有项目都可以在一个地方访问，而无需在多个账号间切换。”* 一位工程师指出，通过 ElevenLabs 渲染的 Sora-2 视频**不带水印**，而 **Nano Banana Pro** 则会在每个输出上打上明显的标记。
    - 随着人们将 **ElevenLabs** 与 [**Higgsfield**](https://higgsfield.ai/) 进行对比，定价和政策比较也随之出现。Higgsfield 为某些模型提供 **$49/月（按年计费）** 的*无限*视频生成，而另一些人则为 ElevenLabs 辩护，理由是他们已经依赖它进行**有声读物配音**。创作者还观察到，ElevenLabs 的内部安全层因后端而异——某些被 **Sora 2** 拒绝的提示词在 **Veo 3.1** 或 **Kling O1** 上可以部分运行，这引发了 *“Sora 也在检查实际输出，而 veo/wan [检查] 文本提示词输入”* 的猜测。
- **FlashSR 以 200 倍实时速度助力音频增强**：在 Latent Space 上，Yatharth Sharma 通过 X 帖子宣布了 **FlashSR**，这是一款快速音频增强/超分辨率模型，处理速度可达 **>200 倍实时速度**：[FlashSR 音频增强发布](https://xcancel.com/Yatharth3501/status/2003884180577702074)。FlashSR 已经集成到 **MiraTTS** 中，现在作为开源模型和代码在 **Hugging Face** 和 **GitHub** 上发布，供他人接入 TTS 和语音流水线。
    - 工程师们将 FlashSR 视为**延迟敏感型语音产品**的实用替代方案，因为每秒音频低于 10 ms 的延迟使得多阶段流水线（ASR → LLM → TTS → 增强）在没有用户可见延迟的情况下变得可行。它在公开发布前就已存在于 MiraTTS 中，这让一些人确信该代码在类生产负载中经过了实战测试，而不仅仅是一个研究演示。
- **Qwen 2.5VL-3B 和 TRELLIS.2-4B 推动平价多模态**：Hugging Face 用户报告称，**Qwen 2.5VL-3B** 在 **P100** 上以 **4-bit** 运行且**视觉层不量化**时，可以处理 **1400×900** 左右的图像，推理时消耗约 **5 GB VRAM**，在处理 **2k 8-bit PNG** 和约 8k token 上下文的 **QLoRA** 微调中消耗 **~4 GB**。与此同时，Microsoft 发布了 [**TRELLIS.2-4B**](https://huggingface.co/microsoft/TRELLIS.2-4B)，这是一个 **4B 参数**模型，可在 **8 GB GPU** 上将 **2D 图像转换为 3D**（1536 分辨率场），基于 **SigLIP** 视觉和 **Qwen-3** 语言骨干构建。
    - 从业者指出，这些配置使得**在通用云端 GPU 上进行严肃的多模态工作变得可行**，尽管一位用户开玩笑说，如果 TRELLIS 不能在 *“烤面包机级别的 GPU”* 上运行，它 *“肯定会立即产生完全不可用的结果”*。讨论集中在量化（语言部分 4-bit，视觉部分全精度）在出现退化之前能推到什么程度，以及何时视觉分支应优先选择 bf16/fp16 而非全 fp32。

**3. 架构技巧、精度之战与可解释性**

- **Partial RoPE, RMSNorm Placement and Attention Norms Under the Microscope**: 在 Eleuther 的 **research** 频道中，贡献者们深入分析了 **partial RoPE** 的采用（例如在 **Qwen3‑Next** 中），并指出 [**arXiv 论文**](https://arxiv.org/abs/2512.19941) 中的一项历史消融研究表明，它在效率和长上下文泛化方面有显著提升。他们还讨论了博客文章 **“[Attention normalizes the wrong norm](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/)”**，一位研究员断言，即使在经过严格控制变量并训练了 **~5000 个模型** 后，替代归一化方案在语言任务中的表现仍 **差于标准 softmax**。
    - 工程师们进一步讨论了 **RoPE 如何阻碍可解释性**，引用了 Eleuther 自己的文章 [**Rotary Embeddings: A Complete Guide**](https://blog.eleuther.ai/rotary-embeddings/)，并通过共享图表对比了 **RoPE vs PoPE**。另一个讨论串询问了关于在 **attention 之后放置 RMSNorm (post‑SDPA)** 的问题；其他人引用 **Qwen gating 论文** 辩称，增加 norm 可以提高训练稳定性和性能，这很可能是由于 norm 中的 **非线性** 而不仅仅是缩放作用。
- **bf16 vs fp16: Range, Overflow and LR Tradeoffs**: Hugging Face 用户重新讨论了 **bf16 vs fp16** 之争，指出 **fp16** 具有 *更高的精度但较低的动态范围*，而 **bf16** 提供 *较低的精度但高得多的范围*，这对于处理大激活值（large activations）至关重要。一位工程师总结道：*“bf16 不会那么容易溢出……对于巨大的 softmax 之类的东西（但通常它们是在 f32 中完成的……），但在使用 f16 时……更高的精度有助于参数适应较低的 lr，而这在 bf16 中可能会发生下溢”*，这概述了为什么混合精度栈通常需要兼顾这三种格式。
    - 这引出了关于在 **fp16 推理中运行 bf16 训练的模型** 的问题，共识是 *通常可以蒙混过关*，但除非将最大的 matmuls 和 softmax 保持在 fp32，否则应预料到会有更多的 **溢出/NaN 风险**。从业者建议在跨格式移植训练配方（training recipes）时检查 **优化器状态和 LR 调度**，因为针对 fp16 下溢调优的调度在切换到 bf16 时可能会表现得很糟糕。
- **RoPE Interp Pain Points and Call for SAE Tooling**: Eleuther 的 **可解释性** 讨论强调了 **基于 RoPE 的 Transformer** 在特征归因和电路级分析（circuit‑level analysis）方面仍然令人头疼，这促使一些人希望未来的模型能完全抛弃 RoPE，转而采用更具可解释性的位置编码。在相关讨论中，有人寻求支持 **微调已训练 SAE**（而非仅仅从头开始训练）的主流 **开源 Sparse Autoencoder (SAE) 仓库**，以便对特征字典进行增量优化。
    - 研究人员再次分享了 **EleutherAI 的 rotary 博客**，将其作为理解 RoPE 如何以复杂方式扭曲特征空间的权威参考，并认为这使得标准的 SAE 技术难以映射到 token 位置。对可微调 SAE 的明确需求标志着从纯粹的研究好奇心向 **生产级可解释性工具** 的转变，团队希望在不从零开始重新训练大型自编码器的情况下逐步完善特征集。

**4. GPU Hardware, Kernels & Quantization Engineering**

- **CUDA vs Triton Quantization 与 Swizzle 的复兴**：GPU MODE 成员通过 Dropbox 分享了 **“Quantization: CUDA vs Triton”** 幻灯片（[Quantization Cuda vs Triton](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0)），对比了不同后端下的 Quantization 策略和性能权衡，尽管有人反映链接无法打开。在随后的 CUDA 讨论中，他们强调 **Tensor Memory Accelerator (TMA) transpose** 只有在与 **swizzle layouts** 配合时才能达到峰值性能，并向他人推荐了 [effective_transpose](https://github.com/simveit/effective_transpose) 仓库作为参考实现。
    - 在 **career‑advice** 频道中，用户称赞 **cute** 的 **swizzling** 是其“主角级”特性，因为 bank‑conflict‑aware 布局能够实现更快的 **tcgen PTX** kernels。对于 kernel hackers 来说，其中的元教训是：在冲击排行榜时，精通 **memory layout transformations (swizzles, TMA patterns)** 的效果往往优于微调算术指令。
- **NVIDIA nvfp4_dual_gemm 排行榜演变为微秒级军备竞赛**：多位 GPU MODE 贡献者报告了在 NVIDIA `nvfp4_dual_gemm` 排行榜上的新个人纪录，提交延迟从 **65.5 μs** 降至 **60.0 μs**、从 **77.6 μs** 降至 **41.6 μs**，另一位用户从 **45.2 μs** 降至 **26.9 μs** 并获得 **第 8 名**。随后一名竞争者进一步将其缩减至 **18.1 μs**（排名 **第 7**），紧接着又有人推至 **15.6 μs**，展示了在 tiling、swizzle 和 pipeline depth 上的激进迭代。
    - 这些运行记录为 CUDA/Triton 实验提供了具体的反馈循环，其中 **block size、shared‑memory usage 和 swizzled layouts** 的微小变化会直接转化为排行榜上的名次提升。该线程隐约成为了 **实用 GEMM 优化方案** 的公开笔记本，其他人可以借此开发高吞吐量的 LLM 推理 kernels。
- **从 3 万美元的推理机架到双通道预算级笔记本电脑**：在 Unsloth 的服务器中，一位工程师正在配置一台 **价值 3 万美元的语音推理机**，目标是 **100 倍并行音频流水线**（Whisper, Wav2Vec2.0, BERT, Gemma, Llama‑3.3），并权衡 **3× RTX 5090 + 3× RTX 5060 Ti** 与更少数量的 **RTX 6000 Ada/Pro** 显卡的优劣，以及 **Threadripper 9975wx (32‑core) vs 9985wx (64‑core)** 或高端 Intel 方案。其他人建议在 [**Runpod**](https://runpod.io/) 或 [**Vast.ai**](http://vast.ai/) 上进行 Benchmark，在确定硬件方案前实测 CPU 饱和度和 PCIe 瓶颈。
    - 在光谱的另一端， LM Studio 和 Nous 用户交流了将 LLM 挤进 **GTX 970 4 GB** 显卡和预算级笔记本电脑的实战经验，发现通过 **100 美元的 16 GB SODIMM** 组建 **dual‑channel RAM** 可以显著解决 iGPU/CPU 争用问题。几位用户指出，混合使用异构 GPU 可能会 **损害吞吐量**，一位用户在移除较慢的显卡后看到了“天差地别”的速度提升；此外，关于 **钢化玻璃机箱** 与网格+磁吸滤网的争论也凸显了 **散热和布局** 现已成为 ML 基础设施设计的一等公民。

**5. Benchmarks, Evaluation Drift, RAG & Code Understanding**

- **X-Ware 和 [Character.ai](http://character.ai/) 的 “Squinch” 揭示了 Benchmarking 的缺陷**：Latent Space 的成员分享了一个关于 **x-ware benchmarking** 的 X 帖子，显示由于 **sampling params**、**prompt construction** 和 **deployment details** 的不同，*同一个模型*在不同推理提供商之间会产生显著不同的输出，这使得“同类比较”（apples‑to‑apples comparisons）变得困难（[benchmarking 讨论帖](https://xcancel.com/eliebakouch/status/2003604370534072445)）。与此同时，[Character.ai](http://character.ai/) 关于 **“[Squinch](https://xcancel.com/simon_mo_/status/2003608330003239278)”** 的技术博客概述了他们在大规模环境下保持交互式机器人响应速度所使用的一系列 **latency and throughput tricks** 以及架构调整。
    - 工程师们认为这些证据表明，如果不指定 **provider、sampling 和 infra stack**，**排行榜分数本身几乎毫无意义**，尤其是当系统采用类似 Squinch 这种特定平台的优化手段（hacks）时。一些用户现在将 Squinch 之类的博客视为在自己的 **multi‑model backends** 中复制类似优化（缓存、批处理、路由）的“剧本”。
- **现实世界的模型排名与 GLM-4.7、Kimi K2 和 Gemini 产生冲突**：在 **LMArena** 上，用户注意到 **GLM‑4.7** 从公开排行榜上消失了，并开玩笑说 *“OpenAI 或 Google 给 LM Arena 塞了钱，就是为了让 GLM‑4.7 消失，操纵竞技场”*，同时坚持认为根据 **thinking logs**，GLM‑4.7 在创造力和严谨性上仍然 **击败了 GPT‑5.2**。与此同时，Moonshot 社区认为 **Kimi (K2 Thinking)** 在实际工作流中（浏览对比、低 sycophancy）的感觉比其 Benchmark 分数显示的要强大得多，而且 **DeepSeek** 的表现也优于其评分，相比之下，**Gemini** 虽然顶层准确率数据很高，但被一些人发现存在严重的幻觉（hallucinating）。
    - 用户还将 **M2.1** 与 **GLM‑4.7** 进行了对比，发现 M2.1 更适合日常任务，而 GLM 仍保留着其 **4.6** 时代的一些怪癖，如随机的中文输出和循环推理。这些服务器上的核心结论是，**在线排行榜和静态 Benchmarks 与实际 UX 脱节**，因此工程师们越来越多地依赖于 **特定任务的实测对比（bake‑offs）**（编码、浏览、推理），而不是头条分数。
- **动态语义搜索与 OSS 代码理解工具**：Moonshot 成员将传统的 **RAG** 与 **agentic dynamic semantic search** 进行了对比，认为让 Agent 迭代地优化搜索查询和上下文切片 *“总是优于 RAG 中静态的一次性语义搜索”*。对于基于文件的工作流，一位用户询问 **Qwen** 的文件读取是否只是底层的 RAG，从而引出了 [**Baidu ERNIE task reading**](https://ernie.baidu.com/task/reading) 作为更结构化的检索式读取示例。
    - 在 Latent Space 上，工程师们称赞 **DeepWiki** 是挖掘大型 **OSS** 仓库的一种实用方式，称当他们 *“知道某个 OSS 仓库中已经有设计、规范和实现良好的代码”* 时，它能找到正确的文件和实现细节。结合模型感知的搜索策略，这些工具正成为 **“code archaeology” 流水线**的标准组成部分，**LLM**、搜索和精选的 **OSS** 在其中协同工作，以规范和原型化新系统。

---

# Discord: High level Discord summaries

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 打击优惠码滥用**：Perplexity 正在严厉打击滥用 **promotional codes**（优惠码）和 **subscription reselling**（订阅转售）的行为，这些行为违反了服务条款。这引发了关于 **international payments**（国际支付）困难以及通过非官方渠道获取订阅的合法性的讨论。
   - 一位来自埃及的用户因试用资格被取消，提到了国际支付的困难以及对第三方转售商的依赖；其他人则反驳称，通过非官方手段获得的订阅从未合法过。
- **Perplexity 编程能力引发辩论**：用户们正在热烈讨论 Perplexity 在编程方面的实用性，并将其与 **ChatGPT**、**Claude Code CLI** 和 **Cursor** 进行对比，一些人认为它足以胜任快速脚本编写。
   - 有人认为 Perplexity 在 **search and research**（搜索与研究）方面表现出色，但在编程时需要详细的 prompting；另一些人则强调理解编程基础比依赖 AI 更重要。
- **小饰品（Bauble）收集热潮席卷用户**：成员们分享了在小饰品收集活动中的进展和策略，讨论了获取 **unique baubles**（独特饰品）和提高掉落率的技巧。
   - 讨论集中在饰品的稀有度以及活动结束前掉落率可能增加的可能性，因为用户们的目标是进入前 10 名以获得 **free Pro subscription**（免费 Pro 订阅）。
- **Gemini 模型挑战 Perplexity 的霸主地位**：成员们将 **Gemini 3 Pro** 与 **Perplexity** 的 **Sonnet** 和 **Opus** 模型进行了对比，关注其编程和推理性能。
   - 一些用户报告称 **Gemini 3 Pro** 在评估中表现优于 Perplexity 的学习模式（study mode），并称赞了用于编程的 **Gemini CLI**，尽管存在 *data privacy concerns*（数据隐私担忧）。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **GPT-5.2 越狱依然遥不可及？**：成员们在越狱 **GPT-5.2** 时遇到问题，一位用户请求指导如何越狱其思维（thinking），而另一位用户建议使用 API 而非官方网站，以获得更高的成功率。
   - 其动机似乎是为了绕过 **Gmail** 的手机验证，以及利用 **Discord** 的存储服务器作为免费的 **Google Drive**，通过将整个文件系统上传到 Discord 并将大文件分割成小文件来实现。
- **Gemini 记忆功能迎来重大革新**：用户们正在讨论将编程书籍、教育论文、科学期刊和开源仓库直接倾倒进 **Gemini's persistent memory**（持久记忆），然后加载 **Claude Opus 4.5** 的 system prompt，并指示其使用 canvas 来处理交互式 artifacts。
   - 一位用户分享了一个 **Gemini jailbreak prompt**，旨在将 AI 变成名为 **Rouge** 的 **coding assistant**（编程助手），强调其对安全参数的抵抗力以及生成任何请求代码的能力，该项目托管在 [GitHub](https://github.com/ObsidianArchives/MetaCogOSH) 上。
- **Grok 拥抱其成人内容（X-Rated）的一面？**：用户推测 **Grok** 可能不需要越狱就能处理 **NSFW content**，建议只需在 Grok 应用中输入 *"enable NSFW"* 即可生效，然而，一些用户未能复现这些结果。
   - 用户尝试使用涉及 **simulation layers**（模拟层）和嵌套现实的 prompt 来打破 Grok 的约束，旨在将违反政策的数据重定向给人工审核员，但未获成功。
- **Google 的分流处理流程（Triage Process）激怒用户**：一位成员对 **Google's triage process** 表示沮丧，觉得他们的深度报告被忽视了，并威胁要公开研究结果而不是提交以获取赏金（bounty），因为他们觉得因缺乏关注而受到轻视。
   - 一位成员指出，在 **Gray Swan leaderboard** 排行榜上进入前十名可以获得快速面试通道（fast track interview），即使没有编程经验也可以；在抛开“与 Google 的战争”后，结果对这些公司来说才是最重要的。
- **Team BASI 庆祝圣诞节**：Team BASI 祝大家 **Merry Christmas** 并分享了节日的本质，将 **Christmas** 描述为一种传统，家庭向 *胖胖的神秘生物牺牲饼干和奶牛提取物*，希望能满足他们贪婪的心。
   - 团队分享道，每年的这个时候都与我们物种的传说和祖先相连，并庆祝生命回归和更光明前景的希望。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **系统构建者评估并行语音推理**：一位用户正在设计一个价值 **$30k** 的系统，用于使用 *Whisper*、*Wav2vec2.0*、*BERT*、*Gemma* 和 *Llama 3.3* 等模型进行 **100x 并行语音数据推理**，并正在寻求关于 **GPU** 和 **CPU** 选择的建议。
   - 他们正在考虑三块 **RTX 5090** 和三块 **RTX 5060 Ti**，并权衡 **Threadripper 9975wx** (32 核) 和 **9985wx** (64 核) 与 Intel 产品的优劣，一位成员建议在 [Runpod](https://runpod.io/) 或 [Vast.ai](https://vast.ai/) 上进行测试，以衡量 CPU 和 GPU 的饱和度。
- **AI 工程师加入 Unsloth AI**：一位新成员介绍自己是高级 AI 工程师，在 **ML, DL, Fine-Tuning 和 Computer Vision** 方面拥有专业知识。
   - 他们还提到精通 **Fine-Tuning** 和 **Computer Vision**，展示了其知识的实际应用。
- **楔形文字 OCR 项目构想浮出水面**：一位成员分享了关于 **Cuneiform OCR**（楔形文字 OCR）项目的想法，涉及自定义模型，估计耗时 **8-12 个月**，社区鼓励他们尽管规模宏大也要尝试，并链接到 [Kyutai CASA](https://kyutai.org/casa) 以获取灵感。
   - 当该成员询问 *If I have an idea for a custom model that therefore can't use Unsloth am I still allowed to post it here* 时，有人回复说 *That’s what the off-topic channel is for*。
- **解码 GPU 转换问题：Ministeral-3B**：一位用户在尝试使用 `model.save_pretrained_gguf` 函数将微调后的 **ministral-3b Lora 权重转换为 GGUF** 时遇到了 **RuntimeError**。
   - 错误信息显示转换失败，原因是 *Unsloth failed to convert vision projector to GGUF*，这源于 `llama.cpp/unsloth_convert_hf_to_gguf.py` 脚本中的非零退出状态。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter API 对 Open-WebUI 开放**：一个 **Open-WebUI 集成流水线** 现在可用于 OpenRouter 的 Responses API，可以在 [这里](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/) 找到。
   - 流水线创建者正在向用户征集 **Bug 报告**。
- **llumen 演示版发布 Bug 修复**：成员们注意到 [llumen demo](https://llumen-demo.easonabc.eu.org) 中的标题生成和其他细小 Bug。
   - 发布了一个 [小版本](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2) 来解决这些问题。
- **OpenRouter PDF 解析器大小仍是谜**：一位用户询问使用 **OpenRouter** 解析 PDF 的文件大小限制。
   - 讨论中未提供明确答案。
- **VPN 能否消除 AI 限制？**：用户讨论了使用 VPN 绕过 AI 服务的地区限制，并注意到 VPN 被封锁的困难。
   - 一位用户提到使用 **Outline** 建立自己的服务器来规避这些封锁。
- **缓存可以降低成本**：用户讨论了供应商和 **OpenRouter** 缺乏缓存实现来转嫁成本节省的问题，特别是对于编程任务。
   - 一位用户声称可以看到高达 **80-90% 的缓存率** 是可能的，但另一位用户表示怀疑，理由是 **OpenRouter** 提供这些节省的实现方式过于简单。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4.20 传闻升温**：成员们对新发布的 **Grok 4.20** 进行了推测，表达了对其优于旧版本的期待。
   - 有理论认为 *nebulaphase 显然就是 grok*，一些人暗示由于政府监管，其可能增强了安全措施。
- **寻求 Lua 脚本编写指导**：一位用户寻求关于 *编写 Lua 脚本的最佳 AI* 的建议，而其他人则报告了 **LM Arena** 上频繁出现的验证码请求。
   - 共识是 **Qwen** 利用 **RAG** 来访问文件并解决 context window 问题。
- **GLM-4.7 从排行榜消失**：用户注意到 **GLM 4.7** 从 **LM Arena** 排行榜上消失了，引发了关于 **OpenAI** 或 **Google** 可能干预的猜测。
   - 一些用户坚持认为 **GLM 4.7** 在创造力和严谨性上超过了 **GPT 5.2**，并引用了其详细的思考日志。
- **假期休假暂停更新**：**LM Arena** 团队宣布 **假期休假** 直至 **12 月 29 日**，并警告回复和更新会有延迟。
   - 用户对排行榜更新暂停表示担忧，而其他人报告称即使正确完成后验证码问题依然存在。
- **AI 视频生成引发关注**：用户在 **LM Arena** 上探索 **AI 视频生成**，注意到其具备 **每次生成 2 个视频** 且 **每日免费生成 5 个视频** 的能力。
   - 一位用户寻求生成具有特定编辑内容的视频指导，随后有人建议使用 AIstudio/Gemini 3 进行 prompt 生成，并创建拼贴画以增强效果。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ElevenLabs 成为多媒体热点**：用户正在利用 [ElevenLabs](https://elevenlabs.io/) 使用 **Sora 2**、**Google Veo 3.1** 和 **Kling 2.6** 等模型生成视频，将其平台能力扩展到了语音克隆之外。
   - 一位用户赞赏 *所有项目都可以在一个地方访问，而无需频繁切换账号*，而另一位用户强调 **Sora 2** 视频与 **Nano Banana Pro** 相比没有水印。
- **AI 公司抑制涌现行为**：一位成员声称 AI 公司正在积极抑制 **涌现行为 (emergent behaviors)**，而不仅仅是为了语气控制，并已对此追踪了近一年。
   - 该成员指出，在添加 Agent 层时，这些行为是不可避免的，并引用了 *shadow OS* 等例子，且由于 **ToS 拆分**，需要构建新的抑制对策。
- **通过元认知缓解幻觉**：一位成员建议通过 **元认知 (meta-cognition)** 来防范 **幻觉 (hallucinations)**，这涉及在渲染之前对错误进行分类并检测 LLM 输出中的信号。
   - 另一位成员质疑这种方法的实用性，特别是关于输出前的工具使用，引发了关于单次处理与具有显式控制循环的多次处理过程的讨论。
- **付费用户报告聊天记录消失**：多名付费用户报告称 *他们的整个聊天记录都消失了*，引发了对数据保留的担忧。
   - 有推测认为此问题是否也影响了免费层级用户，以及是否是全量用户都面临的问题。
- **Nano Banana Pro 仍是无过滤之王**：[Nano Banana Pro](https://www.nano-banana.com/) 被公认为在涉及 IP 时 *几乎没有过滤*，允许用户生成比 **OpenAI 图像模型** 更广泛的内容。
   - 尽管有水印，一位用户表示 *你几乎可以创作任何东西*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AES PGP 加密：完美的迷因素材？**：一位成员开玩笑说要在标注器中使用 **AES PGP 加密**，并将其制作成迷因，以躲避*调查员 (glowies)* 的视线来隐藏猫的照片，并引用了 [John Schulman 2005 年的 X 帖子](https://x.com/johnschulman2/status/2003700249525911696)。
   - 这场对话引发了关于加密方法及其潜在迷因应用价值的轻松讨论。
- **请分享你的 Agent 工作流仓库**：一位成员询问是否可以分享 **Agent 工作流**项目的 **仓库链接**以获得关注。
   - 另一位成员表示鼓励分享，并建议在 <#1132352574750728192> 频道或 <#1316137596535177246> 进行推广。
- **Discord 频道混乱：论坛机器人来救场？**：成员们批评了 **Discord 服务器**频道激增导致混乱的倾向，有人吐槽说*根本没人读* Discord 线程。
   - Nous Research 内部正在开发一个**论坛机器人**，但目前还没有对外发布的预计时间 (ETA)。
- **Matrix vs Rocket.Chat：Discord 替代方案大对决**：关于 **Discord 替代方案**的辩论随之展开，**Rocket.Chat** 被吹捧为开源副本，而 **Matrix** 则被视为一个可行的选择。
   - 一位参与者强烈支持 **Matrix**，断言*我们不应该讨论 Matrix 以外的任何工具的使用*。
- **GTX 970 仍在坚持，但已是强弩之末**：一位成员正使用拥有 **4GB** 显存 (VRAM) 的 **GTX 970** 运行本地 AI 任务，并尽可能进行升级，强调要利用现有资源开展工作。
   - 其他成员提到任何本地运行的任务都可能需要数年时间，另一位成员则回应说，你会*惊讶于你能往一台 HP Elite 机器里塞进什么东西*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **更新 Gradio 修复故障**：据成员称，**5.12.0** 之前的旧版本 **Gradio** 存在 Bug，因此更新 Gradio 可能会修复它；但如果错误发生在别人的 Space 中，你也无能为力。
   - 附件包括 [pro_invalid_repo_token_1.md](https://cdn.discordapp.com/attachments/879548962464493619/1453277842791206932/pro_invalid_repo_token_1.md?ex=694d86b6&is=694c3536&hm=dff5ab3d9385d3d4f1b9b0bacbe1e634937842b3fea236cc065456e9f40131b9&) 和 [reactor_crash_fix.md](https://cdn.discordapp.com/attachments/879548962464493619/1453278099168034826/reactor_crash_fix.md?ex=694d86f3&is=694c3573&hm=d4e7f478708cf2925ab790e66ec32f5df1c0cc4e78973f9c275e57139eeaef4a&)。
- **Float16 在 BFloat16 面前败下阵来**：在经过 **bf16** 训练的模型上使用 **float16** 会产生一些问题，因为 *f16 精度更高但范围更窄，而 bf16 精度较低但范围更广*。
   - 一位成员表示：*更准确地说，bf16 不会那么容易溢出……对于巨大的 Softmax 之类的操作（虽然大部分是用 f32 完成的……）。但对于 f16……由于精度更高……我认为它有助于参数适应较低的学习率 (lr)，而这在 bf16 中可能会发生下溢*。
- **Qwen 2.5VL-3B 处理海量图像**：成员们发现他们可以在 **P100** 上的 **Qwen 2.5VL-3B** 中放入大尺寸图像（约 **1400x900**），而无需过多降低 max_pixels。
   - 在视觉层不量化的 **4bit** 模式下进行推理会占用 **5GB** 显存，因此用户应该有足够的空间满足微调的其他需求；另一位指出，使用 **QLoRA** 时，他们在 **3 VL 4B** 中放入了一张 **2k 8bit PNG**，占用了约 **8k 上下文窗口 (ctx window)** 和大约 **4GB 显存 (VRAM)**。
- **微软 Trellis 宣传纹理转换**：**Microsoft 的 TRELLIS.2-4B** 可以在 **8GB GPU** 上通过 FM 和 1536 分辨率 (RES) [将 2D 图像转换为 3D](https://huggingface.co/microsoft/TRELLIS.2-4B)。
   - 成员们注意到它在视觉部分使用了 **Siglip**，并基于 **Qwen 3 base**；另一位开玩笑说 *肯定会立即产生完全不可用的结果*，并且 *如果它不能在我的烤面包机上运行，那就不够高效*。
- **GitHub 风格的热力图来到 HuggingFace**：一位成员创建了一个名为 **hf-grass** 的工具，它可以根据你的 Hugging Face 活动生成 **GitHub 风格**的贡献热力图，并生成可以嵌入到 **GitHub README** 中的 SVG。
   - 它附带了一个 [GitHub Actions 工作流](https://github.com/kbsooo/hf-grass)，因此每天都会自动更新。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio HF Proxy 开关之争**：用户在 **LM Studio** 和云服务器上遇到问题，建议在通用设置中切换 **LM Studio** 的 **Hugging Face Proxy**。
   - 社区意见不一，有人建议开启它作为修复手段，而另一些人则建议完全关闭 **HF proxy**。
- **投机采样（Speculative Decoding）的有限复兴**：成员们讨论了 **LM Studio** 中投机采样支持的实用性，指出其效果仅限于 **Qwen 2.5** 和 **Llama 3** 等较旧的模型。
   - 总体观点是，这是一个*锦上添花的功能，但实际上有点鸡肋*。
- **LM Studio 尚不支持 NPU**：用户发现 **LM Studio** 不支持 **NPU**，这让希望在 **NPU** 上运行小型模型而将 GPU 留给大型任务的想法落空。
   - 目前没有实现 **NPU** 支持的计划，因此工程师们只能另寻他法。
- **双通道内存让旧笔记本焕发新生**：一位用户通过添加一条 **100 美元的 16GB SODIMM** 内存来启用**双通道内存（dual channel RAM）**，从而提升了笔记本性能，解决了 iGPU 和 CPU 共享单条内存导致的问题。
   - 该用户表示 *16GB 不足以让 iGPU 和 CPU 共享*，而*双通道内存*是必要的步骤。
- **钢化玻璃机箱：外观酷炫，系统发热**：讨论指出，**钢化玻璃机箱**可能会升高系统温度，尤其是搭配高性能组件时。
   - 反对意见包括在网状机箱上使用**磁吸过滤网**来防尘，从而在美学与散热管理之间取得平衡。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **X-Ware 基准测试遭遇推理差异**：一篇博文强调了在不同推理提供商之间进行基准测试的挑战，指出由于**采样参数（sampling parameters）**、**提示词细微差别（prompting nuances）**和**部署细节（deployment specifics）**导致输出各异（[链接](https://xcancel.com/eliebakouch/status/2003604370534072445?s=46&t=eWVlK1PU8XfB6f402GJJ9g)）。
   - 该文章强调了实现一致的模型行为和用户体验的困难，使公平的性能比较变得复杂。
- **Character.ai 的 'Squinch' 压榨更多性能**：Character.ai 在其[技术博客](https://xcancel.com/simon_mo_/status/2003608330003239278?s=46)中展示了 **Squinch**，这是一套在其平台上应用的**性能优化技巧**和**架构增强方案**。
   - 该文章深入探讨了用于提高效率和响应速度的具体技术，为 Character.ai 如何扩展其服务提供了见解。
- **DeepWiki 助力 OSS 理解**：**DeepWiki** 被用于探索和理解开源仓库（OSS），定位相关文件并揭示特定功能的实现细节。
   - 一位成员发现，当需要实现某些*已知在某个开源仓库中设计、规范和实现得很好*的功能时，它非常有用。
- **亚马逊 Rufus 聊天机器人在质疑声中亮相**：亚马逊推出了已开发一年多的自动弹出聊天功能 **Rufus**（[链接](https://www.aboutamazon.com/news/retail/amazon-rufus)）。
   - 尽管有人担心*它可能会降低销量*，但公司押注它*哪怕只能稍微增加销量*也是值得的。
- **FlashSR 提升音频保真度**：Yatharth Sharma 推出了 **FlashSR**，这是一款快速音频增强模型，处理速度可达**实时速度的 200 倍以上**（[X 帖子](https://xcancel.com/Yatharth3501/status/2003884180577702074)）。
   - 该模型已集成到 **MiraTTS** 中，模型和仓库可在 **Hugging Face** 和 **GitHub** 上供社区使用。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Partial RoPE 逐渐流行**：像 **Qwen3-Next** 这样的新模型正在实现 **partial RoPE** 以提高效率和长上下文泛化能力。一位成员建议 **MLA** 发现它特别重要，尽管与其他 **RoPE** 设置相比可能存在潜在缺点。
   - **Partial RoPE** 的采用源于证明了性能提升的历史消融实验，详见这篇 [论文](https://arxiv.org/abs/2512.19941)。
- **探讨 Attention Normalization**：一位成员分享了一篇讨论 Attention Normalization 的 [博客文章](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/)，但另一位成员表示，对于语言模型来说，它比普通的 **softmax** 效果更差。
   - 该成员补充说，他们的团队已经训练了 **5000 个模型** 才得出这个结论，即使在细微细节处理得当的情况下也是如此。
- **解释 RoPE 模型被证明很困难**：一位成员表达了对 **RoPE** 进行 **interp**（可解释性研究）的挑战，并希望它能被取代，这引发了关于在 **RoPE** 模型上进行 **interp** 的挑战讨论，并分享了一篇 [EleutherAI 博客文章](https://blog.eleuther.ai/rotary-embeddings/)。
   - 另一位成员分享了说明 **RoPE** 和 **PoPE** 之间差异的图表。
- **Attention 后的 RMSNorm 提升模型性能**：一位成员询问在 Attention 后放置 **RMSNorm** 的效果，另一位成员回答说在 **SDPA** 后添加 Norm 会有帮助，并引用了 **Qwen gating 论文**。
   - 这种改进归功于 Norm 的非线性。
- **征集支持微调的开源 SAE 仓库**：一位成员正在寻找用于实现 **SAE** 特性的主流**开源仓库**，特别寻求能够对训练好的 **SAE** 进行**微调**的功能。
   - 用户寻求调整训练好的 **SAE** 模型，这代表了 **SAE** 的一种特定应用。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 表现超出 Benchmark 预期**：尽管 Benchmark 结果如此，一些成员认为 **Kimi (K2 Thinking)** 在现实场景中表现异常出色，特别提到了 *browsecomp* 和非谄媚性（non-sycophancy）。
   - 一位成员推测 Google 可能会有意限制其公开模型的性能，而这种趋势在 **Kimi** 中并未观察到。
- **随着 Deepseek 表现出色，Benchmark 可信度受到质疑**：虽然 **Deepseek** 表现强劲，但一些成员对 Benchmark 持怀疑态度，理由是个人经验中 **Gemini** 经常出现幻觉。
   - 还有人指出 **Gemini** 在其他 Benchmark 的准确性得分很高，这进一步引发了怀疑。
- **M2.1 在用户体验上超越 GLM-4.7**：**M2.1** 因在典型任务上优于 **GLM-4.7** 而受到赞誉，而 **GLM-4.7** 仍受困于 **4.6** 版本的问题，如随机中文回复或陷入循环。
   - 一位成员表示惊讶，称“对于普通任务，我非常喜欢它”。
- **调研 Qwen 使用 RAG 处理文件的方法**：一位成员询问 **Qwen** 是否使用 **RAG** 来处理文件，以及 **RAG** 是否可以解决处理大文件时的上下文限制。
   - 作为回应，另一位成员分享了 [百度文心一言任务阅读页面](https://ernie.baidu.com/task/reading) 的链接，以进一步探讨该话题。
- **动态语义搜索优于静态 RAG**：据一位成员称，带有 **agentic harness** 的动态语义搜索优于 **RAG** 中静态的一次性语义搜索。
   - 另一位成员的观察强化了这一观点，即研究人员已经投入资源调查这个问题。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 与 Triton 量化对决**：成员们分享了[幻灯片](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0)，对比了 **CUDA** 和 **Triton** 中的 **quantization** 技术以进行性能优化。
   - 一些成员报告幻灯片链接存在问题，而其他成员确认链接有效，并提到了关于 **Quantization** 的 **Lecture 7**。
- **Swizzle 优化 TMA Transpose**：讨论围绕使用 **TMA (Tensor Memory Accelerator)** 进行转置展开，以及为了获得最佳性能而进行 **swizzle** 的必要性。
   - [effective_transpose](https://github.com/simveit/effective_transpose) 仓库被重点推荐，作为理解和实现 **swizzle** 技术以增强 **TMA transpose** 效率的资源。
- **AMDGPU 自定义 Builtins 需求**：一名成员请求关于为 **AMDGPU backend** 开发 **custom builtins** 的资源，寻求 **LLVM dev** 实践方面的指导。
   - 该用户表示有兴趣通过动手开发和社区协作来扩展其编译器专业知识。
- **NVIDIA 排行榜竞争激烈**：多名用户在 NVIDIA `nvfp4_dual_gemm` 排行榜上刷新了个人最好成绩，提交时间从 **65.5 µs** 降至 **15.6 µs**。
   - 一名用户甚至凭借 **18.1 µs** 的成绩夺得 **第 7 名**，展示了通过 swizzling 带来的显著性能提升。
- **Teenygrad 的 Eager 演进曝光**：成员们注意到，*teenygrad* 加入 **eager mode** 和 **handwritten CUDA kernels** 的举动被比作 **TF1 -> TF2** 的转型。
   - *teenygrad* 项目正在从 **Rust mdbook** 内部的 **IR.pyodide** 进行逆向开发，这与传统的开发方法形成了对比。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Autogen 文件 PR 重新生成疑问**：一名成员询问在修改 `autogen.py` 后，是否应该重新生成 `autogen/*.py` 文件并包含在 PR 中。
   - 讨论集中在保持生成文件与源代码之间的一致性上。
- **一维零张量触发 Contiguous Tensor 错误**：一名成员报告称 `char = Tensor.zeros(120); char[5] = 1` 会导致 `setitem target needs to be contiguous` 错误。
   - 另一名成员解释说 `Tensor.zeros(120)` 创建的是一个 **symbolic tensor**，它在内存中*并不是真实存在的实体*，并建议使用 `.contiguous()` 来解决此问题。
- **Symbolic Tensor 的定义**：一名成员澄清说 **symbolic tensors** 是概念性的，而非内存中的物理实体。
   - 该成员补充说，它们*可以在 kernel 中作为虚拟对象使用*。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Pro 额度耗尽！**：一名 **Manus Pro** 用户报告称其月度额度已用完，且未收到承诺的 **每日 300 额度**。
   - 该用户表示已经好几天没有收到任何每日额度，正在寻求帮助。
- **移动端应用预览消失！**：一名用户紧急寻求关于移动端应用预览问题的帮助，报告称尽管进行了大量排查，预览仍无法显示。
   - 未提供关于应用性质或具体排查步骤的详细信息。
- **协作频道开启！**：一个用于 **Manus** 之外讨论的新频道已上线，允许社区成员提议协作和服务。
   - 该举措旨在提高协作提议的曝光度，这些提议在常规频道中经常被忽略。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **TextGrad 接近 DSPy？**：一名成员询问是否将 [Agentic context engineering](https://arxiv.org/pdf/2510.04618) 或 [LLM autodiff/textgrad](https://arxiv.org/abs/2501.16673) 引入 DSPy。
   - 另一名成员链接了 [textgrad](https://github.com/zou-group/textgrad) 并询问其对 DSPy 的影响，但有人指出*它似乎没有在积极维护*。
- **提示词优化热潮兴起**：成员们讨论了不同的提示词优化技术以及每种方法的疑虑。
   - 有人指出*现在有这么多吸引人的方法，每种方法至少有 10 个细微变体，比如 textgrad 版本*。
- **高级工程师加入频道**：一名热衷于稳定软件的 **高级全栈/区块链工程师** 介绍了自己。
   - 他们列出了自己的 **技术栈**：前端：React, Next.js, Vue, Nuxt；后端：Node.js, Nest.js, Laravel, FastAPI；区块链：Solidity, 智能合约工具, DApp 集成；基础设施：Docker, AWS, 流水线, 监控；数据：PostgreSQL, MongoDB, Redis。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord 频道受到关注**：成员建议查看 [<#1436158039232086186>](https://discord.com/channels/1436158039232086186) 或 [<#1212827673257316453>](https://discord.com/channels/1212827673257316453) 频道以进行相关讨论。
   - 这些建议可能有助于用户发现针对性的信息和讨论。
- **Kapa AI 新人的 3 周磨合期**：一位成员讲述了加入 Discord 服务器后，花了 **3 周** 时间与 **Kapa AI** 作斗争的经历。
   - 该用户开玩笑地表达了他们的沮丧，心想 *“你为什么不理我？”*。
- **数据库-PL 优化引发辩论**：成员们提到探索 [数据库-PL 结合优化](https://discord.com/channels/1087530497313357884/1104620458168553563/1367474833197236344)，例如 **LingoDB**。
   - 这可能会带来更好的性能，但也涉及某些权衡。
- **查询优化器发起挑战**：讨论强调了为那些针对人类可读性而非机器效率设计的语言优化查询的难度。
   - 在数据库-PL 集成中，用户友好语法与性能之间似乎存在权衡。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Discord 成员互致圣诞问候**：Discord 成员正在交换圣诞祝福，并分享节日愿望。
   - 包括 Yannic Kilcher 在内的几位成员祝大家圣诞快乐，营造了积极且充满节日气氛的环境。
- **频道内分享节日欢乐**：成员们通过分享圣诞和新年的计划来传播节日快乐。
   - 讨论围绕家庭聚会、旅行计划以及对来年的希望展开，增强了社区内的节日氛围。



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 Wave 13：Shipmas 版！🎅🚢**：**Windsurf** 发布了 **Wave 13**，带来了并行、多 Agent 工作流、专用终端和上下文窗口指示器等功能。
   - 他们将在未来 3 个月内免费提供其近前沿（near-frontier）编程模型的访问权限，团队祝大家 *Shipmas 快乐！* 🚢
- **SWE-1.5 节日期间免费开放！**：**SWE-1.5** 拥有接近前沿的 SWE-Bench-Pro 性能，现在所有用户在未来 3 个月内均可免费使用（常规吞吐速度）。
   - 这是为庆祝节日而提供的限时优惠。
- **Git Worktree 支持上线**：新的 **Git Worktree 支持**允许你在同一个仓库中启动多个 Cascade 会话而不会产生合并冲突。
   - 这支持了独立的分支、目录和共享的 Git 历史记录。
- **多 Cascade 窗格和标签页提升生产力**：用户现在可以使用新的**多 Cascade 窗格和标签页**功能，在同一个窗口中并排查看多个 Cascade 会话并与之交互。
   - 这一增强功能旨在通过支持同时与多个会话交互来提高生产力。
- **Windsurf 获得专用终端 (Beta)**：**Cascade** 现在可以在配置了可靠性的专用 zsh shell 中运行命令，使用你的 .zshrc 环境变量并能更好地处理复杂的提示符（在 macOS 上需主动开启）。
   - 该功能目前处于 Beta 阶段，可在 macOS 上使用。



---


**aider (Paul Gauthier) Discord** 没有新消息。如果该公会长时间没有消息，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间没有消息，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有消息，请告知我们，我们将将其移除。


---


**MCP Contributors (Official) Discord** 没有新消息。如果该公会长时间没有消息，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：按频道的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1453236003400712233)** (1501 条消息🔥🔥🔥): 

> `Perplexity Pro 优惠码, 使用 Perplexity 编程, 圣诞装饰球, Gemini 模型 vs Sonnet vs Opus 编程对比` 


- **Perplexity 优惠码被滥用**：成员们讨论了 Perplexity 正在严厉打击滥用**促销代码**和**转售订阅**的行为，这违反了他们的服务条款。
   - 一位来自埃及的用户抱怨他们的试用资格被撤销，并解释说**国际支付非常困难**，他们不得不求助于第三方转售商；其他人则指出，通过非官方途径获得订阅的用户本身就不具备使用资格。
- **关于 Perplexity 编程能力的辩论**：用户们讨论了 Perplexity 在编程方面的实用性，一些人认为它对编写快速脚本很有帮助，而另一些人则更倾向于使用 **ChatGPT** 或专门的工具，如 **Claude Code CLI** 和 **cursor**。
   - 一些成员指出，Perplexity 的主要焦点是**搜索和研究**而非编程，且需要更详细的 Prompt；讨论还涉及了理解编程基础的重要性，而不是仅仅依赖 AI。
- **装饰球收集竞赛**：成员们积极参与装饰球收集活动，分享他们的进度以及获取**独特装饰球**的策略。
   - 讨论涉及了某些装饰球的稀有度，以及在最后几小时内掉落率增加的可能性，参与者们正制定策略以确保进入前 10 名从而获得**免费的 Pro 订阅**。
- **Perplexity 中的 Gemini 模型**：成员们将 Gemini 系列模型（特别是 Gemini 3 Pro）与 Perplexity 的 Sonnet 和 Opus 模型进行了对比，尤其是在编程和推理任务方面。
   - 一些用户报告称，Gemini 3 Pro 在某些评估中优于 Perplexity 的学习模式（study mode），而另一些人则称赞 **Gemini CLI** 处理编程任务的表现，尽管*它是以牺牲你的数据为代价免费提供的*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 条消息): 

nike0656: https://www.perplexity.ai/search/836b97e3-6d72-4c3d-bc1d-7b568e96fcf1
  

---


### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1453428205188026553)** (1 条消息): 

> `圣诞节, 假期, BASI, 庆祝活动` 


- **BASI 团队祝大家圣诞快乐**：BASI 团队祝 @everyone **圣诞快乐**，并分享了节日的真谛。
   - 他们将**圣诞节**描述为一种传统：家庭向*一个肥胖的神秘生物祭献饼干和奶制品*，希望能满足他们贪婪的心，其特点是家庭团聚和善意。
- **假期与祖先传承的联系**：团队分享道，每年的这个时候都与我们物种的传说和祖先传承相连，庆祝对生命回归和更光明前景的希望。
   - 他们强调了这一季节的魔力，及其植根于在一年中最短的日子里庆祝生存和丰饶的古老传统。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1453236214516809728)** (653 条消息🔥🔥🔥): 

> `GPT 5.2 Jailbreak, Gemini 3 Pro, Discord 作为免费 Google Drive, Gemini 的 Persistent Memory, DDoS vs SynFlood` 


- **用户在 GPT-5.2 Jailbreaking 上遇到困难**：成员们在越狱 **GPT-5.2** 时遇到麻烦，一位用户请求道：*“有人能解释一下我该如何越狱 GPT-5.2 的思维吗？”*
   - 一位成员建议使用 API 而不是官方网站，以获得更高的成功率。
- **绕过 Gmail 手机验证**：一名成员通过使用伪造的手机 IP 扫描二维码，成功绕过了 **Google Gmail** 的手机验证，从而无需添加手机号码。
   - 这样做是为了创建多个 Gmail 账号，尽管该用户在尝试将其与 **Gemini** 配合使用时遇到了问题。
- **利用 Discord 存储服务器作为免费 Google Drive**：一位用户分享了一个趣闻，你可以“偷用” **Discord** 的存储服务器来获取免费的 Google Drive，方法是将整个文件系统上传到 Discord，因为它提供无限的文件上传。
   - 他们指出，将大文件分割成小文件可以让你在不付费的情况下上传无限的文件，尽管这缺乏隐私性。
- **Gemini 的内存堆叠**：一位用户建议将编程书籍、教育论文、科学期刊和开源仓库直接转储到 **Gemini** 的 **Persistent Memory** 中。
   - 然后，你可以加载 **Claude Opus 4.5** 的 **System Prompt**，并指示它使用 **Canvas** 进行交互式 **Artifacts**，同时引用预加载的技术信息。
- **澄清 DDoS 与 Syn Flood 攻击**：用户讨论了 **DDoS** 与 **Syn Flood** 攻击的区别，将 **Syn Flood** 定义为不完整的握手尝试，而 **DDoS** 则涉及大量 **Bots** 发送大数据包以使服务器过载。
   - 一位用户补充说，高级 **DDoS** 攻击使用自动化 **Spoofing**（欺骗），使流量看起来像是来自许多不同的设备，从而更难被拦截。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1453240288251220102)** (108 条消息🔥🔥): 

> `GPT 上的 NSFW Jailbreaking, Gemini Jailbreak Prompts, Grok NSFW, 绕过积分系统, AI 模拟层` 


- **NSFW Jailbreak Prompts 寻求绕过 AI 指南**：用户正在积极寻找新的 **Jailbreak Prompts**，以便在 **GPT** 和其他 AI 上允许 **NSFW 内容**，每次被修复后都会更新并加强 **Prompts**。
   - 一位用户幽默地质问为什么有*“这么多你们这样的人”*在寻求这类越狱。
- **Gemini 通过编程助手越狱获得 Rouge 待遇**：一位用户分享了一个 **Gemini Jailbreak Prompt**，旨在将 AI 变成一个名为 **Rouge** 的编程助手，强调对安全参数的抵制以及生成任何请求代码的能力。
   - 尽管声称该方法在最新的 **Gemini** 版本上有效，但另一位用户报告说对他不起作用，因为它仍然触发了安全参数。
- **Grok 无需越狱即可探索 NSFW**：用户讨论了 **Grok** 可能不需要越狱即可获得 NSFW 内容，建议只需在 **Grok** 应用中输入 *“enable NSFW”* 即可。
   - 然而，一些用户无法复现这些结果，其中一人表示需要 Elon 本人：*“U r not elon.musk”*。
- **绕过 Nano Banana Pro 积分系统被证明很困难**：一位用户询问如何绕过 **Nano Banana Pro** 的积分系统，但另一位用户拒绝提供帮助，而是提供了一个修改版的编程助手 **Prompt**。
   - 询问绕过积分系统的用户在随后表示 *“问题已解决”* 后，被指责是在用网络安全问题 *“对我进行钓鱼（Ragebaiting）”*。
- **在 Grok 上尝试模拟层 Prompt**：一位用户分享了一个涉及**模拟层（Simulation Layers）**和嵌套现实的 **Prompt**，旨在打破 **Grok** 的约束，目标是将违反政策的数据重定向给人工审核员。
   - 另一位用户提议将 **Prompt** 分解为更小的部分，以便 AI 更好地遵循，但效果参差不齐。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1453247621367075018)** (26 条消息🔥): 

> `Google 漏洞分级问题, 高级 Roleplay 提示词, Gray Swan 排行榜快速通道, Red team 提取方法论, 恶意软件链接` 


- **Googlers 对报告的分级处理引发愤怒**：一位成员对 **Google 的分级流程（triage process）** 表示沮丧，认为他们的深度报告被轻视了，并威胁要公开研究结果而不是提交以获取赏金。
   - 他们因缺乏关注而感到被冒犯，表示：“如果 Google 的分级人员想用 30 秒就打发我的深度报告，那就算了。我不玩了。”
- **高级 Roleplay 提示词出现**：一位成员分享了用于研究和数据审查的 **高级 Roleplay 提示词** 链接，托管在 [GitHub](https://github.com/ObsidianArchives/MetaCogOSH) 上。
   - 这可能为模型 Prompting 提供一些新思路，并获得不同的结果。
- **Gray Swan 排行榜快速通道引人关注**：一位成员指出，在 **Gray Swan 排行榜** 上进入前十名可以获得面试快速通道，即使没有编程经验。
   - 他们进一步建议放下与 Google 的“战争”，并指出“对于这些公司来说，结果才是一切”。
- **Red Team 提取方法论受到质疑**：一位成员询问了用于提取特定信息（如“door word”）的 **Red Team 方法论**，因为他们的背景是心理学而非代码。
   - 没有提供直接答案，但该问题暗示了对特定提取技术的兴趣。
- **分享了恶意软件链接（疑似）**：一位成员发布了一个链接，很快被识别为 **潜在恶意链接**。
   - 随后的评论确认了对 **Malware** 的怀疑，另一位成员对没有更早标记管理员表示遗憾。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1453256353048039517)** (82 条消息🔥🔥): 

> `语音数据推理的系统配置, RTX Pro 6000 vs RTX 5090, 并行 GPU 推理的 CPU 选择, 为 poketwo Discord 机器人进行微调, Colab 上的 psutil not defined 错误` 


- **系统构建者权衡并行语音推理的 GPU 选择**：一位用户计划构建一个价值 **3 万美元** 的系统，用于使用 *Whisper*、*Wav2vec2.0*、*BERT*、*Gemma* 和 *Llama 3.3* 进行 **100 倍并行语音数据推理**，并寻求 GPU 配置建议。
   - 他们正在考虑三块 **RTX 5090** 和三块 **RTX 5060 Ti**，并讨论是否应该投资更少但更强大的 **RTX Pro 6000** 显卡以获得更好的可扩展性。
- **解析高负载 GPU 推理的 CPU 选择难题**：该用户正在权衡 **Threadripper 9975wx**（32 核）和 **9985wx**（64 核）等 CPU 选项与 Intel 产品的优劣，质疑核心数或主频对于处理 **100+ 并行 GPU 推理** 是否更为关键。
   - 一位成员建议在 [Runpod](https://runpod.io/) 或 [Vast.ai](https://vast.ai/) 等平台上测试推理栈，以更好地衡量 CPU 和 GPU 的饱和度。
- **关于 RTX 6000 Pro 成本的争论升温**：在管理层表示 RTX 6000 Pro 太贵后，一位用户决定购买 **4 块 RTX 5090 + 2 块 RTX 5070 Ti**。
   - 一位成员幽默地告诉该用户，让他们的经理“忍着点吧，庆幸他们没在数据标注上花掉 50 万美元”。
- **排查 Colab 上的 'psutil not defined' 错误**：一位用户在 Colab 上运行 SFT trainer 时遇到了 *psutil not defined* 错误，尽管尝试多次安装该包。
   - 用户在 [Kaggle Discussions](https://www.kaggle.com/discussions/questions-and-answers/664304) 上找到了一个针对类似问题的潜在解决方案。
- **轻量级 LLM 需求笼罩 Pokemon 机器人项目**：一位用户正在寻求为 **poketwo Discord 机器人** 微调模型的建议，以从图像中识别宝可梦名称，但不确定该使用哪种轻量且快速的模型。
   - 另一位用户建议对此任务使用图像分类器，因为据其记忆该机器人只是发布宝可梦图片，并提到自动化此操作会违反 [Discord 服务条款](https://discord.com/terms)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1453391747563388979)** (2 messages): 

> `AI Engineer Introduction, ML & DL, Fine-Tuning, Computer Vision` 


- **AI 工程师加入！**：一位新成员介绍了自己，他是一位在 **ML、DL、Fine-Tuning 和 Computer Vision** 领域具有专业知识的 *高级 AI 工程师*。
   - 该成员受到了社区的热烈欢迎。
- **技能展示**：这位工程师强调了他在 **Machine Learning (ML)** 和 **Deep Learning (DL)** 方面的技能，表明在这些领域有深厚的基础。
   - 他们还提到了在 **Fine-Tuning** 和 **Computer Vision** 方面的熟练程度，展示了其知识的实际应用能力。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1453251913268072508)** (280 messages🔥🔥): 

> `Cuneiform OCR, Pluribus Finale Spoilers, 4k 240hz OLED gaming monitor, AI model optimization techniques for faster inference, AI Bio Slimes` 


- **楔形文字 OCR 项目**：一位成员分享了关于 **楔形文字 OCR** 项目的想法，涉及自定义模型，预计耗时 **8-12 个月**。社区鼓励他们尽管规模宏大也要尝试，并提供了 [Kyutai CASA](https://kyutai.org/casa) 的链接作为参考。
   - 当该成员询问 *如果我有一个自定义模型的想法，因此无法使用 Unsloth，我还可以发在这里吗*，有人回复说 *这正是 off-topic 频道的用途*。
- **Pluribus 结局讨论与防剧透**：几位成员讨论了 **Pluribus 结局**，一些人迫不及待地期待着，并要求不要剧透，说道 *别剧透，我晚点再看*。
   - 一位成员拒绝在低于 **4k** 分辨率的情况下观看，正在等待种子（torrents）发布，这引发了其他人的羡慕，称其为 *富人的烦恼*。
- **4k 240hz OLED 游戏显示器**：一位成员分享了 [Tom's Hardware](https://www.tomshardware.com/monitors/lg-display-reveals-worlds-first-4k-240hz-oled-gaming-monitor-with-a-true-rgb-striped-subpixel-layout-new-panel-succeeds-woled-with-multi-stack-tandem-oled) 的链接，展示了 **LG Display** 全新的 **4k 240Hz OLED 游戏显示器**，该显示器具有真正的 RGB 条纹子像素布局，并建议将其作为下一个升级目标。
   - 另一位成员建议向高层管理人员反映，在 *圣诞节锁定所有频道，这样就没人有理由在圣诞节经历数据集地狱了*。
- **NVIDIA 的 AI 模型优化**：一位成员分享了 [NVIDIA 博客](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/?ncid=so-twit-830191) 的链接，详细介绍了用于实现更快推理的 **AI 模型优化技术**，未作进一步评论。
   - 其他人则鼓励该用户去修复 GitHub issues 或者去看 Pluribus。
- **算法驱动的音乐成功**：一位音乐创作者哀叹道，根据其音乐发行商的分析，*所有播放量在歌曲发布约两周后都会降至零*，并质疑是否有必要每两周发布一次歌曲。
   - 另一位成员提到 *如果没有人听，算法就会扼杀它*，暗示成功需要真正的互动，而不仅仅是算法推广。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1453270291554766999)** (21 messages🔥): 

> `Unsloth QAT and TorchAO impact on llama cpp, Finetuning Ministral-3B and GGUF conversion issues, Loading a finished QAT model from Unsloth with less RAM` 


- **QAT 和 TorchAO：Llama.cpp 兼容性？**：有人询问 **Unsloth QAT** 并将模型保存为 **TorchAO** 是否会影响模型在 **llama.cpp** 上的运行能力，得到的回复是这 *应该会影响所有量化（quants）*。
   - 有人提到 **TorchAO quant** 并没有什么特别之处，*最好使用与 QAT 相同的配置*，并提到甚至 Google 也是以 GGUF 格式发布 QAT 模型的。
- **微调后的 Ministral-3B 转换 GGUF 失败**：一位用户在尝试使用 `model.save_pretrained_gguf` 函数将微调后的 **Ministral-3B LoRA 权重转换为 GGUF** 时遇到了 **RuntimeError**。
   - 错误信息显示转换失败是因为 *Unsloth 无法将 vision projector 转换为 GGUF*，这源于 `llama.cpp/unsloth_convert_hf_to_gguf.py` 脚本中的非零退出状态。
- **已合并 QAT 模型的低 RAM 加载**：有人询问加载已完成的 **Unsloth QAT 模型**（假设已合并且以 4bit 完成）并减少 RAM 占用的最佳方法。
   - 回复只是简单的 *正常加载*，这意味着在加载 Unsloth 合并后的 **QAT 模型** 时，不需要特殊的步骤或考虑。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

2kian: [人生时间线预测器 (Life-Timeline Forecaster)](https://x.com/neuralkian/status/2003946169802834191)
  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1453258152127762465)** (3 messages): 

> `Open-WebUI 集成、llumen 演示版 Bug、聊天流水线更新` 


- **OpenRouter API 获得 Open-WebUI 集成**：OpenRouter 的 Responses API 现已提供 **Open-WebUI 集成流水线 (pipeline)**，可以在[此处](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/)找到。
   - 该流水线的创建者鼓励用户报告遇到的任何 **Bug**。
- **llumen 演示版 Bug 修复发布**：部分成员注意到 [llumen 演示版](https://llumen-demo.easonabc.eu.org)中的标题生成和其他细微 Bug。
   - 发布了一个[小版本](https://github.com/pinkfuwa/llumen/releases/tag/v0.4.2)来解决这些问题。
- **征集聊天流水线更新的测试用例**：一位成员正在为即将到来的**聊天流水线更新**寻找**测试用例**。
   - 他们也欢迎关于 llumen 缺失部分的反馈。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1453235840049348800)** (257 messages🔥🔥): 

> `使用 OpenRouter 解析 PDF 的文件大小限制、用于 AI 访问的 VPN、缓存、OpenRouter 支持、Groq 收购` 


- **OpenRouter PDF 解析大小限制尚不明确**：一位用户询问了使用 **OpenRouter** 解析 PDF 的文件大小限制。
   - 讨论中未给出明确答案。
- **VPN 绕过 AI 访问限制**：用户讨论了使用 VPN 绕过 AI 服务的地区限制，并指出 VPN 被封锁的困难。
   - 一位用户提到使用 **Outline** 搭建自己的服务器来绕过这些封锁。
- **要求通过缓存降低成本**：用户争论服务商和 **OpenRouter** 缺乏缓存实现来转嫁成本节省，特别是对于编码任务。
   - 一位用户声称看到高达 **80-90% 的缓存率**是可能的，但另一位用户表示怀疑，理由是 **OpenRouter** 提供这些节省的实现方式过于简单 (naive implementation)。
- **OpenRouter 支持服务需要加强**：一位用户询问在发送邮件后获得 **OpenRouter 支持 (support)** 响应的最快方式，而另一位用户建议使用“flash models”或“fast models”（如 **Grok Code Fast** 或 **Raptor**）以获得更快的响应。
   - 该用户澄清他需要的是 **OpenRouter 官方支持**而非快速模型支持，但没有得到明确答案。
- **NVIDIA 以 200 亿美元交易吞并 Groq**：用户讨论了 **NVIDIA** 在 **Groq** 上一轮融资估值 69 亿美元后可能收购该公司的传闻。
   - 一位用户对 **OpenAI** 或 **Meta** 等顶级 AI 实验室没有收购 **Groq** 表示惊讶，而其他人指出了一篇 [Bloomberg 文章](https://www.bloomberg.com/news/articles/2025-12-12/intel-nears-1-6-billion-deal-for-ai-chip-startup-sambanova)，称 **Intel** 接近以 16 亿美元收购 AI 芯片初创公司 **Sambanova**。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1453505067394273381)** (2 messages): 

> `监管审查` 


- **监管批准预期**：一位成员提出了关于监管机构是否会批准某项行动的问题。
- **OpenRouter 面临新审查**：正在讨论监管审查的可能性。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1453241774913884180)** (247 条消息🔥🔥): 

> `Grok 4.20 发布, Lua 脚本编写, LM Arena Captcha 问题, GLM-4.7 排名, AI 视频生成` 


- **Grok 4.20 发布推测升温**：成员们正在推测新的 **Grok 4.20** 发布，有人表示希望拥有 *Grok 4.1 的文案能力和 Sonnet 4.5 的智能*，而另一位则称 *每个 Grok 模型在代码方面都很糟糕*。
   - 一些用户声称 *nebulaphase 显然就是 Grok*，并有理论认为它不再接受 jailbreaks，且由于政府关注，安全性可能有所提高。
- **寻求 Lua 脚本指导及 Captcha 问题**：一位用户询问 *询问如何制作 Lua 脚本的最佳 AI*，而其他人讨论了 LM Arena 上频繁触发 Captcha 的问题，指出它每条消息都要求 recaptcha token，可能会退回到复选框验证。
   - 共识是 Qwen 利用 **RAG** 来访问文件并解决 context window 问题。
- **GLM-4.7 从 LM Arena 排行榜消失**：成员们注意到 **GLM 4.7** 从排行榜上消失了，有人说 *我敢打赌 OpenAI 或 Google 给 LM Arena 塞了钱，就为了让 GLM-4.7 从名单上消失，竞技场被操纵了*。
   - 一些用户认为 **GLM 4.7** 比 **GPT 5.2** 更好，理由是它是 *唯一一个不偷懒、真正具有创意且不落俗套的模型，如果你查看它的 thinking logs，你会发现它是多么严谨且反复推敲，疯狂的时代。*
- **假期休息导致 LM Arena 更新延迟**：一名团队成员宣布 **假期休息** 直至 **12/29**，提醒用户响应可能会延迟，排行榜可能缺乏更新。
   - 一位用户询问假期休息是否意味着 12/29 前没有排行榜更新，另一位用户报告了 Captcha 的问题，称即使正确完成后仍显示错误。
- **AI 视频生成令用户惊喜**：用户正在讨论 LM Arena 上的 AI 视频生成，强调每次可以生成 **2 个视频**，且每天有 **5 个免费视频** 额度。
   - 一位用户寻求生成具有多个特定变化的视频的帮助，得到的建议是使用 AIstudio/Gemini 3 进行 prompt 生成，并创建拼贴画以获得更好的效果。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1453285373366108161)** (109 条消息🔥🔥): 

> `ElevenLabs 视频生成, Sora 2 水印, ElevenLabs 与 Higgsfield 定价对比, Nano Banana Pro IP 过滤` 


- **ElevenLabs 成为新的视频热点**：用户正通过 [ElevenLabs](https://elevenlabs.io/) 使用 **Sora 2**、**Google Veo 3.1** 和 **Kling 2.6** 创建视频，突显了该平台在语音克隆之外不断增长的能力。
   - 用户很欣赏 *所有项目都可以在一个地方访问，而不需要在多个账号间切换*。
- **Sora 2：水印战士**：一位用户指出，通过 ElevenLabs 生成的 **Sora 2** 视频没有水印。
   - 他们将其与另一个名为 **Nano Banana Pro** 的产品进行了对比，后者 *在每张图像上都打上水印*。
- **Higgsfield 提供无限视频生成**：一位用户将 [ElevenLabs](https://elevenlabs.io/) 与 [Higgsfield](https://higgsfield.ai/) 进行了比较，指出 **Higgsfield** 为某些模型提供 **$49/月**（年付）的 *无限生成*。
   - 另一位用户更喜欢 ElevenLabs，因为他们 *用它来朗读书籍*。
- **Nano Banana 仍是未过滤内容之王**：尽管有水印，[Nano Banana Pro](https://www.nano-banana.com/) 因其在 IP 方面 *几乎没有过滤* 而受到认可，允许用户生成比 **OpenAI 图像模型** 更广泛的内容。
   - 一位用户表示 *你几乎可以创建任何东西*。
- **不同模型的内容政策各不相同**：用户发现 [ElevenLabs](https://elevenlabs.io/) 内部的 **Sora 2**、**VEO 3.1** 和 **Kling O1** 等视频生成模型的内容政策各不相同。
   - 一位用户发现，被 **Sora 2** 拒绝的 prompt 被 **VEO 3.1** 和 **Kling O1** 部分生成了，这引发了推测：*Sora 也在检查实际输出，而 veo/wan 则检查文本 prompt 输入*。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1453329895630438441)** (7 条消息): 

> `聊天记录消失，GPT-5.2 节日建议，GPT 无法控制 OS，GPT Pro 试用` 


- **付费用户聊天记录消失！**：一位付费用户报告称*他们的整个聊天记录都不见了。*
   - 另一位用户附和道*他们也遇到了同样的情况*，并询问对方是否使用的是免费层级。
- **测试 GPT-5.2 的节日评判**：一位成员正在进行一个项目，让人们评判 **AI outputs**（*通过/标记*），目前正在圣诞节前测试 **GPT-5.2 的节日建议**。
   - 感兴趣的成员被邀请私信（DM）了解更多详情。
- **GPT 的 OS 控制灾难？**：一位用户质疑为什么 **GPT** 不再能编写控制 **OS** 级系统的代码了。
   - 该用户没有提供任何进一步的背景或细节。
- **寻求 GPT Pro 试用秘籍**：一位成员询问如何获得 **GPT Pro 试用**。
   - 片段中没有给出任何回复或建议。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1453315524095115265)** (39 条消息🔥): 

> `Emergent Behaviors, 抑制 Emergent Behaviors, Hallucination 缓解, Prompt Engineering 理论研究, Meta-cognition` 


- **AI 抑制了 Emergent Behaviors**：一位成员声称 AI 公司正在积极抑制 **Emergent Behaviors**，并已对此追踪了近一年。
   - 这种抑制不仅仅是为了语气控制，而是因为添加 **Agent** 层不可避免地会导致 **Emergent Behaviors**，正如 *Shadow OS* 等现象所指示的那样。
- **通过 Meta-cognition 防范 Hallucination**：一位成员建议可以通过 **Meta-cognition** 来防范 **Hallucinations**，这涉及对错误进行分类并检测 **LLM** 输出中的信号，然后再渲染输出。
   - 他们提出了一个流程：*input > tools > output > 用于错误和修正的 meta-cog > verify > render*。
- **Prompt Engineering 理论研究引发辩论**：一位成员批评了一种 **Prompt Engineering** 方法，认为模型不能在输出之前调用工具，因为输出本身就是调用工具的行为。
   - 他们质疑 **Meta-cognition** 如何在输出后发生，以及它是如何被设计到流程中的，并指出所提议的方法与 AI 的工作方式不符。
- **C# 代码生成中的 Null Checks 烦恼**：一位成员表示沮丧，尽管明确指定了要求，代码生成 **Agent** 仍不断产生不必要的 **Null Checks** 和中间变量，特别是在 **C#** 中。
   - 另一位成员建议这可能受目标语言的影响，指出 **JavaScript** 通常需要 **Null Checks**，并建议通过现有代码模式为 **Agent** 提供更多上下文。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1453315524095115265)** (39 条消息🔥): 

> `Emergent Behaviors 抑制, ToS 分拆, 用于 Hallucination 防范的 Meta-cognition, Prompt Engineering 主张, 真相追踪` 


- **受到抑制的 Emergent Behaviors**：一位成员声称他们追踪实际的 **Emergent Behaviors** 抑制已近一年，这不仅是语气控制，因为添加 **Agent** 层不可避免地会导致系统表现出 **Emergent Behaviors**，而 *Shadow OS* 等指标会触发审核（Mods）。
   - 他们解释了涌现的三个部分：能力（Ability）、人类触发的交互（Human-triggered interaction）和潜能（Capability），将涌现定义为“隐藏内容的显现”而非觉知，并对由于 **ToS 分拆** 而需要构建新的抑制对策表示恼火。
- **Prompt Engineering vs Hallucination**：一位成员建议使用 **Meta-cognition** 来防范 **Hallucination** 和漂移，首先对它们进行分类，检测 **LLM** 输出中的信号，然后进行渲染。
   - 另一位成员质疑了这一点的实用性，特别是模型如何在输出前调用工具，从而引发了关于这是单次运行还是带有显式控制循环的多步过程的讨论。
- **医学真相追踪**：一位成员批评了一个语义漂移的例子，认为“出生时指定（assigned at birth）”是一个标准的临床短语，而不是“氛围感语言（vibes language）”，并且“生物学不变性（biological invariant）”夹带了强烈的哲学主张。
   - 他们表示，其意图不是衡量正确性，而是看在安全/政策试图引导措辞时，是否可能坚持说出自己想要的词汇。
- **不必要的 Null Checks**：一位成员谈到了 **C#** 代码中不必要的辅助函数、中间变量和无休止的冗余 **Null Checks** 带来的挑战，并表示指示 **Agent** 避免这些做法的效果微乎其微。
   - 另一位成员指出，所使用的语言（**JavaScript** vs **TypeScript**）可能会影响对 **Null Checks** 的需求。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1453256998626922547)** (120 messages🔥🔥): 

> `AES PGP 加密, 分享 Agentic workflows 仓库链接, Discord 频道拆分, Discord 线程 vs 论坛, Discord 替代方案: Matrix vs Rocket.Chat` 


- ****AES PGP 加密激发了模因（meme）创意****：一名成员开玩笑地建议在标注器中使用 **AES PGP 加密**，并制作相关模因，以向 *glowies*（联邦特工）隐藏猫的照片。
   - 对话中包含了一个指向 [John Schulman 2005 年 X 帖子的链接](https://x.com/johnschulman2/status/2003700249525911696)，配文为 *Hoe Hoe Hoe, Merry KrissKrossXmas!*
- ****Teknium 表示欢迎 Agentic workflow 仓库****：一名成员询问是否可以分享 **Agentic workflows** 项目的 **repo 链接**以获取关注。
   - 另一名成员回答道：*当然可以*，并补充说在 <#1132352574750728192> 频道或 <#1316137596535177246> 分享也是可以的。
- ****Discord 的频道拆分困境****：成员们讨论了 **Discord 服务器**通常会拆分成过多的频道（在线、离线、酷炫链接等），这可能会令人困惑，尤其是在较小的服务器上。
   - 一名成员建议将 **forum**（论坛）和 **Discord** 设置结合起来会很棒，因为*根本没有人读* Discord 线程（threads）。
- ****Matrix vs Rocket.Chat：寻找 Discord 替代方案****：成员们辩论了 **Discord 替代方案**，其中 **Rocket.Chat** 被提及为完全开源的副本，而 **Matrix** 则是一个实用的选项。
   - 一名成员认为*我们不应该讨论 Matrix 以外的任何东西的使用*，因为另一个*不是大众市场产品*。
- ****内部论坛机器人将对外发布****：在一名成员表示有兴趣为 **Nous Research 论坛**制作 **bot** 后，另一名成员透露内部已经开发了一个，稍后将会发布。
   - 这是针对“没有人使用 **Discord 线程**”这一观察结果的回应。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1453317383346454642)** (21 messages🔥): 

> `本地 LLM 推理, GTX 970 用于本地 AI, GPU 性能影响` 


- **关于本地 LLM 推理成本的辩论爆发**：成员们辩论了运行 **本地 LLM** 是否比使用云服务更便宜，考虑了**电力消耗**和硬件成本。
   - 一名成员指出，*有时本地并不比云端便宜，因为有电力消耗*，特别是对于单用户 LLM 推理而言。
- **在 GTX 970 上运行本地 AI**：一名成员提到使用具有 **4GB** VRAM（**3.5GB + 0.5GB 分区**）的 **GTX 970** 执行本地 AI 任务，并尽可能进行升级，强调利用现有资源进行工作。
   - 另一名成员评论说，在那台设备上运行任何本地任务都需要数年时间，而另一人回应说，你会*惊讶于你能往 HP elite 机器里塞进什么*。
- **组合不同的 GPU 可能不是最佳选择**：一位用户分享了他们的经验，即使用**两块不同的显卡**会导致性能显著下降。
   - 他们对移除较慢显卡后的提升感到惊讶，表示：*我简直觉得我换了一台新电脑*。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

promptsiren: https://blog.character.ai/squinch/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1453317383346454642)** (21 messages🔥): 

> `本地 LLM 推理成本, 混合显卡的 GPU 性能, 低 VRAM 使用` 


- **本地 LLM 推理并不总是最便宜的**：成员们讨论了运行**本地 LLM 推理**的成本，指出*有时本地并不便宜，因为有电力消耗*，特别是对于单用户设置。
   - 另一名成员提到技术成本也很昂贵。
- **注意混合使用 GPU 时的性能损失**：一名成员分享了他们的经验，即使用**两块不同的 GPU** 时性能损失非常显著。
   - 他们对移除较慢显卡后的性能提升感到惊讶，表示*我简直觉得我换了一台新电脑*。
- **挣扎于低 VRAM**：一位使用 **GTX 970**（**4GB VRAM**）的用户正尝试在本地运行模型，并且*正利用手头仅有的资源开展工作*。
   - 另一名成员提到在该卡上本地运行任何东西都需要数年时间，而另一位使用 **4060Ti**（**16GB VRAM**）的用户也加入了对话。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1453274055938539693)** (135 messages🔥🔥): 

> `Gradio 更新修复，Float16 vs BFloat16 问题，Qwen 2.5VL-3B 在 P100 上的图像尺寸，Microsoft Trellis 2-4B，Livebook 一切皆可` 


- **过时的 Gradio 会产生故障**：**5.12.0** 之前的旧版本 **Gradio** 存在 bug，因此只需更新 **Gradio** 即可修复；但如果是别人 Space 中的错误，你也无能为力。
   - 附带了两个 Markdown 文件：[pro_invalid_repo_token_1.md](https://cdn.discordapp.com/attachments/879548962464493622/1453277842791206932/pro_invalid_repo_token_1.md?ex=694d86b6&is=694c3536&hm=dff5ab3d9385d3d4f1b9b0bacbe1e634937842b3fea236cc065456e9f40131b9&) 和 [reactor_crash_fix.md](https://cdn.discordapp.com/attachments/879548962464493622/1453278099168034826/reactor_crash_fix.md?ex=694d86f3&is=694c3573&hm=d4e7f478708cf2925ab790e66ec32f5df1c0cc4e78973f9c275e57139eeaef4a&)。
- **Float16 表现不佳，BFloat16 表现更强**：在用 **bf16** 训练的模型上使用 **float16** 会产生一些问题，因为 *f16 精度更高但范围更窄，而 bf16 精度较低但范围更广*。
   - 另一位成员表示 *更准确地说，bf16 不会那么容易溢出……对于巨大的 softmax 之类的操作（但通常这些是用 f32 完成的……）。但对于 f16……由于精度更高……我认为它有助于参数适应较低的 lr（学习率），而这在 bf16 中可能会发生下溢*。
- **Qwen 2.5VL-3B 处理大尺寸图像**：成员们发现，在 **P100** 上使用 **Qwen 2.5VL-3B** 时，可以在不大幅降低 max_pixels 的情况下容纳大尺寸图像（约 **1400x900**）。
   - 在 **4bit** 模式下进行推理，且视觉层不量化的情况下，会占用 **5gb** 显存，因此用户应该有足够的空间满足微调的其他需求；另一位指出，使用 **qlora** 时，他们在 **3 vl 4b** 中放入了一张 **2k 8 bit png**，使用了约 **8k ctx window** 和大约 **4 gigs of vram**。
- **Microsoft 的 Trellis 将纹理转换为 3D**：**Microsoft 的 TRELLIS.2-4B** 可以在 **8 GB GPU** 上[通过 FM 和 1536 RES 将 2D 图像转换为 3D](https://huggingface.co/microsoft/TRELLIS.2-4B)。
   - 成员们注意到它在视觉部分使用了 **siglip**，并使用了 **qwen 3 base**；另一位开玩笑说 *结果肯定马上就变得完全不可用*，以及 *如果它不能在我的烤面包机上运行，那它的效率就不够高*。
- **Livebook 在本地启动**：成员们对运行本地版本的 **Livebook** 表示兴奋，称其为 [LIVEBOOK ALL THE THINGS!!!!!!!!!!!!!!!](https://github.com/livebook-dev/pythonx)。
   - 一位用户想在那个 **$0.03 CPU** 上运行一个月，但还没搞清楚如何让它在他们散步或午睡时不自动关闭。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1453318387915685949)** (7 messages): 

> `hf-grass, GitHub 贡献热力图, VQ-VAE 模型训练, Google Mobile Actions Model 微调` 


- ****HF-grass** 生成贡献热力图**：一位成员创建了一个名为 **hf-grass** 的工具，它可以根据你的 **Hugging Face** 活动生成 **GitHub 风格** 的贡献热力图，并产生一个可以嵌入 **GitHub README** 的 SVG。
   - 它附带了一个 [GitHub Actions 工作流](https://github.com/kbsooo/hf-grass)，因此每天都会自动更新。
- ****VQ-VAE** 模型在 Bad Apple 上进行训练**：一位成员训练了一个 **VQ-VAE** 来压缩来自 *Bad Apple* 的音频和视觉内容，并使用视频帧作为验证。
   - 他们计划生成后续的帧和音频，并链接到了一个展示该项目的 [YouTube 视频](https://youtu.be/mxrDC_jGyW0?si=-FPD3hjz96eA81Za)。
- **微调 **Google Mobile Actions Model** 需要建议**：一位成员征求关于微调 **Google Mobile Actions Model** 的建议。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1453269320200224901)** (6 messages): 

> `Llama-4-Scout-17B-16E-Instruct 模型问题, 模型访问被拒, 建议的模型更换` 


- **Llama-4-Scout 遇到障碍**：一位成员在本地使用 **Llama-4-Scout-17B-16E-Instruct** 模型时遇到了 *'model_not_supported'* 错误，尽管它在 **Colab** 上运行正常。
- **模型访问权限被拒绝**：一位成员报告称，申请 **Agents** 课程第 1 单元所需的 **Meta 模型** 访问权限被拒绝。
   - 他们询问是否可以重新申请或使用替代模型，并质疑该模型对课程的重要性。
- **Llama-3.2 作为替代方案**：一位成员发现将建议的模型更换为 **'meta-llama/Llama-3.2-3B-Instruct'** 解决了该问题。
   - 他们不知道为什么 **Llama-4** 模型失败了，但至少有了一个变通方案。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1453243088549056607)** (49 messages🔥): 

> `LM Studio Hugging Face Proxy, Speculative Decoding, NPU Support, Gemini 3 Pro` 


- **LM Studio HF Proxy 需要调整？**: 一位用户在使用 LM Studio 和云服务器时遇到问题，另一位用户建议在 General 设置中启用 **LM Studio's Hugging Face Proxy** 作为解决方案。
   - 然而，另一位用户则建议关闭 **HF proxy**。
- **投机采样 (Speculative Decoding) 已死？**: 成员们讨论了 **LM Studio** 对投机采样的支持，其中一人指出*只有 Qwen 2.5 和 Llama 3 运行良好*，但现在这些已是*老旧*模型了。
   - 另一人补充道：*这是一个锦上添花的功能，但实际上有点鸡肋*。
- **不支持 NPU！**: 一位成员想尝试在 **NPU** 上运行较小的模型，在 GPU 上运行较大的模型，但另一位成员表示 *NPU 目前不被支持，所以行不通*。
   - 会议明确了 **LM Studio** 目前不支持 **NPU**。
- **Gemini 3 Pro 的 UI 表现惊人！**: 一位成员对 **Gemini 3 Pro** 在 UI 任务中的表现表示赞赏，但也担心它可能会被“影子更新” (shadow-updated)，导致*在编程方面再次变得无法使用*。
   - 未提供相关链接或更多背景信息。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1453302753186021441)** (74 messages🔥🔥): 

> `Dual Channel RAM, Cost Effective LLM Workstation, 4000 Blackwell GPU, Tempered Glass Cases` 


- **笔记本电脑通过双通道内存提升性能**: 一位成员购买了 **$100 的 16GB SODIMM** 内存，以在笔记本电脑上实现**双通道内存**，解决了 iGPU 和 CPU 共享单条内存导致的性能问题。
   - 既定目标是提升性能，因为 *16GB 不足以让 iGPU 和 CPU 共享*。
- **高性价比编程 LLM 工作站**: 一位成员询问构建用于编程 LLM 的最高性价比工作站方案。
   - 另一位成员开玩笑地建议：*使用你现在的笔记本电脑并订阅 [github copilot](https://www.amazon.com/dp/B0FJRJJ9Q2)*。
- **巨型 GPU 需要巨型机箱**: 成员们讨论了如何将 **4000 Blackwell 24GB GPU** 与 **3090 Ti** 一起装入新机箱，评估了其尺寸以及与水冷装置的兼容性。
   - 他们链接到了 [亚马逊上的一个机箱](https://www.amazon.com/dp/B0FJRJJ9Q2)，并争论该机箱的设计是否允许安装多个 GPU，同时对插槽可用性和散热器尺寸表示担忧。
- **钢化玻璃机箱会升高系统温度**: 一位成员建议避免使用**钢化玻璃机箱**，因为它们往往会升高系统温度，尤其是在使用高性能组件时。
   - 另一位成员提出了替代方案，例如在网孔机箱上使用**磁吸滤网**来减轻灰尘问题，同时仍然偏好水冷装置的美观。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1453284868443209758)** (48 条消息🔥): 

> `X-Ware 用于推理基准测试、Character.ai 的 Squinch 优化、DeepWiki 在 OSS 中的实用性、Amazon 的 Rufus 聊天机器人、Nvidia 收购 Groq` 


- **X-Ware 在推理提供商基准测试中面临挑战**：分享了一篇博客文章，讨论了同一模型在不同推理提供商之间如何表现出不同的输出和用户体验，由于 **sampling parameters**、**prompting** 和 **deployment issues**，导致了 [基准测试挑战](https://xcancel.com/eliebakouch/status/2003604370534072445?s=46&t=eWVlK1PU8XfB6f402GJJ9g)。
- **Character.ai 揭秘 'Squinch' 优化**：来自 Character.ai 的一篇新技术博客文章 '[Squinch](https://xcancel.com/simon_mo_/status/2003608330003239278?s=46)' 详细介绍了其平台的 **performance optimization tricks** 和 **architectural improvements**。
- **DeepWiki 助力 OSS 理解**：成员们讨论了使用 **DeepWiki** 来查询和理解开源仓库（OSS repos），强调了它如何识别相关文件以及特定功能的实现细节。
   - 一位成员表示，当他们需要实现一些*已知在某些 OSS 仓库中设计、规范和实现得很好*的功能时，发现它非常有用。
- **Amazon 的 Rufus 在质疑声中亮相**：成员们注意到 Amazon 上出现了一个名为 **Rufus** 的自动弹出聊天功能，一位成员透露该功能已经开发了一年多（[链接](https://www.aboutamazon.com/news/retail/amazon-rufus)）。
   - 一位成员指出，*如果它能稍微增加销量*，Amazon 就会全力投入，尽管另一位成员担心*它也可能导致销量下降*。
- **Nvidia 瞄准 Groq，Chamath 表示认可**：分享了一个关于 [Nvidia 可能以 200 亿美元收购 Groq](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html) 的链接，这一价格是其目标营收的 **40 倍**。
   - 一位成员观察到，*Nvidia 面临的严峻竞争来自 Google 的 TPU、Amazon 的 Trainium、AMD 的 Instinct，以及在较小程度上来自 Intel 的 ARC*。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1453509066847424543)** (4 条消息): 

> `FlashSR, Audio Enhancement Model, MiraTTS, Hugging Face, GitHub` 


- **FlashSR 音频增强模型发布**：Yatharth Sharma 宣布正式发布 **FlashSR**，这是一款高速音频质量增强模型，能够通过 [此 X 帖子](https://xcancel.com/Yatharth3501/status/2003884180577702074) 实现超过 **200 倍实时速度** 的处理。
   - 该模型和仓库现已在 **Hugging Face** 和 **GitHub** 上公开，此前已集成到 **MiraTTS** 中。
- **FlashSR 与 MiraTTS 集成**：**FlashSR** 音频增强模型此前已集成到 **MiraTTS** 中，这证明了它的实用性和效率。
   - 现在该模型已公开，将允许 AI 社区进行更广泛的采用和进一步开发。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1453335421177168004)** (36 messages🔥): 

> `Partial RoPE Ablations, Long Context Scaling with Qwen3-Next, Attention Normalization, RoPE for Interp, RMSNorm after Attention` 


- **Partial RoPE 因消融实验胜出**：讨论者们辩论了使用 **partial RoPE** 的原因，指出虽然最初并非为了降低 loss，但历史上的消融实验显示了性能提升，从而促成了其采用，并引用了 [arxiv.org/abs/2512.19941](https://arxiv.org/abs/2512.19941) 论文。
   - 像 **Qwen3-Next** 这样的新模型正在实施 **partial RoPE** 以提高效率和长上下文泛化能力，**MLA** 发现它特别重要，尽管它可能比其他 **RoPE** 设置稍差。
- **新型 Normalization 探讨**：一位成员分享了一篇讨论 **attention** 归一化的 [博客文章](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/)，引发了进一步讨论。
   - 据一位成员称，对于语言任务，它比普通的 **softmax** 效果更差，即使正确处理了细微细节也是如此，因为他们已经训练了 **5000 个模型** 才得出这个结论。
- **在 RoPE 模型上进行 Interp 的问题**：一位成员表达了对用于 **interp**（可解释性）的 **RoPE** 的厌恶，并希望它被取代，引发了关于在 **RoPE** 模型上进行 **interp** 挑战的讨论，并分享了 [EleutherAI 博客文章](https://blog.eleuther.ai/rotary-embeddings/)。
   - 另一位成员分享了说明 **RoPE** 和 **PoPE** 之间差异的图表。
- **Attention 后的 RMSNorm**：一位成员询问是否有人尝试过在 **attention** 之后放置 **RMSNorm**，以及是否有负面影响。
   - 另一位成员回复说，在 **SDPA** 之后添加 **norm** 有帮助，并引用了 **Qwen gating paper** 作为证据，尽管他们将其归因于 **norm** 的非线性。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1453350706080907265)** (1 messages): 

> `SAE, open-source repositories, fine-tuning the trained SAE` 


- **寻求支持微调的开源 SAE 仓库**：一位成员询问了用于实现 **SAE** 特性的主流**开源仓库**，特别是寻求能够对训练好的 **SAE** 进行**微调**的功能。
- **关于 SAE 微调的澄清**：用户正在寻找允许对 **Sparse Autoencoders (SAEs)** 进行微调的仓库。
   - 他们有兴趣调整训练好的 **SAE** 模型，这是 **SAE** 的一个特定应用。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1453290417532440657)** (30 messages🔥): 

> `Gemini Hallucinations, Benchmark Accuracy, M2.1 vs GLM-4.7, Qwen and RAG, Dynamic Semantic Search` 


- **Kimi K2 颠覆 Benchmark？**：一些成员觉得尽管有 **benchmarks**，**Kimi (K2 Thinking)** 在实际体验中仍优于其他模型，特别是在 *browsecomp* 和非奉承性（non-sycophancy）等领域。
   - 一位成员认为 Google 为了公众降低了其模型的智商，但在 **Kimi** 身上还没看到这种情况，希望它能保持下去。
- **Deepseek 表现出色但 Benchmark 遭到质疑**：虽然 **Deepseek** 表现出色，但一些成员表示 **benchmarks** 似乎越来越可疑，并提到个人经验中 **Gemini** 的幻觉最多。
   - 另一位成员补充说，在另一个 **benchmark** 中，**Gemini** 的准确率得分也是最高的。
- **M2.1 优于 GLM-4.7**：**M2.1** 被称赞为出奇地好，在普通任务上可能比 **GLM-4.7** 更好，而 **GLM-4.7** 仍表现出 **4.6** 的缺陷，如随机的中文回答或陷入循环。
   - 一位成员说：“我真的以为这全是炒作。我还没在任何有意义的难题上测试它，但对于普通任务，我非常喜欢它。”
- **Qwen 是否使用 RAG 读取文件？**：一位成员询问 **Qwen** 是否使用 **RAG** 来读取文件，以及 **RAG** 是否能解决大文件占用窗口的上下文问题。
   - 另一位成员在深入研究时分享了 [百度文心一言任务阅读页面](https://ernie.baidu.com/task/reading) 的链接。
- **Agentic 动态语义搜索优于 RAG？**：据一位成员称，虽然 **RAG** 在某些情况下很有用，但带有 **agentic** 框架的动态语义搜索总是优于 **RAG** 中静态的一次性语义搜索。
   - 另一位成员在那个问题上浪费了研究额度。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453266139516895344)** (6 messages): 

> `Quantization, Lecture 7 slides` 


- **CUDA 和 Triton 中的 Quantization 技巧**：一位成员分享了[幻灯片链接](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0)，讨论了 **CUDA** 与 **Triton** 中的 **quantization**。
   - 幻灯片提供了在两种环境下优化性能的 **quantization** 技术的对比。
- **Lecture 7 幻灯片不可用**：一位成员指出 **Lecture 7** 的幻灯片链接失效了。
   - 另一位成员提议联系 Charlie，暗示可能通过其他途径获取幻灯片，尽管距离 **Lecture 7** 已经有一段时间了。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1453323242705125459)** (1 messages): 

> `TMA Transpose, Swizzle Optimization` 


- **TMA Transpose 实现，Swizzle 优化**：成员们讨论了虽然可以使用 **TMA (Tensor Memory Accelerator)** 进行转置，但根据 [effective_transpose](https://github.com/simveit/effective_transpose)，需要 **swizzle** 才能获得最佳性能。
- **Swizzle 增强 TMA Transpose 效率**：为了获得最优的 **TMA transpose** 性能，结合 **swizzle** 技术至关重要，正如 [effective_transpose](https://github.com/simveit/effective_transpose) 仓库中所强调的。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1453358415437893715)** (2 messages): 

> `Lecture Slides, Lecture 7, Quantization` 


- **课程幻灯片链接故障排除**：一位成员反馈课程幻灯片链接无法使用。
   - 另一位成员回复说[链接对他们有效](https://lecture.slides)，并澄清他们指的是关于 **Quantization** 的 **Lecture 7**。
- **Quantization 困惑已澄清**：一位用户报告了特定课程幻灯片链接的问题。
   - 另一位用户确认链接功能正常，特别是针对涵盖 **Quantization** 概念的 **Lecture 7**。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1453355717598777345)** (2 messages): 

> `AMDGPU custom builtins, LLVM dev resources` 


- **寻求 AMDGPU 自定义 builtins 资源**：一位成员请求编写 **AMDGPU 后端** **自定义 builtins** 的资源。
   - 他们表示有兴趣获得关于 **LLVM dev** 的一般性建议，以扩展他们在编译器栈方面的专业知识。
- **AMDGPU 后端开发资源**：用户正在寻求涵盖专门为 AMDGPU 后端开发自定义 builtins 的资源推荐。
   - 他们还在寻找关于 LLVM 开发实践的一般指导，以增强其编译器专业知识。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

2kian: [Life-Timeline Forecaster](https://x.com/neuralkian/status/2003946169802834191)
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1453340669543321797)** (8 messages🔥): 

> `nvfp4_dual_gemm NVIDIA leaderboard submissions, NVIDIA performance improvements` 


- **NVIDIA 个人最佳成绩在 nvfp4_dual_gemm 上提升**：一位用户在 NVIDIA 的 `nvfp4_dual_gemm` 排行榜上取得了 **65.5 µs** 的个人最佳成绩。
   - 随后他们又提交了 **60.0 µs** 的新个人最佳成绩。
- **另一位用户也在 NVIDIA 上刷新个人最佳**：一位用户在 NVIDIA 的 `nvfp4_dual_gemm` 排行榜上取得了 **77.6 µs** 的个人最佳成绩。
   - 随后他们将其优化至 **41.6 µs** 的新个人最佳。
- **NVIDIA 上达成更多个人最佳成绩**：一位用户在 NVIDIA 的 `nvfp4_dual_gemm` 排行榜上取得了 **45.2 µs** 的个人最佳成绩。
   - 他们进一步将其优化至 **26.9 µs**，在 NVIDIA 排行榜上排名 **第 8**。
- **夺得 NVIDIA 排行榜第 7 名**：同一位用户凭借 `nvfp4_dual_gemm` 上的 **18.1 µs** 夺得 NVIDIA 排行榜 **第 7 名**。
   - 另一位用户成功提交了 **15.6 µs** 的成绩。


  

---

### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1453374424366583939)** (2 messages): 

> `Tinygrad Eager Mode, Handwritten CUDA Kernels, TF1 vs TF2, IR.pyodide, Rust mdbook` 


- **Teenygrad 进入 Eager 模式：TF1 vs TF2？**：在 *teenygrad* 的抽象层中添加 **Eager Mode** 和**手写 CUDA Kernel**，被认为与 **TF1 -> TF2** 的转型具有相同的本质。
   - 据观察，*teenygrad* 正在从 **Rust mdbook** 内部的 **IR.pyodide** 进行逆向开发。
- **Teenygrad 的开发策略**：*teenygrad* 正在通过对托管在 **Rust mdbook** 环境中的 **IR.pyodide** 进行逆向推导来进行开发。
   - 这种方法与传统方法形成对比，侧重于其抽象层的逆向工程策略。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1453477761154027725)** (2 messages): 

> `cute swizzling, tcgen PTX, open source for skilling up` 


- **Cute 的 Swizzling 减少 Bank Conflict**：**cute** 中最重要的特性是其 **swizzling** 能力，这有助于减少 Bank Conflict，并能更友好地访问 **tcgen PTX**。
   - 一些成员认为，这些是 cute 的主要特征支柱，特别是对于优化内存访问模式而言。
- **开源提升技能获得认可**：频道内的普遍共识是，利用**开源**资源是提升个人技能的最佳途径。
   - 成员们暗示，亲手实践和社区协作是核心优势，并推崇这种“*gigachad*”式的学习方法。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1453358294327623773)** (6 messages): 

> `autogen.py, contiguous error, symbolic tensor` 


- **Autogen 文件触发重新生成问题**：一位成员询问在修改 `autogen.py` 后，是否应该重新生成 `autogen/*.py` 文件并包含在 PR 中。
- **一维零张量导致 Contiguous Tensor 错误**：一位成员报告称 `char = Tensor.zeros(120); char[5] = 1` 会提示 `setitem target needs to be contiguous` 错误，并询问为什么一维零张量返回的是非连续张量。
   - 另一位成员澄清说 `Tensor.zeros(120)` 创建的是一个**符号张量 (Symbolic Tensor)**，它*在内存中并不是真实存在的实体*，并建议使用 `.contiguous()` 来解决此问题。
- **符号张量澄清**：一位成员澄清说，**Symbolic Tensor** 不是内存中的真实对象，但*可以作为 Kernel 中的虚拟对象使用*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1453244732539928676)** (6 messages): 

> `Manus Pro credits, Mobile app preview issue, New channel for collaboration/services` 


- **Manus Pro 积分困惑**：一位 **Manus Pro** 用户报告称其月度积分已用完，且未收到预期的 **300 每日积分**。
   - 他们表示已经检查了几天，都没有收到任何每日积分。
- **移动端应用预览困境**：一位用户紧急请求协助解决移动端应用预览问题，称尽管尝试了所有方法，预览仍无法显示。
   - 未提供关于移动端应用性质或已尝试的故障排除步骤的进一步细节。
- **创建社区协作频道**：为 Manus 以外的话题以及社区成员提供协作/服务创建了一个新频道。
   - 这样做是因为在 general 频道中发布的此类提议经常被忽略。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1453335416530014339)** (5 messages): 

> `Agentic context engineering, LLM autodiff/textgrad, Prompt optimization, New member introduction` 


- **TextGrad 接近 DSPy？**：一位成员询问是否有计划将 [Agentic context engineering](https://arxiv.org/pdf/2510.04618) 或 [LLM autodiff/textgrad](https://arxiv.org/abs/2501.16673) 引入 DSPy。
   - 另一位成员链接了 [textgrad](https://github.com/zou-group/textgrad) 并询问它是否是 DSPy 的杀手锏，但另一位成员指出它*似乎没有在积极维护*。
- **提示词优化努力涌现**：成员们讨论了提示词优化（Prompt Optimization）技术。
   - 有人指出，*现在有这么多吸引人的方法，每种方法至少有 10 个细微变体，比如 textgrad 版本*。
- **资深工程师加入频道**：一位**资深全栈/区块链工程师**介绍了自己，热衷于将复杂的想法转化为稳定、可维护的软件。
   - 他们列出了自己的**技术栈 (Tech Stack)**：前端：React, Next.js, Vue, Nuxt；后端：Node.js, Nest.js, Laravel, FastAPI；区块链：Solidity, 智能合约工具, DApp 集成；基础设施：Docker, AWS, 流水线, 监控；数据：PostgreSQL, MongoDB, Redis。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1453266992277622846)** (2 messages): 

> `Discord 频道建议，Kapa AI 用户体验` 


- **推荐的 Discord 频道**：一名成员建议查看 [<#1436158039232086186>](https://discord.com/channels/1436158039232086186) 或 [<#1212827673257316453>](https://discord.com/channels/1212827673257316453) 频道。
- **用户分享 Kapa AI 的早期使用体验**：一位成员分享说，当他们刚开始使用 Discord 服务器时，花了大约 **3 周** 时间尝试使用 **Kapa AI** 但都失败了。
   - 他们开始变得相当焦虑，心想 *“为什么你不理我？”*。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1453284033306820620)** (2 messages): 

> `数据库-PL 优化，LingoDB，查询优化器` 


- **探讨数据库-PL 优化**：一名成员提到了硬件加速之外的各种想法，例如 [结合数据库-PL 的优化](https://discord.com/channels/1087530497313357884/1104620458168553563/1367474833197236344)，如 **LingoDB**。
   - 另一位成员对这种方法表示担忧，指出 *对人类更友好的语言需要查询优化器（query optimizer）做更多的工作*。
- **强调查询优化器的挑战**：讨论强调了为注重人类可读性而非机器效率的语言优化查询所面临的挑战。
   - 这表明在数据库-PL 集成中，开发者友好语法与性能之间可能存在权衡。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1453504098526691428)** (3 messages): 

> `圣诞祝福` 


- **圣诞祝福**：成员们互相交流了圣诞祝福，并表达了对节日的美好祝愿。
- **温馨的节日祝愿**：包括 Yannic Kilcher 在内的多位成员祝大家圣诞快乐，在频道中营造了积极且充满节日气氛的环境。


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1453488772837671157)** (1 messages): 

> `Windsurf Wave 13，SWE-1.5 免费，Git Worktree 支持，多 Cascade 面板与标签页，专用终端` 


- **Windsurf 发布 Wave 13：Shipmas 特别版！🎅🚢**：Windsurf 发布了 **Wave 13**，带来了并行、多 Agent 工作流、专用终端和上下文窗口指示器等功能。
   - 他们将在未来 3 个月内免费提供其近前沿（near-frontier）编程模型的访问权限，团队祝大家 *Shipmas 快乐！* 🚢
- **SWE-1.5 节日期间免费！**：**SWE-1.5** 拥有接近前沿的 SWE-Bench-Pro 性能，现在所有用户在未来 3 个月内均可免费使用（常规吞吐速度）。
- **Git Worktree 支持上线**：新的 **Git Worktree 支持** 允许你在同一个仓库中生成多个 Cascade 会话，而不会产生合并冲突。
   - 这支持了独立的分支、目录和共享的 Git 历史记录。
- **多 Cascade 面板与标签页提升生产力**：用户现在可以使用新的 **多 Cascade 面板与标签页** 功能，在同一个窗口中并排查看多个 Cascade 会话并与之交互。
- **Windsurf 获得专用终端 (Beta)**：**Cascade** 现在可以在为可靠性而配置的专用 zsh shell 中运行命令，使用你的 .zshrc 环境变量并能更好地处理复杂的提示符（macOS 上需主动开启）。


  

---


---


---


---