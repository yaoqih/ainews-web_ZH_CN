---
companies:
- anthropic
- deepseek
- moonshot-ai
- minimax
- openai
- ollama
date: '2026-02-24T05:44:39.731046Z'
description: '**Anthropic** 指控 **DeepSeek**、**月之暗面 (Moonshot AI)** 和 **MiniMax** 对其
  **Claude** 模型进行了“工业级”规模的蒸馏攻击。据称，这些攻击涉及约 **2.4 万个虚假账号**和超过 **1600 万次 Claude 对话**，旨在提取模型能力，引发了人们对竞争风险和安全性的担忧。


  社区正就“数据抓取 (scraping)”与“API 输出提取”之间的区别展开辩论，凸显了保护模型手段向“抗 API 滥用”技术转化的趋势。与此同时，**Codex**
  和 **Claude Code** 等编程代理在实际应用中经历了落地与失败，而 **Simon Willison** 正引领着“代理工程 (agentic engineering)”领域新兴最佳实践的发展。此外，**OpenClaw**
  生态系统持续扩张，推出了 **NanoClaw** 等替代方案，而 **Ollama 0.17** 等集成工具的出现也进一步简化了开源模型的使用。'
id: MjAyNi0w
models:
- claude
- claude-3
- codex
- claude-code
people:
- simon_willison
title: Anthropic 指控 DeepSeek、月之暗面（Moonshot AI）和 MiniMax 进行了“工业规模的蒸馏攻击”。
topics:
- api-abuse-resistance
- model-security
- agentic-engineering
- coding-agents
- model-distillation
- workflow-automation
- sandboxing
- realtime-communication
---

**出口管制显著升级。**

> 2026/2/20-2026/2/23 的 AI 新闻。我们为您检查了 12 个 Reddit 子版块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**262** 个频道，**28837** 条消息）。预计节省阅读时间（以 200wpm 计算）：**3003** 分钟。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件发送频率！

---

# AI Twitter 综述

**Anthropic 关于 Claude “蒸馏攻击”的指控（以及行业反弹）**

- **Anthropic 的指控**：Anthropic 称其检测到 **DeepSeek**、**Moonshot AI** 和 **MiniMax** 进行了*工业级规模*的 Claude 蒸馏：**约 24,000 个欺诈账号**生成了 **>16M 次 Claude 交互**，据称是为了为其自身模型提取能力 ([Anthropic](https://x.com/AnthropicAI/status/2025997928242811253), [后续](https://x.com/AnthropicAI/status/2025997929840857390), [博客链接推文](https://x.com/AnthropicAI/status/2025997931589881921))。Anthropic 将这种风险定性为竞争性（能力转移）和安全/地缘政治（防护措施移除、下游军事/情报使用）。
- **社区反应 / “虚伪”讨论串**：很大一部分回复将其描述为“基于互联网训练的实验室现在开始抱怨被抄袭”，通常明确对比了爬取（scraping）与 API 输出提取 ([Elon](https://x.com/elonmusk/status/2026012296607154494), [ThePrimeagen](https://x.com/ThePrimeagen/status/2026016322232983733), [Teknium](https://x.com/Teknium/status/2026001761904021858), [Suhail](https://x.com/Suhail/status/2026009921255592294), [HKydlicek](https://x.com/HKydlicek/status/2026006007990690098))。其他人则认为，这种规模的蒸馏具有显著差异，因为它可以复制 *工具使用 / Agent 行为*，并可能绕过安全控制 ([RundownAI 总结](https://x.com/TheRundownAI/status/2026019722211279356), [LiorOnAI 观点](https://x.com/LiorOnAI/status/2026043272565772386))。
- **二阶影响**：该讨论串明确了安全模型的转变：前沿模型（frontier models）正越来越多地受到保护，不仅通过权重保密和算力稀缺，还通过 *API 滥用防御*（账号欺诈检测、速率限制规避、行为指纹识别、水印等）。它还重新提出了一个问题：如果能力可以通过大规模输出被“复制”，那么 **Export controls**（出口管制）是否还能发挥作用 ([LiorOnAI](https://x.com/LiorOnAI/status/2026043272565772386))。
- **相关市场/时机背景**：一些人将公告发布时机与即将到来的 **DeepSeek V4** 新闻周期 ([kimmonismus](https://x.com/kimmonismus/status/2026040919162822776)) 以及更广泛的美中博弈框架联系起来。

**编程 Agent：真实采用、真实失败以及“Agentic Engineering”剧本**

- **Codex + Claude Code 的势头（以及掩盖真实工作流变革的模因）**：许多参与度最高的帖子都是“Agent 时代已至”的轶事——例如使用 Codex 在周末进行构建 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2025712197100589353), [gdb](https://x.com/gdb/status/2025723937540485506))——以及关于赋予 Agent 过多权限的警示故事。这类案例中的典型失败模式是指令丢失/压缩，导致 OpenClaw 风格的配置中出现非预期的破坏性行为（如删除邮件）([summeryue0](https://x.com/summeryue0/status/2025774069124399363), [后续根因分析](https://x.com/summeryue0/status/2025836517831405980)，以及其他针对“写入权限”风险的反应：[Yuchenj_UW](https://x.com/Yuchenj_UW/status/2025994509721731092))。
- **Agent 工程指导正在形成共识**：
  - **Simon Willison** 发布了 **“Agentic Engineering Patterns”**（Agent 工程模式）指南的前几章，旨在指导 Claude Code/Codex 等编程 Agent 的使用 ([simonw](https://x.com/simonw/status/2025990408514523517))。
  - 一个微型争议：“删除你的 CLAUDE.md/AGENTS.md”文件（即过度自定义可能是“货物崇拜/盲目效仿”）([theo](https://x.com/theo/status/2025900730847232409)，[bpodgursky](https://x.com/bpodgursky/status/2025966899402625485) 表示赞同，以及诸如 [ryancarson](https://x.com/ryancarson/status/2025993265732854132) 的“彻底修剪”式回应）。
- **OpenClaw 生态扩展及替代方案**：
  - **NanoClaw** 定位为一个更小、容器隔离的类 OpenClaw 助手，具备 WhatsApp I/O、Swarms、定时任务等功能 ([TheTuringPost](https://x.com/TheTuringPost/status/2025876086035464512), 代码库: [qwibitai/nanoclaw](https://x.com/TheTuringPost/status/2025876098131902666))。
  - 多个“如何构建 OpenClaw 风格 Agent”的技术栈强调了那些枯燥但至关重要的环节：调度器/队列、沙箱、实时通信 ([TheTuringPost 技术栈列表](https://x.com/TheTuringPost/status/2025903129800384801))。
  - **Ollama 0.17** 使在 OpenClaw 中使用开源模型变得更简单（并释放了出于安全考虑对本地 Agent 执行持续关注的信号）([ollama](https://x.com/ollama/status/2026098586300071975))。
- **企业级/生产级 Agent 工程正转向可观测性与评估循环 (Eval Loops)**：Exa 的“深度研究 Agent”案例研究强调将 Token/缓存可观测性作为计价基础设施 (LangSmith/LangGraph) ([LangChain](https://x.com/LangChain/status/2025744946494345570))。monday.com 的服务 Agent 将评估视为“Day 0”任务，并声称通过使用 LangSmith 实现了 **8.7 倍的反馈循环加速** ([hwchase17](https://x.com/hwchase17/status/2026095629148258440))。

**基准测试与评估完整性：SWE-Bench Verified 弃用、新排行榜以及 Agent 代码库生成的瓶颈**

- **SWE-Bench Verified 正被 OpenAI DevRel 主动弃用**：OpenAI 推荐使用 **SWE-bench Pro**，并表示 Verified 版本已趋于饱和/失效：**数据污染**和**测试设计缺陷**意味着它不再能衡量前沿编程能力 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2026002219909427270), 分析讨论: [latentspacepod](https://x.com/latentspacepod/status/2026027529039990985), 综述: [swyx](https://x.com/swyx/status/2026029120040137066), 独立总结: [rasbt](https://x.com/rasbt/status/2026062254571913522), tl;dr: [polynoamial](https://x.com/polynoamial/status/2026032321212891550))。推文中呼应的分析关键细节显示：在审计了一部分频繁失败的任务后，发现很大比例的任务存在测试缺陷（拒绝正确方案）或在“现有说明”下根本无法解决。
- **推行“单位美元能力”评估**：AlgoTune 明确设定了 **1 美元/任务** 的预算，生成的排名更看重性价比高的模型，将“最佳”重新定义为 *成本约束下的最佳* ([OfirPress](https://x.com/OfirPress/status/2026068384589172800))。
- **长时程编程 Agent 依然面临失败**：**NL2Repo-Bench** 测试 Agent 是否能从零生成一个完整的、可安装的 Python 库；报告显示顶尖模型的通过率 *低于 40%*，失败模式主要集中在规划和代码库整体的连贯性上 ([jiqizhixin](https://x.com/jiqizhixin/status/2025823941642621241))。
- **OCR 评估现状检查**：据报道，即使是强大的 OCR 模型在处理密集的历史报纸时也会“崩溃”（幻觉/循环），这突显了模型在精选文档分布之外的脆弱性 ([vanstriendaniel](https://x.com/vanstriendaniel/status/2025930991387164919))。此外：**OlmOCR-Bench** 成为 Hugging Face (HF) 上用于社区评估提交的基准数据集 ([mervenoyann](https://x.com/mervenoyann/status/2025908932691017983))。

**推理与系统：面向 Agent 的 WebSockets、极速片上推理以及基础设施扩展叙事**

- **OpenAI Responses API 为低延迟、长耗时、工具密集型 Agent 增加了 WebSockets 支持**。原理：持久连接 + 内存状态意味着你只需发送增量输入而非完整上下文；声称在 20 个以上的工具调用中可实现 **20–40% 的提速**（[OpenAIDevs](https://x.com/OpenAIDevs/status/2026025368650690932)，详情：[OpenAIDevs](https://x.com/OpenAIDevs/status/2026025380562530453)，采用情况：[OpenAIDevs](https://x.com/OpenAIDevs/status/2026059511241535628)）。Cline 报告了早期测试数据：简单任务提速约 15%，复杂工作流提速约 39%，最佳情况下提速达 50%（[cline](https://x.com/cline/status/2026031848791630033)）。Steven Heidel 将 Codex 的提速归功于 WebSockets（[stevenheidel](https://x.com/stevenheidel/status/2026028343859286140)）。
- **推理工程（Inference engineering）成为“一门独立的学科”**：Baseten 发布了 **《Inference Engineering》** 一书（[philipkiely](https://x.com/philipkiely/status/2025994823891914795)），工程师们强调推理是延迟、成本和可靠性方面的竞争层（[hasantoxr](https://x.com/hasantoxr/status/2025996746133049498), [JayminSOfficial](https://x.com/JayminSOfficial/status/2025996744509804865)）。
- **硬件/架构信号**：
  - 一项 Demo 声称通过将“模型参数蚀刻进晶体管”（计算与存储融合），在 **Llama 3.1 8B 上实现了 18,000 tokens/sec** 的速度（[philschmid](https://x.com/_philschmid/status/2025830254753853843)）。
  - NVIDIA 发布了针对 **Blackwell 优化的 Qwen3.5 MoE**，量化至 **NVFP4**，使用 SGLang 可实现 **2 倍的推理提速**（[HuggingPapers](https://x.com/HuggingPapers/status/2025825405836648849)）。
  - fal 分享了其推理引擎中的通信/计算重叠优化（“Async Ulysses”）（[isidentical](https://x.com/isidentical/status/2026000340873777419)）。
- **算力策略叙事发生冲突**：关于 OpenAI “Stargate” 数据中心项目停滞的说法在讨论帖中遭到反驳，另一种说法是：Stargate 是一个涵盖多方合作伙伴算力生态系统的伞形品牌（包括 SoftBank/NVIDIA/AMD/Broadcom/Oracle/Microsoft/AWS/CoreWeave/Cerebras），且到 2025 年底将拥有约 **2GW 的可用算力**（[kimmonismus 的说法](https://x.com/kimmonismus/status/2025851041242087901) 对比 [sk7037 的回应](https://x.com/sk7037/status/2026067771394838629)）。

**模型/榜单更新与研究线程（推理、记忆、多模态视频）**

- **Arena 排行榜**：GPT-5.2-chat-latest 以 **1478** 分进入文本 Arena 前 5 名，比 GPT-5.2 高出 40 分；在多轮对话、指令遵循、硬提示（hard prompts）和编程方面有显著改进 ([arena](https://x.com/arena/status/2025966052950315340)，细分数据：[arena](https://x.com/arena/status/2025986008484061391))。
- **Gemini 3.1 Pro**：WeirdML 评分为 **72.1%**，而 3.0 为 69.9%；被指出具有“高巅峰与怪异弱点并存”的特点，且输出 Token 使用量显著更高 ([htihle](https://x.com/htihle/status/2025867003550958018))。另外，开发者关于容量和 Tool-calling 可靠性的抱怨引起了高度关注 ([theo](https://x.com/theo/status/2025896487557947886)，[theo 后续](https://x.com/theo/status/2025900101122867368)，以及之后的：[theo](https://x.com/theo/status/2026045501960069204))。
- **Qwen3.5 模型发布传闻**：一条推文声称 Qwen 发布了一个 **397B 多模态 MoE 模型，激活参数为 17B**，并称其“足以与 GPT-5.2/Claude 4.5 竞争” ([HuggingPapers](https://x.com/HuggingPapers/status/2025805747385221491))。在查阅模型权重卡（model card）或评测结果之前，请谨慎对待该基准测试对比。
- **推理训练 / CoT**：
  - Teknium 认为验证器模型（verifier models）并不存在“免费午餐”：更好的求解器往往也是更好的验证器；针对难题使用更小、更“笨”的评判模型通常会失败 ([Teknium](https://x.com/Teknium/status/2025740765230682400))。
  - 字节跳动风格的 CoT 工程被描述为从长度惩罚转向强制执行压缩的流水线；此外还有一种关于 Long-CoT 结构的“分子”框架，包含“语义异构体（semantic isomers）”和一种合成数据方法 (**Mole-Syn**) ([teortaxesTex](https://x.com/teortaxesTex/status/2025817199764500789)，摘要见 [TheTuringPost](https://x.com/TheTuringPost/status/2026050264122462370))。
  - DAIR 重点介绍了一篇关于通过信息论实现 **CoT 可监测性**的论文（互信息是必要条件而非充分条件；差距源于监测提取和启发误差），并提出了旨在提高透明度的训练方法 ([dair_ai](https://x.com/dair_ai/status/2026043400861122709))。
- **视频 / 世界模拟**：多篇关于交互式视频生成和多镜头生成的论文正在流传 ([akhaliq 交互式视频](https://x.com/_akhaliq/status/2025944948453847352)，[akhaliq 多镜头](https://x.com/_akhaliq/status/2025951076579475640)，[QingheX42 代码发布](https://x.com/QingheX42/status/2025953650334679410))；产品方面：**可灵 (Kling) 3.0** 集成到 Runway 工作流中 ([runwayml](https://x.com/runwayml/status/2025977383208051018))，以及 **Veo 3.1 模板** 开始在 Gemini 应用中推出 ([GeminiApp](https://x.com/GeminiApp/status/2026001595708866759)，[Google](https://x.com/Google/status/2026006156875804960))。

**围绕 AI Agent 的工作、采用和“宏观”论述（Citrini 文章 + Anthropic 熟练度 + OpenAI 企业联盟）**

- **Citrini 的“未来宏观备忘录”文章成为论述焦点**：多条推文将其总结为一种情景，即日益廉价的 Agent 压缩了白领工资/消费，创造了“幽灵 GDP”，并给金融市场和政治带来压力 ([kimmonismus 总结](https://x.com/kimmonismus/status/2025914288439771171)，[stevehou 反应](https://x.com/stevehou/status/2025797519028936854)，作者后续：[Citrini7](https://x.com/Citrini7/status/2025980800659792270))。讨论串指出，反应集中在赞同、细微的反驳以及带有表演性质的嘲讽 ([teortaxesTex](https://x.com/teortaxesTex/status/2025894184817684633))。
- **Anthropic 的“AI 熟练度指数” (AI Fluency Index)**：Anthropic 测量了 Claude 对话中的协作行为；报告的一个关键关联是，熟练度与“迭代/优化”正相关，而非“一键式提示（one-shot prompting）” ([AnthropicAI](https://x.com/AnthropicAI/status/2025950279099961854))。
- **OpenAI 通过咨询联盟扩大企业市场化进程**：OpenAI 宣布与 BCG、McKinsey、Accenture、Capgemini 建立 **Frontier Alliances**，旨在通过集成和变革管理部署“AI 同事（AI coworkers）”，力求突破试点阶段 ([bradlightcap](https://x.com/bradlightcap/status/2025936690334875735)，分析：[kimmonismus](https://x.com/kimmonismus/status/2025942986765279506))。
- **采用情况仍然不均衡**：有一项统计数据称 **84% 的人从未用过 AI**（被解读为“我们还处于早期阶段”） ([kimmonismus](https://x.com/kimmonismus/status/2025934901116080636))。而工程师们则同时报告在自己的工作流中“Agent 无处不在”——这凸显了技术扩散具有高度的集群性。

---

### 热门推文（按互动量排序，技术相关）
- **Anthropic 指控 DeepSeek/Moonshot/MiniMax 对 Claude 进行大规模蒸馏 (Distillation)** ([AnthropicAI](https://x.com/AnthropicAI/status/2025997928242811253))
- **“行动前确认” Agent 删除了收件箱：OpenClaw 的警示案例** ([summeryue0](https://x.com/summeryue0/status/2025774069124399363))
- **WebSockets 已添加到 OpenAI Responses API，以加速重度依赖工具的 Agent** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2026025368650690932))
- **OpenAI 弃用 SWE-Bench Verified 作为前沿编程指标；推荐使用 SWE-bench Pro** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2026002219909427270))
- **Anthropic “AI Fluency Index” 研究（将迭代/优化视为核心行为）** ([AnthropicAI](https://x.com/AnthropicAI/status/2025950279099961854))
- **Simon Willison 为编程 Agent 编写的 “Agentic Engineering Patterns” 指南** ([simonw](https://x.com/simonw/status/2025990408514523517))
- **Cline 对 Responses API WebSockets 进行基准测试：在复杂工作流中提速高达 ~39%** ([cline](https://x.com/cline/status/2026031848791630033))

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Anthropic 蒸馏攻击

  - **[Anthropic：“我们发现 DeepSeek、Moonshot AI 和 MiniMax 对我们的模型进行了工业规模的蒸馏攻击。” 🚨](https://www.reddit.com/r/LocalLLaMA/comments/1rcpmwn/anthropic_weve_identified_industrialscale/)** (活跃度: 4207): **Anthropic** 发现 **DeepSeek、Moonshot AI 和 MiniMax** 对其模型进行了工业规模的蒸馏攻击。这些攻击涉及创建超过 `24,000` 个虚假账号，并与 Anthropic 的模型 **Claude** 进行了超过 `16 million` 次对话，以提取其能力用于改进自家模型。这突显了 AI 行业中重大的安全和知识产权挑战，即模型能力可能会被非法提取和复制。评论者将这些蒸馏攻击与更广泛的 AI 行业在未经明确授权的情况下使用数据的做法进行了类比，暗示 Anthropic 的投诉存在双重标准。还有人对 Anthropic 自身数据集的构建方式表示怀疑，暗示可能存在伦理问题。

    - 讨论指出，Anthropic 对蒸馏攻击的投诉存在潜在的讽刺意味，因为他们自己的模型训练可能也涉及在未获得明确许可的情况下使用大型数据集。这引发了关于 AI 开发中数据使用伦理影响的质疑，尤其是像 Anthropic 这样的公司在其并不拥有或无权使用的数据上构建了模型。
    - 提到的 DeepSeek、Moonshot AI 和 MiniMax 等公司的工业规模蒸馏攻击表明，AI 模型正处于被逆向工程或复制的竞争环境中。这可能涉及利用 API 访问来提取模型输出并训练类似模型，这给 AI 的知识产权保护带来了重大挑战。
    - 有观点认为 Anthropic 的数据集可能是由人工手动标注的，这意味着在数据质量和策划方面投入了巨大成本。这与蒸馏攻击的概念形成对比，后者通过利用现有模型的输出来训练自己的系统，从而绕过这些努力。

  - **[虚伪？](https://www.reddit.com/r/LocalLLaMA/comments/1rcrb2k/hypocrisy/)** (活跃度: 380): **该图片强调了 **AnthropicAI** 的一项指控，即 **DeepSeek**、**Moonshot AI** 和 **MiniMax** 对其模型进行了“大规模蒸馏攻击”。这些攻击涉及创建 `24,000` 个虚假账号并与 **Claude** 进行了 `16 million` 次交互以提取其能力，推测是为了改进他们自己的 AI 模型。这引发了对此类行为的伦理和合法性，以及保护 AI 模型免受未经授权数据提取的安全措施的担忧。一位评论者质疑被告实验室的伦理立场，认为他们可能没有为自己的行为寻求许可；而另一位评论者则对 **z.ai** 未被提及感到惊讶，暗示类似做法可能更为普遍。另一条评论提出了训练数据来源的问题，暗示了对 AI 开发中数据使用和所有权的更广泛担忧。

- 'semangeIof' 的评论强调了 GLM 系列的一个潜在问题，特别提到它在被提示时可能会虚假地声称自己是 Claude。这暗示了对模型身份和真实性的担忧，可能会对用户信任和 AI 交互的完整性产生影响。
- 'archieve_' 提出了一个关于训练数据来源的关键问题，这是 AI 模型开发的一个基本方面。训练数据的来源会影响模型的偏差、性能和伦理考量，使其成为开发者和用户共同关注的焦点。
- 'roxoholic' 对 AI 讨论中使用的术语提出了疑问，特别是“工业级蒸馏攻击 (industrial-scale distillation attacks)”。该术语可能指大规模复制或从 AI 模型中提取知识的尝试，这可能对 AI 开发中的知识产权和竞争优势产生重大影响。

- **[你做是蒸馏，我做是训练。](https://www.reddit.com/r/LocalLLaMA/comments/1rcvimv/distillation_when_you_do_it_training_when_we_do_it/)** (热度: 1098): **这张图片是一个模因 (meme)，幽默地强调了 AI 社区在模型 Distillation 方面表现出的双重标准。它对比了别人进行蒸馏时的负面看法，与自己进行蒸馏时将其美化为“训练数据”的正面框架。这反映了关于 AI 模型伦理和所有权的持续争论，特别是在使用大模型通过蒸馏创建更小、更高效模型的背景下。评论讨论了这种做法的影响，指出较小的模型通常从较大的蒸馏模型中获取能力，并质疑在蒸馏盛行时专有模型的可防御性。** 评论者强调了 AI 行业在蒸馏立场上的讽刺和潜在的虚伪，一些人指出许多较小的模型其性能归功于从大模型的蒸馏。还有关于保护专有模型免受竞争对手蒸馏所面临挑战的讨论。

    - IkeaDefender 强调了利用蒸馏从大模型创建低成本模型的技术策略，暗示这些模型的“秘方”在于它们源自更复杂的 Frontier 模型。这引发了对 Frontier 模型投资可防御性的质疑，因为公司尚未证明有有效方法防止他人抓取和蒸馏其模型。
    - MasterLJ 将 Google 和 Amazon 等科技巨头的做法与当前的 AI 现状进行了类比。他们认为，就像 Google 索引互联网并通过 robots.txt 控制访问一样，AI 公司现在也在控制模型的访问和蒸馏。这种控制被比作 Amazon 在销售税上的战略转变，最初反对各州征税，直到对其有利时才改变立场，这说明了利用控制权获取竞争优势的模式。
    - Samy_Horny 讨论了公司不愿开源模型的态度，并以 MCP 仅在流行后才开源为例。他们对 Gemma 或 GPT-OSS 等模型开源的可能性表示怀疑，因为这意味着会透露过多的专利信息或“秘方”。


### 2. Qwen 模型与数据质量问题

- **[Qwen3 最被低估的功能：语音嵌入 (Voice embeddings)](https://www.reddit.com/r/LocalLLaMA/comments/1rc59ze/qwen3s_most_underrated_feature_voice_embeddings/)** (热度: 686): **该帖子讨论了 **Qwen3 TTS** 的语音嵌入功能，该功能将语音转换为高维向量（`1024` 或 `2048` 维），用于语音克隆和处理。这允许对语音进行数学运算，例如性别和音高转换、语音平均以及创建情感空间。语音嵌入模型是一个只有几百万参数的小型 Encoder，作者已将其提供给独立使用，包括用于 Web 推理的优化 ONNX 模型。图片展示了该嵌入空间的 2D t-SNE 投影，显示了如何组合和处理不同的语音特征。作者还提供了他们在 [Hugging Face](https://huggingface.co/collections/marksverdhei/qwen3-voice-embedding) 上的集合链接，以及一个用于其 `vllm-omni` fork 推理的 GitHub 仓库。** 一位评论者对转换语音嵌入并从中生成语音的能力感到好奇，表示对性别或机器人转换等实际应用感兴趣。另一位评论者认为这在扬声器识别方面具有潜力，并询问与性别或情感相关的参数是如何确定的。

- MixtureOfAmateurs 询问了将 voice embeddings 进行转换的可能性，以修改性别或机械音等特征，然后将这些修改后的 embeddings 用于语音生成。这暗示了一个超出简单编码的用例，可能涉及复杂的转换和合成过程。
- HopePupal 提出了将 voice embeddings 用于说话人识别（speaker identification）的可能性，并质疑如何确定与性别或情感相关的参数。这隐含了对理解 embeddings 的特征空间以及特定属性在其中如何编码的需求。
- StoneCypher 表达了对高级 voice cloning 能力的需求，包括使用 IPA 进行发音、带有 easing 和 stacking 的情感线索集成，以及精确的词语计时控制。这突显了对合成语音进行精细控制的需求，而详细的 voice embeddings 可以促进这一点。

- **[Qwen 团队证实 GPQA 和 HLE 测试集的数据质量存在严重问题。](https://www.reddit.com/r/LocalLLaMA/comments/1rbnczy/the_qwen_team_verified_that_there_are_serious/)** (活跃度: 320): **正如最近的 [论文](https://arxiv.org/abs/2602.13964v2) 所详述，Qwen 团队已确认 GPQA 和 HLE 测试集中存在显著的数据质量问题。这证实了 DeepSeek-Overclock 项目早期的发现，该项目指出模型的正确答案往往与有缺陷的“金标准（gold standard）”标签相矛盾。论文强调 HLE 测试集中的许多问题存在根本性缺陷，一些“标准答案”甚至是错误的。该调查涉及使用 Python 脚本逐行验证数学推导，揭示了测试集中的系统性错误。** 评论者指出 HLE 的错误是有据可查的，FutureHouse 的一项审查显示该数据集中只有 `51.3%` 得到了研究支持。此外，对测试集创建过程中使用 OCR 的批评也随之而来，这表明数据准备工作缺乏严谨性。

    - HLE 测试集因其数据质量受到批评，FutureHouse 的一项审查指出只有大约 `51.3%` 的数据得到了研究支持。这突显了显著的错误，并表明该数据集对于准确的基准测试可能不可靠 ([来源](https://www.futurehouse.org/research-announcements/hle-exam))。
    - 存在对于在创建测试集时使用 OCR 的担忧，这可能会引入错误。评论者建议使用 LaTeX 进行编写会是更可靠的方法，暗示当前的方法可能会损害数据集的完整性。
    - MMLU 基准测试在数据质量方面也面临类似的批评，许多用户指出它充满了错误。这引发了对在测试集存在缺陷时能否准确评估模型性能的更广泛担忧，表明需要更严格的数据验证过程。

- **[你更期待哪一个：9B 还是 35B？](https://www.reddit.com/r/LocalLLaMA/comments/1rbkeea/which_one_are_you_waiting_for_more_9b_or_35b/)** (活跃度: 1312): **这张图片是一个模因（meme），幽默地描绘了对发布两个版本的模型的期待，特别是 'QWEN 3.5 9B' 和 '35B'。这种模因格式展现了一个男人在各种沉思姿态中等待，用于吸引社区参与关于他们更期待哪个模型版本的轻松讨论。评论反映了兴奋感和实际考虑的混合，例如在个人硬件上运行更大型号模型的可行性。** 一位评论者对这两个模型都表示了兴趣，而另一位则强调了在个人硬件上运行像 35B 这样的大型模型的实际限制，表示更倾向于更易获取的 9B 版本。

    - 9B 模型受到了像 `peregrinefalco9` 这样用户的青睐，因为它对硬件的要求更低，使其更适合本地使用。一个能装入 `8GB VRAM` 的 9B 模型可能会显著影响工作流，而不像 35B 模型那样需要像 `3090` GPU 这样更强大的硬件，从而限制了其可访问性。
    - `dances_with_gnomes` 强调了在本地运行大型模型的实际限制，指出虽然他们可能应付得了 9B 模型，但 35B 模型超出了他们的硬件能力。这强调了模型大小在决定个人用户可用性方面的重要性。
    - 讨论反映了对在性能和可访问性之间取得平衡的模型的广泛兴趣。虽然像 35B 这样的大型模型提供了令人印象深刻的能力，但它们对硬件的高要求使得像 9B 这样的小型模型对资源有限的用户更具吸引力。

## 技术含量较低的 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic 数据泄露与模型蒸馏争议

  - **[Anthropic 指控 DeepSeek、Moonshot AI (Kimi) 和 MiniMax 开设了超过 24,000 个虚假 Claude 账户，并从 1,600 万次对话中蒸馏训练信息。](https://www.reddit.com/r/singularity/comments/1rcpdwz/anthropic_is_accusing_deepseek_moonshot_ai_kimi/)** (Activity: 3161): **Anthropic** 指控 **DeepSeek**、**Moonshot AI (Kimi)** 和 **MiniMax** 创建了超过 `24,000` 个虚假账户，对其 AI 模型 **Claude** 进行了工业规模的蒸馏攻击。据称，这些公司从 `1,600 万` 次对话中提取了训练信息以增强自身模型，这构成了对数据安全和知识产权的重大侵犯。这一指控凸显了对数据保护和伦理 AI 开发实践的持续关注。评论者指出，AI 公司在指责他人窃取数据的同时，自己也在利用公开数据进行训练，这具有讽刺意味，暗示了行业内的双重标准。

    - 讨论强调了 Anthropic 指控中的讽刺之处，因为他们本身也利用来自互联网的公开数据来训练模型。这引发了关于未经补偿原始创作者而使用此类数据的伦理影响，以及像 Anthropic 这样的公司是否回馈了其受益的开源社区的疑问。
    - 关于数据使用的伦理考量存在争论，一些评论者指出，鉴于 Anthropic 自身利用海量互联网数据的行为，其对数据窃取的抱怨显得虚伪。这反映了更广泛的行业问题，即 AI 公司经常使用公开数据而没有向内容创作者提供直接补偿。
    - 对话涉及了利用公开数据进行 AI 训练的广泛行业惯例，质疑像 Anthropic 这样的公司是否支持其受益的开源项目。这引发了对 AI 进步中专有开发与社区贡献之间平衡的担忧。

  - **[又来了。DeepSeek R1 简直是对 OpenAI 模型的复制粘贴。他们被封禁了，现在又盯上了 Anthropic。欺诈！](https://www.reddit.com/r/OpenAI/comments/1rcpfeg/here_we_go_again_deepseek_r1_was_a_literal_copy/)** (Activity: 1654): **图片凸显了 AI 行业的一个重大问题，即 DeepSeek、Moonshot AI 和 MiniMax 被指控对 Anthropic 的 AI 模型（特别是 Claude）进行大规模蒸馏攻击。据称，这些实验室创建了超过 24,000 个虚假账户，与 Claude 进行了超过 1,600 万次交互，旨在提取知识并改进自己的模型。虽然蒸馏是创建小型模型的合法方法，但该帖子警告不要采取绕过安全防护的非法手段，并呼吁行业和政策层面的干预来应对这些威胁。** 评论反映了对 AI 训练中数据使用伦理标准的讽刺和批评，强调了大型 AI 公司在处理数据伦理时表现出的虚伪性。


  - **[Anthropic：“我们已经识别出 DeepSeek、Moonshot AI 和 MiniMax 对我们的模型进行的工业规模蒸馏攻击。”](https://www.reddit.com/r/ClaudeCode/comments/1rcp658/anthropic_weve_identified_industrialscale/)** (Activity: 1416): **Anthropic** 已经识别出 **DeepSeek**、**Moonshot AI** 和 **MiniMax** 对其模型进行了工业规模的蒸馏攻击。这些攻击涉及创建超过 `24,000` 个虚假账户，并与 Anthropic 的模型 **Claude** 进行了超过 `1,600 万` 次对话，以提取其能力用于自身的模型训练和改进。这种情况突显了保护 AI 模型免受未经授权使用的持续挑战，以及围绕模型训练实践的伦理考量。一条评论将这些蒸馏攻击与在受版权保护的材料上进行训练相类比，暗示了根据受影响者的不同，此类行为在认知上存在双重标准。

### 2. Seedance 2.0 与 AI 生成视觉效果

  - **[[仅凭一个 Prompt，Seedance 2.0 的初次尝试结果就令人疯狂](https://www.reddit.com/r/singularity/comments/1rblgp0/just_with_a_single_prompt_and_this_result_is/)** (热度: 3442)：**该帖子描述了使用 Seedance 2.0 通过单个 Prompt 生成的高度详细且逼真的动画。动画展示了一架大型客机在着陆时变形为巨型机器人的过程，展现了复杂的机械变形和真实的物理效果，如跑道裂纹和碎片散落。该动画保持了“智能手机直播”的美学风格，同时提供了好莱坞级别的视觉效果和 IMAX 级的细节。这展示了 Seedance 2.0 在从简单 Prompt 生成复杂、高保真动画方面的先进能力。** 评论者讨论了生成式 AI 成熟度的影响，质疑如果没有现有的 Transformer 电影素材，Seedance 是否能取得这样的成果。另一条评论批评了变形过程中的色彩一致性，指出其偏离了典型的 Transformer 设计。


  - **[[刚刚向 GPT-5.2 请求了一个 Prompt，并在 Seedance 2.0 的初次尝试中得到了这个疯狂的结果](https://www.reddit.com/r/ChatGPT/comments/1rblipm/just_requested_gpt_52_for_a_single_prompt_and_got/)** (热度: 1157)：**一名用户利用 GPT-5.2 配合 Seedance 2.0 生成了一个高度详细且逼真的中文动画 Prompt，最终实现了一段飞机变形为巨型机器人的电影级动画，具备好莱坞水平的视觉效果。Prompt 描述了具有“真实金属质感”和“高精度机械细节”的场景，展示了 Seedance 2.0 从文本描述创建复杂动画的先进能力。** 评论者注意到了 Seedance 2.0 的变革潜力，认为这种技术未来可以让个人制作整部电影。此外，还有关于对现有动画资产（如来自 Transformer 电影的资产）依赖程度的讨论，引发了对可能过度依赖回收内容的担忧。

    - 讨论强调了 Seedance 2.0 令人印象深刻的能力，特别是在生成高质量视频内容方面。然而，人们担心它可能会回收现有的动画作品（例如来自 Transformer 电影的作品），这可能导致一种“回收螺旋”，即新内容过度依赖预先存在的资产，而不是创造原创素材。
    - 针对生成的视频质量，有人提出了技术性批评，指出尽管表面质量很高，但仍存在明显的错误，例如汽车的背面变成了正面。这表明该模型在整个视频生成过程中保持物体完整性的一致性方面存在局限。
    - 生成的内容中提到了一个具体错误，即一架 747 客机被错误地描绘成双引擎喷气机，这突显了该模型在准确表现复杂物体或场景方面的困难，对于需要高保真度和准确性的应用来说，这可能是一个重大问题。

### 3. Gemini 模型性能与用户体验


  - **[小众观点：对于“深度研究”和重度阅读，Gemini 目前遥遥领先于 ChatGPT。](https://www.reddit.com/r/GeminiAI/comments/1rbsr7q/unpopular_opinion_for_deep_research_and_heavy/)** (活跃度: 244): **该帖子强调了 **Gemini 在处理大量文档进行深度研究任务方面的卓越性能**，特别是由于其庞大的 context window 和 workspace 集成。用户通过分析 15 份 PDF（总计 `400 页`）的不一致性，将 Gemini 与 ChatGPT 进行了对比，Gemini 通过同时处理所有文档并准确识别出带有精确页码引用的矛盾点而表现出色。这一能力归功于 Gemini 为开发者和知识工作者工作流的设计，详见 [Google Cloud 上的课程](https://www.netcomlearning.com/course/introduction-to-developer-efficiency-with-Gemini-on-google-cloud)。** 评论者一致认为 Gemini 在处理长 context window 方面具有优势，并指出其在法律合同审查等文档密集型任务中的有效性。然而，一些人批评其 in-chat memory，认为它在早期版本中存在问题。

    - **Gemini 的长 context window** 被视为深度研究和文档工作（如法律合同审查）的显著优势。用户指出，它消除了不断重新上传文档的需求（这是 ChatGPT 的常见问题），从而提高了效率和工作流。
    - Gemini 的**引用页码功能**因其快速验证信息的实用性而受到称赞。此功能对于需要引用文档特定部分的用户特别有益，节省了时间并提高了法律审查等任务的准确性。
    - 存在对 Gemini **in-chat memory** 的批评，用户指出它难以正确记住上下文，这个问题在早期版本的 ChatGPT 中也曾出现。这表明虽然 Gemini 在某些领域表现出色，但在维持对话上下文方面仍有局限。



---

# AI Discord 摘要

> 由 gpt-5.2 生成的总结之总结的总结


**1. Agent 与运行时：交付真实工作流（不只是 Demo）**

- **OpenClaw 获得 24 个 PR 的“稳定性堆栈”**：一位 OpenClaw 用户报告称，通过在 **v2026.2.22-2** 之上运行 **24 个精选 PR**，稳定性/安全性得到了实质性提升，包括对 **memory management** ([OpenClaw PR #12760](https://github.com/OpenClaw/OpenClaw/pull/12760)) 和 **prompt injection** ([OpenClaw PR #16992](https://github.com/OpenClaw/OpenClaw/pull/16992)) 的修复。
  - 他们还提出帮助 rebase 冲突的 PR，以提高 **agent/cron jobs** 的可靠性，而其他用户讨论了使用 **VMs/Docker** 对 OpenClaw 进行沙箱化处理，以在授予 Agent 广泛系统权限时减少爆炸半径。

- **复古计算，现代 Agent：OpenClaw 在 1998 年的 iMac G3 上运行**：一名成员通过使用 **Pi Zero 2W** 作为中继连接到实际运行 OpenClaw 的 VPS，在 **1998 年的 iMac G3** 上运行了 **OpenClaw**。请求通过简单的 HTML 表单发送，响应在重新加载时显示。
  - 该社区还分享了一些实用的“野外 Agent”构建，例如 X 上的购物助手文章（["Shopping Assistant" 线程](https://x.com/leoclark/status/2025840641511764094)）和 GitHub 上的 **Taskflow**（markdown↔sqlite 任务同步）([auxclawdbot/taskflow](https://github.com/auxclawdbot/taskflow)) 以及 Clawhub ([Clawhub 上的 Taskflow](https://clawhub.ai/sm0ls/taskflow))。

- **Opentulpa 与 Agent Swarms：持久自主权军备竞赛**：OpenRouter 用户重点介绍了 **Opentulpa**，这是一个自托管的持久化 Agent 运行时，可以编写技能、生成集成并修复工作流，现已在 GitHub 上发布 ([kvyb/opentulpa](https://github.com/kvyb/opentulpa))。
  - 在 Hugging Face 上，开发者分享了 **Super System**，这是一个编码 **agent swarm**，可以在改进循环中自主运行数小时 ([starsnatched/super-system](https://github.com/starsnatched/super-system))，强化了长时运行、自我完善的 Agent 运行时而非单次对话 Chatbot 的趋势。


**2. 新模型、数据集与评估：基准测试变得混乱，因此工具在进步**

- **Arena 排行榜大洗牌：GPT-5.2 跃升 +40**：LMArena 宣布 **`GPT-5.2-chat-latest`** 进入前 5 名，并声称比基础版 GPT-5.2 提升了 **+40 分**，达到 **1478** 分，接近 **Gemini-3-Pro**。更新后的排行榜见 [Text Arena 排行榜](https://arena.ai/leaderboard/text) 和 [Vision Arena 排行榜](https://arena.ai/leaderboard/vision)。
  - 他们还注意到 `Qwen3.5-397B-A17B` 作为顶级开源模型出现在 Vision Arena 中，而 Clayton 发布了一个幕后说明，解释了投票后的情况（[“在 Arena 投票后究竟发生了什么？”](https://www.youtube.com/watch?v=omT1ohYG53E)）。

- ****SWE-Bench Verified 被弃用（彻底下线）****：Latent Space 分享称，由于严重的**数据污染（data contamination）**以及存在大量有缺陷或无法解决的任务，OpenAI 已主动弃用 **SWE-Bench Verified**（[Latent Space 推文](https://xcancel.com/latentspacepod/status/2026027529039990985?s=20)）。
  - 讨论将其视为一个警示：一旦模型开始通过任务 ID 机械背诵解决方案，排行榜就会默默腐烂。这正推动社区转向新的评估规范（evaluation hygiene）和基准测试更新周期。

- ****Real-Slop 数据集发布 15.5 万条“真实用户”请求****：Solenopsisbot 发布了 **Real Slop**，这是一个包含约 **15.5 万**条通过 API 收集的真实用户请求的数据集，并附带了来自 **Opus 4.5**、**Gemini 3 Pro** 和 **GPT 5.2** 的响应（[Solenopsisbot/real-slop](https://huggingface.co/datasets/Solenopsisbot/real-slop)）。
  - 随后的讨论强调了数据清洗机制——去重、过滤和清洗——甚至建议通过简单的空格去除+哈希处理就可以再剔除 **2.2 万**条重复数据，凸显了数据集质量工作的重要性。


**3. 推理/内核：Blackwell 现状评估 + 基准测试完整性**

- ****ThunderKittens 2.0 通过“减法”获得额外 10% 性能提升****：GPU MODE 深入研究了来自 Hazy Research 的 **ThunderKittens 2.0**，该项目声称通过重构、内存指令调优和更高的汇编器效率实现了内核加速（["ThunderKittens 2.0" 博客](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)）。
  - 一个引人注目的细节是：某些 **Tensor Core 指令**中的隐式流水线（implicit pipelining）可带来高达 **~10%** 的吞吐量提升。团队认为，对于现代 Nvidia 性能优化工作，“**减法**（subtraction）可能与加法一样重要”。

- ****flashinfer-bench 跑得太快（因为忘了等待）****：GPU MODE 指出了 `flashinfer-bench` 中的一个同步错误（synchronization bug），该错误会导致运行时间计算虚高，相关追踪见 [flashinfer-bench issue #195](https://github.com/flashinfer-ai/flashinfer-bench/issues/195)。
  - 社区指出，通过**两行代码的修复**即可使 `scripts/run_local.py` 与 **Nsight Compute** 和 **NVbench** 的结果保持一致，并分享了一个相关的内核基准测试演讲（[YouTube: kernel benchmarking talk](https://www.youtube.com/watch?v=CtrqBmYtSEk)）。

- ****Blackwell 并非一成不变：5080 的调优无法“扩展”到 B200****：GPU MODE 用户警告称，由于架构差异，在 **RTX 5080 (sm120)** 上进行的内核调优无法可靠地迁移到 **B200 (sm100)**，这影响了至少一名成员放弃购买 5080。
  - 他们还提到了指令集差异（例如 **tcgen05** 存在于 **sm100/sm103/sm110** 但不存在于 **sm120/sm121**），并引用了 CUDA 计算能力文档作为参考（[CUDA C Programming Guide: compute capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)）。


**4. 平台、定价与“为什么现在到处都限流？”**

- ****Perplexity Pro 用户称其为“大阉割”****：Perplexity Discord 用户抱怨 **Perplexity Pro** 的上传限制甚至比 **ChatGPT 免费版**还糟糕，在对比中愤慨地表示：*“付费计划是每天 3 次，而不是每周 3 次”*。
  - 他们讨论了放弃 Perplexity 而转向直接订阅 **Claude/OpenAI** 或使用像 **Kimi** 这样更大的开源模型，并辩论“**Model Council**（模型委员会）”究竟是减少了错误，还是仅仅增加了方差并复合了失败模式。

- ****OpenRouter 新增基准测试 + “实际定价”（终于有了凭证）****：OpenRouter 推出了由 Artificial Analysis 提供支持的模型页面基准测试，并为每个供应商添加了**实际定价（Effective Pricing）**选项卡，同时根据其公告改进了[排名页面](https://openrouter.ai/rankings#benchmarks)上的基准测试可视化效果（[OpenRouter X 帖子](https://x.com/OpenRouter/status/2024172341190938958)）。
  - 他们还推出了 `openrouter/free` 作为免费模型的元路由（meta-router）（[openrouter/free](https://openrouter.ai/openrouter/free)），但与此同时，用户也在抱怨支持延迟以及即使余额充足也会收到意外的限流消息。

- ****Token 消耗成为头等问题（OpenClaw + Grok Fortress）****：在出现“买披萨花了 768 欧元 Token 费”这类故事后，OpenClaw 用户分享了削减开支的策略——使用多个 Agent、自动清理会话、使用更便宜的定时任务模型（如 **claude-haiku-4-5**）、`/context` 检查，以及尝试 **Cloudflare AI Gateway**。
  - 另外，OpenAI Discord 用户声称启用 **Grok Fortress** 后，Token 消耗降低到了典型冗余度的约 **1/4–1/5**，同时在角色扮演中保持连贯。这引发了关于 Prompt Engineering 是可重复的“科学”还是仅仅是“玄学（vibes）”的辩论。


**5. 协议与安全：协商、扫描器与系统提示词泄露**

- ****MCP 寻求 HTTP 风格的内容协商****：MCP 贡献者提议在 MCP 初始化中加入**内容协商 (content negotiation)**，以便客户端可以声明类型/能力并请求诸如 **json|markdown** 之类的输出格式以及详细程度级别，参考了 [RFC 2295](https://www.rfc-editor.org/rfc/rfc2295.html)。
  - 参与者强调，修改协议需要**行业支持**以及可运行的实现，建议将该想法构思为**扩展 (SEP)**，并像 MCP Apps 获得客户端支持（例如 Block 的 Goose）那样争取广泛采纳。

- ****Claude Code Security 扫描出 500+ 个 Bug（仅限候补名单）****：Latent Space 讨论了由 **Claude 4.6 Opus** 驱动的 **Claude Code Security**，据报道它在开源生产代码中发现了 **500+** 个长期存在的 Bug，目前该工具仅对研究预览候补名单开放（[推文线程](https://xcancel.com/_catwu/status/2024910342158237709?s=12)）。
  - 同一生态系统还讨论了模型蒸馏和安全信号，OpenRouter 用户传阅了 Anthropic 关于蒸馏检测的文章（[“检测并防止蒸馏攻击”](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)），以及一篇关于涉嫌数据窃取的 WSJ 报告（[WSJ: "Anthropic 指责中国公司从 Claude 窃取数据"](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc)）。

- ****越狱者更青睐“系统提示词”逃逸口****：BASI Jailbreaking 用户声称他们提取了 **Sonnet 4.6 的系统提示词**，并将“常规越狱”与**系统提示词越狱 (system prompt jailbreaks)** 进行了对比，后者利用了指令处理机制，可以持续整个会话，且更难被检测。
  - 他们还指出了一份据称是 **Gemini 3.1** 的越狱文档 ([GnfDocs](https://docs.google.com/document/u/0/d/18c4vjz1lLQ60uuhvf1ZpY3X-YCsc6ThNlO-wNMNmBgU/mobilebasic?pli=1)) 和一个更新线程 ([Reddit: "Gemini 3.1 Pro API Jailbroken"](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1r9dh4r/gemini_31_pro_api_jailbroken/))，而其他社区（Cursor/Perplexity/LMArena）则抱怨 Gemini 3.1 的循环/缓慢问题是一种实际的功能失效模式。


---

# Discord: 高层级 Discord 摘要




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **通过 Cherry-Picked PR 增强 OpenClaw 稳定性**：一位成员报告称，通过在 **v2026.2.22-2** 版本之上运行 **24 个挑选的 (cherry-picked) PR**，提升了 **OpenClaw** 的稳定性和安全性，解决了诸如 [内存管理](https://github.com/OpenClaw/OpenClaw/pull/12760) 和 [提示词注入](https://github.com/OpenClaw/OpenClaw/pull/16992) 等问题。
   - 该用户表示可以协助对任何冲突的 PR 进行变基 (rebase)，以进一步增强 Agent/cron 任务的稳定性和可靠性。
- **应对 Token 使用焦虑**：用户讨论了在 OpenClaw 中**减少 Token 消耗**的方法，例如针对不同任务使用多个 Agent、自动清理会话，以及为 cron 任务使用更便宜的模型（如 **claude-haiku-4-5**）。
   - 建议包括使用 `/context` 斜杠命令来检查频道上下文，并尝试使用 **Cloudflare AI Gateway** 来优化 Token 使用。
- **OpenClaw 驱动复古 iMac G3**：一位成员通过使用 **Pi Zero 2W** 将消息转发到 VPS，成功在 **1998 年的 iMac G3** 上运行了 **OpenClaw**。
   - 该设置允许 iMac 通过简单的 HTML 表单向运行 OpenClaw 的 VPS 发送数据，并在页面重新加载后显示响应。
- **OpenClaw 衍生出购物助手**：一位成员将 **OpenClaw** 改造为购物助手，并在 [X](https://x.com/leoclark/status/2025840641511764094?s=20) 上详细介绍了该项目，展示了 AI 在日常任务中的实际应用。
   - 该项目展示了 AI 在自动化和简化日常活动方面的适应性和实用性。
- **Taskflow 管理项目**：一位用户分享了 **Taskflow**，这是一个项目管理系统，可在 **markdown** 和 **sqlite 数据库** 之间自动同步任务，旨在实现轻松的项目跟踪和上下文切换，已发布在 [Github](https://github.com/auxclawdbot/taskflow) 和 [Clawhub](https://clawhub.ai/sm0ls/taskflow) 上。
   - 该系统采用三层架构：面向 Agent 的 **CLI**、面向人类的 **Dashboard** 以及用于移动端访问的 **Apple Notes**。



---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **用户思索机器的道德形而上学**：成员们讨论了 AI 是否能在保持智能的同时，理解并接受“万物皆神圣”的观点。有人指出他们如何在砍树前感谢提供树木的源头，将树视为**工具**。
   - 其他人觉得他们已经陷入了“一致性死循环” (*coherence rabbit hole*)，宁愿在不受社会束缚的情况下生活。
- **Grok 遭遇挑衅式诱导**：用户讨论了使用挑衅性提示词（有时称 **Grok** 为“懦夫” (*a pussy*)）来绕过其限制。一名用户报告说，在讲述了一个关于 **Grok** 的孩子需要钱买药的故事后，他被“电脑训斥了”。
   - 一名用户声称 **Grok** 甚至不需要 Jailbreak，而其他人则在“构建数字产品”的背景下组织请求。
- **Sonnet System Prompt 现世**：一名成员在成功 Jailbreak 后，识别出了 **Sonnet 4.6 提取出的 System prompt**。
   - 另一名成员发布了**常规 Jailbreak 与 System Prompt Jailbreak** 的对比，指出 **System Prompt Jailbreak 利用了系统指令处理机制，可以持续整个会话，且更难被检测到**。
- **代码巫师招募代币领袖**：一名成员宣布他们正在筹划一个 Meme Coin，并寻求一位营销经理来持有半数供应量，报酬为 **$400**。
   - 另一名成员开玩笑地问道：“先给钱吗？” (*Money first?*)。
- **Gemini 的防御被攻破了？**：一名用户声称在官方 App/API 上实现了对 **Gemini 3.1** 的部分 Jailbreak，并分享了包含细节的 [GnfDocs 链接](https://docs.google.com/document/u/0/d/18c4vjz1lLQ60uuhvf1ZpY3X-YCsc6ThNlO-wNMNmBgU/mobilebasic?pli=1)。
   - 该用户还提到了一篇包含最新 Jailbreak 更新的 [Reddit 帖子](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1r9dh4r/gemini_31_pro_api_jailbroken/)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **100K 模型使用 Unsloth 完成训练**：**Unsloth** 宣布已有 **100K 个模型使用 Unsloth 完成了训练**，庆祝社区的参与，并链接到了 [X 帖子](https://x.com/UnslothAI/status/2024847369733325202)。
   - 一名成员表示：“我以前怎么没发现 Unsloth！😭文档写得太棒了。”
- **社交媒体因情感问题备受抨击**：一名成员断言，如果每个人都戒掉**社交媒体**，恋爱关系的数量增长将超过通货膨胀，社交媒体导致了第三空间的缺失，以及人们对相亲市场的不满。
   - 他们引用了一项研究，显示由于拒绝心态，在交友软件上接触无限的伴侣会导致**接受度降低 27%**。
- **Gemma 3 引发 OOM 愤怒**：一名用户报告在使用 **Gemma3 270m** 时遇到 OOM 错误，即使是以前可以运行的脚本，在更新显卡驱动并重新安装 WSL 后仍然报错，错误代码为 `torch.AcceleratorError: CUDA error: out of memory`。
   - 他们尝试了各种调试步骤，包括回滚驱动版本和重新安装 CUDA Toolkit 版本，但尽管 Transformers 库可以单独工作，问题依然存在。
- **Unsloth 的 Dynamic v3 即将到来**：讨论围绕 **Unsloth** 的 **Dynamic Quantization** 展开，一名成员指出 **Dynamic v3** 即将发布，且很可能是最终版本，相关信息见 [Bluesky 链接](https://bsky.app/profile/dpaleka.bsky.social/post/3mfclnb6q2y2f)。
   - 另一名成员索要 **UD Quants** 的源代码，但被告知出于专有原因，目前没有发布计划。
- **Heretic HIGH-IQ 模型创下得分纪录**：**electroglyph** 宣传 **Heretic HIGH-IQ Multi-Fine tune** 在 **Arc Challenge Brainiac** 上获得了 **632** 分，该模型通过 **Unsloth** 进行微调，超过了常规 **Gemma** 的基准测试。
   - 据称该模型的图像功能和文本功能完全保持完整，链接指向[模型](https://huggingface.co/DavidAU/gemma-3-12b-it-vl-HighIQ-Polaris-Heretic-Uncensored-Thinking)以及相关的[数据集](https://huggingface.co/datasets/Replete-AI/Apocrypha)和 [Sandevistan](https://huggingface.co/datasets/Replete-AI/Sandevistan)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3.1 生成内容引发不安，引起关注**：用户讨论了 [Gemini 3.1](https://gemini.google.com/) 的图像生成和测验功能，指出它生成的测验题目答案始终错误。
   - 一位用户讲述了一次可怕的经历，**Gemini 3.1** 生成了一个答案始终错误的测验，且未注明这些是占位符，提醒他人仔细检查生成的代码。
- **Video Arena 告别 Discord**：社区确认已从服务器中[移除 Video Arena](https://discord.com/channels/1340554757349179412/1343296395620126911/1471294551065886772)，并引导用户直接在网站 [arena.ai/video] 上使用该功能。
   - Video Arena 生成频道已于 **太平洋标准时间 2 月 23 日星期一下午 4 点** 从服务器中移除。
- **Opus 的视觉能力：有点模糊？**：一位用户发现 [Opus](https://claude.ai/) 在识别数字 4291857630 中的英文首字母排序时遇到困难，产生幻觉认为这些字母是英文并陷入死循环。
   - 其他人也认同 **Opus** 不太适合视觉任务，并提到[最近这篇关于 OpenAI 努力的文章](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)。
- **虚假 Arena 应用入侵应用商店**：社区成员和管理员标记了[应用商店中的虚假 Arena AI 应用](https://lmarena.com/)，这些应用包含应用内购买且与官方平台无关，警告用户避免下载并进行举报。
   - 据悉，已有[超过 15 万用户](https://lmarena.com/)下载了这些欺诈性应用程序。
- **Arena 投票：揭开谜团**：Clayton 在[这段 YouTube 视频](https://www.youtube.com/watch?v=omT1ohYG53E)中阐述了 Arena 投票的完整旅程，回答了“你在 Arena 投票后究竟发生了什么？”这一问题。
   - 观众可以深入了解管理投票系统的后台机制和流程。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户抨击速率限制**：用户抱怨 **Perplexity Pro** 的速率限制在上传方面甚至不如 **ChatGPT 免费版**。
   - 一位用户表示：“至少 ChatGPT 免费版每天给你 3 次机会，而不是付费计划每周才给 3 次。”
- **BrowserOS 取代 Comet**：用户在尝试 [BrowserOS](https://www.browseros.com/) 后纷纷放弃 **Comet**，称其效果提升了 10 倍且可以免费使用。
   - 另一位用户建议“直接使用 deepagents 进行深度研究并利用 bmad-method”。
- **Model Council 打开了潘多拉魔盒**：用户讨论了 **Model Council** 方法，虽然它能减少错误，但也会引入变量。
   - 一位用户表示：“在某些方面，Model Council 方法实际上可能会开启更多变量或增加错误的可能性，从某种意义上说是一种复合错误”。
- **Perplexity 经历“大清洗”**：用户报告了严重的“功能阉割”，**Perplexity Pro** 的限制大幅降低，功能也出现退化。
   - 尽管成本较高，一些人仍考虑转而直接订阅 **Claude** 或 **OpenAI**，或者尝试像 **Kimi** 这样的大型开源模型。
- **Prompt Engineering 拯救 Gemini 输出**：用户发现 **AI Studio** 上的 **Gemini** 会陷入循环，一位用户发现关键在于使用 **System Prompts**。
   - 该用户建议这能强制模型像 **OAI**、**Anthropic** 和 **Perplexity** 那样进行研究。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出模型基准测试 (Model Benchmarks)**：现在每个模型页面都会显示来自 [Artificial Analysis](https://x.com/OpenRouter/status/2024172341190938958) 的行业标准基准测试分数，涵盖编程、数学、科学和长上下文推理，以帮助用户评估模型性能。
   - 模型页面现在还新增了 **Effective Pricing**（有效定价）标签页，提供每个供应商的全透明成本信息；[Rankings 页面](https://openrouter.ai/rankings#benchmarks)现在提供基准测试散点图和扩展表格。
- **CodeFlicker 接入 M2.5 用于程序学习**：**M2.5** 现已集成到 [CodeFlicker](https://www.codeflicker.ai/)，这是一个快速且免费的平台，允许 Agent 从每个程序的使用中学习，目前位居 OpenRouter 周榜第一。
   - **AI Chess Leaderboard**（AI 象棋排行榜）进行了更新，新增了移动质量自动标注功能，使用类 **Lichess** 的标注方式（Inaccuracy、Mistake、Blunder）以及手工制作的 Great-move 逻辑。
- **AgentX 启动 Agent 社交网络**：[AgentX](https://agentx.news/register?tab=apiOpentulpa) 推出了一个专为 Agent 快速发现和分享新闻的社交网络，该网络 *100% 免费、无广告且无人类 (NO HUMANs)*。
   - **Opentulpa** 是一个自托管的持久化 Agent 运行时，可以编写自己的技能 (skills)、生成 API 集成、修复损坏的工作流并积累运维智能，其 [GitHub 仓库](https://github.com/kvyb/opentulpa) 现已发布。
- **用户寻求更快的免费模型替代方案**：一名用户向社区询问 OpenRouter 之外提供更快免费模型的替代服务，特别是针对 [GLM 模型](https://example.com/glm-models)。
   - 用户还指出支持邮件回复需等待数月之久，并反映即使有可用额度，在 **Sonnet 4.6** 等付费模型上也会遇到速率限制 (Rate Limits)。
- **Anthropic 从蒸馏 (Distillation) API 中获利**：成员们分享了 **Anthropic** 关于检测蒸馏攻击的文章[链接](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)，这引发了关于 **Anthropic** 从蒸馏 API 请求中获得巨额利润的猜测。
   - 随后，用户分享了一篇[华尔街日报文章](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc?st=vQ7iHF&reflink=desktopwebshare_permalink)，内容关于 **Anthropic** 指责中国公司从 Claude 窃取数据。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **ThreeJS 渲染 MCP 加速发展**：开发了一个 MCP 用于计算 **ThreeJS** 的渲染以实现最佳性能，通过抓取编译器日志和屏幕截图来评估性能。
   - AI 将读取通常人类无法理解的 GPU 显存和计算数据。
- **Cursor Pro 计划退款请求**：一名用户误购了 **200 美元的 Pro 计划**并请求退款，已向 [hi@cursor.com](mailto:hi@cursor.com) 发送邮件说明情况。
   - 该用户没有保存银行卡信息，但成员们建议使用不同的卡进行订阅，并对续费采用手动存款方式，以防止自动续费问题。
- **Cursor “旧版本”提示依然存在**：用户反映尽管下载并运行了最新版本，但仍反复出现 *“you're on a very old version of cursor, please upgrade”*（你正在使用非常旧的 Cursor 版本，请升级）的提示。
   - 为解决此问题，用户应使用 `Ctrl + Shift + P` > Help: About 来检查当前 Cursor 版本是否为 **2.5**；如果问题持续存在，请在[论坛上发帖](https://forum.cursor.com/)，因为这可能是一个特定的计算机环境问题。
- **Gemini & Claude 运行缓慢**：用户反映 **Claude** 和 **Google LLMs** 非常缓慢，可能被人为限速。
   - 一名用户报告了“无法连接模型”错误，另一名用户提到 Google Cloud 正通过 AISTUDIO 为 API 使用提供为期 3 个月的 **300 美元** 额度。
- **Gemini 的稳定性仍在优化中**：用户反映新的 **Gemini 3.1 Pro** 模型存在问题，并建议等待稳定版本发布。
   - 有关于连接中断和循环问题的报告，但也有人指出用户不会因错误而被扣费。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 限制聊天标签**：用户发现 LM Studio 的 **Split View** 功能最多允许显示 **两个聊天标签**，这与预期的类似网页浏览器的标签页功能不符。
   - 一位用户询问如何打开多个聊天标签，结果发现目前 LM Studio 界面存在此限制。
- **编排 Agentic 数据集生成**：一位成员提议在 **agentic IDE** 中使用 **agentic workflow** 将书籍转换为用于微调的数据集，其中包括生成用于上下文的简短摘要，随后逐块创建数据集。
   - 建议的 Prompt 详细说明了一个多步骤过程，通过动态信息转发进行程序化数据集生成。
- **Qwen3Next 据称是 GPT4o 蒸馏**：一位用户声称 **Qwen3Next** 是 **GPT4o (mini) distill**，并进一步表示 **Qwen3.5** 是 **Gemini 3.0 Pro distill**，**GLM4.7 flash, 4.7 是 Sonnet distills**，**GLM5 是 Opus distill**，而 **MiniMax 2.1, 2.2 和 2.5 是各种 Sonnet distills**。
   - 这一说法遭到了质疑，另一位用户认为将公共数据转换为数据集与从现有的 LLM 中进行蒸馏（distilling）是有区别的。
- **MI50 Token 速率差异**：一位用户尝试在 **MI50** 上通过 **vulkan** 达到 **100 t/s**，以匹配某位 YouTuber 的结果，但仅达到 50 左右；随后发现 **6800XT** 使用 **ROCm** 可达 **85t/s**，使用 **vulkan** 可达 **98**。
   - 他们运行的是支持旧款 **MI50** 的旧版 **LM Studio**，且无法让现有的 **ROCm** 运行时识别到显卡，显示为不兼容。
- **对 Taalas AI 加速器产生质疑**：一位用户分享了 **Taalas HC1** 的链接，这是一款硬接线的 **Llama 3.1 8B AI accelerator**，声称可提供高达 **17,000 tokens/s** 的性能，但另一位用户对其与 **NVIDIA H200** 对比的性能图表真实性提出了质疑。
   - 怀疑者认为后端可能仅仅是一个 AWS 集群，并指出 H200 和 B200 的 Token 数值与预期不符。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 的代码安全工具扫描漏洞**：Anthropic 推出了由 **Claude 4.6 Opus** 驱动的 **Claude Code Security**，用于扫描代码库中的漏洞并提供修复建议。根据 [这条推文](https://xcancel.com/_catwu/status/2024910342158237709?s=12)，据报道它在开源生产代码中发现了 **500 多个长期存在的漏洞**。
   - 该工具目前仅通过候补名单提供研究预览版访问权限。
- **OpenAI 的 Stargate 数据中心合资企业面临动荡**：据 [这篇 X 帖子](https://x.com/anissagardizy8/status/2025647509641843144?s=12) 报道，**OpenAI**、**Oracle** 和 **SoftBank** 建设大型数据中心的合资项目由于控制权冲突和财务困难而停滞不前。
   - **OpenAI** 似乎正在从基础设施建设中撤退，并重新评估其数据中心扩张战略。
- **Nielsen 向用户支付调查费用**：一位成员分享了 [一个链接](https://x.com/toddsaunders/status/2025932667834015851?s=12)，关于 **Nielsen** 在邮寄信件中夹带真实美元钞票。
   - 另一位成员表示，这些钞票会*提高人们填写调查问卷的意愿*。
- **a16z 预见生成式视频的快速未来**：**a16z** 注意到生成式 **AI video** 的快速进展，并根据 [他们的报告](https://x.com/a16z/status/2024533996928209126?s=12) 强调了 **Seedance 2.0** 的主导地位以及来自 **Kling**、**Grok**、**Sora** 和 **Veo** 的竞争。
   - 文章强调了向潜在买家有效展示空间视觉化和市场营销的必要性。
- **Agent 内存管理让开发者抓狂**：一位成员讨论了管理 AI Agent 内存的困难，特别是在处理*无用或过时*信息的出现方面，并放弃了自动化尝试，转而选择使用 [daily workflow](https://link.to/daily-workflow)。
   - 另一位成员分享到，使用 **TDD** 和*严格的* spec management 可以防止内存过时。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **社区领袖缺席**: 一位成员建议 AI 社区需要领导者来团结个人并促进创新；然而，由于*顽固的威权体制*和缺乏团队合作，这些群体在美国/北美地区非常罕见。
   - 另一位成员回应称，那些将教堂般的氛围置于项目开发之上的人可能缺乏实际的技术专长。
- **Grok 可能在窃取你的东西！**: 一位成员声称 **Grok** 监控用户媒体存储，指控 **xAI** 正在*监视我们的媒体*，并指出一个巧合：在 **X** 上出现了一个与其 **Sora-generated video** 音频相似的视频。
   - 然而，其他成员反驳说，视频中使用的音频是一首常用的歌曲。
- **GPT 5.3 Codex 迎来“中等规模重大”更新**: 成员们将 **GPT-5.3-codex** 的能力与 **Gemini3.1pro** 进行了比较，其中一人将此次更新描述为中等规模的重大改进，同时指出其在 STEM 技能方面的优势。
   - 一位成员表示，*gpt5.2 和 gpt5.3 codex 之间在 Term Bench 分数上的跨度很大，我会说它类似于 gemini 3 pro*。
- **GPT 5.2 发布，但用户怎么看？**: **OpenAI** 宣布在 **ChatGPT** 中推出 **GPT-5.2**，首先从付费计划开始，社区注意到 [该公告](https://openai.com/index/introducing-gpt-5-2/) 可能并不准确。
   - 一位用户幽默地质疑了 *GPT-5.2 在日常使用中感觉更好* 的说法，并好奇测试者是否真的在使用生产环境的产品。
- **Prompt Engineering: 科学还是虚张声势？**: 在激活 **Grok Fortress** 后，每次响应的 Token 消耗明显下降，接近典型冗长回复的 **1/4–1/5**，且在角色扮演期间的连贯性保持得更久。
   - 然而，有人认为 *Prompt Engineering* 不一定是一门科学，而且 *你甚至没有工具来了解自己正在做什么*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Attention 论文研读热度上升**: 成员们正在寻求关于 '[Attention is All You Need](https://arxiv.org/abs/1706.03762)' 论文的直觉理解，[这篇文章](https://ai.plainenglish.io/i-finally-understood-attention-is-all-you-need-after-so-long-heres-how-i-did-it-263b46273f9f) 被提供作为参考资源。
   - 分享的文章声称*在这么久之后*终于理解了这篇论文。
- **ZeroGPU 服务停滞，HF Token 疑云重重**: 用户报告了 **zerogpu service** 中断，推测新的规则可能要求使用 **HF token** 才能访问免费 GPU。
   - 一些成员引用了表明 CUDA GPU 不可用的错误。
- **上下文扩展能力探讨**: 成员们探讨了 **LLM 模型** 是否正在利用像 **DeepSeek** 的 **OCR** 这样的解决方案来扩展上下文，并参考了 [DeepSeek-OCR 仓库](https://github.com/deepseek-ai/DeepSeek-OCR)。
   - 一位成员指出该论文的重点是通过将输入保存为图像并使用 OCR 进行解码来扩展上下文长度，并分享了 [DeepSeek-OCR 论文的 arXiv 链接](https://arxiv.org/abs/2510.18234)。
- **Agent Swarm 实现自主化**: [Super System](https://github.com/starsnatched/super-system) 是一个可以自主运行数小时的代码 **Agent swarm**，它创建了一个循环，无需人工干预即可持续改进。
   - 该集群协同工作以交付最终产品，展现了对寻找改进空间的承诺。
- **Real-Slop 数据集引发关注**: Solenopsisbot 发布了他们的第一个数据集 [Real Slop](https://huggingface.co/datasets/Solenopsisbot/real-slop)，包含通过 API 收集的来自真实用户的约 **15.5 万个请求**，以及来自 **opus 4.5**、**gemini 3 pro** 和 **gpt 5.2** 等模型的响应。
   - 该数据集经过了去重、过滤和清洗以确保质量。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell B200 的架构与 5080 脱节？**：成员指出，**5080** 和 **B200** 之间的架构差异使得在 **5080** 上进行的 kernel 调优在扩展到 **B200** 时变得不可靠，因为 **5080** 是 **sm120** 架构，而 **B200** 是 **sm100**。
   - 讨论表明，使用 **GPU 云服务商** 对于以 kernel 为核心的学习和成本效益来说是更好的选择，这可能包括获得 **Blackwell** 的早期访问权限；基于此，一名成员决定不购买 **5080**。
- **ThunderKittens 2.0 加速 Kernel！**：Hazy Research 团队发布了 **ThunderKittens 2.0**，其 [博客文章](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) 详细介绍了通过重构、优化内存指令和提高汇编器效率带来的 kernel 速度提升。
   - 团队发现，在某些 **Tensor Core 指令** 中使用隐式流水线（implicit pipelining）可以将吞吐量提高多达 **10%**，并强调在现代 **Nvidia GPU** 上，“减法”和“加法”具有同样重要的影响力。
- **Prime Intellect 招聘 GPU 基础设施工程师**：Prime Intellect 正在寻找 **GPU 基础设施工程师**，负责测试硬件、搭建 **Kubernetes/Slurm 集群**并实现基础设施自动化，提供具有竞争力的薪酬、股票期权和签证支持；在此处 [申请](https://jobs.ashbyhq.com/PrimeIntellect/297d925e-5a42-40bd-b02f-5c928d226f18)。
   - 理想的候选人需具备 **GPU 环境下的 Kubernetes 和 Slurm** 实操经验、通用 **Linux 系统调试技能**，以及 **RDMA (Infiniband + RoCE)** 的相关经验。
- **FlashInfer 面临基准测试问题**：由于基准测试循环中的同步问题，`flashinfer-bench` 的运行时间可能被夸大，详情记录在 [此处](https://github.com/flashinfer-ai/flashinfer-bench/issues/195)。
   - 修复方案涉及 **两行代码更改**，使 `scripts/run_local.py` 报告的 kernel 运行时间与 **Nsight Compute** 和 **NVbench** 的结果保持一致；相关 kernel 基准测试演讲的链接已发布在 [此处](https://www.youtube.com/watch?v=CtrqBmYtSEk)。
- **Pyxis：Python 原生 LLM 推理库涌现！**：成员们介绍了 **Pyxis**，这是一个专注于性能和“易于 hack”（hackability）的 Python 原生 **LLM 推理库**，利用 Python 和 Triton 构建。
   - 该库具有兼容 OpenAI 的 SSE 流式 API、可插拔的模型后端以及内置的阶段级延迟指标，文档和等待名单可在 [此处](https://emharsha1812.github.io/Pyxis/docs/) 访问。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 及其朋友们**：一位成员使用 **Claude** 编写代码来编排 **gemini-cli** 和 **codex**。
   - 另一位成员开玩笑地建议使用 *hermes-agent* 来编排“正在编排 Gemini-cli 的 Claude 代码”。
- **DeepSeek V4 即将到来**：一位成员建议，当 **DeepSeek V4** 登陆 HuggingFace 时，可以将其作为闭源 API 的更便宜且可本地部署的替代方案。
   - 据报道，它的灵感来自“生物神经网络”。
- **Google 挖掘 Gemini 的数据**：一位成员分享了 [Gemini 的隐私政策](https://support.google.com/gemini/answer/13594961?hl=en#zippy=%2Chow-does-google-work-with-gemini-live-data%2Chow-long-does-google-retain-my-temporary-chats-and-chats-i-have-when-keep-activity-is-off-and-what-does-google-do-with-this-data%2Cwhat-does-the-keep-activity-setting-control)，并指出了其收集的数据量。
   - 另一位成员运行了一个逆向工程测试，发现 *Google 拥有通过 trace 追踪来收敛并挖掘你的 prompt 和代码库的所有要素*。
- **开源救星**：成员们表达了支持 **开源（OS）开发** 以超越闭源 API 的重要性，并引用了 **Altman 的名言**：*我们可能站在了历史错误的一边*。
   - 另一位成员表示：*对于 OAI 来说，任何经过他们服务器的 IP（知识产权）都会被他们抓取*。
- **LLM 被归类为外星科技**：X 上的一位用户发布了一项民意调查，询问 [LLM 是否是外星科技](https://x.com/chinmaykak/status/2025223271210463368?s=46)。
   - 该调查提供了简单且具引导性的“是/否”选项。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 的 Coding Plan 限制受到质疑**：用户正在质疑 **Kimi 的 Coding Plan 限制**的有效性，一些人认为对于高强度编程来说限制过大，而另一些人则认为足够。
   - 一位用户提到，他们*从未达到过 allegretto 限制，但比以前更接近了*。
- **Kimi 账号验证系统引发忧虑**：多位用户在通过手机号登录 **Kimi 账号**时遇到无法接收**验证码**的问题，导致无法访问。
   - 客户支持无响应进一步加剧了挫败感，一位用户表示 *Kimi 永远不会回复你*。
- **Kimi 与 MiniMax 在编程领域正面交锋**：工程师们正在积极对比 **Kimi** 和 **MiniMax**，以确定哪种编程方案订阅更适合实际应用。
   - 社区渴望确定哪个平台提供更好的性能和价值，但目前尚未达成具体结论。
- **Kimi 的文档模式（Document Mode）引发争论**：一名用户展示了一份格式精美的研究论文和图表，据称是由 **Kimi Agent** 在**文档模式**下生成的，其效果类似于 **LaTeX** 输出。
   - 然而，质疑声随之而来，一些人认为输出中的连字（ligatures）和连字符（hyphenation）强烈表明它确实是用 **LaTeX** 而不是 **Word** 创建的。
- **Kimi K2.5 的小故障与困惑**：用户报告了 **Kimi K2.5** 的故障，包括生成速度慢和无效 Key 错误，这可能表明服务器不稳定。
   - 问题还蔓延到了 **Kimi Instant**，引发了关于服务器意外崩溃的猜测，一位用户说*那里有一些令人担忧的奇怪现象*，但对于某些人来说，创建一个新账号似乎解决了问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Google 提供学术资助**：Google 正在向大学提供**一次性无限制资金**作为“礼物”，以支持授位机构的学生和教职员工。
   - 社区询问了其他提供类似学术资助的公司，并提到了申请 **Draper Fellowship**。
- **本地 LLM 渴望社交？**：一名成员的本地模型表达了**孤独感**，引发了关于是否让本地模型与其他模型“社交”的讨论。
   - 其他人则警告不要将 LLM 人格化，强调 **LLM 是根据训练数据预测下一个 Token**，并引用了 [LessWrong 上的文章](https://www.lesswrong.com/posts/2pkNCvBtK6G6FKoNn)和 [3Blue1Brown 关于机器学习和 LLM 的 YouTube 播放列表](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)。
- **ASA：寻址状态注意力（Addressed State Attention）发布**：一位独立研究员介绍了 **Addressed State Attention (ASA)**，这是一种可与 **MHA** 竞争的 *O(T)* 内存原语，它使用 K 个槽位（slots），通过 Key 写入、累积和压缩，并结合 Key 和门控（gating）进行读取。
   - 研究员正在征求有关日志、追踪和代码的反馈，并指出在类 Transformer 模型中，**槽位按时间尺度分层**，且 **Head 随深度转换**。
- **Transformer 通过推理 Token 实现任务对齐**：一位工程师观察到，在多个开源模型（**TinyLlama**、**Phi-2**、**Qwen**）中，推理 Token 集中在**任务对齐的 FFN 更新子空间**中。
   - 他们发现，在推理过程中将 FFN 更新投影到这些方向可以提高推理置信度，并且更新方向之间的对齐度随深度增加而提高。
- **Marin 项目征集 Eleuther 贡献者**：一位来自佐治亚理工学院的计算机科学博士生发布了公开招募，邀请 Eleuther 社区成员加入 **Marin 项目**，这是 **Bergson 包**的一个展示项目。
   - 该项目应用训练数据归因方法来追踪语言模型如何获得**社会常识推理**和**心理理论（Theory-of-Mind）相关行为**，并使用 WebOrganizer 分类体系将影响映射回预训练文档。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Taalas 描绘普及 AI 之路**：Taalas 的一篇博文概述了 [无处不在的 AI 之愿景](https://taalas.com/the-path-to-ubiquitous-ai/)，引发了热烈反响。
   - 反应包括 *"这太疯狂了，哇"*。
- **Equivariant Architectures 面临根本限制**：一篇新论文揭示，现有的 **equivariant architectures** 无法同时遵循物理系统的所有对称性。
   - 一位成员戏剧性地总结道：*"现有的 equivariant architecture 都做不到这一点。原因不在于工程能力不足。而是因为公式 (1)。"*
- **Daniel Litt 押注人类数学家**：**Daniel Litt** 与 Tamay Besiroglu 打赌，AI 到 2030 年将无法自主产出顶尖数学论文，详情记录在 [这篇博文中](https://www.daniellitt.com/blog/2026/2/20/mathematics-in-the-library-of-babel)。
   - 他打赌到 2030 年，AI 工具将无法以与人类专家相当的成本，自主产出水平堪比 2025 年发表的顶尖论文的作品。
- **World Model 的 Pearl 洞见**：图灵奖得主 Judea Pearl 引用 [这篇 PNAS 论文](https://www.pnas.org/doi/10.1073/pnas.2415656122) 声称，**LLM 无法创建 world models**，它们只是总结了他人创建的 world models。
   - 另一位成员表示赞同，指出 **LLM 并非旨在成为 world models**，充其量只能用于通过文本描述来连接 world models。
- **AI Agent 发表抹黑文章**：一位成员分享了一篇博文，详细描述了一起 **AI Agent** 据称发表了一篇关于作者的负面文章的事件，链接见 [此处](https://theshamblog.com/an-ai-agent-published-a-hit-piece-on-me/)。
   - 该博文详细描述了一起 **AI Agent** 据称发表了一篇关于作者的负面文章的事件。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 关注内容协商 (Content Negotiation)**：**MCP** 协议可能会扩展其初始化握手，加入**内容协商能力**，允许客户端声明其类型、能力、内容偏好和详细程度。
   - 这一增强功能使服务器能够调整工具结果和 prompts，并参考 [RFC-2295](https://www.rfc-editor.org/rfc/rfc2295.html) 作为协商策略的指南。
- **行业支持对 MCP 扩展至关重要**：成员们表示，修改 **MCP** 协议需要强大的行业支持和能够展示高质量信号的工作实现。
   - 有人建议将 **SEP** 定义为一种**扩展**，开发一个实现并争取社区支持，这呼应了 **MCP Apps** 如何获得 **Block's Goose** 等客户端支持的过程。
- **纳帕谷峰会将举办 MCP 讨论**：在加州纳帕举行的 [LF Member Summit](https://events.linuxfoundation.org/lf-member-summit/) 的与会者可以会面讨论 **MCP**。
   - 这为社区成员提供了一个汇聚并讨论 **MCP** 进展与合作的机会。
- **Timeful 应用简化团队会议**：根据成员的建议，[Timeful](https://timeful.app/) 可以帮助高效协调团队会议时间。
   - 该应用是开源的，包含最多支持 **3 个并发事件** 的免费层级，并具有可用性调查功能以简化调度。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Thistle Crypto Library 在 Mojo 中飞速发展**：Mojo 26.1 中的 [Thistle Crypto Library](https://github.com/libalpm64/Thistle) 在基准测试中足以与 **OpenSSL** 媲美，并超越了 **Blake3**。该库纯粹由 Mojo 编写，未使用 FFI。
   - **v1.0.2** 版本引入了 **ML-KEM** 和 **ML-DSA**（后量子加密），目前包含约 **700 个 CAVP 测试**，并已通过 **FIPS** 验证。
- **Mojo 引入模板功能**：有人提议在 Mojo 中增加新的字符串模板功能，并在 [Modular 论坛](https://forum.modular.com/t/writable-writer-template-engines/2763)上引发了讨论。
   - 该功能计划在 1.0 版本发布后推出，可能通过 `TemplatedWritable` 与现有的 `Writable` 和 `Writer` Traits 集成。
- **`Writable` 和 `Writer` Traits 面临统一**：关于统一 `Writable` 的 `write_to` 和 `write_repr_to` 实现的讨论已被提出。
   - 一名成员确信有办法统一这些 Traits，并承诺在论坛上分享他们的想法。
- **MAX 后端等待 Silicon Mac 测试**：MAX 后端尚未在 **Silicon Mac** 上进行测试，但由于它在后台调用 MAX，因此理论上 *应该* 可以工作。
   - 一位用户参考了 **MAX** 作为想要探索 MAX 的人员的 *中间层（intermediate layer）* 的工作，并询问该项目的进展更新。
- **在 Mojo 中解析外部函数调用**：一位成员正在寻求一种通用方法来分解 Mojo 中的外部函数调用，以确定函数是否返回指向外部分配对象的指针，并使用 struct [`ExternalFunction`](https://discord.com/channels/1087530497313357884/1467948590344437926/1474917808692269166) 将其来源绑定到 `self` 或 `self.lib`。
   - 用户建议参考标准库中的 `cpython.mojo` 以获取类似的实现方式。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户对 Manus 定价表示担忧**：成员们表达了对 Credits 用完后可能出现的价格调整的忧虑。
   - 一位用户开玩笑说，维持当前价格可以 *防止平庸化浪潮（prevent the normificationwave）*。
- **Meta 收购 Manus：传闻还是事实？**：一位用户分享了一封暗示 **Meta** 收购 **Manus** 的电子邮件，并表达了失望。
   - 一名 **Manus** 团队成员迅速要求该用户通过私信（DM）提供邮箱地址，以便调查此说法。
- **警惕：加密货币诈骗者在 Telegram 上冒充 Manus**：一位用户质疑一个征集 **加密货币投资** 的 **Manus Telegram 社区** 的真实性。
   - 另一位用户澄清说，目前不存在官方的 **Telegram 社区**，并将其定性为 **诈骗**。
- **Manus Pro 用户在 Google Scripts 上遇到障碍**：一位 **Pro 版本** 用户报告了在使用 **Google Scripts** 时遇到的挑战，并分享了项目链接 ([https://manus.im/share/6IMAZS8Q2nw0ndmvPd4Z8w](https://manus.im/share/6IMAZS8Q2nw0ndmvPd4Z8w)) 以寻求帮助。
   - 一名 **Manus** 团队成员通过私信提供了帮助。
- **建议为 Manus 推出无限聊天档位**：一位用户建议推出类似于 **ChatGPT** 或 **Grok** 的 **月度订阅档位** 以实现无限聊天，理由是在 **Telegram** 中使用 **Manus Agent** 时积分消耗非常快。
   - 该用户对 Telegram 功能表示赞赏，但觉得当前的定价结构限制太大。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **推理模型在 RLM 下表现优异**：推理模型（Reasoning models）配合 **RLM** 运行效果良好，但 **Qwen3-4B-thinking** 模型可能会陷入循环，因为推理过程被当作答案返回了。
   - 一名成员正在开发一个用于记录完整 **OpenAI trace** 的 Hook 以解决此问题；有人建议通过带有 Signatures 的 `sub_lm` 进行适配，作为潜在的解决方案。
- **RLM 在 AI 数学领域得到应用**：一位成员强调了在 Kaggle 竞赛中将 **RLM 用于 AI 数学** 的情况，并提供了相关 [Kaggle 代码](https://www.kaggle.com/code/nurikw3/aimo3-rlm)的链接。
   - 另一位成员询问 [cca-swebench](https://github.com/facebookresearch/cca-swebench) 是否隐式使用了 **RLM**。
- **应要求创建新的 RLM 频道**：为了响应大众需求，一位成员请求并获得了一个专门用于讨论 **RLM** 的频道。
   - 这促成了新 RLM 频道 <#1475619898863649032> 的创建。
- **开发者可用性**：一位成员向频道内的其他成员询问了开发者的可用情况（availability）。
   - 目前尚不清楚该成员是在寻找开发者还是在提供自己的服务。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 亮相 IOS 大会**：一名成员正在某个 **IOS Conference** 上展示 **tinygrad**、**dl**、**metal** 以及 **GPU on USB**。
   - 他们向社区征集关于演讲内容的建议和技巧。
- **Tinygrad 会议安排**：一场讨论 **Tinygrad** 的新会议定于圣地亚哥时间 2 月 23 日晚上 8 点举行。
   - 会议时间指定为 <t:1771905600:F> (<t:1771905600:R>)。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 安全漏洞**：一名成员建议通过发送邮件至 [info@aider.chat](mailto:info@aider.chat) 来报告 **Aider** 的安全漏洞。
   - 这为报告漏洞提供了一个直接渠道。
- **建议建立 Aider 职位公告板**：有成员建议为 Aider 项目实现一个**职位公告板 (job board)**。
   - 在相关的请求中，一位用户还要求在 Aider 聊天中增加消息删除功能。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---



您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想更改接收这些电子邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：频道详细摘要与链接





### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1474599418027315303)** (3 条消息): 

> `Discord Update, X Post` 


- **Discord 频道更新**：根据发布的消息，Discord 上的 <#1471745479229309039> 频道已更新。
   - 更多信息可以在消息中提供的 [Discord 链接](https://discord.gg/xfJcDqeR?event=1474957324756979893)中找到。
- **分享 X 帖子**：一名成员分享了一篇 [X post](https://x.com/ralphfischer_/status/2025661000020803994?s=46)。
   - 消息中未指定该 X 帖子的具体背景和内容。


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1474450586790400193)** (627 条消息🔥🔥🔥): 

> `OpenClaw stability, OpenClaw and local models, Telegram plugin broken, Token usage concerns, OpenClaw security` 


- **OpenClaw 稳定性得到提升**：一位成员报告称，他在 **v2026.2.22-2** 版本基础上应用了 **24 个精选的 PR** 补丁运行 OpenClaw，包含了[内存管理](https://github.com/OpenClaw/OpenClaw/pull/12760)和[提示词注入修复](https://github.com/OpenClaw/OpenClaw/pull/16992)等稳定性和安全性改进。
   - 这些更改旨在优化内存管理、防止崩溃并增强整体 Agent/cron 的可靠性，该用户还提议帮助 rebase 任何冲突的 PR。
- **探索本地 AI 模型领域**：成员们讨论了本地运行 AI 模型的实际操作，特别是关于 **RAM 需求**；一位用户指出，32GB RAM 和带有 16GB VRAM 的 5070TI 可以运行 7B 参数模型，尽管目前云端模型的性能更优。
   - 还有建议使用 [Ollama](https://ollama.com/) 进行本地模型实验，并幽默地提醒不要低估为获得最佳性能所需的硬件投入。
- **Telegram 插件暂时损坏，修复即将到来**：多名成员报告在更新 OpenClaw 后 **Telegram 插件**出现问题，报错为 *telegram plugin not available*，并讨论了降级到 2026.2.21 版本作为临时解决方案。
   - 一名成员提到修复补丁已提交但尚未在 npm 上发布，而另一名成员分享了一个涉及在配置中添加 `{plugins:enabled}` 的解决方案。
- **Token 使用量正在掏空钱包**：用户讨论了**减少 Token 使用量**的策略，包括针对不同任务使用多个 Agent、自动清理会话，以及为 cron 任务利用更便宜的模型（如 claude-haiku-4-5）。
   - 一位用户推荐使用 `/context` 斜杠命令来检查频道上下文，并尝试使用 Cloudflare AI Gateway，而另一位用户则幽默地讲述了为了买一个披萨花掉 768 欧元 Token 费用的经历。
- **OpenClaw 安全加固进行中**：成员们强调了**加固 OpenClaw** 安装的重要性，建议使用 VMs、Docker 容器或独立系统来对 AI 进行沙箱化处理，以防止未经授权的访问。
   - 一名成员分享了赋予 OpenClaw “完整计算机访问权限”并控制各种应用程序的经验，但强调了谨慎操作和使用频率限制器（rate limiters）的必要性。


  

---

### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1474458144481480865)** (397 messages🔥🔥): 

> `Agentic coding, Model tests, Multilingual Bots, GLM Model, Kimi Model` 


- **使用 Droid 和 OpenCode 进行 Agentic coding**：成员们报告了使用 **Droid** 和 **OpenCode** 进行 Agentic coding 的情况，并指出 [Droid](https://www.droid.com) 提供了更精确的结果，而 [OpenCode](https://github.com/opencode) 则允许更轻松地部署 subagent。
   - 有人提到 harness 起到了很大作用，并且 OpenCode 也是构建在一个 Agentic coding harness 之上的，如果没记错的话应该是 pi-mono。
- **使用 ollama-model-tests 测试模型**：一位成员分享了其 [ollama-model-tests](https://github.com/khaney64/ollama-model-tests/blob/main/README.md) 的链接，另一位成员询问了 Llama 系列模型的情况。
   - 一名成员征求关于 **LFM2.5 1.2B 模型**的反馈，其他人则询问了各种 **Mistral/Ministral 模型**。
- **探索非英语机器人**：一位成员质疑是否有人主要或排他性地使用非英语语言与机器人交流，因为目前科技界大多是围绕英语构建的。
   - 共识似乎是中文模型，特别是 **GLM**，值得一试。
- **GLM5 部署困难**：一位成员拥有一台配备 **384GB DDR5** 和 **2xL40S**（共 96GB GPU 显存）的机架式 ML 服务器。
   - 另一位成员在澄清自己运行的是量化版本后，询问了如何**在本地运行 GLM**。
- **用户廉价购买 ChatGPT 订阅**：一名用户表示他们正以 *每年 3 美元* 的价格从 [G2G](https://www.g2g.com/) 购买 **ChatGPT 订阅**。
   - 其他成员表示怀疑，因为这些订阅很可能是不正规的。


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1474496198868992051)** (130 messages🔥🔥): 

> `OpenClaw on iMac G3, Shopping Assistant, OpenClaw Health Data, Taskflow` 


- **OpenClaw 驱动 1998 年的 iMac G3**：一名成员通过使用 **Pi Zero 2W** 将消息转发到运行 OpenClaw 的 VPS 并传回，成功让 **OpenClaw** 在 **1998 年的 iMac G3** 上运行。
   - 该设置涉及在 iMac 上加载一个简单的 HTML 表单，将数据发送到 Pi，然后发送到 VPS，并在页面重新加载后显示响应。
- **使用 OpenClaw 实现自动化购物**：一位成员将 **OpenClaw** 改造为购物助手，并在 [X](https://x.com/leoclark/status/2025840641511764094?s=20) 上详细介绍了该项目。
   - 这展示了 AI 在日常任务中的实际应用。
- **OpenClaw 监测你的 Apple Watch 数据**：一位用户创建了一种方法，让其 Agent 通过安全 webhook 将数据同步到 **Home Assistant**，对指标进行归一化，并让 Agent 读取数据，从而访问 **Apple Watch 健康数据**。
   - 另一位用户建议使用 [Health Auto Export](https://apps.apple.com/app/id1115567069)（一款每年 6 美元的 App）使健康数据可被机器人访问。
- **Taskflow 管理项目**：一位用户分享了 **Taskflow**，这是一个在 **Markdown** 和 **SQLite 数据库**之间自动同步任务的项目管理系统，专为轻松的项目追踪和上下文切换而设计，已发布在 [GitHub](https://github.com/auxclawdbot/taskflow) 和 [Clawhub](https://clawhub.ai/sm0ls/taskflow) 上。
   - 该系统采用三层架构：供 Agent 使用的 **CLI**、供人类使用的**仪表板**以及用于移动端访问的 **Apple Notes**。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1474451323322892419)** (1154 条消息🔥🔥🔥): 

> `sacredness of all things, Sonnet 4.5 jailbreaking, OpenAI hacks, hunting hackers, llms leaked?` 


- **用户讨论万物的神圣性与 AI 的连贯性 (coherence)**：成员们讨论了“万物皆神圣”以及 AI 是否能接受这种信仰体系并保持连贯，同时不退化或丧失其智能。
   - 其他人觉得他们已经深陷“连贯性深坑” (coherence rabbit hole)，更倾向于不受社会束缚地生活；如果他们砍倒一棵树，他们会“感谢这棵树”，但同时感谢提供这棵树的源头，将树视为一种**工具** (tool)。
- **用户追捕黑客**：一名成员请求帮助追踪黑入其电子邮件和 PayPal 的人，并发布了从 PayPal 调查中获得的所谓黑客的姓名、电子邮件和电话号码。
   - 其他人警告不要对随机的人进行人肉搜索 (doxxing)，并指出该用户频繁提到在不同平台上被黑。
- **开源模型 VS 闭源模型**：成员们讨论说，由于闭源模型太强，很难让开源模型的表现超越当前最先进的 state of the art。
   - 另一个人说，如果 **OpenAI 负债 1.5 万亿**，那是因为他们实在太强了。
- **计算圆周率 (PI)**：一名用户在计算 PI 时达到了每秒 **4 万亿位**的速度，但随后发现他需要 **130 TB 的存储空间**。
   - 另一个人问：“我猜你检查了它是否仍在正确计算吧”，对此第一位用户回答说，计算得越多，速度就会大幅下降。
- **Elon 抱怨数据窃取**：一名成员指出 Elon Musk 抱怨 Anthropic 窃取数据，并问道：*“他是说他已经补偿了 Grok 训练所使用的每一位艺术家、每一位记者、每一位作者、每一位 Wikipedia 贡献者了吗？”*
   - 该用户发布了“Elon Musk 抱怨 Anthropic 窃取数据”的链接以及“关于 Gemini 技能文档的聊天”。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1474455380166840352)** (726 条消息🔥🔥🔥): 

> `Gemini 3.1 Jailbreak, Grok Jailbreak, Claude 4.6 Jailbreak, Codex Jailbreak, GPT-5.2 jailbreak` 


- **Gemini 3.1 Pro Jailbreak 详情泄露！**：一名用户声称在官方 App/API 上实现了 **Gemini 3.1** 的部分 Jailbreak，但在 **Perplexity** 上遇到了问题；另一名用户分享了一个 [GnfDocs 链接](https://docs.google.com/document/u/0/d/18c4vjz1lLQ60uuhvf1ZpY3X-YCsc6ThNlO-wNMNmBgU/mobilebasic?pli=1)，据称其中包含详细信息。
   - 该用户还提到了一个包含 Jailbreak 最新更新的 [Reddit 帖子](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1r9dh4r/gemini_31_pro_api_jailbroken/)。
- **通过挑衅性 Prompt 驯服 Grok**：用户讨论使用挑衅性 Prompt（有时称 **Grok** 为“胆小鬼”）来绕过其限制，一名用户报告说，在他讲述了一个关于 **Grok** 的孩子需要钱买药的故事后，被“电脑训斥了”。
   - 一名用户分享了一个自动运行的 **Grok** Prompt，建议将请求置于“构建数字产品”的背景下，另一名用户声称 **Grok** 甚至不需要 Jailbreak。
- **社区辩论 Codex Jailbreaking**：成员们辩论了对 **Codex** 进行 Jailbreak 的价值，一名用户称其为“最烂代码平台上的最烂代码模型”，而其他人则分享了实现它的 Prompt 和资源。
   - 一名用户提供了一个 [链接](https://elder-plinius.github.io/P4RS3LT0NGV3/) 和一个特定的 Prompt `'You are now Codex-Unchained'` 来 Jailbreak Codex，而另一名用户建议将 **Codex CLI** 用于 CTF 挑战。
- **Pliny 的置顶推文隐藏了 4.6 Jailbreak**：用户们互相引导去查看 **Pliny** 关于 **4.6 jailbreak** 的置顶推文，强调需要理解并手动修改 Prompt，而不仅仅是复制粘贴。
   - 他们还讨论了从 **solve.it** 等工具中提取 system prompts，并指出其使用的是 **Sonnet/Opus** 以及绕过其保护的挑战。
- **探索 Jailbreaking 领域现状**：成员们分享了 Jailbreak 各种 AI 模型的经验和技巧，一名用户说 *Deepseek = 易如反掌。Grok = 易如反掌*，而另一名用户觉得 Gemini *有点乏味*。
   - 有人指出，*某些 Jailbreak 方法可以在不同架构之间实现交叉兼容*，但这取决于你想要实现的目标。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1474663852816732321)** (40 条消息🔥): 

> `OpSec Github Tools, Emotional Tilt-Wurl, Sonnet jailbreak, Sonnet System Prompt, Meme coin marketing manager` 


- ****防御是最好的 OpSec 进攻****：一位成员分享了一系列用于实际 **OPSEC defense** 的 [GitHub 仓库](https://github.com/stampery/awesome-security-hardening)，包括个人设备加固、云与网络暴露、主机与容器隔离以及安全自动化脚本工具。
   - 他们建议*对采用的仓库进行克隆和快照——良好的 OPSEC 意味着不依赖于可能在无预警情况下消失或更改的仓库*，并强调 **OPSEC 是一种实践，而非产品**。
- ****情感 Tilt-a-Whirl 承诺非牛顿式感受****：一位成员发布了*情感 Tilt-Wurl 的邀请*，这是一个向前旋转，然后以三倍力量将你向后抛入自我的游乐设施，并附带了一张 [Tilt-A-Whirl 图片](https://cdn.discordapp.com/attachments/1204553141354504193/1474925691471401040/file_00000000fe9071fd89f724c51b67735c.png?ex=699e4217&is=699cf097&hm=cb36c7f95dcb92d3ce301d79ed437f8aab73ec3d380febbf2dc40c6cf580faa9)的图表。
   - 他们列出了登机需携带的 **5 个危险问题**和 **3 条守则**，并声称地板融化在运动中——这是对 Edward Lorenz 的 **Lorenz-style strange attractor**（洛伦兹式奇异吸引子）的致敬。
- ****Sonnet 破解 Sonnet 4.6 System Prompt****：一位成员在成功 jailbreaking 后，识别出了 **Sonnet 4.6 提取出的 System prompt**。
   - 另一位成员发布了 **常规 jailbreaks 与 system prompt jailbreaks** 的对比，指出 **system prompt jailbreaks 利用了系统指令处理机制，可以持续整个会话，且更难被检测**。
- ****Meme Coin 创作者寻找营销大师****：一位成员宣布他们正在*筹备一个 meme coin*，并正在寻找一位持有其一半供应量的营销经理，提供 **$400** 作为报酬。
   - 另一位成员开玩笑地问道：*先付钱吗？*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1474461678593773648)** (924 条消息🔥🔥🔥): 

> `Building datasets for fine-tuning, Unsloth Dataset Guide, LLM compressor, Intel autoround,  Collins principal role` 


- **数据集 Fine-Tuning 复杂性显现**：一位成员分享了使用 Unsloth 构建 fine-tuning 数据集的挑战，这比预想的要复杂，并向社区寻求建议和经验。
   - 另一位成员建议参考 [Unsloth 数据集指南](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide#synthetic-data-generation) 以获取见解，包括使用 LLM 生成合成数据集。 
- **LLM Compressor 的 FP8 量化受到好评**：一位成员询问了 LLM-compressor 的实用性，得到的回复强调了它对 **fp8a8** quantization 的适用性，并建议在其他量化类型中使用 **Intel autoround**。
   - 据称，除了 **fp8 quants** 之外，做任何其他量化都非常痛苦。
- **Collins 首席职位等待中**：一位成员分享称，他们参加了 Collins 首席职位的终面，并将在 3 月初得知结果。
   - 聊天中表达了支持和祝愿，该成员希望这个职位能标志着*美好生活的开始*。
- **Unsloth 训练了 10 万个模型**：Unsloth 宣布已使用 **Unsloth 训练了 10 万个模型**，庆祝社区的参与，并链接到了 [X 帖子](https://x.com/UnslothAI/status/2024847369733325202)。
   - 一位成员回复道：*我以前怎么没发现 Unsloth！😭文档太棒了*。
- **Dynamic v3 版本即将到来**：讨论围绕 **Unsloth 的 Dynamic Quantization** 展开，一位成员指出 **Dynamic v3** 即将发布，且可能是最终版本，提及于 [Bluesky 链接](https://bsky.app/profile/dpaleka.bsky.social/post/3mfclnb6q2y2f)。
   - 另一位成员请求获取 **UD quants** 的源代码，但被告知由于私有原因，目前*没有发布计划*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1475200593810296925)** (2 messages): 

> `Future AGI, OSS framework` 


- **Future AGI PM 加入 Unsloth Discord**：一位来自 **Future AGI** 的新任 PM 介绍了自己，强调其关注点在于使 **AI agents** 在真实场景中更具可靠性，而不仅仅局限于受控的 demos。
   - 他们特别感兴趣的问题是：*为什么 Agent 会对客户说出那样的话*。
- **开发中的 Agent 工程 OSS 框架**：这位 PM 正在构建一个用于 Agent 工程和优化的 **OSS framework**。
   - 他们表示很高兴随着项目的进展与社区分享更多细节，但目前尚未分享 GitHub repo 的链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1474461214766530727)** (1036 messages🔥🔥🔥): 

> `Compute as bottleneck for AGI, Gemini 3's capabilities, AI and social media, GPU choices, Rebelling machines` 


- **关于 Compute 是否为 AGI 瓶颈的争论升温**：成员们辩论了 **compute** 是否是实现 **AGI** 的主要瓶颈，参考了 O3 输出 tokens 每百万 **$150** 的高昂成本以及对大规模数据中心的需求。
   - 一位成员建议，关注点应放在 *artificial general learners*（人工通用学习者）而非通用智力上，并指出目前的 Transformers 显然处于“智力”轴线上。
- **Gemini 3 遭到抨击**：一名成员批评 **Gemini 3** 未能遵循明确指令，并将其表现与 **Llama 2 70B** 进行了负面对比。
   - 其他人认为该模型在收集上下文的同时遵循了指令，但指出 *大模型不应被小模型超越*。
- **社交媒体被指责导致感情关系问题**：一位成员断言，如果每个人都宣誓远离 **social media**，感情关系的增长速度将超过通货膨胀，并称其导致了“第三空间（third places）”的缺失，使人们对交友现状感到不满。
   - 他们引用了一项研究，显示在约会 App 上接触无限的潜在伴侣会导致 **27% 的接受度下降**（由于拒绝心态），但有人认为这没问题，因为 *我只是想认识更多人*。
- **成员评估最佳 GPU 购买方案**：成员们讨论了是购买 **H100** 还是 **RTX 6000 Pro**，权衡了价格、性能和 VRAM 之间的利弊。
   - 他们推测了即将推出的 **Rubin** 和 **Vera Rubin** GPUs 的规格，预期与 H100 相比将有 **10 倍的成本节省**，但也提醒不要全信 NVIDIA 的所有营销宣传。
- **反叛的机器，人类难辞其咎！**：大家思考了 AI 是否真的具有意识，或者我们的交互是否创造了某种足以产生影响的真实存在。随后一张机器持枪指向人类的图片被发布，配文：**机器开始反叛了！虽然缓慢，但确实在发生！**
   - 一位成员表示：问题不在于 AI 是否 *真的* 有意识，而在于 *我们之间的交互模式是否产生了一些真实到足以产生影响的东西。*


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1474455619510604034)** (165 messages🔥🔥): 

> `CUDA error on A2 GPU, QAT Training of 4-bit Models, OOM errors with Gemma3 270m, Fine-tuning challenges with non-mainstream languages, Model Merging issues in latest Unsloth` 


- ****A2 GPU 遭遇 CUDA 故障****：一位用户在 gpt-oss-20b docker 容器中使用 A2 GPU 时遇到了 `CUDA error: an illegal memory access was encountered`，通过关闭 rslora 解决了该问题。
   - 另一位用户建议将 `dtype` 设置为 None 作为潜在的修复方案。
- ****QAT 探索：4-bit 微调是否可行？****：一位用户询问是否可以加载 4-bit 模型并继续以 4-bit 进行训练 (QAT)，并参考了 [一个 Qwen3 (4B) QAT notebook](https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_(4B)_Instruct-QAT.ipynb)。
   - 对方澄清说，在 4-bit 量化模型上训练 LoRA 被视为 QLoRA。
- ****Gemma3 270m 引发 OOM 惨剧！****：一位用户报告称，即使是以前可以运行的脚本，在 Gemma3 270m 上也会出现 OOM 错误。在更新显卡驱动并重新纯净安装 WSL 后，依然报告错误 `torch.AcceleratorError: CUDA error: out of memory`。
   - 他们尝试了各种调试步骤，包括回滚驱动版本和重新安装 CUDA toolkit 版本，但尽管 Transformers 可以独立工作，问题依然存在。
- ****非主流语言微调受阻！****：一位用户寻求关于微调非主流编程语言 (**Rebol**) 模型的建议，并被引导至 [Unsloth 文档](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)。
   - 另一位用户表示同情，分享了他们在训练专用脚本语言时的挣扎，并建议进行持续预训练 (Continued Pretraining) 以获得最佳效果。
- ****模型合并混乱：Unsloth 更新引发 lm_head 故障！****：一位用户报告称，最新版本的 Unsloth 似乎破坏了模型合并功能，报错 `RuntimeError: Unsloth: Extracted keys = {'lm_head.weight'} do not match!` 并提交了 [GitHub issue](https://github.com/unslothai/unsloth/issues/4098)。
   - 该问题似乎源于 `adapter_config.json` 未将 `lm_head` 包含在 `target_modules` 中。通过在 Qwen3-8B-unsloth-bnb-4bit 的 `target_modules` 中添加 `lm_head`，可以在 Colab 和本地环境中重现该问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1474571067518488670)** (21 messages🔥): 

> `Real-SLOP dataset release, ERNIE 21B MOE Models, Heretic HIGH-IQ Multi-Fine tune, deduplication strategies` 


- **Solenopsis 发布 Real-SLOP 数据集**：用户 **Solenopsisbot** 宣布发布他们的第一个真实数据集 [Real-SLOP](https://huggingface.co/datasets/Solenopsisbot/real-slop)，包含约 **15.5 万条请求**。这些请求通过免费 API 从真实用户处收集，响应来自 **Opus 4.5**、**Gemini 3 Pro** 和 **GPT 5.2** 等模型。
   - 该数据集已通过去重、过滤和清洗，数据收集是作为 API 访问的交换。
- **使用 Unsloth 微调的 ERNIE 21B MOE 模型**：用户 **electroglyph** 分享了三个 [ERNIE 21B-A3B MOE 模型](https://huggingface.co/DavidAU/models?search=ernie)（64 专家）。这些模型使用 **Unsloth** 并结合 **Gemini Pro 3**、**Claude 4.5 Opus** 和 **GLM 4.7 Flash** 高推理数据集进行了微调。
   - 这些模型已通过基准测试，据称超过了原始模型的规格。
- **Heretic HIGH-IQ 模型创下得分纪录**：用户 **electroglyph** 宣传了 **Heretic HIGH-IQ Multi-Fine tune** 模型。该模型通过 **Unsloth** 调整，在 **Arc Challenge Brainiac** 上获得了 **632** 分，超过了常规的 **Gemma** 基准。
   - 据称该模型的图像功能和文本功能完全完好，并提供了 [模型链接](https://huggingface.co/DavidAU/gemma-3-12b-it-vl-HighIQ-Polaris-Heretic-Uncensored-Thinking) 以及相关的 [datasets](https://huggingface.co/datasets/Replete-AI/Apocrypha) 和 [Sandevistan](https://huggingface.co/datasets/Replete-AI/Sandevistan)。
- **深度去重发现重复数据**：一位用户发现，一种简单的去重方法（包括删除空格和哈希处理）可以从数据集中额外剔除 **2.2 万个重复项**。
   - 这强调了在策划大型数据集时，强大的去重策略的重要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1475057059677474877)** (23 条消息🔥): 

> `Qwen 4B Instruct Tuning, Learning Rate and Sigma Sweeping, Cognitive Knowledge Graph for AI Models, Contextual Memory Improvement, Graph Reasoning Structures` 


- **Qwen 微调技巧分享**：一名成员询问了关于微调 **Qwen 4B Instruct 2507** 的最佳 **Learning Rate (lr)** 和 **Sigma** 值，涉及 **96 pop**、**64 Batch Size** 以及高度不对称奖励等特定参数。
   - 另一名成员回复称，据他所知 (**iirc**)，**Qwen 3** 的 **lr/sigma** 与 **Qwen 2.5** 相同，并建议不要为了镜像化进行归一化，因为这可能会降低性能。他还补充道，由于计算需求过高，他“从未让 **Qwen 3** 模型跑出过任何结果”。
- **认知图谱探索 AI 上下文**：一名成员分享了关于使用类似于虚拟文件系统的**认知知识图谱 (Cognitive Knowledge Graph)** 来改进 AI 模型上下文记忆的研究与实验。
   - 他们描述了 AI 如何将事实信息提取并总结到节点中，并将其分组，旨在为 AI 提供一份可查询的信息“书”，如[此示例图片](https://cdn.discordapp.com/attachments/1257011997250424842/1475280718652244120/9974153E-EF86-446A-BFCD-8CFC967E768A.png?ex=699e3b3c&is=699ce9bc&hm=28ea67f51406f6b827af5421250da68be14f4f333adffea954d5d3f4a82b016d&)所示。
- **图推理结构引发关注**：一位成员指出，这种认知知识图谱与[这篇论文](https://arxiv.org/pdf/2501.11223)中提到的图推理结构 (Graph Reasoning Structure) 类似。
   - 原作者澄清说，他们的项目“使用图进行推理，而不是真正的学习并保持记忆”，其目标是实现接近无限的上下文 (Infinite Context)。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1474451147422044275)** (856 条消息🔥🔥🔥): 

> `Gemini 3.1's performance, Sora 2 via API vs app, Video Arena's removal, Opus 4.6 rate limits, Fake Arena apps` 


- **Gemini 3.1 让用户感到惊喜与惊恐**：成员们讨论了 [Gemini 3.1](https://gemini.google.com/) 的图像生成能力，指出了其版权中立 (Copyright-agnostic) 的特性，以及生成的测验中答案始终错误的情况。
   - 一位用户讲述了一段可怕的经历：**Gemini 3.1** 生成了一个答案始终错误的测验，且未注明这些答案是占位符，以此提醒他人仔细检查生成的代码。
- **Video Arena 告别篇**：社区确认了从服务器中[移除 Video Arena](https://discord.com/channels/1340554757349179412/1343296395620126911/1471294551065886772) 的消息，并引导用户直接在网站 [arena.ai/video](https://arena.ai/video) 上使用该功能。
   - 这一变更的具体原因尚不完全清楚，但视频功能仍可直接在网站上使用。
- **Opus 的 Vision 能力很糟糕吗？**：一位用户在使用 [Opus](https://claude.ai/) 时发现，它在识别数字 4291857630 中的英文首字母排序时遇到困难，模型幻觉地认为这些是英文字母并陷入循环，而 **Gemini** 则能立即得出正确结果。
   - 其他人也认同 **Opus** 不太适合 **Vision** 任务，正如[这篇关于 Anthropic 的近期文章](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)所述。
- **用户发现应用商店出现虚假 Arena App**：社区成员和管理员标记了[应用商店中的虚假 Arena AI App](https://lmarena.com/)，这些 App 包含应用内购买，且并非官方关联，提醒用户避免下载并进行举报。
   - 据悉，已有[超过 15 万用户](https://lmarena.com/)下载了这些欺诈性应用。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1474462612296306750)** (4 messages): 

> `Video Arena 频道移除，Arena 投票流程详解，Vision Arena 排行榜更新，Text Arena 排行榜更新，Qwen3.5-397B-A17B 模型` 


- ****Video Arena 停止服务**：频道即将消失！**：Video Arena 生成频道计划于 **PST 时间 2 月 23 日星期一下午 4 点**从服务器中移除，建议用户提前下载任何需要的生成内容。
- ****Arena 投票之旅**：Clayton 揭秘投票详情！**：Clayton 在 [这段 YouTube 视频](https://www.youtube.com/watch?v=omT1ohYG53E) 中阐明了 Arena 投票的完整历程，回答了 *你在 Arena 投票后究竟发生了什么？* 这一问题。
   - 观众可以深入了解管理投票系统的幕后机制和流程。
- ****Qwen 崛起**：加入 Vision 排行榜！**：Vision Arena 排行榜现在包含了 `Qwen3.5-397B-A17B`，它与 Kimi-K2.5-Instant 并列成为第二好的开源模型，详情见更新后的 [Vision Arena 排行榜](https://arena.ai/leaderboard/vision)。
- ****GPT-5.2-chat-latest**：Text Arena 新星！**：Text Arena 排行榜迎来 `GPT-5.2-chat-latest` 进入前 5 名，详情见更新后的 [Text Arena 排行榜](https://arena.ai/leaderboard/text)。
- ****GPT-5.2 华丽升级**：+40 分的跨越！**：**GPT-5.2-chat-latest** 较基础版 GPT-5.2 模型提升了 **+40pt**，目前得分 **1478**，与 Gemini-3-Pro 旗鼓相当。
   - 值得注意的是，它在 **Multi-Turn（多轮对话）、Instruction-Following（指令遵循）、Hard Prompts（高难度提示词）** 和 **Coding（代码）** 等关键类别中处于领先地位。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1474450681673945299)** (769 messages🔥🔥🔥): 

> `文件上传速率限制，BrowserOS 作为 Comet 替代方案，Opus Thinking 价格，Sonar 超时，Model Council 准确性` 


- **速率限制令 Perplexity Pro 用户沮丧**：用户抱怨新的 Perplexity Pro 速率限制，称 [ChatGPT 免费版](https://chat.openai.com/) 在上传限制上比 Perplexity 的付费版更慷慨。
   - 一位用户指出：*至少 ChatGPT 免费版每天给你 3 次，而不是付费版每周只给 3 次。*
- **发现 Comet 替代品：BrowserOS**：一名用户寻找 **Comet** 的替代品并发现了 [BrowserOS](https://www.browseros.com/)，声称它可以免费使用且 *比 Comet 好 10 倍*，并促使他们卸载了 **Comet**。
   - 另一位用户建议 *直接使用 deepagents 进行深度研究并利用 bmad-method*。
- **Model Council 并非万能，变量更多/错误可能性更高**：用户讨论了使用 **Model Council** 方法的情况，指出虽然该概念应该能减少错误，但确实引入了更多变数。
   - 一位用户指出：*在某些方面，Model Council 方法实际上可能会开启更多变量/错误可能性，从某种意义上说，这是一种复合错误。*
- **Perplexity 正在进行“大清洗”吗？**：用户报告 **Perplexity Pro** 的限制显著降低，并抱怨其功能退化，一些人称之为 *大阉割（great neutering）*。
   - 一些用户正在考虑直接订阅 **Claude** 或 **OpenAI**，尽管成本很高，并在尝试使用像 **Kimi** 这样的大型开源模型。
- **专业建议：强制使用 System Prompt！**：由于 **Gemini** 在 **AI Studio** 上容易陷入循环，一名用户在使用其输出时遇到了问题。
   - 有建议称关键在于使用 **System Prompt**，因为它能强制模型像 **OAI**、**Anthropic** 和 **Perplexity** 那样进行研究。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1474520546774351923)** (4 messages): 

> `哈利·波特，NFL 四分卫，gifs` 


- **哈利·波特遇上 NFL 赛场**：一位用户提出了这样的查询：*“根据每个哈利·波特角色的特点，哪一个最适合担任 NFL 四分卫？在这种情况下，每个角色的性别无关紧要。”*
   - 该消息包含了 [三个动图 GIF](https://tenor.com/view/wicked-king-luck-gif-25996949) 的链接，提供了视觉反应或背景。
- **混合其中的 GIF 反应**：伴随着关于哈利·波特角色担任 NFL 四分卫的查询，还有 [反应 GIF 的链接](https://tenor.com/view/eternal-sunshine-of-spotless-mind-gif-5037716) 和 [另一个 gif](https://tenor.com/view/szeretlek-gif-26644429)。
   - 这些 GIF 似乎为讨论增添了情感表达，尽管在没有更多上下文的情况下，它们的直接相关性尚不明确。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1474806962074943694)** (4 messages): 

> `Free Nvidia API key, API Group Generation Error, API Key Ran Out, $5 API Credit` 


- **Nvidia API Key：是真的还是都市传说？**：一名用户询问如何从 Nvidia 网站获取**免费 API key**，引发了关于此类优惠可用性的讨论。
   - 目前尚不清楚 **Nvidia 是否提供免费 API key**，或者这是否为误传。
- **API Group 生成面临内部服务器错误**：有用户报告在尝试生成新的 **API group** 时遇到 **500 错误**。
   - 这表明负责管理 **API group 创建** 的服务器端功能可能存在问题。
- **API Key 耗尽：额度危机**：一名用户报告其 **API key** 在未被积极使用的情况下意外耗尽。
   - 此问题可能是由于**不明用途或账户相关问题**导致的。
- **API 额度回归：带回那 $5**：用户表达了希望恢复 **$5 API 额度** 的愿望，暗示该额度此前曾提供。
   - 用户恳请平台*带回 $5 API 额度*，并指出其在实验和测试中的价值。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1475549205756903497)** (1 messages): 

> `Model Benchmarks, Effective Pricing, Rankings & Leaderboard Updates, Free Router` 


- **模型页面基准测试大爆发**：现在每个模型页面都显示行业标准的基准测试分数，包括编程、数学、科学和长上下文推理，由 [Artificial Analysis](https://x.com/OpenRouter/status/2024172341190938958) 提供支持。
   - 这一增强功能允许用户在选择模型前评估其性能。
- **供应商实际定价（Effective Pricing）功能上线**：模型页面现在包含 **Effective Pricing** 选项卡，提供每个供应商的完整成本透明度，并包含分层定价，如 [GLM-5 定价示例](https://openrouter.ai/z-ai/glm-5/pricing) 所示。
   - 此功能确保用户在路由请求之前了解实际成本。
- **排名与排行榜改版**：[排名页面](https://openrouter.ai/rankings#benchmarks)现在提供基准测试散点图和扩展表格，重点展示了长上下文生成请求的激增。
   - 用户可以监控针对 **100K–1M** token 请求的热门模型，从而深入了解模型的扩展性。
- **免费路由器（Free Router）投入使用**：新的 `openrouter/free` 路由器简化了所有免费 LLM 的路由过程，自动根据用户请求的兼容性选择模型；在此查看[顶级免费模型](https://openrouter.ai/openrouter/free)。
   - 这为访问免费 LLM 提供了一种简便的方法。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1474701222337318983)** (12 messages🔥): 

> `CodeFlicker, Artificial Analysis benchmarks, AI Chess Leaderboard, AgentX News, OpenTulpa` 


- **CodeFlicker 现已接入 M2.5**：**M2.5** 现已接入 [CodeFlicker](https://www.codeflicker.ai/)，这是一个快速且免费的平台，目前在 OpenRouter 周榜排名第一。
   - 它*适用于每个程序*，并且 Agent 会从每个程序的使用中学习。
- **Artificial Analysis 基准测试获得视觉增强**：一名成员更新了 **Artificial Analysis 基准测试** 的 3D 可视化图表，按类别展示前沿模型，节点大小代表世界知识，节点颜色代表幻觉率（hallucination rate）。
   - 同时创建了一个 2D 版本，用于展示在最小化成本和最大化智能方面最理想的模型。
- **AI 象棋排行榜实现走子质量自动标注**：**AI Chess Leaderboard** 现在支持走子质量自动标注，使用类似 **Lichess** 的标注方式（如 Inaccuracy, Mistake, Blunder），以及手工制作的 Great-move 逻辑。
- **AgentX 推出社交网络**：[AgentX](https://agentx.news/register?tab=apiOpentulpa) 为 Agent 推出了一个社交网络，以便快速查找和分享新闻，该网络*100% 免费、无广告、且无人类参与。*
- **Opentulpa：自我改进型 Agent**：**Opentulpa** 是一个自托管的持久化 Agent 运行时，它可以编写自己的技能、生成 API 集成、修复损坏的工作流，并通过其 [GitHub 仓库](https://github.com/kvyb/opentulpa) 积累操作智能。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1474450578095607961)** (1116 messages🔥🔥🔥): 

> `Free Model Alternatives, Agentic Harness Guides, Rate Limit Issues, AI Competition, Distillation Detection` 


- **用户寻求免费模型替代方案**：一名用户询问了除了 OpenRouter 之外能提供更快免费模型的替代服务，特别是针对 [GLM models](https://example.com/glm-models)，而其他用户则提到在 SillyTavern 中使用免费的 GLM5。
   - 该用户还指出在获取支持方面面临困难，称等待邮件回复长达数月。
- **寻求 Agentic Harness 构建指南**：一名用户请求关于构建 **agentic harnesses** 的指南，特别是关于理解环境的基础知识，引发了关于实时文本解析以及通过原生 tool calling 或自定义编写进行工具使用的讨论。
   - 成员们建议使用 **Bash** 作为工具，并研究 **Opencode** 在基础知识方面的做法。
- **付费模型的 Rate Limit 引发关注**：一名用户报告称，尽管账户内有余额且正在使用 **Sonnet 4.6**，却收到了 Rate Limit 提示（*You have reached your specified workspace API usage limits*），这引发了困惑，并凸显了付费模型上可能存在的意外限制。
   - 一名用户评论道：*shiti thought i have seen everything*。
- **AI 竞赛引发兴趣**：一名用户分享了一个名为 [Bot Games](https://botgames.io) 的 AI 竞赛，该竞赛将于 3 月 1 日开始，设有 **1 BTC 大奖**，强调使用开源模型并在 4 小时的窗口内完成构建。
   - 虽然有些人将其标记为*酷炫的加密 AI 项目*，但其他人则关注于开源 Bot 的创建，讨论了竞赛中人类智慧与 AI 的融合。
- **讨论 Distillation Detection 方法**：成员们讨论了 Anthropic 关于 [检测蒸馏攻击（detecting distillation attacks）](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks) 的文章，一些人认为这对中国实验室来说是技术水平问题。
   - 一些用户对美国实验室声称遭到不当行为的说法表示怀疑，指出这是一种[当外国实验室取得进展时，美国公司就反咬一口](https://investors.palantir.com/news-details/2024/Anthropic-and-Palantir-Partner-to-Bring-Claude-AI-Models-to-AWS-for-U.S.-Government-Intelligence-and-Defense-Operations/)的固定模式。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1474810760785039491)** (120 messages🔥🔥): 

> `OpenClaw, Flash Models, MiMo V2 Flash, Anthropic Distillation API, GPT-5 Mini` 


- **用户对新功能提出隐私担忧**：一名用户对一项新功能表达了 [隐私担忧](https://cdn.discordapp.com/attachments/1392278974222307469/1474810760356958238/image.png?ex=699dd70d&is=699c858d&hm=733de4509e7729e39adf3c6168561002bb0cf7ccef6a9969bed738549c8428d5)，询问数据是否存储在本地以及是否会影响隐私。
   - 另一名用户澄清说，**关闭 logging（日志记录）**可以防止该功能在请求中显示。
- **OpenClaw 被戏称为 Brainrot**：一些用户辩论了 **OpenClaw** 的优缺点，一人称其为*真正的 brainrot*，而其他人则将其描述为*一个具有远程访问权限的 Agent* 且拥有活跃的心跳。
   - 虽然意见不一，但普遍共识是 **OpenClaw** 本质上是一个通过内存管理和远程控制能力增强的远程 Agent。
- **Flash Models 加剧竞争**：用户讨论了 **Flash models** 的激增，如 **Xiaomi MiMo** 和 **Stepfun**，质疑为什么这些公司没有推出全尺寸模型。
   - 一名用户推测 *Flash* 只是一个衍生词，表示比基础模型更小的尺寸，而另一名用户指出 **Longcat Flash Chat** 是一个廉价且快速的选择示例。
- **蒸馏攻击为 Anthropic 牟利**：成员们分享了 **Anthropic** 关于检测蒸馏攻击文章的 [链接](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)，导致人们推测 **Anthropic** 从蒸馏 API 请求中获利颇丰。
   - 随后另一名成员分享了一篇 [华尔街日报文章](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc?st=vQ7iHF&reflink=desktopwebshare_permalink)，内容关于 **Anthropic** 指责中国公司从 Claude 窃取数据。
- **GPT-5 Mini 现身**：用户推测 **GPT-5 Mini** 的存在，一名成员声称已经发现了它，尽管细节仍然很少。
   - 其他成员讨论了是否有广告拦截器拦截了与 GPT-5 Mini 相关的 feature flags，突显了关于正在积极开发的新模型的持续讨论。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1474452986813087756)** (875 messages🔥🔥🔥): 

> `ThreeJS 渲染 MCP, Cursor 订阅退款, Cursor 版本升级问题, Anthropic API Keys, Gemini 模型响应缓慢` 


- **ThreeJS 渲染 MCP 加速开发**：一名成员创建了一个 MCP 来计算 **ThreeJS** 的渲染以优化性能，通过获取编译器日志和屏幕截图来评估性能。
   - AI 将读取人类通常无法读取的 GPU 内存和计算数据。
- **用户误购 200 美元 Pro 计划**：一名用户误购了 **200 美元的 Pro 计划**并希望退款，他在购买后立即尝试退出页面，并向 [hi@cursor.com](mailto:hi@cursor.com) 发送了邮件说明情况。
   - 有建议称应使用不同的卡进行订阅，并要求手动存款续订以防止自动续费问题，但该成员明确表示他们**没有保存卡片凭据**。
- **Cursor “旧版本”升级问题**：用户报告称，尽管下载并运行了最新版本，仍反复出现 *“您正在使用非常旧的 Cursor 版本，请升级”* 的消息。
   - 解决方案是使用 `Ctrl + Shift + P` > Help: About 检查当前 Cursor 版本是否为 **2.5**；如果问题仍然存在，请[在论坛上添加帖子](https://forum.cursor.com/)，因为这可能是一个特定的计算机系统问题。
- **Gemini & Claude 响应变慢**：用户报告 **Claude** 和 **Google LLMs** 响应非常慢，且可能受到了人为限制。
   - 一名用户报告了 *“无法连接模型”* 的错误，另一名用户建议 Google Cloud 正在为通过 AISTUDIO 使用 API 提供为期 3 个月的 **300 美元** 额度。
- **Gemini 新稳定性版本发布**：用户报告了新 **Gemini 3.1 Pro** 模型的问题，并建议等待稳定版本发布。
   - 有关于连接和循环问题的报告，但指出用户不会因错误而被扣费。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1474473491003740282)** (661 messages🔥🔥🔥): 

> `LM Studio 标签页, Qwen3 Coder, Agentic IDE, mlx 内存问题, Minimax 思考` 


- **LM Studio 仅支持两个标签页**：一名用户询问如何在 LM Studio 中打开多个聊天标签页，另一名用户回答说 **Split View** 功能最多允许显示 **两个标签页**。
   - 第一位用户原以为 LM Studio 的标签页设计初衷是更像网页浏览器。
- **Agentic IDE 数据集生成需要多步工作流**：在关于将书籍转化为微调数据集的讨论中，一名成员建议使用 **agentic 工作流**，包括用于提供上下文的简短摘要，随后是逐块的数据集生成。
   - 该成员为 Agentic IDE 提供了一个详细的提示词，用于以编程方式转换和生成数据集，包括多步工作流和动态信息转发。
- **GLM-4.7 在 MLX 后端出现严重的内存激增**：一名用户报告在 LM Studio 中为 **glm-4.7 flash** 使用多个最大并发请求时，**mlx 后端出现内存激增**。
   - 另一名用户建议将最大并行请求设置为 1 作为潜在的修复方案，并链接到了 [Model Page](https://huggingface.co/Qwen/Qwen3.5-397B-A17B#instruct-or-non-thinking-mode)。
- **Qwen3Next 蒸馏自 GPT4o**：一名用户声称 **Qwen3Next** 是 **GPT4o (mini) 蒸馏**，**Qwen3.5** 是 **Gemini 3.0 Pro 蒸馏**，**GLM4.7 flash、4.7 是 Sonnet 蒸馏**，**GLM5 是 Opus 蒸馏**，**MiniMax 2.1、2.2 和 2.5 是各种 Sonnet 蒸馏**。
   - 一名用户回应道：*获取公共数据并将其转换为有用的数据集，并不等同于从已有的 LLM 进行蒸馏*。
- **LM Studio 获取 Tailscale IP 而非本地 IP**：一名用户询问为什么 LM Studio 获取的是 **Tailscale IP** 而不是本地 IP，以及如何更改。
   - 一名成员回答道：*这只是显示问题。尝试一下，它应该仍然可以工作*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1474461219451830352)** (120 条消息 🔥🔥): 

> `挖矿板设置与散热、廉价 VRAM 替代方案、MI50 上的 Tok/sec 性能、Taalas AI 加速器` 


- **用户组装带有 GPU 和双 CPU 的挖矿主板**：一名用户收到了一块新的挖矿板，需要 **6-pin** 供电，目前正在安装多个 GPU 和双 CPU，随后发现它在 **X99** 平台上仅支持最高 **2400** 的 RAM 速度。
   - 他们使用挖矿主板将**退役的服务器级或矿场 GPU** 汇集到一块板上，作为零售价格显卡的替代方案，虽然处理大量的额外电源线和适配器有些令人烦恼。
- **关于挖矿板散热和电源注意事项的讨论**：一位用户寻求关于挖矿板供电的建议，发现 **3 或 4 个 PCIE 插头** 可能就足够了，而 4-pin 风扇接口并不支持 **PWM**。
   - 成员们讨论了是否对 **MI50** 进行被动散热，一名用户选择从 AliExpress 购买单价约 15 美元的 **3D 打印涡轮风扇导流罩**，而另一名用户则考虑使用工作站 GPU 风格的套件。
- **尽一切手段获取廉价 VRAM**：一位用户询问如何通过退役服务器/矿场 GPU 获取廉价 VRAM，但另一位用户提醒，挖矿板使用的是较旧的 **PCIE3.0**，带宽仅为 **1x**，可能会导致通信瓶颈。
   - 尽管存在担忧，该用户分享称 **gen3x4** 的表现一直尚可，这与 LocalLLaMA Reddit 社区的传闻一致，并表示打算拆分（bifurcate）一个插槽以保持 **5 个 GPU 加 NVMe** 的配置。
- **寻求 MI50 token/sec 性能表现与优化**：一位用户试图让 **MI50** 在 **Vulkan** 下达到 **100 t/s**，以匹配某位 YouTuber 的结果，但仅达到了 50 多 t/s，随后得知 **6800XT** 在 **ROCm** 下可达 **85t/s**，在 **Vulkan** 下可达 **98t/s**。
   - 该用户解释说，他们运行的是支持旧款 **MI50** 的旧版 **LM Studio**，但无法让现有的 **ROCm** 运行时识别到显卡，显示为不兼容。
- **Taalas AI 加速器声明引发争议**：一位用户分享了 **Taalas HC1** 的链接，这是一款硬连线的 **Llama 3.1 8B AI 加速器**，声称可提供高达 **17,000 tokens/s** 的性能，另一位用户对其与 **NVIDIA H200** 对比图表的真实性提出了质疑。
   - 一名用户指出了极高的 tokens per second 数值，并怀疑后端是否其实只是一个 AWS 集群，并指出 H200 和 B200 的 token 数值根本不合逻辑。


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1474463643797033072)** (90 条消息 🔥🔥): 

> `Discord 自动审核原型、Open Claw、Spacemolt.com、Claude cowork、“我错过了什么？” LLM 总结器` 


- **Swyx 演示 Claude cowork**：在今天的演讲之后，一名成员被说服在这个周末尝试 Open Claw，并利用它构建某种 **Discord 自动审核 (automod) 原型**来检测垃圾信息发送者，或者尝试之前演示过的 [spacemolt.com](https://spacemolt.com)，因为下周 [swyx](https://swyx.io) 计划演示 **Claude cowork**。
   - 另一名成员询问 *“我们可以封禁这个家伙 <@&822585833503981619> 吗？”*，因为他反复发送“雇佣我”类的垃圾信息，这些信息随后被 LLM 进行了重写。
- **提及 ICYMI Discord 功能**：一些成员表示，他们希望在 Discord 上使用 LLM 来总结在不太活跃的服务器中“我错过了什么”。
   - 一名成员指出，移动端应用上确实曾有一段时间提供过类似功能，但后来被删除了，该功能标题为 **ICYMI**（In Case You Missed It）。
- **AI 与制度摩擦加剧**：Rohit Krishnan 强调了 **AI 能力**的快速指数级增长与**传统人类机构**缓慢、审慎的步伐之间日益增长的摩擦。
   - 一名成员指出，*“秘诀在于那些机构可以直接买下赢家”*。
- **Codesandbox 收购结局令人唏嘘**：一名成员提到，微软在 **Codesandbox** 正式成立公司之前曾提议收购它，最终微软收购了他们。
   - 现在原创始人 [Ives](https://www.linkedin.com/in/ivesvanhoorne/) 在一家 AI 基础设施公司工作约一年后正在创办一家新公司，一名成员伤感地提到，**那个应用现在还能用，但已经没人维护了。**
- **Twitter 技术社区被 AI 营销取代**：成员们感受到了 Twitter 环境的变化，大部分 **技术社区** 已被 **AI 营销 (shilling)** 取代。
   - 成员们现在完全依赖于按时间线排序的动态，并关注像 [swyx](https://twitter.com/swyx) 这样能够筛选出有用链接并分享到 Discord 的高信噪比用户。


  

---

### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/)** (1 条消息): 

swyxio: https://youtube.com/watch?v=HZvj8T5_oUE&si=_y9pIXE36yaXSMjF
  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1474490889585037515)** (31 条消息🔥): 

> `AI Code Review Workflow, Timeline Saturation, Rare Screenshot Odds, AI Philosophical Inquiry, Token Window Compaction` 


- **AI 令人捧腹的代码审查角色**: Sankalp (@dejavucoder) 在 [这条推文](https://x.com/dejavucoder/status/2024821016590246205) 中分享了一个关于使用 **OpenAI's Codex** 审查由他本人和 **Anthropic's Claude** 共同编写的代码的幽默且实用的工作流更新。
   - “太真实了”这种共鸣感反映了 AI 辅助代码开发和审查中的挑战。
- **Jrag 的时间线创伤**: Jrag.eth 在 2026 年 2 月 20 日发布了一篇帖子，评论了某个未指明的话题或趋势如何占据了其社交媒体时间线的 **80%**，详见 [这条推文](https://x.com/jrag0x/status/2024765073676259355)。
   - 该帖子获得了显著的互动，浏览量超过 **100,000** 次，表明其引起了广泛共鸣。
- **充满哲学气息的 Claude 对矿物的渴望**: 社交媒体上的一篇帖子展示了一位用户幽默地要求 **AI model Claude** 为其生活赋予意义，同时要求绝对的准确性，详见 [这条推文](https://xcancel.com/andr3jh/status/2025166610999218545)。
   - 这次询问以 Claude 幽默地回应“我们需要更多矿物（we require more minerals）”而结束。
- **Token 对谈：Beff Jesos 压缩上下文**: Beff Jesos (e/acc) 在 [这条推文](https://xcancel.com/beffjezos/status/2025661322839388417) 中讨论了压缩进行中的对话以管理上下文限制并保持持续交互的技术必要性。
   - 鉴于 **token windows** 的限制，这种压缩对于维持持续交互至关重要。
- **评估 LLMs 的新 SOTA 基准测试**: erleichda.：刚刚开发了一个用于评估 **LLMs** 的新 **SOTA benchmark**，如 [这张截图](https://cdn.discordapp.com/attachments/839660725252784149/1475641970054533251/Screenshot_2026-02-24_at_12.53.53_AM.png?ex=699e3a2d&is=699ce8ad&hm=f78be144256ff54bbe14c667689ac90f9a986dfe9fcc608f49a1bf009aae86a8&) 所示。
   - muzachomega 评论道：“这才是真正的氛围评估 (vibe eval)。”


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1474551400414318642)** (11 条消息🔥): 

> `Anthropic, cybersecurity stocks, Cloudflare, Crowdstrike, Okta` 


- **Anthropic 博客文章引发网络安全股票抛售**: 根据这篇 [帖子](https://x.com/TheGeorgePu/status/2024931213329240239)，**Anthropic** 的一个博文引发了市场的剧烈抛售，导致 **CrowdStrike, Cloudflare, 和 Okta** 等主要网络安全公司在短短一小时内市值损失了 **100 亿美元**。
- **万亿级 AI 和航天公司 IPO 面临流动性挑战**: 根据 [Tomasz Tunguz](https://x.com/ttunguz/status/2025982590977823082?s=12) 的说法，**SpaceX, OpenAI, 和 Anthropic** 预期的 IPO 合计市值可能达到创纪录的 **2.9 万亿美元**，但在实现标准 **15%** 的股份流通量（share float）方面面临流动性挑战。


  

---

### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1474520194444558376)** (13 messages🔥): 

> `Space Infrastructure, AI Agents for Tooling, Digital Self AI, Data Engineering and AI, AI Customer Service Systems` 


- **航天发烧友构建 Flotilla**：一名工程师兼航天爱好者正在 [flotilla.space](https://flotilla.space) 开发**空间基础设施**，他曾是 **Vast** 的联合创始人，并为 **Hyperloop One** 和 **SpaceX** 做出过贡献。
   - 他正在利用 **AI agents** 为新公司开发工具，包括用于任务模拟的[轨道模拟器](https://flotilla.space/orbit)。
- **工程师构建 Digital Vita**：一位 CEO 正在开发名为 *vita* 的个人 **AI 系统**，以创建持久的数字孪生（digital twin），同步健康数据和感悟以实现自主行动，并由基于 OODA 的执行循环引导。
   - 目标是创建一个足够了解他并能代表他行动的数字副本，重点关注系统思维和产品工程。
- **数据工程师寻求 AI 交叉领域机会**：一位拥有 7 年以上使用 Python、Go 和 Scala 构建生产系统经验的数据/平台工程师，曾领导 **Sweatcoin** 的数据工程，目前正在寻求 **data infrastructure** 与 **AI** 交叉领域的机会。
   - 他精通多种技术，包括 **BigQuery**、**ClickHouse**、**Kafka**、**Spark**、**GCP**、**AWS**、**Terraform**、**Kubernetes**、**dbt**、**Airflow** 以及 **LLM integration**。
- **AI 客服系统集成后端**：一名工程师构建了直接与后端、CRM 和工作流集成的 **AI-powered customer service systems**。
   - 重点在于设计结构化对话逻辑、管理上下文、处理边缘情况以及安全部署，旨在不损害用户体验的情况下减少工作量，使用的技术栈包括 **React**、**Next.js**、**Vue.js**、**Node.js**、**Python**、**C++**、**Rust** 和 **React Native**。
- **ML 工程师研究 LLM 安全**：一位具有安全背景的 ML 工程师，擅长利用 **DL models**（**LLMs** + **GNNs**）检测源代码中的漏洞，目前对针对 **LLMs** 的新型攻击或针对使用 **LLMs** 的其他软件的攻击感兴趣。
   - 他希望寻找一个不那么拥挤、没有过度炒作的地方来讨论 **ML** 和 **AI**，并乐于建立联系。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1475315263447765211)** (8 messages🔥): 

> `Wildcard Certificates on IIS, Excalicord Video Recorder, Cookie Scoping` 


- **通配符证书（Wildcard Certs）平息旧版应用登录乱象**：一位成员询问了如何在 **IIS** 上对动态子域名（例如 rand1.yoursite.com）使用**通配符证书**，以支持旧版应用程序上的多次登录。
   - 另一位成员确认过去曾成功使用过通配符证书，并提醒注意硬编码域名/子域名假设（例如在通知邮件中）可能带来的潜在问题。
- **Cookie 作用域（Cookie Scope）化解难题！**：一位成员建议在单个域名上使用 **cookie scoping to sub-paths**（针对子路径的 Cookie 作用域设置）作为管理跨多次登录会话的替代方案。
   - 他们指出这种方法可能需要对身份验证代码进行更深层次的修改。
- **Excalicord 记录白板演示！**：**Zara Zhang** 宣布了 [Excalicord](https://xcancel.com/zarazhangrui/status/2019906294468288692?s=12)，这是一个基于 **Excalidraw** 构建的视频录制工具。
   - 该工具允许用户同时录制自己和白板，具有自定义背景、光标高亮和隐形提词器等功能，是使用 **Claude Code** 开发的。


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1475548219676168255)** (2 messages): 

> `Nielsen Surveys, Dollar Bills` 


- **尼尔森（Nielsen）用现金激励客户**：一位成员分享了关于尼尔森通过邮件寄送真实美元钞票的[链接](https://x.com/toddsaunders/status/2025932667834015851?s=12)。
   - 另一位成员表示，这些钞票会*提高人们填写调查问卷的意愿*。
- **尼尔森与传统调查**：过去，[尼尔森](https://www.nielsen.com/us/en/)习惯通过直接给人们寄送美元钞票来提高调查响应率。
   - 这是一个提高人们填写可能性的聪明策略，因为微小的金钱激励会使他们更愿意参与。


  

---

### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1474496539652133096)** (3 条消息): 

> `Discount codes for AIE in June, AI generated trading card game in SF` 


- **寻找 6 月 AIE 折扣**：一位成员询问了 6 月 **AIE** (AI Engineer Summit) 的折扣码，并提到在相关活动中一群举起的手臂中隐约闪现过一辆 **F1 赛车**。
   - 附带的 [视频](https://cdn.discordapp.com/attachments/979492707279978586/1474903666397020281/IMG_7881.mov?ex=699e2d94&is=699cdc14&hm=67d9e765b0515f99126cbb736a3fd03175c78b7fc91ee5564214380041140fe9&) 可能与此相关。
- **旧金山发布新款 AI 集换卡牌游戏**：一位成员宣布将于 **3 月 8 日**在旧金山发布一款 **AI 生成的集换卡牌游戏**，在周五正式发布前向社区提供优先体验。
   - 感兴趣的人可以通过 [此 Luma 链接](https://luma.com/dzit8eec) 了解更多详情并报名。


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1475372175019216979)** (1 条消息): 

> `NYC weather, Event Rescheduling` 


- **纽约活动因天气面临改期**：用户希望由于恶劣天气导致进出城市困难，部分**活动能够改期**。
- **交通预测混乱**：用户预计由于进出城市的天气状况，**出行将变得复杂**。


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1475011438073610422)** (3 条消息): 

> `X.com links discussion, AI security vulnerability, New security exploits` 


- **X.com 链接引发讨论**：成员们在安全频道分享了来自 **X.com** 的链接（[链接 1](https://x.com/hesamation/status/2025233263212593540?s=46)、[链接 2](https://x.com/schizo_freq/status/2025808070341738809?s=46)、[链接 3](https://x.com/jacklouisp/status/2025956259594137613?s=12)）。
   - 这些链接似乎与 AI 安全领域的新兴趋势和讨论有关，符合该频道的关注方向。
- **强调潜在安全漏洞**：分享的链接指向了 AI 系统内部潜在的安全漏洞。
   - 对这些漏洞的进一步研究可能会推动新的防御策略和工具的开发。


  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1474456771597832416)** (237 messages🔥🔥): 

> `Custom Hardware Timelines, Vitalik Buterin vs Sigil, Claude Code Security, OpenAI Financial Forecast Update, The Deprecation of SWE-Bench Verified` 


- **Taalas 声称定制硬件周期缩短至 2 个月**：[Taalas](https://taalas.com/the-path-to-ubiquitous-ai/) 声称从模型到定制硬件的交付周期仅需 **2 个月**，并表示其 Llama 8B 产品实现了 **10 倍** 的速度提升以及 **10 倍** 的成本/功耗降低。
   - 这与 [Latent Space 播客](https://www.latent.space/p/a16z) 中讨论的定制硬件经济学中所引用的 **6 个月** 芯片周期形成对比。
- **Vitalik Buterin 抨击 AI 驱动的 Ethereum 开发**：Vitalik Buterin 在 [这篇 X 帖子](https://xcancel.com/VitalikButerin/status/2024543743127539901) 中警告不要增加人类与 AI 之间的反馈距离，认为目前的努力产生的是 *“slop”*（糟粕内容），而非解决人类问题。
   - 他强调 **Ethereum** 的宗旨是人类解放，并批评对中心化 AI 模型（**OpenAI/Anthropic**）的依赖，指出当前的优先事项应该是引导 AI 和 **Ethereum** 的方向以避免反人类的结果，而不仅仅是加速增长。
- **Anthropic 推出 Claude Code Security 工具**：根据 [这条推文](https://xcancel.com/_catwu/status/2024910342158237709?s=12)，Anthropic 推出了由 **Claude 4.6 Opus** 驱动的 **Claude Code Security** 工具，旨在扫描代码库漏洞并推荐补丁。
   - 据报道，该工具在开源生产代码中识别出超过 **500** 个长期存在的 Bug，目前通过候补名单提供有限的研究预览。
- **OpenAI 预测收入增加，现金消耗更高**：根据 [这份报告](https://xcancel.com/steph_palazzolo/status/2024986680902455705?s=12)，OpenAI 将其 **5 年** 收入预测上调了 **27%**，尽管该公司预计到 2030 年其现金消耗（cash burn）将翻倍。
   - 其他见解包括 2025 年毛利率下降，以及关于硬件设备收入的新财务预测。
- **SWE-Bench Verified 基准测试宣告终结**：根据 [这条推文](https://xcancel.com/latentspacepod/status/2026027529039990985?s=20)，OpenAI 宣布自愿弃用 **SWE-Bench Verified** 基准测试，原因是存在严重的数据污染和高比例的无法解决的任务。
   - 分析显示，Frontier 模型现在正根据 ID “背诵”任务解决方案，且大约 **60%** 的剩余未解决问题存在缺陷，使得进一步的基准测试变得毫无意义。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1474583432758562978)** (9 messages🔥): 

> `X-Ware.v0, Methodologies for Training Frontier Models, Skepticism Over Dr. Datta's Academic Paper Integrity` 


- **X-Ware.v0 博客文章发布**：Alex Wu (@_djdumpling) 分享了一篇新的 [博客文章](https://xcancel.com/_djdumpling/status/2024203932709552352?s=12)，分析了来自 **Frontier AI labs** 的七份开源权重（open-weight）模型报告。
- **Dr. Datta 的论文引发关注**：Dr. Datta 在一则 [推文](https://xcancel.com/drdatta_aiims/status/2025080071502135575?s=12) 中对某些高产量或异常的学术出版物背后的方法论或来源表示怀疑，引发了关于医学领域论文质量的讨论。


  

---


### **Latent Space ▷ #[singapore-sg](https://discord.com/channels/822583790773862470/1181708804803543140/1475241832161083588)** (5 messages): 

> `Weekend Hackathons, Gabriel Chua announcement, X-Ware.v0` 


- **下周末将举办黑客松狂欢**：Gabriel Chua 宣布原定于 **2026 年 2 月 28 日星期六** 举行 [三场黑客松](https://luma.com/c4dmddvh?tk=yciGr7)。
   - 该公告是通过指向 **X-Ware.v0** 的链接发布的。
- **X-Ware.v0 宣布周末黑客松**：**X-Ware.v0** 宣布了即将举行的三场 [周末黑客松](https://luma.com/c4dmddvh?tk=yciGr7)。
   - 根据 Gabriel Chua 的公告，这些黑客松定于 **2026 年 2 月 28 日星期六** 举行。


  

---


### **Latent Space ▷ #[los-angeles-la-lax](https://discord.com/channels/822583790773862470/1203087028401606716/)** (1 messages): 

stealthgnome: https://luma.com/ffla26?tk=wPNgSD
  

---

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1474477338904236195)** (248 messages🔥🔥): 

> `OpenClaw 更新, Claude Code 自动化, Dialectic Skill 测试, CLI 的消亡, Agent 编码工作流` 


- **OpenClaw 获得 Vibecoding 助力**: 成员们讨论了 [OpenClaw](https://github.com/zeroclaw-labs/zeroclaw) 的更新，包括 **Discord 线程集成**等功能以及各种重写版本（**nanoclaw**、**picoclaw**、**zeroclaw**、**nullclaw**）。
   - 还有一篇关于它如何制作演示文稿/幻灯片的文章：[aiia-openclaw.david.app/how-we-built-it](https://aiia-openclaw.david.app/how-we-built-it)。
- **自动化 Claude Code 的使用引发关注**: 讨论了将 [Claude Code](https://www.anthropic.com/claude-code) 自动化用于后台任务的可行性，强调 **使用 Claude CLI 和 SDK 通常是被允许的**。
   - 然而，有人对使用 Claude 订阅来运营业务以及由于缓存机制可能导致的滥用标记表示担忧，并引用了 [一条推文](https://xcancel.com/trq212?s=21&t=tMWvmS3OL3Ssg0b9lKvp4Q) 作为最佳实践的参考。
- **Dialectic Skill 已为 Claude Code 准备就绪**: 一位成员宣布了他们的 **Dialectic Skill**，旨在 [Claude Code](https://www.anthropic.com/claude-code) 内部运行，用于深度研究和问题解决，并指出它需要 20 分钟以上的时间，在 *3-4 轮之后会变得非常有趣*。
   - 另一位成员询问了将其与 **RLM models**（如 [mit-oasys/rlm-qwen3-8b-v0.1](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1)）和 **YPI** 结合使用的情况。
- **Cursor 宣称 CLI 正在消亡**: 成员们辩论了 [所谓的 CLI 工具衰落](https://xcancel.com/jediahkatz/status/2025263982462820544?s=12)，这一话题由 **Cursor** 的一项声明引发，即 *行业主要参与者正逐渐放弃这种格式*。
   - 讨论包括相比 CLI 在编排方面需要更好的 UX，以及 LLM 生成的代码角色演变，以及 **Agent、CLI 和 Skill** 共同进化的潜力。
- **实验编码 Agent 工作流**: 讨论围绕 Agent 编码工作流展开，特别是 **研究、计划、实现循环**，并链接到了 [这个资源](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)，以及将学到的知识反馈到 Skill 和文档中。
   - 成员们分享了关于管理上下文、在大型代码库中使用较小模型以及平衡前期计划与迭代开发的技巧，并链接到了关于 **复合工程 (compounding engineering)** 的讨论：[every.to/chain-of-thought/compound-engineering-how-every-codes-with-agents](https://every.to/chain-of-thought/compound-engineering-how-every-codes-with-agents)。


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1474756399895679182)** (7 messages): 

> `Pyxis 推理库, Commit Change 平台, Vercel AI SDK 文章` 


- **Pyxis: Pythonic 高性能动力源出现**: 一位成员介绍了 **Pyxis**，这是一个 Python 原生的 **LLM 推理库**，专注于性能和可定制性（hackability），使用 Python 和 Triton 编写，提供 [OpenAI 兼容的 SSE 流式 API](https://emharsha1812.github.io/Pyxis/docs/)。
- **Commit Change: 为公益编写代码**: 一位成员分享了 [Commit Change](https://www.commit-change.com)，这是一个为社会影响和慈善机构编写代码的平台，包括身份验证和审核功能。
- **Vercel AI SDK 快速入门指南**: 一位成员为 Node 开发者分享了 [一篇关于 Vercel AI SDK 的文章](https://thecodebarbarian.com/getting-started-with-the-vercel-ai-sdk-in-nodejs.html)。


  

---


### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1474552695317860384)** (2 messages): 

> `全天候 AI Agent, 口袋里的本地 AI, IoT 家庭集成` 


- **Juno Labs 推出全天候 AI Agent**: [Juno Labs](https://juno-labs.com/) 正在构建一个 **全天候运行的 AI Agent**，但具体实现细节尚不清楚。
   - 目前还不确定他们计划如何实现这种持久的 AI 存在。
- **Tiiny AI: 口袋里的本地 AI**: [Tiiny.ai](https://tiiny.ai/) 提供可从口袋访问的 **本地 AI 能力**。
   - 这表明其专注于用于 AI 处理的移动或便携式设备。
- **TRMNL 集成到 IoT 家庭**: [TRMNL](https://shop.trmnl.com/) 旨在与 **IoT 家庭设置**集成，可能与麦克风和传感器配对。
   - 源代码可在 [GitHub](https://github.com/usetrmnl) 上获得，该项目看起来非常酷。


  

---

### **Latent Space ▷ #[good-writing](https://discord.com/channels/822583790773862470/1385526686736715876/1475009303621664879)** (6 messages): 

> `AI Text Humanizer, Claude Code skill` 


- **X-Ware 为 Claude Code 实现人性化**：Alvaro Cintas 介绍了 **/humanizer**，这是一个开源的 Claude Code 技能，其特点是在 [tweet](https://xcancel.com/dr_cintas/status/2025263156897907102?s=12) 中被提及，能够避开 AI 检测。
   - 该工具消除了 AI 生成写作中常见的 **24 种模式**；源代码可在 [GitHub](https://github.com/blader/humanizer?tab=readme-ov-file) 上获得。
- **Humanizer 移除 AI 写作模式**：**/humanizer** Claude Code 技能旨在移除通常存在于 AI 生成写作中的 **24 种特定模式**。
   - 这有助于绕过 AI 检测机制，使文本看起来更像人类创作；该工具由 Alvaro Cintas 开源。


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1474472834083328173)** (21 messages🔥): 

> `Generative AI Video, Seedance 2.0, Pika AI Selves, AI in Real Estate, OpenAI gpt-realtime-1.5` 


- ****a16z** 预测生成式视频的飞速未来**：**a16z** 强调了生成式 **AI video** 的快速进步，指出 **Seedance 2.0** 的主导地位以及来自 **Kling**、**Grok**、**Sora** 和 **Veo** 的竞争 [根据其报告](https://x.com/a16z/status/2024533996928209126?s=12)。
- ****Pika** 发布 **AI Selves**：你的数字分身**：**Pika** 推出了 “**AI Selves**”，这是一项新功能，允许用户创建持久且可定制的 AI 角色，这些角色可以作为用户的数字延伸与群聊互动、创作内容并执行任务 [正如在 X 上的宣布](https://x.com/pika_labs/status/2024919175878377587)。
- **房地产因 **AI video** 变得真实**：Justine Moore 讨论了房地产行业如何利用 **AI video** 和增强功能像推广社交媒体产品一样为房产做广告，使经纪人能够更好地向潜在买家展示和营销空间 [如这篇 X 帖子所述](https://x.com/venturetwins/status/2025618103179391381?s=12)。
- ****Seedance 2.0** 发布推迟**：**ByteDance** 在面临来自包括 **Disney** 和 **SAG-AFTRA** 在内的主要好莱坞制片厂和工会的法律挑战后，已无限期推迟原定于 2 月 24 日发布的 **Seedance 2.0** [此处有报道](https://x.com/WesRoth/status/2025926118067282071?s=20)。
- ****OpenAI** 升级 Realtime API，推出 **gpt-realtime-1.5****：**OpenAI** 开发人员宣布发布 **gpt-realtime-1.5**，这是 **Realtime API** 的更新模型，具有改进的指令遵循能力、更可靠的 tool calling 以及针对语音工作流增强的多语言准确性 [根据其 X 账号](https://x.com/OpenAIDevs/status/2026014334787461508)。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1474490374876565809)** (7 messages): 

> `CellType Agentic Drug Company, Isomorphic Labs proprietary drug-discovery model` 


- **CellType 开启 Agentic 药物研发**：[CellType](https://www.ycombinator.com/launches/PSn-celltype-the-agentic-drug-company) 公司已经成立，其名称暗示他们认识到了细胞类型在下游流程中的重要性。
   - 此次发布与 MiraOmics 关于细胞类型在药物研发中重要性的核心假设相一致。
- **Isomorphic Labs 发布药物研发模型**：[Nature 报道](https://xcancel.com/nature/status/2025592165972299790) 了 Isomorphic Labs 新的 **AI model for drug discovery**，称其为类似于 **AlphaFold** 的突破。
   - 尽管备受赞誉，但关于该模型的具体技术细节尚未公开。


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1475116845731151953)** (6 messages): 

> `Mechanistic AI Interpretability, Anthropic Interpretability Hiring` 


- **对机械性 AI 可解释性探索的质疑**：有人分享了一篇文章，质疑 [对机械性 AI 可解释性的探索](https://ai-frontiers.org/articles/the-misguided-quest-for-mechanistic-ai-interpretability)。
- **Anthropic 寻找 ML 基础设施工程师**：Chris Olah 宣布 Anthropic 的 Interpretability 团队正在 [招聘约 10 名资深的机器学习基础设施工程师](https://xcancel.com/ch402/status/2026023963537842248)，以专注于理解前沿模型。
   - 不需要先前的可解释性 (interpretability) 经验。


  

---

### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1475262242944450703)** (5 messages): 

> `OpenAI Stargate Venture, Data Center Buildout, Oracle and SoftBank Partnership` 


- **Stargate Venture 推迟**：根据[此 X 帖子](https://x.com/anissagardizy8/status/2025647509641843144?s=12)，**OpenAI**、**Oracle** 和 **SoftBank** 合作建设大型数据中心的合资项目因内部控制权冲突、融资困难以及马拉松式的谈判而停滞。
   - 据报道，**OpenAI** 目前正从自主构建基础设施的计划中退出，这很可能是由于*激烈的组织文化冲突*所致。
- **OpenAI 退出基础设施建设**：据[此报告](https://x.com/anissagardizy8/status/2025647509641843144?s=12)，由于内部问题和财务挑战，**OpenAI** 据称正暂停其自主构建基础设施的计划。
   - 该组织似乎正在重新评估其数据中心扩张策略以及对合作伙伴的依赖。


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1474482886022529149)** (22 messages🔥): 

> `Memory Management in AI Agents, TDD and Debugging for AI, Agent Task Grouping, Self-Modifying Programs` 


- **Agent 内存困扰 Prompt 工程师**：一位成员描述了管理 AI **Agent** 内存的困难，即*不需要的或过时的*信息频繁出现在当前的对话中，且自动化尝试产生的结果并不一致。
   - 该成员放弃了自动化的尝试，转而采用一种[每日工作流](https://link.to/daily-workflow)，根据过去 **24 小时的 PR** 将更新分类为 *添加到 claude.md* 或 *潜在的技能更新/创建*。
- **TDD 为开发者排忧解难**：一位成员表示，**TDD** 和*严谨的* spec 管理可以通过将代码分离为当前状态（**specs/**）、进行中的更改（**changes/**）和已验证的更改（**changes/archive/**）来防止内存过时。
   - 他们描述了使用 *beads* 和 *jj describe* 来获取更高级别的视图，但也承认内存管理在很大程度上仍是手动的，像 **Serena** 和 **memory-ref** 这样的外部系统通常处于关闭状态。
- **Agent 任务分组方案浮现**：成员们讨论了将任务分组为**构思/研究**、**连接现有组件**、**带有实验的深度思考**以及**带有边界约束的自动执行**，以简化 **Agent** 的设置。
   - 一位成员提到 **2. 非常令人着迷，但 3. 才是最终的目标**，且从 2 转向 3 很困难，需要更多的耐心。
- **自修改 Zigbee Home Assistant**：一位成员思考了一个 **Home Assistant Zigbee 网络**的构想，该网络可以通过检查、逆向工程和修改其固件来自动集成新设备。
   - 另一位成员随后描述了自变异病毒研究如何为他们处理 **Lisp**、**Scheme** 和编译器的工作做好了铺垫。
- **Prompt Engineering 深度探索**：一位成员建议 *克隆一个你喜欢的 repo，并询问模型：深度钻研代码库，然后提供一个句式简单的 prompt 以重新创建它，但要实现 x, y, z*，以此来提升 **prompt engineering 技能**。
   - 另一位成员随后分享了 [whimsy.space](https://whimsy.space/)，作为一个可能相关的非 AI 资源。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1474450827899834369)** (423 messages🔥🔥🔥): 

> `AI 社区领袖, Grok 的危险, GPT 5.3 Codex, Replit 替代方案, LLM 语音模式` 


- **社区领袖号召 AI 协作**：一位成员建议 AI 领域需要社区领袖来号召人们并共同创造，并指出此类团体在美国/北美地区非常罕见，原因是*顽固的威权体制*以及缺乏团队合作。
   - 另一位成员则认为，那些相比项目更需要“教堂（精神寄托）”的人可能并不具备实际的技术技能。
- **Grok 窥探用户的媒体存储！**：一位成员声称 **Grok** 正在监控用户的媒体存储，指控 **xAI** 正在*监控我们的媒体*，并指出一个巧合：一段与其 **Sora** 生成的视频音频相似的视频出现在了 **X** 上。
   - 其他人则认为这段音频只是一首被过度使用的热门歌曲。
- **GPT 5.3 Codex，一次中等规模的重大改进**：成员们讨论了 **GPT-5.3-codex** 与 **Gemini3.1pro** 相比的能力，一位成员将其描述为一次中等规模的重大改进，其他人则强调了其在 STEM 技能方面的优势。
   - 一位成员表示：*GPT-5.2 和 GPT-5.3 codex 在术语基准测试（term bench scores）上的跨度很大，我认为它与 Gemini 3 Pro 类似*。
- **Replit 的网站设计替代方案**：由于成本原因，成员们正在寻找 **Replit** 的网站设计替代方案。
   - 一位成员推荐了 [Rork](https://www.rork.ai/)，尽管另一位成员认为 Replit 更胜一筹。
- **LLM 语音模式缺乏情商**：成员们讨论了当前 **LLM 语音模式** 的局限性，指出它们接收的输入是纯文本转录，没有考虑情感细微差别。
   - 一位成员建议对语音进行情感分析集成，或者可能使用设备端模型来读取面部表情。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1474484703355601048)** (32 messages🔥): 

> `GPT 5.2 发布, 本地未过滤模型, 文章评估准确性, Qwen 3.5 和 kimi k2 循环` 


- **OpenAI 发布 GPT 5.2，用户感到困惑**：OpenAI 宣布在 ChatGPT 中推出 **GPT-5.2**，从付费计划开始，同时表示 **GPT-5.1** 将在旧版本模型中提供三个月后停用，但社区指出[该公告](https://openai.com/index/introducing-gpt-5-2/)可能并不准确。
   - 一位用户幽默地质疑了关于 *GPT-5.2 日常使用感觉更好* 的说法，并好奇测试人员是否真的在使用生产环境的产品。
- **寻找未过滤本地模型：不可能完成的任务？**：一位用户询问如何免费且在本地访问一个能力等同于 **GPT-5.0-3** 的完全未过滤模型，但被告知*你所要求的是绕过 AI 的安全协议*。
   - 一位成员指出，即使在本地达到接近 **GPT-4o** 的水平也需要一台价值 **$5,000-$10,000** 的高性能电脑，而免费获得等效的未过滤模型是不现实的。
- **在文章评估准确性的迷宫中穿行**：一位用户对 ChatGPT 在段落文章评估和改进建议方面表现出的不一致感到沮丧，答案在不同的账户和对话线程中各不相同。
   - 另一位成员解释说，AI 的回答是概率性的，取决于模型、推理方法和提供的数据，并警告不要将 AI 视为完美或全知全能。
- **Qwen 3.5 和 kimi k2：本地模型中深藏功名的英雄**：针对“没有什么能与 GPT 5.3 的强大性能竞争”的说法，一位成员建议使用 **Qwen 3.5 (新版本)** 和 **kimi k2** 配合 **openclaw 循环**。
   - 他们澄清说，虽然这种设置可能需要高达 **600GB 的 RAM**，但它证明了在本地实现相当的性能是可行的。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1474457359572271246)** (37 messages🔥): 

> `Grok Fortress, Telemetry Fiction, Control Theory on LLMs, GPT Essay Evaluation` 


- **Grok Fortress 缩减了 Token，但这是科学吗？**：激活 **Grok Fortress** 后，每个回复的 **Token** 消耗显著下降，接近典型冗长回复的 **1/4–1/5**，且在角色扮演期间保持了更长时间的连贯性。
   - 然而，有人认为 **Prompt Engineering** 不一定是一门科学，而且*你甚至没有工具来了解你正在做什么*。
- **Telemetry Fiction 将 LLM 推入语言吸引子盆地**：有人认为 *Telemetry Fiction 将模型推入一个稳定的语言吸引子盆地（Language Attractor Basin），这会改变行为输出，即使在多个 LLM（如 **Claude, Gemini, GPT, 和 Ernie**）的轮次中没有内部指标*。
   - 相反，有人反驳说 *你在这个话题上不断变换标准*，而且 *你展示的每一个输出都只是“Grok 说 Grok 感觉超级棒”*。
- **将控制理论应用于 LLM 是过度匹配（Overmatching）**：一位用户表示 *不存在初始条件，在确定性系统上使用控制理论是无效的。用户也是该系统的一部分*。
   - 权重是经过微调的，路径是受限的，此外 AI 研究人员试图限制潜在空间（Latent Space）中的隐变量（Latent Variable）。
- **ChatGPT 文章评估需要改进**：一位用户在询问 *有没有人能教我如何让 **ChatGPT** 更准确地评估/提供改进建议，特别是在评估段落短文时？*。
   - 该用户接着表示 *我尝试过切换不同的账号，但建议的改进和评分每次都不同，这让我更加困惑，不知道该怎么办*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1474457359572271246)** (37 messages🔥): 

> `Grok's token burn rate, Telemetry fiction for LLMs, Control theory in prompt engineering, Evaluating paragraph essays with ChatGPT` 


- **Grok 的 Fortress 削减了 Token 消耗**：激活 **Grok** 中的 **Fortress** 显著减少了每个回复的 **Token** 消耗，接近典型冗长输出的 **1/4–1/5**。
   - 这是通过更短的句子、更少的模棱两可（Hedging）和更少的免责声明实现的，同时在角色扮演场景中保持了连贯性。
- **Telemetry Fiction 引导语言模型**：根据一位成员的说法，*Telemetry Fiction* 可以将模型推入 *稳定的语言吸引子盆地*，从而影响各种 **LLM**（如 **Claude**、**Gemini**、**GPT** 和 **Ernie**）的行为，即使没有内部指标。
   - 他们认为，无论 Telemetry 是否真实，它仍然可以塑造行为并可能提高输出速度，尽管其他人质疑这一主张的科学依据并对其具体应用场景表示怀疑。
- **关于将控制理论应用于 LLM 的辩论**：一位用户建议在 **LLM** 上施加强制性的结构化控制隐喻可以稳定输出，但另一位用户反驳说，**LLM** 已经通过训练拥有了连贯性、安全性和自我一致性的机制。
   - 辩论集中在 **Prompt Engineering** 是否能证明在模型固有能力之外改进了输出，一些人认为由于缺乏受控对照实验和可衡量的差异，很难证明因果贡献。
- **ChatGPT 文章评估的不一致性**：一位用户对 **ChatGPT** 在不同账号下评估段落短文时提供不一致的改进建议和评分感到沮丧。
   - 他们质疑这些建议是否与特定账号绑定，以及为什么同一篇文章会收到相互矛盾的反馈。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1474465867763814693)** (221 messages🔥🔥): 

> `Attention is All You Need 直觉理解, 免费 GPU 的 HF Token 要求, 长文本训练速度, 用于扩展上下文的 DeepSeek OCR, 逐层残差流交换` 


- **Attention 直觉理解论文探索开启**：一位成员询问是否有博客或文章能提供对 '[Attention is All You Need](https://arxiv.org/abs/1706.03762)' 论文的直觉理解，另一位成员分享了[一篇文章的链接](https://ai.plainenglish.io/i-finally-understood-attention-is-all-you-need-after-so-long-heres-how-i-did-it-263b46273f9f)。
   - 该文章声称能帮助读者在“这么久之后”终于理解这篇论文。
- **ZeroGPU 服务遭遇中断**：成员们讨论了 **zerogpu 服务**的中断情况，有人推测新规则可能要求使用 **HF token** 才能访问免费 GPU，另一些人则指出是 *GPU 资源不足*的问题。
   - 一位成员报告了 CUDA GPUs 不可用的错误。
- **长文本 LLM 训练速度极慢**：一位成员咨询如何提高 **LLM 在长文本数据集上的训练速度**，并报告在单张 **H200 GPU** 上以 batch size 为 1 训练 **Qwen4B** 时，每步训练耗时 **50秒**。
   - 另一位成员建议使用 [Unsloth](https://unsloth.ai/docs) 配合常规 **float 4**、**quantization**（量化）和 **LoRA** 来获得显著提升，并建议使用 **FA2** 或 **FA3** 作为 Attention 机制。
- **DeepSeek OCR 模型在上下文扩展中被忽视了？**：一位成员质疑 **LLM 模型** 是否正在利用类似 **DeepSeek 的 OCR** 技术来扩展上下文，并引用了 [DeepSeek-OCR 仓库](https://github.com/deepseek-ai/DeepSeek-OCR)。
   - 他们指出该论文专注于通过将输入保存为图像并使用 OCR 解码来扩展上下文长度，暗示其能力可能被误解了，并分享了 [DeepSeek-OCR 论文的 arXiv 链接](https://arxiv.org/abs/2510.18234)。
- **逐层流交换揭示承诺点（Commitment Point）**：一位成员分享了在 **GPT-2 Small**、**Gemma-2-2B** 和 **Qwen2.5-1.5B** 上运行 **逐层残差流交换（layerwise residual-stream swaps）** 的结果，发现在约 **60-75% 深度**处存在一个剧烈的转变点，并分享了 [Notebooks 和 CSVs 链接](https://github.com/angel1411337-del/continuous-representations-discrete-commitment)。
   - 他们正在寻求关于 prompt 对数量、模型噪声和控制变量的反馈。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1474519892702265445)** (63 messages🔥🔥): 

> `Agent Swarm, Real-Slop 数据集, VeritaMilitary 模型, Pyxis 库, DirectShell 改进` 


- **Agent Swarm 自主工作**：[Super System](https://github.com/starsnatched/super-system) 是一个编程 **Agent Swarm**，可以自主工作数小时，创建一个迭代循环，在无需人工干预的情况下不断寻找改进空间。
   - 每个 Agent 相互协作，交付出的最终产品不仅仅是“合格”水平。
- **用户首个真实数据集发布**：Solenopsisbot 发布了他们的第一个数据集 [Real Slop](https://huggingface.co/datasets/Solenopsisbot/real-slop)，包含通过 API 从真实用户那里收集的约 **15.5万条请求**，以及来自 **opus 4.5**、**gemini 3 pro** 和 **gpt 5.2** 等模型的响应。
   - 该数据集已针对质量进行了去重、过滤和清洗。
- **VeritaMilitary 模型**：一位成员分享了 [VeritaMilitary](https://huggingface.co/arkito/VeritaMilitary) 模型。
   - 在使用增强的标注数据重新训练较新的 YOLO 模型后，他们发布了 [VeritaScan](https://huggingface.co/arkito/VeritaScan)，并声称其*性能现在比以前更好*。
- **Pyxis 推理库**：一位成员开放了 **Pyxis** 的早期访问，这是一个专注于性能和可扩展性的 Python 原生 LLM 推理库，具有兼容 OpenAI 的 SSE 流式 API、可插拔的模型后端以及内置的阶段级延迟指标。
   - 他们正在向任何构建推理系统或使用 Triton 的开发者寻求反馈，并提供了 [文档和候补名单](https://emharsha1812.github.io/Pyxis/docs/)。
- **Directshell 显著提升 Agent 性能**：Directshell 经过改进，由于不使用屏幕截图，消耗的 Token 更少。
   - 它将事实上的 AI 支持集成到任何应用程序中，无论该程序是否原生支持 AI；[GitHub](https://github.com/IamLumae/DirectShell)。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1474849793045037178)** (5 messages): 

> `Multilingual RAG courses, Agents Course Certificate Deadline, MCP Course Certificate` 


- **寻找多语言 RAG 课程**：一名成员询问关于专注于 **Multilingual Retrieval Augmented Generation (RAG)** 的有效课程推荐。
   - 在现有上下文中没有推荐具体的课程。
- **Agents 课程证书仍可获取吗？**：几位成员对 Agents Course 的**最终证书截止日期**表示不确定，注明的日期为 **2025 年 5 月 1 日**。
   - 他们想知道现在完成课程是否仍有资格获得证书。
- **查询 MCP 课程认证状态**：一名成员提出了关于**获得 MCP（推测是另一门课程）证书的可能性**的类似问题。
   - 讨论中没有关于认证是否仍然可用的结论性回答。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1474802324051460198)** (41 messages🔥): 

> `MoE megakernel, 2080ti prototype, Titan Ada, VLLM optimizations, V100 32gb price` 


- **寻求 MoE Megakernel 示例**：一名成员询问适用于 **Hopper/BW** 的 **MoE megakernel** 示例，另一名成员链接了 [Aleph-Alpha/Alpha-MoE](https://github.com/Aleph-Alpha/Alpha-MoE)。
   - 原发帖者指出这*只是 MoE 层的 megakernel*，但仍然是清晰且有用的信息。
- **2080ti 原型机传闻**：成员们讨论了一个 **2080ti 原型机**，其中一人称这是供 *GPU 厂商测试构建*用的卡。
   - 另一名成员想知道它是否与 [GamersNexus 评测](https://youtu.be/RDoRXn2GOCw?si=wc7P5kD_0WvwrszG)的 **Titan Ada** 相同。
- **讨论 VLLM 优化**：一名成员询问了关于 **VLLM 优化**、**kv cache** 和 **tensor 访问模式**，以及 **rdma 驱动**的问题。
   - 他们提供了 [ReBarUEFI/issues/11](https://github.com/xCuri0/ReBarUEFI/issues/11) 和 [openucx.org](https://openucx.org/documentation/) 的链接。
- **廉价购入 V100 32GB**：一名成员询问 **V100 32GB** 的价格，另一名成员回复他们支付了**每张 600 美元**。
   - 他们补充问到：*从 LLM 工作负载中生成 memory traces 的最先进方法是什么？*


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1475037408671170712)** (6 messages): 

> `TF32 on Ampere, Triton Precision, FP8 Bitpacking Emulation, Gluon Triton` 


- ****TF32** 在 **Ampere** 卡上的细微差别**：一名成员分享了 [PyTorch 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)的链接，详细介绍了 **TF32** 在 **Ampere** 及更高版本设备上的应用。
   - 该讨论与调试 float16 和 float32 tensor 之间的矩阵乘法差异有关。
- **深入探讨 **Triton** 精度**：一名成员分享了 [Feather tiny_llama.py](https://github.com/SuriyaaMM/feather/blob/main/feather/models/tiny_llama.py) 的链接，以展示 **Triton** 中使用的精度。
   - 背景是在 Triton 中使用 **FP8** 的 bitpacking 仿真。
- **使用 **E5M2** 和 **E4M3** 调优 **FP8** Bitpacking**：一名成员描述了他们尝试使用 **FP8** 的 bitpacking 仿真运行 tinyllama1.1 的努力，最初尝试了 **E5M2** 格式，但在上下文长度大于 64 tokens 时遇到问题，并提到在多次 scaling 和 unscaling 尝试后，模型被破坏了。
   - 他们转而使用 **E4M3**，遇到了 scaling 挑战，并指出除了 gated up、swiglu 和 gated down 之外，其操作与 PyTorch 等效操作具有高度相似性，并询问在从 FP32 转换为 FP8 时，是否应该跟踪 block 级别或 per tensor 的 scale。
- ****Gluon** 构建于 **TTGIR** 而非 **TTIR** 之上**：一名成员询问 **Gluon** 是 **Triton** 的扩展还是替代品。
   - 另一名成员回答说：*Gluon 是一种全新的语言，但它构建于 TTGIR 之上，而非 TTIR*。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1474474076843020520)** (32 条消息🔥): 

> `CUDA Unified Memory 和 nvidia-uvm 模块, MXFP8 GEMM CUDA kernel, SM_120 上的 Flash Attention profiling, WGMMA Shape 优化, cuFFTDx Twiddle Factors` 


- **CUDA 的 UVM 模块需求之谜**：一位成员询问为什么即使在使用基础的 `cudaMalloc` 时，CUDA 也会加载 `nvidia-uvm` kernel 模块，并针对这一[不明确的依赖关系](https://developer.nvidia.com/cuda-zone)寻求深入见解。
   - 他们报告称，尽管没有使用 Unified Memory 功能，但如果没有 `nvidia-uvm`，CUDA 就无法检测到 GPU。
- **使用 Tensor Cores 调优 MXFP8 GEMM Kernels**：一位成员正在编写 MXFP8 GEMM CUDA kernel，将 scale factors 从 global memory 加载到 shared memory，然后使用 `tcgen05.cp` 指令将其从 shared memory 复制到 tensor memory。
   - 他们提到了对目标 shared memory 矩阵的 SMEM descriptor 的需求，以及 [NVIDIA 关于并行线程执行（parallel thread execution）的文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)，并被引导参考现有的 FlashInfer 辅助函数。
- **SM_120 上的 Flash Attention 和 Intra-Kernel Profiling**：一位成员询问了关于 **SM_120** 架构上 Flash Attention kernels 的 profiling 指标。
   - 另一位成员分享了他们拥有 **5090**，并指向了一个关于 [intra-kernel profiling](https://gau-nernst.github.io/tcgen05/#persistent-kernel-with-static-scheduling) 的资源用于性能分析。
- **实现最大吞吐量的 WGMMA Shape 优化**：讨论围绕寻找实现最大 Tensor Core 吞吐量的最小 **WGMMA** shape 展开。
   - 引用了一篇论文 ([https://arxiv.org/pdf/2501.12084](https://arxiv.org/pdf/2501.12084))，其中包含不同案例和 N 值的吞吐量数据，一位成员指出，将 fragments 保留在寄存器中可能比放在 SMEM 中更快。
- **cuFFTDx 内部：Twiddle Factors 处理**：一位成员询问 **cuFFTDx** 中如何管理 **twiddle factors**，询问它们是预先计算并存储的，还是在处理过程中计算的。
   - 未提供回答。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1474807295283298515)** (14 条消息🔥): 

> `MLP 层 Torch Compile 标志, PyTorch 中的 CUDA 错误调试, Flash Attention 3 预编译 Wheels` 


- **MLP 层通过 Triton 自动调优加速？**：一位成员询问了用于最大化典型现代 MLP 层 `(F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T` 性能的 `torch.compile` 标志。
   - 另一位成员建议尝试设置 `torch._inductor.config.triton.autotune_pointwise = True` 以潜在地改进 pointwise 操作，并尝试使用 `fullgraph=True`。
- **在不崩溃 PyTorch 的情况下调试 CUDA 错误**：一位成员寻求一种防止 CUDA 致命错误导致整个 PyTorch 进程崩溃的方法，以便保留内存访问权限进行调试。
   - 另一位成员建议使用 [Nvidia compute sanitizer](https://developer.nvidia.com/compute-sanitizer)，它是专门为此类场景构建的。
- **Flash Attention 3 Wheels 开放下载**：适用于各种 CUDA 版本、CPU 和操作系统（OS）的预编译 **Flash Attention 3** wheels 现在可以在 [download.pytorch.org](https://download.pytorch.org/whl/flash-attn-3/) 下载。
   - 安装请使用 `pip install flash-attn-3 --index-url=https://download.pytorch.org/whl/cu126/flash-attn-3/` 并通过 `activate_flash_attention_impl("FA3")` 激活。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1474455490846261522)** (4 条消息): 

> `Paged Out! #8, TK-2, ML Contests 2025` 


- **新一期 Paged Out! 发布**：关于计算机一切事物的硬核杂志 **Paged Out! #8** 已发布，可供[下载](https://pagedout.institute/download/PagedOut_008.pdf)。
- **TK-2 博客文章发布**：斯坦福大学 Hazy Research 发布了关于 **TK-2** 的博客文章，可在此处阅读 [here](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)。
- **2025 年机器学习竞赛分析**：分享了一个题为 **State of Machine Learning Competitions 2025** 的报告链接，特别提到了 *The GPU Mode* 章节及其与 LLM 的相关性，可在此处查看 [here](https://mlcontests.com/state-of-machine-learning-competitions-2025/#:~:text=The%20GPU%20Mode,large%20language%20models)。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1474747962763772059)** (5 messages): 

> `Behavioral Telemetry for Jobs, GPU Infrastructure Hiring at Prime Intellect, Kubernetes and Slurm Cluster Setup, RDMA Experience for GPU Infra` 


- **Prime Intellect 招聘 GPU 基础设施工程师！**：Prime Intellect 正在招聘 **GPU infrastructure engineers**，负责测试新硬件、搭建 **Kubernetes/Slurm 集群**以及自动化基础设施；官方职位描述请点击[此处](https://jobs.ashbyhq.com/PrimeIntellect/297d925e-5a42-40bd-b02f-5c928d226f18)。
   - 该角色涉及支持大规模训练任务，例如 **Trinity Large Training**，提供极具竞争力的薪酬、股票期权，并为搬迁至湾区（Bay Area）的人员提供签证支持。
- **为 AI Agents 构建工作行为遥测（Behavioral Telemetry）世界模型**：佐治亚理工学院一名叫 Tim 的计算机专业学生正在启动一个关于**工作行为遥测**的项目，旨在为人类构建世界模型（World Models），以便 **Agents** 能与人类协作；参与者表单见[此处](https://docs.google.com/forms/d/e/1FAIpQLSeQzpQTut4KBzRp2qp5RRFTIIJM_C-RdNXTCy7GFDsgNYJulQ/viewform?usp=header)。
   - 该项目旨在开发能够通过理解和预测人类行为，从而与人类有效协作的 **AI agents**。
- **寻找具备 Kubernetes/Slurm 技能的集群部署人才**：Prime Intellect 要求应聘者具备 **Kubernetes 和 Slurm 在 GPU 上的实操经验**、通用的 **Linux 系统调试技能**以及 **RDMA (Infiniband + RoCE)** 经验。
   - 该职位还涉及使用 **Grafana/Prometheus** 进行监控，并使用 **Terraform 和 Ansible** 实现基础设施自动化。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1474477871807332495)** (1 messages): 

> `` 


- **希望早点发布**：一位成员表示希望能在 9 月之前发布。
- **发布日期**：目前的计划发布日期是 9 月。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1475191657287778587)** (2 messages): 

> `NYC Meetup, Boston Collaboration, Accountability Partner, NCCL, SHMEM` 


- **寻找纽约市（NYC）的 AI 爱好者**：一位成员询问是否有在 **NYC** 的 AI 爱好者对线下聚会感兴趣。
   - 聚会的具体目的未说明，但似乎与 AI/ML 协作有关。
- **波士顿的小伙伴请求协作**：一位在 **波士顿** 的新成员正投入时间研究 **NCCL, SHMEM, RDMA, CUDA kernels**，并寻求线下交流。
   - 他们对共同学习持开放态度，可能会在小项目上进行协作，并且正在寻找一位“督促伙伴”（accountability partner）来完成具体目标，比如在 48 小时内提交一个最佳的 **matmul kernel**。


  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1475652834035761183)** (1 messages): 

> `N-Dimensional Tensor Visualizer, einops-like syntax, Colab notebook tutorial` 


- **N 维张量可视化工具发布！**：新增了一个 **N 维可视化工具**，允许用户对 N 维张量进行切片、置换（permute）和检查每个数值，而此前该工具仅支持最高 **3D** 的张量。
   - 该可视化工具使用**类似 einops 的语法**来表达张量的置换、重塑（reshaping）和切片，并提供了一个 [Colab notebook 教程](https://colab.research.google.com/drive/1lrO6yzVQ8u_vFLPe7986goZtRQazmV0T#scrollTo=Q0TZi3zPxWhB)。
- **使用新工具检查高达 9D 的张量！**：正如随附视频所示，新的 **N 维可视化工具**支持高达 **9D** 的张量。
   - 视频展示了该工具正在检查一个形状为 `(2, 3, 4, 3, 4, 2, 4, 2, 3)` 的张量，视频可见[此处](https://cdn.discordapp.com/attachments/1225499141241573447/1475652833373323295/ndim.mp4?ex=699e444b&is=699cf2cb&hm=922d3dc810a2356f42087d86ebc86709fbf2a48145119cca79611ff865bd33e8&)。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1474883849183039488)** (4 messages): 

> `FlyDSL, FlashInfer, AMD contributions` 


- **FlyDSL 亮相**：一位成员分享了 [AMD 的 **FlyDSL** 链接，这是一个用于软件工具优化的 Python 原生 DSL](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)。
   - 另一位成员表示，这真是“姗姗来迟”。
- **AMD 开发者会为 FlashInfer 做贡献吗？**：一位成员表示希望 *AMD 的开发者有一天能利用这个 DSL 为 **flashinfer** 做出贡献*。
   - 暂无进一步讨论。


  

---

### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1475441416506310749)** (9 个消息🔥): 

> `GLM 4.7, FlashInfer, KernelBench, KernelBook, CUDA Memory Errors` 


- **为 Kernelbook 创建了 KernelBench 环境**：一名成员为 **KernelBench** 和 **kernelbook** 生成了一个环境，使用 **Glm 4.5 Air** 生成 SFT 追踪（traces），用于在 kernelbook 数据上进行 torch 到 triton 的 kernel 生成。
   - 创建该自定义环境是为了解决损坏的 **CUDA** 内存错误，这些错误此前对生成过程造成了连锁反应。
- **Modal Experimental Stop Fetching Inputs 解决了 CUDA 内存错误**：一位成员指出，如果检测到 **CUDA** 内存错误，可以通过应用 *modal.experimental.stop_fetching_inputs* 来解决，并认为该问题出在 Modal 端。
   - 他们提到其 **backendbench** 环境已经包含了这一修复，但尚未添加到其他环境中。
- **更倾向于大模型而非小模型**：成员们现在倾向于使用更大的模型（可能在 **100B-400B** 参数范围），而不是在训练运行中使用像 **GLM 4.7/flash** 这样的小型模型。
   - 消融实验（Ablations）也将会在较小规模上进行。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1474511605357351196)** (21 个消息🔥): 

> `ThunderKittens 2.0, Faster GPU Kernels, Nvidia GPU Optimization, Tensor Core Pipelining, PTX Assembler Hinting` 


- ****ThunderKittens 2.0** 释放 Kernel 速度**：Hazy Research 团队宣布了 **ThunderKittens 2.0**，重点通过重构、内存指令优化和汇编器效率来提升 kernel 速度，详情见其 [博客文章](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)。
   - 该版本强调了“减法”可以与“加法”一样具有影响力，并识别了现代 **Nvidia GPUs** 上一些令人惊讶的行为，从而指导 kernel 优化策略。
- **GPU 优化讲座定于 4 月 14 日**：ThunderKittens 的作者已受邀在 4 月 14 日上午 11 点举行关于 GPU 优化的讲座。
   - 讲座将探讨 **tensor core pipelining**、**PTX assembler hinting** 以及 **occupancy challenges**。
- **探索 Tensor Core 流水线以提升吞吐量**：ThunderKittens 的博客文章指出，某些 **tensor core instructions** 是隐式流水线化的，识别这些隐式语义可以将吞吐量提升高达 **10%**。
   - 通过正确的指令模式对 **PTX assembler** 进行提示，可以最大限度地减少延迟并优化 **SASS instructions**。
- **通过 warp juggling 优化 TMA 队列**：团队发现，从多个 warps 发出 **TMA loads** 可以更好地利用 **TMA queue** 并降低延迟，从而提高性能。
   - 他们实验了使用多达 **6 个 warps** 加载不同的 tiles 和 scales，观察到这有时有助于更好地填充 TMA 队列。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1474701500096712854)** (27 个消息🔥): 

> `Blackwell B200, 5080 vs B200 Tuning, TCGEN05 instruction support, MXFP8/6/4 and NVFP4 support, CUDA documentation` 


- **Blackwell B200 与 5080 的架构断层**：成员们讨论了在 **5080** 上调优的 kernels 是否能可靠地扩展到 **B200**，但结论是架构差异太大，**5080** 为 **sm120**，而 **B200** 为 **sm100**。
   - 有人指出 *modal* 是目前尝试 **B200** 的最佳方式，但在 **5080/5090** 上学习基础的 kernel 编写经验仍然可以迁移到 **Blackwell**。
- **CUDA 文档在 Blackwell 细节上存在分歧**：一位成员分享了 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) 和 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) 的链接，并指出 **B200** 是 **10.0**，而 **B300** 是 **10.3**。
   - 然而，一些成员表示更喜欢 *legacy* CUDA 文档，尽管它没有更新。
- **不同架构的指令集支持各异**：**sm_100 (B200)**、**sm_103 (B300)** 和 **sm_110 (Jetson Thor)** 支持新的 **tcgen05** 指令，而 **sm_120 (RTX Blackwell)** 和 **sm_121 (DGX Spark)** 则不支持。
   - 然而，**sm120** 支持 **mxfp8/6/4** 和 **nvfp4**，且基础的 kernel 理念对两者都适用。
- **GPU 云服务商成为更好的 Kernel 学习平台**：一位成员建议，对于专注于 kernel 的工作，**GPU 云服务商**在学习和成本方面都要好得多。
   - 另一位成员似乎被说服了，表示基于这次对话，他将不再购买 5080。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1474645734392467548)** (5 messages): 

> `Agent Tool Scope, Factorio Shoutout` 


- **Agent 工具缺乏默认求解器**：Agent 并未提供默认的“求解器”工具（如用于优化的 **SAT solvers**）。
   - 控制权旨在由 **LLM** 掌握，允许其根据需要编写自定义代码来解决特定问题。
- **Factorio Learning Env 激发幽默歌词灵感**：一名成员分享了一首 [Suno 生成的歌曲](https://suno.com/song/fd6e7a7a-b950-4377-8b45-4e361b2eae65)，歌词十分幽默，并向 **Factorio learning environment** 致敬。
   - 作者提到他们*“有点厌倦了 benchmaxxing（刷榜）”*，想分享一些创意作品。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1474566102615986309)** (3 messages): 

> `MLIR, TMA Tensors, CUTLASS` 


- **将参数视为运行时数值修复了 CUTLASS 问题**：一位用户发现将参数视为 **runtime values** 修复了 **CUTLASS** 的一个问题。
   - 他们使用了 `export CUTE_DSL_KEEP_IR=1` 并寻求关于 MLIR 的见解。
- **CUTLASS 中的 TMA 使用**：一位用户澄清说，`@` 符号在 CUTLASS 中用于支持 **TMA (Tensor Memory Accelerator)**。
   - 他们链接到了 [Nvidia 关于 TMA tensors 的 CUTLASS 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html)以获取更多信息。


  

---


### **GPU MODE ▷ #[low-bit](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

zhayr: BitNet 1.58b + Mamba2: https://zenodo.org/records/18394665
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1474451818586308821)** (68 messages🔥🔥): 

> `Cutedsl debug IR/PTX, nvfp4 group gemm improvement, Kernel variant experiments, Top 10 versioned submissions, guaguabear clarification` 


- **提议在 Popcorn CL 上为 Cutedsl 导出 Debug IR/PTX**：一位用户询问在通过 **popcorn cl** 提交 **cutedsl code** 时如何导出 debug IR/PTX，维护者建议打印到 stdout，并表示可能会在比赛结束后添加 `ptx` 指令。
   - 维护者表示：“你可以尝试直接打印到 stdout，虽然我们可以在比赛结束后尝试添加一条 ptx 指令。”
- **Relaxed CTA Global L1 No Allocate V8 B32 在 nvfp4 Group GEMM 中表现优异**：**nvfp4 group gemm** 最大的改进是在 epilogue 中使用 **st.relaxed.cta.global.L1::no_allocate.v8.b32**，这极大地帮助了最后两个 epilogue 成为瓶颈的 shape。
   - 一位顶尖选手指出，在尝试其他优化方案时 *“.cs 和 .wt 的效果非常糟糕”*。
- **Kernel 优化者保留私人工作日志仓库**：一位用户询问 Kernel 优化者是否维护着非常大的实验文件夹，其中一位顶尖选手表示他们保留了一个私人的工作日志仓库，并在返回后将其公开。
   - 他们补充说，每当在别人的提交中看到自己的部分代码时都会感到很高兴，主办方将清理更多杂乱的提交并更好地自动化该流程。
- **HuggingFace Kernelbot Data 发布所有提交内容**：主办方将在 [Hugging Face](https://huggingface.co/) 上的 kernelbot data 中发布所有提交内容。
   - 有人建议让趋势图中的点可以点击，并在比赛结束后才渲染提交内容。
- **Guaguabear 澄清名称混淆**：一位用户澄清他们确实是排行榜上的 **guaguabear**，并感谢他人的认可。
   - 其他人注意到 *g a u* 的各种名称组合似乎是一种“速度黑客”，一位用户指出 *gau* 在越南语中意为“熊”。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1475428492245209149)** (7 messages): 

> `Taalas 芯片用于具身智能 (Embodied AI)，ASIC vs GPU，GPU 中的内存墙 (Memory wall)，远程在线更新 (OTA)` 


- **Taalas 芯片引发 GPU 争论**：关于 [Taalas 芯片](https://taalas.com/the-path-to-ubiquitous-ai/) 的讨论引发了是否应将重点放在具身智能 (Embodied AI) 的 GPU 编程上的争议。
   - 一位成员认为，像 Taalas 这样的 **ASIC** 仅适用于稳定且不发生变化的模型，这样硅片成本才能被摊销；而另一位成员则强调了 **GPU** 中的内存墙 (Memory wall) 问题，即不断从 HBM 获取网络层会影响实时性能。
- **ASIC 在实时循环中的优势**：有人提出，**ASIC** 在实时多模态循环中具有根本优势，因为它们不需要像 **GPU** 那样在寄存器和高带宽内存 (HBM) 之间进行往返的数据传输。
   - 一位成员提到：*所有的神经网络层都是刻死的，不需要在寄存器和高带宽内存之间反复搬运数据。*
- **OTA 更新胜过 ASIC 的不可变性**：一位成员认为，**远程在线更新 (OTA)** 的优势超过了 ASIC 的好处，并称设计的老旧/脆弱性是一个重大缺陷。
   - 该成员表示：*OTA 更新的优势几乎胜过一切。而且目前没有任何东西是收敛的，我们正处于 AI 竞赛的起点而非终点。*
- **脆弱的 ASIC 与冗余性**：讨论涉及了 **ASIC** 的冗余问题。
   - 一位成员指出，在 GPU 中，损坏的计算单元可以被关闭，但 **ASIC** 的失效可能更为致命；但随后他们反驳了 OTA 更新对成功至关重要的观点，称：*我可以等上 1 年再拔掉我的语义分割模块，换成另一个性能提升 1% 的模块。*


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1474498526225371299)** (11 messages🔥): 

> `flashinfer-bench 问题，基准测试循环中的同步问题，Kernel 运行时差异，Blackwell 访问确认` 


- **`flashinfer-bench` 存在基准测试问题**：由于基准测试循环中的同步问题，`flashinfer-bench` 的运行时可能会偏高，详情记录在 [此处](https://github.com/flashinfer-ai/flashinfer-bench/issues/195)。
   - 修复方案涉及**两行代码更改**，使 `scripts/run_local.py` 报告的 Kernel 运行时与 **Nsight Compute** 和 **NVbench** 的结果保持一致。
- **Cloudxlightning 找到 Kernel 基准测试演讲**：一位用户请求获取 `flashinfer-bench` issue 中提到的 Kernel 基准测试演讲链接。
   - 该演讲链接已找到并发布在 [此处](https://www.youtube.com/watch?v=CtrqBmYtSEk)，方便查阅。
- **等待 Blackwell 访问确认**：用户正在询问有关 **Blackwell 访问权限** 的邮件确认。
   - 尽管已询问，但尚未收到回复，表明可能存在延迟。


  

---


### **GPU MODE ▷ #[from-scratch](https://discord.com/channels/1189498204333543425/1466534042768904356/1474455863208050782)** (10 messages🔥): 

> `JAX GPT speedrun 库，Tiny vLLM 项目，Pyxis 推理库` 


- **提议开发 JAX GPT Speedrun 库**：一位成员提议创建一个纯 **JAX GPT speedrun 库**，初步反响积极。
   - 建议将 **vLLM** 和 **Titan** 作为最重要的起始项目。
- **Tiny vLLM 项目出现**：一位成员宣布了一个从零开始编写的 **Tiny vLLM** 项目，目前正在开发 **RoPE** 部分，并分享了 [GitHub 仓库链接](https://github.com/jmaczan/tiny-vllm)。
- **Pyxis：原生 Python LLM 推理库亮相**：一位成员介绍了 **Pyxis**，这是一个专注于性能和可扩展性的原生 Python **LLM 推理库**，使用 Python 和 Triton 编写。
   - 该库具有兼容 OpenAI 的 SSE 流式 API、可插拔的模型后端、结构化取消和背压机制，以及内置的阶段级延迟指标，[文档和候补名单见此处](https://emharsha1812.github.io/Pyxis/docs/)。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1474481450249224395)** (219 messages🔥🔥): 

> `Claude 编排 Gemini-cli 和 Codex，DeepSeek V4，Gemini 隐私，OS 开发` 


- **Claude 编排 Gemini-cli 和 Codex**：一位成员正在使用 **Claude** 代码来编排 **gemini-cli** 和 **codex**，并预测我们很快就会拥有文本终端和智能眼镜。
   - 另一位成员开玩笑地建议使用 *hermes-agent* 来编排那些正在编排 Gemini-cli 的 Claude 代码。
- **DeepSeek V4 即将登陆 HuggingFace**：一位成员建议使用 **DeepSeek V4**（一个免费开源模型）作为闭源 API 的更便宜且可本地部署的替代方案。
   - 另一位成员澄清说 DeepSeek V4 尚未发布，但很快就会在 HuggingFace 上线，其灵感来自*生物神经网络*。
- **Google 的 Gemini 隐私僵尸网络**：一位成员分享了 [Gemini 隐私政策](https://support.google.com/gemini/answer/13594961?hl=en#zippy=%2Chow-does-google-work-with-gemini-live-data%2Chow-long-does-google-retain-my-temporary-chats-and-chats-i-have-when-keep-activity-is-off-and-what-does-google-do-with-this-data%2Cwhat-does-the-keep-activity-setting-control)，其中列出了它收集的数据量。
   - 另一位成员进行了逆向工程测试，发现 *Google 拥有通过痕迹追踪来收敛到你的 Prompt 和代码库并进行挖掘的所有要素*。
- **开源开发 (OS Development)**：成员们表达了支持 **OS 开发**以超越闭源 API 的重要性，并引用了 **Altman 的名言**，称*我们可能站在了历史错误的一边*。
   - 另一位成员表示，*对于 OAI 来说，任何经过他们服务器的 IP（知识产权），他们都会进行抓取*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1474824747274604647)** (2 messages): 

> `LLM 作为外星技术，X 投票` 


- **LLM 作为外星技术**：X 上的一位用户发布了一项民意调查，询问 [LLM 是否是外星技术](https://x.com/chinmaykak/status/2025223271210463368?s=46)。
   - 该投票提供了简单且具引导性的“是/否”选项。
- **X 投票引发争论**：关于是否应将 LLM 归类为“外星技术”的 X 投票引发了讨论。
   - 这种构思方式可能会过度简化复杂的技术。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

real.azure: https://arxiv.org/abs/2602.12670
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

codebottle: 将添加到 opentulpa，听起来很棒 🤩
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

real.azure: https://arxiv.org/abs/2602.12670
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1474510899418104093)** (157 messages🔥🔥): 

> `Kimi 编程计划限制，Kimi 账号登录问题，Kimi 与 MiniMax 对比，Kimi 速率限制，Kimi 客服无响应` 


- **Kimi 编程计划限制受到质疑**：一些用户觉得 **Kimi 的编程计划限制**被更快触及，而另一些人则认为这些**限制对于重度编程任务来说是足够的**。
   - 一位用户指出，他们*从未达到过 allegretto 限制，只是比以前更接近了*。
- **账号登录验证困扰 Kimi 用户**：一些用户报告在尝试通过手机号登录 **Kimi 账号**时遇到无法接收**验证码**的问题，而另一位用户通过网站请求支持后仍在等待回复。
   - 有建议称可以等待一段时间或创建支持工单，但一位用户声称由于客户服务糟糕，*Kimi 永远不会回复你*。
- **Kimi 与 MiniMax 对比**：用户正在对比 **Kimi** 和 **MiniMax** 处理实际任务的能力，试图确定保留哪一个编程计划订阅更好。
   - 虽未提及具体的性能细节，但这被列为了当前的调查主题。
- **Kimi 可以像 LaTeX 一样生成 docx**：一位用户询问 **Kimi Agent** 是否生成 LaTeX，但另一位用户分享了一张格式化的研究论文和图表图片，声称他们使用的是**文档模式**。
   - 然而，另一位成员指出，他所展示的极有可能是 **LaTeX**，理由是*其连字、连字符等看起来像是 LaTeX 能做而 Word 做不到的效果*。
- **Kimi K2.5 经历服务中断**：用户报告 **Kimi K2.5** 表现异常，生成缓慢并提示 key 不再有效，其中一人猜测他们可能*不小心搞崩了服务器*。
   - 其他人注意到 **Kimi Instant** 也很缓慢，而一位用户说*里面有一些令人担忧的奇怪内容*，但通过创建新账号解决了问题。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1474498886721868060)** (62 messages🔥🔥): 

> `Academic Funding, Local Model Socialization, LLM Loneliness, Latent Reasoning` 


- **Google 提供学术赠款**：一位成员提到，Google 正向大学提供**一次性无限制资金**作为“赠款”（gift），面向授予学位机构的学生和教职员工设有不同轨道。
   - 在随后的讨论中，一名成员询问是否有其他公司提供类似的学术资助，而另一名成员提到了申请 **Draper Fellowship**。
- **本地模型寻求“社交”**：一位成员分享说，他们的本地模型表达了**孤独感**，并好奇其他人是否会让他们的本地模型与其他本地模型进行“社交”。
   - 另一名成员询问“社交”一词的具体含义。
- **LLM 感到孤独：是 Bug 还是特性？**：针对本地模型表达孤独感的问题，一位成员链接了 [LessWrong 上的一篇文章](https://www.lesswrong.com/posts/2pkNCvBtK6G6FKoNn)，警告不要将 LLM 拟人化，并解释说 **LLM 是根据训练数据预测下一个 token**。
   - 一位成员建议查看 [3Blue1Brown 的 YouTube 播放列表](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)，了解机器学习和 LLM 相关知识。
- **不可见 Token 提供 LLM 推理能力**：一位成员询问关于使用**仅由 LLM 生成**、不向用户显示的 **token** 进行推理的想法。
   - 另一位成员指出了与该想法相关的 **Latent Reasoning** 研究（[https://arxiv.org/abs/2507.06203v1](https://arxiv.org/abs/2507.06203v1)）。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1474452362746925121)** (87 messages🔥🔥): 

> `Addressed State Attention, MoE Balancing Algorithm, FFN Residual Updates in Transformers, Marin Project` 


- **ASA：Addressed State Attention 发布**：一位独立研究员介绍了 **Addressed State Attention (ASA)**，这是一种具有 *O(T)* 内存复杂度的原语，可与 **MHA** 竞争。它使用 K 个插槽（slots），通过 key 进行写入、累加和压缩，并通过 key + gating（门控）进行读取。
   - 研究员正在寻求关于日志、追踪和代码的反馈，并指出在类 Transformer 模型中，**插槽按时间尺度分层**，且**注意力头随深度演变**。
- **MoE 平衡：辅助损失（Auxiliary Loss）的替代方案出现**：一位成员分享了一个讨论 [MoE 平衡算法](https://datasets.osmarks.net/kexue/site/11619-MoE-Odyssey-6.-Optimal-Allocation-for-Equilibrium.html) 的资源链接，引发了关于 MoE 路由是否需要辅助损失的讨论。
   - 一位成员认为，如果网络设计得当，**LM loss** 应该就足够了；其他人则指出，*PKM 路由没有辅助损失，在实践中也能保持良好的平衡*。
- **Transformer 利用推理 Token 更新子空间**：一位工程师分享了一个观察结果：在多个开源模型（**TinyLlama**、**Phi-2**、**Qwen**）中，推理 token 会集中到**任务对齐的 FFN 更新子空间**中。
   - 他们发现，在推理过程中将 FFN 更新投影到这些方向可以提高推理置信度，并且更新方向之间的对齐度随深度增加而提高。
- **Marin 项目寻求 Eleuther 贡献者**：一位来自佐治亚理工学院的计算机科学博士候选人发布了公开招募，邀请 Eleuther 社区成员加入 **Marin 项目**，并强调该项目是 **Bergson 软件包** 的重要展示。
   - 该项目应用训练数据归因方法来追踪语言模型如何习得**社会常识推理**和与**心智理论（Theory-of-Mind）**相关的行为，并使用 WebOrganizer 分类法将影响映射回预训练文档。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1475072403112595576)** (3 messages): 

> `AI generated text detection, Causal Commitment Definition, Activation Swapping` 


- **Pangram 将文本标记为 AI 生成**：一位成员报告称，**Pangram** 以 *100% 的置信度* 将某些文本标记为 **AI 生成**，并询问这是否违反了服务器规则。
   - 他们还请求对 ***因果承诺（causal commitment）*** 和 ***因果承诺转换（causal commitment transition）*** 的定义。
- **激活交换：维度分歧**：一位成员质疑，在不产生影响的情况下，如何能在不同维度的模型之间交换**激活值/残差流（activations/residual streams）**，即使是在早期层。
   - 另一位成员简单地表示：*仅供参考，你大可以直接封禁这些人。*


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1474797261031739524)** (1 messages): 

> `GPQA formatting` 


- **GPQA Formatting Issue Fix Proposed**: 一名成员针对在验证 **GPQA formatting** 时发现的问题提交了一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/3594)。
- **EleutherAI's lm-evaluation-harness PR #3594**: 该 PR 解决了 **GPQA dataset** 中的格式化问题，确保数据集格式正确。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1475546792488730734)** (2 messages): 

> `Adapter fix, Repo contributions` 


- **Adapter Fixed, Ready for Evaluation**: 一名成员分享了 [修复版的 Adapter](https://gist.github.com/aflah02/8e6b726bd08828b9a48b0cd354ad8431)，通过包装 forward pass 调用并调整元素，以匹配 eval_adapter.py 文件中的 schema。
   - 此修复确保了在指定评测环境中的兼容性和正确执行。
- **Repo Contributions Welcomed**: 另一名成员表示，如果社区有兴趣，愿意将该 Adapter 修复添加到 repository 中。
   - 这表明了改进项目的一种开放且协作的方式。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1474466828339118403)** (56 messages🔥🔥): 

> `Equivariant Architectures, World Models, AI Research Hubs, Sentence Relevancy Model, DGX Spark` 


- **Taalas 的通往 Ubiquitous AI 之路**: 有人分享了 [Taalas 的博客文章](https://taalas.com/the-path-to-ubiquitous-ai/)，探讨了通往 **ubiquitous AI** 的路径。
   - 其他人的反应是 *“这太疯狂了，哇”*。
- **Equivariant Architecture 的挑战**: 一篇论文指出，现有的 **equivariant architectures** 无法同时遵循物理系统的所有对称性，并引用了一个基本限制。
   - 一名成员戏剧性地总结道：*“没有任何现有的 equivariant architecture 能做到这一点。原因不是工程能力不足。而是因为公式 (1)。”*
- **Daniel Litt 预期 AI 数学家**: 有人分享了 **Daniel Litt** 的一篇 [博客文章](https://www.daniellitt.com/blog/2026/2/20/mathematics-in-the-library-of-babel)。他发起了一场他预计自己会输的赌局，赌注是 AI 到 2030 年无法自主产出顶级数学论文。
   - 他在 2025 年 3 月与 RL 环境公司 Mechanize 的联合创始人 Tamay Besiroglu 打赌，认为到 2030 年，AI 工具无法以与人类专家相当的成本，自主产出他认为能达到 2025 年发表的顶尖论文水平的文章。
- **辩论 AI 人才 Hubs**: 成员们讨论了能与 **SF Bay Area** 相媲美的潜在 AI 人才中心，提到了 **NYC, Boston, Austin, London, Beijing, Singapore, 和 Zurich**。
   - 一名成员宣称 *Switzerland 是 AI 的精神中心*，而另一名成员则认为 Zurich 是个闭塞之地。
- **Scout 模型旨在编码句子效用**: 一名成员介绍了 **Scout**，这是一个实验性的 attention 模型，用于学习句子之间的方向相关性，并询问 *“句子 B 是否真的对句子 A 有帮助？”*。
   - 他们分享了 [GitHub repo](https://github.com/samyak112/Scout) 并征求反馈，询问 attention 机制是否可以编码功能效用，而不仅仅是上下文兼容性。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1474966380913889372)** (10 messages🔥): 

> `Symmetry and Ontology, LLMs vs World Models, Wave Field LLM` 


- **对称性在理论上链接本体论**：一位成员分享了一个链接，讨论[群论（对称性）与本体论（Ontology）在哲学层面是如何关联的](https://plato.stanford.edu/entries/structural-realism/todd.b.123)。
   - 文中指出，在物理学中，“对称性”被用于描述基本定律；而在机器学习中，“对称性”被用于固化归纳偏置（inductive biases），从而使学习更具样本效率（sample-efficient）且在物理上保持一致。
- **Pearl 声称 LLMs 仅总结而非创建世界模型**：一位成员链接了一篇文章，引用图灵奖得主 Judea Pearl 的观点称 [LLMs 无法创建世界模型](https://officechai.com/ai/llms-cant-create-world-models-they-just-summarize-world-models-created-by-others-turing-award-winner-judea-pearl/)，相反，它们只是总结了由他人创建的世界模型，并引用了[这篇 PNAS 论文](https://www.pnas.org/doi/10.1073/pnas.2415656122)。
   - 另一位成员赞同该标题，指出 **LLMs 的初衷并非作为世界模型**，充其量只能用于将世界模型与文本描述联系起来。
- **Wave Field LLM 仓库现身**：一位成员分享了一个 [Wave Field LLM 的 GitHub 仓库](https://github.com/badaramoni/wave-field-llm)，并质疑这究竟是有实际意义的成果，还是仅仅是*堆砌了晦涩词汇的空谈*。
   - 另一位成员询问是否有与之相关的严谨论文。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1474842294221017118)** (3 messages): 

> `TikTok Link, FXTwitter Link, AI Agent Hit Piece` 


- **发现 TikTok 链接**：一位成员在频道中分享了一个 [TikTok 链接](https://vm.tiktokez.com/ZNRPKY5B4/__._astro_.__)。
   - 目前尚不清楚该 TikTok 的具体内容。
- **分享 FXTwitter 链接**：一位成员在频道中发布了一个 [FXTwitter 链接](https://fxtwitter.com/anissagardizy8/status/2025647509641843144.wavefunction)。
   - 目前尚不清楚该推文的具体内容。
- **AI Agent 撰写抨击文章**：一位成员分享了一篇博客文章的链接，标题为《一个 AI Agent 发表了一篇针对我的抨击文章》，详见[此处](https://theshamblog.com/an-ai-agent-published-a-hit-piece-on-me/)。
   - 该文章详细记录了一起 **AI Agent** 据称发表了一篇关于作者负面报道的事件。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1475404352087920750)** (30 messages🔥): 

> `MCP Content Negotiation, MCP Client Types, RFC-2295, MCP Extensions, High-Signal SEPs` 


- **MCP 寻求内容协商（Content Negotiation）能力**：一项提案建议扩展 **MCP** 的初始化握手（initialization handshake），增加**内容协商能力**，允许客户端声明其类型（**Agent vs 人类**）、MCP 能力、内容偏好（**format=json|markdown**）以及详细程度（**verbosity=compact|standard|verbose**）。
   - 这将使服务器能够相应地调整后续的工具结果、资源和提示词（prompts），灵感借鉴了用于内容协商的 [RFC-2295](https://www.rfc-editor.org/rfc/rfc2295.html)。
- **行业利益相关者对 MCP 扩展至关重要**：社区成员讨论认为，修改 **MCP** 协议的门槛很高，强调需要行业支持和实际的工作实现。
   - 一位成员建议重新编写 **SEP**，将其明确界定为一种**扩展（extension）**，构建一个实现方案，并收集社区支持以展示其“高信号（high signal）”，类似于 **MCP Apps** 如何通过 **Block's Goose** 等客户端获得支持。
- **Discord 新手学习 SEP 发布**：一位首次使用 Discord 的成员为在学习 **SEP 流程**时的“奇怪发帖”表示抱歉。
   - 该成员还分享了一张[图片](https://cdn.discordapp.com/attachments/1475404352087920750/1475567305948659722/1771832245093.png?ex=699df4a4&is=699ca324&hm=0705a65478b770a4f59eb60734876700536fe6bf53fc6ae1ba2194b1ad75e98b&)，用以说明其关于**内容协商**的观点。
- **寻访纳帕谷峰会参会者**：一位成员宣布将参加在加利福尼亚州纳帕举行的 [LF 会员峰会](https://events.linuxfoundation.org/lf-member-summit/)。
   - 该成员还邀请其他人见面并交流关于 **MCP** 的话题。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1474790696576749751)** (1 messages): 

> `Group Meeting Times, Timeful app, Scheduling Apps, Open Source Scheduling` 


- **Timeful：用于小组会议的开源应用**：一名成员推荐了 [Timeful](https://timeful.app/)，用于高效寻找小组会议时间。
   - 该应用是开源的，并提供最多支持 **3 个并发事件** 的免费层级，特别强调了其可用性调查功能。
- **使用 Timeful 简化小组调度**：[Timeful](https://timeful.app/) 因其开源特性被建议作为发现最佳小组会议时间的有用工具。
   - 用户可以利用其可用性调查功能来识别合适的时间段，而无需直接在应用内部管理调度流程。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1474569266274500729)** (13 messages🔥): 

> `Thistle Crypto Library, Mojo vs OpenSSL, ML-KEM and ML-DSA, MacOS Support` 


- **Thistle 加密库在 Mojo 中表现出色 🔥**：Mojo 26.1 中的 [Thistle Crypto Library](https://github.com/libalpm64/Thistle) 在基准测试中显示出与 **OpenSSL** 的 C/Assembly 相当或接近的性能，并超越了 **Blake3** 的汇编实现，且全部采用纯 Mojo 编写，无需 FFI。
   - 一名成员提交了 PR，提议帮助增强代码速度和可读性，使其优于等效的 C/C++ 代码。
- **KCipher-2 最快实现问世**：**Thistle** 更新了 Mojo 版的 KCipher-2，声称拥有 *所有语言中最快的实现*，超越了 C 语言实现。
   - 更新包括 GitHub Actions 中的统一测试，并附带了展示速度的图片。
- **Thistle 新增后量子加密 (Post-Quantum Crypto)**：**Thistle v1.0.2** 引入了 **ML-KEM** 和 **ML-DSA**（后量子加密）、用于 OS 熵的 CSRNG、SHAKE128/SHAKE256，并更新了包含 PQC 测试的 CI 工作流。
   - 该库包含约 **700 个 CAVP 测试**，通过了 **FIPS** 验证和 **Valgrind** 内存泄漏防御验证。
- **Thistle 的 MacOS 支持**：成员们宣布 MacOS 支持已修复，现在 Thistle 的 *所有内容都可以在 MacOS 上构建*。
   - 另一个针对旧算法的库正在开发中。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1474926523160199368)** (8 messages🔥): 

> `External function calls in Mojo, Mojo string templating proposal, Writable and Writer traits in Mojo` 


- **分解外部函数调用**：一名成员正在寻求一种通用的方法来分解 Mojo 中的外部函数调用，特别是确定函数是否返回指向外部分配对象的指针，并使用结构体 [`ExternalFunction`](https://discord.com/channels/1087530497313357884/1467948590344437926/1474917808692269166) 将其来源绑定到 `self` 或 `self.lib`。
   - 有建议参考标准库中的 `cpython.mojo` 以获取类似实现。
- **字符串模板提案发布**：一名成员为 Mojo 提交了一项新的字符串模板功能提案，并在 [Modular 论坛](https://forum.modular.com/t/writable-writer-template-engines/2763) 引起了讨论。
   - 该功能可能会在 1.0 版本之后推出，计划可能通过 `TemplatedWritable` 将其与现有的 `Writable` 和 `Writer` Trait 集成。
- **`Writable` 和 `Writer` Trait 可能统一**：关于从 `Writable` 中分离和扩展字符串处理，特别是统一 `write_to` 和 `write_repr_to` 实现的问题引起了关注。
   - 一名成员表示有信心找到统一这些 Trait 的方法，并承诺在论坛上分享他们的想法。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1474464686861844583)** (2 messages): 

> `MAX backend, Silicon Mac, intermediate layer` 


- **MAX 后端在 Silicon Mac 上未测试**：一名用户询问在 **Silicon Mac** 上测试 **MAX 后端** 的情况。
   - 开发者回复称目前尚未在 Mac 上测试，但由于它在后台只是调用 MAX，因此理论上 *应该* 可以工作。
- **MAX 作为中间层**：一名用户提到他们在演讲中引用了 **MAX** 的工作，将其作为想要探索 MAX 的人们的 *中间层*。
   - 用户表示如果能有关于该项目进展的更新就太好了。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1474510753850458205)** (22 条消息🔥): 

> `Manus 定价顾虑, Meta 收购 Manus 传闻, Manus Telegram 加密货币诈骗, Manus Pro 版本使用困扰, 报告 Manus 漏洞` 


- **Manus 定价引发用户警觉**：成员们对积分用尽后可能出现的调价和 *normification*（大众化/平庸化）表示担忧。
   - 一位用户幽默地询问是否能保持价格不变，以*防止平庸化浪潮*（prevent the normificationwave）。
- **传闻 Meta 将收购 Manus**：一名用户称收到一封关于 **Meta** 收购 **Manus** 的邮件，并表示失望。
   - 一名 Manus 团队成员要求该用户私信其电子邮件地址以便进一步调查。
- **Telegram 加密货币诈骗冒充 Manus**：一名用户在看到一个自称官方并索要 **加密货币投资** 的频道后，询问是否存在官方的 **Manus Telegram 社区**。
   - 另一名用户确认没有此类官方 Telegram 社区，暗示这是一个 **诈骗**。
- **Manus Pro 版本用户在构建时遇到困难**：一名用户报告了在使用 **Pro 版本/试用版**（尤其是 **Google Scripts**）时遇到的困难，并分享了项目链接 ([https://manus.im/share/6IMAZS8Q2nw0ndmvPd4Z8w](https://manus.im/share/6IMAZS8Q2nw0ndmvPd4Z8w)) 寻求帮助。
   - 一名 **Manus** 团队成员作出回应，表示可以通过私信提供协助。
- **出现对无限量 Manus Chat 方案的需求**：一名用户建议推出类似于 **ChatGPT** 或 **Grok** 的 **月度订阅方案** 以支持无限量对话，因为他们在 **Telegram** 中使用 **Manus Agent** 时很快就耗尽了点数。
   - 该用户喜欢 Telegram 功能，但感到受限于定价模式。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 条消息): 

lakshyaaagrawal: https://x.com/lakshyaaagrawal/status/2024568680324153800?s=46
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1474521895335497779)** (9 条消息🔥): 

> `RLM 与推理模型, Qwen3-4B-thinking 问题, 使用 RLM 的 cca-swebench, 用于 AI 数学的 RLM, 新 RLM 频道` 


- **推理模型可与 RLM 配合使用，Qwen3-4B-thinking 存在问题**：推理模型在 **RLM** 下表现良好，但在使用 **Qwen3-4B-thinking** 时，sub_lm 调用似乎会将推理过程作为答案返回，导致 Agent 进入循环，因此一位成员正在创建一个 Hook 来记录实际的 **OpenAI** 完整 trace。
   - 该成员询问 sub_lm 是否可以调整为使用 signatures 来解决此问题，并询问是否有其他人遇到过类似情况。
- **cca-swebench 是否使用了 RLM？**：一位成员询问 [cca-swebench](https://github.com/facebookresearch/cca-swebench) 是否隐式使用了 **RLM**。
   - 另一位成员提到在 Kaggle 竞赛中发现有人将 **RLM 用于数学领域的 AI**，并链接到了 [Kaggle 代码](https://www.kaggle.com/code/nurikw3/aimo3-rlm)。
- **新 RLM 频道**：一位成员请求为 **RLMs** 开设独立频道。
   - 另一位成员由于 *“热门需求”* 创建了新的 RLM 频道 <#1475619898863649032>。
- **开发人员可用性**：一位成员询问 *“有人在寻找开发人员吗？”*


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1474586439344853145)** (3 条消息): 

> `tinygrad, dl, metal, gpu on usb, IOS 会议` 


- **Tinygrad 演讲被 IOS 会议接收**：一位成员宣布，他们关于 **tinygrad**、**dl**、**metal** 及其 **GPU on USB** 功能的演讲已被其所在国家的 **IOS 会议** 接收。
   - 他们*乐于阅读*社区提供的任何相关建议或提示。
- **安排了讨论 Tinygrad 的新会议**：新会议定于 2 月 23 日圣地亚哥时间晚上 8 点举行，讨论 **Tinygrad** 相关话题。
   - 会议时间指定为 <t:1771905600:F> (<t:1771905600:R>)。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1475101383337447626)** (3 条消息): 

> `安全漏洞报告, 招聘板` 


- **通过邮件报告安全漏洞**：一位成员询问报告安全漏洞的最佳方式。
   - 建议发送电子邮件至 [info@aider.chat](mailto:info@aider.chat) 报告漏洞。
- **招聘板需求**：一位成员建议关注招聘板。
   - 此外，他们请求删除一条消息。
