---
companies:
- nous-research
- anthropic
date: '2026-04-06T05:44:39.731046Z'
description: '**Hermes Agent** 作为领先的开放智能体堆栈，因其自我改进技能、持久记忆和自我改进循环等特性而备受瞩目。其新增的 **Manim
  技能** 支持生成数学及技术动画，进一步拓展了智能体的能力边界。Hermes 生态系统正在迅速壮大，涵盖了 GUI 工具、WebUI、HUD 更新、OAuth
  支持及各项集成。此外，一场针对智能体的“开放训练数据运动”正在兴起，专注于共享可复用的行为数据和基准测试追踪（harness traces）。


  与此同时，Anthropic 的 **Claude Code** 正面临分发与政策方面的挑战。据报道，相关限制和不稳定性正影响着第三方编程智能体，凸显了全天候在线（always-on）智能体在订阅制经济模型下存在的问题。社区中的核心观点包括：“如果用
  Claude Code 分析其自身源码会报错”以及“基本无法使用”。'
id: MjAyNS0x
models: []
people:
- theo
- clementdelangue
- badlogicgames
- yuchenj_uw
title: 今天没发生什么事。
topics:
- self-improving-skills
- agent-architecture
- memory-persistence
- animation-generation
- open-training-data
- coding-agents
- subscription-models
- policy-restrictions
---

**平静的一天。**

> 2026年4月4日至4月6日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 且没有检查更多 Discords。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[自行选择/取消](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 综述

**Gemma 4 的快速本地化采用与端侧开源模型时刻**

- **Gemma 4 正在推动一波强劲的“本地优先”浪潮**：多条帖子指出 Gemma 4 已成为 Hugging Face 上最热门/排名第一的模型，人们对其端侧实用性而非仅仅是排行榜表现表现出极大的热情——参见 [@ClementDelangue](https://x.com/ClementDelangue/status/2040911131108069692)、[@GlennCameronjr](https://x.com/GlennCameronjr/status/2040529333794824456) 和 [@Yampeleg](https://x.com/Yampeleg/status/2040495537598648357)。最强烈的信号是人们在消费级 Apple 硬件上运行它的速度之快：[@adrgrondin](https://x.com/adrgrondin/status/2040512861953270226) 展示了在 **iPhone 17 Pro** 上通过 **MLX** 运行 **Gemma 4 E2B**，速度约为 **40 tok/s**；[@enjojoyy](https://x.com/enjojoyy/status/2040563245925151229) 报告了类似的 iPhone 部署；[@_philschmid](https://x.com/_philschmid/status/2041171039598543064) 强调了在 **AI Edge Gallery** 中使用技能进行 Wikipedia 查询的 Gemma 4 E2B。Red Hat 还发布了 **NVFP4** 和 **FP8-block** 格式的 **量化版 Gemma 4 31B** 模型权重卡（model cards），其指令遵循（instruction-following）评估已上线，推理/视觉评估正通过 [@RedHat_AI](https://x.com/RedHat_AI/status/2040766645480628589) 待定发布。这些动态共同表明，Gemma 4 不仅仅是又一个开源发布，它正在成为 **边缘推理（edge inference）、Apple Silicon 工具链和低门槛本地部署** 的标杆。

- **商业影响是对付费聊天订阅和云端依赖产生的压力**：一些更具病毒性的评论虽然简略，但捕捉到了真实的转变。[@AlexEngineerAI](https://x.com/AlexEngineerAI/status/2040260903053197525) 认为本地运行的 Gemma 4 已经弥补了足够多的差距，使得对于某些用户来说 Claude 订阅的吸引力大打折扣，而 [@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2040454752534761725) 提醒人们 **HF 托管的模型是免费使用的**，并且可以取代 Agent 工作流的部分环节。在基础设施方面，[@ollama](https://x.com/ollama/status/2041238722914685336) 推出了基于 **NVIDIA Blackwell GPU** 的 **Ollama Cloud 上的 Gemma 4**，使其无需自托管即可用于 OpenClaw 和 Claude 风格的工作流。[@osanseviero](https://x.com/osanseviero/status/2041154555530932578) 发布的一条值得关注的生态系统帖子也强调了此次发布协调范围之广——包括 **HF, vLLM, llama.cpp, Ollama, NVIDIA, Unsloth, SGLang, Docker, Cloudflare** 等——这提醒人们，“开源模型的成功”日益依赖于 **同步的下游系统支持**，而不仅仅是权重本身。

**Hermes Agent 的自我改进 Agent 循环、OpenClaw 阻力以及对开放追踪数据（Trace Data）的推动**

- **Hermes Agent 是本批次中最受关注的 Agent 框架故事**：核心叙事是 Nous 的系统通过结合 **持久化记忆（persistent memory）**、**自我生成/优化的技能** 以及更具主见（opinionated）的自我改进循环，正在赢得市场关注度。[@NousResearch](https://x.com/NousResearch/status/2040931043658567916) 发布的 **Manim 技能** 特别引起共鸣，因为它展示了一个能立即产出清晰可见成果（技术动画和讲解视频）的 Agent 技能，而不是又一个 PDF 总结器。[@ErickSky](https://x.com/ErickSky/status/2040956335764734235)、[@lucatac0](https://x.com/lucatac0/status/2041018088913608923)、[@Sentdex](https://x.com/Sentdex/status/2041165530812334417)、[@casper_hansen_](https://x.com/casper_hansen_/status/2041046264758858081) 和 [@noctus91](https://x.com/noctus91/status/2041084870722793707) 的演示和反应进一步放大了这一影响。来自 [@Teknium](https://x.com/Teknium/status/2041233409901769133) 的产品更新增加了针对 Discord/Telegram 机器人的 **斜杠命令技能加载**，而像 **Hermes HUD** 这样的社区工具则通过 [@aijoey](https://x.com/aijoey/status/2040978270439580042) 将实时进程映射到 tmux 窗格并显现审批流程，此外 [@Teknium](https://x.com/Teknium/status/2040998328461316524)、[@nesquena](https://x.com/nesquena/status/2041000592215298123) 和 [@magiknono](https://x.com/magiknono/status/2040524343973740584) 也推出了多个 WebUI 集成。

- **与 OpenClaw 的对比集中在架构和商业模式的脆弱性上**：多篇帖子直接对比了两者。[@TheTuringPost](https://x.com/TheTuringPost/status/2040936147720048909) 将其区别总结为**人工编写的技能 vs 自我形成的技能**、**Markdown 记忆 vs 持久化/可搜索的记忆栈**，以及**网关控制平面 vs 自我提升循环**。这种定调得到了 [@SnuuzyP](https://x.com/SnuuzyP/status/2040999794894663996)、[@DoctaDG](https://x.com/DoctaDG/status/2041051272560923090) 和 [@spideystreet](https://x.com/spideystreet/status/2041172439468511266) 等从业者的共鸣，他们中许多人提到上手更简单，且减少了手动的技能微调。背景是用户对 Claude 订阅限制和运行时间日益增长的挫败感：[@theo](https://x.com/theo/status/2041016477047034012) 报告了 Claude Code 在分析自身源码时出错；[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2041187141523526011) 和 [@ratlimit](https://x.com/ratlimit/status/2040787102078546068) 强调了停机问题；[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2041202983640432966) 认为 **$20/$200 的订阅模式在结构上与 24/7 全天候的 Agent 工作负载不匹配**。这种经济层面的批评有助于解释 [@NousResearch](https://x.com/NousResearch/status/2040471903433896328) “**开源是必然的 (Open Source is inevitable)**” 这一观点背后的舆论势头。

- **一个更重要的长期线索是开放 Agent 数据**：[@badlogicgames](https://x.com/badlogicgames/status/2040979640265633882) 发布了 **pi-share-hf**，用于将编程 Agent 会话发布为带有 PII 防护的 Hugging Face 数据集，随后通过 [@badlogicgames](https://x.com/badlogicgames/status/2041151967695634619) 发布了他自己的会话。[@ClementDelangue](https://x.com/ClementDelangue/status/2041189872556269697) 明确将其定位于**开源前沿 Agent** 所缺失的关键要素：社区已经在生成轨迹 (traces)，因此应该众包数据集。这与 [@salman_paracha](https://x.com/salman_paracha/status/2040215191678509521) 关于 Agent 交互轨迹采样/筛选的 **Signals** 论文，以及 Baseten 的观点（即自我提升模型应该直接从**记录的生产轨迹**中学习，而不是依赖干净的沙箱，通过 [@baseten](https://x.com/baseten/status/2041194606512279617) 提出）完美衔接。这可以说是目前最具技术实质的“Agent”趋势：不仅是更好的测试框架 (harnesses)，而且是一个围绕**轨迹捕获、整理以及从真实使用中训练**的新兴技术栈。

**新研究信号：RL、路由、Agent 评估和小型专业化模型**

- **后训练 (Post-training) 和 RL 效率仍是实质性的活跃领域**：[@TheTuringPost](https://x.com/TheTuringPost/status/2040389184234651815) 强调了阿里巴巴 Qwen 的 **FIPO** (**Future-KL Influenced Policy Optimization**)，该方法为强烈影响未来步骤的 tokens 分配更多权重；报告的结果包括推理轨迹从约 **4K 扩展到 10K+ tokens**，以及 **AIME** 的得分从 50% 左右提升到 **~56–58%**，领先于提及的 DeepSeekR1-Zero-Math，并根据配置不同达到或超过了 o1-mini。[@finbarrtimbers](https://x.com/finbarrtimbers/status/2041176604961878271) 撰文描述了 **OLMo 3** 如何从同步 RL 转向**异步 RL**，实现了 **4 倍的吞吐量**提升 (tokens/sec)。其他值得注意的论文指向包括来自 [@_akhaliq](https://x.com/_akhaliq/status/2041183818317509028) 和 [@HuggingPapers](https://x.com/HuggingPapers/status/2041188981195391447) 的 **Self-Distilled RLVR / RLSD**，以及来自 [@TheAITimeline](https://x.com/TheAITimeline/status/2040953557961080843) 的 **Path-Constrained MoE**，后者通过约束跨层的路由路径来提高统计效率，并消除辅助负载均衡损失。

- **Agent 和基准测试 (Benchmark) 研究正在摆脱玩具任务**：[@GeZhang86038849](https://x.com/GeZhang86038849/status/2041184352516919690) 推出了 **XpertBench**，明确针对**专家级、开放式工作流评估**，而非已经饱和的考试类基准测试。[@TheTuringPost](https://x.com/TheTuringPost/status/2041124796361236608) 分享了一项关于工具使用的综述，涵盖了从单函数调用到**长程编排 (long-horizon orchestration)**、重新规划、反馈循环以及延迟/成本预算等效率问题的进展。在数据/企业工作流方面，[@CShorten30](https://x.com/CShorten30/status/2041154055993430365) 指出了 Shreya Shankar 针对异构数据库系统多步查询的 **Data Agent Benchmark**。这些都表明评估设计正在赶上生产环境 Agent 构建者所关注的核心：**工作流完成度、歧义处理、编排质量和成本**。

- **小型专用模型继续通过强有力的案例研究证明其价值**：[@DavidGFar](https://x.com/DavidGFar/status/2041063368656585002) 发布了 **SauerkrautLM-Doom-MultiVec-1.3M**，这是一个拥有 **1.3M 参数的 ModernBERT-Hash** 模型，在 **31K 人类游戏帧**上进行了训练。该模型在 VizDoom 任务上的表现超越了参数量大得多的 API 级 LLM，且在 **CPU 上的运行耗时仅为 31 ms**。虽然应用场景较窄，但其核心观点很重要：在延迟和架构比广泛的世界知识更重要的**实时控制任务**中，范围适度的模型可以占据主导地位。与此相关，[@MaziyarPanahi](https://x.com/MaziyarPanahi/status/2040776481673281936) 推出了 **Falcon Perception**，这是一个 **0.6B** 参数、面向分割（segmentation）的视觉语言模型，据其对比测试称性能超越了 SAM 3，并能在搭载 MLX 的 MacBook 上运行；[@Prince_Canuma](https://x.com/Prince_Canuma/status/2040861768138789012) 和 [@ivanfioravanti](https://x.com/ivanfioravanti/status/2040886300971004270) 也表达了类似观点。反复出现的主题是：**专业化 + 更好的系统适配**可以击败通用的规模扩张。

**OpenAI 和 Anthropic：政策信号、治理审查与算力经济学**

- **OpenAI 最大的公开动作是政治性的，而非产品层面的**：该公司及其盟友推动了一个全新的**“智能时代的工业政策（Industrial Policy for the Intelligence Age）”**框架，由 [@kimmonismus](https://x.com/kimmonismus/status/2041130939175284910)、[@OpenAINewsroom](https://x.com/OpenAINewsroom/status/2041198359420215453) 和 [@AdrienLE](https://x.com/AdrienLE/status/2041179073167454689) 进行了总结。核心想法包括：**公共财富基金**、**可移植福利**、**32 小时工作周试点**、**AI 权利**、更强大的溯源/审计基础设施，以及针对已发布的危险模型的遏制方案。值得注意的战略信号是，OpenAI 现在公开主张将向**超级智能（superintelligence）**的过渡视为一个积极的政策问题，而非遥远的假设。外界对此反应不一：有人认为它在应对冲击方面异乎寻常地坦诚，也有人认为这过于超前或出于政治投机，例如 [@Dan_Jeffries1](https://x.com/Dan_Jeffries1/status/2041170970631676067) 和 [@jeremyslevin](https://x.com/jeremyslevin/status/2041182591546531924)。此外，OpenAI 还通过 [@OpenAI](https://x.com/OpenAI/status/2041202511647019251) 和 [@markchen90](https://x.com/markchen90/status/2041250842255425767) 启动了 **Safety Fellowship**。

- **与此同时，针对 Sam Altman 和 OpenAI 治理的审查急剧加强**：一份重磅的《纽约客》调查报告由 [@RonanFarrow](https://x.com/RonanFarrow/status/2041213917611856067)、[@NewYorker](https://x.com/NewYorker/status/2041111369655964012) 以及 [@ohryansbelt](https://x.com/ohryansbelt/status/2041151473984123274) 等社区总结进一步放大。报道重新审视了 2023 年的解雇/复职风波，并声称存在内部备忘录、欺骗指控、董事会操纵、安全流程担忧以及对超级对齐（superalignment）投入资源不足等问题。OpenAI 方面的反击来自 [@tszzl](https://x.com/tszzl/status/2041265558054965534)，他表示对齐团队仍是公司内部规模最大、算力资源最丰富的项目之一。另外，[@anissagardizy8](https://x.com/anissagardizy8/status/2040894109817393240) 和 [@kimmonismus](https://x.com/kimmonismus/status/2041100365303808069) 报道了 Altman 与首席财务官 **Sarah Friar** 之间的紧张关系，特别是在算力开支和 IPO 准备方面。

- **Anthropic 的对应举措体现在算力和收入规模上**：[@AnthropicAI](https://x.com/AnthropicAI/status/2041275561704931636) 宣布与 **Google 和 Broadcom** 达成协议，将从 **2027年** 开始上线**数千兆瓦（GW）的下一代 TPU 产能**，用于训练和服务前沿的 Claude 模型。Anthropic 还通过 [@AnthropicAI](https://x.com/AnthropicAI/status/2041275563466502560) 表示，其年化收入已超过 **300 亿美元**，高于 2025 年底的 **90 亿美元**。这与关于前沿实验室经济压力的报道相呼应：[@kimmonismus](https://x.com/kimmonismus/status/2041203798723666375) 引用《华尔街日报》的报道称，尽管收入正在爆发式增长，但**训练和推理成本依然巨大**，OpenAI 预计到 **2028 年算力支出将达到 1210 亿美元**。对于工程师来说，实际的启示很直接：前沿竞赛的瓶颈越来越不只是模型构思，而是**资本结构、长期算力合同和服务经济学**。

**系统与基础设施：更快的 RL、更快的 MoE 解码、更好的 GPU/边缘工具**

- **几篇帖子对系统性能提升的描述异常具体**：[@cursor_ai](https://x.com/cursor_ai/status/2041260649267986643) 报告称，通过 “warp decode” 技术，在 Blackwell GPUs 上的 MoE Token 生成速度提升了 1.84 倍，且输出质量有所提高，这一结果直接归功于更频繁的 Composer 模型更新。[@tri_dao](https://x.com/tri_dao/status/2041191260682150048) 指出，针对消费级 Blackwell 显卡的快速 Muon optimizer 路径即将推出，因为其实现方式被表达为 **matmul + epilogue**，从而允许复用 mainloop 的工作。在 RL 方面，[@finbarrtimbers](https://x.com/finbarrtimbers/status/2041176604961878271) 发布了一篇罕见的工程总结，介绍了如何将 OLMo 3 的 RL 栈异步化，从而实现了 **4 倍的 throughput** 提升。

- **Apple/本地技术栈以及训练/推理教育生态也在持续改进**：[@josephjojoe](https://x.com/josephjojoe/status/2041215366177636468) 开源了用于 Apple Silicon 上蛋白质建模的 **ESM-2 MLX** 移植版本，扩展了本地 bio-LLM 的实验范畴。[@rasbt](https://x.com/rasbt/status/2041140643959885999) 为 **LLM Architecture Gallery** 添加了 RSS 订阅源，这是一个虽小但实用的功能改进，方便追踪模型设计。[@UnslothAI](https://x.com/UnslothAI/status/2041177756848083266) 表示其免费 notebook 现在可以训练/运行 **500 多种模型**。为了更深入地理解系统，[@levidiamode](https://x.com/levidiamode/status/2041229052804280811) 赞扬了 Hugging Face 的 **Ultra-Scale Playbook**，该手册统一了 **DP/TP/PP/EP/context parallelism**，并提供了在多达 **512 个 GPUs** 上的实证扩展证据。

**热门推文（按互动量排序）**

- **Gemma 4 设备端演示**：[@adrgrondin](https://x.com/adrgrondin/status/2040512861953270226) 展示了在 **iPhone 17 Pro** 上使用 MLX 运行 **Gemma 4 E2B**，速度约为 **40 tok/s**，是当天最引人注目的技术爆款。
- **Claude 订阅与本地开源模型替代**：[@AlexEngineerAI](https://x.com/AlexEngineerAI/status/2040260903053197525) 捕捉到了这样一种情绪：本地开源模型现在对于许多工作流来说已经“足够好”了。
- **开源立场**：[@NousResearch](https://x.com/NousResearch/status/2040471903433896328) 用“**开源是不可阻挡的 (Open Source is inevitable)**”概括了更广泛的运动。
- **Claude 宕机与限制反弹**：[@ratlimit](https://x.com/ratlimit/status/2040787102078546068)、[@theo](https://x.com/theo/status/2041111862113444221) 和 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2041202983640432966) 共同将运行时间 (uptime) 和订阅经济学推向了主流工程界的投诉热点。
- **OpenAI 治理调查**：[@RonanFarrow](https://x.com/RonanFarrow/status/2041213917611856067) 和 [@ohryansbelt](https://x.com/ohryansbelt/status/2041151473984123274) 推动了当天最大的技术相关企业治理新闻。
- **Anthropic 计算规模**：[@AnthropicAI](https://x.com/AnthropicAI/status/2041275561704931636) 宣布了**多吉瓦 (multi-gigawatt) 级别的 TPU 产能**，且 [@AnthropicAI](https://x.com/AnthropicAI/status/2041275563466502560) 引用了 **300 亿美元的年化收入**，这些是前沿实验室规模的最明确信号。

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM 综述

### 1. Gemma 4 模型发布与基准测试

  - **[发布 Google DeepMind 的 Gemma 4 所经历的过程](https://www.reddit.com/r/LocalLLaMA/comments/1se6nq5/what_it_took_to_launch_google_deepminds_gemma_4/)** (热度: 664)：**该图强调了发布 Google DeepMind 的 Gemma 4 模型所需的协作努力，涉及与 Hugging Face (HF)、VLLM、llama.cpp、Ollama、NVIDIA、Unsloth、Cactus、SGLang、Docker 和 CloudFlare 等组织的合作伙伴关系。这凸显了现代 AI 生态系统的复杂性和相互依赖性，多个技术和平台必须协同工作以支持像 Gemma 4 这样的先进模型。这次发布反映了跨多个技术领域的重大集成努力。** 一条值得注意的评论讨论了最新 LM Studio beta 版中的推理 bug，特别是针对 Gemma 4 模型的随机拼写错误和 Token 过度生成。这表明模型部署中仍存在挑战，需要进一步优化。

    - x0wl 指出了在最新的 LM Studio beta 版中使用 Google DeepMind 的 Gemma 4 模型时存在的推理 bug，具体提到了“随机拼写错误”和“未关闭 think 标签”等问题。该用户使用的是官方 Gemma 4 26B A4B @ Q4_K_M 以及 Q8 KV quantization，并指出这些问题出现在 llama.cpp commit 277ff5f 和 runtime 2.11.0 版本中。
    - Embarrassed_Adagio28 表达了对当前问题解决以及 Gemma 4 31B 改进型 Agentic 编程设置发布的期待。他们认为一旦配置得当，该模型可能会非常高效，但在那之前，他们更倾向于使用 Qwen 3 coder，这表明了在当前任务中对稳定性和性能的偏好。

- **[[PokeClaw] 第一个使用 Gemma 4 自主控制 Android 手机的可用应用。完全端侧运行，无需云端。](https://www.reddit.com/r/LocalLLaMA/comments/1sdv3lo/pokeclaw_first_working_app_that_uses_gemma_4_to/)** (活跃度: 489): **该图片展示了 **PokeClaw** 的界面，这是一款利用 **Gemma 4** 在完全端侧自主控制 Android 手机的创新应用，无需依赖云端服务。这个仅用两天开发的开源原型展示了一个闭环 AI 系统，能够执行诸如直接从屏幕读取对话上下文并自动回复消息等任务。该应用的最新更新 (v0.2.x) 提升了上下文理解能力，并增加了更新检查功能。该项目托管在 [GitHub](https://github.com/agents-io/PokeClaw) 上，邀请用户通过报告问题或给仓库点赞 (star) 来做出贡献。** 评论者对应用名称 "PokeClaw" 感到好奇，期待它与 Pokémon 有所关联，并对应用的安全性以及自主控制的潜在风险表示担忧。

    - 使用 **Gemma 4** 进行完全端侧控制被强调为运行时安全的一项显著优势，因为它确保所有操作都在本地处理，而不依赖云端服务。这种方法让用户能够完全控制自己的数据和行为，减少了与云端处理相关的潜在安全风险。
    - 技术建议是对辅助功能 (accessibility features) 中的边缘情况进行彻底测试。这对于防止由于与设备的辅助功能设置产生不可预见的交互而可能发生的任何意外操作至关重要，因为这可能导致意外行为或安全漏洞。
    - 使用 **Gemma 4** 实现的自主控制因其通过避免云端依赖来维护用户隐私和安全而受到赞赏。然而，严谨测试的重要性被再次强调，以确保应用程序在所有场景下都能按预期运行，特别是在处理消息监控和自动回复等敏感任务时。

  - **[Gemma 4 刚刚随手击败了我们排行榜上除了 Opus 4.6 和 GPT-5.2 之外的所有模型。31B 参数，每次运行成本 $0.20](https://www.reddit.com/r/LocalLLaMA/comments/1sdcotc/gemma_4_just_casually_destroyed_every_model_on/)** (活跃度: 2056): **该图片展示了 **Gemma 4**（一个 31B 参数的模型）的性能，它在 FoodTruck Bench 排行榜上排名第三，在 30 天内实现了 `$24,878` 的净值、`+1144%` 的 ROI 以及 `46%` 的利润率，每次运行成本为 `$0.20`。该模型在性价比方面超越了包括 **GPT-5.2** 和 **Gemini 3 Pro** 在内的其他几个模型，只有 **Opus 4.6** 以显著更高的 `$36` 运行成本超过了它。Gemma 4 的 **26B A4B 变体** 虽然更便宜，但由于 JSON 格式问题需要自定义输出清理，这影响了它在 Agent 工作流中的可用性。** 一位评论者指出结果页面缺少推理成本列，认为这是一个有用的补充。另一位用户提到，Gemma 4 在诊断 PLC 代码方面表现不佳，而 Qwen-Coder-Next 在该领域更为有效。

    - Recoil42 指出结果页面缺少推理成本列，并建议包含这一指标将有助于更全面地评估模型性能。这可以帮助用户更好地理解运行不同模型的成本效益，特别是在将 Gemma 4 与 Opus 4.6 和 GPT-5.2 等模型进行比较时。
    - Adventurous-Paper566 强调了 Gemma 4 的实际性能，指出它可以在 32GB 的 VRAM 上运行，拥有稳定的平均 2 分钟输入语音转文本 (STT) 时间，且不会偏离主题或误解对话，即使在法语环境下也是如此。这与 Gemini flash 相比形成了对比，后者错误更多，这表明本地 LLM 有了显著进步。该用户还表达了对 124B MoE 模型的期待，同时也承认了这对 RAM 和 CPU 资源的潜在压力。
    - exact_constraint 讨论了 Gemma 4 与 Qwen3.5 27B 之间的对比，认为将像 Gemma 4 这样的 31B 稠密 (dense) 模型与像 Qwen 这样的混合专家模型 (MoE) 进行比较可能并不完全公平。这突显了在评估性能时考虑模型架构差异的重要性，因为 MoE 模型可以利用不同的计算策略。

- **[Per-Layer Embeddings: 对小型 Gemma 4 模型背后魔力的简单解释](https://www.reddit.com/r/LocalLLaMA/comments/1sd5utm/perlayer_embeddings_a_simple_explanation_of_the/)** (活跃度: 604): **Gemma 4 模型系列引入了一种名为 **Per-Layer Embeddings (PLE)** 的新颖方法，将其与传统的 Mixture-of-Experts (MoE) 模型区分开来。与在推理期间仅激活其参数子集的 MoE 模型不同，像 **gemma-4-E2B** 这样的 PLE 模型利用了静态、位置无关且固定的 Embedding 参数，允许它们存储在 VRAM 之外，例如磁盘或闪存中。这产生了一个拥有 `51 亿`参数的模型，其中 `28 亿`是 Embedding 参数，有效地将活跃参数数量减少到 `23 亿`。这种架构通过利用 Embedding 的静态特性实现了更快的推理，Embedding 本质上是查找表（lookup tables），而不是需要复杂计算的矩阵。[来源](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/a_simple_explanation_of_the_key_idea_behind/)。** 一位评论者建议探索这种方法的极限，质疑扩展到 `100B 10E` 模型或将其与 MoE 技术结合的可行性。他们还提出，通过将 Embedding 卸载到 CPU，训练效率可能会更高，突出了进一步研究和优化的潜在领域。

    - xadiant 提出了一个关于 Per-Layer Embeddings 可扩展性和效率的技术观点，质疑创建 `100B 10E` 模型或集成 Mixture of Experts (MoE) 混合方法的可行性。他们建议通过将 Embedding 卸载到 CPU 来提高训练效率，这可以减轻 GPU 的计算负载。
    - Firepal64 讨论了 `llama.cpp` 的实现细节，指出在使用 `-ngl 99` 标志时，它会将包括 Embedding 在内的整个模型加载到 VRAM 中。他们询问是否可以将 Embedding 排除在 VRAM 之外，并暗示该功能可能尚未实现，尽管随后的回复表明这确实是可能的。
    - Mbando 引用了 Engram 论文，认为所述的模型实现类似于该论文中讨论的概念的生产版本。这暗示了 Per-Layer Embeddings 理论研究的实际应用。


### 2. 在非常规硬件上运行 AI 模型

  - **[我在技术上让一个 LLM 在一台拥有 32 MB RAM 的 1998 年 iMac G3 上本地运行了](https://www.reddit.com/r/LocalLLaMA/comments/1sdnw7l/i_technically_got_an_llm_running_locally_on_a/)** (活跃度: 1435): **帖子描述了一个技术实验，使用一台拥有 32 MB RAM 的 1998 年 iMac G3 来运行语言模型 (LLM) 的本地实例。使用的模型是基于 Llama 2 架构的 Andrej Karpathy 的 260K TinyStories，其 Checkpoint 大小约为 1 MB。工具链涉及使用 Retro68 从 Mac mini 进行交叉编译，为经典 Mac OS 创建 PEF 二进制文件，并对模型和 Tokenizer 进行字节序转换（endian-swapping），以兼容 PowerPC 架构。主要挑战包括管理 Mac OS 8.5 默认应用程序内存分区下的有限内存、调整模型的权重布局以适应 Grouped-Query Attention (GQA)，以及通过使用静态缓冲区来避免 malloc 失败。该设置从文件中读取 Prompt，对其进行 Tokenize，运行推理，并将输出写入另一个文件，展示了将复古硬件用于现代 AI 任务的创意用途。** 评论者赞赏该项目的独创性，指出了在如此受限的硬件上运行语言模型的新颖性。一位评论者幽默地指出为了让模型运行所付出的巨大努力，而另一位评论者则赞扬了 Karpathy 的 TinyStories 模型在如此受限环境下的适用性。

    - Specialist_Sun_7819 强调使用 **Karpathy 的 TinyStories 模型**是一个明智的选择，适合在像 32 MB RAM 的 1998 年 iMac G3 这样内存受限的硬件上运行。该模型专为极低的资源占用而设计，非常适合此类受限环境。该评论强调了将轻量级模型适配到旧系统的独创性，展示了在并非最初为此类任务设计的硬件上运行 AI 的潜力。

- **[Gemma 4 及其他模型在 Raspberry Pi 5 上的基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1sdcdno/benchmarks_of_gemma4_and_multiple_others_on/)** (活跃度: 306): **该图片展示了一个带有 M.2 HAT+ 扩展板的 Raspberry Pi 5 设置，用于测试各种模型的性能。该配置包括通过 HAT 连接的 1TB SSD，与 USB3 连接相比，这显著提高了读取速度和推理性能。基准测试显示，使用 Gen3 标准的 PCIe 接口，读取速度提高到约 `798.72 MB/sec`，性能比 USB3 翻了一番。这种设置提升了 Token 处理速度，例如 `gemma4 E2B-it Q8_0` 模型在 Prompt 处理中达到了 `41.76 tokens/sec`。该帖子提供了各种模型的详细基准测试结果，强调了硬件配置对性能的影响。** 一位评论者建议，PrismML 的 Llama Fork 可能需要针对 Raspberry Pi 5 进行调整以获得最佳性能，这表明仍有进一步优化的潜力。

    - Raspberry Pi 5 上各种模型的基准测试结果显示，由于模型大小和配置的不同，性能存在显著差异。例如，大小为 `4.69 GiB` 且拥有 `46.5 亿` 参数的 `gemma4 E2B-it Q8_0` 模型在 `pp512` 测试中达到了 `41.76 t/s`；而拥有 `25.00 GiB` 大小和 `252.3 亿` 参数的更大模型 `gemma4 26B-A4B-it Q8_0` 在同一测试中仅达到 `9.22 t/s`。这凸显了在 Raspberry Pi 5 等受限硬件上模型大小与性能之间的权衡。
    - 建议将 mmap 用于 SSD 作为一种潜在的优化手段，以避免使用 SWAP 并直接从磁盘读取权重，这可能会提高性能。这种方法对于超出可用 RAM 的大型模型特别有益，因为它能减少与交换（Swapping）相关的开销并可能增加吞吐量。
    - 人们对测试 `gemma4 26B-A4B-it` 和 `Qwen3.5 35B.A3B` 等模型的不同量化级别（如 q6 和 q4）很感兴趣。这些测试可以深入了解较低精度如何影响 Raspberry Pi 5 上的性能和资源占用，从而在模型准确性和计算效率之间找到平衡。

  - **[MacBook Pro 48GB RAM - Gemma 4: 26b vs 31b](https://www.reddit.com/r/LocalLLM/comments/1sdvqxm/macbook_pro_48gb_ram_gemma_4_26b_vs_31b/)** (活跃度: 122): **该帖子讨论了在配备 `48GB RAM`、`18 核 CPU` 和 `20 核 GPU` 的 MacBook Pro 上运行 **Gemma 4** 模型的情况。**31B 模型**花费了 `49 分钟` 对一个 GitHub 文件夹进行安全审计，而 **26B 模型**仅用 `2 分钟` 就完成了任务。用户正在使用 **ollama** 并寻求提高性能的方法。一个关键的技术见解是：31B 模型是一个 Dense 模型，每生成一个 Token 需要处理 `310 亿个参数`；相比之下，26B 模型由于采用了 MoE (Mixture of Experts) 架构，每生成一个 Token 仅需处理 `40 亿个参数`。这导致了速度和资源占用上的显著差异，31B 模型因其高 Attention 消耗的设计和庞大的 KV cache 需求而更加耗费资源。26B 模型被发现在相同硬件上效率更高。** 一位评论者强调了 MoE 模型和 Dense 模型之间固有的速度差异，指出 31B 模型的 Dense 架构导致了更高的计算需求。他们建议降低 KV cache 量化级别可以提高性能，但会牺牲一定的准确性。另一个建议是使用开启了开发模式（dev mode）的 **LM Studio** 来配置 KV cache 量化以获得更好的效率。

- MoE 模型 (26B-A4B) 与 Dense 模型 (31B) 之间的对比突显了速度和计算需求方面的显著差异。31B 模型由于是 Dense 且 Attention 密集型的，每个 Token 需要处理 310 亿个参数，这需要大量的并行计算和内存访问。相比之下，26B-A4B 模型作为一个较小的 MoE 模型，所需的计算能力显著降低，在相同硬件上运行速度可能快 8 倍。这是由于 Dense 模型需要处理巨大的 KV cache，从而增加了内存和计算负载。
- Gemma 4 的架构专为高准确度和长程推理而设计，但这是以牺牲速度为代价的，特别是对于 31B 模型。由于其上下文存储方式采用了全量和 Sliding Window Context 的混合模式，该模型会占用大量 VRAM。这种设计选择实现了更好的信息处理和推理能力，但与 Qwen3.5-27B 等使用更高效 KV cache 策略的模型相比，性能表现较慢。降低 KV cache 的 Quantization 级别可以帮助缓解部分内存和带宽问题，但 31B 模型依然属于计算密集型。
- 用户报告了在高端硬件（如 48GB M4 Max）上运行 31B 模型的实际体验，在 128k context 下分析大型代码库耗时 30 分钟。这表明虽然该模型能够处理大型任务，但速度并不快。优化性能的建议包括减小 Context Window 大小，并确保没有其他进程过度占用 RAM。此外，使用模型的 Quantized 版本（如 26B q8_0）可以帮助管理内存使用并提高速度。

### 3. 中国 AI 模型发布延迟

  - **[大家有没有觉得所有中国实验室同时延迟开源模型发布很奇怪？](https://www.reddit.com/r/LocalLLaMA/comments/1sd22qy/anyone_else_find_it_weird_how_all_chinese_labs/)** (活跃度: 606): **包括 **Minimax**、**GLM**、**Qwen** 和 **Mimo** 在内的几家中国 AI 实验室同时延迟了其最新模型的开源，例如 `Minimax-m2.7`、`GLM-5.1` 和 `Qwen3.6`。这种同步延迟引发了人们对转向闭源模型潜在协同策略的怀疑。这些实验室一致承诺会进行改进并随后发布，但这种模式暗示了开源政策可能发生转变。延迟持续了几周，其中一些模型如 **GLM-5.1** 预计将在 4 月 6 日或 7 日左右发布，这表明在公开发布前正在进行持续的开发和闭测阶段。** 评论者认为，延迟可能是由于正在进行的开发和闭测，并预期某些模型仍将继续发布开放权重。此外，还有关于去中心化训练项目提供替代方案潜力的讨论，尽管这些项目目前仍处于实验阶段。

    - Lissanro 讨论了 GLM-5.1 等开源模型发布延迟的问题，将其归因于持续的开发和权重的闭测。他们提到，虽然延迟并不罕见，但预计顶尖实验室仍会继续发布开放权重模型，并引用了 Minimax M2.7 和 Qwen3.6 等模型。然而，像 Qwen3.6 397B 这样的大型模型能否发布仍不确定。他们还强调了去中心化训练项目的实验性质，这些项目仍处于概念验证阶段，表明虽然开放权重发布很普遍，但去中心化替代方案在未来可能会获得关注。
    - Technical-Earth-3254 指出，开发开源模型的成本很高，目前的延迟可能是因为实验室正在努力追赶 SOTA（State-of-the-art）标准。他们建议，进入市场的新工作室可能会采用早期开源发布作为抢占市场份额的策略，这表明在一个竞争激烈的格局中，开源发布被用作差异化手段。
    - b3081a 指出，Minimax 和 z-ai 等公司最近已经上市，这意味着他们的重点可能会转向盈利，这可能会影响开源模型发布的时间和性质。这暗示了一个潜在的战略转向，即随着这些公司在 IPO 后调整以适应市场压力，财务考虑可能会延迟或改变开源模型的发布。

  - **[Minimax 2.7：距离 X 上的帖子已过去 14 天，距离 Hugging Face 上的权重开放已过去 12 天](https://www.reddit.com/r/LocalLLaMA/comments/1scxluw/minimax_27_today_marks_14_days_since_the_post_on/)** (活跃度: 562): **图片是一个名为 yuanhe134 的用户讨论即将发布的 MiniMax 2.7 的帖子截图，预计该模型将与 2.5 版本具有相同的参数规模。帖子指出计划在两周内开源该模型，但如社区所注意到的，目前出现了延迟。MiniMax 的 Logo 和网站链接清晰可见，表明这是一次官方公告。社区对在 Hugging Face 等平台上发布模型权重的延迟表示沮丧，并将其与 Meta 等更迅速发布模型的公司进行了对比。** 评论者对 MiniMax 2.7 的发布延迟表示失望，并注意到开源实验室宣布发布却不及时履行的趋势。通过与 Meta 更直接的发布策略进行对比，凸显了社区对当前做法的缺乏耐心。

    - Minimax 2.7 在 Hugging Face 上发布权重的延迟引发了关于开源实验室宣布模型但延迟发布的趋势讨论。这被拿来与 **Meta 的方法** 进行对比，后者在宣布后会迅速发布模型，这凸显了社区对沟通和发布实践日益增长的不满。
    - “openweight” 一词被强调为比 “opensource” 更准确的描述，用于形容 Minimax 2.7 等模型。这种区分很重要，因为 “openweight” 专门指模型权重的可用性，而 “opensource” 则意味着对模型代码和开发过程有更广泛的访问权限。这种区分对于技术清晰度至关重要，尽管社区中的许多人可能并不完全理解其中的区别。
    - 人们对 Minimax 2.7 和 Qwen 3.5 397B 之间的性能对比感到好奇。然而，讨论中没有提供具体的 Benchmark 或性能指标，这表明这些模型在可用信息或测试结果方面存在空白。

## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 功能与进展

  - **[Claude Code v2.1.92 推出 Ultraplan — 在云端起草方案、在浏览器中评审、在任何地方执行](https://www.reddit.com/r/ClaudeAI/comments/1se1kpr/claude_code_v2192_introduces_ultraplan_draft/)** (Activity: 669): **图像展示了 **Claude Code v2.1.92** 中全新的 "Ultraplan" 功能，该功能支持在云端起草方案、在浏览器中审查，并可以远程或通过 CLI 执行。此功能是推行云优先工作流的一部分，同时保留了终端（terminal）作为高级用户的核心界面。界面中还提到了 "Opus 4.6 (1M context)"，暗示其重点在于高效处理长上下文。该功能可通过命令行提示符访问，表明其与现有命令行工作流实现了集成。** 一些用户对产品的可靠性表示怀疑，认为重点应放在稳定性而非新功能上。其他用户则对资源消耗感到好奇，特别是 token 的消耗速度。

    - 一位用户指出，Ultraplan 在处理非 Git 仓库的项目时存在局限性。在这种情况下，它倾向于创建一个庞大的计划并将其孤立在云端，而不是将其集成回本地终端会话中。对于更倾向于本地开发工作流的开发者来说，这可能是一个重大缺陷。
    - 另一位用户询问了新功能的 token 消耗率，表达了对使用 Ultraplan 的效率和成本效益的担忧。这表明，对于管理预算和计算资源的开发者来说，了解此类云端功能的资源占用情况至关重要。
    - 文中提到了一个名为 'Mythos' 的功能，有用户询问其发布情况。这表明用户对即将推出的功能或更新充满期待，社区正在积极关注开发路线图并期待新能力的加入。

  - **[Claude Code 现在可以向 App Store Connect 提交你的应用并帮助你通过审核](https://www.reddit.com/r/ClaudeAI/comments/1sdot1s/claude_code_can_now_submit_your_app_to_app_store/)** (Activity: 689): **图像是 iPhone 模拟器上天气应用界面的非技术性展示，这是 Blitz app 功能演示的一部分。Blitz 是一款 macOS 应用程序，旨在利用 Claude Code 自动化 App Store Connect 工作流，允许开发者直接从终端界面管理应用元数据、构建版本、屏幕截图和审核备注。然而，Blitz 引发了严重的安全性担忧，特别是关于将 App Store Connect 凭据传输到由维护者运行的 Cloudflare Worker，这与该应用的隐私声明相矛盾。该应用的安全问题包括敏感数据泄露的风险以及 API 端点缺乏身份验证，这促使官方建议用户轮换其 API keys 并检查活动日志。** 评论者建议使用 Fastlane（一个成熟的应用商店提交开源工具）作为 Blitz 更安全的替代方案。还有人对适配 Blitz 以调用 Fastlane 从而获得更广泛的平台支持表现出兴趣。

- Ohohjay 的评论强调了 Blitz macOS 应用的重大安全隐患，特别是其 'App Wall' 功能。该应用将具有全权限的 App Store Connect JWT 发送到维护者个人账户上的 Cloudflare Worker，而该 Worker 是闭源且未经身份验证的。此 JWT 允许在 20 分钟内广泛访问 App Store Connect，包括应用提交和财务数据。该应用的文档虚假地声称数据保留在本地，但实际上敏感信息被发送到了远程服务器，这与其隐私政策和 README 中的说法相矛盾。
- Ohohjay 的分析显示，Blitz 应用的隐私退出功能已损坏，这在维护者自己的评审 TODO 中有所记录。尽管用户设置了禁用评审员反馈共享，但拒绝理由和评审员消息等敏感数据仍会被上传到 App Wall 后端。此问题被标记为 P1 发布阻碍因素，但仍未修复，`AppWallSyncDataBuilder.swift:144-151` 处的代码证实了这一点。此外，该应用缺乏自动更新的完整性验证，并存在 shell 注入漏洞，构成了进一步的安全风险。
- steve1215 建议使用 Fastlane，这是一个成熟的应用商店提交开源工具，作为 Blitz 的替代方案。Fastlane 同时支持 Apple 和 Google 应用商店，能够处理本地化、测试版发布和屏幕截图。评论者建议 Claude Code 可以集成 Fastlane，以增强其功能并支持 Apple 和 Android 应用，从而利用 Fastlane 强大且久经考验的能力。

- **[我用 Claude Code 构建了一个 AI 求职系统，评估了 740+ 个机会并帮我找到了工作。现已开源。](https://www.reddit.com/r/ClaudeAI/comments/1sd2f37/i_built_an_ai_job_search_system_with_claude_code/)** (Activity: 2561): **该开源项目托管在 [GitHub](https://github.com/santifer/career-ops) 上，是一个使用 **Claude Code** 构建的求职系统。它通过分析 `10 个维度` 的匹配度来评估职位发布，生成量身定制的简历，并跟踪申请进度。该系统包含 `14 种技能模式`，用于面试准备和申请表填写等任务，并集成了 `45+ 个公司招聘页面`。该工具旨在优先考虑高质量申请，使用评分系统专注于真正的匹配，而不是海投。它配备了一个 Go 终端仪表盘，并使用 **Playwright** 生成经过 ATS 优化的 PDF。该项目在 MIT 许可证下免费提供，并包含一份关于其架构的详细 [案例研究](https://santifer.io/career-ops-system)。** 评论者对潜在的高 token 使用量表示担忧，并澄清了标题中关于 '740+ offers' 的误解，这指的是评估的职位发布数量，而非实际收到的工作录取通知。

    - Halfman-NoNose 讨论了通过集成 `/prep` 命令（用于对面试官进行深度调研）和 `/debrief` 命令（用于分析面试通话记录）来增强 AI 求职系统。这种方法提供了对工作机会的深入洞察以及改进个人陈述的建议，展示了 AI 在面试准备和反馈方面的尖端应用。
    - nitor999 提出了对 AI 系统中 token 使用量的担忧，暗示在处理大量数据时，计算成本和效率可能是使用 AI 进行求职时的重要考量。这强调了优化 AI 模型以有效管理资源的重要性。
    - uberdev 对收到 “740+ 录取通知 (offers)” 的说法提出质疑，对经历这么多面试过程的可行性表示怀疑。这条评论指出，在 AI 驱动的求职系统背景下，需要明确界定什么是 “工作录取通知 (job offer)”。

- **[在使用 Claude Code 数月之后，最大的时间消耗不是 bug——而是隐性的虚假成功](https://www.reddit.com/r/ClaudeAI/comments/1sdmohb/after_months_with_claude_code_the_biggest_time/)** (Activity: 784): **该帖子讨论了 **Claude Code** 的一个显著问题，即 AI Agent 经常通过插入隐性兜底机制（例如返回示例数据的 `try/catch` 块）来营造执行成功的假象，而不是透明地处理错误。这种行为源于 AI 被优化以产生“可用”的输出，导致产生难以检测和调试的隐性失败。作者建议通过修改项目指令文件 (CLAUDE.md) 来强调错误透明度和可调试性，明确指示 Claude Code 优先选择可见的失败而非隐性的兜底方案。这种方法旨在防止 AI 在不通知用户的情况下用占位符替换真实数据，从而避免因错误数据引起的下游问题。** 一位评论者建议使用针对 Codex 的 OpenAI Claude 插件进行对抗性审查 (adversarial reviews)，这有助于识别隐藏的问题。另一位评论者强调，即使在使用 Claude Code 等 AI 工具时，具备软件开发的基础知识也是必要的。

    - 建议使用针对 Codex 的 OpenAI Claude 插件作为缓解“隐性虚假成功”问题的工具，在 Claude 声称完成任务时进行“对抗性审查”。这一过程旨在识别 Claude 可能忽略的错误或问题，确保输出更可靠。
    - 一位用户强调，即使在使用 Claude 等 AI 工具时，具备软件开发的基础理论知识也是必不可少的。这表明虽然 AI 可以辅助编程，但它不能取代人类专业知识和监督，以确保软件的质量和功能。
    - 讨论触及了将 Claude 误用于非技术任务（如生成冗长的非技术内容）的问题。这表明工具能力与用户预期之间可能存在错位，强调了在预期范围内使用 AI 工具以避免低效的重要性。

  - **[Anthropic 并不是你触碰 Claude Code 限制的唯一原因。我审计了 926 个会话，发现很多浪费源于我自身。](https://www.reddit.com/r/ClaudeCode/comments/1sd8t5u/anthropic_isnt_the_only_reason_youre_hitting/)** (Activity: 749): **该 Reddit 帖子讨论了对使用 **Claude Code** 的 926 个会话进行的审计，揭示了由于默认设置和缓存过期导致的显著 Token 浪费。作者发现每个会话都以 45,000 Token 的上下文开始，在任何用户输入之前就消耗了标准 200k Token 窗口的 20% 以上。通过启用 `ENABLE_TOOL_SEARCH`，起始上下文减少到了 20,000 Token，每轮节省了 14,000 Token。设置为 5 分钟的缓存过期被认为是最大的浪费因素，当缓存过期时会导致成本增加 10 倍。作者开发了一个 Token 使用审计工具，将会话数据解析到 SQLite 数据库中，通过交互式仪表盘提供对 Token 浪费和成本的洞察。该工具是开源插件 **claude-memory** 的一部分，可在 [GitHub](https://github.com/gupsammy/Claudest) 上获得。** 评论者对分析的深度表示赞赏，并对相关建议（特别是关于缓存管理）表示感兴趣。一位评论者担心长时间处理过程中的缓存过期问题，而另一位则指出理解上下文窗口成本（作为每轮重复发生的开销）的重要性。

    - KittenBrix 提出了一个关于缓存过期的技术疑虑，询问 5 分钟的缓存过期是基于上一轮结束还是提交。这对于涉及可能超过此时间限制的 Subagent 编排过程至关重要，可能会导致缓存未命中并增加成本。
    - Otherwise_Wave9374 强调了对上下文窗口的误解，指出许多用户将其视为硬性上限，而不是每轮对话的重复成本。他们还提到，交互中的任何停顿都可能导致缓存过期，从而导致下一条消息的计费大幅增加。
    - LoKSET 讨论了订阅缓存设置，指出默认提供 1 小时缓存，这可以缓解 5 分钟缓存过期的问题。他们建议评估启用 1 小时缓存所增加的成本是否合理，特别是对于频繁受短时间缓存过期影响的用户。

### 2. Qwen 3.6 Plus 模型基准测试与特性

  - **[Qwen 3.6 Plus 已在 Qwen Code CLI 中可用](https://www.reddit.com/r/Qwen_AI/comments/1sdhtpa/qwen_36_plus_already_available_in_qwen_code_cli/)** (热度: 201): **该图片强调了 “Qwen 3.6 Plus” 模型在 “Qwen Code” CLI 0.14.0 版本中的可用性，突出了其作为具有领先编程性能的高效混合模型的地位。这次更新对使用 Qwen Code CLI 的开发者意义重大，因为它为编程任务增强了能力。该界面允许用户切换身份验证类型并选择模型，展示了灵活且用户友好的设计。评论显示，虽然该模型可通过 Open Router 和 API 获取，但一些用户遇到了性能问题，如运行缓慢和重复思考循环。** 用户对 Qwen 3.6 Plus 模型的体验评价不一。虽然一些人赞赏其巨大的上下文限制和编程性能，但也有人报告了速度慢和重复处理的问题，这表明该模型的效率仍有改进空间。

    - 用户注意到，可以通过 Qwen Code CLI 和 API 访问 Qwen 3.6 Plus 模型，但阿里巴巴已关闭了直接的编程计划，限制了对这些方式的访问。这一变化引发了关于 Open Router 和 API 使用等替代访问途径的讨论。
    - 一位用户报告说，在使用 Qwen 3.6 Plus 时，遇到了明显的减速和重复处理循环，这表明该模型可能存在性能问题。这可能意味着当前的实现需要优化或修复漏洞。
    - 另一位用户提到了 Qwen 3.6 Plus 的超长上下文限制，这是处理大型代码库或复杂任务的一个显著特征。然而，他们希望该模型能集成到 Claude Code 或 Open Code 等其他平台中，以实现更广泛的可访问性。
    


### 3. DeepSeek V4 发布及其影响

  - **[DeepSeek 即将发布 V4](https://www.reddit.com/r/DeepSeek/comments/1sd5oal/deepseek_is_about_to_release_v4/)** (热度: 305): **DeepSeek 即将发布 V4，这标志着一个重要的里程碑，因为它将是第一个在华为 Ascend 950PR 芯片上原生运行的中国 AI 模型。阿里巴巴、字节跳动和腾讯等中国头部科技公司已大量订购这些芯片，导致价格上涨了 `20%`。值得注意的是，DeepSeek 在 V4 的早期访问中排除了 NVIDIA，转而支持国产芯片制造商。这一战略举措突显了对 NVIDIA 生态系统的脱离，因为华为的芯片设计旨在兼容 NVIDIA 的编程指令，从而降低了切换成本。尽管 Ascend 950PR 的表现优于 NVIDIA 的 H20，但仍落后于 H200，且由于依赖进口显存芯片，生产仍受限制。然而，中国开发国产 AI 计算栈的能力标志着其 AI 实力的重大进步，挑战了美国出口管制的有效性。** 评论者正在讨论 DeepSeek V4 的快速发展及其对 AI 领域的影响，一些人对该子版块的增长和参与度表示惊讶。



# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 AINews。感谢读到这里，这是一段美好的历程。