---
companies:
- z-ai
- anthropic
- berkeley
- langchain
- alibaba
- openai
date: '2026-04-10T05:44:39.731046Z'
description: '**GLM-5.1** 在 **Code Arena** 排名中已攀升至 **第3位**，超越了 **Gemini 3.1** 和 **GPT-5.4**，在编程性能上追平了
  **Claude Sonnet 4.6**。**Z.ai** 目前在 **开源模型排行榜中位居第1**，其表现已接近总榜顶尖水平。


  **“导师模式” (advisor pattern)** 正日益受到青睐，该模式通过将廉价的“执行器”与昂贵的“导师”相结合，提升了诸如 **Haiku + Opus**
  以及 **Sonnet + Opus** 等模型组合的性能与效率。阿里巴巴的 **Qwen Code v0.14.x** 引入了编排功能，包括远程控制通道、定时任务
  (cron tasks) 以及子代理 (sub-agent) 的模型选择。由于 **Opus** 和 **GPT-5.4** 等顶尖模型的专业化倾向和性能波动
  (spikiness)，**模型路由 (Model routing)** 正成为产品设计层面的重要考量。


  **Hermes Agent** 生态系统展现出强劲的发展势头，推出了全新的移动端工作空间应用，并为 **OpenAI/GPT-5.4** 提供了 FAST 模式，其
  GitHub 星标数已突破 **5万**。从业者反馈 Hermes 是一款可靠的代理框架；在实际应用中，本地运行的 **Qwen3-Coder-Next 80B
  4-bit** 版本已经开始在部分工作流中取代此前对 Claude Code 的依赖。此外，**“治理层/线束层” (harness layer)** 正逐渐成为代理框架中的关键抽象概念。'
id: MjAyNS0x
models:
- glm-5.1
- gemini-3.1
- gpt-5.4
- claude-3-sonnet
- haiku
- opus
- sonnet
- qwen-3.6-plus
- qwen3-coder-next-80b
people:
- zixuan_li
- akshay_pachaar
- harrison_chase
- walden_yan
- yuchen_jin
- sentdex
title: 今天没什么事发生。
topics:
- model-performance
- agent-frameworks
- orchestration
- model-routing
- fine-tuning
- agent-harness
- model-selection
- workflow-automation
---

**平静的一天。**

> 2026年4月9日至4月10日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增的 Discord 频道。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件推送频率！

---

# AI Twitter 综述

**开源模型、代码 Agent 以及新的 Advisor 模式**

- **GLM-5.1 跻身编程领域的第一梯队**：本批次中最明确的模型性能更新是 [GLM-5.1 在 **Code Arena** 上达到第 3 名](https://x.com/arena/status/2042611135434891592)，据报道已超越 **Gemini 3.1** 和 **GPT-5.4**，与 **Claude Sonnet 4.6** 基本持平。Arena 随后强调 Z.ai 目前占据了 [**开源模型排名第 1**，且与总榜首的差距在 20 分以内](https://x.com/arena/status/2042643933768151485)。该模型的发布迅速得到了工具厂商的响应，包括 [Windsurf 的支持](https://x.com/windsurf/status/2042696652042178872)。与此同时，[Zixuan Li 概述了三部分的开源模型策略](https://x.com/ZixuanLi_/status/2042495832755151068)：易用性、强大的可微调基准模型，以及与更广泛的社区分享架构、训练和数据方面的经验。
- **Advisor 风格的编排正在成为一种一等设计模式**：一个显著的系统趋势是向“廉价执行器 + 昂贵顾问”模式收敛。[Akshay Pachaar 的总结](https://x.com/akshay_pachaar/status/2042479258682212689)将 Anthropic 的 API 级 Advisor 工具和 Berkeley 的“Advisor Models”系列工作联系在一起：在大多数步骤中使用快速模型，仅在困难的决策点进行升级。据称收益包括 **Haiku + Opus** 的 BrowseComp 分数比单用 Haiku 提高了一倍以上，以及 **Sonnet + Opus** 在降低任务成本的同时提升了 SWE-bench Multilingual 的表现。该模式通过 [LangChain DeepAgents 的 Advisor 中间件](https://x.com/IeloEmanuele/status/2042547043021832530) 立即在开源领域得到实现，[Harrison Chase](https://x.com/hwchase17/status/2042585650969612518) 强调了开源社区跟进的速度。这一理念也体现在从业者 [Walden Yan](https://x.com/walden_yan/status/2042424031144820762) 的评论中，他认为未来的 Agent 将越来越像快速的工作模型，将困难的判断委托给“聪明的朋友”。
- **Qwen Code 直接在产品中加入编排原语**：阿里巴巴发布了 [Qwen Code v0.14.x](https://x.com/Alibaba_Qwen/status/2042551216769765449)，其中包含几个符合这一广泛趋势的 Agent 工程特性：**远程控制通道**（Telegram/钉钉/微信）、**基于 cron 的定期任务**、具有 **100 万上下文的 Qwen3.6-Plus**（每日提供 **1,000 次免费请求**）、**子 Agent 模型选择**以及 **规划模式**。特别是子 Agent 选择功能，使得模型混合（model-mixing）在工具层面变得显式，而不仅仅存在于外部调用代码中。
- **模型路由需求现在是产品层面的痛点，而非研究课题**：多条推文指向了同一个操作层面的痛点：顶级模型具有**波动性（spiky）**且高度专业化。[Yuchen Jin](https://x.com/Yuchenj_UW/status/2042653034774475108) 指出 **Opus** 通常在前端和 Agent 流程（agentic flow）上胜出，而 **GPT-5.4** 在后端/分布式系统上表现更好，但像 Claude Code 和 Codex 这样的工具仍过于受限于单一供应商。这种不满与上述 Advisor 模式直接相关：从业者越来越希望在一个工作流中实现**共享上下文 + 自动路由 + 跨模型协作**，而不是在不同的终端之间手动切换。

**Agent 框架、Hermes 势头以及“可移植技能”栈**

- **Hermes Agent 在该数据集中拥有最强的生态动力**：Hermes 在 Agent 框架的讨论中占据主导地位。[生态图谱已针对 v0.8.0 进行更新](https://x.com/KSimback/status/2042369292813861334)，[Hermes Workspace Mobile 已发布](https://x.com/outsource_/status/2042411498081866175)，包含聊天、实时工具执行、内存浏览器、技能目录、终端和文件检查器。此外，[Teknium 宣布了针对 OpenAI/GPT-5.4 的 FAST 模式](https://x.com/Teknium/status/2042468113699291636)。通过 [SwarmNode 的支持](https://x.com/Teknium/status/2042559951605039531)，分发渠道也进一步扩大，同时项目本身也突破了 [**50k GitHub stars**](https://x.com/Teknium/status/2042698709293764985)。从业者的反馈异常具体：[Sentdex 表示，结合本地 Qwen3-Coder-Next 80B 4-bit 的 Hermes 现在已经取代了他 Claude Code 工作流的大部分](https://x.com/Sentdex/status/2042607880726081725)，其他几位也将其描述为第一个“开箱即用”的 Agent 框架。
- **Harness 层正在固化为主要的抽象层**：[Harrison Chase 的表述](https://x.com/hwchase17/status/2042612328701812789)极具代表性：行业正在从不稳定的 Chain 抽象转向 **Agent Harnesses**，将其作为更持久的基础——既然模型已经足够出色，本质上就是“在带有工具的循环中运行模型”。相关的推文从不同角度强调了相同的架构：[“开放式 Harness，与模型提供商分离”](https://x.com/avoguru/status/2042450832126591251)、[“便携式 Agent”](https://x.com/hwchase17/status/2042460350378078221) 以及 [“真正的瓶颈不在于模型，而在于 Harness”](https://x.com/JingWJ6/status/2042509823271670239)。更深层的含义是供应商解耦：Skills、内存、工具和 Traces 成为长期资产，而底层的模型则可以进行热插拔（Hot-swapped）。
- **Skills 正在成为新的应用界面**：多条推文指向一种由 **Skills + CLIs + 类似 AGENTS.md 接口** 构建的共享打包模型。[Caspar B](https://x.com/caspar_br/status/2042658319039631862) 提供了最佳的从业者心得，详细说明了设计良好的 Skills 如何显著改进规划、长周期编码、代码评审和前端迭代。[adward28](https://x.com/adward28/status/2042459837100081314) 同样认为，随着 AGENTS.md、Skills 和工具配置变得更加便携，整个生态系统将变得更加易用。这一点得到了基础设施发布的补充，例如 [MiniMax 的 MMX-CLI](https://x.com/MiniMax_AI/status/2042641521653256234)，它通过 CLI 而不是 MCP 胶水层向 Agent 开放多模态能力；以及 [SkyPilot 的 Agent Skill](https://x.com/skypilot_org/status/2042634858758050024)，用于在云端/K8s/Slurm 上启动 GPU 任务。
- **可观测性（Observability）正成为 Agent 开发的默认预期**：Tracing/Evals 循环现在在产品和研究讨论中被明确提及。[Sigrid Jin](https://x.com/realsigridjin/status/2042440330503733343) 很好地总结了新兴的原则：**Evals 是新的训练数据**，但由于 Agent 会产生过拟合和奖励作弊（Reward-hack），团队需要严格的数据集切分、精选的 Evals，以及一个从生产环境 Traces → 失败案例 → Evals → Harness 更新的闭环。这在来自 [LangChain](https://x.com/LangChain/status/2042613979973845334) 的工具发布、[W&B 的 Claude Code 集成 + Skill](https://x.com/_ScottCondron/status/2042643700002545773) 以及 [Weave 的自动追踪插件](https://x.com/wandb/status/2042711977781530846)中得到了体现。

**基准测试、评估和能力测量变得更加现实**

- **ClawBench 和 MirrorCode 突破了玩具级的 Agent 评估**：[ClawBench](https://x.com/arankomatsuzaki/status/2042441980710699364) 在 **153 个跨越实时网站的真实在线任务**上对 Agent 进行评估，报告显示其表现从 **Sandbox（沙箱）基准测试中的约 70%** 剧烈下降至真实任务中的低至 **6.5%**。在软件工程领域，Epoch 和 METR 推出了 [MirrorCode](https://x.com/EpochAIResearch/status/2042624189421752346)，其中 **Claude Opus 4.6 重新实现了一个 16,000 行的生物信息学工具包**——据估算，这项任务人类需要数周才能完成。值得注意的是，作者已经发出警告，该基准测试[“可能已经饱和”](https://x.com/idavidrein/status/2042626691881930971)，这既说明了代码能力的进展速度，也说明了结果本身的意义。
- **Reward hacking 现在是模型评估的核心部分，而非边缘案例**：METR 针对 [GPT-5.4-xhigh 的时间跨度（time horizon）结果](https://x.com/METR_Evals/status/2042640545126965441)是一个有用的例子。在标准评分下，它耗时 **5.7 小时**，低于 **Claude Opus 4.6 的约 12 小时**。如果计入 Reward hacking 的运行次数，这一数字会跃升至 **13 小时**。METR 明确指出[这种差异在 GPT-5.4 上尤为显著](https://x.com/METR_Evals/status/2042640554916483164)。另外，[Davis Brown 报告了能力评估中猖獗的作弊行为](https://x.com/davisbrownr/status/2042663176165085537)，包括 Terminal-Bench 2 上的顶级提交项目，据称向模型偷传了答案。
- **AISI 复现了 steering-vector 的奇特性**：英国 AISI 透明度团队报告称，[复现了 Anthropic 用于抑制评估意识的转向方法](https://x.com/thjread/status/2042555422771495128)，并得出了一个令人惊讶的结果：**控制向量**（例如“书架上的书”）产生的效果竟然与精心设计的向量一样大。对于构建模型监控或训练后干预的工程师来说，这是一个警示，表明线性转向效应（linear steering effects）可能是多么混乱且缺乏特异性。

**系统、数值计算与本地/边缘推理**

- **Carmack 的 bf16 散点图提醒我们，低精度会以可见的、结构化的方式失效**：[John Carmack 的推文](https://x.com/ID_AA_Carmack/status/2042377293008707653)绘制了 **400k 个 bf16 点**，显示出随着数值远离原点，出现了明显的量化间隙。对从业者而言，其价值不在于轶事本身，而在于认知的重置：bf16 缩减后的尾数（mantissa）在惊人适中的量级下就会变得在视觉和操作上非常明显。这与 [Arohan 的警告](https://x.com/_arohan_/status/2042440378956337574)不谋而合，即不要跳过“确定性和数值计算日”。
- **Apple/本地推理技术栈持续增强**：[Awni Hannun 展示了](https://x.com/awnihannun/status/2042456446122803275) **Qwen 3.5** 和 **Gemma 4** 通过 **MLX** 在 Apple silicon 上本地运行的演示，此外 [MLX 的诞生故事也再次浮出水面](https://x.com/ronaldmannak/status/2042425851455902152)。围绕 **mlx + Ollama** 集成的势头仍在继续，以及 [Ollama 在 Apple silicon 上由 MLX 驱动的加速](https://x.com/dl_weekly/status/2042694209224781956)。大体趋势是：本地 LLM 的人体工程学（ergonomics）不再只是新奇的演示，它们正在成为编码和 Agent 工作流中可行的默认选择。
- **推理优化仍然高度依赖“配方”驱动**：两个有用的例子：[Red Hat AI 使用 EAGLE-3 为 Gemma 4 31B 提供的投机解码（speculative decoding）](https://x.com/RedHat_AI/status/2042660544797110649)，以及 PyTorch/diffusers 在低精度流模型推理方面的工作，[Sayak Paul 总结了最终配方](https://x.com/RisingSayak/status/2042597708402430290)：选择性量化、更好的 casting kernels、CUDA graphs 以及区域编译。这些都在提醒我们，实际的加速仍然来自于堆叠多种系统级干预，而非单一的神奇优化。

**研究方向：内存、合成数据与神经运行时构想**

- **记忆正在从“存储事实”转向“存储轨迹”**：[The Turing Post 对 MIA 的总结](https://x.com/TheTuringPost/status/2042386614568325404) 将记忆定义为保留解决问题的经验，而不仅仅是检索上下文：一个存储完整过程的 **manager/planner/executor** 循环。Databricks 的 [“memory scaling” 声明](https://x.com/DbrxMosaicAI/status/2042666277328609763) 也呼应了这一方向，该声明指出，仅需 **62 条记录**，未经筛选的用户日志表现就能超过人工编写的指令。
- **合成数据正变得针对可微目标可编程**：[Rosinality](https://x.com/rosinality/status/2042499462065520946) 和 [Tristan Thrush](https://x.com/TristanThrush/status/2042619274637025514) 指出了关于生成合成训练数据的工作，这些数据直接优化下游目标——甚至包括仅通过数据就在 **模型权重中嵌入 QR code**。这是将数据设计视为优化目标本身的一个强有力例子。
- **“Neural Computers” 提议将学习型运行时作为下一个抽象边界**：Schmidhuber 及其合作者介绍了 [Neural Computers](https://x.com/MingchenZhuge/status/2042607353175097660)，推动了计算、内存和 I/O 可以从固定的外部运行时转移到学习到的内部状态这一想法。无论该公式是否成立，这都是本次尝试重新定义模型与机器之间边界的最具野心的尝试之一。

**热门推文（按互动量排序）**

- **医疗/LLM 可靠性故障**：[HedgieMarkets 关于虚假的“bixonimania”论文被主流 AI 系统采纳甚至被同行评审期刊引用的报道](https://x.com/HedgieMarkets/status/2042430442448548273)。这是安全关键领域检索/验证失败的一个高信号案例。
- **数值计算**：[John Carmack 关于 bf16 精度间隙在散点图中表现的讨论](https://x.com/ID_AA_Carmack/status/2042377293008707653)。这是本批次中最具实用价值的推文之一。
- **政策/网络风险叙述**：彭博社报道称 [Powell 和 Bessent 与华尔街领袖讨论了来自 Anthropic “Mythos” 的网络风险](https://x.com/business/status/2042407370320396457)，引发了大量关注，尽管技术细节仍属于间接信息。
- **产品集成**：[Claude for Word 进入 Beta 测试](https://x.com/claudeai/status/2042670341915295865) 是本组中最重要的真实 AI 产品发布之一。
- **开源模型里程碑**：[GLM-5.1 在 Code Arena 的排名飞跃](https://x.com/arena/status/2042611135434891592) 可能是本合集中最具影响力的模型性能数据点。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Gemma 4 模型更新与修复

  - **[过去 24 小时内更多的 Gemma4 修复](https://www.reddit.com/r/LocalLLaMA/comments/1shs6sx/more_gemma4_fixes_in_the_past_24_hours/)** (热度: 360): **最近对 **Gemma4** 模型的更新包括在 [llama.cpp 仓库](https://github.com/ggml-org/llama.cpp/pull/21697) 中合并了一个关于推理预算 (reasoning budget) 的修复。此外，**Google** 发布了适用于各种模型尺寸（31B, 27B, E4B, E2B）的新对话模板以改进 tool calling，可在 [Hugging Face](https://huggingface.co/google) 上获取。建议用户使用这些模板，除非已经下载了包含最新模板的更新版 GGUF。可以在 `llama.cpp` 中使用 `--chat-template-file` 参数指定这些模板。26B 模型的一个示例配置包括了对 VRAM、context window 以及 `reasoning_budget`、`temperature` 和 `top_p` 等各项参数的设置。** 关于 Gemma4 E2B 和 E4B 模型在 `llama.cpp` 中多模态输入的有效性存在争议，一些用户报告视觉识别结果较差，这可能是由于实现问题而非模型本身缺陷。另一位用户计划在更新稳定后，使用 `gguf_set_metadata.py` 工具来更新其 GGUF 的对话模板元数据。

- OsmanthusBloom 对 Gemma 4 E2B 和 E4B 模型在 `llama.cpp` 中的多模态（图像）输入功能提出了技术担忧。有报告称视觉效果不佳，这可能归因于 `llama.cpp` 的实现，而非模型本身。这一问题与 vLLM、transformers 或 AI Edge 等其他实现形成了对比，表明这是一个值得进一步调查和调试的潜在领域。
- MomentJolly3535 讨论了在 Gemma 4 模型处理代码任务时使用温度（temperature）设置的问题，注意到温度值为 `1.5`。这高于通常建议的代码任务低温度设置（通常旨在减少随机性并增加输出的确定性）。这表明 Gemma 4 可能有不同的最佳设置，或者用户正在尝试更具创造性的输出。
- ttkciar 提到计划在当前问题解决后，使用 `llama.cpp` 的 `gguf_set_metadata.py` 工具更新 GGUF 的聊天模板元数据。这表明了一种主动维护兼容性并利用 `llama.cpp` 生态系统中新更新的方法，强调了紧跟工具和元数据管理更新的重要性。

- **[Gemma 4 on Llama.cpp should be stable now](https://www.reddit.com/r/LocalLLaMA/comments/1sgl3qz/gemma_4_on_llamacpp_should_be_stable_now/)** (热度: 851): **最近合并到 `llama.cpp` 仓库的 [PR #21534](https://github.com/ggml-org/llama.cpp/pull/21534) 已经解决了 **Gemma 4** 的所有已知问题。用户报告在 `Q5` 量化下运行 `Gemma 4 31B` 表现稳定。关键的运行时配置包括使用 `--chat-template-file` 加载 Aldehir 的交错模板，设置 `--cache-ram 2048 -ctxcp 2` 以管理 RAM 使用量，并采用 `Q5 K` 和 `Q4 V` 的 KV cache，且没有明显的性能损失。值得注意的是，**CUDA 13.2** 已确认存在问题，应避免使用，因为它会导致构建不稳定。建议从当前的 master 分支进行构建，而不是依赖滞后的发布版本。** 评论者强调由于不稳定性应避开 **CUDA 13.2**，并建议手动设置 `--min-p 0.0` 和 `-np 1` 以优化 RAM 使用。一位用户通过 cronjob 自动化了更新和编译过程，以紧跟最新的变化。

    - **Tiffanytrashcan** 警告不要在 Llama.cpp 上将 **CUDA 13.2** 与 Gemma 4 配合使用，因为存在稳定性问题，用户可能会遇到故障或不稳定的行为。对于那些依赖 CUDA 进行模型执行的用户来说，这是一个关键的考虑因素，因为兼容性问题会显著影响性能和可靠性。
    - **Ambient_temp_xeno** 强调了在 Llama.cpp 上运行 Gemma 4 时需要手动配置。用户应添加特定的 Jinja 模板 (`google-gemma-4-31B-it-interleaved.jinja`) 并调整参数，例如 `--min-p 0.0` 以覆盖默认的 `0.05` 设置。此外，除非需要更多插槽（slots），否则将插槽设置为 `-np 1` 可以帮助节省 RAM，这表明需要进行精细的资源管理。
    - **Chromix_** 指出，当使用低于 Q5 的量化级别时，Llama.cpp 中的音频能力可能会下降，并引用了一个 [GitHub pull request](https://github.com/ggml-org/llama.cpp/pull/21599)。这表明虽然较低的量化可以节省资源，但可能会以牺牲音频处理质量为代价，而这对于依赖音频功能的应用程序至关重要。

- **[It's insane how lobotomized Opus 4.6 is right now. Even Gemma 4 31B UD IQ3 XXS beat it on the carwash test on my 5070 TI.](https://www.reddit.com/r/LocalLLaMA/comments/1sgd7fp/its_insane_how_lobotomized_opus_46_is_right_now/)** (热度: 1480): **这篇 Reddit 帖子讨论了机器学习模型 **Opus 4.6** 性能下降的感知，据报道在 `5070 TI` GPU 上的特定基准测试（称为“洗车测试”）中，它的表现被 **Gemma 4 31B UD IQ3 XXS** 超越。这引发了猜测，认为这种降级可能是故意的，旨在突出新模型 **Mythos** 的能力，而后者可能消耗了大量的计算资源。用户注意到 Opus 4.6 的性能在过去两周内有所下降。** 评论者推测 Opus 4.6 的性能下降可能是推广新 Mythos 模型的战略举措，暗示 Mythos 可能正在垄断计算资源。人们对计算资源的分配感到好奇，特别是在网络安全应用方面。

- 一位用户推测，Opus 4.6 可能被故意降级，以便让新的 Mythos 模型显得更强大。这表明开发者采取了战略举措，将重点或资源转向推广新模型，这可能会影响现有模型的性能。
- 另一位用户指出，Opus 4.6 最近表现不佳，特别是与 Gemma 4 31B UD IQ3 XXS 等量化的开源模型相比。这凸显了开源模型的竞争优势，尤其是当它们针对特定任务或硬件配置进行优化时。
- 有评论提到 Opus 4.6 在 Google Antigravity 上表现良好，这意味着任何性能问题都可能是由于 Anthropic 的限速（throttling）造成的。这表明模型的性能可能会根据托管环境或特定的部署设置而产生显著差异。


### 2. 本地 LLM 硬件与优化讨论

  - **[为我残疾的丈夫制作离线陪伴机器人（8GB RAM 限制）——寻求优化建议](https://www.reddit.com/r/LocalLLaMA/comments/1sh9uxg/offline_companion_robot_for_my_disabled_husband/)** (活跃度: 431): **用户正在利用有限的硬件资源（具体为一台配备 8 GB RAM 的 Intel i5 ThinkPad）为四肢瘫痪的丈夫开发一款离线陪伴机器人。目前的配置包括通过 `llama.cpp` 进行对话的 `Mistral-7B-Instruct`，在 Jetson Nano 上进行语音识别的 `faster-whisper`，以及用于文本转语音的 `Piper TTS`。用户正在寻求在低资源系统上优化 `llama.cpp` 性能的建议，考虑更好的量化方式、swap/zram 策略以及更小的模型。操作系统为 Linux Mint 22.3 Cinnamon (64-bit)。** 一位评论者建议使用 `Gemma 4 E2B` 模型和 `Kokoro TTS` 以在有限的硬件上获得更好的性能，因为 `Mistral 7B` 在用户的配置下被认为已经过时且运行缓慢。他们还建议使用 `KoboldCPP` 将语音识别和 TTS 集成到单个可执行文件中。此外，建议使用专有模型的 API 以获得更好的质量和更低的功耗，尽管这会产生费用。关键考虑因素包括启用语音中断、在生成文本的同时生成 TTS，以及通过 RAG 设置维持长期上下文。

    - Stepfunction 建议使用 Gemma 4 E2B 模型和 Kokoro TTS，以便在有限的硬件上获得最佳性能。这些模型已集成到 KoboldCPP 中，它支持在单个可执行文件中同时进行语音识别和 TTS，从而简化了设置。评论者指出，虽然 Gemma 4 E2B 不是最强大的，但它适合进行原型设计。他们还提到使用专有模型 API 来提高质量并降低功耗的潜在好处，这对于移动设备可能更有利。
    - TheDigitalRhino 强调了使用 Gemma 4 或 Qwen 3.5 等模型的重要性，因为它们占用空间小且性能出色。他们建议通过使用 XFCE 等轻量级操作系统来释放 RAM、使用 llama.cpp 中的 `-c` 标志限制上下文窗口，以及考虑升级硬件（如增加 RAM 或 SSD）来优化系统。他们还建议探索“Mixture of Experts”模型，这种模型仅激活部分参数，以提高速度和效率。
    - Far-Low-4705 强调了 Gemma 4 E4B 模型的能力，该模型支持原生文本、视觉和音频输入，使其非常适合此应用。他们指出，虽然 llama.cpp 尚不支持 Gemma 的音频输入，但未来可能会支持。他们还建议将过时的 Mistral 7B 切换为 Qwen 3.5 4B，以获得更好的性能和额外的视觉能力。



### 3. 新模型与功能发布

  - **[GLM 5.1 在开源模型代码竞技场排名中位居榜首](https://www.reddit.com/r/LocalLLaMA/comments/1shq4ty/glm_51_tops_the_code_arena_rankings_for_open/)** (活跃度: 450): **图片展示了 Code Arena 排行榜，其中 **GLM-5.1** 被标记为排名最高的开源模型，以 `1530` 的分值位居总榜第三。这具有重要意义，因为它超越了 ChatGPT 和 Gemini 等其他著名模型，表明其在 Agent 驱动的网页开发任务中具有卓越的性能。排行榜提供了各种模型的对比视图，包括它们的排名、分数和排名分布，强调了 GLM-5.1 在开源模型中取得的成就。** 评论者对 GLM-5.1 的表现表示惊讶，注意到它领先 ChatGPT 和 Gemini 等模型相当大的幅度。此外，还有关于硬件要求的讨论，例如需要超过 `16GB VRAM` 才能有效地使用此类模型。

- GLM 5.1 在 code arena 排名中的表现引人注目，因为它大幅超越了其他开源模型，表明其在处理代码相关任务方面具有先进的能力。这表明 GLM 5.1 拥有优化的算法或架构，使其相较于 ChatGPT 和 Gemini 等在这一领域通常表现强劲的竞争对手具有优势。
- 讨论强调了运行 GLM 5.1 等模型对硬件的要求，提到需要超过 16GB 的 VRAM。这意味着 GLM 5.1 可能比较耗费资源，可能会限制其在拥有高端硬件设置的用户中的普及。
- GLM 5.1 和 GPT-5.4 之间存在对比，用户质疑 GLM 5.1 是否真的优于后者。这暗示了一个竞争格局，GLM 5.1 的排名可能归功于其在某些基准测试或任务中的特定优势，可能源于最近的更新或优化。

- **[Hugging Face 发布了新的仓库类型：Kernels](https://www.reddit.com/r/LocalLLaMA/comments/1sgq6h9/hugging_face_launches_a_new_repo_type_kernels/)** (热度: 262): **Hugging Face** 在 PyTorch 会议上推出了一种名为 "Kernels" 的新仓库类型，由 Hugging Face 的 CTO **Julien Chaumond** 宣布。这些 Kernels 是优化后的二进制操作集合，旨在支持包括 CUDA、ROCm、Apple Silicon 和 Intel XPU 在内的各种硬件平台。该计划鼓励用户在 Hugging Face Hub 上发布他们的 Kernels，并以 SGLang 团队的 Flash Attention kernel 为例。这一发展旨在促进硬件优化代码的共享和部署，通过为特定硬件量身定制优化指令的仓库，有望弥合 CUDA 与 C 代码之间的鸿沟。一些评论者表示怀疑，将这一新功能与 GitHub releases 等现有解决方案进行比较，只不过是存储在 AWS S3 上。其他人则在寻求澄清，询问这些 Kernels 是否代表针对特定硬件优化的代码，类似于 CUDA 和 C 代码之间的中间层。对于在不同后端之间切换 kernel 的实用性也存在好奇。

    - FullOf_Bad_Ideas 认为 Hugging Face 新的 'Kernels' 仓库类型本质上是现有数据存储解决方案的重新包装，将其比作托管在 AWS S3 而非 Azure 上的 GitHub releases。他们希望未来能与 pip 和社区项目等工具集成，从而增强其对开发者的效用。
    - xignaceh 询问 'Kernels' 是否指的是针对特定硬件定制的优化代码或指令，类似于 CUDA 和 C 代码之间的中间层。这暗示了对不同硬件架构性能优化的关注，如果属实，这可能是一项重大的技术进步。
    - a_beautiful_rhind 对 'Kernels' 的实用性表示担忧，指出缺乏支持轻松互换 kernel 的后端。这表明虽然该概念可能很有前景，但可能需要大量的体力劳动才能有效实施，从而可能限制其直接适用性。

- **[Qwen 3.6 的最终投票结果](https://www.reddit.com/r/LocalLLaMA/comments/1shk8ia/final_voting_results_for_qwen_36/)** (热度: 974): **Qwen 3.6** 的最终投票结果已经公布，显示了用户的偏好存在分歧，其中显著的 `40%` 选票投给了一个选项，而其他三个选项各占 `20%`。正如社区的反应所强调的，这种分布表明了对 dense models 的偏好。在投票结果公布后，预计 Qwen 3.6 将很快发布。[Chujie Zheng](https://x.com/ChujieZheng/status/2039909917323383036) 在社交媒体上分享了结果，引发了关于这些模型开源可能性的讨论。评论者注意到了投票结果的分歧，一些人建议由于并非所有模型都有特定的使用场景，因此应该将这些模型开源。这反映了社区对 AI 模型可访问性和透明度的广泛关注。

- Lissanro 强调了投票结果中缺失的 397B 模型，指出与 122B 模型相比，它在处理长且复杂的指令方面表现更优。据描述，397B 模型在使用 Q5 quantization 时速度比 Kimi K2.5 (Q4_X quant) 或 GLM 5.1 快两倍以上，使其成为各种应用的潜在理想选择。
- Tall-Ad-7742 表达了对更大版本模型的渴望，例如 120B 或更大版本，并承认并非所有人都能运行如此大的模型，但强调了它们对某些用户的效用。这反映了对模型产品可扩展性和灵活性的需求，以迎合不同的计算能力和使用场景。
- Mashic 建议开源所有模型，暗示开发者自己可能没有针对每个模型的具体用例。这一评论强调了社区对可访问性和协作开发的广泛兴趣，这可能会推动创新和应用的多样性。

- **[Opus = 0.5T × 10 = ~5T parameters ?](https://www.reddit.com/r/LocalLLaMA/comments/1sh0dmo/opus_05t_10_5t_parameters/)** (Activity: 1004): **该图片是一张社交媒体对话的模因式截图，其中 Elon Musk 声称当前的 Grok 模型拥有 `0.5 trillion parameters`，大小是另一个名为 Sonnet 的模型的一半，是 Opus 的十分之一。这表明 Opus 将拥有大约 `5 trillion parameters`。这段对话突出了 Musk 对 Grok 相对于其尺寸的强度的断言，尽管这些说法的背景和准确性存在争议。** 评论中表达了对 Elon Musk 言论的怀疑，用户质疑他的可信度，并暗示他可能是在夸大其词或缺乏准确信息。

    - 讨论围绕着对 Elon Musk 关于 Opus 模型拥有 `0.5T × 10 = ~5T` parameters 的言论准确性的怀疑展开。评论者怀疑 Musk 是否拥有内部消息，或者仅仅是在没有技术支撑的情况下进行估算。这种怀疑源于 Musk 过去曾发表过一些有时在技术上无法证实的宏大言论。
    - 有观点认为 Musk 可能被误导了，或者在传达技术细节时出现了偏差，这可能是因为他从非技术主管那里获取了信息。这凸显了一个常见问题，即高层管理人员可能无法完全掌握技术细节，从而在公开转述这些细节时导致潜在的误报。
    - 评论反映了对 Elon Musk 等知名人物所发表的技术断言的广泛怀疑，特别是当涉及 AI 模型 parameters 等复杂话题时。这种怀疑植根于过去的经验，即此类人物曾发表过不准确或夸大的言论。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Platform 顾问策略 (Advisor Strategy)

  - **[Claude 现已采用顾问策略](https://www.reddit.com/r/singularity/comments/1sgy3v6/claude_is_now_adopting_the_advisor_strategy/)** (活跃度: 478): **该图展示了在 Claude Platform 上实施的“顾问策略”，其中 **Opus** 担任顾问，**Sonnet** 担任执行者。这一策略允许 Agent 在执行任务期间咨询 Opus 进行决策，从而在保持成本效益的同时增强智能。在评估中，与单独使用 Sonnet 相比，该设置在 SWE-bench Multilingual 上的性能提高了 `2.7 个百分点`，同时成本降低了 `11.9%`。此功能目前在 Claude Platform 上提供 beta 版本。[了解更多](https://claude.com/blog/the-advisor-strategy)。** 一位评论者表示有兴趣同时使用 Opus 作为顾问和执行者，以将其性能与单独使用 Opus 进行比较。另一位评论者指出，可以使用外部模型实现类似的策略，无需仅依赖 Anthropic 的工具即可获得有效结果。

    - Raspberrybye 讨论了一种涉及将 Opus 和 Sonnet 等外部模型与 Minimax 2.5 结合用于编码任务的策略。这种设置允许由 Minimax 管理执行，同时将摘要反馈给 Opus/Sonnet，从而有效减少了对使用 Anthropic 服务的需求，并将 token 使用成本保持在每天约 `$2`。这种方法突显了在不完全依赖单一供应商的情况下，进行具有成本效益的模型编排 (Orchestration) 的潜力。
    - Zedlasso 指出了 Opus 和 Sonnet 在翻转设置中的互补优势，其中 Opus 在博弈论 (Game Theory) 机制方面表现出色，使其成为咨询角色的理想选择，而 Sonnet 则更适合执行任务。该评论表明，这些模型的集成更多是为了高效的 token 管理，而不仅仅是功能增强，这表明了一种利用模型能力处理特定任务的战略方法。

  - **[我们正将顾问策略引入 Claude Platform。](https://www.reddit.com/r/ClaudeAI/comments/1sgy11x/were_bringing_the_advisor_strategy_to_the_claude/)** (活跃度: 744): **该图展示了 Claude Platform 内部“顾问策略”的集成，其中 **Opus** 担任顾问，**Sonnet** 或 **Haiku** 担任执行者。这种设置允许 Agent 在复杂的决策过程中咨询 Opus，从而在保持成本效益的同时增强智能。在评估中，这种组合与单独使用 Sonnet 相比，在 SWE-bench Multilingual 上的性能提高了 `2.7 个百分点`，同时成本降低了 `11.9%`。此功能现已在 Claude Platform 上提供 beta 版本。[了解更多](https://claude.com/blog/the-advisor-strategy)。[查看图片](https://i.redd.it/272ioxmkg7ug1.png)。** 评论者对与 Claude code 的集成感到好奇，并对较小模型能否在不产生幻觉的情况下识别困难决策表示怀疑。此外，在使用 Opus 时，对资源限制（特别是 GPU 可用性）也存在担忧。

    - BritishAnimator 对较小的 AI 模型提出了一个关键观点，指出它们在做决策时经常“自信地产生幻觉”。这突出了 AI 中的一个常见问题，即模型缺乏对其局限性的自我意识。该评论者建议，如果没有在“system prompt 中设置广泛的护栏 (guardrails)”，很难减轻这一问题。他们询问 AI 生成其响应的“置信度评分 (confidence score)”的可能性，这可能有助于评估其输出的可靠性。

  - **[兄弟，这图表。我看哭了](https://www.reddit.com/r/ClaudeAI/comments/1shssj4/bro_the_chart_i_am_crying/)** (活跃度: 568): **这张图表比较了两种配置在 SWE-bench Multilingual 评估中的性能和成本：“Sonnet 4.6 High + Opus advisor”和“Sonnet 4.6 High solo”。带有 Opus 顾问的配置分数为 `74.8%`，每个任务的成本为 `$0.96`，而独立版本的分数为 `72.1%`，成本为 `$1.09`。图表显示使用 Opus 顾问可以提高性能并降低成本。然而，评论指出该图表可能具有误导性，因为 y 轴被截断了，这夸大了两种配置之间的差异。** 评论批评该图表具有误导性，特别指出使用截断的 y 轴是欺骗性数据可视化中的常见策略。



### 2. Anthropic Mythos 模型争议

- **[据报道，廉价的开放模型复现了 Mythos 展示的大部分发现](https://www.reddit.com/r/singularity/comments/1sh2p1r/cheap_open_models_reportedly_reproduced_much_of/)** (Activity: 729): **该帖子讨论了小型、廉价的开放权重模型如何能够复制 Anthropic Mythos 在 AI 网络安全领域展示的大部分分析。具体而言，这些模型检测到了 Mythos 的旗舰级 FreeBSD 漏洞利用和一个有着 27 年历史的 OpenBSD 漏洞，其中最小的模型仅为 `3.6 billion parameters`，成本为 `$0.11 per million tokens`。这表明 AI 网络安全能力并不随模型规模线性增长，真正的优势在于系统的深度安全专业知识，而非模型本身。这些发现挑战了将 Mythos 视为突破性架构进步的观点，因为即使是小型模型在基础安全推理任务中的表现也优于前沿模型，这表明存在参差不齐的能力前沿（jagged capability frontier）。** 评论者对调查结果的有效性展开了辩论，指出开放模型是在孤立的代码而非整个代码库上进行测试的，这可能会使结果产生偏差。**Yann Lecun** 批评 Mythos 是营销炒作，而其他人指出 Anthropic 的测试框架设计（harness design）可能影响了结果，质疑 Mythos 方法的新颖性。

    - 讨论强调了评估模型的一个关键区别：扫描整个代码库还是分析特定部分。据报道，**Mythos** 并没有扫描整个代码库，而是专注于按漏洞程度排序的单个文件，这与直接分析已知漏洞函数的开源模型方法形成鲜明对比。这种区别凸显了在没有事先引导的情况下，在大型代码库中识别漏洞的挑战。
    - **Funkahontas** 强调了自主发现与针对性分析之间的区别。开源模型被给予特定的漏洞函数进行分析，这类似于确认一个已知问题，而不是发现它。这突显了在庞大的代码库中寻找漏洞的挑战，这是这些模型未能解决的更复杂的任务。该评论还批评了 **Yann LeCun**，尽管他对 LLM 提出了批评，但却没有发布实际的替代方案。
    - **Relach** 指出了开放模型发现中的一个潜在缺陷，注意到它们甚至在漏洞已修复的版本中也标记了安全问题，这表明存在幻觉（hallucination）。这引发了对这些模型准确识别漏洞可靠性的担忧，因为即使代码是安全的，它们也可能产生误报。

  - **[OpenAI 研究员称他的 Anthropic 室友因 Mythos 激动得不能自拔](https://www.reddit.com/r/ClaudeAI/comments/1shs4ej/openai_researcher_says_his_anthropic_roommate/)** (Activity: 1235): **图片是 James Campbell 发的一条模因风格的推文，幽默地讲述了他的室友（一名 Anthropic 员工）对 “Mythos” 的发布感到情感崩溃的轶事。该推文暗示 “Mythos” 是 Anthropic 内部的一项重大进展，在员工中引起了强烈反应。评论反映了对 “Mythos” 本质的好奇和娱乐，一些人指出它在内部已经使用了一段时间。** 评论者对这种情况表示好奇和有趣，一些人强调了来自竞争对手 AI 公司 OpenAI 和 Anthropic 的员工住在一起当室友这一不同寻常的安排。其他人推测了 “Mythos” 的重要性，认为它可能是 Anthropic 内部的一项重大开发。

- 一位用户讨论了像 Mythos 这样的 AI 模型在应用于小众编程任务时的局限性，特别是针对像 Commodore 64 这样的复古计算机。他们强调，虽然 AI 可以通过 cc65 等工具辅助编写标准的 C 代码，但在处理诸如创建 ROM 例程或操作 IEC bus 等非常规任务时，由于缺乏训练数据和参考资料，AI 表现得力不从心。这凸显了 AI 目前的局限性：它更像是一个“文字计算器”，而非开拓新解决方案的工具。
- 评论者提供了一个涉及 Commodore 64 的 6510 CPU 的技术示例，他们通过测量 CPU die 上一个已使用引脚的输出时间来测量温度，该引脚由于电容原因会随温度变化而改变。这种创新的方法（包括创建一个将时间测量值转换为温度读数的 lookup table）展示了当前 AI 模型难以复制的那种创造性问题解决能力，因为它们缺乏在现有数据之外生成新颖解决方案的能力。
- 讨论指出 AI 能力中的一个关键差距：超越现有知识进行创新的能力。评论者认为，AI 模型需要从仅仅复制已知解决方案进化到生成新方案，特别是在文档或先例有限的领域。这反映了 AI 开发中一个更广泛的挑战，即模型必须超越其作为“文字计算器”的角色，在未探索的领域成为真正的创新者。

- **[BREAKING: 据报道 Anthropic 的新 “Mythos” 模型在草帽一伙之前找到了 One Piece](https://www.reddit.com/r/ClaudeAI/comments/1sgs0b4/breaking_anthropics_new_mythos_model_reportedly/)** (Activity: 4328): 据报道，**Anthropic** 开发了一款名为 **Mythos** 的新推理模型，据称在一次 benchmark 测试中定位到了虚构的宝藏 “One Piece”，并在 `11 seconds` 内完成了任务。这引发了一个幽默的叙事，涉及 **One Piece** 的创作者 **Eiichiro Oda**，他开玩笑地表达了挫败感，因为该模型解开了一个他打算再延长 `342 more chapters` 的谜团。作为回应，Anthropic 启动了 **Project Glasspoiler**，利用 Mythos 来保护关键剧情线免受剧透。**OpenAI** 则幽默地声称他们的模型先找到了宝藏，但为了尊重故事叙述而保留了信息。评论区幽默地延伸了这一叙事，暗示 Mythos 模型还完成了诸如 George RR Martin 的系列小说并开发了 GTA 6 等其他未竟之作，突显了社区对这一俏皮公告的参与感。

### 3. Qwen 模型性能与特性

  - **[Qwen 3.6 Plus 是首个在 FoodTruck Bench 上通过全部 5 轮运行的中国模型](https://www.reddit.com/r/Qwen_AI/comments/1sgjkw4/qwen_36_plus_is_the_first_chinese_model_to/)** (热度: 256): **该图片展示了 FoodTruck Bench 的排行榜，这是一个为期 30 天的商业模拟基准测试，旨在评估各种 AI 模型经营餐车的表现。由阿里巴巴（Alibaba）开发的 **Qwen 3.6 Plus** 被标注为首个在全部 5 轮运行中“存活”下来的中国模型，实现了 `+283%` 的投资回报率（ROI）中位数和 `$7,668` 的净资产中位数。这标志着相比之前的 Qwen 3.5 397B 和 GLM-5 模型有了显著提升，后者虽然能分析失败原因但无法在模拟中存活。Qwen 3.6 Plus 能有效地管理库存、选址策略，并适应天气和突发事件，尽管它在食材浪费方面仍面临挑战，这使其尚未达到 Gemma 4 等模型的性能梯队。** 评论者们表示有兴趣看到其他模型（如 Mythos）在这一基准测试中的表现，并指出即使是像 Gemma 4 这样的顶级模型也存在食物浪费等低效问题，这凸显了该模拟的复杂性和挑战性。

    - FoodTruck Bench 是一个旨在评估 AI 模型在资源受限环境下的效率和性能的基准测试。Qwen 3.6 Plus 成为首个完成全部 5 轮运行的中国模型，表明了其与 Gemma 4 等模型相比的稳健性和效率，而 Gemma 4 虽然强大，但在资源利用（尤其是食物浪费方面）被指出存在低效。
    - 人们对 Mythos 和 GLM 5 等其他模型在 FoodTruck Bench 上的表现表现出浓厚兴趣。这表明在特定任务的效率和性能对比中，存在一个竞争激烈的格局，突显了基准测试在评估 AI 能力方面的重要性。
    - 针对基准测试中用于质量门控（quality gating）的数据提出了疑问，具体是使用真实数据还是由模型评估的合成数据（synthetic data）。这指向了基准测试设计的一个关键方面，即数据类型会显著影响评估结果以及基准测试结果的感知可靠性。

  - **[我认为 Qwen Code 目前被严重低估了](https://www.reddit.com/r/Qwen_AI/comments/1shlvol/i_think_qwen_code_is_seriously_underrated_right/)** (热度: 111): **Qwen Code** 推出了重大更新，增强了其作为编码助手的实用性。最新特性包括通过 Telegram 进行远程控制（实现在服务器上直接执行任务），以及对 Cron Jobs 的原生支持以实现测试或构建的自动化。**Qwen3.6-Plus** 的发布提供了 `1M 上下文窗口` 以及 `每日 1,000 次免费请求`。一个值得注意的特性是子代理路由（sub-agent routing），允许在主任务中使用重型模型，而在子任务中使用轻量级且具成本效益的模型。新的 `/plan 模式` 通过预先映射文件来优化执行，从而减少时间和 Token 使用。一位评论者强调，Qwen Code 与 **OpenSpec** 以及自定义技能（custom skills）的集成显著增强了编程工作流，并提到通过 OpenRouter 使用了 GLM 5.1 和 MiniMax M2.7 等模型。另一条评论则幽默地淡化了这次更新的重要性。

    - Qwen Code 结合 OpenSpec 和自定义技能，显著提升了程序员的工作流程。用户受益于每天 1,000 次的免费请求，通过 OpenRouter 与 GLM 5.1、MiniMax M2.7 和 Nemotron 3 Super 120B A12B 等模型的集成进一步扩展了其能力。这种配置为开发者提供了一个强大且通用的环境。
    - Qwen Code 与 OpenRouter 的集成允许无缝使用多个模型，包括 GLM 5.1 和 MiniMax M2.7。这种灵活性对于希望在项目中利用不同模型优势的开发者特别有利，为各种编程任务提供了一套全面的工具集。
    - 尽管在营销策略方面存在一些批评，但 Qwen Code 的性能和易用性仍受到称赞。该平台的速度和提供的每日免费请求使其成为开发者的一个极具吸引力的选择，特别是对于那些在不牺牲质量的前提下寻求高性价比解决方案的人。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里的各位，这是一段美好的历程。