---
companies:
- openai
- anthropic
- sakana-ai-labs
- vercel
- artificial-analysis
date: '2026-06-22T05:44:39.731046Z'
description: '**OpenAI** 扩展了其 **Daybreak** 计划，推出 **GPT-5.5-Cyber** 模型，重点开发面向网络安全的闭环补丁生成能力。该模型扫描了超过
  3000 万次代码提交，覆盖 cURL、Python 等主要项目。这一发布引发了有关政策和出口管制的争议，与 **Anthropic** 对 **Mythos/Fable**
  采取的限制性访问形成对比。**Sakana Fugu** 推出了一个编排 API，能够在多个模型之间学习模型选择和任务委派，但因基准测试不透明、成本报告不充分而受到批评。与此同时，**GLM-5.2**
  作为一种适用于智能体应用和基础设施部署的开放权重模型，正受到越来越多关注。*“值得注意的转变，是从‘发现漏洞’转向在人工审核下生成闭环补丁”*，以及*“在测试阶段进行协调，在长周期任务中可能胜过调用单一大型模型”*，概括了其中的关键技术洞见。

  '
id: MjAyNS0x
models:
- gpt-5.5-cyber
- mythos
- fable
- glm-5.2
people:
- sama
- blackhc
- shashj
- levie
- audreyt
- eliebakouch
- blancheminerva
title: '今天没发生什么特别的事。

  '
topics:
- cybersecurity
- closed-loop-patch-generation
- model-orchestration
- test-time-scaling
- agentic-ai
- model-selection
- infrastructure-adoption
- benchmarking
- cost-accounting
---

**平静的一天。**

> 2026 年 6 月 20 日至 22 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/)支持搜索往期全部内容。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览


**OpenAI Daybreak、GPT-5.5-Cyber，以及政策与安全的分歧**

- **OpenAI 将其网络安全能力从漏洞发现扩展到了漏洞修复**：[OpenAI](https://x.com/OpenAI/status/2069104283824640023)宣布扩大 **Daybreak** 计划，推出 **Codex Security plugin**、面向可信防御者的完整 **GPT-5.5-Cyber** 模型、**Cyber Partner Program**，以及用于保护关键 OSS 的 **Patch the Planet**。后续帖子给出了更具体的范围：[已扫描 3000 多万个 commit，覆盖 3 万多个代码库，发现 7 万多个由审查者标记的修复，以及额外自动检测出的 50 多万个修复](https://x.com/reach_vb/status/2069110672886002140)；[cURL、Go、Python、Sigstore 和 pyca/cryptography 等大型项目均在覆盖范围内](https://x.com/gdb/status/2069112120206332130)；此外，[该 plugin 支持深度扫描、威胁建模、生成补丁，并导出到现有工作流中](https://x.com/gdb/status/2069128701850386834)。值得注意的是，重点已经从“找出 bug”转向了**由人类审核的闭环补丁生成**。
- **能力声明正与出口管制逻辑发生冲突**：OpenAI 明确声称，GPT-5.5-Cyber 在 CyberGym 上达到了 [SOTA](https://x.com/sama/status/2069121360744550796)，与此同时，围绕 Anthropic 受限的 **Mythos/Fable** 访问权限的公开争论仍在继续。[@BlackHC](https://x.com/BlackHC/status/2069168353919263002)提出了一个显而易见的政策问题：如果 OpenAI 最新的网络安全模型更强，为什么它没有受到同等程度的管制？[@shashj](https://x.com/shashj/status/2069078104941961293)还对 Mythos 事件作出了一个重要修正：NSA 所说的“几小时，而不是几周”，指的是**在假定已经拥有初始访问权限的前提下进行红队测试**，而据报道，那些红队目前已经无法继续访问 Mythos。结果是，**模型能力报告**与**连贯一致的治理标准**之间的鸿沟正在扩大。

**Sakana Fugu 的编排发布，以及对基准测试透明度的反弹**

- **Fugu 将“发布模型”重新定义为在模型池上进行学习式编排**：Sakana 推出了 [**Fugu**](https://x.com/SakanaAILabs/status/2068973497905545461)，将其描述为一个统一 API，能够在多个前沿模型之间学习**模型选择、任务委派、验证与综合**；[Vercel](https://x.com/vercel_dev/status/2069009248952942605)很快将 **Fugu Ultra** 加入 AI Gateway。对于已经观察到真实系统正逐渐转向编排层的工程师来说，这一产品理念很有吸引力：[@levie](https://x.com/levie/status/2068917230570795178)认为路由和编排很可能会成为高价值的一层，[@audreyt](https://x.com/audreyt/status/2068937870757548096)则表示，将 Fugu Ultra 作为规划器/顾问，与快速的 driver loop 搭配使用，效果很好。随后，Sakana 发布了一系列应用案例——autoresearch、金融、盲棋和 CAD——认为在长时程任务中，**测试时协调**可以胜过单一模型调用（[1](https://x.com/SakanaAILabs/status/2069084332879462779)、[2](https://x.com/SakanaAILabs/status/2069086336955646322)、[3](https://x.com/SakanaAILabs/status/2069088009790861312)、[4](https://x.com/SakanaAILabs/status/2069089571208679469)）。
- **批评随即而来：基线不透明、缺少成本核算，报告方式也存在疑问**：最详细的拆解来自 [@eliebakouch](https://x.com/eliebakouch/status/2068939729811468503)。他认为，Fugu 本质上是一个**路由器/分类器**，再加上一套预先规划好的多步骤工作流系统，并指出了几个核心问题：它在 **SWE-Bench Pro** 上比 **Opus** 低约 10 分；拿匿名的“Model A/B/C”进行比较；没有披露 best-of-N 式编排的** token/成本数据**；而且应该与其他**测试时扩展**方案比较，而不是与普通基础模型比较。[@BlancheMinerva](https://x.com/BlancheMinerva/status/2069009885958668340)进一步加剧了质疑，基于 Sakana 过去发生的事件，以及其早期工作中据称不可能实现的性能声明，对 Sakana 的可信度提出挑战。尽管这次发布在技术上仍然很重要，但讨论的重点已经从“编排有没有用？”转向了“应该如何评估和披露编排系统？”

**GLM-5.2 的突破：开放权重 Agent、基础设施采用，以及真实 harness 上的胜利**



- **GLM-5.2 正逐渐成为首个在 Agent 应用领域被广泛视为接近前沿水平的开放权重模型**：多篇帖子都指向同一个结论。[Artificial Analysis](https://x.com/ArtificialAnlys/status/2069121548670406947) 显示，**GLM-5.2** 在 **GDPval-AA** 上以 **1524 Elo** 排名**总榜第 3**，仅次于 Claude Fable 5 和 Opus 4.8，同时与部分闭源模型持平，甚至更胜一筹；他们还特别指出，GLM 是目前**领先的开放权重模型**，并且在 [AA-Briefcase 性价比前沿](https://x.com/ArtificialAnlys/status/2069148772446425563) 上表现突出。[[@natolambert](https://x.com/natolambert/status/2069073545632813193)](https://x.com/natolambert/status/2069073545632813193) 称，这可能是 Agent 领域的**“DeepSeek 时刻”**；而 [@AravSrinivas](https://x.com/AravSrinivas/status/2069146151325257913) 则认为，它重新激发了人们对开源的浓厚兴趣，因为在中位水平的生产知识工作中，它已经能够“通过盲测”。
- **最有说服力的证据来自真实的 harness，而不是抽象的基准图表**：[Cline](https://x.com/cline/status/2069171146994729078) 使用相同的 harness，让 GLM-5.2 和 Opus 4.8 修复 Cline 仓库中的一个真实 bug。结果显示，GLM 的速度更慢、工具调用更多，但成本更低（**0.41 美元 vs 0.81 美元**），验证环节也更可靠：它清理了无用代码，并确认生产构建成功；而 Opus 虽然通过了测试，却遗留了类型错误。[[@askalphaxiv](https://x.com/askalphaxiv/status/2069074178829901974)](https://x.com/askalphaxiv/status/2069074178829901974) 表示，GLM-5.2 是他们尝试过的首个能够执行**真实 autoresearch 任务**的开放权重模型，包括在两台 8xH100 节点上运行异步和 colocated RL 训练。在工具链层面，[[@_xjdr](https://x.com/_xjdr/status/2069030608727408993)](https://x.com/_xjdr/status/2069030608727408993) 介绍了他们如何将 GLM 提升为 ncode 的**默认模型**：整个周末都在加固容量、解析工具流，并为标准会话和 **1M context** 会话拆分 endpoint；另一篇帖子则详细说明了要让 OSS 模型顺利接入，竟然需要大量**针对模型的 parser 和 harness 工作**（[详情](https://x.com/_xjdr/status/2069038936362803544)）。
- **分发和 serving 的速度异常迅猛**：GLM-5.2 已登陆 [AWS Marketplace](https://x.com/CarolGLMs/status/2068902098696339811)，进入 [Baseten 的模型库，速度超过 280 tok/s、TTFT 低于 0.8 秒](https://x.com/baseten/status/2069153790503080251)，通过 [Fireworks 接入 Droid](https://x.com/FactoryAI/status/2069161306410942900)，集成进 [LangChain 的 deepagents 代码](https://x.com/sydneyrunkle/status/2069028200181539181)，并覆盖众多服务商——[有统计称已达到 20 家](https://x.com/paradite_/status/2069132200927522848)。与此同时，面向实际使用的教程也越来越多，例如这篇介绍如何通过 Baseten 的 OpenAI-compatible endpoint，在 Claude Code 中运行 GLM-5.2 的[指南](https://x.com/thealexker/status/2069163621469335757)。这里的关键在于：**开放模型的质量如今已经跨过了一个门槛，足以让推理服务商和 Agent 工具开发者围绕它进行大规模优化**。

**Agent 基础设施：Gemini Interactions API、Hermes 的扩展，以及以 harness 为先的工程实践**


- **Google 将 Interactions API 提升为面向 Agent 的主要 Gemini 接口**：[Google](https://x.com/Google/status/2069108942102310957) 和 [@OfficialLoganK](https://x.com/OfficialLoganK/status/2069115284519346263) 宣布，**Interactions API** 现已正式 GA，并成为 Gemini 模型和 Agent 的新默认接口。它的功能范围相当亮眼：通过一个 API 同时支持模型和 Agent、**后台异步执行**、更广泛的工具支持、多模态生成、托管式 Agent，以及一个名为 **Antigravity** 的隔离远程 Linux 沙盒（见 [@_philschmid](https://x.com/_philschmid/status/2069108134044467487)）。这让 Google 的技术栈越来越像是对“Agent harness”问题的官方一体化解法，而不只是一个模型端点。
- **Skills、通信协议和有状态会话正成为基础设施中的一等公民**：为了简化迁移，Google 发布了可安装的 [Gemini Interactions skill](https://x.com/_philschmid/status/2069137029359645007)，帮助编程 Agent 掌握新的 SDK 模式和当前模型版本。与此同时，[@omarsar0](https://x.com/omarsar0/status/2069066883995758814) 分享了一份关于**九种开源 Agent 通信协议**的实用调研，并指出，围绕**混合载荷加会话状态持久化**的标准正在形成，而去中心化发现机制仍不成熟。共同趋势是：团队正在围绕**有状态、工具丰富、可长时间运行的 Agent 工作流**进行标准化，但还没有形成完整的协议栈。
- **Hermes 作为本地化/个人 Agent 平台，覆盖范围仍在扩大**：Hermes 的更新包括[无需 Mac 即可访问 iMessage](https://x.com/tonbistudio/status/2068922944576008696)、[将 Raft 作为共享工作区中的外部 Agent 集成](https://x.com/raft_hq/status/2069040502507483192)，以及最重要的——[让任意模型都能控制 Windows 或 Linux 桌面应用的 GUI](https://x.com/Teknium/status/2069126072504074356)。该仓库的 Star 数也已突破 [200K](https://x.com/Teknium/status/2069088568161771522)，这进一步说明，大量开发者精力正投入到 **Agent UX 和 harness 易用性**上，而不只是基础模型质量。

**推理经济学、基础设施规模，以及向“自有智能”的转变**

- **Baseten 15 亿美元的 Series F，直接押注于后训练开源模型，以及作为企业控制平面的推理能力**：[Baseten](https://x.com/baseten/status/2069097489794527537) 和 CEO [@amiruci](https://x.com/amiruci/status/2069095112186196175) 表示，企业越来越希望**掌握自己的智能层**：运行开源或专用模型，利用自有数据和评测集进行后训练，并保留对持续学习的控制权。他们的客户名单——Abridge、Cursor、Decagon、Harvey、Notion、OpenEvidence 等——表明，这种做法已经在应用层发生。这也与当天更广泛的迹象一致：更强的开源模型加上更好的基础设施，正将**后训练从前沿实验室的专属能力，转变为应用公司的能力**。
- **算力租赁正成为一个独立的战略市场**：有关 [Reflection 与 SpaceX 签署 63 亿美元 GB300 算力使用协议](https://x.com/AndrewCurran_/status/2069078511948910820)的报道引发了广泛讨论；[@jaminball](https://x.com/jaminball/status/2069099044413304840) 将其与 SpaceX/xAI 此前和 Anthropic、Google 达成的其他大型算力协议放在一起分析，并指出，Blackwell 的隐含价格超过 **10 美元/小时**，且设有**90 天退出条款**。如果消息属实，这意味着“neocloud”算力和 GPU 经纪服务，正成为模型开发者与硬件供应之间越来越重要的战略层。
- **热门推文（按互动量排序）**：
  - **OpenAI Daybreak / GPT-5.5-Cyber**：[@OpenAI](https://x.com/OpenAI/status/2069104283824640023)、[@sama](https://x.com/sama/status/2069121360744550796)
  - **GLM-5.2 的真实世界验证**：[@cline](https://x.com/cline/status/2069171146994729078)
  - **Google 的 Interactions API 正式 GA**：[@Google](https://x.com/Google/status/2069108942102310957)
  - **Baseten Series F / “自有智能”论点**：[@amiruci](https://x.com/amiruci/status/2069095112186196175)
  - **Sakana Fugu 发布**：[@SakanaAILabs](https://x.com/SakanaAILabs/status/2068973497905545461)

**基准测试、评测方法，以及从静态分数转向真实工作流**



- **Judge 的可靠性正受到重新审视**：[[@dair_ai](https://x.com/dair_ai/status/2069063719817265463)] 总结了一项大规模的 LLM-as-a-Judge 审计，涵盖 **21 个评审模型**、**9 家提供商**以及约 **54.1 万次评判**。关键结论在于方法论：**完全匹配率会显著高估评审模型的质量**；而改用 **Cohen’s kappa** 后，MT-Bench 上的评审一致性会下降 **33–41 个百分点**，评审模型的排名也会明显变化。对于将评审模型作为内部评测基础设施的团队来说，这是一个强烈的警示。
- **人们越来越希望把 Agent 当作系统来评估，而不只是聊天机器人**：[Jules](https://x.com/julesagent/status/2069095582422200732) 对此进行了明确阐述：目标不只是让 Agent 做出反应，而是让它能够发现问题、提前行动并与人协作。与此同时，[@rseroter](https://x.com/rseroter/status/2069097330490446193) 强调了“使用 coding agent”和“构建一个**自主 coding harness**”之间的区别。当天最有实质内容的几个话题——Cline 中的 GLM、OpenAI Daybreak，以及对 Fugu 的批评——本质上都在讨论 Agent 在工具、记忆、验证和长时任务执行下的**系统行为**，而不是单轮对话中的原始智力。


---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. GLM-5.2 的价格/性能与 Homelab 部署

  - **[GLM-5.2 出现在 DeepSWE 中](https://www.reddit.com/r/LocalLLaMA/comments/1uc79ho/glm52_is_on_deepswe/)**（热度：606）：**这张图片是一张面向 coding agent/model 的 **DeepSWE 成本-得分基准图**，链接如下：[图片](https://i.redd.it/8qaktqtjjq8h1.png)。图中突出显示了 **GLM-5.2 [max]**：DeepSWE 得分为 `44%`，平均成本为 `$3.92/task`。它的得分低于 GPT-5.x、Claude 等顶级闭源模型，但在成本与性能之间取得了相对不错的平衡，尤其是考虑到帖子提到 DeepSeek 的定价可能已经过时，因为之后又推出了 `75%` 的折扣。帖子还将 DeepSWE 与 [ArtificialAnalysis 的 coding-agent 得分](https://artificialanalysis.ai/agents/coding-agents) 以及 [SWE-rebench](https://swe-rebench.com/) 联系起来，同时指出，DeepSWE 早先受到的部分批评已被原作者收回。**评论者对 GLM-5.2 总体持谨慎乐观态度，认为它“用起来”已经能与 Sonnet/Kimi 竞争。作为一个 open-weight model，它能够与 Opus/GPT 级别的系统在同一话题中被比较，这一点也相当引人注目。评论中也有人批评图表设计，尤其是成本轴将零点放在右侧；还有人调侃 Gemini 在这个基准上的表现竟然不如一些 open model。

    - 一位评论者认为，DeepSWE 的结果大致符合实际使用体验：**GLM-5.2** 感觉比 **Claude Sonnet** 和 **Kimi** 更强，但仍落后于 **Opus 4.8/GPT-5.5**。他们强调，GLM-5.2 的技术意义在于：这是一个**接近 frontier model 水平的 open-weight model**，可以自行部署；虽然硬件成本和配置复杂度都很高，但部署完成后就不再需要按 token 支付 API 费用。
    - 有人进一步审视了基准图中的成本/性能关系：一位用户询问 **GPT-5.5 Medium** 是否同时比 GLM-5.2 *更便宜且更强*；另一位用户则指出，**Fable Low** 似乎比 **Gemini 3.5 Flash** 和 GLM 更便宜。这个讨论说明，读者比较 DeepSWE 结果时，并不只看原始得分，也会关注专有模型与 open/open-weight model 之间经过价格归一化后的性能。
    - 一位评论者指出了基准图可视化方面的问题：图表似乎把坐标轴上的 `0` 放在了右侧，导致隐含的原点不一致——*“如果两个坐标轴都从 0 开始，原点应该是 0,0，而不是 0,-25。”* 这会影响技术解读，因为不寻常的坐标方向或经过平移的原点，可能扭曲人们对模型排名以及成本/性能取舍的判断。

  - **[GLM5.2 @7tg on 4x3090 + 192GB on budget motherboard + cpu](https://www.reddit.com/r/LocalLLaMA/comments/1ucknck/glm52_7tg_on_4x3090_192gb_on_budget_motherboard/)**（热度：838）：**一位 Homelab 爱好者分享了一台消费级工作站：配备 **4 张 RTX 3090 / 192GB DDR5**，总成本约为 `$6000`。在 Linux 下，每张 GPU 的功耗被限制在 `200W`；在一套预算型预装平台上，他还将内存频率从 `5200` 超频到 `5600 MT/s`，并升级到了 `1250W Platinum` 电源。报告中的本地工作负载包括：将 **GLM 5.2** 用作 planner，速度约为 `~7 tok/s`；将完全装入显存的 **MiniMax 2.7** 用作 coding model，速度约为 `~45 tok/s`；使用 **Qwen3.6 27B q8** 进行检查/测试，速度约为 `~50 tok/s`；此外，**Flux2Klein** 在两张 GPU 上批处理时，扩散生成速度约为 `1 image / 6s`。**评论主要集中在一些缺失的实现细节上：模型使用了什么 **quantization format**，为何选择 MiniMax 2.7 而不是 MiniMax M3，四张 GPU 的主板/PCIe 通道拆分是如何配置的，以及这种由太阳能供电的消费级硬件方案，与 ECC/server 或 Threadripper 平台相比，在成本和价值上究竟如何。



- 几位评论者都关注了在 `4x RTX 3090 + 192GB RAM` 上运行 **GLM5.2** 时缺失的**量化细节**，询问使用的是哪种量化版本，以及实际使用体验如何。其中一位用户特别问到，为什么不选择 **MiniMax M3**，这实际上是在比较模型质量、性能和显存占用是否匹配。
    - 大家还对平台拓扑结构很感兴趣：有人询问使用的是哪款**预算型主板**，以及连接 `4` 张 GPU 是否需要 **PCIe 分拆器/延长转接线**。这是一个重要问题，因为 `4x3090` 配置会受到插槽间距、PCIe 通道分配，以及主板和 BIOS 对多 GPU 支持情况的限制。
    - 一位正在组装类似开放式系统的评论者——`4×3090`、`256GB RAM`、**Threadripper Pro 5975WX**、**ASUS Pro WS WRX80E-SAGE SE WIFI**——询问了散热需求。讨论的重点是：在没有机箱的多张 3090 配置中，除了 CPU 散热和机箱风扇之外，是否还需要额外的定向气流；毕竟相邻 GPU 之间热量密集，也容易出现热风循环回流的问题。

  - **[Tokenomics](https://www.reddit.com/r/LocalLLaMA/comments/1ubrcwj/tokenomics/)**（活跃度：1984）：**图片是一张 [推文截图](https://i.redd.it/oqzbrucwan8h1.jpeg)，其观点是本地推理的“tokenomics”（代币经济学）可能在经济上并不划算：截图使用一个没有来源的例子，假设投入 **约 2 万美元的硬件**只能达到 **约 20 tokens/s**，并据此估算，相比 GLM-5.2 API 约为 **`$1.40/$4.40` 每百万 tokens** 的价格，需要 **约 5.5 年才能回本**。其技术意义并不在于具体计算结果——评论者质疑这些是*“编出来的数字”*——而在于它提出了一个更广泛的观点：云端 LLM 推理可以通过批处理和提高硬件利用率获益，而仅从单位成本来看，自托管则更难证明其合理性。**不过，评论者普遍认为，本地部署的价值仍然体现在**隐私、可靠性/不中断运行、控制权、兴趣爱好、微调/实验，以及高利用率的中小企业工作负载**上，而不一定是为了节省每个 token 的成本。还有人指出，具有竞争力的开源模型和云端模型定价，可能会让其利润率长期低于专有前沿模型 API。

    - 评论者质疑了帖子中的成本和性能假设，指出其中引用的 **`$20k` 硬件成本**和 **`20 tokens/s`** 数据都没有来源。有人认为，很少有用户会自托管 **GLM-5.2** 这样的大模型，但对于已经商品化、竞争激烈的模型，托管推理市场的竞争会使 API 利润率低于专有前沿模型的定价水平。
    - 讨论中还出现了围绕利用率的技术成本比较：云端批量推理通常比单用户本地推理更便宜，因为云服务商能更高效地让硬件满负载运行。不过，对于能够让 GPU 保持高利用率、重视隐私和控制权，或需要进行微调/REAP 类工作流的中小企业和重度用户来说，本地设备可能具有经济合理性。
    - 多条评论强调了摊销和风险问题：API 支出使用多年后无法回收，而购买的硬件仍然具有转售价值，并且可以保证本地可用。评论者还指出，托管 API 的价格不一定能长期保持稳定；因此，尽管本地设备的利用率可能较低，但出于隐私、不间断访问和长期成本控制等原因，本地推理仍然很有吸引力。


### 2. 本地 LLM 推理调优与 KV 量化

  - **[本地 LLM 推理优化：完整指南](https://www.reddit.com/r/LocalLLaMA/comments/1uc3wg9/local_llm_inference_optimization_the_complete/)**（活跃度：577）：**一篇新的 [llama.cpp 本地推理优化指南](https://carteakey.dev/blog/local-inference/local-llm-optimization/) 总结了针对消费级 GPU/CPU 的实用调优方法，重点涵盖**显存适配、KV cache 大小与量化（`-ctk/-ctv q8_0`）、Flash Attention、MoE 层放置、MTP/推测解码评估、CPU/P-core 调优、XMP/EXPO，以及常见的 OOM/加载时间故障。评论者特别指出了多模态场景中的一些陷阱：`mmproj` 在加载时需要**连续的显存空间**，因此视觉模型可能需要预留更多余量，例如使用 `--fit-target 2048`；同时，`--ubatch-size` 必须大于图像 token 数量，否则 llama.cpp 在视觉推理时可能触发断言。作者还分享了自己的基准测试追踪器 [l3ms.carteakey.dev](https://l3ms.carteakey.dev/)，测试设备为 **RTX 4070 12GB + i5-12600K + 32GB DDR5-6000**。**总体而言，大家对文章内容的技术反馈较为正面，尤其认可其中对各种实际故障模式的说明。不过，有一位评论者认为文章的行文风格很像 AI 生成，指出信息虽然有用，但读起来不够顺畅，并建议进行人工编辑。



    - 一位评论者指出了几个 **llama.cpp/GGUF 视觉推理中的常见陷阱**：应先使用模型卡片中的默认配置；加载 `mmproj` 时需要 **连续的 VRAM**；而过于激进的 `--fit-target` 值可能会导致加载阶段崩溃，而不是推理失败。对于多模态模型，他们还指出，图像可能会被分词成 **数百个 token**，因此 `--ubatch-size` 必须至少达到图像 token 的数量，否则 llama.cpp 可能会在视觉推理过程中触发断言。建议对视觉模型使用 `--fit-target 2048` 作为缓解措施。
    - 一位用户在 [l3ms.carteakey.dev](https://l3ms.carteakey.dev/) 分享了一个具体的本地推理基准测试环境：**RTX 4070 12GB**、**i5-12600K** 和 **32GB DDR5-6000**。这为根据实际硬件受限的测量结果比较优化建议提供了一个有参考价值的基准，尤其适用于 12GB 显存级别的消费级 GPU。
    - 有一篇技术性评论认为，指南中的 `ik_llama.cpp` 部分应该删除或重写，因为其中没有说明用户实际选择它的原因。评论者还强调，`ik_llama.cpp` 的改动**预计不会被正式或直接合并到 llama.cpp 上游**，因此如果仅仅将其描述为“尚未上游化”，可能会误导读者理解该项目与 llama.cpp 上游之间的关系。

  - **[Gemma 4 QAT seems to respond significantly better to KV cache quantization](https://www.reddit.com/r/LocalLLaMA/comments/1ubl0df/gemma_4_qat_seems_to_respond_significantly_better/)**（活跃度：329）：**帖子中的图表（[图片](https://i.redd.it/wxvhm0r1ml8h1.png)）报告了在 WikiText 的 `16k` 上下文下，**Gemma 4 26B** 使用不同 KV cache 量化方案时，相对于完整 16-bit KV cache 的 **KL divergence**，并比较了非 QAT 与 QAT 版本。关键技术结论是：**QAT 模型对 KV 量化的适应性强得多**：在非 QAT v4/v6/v8 中，`99.9%` KLD 大约从 `18.815 / 17.256 / 14.576` 降至 QAT 中的 `4.409 / 3.436 / 2.385`，这表明对于 Gemma 4 QAT 模型，`Q8_0` KV cache 可能又变得可行。**评论主要询问这些 KLD 数值具体代表什么，并表示希望在 `24 GB` GPU 上复现该基准测试。有人指出，这可能是 QAT 带来的意外副作用。

    - 一位拥有 `24 GB` GPU 的用户表示，如果有人提供代码，愿意复现并测试上述 Gemma 4 QAT KV-cache 量化结果；他们认为该讨论缺少足够的方法细节，因此很难解读这些数值或验证结果。
    - 一位评论者在 **Gemma 31B** 模型的视觉相关工作负载上报告了相反的经验：使用 `q8` KV cache 得到的结果*“更差或不够准确”*，不如 `bf16` KV cache，因此他们改回了 `bf16`。这说明 KV-cache 量化带来的收益可能取决于具体任务和模型，并不一定普遍提升效果。
    - 另一位评论者推测，对 KV-cache 量化容忍度的提升可能是 **QAT** 本身带来的非预期副作用；另有评论指出 **QAT Gemma** 存在已知问题，并询问这些问题是否已经修复。

  - **[My experience so far with 100% LOCAL LLM + RTX 5090 🤔](https://www.reddit.com/r/LocalLLM/comments/1ubkczr/my_experience_so_far_with_100_local_llm_rtx_5090/)**（活跃度：859）：**图片是一张用于在本地运行 **Qwopus3.6 27B v2 MTP** 的 **LM Studio 配置截图**，硬件为 **RTX 5090 32GB**。截图展示了约 `160,768` token 的长上下文配置，并启用了 GPU offload、KV cache offload 和 Flash Attention；内存估算值也接近显存上限（[图片](https://i.redd.it/xzc7aq0efl8h1.png)）。结合上下文来看，这篇帖子是一份关于如何将高密度本地编程/聊天模型装入 `32GB` 显存的实践报告，重点讨论了尽可能实现 `100%` GPU offload、`Q8_0`/之后改用 `Q5_1` 的 KV-cache 量化取舍，以及使用 LM Studio + Cline/OpenCode 进行分步骤“氛围编程”，而不是一次性生成全部内容。**评论者普遍认同作者对工作流的结论：将任务拆分得更小、设置检查点，以及使用持久化的规则/skills 文件，都能提升本地 Agent 的可靠性。一位技术评论者建议使用 `Q5_1` V-cache 量化，并增大 evaluation batch size / physical batch size，以优化更长上下文下的性能和速度；作者后来在 LM Studio 中进行了测试，但结果喜忧参半。



- 一位评论者进一步强调了这一工作流观点：与其使用大型“英雄式提示词”（hero prompts），不如将任务拆分为更小的范围，设置严格的检查点，并逐步迭代，这样本地 LLM 的表现会更好。他还指出，应将 `rules`/`skills` 文件维护成模型持续更新的操作手册，类似于运行手册和定期评审机制；他还参考了 [aiosnow.com](https://www.aiosnow.com/) 上的一个示例结构。
    - 有人提出了一项技术优化：对 KV-cache 进行量化，具体来说是将 **V cache 降至 `Q5_1`**。根据相关基准测试，这样做只会带来极小的质量损失，却能显著节省 VRAM 和上下文内存：[长上下文 KV cache 量化基准测试](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context#section-8)。同一位评论者还建议将 **Evaluation Batch Size** 和 **Physical Batch Size** 都提高 **2–4 倍**，并表示这能让其配置下的生成速度得到大幅提升。
    - 另一位评论者只推荐了使用 `llama.cpp`，暗示这是一套针对消费级 GPU/CPU 优化的本地推理方案，适用于常见的 GGUF 量化模型工作流。



### 3. 本地 AI 硬件供应情况

  - **[Chinese Hackers Latest Masterpiece with NVIDIA](https://www.reddit.com/r/LocalLLaMA/comments/1ucokod/chinese_hackers_latest_masterpiece_with_nvidia/)**（热度：886）：**一位 Bilibili 硬件改装者称，自己花了约 `1 年`时间，逆向分析 NVIDIA **Tesla V100** 的封装/电路板接口，共涉及 `2,963` 个引脚信号，并将其重新设计成一块**单槽位/半高规格的“Tesla V100 v4”** PCB，据称支持 **NVLink**，最多可扩展至 `8 路`配置（[帖子](https://t.bilibili.com/1211458176581369862)、[工程师主页](https://space.bilibili.com/1560089206)、[视频](https://www.bilibili.com/video/BV13JEa6sEtb/)）。其标价对于 V100 级别的硬件来说低得惊人：`16 GB` 版本售价 `1499 元`（约 `$220`），`32 GB` 版本售价 `3999 元`（约 `$590`）；此外，`2 路`和 `8 路` NVLink 适配器的价格分别为 `199 元`和 `799 元`。评论者还提到，中国有人在制作采用 MCIO 风格连接的逆向 **NVLink** 适配卡，声称四张 GPU 之间的带宽约为 `100 GB/s`。目前最大的技术疑问在于可靠性：重新处理二手 V100 的 BGA 封装时可能会损伤旁边的 **HBM**，因此长期良品率和保修可信度仍是未知数。**评论者普遍对这项逆向工程和 PCB 小型化工作印象深刻，并对高密度多 GPU/HBM 配置表现出兴趣，尤其是通过 NVLink 连接的 `4x32 GB` V100 节点。一位评论者表示，如果有兼容的单槽位水冷头，自己愿意购买很多张 `32 GB` 卡；而原帖作者则澄清说，自己只是分享这个项目，并不是在推广或销售产品。**

    - 评论者讨论了一款据称来自中国、经过逆向设计的 **NVIDIA NVLink** 互连适配器：这是一张通过 **MCIO** 将 GPU 连接起来的 `4 路`卡，据称四张 GPU 之间可提供 `100 GB/s` 的带宽。一位用户特别看重其优势：以这样的链路速度，将四张 `32 GB` 卡上的 HBM 汇聚成 `128 GB`。此外，还有传闻称，一款支持 `8 路` NVLink 的适配器正在开发中。
    - 围绕散热和外形尺寸，有人从硬件改装角度展开讨论：一位评论者表示，如果有人推出**单槽位水冷头**，自己会购买多张 `32 GB` 卡。这说明在一台机箱中部署多张此类卡时，空间密度可能是主要限制因素。
    - 也有人怀疑这项工作是否真的属于逆向工程，而不是使用了泄露的设计资料：一位评论者指出，据称 **V100 SXM PCB 文件**已经广泛流传，这意味着相关适配器的制作可能借助了现成的原理图或电路板文件，并非完全通过洁净室式的独立逆向工程完成。

  - **[been tracking EU DDR5 data for 25 days: Prices are dropping, and the DE vs. NL gap is wild (good news for local LLM builders in EU)](https://www.reddit.com/r/LocalLLaMA/comments/1ucixz9/been_tracking_eu_ddr5_data_for_25_days_prices_are/)**（热度：354）：**原帖作者分享了一个处于测试阶段的欧盟 RAM/CPU 价格追踪器 [PriceSquirrel](http://www.pricesquirrel.com)。数据显示，DE/NL/ES/BE 多地的 DDR5 套装价格在 `25 天`内明显下降。例如，**G.Skill DDR5 Aegis 2x16GB 6000** 从 `€579` 降至 `€419`（下降 `-28%`），**Kingston FURY Beast RGB 2x16GB 6000** 从 `€499` 降至 `€369`（下降 `-26%`），**G.Skill Trident Z Neo 2x32GB 6000** 从 `€1200` 降至 `€927`（下降 `-23%`）。目前提到的最大价差，是同一 EAN 的 **G.Skill Trident Z5 RGB 2x32GB DDR5-6400**：德国 NBB 售价为 `€799`，而荷兰 Megekko/Azerty 售价为 `€1180`。总体而言，德国价格通常比荷兰/比利时低 `10–20%`；原帖作者认为，**DDR5-6000 2x16GB** 正逐渐成为本地 LLM 推理的入门级“性价比甜点位”。**评论者指出，欧盟消费级 DDR5 的降价趋势，与美国注册型/服务器 DDR5 的情况形成对比：一位追踪者发现，`64GB DDR5-4800 RDIMM` 的价格在 6 月初从 `$1530` 涨到 `$1800`，之后仍维持在高位。还有人认为，RAM 价格正在全面扭曲游戏机和工作站的升级成本；一位用户比较称，如今 AM5/AM6 平台的升级成本接近 `€2000`，而过去购买同级别内存的成本还不到 `€500`。**

    - 一位追踪**美国注册型/服务器 DDR5 RAM** 的评论者表示，其价格走势与欧盟桌面级 DDR5 正好相反：`64GB DDR5-4800 RDIMM` 在 6 月初从约 **`$1530` 涨至 `$1800`**，之后一直维持在这一水平。这表明服务器级内存可能仍然受到供应限制，或者面临着与消费级 DDR5 不同的需求压力。
    - 对于本地 LLM 组机，一位用户认为，在依赖系统 RAM 的情况下，**较老的 DDR4 工作站/服务器平台可能比 DDR5 台式机更便宜、速度也更快**。他声称，一套约 10 年前的**六通道 Xeon DDR4-2400** 配置，其内存带宽可以超过双通道 **DDR5-7000 台式机**；如果模型层被卸载到系统 RAM 中，那么与内存容量和带宽相比，**PCIe 代际**在实际使用中的影响很小。
    - 对于德国本地的硬件价格追踪，一位评论者推荐使用 **Geizhals**，称它是查询历史科技产品价格和比较零售商报价的常用来源。




## 技术含量较低的 AI 子版块摘要

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo


### 1. Claude 身份验证上线

  - **[Anthropic 正在逐步推出身份验证。就在昨天刚更新。](https://www.reddit.com/r/ClaudeAI/comments/1uboasr/anthropic_is_rolling_out_identity_verification/)**（热度：3429）：**这张[图片](https://i.redd.it/5blf6lxykm8h1.jpeg)展示了 Anthropic 最近更新的 Claude 帮助页面——**“在 Claude 上进行身份验证”**。页面称，Anthropic 正在针对某些使用场景逐步推出身份验证，以防止滥用、执行相关政策并履行法律义务。帖子指出，验证由第三方服务商 **Persona Identities** 负责，可能需要政府签发的带照片身份证件以及配备摄像头的设备；[这里](https://web.archive.org/web/20260415064244/https://support.claude.com/en/articles/14328960-identity-verification-on-claude)提供了已存档的支持页面。**热门评论普遍持强烈负面态度，主要担忧隐私和供应商信任问题，尤其反对 Persona 与 Peter Thiel 的关联。多位评论者表示，他们会停止为 Claude 付费，或预计此举会推动用户转向中国模型或开源模型。

    - 一个较为深入的隐私与安全讨论聚焦于 Anthropic 使用 **Persona** 进行身份验证。有评论称，该流程需要**政府签发的带照片身份证件加实时自拍**，也就是会处理面部几何特征等生物特征信息。评论者指出，据称这项政策适用于 **Free、Pro 和 Max** 消费者账户，但不适用于 **Team、Enterprise 或 Platform**，因此高阶消费者订阅用户会直接受到影响。
    - 一个与技术相关的担忧是第三方如何处理数据：评论者援引报道称，Persona 的子处理商包括 **AWS、Google、OpenAI、Stripe 和 Twilio**，这意味着身份证件和生物特征验证数据可能会经过更广泛的供应商链路，而不只是留在 Anthropic 内部。他们还指出，Anthropic 的支持材料据称没有明确说明身份验证数据的**保留期限**，并将其视为一个严重的隐私与合规缺口。
    - 该讨论还将此次上线与更广泛的平台风险控制联系起来：Anthropic 给出的理由被解读为与**触及现实世界服务的 Agent 能力**、平台完整性检查，以及 **EU AI Act** 和生物特征隐私法律等监管压力有关。不过，评论者批评其验证触发条件含糊不清，例如*“某些能力”*和*“平台完整性检查”*，认为这种模糊表述让用户难以判断何时需要提交敏感的身份验证信息。

  - **[Anthropic 将从 2026 年 7 月 8 日起，针对某些能力逐步推出身份验证](https://www.reddit.com/r/singularity/comments/1ubkpe5/anthropic_is_rolling_out_identity_verification/)**（热度：1180）：****Anthropic** 更新了 Claude 的政策文档，新增了自 `2026 年 7 月 8 日` 起生效的 **“验证数据”**处理条款，该条款涉及 Claude 中未明确说明的“某些能力”/“高级能力”的身份检查（[支持文章](https://support.claude.com/en/articles/14328960-identity-verification-on-claude)、[隐私政策更新](https://privacy.claude.com/en/articles/10301952-updates-to-our-privacy-policy)）。帖子称，验证由第三方身份验证服务商 **Persona** 负责，因此围绕收集政府身份证件以解锁受限模型功能，引发了数据保留和隐私方面的担忧。**评论者强烈反对基于身份证件的访问限制，认为付款已经足以完成验证，并警告“高级能力”的范围可能会随时间扩大，例如涵盖安全分析、漏洞发现或代码加固提示。多人认为这很可能会成为整个行业的趋势，并希望开源模型能够迎头赶上，从而避免强制性的类似 KYC 的访问控制。

    - 评论者推测，此次上线可能与 Anthropic 更高能力系统的**出口管制限制**有关，特别提到 **Mythos** 仅限**美国公民**使用。他们担心，随着模型能力被归类为敏感能力，访问模型可能越来越需要进行身份、国籍或资质核验。
    - 有人提出技术层面的担忧：所谓“高级能力”可能包括漏洞发现、漏洞利用分析或代码加固等与安全相关的工作流，导致原本合法的软件安全使用场景触发身份验证。用户担心，这条边界可能会随着时间推移，从少数高风险功能扩展到更广泛的编程或分析功能。
    - 多条评论批评 Anthropic 的运营可靠性和产品控制，提到据称存在**模型性能在未告知用户的情况下下降**、Token 消耗统计不一致或存在 Bug，以及付费订阅无法在 Anthropic 自有应用之外使用等限制。还有一位评论者提到，Anthropic 选择 **Persona** 作为身份验证服务商。





### 2. Anthropic Frontier Model 传闻

  - **[Claude Sonnet 5 “Fennec” 泄露，拥有 1M context，预计下周发布](https://www.reddit.com/r/ClaudeCode/comments/1uc1aj4/claude_sonnet_5_fennec_leak_1m_context_expected/)**（热度：1823）：**这张[图片](https://i.redd.it/4ppk5ty2bp8h1.jpeg)是一张橙色背景的**宣传风格图片**，上面写着“Claude Sonnet 5”；它并不能为所谓的泄露消息提供技术证据。帖子声称，Anthropic 下一代 Sonnet model 的代号可能是 **“Fennec”**，最早或许会在下周发布，具备 `1M` token context window、出色的 coding performance、更快的 inference，以及相比 Opus/Fable 更好的价格性能比，但帖子没有提供来源或 benchmark data。**评论区普遍对这则泄露的可信度持怀疑态度，例如有人说：“这个泄露是和我们处在同一个房间里吗？”以及“这是 OP 在梦里得知的吧。”不过，也有一位评论者指出，鉴于此前据报道 Anthropic 的 Sonnet models 曾经击败当时的 Opus variants，这一传闻至少并非完全不可能。

    - 一位评论者认为，传闻中的 **Claude Sonnet 5 “Fennec”** 至少有一定可能性，因为 Anthropic 今年早些时候曾有一个 **Sonnet-tier model 击败当时的 Opus**，这说明较低定位的 model 超越旧款旗舰 model 并非没有先例。
    - 另一位评论者声称，**“Fennec”** 并不是新泄露的代号，而是一个更早的内部 codename，据称最早在 2 月就被用来指代 **Sonnet 4.6**。如果属实，那么这会削弱人们将它解读为 Sonnet 5 即将发布信号的依据。

  - **[Anthropic 内部的 Mythos 后继者现身](https://www.reddit.com/r/singularity/comments/1ubwtut/anthropics_internal_mythos_successor_emerges/)**（热度：1644）：**这张图片是 [Andrew Curran 的一条推文](https://i.redd.it/qrjnoo6zdo8h1.png)的截图，该推文转发并放大了一则**传闻**：Anthropic 已经训练出了一个能力更强的内部 model，作为尚未发布的 “Mythos” model 的后继者，名称可能是 **Mythos 5.1** 或 **Mythos 6**。帖子没有提供 benchmark、architecture details、evals 或 release plans；其技术意义主要在于，它声称 frontier labs 可能会继续推进内部 checkpoint，即使暂时不对外发布 model。**评论者大多认为这一说法属于有一定可能性的推测，并指出几个月时间足以进行新一轮 post-training，甚至重新开展一次 pretraining。部分讨论进一步延伸到对 access restrictions 的不满，有用户认为，禁止访问或不发布 model，可能会推动 acceleration 转向中国、欧洲，或 GLM 5.2 等 alternative models。

    - 一位评论者认为，传闻中 Anthropic “Mythos 后继者”的时间线在技术上是说得通的：如果第一个 Mythos checkpoint 大约在 **1 月或 2 月**已经存在，那么约 `5 months` 的时间足以进行另一轮 **post-training run**，对于大型 model 来说，甚至可能足以再进行一次 **pretraining run**。

  - **[据 The Economist 报道，NSA 称 Mythos 在数小时内入侵了其几乎所有机密系统](https://www.reddit.com/r/singularity/comments/1ubets2/nsa_says_mythos_broke_into_almost_all_of_their/)**（热度：2838）：**这张[图片](https://i.redd.it/o4nb07y8wj8h1.jpeg)是 X 用户 “Jimmy Apples” 发布的一条帖子的截图。该帖声称，**The Economist** 报道称，一个名为 **Mythos** 的 AI system “入侵了 NSA 几乎所有的机密系统”，用时“不是几周，而是几小时”；而 Reddit 标题则将其表述为 NSA 的官方说法。相关链接指向 The Economist 一篇需要付费阅读的、讨论 AI/export controls 的 briefing。评论者指出，其中的摘录似乎是在将 AI controls 与历史上对 “military encryption” 的管控进行比较，而不是提供经过独立佐证的技术事故细节。**评论区对此高度怀疑：有人问，如果 NSA 遭遇了如此灾难性的入侵，为什么没有被广泛报道；也有人认为，这一说法或许更多反映的是 NSA 的安全水平，而不是 Mythos 的能力。评论者还反驳了“Encryption is a potent technology, but narrow in its application”这句话，认为没有任何 AI 能合理地通过暴力破解 `AES-128` 或 `RSA-2048`；另一些人则将其理解为一种出口管制类比，即 AI 的 dual-use scope 更广。



- 有评论者质疑文章关于 AI 比加密技术*“用途更广”*、因而更值得关注出口管制的技术表述。有人指出，现代密码学不太可能被 AI 通过单纯的暴力搜索攻破：*“没有哪个 AI 会去暴力破解 `AES-128`，甚至是 `RSA-2048`。”* 这意味着，所谓的攻破更可能涉及软件漏洞、凭据窃取、配置错误，或社交工程及运营层面的攻击路径，而不是破解加密算法本身。
- 一条结合付费墙背景的评论认为，**The Economist** 可能是在将历史上对“军用加密技术”的出口管制，与当前对 AI 的出口管制进行比较，并据此主张 AI 的两用适用范围可能比加密技术更广。技术层面的反驳是：“加密”是一种较为狭窄的基础技术，而 AI 系统可以协助开展侦察、漏洞利用代码生成、自动化操作和补丁分析等工作——但如果不明确说明具体的攻破机制，这种宽泛的说法仍然缺乏说服力。


# AI Discord 社区

很遗憾，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但很快会推出全新的 AINews。感谢你一直读到这里，这段旅程曾经很美好。