---
companies:
- nous-research
- hugging-face
- cloudflare
date: '2026-06-19T05:44:39.731046Z'
description: '**GLM-5.2** emerges as a leading open-weight coding model rivaling **Opus
  4.8** and **GPT-5.5** in software engineering tasks, emphasizing the strategic importance
  of open models for provider competition, on-prem deployment, and fine-tuning rights.
  Experts like **Patrick Toulme** and **Thomas Wolf** highlight its frontier capabilities
  and structural impact on the AI ecosystem. The usability of GLM-5.2 heavily depends
  on serving infrastructure and agent harnesses, with tools like **sglang cookbooks**
  and **deepagents code** enhancing evaluation and deployment. In agent engineering,
  the focus shifts to orchestration patterns such as **agent fan-out** and **loop
  engineering**, with **Hermes Agent v0.17.0** advancing as a robust open agent stack
  supported by community-driven deployments. Additionally, **Cloudflare** is becoming
  a significant player in agent infrastructure.'
id: MjAyNS0x
models:
- glm-5.2
- opus-4.8
- gpt-5.5
people:
- patrick_toulme
- thomas_wolf
- andrew_ng
- meryem_arik
- banteg
- graham_neubig
- harrison_chase
- jared_from_cognition
- omar_sanseviero
- teknium
title: not much happened today
topics:
- open-weight-models
- coding
- agent-engineering
- agent-fan-out
- loop-engineering
- model-serving
- infrastructure
- software-engineering
- model-evaluation
- open-agent-stack
- session-compression
---

**平静的一天。**

> 2026年6月18日至6月19日的 AI 新闻。我们检查了 12 个 Reddit 分区、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未发现新的 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 综述

**GLM-5.2 的突破、开源代码能力以及智谱/DeepSeek 动态**

- **GLM-5.2 似乎是本周最具影响力的模型新闻**：多位从业者独立报告称，[GLM-5.2](https://x.com/gneubig/status/2067936197888930263) 是第一个让他们认真考虑在许多工作流中替代闭源模型的开源（Open-weight）代码模型，尽管在视觉和 Serving 方面仍有一些局限。[Patrick Toulme](https://x.com/PatrickToulme/status/2068134212587184442) 称其为“真正的领先级代码模型”，理由是它在本地部署时具有强大的工具使用能力、自主嵌套子 Agent、长程规划能力，以及接近 Opus 质量的代码生成。[Yuchen Jin](https://x.com/Yuchenj_UW/status/2068182756132376668)、[@_xjdr](https://x.com/_xjdr/status/2068138331192602730) 和 [@hrishioa](https://x.com/hrishioa/status/2068036265484992938) 也表达了类似观点，认为 GLM-5.2 在代码和设计任务上的表现经常接近 **Opus 4.8 / GPT-5.5** 级别。目前的共识并非它是“整体最强模型”，而是“开源模型现在已经真正进入了领先级 SWE（软件工程）范畴”。
- **实际意义在于模型独立性，而非仅仅是 Benchmark 吹嘘**：[Thomas Wolf](https://x.com/Thom_Wolf/status/2067996287530684826) 将 GLM-5.2 视为开源权重在结构上带来改变的证明：供应商竞争、本地部署（On-prem）、微调权以及更低的供应商锁定风险。这一主题在 [Andrew Ng](https://x.com/AndrewYNg/status/2068039709126017356) 和 [Meryem Arik (通过 ET Now)](https://x.com/etnshow/status/2067978328641007809) 的推文中反复出现，双方都认为最近对领先级专有模型访问的限制增加了开源模型的战略价值。此外还有成本角度：[banteg](https://x.com/banteg/status/2067960853844865306) 对“在家运行”的经济效益提出了质疑，认为以目前的 Token 价格，本地硬件投入相对于托管 API/订阅服务通常是不理智的。
- **Serving（部署服务）和测试框架（Harnesses）与模型本身同样重要**：多条推文强调，GLM-5.2 的可用性在很大程度上取决于基础设施和 Agent 框架的选择。[Graham Neubig](https://x.com/gneubig/status/2067952120540631493) 强调了 **sglang cookbooks** 为不同模型/硬件提供的精确部署设置，而 [@multimodalart](https://x.com/multimodalart/status/2068026613787217943) 展示了它可以通过 Hugging Face 路由到兼容 Claude Code 的接口。其他人则认为专有框架可能会低估开源模型的质量：[Harrison Chase](https://x.com/hwchase17/status/2068075256993169619) 建议使用 **deepagents code**，认为这比针对 Claude Code/Codex 优化的环境更适合作为评估 GLM-5.2 的模型无关方法。

**Agent 工程：扇出、循环可靠性以及 Hermes 的快速迭代**

- **Agent 工程的重心正在从“单个智能 Agent”转向编排模式**：[来自 Cognition 的 Jared](https://x.com/imjaredz/status/2068001458205720751) 将“**Agent 扇出**”（agent fan-out）描述为 Devin 内部的一种常见工作流：一个主 Agent 分解任务，并行产生 5-100 个子 Agent，并合并输出结果。其基本原理直接且在技术上合理：Agent 在上下文较小的窄任务上表现更好，而并行 VM（虚拟机）使得这种任务分解在经济上具有吸引力。这与日益强调的**循环工程**（loop engineering）这一核心学科相呼应，这一点在 [Omar Sanseviero 的推文](https://x.com/omarsar0/status/2068010014808092674)以及 [threepointone 计划深入探讨](https://x.com/threepointone/status/2067970619929510350)的关于构建跨客户端/服务器/推理故障的弹性 Agent 循环中可见一斑。
- **Hermes 正在迅速成熟为一套严肃的开放 Agent 栈**：Nous 发布了 [Hermes Agent v0.17.0 “The Reach Release”](https://x.com/NousResearch/status/2068056222457115126)，[Teknium](https://x.com/Teknium/status/2068058591186346450) 进一步补充了关于 Agent 分发（“agent distributions”）、会话压缩行为以及更广泛易用性的发布说明和使用技巧。社区动态展示了实际部署的势头：[iMessage 支持](https://x.com/Lonely__MH/status/2068171726291435808)，通过 Hermes 加 Kimi 即时生成的 GIS 工具（[Randy George](https://x.com/RandyGeorge17/status/2068020022019326201)），以及用户越来越多地发现隐藏的系统行为，例如上下文压缩规则（[@witcheer](https://x.com/witcheer/status/2068027535955468533)）。
- **Cloudflare 正悄然成为关键的 Agent 基础设施**：[Workers 上的临时账户](https://x.com/Cloudflare/status/2067956828290302374)允许 Agent 运行 `wrangler deploy --temporary` 而无需手动进行 OAuth，减少了最令人头疼的部署瓶颈之一。此外，Cloudflare 通过使 [Durable Objects 在活跃的出站连接和 WebSockets 期间保持存活](https://x.com/CFchangelog/status/2068047758271861012)，解决了长期运行 Agent 的一个关键问题，并添加了 [APAC 位置提示](https://x.com/CFchangelog/status/2067994912713322811)以降低延迟。这些虽是发布说明中的细微项目，但共同解决了长达数小时的 Agent 会话和部署循环中的实际运维痛点。

**模型访问、主权以及 Anthropic “Mythos/Fable” 冲击**

- **Anthropic 顶级模型的访问限制所产生的影响已远超一家公司**：多篇推文提到了 **Mythos/Fable** 可用性持续受阻的情况，有报告称[一些早期用户通过 Project Glasswing 保留了访问权限](https://x.com/kimmonismus/status/2067876984206537188)，随后又有消息称大约 [~200 家机构可能仍拥有访问权限](https://x.com/kimmonismus/status/2068038020394021000)。更大的战略启示在于：[Andrew Ng](https://x.com/AndrewYNg/status/2068039709126017356) 认为，厂商政策变化与美国政府出口控制的结合，正在加速全球对 AI 主权和开放替代方案的需求。如果对前沿智能的访问可以被突然撤销，那么这种依赖性本身就变成了产品风险。
- **治理对话正变得更加具体且由基准驱动**：[Rohan Paul](https://x.com/rohanpaul_ai/status/2067947789578125391) 总结了一个可能的转变，即从“消除所有越狱”等不可能实现的目标，转向对绕过严重程度、可重复性、暴露的能力以及下游危害进行分级评估。这比二元化的安全声明更具操作性，并符合行业向 Agent 和模型部署建立明确评估/控制平面的大趋势。
- **开源越来越多地被视为工程杠杆和地缘政治对冲**：[Natolambert](https://x.com/natolambert/status/2067974681135862167) 认为禁止开源 AI 将是一个错误，而 [Harry Stebbings 引用 Everett Randle 的话](https://x.com/HarryStebbings/status/2068032816286150974)指出西方开源模型相对于中国的弱势。本周反复出现的政策-工程综合论调是：开放权重不再仅仅是开发者的偏好，它们正被作为**主权基础设施**进行讨论。

**基础设施、推理与系统：Speculative Decoding、TPU 和文档解析**

- **推理工程持续高速发展，尤其是在吞吐量方面**：Modal 和 Z Lab 发布了[针对 Qwen 3.x 的六个新 Speculative Decoders](https://x.com/charles_irl/status/2068124629433262210)，最引人注目的声明是 Qwen 3.5 122B-A10B 在 **B200** 上实现了 **1k+ output tokens/sec**。如果这些数据在生产级工作负载中能够保持，Speculative Decoding 仍是实质性改变推理服务经济性最明确的杠杆之一。与此同时，Google 详细介绍了 **TPU 8i**，它针对训练后（Post-training）和高并发推理进行了优化，拥有更大的片上 SRAM、一个 **Collectives Acceleration Engine** 以及一种名为 [Boardfly](https://x.com/GoogleCloudTech/status/2068000858713841924) 的新服务拓扑结构。
- **开源文档提取领域迎来了一位重量级新成员**：[Vik Paruchuri](https://x.com/VikParuchuri/status/2067941596306231421) 发布了一个用于从文档中提取结构化数据的开源 **9B 模型**。据报告，该模型在内部基准测试中得分为 **90.2%**，而 Gemini 3.5 Flash 为 **91.3%**，且远超 **NuExtract3 (81.5%)** 等提取专家模型。其 **p50 耗时为 9.5s**，并支持基于 JSON-schema 的输出。对于构建文档工作流的团队来说，这是该系列发布中最具实际落地意义的项目之一。
- **不使用 VLM 的解析方案仍有胜出空间**：[Jerry Liu](https://x.com/jerryjliu0/status/2068005414369906856) 强调了 **LiteParse**，这是一个纯代码驱动的解析器。据称在处理 Markdown 密集型文档时，它的表现优于某些 VLM/OCR 系统，同时保持了免费和高效。这提醒我们，并非所有的文档智能问题都需要用到生成式多模态栈。

**科学、记忆与研究方向**

- **AI for Science 领域见证了强力的机制建模更新**：Google DeepMind 的研究人员推出了 [ATLAS (Active Theory Learning for Automated Science)](https://x.com/EltetoNoemi/status/2067920123122336138)，这是一个从数据中生成可解释机制模型并选择后续实验进行验证的流水线。这符合长期以来系统不仅做预测，还要提出结构化理论并选择干预措施的趋势。
- **Agent 记忆研究正变得更具部署性**：[DAIR.AI 对 AtomMem 的划重点](https://x.com/dair_ai/status/2067984002376749525)值得关注，因为它针对的是长效 Agent 中的真实失效模式：粗略的摘要会产生偏差，而无约束的记忆更新会损坏状态。AtomMem 采用**原子事实提取（Atomic Fact Extraction）**、分层事件结构和基于图的关联检索，在 **LoCoMo** 上报告了 SOTA 性能，同时力求保持足够的计算廉价以供产品使用。
- **从轨迹中挖掘技能（Skill Mining）仍有前景但尚不成熟**：[Omar Sanseviero 对一篇关于自动生成 `SKILL.md` 论文的总结](https://x.com/omarsar0/status/2067986774241251433)是一个很好的现实检查。该流水线可以将 GUI 轨迹聚类为高纯度的可读技能，但 RL 收益有限：**技能步骤准确率仅从 18.5% 提升至 20.5%**，BrowseComp+ 表现持平，且简单的先验方案仍具竞争力。良好的分解还不等同于有用的能力迁移。

**热门推文（按互动量排序）**

- **前沿实验室的人才流动**：[John Jumper 离开 Google DeepMind 加入 Anthropic](https://x.com/JohnJumperSci/status/2068001285173834106)，这是今年 AI 领域最大的人事变动之一。[Demis Hassabis](https://x.com/demishassabis/status/2068002732250640603) 的回应凸显了这次损失的规模。此前不久 Noam Shazeer 刚刚离职，引发了外界对 DeepMind 人才留存和产品走向的广泛关注。
- **具有实际技术内涵的应用 AI 奇闻**：[一个由 1,800 个使用 DeepSeek API 的机器人填充的《魔兽世界》(WoW) 私服](https://x.com/kimmonismus/status/2067924419947995471)是互动量最高的技术相关帖子。在梗的背后，它指向了一个反复出现的系统性问题：当模型推理足够廉价，可以模拟软件的整个社交层时，会发生什么？
- **Anthropic 使用限制重置**：[ClaudeDevs 重置了所有计划的 5 小时和每周限制](https://x.com/ClaudeDevs/status/2068122937308426676)。这一高关注度的运营调整可能反映了在 Fable/Mythos 中断期间的需求压力和用户不满。
- **Figure 的部署里程碑**：[“在 Figure，机器人的数量首次超过了人类”](https://x.com/adcock_brett/status/2068040783295627609)。虽然缺乏运营细节，但作为具身 AI (Embodied AI) 领域规模化叙事和劳动力替代构想的一个信号，仍值得注意。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. GLM-5.2 基准测试与本地推理

- **[新的 Agentic 基准测试发布：Claude Fable 和 GLM 5.2 在同类模型中领先](https://www.reddit.com/r/LocalLLaMA/comments/1u9yt6v/new_agentic_benchmark_out_claude_fable_and_glm_52/)** (热度: 328): **该[图片](https://i.redd.it/9xcqvw4ny78h1.png)是来自 **Artificial Analysis** 的技术柱状图，展示了 **AA-Briefcase Elo**。这是一个新的 Agentic 知识工作基准测试，旨在测试 LLM 的规划/执行能力而非静态问答；该帖子在此链接了[方法论/文章](https://artificialanalysis.ai/articles/aa-briefcase)。** **具有回退机制的 Claude Fable 5** 以 `1587` 分领跑，远高于 **Claude Opus 4.8** 的 `1356` 和 **GLM-5.2** 的 `1266`。图中包含置信区间，数据日期为 **2026 年 6 月 18 日**；正文强调该基准测试“尚未饱和”，减少了明显的刷榜（benchmark-gaming）疑虑。** 评论集中在模型排名含义上——例如，担心 **Mistral** 远远落后，以及对“Claude Fable”是否真实或命名是否准确表示怀疑。最专业的一条技术评论认为，Agentic 基准测试需要具有重复运行、方差、工具权限详情、超时策略和失败类别的可复现环境，因为*“一次幸运的轨迹”*可能会夸大一个不稳定的 Agent 的分数。

    - 一位评论者认为，在标题排名具有意义之前，基准测试需要更强的可复现性元数据：**重复运行**、得分方差、工具权限、超时策略以及分类的故障模式。他们指出，在 Agentic 评估中，如果结果基于过少的试验，*“一次幸运的轨迹”*可能会夸大模型表象上的可靠性。
    - 一个技术对比贴指出，**Mistral Medium 的排名据报道高于 Gemini 3.1 Pro**，这令人惊讶，同时仍将 **Mistral 3.5 Medium** 视为本地实验室（local-lab）部署的实际选择。同一评论者强调了 **MiniMax 3** 的良好表现，暗示其训练或微调可能优先考虑了 Agentic 工作流，而非广泛的基准测试优化。

  - **[GLM-5.2 成为 Artificial Analysis Intelligence Index 上新的领先开源权重模型](https://www.reddit.com/r/LocalLLaMA/comments/1u9zqlx/glm52_is_the_new_leading_open_weights_model_on/)** (热度: 468): **[Artificial Analysis](https://artificialanalysis.ai/articles/glm-5-2-is-the-new-leading-open-weights-model-on-the-artificial-analysis-intelligence-index) 报告称 **Z.ai GLM-5.2** 现已成为 Intelligence Index v4.1 上排名第一的开源权重模型，得分为 `51`，同时保持了 GLM-5.1 的 `744B` 总参数 / `40B` 激活参数的 MoE 架构。报告显示最大的提升在于科学/Agentic 评估——`CritPt +16`、`HLE +12`、`TerminalBench v2.1 +16` 以及 `GDPval-AA v2 = 1524`——采用 **MIT 许可证**，具备 `1M` 上下文，API 价格为每 `1M` tokens：输入 `$1.4` / 缓存命中 `$0.26` / 输出 `$4.4`。它处于智能与成本的帕累托前沿（Pareto-frontier）位置，尽管其每个任务平均输出 token 数高达 `43k`。** 评论者对 **GLM**、**DeepSeek** 和 **Qwen** 等开源权重的中国前沿模型表现出比 Fable 更浓厚的兴趣，同时也询问了是否有更小或变体版本（如 “Flash”/“Air”）发布，并指出了缺乏视觉支持的问题。

    - 针对技术层面，有人提出 **GLM-5.2** 是否可以蒸馏到其他大型开源权重架构中，如 **Qwen 3.6 122B** 或 **Nemotron 3 Super**，这暗示了用户对于将 GLM-5.2 的推理/性能特性转移到更易获取或具有不同优化方向的基础模型中的兴趣。
    - 一位用户报告了一个轶事性的软件架构测试，其中 **GLM-5.2** 犯了多个实现错误：选择了*过时或冗余的 crates*，并因为在每次块写入后调用 `fsync` 而引入了严重的性能问题。在同一个 prompt 下，据报道 **MiniMax 3** 产生了更好的结果，导致该评论者推测 GLM-5.2 可能具有强大的后训练（post-training），但可能使用的是较旧或较弱的代码数据集。
    - 功能缺失的一个主题是 GLM-5.2 缺乏**视觉/多模态支持**，评论者还询问了更小、更快的变体，如 **GLM-5.2 Air** 或 **Flash**，可能是为了低延迟或更廉价的部署场景。

- **[GLM-5.2 现在可以在 llama.cpp 和 Unsloth Studio 中本地运行。](https://www.reddit.com/r/LocalLLaMA/comments/1u9vfhf/glm52_can_now_run_locally_in_llamacpp_and_unsloth/)** (热度: 435): **该图片是 [GLM-5.2-GGUF 量化](https://i.redd.it/3hzm5bu8078h1.jpeg) 的技术基准测试散点图，展示了磁盘占用大小与 top-1 token 一致性（以 `Q8_0` 作为 `100%` 参考）。核心观点是 **Unsloth 将 GLM-5.2 从 `1.51TB` 压缩到了 `238GB`**，通过 2-bit GGUF 变体保留了约 `82%` 的 token 一致性，从而能够在拥有超大内存的系统（如 `256GB` 的 Mac 或 RAM/VRAM 组合配置）上通过 **llama.cpp** 或 **Unsloth Studio** 进行本地推理。提供的链接包括 [Unsloth GLM-5.2 指南](https://unsloth.ai/docs/models/glm-5.2) 和 [Hugging Face 上的 GGUF 权重](https://huggingface.co/unsloth/GLM-5.2-GGUF)。** 评论大多持怀疑态度或开玩笑：一位用户认为 `~82%` 的一致性意味着大部分输出可能不可靠，而另一些人则调侃说，由于其极高的内存要求，llama.cpp 的支持并不能让大多数用户实际运行该模型。

    - 一位评论者认为报告的 **`82%` 准确率** 具有误导性，因为它是针对 `llama.cpp` 中的 `Q8_0` 输出测量的，而不是 `BF16` 参考基准。他们还指出，据称 `llama.cpp` 缺乏适当的 GLM-5.2 实现，并且产生的输出已经与参考实现产生偏差，引用了 [ggml-org/llama.cpp issue #24730](https://github.com/ggml-org/llama.cpp/issues/24730)。另一位评论者补充说，**top-1 token 一致性** 可能不足以评估本地实现的正确性或保真度。

  - **[GLM-5.2 是最佳的开放权重创作模型](https://www.reddit.com/r/LocalLLaMA/comments/1u98pc9/glm52_is_the_best_open_weight_creative_writing/)** (热度: 371): **该图片是来自 Sam Paech 的 EQ-Bench [创作基准测试](https://eqbench.com/creative_writing.html) 的技术排行榜截图，显示 **GLM-5.2** 是排名最高的“开放权重”创作模型，其 **Elo 评分为 `1821.0`**，**Rubric 评分为 `82.20`**。它排在 **claude-fable-5**、**claude-opus-4-7** 和 **gpt-5.5** 等闭源领先模型之下，但高于 **Kimi-K2.6** 和 **Kimi-K2-Instruct** 等其他开放权重竞争者，这使得帖子声称其为最佳开放权重创作模型的观点与展示的表格一致。图片：[https://i.redd.it/oj35cq74328h1.png](https://i.redd.it/oj35cq74328h1.png)** 评论者对 GLM-5.2 显而易见的性价比印象深刻，并认为创作基准测试可能比标准评估更难通过“刷榜”（benchmaxx）来优化。提出的一个疑虑是 **Claude 被用作 LLM 评委**，因此评论者质疑它是否可能偏向类 Claude 的写作风格或 Anthropic 模型。

    - 评论者指出 **GLM-5.2** 在创作基准测试中得分很高，而据报道其成本显著低于排名更高的模型。一位用户认为，这种类型的基准测试可能比标准的推理/问答排行榜更不容易受到“刷榜”优化的影响。他们还强调了 **GLM 在 EQBench 上的快速进步**，推测未来的 **GLM-6** 可能会在创作评估中超越 **Claude Opus 4.7o**。
    - 几位用户对使用 **LLM-as-judge**（LLM 作为评委）来设定主观写作质量的有效性表示怀疑，尤其是因为 **Claude** 显然被用作评审模型，可能会偏向与其自身风格相似的输出。一个更合理的用例建议是进行客观的指令遵循检查——例如长度限制、提示词主题匹配——而不是定性的文学排名。
    - 一位评论者在基准测试中检查了最近的中型模型，发现了 **Gemma-4-31B** 和 **Gemma-4-26B-A4B** 的条目，但注意到缺少类似的 **Qwen3.6/Qwen3.5 中型模型**。他们链接了排行榜的截图：https://preview.redd.it/oo52ln0t828h1.png?width=1194&format=png&auto=webp&s=b37390b89f1f577661e587ed10692ffea3f2939b


### 2. Open Agentic Research and Coding Models

- **[研究人员使用 32 张 H100 训练了一个 Deep Research Agent 并开源了所有内容](https://www.reddit.com/r/LocalLLaMA/comments/1u9w6my/researchers_trained_a_deep_research_agent_with_32/)** (热度: 816): **这张 [图片](https://i.redd.it/hdrqhare878h1.png) 是一个技术基准测试图表，而非梗图：它展示了来自 **Ohio State University** 的开源 “Deep Research” Agent —— **QUEST-35B**，该模型在包括 `BrowseComp`、`Mind2Web 2`、`HLE`、`DeepResearch Bench`、`GAIA` 和 `LiveResearchBench` 在内的多个排行榜上表现突出。根据帖子内容，QUEST-35B 据报道是使用约 `32× H100` GPU 在大约 `8K` 个合成样本上训练而成的，其代码、权重、数据集和训练配方均已开源；图表将其定位为与 **Gemini**、**Claude/Opus**、**GPT** 和 **Kimi** 等封闭系统具有竞争力的产品，包括在 `Mind2Web 2` 和 `GAIA` 上取得的顶级名次。** 评论者质疑到底发布了什么——是基座模型、微调模型还是完整的 Agent 框架（Harness）——以及基准测试的提升是否反映了真实的科研能力、预设的推理/搜索脚手架，还是可能的合成数据过拟合。还有人对仅凭 `8K` 个合成样本就得出强有力结论表示怀疑。

    - 评论者质疑实际开源的内容：该工作是一个 **新的基座模型、一个微调模型、一个 Agent 框架，还是仅仅一个 Prompting/Thinking 方案**。核心技术考量在于，“Deep Research Agent” 不仅仅需要模型权重，还需要工具调用编排（Tool-use Orchestration）、搜索/检索、引用处理、评估框架和工作流逻辑，因此其实用性取决于是否包含这些基础设施。
    - 一位评论者对所报道的评估规模表示怀疑，指出 *“人们在 2026 年仍然相信 `8k` 样本的结果。”* 这意味着，除非有更大规模、多样化的基准测试和稳健的 Agent 评估协议支持，否则关于深度研究能力的说法在统计或方法论上可能是薄弱的。
    - 另一个技术问题是，为什么深度研究根本需要一个 **Fine-tuned 模型**，因为像 **ChatGPT** 和 **Claude** 这样的前沿系统使用其标准模型就能展示研究模式。这引发了微调与 Agent 工作流之争：研究性能主要来自模型专业化，还是来自诸如规划、网页搜索、检索、验证和报告综合等外部编排。

  - **[poolside/Laguna-M.1 · Hugging Face - 225B-A23B](https://www.reddit.com/r/LocalLLaMA/comments/1u9b2i3/poolsidelagunam1_hugging_face_225ba23b/)** (热度: 354): ****poolside** 发布了 [`Laguna-M.1`](https://huggingface.co/poolside/Laguna-M.1)，这是一个采用 **Apache-2.0 协议的 Open-weight** 文本 MoE 编程/Agent 模型，拥有 `225B` 总参数 / `23B` 激活参数、`70` 层、`67` 个稀疏 MoE 层、`256` 个专家（`top-k=16`）、全局注意力、RoPE+YaRN 以及 `262,144` Token 的上下文窗口。报告的编程 Agent 基准测试显示其在 SWE-bench Verified 上为 `74.6%`，在 SWE-bench Multilingual 上为 `63.1%`，在 SWE-bench Pro 上为 `49.2%`，在 Terminal-Bench 2.0 上为 `45.8%`——与 Devstral 2 和 GLM-4.7 等开源模型具有竞争力，但在多个列出的指标上低于 DeepSeek-V4 Flash / Qwen3.5。一位评论者指出，此次发布包括 **BF16、FP8 和 NVFP4** 格式的基座和后训练变体，而另一位评论者指出，较小的 [`Laguna-XS.2`](https://huggingface.co/poolside/Laguna-XS.2) / `33B-A3B` 模型仍在等待 [`llama.cpp` 支持](https://github.com/ggml-org/llama.cpp/issues/23249)。** 评论者普遍对 poolside 发布开放权重的旗舰模型持积极态度，认为这类发布尽管缩小了与私有编程 Agent 的差距，但仍未得到足够重视。一位评论者建议对比应包括 **Mistral Medium 3.5 128B**，但将 Laguna M.1 描述为可能是美国训练的最强 Open-weight 编程模型。

- **poolside Laguna M.1** 被强调为罕见的 Apache-2.0 开源权重“旗舰级” coding-agent 发布：模型规模为 `225B-A23B`，提供 base 和 post-trained 变体，支持 `BF16`、`FP8` 和 `NVFP4` 权重，并在 **SWE-Bench Pro** 上取得了 `49.2%` 的成绩。一位评论者指出，OpenRouter 的非正式测试表明该模型“整体表现确实出色且均衡”，尽管其体积对于典型的本地硬件来说过大。
- 针对较小的 **Laguna-XS.2 / 33B-A3B** 模型存在实现与支持方面的担忧：据报道该模型仍待 `llama.cpp` 的支持，相关讨论见 [`ggml-org/llama.cpp#23249`](https://github.com/ggml-org/llama.cpp/issues/23249)，模型托管于 [`poolside/Laguna-XS.2`](https://huggingface.co/poolside/Laguna-XS.2)。评论者特别强调需要 `llama.cpp` 的支持，以使本地推理（local inference）更具实用性。
- 一位评论者认为，基准测试对比集应包括 **Mistral Medium 3.5 128B**，并建议将其作为评估 Laguna M.1 编程性能更具相关性的基准（baseline）。他们将 Laguna M.1 视为目前来自美国公司的最强开源权重编程模型，但暗示这一结论仍取决于更广泛的直接对比评估。

### 3. 开放模型成本与采用趋势

  - **[开源模型在性价比上开始超越前沿（Frontier）模型](https://www.reddit.com/r/LocalLLM/comments/1u9gohq/open_source_is_starting_to_beat_frontier_on/)** (热度: 441)：**该帖子引用了一张散点图 ([图片](https://i.redd.it/a4obm010k38h1.jpeg))，将 “Artificial Analysis Intelligence Index” 与对数尺度的美元运行成本轴进行对比，认为 **DeepSeek, GLM, Qwen, Kimi/MiniMax** 等开源/开放权重模型正在进入高智能/低成本的“绿色象限”。帖子的技术观点是，虽然像 **Claude Opus/Fable** 或 **GPT-5.5** 这样的封闭前沿 API 在能力上可能仍然更高，但对于许多不需要绝对峰值能力的生产负载（Production Workloads）来说，性价比前沿（Cost-performance frontier）正在向开放模型转移。** 评论者意见分歧：一些人认为这种情况已经持续多年，本地模型现在已经可以媲美几年前的顶级模型；另一些人则批评该图表过于简化，因为真正的性价比取决于**特定任务的有用工作**、Token 效率、提示词（Prompting）、编排（Orchestration）以及部署框架（Harness），而不仅仅是两个聚合基准测试轴。

    - 一位评论者认为，**性价比无法通过双基准图表来体现**，因为真正的指标是“完成每项有用工作的成本（Cost per useful work accomplished）”。他们指出，Token 使用量因任务、模型、提示词、框架和编排策略的不同而有很大差异，因此仅凭基准测试分数可能会误导实际效率。
    - 几位评论者将开源/本地模型定位为已经达到**大约几年前的前沿模型水平**，这使得它们对许多用户来说已经足够好，即使不是最先进的（State of the art）。提出的一个警告是，如果开源模型主要是从前沿模型**蒸馏（Distilled）**而来，而不是独立推进前沿，那么它们在结构上可能仍然落后。
    - 一个轶事式的编程对比声称，**GLM 5.2** 在修复损坏的实现方面表现优于 “Sonnet 4.6”：据称 GLM 避免了破坏无关功能，而 Sonnet 则持续尝试修复。这虽然不是基准测试，但它突显了任务级别的差异，在特定的调试工作流中，低成本/开源模型可能是更好的选择。

  - **[OSS 模型在市场份额上决定性地超越了私有模型（基于过去 3 个月的 OpenRouter 数据）](https://www.reddit.com/r/LocalLLaMA/comments/1u96545/oss_models_decisively_overtook_proprietary_models/)** (热度: 319)：**Dirac 的 [OpenRouter Token 份额仪表板](https://dirac.run/labs-market-share) 声称，在 **OpenRouter API 流量**中，开源/开放权重模型实验室在过去约 3 个月内逆转了市场份额：从 2026 年 3 月的约 `40% OSS / 60% 私有` 变为 2026 年 6 月中旬的约 `60% OSS / 40% 私有`，总使用量接近 `~6T tokens/day`。该分析按*模型创建实验室*而非 API 托管方聚合输入+输出 Token，并明确排除了 3 月 18 日至 4 月 2 日期间小米 `mimo-v2-pro-20260318` 免费模型的流量，以避免扭曲份额计算。** 评论者质疑 OpenRouter 是否具有更广泛 LLM 市场的代表性：**Claude** 或 **GPT** 的用户通常通过第一方订阅或直接 API 访问，而不是通过 OpenRouter，因此该图表可能主要反映了 OpenRouter 的用户群体而非全球采用情况。“决定性地”一词也受到挑战，因为消费者订阅的使用情况并未被 API Token 市场份额捕获。

    - 几位评论者挑战了这一方法论，认为 **OpenRouter 的流量不能代表整个 LLM 市场份额**，因为大多数 GPT/Claude 的使用发生在第一方订阅或直接 API 上，而不是通过 OpenRouter。关键的技术警告是，数据可能仅反映了**路由/API 用户子群体**，而不是更广泛的消费者或企业市场。
    - 一位评论者强调了核心图表的主张：在 OpenRouter 过去 3 个月的使用中，OSS 模型的份额据称从大约 **`40%` vs `60%` 私有** 变成了相反的 **`60%` OSS vs `40%` 私有**。这支持了 *OpenRouter 流量内部* 的强劲转变，但不一定代表整个 LLM 市场。

## 低技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic Fable/Mythos 访问限制

  - **[Anthropic 国际总经理表示，“有信心在未来几天内 [Fable 5] 将再次可用”](https://www.reddit.com/r/ClaudeAI/comments/1u95x3l/anthropic_is_confident_that_in_the_coming_days/)** (热度: 1019): **Anthropic** 的国际总经理表示，公司“有信心”在未来几天内恢复对 **Mythos/Fable 5** 的访问。此前，为响应限制外籍人士访问的白宫安全指令，这些模型在全球范围内被禁用（[Korea JoongAng Daily](https://www.koreajoongangdaily.com/business/anthropic-confident-of-reenabling-mythos-fable-5-access-in-coming-days-executive/12727522)）。报告将问题核心指向 **Mythos 先进的网络安全/代码分析能力** 以及 **Project Glasswing**。这是一个包含约 `150` 个合作伙伴的受控访问计划，其中包括美国科技公司以及 **Samsung Electronics**、**SK hynix** 和 **SK Telecom** 等韩国公司；首尔会议的背景表明，Anthropic 预计恢复范围将是全球性的，而非仅限于美国。评论者对 Anthropic 能否在政策多变的背景下自信预测时间点表示怀疑，有人称之为“对这种事感到自信是很愚蠢的”。另一位评论者提到，企业客户已经在要求供应商保证脱离**美国拥有的 AI 解决方案**，这暗示停运正在加速主权/欧盟对齐的采购讨论。

    - 一位评论者报告了可用性争议对企业的具体影响：据称有*三个独立客户*要求保证其组织正在**脱离美国拥有的 AI/云解决方案**，从而启动了独立的**欧洲办公室 / 欧盟托管解决方案轨道**。技术层面的相关影响是，对司法管辖区隔离、数据驻留以及针对美国 AI 供应商的供应商风险缓解的需求增加。
    - 另一位评论者认为，如果 Anthropic 的先进模型发布可以因“安全风险”标签而被阻止，那么 Anthropic 在受影响的市场中实际上可能被限制在 **Opus 级别产品**。令人担忧的是，未来的前沿版本可能会面临反复的监管/出口式中断，使得高端 Anthropic 模型的可用性保证变得不可靠。

  - **[约 200 家公司在美国停运令后仍可访问 Anthropic Mythos](https://www.reddit.com/r/ClaudeAI/comments/1u9sq81/about_200_companies_still_have_access_to/)** (热度: 949): **Bloomberg** 报道称，**Anthropic 的 Project Glasswing**（一个在漏洞研究场景下测试先进 AI 系统的网络安全合作伙伴计划）中约有 `200` 个组织在最近美国政府限制广泛访问 **Fable 5** 和 **Mythos 5** 之后，仍保留对 **Mythos Preview** 的访问权限（[Bloomberg](https://www.bloomberg.com/news/articles/2026-06-19/early-users-of-anthropic-mythos-still-have-access-after-us-order?embedded-checkout=true)）。据报道，保留访问权限的早期参与者包括 **Cisco**、**Amazon Web Services (AWS)** 和 **JPMorgan Chase & Co.**，而更广泛的访问仍处于停滞状态。评论者关注 **Amazon/AWS** 保留访问权限一事，并指出其中的讽刺意味：据称 Amazon 向政府投诉了 Anthropic，却未被从特权访问组中移除。

    - 一位评论者指出，尽管有停运令，据报道 **Amazon 仍能访问 Anthropic Mythos**，并指出明显的矛盾点，即 Amazon 据称也是向政府投诉 Anthropic 的方之一。这与其说关乎模型性能，不如说关乎政府指令后的**选择性访问控制 / 执行范围**。

  - **[更新：Anthropic 提议取消美国对 Mythos 和 Fable AI 模型的限制](https://www.reddit.com/r/ClaudeAI/comments/1u9fd4z/update_anthropic_floats_proposal_to_lift_us/)** (热度: 947): 据报道，**Anthropic** 已向美国商务部（致商务部长 **Howard Lutnick**）提交了一份框架提案，旨在取消对其 **Mythos/Fable AI 模型**的访问限制。该框架的核心是加强与白宫的沟通、正式的合作承诺以及更快地解决政府的安全顾虑。帖子未提供 Model Card 详情、基准测试、能力评估、威胁模型细节或实现更改；报告的状态仅为谈判“进展顺利”，尚无公开的时间表。热门评论大多非技术性且持怀疑态度，暗示监管结果可能受资金或政治影响，且包含与 Epstein 相关的离题引用，而非对出口管制、模型安全或安全审查标准的实质性讨论。

### 2. 前沿模型竞赛传闻

  - **[Z.ai 创始人有信心在年底前打造出 Fable 级别的 GLM 模型](https://www.reddit.com/r/singularity/comments/1u9b5vb/zai_founder_is_confident_that_they_can_make_a/)** (活跃度: 1341)：**这张[图片](https://i.redd.it/qmh9mjvjj28h1.png)是一个深色模式下的 X/Twitter 交流界面，其中 **Elon Musk** 估计中国可能在 **Q1** 达到 “Fable 级别” 的 AI 能力，而 **jietang/Z.ai** 回复道 *“用不了那么久，”* 暗示 Z.ai 预计在年底前推出该级别的 **GLM 家族模型**。由于文中没有展示基准测试、架构细节、评估结果或发布计划，因此该帖子主要是一个**主张/预测，而非技术证据**。** 评论者们持怀疑态度，有人评论道 *“空口无凭”*，并认为 Z.ai 在讨论 “Fable 级别” 能力之前，应该先展示出 **Opus 级别**的模型；另一些人则对更强大的开源前沿模型表示欢迎。

    - 一个有实质内容的讨论帖质疑了在尚未展示 **“Opus 级别”** 模型之前就声称近期能实现 **“Fable 级别” GLM** 的可信度，将此问题视为能力扩展的里程碑，而非路线图上的声明。另一位评论者引用了 **OpenAI Sora** 之后竞争对手迅速涌现的先例，认为中国实验室可能仅落后前沿 SOTA `3–6 个月`，这是能力快速扩散的体现。

  - **[据报道 DeepMind 目前正艰难应对与 Anthropic 和 OpenAI 的竞争，而 3.5 Pro 并不是他们保持竞争力所需的阶跃式进步](https://www.reddit.com/r/singularity/comments/1ua75fz/deepmind_is_now_reportedly_struggling_to_compete/)** (活跃度: 958)：**Reddit 的一则帖子引用了来自 **synthwavedd** 未经证实的 X 传闻，称 **Google DeepMind/Gemini 3.5 Pro** 可能仍落后于 **Anthropic** 和 **OpenAI**，发布者预计它在创意/百科知识任务上会比 **Agent 式编程**或递归自我改进型工作流更强（[来源](https://x.com/synthwavedd/status/2068000857757741251)）。评论者认为 Gemini 的产品/模型界面分散在 **AI Studio**、**Gemini 网页端/移动端**和 **Antigravity** 中，同时 Gemini/Flash 的定价和编程性能被认为相对于一些中国实验室和前沿竞争对手正在恶化。** 主要争论点在于 Google 的基础设施/数据/现金流优势是否应该转化为模型领导力，以及 Google 的公司/产品扩张冗余是否减缓了 DeepMind 的执行速度。几位评论者对 **Gemini 3.5 Pro** 抱有较低的期望，认为如果它是一个重大的阶跃式进步，很可能已经在 I/O 大会上展示了。一位评论者将 **John Jumper** 转投 Anthropic 视为 Google DeepMind 研究优势的战略性损失。

    - 评论者认为 **Gemini 的产品/模型碎片化**可能会损害其采用率：Gemini 网页端/移动端、AI Studio、Antigravity 以及 Flash 的价格变动被指造成了生态系统的分裂。一种技术层面的批评是，与领先的 OpenAI/Anthropic 模型相比，Gemini 虽然拥有强大的通用/百科知识，但在编程方面“极其懒惰”且表现较弱，而中国实验室被认为在某些模型发布中正在追赶或超越 Google。
    - 一场实质性的战略辩论将 **Google DeepMind 更广泛的 AGI 论题**与 Anthropic/OpenAI 以 LLM 为中心的方法进行了对比。一位评论者指出，DeepMind 正在投资*语言模型、世界模型和更广泛的 AI 系统*，这符合 Demis Hassabis 的观点，即仅靠 LLM 可能不足以实现 AGI；而 Dario Amodei 则被描述为更乐观地认为规模化的 LLM 式系统可以实现这一目标。
    - 几条评论将 Google 的问题归结为组织架构而非纯技术问题：大公司的指标优化可能更倾向于渐进式的产品改进，而非高风险的模型突破。一位评论者链接了 Steve Yegge 关于 Anthropic 工程文化的文章 [《Anthropic 的蜂群思维》(The Anthropic Hive Mind)](https://steve-yegge.medium.com/the-anthropic-hive-mind-d01f768f3d7b)，认为 Anthropic 让工程师探索许多投机性想法的意愿，可能会比 Google 这种 KPI 驱动的结构产生更多前沿模型的创新。

### 3. 实战 AI 工具发布

  - **[已发布的实时捕捉政客谎言的事实查核工具](https://www.reddit.com/r/ChatGPT/comments/1u9ctgb/published_factchecker_that_catches_politicians/)** (热度: 1317): **作者发布了 **InTruth**，这是一个 BYOK Chrome 扩展程序，用于对任意视频进行实时政治事实查核，其工作流为：**Deepgram 转录 → Serper 搜索验证来源 → Claude 生成裁定结果**；演示基于 `2024` 年美国总统辩论。Chrome Web Store 列表位于 [此处](https://chromewebstore.google.com/detail/intruth/ikmpglbpcdoapfelcbfpoaddmhmaaocg)；引用的 Reddit 演示视频因 `403 Forbidden` 无法访问。** 热门技术反馈询问该项目是否会在 GitHub 上开源以及如何实现主张检测（claim detection）；一位评论者建议将类似的流水线集成到未来的智能眼镜中。

    - 评论者关注系统的**主张检测流水线**，询问它如何实时识别可查证的事实主张，而不仅仅是对明显的陈述做出反应。一个关键的技术担忧是模型在检索/验证之前是否执行了显式的主张提取，特别是对于陈述可能存在歧义、复合或带有修辞色彩的现场政治演说。
    - 几条评论质疑演示是依赖于模型训练数据中已有的事实，还是真正的**实时检索增强事实查核（RAG fact-checking）**工作流。一位评论者指出，对于实际部署，证据需要从多个来源提取并动态评估，而不仅仅是与 AI 模型中可能已经编码的详尽记录的主张进行匹配。
    - 提出的一个主要可靠性问题是**来源信任与检索操纵**：如果系统使用网页搜索结果验证主张，它如何确定这些来源是真实的？评论者特别提出了 SEO 优化或对抗性页面可能影响证据集的风险，暗示需要来源排名、溯源检查以及对搜索结果污染（search-result poisoning）的抵抗力。

  - **[我为 FLUX.2 [klein] 构建了一个单一 ComfyUI 节点：包含 T2I, I2I, 编辑, 局部重绘, 扩图, 草图, 换脸等功能](https://www.reddit.com/r/StableDiffusion/comments/1u9g3vy/i_built_a_single_comfyui_node_for_flux2_klein_t2i/)** (热度: 935): **作者发布了 **One Node · FLUX.2 [klein]**，这是一个独立的 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 自定义节点，它将包括**文生图（T2I）、图生图（I2I）、编辑、局部重绘（inpaint）、扩图（outpaint）、草图（sketch）和换脸（faceswap）**在内的 FLUX.2 工作流整合到一个组件中，并在 [YouTube](https://youtu.be/L4ItbBWXqCo) 上提供了设置/教程，源代码托管在 [GitHub](https://github.com/yanokusnir-ai/one-node-flux-2-klein)。2026 年 6 月 19 日的更新增加了**外部加载器支持**（包括 `GGUF`）、**模型刷新**按钮以及对草图功能的**数位板/压感**支持，详见项目 [更新日志](https://github.com/yanokusnir-ai/one-node-flux-2-klein#changelog)。** 热门评论非常正面，称其为他们见过的“最好的节点之一”，并对计划中/相关的 *“即将登陆 ltx”* 的移植版本表示关注；提供的评论中没有出现实质性的技术批评或基准测试讨论。

    - 一位用户报告了一个初始的 UI/显示 Bug：生成过程“干净且快速”完成，输出出现在媒体资产中，但图像预览未在节点窗口内显示。他们表示使用 Claude Code 修复了该自定义节点，随后成功测试了 LoRA 设置以及 I2I、编辑和换脸工作流。
    - 几位评论者认为该节点有效地将 **A1111 风格的全功能工作流引入了 ComfyUI**，将 T2I/I2I/编辑/局部重绘/扩图/草图/换脸整合到一个界面中，而不需要许多独立的图节点。
    - 一位评论者指出，相同风格的集成节点 *“即将登陆 ltx”*，暗示了对 FLUX.2 [klein] 之外的 **LTX** 模型提供计划支持或类似的统一工作流。





# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新版的 AINews。感谢读到这里，这是一段美好的历程。