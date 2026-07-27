---
companies:
- openai
- broadcom
- qualcomm
- modular
- nvidia
- skypilot
- modal
- anthropic
- hugging-face
date: '2026-06-24T05:44:39.731046Z'
description: 'OpenAI 宣布推出 Jalapeño，这是其首款为大语言模型推理定制的 AI 芯片，由该公司与 Broadcom 联合打造。OpenAI
  希望借此掌控更多 AI 技术栈，并通过仅 9 个月的快速设计周期改善算力经济性。社区分析认为，Jalapeño 配备了 216GB HBM3E，带宽约为 7.1–7.4
  TB/s，FP4 性能约为 10 PFLOPS，这表明，面向超大规模云服务商的推理芯片正成为新的行业标准。


  与此同时，Qualcomm 正在收购 Modular，Mojo 的开源工作也在按计划推进，显示出除 NVIDIA/CUDA 之外，垂直整合的推理技术栈正面临越来越激烈的竞争。在基础设施方面，NVIDIA
  的 NeMo AutoModel 可将 MoE 模型的训练吞吐量提升 3.4–3.7 倍，SkyPilot 和 Modal 等初创公司也在推动统一化、开源的推理解决方案。对
  DFLASH 模型进行定制训练后，解码速度可提升 30–50%。


  在用户体验方面，Anthropic 原生运行于 Slack 的 Claude 智能体，将智能体交互模式从“调用工具”转向“与同事协作”，但也引发了围绕身份、权限和厂商锁定的新安全与成本问题，同时关于能力型安全和责任归属的讨论也在持续。Hugging
  Face 则推出了自托管的 Slack 编程智能体 Moon Bot 作为回应。

  '
id: MjAyNS0x
models:
- dflash
- nemo-automodel
- claude
people:
- gdb
- kimmonismus
- scaling01
- clattner_llvm
- karpathy
- gallabytes
- dabit3
- kentonvarda
- random_walker
- jubbaonjeans
- victormustar
title: '今天没发生什么特别的事。

  '
topics:
- hardware
- inference
- performance-optimization
- model-training
- agent-ux
- security
- capability-based-security
- open-source
- fine-tuning
- infrastructure
- model-optimization
---

**平静的一天。**

> 2026 年 6 月 23 日至 6 月 24 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看其他 Discord。你可以通过 [AINews 网站](https://news.smol.ai/) 搜索往期所有内容。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同的邮件发送频率！




---

# AI Twitter 速览


**OpenAI 的 Jalapeño Chip，以及迈向全栈 AI 基础设施的竞赛**

- **OpenAI 更深入地布局硬件**：[OpenAI](https://x.com/OpenAI/status/2069770172802773292) 宣布推出 **Jalapeño**，这是其首款为 LLM 推理定制的 AI 芯片，由 **Broadcom** 共同打造，计划用于 ChatGPT、Codex、API 流量以及未来的 Agent 产品。其战略意图非常明确：掌控更多技术栈——芯片、内核、内存、网络、调度和部署——从而降低计算成本和产品行为对市场化 GPU 供应的依赖。[上文提到的 @gdb](https://x.com/gdb/status/2069809298612621629) 强调了出色的**每瓦性能**；而 [@kimmonismus](https://x.com/kimmonismus/status/2069795647956373632) 则指出，据报道其从设计到流片仅用了 **9 个月**。对于高性能 ASIC 来说，这一周期异常短，据称得益于 OpenAI 自有模型的加速。
- **技术解读与生态影响**：社区的逆向分析显示，Jalapeño 似乎具有类似 TPU 的设计：[@scaling01](https://x.com/scaling01/status/2069867464716939413) 估计它采用接近光罩尺寸的芯片裸片，配备约 **216GB HBM3E**，带宽约为 **7.1–7.4 TB/s**，并可达到约 **10 PFLOPS FP4**。即便这些数字尚未得到官方确认，其释放的信号仍然是：对于顶尖 AI 实验室而言，采用类似超大规模云服务商的推理芯片，已经成为基本配置。同一天，编译器和运行时领域也发生了变化：[Chris Lattner 宣布](https://x.com/clattner_llvm/status/2069769232477192354)，**Qualcomm 将收购 Modular**；而 [Modular 表示](https://x.com/Modular/status/2069787078032834635)，**Mojo 开源计划仍在按原计划推进**。这一组合表明，除了 NVIDIA/CUDA 之外，围绕垂直整合推理技术栈的竞争正在变得更加激烈。
- **服务部署和吞吐量仍是活跃的竞争方向**：在基础设施方面，[NVIDIA](https://x.com/NVIDIAAI/status/2069813582825418828) 表示，借助 Expert Parallelism、DeepEP 和 TransformerEngine 内核，**NeMo AutoModel** 可通过 MoE 模型实现 **3.4–3.7 倍更高的训练吞吐量**。[SkyPilot](https://x.com/skypilot_org/status/2069815107891388477) 推出了 **Endpoints**，用于统一管理自有集群上的推理服务；[Modal](https://x.com/modal/status/2069818060991762809) 则声称，其开源推理方案在延迟方面优于专有服务商。在本地优化方面，[@jon_durbin](https://x.com/jon_durbin/status/2069876870628155397) 报告称，通过训练定制的 **DFLASH** 草稿模型/推测模型，实际解码速度提升了 **30–50%**。

**Agent 的用户体验正从“工具”转向“同事”，同时带来新的安全和成本问题**




- **Anthropic 基于 Slack 的 Agent 模型是最大的 UI 亮点**：多条推文都认为，Claude 嵌入 Slack 和团队工作流具有重要意义。[@karpathy](https://x.com/karpathy/status/2069822834160124091) 认为，人们低估了它的价值，因为它并不只是一个“功能”或 Slack 机器人，而是一个**组织级的工作框架（harness）**。[@gallabytes](https://x.com/gallabytes/status/2069808735212716225) 将这种体验上的跃迁描述为：Claude Code 是“结对协作伙伴”，而 Tags 则像是在“管理一个团队”。[@dabit3](https://x.com/dabit3/status/2069785904206508241) 进一步提出，最终用户甚至可能不需要再明确标记 Agent。
- **真正棘手的是身份、权限和锁定效应**：Anthropic 在[这条帖子](https://x.com/ClaudeDevs/status/2069895377080443271)中详细介绍了它的 **Agent 身份**模型：Claude 拥有自己的凭证，所有操作都会以该身份记录并接受审计，同时访问权限也可以集中撤销。这一设计既获得了赞赏，也引发了担忧。[@KentonVarda](https://x.com/KentonVarda/status/2069765917018382568) 认为，为每个 Agent 单独配置权限无法扩展，主张采用基于能力的安全机制（**capability-based security**），提供细粒度、限定任务范围的访问权限。[@random_walker](https://x.com/random_walker/status/2069760540709208306) 将 Claude Tag 描述为“一个什么都记得、并按思考次数收费的同事”，并警告说：一旦一个共享 Agent 深度嵌入组织工作流，就可能出现隐性知识锁定、提示注入风险以及预算不透明等问题。[@JubbaOnJeans](https://x.com/JubbaOnJeans/status/2069798018879238517) 同样指出，写入操作的归属可能变得模糊，而在 Slack 这类边界清晰的环境之外，未来的访问控制也会更加复杂。
- **开放式 / DIY 的回应很快就出现了**：Hugging Face 在[一条博客推文](https://x.com/victormustar/status/2069696147526947290)中介绍了其内部基于 Slack 的 coding Agent **Moon Bot**，强调支持自托管、自定义工具、可审计会话，并且不存在供应商锁定。随后，[@calebfahlgren](https://x.com/calebfahlgren/status/2069768499510013978) 列出了覆盖 GitHub、Athena、analytics、MongoDB、Elasticsearch 和 HF Buckets 的生产环境集成。更大的趋势是：团队越来越需要 Agent 原生的 UX，但很多团队更愿意自己掌控工作框架和记忆层，而不是把组织智能外包给供应商。

**Qwen-AgentWorld、OpenThoughts-Agent，以及作为下一条 Agent 扩展轴心的 Memory**

- **Qwen-AgentWorld 推动面向 Agent 的“语言世界模型”**：Alibaba Qwen 发布了 [Qwen-AgentWorld](https://x.com/Alibaba_Qwen/status/2069720365442719867)，将其定位为原生的**语言世界模型（language world model）**，能够在单个模型中模拟 **7 个环境**——MCP、Search、Terminal、SWE、Web、OS 和 Android。Qwen 表示，该项目有两条路径：一是构建模拟器本身，二是将世界建模用于 Agent 预训练。他们开源了 [Qwen-AgentWorld-35B-A3B 和 AgentWorldBench](https://x.com/Alibaba_Qwen/status/2069720412481888400)，其中模型采用 **35B MoE / 3B active** 架构，并支持 **256K context**。一个值得注意的结果是：单轮环境预测可以迁移到多轮 Agent 任务中，并且在领域内和跨领域 benchmark 上都带来提升，具体总结见[后续介绍](https://x.com/Alibaba_Qwen/status/2069720397747220493)。
- **OpenThoughts-Agent 提供了一套扎实的开放数据方案**：[@iScienceLuvr](https://x.com/iScienceLuvr/status/2069643721155793114) 和 [@RichardZ412](https://x.com/RichardZ412/status/2069827815403557287) 重点介绍了 **OpenThoughts-Agent**，这是一个面向 Agent 模型的开放式数据筛选与训练流程，包含 **100 多组受控消融实验**。团队构建了一个包含 **10 万条样本**的训练集，并对 **Qwen3-32B** 进行微调，在 7 个 Agent benchmark 上取得了 **44.8% 的平均准确率**。这些关键发现对实践者很有参考价值：指令的选择影响远超一般预期，benchmark 表现最强的 teacher 不一定是最好的 teacher，更长的执行轨迹会带来帮助，而在大规模训练中，数据源多样性优于反复使用少数来源。
- **Memory 正在成为系统中的一等层级**：大量高价值讨论都围绕 Agent 中尚未解决的 Memory 问题展开。 [Weaviate 的 Engram GA](https://x.com/victorialslocum/status/2069722431460168171) 将 Memory 视为一种异步基础设施：它会提取、去重、整合并确定记忆的作用范围，而不是把所有内容一股脑塞进 context。[@hwchase17](https://x.com/hwchase17/status/2069857129272627626) 展示了一个 LangSmith / Context Hub 工作流，用于执行“睡眠时间计算（sleep-time compute）”：系统在线下分析 traces，再将结果写回 Memory。[@dair_ai](https://x.com/dair_ai/status/2069846777977880769) 提到的一篇论文认为，应当把 Agent Memory 作为完整的**数据管理层**来评估——包括存储、检索、更新、整合和生命周期管理，而不是将其视为只根据最终任务成功率评判的黑盒。Agent 之间的差异化能力似乎正越来越集中在这一层。

**中国开放模型持续缩小差距：GLM-5.2、Kimi 的分发，以及计算规模**



- **GLM-5.2 继续主导开源模型讨论**：多条推文都将 **GLM-5.2** 视为当前最强的开放权重模型竞争者。[CoreWeave](https://x.com/CoreWeave/status/2069874833576321150) 表示，它在 Artificial Analysis 和 Agent Arena 的开放模型排名中均位居榜首；与此同时，[Baseten](https://x.com/baseten/status/2069832610289709156) 和 [Cursor availability](https://x.com/ZixuanLi_/status/2069921339817795869) 的动态也显示，其服务部署和分发正在迅速扩大。[​​@nutlope](https://x.com/nutlope/status/2069827178569638243) 将 GLM 5.2 与 Opus 4.8 在网页任务上的表现进行了对比，称二者**质量相近**，但 GLM 5.2 的**输出 token 数约为 2 倍**，同时**速度更快**，成本大约**低 3 倍**。[Arena](https://x.com/arena/status/2069885722333769963) 还表示，GLM-5.2 Max 在 Code Arena：Frontend 中领先于一众实力强劲的模型。
- **基准测试的细节很重要**：GLM-5.2 也出现在 ARC-AGI-2 测试中。[​​@fchollet](https://x.com/fchollet/status/2069858556552298519) 称，这是迄今为止开源模型在 ARC-AGI-2 上取得的**最强成绩**；与此同时，也有人围绕其 **22.8%** 的成绩究竟如何与西方前沿模型比较展开讨论。更广泛的结论并不在于某一个基准测试，而在于中国开源模型已经持续出现在编程、Agent 和知识工作等多个领域的竞争中。
- **商业化与基础设施正在加速**：[Moonshot 的 Kimi API](https://x.com/Kimi_Moonshot/status/2069718757338202140) 现已上线 **AWS Marketplace**，企业可以通过统一账单和 EDP 额度抵扣，更便捷地完成采购。与此同时，中国国内算力仍是一个重要议题：[@teortaxesTex](https://x.com/teortaxesTex/status/2069760099925524864) 提到有报道称，Huawei 可能会展示一套 **950 SuperPOD** 规模的系统，这意味着中国有望以相当大的规模生产国产 NPU 集群。如果属实，这将显著改善中国模型服务生态的经济性和韧性。

**政策、人才与前沿实验室战略正在重塑竞争格局**

- **Anthropic 仍处于政策争议的中心**：[@kimmonismus](https://x.com/kimmonismus/status/2069704003311567045) 报道称，有人对 Trump 政府时期的 AI 出口管制发起了首起重大法律挑战。Legion 认为，通过托管服务访问模型，并不等同于出口模型权重或技术数据。与此同时，备受讨论的 Mythos 事件也有了更多背景：[这里汇总了 Reuters/AP 的细节](https://x.com/kimmonismus/status/2069692592250360126)，报道称 Anthropic 的模型在一次受限测试中发现了美国敏感系统中的漏洞，不过也有评论者提醒，早期报道可能夸大了事件的影响。
- **蒸馏与访问控制正成为地缘政治问题**：[@kimmonismus](https://x.com/kimmonismus/status/2069879640835961277) 还报道称，Anthropic 指控与 Alibaba 有关联的运营方使用了**约 25,000 个欺诈账户**，进行了 **2,880 万次 Claude 交互**，以将前沿模型能力蒸馏到 Qwen 类系统中。如果情况属实，这会让“对抗性蒸馏”争议从传闻层面升级为更接近执法和国家战略的问题。
- **人才与新实验室**：当天还出现了人才流动和新机构成立的消息。[Arthur Conmy 加入 Anthropic](https://x.com/ArthurConmy/status/2069820098890674334) 一事在 alignment 领域尤其值得关注。[Mirendil AI 宣布成立](https://x.com/bneyshabur/status/2069860934148079800)，并获得 **2 亿美元种子轮融资**，其理念是利用能够自我加速的 AI 推动科学领域的 R&D。在英国，[BOLD Lab 和 SOFAIR](https://x.com/KanishkaNarayan/status/2069777169551671420) 获得了 **6,000 万英镑**种子资金，用于成立两个新的国家级基础 AI 实验室；[UCL DARK 并入 BOLD](https://x.com/_rockt/status/2069713868918587399) 也在同期发生。在商业领域，[Bloomberg 报道的 Google DeepMind 员工离职并转向 Anthropic](https://x.com/kimmonismus/status/2069870513283871203) 一事，则进一步说明创业公司的上升空间仍在持续吸引前沿领域人才。

**热门推文（按互动量排名）**

- **OpenAI Jalapeño**：[OpenAI 宣布首款定制推理芯片](https://x.com/OpenAI/status/2069770172802773292)——这是这组消息中影响最深远的产品与基础设施发布。
- **GPT-5.5 Instant 更新**：[OpenAI 推出更新版 GPT-5.5 Instant](https://x.com/OpenAI/status/2069843083701915755)，改进了意图理解、约束处理和对话风格。
- **Qwen-AgentWorld**：[Alibaba Qwen 发布并开源面向 Agent 的语言世界模型](https://x.com/Alibaba_Qwen/status/2069720365442719867)。
- **Anthropic 的 Agent 身份模型**：[Claude 在 Slack 中现已使用自己的凭证和审计轨迹](https://x.com/ClaudeDevs/status/2069895377080443271)，厘清了企业级 Agent 设计中最棘手的问题之一。
- **Cursor x Notion**：[现在可以直接从 Notion 委派 Cursor 任务](https://x.com/cursor_ai/status/2069872515548340407)，这再次表明，Agent 工作流正在进入现有的团队软件，而不是停留在独立的聊天应用中。


---




# AI Reddit Recap

## /r/LocalLlama + /r/localLLM 回顾

### 1. 中国 AI 芯片生态与管控

  - **[已有 7 家中国公司开始出货 H100/H200 级别的 AI 芯片，其中大多数在过去 6 个月内完成 IPO。我把它们全部整理出来了。](https://www.reddit.com/r/LocalLLaMA/comments/1udkxde/7_chinese_companies_are_already_shipping/)**（活跃度：1423）：**这篇帖子梳理了 7 家中国 AI 加速器厂商**——Huawei Ascend、Alibaba T-Head、Baidu Kunlunxin、MetaX、Moore Threads、Biren 和 Iluvatar CoreX——并称它们当前的产品大致达到 **H100 级别**，下一代产品则瞄准 **H200 级别**。这些说法主要依据 CHITEX/Dmitry Shilov 的演示文稿，以及作者链接的 [X 帖子](https://x.com/superalesha/status/2069415581237813437)。帖子列举的关键规格包括：**Huawei Ascend 910C/910D/950** 采用国产 HBM 的路线图；Alibaba 的 `16×96GB` PG1 服务器，总显存达到 `1.536TB`；MetaX C600 配备 `144GB HBM3e`；Moore Threads S5000 配备 `80GB` 显存和 `1 PFLOPS` 算力；以及 Biren/Iluvatar 的路线图，加入 FP8/FP4 和边缘推理模块。更大的判断是，中国 AI 基础设施正在摆脱对 NVIDIA/CUDA 的依赖，转向国产技术栈：采用类似 OAM 的模块、专有互连技术、由 SMIC 生产，并实现接近 100% 的利用率；与此同时，Qwen/DeepSeek/GLM 等中国开源权重模型也越来越多地优先针对非 NVIDIA 加速器进行调优。**不过，热门评论对实际获取和部署这些产品持怀疑态度：有人询问这些系统是否会在欧洲上市，甚至能否通过 AliExpress 购买；最实质性的担忧则是，无论原始硬件规格多么出色，真正的瓶颈都会落在*“软件栈”*上——包括 CUDA 兼容性、驱动、编译器/运行时的成熟度，以及与各类框架的集成。**

    - 一条技术细节丰富的评论认为，这篇帖子的说法夸大了实际部署能力：`1,536 GB` 的总显存，在考虑运行时开销、KV cache、激活值、内存碎片以及分布式执行要求后，并不足以运行一个约 `1,510 GB` 的 BF16 模型。评论者还质疑“H100/H200 级别”的说法，并指出，据报道 Huawei Ascend 950PR 配备 `128GB` 显存，带宽为 `1.6TB/s`，FP8 算力为 `1 PFLOPS`；相比之下，NVIDIA H200 的对应规格为 `144GB`、`4.8TB/s` 和 `2 PFLOPS dense FP8`。这意味着，尽管厂商声称产品属于同一级别，其显存带宽和计算能力实际上明显更低。
    - 有几项说法被指出更像是“即将出货”，而不是已经在出货。例如，评论者表示，关于 Kunlun M100，目前公开渠道找不到显存容量、带宽或 TFLOPS 等核心规格；而现有的 `vLLM` 支持似乎针对的是较老的 Kunlun 芯片，而不是 M100。
    - Moore Threads 及其 C 系列的相关说法也受到质疑：评论者称，目前出货的产品似乎仍是 C500/C550 级别，规格没有那么突出，可能配备 `64GB` GDDR6；而 C600 宣传中的 `144GB HBM3e` 和 H200 定位，仍属于未来大规模量产的说法。他们强调，要将 GDDR6 产品升级为可大规模生产的 HBM3e 产品，需要跨越一个尚未得到验证的重大制造和集成难关。

  - **[这个社区似乎可能错过了这件事：要求追踪 AI 芯片位置的法案获得业界支持 | 已有六家公司表示支持《芯片安全法案》，该法案将要求美国最先进的计算芯片配备位置追踪机制。](https://www.reddit.com/r/LocalLLaMA/comments/1ue2fd7/seems_this_community_might_have_missed_it_bill/)**（活跃度：440）：**这篇帖子援引了几天前的相关报道（[r/politics](https://www.reddit.com/r/politics/comments/1uahgcs/bill_that_would_mandate_location_tracking/) 和 [r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1ubz5xh/us_to_require_location_tracking_for_ai_and/) 也讨论过此事），称拟议中的《芯片安全法案》将要求美国最先进的 AI 加速器配备位置追踪机制。从技术角度看，这意味着要在受出口管制的计算设备中加入某种硬件或固件级别的地理定位、设备证明或上报能力，目标是防止高端 AI 芯片被转运到受限制的司法管辖区。**热门评论普遍持反对态度，认为这一要求可能削弱美国相对于中国的竞争力，并引入新的安全和隐私风险；有一位评论者讽刺称，这将是*“最好、最安全的位置追踪机制”*，而且*“不存在任何安全问题。”*






### 2. OCR 与 Agent 模拟领域的开放模型发布

  - **[Unlimited-OCR 现已登陆 ModelScope！一款 3.3B 多语言 OCR 模型，可一次性解析单张图像、多页文档和 PDF。许可证：MIT](https://www.reddit.com/r/LocalLLaMA/comments/1ue51uk/unlimitedocr_is_now_on_modelscope_a_33b/)**（热度：948）：****Baidu 的 **Unlimited-OCR** 已在 [ModelScope](https://x.com/ModelScope2022/status/2069335055965491525) 上发布。这是一款采用 **MIT 许可证**的 **`3.3B` 多语言 OCR/文档解析模型**，可一次性解析单张图像、多页文档和 PDF；针对较长的 OCR 序列，输出上限最高可达 **`32K` token**。[GitHub 仓库](https://github.com/baidu/Unlimited-OCR)介绍了基于 Transformers 的推理方式，以及支持 OpenAI 兼容流式输出的 **SGLang 部署**，并提供两种图像/布局模式：`base` 和 `gundam`。**技术讨论者主要询问它与 **PaddleOCR-VL-1.6** 的对比、`32K` 输出限制最多能容纳多少页，以及 `gundam`/“gundan”模式的含义；此外，也有人质疑它是否缺少对 **Paddle** 的支持。**

    - 评论者希望看到与 **PaddleOCR-VL-1.6** 的具体对比评测，尤其关注吞吐量与准确率之间的权衡，以及在多页/PDF 解析过程中，模型的 `32k` 上下文限制最多能容纳多少页文档。
    - 一些用户对发布信息中的术语不够明确表示疑问，尤其是 *“gundam mode”*。他们认为 ModelScope/Hugging Face 文档需要说明这一模式的定义，以及它会如何影响 OCR 行为或文档解析。Hugging Face 模型卡链接如下：https://huggingface.co/baidu/Unlimited-OCR

  - **[Qwen-AgentWorld-35B-A3B：一款每个 token 仅激活 3B 参数、经过训练可模拟 MCP、终端、SWE、Android、Web 和操作系统环境的模型](https://www.reddit.com/r/LocalLLaMA/comments/1ue5149/qwenagentworld35ba3b_a_3bactive_moe_trained_to/)**（热度：292）：****Qwen** 发布了 [`Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)。这是一款拥有 `35B` 参数、每个 token 大约激活 `3B` 参数的 MoE 模型，其定位是**语言世界模型**，而不是聊天/指令模型或自主 Agent。模型经过训练，能够根据 Agent 的动作，预测 **MCP/工具调用、搜索、终端、SWE、Android、Web 和操作系统 GUI** 等环境中的后续观测结果，从而在不实际调用工具的情况下，进行模拟 Agent 循环，用于离线评测、合成轨迹生成、工具使用工作流测试，以及类似沙盒的训练。**评论整体较少，但有一条技术性评论指出，它可以通过模拟 `ls -la` 等动作来服务于评测；也有人开玩笑或持怀疑态度地认为，这种训练可能类似于互换用户/助手角色，或者直接提示模型“*You are an MCP server now.*”**

    - 一位评论者提出了一个具体用途：训练模型预测环境响应。例如，给定用户输入的 `ls -la` 命令，生成相应的终端输出。他认为这对**评测框架或模拟环境**很有帮助，因为这样可以在不调用真实终端或外部工具的情况下模拟 Agent 的动作。
    - 另一条具有技术相关性的讨论将 Qwen-AgentWorld-35B-A3B 视为 LLM Agent 的一种潜在**世界模型式组件**，并在概念上将其与 Yann LeCun 的世界模型研究进行比较。评论者指出，如果将 MCP、SWE、Android、Web、操作系统和终端等环境中的模拟能力直接应用于 LLM 的推理和训练，并且基准测试中的能力能够泛化，或许可以提升 Agent 的能力。




## 技术性较低的 AI 子版块动态

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. Krea 2 开源图像模型与生成式 AI 的保真度

  - **[我们是 Krea 2 背后的团队，欢迎向我们提问！](https://www.reddit.com/r/StableDiffusion/comments/1udnm0a/we_are_the_team_behind_krea_2_ask_us_anything/)**（热度：1017）：****Krea** 宣布，**Krea 2** 这款由其内部训练的开源文生图模型现已发布，代码和权重可通过 [krea.ai/krea-2-open-source](http://krea.ai/krea-2-open-source)、[GitHub](http://github.com/krea-ai/krea-2) 获取；Hugging Face 上也提供了 [`Krea-2-Raw`](http://huggingface.co/krea/Krea-2-Raw) 和 [`Krea-2-Turbo`](http://huggingface.co/krea/Krea-2-Turbo) 的 checkpoint。Krea 研究负责人表示，这是他们首次完全由内部训练并开源的模型，同时也在考虑发布更多产物，例如**不使用 guidance/step distillation 的 Turbo checkpoint**、**5B 版本**，以及针对**图像参考、编辑、边界框、更好的文字渲染和真实感**等能力的改进。**评论者主要关注产品路线和架构问题：是否会发布**图像编辑**版本，以及 Krea 为什么选择 **Qwen VAE** 而不是 **Flux 2 VAE**。**



- 一位 Krea 2 研究人员表示，这是他们首次完全在内部训练并发布开源模型，并且同时发布了 **raw** 和 **turbo** checkpoint。他们正根据社区反馈考虑推出更多开放版本，包括**不使用 guidance / step distillation 的 Turbo checkpoint**、**5B checkpoint 变体**，以及围绕具体能力的改进，例如图像参考、图像编辑、边界框、更好的文字渲染和更强的真实感。
    - 多位评论者重点关注模型组件和训练选择的透明度，尤其是 Krea 2 为什么使用 **Qwen VAE** 而不是 **FLUX.2 VAE**。另一项技术相关的请求是希望 Krea 发布其**审美奖励模型（aesthetic reward model）**；评论者认为，目前的开源图像生成领域缺乏用于偏好优化和审美优化的强大奖励模型。
    - 功能请求主要集中在下游可控性上：用户询问 Krea 2 是否会推出**图像编辑**版本，以及未来是否会支持**风格迁移**。这些请求与研究人员提到的潜在后续能力扩展相吻合，包括图像参考和编辑工作流。

  - **[我让一张自己的照片变老并进行了修复](https://www.reddit.com/r/ChatGPT/comments/1ud6wuy/i_aged_and_restored_a_photo_of_myself/)**（活跃度：3288）：**这张图片（[链接](https://i.redd.it/rqbz1fkqhy8h1.png)）是一个针对 **ChatGPT 图像修复/上色**能力的受控四格测试：发帖者从一张已知的原始肖像开始，先人为地让照片变旧并制造损伤，然后要求 ChatGPT 对其进行修复。结果展示了生成式修复的一个关键局限：模型并没有恢复原本的面孔，而是**根据退化后的输入，臆造出看似合理的面部细节**，使人物看起来像是另一个年纪更大的人，胡须和脸部结构都发生了变化，还出现了被锐化的虚构细节。**评论者大多认为，这说明 AI “修复”并不是忠实的重建，而是基于受损输入进行生成。有人将这一点与人脸识别和安全系统中的风险联系起来，也有人开玩笑说修复后的样子像 Jack Black。

    - 一位评论者认为，这一结果展示了 AI 年龄变换/修复工作流的核心局限：输出可能变成*“一个完全不同的人”*，而不是保持原有身份。他明确指出，这种身份漂移可能导致基于 AI 的人脸识别和安全系统出现故障。
    - 一位用户将 **“Aged by Gemini”** 的结果裁剪回原始构图，然后用 **NanoBananaPro** 进行处理，声称它*“在修复方面仍然好得多”*，而且第一次尝试就得到了更好的结果。他指出，Gemini 生成的老化图像似乎缩小了画面，因此构图和裁剪会实质性地影响修复流程；同时，第二张图*“做了大量”*重建工作。

  - **[日本动画师使用 Seedance 将简单的 3D 模型渲染成动画](https://www.reddit.com/r/singularity/comments/1ue6yoh/japanese_animator_using_seedance_to_render_anime/)**（活跃度：2674）：**一篇 Reddit 帖子展示了一位日本动画师，据称使用 **Seedance**，根据简单的 **3D 模型**生成/渲染动画片段。这似乎是一种工作流：先用粗略的 3D 场景/构图提供空间和时间上的一致性，再交给 AI 生成视频。由于所提供的链接返回 **HTTP 403 Forbidden**，无法访问其中的 Reddit 视频，但评论者确认这位动画师是 [**Tetsurou**](https://x.com/craftcapitallab)，据称是动画行业资深人士，参与制作过 **TRIGUN STAMPEDE** 和 **TRIGUN STARGAZE**。**评论者认为，这可能是制作长篇 AI 视频、并保持世界模型一致性的一条可行路径；同时，他们也在讨论动画师使用 3D 控制和输入，是否足以让这类作品称得上艺术。一位评论者认为，其效果比典型的动画 CGI 更好，同时将反对 AI 艺术的观点斥为守门行为。



- 评论者认为，这套工作流很可能适用于**长篇视频的一致性控制**：先用简单的 3D 模型和布局，作为稳定的场景、姿态与世界观表示，再让 **Seedance** 渲染出最终的 anime 画面。有用户指出，这种方式还可以通过修改提示词来切换风格，例如从 anime 切换到**写实风格**（**photoreal**）或**复古漫画风格**（**retro comic**），同时保留底层预设的动作与构图。
- 从制作技术角度看，一个值得关注的方向是让 AI 负责 **inbetweening**——也就是生成关键帧之间的中间帧。一位评论者认为，这是动画制作中成本很高的一环，但相比 layout、表演或关键动画，它对观众感知到的创意质量贡献相对较小。这意味着，Seedance 这类工作流有望在保留人工创作指导的同时降低制作成本，而人工指导则可以通过 3D blocking 和提示词来实现。
- 该作品的创作者是 [**Tetsurou**](https://x.com/craftcapitallab)。据称，他拥有超过 `10 年`的 anime 行业经验，近期参与过 **TRIGUN STAMPEDE** 和 **TRIGUN STARGAZE** 的制作。这一背景在技术层面很重要，因为这段演示看起来不太像是原始的 text-to-video 生成，更像是一位经验丰富的动画师，利用 AI 在有意设计的 3D 场景调度之上完成渲染与合成。



### 2. AI 数据中心引发的反弹与防御

  - **[弗吉尼亚州邻居被数据中心噪音激怒：“你只想骂人”——居民在窗户上加装床垫和有机玻璃，以阻挡弗吉尼亚州这座数据中心发出的噪音。为其供电的天然气涡轮机发出高频啸叫，噪音 `24/7` 从不停歇。- NewsNation](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/)**（热度：2474）：**据报道，弗吉尼亚州一座数据中心持续产生 `24/7` 的高频噪音，声源是现场的**天然气涡轮机**，噪音严重到附近居民不得不在窗户上加装床垫和有机玻璃来隔音。评论者主要关注技术与选址问题：如果该设施没有接入电网，主要只需要光纤/网络接入以及自行发电，那么把由涡轮机供电的基础设施建在住宅区附近，更像是分区规划或审批失误，而不是数据中心本身的必然要求。热门评论质疑这样的项目为何能通过美国郊区的分区规划审批，并表示数据中心确实有必要增加，但不应建在住宅区；总体而言，大家认为这个选址难以辩护。**

    - 评论者认为，问题主要在于**发电设施的选址**，而非数据中心本身必然会产生噪音：据报道，该设施没有使用电网供电，而是依靠**现场天然气涡轮机**，持续发出高频啸叫，按理说不应靠近住宅区。一个重要的技术结论是，数据中心的选址其实具有较大灵活性，因为它们主要需要**电力、冷却和网络连接**。因此，评论者认为，没有多少工程上的理由要把由涡轮机供电的基础设施建在居民区内。
    - 多条评论质疑监管和规划环节为何失效：用户将这一弗吉尼亚案例与**欧盟/英国的规划制度**进行对比，指出在这些地区，燃气涡轮机等工业噪声源通常会面临更严格的许可要求、环境噪声评估，以及与住宅区保持距离等规定。讨论强调，更严格的分区规划或审批制度本可以要求项目接入电网、采取声学降噪措施，或迁移设施，而不是允许涡轮机紧邻住宅 `24/7` 运转。

  - **[John Carmack 谈数据中心问题](https://www.reddit.com/r/singularity/comments/1ue1sya/john_carmack_weighs_in_on_datacenters/)**（热度：2034）：**图片是一张 [X/Twitter 对话截图](https://i.redd.it/mius3v4nc59h1.png)，其中 **John Carmack** 认为，公众对数据中心的反对可能会变得类似美国社会曾经的反核情绪，从而拖慢 AI 基础设施的部署。他将数据中心需求视为 AI 驱动重大转型中“*真正的价值和进步*”的证据，而 **Markus “notch” Persson** 则用一个简单的“*为什么？*”提出质疑。评论者反驳了 Carmack 的说法，认为应采取折中方案：只要数据中心不会给当地造成扰民，就应允许建设，同时运营方还应自行提供电力和用水。另一些人指出，反核情绪在一定程度上曾受到化石燃料利益集团的影响，并猜测如今同样的利益集团可能会从 AI 数据中心不断增长的能源需求中获益。**

    - 多位评论者认为，数据中心扩张本质上是一个**基础设施选址和资源供给问题**：只要设施不会对居民造成干扰，就应允许其自由建设；同时还应要求运营商自行建设或保障**电力和用水**，而不是让当地电网或市政公用设施承担额外压力。噪音和废热也被明确指出是选址限制因素，大家反对把大型设施建在城镇附近，因为冷却排热和声学负荷会影响居民。
    - 一个反复出现的技术与政策观点是，大规模 AI 数据中心的增长应当与**新增可靠电源**同步，尤其是在进一步扩大规模之前建设核电。评论者认为，相比单纯扩大化石燃料供电能力，“安全的核电”更适合满足数据中心高负荷、持续运行的用电特征；同时，他们也担心，如果不建设新的清洁基荷电源，AI 负荷增长最终可能让石油和煤炭利益集团受益。




### 3. Gemini 和 Fable 的模型发布传闻

  - **[本周发布 3.5 pro](https://www.reddit.com/r/GeminiAI/comments/1uei7js/35_pro_coming_this_week/)**（活跃度：1211）：**这张图片是一条未经证实的推文截图，声称 **“Gemini 3.5 Pro”** 将于“本周”发布，并传闻会升级视觉/多模态推理能力、记忆和上下文保持能力、Agent 工作流、SVG/前端生成能力，加入原生图像模型和“Gemini Super App”，以及所谓的 `2.5M` token 上下文窗口（[图片](https://i.redd.it/kxh47zuxa99h1.png)）。从技术角度看，这篇帖子更像是推测，而不是正式公告：评论者指出其中没有提到任何编程基准测试成绩，并质疑它是否能胜过现有的 Gemini 3.x/2.5 Pro 预览版，或与 GPT/Claude/Fable 级别的编程模型竞争。**评论大多持怀疑态度，有人表示 Google 应该“先把它发布出来”，并避免出现性能倒退；也有人认为 `2.5M` 的上下文窗口听起来像假的，预计最终还是 `1M`。

    - 评论者质疑 **3.5 Pro** 是否真的会比 **3.1 Pro Preview** 更强。有人指出，如果这条泄漏消息可信，而且模型确实很强，公告很可能会重点强调**领先的编程基准成绩**；而没有提到这类成绩，可能意味着它未必能击败当前顶尖的编程模型。
    - 用户对所谓的 **`2.5M` 上下文窗口**持怀疑态度，认为 **`1M` tokens** 更可信，并表示这个夸张的上下文规模反而让这条泄漏消息看起来更像假的。
    - 一个与技术实际使用相关的担忧是高负载下的模型路由问题：用户开玩笑并抱怨说，即使是付费的 **Pro** 订阅者，在“高强度使用”期间也可能收到备用模型的回复。这样一来，即便用户有权使用宣传中的模型，实际体验和质量仍可能不稳定。

  - **[Fable 5 回归传闻：CC 中出现了一些线索](https://www.reddit.com/r/ClaudeAI/comments/1uehr3a/fable_5_return_rumored_with_some_hints_in_cc/)**（活跃度：845）：**一则基于 **Claude Code `v2.1.190` 字符串变更**的传闻称，Anthropic 可能正在准备将 **Fable 5** 永久纳入订阅，并设置每周配额：新增的字符串 *“You've used your Fable 5 usage for this week”*，以及删除的 *“purchased separately from your plan”* 被视为相关证据（[来源](https://x.com/synthwavedd/status/2069813760622043483)）。如果属实，这意味着 Fable 5 的使用方式可能会从限时提供或需要单独购买，转变为订阅中包含的周期性限额使用。**评论大多是兴奋和猜测；唯一比较实质性的偏好是，与其提供短暂的临时订阅窗口，用户更希望设置较低的每周配额，因为这样可以持续使用。

    - 一位评论者提出了一个具体的产品访问方面的担忧：相比只提供 `two-week` 限时访问的订阅模式，他们更希望 Fable 采用**较低的每周使用上限**，因为相比限时开放，持续但受限的访问更实用。




# AI Discord 社区

很遗憾，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但很快会推出全新的 AINews。感谢你一直读到这里，这段旅程曾经很美好。