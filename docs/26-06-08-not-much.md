---
companies:
- cognition
- frontiercode
- moonshot
- google
- claudedevs
- magicpath
- langsmith
- modal
date: '2026-06-08T05:44:39.731046Z'
description: '由 **Cognition** 推出的 **FrontierCode** 基准测试凸显了编程任务的挑战性：表现最出色的模型 **Opus
  4.8** 在最难的任务子集上仅获得了约 **13%** 的分数，这表明编程问题的解决程度远不及现有基准测试所暗示的那样理想。


  目前，将“**循环**（loops）”作为编程智能体控制隐喻的趋势十分显著，强调明确的目标、验证和迭代，但也有专家提醒要警惕对循环的过度依赖。随着 **ClaudeDevs**、**MagicPath**、**LangSmith**
  和 **Modal** 推出的可观测性仪表盘、沙箱环境及工作流工具，智能体的使用体验（ergonomics）正在不断优化。


  **月之暗面（Moonshot）** 旗下的 **Kimi** 发布了重大更新，包括更强大的编程智能体，以及一款支持多达 **300 个本地子智能体** 的桌面端智能体产品。此外，**谷歌（Google）**
  通过升级 **Gemma 4** 检查点，进一步推动了高效的本地化部署。'
id: MjAyNS0x
models:
- opus-4.8
- gemma-4
people:
- swyx
- dzhng
- claudecode
- bcherny
- reach_vb
- omarsar0
- gneubig
- hamelhusain
- angaisb_
title: 今天没发生什么事。
topics:
- coding-evaluation
- agent-control
- verification
- agent-ergonomics
- sandbox-environments
- local-inference
- workflow-optimization
- cli-tools
- plugin-integration
- persistent-memory
---

**平静的一天。**

> 2026年6月5日至6月8日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现更多的 Discord 消息。[AINews 网站](https://news.smol.ai/) 允许你搜索所有历史发布。提醒一下，[AINews 现在是 Latent Space 的一部分](https://www.latent.space/p/2026)。你可以 [选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 综述


**编程 Agent、循环（Loops）以及从“通过测试”到“可合并软件”的转变**

- **FrontierCode 提高了编程评估的标准**：Cognition 推出了 **FrontierCode**，这是一个明确针对代码是否真正**可合并（mergeable）**而非仅仅通过单元测试的新基准测试。任务是与开源维护者共同构建的，每个任务耗时 **40+ 小时**，并从回归安全性、代码整洁度、影响范围、测试正确性和可维护性等维度进行评估。核心结果显示，表现最好的模型 **Opus 4.8** 在最难的子集上仅获得约 **13%** 的分数——远低于 SWE-Bench 类评估中常见的 50% 以上的水平，这表明编程问题的解决程度远低于流行基准测试所暗示的那样 ([Cognition 官方公告](https://x.com/cognition/status/2064061031912288715), [Scott Wu 的总结](https://x.com/ScottWu46/status/2064073699368800475), [swyx 的分析](https://x.com/swyx/status/2064081945567580323), [theo 关于方差/可复现性的提问](https://x.com/theo/status/2064126021088215385), [Cognition 的回复](https://x.com/cognition/status/2064215347503452649))。
- **“循环（Loops）”正在成为主流的 Agent 控制隐喻——但仍需注意限制**：当天最响亮的实践主题是，应该给编程 Agent 提供**明确的目标、验证标准和迭代结构**，而不是单次提示（one-shot prompts）。流行的例子包括 [dzhng 的“不要使用循环，设计状态机”](https://x.com/dzhng/status/2063931263312892406)，[Claude Code 关于自动模式、例程（routines）和验证的回顾](https://x.com/ClaudeDevs/status/2064032814392352816)，[bcherny 的推文串](https://x.com/bcherny/status/2064034799711588805)，[OpenAI Codex 关于结果优先提示的技巧](https://x.com/reach_vb/status/2064028260070215772) 以及 [“帮我批准（Approve-for-me）”默认设置](https://x.com/reach_vb/status/2064044955421769755)，还有 [LangChain OSS 的“评分标准（rubrics）”](https://x.com/sydneyrunkle/status/2064034061165682931)。但一些从业者对盲目的循环炒作提出了反对意见：[Omar Sar0](https://x.com/omarsar0/status/2064024230396604469) 和 [Greg Neubig](https://x.com/gneubig/status/2064011013637234728) 强调，在不易验证的领域之外，人工检查点（human checkpoints）仍然至关重要，而 [Hamel Husain](https://x.com/HamelHusain/status/2064019243990188259) 则开玩笑说要屏蔽掉这个词。
- **Agent 的易用性（Ergonomics）在验证和编排方面有所提升**：整个技术栈的产品变化反映了这一转变。[ClaudeDevs 为 MCP 连接器开发者增加了可观测性仪表盘](https://x.com/ClaudeDevs/status/2064072801062121906)，包括采用率、延迟和错误视图。[MagicPath 推出了 Builder 计划](https://x.com/skirano/status/2064035120483352776)，用于外部 Agent 工作流和多人协作 Canvas 编辑。[LangSmith Sandboxes](https://x.com/LangChain/status/2064030008738296065) 和 [Modal 的沙箱扩展案例](https://x.com/AmplifyPartners/status/2063998736703856737) 都指向了同一个基础设施趋势：Agent 需要隔离、可检查且可长时间运行的环境。
- **实际使用模式正在趋于稳定**：最强力的操作建议集中在可衡量的结果、有限的自主权和 Thread 卫生。 [Angaisb_ 警告说，过长的 Codex Thread 会导致性能下降](https://x.com/Angaisb_/status/2064103464142065852)，而 [reach_vb 则报告了单 Thread 上下文累积的成功经验](https://x.com/reach_vb/status/2064115851503059418)。这种差异本身就是一个有用的信号：当前的 Agent 性能仍然在很大程度上受 **Harness 行为和工作流选择**的影响，而不仅仅取决于基础模型（Base-model）的质量。

**模型发布、本地推理及服务栈升级**

- **Kimi 发布了更强大的编程 Agent 和桌面端 Agent 产品**：月之暗面（Moonshot）发布了其开源编程 Agent **Kimi Code** 的重大更新，新增了**一行命令 CLI 安装**、拖拽**视频作为编程上下文**、ACP 支持、插件以及 IDE 集成（[公告](https://x.com/KimiDevs/status/2063981516708024369)）。同时，它还推出了桌面端 Agent 产品 **Kimi Work**，支持多达 **300 个本地子 Agent**、通过扩展实现的 browser-use、针对财务工具的接入以及持久化记忆（[产品发布](https://x.com/Kimi_Moonshot/status/2063990409903112344)，[桌面端可用性](https://x.com/crystalsssup/status/2063992904209842215)）。
- **Google 大力推动高效本地部署**：Gemma 获得了多项显著升级。据报道，新的 **QAT Gemma 4** 权重在保持性能的同时减少了 **~4 倍内存占用**，其中 **Gemma 4 E2B** 使用移动端量化格式可缩减至约 **1GB**（[@_philschmid](https://x.com/_philschmid/status/2063990553826439378)）。此外，**Gemma 4 MTP** 已合并至 **llama.cpp**，配合 QAT 权重可实现更快的解码速度（[Gemma 团队](https://x.com/googlegemma/status/2064030477628182814)）。同时，[llama.cpp 也增加了视频输入支持](https://x.com/osanseviero/status/2063985470489448887)，扩展了本地多模态用例。
- **开源/开放权重模型竞争依然激烈**：[Artificial Analysis 报告称 MiniMax-M3 在其智能指数（Intelligence Index）中得分 55](https://x.com/ArtificialAnlys/status/2064066303863005254)，一旦权重发布，它将成为领先的开放权重模型。M3 增加了**原生多模态能力**和 **100 万 token 上下文窗口**，在 GPQA/MMMU-Pro 上表现强劲，但在对幻觉敏感的评估中存在明显的拒答现象。与此同时，[norpadon 宣布了针对 Apple 硬件优化的量化版 Qwen3.5 权重](https://x.com/norpadon/status/2064040631479976240)。
- **服务基础设施正从文本 LLM 扩展到世界模型和全能（Omni）模型**：**vLLM-Omni 0.22.0** 增加了对 **NVIDIA Cosmos 3 世界模型**的同步支持、机器人服务 API、诸如 **Qwen3-TTS** 和 **VoxCPM2** 的 TTS 模型、更快的图像/视频推理，以及更广泛的量化/硬件覆盖（[发布说明](https://x.com/vllm_project/status/2064013506882703421)）。这反映了从仅限文本的推理栈转向通用多模态服务的更广泛趋势。

**基准测试、评估方法论与真实世界 Agent 衡量**

- **Agent 评估正从合成任务转向真实环境下的遥测数据**：Arena 推出了 **Agent Arena**，这是一个基于超过 **100 万个真实世界会话**的排行榜。它使用**因果追踪（causal tracing）**而非单纯投票来估算编排器/框架在五个维度上的处理效果：**确认成功、好评与差评、可控性、Bash 恢复以及工具幻觉**（[综述](https://x.com/arena/status/2064021507681276234)，[方法论推文](https://x.com/ml_angelopoulos/status/2064028763697127844)）。虽然该方法论是否经得起推敲尚待观察，但它是目前利用实际使用轨迹对已部署 Agent 进行基准测试的最明确尝试之一。
- **专业基准测试持续向新的输出领域扩散**：Hugging Face 和 Mecado 发布了 **CADGenBench**，这是一个用于根据图纸或 STEP 修改生成和编辑**工程级 3D CAD 零件**的基准测试，其指标涵盖了几何、拓扑、接口兼容性和 CAD 有效性（[发布推文](https://x.com/MikushRab/status/2063999885796614522)，[Thom Wolf 总结](https://x.com/Thom_Wolf/status/2064029993638764672)）。这是一个有意义的转变：评估正在超越文本/代码，进入到正确性由物理和几何定义的结构化产物领域。
- **一个反复出现的论点：好的基准测试会演变为训练流水线**：[Ofir Press 认为](https://x.com/OfirPress/status/2063990430350340575)，最好的基准测试是可扩展的，并植根于**真实世界的爬取数据源**，这使得它们不仅对衡量有用，对数据生成也同样有用。这一观点在 FrontierCode 和 Agent Arena 中都有隐式体现：基准测试不再是静态的计分板，它们正在成为**产品和 RL（强化学习）改进的反馈循环**。

**Google, Apple 以及消费者 AI 平台之争**

- **Google 扩展了 AI 产品组合、搜索和开发者界面**：Google 发布了功能更强大的 **NotebookLM**，具备 agentic chat（智能体对话）、更强的推理能力，并为 Ultra 订阅用户提供了更多输出格式（[发布详情](https://x.com/NotebookLM/status/2064016460964585549)）。同时将 **Google AI Plus** 的价格从 **$7.99 降至 $4.99/月**，并将存储空间翻倍至 **400GB**（[价格更新](https://x.com/NewsFromGoogle/status/2064066310393209100)）。在平台方面，[Google 强调了搜索功能的重大升级](https://x.com/Google/status/2064034586762354893)，包括多模态搜索以及将 **Gemini 3.5 Flash** 作为 AI Mode 的新默认模型。
- **Apple WWDC 的 AI 叙事聚焦于集成而非前沿领导力**：围绕 WWDC 的评论集中在重建的 **Siri AI**，它具有屏幕感知、App Action、个人上下文和更好的语音交互，同时也引发了对 **EU availability**（欧盟可用性）和硬件限制的担忧（[kimmonismus 实时推文](https://x.com/kimmonismus/status/2064059964709388774)，[区域限制说明](https://x.com/kimmonismus/status/2064047278105464868)）。[awnihannun](https://x.com/awnihannun/status/2064202168618422396) 提供了一个技术细节：据报道，Apple 的端侧模型是一个 **20B 参数的查询路由架构（query-routed architecture）**，每次查询会将专家模型从 NAND 加载到 RAM 中，这是一种针对设备限制优化的非标准设计。

**研究方向：Continual Learning、Agent 训练与优化器之争**

- **Anthropic 将科学领域 AI 的核心阻碍归结为基础设施不匹配**：其新的科学博客指出，AI 在编程领域的进步快于生物学，是因为生物数据库和工具并非为 Agent 使用而设计；瓶颈不在于原始智能，而在于**兼容 Agent 的科学基础设施**（[Anthropic 博客推文](https://x.com/AnthropicAI/status/2064054837294354677)）。这与广泛呼吁的 harness/environment 标准化相契合。
- **开源 RL 和环境协议正在成为协作重点**：[OpenEnv 已移交给一个联盟，包括 Hugging Face、Meta-PyTorch、Reflection、Unsloth、Modal、Prime Intellect、NVIDIA 等](https://x.com/ben_burtenshaw/status/2063991191415267492)。其核心理念是：前沿实验室通过紧密耦合的 harness 共同训练模型，而开源生态则需要在模型、harness、环境和 trainer 之间建立一个**共享协议层**。
- **Agent 的 Continual Learning 重新成为一个实际的系统问题**：[Hivemind 宣布了一个系统，可以将来自 Claude Code、Codex、Cursor 和 Hermes 等 Agent 的 trace 转化为可复用的技能](https://x.com/kimmonismus/status/2064001045391462907)，并声称在各种设置下都有显著收益。与此相关，[Nando de Freitas 发布了一长串推文](https://x.com/NandoDF/status/2063938859583389837)，概述了一个围绕从**交互后果（interaction consequences）**而非仅仅从 Token 序列中学习的研究计划。
- **优化器讨论异常活跃**：多个推特线程争论 **Muon** 是否与 **Shampoo** 有实质性区别，[Arohan 暗示存在优于 Shampoo 的优化器](https://x.com/_arohan_/status/2064036303021494418)，而 [Keller Jordan 公开了 Shampoo 和 Spectral Descent 的基准测试](https://x.com/kellerjordan0/status/2064062891607888058)。这场风波背后的实质点在于：人们重新开始渴望通过**优化器层面的改进**作为真正的前沿杠杆，而不仅仅是基准测试的噪声。

**热门推文（按互动量排序）**

- **Signal 关于英国设备扫描的立场**：互动量最高的技术相关帖子是 [Signal 反对英国要求进行端侧扫描和与年龄验证挂钩的内容审查的声明](https://x.com/signalapp/status/2064069692168519931)。这更多属于隐私/安全政策而非 AI，但与客户端推理和平台信任直接相关。
- **OpenAI 的公司方向与流动性**：[Sam Altman 分享了 OpenAI 的当前计划](https://x.com/sama/status/2064088940932641225)，随后不久 [OpenAI 宣布已秘密提交 S-1 文件](https://x.com/OpenAINewsroom/status/2064094175541461220)。对于 AI 工程师来说，关键的战略意义在于：OpenAI 和 Anthropic 现在似乎都在保留 IPO 选择权，同时在提升算力和产品广度。
- **NotebookLM 和 FrontierCode 是当天最重磅的纯产品/评测发布**：[NotebookLM 的升级](https://x.com/NotebookLM/status/2064016460964585549)、[Kimi Code](https://x.com/KimiDevs/status/2063981516708024369)、[Kimi Work](https://x.com/Kimi_Moonshot/status/2063990409903112344) 以及 [FrontierCode](https://x.com/cognition/status/2064061031912288715) 占据了技术讨论的主导地位，尤其是 FrontierCode 正在重塑关于“优秀编程表现”定义的讨论。


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Commodity-Hardware LLM Inference Updates

- **[llama.cpp Gemma4 MTP support merged!](https://www.reddit.com/r/LocalLLaMA/comments/1tzbcyp/llamacpp_gemma4_mtp_support_merged/)** (热度: 1097): **llama.cpp** 合并了 [PR #23398](https://github.com/ggml-org/llama.cpp/pull/23398)，通过 `--spec-type draft-mtp` 和一个 draft/assistant GGUF 模型增加了 **Gemma 4 多标记预测 (MTP)** 支持，为支持的 Gemma 4 变体开启了投机采样风格的解码。一位评论者报告称，在 **RTX 4070 Super** 上使用 **12GB VRAM** 运行 **Gemma 4 12B**，配合 [Unsloth QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)、[MTP assistant/drafter Q8_0 GGUF](https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF) 以及 `--spec-draft-n-max 4`，速度达到了 **`140 tok/s`**。PR 的 `mtp-bench` 结果显示，与非 MTP 相比，其 **吞吐量提升了约 2 倍以上**，而据报道 MoE 变体在作者的系统上并未加速。该实现据称在 31B 和 26B-4B 模型上还原了 Gemma 团队约 **~87%** 的 AIME-26 性能；E4B/E2B 变体目前尚不支持，且多 GPU 可能需要 `--spec-draft-device` 配合 `-sm layer`。评论者对 **QAT + MTP** 的结合充满热情，并特别感谢了贡献者 [u/am17an](https://www.reddit.com/user/am17an/) 在 llama.cpp 集成方面的工作。

    - 一位用户报告称，利用新合并的 llama.cpp MTP 支持、**Unsloth QAT GGUF** 权重和 MTP drafter 模型，在拥有 **12GB VRAM 的 RTX 4070 Super** 上运行 **Gemma 4 12B** 可达 `140 tok/s`。其命令使用了 `--model-draft`、`--spec-type draft-mtp`、`--spec-draft-n-max 4` 以及超大的 `--ctx-size 131072`，模型链接指向 [Unsloth QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) 和 [MTP assistant/drafter Q8_0 GGUF](https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF)。
    - 在 **NVIDIA GB10 Grace Blackwell / Asus Ascent GX10** 上的一项基准测试对 `Gemma-4-31B-it-Q8_0.gguf` 配合 `gemma-4-31B-it-MTP-Q8_0.gguf` 进行了测试，将 Q8 描述为“基本等同于全精度”。在没有 MTP 的情况下，吞吐量稳定在 `6.2–6.4 tok/s` 左右；使用 `--spec-type draft-mtp --spec-draft-n-max 7` 后，吞吐量提升至 `15.7–31.2 tok/s`（取决于任务），在通过 `--reasoning on` 保留推理模式的同时，实现了约 **3–5 倍的加速**。
    - 详细的 MTP 基准测试显示了任务相关的接受率行为：翻译任务达到 `31.2 tok/s`，草稿接受率为 `0.699`；摘要任务达到 `29.4 tok/s`，接受率为 `0.645`；而创意写作速度低得多，仅为 `15.7 tok/s`，接受率仅为 `0.277`。这表明 Gemma 4 MTP 加速对工作负载高度敏感，确定性或受限的任务比开放式的创意生成更能从投机多标记预测中获益。

  - **[You don't need a GPU to run gemma-4-26B-A4B](https://www.reddit.com/r/LocalLLaMA/comments/1tz5ffp/you_dont_need_a_gpu_to_run_gemma426ba4b/)** (热度: 902): **原帖作者报告在 **Intel i5-8500 + `32GB` RAM**、Linux 环境下，通过 [KoboldCpp](https://github.com/LostRuins/koboldcpp) 纯 CPU 运行 **Gemma `26B-A4B`**，实现了约 **`7 tok/s`** 的速度且无需 GPU；此前的 `~12B` 稠密模型虽然可用但速度较慢。评论者指出，关键的技术原因是该模型尽管总参数量为 `26B`，但 **激活参数量仅约 `4B`**，因此只要量化后的权重能装入系统 RAM，CPU 推理就是切实可行的。** 评论普遍认为，强大的本地推理并不一定需要云端接入或高端 GPU 设备，不过一位评论者认为，即使是带有 `8GB` VRAM 的廉价二手 GPU 也会带来巨大的提速。

    - 评论者指出，**Gemma 26B-A4B** 在 CPU/消费级硬件上相对可行，是因为尽管其总参数量较大，但每个 token 仅有约 **`4B` 激活参数**；主要的限制在于将模型权重装入系统 RAM，而非需要高端 GPU 算力。
    - 提出的技术忠告是，即使是带有 **`8GB` VRAM** 的小型二手 GPU 也能显著改善可用性，一位评论者估计，假设模型或激活工作集能从 GPU 加速中受益，其性能将比纯 CPU 执行提升约 **`5 倍`**。

- **[小米刚刚声称在标准 8-GPU 服务器上实现了 1T 模型的 1,000+ tps 吞吐量](https://www.reddit.com/r/LocalLLaMA/comments/1u0buhm/xiaomi_just_claimed_1000_tps_on_a_1t_model_using/)** (热度: 818): **Xiaomi MiMo** 声称 [`MiMo-V2.5-Pro-UltraSpeed`](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps) 在单个“标准” **8-GPU 通用节点**上，针对 **`1T` 参数的 MoE 模型**，达到了 **`1000+` tokens/s 的解码吞吐量**——据报道最高可达约 `1200 tps`。这是通过 TileRT 持久化/融合/流水线内核以及 **DFlash 投机解码（speculative decoding）**实现的，其接受长度（acceptance lengths）约为 `4.3–6.3` 个 tokens。模型侧的关键优化是选择性 **MXFP4 QAT**：小米表示，简单地应用 FP4 会损害推理和代码能力，因此他们只对 **MoE 专家层**（占据了大部分参数且是对量化最宽容的模块）进行量化，而让其他模块保持原始精度，从而在减少带宽压力的同时尽可能降低质量损失。该访问权限被定位为 **2026 年 6 月 9 日至 23 日**的限量企业/API 试用，促销价格为 **MiMo-V2.5-Pro 的 3 倍**。评论者集中讨论了“标准 8-GPU 节点”是否定义不明——询问使用了哪些 GPU——并认为这一结果证明了尽管此前存在质疑，压缩稀疏/MoE 架构正变得越来越经济。一位评论者认为，真正的“Token 寒冬”不在于模型能力，而在于消费级硬件的稀缺和定价，与此同时，数据中心却垄断了 GPU 进行低效的推理。

    - 评论者强调，小米报告的 `1,000+ TPS` 很大程度上取决于未指明的“标准 8-GPU 服务器”配置，并质疑 GPU 是数据中心级显卡还是诸如 `RTX 5090/3090` 之类的消费级 GPU，这使得在没有硬件细节的情况下很难评估其吞吐量声明。
    - 一个关键的技术点是小米针对 **MiMo-V2.5-Pro** 的选择性 FP4 量化策略：他们没有对整个模型应用 FP4，而是仅量化了 **MoE 专家层**，这些层包含大部分参数且对量化更具容忍度，同时保持非专家模块的原始精度。引用的观点是，**FP4 QAT** 在保留推理/代码能力的同时，减小了模型体积并提高了显存带宽利用率。
    - 已发布的权重已链接至 Hugging Face：[XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash)。一位评论者还质疑了架构标注，询问该模型是否实际上是 `1T-A1B`，暗示这是一个总参数量极大但每个 token 激活参数量却小得多的 MoE 模型。

  - **[Gemma 4 聊天模板现在支持保留思维（preserve thinking）](https://www.reddit.com/r/LocalLLaMA/comments/1u084qi/gemma_4_chat_template_now_has_preserve_thinking/)** (热度: 447): 根据 [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it/discussions/118#6a26a7088a64389a0709d3d2) 相关的 Hugging Face 讨论，**谷歌的 Gemma 团队已更新官方 `google/gemma-4-31B-it` 聊天模板以支持 `preserve_thinking`**。该帖子还记录了这款多模态 31B 指令模型的实际推理/部署路径，包括 `transformers` 的 `pipeline` / `AutoProcessor` + `AutoModelForImageTextToText`，以及通过 **vLLM** 和 **SGLang** 实现的 OpenAI 兼容服务。评论者认为官方支持 `preserve_thinking` 是对早期社区“第三方”聊天模板修改的认可，其中一位指出他们“知道它效果非常好”。几位用户希望推出更大规模的 **Gemma 4 `124B` MoE** 变体，以更好地利用更新后的模板，特别是针对 Agentic 编程工作负载。

    - 用户指出，官方 **Gemma 4 聊天模板**似乎正在添加 `preserve_thinking`，这是一种部分用户此前已通过第三方/自定义模板启用并发现有效的行为。技术观点认为，在多轮对话中保留隐藏的/结构化的推理对于 **Agentic 编程工作负载**特别有用，因为在这种场景下，工具调用和多步上下文的连续性至关重要。
    - 一位评论者提醒说，这一更改可能尚未正式上线：他们报告这目前仍是一个 **未合并的公开 PR**，且模型文件已有大约 `21 天` 没有更新。这表明用户在认为官方 Gemma 4 资源已具备 `preserve_thinking` 之前，应先核实模板版本。

## 非技术类 AI 版块总结

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 安全、隐私与 Token 限制

- **[一场针对 Claude Code 的活跃攻击正在植入后门。如果你使用 npm，你的凭据可能已经泄露。](https://www.reddit.com/r/ClaudeAI/comments/1u05t5e/an_active_attack_is_planting_backdoors_inside/)** (热度: 1039): **该帖子声称存在一场针对 `@redhat-cloud-services` 软件包（涉及 `32` 个包，约每周 `117k` 次下载）以及随后出现的 “Phantom Gyp” 浪潮（涉及 `57` 个包，约每月 `647k` 次下载）的活跃 npm 供应链攻击。在这场攻击中，恶意的安装/构建钩子（hooks）会窃取凭据，并通过 `~/.claude/settings.json` 中的 **Claude Code** `SessionStart` 钩子以及 `.vscode/tasks.json` 中的 `folderOpen` 任务实现持久化；引用的来源包括 **Microsoft** 的 Miasma 报告、**StepSecurity** 关于 [`binding.gyp` 滥用](https://www.stepsecurity.io/blog/binding-gyp-npm-supply-chain-attack-spreads-like-worm)的文章，以及 **Snyk** 的清理指南。建议的事件响应顺序为：检查依赖树/lockfiles 以发现受影响的包/版本；检查编辑器的持久化配置；在重置密钥前先断开连接并进行清理；然后从受信任的机器上重置 npm/GitHub/SSH/cloud/Kubernetes/Vault 的密钥；审计 npm 发布历史、GitHub 安全日志、自托管 runners、OIDC 信任关系；并临时使用 `npm install --ignore-scripts`，配合 lockfile 完整性哈希和最小权限的 CI/CD 令牌。** 热门评论大多集中在操作层面：一位评论者感谢了作者，而另一位则询问这是否与之前的事件相同，还是属于一场全新的攻击活动。

    - 一份详细的补救清单确认了可能受影响的 npm 软件包：`@redhat-cloud-services`、`@vapi-ai/server-sdk` 和 `ai-sdk-ollama`；建议使用 `npm ls` 进行检查，并审查在 `6 月 1 日` 和 `6 月 3–4 日` 左右发布的 lockfile 版本。该指南强调 **“先抑制再重置令牌”**：检查 `~/.claude/settings.json` 是否存在异常的 `SessionStart` 钩子，以及 `.vscode/tasks.json` 中是否有可疑的 `folderOpen` 任务，然后从受信任的机器断开连接并清理后，再重置凭据。
    - 评论描述了横跨 GitHub/npm 供应链层面的疑似蠕虫行为：检查 [GitHub 安全日志](https://github.com/settings/security-log)中是否存在异常的仓库、GitHub Actions 工作流、自托管 runners，以及诸如 “Miasma” 或 “Shai-Hulud” 的引用。它特别指出 **GitHub Actions OIDC 信任关系** 是高价值的重置目标，并提到这据称是 Red Hat 遭受侵害的漏洞所在，同时建议审查 npm 发布历史以发现未经授权重新发布的包版本。
    - 讨论的缓解措施包括：使用 **完整性哈希（integrity hashes）** 锁定依赖项，以便在执行前拦截内容被篡改后重新发布的包；临时使用 `npm install --ignore-scripts` 来阻止恶意的安装钩子、`binding.gyp` 和 `node-gyp` 的构建时执行。另一位评论者质疑为何可以直接向 Red Hat 仓库推送代码，认为受保护的 `main`/`master` 分支应当要求基于 PR 的合并及多名审批者。

  - **[Anthropic 今天更改了隐私政策，其中有一项每个 Claude 用户都需要了解的特定条款](https://www.reddit.com/r/ClaudeAI/comments/1u0kq84/anthropic_changed_their_privacy_policy_today_and/)** (热度: 784): **发帖者（OP）声称 **Anthropic** 在 `2026-06-08` 发布了修订后的[隐私政策](https://www.anthropic.com/legal/privacy)，并将于 `2026-07-08` 生效；该政策将向执法部门披露信息的条件从“外部强制的法律程序”更改为基于 Anthropic 内部的 *“诚实信用原则（good faith belief）”* 认为有必要。帖子认为，这为自动安全分类器的误报带来了风险——特别是角色扮演、虚构作品、叙事语境中的威胁或心理健康宣泄——因为据称对话可能会在没有法院命令、用户通知、申诉途径或明确证据阈值的情况下被移交给当局。OP 还将其与 OpenAI/Mistral 的政策进行了对比，认为其表现更差，并提出了对英国 GDPR/DBS 的担忧，但帖中未提供直接的政策变更链接；一位热门评论者明确要求提供来源 URL。** 热门评论态度非常消极，将这一变化定性为重大的隐私倒退，是更广泛的“平台劣化（enshittification）”的一部分；一位评论者表示，由于感知到的高昂成本、限制性行为和削弱的隐私保护，他们将迁回使用 Codex。另一位评论者要求提供链接以验证所声称的新政策。

- 一位评论者将 Anthropic 的政策变化与更广泛的 AI 提供商注意义务（duty-of-care）问题联系起来，并引用了针对 **OpenAI/Sam Altman** 的一起诉讼。在该诉讼中，受害者家属指控一名大规模枪击案枪手对 ChatGPT 的使用 *在内部已被标记但未向警方报告* ([BIV report](https://www.biv.com/news/tumbler-ridge-families-likely-to-seek-us1-billion-in-lawsuit-against-openai-lawyer-12209582))。这意味着，当内部安全系统识别到严重风险时，提供商可能会越来越多地保留监控或升级（escalate）用户活动的权利。
- 另一位评论者认为，对于高严重性的滥用行为，Anthropic 的升级机制可能是合理的，并专门提到了 Anthropic 自身的 **biorisk red-team work** ([Anthropic Red Teaming: Biorisk](https://red.anthropic.com/2025/biorisk/))。这将隐私政策担忧与具体的威胁模型（如 AI 辅助生物危害）联系起来，在这种情况下，对用户内容的审查或报告可被视为一种安全控制手段。

- **[Claude 的新使用限制极其离谱。](https://www.reddit.com/r/ClaudeAI/comments/1tzwrxs/claudes_new_usage_limits_are_insane/)** (热度: 1122)：**截图 ([image](https://i.redd.it/6x64517caz5h1.png)) 显示，一个在 **Opus 4.8 搭配 1M context** 下进行的 Claude 编程会话，在约 `12m 54s` 内消耗了 `1.1M tokens`，仅一次 prompt 就让用户的 5 小时限额仅剩 `21%`。该帖认为，结合使用 **Opus + 1M context + UltraCode** 会导致 token 使用量倍增，因为多个并行的 Agent 可能会分别读取庞大的上下文，使得一次请求的表现更像是多次昂贵的调用，而非单次高效的 inference pass。** 评论者大多对该投诉持反对意见，认为在使用最昂贵的模型/上下文/Agent 模式组合时，这是预期内的行为——*“用挖掘机碾死蚂蚁”* 是一个形象的比喻。他们强调 **UltraCode 刻意没有进行 token 优化**，应该保留给特定、高价值的任务，而不是作为默认的 “Max thinking” 模式。

    - 几位评论者认为高使用量是预料之中的，因为用户组合了最耗费 token 的设置：**Ultra Code**、高 “thinking” 等级以及大 context。技术上的结论是，Ultra Code *并不是* “Max thinking” 的 token 高效替代方案；它是为一类更窄的任务设计的，在这些任务中，极高的 token 消耗和成本是可以接受的。
    - 一个反复出现的观点是，开发者需要根据任务复杂度和成本约束来选择模型/工具配置。评论者将此问题视为一个优化问题：在常规工作中使用性能过剩的编程模式，预见性地会耗尽额度，因此工作流应当将 Ultra Code 风格的模式保留给那些额外的 reasoning/context 预算能实质性改善结果的情况。

### 2. Mythos 5 和 Ideogram 4.0 创意模型报告

  - **[Mythos 5：我们还没准备好](https://www.reddit.com/r/ClaudeAI/comments/1tzg6dk/mythos_5_were_not_ready/)** (活跃度: 1348)：**一篇帖子声称 Anthropic 的 “Mythos 5”** 测试模型在**基于 SVG/代码的视觉生成**、前端/UI 创建、游戏、网站甚至代码生成的音乐方面表现出异常强大的实力，某些输出有时需要几分钟才能生成。帖中还引用了据称是 Anthropic 内部的结果：与熟练人类约 `4倍` 的速度相比，该模型在训练代码优化方面实现了高达 `52倍` 的加速；并预计公开版本相对于测试模型将**价格昂贵且可能被削弱 (nerfed)**。热门评论大多持怀疑或讽刺态度：评论者质疑“过于危险的 SVG 生成”这一说法，其中一位评论者认为唯一可信的说法是任何公开模型都将是降级/削弱后的版本；另一位则反对预期的高昂成本。

    - 一位评论者对发布的模型可能与内部测试版本存在实质性差异表示怀疑，并引用了该主张：*“公开版本很可能是当前测试模型的削弱版。”* 技术含义在于，如果 Anthropic 在公开发布前应用后训练限制、能力门控或安全/性能权衡，那么针对 **Mythos 5** 所报告的任何能力主张可能都无法转化到生产环境中。
    - 一个实质性的建议是，如果 **Mythos 5** 的运行成本显著更高，Anthropic 可能需要交付**更小、更便宜、领域专精的模型**，而不是仅仅依赖于单一的前沿通用 (frontier generalist) 模型。这反映了部署中常见的权衡：专门化模型可以在降低推理成本和延迟的同时，在受限领域内保持任务性能。

  - **[Ideogram 4.0 对角色和 IP 的理解对于开源模型来说非常疯狂](https://www.reddit.com/r/StableDiffusion/comments/1u0e1g0/ideogram_40s_understanding_of_characters_and_ip/)** (活跃度: 1081)：**该帖子报告了在 **ComfyUI** 中本地运行的 **Ideogram 4.0** 具有强大的零 LoRA 角色/IP 还原能力，其使用了 **INT8** 模型变体、分辨率为 `1440×1024` (~`1.5 MP`)，并配合 **Kijai 的 Ideogram 4 Prompt Builder KJ** 节点和 **SilverOxide 的工作流** ([Pastebin](https://pastebin.com/xpYezwZp))。作者还强调了 Ideogram 4.0 的局部重绘 (inpainting) 质量，可选择使用 [`ComfyUI-Inpaint-CropAndStitch`](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)，并分享了一个马力欧/索尼克提示词 JSON，使用了 `high_level_description`、`style_description` 和基于边界框的 `compositional_deconstruction` 等结构化字段。** 评论者对这些样本**未使用 LoRA** 感到非常惊讶，有人询问 Ideogram 4.0 的 LoRA 训练是否已经具有实际意义。另一位评论者赞扬了其对特定 IP/细节的处理，例如*“从林克 (Link) 给塞尔达 (Zelda) 的便条”*。

    - 发帖者报告称，**Ideogram 4.0 可以在没有 LoRA 的情况下重现可识别的角色/IP 概念**，称其是他们尝试过的该用途下最强的开源模型。图像是在 **ComfyUI** 中以 `1440x1024` (`~1.5 MP`) 分辨率本地生成的，使用了 **INT8 Ideogram 4.0 模型**，以及 **Kijai 的 Ideogram 4 Prompt Builder KJ 节点**和 **SilverOxide 的工作流** ([pastebin](https://pastebin.com/xpYezwZp))。
    - 共享的一个技术工作流细节是使用了带有 `high_level_description`、`style_description` 和 `compositional_deconstruction` 字段的结构化提示词 JSON，包括对象级的 `bbox` (边界框) 区域和描述。示例提示词明确地通过边界框布置了马力欧和索尼克，包括姿势、面部表情和背景系列脉络，这表明 Ideogram 4.0 受益于空间分解提示 (spatially decomposed prompting)。
    - 发帖者还指出 **Ideogram 4.0 在局部重绘 (inpainting) 方面表现良好**，通常不需要后续清理，但在需要修复蒙版人脸/细节时，他们会使用 **ComfyUI-Inpaint-CropAndStitch** ([GitHub](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch))。这实现了一种实用的工作流：先以较低像素生成，然后对有问题的区域进行选择性重绘以获得更高保真度。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。