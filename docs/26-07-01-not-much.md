---
companies:
- anthropic
- cursor
- cognition
- perplexity
- z-ai
- langchain
- vllm-project
- deepseek-ai
date: '2026-07-01T05:44:39.731046Z'
description: 'Anthropic 重新启用了 Claude Fable 5，并更新了网络安全防护机制，将部分请求转交给 Opus 4.8 处理。这次重新发布推动了
  Cursor、Devin 和 Perplexity 对相关工具的采用。开发者正在通过多模型编排和模型组合策略来适应前沿模型的限制，而不是依赖单一模型。Fable
  5 在 Remote Labor Index 上取得了 16.10% 的成绩；Sonnet 5 则在 AA-Briefcase 中排名第二，但在成本与性能之间存在权衡。


  与此同时，Z.ai 推出了 ZCode——一个面向 GLM-5.2 的开发环境，支持 BYOK 和跨平台使用；LangChain 提供了相关指南，hwchase17
  也记录了开发者对它的采用情况。基准测试显示，GLM-5.2 在 APEX-SWE 上表现领先，在 Integration 项目中的 Pass@1 达到 55.3%，紧随其后的是
  Kimi K2.7，表明两者在编程能力上的差距正在缩小。


  推理方面，vLLM 已为 DeepSeek 系列模型加入 DSpark 推测解码，速度约为 250 tok/s；此外，GLM-5.2 DSpark 的预览版解码速度提升了
  1.5 倍。

  '
id: MjAyNS0x
models:
- claude-fable-5
- opus-4.8
- sonnet-5
- glm-5.2
- kimi-k2.7
people:
- claudeai
- theo
- omarsar0
- mparakhin
- kimmonismus
- artificialanlys
- claudedevs
- cursor_ai
- cognition
- perplexity_ai
- zai_org
- hwchase17
- mercor_ai
- scaling01
- vllm_project
- mgoin_
- jon_durbin
title: '今天没发生什么特别的事。

  '
topics:
- multi-model-orchestration
- model-combination-strategies
- cybersecurity
- coding-ide
- benchmarking
- inference-optimization
- speculative-decoding
- pass-at-1
- integration-testing
---

**平静的一天。**

> 2026 年 7 月 1 日 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有继续查看其他 Discord 服务器。你可以在 [AINews 的网站](https://news.smol.ai/)上搜索所有过往期刊。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以[选择订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同的邮件推送频率！




---

# AI Twitter 综述


**Coding Models、Agent Harnesses 与 Fable 5 的重新上线**

- **Anthropic 重新启用了 Claude Fable 5，但安全降级机制依然十分明显**：在经历了一天的需求积压后，[@claudeai](https://x.com/claudeai/status/2072402636813607381) 宣布 **Fable 5 回归**，并补充说明：更新后的网络安全防护措施可能会将部分请求转交给 **Opus 4.8**，而生物学和化学分类器目前仍然过于宽泛 [@claudeai](https://x.com/claudeai/status/2072402638247968855)。重新上线后，它很快便被各类工具接入：**Cursor** 表示，Fable 5 在其评测中表现领先，但**单个任务的成本最高** [@cursor_ai](https://x.com/cursor_ai/status/2072403323844428217)；**Devin** 已将其加入 Cloud、Desktop 和 CLI [@cognition](https://x.com/cognition/status/2072405137117548601)；**Perplexity** 也恢复了它作为编排模型的身份 [@perplexity_ai](https://x.com/perplexity_ai/status/2072433125104505226)。Anthropic 还在模型恢复上线后为用户重置了速率限制 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2072429181565288665)。
- **真正有意思的地方，与其说是“模型回来了”，不如说是“人们如何适应前沿模型的限制”**：许多开发者不约而同地转向了**多模型编排**，而不是依赖单一模型。[ @theo](https://x.com/theo/status/2072481845363822914)介绍了自己的做法：只让 Fable 负责价值更高的推理和规划，再把实现、验证以及计算机操作交给其他模型；据他称，这显著提升了端到端 PR 的产出率 [@theo](https://x.com/theo/status/2072482460122964067)。[@omarsar0](https://x.com/omarsar0/status/2072400978079261041)也表达了类似观点，认为团队应当设计**模型组合策略**，而不是围绕某一个前沿模型构建系统。[@MParakhin](https://x.com/MParakhin/status/2072275413116784961)则对“简单任务预分类器”提出了质疑，认为可靠的路由往往需要先解决任务本身。在基准测试方面，[@kimmonismus](https://x.com/kimmonismus/status/2072376968729817531)特别提到 **Fable 5 在 Remote Labor Index 上取得了 16.10% 的成绩**；[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2072427328689619241)则报告称，**Sonnet 5** 在 **AA-Briefcase** 中排名第二，但需要更多轮交互；在较低的 effort 设置下，其成本与性能之间的权衡也更差。

**Open Models、Chinese Labs，以及围绕 GLM-5.2 不断扩展的 Coding 工具栈**



- **Z.ai 正在围绕 GLM-5.2 打造完整的产品矩阵，而不只是发布一个 checkpoint**：最具体的发布成果是 **ZCode**，这是官方为 **GLM-5.2** 打造的开发环境，支持 BYOK、跨平台使用，并为 coding-plan 订阅用户提供配额加成 [@Zai_org](https://x.com/Zai_org/status/2072349453361557898)。[@kimmonismus](https://x.com/kimmonismus/status/2072378141041991702) 的评论将其描述为一款 AI 原生编码 IDE，针对 GLM 工作流和长时间运行的自主任务进行了优化。周边生态也在快速发展：**LangChain** 发布了在编码流程中使用 GLM-5.2 的指南 [@LangChain](https://x.com/LangChain/status/2072334663457067064)，而 [@hwchase17](https://x.com/hwchase17/status/2072344890755977571) 也明确提到，开发者正在将 GLM-5.2 用作日常主力模型。
- **基准测试表明，开源编码模型正在弥补某些特定差距，尽管它们整体上还未达到前沿模型的领先水平**：[@mercor_ai](https://x.com/mercor_ai/status/2072448918751941041) 报告称，**GLM 5.2** 成为首个在 **APEX-SWE** 某个类别中取得领先的开源模型，在 **Integration** 项目上取得 **55.3% Pass@1**，并在该测试中成为综合表现最佳的开源模型；**Kimi K2.7** 紧随其后。这与 [@scaling01](https://x.com/scaling01/status/2072346101068238946) 的观点相互补充：他提醒人们不要夸大 GLM 已经超越西方顶尖前沿模型这一说法，但同时也承认编码能力方面的差距正在快速缩小。
- **围绕开源模型的推理优化，正在成为整个生态的重要组成部分**：[@vllm_project](https://x.com/vllm_project/status/2072545387639189798) 在 **vLLM** 中加入了针对 DeepSeek 模型的原生 **DSpark speculative decoding** 支持，并报告称，在 8×B300 上速度约为 **250 tok/s**，相比 MTP 具有更高的接受率；[@mgoin_](https://x.com/mgoin_/status/2072525522639212825) 则发布了 **GLM-5.2 DSpark preview**，声称解码速度大约提升 **1.5 倍**。此外，[@jon_durbin](https://x.com/jon_durbin/status/2072293557172363720) 报告称，在 **Qwen3-32B** 上使用内部开发的 **dflash** drafter，在相同硬件上可带来 **约 50% 的吞吐量提升**。

**Agent 基础设施：记忆、Wiki、技能组合与结构化工作流**

- **“Wiki memory” 正在成为 Agent 的一种实用设计模式**：[@sydneyrunkle](https://x.com/sydneyrunkle/status/2072311589072486879) 认为，**wiki-structured memory** 可以作为一种简单且易扩展的基础设施，而这一想法很快就转化成了产品发布。**LangChain** 推出了 **OpenWiki**，通过 `openwiki --init` 生成并维护 Agent 可使用的代码库文档 [@BraceSproul](https://x.com/BraceSproul/status/2072375499125596262)、[@LangChain](https://x.com/LangChain/status/2072376975545798792)。不同帖子中的动机基本一致：Agent 在不同线程之间反复丢失工作上下文，因此需要一个持续维护、可检查的知识层，而不是原始日志 [@caspar_br](https://x.com/caspar_br/status/2072420582717858292)。
- **记忆系统正在从“只负责检索”转向“协调与维护”**：Weaviate 对 **Engram** 的介绍很好地体现了这一趋势：系统先提取候选记忆，再结合已有记忆进行转换，最后才提交保存。这样可以一次性解决矛盾，而不必在每次查询时重新处理 [@PrajjwalYd](https://x.com/PrajjwalYd/status/2072291317695324410)。[@bpalit](https://x.com/bpalit/status/2072378273343082537) 将同样的观点延伸到了企业场景：Agent 记忆必须具备治理能力、权限感知能力和共享能力，而不能只是一个存放 Markdown 文件的文件夹。
- **结构化组合正在取代“把所有工具都交给模型”这种朴素做法**：[@omarsar0](https://x.com/omarsar0/status/2072430551446032847) 重点介绍了 **SkillComposer**。它将技能选择视为一个联合自回归组合问题，并报告称，相比不使用技能的基线，在 SkillsBench 上分别提升了 **+23.1pp / +18.2pp**。在框架方面，Deep Agents 增加了对 **recursive language model workflows** 的支持 [@sydneyrunkle](https://x.com/sydneyrunkle/status/2072348322526810594)，而 [@hwchase17](https://x.com/hwchase17/status/2072377816780624266) 则将 **dynamic subagents** 与 **Agentic MapReduce** 等模式联系起来。这一总体方向——更明确的工作流结构、fan-out/fan-in 模式，以及由代码强制执行的编排——在各种产品和基准测试中反复出现。

**安全性、评测与 Agentic MapReduce**



- **Cognition 的 Devin Security Swarm 是围绕真实企业工作流构建 Agent 架构的较清晰案例之一**：该系统使用 **Agentic MapReduce**，将边界明确的 Agent 分发到整个代码库中，汇总发现，并在呈现已确认的漏洞之前验证其可利用性 [@cognition](https://x.com/cognition/status/2072368168182432109)。Cognition 声称，与其他方案相比，这种方式**成本效益更高、准确率也更高**；该公司还表示，一项 Fortune 500 企业试点在生产代码仓库中发现并修复了**超过 1,000 个漏洞** [@walden_yan](https://x.com/walden_yan/status/2072377406267273248)。[@jakejluo](https://x.com/jakejluo/status/2072380678419705949) 和 [@levie](https://x.com/levie/status/2072519377371459836) 等开发者的普遍反应是，这种模式可以推广到大规模文档、代码和知识工作流中。
- **AI Agent 评测正迅速发展为一个独立的子领域**：[@random_walker](https://x.com/random_walker/status/2072375245969719374) 提到，近期有多篇论文推动了 Agent 评测的发展，并将其描述为一门独立学科。实际案例包括：**Agent Arena** 重新启用了 Fable 5 的 Agent 模式 [@arena](https://x.com/arena/status/2072423538641031372)；用于评测每兆瓦功耗下 Agent 性能的 **AA-AgentPerf** [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2072254061244825981)；以及 **WorldModelGym**，它评估 world model 是否真的能支持良好的决策，而不只是生成看似合理的模拟结果 [@RekaAILabs](https://x.com/RekaAILabs/status/2072325792558956573)。
- **人们也在推动构建更完善的 AI 失败案例报告流程**：由网络安全和 AI 安全研究人员组成的联盟共同推出 **FLARE-AI**，旨在标准化**缺陷和事故报告**，让问题能够被转交给合适的开发者和登记机构，而不是消失在彼此割裂的受理表单中 [@ClementDelangue](https://x.com/ClementDelangue/status/2072401982569025742)、[@ShayneRedford](https://x.com/ShayneRedford/status/2072408461015707883)。

**值得关注的系统、推理与架构工作**

- **NVIDIA 的 TwoTower 成果展现了生成架构在速度与质量之间的一种具体权衡**：[@NVIDIAAI](https://x.com/NVIDIAAI/status/2072394812301480067) 推出了 **Nemotron-Labs-TwoTower**，将一个 30B 模型改造成 diffusion 风格的语言模型，通过双副本设置并行生成 token。其宣称的结果是：**生成速度提升 2.42 倍**，同时保留原模型**98.7%** 的质量。[@LiorOnAI](https://x.com/LiorOnAI/status/2072402904867365167) 将这一技巧概括为：复用一个冻结的上下文模型和一个经过训练的写作模型，从而避免从头开始进行完整训练。
- **得益于 Agent 式优化和专用运行时，端侧与浏览器推理仍在持续受益**：[@googlegemma](https://x.com/googlegemma/status/2072416614188974274) 展示了 **WebGPU Gemma 4** 在 M4 上达到 **255 tok/s** 的运行效果，据称得益于使用 Fable 5 编写的 kernel。[@andimarafioti](https://x.com/andimarafioti/status/2072335408294236164) 演示了一套完全开源的实时语音技术栈，围绕 **Gemma 4 31B** 和 **Cerebras** 推理构建，目标是成为 OpenAI realtime API 的即插即用替代方案。在 kernel 层面，Hugging Face 的 kernels 库现已提供 MiniMax 的 **MSA kernel** [@RisingSayak](https://x.com/RisingSayak/status/2072277942554841292)，而 Mac 上运行 Triton 也引起了不少关注 [@QuixiAI](https://x.com/QuixiAI/status/2072345855093289005)。
- **对 vanilla LLM 扩展之外的架构研究也开始受到关注**：[@gklambauer](https://x.com/gklambauer/status/2072213633640075366) 提到了 **AdaJEPA**，这是一种由 LeCun 领衔的 world model 方法，通过潜在状态预测误差实现**测试时自适应**；[@LiorOnAI](https://x.com/LiorOnAI/status/2072380547603829224) 将 **NEO** 概括为学习可复用的因果“程序”，而不只是预测下一帧；[@ziv_ravid](https://x.com/ziv_ravid/status/2072402889092616309) 则强调，“在想象中训练”正在成为一种活跃的范式，而不再只是推测。

**热门推文（按互动量排序）**



- **Fable 5 的可用性占据了技术讨论的主流**：[​@claudeai：“Fable 5 回来了。”](https://x.com/claudeai/status/2072402636813607381)、[​@ClaudeDevs 关于速率限制重置的说明](https://x.com/ClaudeDevs/status/2072429181565288665)，以及[​@cursor_ai 宣布 Fable 5 在 CursorBench 中排名第一](https://x.com/cursor_ai/status/2072403323844428217)。
- **覆盖面广泛的系统/基础设施发布**：[​@NVIDIAAI 宣布 TwoTower 在保持 98.7% 质量的同时，生成速度提升 2.42 倍](https://x.com/NVIDIAAI/status/2072394812301480067)。
- **Open model 生态持续升温**：[​@Zai_org 为 GLM-5.2 发布 ZCode](https://x.com/Zai_org/status/2072349453361557898)，以及[​@TogetherCompute 宣布完成 8 亿美元 C 轮融资，估值达到 83 亿美元](https://x.com/vipulved/status/2072321276094673083)。
- **高关注度的工具与知识层发布**：[​@LangChain/OpenWiki](https://x.com/LangChain/status/2072376975545798792) 和 [​@cognition/Devin Security Swarm](https://x.com/cognition/status/2072368168182432109)。



---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Open-Weight Model 发布与本地运行时基准测试

  - **[I extended Gemma4-31B to 44B (88 layers)  — since Google won't give us anything bigger than 31B](https://www.reddit.com/r/LocalLLaMA/comments/1ul0cx9/i_extended_gemma431b_to_44b_88_layers_since/)**（热度：747）：**这张图片是一张**技术架构信息图**，展示了帖子中所声称的 Gemma4 扩展方案：图中将类似 Gemma4-31B 的 `60` 层混合基础模型，通过插入 attention 层扩展到 `80` 层；随后再通过复制模块，扩展为 `88` 层、约 `44–47B` 参数的变体，重点强调了**恒等初始化**、权重零初始化，以及将 `layer_scalar = 1.0` 设为固定值以保证稳定性。在相关背景中，作者表示，这样做的目标是为韩语法律和 STEM 微调增加“空白容量”，同时不覆盖基础模型中原有的密集知识，并在 [Hugging Face model card](https://huggingface.co/TOTORONG/extGemma4-44B) 中提供了实现和说明；图片本身位于：[https://i.redd.it/qbkvzo4s3pah1.png](https://i.redd.it/qbkvzo4s3pah1.png)。**评论中的主要技术反馈是，应当将该方法与更简单的 **RYS / “repeat yourself”** 基线进行比较，也就是直接复制连续层，作为一种快速且较粗糙的模型扩展策略。其他评论大多是鼓励或非技术性建议，没有提供实质性的评估。

    - 有评论者建议，将这个 44B/88 层的 Gemma 扩展模型与 **RYS（Repeat Yourself）** 基线进行基准测试。RYS 会直接复制原始模型中的连续层，是一种快速且较粗糙的参数规模扩展方法。他们认为，这样的对照测试有助于判断：在模型规模相近的情况下，提出的层扩展策略是否优于简单的层重复。
    - 社区对后续的**量化**工作表现出兴趣，尤其是如果社区能够提供相关构建版本。这意味着，对于不具备数据中心级硬件的用户而言，44B 模型的实用性将取决于低精度版本的发布。另一位评论者认为，这种方法类似于 **Llama 2 / Llama 3** 时代早期的一些“大杂烩（Frankenstein）”大型模型实验；在官方更大规模 checkpoint 尚未推出之前，人们曾探索过合并或扩展模型架构。

  - **[nvidia/Qwen3.6-27B-NVFP4 just dropped](https://www.reddit.com/r/LocalLLaMA/comments/1ujlltn/nvidiaqwen3627bnvfp4_just_dropped/)**（热度：702）：****NVIDIA** 发布了 [`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)，这是 **Qwen3.6-27B** 的 NVFP4/混合精度量化版本。评论者指出，该模型发布后的大小约为 `22 GB`。与大约 `26 GB` 的 [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) 相比，它在 `32 GB` 显存环境下明显更合适；但对于所谓的“4-bit”模型来说，它仍比一些人的预期更大，因为 NVFP4 部署通常会包含缩放信息/元数据，以及 `F8_E4M3` 等混合 FP8 组件——FP8 使用 4 位表示指数、3 位表示尾数。**讨论的重点在于如何设定预期：用户原本希望 NVFP4 的大小能接近 Q8/FP8 的一半，而另一些人则认为，混合精度带来的额外开销可以解释为何压缩效果没有预想中那么明显。人们还期待将其与 Unsloth 版本进行直接的质量/性能对比，并希望未来能推出 GGUF 转换版本。



- 评论者比较了 `Qwen3.6-27B` 的 **NVIDIA** 和 **Unsloth** NVFP4 版本：据称 NVIDIA 的文件约为 `22 GB`，而 Unsloth 的约为 `26 GB`，因此 NVIDIA 版本对配备 `32 GB` VRAM 的显卡来说更实用。一位用户指出，由于两者似乎都是混合精度格式，相比 FP8，它们的体积缩减幅度没有名义上“4-bit”模型那么明显。
    - 有人不明白为什么经过 `NVFP4` 量化的 `27B` 模型仍然有 `22 GB`，原本预计它的大小应该更接近 Q8 的一半。讨论中还提出了一个精度格式问题：`F8_E4M3`，也就是指数位为 `4` 位、尾数位为 `3` 位的 FP8，在某些混合精度布局中用于存储主要权重。
    - 用户询问 NVIDIA 的版本与 [`unsloth/Qwen3.6-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) 相比如何，以及是否会发布适用于 llama.cpp 风格推理的 **GGUF** 转换版本。另一个技术问题是，该模型是否支持推理期间的 **MTP**。

  - **[[audio.cpp] VibeVoice 1.5B 已发布——22.95 分钟生成 90 分钟播客，达到 4.08 倍实时速度；未量化时也比 Python 快 2.86 倍。原生 C++/ggml](https://www.reddit.com/r/LocalLLaMA/comments/1uk7khq/audiocpp_vibevoice_15b_released_90min_podcast_in/)**（活跃度：583）：****audio.cpp** 为 **VibeVoice 1.5B** 增加了原生 C++/`ggml` 支持。在 **RTX 5090** 上进行多说话人 TTS 生成测试时，生成时长为 `5615.73s` / `93.60 min` 的内容耗时 `1376.84s` / `22.95 min`，`RTF=0.245`，即达到实时速度的 `4.08×`，比 Python 基线快 `2.86×`；测试未使用量化，并采用了 `10` 个扩散步骤。作者将其视为长文本 TTS 运行时的一项里程碑，重点关注可复用会话、类似服务器的本地推理、稳定的内存表现以及 CUDA 优化；目前 [audio.cpp 仓库](https://github.com/0xShug0/audio.cpp) 已发布 `16/28` 个模型系列。**评论大多表示支持，并对实现所需的工作量感到好奇；有评论者表示，这样的加速会让 TTS/语音转换对他们来说变得实用。作者还征集了对其他模型的支持需求，以及跨 GPU/CPU 的性能数据。

    - 一位评论者分享了此前关于 `audio.cpp` 性能的讨论，其中涵盖 **Qwen3-TTS** 和 **PocketTTS** 等其他 TTS 后端，可用于将此次 VibeVoice `1.5B` 原生 C++/ggml 的吞吐量与早期本地 TTS 基准进行比较：[此前的性能讨论](https://www.reddit.com/r/LocalLLaMA/s/GNRnwiL7Nh)。
    - 社区明确希望将 `audio.cpp` 的支持范围扩展到 VibeVoice `1.5B` 之外，包括请求支持更大的 **VibeVoice 7B** 模型。这说明用户希望在同一个 C++/ggml 运行时中，对不同模型规模下的质量与速度取舍进行基准测试。
    - 一位用户认为，报告中的实时生成速度 `4.08x` 以及相较 Python 的 `2.86x` 加速，可能会让本地 **TTS 和语音转换** 真正适用于他们的工作流程；同时他们还询问了实现所需的工作量，以及编码模型是否确实能帮助完成底层 C++ 工作。

  - **[Huawei 开源 OpenPangu-2.0-Flash——总参数量 92B，激活参数量 6B](https://www.reddit.com/r/LocalLLaMA/comments/1ujn5u3/huawei_opensources_openpangu20flash_92b_total6b/)**（活跃度：512）：****Huawei** 开源了 **OpenPangu-2.0-Flash**，这是一个上下文长度为 `512K` 的 MoE 模型，号称总参数量为 `92B`、激活参数量为 `6B`；根据 [X](https://x.com/Chinazhidx/status/2071877413685109071) 上的公告，此次同时发布了**权重、推理代码和训练算子**。同一篇帖子还表示，**OpenPangu-2.0-Pro** 计划于 7 月发布，定位为更大型的上下文长度 `512K` 旗舰模型，总参数量为 `505B`、激活参数量为 `18B`；今年晚些时候还将陆续开源更多组件，后续的基准测试/相关声明讨论见[这里](https://x.com/CalatheaAI/status/2071917592810496273)。**评论者对 Huawei 发布更完整的开源技术栈持谨慎乐观态度，但也质疑模型质量以及基准测试的具体性。一项技术层面的批评是，诸如*“超过 Gemma 4”*这样的说法过于模糊，却没有说明具体比较的是 Gemma 的哪个版本，例如是否指 `26B-A4B`。



- 评论者指出，**OpenPangu-2.0-Flash** 在技术上最值得关注的地方，可能是其发布方式，而不是原始基准测试成绩：Huawei 似乎正朝着“完全开源”迈进，公开发布了**模型权重、数据集和训练细节**。对于一家正在打造完整模型 + 运行时生态的硬件厂商来说，这一点相当值得注意。
- 有人对“超越 Gemma 4”这一说法持怀疑态度，指出其比较标准并不明确——例如，Huawei 比较的究竟是 **Gemma 3/4 风格的稠密模型，还是 `26B-A4B` 这类 MoE 变体**。令人担忧的是，对于一个总参数量为 **`92B`、激活参数量为 `6B`** 的 MoE 模型来说，如果只是击败一个激活参数量较小的基线模型，这并不能算是特别强的结果。
- 有人提出了一个技术上很重要的观点：**Pangu 可能完全使用 Huawei 加速器训练，而不是 NVIDIA GPU**。在出口管制限制的背景下，这使它具有重要的战略意义。一位评论者将其与 DeepSeek 此前据称使用 Huawei 芯片训练的计划进行了对比：据称该计划后来由于集群调试问题，主要退回到使用 Huawei 芯片进行推理，并认为 Pangu 证明了在非 NVIDIA 的国产硬件上同样可以训练出实用的 LLM。




## 低技术含量的 AI 子版块回顾

e /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. Claude Sonnet 5 发布基准测试

  - **[Introducing Claude Sonnet 5, our most agentic Sonnet yet.](https://www.reddit.com/r/ClaudeAI/comments/1ujwggp/introducing_claude_sonnet_5_our_most_agentic/)**（活跃度：3549）：**[基准测试表](https://i.redd.it/gspb3e6begah1.png)支持 Anthropic 将 **Claude Sonnet 5** 定位为更具 Agent 能力的 **Sonnet 4.6** 后继版本这一说法。数据显示，它在编码、推理、计算机操作和知识工作任务上均有所提升。据报道，其在 **SWE-bench Pro** 上的得分为 `63.2%`，在 **Terminal-Bench 2.1** 上为 `80.4%`，在 **OSWorld-Verified** 上为 `81.2%`，整体表现接近 **Opus 4.8**。帖子还声称，Sonnet 5 价格更低，并且在 Free/Pro 计划中有更广泛的默认可用性。**评论者关注的重点并不是原始基准成绩，而是产品取舍：有人表示，如果 Sonnet 5 没那么啰嗦，却能达到接近 Opus 的表现，那会很受欢迎，并开玩笑说：*“Opus 4.8 说起话来像一个不停灌糖的幼儿。”* 其他人则对 Haiku 等更小型模型感到失望，或开玩笑地要求推出一个假想中的“Fable”模型。

    - 评论者认为，如果 **Claude Sonnet 5** 能以远低于 **Opus 4.8** 的输出量，达到接近后者的质量，它可能会很有吸引力。一位用户表示，只要它能做到*“用三分之一的输出量，达到几乎和 Opus 4.8 一样的效果”*，自己就会采用它。这反映出用户对更低的冗长度、更少的 token 成本以及更快 Agent 循环的兴趣。
    - 有人介绍了一种技术工作流：使用 **Opus** 负责高层规划和编排，再将执行工作委派给成本更低的 **Sonnet agents**。该评论者认为，Sonnet 的改进很重要，因为更强的低成本模型能让多 Agent 架构变得更实用、更容易普及，而不必每项任务都依赖 Opus/Fable 级别的模型。

  - **[Looks like Anthropic quietly updated the Sonnet 5 'Agentic search' benchmark graph overnight](https://www.reddit.com/r/ClaudeAI/comments/1ukgqwr/looks_like_anthropic_quietly_updated_the_sonnet_5/)**（活跃度：1173）：**这张图片对比了 Anthropic 的 **“Agentic search performance by effort level”** BrowseComp 图表的两个版本。新版似乎重新缩放并扩展了两个坐标轴，并且明显改变了 **Sonnet 5**、**Opus 4.8** 和 **Sonnet 4.6** 在**通过率与单任务成本**之间的相对位置。这里的技术意义并不在于出现了新的基准测试结果，而在于其展示方式和可复现性存在问题：更新后的图表让这些模型看起来集中在更高的通过率和成本区间，引发了人们对原图是否存在坐标缩放错误、绘制数值错误，或是否在没有任何说明的情况下被悄悄修改的疑问。[图片](https://i.redd.it/rwtrj2vq6lah1.jpeg)** 评论者对这一基准测试可视化结果高度怀疑，称其为 *“trust me bro” 图表* 和 *“凭感觉画图”*。争论的核心在于：这是一次善意的修正，还是说明厂商发布的基准测试图表过于不透明，若没有原始数据和变更记录，就不值得信任。

    - 评论者提出了一个方法论问题：Anthropic 的“Agentic search”基准测试可视化结果似乎变成了一张*截然不同的图表*，而不只是修正坐标轴比例或替换某个模型的数据。技术层面的主要结论是：如果没有可复现的数据、有版本记录的方法说明或变更日志，就应当对厂商发布的基准测试图表保持怀疑；这类图表实际上就是所谓的“trust me bro”图表。



  - **[同样价格下，Sonnet 5 在 high 和 xhigh 档位不如 Opus？](https://www.reddit.com/r/ClaudeAI/comments/1ujx3rw/sonnet_5_is_worse_than_opus_at_the_same_price_at/)**（热度：1173）：**这张[图片](https://i.redd.it/usofw9d8pgah1.jpeg)是一张基准测试图表，比较了 **Sonnet 5**、**Opus 4.8** 和 **Sonnet 4.6** 在不同 effort 档位下的 BrowseComp Agentic Search 性能与单任务成本。图表显示，在 `high` 和 `xhigh` effort 下，**Opus 4.8 的性价比可能高于 Sonnet 5**：在成本相近的情况下，Opus 的通过率约为 `70–72%`，而 Sonnet 5 最高约为 `65–69%`。这与帖子标题的观点一致：在相同价格档位下，Sonnet 5 可能不如 Opus。**评论者普遍对此感到失望，认为如果 Opus 在相近成本下更快或更好，那么在 `high/xhigh` 档位使用 Sonnet 5 就“没有意义”。一位用户表示，Sonnet 5 完成一项任务耗时 `17 分钟`，并占用会话额度的 `9%`；而 Opus 4.6/4.8 只用了约 `3 分钟` 和 `4–5%`，进一步引发了对延迟和会话成本效率的担忧。

    - 用户反映，Sonnet 5 在高档位下的**延迟和额度效率较差**：一位评论者称，一项基于标准的提纲评分任务耗时 `17 分钟`，并消耗了 `5X` 会话额度的 `9%`；相比之下，**Opus 4.6/4.8** 据称只需约 `3 分钟`，占用 `4–5%` 的会话额度。这表明，尽管宣传价格相近，Sonnet 5 在某些实际工作负载下的吞吐量和成本表现可能明显更差。
    - 也有人指出，这种比较取决于如何解读图表中的档位：有人认为，**Sonnet 5 High** 的成本与 **4.6 Low** 大致相同，但性能据称有所提升；而 **Sonnet 5 Medium** 的价格则比 **4.6 overall** 低得多，同时性能大致相当。双方争论的技术核心在于：应该比较 high/xhigh 档位，还是应从 medium/low 档位的成本性能定位来判断。


### 2. Claude Fable 5 的出口管制与安全措施

  - **[Claude Mythos 5/Fable 5 的出口限制解除](https://www.reddit.com/r/ClaudeAI/comments/1uk5ihe/claude_mythos_5fable_5_export_restrictions_lifted/)**（热度：1602）：**这张[图片](https://i.redd.it/39qj3w9waiah1.jpeg)是一封美国商务部于 **2026 年 6 月 30 日** 出具的信函。信中表示，此前 6 月 12 日信函针对 **Anthropic 的 Claude Mythos 5 和 Claude Fable 5** 施加的出口许可要求已被**撤回**。从技术层面看，这意味着这两款模型的权重或服务在出口、再出口以及境内转移时，不再需要商务部此前要求的特定许可证；这似乎是 Anthropic 针对安全风险采取缓解措施后得到的结果。帖子还附上了 [Anthropic 在 X 上发布的公告](https://x.com/AnthropicAI/status/2072106151890809341)。**评论者主要关注产品何时恢复，而不是政策细节；他们询问 Anthropic 何时会“重新启用”访问权限，并开玩笑或请求“提前重置”，这表明用户期待限制解除后服务恢复或额度发生调整。

    - 一位评论者认为，解除出口限制后，应当将当前表现与此前 Claude Mythos 5/Fable 5 的结果进行**对比基准测试**。他指出，训练阶段或后训练阶段为了削弱模型在某一领域的能力而采取的干预措施，可能会无意中损害模型在其他领域的表现。这里关注的重点是检测能力退化，而不能仅仅因为访问恢复，就默认模型行为没有变化。

  - **[Fable 5 回来了。](https://www.reddit.com/r/ClaudeAI/comments/1ukvjyn/fable_5_is_back/)**（热度：2607）：****Anthropic 表示，在与美国政府协商后，Fable 5 已重新部署，并加入了更新后的网络安全防护措施。这些措施可能会暂时增加安全系统的误判回退；被标记的请求将改由 **Opus 4.8** 处理。生物学/化学分类器与发布时保持不变，对于一些基础的、与生物学相关的查询，触发回退的范围仍然较广；Anthropic 承诺很快修复这一问题。付费套餐用户可享受截至 **7 月 7 日** 的促销访问权限，但上限为**每周使用量的 50%**；之后仍可通过使用额度继续访问（[支持详情](https://support.claude.com/en/articles/15424964-claude-fable-5-promotional-access)、[博客文章](https://www.anthropic.com/news/redeploying-fable-5)）。**评论总体上较为欢欣，但有一个值得注意的担忧是：一旦 Fable 5 恢复按使用额度计费，许多用户可能会觉得它贵到难以经常使用。



    - 一位 `$100` 套餐用户表示，让 **Fable 5** 审查近期新增功能后，它生成了 `18` 个 Fable 子 Agent，很快耗尽了一个 `5 hour` 使用时段中剩余约 `50%` 的额度。即使用户中断任务并要求它停止或限制 token，这些 Agent 也只是开始收尾；账号在大约 `120 seconds` 内就达到了额度上限的 `101%`，凸显出自主子 Agent 扇出可能带来的严重额度消耗问题。
    - 多位评论者担心，Fable 恢复按**使用额度**计费后，许多用户可能会用不起。此次子 Agent 的行为表明，除非系统提供更严格的并发、token 数量或 Agent 生成控制，否则成本的可预测性可能会成为一大问题。

  - **[Fable available for plans until July 7th after which it becomes usage credit based](https://www.reddit.com/r/ClaudeAI/comments/1ukafrm/fable_available_for_plans_until_july_7th_after/)**（活跃度：2039）：****Anthropic** 表示，**Fable 5** 正在 Claude Platform、[Claude.ai](https://claude.ai)、Claude Code 和 Claude Cowork 上面向全球重新部署；Pro/Max/Team/部分 Enterprise 套餐可使用该模型，但截至 **July 7**，使用量最高限制为每周额度的 `50%`，之后将改为按**使用额度**计费（[公告](https://www.anthropic.com/news/redeploying-fable-5)）。通过 **AWS**、**Google Cloud** 和 **Microsoft Foundry** 提供的云端访问也在恢复，而 **Mythos 5** 仍仅限获批准的美国机构使用；Anthropic 还表示，正在与主要云合作伙伴共同制定一套统一的越狱严重性评估框架，并为提交 Fable 5 网络安全越狱报告推出 **HackerOne** 渠道。**评论区的主要观点强烈反对这一变动：原本预期的访问周期被缩短为 `7` 天且只能使用一半额度；还有多位用户认为按使用额度计费会贵得令人难以承受，其中有人声称一次会话在 Opus 4.8 上就花费了 `$124`。另一些人则嘲讽 Anthropic 关于越狱分类的说法，认为这种表述过于简单化，或带有政治动机。

    - 用户担心，Fable 的发布安排与原本预期的 `14` 天套餐内访问相比发生了重大变化：现在大约只有 `7` 天，且在 July 7 之后将改为按使用额度计费。评论中最具体的成本数据是：据称某次会话在 **Opus 4.8** 上消耗了 `$124` 的额度；评论者认为，这使得许多用户难以长期、持续地使用它。
    - 一些评论者认为，从订阅或套餐内访问切换到按使用量计费的额度模式，不只是可用性变化，更是一次严重的计费模式倒退。讨论重点并不在功能质量，而在按量推理成本、缩短的访问窗口，以及减少的套餐内可用额度对实际使用造成的影响。

  - **[Fable is going to be redirecting coding task to Opus 4.8](https://www.reddit.com/r/ClaudeAI/comments/1ukcmji/fable_is_going_to_be_redirecting_coding_task_to/)**（活跃度：1043）：**图片是一张 **Anthropic X 帖子**的截图。帖子称，**Claude Fable 5** 将再次面向全球开放，但会采用更严格的安全分类器，拦截更多与网络安全相关的任务；在大约 **July 7** 之前，常规编码和调试工作会暂时转交给 **Opus 4.8**。其技术意义在于：一个本应具备顶级编码能力的模型，正受到安全缓解措施和回退路由的限制，这引发了人们对基准测试成绩与实际可用性、实用价值之间差距的讨论。[图片](https://i.redd.it/1opie5x50kah1.jpeg) 评论者对该模型在网络安全、生物学/化学以及现在的编码领域都受到限制感到不满，认为它最终主要只适合跑基准测试，而不适合实际工作。还有人反复呼吁开发开源的“mythos-level”模型，以对抗专有模型的安全门控。

    - 一位评论者澄清说，这项政策被误读了：根据所引用的文档，并不是**所有编码任务**都会转交给 Opus 4.8；只有被判定为存在**安全风险**的提示词，才会回退到 Opus。由此，关键技术问题在于：负责判断代码相关请求何时越过风险边界的安全分类器，其行为和判断准确性究竟如何。




# AI Discord 社区

很遗憾，Discord 今天终止了我们的访问权限。我们不会以这种形式恢复它，但很快会推出全新的 AINews。感谢你读到这里，这段旅程曾经很美好。