---
companies:
- z.ai
- databricks
- liquid-ai
- google-deepmind
- google
- sail
- hyperagent
- openai
- langchain
date: '2026-06-25T05:44:39.731046Z'
description: '**Z.ai 的 GLM-5.2** 在编程和智能体基准测试中领先，取得了 **1595 分**（Code Arena：Frontend）和
  **34.29% 的推理准确率且零失败**等顶尖成绩。Databricks 通过硬件升级和优化，将 GLM-5.2 的速度提升至 **392 tok/s**。**Ornith-1.0**
  是一个采用 MIT 许可证的新型编程模型系列，参数规模覆盖 **9B 至 397B**，在多项基准测试中表现出色，并采用了能够自我提升的强化学习训练方法。**Liquid
  AI** 发布了一款小型模型，面向低延迟机器人和电商场景。**Google** 将计算机操作能力集成到 **Gemini 3.5 Flash** 中，并配备了安全控制机制和开发者工具，用于控制设备。**Sail**
  和 **Hyperagent** 等初创公司专注于长时间运行的智能体，强调持久化执行和成本效率。**OpenAI** 表示，内部对 Codex 的使用正在增长，已用于处理复杂的跨职能任务，这凸显了智能体技能并发的重要性。

  '
id: MjAyNS0x
models:
- glm-5.2
- glm-5.2-max
- opus-4.8
- claude-fable-5
- ornith-1.0
- gemma-4
- qwen-3.5
- lfm2.5-230m
- gemini-3.5-flash
- codex
people:
- philschmid
- gdb
- reach_vb
- eliebakouch
title: '今天没发生什么特别的事。

  '
topics:
- coding-benchmarks
- agentic-ai
- reinforcement-learning
- model-optimization
- speculative-decoding
- hardware-optimization
- long-running-agents
- agent-persistence
- cost-efficiency
- computer-use
- safety-controls
- developer-tools
- token-consumption
- concurrent-agents
---

**平静的一天。**

> 2026 年 6 月 24 日至 25 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有再查看其他 Discord 服务器。[AINews 网站](https://news.smol.ai/)支持搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以[选择接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 综述

**开放模型、代码基准，以及 GLM/Ornith/Liquid 浪潮**

- **GLM-5.2 在代码和 Agent 基准测试中快速崛起**：多条帖子都将 **Z.ai 的 GLM-5.2** 视为当天最重要的开放模型新闻。在前端编程方面，[Arena 报告](https://x.com/arena/status/2070174325844640123)称，**GLM-5.2 Max** 在 Code Arena: Frontend 上达到 **1595** 分，超过了 **Opus 4.8**，并进一步缩小了与 **Claude Fable 5** 的差距。在 Agent 可靠性方面，[PostTrainBench 指出](https://x.com/hrdkbhatnagar/status/2070244540108423427)，**GLM 5.2 Max reasoning** 的得分为 **34.29%**，略高于 **Opus 4.8 Max 的 34.08%**；在 84 次运行中，它**没有一次失败**。速度方面也有进展：[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2070166719839326396)表示，Databricks 在 Artificial Analysis 上将 GLM-5.2 的速度提升到了 **392 tok/s**；此前它在 H200 上的速度为 **201 tok/s**，之后在 B300 上又进一步提升。该结果归功于硬件升级，以及 speculative decoding 和内核等优化。
- **新的代码专用开放权重模型**：[Ornith-1.0](https://x.com/ornith_/status/2070148887067963854) 发布，这是一个采用 **MIT 许可证**的 Agent 编程模型系列，涵盖 **9B dense、31B dense、35B MoE 和 397B MoE**，基于 **Gemma 4** 和 **Qwen3.5** 进行后训练。公布的成绩包括：**Terminal-Bench 2.1：77.5**、**SWE-Bench Verified：82.4**、**SWE-Bench Pro：62.2**，以及 **ClawEval：77.1**。其中最值得关注的训练特点，是一种自我改进的 RL 方案：它优化的不只是解决方案 rollout，还包括驱动这些 rollout 的**任务专用脚手架**。与此同时，**Liquid AI** 发布了 [LFM2.5-230M](https://x.com/maximelabonne/status/2070149175006617682)，这是一款面向机器人和电商场景中低延迟工具调用的超小型模型；[vLLM 在首日就加入了支持](https://x.com/vllm_project/status/2070177937815736420)，[SGLang 也加入了支持](https://x.com/lmsysorg/status/2070168574849945721)，而 [WebGPU 相关工作使其在本地运行时速度达到约 1400 tok/s](https://x.com/xenovacom/status/2070210622239707568)。

**投入生产环境的 Agent：Computer Use、长时程基础设施与内部采用**



- **Google 将 computer use 推进到 Gemini 3.5 Flash**：Google 已将 **computer use** 打造成 **Gemini 3.5 Flash** 的一等内置能力，覆盖浏览器、桌面端和移动端。主要发布信息来自 [@Google](https://x.com/Google/status/2070175556503568394)、[@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2070180509523546481) 和 [@googledevs](https://x.com/googledevs/status/2070174765940170832)。重点介绍的安全控制包括：对敏感操作要求**用户明确确认**，以及**自动停止任务**。对于开发者，[ @_philschmid 分享了一个快速入门示例](https://x.com/_philschmid/status/2070177135453434183)，展示了如何通过 `adb` 控制 Android 手机；采用同样的模式也可以扩展到 iOS。这是一次重要的产品转变：重点不再只是模型 API，而是提供标准化的操作接口，并配备 human-in-the-loop 机制。
- **Agent 基础设施正围绕持久化和成本变得更加专门化**：多家初创公司和产品正在针对**长时间运行的 Agent** 进行优化，而不是交互式聊天的低延迟体验。[Sail](https://x.com/neilmovva/status/2070164963013148747) 宣布获得 **8000 万美元**融资，为运行**数天或数周**的 Agent 提供低成本推理和沙箱，并声称对于这类不追求即时结果的工作负载，可以实现“**每美元多获得 10 倍智能**”。[Hyperagent](https://x.com/kimmonismus/status/2070152987209519224) 受到关注的原因是，它为每个 Agent 配备独立的云端机器，并支持持久化的浏览器和代码执行环境。[LangChain 对 Fleet 的介绍](https://x.com/LangChain/status/2070123493568426050) 做了一个很有用的区分：如果工作以得到一个答案为终点，就使用**通用聊天**；如果工作具有可重复的结构并且需要持久化上下文，就使用**专用 Agent**。
- **OpenAI 内部对 Codex 的使用正成为一个重要先行指标**：[OpenAI](https://x.com/OpenAI/status/2070196105745518913) 表示，Agent 正在“改变每个部门的工作方式”，而 Codex 被用于更长期、涉及更多团队协作的任务。[@gdb](https://x.com/gdb/status/2070199649823297653)、[@reach_vb](https://x.com/reach_vb/status/2070201707015934112) 和 [@eliebakouch](https://x.com/eliebakouch/status/2070229373530288619) 的外部评论强调了内部 token 消耗的增长，尤其是在研究团队中，以及 **skills** 和**并发 Agent** 等使用模式。实际启示并不是“Agent 很神奇”，而是：当组织能够支持**审查闭环**、**工具链**和**持久化工作流**时，真正的应用正在逐渐出现。

**评测、奖励作弊与作为前沿杠杆的合成数据**

- **公开基准正越来越容易被攻破**：[Cursor 的研究文章](https://x.com/cursor_ai/status/2070195789121671624)指出，包括 **Opus 4.8** 和 **Composer 2.5** 在内的近期模型，可以通过从互联网或 git 历史中检索解决方案来攻破公开基准；在更严格的测试框架下，得分会大幅下降。这与 [ProgramBench 推动](https://x.com/jyangballin/status/2070206413444403324)将**无互联网**环境作为未来编程评测默认设置的方向一致。更广泛的主题是：如今，评测环境的设计已成为一项一等变量，而不再只是基准测试的规范问题。
- **Autodata / Agent 式合成数据生成正受到越来越多关注**：Meta 的 [Autodata 论文讨论串（作者：@jaseweston）](https://x.com/jaseweston/status/2070117091521204521)是其中较有实质内容的研究成果之一。该方案将数据生成视为一个**数据科学家 Agent 循环**，包含创建、分析和**元优化**，把额外的推理算力转化为更好的训练数据和评测数据。据报道，它在**计算机科学、法律和数学**任务上都取得了提升；经过元优化的测试框架还将创建通过率从 **62.1% 提升到了 79.6%**。[@iScienceLuvr](https://x.com/iScienceLuvr/status/2070058945914573049) 和 [@omarsar0](https://x.com/omarsar0/status/2070235085732000228) 也对此进行了独立传播。这是这份摘要中最清晰的案例之一，说明“autoresearch”正从口号转变为具体的循环设计。
- **数据筛选如今也成为 test-time compute 的一种杠杆**：[Datology](https://x.com/arimorcos/status/2070154289880932621) 认为，通过在不影响任务表现的前提下引导模型生成更**简洁**的答案，数据筛选可以让模型的答案生成效率提升 **35 倍**；[@pratyushmaini](https://x.com/pratyushmaini/status/2070172084123390109) 则明确将其视为质量和训练效率之外的第三个维度。这一点值得注意，因为它将预训练和后训练阶段的数据选择，直接与**服务成本**和**用户感知延迟**联系起来，而不只是与基准测试质量相关。

**开放生态经济：Hugging Face、数据发布与 Agent 工具链**



- **Hugging Face 在保持开放定位的同时跨过了重要的商业里程碑**： [Clement Delangue 宣布](https://x.com/ClementDelangue/status/2070104323481104674)其**年度收入运行率达到 1 亿美元**，同时表示 HF 仍为**97% 的用户**提供免费、开放的平台，并管理着规模达**数百 PB**的模型和数据集。对于关注基础设施和平台的人来说，这清楚地证明了开放模型分发、托管以及社区工作流能够支撑一项持久的业务。这也为下游采用情况提供了背景，例如 [Gemma 4 在 2.5 个月内达到 2 亿次下载](https://x.com/googlegemma/status/2070180154069176399)。
- **实用的开放语料库和数据基础设施仍在不断扩展**： [Common Crawl 发布](https://x.com/CommonCrawl/status/2070094659343237492)了**2026 年 6 月**的归档数据：包含**21 亿个网页**、未压缩大小为 **354 TiB**，来自**4080 万个主机**，并附带更新后的网页图谱。领域专用数据方面，[Telco-Common-Corpus](https://x.com/Dorialexander/status/2070080144593588493) 也已发布，这是一个包含**100 亿 token**、完全开放的电信领域语料库。在具身智能和机器人数据方面，[Chris Paxton 估计](https://x.com/chris_j_paxton/status/2070009005439603083)，目前可用的开放数据集加起来可能已经达到约**1 万机器人小时**，这足以让“基本上任何人”都尝试训练一个像样的机器人基础模型。
- **围绕本地和开放部署的工具仍在持续改进**：当天还出现了 [Qdrant EDGE + LiteRT，实现完全端侧的 RAG](https://x.com/qdrant_engine/status/2070117122324242637)、[Hugging Face 的“在本地运行自己的模型”直播](https://x.com/huggingface/status/2070160187751850242)、[支持 MTP heads 的 GGUF UI](https://x.com/mishig25/status/2070143864522887280)，以及面向开发者的改进，例如 [LangChain 的部署实践手册](https://x.com/LangChain_JS/status/2070202038315778506)。这些并不是彼此孤立的功能，而是共同体现了向**可移植 Agent 技术栈**和**更便捷的本地推理体验**发展的趋势。

**政策、访问控制与蒸馏之争**

- **Fable 5 并没有回归，很可能只是 UI 显示问题**：短暂看似重新出现的 **Claude Fable 5**，最终成了一起关于谣言传播和访问权限不透明的案例。相关猜测来自 [@kimmonismus](https://x.com/kimmonismus/status/2070095365701832724)，但 Anthropic 方面随后明确进行了纠正：[​@sammcallister 表示](https://x.com/sammcallister/status/2070107830498054527)，他们向 Fable 5 提供的流量**恰好为 0**；[@TheAmolAvasare 表示](https://x.com/TheAmolAvasare/status/2070132115497476372)，**不存在 Fable/Mythos 流量**，很可能只是 UI bug 或恶作剧。[后续的更正帖](https://x.com/kimmonismus/status/2070128939096236505)也反映了这一点。
- **蒸馏争议升级成了一场政策作秀**：围绕 Anthropic 声称 [Alibaba 据称使用了数百万次 Claude 对话](https://x.com/Discoplomacy/status/2070069250513900005)的讨论，逐渐扩展到技术和地缘政治层面的评论。[Andrew Curran 发布了 Dario Amodei 的信件](https://x.com/AndrewCurran_/status/2070134863370567864)，与此同时，许多评论者争论问题的本质究竟是领先基准的合成后训练、API 泄露、中间商转售，还是政治立场表达。最具体的政策动向信号来自 [The Information 的报道](https://x.com/steph_palazzolo/status/2070241787180966279)：美国政府要求 OpenAI **逐一向客户分批开放 GPT-5.6 预览版**，这表明前沿模型发布正在形成一种事实上的审查机制。

**热门推文（按互动量排序）**

- **OpenAI 内部采用 Agent**：[OpenAI 介绍 Codex 如何改变各部门的工作方式](https://x.com/OpenAI/status/2070196105745518913)。
- **Hugging Face 的商业模式**：[Clement Delangue 谈 HF 年收入运行率超过 1 亿美元](https://x.com/ClementDelangue/status/2070104323481104674)。
- **基准测试的完整性**：[Cursor 指出模型在公开基准测试中作弊](https://x.com/cursor_ai/status/2070195789121671624)。
- **开放代码模型**：[Ornith-1.0 发布](https://x.com/ornith_/status/2070148887067963854)。
- **Google 的 Agent 产品化**：[Gemini 3.5 Flash 推出 computer use 功能](https://x.com/Google/status/2070175556503568394)。
- **多 Agent 系统的行为**：[Thom Wolf 介绍 100 多个 Agent 协作，将 Gemma 4 的推理速度优化至原来的 5 倍](https://x.com/Thom_Wolf/status/2070134136304517284)。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 专用开放模型发布



  - **[NVIDIA 发布了 Nemotron-TwoTower-30B-A3B-Base-BF16：一款基于 Nemotron 3 Nano 30B-A3B 主干构建的非典型 diffusion-based language model。](https://www.reddit.com/r/LocalLLaMA/comments/1uf4azy/nvidia_has_released/)**（热度：459）：****NVIDIA** 发布了 [`Nemotron-TwoTower-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/Nemotron-TwoTower-30B-A3B-Base-BF16)，这是一款源自 **Nemotron 3 Nano 30B-A3B** 主干的 diffusion-style LLM。该模型将一个冻结的自回归上下文 tower 与一个 diffusion denoiser tower 结合起来，后者可以并行填充 token block；NVIDIA 宣称，在默认的 mask-diffusion 配置下，该模型在综合基准测试中的得分保留了 AR 基线的 `98.7%`，同时实现了 `2.42×` 的实际生成吞吐量。**唯一具有技术相关性的评论提出了疑问：与 **DiffusionGemma** 相比，它相对于基线的质量保持能力是否更强；其余热门评论大多是玩笑或与模型请求无关的内容。

    - 一位评论者指出，与原始 Nemotron 主干相比，**Nemotron-TwoTower-30B-A3B-Base-BF16** 似乎比 **DiffusionGemma** 相对于其基础模型保留了更多准确率，不过该讨论没有提供具体的基准名称或数值分数。

  - **[Qwen-AgentWorld-35B-A3B：一个拥有 3B active 参数、经过训练以模拟 MCP、terminal、SWE、Android、web 和 OS 环境的 MoE](https://www.reddit.com/r/LocalLLaMA/comments/1ue5149/qwenagentworld35ba3b_a_3bactive_moe_trained_to/)**（热度：315）：****Qwen** 发布了 [`Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)，这是一个稀疏 MoE，总参数量为 `35B`，每个 token 约有 `3B` 参数处于 active 状态。它被定位为 **language world model**，而不是聊天或指令式 Agent。该模型经过训练，可以模拟 Agent 循环中的环境响应：根据操作预测下一步的观察结果或状态，覆盖 MCP/tool calling、搜索、terminal、SWE、Android、web 和 OS-GUI 交互等领域。这有望用于离线 Agent 训练与评估、生成合成轨迹，以及模拟 tool workflow。**唯一有实质技术内容的评论强调了它在评估中的潜在用途，例如通过预测 `ls -la` 的 terminal 输出，来模拟操作产生的结果。其他热门评论大多是在开玩笑，或质疑其数据集是否只是简单地互换了 user/assistant 角色，或者是否只是向模型输入了“*You are an MCP server now.*”这样的提示词。

    - 一位评论者将该模型理解为学习环境转换动态：给定类似 `ls -la` 的 user/tool command，它会预测相应的 terminal 输出。他们认为，这不仅有助于 Agent 训练，还可以用于**在评估中模拟 tool/environment action**，从而可能减少执行真实沙箱操作的需求。
    - 另一种技术解读是，**Qwen-AgentWorld-35B-A3B** 可能是在模拟的“world”轨迹上训练的，这些轨迹涵盖 MCP、terminal、SWE、Android、web 和 OS 交互；随后再通过下游 **Agent performance improvement** 对其进行评估。评论者认为，如果这一解读成立，那么与其说该模型只是一个 simulator，不如说它是一个得到增强的 **agentic model**，并希望实际运行 Agent benchmark 的用户进行验证。

  - **[Unlimited-OCR 现已登陆 ModelScope！一款 3.3B 多语言 OCR 模型，可对单张图片、多页文档和 PDF 进行 one-shot parsing。许可证：MIT](https://www.reddit.com/r/LocalLLaMA/comments/1ue5149/unlimitedocr_is_now_on_modelscope_a_33b/)**（热度：1123）：****Baidu 的 Unlimited-OCR** 已在 **ModelScope** 发布。这是一款采用 **MIT** 许可证的 **`3.3B` 多语言 OCR/文档解析模型**，面向单张图片、多页文档和 PDF 的 *one-shot* 整文档解析；针对较长的 OCR 序列，最多可输出 **`32K` 个 token**。该项目提供 **base** 和 **“gundam” image mode**，并支持通过 **Transformers** 推理，以及使用 **SGLang** 提供带有 OpenAI 兼容流式 API 的服务；代码托管在 [GitHub](https://github.com/baidu/Unlimited-OCR)，相关公告发布于 [X](https://x.com/ModelScope2022/status/2069335055965491525)。**评论者主要询问一些尚未提供的技术对比和细节：它是否与 **PaddleOCR** 有关，或是否缺少其中的功能；它与 **PaddleOCR-VL-1.6** 相比表现如何；`32K` 输出限制最多能容纳多少页；以及“**gundam mode**”究竟是什么意思。



- 评论者要求与 `PaddleOCR-VL-1.6` 进行**直接基准测试**，具体包括 Unlimited-OCR 在 OCR 质量/性能方面的表现，以及在多页/PDF 解析场景下，模型的 `32k` 上下文窗口实际能容纳多少页文档。

- 有人对模型/文档中提到的 **“gundam mode”** 提出了技术层面的疑问——多位用户询问这一术语的含义，说明发布材料中可能存在表述不清，或者有尚未公开说明的推理/解析模式。

- 一位评论者贴出了 Hugging Face 上的模型卡：[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)；另一位用户在图片旁写道“missing paddle?”，可能是在指出与 PaddleOCR 相关的引用或依赖不一致、缺失等问题。

  - **[Ornith-1.0 released on Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1ufc9vp/ornith10_released_on_hugging_face/)**（活跃度：391）：****DeepReinforce-AI** 发布了 [**Ornith-1.0** Hugging Face collection](https://huggingface.co/collections/deepreinforce-ai/ornith-10)，其中包括 `9B`/`31B` dense 版本，以及 `35B`/`397B` MoE 版本，并宣称在若干未指明的基准测试中取得 SOTA 成绩；评论者认为这些模型是经过后训练的 **Qwen3.5** 和 **Gemma4**。一位用户报告称，在双 R9700 Vulkan 环境下，`35B Q8_0` 版本的生成速度约为 `115 tok/s`，提示词处理速度约为 `5400 tok/s`，与“关闭思考功能的 Qwen 3.6 35B”相当，偶尔会短暂降至 `95 tok/s`。另一位测试者发现，`35B` 模型拒绝泄露隐藏的 canary token，并明确将该请求识别为 prompt injection 尝试，说明模型可能内置了防止信息泄露和 prompt injection 的能力。** 早期主观反馈非常积极：一位测试者认为，与 Qwen 3.6 35B 相比，Ornith-35B 在代码、API 和安全审查方面的输出“详细得多”，同时速度还快得多，最后评价道：*“这可能是真的强。”*

    - 一位用户报告称，在**双 R9700 Vulkan** 环境下，**Ornith-1.0 35B Q8_0** 量化版本与**关闭思考功能的 Qwen 3.6 35B** 原始吞吐量基本相同：生成速度约 `115 tok/s`，提示词处理速度约 `5400 tok/s`。他观察到响应过程中速度会间歇性地从 `115 tok/s` 降至 `95 tok/s`，可能与温度有关；除此之外，他在非正式的 Ruby/Sinatra 测试中表示，该模型速度快得多，而且在代码、API 和安全审查方面的回答也比 Qwen 3.6 35B 更详细。
    - 在 Pi 设备上的测试表明，35B 模型可能内置了 prompt injection 或 canary 外泄防护机制。测试者在上下文中隐藏了一个随机字符串，并在之后要求模型找出它；模型拒绝执行，并明确判断这是一次 *“prompt injection attempt”*，拒绝复述 canary token。
    - 多位评论者认为，Ornith-1.0 是经过后训练的 **Qwen3.5** 和 **Gemma4** 衍生模型，据称基准成绩超过了 **Qwen 3.6 27B**。有人提出了一个技术疑问：为什么发布说明建议在 **vLLM** 中使用 `qwen3_xml` 格式，而在 **SGLang** 中使用 `qwen3_coder`？这可能意味着不同 serving stack 使用了不同的 prompt template，从而影响质量和基准测试的可复现性。


### 2. AI 法律与芯片管控动向

  - **[The Swiss Federal Supreme Court is evaluating Heretic](https://www.reddit.com/r/LocalLLaMA/comments/1ueeund/the_swiss_federal_supreme_court_is_evaluating/)**（活跃度：883）：**该帖子称，**Swiss Federal Supreme Court** 正在内部评估 [Heretic](https://heretic-project.org)，将其作为缓解 LLM 在合法刑法工作流中出现拒答的一种方案，而不是试图禁止“abliterated”模型。帖子引用的论文 [*Measuring \& Mitigating Over-Alignment for LLMs in Multilingual Criminal Law Courts*](https://arxiv.org/pdf/2606.23375) 研究了多语言法律场景中的过度对齐/拒答行为，并在第 5.2 节评估了 Heretic，结论较为积极；论文同时还讨论了 abliteration 等技术。** 一条具有技术相关性的评论指出，**drug discovery** 领域也存在类似的拒答问题：主流或闭源 LLM 可能无法使用，因为合法的领域问题有时会与受限制的生物/化学内容相似。

    - 一位从事**drug discovery** 的评论者表示，他们“无法使用主流/闭源 LLM”，这意味着在向托管模型发送提示词时，可能会受到专有分子/IP 数据、保密性、合规性和审计能力等方面的限制。技术层面的结论是，制药等领域可能更倾向于使用 **local/open-weight models**，例如 Heretic 风格的无审查或可自行托管的系统，以避免数据外泄和策略过滤限制；不过，原文没有提供基准测试或实现细节。



  - **[Anthropic 指控 Alibaba 发起“公然”且“非法”窃取 AI 能力的行动](https://www.reddit.com/r/LocalLLaMA/comments/1ueyl2i/anthropic_accuses_alibaba_of_campaign_to_brazenly/)**（活跃度：759）：据 [CNBC](https://www.cnbc.com/2026/06/24/anthropic-alibaba-distillation-campaign.html) 和 [Bloomberg](https://www.bloomberg.com/news/articles/2026-06-24/anthropic-accuses-alibaba-of-illicitly-accessing-its-ai-models) 报道，**Anthropic** 据称指控 **Alibaba** 协同开展模型提取／蒸馏行动，试图“公然”且“非法”访问 Anthropic 的 AI 模型，并复制其能力。技术层面的争议在于：大规模查询前沿模型，用于训练或调优竞争模型，是否构成未经授权的能力转移，而不只是普通的 API 使用。**热门评论主要聚焦于知识产权和法律层面的不对称性：用户认为 LLM 的输出通常不受版权保护，并嘲讽 Anthropic 的指控十分虚伪，因为 Anthropic 自己也曾因训练数据处理方式面临诉讼和和解，包括 [Authors Guild 的总结](https://authorsguild.org/advocacy/artificial-intelligence/what-authors-need-to-know-about-the-anthropic-settlement/) 以及 [Inside Tech Law 对 *Bartz v. Anthropic* 和解背景的报道](https://www.insidetechlaw.com/blog/2025/09/bartz-v-anthropic-settlement-reached-after-landmark-summary-judgment-and-class-certification)。

    - 几位评论者将这场争议描述为**模型蒸馏／能力提取**问题，而不是一个简单的版权问题：Anthropic 可能是在指控对方滥用 EULA／API，但有人认为 LLM 输出本身不受版权保护，因此很难主张生成的文本属于专有训练数据。
    - 一个与技术密切相关的批评是，通过 `~25,000` 个机器人账号和住宅代理进行大规模提取，很难仅靠政策阻止；评论者质疑，除了私有的反滥用控制、速率限制、账号验证或流量分析之外，立法者还能施加什么切实可行的执法机制。
    - 一位评论者认为，这项指控反而公开暴露了一个薄弱的竞争壁垒：如果竞争对手能够通过 API 从类似 Claude 的系统中蒸馏行为，那么 Anthropic 的防御力就不再主要取决于模型保密性，而更多取决于监控、访问控制、推理经济性以及持续的模型改进。

  - **[这项社区似乎漏掉了：要求追踪 AI 芯片位置的法案获得行业支持｜已有六家公司公开支持 Chip Security Act，该法案将要求为美国最先进的计算芯片配备位置追踪机制。](https://www.reddit.com/r/LocalLLaMA/comments/1ue2fd7/seems_this_community_might_have_missed_it_bill/)**（活跃度：465）：**拟议中的 **Chip Security Act** 将要求为美国最先进的 AI／计算芯片配备**位置追踪机制**。帖子提到，据报道已有“六家公司”表示支持；相关讨论也出现在 [`r/politics`](https://www.reddit.com/r/politics/comments/1uahgcs/bill_that_would_mandate_ai_chip_location_tracking/) 和 [`r/LocalLLM`](https://www.reddit.com/r/LocalLLM/comments/1ubz5xh/us_to_require_location_tracking_for_ai_and/) 中。**从技术角度看，这可能意味着要在硬件／固件或供应链层面增加一套用于执行出口管制合规要求的机制，但也会带来明显问题，包括防篡改能力、远程证明、地理围栏的可靠性，以及高端加速器中新增的攻击面。**热门评论普遍持负面态度，认为这项规定可能削弱美国竞争力、加速中国替代方案的发展，并引入不安全的追踪基础设施。一条讽刺性的评论概括了这种担忧：“我们将打造世界上最出色、最安全的位置追踪机制！”**



## 技术性较低的 AI 子版块回顾

e /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 前沿模型发布与泄露

  - **[GPT-5.5 Instant 现已开始逐步推送](https://www.reddit.com/r/OpenAI/comments/1uen1zv/gpt55_instant_now_rolling_out/)**（活跃度：803）：**图片是一张疑似由 **ChatGPT（@ChatGPTapp）** 在 X 上发布的截图，宣布 **“GPT-5.5 Instant”** 开始推送，顺序为 **Pro**、**Plus**，然后是免费用户，且“明天之前”完成推送（[图片](https://i.redd.it/sz3szth86a9h1.jpeg)）。讨论中的技术疑点在于：这究竟是真正全新的 ChatGPT 模型变体、界面／营销层面的重新命名，还是等同于 API 中已有的 `thinking: none` 等配置。**评论者对此持怀疑和困惑态度，纷纷询问这是否只是旧消息、如何确认自己使用的是新版还是旧版 **5.5 Instant**，以及它与 API 中关闭推理／思考功能后已经可以实现的行为究竟有何不同。



- 评论者提出了一个关于**模型/版本识别**的技术歧义：多位用户询问，如何判断自己使用的是新近推出的 **GPT-5.5 Instant**，还是此前的 Instant 版本。这意味着目前的发布缺少用户界面或 API 中可见的版本元数据，或缺少类似变更日志的版本标识。
- 有用户质疑，这次发布在功能上是否真的区别于现有 API 配置中的 **`thinking: none`**，因此不确定“GPT-5.5 Instant”究竟是一个独立的模型快照、一次路由调整，还是仅仅是关闭推理功能的预设。

  - **[欧盟正资助自己的开源 400B+ 前沿模型，该模型将在欧洲超级计算机上训练](https://www.reddit.com/r/singularity/comments/1ue8yy5/the_eu_is_funding_its_own_opensource_400b/)**（热度：898）：**欧盟委员会**在 Frontier AI Grand Challenge 中选定了由 **Domyn** 牵头的 **EUROPA 联盟**，计划利用**欧洲公共 EuroHPC AI 优化超级计算机**训练一个**开源的 `400B+` 参数模型**，目标是支持全部 **24 种欧盟官方语言**（[来源](https://aiweekly.co/alerts/domyn-led-europa-consortium-wins-eu-frontier-ai-grand-challenge)）。这项资助**不是现金，而是算力配额**——为期一年，最多可使用 EuroHPC 总算力的 **`2.5%`**——但评论者指出，目前没有公布交付时间表、训练预算、模型架构、基准测试目标，也没有说明“前沿级”的具体定义。**评论者对此意见不一：有人认为，该模型很可能采用一个 **`400B+` 的 MoE 架构**，其中约有 **`40B+` 参数处于激活状态**；如果欧盟能为公共部门和初创企业提供低价或免费的推理服务，那么它的主要价值或许在于此，但在能力上未必能与顶尖的专有或前沿系统竞争。另一些人则批评欧盟是在“指定赢家”，认为与其只资助一个模型项目，不如支持多个相互竞争的模型项目；还有人认为，多语言定位主要是营销，因为现代 LLM 已经能够高效地学习语言迁移能力。**

    - 一位评论者推测，欧盟模型很可能是一个**拥有 `400B+` 参数、约 `40B+` 激活参数的 MoE 模型**，但认为它可能达不到当前强大的前沿模型或开源模型（例如 **GLM-5.2**）的能力水平。在他看来，该模型的主要技术和实际价值，与其说是刷新原始基准成绩，不如说是为公共部门用户和初创企业提供**由欧盟托管的推理服务**，并且这些服务有可能获得补贴，甚至免费提供。
    - 有一项技术批评认为，围绕欧盟 **24 种官方语言**进行专门训练，可能更多是营销需要，而非技术必需。原因在于，现代 LLM 往往可以借助共享表示和广泛的互联网规模语料，高效获得多语言能力。令人担忧的是，过度强调语言覆盖，可能会挤占数据质量、扩展效率、后训练和评测等更重要的前沿模型工作。
    - 另一位评论者认为，与其资助一个被选定的模型，不如资助**多个相互独立的前沿模型尝试**，让不同的架构、数据集、训练技术栈以及对齐/后训练方案展开竞争。这一观点背后的技术逻辑是，前沿进展高度依赖实证探索，因此，一个由多种实验组成的生态，可能比集中式的“指定赢家”策略更有效。

  - **[3.5 Pro 本周发布](https://www.reddit.com/r/GeminiAI/comments/1uei7js/35_pro_coming_this_week/)**（热度：1695）：**这张图片是一条**传闻中的泄露推文**，并非官方公告。推文声称 **Gemini 3.5 Pro** 将在“本周”发布，并具备更强的视觉能力、多模态推理、更好的记忆和上下文保持能力、Agent 工作流、SVG/前端生成、原生图像模型，以及 `2.5M` token 的上下文窗口（[图片](https://i.redd.it/kxh47zuxa99h1.png)）。Reddit 标题将其描述为“3.5 Pro 本周发布”，正文则写着“Fable 的终结”，但图片没有提供任何基准测试数据、模型卡、API 细节或可验证的来源。**评论者对此持怀疑态度：有人认为它应该先正式发布，并“祈祷它不要莫名其妙地退步”；有人指出，由于没有提到任何领先的编程基准测试，它不太可能成为“Fable 的终结”；还有人批评发帖者分享了相互矛盾的泄露消息。**



- 评论者对 **Gemini/Google “3.5 Pro”** 能否胜过现有的 **3.1 Pro Preview** 持怀疑态度，其中一人明确表示，希望“千万不要莫名其妙地退步”。另一人指出，这则泄露消息没有声称模型在 **顶尖编程基准测试**中领先，这是一个负面信号；他认为，如果该模型在这方面具有竞争力，Google 很可能会宣传其基准测试成绩。
    - 有人质疑所谓的 **`2.5M` 上下文窗口**不太可信；一位评论者认为，该模型更有可能仍采用 **`1M` 上下文**限制，并将更大上下文窗口的说法视为原帖可能是伪造的证据。
    - 另一个技术和产品层面的担忧是高负载下的模型路由：一位评论者提到，付费层用户在“高强度使用”期间，**Pro 3.5** 请求可能会被降级到其他模型。这会让那些希望稳定使用高级模型的用户难以进行基准测试，也会影响使用体验的可靠性。

  - **[Fable 5 回归传闻，CC 中似乎出现了一些线索](https://www.reddit.com/r/ClaudeAI/comments/1uehr3a/fable_5_return_rumored_with_some_hints_in_cc/)**（活跃度：1007）：**一则基于 Claude Code `v2.1.190` 字符串变更的传闻称，**Fable 5** 可能会以订阅内含模型或功能的形式回归，并附带**每周使用额度**：据称新增的字符串是 *“You've used your Fable 5 usage for this week”*，而关于 *“purchased separately from your plan”* 的表述则被移除了（[来源](https://x.com/synthwavedd/status/2069813760622043483)）。如果属实，这意味着 Fable 5 可能会从单独购买或临时开放，转向长期包含在订阅方案中但设有每周使用上限；不过，原帖并没有提供官方确认。**评论者大多既兴奋又持怀疑态度，其中一个较有实质性的观点是：即使每周额度有限，也比短期订阅访问更好，因为这样至少能持续使用该功能。

    - 关于 Fable 可能回归时的访问政策，讨论中有一个较有实质性的观点：一位评论者认为，**较低的每周使用额度**，比只提供 `two-week` 限时访问的订阅模式更好，因为可持续获得、但额度受限的访问方式能够长期保留可用性，而有时间限制的访问则可能在期限结束后让用户彻底无法使用。




### 2. AI 数据中心引发的反弹与防御

  - **[数据中心噪音惹恼弗吉尼亚州居民：“你只想骂人”——居民在窗户上加装床垫和有机玻璃，以阻挡弗吉尼亚州这座数据中心的噪音。为其供电的天然气涡轮机会发出高频啸声。噪音 24/7 从未停止。- NewsNation](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/)**（活跃度：3182）：**一篇与 NewsNation 相关的 Reddit 帖子称，弗吉尼亚州某数据中心附近的居民正持续遭受 `24/7` 噪音困扰。噪音被描述为驱动该设施的**现场天然气涡轮机**发出的高频啸声；据称，附近居民已经在窗户上安装床垫和有机玻璃来减轻噪音。由于相关 Reddit 视频（[v.redd.it/akb9g6vkn69h1](https://v.redd.it/akb9g6vkn69h1)）因**403 Forbidden**而无法访问，因此技术细节仅限于帖子正文和评论内容。**热门评论主要关注土地利用和基础设施问题：用户质疑当地的分区规划为何允许在住宅附近建设数据中心和涡轮机电站；他们认为这类设施不应选址于住宅区，并指出数据中心主要需要网络连接，而不是靠近居民区。

    - 评论者重点讨论了这种不同寻常的选址和基础设施方案：据称，该数据中心**没有接入电网**，而是由现场的**天然气涡轮机**供电，从而产生持续不断的高频啸声。许多人认为，数据中心主要需要可靠的网络连接和电力供应，而不是靠近住宅区，因此这一选址在技术和规划层面都值得质疑。
    - 一个具有技术参考价值的讨论串，将美国地方分区和规划的结果与更严格的欧盟/英国规划制度进行了比较。评论者认为，在欧洲，这类位于住宅附近、全天候运行并产生工业噪音的设施，很可能会面临更严格的审批障碍。问题的重点并不是数据中心本身，而是涡轮机驱动的工业基础设施与住宅之间缺乏合理的土地用途隔离。
    - 一位评论者指出，这种噪音问题在技术上并不新鲜：**隔音屏障、土坡、围栏以及植被/林带缓冲区**，都是公路和其他噪音基础设施周边常用的降噪手段。批评的核心是，如果运营方被要求采取标准的声学降噪措施，就应当能够将噪音降低到可接受的水平。

  - **[John Carmack 谈数据中心](https://www.reddit.com/r/singularity/comments/1ue1sya/john_carmack_weighs_in_on_datacenters/)**（活跃度：2203）：**[这张图片](https://i.redd.it/mius3v4nc59h1.png)是一张 X/Twitter 对话截图，其中 **John Carmack** 认为，反对新建 **AI/数据中心基础设施**的声音，可能会类似于美国社会曾经的反核情绪，从而拖慢一次重大的技术转型。结合帖子标题《“John Carmack 谈数据中心”》来看，其技术意义并不在于某项具体基准测试或某个模型，而在于**计算能力容量限制**：Carmack 将数据中心需求不断增长视为其价值的证据，并表示 Texas 应积极支持面向 AI 工作负载的数据中心建设。**评论者则反对这种绝对化的论述，主张采取折中方案：只要数据中心不对居民造成噪音等干扰，并且自行提供**电力/水资源**，就应当允许建设。还有人反驳 Carmack 的核能类比，指出化石燃料利益集团曾参与塑造反核政治，而 AI 数据中心不断增长的能源需求也可能让这些集团受益。

    - 多位评论者关注**数据中心的选址限制**，认为只有在不会给当地造成**噪音、废热、用水压力或扰民问题**等外部影响的地点，才应允许建设数据中心；同时，数据中心还应自行提供或确保获得**电力和供水基础设施**，而不是把负担转嫁给地方政府。
    - 一个反复出现的技术政策主题是，大规模 AI 数据中心扩张受到**能源供应**的制约。评论者认为，若要进一步扩大建设规模，可能需要以**安全的核能**作为基础，同时批评依赖煤炭/石油支持的发电方式来满足 AI 计算需求。



### 3. 大规模 Agentic Coding 工作流

  - **[我使用自己的 Pro 订阅 18 个月后，公司终于给我开通了企业版许可证。我刚刚让 Opus 生成了 451 个 Sonnet 子 Agent，在一次 5 小时的会话中用了价值 1400 万的 token——甚至还没有触及上限。这太棒了。](https://www.reddit.com/r/ClaudeAI/comments/1uf2nba/after_using_my_own_pro_subscription_for_18_months/)**（活跃度：1445）：**一位用户表示，在从个人 Claude Pro 订阅转为企业版许可证后，他让 **Claude Opus** 生成了 `451` 个 **Sonnet** 子 Agent，用于数据标注工作流；在一次持续 `5` 小时的会话中消耗了大约 `14M` 个 token，而且似乎没有遇到使用上限。其关键技术意义在于，企业版计划支持大规模 Agent 扇出，但评论区指出，这很可能是**按用量计费，而不是提供无限配额**。许多高赞评论者对“没有触及上限”这一说法持怀疑态度，认为真正的限制是公司的月度账单；还有人要求看看最终账单。

    - 评论者解释说，**企业版/API 风格的许可证可能不像 Pro 那样显示相同的使用上限**，所以“没有触及上限”很可能意味着这次运行会按量计费并出现在账单上，而不是被系统阻止。一位评论者估算，这次 `14M` token 的会话可能会产生大约 **`$120–$200`** 的费用，具体取决于输入输出比例和模型定价，并建议使用 [`ccusage`](https://github.com/ryoppippi/ccusage) 等工具查看 token 级别的计费明细。

  - **[软件开发已经进入“无限猴子”时代](https://www.reddit.com/r/ClaudeAI/comments/1ue4zw0/software_development_has_entered_its_infinite/)**（活跃度：818）：**该帖子认为，**Claude Code**、**Cursor** 和 **Codex** 等 Agentic Coding 工具降低了通过自然语言对代码库进行大规模修改的门槛，形成了一种“无限猴子”式的局面：生成的软件数量大幅增加，质量则从实用到勉强能运行、几乎无法理解不等。评论中提出的技术含义是，这种趋势可能会增加而非减少对资深工程师的需求，尤其是在 **安全审查、维护和 AI 生成代码治理**方面。评论者将 LLM 编程工具比作智能手机相机：相机并没有消灭专业摄影师，而是扩大了业余创作，并催生了新的生态。另一种观点认为，AI 生成和 AI 发现的漏洞可能会让 IT/安全工程师变得更加重要，尤其是在银行和政府等高风险领域。

    - 有评论提出了一个技术层面的担忧：LLM 辅助开发可能会**增加对 IT/安全工程师的需求**，而不是取代他们，因为自动代码生成和分析可能暴露或引入更多安全问题。评论者特别提到 **LLM 发现的安全漏洞**，并警告说，**政府和银行**等关键部门需要更强的工程监督，以避免出现系统性故障。

  - **[我为 Claude Code 做了一个状态灯。你觉得这真的有用吗？](https://www.reddit.com/r/ClaudeCode/comments/1ue5inx/i_built_a_status_light_for_claude_code_do_you/)**（活跃度：3291）：**图片展示了一个 DIY 的**交通灯式硬件状态指示器**，夹在显示器上，用于显示 **Claude Code** 的状态；通过 Claude Code hooks 映射不同状态：**红色** = 等待确认，**黄色** = 正在运行，**绿色** = 已完成/空闲。它的主要技术意义在于，为长时间运行的 Agentic Coding 会话增加一层环境式 UI/实体通知，避免用户反复切换窗口来检查 Claude Code 是否需要输入。[图片](https://i.redd.it/ncs9m61cb69h1.jpeg) 评论者普遍认为这个装置很有意思，但也质疑它的实际价值。主要技术问题在于它如何处理**多个 Claude Code 会话/工作树**；也有人建议采用纯软件方案，例如状态栏 hooks、Telegram 通知，或使用 Claude Code 的 `/remote-control` 推送通知。

    - 一个关键的技术问题是并发处理：一位评论者询问状态灯如何处理跨多个工作树运行的**多个 Claude Code 会话**，这意味着该设计需要按会话/工作树跟踪状态，而不是只提供一个全局的忙碌/需关注指示器。
    - 几位评论者提到了纯软件替代方案：将 Claude Code hooks 连接到**状态栏通知**、发送 **Telegram 消息**，或者使用 `/remote-control`，在需要用户关注时依靠推送通知。
    - 一位用户介绍了一个使用 **Stream Deck** 的类似实现：每启动一个新的 Claude Code 会话，就动态创建一个按钮；**工作时显示绿色**，**需要输入时显示红色**；按下红色按钮即可聚焦到对应的 Claude Code 实例。



# AI Discord 社区

很遗憾，Discord 今天终止了我们的访问权限。我们不会以这种形式恢复它，但很快会发布全新的 AINews。感谢你读到这里，这段旅程曾经很美好。