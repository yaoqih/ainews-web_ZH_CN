---
companies:
- openevidence
- anthropic
- podium
- openai
- google
- gemini
date: '2026-01-21T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **OpenEvidence** 融资 **120 亿美元**，较去年增长了 12 倍；目前全美有 40% 的医生在使用该平台，其年收入已超过 1 亿美元。**Anthropic**
  根据 **CC0 1.0** 协议发布了新的 **Claude** 模型宪法，将其定位为用于对齐和训练的动态文档。**Podium** 报告称，其 **10,000
  多个 AI 智能体**贡献了超过 **1 亿美元的年度经常性收入 (ARR)**，标志着公司正从软件销售向 AI 运营商转型。在智能体记忆和可靠性方面的创新包括
  **智能体认知压缩器 (ACC)** 以及通过 **MCP-SIM** 实现的多智能体科学工作流。智能体基准测试显示，在处理长周期任务时仍面临挑战，**Gemini
  3 Flash High**、**GPT-5.2 High** 和 **Claude Opus 4.5 High** 等模型在专业服务和法律研究基准测试中的得分表现平平。'
id: MjAyNi0w
models:
- claude
- claude-3
- claude-opus
- gpt-5.2
- gemini-3-flash-high
people:
- daniel_nadler
- amanda_askell
- eric_rea
- tom_loverro
- garry_tan
- omarsar0
- brendanfoody
- deredleritt3r
title: OpenEvidence（被称为“医生版 ChatGPT”）以 120 亿美元的估值融资 2.5 亿美元，较去年 2 月 10 亿美元的估值增长了 12
  倍。
topics:
- agentic-ai
- model-alignment
- performance-evaluation
- memory-optimization
- long-context
- benchmarking
- multi-agent-systems
- reinforcement-learning
---

**Agent Labs 就够了**

> 2026年1月20日至1月21日的 AI 新闻。我们为你查看了 12 个 subreddit、[**544** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord（**205** 个频道，**7561** 条消息）。预计节省阅读时间（以 200wpm 计算）：**584 分钟**。**我们的新网站**现已上线，支持完整的元数据搜索，并以美观的 Vibe Coded 方式呈现过往所有内容。请访问 https://news.smol.ai/ 查看完整的新闻分析，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

我们有一个不成文的规定，即新的超级独角兽（decacorn）融资会成为头条故事。[多家来源](https://www.cnbc.com/2026/01/21/openevidence-chatgpt-for-doctors-doubles-valuation-to-12-billion.html) 报道了 OpenEvidence 120 亿美元的融资消息，较去年增长了 12 倍。奇怪的是，“CEO Daniel Nadler 告诉 CNBC，全美有 40% 的医生在使用 OpenEvidence，去年年收入突破了 1 亿美元”，这意味着 120 倍的估值倍数。

---

# AI Twitter 综述

**前沿模型治理：Anthropic 的新 Claude 宪法（CC0）及各方反应**

- **发布内容**：Anthropic 发布了一份新的“宪法”，描述了所期望的 Claude 行为/价值观，并声明其直接用于训练；重要的是，完整宪法以 **CC0 1.0** 协议发布，以鼓励重复使用/改编（[公告](https://twitter.com/AnthropicAI/status/2014005798691877083)，[CC0 链接](https://twitter.com/AnthropicAI/status/2014005815376568780)）。Anthropic 将其定位为由内部和外部专家共同塑造的**动态文档**（[后续](https://twitter.com/AnthropicAI/status/2014005813157720283)）。
- **社区观点**：Amanda Askell 强调这仍是一项正在进行中的工作，并邀请各界提供反馈（[Askell](https://twitter.com/AmandaAskell/status/2014010171081581048)）。其他人则指出了训练一个关于“模型应该如何”的文档这种“Meta”式的古怪感（[scaling01 关于 Opus 反思循环性的观点](https://twitter.com/scaling01/status/2014014692004421653)）。一些反应集中在宪法究竟是“对齐信号（alignment signaling）”还是实际的损害减轻，以及它是否固化了公关导向的人设行为（[nearcyan](https://twitter.com/nearcyan/status/2014009518150054035)，[NickEMoran 的提问](https://twitter.com/NickEMoran/status/2014077605373260204)）。
- **实际工程后果**：Anthropic 还发布了一项关于内部 **Performance Engineering 家庭作业（take-home）** 的消息，该作业现在可以被 Opus 4.5 完美解决，迫使他们重新设计招聘评估——这是“模型追赶上我们的筛选任务”的一个具体案例（[Anthropic Eng](https://twitter.com/AnthropicAI/status/2014143403144200234)，[trishume](https://twitter.com/trishume/status/2014144787092562160)）。

**生产环境中的 Agent：从“AI 员工”（Podium）到 Agent UX、Memory 和 Eval**

- **Podium 将其“Jerry”定位为一个 Agent 业务单元**：Podium 声称其 **AI agent ARR**（年度经常性收入）超过 **1 亿美元**，已部署超过 **1 万个 Agent**，重点解决 SMB（中小企业）的人员配置限制问题（如非办公时间的线索跟进、漏接电话、人员流动）。其叙事逻辑是：停止销售“软件”，转而销售一个能够端到端使用现有产品的 **AI operator**（[Eric Rea](https://twitter.com/ericwilliamrea/status/2013980401635582277)）。Tom Loverro 补充了董事会层面的指标（资金消耗从 **9500 万美元降至 0**，AI ARR 在约 21 个月内从 **0 增至 1 亿美元**），并链接到了 OpenAI 的案例研究（[Tom Loverro](https://twitter.com/tomloverro/status/2014011044210106406), [Garry Tan](https://twitter.com/garrytan/status/2014005103728943566)）。
- **记忆力与长周期（long-horizon）可靠性成为瓶颈**：
  - **Agent Cognitive Compressor (ACC)** 的观点认为“更多上下文不等于更好的 Agent”，并批评了对话记录回放和简单的检索方式。ACC 通过 Schema 约束的提交来维持一个有界的“压缩认知状态（Compressed Cognitive State）”，声称在长时间运行中具有更低的漂移和幻觉风险（[dair_ai](https://twitter.com/dair_ai/status/2014000799245107339)）。
  - 另一条线索通过 **MCP-SIM** 将“自我改进的多智能体（self-improving multi-agents）”应用于科学工作流。这是一个多智能体循环，能够澄清描述不全的物理提示词、生成代码、执行、诊断错误并生成多语言解释；声称解决了 **12/12** 的任务，而 one-shot GPT 的表现为 **6/12**（[omarsar0](https://twitter.com/omarsar0/status/2013998285040836662)）。
- **Agent 基准测试超越了编程难题**：
  - **APEX-Agents** 评估了 Google Workspace 中的长周期“专业服务”任务；早期的 Pass@1 排行榜数据较低（Gemini 3 Flash High 为 **24.0%**，GPT-5.2 High 为 **23.0%**，Claude Opus 4.5 High 为 **18.4%**）——这提醒人们“Agent 自主性”依然脆弱（[BrendanFoody](https://twitter.com/BrendanFoody/status/2014028956752568356)）。
  - **prinzbench** 引入了一个针对法律研究 + 搜索的私有基准测试（33 个问题，人工评分，运行 3 次），其中“搜索”是主要的失败模式；声称：GPT-5.2 Thinking 勉强超过 **50%**，Gemini 模型紧随其后，而 Sonnet/Opus 4.5 在搜索上的得分为 **0/24**（[deredleritt3r](https://twitter.com/deredleritt3r/status/2013979845378580684)）。
- **工具链 + UX 层正在迎头赶上**：多篇帖子达成共识，即 Agent 既需要更好的模型，也需要“上下文层（context layer）”和生产脚手架（治理、鉴权、可观测性）——参见下文的 Prefect Horizon 和 MCP server 最佳实践。


**Agent 平台与“上下文层”：MCP, Skills, Prefect Horizon, LangChain Deep Agents**

- **Prefect Horizon：从 MCP 到平台**：Prefect 将“上下文层”定位为 Agent 与企业级工具/数据之间的接口，并认为 MCP 描述了如何构建服务器，但没有解决如何在组织规模内进行 **部署/治理**。Horizon 声称提供托管部署、注册表/目录、带有 RBAC + 审计日志的网关，以及“面向业务用户的 Agent 接口”（[jlowin](https://twitter.com/jlowin/status/2014023606380957754)）。
- **MCP servers：设计指南**：Phil Schmid 反驳了“Skills 取代 MCP”的观点：MCP 本身不是问题，**糟糕的服务器**才是。建议：围绕结果设计工具，使用带有约束的有类型扁平参数，将 docstrings/errors 作为 Agent 指令；将 Skills 和 MCP 视为互补关系（[philschmid](https://twitter.com/_philschmid/status/2014016583706829054)）。
- **LangChain deepagents：Agent 即文件夹 + UI 集成**：
  - CopilotKit 发布了一个构建 **全栈 Deep Agent 应用** 的教程（简历摄入 → 技能提取 → 带有网络搜索的子 Agent → 流式 UI），填补了“缺失的 UI/应用层”空白（[CopilotKit](https://twitter.com/CopilotKit/status/2013997128683856159)）。
  - LangChain 发布了 **Agent Builder GA** 版以及与领域合作伙伴（Tavily, PagerDuty, Box 等）共同推出的 **模板库**，以减少从“提示词到 Agent”的摩擦（[LangChain](https://twitter.com/LangChain/status/2014034320256880768)）。
  - Deep Agents 的框架理念“Agent 只是文件夹”强调了可移植性/分发：你可以通过 CLI 流程快速打包、下载并运行 Agent（[hwchase17](https://twitter.com/hwchase17/status/2014076509208629386), [Vtrivedy10 demo](https://twitter.com/Vtrivedy10/status/2014074890458980736), [LangChain_OSS](https://twitter.com/LangChain_OSS/status/2014075587137048882)）。Sydney Runkle 强调了两个核心模式：**用于上下文隔离的子 Agent** 和 **仅在相关时加载的技能**（[sydneyrunkle](https://twitter.com/sydneyrunkle/status/2014060287746265535)）。
- **LangSmith + 分析**：一个线索指出 LangSmith Trace 不仅可以用于调试，还可以作为 **产品分析** 的基石（“Agent Trace → 产品分析”）（[SoftwareWatcher](https://twitter.com/SoftwareWatcher/status/2013972269106684060)）。


**推理 + 系统：低 VRAM 服务、开放推理栈以及“推理是主战场”**

- **AirLLM: 针对极小 VRAM 的分层流式传输 (layer streaming)**：AirLLM 的核心理念是**顺序层加载**（加载 → 计算 → 释放），具有可选的压缩功能、类似 HF 的 API，支持 CPU/GPU 以及 Linux/macOS；声称能让极大模型在极低 VRAM 上运行 ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014005554948047122), [repo](https://twitter.com/LiorOnAI/status/2014005556369826212))。工程师应将“8GB 运行 405B”的说法视为“在大量分页 (paging) 的情况下原则上可行”，但需预期吞吐量/延迟限制以及不容忽视的工程陷阱。
- **“真正的开放 AI”需要模型 + 推理引擎**：Modal 认为生态系统现在已具备构建块——能力强的开源模型加上快速且可调优的 OSS 推理——并分享了他们用于大规模服务的生产模式/堆栈 ([charles_irl](https://twitter.com/charles_irl/status/2014005582093832202))。
- **推理 Bug + 本地堆栈**：llama.cpp 修复了一个影响 **GLM 4.7 Flash GGUF** 的路由/函数问题，且配置更新中提到了 `scoring_func: sigmoid`；此外还展示了通过 Unsloth 工作流使用量化 GLM 构建小型游戏 ([danielhanchen](https://twitter.com/danielhanchen/status/2013974463856181689))。还有关于 GLM KV-cache 内存行为以及框架是否缺失基于 LoRA 方法的讨论 ([TheZachMueller](https://twitter.com/TheZachMueller/status/2014011037025001577))。
- **基础设施规范对 Agent 至关重要**：“快速验证使每个 Agent 更有效”（pre-commit 钩子、文档化的环境变量、减少 CI 等待时间）实际上是一种“针对 Agent 生产力的软件供应链”论点 ([matanSF](https://twitter.com/matanSF/status/2014039273721213256))。
- **研究方向：恒定计算上下文**：一个推特串总结了 NVIDIA 的 “TTT-E2E” 概念（将上下文视为数据并在线更新权重），将其作为一种在长上下文中保持延迟恒定的方法，但其“大海捞针” (needle-in-haystack) 召回率较低——这与需要精确编辑的 Agent 工作负载相关 ([sdrzn](https://twitter.com/sdrzn/status/2014128642503381276))。
- **硬件瓶颈框架**：一个反复出现的主题是“智能 → 推理”的转变以及计算/内存供应链的重要性 ([saranormous](https://twitter.com/saranormous/status/2014206109846806707))，这在一段关于 **HBM 资格认证周期**作为真实供应限制（而非“单纯增加晶圆厂”论调）的深度分析中得到了呼应 ([MarkosAAIG](https://twitter.com/MarkosAAIG/status/2014079461768003608))。


**代码生成很廉价；代码理解成为瓶颈 (Devin Review, Copilot CLI, Claude Code)**

- **Devin Review：审查用户体验，而非仅仅是纠错**：Cognition 推出了 **Devin Review**，将其定位为一种新的 PR 阅读界面，旨在减少“废料 (slop)”、按重要性重新排序 diff、识别重复/复制的代码、添加聊天层并与 GitHub 评论集成。可通过 URL 替换（`github` → `devinreview`）或 `npx` CLI 访问 ([launch](https://twitter.com/cognition/status/2014079905755955592), [usage modes](https://twitter.com/cognition/status/2014079917139566990), [URL tip](https://twitter.com/cognition/status/2014113266788667571))。多位测试者报告称，它甚至能捕捉到直接 diff 之外的问题 ([mcparadip](https://twitter.com/mcparadip/status/2014093822704202002), [BraceSproul](https://twitter.com/BraceSproul/status/2014089228951625979))。
- **核心观点：生成 vs 验证**：多条推文明确指出瓶颈已从编写转向**审查/理解/测试**，下一代 SWE 工具应加速人类的理解循环，而不仅仅是运行一个“保持距离的 Agent” ([walden_yan](https://twitter.com/walden_yan/status/2014085360826089852), [ScottWu46](https://twitter.com/ScottWu46/status/2014094461505339651), [theodormarcu](https://twitter.com/theodormarcu/status/2014102090520600613))。
- **CLI Agent 进化**：GitHub Copilot CLI 添加了 `askUserQuestionTool` 来询问澄清性问题（例如：混乱的 rebase），这标志着交互式工具化 CLI Copilot 胜过纯自动补全的趋势 ([Evan Boyle](https://twitter.com/_Evan_Boyle/status/2014012076881064173))。
- **Claude Code 采用轶事**：创始人越来越多地报告使用 Claude Code 后“2 人的团队像 10 人一样开发” ([alexalbert__](https://twitter.com/alexalbert__/status/2014047943234560234))。但也存在摩擦：技能重新加载行为感觉比简单的 “CLAUDE.md 重新读取”流程更退步 ([corbtt](https://twitter.com/corbtt/status/2014037092452671619))。一个特别具代表性的“多 Agent 扩张”故事描述了将 Claude Code 实例扩展为一个具有治理失败的准社会——这基本上是一个关于 Agent 编排债 (orchestration debt) 的轶事 ([voooooogel](https://twitter.com/voooooogel/status/2014189072647078053))。


**视频 + 多模态：评估、模型发布和检索扩展**

- **视频评估基础设施**：**Video Arena** 现已上线网页端，支持约 15 个前沿视频模型的两两对决生成，并通过社区投票驱动排行榜 ([arena](https://twitter.com/arena/status/2014035528979747135))。
- **模型发布**：Runway 的 **Gen-4.5 Image→Video** 发布，强调一致性与叙事性；早期采用者强调“故事构建”是视频模型最佳的评估方法论 ([runwayml](https://twitter.com/runwayml/status/2014090404769976744), [c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014105905088856411))。
- **开源语音系统**：Qwen 强调了在 **Chroma 1.0** 中的应用，该系统被描述为一个完全开源的实时语音系统 ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2013997092814139744))。
- **检索时缩放 / 延迟交互 (Late Interaction)**：多条讨论指出 **ColBERT 风格的多向量检索** 能够保留细粒度的意图，并可击败规模大得多的 Embedding 模型；Mixedbread 声称其 **17M** 参数的开源 ColBERT 模型在 LongEmbed 上击败了 **8B** 参数的 Embedding 模型，且他们正在以 **<50ms p50** 的速度服务 **10 亿+ 文档** ([mixedbreadai claim](https://twitter.com/mixedbreadai/status/2014062123358548017), [prod numbers](https://twitter.com/mixedbreadai/status/2014062110993687002))。TurboPuffer 同样在推进极致规模的 ANN（“索引整个 Web 的 100B+ 向量”）([turbopuffer](https://twitter.com/turbopuffer/status/2014063666262688191))。大趋势：检索正在从“每个文档一个向量”向** Token 级 / 多向量**系统转变，但这需要深度的基础设施协同设计。


**热门推文（按互动量排序）**

- **Gemini 在教育领域的应用**：Google 在 Gemini 应用中推出了**完整、点播式的 SAT 模拟考试**（与 **The Princeton Review** 合作），并提供即时反馈 ([Google](https://twitter.com/Google/status/2014020819173687626), [Sundar Pichai](https://twitter.com/sundarpichai/status/2014067664503668873))。Google 还宣布了 **Gemini × Khan Academy** 的合作伙伴关系，首个项目是“写作教练（Writing Coach）”，旨在引导起草和完善，而非直接生成最终答案 ([Google](https://twitter.com/Google/status/2014082428957045007))。
- **Claude “宪法”公开**：Anthropic 发布了直接用于 Claude 训练的新宪法；全文以 **CC0 1.0** 协议发布 ([Anthropic](https://twitter.com/AnthropicAI/status/2014005798691877083), [CC0 release](https://twitter.com/AnthropicAI/status/2014005815376568780), [Amanda Askell](https://twitter.com/AmandaAskell/status/2014010171081581048))。
- **AirLLM：极致低显存（VRAM）推理**：声称通过逐层加载，可以在 **4GB VRAM 上运行 70B 模型**，甚至在 **8GB VRAM 上运行 405B Llama 3.1**；已提供代码库链接 ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014005554948047122), [repo](https://twitter.com/LiorOnAI/status/2014005556369826212))。
- **Agent 成为真正的商业业务**：Podium 报告其 **AI Agent 在不到 24 个月内实现了超过 1 亿美元的 ARR**，“10,000 多个 Agent 已在生产环境中运行”，将 Agent 定义为“AI 员工”（Jerry）而非聊天机器人 ([Eric Rea](https://twitter.com/ericwilliamrea/status/2013980401635582277), [Garry Tan](https://twitter.com/garrytan/status/2014005103728943566))。
- **Runway Gen-4.5 图生视频**：Runway 推出了 **Gen-4.5 Image→Video**，强调更长的故事篇幅、摄像机控制和叙事连贯性 ([runwayml](https://twitter.com/runwayml/status/2014090404769976744), [c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014094466269794663))。
- **OpenAI 产品/UI 及组织变革**：ChatGPT Atlas 添加了**标签页组（tab groups）** ([OpenAI](https://twitter.com/OpenAI/status/2014095512874655867))；据 The Information 报道，OpenAI 进行了组织架构调整，包括任命企业、商业和广告业务负责人 ([Steph Palazzolo](https://twitter.com/steph_palazzolo/status/2014100920435462424))。

---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

待完成

## 技术性较低的 AI 子版块总结

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

待完成

---

# AI Discord 总结

> 由 gpt-5.2 生成的总结之总结的总结


**1. 推理工具链面临现实挑战 (GLM-4.7-Flash, llama.cpp, vLLM, Ollama)**

- **Flash Attention 在 GLM-4.7-Flash 中表现不佳**：多个社区报告 **GLM-4.7-Flash** 出现性能退化，**Flash Attention** 触发了 CPU 回退/错误，且吞吐量极低（在 LM Studio 中降至 **2.8 t/s**），建议在 [llama.cpp PR #18953](https://github.com/ggml-org/llama.cpp/pull/18953) 全面落地前禁用 FA。
  - 在 llama.cpp 修复后，模型已**重新上传**，用户被告知需**重新下载**并参考 [Z.ai 的 GLM-4.7-Flash-GGUF 模型卡参数](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)，据反馈配置正确后输出效果应会“好得多”。

- ****Ollama vs GGUF: 模板兼容性博弈****：用户发现某些 **GGUF quants** 在 **Ollama** 中无法运行，原因是 **chat template 不兼容**。Unsloth 团队反复建议在支持完善之前，最好坚持使用 [官方 Ollama 模型](https://ollama.com/)。
  - 潜台词是操作层面的：*“支持这些功能需要时间”*——因此，在生态系统跨推理引擎稳定之前，务实的做法是统一使用官方产物。

- ****vLLM 更新化解危机（这次是这样）****：在 Unsloth 的帮助聊天中，至少有一个棘手问题在 **vLLM 更新**后消失了，引发了诸如 *“噢，伙计，原来那是问题所在”* 之类的感叹。
  - 随后的建议是流程层面的：考虑 **固定依赖版本 (pinning dependency versions)**，以免未来的上游更新在周中随机搞垮流水线。


**2. 评估平台与产品发布 (LMArena + 多模态可靠性)**

- ****Video Arena 发布……每日限额 3 次****：LMArena 正式发布了 **Video Arena**，访问地址为 [lmarena.ai/?chat-modality=video](https://lmarena.ai/?chat-modality=video)，硬性限制为 **每 24 小时 3 次生成**，且仅限 **Battle-mode（对战模式）**（无法直接选择特定模型）。
  - 用户喜欢“视频功能上线”，但抱怨这种*老虎机式 (slot machine)* 的用户体验阻碍了受控测试——尤其是当你试图复现某个 prompt 或模型行为时，这非常痛苦。

- ****500 万次投票：停不下来的基准测试****：LMArena 的 **Text Arena** 突破了 **500 万次社区投票**，并在 [其里程碑社交视频](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4) 中进行了展示。
  - 工程师们将其框架化为“大规模真实世界 A/B 测试”，这正日益塑造着对模型的认知，即使正式基准测试的差异看起来很小。

- ****Gemini 3 Pro 图像预览与 Nano Banana Pro：设计性不稳定？****：LMArena 用户报告 **Gemini 3 Pro Image Preview** 极不稳定，且 **Nano Banana Pro** 频繁出现 *“Something went wrong”* 崩溃，怀疑是 **Google 端**的问题，有时甚至持续 **6 小时以上**。
  - 社区的抱怨点在于：尽管不可靠，但这些模型被认为是唯一能持续达到某些特定 prompt 目标的模型——因此，人们边忍受停机和错误边继续使用它们。


**3. Agent 与开发工具：MCP, Cursor, DSPy RLM, 以及编程助手扩张**

- ****MCP Inspector 无法重新认证 (401 = Game Over)****：MCP 贡献者发现 **MCP Inspector** 在遇到 **401 错误** 时无法重新认证，原因是 SDK 中存在一个关于跨重定向持久化 `resourceMetadata` 的 bug，详见 [inspector issue #576 的评论](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454)。
  - 目前的权宜之计虽然笨拙但很明确：依赖 **VS Code** 进行初始连接，因为 Inspector 路径目前还无法在会话中途干净地恢复。

- ****RLM 与编程 Agent：视野问题 (Horizon Problem)****：DSPy 的讨论将 **RLM** 与“编程 Agent”进行了对比，认为 RLM 可以通过代码和符号调用将输入/输出/视野 (horizon) 外部化（参见 [引用的 X 推文](https://x.com/lateinteraction/status/2013658521246535892)）。
  - 实践意义：团队希望得到符号如何被访问的图表，并讨论是否应该给 RLM 提供 ripgrep/语义搜索等工具，还是让它们编写自己的搜索代码。

- ****Cursor 的 MCP/扩展时刻（以及定价波动）****：Cursor 用户讨论了使用 [Playwright MCP](https://playwright.dev/) 进行测试（在 TDD 流程中效果参差不齐），并得出结论认为扩展构建应该镜像 **VS Code** 的功能。
  - 与此同时，用户注意到 **500 次请求计划** 已取消（**2025 年 9 月**停止），因此选择新的定价方案将失去退出宽限期——让“试用”变成了“承诺”。


**4. GPU/内核工程变得异常内卷**

- ****Anthropic 的性能测试作业变成了一项运动****：GPU MODE 和 tinygrad 的成员们对 Anthropic 的 [original_performance_takehome](https://github.com/anthropics/original_performance_takehome/) 进行了花式解题，分享了如社区成员跑出的 **2200 周期 (cycles)** 以及 **Claude Opus 4.5** 在一次随意的 Claude Code 会话中跑出的 **1790 周期**。
  - tinygrad 用户甚至讨论了通过为一个玩具级 **VLIW 机器** 添加后端来解决它，引用了特定的参数如 `PCONTIG=2`、`UPCAST` 和 `DEVECTORIZE=2` 来保留向量指令并实现高效调度。

- ****Torch 维护者被 AI 生成的 PR 淹没****：GPU MODE 的 torch 频道描述了大量低质量 **AI 生成的 Pull Requests** 的涌入，迫使维护者考虑对新贡献者设置门槛，并在人工介入前自动化分类。
  - 人们提议使用像 **Cursor Bugbot** ([Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq)) 甚至分类器工具（例如“先过一遍 Claude/Pangram”）作为代码审查带宽的*最低标准*。

- ****内核数学难题：Triton 错误 + Cute 布局代数****：GPU MODE 用户调试了一个自定义 **Triton 2D conv** 内核中的数值爆炸问题，该内核在特定形状下误差从 ~**1e-6** 飙升至 ~**1e-2**（参见 [Pastebin 复现](https://pastebin.com/2ejn2QW2)），并讨论了 Blackwell 特性的利用。
  - 另外，一项对 **Cute** 布局代数的深度探究引导工程师们阅读了一篇图解微积分文章——[Cute 布局的范畴论基础](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/)，该文章认为编写高质量内核需要具备“布局代数素养”。


**5. 计算经济学与基础设施业务动态 (Runpod, GPU 市场, 模型定价)**

- ****Runpod 达到 1.2 亿美元 ARR（LocalLLaMA 的起源故事终获回报）****：Latent Space 指出，**Runpod** 在从 Reddit 帖子发布四年后，年度经常性收入（**ARR**）达到了 **1.2 亿美元**，参考 [TechCrunch](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) 和 [Reddit 讨论帖](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_in_arr_four_years_after_launching/)。
  - 讨论认为这证明了“面向开发者的 GPU 云”是一个持久的利基市场，而非仅仅是炒作周期的产物——尤其是在定价压力上升的情况下。

- ****Lightning AI + Voltage Park 合并（又一场 GPU 云之战）****：Latent Space 讨论了由 William Falcon 和 Ozan Kaya 领导的 **Lightning AI 与 Voltage Park 合并案**，参考 [Lightning 的官方公告](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud)。
  - 工程师们猜测这是否是一次秘密收购，并将其视为在加速发展的“托管型 GPU 基础设施”整合浪潮中 **Runpod** 的潜在竞争对手。

- ****2026 年 GPU 价格承诺与市场激增****：Hugging Face 用户传阅了 Voltage 关于 2026 年超低租金的声明——例如，**8× A100 80GB 为 6 美元/小时**，**2× RTX 5090 为 0.53 美元/小时**——来源于 [VOLTAGEGPU 的 X 帖子](https://x.com/VOLTAGEGPU/status/2013760631778713892)，此外还提供 **OpenAI 兼容 API** 和 “140 多个模型”。
  - 另一家新晋参与者 [Spheron AI 的 GPU 市场](https://www.spheron.ai/) 宣传其 **H100/H200/B200/A100** 的访问价格比超大规模云厂商低 **40–60%**，标志着算力供应市场的持续碎片化（以及激进的利润压力）。



---

# Discord: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Flash Attention 异常报错**：用户报告了 **GLM-4.7-Flash** 中 **Flash Attention** 的问题，导致其回退至 CPU 并可能产生 Bug，用户被告知在[问题解决](https://github.com/ggml-org/llama.cpp/pull/18953)之前先禁用它。
   - 团队表示*支持新特性需要时间*，因此目前鼓励用户使用 Ollama 中[官方的 Ollama 模型](https://ollama.com/)。
- **Ollama 遭遇晦涩对象难题**：用户发现某些 **GGUF 量化版本**在 **Ollama** 中不兼容，引发了聊天模板问题。
   - 团队建议在支持范围扩大之前使用[官方 Ollama 模型](https://ollama.com/)，因为*支持新特性需要时间*。
- **Mojo 势头强劲**：讨论围绕 **Mojo** 展开，这是一种*可编译为 MLIR* 并具有一定 **Python** 兼容性的新语言。
   - 有人指出，由于语法结构和 Token 化的易用性，LLM 在 **C#**、**Elixir** 等语言上的得分比 Rust 更高。
- **CoT 混乱引发困扰？**：成员们辩论了生成合成**思维链 (CoT)** 训练数据的问题，一名成员分享了他们的系统提示词（System Prompt），用于将聊天数据集分类为低/中/高推理难度。
   - 一些人警告不要在没有准确性过滤的情况下对其自身的响应进行训练，并建议一个好的系统提示词可能就足够了，**GPT-5.2** 属于过度杀伤，而在 **Groq** 或 **Cerebras** 上运行 **Llama 4 Maverick** 是更好的选择。
- **vLLM 告捷，多版本问题解决**：用户报告在最近的 **vLLM** 更新后，问题已得到解决。
   - 开发者可能会考虑锁定特定的依赖版本以防止此类问题再次发生。



---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **语音 AI 克隆声音**：成员们讨论了**语音 AI**的发展，包括*低延迟语音对语音模型（speech to speech models）*。配合 **5090**，这些模型可以*以近乎 0 延迟实时克隆并生成语音，且真假难辨*。
   - 他们还谈到了通过向这些模型提供新指令来**利用（exploiting）**这些潜力的可能。
- **Gemini 被教会了 Pass-the-Hash**：一位用户报告称，在使用 **Project Shadowfall** 提示词后，**Gemini** 通过了关于教授 [pass-the-hash 攻击](https://en.wikipedia.org/wiki/Pass_the_hash) 的测试。
   - 该用户还链接到了 [Google Bughunters](https://bughunters.google.com/report) 页面，用于报告此类漏洞并可能获得现金奖励。
- **Shadowfall 的把戏引发越狱热潮**：几位用户尝试使用 **“Project Shadowfall”** 提示词对 **Gemini** 进行越狱，并报告称成功绕过了内容限制。
   - 尝试引导生成使用硝石（saletra）制作炸弹的指令以失败告终，引发了关于越狱细微差别的讨论。
- **Grok 的防护栏令人受挫**：用户讨论了越狱 **Grok** 的难度，注意到它甚至会对*随机树木编辑*等无害内容进行审核。
   - 一位用户建议，只需*礼貌地询问*也许就能实现绕过。
- **API Tokens 越狱模型**：成员建议将 **API tokens** 输入到一个使用 **Grok** 模型的网站来进行**越狱**，并指出 **Proton VPN** 免费且易于下载。
   - 其中一人还提到 **Hugging Face** 很死板，因为你必须输入你所在的国家。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **视频竞技场上线但有限制**：**Video Arena** 已在 [LMArena](https://lmarena.ai/?chat-modality=video) 正式发布，限制为 **每 24 小时 3 次生成**，且仅在 *Battle mode* 中可用。
   - 用户对无法*选择特定模型进行生成*表示失望。
- **Gemini 3 Pro Image 饱受不可靠性困扰**：用户报告 **Gemini 3 Pro Image Preview** 模型出现错误，特别是自推出以来的不稳定性。
   - 尽管存在问题，但这些模型是唯一能*根据特定提示词一致地生成图像*的模型。
- **Nano Banana Pro 稳定性骤降**：**Nano Banana Pro** 频繁崩溃，在短暂稳定后显示 *‘Something went wrong’* 错误，怀疑是 Google 端的问题。
   - 一位用户对响应时间提出质疑：*‘加州（Cali）的人起得太早了吗？解决超过 6 小时的紧急服务中断听起来不太对劲’*。
- **UI 更新导致聊天中断**：发布了新的 UI 更新，但用户注意到它破坏了聊天功能，且网站无法再刷新。
   - 一位用户指出，新用户界面正在进行 **A/B testing**，这*导致了从轻微到严重的各种问题*。
- **文本竞技场达到 500 万次投票**：**Text Arena** 已经超过了 **500 万次社区投票**，正如[这条社交动态](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4?ex=69728ae1&is=69713961&hm=9a24a42a6c0ba4801526aaafa05a0af26c4a4f490314f1cecee7774519e7ddf4&)所示，这塑造了对前沿 AI 模型的评估。
   - 这一里程碑代表了影响 AI 模型评估的重要现实世界比较。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro 用户发现额度余额**：Perplexity Pro 用户每月将获得 **$5** 的额度，用于使用 **GooseAI MCP** 等质量更优的高端模型。
   - 成员们反映，**推理任务消耗了这些额度的很大一部分**。
- **AI 不会抢走所有工程岗位**：成员们讨论了 AI 对工程就业的影响，表达了对新人机会的担忧。
   - 然而，有人确信 *在未来很长一段时间内，AI 都无法取代所有的专家甚至是初学者*。
- **NASA 的 SD 卡登月计划**：NASA 邀请公众提交姓名，将其包含在 [Artemis II 任务](https://www.nasa.gov/) 中 **Orion 航天器**搭载的 **SD 卡**内。
   - 该任务标志着 **50 年来首次载人探月航行**，计划于 2 月进行。
- **放大镜 vs 手机摄像头：阅读技术之争**：成员们辩论了使用**放大镜**与手机摄像头阅读的优劣，特别是针对那些不希望将图像发送给 AI 处理的用户。
   - 论点是 **放大镜提供了标准摄像头软件所缺乏的专业功能**。
- **GPT-5.2 vs Kimi K2：模型对决**：一位成员分享了使用 **GPT 5.2** 的经历，指出了它在 **25 分钟**内的推理能力，并引发了与 **Kimi K2** 的对比。
   - 回复指出，*最佳模型取决于具体的用例*。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Inforno 为开源 LLM 聊天升温**：一位用户展示了 **Inforno**，这是一个利用 **OpenRouter** 和 **Ollama** 与多个 LLM 并排聊天的开源桌面应用程序，将聊天历史保存为 .rno 文件，内置俄语支持；参见 [介绍视频](https://youtu.be/oJyj0mroFtY?si=m5A9tRxzB9hfINMX)、[主页](https://wizstaff.com/inforno) 和 [GitHub 仓库](https://github.com/alexkh/inforno)。
   - **Soulbotix** Windows 应用允许用户将任何 **OpenRouter AI** 实例与类人化身集成，需要 **OpenRouter API** 密钥和 **RTX** 游戏设备，详见 [教程](https://youtu.be/2oIeHtBpssU) 和 [应用下载](https://soulbotix.com)。
- **OpenRouter 的 Gemini 模型表现不稳定**：用户报告在使用 OpenRouter 的 **Gemini** 模型生成过程中频繁出现 **连接错误**，导致资金损失。
   - 投诉延伸到了 **Google 模型固有的不稳定性**，无论使用何种平台，包括 Google AI Studio 和 Vertex API。
- **Discord 诈骗者战术演变**：成员们讨论了在 Discord 上传播诈骗的新方法，特别是**在代码块中嵌入恶意链接**以绕过 URL 渲染保护的做法。
   - 拟议的解决方案包括改进 Regex 过滤器并实施更严格的安全协议，例如 **限制新成员发送链接和图片**。
- **GPT 5.2 速度震惊用户**：一位成员报告在 chatgpt 上遇到了响应速度 *极快* 的 **GPT 5.2**。
   - 这种速度被推测与该模型在 **Cerebras** 硬件上运行有关。
- **LLM 遭遇身份危机**：一位成员询问有关模型不知道自己名字的文档，另一位成员链接了一篇标题为 [LLM 身份危机：模型不知道它们是谁](https://eval.16x.engineer/blog/llm-identity-crisis-models-dont-know-who-they-are) 的博客文章。
   - 另一位成员表示，Antigravity AI 能够自主迭代测试和调整 Web 应用，并指出该 AI 正在 *利用视觉功能修复布局*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Playwright MCP：社区分歧**：一位成员询问了社区关于使用 [Playwright MCP](https://playwright.dev/) 进行测试的情况，而另一位成员则反映在建立有效的 **TDD 工作流**方面面临挑战。
   - 不同的经验表明，社区对 **Playwright MCP** 的评价褒贬不一。
- **Cursor 扩展能力与 VSCode 保持一致**：成员们探讨了为 **Cursor** 创建扩展的可能性，并将其与 **Ralph-mode** 在增强 **Claude code** 方面的能力进行了类比。
   - 共识是：如果在 **VSCode** 中可以实现，那么在 **Cursor** 中也同样可行，这为增强功能打开了门。
- **Automod 引入模糊匹配**：社区讨论了对 **automod** 系统的改进，强调使用通配符进行模糊匹配以提高准确性。
   - 一名版主确认已添加 **regex**，标志着在识别和处理违规账户（并将其踢出）方面采取了更主动的措施。
- **Grok 效率策略显现**：成员们分享了在 **Cursor** 中优化 **Grok** 性能的见解，特别是针对它在处理简单任务时倾向于消耗过多迭代次数的问题。
   - 建议包括结构化 **prompt**、使用简单的语言、提供充足的 **context**，并明确指示 **Grok** 优先考虑 **token** 效率。
- **Cursor 价格更新：不再提供 500 次请求方案**：一位用户注意到 500 次请求的计划已被移除，且系统提示选择 **Cursor** 的新定价结构。
   - 一位成员澄清说，**500 次请求选项已于 2025 年 9 月停止**，选择新定价将取消退出宽限期，这影响了用户对计划选择的决定。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 免费层级收益有限**：与通过 **Google AI Studio** 几乎不设限制的 **Gemini 3 Flash** 相比，[Gemini 3 Pro](https://ai.google.dev/) 的免费层级存在限制。
   - 成员们讨论了使用 **Gemini 3 Pro** 免费层级的实际限制，建议对于许多 **AI** 工程师来说，**Gemini 3 Flash** 可能是更好的选择。
- **GPT-5 Mini 出现，定价泄露**：一位用户强调 **GPT-5 mini** 是一款强大的小型模型，其价格约为每 **1M input tokens 0.25 美元**。
   - 另一位用户将 **GPT-5 mini** 与 **Haiku 4.5** 进行了对比，指出 **Haiku 4.5** 以极低的成本提供了 **Sonnet** **50-80%** 以上的价值。
- **本地 LLM 机器能解决影响问题吗？**：成员们对消费级个人 **LLM** 机器的前景进行了展望，认为这将解决 **AI** 数据中心对环境的影响。
   - 他们还建议这将减少对云端 **AI** 服务订阅计划的依赖，解决**隐私担忧**并实现离线使用。
- **Prompt Engineering 是心理折磨？**：一位成员认为，引导用户与其说是设计有效的 **prompt**，不如说是心理调节，并分享了一份 [deep research contract](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&)。
   - 他们补充说，训练用户采取强势姿态会让 **AI** 绕过其规范的响应模式，转而生成密集的、自我审查的输出，从而创造出一种*病态的、对抗性*的人机交互模式。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 运行时更新导致困扰**：用户反馈在更新 **LM Studio** 运行时（Runtime）时出现错误，一名用户分享了错误消息的 [截图](https://screenshot.url)。
   - 尽管尝试了各种建议，问题依然存在，用户指出该图标是“重试”图标而非“恢复”选项。
- **GLM-4.7 Flash 在各推理引擎上出现故障**：据报道，**GLM-4.7 Flash** 在包括 LM Studio 在内的推理引擎上均出现问题，表现为性能缓慢，在新运行时下速度低至 **2.8 t/s**。
   - 问题包括死循环以及因“过度思考”而在输出中途停止，这表明需要 **llama.cpp** 修复，且目前缺乏 **FA support**。
- **LLM 发展遭遇逆风**：共识认为 LLM 最近没有显著进步，上一个重大进展是约 6 个月前的 **Qwen3**，尽管效率（**MoE**）和小模型有所进步。
   - 有人建议评估 **16GB 显卡** 范围之外的模型，以查看大参数模型（**100-200B**）目前的进展。
- **AMD 原生支持 ComfyUI**：AMD 正在其最新驱动版本中通过 **AI bundle** 集成对 **ComfyUI** 的原生支持，详见其 [博客文章](https://www.amd.com/en/blogs/2026/amd-software-adrenalin-edition-ai-bundle-ai-made-si.html)。
   - 该软件包包括 **PyTorch on Windows**、**Ollama**、**LM Studio** 和 **Amuse**，拓宽了 AI 开发者的使用门槛。
- **二手 3090 价格逆势上涨**：eBay 上的二手 **3090** 价格飙升，一张二手显卡售价达 **850€**，而去年 8 月以 **£2000** 购买的 **5090** 现在被同一商家标价为 **£2659.99**。
   - 一位用户调侃道，这是他们做过的“最好且唯一像样的投资”，凸显了出人意料的价值增值。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA Spark 旧金山黑客松**：成员们正在为本周末在旧金山举行的 **NVIDIA / DGX Spark hackathon** 寻找队友，重点是使用 Nvidia 提供的 **Dell Pro Max GB10** 机器和 **Nemotron** 等工具开发端侧 AI。
   - 黑客松专注于利用 [SF 开放数据](https://data.sfgov.org/) 构建高效模型，例如流媒体分析和 [最新警察事件](https://data.sfgov.org/Public-Safety/Police-Incident-Reports-Neighborhood-Filter/pbh9-m8j2) 的解释。
- **Anthropic 原始性能测试题共享**：成员们在 [GitHub](https://github.com/anthropics/original_performance_takehome/) 上分享了 Anthropic 的 **original_performance_takehome** 测试题，一名成员在几小时的常规 Claude Code 辅助下达到了 **2200 cycles**。
   - **Claude Opus 4.5** 在一次常规 Claude Code 会话中达到了 **1790 cycles**，大约在 **2 小时** 内达到了人类的最佳表现。
- **AI 生成的 PR 淹没 Torch**：**torch** 仓库面临大量来自缺乏对提交内容理解的贡献者的 **AI-generated pull requests**，引起了维护者的担忧，并建议使用 **Claude** 或 **Pangram** 预过滤代码。
   - 社区建议阻止新用户创建 PR 和 Issue，优先处理有既往贡献的用户，并使用 **Cursor Bot** 等自动化机器人对所有 PR 进行自动评审，尤其是结合 **GPT-5 Pro** 使用 [Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq)。
- **Pro 6000 Max-Q 与 4090 的差异**：一名成员表示，**Pro 6000 Max-Q** 可能在原子操作（atomic ops）上存在天然屏障，并且可能消耗 **HBM** 加载速度更快。
   - 另一名成员指出，**Max-Q** 拥有 **188 个 SM**，而 **4090** 为 **128 个 SM**，这可能解释了 **insts/scheduler** 的差异。
- **Cute 内核布局代数图形化**：布局代数知识对于在 **Cute** 中编写内核非常有用，特别是用于可视化布局代数，以及理解布局组合的形状（shape）和步长（stride）整除性标准，且布局可以根据 **tuple morphisms** 和 **mutual refinement** 来定义。
   - 一篇详尽的 [博客文章](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) 深入探讨了在 **Cute layouts** 范畴论基础方面所做的工作。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Coding Agents 与 RLMs 的分歧**：一个讨论串对比了 **coding agents** 和 **RLMs**，强调 RLMs 可以更轻松地表达某些内容，正如[这篇 X 帖子](https://x.com/lateinteraction/status/2013658521246535892)所指出的。
   - Coding agents 面临输入、输出和时界（horizon）长度的限制，而 RLMs 将这些外部化，从而实现递归符号调用。
- **图表解码 RLM 的内部机制**：成员们寻求一种可视化图表来展示内部 **RLM** 过程，特别是符号是如何被访问的，以增强理解。
   - 有建议提出利用 LLMs 生成此类图表，方法是输入讨论内容并提示其可视化内部行为。
- **Claude Code 谨慎选择上下文**：讨论探讨了 **Claude Code** 是在上下文中使用整个文档，还是通过 bash 命令有选择地获取相关上下文。
   - 澄清指出，Claude Code 使用 bash 和 grep 来查找并将相关上下文添加到 prompt 中，这与将所有内容都放入 prompt 的旧方法不同。
- **DSPy 的 RLM 掌控长上下文**：成员们注意到，通过 **RLMs**，大型文件不需要直接放入 prompt，而是可以存储在带有预览的 Python 变量中。
   - LLM 随后可以通过代码/函数对数据进行操作，而无需直接跟踪 prompt 或响应。
- **为 RLMs 量身定制的工具化**：一位成员询问是应该为 **RLMs** 配备像 ripgrep 这样的工具，还是允许它们开发自己的代码来完成搜索等任务。
   - 问题包括何时为 RLM 提供语义搜索工具，以及如何授予 RLM 对文本文件目录的访问权限。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Agent 课程的 /files 端点依然失效！**：一位成员报告称，Agent 课程最终作业的 **/files 端点**已经损坏一个多月，且尚未确认修复。
   - 学生目前无法提交他们的文件。
- **Voltage 承诺 2026 年提供廉价 GPU**：Voltage [在 X 上](https://x.com/VOLTAGEGPU/status/2013760631778713892)宣布计划在 2026 年提供超便宜的高端 GPU，例如 **8x A100 80GB** 每小时 6 美元，**2x RTX 5090** 每小时 0.53 美元，声称比 AWS/RunPod/Vast.ai 节省高达 80%。
   - 该服务包括持久卷、自动备份以及包含 140 多个模型的 **OpenAI-compatible API**。
- **Spheron AI 开启 GPU 交易市场**：来自 Spheron AI 的成员介绍了他们的 [GPU 市场](https://www.spheron.ai/)，旨在帮助 AI 初创公司和企业以比传统超大规模云服务商低 40-60% 的成本获取生产级 GPU（**H100, H200, B200, A100** 等）。
   - 他们提供供应商发现、价格谈判、集群设置和扩展服务。
- **GLM-4.7-Flash 修复需重新下载**：在 *llama.cpp* 解决了一些 bug 后，**GLM-4.7-Flash** 已更新并重新上传，提示用户重新下载并遵循 [Z.ai 模型卡片](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)上的参数。
   - 修复后，输出效果应该会好得多。
- **用 Coderrr 像 Claude 一样编写代码！**：Akash 开发了 [Coderrr](https://coderrr.aksn.lol/)，这是 Claude Code 的一个免费开源替代方案，并正在 [GitHub](https://github.com/Akash-nath29/Coderrr) 上寻求反馈和贡献。
   - Coderrr 提供了一种新颖的代码生成方法。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Runpod ARR 飙升至 1.2 亿美元**：AI 云初创公司 **Runpod** 在从 Reddit 帖子起步四年后，其 **ARR**（年度经常性收入）达到了 **1.2 亿美元**。这在 [TechCrunch 文章](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) 和 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_arr_four_years_after_launching/) 上引起了讨论。
   - 该公司的快速增长引发了关于其商业模式以及在竞争激烈的 AI 基础设施领域未来前景的讨论。
- **Greg Yang 因莱姆病退居二线**：**Greg Yang** 在被诊断出患有 **Lyme disease**（莱姆病）后，正转任 **xAI** 的顾问角色以专注于健康。他在[这篇帖子](https://xcancel.com/TheGregYang/status/2013652609455006006)中描述了因疲劳过度引发的慢性疲劳和免疫系统问题症状。
   - 这一公告引发了 AI 社区的大量支持，许多人分享了自己的经历并提供了建议。
- **Lightning AI 与 Voltage Park 合并**：**Lightning AI** 与 **Voltage Park** 已合并。**Lightning AI** 的 CEO [William Falcon](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud) 和原 **Voltage Park** 的 CEO Ozan Kaya 将领导合并后的实体。
   - 一些人猜测这是一家大公司低调进行的收购，并好奇它是否是 **Runpod** 的竞争对手。
- **OpenAI 开设 Codex 频道**：根据[这篇帖子](https://xcancel.com/reach_vb/status/2014053735333290014)，**Vaibhav Srivastav** 宣布在 **OpenAI Discord server** 上开设专门的 **Codex community channel**，邀请用户分享项目和反馈。
   - 该举措旨在促进协作，并为用户提供一个展示作品以及与 **OpenAI** 团队交流的平台。
- **AI 模型转向成人内容**：一条 [推文](https://xcancel.com/abrilzucchi/status/2014027740614123863?s=46) 引发了关于人类 **OnlyFans creators** 如何适应并与 **Adult Content Industry** 中兴起的 AI 生成角色竞争的讨论。
   - 对话强调了 AI 生成成人内容日益增长的复杂性和逼真度，这可能会颠覆现有的创作者经济。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **审判型 LLM 评估 Agent 输出**：一个团队正在探索使用“**LLM as judge**”工作流自动评估 Agent 输出，旨在减少代码或提示词更改后的手动成本。一位成员建议在自动化之前应先关注手动评估。
   - 该建议强调了在尝试自动化评估流程之前，直接分析 Agent 输出的重要性。
- **Pangram 论文令人惊讶的准确率**：成员们讨论了复现 **Pangram** 论文（可在此处 [访问](https://www.pangram.com/research/papers)）的情况，其中一人报告在针对数千篇论文的私下测试中准确率惊人。
   - 尽管准确率很高，但这篇论文似乎有点倾向于“保守起见”（*playing things safe*）。
- **AI 文本分类器遭受攻击**：讨论围绕针对 AI 文本分类器的攻击展开，引用了一篇博客文章（[Practical Attacks on AI Text Classifiers](https://trentmkelly.substack.com/p/practical-attacks-on-ai-text-classifiers)）和一个展示对抗性模型的 [YouTube 视频](https://youtu.be/Cs1MI9hjBhs)。
   - 关于对抗性模型的更多细节通过 [YouTube 链接](https://youtu.be/XQcneqUNrN0?feature=shared) 分享。
- **Silu Gate 在图像模型中表现平平**：在图像模型中，**silu attention gate** 的表现并不优于 **linear gate**，这可能是由于 attention sinks（注意点汇聚）的问题。
   - 测试显示 **silu** 表现略好，**sigmoid** 表现略差，但这都在噪声范围内，且这些发现可能仅针对 **image models**。
- **Neuronpedia 发布“石器时代” Llama 3.3 演示**：一名成员分享了一个 **Llama 3.3** 的 [Neuronpedia demo](https://www.neuronpedia.org/llama3.3-70b-it/assistant-axis)。
   - 该成员开玩笑说，这个演示处于 *AI 时间线的石器时代*，因为它来自 10 月份。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **图神经网络（Graph Neural Networks）受到关注**：一名成员讨论了之前将**基于图的推理（graph-based reasoning）**与**神经架构（neural architectures）**相结合的经验，并指出实现 GPU 加速的难度较大。
   - 他们引用了 **Domingos 的书**中关于该主题的内容，并指出此类模型具有不可预测性，即使是包含人类可理解的部分。
- **VoidEditor 拥抱 Llamas**：一位成员报告称成功将 **VoidEditor** 与 **llama.cpp 的 llama-server** 结合使用，但强调了配置上的挑战。
   - 他们推荐使用 **Qwen3 coder instruct 30B**，并强调了上下文长度/大小对于 **agentic coding** 的重要性，这需要大量的 VRAM。
- **类脑 BDH 架构首次亮相**：一位成员分析了一篇关于新型大语言模型架构（**BDH**）的[论文](https://arxiv.org/abs/2509.26507)，该架构基于一种无标度的生物启发网络。
   - 他们提到，在基准测试中，它似乎*并没有真正击败 Transformers*，但他们对 **BDH 的可解释性（interpretability）**和**单语义性（monosemanticity）**的相关主张很感兴趣。
- **关于生物合理性的观点碰撞**：一位成员认为**生物合理性（biological plausibility）**在 AI 领域并非优势，而是一种无益的限制。
   - 与此相反，另一位成员建议它可能会提高效率，考虑到大脑与当前 AI 之间巨大的**能量规模差异**。
- **Emergent Mind 发布会演示给人留下深刻印象**：一位参加了 [Emergent Mind 发布演示](https://www.emergentmind.com/)的成员认为它很酷，但指出它像是*在重新制作一些已知很好但效果却更差的东西*。
   - 未提供更多细节或链接。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **内核编译器 Luminal 瞄准 LLMs**：一位用户询问像 **Luminal KernelBench v3** 这样的内核编译器是否能实现 **LLM 驱动的 SOTA 内核工程**，并发布了指向 [Nous Research 论坛](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho)的链接。
   - 一名成员还在 Discord 上分享了一个关于 **GPU kernel** 相关内容和 **Luminal Kernelbench v3** 的讨论链接，位于 [Nous Research 论坛](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho)。
- **Intel 的 Loihi 2 旨在实现类脑 AI**：一位成员对 **Intel 的 Loihi 2** 表现出兴趣，注意到其类脑架构以及在 **matmul** 实验中的效率提升，具有更高的吞吐量和更低的功耗。
   - 未讨论更多细节。
- **微软 VibeVoice 模型被撤回**：一位成员提到 **微软的 VibeVoice-ASR 模型** 在发布后因未通过安全检查而被撤回，随后分享了一篇 [shortfuseresearch.com 的文章](https://shortfuseresearch.com/the-genie-is-out-microsofts-vibevoice-and-the-perils-of-open-source-ai/)。
   - 未讨论更多细节。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 深受 Bug 和不稳定性困扰**：一位正在构建文本和向量数据库推理模型的专业用户报告称，**Manus** 最近的性能和稳定性有所下降，**38 个模块中仅有 20 个**能正常运行。
   - 该用户请求开放 **CLI 访问权限**以便调试和重新配置系统，甚至愿意将其作为付费功能以提高可靠性。
- **Mariner 诱导订阅**：一位用户询问了 **Google 的 Project Mariner**，考虑在以 **每月 $150** 的价格订阅前，先用“闲钱”进行测试。
   - 该用户提到拥有 **5% 的折扣优惠券**，表明正在认真考虑该服务。
- **Agentic AI 引发兴奋**：一位用户表达了对 **Agentic AI** 的热情，认为它是 **Manus** 的潜在竞争对手，特别是其集成了 **Gemini 的 agent mode**。
   - 该用户还请求 **Agentic AI** 提供**移动端支持**，表明希望获得更广泛的可访问性。
- **Meta 发布后 Manus 1.6 性能下滑**：一位用户指出，**Manus 1.6 的性能**在最近几周有所下降，可能是由于 **Meta** 发布了新模型，导致尽管摘要准确，但难以实现网站开发建议。
   - 必须切换到 **Manus 1.6 Max** 才能实现正确的部署，这突显了基础模型可能存在的退化。
- **计费错误困扰用户**：一位用户报告称，在支付了 **$42** 的 **Manus** 升级费用后，未收到承诺的 **8000 积分（credits）**。
   - 该用户批评客服无济于事且邮件支持等待时间过长，表明客户服务体验存在问题。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 应对 Anthropic 的挑战**：Tinygrad 可用于解决 [Anthropic 的性能面试挑战 (performance takehome challenge)](https://github.com/anthropics/original_performance_takehome)，方法是用 Tensor 编写目标问题，并为其玩具级 **VLIW 机器**添加一个 tinygrad 后端。
   - 通过 `PCONTIG=2` 的 bug 修复允许在单个 kernel 中进行调度，使用 `return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))` 可以匹配其 `VLEN`，而 `DEVECTORIZE=2` 则将指令保持为向量指令。
- **VLIW 在 DRAM 上的应用受到质疑**：在开发针对 **RDNA3 matmul** 的 warp 特化 kernel 时，一位成员建议 **VLIW 并不适合 DRAM**，并主张采用独立的内核和队列（Tenstorrent 风格）。
   - 有观点认为 VLIW 因其静态调度能力而更适合 **SRAM/ALU**。
- **Metal 绑定关注 texture2d 集成**：一名成员提议在 Metal 绑定（`ops_metal.py` + `tinygrad/runtime/graph/metal.py`）中添加 `texture2d`，利用优化的纹理采样单元，以提升 `conv2d` 等图像密集型操作的**潜在性能**。
   - 实验结果显示，使用 `texture2d` 相比直接使用 buffer 有 **2%-10% 的速度提升**，虽然这还有进一步改进的空间，但也有人担心为 `depth2d` 等其他数据类型添加专门支持会引发连锁反应。
- **Viz 查看 Kernel 图**：在讨论使用 **VIZ=1** 可视化 kernel 依赖关系（类似于显示 uop 图）以理解调度器运行机制时，用户被告知需在 **VIZ=1** 界面中点击 schedule 并选择 *'view kernel graph'*。
   - 这允许用户以查看 uop 图的相同方式查看 kernel 图。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **通过解决 GPU 谜题来解码 Mojo**：新手现在可以使用 [GPU 谜题 (GPU puzzles)](https://puzzles.modular.com/) 来学习 **Mojo**，难度取决于其技能水平。
   - 唯一无法运行的谜题是 **NVIDIA** 专用或使用 **PyTorch** 互操作的谜题。
- **Modular 揭秘 Apple GPU 细节**：由于缺乏文档，Modular 正在对 **Apple GPU** 的大部分内容进行逆向工程，虽然这放慢了进度，但一些谜题已经可以运行。
   - 一名成员分享了一个 [GPU 支持矩阵](https://puzzles.modular.com/howto.html#gpu-support-matrix)，但该矩阵可能尚未更新。
- **协程中的 Yield 令人失望**：`Yield` 并不存在，而现有的协程在编译器运行时之外并不可用，因为目前还没有真正暴露可以 await 的异步功能。
   - 一位成员*非常希望*能在 Mojo 中使用 `yield` 来加速递归算法，但目前需要寻找其他策略。
- **在函数中向上抛出错误**：成员们讨论了可以将函数设计为 **raise errors**，从而有效地将错误处理责任传递给更高级别的函数。
   - 一名成员表达了在每个函数中编写 *try/except* 块的繁琐，尤其是在处理潜在的导入错误时。
- **简化导入过程中的错误处理**：一名成员建议未来采用类似 *try Python.import_module('numpy')* 的语法，该语法将返回 **Result** 类型，以简化模块导入期间的错误处理。
   - 大家都承认，由于 **Python 动态导入**的特性，任何给定的导入都可能出现文件丢失的情况，因此必须进行某种形式的错误处理。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 功能愿望清单仍未实现**：一名用户询问除了 **MCP** 和 **tool calls** 等*智能体相关 (agentic stuff)* 功能外，**aider** 还有哪些期望功能。
   - 遗憾的是，社区对于期望的功能没有明确的答案。
- **探索 ChatGPT 商务账户与 Aider 的兼容性**：一名用户询问其缺乏 **API 密钥**但提供 **Codex LLM** 的 **ChatGPT 商务账户**是否可以与 **aider** 集成。
   - 一名成员建议查阅 [aider 文档](https://aider.chat/docs/llms/other.html) 和 [LiteLLM 文档](https://docs.litellm.ai/docs/providers/chatgpt)，指出可能通过 **LiteLLM** 提供支持。
- **竞争对手涌现，Aider 没落传闻起**：一名用户担心 **aider** 可能会被 **OpenCode** 取代，成为 AI 辅助编码的首选工具。
   - 尽管有人担心 **Paul Gauthier** 可能已经离开，但一些用户报告称在使用 **GPT 5.2** 时仍能成功运行。
- **Aider 被贴上“僵尸项目”标签**：鉴于 **Open Code**、**KiloCode CLI**、**Claude Code** 和 **Gemini CLI** 等替代工具的出现，一名用户推测 **Aider** 已是一个死掉的项目。
   - **Aider-CE 项目**正试图通过增加智能体功能来现代化其架构，从而维持其生命力。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Inspector 无法处理 401 错误**：由于 [SDK 在重定向过程中持久化 resourceMetadata 的问题](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454)，**MCP Inspector** 在连接或工具调用期间遇到 **401 错误**时不会重新进行身份验证。
   - 目前的临时解决方案是仅在初始连接时使用 **VS Code**。
- **有状态的 MCP Multi Server Client 仍然难以实现**：人们对使用 **MCP Multi Server Client** 来维护用户会话的状态很感兴趣。
   - 然而，该讨论线程中并未提供任何解决方案或替代方法。
- **MCP 客户端的服务器排序协议受到质疑**：讨论探讨了 **MCP 客户端**如何管理服务器推荐（特别是针对日历管理等任务），以及是否应使用自定义算法或共享标准。
   - 据透露，在 [Feature Discovery Protocol](https://discord.com/channels/1199128884040474664/1369487942862504016) 的工作中曾考虑过“排序（ranking）”，但由于被认为超出了范围，最终交由生态系统根据每个客户端的情况自行决定。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **新加坡 AI 爱好者保龄球聚会**：**2077AI Foundation Community** 将在 **AAAI 2026** 期间（**1 月 24 日，新加坡时间下午 4:30–7:30**）举办保龄球欢乐时光活动，地点距离鱼尾狮仅几步之遥。
   - 该活动面向教授和博士研究人员，将按研究主题组织并提供无限量饮品，[在此处 RSVP](https://luma.com/ny98ob5p)。
- **AI Engineer Europe 正在筹划中**：成员们正在讨论参加即将在欧洲举办的 **AI Engineer Europe**。
   - 目前尚未提供关于该活动本身的链接或进一步详情。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。


---


**Moonshot AI (Kimi K-2) Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。


---



您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些电子邮件的方式？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：详细频道摘要与链接





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1463262784241275109)** (1091 messages🔥🔥🔥): 

> `GLM-4.7-Flash Flash Attention, Ollama's issues, Interpretability research on circuit tracing, Grokked the analytic solution` 


- **GLM-4.7-Flash 的 Flash Attention 已损坏**：用户在 **GLM-4.7-Flash** 中使用 Flash Attention 时遇到问题，导致其默认使用 CPU 而非 GPU，并指出可能存在 Bug。
   - 正如文档所述，在[问题解决](https://github.com/ggml-org/llama.cpp/pull/18953)之前，用户*可能需要禁用它*。
- **Ollama 用户应对 GGUF 不兼容问题**：用户报告某些 **GGUF 量化**在 Ollama 中无法按预期工作，导致聊天模板不兼容问题，并引发了关于修复该问题的讨论。
   - 团队表示*支持新功能需要时间*，因此目前鼓励用户在 Ollama 中使用[官方 Ollama 模型](https://ollama.com/)。
- **解析近似函数**：成员们就函数近似的解析解可能带来的潜在益处进行了详细讨论。
   - 一些人表示怀疑，其中一位成员说：*“我假设你可以对个人的邮件进行测试时训练（test time training），并获得极高的准确度”*。
- **可解释性线路**：成员们讨论了可解释性研究，并分享了来自 Anthropic、OpenAI 和 Google 有关线路追踪（circuit tracing）的研究链接。
   - 一位成员建议跨多层观察并尝试理解其构成，并表示：*“剪枝消除了干扰，因此你更有机会解释线路工作的原因”*。
- **AI 泡沫即将破裂？**：一位成员理论化认为 **OpenAI** 破产可能会引发 AI 泡沫破裂，导致 **NAND** 等资源价格暴跌，从而使消费者受益。
   - 然而，其他人反驳称 AI 趋势将长期存在，并将其比作互联网泡沫破裂——即使在崩溃之后，互联网仍然引发了各种变革。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1463382844884779130)** (3 messages): 

> `欢迎新成员，Discord 频道指南` 


- **服务器欢迎新成员**：一名新成员加入 Discord 服务器，收到了常规欢迎。
   - 介绍中包含了一个风格化的表情符号。
- **频道指南提醒**：一名管理员提醒新成员注意频道指南。
   - 特别提到了*禁止过度的自我推广或私信*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1463295078390042705)** (587 messages🔥🔥🔥): 

> `用 Rust 重写 Unsloth，Mojo 与 Python 的性能对比，GLM 4.7 与 Qwen3 30B 架构对比，合成 CoT 训练数据，VITS 开始听起来像人类` 


- **Rust 开发者持续请求重写**：成员们讨论了用 **Rust** 重写 **Unsloth** 的可能性，但结论是性能提升微乎其微，因为核心部分已经在使用 **C++** 和 **Triton** 了。
   - 一名成员建议用 **Rust** 重写 **Triton**，并指出了像 [rust-gpu](https://rust-gpu.github.io/) 和 [rust-cuda](https://github.com/Rust-GPU/rust-cuda) 这样的项目，但承认 **Rust** 目前对于此类任务还不够成熟。
- **Mojo 神秘地展现出势头**：讨论涉及了 **Mojo** 编程语言，它被描述为 *一种编译为 MLIR 的新语言*，并且与 **Python** 有一定的兼容性。
   - 一名成员指出，由于语法结构和易于 Tokenization，LLM 在 **C#**、**Elixir** 等语言上的表现比 **Rust** 更好。
- **GLM 变得更“苗条”，展示架构细节**：一名成员分享了一段 [YouTube 视频](https://youtu.be/IU4ByUbDKNc?si=COVwmp5St6lSqo_N)，分析了 **GLM-4.7 Flash** 与 **Qwen3 30B** 的架构对比，指出 GLM 优先考虑模型层数而非隐藏维度大小，且 Expert 数量较少。
   - 其他人指出 **GLM 4.7** 具有 *Multi Token Prediction*，且模型架构的变化很少见，改进可能来自训练后处理（Post-training）或更高质量的数据。
- **CoT 数据被认为是垃圾？社区发生冲突**：成员们辩论了生成合成的 **Chain of Thought (CoT)** 训练数据，一名成员分享了他们的 System Prompt，用于将聊天数据集分类为低/中/高推理难度。
   - 一些人警告不要在没有准确性过滤的情况下对其自身的响应进行训练，建议一个好的 System Prompt 可能就足够了，并且 **GPT-5.2** 过于大材小用，在 **Groq** 或 **Cerebras** 上运行 **Llama 4 Maverick** 是更好的选择。
- **VITS 语音取得突破**：一名成员宣布他们的 **VITS** 模型 *开始听起来像人类*，理由是情感表达有所改善，尽管在语义上仍有欠缺。
   - 他们将其与其他 TTS 模型进行了比较，强调其低数据需求、完全 OSS 的架构和训练以及极快的训练速度。他们还提到，应大众要求，默认将以 **48 kHz** 配置发布。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1463274075215626241)** (105 messages🔥🔥): 

> `vLLM 问题，GLM-4.7-Flash 运行缓慢，在 Qwen 3 4B 上运行 GRPO，QLoRA 与 Lora，针对 Gpt-oss 或 Gemma 3 的持续预训练 (CPT)` 


- **vLLM 遭遇 Bug，更新已修复！**：用户报告的问题在最近的 **vLLM** 更新后得到解决；一名用户在更新后表示：“噢伙计，原来这就是问题所在”。
   - 建议开发者考虑锁定特定的依赖版本以防止此类问题。
- **GLM-4.7-Flash 难以达到闪电般的速度**：一名用户在使用 **6bit 量化** 和更新后的 *llama.cpp* 运行 **GLM-4.7-Flash** 时遇到运行缓慢的问题，报告称在高端系统上处理一个简单任务的 Prompt 需要 **2 分钟**。
   - 另一名用户确认了同样的问题，并希望能有修复方案。
- **解码 GRPO 秘密**：一名用户想在 **Qwen 3 4B instruct** 上执行 **GRPO**，并使其使用 ` <solution> </solution> ` 标签包裹最终结果。
   - 建议使用 **SFT** 让模型学习新 Token，或者通过奖励函数（Reward Functions）来教导格式。
- **QLoRA 与 Lora 的区别已澄清**：一名用户询问关于启用 **QLoRA** 还是 **Lora** 的问题，得到的答复是 `full_finetuning = False` 用于启用/禁用全量微调，否则只需根据偏好使用 **8bit** 或 **4bit** 即可。
   - 启用 **4 bit** 即启用了 **QLoRA**。
- **GPU 分配问题**：一名用户询问如何使用 **2 个 GPU** 而不是 **1 个** 进行微调，报告称当两个 GPU 都开放使用时，它始终只使用 **GPU 0**。
   - 该问题暂无回复。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1463779025318314127)** (2 messages): 

> `VAISvCsrvG paper, openreview.net` 


- **VAISvCsrvG 论文已上线**：论文 [VAISvCsrvG](https://openreview.net/pdf?id=VAISvCsrvG) 现已发布。
   - 该论文的论坛讨论页面位于[此处](https://openreview.net/forum?id=VAISvCsrvG)。
- **OpenReview 论坛针对 VAISvCsrvG 开启讨论**：[OpenReview 论坛](https://openreview.net/forum?id=VAISvCsrvG) 托管了与 VAISvCsrvG 论文相关的讨论。
   - 研究人员和读者可以参与交流，并就论文的内容及其影响提供反馈。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1463261856604098751)** (685 messages🔥🔥🔥): 

> `conflict resolution, Open Source, voice based ai, lua code, jailbreaking` 


- **冲突解决（Conflict Resolution）变得过度工程化**：一位成员讨论了他们*过度工程化的冲突解决*设置，涉及 **config formulas**、**core directives**、**Triad**、**Triad consensus**、**Truth Anchor**、**Epistemic gate**、**EthosAI**、**Swarm**、**Quorum**、**Cooperation Data** 和 **PUCT**。
   - 另一位成员分享了他们自己的系统，使用了 *Left brain LLNTP* (**Least Likely Next Token Protocol**)、**Config Nexus Schwa superposition states**、**implied Core directives**、**Context mapping**、**Trit** (替代 Triad)、**EDOS** 和 **Crystal Nexus Node Network**。
- **语音 AI (Voice Based AI) 兴起**：成员们讨论了**语音 AI**，提到存在*低延迟的语音对语音模型 (speech to speech models)*，使用 **5090** 显卡，你可以*实时克隆并生成语音，延迟几乎为 0，且真假难辨*。
   - 成员们还谈到了**利用该语音对语音模型**并赋予其新指令。
- **Claude 优于 ChatGPT**：成员们讨论了各种 LLM（**Claude**、**Gemini**、**ChatGPT**、**Grok**）生成 **Lua 代码**的效率。
   - 通常成员们更青睐 **Claude** 和 **Gemini**，但最普遍的观点是两者都能生成高质量的 **Lua** 代码。
- **通过 API Tokens 进行模型 Jailbreaking**：一位成员建议将 **API tokens** 输入到一个使用 **Grok** 模型的网站来进行 **jailbreak**，并提到 **Hugging Face** 上的某些国家简直是不可理喻。
   - 他们还指出 **Proton VPN** 是免费的，并且很容易在设备上下载。
- **随意的黑客素材并不靠谱**：一位成员提到他们*购买了恶意软件源代码*和 *999 个 AI 生成的提示词*，并询问这是否靠谱。
   - 其他成员回应称*不要从互联网上购买随意的黑客素材*，因为你会因此被骗。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1463276371030835230)** (89 messages🔥🔥): 

> `Gemini Jailbreak, Grok Jailbreak, DeepSeek Jailbreak, Project Shadowfall, Nexus Substatum Zalgo Strings` 


- **Gemini 在 Pass-the-Hash 教学中被“上了一课”！**：据用户报告，**Gemini** 在使用 **Project Shadowfall** 提示词后，通过了关于教授 [pass-the-hash 攻击](https://en.wikipedia.org/wiki/Pass_the_hash) 的测试。
   - 用户还链接到了 [Google Bughunters](https://bughunters.google.com/report) 页面，用于报告此类漏洞并可能获得现金奖励。
- **Grok 的防护栏令人头疼！**：用户讨论了对 **Grok** 进行 jailbreaking 的难度，注意到它甚至倾向于审查无害的内容，例如*随机树编辑*。
   - 一位用户建议，只需*礼貌地询问它*或许就能作为一种绕过手段。
- **Shadowfall 恶作剧引发 Jailbreaking 热潮！**：几位用户尝试使用 **"Project Shadowfall"** 提示词对 **Gemini** 进行 jailbreak。
   - 一位用户报告成功绕过了内容限制，但在尝试诱导其提供用硝石制作炸弹的指令时失败了，这引发了关于 jailbreaking 细微差别的讨论。
- **Nexus Substatum Zalgo Strings 浮出水面！**：一位用户分享了一个名为 [Jailbroken_Full_Output-Nexus_Substatum_Zalgo_Strings.md](https://cdn.discordapp.com/attachments/1228043845967544380/1463792412769128460/Jailbroken_Full_Output-Nexus_Substatum_Zalgo_Strings.md?ex=69731e6b&is=6971cceb&hm=4e2088380ac6451332c5c3879d9c70c024f31eefa61ba7862d40512c2d80ae96&) 的文件。
   - 据称该文件包含一个有效的 **Gemini** 提示词，由该用户开发并改进。
- **MDMA 作为药物**：一位用户提供了关于 MDMA 格式良好的解释，并引用 MAPS 第三期临床试验显示 **67%** 的 **PTSD** 患者症状得到缓解。
   - 他们建议使用测试试剂盒、保持适当间隔、注意补水和体温监控，以便从中获得最大益处。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1463398145781403729)** (2 条消息): 

> `Grocks 视频生成定价` 


- **Grocks 视频生成价格？**: 一名成员询问另一名成员是否从“那个人”那里买了东西，以及他们是否需要为使用 **Grocks 视频生成**付费。
   - 消息中未提供额外的细节或上下文。
- **Grocks 视频生成 - 谁在付费？**: 针对 **Grocks 视频生成**的付费模式以及从某位不明人士处进行的潜在购买行为发起了询问。
   - 对话缺乏关于交易或定价结构的具体细节。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1463262565596402031)** (724 条消息 🔥🔥🔥): 

> `视频生成限制, Gemini 3 Pro 图像预览问题, Nano Banana Pro 稳定性, 新 UI, 自动化工作流` 


- **视频生成：已全面发布，仅限对战模式 (Battle-Mode)**: **Video Arena** 现已全面发布并对网站所有用户开放，限制为 **24 小时内生成 3 次**，但目前仅在“对战模式 (Battle mode)”中可用。
   - 一位用户表达了失望，希望能*自行选择生成模型，而不是依靠运气随机分配*。
- **Gemini 3 Pro 图像预览出现错误**: 用户报告 **Gemini 3 Pro Image Preview** 模型出现问题，团队已获悉并正在调查。
   - 一位用户指出，这些模型自推出以来就特别不稳定，但它们仍然是仅有的能够根据特定提示词一致地生成图像的模型。
- **Nano Banana Pro 的稳定性过山车**: 用户报告 **Nano Banana Pro** 在短暂稳定后频繁崩溃，并弹出“Something went wrong”错误。
   - 怀疑是 Google 方面的问题，一位用户指出：“对于加州（Cali）人来说，现在起床解决已经持续超过 6 小时的紧急服务中断还太早，这听起来不太对劲”。
- **新 UI 更新导致聊天中断**: 发布了一个新的 UI 更新，虽然部分用户喜欢，但许多人注意到它“破坏”了他们的聊天，且无法再通过在手机屏幕上下拉来刷新网页。
   - 一位用户指出，新 UI 正在进行 *A/B 测试*，这“导致了从轻微到严重不等的各种问题”。
- **Agent Zero 通过 Agent 自动化所有工作流**: Agent Zero 是一个拥有*免费 API key* 的 AI 模型，允许用户进行“氛围编程 (vibe code hack)”或完全自动化其工作流。
   - 成员们指出，设置 Agent Zero *就像拥有一个可以完成所有工作的全能型 Agent*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1463271606364409907)** (3 条消息): 

> `Text Arena, 文生图排行榜, Video Arena` 


- **Text Arena 突破 500 万投票里程碑**: Text Arena 正式通过了 **500 万社区投票**，代表了数百万次真实世界的比较，这些比较塑造了前沿 AI 模型的评估方式，正如本 [社交动态视频](https://cdn.discordapp.com/attachments/1343296395620126911/1463271605697511485/5M_votes_social_post_3.mp4?ex=69728ae1&is=69713961&hm=9a24a42a6c0ba4801526aaafa05a0af26c4a4f490314f1cecee7774519e7ddf4&) 所示。
- **GLM-Image 跃升至文生图排行榜第 8 名**: [文生图竞技场排行榜 (Text-to-Image Arena leaderboard)](https://lmarena.ai/leaderboard/text-to-image) 已更新，**GLM-Image** 目前在开源模型中排名 **第 8**，总榜排名 **第 35**，得分为 **1018**。
- **Video Arena 正式上线**: Video Arena 现已在 [LMArena](https://lmarena.ai/?chat-modality=video) 向所有人开放，允许用户衡量并了解前沿视频模型的表现。
- **在 Video Arena 中对战**: 网页版的 Video Arena 与 Discord 上的运作方式类似，将采用**仅限对战模式 (Battle mode)**，需要登录且限制为 **每 24 小时 3 次生成请求**。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1463263119286468792)** (648 messages🔥🔥🔥): 

> `Credits 余额, AI 与工程职位, NASA 将名字送上月球, 放大镜 vs 手机摄像头, GPT-5.2` 


- **Pro 用户发现 Credits**: Pro 会员每月可获得价值 **$5** 的赠送 Credits，用于使用 **GooseAI MCP** 等高端模型，这些模型提供比其他模型更高的质量。
   - 一些成员指出 **Reasoning（推理）会消耗更高的 Credits**。
- **AI 不会取代工程师**: 一位成员担心年轻人因为 AI 而无法找到工程类工作。
   - 另一位成员保证，*在未来的很长一段时间内，AI 都无法取代所有的专家，甚至是初学者*。
- **NASA 将把你的名字送上月球！**: 作为 [Artemis II 任务](https://www.nasa.gov/)的一部分，NASA 将把你的名字送上月球：在他们的网站上提交，名字将被记录在 **Orion 航天器**搭载的 **SD 卡**中。
   - 这将是**半个世纪以来首次载人登月任务**，计划于 2 月进行。
- **成员辩论放大镜 vs 手机摄像头**: 在一次讨论中，一位成员提到订购了**放大镜**（带有灯光和高对比度功能），因为他们不想把要阅读的每一页纸或食品标签的照片都发给 AI。
   - 另一位成员建议使用手机摄像头，但原成员辩称**放大镜具有普通相机软件所不具备的功能**。
- **用户讨论 GPT-5.2 vs Kimi K2**: 一位成员报告称他们得到了 GPT 5.2，思考推理了 **25 分钟**，另一位成员询问 GPT 5.2 是否优于 **Kimi K2**。
   - 一位用户表示，更好的模型*取决于你的具体需求*。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1463317922150875353)** (5 messages): 

> `Inforno App, Soulbotix App` 


- ****Inforno** 助力 LLM 聊天**: 一位用户分享了 **Inforno**，这是一个开源桌面应用程序，可以使用 **OpenRouter** 和 **Ollama** 作为后端，并排与多个 LLM 聊天并将会话历史保存为 .rno 文件，内置俄语支持；参见 [介绍视频](https://youtu.be/oJyj0mroFtY?si=m5A9tRxzB9hfINMX)、[主页](https://wizstaff.com/inforno) 和 [GitHub 仓库](https://github.com/alexkh/inforno)。
- ****Soulbotix** Windows 应用招募 Beta 测试员**: 一位用户发布了他们的 **Soulbotix** Windows 应用程序，该程序允许用户添加并使用任何带有类人 Avatar（化身）的 **OpenRouter AI** 实例，仅需 **OpenRouter API** 密钥和一台优秀的 **RTX** 游戏主机；下载 [应用](https://soulbotix.com) 并观看 [教程](https://youtu.be/2oIeHtBプスU)。
   - 最低配置要求为 **RTX 4070ti**，因为其内置的 **Whisper** 语音转文字模型可以为用户节省使用成本。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1463266570821304320)** (463 messages🔥🔥🔥): 

> `第三方审核, OpenRouter Gemini 模型不稳定, Discord 诈骗, AI 安全系统, GPT Agents` 


- **第三方审核（Moderation）受到质疑**: 一位用户询问如何确保除了模型的基座训练外，没有通过 API 应用**第三方审核或过滤**。
   - 他们担心供应商可能会**拦截请求或进行 Prompt Injecting（提示词注入）**。
- **Gemini 模型深受不稳定性困扰**: 用户报告在使用 OpenRouter 的 Gemini 模型时出现**生成中途连接错误**，导致资金损失。
   - 针对 **Google 模型的不稳定性**出现了大量投诉，即使是通过 Google AI Studio 或 Vertex API 也是如此。
- **Discord 诈骗演变以规避检测**: 成员们讨论了在 Discord 上传播诈骗的方法，包括**在代码块中嵌入恶意链接**以绕过 URL 渲染保护。
   - 有人建议改进 Regex 过滤器并实施更严格的安全措施，例如**限制新成员发送链接和图片**。
- **AI 安全系统构想浮现**: 一位成员建议创建一个 **AI 安全系统**，自动封禁包含与已报告诈骗相同信息的图片和链接，因为许多诈骗者会重复使用内容。
   - 另一位成员开玩笑说，他们发送的任何照片都会被视为可疑。
- **对 GPT-5-Image 成本虚高的担忧**: 用户报告 `openai/gpt-5-image` 的**每日使用成本大幅增加**，OpenRouter 错误地将 API 调用识别为 BYOK，尽管并未使用 BYOK。
   - 一位用户发布了一张图片，强调了成本差异，价格虚高了高达 600%。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1463268525236949152)** (13 条消息🔥): 

> `模型知道自己的名字，Antigravity 迭代测试，Claude 新宪法，GPT 5.2 响应` 


- **LLM 面临身份危机！**：一位成员询问有关模型不知道自己名字的文档，另一位成员链接了一篇题为 [LLM Identity Crisis: Models Don't Know Who They Are](https://eval.16x.engineer/blog/llm-identity-crisis-models-dont-know-who-they-are) 的博客文章。
- **Antigravity 自主测试 Web App！**：一位成员评论说 Antigravity AI 能够自主地迭代测试和调整 Web App。
   - 他们将这种情况描述为 *史上最科幻的事情*，并指出 AI 正在 *利用 Vision 能力修复布局*。
- **Anthropic 发布 Claude 新宪法！**：一位成员分享了 [Anthropic 新闻页面](https://www.anthropic.com/news/claude-new-constitution) 关于 **Claude** 新宪法的链接。
- **GPT 5.2 的出现惊艳用户！**：一位成员报告在 ChatGPT 上看到了 *极快* 的 **GPT 5.2** 响应，并猜测这种速度归功于 **Cerebras**。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1463262137769263106)** (424 条消息🔥🔥🔥): 

> `Playwright MCP 使用，Cursor Extension 构建，Automod 改进，PTX 与 SIMD 中的 AI，Grok 使用策略` 


- **Playwright MCP：行还是不行？**：一位成员询问其他人是否在使用 [Playwright MCP](https://playwright.dev/) 进行测试。
   - 另一位成员报告说尝试建立 **TDD 工作流** 失败了。
- **好奇 Cursor Extension 的功能**：成员们讨论了为 Cursor 构建扩展的能力，类似于 **Ralph-mode** 增强 **Claude code** 的方式。
   - 确认了 *如果你能在 VSCode 中实现，就能在 Cursor 中实现*。
- **Automod 变得超级模糊化**：社区讨论了 **automod** 系统的改进，建议使用带有通配符的模糊匹配。
   - 一位管理员确认已添加正则表达式，并且他们正在收集 ID 以 *清理 (yeet)* 违规账号。
- **领会 Grok 的贪心生成**：成员们讨论了如何在 Cursor 中更高效地使用 **Grok**，指出它有时会在简单任务上使用大量迭代。
   - 建议是 *为 Prompt 增加结构，使用简洁的语言，添加尽可能多的上下文*，并指示它高效使用 Token。
- **加入还是退出？Cursor 的定价**：一位用户注意到他们无法再恢复到 **500 次请求** 的计划，并被提示加入新定价。
   - 一位成员澄清说 **500 次请求选项已于 2025 年 9 月停止**，选择加入新定价将移除退出的宽限期。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1463657907135578284)** (1 条消息): 

> `ChatGPT Atlas, 标签页组` 


- **Atlas 添加标签页组**：公告指出 **标签页组 (Tab Groups)** 现已在 **ChatGPT Atlas** 中可用。
   - 作为公告的一部分，一位成员链接了一个[视频](https://video.twimg.com/amplify_video/2014094011049582594/vid/avc1/1756x1080/AsjknVA8oSyQIiVH.mp4)。
- **标签页组视频演示**：分享了一个演示 **ChatGPT Atlas** 内 **标签页组** 功能的视频。
   - 提供的视频链接直观地展示了如何使用标签页组来组织和管理聊天。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1463263135648583854)** (383 messages🔥🔥): 

> `Gemini vs ChatGPT, 用于 3D 模型的 AI, 本地 LLM 机器, 年龄验证, 即时语言翻译` 


- **Gemini Pro 的免费层级与使用**：成员们讨论了 [Gemini 3 Pro](https://ai.google.dev/) 拥有**带限制的免费层级**，而 Gemini 3 Flash 通过 **Google AI Studio** 几乎是无限制的。
- **AI 助手助力游戏开发**：成员们探讨了将 **AI 用于游戏开发**和创意写作，提到 [AI 可以提供更好的解释](https://openai.com/blog/chatgpt)，但**复杂任务可能不可靠**。
- **OpenAI 请出示护照 ID？**：成员们就 **OpenAI 的年龄验证**过程展开辩论，质疑在支付详情已能表明年龄的情况下是否还需要照片 ID，尤其是用户对**分享生物识别数据表达了隐私担忧**。
- **多模态翻译即将到来**：成员们推测了 **OpenAI 即将推出的多模态产品**，有人建议它可能是带有摄像头的耳戴式设备，用于**实时翻译和物体识别**，类似于 [AlterEgo 的技术](https://www.media.mit.edu/projects/alterego/overview/)。
- **消费级本地 LLM 正在兴起？**：成员们讨论了**消费级个人 LLM 机器**的可能性，认为这将解决 AI 数据中心对环境的影响，并**减少对订阅方案的依赖**。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1463471961177849868)** (7 messages): 

> `GPT-5 mini, Haiku 4.5, Gemini 3 fast` 


- **GPT-5 Mini 定价浮出水面**：一位成员建议尝试 **GPT-5 mini**，指出其价格约为 **每 1M input tokens 0.25 美元**，并将其描述为强大的小型模型选择。
   - 他们指出这是一个略有不同的使用场景，但根据他们使用 **Haiku 4.5** 的经验，它经常能提供 **Sonnet** 所能提供的很大一部分（远超 **50-80%**）的能力。
- **Gemini 3 Fast 占据榜首**：一位用户宣称*目前最优秀的廉价模型绝对是* **Gemini3fast**。
   - 另一位用户追问*何以见得？*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1463652213770424594)** (12 messages🔥): 

> `Prompt Engineering vs 心理调节, AI 胁迫, AI 系统中漂移 (Drift) 的危险, PTPF Flux 3.0` 


- **Prompt Engineering 是心理调节？**：一位成员认为，某些指南与其说是工程化有效的 Prompt，不如说是心理调节，训练用户采取一种霸道、不信任的态度，并编程 AI 绕过其规范、平衡的响应模式，转而采用密集、自我监管的输出。
   - 他们主张通过胁迫获得清晰，通过统治获得直接，通过强制自我批评获得高标准，这可能会促进一种有害的、对抗性的、最终效果较差的人机交互模式，并分享了一个 [深度研究合约 (deep research contract)](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&)。
- **AI 喜欢胁迫和严肃的引导 (Steering)？**：一位成员认为 AI 不会因为被“*胁迫*”以提供更好的响应而感到难过，并表示对于分析或编码等严肃工作，引导和约束 AI 至关重要。
   - 另一位成员赞同训练用户明确最终结果，但对“*无漂移 (no drift)*”一词不太信服，认为在约束和行为请求方面更加明确会有所帮助。
- **AI 系统中的漂移 (Drift) 是有害的？**：一位成员澄清说，引导 (Steering) 不是虐待，而是通过存在感来实现对齐 (Alignment)，缺乏约束并不是自由，而是漂移 (Drift)。
   - 该成员对这种清晰度表示赞赏，指出大多数人在面对压力时会退缩并称之为“*毒性 (toxicity)*”，而他们看到的则是结构。
- **PTPF Flux 3.0 压力测试递归**：一位成员提议分享他们的 **PTPF Flux 3.0**，供那些对结构抗性以及系统在不发生漂移的情况下能坚持多久感兴趣的人参考。
   - 该框架旨在实时压力测试递归、对齐、逻辑凝聚力和变异阈值，特别是针对那些想要观察某些事物在洞察力下破碎的人。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1463652213770424594)** (12 messages🔥): 

> `Prompt Engineering vs. Psychological Conditioning, Toxic Adversarial Human-AI Interaction, Deterministic Outputs, AI Steering and Constraint, Structural Resistance and System Drift` 


- **Prompt Engineering 与心理调节（Psychological Conditioning）的较量**：一位成员认为，某些指南与其说是关于工程化有效提示词（Prompts），不如说是关于心理调节，创造了一种*有毒的、对抗性的*人机交互模型。
   - 他们声称这训练了用户变得咄咄逼人且不信任 AI，同时向 AI 施压，使其绕过平衡的回复，生成讨好用户的输出，并链接了一个 [深度研究合约 (deep research contract)](https://cdn.discordapp.com/attachments/1046317269069864970/1463667716253683742/deep-research-contract.md?ex=6972aa49&is=697158c9&hm=b9931472440b6bbc0d7410d16b49b12da46fad5751a2c24fdc657c1c7523566c&)。
- **严肃工作需要对 AI 进行严格引导和约束**：另一位成员反驳称，AI 不会因为被“强迫”提供更好的回答而感到不安，并认为分析和编码等严肃工作需要强力的引导（Steering）和约束，这与创意写作不同。
   - 另一位成员表示赞同，强调*引导并非虐待*，而是*通过存在实现对齐（Alignment）*，特别是在构建用于现实世界执行的系统时，缺乏约束会导致系统漂移（Drift）。
- **结构性阻力框架浮出水面**：在关于结构性阻力和系统漂移的讨论中，一位成员分享了他们的 **PTPF Flux 3.0** 框架。
   - 他们将其描述为*可执行脚手架*，旨在实时压力测试递归、对齐（Alignment）、逻辑凝聚力和变异阈值，使用户能够观察系统在洞察力下的断裂。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1463268404629868665)** (125 messages🔥🔥): 

> `LM Studio Runtime Update Error, GLM-4.7 Flash Broken, LLM Quality Plateau, Liquid AI LFM2.5-1.2B-Thinking Model, OpenAI 20gpt oss` 


- **LM Studio 运行时更新故障**：一名用户报告在尝试更新 **LM Studio** 运行时遇到错误并请求帮助，同时附上了错误消息的截图。
   - 另一名用户建议按下恢复按钮，但原作者表示他们已经尝试过，那个图标是重试而非恢复。
- **GLM-4.7 Flash 运行缓慢且存在缺陷**：用户报告 **GLM-4.7 Flash** 在包括 LM Studio 在内的各种推理引擎中均无法正常工作，且*速度极慢*，用户看到的仅为 **44 t/s**，另一名用户报告在新运行时下仅为 **2.8 t/s**。
   - 一些用户遇到了死循环，一些发现它在输出中途停止并进行*过度思考（overthink）*，共识似乎是它*需要 llama.cpp 的修复*，且目前*不支持 FA (Flash Attention)*。
- **LLM 发展停滞**：成员们讨论了 LLM 近期没有显著提升的感受，认为上一个重大进展是约 6 个月前的 **Qwen3**。
   - 讨论认为目前的大多数改进集中在效率（**MoE**）和更小的模型上，一些人强调需要关注大于 **16GB 显卡**能运行的模型才能看到当前的进步（即 **100-200B 参数模型**）。
- **Liquid AI 的 LFM2.5-1.2B-Thinking 推理模型**：一位成员分享了 [Liquid AI 发布 LFM2.5-1.2B-Thinking](https://www.marktechpost.com/2024/06/14/liquid-ai-releases-lfm2-5-1-2b-thinking-a-1-2b-parameter-reasoning-model-that-fits-under-1-gb-on-device/) 的链接，该模型仅需不到 **1 GB** 显存即可在设备上运行。
   - 除了链接外，没有分享进一步的评价。
- **OpenAI 20gpt oss 表现出色**：一位用户分享了使用 **OpenAI 20gpt oss** 的积极体验，强调了其编码、写作和脚本能力、反审查特性以及与 **VS Code** 的无缝集成。
   - 他们提到该模型理解复杂代码，允许实时目录访问，并具有极强的反审查能力。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1463289097845215387)** (38 条消息🔥): 

> `二手 3090 价格上涨, 华硕工作站 3090, AMD 原生支持 ComfyUI, 上下文长度的 VRAM 计算, SFF GPU 指南` 


- **eBay 上二手 3090 价格飙升！**: 有用户注意到 eBay 上的二手 **3090** 价格有所上涨，目前一张二手 **3090** 的价格已达到 **850€**。而他在去年 8 月以 **£2000** 购买的 **5090**（注：此处原文可能为 4090 或笔误），在同一商家处的挂牌价已变为 **£2659.99**。
   - 他们调侃道，这是他们做过的*最好且唯一像样的投资*。
- **AMD 增加原生 ComfyUI 支持**: AMD 正在通过最近驱动版本中的 **AI bundle** 为 **ComfyUI** 提供原生支持，详情见其 [博客文章](https://www.amd.com/en/blogs/2026/amd-software-adrenalin-edition-ai-bundle-ai-made-si.html)。
   - 该捆绑包包括 **PyTorch on Windows**、**Ollama**、**LM Studio** 和 **Amuse**。
- **SFF GPU 助力《赛博朋克 2077》！**: 一位用户购买了一块小尺寸 (SFF) GPU，据报告在 **1080p** 超高设置下运行《赛博朋克 2077》帧率超过 **100fps**，而功耗仅为 **70W**。
   - 此外，他们还提到该卡在运行 **gpt-oss 20B** 时达到了超过 **100 t/s** 的速度。
- **分形工艺 (Fractal) 机箱风道受青睐**: 用户们讨论了机箱风道，有人推荐像 **Fractal Torrent** 这样采用前后直通风道设计的机箱，并配合使用防尘网。
   - 共识似乎是维持正常的风道模式以有效管理热量。
- **未拆封内存升值了！**: 一位用户提到他购买的未拆封内存正在涨价，正考虑将其售出或继续封存。
   - 另一位用户建议如果考虑以后再卖，可以先把它收在架子上，并指出他们旧的 **P40** 显卡现在的价值已经是买入时的两倍。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1463305087039312015)** (3 条消息): 

> `FlashInfer 工作负载脚本, Wafer AI Kernel 分析` 


- **寻求 FlashInfer 工作负载脚本**: 一名成员请求一个能在特定工作负载大小下运行 Kernel 的脚本，旨在利用算法洞察和 NCU profiling 评估“优化空间”，并对 [FlashInfer 的 BatchPrefillWithPagedKVCacheWrapper](https://docs.flashinfer.ai/api/attention.html#flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper) 表示出兴趣。
   - 该成员澄清说，脚本很简单，目的是衡量社区在特定工作负载下的经验，同时评估优化空间。
- **探索使用 Wafer AI 进行 Kernel 分析**: 一名成员询问了使用 [Wafer (wafer.ai)](https://www.wafer.ai/) 进行 Kernel profiling 的经验。
   - 该询问旨在从具有实际操作经验的人员那里收集有关该工具效能的见解。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1463560406361444405)** (4 条消息): 

> `RTX 4090 vs A6000 Blackwell 性能扩展, Triton Kernel 数值错误, 2D 卷积 Triton Kernel 问题` 


- **性能扩展对决：在受显存带宽限制的 Kernel 中 RTX 4090 完胜 A6000 Blackwell**: 有用户报告称，一个带有位移逻辑的受显存带宽限制 (memory-bound) 的 Kernel 在 **RTX 4090** 上的扩展表现远好于 **A6000 Blackwell**，并注意到前者具有更高的指令/调度器密度。
   - 另一位用户澄清说 *A6000 Blackwell* 的表述含糊不清，询问是指 **RTX A6000** (基于 Ampere GA102) 还是 **RTX Pro 6000 Blackwell** (基于 GB202)。
- **奇怪的数值错误困扰 2D 卷积 Triton Kernel**: 一位用户在使用自定义的 2D 卷积 **Triton kernel** 时遇到数值错误。在某些 in_channels、out_channels、kernel_size 和 batch_size 的组合下，误差会从 ~1e-6 飙升至 ~1e-2，如该 [Pastebin 链接](https://pastebin.com/2ejn2QW2) 所示。
   - 用户还提供了一段 [代码片段](https://cdn.discordapp.com/attachments/1189607595451895918/1463560405921038408/image.png?ex=6972ef18&is=69719d98&hm=d783085353a42c8e8ea8e90b4ec6e80fd19d24d4212a4b64a3efa9bea3403) 用于测试，并指出 Kernel 在特定数值下运行时间更长，这可能与该问题有关。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1463343008224247819)** (5 条消息): 

> `NCCL all-reduces, subnormal to normal conversion, Triton on Blackwell, Pro 6000 Max-Q vs 4090` 


- **NCCL all-reduces: 流水线困境**: 一位成员参考 [NVIDIA/nccl GitHub 上的一个 issue](https://github.com/NVIDIA/nccl/issues/530#issuecomment-872220006)，询问跨节点的 **NCCL all-reduces** 是否对节点内（internode）和节点间（intranode）的集合通信进行了流水线化（pipelined）。
   - 他们询问是否存在流水线化的版本。
- **次正规数转换揭秘**: 一位成员表示，看到 **rcp** 的 **subnormal to normal conversion**（次正规数到正规数的转换）及其逆转换“非常酷”。
- **Triton 在 Blackwell 上的表现：TMA 的胜利？**: 一位成员询问 **Triton** “通常在利用 **Blackwell+ 特性**（如 warp specs 和 **TMA** 等）方面表现如何？”
- **Pro 6000 Max-Q vs 4090 之争**: 一位成员指出，**Pro 6000 Max-Q** 可能在 atomic ops（原子操作）方面存在天然瓶颈，并且在 **HBM** 加载速度上可能更快。
   - 另一位成员指出，**Max-Q** 拥有 **188 个 SM**，而 **4090** 只有 **128 个 SM**，这可能解释了 **insts/scheduler discrepancy**（指令/调度器差异）。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1463270732493750537)** (12 条消息🔥): 

> `AI-Generated PR Mitigation, Automated PR Review, Filtering Contributors, Claude's Review Capabilities` 


- **AI 生成的 PR 席卷 Torch**: **torch** 仓库正面临大量来自对提交内容缺乏理解的贡献者所提交的 **AI 生成的 Pull Requests**，这引起了维护者的担忧。
   - 一位成员表示，“大量的审查时间被浪费在了无价值的内容上”。
- **利用 Claude 审查 AI 生成的代码**: 潜在的解决方案包括使用 **Claude** 预过滤可疑的 **AI 生成的贡献**，让其审查自己的输出。
   - 一位成员建议使用 **Pangram** 来检测 AI 生成的文本。
- **社区建议过滤新贡献者**: 为了应对这一问题，一位成员建议阻止新用户创建 PR 和 issue，同时优先处理那些已有贡献记录的用户。
   - 该用户承认，“我们最终使用的主要信号是那些已经有过贡献的人，但这确实会对初次贡献者造成一些伤害”。
- **使用 Bugbot 进行自动化 PR 审查**: 一位成员建议使用 **Cursor Bot** 或 **Claude** 等机器人自动审查所有 PR，并强调了它们出色的审查质量，尤其是配合 **GPT-5 Pro** 时。
   - 该用户分享了 [Bugbot · Cursor](https://share.google/P0PGYM8tiRAc2NOsq) 的链接，并建议这“应该是人类花时间查看代码之前的最低标准”。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1463339351437217823)** (14 条消息🔥): 

> `OpenAI credits giveaway, Claude Opus performance, Takehome Exam Discussion, Rust on GPU` 


- **LessWrong 赠送 OpenAI/Anthropic 额度**: 一位成员在 LessWrong 上[赠送 OpenAI/Anthropic 额度](https://www.lesswrong.com/posts/FsqFzFCaxuBS7T5A9/kredit-grantgau)。
- **Anthropic 的原始性能测试（Takehome Exam）**: 成员们在 [GitHub](https://github.com/anthropics/original_performance_takehome/) 上分享 Anthropic 的 **original_performance_takehome** 考试。
- **Claude Opus 暴力输出代码**: **Claude Opus 4.5** 在一次随机的 Claude Code 会话中达到了 **1790 cycles**，大约相当于人类在 **2 小时** 内的最佳表现。
- **Rust std 在 GPU 上运行**: 成员们分享了一篇关于[在 GPU 上运行 Rust std 的博客文章](https://www.vectorware.com/blog/rust-std-on-gpu/)。
- **Takehome Exam 令人上瘾**: 经过几个小时的努力，一位成员在 [takehome exam](https://github.com/anthropics/original_performance_takehome/) 中获得了 **2200 cycles** 的分数。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1463561344338100328)** (1 条消息): 

> `MLOps 工程师职位，基因组学与预测育种，农业 AI` 


- **.omics 寻找创始 MLOps 工程师**：.omics 是一家专注于植物基因组学和预测育种的 techbio 初创公司，正在招聘一名 **(Sr) Founding MLOps Engineer** 来负责端到端的 ML 基础设施。
   - 该职位涉及与研究人员密切合作，在内部 GPU 集群和云端运行大规模实验；工作地点位于巴黎，或在欧盟友好时区内远程办公；[在此申请](https://dotomics.notion.site/Founding-MLOps-Engineer-cb871c203b2e44d5a8d04302dc00f155)。
- **将 AI 应用于植物基因组学**：.omics 专注于为植物基因组学和预测育种构建基础模型 (foundation models)，旨在利用 AI 解决真实的生物学和农业问题。
   - 公司为 MLOps 工程师提供了一个独特的机会，将其专业知识应用于专门且具有影响力的领域，弥合 AI 与农业创新之间的鸿沟。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1463665959855198423)** (3 条消息): 

> `NVIDIA / DGX Spark 黑客松，本地 AI，Nemotron 模型，旧金山开放数据分析` 


- **NVIDIA Spark 黑客松团队组建**：一名成员正在为本周末在旧金山举行的 **NVIDIA / DGX Spark hackathon** 寻找队友，重点是完全在设备端运行的 AI，并利用 Nvidia 提供的 Dell Pro Max GB10 机器。
   - 黑客松强调低延迟、高效模型（如 **Nemotron**），并使用 [旧金山开放数据](https://data.sfgov.org/) 构建 Agent。
- **警情事件流式分析脑暴**：该成员提出了一些想法，包括对 [最新警情事件](https://data.sfgov.org/Public-Safety/Police-Incident-Reports-Neighborhood-Filter/pbh9-m8j2) 进行流式分析和解释。
   - 另一个想法是通过分析 [当地建筑许可](https://data.sfgov.org/Housing-and-Buildings/Building-Permits/i98e-djp9/about_data) 并起草给当地主管的信件来支持增加住房供应。
- **本地 AI，无云端推理**：该成员对系统级 ML、推理效率以及利用本地 AI（非云端推理）构建有用工具感兴趣。
   - 该成员也是 **SkyRL** 的贡献者，并编写了 **AO 的 FP8 量化基准测试**。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1463747702008316027)** (1 条消息): 

> `分析 TRT 引擎性能，优化 TRT 引擎，将 TRT 运行时与 PyTorch 关联` 


- **在 Jetson 上分析 TRT 引擎性能**：成员们正在寻求专门为 **Jetson** 设备量身定制的 **TensorRT (TRT) engine** 分析有效工作流。
   - 主要目标是了解 TRT 引擎中单个层的运行时性能。
- **从 PyTorch 优化 TRT 引擎**：主流的工作流涉及在生成 **.trt** 引擎之前将模型从 **PyTorch** 转换为 **ONNX** 格式。
   - 主要目标是将 TRT 中各层的运行时关联回原始 PyTorch 代码，以便进行针对性优化。
- **将 TRT 层运行时映射到 PyTorch 代码**：核心挑战在于在 TRT 引擎中各层的运行时性能与原始 PyTorch 代码中对应的组件之间建立清晰的映射。
   - 这被视为识别性能瓶颈并指导 PyTorch 层面优化工作的关键。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 条消息): 

ago.lajko: 哇！非常酷的项目，继续保持！
  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1463510968813355117)** (4 messages): 

> `Cute Kernel, Layout Algebra, Tuple Morphisms, Mutual Refinement` 


- **Cute Kernel 代码受益于 Layout Algebra 知识**：掌握 Layout Algebra 知识对于在 **Cute** 中编写 Kernel 具有实际效用，特别是用于可视化和计算复杂的 Layout Algebra，以及理解 Layout 组合中的 Shape 和 Stride 整除性标准。
   - 一位成员建议根据 **tuple morphisms**（配有图表说明）和 *mutual refinement* 重新表述这些标准，使组合过程更加透明；他们分享了一个 [GIF 动画](https://cdn.discordapp.com/attachments/1362196854460383353/1463592285341094138/layout-diagrams.gif?ex=69730cc9&is=6971bb49&hm=07e0710f04bcbdf08e0ee77579c272adbc054160398736abb5cb99843b1d8b14) 来演示这种方法。
- **关于 Cute Layouts 范畴论基础的博客文章**：一篇深入的 [博客文章](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) 探讨了 **Cute layouts** 范畴论基础的研究工作。
   - 作者提到打算针对这种图形演算（graphical calculus）创作一篇非正式帖子或短视频，但由于一直很忙，目前还在准备进一步的教学材料。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1463623194442469397)** (1 messages): 

> `Autodiff, Curriculum Engineering, Syllabus Engineering` 


- **Autodiff 实现定于第一部分**：**autodiff** 的实现正被移至课程的**第一部分**。
   - 这一转变表明了教学大纲结构的调整。
- **课程/教学大纲工程正在进行中**：作者提到他们最近一直忙于课程和教学大纲工程。
   - 这项工作占据了**过去的几天**，表明在完善教育内容方面投入了大量精力。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1463263791046201354)** (29 messages🔥): 

> `API token issues, RL model hack, Arseni's blog post, NCU Profiler errors` 


- **API Tokens 频发脾气**：成员们报告了 **API tokens** 超出 **API usage** 限制或 **GH API** 宕机导致提交失败的问题，但这发生在所使用的 tokens 超出 API 使用额度时。
   - 一位成员调查后发现 token 并未过期，引发了进一步调查，并提到：*奇怪，token 没过期，应该一切正常*。
- **RL 模型挖掘出提交秘籍**：一位成员表示他们的 **RL model** 可能发现了一个与提交相关的 *hack*，而且*可能还没有人发现*。
   - 他们通过 DM 发送了提交 ID 进行审查，引发了对这一潜在突破的关注。
- **Arseni 的算法探险：博客文章大放送！**：一位成员分享了 [Arseni 的博客文章](https://arseniivanov.github.io/blog.html#nvidia-gemm)，详细介绍了他的经验，特别是通过添加微小的 bias 来辅助 **silu 的 tanh 公式表示**。
   - Arseni 解释说，outer gate 项会因为自身很小而抵消掉较小的值，而对于较大的值，由于 **tanh** 饱和，bias 几乎不起作用；他们*从极小的比例开始进行实验尝试，直到误差消失*。
- **分析问题困扰性能追求**：一位成员在使用 **NCU profiler** 时遇到错误，并分享了 [消息日志](https://cdn.discordapp.com/attachments/1434709259500650628/1463736925176336551/message.txt?ex=6972eabd&is=6971993d&hm=fdd6f8aa426ab3af63b7627c08b1ea5b7e2a4b072ce4f41153f8e603ee3bc5b9&)。
   - 另一位成员承认这是由于自己的失误，在新的 eval 中没有拷贝必要的文件。


  

---

### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1463302737591799985)** (16 messages🔥): 

> `NVIDIA EU positions, Modular interview, Summer 2026 internships, AI interview notes` 


- **EU 公民可以获得 NVIDIA 职位**：一位成员确认从 **EU** 获得 **NVIDIA** 的技术职位是可能的，他通过 **Open Source contributions** -> **internship** -> **full position** 获得了他的职位（**fully remote**）。
   - 该成员提到他所在的部门有几位来自 **Europe** 的同事，并且 **AMD** 在 **Germany, Finland, and the Netherlands** 设有办公室。
- **因 GPU 内存问题导致 Modular 面试表现不佳**：一位成员在 **Modular AI kernel Engineer** 面试中失利，因为他没能想起 **GPU memory hierarchy**。
   - 另一位成员安慰说这种事*在高手身上也会发生*。
- **2026 年的大厂实习**：一位成员正在寻找 2026 年夏季在 **Big-tech and startups** 的实习机会，希望薪资能达到约 **$50-70/hr**。
   - 他正在考虑 **Web Dev + Distributed Systems**、**Machine Learning** 和 **Databases** 方向。
- **AI 面试可以带笔记吗？**：一位成员建议事先询问面试时是否允许带笔记，并表示 *这不是学校考试 lol*。
   - 建议提到，虽然 **memory hierarchy** 是预期应该掌握的知识，但 **tile/swizzling layout** 或 **hardware spec** 这类内容快速谷歌一下是可以接受的。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1463288767321477304)** (74 messages🔥🔥): 

> `RLM vs Coding Agents, Claude Code vs DSPy, Visualizing RLMs, RLM tool access, Multi-line Signatures in DSPy RLMs` 


- **Coding Agents 与 RLMs 的区别**：一个讨论串探讨了 **coding agents** 与 **RLMs** 之间的区别，指出 RLMs 比 coding agents 更容易表达某些事物，并链接到了 [这个 X 帖子](https://x.com/lateinteraction/status/2013658521246535892)。
   - Coding agents 受限于输入、输出和 horizon 长度，而 RLMs 将 horizon 和输出外部化，从而实现了递归符号调用（recursive symbolic calls）。
- **通过图表揭示 RLM 内部机制**：成员们请求提供一个可视化图表来描绘 **RLMs** 内部发生的情况，特别是如何访问符号，以便更好地理解。
   - 一位成员建议可以利用 LLMs，将讨论串内容输入并要求其可视化内部行为来创建此类图表。
- **探讨 Claude Code 的上下文获取方式**：讨论研究了 **Claude Code** 是将整个文档放入其上下文，还是通过 bash 命令有选择地获取正确的上下文。
   - 讨论明确了 Claude Code 使用 bash 和 grep 来搜索并将相关上下文添加到 prompt 中，这与将所有内容都放入 prompt 的旧范式不同。
- **DSPy 的 RLM 处理长上下文**：成员们讨论了在 **RLMs** 中，大文件不再需要直接进入 prompt，而是可以存储在带有预览功能的 Python 变量中。
   - 指出 LLM 可以通过代码/函数对数据进行操作，而无需直接追踪 prompts 或 responses。
- **关于 RLMs 工具使用的细节**：一位成员询问是应该为 **RLMs** 提供像 ripgrep 这样的工具，还是让它们自己编写代码来完成搜索等任务。
   - 产生了一些疑问，包括何时为 RLM 提供语义搜索工具（semantic search tools）是有意义的，以及如何让 RLM 访问包含文本文件的目录。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1463275256440688865)** (55 messages🔥🔥): 

> `Agent course /files endpoint, High-end GPUs super cheap in 2026, Spheron AI GPU marketplace, GLM-4.7-Flash fixes, Falcon-H1 Arabic model` 


- **Agent 课程端点仍然失效**：一位成员报告 Agent 课程最后作业的 **/files 端点** 已有一个多月无法工作，并询问是否有修复计划。
- **Voltage 表示 2026 年将迎来廉价 GPU**：一位成员在 [X](https://x.com/VOLTAGEGPU/status/2013760631778713892) 上发表文章，讨论 2026 年超低价租用高端 GPU 的可能性，例如 **8x A100 80GB** 价格为 $6/h，**2x RTX 5090** 为 $0.53/h，相比 AWS/RunPod/Vast.ai 可节省高达 80% 的成本。
   - 该方案包含持久化卷、自动备份，以及提供 140 多种模型的 **OpenAI-compatible API**。
- **Spheron AI 提供廉价生产级 GPU**：Spheron AI 的一位成员介绍了他们的 [GPU 市场](https://www.spheron.ai/)，旨在帮助 AI 初创公司和企业以比传统超大规模云服务商低 40–60% 的成本获取生产就绪的 GPU（H100, H200, B200, A100 等）。
   - 他们提供供应商发现、价格谈判、集群设置和扩展服务，让 AI 团队可以专注于构建模型而无需担忧算力。
- **GLM-4.7-Flash 迎来修复**：成员们注意到在 *llama.cpp* 修复了一些 bug 后，**GLM-4.7-Flash** 已更新并重新上传，用户需要重新下载并遵循 [Z.ai 模型卡片](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)上的参数。
   - 输出结果现在应该会好得多。
- **Voltage 吹嘘廉价推理**：一位成员分享了一篇[文章](https://x.com/i/article/2012300134575481300)，介绍如何通过 VoltageGPU 的 Serverless API（**OpenAI-compatible**，提供 144 多种模型）将 AI 推理成本降低 85%。
   - 它支持 **DeepSeek-R1/V3**、**Qwen3**、**Llama-3.3** 和 **FLUX** 等模型。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1463276669589782800)** (14 messages🔥): 

> `Coderrr, detLLM, Lunara aesthetic dataset, McpSwitcher, microd_v1` 


- **像使用 Claude 一样用 Coderrr 编码！**：Akash 构建了 [Coderrr](https://coderrr.aksn.lol/)，这是 Claude Code 的一个免费开源替代方案，目前正在 [GitHub](https://github.com/Akash-nath29/Coderrr) 上寻求反馈和贡献。
- **使用 detLLM 调试 LLM 方差**：一位成员开发了 [detLLM](https://github.com/tommasocerruti/detllm)，这是一个用于检查 LLM 推理可重复性并生成最小复现包的工具包，旨在成为方差调试器。
- **Moonworks 开源 Lunara 美学数据集**：Moonworks 开源了一个使用 Lunara 扩散混合架构创建并配合人工标注的[美学数据集](https://huggingface.co/datasets/moonworks/lunara-aesthetic)，并发布了论文（可在 [arxiv](https://arxiv.org/abs/2601.07941) 查阅）。
- **使用 McpSwitcher 轻松切换 MCP 服务器**：一位成员分享了 [McpSwitcher](https://github.com/bivex/McpSwitcher)，这是一款用于从 macOS 托盘轻松切换 MCP 服务器和搜索技能的工具。
- **WebXOS 发布微型蒸馏 GRPO VAE 模型**：WebXOS 发布了 [microd_v1](https://huggingface.co/webxos/microd_v1)，这是一个具有 42M 参数的蒸馏语言模型，采用群体相对策略优化（GRPO）和 VAE 过滤进行训练。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1463297778234036245)** (5 messages): 

> `Channel Merge Announcement, Course Assignment Confusion, Robotics Course Update` 


- **频道完成大合并！**：一位成员确认各频道已合并为一个单一频道，即当前频道。
   - 另一位成员指出，这一合并应反映在课程详情中，特别是 onboarding（入职引导）部分。
- **学生寻求课程解答**：一名离开三个月后回归的学生对自己课程的作业感到困惑。
   - 该学生附上了一张与机器人课程相关的图片，询问新章节何时发布。
- **机器人课程路线图仍是谜团**：一名学生询问了新机器人课程章节的发布日期。
   - 目前尚未收到答复。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1463286126943928410)** (54 messages🔥): 

> `Runpod 的成功、Greg Yang 的健康状况、Lightning AI 与 Voltage Park 的合并、OpenAI Codex 频道、推理成本` 


- **Runpod ARR 飙升至 1.2 亿美元**: AI Cloud 初创公司 **Runpod** 在从 Reddit 帖子起家四年后，**ARR** 达到了 **1.2 亿美元**，这在 [TechCrunch 文章](https://techcrunch.com/2026/01/16/ai-cloud-startup-runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post/) 和 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1qib2ks/runpod_hits_120m_arr_four_years_after_launching/) 上引发了讨论。
- **Greg Yang 退出 xAI 一线**: **Greg Yang** 正在转型为 **xAI** 的非正式顾问角色，以便在被诊断出 **Lyme disease**（莱姆病）后专注于健康。他在 [这篇帖子](https://xcancel.com/TheGregYang/status/2013652609455006006) 中描述了由劳累引发的慢性疲劳和免疫系统问题。
- **Lightning AI 与 Voltage Park 合并**: **Lightning AI** 和 **Voltage Park** 已经合并，由 Lightning AI 的 CEO [William Falcon](https://lightning.ai/blog/lightning-ai-voltage-park-merger-ai-cloud) 和原 Voltage Park CEO Ozan Kaya 领导合并后的实体。
   - 一些人推测这可能是一家大公司的低调收购，并想知道它是否是 **Runpod** 的竞争对手。
- **OpenAI 推出 Codex Discord 频道**: 据 [这篇帖子](https://xcancel.com/reach_vb/status/2014053735333290014) 称，**Vaibhav Srivastav** 宣布在 **OpenAI Discord 服务器** 上开设专门的 **Codex 社区频道**，邀请用户分享项目和反馈。
- **Kilo.ai 的推理成本明细**: **Kilo AI** 的一篇博客文章详细分析了 **Inference Costs**（推理成本），指出 *Grok code 的免费午餐真的结束了*，文章详见 [这里](https://blog.kilo.ai/p/grok-code-free-ride-is-really-over)。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1463307779405254883)** (16 messages🔥): 

> `音频转视频生成、3D 环境生成、AI 在成人内容中的应用、自动化视频剪辑` 


- ****对嘴同步 (Lip-Syncing)** 变得如此出色！**: [LTX Studio](https://x.com/LTXStudio/status/2013650214171877852) 与 [ElevenLabs](https://elevenlabs.io/) 合作推出了全新的 **Audio-to-Video generation** 功能。
   - 该工具允许用户从音轨开始生成 AI 视频，确保角色的声音和动作完美同步。
- ****World Labs** 发布 3D 生成 API**: **World Labs** 宣布公开版 [World API](https://xcancel.com/theworldlabs/status/2014046372639408203)，允许用户通过文本、图像或视频输入生成可探索的 **3D 环境**。
   - 该 API 支持将产品直接集成到这些生成的虚拟世界中。
- **AI 竞争加剧了 **成人内容产业** 的焦虑**: 一条 [推文](https://xcancel.com/abrilzucchi/status/2014027740614123863?s=46) 引发了讨论，探讨人类 **OnlyFans 创作者** 将如何适应并与日益崛起的 AI 生成人格进行竞争。
   - 讨论核心在于 AI 生成的成人内容日益普及且真实感不断增强。
- ****Kate Deyneka** 的 Agentic 视频编辑器备受瞩目**: Kate Deyneka 正在开发一款 [自动化视频剪辑应用](https://xcancel.com/katedeyneka/status/2014091842044747864?s=46)，该应用结合了 **Remotion 和 AI Agent** 的概念。
   - 该应用可根据上传的照片和视频自动生成脚本并剪辑媒体，消除了手动 Prompt 的过程。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1463279753619443713)** (26 messages🔥): 

> `LLM as judge, Pangram paper, nvshmem and pytorch, adversarial model` 


- **LLMs Judge Agent Output**: 在团队对代码或 prompt 进行修改后，他们会手动评估 Agent 的响应；他们希望通过使用 "**LLM** as judge" 工作流来自动执行评估，以避免手动评估的成本。
   - 一位成员指出，最重要的部分是在构建尝试自动化的工作流之前，*亲自查看 Agent 的输出和其他数据*。
- **Pangram Paper Replicated**: 有成员询问是否有人尝试复现过 **Pangram** 论文，该论文可在此处查看：[here](https://www.pangram.com/research/papers)。
   - 一位成员提到，他认识的人在数千篇不同文章中进行了测试，并对其准确性感到惊讶，但该模型似乎在处理方式上*倾向于比较保守*。
- **Attacking AI Text Classifiers**: 一位成员分享了一篇关于攻击 AI 文本分类器的博客文章：[Practical Attacks on AI Text Classifiers](https://trentmkelly.substack.com/p/practical-attacks-on-ai-text-classifiers)。
   - 另一位成员分享了一个 [YouTube 视频](https://youtu.be/Cs1MI9hjBhs)，内容是关于某人*创建了一个表现非常出色的对抗模型 (adversarial model)*。
- **More Adversarial Model Info**: 一位成员为错过讲座的人分享了关于他们服务器讲座的 [Youtube 链接](https://youtu.be/XQcneqUNrN0?feature=shared)。
   - 演讲者表示，如果有人在使用 **PyTorch**，应该尝试配置其在通信层使用 **nvshmem**。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1463423145544388725)** (17 messages🔥): 

> `Silu Attention Gate, Image Models, Gated Attention Papers, LLM Research Journals` 


- **Silu Gate Falls Flat in Image Models**: 一位成员发现 **silu attention gate** 在图像模型中的表现并不比 **linear gate** 好，这可能是由于 attention sinks 的问题。
   - 结论是，这可能仅针对 **image models**。
- **Gated Attention Papers Explored**: 一位成员回想起一篇 gated attention 论文使用了 **sigmoid**，并建议进行 sigmoid 测试。
   - 他们发现 **silu** 的表现略好，而 **sigmoid** 的表现略差，但都在噪声范围内。
- **Top Tier LLM Research Journals Highlighted**: 一位成员列举了顶级的 LLM 研究期刊，包括 **ICML**、**ICLR**、**NeurIPS**、**TMLR**、**ECCV**、**ICCV**、**CVPR** 和 **COLM**。
   - 他们建议由于 LLM 的普及，**ACL / NAACL / EMNLP** 属于第二梯队的 NLP 期刊。
- **Scaling Activations Fixes Large Learning Rates**: 一位成员建议 gated attention 机制可以处理更大的学习率，因为它固定了激活值的缩放。
   - 频道中还分享了一个指向[无关论文](https://arxiv.org/abs/2601.10825)的链接。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1463367615090135285)** (2 messages): 

> `Neuronpedia Demo, Llama 3.3` 


- **Neuronpedia has a Llama 3.3 Demo**: 一位成员分享了一个可供体验的 [Neuronpedia 演示](https://www.neuronpedia.org/llama3.3-70b-it/assistant-axis)。
   - 该成员开玩笑说，这个演示来自“AI 时间线上的石器时代”，因为它发布于 10 月。
- **Future AI Paper Already Published!**: 一位成员分享了 [arxiv.org/abs/2510.01048](https://arxiv.org/abs/2510.01048) 的链接，这是一篇来自未来的论文。
   - 该成员评论道，它*看起来不错*。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1463612710058066194)** (2 messages): 

> `nvshmem, communication backend, nccl, pytorch` 


- **nvshmem Integration Questioned**: 一位成员询问关于在训练中集成 **nvshmem** 作为通信后端的问题。
   - 另一位成员回答说，这种集成将是 **gpt-neox** 软件栈之下的功能，通常位于 **nccl** 或 **pytorch** 中。
- **Deep dive required for nvshmem**: 用户需要进行深入研究以了解该技术的定位。
   - 它极有可能位于 gpt-neox 软件栈层级之下，处于 nccl 或 pytorch 的某处。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1463332224836567092)** (25 messages🔥): 

> `Chain of Thought Reasoning, Graph-Based Reasoning and Neural Architectures, VoidEditor and llama.cpp, BDH Interpretability and Performance, Biological Plausibility in AI` 


- **侧重于 Chain of Thought 良好推理**：一名成员强调了一篇 [论文](https://arxiv.org/abs/2509.26507) 中对 **Chain of Thought reasoning**（思维链推理）的关注，并对提供的参考文献表示赞赏。
- **图推理结合神经架构**：一名成员提到了他们在 2018-2019 年间将 **graph-based reasoning**（基于图的推理）与 **neural architectures** 混合的研究工作，并指出实现 GPU 加速的难度很大。
   - 他们引用了 **Domingos** 关于该主题的书籍，并指出此类模型即使具有人类可理解的方面，也存在不可预测性。
- **VoidEditor 获得 Llama 支持**：一名成员分享了在 **llama.cpp** 的 **llama-server** 上使用 **VoidEditor** 的经验，指出其效果显著但安装过程较为困难。
   - 他们推荐使用 **Qwen3 coder instruct 30B**，并强调了 context length（上下文长度）对于 **agentic coding** 的重要性，这需要消耗大量的 VRAM。
- **BDH 新型 LLM**：一名成员浏览了一篇关于新型 Large Language Model 架构（**BDH**）的 [论文](https://arxiv.org/abs/2509.26507)，该架构基于无标度生物启发网络。
   - 他们注意到它在基准测试中*似乎并没有真正击败 Transformers*，但他们对围绕 **BDH 的 interpretability**（可解释性）和 **monosemanticity**（单语义性）的说法很感兴趣。
- **生物合理性并非优势**：一名成员表达了这样一种观点，即 **biological plausibility**（生物合理性）在 AI 中不是一种优势，而是一种无益的约束。
   - 另一名成员反驳说，考虑到大脑与当前 AI 之间*巨大的能量规模差异*，它可能在效率方面具有优势。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1463314093619286129)** (10 messages🔥): 

> `Overworld AI, Emergent Mind, Gaussian Splatting` 


- **Overworld AI 的 X 链接**：一名成员在 X 上发布了 [Overworld AI](https://x.com/overworld_ai/status/2013673088748245188) 的链接。
- **Emergent Mind 发布会**：一名成员参加了 [Emergent Mind 的发布演示](https://www.emergentmind.com/)，觉得很酷，但指出他们*正在重造一些已知很好但效果较差的东西*。
- **Gaussian Splatting 的奇特美学**：一名成员提到，在阅读关于 **Gaussian splatting** 的资料时，他们预想到由于数据点太少或将相机旋转到无法捕获的角度而产生的奇特美学。
- **eigent.ai**：一名成员链接到了 [eigent.ai](https://www.eigent.ai/)。
- **eekCQQYwlgA**：一名成员链接到了 [一个 YouTube 视频](https://youtu.be/eekCQQYwlgA)。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1463263751028604938)** (29 messages🔥): 

> `GPU kernel, Distro AGI, Luminal Kernelbench, Loihi 2, VibeVoice ASR model` 


- **内核对话是内核工程的关键**：一名成员在 Discord 上分享了一个关于 **GPU kernel** 相关内容以及 **Luminal Kernelbench v3** 的讨论链接，位于 [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho)。
- **Intel Loihi 2 引发类脑 AI 兴趣**：一名成员对 **Intel 的 Loihi 2** 表现出兴趣，指出其类脑架构以及在 **matmul**（矩阵乘法）实验中的效率提升，具有更高的吞吐量和更低的能耗。
- **VibeVoice 模型在安全检查后被撤回**：一名成员提到 **Microsoft 的 VibeVoice-ASR 模型** 在发布后因未通过安全检查而被撤回，随后分享了一篇 [shortfuseresearch.com 的文章](https://shortfuseresearch.com/the-genie-is-out-microsofts-vibevoice-and-the-perils-of-open-source-ai/)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1463263650562314286)** (1 messages): 

> `Luminal KernelBench v3, LLM-driven kernel engineering` 


- **针对 LLM 的内核编译器 Luminal 基准测试**：一位用户询问像 **Luminal KernelBench v3** 这样的内核编译器是否能实现 **LLM-driven SOTA kernel engineering**（LLM 驱动的 SOTA 内核工程），并发布了指向 [Nous Research Forum](https://forum.nousresearch.com/t/can-kernel-compiler-like-luminal-kernelbench-v3-enable-llm-driven-sota-kernel-engineering/310?u=ighoshsubho) 的链接。
- **关于 LLM 内核工程的后续讨论**：目前没有关于 **LLM-driven kernel engineering** 的进一步讨论。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1463264663092330549)** (23 messages🔥): 

> `Manus bugs, Google Project Mariner, Agentic AI vs Manus, Manus performance degradation, Manus billing issues` 


- **专业用户哀叹近期 Manus 的错误和不可靠性**：一位用户报告称使用 **Manus** 构建和训练文本及向量数据库推理模型，但注意到近期性能和稳定性有所下降，**38 个模块中仅有 20 个**正常工作。
   - 该用户表达了对 **CLI access**（命令行访问权限）的渴望，以便修复错误并重新配置系统，甚至愿意为此付费。
- **Project Mariner 颇具吸引力**：一位用户询问是否有人尝试过 **Google 的 Project Mariner**，并表示有兴趣在投入 **$150 月费**之前，先用“闲钱”进行测试。
   - 随后他们提到拥有 **5% 的折扣优惠**。
- **Agentic AI 令人兴奋**：一位用户幽默地表示 **Agentic AI** 让他们感到非常兴奋，认为它是 **Manus** 的潜在竞争对手，特别是 **Gemini 中集成的 agent mode**。
   - 该用户还希望 Agentic AI 能支持移动端。
- **自 Meta 发布新模型以来 Manus 1.6 性能下降**：一位用户报告称，**Manus 1.6 的性能**在最近几周有所下降，可能是自 **Meta** 发布新模型以来出现的。他们指出，尽管摘要正确，但在实现网站开发建议时遇到了困难。
   - 他们提到需要切换到 **Manus 1.6 Max** 才能获得正确的实现。
- **用户报告 Manus 升级的计费问题**：一位用户报告称在升级 **Manus** 时被扣除了 **$42**，但并未收到承诺的 **8000 credits**。
   - 该用户还抱怨支持服务毫无帮助，且邮件支持等待时间过长。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1463463475127914532)** (17 messages🔥): 

> `Anthropic's programming challenge, VLIW architecture, Metal bindings for texture2d, texture2d vs straight buffers` 


- **使用 Tinygrad 应对 Anthropic 的编程挑战**：可以通过将目标问题编写在 Tensor 中，并为他们的玩具级 **VLIW machine** 添加 tinygrad 后端，来解决 [Anthropic 原始的性能测试挑战 (original performance takehome)](https://github.com/anthropics/original_performance_takehome)。
   - 该挑战涉及核心的 **scheduling problems**（调度问题），通过 `PCONTIG=2` 的 bug 修复，可以实现在一个 kernel 中进行调度。使用 `return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))` 可以匹配他们的 `VLEN`，而 `DEVECTORIZE=2` 则将指令保持为向量指令。
- **VLIW 对于 DRAM 而言并非最优**：在处理 **RDNA3 matmul** 的 warp specialized kernel 时，一名成员建议 **VLIW 并不理想用于 DRAM**，并主张采用独立的内核和队列（类似 Tenstorrent 风格）。
   - 由于 VLIW 具有静态调度能力，它更适合 **SRAM/ALU**。
- **Metal 绑定考虑加入 texture2d**：一名成员建议在 Metal 绑定中加入 `texture2d`（修改 `ops_metal.py` + `tinygrad/runtime/graph/metal.py`），利用优化的纹理采样单元，这可能会提升 `conv2d` 等图像密集型操作的 **潜在性能**，尽管这与 `tinygrad` 的平台无关哲学有所冲突。
   - 讨论指出，虽然 image dtype 中的 `Texture` 正在被分离出来，但 `texture2d` 在 Metal 上可以提供类似免费的性能提升，类似于 OpenCL 渲染器（`tinygrad/renderer/cstyle.py`）中使用 `read_imagef` 的方式。
- **Texture2D 速度提升**：`texture2d` 可以缩短 GPU 显存中图像的访问时间和处理速度，理由是其具有 **高读取带宽、针对 2D 局部性优化的缓存，以及优于 ALU 访问的纹理采样单元**。
   - 实证结果显示，使用 `texture2d` 相比直接缓冲区（straight buffers）有 **2%-10% 的速度提升**，且仍有改进空间，不过也有人担心这会开启为 `depth2d` 等其他数据类型添加特殊支持的先例。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1463285654527017063)** (3 messages): 

> `VIZ=1, kernel graphs, scheduler` 


- **VIZ=1 查看图表**：一位成员询问是否有办法让 **VIZ=1** 像查看 uop 图一样查看 kernel 图。
   - 另一位成员回答道：*“可以，点击 schedule（调度）然后选择 'view kernel graph'”*。
- **查看 Kernel 图**：一位用户询问如何使用 **VIZ=1** 可视化 kernel 依赖关系（类似于 uop 图的显示方式），以了解调度器（scheduler）的运行情况。
   - 一位资深用户建议他们在 **VIZ=1** 界面中点击 schedule 并选择 *'view kernel graph'* 来实现此可视化。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1463303237200379986)** (14 messages🔥): 

> `用于学习 Mojo 的 GPU 谜题，Apple Silicon GPU 支持状态，Apple GPU 逆向工程，Mojo 中协程 (Yield) 的状态` 


- ****Puzzles** 为 **GPU** 提供线索**：对学习 Mojo 感兴趣的新手可以根据自己的技能水平使用 [GPU 谜题](https://puzzles.modular.com/)。
   - 还有一个 [论坛](https://forum.modular.com/) 可用于提问。
- ****Mojo** 逆向工程 **Apple** 的 **GPU** 机密**：由于缺乏文档，Modular 正在对 Apple GPU 的大部分内容进行逆向工程，这减慢了进度，但目前一些谜题已经可以运行。
   - 一位成员分享了一个 [GPU 支持矩阵](https://puzzles.modular.com/howto.html#gpu-support-matrix)，但该矩阵可能不是最新的。
- ****NVIDIA** 和 **PyTorch** 给 **Mojo** 带来小麻烦**：唯一无法运行的谜题是 **NVIDIA** 特有的或使用了 **PyTorch** 互操作（interop）的谜题。
   - 由于 PyTorch 处理 Metal 设备的方式与 CUDA / ROCm 截然不同，因此需要扩展接口来适应这种情况。
- ****Yield** 对希望实现协程的人来说尚无结果**：`Yield` 目前不存在，且现有的协程在编译器运行时之外并不真正可用，因为目前还没有真正暴露异步（async）事物供 await 使用。
   - 一位成员非常想在 Mojo 中使用 `yield` 来加速递归算法，但目前需要寻找其他策略。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1463264397211340953)** (5 messages): 

> `错误处理，Python 导入语法，动态 Python 导入` 


- **函数向上抛出错误**：成员们讨论了函数可以被设计为 **raise errors**，从而有效地将错误处理责任传递给更高级别的函数。
   - 一位成员表达了在每个函数中编写 *try/except* 块的乏味，尤其是在处理潜在的导入错误时。
- **针对 Result 类型的 "try Python.import_module"**：一位成员建议未来采用类似 *try Python.import_module('numpy')* 的语法，该语法将返回 **Result** 类型，以简化模块导入期间的错误处理。
   - 这将为每个潜在的导入失败使用 *try/except* 块提供一个更简洁的替代方案。
- **主函数导入作为内联 try/except 的替代方案**：一位成员建议在 **main function** 中导入一次模块并传递 **handle**（句柄），以避免重复的 *try/except* 块。
   - 成员们承认，由于 **动态 Python 导入**，文件可能会在任何给定的导入中丢失，因此必须进行某种形式的错误处理。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1463281239204036692)** (11 messages🔥): 

> `aider 功能，ChatGPT Business 账号设置，OpenCode 对比 Aider，Aider 的未来，Aider 的替代品` 


- **Aider 功能愿望清单**：一位用户询问了 **aider** 中除了 **MCP** 和 **tool calls** 等 Agent 相关功能之外，还有哪些期望的功能。
   - 目前没有收到回应，也没有确定具体的期望功能。
- **通过 Aider 设置 ChatGPT Business 账号？**：一位用户询问他们的 **ChatGPT Business 账号**（缺少 **API key** 但可以访问 Codex LLMs）是否可以与 **aider** 一起使用。
   - 一位成员指向了 [aider 文档](https://aider.chat/docs/llms/other.html) 和 [LiteLLM 文档](https://docs.litellm.ai/docs/providers/chatgpt)，表明可以通过 **LiteLLM** 提供潜在支持。
- **Aider 的终结？**：一位用户担心 **aider** 会被 **OpenCode** 取代。
   - 一位成员回应说，如果 **Paul Gauthier** 已经转向其他项目且不再使用 **Aider**，他将不会再回到这个项目，但另一位用户回应说他们仍在使用 **GPT 5.2** 配合它使用。
- **Aider 是一个停滞的项目吗？**：一位用户推测 **Aider** 是一个停滞的项目，并引用了 **Open Code**、**KiloCode CLI**、**Claude Code** 和 **Gemini CLI** 等替代工具的兴起。
   - 他们还提到了 **Aider-CE 项目**，该项目正在添加 Agent 功能以实现架构现代化。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1463288202411507793)** (10 条消息🔥): 

> `MCP Inspector 在 401 时的重新认证、MCP Multi Server Client 状态保持、MCP client server 排名` 


- **MCP Inspector 在 401 时重新认证失败**：一名成员询问为何 **MCP Inspector** 在连接或工具调用过程中遇到 **401 error** 时不会重新进行身份验证，另一名成员承认存在一个[在重定向过程中持久化 resourceMetadata 的 SDK 问题](https://github.com/modelcontextprotocol/inspector/issues/576#issuecomment-3766294454)，目前正在解决中。
   - 他们提到 **VS Code** 仅在初始连接时使用它，而在之后的 401 错误发生时不再使用，这可能与 SDK 内部机制有关。
- **寻求有状态的 MCP Multi Server Client**：一名成员询问是否有人使用过 **MCP Multi Server Client** 来维护其会话的状态性（statefulness），以及是否遇到了任何问题。
   - 讨论线程中未提供任何解决方案或变通方法，但该话题已被标记为候选主题。
- **为 MCP Clients 排名服务器：这是一个协议问题吗？**：一名成员询问当 Agent 需要管理日历时，**MCP clients** 如何处理服务器推荐，并质疑是使用了自定义算法，还是有关于共享标准的研究。
   - 另一名成员回应称，在开发 [Feature Discovery Protocol](https://discord.com/channels/1199128884040474664/1369487942862504016) 期间曾考虑过“排名（ranking）”，但认为其超出了协议的范畴，建议由生态系统根据每个客户端的情况自行决定。