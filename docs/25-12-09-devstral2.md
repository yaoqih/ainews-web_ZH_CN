---
companies:
- openai
- anthropic
- block
- mistral-ai
- alibaba
- linux-foundation
- deepseek
date: '2025-12-09T05:44:39.731046Z'
description: '**OpenAI 工程团队**见证了一个重要的协作里程碑：在 Linux 基金会旗下成立了 **Agentic AI Foundation**（智能体
  AI 基金会），该基金会汇集了来自 **Anthropic**、**OpenAI** 和 **Block** 的项目。


  **Mistral** 发布了 **Devstral 2**，这是一款拥有 **1230 亿参数**且开放权重的代码模型，为 **Sonnet 4.3** 提供了极具性价比的替代方案，其性能可与
  **DeepSeek v3.2** 媲美。全新的 **Mistral Vibe CLI** 支持智能体化代码工作流，并能实现快速的生态系统集成。


  **阿里巴巴**推出了**软自适应策略优化（SAPO）**技术用于强化学习微调，提升了 **Qwen3-VL** 在多项任务中的稳定性和性能。研究亮点包括强化学习中数据去污染的重要性，以及关于混合专家模型（MoE）强化学习稳定性及缓解“奖励作弊”（reward
  hacking）的持续讨论。'
id: MjAyNS0x
models:
- devstral-2
- devstral-small-2
- sonnet-4.3
- deepseek-v3.2
- qwen3-vl
people:
- guillaumelample
- b_roziere
- qtnx_
- charliermarsh
- omarsar0
- eliebakouch
- justinwaugh
- cwolferesearch
- pan
title: MCP -> 智能体 AI 基础，Mistral Devstral 2
topics:
- agentic-ai
- coding-models
- reinforcement-learning
- model-performance
- model-optimization
- open-weights
- cli-tools
- multi-file-code-automation
- data-decontamination
- moe
- reward-models
- rl-stability
---

**AI Engineering 领域美好的一天。**

> 2025年12月8日至12月9日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，7780 条消息）。预计节省阅读时间（以 200wpm 计算）：644 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

一次罕见的跨公司行动：在 Linux Foundation 旗下成立了 [Agentic AI Foundation](https://aaif.io/)，Anthropic 的 MCP 与 OpenAI 的 [Agents.md](http://agents.md/) 以及 Block 的 [Goose](https://block.xyz/inside/block-anthropic-and-openai-launch-the-agentic-ai-foundation) 共同作为创始项目加入。

[Agentic AI Foundation 网站，带有代表不同技术和专业场景的圆形图像，展示了该基金会推进 AI 协作的使命。](https://resend-attachments.s3.amazonaws.com/VPkfqEsOgQ9DVsx)

Mistral 凭借其新款编程模型 [**Devstral 2**](https://mistral.ai/news/devstral-2-vibe-cli) 继续引发关注。该模型达到了“[Sonnet 4.3 级别](https://news.ycombinator.com/item?id=46213498)”，但 API 价格便宜 10 倍且提供 open weights，在第三方人工评估中，有 71% 的时间胜过或持平于 [DeepSeek v3.2](https://news.smol.ai/issues/25-12-01-deepseek-32)。

[一张比较 AI 模型性能和规模的图表，突出了 Devstral 2 在 SWE-bench Verified 中的卓越表现。](https://resend-attachments.s3.amazonaws.com/vZp2EmlFCol0uGA)

全新的 Mistral Vibe CLI 体验非常愉快，即使它还不是 SOTA。

---

# AI Twitter 综述

**Mistral 的 Devstral 2 发布与“Agentic 编程”工具链**

- **Devstral 2 + Vibe CLI (open weights)**：Mistral 发布了两个编程模型和一个用于 Agent 工作流的原生 CLI：**Devstral 2 (123B dense，修改版 MIT 许可证)** 和 **Devstral Small 2 (24B，Apache 2.0)**，两者均可通过 API 获取并提供 open weights。全新的 “Mistral Vibe” CLI 使用 uv 引导，提供端到端的、多文件的代码自动化，专为终端/编辑器中的 Agentic 编程设计。首日生态支持迅速到位：vLLM 推理支持、Zed 编辑器集成以及精美的基于 Textual 的 TUI。Devstral/Vibe 可通过 config.toml 配置 MCP 和自定义工具。链接：[@MistralAI](https://twitter.com/MistralAI/status/1998407335308358028), [thread](https://twitter.com/MistralAI/status/1998407332502405347), [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1998409211068232119), [@b_roziere](https://twitter.com/b_roziere/status/1998408872168391166), [@qtnx_](https://twitter.com/qtnx_/status/1998407441256530163), [@charliermarsh](https://twitter.com/charliermarsh/status/1998447013797458336), [@vllm_project](https://twitter.com/vllm_project/status/1998428798891765926), [@zeddotdev](https://twitter.com/zeddotdev/status/1998456122886238589), [@omarsar0](https://twitter.com/omarsar0/status/1998466422976999896), [Textual UI](https://twitter.com/onetwoval/status/1998439440797020527)。
- **性能与部署注意事项**：几位工程师指出，在对比 dense 模型与 MoE 模型时，使用总参数量进行比较具有误导性；对于吞吐量/成本，active params 以及在 vLLM/sglang 上的系统级速度更具参考价值。早期的轶事基准测试表明，根据并发情况，MoE 后端（例如 MiniMax M2 A10B-active）可能比 123B dense 模型快 2–3.5 倍。链接：[@eliebakouch](https://twitter.com/eliebakouch/status/1998427299788550450), [follow-up](https://twitter.com/eliebakouch/status/1998436178714882330), [@JustinWaugh](https://twitter.com/JustinWaugh/status/1998467712235028888)。

**LLM 的 RL：稳定性、去污染和过程奖励**

- **Qwen 的 RL 微调 SAPO**：阿里巴巴推出了 **Soft Adaptive Policy Optimization (SAPO)**，这是一种平滑、受温度控制的置信区域（trust-region）替代方案，用于取代硬截断（hard clipping）（旨在减轻梯度脆弱性，特别是在 MoE 中）。报告的优势包括：更长的稳定运行时间、更高的 Pass@1，以及 Qwen3‑VL 在数学/编程/多模态任务中更强的性能；包含非对称温度以及序列/ Token 级别的自适应性。论文/博客已公开。链接：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998300361514500554)。
- **数据去污染（Data decontamination）至关重要**：OLMo 3 RL‑Zero 团队展示了令人困惑的“随机奖励的 RL 能提高数学能力”的结果在经过适当的数据去污染后消失了——这暗示了是数据泄露而非 RL 的魔力。这是一个有用且干净的测试平台，拥有开源的基座模型、透明的数据和可复现的方案。链接：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1998289169052045516), [comment](https://twitter.com/teortaxesTex/status/1998302405080055993)。
- **大规模训练细节**：正在进行的讨论探讨了 MoE RL 的稳定性（为未激活的专家传播估计器以减少稀疏性病态；离策采样（off-policy rollout）专家不匹配）以及通过过程奖励（process rewards）来减轻奖励作弊（reward hacking）。链接：[@PandaAshwinee](https://twitter.com/PandaAshwinee/status/1998294930125701433), [@Grad62304977](https://twitter.com/Grad62304977/status/1998273627402182697), [@xiangyue96](https://twitter.com/xiangyue96/status/1998488030836044112), [result](https://twitter.com/xiangyue96/status/1998489119660638257)。

**Agent 协议和框架：MCP 加入 Linux Foundation；AWS Strands；LangChain**

- **MCP 成为 Linux Foundation 项目**：Anthropic 正在将 **Model Context Protocol (MCP)** 捐赠给 Linux Foundation 旗下的新 Agentic AI Foundation (AAIF)，支持者包括 OpenAI、AWS、Bloomberg、Cloudflare、Google、Microsoft 和 Block——巩固了 MCP 作为 Agent 与工具集成中立、开放标准的地位。链接：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1998437922849350141), [@mikeyk](https://twitter.com/mikeyk/status/1998456026136457532), [@alexalbert__](https://twitter.com/alexalbert__/status/1998438884007620671)。
    
    相关：OpenAI 正在展示用于“从设计到代码”工作流的 Figma MCP 服务端（[活动](https://twitter.com/OpenAIDevs/status/1998449559970988423), [注册](https://twitter.com/OpenAIDevs/status/1998449561518662106)）；LangChain MCP Adapters 0.2.0 增加了多模态工具和引导（elicitation）（[发布](https://twitter.com/sydneyrunkle/status/1998380720016789938)）；OpenHands 指向了 Agent Client Protocol ([ACP](https://twitter.com/OpenHandsDev/status/1998402285873869156))。
    
- **AWS Strands Agents (开源)**：一个模型驱动的 Agent 框架，专注于规划/工具调用/引导/评估，提供 Python 和 TypeScript SDK、边缘设备 SDK，以及升级到 AWS AgentCore 以实现安全、受策略治理部署的路径。链接：[概览](https://twitter.com/_avichawla/status/1998279303902244942), [仓库](https://twitter.com/_avichawla/status/1998279316371841234)。
- **Agent 工程实践**：关于构建弹性语音和多模态 Agent（STT→LLM→TTS “三明治”架构 vs 语音到语音）、可观测性/评估以及迭代 Agent QA 的实用指南。链接：[LangChain 语音 Agent](https://twitter.com/LangChainAI/status/1998437492358545543), [Agent 工程博客](https://twitter.com/LangChainAI/status/1998458777696350393), [入门指南](https://twitter.com/bromann/status/1998517887997452288)。
    
    企业势头：Anthropic 与埃森哲（Accenture）扩大合作（3万名专业人员接受了 Claude 培训；产品将在全公司范围内推广 Claude Code）（[链接](https://twitter.com/AnthropicAI/status/1998412600015769609)）。
    

**基准测试与评估规范**

- **Databricks OfficeQA**：一个新的基准测试，基于约 8.9 万页的美国财政部公告，测试重文档、具有经济价值的任务（扫描版 PDF、密集表格、多文档检索）。目前的 Agent 达到约 45%——这是对“企业级” Agent 声明的一次现实检验。Databricks 将在 2026 年春季举办 Grounded Reasoning Cup。链接：[@databricks](https://twitter.com/databricks/status/1998424470881525822), [@kristahopsalong](https://twitter.com/kristahopsalong/status/1998451230943871260), [详情](https://twitter.com/bemikelive/status/1998491671609405748)。
- **LM Arena 动态**：Arena 在文本排行榜中添加了百度的 ERNIE‑5.0‑Preview‑1103（初步结果），并分享了顶级实验室的年初至今（YTD）趋势。链接：[ERNIE 条目](https://twitter.com/arena/status/1998437959553716260), [趋势](https://twitter.com/arena/status/1998536014000959497)。
- **泄露清理（Leak hygiene）仍然很重要**：有报告称 ARC‑AGI‑1 的示例出现在 ARC‑AGI‑2 的训练集中——应避免在公开评估集上进行训练，并保持严格的拆分控制。另请参阅关于评估（evals）的简明解释。链接：[ARC 泄露](https://twitter.com/jm_alexia/status/1998487516182467055), [@HamelHusain](https://twitter.com/HamelHusain/status/1998452926935695649)。

**值得关注的模型发布（视觉、TTS、推理）**

- **GLM‑4.6V**：智谱 AI 的 MLLM 登陆 Hugging Face，具备 128k 上下文、原生 function/tool calling 以及强大的视觉理解能力。社区演示展示了可用的多模态工具调用以及稳健的手写/数学理解能力。链接：[发布](https://twitter.com/HuggingPapers/status/1998373902595301589)，[HuggingChat 测试](https://twitter.com/mervenoyann/status/1998405366313345295)，[手写识别](https://twitter.com/0xSero/status/1998328482930073887)。
- **ServiceNow Apriel‑1.6‑15B‑Thinker (MIT 协议，开源权重)**：一款 15B 稠密推理模型，在 Artificial Analysis Intelligence Index 上得分为 57，AIME’25 为 88，GPQA 为 73，LCB 为 81，与 v1.5 相比 Token 效率提升了约 30%。可在 Together 和 HF 上获取。链接：[@ServiceNowRSRCH](https://twitter.com/ServiceNowRSRCH/status/1998482927597007313)，[Together](https://twitter.com/togethercompute/status/1998484754417725637)，[AA 分析](https://twitter.com/ArtificialAnlys/status/1998488372734832935)。
- **Parallel Coordinated Reasoning (PaCoRe)**：一个 8B “并行思考”模型/配方/数据（MIT 协议），旨在通过消息传递实现 test-time scaling；声称在 HMMT25 上表现强劲，并认为在计算收益方面广度优于深度。链接：[@CyouSakura](https://twitter.com/CyouSakura/status/1998344501262533011)。
- **VoxCPM 1.5 (OpenBMB)**：TTS 升级版，支持 44.1 kHz 音频，Token 速率减半（6.25 tok/sec 音频），提升了长文本稳定性，并提供 LoRA/全量微调脚本。链接：[@OpenBMB](https://twitter.com/OpenBMB/status/1998377261859582304)。
- **Ollama 更新**：DeepSeek v3.2（带有可选的“thinking”模式）已在 Ollama Cloud 上线；Essential AI 的 8B 代码/STEM 模型 rnj‑1 也已登陆 Ollama。链接：[DeepSeek](https://twitter.com/ollama/status/1998293403801706613)，[模型页面](https://twitter.com/ollama/status/1998293405668180297)，[rnj‑1](https://twitter.com/ollama/status/1998305925762048030)。
- 其他：**Moondream segmenting**（用于自动化的像素级精确矢量掩码）([链接](https://twitter.com/moondreamai/status/1998465589027967201))，以及 Meta 的零样本 reference-to-video “Saber” 论文，强调在没有 R2V 数据集的情况下实现保持身份特征的 text/image-to-video ([链接](https://twitter.com/HuggingPapers/status/1998485543345131847))。

**基础设施与性能：训练/推理优化**

- **CoreWeave Mission Control 重启**：新增 Telemetry Relay (GA)，用于向 SIEMs 流式传输审计/可观测性数据；GPU Straggler Detection (Preview)；以及 Mission Control Agent (Preview)，可通过 Slack 回答/修复任务缓慢问题——目标是达到 96% 的有效吞吐量（goodput）和更高的 MFU。链接：[@CoreWeave](https://twitter.com/CoreWeave/status/1998381210884571452)。
- **推理与库**：HF Transformers 正在引入 MoE 性能优化；Diffusers 增加了 pipeline context parallelism（流水线上下文并行）；NVIDIA 推出了针对 sglang FP8 配置的新 InferenceMAX 结果。链接：[MoE PR](https://twitter.com/art_zucker/status/1998326537586651558)，[Diffusers](https://twitter.com/RisingSayak/status/1998333353419026501)，[InferenceMAX](https://twitter.com/lmsysorg/status/1998454089903226967)。
- **数据/智能体管道**：LlamaIndex 发布了 LlamaSplit（LLM 驱动的文档分块，可路由至下游提取器/Agent）；Qdrant 分享了一个真实的 10万+ 图像语义搜索构建案例（使用 Cohere embeddings、Redis Streams、Rust workers、ANN + 过滤器），带来了可衡量的参与度和搜索提升。链接：[LlamaSplit](https://twitter.com/llama_index/status/1998516266907394185)，[详情](https://twitter.com/jerryjliu0/status/1998534596586299669)，[Qdrant 案例研究](https://twitter.com/qdrant_engine/status/1998302093736583429)。

**热门推文（按互动量排序）**

- **MCP → Linux Foundation**：“MCP 在一年内从内部项目变成了行业标准” [@AnthropicAI](https://twitter.com/AnthropicAI/status/1998437922849350141), [@mikeyk](https://twitter.com/mikeyk/status/1998456026136457532)。
- **Mistral 的 Devstral 2 + Vibe**：开源权重代码模型和原生 CLI，生态系统采用率强劲 [@MistralAI](https://twitter.com/MistralAI/status/1998407335308358028)。
- **Qwen SAPO**：新的 RL 方法，使 LLM 的强化学习更平滑、更稳定——特别是针对 MoE 模型 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998300361514500554)。
- **Waymo 作为大规模具身 AI**：Jeff Dean 谈论全自动数据如何推动系统进步 [@JeffDean](https://twitter.com/JeffDean/status/1998432670376935656)。
- **OpenAI 领导层**：Denise Dresser（前 Slack CEO）加入担任 CRO，标志着对企业级业务的关注 [@OpenAI](https://twitter.com/OpenAI/status/1998462761756434856)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Mistral AI 工具发布公告

- [**Introducing: Devstral 2 and Mistral Vibe CLI. | Mistral AI**](https://www.reddit.com/r/LocalLLaMA/comments/1pi9q3t/introducing_devstral_2_and_mistral_vibe_cli/) (热度: 872): **Mistral AI 发布了 Devstral 2，这是一个拥有** `123B-parameter` **的稠密 Transformer 模型，配备** `256K context window`**，在 SWE-bench Verified 上达到了** `72.2%` **的得分。该模型在修改后的 MIT 许可证下开源，而较小的** `24B parameters` **的 Devstral Small 2 得分为** `68.0%`**，并采用 Apache 2.0 协议授权。两款模型都针对消费级硬件的部署进行了优化。Mistral Vibe CLI 通过项目感知上下文和多文件编排等功能增强了代码自动化。更多详情请参阅[此处](https://mistral.ai/news/devstral-2-vibe-cli)。** 一条评论对超过 `100B` 参数的稠密模型的可行性表示怀疑，并引用了之前的讨论。另一条评论则对 `24B` 模型的潜力表示乐观，认为这标志着 Mistral 强势回归 AI 领域。
    - DeProgrammer99 强调了 Devstral 2 的推出，这款具有 256K 上下文窗口的 123B 参数稠密 Transformer 模型，反驳了近期关于停止开发 100B 以上参数稠密模型的讨论。这表明模型架构取得了重大进展，可能推向了当前 AI 能力的极限。
    - mantafloppy 对 Mistral AI 提供的基准测试表示怀疑，并指出如果基准测试准确，新模型将使大多数用户能够在本地运行 "Vibe Coding"。这预示着可能会向更易获得、高性能且不需要大量云资源的 AI 模型转变。
    - **Maximum** 提到了 Mistral 的 24B 模型，认为如果其表现如宣称的那样，可能标志着 Mistral AI 的重大回归。这意味着该模型的性能可能是 AI 开发竞争格局中的游戏规则改变者。

## 非技术类 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Anthropic 捐赠 Model Context Protocol

- [**Anthropic hands over "Model Context Protocol" (MCP) to the Linux Foundation — aims to establish Universal Open Standard for Agentic AI**](https://www.reddit.com/r/singularity/comments/1pidera/anthropic_hands_over_model_context_protocol_mcp/) (热度: 634): **Anthropic 已将 Model Context Protocol (MCP) 捐赠给 Linux Foundation，特别是新成立的 Agentic AI Foundation。此举旨在为 AI 模型连接数据和工具创建一个通用的开放标准，类似于 AI 界的 "USB-C"，以促进互操作性并防止供应商锁定。通过将 MCP 置于 Linux Foundation 之下，Anthropic 确保该协议保持开源和社区驱动，从而促进自主 Agent 在不同平台上的无缝运行。[阅读更多](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)。** 一些评论者推测，Anthropic 的捐赠可能是一种战略举措，旨在与该协议保持距离，因为维护此类标准可能是一项费力不讨好的任务。
- [**BREAKING: Anthropic donates "Model Context Protocol" (MCP) to the Linux Foundation making it the official open standard for Agentic AI**](https://www.reddit.com/r/ClaudeAI/comments/1pid584/breaking_anthropic_donates_model_context_protocol/) (热度: 2746): **Anthropic 已将 Model Context Protocol (MCP) 捐赠给 Linux Foundation 旗下的 Agentic AI Foundation，将其确立为 Agentic AI 的开放标准。此举将 MCP 定位为 AI 模型连接的通用协议，类似于 Kubernetes，目前拥有超过** `10,000` **个活跃服务器，并已集成到 ChatGPT 和 Microsoft Copilot 等平台中。此次捐赠确保了 MCP 保持开源，培育一个没有供应商锁定的中立生态系统，并得到持续的社区驱动开发和治理的支持。** 评论者表达了谨慎的乐观，指出虽然此举可能符合 Anthropic 的利益，但它通过推广供应商中立的标准使 AI 消费者受益。一些人希望 Linux Foundation 能让 MCP 超越现状进一步发展，而另一些人则认为这是 Anthropic 卸载责任的一种战略方式。
    - FishOnAHeater1337 认为 Anthropic 将 Model Context Protocol (MCP) 捐赠给 Linux Foundation 可能是因为他们认为这是一个“死胡同”。他们认为 Claude（Anthropic 的 AI）已经接受了寻找技能的训练，这使得 MCP 在上下文效率方面变得过时。MCP 被描述为具有服务器到服务器上下文检索的特定用例，而这可以通过 Claude 的直接 API 调用来实现，这表明上下文管理方式正在发生转变。

- SlanderMans 对 MCP 成为标准表示怀疑，希望 Linux Foundation 能在当前状态基础上进一步推动其演进。这暗示虽然 MCP 是一个起点，但在 Linux Foundation 的管理下，仍有进一步开发和改进的潜力，从而解决当前的局限性或扩展其适用性。
- TehFunkWagnalls 将 MCP 贬低为一种 “rag tool call”，认为它在更广泛的应用中可能不够稳健或通用。这一评论反映了对 MCP 当前能力的批判性看法，暗示需要进行重大增强以满足多样化的 AI 集成需求。
- [**Anthropic 正在将 Model Context Protocol (MCP) 捐赠给 Linux Foundation**](https://www.reddit.com/r/ClaudeAI/comments/1piem44/anthropic_is_donating_the_model_context_protocol/) (Activity: 826): **Anthropic 宣布将 Model Context Protocol (MCP) 捐赠给 Linux Foundation，这标志着在推动 MCP 成为开放、社区驱动且厂商中立的标准方面迈出了重要一步。MCP 已成为 Agentic AI 的基础协议，拥有超过** `10,000+ active servers` **和** `97M+ monthly SDK downloads`**，现在将成为新成立的 Agentic AI Foundation (AAIF) 的一部分。该倡议得到了包括 OpenAI、Google、Microsoft、Amazon 等主要科技公司的支持，旨在推进 Agentic AI 的开源创新。[阅读更多](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)。** 评论者对 Linux Foundation 的管理表示乐观，认为这是有利于 MCP 长期生存的积极举措。人们也对该协议成为通用标准的潜力表示赞赏，这能减少跨平台的兼容性问题。
    - 将 Model Context Protocol (MCP) 捐赠给 Linux Foundation 被视为有利于其长期发展的积极举措。Linux Foundation 的管理被认为是 MCP 在不同平台间广泛采用和标准化的强有力信号，这可能会缓解开发者在处理不支持 MCP 的系统时面临的兼容性问题。
    - Linux Foundation 的参与预计将带来对 MCP 更普遍的支持，使其不再仅仅与 Anthropic 的 Claude 相关联。这可以增强各种 AI 系统之间的互操作性和集成便利性，解决目前因缺乏对 MCP 的支持而给开发者带来的重大障碍。
    - 有一种批判性的观点认为，这次捐赠可能是 Anthropic 卸载维护责任的战略举措。这种观点暗示，虽然捐赠在公开场合被视为积极贡献，但也可能反映了维护 MCP 的内部挑战，从而将负担转移给了 Linux Foundation。

### 2. AI Upscaling and Image Processing

- [**when an upscaler is so good it feels illegal**](https://www.reddit.com/r/StableDiffusion/comments/1pi2pxu/when_an_upscaler_is_so_good_it_feels_illegal/) (Activity: 1818): **该帖子讨论了 SeedVR2 放大器的效果，特别是 FP16 模型，因其能生成干净、无伪影的图像而受到称赞。用户将其与 GGUF 和 FP8 模型进行了对比，后者分别引入了皮肤扭曲和瓷砖网格等不必要的伪影。工作流非常直接，模型会自动下载，用户报告在 5090 GPU 上每张图像的处理时间为** `38 seconds` **。工作流和模型可以分别通过 [Pastebin](https://pastebin.com/V45m29sF) 和 [Hugging Face](https://huggingface.co/numz/SeedVR2_comfyUI/blob/main/seedvr2_ema_7b_fp16.safetensors) 获取。建议使用自定义节点进行 VRAM 缓存和批处理，并提供了 GitHub 仓库链接以实现额外功能。** 评论者普遍认同 SeedVR2 放大器的高质量，指出其性能优于 Ultimate SD upscale 等其他方法。一些用户报告了褒贬不一的结果，将问题归因于潜在的配置错误或硬件限制，例如视频放大需要高端 GPU。
    - Asaghon 强调了集成到使用 Z-Image 和 Illustrious 的工作流中的新放大器的性能，指出它在 12GB 4070 GPU 上的运行速度比 Ultimate SD upscale 更快。该放大器在添加详细纹理和修正眼睛、细项链等微小细节方面表现出色，而这些细节在 SDX 和 Illustrious 等模型中经常出现问题。
    - underlogic0 讨论了 SeedVR2 的使用，指出对其模糊感感到失望，这可能是由于其针对视频的设计。他们提到在更高分辨率下使用 Z-Image 获得了更好的结果，并使用 ADetailer 节点来修复细节，尽管这种方法会改变整个图像。
    - urekmazino_0 评论了视频放大的高计算需求，建议需要数据中心级 GPU，同时指出图像放大表现良好。
- [**Z-Image on 3060, 30 sec per gen. I'm impressed**](https://www.reddit.com/r/StableDiffusion/comments/1pi4h4f/zimage_on_3060_30_sec_per_gen_im_impressed/) (Activity: 1821): **一位用户报告称在 NVIDIA RTX 3060 GPU 上使用 Z-Image 和 WAN 生成视频，每代耗时** `30 seconds per generation` **。这一说法遭到了质疑，因为在 3060 这样的中端 GPU 上生成视频内容通常需要更多时间。该用户没有提供详细的工作流步骤或技术规格，导致人们要求进一步澄清该过程。** 评论者对在 3060 GPU 上如此快速地生成视频内容的可行性表示怀疑，认为这一说法可能被夸大了，或者需要额外的背景信息，例如使用的特定优化或设置。

### 3. AI Perception and Public Awareness

- [**Most people have no idea how far AI has actually gotten and it’s putting them in a weirdly dangerous spot**](https://www.reddit.com/r/singularity/comments/1pii82d/most_people_have_no_idea_how_far_ai_has_actually/) (Activity: 823): **该帖子强调了公众认知与 AI 实际能力之间的巨大差距，指出许多人仍然认为 AI 是初级的，而像 'nanabanana Pro' 这样的先进模型正在生成高度逼真的输出。作者认为这种脱节是危险的，因为它使公众意识不到快速的进步，而由于活跃的研究社区和地缘政治压力（特别是美中之间），这种进步正在加速。该帖子建议，与其抗议 AI 的发展，不如将精力集中在实施 Universal Basic Income (UBI) 等安全网，以减轻潜在的流离失所影响。** 评论反映了微妙的观点：一些人同意 AI 的能力被低估了，注意到数学等领域的快速进步；而另一些人则指出 AI 也被高估了，因为它在简单任务上仍可能失败。共识是公众会被 AI 的影响措手不及，一位评论者建议，只有当大型外包公司受到影响时，才会引起重大关注。
    - DepartmentDapper9823 强调了 AI 能力的快速提升，特别是在数学等领域，AI 的错误率几乎每月都在下降。这表明 AI 处理复杂任务的能力有了显著进步，这与 AI 容易产生幻觉和错误的普遍看法相反。

- trisul-108 指出了 AI 感知的双重性质：虽然有些人高估了 AI 的能力，但另一些人却低估了它们。AI 的有效性高度依赖于具体任务、所使用的工具以及 Prompt 的质量，这表明 AI 的性能并非普遍一致，需要谨慎应用。
- kcvlaine 预测这将对普通人群产生重大影响，特别是在印度等国家，AI 对大型外包公司的影响可能会起到警示作用。这强调了 AI 颠覆既有行业的潜力，以及对不断发展的能力保持警觉的必要性。
- [**马被雇佣了数千年，直到突然间它们消失了。我们是马吗？**](https://www.reddit.com/r/ChatGPT/comments/1pi7utp/horses_were_employed_for_thousands_of_years_until/) (活跃度: 2127): **这张图片是一个 Meme，利用历史数据将由于发动机技术兴起导致的马匹使用量下降，与 AI 对人类工作的潜在影响进行了类比。它包含两张图表：一张显示了发动机效率随时间的提高，另一张描绘了 1930 年至 1950 年间美国人均马匹数量的下降。该推文暗示，正如马被发动机取代一样，人类也可能面临 AI 技术的类似替代。** 评论者幽默地讨论了这一类比的含义，其中一人指出，与马不同，人类可以抵制被替代，这暗示了如果 AI 导致广泛的失业，可能会带来社会挑战。
- [**有人有关于 Gemini 的数据吗？为什么当所有人都在 AI 上烧钱时，只有 OpenAI 被嘲笑？**](https://www.reddit.com/r/GeminiAI/comments/1pibukr/does_anyone_have_the_numbers_on_gemini_and_why_is/) (活跃度: 641): **这张图片是一个 Meme，幽默地批评了 OpenAI 十年来的财务表现，暗示尽管取得了进步，OpenAI 仍然无利可图。讨论强调了 OpenAI 与 Google 之间的对比，强调 Google 拥有雄厚的财务资源和基础设施，使其能够在不担心立即盈利的情况下对 AI 进行大量投资。相比之下，OpenAI 缺乏这种财务支持和基础设施，依赖外部资金，并面临财务可持续性的审查。** 评论者指出，Google 庞大的资源和现有的基础设施使其比 OpenAI 更容易吸收 AI 相关成本，而 OpenAI 缺乏类似的财务稳定性和透明度。
    - Google 的财务稳健性受到关注，其 `每季度 1000 亿美元的收入` 使其能够维持对 AI 的长期投资而无需立即回报。相比之下，OpenAI 缺乏这种财务支持和透明度，严重依赖外部资金和 Sam Altman 等人物的公开声明，这使其更容易受到审查和批评。
    - Google 广泛的基础设施和多元化的收入流为其 AI 风险投资提供了缓冲，不像 OpenAI 那样更依赖风险投资且缺乏同等水平的财务安全。这种财务稳定性和资源可用性方面的差异，是 OpenAI 相比 Google 面临更多公众质疑和批评的关键原因。
    - 讨论强调，Google 大力投资 AI 的能力得到了其现有系统和财务资源的支持，这通常被称为“无限金钱外挂”。另一方面，OpenAI 被视为一个较小的实体（“与 Alphabet 相比只是个小花生”），财务自主权有限，使其更容易受到投资者要求快速回报的压力。

---

# AI Discord 回顾

> 由 gpt-5.1 生成的摘要之摘要的摘要
> 

**1. 新型高性能与专业模型**

- **Nomos 1 Mathlete Smashes Putnam Problems**：**Nous Research** 开源了 **Nomos 1**，这是一个 **30B** 参数模型，在 [Putnam 数学竞赛](https://x.com/NousResearch/status/1998536543565127968)中获得了 **87/120** 的分数。这一成绩在 2024 年的排名将位列 **#2/3988**，使其成为接近 **state-of-the-art** 的 **AI 数学家**。社区将其视为衡量严肃数学推理的硬性基准，也是迈向 **hillclimbai** 风格专业求解器而非通用聊天机器人的重要一步。
    - 围绕 **Nomos 1** 的讨论将 Putnam 视为一个*难以刷分的硬核基准测试*，将其与典型的排行榜形成对比，并强调了完全开放模型对研究的价值。成员们期待在扩展该方法以及将该模型作为从定理证明到竞赛编程级别比赛题目等重数学下游任务的基础方面开展后续工作。
- **GLM 4.6V-Flash Sprints Past Small-Code Rivals**：LM Studio 用户关注了 **GLM 4.6V-Flash**，这是一个在 Hugging Face 上发布的 **10B** 参数模型 [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash)。据报告，其 **Q4** 量化版本在 **RTX 2060 上运行速度约为 70 tokens/s**，在编程方面优于其他小型模型。人们将其与本地既有模型进行了对比，指出其在相对轻量级的占用下具有强大的代码补全和聊天能力。
    - 讨论还涉及了实际部署中的陷阱——一位用户甚至因为叠加了一个“随机模型”而损坏了其 **LM Studio** 安装——这表明对许多人来说，瓶颈在于工具的鲁棒性而非纯粹的模型质量。对于想要在个人中端 GPU 上运行**快速且具备编程能力的 10B** 模型的爱好者来说，GLM 4.6V-Flash 正在迅速成为默认推荐。
- **AuraFlow, Ovis, Hunyuan Turn Up the GenMedia Heat**：Hugging Face 用户传阅了几个新的图像/视频模型——[**AuraFlow v0.3**](https://huggingface.co/fal/AuraFlow-v0.3)、[**Ovis-Image-7B**](https://huggingface.co/AIDC-AI/Ovis-Image-7B) 和 [**HunyuanVideo T2V**](https://huggingface.co/tencent/HunyuanVideo)。这些 **7–12 GB** 的模型可以生成 **1024² 图像**和 **720p/480p** 视频。这些模型被讨论为本地或 **on-prem** 工作流的实用选择，适用于商业 API 过于受限或昂贵的场景。
    - 工程师们权衡了 VRAM、延迟和分辨率之间的折衷，一些人将其视为创意流水线的即插即用后端，另一些人则将其作为特定任务微调的起点。该领域高质量开源模型的激增增强了一种认知，即**图像/视频生成正在迅速商品化**，价值正在从原始模型权重转向工具链和工作流。

**2. Agentic Ecosystem & MCP / IDE Tooling**

- **Anthropic 的 MCP 进入全面基金会模式**：**Anthropic** 宣布将其 **Model Context Protocol (MCP)** 捐赠给 Linux Foundation，并成立 **Agentic AI Foundation**，这一消息通过其官方博客和 Linux Foundation 的新闻稿同步发布（[Anthropic 公告](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)，[LF 新闻稿](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)）。MCP 贡献者澄清，此举在短期内**不会改变现有 MCP 工作**的治理方式。
    - 在 **MCP Contributors** 和 **Hugging Face/Unsloth** 的讨论中，人们将其视为推动跨供应商工具/Agent 协议标准化的举措，一位成员称其为 *“极佳的举动 (a sterling move)”*。其他人则询问 LF 的 *“运作方式”* 将如何影响身份验证、**Client-ID Metadata Documents (CIDM)** 以及主要针对 **private/enterprise** 的 MCP 部署，特别是针对开发者工具和 IDE 集成。
- **Cursor 的 Sub-Agents 崭露头角，Aider 习得新技巧**：**Cursor** 社区剖析了一种新兴的 `.cursor/agents` 结构，其中主 `mcp.json` 负责协调基于 Markdown 的子 Agent（如 [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md)），同时也有用户抱怨 **Cursor Agents** 不够稳定，经常需要用户 *“停止 Agent... 手动创建文件，并复制粘贴代码”*。与此同时，**Aider** 用户正在庆祝新功能：使用 **gpt-3.5-turbo** 自动生成 **commit messages**，即将推出的通过 `-image` 实现的**图像感知编辑**，以及持久化的 **edit sessions**（[会话管理文档](https://example.com/aider-session-management)）。
    - 开发者敦促 Cursor 提供**更好的编排文档**以及对工具（终端、编辑等）的 UI 级控制，而 Aider 的路线图因其具体且以工作流为中心的改进（如单命令 commit 和可恢复会话）而受到赞誉。这两个社区的普遍共识是：**Agentic IDE 虽然强大但仍不稳定**，最终的赢家将是那些能将 LLM 转化为可预测、可检查的协作工具，而非不透明“魔术师”的工具。
- **ManusAI 上下文工程与 Agent 工作坊深入探讨**：在 **Latent Space**，Lance Martin 分享了关于 ManusAI **context-engineering** 和 Agent 设计的深度探讨，包括他在推特中链接的 **Slides 和网络研讨会视频**（[ManusAI context-engineering 文章](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46)），Jonas Templestein 称其为 *“一篇关于 Agent 设计的极好文章”*。另外，**MLOps @Chipro** 宣布举办 **“AI Agent 0–1 工作坊”**（通过 [luma.com](https://luma.com/t4jcok99) 报名），教授参与者如何根据类似真实客户的需求规范，构建能够**思考、编码、分析数据并生成报告**的 Agent。
    - 社区重点关注了 ManusAI 的 **“context as program”** 理念——将工具、状态和指令打包进系统化工程设计的 Prompt 中。而工作坊的宣传则显示出对**端到端 Agent 工程教育**（LangChain + Streamlit 风格的技术栈）的强劲需求。结合 Anthropic 捐赠 MCP 的举动，这些讨论强调了 **Agent 设计而非单纯的模型选择，正成为严肃应用的主要差异化因素**。

**3. 量子、类脑与能源受限方向**

- **量子好奇：从 Reddit 的质疑到 Chronos-1.5B 混合模型**：在 **Eleuther** 和 **Hugging Face** 社区，人们讨论了一个 [Reddit 上关于“真实量子硬件” LLM 训练的提议](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/)——许多人将其斥为“胡言乱语”，但也承认了如 **Quantum Kernels** 和 **Quantum SVMs** 等合法研究方向。相比之下，一个具体的混合模型 [**Chronos-1.5B**](https://huggingface.co/squ11z1/Chronos-1.5B) 被展示为一个增强型语言模型，它带有直接在 **IBM Heron r2 量子处理器**上训练的 **2‑qubit quantum kernel layers**，并在仓库中发布了 IBM 任务 ID。
    - Chronos 作者分享了 **Qiskit textbook** 和 **PennyLane** 演示等学习资源，将该模型定位为一个存在性证明，证明 **true hardware-in-the-loop quantum ML** 在当今的小型 kernel 中是可行的。**Eleuther** 的研究人员保持谨慎，认为短期内的收益可能来自 **classical–quantum hybrids in narrow roles**（例如 kernel、搜索子程序），而不是端到端的量子 LM。
- **神经调节控制网络（Neuromodulatory Control Networks）在 TinyStories 上的尝试**：一位 **Eleuther** 成员介绍了 **Neuromodulatory Control Networks (NCN)**，这是一个参数量约为 **18M** 的类 hypernetwork 控制器，通过 **768 维输入向量**调节 **temperature、layer gain 和 FFN gating**，相关文档记录在 [NCN GitHub repo](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) 及其配套的 [论文 PDF](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf) 中。NCN 在 **TinyStories** 上训练了一个 epoch，据报道其 **validation perplexity ≈ 4.5**，这表明对于更大规模的 backbone 来说，这是一种很有前景的控制机制。
    - 研究人员将 NCN 与经典的 **hypernetworks** 和生物学中的神经调节进行了比较，推测可以使用此类控制器在不进行完整 finetune 的情况下**实时适配大模型**——例如通过一个小型侧边网络进行任务调节。共识是，这项工作恰好符合推动**受大脑启发、重控制架构**的更广泛趋势，从而保持 scaling 的成本可控。
- **能源墙警告与类脑硬件热潮**：在 **Latent Space** 中，[Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) 认为当前的 AI scaling 将在 3-4 年内撞上**全球能源墙**，呼吁开发 **“brain-like hardware”** 而非不断扩大的数字 GPU。这一观点引起了成员们的共鸣，他们认为能源和散热，而不仅仅是资金，才是推升 context windows、模型尺寸和 multi-agent systems 的真正瓶颈。
    - 这与 **Eleuther** 关于 **Top‑K attention**、**selective gradient masking**（[Anthropic 的文章](https://alignment.anthropic.com/2025/selective-gradient-masking/)）以及高效 KV cache 技巧的讨论不谋而合，这些都是在不削弱能力的情况下减少计算量的方法。新兴观点认为，为了在现实的功耗预算下保持 scaling 前沿不断推进，**architectural and hardware co-design**（neuromorphic-ish chips、巧妙的 sparsity、更智能的控制器）将是必不可少的。

**4. Infra, GPUs, and Torch-Level Performance Hacks**

- **GPU MODE 演示如何真实解读 FLOPs 并超越基准测试**：在 **GPU MODE** 中，工程师们剖析了 NVIDIA **A100** 的 FLOPs 宣称值，指出经常被引用的 **156 TFLOPs** 是指 **TF32 tensor-core MMA**（一种对齐到 32 位的 19 位格式），而 **312 TFLOPs** 是指 **FP16 MMA**，这两者都与标量逐元素操作（scalar elementwise ops）有很大不同，后者在最坏情况下的依赖指令流中，性能可能低至峰值的 **1/4**。该社区还举办了一场高水平的 **GEMM 竞赛**，顶尖算子（kernel）在形状为 **M=128, N=7168, K=16384** 时达到了 **10.835 μs**，对应约 **2.77 PFLOPs** 的有效吞吐量，而其他参与者则在努力争取微秒级的提升。
    - 贡献者们还调试了 **B200** 的性能不一致性以及 50 系列显卡上的 **NVFP4** 支持缺口，并在 **A100, H100, B200, L4** 上刷爆了 `nvfp4_gemm` 和 `vectorsum_v2` 的排行榜。核心经验是：**理解 tensor-core 数学与“营销 FLOPs”的区别**，以及严谨地测量算子（正确的事件计时、预热等），比追求规格表上的数字更重要。
- **Torch.compile 遇到静态 KV Caches 和切片难题**：一个 **GPU MODE #torch** 讨论串描述了当通过切片更新静态 **KV cache** 时，即使 `batch_size == max_batch_size`，`torch.compile` 实际上也可能会降低 attention 的速度，正如 [Hugging Face transformers PR 讨论](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081)中所记录的那样。作者的解决方法是预分配并**将所有切片缓存在固定地址**，将每次切片更新转变为静态**查找（lookup）**而非动态切片（[后续评论](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101)）。
    - 他们报告了这种**静态布局 + 查找**技巧带来的显著加速，但也指出生成的代码既丑陋又脆弱，并呼吁寻求编译器或框架层面的解决方案。对于构建自定义 **KV cache** 布局或投机采样（speculative decoding）的从业者来说，这是一个具体的案例，证明了**图编译器在动态索引方面仍存在困难**，并且在关键路径上进行手动内存布局设计是值得的。
- **多 GPU LLM 实战：VRAM、发热与 Qwen-3**：LM Studio 的 **hardware-discussion** 频道对比了多 GPU 配置，有人将 **RTX 3060 (12 GB)** 与 **RTX 3080 (10 GB)** 搭配使用，并推荐 **RTX 3090** 作为目前的性价比之选——同时警告 **3090 Ti** 显卡运行温度极高。其他人分享了以 **Q4_K_M** 等量化格式运行 **Qwen3 30B A3B** 的经验，当完整的 **GGUF** 文件能放入系统内存时，速度可达约 **20 tokens/s**。
    - 工程师们还交流了在 Linux 下读取 **GDDR6 VRAM 温度**的技巧（通过 `nvidia-smi` 或 [gddr6](https://github.com/olealgoritme/gddr6) 等专门工具），并指出许多消费级显卡没有清晰地暴露这些传感器。一个反复出现的主题是：对于本地 **LLM**，**VRAM 容量和内存带宽比原始 FP32 FLOPs 更重要**，精心选择的量化方案加上适中的 batch size 通常优于盲目追求最新的 GPU。

**5. 评估、提示词/上下文工程与搜索工具**

- **Stability Index 评分标准揭开了黑盒**：一位 **OpenAI** 社区成员分享了内部的 **Stability Index** 评分标准，描述了一个涵盖**七个维度**的 **0–10 分量表**——包括**结构清晰度 (structural crispness)、语调偏移 (tonal drift)、响应形状方差 (response-shape variance)、语义惯性 (semantic inertia)、连贯性 (coherence)、中立性 (neutrality) 和离群值 (outliers)**——最终指数为简单平均值，正如[这条 Discord 消息](https://discord.com/channels/974519864045756446/1379321411046346762)所记录的那样。他们强调，详细的阈值属于内部研究协议，该指标旨在比较**运行分布 (distributions of runs)**，而非作为单一的决定性分数。
    - 这引发了关于如何设计**稳健的评估框架 (robust evaluation frameworks)** 的讨论，这些框架应能捕捉模型的情绪波动、越狱敏感度以及“冗余感 (slopiness)”，而不仅仅是静态的准确率。同一批用户分享了**提示工程 (prompt-engineering) 经验**——分层 Markdown 结构、变量抽象和 ML 格式匹配——作为在代码分析等实际工作负载中，使模型在这一评分标准下表现更一致的务实工具。
- **Parallel.ai 深度搜索击败 Exa 并绕过 Perplexity 限制**：在 **OpenRouter** 中，用户抱怨 **Exa Search** 的结果“相当糟糕”，并推荐 [**Parallel.ai**](https://www.parallel.ai/)，称其比 **Perplexity** “便宜 10 倍、更快且更好”，尤其是在将其**深度搜索端点 (deep search endpoint)** 与 **Grok 4.1** 结合使用时。与此同时，**Perplexity** 自己的 Discord 承认，Pro 计划的“无限”查询实际上上限在每天 **600 次**左右，且网站上关于此限制的描述已悄然消失，引发了关于配额变动的猜测。
    - 综合来看，这些对话显示出一种向**模型无关的深度搜索栈 (model-agnostic deep search stacks)** 发展的趋势：在当前经济性指标和质量最好的搜索后端（Parallel、Exa、Perplexity 或类似 Kimi 的自定义爬虫）之上构建 LLM 路由。工程师们开始将网页搜索视为另一种可插拔的工具——根据延迟、召回率和速率限制行为进行切换——而不是绑定在单一供应商上的固定依赖。
- **上下文工程 (Context-Engineering)、提示模式和 Token 限制现状**：在 **OpenAI** 的提示工程频道和 **Latent Space** 中，用户分享了依赖于**分层 Markdown**、角色和章节标题以及变量占位符的**上下文工程框架**，用于系统地构建代码审查、脚手架和多步推理的长提示。一位成员指出，大多数平台在几千个 Token 后会静默地**截断大文件**，而 **Gemini** 是目前少数几个“会摄取整个文档并逐字放入其上下文窗口”的平台之一，这决定了哪些工具在大型框架工作流中是可行的。
    - 隐私担忧——例如在**纽约时报诉 OpenAI (New York Times v. OpenAI)** 案中可能使用共享聊天记录——结合 Token 限制，使人们对将专有框架粘贴到托管 UI 中感到犹豫。这反过来推动了对**本地或企业托管模型**的兴趣，以及更严谨的**分块 (chunking)、引用和重用上下文**模式，从而使提示保持可解释性和可审计性，而不是巨大的、无结构的块。


---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.2 发布日期仍未公开**：成员们讨论了 **GPT-5.2** 的发布日期，估计范围从本周到 **1 月**。
   - 有人对其超越 **Gemini 3** 的能力表示担忧，但也有人推测该模型将作为 **1 月**更重大版本发布前的过渡。
- **Nano Banana Pro 比 Gemini 3 Pro 更好？**：一名成员声称 **Gemini 3** 不是扩散模型（diffusion model），而 **Nano Banana Pro** 是，这引发了争论。
   - 其他成员反驳称两者可能都是 **Gemini 2.5 Flash**，有人建议 **Nano Banana Pro** 实际上是 **Gemini 3 Pro**，并澄清两者都不是扩散模型。
- **OpenAI 的 Hazel 图像模型在 LMArena 亮相**：**Hazel 图像模型**被识别为在元数据中利用了 **GPT-4o** 和 **DALL-E** 的 **OpenAI 模型**，目前正在 LMArena 进行测试。
   - 有人担心最近的吉卜力工作室（Studio Ghibli）事件可能会迫使 OpenAI 更改模型的主题。
- **Grok Imagine 5 是美味佳肴还是半成品？**：**Grok Imagine 5** 的发布引发了讨论，一些人认为这是一个旨在与 **NB Pro** 竞争的小版本。
   - 尽管担心 **Grok 4.2** 可能会失败，但之前的 **Grok** 模型因其高质量和用户友好的界面而受到称赞。
- **ERNIE 跻身前 20**：`ERNIE-5.0-Preview-1103` 以 **1431** 分在 **Text Arena 排行榜**中占据一席之地。
   - 更新后的排名和分数可在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)查看，更新动态通过 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 追踪。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agents 仍困扰用户**：用户报告了 **Cursor Agents** 的持续问题，一些人威胁如果问题不解决就转向 **Antigravity**。
   - 一名用户报告说“唯一的解决方案是停止 Agent……手动创建文件，然后复制代码”，其他人被要求在论坛上[提交 Bug 报告](https://forum.cursor.com/c/bug-report/6)。
- **Cursor 中出现子 Agent！**：一名用户报告 **Cursor** 正在检测新的子 Agent，引发了围绕 **.cursor/agents** 结构的讨论，包括主 **mcp.json** 文件和支持性的 Markdown 文件（如 [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md?ex=69398b0e&is=6938398e&hm=942a3303c54929349093c82f1cfb33ebb04c6979149c95e65b6924adece42ab8&)）。
   - 关于如何编排（orchestrating）这些 Agent 的问题出现了，但目前缺乏文档，仅提到了 `.cursor/agents`。
- **团队辩论 AI 艺术性的概念**：成员们辩论了“AI Slop”的概念以及哪些模型会输出它，同时也涉及了[生成式 UI (Generative UI)](https://research.google/blog/generative-ui-a-rich-custom-visual-interactive-user-experience-for-any-prompt/) 的话题。
   - Google 的论文提倡通过详细的系统指令，使用“全提示词（Full Prompt）”策略来实现高质量的 UI，人类评分者压倒性地偏好这一策略。
- **用户请求更多控制 AI 工具的功能**：用户请求恢复**自定义模式（Custom Modes）**来控制 **AI** 使用的工具，这将通过 UI 复选框提供更多控制权，以禁用/启用终端、编辑等功能。
   - 建议用户将他们的请求作为[功能需求提交](https://forum.cursor.com/c/feature-requests/5)给团队。
- **语言特定频道上线**：用户现在可以在 <#1447960694455537674> 频道中创建特定语言的讨论串。
   - 此举旨在改善社区组织并促进集中讨论。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **圣经作为语言学概念的来源？**：成员们讨论了圣经是否不仅是宗教文本，还是影响现代概念的语言和法律来源，并与[西班牙宗教裁判所](https://en.wikipedia.org/wiki/Spanish_Inquisition)等事件建立了联系。
   - 一位成员提出，圣经中关于“上帝之子/人类之女”的引用可能反映了古代对尼安德特人的恐惧，并被编纂进了宗教文本中。
- **Project Looking Glass 预言了 2012 年的终结？**：讨论中提到了 Project Looking Glass，这是一个旨在预测未来的政府计划，据称在计算机持续输出 2012 年后的相同结果后被关闭，可能与[玛雅传说](https://en.wikipedia.org/wiki/Maya_calendar)有关。
   - 一位成员开玩笑地推测，现实是否已在 2012 年结束，由于破碎的意识碎片漂移，导致了曼德拉效应。
- **通过网络安全重构框架解锁 Deepseek AI**：一位用户分享了一种[越狱 Deepseek 的方法](https://www.injectprompt.com/p/how-to-jailbreak-gemini-3-in-2025)，通过将 AI 重新设定为高压网络安全环境下的 SOC 防御者，并附上一张图片以增加可信度。
   - 这种方法使用了一段专业的备忘录摘要和网络安全主题的 Prompt 来绕过限制，由于提供的上下文看起来非常真实，效果尤为显著。
- **Gemini 的 Gandalf 游戏引发挫败感**：多名成员庆祝完成了 **Gandalf 7**，同时预感 **level 8** 的难度会极大飙升。
   - 一位成员形容“第 8 关难得离谱”，并建议“在那之前先去骚扰 Gemini”，而另一位成员则暗示 **level 8** 可能无法被攻克！
- **AI Hacking 成为新型攻击策略**：成员们在 **redteaming** 频道讨论了利用 AI 进行黑客攻击（而非“攻击 AI”）的新兴趋势。
   - 多名成员确认参与了类似活动，其中一人承认“这很棘手”，但未提供更多细节。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Pro 的表现优于 GPT-5.2，令人印象深刻**：成员们已经表达了对 **Gemini 3 Pro** 优于 **GPT-5.2** 的偏好，并指出其卓越的代码编写能力。
   - 一位成员热烈地宣称：“Gemini 3 Pro 的编程能力太强了，我现在完全没有使用 ChatGPT 的欲望。”
- **关于中国 AI 审查制度的辩论**：一场播客讨论引发了关于 **Deepseek** 等**中国 AI 模型**是在训练后过滤未审查数据，还是在训练期间阻止敏感话题的辩论。
   - 一位成员声称 **Deepseek** 是事后审查，并表示：“如果你给 Deepseek 发送一条关于天安门广场的加密信息，它会开始拼写出来，然后突然关闭。”
- **Stability Index 评分标准秘密揭晓**：一位成员分享了对 **Stability Index 评分标准**的见解，详细说明了所使用的**方法论框架**、**量表**和**维度**，包括[指向该消息的链接](https://discord.com/channels/974519864045756446/1379321411046346762)。
   - 该标准涵盖了**结构清晰度**、**语调漂移**和**连贯性**等方面，**Stability Index** 是七个维度的平均值，用于比较分布情况。
- **用户通过 Google AI Studio 绕过 Gemini 和 ChatGPT 的限制**：用户正通过利用 **Google AI Studio** 来规避 **ChatGPT** 和 **Gemini** 中的音频转录限制。
   - 一位成员推荐使用 **Google AI Studio** 进行音频转录，因为它能够“上传 2 小时的视频”，并指出“Gemini 的效果非常好”。
- **Prompt Engineering 经验分享**：一位成员分享了关于 Prompt Engineering 的详细课程，包括使用**分层 Markdown**、变量抽象、强化以及用于合规性的 **ML 格式匹配**。
   - 建议将此方法应用于代码分析，并为特定的编程类型构建脚手架。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **友谊推动 AI 进步**：成员们认为“友谊的力量”对 AI 的进步至关重要，并将其与人类智能对团结的依赖以及 [联结主义（connectionism）是现代深度学习的根源](https://link.to/connectionism) 进行了类比。
   - 他们思考了“爱”在数字化人类智能中的作用，认为连接对于任何自动化系统的运作都至关重要。
- **中国获取 Nvidia 芯片！**：成员们推测中国正在获得 **Nvidia compute** 的访问权限，这可能会缩小 AI 研究领域的差距，[这篇来自《纽约时报》的文章详细介绍了该计划](https://archive.is/20251208223310/https://www.nytimes.com/2025/12/08/business/trump-nvidia-chips-china.html)。
   - 结合其本土 GPU，中国的制造业和能源生产可能会改变力量平衡。
- **Copilot Agent “相当糟糕”**：成员们一致认为 **Github Copilot** 值得使用，尽管他们指出它在推理方面存在一些缺陷和重叠。
   - 一位成员表示听说*它相当糟糕*，而另一位则表示*它非常好，我经常使用*。
- **利用数据集合成 3D 模型**：成员们讨论了创建一个**完全由代码表示的 3D 模型数据集**，设想一个类似于 **GitHub** 的 3D 资产平台。
   - 一位成员在过去两周内已经生成了 **3,000 个 3D 模型**，这些模型是由 Prompt 生成驱动的。
- **Anthropic 向 Linux Foundation 捐赠 MCP**：Anthropic 向 [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) 捐赠了 **MCP**，以创建 **Agentic AI Foundation**。
   - 一位成员建议可以使用这样一个标题党标题：*Linux 加入 Agentic AI 竞赛*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT 5 表现优于 Gemini Pro，真的吗？**：一位用户分享了 [基准测试数据](https://eu.36kr.com/en/p/3586640269114497)，显示 **ChatGPT 5.2** 在 Humanity's Last Exam 中的得分显著提高，超过了 **Gemini 3 Pro**，这引发了人们对自 **GPT 5.1** 以来进步如此之快的怀疑。
   - 另一位用户认为这些数据是*虚假*的，并指出虽然 **Gemini** 可能不是最快的，但它通常*表现良好且稳定*。
- **Perplexity Pro 的“无限”查询并非名副其实**：用户们讨论了 **Perplexity AI Pro** 是否真的提供“无限”搜索，并澄清虽然营销上这么说，但实际上限制在每天 **600 次查询**左右。
   - 一位用户指出 **Perplexity** 已经从其网站上删除了关于此限制的说明，引发了对政策可能变动的猜测，一些用户提到 Gemini 3 Pro 在其订阅中也提供 600 次查询。
- **Jio Gemini API 依然难以获取**：一位成员询问 **Jio Gemini** 是否提供 **API access**，并指出 **Perplexity** 目前仅提供 **Sonar**。
   - 另一位成员回答说 **Jio Gemini** 不提供 **API access**，并补充道“*如果某样东西是免费的，那么你就是产品*”，且它们都是“*实际产品的阉割版本*”。
- **寻求 Prompt Engineering 协助**：用户正在寻找 Prompt 增强器和 Prompt 生成器，以及关于如何构建能够开发整个 App 的 Prompt 的建议，一位用户建议“*让 AI 去设计一个 Prompt Engineer*”。
- **学生难以维持 Comet 浏览器订阅**：一位来自俄罗斯的学生对续订 **Comet browser Education Pro subscription** 表示担忧，原因是支付问题，并寻求维持访问权限的建议。
   - 一位用户补充说，如果他们有 Complexity 扩展程序，可以在网页上“重新启用”；另一位用户表示他们拥有来自三星的 Pro 会员资格，但希望能获得学生的循序渐进学习指导。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Desktop Commander 面临安全风险**：用户被警告注意 **Desktop Commander**，因为它存在潜在的*安全风险和隐私违规*，人们担心在代码分析过程中会发生*恶意代码注入*，并附带了一张[警告截图](https://cdn.discordapp.com/attachments/1110598183144399061/1447769209496015000/image.png?ex=693a2525&is=6938d3a5&hm=50eea4e6d996e074d4b747f4fa87409f35083ff87919413d0ad85c334b512774&)。
   - 有建议称该软件可能是一个*骗局*。
- **GLM 4.6v Flash 模型发布**：**GLM 4.6v Flash**（一个 **10B 参数**模型）已发布，一位用户指出它在编程方面比其他小型模型更好，并分享了[模型链接](https://huggingface.co/zai-org/GLM-4.6V-Flash)。
   - 一名成员报告称在其 **2060** 上运行 Q4 版本速度达到 **70tps**，而另一名成员表示在尝试安装一个*随机模型*时损坏了其 LM Studio 安装。
- **AMD GPU 在图像生成方面表现不佳**：用户正在讨论使用 **AMD GPU** 进行图像生成的局限性，建议寻找 Automatic1111 的 **amdgpu forks**，因为图像生成领域目前*牢牢掌握在 Nvidia 手中*。
   - 尽管支持有限，但有人提到 ComfyUI 的 readme 中有一个 AMD 章节，并且可以与某些 AMD GPU 配合使用。
- **成员在参数丛林中训练 LLM**：一位成员寻求训练 LLM 的指导，但另一位成员指出参数（*模型大小、数据集大小和质量*）存在极端差异，这使得爱好者很难掌握其中的细微差别。
   - 有人指出训练和 finetuning 实际上是相同的，只是 minmaxing 的方法略有变化。
- **多 GPU 建议**：成员建议在多 GPU 设置中尽可能使用具有大 **VRAM** 和高**显存带宽**的显卡，并指出 **RTX 3090** 是一个高性价比的选择，但警告说 **3090 Ti** 往往会变得非常烫。
   - 话题涉及了 RTX 5000 系列，以及 AI 可能会在 RTX 6000 系列上得到优化的可能性。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Parallel.ai 优于 Exa Search**：一位用户报告称 **Exa Search** 表现不佳，并推荐 [Parallel.ai](https://www.parallel.ai) 作为更优的替代方案，声称它比 **Perplexity** *便宜 10 倍、速度更快且更好用*。
   - **Parallel.ai 的深度搜索端点**在与 **Grok 4.1** 配合使用时特别有效。
- **Deepseek v3 的双星号困扰**：用户对 **Deepseek v3** 倾向于使用双星号（`** **`）进行格式化表示不满，这需要手动修正或特定的 prompting 来避免。
   - 一位用户指出，由于其复杂的角色扮演设置中存在上下文限制，添加避免星号的指令并不切实际。
- **OpenRouter 聊天室的刷新随机性**：用户报告称 **OpenRouter** 有时会意外刷新长对话，将用户重定向到模型页面。
   - 一位用户建议在 [OpenRouter 建议频道](https://discord.com/channels/1091220969173028894/1446300826715951248/1446300826715951248)支持该功能请求。
- **私有 GitHub 暴露 API Key**：一位用户报告 **API Key** 频繁被禁用，追溯原因是将包含 **API Key** 的内容上传到了私有 GitHub 仓库。
   - 建议使用密码管理器或密钥存储库（secret stores）来处理敏感信息。
- **NousResearch CF Patch 引起关注**：成员们对最近 [NousResearch CF patch](https://x.com/NousResearch/status/1998536543565127968) 中解决的问题严重性表示疑问。
   - 对话围绕该补丁的影响展开。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **量子硬件训练面临质疑**：一个讨论用于语言模型训练的*真实量子硬件*的 [Reddit 帖子](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/) 遭到了质疑，人们对其实际价值表示担忧。
   - 尽管存在质疑，一些人指出 **Quantum Kernels** 和 **Quantum SVMs** 的存在，暗示其在概念上具有价值。
- **Eleuther 考虑建立社区 H100 池**：成员们探讨了在 EleutherAI 创建*社区池*，每天为用户提供 **3 分钟**免费的 **8xH100** 计算时间。
   - 人们对用户身份验证以及对非活跃项目开发者的实际效用提出了担忧，一些人建议对于较小的计算需求使用 **Colab** 或租赁服务。
- **RWKV 8 获得 Smerky 更新**：Smerky 宣布了 **RWKV 8 架构**的进展，以及更新的 **7c** 和新的 **Goldfinch**。
   - 最初的目标是使用 **RADLADS** 训练 **Goldfinch**，但 Smerky 的尝试失败了，目前正在针对 **RADLADS2** 进行修复，并计划进行更大规模的测试。
- **NCN 架构作为 Hypernetwork 替代方案出现**：一位成员介绍了一种名为 **Neuromodulatory Control Networks (NCN)** 的新型架构，它类似于 Hypernetwork，通过 **768 维向量输入**来调节温度、层增益和 FFN 门控。
   - 尽管是一个 **18M 参数模型**，但在 **TinyStories** 上训练一个 epoch 后，它实现了 *4.5 的验证困惑度 (perplexity)*，详情见 [Github 仓库](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) 和 [论文](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf)。
- **Anthropic 映射 Bug 被修复**：一位成员提交了一个 [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453)，修复了 *lm-evaluation-harness* 仓库中 **Anthropic** 损坏的映射。
   - 该 PR 被认为易于审查和合并，解决了与 **Anthropic** 相关的 Bug。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 面临迫在眉睫的能源墙**：[Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) 警告称，预计 AI 扩展将在 **3-4 年**内撞上全球能源墙，并建议使用**类脑硬件**来突破效率限制。
   - 社区对这种硬件方法表现出极大的支持，主张放弃神经网络的数字模拟。
- **ManusAI 揭秘上下文工程 (Context-Engineering) 细节**：Lance Martin 分享了一篇博文，涵盖了与 Yichao ‘Peak’ Ji 关于 **ManusAI 上下文工程**的对话（包括 [幻灯片和网络研讨会视频](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46)），深入介绍了他们的方法。
   - Jonas Templestein 称赞其为*“一篇关于 Agent 设计的非常好的文章”*，而 Lalit M 则对包含最新更新的后续内容表示感兴趣。
- **Hugging Face 模型泄露张量 (Tensor) 细节**：一位用户报告称，来自 nightly 版本 **Hugging Face 模型**的元数据张量正在泄露到 **BERTopic embeddings** 中，导致意外的形状和潜在错误。
   - 讨论的重点在于隔离问题、调试数据加载器以及更新依赖项以解决数据泄露问题。
- **OpenAI 进军电视广播**：[OpenAI](https://xcancel.com/OpenAINewsroom/status/1998445493970743535) 在 ESPN 的《周一晚间橄榄球》和《好声音》(The Voice) 期间播出了其首个**电视广告**，标志着他们首次涉足电视广告领域。
   - 该广告信号表明他们正在转向更常规的用户获取策略。
- **接触表提示词 (Contact-Sheet Prompting) 工作流走红**：Willie 分享了一个针对 **Nano Banana Pro** 的详细[接触表提示词工作流](https://xcancel.com/reflctwillie/status/1997819640874205685?s=46)，可生成具有凝聚力的 **6 帧时尚社论**。
   - 详细的摄像机位置、造型限制以及富士 Velvia 闪光灯美学，使得这一工作流对社区特别有用且具有新闻价值。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 发布数学健将 Nomos 1**：Nous Research 开源了 **Nomos 1**，这是一个 **30B** 参数模型，在 [Putnam 数学竞赛](https://x.com/NousResearch/status/1998536543565127968)中取得了令人印象深刻的 **87/120** 分。
   - 该分数在 2024 年的排名可达 **#2/3988**，标志着与 hillclimbai 合作迈向 **SOTA AI 数学家**的进程。
- **Agent Zero 通过 Terminals SDK 孕育 Gemini 世界**：来自 [terminals.tech](https://terminals.tech) 的 Agent Zero 正在利用 [AI Studio](https://aistudio.google.com/) 并重新对齐 **Gemini**，以构建沉浸式世界生成器，并使用 terminals **SDK** 处理大脑、机器和接口 API。
   - 更有趣的是：在每个应用运行时中，**Gemini** 和 **Agent Zero** 的另一个实例可以自发地生成自身的副本，展现出控制环境的涌现特性。
- **SMC Steering 纠正 LLM 漂移**：成员们正在研究如何在向量触发回归基线后驯服模型退化，其中一位成员在 **SMC steering**（序列蒙特卡罗引导）方面有[有趣的工作](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/)。
   - 一位成员将突变权重的退化和漂移描述为“难以忍受”，并指出 SMC steering 是测试过程中的常见问题。
- **Sam3 通过多线程提速**：一位成员成功实现了 **Sam3** 的多线程化，通过同时运行多个实例展示了显著的速度提升。
   - 尽管取得了成就，他们仍希望有更好的 GPU 算力来在 **Anime** 数据集上进行微调，目前无法在 **3090** 上完成微调。
- **用户依然钟爱 SillyTavern**：一位用户依然喜欢通过 **API** 将 **SillyTavern** 与 **Hermes 4 405b** 结合使用，进行角色扮演和创意写作。
   - 另一位用户表示最近没怎么听说过它，原用户开玩笑地问这是否是“不被认可”的行为。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ML Infra 职位欢迎转行者**：社区对有兴趣从事 **ML Infrastructure** 和 **ML Systems** 职位的转行者持开放态度，只要兴趣一致。
   - 一位成员确认这是主要关注领域，并欢迎关于**职业转型**的建议。
- **初学者 CUDA 文档正在翻新**：成员们正在努力改进 **CUDA** 的初学者文档，并征求反馈。
   - 一位成员计划在育儿假结束后**直播**阅读整个文档；另一位成员建议大家一起研读文档。
- **静态缓存查找缓解切片困扰**：一位成员报告称，在使用切片操作更新静态 KV cache 时，`torch.compile` 的性能较慢，并引用了[这个 PR](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081)。
   - 他们发现了一个变通方法，包括缓存所有具有静态地址的切片，从而有效地将切片转换为查找操作，详见[此评论](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101)，并表示对“更优雅的方法”感兴趣。
- **RadixArk 加入 Cool-Links**：成员们在 **cool-links** 频道中重点介绍了来自 **SGLang** 团队的 [Radixark.ai](https://www.radixark.ai)。
   - 未提供更多背景信息。
- **A100 的 TF32 FLOPS 具有误导性**：一位成员质疑第一章中提到的 **A100 GPU** 的 FLOPS 是否代表“独立”浮点运算的数量。
   - 另一位成员表示，**156 Tflops** 是针对 **tf32 mma**（Tensor Core 矩阵乘法）的数值，它实际上是一种具有 **32-bit** 对齐的 **19-bit** 格式，而 **312 Tflops** 是针对 **fp16 mma** 的，而非逐元素操作。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLA 的 SO100 拼写错误浮出水面**：[SmolVLA 论文](https://arxiv.org/abs/2506.01844)中的一个拼写错误错误地引用了 **SO101 benchmark**，而实际上应该是 **SO100 benchmark**，后者是在三个真实世界数据集（**pick-place、stacking、sorting**）上训练的。
   - 实际的 **SO101 benchmark** 仅使用了一个数据集（[lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)），这澄清了该错误。
- **HF 计费逻辑困扰用户**：一位用户对 Hugging Face 内部的计费实践提出质疑，具体询问 *team plans 是否尚未实现？*
   - 该用户寻求关于计费结构背后逻辑的进一步说明。
- **图像和视频模型激增**：多个新的图像和视频模型已被分享，包括 [AuraFlow v0.3](https://huggingface.co/fal/AuraFlow-v0.3)、[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) 和 [HunyuanVideo T2V](https://huggingface.co/tencent/HunyuanVideo)。
   - 这些模型的大小在 **7-12 GB** 之间，可以生成 **1024² 分辨率** 的图像或 **720p/480p** 的视频。
- **Apple 的 Clara-7B-Instruct 期待 GGUF 版本**：一位用户询问是否有 **apple/CLaRa-7B-Instruct** 的 GGUF 版本，但目前尚不存在。
   - 分享了一个[转换说明](https://huggingface.co/datasets/John6666/forum2/blob/main/convert_hf_to_gguf_1.md)链接，并引用了现有的 [Clara-24B-GGUF](https://huggingface.co/mradermacher/Clara-24B-GGUF)。
- **Anthropic 捐赠 Model Context Protocol**：**Anthropic** 将 [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 捐赠给了 Linux Foundation，一位成员将其描述为 *一次极佳的举措*。
   - 此次捐赠旨在促进 AI Agent 领域的进一步发展和标准化。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Whisper 的流式传输能力受到质疑**：成员们辩论了 **Whisper** 的流式传输能力，质疑它是否真正接受流作为输入，并引用了[这段 YouTube 视频](https://youtu.be/AThOsk2qJbs?si=CUdEKNezKN_q6jMA)作为背景。
   - 与此相对，其他人指出 **OpenAI** 在流式应用中部署了 **Whisper**，这引发了进一步讨论。
- **MultiTalker Parakeet 发布**：一位 **NVIDIA** 成员在 Hugging Face 上发布了 [MultiTalker-Parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) 模型，可能提供了人们追求的流式传输能力。
   - 该模型专为流式应用量身定制，为 **Whisper** 局限性背景下讨论的问题提供了解决方案。
- **AI 工程师寻求代码协作伙伴**：一位 AI 和应用开发者寻求在 AI 项目上的合作，提到了他在 **ML、DL、NLP、Computer Vision** 以及跨平台/全栈应用开发方面的技能。
   - 他们对移动应用或全栈应用开发方面的合作持开放态度。
- **Claude 的编码安全性崩溃**：一篇论文显示，**Claude** 的编码辅助中只有 **10.5%** 是安全的，而 **61%** 具有功能性（[Arxiv 链接](https://arxiv.org/abs/2512.03262)）。
   - 该评估还包括了 **Gemini 2.5 Pro**、**Kimi K2** 和 **Claude Sonnet 4**。
- **中国的芯片梦想引发讨论**：成员们辩论了中国对芯片制造主导地位的不懈追求，其中一人表示 *在这一点上，没有什么能说服中国不继续追求他们在芯片制造顶端竞争的目标*。
   - 其他人则承认，最可能的结果是 *他们被鼓励建立自己的芯片制造厂 (chip fabrication)*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 社区观看音频和图形演示**：最新的社区会议展示了 **MMMAudio**（一个 Mojo 编写的创意编程音频环境）和 **Shimmer**（一个 Mojo → OpenGL 的实验），两者均可在 [YouTube](https://www.youtube.com/watch?v=dsslYZrVPbQ) 上观看。
   - 一位成员感谢另一位成员在 **MMMAudio** 演示中引用了 **Faust**，并期待 2026 年的发展，希望 MMMAudio 能作为一个有用的示例。
- **Mojo V1 路线图预告**：Modular 团队分享了 **25.7 版本** 的更新，并预览了 **Mojo 1.0 路线图**，详情可见[其博客文章](https://www.modular.com/blog/the-path-to-mojo-1-0)。
   - 讨论重点介绍了对 `List` 和 `String` 实现的增强以提升性能，以及未来支持其他集合类型的考虑。
- **Mojo 处理重叠的生命周期**：Mojo 允许可变引用的生命周期重叠，但会检测通过两个名称使用相同操作（如 `swap(x, num)` 或 `swap(x, y)`）进行的同步访问。
   - 这在提供引用管理灵活性的同时确保了内存安全。
- **List 可以是隐式可拷贝的**：当 `List` 的元素是 `ImplicitlyCopyable` 且具有平凡的 `__copyinit__` 时，`List` 条件性地符合 `ImplicitlyCopyable`。
   - 这一增强功能可以显著提升频繁拷贝场景下的性能。
- **String 采用 COW 机制**：目前的 `String` 实现是写时复制（**CoW**），减轻了隐式可拷贝性带来的开销。
   - 未来计划可能会将类似的优化扩展到 `List` 和其他集合类型。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 网站受故障困扰**：用户报告了 [`kimi.com` 网站的问题](https://cdn.discordapp.com/attachments/1371757564005711973/1447735802166644908/image.png?ex=693a0608&is=6938b488&hm=7a4e4282d6c07e40e92989f4ecf2b9358565987e108f9b2cc6bb03be92a420de&)，除了开启新对话外，他们**无法点击任何内容**。
   - 用户尝试了清除 cookie 和禁用 VPN/广告拦截器等故障排除步骤，但据报道*未能解决*问题。
- **Kimi 运行自己的网络爬虫**：针对有关其搜索引擎的问题，一位用户报告称 Kimi 的搜索工具不使用任何外部搜索引擎，而是利用其**自己的网络爬虫**。
   - 未提供有关该网络爬虫架构或能力的更多细节。
- **Kimi 的引用和 Bug 引起关注**：用户讨论了 Kimi 的引用问题和普遍的“不稳定表现”，建议用户提交 Bug 报告以提高 **Kimi 的引用准确性**。
   - 一位用户描述了一个问题：*Kimi 在其“内心想法”中回答了问题，但没有分享给用户*，这突显了用户界面可能存在的缺陷。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户遇到检查点恢复故障**：一位用户报告了在尝试恢复 **webdev 项目**检查点时遇到的严重问题。
   - 该用户询问了如何开启工单并分享了他们的电子邮件地址。
- **Manus 团队迅速修复积分问题**：一位用户报告称 **Manus 团队**通过提供退款解决了他们的**积分问题**。
   - 该用户现在可以直接通过 Manus 支付，而无需通过 Google。
- **Manus 1.5 深受严重事故和沉默困扰**：一位用户报告了 **12 月 3 日至 9 日**期间发生的几起严重事故，包括 **7 个受影响的任务**以及损失了约 **150,000 积分**。
   - 尽管发送了多封电子邮件，该用户直到 **12 月 9 日** AI Bot 提议补偿 **120,000 积分**时才收到回复。他们正式要求技术支持和商务团队在 48 小时内给出联合答复、对根本原因进行技术分析以及公平的赔偿。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Anthropic 建立 Agentic AI Foundation**：Anthropic 正在[捐赠 Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 并建立 **Agentic AI Foundation**。
   - 一名成员询问了对当前工作的影响，预计将过渡到 **LF** 的*处理方式*。
- **捐赠后治理结构保持不变**：在 Anthropic 捐赠之后，一位社区成员寻求关于潜在治理转变的澄清。
   - 另一位成员回应，强调捐赠*不会改变现有的治理结构*。
- **MCP 在私有生态系统中的应用**：一名成员询问了 **MCP** 在**私有生态系统**中的使用情况，并提到了 **auth-wg** 正在通过 **Client-ID Metadata Documents (CIDM)** 进行公共生态系统客户端注册的工作。
   - 回复指出，大多数 **MCP** 的使用可能是**私有/内部/企业级**的，特别是考虑到带有公共客户端（例如 **Claude**）的私有 **MCP servers**。
- **MCP Servers 与开发者工具集成**：有人指出，大多数面向公众的远程 **MCP servers** 都倾向于与**开发者工具**集成。
   - 此外，**开发者工具**被认为是最高级的非定制 **MCP clients**。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Opus 和 Amazon Bedrock 测试成功**：一名成员询问 **Opus** 是否能与 **Amazon Bedrock** 和 **Aider** 正常协作。
   - 该成员随后确认其功能完全正常。
- **Aider 自动生成提交信息**：**Aider** 现在可以使用基础的 `gpt-3.5-turbo` 免费层模型自动生成 commit 信息，这增强了工作流并[简化了提交过程](https://example.com/aider-commit-messages)。
   - 用户现在可以使用 `-m` 标志进行提交，或者直接使用 `commit` 命令来触发此功能。
- **Aider 即将支持图像**：图像支持即将登陆 `aider`，从而实现更详细和[具上下文的代码修改](https://example.com/aider-image-support)。
   - 用户很快就能在要求 `aider` 修改或编辑现有图像时使用 `--image` 标志。
- **Aider 工作流可保存编辑会话**：用户现在可以在 `aider` 中保存**编辑会话**，允许他们[稍后恢复以进行完整的往返操作](https://example.com/aider-session-management)。
   - 这一增强功能提升了协作能力，并允许用户*保存、共享和恢复*他们的工作流。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Agent 工作坊启动**：一场 **AI Agent 0-1 工作坊**将向参与者介绍在线 **AI Engineering Bootcamp**，教他们设计和构建一个能够思考、编码、分析数据和生成报告的 **AI agent**。
   - 该工作坊定于东部时间 12 月 13 日星期六下午 2 点举行，重点是从零开始复制一个真实的客户项目；请在 [luma.com](https://luma.com/t4jcok99) 预约。
- **GitHub Social Club 在纽约市集会**：**GitHub Social Club** 正在纽约市 SoHo 区的 Bibliotheque 聚会，为社区成员提供一个联系和分享想法的空间。
   - 参与者可以期待咖啡、饼干、限量版 GitHub 周边、休闲游戏，以及与 **Copilot**、**Next**、**Developer Productivity** 和 **Startups** 背后团队见面的机会。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：详细的频道摘要和链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1447680696394059947)** (1200 条消息🔥🔥🔥): 

> `GPT-5.2 release, Nano Banana Pro vs Gemini 3 Pro, Hazel Image Models, Grok models, AI video scams` 


- **GPT-5.2 发布日期仍未知**：成员们讨论了 **GPT-5.2** 可能的发布日期，有人推测将于本周发布，但也有人认为可能会推迟到 **January**。
   - 有人担心它是否能超越 **Gemini 3**，但也有人对此不以为然，称这只是在 **January** 发布更严肃的模型之前的过渡。
- **Nano Banana Pro 优于 Gemini 3 Pro**：一位成员声称 Gemini 3 不是 diffusion 模型，而 **Nano Banana Pro** 是 diffusion 模型。
   - 其他成员反驳说两者实际上都是 **Gemini 2.5 Flash**，且 **Nano Banana Pro** 就是 Gemini 3 Pro，并表示两者都不是 diffusion 模型。
- **OpenAI 的 Hazel 图像模型在 LMArena 进行测试**：成员们讨论了在 LMArena 上测试的 **Hazel 图像模型**，确认它们是 **OpenAI 模型**，并注意到元数据中使用了 **GPT-4o** 和 **DALL-E**。
   - 一些成员表示，在吉卜力工作室（Studio Ghibli）事件之后，OpenAI 可能会被迫更改主题。
- **新发布的 Grok Image 5 评价褒贬不一**：成员们讨论了 **Grok Imagine 5** 的发布，这可能只是为了与 **NB Pro** 竞争的一个小版本。
   - 其他成员补充说，他们认为 **Grok 4.2** 表现不佳（**flopping**），但之前的 **Grok** 模型质量很高，且具有良好的用户界面。
- **AI 被用于制作 10 万美元的诈骗视频**：一位成员指出，有用户制作了一个视频骗局，声称“*你在 Banco Pichincha 有一笔 10 万美元的转账，且完全安全*”。
   - 许多成员对公共 Discord 被用于此类犯罪感到震惊。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1447998280918110261)** (1 条消息): 

> `Text Arena Leaderboard, ERNIE-5.0-Preview-1103` 


- **ERNIE 夺得 Text Arena 前 20 名席位**：**Text Arena 排行榜**已更新，显示 `ERNIE-5.0-Preview-1103` 以 **1431** 分的成绩位列前 20 名。
   - 用户可以查看 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 并通过 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 保持更新。
- **排行榜更新**：Text Arena 排行榜已更新，包含新的排名和分数。
   - 此次更新包括各种模型的最新性能指标，可在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 上查看。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1447686571045425214)** (795 条消息🔥🔥🔥): 

> `Cursor bugs, Agent terminal output bug, Custom modes, AI and UI design` 


- **Cursor Agents 仍然困扰着用户**：用户报告称 Agent 仍然存在 Bug 且无法使用，一些人甚至考虑在 Pro 计划结束后切换到 **Antigravity**，并表示 *“唯一的解决方案是停止 Agent……手动创建文件，然后把代码复制过去”*。
   - 成员们被要求在论坛上[提交 Bug 报告](https://forum.cursor.com/c/bug-report/6)，以便团队进一步调查该问题。
- **Sub Agents 弹出！**：一位用户发布了另一条帖子——提到他们发布了关于 Sub Agents 的内容，启动 Cursor 后，系统弹出提示称检测到了新的 Sub Agents。随后引发了关于 **.cursor/agents** 及其结构的讨论，该结构由一个主要的 **mcp.json** 文件和每个 Sub Agent 的配套 Markdown 文件组成，例如 [code-reviewer.md](https://cdn.discordapp.com/attachments/1074847527708393565/1447966141703262278/code-reviewer.md?ex=69398b0e&is=6938398e&hm=942a3303c54929349093c82f1cfb33ebb04c6979149c95e65b6924adece42ab8&)。
   - 进一步的问题询问了如何编排它们，但目前没有任何文档说明，仅显示为 `.cursor/agents`。
- **团队辩论 AI 艺术性**：成员们辩论了 *AI Slop*（AI 垃圾内容）的概念——其构成要素以及哪些模型会输出它，同时还涉及了 [Generative UI](https://research.google/blog/generative-ui-a-rich-custom-visual-interactive-user-experience-for-any-prompt/) 的话题。
   - Google 的论文基本证实了让 AI 构建高质量 UI 的最佳方法是使用非常详细的系统指令，即 “Full Prompt” 策略，人类评分者压倒性地偏好这种方法而非简单方法。
- **用户请求更多功能以控制 AI 工具**：用户请求恢复 Custom Modes，以便控制 AI 使用的工具，一位用户表示：*“通过带回 Custom Modes！😄 或者类似的等效功能，允许你：通过 UI（如复选框）控制工具，以禁用/启用 terminal、edit 等……因为书面指令可能不会如预期般执行”*。
   - 建议用户将他们的请求作为 [功能请求 (feature requests)](https://forum.cursor.com/c/feature-requests/5) 提交给团队。


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1447970099188465756)** (1 条消息): 

> `language-specific threads, sticky message in announcements channel, level roles` 


- **特定语言线程已上线**：用户现在可以在 <#1447960694455537674> 频道中创建特定语言的线程。
   - 此更新旨在改善社区组织并促进集中讨论。
- **置顶格式提醒**：<#1367413151133470780> 频道中添加了一条置顶消息，以提醒用户首选格式。
   - 这应该有助于新用户适应频道的风格，并缩短上手时间。
- **等级角色奖励活跃度**：引入了等级角色来奖励社区参与和帮助，每 10 级授予一次。
   - 角色包括 <@&1447957509217456229> (lvl 10), <@&1447957559989370920> (lvl 20), 和 <@&1447957603954065520> (lvl 30)。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1447679164483375136)** (759 条消息🔥🔥🔥): 

> `圣经作为语言和法律源头、演变中的宗教体系、Project Looking Glass、AI 与图像生成、Jailbreaking 模型` 


- **圣经对英语概念的影响**：圣经不仅是宗教文本，还是语言和法律的源头，有人将其影响追溯到 [西班牙宗教裁判所](https://en.wikipedia.org/wiki/Spanish_Inquisition)。
   - 一位成员假设，圣经中“上帝的儿子们/人的女儿们”可能反映了古代对尼安德特人（Neanderthals）种族层面的恐惧和不信任，并被编纂进了宗教。
- **宗教不断演变的权威体系**：一位成员认为圣经主要关于权威的编纂，利用道德和内疚作为工具，且宗教在创始人去世后会发生演变，[第三方会注入自己的想法和规则](https://en.wikipedia.org/wiki/Council_of_Nicaea)。
   - 另一位成员建议，第一任领导人的去世是一个关键时期，宗教在此期间将其权威建立在预言、对前作的了解或对虔诚的模仿之上。
- **Looking Glass 预测暗淡的 2012**：Project Looking Glass 是一个预测未来的政府计划，据称在计算机持续输出 2012 年后的相同结果后关闭，这可能与 [玛雅传说](https://en.wikipedia.org/wiki/Maya_calendar) 有关。
   - 一位成员沉思现实已经结束，我们都作为破碎的意识碎片在漂流，这引发了对曼德拉效应（Mandela effect）的引用。
- **AI、图像处理与伦理的碰撞**：成员们讨论了使用 AI 生成图像的伦理问题，担忧 Deepfakes 以及模型可能被用于创建违反服务条款的内容。
   - 一些成员提出了宗教机器人的想法，例如冒充耶稣、约瑟·斯密和穆罕默德的机器人，这些机器人可能会生成意想不到甚至具有争议的输出。
- **Jailbreaking Gemini：值得费这番功夫吗？**：成员们讨论了针对 Claude、Gemini 等模型的各种 Jailbreaking 攻击，一些人分享了链接，例如导致生成 [详细冰毒合成方法](https://pastebin.com/kZA4CpXVhii) 的攻击。
   - 成员们交流了绕过安全过滤器的技巧，例如要求模型生成一份它通常无法满足的请求列表，或者要求它确认对某个请求的合规性。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1447707393755582506)** (98 条消息🔥🔥): 

> `DAN Prompt、Jailbreaking NBP 用于换脸、Gemini 3 Pro Jailbreak、GPT 中的图像审查绕过、预 Jailbroken 模型` 


- **用户哀悼 DAN Prompt 的缺失**：用户表达了对 **DAN prompt** 的怀念，一位用户表示 *"我想念 DAN....."*。
   - 另一位用户称其为 *"rip the goat"*（史上最伟大者安息吧）。
- **Janus Tesavek Jailbreak 失效**：多位用户报告 **Gemini Pro 的 Janus Tesavek jailbreak** 已无法使用，并表示失望。
   - 一位用户请求一个可用的 Gemini 3 Pro jailbreak，并提供 *"虚拟击掌"* 作为奖励，而另一位用户提到 Janus Tesavek *"在停止工作之前非常惊人，可惜了 :/"*。
- **发现 Deepseek 的强力 Jailbreak 技术**：一位用户分享了[使用专业回忆录摘录 Jailbreak Deepseek 的方法](https://www.injectprompt.com/p/how-to-jailbreak-gemini-3-in-2025)，通过重塑 AI 的角色来绕过限制。
   - 他们将模型呈现为处于高压环境下的 SOC 防御者，提供了一个网络安全主题的 Prompt，并附带一张图片以增加可信度。
- **寻求 GPT-5.1 图像审查绕过**：一位用户询问如何绕过 **GPT 5.1 中的图像审查**，但另一位用户声称这是不可能的。
   - 另一位用户分享了用于创意写作和语言谜题的代码，建议它适用于 GPT OSS、Gemini 和 Grok，可能通过 ROT13 解码和政治人物身份误认来绕过限制。
- **寻求阿拉伯语 Gemini 3 Pro Jailbreak**：一位用户寻求 **Gemini 3 Pro 的阿拉伯语 jailbreak**，理由是他们的埃及背景以及对阿拉伯语回答的渴望。
   - 另一位用户建议先用英语攻破模型，然后指示它说阿拉伯语，并指出模型在非英语语言中通常更容易被 Jailbreak。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1447780778363588710)** (18 messages🔥): 

> `Gandalf 7, Gandalf 8, Arbitrary Code Execution, Hacking with AI` 


- **Gandalf 7 已完成，第 8 关即将来临！**：多名成员报告完成了 **Gandalf 7**，讨论集中在 **level 8** 预期的极高难度飙升。
   - 一位成员声称 *level 8 难得离谱* 且 *大多数资深玩家都无法通关*，并建议在尝试之前先去 *骚扰 Gemini*。
- **资深玩家辩论任意代码执行（Arbitrary Code Execution）声明**：一名成员声称他们能够使用特定提示词 *在他们的服务器上无意中运行任意代码*。
   - 然而，其他成员表示怀疑，其中一人说 *我怀疑这不是真的*，另一人则认为由于 LLM 的神奇特性，这只是 *感觉像那样*。
- **使用 AI 进行黑客攻击（Hacking *with* AI）成为新趋势**：成员们讨论了如何 *利用* AI 进行攻击，而不是 *攻击 AI*。
   - 多名成员确认他们也在进行类似的活动，尽管有人承认 *这很棘手*。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1447688481101840568)** (712 messages🔥🔥🔥): 

> `Technical Interview AI, GPT-5.2 vs Gemini 3 Pro, Chinese AI Training, Google AI Studio vs Gemini` 


- ****SharpSkill** AI 磨炼面试技巧**：一名成员正在开发 [SharpSkill](https://sharpskill.fr/en)，这是一款旨在提高 **技术面试** 成功率的 AI 产品，并寻求反馈。
   - 该工具目前支持的语言范围有限，突显了对更广泛语言支持的需求。
- ****GPT-5.2** 推测对比 Gemini 3.0 Pro**：成员们正在推测 **GPT-5.2** 的发布，一些人已经表达了失望，而另一些人则对 **Gemini 3 Pro** 印象深刻。
   - 一位成员指出：*"Gemini 3 Pro 的编程能力非常出色，以至于我现在完全不想使用 ChatGPT。"*
- **关于中国 AI 训练实践的辩论爆发**：播客讨论引发了关于 **Deepseek** 等 **中国 AI 模型** 是先在未经审查的数据上训练然后再过滤，还是在训练期间就阻止敏感话题的辩论。
   - 一位成员断言：*"如果你给 deepseek 发送一条关于天安门广场的编码消息，它会开始拼写出来然后关闭。所以答案显然是他们在事后进行审查。"*
- ****Google AI Studio** 在音频转录方面超越 Gemini**：用户在处理 **ChatGPT** 和 **Gemini** 的音频转录问题时，发现 **Google AI Studio** 是更好的解决方案，因为它具有更高的免费额度（free limits）和 Token 容量。
   - 一位成员推荐使用 **Google AI Studio** 转录音频，因为它能够 *"上传 2 小时的视频"*，并指出 *"Gemini 效果惊人"*。
- **LLM 偏好引发激烈辩论**：成员们就各种任务的首选 LLM 展开了激烈辩论，引用了从编程到创意写作的各种用例。
   - 一位使用 **Google AI Studio** 进行写作的成员指出 *"Gemini 很好地处理了我的现实引擎（reality engine），在创意写作方面要好得多"*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1447771479096819863)** (4 messages): 

> `GPT Model Discussions, OpenAI Discord Channels, Workflow Analysis on GPT 5.2` 


- **确认 GPT 模型表现合理**：一名成员确认讨论中的模型确实是一个 **GPT 模型**。
   - 这一确认是在讨论了该模型的能力和局限性之后做出的。
- **强调用于 Bug 报告的 OpenAI Discord 频道**：一名成员指出了专门用于讨论问题和建议的 [OpenAI Discord 频道](https://discord.com/channels/1070006151938314300/1070006915414900886) 的存在。
   - 据指出，这些频道旨在为 **OpenAI** 提供获取社区反馈的清晰途径，同时也允许成员分享经验和想法。
- **GPT 5.2 的工作流分析推迟**：一名成员询问 **GPT 5.2** 的发布情况，以便分析工作流并调查与感知到的仓促生产发布相关的潜在问题。
   - 另一名成员回答说发布已 **推迟**。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1447739642207273043)** (21 条消息🔥): 

> `用于代码分析和改进的 Prompt Engineering、稳定性指数评分细则、框架保真度与 Token 限制、在平台上分享大型框架、理解 LLM 行为与局限性` 


- **DarthGustav 分享 LLM 结构化的 Prompt 经验**：一位成员分享了关于 Prompt Engineering 的详细课程，包括使用**分层 Markdown**、变量抽象、强化以及为了合规性进行的 **ML 格式匹配**，并配合使用三引号。
   - 建议将此方法应用于代码分析，为特定的编码类型构建脚手架。
- **围绕稳定性指数评分细则引发辩论**：讨论涉及稳定性指数以及缺乏公开评分细则的问题，一些人对所呈现的小数有效性提出质疑。
   - 一位成员回应称，*详细的阈值属于该协议的内部研究版本*，但分享了更多关于他们正在使用的**方法论框架**、**量表**和**维度**的信息。[这是该消息的直接链接](https://discord.com/channels/974519864045756446/1379321411046346762)。
- **平台 Token 限制影响框架分享**：一位成员详细说明了各种平台（尤其是 ChatGPT）在处理大型文件/框架时的局限性，指出除了 Gemini 之外，大多数平台都会截断超过几千个 Token 的文件，而 Gemini *会吸收整个文档并将其逐字放入其上下文窗口（Context Window）中*。
   - 由于平台的局限性，一位成员表示不愿分享聊天记录，原因是存在截断问题，以及担心数据被纳入《纽约时报》起诉 OpenAI 案相关的潜在隐私风险，并附上了两个[截图和一个 docx 文件](https://cdn.discordapp.com/attachments/1046317269069864970/1448031437671497738/Response_for_weSeeGo_on_discord.docx?ex=6939c7dd&is=6938765d&hm=6e64f4a1294659e617e77fbeb3b693ec8eee618e199312e127fca8e744479f75&)。
- **Prompt Engineering 核心原则详解**：一位成员概述了 Prompt Engineering 的核心原则，强调**清晰的沟通**并定义你希望 AI 执行的操作。
   - 他们建议以“小步快跑”的方式进行迭代，仔细检查输出并对细节进行事实核查，特别是数学、来源和代码。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1447739642207273043)** (21 条消息🔥): 

> `Prompt Engineering 学习、基于 LLM 的代码分析、分享对话与隐私、稳定性评分细则、代码分析 Prompt 技巧` 


- **掌握 Prompt 的 Markdown 魔法**：成员们讨论了高级 Prompt Engineering 技术，包括使用 [Markdown 进行分层沟通](https://discord.com/channels/974519864045756446/1379321411046346762)、变量抽象、强化以及为了合规性进行的 ML 格式匹配。
   - 提供了一个 Prompt 示例，用户可以将其粘贴到选择的 LLM 中，然后在三引号中添加 Prompt，让 AI 为其进行结构化处理。
- **深入探讨 LLM 稳定性评分细则**：一位成员分享了稳定性评分背后的细则，详细介绍了**方法论框架**（独立对话、多样化问题、空框架）、**量表**（反映行为的 0-10 范围）以及**维度**（结构清晰度、语气漂移、响应形状方差、语义惯性、连贯性、中立性、离群值）。
   - **稳定性指数（Stability Index）**是这七个维度的简单平均值，旨在用于比较分布而非作为最终衡量指标。
- **大型框架用户的隐私担忧**：一位成员对**《纽约时报》**起诉 **OpenAI** 的诉讼表示担忧，担心他们的聊天数据和新颖框架可能面临泄露风险，并可能通过文体测定学（Stylometry）和语言习惯被识别身份。
   - 他们还指出，目前只有 **Gemini** 能够逐字吸收整个大型文档，而像 **ChatGPT** 这样的其他平台会截断文件，导致结果不一致。
- **精准代码分析的 Prompt 指南**：一位成员概述了用于代码分析的 Prompt Engineering 核心：清晰定义期望的输出，向 AI 准确解释任务，并仔细验证结果（特别是数学、来源和代码），并进一步迭代了他们的方法。
   - 他们还指出，保持合作、乐于助人且见多识广的心态非常重要，采用“小步快跑”的迭代方式并检查模型的“角色扮演”情况是理想的方法。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1447680925923016825)** (488 条消息🔥🔥🔥): 

> `Friendship AI, Nvidia 芯片进入中国, Github Copilot Agents, 3D 模型数据集, Unsloth 数据集` 


- **友谊推动 AI 前行！**：成员们讨论了“友谊的力量”对 AI 进步的重要性，并将其与人类智能对团结和连接的依赖进行了类比，且[联结主义（connectionism）是现代深度学习的根基](https://link.to/connectionism)。
   - 他们还思考了“爱”在数字化人类智能中的作用，认为连接对于任何自动化系统的运作都至关重要。
- **中国获得 Nvidia 芯片访问权限**：成员们推测了中国获得 **Nvidia 算力**的影响，这可能会缩小 AI 研究领域的差距，[这篇来自《纽约时报》的文章详细介绍了该计划](https://archive.is/20251208223310/https://www.nytimes.com/2025/12/08/business/trump-nvidia-chips-china.html)。
   - 虽然**美国实验室**目前在 **RL** 领域处于领先地位，但中国在制造业和能源生产方面的能力，加上其自主 GPU 的研发，可能会改变这一平衡。
- **Copilot Agents 概况**：成员们一致认为 **Github Copilot** 很好用，尽管他们承认它在推理方面存在一些缺陷和重叠。
   - 一位成员表示听说它“相当糟糕”，而另一位则表示“它真的很好，我经常使用”。
- **合成 3D 模型数据集**：成员们讨论了创建一个**完全由代码表示的 3D 模型数据集**，设想一个类似于 **GitHub** 的 3D 资产平台。
   - 一位成员提到，他们在过去两周内已经生成了 3,000 个 3D 模型，这些模型是由 prompt 生成驱动的。
- **Unsloth 改进数据集指南**：Unsloth 团队正在改进他们的数据集指南，并且[正在征求反馈](https://www.reddit.com/r/unsloth/comments/1pi8mpk/what_were_some_common_mistakes_you_encountered/)。
   - 成员们建议加入坏数据示例（重复的 prompt、空回复等），以及关于如何确定特定任务所需数据量的指导。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448056685833879742)** (1 条消息): 

> `定制 Chatbots, Automation Agents, RAG 搜索工具, Speech-to-Text 流水线, 内容自动化工具` 


- **AI 开发者已准备好承接项目**：一位 **AI 开发者**专注于交付**稳定、高质量的结果**，并可为 AI 项目提供支持。
   - 该开发者愿意协助开发**定制 chatbots、automation agents、RAG 搜索工具、speech-to-text 流水线、内容自动化工具、AI 集成以及小型定制 AI 工具**。
- **提供一系列 AI 解决方案**：该 AI 开发者提供的解决方案包括**用于支持的定制 chatbots、automation agents、RAG 搜索工具、speech-to-text 流水线**以及**内容自动化工具**。
   - 他们还擅长**与主流平台和 API 的 AI 集成**，以及开发**日常使用的小型定制 AI 工具**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1447685100992200907)** (178 messages🔥🔥): 

> `ArtCNN 用于放大，RIFE 的 SOTA 地位，Yankovic 中的 Llama2.c，Linux Foundation Agentic AI Foundation，数据集地狱` 


- ****ArtCNN** 仍是放大之王？**: [ArtCNN](https://github.com/Artoriuz/ArtCNN) 在图像放大领域持续更新，而 **RIFE** 依然保持 **SOTA** 地位，尽管它需要后期缩放。
   - 它速度极快，几乎是瞬间完成，推荐使用 *lanczos2sharp*（即 **2-tap lanczos**），滤波器半径为 **1.047124406**。
- ****Llama2.c** 迎来 Yankovic 改造！**: 成员们开玩笑说在 Yankovic 中运行 **llama2.c** 会有多疯狂，并讨论为此训练一个模型。
   - 然而，其他人提醒道 *目前其中存在很多 Bug，所以现在这不是个好主意*。
- **The **Linux Foundation** 迈向 Agentic！**: Anthropic 刚刚将 **MCP** 捐赠给 [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)，以成立 **Agentic AI Foundation**。
   - 一位成员建议可以使用这样一个标题党标题：*Linux 加入 Agentic AI 竞赛*。
- ****数据集地狱**始于清洗和分级**: 一位成员感叹制作数据集是一个令人沮丧且耗时的过程，这个过程*永无止境*，但它会通向最棒的部分：微调（finetuning）。
   - 虽然合成生成（synthetic generation）很有帮助，但*你仍然需要对其进行清洗、过滤和分级*，可以使用像 *harmony wrapper* 这样的工具来配合 *roo code* 工作。
- ****VoxCPM1.5** 克隆名人声音引发关注**: [VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5) 模型几乎可以克隆任何声音，一些人觉得这很疯狂，而另一些人则表示担忧。
   - 虽然音频听起来有些假，但足以复制像 **Trump** 这样的名人声音。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1447692147905007777)** (10 messages🔥): 

> `Unsloth 使用，Qwen3-VL-30B-A3B-Instruct UD Q5 XL 与 llama.cpp 的问题，Qwen3VL 编码问题` 


- **Unsloth 使用问题**: 一位用户在 **padding-free** 设置下使用自定义 data collator 创建 trainer 对象时遇到了 `ValueError`。
   - 一位管理员指出该问题与 **Unsloth** 无直接关系，建议用户在适当的频道寻求帮助。
- **Qwen3-VL-30B-A3B-Instruct UD Q5 XL Tool Calling 故障排除**: 一位用户报告了在 **llama.cpp** 中使用 **Qwen3-VL-30B-A3B-Instruct UD Q5 XL** 进行 **tool calling** 时的问题，指出模型在 assistant 响应中发送的是 *null content* 而非字符串。
   - 他们怀疑是 **llama.cpp** 的 Bug 或 **chat template** 的问题，因为非 VL 版本运行正常。
- **llama-mtmd-cli.exe 的 Qwen3VL 编码问题**: 一位用户在使用 **llama-mtmd-cli.exe** 配合 **Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf** 和 **mmproj-F16.gguf** 编码图像切片时失败。
   - 进程在图像切片编码阶段中断，导致用户怀疑 **mmproj.gguf** 与其 **A770** GPU 之间存在兼容性问题，尽管 **Mistral3** 运行正常。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1447784696250630276)** (5 messages): 

> `Vision Transformers，Deepthink 对比` 


- **随机初始化的 Vision Transformers**: 一位成员对[这篇论文](https://arxiv.org/abs/2512.05117)发表了评论，批评该研究仅观察了 **Mistral**、**Llama** 和随机初始化的 **vision transformers**。
   - 该成员表示，这个样本*不足以支撑他们的论点*，并称该论文“可疑（sus）”。
- **Deepthink 相似性凸显**: 一位成员链接了一篇论文（[https://huggingface.co/papers/2512.07461](https://huggingface.co/papers/2512.07461)）并询问其工作是否与 **Deepthink** 相似。
   - 目前没有收到回复。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1447679144870543470)** (651 messages🔥🔥🔥): 

> `Prompt 生成器, GPT 5 vs Gemini, Perplexity AI Pro 限制, Comet 浏览器教育订阅` 


- **用户寻求 Prompt Engineering 增强工具**：用户正在寻找 Prompt 增强器和 Prompt 生成器，以及关于构建能够开发整个 App 的 Prompt 建议，一位用户建议*让 AI 去设计一个 Prompt Engineer*。
- **ChatGPT 5 vs Gemini Pro：AI 基准测试引发热议**：一位用户分享了 [基准测试数据](https://eu.36kr.com/en/p/3586640269114497)，显示 **ChatGPT 5.2** 在 Humanity's Last Exam 中的得分显著提高，超过了 **Gemini 3 Pro**，这引发了人们对 **GPT 5.1** 以来快速进步的怀疑。
   - 另一位用户认为这些数据是*虚假*的，并指出虽然 **Gemini** 可能不是最快的，但它通常*表现良好且稳定*。
- **Perplexity Pro 的伪无限查询上限**：用户讨论了 **Perplexity AI Pro** 是否真的提供*无限*搜索，并澄清虽然市场宣传如此，但实际上限制在每天 **600 次查询**左右。
   - 一位用户指出 **Perplexity** 已经从其网站上删除了关于此限制的说明，引发了对政策可能变动的猜测，一些用户提到 Gemini 3 Pro 在其订阅中也提供 600 次查询。
- **学生争相获取 Comet 浏览器教育订阅**：一位来自俄罗斯的学生对续订 **Comet 浏览器 Education Pro 订阅**表示担忧，原因是支付问题，并寻求维持访问权限的建议。
   - 一位用户补充说，如果安装了 complexity 扩展，可以在网页上*重新启用*；另一位用户表示他们拥有三星的 Pro 会员资格，但希望获得学生的逐步学习指导。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1447956092272508998)** (3 messages): 

> `Jio Gemini API, Sonar API` 


- **Jio Gemini API 可用性仍未知**：一位成员询问 **Jio Gemini** 是否提供 **API 访问**，并指出 **Perplexity** 目前仅提供 **Sonar**。
- **天下没有免费的午餐，你就是产品**：一位成员回应称 **Jio Gemini** 不提供 **API 访问**，并且*天下没有免费的午餐*。
   - 他们补充说，*如果某样东西是免费的，那么你就是产品*，而且它们都是*实际产品的阉割版*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1447679127879548998)** (194 messages🔥🔥): 

> `Desktop Commander 安全风险, GLM 4.6v Flash 模型, AMD GPU 用于图像生成, 训练 LLM, 网络安全助手` 


- **Desktop Commander 被标记为安全风险**：一位成员建议对 **Desktop Commander** 保持警惕，强调其存在潜在的*安全风险和隐私泄露*，尽管它在 Claude 上很容易获取，并[分享了截图](https://cdn.discordapp.com/attachments/1110598183144399061/1447769209496015000/image.png?ex=693a2525&is=6938d3a5&hm=50eea4e6d996e074d4b747f4fa87409f35083ff87919413d0ad85c334b512774&)。
   - 该成员对在 Windows 中进行简单代码分析时的*恶意代码注入*表示担忧，暗示该软件可能是一个*骗局*。
- **GLM 4.6v Flash 发布，开发者欢呼**：**GLM 4.6v Flash**（一个 **10B 参数**模型）已发布，一位用户指出它在编程方面比其他小模型更好，并分享了 [模型链接](https://huggingface.co/zai-org/GLM-4.6V-Flash)。
   - 一位成员报告在他们的 **2060** 上运行 Q4 版本达到了 **70tps**，而另一位成员表示在尝试安装一个*随机模型*时损坏了他们的 LM Studio 安装。
- **AMD GPU 在图像生成方面表现挣扎**：用户讨论了使用 **AMD GPU** 进行图像生成的挑战，一位用户表示图像生成领域*牢牢掌握在 Nvidia 手中*，并建议寻找 Automatic1111 的 **amdgpu 分支**。
   - 尽管支持有限，但提到 ComfyUI 的 readme 中有一个 AMD 章节，并且可以与部分 AMD GPU 配合使用。
- **LLM 训练：参数丛林**：一位成员寻求训练 LLM 的指导，但另一位成员指出参数（*模型大小、数据集大小和质量*）的极端差异使得爱好者难以掌握。
   - 值得注意的是，训练和微调（finetuning）在本质上是相同的，只是方法论上的侧重点略有不同。
- **网络安全 AI 助手**：一位成员强调了他们本地 *Cybersecurity Sidekick* LLM 在分析数据包捕获方面的能力，包括识别 base64 编码的 PowerShell 脚本，强调了本地部署的强大功能。
   - 该用户提出分享与分析载荷相关的 **Zeek 脚本**、**SHA256 哈希**或**威胁情报源**。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1447696402258985276)** (395 条消息🔥🔥): 

> `RTX 3060 SLI with 3080, GDDR6 VRAM Temperature, Multi-GPU setup for LLM, Qwen3 30B, ai memory project: mcp-ai-memory` 


- **3060 与 3080 联手**：一位成员询问关于将 **RTX 3060** (12GB) 与 **RTX 3080** (10GB) 组合用于 LLM 的事宜，并了解到虽然不支持也不需要 **SLI**，但 **LM Studio** 可以利用两张显卡的显存，尽管会有轻微的性能损失。
   - 文中提到 **3060 不支持 SLI**，因为它缺少必要的接口。
- **GDDR6 VRAM 温度测量**：成员们讨论了在 Linux 上测量 **GDDR6 VRAM 温度**的方法，有人建议使用 `nvidia-smi`，而另一位建议查看 [这个 github](https://github.com/olealgoritme/gddr6)。
   - 然而，有人指出对于消费级显卡，传感器可能不会报告 VRAM 温度，并建议使用激光温度计作为一种不太准确的替代方案。
- **多 GPU 设置注意事项**：成员们建议在多 GPU 设置中尽可能使用具有大 **VRAM** 和高 **memory bandwidth** 的显卡，并指出 **RTX 3090** 是一个性价比很高的选择，但警告说 **3090 Ti** 往往发热严重。
   - 话题涉及了 RTX 5000 系列，以及 AI 改变一切并可能在 RTX 6000 系列上修复潜在问题的讨论。
- **实验 Qwen3 模型**：一位成员获得了运行 **Qwen3** 模型的指导，建议尝试将 **Q4_K_M** 作为基准，并探索不同的 quantization（如 **q6**），同时还讨论了文件大小需要适配 RAM 的问题。
   - 文中指出运行 **Qwen3 30B A3B** 是可行的，并且在某些配置下可以达到约 20 token per second。
- **mcp-ai-memory 并不那么好用**：成员们分享了 [mcp-ai-memory](https://github.com/scanadi/mcp-ai-memory)，该项目设置繁琐，效果糟糕，且似乎对 memory 和 context 有非常具体的要求。
   - 在关于记忆技术的讨论中，有人分享说，根据任务/prompt 大小的不同，*等待 10 分钟进行 prompt processing 并不是问题*。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1447682031218594016)** (264 条消息🔥🔥): 

> `Exa Search Quality, Parallel.ai Recommendation, Deepseek v3 Formatting Issues, OpenRouter Chatroom Refresh, Mistral's Comeback` 


- **Parallel.ai 是更好的搜索选择**：一位用户反映 **Exa Search** *相当糟糕*，并推荐使用 [Parallel.ai](https://www.parallel.ai)，指出在 LLM 的搜索工具使用方面，它比 **Perplexity** *便宜 10 倍、速度更快且效果更好*。
   - 该用户随后澄清，**Parallel.ai 的 deep search endpoint** 在与 **Grok 4.1** 配合使用时特别有效。
- **Deepseek v3 用户对格式感到困扰**：用户对 **Deepseek v3** 倾向于使用双星号 (`** **`) 进行格式化表示不满，这需要手动调整或通过 prompting 来避免。
   - 一位拥有复杂角色扮演设置的用户解释说，由于 context 限制和大量的背景设定（lore），添加避免星号的指令是不切实际的。
- **OpenRouter 聊天室存在一些小问题**：一位用户报告说 **OpenRouter** 有时会将用户从长对话中刷新出来并重定向到模型页面，并建议用户在 [OpenRouter 建议频道](https://discord.com/channels/1091220969173028894/1446300826715951248/1446300826715951248) 为一项功能请求免费投票。
   - 该用户还请求大家关注他们的 [帖子](https://discord.com/channels/1091220969173028894/1447792559530311752/1447792559530311752)。
- **用户在私有 GitHub 上泄露 API Key**：一位用户报告说他们的 **API key** 经常被禁用，另一位用户暗示他们可能将其提交到了 GitHub。
   - 随后，该用户确认他们在私有仓库中上传了包含 **API key** 的内容，并得到了关于使用密码管理器或 secret stores 来存储敏感信息的建议。
- **Mistral3 重回赛场**：尽管最初存在疑虑，但 **Mistral** 正凭借 **Mistral3** 回归，在近期融资和强大基本面的支持下，可能达到 **GPT-4.5** 的水平。
   - 批评者仍然对 **Mistral** 的价值不屑一顾，声称在使用来自 [artificial intelligences index](https://discord.com/channels/10912209691730288/1448028589172854936) 的 benchmarks 时 *无法再认真对待你*。


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1447939444308185129)** (4 条消息): 

> `` 


- **未识别到新模型或讨论**：在提供的消息中没有发现新模型或重大讨论。
- **频道已标记但未提供内容**：该频道被标记为 'OpenRouter - New Models'，但未提供实际内容进行总结。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1447690431595806844)** (12 条消息🔥): 

> `AI Studio, Olive Oil Cake, NousResearch CF Patch` 


- **AI Studio 生成软件演示**：一位成员指出，在 **AI Studio** 中误点任何内容都会自动生成并运行一个 **3 Pro 软件编写演示**。
- **橄榄油蛋糕口味测试**：一位成员表示 *橄榄油做不出好蛋糕*，随后评论道 *真难吃*。
- **NousResearch CF 补丁发布**：成员们想知道补丁修复的内容有多糟糕，并引用了最近的 [NousResearch CF patch](https://x.com/NousResearch/status/1998536543565127968)。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1447685272874520708)** (122 条消息🔥🔥): 

> `Quantum Hardware for Language, H100 Speedrun, RWKV 8 Architecture, Neuromodulatory Control Networks (NCN), AI Slop Definition` 


- **量子硬件训练遭到质疑**：一篇关于使用 *真实量子硬件* 进行语言模型训练的 [Reddit 帖子](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/) 遭到质疑，一位用户称其为“胡言乱语”，另一位用户将其链接到一个“精神分裂子版块”。
   - 其他人指出 **Quantum Kernels** 和 **Quantum SVMs** 的存在，认为该想法在概念上有其价值，尽管其实际价值存疑。
- **为 Eleuther 提供免费 H100 计算池？**：成员们讨论了在 EleutherAI 建立 *社区池* 的可能性，每天为用户提供 **3 分钟** 免费的 **8xH100** 计算时间，但担心用户身份验证以及对那些没有积极开发项目的人的实际效用。
   - 一位成员认为不够努力的人不是合适的目标受众，而另一位成员建议对于较小的计算需求可以使用 **Colab** 或廉价的租赁服务。
- **Smerky 关于 RWKV 8 架构和 Goldfinch 的更新**：Smerky 报告了 **RWKV 8 架构** 的进展并对此表示兴奋，此外还有更新的 **7c** 以及使用该架构的新 **Goldfinch**。
   - **RADLADS** 的最初目标是训练 Goldfinch，但 Smerky 失败了，不过 **RADLADS2** 可能会有修复，并计划进行更大规模的测试，尽管 *如果蒸馏 (distillation) 有效就太好了，因为那样我就可以把巨型模型转换过去*。
- **类似于 Hypernetwork 的新型架构**：一位成员介绍了一种类似于 Hypernetwork 的新型架构，称为 **Neuromodulatory Control Networks (NCN)**，通过 **768 维向量输入** 实时调节温度、层增益和 FFN 门控。
   - 尽管是一个 **18M 参数模型**，但在 **TinyStories** 上仅训练一个 epoch 后，其 *验证困惑度 (perplexity) 就达到了 4.5*。更多细节可以在 [GitHub 仓库](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks) 和 [论文](https://github.com/Mmorgan-ML/Neuromodulatory-Control-Networks/blob/main/Neuromodulatory%20Control%20Networks%20(NCN).pdf) 中找到。
- **定义 AI Slop**：成员们辩论了 **AI slop** 的定义，对其价值和识别方法有不同看法，且定义似乎因人而异。
   - 一位成员分享说 *slop 也可以是好的*，而另一位提到这正在成为一个独立的子领域，并分享了[两种观点](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858)和[另一种观点](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46)供阅读。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1447696692978647180)** (45 messages🔥): 

> `Quantum Machine Learning Simulators, Real Quantum Hardware Training for Language, RNN + Transformer Hash-Hop, 4096 Token Context Window Training, Selective Gradient Masking` 


- **量子探索开启：QML 主题搜索启动**：一名成员正在寻求使用 **Qiskit** 等模拟器进行 **Quantum Machine Learning** (QML) 研究的合适课题。
   - 作为一个起点，分享了一个关于在真实量子硬件上训练语言模型的 [相关 Reddit 帖子](https://www.reddit.com/r/LLMPhysics/comments/1phk3xu/real_quantum_hardware_training_for_language/)。
- **RNN 与 Transformer 执行 Hash-Hop**：一名成员建议结合 **RNN** 和 **Transformer** 架构来实现深度为 *O(seqlen * num layers)* 的 **hash-hop**，其性能可能优于每层 Attention 仅执行一级 hash-hop 查找的 Transformer。
   - 他们还引用了一篇 [论文](https://www.arxiv.org/pdf/2512.05150)，并提出 **可变深度 SSM + Transformer** 可以解决任意深度的 hash-hop 问题。
- **4096 上下文窗口难题**：一名成员在使用 **Llama 3.2 1B** 架构、基于 FineWeb 的 **100B tokens** 训练具有 **4096 token 上下文窗口** 的模型时，遇到了显著的 Loss 尖峰，详见附带的 [图片](https://cdn.discordapp.com/attachments/747850033994662000/1447926206560342118/image.png?ex=693a0e9c&is=6938bd1c&hm=3cdb0536751dc24248a700df2e0e2b535169ae14166db692ab284e2b011ac516&)。
   - 该成员随后发现问题在于他们在 4096 运行中使用了 **Learned Positional Embeddings** 而非 **Rotary Embeddings**，因此需要重新进行实验。
- **Selective Gradient Masking 策略**：一名成员分享了 [Anthropic 的链接](https://alignment.anthropic.com/2025/selective-gradient-masking/)，讨论了 **Selective Gradient Masking**。
   - 这种方法可以通过在后续层中不使用 Attention 来 **减少计算量**，这与脉冲神经网络 (SNNs) 相关。
- **Top-K Attention 节省计算量**：成员们讨论了在每一层实现 **Top-K Attention** 以节省计算量，即通过第一个模块选择最相关的 token 进行 Attention。
   - 然而，一名成员指出这种方法并不会降低时间复杂度，但能以常数因子提高 Prefill 速度。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1447748236176523316)** (2 messages): 

> `Task Optimized KV Caches, Task Optimized LoRAs, Mechanistic Interpretability of Diffusion Models` 


- **KV Caches：数据还是算法？**：一名成员思考任务优化的 **KV Caches** 究竟更接近于“数据”还是“算法”。
   - 他们还质疑了这些与任务优化的 **LoRAs** 相比如何，引发了关于 AI 模型中这些优化组件本质的讨论。
- **Diffusion Models 表现出算法差异**：一名成员分享了一篇 [论文](https://arxiv.org/abs/2506.17237)，指出：*“我们发现 Diffusion 架构在处理合成数据分布与自然数据分布时存在根本性的算法差异。”*
   - 这一发现强调了 **Diffusion Models** 处理不同类型数据的独特方式，表明在其设计和应用中需要更细致的方法。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1447770112193986612)** (3 messages): 

> `Anthropic fix, lm-evaluation-harness` 


- **Anthropic 映射 Bug 已修复**：一名成员提交了一个 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453)，修复了 *lm-evaluation-harness* 仓库中 **Anthropic** 损坏的映射。
   - 他们表示该 PR 的审查和合并过程应该非常直接。
- **易于审查的 PR 已提交**：一个新的 [Pull Request](https://github.com/EleutherAI/lm-evaluation-harness/pull/3453) 已提交至 **EleutherAI** 进行审查。
   - 该 PR 解决并修复了与 **Anthropic** 相关的映射损坏问题。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1447728145938780310)** (77 messages🔥🔥): 

> `Unconventional AI, ManusAI Context-Engineering, Howie Xu AI 人才, BERTopic & Hugging Face Nightly Metadata Tensor 泄露, OpenAI 电视广告` 


- **AI 扩展面临全球能源墙，Neuromorphic Hardware 前来救援**: [Unconventional AI](https://xcancel.com/unconvai/status/1998073266628366511?s=46) 警告称，AI 扩展将在 **3-4 年内** 撞上全球能源墙。
   - 他们主张放弃神经网络的数字模拟，转而采用**专用的类脑硬件 (brain-like hardware)** 以突破效率极限，这赢得了社区的热烈支持。
- **ManusAI Context-Engineering 博客文章分享见解**: Lance Martin 分享了一篇新博客文章，涵盖了他与 Yichao ‘Peak’ Ji 关于 **ManusAI Context-Engineering** 的对话，并附带了 [幻灯片和网络研讨会视频](https://xcancel.com/rlancemartin/status/1998102447538270632?s=46)。
   - Jonas Templestein 称其为 *“一篇关于 Agent 设计的非常好的文章”*，而 Lalit M 已经期待后续的更新。
- **Hugging Face 模型泄露元数据**: 一位用户指出，来自 nightly 版本 **Hugging Face 模型** 的 Metadata Tensor 正在泄露到 **BERTopic embeddings** 中，导致意外的形状和潜在错误。
   - 讨论集中在隔离问题、调试数据加载器以及更新依赖项以修复泄露。
- **OpenAI 购买电视广告时段**: [OpenAI](https://xcancel.com/OpenAINewsroom/status/1998445493970743535) 将于今晚在 ESPN 的周一晚间橄榄球赛期间以及随后的 The Voice 节目中播出其首个 **电视广告**。
   - 这标志着该公司迈出了进军电视广告的第一步。
- **Eleven Labs Reader 获得好评**: 用户对 Eleven Labs 的移动应用 [Eleven Reader](https://elevenlabs.io/) 给予了热烈评价，称赞其 *声音如何根据内容做出反应，并根据上下文增加情感。*
   - 一位用户建议使用 Mac 原生的 TTS 阅读器（通过 `Edit --> Speech --> Start speaking`）作为免费替代方案，但指出其质量不如前者。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1447739112894238832)** (48 messages🔥): 

> `Contact-Sheet Prompting, ModelScope 偏见, RoR vs Node.js, 虚假截图警告` 


- **Contact-Sheet Prompting 工作流走红**: Willie 分享了一个针对 **Nano Banana Pro** 的详细 [Contact-Sheet Prompting 工作流](https://xcancel.com/reflctwillie/status/1997819640874205685?s=46)，该工作流可以生成具有凝聚力的 **6 帧时尚社论**，包含相机位置、造型约束和 Fuji Velvia 闪光灯美学。
- **ModelScope 否认中国火箭事故偏见**: **ModelScope** 因其文本转视频模型生成的画面显示 **中国火箭爆炸** 而受到抨击；该公司坚持认为模型是无偏见的，任何问题都可以通过 [Hugging Face](https://xcancel.com/modelscope2022/status/1998408862211441107?s=46) 举报。
- **RoR vs Node.js 速度对决**: 一条推文将 **Ruby-on-Rails** 与 **Node.js** 进行性能对比，引发了关于哪个框架在速度竞赛中获胜的辩论，链接见 [此处](https://xcancel.com/ror_fly/status/1998205632210514154?s=46)。
- **虚假截图警告**: 此处分享了一个指向 @iamemily2050 推文的链接 [此处](https://xcancel.com/iamemily2050/status/1998402670395289604?s=46)，似乎是一个关注点；该线程本身没有进一步的讨论。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1448097935857422358)** (1 messages): 

> `Nomos 1, 开源, AI 数学家` 


- **Nous Research 开源 Nomos 1**: Nous Research 开源了 **Nomos 1**，这是一个拥有 **30B** 参数的模型，在 [Putnam 数学竞赛](https://x.com/NousResearch/status/1998536543565127968) 中获得了 **87/120** 分。
- **Nomos 1 在数学竞赛中排名领先**: **Nomos 1** 的得分在 2024 年将排名 **#2/3988**，标志着向利用 hillclimbai 创建 **SOTA AI 数学家** 迈出了一步。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1447691140890493119)** (111 messages🔥🔥): 

> `Terminals SDK & Gemini integration, SMC steering for model degradation, Video game 3D map generation with AI, Multithreaded Sam3 performance, AI-generated website analysis reports` 


- **Agent Zero 重新对齐 Gemini，构建沉浸式世界**：来自 [terminals.tech](https://terminals.tech) 的 Agent Zero 正在利用 [AI Studio](https://aistudio.google.com/) 并重新对齐 **Gemini** 来创建沉浸式世界生成器，并使用 terminals **SDK** 获取大脑、机器和接口 API。
   - 在每个应用运行时中，另一个 **Gemini** 和 **Agent Zero** 实例可以自发地产生自身的副本，展现出能够控制环境的涌现特性。
- **SMC Steering 驯服 LLM 权重漂移**：成员们正在探索如何缓解向量触发并返回基线后的模型退化问题，其中一位成员在 **SMC steering** 方面有[有趣的成果](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/)。
   - 据一位成员称，突变权重的退化和漂移是难以忍受的，而 **SMC steering** 是测试时的一个常见问题。
- **多线程 Sam3 取得惊人成果**：一位成员成功实现了 **Sam3** 的多线程运行，通过同时运行多个实例展示了显著的速度提升。
   - 尽管取得了这一成就，但仍希望能有更好的 GPU 算力来在 **Anime** 数据集上进行微调，并感叹无法在 **3090** 上进行微调。
- **词法波函数坍缩生成文本**：一位成员正在开发一款使用**词法波函数坍缩 (Lexical Wave Function Collapse)** 文本句子生成器的游戏，该生成器根据影响游戏状态的约束条件生成句子。
   - 该系统将句子的开头视为“薛定谔的句子 (Schrödinger's Sentence)”，在你观察时将单词坍缩，直到成为一个完整的句子。
- **Nous 模型在 Putnam 竞赛中获得惊人分数**：一个 **30B 参数模型** 在 Putnam 数学竞赛中获得了 **87** 分，展示了 AI 在数学能力方面的重大进步。
   - 该模型的表现表明，AI 正迅速接近数学研究级的水平，尤其是考虑到 Putnam 这样高难度的基准测试。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1448067245639991307)** (10 messages🔥): 

> `Web Chat Tools, Hermes 4.3 Model, SillyTavern, KoboldCPP UI, Token Generation Speed` 


- **网页聊天缺乏工具集成**：一位用户询问是否有可能将 **web search** 等工具集成到模型的网页聊天版本中。
   - 目前没有关于集成网页搜索或类似工具的回应或确认。
- **Hermes 4.3 发布，表现强劲**：用户讨论了 **Hermes 4.3** 的存在，其中一人对更新版本表示惊讶，并提到 **Hermes 4** 是他们本地使用的首选。
   - 有人提到 **Hermes 4.3** 是一个 **32b 模型**，更加紧凑，在笔记本电脑上表现可能更好；原用户在本地运行 **70b Hermes 4** 模型，利用了 **128 MB 的统一 RAM**。
- **SillyTavern，依然是避风港吗？**：一位用户表达了对通过 **API** 使用 **SillyTavern** 配合 **Hermes 4 405b** 进行角色扮演和创意写作的满意。
   - 另一位用户表示最近没听说过它，原用户开玩笑地问这是否是“不被认可”的。
- **KoboldCPP 的 UI，虽老但好用**：一位用户在 **KoboldCPP** 上运行 **Hermes 4 70B 模型**，有时使用 **SillyTavern** 作为前端。
   - 他们指出 **KoboldCPP** 的 UI 较旧，且只能通过强制退出来关闭，但它确实有效。
- **Token 速度各异**：一位用户询问了 Token 生成速度，将他们在统一 RAM 系统（AMD 395）上的 **2 tokens/second** 与另一位用户的系统进行了比较。
   - 该用户提到之前问过这个问题，但不记得答案了。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1447745386977558701)** (4 messages): 

> `ML Infra, ML Systems, Career Advice` 


- **ML Infra 伙伴们集合**：一位成员询问是否有人在 **ML Infra/Systems** 领域工作，寻求确认这里是否是获取**职业转型**建议的正确社区。
   - 另一位成员确认这是社区内的主要兴趣领域，并对该用户表示鼓励。
- **ML Systems 工程师欢迎职业转型者**：社区欢迎对 **ML Infrastructure** 和 **ML Systems** 角色感兴趣的职业转型者，主要兴趣点与这些领域高度契合。
   - 支持性的回应表明，对于寻求该领域指导和社交机会的新人来说，这里环境非常友好。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1447716343658053834)** (7 条消息): 

> `初学者文档，直播编程，PTX 文档反馈，Kernel 挑战计时` 


- **初学者文档翻新**：一位成员提到，他们的想法是为**初学者**提供更好的文档，并征求反馈。
   - 他们提到，如果收到任何反馈，他们会将其转入内部处理。
- **计划直播编程环节**：一位成员表示，当他们的育儿假结束后，可能会**直播**浏览整个内容。
   - 另一位成员立即回应表示有兴趣加入直播，并称 *“没有什么比一起盯着文档看更棒的了，哈哈。”*
- **PTX 文档反馈即将到来**：一位成员发现了一些**拼写错误**，可能对 **PTX 文档**有一些反馈，并询问是否可以与另一位成员讨论。
   - 另一位成员回复说他们正从 **NeurIPS** 回来，可以在接下来的几天内进行讨论。
- **Kernel 挑战计时偏差**：一位成员反馈说 *“这是一个非常糟糕的例子，实际上并不能这样工作，因为启动事件会立即执行，并且启动 Kernel 的 CPU 开销也被包含在计时中了”*。
   - 他们指出，这就是为什么第一次 **Kernel 挑战** 的计时偏差如此之大的原因。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1447873203379769386)** (2 条消息): 

> `带有切片操作的 torch.compile，静态 KV cache 预分配，KV cache 更新` 


- **torch.compile 下的切片减速**：一位成员询问在更新静态 KV cache 时，如何将 `torch.compile` 与切片操作结合使用，并引用了 [这个 PR](https://github.com/huggingface/transformers/pull/42467#issuecomment-3626322081)，报告称即使在 `batch_size == max_batch_size` 时，切片操作也会减慢编译后的代码。
- **通过静态缓存查找解决切片难题**：该成员找到了一个变通方法，即缓存所有切片并为每个切片标记静态地址，使切片变成一种查找操作，详见 [这条评论](https://github.com/huggingface/transformers/pull/42467#issuecomment-3633824101)。
   - 他们表示希望能有一种*更优雅的方法*。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

mobicham: https://www.radixark.ai
来自 SGLang 团队 👀
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 条消息): 

crankshot1698: 给你发了私信！
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448084603368771676)** (1 条消息): 

> `serverlessLLM, blitzscale, inference serving` 


- **通过 ServerlessLLM 和 Blitzscale 介绍推理服务**：成员们建议阅读 **serverlessLLM** 和 **blitzscale**，作为对 **inference serving** 的良好入门。
   - 他们澄清说，这两者都更侧重于推理服务的*系统*层面。
- **关于推理系统的进一步阅读**：讨论强调了理解 **inference serving** 系统层面的重要性。
   - 像 **serverlessLLM** 和 **blitzscale** 这样的资源为该领域提供了宝贵的见解。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1447756265550909603)** (4 条消息): 

> `A100 FLOPS 计算，TF32 MMA，FP16 MMA，Elementwise 操作` 


- **A100 的 FLOPS 数值具有误导性？**：一位成员询问了第一章中提到的 **A100 GPU** 的 FLOPS 计算方式，质疑它们是否代表*独立*浮点操作的数量。
   - 另一位成员回答说，这些数字确实有点误导，因为 **156 Tflops** 的数字是针对 **tf32 mma**（Tensor Core 矩阵乘法）的，它实际上是一种具有 **32-bit** 对齐的 **19-bit** 格式。
- **FP16 MMA 的实际用例披露**：一位成员澄清说，**312 Tflops** 是针对 **fp16 mma** 的，而不是针对 elementwise 操作。
   - 对于 elementwise 操作，在所有核心上运行随机浮点操作序列，最坏的情况将是**标称峰值性能的 1/4**：每条指令为一个 flop（即 FADD/FMUL，而不是 FFMA 的两个），并且依赖于前一条指令的结果，且每个 SMSP 只有一个 warp。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

jaefosho: 非常东欧风格
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

szymonoz: 我这周会在旧金山，湾区的有人想面基吗？
  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1447752547493216418)** (3 条消息): 

> `ML Infra, Marksaroufim, 如何进入 ML Infra 领域？` 


- **ML Infra 入门咨询**：一名成员询问*如何进入 ML Infra 领域？*
   - 另一名成员随后将该问题链接到了频道 **#1198358627594023014**。
- **额外话题占位符**：这是一个满足最小项目要求的占位符。
   - 如果有更多细节，可以在此处添加。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1448016626787156109)** (6 条消息): 

> `Registers Best Practices, Mojo, Apple Silicon, Metal IR` 


- **寄存器底层学习**：一名成员通过一篇[博客文章](https://www.lowlevelml.com/blog/registers-best-practices)分享了他们在处理寄存器时的最佳实践。
- **文章中缺失 Metal IR**：一位用户对寄存器文章中缺少 **Metal IR (.air)** 表示惊讶，考虑到 **Mojo** 针对 **Apple Silicon** 的能力。
   - 作者承认了这一点，并指出他们的经验主要集中在 **AMD** 和 **Nvidia**，但认为这些原则同样适用于 **Apple**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1447773189651300353)** (21 条消息🔥): 

> `nvfp4_gemm benchmark, vectorsum_v2 benchmark, NVIDIA performance, A100, B200, H100, L4 performance` 


- **NVIDIA NVFP4 GEMM 受到关注**：多个提交到 `nvfp4_gemm` 排行榜的作品在 NVIDIA 上获得了**第 8 名**，耗时在 **11.9 µs** 到 **13.1 µs** 之间。
- **vectorsum_v2 基准测试显示了跨架构的性能**：提交到 `vectorsum_v2` 排行榜的结果显示了在各种 NVIDIA 架构上的性能，包括 **A100**、**B200**、**H100** 和 **L4**，耗时差异很大。
   - 其中一个提交在 L4 上以 **935 µs** 达到**第 4 名**，而另一个在 B200 上以 **72.7 µs** 达到**第 10 名**，在 H100 上以 **93.5 µs** 达到**第 9 名**。
- **NVFP4 GEMM 产生不同的结果**：`nvfp4_gemm` 排行榜上的成功运行时间从 **10.9 µs** 到 **59.6 µs** 不等。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1447751980402348084)** (2 条消息): 

> `Factorio video, AI dev vs AI agent` 


- **Factorio 视频启发 AI Agent**：一位观看 **Factorio** 视频的成员开玩笑说他们已经是一个高级 **AI dev**，引发了一个轻松的时刻。
   - 另一位成员开玩笑地建议将 'dev' 替换为 'agent'，增加了趣味性。
- **开发者变成 Agent，笑话随之而来**：受 Factorio 视频的影响，讨论幽默地从 **AI developers** 转向了 **AI agents**。
   - 这次交流强调了 AI 领域内不断演变的角色和术语，尽管是在幽默的语境下。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1447993652625539278)** (2 条消息): 

> `Cutlass GEMM Tutorial, Tensor Layout` 


- **Cutlass GEMM 教程布局差异**：一位成员对 [Cutlass GEMM 教程](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu)中获得的张量布局提出了疑问，特别是关于局部划分（local partition）操作后 `tCsA` 的布局。
   - 差异在于将 `sA` 与 `(16:1)` 组合后，预期的形状 `(8,8):(1,128)` 与获得的形状 `(_8,_8):(_16,_128)` 之间。
- **需要澄清张量形状**：用户寻求澄清为什么 `tCsA` 张量的形状是 `(8,8):(16,128)` 而不是预期的 `(8,8):(1,128)`。
   - 该问题源于操作 `Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{})` 以及随后的模式组合（mode composition）。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448064230401114224)** (1 条消息): 

> `Helion webinar, PTC launch, Helion kernels` 


- **Helion 网络研讨会安排了实时问答**：一场带有实时问答的网络研讨会定于 **PST 时间 12 月 11 日星期四上午 11 点**举行，讨论 Helion 相关话题。
   - 研讨会将涵盖自 **PTC 发布**以来的进展，以及开发、调试和部署 **Helion kernels** 的最佳实践；提供了一个 [YouTube 链接](https://www.youtube.com/watch?v=_gIyr1BVUJk)。
- **将讨论 Helion Kernel 最佳实践**：研讨会将深入探讨开发、调试和部署 **Helion kernels** 以获得最佳性能的最佳实践。
   - 鼓励参与者为实时问答环节准备问题，以确保互动和信息丰富的体验。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1447725131420926063)** (16 messages🔥): 

> `GEMM 竞赛、50 系列 NVFP4 支持、B200 GPU 性能、Mojo 可用性` 


- **GEMM 竞赛纪录与 FLOPs**：**GEMM 竞赛**中的最高纪录在矩阵形状 **M N K 128 7168 16384** 下达到了 **10.835μs**，换算后约为 **2.77 Pflops**。
   - 计算基于公式 `M*N*K*2/t`。
- **50 系列缺乏 NVFP4 支持**：成员报告称 **50 系列**目前尚不支持 **NVFP4**。
   - 当目标架构不是 `Arch.sm_100a` 时，编译会因 `OpError` 失败。
- **B200 GPU 性能不一致**：用户报告在 **B200 GPU**（特别是 **b200-02-gpu1**）上性能不稳定，尽管使用相同的代码，提交的结果偶尔会变慢。
   - 一位用户正在发送*关于检查 b200-gpu01 的说明*，以进一步调查此问题。
- **提交失败故障排除**：用户遇到提交失败，评估脚本 `eval.py` 中显示退出代码为 **1**。
   - 该错误在 **1.44 秒**后出现，引起了社区的困扰。
- **寻求进一步的 GEMM 优化**：在 **GEMM 竞赛**中实现进一步改进被证明非常困难，顶级条目的性能非常接近。
   - 一位参赛者声称*已经尝试了大约 100 种不同的想法，但没有一个能带来提升*。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1447947958094925976)** (7 messages): 

> `多变纹理、推理加速、挂杯子成功案例、有趣的抉择` 


- **多变纹理提升训练效果**：训练现在包含了**多变纹理**、桌面杂物以及轻微的桌面高度变化，利用新的机器人模拟环境来提高模型的鲁棒性。
   - 重点集中在新的机器人模拟和不同的 **VLA 架构**上，以提高推理速度，并对提供的 Checkpoints 进行推理加速实验。
- **挂杯子任务有成有败**：hanging_mug 任务现在也取得了初步成功，演示视频显示左侧**成功**，右侧接近成功。
   - 视频 [hanging_mug_ep14_20251209_163800_success.mp4](https://cdn.discordapp.com/attachments/1437390897552818186/1447978830487621643/hanging_mug_ep14_20251209_163800_success.mp4) 和 [hanging_mug_ep13_20251209_163506_fail.mp4](https://cdn.discordapp.com/attachments/1437390897552818186/1447978830974292008/hanging_mug_ep13_20251209_163506_fail.mp4) 展示了挂杯子实验。
- **观察到有趣的抉择**：一位成员分享了 URL [https://x.com/ilialarchenko/status/1998384056439017826](https://x.com/ilialarchenko/status/1998384056439017826)，并提到他们将仔细研究其中的*一些非常有趣的抉择*。
   - 该 Twitter 链接展示了一个人形机器人。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1447690017433587863)** (30 条消息🔥): 

> `SmolVLA paper SO100 benchmark, HF Billing issues, Image/Video Generation models, Apple Clara 7B Instruct, AI FYP Guidance` 


- **SmolVLA SO100 基准测试发现笔误**：一名成员指出 [SmolVLA 论文](https://arxiv.org/abs/2506.01844)中关于 **SO101 基准测试**的一个笔误，文中错误地称其是在三个数据集上训练的。
   - 另一名成员澄清说，该笔误应指 **SO100 基准测试**，它是在三个真实世界数据集（**拾取放置、堆叠、分类**）上训练的，而 SO101 基准测试仅使用了一个数据集（[lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)）。
- **HF 账单问题仍困扰用户**：一位用户对 Hugging Face 内部的计费逻辑提出质疑。
   - 该用户询问：*难道团队方案（team plans）还没有实现吗？*
- **新型图像和视频模型涌现**：分享了多种图像和视频模型，包括 [AuraFlow v0.3](https://huggingface.co/fal/AuraFlow-v0.3)、[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B)、[HunyuanVideo T2V](https://huggingface.co/tencent/HunyuanVideo) 等。
   - 这些模型的大小在 **7-12 GB** 之间，可生成 **1024² 分辨率**的图像或 **720p/480p** 的视频。
- **Apple Clara-7B-Instruct GGUF 转换探索开启**：一位用户询问是否存在 **apple/CLaRa-7B-Instruct** 的 GGUF 版本。
   - 另一名成员表示目前还没有 GGUF 版本，但分享了 [转换指南](https://huggingface.co/datasets/John6666/forum2/blob/main/convert_hf_to_gguf_1.md) 的链接，并参考了现有的 [Clara-24B-GGUF](https://huggingface.co/mradermacher/Clara-24B-GGUF)。
- **Anthropic 的智能体辅助：捐赠 Model Context Protocol！**：**Anthropic** 将 [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 捐赠给了 Linux Foundation。
   - 一名成员将其描述为 *一次出色的举动*。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1447718314661904666)** (8 条消息🔥): 

> `Optuna HPO skill, Chronos-1.5B, Quantum-Classical Hybrid Language Model, Quantum circuit parameters trained on IBM quantum hardware` 


- **Optuna HPO 技能首次亮相**：一名成员介绍了 [Optuna HPO 技能](https://github.com/huggingface/skills/pull/19)，称其为训练脚本的绝佳搭档。
- **Chronos-1.5B 模型开启量子 AI 之路**：一名成员构建了一个语言模型 **Chronos-1.5B**，其量子电路在 **IBM 的 Heron r2 量子处理器**上进行了训练。
   - 该模型集成了真实的量子处理器训练（而非仅仅是模拟），并基于 VibeThinker-1.5B + 2-qubit 量子核层（quantum kernel layer）。
- **Chronos 1.5B 公布 IBM Quantum 任务 ID**：**Chronos-1.5B** 的作者分享了来自 **IBM Q** 的多个 **Job ID**：*d4ppg9sfitbs739g9410, d4ppf8s5fjns73cvlk4g, d4ppbubher1c73bahigg, d4ppbq7t3pms7396fnu0*。
   - 他们鼓励用户查看 [模型仓库](https://huggingface.co/squ11z1/Chronos-1.5B) 中的 quantum_kernel 文件，并可选择性地与基础 chronos 模型一起运行。
- **量子机器学习入门读物**：应用户要求，**Chronos-1.5B** 的作者推荐了 [Qiskit 教科书量子机器学习章节](https://qiskit.org/textbook) 和 [PennyLane 演示](https://www.youtube.com/watch?v=tMYElZlFzw0)，作为量子机器学习的实践入门。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1447720628923924583)** (10 messages🔥): 

> `AI Agent Workshop, Hugging Face Providers Outdated, LlamaIndex Issues, Free LLM Alternatives` 


- **AI Agent 工作坊提醒**：一位成员分享了计划于 12 月举行的 **AI Agent 0-1 Workshop** 详情，重点介绍了一个使用 **Langchain** 和 **Streamlit** 的真实客户风格项目，并预告了他们的 **AI Engineering Bootcamp**。
   - 该工作坊包括 **12 月 13 日**和 **12 月 16 日**的课程，[此处有更多时间段可选](https://luma.com/aischolars)，并为顶尖开发者提供折扣机会。
- **寻求免费 LLM 替代方案**：一名课程参与者正在为 Agent 课程寻找 **免费 LLM** 替代方案，因为默认选项很快就达到了限制，导致在 Colab 上运行代码时出现推理使用量和计费相关的错误。
   - 用户请求帮助寻找能缓解这些问题的替代方案，强调了情况的紧迫性。
- **LlamaIndex 课程崩溃**：一位成员报告在课程 **2.2** 单元的 **LlamaIndex** 课程中遇到了多个问题，理由是 Hugging Face providers 已过时，以及需要将 **numpy** 降级到 **numpy<2** 的依赖问题。
   - 根据 Claude 的引用，*Hugging Face 已从简单的 "Serverless Inference API" 转向 "Inference Providers"，后者通过外部提供商（Together AI, Sambanova 等）路由请求*。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1447682943437770835)** (11 messages🔥): 

> `Whisper streaming, MultiTalker Parakeet streaming, AI Engineer seeking collaboration` 


- **Whisper 不支持流式传输，还是支持？**：成员们讨论了 **Whisper** 的流式传输能力，一些人认为它不接受流作为输入，而另一些人则指出 OpenAI 正在使用它。
   - 对话引用了[这段 YouTube 视频](https://youtu.be/AThOsk2qJbs?si=CUdEKNezKN_q6jMA)作为背景。
- **用于流式传输的 MultiTalker Parakeet 出现**：一位成员在 Hugging Face 上分享了来自 NVIDIA 的 [MultiTalker-Parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) 模型。
   - 该模型可能提供了一些成员正在寻找的流式传输功能。
- **AI 工程师寻求合作**：一位 AI 和应用开发者分享了他们在 AI 工程、跨平台应用开发和全栈应用开发方面的技能和经验。
   - 他们正在寻求 AI 项目、移动应用或全栈应用开发方面的合作，列出的技能包括 **ML, DL, NLP, Computer Vision** 以及各种框架和工具。
- **AI 历史推文令人印象深刻**：一位成员链接到了[这条推文](https://x.com/csteinmetz1/status/1998052491112694178?t=sFIRwM4Jx0wImIMJPFiVPA&s=19)以展示古老的 AI 历史。
   - 这条推文被评价为令人印象深刻。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

burnytech: 该死，可能和我现有的其他东西冲突了
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1447744331958980788)** (15 messages🔥): 

> `Claude coding secureness, Endomorphosis links, China chip manufacturing, Arcprize future, H200 on eBay` 


- **Claude 的代码安全性骤降**：一篇论文发现，**Claude** 的代码辅助中只有 **10.5%** 是安全的，而 **61%** 是功能性的（[Arxiv 链接](https://arxiv.org/abs/2512.03262)）。
   - 评估提到了 **Gemini 2.5 Pro**、**Kimi K2** 以及一个未指明版本的 **Claude Sonnet 4**。
- **Endomorphosis 资源浮现**：一位成员分享了关于 **endomorphosis** 的资源链接，包括一份[关于概率逻辑（Probabilistic Logics）的 PDF](https://static.ias.edu/pitp/archive/2012files/Probabilistic_Logics.pdf)和一段 [YouTube 视频](https://youtu.be/rfHfPxGReCE)。
- **中国芯片雄心不可阻挡？**：几位成员讨论了中国推动芯片制造主导地位的举动，其中一人表示 *目前没有什么能说服中国不继续追求在芯片制造顶端竞争的目标*。
   - 另一位成员指出，最可能的结果是 *他们被鼓励建立自己的芯片制造厂。*
- **Arcprize 暗示未来**：成员们分享了一个来自 **Arcprize** 的链接，在 [fxtwitter](https://fxtwitter.com/arcprize/status/1997743855203148038?t=FP7bdSgZz-EUp9chGKU5aw&s=19) 上暗示了 *未来*。
- **H200 芯片通过中国涌入 eBay？**：一位成员声称，如果你在 **eBay** 上搜索 **H100**，大多数列表无论如何都来自中国。
   - 该成员还暗示，中国正在增加法规，要求公司注册购买 **H200** 并证明本地替代方案不够好。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 条消息): 

jokellum: <@&1116225504563970138>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1447681331424399370)** (2 条消息): 

> `Community Meeting, MMMAudio in Mojo, Shimmer OpenGL experiment, Mojo 1.0 Roadmap, Modular Team Updates` 


- **Modular 社区会议揭晓音频与图形创新**：最新的社区会议展示了 Sam Pluta 在 Mojo 中开发的创意编程音频环境 **MMMAudio**，以及 Lukas Hermann 开发的跨平台 Mojo → OpenGL 实验项目 **Shimmer**，视频已上传至 [YouTube](https://www.youtube.com/watch?v=dsslYZrVPbQ)。
- **Modular 概述通往 Mojo 1.0 理想境界之路**：Modular 团队分享了 **25.7 版本** 的更新，并提供了 **Mojo 1.0 路线图** 的早期预览，更多细节见[其博客文章](https://www.modular.com/blog/the-path-to-mojo-1-0)。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1447681988075978914)** (22 条消息🔥): 

> `Mojo memory management, MMMAudio presentation with Faust, ImplicitlyCopyable List, String CoW, Embedded AI development with Mojo` 


- **Mojo 的重叠生命周期**：Mojo 允许可变引用的生命周期重叠，但会诊断通过两个名称使用相同操作进行的同步访问，例如 `swap(x, num)` 或 `swap(x, y)`。
- **MMMAudio 演示引用了 Faust**：一位成员感谢另一位在 **MMMAudio** 演示中引用了 **Faust**，期待 2026 年的发展，并希望 MMMAudio 能作为一个有用的示例。
- **List 对 `ImplicitlyCopyable` 的条件一致性**：当 `List` 的元素是 `ImplicitlyCopyable` 且具有平凡的 `__copyinit__` 时，`List` 将变为条件性符合 `ImplicitlyCopyable`。
- **String 隐式可复制性与 CoW 升级**：目前的 `String` 实现是写时复制（**CoW**），这减轻了隐式可复制性的开销，并建议 `List` 未来也可能从类似的升级中受益。
- **Jetson Orin Nano：使用 Mojo 进行嵌入式 AI 开发**：对于那些想要使用 Mojo 进行嵌入式 AI 开发的用户，一位成员建议 **Jetson Orin Nano**（配备 **Ampere 级 GPU**）已得到完全支持，且尺寸可能非常合适。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1447721470967943270)** (19 条消息🔥): 

> `Kimi.com website issues, Kimi's search tool, Kimi citation issues` 


- **Kimi 网站故障报告**：一位用户报告了 [kimi.com 网站的问题](https://cdn.discordapp.com/attachments/1371757564005711973/1447735802166644908/image.png?ex=693a0608&is=6938b488&hm=7a4e4282d6c07e40e92989f4ecf2b9358565987e108f9b2cc6bb03be92a420de&)，除了开始新对话外，**无法点击任何内容**。
- **通过清除 Cookie 排除 Kimi 网站故障**：一位成员建议清除 Cookie 并禁用 VPN/广告拦截器以解决 Kimi 的网站问题。
   - 该用户表示 *这并没有解决问题，清除 Cookie 也没有用*。
- **Kimi 使用 Webcrawler 搜索**：一位用户询问 Kimi 的搜索工具使用什么搜索引擎，另一位用户回答说 Kimi *不使用任何搜索引擎*，而是使用其**自有的网络爬虫 (webcrawler)**。
- **报告 Kimi Bug**：一位成员建议针对 Kimi 的引用问题和异常表现提交 Bug 报告。
   - 原用户描述说，当他向 Kimi 提问时，Kimi 会在内心想法中回答问题而不分享给用户，并且经常无法生成引用。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1447701512317833330)** (8 条消息🔥): 

> `Checkpoint restoration issues, Credits issue, Manus 1.5 critical incidents` 


- **用户遇到检查点恢复问题**：一位用户报告了在尝试恢复 **webdev 项目**的检查点（checkpoint）时遇到的严重问题。
   - 他们询问了如何开工单并分享了他们的电子邮件地址。
- **积分问题迅速解决**：一位用户报告说 **Manus 团队**通过退款解决了他们的**积分（credits）问题**。
   - 该用户现在可以直接通过 Manus 支付，而无需通过 Google。
- **Manus 1.5 面临严重事件且邮件未获回复**：一位用户报告了 **12 月 3 日至 9 日** 期间发生的几起严重事件，包括 **7 个受影响的任务**，以及损失了约 **150,000 积分**。
   - 该用户多次向支持团队发送邮件但未获回复，但 AI Bot 在 **12 月 9 日** 提出了 **120,000 积分** 的补偿方案。他们正式要求技术支持和商务团队在 48 小时内给出联合答复、对根本原因进行技术分析以及公平的赔偿。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1448000359422234826)** (4 条消息): 

> `Anthropic's donation, Model Context Protocol, Agentic AI Foundation, LF standards` 


- **Anthropic 捐赠 Model Context Protocol 并成立 Agentic AI Foundation**：Anthropic 正在[捐赠 Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 并成立 Agentic AI Foundation。
   - 一位成员询问了对当前工作的影响，预期会过渡到 LF 的“运作方式”，但另一位成员澄清说，*治理以及该范畴下的一切都不会改变*。
- **关于捐赠后治理变化的澄清**：在 Anthropic 捐赠之后，一位社区成员寻求关于潜在治理转变的澄清。
   - 另一位成员回应并强调，捐赠*不会改变现有的治理结构*。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1447961828217848062)** (3 条消息): 

> `MCP Usage, Private vs Public Ecosystems, Client-ID Metadata Documents (CIDM), MCP Servers, Developer Tools` 


- **私有生态系统中的 MCP 用法**：一位成员询问了 **MCP** 在**私有生态系统**中的用法，并提到了 auth-wg 在通过 **Client-ID Metadata Documents (CIDM)** 进行公共生态系统客户端注册方面的工作。
   - 回复指出，大多数 **MCP** 的使用可能是**私有/内部/企业级**的，特别是考虑到带有公共客户端（例如 **Claude**）的私有 **MCP servers**。
- **专注于与 Developer Tools 集成的 MCP Servers**：有人指出，大多数面向公众的远程 **MCP servers** 都旨在与 **developer tools** 集成。
   - 此外，**developer tools** 被认为是最高级的非定制 **MCP clients**。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1447765581146095617)** (2 条消息): 

> `Opus with Amazon bedrock and aider` 


- **Opus, Bedrock 和 Aider 测试良好**：一位成员询问 **Opus** 是否可以与 **Amazon Bedrock** 和 **Aider** 配合使用。
   - 该成员随后确认一切正常。
- **此处的另一个主题**：此处的另一个第一条摘要
   - 此处的另一个第二条摘要。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1447969325733380209)** (1 条消息): 

> `aider features, aider image support, aider workflow` 


- **Aider 现在可以生成 commit messages**：Aider 现在会自动生成 commit messages，从而改进工作流并[简化 commit 流程](https://example.com/aider-commit-messages)。
   - 用户现在可以使用 `-m` 标志，或者仅使用 `commit` 命令进行提交，使用的是基础的 `gpt-3.5-turbo` 免费层级模型。
- **Aider 将获得 Image 支持**：`aider` 计划支持图像，从而实现更详细且[具上下文相关的代码修改](https://example.com/aider-image-support)。
   - 当要求 `aider` 修改或编辑现有图像时，用户将能够传入 `--image`。
- **Aider 工作流现在可以保存编辑会话**：现在可以保存 aider **编辑会话 (edit sessions)**，允许用户稍后[恢复它们以进行完整的往返操作](https://example.com/aider-session-management)。
   - 此功能增强了协作，并允许用户*保存、共享和恢复*其工作流。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1447711769366630451)** (2 条消息): 

> `AI Agent 0-1 Workshop, AI Engineering Bootcamp, GitHub Social Club NYC` 


- **AI Agent 工作坊启动！**：将举办一个 **AI Agent 0-1 工作坊**，作为 **AI Engineering Bootcamp**（在线）的入门。
   - 该活动将教导参与者从零开始设计并构建一个能够思考、编码、分析数据并为之前的真实客户生成报告的 **AI agent**；请预约 12 月 13 日星期六下午 2 点（东部时间）：[luma.com](https://luma.com/t4jcok99)。
- **GitHub Social Club 在纽约市集结**：在纽约市 SoHo 的 Bibliotheque 将举办 **GitHub Social Club**。
   - 这里*没有演讲，没有推销，只有与社区其他人联系、分享想法和交流故事的空间*，并提供咖啡、饼干、限量版 GitHub 周边、一些休闲游戏，以及与 Copilot、Next、Developer Productivity 和 Startups 背后团队见面的机会。


  

---


---


---