---
companies:
- anthropic
- openai
- microsoft
- perplexity-ai
- huggingface
- groq
- cerebras
- togethercompute
date: '2025-10-16T05:44:39.731046Z'
description: '**Anthropic** 凭借 **Claude** 的新功能 **Skills** 连续占据 AI 新闻头条，成就斐然。这是一种构建专用智能体（Agents）的新颖方式，通过
  Markdown 文件、脚本和元数据来处理创建和读取 PDF、文档及 PPT 等任务。Simon Willison 称其为“比 MCP 意义更重大的突破”，并预言“Skills
  将迎来寒武纪大爆发”。


  与此同时，**Anthropic** 发布了 **Claude 4.5 Haiku**，该模型具备强大的推理和长上下文处理能力，且定价极具竞争力。其他更新还包括
  **OpenAI** 的 ChatGPT 记忆管理改进、**Windows 11 Copilot** 的语音和视觉功能，以及 **HuggingChat Omni**
  实现了在来自 15 个供应商的 115 个开源模型间进行路由调度。这些动态凸显了智能体技能、文档处理、长上下文推理和多模型路由领域的最新进展。'
id: MjAyNS0x
models:
- claude-4.5-haiku
- claude
- chatgpt
- huggingchat-omni
people:
- simonwillison
- alexalbert__
- mustafasuleyman
- yusuf_i_mehdi
- aravsrinivas
title: Claude Agent Skills —— 是美化版的 AGENTS.md，还是 MCP 杀手？
topics:
- agent-skills
- document-processing
- long-context
- reasoning
- multi-model-routing
- memory-management
- voice
- vision
---

**Claude is all you need**

> 2025/10/15-2025/10/16 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord（197 个频道，6365 条消息）。预计节省阅读时间（以 200wpm 计算）：492 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 风格展示。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

对于任何实验室来说，这都是一项罕见的成就，Anthropic 凭借今天发布的 [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) 连续两次登上了 AINews 的头条新闻，这是一种“使用文件和文件夹构建专业 Agent 的新方法”。事实证明，Claude 最近新增的创建和读取 PDF、Docs 以及 PPT 的能力全都是 Skills。

[](https://resend-attachments.s3.amazonaws.com/G0aCs2pSirnjnWA)

- [介绍博客和视频](https://www.anthropic.com/news/skills)
- [工程技术文章](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [HN 讨论](https://news.ycombinator.com/item?id=45607117)
- [Simon Willison 称其为“比 MCP 更重大的进展”](https://simonwillison.net/2025/Oct/16/claude-skills/)

引用 Simon 的话：

> Skills 在概念上极其简单：一个 skill 就是一个 Markdown 文件，告诉模型如何执行某项任务，并可选地附带额外的文档和预编写的脚本，模型可以运行这些脚本来帮助其完成 skill 中描述的任务。
> 
> 
> Claude 在 9 月份随其[新 code interpreter 功能](https://simonwillison.net/2025/Sep/9/claude-code-interpreter/)推出的[新文档创建能力](https://www.anthropic.com/news/create-files)，事实证明完全是使用 skills 实现的。这些功能现在已在 [Anthropic 的 GitHub 仓库](https://github.com/anthropics/skills/tree/main/document-skills)中提供，涵盖了 `.pdf`、`.docx`、`.xlsx` 和 `.pptx` 文件。
> 

以及：

> 我预计我们将看到 Skills 的寒武纪大爆发，相比之下，今年的 MCP 热潮将显得平淡无奇。
> 
> 
> Skills 是包含少量 YAML 元数据和一些可选脚本的 Markdown，这些脚本可以在你设定的任何可执行环境中运行。它们感觉更接近 LLM 的精神——丢进一些文本，让模型自己去搞定。
> 

[](https://resend-attachments.s3.amazonaws.com/M84445mZrQ4drWs)

---

# AI Twitter 综述

**助手平台：ChatGPT Memory, Sora 2, Claude 4.5 Haiku 和 “Skills,” Windows Copilot, Perplexity, HuggingChat Omni**

- **OpenAI 产品更新**：ChatGPT 现在自动管理已保存的记忆（不再有“记忆已满”的情况），支持搜索/排序和重新优先级排序；正在全球范围内的 Web 端向 Plus/Pro 用户推出 [@OpenAI](https://twitter.com/OpenAI/status/1978608684088643709)。Sora 2 为 Pro 用户在 Web 端增加了 Storyboards（故事板），并延长了视频长度（所有用户在 App/Web 端最高可达 15 秒；Pro 用户在 Web 端最高可达 25 秒）[@OpenAI](https://twitter.com/OpenAI/status/1978661828419822066), [@billpeeb](https://twitter.com/billpeeb/status/1978662020947087869)。
- **Anthropic 的性价比层级和 Agent 升级**：Claude 4.5 Haiku 发布，价格为每 100 万输入/输出 Token $1/$5；在推理模式下，它在 Artificial Analysis 指数上得分为 55，比 Sonnet 4.5 便宜 3 倍，根据 [Artificial Analysis](https://twitter.com/ArtificialAnlys/status/1978661658290790612) 的数据，其在长上下文/编程方面表现强劲。社区排名将其列为 LMArena 总榜第 22 位，在编程和长查询方面具有优势 [@arena](https://twitter.com/arena/status/1978966289248063885)。Anthropic 还推出了 **Skills**——在运行时加载的打包指令文件夹/脚本/资源——可在 [claude.ai](http://claude.ai/)、Claude Code 和 API 中使用，并附带文档和工程笔记 [@alexalbert__](https://twitter.com/alexalbert__/status/1978877498411880550), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978896757489594404)。企业级功能现在包括 Microsoft 365（SharePoint、OneDrive、Outlook、Teams）集成和企业搜索 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978864348236779675)。
- **Windows 和 Perplexity 发布 UX 原语**：Windows 11 增加了 Copilot Voice（“Hey Copilot”）、跨桌面/应用/文档的 Vision 功能，以及即将推出的针对本地文件的 Copilot Actions [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1978808627008847997), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1978808604200259785)。Perplexity 在 iOS/Web 端发布了内置的语言学习体验和新的财经功能（内幕交易追踪器）[@perplexity_ai](https://twitter.com/perplexity_ai/status/1978859991152165125), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978941079182545338)。
- **跨多个 OSS 模型的路由**：HuggingChat v2 推出了 “Omni”，这是一种基于策略的自动模型选择功能，涵盖 115 个 OSS 模型和 15 个提供商（如 Groq、Cerebras、Together）。Omni 可以在一个会话中在编程模型和写作模型之间路由任务；100% 开源 [@victormustar](https://twitter.com/victormustar/status/1978817795312808065), [@reach_vb](https://twitter.com/reach_vb/status/1978854312647307426)。
- 其他值得关注的：网站正在推广“使用 ChatGPT 登录”；使用 OpenAI 模型的成本可以通过这种方式转移给终端用户 [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1978835849379725350)。Google 的 AI Studio 现在拥有一个统一的 Playground，支持 Chat/GenMedia/Live 模型 [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1978861583078961263)。

**系统与基础设施：TPU 上的 vLLM、Google 的 TPU 推进以及本地开发机的现实**

- **TPU 推理栈落地**：vLLM 展示了与 Google 共同开发的重新构思的 TPU 后端，通过单一的 JAX-to-XLA lowering 路径统一了 PyTorch 和 JAX，具有默认 SPMD、Ragged Paged Attention v3，吞吐量是其 2 月份原型的 2 到 5 倍；支持 Trillium (v6e) 和 v5e [@vllm_project](https://twitter.com/vllm_project/status/1978855648176853100), [@_philschmid](https://twitter.com/_philschmid/status/1978889178067743210)。
- **Google 扩大 TPU 访问权限**：TPU 现在向外部客户销售，直接与 NVIDIA 竞争 [@zephyr_z9](https://twitter.com/zephyr_z9/status/1978835094216343820)。Baseten 报告称，通过早期采用支持 KV-cache 感知路由的 NVIDIA Dynamo，延迟降低了 50%，吞吐量提升了 60% 以上 [@basetenco](https://twitter.com/basetenco/status/1978883986924634551)。
- **本地 vs 云端**：来自真实用户的实践笔记——Mac Mini M4 Pro 非常适合本地 LLM 推理，但不适合持续的微调工作负载；由于 MPS 的不稳定性，CUDA 对于 PyTorch 训练仍然至关重要，许多人选择云端 GPU，而不是噪音大、发热高的多 GPU 桌面机 [@rasbt](https://twitter.com/rasbt/status/1978608882156269755)。PyTorch 的 Soumith 指出 Apple 对 MPS 后端的投入断断续续，Meta 工程师承担了大部分工作；并警告不要指望在训练方面能与 NVIDIA 平起平坐 [@soumithchintala](https://twitter.com/soumithchintala/status/1978848796953161754)。
- **管道与晶圆厂**：台积电 (TSMC) N2 量产计划在年底前开始 [@TechPowerUp](https://twitter.com/TechPowerUp/status/1978737339171017215)。Google 发布了 torchax，探索 PyTorch→JAX 的 lowering [@gallabytes](https://twitter.com/gallabytes/status/1978860154008240142)。

**推理、RL、长上下文和评估 (evals)**

- **预见未来的 RL 缩放定律 (scaling laws)**：Meta 及其合作者发布了《LLM 强化学习计算缩放的艺术》(The Art of Scaling Reinforcement Learning Compute for LLMs)，这是一项耗时 40 万 GPU 小时的系统研究，提出了 ScaleRL（具有 8 步离策性的 PipelineRL）、CISPO 损失、FP32 logits 以及基于中断的长度控制。关键结果：目标计算量下的性能可以通过一半计算量的运行结果来预测；许多微小的决策会实质性地影响稳定性/上限 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978793969384624226), [@_lewtun](https://twitter.com/_lewtun/status/1978826407376458125), [@omarsar0](https://twitter.com/omarsar0/status/1978865039529689257)。
- **推理时递归优于上下文腐烂 (context rot)**：递归语言模型 (RLMs) 表明，在无限上下文上进行递归自调用/工具调用，在长上下文任务上的表现可以超越标准的 GPT-5，且即使在 10M+ token 下仍能保持成本效益。可尝试的极简要点：[@a1zhang](https://twitter.com/a1zhang/status/1978948676287340753)，评论 [@dbreunig](https://twitter.com/dbreunig/status/1978873161841066464)。
- **计算高效的路由与 RL 理念**：Dr.LLM 动态地跳过/重复 Transformer 层以减少计算量并提高准确性，其逐块路由通过离线 MCTS 和 focal loss 进行训练——推理时采用贪婪路由 [@omarsar0](https://twitter.com/omarsar0/status/1978829550709866766)。“串联训练”(Tandem training) 在 RL 期间间歇性地从冻结的弱模型中采样 token，以使解决方案对较弱的协作模型保持可理解性 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978794773747314765)。
- **小模型，大难题**：Tiny Recursion Model (TRM, ~7M) 在 ARC-AGI-1 上达到了 40% 的准确率，成本约为每任务 1.76 美元（已发布权重和方案），这进一步证明了专门的推理程序至关重要 [@arcprize](https://twitter.com/arcprize/status/1978872651180577060)。AssistantBench：o3 目前在普林斯顿的任务中超越了 GPT-5-med [@OfirPress](https://twitter.com/OfirPress/status/1978925179876020247)。
- **评估 (Evals) 优于直觉 (vibes)**：吴恩达 (Andrew Ng) 为 Agent 系统中的评估和错误分析提出了一个务实的框架——原型设计、检查输出、定义自定义指标/评判器、迭代评估，最后再进行优化 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1978867684537438628)。

**编程 Agent 与检索：快速上下文优于长上下文**

- **Cognition 通过 SWE-grep 实现的快速上下文**：一个新的模型系列（>2,800 TPS），用于快速、多轮的 Agent 搜索，定位“正确文件”的速度比 Claude 4.5 Haiku 快约 20 倍，同时在 Cognition 的 CodeSearch 评估中与前沿模型并驾齐驱；目前正通过 Fast Context 子 Agent 和 Playground 在 Windsurf 中推出 [@cognition](https://twitter.com/cognition/status/1978867021669413252), [@silasalberti](https://twitter.com/silasalberti/status/1978871477605929229), [@swyx](https://twitter.com/swyx/status/1978874342743343254)。由 Cerebras 支持的部署在实践中进一步降低了延迟 [@draecomino](https://twitter.com/draecomino/status/1978898418354561225)。
- **Agent 原语与开源 (OSS) 工具链**：Cline CLI（预览版）公开了一个可脚本化的、开放的“原语 Agent 循环”，IDE Cline 可以对其进行编排——专为子 Agent 和可组合工作流设计 [@cline](https://twitter.com/cline/status/1978874789193486749)。“Open Agent Builder”是一个 n8n 风格的开源画布，连接了 Firecrawl、LLM、逻辑节点和 MCP，用于可部署为 API 的工作流 [@firecrawl_dev](https://twitter.com/firecrawl_dev/status/1978878728827478289), [@CalebPeffer](https://twitter.com/CalebPeffer/status/1978852506286571737)。Surfer 2 在 WebVoyager/AndroidWorld/WebArena/OSWorld 等跨平台计算机使用 (computer-use) 任务中报告了 SOTA [@hcompany_ai](https://twitter.com/hcompany_ai/status/1978935436111229098)。
- **用于代码的 Anthropic Skills**：开发者报告称，通过将特定领域的脚本/资源作为运行时 Skills 分层接入，Claude Code 变得更敏锐、更精准——这是一种补充 MCP/工具的结构化上下文工程 [@omarsar0](https://twitter.com/omarsar0/status/1978919087137804567)，文档见 [@alexalbert__](https://twitter.com/alexalbert__/status/1978877611159003542)。

**视觉与多模态：实时世界、OCR/VLM 以及图像/视频编辑**

- **实时世界模型**：World Labs 的 RTFM 是一款实时、持久、3D 一致的自回归扩散 Transformer，在大型视频数据集上训练而成——在 H100 速度下实现交互式流式传输，并提供现场演示 [@theworldlabs](https://twitter.com/theworldlabs/status/1978839171058815380), [@drfeifei](https://twitter.com/drfeifei/status/1978840835341914164), [@jcjohnss](https://twitter.com/jcjohnss/status/1978842517605843391)。
- **视频与编辑流水线**：Google Veo 3.1 已在 LTX Studio 和 Synthesia 上线（具有更高的写实度、音频支持和完整的 Keyframe 支持）[@LTXStudio](https://twitter.com/LTXStudio/status/1978827563926716704), [@synthesiaIO](https://twitter.com/synthesiaIO/status/1978836856419635561)。Sourceful 的 Riverflow 1 在 Artificial Analysis 的图像编辑“全部”列表中首次亮相即排名第一，它结合了 VLM 与开放扩散模型；价格为 66 美元/1k 张图像（“mini”版为 50 美元/1k 张）[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1978891167795417092)。
- **文档 AI 与定位 VLM**：PaddleOCR-VL (0.9B) 针对工业级文档智能（文本、表格、公式、图表、手写），由 NaViT + ERNIE 驱动；支持 109 种语言 [@PaddlePaddle](https://twitter.com/PaddlePaddle/status/1978809999263781290)。字节跳动的 Sa2VA 结合了 SAM2 和 LLaVA，用于图像/视频的密集定位理解 [@HuggingPapers](https://twitter.com/HuggingPapers/status/1978745567258829153)。阿里巴巴的 Qwen3-VL-Flash 带来了 256K 上下文、更好的空间推理/3D 定位、OCR 以及更严格的安全限制 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978841775411503304)。Google “Nano Banana” 图像编辑功能登陆 Lens/AI 模式（最初在美国/印度推出）[@Google](https://twitter.com/Google/status/1978857184566837735)。

**开源模型与微型模型的胜利：nanochat、MobileLLM-Pro、ColBERT minis 以及安全组件**

- **野外微型模型**：Karpathy 的 nanochat d32（1k 美元从零训练）将 CORE 提升至 0.31（> GPT-2 ~0.26），GSM8K 提升至 ~20%，并发布了完整报告/脚本；社区正在将其集成到 Transformers 和 vLLM 中 [@karpathy](https://twitter.com/karpathy/status/1978615547945521655), [@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1978832914952401081)。
- **端侧级别 LM**：Meta 的 MobileLLM-Pro (1B) 发布了基础版 + 指令版检查点（包含量化变体），旨在实现高质量、高效的端侧推理；在少于 2T 的开放 Token 上预训练，在推理/知识/长上下文检索方面优于 Gemma 3 1B 和 Llama 3.2 1B 分别 5.7% 和 7.9% [@_akhaliq](https://twitter.com/_akhaliq/status/1978916251456925757)。
- **超紧凑嵌入/检索**：[mixedbread.ai](http://mixedbread.ai/) 的 mxbai-colbert-edge-v0 (17M, 32M) 提供了可复现的 ColBERT 训练；17M 版本在 LongEmbed 1B 以下模型中排名第一，采用 Apache 2.0 许可 [@mixedbreadai](https://twitter.com/mixedbreadai/status/1978853869557055492), [@bclavie](https://twitter.com/bclavie/status/1978854449062793335)。Nanonets 发布了新的 OCR2-3B 和 1.5B-exp 模型 (Apache-2.0)，可处理表单、水印、图表甚至流程图 [@mervenoyann](https://twitter.com/mervenoyann/status/1978837720353927415)。
- **安全工具**：阿里巴巴开源了 Qwen3Guard 的组件，包括 Qwen3-4B-SafeRL（WildJailbreak 分数从 64.7 跃升至 98.1，且不损害通用性能）和用于分类中间“思考”过程及逐 Token 审核的 Qwen3GuardTest [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978732145297576081)。

**热门推文（按互动量排序）**

- “90 年代的电视 vs 2025 年”的 UX 吐槽引发广泛共鸣 [@karpathy](https://twitter.com/karpathy/status/1978653908663726585)。
- ChatGPT “Memory”（记忆）自动管理功能推出 [@OpenAI](https://twitter.com/OpenAI/status/1978608684088643709)。
- Sora 2 更新：Storyboards（分镜脚本）+ 更长的视频 [@OpenAI](https://twitter.com/OpenAI/status/1978661828419822066)。
- DeepMind 的“视频版图灵测试”预告以及与 CFS 的聚变合作 [@demishassabis](https://twitter.com/demishassabis/status/1978644313824534954), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1978808994811588666)。
- Perplexity 推出语言学习和内幕交易追踪功能 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978865088296542387), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978941079182545338)。
- Anthropic 的 Agent Skills（智能体技能）发布公告 [@alexalbert__](https://twitter.com/alexalbert__/status/1978877498411880550)。
- vLLM 的 TPU 后端统一了 PyTorch + JAX [@vllm_project](https://twitter.com/vllm_project/status/1978855648176853100)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

待完成 (TO BE COMPLETED)

## 技术性较低的 AI 子版块摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

待完成 (TO BE COMPLETED)

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
>

**Theme 1. Model Mania: New Releases and Breakthroughs Shake the Landscape**

- [**Gemini 3 Pro 仅凭单一提示词编写出可运行的游戏**](https://discord.com/channels/1340554757349179412/1340554757827461211/1428095522148847720)：LMArena Discord 频道的用户们对 **Gemini 3 Pro** 的编程实力赞不绝口，因为它仅通过一段 1000 多行的提示词就生成了一个完全可运行的 [HTML 版 Geometry Dash 克隆体](https://link.to.clone/)，而这一任务 **Gemini 2.5 Pro** 此前未能完成。这引发了人们的猜测，即 Gemini 3 Pro 在编程任务上可能会比预期的 **GPT-5 Pro** 强出 **5-10%**。
- [**OpenAI 的 Sora 2 获得分镜脚本升级**](https://discord.com/channels/974519864045756446/977259063052234752/1428171257962041466)：**Sora 2** 现在允许 **Pro** 用户创建 **Storyboards**（分镜脚本）并生成长达 **25 秒**的视频，而普通用户的上限为 **15 秒**。**Sora 2 Pro** 的排名也一路飙升，在 [LMArena 文本转视频排行榜](https://lmarena.ai/leaderboard/text-to-video)上与 **Veo 3** 并列 **#1**。
- [**Anthropic 与 Cognition 发布新模型**](https://discord.com/channels/1027685395649015980/1027688115592237117/1428151932668743721)：**Claude Haiku 4.5** 已在 Windsurf 上线，消耗 **1 倍额度**，据报道其编程性能可媲美 **Sonnet 4**，且成本仅为其三分之一，速度翻倍。与此同时，Cognition 新的 **SWE-grep** 模型正在 Windsurf 逐步推出，承诺通过向编程 Agent 快速定位相关文件，使 [Agent 式搜索提速 20 倍](https://cognition.ai/blog/swe-grep)。

**Theme 2. Platform Pains and Subscription Snafus**

- [**订阅故障导致 Pro 用户降级为免费版**](https://discord.com/channels/1074847526655643750/1074847527708393565/1428095086360789133)：**Cursor Pro+** 和 **Perplexity Pro** 的用户（特别是使用 Airtel 套餐的用户）反映，他们的付费计划意外降级为免费版本，导致无法使用高级功能。一些 Perplexity 用户还反映在免费试用期间被错误扣费，目前支持团队正在调查中。
- [**集成难题困扰 OpenRouter 与 Groq**](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794)：Cursor 用户正面临故障频发的 **OpenRouter** 集成问题，尽管配置正确，服务仍无法处理任何请求。在 DSPy Discord 频道中，一名用户反映即使在 **OpenRouter** 中将 **Groq** 设置为唯一供应商，系统仍会默认调用其他模型，正如其[设置截图](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794)所示。
- [**Tinygrad 的破坏性变更令开发者受挫**](https://discord.com/channels/1068976834382925865/1068976834928193609/1428207732363886743)：tinygrad Discord 频道中的一位 openpilot 开发者对频繁的破坏性变更表示沮丧，这些变更导致了难以理解的错误，并需要繁琐的提交二分法（commit bisection）来修复。该用户指出，针对 **845** 处理器的不稳定 **IMAGE hacks** 以及不断变化的环节变量带来了巨大的维护挑战。

**Theme 3. The Tooling Tribune: Frameworks and Libraries Evolve**

- [**DSPy 辩论 Agentic Search，同时 Mojo 关注游戏开发**](https://discord.com/channels/1161519468141355160/1161519469319946286/1428095494911164486)：DSPy 社区就 "Agentic Search" 的定义展开了辩论，一些人称其为营销术语，而另一些人则试图复制 **Claude Code** 的 *ripgrep* 实现，详见这篇 [Agentic Search for Dummies 博客文章](https://benanderson.work/blog/agentic-search-for-dummies/)。与此同时，Modular 社区正在探索 Mojo 在游戏开发方面的潜力，并指向了 [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3) 等项目，Modular 还开源了整个 [MAX Python API](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379)。
- [**PyTorch 2.9 为多 GPU 协同解锁 Symmetric Memory**](https://discord.com/channels/1189498204333543425/1398843708488552570/1428169904686633081)：一篇 [PyTorch 2.9 博客文章](https://pytorch.org/blog/pytorch-2-9/) 宣布了 **PyTorch Symmetric Memory**，这是一项新功能，简化了在 NVLinks 和 RDMA 网络上编写多 GPU kernel 的编程。这实现了强大的功能，如 **in-kernel communication**、超低延迟远程访问以及为性能工程师提供的自定义通信模式。
- [**HuggingFace 通过 Custom Blocks 增强 Diffusers**](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123)：HuggingFace 团队宣布 **Modular Diffusers** 现在支持 **custom blocks**，允许开发者实现能够无缝集成到核心库中的新功能。目前已提供 [示例 block 集合](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401) 和 [文档](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)。

**Theme 4. Hardware Horizons and High-Performance Hacks**

- [**Intel 的新芯片承诺海量内存带宽**](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144)：工程师们对 Intel 即将推出的硬件议论纷纷，包括仅用于推理的 **Crescent Island** GPU，它拥有 **1.5TB/s** 的带宽和 **160GB** 的内存。此外，**Rubin CPX** 架构也备受关注，它在支持多种数字格式方面表现出色，有可能通过 **1280-bit** 总线简化 **software-level block floats**。
- [**本地 LLM 撞上 GPU 内存墙**](https://discord.com/channels/1110598183144399058/1110598183144399061/1428104621712474193)：LM Studio 的讨论强调了本地 LLM 的一个关键瓶颈：**GPU 内存限制**。用户注意到，无论宣传的 context windows 有多大，大多数模型在超过 **20-40k tokens** 后都会失去连贯性。这引发了硬件辩论，包括 **128GB** 的 **DDR4 3600 RAM** 相比 **64GB** 的性能优势，一位用户指出 *如果你有 DDR5-8000，它的速度会快 4 倍*。
- [**DeepSeek 工程师在受限的 H20 上大显身手**](https://discord.com/channels/1189498204333543425/1189498205101109300/1428132127710384201)：GPU MODE Discord 中流传着一个都市传说，称 **DeepSeek** 工程师巧妙地利用底层 **PTX/SASS** 指令克服了受限硬件上的内存带宽限制。这项创新使他们能够在尽管美国限制将中国可用的 GPU 从 **H100** 降级为 **H20** 的情况下，依然构建出强大的模型。

**Theme 5. AI Ethics and Culture Clashes**

- [**AI 奶奶走红，吸粉 200 万**](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)：一个完全由 AI 生成的网红 **grannyspills**，以“心直口快、拜金的奶奶”形象提供毒舌约会建议，其 Instagram 粉丝量已接近 **200 万**，详见[这篇 X 帖子](https://x.com/venturetwins/status/1978852719335985309)。这引发了关于观众是否在意其虚拟属性的辩论，一些人称赞这种讽刺艺术，而另一些人则担心 AI 对文化的影响。
- [**收到投诉后，OpenAI 在 Sora 中屏蔽 MLK**](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)：应 King Estate 的要求以及公众对 AI 生成视频片段的投诉，OpenAI 通过一篇 [X 帖子](https://x.com/OpenAINewsroom/status/1979005850166648933)宣布，将禁止 **Sora** 描绘 **Dr. Martin Luther King Jr.**。此举遭到用户的强烈批评，被认为是一种“滑坡效应”式的让步，将公众人物私有化，并可能导致无休止的下架要求。
- [**传闻 GPT-5 将取消拒绝机制，引发争议**](https://discord.com/channels/974519864045756446/998381918976479273/1428095910038077502)：OpenAI 社区的传闻暗示 **GPT-5** 可能会采用类似于 **GPT-4o** 的“较少拒绝”人设，这在用户中引发了分歧。虽然有些人欢迎更顺从的模型，但另一些人担心这可能会损害模型的伦理基础，导致它无论道德与否都会同意任何要求。


---

# Discord: 高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Pro 编写 Geometry Dash 代码**：一位用户提示 **Gemini 3 Pro** 生成一个完全可玩的 [HTML 版 Geometry Dash 克隆游戏](https://link.to.clone)，包含音乐和物理引擎，使用的提示词为：*生成一个 Geometry Dash 克隆版的完整 HTML 文件，但要求是 2000 年代的风格，为关卡添加音乐（使用 JS 制作反映关卡变化的音乐），物理引擎与 Geometry Dash 游戏相同，我们需要一个完整的可玩游戏。全部集成在一个 HTML 文件中，代码不少于 1000 行*。
   - 其他模型如 **Gemini 2.5 Pro** 未能使用相同提示词生成可运行的游戏，这引发了对 **Gemini 3** 的期待。
- **Gemini 3 Pro 基准测试备受期待**：成员们讨论了 **Gemini 3 Pro** 在编程方面可能超越 **GPT-5 Pro**，预计会有 **5-10% 的性能提升**。
   - 传闻该模型正在 AI Studio 上进行 A/B 测试，促使用户寻找访问方法，但也有人担心 **Google** 可能会施加 Token 限制以节省服务器资源。
- **LMArena 机器人出现故障**：用户报告了 **Video Arena 机器人** 的问题，例如“生成响应时出错”以及视频生成能力的限制。
   - 一名管理员确认团队已获悉这些问题并正在积极修复，建议用户参考 bugs 频道进行故障排除。
- **Sora 2 登顶**：**Sora 2 Pro** 登上了 [Text-to-Video 排行榜](https://lmarena.ai/leaderboard/text-to-video)，目前与 **Veo 3** 和 **Veo 3 Fast** 并列 **第一**，而 **Sora 2** 位列 **第三**。
   - Pro 账户可以使用更长的视频长度（长达 25 秒）且无水印，这引发了关于视频质量和模型偏好的讨论。
- **社区请求 PDF 上传和消息编辑功能**：LMArena 社区对 **PDF 文件上传**、**消息编辑**、**刷新响应**和**删除消息**等新功能表现出浓厚兴趣。
   - 管理员已确认这些功能已列入未来的开发计划。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Airtel 套餐提前到期！**：几位 **Airtel Perplexity Pro** 用户报告他们的订阅突然结束，支持团队正在 *investigating*（调查），但尚未回复部分用户。
   - 此外，一些用户报告在 **Perplexity Pro** 的免费试用期内被计费。
- **Comet Browser 推荐奖励 Bug 困扰用户**：一些用户报告通过 **Comet Browser** 推荐每位用户赚取了 **$5-$15 USD**，而其他用户则在领取奖励时遇到困难。
   - 工作人员已意识到 [该推荐奖励 Bug](https://discord.com/channels/1047197230748151888/1047649527299055688/1428408489593479258) 并正在调查潜在的推荐计划滥用行为；一名用户因分享 **Perplexity Pro** 代码被禁言。
- **Comet Browser “漏洞”已解决？**：用户讨论了一个 *Comet Jacking* 漏洞，但这似乎是一个已解决的 [Prompt Injection 问题](https://discord.com/channels/1047197230748151888/1047649527299055688/1428339356716496957)。
   - 修改内部设置和访问其他用户的数据违反了服务条款，可能导致 **Perplexity** 账号被封禁。
- **Perplexity 中的图像生成略显不稳定**：用户报告了图像生成的问题，包括错误的帧比例和幻觉信息，特别是在数学问题中。
   - 一名用户建议使用 **Gemini Pro** 以获得更好的准确性，因为 **Perplexity** 上的几位用户仍在等待支持团队的回复。
- **Sonar Deep Research 模型超时**：一名用户报告了 **Perplexity Sonar Deep Research Model** 的 **timeout issue**（超时问题），并在 [社区论坛](https://community.perplexity.ai/t/perplexity-sonar-deep-research-model-timeout-issue-seeking-solution/2094) 发帖但未获回复。
   - 另一名用户指出他们的 **Spaces** 账号无法创建 **new chats**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 获得记忆功能升级**：**ChatGPT** 现在具备自动记忆管理功能，允许用户按时间先后顺序对记忆进行排序和搜索。
   - 该更新正向 **Plus** 和 **Pro** 用户推出，解决了之前的记忆容量限制问题。
- **Sora 新增 Storyboards，延长视频长度**：[Sora](https://video.twimg.com/amplify_video/1978653248572567552/vid/avc1/704x1280/lIHMEtPzOCUTOkfm.mp4) 现在为 **Pro** 用户提供 **Storyboards**（分镜脚本），增强了视频创作能力。
   - 普通用户可以生成最长 **15 seconds** 的视频，而 **Pro** 用户可以将视频延长至 **25 seconds**。
- **Transformer 音乐模型遭遇训练难题**：一名成员报告在训练音乐生成模型时遇到 **NaN loss** 问题，尽管尝试了调整 **learning rates** 并对数据进行预归一化（使用 **300-hour** 的钢琴音乐数据集）。
   - 探索了 Tokenization 方法（**REMI**）和不同的模型大小（**3M to 25M parameters**），问题通常在第 **120-150** 步左右出现。
- **GPT-5 将采用无过滤人格？**：传闻暗示 **GPT-5** 可能会采用类似 **GPT-4o** 的“更少拒绝”策略，引发了社区成员的辩论。
   - 一些人欢迎这种改变，而另一些人则担心这可能会损害模型的伦理基础。
- **AI Safety 竞赛提供奖金**：一项 AI Safety 竞赛鼓励创作创意内容（故事、漫画、视频），奖金为 **$10,000**，详情见 [keepthefuturehuman.ai/contest/](https://keepthefuturehuman.ai/contest/)。
   - 成功提交作品的推荐人可以获得 **$30 Amazon gift cards**，如 [视频概述](https://siliconversations.com/) 中所述。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Bot 对抗垃圾信息发送者**：成员们探索了使用 **Unsloth bot** 来对抗垃圾信息发送者，通过在 **madpinger's Discord bot dataset** 或 **Reddit bot dataset** 等数据集上对其进行训练。
   - 讨论中开玩笑地提到了 **Ultron** 的诞生，而其他人则建议现有的用于垃圾信息检测的 Discord 机器人已经非常成熟。
- **拼写错误干扰训练**：一名用户在 [Unsloth Windows 安装文档](https://docs.unsloth.ai/get-started/install-and-update/windows-installation)中发现并报告了一个拼写错误（**losedowsinstall** 而不是 **Windows install**）。
   - 在用户建议由于支持更好而推荐使用 **WSL** 而非原生 Windows 安装后，该拼写错误被迅速修复。
- **DGX SPARK 引发辩论**：Unsloth 通过一条 [推文](https://x.com/UnslothAI/status/1978456629613084926) 强调了他们对 **DGX SPARK** 的支持，引发了关于其在 **nvfp4 training** 性能方面的对话。
   - 观点不一，有些人认为它对于推理（inference）来说已经足够，而另一些人则认为由于带宽限制，它可能仅属于“爱好者”级别。
- **Svelte 超越 React... 也许吧**：在苦苦挣扎了三周使用 **Svelte** 修改 **Open-WebUI** 后，一名成员在观看了一个 Fireship 视频后宣称，它“只是步骤更多的 **React**，但没那么烦人”。
   - 该成员将最初的困惑描述为“特性而非 Bug”，并最终认为 **Svelte** 实际上非常酷。
- **国际象棋 LLM 做出违规移动**：一名成员建议，[AI vs AI 国际象棋平台](https://github.com/Laszlobeer/chess-llm-vs-llm) 可以通过实现**尝试做出合法移动的多轮尝试**来改进，而不是在失败后使用随机的合法移动。
   - 该成员理论上认为这一改变可能会提高国际象棋移动的性能和整体质量。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 计划混乱导致用户降级为免费版**：几位用户报告他们的 **Cursor Pro+ plans** 降级回了**免费版本**，影响了依赖自定义模型的特性（如 **Agent Window**），并在设置中出现错误。
   - 成员们推测可能是由于维护、价格变动或 **Cheetah** 被撤下；一名成员确认了一个类似的故障，即 Cursor 的 Agent 和 Edit 停止了正确计费。
- **Cheetah 消失，Haiku 登场**：用户注意到 **Cheetah** 消失了，并确认新的 **Claude-4.5-Haiku** 已启用。
   - 一位成员在比较成本时指出：“Haiku 虽然比 Cheetah 便宜，但 Haiku 比 Claude 4.5 Sonnet 便宜吗？Cheetah 是 1.25 / 10，Haiku 看起来是 1 / 5”。
- **OpenRouter 集成问题频发**：成员们报告了将 **OpenRouter** 与 Cursor 集成时的问题，指出 Cursor 没有向 OpenRouter 发送请求。
   - 建议的解决方案包括禁用其他模型并移除前缀以解决集成问题。
- **Tokenizer 问题引发 Bug 讨论**：一名成员在 background-agents 频道报告了 **Tokenizer layer** 无法完全完成任务的问题，怀疑是 Bug 还是目标定义的问题。
   - 他们提到，以前指定一个规范（spec）然后命令“完成 Tokenizer 层并确保其符合规范并通过所有测试”是有效的。
- **仪表盘工具追踪 Cursor 的 Token 成本**：一名成员展示了他们用于追踪产品中 Token 成本的 [仪表盘](https://token-watch.vercel.app/)；另一名成员也展示了他们构建的仪表盘。
   - 成员们赞扬了仪表盘的使用，一些人还为该仪表盘贡献了代码和调试建议。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LLM 对 YouTube 洞察发表见解**：成员们发现，使用 LLM 分析 YouTube 转录文本时，最有价值的是让 LLM 对转录内容*发表见解*以提取关键信息。
   - 这超越了简单的摘要，能够实现对视频内容的更深层次理解。
- **上下文限制阻碍矛盾检测**：成员们发现，由于上下文限制（特别是在本地环境中），对 **8-10 份 PDF 文档**进行矛盾检测需要类似 Agent 的设置。
   - 尽管排列组合对和摘要策略有所帮助，但即使是具有 *1M Token 上下文窗口*的模型在规模化处理时也可能难以保证准确性。
- **本地 LLM 限制凸显**：用户讨论了本地 LLM 经常受到 **GPU 显存限制**的瓶颈影响，并指出宣传的上下文长度并不保证能被有效利用。
   - 大多数本地模型在超过 **20-40k Token** 后开始失去上下文连贯性，这强调了上下文长度与性能之间的平衡。
- **上下文工程催化内容理解**：使用 LLM 进行有效的内容分析（特别是矛盾检测）需要精细的 Prompt Engineering 和上下文工程。
   - 建议采用带有示例和调优提示的结构化、迭代方法，而不是直接倾倒大量文本，以避免结果稀释和处理缓慢。
- **硬件配置中的 RAM 表现**：LLM 在 **128GB** **DDR4 3600** RAM 与 **64GB** 上的性能表现表明，足够的内存至关重要，且内存带宽会极大影响性能。
   - 一位用户指出，*如果是 DDR5-8000，速度会快 4 倍*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **访问令牌引发账号焦虑**：一位用户在意外创建并删除一个读取令牌后，遇到了无法创建新访问令牌的问题，导致 UI 在权限设置中显示 **“未找到结果”**，甚至考虑[注销账号](https://tenor.com/view/rustic-relic-hunter-snap-out-of-it-are-you-outta-your-mind-gif-9081070089877950090)。
   - 其他用户建议清除浏览器缓存或使用无痕模式作为潜在修复方案，并建议联系 HF 支持团队，发送邮件至 *billing@huggingface.co* 或 *website@huggingface.co*。
- **自定义模块增强 Diffusers**：**Custom blocks** 可以实现库中尚未提供但能无缝集成的功能，如该[集合](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401)所示。
   - **Modular Diffusers** 允许通过自定义模块扩展功能，并与现有的 **Diffusers Library** 无缝集成，详见[文档](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)。
- **GitHub 限制大文件上传**：一位用户询问如何向 **GitHub** 上传大于 **25MB** 的文件，因为他们在向 **Google Drive** 上传时耗时过长。另一位成员建议根据[这些说明](https://cdn.discordapp.com/attachments/879548962464493619/1428283674251624521/github_over_25mb.md?ex=68f29914&is=68f14794&hm=4f2a32cd8bd636b8aa719aacc55bee90d5ee9c868e58e7430a2e7a5d7d996c6f&)使用 **GitHub CLI**。
   - 该用户随后遇到了 **git LFS** 错误，但不理解其中的 “ref” 错误。
- **FRAI 框架迎来首批反馈**：一位用户分享了一个名为 **FRAI** 的开发者优先的 **Responsible AI** 框架，其 CLI 版本已在 [GitHub](https://github.com/sebuzdugan/frai) 上发布。
   - 作者正在寻求 **Star 和反馈**以改进该框架，并提供了指向[其 YouTube 频道](https://m.youtube.com/@sebuzdugan)的链接，其中包含与 FRAI 相关的视频。
- **影响函数引发探究**：一位成员对**影响函数（influence functions）**表现出浓厚兴趣，并寻求与有相关经验的人建立联系，引用了论文 [(1)](https://arxiv.org/abs/2308.03296) 和 [(2)](https://arxiv.org/abs/2411.12580v1) 作为理解和应用该技术的资源。
   - 该成员正在其工作组内探索可能受益于此方法论的**新研究问题**，并愿意与该领域的专业人士合作。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini 2.5 Flash 击败 Haiku 和 Deepseek**：一位成员声称 **Gemini 2.5 Flash** 的表现优于 **Haiku**、**Deepseek R1** 和 **Kimi K2**，但另一位用户发现 **Flash** 相当笨拙，特别是在 Python 编程方面，并[发表评论](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144)称，即使是 **Gemini 2.5 Pro** 在被明确告知不要添加注释时，仍会在代码中添加注释。
   - 他们认为编程仍然是 Anthropic 的强项。
- **Haiku 针对代码优化，但在其他方面表现不佳？**：成员们讨论了 **Haiku 4.5**，指出它似乎针对编码进行了优化，但在大多数其他任务中可能会崩溃，一位成员建议它*可能值得用于编码，但不适用于其他一切*。
   - 还有观点认为 **Gemini** 很聪明，但没有针对 Agent 任务进行良好的训练，而 **Haiku** 的注意力跨度较低，可能会影响其编码能力。
- **Tensor Logic：连接逻辑与智能的桥梁？**：重点介绍了一篇关于 **Tensor Logic** 的[论文](https://arxiv.org/abs/2405.08793)，认为它可以通过将逻辑推理转化为纯张量代数，成为逻辑与智能之间的桥梁。
   - 这种方法将布尔推理、概率推理和谓词逻辑嵌入到一个单一的可微框架中，可能使模型能够*不仅预测真理*，而且*证明真理*。
- **英特尔 Crescent Island 拥有 1.5TB/s 带宽**：一位成员对英特尔即将推出的 **Crescent Island** 表示兴奋，这是一款仅用于推理的 GPU，采用 **Xe3-LP** 架构和 **160GB** 内存，拥有 **1.5TB/s** 的带宽，称这是一个快速进步的时代，*就像 2000 年的游戏业再次降临*。
   - 还提到应该可以给它双倍的内存，因为英特尔使用的是 32Gb 芯片，而 LPDDR5x 最高可达 128Gb。
- **通过视觉基础编码器对齐 Token**：[这条推文](https://x.com/bowei_chen_19/status/1973085809365405705)重点介绍了一篇关于使用 **Visual Foundation Encoders** 作为 Diffusion 模型的 Tokenizer 的论文。
   - 提供了[项目页面](https://aligntok.github.io)和 [Arxiv 论文](https://arxiv.org/pdf/2509.25162)的链接。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **谷歌利用 Gemma 将细胞转化为句子**：Google AI 开发者推出了 **C2S-Scale 27B Gemma 模型**，该模型将单细胞基因表达数据 Tokenize 为 LLM 可读的“细胞句子”，并可在 [Hugging Face](https://huggingface.co/) 上获取。
   - 在验证研究中，该模型提出了一个新的假设（**silmitasertib 增强了肿瘤中的免疫信号**），实验分析显示 **抗原增加了 50%**，这是一个潜在的新免疫治疗靶点。
- **尽管有 GPT-5，Anthropic 的收入仍攀升至 70 亿美元**：尽管推出了 GPT-5 和 Codex，**Anthropic 的年化收入**已从 2025 年 1 月的 **10 亿美元**迅速攀升至 10 月中旬的 **70 亿美元**。
   - 成员们一致认为，*他们的 CLI Agent 目前遥遥领先，甚至到了离谱的地步——至少在我使用它们的方式上是这样*。
- **AI 奶奶吸引了 200 万粉丝**：一个名为 **grannyspills** 的完全由 AI 生成的网红，一个直言不讳、拜金的奶奶，散布毒舌约会建议，于 7 月推出，即将突破 **200 万** Instagram 粉丝，详情见[此 X 帖子](https://x.com/venturetwins/status/1978852719335985309)。
   - 关于观众是否在意她是虚构的争论正在酝酿，一些人称赞这个讽刺角色，而另一些人则担心 AI 对文化的影响。
- **收到投诉后，OpenAI 在 Sora 中屏蔽了马丁·路德·金**：在收到关于 **马丁·路德·金博士 (Dr. Martin Luther King Jr.)** 的不尊重 AI 生成视频剪辑的投诉后，OpenAI 已暂停任何描绘金博士的 **Sora** 输出，同时增加新的护栏，根据[此 X 帖子](https://x.com/OpenAINewsroom/status/1979005850166648933)。
   - 大多数用户批评此举是滑坡效应的让步，将公众人物私有化，并可能引发无休止的删除请求，此前 **金氏家族遗产委员会 (King Estate)** 要求禁止使用这位历史人物的肖像。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 社区辩论“代理式搜索 (Agentic Search)”的定义**：成员们讨论了在 DSPy 中实现“代理式搜索”的方法，建议可以通过传递一个名为 `hybrid_search` 的函数来实现。
   - 一些人认为“代理式搜索”是一个营销术语，涉及模型使用工具来达成目标，其中一位成员对该领域人为制造的复杂性表示沮丧：*这个领域有太多的营销和人为的复杂性，真令人气愤*。
- **Claude Code 的代理式搜索启发了 DSPy 的复现**：一位成员在对语义搜索感到不满后，试图在 DSPy 中复现 **Claude Code 的代理式搜索**。他发现 **Claude Code** 使用 *ripgrep* 在语料库中进行术语搜索，在将上下文添加到 LLM 之前先筛选文档。
   - 他们分享了 [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/) 来解释这一策略。
- **OpenAI 传出实现 RLM 的传闻**：一位用户发布了一张截图，似乎暗示 OpenAI 内部使用了 **RLM (Recurrent Language Model)**。
   - 这暗示了 *RLM* 可能正在影响 OpenAI 的更新，引发了社区的兴趣和推测，并分享了 [Alex Zhang 关于 RLM 的博客文章](https://alexzhang13.github.io/blog/2025/rlm/)。
- **Groq 在 OpenRouter 中出现故障？**：一位用户报告了在 **OpenRouter** 中使用 **Groq** 时遇到的问题，即使将其配置为唯一供应商也是如此，并发布了 [OpenRouter 设置页面](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794)的截图。
   - 尽管进行了特定配置，系统仍默认使用其他供应商而非 **Groq**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **租用 GPU 的烦恼：vllm Profiler 权限问题**：由于供应商对内核级操作的限制，一位用户在租用的 GPU 上使用 **vllm 的 profiler** 时遇到了 `CUPTI_ERROR_NOT_INITIALIZED` 错误。
   - 使用 `sudo` 的建议没有帮助，因为用户没有 sudo 权限，这导致他们开始寻找可以进行 profiling 的单 GPU 租用服务。
- **DeepSeek 在受限的 H20 上大显身手**：都市传说称 DeepSeek 使用 **PTX/SASS** 指令来规避内存带宽限制，从而在资源受限的情况下实现了一个强大的模型。
   - 尽管美国的限制导致中国的 **H100** 被降级为 **H800**，随后又进一步降级为 **H20**，但足智多谋的工程师们仍在继续创新。
- **谷歌 Jaten Op 扩展系统评析**：成员们讨论了 [Google 的 torchax/ops/jaten.py](https://github.com/google/torchax/blob/main/torchax/ops/jaten.py)，称赞了其**算子注册扩展系统 (op registration extension systems)** 的易用性。
   - 一位成员对那些重新实现复杂特殊函数的人表示同情，并引用了[文件中的第 4654-4776 行](https://github.com/google/torchax/blob/c3d6ee322ad864eac0e1d3f557d459628e09819d/torchax/ops/jaten.py#L4654-L4776)。
- **对称内存 (Symmetric Memory) 释放多 GPU 内核的协作能力**：一位成员分享了 [PyTorch 2.9 博客文章](https://pytorch.org/blog/pytorch-2-9/)，介绍了 **PyTorch Symmetric Memory**，它简化了在 NVLink 和 RDMA 网络上编写多 GPU 内核的程序。
   - 这一创新实现了**内核内通信 (in-kernel communication)**、**超低延迟远程访问**以及**自定义通信模式**。
- **Rubin CPX 支持奇特的数据格式！**：一位成员指出 **Intel** 在支持多种数据格式方面表现出色，以极低的计算开销简化了**软件级块浮点 (software-level block floats)**。
   - 结合可能支持 **CXL** 的架构，以及用于实现 **1.5 TB/s** 内存带宽的 **1280 位总线**，这使得 Intel 成为连接 CPU 和内存的强有力竞争者。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Google 发布 Coral NPU Verilog**: Google 已在 **Apache 2** 许可证下开源了 **NPU 模块**的 Verilog 代码，可在 [GitHub](https://github.com/google-coral/coralnpu) 上获取。
   - 该矩阵核心类似于 **AMD 的 NPU**，但使用了 **RV32 核心**，可能作为 **Mojo 可移植性**的测试平台。
- **窥探 Mojo 标准库内部**: 一位标准库贡献者警告不要过度使用 `__type_of()`，因为其语义仍在演变，尽管团队已决定去掉 `__` 前缀，因此这些很快将被命名为 `type_of(x)` 和 `origin_of(x)`。
   - 有用户报告在使用 `__type_of()` 时收到警告，澄清是因为它使用了名称但未使用值，需要使用 `_a` 来消除错误。
- **Mojo 的游戏梦想**: 用户们正关注 Mojo 在游戏开发方面的潜力，因其具有类 Python 的语法和系统级语言的性能，目标是通用化用途。
   - 爱好者们强调了像 [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3) 这样的项目，并对 Textual 移植以及完整的音频/MIDI 功能表示了兴趣。
- **Mojo 中涌现 TUI 框架**: 社区讨论了 Mojo 的 TUI 框架，借鉴了 Textual、ELM 应用（如 Go 语言中的 BubbleTea），并参考了 [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo) 等仓库。
   - 一位用户暂停了其受 ELM 启发的 TUI 框架 [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo)，以等待 Mojo 进一步成熟。
- **MAX Python API 正式开源**: Modular AI 已完全开源 **MAX Python API**，增强了社区的访问和贡献。
   - [这篇论坛帖子](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379)中提供了新开源 Python 模块的完整列表。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户吐槽 Jotform 联系帮助**: 一位用户批评了 **Manus AI** 的联系帮助系统，建议使用 **AI Agent** 而非*简单的表单*，并提议建立一个包含用户反馈积分的订阅机制。
   - 他们抱怨响应时间过长，指出*用户需要的是服务，而不是因为服务标准而等待 3 天才得到的回复*。
- **Beta 项目备受期待**: 多位用户询问如何加入 **Beta 项目**以体验扩展的 **Manus Agent** 功能，该项目目前仅对 **Manus Fellows** 开放。
   - 一位用户赞叹道：**Manus Agent** *解决了我的所有问题，它是我用过最好的工具！*
- **用户被锁定，需要账号支持**: 一位用户报告因**手机验证问题**被锁定在账号之外，发现帮助中心起不到作用。
   - 一位社区成员提出协助，要求提供账号邮箱地址，并保证说：*在这里大概能得到帮助*。
- **OpenAI 依赖引发部署僵局**: 一位用户报告部署失败，因为 **OpenAI 需要编译 pydantic_core**，而 **Manus 部署环境**不支持。
   - 一位成员提出通过利用预配置的 **OPENAI_API_KEY** 环境变量和更简单的 **HTTP client**，创建一个不带 **OpenAI 依赖**的版本。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 性格蒸馏吸引爱好者**: 成员们探索使用 **10 万条 Kimi K2** 回复示例来微调 (finetuning) **Qwen 3-4B** 基础模型，旨在蒸馏其性格。
   - 虽然注意到目前缺乏现成模型，但他们建议像 [Unsloth](https://github.com/unslothai/unsloth) 这样的工具现在让微调变得更加容易。
- **API 成本阻碍 Kimi K2 微调**: 对 **1B** 版本进行 **Kimi K2** 微调的兴趣因获取 **10 万条示例**的 **API** 高昂成本而受挫。
   - 有人建议使用 **1 万条示例**或更少的过滤数据集作为务实的替代方案，以减轻开支。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 破坏性变更引发困扰**：一位偶尔使用 **tinygrad** 的用户报告称，在交付新的 openpilot 模型时，频繁的破坏性变更（breaking changes）需要进行提交二分查找（commit bisection），这导致了难以理解的错误。
   - 该用户指出，针对 **845** 的 **IMAGE hacks** 是不稳定的根源，并对管理频繁更改的环境变量提出了挑战。
- **Shapetracker 弃用动摇 Tinybox 保修状态**：随着 **Shapetracker** 的弃用，一位用户询问了 **Tinybox** 的保修状态。
   - 另一位用户提供了 [Di Zhu 关于 Shapetracker 的文章](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md) 链接来解释当前情况。
- **Tinygrad 应提供明确的失败消息**：一位用户建议 **tinygrad** 应该针对使用 **IMAGE=**、**NOLOCALS=** 和 **GRAPH_ONE_KERNEL=** 时不支持的设备/硬件组合提供明确的失败消息。
   - 该用户表示，真实的编译失败与错误的配置之间容易产生混淆，这阻碍了他们的调试工作。
- **请在 Python 中设置默认设备**：一位用户询问如何在 Python 中设置默认设备，类似于 **Device.set_default('CL')**，以便在 Python 脚本中交叉检查不同的 backends。
   - 另一位成员澄清说，设置 **Device.DEFAULT = "CL"** 即可实现所需结果。
- **Tinygrad Stats 网站缺少历史数据**：一位用户注意到 [tinygrad stats 网站](https://stats.tinygrad.win/) 仅包含过去 25 天的数据，限制了对长期性能趋势的评估。
   - 他们希望为失败的编译编写测试，但考虑到失败与模型架构、设备和 FP16 配置的特定相关性，面临着挑战。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 获得 Aider CLI 快速启动**：一位用户提出了一种通过命令 `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff` 开始在 aider CLI 中使用 **Grok** 模型的方法。
   - 这允许立即使用该模型编辑代码。
- **Ollama 的 Qwen 2.5 Coder 7b 模型缺少元数据**：一位用户报告了从 [ollama.com](https://ollama.com) 获取 **Qwen 2.5 Coder 7b** 模型的有效 `metadata.json` 文件时遇到问题。
   - 该用户指出模型输出乱码，并请求一个 `metadata.json` 示例以解决此问题。
- **Ollama 面临文件集成问题**：一位用户在将 Ollama 模型与 `filename` 进行文件集成时遇到问题，希望聊天机器人能在消息后自动整合指定文件（例如 `SomeFile.txt`）的内容，但未能如预期工作。
   - 问题集中在将外部数据与 ollama 集成以获取聊天机器人上下文（context）方面。
- **Ollama 模型出现故障**：一位用户在使用特定的 Ollama 模型 **Qwen 2.5 Coder 7b** 时遇到了不明原因的麻烦。
   - 该用户表示他们一直在使用许多 Ollama 模型，但唯独 **Qwen 2.5 Coder 7b** 无法正常协作。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 特性矩阵中的 Discovery 解析**：一位成员寻求关于 [Model Context Protocol 特性支持矩阵](https://modelcontextprotocol.io/clients#feature-support-matrix) 中 *Discovery* 含义的澄清。
   - 它指的是 *支持通过 tools/list_changed 通知来发现新工具*。
- **MCP 关注工具分组 SEP**：反馈建议制定一个分组 SEP，以支持所有 **MCP primitives**（Tools、Prompts 和 Resources）的分组，而不仅仅是工具组。
   - 关于层级分组（hierarchical groups）的讨论正在 [GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1567#discussioncomment-14680104) 上进行。
- **贡献者询问：规划 MCP 的航向**：一位成员询问了为分组功能创建新 **SEP document** 的后续步骤，包括社交化、反馈收集以及潜在的原型实现。
   - 鉴于下一个规范版本将于 2025 年 11 月发布，该成员旨在优先推动相关工作以纳入 **SEP roadmap**。

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Claude Haiku 4.5 登陆 Windsurf**：**Claude Haiku 4.5** 现在已在 Windsurf 中上线，消耗 **1x credits**，其编程性能与 **Sonnet 4** 相当。
   - 正如 [Windsurf 的 X 帖子](https://x.com/windsurf/status/1978512184343662707)所宣传的，它的成本仅为三分之一，且速度提升了 2 倍以上。
- **SWE-grep 模型推送到 Windsurf**：专为快速 Agent 搜索（>2800 TPS）设计的全新 **SWE-grep** 和 **SWE-grep-mini** 模型正逐步向 Windsurf 用户开放。
   - 根据 [Cognition 的博客文章](https://cognition.ai/blog/swe-grep)和 [X 帖子](https://x.com/cognition/status/1978867021669413252)，这些模型通过 Fast Context 子 Agent 集成，将正确的文件呈现给你的编程 Agent 的速度比以前**快 20 倍**。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道沉寂太久，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道沉寂太久，请告知我们，我们将将其移除。

---

你收到此邮件是因为你通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
你可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1428095522148847720)** (1247 messages🔥🔥🔥): 

> `Geometry Dash 克隆, Gemini 3 Pro, AI 视频生成, LMArena 限制, Sora 2 代码` 

- **使用 Gemini 3 Pro 编写 Geometry Dash**：一位用户成功引导 **Gemini 3 Pro** 在约 30 秒内生成了一个完全可玩的 [HTML 版 Geometry Dash 克隆](https://link.to.clone)，包括音乐和物理效果。使用的 Prompt 为：*Generate full HTML file of a clone of Geometry Dash, but if it was made in the 2000s, add music to levels (make the music using JS varied music that reflect levels) same physics as Geometry Dash game we want a full playable game. All in one HTML file, minimum 1k lines*。
   - **Gemini 2.5 Pro** 等其他模型未能使用相同的 Prompt 生成可运行的游戏，这引发了对 **Gemini 3** 的期待。
- **Gemini 3 Pro 预期**：成员们讨论了 **Gemini 3 Pro** 在编程方面可能超越 **GPT-5 Pro**，并推测其**性能将提升 5-10%**。
   - 传闻该模型正在 AI Studio 上进行 A/B 测试，促使用户寻找访问方法，但也有人担心 **Google** 可能会施加 Token 限制以节省服务器资源。
- **LMArena Video Arena 机器人：故障与解决方案**：用户报告了 **Video Arena 机器人** 的问题，例如“生成响应时出错”以及视频生成能力和机器人的整体限制。
   - 管理员确认团队已意识到这些问题并正在积极修复，建议用户参考 bugs 频道进行故障排除。
- **Sora 2 霸榜视频排行榜**：**Sora 2 Pro** 已登上 [Text-to-Video 排行榜](https://lmarena.ai/leaderboard/text-to-video)，目前与 **Veo 3** 和 **Veo 3 Fast** 并列 **#1**，而 **Sora 2** 位列 **#3**。
   - Pro 账户可以使用更长的视频长度（长达 25 秒）且无水印，这引发了关于视频质量和模型偏好的讨论。
- **社区请求 PDF 上传和消息编辑功能**：LMArena 社区对新功能表现出浓厚兴趣，如 **PDF 文件上传**、**消息编辑**、**刷新响应**和**删除消息**。
   - 管理员已确认这些功能已列入未来实施的计划中。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1428528481007960194)** (1 messages): 

> `排行榜更新, Claude-Haiku-4-5, Text Arena` 

- **Claude-Haiku-4-5 占据第 22 位！**：Text 排行榜已更新，**Claude-Haiku-4-5** 已经上榜，目前排名 **#22**。
   - 鼓励用户查看 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)并在指定频道分享想法。
- **Text Arena 排行榜已刷新**：Text 排行榜已根据最新排名进行了刷新。
   - 查看 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)以了解更新后的排名情况。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1428097062351802468)** (1279 条消息🔥🔥🔥): 

> `Comet Browser, Perplexity Pro, AI Models, Claude, RAG` 


- **Perplexity Pro 用户 Airtel 套餐被撤销！**：多位使用 **Airtel Perplexity Pro** 套餐的用户报告称其订阅突然终止，支持团队声称正在调查此问题，但部分用户反映未得到回复。其他用户表示在免费试用期间被扣费。
- **Comet Browser 推荐计划及盈利潜力**：部分用户报告通过 **Comet Browser** 的推荐（**每位用户 $5-$15 USD**）赚到了钱，而其他用户则反映难以收到推荐奖金，工作人员已意识到这一潜在的 [bug](https://discord.com/channels/1047197230748151888/1047649527299055688/1428408489593479258)。
   - 一名用户因在分享频道分享 **Perplexity Pro** 代码被禁言一天，另有一名用户报告了推荐计划可能存在的滥用行为，工作人员正在 [调查](https://discord.com/channels/1047197230748151888/1047649527299055688/1428335328748851240)。
- **Comet Jacks！这个漏洞是真的吗？**：部分用户一直在讨论一个名为 *Comet Jacking* 的漏洞。然而，其他人报告称这只是一个已解决的 [prompt injection issue](https://discord.com/channels/1047197230748151888/1047649527299055688/1428339356716496957)（提示词注入问题）。
   - 有人表示修改内部设置和访问其他用户的数据违反了服务条款，可能导致 **Perplexity** 账号被封禁。
- **Perplexity 幻觉与数学问题**：用户报告了在生成某些图像时遇到的挑战和错误（如帧比例问题），以及幻觉信息和数学题处理问题。此外，一位用户推荐使用 Gemini Pro，因为它可能提供最佳的准确度。
   - **Perplexity** 上的多位用户仍在等待支持团队的回复。
- **讨论 Perplexity 模型与限制**：成员们一直在讨论 **GPT-5**、3.0 以及 Pro 模式下的图像生成限制。此外，一位用户询问 **Comet** 如何处理包含 13300 页的 PDF 文件。
   - 一位用户建议使用 **RAG**，因为这是最佳选择，毕竟无法将所有内容都加载到 context 中。据了解，**Perplexity** 拥有 **RAG** 功能，但不确定是否允许上传该文件。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1428333666479640606)** (2 条消息): 

> `Perplexity Apps, Perplexity Game` 


- **Perplexity Game 上线！**：一位成员分享了一个基于 **Perplexity Apps** 的*游戏*链接供他人尝试：[Perplexity Game](https://www.perplexity.ai/apps/1a78bb4a-d123-4691-8810-38a5469ed917)。
- **深入探索 Perplexity Apps**：探索 **Perplexity** 内部由社区创建的应用程序，这些程序提供了独特的功能和体验。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1428099041853706292)** (5 条消息): 

> `Perplexity API Statistics Page, Timeout issue on Perplexity Sonar Deep Research Model, Spaces not allowing new chats` 


- **呼吁检查 Perplexity API 统计页面**：一位用户询问是否有人检查过关于 **API statistics** 的 Perplexity 页面。
- **用户报告 Perplexity Sonar Deep Research Model 的超时问题**：一位用户报告了 Perplexity **Sonar Deep Research Model** 的 **timeout issue**（超时问题），并在两小时前发布到 [社区论坛](https://community.perplexity.ai/t/perplexity-sonar-deep-research-model-timeout-issue-seeking-solution/2094) 后，请求团队成员进行调查。
   - 截至消息发布时，该用户尚未收到回复。
- **Spaces 故障导致无法开启新对话**：一位用户询问为什么他们的 **Spaces** 账号无法在任何现有空间内创建 **new chat**（新对话）。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1428171257962041466)** (2 条消息): 

> `ChatGPT Saved Memories, Sora Updates` 


- **ChatGPT 获得记忆管理功能**：**ChatGPT** 现在可以自动管理保存的记忆，用户可以按新鲜度搜索/排序记忆，并在设置中重新调整优先级。
   - 该功能从今天起在全球范围内向网页端的 **Plus** 和 **Pro** 用户推出，解决了“记忆已满”的问题。
- **Sora 的分镜脚本与更长的视频**：[Sora](https://video.twimg.com/amplify_video/1978653248572567552/vid/avc1/704x1280/lIHMEtPzOCUTOkfm.mp4) 更新已发布：**Storyboards**（分镜脚本）现已在网页端向 **Pro** 用户开放。
   - 所有用户现在可以生成长达 **15 秒** 的视频，而 **Pro** 用户可以在 **Sora** 应用和网页端创建长达 **25 秒** 的视频。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1428095910038077502)** (618 条消息🔥🔥🔥): 

> `Transformer 音乐模型, GPT-5 推测, AI Safety 竞赛, AWS vs Azure 用于 AI, Gemini 3.0 访问权限` 


- **基于 Transformer 的音乐模型遭遇 NaN Loss 瓶颈**：一名成员在训练一个基于 Transformer 架构的音乐生成模型时遇到了 **NaN loss** 问题，尽管尝试了各种 **learning rates** 和 pre-normalization，数据集包含 **300 小时** 的钢琴音乐。
   - 该成员探索了 tokenization 方法（**REMI**）和模型规模（**3M 到 25M 参数**），并报告该问题出现在 **120-150 step** 左右，无论如何调整超参数都会发生，因此怀疑是与数据相关的 bug。
- **关于 GPT-5 “更少拒绝”人格的传闻**：一位用户提到一个帖子称 **GPT-5** 将会 *更少拒绝* 且 *更像 4o*，但社区成员对此做法褒贬不一。
   - 一些人认为 *更少拒绝总是一件好事*，而另一些人则担心这可能导致模型无论道德与否都会同意所有要求。
- **AI Safety 内容竞赛广告**：一项 AI safety 竞赛正为推广 AI safety 的创意内容（故事、漫画、视频）提供 **10,000 美元** 奖金，链接见 [keepthefuturehuman.ai/contest/](https://keepthefuturehuman.ai/contest/)。
   - 参与者推荐成功的投稿还可以获得 **30 美元亚马逊礼品卡**，[Siliconversations](https://siliconversations.com/) 提供了一个关于该竞赛的 **3 分钟视频** 概述。
- **LLM 基础设施之争：Azure vs AWS**：成员们正在辩论用于 LLM 基础设施的 **Azure vs AWS**，指出 **AWS** 拥有 *Claude*、*Titan* 和 *open weights* 模型，而 **Azure** 拥有 *OpenAI* 以及大量的 *open weights* 模型。
   - 有人对中型企业使用 **Azure** 的潜在限制及其对 **Microsoft Power Platform** 的替代方案提出了担忧。
- **Gemini 3.0 泄露与 Comet 的早期访问**：成员们在 **Gemini.google.com** 网站上发现了硬编码的 **Gemini 3.0 Pro** 引用，暗示即将发布。一位成员还分享了一个关于通过 **Comet** 访问 **Gemini 3.0** 的 [YouTube 视频](https://www.youtube.com/watch?v=ba3ZZJkZxAY)。
   - 成员们还在讨论语音 AI 助手，有人提到 **Chat GPT** 是一个不错的选择。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1428429200016277614)** (3 条消息): 

> `Sora 在德国, AI 语音助手, 基础 AI 支持` 


- **Sora 希望进入德国**：一位成员表达了希望 **Sora** 能在 **德国** 使用的愿望。
   - 这表明了用户对 **Sora** 国际可用性和可访问性的兴趣。
- **语音助手开发咨询**：两名成员询问另一名成员是否有构建 **AI 语音助手** 的经验。
   - 该问题表明了对 **AI 语音技术** 实际开发和实现的兴趣。
- **正在使用的基础 AI 支持**：一位成员表示他们正在利用基础的 **AI 支持** 进行工作审查、笔记记录和脚手架搭建（scaffolding）。
   - 这突显了 **AI** 在日常任务和工作流程中的实际应用。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1428098039326703698)** (41 messages🔥): 

> `Reporting in OpenAI Discord, Cursed 8-ball, October University of Poly-Disciplinary Studies, Request for AI torture prompts, Manipulating sound with prompts` 


- **报告流程说明**：一名成员澄清说，机器人 <@1052826159018168350> 没有举报功能，但用户可以通过 App 或 Modmail 进行举报，详见 [instructions channel](https://discord.com/channels/1046317269069864970/1107330329775186032)。
   - 他们演示了如何使用 App 举报消息：将鼠标悬停在消息上，点击“...”，然后选择“apps”和“report message”，如[此图](https://cdn.discordapp.com/attachments/1046317269069864970/1428103477795487826/image.png?ex=68f29a01&is=68f14881&hm=67a6cc078b349abe409c54254d49ab4be036f4364326065b3ae91e5fcf087cdc)所示。
- **被诅咒的 8-Ball**：在使用 8-ball 进行咨询后，一位用户开玩笑说 *“这个 8-ball 似乎比我知道的还多”*，并推测它可能被诅咒了，调侃说这可能是因为现在是十月。
   - 另一位用户回复了一个 [ChatGPT 链接](https://chatgpt.com/share/68effa7b-b67c-8011-bca9-b9384a76ed6e)，其中的回答是：*“结果不确定。可能闹鬼。”*
- **不欢迎伤害幻想**：一名用户请求能让 *"ChatGPT 感到痛苦"* 的 Prompt，因违反社区准则（特别是关于尊重和友善的规定）而被制止。
   - 另一名成员强调，该频道是为了构建有用的系统，伤害幻想是不恰当的，建议转向可衡量的安全评估，例如 *Refusal rates*（拒绝率）或 *Jailbreak resistance*（抗破解能力）。
- **动漫 Prompt 复制模板**：一名成员分享了一个用于通过 **Sora** 生成动漫图像的[模板](https://discord.com/channels/1046317269069864970/1046317270017093705/1200514049426374726)，包括风格、设定、反派、音频和镜头动作等部分。
   - 该模板详细规定了动画风格、环境、色调、反派特征、音乐氛围以及带有时间点的逐镜头动作。
- **通过直接告诉模型来生成图像**：针对用户的请求，一名成员建议他们可以通过告诉模型想要什么来“制作 Prompt”以生成图像，就像询问他人一样。
   - 他们鼓励用户在新的对话中对 Prompt 进行迭代，清晰地解释他们想要在新图片中体现的细节和变化。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1428098039326703698)** (41 messages🔥): 

> `Reporting messages, Cursed 8-ball, October University of Poly-Disciplinary Studies, Sound of music with prompts, futuristic robot in a storm` 


- **举报消息变得简单**：一名成员询问如何举报消息，并获知机器人没有此功能，但可以通过 App 或发送 Modmail 举报：将鼠标悬停在消息上，点击“...”，然后选择“apps”，再选择“report message”。
   - 对同一条消息进行多次举报是完全可以的；一旦 Mod 有空，他们可以轻松处理举报并检查被举报的消息。
- **被诅咒的 8-ball 比你懂得多**：一名成员开玩笑说他们的 8-ball 似乎比他们知道的还多，另一名成员回复了一个 [ChatGPT 链接](https://chatgpt.com/share/68effa7b-b67c-8011-bca9-b9384a76ed6e)，显示内容为 *"结果不确定。可能闹鬼。"*。
   - 第二名成员觉得很有趣，因为他们在对话中要求模型执行的主要“后台任务”之一是在“October University of Poly-Disciplinary Studies”的设定下模拟一个世界和大量角色，这具有一套复杂的含义。
- **这里不欢迎伤害幻想**：一名成员请求能让 ChatGPT 感到痛苦的 Prompt，另一名成员回复说这不合适，并且 *"这个频道是为了构建有用的、可测试的系统。"*。
   - 第二名成员补充说，如果你正在测试安全性，请开启一个新话题，并提供可衡量的 Eval（Refusal rate、Jailbreak resistance、挑衅下的文明程度）和通过/失败表格。
- **将整段内容复制并粘贴到 Sora**：一名成员分享了一个 Prompt 模板，用于在 Sora 中通过 Prompt 操纵音乐声音，主要针对动漫，包括 STYLE、SETTING / PALETTE、ANTAGONIST、AUDIO、CAMERA / ACTION BEATS 和 ANIMATION NOTES 等部分。
   - 它使用 Markdown 表格来帮助组织 CAMERA / ACTION BEATS，包含 Time、Shot & Action 以及 Notes 列。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1428095116207325294)** (264 messages🔥🔥): 

> `Spammer bot, Ultron, Windows spelling error, vLLM Assertion Error, Mobile 5090` 


- **Unsloth Bot 能否阻止垃圾邮件发送者？**：成员们讨论了是否可以对 **Unsloth bot** 进行编程以抓取垃圾邮件发送者，建议包括使用现有的数据集（如 **madpinger 的 Discord bot 数据集**或 **Reddit bot 数据集**）来制造“机器人对抗机器人”的局面。
   - 一位用户开玩笑地提到这种情景会导致 **Ultron** 的诞生，而其他人则指出目前已有现成的用于垃圾邮件检测的 Discord bots。
- **Unsloth 修复文档中的 Windows 拼写错误**：一位用户报告了 [Unsloth Windows 安装文档](https://docs.unsloth.ai/get-started/install-and-update/windows-installation) 中的一个拼写错误（将 Windows install 写成了 losedowsinstall），并提供了截图作为证据。
   - 另一位用户感谢了他的发现，并建议使用 **WSL** 而非原生 Windows 安装，因为其支持更好；随后该拼写错误被迅速修复。
- **vLLM Assertion Error 的困扰**：一位用户在使用 **vLLM 0.11.0** 部署 **Qwen3-VL-4B-Instruct-unsloth-bnb-4bit** 模型时遇到了 **AssertionError**，具体与 `linear.py` 中的形状不匹配有关。
   - 该问题指向了一个相关的 [GitHub issue](https://github.com/unslothai/unsloth/issues/1886)，引发了关于这是模型问题还是 vLLM 问题的讨论，其他人则建议检查 **CUDA** 安装情况。
- **DGX SPARK 引起关注**：Unsloth 团队通过一条 [推文](https://x.com/UnslothAI/status/1978456629613084926) 强调了他们对 **DGX SPARK** 的支持，引发了关于其性能以及是否适合 **nvfp4** 训练的讨论。
   - 观点各异，有人认为它在推理方面表现尚可，而另一些人则认为由于带宽限制，它可能属于“发烧友（hobbyist）”级别，一位用户甚至误将 hobbyist 说成了 *hobbit*。
- **Qwen3-VL 在 LM Studio 的结果引发困惑**：一位用户观察到，在默认的 Unsloth notebook 中运行 **Qwen3-VL-8B** 的推理结果，与在 **LM Studio** 中本地运行默认 **MLX** 模型的结果存在显著差异。
   - 一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1o7l1io/lm_studio_and_vl_models/) 链接表明，**LM Studio** 可能会将图像缩放至 **500px**，这可能会影响模型性能。其他评论还提到了 **Jan AI** 的使用以及建议“双重检查 chat template”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1428205460548358265)** (9 messages🔥): 

> `New Spark Owners, Microcenter selling sparks` 


- **新 Spark 车主引发欢迎热潮**：成员们正在欢迎新的 **Nvidia Spark** 车主。
   - 一位新车主提到，他们是在 **Microcenter** 排队买到的 **Spark**，而他们直接从 **Nvidia** 预订的设备将于次日送达。
- **GLM-4.6 快速入门指南**：一位新用户正在等待 **DAC 线缆**以便在本地尝试 **GLM-4.6**，并表示这个频道看起来很有趣！
   - 另一位成员提到 [这里](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally) 有一份关于 **GLM-4.6** 的指南。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1428095337259733073)** (230 条消息🔥🔥): 

> `Svelte vs React, 黑客松信息延迟, Context Collapse & Synth Data, PeftBase 函数位置, 学习率退火` 


- **Svelte 被认为优于 React，步骤虽多但更少烦恼**：在花了三周时间使用 **Svelte** 修改 **Open-WebUI** 后，一位成员发现看了 Fireship 的视频后觉得它“就是步骤更多的 **React**，但没那么烦人”。
   - 该成员将最初的困惑描述为“是特性而非 Bug”，并认为 **Svelte** 实际上非常酷。
- **黑客松信息延迟，参赛者一头雾水**：成员们对即将到来的黑客松缺乏信息表示沮丧，距离活动开始仅剩两天。
   - 参赛者哀叹缺乏关于准备工作的细节，且不得不依赖在*开赛当天*获取信息，担心会*措手不及*。
- **通过常规程序解决 Context Collapse 与合成数据问题**：一位成员建议，使用常规程序解析上下文添加可以阻止上下文工程方法的 Context Collapse，并进一步将其与 Early Experience 或 SEAL 结合，以获得更好的新型合成数据（Synth Data）。
   - 这将产生“常规但更好的合成数据”。
- **为了可维护性讨论 PeftBase 函数的放置位置**：维护者们在 **OneTrainer** 项目中就如何实现恢复训练时的 rank/alpha 不匹配检查产生了分歧，具体在于代码中该检查函数的放置位置。
   - 一个建议是，如果该函数是所有 **PEFT** 模块类型的通用工具，且不依赖于 **LoRA** 特有的细节，则应将其放在 **PeftBase** 中。
- **验证集损失（Val Loss）上升预示需要退火**：当在固定学习率下出现 **loss plateau**（损失平台）时，验证集损失上升信号预示着需要进行退火（anneal），并在继续训练前开始降低学习率。
   - 另一位成员建议，模型可能仍在 **val_loss** 覆盖范围之外进行学习，而另一位成员则认为模型可能在学习当前 batch 内容的同时，正在“遗忘”当前 batch 之外的其他部分。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1428282111244242944)** (40 条消息🔥): 

> `硬件优化, 设备映射 (Device Mapping), SageMaker 环境, Gemma 3` 


- **硬件优化探讨**：一位用户询问了 Unsloth 中的**硬件级优化**，例如调整频率水平或功耗限制；回复澄清说 Unsloth 专注于 **kernel、内存优化和量化**，而非直接的硬件调整。
   - 提供了一篇研究论文链接，讨论 [用于能效神经网络推理的自适应近阈值计算](https://dl.acm.org/doi/abs/10.1145/3731545.3735119)；虽然欢迎贡献，但由于保修和敏感性问题，不太可能直接修改用户硬件。
- **设备映射 (Device Mapping) 'balanced' 行为**：一位用户分享了在使用和不使用 `device_map="balanced"` 时的性能差异图像，指出 balanced 设置似乎耗时翻倍且步数翻倍；对此澄清说 **batch size 总是乘以 GPU 的数量**。
   - 一位社区成员问道：*`device_map="balanced"` 真的适合微调更大的模型吗？看起来它总是只占用一个 GPU 约 1GB 显存，而占满了另一个 GPU*。
- **SageMaker 环境困扰**：一位用户在 AWS SageMaker 上复现 Colab 笔记本 (**Orpheus_(3B)-TTS.ipynb**) 时遇到了版本兼容性问题，特别是 **PyTorch 2.8.0+cu126**。
   - 解决方案是使用 [Unsloth Docker 镜像](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker)，用户还提醒说安装最新的 PyTorch 需要正确的 CUDA 版本。
- **Gemma 3 再次受阻**：一位用户报告说 **Gemma 3** 在 Unsloth 中再次无法工作，调用 `.train()` 会导致错误，并附带了一个 [Discord 消息](https://discord.com/channels/1179035537009545276/1428389199932817469) 链接。
   - 未提供更多细节。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1428451128101834752)** (3 messages): 

> `AI vs AI Chess, Multi-turn Attempts at Legal Moves` 


- **LLM 在国际象棋对决中展开较量**：一名成员创建了一个 [AI vs AI 国际象棋平台](https://github.com/Laszlobeer/chess-llm-vs-llm) 来评估具备下棋能力的 LLM。
   - 该项目让两个 LLM 在国际象棋比赛中对决，以观察哪一个表现更好。
- **LLM 国际象棋游戏获得多轮合法步法尝试**：一位用户建议，可以通过实现**进行合法步法的多轮尝试**来改进国际象棋平台，而不是在失败时求助于随机的合法步法。
   - 该成员理论上认为，这种改变可能会提高性能和棋局步法的质量。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

noimnull: 有人知道关于 Unsloth 的 fastinference 如何工作的博客吗？
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1428095086360789133)** (524 messages🔥🔥🔥): 

> `Cursor Pro+ plan issues, Cheetah pulled?, Cursor expensive with Claude 4.5, OpenRouter issues, Gemini 3.0` 


- **Cursor 订阅追踪混乱**：多名用户报告其 **Cursor Pro+ 计划**退回到了**免费版本**，问题影响了依赖自定义模型的功能；一名成员确认了类似的故障，称 *“某些东西刚刚弄坏了我的 Cursor，Agent 和 Edit 依赖于无法通过 API Key 计费的自定义模型。”*
   - 成员们猜测可能存在维护、价格变动或 Cheetah 被撤下的情况，一些人注意到 **Agent Window** 和设置中出现了错误。
- **Cheetah 失踪，Haiku 到来**：用户注意到 **Cheetah** 消失了，一些人猜测它会以不同的发布名称回归，并确认新的 **Claude-4.5-Haiku** 已启用。
   - 一位成员指出 *“Haiku 比 Cheetah 便宜，但 Haiku 比 Claude 4.5 Sonnet 便宜吗？Cheetah 是 1.25 / 10，Haiku 看起来是 1 / 5”*。
- **OpenRouter 的 API Key 集成问题**：成员们报告了将 **OpenRouter** 与 Cursor 集成时的问题，其中一人表示：*“我无法在 Cursor 中使用 OpenRouter，有人知道如何解决吗？”*
   - 建议的解决方案包括禁用其他模型和移除前缀，但监控流量的用户注意到 Cursor 并没有向 OpenRouter 发送请求。
- **Gemini 3.0：未来？**：成员们对 **Gemini 3.0** 表示期待，并认为 Gemini 在定价和上下文窗口长度方面具有优势。
   - 有人表示：*“到目前为止，Agent 化使用对我来说是最重要的基准测试。”*
- **Dashboard 工具追踪 Cursor 的 Token 成本**：一名成员展示了他们用于追踪产品中 Token 成本的 [Dashboard](https://token-watch.vercel.app/)；另一名成员也展示了他们自己构建的 Dashboard。
   - 成员们赞扬了 Dashboard 的使用，一些人贡献了代码和调试建议。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1428436530170564768)** (1 messages): 

> `Tokenizer layer, Agents Bugs, Goal Post` 


- **Tokenizer 问题引发 Bug 讨论**：一名成员报告了 **Tokenizer 层**无法完全完成任务的问题，怀疑这是一个 Bug 或是定义目标时出了问题。
   - 他们提到，以前指定一个 spec（规范）然后命令“完成 Tokenizer 层并确保其符合规范并通过所有测试”是有效的。
- **规范合规性担忧浮现**：讨论强调了关于 **Tokenizer 层**是否能一致地满足指定要求并通过所有测试的担忧。
   - 该用户的经历表明，Agent 遵守详细规范和测试协议的能力可能出现了退化，这引发了进一步的调查。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1428104621712474193)** (192 条消息🔥🔥): 

> `使用 LLM 分析 YouTube 转录文本，使用 LLM 进行矛盾检测，LLM 的 GPU 显存限制，针对特定任务的本地与云端 LLM 对比，LLM 使用的硬件建议` 


- **LLM 分析 YouTube 转录文本**：成员们讨论了使用 LLM 分析 YouTube 转录文本以获取有价值的见解，并指出真正的价值在于利用 LLM 对转录文本的 *内容发表见解*。
   - 这种方法可以更深入地理解并从视频内容中提取关键信息。
- **硬件限制阻碍本地 PDF 矛盾检测**：一位用户寻求关于使用模型在 **8-10 份 PDF 文档**中查找矛盾的建议，但成员们建议，由于 **Context** 限制，这需要更复杂的类似 Agent 的设置，而不是直接使用模型。
   - 他们建议采用排列组合对和摘要等策略来减小 Context 大小，但也警告说，即使是声称拥有 *100 万 Token Context Window* 的模型，在如此规模下也可能难以保证准确性。
- **GPU 显存限制本地 LLM 性能**：用户讨论了由于 **GPU 显存限制**导致的本地 LLM 局限性，一位成员强调，模型广告宣传的最大 Context 长度并不保证其能够有效利用它。
   - 有人指出，大多数本地模型在超过 **2-4 万 Token** 后就开始失去 Context 连贯性，强调了平衡 Context 长度与实际性能的必要性。
- **基于 LLM 的内容分析需要 Context 工程**：成员们强调了 Prompt Engineering 和 Context 工程对于有效利用 LLM 分析内容的重要性，尤其是在寻找矛盾时。
   - 有人警告不要简单地将大量文本塞进模型，因为这会导致结果被稀释且处理速度变慢，建议采用更结构化、迭代的方法，并配合示例和调优后的 Prompt。
- **探索无审查 AI**：一位用户询问了不会拒绝简单问题的 *未过滤 AI 模型*，随后有人推荐了经过 *uncensored*（无审查）和 *obliterated*（抹除拒绝）处理的模型，这些模型已移除了拒绝回答和审查机制。
   - 成员们建议搜索特定的微调者，如 **huihui-ai**、**TheDrummer**、**mlabonne** 和 **Jinx**，以找到此类模型。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1428109249258717215)** (258 条消息🔥🔥): 

> `GPU 利用率，DGX Spark，多 GPU，冷却方法，RAM 速度` 


- **9070XT 获得 LLM 任务**：一位拥有 **9070XT 16GB** 和 **RTX 2070 8GB** 的用户寻求关于利用 9070 的建议，提议它可以运行一个本地行政助理聊天机器人。
   - 另一位成员建议，凭借 16GB VRAM，它可以处理 **Q4** 量化的 **gemma3 12b**、**gpt-oss-20b** 或 **qwen3 14b** 等模型，并留有 Context 空间。
- **DGX Spark 遭到质疑**：一位成员认为 **DGX Spark** 的视频是赞助内容，并引用了一篇[评测](https://youtu.be/md6a4ENM9pg)，该评测建议通过搭建类似“比特币”的挖矿机可以获得更好的 4K 性能。
   - 它被描述为一种“开箱即用”的解决方案，可以运行大量模型以及不同的 Agent 设置和流水线，而无需了解任何 AI 知识。
- **混合 GPU 会干扰模型**：一位用户询问在 **3090** 的基础上增加 **3050** 或 **2060** 是否会对计算有显著贡献，还是仅仅作为 VRAM 扩展。
   - 有人指出，虽然额外的 VRAM 有帮助，但使用多张显卡可能会降低速度，因为它们 *不会同时工作*。
- **奇葩冷却装置征服计算**：一位用户分享了一张简陋 PC 设置的照片，使用了用胶带粘在机箱上的笔记本冷却垫。
   - 另一位用户展示了他们的廉价水冷设置，并附上了[他们的装备照片](https://cdn.discordapp.com/attachments/1153759714082033735/1428375752583417886/w4pfwsXSU-wPCbLSuPjtTXj6tymx0bu02dpMBv1e0yI.jpg?ex=68f2eed5&is=68f19d55&hm=b1fdc6b0bc368e8c240479168f9d999a81877d6186354b36a96bbca4dd6c1f77&)。
- **RAM 赛道启示录**：讨论内容包括对于 LLM 来说，**128GB** 的 **DDR4 3600** RAM 是否比 **64GB** 能显著提升性能。
   - 一位成员指出，对于 LLM 来说，拥有足够的内存至关重要，而且内存带宽极大影响性能，并提到 *如果你有 DDR5-8000，速度会快 4 倍*。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1428097262336213154)** (288 条消息🔥🔥): 

> `GPT 代码质量, Discord.py 命令更新, 上传大文件到 GitHub, 位精度优化, Hugging Face 访问令牌问题` 


- **Discord.py 命令需要更新！**: 一位成员报告 Discord.py 命令无法工作，机器人响应 *"No command called 'command' found"*，另一位成员建议是时候[更新命令了](https://tenor.com/view/walk-in-pizza-fire-community-sol-marms-marms-nft-gif-16842297911040510242)。
   - 机器人所有者回复称他们将 *"修复它"*。
- **GitHub 大文件传奇！**: 一位用户询问如何向 GitHub 上传大于 **25MB** 的文件，因为他们面临向 Google Drive 上传时间过长的问题，另一位成员建议使用 **GitHub CLI** 并参考[这些说明](https://cdn.discordapp.com/attachments/879548962464493622/1428283674251624521/github_over_25mb.md?ex=68f29914&is=68f14794&hm=4f2a32cd8bd636b8aa719aacc55bee90d5ee9c868e58e7430a2e7a5d7d996c6f&)。
   - 该用户随后遇到了 **git LFS** 错误，但不理解 'ref' 错误。
- **HF 访问令牌故障排除！**: 一位用户在意外创建并删除一个读取令牌后，在创建访问令牌时遇到问题，导致 UI 显示权限 **'no results found'**，并考虑[注销账号](https://tenor.com/view/rustic-relic-hunter-snap-out-of-it-are-you-outta-your-mind-gif-9081070089877950090)。
   - 其他用户建议清除浏览器缓存或使用无痕模式作为潜在修复方案，并指出应联系 HF 支持，建议发送邮件至 *billing@huggingface.co* 或 *website@huggingface.co*。
- **保障 Agentic 工作流安全：防御的艺术！**: 一位成员发起了关于减轻 Agentic 工作流（涉及电子邮件或个人账户）中潜在黑客风险的讨论，特别是关于 Prompt Injection（提示词注入），一位成员指出 *"激进的沙箱化和上下文隔离可能会有所帮助"*。
   - 另一位用户强调了*最小权限原则*，建议不要给 AI 任何不必要的权限以防止潜在滥用，并认为深度防御非常重要。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1428550796680892538)** (1 条消息): 

> `Influence Functions, 研究协作` 


- **Influence Functions 激发研究兴趣**: 一位成员对 **influence functions** 表现出兴趣，并寻求与在该领域有经验的人建立联系，引用了论文 [(1)](https://arxiv.org/abs/2308.03296) 和 [(2)](https://arxiv.org/abs/2411.12580v1) 作为理解和应用它们的资源。
   - 该成员正在其工作组内探索可能受益于该方法论的**新研究问题**，并愿意与该领域的专业人士合作。
- **Influence Functions 研究协作**: 发帖者有兴趣寻找一位在 influence functions 方面具有专业知识的合作者，以探索新的研究问题。
   - 他们提供了两篇相关论文的链接（[https://arxiv.org/abs/2308.03296](https://arxiv.org/abs/2308.03296) 和 [https://arxiv.org/abs/2411.12580v1](https://arxiv.org/abs/2411.12580v1)）作为背景材料。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1428105918805184533)** (15 条消息🔥): 

> `FRAI 框架, brain tuning, Mikus 冲突, 共振, 快餐` 


- **用户称他正在进行另一种 brain tuning！**: 一位用户声称他正在进行 A=1760 的 **brain tuning**，这比 A=440 快 4 倍。
   - 他开玩笑说，他**高效的打字速度**让别人很难理解。
- **用户说 Mikus 不喜欢他**: 一位用户报告了与 **Mikus** 的冲突，并希望深入了解。
   - 他开玩笑说 *也许他业余时间是个黑客，像黑客一样谁也不喜欢*。
- **用户讨论游戏成瘾**: 一位用户承认每月在**手机游戏上花费 300 美元**，并认为这是他最大的缺点。
   - 他计划减少游戏时间以专注于**模型训练**，开玩笑说 *我会尝试停止游戏转而进行模型训练（开玩笑的... 目前我两者兼顾）*。
- **用于责任制 AI 的 FRAI 框架亮相**: 一位用户分享了一个名为 **FRAI** 的开发者优先的 **Responsible AI** 框架，其 CLI 版本已在 [GitHub](https://github.com/sebuzdugan/frai) 上发布。
   - 作者正在寻求 **Star 和反馈**以改进该框架，并提供了指向其 [YouTube 频道](https://m.youtube.com/@sebuzdugan) 的链接，其中包含与 FRAI 相关的视频。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1428591577688707123)** (1 messages): 

> `Custom Blocks, Diffusers Library, Modular Diffusers` 


- **自定义模块助力 Diffusers**：Custom blocks 可以实现库中尚未提供但能无缝集成的功能。
   - 可以在[此处](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks-68c8e37c62a6b2a30fd58401)查看一些 custom blocks，并参考[文档](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/pipeline_block)。
- **Modular Diffusers 详解**：Modular Diffusers 支持通过 custom blocks 扩展功能。
   - 正如链接集合所示，这些模块可以无缝集成到现有的 Diffusers Library 中。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1428189575943028768)** (2 messages): 

> `Pixel Removal, Image Hole Filling` 


- **建议移除并填充图像像素**：一位用户建议从图像中移除所有亮度为 **[255, 255, 255]** 的像素。
   - 该用户随后建议填充因移除像素而产生的空洞。
- **讨论图像像素编辑**：一位用户提出了一种涉及移除特定像素亮度的图像编辑方法。
   - 该提议包括随后填充图像中产生的间隙或空洞的步骤。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1428357437592830125)** (1 messages): 

> `HuggingFace Inference Provider credits, AI Agents Hackathon, MCP, Production Hacks` 


- **HuggingFace 发放推理额度！**：所有黑客松参与者现在都可以获得 **免费** 的 [HuggingFace Inference Provider credits](https://huggingface.co/Agents-MCP-Hackathon-Winter25) 以参加在线黑客松。
   - 参与者将在赢取丰厚现金奖励的同时，学习 **AI Agents**、**MCP** 和生产级技巧。
- **AI Agents 冬季黑客松宣布！**：专注于 **AI Agents** 和 **MCP** 的最大规模在线[黑客松](https://huggingface.co/Agents-MCP-Hackathon-Winter25)已宣布于 25 年冬季举行。
   - 为参与者提供了专门的支持频道。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1428110591184211969)** (4 messages): 

> `PEFT Configuration Incomplete, TrackIO Dependency Issue, Lora Adapter Testing, Lighteval Support for Lora` 


- **PEFT 的 Smol Model 需要指定目标模块**：一位用户报告了 `ValueError`，原因是 Unit 1 实践练习中的 **PEFT configuration** 不完整，特别是需要针对 `["q_proj", "v_proj"]` 等模块。
   - 他们指出 `smolm3` 架构模型在 [PEFT 源代码](https://github.com/huggingface/peft/blob/e6f927bfecba238f81e940b7f560284e5829dc2e/src/peft/utils/constants.py#L87)中未被引用。
- **TrackIO 的依赖问题**：为 trackio 构建的 Space 失败，因为 `requirements.txt` 中缺少 **trackio==0.5.1.dev0** 的依赖。
   - 一位用户通过手动将其更改为 **0.5.2** 修复了此问题，并强调如果不进行此修复，第一次运行将不会被记录。
- **敦促用户修正微调模型测试**：一位用户建议修改针对 Lora 训练模型的[测试说明](https://huggingface.co/learn/smol-course/unit1/4#test-the-fine-tuned-model)，以包含加载 **Lora adapter** 的步骤。
   - 他们建议在对比基础模型与微调模型时，应更明确地说明使用哪个 tokenizer。
- **Lighteval 暂不支持 Lora**：一位用户指出，[Unit 1 训练模型](https://huggingface.co/learn/smol-course/unit1/5#lorapeft-on-jobs-optional)解释了如何使用 HF Jobs 和 **Lora** 训练模型，但 [lighteval vvlm](https://huggingface.co/learn/smol-course/unit1/6#4-evaluate-the-model-using-hf-jobs) 目前还不支持评估带有 Lora adapter 的模型。
   - 他们链接了一个[相关的 GitHub pull request](https://github.com/huggingface/lighteval/pull/611)，暗示需要一个解决方案。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1428100322936619092)** (3 messages): 

> `Agents course progress, Agents course timeline` 


- **课程进度显示异常**：一位成员报告了 Agents 课程的问题，特别是刷新页面后无法跟踪进度或查看已完成的测验。
   - 在给定的上下文中未提供解决方案或解释。
- **Agents 课程的时间线不明确**：一位成员询问了完成 Agents 课程的时间线。
   - 在给定的上下文中未提供具体的时间线。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1428104163606401144)** (197 条消息🔥🔥): 

> `Kimi K2 vs Claude 4.5 Haiku, Gemini 2.5 Flash 性能, Haiku 注意力跨度, CPU 推理, Tensor Logic` 


- **Gemini 2.5 Flash 碾压 Haiku、Deepseek R1 和 Kimi K2 —— 真的吗？**：一位成员声称 **Gemini 2.5 Flash** 的表现优于 **Haiku**、**Deepseek R1** 和 **Kimi K2**，但另一位用户发现 **Flash** 相当笨拙，尤其是在 Python 编程方面。
   - 他们补充说，即使是 **Gemini 2.5 Pro** 也会在明确要求不要加注释的情况下在代码中添加注释，这突显了编程依然是 Anthropic 的强项。
- **Haiku 编程能力强化：针对编程优化，其他方面表现崩溃？**：成员们讨论了 **Haiku 4.5**，指出它似乎针对编程进行了优化，但在大多数其他任务中可能会表现不佳，一位成员建议它*可能值得用于编程，但不适合其他任何事情*。
   - 还有观点认为 **Gemini** 很聪明，但没有针对 Agent 任务进行良好的训练，而 **Haiku** 的注意力跨度（attention span）较低，可能会影响其编程能力。
- **Tensor Logic 揭晓：连接逻辑与智能？**：一篇关于 **Tensor Logic** 的[论文](https://arxiv.org/abs/2405.08793)受到关注，该论文认为通过将逻辑推理转化为纯粹的张量代数（tensor algebra），它可以成为逻辑与智能之间的桥梁。
   - 这种方法将布尔推理、概率推理和谓词逻辑嵌入到一个单一的可微框架中，可能使模型能够*不仅仅是预测真理*，而是*证明真理*。
- **本地 AI 装备搭建：EPYC vs Mac Studio 内存带宽**：一场关于本地 AI 装备内存带宽对比的讨论展开，EPYC 配置可能超越 Mac Studio，尽管达到理论最大带宽具有挑战性。
   - 对话涉及了内存成本，一位成员开玩笑说与其花几千美元买硬件，不如去[下载更多 RAM](https://downloadmoreram.com/)。
- **Intel Crescent Island：具有 1.5TB/s 带宽的仅推理 GPU**：一位成员对 Intel 即将推出的 **Crescent Island** 表示兴奋，这是一款采用 **Xe3-LP** 架构、拥有 **160GB** 内存的仅推理 GPU，拥有 **1.5TB/s** 的带宽，称这是一个进步飞快的时代，*就像 2000 年的游戏界再次降临*。
   - 还有人提到，给它双倍内存应该是可能的，因为 Intel 使用的是 32Gb 芯片，而 LPDDR5x 最高可达 128Gb。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

teknium: 运行已完成，模型尚未发布。
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1428239348779974656)** (1 条消息): 

> `在语义层面运行的视觉模型, RAE: 用于 DiT 的表示自动编码器 (Representation Autoencoders), 视觉基础编码器作为 Diffusion Models 的分词器 (Tokenizers)` 


- **视觉模型进入 Nano Banana 时代**：关于在语义层面运行的视觉模型的讨论，以 **Google 的 Gemini Flash Image（又名 nano banana）** 为例，它利用 *vllm* 和 cross-attention 来理解多个参考图像潜变量（latents）中的视觉元素，详见[此推文](https://x.com/ditpoo/status/1970110646038548713)。
- **RAE 助力 DiT**：[此推文](https://x.com/sainingxie/status/1977936710135669130)提到了一篇关于 **用于 DiT 的表示自动编码器 (RAE)** 的新论文，并附带了[论文](http://arxiv.org/abs/2510.11690)和[博客](http://rae-dit.github.io)链接。
   - RAE 是一种用于改进 **Diffusion Transformers (DiT)** 的技术。
- **通过视觉基础编码器对齐 Token**：[此推文](https://x.com/bowei_chen_19/status/1973085809365405705)重点介绍了一篇关于使用 **视觉基础编码器** 作为 Diffusion Models 分词器（tokenizers）的论文。
   - 提供了[项目页面](https://aligntok.github.io)和 [Arxiv 论文](https://arxiv.org/pdf/2509.25162)链接。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1428116934028955740)** (89 messages🔥🔥): 

> `C2S-Scale 27B Gemma model, Dedalus Labs $11M seed, Amp Free with Ad-Supported Tier, Anthropic’s Revenue Soars to $7B, Clerk $50M Series C` 


- **Google 使用 Gemma 将细胞转化为句子**：Google AI Developers 发布了 **C2S-Scale 27B Gemma 模型**，该模型将单细胞基因表达数据标记化（tokenize）为 LLM 可读的“细胞句子”，并已在 [Hugging Face](https://huggingface.co/) 上线。
   - 在验证研究中，该模型提出了一个新假设（**silmitasertib 可增强肿瘤中的免疫信号**），实验分析显示**抗原增加了 50%**，这是一个潜在的新免疫疗法靶点。
- **Dedalus Labs 为“五行代码” AI Agents 融资 1100 万美元**：Dedalus Labs 获得了 **1100 万美元种子轮融资**，用于开发 **5 行代码 AI Agent**，引发了对其公司前景的积极反响。
   - 目前尚未提供关于该公司或产品的更多信息。
- **Amp 推出免费版并加入广告**：**Amp** 宣布了一种新的、可选的“免费”模式，通过展示*得体的广告*来覆盖 Token 成本，同时保持代码片段的私密性；如果用户愿意，可以继续使用现有的“智能”模式。
   - 讨论中既有关于隐私和模型质量的严肃提问，也有关于边写代码边看广告的调侃，以及对 Slack 避开流量的视频噱头和[过往推文](https://xcancel.com/sqs/status/1907300401352999030)的讨论。
- **Anthropic 的 CLI 表现出色，营收攀升至 70 亿美元**：尽管 GPT-5 和 Codex 已经发布，**Anthropic 的年化收入**已从 2025 年 1 月的 **10 亿美元**迅速攀升至 10 月中旬的 **70 亿美元**，这引发了观察者对 OpenAI 自身收入轨迹的质疑，并对 Anthropic 的持续增长感到惊叹。
   - 成员们一致认为，*他们的 CLI Agent 目前遥遥领先，这甚至不是开玩笑——至少在我使用它们的方式上是这样*。
- **Clerk 融资 5000 万美元以攻克 Agent 身份识别**：Clerk 完成了由 Menlo Ventures 和 Anthropic 的 Anthology Fund 领投、Georgian Capital 参投的 **5000 万美元 C 轮融资**，旨在为 AI 驱动的应用解决“Agent Identity（Agent 身份识别）”问题，同时扩展其身份验证、多租户和计费产品。
   - 该公告引发了开发者社区广泛的祝贺和功能需求。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1428544072125251635)** (1 messages): 

> `Local Workflows, M4 Max Setups` 


- **成员庆祝购入新 M4 Max，寻求工作流建议**：一位成员庆祝从旧的 Windows ThinkPad 换到了拥有 **128GB RAM 的 M4 Max**。
   - 他们热衷于探索本地工作流（local workflows）和配置，邀请社区提供建议和链接以供实验。
- **鼓励本地配置**：该成员专门寻求有关本地工作流和配置的建议，以发挥新机器的性能。
   - 这表明其关注点在于利用 M4 Max 的性能进行设备端处理和开发，而不是依赖云端解决方案。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1428567264403390504)** (8 messages🔥): 

> `AI Granny, OpenAI Sora MLK` 


- **AI 奶奶“拜金女”人设吸粉 200 万**：一个名为 **grannyspills** 的完全由 AI 生成的网络红人于 7 月上线，她是一个言辞犀利、追求金钱的奶奶，专门提供毒舌约会建议，目前 Instagram 粉丝即将突破 **200 万**，详情见[这条 X 帖子](https://x.com/venturetwins/status/1978852719335985309)。
   - 关于观众是否在意她是虚构人物的争论正在升温，一些人称赞这个讽刺性角色，另一些人则担心 AI 对文化的影响；一名用户声称创作者就住在他们的大楼里。
- **OpenAI 在 Sora 中屏蔽马丁·路德·金（MLK）**：在收到关于 **马丁·路德·金博士** 的 AI 生成视频片段不尊重死者的投诉后，OpenAI 已暂停任何描绘金博士的 **Sora** 输出，同时增加新的护栏（guardrails），根据[这条 X 帖子](https://x.com/OpenAINewsroom/status/1979005850166648933)。
   - 大多数用户批评此举是滑坡效应式的妥协，使公众人物私有化，并可能招致无休止的下架要求，此前 **金博士遗产委员会（King Estate）** 要求禁止使用这位历史人物的肖像。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1428095494911164486)** (70 messages🔥🔥): 

> `DSPy agentic search, Claude Code's search capabilities, RLM in OpenAI, Groq issues in OpenRouter` 


- **DSPy 社区辩论 “Agentic Search” 的定义**：成员们讨论了在 DSPy 中实现 “Agentic Search”，其中一位建议通过传递一个名为 `hybrid_search` 的函数，DSPy 就可以实现 Agentic Search。
   - 其他人则认为 “Agentic Search” 只是一个营销术语，本质上是模型使用工具来实现目标。一位成员指出：*这个领域有太多的营销和人为的复杂性，真的很令人气愤*。
- **Claude Code Agentic Search 的实现**：一位成员在对语义搜索感到不满后，试图在 DSPy 中复制 **Claude Code 的 Agentic Search**。
   - 他们发现 **Claude Code** 使用 *ripgrep* 在语料库中进行术语搜索，在将上下文添加到 LLM 之前先筛选文档，并分享了 [Agentic Search for Dummies](https://benanderson.work/blog/agentic-search-for-dummies/)。
- **OpenAI 关于 RLM 的传闻**：一位用户发布了一张截图，似乎暗示 OpenAI 内部使用了 **RLM (Recurrent Language Model)**。
   - 这暗示了 *RLM* 可能正在影响 OpenAI 的更新，引发了社区的兴趣和猜测，并分享了 [Alex Zhang 的 RLM 博客文章](https://alexzhang13.github.io/blog/2025/rlm/)。
- **OpenRouter 中的 Groq 故障？**：一位用户报告了在 **OpenRouter** 中使用 **Groq** 时遇到的问题，即使将其配置为唯一供应商也是如此，并发布了 [OpenRouter 设置页面](https://discord.com/channels/1161519468141355160/1161519469319946286/1428621175562440794)的截图。
   - 尽管进行了指定配置，系统仍默认使用其他供应商而非 **Groq**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1428132127710384201)** (18 messages🔥): 

> `vllm profiler errors, NVIDIA architecture, CUDA-Q support, distributed GPU talks, torchax/ops/jaten.py` 


- **VLLM Profiler 因权限不足而受阻**：一位用户在租用的 GPU 上运行 **vllm 的 profiler** 时遇到了 `CUPTI_ERROR_NOT_INITIALIZED` 错误，并被供应商告知他们不允许进行内核级操作。
   - 另一位成员建议使用 `sudo` 来更改 **NVIDIA** 的 profiling 限制，但该用户在租用的机器上没有 sudo 权限；他们正在寻找可以租用单个 GPU 进行 profiling 的地方。
- **适用于 Jetson Nano 的 Maxwell 酷炫反汇编器**：一位正在准备关于 **NVIDIA 架构** 演示的成员得到建议：虽然 **Blackwell** 是最前沿的，但 **Maxwell** 拥有酷炫的反汇编器，并且被用于第一代 **Jetson Nano**，更适合在受限条件下工作。
   - 该用户已经决定选择 **Hopper**，因为它支持 **CUDA-Q**，并且在 **AI** 和量子计算方面表现稳健。
- **Hopper Hacking 助力 DeepSeek**：一个都市传说描述了 DeepSeek 如何利用 **PTX/SASS** 指令来处理内存带宽问题，从而用较少的资源构建了强大的模型。
   - 在美国将 **H100** 限制为 **H800**，随后进一步限制 GPU 之后，中国只能合法获得 **H20**，但人们依然在用它们大显身手。
- **GPU 演讲已发布**：一位成员询问在哪里可以找到 **GPU Mode** 的分布式 GPU 演讲。
   - 另一位成员分享了 [GPU MODE YouTube 频道](https://www.youtube.com/@GPUMODE/videos)的链接。
- **对 Google 的 Jaten Op 扩展系统的评价**：成员们讨论了 [Google 的 torchax/ops/jaten.py](https://github.com/google/torchax/blob/main/torchax/ops/jaten.py)，其中一人对 **算子注册扩展系统 (op registration extension systems)** 的易用性表示惊讶。
   - 另一位成员对必须重新实现那些奇怪的特殊函数的人表示同情，指的是 [文件中的第 4654-4776 行](https://github.com/google/torchax/blob/c3d6ee322ad864eac0e1d3f557d459628e09819d/torchax/ops/jaten.py#L4654-L4776)。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1428474388226773203)** (2 messages): 

> `Distributed Triton, Non-ML Kernels with Triton DSL` 


- **寻找分布式 Triton 的尖端工具**：一位成员询问了目前可用于分布式 **Triton** 编程的最先进工具。
   - 他们正在寻求关于最佳资源和框架的建议，以促进 **Triton** 代码在多个设备或节点上的分发和执行。
- **在 Triton 中使用 Stencils？**：一位成员询问是否可以使用 **Triton DSL** 编写非机器学习内核，例如 **Stencils**。
   - 这个问题探讨了 **Triton** 在机器学习工作负载之外的通用性，以及它在更通用的计算任务中的适用性。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1428125508809523220)** (5 messages): 

> `NCU Timeline View, TMA Multicast Bandwidth` 


- **异步 CUDA Kernel 期望 NCU 时间线视图**：一位成员表达了对 **NCU** (NVIDIA Compute Unified Device Architecture) 中时间线视图的强烈需求，以便更好地分析异步、流水线化、持久化的 **CUDA kernels**。
   - 他们指出 **Pallas** 和 **Proton profiler** 等其他工具具有类似功能，并想知道 **NCU** 是否可以通过检测来支持此功能，例如利用空闲的共享内存 (**smem**) 来存储时钟数据。
- **TMA 多播带宽缩放问题**：有人提出了关于 **TMA (Thread Memory Accelerator) 多播**在将相同部分加载到不同 block 时带宽缩放的问题。
   - 该成员询问带宽是否与 cluster 中的 **CTAs (Cooperative Thread Arrays)** 数量成正比，或者 **TMA 多播**是否主要用于提高缓存命中率；**cutensormapL2Promotion** 标识符可能与之相关，因为加载到 L2 *可能* 是一种提升（promotion）。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1428487088323428474)** (3 messages): 

> `Free-Threading in PyTorch, Accessing Backward Functions, Custom Backward Kernels` 


- **PyTorch 开启 Free-Threading**：根据[这篇博文](https://trent.me/articles/pytorch-and-python-free-threading/)，**PyTorch** 正在实现 **PyTorch** 模型的多线程并行推理。
- **GELU 仅公开 Forward API**：一位成员询问如何在不使用 autograd 的情况下访问 backward 函数，特别是为了在自定义 backward 的融合 kernel 中使用 autograd 可以访问的 kernel。
   - 他们注意到 **GELU** 仅公开了 forward API。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1428224093337161789)** (2 messages): 

> `PMPP for AMD HPC hardware, CUDA and AMD` 


- **PMPP 适用于 AMD HPC 硬件**：一位成员询问 **PMPP** (Parallel and Multiprocessing Programming) 是否足够通用，可以用于入门 **HPC** 领域的 **AMD 硬件**。
   - 另一位成员回答说这应该足够了，并暗示 **PMPP** 并非过度针对 Nvidia。
- **CUDA 与 AMD 编程**：讨论围绕通用并行编程 (**PMPP**) 是否过于关注 **Nvidia 特性**而非适用于 **AMD 硬件**展开。
   - 共识似乎是，即使最终目标是在 **HPC** (High-Performance Computing) 领域使用 **AMD**，**PMPP** 也提供了足够的理论基础。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

shvdyside: 有人在伦敦吗？
  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1428148988661469204)** (1 messages): 

> `Intel Rubin CPX, Software-Level Block Floats, 640-bit vs 1280-bit bus, CXL-capable Intel` 


- **Rubin CPX 支持奇特的数据格式！**：一位成员提到，**Intel** 非常擅长的一点是支持各种奇特的数据格式，这使得**软件级 block floats** 更容易实现。
   - 他们补充说，如果 block floats 带来的计算开销并不大，*你可以尝试很多有趣的事情*。
- **海量内存带宽！**：成员们正在讨论 **640-bit** 与 **1280-bit 总线**，其中一人倾向于 **1280-bit** 以获得 **1.5 TB/s** 的内存带宽。
   - 他们补充说，如果 **Intel** 使其支持 **CXL**，它极具竞争力，因为 **CXL** 将大幅降低 CPU 与其通信并驱动它的成本。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1428346215237685278)** (1 messages): 

> `Alignment Evals Hackathon, Red Teaming Evals, Interp based Evals for LLMs` 


- **黑客松聚焦 Alignment Evals**：一场针对 **red teaming evals** 并旨在构建更稳健评估体系的 Alignment Evals 黑客松定于 11 月 1 日举行：[luma.com/h3hk7pvc](https://luma.com/h3hk7pvc)。
   - 之前黑客松的一个团队[在 ICML 展示了他们的工作](https://www.linkedin.com/feed/update/urn:li:activity:7352097355392786432/)。
- **基于 Interp 的 Evals 亮相**：一个团队在 1 月份创建了[首批基于 Interp 的 LLM Evals 之一](https://github.com/gpiat/AIAE-AbliterationBench/)。
   - 该项目专注于 **AIAE AbliterationBench**。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1428313346305953862)** (7 messages): 

> `GH200 编译问题，H100 Attention Kernel 错误，ThunderKittens 中损坏的 Kernel，ThunderKittens 开发协助` 


- **GH200 编译 H100 Attention Benchmark 的困扰**：一名成员在尝试编译并运行 **H100** attention benchmark 时，在 **GH200** 机器上遇到了编译问题，并引用了[之前的 issue](https://github.com/HazyResearch/ThunderKittens/issues/150) 中的类似问题。
   - 他们进行了诸如添加显式转换、添加 `warp::` 前缀以及禁用 causal 模式等更改，但现在触发了 *"unspecified launch error"*。
- **寻求 H100 Attention 的 TK Kernel 帮助**：一名成员请求协助处理 **H100** attention kernel 的现状，向一位一直活跃在该项目中的用户寻求帮助。
   - 另一名成员做出了回应，指出他们计划修复损坏的 kernel，但目前很忙，并提议通过私信（DM）分享他们个人的、可运行的 **H100** attention 前向（forward）实现。
- **ThunderKittens 的 ROCm 版本即将发布**：一名成员提到他们正在与 **AMD** 合作开发适用于 **ROCm** 的新版 **ThunderKittens**，暗示即将发布。
   - 他们补充说，正与 **AMD** 合作开发适用于 **ROCm** 的新版 **ThunderKittens**！
- **社区着手解决 Kernel 故障**：一名用户提议协助 **ThunderKittens** 的开发，建议从最新的更新中同步相关更改。
   - 作为回应，另一名用户分享了他们编译 **H100** kernel 的尝试，并提供了[他们的 commit 链接](https://github.com/aehmttw/ThunderKittens/commits/main/)，这些 commit 修复了类型转换、移除了 causal attention 并添加了 `warp::` 前缀，这至少使编译得以通过，但导致了运行时错误。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/)** (1 messages): 

notiurii: <@&1231246776103604326> 或者别的什么，我也不太清楚
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

noddybear: https://www.anthropic.com/news/skills
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1428620276354125825)** (1 messages): 

> `识别 Discord 用户` 


- **启动 Discord 用户搜索**：一名用户请求帮助识别两名 Discord 用户 **anuragj0803** 和 **meem**，并请认识他们的人发送私信（DM）。
   - 该用户在消息中附带了一张[图片](https://cdn.discordapp.com/attachments/1359640791525490768/1428620276316246067/image.png?ex=68f329d0&is=68f1d850&hm=089ab0801595780d085e976cd94ed235af2db60eb047632c9cbe35a036e8ead&)，推测是为了提供上下文。
- **额外的用户搜索**：扩大对 Discord 用户 **anuragj0803** 和 **meem** 的搜索范围。
   - 该用户正积极寻求帮助，以便在社区中识别这些个人。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1428347733034663939)** (1 messages): 

> `CuTe 库，Volta 架构，HMMA NT 操作，矩阵乘法，累加器映射` 


- **对 CuTe 库 MMA 操作的困惑**：一名用户在理解 **CuTe** 中单个 **MMA** 操作的工作原理时遇到困难，特别是文档中针对 **Volta** 架构的 **8x8x4 HMMA NT** 操作示例。
   - 用户对矩阵 **A** 和 **B** 的输入布局如何与 **T0** 中的结果累加相关联感到困惑，并请求进一步的资源。
- **A 和 B 布局 vs 累加器映射**：用户正在寻求澄清，即 **A** 和 **B** 布局是否用于矩阵乘法本身，而 **C** 布局是否用于与 **D** 的累加。
   - 用户试图理解文档中描述的“**A 和 B 布局映射**”与“**累加器映射**”之间的关系。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1428377356548051087)** (8 条消息🔥): 

> `Software 1.0 vs 2.0 优化、Torch.compile 局限性、Flash Attention API、避开编译器的原因、算法重写` 


- **Software 2.0：手动优化 vs. Torch.compile**：讨论集中在性能工程师必须进行的哪些优化是 **torch.compile()** 无法处理的，并探讨何时使用更底层的工具，如 **Helion**、**Triton**、**Gluon**、**CUDA** 和 **PTX**。
- **Torch 的显式 Flash Attention API 哲学**：Tensor 前端的 **显式 flash attention/flex attention API** 被强调为 **Torch** 哲学的一部分，这与 **Tinygrad** 和 **Luminal** 等基于搜索的发现方法形成对比。
- **避开编译器拥抱的原因**：避免使用编译器的原因包括不可接受的 **JIT 开销**、数值偏差、特定算子的保证融合、使用最前沿硬件、缺乏硬件自动调优以及算法重写的必要性。
   - 在编译器之外，考虑因素包括 **batch sizes**、**浮点精度**、**异步 IO**、**checkpoints**、更高效的优化器以及分片 (sharding)。
- **算法重写是终极优化**：最重要的优化可能是 **算法重写** 而非融合，这意味着需要解决编译器级别增强之外的基础计算方法。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1428396010975006790)** (3 条消息): 

> `AMD 分布式 Kernel 挑战赛、未来的比赛` 


- **AMD Kernel 挑战赛受到好评**：成员们感谢了 **AMD 分布式 Kernel 挑战赛** 的贡献者。
- **更多比赛即将到来**：一名成员回应称，很快就会有更多比赛。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1428169904686633081)** (8 条消息🔥): 

> `RTX 6000 Pro、Blackwell、第五代 Tensor Core 堆栈、对称内存编程、多 GPU Kernel` 


- **Blackwell 热议：RTX 6000 Pro 宣称“正统”地位**：一名成员声称 *只有* **RTX 6000 Pro** 拥有 *真正的 Blackwell 套件*，引发了关于什么是 *真正的 Blackwell* 的辩论。
   - 该成员澄清说，**RTX 5090** 和其他消费级套件 *没有第五代 Tensor Core 堆栈*。
- **Tensor Core 之争：RTX 6000 Pro 加入战局**：随后进行了澄清，指出 **RTX 5090** 和 **RTX 6000 Pro** 都不具备 `tcgen05` 或 *第五代 Tensor Core 堆栈*。
   - 这两款显卡都是 `sm_120` 架构，并使用相同的芯片。
- **对称内存交响乐：PyTorch 2.9 协调多 GPU Kernel**：一名成员链接到了 [PyTorch 2.9 博客文章](https://pytorch.org/blog/pytorch-2-9/)，该文章介绍了用于编写跨 NVLinks 和 RDMA 网络工作的多 GPU Kernel 的 **PyTorch Symmetric Memory**。
   - 这开启了诸如 **Kernel 内通信 (in-kernel communication)**、**超低延迟远程访问** 以及 **自定义通信模式** 等机会。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1428119371309383813)** (3 条消息): 

> `Lotion 论文、FP8、QAT、torchao` 


- **Lotion 论文：QAT 的替代品？**：一名成员询问了 [Lotion 论文](https://arxiv.org/pdf/2510.08757)，以及它是否可以替代 **FP8** 的 **QAT**（他们认为 QAT 由 **torchao** 处理）。
- **论文链接**：成员发布了论文链接 [https://arxiv.org/pdf/2510.08757](https://arxiv.org/pdf/2510.08757)


  

---


### **GPU MODE ▷ #[llmq](https://discord.com/channels/1189498204333543425/1421956177549332662/1428177736777732176)** (2 条消息): 

> `llmq、单元测试、Kernel` 


- **以单元测试作为 llmq 的入门方式**：一名成员询问编写 **单元测试 (unit tests)** 是否是熟悉 **llmq** 仓库并进行贡献的好方法。
   - 另一名成员回应说，虽然单元测试可能无助于理解整体架构，但对于熟悉特定的 **Kernel** 很有意义。
- **针对特定 Kernel 的单元测试**：讨论表明，专注于 **llmq** 仓库中特定 **Kernel** 的 **单元测试** 可能是一种实用的方法。
   - 这种策略允许贡献者深入研究特定组件，而无需全面了解整个代码库。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1428562254198079511)** (2 messages): 

> `Google Coral NPU, Verilog Open Source, Apache 2 License, RV32 cores, Mojo Portability` 


- **Google 开源 Coral NPU Verilog！**: Google 已在 **Apache 2** 许可证下开源了 **NPU block** 的 Verilog 代码，可在 [GitHub](https://github.com/google-coral/coralnpu) 上找到。
- **Coral 的核心：类似 AMD，但是 RV32**: 矩阵核心本身看起来有点像 **AMD 的 NPU**，但它们是 **RV32 核心**。
- **Coral NPU：新的 Mojo 可移植性测试平台？**: 开源的 NPU 作为测试 **Mojo 可移植性** 的平台可能非常有趣，因为应该可以在客户端硬件上对其进行模拟。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1428204016558538759)** (35 messages🔥): 

> `Mojo Standard Library, __type_of() Semantics, Mojo for Game Development, Mojo TUI Frameworks, LayoutTensor` 


- **解码 Mojo 标准库**: 一位标准库贡献者分享了关于语言内部机制的见解，并警告不要过度使用 `__type_of()`，因为其语义和语法正在演变；团队已决定去掉 `__` 前缀，因此这些很快将被命名为 `type_of(x)` 和 `origin_of(x)`。
   - 当被问及从哪里了解到这些时，该贡献者回答道：*我是标准库贡献者，所以我知道很多关于这门语言的古怪细节*。
- **`__type_of()` Bug 追踪引发辩论**: 一位用户报告了在使用 `__type_of()` 时出现未使用的变量警告，引发了关于这是 Bug 还是预期行为的讨论。团队实际上在昨天决定去掉 `__` 前缀，所以这些很快将命名为 `type_of(x)` 和 `origin_of(x)`，我认为很快就会进入 Nightly 版本。
   - 讨论中澄清了 `__type_of(a)` 使用了名称 "a" 但没有使用其值，因为它需要在编译时知道返回类型，而 "a" 直到运行时才存在；要消除该错误，可以将变量命名为 `_a`。
- **Mojo 关注游戏开发领域的扩张**: 用户探讨了 Mojo 在 AI 之外的潜力，特别是在游戏开发方面，注意到它具有类似 Python 的易用性和系统级语言的性能，且 Mojo 的目标是成为一种通用编程语言。
   - 爱好者们指出，现有的项目如 [Stargine](https://forum.modular.com/t/stargine-a-game-engine-in-mojo/2266/3)（论坛项目）和 raylib 绑定是很有前景的起点，而另一位用户则对 Textual 移植以及完整的音频/MIDI 功能表示感兴趣。
- **Mojo 的 TUI 未来初具规模**: 社区讨论了在 Mojo 中开发 TUI 框架的可能性，从 Textual、ELM 应用（如 Go 语言中的 BubbleTea）等项目中汲取灵感，并分享了相关仓库的链接，如 [ui-terminal-mojo](https://github.com/rd4com/ui-terminal-mojo)。
   - 一位用户分享了他们正在开发中的受 ELM 启发的 TUI 框架 [banjo](https://github.com/thatstoasty/banjo/blob/main/examples/multi.mojo)，目前暂停了开发以等待 Mojo 进一步成熟。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1428507082268737698)** (1 messages): 

> `MAX Python API, Open Sourcing` 


- **Modular 开源 MAX Python API**: Modular AI 已经**开源**了 **MAX Python API** 的剩余部分。
   - 列出所有新开源 Python 模块的论坛帖子可以在[这里](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379)找到。
- **MAX Python API 现已开放！**: **MAX Python API** 现在完全开源，允许更多的社区贡献和访问。
   - 访问 [Modular.com](https://forum.modular.com/t/open-sourcing-all-of-the-max-python-api/2379) 上的论坛帖子，查看新可用模块的完整列表。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1428137209252417678)** (36 条消息🔥): 

> `Jotform 替代方案, Beta 计划, Manus 1.5, 账户支持, 部署错误` 


- **用户吐槽 Jotform 联系帮助**：一位用户对 **Manus AI** 的联系帮助表示不满，建议使用下一代 **AI Agent** 而不是*简单的表单*来获取详细信息并通过电子邮件回复，并建议**创新订阅**可以包含用户反馈的积分。
   - 他们抱怨响应时间过长，强调*由于服务标准的原因，用户需要的是某种服务，而不是 3 天后的回复*。
- **用户渴望 Beta 测试扩展的 Agent 功能**：多位用户询问如何加入扩展 **Manus Agent** 功能的 **Beta 计划**，但被告知目前只有 **Manus Fellows** 拥有访问权限。
   - 一位用户兴奋地宣称 **Manus Agent** *解决了我所有的问题，它是我用过的最好的工具！*
- **用户被锁定，寻求账户支持**：一位用户报告由于**手机验证问题**被锁定在账户之外，且帮助中心没有提供有效帮助。
   - 一位社区成员提供了协助，并要求提供两个账户的电子邮件地址以解决问题，称*在这里可能可以获得帮助*。
- **OpenAI 依赖导致部署混乱**：一位用户报告部署失败，原因是 **OpenAI 需要 pydantic_core**，而该库需要编译，但 **Manus 部署环境不支持**。
   - 一位成员提议通过使用预配置的 **OPENAI_API_KEY** 环境变量和更简单的 **HTTP client**，创建一个不需要 **OpenAI 依赖**即可运行的版本。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1428287920816525343)** (17 条消息🔥): 

> `Kimi K2 微调, unsloth` 


- **用户寻求 Kimi K2 性格蒸馏**：成员们讨论了在 **10 万个 Kimi K2** 回答样本上微调 **Qwen 3-4B** 基础模型以蒸馏其性格的潜力。
   - 一位成员指出了可能性，但注意到目前没有任何现有模型，而另一位成员建议现在使用 [Unsloth](https://github.com/unslothai/unsloth) 等工具微调 **LLM** 变得更加容易。
- **昂贵的 API 阻碍了 Kimi K2 微调**：一位成员表示有兴趣微调 **1B** 版本的 **Kimi K2**，但另一位成员指出，获取 **10 万个样本** 的 **API** 访问费用太高。
   - 他们建议在过滤后仅使用 **1 万个样本** 或更少。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1428207732363886743)** (8 条消息🔥): 

> `Tinygrad 破坏性变更, Tinybox 保修, Shapetracker 弃用, tinygrad 统计网站, Tinygrad 默认设备设置` 


- **Tinygrad 的破坏性变更令用户沮丧**：一位偶尔使用 **tinygrad** 的用户对频繁的破坏性变更表示沮丧，这些变更在尝试发布新的 openpilot 模型时需要进行提交二分查找（commit bisection），并且在各种配置中会出现难以理解的错误。
   - 他们指出了管理频繁更改的环境变量的挑战，并认为针对 **845** 的 **IMAGE hacks** 导致了不稳定性。
- **Tinybox 的保修状态受到质疑**：在 **Shapetracker** 被弃用后，一位用户询问了 **Tinybox** 的保修情况，Di Zhu 在一份报告中对此进行了说明。
   - 另一位用户提供了 [Di Zhu 关于 Shapetracker 的报告](https://github/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md)的链接。
- **用户建议为不支持的配置提供明确的失败消息**：一位用户建议，当遇到不支持的设备/硬件组合时，特别是涉及 **IMAGE=**、**NOLOCALS=** 和 **GRAPH_ONE_KERNEL=** 时，**tinygrad** 应该提供明确的失败消息。
   - 该用户表示难以区分真实的编译失败和错误的配置，这阻碍了他们的调试过程。
- **请求在 Python 中设置默认设备**：一位用户询问是否可以在 Python 中设置默认设备，类似于 **Device.set_default('CL')**，以便在 Python 脚本中方便地交叉检查不同的 **backends**。
   - 有人指出 **Device.DEFAULT = "CL"** 可以实现该结果。
- **Tinygrad 统计网站数据可用性**：一位用户提到 [tinygrad 统计网站](https://stats.tinygrad.win/) 似乎只有过去 25 天的数据，这使得评估长期性能趋势变得困难。
   - 他们表示有兴趣为失败的编译编写测试，但发现由于失败与模型架构、设备和 **FP16** 配置具有高度相关性，这具有挑战性。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1428402139310330009)** (3 messages): 

> `aider benchmark, sonnet 4.5, haiku 4.5, gpt-5 pro, openrouter/x-ai/grok-code-fast-1` 


- **Aider Benchmark 位置引发讨论**：一位用户询问了针对 **Sonnet 4.5**、**Haiku 4.5** 和 **GPT-5 Pro** 等模型的 aider 基准测试结果位置。
   - 另一位用户回答说，*没有人愿意为这些模型付费测试*。
- **Grok 模型在 Aider CLI 中快速启动**：一位用户建议了一种在 aider CLI 中快速开始使用 **Grok** 模型的方法。
   - 他们提供了命令 `aider --model=openrouter/x-ai/grok-code-fast-1 --edit-format diff` 以快速入门。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1428111328563695787)** (3 messages): 

> `Ollama Qwen 2.5 Coder 7b metadata, Ollama file integration, Ollama model issues` 


- **用户在 Ollama 中获取 Qwen 2.5 Coder 7b 元数据遇到困难**：一位用户报告称，无法从 [ollama.com](https://ollama.com) 为 **Qwen 2.5 Coder 7b** 模型检索到有效的 `metadata.json` 文件。
   - 用户表示该模型输出乱码，并专门请求一个 `metadata.json` 示例来解决问题。
- **Ollama 文件集成遇到困难**：一位用户在使用 `filename` 进行 Ollama 模型文件集成时遇到问题。
   - 用户希望聊天机器人在消息后自动合并指定文件（例如 `SomeFile.txt`）的内容，但未能按预期工作。
- **用户发现一个 Ollama 模型存在问题**：出于某种原因，用户无法让其中一个模型正常工作。
   - 用户拥有许多 Ollama 模型，但只有 **Qwen 2.5 Coder 7b** 给他带来了麻烦。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1428155249922936934)** (4 messages): 

> `Model Context Protocol (MCP), MCP Feature Support Matrix, MCP Tool Discovery, MCP Grouping SEP, MCP Schema Enhancement` 


- **MCP 功能矩阵得到澄清**：一位成员就 [Model Context Protocol 的功能支持矩阵](https://modelcontextprotocol.io/clients#feature-support-matrix)中“Discovery”的含义寻求澄清。
   - 它指的是*对响应 tools/list_changed 通知并查找新工具的支持*。
- **MCP 关注工具分组 SEP**：来自过去审查的反馈建议制定一个分组 SEP，以支持所有 **MCP primitives**（Tools、Prompts 和 Resources）的分组，而不仅仅是工具组。
   - 关于层级分组的讨论正在 [GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1567#discussioncomment-14680104) 上进行。
- **贡献者提问：MCP 的下一步计划是什么？**：一位成员询问了为分组功能创建新 **SEP 文档**的后续步骤，包括社交化推广、反馈以及潜在的原型实现。
   - 鉴于下一个规范版本将于 2025 年 11 月发布，该成员旨在优先开展工作，以便将其纳入 **SEP 路线图**。


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1428151932668743721)** (2 messages): 

> `Claude Haiku 4.5, SWE-grep, SWE-grep-mini, Agentic Search` 


- **Haiku 登陆 Windsurf！**：**Claude Haiku 4.5** 现已在 Windsurf 中可用，消耗 **1x 额度**，其编程性能与 **Sonnet 4** 相当，但成本仅为三分之一，速度提升超过 2 倍。
   - 可以通过重新加载或从 [Windsurf on X](https://x.com/windsurf/status/1978512184343662707) 下载 Windsurf 来体验。
- **SWE-grep 席卷 Fast Context！**：专为快速 Agentic Search（>2800 TPS）设计的新模型系列 **SWE-grep** 和 **SWE-grep-mini** 正在通过 Fast Context 子代理逐步向 Windsurf 用户推出，为你的编程 Agent 寻找正确文件的速度比以前快 **20 倍**。
   - 可以在[新 Playground](https://playground.cognition.ai) 中试用，并加入 [Reddit](https://www.reddit.com/r/windsurf/comments/1o8bo77/fast_context_is_here_swegrep_and_swegrepmini/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 上的讨论；在 [Cognition 博客](https://cognition.ai/blog/swe-grep)和 [Cognition 的 X 帖子](https://x.com/cognition/status/1978867021669413252)中阅读更多信息。