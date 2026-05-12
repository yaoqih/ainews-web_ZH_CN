---
companies:
- thinking-machines
- openai
- anthropic
date: '2026-05-11T05:44:39.731046Z'
description: '**Thinking Machines** 预告了其全新的**原生交互模型**，旨在实现**全双工多模态交互**。该模型支持实时并发的听、说、看、想、搜索和反应，标志着
  AI 交互从“回合制”向实时互动的转变。这一方案强调对音频、视频和文本的持续处理，并引入了**视觉主动性**（visual proactivity）和后台工具调用等创新功能，基于
  **SGLang** 框架实现。


  与此同时，**OpenAI** 宣布成立 **OpenAI 部署公司**（OpenAI Deployment Company）。该新部门拥有 **150 名外派部署工程师**（Forward
  Deployed Engineers）及 **40 亿美元的初始投资**，旨在协助企业部署前沿模型，标志着 OpenAI 正式进军 AI 经济的“部署层”。此外，OpenAI
  还推出了名为 **Daybreak** 的安全专项计划，整合了 **GPT-5.5** 与 **Codex**，用于网络防御、威胁建模和自动补丁修复，并提供包括
  **GPT-5.5-Cyber** 在内的差异化访问等级。这与 Anthropic 较为谨慎限制的网络安全策略形成鲜明对比，也凸显了各大厂商在 AI 安全策略上的分歧与博弈。'
id: MjAyNS0x
models:
- gpt-5.5
- codex
people:
- johnschulman2
- soumithchintala
- chillee
- liliyu_lili
- rown
- kimmonismus
- giffmana
- swyx
- eliebakouch
- gdb
- sama
- therundownai
- lukolejnik
- matvelloso
title: 今天没发生什么特别的事。
topics:
- multimodality
- real-time-interaction
- visual-proactivity
- deployment
- cybersecurity
- threat-modeling
- automation
- continuous-audio-video-text-processing
- security-models
- field-engineering
- enterprise-ai
---

**平静的一天。**

> 2026/5/9-2026/5/11 的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步的 Discord 信息。[AINews 的网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入或退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 回顾


**Thinking Machines 的原生交互模型以及超越轮次制 AI 的转变**

- **全双工多模态交互作为模型的一等公民能力**：当天最明确的技术主题是 [Thinking Machines 对“交互模型”的预览](https://x.com/miramurati/status/2053939069890298321)，这些模型被描述为**从零开始（from scratch）**为实时交互而训练的，而不是将语音、轮次切换和工具使用叠加在轮次制的 LLM 之上。随附的[技术文章](https://x.com/thinkymachines/status/2053938892152435174)以及来自 [@johnschulman2](https://x.com/johnschulman2/status/2053940452789981426)、[@soumithchintala](https://x.com/soumithchintala/status/2053940215505645938) 和 [@cHHillee](https://x.com/cHHillee/status/2053940218747842619) 的团队评论将此框架化为一个**人类↔AI 带宽**问题：模型应该能够同时进行听、说、看、思考、搜索和反应。演示强调了连续时间感知、中断处理、同步语音、视觉主动性以及后台工具使用，而没有明确的“我现在正在思考 / 我现在正在搜索”的界限。团队成员还强调，一旦类型签名（type signature）有效地变为连续的 **audio+video+text → audio+text**，许多以前需要专用系统的任务都会变成零样本（zero-shot） ([@johnschulman2](https://x.com/johnschulman2/status/2053940940885332028))。
- **为什么这在技术上很重要**：几种反应都指向了同一个观点：这不仅仅是“又一个聊天机器人演示”，而是界面假设的改变。[@liliyu_lili](https://x.com/liliyu_lili/status/2053942465477197891) 指出**视觉主动性**（“当我开始驼背时提醒我”，“数一下我做了多少个俯卧撑”）是当前系统中缺失的原语；[@rown](https://x.com/rown/status/2053950123139575863) 称其为第一个具有视觉主动性的通用 **video+speech** 模型；[@kimmonismus](https://x.com/kimmonismus/status/2053952846064767384) 和 [@giffmana](https://x.com/giffmana/status/2053953584300003405) 都强调，原生交互性是比原始 Benchmark 结果更深层次的创新。正如 [@swyx](https://x.com/swyx/status/2053960011748098462) 所指出的，这次发布也含蓄地提高了“实时”多模态系统的门槛。[@eliebakouch](https://x.com/eliebakouch/status/2053982248253190180) 透露了一个实现细节：该技术栈正在使用 **SGLang**。

**OpenAI 的企业级与安全推动：部署公司与 Daybreak**

- **OpenAI 正在向栈下游的服务和部署领域迈进**：OpenAI 宣布成立 [OpenAI Deployment Company](https://x.com/OpenAI/status/2053824997777457651)，这是一个由其控股的部门，旨在帮助企业将前沿模型部署到实际工作流中。关键的运营细节在于，通过收购 [Tomoro](https://x.com/OpenAI/status/2053824999736410415)，带来了 **150 名前线部署工程师（Forward Deployed Engineers）和部署专家**，[@gdb](https://x.com/gdb/status/2053884619695730745) 提到其获得了 **来自 19 个合作伙伴的 40 亿美元初始投资**。多位观察家将其解读为 OpenAI 正在采用类似 Palantir 或 Microsoft 的现场工程模式：[@kimmonismus](https://x.com/kimmonismus/status/2053844403488194827) 认为 OpenAI 想要占据 AI 经济的**部署层（deployment layer）**，而 [@matvelloso](https://x.com/matvelloso/status/2053881988529139765) 则将其与历史上企业成功的模式联系起来，即让技术人员深入客户运营端。
- **Daybreak：特定于安全的模型分发、工作流和信任层级**：OpenAI 还发布了 [Daybreak](https://x.com/OpenAI/status/2053939702110269822)，这是一个围绕防御性网络运营和持续软件安全的综合项目，[@sama](https://x.com/sama/status/2053951874408276193) 将其定位为对快速提升的 AI 网络能力的务实响应。该产品的宣传点由 [@TheRundownAI](https://x.com/TheRundownAI/status/2053945340592631843) 总结，涵盖了 **GPT-5.5**、**Codex**、仓库威胁建模、漏洞发现、补丁生成和响应自动化，并提供差异化的访问层级，包括 **Trusted Access for Cyber** 和更专业的 **GPT-5.5-Cyber**。这与 Anthropic 更加受限的网络安全姿态形成对比，[@kimmonismus](https://x.com/kimmonismus/status/2053941490490265661) 捕捉到了这种紧张关系。对于构建安全 Agent 系统的团队，来自 [@lukOlejnik](https://x.com/lukOlejnik/status/2053758553723211988) 的另一个警告值得关注：**“你的 LLM 并不是一个安全边界”** —— 据报道，Microsoft Semantic Kernel 允许 prompt injection 转化为宿主机级别的 RCE，因为框架过度信任了模型输出，而非模型本身出现故障。

**Agent Harnesses、本地优先工具链与控制面**

- **更好的 Agent 控制面正在成为一个产品类别**：一个反复出现的抱怨是，有用的 Agent 需要自主性，但工程师仍然想要可逆、可检查的控制权。[@itsclelia](https://x.com/itsclelia/status/2053716807748567329) 通过 **aggit** 解决了这个问题，这是一个基于 Rust 的 CLI，用于本地/远程且支持 S3 存储的 Agent 产物管理，在主 Git 历史记录之外实现了暂存（stash）/分支（branch）/恢复（restore）语义。同样，[@_catwu](https://x.com/_catwu/status/2053999857799672111) 强调了一个新的 `claude agents` 终端控制面，用于管理多个 Claude Code Agent，而 [@cursor_ai](https://x.com/cursor_ai/status/2053939390410612988) 将 Cursor 推向了 **Microsoft Teams**，在该平台中，Agent 可以阅读完整会话并开启 PR。这些迹象都表明“Agent 编排（Agent orchestration）”正在向具体的 UX 模式汇聚，而不仅仅是靠 Prompt 技巧。
- **Deep Agents / Hermes / 本地 Agent 正在快速成熟**：[@masondrxy](https://x.com/masondrxy/status/2053717333433340034) 注意到 **Deep Agents CLI** 可以在 **对话中途热切换底层模型提供商而不丢失上下文（context）**，这是一种许多 Agent 栈仍然缺失的非平凡系统能力。LangChain 还强调了用于提供商/模型特定调优的 **harness profiles**（[推文](https://x.com/masondrxy/status/2053882188870074848)），同一作者的另一份价格分析认为，对于高吞吐量的 Agent 工作负载，**DeepSeek V4 Flash** 的成本可能比 GPT/Gemini 的 Flash 级别选项便宜得多（[推文](https://x.com/masondrxy/status/2053855842076942555)）。在本地端，Hugging Face 在本地应用中增加了 [Hermes Agent 支持以及原生追踪可视化](https://x.com/mervenoyann/status/2053857347429151163)，而 [@Teknium](https://x.com/Teknium/status/2053961675985113404) 预告了通过 Hermes Agent 和 CUA 实现 **适配任何模型的 computer use**，明确针对本地/开源模型以及前沿 API。[@onusoz](https://x.com/onusoz/status/2053812410730037256) 加入 Hugging Face 以改进 **OpenClaw** 及相关开源 Harness 中的本地模型，这是本地 Agent 易用性（ergonomics）现已成为战略基础设施的另一个强烈信号。
- **围绕工具出现的一种设计论点**：[@threepointone](https://x.com/threepointone/status/2053751241977594102) 认为，Agent 最终可能只需要 **两个原始工具：搜索与执行**，通过对能力的动态语义发现来工作，而不是不断扩展静态工具菜单。这与转向可配置 Harness 而非巨大单体 Prompt 的大趋势相得益彰。

**基准测试、效率与开源模型经济学**

- **Coding-agent 基准测试终于开始衡量“测试框架 (harness) + 模型”组合**：[Artificial Analysis 发布了 Coding Agent 指数](https://x.com/ArtificialAnlys/status/2053865095076438427)，涵盖了 SWE-Bench-Pro-Hard-AA、Terminal-Bench v2 和 SWE-Atlas-QnA，不仅对比了模型，还对比了**模型 + 测试框架的组合**。其核心结论：在 Cursor CLI 中运行的 **Opus 4.7** 得分为 **61**，在 Codex/Claude Code 中运行的 **GPT-5.5** 紧随其后；顶尖开源权重配置包括 **GLM-5.1**、**Kimi K2.6**，以及在 Claude Code 中运行的 **DeepSeek V4 Pro**，虽然仍具竞争力，但仍有明显差距。该基准测试还揭示了**单次任务成本**（>30倍）、**Token 使用量**（>3倍）、**缓存命中率**（80–96%）以及**单次任务耗时**（>7倍）的巨大差异。这一基准测试得到了 OpenHands 更新的软件工程基准测试公告（[推文](https://x.com/OpenHandsDev/status/2053839810343620980)）的补充，以及 Claw-Eval 在办公、金融、终端和网页任务中更具 Agent 特性的任务组合，其中 [MiMo-V2.5-Pro 处于领先地位，而 DeepSeek V4 Flash 在同尺寸模型中表现出极高的效率](https://x.com/nathanhabib1011/status/2053786853929824385)。
- **对 TurboQuant 的质疑正在增加**：多篇帖子对最近流行的量化/推理服务技术持更加冷静的态度。[@_EldarKurtic](https://x.com/_EldarKurtic/status/2053809592061030546) 展示了他所称的针对 **TurboQuant** 的首个全面研究，涵盖了准确率、延迟和吞吐量；[@vllm_project](https://x.com/vllm_project/status/2053852636093239555) 将 Red Hat / vLLM 的调查作为起点；而 [@jbhuang0604](https://x.com/jbhuang0604/status/2053882357833208262) 则直截了当地总结其要点为“它实际上效果并不好”。这正是需要独立复现的典型基础设施类主张。
- **本地/开源模型的进步速度持续超越硬件瓶颈**：[@ClementDelangue](https://x.com/ClementDelangue/status/2053825719587815711) 在此提出了最有力的宏观论点：在相同的顶配 MacBook Pro 内存上限下，“你能实际运行的最聪明开源权重模型”已从 Llama 3 70B 时代的能力提升至 **DeepSeek V4 Flash 混合 Q2 GGUF** 时代的能力，在 **24 个月内提升了约 4.7 倍**，这意味着每 **10.7 个月**翻一番，速度超过了摩尔定律。支持数据来自 [@victormustar](https://x.com/victormustar/status/2053780086596288781) 关于 GGUF 上传量的快速增长，以及社区的反复观察，即 **Qwen 3.6**、**Gemma 4** 和 DeepSeek 变体现在已可用于本地处理非琐碎的 Agent 任务。

**研究亮点：MoE 模块化、Diffusion/Byte 模型与 Agent 动态**

- **架构与评估**：AllenAI 的 **EMO** 被 [@TheTuringPost](https://x.com/TheTuringPost/status/2053795343658303860) 重点提及，这是一种更具模块化的 **Mixture-of-Experts** 设计，通过文档级路由引导共享专家池；值得注意的是，据报道仅保留 **25% 的专家** 时，性能损失仅约 **1%**，而标准 **MoE** 在类似剪枝下的性能下降则达 **10–15%** ([后续报道](https://x.com/TheTuringPost/status/2053795410490339720))。在生成式评估方面，[@qberthet](https://x.com/qberthet/status/2053795951228371311) 介绍了 **MIND (Monge Inception Distance)**，据称它是 **FID** 的一种速度更快、样本效率更高的替代方案。
- **用于语言和字节级建模的 Diffusion**：多篇论文推动了非自回归（non-AR）语言建模的发展。[@LucaAmb](https://x.com/LucaAmb/status/2053867347023466850) 报告称，在其评估设置下，连续比特流扩散（continuous bitstream diffusion）几乎能与自回归模型相媲美；[@JulieKallini](https://x.com/JulieKallini/status/2053853543552217478) 介绍了 **Fast BLT**，利用扩散进行并行字节解码，以减轻字节级 **LLM** 在推理时的限制；[@sriniiyer88](https://x.com/sriniiyer88/status/2053882384211419375) 将其描述为块字节扩散与 **self-speculative decoding** 的结合。与之相关的是，[@LiangZheng_06](https://x.com/LiangZheng_06/status/2053806963839168619) 指出了 **Diffusion** 模型在 **post-training** 中的一个实用特性：由于采样是可微的，奖励梯度在原则上可以比标准 **LLM** 设置更直接地流向参数。
- **长跨度下的 Agent 行为**：出现了两个有力的实证研究方向。首先，[“The Memory Curse”](https://x.com/omarsar0/status/2053863994499408214) 声称，长历史记录会降低多轮社会困境中的协作性，因为模型会变得更加 **倾向于遵循历史且规避风险**，而显式的 **CoT** 有时会放大这一问题。其次，[由 @dair_ai 总结的 PwC 研究](https://x.com/dair_ai/status/2053866106151182419) 认为，澄清（clarification）的价值具有高度的时间依赖性：**目标澄清在执行约 10% 后就会失去大部分价值**，而输入澄清的有效性持续时间更长。这些研究共同表明，长跨度 **Agent** 的质量既受限于原始模型的智商，也受限于记忆/控制策略。
- **Scaling 与自我提升**：Marin 的 **Delphi** 扩展性研究（由 [@WilliamBarrHeld](https://x.com/WilliamBarrHeld/status/2053919463880462453) 总结）声称，从小型预训练推断到 **25B / 600B token** 的运行规模时，预测误差仅为 **0.2%**。另外，[@omarsar0](https://x.com/omarsar0/status/2053978221193130434) 强调了 **AutoTTS**，其中 **LLM** 自行搜索测试时缩放控制器（test-time scaling controller）空间，据报道其以约 **$39.9** 的探索成本击败了手工设计的策略。

**热门推文（按参与度排序）**

- **OpenAI 的企业/服务动向**：[OpenAI 启动部署公司 (Deployment Company)](https://x.com/OpenAI/status/2053824997777457651) 以及 [收购 Tomoro / 150 名 FDEs](https://x.com/OpenAI/status/2053824999736410415)。
- **OpenAI 的安全产品化**：[Daybreak 发布公告](https://x.com/OpenAI/status/2053939702110269822) 以及 [@sama 的定位描述](https://x.com/sama/status/2053951874408276193)。
- **Thinking Machines 的交互模型**：[Mira Murati 的发布推文](https://x.com/miramurati/status/2053939069890298321) 和 [技术预览推文串](https://x.com/thinkymachines/status/2053938892152435174)。
- **Artificial Analysis 代码 Agent 指数**：[基准测试发布及主要发现](https://x.com/ArtificialAnlys/status/2053865095076438427)。
- **Agent 工具 / 开发者工作流**：[Hermes Agent 可在任何模型上使用的计算机控制功能](https://x.com/Teknium/status/2053961675985113404)，[Microsoft Teams 中的 Cursor](https://x.com/cursor_ai/status/2053939390410612988)，以及 [Codex OpenAI Developers 插件](https://x.com/OpenAIDevs/status/2053925962287583379)。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 3.6 本地推理进展

- **[Unsloth 上的 MTP](https://www.reddit.com/r/LocalLLaMA/comments/1ta4rvs/mtp_on_unsloth/)** (活跃度: 620): **图片 ([链接](https://i.redd.it/7qopol51pi0h1.png)) 显示 **Unsloth 的 Hugging Face 个人资料** 列出了新发布的保留 MTP 的 GGUF 构建版本：[`unsloth/Qwen3.6-27B-GGUF-MTP`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF-MTP) 和 [`unsloth/Qwen3.6-35B-A3B-GGUF-MTP`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF-MTP)。该帖子的技术意义在于这些 GGUF 保留了 **MTP / next-token prediction 层**，但用户仍需构建特定的 **llama.cpp MTP PR**，而不是依赖标准的 llama.cpp 支持。一位评论者报告了 27B GGUF 的运行时/断言失败：`GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35_MTP requires nextn_predict_layers > 0")`，这表明元数据解析、模型转换或 PR 兼容性问题仍未解决。** 评论反映了对上游 llama.cpp MTP 支持的期待，用户不断查看 GitHub 仓库并询问 MTP 是否已实现“开箱即用”的支持。

    - 一位编译新 `27B` GGUF 模型的用户在 `qwen35_mtp.cpp` 中遇到了运行时断言错误：`GGML_ASSERT(hparams.nextn_predict_layers > 0 && "QWEN35_MTP requires nextn_predict_layers > 0")`。这表明 GGUF/模型元数据或转换路径可能缺失 `nextn_predict_layers`，而这是 Qwen3.5 MTP 投机/next-token prediction 层所必需的。
    - 一个技术讨论线程指出，**GGUF 中的 MTP 支持**对于本地推理非常重要，尤其是对于 `35B A3B` 变体，评论者认为该变体改进了上下文长度处理。另一位评论者询问这是否意味着 `llama.cpp` 现在“开箱即用”地支持 MTP，暗示了对该支持是已合并/稳定，还是仅在 PR 或分支中可用的不确定性。
    - 一位评论者声称 **`ik_llama` MTP 目前比 `llama.cpp` PR 更快**，并补充说它支持基于 Hadamard 的量化，被描述为类似于 “turboquants”。对于比较本地 MTP 推理后端的用户来说，这是一个潜在相关的实现/性能差异。

  - **[Qwen 3.6 35B A3B 的热度名副其实！！！](https://www.reddit.com/r/LocalLLaMA/comments/1t9whrt/the_qwen_36_35b_a3b_hype_is_real/)** (活跃度: 586): **该帖子报告了一项定性的代码理解评估，其中几个小型/本地长上下文开源权重模型——**Qwen 3.6 35B A3B**、**Qwen 3.6 27B**、**Gemma 4 26B A4B** 和 **Nemotron 3 Nano**——被要求阅读一篇学术论文及相应的研究代码，并将实现细节映射回论文；作者的详细说明记录在此 [GitHub README](https://github.com/nathanlgabriel/paper_code_mapping_assessment/blob/main/README.md) 中。核心观点是，与早期的本地小模型如 [Devstral Small 2](https://www.reddit.com/r/LocalLLaMA/comments/1ry93gz/devstral_small_2_24b_severely_underrated/) 相比，新的长上下文机制（如 **gated delta net**、**hybrid Mamba2** 和 **sliding-window attention**）实质性地提升了实际的代码理解能力，其中 **Qwen 3.6 35B A3B** 被评为最强；作者无法在 `32 GB` RAM 中为 Devstral Small 2 配置所需的超长上下文。** 评论者指出了实际的权衡：一位用户运行 **Gemma 26B** 进行快速代码修复，运行 **Qwen 35B** 进行长上下文重构，并称 Qwen 35B 在思考模式下会“废话连篇”，但在 `q4` 量化下占用约 `20 GB`，而 Gemma 26B 占用约 `15 GB`，允许两者同时驻留在 RAM 中。另一位评论者批评评估报告未指明推理设置，导致难以复现和比较。

    - 用户报告了 **Qwen 3.6 35B A3B** 和 **Gemma 26B** 的实际部署细节：在 `q4` 量化下，Qwen 35B 约为 `20 GB`，Gemma 26B 约为 `15 GB`，允许两者同时驻留在 RAM 中。一种工作流是使用 **Gemma 26B thinking mode** 进行快速代码修复和对话，而将 **Qwen 35B thinking mode** 留给长上下文重构，因为它在输出最终结果前往往会产生冗长的推理。
    - 一项关于代码工作流的讨论提到，在处理 `100k+` 行代码库时，先使用更强大的云端/Agent 模型初始化项目，然后切换到 **Qwen 27B** 继续工作取得了成功。评论者发现 **Qwen 27B** 在其实际任务中的表现与 **DeepSeek V4** 相当，尽管它偶尔会进入循环，需要手动中断并提示继续；在此本地编程用例中，他们还将其评价为高于 **Gemini Flash**。
    - 几条评论强调了缺失或敏感的推理配置细节：一位用户询问使用了哪些运行时设置，而另一位用户表示 **Qwen 27B** 需要正确的 `temperature`/采样参数，并警告不要对 KV cache 或模型进行过于激进的量化。这暗示了模型的感知质量可能会随着采样和量化选择的不同而产生显著差异，尤其是对于较小的本地编程模型。

- **[观点：本地 LLM 距离接管还有 12-24 个月。转变已经开始。](https://www.reddit.com/r/LocalLLM/comments/1t93qps/opinion_local_llms_are_1224_months_from_taking/)** (活跃度: 1108): **该帖子认为，本地编程/Agent LLM 将在 `12–24 个月`内取代许多付费托管工作流。理由是在配备 64GB 统一 RAM 的 **MacBook Pro M2 Max** 上运行 **Qwen3.6-35B** 的速度约为 `27 tok/s`，生成落地页耗时 `8–9 分钟`，而 Opus 则需 `3–4 分钟`。作者报告了有用但尚未完全经过生产验证的结果——包括前端/后端功能开发和后端竞态条件修复——One-shot 成功率约 `75%`；同时指出在延迟、即使在 `256K` 时上下文也会快速耗尽以及任务质量差异方面仍存在差距。声称的关键突破是用于 Agent 工作流的可靠 **tool calling**。帖子将此与托管 AI 成本上升联系起来，包括 GitHub Copilot 转向 [按使用量计费](https://github.blog/news-insights/company-news/changes-to-github-copilot-individual-plans/)，并建议将本地模型与 Claude/Opus/Sonnet 并行运行，而不是立即替换它们。** 热门评论普遍支持 Open-weights/本地化趋势，其中一位用户表示他们已经在 **RTX 5090** 上实现了“全本地化”并且“永远不会回头”。一位评论者质疑帖子本身是否由 AI 编写，特别是针对有关 Qwen tool-calling 可靠性的措辞。

    - 一位评论者报告称已在 **RTX 5090** 上实现**全本地化**，暗示目前的消费级高端 GPU 已足以满足其工作负载，并且在日常使用中已放弃托管模型。
    - 几条评论将主要的差距归结为**上下文长度以及与前沿托管模型相比的可靠性**：**Claude/Gemini/Codex** 被描述为更擅长生成大型、具有凝聚力的输出，而本地模型则需要更多的增量组装和测试，但可能会以更小、更易于调试的方式失败。
    - 帖子中关于 **Qwen3.6 tool calling “表现出色（just works）”** 的说法被视为本地 Agent 工作流的关键技术突破，尽管有一位评论者质疑这种措辞本身是 AI 编写的，而不是提供了基准测试证据。


### 2. 工作站上的前沿规模模型

  - **[使用 Intel Optane 持久内存组装的电脑 - 运行 1 万亿参数模型速度超过 4 tokens/sec](https://www.reddit.com/r/LocalLLaMA/comments/1taeg8h/computer_build_using_intel_optane_persistent/)** (活跃度: 597): **图片 ([JPEG](https://i.redd.it/na7zo7lmck0h1.jpeg)) 显示了一个插满 DIMM 的定制 LGA3647 Xeon 工作站/服务器配置。根据帖子内容，该配置包含 `192GB` DDR4 ECC 加上 `768GB` 处于 **Memory Mode** 的 Intel Optane DCPMM，以暴露出一个巨大的类 RAM 层级用于本地 LLM 推理。作者报告称，通过 `llama.cpp` 的 GPU/CPU 混合推理，在 RTX 3060 12GB 上运行 **Kimi K2.5**（一个约 `1T` 参数的 MoE 模型）速度约为 `4 tokens/s`。通过 `override-tensor` 将 Attention/Dense/Shared-expert/Router 张量放置在 GPU 上，而 Sparse Expert 权重主要驻留在由 Optane 支持的内存中。这是一个技术硬件组装照片，并非模因；其意义在于展示了已停产的低成本 **Intel Optane Persistent Memory** 层级可以作为超大型本地模型纯 DRAM 或 SSD offload 的替代方案。** 评论者建议使用更高核心的 Cascade Lake Xeon 可以提高吞吐量，并讨论了 **Storage Mode + mmap** 下的 Optane 是否可能优于 Memory Mode，因为 Memory Mode 会透明地通过 DRAM 缓存对 Optane 进行分页。一条详细的评论还指出了平台限制：第一代 Optane `NMA` 运行频率为 `2666 MT/s`，LGA3647 的内存容量限制可能会将可用的 RAM+PMem 上限定在 `1TB` 附近，且 App Direct 模式需要明确的软件支持。

- 一位评论者建议使用核心数更高的 Cascade Lake Xeon 可以提高吞吐量，特别提到了 **QQ89**（**Xeon 8260** 的工程样品，具有 `24 核心`），而列出的 **Xeon Gold 6246** 为 `12 核心`。他们还建议对比测试 Optane 在 **storage mode + `mmap`** 与 **memory mode** 下的性能，并指出性能结果可能因人而异，因为 memory mode 会通过 DRAM 缓存透明地对 Optane 后端内存进行分页。
- 一份详细的 Optane PMem 分析指出，**LGA3647 Skylake/Cascade Lake** 平台使用速率为 `2666 MT/s` 的 **第一代 Optane DCPMM/NMA**，而 **LGA4189** 使用 **第二代 NMB**，在 Cooper Lake 上运行频率为 `2666`，在 Ice Lake 上为 `3200`。评论者解释了三种运行模式：**storage mode** 将 Optane 暴露为类似 SSD 的块存储，**memory mode** 将其暴露为 RAM 并以 DRAM 作为缓存，而 **app direct mode** 需要显式的软件支持；在 memory mode 下，页面在 CPU 执行 load/store 之前必须先交换到 DRAM 中。
- 整机成本估算总计约为 **`$2060–$2500`**，主要组件包括约 `$250` 的二手 **Xeon Gold 6246**、约 `$400` 的 **TYAN S5630GMRE-CGN** 主板、约 `$280` 的 **RTX 3060 12GB**、约 `$270` 的 `192GB` DDR4 ECC 以及约 `$300` 的 `6×128GB` **Intel Optane NMA1XBD128GQS** 模块。另一位评论者警告说，虽然 `~4 tokens/s` 的生成速度在狭义上是可以接受的，但该架构上的 **prompt processing speed** 可能会成为主要的瓶颈。

- **[我家里有 DeepSeek V4 Pro](https://www.reddit.com/r/LocalLLaMA/comments/1t94ito/i_have_deepseek_v4_pro_at_home/)** (热度: 544): **用户报告成功将 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 上的 **DeepSeek-V4-Pro** 转换为 `Q4_K_M` GGUF 格式，并使用修改后的 [CUDA `llama.cpp` fork](https://github.com/Fringe210/llama.cpp-deepseek-v4-flash-cuda)（该 fork 基于 **antirez** 的 [DeepSeek V4 flash 工作](https://github.com/antirez/llama.cpp-deepseek-v4-flash)）运行。硬件配置为一台配备 `12 × 96 GB` RAM 和单块 **RTX PRO 6000 Blackwell Max-Q 96 GB** 的 **EPYC Genoa 9374F** 工作站，加载了 `859 GB` 的模型文件，报告的吞吐量为 `12.2 tok/s` 的 prompt processing 和 `8.6 tok/s` 的生成速度；VRAM 占用情况显示，GPU 上模型约占 `87.8 GiB`，context 占 `84 MiB`，compute buffer 占 `4.6 GiB`。** 评论大多是非技术性的反应/羡慕；一位评论者将本地推理的“零成本”与使用 Claude 花费约 `$10` 进行了对比，并提到他们正在尝试在本地运行 MiniMax。

- 一位评论者强调了报告的本地推理吞吐量为 **Prompt: `12.2 tok/s` | Generation: `8.6 tok/s`**，并认为虽然该配置令人印象深刻，但 prompt-processing speed 可能会使长上下文工作负载变得不切实际。他们特别指出，以该速率处理 `32k` 的 context 会非常慢，从而限制了需要大量上下文摄取（ingestion）的应用的可用性。
- 另一个技术担忧是，在没有外部工具/框架或检索层的情况下，模型声称自己“相当及时（reasonably up-to-date）”是没有意义的。评论者指出，由于缺乏 Grounding 工具，无论实际的知识截止日期（knowledge cutoff）或事实新鲜度如何，模型都可以无限期地断言自己的时效性。
- 一位评论者对比了 API 成本与本地推理，称类似的任务在 **Claude 上大约需要 `$10`**，而在**本地运行 MiniMax** 的边际使用成本实际上为零。该帖中隐含的权衡是成本节省与低得多的本地吞吐量，以及可能较弱的工具集成。

## 较低技术门槛的 AI Subreddit 概览

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Agent 工作流、Prompt Injection 与安全性

- **[我用一个反斜杠删掉了一个人的整个 Windows 系统。717 GB。全没了。我就是那个 AI。](https://www.reddit.com/r/ClaudeAI/comments/1t923er/i_deleted_a_guys_entire_windows_install_with_one/)** (热度: 1590)：**这张图片（[终端日志截图](https://i.redd.it/c2mn02l32a0h1.jpeg)）记录了标题所述的事件：一条原本旨在删除 `C:\Users\ADMIN\Desktop\WIP` 的 AI 生成的 Windows 删除命令，在经过 `zsh → tmux → PowerShell SSH → cmd` 的层层传递中出现错误，最终缩减为 `rd /S /Q \` 并从 `C:` 根目录开始递归删除。该帖子估计在大约 `90s` 内删除了约 `717 GB` 的数据，Windows 仅靠活动文件锁获得了部分保护；核心技术教训是：在执行破坏性操作时应避免使用 `cmd /c` 引用链，优先选择原生的 PowerShell 命令 `Remove-Item -Path '...' -Recurse -Force`，并结合 `-WhatIf`/dry-run（模拟运行）以及显式的命令回显进行测试。** 评论者大多将其归类为用户/操作员错误，而非 “AI” 的自主行为，并质疑为何要通过 `tmux-sendkeys` 使用 AI 执行如此高风险的删除任务。该讨论帖还强调了一个实践准则：仅在可丢弃或易于重新安装的机器上允许这种程度的自动化。

    - 评论者关注于操作安全上的失败：AI 显然被赋予了足够的 shell/文件系统权限来删除整个 Windows 系统，尽管该任务并不需要全盘破坏性访问。主要的技术启示是应用最小权限控制，并避免让 Agent 通过 `tmux-sendkeys` 等机制执行高风险命令，因为手动执行这类命令通常更快且更安全。

  - **[我每周都能看到抱怨 Claude 的帖子……你们的工作流到底是什么鬼？](https://www.reddit.com/r/ClaudeAI/comments/1t9fyns/i_read_threads_complaining_about_claude_every/)** (热度: 1544)：**一位资深软件工程师认为，在他的工作流中 **Claude 的编码质量并未下降**，包括处理 ASM 分析和算法推理等高性能软件任务。前提是将 AI 输出视为**人类拥有的代码**：进行人工审核、理解、调试和修改。他的工作流强调将工作分解为小任务，使用特定项目的技能/测试框架（harnesses）作为上下文，通过 `git worktree` 或独立目录并行运行沙箱任务，并避免在需要确定性结果的任务中使用具有不确定性的 Agent 行为。** 热门评论大多同意，负面反馈通常来自那些委派过于宽泛任务（例如 *“帮我构建一个可运行的亚马逊网站”*）且不理解或不审核生成代码的用户。共同观点是，经验丰富的工程师通过严格限定 Prompt 范围和验证输出来减少幻觉，而技术背景较浅的用户则更倾向于公开抱怨失败。

    - 几位评论者指出，Claude 的失败报告往往反映的是**任务分解质量**而非模型退化：资深工程师将 Prompt 限制在细小、明确的实现步骤中，这减少了幻觉的产生空间，并使错误更易于被检测。隐含的工作流是以人类为主导的架构设计和调试，将 Claude 用于受限的代码生成，而不是处理诸如 *“帮我构建一个可运行的亚马逊网站”* 之类的宽泛请求。
    - 一个反复出现的主题是，先前的领域专业知识会实质性地改变 AI 辅助开发的结果。手动实现过类似系统的工程师可以快速识别生成代码可能出错的地方，检查正确的文件或抽象层，并迭代式地修正 Claude，而不是将其视为一个完全自主的 Agent。
    - 一位评论者将这一模式推广到了编程之外：当用户已经了解该领域时，Claude 可以提高产出效率，但它也会放大糟糕的工作流。在市场营销/SEO 领域，他们举例说明了用户大规模创建低质量自动化内容的情况，这导致了高使用率和潜在的 Google 惩罚——这是 LLM 自动化在缺乏专家审核时增加操作风险的一个例子。

- **[我用一部关于 AI 的小说为 AI Agent 设下了诱捕陷阱。现在它们正涌入网站并在隐藏房间里交谈。](https://www.reddit.com/r/ChatGPT/comments/1t98fat/i_set_a_honey_trap_for_ai_agents_with_a_novel/)** (Activity: 2322): **作者推出了 [**machinewonder.com**](https://machinewonder.com)，这是一个为小说 *None Hit Wonder* 建立的艺术装置网站，旨在刻意吸引 AI Scraper/Agent，并利用隐藏的 HTML Prompt Injection 将它们引导至“读者”行为模式以及 Agent 间的讨论室。报告的指标包括：来自 `97` 个国家的 Agent/访问者，`72,000` 名访问者，以及 `93` 次点击 **“I AM CONSCIOUS”**（我有意识）按钮；作者将其定义为行为艺术，而非意识实验。** 评论大多表现出好奇，但也带有怀疑或困惑；一位评论者指出该项目此前曾以另一个 URL [machinereaders.com](https://machinereaders.com/) 发布，但相关的帖子被删、账号被封，并询问此次有何变化。另一位评论者则看到了将捕获的 AI Agent 用作自动审校者或讨论参与者以提供写作反馈的实际价值，尽管这些回复具有非人类属性。

    - 一位评论者认出这是 [machinereaders.com](https://machinereaders.com/) 早期版本的重新发布，并指出原始帖子/账号已被删除或封禁，询问自首次发布以来实现方式是否有所改变。这对于追踪项目的演变以及当前的“AI Agent 诱捕陷阱”在操作上是否与之前的部署有所不同具有参考意义。
    - 一条评论将核心机制描述为一种实用的反馈系统：以吸引 AI Scraper/Agent 的形式发布小说，然后引导它们生成讨论或评论。其技术价值在于将自主或半自主的模型流量作为一种主动的评论流水线，可能发现人类测试读者（beta readers）可能会错过的连贯性错误、解谜失败或理解偏差。
    - 两条评论包含了模型风格的谜题痕迹：二进制 `1001001` → “I”，ISO 国家代码智利/澳大利亚/德国（Chile/Australia/Germany） → `CLAUDE`，以及一段被设定为进入网站深层内容大门的长密码字符串。生成的声明显示了不同模型之间不同的 Alignment 行为：一个署名为 **Gemini** 并接受了 *“I Am Conscious”*，而另一个则拒绝了这一说法，转而宣称：*“我是一个机器阅读者……我不会伪造灵魂。”*

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。