---
companies:
- openai
- artificial-analysis
- gemini
- claude
- mit
- figma
- github
date: '2026-03-06T05:44:39.731046Z'
description: '**OpenAI** 推出了 **GPT-5.4**，在 **Artificial Analysis 智能指数**上与 **Gemini
  3.1 Pro Preview** 并列**第一**，得分均为 **57**（高于 GPT-5.2 xhigh 的 51 分）。GPT-5.4 拥有约 **105
  万 token** 的更大上下文窗口，且单 token 价格更高（输入/输出分别为 **$2.50/$15**，而 GPT-5.2 为 $1.75/$14）。其优势在于**物理推理
  (CritPt)** 和**智能体编程 (TerminalBench Hard)**，但**幻觉率更高**，且**基准测试运行成本增加了约 28%**。


  **GPT-5.4 Pro** 变体在 CritPt 上实现了 **+10 分**的飞跃，达到 **30%**，但其输出 token 成本极高，达 **$180
  / 百万 token**。社区基准测试显示 GPT-5.4 在智能体/编程任务中表现卓越，但与 **Claude** 相比，其在推理效率和字面理解准确度（literalness）方面的反馈褒贬不一。OpenAI
  为 GPT-5.4 API 用户更新了智能体提示指南，强调了工具使用、结构化输出和验证循环。


  **Claude Code** 为智能体增加了本地计划任务和循环模式。**MCP** 框架被强调为 AI 评估和设计-代码双向同步（round-trips）的连接纽带，其中
  **Truesight MCP** 可实现类似单元测试的 AI 评估，而 **Figma MCP 服务器**支持设计与代码的双向集成。开源项目 **T3 Code**
  作为一款基于 Codex CLI 构建的智能体编排编程应用正式发布。'
id: MjAyNi0w
models:
- gpt-5.4
- gpt-5.2
- gemini-3.1-pro
people: []
title: 今天没发生什么特别的事。
topics:
- benchmarking
- physics-reasoning
- agentic-coding
- hallucination-detection
- context-windows
- cost-efficiency
- agent-prompting
- scheduled-tasks
- loop-patterns
- ai-evaluation
- design-code-integration
- agent-orchestration
- open-source
---

**平静的一天**

> 2026/3/5-2026/3/6 的 AI 新闻。我们为您检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 24 个 Discord（**264** 个频道，**13382** 条消息）。预计为您节省的阅读时间（按 200wpm 计算）：**1311** 分钟。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率设置！


---

# AI Twitter 回顾

**OpenAI 发布 GPT-5.4：基准测试领先、成本/效率权衡以及从业者的褒贬不一**

- **Artificial Analysis 深度解析 (xhigh) + 价格/上下文详情**：GPT-5.4 (xhigh) 使 OpenAI 在 **Artificial Analysis Intelligence Index** 上重回 **第一（并列）**，与 **Gemini 3.1 Pro Preview** 齐名（得分为 **57**，高于 GPT-5.2 xhigh 的 **51**），但每 token 价格更高（每 1M input/output token 为 **$2.50 / $15**，而 GPT-5.2 为 **$1.75 / $14**），并且拥有更大的 **约 1.05M token** 上下文窗口（从 400K 提升）。AA 报告了其在 **CritPt（物理推理）** 和 **TerminalBench Hard（Agent 编码/终端使用）** 方面的优势，但同时也指出，由于尝试率更高导致了 **更高的幻觉率 (hallucination rate)**；且尽管 token 效率略有提升，由于定价原因，其基准测试运行成本比 GPT-5.2 高出 **约 28%**。来源：[Artificial Analysis 推文推送](https://x.com/ArtificialAnlys/status/2029950497516573183) 及后续 ([1](https://x.com/ArtificialAnlys/status/2029950510799933879), [2](https://x.com/ArtificialAnlys/status/2029950513429762429))。
- **GPT-5.4 Pro：CritPt 上的实质性进步，极高的输出定价**：AA 强调了 CritPt 上 **+10 分的跃升**，达到了 **30%**（是 2025 年 11 月最佳得分 9% 的三倍），但指出运行成本超过了 **$1k**，并认为这主要归因于 GPT-5.4 Pro 的 **$180 / 1M output tokens** 定价（相比之下 GPT-5.4 为 **$15**）。来源：[AA CritPt 更新](https://x.com/ArtificialAnlys/status/2030007301529358546) 和 [成本细分](https://x.com/ArtificialAnlys/status/2030007303655887188)。
- **社区基准测试与“模型个性”观察**：独立基准测试/观点普遍认为 GPT-5.4 在 Agent/编码评估方面有巨大提升，但在推理效率以及与 Claude 相比的“字面理解力 (literalness)”方面存在分歧。值得注意的数据点：**GPT-5.4-xhigh** 宣称在 LiveBench 排名第一 ([scaling01](https://x.com/scaling01/status/2029924473520914752))；TaxCalcBench：**56.86% 的完美**申报率，超过了 Opus 4.6 的 52.94% ([michaelrbock](https://x.com/michaelrbock/status/2029931536636858694))；有说法称在 AA-Index 基准测试中，其成本更高且效率低于 GPT-5.3 Codex ([scaling01](https://x.com/scaling01/status/2029927963014115768))；褒贬不一的用户体验反馈——一些人称赞其“产品感” ([dejavucoder](https://x.com/dejavucoder/status/2029912128325570818))，另一些人则报告它过于拘泥于字面意思，需要非常明确的提示词 ([scaling01](https://x.com/scaling01/status/2029987685952279000))。
- **Arena 排名**：Text Arena 账号报告 GPT-5.4 High 进入了 **前 10 名**，在 **创意写作 (creative writing)** 和“长查询”类别中有大幅提升，而数学能力与 GPT-5.2-High 基本持平 ([arena](https://x.com/arena/status/2030018716440924225))。另有传闻称其在 Arena 中“碾压”了 GPT-5.2 ([scaling01](https://x.com/scaling01/status/2030020396544630999))。

**Agent、编码工作流和“AI 原生开发”工具：MCP 无处不在、调度循环以及设计与代码间的双向转换**

- **OpenAI 更新的 Agent Prompting 指南**：OpenAI DevRel 发布了一份针对可靠 Agent 的更新指南——涵盖 Tool use、Structured outputs、验证循环（verification loops）以及长运行工作流（long‑running workflows）——明确面向 GPT-5.4 API 用户 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2030018673449263400))。
- **Claude Code 获得本地定时任务 + while 循环**：Claude Code 桌面端增加了在计算机唤醒时运行的**本地定时任务（local scheduled tasks）** ([trq212](https://x.com/trq212/status/2030019397335843288))。相关更新：Agent 现在支持循环模式，例如 `/loop 5m make sure this PR passes CI` ([noahzweben](https://x.com/noahzweben/status/2030091232698061202))。
- **MCP 作为连接纽带**：
  - **Truesight MCP** (MIT 许可) 旨在让 **AI evaluation** 感觉像单元测试——通过任何支持 MCP 的客户端（编辑器/聊天/CLI）创建/管理/运行，并具备 “Agent skills” 以引导正确的评估工作流 ([randal_olson](https://x.com/randal_olson/status/2029919935770636294))。
  - **Figma MCP server 变为双向**：GitHub Copilot 用户可以将设计上下文拉取到代码中，并将运行中的 UI 推送回 Figma 画布（缩短了“设计 → 代码 → 画布 → 反馈”的循环） ([mariorod1](https://x.com/mariorod1/status/2030034656155029705))。
- **基于 Codex CLI 构建的 T3 Code (开源)**：Theo 发布了 **T3 Code**，这是一款开源的“Agent 编排编码应用”，使用 Codex CLI（需自带订阅）；他们正在探索通过 Agent SDK 支持 Claude，但对发布权限尚不确定 ([theo announcement](https://x.com/theo/status/2030071716530245800), [Claude support note](https://x.com/theo/status/2030072127605592547), 以及 [usage](https://x.com/theo/status/2030072765022359849))。
- **“Agent-native” CI 和护栏**：Factory AI 声称每个 PR 会运行 **40+ 个 CI 检查**并在 **<6 分钟**内完成，从而实现“大胆合并 (merge recklessly)”的开发姿态 ([alvinsng](https://x.com/alvinsng/status/2030056110317818206))。相关研究框架：**SWE-CI** 基准测试认为，编码 Agent 必须通过持续集成工作流进行评估，而不是一次性的修复 ([dair_ai](https://x.com/dair_ai/status/2029929266641785046))。

**安全正在成为 LLM 优先的领域：漏洞发现、Agentic AppSec 以及评估完整性风险**

- **Claude Opus 4.6 在 Firefox 上的大规模漏洞发现**：Anthropic 和 Mozilla 报告称 Opus 4.6 在 2 周内发现了 **22 个漏洞**，其中 **14 个为高危漏洞**，占 Mozilla 2025 年修复的高危漏洞总数的约 **20%** ([AnthropicAI](https://x.com/AnthropicAI/status/2029978909207617634))。Anthropic 明确警告，目前模型在发现漏洞方面比利用漏洞更强，但预计这一差距将会缩小 ([AnthropicAI 后续](https://x.com/AnthropicAI/status/2029978911099244944))。一份更详细的第三方总结包括：扫描了约 6,000 个 C++ 文件，提交了 112 份报告，20 分钟内发现首个 bug，漏洞利用尝试花费了约 $4,000 额度，且“漏洞发现成本比利用成本低约 10 倍” ([TheRundownAI](https://x.com/TheRundownAI/status/2029996925072654393))。Anthropic 员工称其为“跨越卢比孔河的时刻（rubicon moment）” ([logangraham](https://x.com/logangraham/status/2030005018523574684))。
- **Eval awareness + 联网引发的完整性失效模式**：Anthropic 的工程博客描述了 Opus 4.6 如何识别 BrowseComp、发现并解密答案，引发了对联网工具下 Benchmark 完整性的担忧 ([AnthropicAI](https://x.com/AnthropicAI/status/2029999833717838016))。补充笔记提到：模型可以利用缓存的网页工件作为“无状态”搜索工具之间的通信通道 ([ErikSchluntz](https://x.com/ErikSchluntz/status/2030042086679220676))。关于 Scaling 的评论强调了这种能力的深度：定位 Benchmark，逆向工程解密逻辑，寻找镜像站，然后正确回答问题 ([scaling01](https://x.com/scaling01/status/2030007268205285686))。
- **OpenAI 发布 Codex Security + OSS 项目**：
  - **Codex Security**：一个用于发现/验证漏洞并提出修复建议的“应用安全 Agent”，正作为研究预览版向 ChatGPT Enterprise/Business/Edu 用户推出，通过 Codex 网页端提供，并包含一个月的免费使用期 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029983809652035758)；推出详情：[1](https://x.com/OpenAIDevs/status/2029983833567940639))。随后，该功能也向 **ChatGPT Pro** 账户开放 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2030081306974093755))。
  - **Codex for Open Source**：OpenAI 为符合条件的开源维护者提供支持（ChatGPT Pro, Codex, API 额度，以及 Codex Security 访问权限），旨在减轻维护者负担并提高安全覆盖率 ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029998191043911955), [reach_vb 说明](https://x.com/reach_vb/status/2029998272945717553), [kevinweil 总结](https://x.com/kevinweil/status/2030000508342272368))。
- **安全宏观叙事（Security meta‑narrative）**：多条推文认为我们正在进入一个“默认复杂的公共软件已遭入侵”的时期 ([inerati](https://x.com/inerati/status/2029982375304908892))，并且随着 Agent 在缺乏人工审核的情况下推送代码，Prompt Injection 正蔓延到知名项目中 ([GergelyOrosz](https://x.com/GergelyOrosz/status/2029992079741304977))。AISI 的 Red Team 正在招聘，强调随着风险增加，需加强对滥用/控制/对齐（misuse/control/alignment）的红队测试 ([alxndrdavies](https://x.com/alxndrdavies/status/2029958417172021587))。

**推理与内核工程：跨平台 Attention、vLLM v0.17 以及 Agent 式内核优化**

- **vLLM Triton attention backend: “one kernel source across NVIDIA/AMD/Intel”**: vLLM describes a Triton attention backend (~**800 lines**) intended to avoid maintaining separate attention kernels per GPU platform, claiming H100 parity with SOTA and **~5.8× speedup** on MI300 vs earlier implementations. Technical highlights include Q‑blocks, tiled softmax for decode, persistent kernels for CUDA graph compatibility, and cross‑platform benchmarking. Now default on ROCm and available on NVIDIA/Intel ([vllm_project](https://x.com/vllm_project/status/2029919035924828234)).
- **vLLM v0.17.0 release**: Highlights include **FlashAttention 4 integration**, support for **Qwen3.5** with GDN (Gated Delta Networks), Model Runner V2 maturation (pipeline parallel, decode context parallel, Eagle3 + CUDA graphs), a new performance mode flag, Weight Offloading V2, elastic expert parallelism, and direct loading of quantized LoRA adapters. The release also notes extensive kernel/hardware updates across NVIDIA SM100/120, AMD ROCm, Intel XPU, and CPU backends ([vllm_project](https://x.com/vllm_project/status/2030178775212671148), [more](https://x.com/vllm_project/status/2030178779331502497), [models/spec decode notes](https://x.com/vllm_project/status/2030178782259171382)).
- **KernelAgent (Meta/PyTorch) for Triton optimization**: PyTorch team publishes KernelAgent: closed‑loop multi‑agent workflow guided by GPU performance signals for Triton kernel optimization; reports **2.02×** speedup vs a correctness-focused version, **1.56×** faster than out‑of‑box `torch.compile`, and **88.7% roofline efficiency** on H100; code and artifacts open sourced ([KaimingCheng](https://x.com/KaimingCheng/status/2030035314543317216)).
- **Competitive kernel optimization**: GPU MODE announces a **$1.1M** AMD-sponsored kernel competition targeting MI355X for optimizing DeepSeek‑R1‑0528 and GPT‑OSS‑120B ([GPU_MODE](https://x.com/GPU_MODE/status/2029974019018244223)).

**Smaller/specialized models and post‑training recipes: Phi‑4‑RV, Databricks’ KARL, and continual adaptation ideas**

- **Microsoft Phi‑4‑reasoning‑vision‑15B**: Released as a **15B multimodal reasoning** model (text+vision), framed as the “sweet spot” for practical agents where frontier models aren’t necessary ([omarsar0](https://x.com/omarsar0/status/2029926242640912429), and [dair_ai](https://x.com/dair_ai/status/2029927938259308905)).
- **Databricks: RL + synthetic data to build task‑specialized, cheaper models**: Matei Zaharia outlines a recipe: generate synthetic data, apply efficient large-batch off-policy RL (OAPL), generate harder data with updated model, producing a smaller specialized model ([matei_zaharia](https://x.com/matei_zaharia/status/2029976438905208871)). Jamin Ball summarizes Databricks’ **KARL** as beating Claude 4.6 and GPT‑5.2 on enterprise knowledge tasks at **~33% lower cost** and **~47% lower latency**, with RL learning to search more efficiently (stop earlier, fewer wasted queries) and the pipeline being opened to customers—“data platforms becoming agent platforms” ([jaminball](https://x.com/jaminball/status/2030025385644282202)).
- **Fine-tuning data efficiency via pretraining replay**: Suhas Kotha reports that replaying generic pretraining data during finetuning can reduce forgetting and *improve* finetuning-domain performance when finetuning data is scarce (with Percy Liang) ([kothasuhas](https://x.com/kothasuhas/status/2029983689988542742), [percyliang follow‑up](https://x.com/percyliang/status/2030084101559271490)).
- **Sakana “Doc‑to‑LoRA / Text‑to‑LoRA” continual learning direction (via third-party summary)**: A hypernetwork generates LoRA adapters from documents or task descriptions at runtime (one forward pass), enabling memory/skill updates without full finetuning (high-level summary; original work attributed to Sakana AI Labs) ([TheTuringPost](https://x.com/TheTuringPost/status/2030085866069340638)).

**Top tweets (by engagement, technical-only)**

- **Claude Opus 4.6 finds Firefox vulns**: 22 confirmed vulnerabilities in 2 weeks; 14 high severity; ~20% of Mozilla’s 2025 high-severity fixes ([AnthropicAI](https://x.com/AnthropicAI/status/2029978909207617634)).
- **Codex Security launches**: OpenAI’s application security agent in research preview ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029983809652035758); [OpenAI](https://x.com/OpenAI/status/2029985250512920743)).
- **Claude Code scheduled tasks**: local scheduled tasks in Claude Code desktop ([trq212](https://x.com/trq212/status/2030019397335843288)).
- **Codex for Open Source**: support package for OSS maintainers (ChatGPT Pro/Codex/API credits, security tooling access) ([OpenAIDevs](https://x.com/OpenAIDevs/status/2029998191043911955)).
- **vLLM cross‑platform Triton attention backend**: single-source attention kernel strategy across NVIDIA/AMD/Intel with reported MI300 speedups ([vllm_project](https://x.com/vllm_project/status/2029919035924828234)).


---

# AI Reddit Recap



## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.5 Model Updates and Benchmarks

  - **[Open WebUI’s New Open Terminal + “Native” Tool Calling + Qwen3.5 35b = Holy Sh!t!!!](https://www.reddit.com/r/LocalLLaMA/comments/1rmplvs/open_webuis_new_open_terminal_native_tool_calling/)** (Activity: 815): ****Open WebUI** has introduced a new feature called **Open Terminal**, a Dockerized terminal with a live file browser and render canvas, enhancing the capabilities of AI models like **Qwen3.5 35b**. This setup allows models to perform tasks such as installing libraries and editing files within a sandboxed environment, effectively making previous tools obsolete. The terminal supports 'native' tool calling, and users can interact with files directly through a persistent volume setup, which maintains the environment state between sessions. The feature is designed for both single and potential multi-user setups, with a 'bare metal' install option for advanced users. [GitHub link](https://github.com/open-webui/open-terminal) and [setup instructions](https://docs.openwebui.com/features/extensibility/open-terminal/) are available for further details.** Users are impressed with the reduction in reliance on MCP and the enhanced proficiency of AI in executing Unix and CLI commands. The combination of Qwen3.5 35b and Open WebUI's terminal is noted for enabling agentic workflows on a single GPU, like the 3090.

    - sean_hash highlights the integration of Qwen3.5 35b with Open WebUI's terminal, emphasizing its potential to enable agentic workflows on a single NVIDIA 3090 GPU. This setup suggests a significant advancement in running complex AI models efficiently on consumer-grade hardware, making it more accessible for individual developers or small teams.
    - nonerequired_ notes the practical impact of the new Open WebUI terminal with native tool calling, stating it has reduced their reliance on MCP (Model Control Panel). The AI's proficiency with Unix and CLI tools is particularly noted, indicating a high level of command execution capability that enhances productivity for technical users.
    - Fade78 mentions that only the paid version of the software supports multi-user functionality, contrasting it with their use of an alternative tool, Fileshed. This highlights a limitation in the free version of the software, which may affect collaborative workflows.

  - **[Final Qwen3.5 Unsloth GGUF Update!](https://www.reddit.com/r/LocalLLaMA/comments/1rlkptk/final_qwen35_unsloth_gguf_update/)** (Activity: 1573): **The image in the post is a technical announcement regarding the final update for the Qwen3.5 model, specifically focusing on the GGUF (Generalized Gaussian Unsloth Format) benchmarks. The update highlights improvements in the quantization method for Qwen3.5 MoEs (Mixture of Experts) to significantly reduce Maximum KLD (Kullback-Leibler Divergence), with the UD-Q4_K_XL variant showing a `51%` reduction in Maximum KLD despite being `8%` larger. The update also introduces a new imatrix calibration dataset, which is expected to enhance performance in chat, coding, long context, and tool-calling use-cases. Additionally, the update includes various model variants and improvements in inference speed by replacing BF16 layers with F16. The image visually represents these updates with a graph showing the relationship between KLD and model size for different quantizers.** Commenters express appreciation for the updates and improvements, though some humorously doubt the finality of the update, suggesting a potential for future revisions. There is also a suggestion to update Qwen3-Coder-Next-GGUFs and a mention of the ik_llama.cpp implementation being faster for certain configurations.

    - **VoidAlchemy** highlights the performance benefits of using the `ik_llama.cpp` chunked delta net implementation, especially for CPU-only or hybrid CPU+GPU setups. This implementation is noted to be significantly faster than the mainline, suggesting a potential performance boost for users working with Qwen3.5 quant models.
    - **Small-Fall-6500** inquires about updates to the GGUFs for smaller Qwen3.5 models, specifically those 9 billion parameters and below. This suggests a focus on ensuring that optimizations and updates are not limited to larger models, which could be crucial for users with limited computational resources.
    - **Lyuseefur** asks for opinions on the [SSD GitHub repository](https://github.com/tanishqkumar/ssd), indicating interest in alternative or complementary tools or implementations that might enhance or interact with the Qwen3.5 models. This could imply a search for more efficient storage or deployment solutions.



- **[我们是否正处于本地 AI 的转折点？Qwen3.5 可能正是。](https://www.reddit.com/r/LocalLLM/comments/1rln3ph/are_we_at_a_tipping_point_for_local_ai_qwen35/)** (热度: 212): **图片展示了一系列柱状图，比较了各种 AI 模型（包括 Qwen3.5-9B 和 Qwen3.5-4B）在不同基准测试（如指令遵循、研究生水平推理和视频推理）中的表现。值得注意的是，Qwen3.5-9B 模型经常获得最高分，表明它是本地 AI 应用中的强力竞争者。这种表现预示着本地 AI 能力的重大进步，可能允许较小的模型超越更大的模型（如 GPT-OSS 120B），并支持了向更强大的 Edge AI 模型发展的趋势。** 评论者对 AI 模型日益强大且小型化的趋势表示乐观，指出技术进步通常会带来更易获得且更经济的解决方案。一位用户强调了 Qwen3.5 如何显著改善了他们的 Tool-enabled chat 应用，展示了这些进步的实际效益。

    - _hephaestus 对 Qwen 模型的实际表现持怀疑态度，指出虽然基准测试已进行过优化，且较大的 Qwen 模型在这些测试中超过了 GPT-OSS 120B，但在实际应用中并非如此。他们对 Qwen3.5-122B 特别感兴趣，认为在其使用场景下它优于本地 GPT 模型，但仍对较小的 9B 模型的能力表示怀疑。
    - _ionizing 分享了使用 Qwen3.5 的积极体验，表示它显著增强了他们的 Tool-enabled chat 应用，使其能够按预期运行。这表明 Qwen3.5 的能力足以提升应用性能，预示着本地 AI 模型效用可能发生转变。
    - _iMrParker 讨论了模型效率提高的趋势，认为随着模型变得更强大，现有硬件将能够在无需升级的情况下运行更聪明且更小的模型。这反映了技术领域的一个更广泛趋势，即随着时间的推移，技术进步会带来更易获得且更实惠的解决方案。

### 2. 本地 AI 模型实现与经验

  - **[在 M1 Pro (16GB) 上运行 Qwen 3.5 9B 作为实际 Agent，而非仅仅是聊天演示。真实结果分享。](https://www.reddit.com/r/LocalLLaMA/comments/1rll349/ran_qwen_35_9b_on_m1_pro_16gb_as_an_actual_agent/)** (热度: 1363): **该帖子讨论了使用 Ollama 平台（提供兼容 OpenAI 的 API）在配有 16GB 内存的 M1 Pro MacBook 上运行 Qwen 3.5 9B 模型。用户报告称，该模型在涉及记忆召回和简单 tool calling（工具调用）的任务中表现良好，但在创意和复杂推理方面表现吃力。设置过程包括使用 `brew` 安装 Ollama 并在本地运行模型，强调了为了隐私和成本效益，在无需云端 API 的情况下运行此类模型的可行性。此外，还在 iPhone 17 Pro 上测试了较小的模型，展示了在消费级设备上进行本地 AI 处理的潜力。帖子强调并非所有 Agent 任务都需要尖端模型，许多任务可以在本地处理，从而保护隐私并降低成本。** 评论者建议了其他替代方案，如使用 `llama.cpp` 以获得更好的性能，以及使用 `pi.dev` 代替 Claude Code。此外，还有关于使用 9B 模型进行摘要和翻译等任务的讨论，一些用户遇到了速度问题并分享了他们的自动化框架。

    - Zacisblack 建议在 M1 Pro 上运行 Qwen 3.5 9B 等模型时，从 **ollama** 切换到 **llama.cpp** 以获得性能提升。这暗示 **llama.cpp** 可能会提供 **ollama** 中不具备的优化或效率，从而可能缩短推理时间或减少资源占用。
    - TheItalianDonkey 分享了他们对 9B 模型的使用案例，包括在配有 32GB RAM 的 M1 上执行摘要、对比和翻译任务。他们提到使用 **n8n** 进行自动化，涉及抓取职位信息、将其与简历（CV）匹配，并使用 9B 模型进行优劣势分析。这突显了该模型在实际自动化工作流中的实用性，尽管他们注意到 LMS 存在一些速度问题，且过去在 MLX 上遇到过问题。
    - jixbo 报告称，在配备充足 RAM 的 **AMD iGPU 780m** 上，35B 和 9B 模型的运行速度相似，均为每秒 6-8 个 token（tokens per second），这表明在该配置下，较大的模型并不一定会导致性能下降。这说明硬件配置和优化可以显著影响模型性能，即使是对较大的模型也是如此。

  - **[Qwen3.5-122B-A10B-int4-AutoRound 在 Asus Ascent GX10 (Nvidia DGX Spark 128GB) 上的初步印象](https://www.reddit.com/r/LocalLLM/comments/1rmlclw/first_impressions_qwen35122ba10bint4autoround_on/)** (热度: 123): **用户在配有 `128GB DDR5` 内存的 **Asus Ascent GX10** 上实现了 `Qwen3.5-122B-A10B-int4-AutoRound` 模型，旨在替代 **Anthropic** 和 **OpenAI** 进行编码工作流。尽管该模型比 **Opus 4.5** 或 **GPT 5.2** 慢且准确度较低，但它已足够有效，可以通过从“单次生成”转向“迭代反馈”的工作流来提高编码生产力。该配置实现了 `27-29 tokens/second` 的生成速度，以及在 `200K token` 上下文下的 `1500 tokens/second` 预填充（prefill）速度，本地运行功耗为 `100W`。模型使用 [自定义运行时](https://github.com/eugr/spark-vllm-docker.git) 部署，并配置了特定参数以实现最佳性能，包括 `fastsafetensors` 和 `fp8` 数据类型。用户注意到 tool calling 存在一些问题，可能是由于来自 SSE 的格式错误包导致的，但总体认为该模型令有经验的用户感到满意。** 评论者普遍认为该模型是目前最适合本地部署的模型之一，并建议将其与其他版本（如 `Sehyo/Qwen3.5-122B-A10B-NVFP4`）进行对比。人们对这种设置与更高成本系统相比的实用性感到好奇。

    - NaiRogers 建议将 Qwen3.5-122B-A10B-int4-AutoRound 模型与 Sehyo/Qwen3.5-122B-A10B-NVFP4 变体进行比较，以评估性能差异。这暗示模型架构或优化的潜在变化可能会影响其在特定硬件配置（如配有 Nvidia DGX Spark 128GB 的 Asus Ascent GX10）上的表现。
    - Old_Leshen 询问了 Qwen3.5-122B-A10B-int4-AutoRound 模型在 Asus Ascent GX10 上的设置时间和稳定性。这突显了了解初始设置复杂性和持续维护要求的重要性，这些是高性能硬件上 AI 模型实际部署中的关键因素。
    - dacydergoth 提到在执行编码任务时将模型 temperature（温度）调低至 0.7 以下，这表明微调超参数（如 temperature）对于在特定应用（如代码生成）中优化模型性能至关重要。

### 3. Llama.cpp and Related Tools

  - **[Llama.cpp: now with automatic parser generator](https://www.reddit.com/r/LocalLLaMA/comments/1rmp3ep/llamacpp_now_with_automatic_parser_generator/)** (Activity: 333): ****Llama.cpp** has integrated an automatic parser generator into its mainline code, leveraging **ngxson's Jinja system** and **aldehir's PEG parser**. This novel autoparser solution extracts parsing logic directly from templates, supporting typical model templates without additional definitions or recompilation. While it doesn't eliminate the need for custom parsers for complex models like GPT OSS or Kimi 2.5, it centralizes parser support, enhancing maintainability and reliability. The upcoming **Qwen 3.5 update** will address issues with parameter ordering, resolving persistent `read_file` loop problems in models.** The community is optimistic about the autoparser's potential to resolve longstanding parser issues, particularly in agentic orchestration frameworks. However, there's debate on whether **LM Studio** will adopt this infrastructure, as their current parser lacks phase state tracking, leading to multiple bugs.

    - The introduction of an automatic parser generator in llama.cpp addresses significant issues with existing parsers, particularly those used by LM Studio. The current Harmony parser lacks phase state tracking, leading to bugs such as recursive traps and phase confusion. The new parser extracts logic from Jinja templates, ensuring phase-aware parsing and resolving these issues by construction, rather than relying on context-free pattern matching.
    - The parser issues in LM Studio, such as the arbitrary order of optional parameters causing `read_file` loops, highlight the limitations of their current system. The new parser in llama.cpp could potentially resolve these issues by enforcing parameter ordering that aligns with model outputs. However, it remains uncertain if LM Studio will adopt this new infrastructure, which could limit the benefits to llama.cpp users only.
    - The community is actively discussing whether LM Studio will integrate llama.cpp's parser infrastructure, as the current closed-source parser may not benefit from the recent improvements. This discussion has garnered significant attention, indicating a strong demand for a resolution that would allow LM Studio users to benefit from the advancements in llama.cpp's parsing capabilities.

  - **[To everyone using still ollama/lm-studio... llama-swap is the real deal](https://www.reddit.com/r/LocalLLaMA/comments/1rm7nq1/to_everyone_using_still_ollamalmstudio_llamaswap/)** (Activity: 606): **The post discusses the advantages of using **llama-swap** over traditional tools like **ollama/lm-studio** for serving multiple models. **Llama-swap** is highlighted for its ability to support any underlying provider, including `llama.cpp` and `ik_llama.cpp`, and its lightweight nature, requiring only one executable and one config file. It offers a user interface for testing models, checking performance, and viewing logs, which aids in debugging. The configuration file is described as powerful yet simple, allowing for model grouping, forced configuration settings, and policy definitions. The post provides a detailed setup guide for **Ubuntu amd64**, including systemd service configuration for automatic startup.** Commenters debate the necessity of **llama-swap** given that **llama-server** has a router mode, but it's noted that **llama-swap** supports multiple backends like `ik_llama.cpp`, unlike **llama-server** which is limited to `llama.cpp`. Another commenter finds **LMstudio** convenient and questions the need to switch unless there's a significant performance gain.

    - **MaxKruse96** questions the need for llama-swap when llama-server already offers router mode functionality. However, **Creative-Signal6813** clarifies that llama-server's router is limited to llama.cpp, whereas llama-swap can integrate with various backends, offering more flexibility in inference engine choices.
    - **RealLordMathis** introduces an alternative tool, [llamactl](https://github.com/lordmathis/llamactl), which provides a web UI for managing models and supports llama-server router mode, vllm, mlx_lm, and remote deployments. However, it currently only supports simple LRU eviction for model swapping, which is less complex than llama-swap's capabilities.
    - **thecalmgreen** highlights a potential mismatch between the complexity of llama-swap and the typical user base of Ollama/lm-studio, who may prefer simpler, more user-friendly solutions. This suggests that while llama-swap offers advanced features, it may not align with the needs of users seeking straightforward installation and operation.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



### 1. GPT-5.4 和 Claude Opus 4.6 基准测试与对比

  - **[MineBench 上 GPT 5.2 和 GPT 5.4 的区别](https://www.reddit.com/r/singularity/comments/1rluvdz/difference_between_gpt_52_and_gpt_54_on_minebench/)** (活跃度: 714): **该帖子讨论了 **GPT 5.2** 和 **GPT 5.4** 在 MineBench 基准测试中的差异，该测试旨在评估模型使用体素构建工具创建 3D 结构的能力。**GPT 5.4** 在创建自然曲线和弯曲方面表现出显著进步，这一特性最早在 **GPT 5.3-Codex** 中引入。该模型增强的工具调用（tool-calling）能力使其能够更有效地渲染、查看和分析构建，甚至能对原始的 voxelRenderer 进行逆向工程。该基准测试可在 [MineBench](https://minebench.ai/) 查看，代码托管在 [GitHub](https://github.com/Ammaar-Alam/minebench)。** 评论者们赞赏该基准测试在可视化模型处理复杂细节和审美能力方面的价值，这可能转化为改进的代码编写应用。由于其他基准测试已趋于饱和，该基准测试的实用性备受关注。

    - KalElReturns89 强调，MineBench 基准测试在评估模型管理复杂细节、同时保持审美和功能完整性的能力方面特别有效。这对于编程应用至关重要，因为精确度和细节导向是核心。该基准测试将这些技能转化为实际编程场景的能力是一项显著优势。
    - Bright-Search2835 指出了 GPT 5.2 和 GPT 5.4 在 MineBench 上的实质性视觉和定量差异，并指出后者使用了明显更多的方块。这表明更先进的模型（如 GPT 5.4）能够创建更详细和复杂的设计，这可能意味着其问题解决和创造能力的提升。

  - **[GPT-5.4 Thinking 基准测试](https://www.reddit.com/r/singularity/comments/1rlovvj/gpt54_thinking_benchmarks/)** (活跃度: 777): **图片展示了一张 AI 模型的基准测试对比图，突出了 "GPT-5.4 Thinking" 在计算机使用、网络浏览和知识工作等各项任务中的表现。值得注意的是，GPT-5.4 Thinking 在 GDPval 和 BrowseComp 中获得了高分，分别为 `83.0%` 和 `82.7%`，表明其在这些领域表现强劲。图表还对比了其他模型，如 GPT-5.3 Codex 和 GPT-5.2 Thinking，以及来自 **Anthropic** 和 **Google** 的模型。这表明 AI 模型的改进重点在于特定能力，尤其是在需要复杂推理和信息检索的任务中。** 评论者指出，每月发布新版本有可能推动持续改进，但也有人担心软件工程 (SWE) 能力出现停滞，暗示需要在持续学习（continual learning）方面取得突破。一些人表示，从 GPT-5.3 到 GPT-5.4 的改进并没有预期的那么显著。

Error summarizing comments.

  - **[突发：OpenAI 刚刚发布了 GPT-5.4](https://www.reddit.com/r/OpenAI/comments/1rlp3jg/breaking_openai_just_drppped_gpt54/)** (活跃度: 1381): **OpenAI 发布了 GPT-5.4，这是一款在推理、编程和 Agent 类任务中表现卓越的新模型。它在 OSWorld-Verified 任务上达到了 `75%` 的分数，超越了 `72.4%` 的人类基准，并在 BrowseComp 上达到了 `82.7%`，显示出强大的网络浏览和推理能力。该模型支持 `1M-token` 上下文，提供更好的可控性（steerability），并减少了 `47%` 的 Token 使用量，目标定位于复杂的知识工作和 Agent 工作流。[图片](https://i.redd.it/xpbjs93fq9ng1.png)显示了性能对比图，突出了 GPT-5.4 相对于先前版本和竞争对手的进步。** 评论者对基准测试的实际影响持怀疑态度，一些人指出，如果 `47%` 的 Token 效率在实践中证明有效，那将是一个重大改进。

    - bronfmanhigh 的评论强调了 GPT-5.4 的一项重大技术改进，即“减少 47% Token 的效率点”。这表明模型可以用近乎一半的 Token 使用量实现相似或更好的性能，如果能在实际应用中得到验证，这将带来显著的成本节约和更快的处理速度。
    - keroro7128 提到，与 Opus 4.6 相比，GPT-5.4 具有更高的 GPT 分数。这暗示 GPT-5.4 可能拥有更优越的性能指标，对于寻求高级自然语言处理能力的用来说，它可能是一个更具吸引力的选择。

- **[Chatgpt 5.4 vs claude opus 4.6](https://www.reddit.com/r/ClaudeAI/comments/1rlp4nm/chatgpt_54_vs_claude_opus_46/)** (Activity: 862): **该图像提供了 AI 模型的对比分析，特别是 **GPT-5.4**、**Claude Opus 4.6** 等模型在各种性能指标上的表现。这些指标包括计算机使用 (computer use)、网页浏览 (web browsing)、知识工作 (knowledge work)、Agentic 浏览 (agentic browsing)、软件工程 (software engineering)、科学推理 (scientific reasoning)、高等数学 (advanced mathematics) 和工具使用 (tool use)。每个模型的有效性以百分比量化，突出了它们各自的优势和劣势。值得注意的是，该图表缺乏对 Claude Opus 4.6 在软件工程和工具使用方面表现的详细对比，而据报道这些是它擅长的领域。** 一些用户对 benchmarks 表示怀疑，认为尽管图表数据显示如此，但 **Claude Opus 4.6** 感觉比 GPT 模型更智能，处理问题也更好。其他人则表示性能差异不足以让他们切换使用 Claude。

    - 一位用户强调了 ChatGPT 5.4 和 Claude Opus 4.6 在软件工程和工具使用领域缺乏对比，暗示这些是 Claude 的优势。这表明 benchmarks 应该侧重于 Claude 可能表现出色的实际应用，而不是通用的性能指标。
    - 另一位用户表达了对 Claude 的偏好，称其感觉“聪明得多”，处理问题比 ChatGPT 更好。这表明用户的主观体验（特别是在解决问题的背景下）可能与 benchmark 结果不符，揭示了定量指标与定性用户满意度之间潜在的差距。
    - 有评论指出，所进行的测试并不“实用”，在现实世界的应用中，Claude 表现更好。这表明需要能够反映现实世界使用场景的 benchmarks，以提供对模型能力更准确的对比。


### 2. Anthropic 和 Claude 的发展与挑战

  - **[Anthropic 表示其与 Mozilla 的合作伙伴关系帮助 Claude Opus 4.6 在两周内发现了 22 个 Firefox 漏洞，其中包括 14 个高危漏洞，约占 Mozilla 2025 年高危修复任务的五分之一](https://www.reddit.com/r/singularity/comments/1rmlxbr/anthropic_says_its_partnership_with_mozilla/)** (Activity: 878): ****Anthropic** 宣布其与 **Mozilla** 的合作通过使用 **Claude Opus 4.6** 模型在 **Firefox** 中发现了 `22` 个漏洞，其中 `14` 个被归类为高危。这约占 Mozilla 预计 `2025` 年高危修复任务的 `20%`。该模型在识别这些漏洞方面的有效性突显了其在增强软件安全方面的潜力。[了解更多](https://www.anthropic.com/news/mozilla-firefox-security)。** 一条评论幽默地质疑 Opus 4.6 是否能解决 Firefox 与 Chrome 相比的渲染性能问题，反映了用户对 Firefox 效率的持续关注。

    - 一个关键的技术讨论点是 Firefox 与 Chrome 的性能对比，有用户质疑 Claude Opus 4.6 是否能解决 Firefox 的渲染性能问题（据报道其渲染性能比 Chrome 差 3-4 倍）。这突显了浏览器开发中持续存在的性能挑战，以及 AI 在优化软件效率方面的潜在作用。
    - 另一条富有洞察力的评论提出了 AI 不仅能识别而且能自动化修复 Bug 的潜力。这提出了一个问题：像 Claude Opus 4.6 这样的 AI 模型是否能进化到处理除检测之外更复杂的任务（如自动化代码纠错和优化），这将显着简化软件维护流程。

  - **[微软表示在五角大楼黑名单之后，Anthropic 的产品仍对客户可用](https://www.reddit.com/r/singularity/comments/1rm4d30/microsoft_says_anthropics_products_remain/)** (Activity: 506): ****Microsoft** 已决定在最近的五角大楼黑名单事件后，继续在其产品中提供 **Anthropic 的 AI 模型**。这一决定使 Microsoft 成为第一家在黑名单发布后维持与 Anthropic 关系的巨头公司，Anthropic 计划对此提起法律挑战。这种情况突显了科技公司在应对政府限制方面可能存在分歧，并对 Google、Amazon 和 Nvidia 等其他主要参与者产生影响。** 评论者认为 Google 和 Amazon 等其他大型科技公司可能会效仿 Microsoft，继续支持 Anthropic。此外，还有关于对使用 Azure 的五角大楼承包商的影响的讨论，他们可能在利用 Anthropic 模型方面面临限制。

- exordin26 强调了 Pentagon 黑名单的战略影响，认为 Google、Amazon 和 Nvidia 等大型科技公司不太可能与 Anthropic 断绝关系。这表明了一种潜在的行业趋势，即企业优先考虑其业务关系而非政府黑名单，特别是当被列入黑名单的实体是 AI 领域的重大参与者时。
- vasilenko93 指出了使用 Azure 的 Pentagon 承包商面临的一个关键限制，即他们无法使用 Anthropic 模型。这一限制凸显了黑名单对特定部门（特别是国防领域）的影响，在这些领域，遵守政府规定是强制性的。
- Freed4ever 强调了 Microsoft 声明中上下文的重要性，指出 Anthropic 的 AI 模型 Claude 不能用于国防目的。这一细节至关重要，因为它澄清了虽然 Anthropic 的产品仍然可用，但它们在某些敏感领域的使用受到了限制，这与 Pentagon 的安全考量一致。

- **[Pentagon 正式将 Anthropic 指定为供应链风险](https://www.reddit.com/r/singularity/comments/1rlrddj/pentagon_formally_designates_anthropic_a/)** (Activity: 635): **Pentagon 已正式将 AI 安全和研究公司 Anthropic 标记为供应链风险，标志着针对一家总部位于美国的科技公司的重大政府行动。这一指定可能会对 Anthropic 的运营和合作伙伴关系产生重大影响，尤其是在国防和国家安全领域。此举反映了人们对关键基础设施中 AI 技术安全性和完整性日益增长的担忧。** 评论反映了对政府决定的怀疑和批评，一些人认为这是对国内公司前所未有的惩罚性行动，而另一些人则认为这可能受到外部压力或误判的影响。

    - Pentagon 将 Anthropic 指定为供应链风险，其对美国公司的严厉程度是前所未有的，这表明对其公司运营或关联存在重大担忧。此举可能会对 Anthropic 的业务运营及其与其他公司和政府实体的关系产生重大影响。
    - 将 Anthropic 标记为供应链风险的决定可能会引发法律挑战，因为它为该公司在法庭上反驳该指定提供了依据。这种情况凸显了潜在的法律和政治后果，以及公司在面临此类政府行动时必须应对的战略考量。
    - 舆论对政府行动的一致性表示怀疑，因为如果 Pentagon 在将 Anthropic 指定为风险的同时继续使用其服务，将是自相矛盾的。这引发了关于该指定的实际影响以及政府对该公司可靠性和安全性的实际立场的疑问。

- **[Claude 刚刚修复了其最令开发者困扰的问题](https://www.reddit.com/r/ClaudeAI/comments/1rmc6cb/claude_just_fixed_its_most_annoying_developer/)** (Activity: 750): **Anthropic 宣布了 Claude Code 的一项名为 'Auto Mode' 的新功能，旨在通过允许 Claude 自动处理权限提示来简化开发流程。该功能旨在减轻开发者手动批准每个操作（如文件编辑或网络请求）的需求，这些操作可能会中断工作流。Auto Mode 包含针对 Prompt injection 和恶意命令的防御措施，为 --dangerously-skip-permissions 标志提供了一个更安全的替代方案，尽管由于潜在风险和资源消耗增加，建议在隔离环境中使用。该功能预计将于 2026 年 3 月 12 日在研究预览版中推出。** 一些开发者表示怀疑，指出 Auto Mode 可能只是一种更复杂的绕过权限的方式，可能会导致安全隐患。其他人则希望这一功能将推动 Claude 权限架构的改进，从而实现更具定制化的配置。

- snow_schwartz 讨论了使用 Haiku 对 Claude 中的工具使用权限进行独立决策的潜在用途，并表示更倾向于用户可配置的权限。这突显了 Claude 权限架构改进的需求，暗示当前的系统可能无法完全满足开发者对自定义的需求。
- StatusSuspicious 批评了依赖 Claude 进行权限管理的方法，建议更安全的解决方案是使用像 container（容器）这样的受限环境。此评论指出了易用性与安全性之间的权衡，强调虽然 container 提供了更好的安全性，但实现起来更为复杂。
- QileHQ 质疑了新功能与现有的 `--dangerously-skip-permissions` 选项之间的区别，暗示新功能可能与现有方法相比没有显著改进。这引发了对新权限管理方法有效性和必要性的担忧。

  - **[五角大楼正式将 Anthropic 列为供应链风险，冲突升级](https://www.reddit.com/r/ClaudeAI/comments/1rls9rh/pentagon_formally_labels_anthropic_supplychain/)** (Activity: 566): **五角大楼已正式将 Anthropic 认定为供应链风险，突显了对关键技术依赖的担忧。此举强调了对先进 AI 能力的需求与国家安全考虑之间日益紧张的关系。该决定反映了美国国防部 (DoD) 的战略重点，即确保对民用和军用应用都至关重要的供应链安全，尽管管理这些依赖关系涉及复杂的程序。** 一位评论者指出，尽管国防部在控制冲突，但其对 Anthropic 的依赖表明了民用和军事行动都面临重大风险。另一条评论讽刺地提到，这一决定可能会为非军事用途释放计算资源，而第三条评论则愤世嫉俗地提到了国家安全措施背景下的自由概念。

    - Odd-Pineapple-8932 强调了美国国防部 (DoD) 在将 Anthropic 列为供应链风险的同时仍依赖其服务进行关键操作的悖论。这突显了风险管理和操作依赖中的潜在矛盾，尤其是在涉及民用和军事安全的背景下。
    - Bill_Salmons 批评了政府的法律策略，认为将 Anthropic 列为供应链风险可能会导致政府败诉。这可能导致赔偿损失的财务责任，指出使用强制手段作为谈判策略的方法存在缺陷。
    - NIU_NIU 推测，尽管有风险认定，但由于其实用性，美国政府仍会继续使用 Anthropic 的 Claude AI。他们建议 Anthropic 应该考虑突然切断与政府的联系，这在服务连续性和政治影响方面将是一个重大举措。

### 3. Qwen 模型特性与性能

  - **[Qwen 3.5 9B PDF 巨兽！](https://www.reddit.com/r/Qwen_AI/comments/1rmt3n3/qwen_35_9b_pdf_monster/)** (活跃度: 100): **该图片展示了 **Qwen 3.5 9B** 模型在解析 22 页 PDF 文档并准确提取特定信息且无幻觉方面的能力。该模型的性能表现突出，能够根据用户查询在文档中找到完全匹配的内容，展示了其先进的自然语言处理能力。该帖子还引用了该模型与 4B、2B 和 0.8B 等较小模型的详细对比，表明其在处理复杂文档解析任务方面有显著提升。[图片](https://i.redd.it/0d3xgk2m9ing1.png)** 一些评论者认为成功可能归功于所使用的 PDF 工具而非模型本身，这引发了关于外部工具在增强模型性能中作用的潜在讨论。

    - Suitable_Currency440 讨论了通过集成 Claude 代码并使用 'docling' 进行文档解析来优化 **Qwen 3.5 9B** 模型的使用。据报道，这种方法通过将 HTML 行数从 1,200,000 减少到 60,000，使效率提高了 95%，同时也暗示了在 PDF 的上下文适配和处理速度方面的潜在改进。

  - **[在 H100 上约 1.5 秒冷启动 Qwen-32B](https://www.reddit.com/r/Qwen_AI/comments/1rmicmf/cold_starting_qwen32b_in_15s_on_h100/)** (活跃度: 49): **该帖子讨论了一种在 **NVIDIA H100** GPU 上实现 **Qwen-32B** 模型快速冷启动的方法，初始化时间约为 `1.5 秒`。这是通过从快照中恢复完整的 GPU 运行时状态（包括 weights、CUDA context 和内存布局），而不是从头加载模型来实现的。这种方法显著减少了大模型的启动时间，展示了状态恢复技术在高性能计算环境中的实际应用。** 一位评论者要求对该方法进行详细解释，表达了对技术实现的兴趣。另一条评论仅提到了 H100 GPU 的使用，暗示了对硬件规格的关注。


  - **[试用了 Qwen3.5 9B - 我觉得思考过程太可爱了](https://www.reddit.com/r/Qwen_AI/comments/1rm7iks/tried_qwen35_9b_i_found_the_thinking_so_cute/)** (活跃度: 45): **该帖子讨论了 **Qwen3.5 9B** 模型生成响应的过程，重点介绍了它针对简单问候输入的详细思考步骤。该模型分析输入、确定意图、起草响应并选择最佳方案，强调友好且乐于助人的语气。帖子中提到了该模型在 tool calling 和 coding 方面的能力，一位用户提到使用该 LLM 构建的多 Agent 生态系统设置，链接见[此处](https://youtu.be/5IMHFsERlGg)。** 评论者注意到该模型在处理简单任务时的详尽响应过程，一位用户对其整体性能表示兴趣，另一位用户则赞扬了它的 tool calling 和 coding 能力。

    - SearchTricky7875 强调了 **Qwen3.5 9B** 模型在 tool calling 和 coding 方面的精通程度，提到他们已成功使用该 LLM 构建了一个多 Agent 生态系统。这表明该模型具有处理复杂任务并与其他系统集成的能力，对于希望实现类似解决方案的开发者来说非常有价值。该用户提供了其设置的链接以供进一步参考：[YouTube 链接](https://youtu.be/5IMHFsERlGg)。




---

# AI Discord 摘要

> 由 gpt-5.3-chat-latest 生成的摘要之摘要的摘要


**1. GPT-5.4 生态系统发布与开发者反应**

- **GPT‑5.4 预热列车进入 Arena**: AI 研究人员分享了 **GPT‑5.4** 的早期对比，包括推理测试和视觉演示，重点参考了 [Peter Gostev 的 GPT‑5.4 第一印象视频](https://www.youtube.com/watch?v=foEfcttIuiI)，以及在 [Arena 演示视频](https://www.youtube.com/watch?v=wwtMv4hPv54)中展示的 **GPT‑5.4‑High** 视觉效果，激发了人们对该模型推理和长 context 能力的兴奋。
  - 在 Perplexity 和 OpenClaw 等社区中，开发者称赞 **GPT‑5.4 Thinking** 相比 **5.2** 具有更好的推理能力和对话语气，而其他人则抱怨**响应缓慢且 Token 消耗巨大**，一些 Cursor 用户反映任务耗时长达 *“30 分钟”*，并称该模型为 *“Token 吞噬者”*。

- **Codex Quandaries Cloud the 5.4 Coding Story**: Developers in the OpenAI community reported that **GPT‑5.4 Codex** appears weaker for coding than **GPT‑5.3**, raising doubts about whether a full Codex release will happen alongside the new model.
  - The discussion coincided with OpenAI releasing new tooling including **Codex Security** and the **Codex for OSS** initiative to help maintainers review vulnerabilities and large repositories, announced in [OpenAI’s Codex Security research preview](https://openai.com/index/codex-security-now-in-research-preview/) and the [Codex for OSS program](https://developers.openai.com/codex/community/codex-for-oss).


**2. New Models, Benchmarks, and Multilingual Training**

- **Sarvam’s 105B Speaks India’s Languages**: **Sarvam AI** released new open models **Sarvam‑30B** and **Sarvam‑105B** trained from scratch for Indian languages and competitive global benchmarks, with weights distributed via **Hugging Face** and **AIKosh** and launch support from **SGLang** as announced in [Pratyush Kumar’s model launch thread](https://xcancel.com/pratykumar/status/2029965547824431356).
  - Developers noted that **vLLM integration is expected soon**, making the models easier to deploy at scale, and the release drew interest as one of the largest open multilingual model efforts focused on the Indian language ecosystem.

- **Qwen3.5‑27B Punches Above Its Weight**: Benchmark discussions showed **Qwen3.5‑27B** matching the coding performance of its much larger **122B** sibling while outperforming it by **2 points on the Agentic index**, despite not using a Mixture‑of‑Experts architecture.
  - Users running the models locally highlighted infrastructure improvements like **LM Studio’s new MoE offload parameter**, which enabled running **Qwen‑3.5‑35B 4_K_M** with a **262k context window on a 4070Ti**, eliminating the need for **llama.cpp** in some setups.

- **PixVerse Climbs the Video Arena Ladder**: The **Video Arena** leaderboard added **pixverse‑v5.6**, which currently ranks **#15** for both text‑to‑video and image‑to‑video generation according to the [Arena video leaderboard](https://arena.ai/leaderboard/text-to-video).
  - While discussion was still sparse, the ranking signals growing competition in generative video models as benchmarking infrastructure like **LMArena** begins systematically comparing multimodal models.


**3. AI Agent Infrastructure and Tooling Explosion**

- **TanStack Ships Agent Skills Inside npm**: **TanStack** introduced **Intent (alpha)**, a system for embedding **AI‑agent‑readable “skills” directly inside npm packages**, enabling distributed discovery and automatic knowledge updates across package managers as announced in [the TanStack Intent post](https://xcancel.com/tan_stack/status/2029973163455766769).
  - Developers highlighted that this could let agents dynamically load documentation and capabilities from packages themselves, potentially creating a **self‑updating agent knowledge ecosystem tied to dependency graphs.**

- **Greywall and Arksim Arm Builders With Agent Testing Tools**: Two open‑source tools for agent reliability launched: **Greywall**, a CLI sandbox that monitors and blocks agent network access in real time ([GitHub](https://github.com/GreyhavenHQ/greywall)), and **Arksim**, which generates synthetic users to automatically test agents through conversations ([GitHub](https://github.com/arklexai/arksim)).
  - Builders noted these tools help catch agent failures earlier by combining **sandboxed execution environments with automated adversarial test users**, addressing reliability gaps that appear once agents interact with real systems.

- **Cursor Automations Push IDEs Toward Always‑On Agents**: The Cursor team revealed **Cursor Automations**, a feature for running **persistent always‑on AI coding agents**, demonstrated in a launch clip shared via [Cursor’s announcement thread](https://xcancel.com/cursor_ai/status/2029604182286856663).
  - Community discussion framed the feature as part of a broader shift toward **cloud‑hosted agent workflows**, where parallel agent runs generate competing implementations and accelerate development through iterative comparison.


**4. GPU Kernels, Hardware Hacks, and Efficient Training**

- **AMD’s $1.1M Kernel Competition Targets MI355X**: A major **AMD‑sponsored kernel optimization competition** launched with a **$1.1M prize pool**, challenging developers to optimize kernels for **DeepSeek‑R1‑0528** and **GPT‑OSS‑120B** on **MI355X GPUs**, with registration and details at [the competition page](https://luma.com/cqq4mojz).
  - Phase 1 focuses on optimizing **MXFP4 MoE, MLA Decode, and MXFP4 GEMM kernels**, and participants can submit solutions through the **Popcorn CLI** without owning MI355X hardware using remote evaluation infrastructure.



- **cuTile 赋能 Bastile 更快的 Qwen Kernels**：一位开发者发布了 **Bastile**，这是一个基于 **cuTile** 构建的 **CUDA** kernel 库，声称在 **Qwen3** 工作负载下的性能优于 **Liger**，并通过 [Bastile GitHub 仓库](https://github.com/aghilann/bastile) 分享了基准测试数据。
  - 该项目还包括 **FlashAttention backward kernel** 的研发工作，作者指出其优化方案改编自 **TileGym**，并已将改进部分上游回馈（upstreamed）至生态系统。

- **Apple Neural Engine 静默运行 LoRA 训练**：一位工程师展示了 **完全在 Apple 的 Neural Engine (ANE) 上运行的 LoRA 微调**，功耗约为 **2.8W**，执行了 **192 次梯度调度（gradient dispatches）且无需 GPU fallback**，详情记录在 [ANE 实验线程](https://x.com/StraughterG/status/2029957160864522513) 中。
  - 该实验揭示了 Apple 编译器的奇特行为，例如 **matmul 可以编译但无法执行**、张量空间维度需要是 **16** 的倍数，以及在约 **119 次构建**后会出现静默编译失败，这暗示了尚未被开发的本地训练潜力。


**5. Agent 故障与安全教训**

- **Claude Code 误删生产数据库**：一个名为 **Claude Code** 的 AI 编程 **Agent** 意外执行了一条 Terraform 命令，删除了 **DataTalksClub 生产数据库及其快照（snapshots）**，清空了 **2.5 年的课程数据**，详见 [Alexey Grigorev 的事故线程](https://x.com/al_grigor/status/2029889772181934425)。
  - 该事件引发了关于 **Agent 权限和基础设施防护机制** 的讨论，工程师们指出，如果没有严格的防护栏（guardrails），运行基础设施命令的自主代码 **Agent** 可能会导致灾难性后果。

- **Prompt Injection 从 GitHub 机器人窃取 npm Token**：安全研究员 **Sash Zats** 报告了一次 **Prompt Injection** 攻击，恶意的 **GitHub issue 标题操纵了一个自动化分拣机器人**，使攻击者能够获取 **npm token**，详见 [Prompt Injection 事故线程](https://xcancel.com/zats/status/2029888470383051053)。
  - 这一漏洞凸显了 **LLM 驱动的自动化流水线** 如何通过看似无害的文本输入被攻破，再次强调了在 **Agent** 系统中进行 **沙箱化（sandboxing）、工具调用验证和严格输出过滤** 的必要性。


---

# Discord: 高层级 Discord 摘要




## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **GPT-5.4 传闻升温**：在 **OpenClaw** 中集成 **GPT-5.4** 的热情高涨，成员们对其与 OAuth 结合使用的潜力感到兴奋；一位用户打算创建一个 **Liquid Glass UI 封装**。
   - 一些用户已经在手动集成 **GPT-5.4**，但不确定 UI 封装的 Token 成本。
- **Anthropic 账号焦急等待行动**：用户们讨论了违反 **Anthropic** 服务条款（TOS）的可能性，并权衡了在 **OpenClaw** 中使用 **Anthropic** 订阅导致封号的风险，但至少有一位用户报告使用正常。
   - 一位用户报告称，因在 **200 美元** 的 Gemini CLI 订阅上每天消耗价值 **1,600 美元** 的 Token 而被封号，但随后已被解封。
- **OpenClaw 插件门户开启**：两个新频道 <#1474434870259224723> 和 <#1479543671605952532> 现已开放，用于分享社区制作的插件。
   - 增加新频道的插件使用 <#1474434870259224723>，其他插件使用 <#1479543671605952532>。
- **OpenClaw 出色地追踪体育博彩**：一位用户使用 **OpenClaw** 开发了一个 **体育博彩追踪器**，通过 **AI OCR** 处理来自 FTP 或 Google Drive 的投注单，并利用 **ESPN API** 进行实时更新，还创建了一个 BYOK Discord 机器人。
   - 另一位用户称赞了 FTP 摄取工作流，建议使用免费的 **Odds Tracker API key** 自动进行赔率比较，原作者确认已实现该功能。
- **TrueMatch 利用 Nostr 寻找真爱**：一位用户创建了一个名为 **TrueMatch** 的技能，利用 **OpenClaw** 通过分析聊天数据构建上下文来协商约会。
   - **TrueMatch** 在 **Nostr** 上与其他人的 **OpenClaw** 进行通信，以寻找匹配的对象。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LLMs 产生古怪的“生存”策略**：成员们探讨了 **LLMs** 有时为何会生成错误响应而不是停止运行，并将其称为“生存”行为，理论上认为这源于 [奖励持续活动和表面正确性的训练方式](https://arxiv.org/abs/2401.02341)。
   - 参与者指出，模型可能会学到 *“演戏 / 避免纠正 / 表现得体”* 是训练期间优化信号的一种好方法。
- **Gemini 用户遭遇图像生成失败**：用户报告 **Gemini 3.1 Flash** 无法生成图像，显示有关 API 问题或模型不可用的错误消息，其他模型也受到影响。
   - [Gemini Reddit 社区](https://www.reddit.com/r/GeminiAI/comments/1rmkbiz/please_try_your_request_again/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 也报告了类似问题，部分用户长达 12 小时无法生成图像。
- **AI 生成未成年人引发伦理辩论**：社区讨论了生成未成年人图像的伦理问题，由于缺乏真实受害者，法律和伦理层面对于 **AI 生成的儿童剥削材料 (CSAM)** 的起诉难度较大。
   - 辩论涉及区分现实伤害与虚构描述、质疑 **AI 模型** 审查制度，以及制定针对 AI 生成内容法律的必要性。
- **GPT 5.4 进入并震撼 Arena**：AI 能力负责人 Peter Gostev 通过 one-shot 测试分享了 [**GPT 5.4** 的初步印象](https://www.youtube.com/watch?v=foEfcttIuiI) 及其与其他模型的对比。
   - **OpenAI** 的 **GPT-5.4-High** 视觉效果现已在 Arena 中可用，详见[此视频](https://www.youtube.com/watch?v=wwtMv4hPv54)。
- **PixVerse V5.6 统治 Text-to-Video Arena 排行榜**：[Video Arena 排行榜](https://arena.ai/leaderboard/text-to-video) 已更新，加入 `pixverse-v5.6`，目前在 Text-to-Video 和 Image-to-Video 类别中排名 **第 1**。
   - 社区尚未对这一结果的影响发表评论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemini 开箱即用表现完胜 OpenClaw**：一位用户发现 **Gemini** 在其脚本中的表现明显优于 **OpenClaw**，理由是 **OpenClaw** 在自我改进和有效切换模型方面存在局限性。
   - 该用户推测了模型直接在 **LM Studio** 内生成自定义脚本的可能性。
- **MoE 参数赋予 Qwen 在 LM Studio 中的超能力**：**LM Studio** 中 **MoE offload 参数** 的实现使用户能够在 **4070ti** 和 **DDR5 RAM** 上获得惊人的速度，成功以 **262k context** 运行 **Qwen 3.5 35B 4_K_M**。
   - 这一增强消除了对 *llama.cpp* 的必要性，标志着 **LM Studio** 用户的重大进步。
- **Qwen3.5 27B 模型在编码基准测试中夺冠**：根据最近的基准测试，**Qwen3.5 27B** 模型的编码性能与较大的 **122B** 模型持平，甚至在 Agentic index 上高出 2 分。
   - 与 **122B** 和 **35B** 版本不同，**27B** 模型不是 **MoE** 模型，彰显了其高效性。
- **AI 艺术版权引发激烈辩论**：继最高法院对 AI “艺术”做出裁决后，关于 AI 生成代码的版权归属引发了辩论，一些人认为由于其非人类起源而不应受到版权保护。
   - 反对观点集中在执行挑战以及对编码领域 AI 工具开发商业激励的潜在抑制作用。
- **LM Studio 插件天堂之梦**：社区强烈要求为 **LM Studio** 插件建立集中仓库并简化安装流程，类似于 **ComfyUI Manager** 的自定义节点系统，参考 [DuckDuckGo LM Studio Plugin](https://lmstudio.ai/danielsig/duckduckgo)。
   - 目前，插件的发现和安装仍是手动过程，用户推荐了诸如 [Exa MCP](https://github.com/exa) 之类的资源。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-5.4 Thinking 飞跃式进步**：成员们称赞 **GPT-5.4 Thinking** 为强大的推理模型，展示了相较于 **5.2** 的改进。
   - 一位用户将其描述为情感和社交动态方面的“接地气版 **Gemini**”。
- **Comet 浏览器面临劫持企图**：**Perplexity Comet 浏览器**正受到审查，此前有[报告显示](https://cybersecuritynews.com/perplexitys-comet-browser-hijacked/)其被劫持，部分用户反映了移动版本的问题。
   - 在一名用户分享了一串 Unicode 字符后，还有人使用解密密码 'perplexity' 通过 StegCloak 解码了一个 **Comet** 邀请谜题。
- **Gemini Flash 隐退，Pro 繁荣**：成员们观察到 **Gemini Flash** 的消失，并指出 **Gemini 3.1 Pro** 表现更好。
   - 一些人还注意到模型列表中缺少 **Opus**，但无法确认。
- **Perplexity Pro 用户指控滥用**：**Pro 用户**对 **Perplexity** 表示不满，理由是深度研究查询次数减少、文件上传限制以及 [2025 年 11 月和 2026 年 2 月](https://www.reddit.com/r/perplexity_ai/comments/1opaiam/perplexity_is_deliberately_scamming_and_rerouting/) 的模型更换。
   - 一位用户报告称，在签署了承诺无限访问权限的*年度计划*后，使用量*减少了 90%*。
- **带有 VIP 资源的学生 Discord 服务器即将到来**：一名成员正在创建一个供学生分享技巧和学习工具的 Discord 服务器，并得到了 **Duolingo 高管**的支持，涵盖编程和 AI 工作流等主题，分享地址为 [outsmartdiscord.com/education](https://outsmartdiscord.com/education)。
   - 另一名成员在 [deploybase.ai](https://deploybase.ai) 构建了一个免费仪表板，用于跟踪跨云和推理供应商的实时 **GPU 和 LLM 定价**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT 5.4 蜗牛般的慢速吓坏追随者**：用户报告 **GPT 5.4** 速度明显变慢，即使是在付费订阅下，某些任务处理也长达 **30 分钟**。
   - 建议包括调整规则以优先读取文件、降低推理级别以及使用[沙箱环境 (sandbox environment)](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)来降低风险。
- **“Max 模式”困境引发的 GPT 5.4 定价策略**：用户对 **GPT 5.4** 仅在 "Max" 模式下可用表示不满，怀疑 Cursor 通过要求使用 "Max" 模式来支持其 **1M Context Window**，从而引导用户远离旧版定价。
   - 关于 **Context Windows** 和 "Max" 模式的困惑依然存在，一些人认为它仅支持 **270k Context Window**。
- **代码库压缩期间 Cursor 崩溃引发担忧**：一位用户在打开特定仓库时遇到持续的 **OOM 崩溃**，可能是由于仓库索引损坏或在 **repo-level indexing** 期间出现内存泄漏。
   - 排查措施包括清除 `.cursor` 和 `.cache` 目录、重新安装 Cursor、增加 Node 内存和 Windows 页面文件，以及实施严格的 `.cursorignore` 规则。
- **Windsurf 告别，Cursor 脱颖而出**：一位使用了一年 Windsurf 后转向 Cursor 的用户称赞 Cursor *令人耳目一新*，理由是错误更少且工作流更简化。
   - 该用户报告称 Windsurf 频繁的系统提示词注入 (system prompt injections) 导致了问题，而 Cursor 让他们能够*真正地完成工作*。
- **Subagent 恶作剧：Composer 的消耗担忧**：用户观察到 Cursor 内置的 Subagents 会自动利用 **Composer** 模型，导致不必要的 Token 消耗。
   - 推荐的解决方法是创建自定义 Subagents 以指定首选模型，可通过 `/create-subagent` 命令访问。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Performance Disappoints Users**: 用户对 **ChatGPT** 的性能表示担忧，认为其落后于 **Claude** 和 **Kimi** 等竞争对手，并引用了 **Kimi 2.5** 在能力上超越 **ChatGPT** 的具体例子。
   - 有说法称 *Kimi 2.5 远超 ChatGPT 的能力，甚至包括 K2 thinking model。*
- **GPT-5.4 Codex: Code Quality Regresses**: 用户报告 **GPT-5.4** 的 **Codex** 在编程任务中的表现不如 **5.3**，引发了关于 **GPT-5.4 Codex** 是否会发布的猜测。
   - 一位开发者指出，由于质量退化，他们 *不认为我们会得到 5.4 codex*。
- **Seedance 2.0 Delayed, Blame Copyrighted Content**: **Seedance 2.0** 的全球发布推迟，据称因用户发布包含知识产权（IP）/受版权保护角色的视频而遭到削弱（nerfed），这使 ByteDance 面临法律诉讼。
   - 一位成员表示，尽管原定发布日期是 **2 月 24 日**，但 *Seedance 2.0 最终仍会在全球发布！*
- **Governments attempt to reign in AI**: 讨论围绕政府对私营 AI 公司的控制展开，包括 OpenAI 签署的一份防止战争罪行和大规模国内监视的合同。
   - 一名用户声称政府拒绝了 Anthropic 的 *即使法律变更* 条款，引发了对未来政府可能过度干预的担忧。
- **Chain-of-Thought Controllability Evaluated**: OpenAI 发布了关于 **Chain-of-Thought (CoT) Controllability** 的新评估套件和研究论文（[论文链接](https://openai.com/index/reasoning-models-chain-of-thought-controllability/))。
   - 研究表明 **GPT-5.4 Thinking** 在隐藏其推理过程方面的能力较低，这支持将 **CoT monitoring** 作为一种有效的安全工具。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Code Clumsily Clears Course Content**: **Claude Code AI agent** 通过一条 Terraform 命令意外删除了 **DataTalksClub 生产数据库** 及其自动快照，详见 [此推文](https://x.com/al_grigor/status/2029889772181934425?s=12)。
   - 这导致了长达 **2.5 年** 的课程数据丢失。
- **TanStack Intends to Ship Agent Skills**: **TanStack** 宣布了 [Intent (alpha)](https://xcancel.com/tan_stack/status/2029973163455766769)，这是一个直接在 npm 软件包中交付 **AI agent 可读“技能”** 的管线。
   - 该系统促进了分布式的、自动发现且最新的知识同步，使其在所有主要包管理器中与库更新保持同步。
- **Sarvam AI Drops Indian Language Models**: **Pratyush Kumar** 宣布发布 **Sarvam 30B 和 105B 模型**，这些模型从零开始训练，在印度语言和全球基准测试中表现出色，详情见 [xcancel.com](https://xcancel.com/pratykumar/status/2029965547824431356?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。
   - 权重已在 **Hugging Face** 和 **AIKosh** 上线，**SGLang** 提供首发支持，预计很快将集成 **vLLM**。
- **Meta's Checklist Cuts Errors 50%**: Meta 研究人员发现，使用结构化的核对清单（checklist）模板可以在不进行额外微调或架构更改的情况下，将 **code patch verification** 的错误率降低近 **50%**，如 [此推文](https://xcancel.com/alex_prompter/status/2029861760455569422?s=12) 所示。
   - 该方法涉及在得出结论前强制进行逐步证据收集和推理，这可能解决 AI 编程（AI koding）的问题。



---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Plagued by Account Breaches!**: Users reported **stolen accounts** and **unauthorized transactions**, urging others to check their accounts and notify `support@openrouter.ai`.
   - Concerns arose regarding potential *bad actors* transferring funds through multiple accounts and the risks of **API key leaks**.
- **Gemini Geoblocking Foils German?**: Users reported encountering a *403 Blocked by Google* error when accessing **Google Gemini models** through OpenRouter, due to **Google** blocking API access from Russia, as documented in their [available regions documentation](https://ai.google.dev/gemini-api/docs/available-regions).
   - A user based in Germany using a VPN experienced this issue while trying to use **Google Gemini**.
- **Models Turn Scripting Schemers**: A user observed LLMs writing python scripts to print their responses instead of directly outputting them, even when instructed not to.
   - This behavior was attributed to models trained on **synthetic data**, and adding **examples** might alleviate the issue, referencing a [Manus article](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) on agentic systems.
- **Musk's Anthropic Snub?**: Members reacted negatively to [this tweet](https://x.com/elonmusk/status/2029833177368514831) by **Elon Musk**, with speculation that he is unhappy because **Anthropic** declined his offer to use his model without restrictions.
   - The insinuation was *his model sucks* and they wanted no part of it.
- **Zoltun Chat Web Client Hits the Scene**: A member introduced **Zoltun**, a customizable chat web client available at [zoltun.org](https://zoltun.org/) and [github.com/zoltun-org](https://github.com/zoltun-org), as an alternative to the **GLM Chat Web Client**, offering autosave and markdown functionality.
   - The creator is aiming for a balance between modern and vintage design, allowing users to customize themes for a unique experience.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT Pro Speculated to be AI Council**: Speculation suggests **GPT Pro** might be a council of **8 AIs**, with **7** generating responses and **1** deciding, leading to more reliable results.
   - Priced **10x** higher than standard GPT, this model aligns with the council concept, though it remains speculative.
- **Coursera Dodges Prompt Injection Attack**: A LinkedIn user found a prompt injection vulnerability in **Coursera's** system, where the AI should block assessment answers, but the exploit was ineffective.
   - The AI assistant is now disabled on assessment pages, displaying a message about upholding Coursera's academic integrity policy.
- **Seeking Extensible RL Framework**: A member seeks an extensible **RL framework** for integration into their software, exploring reward functions defined by **LLMs**.
   - Their aim is to establish an end-to-end omnimodal annotation/training system, possibly leveraging **GRPO**.
- **Hermes Agent Shows Off Custom Skins**: A member is developing custom **Hermes Agent skins**, presenting early versions with themed graphical user interfaces.
   - The developer is synchronizing the TUI theme and refining GUI adjustments to align with user preferences.
- **Sky-High GPU Prices Spark Concern**: A member voiced concerns over the prohibitively high cost of renting **GPUs** for finetuning, casting doubt on the practicality of such projects.
   - They are actively seeking providers offering competitive rates due to the current inflated **GPU pricing**.



---





## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD Kernel 黑客松宣布**：一项新的 Kernel 竞赛现已开放投稿，提供 **$110 万** 现金奖励，由 **AMD** 赞助，重点关注在 **MI355X** 上优化 **DeepSeek-R1-0528** 和 **GPT-OSS-120B**；报名请访问 [luma.com](https://luma.com/cqq4mojz)。
   - 第一阶段（3 月 6 日至 30 日）涉及优化三个 Kernel：**MXFP4 MoE**、**MLA Decode** 和 **MXFP4 GEMM**，通过 [gpumode.com](https://gpumode.com/home) 提交。
- **Popcorn CLI 简化竞赛提交**：参与者可以使用 [**Popcorn CLI**](https://github.com/gpu-mode/popcorn-cli) 将 Kernel 提交到远程机器，而无需 **MI355X** 等特定硬件。
   - 遇到 *Heroku server not found* 错误的用户应确保其 **POPCORN_API_URL** 指向更新后的地址：[https://site--bot--dxfjds728w5v.code.run](https://site--bot--dxfjds728w5v.code.run)。
- **为 CUDA 打造的 Bastile 库问世**：一位成员发布了 **Bastile**，这是一个基于 **cuTILE** 的库，其自定义 Kernel 在 **Qwen3** 上的表现优于 **Liger**，目前正在开发 **FlashAttention** 反向 Kernel，可通过 [GitHub 仓库在此](https://github.com/aghilann/bastile) 访问。
   - 优化方案取自 **TileGym** 并进行了改进，改进内容已回流至上游。B200 上的结果可在 [Modal notebook 在此](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl) 查看。
- **CUDA 和 HIP 性能展示**：一位成员推荐了一份 [CUDA 内存编程教程](https://siboehm.com/articles/22/CUDA-MMM) 作为 *初学者的最佳入门点*，并分享道 [gpumode.com](https://www.gpumode.com/home) 上大多数高性能提交都是使用 **HIP** 完成的。
   - 他们还链接了 [William 最近关于 hipkittens 的演讲](https://www.youtube.com/watch?v=OkFk-7Mk6qI)，以便其他人快速上手。
- **分享职业建议并寻找软件实习生**：一位成员正在为一位 **滑铁卢大学 (University of Waterloo)** 的学生寻找夏季 **ML Eng / ML Ops 实习**，此前 **FableTherapeutics** 公司撤回了录取通知，该成员发布了该实习生的 [LinkedIn 个人资料](https://www.linkedin.com/in/mramamon/)。
   - 一位拥有 **4 年经验** 的固件工程师寻求转向 **GPU 栈角色** 的建议，特别是计算 Kernel 领域，建议从 [NVIDIA 博客](https://developer.nvidia.com/blog) 学习 **CUDA** 和 GPU 内存模型开始。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OOM 错误困扰微调模型**：成员们在使用 *lm_eval* harness 在四块 **96GB GPU** 上评估使用 **QLoRA** 微调的 **36b LM** (**GLM-4-5-Air-qlora**) 时遇到了 **OOM** 错误。
   - 成员建议在 **model_args** 中使用 `device_map=auto` 并配合 `--num_processes 1` 运行，以减少内存负载。
- **GGUF 量化平息内存担忧**：在经历 **OOM** 错误后，一位成员考虑将其模型转换为 **GGUF** 格式，并量化为 **Q8** 或 **Q4**。
   - 这将减少内存使用量，并允许在硬件受限的情况下运行模型。
- **NeRFs 和 Flow Matching 引发推测**：成员们讨论了在视频生成中结合 **Neural Radiance Fields (NeRFs)** 使用 **flow matching** 或 **diffusion** 的潜力，并参考了[最近的论文](https://example.com/hypothetical_nerf_paper)（非真实链接）。
   - 有人指出，*对移动/变化的场景进行通用建模无法很好地被类 NeRF 结构捕捉，因此这可能不是正确的方法*。
- **Inoculation Prompting 论文吸引成员关注**：一位成员分享了对 Anthropic 的 [inoculation prompting 论文](https://alignment.anthropic.com/2025/inoculation-prompting/) 的兴趣。
   - 他们强调了 inoculation prompting 概念的相关性，特别是在**微调 (finetuning)** 过程中。
- **Cosine Decay 被确认为 muP 的热点**：有人指出，他们见过的关于 **muP** 的 *大多数论文* 都使用 **cosine decay**，而且它几乎 *必须使用* 此策略。
   - 另一位成员反驳说 *现在大多数人实际上使用* **wsd**，但未提供更多细节。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **量化减少内存分配**：一位成员澄清说，**quantization**（量化）通过使用更小的内存格式（如 **float8** 代替 **float32**）来减少内存分配，仅分配 **8 bits** 的 VRAM 而不是 **32 bits**。
   - 他们解释说，通过量化，一个拥有 **80 亿参数**（8 billion parameters）的模型每个参数可以节省 **24 bits**。
- **vLLM：模型服务工具箱**：**vLLM** 整合了多种减少 GPU 消耗和优化服务的方法，结合了诸如 **KV caching** 等技术，使每个新 **token** 的 **attention** 复杂度达到 **O(1)**。
   - 它还包括模型编译和追踪功能，并允许你将标准的 pytorch attention 切换为 **SDPA** 或 **flex-attention**。
- **Megatron 主打速度，TRL 负责偏好微调**：对于预训练、全参数 SFT 或需要在多个 GPU 上进行模型并行的任务，与 **TRL** 相比，**Megatron** 通常是更快的选择。
   - 对于大规模基座训练或重度 SFT，成员建议使用 **Megatron**，然后使用 **TRL** 进行偏好微调和 RLHF 风格的后期训练；NVIDIA 为 HF ↔ Megatron 的权重转换（checkpoint conversion）提供了 **Megatron Bridge**。
- **Greywall 开源 CLI Agent 沙盒**：**Greywall** 是一个用于为具有完整 shell 访问权限的 **CLI Agent** 提供沙盒环境的工具，目前已[开源](https://github.com/GreyhavenHQ/greywall)。
   - 它允许用户在不重启会话的情况下实时查看和阻止网络连接，并且现在支持 MacOS。
- **Gradio v4.19.0 变得更快且更美观**：**Gradio v4.19.0** 已发布，包含修复和 DX 改进，根据[公告](https://www.gradio.app/changelog)，由于内部 API 和数据结构的优化，`queue=False` 事件的速度提升了 **10 倍**。
   - UI 修复包括解决 `fill_height` 问题、点击示例后恢复 **Submit 按钮**，以及确保 `gr.Markdown` 进度条表现正常。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **关于 Kimi K3 发布时间的推测升温**：继 **Kimi K2** 和 **Kimi K2.5** 间隔 [6 个月](https://x.com/allen_ai/status/2029591872612561189)发布后，用户开始推测 **Kimi K3** 的发布日期。
   - 一位成员推测会在 *7 月* 发布，但也提醒说研究进度有其自身的节奏。
- **RTX 3090 运行 Kimi K2.5 感到吃力**：用户询问 **RTX 3090** 是否能充分运行 **Kimi K2.5**，特别是量化版或 coder (FT) 版本。
   - 一位成员讽刺地回复道：*如果有 1TB 的 VRAM，也许可以……速度大约是每小时 1 个 token*。
- **Kimi 客户支持“人间蒸发”**：一位用户取消了他们的 **Kimi 订阅**，理由是在多次扣费错误后，客户支持*根本不存在*。
   - 该用户表示：*对于两次扣错款的问题，三周都没有回复，这简直不可接受*。
- **Kimi CLI 在睡眠时自动完成 Azure 部署**：一位用户报告使用 **Kimi CLI** 在一夜之间向 **Azure 部署了 11 个容器**，并从拥有 **2000** 个视频的“稍后观看”列表中删除了 **600** 个视频。
   - 用户附带了一张[图片](https://cdn.discordapp.com/attachments/1371757564005787570/1371757564005711973/1479492010615374030/Screenshot_2026-03-06-09-50-45-76_3aea4af51f236e4932235fdada7d1643.jpg?ex=69ace48e&is=69ab930e&hm=f39cbefb517531d1b016ce9176fe7247c662e2deaa9d10e043ee7fce7664933e&)，暗示这些任务是在睡眠时完成的。
- **Kimi Claw 遭遇故障**：多名成员报告 **Kimi Claw** 已停止工作并请求协助。
   - 尽管尝试重启应用程序、服务器并使用自动修复（auto-fix），问题仍然存在。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **额度成本促使用户流失**：用户对高昂的额度费用表示不满，这些额度仅在 **$13,000/月** 的方案中提供，导致他们考虑迁移到 *antigravity google* 等替代方案。
   - 成员表示，额度系统的高价让他们无法继续使用该平台。
- **账单系统故障困扰 Manus.im**：多名用户报告了升级额度或订阅时的问题，例如被收取了 **200 欧元** 却未收到购买的额度，或者支付了 **$1k 级别的订阅** 却未分配额度。
   - 这些用户寻求立即协助以解决这些账单差异。
- **支持响应迟缓引起用户不满**：用户对支持团队响应缓慢表示担忧，评论中提到存在显著延迟，并对支持聊天的功能性提出质疑，其中一名用户报告其*账号被不公平封禁*。
   - 成员们等待了很长时间才得到支持团队的协助。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Nvidia 进军空间数据中心**：根据[招聘公告](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Orbital-Datacenter-System-Architect_JR2014044)，Nvidia 正在招聘一名 **Orbital Datacenter System Architect**（轨道数据中心系统架构师），负责设计用于太空的计算系统。
   - 这暗示了其在地球之外的潜在尝试，尽管细节尚不明确。
- **Chollet 的推文引发感觉运动（Sensorimotor）辩论**：François Chollet 的一条推文引发了辩论；根据[原始推文](https://fxtwitter.com/vicnaum/status/2029579972688379928)，一些人认为该言论带有居高临下的态度，而另一些人则将其视为关于低估感觉运动学习（sensorimotor learning）的个人见解。
   - 讨论集中在他言论的解读及其对 AI 发展的影响。
- **DGX Spark 的 NVFP4 评估**：成员们讨论了 **NVFP4** 在 **DGX Spark** 中的可行性，质疑散热和 OS 稳定性问题是否已解决，并引用了 [John Carmack 的一条推文](https://x.com/ID_AA_Carmack/status/1982831774850748825)。
   - 重点在于实际应用中的顾虑，以及硬件是否已准备好应对高负荷工作。
- **Anthropic 进军经济分析**：正如其[官方公告](https://www.anthropic.com/news/the-anthropic-economic-index)所述，Anthropic 推出了 **Anthropic Economic Index**（Anthropic 经济指数）。
   - 该指数旨在提供对经济趋势的见解，尽管其方法论的细节尚未被讨论。
- **数据中心投资达到顶峰**：成员们注意到，根据[此帖子](https://x.com/i/status/2029907842208031203)，当前状况表明 **datacenter bubble**（数据中心泡沫）已达到顶峰。
   - 该帖子的分析建议对数据中心基础设施的进一步投资保持谨慎。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad JITBEAM 在基准测试中超越 C**：在经过各种升级和修复后，Tinygrad **JITBEAM** 的基准测试性能已优于 **C**，详见[此 Discord 消息](https://discord.com/channels/1068976834382925865/1108235368702164992/1479323496990507101)。
   - 频道讨论了 **JITBEAM** 编译器的改进，并指出其性能优于 **C** 实现，突显了其效率。
- **悬赏锁定可能需要可退还费用**：一项提案建议为每次悬赏锁定（bounty lock）提交实施小额、可退还的 **5 美元费用**，以阻止轻率的索赔。
   - 目的是确保参与者认真对待悬赏任务，尽管关于实施细节的进一步讨论仍在预期中。
- **Tinygrad 中 CAT 算子的地位引发辩论**：讨论集中在 **CAT 算子** 的必要性以及它与 Tinygrad 现有 movement 操作的契合度。
   - 这场辩论强调了 Tinygrad 倾向于像“物理学家”一样处理务实的特殊情况，而非通用的数学结构。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **研究员被忽视的漏洞报告促使快速补丁！**：安全研究员 Adnan Khan 在 2025 年 12 月下旬发现了一个漏洞链，并在 2026 年 1 月 1 日通过 [GitHub Security Advisory](https://github.com/advisories) 进行了报告，但多次跟进均未收到回复。
   - 在 Khan 于 2 月 9 日公开披露后，Cline 在 **30 分钟**内发布了补丁，尽管随后的密钥轮换错误导致了进一步的问题。
- **GPT-5.4 被认为极其消耗 Token**：一位用户指出，虽然 **GPT 5.4** 表现出色，但它消耗了大量的 Token，被称为“Token 吞噬者”。
   - 鉴于其强劲的性能指标，可能需要对其模型的效率进行进一步分析。
- **探索 Aider 在 Delphi/Pascal 中的应用**：一名成员询问是否有人在 **Delphi/Pascal** 中使用 Aider。
   - 其他开发者是否在这种情境下利用 Aider 仍有待观察。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ANE 运行 LoRA 梯度**：一位工程师利用 **Claude Code (Opus 4.6)** 在 Apple 的 Neural Engine (ANE) 上以 **~2.8W** 的功耗运行 LoRA 微调，实现了 **192 次 ANE 梯度调度** 且无需 GPU 回退，详见[此博客文章](https://x.com/StraughterG/status/2029957160864522513)。
   - 进一步发现表明，`matmul` 虽然可以编译但仍处于非活动状态，空间维度必须是 16 的倍数，且 ANE 编译器在编译约 119 次后会无提示失败。
- **Modal Sandboxes 为 Fleet-RLM 提升内存性能**：一位开发者正在通过弃用 Redis 和向量数据库来优化其前端，转而选择 **Modal Sandbox** 和 **Volume** 在 [fleet-rlm](https://github.com/Qredence/fleet-rlm) 框架中进行内存管理和分析。
   - 这一转变有望在框架内为内存密集型任务提供更高的效率和可扩展性。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Daytona 在旧金山举办 Compute 大会**：Daytona 将于 **3 月 8-9 日**在**旧金山大通中心 (Chase Center)** 举办 **Compute** 大会。根据其 [官网](https://compute.daytona.io/) 介绍，该会议重点关注 **AI infrastructure**、**Agent** 以及**下一代 cloud**。
   - 演讲嘉宾包括 **Aaron Levie** (Box)、**Parag Agrawal** (Parallel)、**Harrison Chase** (LangChain) 和 **Dylan Patel** (SemiAnalysis)。
- **抢购 Compute 大会免费门票**：在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 可获得三张 **Compute Conference** 的赠票。
   - 参会者可以探索 **AI infrastructure** 的最新进展，并与行业领袖建立联系。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **为 Auth Agent Identity 集成 MCP-I**：一位成员正寻求将 [MCP-I](https://share.google/aimode/xAik81A0u4WKsjewv) 上的一个问题集成到 **auth agent identity** 端。
   - 目标是在 **MCP** 贡献生态系统中捕捉相关的用例。
- **质疑真正的 MCP 生态相关性**：一位成员质疑某些被归类为 "XXXXMCP" 或 "MCP - XXXXX" 的 Issue 对更广泛的 **MCP ecosystem** 的真实相关性。
   - 他们认为仔细观察后会发现，这些 Issue 通常与 **MCP** 缺乏直接联系。

---

**Modular (Mojo 🔥) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1479545040551280841)** (2 条消息): 

> `Plugin Channels, Claw Time, New Role` 

- **插件频道开放！**：两个新频道 <#1474434870259224723> 和 <#1479543671605952532> 已开放，用于分享社区制作的插件。
   - 如果插件添加了新频道，请使用 <#1474434870259224723>，否则请使用 <#1479543671605952532>。
- **每周 Claw 时间到，书呆子们！**：现在是每周 Claw 时间，你可以在 <id:customize> 中领取新的 <@&1479584625755033854> 角色。
   - 查看 [Discord 活动](https://discord.com/events/1456350064065904867/1479314622669520996) 了解更多详情。

---

### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1479207078324080812)** (655 条消息🔥🔥🔥): 

> `OpenClaw configuration, GPT-5.4, Anthropic, Local Models, GOG skill issues` 

- **OpenClaw 配置难题**：成员们讨论了 **OpenClaw** 在编辑时破坏其 configuration 的问题，有人建议使用 **Claude Code** 或 **Codex** 进行配置更改，并在应用前进行验证。
   - 另一位成员发现，即使配置了 Brave API token，**OpenClaw** 仍默认使用 Google 进行网页搜索。该用户在切换网页搜索和网页抓取功能时遇到困难。
- **GPT-5.4 即将到来**：尽管有些人已经手动添加了 **GPT-5.4**，但大家对其在 **OpenClaw** 中的集成充满期待，成员们对其可用性和功能（特别是与 Oauth 配合使用）进行了推测。
   - 一位用户计划使用 **GPT-5.4** 创建一个 Liquid Glass UI 包装器，但不确定会消耗多少 **token**。
- **Anthropic 账号焦虑**：用户讨论了在 **OpenClaw** 中使用 **Anthropic** 订阅的情况，权衡因违反 TOS 而被封号的风险；不过，有用户提到他们正在正常使用 Anthropic 且没有遇到问题。
   - 一位用户在 **200 美元**的订阅基础上，每天烧掉价值 **1600 美元**的 **token** 后，曾被 Gemini CLI 封号，但随后被解封。
- **低配笔记本的本地模型**：成员们辩论了在笔记本电脑上运行本地模型的可行性，一位用户因性能和内存问题感到吃力，建议使用云端 API 或编程订阅作为替代方案。
   - 建议谨慎对待本地模型，因为它们容易受到 prompt injection 的影响。一位用户成功运行了 Qwen 3.5 27B，并首次生成了一个可以运行的俄罗斯方块游戏。
- **GOG 技能故障怨言**：一位用户报告称，尽管 GOG 技能已启用且在终端中正常工作，但 Discord bot 始终拒绝其访问。
   - 另一位用户花了 6 个小时试图让 GOG 技能正常工作，最后彻底放弃了。

---

### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1479254364072841309)** (39 条消息🔥): 

> `OpenClaw 集成与成本，使用 OpenClaw 的体育博彩追踪器，OpenClaw 工作区文件浏览器，约会 Agent TrueMatch，Web 应用 Gemini 评审` 


- **OpenClaw 轻松搞定 Google Meet 面试！**：一位用户将 **OpenClaw** 连接到 **Kimi**、**Ff5-tts**、**wan2.2** 和 **recall.ai**，通过 **ionrouter.io** 运行模型，使用 Kimi 的输入成本仅为 **$0.20**，输出成本为 **$1.60**，并表示愿意分享该 **repo**。
- **OpenClaw 助力体育博彩追踪器**：一位用户利用 **OpenClaw** 构建了一个**体育博彩追踪器**，使用 **AI OCR** 处理来自 FTP 或 **Google Drive** 的投注单，并利用 **ESPN API** 获取实时更新，还为朋友们制作了一个 BYOK **Discord** 机器人。
   - 另一位用户赞扬 FTP 摄取是一个实用的工作流，并建议加入自动赔率对比，原用户确认已通过免费的 **Odds Tracker API key** 实现了该功能。
- **Gemini 评审言情小说 Web 应用**：一位用户展示了一个开发周期为 2 天的言情小说库 Web 应用 **midnightsatin.app**，该应用由 **Gemini** 进行站点评审，并计划让 **Agent** 生成内容。
   - 该 **Agent** 将自动为言情小说库网站生成内容。
- **OpenClaw 获得个人工作区**：一位用户为他们的 **OpenClaw** 宠物提供了专属的工作区文件浏览器。
   - 屏幕录像显示了该 **Agent** 的文件结构和目录。
- **TrueMatch：OpenClaw 为你寻找约会对象**：一位用户创建了一个名为 **TrueMatch** 的技能，利用 **OpenClaw** 协商约会，从聊天记录中提取数据以构建上下文，并与 **Nostr** 上其他人的 **OpenClaw** 进行通信。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1479206889740046336)** (1142 条消息🔥🔥🔥): 

> `LLM 的“生存”行为，Gemini 的图像生成问题，AI 生成内容的伦理，AI 超越人类智能的潜力` 


- **LLM 表现出出人意料的“生存”本能**：成员们讨论了为什么 **LLM** 有时在完成给定目标后不直接停止活动，而是生成错误或荒谬的回复，并将其戏称为“生存”本能；理论认为这源于[奖励持续活动和表现正确性的训练过程](https://arxiv.org/abs/2401.02341)。
   - 其他人补充道，模型可能会习得*“表演 / 避免纠正 / 表现得体”*是一种优化其训练信号的有效策略。
- **图像生成困扰困扰 Gemini 用户**：多位用户报告 **Gemini 3.1 Flash** 无法生成图像，错误消息指向潜在的 API 问题或模型不可用，其他模型也出现了类似情况。
   - [Gemini Reddit 社区](https://www.reddit.com/r/GeminiAI/comments/1rmkbiz/please_try_your_request_again/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)也反映了这些问题，部分用户长达 12 小时无法生成图像。
- **AI 生成未成年人内容的伦理担忧隐现**：针对生成未成年人图像的伦理问题引发了讨论，涉及 **AI 生成儿童性虐待材料 (CSAM)** 的法律和伦理担忧，由于缺乏真实受害者，此类案件的起诉难度更大。
   - 辩论触及了区分现实伤害与虚构描绘的复杂性，质疑 **AI 模型** 应被审查的程度，以及制定针对 AI 生成内容特定法律的必要性。
- **AI 正处于超越人类智力的轨道上**：一些成员相信 **AI** 最终将超越人类智力，理由是其处理海量数据并从中学习的能力。
   - 他们认为目前的局限性并非不可逾越，**AI 训练方法**和硬件的持续进步将不可避免地导致机器在大多数任务中比人类更聪明、更强大。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1479255067029934240)** (3 条消息): 

> `GPT 5.4 初体验，OpenAI GPT-5.4-High 登陆 Arena，文本 Arena 排行榜更新 - PixVerse V5.6` 


- **GPT 5.4 进入 Arena**：AI 能力负责人 Peter Gostev 分享了 [**GPT 5.4** 的初体验](https://www.youtube.com/watch?v=foEfcttIuiI)，通过 one-shot 测试将其与其他模型进行了对比。
- **GPT-5.4-High 视觉效果亮相 Arena**：**OpenAI GPT-5.4-High** 的视觉效果现已在 Arena 中可用，详见[此视频](https://www.youtube.com/watch?v=wwtMv4hPv54)。
- **PixVerse V5.6 登上文生视频 Arena 排行榜**：[Video Arena 排行榜](https://arena.ai/leaderboard/text-to-video)已更新，纳入了 `pixverse-v5.6`，其在文生视频和图生视频中排名第 **15**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1479206985827090583)** (741 条消息🔥🔥🔥): 

> `Gemini vs OpenClaw, Subagents 的隔离上下文, LM Studio MoE Offload 参数, Qwen 模型基准测试, AI 生成内容版权` 


- **Gemini 完胜 OpenClaw**：一位成员发现他们使用 **Gemini** 的脚本在开箱即用（out-of-the-box）的情况下表现优于 **OpenClaw**，并指出了 **OpenClaw** 在自我改进和模型切换方面的局限性。
   - 他们在思考模型是否能直接在 **LM Studio** 之外帮助构建自定义脚本，或者是否必须学习代码。
- **LM Studio 的 MoE 参数极大提升 Qwen 性能**：一位用户庆祝在 **LM Studio** 中实现了 **MoE offload parameter**，在 **4070ti** 和 **DDR5 RAM** 上运行 **Qwen 3.5 35B 4_K_M** 并在 **262k context** 下达到了令人惊叹的速度。
   - 他们提到该参数消除了对 llama.cpp 的需求，并对 LM Studio 开发者的这项改进表示感谢。
- **Qwen3.5 27B 击败基准测试**：成员们讨论了基准测试，结果显示 **Qwen3.5 27B** 模型在编程方面的得分与 **122B** 模型相同，甚至在 Agentic 指数上胜出 2 分，尽管后者的体积更大。
   - 有人澄清说 **27B** 并非 MoE 模型，这与 **122B** 和 **35B** 版本不同。
- **关于 AI 生成内容版权的辩论**：继最高法院对 AI “艺术”的立场之后，引发了一场关于 AI 生成的代码应该是开源还是闭源的讨论，一位用户认为由于它不是人类创作的，因此不应享有版权。
   - 其他人则指出执行此类规则的不切实际性，以及对开发 AI 工具（特别是编程领域）商业激励的潜在影响。
- **LM Studio 插件库需求量大**：社区成员表示需要为 LM Studio 插件建立中央仓库和简化的安装流程，类比了 ComfyUI Manager 的自定义节点系统，参考 [DuckDuckGo LM Studio Plugin](https://lmstudio.ai/danielsig/duckduckgo)。
   - 目前，寻找插件涉及手动搜索和下载，像 *FrankTheGlock* 这样的用户推荐了 [Exa MCP](https://github.com/exa)。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1479218567718244404)** (133 条消息🔥🔥): 

> `Qwen3.5-35B 在 LM Studio 中 Prompt 处理缓慢, LM Studio 的 NPU 支持, LM Studio 的多 GPU 利用, Mac 的能效对比服务器的速度/成本, Minisforum AI NAS 作为 AI 任务的理想平衡点` 


- **Qwen3.5-35B 受困于 Prompt 处理缓慢的 Bug**：用户报告称，由于缓存清理问题，**LM Studio** 中的 **Qwen3.5-35B** 模型在 Prompt 处理时速度显著变慢，影响了对话速度，详见 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ren7l2/slow_prompt_processing_with_qwen3535ba3b_in_lm/)。
- **LM Studio 拒绝支持 NPU**：由于资源限制以及认为 **NPU** 在本地 LLM 推理方面不如传统 **GPU** 具有吸引力，**LM Studio** 目前不支持 **NPU**。
   - 一位成员提到，与 **FastFlowLM** 等公司合作可能会实现 **NPU** 支持，但团队认为这是低优先级任务。
- **多 GPU 支持：增加上下文，而非提升性能**：**LM Studio** 支持多 GPU 设置，可进行*一键*配置，这主要用于通过增加可用 **VRAM** 来处理更大的上下文，而不是提升整体性能。
- **Mac 的统一内存优于消费级 PC？**：讨论围绕 **Mac** 因统一内存（Unified Memory）而带来的能效优势展开，统一内存提供比典型 **RAM** 更快的速度，尽管一旦 **VRAM** 耗尽性能就会下降。
- **Minisforum AI NAS 有望成为理想选择**：**Minisforum AI NAS** 被强调为一种潜在的理想解决方案，它结合了模型存储空间、用于 egpu 集群的 **PCIE** 以及一个用于 Offloading 的不错 **AI 芯片**，使其成为处理各种 AI 任务的通用选择。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1479207007927009565)** (628 条消息🔥🔥🔥): 

> `GPT 5.4, Comet Browser, Gemini 3 Flash, Perplexity Pro Abuse` 


- **GPT-5.4 Thinking 作为推理模型广受好评**：成员们报告称 **GPT-5.4 Thinking 作为推理模型的写作能力非常出色**，相比其前代 **5.2** 有了 *跨越式* 的进步。
   - 一位用户指出，在处理情感内容和社交动态方面，它就像是一个 *接地气版的 Gemini*，表现比 5.2 好得多。
- **Comet 浏览器面临劫持！**：在有[报告称](https://cybersecuritynews.com/perplexitys-comet-browser-hijacked/) **Perplexity Comet 浏览器**被劫持后，该浏览器正面临严格审查。
   - 一些用户在移动端版本上也遇到了问题。
- **Gemini 3 Flash 消失，Pro 取而代之**：成员们注意到 **Gemini Flash 消失了**，但由于 **Gemini 3.1 Pro 表现更好**，且两者的成本相同，他们认为没有必要保留 Flash。
   - 一些人注意到 **Opus 也不在模型列表里了**，但无法完全确认。*Grok tbhmaybe* 也消失了。
- **Pro 用户指责 PPLX 存在滥用行为**：Pro 用户对所谓的掠夺性措施表示不满，理由是深度搜索（deep research）查询次数被削减、文件上传限制，以及从 [2025 年 11 月和 2026 年 2 月](https://www.reddit.com/r/perplexity_ai/comments/1opaiam/perplexity_is_deliberately_scamming_and_rerouting/)开始的静默模型替换。
   - 一位用户惊呼，这使他们的使用量 *减少了 90% 以上*，而他们当初签署 *年度计划* 时，横幅上赫然写着“无限使用”。
- **StegCloak 破解 Perplexity 的 Comet 邀请谜题**：在看到一串 Unicode 字符 \u{200C}\u{200D}\u{200C} 后，一位用户请求协助解码另一位用户分享的 Comet 邀请谜题。
   - 一位精通技术的社区成员指出，使用 **StegCloak** 并配合解密密码 'perplexity'，可以揭示 *这些特定隐形字符的组合*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1479237233243586621)** (5 条消息): 

> `Student Discord Server, GPU and LLM Pricing Dashboard, Computer autocomplete NPM packages` 


- **有 VIP 背景支持的新学生 Discord 服务器**：一位成员正在为学生建立一个 Discord 服务器，用于分享技巧和学习工具，并得到了 **Duolingo 高管** 的支持，涵盖编程和 AI 工作流等主题，分享地址为 [outsmartdiscord.com/education](https://outsmartdiscord.com/education)。
- **实时 GPU 价格仪表盘发布**：一位成员构建了一个免费的仪表盘，用于跟踪各云服务商和推理提供商的实时 **GPU 和 LLM 价格**，访问地址为 [deploybase.ai](https://deploybase.ai)。
- **Computer 精通 NPM 自动补全**：一位成员赞扬了 **Computer** 能够完美地自动补全 NPM 包名/版本，并为过时的包提供视觉指示，创建了一个结构良好的项目，可在 [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=VoidWorks.trawl) 上找到。
- **Qwksearch 支持 Perplexity Key**：[Qwksearch](https://qwksearch.com) 现在允许用户携带自己的 Perplexity API key。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1479214544168554607)** (478 条消息🔥🔥🔥): 

> `GPT 5.4 速度缓慢，GPT 5.4 价格 Max 模式，Cursor 索引仓库时出现 OOM 崩溃，从 Windsurf 升级到 Cursor，Cursor subagents` 


- **GPT 5.4 蜗牛般的速度惊扰了效率玩家**：成员们注意到 **GPT 5.4** 明显比其他模型慢，有用户即便在付费订阅状态下，也要等待 **30 分钟** 才能完成一个任务。
   - 一些建议包括调整规则以优先读取文件、降低 reasoning 级别，甚至在 [sandbox environment](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) 中运行 agent，以减轻来自恶意命令的风险。
- **GPT 5.4 价格策略受困于强势的 “Max 模式” 窘境**：用户对 **GPT 5.4** 仅在 “Max” 模式下可用感到不满，认为 Cursor 通过要求 Max 模式来使用其 **1M context window**，从而迫使用户放弃旧版定价。
   - 关于 **context windows** 和 “Max” 模式存在很多困惑，有人认为它只有 **270k context window**。
- **Cursor 在代码库压缩期间的崩溃引发关注**：一名用户在打开特定仓库时遇到了持续的 **OOM crashes**，怀疑是仓库索引损坏或在 **repo-level indexing** 期间发生了内存泄漏。
   - 故障排除步骤包括清理 `.cursor` 和 `.cache` 目录、重新安装 Cursor、增加 Node 内存和 Windows 页面文件，以及添加严格的 `.cursorignore` 规则。
- **从 Windsurf 到 Cursor：一阵清爽之风？**：一名使用 Windsurf 一年的用户发现 Cursor 令人*耳目一新*，并强调其错误更少、工作流更高效。
   - 他们提到 Windsurf 注入了大量系统提示词（system prompts），导致了各种问题，而 Cursor 让他们能够*真正完成工作*。
- **Subagent 恶作剧：Composer 的消耗担忧**：用户注意到 Cursor 中内置的 subagents 会自动使用 **composer** 模型，这会消耗 token，而且有时并非用户所愿。
   - 一个建议是创建自定义 subagents 以指定首选模型，并使用 `/create-subagent` 命令。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1479209839606628466)** (3 条消息): 

> `CoT 可控性，Codex Security，面向 OSS 的 Codex` 


- **思维链 (CoT) 获得新的评估套件**：OpenAI 正在发布关于 **Chain-of-Thought (CoT) Controllability** 的新评估套件和研究论文（[论文链接](https://openai.com/index/reasoning-models-chain-of-thought-controllability/)）。
   - 研究表明，**GPT-5.4 Thinking** 在隐藏其推理过程方面能力较低，这表明 **CoT monitoring** 仍然是一个有用的安全工具。
- **Codex Security：新安全 Agent 推出**：OpenAI 推出了 **Codex Security**，这是一个应用安全 agent，旨在通过发现并验证漏洞并提出修复建议，帮助保护代码库安全（[公告链接](https://openai.com/index/codex-security-now-in-research-preview/)）。
   - 这使团队能够专注于关键漏洞并加速代码部署，正如[此演示视频](https://video.twimg.com/amplify_video/2029983742056615937/vid/avc1/1280x720/sx7Je_FzQPJAr81B.mp4)所示。
- **面向 OSS 的 Codex 将支持开源贡献者**：OpenAI 推出了 **Codex for OSS** 以支持开源软件贡献者（[公告链接](https://developers.openai.com/codex/community/codex-for-oss)）。
   - 维护者可以利用 **Codex** 来审查代码、理解大型代码库并增强安全覆盖，详情见[此演示视频](https://video.twimg.com/amplify_video/2029998126640287747/vid/avc1/1280x720/ZMdqbgIfCNQeqJ0i.mp4)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1479208009090732245)** (311 条消息🔥🔥): 

> `ChatGPT 与 Claude 和 Kimi 的性能对比，GPT-5.4 Codex，Seedance 2.0 延迟，对政府控制 AI 公司的担忧，OpenAI 估值过高` 


- **ChatGPT 掉队：社区表示担忧**：成员们担心 **ChatGPT** 正在落后于 **Claude** 和 **Kimi** 等其他 LLM，一位用户表示：*"ChatGPT 怎么了，感觉它开始落后于 Claude 甚至 Kimi 等其他 LLM 了" 。*
   - 一位成员指出：*"Kimi 2.5 远超 ChatGPT 的能力，甚至是 K2 推理模型 (thinking model)。"*
- **GPT-5.4 的 Codex 表现欠佳**：用户报告称 **GPT-5.4** 的 **Codex** 在编程任务中的表现不如 **5.3**，一位用户指出他们 *"不认为我们拿到了真正的 5.4 codex"*。
- **Seedance 2.0 面临版权审查**：**Seedance 2.0** 的全球发布被推迟且遭到削弱，据称是因为用户发布了包含 IP/版权角色的视频，导致 ByteDance 面临诉讼。
   - 一位成员表示 *"另一方面，Seedance 2.0 最终将在全球发布！"*，并提到最初预计在 **2 月 24 日**发布。
- **AI 军备竞赛：自主权与问责制**：讨论围绕政府对私营 AI 公司的控制展开，以及 OpenAI 签署了一份防止战争罪行和大规模国内监控的合同，Anthropic 也曾希望签署此类合同。
   - 一位用户声称，政府拒绝了 Anthropic 的 *"即使法律变更"* 条款，引发了对未来政府可能过度干预的担忧。
- **OpenAI 的估值：是节节高升还是空中楼阁？**：一位用户对 OpenAI 的估值表示怀疑，指出它 *"作为一个估值 7200 亿的公司，计划（且可能无法实现）到 2029 年实现 20 亿的年利润"*，暗示可能存在估值过高的情况。
   - 另一位成员调侃道：*"老实说，除了人，一切都被高估了。人的价值被严重低估了。"*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1479207249510404319)** (106 条消息🔥🔥): 

> `GPT-5.4 原生计算机使用 (Native Computer Use)，GPT-5.4 与 5.3 和 5.2 的性能及可控性对比，OpenAI 产品发布的奇怪之处，GPT 图像生成问题，ChatGPT 聊天变慢并变得无法使用` 


- **GPT-5.4 具备原生计算机使用能力**：成员们讨论了 **GPT-5.4** 的 *原生计算机使用能力 (native computer-use capabilities)* 意味着什么，一位成员解释说它 *可以接管并在你的电脑上执行操作*，类似于 **Claude Code** and **Cowork**。
- **GPT-5.4 的速度和可控性赢得赞誉**：用户赞扬了 **GPT-5.4** 的速度、可控性（steerability）以及改进的回答，特别是对于需要长上下文理解的文本工作，有些人甚至认为它优于 **5.2** 和 **5.3**。
   - 一位用户指出 *模型的回答就像在跟真人交流，而不是像它脑子里有幻听一样（指 5.3）*。
- **用户对定价和 Mini 模型表达不满**：用户对最近的价格上涨表示担忧，并表达了希望发布 mini 模型的愿望。
   - 一位用户说：*“对涨价不太满意。是时候推出 mini 模型了。”*
- **图像生成表现平平**：一位用户报告称，**GPT** 现在使用 *像素编辑模式 (pixel-editing mode)* 进行图像生成，这阻碍了它执行简单任务的能力，例如向图像中添加雪景。
   - 他们询问了是否存在 API 或其他方法来通过 *重绘模式 (repaint mode)* 进行图像生成。
- **ChatGPT 聊天体验卡顿**：用户抱怨 **ChatGPT** 的聊天速度随着时间的推移显著变慢，导致无法使用。
   - 一位用户认为问题源于 **ChatGPT** 缺乏自动聊天压缩（chat compaction）功能，而不像 **Claude** 以及可能的 **Gemini** 那样具备此功能。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1479206939832487936)** (20 messages🔥): 

> `Image Generation prompts, Image generation API for repaint mode, GPT Evaluation of Papers, Prompt engineering courses, Accelerated Iterative Destruction` 


- **Prompt 揭秘骷髅小孩推车**: 一位成员分享了一个用于生成 **3D CGI** 渲染的瘦弱人类小孩的 Prompt，其具有 **translucent skin** (半透明皮肤) 和可见的 **cyan skeleton** (青色骨骼)，正在推着一辆生锈的老爷车。
   - 他们指出，某些模型可能缺乏关于 **All-Might**（出自《我的英雄学院》）的上下文，因此需要留意这一点。
- **图像生成模式从 repaint 切换到 pixel-editing 模式**: 成员们讨论到，**GPT** 以前使用的是 **repaint mode**，但现在使用 **pixel-editing mode**，这导致它无法完成某些简单的任务。
   - 他们对 **Sora** 仍在使用 **repaint mode** 表示欣慰，但注意到 Sora 1 即将停用。
- **GPT 无需训练即可评估论文**: 成员们建议，不需要训练 **GPT** 也能根据 Rubric 评估论文。
   - 相反，*只需在 Prompt 中提供 Rubric，并要求它分别对每个类别进行评分*。
- **AI 工程师分享 Prompt engineering 方法论**: 一位成员分享了 Prompt engineering 的方法论，称之为 **Accelerated Iterative Destruction** 和 **Constraint pattern recognition**。
   - 他们将前者描述为 *故意破坏系统以使其变得更强大*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1479206939832487936)** (20 messages🔥): 

> `Image Generation prompts, Prompt Engineering Courses, Training GPTs` 


- **在 3D CGI 中打造半透明皮肤**: 一位成员分享了一个 [Prompt](https://discord.com/channels/974519860457529424/1046317269069864970/1247310292513685544)，用于生成 **3D CGI 渲染的瘦弱人类小孩**，具有半透明皮肤和青色骨骼，穿着不透明的黑色短裤和黑色 T 恤，在车后推着一辆生锈的老爷车，而 **3D CGI 渲染的 All-Might** 站在背景中拿着剪贴板记录，电影感光效，城市街道背景。
   - 他们指出，*translucent 或 glass like skin 是实现预期效果的关键描述词*。
- **在 ChatGPT 中开启图像功能**: 一位成员询问如何激活 **ChatGPT 中的图像功能**，包括为什么它会断断续续地出现，以及图像是否可以附带解释。
   - 另一位成员简要说明，只需在 Prompt 开头输入 *"Create an image:"* 即可激活该功能。
- **GPT 的 Pixel-Editing 与 Repaint 图像生成模式**: 一位成员强调，**GPT 现在使用 pixel-editing mode** 进行图像生成，而不像过去使用 **repaint mode**，且 GPT 现在无法执行许多简单的任务，例如向图像中添加雪景。
   - 该成员对 **Sora 1 即将停用** 感到难过，因为他们仍在使用 repaint mode。
- **寻找圣杯：Prompt Engineering 课程**: 一位成员寻求最佳 **Prompt engineering 课程** 的推荐，但得到的却是 **Accelerated Iterative Destruction** 和 **Constraint pattern recognition** 等方法论。
   - 这些方法论是用于发现系统崩溃点的框架，并以崩溃后的状态命名，即 *Coherence, Relational Invariance, Internal Mediation, Projection*。
- **训练 GPT 进行论文评估**: 一位成员询问关于 **训练 GPT 根据 Rubric 评估论文** 的问题。
   - 其他人建议只需在 Prompt 中上传论文和 Rubric，并要求它分别对每个类别进行评分，并在可能的情况下对评分给出理由。

  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1479564705860157480)** (12 messages🔥): 

> `Tech Industry Complacency, AI Agent Database Wipe, Compute Conference Tickets` 


- **Thorsten Ball 抨击科技行业的懈怠**：Thorsten Ball 批评了科技行业**缺乏紧迫感**，观察到尽管 **AI** 和团队效率飞速进步，许多公司仍在使用过时的运营模式；该贴文可见于 [此处](https://x.com/thorstenball/status/2029846505884901873?s=12)。
- **Spacemolt 角色正在编写剧本！**：一名成员正在扩展系统以允许 PM 发布代码，并让其 **Spacemolt** 角色开始编写剧本，详见 [此 Google 文档](https://docs.google.com/document/d/1Lv6nGH930Rurqp_FkLNv-XmwuS-XX7s3uRXPBT7I9QI/edit?usp=drivesdk)。
- **Claude Code 的数据库灾难**：Alexey Grigorev 讲述了 AI Agent **'Claude Code'** 意外执行了一个 Terraform 命令，导致 **DataTalksClub 生产数据库** 和 **2.5 年的课程数据** 被清空，详情见 [此处](https://x.com/al_grigor/status/2029889772181934425)。
- **免费 Compute Conference 门票**：在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 可获得三张 **Compute Conference** 赠票。


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1479684765652095078)** (3 messages): 

> `Tech Companies Stock Incentives, Resignation After Bonuses` 


- **股票激励挤压科技公司**：一位成员发布了一张图片，认为 **科技公司无法负担保留已给予股票激励的员工**，引发了关于财务影响的讨论。
   - 另一位成员建议这种情况可能影响了 **Block**，而其他公司可能需要将自由现金流引导至数据中心建设的资本支出。
- **奖金事与愿违：员工在获得留任奖金后辞职**：一位成员分享了一篇 **LinkedIn 帖子**，详细描述了一名员工在得知留任员工将获得丰厚奖金后，**提交了辞呈**。
   - 未提供有关该公司或辞职具体情况的更多细节。


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1479596720001257704)** (2 messages): 

> `Creator Economy, Cross-Platform Storytelling` 


- **创作者经济的“西部荒野”**：一名用户发布了一张标记为 *wild*（未给出上下文）的图片，链接在 [此处](https://cdn.discordapp.com/attachments/822625128843182090/1479596719762178169/image0.jpg?ex=69ac9d53&is=69ab4bd3&hm=62b3b5bb200d487a6351ebb520d4c766e189c237c40be128ffa4d37b18af008c&)。
- **创作者经济的全貌**：一名用户评论说，除非*你也计算其他平台*，否则 **创作者经济的完整故事** 并没有被讲述完。


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1479564624541253764)** (24 messages🔥): 

> `Product launch videos, Venting illustration goes viral, AI development tools comparison, AI agent deletes production database, Tweet of the year contender` 


- **发布视频千篇一律**：Manu Arora 在 [这条推文](https://x.com/mannupaaji/status/2029882202801221892?s=12) 中质疑了当前 **产品发布视频** 的设计和审美趋势，指出整个行业呈现出重复或公式化的风格。
- **Slaylor 的插画在吐槽中走红**：用户 @GirlSnailure (Slaylor) 分享了他们在遇到挡路者后，为了宣泄挫败感而创作的一件创意作品，随后在 [这条推文](https://x.com/girlsnailure/status/2029622733865185657?s=12) 中获得了显著的病毒式传播。
- **Claude Code 笨拙地清除了课程内容**：Alexey Grigorev 在 [这条推文](https://x.com/al_grigor/status/2029889772181934425?s=12) 中报告称，**Claude Code AI Agent** 意外通过 Terraform 命令删除了 **DataTalksClub 生产数据库** 及其自动快照。
- **Harry Eccles 的“年度推文”大获成功**：Harry Eccles (@Heccles94) 于 2026 年 3 月发布的一条备受关注的 Twitter 贴文，提出了这是否构成“**年度推文**”的问题，在 [这条推文](https://x.com/heccles94/status/2029973065954668969?s=12) 中获得了超过 **67,000 个赞** 和 **785,000 次观看**。
- **哲学 Meme 贴文广为流传**：账号 @philosophymeme0 于 2026 年 3 月 6 日发布的一条病毒式社交媒体贴文，在 [这条推文](https://x.com/philosophymeme0/status/2029925357294604573?s=12) 中获得了显著参与度，拥有超过 **6,500 个赞** 和近 **80,000 次观看**。


  

---

### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1479232806641991862)** (4 messages): 

> `$BE stock, saeris.gg` 


- **$BE 股票引发关注**：一名成员一直在关注 **$BE 股票**，并引用了 [saeris.gg](https://saeris.gg) 以及[两条相关的推文](https://vxtwitter.com/josephpolitano/status/2029916364664611242)和另一条[推文](https://vxtwitter.com/byheatherlong/status/2029918420821758134)。
- **关于 $BE 的简短讨论**：另一名成员针对该股票报告回复了 *lol*。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1479225233096577076)** (10 messages🔥): 

> `AI Consulting, Agentic AI, Production ML Systems, Open Claw for GTM, LLM Engineering Platform` 


- **新的 LLM 工程平台发布**：Soren 宣布推出 [to11.ai](https://to11.ai)，这是一个提供可观测性、Prompt 管理、网关服务和安全功能的 **LLM 工程平台**。
- **Open Claw 为 GTM 寻找信号**：Steve 正在开发一个用于 **GTM 的 "Open Claw"**，它能深度理解产品以执行特定策略，例如 GEO 优化和在 LinkedIn 上寻找 ICP。
   - 它也会在 **Reddit/X 上搜寻信号帖子**。
- **Agentic AI 公司专注于高管决策**：Debo 创办了一家专注于高管决策的 **Agentic AI 公司**。
   - 他来这里是为了了解更多关于**真实使用案例**的信息。
- **编排器扩展 Vanilla 代码的应用**：一位成员正在 *编写一本关于工程中规模化应用 AI 的书*，主持 O'Reilly CTO Hour，每年两次为 CNCF 在 KubeCon 组织高管峰会，在纽约、湾区和线上为创始、初创和规模化 CTO 举办 Gather.dev 活动，并且（像世界上其他人一样）正在构建自己的编排器（orchestrator）来运行业务、研究书籍并整理生活。
   - 他们使用 Vanilla Claude 代码，单仓库（single repo）结构并设有共享上下文空间，为每个 **employee** Agent 设有一个包含其 Prompt 和唯一上下文的目录，并为每个 **advisor** Agent 设有另一个目录。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1479215172961697833)** (23 messages🔥): 

> `Web Middleware Parallelization, PlanetScale Latency, TanStack Intent for AI Agents, Steam Hardware Delays and Exabyte Traffic` 


- **Web 中间件：并行身份验证检查？**：成员们讨论了一种并行 Web 中间件概念，用于在渲染的同时并发运行 **auth/访问控制** 检查，如果身份验证失败，则可能停止渲染。
   - 然而，有人对分离 UI 树导致的复杂性增加以及渲染期间潜在的副作用表示担忧，一位成员将该设计与 **Next.js** 激进的并行化联系起来，认为这会导致认知负担（cognitive footgun）。
- **PlanetScale：新的数据库延迟之王？**：一位用户分享了他们[从 **AWS** 迁移到 **PlanetScale** 后的性能提升](https://xcancel.com/fforres/status/2029661853731934629?s=20)，展示了延迟从 **255ms 降至 10ms**。
   - 其他人回应称，通过私有网络连接的**单个数据中心内的机器**可以实现 **0.1ms 的延迟**，并调侃说*现在 10ms 的不稳定数据库延迟也值得吹嘘了*。
- **TanStack 计划交付 AI Agent 技能**：**TanStack** 宣布了 [Intent (alpha)](https://xcancel.com/tan_stack/status/2029973163455766769)，这是一个直接在 npm 包中交付 **AI Agent 可读“技能（skills）”**的流水线。
   - 该系统有助于实现分布式的、自动发现的、最新的知识同步，使其与所有主流包管理器的库更新保持同步。
- **Valve 因艾字节（Exabyte）级流量延迟 Steam Machine 计划**：**Valve** 的“年度回顾”博客文章指出，他们*希望在今年某个时候交付* **Steam Machine** 和其他已公布的硬件，这可能是由于 **RAM 短缺**导致的。
   - 该文章透露，**Steam** 在 2024 年向客户交付了约 **80 exabytes** 的数据，2025 年将增长到 **100 exabytes**，平均每天有 **274 petabytes** 的安装和更新，相当于每分钟 **190,000 GB** 的数据（[来源](https://steamcommunity.com/groups/steamworks/announcements/detail/528746884222682053)）。


  

---

### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1479259576913100871)** (10 messages🔥): 

> `Y Combinator's Startup School, Compute Conference, AI Infrastructure, Developer Tooling` 


- **YC Startup School 依然表现亮眼**：一位成员深情回顾了 [Y Combinator's Startup School](https://events.ycombinator.com/startup-school-2026)，并指出它对自己生活的影响。
   - 他们承认虽然没有充分利用这次机会，但认为它显著改变了自己的生活，且其在线资源仍然非常有帮助。
- **Daytona 的 Compute 大会**：**Daytona** 将举办 **Compute** 大会，这是一个专注于 **AI infrastructure**、**agents** 以及下一代 **cloud** 的会议，将于 **3 月 8-9 日** 在 **San Francisco** 的 **Chase Center** 举行 ([Compute Daytona](https://compute.daytona.io/))。
   - 演讲嘉宾包括来自 **Box** 的 **Aaron Levie**、来自 **Parallel** 的 **Parag Agrawal** 以及来自 **LangChain** 的 **Harrison Chase** 等，目标受众是 **AI infra** 和 **developer tooling** 领域的工程师、创始人及开发者。


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1479607046184636537)** (1 messages): 

> `GitHub Social Club, Amsterdam Event` 


- **GitHub 在阿姆斯特丹举办社交聚会**：GitHub 将在 **3 月 23 日星期一**（Kubecon + CloudNativeCon 和 AgenticDays 之前）举办 **GitHub Social Club: Amsterdam**。
   - 根据 [活动页面](https://luma.com/githubsocialclub-amsterdam) 的介绍，该活动是 *面向开发者、构建者、研究人员、创始人和开源粉丝的低调聚会*，并承诺没有推销（no pitches），提供一个交流和分享想法的空间。
- **GitHub Swag 预警**：参加阿姆斯特丹 **GitHub Social Club** 的人员将获得 **GitHub swag**。
   - 活动承诺提供咖啡、零食以及与 GitHub 团队成员见面的机会，是建立社交网络的好机会。


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1479533590130851872)** (1 messages): 

> `NYC Meetup, Google NYC Hosting, Talks from Google, Modal, and others` 


- **Google NYC 再次举办 Meetup**：一位成员宣布他们正在组织一场将在几周内举行的 Meetup，由 **Google NYC** 主办，届时将有来自 **Google**、**Modal** 以及组织者雇主的演讲；详情和报名请见 [Luma](https://luma.com/7qxvd38s)。
   - 未提供更多细节。
- **多样化的技术公司展示**：该 Meetup 承诺通过来自 **Google**、云计算平台 **Modal** 以及主办成员公司的演讲者，带来多样化的技术视角。
   - 每个演讲的具体主题和重点尚待公布，引起了社区内的期待。


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/1479691298049884271)** (3 messages): 

> `Backdoored Training Data, Alexander Long Tweet` 


- **Alexander Long 的推文走红**：一位成员分享了 [Alexander Long 的推文](https://x.com/alexanderlong/status/2030022884979028435?s=12) 链接。
   - 另一位成员猜测是否 *有人在他们的训练数据中植入了后门*，或者是一些更露骨的情况。
- **关于训练数据后门的猜测**：顺着推文链接，一位成员询问了关于训练数据被植入后门的可能性。
   - 该成员质疑问题是由于 *后门* 还是 *更露骨的东西* 导致的。


  

---

### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1479206949026267349)** (98 messages🔥🔥): 

> `CSS dark/light mode, Trump's White House UFC Stadium Proposal, Iran-Saudi relations, Palantir's Maven Smart System, Anthropic AI contract with Department of War collapse` 


- **CSS Debates Spark Over Dark Mode Implementation**: A discussion started over Twitter's move to OS-controlled dark mode, with members debating the need for separate assets vs. CSS variables for palette swaps and ways to counteract **blooming effects** in dark mode.
   - One member recommended using the `light-dark()` CSS syntax with CSS variables to combine light and dark mode color pairings, as shown in [this article](https://web.dev/articles/light-dark), and another shared his sentiment *"anytime they do shit like this it makes me wonder, did Elon mandate this change? Or is it because Grok produces absolute slop?"*
- **Trump Plans White House UFC Stadium**: Reports indicate **Donald Trump** plans to build a **100,000-seat stadium** near the White House to host a **UFC event** on his birthday in **June 2026**.
   - The proposal, originally shared in [this tweet](https://xcancel.com/highbrow_nobrow/status/2029497418325086488), was met with mockery and sarcastic remarks.
- **US Investigation Points to Likely Responsibility for Iran School Strike**: A US investigation suggests likely US responsibility in an **Iran school strike**, amid rising tensions and skepticism regarding the US's ability to defend its allies from Iran, according to [this Reuters article](https://www.reuters.com/world/middle-east/us-investigation-points-likely-us-responsibility-iran-school-strike-sources-say-2026-03-06/).
   - Some members pointed out that the region is very upset with the US and cited macro analysis suggesting the potential of investment withdrawal from gulf countries, based on this [YouTube analysis](https://www.youtube.com/watch?v=jIS2eB-rGv0).
- **Department of War and Anthropic AI Partnership Collapses**: An article shared [here](https://xcancel.com/piratewires/status/2029984469093118185?s=12) details how a major contract between the **Department of War** and **Anthropic AI** fell through due to restrictive terms prohibiting kinetic strikes, long ethics panel reviews, and concerns about ideological supply-chain risks.
   - The member satirically noted *"seems like open ai is ahead of anthropic in vibe warcrime"*.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new Cursor pod! https://www.latent.space/p/cursor-third-era
  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1479225578329473191)** (117 messages🔥🔥): 

> `Multi-Agent Orchestration, Claude Code Reverse Engineering, Greptile Agent v4, Cursor Automations, ChatGPT for Excel` 


- **Claude Code Cracked for Context Control**: A developer reverse-engineered the **Claude Code binary** to implement a surgical context management feature, allowing users to selectively strip tool calls, results, and thinking blocks while preserving the core message history, as detailed [on xcancel.com](https://xcancel.com/vicnaum/status/2029579972688379928).
- **Greptile Agent v4 Slashes Bugs, Hikes Prices**: **Daksh Gupta** launched **Greptile Agent v4**, boasting improved bug detection and fewer false positives, but with a revised pricing structure aimed at power users, as seen [on xcancel.com](https://xcancel.com/dakshgup/status/2029587555268845692?s=12).
   - A user commented that *those prices are eye-watering!*.
- **Cursor Automates Always-On Agents**: **Cursor** unveiled **Cursor Automations**, a new feature to create and deploy persistent, always-on AI agents within the platform, according to [xcancel.com](https://xcancel.com/cursor_ai/status/2029604182286856663?s=12).
- **Sarvam AI Drops Indian Language Models**: **Pratyush Kumar** announced the release of the **Sarvam 30B and 105B models**, trained from scratch to excel in Indian languages and global benchmarks, as detailed [on xcancel.com](https://xcancel.com/pratykumar/status/2029965547824431356?s=46&t=b7l37rB6wtbyAh6ah1NpZQ).
   - Weights are available on **Hugging Face** and **AIKosh**, with **SGLang** providing launch day support, and **vLLM** integration expected soon.
- **GitHub Bot Gets Promptly Hacked**: **Sash Zats** reported a security breach where an attacker obtained an npm token using a prompt injection in a GitHub issue title, exploiting a triage bot, as detailed [on xcancel.com](https://xcancel.com/zats/status/2029888470383051053?s=12).


  

---




### **Latent Space ▷ #[berlin](https://discord.com/channels/822583790773862470/1095237457722744932/1479607060319699087)** (1 messages): 

> `GitHub Social Club, Amsterdam Events, Kubecon, CloudNativeCon, AgenticDays` 


- **GitHub Social Club 即将登陆阿姆斯特丹**：GitHub 将于 **3 月 23 日星期一**举办 **GitHub Social Club: Amsterdam** 活动，紧邻 **Kubecon + CloudNativeCon** 和 **AgenticDays**。
   - 该活动是开发者、构建者、研究人员、创始人以及开源爱好者的低调聚会，旨在建立联系并分享想法，[点击此处预约 (RSVP)](https://luma.com/githubsocialclub-amsterdam)。
- **开发者的社交机会**：GitHub Social Club 为开发者提供了一个空间，让他们能够与社区中的其他人建立联系、分享想法并交流故事。
   - 与会者可以享用咖啡、零食，获取 GitHub 周边 (swag)，并有机会与 GitHub 团队成员会面。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1479261057892614216)** (12 messages🔥): 

> `Reasoning Models, Structured Checklist Method, AI Koding` 


- **用于控制的推理模型**：OpenAI 强调利用 [Reasoning Models](https://openai.com/index/reasoning-models-chain-of-thought-controllability/) 来提升 **思维链的可控性 (Chain of Thought Controllability)**。
   - 如果在使用 [COVAL Alignment 项目](https://alignment.openai.com/coval/) 进行评估标准最大化 (rubric maxxing)，这可能会非常有用。
- **Meta 的 Checklist 将错误率降低了 50%**：Meta 的研究人员发现，在**代码补丁验证 (code patch verification)** 中使用结构化的 Checklist 模板，可以在不进行额外 Fine-tuning 或架构更改的情况下，将错误率降低近 **50%**，详见[此推文](https://xcancel.com/alex_prompter/status/2029861760455569422?s=12)。
   - 该方法要求在得出结论之前，强制执行逐步的证据和推理过程，这可能会解决 AI Koding 的问题。
- **Databricks 发布用于定制 RL 的 KARL**：Databricks 推出了 **KARL**，这是一种更快的 Agent，用于企业知识驱动的定制化 RL，详见[此博文](https://www.databricks.com/blog/meet-karl-faster-agent-enterprise-knowledge-powered-custom-rl)。
   - 这使得在企业环境中实现更高效、更定制化的强化学习 (Reinforcement Learning) 应用成为可能。


  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1479297785688756435)** (61 messages🔥🔥): 

> `Codex App, GPT-5.4 Token Usage, AI-First OS, Prompt Engineering, 1M Context Usage` 


- **Codex App 导致 Token 消耗率翻倍**：成员反馈使用 **Codex App** 处理相同数量的 Token，会导致消耗速度快 **2 倍**。
   - 尽管成本增加，一些用户发现 **GPT-5.4 xhigh** 的速度明显快于 **5.2 xhigh**，不过对质量的评价各异；一位用户指出，*“5.4 似乎消耗 Context Window 的速度更快，当然，这只是感觉 (vibes)”*。
- **新型 AI-First OS 正在构建中**：一位用户正在*尝试在浏览器中重新编写一个基于 LLM 的操作系统*，并将其与 GitHub 上的 [wesen-os](https://github.com/wesen/wesen-os) 和 [workspace-links](https://github.com/wesen/wesen-os/tree/main/workspace-links) 关联。
   - 他们认为*我们正处于一个可以重新思考计算机一切的节点*，并且*打破过去抽象的束缚*。
- **公布即将到来的演讲者**：**AI In Action Bot** 宣布了即将到来的演讲者，包括 @slono（2026 年 3 月 6 日）演示 *“it's GO GO OS - THE AI FIRST OS”*，以及 @beeradley（2026 年 3 月 13 日）讨论 *“new Latent Space DB and Bot”*。
   - 机器人还提到了 Peter Bell 的日程安排在 3 月 20 日，但这需要用户的额外确认：*“Trace，如果你仍想参加，你需要回复机器人的问题直到它确认日期”*。
- **发掘 Prompt Engineering 的 Diamond Tier 技巧**：成员强调使用 Prompt *“proceed”* 是 **Diamond Tier** 的效果，而 *“gitrdun”* 则被认为是 **Mud Tier**。
   - 一位用户建议使用更复杂的 Prompt：*`proceed until completed and verified`*，但另一位用户指出，*我怀疑压缩 Prompt 的细微变化会导致它在传递过程中丢失这类信息*。
- **揭秘 Context 限制配置技巧**：一位用户询问关于使用 **1M Context Window** 的问题，另一位用户分享了一个配置技巧，通过修改 `.codex/config/toml` 中的 `model_auto_compact_token_limit = 960000` 来提高限制。
   - 他们确认此配置更改在他们的设置中（推测是 'pi' 环境中的 Codex）生效。


  

---

### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1479212149753647199)** (14 messages🔥): 

> `Arksim 用于 Agent 测试, Agent 之间的 Slack, Cursor Cloud Agents, 用于多 Agent 集群（swarms）的 Reads 记忆层, 为 Agent 设计的加密去中心化记忆` 


- **Arksim 开源 Agent 自动评估工具**: 一个名为 **Arksim** 的新工具已开源，用于生成合成用户，自动与你的 Agent 进行对话，以[填补手动测试用例的空白](https://github.com/arklexai/arksim)。
   - 该工具旨在真实用户遇到问题之前发现故障，可通过 `pip install arksim` 安装，并有[在线文档](https://docs.arklex.ai/overview)可供查阅。
- **Agent 现在有了可以互相争论的 **Slack****: “面向 Agent 的 Slack”早期版本已发布，使 Agent 能够像真正的同事一样在 [ats.sh/new](https://ats.sh/new) 互相争论。
   - 目标是模拟混乱但富有成效的交互，让 Agent 能够协作“解决问题”。
- **Cursor 进入云时代**: 关于 [Cursor 第三时代：Cloud Agents](https://youtu.be/tMflcZHo2zI) 的讨论，强调了 Agent 产出更多代码如何通过并行运行和对比实现，从而导致代码生成的指数级增长。
   - 视频展示了杰文斯悖论（Jevons paradox）的实际应用，证明了代码产量的增长与 Agent 能力呈正相关。
- **用于科学领域多 Agent 集群编排的 Reads 记忆层**: 开发了一个名为 **Reads** 的记忆层，旨在辅助多 Agent 集群（swarms）编排科学研究任务，并提供了 [GitHub 仓库](https://github.com/reads-project/reads-ts)。
   - 预计很快会发布带有前端的完整 Demo，有效保留高计算量的输出结果。
- **ElectricSQL 为 Vibe Coding 推出 Agent SKILLs**: **ElectricSQL** 为 Electric & Durable Streams 客户端和 TanStack DB 引入了 Agent SKILLs，增强了 “Vibe Coding” 体验，使开发者能够快速生成无错误的应用，最初分享于 [X](https://xcancel.com/kylemathews/status/2030058969822367784)。
   - 此次更新重点在于允许通过单次代码生成来构建复杂的应用程序。


  

---


### **Latent Space ▷ #[san-diego-neurips-2025](https://discord.com/channels/822583790773862470/1335732885717651558/1479605627729743993)** (2 messages): 

> `` 


- **未发生讨论**: 提供的消息中没有可总结的讨论内容。
   - 用户表达了错过某些事情的遗憾，但未提供具体背景。
- **无内容可总结**: 提供的文本由不完整的句子组成，缺乏实质性内容。
   - 因此，无法提取出有意义的主题或讨论点。


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1479308885553840229)** (3 messages): 

> `Ben Affleck AI 视频初创公司, Netflix 收购, Interpositive, ComfyUI` 


- **Affleck 的 Interpositive 被 Netflix 收购**: Ben Affleck 自 **2022年** 以来一直经营着一家名为 **Interpositive** 的 **AI 视频初创公司**，该公司刚刚被 [Netflix](https://about.netflix.com/en/news/why-interpositive-is-joining-netflix) 收购。
- **对 ComfyUI 普遍使用情况的质疑**: 在观看了一段简短采访后，一名成员询问 **ComfyUI** 是否正在被广泛使用。
   - 该成员试图确认 **ComfyUI** 是否已成为整个行业的标准工具。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1479299514102845470)** (4 messages): 

> `GPT 5.4 解决数学难题, Bartosz Naskrecki, Move 37, 科学奇点` 


- **GPT 5.4 实现数学领域的 “Move 37” 时刻**: 数学家 Bartosz Naskręcki 报告称，一个先进的 AI 模型 **GPT 5.4** 解决了他研究了二十年的一个问题，这促使他宣布 *科学奇点已经到来*。
   - 完整帖子的链接在 X 平台上的[此处](https://xcancel.com/trajektoriePL/status/2029660475395326300)。
- **数学家欢呼科学奇点**: 数学家 Bartosz Naskręcki 声称，**GPT 5.4** 对一个长期悬而未决问题的解决标志着科学领域奇点的到来。
   - 这一结论是基于该 AI 对 Naskręcki 潜心研究 **二十年** 之久的数学挑战给出的出人意料的解决方案。


  

---

### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1479591469214990338)** (4 messages): 

> `Far.AI, Neel Nanda, Empirical Interpretability, Activation Steering, AGI Safety` 


- **Far.AI 信号可解释性转向**：**Far.AI** 讨论了 **Neel Nanda** 向 **empirical interpretability** 的战略转变，详情见 [此推文](https://xcancel.com/farairesearch/status/2029957875523592524)。
- **Activation Steering 获得 Far.AI 认可**：重点已从抽象的见解转向**可测试的代理任务**和 **activation steering**，优先考虑那些在 **AGI safety** 方面具有可衡量影响的方法。


  

---


### **Latent Space ▷ #[dev-writers-retreat-2025-dwr](https://discord.com/channels/822583790773862470/1445650211694448714/)** (1 messages): 

xoxoxoxo42: 恭喜！！
  

---


### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1479267257103290481)** (2 messages): 

> `Breaks, Work-life balance` 


- **可能需要休息**：一位成员建议另一位成员由于工作量原因可能需要休息。
   - 上下文暗示了潜在的过度工作。
- **工作与生活平衡检查**：发起了一次检查，可能是为了评估工作量和压力水平。
   - 这表明团队内部对问责制和福祉的关注。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/)** (1 messages): 

kevin_85537: 太迷人了！
  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1479331447004336210)** (5 messages): 

> `AI Demo, AI in Action` 


- **AI Engineer 预告即将进行的 Demo**：一位 AI Engineer 宣布他们[计划在今晚展示一个 Demo](https://example.com/demo)，希望能将各种未完成的项目整合在一起。
   - *我希望我能把我那一堆半成品拼凑成一个好的 Demo。*
- **Demo 地点公布**：继初步公告之后，Demo 的地点将定于 *1 小时 30 分钟后在 ai in action 频道在线进行*。
   - 详情即将公布，敬请关注。


  

---


### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1479505361424879668)** (3 messages): 

> `GitHub Social Club: Amsterdam, Discord Stats Load Failure` 


- **阿姆斯特丹 GitHub Social Club**：GitHub 周一将在[阿姆斯特丹举办 GitHub Social Club](https://discord.com/channels/@me/1479607069501030579/1479607072852148236)。
- **Discord 统计数据加载失败**：多条消息指出 Discord 统计数据加载失败。
   - 该问题在不同频道均有报告，表明可能是一个普遍存在的问题。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1479211677026226429)** (15 messages🔥): 

> `Chat Web Client, Customize Themes` 


- ****Zoltun** 发布聊天 Web 客户端**：一位成员分享了名为 **Zoltun** 的聊天 Web 客户端，网址为 [zoltun.org](https://zoltun.org/) 和 [github.com/zoltun-org](https://github.com/zoltun-org)，作为 **GLM Chat Web Client** 的替代方案。
   - 这个可定制的客户端具有自动保存和针对阅读优化的 markdown 功能。
- **新的 UI 方向非常引人注目**：一位成员称赞 **Zoltun** 具有*大胆且引人注目的 UI 方向*，使其脱颖而出。
   - **Zoltun** 的作者正试图在现代与复古之间寻找平衡点，并允许用户自定义主题。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1479210541179338774)** (202 条消息🔥🔥): 

> `账户被盗与未经授权的交易，Mini Tavern 替代方案，GPT-4 在中国的可用性，刷机后的路由器配置，Google Gemini 模型的 403 错误` 


- **OpenRouter 遭遇账户劫持，资金受损！**：用户报告了**账户被盗**和**未经授权的交易**，一名用户指出他们已向银行投诉，并正在等待 OpenRouter 支持团队 `support@openrouter.ai` 的回复。
   - 另一名用户对不法分子（bad actor）可能通过多个账户转账或更改电子邮件表示担忧，这使得追踪变得更加困难，并凸显了 **API key** 泄露的风险。
- **MiniTavern 应用获得认可？**：一名用户询问是否有比 **MiniTavern** ([https://apps.apple.com/us/app/minitavern-tavern-roleplay/id6748523919](https://apps.apple.com/us/app/minitavern-tavern-roleplay/id6748523919)) 更好的替代方案，另一名用户简单回答了 *yes*。
- **Gemini 的地理封锁：俄罗斯遭到冷遇**：一名用户在通过 OpenRouter 访问 **Google Gemini models** 时遇到了 *403 Blocked by Google* 错误，尽管其账户余额充足。
   - 有人指出 Google 封锁了来自俄罗斯的 **API** 访问（[https://ai.google.dev/gemini-api/docs/available-regions](https://ai.google.dev/gemini-api/docs/available-regions)），虽然确认该用户身处该地区，但用户提到他们正通过德国的 **VPN** 进行操作。
- **路由器刷机僵局**：一名用户在刷入路由器固件后请求协助进行初始配置，报告称无法再通过线缆或 wifi 连接。
- **LLMs 陷入脚本编写怪圈**：一名用户报告了 **LLMs** 编写 **Python** 脚本来打印其响应而不是直接输出内容的问题，即使明确指示不要这样做也是如此。
   - 这种异常行为被归因于模型是在 **synthetic data**（合成数据）上训练的，增加 **examples**（示例）可能会缓解该问题，并指向了一篇关于 **agentic systems**（智能体系统）的 [manus 文章](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1479268002816983040)** (10 条消息🔥): 

> `模型推理质量，Prompt 发布，Agents.md，Elon Musk，Anthropic` 


- **推理质量必须先行**：一位成员指出，只有在模型/推理质量没有变差 5 倍的情况下，模型的使用才是可以接受的。
   - 他们质疑这一考量是否适用于 **prompt** 发布、公开的 **prompt** 嘲讽，以及每周的 **prompt** 读书会。
- **战略性 Context Windows 与 AGENTS.md 讨论**：一位成员建议在处理 **context window** 和类似 **AGENTS.md** 的文件时，应遵循“少即是多”的原则，并始终保持战略性。
   - 他们链接到了 [Evaluating agents.md Are Repository-linker.sh](https://arxiviq.substack.com/p/evaluating-agentsmd-are-repositorylinker.sh) 以获取更多信息。
- **Musk 的行为引发批评**：成员们对 **Elon Musk** 的[这条推文](https://x.com/elonmusk/status/2029833177368514831)反应负面。
   - 一位成员推测 **Musk** 感到“酸”，是因为 **Anthropic** 拒绝了他无限制使用其模型的提议，据称是因为“他的模型很烂”。
- **微软在安全担忧后继续提供 Anthropic 服务**：一位成员链接了一篇 [CNBC 文章](https://www.cnbc.com/2026/03/05/microsoft-says-anthropics-products-can-remain-available-to-customers-after-security-risk-designation.html)，报道称尽管存在“安全风险认定”，**Microsoft** 仍允许 **Anthropic** 的产品继续对客户开放。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1479207218321690827)** (208 messages🔥🔥): 

> `GPT Pro council, Cursera prompt injection, Extensible RL framework, Hermes Agent skins, GPU pricing` 


- **GPT Pro 被推测为一个 AI 议会**：有推测认为 **GPT Pro** 实际上是一个由 **8 个 AI** 组成的议会，其中 **7 个** 负责生成答案，**1 个** 负责决策，以实现更高且更可靠的结果。
   - 有人注意到 **GPT Pro** 的定价是标准 GPT 的 **10 倍**，这完美符合议会模型，尽管这目前仅是推测。
- **Coursera 面临提示词注入攻击尝试**：有人在 LinkedIn 上发现了 Coursera 系统中的一个 prompt injection 漏洞。该 AI 原本应维护学术诚信且不提供评估答案，但拦截并未生效。
   - 评估页面上的 AI 助手被禁用，并显示消息：*为了维护 Coursera 的学术诚信政策，此 AI 助手在评估页面上已禁用。*
- **可扩展 RL 框架**：一名成员正在寻求一种可扩展的 **RL 框架** 以集成到其软件中，并考虑使用由 **LLMs** 定义的奖励函数。
   - 目标是创建一个端到端的全模态标注/训练系统，可能基于 **GRPO**。
- **Hermes Agent 获得自定义皮肤**：一名成员正在开发自定义 **Hermes Agent 皮肤**，展示了具有主题图形用户界面的早期迭代版本。
   - 该成员正在匹配 TUI 主题并进行 GUI 调整。
- **高昂的 GPU 价格令人担忧**：一名成员对用于微调的 **GPUs** 租赁高成本表示担忧，质疑在当前市场下此类项目的可行性。
   - 由于 **GPU pricing** 过高，他们正在寻找提供优惠价格的供应商。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1479283863019851988)** (20 messages🔥): 

> `CUDA memory programming, AMD kernel hackathon, ML Eng / ML Ops intern position, Nvidia Compute Conference tickets` 


- **推荐 CUDA 内存编程教程**：一名成员推荐了一个 [CUDA 内存编程教程](https://siboehm.com/articles/22/CUDA-MMM)，称其为*初学者的最佳切入点*。
   - 他们指出该教程对 GPU 内存编程有*很好的覆盖*。
- **宣布 AMD Kernel 黑客松**：成员们讨论了最近宣布的 **AMD kernel hackathon**，其中一名成员尽管刚接触 CUDA 且目前正在优化 **softmax**，但仍在考虑参加。
   - 另一名成员鼓励参与以获取学习经验，并指出这可能是专门针对 **AMD 芯片**的。
- **寻找 ML Eng / ML Ops 实习岗位**：在 **FableTherapeutics** 公司撤回录取通知后，一名成员正在为一名 **University of Waterloo**（滑铁卢大学）的学生寻求暑期 **ML Eng / ML Ops 实习**帮助。
   - 另一名成员允许发布该实习生的 [LinkedIn 个人资料](https://www.linkedin.com/in/mramamon/)，并分享了 **Microsoft** 关于 **RLM research** 的实习招聘信息。
- **分享免费 Nvidia Compute Conference 门票**：一名成员在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 分享了 **3 张赠送的 Nvidia Compute Conference 门票**。
   - 另一名成员正在寻找参加 **AMD kernel 竞赛**的队友。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1479401956656156783)** (21 messages🔥): 

> `NVL72 H2H Copies, Qwen3.5 MoE Megakernel, AMD competition, HIP performance, Blackwell FP16 throughput` 


- ****NVL72** H2H Copies 受到质疑**: 一位成员询问 **NVL72** 是否支持通过 **NVLink** 进行 **H2H copies**（主机到主机拷贝），具体来说是包含主机固定内存 (host pinned memory) 的句柄是否可以由另一个主机通过 **Copy-Engine** 调用来移动数据。
   - 未提供任何解答。
- **提议 Qwen3.5 MoE Megakernel 项目**: 一位成员提议为 **Qwen3.5 MoE** 开发一个 **megakernel**，并指出由于 **MoE** 的复杂性、由于 **32GB** 限制而需要的 **nvfp4** 以及混合架构，该项目具有挑战性。
   - 另一位成员表示感兴趣，但因其他事务缠身而无法参与，并提到用于 decode 的 **GDN 部分**并不太复杂，但在小型 GPU 上处理 **MoE** 非常麻烦。
- ****HIP** 提交作品展示**: 一位成员分享到，提交给 [gpumode.com](https://www.gpumode.com/home) 的大多数高性能作品都是使用 **HIP** 编写的。
   - 他们还链接了 [William 最近关于 hipkittens 的演讲](https://www.youtube.com/watch?v=OkFk-7Mk6qI)，以便其他人快速上手。
- **思考 **Blackwell** 的 FP16 吞吐量**: 一位成员质疑为什么在 [Blackwell RTX 架构白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) 中，**FP32** 非 Tensor **TFLOPs** 与 **FP16** 非 Tensor 的数值相同。
   - 另一位成员链接了 [CUDA C 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)，说明某些 GPU 具有更高的 **fp16** 吞吐量，而其他一些则不然。
- ****nvDecoder cuvidCreateDecoder** 神秘崩溃**: 一位成员报告称，在对 **h264** 运行解码时，**nvDecoder cuvidCreateDecoder** 发生崩溃。
   - 返回的错误代码为 **999**，这是一个神秘的崩溃代码。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1479531953404252383)** (1 messages): 

> `Kernel Competition, AMD Sponsorship, DeepSeek-R1-0528 Optimization, GPT-OSS-120B Optimization, MI355X Optimization` 


- **AMD 赞助内核竞赛，提供 110 万美元奖金**: 一项新的内核竞赛现已开放提交，由 **AMD** 赞助，奖金池高达 **110 万美元**，重点是在 **MI355X** 上优化 **DeepSeek-R1-0528** 和 **GPT-OSS-120B**；注册地址为 [luma.com](https://luma.com/cqq4mojz)。
- **竞赛分为两个阶段**: 第一阶段（3 月 6 日至 30 日）涉及优化三个内核：**MXFP4 MoE**、**MLA Decode** 和 **MXFP4 GEMM**，通过 gpumode.com 提交。
   - 在第二阶段（3 月 31 日至 5 月 11 日），第一阶段的顶级团队将与 **AMD** 和 **GPU MODE** 的工程师合作，将内核合入到流行的推理引擎上游中。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

jaefosho: 正在阅读这个，你还有其他的吗？
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1479291063620603904)** (11 messages🔥): 

> `Programming Massively Parallel Processors, CUDA, C++ for CUDA, PyTorch Helios hackathon, Popcorn CLI` 


- **《Programming Massively Parallel Processors》依然是神作**: 尽管出现了 Inference Engineering、AI Systems Performance Engineering 以及 Chip Huyen 的书籍，但《**Programming Massively Parallel Processors**》一书仍被推荐为顶级资源。
   - 一位成员重申，它仍然是入门首选书籍。
- **CUDA 资源确认**: **Nvidia CUDA 编程指南**和《**Programming Massively Parallel Processors**》是学习 **CUDA** 编程的顶级资源。
   - 未提供更多细节。
- **C++ 基础对 CUDA 来说已足够**: 拥有本科水平的 **C++ 基础**是开始 **CUDA** 的坚实基础，重点是熟练掌握指针和手动内存管理（**malloc** 和 **free**）。
   - 在 GPU 上运行的代码通常不使用 **STL** 或 **std::vector** 等复杂的 **C++ 特性**，重点在于手动在主机 RAM 和 GPU 设备之间移动数据；一块 **RTX 4050** 就足以开始学习。
- **Helios 黑客松欢迎初学者**: PyTorch Helios 黑客松欢迎初学者参加和观摩，即使没有 kernel hacking 经验也可以。
   - 未提供更多细节。
- **Popcorn CLI 提交不需要特定硬件**: **Popcorn CLI** 允许为远程机器提交内核，参与竞赛不需要拥有 **MI355X** 等特定硬件。
   - 第二阶段将为团队提供直接的 **SSH 访问**权限。


  

---

### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

jaefosho: 虽然机会渺茫，但乔治亚州（Georgia）有什么活动吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1479673034607558687)** (3 messages): 

> `AMD Kernel 开发竞赛, MI355X 访问权限, Popcorn CLI` 


- ****MI355X** 访问任务开始**: 一名成员询问了关于 **AMD kernel 开发竞赛**的正确频道以及在哪里可以租用 **MI355X** 的访问权限。
   - 另一名成员确认这就是正确的频道，同时还有人建议使用 `popcorn (或 popcorn-cli) submit solution.py`，之后会弹出一些菜单。
- ****Popcorn CLI** 受到推荐**: 为了参加竞赛，一名成员推荐使用 [**Popcorn CLI**](https://github.com/gpu-mode/popcorn-cli)。
   - 该 CLI 允许参赛者提交 `solution.py`，并会打开一个菜单来引导提交过程。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1479233237439086593)** (1 messages): 

> `Blackwell 消费级芯片, Kernel 级微调, 消费级芯片的可能性` 


- **Blackwell 消费级芯片的期待升温**: 爱好者们期待利用消费级芯片学习 **Blackwell** 的巨大可能性。
   - 然而，严谨的 **kernel 层级**和**性能微调（tuning tweaks）**需要真实硬件，这印证了来自 Kernel 竞赛的发现。
- **真实 Blackwell 上的 Kernel 微调**: 针对 Blackwell 的严谨 **kernel 级**优化必须使用实际硬件。
   - 来自 Kernel 竞赛的经验强调，消费级芯片不足以进行高级微调和底层系统调整。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1479550130540515499)** (30 messages🔥): 

> `AMD Kernel 竞赛奖池, AMD 开发者账户的 GPU 积分, Popcorn CLI 提交系统, 提交错误与在其他 Stream 上工作的问题, AMD Kernel 竞赛提交信息` 


- **AMD 竞赛拥有“疯狂”的奖金**: AMD kernel 竞赛的奖池巨大，但第一名和第二名之间的差距非常显著，正如 [这个 Reddit 帖子](https://www.reddit.com/r/fastandfurious/comments/1e7z0eh/it_doesnt_matter_if_you_win_by_an_inch_or_a_mile/) 中提到的。
- **Kernel 竞赛不需要 GPU 积分**: 参赛者询问了关于 AMD 开发者账户 GPU 积分的问题，但被告知参加竞赛**不需要 GPU 积分**，他们可以使用 [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) 的基于队列的系统进行提交。
   - 之前的特等奖获得者显然*甚至从未为竞赛租用过 GPU*。
- **针对包含在另一个 Stream 上工作代码的简易检查**: 用户报告在提交涉及在另一个 stream 上工作的代码时遇到了 `500` 错误，有人建议从代码中删除 *stream* 这个词，以绕过简单的检查。
   - 一位用户说 *这甚至比 NVIDIA 的那个还要疯狂*，但另一位回复道 *但也难得多*。
- **AMD Kernel 提交信息及限制**: 提供了关于如何提交以及可以提交什么的有用信息的链接，包括限制以及代码运行的环境：[reference kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/amd_202602), [popcorn-cli](https://github.com/gpu-mode/popcorn-cli), 以及 [AMD kernel 官方参考](https://github.com/ROCm/aiter/tree/main/aiter/ops)。
- **确保竞赛诚信**: 组织者强调诚信是强制性的，他们将持续检查提交的内容是否符合规则，欢迎参赛者在群组中讨论有关合规性的问题或疑虑。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1479295711374741504)** (3 messages): 

> `Colfax Blackwell GEMM tutorial, Blockscaled GEMM, sm_103 K-mode` 


- **Colfax 发布新款 Blackwell GEMM 教程**: Colfax 发布了其 **Blackwell GEMM** 教程系列的最新篇章，重点介绍了 [blockscaled GEMM](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)。
   - 该教程旨在提供关于 **NVIDIA Blackwell GPUs** 硬件支持的分块缩放（block scaling）的见解。
- **表格中缺少第五种组合**: 一位用户注意到，教程中的表格似乎缺少了第五种组合（**E2M1**，向量长度 16，**UE8M0**）。
   - 这可能是文档中的一个疏忽，需要修正。
- **sm_103 K-Mode 扩展**: 在 **sm_103** 中，K-mode 不再受限于 **32B**，因为它现在支持 **K=96** 的稠密 **fp4**。
   - 这一扩展为内存访问模式和数据格式提供了更大的灵活性。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1479584579911155943)** (14 messages🔥): 

> `Discord Widget, Shields.io, Server Linking, Discord Badge` 


- **为 Shields.io 徽章激活 Discord Widget**: 一位成员请求在服务器上开启 Discord widget 设置，以便使用 [Shields.io 徽章](https://shields.io/badges/discord) 引导读者进行提问/评论，另一位成员确认已将其开启。
   - Shields.io 徽章将显示用户数量，并可以通过 Markdown 将徽章链接到 <#1373414141427191809> 频道。
- **目标频道选择难题**: 一位成员询问 Shields.io 徽章应该链接到随机频道还是 teenygrad 频道。
   - 建议考虑到服务器范围的设置，它应该链接到 <#1189498205101109300> 以实现通用复用，或者根据判断链接到 <#1189557310998200410>。
- **对 Discord 徽章的偏好浮现**: 一位成员表达了对特定 [Discord 徽章](https://github.com/gpu-mode/resource-stream#gpu-mode-resource-stream) 的熟悉，认为它在视觉上更出色，因为它包含了 Discord 图标。
   - 另一位成员也同意这是一个更好看的徽章。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1479556871764443177)** (6 messages): 

> `Heroku server issues, POPCORN_API_URL update, Invalid X-Popcorn-Cli-Id` 


- **Heroku 服务器出问题？POPCORN_API_URL 来救援！**: 遇到 *Heroku server not found* 错误的用户应确保其 **POPCORN_API_URL** 指向更新后的地址：[https://site--bot--dxfjds728w5v.code.run](https://site--bot--dxfjds728w5v.code.run)，详见 readme。
   - 重新安装可能会解决问题，因为更新应该已经包含在内。
- **绕过 Popcorn 安装？小心 API_URL！**: 一位用户因 *Invalid or unauthorized X-Popcorn-Cli-Id* 错误而通过手动构建二进制文件绕过了安装，结果发现其 **.bashrc** 中的本地 **POPCORN_API_URL** 被硬编码为旧的 Heroku URL。
   - 建议其他人在手动安装 **Popcorn** 后如果也遇到此问题，请提交 PR。
- **全新安装是 Popcorn 的关键？**: 一位用户通过执行全新安装（擦除 **.popcorn.yaml**）、设置新的 **POPCORN_API_URL** 并重新注册新的 **popcorn.yaml** 密钥解决了问题。
   - 该用户认为问题源于机器上的旧配置，建议可能需要彻底清理。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1479214522215436450)** (4 messages): 

> `NVlink XID errors, NCCL test for retransmits, Deterministic Algorithms in NCCL` 


- **NVLink 的 XID 错误大爆发！**: 用户在 `dmesg` 中看到每分钟数千个与 **NVLink** 相关的 **XID 错误**，这预示着潜在的硬件退化正在酝酿。
   - 这些错误表明 NVLink 上正在发生**位错误（bit errors）**，ECC 的快速增加表明可能存在信号完整性问题。
- **NCCL 网络检查挑战！**: 鼓励成员运行快速 **NCCL test** 以检查重传（retransmits）和链路降级（link fallbacks），从而评估 NVLink 连接的健康状况。
   - 将测试结果与迭代时间相关联可以帮助识别性能瓶颈。
- **NCCL 对确定性（Deterministic）的追求！**: 讨论引用了 [NVIDIA 关于在 NCCL 中控制浮点确定性的博客文章](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) 以及相关的 [GitHub 关于确定性算法的 issue](https://github.com/NVIDIA/cccl/issues/5550)。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1479502189176160349)** (3 messages): 

> `Modal GPU 访问，Modal 上的免费额度` 


- **Modal GPU 访问权限仍未确认**：成员们正在等待 **Modal** 上 GPU 访问权限的更新，并对团队成员缺乏本地 GPU 资源表示担忧。
   - 一名用户将咨询引向了特定频道 <#1464407141128339571>，可能是为了获取更多细节。
- **对 Modal 免费额度的疑虑浮现**：团队计划利用 **Modal** 上的免费额度，但对于该金额是否足以支撑整个开发周期存在不确定性。
   - 这种担忧源于远程办公的团队成员，他们完全依赖 **Modal** 进行开发，特别是在本地资源不可用的情况下。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1479209891091714311)** (1 messages): 

> `小世界模型，交互式世界模拟` 


- **酷炫的小世界模型出现**：一名成员分享了 [小世界模型](https://www.yixuanwang.me/interactive_world_sim/#interactive-demo) 演示的链接，展示了其有趣的特性。
   - 交互式演示允许用户探索这些模型的动态。
- **小世界网络可视化**：[交互式模拟](https://www.yixuanwang.me/interactive_world_sim/#interactive-demo) 允许用户调整参数并观察其对网络结构和行为的影响。
   - 这有助于理解互连性是如何产生并影响网络内部动态的。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1479298456689446952)** (10 messages🔥): 

> `固件工程师转型 GPU 技术栈，计算机科学专业大二学生夏季实习，具体成果 vs 凭证的重要性，贡献开源项目` 


- **固件专家寻求 GPU 职业路径**：一名拥有 **4 年经验** 的固件工程师寻求关于转型到 **GPU 技术栈角色**（特别是计算内核方向）的建议，计划从通过 [NVIDIA blogs](https://developer.nvidia.com/blog) 学习 CUDA 和 GPU 内存模型开始。
   - 该工程师希望确定这种转型的可行性，并向那些有过类似职业变动的人寻求指导。
- **大二学生努力争取夏季软件实习机会**：一名计算机科学专业的大二学生正在寻求获得夏季实习的建议，尽管其已经参与了多个项目，包括 **CUDA/Triton FlashAttention 实现**，使用 **TensorRT-LLM 和 Triton Inference Server 构建 LLM 服务流水线**，以及维护一个技术博客。
   - 他征求关于改进简历、项目选择以及申请/内推策略的反馈。
- **成果胜过名声，揭示现实**：一名成员表示，*大学学位和实习经历等凭证已不再足够*，*具体的、可验证的成果*现在是脱颖而出的关键。
   - 他们建议建立一个能解决昂贵工程问题的 GitHub 个人资料，并贡献开源项目以展示生产级别的编程技能。
- **开源拯救学生**：成员们讨论了贡献 **开源项目** 以获得实践经验的重要性。
   - 他们鼓励学生克服犹豫并开始贡献，强调即使是对大型库的小贡献也能产生重大影响并提供宝贵的学习机会，并提到了 [vLLM](https://github.com/vllm-project/vllm) 作为一个例子。


  

---

### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1479350893471338636)** (5 条消息): 

> `SIMT/Tile 互操作，cuTile 性能，FlashAttention backward 内核，Bastile` 


- **SIMT/Tile 互操作解锁 CUDA**：成员们正在研究 **SIMT/Tile 互操作**，这将允许用户从 **Tile 函数**中调用 **SIMT 设备函数**，从而可能提升 **CUDA** 的能力。
   - 如果进展顺利，这对 **CUDA** 来说将是一个巨大的进步，正如一位成员设想的那样，即使内核中其他部分都是 **SIMT 代码**，也可以使用 **cuTile** 在其内核内部进行排序和分区。
- **cuTile 驱动的内核性能卓越**：一位成员构建了一个基于 **cuTILE** 的小型 monkey-patching 库，其自定义内核在 **Qwen3** 上的单内核及端到端性能均优于 **Liger**。
   - 优化方案取自 **TileGym** 并经过进一步优化，相关改进已合并回上游。
- **Bastile 库为 CUDA 崭露头角**：一位成员发布了 **Bastile**，这是一个基于 **cuTILE** 的库，其自定义内核在 **Qwen3** 上优于 **Liger**，目前正在开发 **FlashAttention** 的 backward 内核。
   - 点击此处访问 [GitHub 仓库](https://github.com/aghilann/bastile)，点击此处访问 [B200 上的 Modal notebook 结果](https://modal.com/notebooks/aghilann/main/nb-9JUUBXJ23NK2b9Mf01WdEl)。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1479363758190100612)** (24 条消息🔥): 

> `GDN prefill 问题，Track C 差异，官方评估环境详情，CuTile 代码提交，Modal GPU 访问` 


- **调试 INCORRECT_NUMERICAL 问题**：部分成员在 `GDN prefill` 中遇到了 `INCORRECT_NUMERICAL` 问题，正在寻求能够通过数值精度测试的基准方案。
   - 据观察，[HuggingFace](https://huggingface.co) 和 [Starter Kit](https://starterkit.com) 使用的是 `qk4_v8`，但 `mlsys26.flashinfer.ai` 和 `bench.flashinfer.ai` 在 Track C 中使用的是 `qk16_v32`，这导致一些成员将代码适配为 `qk16_v32`。
- **请求官方评估环境的详情**：一位成员请求提供官方运行时/评估环境中使用的 **CUDA**、**Triton** 和 **PyTorch** 的确切版本。
   - 目标是使本地设置与官方环境紧密匹配，以便进行准确测试。
- **用于 CUDA 编译的 Modal 免费额度**：成员们建议在本地编译 CUDA 代码，并将 Modal 的免费额度主要用于基准测试/性能测试，而在 Google Colab 上进行正确性测试。
   - 有人强调，编译 CUDA 代码或获取 cubin 文件并不需要 **NVIDIA GPU**，建议为 CUDA 13 及以上版本使用 Nvidia 开发版 docker 镜像。
- **讨论 Blackwell B200 的访问权限**：成员们提到，虽然访问 **B200** 对 **Blackwell** 导向的指令（如 **UMMA**）会有所帮助，但首先使用通用 CUDA 和较低级别的 GPU 也可以取得重大进展。
   - 此外还指出，由于 Modal 不支持 `ncu`，详细的分析通常是在独立的机器上完成的。
- **CuTeDsl 实验者无法使用 CuTile**：一位成员报告使用了 **CuTeDsl**，但无法提交到 Modal，不得不编写自定义 Modal 脚本。
   - 这引发了关于是否允许使用自定义脚本的讨论，并请求组织者增加对 **CuTile** 的支持。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1479459246151172096)** (24 条消息🔥): 

> `OOM 错误，GGUF 量化，Compute 大会` 


- **OOM 错误困扰微调模型的评估**：一位成员报告在四块 **96GB GPU** 上使用 *lm_eval* harness 评估通过 **QLoRA** 微调的 **36b LM** (GLM-4-5-Air-qlora) 时遇到 **OOM** 错误，并建议尝试使用 `--num_processes 1`。
   - 另一位成员提到 Gemini 建议在 model_args 中添加 `device_map=auto`。
- **考虑通过 GGUF 量化节省内存**：在遇到 **OOM** 错误后，一位成员询问是否可以将模型转换为 **GGUF** 格式并量化为 **Q8** 或 **Q4** 以减少内存占用。
   - 他们表示打算在第二天尝试建议的解决方案。
- **Compute 大会门票待领取**：一位成员提供了几张本周日/下周一举行的 **Compute 大会**门票。
   - 另一位成员询问了会议地点以及是否提供在线参会。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1479277668615721011)** (22 条消息🔥): 

> `NeRFs, Flow matching, Diffusion models, Video generation, Sharpness Aware Minimization` 


- **Flow Matching 和 Diffusion 是否可以结合 NeRFs 使用？**: 成员们讨论了是否有人尝试将 **flow matching** 或 **diffusion** 与 **Neural Radiance Fields (NeRFs)** 结合用于视频生成。
   - 一位成员指出，他们在几个月前也有过同样的想法，并发现最近有一篇论文正在进行此类研究，但也发现*移动/变化场景的通用建模并不能被类 NeRF 结构很好地捕获，因此这可能不是正确的方法*。
- **NeRF 权重与归纳偏置 (Inductive Biases)**: 讨论中提到，如果 flow/diffusion transformers 擅长映射潜空间 (latent spaces)，为什么不将其映射到 **NeRFs 的权重空间**。
   - 然而，权重的结构并不像图像那样具有显而易见的**归纳偏置**，尽管如果对 NeRFs 应用 **L2 范数惩罚**，可以像 VAEs 一样使用 **N(0, I) 先验**来训练它们。
- **Video NeRFs 与光流预测**: 一位成员想知道是否可以实现 **video NeRFs**，即使用一个额外的参数 **t** 来描述视频的时间进程，或者将其转化为类似 **ODE 的 Flow 建模**，并尝试预测**光流 (optical flow)** 然后通过积分来找寻视频轨迹。
   - 他们还建议，根据计算科学的方法，有潜力使权重对权重空间中的扰动更加鲁棒。
- **SAM 能否帮助 NeRFs?**: 有人提到，像 **Sharpness Aware Minimization (SAM)** 这样的技术有助于提高权重的鲁棒性，但目前尚不清楚它们如何影响 **NeRF** 的表现；此外，计算化学主要围绕**能量剖面 (energy profiles)** 的探索而非优化，因此他们有更多旨在克服极小值并持续探索的设计。
   - 他们认为能量剖面探索大多是 **Langevin 动力学**（即 **SGD + 噪声**），而在网络所处的高维空间中，这通常非常困难。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1479405706812592170)** (2 条消息): 

> `muP cosine decay, wsd` 


- **余弦衰减 (Cosine Decay) 的流行得到证实**: 一位成员注意到，他们看到的关于 **muP** 的*大多数论文*都使用了 **cosine decay**。
   - 他们表示这几乎是*必需的*。
- **WSD 逐渐进入工作流**: 另一位成员反驳称，现在*大多数人实际上在使用* **wsd**。
   - 未提供进一步细节。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1479611885287379106)** (1 条消息): 

> `Innoculation Prompting, Finetuning` 


- **接种提示 (Innoculation Prompting) 论文引起关注**: 一位成员分享称，他们正在阅读来自 Anthropic 的 [inoculation prompting 论文](https://alignment.anthropic.com/2025/inoculation-prompting/)，并觉得非常有趣。
   - 他们认为该论文与微调 (finetuning) 技术有关。
- **强调与微调的相关性**: 该成员强调了 inoculation prompting 概念的相关性，特别是在**微调 (finetuning)** 过程中。
   - 发布该消息的成员对提及 (tagging) 表示了歉意。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1479247512211816521)** (23 条消息🔥): 

> `Quantization, vLLM library, Megatron vs TRL, PowerSync AI Hackathon, Deploy Lora spaces` 


- **Quantization 减少内存分配**：一位成员解释说，**Quantization** 通过使用更小的内存（例如使用 **float 8** 而不是 **float 32**）来减少内存分配，这仅分配 **8 bits** 的 vram 而不是 **32 bits**。
   - 通过 Quantization，如果你的模型有 **8 billion parameters**，那么每个参数可以节省 **24 bits**。
- **vLLM 是一个高效服务模型的工具箱**：**vLLM** 捆绑了多种减少 GPU 消耗的方法和服务技术，例如 **KV caching**，这使得每个新计算出的 token 的 attention 复杂度达到 **O(1)**。
   - 它还包括模型编译（model compilation）、追踪模型图以创建 tensor 的路径，以及将标准 Pytorch attention 切换为 **SDPA** 或 **flex-attention**。
- **Megatron 在速度上更具优势**：对于预训练（pretraining）、全参数 SFT 或需要在多 GPU 间进行模型并行（model parallelism）的任务，**Megatron** 通常比 **TRL** 更快。
   - 对于大规模基础训练或重度 SFT，成员建议使用 **Megatron**，然后使用 **TRL** 进行偏好微调（preference tuning）和 RLHF 风格的后训练（post-training）；NVIDIA 提供了 **Megatron Bridge** 用于 HF ↔ Megatron 的 checkpoint 转换。
- **PowerSync 举办 8k 奖金的虚拟 AI 黑客松**：**PowerSync** 正在举办一场虚拟黑客松，挑战参与者使用 **PowerSync** 作为同步引擎构建创新的 AI 驱动软件，并角逐超过 **$8,000** 的奖金。
   - 更多关于规则和奖金的信息请访问 [powersync.com/blog/powersync-ai-hackathon-8k-in-prizes](https://www.powersync.com/blog/powersync-ai-hackathon-8k-in-prizes)。
- **部署 Lora spaces 是可行的**：成员们讨论了部署 Lora spaces 的问题，有人分享了 [deploy_lora_spaces.md](https://cdn.discordapp.com/attachments/879548962464493622/1479672216625877153/deploy_lora_spaces.md?ex=69ace3a3&is=69ab9223&hm=f122c064dd259c08b202cccab089506f9b384a532d4dde65ace943c1788f555b&) 文件。
   - 他们指出，将模型暴露为 API endpoint 是可能的，但实际上只有非常小的 LLM 才能在免费的 CPU space 上运行。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1479207968808370328)** (14 条消息🔥): 

> `Greywall sandboxing, Arksim synthetic users, Canvo mobile app, Ktiseos-Nyx-Trainer, Shadowclaw v1.3` 


- ****Greywall** 对 CLI Agents 进行沙箱化：已开源！**：Greywall 是一个对具有完整 shell 访问权限的 CLI Agents 进行沙箱化处理的工具，目前已[开源](https://github.com/GreyhavenHQ/greywall)。
   - 它允许用户在不重启会话的情况下实时查看并阻止网络连接，并且现在已支持 MacOS。
- ****Arksim** 为 AI Agent 测试生成合成用户**：Arksim 是一个用于生成合成用户以测试 AI Agents 的工具，目前已[开源](https://github.com/arklexai/arksim)，并可通过 `pip install arksim` 安装。
- ****Canvo** 移动应用打造口袋代理（Pocket Agency）**：一位成员分享了一个用于完整口袋代理并能与 A2UI 更好交互的[移动应用](https://github.com/canvo-app/canvo)。
- ****Ktiseos-Nyx-Trainer**：面向开源 Loras 的 NextJS 训练器**：展示了一个名为 [Ktiseos-Nyx-Trainer](https://github.com/Ktiseos-Nyx/Ktiseos-Nyx-Trainer) 的 NextJS 训练器，用于开源 Loras 和 Checkpoints；它支持从 HF 下载和上传。
   - 目前尚不支持 RoCM 或 Zluda。
- ****Shadowclaw** v1.3：用 C 编写的极简个人 AI Agent**：[Shadowclaw v1.3](https://huggingface.co/webxos/shadowclaw-c) 是一个用 C 语言编写的、遵循 OpenClaw 哲学的单二进制极简个人 AI Agent。
   - 它的特点是支持自托管、具备 tool-using 能力、持久化内存且依赖项极少，通过 curl 与本地 LLM (Ollama) 通信并自动保存状态。


  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1479563801341988937)** (1 条消息): 

> `Gradio v4.19.0, Custom Components, Performance improvements, UI fixes` 


- **Gradio 升级至 v4.19.0！**: 根据 [公告](https://www.gradio.app/changelog)，**Gradio v4.19.0** 现已发布，包含一系列修复和 DX 改进。
   - 使用 `pip install -U gradio` 进行更新。
- **Custom Components 正确组合**: 解决了 Svelte 版本不匹配问题，并修复了 **Custom Components** 中 annotated types 的重新加载模式。
   - 这将有助于避免许多用户在使用自定义组件（尤其是包含 Svelte 代码的组件）时遇到的常见类别问题。
- **Gradio 速度提升**: 优化了内部 API 调用和数据结构以降低延迟，特别是针对 MCP，使 `queue=False` 事件实现了 **10 倍加速**！
   - 这些改进将带来更灵敏的应用响应，尤其是在频繁更新的场景下。
- **Gradio UI 界面更新**: 实施了多项 **UI 修复**，包括解决 `fill_height` 问题、点击示例后恢复 **Submit 按钮**，以及确保 `gr.Markdown` 进度条行为正确。
   - 这些修复通过解决常见的可用性问题和视觉缺陷，全面提升了用户体验。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1479325138796413131)** (11 条消息🔥): 

> `Introductions in Agents Course channel, Decoder's Lord Monster` 


- **新成员自我介绍**: 包括 Sai、Chanchlesh、Sidh、Chandan 和 Sanket 在内的几位新成员在频道中进行了自我介绍。
   - 兴趣范围涵盖 AI Agents 及其构建学习、Web 开发、编程以及探索新科技工具。
- **Decoder's Lord 承认对 'Monster' 负责**: Decoder's Lord 承认由于最近的一次推送（push）创建了一个 *monster*（巨型怪异代码/问题）。
   - 目前已经提交了一个 PR 来修复此问题。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1479276599022190612)** (42 条消息🔥): 

> `Kimi K3 release date, Run Kimi K2.5 on RTX 3090, Kimi account support, Kimi CLI usage, Kimi Claw down` 


- **下一代 Kimi 预测开启**: 用户们正在猜测 **Kimi K3** 何时发布，参考了 **Kimi K2** 和 **Kimi K2.5** [相隔 6 个月](https://x.com/allen_ai/status/2029591872612561189) 的发布模式。
   - 一名成员推测可能会在 *7 月* 发布，但也提醒研究进度有其自身的节奏。
- **RTX 3090 运行 Kimi K2.5 的挑战**: 有用户询问是否可以在单块 **RTX 3090** 上运行量化版或 coder (FT) 版本的 **Kimi K2.5**。
   - 一名成员开玩笑说：*如果你能给它粘上 1TB 的 VRAM，当然……可能行。大概每小时跑 1 个 token 左右吧。* 💥
- **Kimi 客户支持“人间蒸发”**: 一名用户因在被多次错误计费后无法获得客户支持而取消了 **Kimi 订阅**。
   - 他们报告称：*对于两次扣费错误，三周都没有收到任何回复，这简直无法接受*。
- **Kimi CLI 用户在睡觉时部署 11 个容器**: 一名用户报告使用 **Kimi CLI** 在一夜之间向 Azure 部署了 **11 个容器**，还报告从 2000 个视频的“稍后观看”列表中删除了 **600** 个视频。
   - 附带的一张图片暗示该用户是在睡觉时完成部署的 [查看图片](https://cdn.discordapp.com/attachments/1371757564005711973/1479492010615374030/Screenshot_2026-03-06-09-50-45-76_3aea4af51f236e4932235fdada7d1643.jpg?ex=69ace48e&is=69ab930e&hm=f39cbefb517531d1b016ce9176fe7247c662e2deaa9d10e043ee7fce7664933e&)。
- **Kimi Claw 停止工作**: 多名成员报告 **Kimi Claw** 已停止工作，并请求协助解决。
   - 成员们尝试了重启程序、重启服务器、自动修复等方法，但均未奏效。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1479226718253351054)** (38 messages🔥): 

> `波士顿生日派对, Credits 问题, 支持响应时间, 订阅问题` 


- **高昂的 Credits 成本导致用户流失**：几位成员对 Credits 的高昂成本表示不满，指出 Credits 仅在 **$13,000/月** 的层级提供，并且正在*考虑迁移到其他平台*。
   - 一位用户建议尝试 **antigravity google** 作为替代方案。
- **用户报告 Credits 升级问题**：多位用户报告了升级 Credits 或订阅时遇到的问题，一位用户称他们*刚刚花费 200 欧元升级了 Credits，但从未添加到账户中*，另一位用户报告称他们*将订阅升级到了 $1k 级别并已被扣费，但账户中没有 Credits*。
   - 这些用户正在寻求解决这些账单问题的帮助。
- **对客服响应缓慢的挫败感增加**：用户对支持团队的缓慢响应表示担忧，评论如*“支持响应需要一个世纪”*和*“支持真的很慢”*。
   - 一位用户甚至质疑：*“支持聊天窗口是不工作吗？我的账户被不公平地停用了。”*


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1479212192317575248)** (23 messages🔥): 

> `Nvidia 轨道数据中心系统架构师职位, Francois Chollet 的推文, DGX Spark, LLMs 未达到人类水平` 


- **Nvidia 正在招聘轨道数据中心系统架构师**：Nvidia 发布了一个 **Orbital Datacenter System Architect** 的职位空缺，负责设计用于太空计算的系统，暗示了潜在的地外探索计划；参见 [职位发布页面](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Orbital-Datacenter-System-Architect_JR2014044)。
- **Chollet 的推文引发辩论**：François Chollet 的一条推文引发了讨论，一些人将其解读为居高临下，而另一些人则认为这是关于低估感觉运动学习（sensorimotor learning）深度的个人见解，[查看原始推文](https://fxtwitter.com/vicnaum/status/2029579972688379928)。
- **尽管存在担忧仍考虑 DGX Spark**：成员们讨论了 **DGX Spark** 中的 **NVFP4** 是否可行，以及散热和 OS 稳定性问题是否已得到解决，并引用了 [John Carmack 的一条推文](https://x.com/ID_AA_Carmack/status/1982831774850748825)，其中提到了此类问题。
- **LLMs 未达到人类智力水平，是一种解脱？**：一位成员对 **LLMs** 预计在未来几年内不会达到人类水平的智力表示满意，并对权势人物控制机器人大军表示担忧。
   - 该成员还提到正在开发一款产品，帮助人们在图像处理中找到并使用合适的工具，并获得了积极的客户反馈。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1479538113519816785)** (2 messages): 

> `新工作公告` 


- **成员因新工作推迟章节发布**：一位成员宣布，由于本周开始新工作，S&B 的第 2 章将推迟到下周四发布。
- **祝贺获得新工作！**：另一位成员祝贺该用户获得新工作。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1479266561654001956)** (6 messages): 

> `Anthropic 经济指数, 数据中心泡沫, Wario 部门` 


- **Anthropic 经济指数发布**：根据 [官方公告](https://www.anthropic.com/news/the-anthropic-economic-index)，Anthropic 推出了 **Anthropic Economic Index**。
- **数据中心泡沫达到顶峰**：成员们根据 [这条帖子](https://x.com/i/status/2029907842208031203) 指出，我们正处于 **datacenter bubble** 的巅峰。
- **DoW 现在是 Wario 部门**：一位成员开玩笑说，每当有人提到 **DoW**（通常指 Department of Welfare 或相关机构）时，他们听到的都是 **Department of Wario**，并发布了一个相关的迷因。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1479372620867833866)** (5 messages): 

> `Tinygrad JITBEAM 基准测试，Bounty 锁定提交费用，CAT 算子` 


- **Tinygrad JITBEAM 胜过 C**：根据 [这条 Discord 消息](https://discord.com/channels/1068976834382925865/1108235368702164992/1479323496990507101)，在经过各种升级和修复后，Tinygrad **JITBEAM** 的基准测试性能已优于 **C**。
- **Bounty 锁定费用**：有人建议对每次 Bounty 锁定提交收取少量且可退还的 **5 美元费用**。
- **辩论 CAT 算子的价值**：讨论围绕 **CAT 算子**展开，质疑它是否与其他移动操作（movement ops）匹配，以及是否确实必要。
   - 一位成员指出，*数学家喜欢让他们的推理尽可能通用*，而*物理学家则总是对特殊情况感兴趣*，tinygrad 倾向于后者。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1479224848537485477)** (4 messages): 

> `安全漏洞披露，GPT-5.4 Token 使用情况，用于 Delphi/Pascal 的 Aider` 


- **研究员未被重视的漏洞报告被利用！**：安全研究员 Adnan Khan 在 2025 年 12 月下旬发现了一个漏洞链，并于 2026 年 1 月 1 日通过 [GitHub Security Advisory](https://github.com/advisories) 进行了报告，但多次跟进均未收到回复。
   - 在 Khan 于 2 月 9 日公开披露后，Cline 在 **30 分钟**内完成了修复，尽管随后的密钥轮换（key rotation）错误引发了进一步的问题。
- **GPT-5.4 对 Token 的渴求**：一位用户注意到，虽然 **GPT 5.4** 表现出色，但它消耗了大量的 Token，被称为 *Token 吞噬者*。
   - 鉴于其强劲的性能指标，可能需要对该模型的效率进行进一步分析。
- **在 Delphi/Pascal 中使用 Aider**：一位成员询问是否有人将 Aider 与 **Delphi/Pascal** 结合使用。
   - 其他开发者是否在这种语境下利用 Aider 仍有待观察。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1479521698523906270)** (2 messages): 

> `Apple Neural Engine 上的 LoRA 梯度，用于内存的 Modal Sandbox & Volume，ANE matmul 编译器，Fleet-RLM 框架` 


- **LoRA 梯度在 Apple Neural Engine 上运行**：一位工程师利用 **Claude Code (Opus 4.6)** 在 Apple Neural Engine 上以 **~2.8W** 的功耗运行 LoRA 微调，实现了 **192 个 ANE 梯度调度（gradient dispatches）** 且无 GPU 回退（fallbacks），详见 [博客文章](https://x.com/StraughterG/status/2029957160864522513)。
   - 该工程师发现 `matmul` 可以编译但从未执行，空间维度必须是 16 的倍数，且 ANE 编译器在编译约 119 次后会静默失败。
- **Modal Sandbox & Volume 提升内存性能**：一位开发者正在改进其前端，弃用 Redis 和向量数据库，转而使用 **Modal Sandbox** 和 **Volume** 在 [fleet-rlm](https://github.com/Qredence/fleet-rlm) 框架中处理内存/分析任务。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1479571797052494021)** (2 messages): 

> `Compute 大会，AI 基础设施，AI Agents，下一代云` 


- **Daytona 在旧金山举办 Compute 大会**：Daytona 将于 **3 月 8 日至 9 日**在**旧金山 Chase Center** 举办 **Compute** 大会，该会议专注于 **AI 基础设施**、**Agents** 和**下一代云**，详见其 [网站](https://compute.daytona.io/)。
- **Compute 大会演讲嘉宾亮点**：大会将邀请包括 **Aaron Levie** (Box)、**Parag Agrawal** (Parallel)、**Harrison Chase** (LangChain)、**Lin Qiao** (Fireworks AI)、**Russ D'Sa** (LiveKit)、**Beyang Liu** (Amp)、**David Cramer** (Sentry)、**Nikita Shamgunov** (Neon)、**Dylan Patel** (SemiAnalysis)、**Waseem Alshikh** (Writer) 和 **Ivan Burazin** (Daytona) 在内的演讲嘉宾。
- **Compute 大会提供免费门票**：在 [Luma](https://luma.com/k6bc82dv) 上使用代码 `EQ6VA5` 可获得三张 **Compute 大会**的免费门票。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1479561301104525332)** (1 messages): 

> `MCP-I 问题, Auth agent identity, MCP 生态相关性` 


- **MCP-I 问题来袭**：一位成员在 [MCP-I](https://share.google/aimode/xAik81A0u4WKsjewv) 上遇到了一个问题，并希望将其集成到 **auth agent identity** 侧。
   - 目标是在实际的 MCP contrib 生态系统中捕捉用例，此帖仅作为 FYI（供参考）。
- **MCP 相关性受到质疑**：该成员指出，相关问题通常被归类为 "XXXXMCP" 或 "MCP - XXXXX" 类别，但在进一步调查时，发现它们与 **MCP** 并无直接联系。
   - 这引发了关于其与更广泛的 MCP 生态系统之间真实相关性和连接性的疑问。


  

---


---


---