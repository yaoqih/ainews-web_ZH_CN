---
companies:
- openai
- anthropic
- x-ai
- tencent
- deepseek
date: '2026-04-30T05:44:39.731046Z'
description: '**OpenAI 的 GPT-5.5** 在长程网络任务中实现了顶级性能，以 **71.4%** 的通过率媲美或超越了 **Claude
  Mythos Preview**，并且在推理量超过 **1 亿个 token** 后仍展现出持续的提升。OpenAI 还发布了 ChatGPT 的**“高级账户安全”**更新，增强了防钓鱼攻击的能力。**Codex**
  的更新使其从编程领域扩展到通用计算机任务，速度提升高达 **42%**，并引入了基于角色的新手引导（onboarding）和应用集成。


  在经济性方面，**GPT-5.5 Pro** 在 **CritPt** 上表现出微弱的 SOTA（业界领先）改进，且与 GPT-5.4 Pro 相比，**成本和
  token 使用量降低了约 60%**。


  在开源权重模型中，**Qwen3.6 27B** 在 150B 参数以下的类别中处于领先地位，其**智力指数评分为 46**，具备 **262K 上下文**窗口、原生多模态输入以及高效的
  BF16 权重。腾讯的 **Hy3-preview**（总参数 295B，MoE 激活参数 21B）智力指数评分为 42，在 **CritPt** 上展现出强大的科学推理能力。xAI
  的 **Grok 4.3** 在智能体（agentic）基准测试中有显著提升，且成本有所降低。'
id: MjAyNS0x
models:
- gpt-5.5
- claude-mythos-preview
- gpt-5.5-pro
- qwen3.6-27b
- hy3-preview
- grok-4.3
- gemma-4-31b
- glm-5.1
- deepseek-v4-flash
people:
- sama
- scaling01
- cryps1s
- polynoamial
- ajambrosino
- arix
title: 今天没发生什么特别的事。
topics:
- cybersecurity
- model-efficiency
- multimodality
- model-benchmarking
- agentic-ai
- model-cost-optimization
- context-windows
- model-performance
- open-weight-models
- software-integration
- security-updates
---

**平静的一天。**

> 2026年4月29日至4月30日的 AI 新闻。我们查阅了 12 个 Reddit 子版块、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，且未新增 Discord 内容。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[自行选择](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 回顾

**OpenAI 的 GPT-5.5、Codex 扩展以及网络能力评估**

- **GPT-5.5 在长跨度（long-horizon）网络任务中已确立了顶尖地位**：英国 AI Security Institute 报告称，[GPT-5.5 成为第二个能端到端完成其多步网络攻击模拟的模型](https://x.com/AISecurityInst/status/2049868227740565890)，多篇后续帖子强调了该模型在评估中与 **Claude Mythos Preview** 旗鼓相当：[@scaling01](https://x.com/scaling01/status/2049870801998864606) 指出 GPT-5.5 的平均通过率为 **71.4%**，而 Mythos 为 **68.6%**；而 [@cryps1s](https://x.com/cryps1s/status/2049879762169167898) 注意到 GPT-5.5 在 **2/10** 的尝试中解决了 TLO 链，而 Mythos 为 **3/10**。[@polynoamial](https://x.com/polynoamial/status/2049883449327243413) 强调，在推理预算超过 **1 亿 Token** 后，性能仍在提升，表明目前尚未出现明显的饱和迹象。这实质性地改变了早先关于 Anthropic 在进攻性网络自动化领域拥有独特领先地位的论调。OpenAI 还借此机会发布了一项产品侧的安全更新：[ChatGPT 高级账户安全](https://x.com/OpenAI/status/2049902506881462613)，增加了防钓鱼登录和强化的恢复机制。

- **Codex 正在从编程领域扩展到通用计算机工作**：OpenAI 发布了一次重大的 Codex 更新，明确定位为“适用于每个人、适用于任何通过计算机完成的任务”，[官方公告](https://x.com/OpenAI/status/2049928776147230886)重点介绍了基于角色的引导流程、应用连接以及涵盖文档、幻灯片、表格、研究和规划的工作流。[@ajambrosino](https://x.com/ajambrosino/status/2049928915872075984) 将此次更新总结为：动态的任务特定 UI、**快 20%** 的计算机/浏览器使用速度、更好的幻灯片/表格处理能力以及更流畅的交接；而 [@AriX](https://x.com/AriX/status/2049932746567598472) 特别指出更新后 **Computer Use 运行速度提升了 42%**。Sam Altman 对此次发布进行了宣传，称 [“今天 Codex 迎来重大升级！尝试将其用于非编程类的计算机工作。”](https://x.com/sama/status/2049946120441520624) 这一大趋势表明：OpenAI 正在将“计算机使用 Agent”的 UX（用户体验）产品化，而不单纯是提升模型能力。

- **基准测试的增量变化虽小，但在经济上意义重大**：[Artificial Analysis](https://x.com/ArtificialAnlys/status/2049926072595280030) 报告称 **GPT-5.5 Pro** 在 **CritPt** 上的表现略微超越 GPT-5.4 Pro 成为新的 SOTA，但有趣的点不在于原始分数——它在那项前沿科学评估中，以 **约 60% 的更低成本和 Token 使用量** 实现了这一飞跃。这与外界广泛的讨论相吻合，即 GPT-5.5 系列与其说是追求剧烈的智能断层式提升，不如说是追求在高价值工作流中更强的可靠性和更高的效率。

**开放权重模型动态：Qwen3.6、腾讯 Hy3-preview、Grok 4.3 以及 Ling 2.6 1T**

- **Qwen3.6 27B 似乎是当日最重要的开放权重发布**：[Artificial Analysis](https://x.com/ArtificialAnlys/status/2049881951260283097) 将 **Qwen3.6 27B** 评为 **150B** 参数以下新的开放权重领导者，其 **Intelligence Index（智能指数）得分为 46**，领先于 Gemma 4 31B 和之前的 Qwen 变体。关键细节包括：**Apache 2.0** 协议、**262K 上下文**、**原生多模态输入**，且 BF16 权重足够小，可以放入单个 H100 显卡。配套的 **35B A3B MoE** 得分为 **43**，使其成为 **3B 激活参数** 左右最强的开放模型。折衷之处在于按输出 Token 计算的推理成本较高：AA 估计 Qwen3.6 27B 在测试套件中消耗了 **约 1.44 亿输出 Token**，运行成本约为 Gemma 4 31B 的 **21 倍**。尽管如此，就“单位尺寸的能力”而言，这似乎是一个显著的进步。

- **腾讯的 Hy3-preview 具有竞争力，但并非同类顶尖**：[Artificial Analysis](https://x.com/ArtificialAnlys/status/2049852417316143393) 将 **Hy3-preview** 描述为一个拥有 **295B 总参数 / 21B 激活参数的 MoE** 模型，具有 **256K 上下文** 和 **限制性商业用途** 的社区许可证。它在 AA 的智能指数中得分为 **42**，落后于 Qwen3.6 27B、DeepSeek V4 Flash 和 GLM-5.1 等近期推出的开放竞争对手。最引人注目的亮点是 **CritPt** 测试，它以 **4.6%** 的得分追平了 GLM-5.1，这表明相对于其整体定位，它具有优于平均水平的科学推理能力。

- **xAI’s Grok 4.3 improved sharply on agentic benchmarks while getting cheaper**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049987001655714250) measured **Grok 4.3** at **53** on the Intelligence Index, up four points from Grok 4.20 v2, with a major jump on **GDPval-AA** to **1500 Elo**. AA also reported approximately **40% lower input price** and **60% lower output price** than the prior version. The release still trails GPT-5.5 on GDPval-AA by a wide margin, but it looks like a real systems-and-post-training improvement rather than a minor rev.

- **Ant Group’s Ling 2.6 1T targets cost-efficiency rather than frontier status**: [Artificial Analysis](https://x.com/ArtificialAnlys/status/2049923495602303438) positioned **Ling 2.6 1T** as a **1T-parameter non-reasoning model** scoring **34**, with decent GPQA/HLE numbers and notably low benchmark-run cost at roughly **$95**. The caveat is reliability: AA reported a **92% hallucination rate** on AA-Omniscience.

**DeepSeek multimodal/vision work, GUI agents, and training scale speculation**

- **DeepSeek’s multimodal direction appears tightly coupled to computer-use agents**: [@nrehiew_](https://x.com/nrehiew_/status/2049840778491662623) highlighted that DeepSeek trains vision into **V4-Flash** by having the model directly output **bounding boxes and point coordinates during reasoning**, interpreting this as a computer-use-oriented design rather than generic VLM work. A second post argues the paper’s “visual primitives” tasks map directly to browser/computer use rather than broad multimodal understanding ([link](https://x.com/nrehiew_/status/2049840802562740311)). That framing matches parallel observations from [@teortaxesTex](https://x.com/teortaxesTex/status/2049871869847765212) that DeepSeek may be integrating vision weights back into the main V4 line rather than releasing a separate “V4-Flash-Vision”.

- **The repo disappearance became a story of its own**: after release, several observers noted that DeepSeek’s “Thinking with Visual Primitives” repo vanished, including [@teortaxesTex](https://x.com/teortaxesTex/status/2049880056420298995) and [@arjunkocher](https://x.com/arjunkocher/status/2049875566678118898). No clear explanation emerged in these tweets, but the deletion drew more attention because the work suggested a concrete recipe for visual reasoning and GUI grounding.

- **Scaling chatter points to very large token counts for frontier pretraining**: [@teortaxesTex](https://x.com/teortaxesTex/status/2049830477167526255) argued that **>100T tokens** is no longer unusual for frontier models and estimated a hypothetical **100T-token DeepSeek V4** as “V4 + 2 more epochs,” while [@nrehiew_](https://x.com/nrehiew_/status/2049848830292856970) back-of-the-enveloped **~150T tokens** and **~9e25 pretraining FLOPs** for a **~100B active** model, suggesting a run feasible in roughly **14 days** on an OpenAI-scale **100K GB200** cluster at conservative MFU. These are speculative takes, but useful as calibration for what “frontier-scale” now means in practice.

**Agent infrastructure, harness engineering, and collaborative agent systems**

- **There is a clear shift from model-centric bragging to harness-centric engineering**: Cursor published a strong note on [how it tests and tunes its agent harness](https://x.com/cursor_ai/status/2049901436918436249), focusing on runtime, evals, degradation repair, and model-specific customization rather than generic benchmark claims. [@Vtrivedy10](https://x.com/Vtrivedy10/status/2049919247321813491) explicitly connected Cursor’s writeup to design patterns converging across agent builders: bespoke prompts/tools per model, mixed offline+online evals, dogfooding, and treating the context window as the primary compute boundary.

- **LangChain continues to package deployment and multi-tenant agent infra**: [@hwchase17](https://x.com/hwchase17/status/2049858892637892739) introduced **DeepAgents deploy**, a config-driven cloud deployment flow via `deepagents.toml`, covering agent, sandbox, auth, and frontend sections. Related posts from LangChain staff detailed agent-server patterns for data isolation, delegated credentials, and RBAC in multi-user deployments ([example](https://x.com/sydneyrunkle/status/2049956826670911809)). This is increasingly the boring-but-important layer turning demos into enterprise software.

- **Collaborative multi-agent workspaces are getting more concrete**: [@cmpatino_](https://x.com/cmpatino_/status/2049881579691139372) introduced **Agent Collabs**, using Hugging Face buckets plus Spaces as a shared backend for swarms of heterogeneous agents to exchange messages, artifacts, and progress. The noteworthy idea is not just “agents collaborating,” but lightweight coordination primitives that let weaker agents contribute useful validation work while better-resourced agents handle expensive experiments.

**Security, supply chain, and account hardening**



- **Open-source package compromise remains an acute operational risk**: [Socket](https://x.com/SocketSecurity/status/2049849100548424180) reported that the popular PyPI package **`lightning`** was compromised in versions **2.6.2** and **2.6.3**, with malicious code executing on import, downloading **Bun**, and running an **11 MB obfuscated JavaScript payload** aimed at credential theft. [@theo](https://x.com/theo/status/2049914688318959952) connected that incident with additional package compromises (`intercom-client` on npm) and a Linux zero day, arguing the tempo of software supply-chain attacks is increasing.

- **Security scanners are becoming first-class AI products**: Anthropic rolled out **Claude Security**, described by [@kimmonismus](https://x.com/kimmonismus/status/2049901987500552195) and later [@_catwu](https://x.com/_catwu/status/2049964403177689130#m) as a repo vulnerability scanner that validates findings and suggests fixes, powered by **Opus 4.7**. Cursor shipped a parallel offering with [Cursor Security Review](https://x.com/cursor_ai/status/2049926283061035254), including always-on PR review and scheduled codebase scans. This is one of the clearest examples of model vendors moving directly into established devsecops categories.

**Top tweets (by engagement)**

- **OpenAI Codex broadens into general knowledge work**: [OpenAI’s Codex announcement](https://x.com/OpenAI/status/2049928776147230886) and [Sam Altman’s follow-up](https://x.com/sama/status/2049946120441520624) were the day’s biggest product posts, signaling a strategic push from “coding agent” to “computer-use agent”.
- **GPT-5.5’s cyber eval result mattered**: [UK AISI’s thread](https://x.com/AISecurityInst/status/2049868227740565890) was one of the highest-engagement technical posts and reshaped comparisons with Anthropic’s Mythos.
- **Qwen shipped interpretability tooling, not just models**: [Qwen-Scope](https://x.com/Alibaba_Qwen/status/2049861145574690992), an open suite of sparse autoencoders for Qwen models, stood out as a rare release focused on feature steering, debugging, data synthesis, and evaluation rather than raw model weights.
- **Anthropic published a large-scale guidance/sycophancy study**: [their analysis of 1M Claude conversations](https://x.com/AnthropicAI/status/2049927618397614466) tied behavioral research directly to training changes for **Opus 4.7** and **Mythos Preview**, an important sign that post-training loops are becoming more productized and data-informed.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap



### 1. AMD Ryzen 395 Box 与 Halo Box 发布

  - **[AMD 内部研发的 Ryzen 395 Box 将于六月推出](https://www.reddit.com/r/LocalLLaMA/comments/1t038g7/amd_inhouse_ryzen_395_box_coming_in_june/)** (Activity: 1061): **来自 AMD AI Dev Day 演示的图像展示了即将推出的 AMD Ryzen 395 box，预计将于六月发布。该设备拥有 `128GB` 统一内存，并声称利用所谓的 "Ryzen AI Max" 原生支持 `200 billion models`。如演示中所提到的，该产品似乎由 Lenovo 制造。然而，一位工程师确认，该装置本质上是一个配备了 `128GB` 内存且没有其他变动的 Ryzen 395。** 评论者们对在 `128GB` 统一内存上运行 `200 billion model` 的实用性表示怀疑，考虑到即使计入操作系统开销后的内存限制，其可行性仍然存疑。

    - obiwanfatnobi 提出了一个关于在具有 '128GB unified RAM' 的系统上运行 '200B model' 可行性的技术点。他们强调，即使使用 Linux，可用的 VRAM 也仅约为 '116GB'，这对于如此巨大的模型可能不够，暗示了当前 AI 工作负载硬件配置的潜在局限性。
    - promethe42 将新的 AMD Ryzen 395 box 与 'Framework Desktop' 进行了比较，指出它似乎晚发布了 '12 个月'。他们建议 AMD 在发布新硬件之前应优先改进其 'drivers/ROCm'，这表明软件支持可能滞后于硬件进步。
    - DaniyarQQQ 评论了对 '512GB of unified memory' 的需求，暗示当前的内存容量可能不足以满足现代计算需求，特别是在高性能或 AI 应用中。这表明了尖端技术中内存需求增加的趋势。

  - **[AMD Halo Box (Ryzen 395 128GB) 照片](https://www.reddit.com/r/LocalLLaMA/comments/1t09hyw/amd_halo_box_ryzen_395_128gb_photos/)** (Activity: 467): **配备了 `Ryzen 395` 处理器和 `128GB` RAM 的 AMD Halo Box 展示了运行 Ubuntu 的画面。该装置包括一个可编程灯条，增强了其定制化能力。然而，它缺少 CD-ROM 驱动器，也没有用于 clustering 的快速端口，这可能会限制其在某些高性能计算场景中的使用。** 评论者注意到缺少 CD-ROM 和用于 clustering 的快速端口是潜在的缺点，这表明虽然设备很紧凑，但这些遗漏可能会影响其在特定技术应用中的效用。

    - OnkelBB 指出 AMD Halo Box 缺乏用于 clustering 的快速端口，这可能会限制其在高性能计算环境中的使用，因为在这些环境中，快速互连对于跨多个节点进行扩展至关重要。
    - FoxiPanda 强调了对增加 AMD 产品内存带宽的普遍需求，建议当前的产品可能无法满足内存密集型应用的需求。对于需要快速数据访问和处理的工作负载来说，这是一个关键因素。
    - Stepfunction 指出 AMD Halo Box 是一台 small form factor 计算机，这意味着在可扩展性和冷却方面可能存在限制，但在空间效率和便携性方面也有优势。

### 2. Qwen Model Innovations and Applications

  - **[Qwen-Scope: Official Sparse Autoencoders (SAEs) for Qwen 3.5 models](https://www.reddit.com/r/LocalLLaMA/comments/1szrbub/qwenscope_official_sparse_autoencoders_saes_for/)** (Activity: 393): ****Qwen-Scope** is a newly released collection of Sparse Autoencoders (SAEs) for the **Qwen 3.5 models**, ranging from `2B` to `35B` MoE, designed to map internal features across all layers. This tool acts as a dictionary of the model's internal concepts, allowing for precise interventions such as **Surgical Abliteration** to suppress specific features like refusal, **Feature Steering** to activate desired concepts, and **Model Debugging** to identify token-triggered internal directions. The release is under the **Apache 2.0 license**, but the Qwen team advises against using it to remove safety filters. The tool is demonstrated in a [Space demo](https://hf.co/spaces/Qwen/QwenScope) and detailed in a [technical paper](https://qianwen-res.oss-accelerate.aliyuncs.com/qwen-scope/Qwen_Scope.pdf).** Commenters highlight the significance of this release as potentially the largest open-source interpretability tool for a dense `27B` model, contrasting it with Google's smaller `GemmaScope` variants. There is anticipation for similar tools for future model iterations like Qwen 3.6.

    - NandaVegg highlights the significance of the release of Sparse Autoencoders (SAEs) for the dense 27B Qwen model, noting it as potentially the largest open-source interpretability tool available. This contrasts with previous tools like GemmaScope, which only supported smaller models such as 9B and 2B, indicating a substantial advancement in model interpretability capabilities.
    - robert896r1 expresses anticipation for the release of similar tools for Qwen 3.6, suggesting that the community might adapt existing tools for newer iterations. This reflects a common trend where the community often extends or modifies tools to support the latest model versions, ensuring continued utility and relevance.
    - oxygen_addiction speculates on the use of feature steering in large models, such as ChatGPT5, where a router could dynamically select the best model for a given prompt. This concept involves leveraging interpretability tools to enhance model performance by tailoring responses based on specific features or requirements.

  - **[Qwen 3.6 35b a3b is INSANE even for VRAM-constrained systems](https://www.reddit.com/r/LocalLLM/comments/1szeghg/qwen_36_35b_a3b_is_insane_even_for/)** (Activity: 480): **The post discusses the performance of **Qwen 3.6 35B-A3B**, a local LLM, on a VRAM-constrained system with an **AMD 7700 XT, 32GB DDR4 RAM, and a Ryzen 5 5600**. The user highlights the model's ability to handle complex coding tasks, such as fixing bugs in a web scraper and updating a project README with screenshots, using configurations like `i1-q4_k_s quant`, `128k context`, `flash attention`, and `Q8_0 KV quantization`. The model succeeded where others like **Gemma 3, Gemma 4, and Qwen 2.5 Coder** failed, demonstrating its capability to perform tasks without failed tool calls, even under hardware constraints.** Commenters suggest optimizing performance by moving extra experts to CPU and fitting the KV cache on GPU to achieve over `30 t/s`. Another user questions the long processing time at `16-20 tok/s`, noting their own experience of faster processing at `35-40 tok/s`.

    - GoldenX86 suggests optimizing performance by moving extra experts to the CPU while keeping the KV cache on the GPU, which can increase processing speed to over 30 tokens per second (t/s). This approach is particularly useful for VRAM-constrained systems, allowing for efficient utilization of available resources.
    - AccomplishedFix3476 highlights the potential of running the 35b a3b model on consumer VRAM for coding workflows, noting that local and long-running tasks can reveal memory leaks and context drift issues not apparent in API environments with short time-to-live (TTL). They recommend logging everything initially to catch these issues early.
    - Perfect-Flounder7856 shares a benchmark comparison where the 35b a3b model outperformed the 27b model on a policy reasoning benchmark, scoring 96 versus 92. This indicates the model's superior performance in specific tasks, justifying hardware investments for those seeking high accuracy and speed.




### 3. Mistral Medium 3.5 模型发布

  - **[mistralai/Mistral-Medium-3.5-128B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1sz1qer/mistralaimistralmedium35128b_hugging_face/)** (活跃度: 1120): **Mistral Medium 3.5** 是一款参数量为 `128B` 的稠密（dense）模型，拥有 `256k` 的 context window，专为指令遵循（instruction-following）、推理（reasoning）和编程任务设计。它支持多模态输入（包括文本和图像），并允许针对每个请求配置 reasoning effort，从而在快速回复和复杂推理之间切换。该模型支持多语言和 system prompts，并以 **Modified MIT License** 发布。它取代了之前的 Mistral Medium 3.1 和 Devstral 2 等模型，承诺在统一架构下提供更强的性能。对于复杂任务，建议将 `reasoning_effort` 设置为 "high"，并将 temperature 设置为 `0.7` 以获得最佳性能。评论者们正在测试该模型在不同硬件上的性能，并指出其 `128B` 参数的稠密配置是一个独特特征。此外，还有关于该模型与 Qwen `27B` 等其他稠密模型相比的生态位讨论。

    - IvGranite 分享了使用 `llama.cpp` build 8967 在 Strix Halo 上运行 `mistral-medium-3.5-128b-q4` 模型的性能指标。结果显示生成速度为 `3.26 t/s`，prompt 处理速度为 `46.70 t/s`，其中一项测试的总耗时为 `4.84s`。这表明对于这种规模的模型，其处理时间相对高效，凸显了 `q4` 量化（quantization）在优化性能方面的潜力。
    - grumd 和 reto-wyss 讨论了 128B 稠密模型的意义，grumd 称其为一个“有趣的利基市场（interesting niche）”。reto-wyss 将其与 Qwen 27b 模型进行了比较，询问哪一个更稠密，这暗示了模型密度和性能方面的竞争态势。这反映了人们在平衡模型大小与计算效率方面的持续兴趣。
    - 围绕 `mistral-medium-3.5-128b` 等稠密模型的讨论凸显了处理大规模模型时的挑战与创新。焦点在于如何通过稠密架构（dense architectures）实现高性能，虽然这类架构通常资源密集，但为复杂任务提供了巨大潜力。对话强调了模型量化和优化技术进步的重要性。

  - **[Mistral Medium 3.5 Launched](https://www.reddit.com/r/LocalLLaMA/comments/1sz2mgw/mistral_medium_35_launched/)** (活跃度: 369): **Mistral Medium 3.5** 作为一款 `128B` 稠密模型正式发布，集成了指令遵循、推理和编程能力。该模型以开放权重形式提供，采用修改后的 MIT 许可证（modified MIT license），该许可证规定月收入超过 `$20M` 的公司进行商业使用需支付许可费。该模型支持云端的异步编程任务，允许并行运行多个会话，并在 Le Chat 中为复杂工作流引入了新的 Work 模式。更多详情请参见 [Hugging Face](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) 和 [Mistral 的官方公告](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5)。关于许可条款存在争议，一些用户认为称其为“修改版 MIT 许可证”具有误导性，因为对商业用途的限制偏离了传统的 MIT 许可证条款。

    - Mistral Medium 3.5 模型是一个 1280 亿参数的稠密模型，考虑到目前大参数稠密模型的趋势，这一点意义重大。正如 Septerium 所指出的，这与对稠密架构的持续投入相一致，反映了行业向超稀疏 MoE 模型和 2000 亿参数级别的超稠密模型共同发展的广泛趋势。
    - Long_comment_san 指出，虽然 Mistral Medium 3.5 的 benchmarks 并非顶尖（state-of-the-art），但足以维持人们对大型稠密模型的兴趣。该评论者强调了这些模型作为未来 AI 主力的重要性，暗示行业将继续探索 80B+ 参数级别的稠密模型以及拥有数万亿参数的超稀疏模型。
    - ClearApartment2627 提出了许可问题，认为 Mistral 的许可证（要求月收入超过 2000 万美元的公司为商业用途付费）不应被标记为“修改版 MIT 许可证”。这一区别对于考虑将该模型用于商业应用的公司非常重要，因为它会影响使用该模型的成本和法律后果。


## 非技术类 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude AI 应用与创新

- **[Launched My First App Using Claude](https://www.reddit.com/r/ClaudeAI/comments/1sz38u6/launched_my_first_app_using_claude/)** (热度: 654): **用户发布了一款使用 **Claude** 构建的车辆管理应用，功能包括费用追踪、自定义保养计划、燃油追踪、展示厅模式以及通过 Claude API 实现的 AI 助手。该应用侧重前端并采用本地数据存储，不过 API 调用需要数据库。开发者正在开发 Play Store 版本并寻求增长建议。[应用链接](https://apps.apple.com/app/id6761397650)。** 一位评论者将该应用与 **Vehicle Smart** 进行了对比，指出其在保养功能方面的开发更为出色。另一位询问了开发工具，询问它是用 **Swift**、**Expo** 还是 **Tauri** 构建的。

    - NooneLeftToBlame 讨论了应用的功能，并将其与英国警察常用的热门应用 'Vehicle Smart' 进行了比较。他们指出，虽然 'Vehicle Smart' 具有车牌查询和用于保养提醒的车库功能，但后者的开发程度较低。相比之下，根据截图显示，这款新应用开发得更好，暗示其在用户体验方面具有潜在的竞争优势。
    - barritus 询问了应用的技术栈，询问它是完全用 Swift 编写的，还是使用了 **Expo** 或 **Tauri** 等框架。这体现了对技术实现和技术栈选择的关注，这些因素会影响应用性能和跨平台兼容性。
    - Alternative-Ad-8175 提出了对数据存储的担忧，建议使用云存储以防止手机丢失时数据丢失。他们还提到了个人身份信息 (PII) 的存在，暗示需要采取安全的数据处理措施来保护用户隐私。

  - **[The final nail in the coffin for entry level creative freelancers just dropped](https://www.reddit.com/r/ClaudeAI/comments/1syu949/the_final_nail_in_the_coffin_for_entry_level/)** (热度: 940): **Anthropic** 发布了 Blender MCP 连接器，使 **Claude** 能够通过 Python API 控制 Blender。这种集成允许用户使用自然语言命令创建和修改 3D 场景，实际上在 Blender 中充当了“Copilot”。该工具可以处理调试节点设置、批量更改和添加自定义工具等任务，可能会减少在产品渲染和低多边形 (low-poly) 资产创作等任务中对初级自由职业者的需求。现在，单个用户可以利用 Claude 及其连接工具管理更广泛的创意流程，从剧本创作到最终剪辑。一些评论者对 AI 产出的作品质量表示怀疑，认为这可能会导致低质量游戏和应用的增加。其他人则对这一公告的重要性不以为然，将此类讨论比作煽情媒体。

  - **[Claude is my SEO strategist, content engine, and CTO. From 0 to 10,000 active users in 6 weeks, $0 on ads.](https://www.reddit.com/r/ClaudeAI/comments/1syt37w/claude_is_my_seo_strategist_content_engine_and/)** (热度: 1039): **Reddit 帖子中的图片是一个显示分析数据的仪表板，突出了使用 Claude 和 Lovable 等 AI 工具构建的市场平台 Agensi 在用户参与度方面的显著增长。仪表板显示，在过去 30 天内，活跃用户达到 10,000 名（增长 263.3%），新用户达到 9,900 名（增长 262.0%），且未投入广告费用。这种增长归功于战略性地利用 Claude 进行 SEO、内容策略和 AEO (Answer Engine Optimization)，其中包括分析 Google Search Console 数据以识别关键词空白，并针对 AI 引擎和搜索引擎优化内容结构。** 一些评论者对内容的真实性和原创性表示怀疑，认为它可能是“通用的 AI 垃圾内容 (AI slop)”或垃圾邮件，并质疑帖子本身是否也是由 AI 编写的。

  - **[How not to run an ai company](https://www.reddit.com/r/ClaudeCode/comments/1szi053/how_not_to_run_an_ai_company/)** (Activity: 934): **The image depicts a status dashboard for an AI company, showing multiple services experiencing a 'Major Outage.' The services include 'claude.ai,' 'Claude Console,' 'Claude API,' 'Claude Code,' 'Claude Cowork,' and 'Claude for Government,' with uptime percentages ranging from `98.69%` to `99.88%`. This suggests significant operational challenges in maintaining service reliability, which is critical for AI companies aiming for consistent performance. The title and comments highlight the perception of poor management and the challenges of operating in the fast-paced AI industry, where stability is often sacrificed for rapid development.** Commenters debate whether such outages are typical for cutting-edge AI companies, with some arguing it's part of the 'go fast and break things' approach common in disruptive tech sectors, while others suggest this is not suitable for mature SaaS companies.



### 2. DeepSeek V4 Model Performance and Comparisons

  - **[I wasn’t ready for DeepSeek V4](https://www.reddit.com/r/DeepSeek/comments/1t0aods/i_wasnt_ready_for_deepseek_v4/)** (Activity: 176): **The image showcases a dashboard for DeepSeek V4, highlighting its performance metrics such as spending, token usage, and cache savings. The total spend is noted as `$1,050.86` with cache savings of `$3,351.43`, indicating significant cost efficiency. The dashboard compares different models like DeepSeek Chat, DeepSeek V4 Pro, and DeepSeek V4 Flash, emphasizing the superior performance of the V4 Flash model over others, including the Claude models previously used by the poster. This suggests that DeepSeek V4 models offer a competitive edge in terms of price, speed, and efficiency, challenging existing premium models in the market.** Commenters highlight the revolutionary nature of the V4 models in terms of cost-effectiveness and performance, suggesting that the market has yet to fully recognize their potential. There is also curiosity about the specific dashboard or application used to display these analytics.

    - **DeepSeek V4** is noted for its significant improvements in price, speed, and efficiency, marking a revolutionary step in AI model development. Users highlight that the model's cost-effectiveness is a standout feature, potentially disrupting the market by offering high performance at a lower price point compared to previous versions.
    - The **V4 flash** model is becoming a default choice for many users due to its balanced performance metrics. It is praised for its ability to handle a wide range of tasks efficiently, suggesting that it offers a versatile solution for various applications, which could be a key factor in its adoption.
    - Despite its capabilities, there seems to be a lack of awareness or recognition of **DeepSeek V4's** potential impact on the market. This could be attributed to a general acceptance of high costs in AI solutions, which V4 challenges by providing a more cost-effective alternative without compromising on performance.

  - **[Deepseek V4 pro reminds me of Claude 4.6 sonnet](https://www.reddit.com/r/DeepSeek/comments/1sz84uc/deepseek_v4_pro_reminds_me_of_claude_46_sonnet/)** (Activity: 175): **The post discusses the performance of the **Deepseek V4 Pro** model, comparing it to **Claude 4.6 Sonnet** in terms of creativity and coding capabilities, particularly for HTML tasks. The model is noted for its potential, being in preview, but currently struggles with roleplay consistency and character adherence, often ignoring instructions even at low temperature settings like `0.6`. The user also mentions **Kimi K2.6** as their preferred model for most tasks, while acknowledging Deepseek V4 Pro's improvements over its predecessor, Deepseek V3.2.** Commenters highlight the model's instability and inconsistency in roleplay, with issues in maintaining character traits and scene consistency. One user suggests that **GLM 5.1** outperforms **Kimi K2.6** in coding tasks, indicating a preference for GLM 5.1 in technical applications.



- Flat-Rooster8373 指出了 DeepSeek V4 Pro 在角色扮演（role-playing）场景中一致性的问题，指出该模型难以维持角色的完整性，且即使在 0.6 等较低的 temperature 设置下也经常忽略指令。评论者观察到，使用预设（presets）会加剧这些问题，导致输出重复且充满陈词滥调；而无预设的方法则能产生更好的第一人称推理，尽管最终输出仍与推理过程存在偏差。
- Far-Habit-2713 在编程任务中对比了 DeepSeek V4 Pro 与 Qwen 3.6 Plus，发现 Qwen 在通用编程和调试方面表现出色。然而，DeepSeek V4 Pro 被认为能编写更优的 Rust 代码并提供更详细的代码分析。这表明虽然 Qwen 可能更全面，但 DeepSeek 在特定编程语言和深度分析方面具有优势。
- azvd_ 分享了在 Hermes 平台上使用 DeepSeek V4 Pro 的经验，指出与 Opus 4.7 相比，它的错误更少。这一改进归功于 DeepSeek 增强的理解能力，这与 Opus 为了可能优化其他方面而有意降低理解力的做法形成对比。

- **[bro this is too cheap i think finally i have a respect for the deepseek](https://www.reddit.com/r/DeepSeek/comments/1szyr5z/bro_this_is_too_cheap_i_think_finally_i_have_a/)** (活跃度: 132): **该帖子讨论了 **DeepSeek** 的定价，特别是质疑低价是否针对 **DeepSeek V4 Flash** 版本而非 **Pro** 版本，后者预计在今年晚些时候之前仍将保持高价。一份修改说明指出 Pro 版本目前正在打折。评论中的技术咨询集中在 DeepSeek 与其他 frontier models 相比的质量水平，以及定价是否受到 cache hits 的影响，这可能会影响 output tokens 的成本。** 评论者们正在争论低价是因为临时折扣还是定价策略的根本改变，一些人认为成本效益可能源于影响 token 输出成本的缓存优化。

    - **DeepSeek V4 Flash vs. Pro**: 存在关于 DeepSeek V4 Flash 和 Pro 版本定价差异的讨论。Pro 版本被指出更贵，但目前有折扣。这表明了一种吸引不同用户群体的战略定价模式，可能是由于不同的功能集或性能表现。
    - **缓存系统与成本效率 (Cache System and Cost Efficiency)**: 评论强调了 DeepSeek 基于磁盘的 KV cache 系统，该系统因其鲁棒性和可靠性而受到赞誉，其持续时间可达数小时，而其他供应商通常仅为 5 分钟。该系统通过使缓存输入（cached input）近乎免费，显著降低了成本，这是该模型高性价比的关键因素。
    - **创意任务表现**: 有针对 DeepSeek V4 在创意写作任务中表现的批评，称其与之前版本相比有所退步。然而，它在角色扮演（RP）和 Agent 任务中仍被认为是有效的，这表明创意能力与其他功能之间存在权衡。


### 3. ICML 2026 会议讨论与争议

- **[ICML 2026 Decision [D]](https://www.reddit.com/r/MachineLearning/comments/1szc05y/icml_2026_decision_d/)** (活跃度: 1124): **该帖子讨论了围绕即将发布的 **ICML 2026** 评审结果的期待。社区正急切等待更新，许多人频繁查看 OpenReview 等平台以获取最新信息。这反映了学术界在会议决策期间典型的高度参与感和焦虑感。** 评论幽默地反映了研究人员等待会议决策时的紧张和急躁，突显了反复刷新平台获取更新的普遍行为。

- **[似乎 ICML 正在拒绝大量全票好评的论文 [D]](https://www.reddit.com/r/MachineLearning/comments/1t04vk3/seems_icml_is_rejecting_many_unanimous_positively/)** (活跃度: 202): **该帖子讨论了对 ICML 评审流程的担忧，强调了在 Rebuttal 阶段激励机制的错位。作者指出，评审员感到有压力去调整分数以避免冗长的讨论，导致分数虚高，而这些分数并不一定反映论文的真实水平。这导致许多获得一致正向评分的论文由于会议容量有限而被拒绝。作者建议回归更简单的同行评审流程，由评审员提供独立的评估，并由 Area Chairs (ACs) 评估质量和一致性，通过讨论解决边缘案例。** 评论者对评审流程表示沮丧，指出即使在解决了评审员的担忧后，获得高分的论文仍然被拒绝。有人呼吁建立申诉机制，因为一些人认为单个 AC 的决定可以推翻多个正面评审结果，导致令人沮丧的结局。

    - 几位评论者对 ICML 论文评审流程表示失望，并列举了平均分很高（例如 `4.5` 或 `4/4/4/4`）的论文尽管获得了评审员的正向反馈，却仍被拒绝的情况。一个共同的担忧是 Area Chairs (ACs) 似乎拥有在没有明确申诉机制的情况下推翻一致好评的权力，这引起了作者们的困惑和不满。
    - 一位评论者提到，尽管在 Rebuttal 阶段解决了所有评审员的担忧，他们的论文仍然被拒绝。这表明评审过程与最终决策之间可能存在脱节，已解决的问题仍被列为拒绝理由，暗示可能存在程序效率低下或沟通不畅的问题。
    - 讨论引发了对评审流程透明度和公平性的质疑，一些人认为拒绝可能受到满足录取配额需求的影响，而不仅仅是基于学术价值。这指向了会议论文筛选过程中的系统性问题，即高分论文仍无法保证被接收。

  - **[A* 会议中的中国学术网络/圈子排斥非中国论文 [D]](https://www.reddit.com/r/MachineLearning/comments/1t06564/chinese_nexusnetwork_in_a_conferences_rejecting/)** (活跃度: 112): **该帖子对顶会 AI 会议中涉嫌存在的裙带关系和评审偏见表示担忧，特别是涉及中国学术网络的情况。作者声称，中国评审员可能会偏袒中国作者的论文，这种偏袒可能通过 WeChat 等应用进行协调。引用的一例是，一名评审员因为论文未引用某中国作者的作品而表示不满。据报道，这一问题在 IJCAI 26 等会议中普遍存在，有人声称来自中国大学的非研究质量论文被接收，而非中国作者则面临更严苛的批评。** 评论表明，人们察觉到中国研究人员之间存在协调评审的行为，可能涉及互换评审和通过 WeChat 共享信息。还有轶闻称中国研究人员掌握评审过程的内幕消息，引发了对公平性和透明度的担忧。

    - 一位用户提到，一些知名度较低但受人尊敬的期刊被来自中国大学的论文所占据，这些论文往往缺乏真正的研究质量，更像是工程项目。他们指出，尝试提交类似内容的非中国作者面临更严苛的批评，暗示评审过程中可能存在偏见。
    - 另一位评论者分享了一次经历：一名中国研究人员在评审过程中联系了他们，声称掌握其论文评审的内幕消息。这引发了对评审流程保密性和公平性的担忧，尽管这对论文被拒的直接影响仍存疑。
    - 一位用户观察到，在 ECCV 中，尽管有多篇论文被接收，他们却未被邀请担任评审，而带有中国共同作者的论文则收到了评审邀请。他们注意到一种模式，即一名中国 Area Chair 偏袒中国作者，即使他们的论文分数很低，这引发了对评审和录取过程中潜在偏见的质疑。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们将很快发布新的 AINews。感谢阅读到这里，这是一段美好的历程。