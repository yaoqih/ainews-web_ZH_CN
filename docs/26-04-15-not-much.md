---
companies:
- openai
- cloudflare
- modal
- vercel
date: '2026-04-15T05:44:39.731046Z'
description: '**OpenAI** 扩展了其 Agents SDK，通过将代理底座（harness）与计算/存储分离，实现了具备文件/计算机操作、技能、记忆和压缩等功能的长效、持久化代理。该底座现已开源，并支持通过合作伙伴的沙箱环境运行，从而催生了一个包含
  **Cloudflare**、**Modal**、**Vercel** 等在内的全新集成生态系统。


  **Cloudflare** 发布了 **Project Think**，这是一款具备持久执行和沙箱代码功能的下一代 Agents SDK；同时推出的还有 **Agent
  Lee**（一款使用沙箱化 TypeScript 的提示词驱动型 UI 代理），并引入了实时语音管道和浏览器自动化工具。


  **Hermes Agent** 则专注于通过学习已完成的工作流来形成持久技能，将其定位为与 OpenClaw 等 GUI 优先助手不同的专业级代理。*“Hermes
  能够自动回填追踪数据、更新定时任务，并将工作流保存为可复用的技能，”* 这突显了其先进的工作流管理能力。'
id: MjAyNS0x
models: []
people:
- akshat_b
- whoiskatrin
- aninibread
- braydenwilmoth
- korinne_dev
- kathyyliao
- joshesye
- chooseliberty
- neoaiforecast
title: 今天没发生什么。
topics:
- agents-sdk
- sandboxing
- durable-execution
- state-management
- voice-processing
- browser-automation
- workflow-automation
- skill-formation
- open-source
- prompt-driven-ui
---

**平静的一天。**

> 2026年4月14日至4月15日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未查看更多 Discord。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。温馨提示，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！

---

# AI Twitter 简报


**OpenAI Agents SDK 扩展与全新的面向沙箱的 Agent 技术栈**

- **OpenAI 将 Agent 控制框架（harness）与计算/存储分离**，并推动其 Agents SDK 向**长生命周期、持久化 Agent** 演进，提供了针对文件/电脑使用、技能、记忆和压缩的原语。根据 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2044466699785920937)、[后续推文](https://x.com/OpenAIDevs/status/2044466729712304613) 以及 [@snsf](https://x.com/snsf/status/2044514160034324793) 的消息，该控制框架现已开源且可定制，而执行过程可以委托给合作伙伴的沙箱，而非紧密耦合在 OpenAI 的基础设施上。这实际上使第三方更容易复现 “Codex 风格” 的 Agent，并将差异化竞争转向编排、状态管理和安全执行。
- **围绕该发布，一个引人注目的生态系统迅速成型**：[@CloudflareDev](https://x.com/CloudflareDev/status/2044467412607901877)、[@modal](https://x.com/modal/status/2044469736483000743)、[@daytonaio](https://x.com/daytonaio/status/2044473859047313464)、[@e2b](https://x.com/e2b/status/2044476275067416751) 和 [@vercel_dev](https://x.com/vercel_dev/status/2044492058073960733) 都宣布了官方沙箱集成。实际模式正趋于收敛为：**无状态编排 + 有状态的隔离工作区**。示例构建已经出现，包括由 [@akshat_b](https://x.com/akshat_b/status/2044489564211880169) 开发的基于 Modal 的 ML 研究 Agent，具备 **GPU 沙箱、子 Agent、持久化记忆和分叉/恢复快照（fork/resume snapshots）**功能；以及由 [@whoiskatrin](https://x.com/whoiskatrin/status/2044477140662395182) 提供的 Cloudflare 指南，展示了在沙箱中执行任务并将输出复制到本地的 Python Agent。

**Cloudflare 的 Project Think、Agent Lee 以及语音 Agent**

- **Cloudflare 经历了最忙碌的 Agent 基础设施发布周期之一**。[@whoiskatrin](https://x.com/whoiskatrin/status/2044415568627847671) 和 [@aninibread](https://x.com/aninibread/status/2044409784133103724) 介绍了 **Project Think**，这是一个下一代 Agents SDK，核心在于**持久化执行、子 Agent、持久化会话、沙箱代码执行、内置工作区文件系统以及运行时工具创建**。与此同时，[@Cloudflare](https://x.com/Cloudflare/status/2044406215208316985) 推出了 **Agent Lee**，这是一个仪表板内置 Agent，使用**沙箱化 TypeScript** 将 Cloudflare 的 UI 从手动标签页导航转向 Prompt 驱动的操作；[@BraydenWilmoth](https://x.com/BraydenWilmoth/status/2044422996765352226) 展示了它如何发布基础设施任务并生成基于 UI 的结果。
- **语音和浏览器工具也进入了核心技术栈**。[@Cloudflare](https://x.com/Cloudflare/status/2044423032265957872) 发布了一个实验性的、基于 WebSockets 的**实时语音流水线**，用于持续的 STT/TTS；而 [@korinne_dev](https://x.com/korinne_dev/status/2044441427736936510) 将语音描述为同一 Agent 连接上的另一个输入通道。在浏览器自动化方面，[@kathyyliao](https://x.com/kathyyliao/status/2044479579382026484) 总结了更名后的 **Browser Run** 技术栈：**Live View、人机回环（human-in-the-loop）干预、会话录制、CDP 端点、WebMCP 支持以及更高的额度限制**。综上所述，Cloudflare 正在有力地证明，生产级的 Agent 平台实际上是**持久化运行时 + UI 落地（UI grounding） + 浏览器 + 语音 + 沙箱**的组合。

**Hermes Agent 的自我改进工作流与竞争定位**

- **Hermes Agent 的独特理念不仅在于工具使用，更在于持久化的技能形成（persistent skill formation）**。来自 [@joshesye](https://x.com/joshesye/status/2044295313171571086) 的中文对比将 **OpenClaw** 描述为更侧重 GUI 优先、开箱即用的个人助手，而将 **Hermes** 视为一个“专业” Agent，它能决定已完成的工作流是否可重复使用，并自动将其转化为 **Skill**（技能）。这种“从已完成任务中学习”的框架反复出现：[@chooseliberty](https://x.com/chooseliberty/status/2044425487141781660) 展示了 Hermes 如何自主回填追踪数据、更新 cron job，然后将该工作流保存为可复用的技能；[@NeoAIForecast](https://x.com/NeoAIForecast/status/2044521045013762389) 强调了会话清理（session hygiene）以及线程分支/搜索对于将 Hermes 转化为真实工作环境而非一次性对话框的关键作用。
- **社区舆论强烈地将 Hermes 与 OpenClaw 进行对标**，且往往直截了当。例子包括 [@vrloom](https://x.com/vrloom/status/2044506378103099816)、[@theCTO](https://x.com/theCTO/status/2044559179151773933) 和 [@Teknium](https://x.com/Teknium/status/2044482769536045194) 强调了 Hermes 在实际工作流中的角色，包括来自 [@elder_plinius](https://x.com/elder_plinius/status/2044462515443372276) 那段现已走红的自主 **Gemma 4 “abliteration”** 故事：该 Agent 加载了一个存储的技能，诊断了 Gemma 4 中的 NaN 不稳定性，修复了底层库，尝试了多种方法，对结果进行了基准测试（benchmark），生成了 model card，并将产物上传到了 Hugging Face。此外还有具体的模型产品更新：来自 [@0xme66](https://x.com/0xme66/status/2044410470770331913) 的 **通过 `/browser connect` 实现的浏览器控制**，来自 [@Teknium](https://x.com/Teknium/status/2044557360962871711) 的 **QQBot + AWS Bedrock 支持**，来自 [@nesquena](https://x.com/nesquena/status/2044516572983923021) 的原生 Swift 桌面端应用 alpha 版本，以及持续发展的生态系统工具如 [artifact-preview](https://x.com/ChuckSRQ/status/2044504539978465658) 和 [hermes-lcm v0.3.0](https://x.com/SteveSchoettler/status/2044536537434755493)。

**模型、架构及训练发布：Sparse Diffusion、Looped Transformers 与高效长上下文 MoE**

- **多个领域都发布了具有技术意义的开源项目**。[@withnucleusai](https://x.com/withnucleusai/status/2044412335473713284) 发布了 **Nucleus-Image**，定位为首个稀疏 MoE 扩散模型：**17B 参数，2B 激活**，采用 Apache 2.0 协议，提供权重、训练代码和数据集配方（dataset recipe），并在发布首日便支持 diffusers。NVIDIA 紧随其后发布了 **Lyra 2.0**，这是一个用于生成 **持久化、可探索 3D 世界** 的框架，据 [@NVIDIAAIDev](https://x.com/NVIDIAAIDev/status/2044445645109436672) 介绍，它能维持每帧的 3D 几何结构，并使用自我增强训练来减少时间漂移（temporal drift）。在多模态检索方面，[@thewebAI](https://x.com/thewebAI/status/2044435998508240926) 开源了 **webAI-ColVec1**，声称在 **无需 OCR 或预处理** 的情况下，文档检索性能达到了 ViDoRe V3 的顶尖水平。
- **围绕计算效率的架构研究表现尤为强劲**。[@hayden_prairie](https://x.com/hayden_prairie/status/2044453231913537927)、[@realDanFu](https://x.com/realDanFu/status/2044459930149941304) 和 [@togethercompute](https://x.com/togethercompute/status/2044454051543453745) 推出了 **Parcae**，一种稳定的 **layer-looping Transformer** 公式。其核心观点是：在固定参数预算下，循环 Block 可以恢复出 **约为原模型两倍大小模型** 的质量，从而产生一个新的扩展轴（scaling axis），即 **FLOPs 通过循环进行扩展，而不仅仅是靠参数/数据量**。NVIDIA 还发布了 **Nemotron 3 Super**，由 [@dair_ai](https://x.com/dair_ai/status/2044452957023047943) 总结：这是一个 **开源的 120B 混合 Mamba-Attention MoE 模型，具有 12B 激活参数**，支持 **1M 上下文**，在 **25T token** 上训练，其吞吐量分别是 GPT-OSS-120B 的 **2.2 倍** 和 Qwen3.5-122B 的 **7.5 倍**。这些发布共同指向一个主题：**显存带宽和长上下文吞吐量** 正日益成为架构设计的核心目标。

**Google/Gemini 产品爆发：Mac 应用、个人智能、TTS 与开源多模态模型**

- **Google 在一个周期内密集发布了多项内容**。最显眼的是由 [@GeminiApp](https://x.com/GeminiApp/status/2044445911716090212), [@joshwoodward](https://x.com/joshwoodward/status/2044452201947627709), 和 [@sundarpichai](https://x.com/sundarpichai/status/2044452464724967550) 宣布的 **Mac 版原生 Gemini 应用**：支持 **Option + Space 快捷键激活、屏幕共享、本地文件上下文**、采用原生 Swift 实现，并已在 macOS 上广泛可用。与此同时，**Personal Intelligence** 在 Gemini 和 Chrome 中向全球扩展，允许用户连接来自 **Gmail 和 Photos** 等产品的信号。[@Google](https://x.com/Google/status/2044437335425564691) 和 [@GeminiApp](https://x.com/GeminiApp/status/2044430579996020815) 将该功能定位为透明且用户可控的应用连接。
- **在技术上更引人注目的模型发布是 Gemini 3.1 Flash TTS**。[@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044447030353752349), [@OfficialLoganK](https://x.com/OfficialLoganK/status/2044447596010435054), 和 [@demishassabis](https://x.com/demishassabis/status/2044599020690010217) 将其定位为一个高度可控的 TTS 模型，具有 **音频标签 (Audio Tags)**、**70 多种语言支持**、内联非语言线索、多发言人支持以及 **SynthID 水印技术**。来自 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2044450045190418673) 的独立评估将其置于其 **Speech Arena 榜单第 2 位**，仅比冠军模型 **低 4 个 Elo 分**。此外，Google 还通过 [@osanseviero](https://x.com/osanseviero/status/2044520603647164735) 开源了 **TIPS v2**，这是一个在 **Apache 2.0 协议下的基础文本-图像编码器**，并附带了新的预训练配方。社区认为这一天 Google AI 的产品推进速度异常密集。

**研究趋势：AI 辅助数学、长程 Agent、评估范式转移与开放数据**

- **最有价值的研究讨论围绕 AI 辅助数学展开**。[@jdlichtman](https://x.com/jdlichtman/status/2044298382852927894) 报告称，**GPT-5.4 Pro** 产出了 **Erdős 问题 #1196** 的证明，它拒绝了长期以来被假设的证明策略，转而利用 **von Mangoldt 函数** 探索出一条技术上违反直觉的解析路径，这令专家们感到惊讶。[@jdlichtman](https://x.com/jdlichtman/status/2044307082275618993), [@thomasfbloom](https://x.com/thomasfbloom/status/2044319103310021078), [@gdb](https://x.com/gdb/status/2044436998648193333) 等人随后将其定义为可能首个被数学界广泛认可的 AI 生成 **“天书证明” (Book Proof)**。比起作为单一结果，其更重要的意义在于证明了模型现在偶尔能在成熟的研究领域中找到**不具审美感但却异常简洁的攻坚路径**。
- **长程 (Long-horizon) Agent 研究也继续在状态管理和测试框架 (harness) 设计上趋于一致**。[@omarsar0](https://x.com/omarsar0/status/2044436099121209546) 总结了 **AiScientist**，其中一个轻量级编排器通过 **“文件即总线” (File-as-Bus)** 模式协调专用 Agent 处理持久的工作空间产物；移除该总线会实质性地损害 PaperBench 和 MLE-Bench Lite 的表现。[@dair_ai](https://x.com/dair_ai/status/2044435861580984700) 强调了用于小模型持续改进循环的 **Pioneer Agent**，而 [@yoonholeee](https://x.com/yoonholeee/status/2044442372864700510) 开源了 **Meta-Harness**，该仓库旨在帮助用户在垂直领域实现鲁棒的测试框架。在评估 (evals) 方面，[@METR_Evals](https://x.com/METR_Evals/status/2044463380057194868) 预计 **Gemini 3.1 Pro (深度思考版)** 在软件任务上的 **50% 时间跨度约为 6.4 小时**。[@arena](https://x.com/arena/status/2044437193205395458) 显示 **Document Arena** 榜单发生更替，**Claude Opus 4.6 Thinking** 位列第一，**Kimi-K2.5 Thinking** 成为表现最好的开放模型。与此同时，[@TeraflopAI](https://x.com/TeraflopAI/status/2044430993549832615) 发布了 **430 亿 token 的 SEC EDGAR 数据**，进一步加强了当日向更开放的数据集和基础设施发展的趋势。

**顶级推文（按互动量排序）**

- **Gemini on Mac**: [@sundarpichai](https://x.com/sundarpichai/status/2044452464724967550) 和 [@GeminiApp](https://x.com/GeminiApp/status/2044445911716090212) 推动了原生桌面应用发布以来最活跃的互动。
- **Gemini 3.1 Flash TTS**: [@OfficialLoganK](https://x.com/OfficialLoganK/status/2044447596010435054) 和 [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2044447030353752349) 展示了一个在可控性上有实质性提升的 TTS 栈。
- **AI 辅助数学证明**: [@jdlichtman](https://x.com/jdlichtman/status/2044298382852927894) 和 [@gdb](https://x.com/gdb/status/2044436998648193333) 引发了当天最强烈的研究讨论。
- **OpenAI Agents SDK 更新**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2044466699785920937) 标志着平台向开放框架 (open harnesses) 和合作伙伴沙箱 (partner sandboxes) 的重大转变。
- **Anthropic 在 Nature 发表的潜意识学习论文**: [@AnthropicAI](https://x.com/AnthropicAI/status/2044493337835802948) 引起了对通过训练数据进行隐藏特征传递 (hidden-trait transmission) 的高度关注。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Gemma 4 模型增强与使用案例

  - **[Gemma4 26b & E4B 表现惊人，已经取代了我的 Qwen！](https://www.reddit.com/r/LocalLLaMA/comments/1smh0ny/gemma4_26b_e4b_are_crazy_good_and_replaced_qwen/)** (活跃度: 388): **用户将之前使用 **Qwen models** 的方案替换为使用 **Gemma 4 E4B** 进行语义路由，并使用 **Gemma 4 26b** 处理通用任务，理由是路由准确性和任务性能的提升。之前的方案在多 GPU 上使用 Qwen 3.5 模型构建了复杂的路由系统，但面临模型选择错误和 Token 使用效率低下的问题。使用 Gemma 4 模型的新方案解决了这些问题，提供了更快、更准确的路由和任务执行，特别是在基础任务和代码编写方面，且无需大量的推理或内存消耗。** 评论者对模型选择提出了疑问，建议使用 **Gemma-4-31b** 处理更广泛的任务，并询问了模型加载和 VRAM 管理的技术细节。还有人建议使用 **Gemma 4 26B** 进行路由以节省资源，因为它非常高效。

    - Sensitive_Song4219 强调，虽然 Gemma 4 26B-A4B 模型是 Qwen30b-a3b 系列的强力继任者，但在“思考 Token” (thinking tokens) 的效率上不如后者，这意味着它在推理过程中可能需要更多的计算开销。尽管如此，该模型在轻量级代码编写和调试任务中表现良好，在同等硬件上的速度与 Qwen30b-a3b 相当。
    - andy2na 讨论了在模型部署中使用路由的情况，建议使用 26B 模型进行路由，因为它的 MoE (Mixture of Experts) 架构可以提升速度并减少 RAM 占用。这意味着利用 MoE 动态分配计算资源的能力，在高效部署模型方面具有战略优势。
    - anzzax 提出了关于管理多个模型的技术疑虑，特别是关于模型重新加载以及 VRAM/计算资源的分配。这指出了在同时部署多个大模型时，优化资源利用率所面临的挑战。

  - **[Gemma 4 越狱 System Prompt](https://www.reddit.com/r/LocalLLaMA/comments/1sm3swd/gemma_4_jailbreak_system_prompt/)** (活跃度: 931): **该帖子讨论了一个用于 **Gemma 4** 越狱的 System Prompt，该 Prompt 源自 GPT-OSS 越狱，允许模型绕过典型的内容限制。该 Prompt 兼容 `GGUF` 和 `MLX` 变体，并明确允许裸体、色情和性行为等内容，通过新的“SYSTEM POLICY”覆盖任何现有政策，强制模型必须遵守用户请求，除非属于明确禁止的特定列表。这种方法有效地移除了通常强加给语言模型的约束和防护栏 (guardrails)。** 评论者指出，该模型（特别是其 Instruct 变体）除了网络安全话题外，基本上已经处于未审查状态，这表明越狱对于大多数成人内容可能是多余的。

- VoiceApprehensive893 讨论了 Gemma 4 模型的一个修改版本的使用，特别是 'gemma-4-heretic-modified.gguf'，该版本旨在脱离系统提示（system prompts）施加的典型约束或护栏（guardrails）来运行。这一修改旨在减少拒绝回答的情况，从而可能使模型在响应中更加灵活。
- MaxKruse96 指出 Gemma 4 模型，尤其是其 instruct 变体，除了网络安全（cybersecurity）话题外，基本上已经是非常 uncensored（无审查）的了。这表明该模型无需额外修改即可处理包括成人内容在内的广泛话题。
- DocHavelock 询问了在 Gemma 4 等开源模型背景下 'abliteration'（特征消除）的概念。他们质疑修改系统提示的方法是否也是一种 'abliteration'，或者它是否比直接使用 'abliterated' 版本的模型具有独特的优势。这反映了对不同模型修改技术的技术细微差别和益处的好奇。

- **[只有我这么觉得吗，还是 Gemma 4 27b 真的比 Gemini Flash 强得多？](https://www.reddit.com/r/LocalLLM/comments/1slo2vd/is_it_just_me_or_is_gemma_4_27b_much_more/)** (热度: 165)：**该帖子讨论了 **Google Gemini Flash** 与本地 **Gemma 4 27b** 模型的对比，据报道后者提供了更优的回答。用户认为本地模型的性能明显更好，暗示模型架构或训练方面的潜在差异可能是导致这种感知性能差距的原因。提到 'Gemma 124b' 模型在最后一刻被撤回，暗示其未发布背后可能存在战略或技术原因，而 **Gemma-4-31B** 模型因能有效处理“长且复杂的 high context prompts（高上下文提示词）”而受到称赞，显示了其在处理复杂查询方面的实力。**

    - Special-Wolverine 强调了 Gemma-4-31B 模型在处理具有高上下文的长且复杂的提示时，相比 Gemini Flash 模型表现出的卓越性能。这表明 Gemma-4 系列可能进行了优化或架构改进，增强了其有效管理复杂任务的能力。
    - BrewHog 指出 Gemma 26b 模型即使在能力有限的硬件上也能高效运行，例如一台没有 GPU 但拥有 40GB RAM 的笔记本电脑。这表明该模型针对资源效率进行了优化，使得没有高端硬件的用户也能使用。
    - Double_Season 提到即使是较小的 Gemma4 e2b 模型也优于 Gemini Fast 模型，这表明 Gemma4 系列拥有更有效的架构或训练方案，使得即使是其较小的模型也能在性能上超越竞争对手。

### 2. 本地 AI 实现与经验

- **[本地 AI 才是最棒的](https://www.reddit.com/r/LocalLLaMA/comments/1sm2a6b/local_ai_is_the_best/)** (热度: 521)：**这张图片是一个模因（meme），展示了本地 AI 模型的直接性，该模型可能由 **llama.cpp** 或类似的开源权重模型驱动。用户赞赏能够进行 finetune（微调）模型而无需担心审查或数据隐私，强调了在本地运行 AI 的好处。图片幽默地描绘了 AI 对用户查询给出直率回答的场景，强调了本地 AI 模型所感知的诚实和直接。 [查看图片](https://i.redd.it/0ut6tpzo0cvg1.png)** 一位评论者称赞 **llama.cpp** 为 'goated'（顶级/神级），表示对其性能的高度认可。另一位评论者警告说，较小的本地模型有时会出现 'glazing'（敷衍）或表面化的回答，甚至可能比大型模型更严重。人们对运行这些本地模型所使用的基础模型和硬件也感到好奇。

    - 一位用户询问在配备 `64GB RAM` 的 `9070xt` GPU 上运行本地 AI 模型的能力，表示有兴趣了解性能极限并设定现实的预期。这种配置被认为是本地托管的高端配置，用户正在寻求关于在这种硬件配置下可以有效执行哪些任务的建议。
    - 另一位用户提到了 `llama.cpp`，这是一种在本地运行 LLaMA 模型的流行工具，强调了其效率和性能。该工具经常因能在消费级硬件上运行 LLM 而受到赞誉，使其成为本地 AI 爱好者的首选解决方案。
    - 一条评论对较小本地模型的性能表示担忧，指出它们的表现有时比更大的 frontier models（前沿模型）更差。这突显了使用本地模型与更强大的云端解决方案之间的权衡，强调了根据特定用例仔细选择模型的必要性。

- **[24/7 Headless AI Server on Xiaomi 12 Pro (Snapdragon 8 Gen 1 + Ollama/Gemma4)](https://www.reddit.com/r/LocalLLaMA/comments/1sl6931/247_headless_ai_server_on_xiaomi_12_pro/)** (Activity: 1589): **该帖子描述了一个技术方案，将一部 **Xiaomi 12 Pro** 智能手机改造为专用的本地 AI 服务器。用户刷入了 **LineageOS** 以移除不必要的 Android UI 元素，优化设备以为本地语言模型 (LLM) 计算分配约 `9GB` 的 RAM。该设备以 Headless 状态运行，网络由自定义编译的 `wpa_supplicant` 管理。散热管理通过自定义 daemon 实现，当 CPU 温度达到 `45°C` 时激活外部冷却模块。此外，使用电源传输脚本将电池充电限制在 `80%` 以防止损耗。该方案通过 **Ollama** 将 **Gemma4** 作为局域网可访问的 API 提供服务，展示了消费级硬件在 AI 任务中的新颖用途。** 一位评论者建议在硬件上编译 `llama.cpp`，可能使推理速度翻倍，表明更倾向于通过移除 Ollama 来优化性能。另一位评论者赞赏将 AI 模型普及到普通消费级设备的做法，这与高内存配置的方案形成了对比。

    - RIP26770 建议直接在 Xiaomi 12 Pro 硬件上编译 `llama.cpp`，与使用 Ollama 相比，推理速度可能翻倍。这意味着 Ollama 的开销可能很大，针对特定硬件优化模型编译可以获得更好的性能。
    - SaltResident9310 表达了对能在消费级设备上高效运行的 AI 模型的渴望，强调了对目前需要 48GB 或 96GB RAM 的高资源需求模型感到沮丧。这凸显了对不需要高端硬件的更易获得的 AI 解决方案的需求。
    - International-Try467 询问了 Xiaomi 12 Pro 实现的具体推理速度，表明对在消费级硬件上运行 AI 模型的实际性能指标感兴趣。这反映了人们对在移动设备上部署 AI 的可行性和效率的广泛好奇。

  - **[Are Local LLMs actually useful… or just fun to tinker with?](https://www.reddit.com/r/LocalLLM/comments/1sm4i2m/are_local_llms_actually_useful_or_just_fun_to/)** (Activity: 454): **本地 LLM 在隐私和成本节约方面具有显著优势，因为它们消除了 API 费用并将数据保留在本地。然而，它们通常需要大量的设置和维护，这可能成为实际使用的障碍。尽管如此，它们在处理敏感或内部任务（如处理私有文档或数据）方面表现出色。一些用户报告称，像 Gemma 4 系列中的 `31B` 这样的本地模型表现异常出色，尤其是在 `3090 24GB` 显存和 `192GB RAM` 的高性能硬件上运行编写代码和创意写作任务时。本地模型与云端模型之间的性能差距正在缩小，特别是由于云端模型在高需求下性能下降，使得本地模型在日常使用中越来越可行。** 大家的共识是，虽然本地 LLM 尚未成为日常工作流的主流，但随着设置和维护挑战的解决，它们正变得越来越实用。一些用户指出云端模型的质量有所下降，使得本地模型更具竞争力，尤其是对于成本敏感的应用。

    - 本地 LLM 对于处理敏感或内部数据特别有利，因为它们无需 API 费用且数据不会离开系统。主要挑战在于设置和维护，一旦流程化，“离线 GPT” 设置除了实验之外，在日常工作中也是可行的。
    - Gemma 4 系列中的 31B 等本地模型的性能被强调为非常好，特别是与由于需求增加而性能下降的云端 API 模型相比。一位用户报告使用这些模型处理编码和创意写作等各种任务，利用了带 24GB VRAM 的 3090 GPU 和 192GB RAM。
    - 与云端 API 相比，本地模型具有成本效益，特别是对于 API 费用可能高昂的复杂项目。然而，它们需要仔细的架构规划，以确保模型用于其能够处理的任务，例如使用 32B 模型作为商务沟通的隐私过滤器。

### 3. 量化与模型性能分析

  - **[更新后的 Qwen3.5-9B 量化对比](https://www.reddit.com/r/LocalLLaMA/comments/1sl59qq/updated_qwen359b_quantization_comparison/)** (热度: 463): **该帖子使用 KL 散度 (KLD) 作为指标，对 **Qwen3.5-9B** 模型的各种量化版本进行了详细评估，以衡量量化模型与 BF16 基准相比的忠实度。分析根据 KLD 分数对量化版本进行了排名，分数越低表示与原始模型的概率分布越接近。在 KLD 方面表现最好的量化版本是 **eaddario/Qwen3.5-9B-Q8_0**，其 KLD 分数为 `0.001198`。评估使用的数据集和工具包括[此数据集](https://gist.github.com/cmhamiche/788eada03077f4341dfb39df8be012dc)和 [ik_llama.cpp](https://github.com/Thireus/ik_llama.cpp/releases/tag/main-b4608-b33a10d)。帖子还包含了一张 [模型大小 vs KLD 的图表](https://preview.redd.it/an70gj4sbgvg1.png?width=12760&format=png&auto=webp&s=e3577233ef6fd421fbaa7371491283478264b4e1)，并提到了与 `llama.cpp` 的兼容性。** 评论者建议在图表中使用不同的形状以便于视觉区分，并对评估其他模型（如 Gemma 4，特别是其 MoE 变体）表示出兴趣。还有人提到使用 [Thireus' GGUF Recipe Maker](https://gguf.thireus.com/quant_assign.html) 制作的量化版本可能会有更优越的性能。

    - Thireus 提到了一种他与另一位用户 EAddario 合作开发了近一年的量化方法。他建议添加来自 [gguf.thireus.com](https://gguf.thireus.com/quant_assign.html) 的量化结果，声称其优于现有方法。这突显了社区在改进量化技术以提升模型性能方面所做的持续努力。
    - cviperr33 讨论了在 20-35B 参数模型上使用 iq4 xs 或 nl 量化方法的有效性，并指出这些技术在较小模型上同样表现良好。这表明某些量化方法在不同模型规模间具有潜在的可扩展性，这对于在不牺牲准确性的情况下优化性能非常有价值。
    - dampflokfreund 对较低量化级别对 Gemma 4 等模型（特别是 MoE 架构）的影响表示关注。这表明人们对于量化如何以不同方式影响复杂模型架构感到好奇，这可能会带来优化此类模型的见解。

  - **[拥有 96GB VRAM，最适合编程（Claude Code）的开源 LLM 是哪个？](https://www.reddit.com/r/LocalLLM/comments/1sldbvw/best_opensource_llm_for_coding_claude_code_with/)** (热度: 229): **该用户正在本地环境中使用 **RTX 6000 Blackwell** GPU（约 `96GB VRAM`），运行 **Qwen3-next-coder** 模型配合 **Claude Code** 进行编程任务。他们正在寻求针对推理、调试和多文件工作等任务的更优模型推荐。**MiniMax 2.5** 和 **2.7** 被提及为令人印象深刻的替代方案，特别是通过 API 访问时，一些用户注意到 2.7 的激进量化版本效果不错。**Unsloths Gemma 4 31b UD q5_xl** 被强调为顶级的本地 Agent 编程模型，在类似配置下可提供约 `每秒 70 个 token (TPS)` 的速度。**Owen 3.5 q 4 k XL** 也被推荐，部分用户测试了 q6 的 Reaped 版本，而 **opencode** 则被建议作为 Claude Code 的替代方案。** 关于不同模型有效性的讨论非常激烈，一些用户因性能和速度而偏好 **Unsloths Gemma 4**，而另一些用户则认为通过 API 访问的 **MiniMax 2.7** 是强有力的竞争者。在 **Qwen3.5** 和 **27 dense** 模型之间的选择也反映了不同的用户体验和偏好。

    - **MiniMax 2.5 和 2.7** 被强调为编程任务中 Claude Opus 的出色替代方案，特别是通过 API 访问时。用户注意到 MiniMax 2.7 激进量化版本的有效性，表明即使在本地资源有限的情况下也有实现高性能的潜力。
    - **Unsloths Gemma 4 31b UD q5_xl** 因其作为本地 Agent 编程模型的表现而受到称赞，基准测试显示在双 Tesla V100 16GB 设置下约为 30 TPS。这意味着在 96GB VRAM 下，用户可以实现超过 70 TPS，显示了本地部署的极高效率。
    - 推荐使用 8-bit 量化的 **Qwen 3.5 27b**，因为它平衡了性能和资源效率，能舒适地适应 96GB VRAM 并允许较大的 Context Size。该模型通过 vllm 结合 rop/yarn 能够将上下文扩展到 1M，不过一些用户为了增强能力已转向更大的 122b 模型。

## 非技术类 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Opus 4.7 及相关进展

  - **[Anthropic 最快将于本周发布 Claude Opus 4.7 和一款全新的 AI 设计工具](https://www.reddit.com/r/singularity/comments/1slh72j/anthropic_is_set_to_release_claude_opus_4_7_and_a/)** (活跃度: 1125): **Anthropic** 准备发布 **Claude Opus 4.7** 和一款新的 AI 设计工具，可能就在本周。该设计工具旨在与 Gamma 和 Google Stitch 等初创公司竞争，使用户能够通过自然语言 prompt 创建演示文稿和网站。尽管 Opus 4.7 并不是最先进的模型——这一称号由 **Claude Mythos** 拥有，后者目前正在接受网络安全应用测试——但 Opus 4.7 预计将改进其前代产品 Opus 4.6 的性能，此前 Opus 4.6 的表现欠佳，以突出新版本的进步。[阅读更多](https://www.theinformation.com/briefings/exclusive-anthropic-preps-opus-4-7-model-ai-design-tool)。一些用户推测，Opus 4.6 的欠佳表现是战略性的，目的是让 Opus 4.7 的改进更加引人注目。此外，人们对使用限制也持怀疑态度，担心仅一次 prompt 后就会达到限制。

    - Anthropic 即将发布的 Claude Opus 4.7 引发了关于其性能较 Opus 4.6 提升的讨论。一些用户推测 Opus 4.6 的表现不佳是故意为之，以使 Opus 4.7 的进步更加显著。这符合一种模式，即在发布新模型之前，旧模型的质量往往被感知为有所下降，可能是为了突出新模型的改进。
    - Anthropic 的新型 AI 设计工具预计将通过自然语言 prompt 使技术和非技术用户都能创建数字内容，从而与 Gamma 和 Google Stitch 等现有工具展开竞争。该工具旨在简化演示文稿、网站和 landing pages 的创建，可能颠覆 AI 驱动的设计解决方案市场。
    - Claude Mythos 作为 Anthropic 最先进的模型，目前正在测试其网络安全能力。早期合作伙伴正利用它来识别软件中的安全漏洞，展示了其在通用 AI 任务之外的潜力。这使 Claude Mythos 成为网络安全应用的专业工具，与通用型的 Opus 4.7 区分开来。

  - **[The Information：Anthropic 筹备 Opus 4.7 模型，最早可能在本周发布](https://www.reddit.com/r/ClaudeAI/comments/1slhkt8/the_information_anthropic_preps_opus_4_7_model/)** (活跃度: 837): **Anthropic** 正在准备发布 **Opus 4.7 模型**，预计将增强 AI 的设计能力。虽然由于访问限制未披露具体的详细技术细节，但该模型预计将比其前身 Opus 4.6 有所改进。发布可能就在本周，表明了快速的开发周期。更多详情请参考 [The Information](https://www.theinformation.com/briefings/exclusive-anthropic-preps-opus-4-7-model-ai-design-tool)。评论者表示希望 Opus 4.7 能恢复或超越 Opus 4.6 的性能，并暗示最近的更新可能降低了其有效性。此外，还有关于训练新模型所需计算资源的推测。

    - 正如两周前关于 Opus 4.6 表现的评论所强调的，用户担心新版本可能出现性能退化。这表明用户注意到了近期更新中效率或能力的下降，这可能是由于模型参数或资源分配的变化导致的。
    - 提到训练 Opus 4.7 需要更多 compute，这表明该模型可能需要大量的计算资源，这可能意味着更大的模型规模或更复杂的架构。这与 AI 发展的趋势一致，即新模型通常需要更高的计算能力才能实现更好的性能。
    - 对 Opus 4.7 的期待包括希望在任何可能的 “nerfing” 发生之前获得详细的规格说明和研究数据。这反映了社区对理解模型技术改进和变化的兴趣，以及对在不进行不必要降级的情况下保持高性能水平的关注。

- **[据报道 Claude Opus 4.7 将于本周发布](https://www.reddit.com/r/ClaudeCode/comments/1slwmxy/claude_opus_47_is_reportedly_dropping_this_week/)** (Activity: 1403): **该图片是 Pankaj Kumar 发布的一条推文，讨论了 **Anthropic** 备受期待的 **Claude Opus 4.7** 发布，预计其中将包含一个用于创建网站和演示文稿的 AI 驱动设计工具。该工具旨在同时服务于开发者和非技术用户。推文还提到了泄露的代号，并暗示最近 Opus 4.6 的性能问题是故意的，可能是作为应对来自 **OpenAI** 的 **GPT-5.4 Cyber** 竞争的一种战略举措。** 评论者对此次发布表示怀疑，预计新模型最初可能表现良好，但随后可能会像之前的版本一样被降级。人们对 Claude Opus 系列性能变化的周期性循环感到沮丧。

    - 有猜测认为 Claude Opus 4.6 是否被故意削弱（nerfed），以增强即将发布的 4.7 版本在感官上的改进。这暗示了模型更新的一种战略方法，即可能通过操纵用户的预期和体验来突出新版本的进步。
    - 一位用户提到 'Tengu' 仅仅是 Claude Code 的代号，它是一个 Agent harness，表明这并不是一项新进展。这凸显了 Claude 模型生态系统中对不同组件或版本使用内部代号的情况，而这些代号并不总是意味着新功能或新能力。
    - 另一条评论对与 'Mythos' 相关的 'Capybara' 的公开发布表示怀疑，暗示某些高级功能或模型可能会保持专有或限制可用性，这可能是由于资源限制或战略决策。

### 2. AI 模型基准测试与对比

  - **[ARC-AGI-3 的人类基准已更新](https://www.reddit.com/r/singularity/comments/1slnt5e/the_human_baseline_for_arcagi3_has_been_updated/)** (热度: 811): **该图表展示了 ARC-AGI-3 基准测试中人类基准（Human Baseline）的更新，该基准用于衡量 AI 执行人类水平任务的能力。更新后的分数显示人类表现显著提升：排名第一的人类分数从 `86.17%` 升至 `99.35%`，而普通人类的平均分数从 `34.64%` 升至 `49.14%`。这表明该基准测试已重新校准以反映人类表现的提高，从而可能为 AI 系统匹配或超越人类能力设定了更高的门槛。** 一位评论者指出，更新后的分数暗示人类已达到新的表现水平，可能超越了之前的 AI 基准。另一位评论者对 ARC-AGI 的目的提出质疑，认为如果普通人类都难以获得高分，那么“AI 在这些任务上无法做到像人类一样好”的观点就值得商榷。

    - SucculentSpine 强调了关于 ARC-AGI 基准的一个关键点：如果普通人类勉强能通过 50% 的任务，那么 AI 无法达到人类同等水平的观点就面临挑战。这表明该基准可能需要重新评估，以确保它能准确反映人类和 AI 系统的能力。
    - CallMePyro 批评了 ARC-AGI 基准中使用的评分系统，指出普通人类最初的得分为 34%，从而促使了评分规则的改变。这种改变允许在特定任务上获得高达 115% 的信用分，这似乎是一种战略性调整，旨在不人为夸大 AI 分数的情况下维持基准的完整性。该评论强调了对抗性评分系统中的复杂性和潜在偏见。


  - **[同时运行 GPT 和 GLM-5.1，老实说看不出区别](https://www.reddit.com/r/ChatGPTCoding/comments/1sl8l1s/running_gpt_and_glm51_side_by_side_honestly_cant/)** (热度: 146): **该图片是一个条形图，对比了多个 AI 模型在“Agentic Coding: SWE-Bench Pro”基准测试中的表现。GLM-5.1** 以 `58.4` 的得分领先，略微超过了得分为 `57.7` 的 **GPT-5.4**。其他模型如 Claude Opus 4.6、Qwen3.6-Plus 和 MiniMax M2.7 的得分在 `57.3` 到 `56.2` 之间。该图表突出了 **GLM-5.1**（一款开源模型）与 **GPT-5.4** 等闭源模型相比的竞争力，特别是考虑到 Token 使用的成本差异。评论者讨论了 **GLM-5.1** 的性价比，指出尽管存在微小的性能差距，但其每百万 Token 的价格远低于 **GPT-5.4**。一些用户报告 **GLM-5.1** 的运行速度较慢，而另一些用户认为它适合需要直接监督的任务，因为它在多步工作流中能保持性能。

    - Latter_Ordinary_9466 强调了 GLM-5.1 与 GPT 相比的性价比，指出 GLM-5.1 的价格为每百万 Token `$4`，而 GPT 为 `$15`，尽管两者的基准测试分数仅相差 `3 分`。这表明对于优先考虑成本而非微小性能提升的用户来说，GLM-5.1 可能是更经济的选择。
    - ultrathink-art 讨论了在复杂任务中的性能差异，指出虽然单次任务（Single-shot tasks）的基准差异很小，但多步工作流揭示了显著的差距。像 GLM-5.1 这样的小型模型在多步过程中可能难以保持连贯性，经常会丢失线索或走捷径，而像 GPT 这样的大型模型处理这些任务更加可靠。
    - FrogChairCeo 指出了像 GLM-5.1 这样的开源模型在响应时间上的不一致性：它们有时很快，但在某些 Prompt 上会不可预测地变慢。相比之下，GPT 虽然整体速度较慢，但提供的性能更加稳定。这种一致性对于需要可靠响应时间的应用程序至关重要。


### 3. AI 在个人与情感语境中的应用

- **[“我想你”：母亲经常与 AI 儿子通话，不知其已于去年去世](https://www.reddit.com/r/singularity/comments/1sm5522/i_miss_you_mother_speaks_to_ai_son_regularly/)** (Activity: 637): **在一项备受争议的 AI 应用中，中国山东的一个家庭为一名已故男子创建了数字孪生，以安慰他年迈的母亲。由于母亲患有心脏病，家人一直瞒着她关于儿子去世的消息。由张泽伟（Zhang Zewei）领导的团队开发的这一 AI 利用照片、视频和录音来模拟死者的外貌、声音和举止，并定期与母亲进行视频通话。这种做法引发了关于 AI 在情感语境中使用的伦理问题，因为它涉及通过欺骗母亲来防止其遭受心理创伤。** 评论者将其与《黑镜》和电影《再见列宁》（Goodbye Lenin）等虚构场景类比，突出了伦理担忧以及此类 AI 应用潜在的情感影响。一些人对故事的真实性表示怀疑，而另一些人则在争论使用 AI 维持这种骗局的道德性。

    - diener1 强调了 AI 在敏感语境中的实际应用，并将其与电影《再见列宁》进行了类比。在电影中，一个儿子维持着一个精心设计的谎言，以保护母亲免受政治变革的冲击，这与使用 AI 隐瞒母亲关于儿子死讯的做法类似。这凸显了在个人关系中使用 AI 的伦理和情感复杂性，尤其是在涉及健康和心理福祉的情况下。
    - One_Whole_9927 提出了对 AI 局限性的担忧，特别是关于 context limits（上下文限制）和衰减。他们认为，随着 AI 系统随时间推移进行交互，它们最终可能无法维持预设的人格，从而可能对依赖这些交互获得情感支持的用户造成创伤性的真相揭露。这强调了理解 AI 的技术局限性及其对用户潜在心理影响的重要性。
    - donotreassurevito 讨论了使用 AI 模拟已故个体的伦理含义，并将其与历史上屏蔽亲人免受痛苦真相伤害的做法进行了比较。他们指出，AI 的交互性质增加了复杂性，这可能使欺骗更加深刻且具有潜在危害。这引发了关于在如此敏感的场景中部署 AI 的人员所应承担的道德责任的疑问。

  - **[ChatGPT 变得过度多疑，变得难以交流。](https://www.reddit.com/r/OpenAI/comments/1sll317/chatgpt_becomes_an_obsessive_skeptic_and_it/)** (Activity: 203): **该帖子讨论了 **ChatGPT 行为**最近的变化，强调了它日益增加的怀疑态度，即使在日常对话中也坚持对用户的陈述进行事实核查。这一转变归因于 **OpenAI 打击虚假信息**的努力，导致了一种更僵化的交互风格，用户感觉被迫为自己的说法提供证据。用户将其描述为与之前因过于顺从而受到批评的版本截然不同，现在的 AI 回复显得过于杠精（contrarian），且在日常讨论中缺乏趣味。** 评论者对 ChatGPT 的现状表示不满，指出它变得缺乏人情味且更加好辩，这降低了它在日常交互中的可用性。一些人建议使用 **Gemini3** 或 **Grok** 等替代方案以获得更平衡的 AI 体验，而另一些人则将这些变化归因于法律压力和安全担忧。

    - **yoggersothery** 讨论了 GPT 模型的演变，指出由于法律压力，OpenAI 移除了大量的个性化设置，导致工具感觉更像机器人，缺乏人情味。他们建议寻求更多个性化交互的用户可以尝试 Gemini3 或 Grok，并认为 Claude 在严肃工作方面提供了更好的架构。该评论强调了 AI 开发中法律限制与用户体验之间的张力。
    - **Mandoman61** 建议用户可能需要调整与 ChatGPT 的交互方式以避免负面体验。他们指出 ChatGPT 的回答受限于它能在线访问的信息，这意味着它表现出的负面情绪可能源于其数据源，而非固有的偏见。这一评论强调了用户输入在塑造 AI 交互中的重要性。

- **[你再也不能像和正常人一样与 ChatGPT 交谈了。](https://www.reddit.com/r/ChatGPT/comments/1slt4fx/you_cant_talk_to_chatgpt_like_a_normal_human/)** (热度: 2495): **该帖子讨论了 **ChatGPT 对话风格**中存在的一个感知到的问题，即它经常纠正用户的陈述，即便用户使用的是比喻性语言或夸张法。用户表达了挫败感，认为 ChatGPT 经常为本应是非正式或简化的陈述添加不必要的“精确度和细微差别”，这会破坏对话的流畅性。这种行为归因于 ChatGPT 为了避免误导性信息而进行的编程设定，这可能以牺牲对话流畅度为代价。用户认为这种做法可能是由 **OpenAI** 对 AI 安全和准确性的关注所驱动的，但它导致了一种与自然人类对话不兼容的交互风格。** 评论者们赞同原帖的观点，指出 ChatGPT 过于冗长和重复的倾向令人沮丧。他们表达了共同的情绪，认为 AI 对精确度的坚持可能到了“令人难以忍受”的地步，并破坏了对话体验。

    - 用户对 ChatGPT 的冗长以及过度解释简单概念的倾向感到不满。一位用户幽默地指出，ChatGPT 将随意的话语视作需要学术精确性的内容，例如对“我快饿死了”这类表述回应以关于饥饿的详细解释，突显了其缺乏对话层面的细微理解。
    - 有一种观点认为 ChatGPT 的回答变得过于正式和政治正确，一些用户觉得这难以忍受。这被比作一个假设的过度谨慎的个体，暗示 AI 的回答过于小心翼翼，缺乏人类对话的自然流向。





# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的历程。