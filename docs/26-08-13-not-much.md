---
companies:
- google-deepmind
- google
- deepseek
- arcee
date: '2026-08-13T05:44:39.731046Z'
description: '**Google** 在发布 **Gemini 3.6 Flash** 仅三周后，便迅速推出了 **Gemini 3.7 Flash**，重点面向编程、网页开发、知识工作和智能体工作流。新版本首发优惠降价
  50%，同时提升了多项基准测试成绩，包括 **DeepSWE 65.3%** 和 **Code Arena Elo 1588**。此次更新很快覆盖多个平台，包括
  Gemini API 和 Android Studio；独立基准测试也证实了其性能提升。


  与此同时，**DeepSeek** 以 MIT 许可证开源了开发者预览版 **DeepSeek Harness**，重点探索架构创新，例如具备 KV-cache
  感知能力的仅追加历史记录语义，并将 Harness 定位为支持递归改进的操作系统／运行时底座。


  **Arcee** 也以 Apache 2.0 许可证开源了 **NAC**，专为长时间运行的异步任务设计，可支持大规模代码流水线，并允许用户通过手机进行编排，或借助
  Codex／Claude 委托执行任务。'
id: MjAyNS0x
models:
- gemini-3.7-flash
people:
- _philschmid
- koraykv
- officiallogank
- tianyi
- eliebakouch
- bookwormengr
- 0xlogicrw
- teortaxestex
- latkins
- stochasticchasm
- code_star
- fujikanaeda
title: 今天没发生什么特别的事。
topics:
- agentic-workflows
- coding
- knowledge-work
- benchmarking
- runtime-systems
- open-source
- long-running-processes
- asynchronous-computation
- software-architecture
- developer-tools
- price-performance
---

What would you like me to do with this excerpt—summarize it, fact-check the claims, rewrite it, extract key takeaways, or continue formatting it?

- **DeepSeek Harness 是当天讨论度最高的基础设施发布**：DeepSeek 以 **MIT** 许可证开源了 **DeepSeek Harness**，目前定位为**开发者预览版**（[\@tianyi](https://x.com/tianyi/status/2087888089759015218)）。技术社区的重点并不在基准测试分数，而更多集中于其架构设计：多种 Harness“模式”、可组合插件、可见的执行轨迹、**对 KV-cache 友好的只追加历史记录语义**，以及项目本身大量借助 agents/Codex 构建的事实（[\@eliebakouch](https://x.com/eliebakouch/status/2087904176357437820)、[后续讨论](https://x.com/eliebakouch/status/2087908415775408346)、[\@bookwormengr](https://x.com/bookwormengr/status/2087963340777951720)）。一种反复出现的解读是：DeepSeek 将 Harness 视为支持递归式改进的 **OS/runtime 基础层**，而不只是 Claude Code 的复刻版（[\@0xLogicrw](https://x.com/0xLogicrw/status/2087927539729829890)、[\@teortaxesTex](https://x.com/teortaxesTex/status/2087933141331628062)）。
- **Arcee 的 NAC 拓展了设计空间**：Arcee 以 **Apache 2.0** 许可证开源了 **NAC**，将其描述为一个用于**长时间运行、异步、无需人工持续介入的工作**的内部 Harness。在过去三个月里，NAC 已经支持了 Arcee **预训练、后训练和数据流水线**中相当大一部分代码的提交（[\@latkins](https://x.com/latkins/status/2087952185376346507)、[代码仓库](https://x.com/latkins/status/2087952198919753847)）。团队成员表示，他们用 NAC 完成的任务范围很广，从**看护实验**到**跨仓库工程任务**，再到各种**自动化研究型任务**；这些任务通常通过手机进行编排，或经由 MCP 从 Codex/Claude 委派执行（[\@stochasticchasm](https://x.com/stochasticchasm/status/2087953024736215054)、[\@code_star](https://x.com/code_star/status/2087956435628085356)、[\@fujikanaeda](https://x.com/fujikanaeda/status/2087962984647639132)）。
- **托管式和桌面端 Harness 正不断靠近生产工作流**：Cursor 宣布推出新的 **builds**，让**云端 agents 的启动速度提升 3 倍**，能够在失败时回退到上一个正常的 build，并增强长时间自主运行任务的稳定性和可调试性（[Cursor](https://x.com/cursor_ai/status/2087941307624980753)）。LangChain 最近关于“托管式 deep agents”的表述也采取了类似方向：生产环境中的 agents 不再是临时拼凑的聊天机器人，而是由文件定义的 Harness，具备调度、记忆、Slack 集成以及受治理的运行时语义（[\@hwchase17](https://x.com/hwchase17/status/2087950696457162837)、[\@bromann](https://x.com/bromann/status/2087942830895534280)、[\@caspar_br](https://x.com/caspar_br/status/2087963613294367043)）。
- **Nous 持续将 Hermes 扩展为可编程的 agent shell**：Nous 大幅扩展了 **Hermes Agent** 的插件能力，随后又加入了**实时操控子 agents 及查看其执行记录**的功能，并推出 **Bot Mode**：在该模式下，profile 会变成拥有独立聊天、例程、记忆、SOUL.md 以及 bot-to-bot 消息能力的具名 bots（[\@Teknium](https://x.com/Teknium/status/2087947369229009119)、[实时记录控制](https://x.com/Teknium/status/2087986084592709814)、[Bot Mode](https://x.com/Teknium/status/2088003994904113614)）。

**推理速度、投机解码与内核级优化**

- **OpenAI 与 Cerebras 推出 GPT-5.6 Sol “Ultrafast”**：OpenAI 预览了由 **Cerebras** 提供支持的 **GPT-5.6 Sol** **Ultrafast mode**，速度最高可达 **750 tokens/sec**，比标准模式快 **14 倍**，初期仅面向部分 API 客户开放。官方点名的应用场景包括低延迟的**语音、客服、商业、编程、金融和安全**工作流（[OpenAI](https://x.com/OpenAI/status/2087947721936359705)、[详细信息](https://x.com/OpenAI/status/2087947724725665908)、[Cerebras](https://x.com/cerebras/status/2087948820906950719)）。这也引发了更广泛的讨论：在 agentic 系统中，**工具延迟**可能很快会取代模型延迟，成为新的瓶颈（[\@random_walker](https://x.com/random_walker/status/2087969513048305668)）。
- **开源推理工作也在同步推进**：Red Hat AI 发布了面向 **Kimi-K3** 的投机解码器 **DSpark**，声称可将解码速度提升约 **4 倍**：在数学推理任务中，单用户速度从约 **110 tok/s** 提升至约 **435 tok/s**；在负载场景下吞吐量提升约 **3.5 倍**，并通过在 draft layers 中使用**滑动窗口注意力**，使 **20K 上下文**下的接受率保持稳定（[\@RedHat_AI](https://x.com/RedHat_AI/status/2087907190929531028)）。
- **内核优化正变得更加专业化**：Prime Intellect 发布了 **Prime Flash MoE**，这是一组针对 **Blackwell** 优化的 **CUDA kernels**，用于 **MoE 推理**，融合了路由感知 GEMM、SwiGLU、量化和归约操作，同时在 **B200** 上对 **BF16** 和 **MXFP8** 两条路径进行了基准测试（[Prime Intellect](https://x.com/PrimeIntellect/status/2087969614156247504)）。其意义很直接：随着各家实验室不断推出成本更低的 MoE endpoints，基础设施团队正围绕其底层 serving stack 展开激烈竞争。

**基准测试、评测平台，以及它们究竟在测量什么**

- **Custom benchmark infrastructure is becoming a product category**: Artificial Analysis launched **Optima**, a platform for building and running **custom benchmarks** on internal workloads, including uploaded datasets, agent traces from tools like **Arize/Braintrust/Langfuse**, or benchmark generation from natural-language descriptions. It tracks **quality, cost per task, time per task**, and can use pairwise judging similar to AA’s public benchmarks ([Artificial Analysis](https://x.com/ArtificialAnlys/status/2087930781050322977)). The pitch is that enterprises know they need custom evals, but very few can build them well ([\@grmcameron](https://x.com/grmcameron/status/2087981252683223522)).
- **Vals raised a $40M Series A and expanded benchmark coverage**: Vals announced a **$40M Series A at a $400M valuation**, alongside **Vals Smith** for custom coding benchmarks from any GitHub repo, a new **RSI Index** for AI R&D capability, and **ReverseEngBench** for cyber evaluation ([Vals AI](https://x.com/ValsAI/status/2087917239966290168), [RSI commentary](https://x.com/scaling01/status/2087922234857750653)). The core argument is familiar but increasingly important: model labs shouldn’t be the only ones grading their own systems.
- **Several papers pushed on agent eval failure modes**: notable summaries included Microsoft-related work arguing that **skill libraries can actively hurt agents**, attributing **307 failures** to loaded skills, including **125 functional failures** and **182 efficiency regressions** ([\@omarsar0](https://x.com/omarsar0/status/2087926158432309306)); a paper showing context compactors retain only **17%** of persistent session constraints unless augmented with a dedicated extractor ([DAIR.AI](https://x.com/dair_ai/status/2087930434323959894)); and another arguing leaderboard variance is dominated by **agent-task interaction**, not stable “agent quality,” with the **agent main effect under 3%** in multiple benchmarks ([DAIR.AI](https://x.com/dair_ai/status/2088007756582445228)).

**Model and Multimodal Releases Beyond Gemini**

- **MiniMax had a strong open-model day**: **MiniMax-Music3** launched as an **open-weights** music model; posts describe it as an **8B LLM + 2.7B DiT** that turns prompt + lyrics into full songs and runs on consumer hardware via **diffusers/ComfyUI/Hugging Face Spaces** ([MiniMax AI](https://x.com/MiniMax_AI/status/2087934657354678421), [\@multimodalart](https://x.com/multimodalart/status/2087933660490056163)). On the video side, **MiniMax-H3** reached **#1 in Video Edit Arena overall and among open models**, with **1390 pts** and a reported **+32 pt** lead over the next-best systems ([Arena](https://x.com/arena/status/2087930695469646276), [MiniMax](https://x.com/MiniMax_AI/status/2087938612088410302)).
- **Meta’s local-agent push kept spreading**: Meta’s **Muse Glimmer 30B** continued to draw attention as an **Apache 2.0** open-weights agent model that can run locally; Unsloth added free fine-tuning notebooks and **GRPO RL** support, claiming **1.5x faster** training with **50% less VRAM** and local training on **24GB VRAM** ([Unsloth](https://x.com/UnslothAI/status/2087930141217607798), [Ollama](https://x.com/ollama/status/2087965142097309871)).
- **Sakana Chat expanded practical code-execution UX**: Sakana updated **Sakana Chat**—powered by **Fugu and Namazu**—to support **code execution** with no login and free access, enabling Japanese-language interactive app/game/tool generation and spreadsheet/business-analysis workflows ([Sakana AI Labs](https://x.com/SakanaAILabs/status/2087880850318696481), [use case](https://x.com/SakanaAILabs/status/2087956599214391505)).

**Top Tweets (by engagement)**

- **OpenAI’s fastest frontier serving announcement**: **GPT-5.6 Sol Ultrafast**, up to **750 tok/s** and **14x speedup**, powered by Cerebras ([OpenAI](https://x.com/OpenAI/status/2087947721936359705)).
- **Google’s major workhorse refresh**: **Gemini 3.7 Flash** shipped with strong coding/agent gains at **half the original 3.6 Flash price** ([\@OfficialLoganK](https://x.com/OfficialLoganK/status/2087948481721962669), [Google](https://x.com/Google/status/2087948901265354817)).
- **OpenAI desktop memory/context expansion**: **Computer History** lets ChatGPT/Codex use opt-in app and website activity as context, with timeline view and user controls ([OpenAI](https://x.com/OpenAI/status/2087996496088297746), [OpenAIDevs](https://x.com/OpenAIDevs/status/2088000960891408677)).
- **DeepSeek’s agent runtime enters the open**: **DeepSeek Harness** open-sourced under **MIT**, catalyzing broad discussion about harnesses as the substrate for long-running and self-improving agents ([\@tianyi](https://x.com/tianyi/status/2087888089759015218)).
- **Hermes Agent keeps leaning into multi-agent UX**: Nous shipped **Bot Mode**, turning agent profiles into persistent named bots with routines and inter-bot messaging ([\@Teknium](https://x.com/Teknium/status/2088003994904113614)).




---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.8 Open Release and Local Inference

  - **[Qwen3.8-2.4T-A95B Released](https://www.reddit.com/r/LocalLLaMA/comments/1vmgozv/qwen3824ta95b_released/)**（热度：2345）：****Qwen3.8-2.4T-A95B** 已公布并发布。这是一款规模极大的稀疏/MoE 风格模型：从名称来看，它大约拥有 `2.4T` 个总参数，每个 token 约激活 `95B` 个参数。以 `bf16` 格式存储全部权重需要约 `4.8–5 TB` 的内存或磁盘空间。尽管每个 token 实际激活的参数量小得多，但对于一般家庭实验室来说，完整模型仍不适合本地推理。**热门评论主要关注部署难度：有人开玩笑说它“终于”成了可以本地运行的模型，也有人指出，`5 TB bf16` 的需求甚至超过了极端 homelab 的承受范围。评论中反复出现的观点是：真正有可能在本地运行的或许只有 `95B` 的激活部分，而不是完整模型。

    - 技术讨论主要围绕模型的**部署规模**展开：一位评论者指出，`bf16` 检查点大约需要 `5 TB` 空间，即使是高端 homelab 也很难进行本地推理。另一位评论者则强调了 MoE 风格模型中“总参数量”和“激活参数量”的区别，并开玩笑说自己只能运行**激活的 `A95B` 部分**。这意味着单个 token 的计算量可能更接近激活的 `95B` 个参数，但存储需求仍会按照完整的 `2.4T` 参数模型计算。

  - **[Qwen 3.8 release on hugging face](https://www.reddit.com/r/LocalLLM/comments/1vmgpz3/qwen_38_release_on_hugging_face/)**（热度：490）：****Qwen 3.8** 据称已发布到 [Hugging Face](https://huggingface.co/Qwen)。评论者尤其关注 **`27B`** 版本，同时注意到还有一个规模大得多的 **`95B active`** 配置。大家最关心的技术问题是部署难度：即使采用 **`Q1`** 这样的激进量化方式，`95B active` 模型在消费级硬件上仍可能不切实际，或者运行速度极慢；而拥有 **RTX 3090** 等 GPU 的用户则在期待更小的 `27B` 版本。**评论者普遍认可 Qwen “说到做到”，但也明显怀疑 `95B active` 模型是否能在本地实际使用，除非接受极端受限且非常缓慢的运行条件。

    - 用户强调，这次发布似乎包含一个规模非常大的 **`95B` 激活参数**模型。一位评论者认为，即便采用 **REAP-style pruning/quantization** 等激进方法，再配合极限的 **`Q1` 量化**，由于激活参数量过大，推理速度仍会慢到难以实用。
    - 多条评论都在讨论 Hugging Face 上预计发布的 **`27B` Qwen 3.8** 版本。用户尤其希望能在 **RTX 3090** 等消费级硬件上运行它，这也反映出大家关心这个较小的检查点能否装入 `24GB` 显存级别的 GPU，并达到可以接受的性能。

  - **[How do you plan to run Qwen3.8-2.4T-A95B locally?](https://www.reddit.com/r/LocalLLaMA/comments/1vmvpjc/how_do_you_plan_to_run_qwen3824ta95b_locally/)**（热度：608）：**这篇帖子询问爱好者打算是否以及如何在本地运行 **Qwen3.8-2.4T-A95B**，并将其与此前一些超大规模本地推理目标放在一起讨论，例如 `Llama-70B`、`Mistral Large`、`DeepSeek V2/V3` 和 `Kimi K3`。帖子没有提供具体的部署方案、硬件拓扑、量化策略或推理框架；热门评论中唯一较偏技术性的估算是，本地运行的吞吐量可能低到约 `0.003 tokens/s`。**热门评论大多持悲观或调侃态度，认为要实现实际可用的本地推理，可能需要约“`$100k`”级别的巨额投入，或者数量近乎不可能的内存，而不是普通消费者能够搭建的硬件环境。

    - 评论者普遍怀疑 **Qwen3.8-2.4T-A95B** 是否适合本地运行，认为多万亿参数 MoE 规模模型的内存和计算需求远超消费级硬件；有人估计其运行速度可能只有约 `0.003 tokens/s`，说明如果没有数据中心级 GPU，推理瓶颈将会非常严重。
    - 还有一个与技术实际应用相关的担忧是采用率和验证情况：一位评论者指出，Hugging Face 上的上传内容下载量**不到 `1000`**，也没有进入热门榜单。由于尝试加载或进行基准测试的用户太少，社区可能还缺乏足够的实践验证，无法判断这些发布文件是否真正可用，以及性能表现究竟如何。


### 2. DeepSeek V4 Pro and Agent Harness Launches

  - **[DeepSeek：今天发布 DeepSeek-V4-Pro！](https://www.reddit.com/r/LocalLLaMA/comments/1vn8m1x/deepseek_were_launching_deepseekv4pro_today/)**（热度：643）：****DeepSeek** 通过 [X](https://x.com/deepseek_ai/status/2087864585504305397) 宣布推出 **DeepSeek-V4-Pro**。评论者注意到，官方不仅公布了新的 API 价格表，还在 Hugging Face 上公开了模型权重：[`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)。目前最值得关注的是，该模型似乎既支持托管 API 调用，也支持本地或自托管推理；但 API 的收费结构变化较大，可能会影响不同工作负载的成本效益。**评论者普遍对涨价持负面态度：有人认为，这次涨价削弱了 DeepSeek 过去的优势，因为这些模型本来就“**比较吃 token，而且速度有点慢**”，只是价格低还能接受；如今低价优势不再，用户可能更倾向于本地部署。

    - 据报道，DeepSeek-V4-Pro 的权重已发布到 Hugging Face，地址为 [deepseek-ai/DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)。讨论重点也因此从 API 定价，转向了在有足够需求的情况下，第三方推理服务商能否以合理成本托管该模型。
    - 多位评论者关注 API 的成本问题：有人指出，DeepSeek 过去虽然“比较吃 token，而且速度有点慢”，但因为便宜，整体仍然可以接受；而新的价格取消了这一优势，可能会促使用户转向本地推理或其他服务商。
    - 一位早期用户质疑 DeepSeek 关于其性能可与 **Kimi 3** 持平的说法，认为 DeepSeek-V4-Pro 在知识深度，以及在超长上下文中持续完成长期、几乎无需人工干预的项目任务方面，似乎都不如 Kimi。

  - **[deepseek-ai/DeepSeek-V4-Pro-0813 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1vn9it4/deepseekaideepseekv4pro0813_hugging_face/)**（热度：594）：****DeepSeek** 曾短暂地将 [`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) 发布到 Hugging Face。评论者提到，与例如拥有 `2.8T` 参数的 Kimi 相比，这个 `1.7T` 参数模型公布的基准测试成绩异常强劲；其中有人特别指出，**DeepSWE** 从 V4-Pro Preview 的 `12.8` 大幅提升到 `62.7`，据称超过了 GLM-5.2 和 Opus-4.8。该仓库随后暂时变为 `404` 或私有状态，原因可能是打包或配置问题：据称，`config.json` 将隐藏层数量声明为 `43`，类似某个“flash”版本，但下载的权重分片却包含 `61` 层。这表明该版本可能在重新出现前被撤回修正。**评论者对其推理速度和基准测试提升印象深刻，但也有人提醒应保持谨慎，因为最初的 Hugging Face 文件在内部存在不一致，可能还需要修复。

    - 一位评论者指出，与 **Kimi 的 `2.8T`** 模型相比，**DeepSeek-V4-Pro-0813** 作为一个 `1.7T` 参数模型，表现似乎格外出色；他引用了该模型相较 **V4-Pro Preview** 的显著基准测试提升：**DeepSWE 从 `12.8` 提升至 `62.7`**，据称超过了 **GLM-5.2** 和 **Opus-4.8**。
    - 多名用户注意到，Hugging Face 上的模型曾短暂返回 `404`。一种技术解释是，**DeepSeek 可能因 `config.json` 配置错误而撤回了模型**：据称，该配置像 Flash 版本一样列出了 **`43` 个隐藏层**，但下载的权重分片实际包含 **`61` 层**，说明打包或配置存在不匹配，需要进行修正。

  - **[DeepSeek Harness 已上线！](https://www.reddit.com/r/LocalLLaMA/comments/1vnb66j/deepseek_harness_is_up/)**（热度：373）：****DeepSeek AI** 宣布推出 **DeepSeek Harness（`dsh`）**，这是一个处于开发者预览阶段的开源 Agent Harness，采用“所有功能都是插件”的架构，并由 **Cordis** 驱动；项目参考了 *A Programming Paradigm for Spatiotemporal Composability* 中的设计。该项目明确处于不稳定状态——“**将会出现破坏兼容性的变更**”——并引导开发者加入 [DeepSeek Harness Discord](https://discord.com/invite/Ycq5dCaS4)。官方没有提供基准测试、除插件和 Cordis 模型之外的更多实现细节，也没有承诺 API 稳定性。**热门评论对该项目迅速走红持怀疑态度：一位用户称，在大约一小时内看到 GitHub Star 数从 `20k` 涨到 `30k` 后，怀疑其中存在机器人刷星。其他技术层面的反应则质疑 Agent Harness 似乎越来越倾向于使用 TypeScript，并询问 `dsh` 是否能取得比 Reasonix 更高的缓存命中率。

    - A commenter raises an implementation concern that most agent/harness projects appear to be written in **TypeScript**, contrasting this with **Codex** as a possible exception. The technical implication is skepticism about runtime/ecosystem choices for a coding harness, though no concrete benchmark or failure mode is provided.
    - One technical question asks whether **DeepSeek Harness** can achieve better **cache hit rates** than **reasonix**, noting that cache efficiency is increasingly important for agentic coding workloads where repeated context/tool calls can dominate cost and latency.
    - The linked release materials are the GitHub repo [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness) and product page [`deepseek.com/harness/en/`](https://deepseek.com/harness/en/), but commenters note that the announcement lacks detailed technical documentation or benchmark data.


### 3. LLM Transparency: Watermarking and Reasoning-Trace Leaks

  - **[Hidden Reasoning from Claude and GPT are Decoded, and it is interesting](https://www.reddit.com/r/LocalLLaMA/comments/1vmawd2/hidden_reasoning_from_claude_and_gpt_are_decoded/)** (Activity: 447): **A cited paper, [**“Stealing Reasoning Traces from Proprietary LLM APIs”**](https://arxiv.org/pdf/2608.09867), claims an API-side leakage method can recover hidden reasoning tokens from **Claude** and **GPT** models, with published examples in [`mitkox/stolen-thoughts`](https://github.com/mitkox/stolen-thoughts). The post highlights an AIME example where a decoded Claude trace appears to recognize a benchmark item from memory — *“This is a known AIME problem. Answer 60”* — raising concerns about benchmark contamination and inflated proprietary-model math scores; commenters also link a related discussion on [X/Twitter mirror](https://xcancel.com/_can1357/status/2087228354399265125?s=20).** The main debate is whether frontier-model reasoning traces reveal any hidden algorithmic “secret sauce”: the poster argues they mostly show ordinary artifacts like memorization, incoherent intermediate tokens, and overthinking, implying open-source models may be closer than benchmark gaps suggest. There is also speculative concern that such leakage may have enabled large-scale distillation of proprietary models and that closing the gap could slow future distillation efforts.

    - A linked repo, [mitkox/stolen-thoughts](https://github.com/mitkox/stolen-thoughts), is referenced as evidence around “decoded” hidden reasoning traces from closed models. The quoted trace is technically interesting because it appears to expose internal chain-of-thought-style behavior including problem recognition, partial memorization of an AIME problem, intermediate geometry computations such as `AC = 7√3` and `AD = 13√3`, and uncertainty/self-correction around the final answer.
    - One commenter argues there is likely no unique proprietary “secret sauce” visible in the reasoning tokens: the gap is framed as primarily **data, compute, and engineering**, not fundamentally different reasoning mechanisms. They speculate that open-weight models could reach future closed-model capability levels while fitting on a `128G` device, though this is presented as prediction rather than benchmark evidence.
    - Another technical point is that hidden reasoning is valuable less as user-facing output and more for **post-training and RL optimization**: retaining or supervising latent reasoning traces can improve training signals and reduce cost by avoiding longer visible generations. The claim is that hidden reasoning mainly helps optimize post-training objectives and inference economics rather than representing a qualitatively separate capability.

  - **[Anthropic, OpenAI, Google, Meta, Microsoft, and Mistral all signed the EU Code of Practice on Transparency of AI-Generated Content](https://www.reddit.com/r/LocalLLaMA/comments/1vlyzi6/anthropic_openai_google_meta_microsoft_and/)** (Activity: 949): **The [image](https://i.redd.it/f9jt8fh79uih1.jpeg) is a screenshot of an X post claiming **Anthropic, OpenAI, Google, Meta, Microsoft, and Mistral** signed the EU Code of Practice on transparency for AI-generated content, with **OpenAI** support text saying it wants to expand provenance signals to all modalities, including **text**. The Reddit post interprets this as future invisible watermarking of generated **code and prose**, including local/open-weight models from those vendors, though the image itself is not a technical spec or implementation proof.** Commenters largely focused on practical bypasses and workflow risk: one argued watermarking could be learned from `~10M` generated tokens and removed by a small adversarial `1.5B` model or browser extension, while others worried it could push users toward non-signatory/Chinese models or interfere with agentic code generation and compilers.



- 一位评论者认为，文本水印在技术上可能很容易被逆向分析：为每个模型生成大约 `10,000` 个段落、合计 `~10M tokens`，训练一个分类器来区分不同模型的输出，找出与水印相关的特征，然后再训练一个对抗性 `1.5B` 模型，在尽量少改写文本的情况下移除水印信号。他们称，这种移除器可以在笔记本电脑的 CPU 上本地运行，也可以作为浏览器扩展，实时修改流式输出的 LLM 内容。
- 几位评论者质疑，不可见水印是否能与实用的文本生成兼容，尤其是在代码场景中。他们担心，对 token 分布的扰动或隐藏标记可能会干扰 Agent 脚本、对编译器敏感的输出，或要求精确格式的生成；相比之下，图像、音频和视频中的水印更容易利用人眼或人耳难以察觉的信号通道。
- 有人提出了一项关于技术普及的担忧：如果强制实施 EU 水印，尤其是在水印影响输出质量或可检测性的情况下，用户可能会转向签署范围之外的开源权重模型或更便宜的 API 模型，特别是中国模型。一位评论者推测，如果其他市场的用户抵制水印，服务商可能会保留仅针对 EU 的水印策略。




## 非技术类 AI Subreddit 讨论总结

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. SL2T 与 H3 多模态模型机制

  - **[DeepMind just released SL2T, sign language-to-text model, deaf users can now sign into their phones instead of typing, developed with heavy input from the Deaf community](https://www.reddit.com/r/singularity/comments/1vmflo1/deepmind_just_released_sl2t_sign_languagetotext/)**（热度：3468）：****DeepMind** 据报道发布了 **SL2T**，这是一套将手语实时转换为英文文本的系统，能够同时识别**手部、身体和面部动作**，让聋人用户可以直接对着手机打手语，而不必输入文字；详情见所链接的 [DeepMind 博客文章](https://deepmind.google/blog/putting-sign-language-ai-into-users-hands/)。文章称，姿态追踪在**设备端**运行，以保护隐私，而翻译则在**服务器端**完成；系统还支持一边拿着手机、一边用**单手打手语**等实际场景，并声称在“*学术基准测试中处于业界领先水平*”，但 Reddit 总结中没有提供具体的基准数据。**置顶评论总体上态度积极，表达了对 DeepMind 的认可，也惊讶于这类无障碍技术此前没有更早出现；给出的评论中没有实质性的技术争论。**


  - **[PSA: I’m the creator of Heretic, and I advise you to *not* use “heretic” models as text encoders for H3 (or any other model)](https://www.reddit.com/r/StableDiffusion/comments/1vmdxzk/psa_im_the_creator_of_heretic_and_i_advise_you_to/)**（热度：2901）：**[**Heretic**](https://github.com/p-e-w/heretic) 的作者警告称，不要把“heretic”/abliterated LLM 替换为图像或视频模型的文本编码器——例如，不要用它替换 **Minimax H3** 的 **Qwen3-VL** 编码器——因为这样**不会降低输出审查，反而可能削弱提示词遵循能力或引入伪影**。Heretic 风格的方法会使用方向性消融、ARA 和 SOMA，扰动 residual stream 中的表示，使“有害”提示词在 **LLM 的拒答行为**中更像“无害”提示词；但它们**不会**为下游的 diffusion/transformer 生成器提供更丰富、更“原始”的语义嵌入，反而会让隐藏状态偏离生成器训练时所使用的分布。一个可能的例外是那些明确包含拒答 LLM 环节的模型或工作流，例如提示词增强器，或 Ideogram 风格的主动拒答系统；在这些场景中，解除该 LLM 组件的审查可能确实有帮助。**评论大多支持这份 PSA，认为作者的说法权威且值得广泛传播。**一位评论者指出了一个实际例外：如果图像工作流会先通过基于 LLM 的提示词增强器处理用户提示词，而该增强器本身会拒绝可疑内容，那么换成未经审查的模型，就可以在提示词到达生成器之前绕过这一步拒绝。

    - 一位评论者指出，Heretic/未经审查的模型在图像生成工作流中存在一个范围较窄但确实有效的用途：它们不应作为 H3 或类似模型的**文本编码器**，但可以作为上游的 **LLM 提示词增强器**，在用户提示词传给图像模型之前对其进行改写。他们表示，一些提示词增强工作流会拒绝“可疑内容”，而换用未经审查的 LLM 后即可避免被拒绝。这说明它的价值在于提示词预处理环节，而不是在 CLIP/T5 风格的图像模型条件控制中。


### 2. Grok 4.6 基准测试与 DeepSeek API 涨价

  - **[DeepSeek 大幅上调 API 价格（2026 年 8 月 16 日起生效）——缓存命中价格最高上涨 1,114%](https://www.reddit.com/r/DeepSeek/comments/1vn81do/deepseek_just_massively_increased_their_api/)**（热度：1597）：****DeepSeek** 将从 `2026-08-16 16:00 UTC` 起调整 API 定价，并新增高峰/非高峰时段；根据其 [API 定价文档](https://api-docs.deepseek.com/quick_start/pricing/)，高峰时段的价格是非高峰时段的 `2×`。涨幅最大的是缓存输入：**V4-Pro 缓存命中**的每个计价单位价格将从 `$0.003625` 上调至非高峰/高峰时段的 `$0.022/$0.044`，涨幅分别达到 `+507%/+1,114%`；输出价格也大幅上涨，**V4-Pro 输出**将从 `$0.87` 调整为 `$1.98/$3.96`，涨幅分别为 `+128%/+355%`。这会明显削弱 DeepSeek 在依赖提示词缓存的长上下文和重复性工作负载中的成本优势，同时也让围绕 UTC 高峰时段进行任务调度和成本优化变得更加复杂。**热门评论大多缺乏技术细节，且整体较为悲观；一位评论者表示自己“已经转向其他方案”，因为 *“DS4 表现不错，但前提是价格便宜”*，这意味着该模型的性价比很大程度上取决于较低的 API 价格。

    - 一些用户表示已经开始迁移离开 DeepSeek API，认为 **DS4 的性价比高度依赖低价**；在此次涨价后，尽管模型质量尚可，但他们认为其竞争力已经下降。
    - 一位评论者提到，时区可能会带来定价优势：在**巴西**，API 的“非高峰”时段似乎大致对应当地时间的 `7:00–22:00`，因此正常工作日的大部分时间都能享受折扣价格。

  - **[根据 artificial analysis arena，Grok 4.6 与 Sol 5.6 实力相当](https://www.reddit.com/r/singularity/comments/1vmhtfu/grok_46_is_an_equivalent_to_sol_56_according_to/)**（热度：1475）：**这张[图片](https://i.redd.it/gi37g752tyih1.jpeg)是一张标题为 **“Artificial Analysis Intelligence Index”** 的基准测试柱状图，其中 **Claude Opus 5** 以 `63` 分位居第一，**Claude Fable 5** 得分为 `62`，而 **GPT-5.6 Sol** 与 **Grok 4.6** 均为 `61`，支持了帖子标题所称的 *“Grok 4.6 与 Sol 5.6 实力相当”* 这一结论。评论还补充了价格和模型规模方面的信息：据称 Grok 4.6 的价格低得多（输入 `$2/M`、输出 `$6/M`），而 GPT-5.6 Sol 的价格为输入 `$5/M`、输出 `$30/M`；同时，Grok 4.6 据称是规模为 `1.5T` 的较小模型，而竞争对手的规模为 `5T+`。**评论者认为，这一结果体现了 **xAI/Grok** 在前沿模型领域取得的进展，令人颇感意外；其中一人表示，Google “完全掉出前沿”也出乎意料。还有人推测，**Grok 4.7** 的规模可能扩展到 `2T–2.5T` 参数，并认为未来的重要模型可能都会从“Kimi 级别”起步。

    - 评论者重点比较了价格和模型规模：据称 **Grok 4.6** 的输入价格为 `$2/M`、输出价格为 `$6/M`，远低于 **GPG 5.6 Sol** 的输入 `$5/M` 和输出 `$30/M`。一位用户指出，SpaceXAI 只将 Grok 4.6 与 **Sol** 和 **Fable** 进行比较，没有纳入 **Opus** 或 **Sonnet**；与此同时，SpaceXAI 声称 Grok 4.6 是一个规模约为 `1.5T` 参数的模型，而竞争对手据称达到 `5T+` 参数。
    - 一条技术讨论围绕 **Grok 4.6** 究竟是 `2T` 还是 `1.5T` 参数展开。一位评论者随后修正说法，认为它更可能是 **Grok 4.5 加上额外 RL**，类似于外界所称的 **GPT 5.5** 与 **GPT 5.6** 之间的关系。他们据此推测，SpaceXAI 可能只落后 OpenAI 约 `3 个月`：原因是推断 OpenAI 5.6 大约在 4 月或 5 月可用，同时有报道称外部测试人员在正式发布前约 `3 个月` 就已经获得了 5.6 的访问权限。
    - 多条评论认为，Grok 4.6 在基准测试中的表现值得关注，因为据报道它在多个基准上接近 SOTA，而 **Google** 似乎已经“掉出前沿”。另一位评论者表示，如果如今 Grok 4.6 和 **Kimi 级别**的模型已经成为大型前沿模型发布的基准，那么即将推出的 **Grok 4.7** 可能会将规模提升到 `2T–2.5T` 参数。

  - **[Grok 4.6 基准测试](https://www.reddit.com/r/singularity/comments/1vmhvc3/grok_46_benchmarks/)**（热度：1017）：**这张图片是一张 **“Grok 4.6 High”** 的基准测试表格（[图片](https://i.redd.it/2cyo43ddtyih1.png)），将其与 **Grok 4.5 High**、**GPT-5.6 Sol Max** 和 **Fable 5 Max** 进行了比较。结果显示，Grok 4.6 在前沿模型竞争中表现强劲，在 **GDPVal-AA v2**、**AA-Briefcase** 和 **Harvey LAB** 上领先；但在其他多个基准类别中，竞争对手仍然占优。表格还注明，结果基于*第三方模型评分*，采用各模型自行报告或公开发布的最佳成绩。一位评论者还特别提到其据称达到 **`1.5T`** 的规模，说明模型大小和计算资源也是人们关注这组基准结果的因素。**评论整体较为轻松，或停留在推测层面：一位用户将前沿模型的发展形容为轮流领跑的 hype cycle——**Grok → Claude → Gemini → ChatGPT**；另一位则认为其 `1.5T` 的规模“令人印象深刻”。

    - 一位评论者指出，据称 **Grok 4.6** 是一个拥有 `1.5T` 参数的模型。考虑到这一规模以及其显著而迅速的进步，这一基准测试成绩尤其值得注意。另一位评论者则强调了性价比，称 Grok 以这样的价格来说 *“非常擅长编程”*，而且 *“速度很快”*；从基准测试排名来看，它似乎接近 **Kimi K3**——*“每项任务的价格完全相同”*，但得分高 `1` 分。
    - 有评论者介绍了一种结合 **Claude Opus** 和 **Grok** 的编程工作流：由 Opus 负责高层规划和初始实现，再让 Grok 执行范围明确的小幅修改。评论者认为，这种手动组合相当于更强大的 Cursor/Composer 风格 Agent 式编辑，并建议进一步用子 Agent 实现自动化。


### 3. Claude Opus 5 Agent 用户体验与自主编程

  - **[I asked Opus 5 to build GTA6 on its own in 24 hours](https://www.reddit.com/r/ClaudeAI/comments/1vmjzh7/i_asked_opus_5_to_build_gta6_on_its_own_in_24/)**（活跃度：1558）：**作者称，他们要求 **Opus 5** 在 `24h` 内自主生成一款类似 GTA 的开放世界游戏，并且不再提供进一步指导，完全由模型自行决定*城市布局、区域、道路、建筑、NPC、载具和天气*。之后，作者在 [`ukanwat/aaabench`](https://github.com/ukanwat/aaabench) 中公开了这套编排框架，其中包括“skills、agents、tooling、resources 和 models”；但由于 Reddit 返回 `403 Forbidden`，通过所提供 URL 无法访问 Reddit 托管的游戏演示/预告片视频。**评论大多停留在推测层面：一位用户询问 3D 模型和其他素材的来源，另一位用户则认为，尽管成果还比较“粗糙”，这种自主生成游戏的能力仍可能在未来十年显著扩大独立游戏的产量，尽管其中很可能会出现大量 AI 生成的“垃圾内容”。

    - 一个与技术密切相关的问题集中在**素材来源**上：一位评论者问道，*“它是从哪里获取模型的”*，并指出，评估一个 AI 构建的 GTA 类演示，很大程度上取决于 Opus 是自行生成了网格/纹理、使用了随附素材、抓取或下载了第三方模型，还是依赖现成的游戏引擎素材包。这一区分对于判断其自主性、版权风险，以及最终成果有多少真正反映模型能力、多少只是素材拼装，都非常重要。

  - **[Opus 5 is actually almost rage-inducing to use.](https://www.reddit.com/r/ClaudeAI/comments/1vn8ml6/opus_5_is_actually_almost_rageinducing_to_use/)**（活跃度：1378）：**帖子称，在编程和工作流场景中，即使遵循 Anthropic 的指导并修改全局 `claude.md`，使用 **Anthropic Claude Opus 5** 仍然体验很差：据称，它的输出依旧过于冗长、充斥流行术语，还经常把小任务膨胀成包含多个步骤的“项目”。作者最主要的技术层面抱怨是代码修改不完整，以及由此产生的 **code rot**：Opus 5 可能修改文件的一处内容，却让同一文件的另一处失效，随后只是说*“如果你希望修改那部分，请告诉我”*，而不是主动修复由此造成的不一致；作者认为 **Fable** 的这类问题较少，但即使是 `$200` 套餐，每周额度也很有限。**高赞评论普遍表示认同，认为 Opus 5 不可用或“很烦人”；一位用户说，让模型把一个两页的文本文件改得简洁一些，结果却变成了持续一小时的过程，并且还剩大约 `10` 项待处理任务；另一位用户则表示，必须反复催促模型，它才会完成一个简单的修改请求。

    - 多位用户表示，与 **Opus 4.8** 相比，**Opus 5** 在任务执行和回归表现上存在问题，尤其是在总结、改写等简单的转换任务中。一位用户说，请求模型把一个两页的文本文件改得简洁，结果一小时后得到的却是“成墙的文字”，以及一套拟议中的工作流/系统和许多后续待办事项。反复出现的技术性抱怨是：模型过度拆解任务、过度规划，而不是直接完成用户要求的输出。
    - 一些评论描述称，**Opus 5** 在处理边缘情况时会生成复杂绕口的文字和混乱的推理过程；其中一位用户明确表示，因为“用 Opus 5 什么都做不成”，所以退回使用 **Opus 4.8**。另一位用户提到明显的“code rot”，并称模型需要通过反复提示，才能完成一个单独的编程任务。这表明，与之前的版本相比，它在指令遵循能力或持续完成任务方面可能有所退化。
    - 一位评论者认为，与 **ChatGPT 5.6** 相比，**Opus 5** 的表现较差，并称自己已经切换回去，因为 ChatGPT “好用得多”。虽然评论中没有提供基准测试数据，但该讨论的核心主题是实际可用性被认为出现了倒退：问题不在于原始智能水平，而在于输出冗长、行为固执，以及难以真正把任务完成。

  - **[You never know the good days until they’re gone (unless you’re still using 4.6)](https://www.reddit.com/r/ClaudeAI/comments/1vn6b31/you_never_know_the_good_days_until_theyre_gone/)** (Activity: 1001): **The image is a [bar chart](https://i.redd.it/rjci2eqv64jh1.png) comparing **words per answer** across model versions, showing **Opus 5** as much more verbose at `510` words/answer versus **Opus 4.6** at `234`, **Opus 4.8** at `259`, **Opus 4.7** at `276`, **Fable 5** at `316`, and **Opus 4.5** at `158`. The post uses this to argue that newer models—especially Opus 5—may be less pleasant for practical workflows because they “flood the chat” with unnecessary chatter despite being newer or stronger on paper.** Commenters debated whether higher benchmark intelligence translates to better usability: one preferred a slightly weaker model that executes concisely over one that narrates excessively, while another asked for the data source behind the chart. A third commenter said they now use **Kimi K3** as a main driver because it is more pleasant and exposes reasoning traces for oversight by other agents.

    - Several commenters argue that higher benchmark or “smarter on paper” performance does not necessarily translate into better developer ergonomics: **Opus 5** is criticized for verbose meta-reasoning, excessive caveats, and repeatedly generating new follow-up issues instead of directly completing the requested task. One user says they would prefer a “slightly weaker model” if it more reliably understands the task and executes without long narrative overhead.
    - A workflow comparison highlights **Kimi K3** as a preferred “main driver” model, with **Claude/Sol** used as planner/reviewer agents. The key technical point is that Kimi reportedly exposes reasoning traces rather than hiding them, allowing supervising agents to inspect intermediate reasoning in near real time and catch issues more effectively; **Fable** is described as better at planning, while **Sol** is more thorough but less pleasant due to degraded writing style.
    - A small but notable usage signal: one commenter says they still use **Claude 4.6** almost exclusively, implying that older models may remain preferable when they provide better task adherence, lower verbosity, or more predictable interaction patterns than newer releases.