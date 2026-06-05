---
companies:
- nvidia
- anthropic
- togethercompute
- baseten
- modal
- vllm_project
- fireworksai_hq
- ollama
- wandb
- cline
- primeintellect
- nousresearch
date: '2026-06-04T05:44:39.731046Z'
description: '**英伟达（NVIDIA）** 发布了 **Nemotron 3 Ultra**，这是一个完全开源的 **550B MoE**（混合专家）模型。它拥有
  **55B 激活参数**和 **100 万（1M）上下文窗口**，专为长程智能体（agent）任务进行了优化，实现了高达 **5 倍的速度提升**和 **30%
  的成本降低**。该模型采用了 Mamba 与 Attention 混合架构、LatentMoE 以及原生 MTP（多 Token 预测）技术，并在 **20T（20
  万亿）Token** 上使用 NVFP4 低精度格式完成了预训练。基准测试显示其性能强劲，智能指数（Intelligence Index）达到 **47.7**，输出速度超过
  **每秒 400 个 Token**。目前，各大主流推理平台均已支持该模型。此外，英伟达还推出了 **Nemotron 3.5 ASR**，这是一款拥有 **0.6B
  参数**的开源流式语音识别模型，支持 **40 种语言及地区组合**，延迟低于 100 毫秒，专为语音智能体设计。


  **Anthropic** 强调了人工智能领域递归自我改进（RSI）的早期迹象：**Claude** 模型目前编写了超过 **80% 已合并的代码**，使工程师的代码交付量提升了
  **8 倍**。Claude Opus 4 在训练脚本上实现了 **3 倍的速度提升**，而 Mythos 预览版（Mythos Preview）则实现了约 **52
  倍的加速**，且在 **64% 的情况下**提供的高质量研究建议优于人类。'
id: MjAyNS0x
models:
- nemotron-3-ultra
- nemotron-3.5-asr
- claude-opus-4
- mythos-preview
people:
- piotrz_zelasko
title: 今天没什么事发生。
topics:
- mixture-of-experts
- long-context
- model-quantization
- agentic-ai
- streaming-speech
- asr
- low-precision-training
- benchmarking
- recursive-self-improvement
- code-generation
- model-speedup
---

**平淡的一天。**

> 2026年6月3日至6月4日的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增 Discord 信息。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以 [选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件频率！

---

# AI Twitter 综述


**NVIDIA 发布 Nemotron 3 Ultra 与 3.5 ASR**

- **Nemotron 3 Ultra** 是当天最明确的技术发布：这是一个完全开源的 **550B MoE** 模型，拥有 **55B 激活参数**、**1M 上下文**，并明确专注于长时运行的 **Agent** 工作负载。NVIDIA 表示，在 **Agentic** 任务中，该模型的速度提升高达 **5 倍**，成本降低 **30%**，其权重、合成数据、奖励检查点、量化变体以及训练配方均在 **OpenMDW 1.1** 协议下发布 ([NVIDIA 发布公告](https://x.com/nvidia/status/2062522316672667770), [NVIDIAAI 开源资产](https://x.com/NVIDIAAI/status/2062521383582646537), [Pavlo Molchanov 的推文串](https://x.com/PavloMolchanov/status/2062538679470657727))。该架构结合了 **hybrid Mamba/attention**、**LatentMoE** 和原生 **MTP**，并在 **20T tokens** 上使用 **NVFP4** 完成了预训练——值得注意的是，它将低精度预训练推向了新的规模量级 ([技术笔记](https://x.com/ctnzr/status/2062515418884149451), [扩展性讨论](https://x.com/scaling01/status/2062540298933219832))。

- **基准测试和推理服务表现** 对于一个开源发布来说异常强劲。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2062527871529439438) 使用 NVIDIA 推荐的 **NVFP4** 推理权重测得其智力指数（Intelligence Index）为 **47.7**（**BF16** 为 **48.2**），使其成为他们测试过的最强 **美国开源权重** 模型，尽管仍落后于 **Kimi K2.6**。更有趣的是，他们报告通过 BlackBox 实现了 **400+ output tok/s** 的吞吐量，并分别展示了 Nemotron 3 Ultra 在受限轮数的 Terminal-Bench 风格评估中，处于 **任务延迟与性能的帕累托前沿 (Pareto frontier)** ([延迟分析](https://x.com/ArtificialAnlys/status/2062598349757567359), [BlackBox 吞吐量](https://x.com/blackboxai/status/2062546216949588001))。该模型在发布首日 (**day 0**) 即在整个技术栈中上线：[vLLM](https://x.com/vllm_project/status/2062574262163280172), [Modal](https://x.com/modal/status/2062528720104227149), [Together](https://x.com/togethercompute/status/2062520009893576974), [Fireworks](https://x.com/FireworksAI_HQ/status/2062568688201646321), [Ollama cloud](https://x.com/ollama/status/2062591290743853291), [Baseten](https://x.com/baseten/status/2062609272815685759), [CoreWeave/W&B](https://x.com/wandb/status/2062577626242580896), [Cline](https://x.com/cline/status/2062620668085297214), [Prime Intellect](https://x.com/PrimeIntellect/status/2062622550300275088), 以及 [Nous Portal](https://x.com/NousResearch/status/2062554136625766409)。

- **Nemotron 3.5 ASR** 是一个较为低调但实用的伴随发布：一个开源流式 **ASR** 模型，拥有单一的 **0.6B** 检查点，支持 **40 种语言/地域组合**，且延迟低于 **100ms**，基于缓存感知的 **FastConformer / RNN-T** 风格设计，针对语音 **Agent** 和流式语音工作负载进行了优化 ([Piotr Zelasko](https://x.com/PiotrZelasko/status/2062538923776290909), [Together](https://x.com/togethercompute/status/2062520605102993436), [fal 的可用性](https://x.com/fal/status/2062521027020611933))。

**Anthropic 的递归自我提升框架与内部 AI 编程指标**

- Anthropic 发布了当天讨论最多的政策/研究简报，认为当前系统显示出 **递归自我提升 (RSI)** 的早期迹象——虽然在研究方向上尚未实现完全自主，但有明确证据表明 AI 正在加速 AI 的开发 ([Anthropic 帖子](https://x.com/AnthropicAI/status/2062568862479208923))。核心运营数据非常具体：Anthropic 内部 **80% 以上的合并代码** 现在由 Claude 编写，典型工程师每季度的代码交付量是往年的 **8 倍**，并且在内部开放式工程任务中，Claude 的成功率在六个月内从大约 **26% 提升到了 76%** ([编程指标](https://x.com/AnthropicAI/status/2062568864240836995), [Alex Albert 的总结](https://x.com/alexalbert__/status/2062580571214389510))。

- 最引人注目的经验数据点是 Anthropic 反复进行的“加速小模型训练脚本”测试：**Claude Opus 4** 平均提速约 **3x**，而据报道 **Mythos Preview** 实现了约 **~52x** 的提速（[Anthropic 基准测试声明](https://x.com/AnthropicAI/status/2062568869240476050)，[日期修正](https://x.com/AnthropicAI/status/2062634151556292775)）。Anthropic 还表示，在研究人员走入误区的环节中，**64%** 的情况下 **Mythos** 给出的“下一步该做什么”的研究建议优于人类（[研究下一步决策结果](https://x.com/AnthropicAI/status/2062568870872003021)）。他们的核心观点是：自动化“问题选择”仍未解决，但自动化执行和迭代的大部分环节已经成为现实。

- 治理层面的意义与生产力的提升同样重要。Anthropic 明确写道，“如果世界有权选择**放慢或暂时暂停前沿 AI 的开发**，那将是件好事”，并将验证和协调机制框定为应对 RSI 类动态持续发展的紧迫需求（[Anthropic 治理声明](https://x.com/AnthropicAI/status/2062568873321513443)，[讨论](https://x.com/scaling01/status/2062572962117562507)，[评论](https://x.com/a_karvonen/status/2062572851916574730)）。与此同时，据 [@CRSegerie](https://x.com/CRSegerie/status/2062474945377218819) 称，Anthropic 最近**削弱了其 Responsible Scaling Policy 中关于生物/化学风险的部分阈值**，因此引发了批评。另外，包括 **Altman、Amodei、Hassabis 和 Baker** 在内的联盟在美国支持**强制性 DNA 合成筛选和记录保存**，理由是 AI 正在削弱生物知识的门槛（[信函摘要](https://x.com/kimmonismus/status/2062485389949145457)）。

**Cloudflare 收购 VoidZero 并强化全栈 Agent 工具链**

- 开发者平台领域最大的动作是 **Cloudflare 收购了 VoidZero**，该团队是 **Vite、Vitest、Rolldown、Oxc 和 Vite+** 的幕后力量。Cloudflare 和 VoidZero 强调 **Vite 保持开源、MIT 协议且厂商中立**，Cloudflare 还承诺向一个独立 Vite 生态系统开发基金投入 **100 万美元**（[Cloudflare](https://x.com/Cloudflare/status/2062521221132992533)，[Vite 声明](https://x.com/vite_js/status/2062525206158078047)，[尤雨溪 (Evan You)](https://x.com/evanyou/status/2062533668233756677)）。

- 开发者们的战略解读是，这让 Cloudflare 对日益趋向 Agent 友好的应用栈拥有了更紧密的控制力：将前端/构建工具、Runtime、存储、推理、部署原语和安全性集成于一体。[@wesbos](https://x.com/wesbos/status/2062520527151903090) 将其形容为 Cloudflare 正在组装“一个可以直接交给 LLM 来构建网站的整洁软件包”，这与 Cloudflare 自身在统一平台中推进 Agent、MCP、Sandboxes、AI 搜索、支付和可观测性的方向一致（[Cloudflare Agent 文档概览](https://x.com/thomasgauvin/status/2062512156076048447)）。

**Agent、Harness、存储和评估基础设施**

- 多条推文指出，在原始模型发布之外，一个成熟的“Agent 系统”层正在形成。一个反复出现的主题是，瓶颈正日益转向 **Harness/编排器 (Orchestrator)**，而不仅仅是 Prompting。一段流行的视频将 Claude Code 的工作流总结为“我不再向 Claude 发送 Prompt 了，我直接写循环 (Loops)”，而 [@omarsar0](https://x.com/omarsar0/status/2062553527730540611) 则描述了如何将**动态工作流**逆向工程到他自己的编排器中，用于分支研究、验证、分选、数据综合和评估生成。共同的理念是：高阶控制循环，而非单次 Prompt，正在成为真正的工作单元。

- 围绕这些循环的工具也在改进。[LangSmith Sandboxes](https://x.com/LangChain/status/2062512156688466083) 达到 GA 阶段，具备 Dockerfile 快照、交互式控制台、TCP 隧道和标准 Linux 工具。Hugging Face 推出了两个相关的想法：在 Hub 上发布自定义 **Kernels** 的分发路径（[公告](https://x.com/RisingSayak/status/2062471134260687264)），以及加强对将 **Agent Trace** 作为一等资产存储的支持，这得到了 [@ClementDelangue](https://x.com/ClementDelangue/status/2062542713463980303) 的回应。[@julien_c](https://x.com/julien_c/status/2062524414034423969) 发布了 **SynthTraces**，这是一个极简的 Harness，通过让开源模型扮演编程 Agent，并由本地模型模拟用户，生成了 **2,000 多个合成编程 Agent 会话轨迹**。

- 评估也转向了现实世界的 Agent 工作。**Arena** 推出了 **Agent Arena / Agent Mode**，通过包含网页搜索、文件系统、bash 和图像生成等工具的**数百万次实时会话**来衡量 Agent 性能。他们目前的排名中，**GPT-5.5** 位居第一，随后是 **Claude Opus 4.7**、**GLM-5.1**、**Gemini 3.1 Pro** 和 **Kimi-K2.6**。其方法论基于 **30 万个以上任务**、**200 万次以上工具调用**以及 **4000 万行代码**中的任务成功率、可控性、恢复能力、用户好评/投诉以及工具幻觉 ([发布地址](https://x.com/arena/status/2062566749418233981)，[方法论](https://x.com/arena/status/2062566769659912281))。在企业端，**Cognition** 为 Devin 推出了 **AI Productivity Guarantee**（AI 生产力保障）——如果产品未能产生积极的工程价值，将涵盖高达 **1000 万美元**的使用费用——该保障由一套内部衡量系统支持，涵盖了超过 **258 个企业会话**，任务时长长达 **64 小时以上** ([保障详情](https://x.com/cognition/status/2062597242167628019), [技术报告](https://x.com/cognition/status/2062597246001324518))。

**记忆、多模态以及模型/基准测试更新**

- **OpenAI 向美国的 Plus 和 Pro 用户推出了功能更强大的 ChatGPT 记忆系统**，具备**记忆摘要**、更多控制选项以及 **2 倍的记忆容量**。公司将其定位为一个长期的研究弧线，从保存的记忆到“梦境 (dreaming)”再到当前的系统 ([OpenAI](https://x.com/OpenAI/status/2062567556524003631)，[控制选项](https://x.com/OpenAI/status/2062567559673856346)，[Christina Kim 的解释](https://x.com/ChristinaHartW/status/2062585124450172956))。相关的开发者端更新包括 **Responses 和 Completions API 中的审核分数 (moderation scores)** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2062619558440267801))，以及一个广为流传的全新 **Codex iOS app plugin** 演示，该插件支持在浏览器中通过热重载查看和测试应用 ([OpenAIDevs 演示](https://x.com/OpenAIDevs/status/2062599291479478275))。

- 另外几个模型/数据的发布也值得关注。**Gemma 4 12B** 继续作为本地代码模型的替代方案以及高压缩版本受到关注：[Unsloth](https://x.com/UnslothAI/status/2062470072179044447) 发布了大小为 **4.66 GB** 的 **2-bit GGUF**。[@_philschmid](https://x.com/_philschmid/status/2062546814075609413) 重点介绍了一份架构解析，解释了 Gemma 4 如何在没有独立编码器的情况下处理文本/图像/音频。在多模态研究中，[@skalskip92](https://x.com/skalskip92/status/2062549751246066144) 指出 **Molmo2** 是 CVPR 上一个强有力的开源 VLM 候选模型，支持视频指向、跟踪、计数和多图推理。对于文档理解，来自 LlamaIndex 的 **ParseBench** 引入了一个开源基准测试，包含 **2000 多个经人工验证的页面**和 **16.7 万条以上测试规则**，涵盖表格、图表、忠实度、格式化和溯源性 (grounding) ([基准测试发布](https://x.com/llama_index/status/2062525204262236266))。

**热门推文（按参与度排序，已过滤技术相关性）**

- **Anthropic 关于 RSI（递归自我改进）和内部自动化**：Claude 现在编写了 Anthropic **80% 以上**的合并代码，工程师的代码交付量提升了 **8 倍**，公司表示 AI 加速 AI 开发正变得可行 ([Anthropic](https://x.com/AnthropicAI/status/2062568862479208923))。
- **OpenAI 记忆系统升级**：为美国的 Plus/Pro 用户提供更强大的 ChatGPT 记忆系统，包含摘要、控制选项和 **2 倍**的记忆容量 ([OpenAI](https://x.com/OpenAI/status/2062567556524003631))。
- **Cloudflare + VoidZero**：Cloudflare 引入了 VoidZero 团队，同时保持 **Vite 的 MIT 协议和供应商中立**，并为生态系统提供 **100 万美元的 OSS 基金** ([Cloudflare](https://x.com/Cloudflare/status/2062521221132992533), [Vite](https://x.com/vite_js/status/2062525206158078047))。
- **Nemotron 3 Ultra 发布**：面向长期运行 Agent 的开源 **550B/55B-active** 混合 MoE，提供完整训练方案和极高的速度声明 ([NVIDIA](https://x.com/nvidia/status/2062522316672667770))。
- **Cursor 画布 (canvases) + 上下文浏览器**：可共享的用于应用/报告/内部工具的画布，以及 Agent 上下文消耗去向的交互式分解 ([Cursor](https://x.com/cursor_ai/status/2062611883249783083))。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Gemma 4 12B 发布与基准测试

- **[google/gemma-4-12B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1tvtn6m/googlegemma412b_hugging_face/)** (活跃度: 1610): **Google DeepMind** 发布了 [`google/gemma-4-12B`](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)，作为 **Gemma 4** 开源权重家族的一部分。该家族涵盖了 `E2B`、`E4B`、`12B`、`26B A4B` 和 `31B` 变体，包含 Dense 和 MoE 架构、指令微调（instruction-tuned）/预训练检查点，支持多模态输入、`140+` 种语言的多语言支持，以及高达 `256K` tokens 的上下文窗口。该帖子强调了对原生 `system` 角色的支持、可配置的推理/思考模式（reasoning/thinking modes）、函数调用（function-calling）/Agent 用例、编程能力的改进，以及通过来自 [`ggml-org`](https://huggingface.co/ggml-org/gemma-4-12b-it-GGUF) 和 [`unsloth`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) 的 GGUF 版本实现的本地部署。一条热门评论链接了 Maarten Grootendorst 的 [视觉指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b)，特别指出了该模型的 *“无编码器架构（encoder-free architecture）”*。评论者主要关注实际的编程性能，其中一位明确表示想测试 Gemma 4 12B 是否能在编程任务上击败 **Qwen 3.5 9B**。评论中未提供具体的 Benchmark 结果。

    - **Maarten Grootendorst** 链接的技术指南强调了 Gemma 4 12B 的 **无编码器架构（encoder-free architecture）**，将其作为对模型内部结构感兴趣的读者的一个显著设计点：https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b 。
    - 几位评论者将 **Gemma 4 12B** 定位在较小的 Gemma 变体（如 `E4B`）和较大的模型（如 `26B`）之间的一个实用尺寸层级，一位用户还表示有兴趣了解它在编程任务上是否能超越 **Qwen 3.5 9B**。
    - 提出的一个技术问题围绕该模型显而易见的 **音频能力**，并推测如果多模态支持足够稳健，这可能会使 Gemma 4 12B 在 **语音/音频翻译** 工作流中非常有用。

- **[New Google Gemma 4 12B Claims Near-26B Performance - We Tested Both!](https://www.reddit.com/r/LocalLLaMA/comments/1tw4tmf/new_google_gemma_4_12b_claims_near26b_performance/)** (活跃度: 984): **一个本地单卡 `RTX 4090` 的对比测试声称，**Google Gemma 4 26B-A4B** 使用了 `15 GB` VRAM，以 `138 tok/s` 的速度生成了 `6.9k` tokens，并在三个 HTML5 Canvas 物理代码任务（高尔顿板、双块碰撞和混沌三摆）中表现优于 **Gemma 4 12B**。后者使用了 `9 GB` VRAM，以 `80 tok/s` 的速度生成了 `8.9k` tokens。发布者认为，尽管 MoE 架构的 `26B-A4B` 模型总参数量更大，但由于只有约 `4B` 参数处于激活状态，其速度比 12B 模型快约 `1.7×`；而 `12B` 模型对于 `16 GB` 显存的笔记本电脑仍具吸引力。该测试还被用来推广作者的本地 AI 应用 [atomic.chat](https://atomic.chat/)。** 热门评论对所谓的胜出者提出了质疑，认为视频似乎显示 **Gemma 4 12B** 在场景 2 和 3 中表现更好，其中一位询问标签是否被弄反了。另一位评论者要求与 **Qwen3.6 35B-A3B** 进行对比基准测试。

    - 多位评论者质疑测试标签/结果，称在视频对比中 **Gemma 4 12B** 的输出看起来比更大的模型更强——尤其是视频 2 和 3——其中一位指出，唯一的明显缺陷是第一次测试中 *“球的初始速度似乎太高了”*。
    - **Gemma 4 12B** 被强调的一项技术优势是多模态能力：它可以处理 **音频和视频**，同时能够适配 **显存（VRAM）较小** 的设备，这使得接近 26B 的性能在本地或受限部署中非常实用。
    - 评论者要求提供更广泛的基准，例如 **Qwen3.6 35B A3B**，并认为评估应区分任务领域：预计 **Qwen** 在定量/编程基准测试中领先，而 **Gemma 4** 在创意写作和翻译等定性语言任务上可能更具竞争力。

- **[gemma-4-12b-it vs Qwen3.5-9B 在共享基准测试中的表现：尽管参数规模更小，Qwen 在 5/8 个基准测试中击败 Gemma 成为综合赢家](https://www.reddit.com/r/LocalLLaMA/comments/1tw0lua/gemma412bit_vs_qwen359b_on_shared_benchmarks_qwen/)** (热度: 520): **该图片是一个对比 **Gemma 4 12B Unified** 与 **Qwen3.5-9B** 的技术基准测试表格，数据编译自官方 Hugging Face 模型卡（model-card）得分。尽管 **Qwen3.5-9B** 的参数规模更小且据称拥有更轻量的 KV cache，但它在 **5/8** 个共享基准测试中获胜 ([图片](https://i.redd.it/20s4116kg45h1.png))。Qwen 在 **MMLU-Pro, GPQA Diamond, TAU2, MMMU-Pro 和 MedXpertQA-MM** 上处于领先地位，而 Gemma 在 **LiveCodeBench v6, MMMLU 以及在 MathVision/MATH-Vision** 上以微弱优势领先。该帖子认为 Qwen 在单位显存性能（“GB for GB”）上更强，除了在编程领域，Gemma 或像 **OmniCoder-9B** 这样的 Qwen 微调版可能具有竞争力。** 评论者对仅凭基准测试得出的结论表示反对：有人认为 Qwen 可能被“针对基准测试过度优化（benchmaxxed）”，并表示 Gemma 在通用助手、创意写作和角色扮演方面通常感觉更好，而 Qwen 在编程方面很强。其他人则认为 Qwen 与 Gemma 的争论被夸大了，因为两者在实际的脚本/编程任务中都非常有能力，不过 Qwen 的推理模式因在上下文中填充低价值的推理文本而受到批评。

    - 几位评论者指出 **Qwen** 似乎存在“刷榜（benchmaxxed）”嫌疑，尤其是在针对编程的基准测试中，其真正的优势主要集中在涉及代码生成、工具使用或编程风格逻辑的任务。在实际使用中，用户报告 **Gemma 4 31B / Gemma 3.6 27B** 和 **Qwen** 都能生成可用的脚本，但在采纳输出之前仍需要人工检查。
    - 一个反复出现的技术投诉是 **Qwen 推理模式（reasoning mode）** 会通过生成过多的类似思维链（chain-of-thought）文本来浪费上下文空间，一位用户估计生成的推理内容中只有大约 `20%` 是有用的。这表明对于某些本地/SLM 工作流，禁用推理功能可能会提高有效上下文利用率并减少噪声。
    - 用户反馈 **Gemma** 在非编程任务上表现更好，例如通用助手、创意写作、摘要、角色扮演，甚至在某些视觉/图像理解案例中也是如此。一个引用的例子是手写笔记转录：**Qwen** 反复将一段由箭头连接的零散词段误识别为小标题，而 **Gemma 26B** 则推断出它属于正文；另一位评论者建议在 **EQBench** 和创意写作基准测试上进行测试，他们预计 Gemma 在这些方面的表现会优于 Qwen。

### 2. 长上下文扩展与 KV Cache 效率

  - **[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1twla1k/nvidianvidianemotron3ultra550ba55bbf16_hugging/)** (活跃度: 542): **NVIDIA** 发布了 [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)，这是一个拥有 `550B` 参数的 **LatentMoE** 混合模型，具有 `55B` 激活参数，交织了 **Mamba-2**、MoE、选定的 Attention 层和 **Multi-Token Prediction**；它宣称支持高达 `1M` token 的上下文，并可通过 `enable_thinking=True/False` 配置推理。该模型针对前沿推理、Agent 工作流、工具使用、多语言 RAG 和长上下文分析，声明的最小推理服务占用空间为 **`8x` GB200/B200/GB300/B300, `16x` H100, 或 `8x` H200** GPU，并遵循 [OpenMDW 1.1 许可证](https://raw.githubusercontent.com/OpenMDW/OpenMDW/refs/heads/main/1.1/LICENSE.OpenMDW-1.1)。热门评论大多在调侃对本地用户而言不切实际的硬件要求——例如：“希望我能在我的诺基亚 3310 上跑起来”和“该死，我只有 7 张 H200...”，而不是讨论模型质量或架构。

    - 一位评论者强调了列出的 **NVIDIA Nemotron-3-Ultra-550B-A55B-BF16** 极高的推理硬件要求：最小配置包括 `8x GB200/B200/GB300/B300`、`16x H100` 或 `8x H200`，这意味着该模型仅适用于大型多 GPU/数据中心部署，而非个人用户或小型实验室。
    - 提出的一个技术点是，即使该模型的输出质量略低于 **GLM** 等替代方案，它作为一种**大型、低延迟的开源模型**可能仍具有价值。讨论的权衡点是，对于延迟敏感的应用，更快的响应/处理速度可能比绝对的 Benchmark 质量更重要。

  - **[KVarN: new KV-cache quant from Huawei. 3–5× KV cache compression with actual speed-up instead of slow-down, and unlike TurboQuant it holds up on reasoning (Apache 2.0, vLLM single flag)](https://www.reddit.com/r/LocalLLaMA/comments/1twptw2/kvarn_new_kvcache_quant_from_huawei_35_kv_cache/)** (活跃度: 438): **Huawei CSL** 开源了 **KVarN**，这是一种通过单个 flag 集成到 **vLLM** 的 Apache-2.0 KV-cache 量化方法，声称相比 FP16 有 `3–5×` 的 KV-cache 压缩，达到 FP16 吞吐量的 `~1.4×`，以及 **TurboQuant** 吞吐量的 `~2.4×`，同时保持 FP16 级别的质量（[代码库](https://github.com/huawei-csl/KVarN), [论文](https://arxiv.org/abs/2606.03458)）。该帖子将 KVarN 与 vLLM FP8 KV cache（`~2×` 容量，接近 BF16 吞吐量）和 **Google TurboQuant** 进行了对比，并引用了 [vLLM/Red Hat AI 的研究](https://vllm.ai/blog/2026-05-11-turboquant)，该研究指出 TurboQuant 虽然实现了压缩，但在 AIME25 和 LiveCodeBench 等 Benchmark 的低比特模式下，吞吐量会降至 BF16 的 `66–80%`，并损失约 `20` 个推理分。核心技术声明是 KVarN 避免了 Attention 中显式的 BF16 反量化开销，并在更高的压缩比下保持推理/代码/数学精度，且无需模型更改、重新训练或校准。评论大多对这些声明表示怀疑，并担心又会出现一波低质量的量化 PR，但一位评论者主动提出要在 **B200** 上对 Qwen/Gemma 的 MTP 和非 MTP 工作负载进行 Benchmark 测试，以验证扩展性和精度保留情况。

    - 一位评论者认为关键的验证在于**并发服务**，特别是 `batch=16` 而非 `batch=1`，因为许多 KV-cache 量化方法在反量化开销占主导地位的高并发情况下会失去其显存优势。他们指出，KVarN 声称的“提速而非降速”是生产环境的关键信号，特别是如果压缩开销可以通过单个 flag 在 **vLLM** 现实的请求组合中被摊销。
    - 一位用户计划在 **NVIDIA B200** 上测试 KVarN，比较 **Qwen** 和 **Gemma 4** 的 **MTP 和非 MTP** 工作负载。这将有助于验证声称的 `3–5×` KV-cache 压缩和速度提升是否能在高端推理硬件上（而非仅在论文环境中）进行扩展。
    - 另一位评论者怀疑 KV 量化结果是否能推广到新架构，认为许多方法有效是因为当前模型在 KV cache 中存储信息的方式不够高效。他们特别要求在 **Qwen3.5** 和 **DeepSeek V4 风格的架构**上进行评估，在这些架构中 KV 信息可能存储得更密集，因此对激进压缩的耐受度可能更低。


## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. 开放图像模型与本地生成工作流 (Open Image Models & Local Generation Workflows)

  - **[Ideogram 4.0 刚刚开源！](https://www.reddit.com/r/StableDiffusion/comments/1tvtu2u/ideogram_40_just_open_sourced/)** (热度: 1087): **该[图片](https://i.redd.it/9ajk9fuu935h1.jpeg)是该帖子声称 **Ideogram 4.0** 现已开放权重（open-weight）并“已登陆 Comfy”的宣传/非技术横幅，展示了一个带有 Ideogram 徽标的电影感霓虹灯场景，而非基准测试图表或架构图。正文描述了一个具有 `9.3B` 参数的文本转图像 DiT 模型，提供 `fp8`/`nf4` Checkpoints，原生支持 ComfyUI，使用 **Qwen3-VL-8B-Instruct** 文本编码，支持包含十六进制颜色/边界框/文本元素的 JSON 结构化提示词，并报告了 `0.97` 的 X-Omni 英文 OCR 准确率。** 评论者较少关注宣传图片，而更多关注安全行为：多位用户报告该模型经过了严格的审查/“safetymaxxed”，特别是对于 NSFW 提示词，有人预测社区将尝试“abliterate”或移除这些限制。

    - 用户报告发布的 **Ideogram 4.0** 模型似乎经过了严格的安全过滤：**comfyanonymous** 指出某些被拦截的输出是因为模型被 *“safetymaxxed”*，而非 **ComfyUI** 的问题，示例图片见[此处](https://preview.redd.it/7lrd6rekg35h1.png?width=1024&format=png&auto=webp&s=988d678c1ecca642b6182749c6ade74e0c7ffaa1)。多位评论者还将其描述为对 NSFW 生成进行了硬性审查，暗示这种限制是嵌入在模型/提示词层面的，而不仅仅是在 UI 端。
    - 提出了几个技术采用障碍：评论者提到了 **watermarking**（水印）、**强力审查**以及**没有商业许可**，认为这些限制使得该开源版本对于生产或下游微调工作流的用处较小。一位用户明确总结了这种担忧：*“有水印、被审查、无商业许可。”*
    - 一位评论者强调了 **边界框 JSON 提示词** 能力是一个显著特征，并在[此处](https://preview.redd.it/0bmpbik2e35h1.png?width=1024&format=png&auto=webp&s=8ea4876bd32c8d93e34e5c226ab7a06a1720c68c)展示了示例输出。这表明 Ideogram 4.0 可能支持通过 JSON 风格的空间约束进行更结构化的布局控制，这对于确定性构图或 UI/设计生成工作流非常有用。

  - **[多角色 Anima 生成效果极佳。虽然有些渗漏（bleeding），但只会越来越好](https://www.reddit.com/r/StableDiffusion/comments/1tvv4j1/multiple_characters_anima_generations_are_so_good/)** (热度: 932): **该帖子展示了使用 **Anima** 进行的多角色图像生成，工作流已发布在作者的 [Civitai 个人资料页](https://civitai.red/user/Smexlo)上；作者指出在提示词控制、角色/细节渗漏以及解剖结构方面仍存在问题。其中一张图片使用 **Grok** 进行了后期编辑，添加了“女巫布莱尔”风格的木棍小人，而其余图片均在 Anima 中生成，作者表示期待 **WAI Anima**。** 评论者赞扬了 Anima 的多角色构图和提示词遵循能力，有人将其与 **NovelAI Diffusion V4.5** 进行了积极对比，并强调其自然语言解析能力在仅有 `500M` 参数文本编码器的情况下令人惊讶。另一位评论者报告他们“通常甚至没有渗漏问题”，暗示渗漏的严重程度可能取决于工作流或提示词。

    - 用户关注于 **Anima 的多角色提示词遵循能力**，指出它可以通过自然语言提示词构建细节丰富的场景，且角色/颜色/细节的渗漏（bleeding）相对较少。一位评论者将其与 **Illu/Pony 工作流** 进行了对比，在后者中，多角色生成通常需要强大的 Checkpoint 加上角色 LoRA，但仍受困于*“严重的渗漏”*，部分原因是 **Danbooru-tag 提示词在指定复杂场景关系方面更加受限**。
    - 一个技术上值得注意的观点是，尽管 Anima 仅使用 **`500M` 参数的文本编码器**，却实现了强大的自然语言解析能力，一位用户将其提示词遵循效果与 **NovelAI Diffusion V4.5**（作为尖端提示词遵循能力的基准）进行了对比。讨论将 Anima 视为一个早期的基准，通过社区微调和类似于 **SDXL** 时期的“民间工程（backyard engineering）”，它有望进一步改进。
    - 一位用户分享了一张宽度为 **`2560px`** 的示例输出，并表示他们 *“通常甚至没有渗漏问题”*（[图片](https://preview.redd.it/9cg06yjwo35h1.png?width=2560&format=png&auto=webp&s=bbc1ae3f5a825fb744fb7e351bc0d23d7f61def8)），这表明在 Anima 多角色生成中，渗漏可能取决于提示词或模型，而非普遍存在。

### 2. Claude Code 处理实时数据流

  - **[我通过 MCP 将 Claude Code 接入了包含每个 Polymarket 钱包和交易的数据库。接下来你们想让我问它什么？这是我目前的发现：](https://www.reddit.com/r/ClaudeAI/comments/1tvefqd/i_wired_claude_code_into_a_database_of_every/)** (热度: 1801): **作者声称他们通过 Postgres MCP 将 Claude Code 连接到了一个包含约 `1.3B` 次交易和 `2.7M` 个钱包的实时 Polymarket 账本，允许进行自然语言查询，Claude 会将其翻译为 SQL 并执行；链接的文章描述了一个类似的设置，使用 `@modelcontextprotocol/server-postgres` 在预聚合表上运行，涵盖了 `1,560,894` 个钱包的约 `1.3B` 次交易 ([CrowdIntel](https://crowdintel.xyz/blog/claude-mcp-polymarket-ledger))。报告的发现包括只有约 `20%` 的钱包实现了净盈利，`2.4%` 的钱包利润超过 `$1,000` 美元，利润极度集中在排名前 `0.1%` 的钱包中，作者还声称 Claude 发现了暗示内部交易或类机器人交易的可疑模式。** 热门评论鼓励向包括《纽约时报》/《福布斯》在内的调查记者反映情况，并建议进行更严谨的分析：将观察到的 PnL 分布与模拟的“公平市场”零模型 (null model) 进行比较，并将大额亏损的钱包/投注视为可能的洗钱或内部转账信号，而不仅仅是散户损失。

    - 一位评论者建议为公平市场下（无内部投注）的 Polymarket 钱包/交易分布建立一个**基准零模型 (baseline null model)**，然后将这些预期分布与观察到的结果进行比较。他们还建议对**大额亏损钱包/投注**进行细分，以区分潜在的内部资金提取与可能的洗钱行为。
    - 另一个技术讨论贴询问分析是否仅涵盖直接参与 Polymarket 市场的钱包，还是也执行了**资金流向追踪 (fund-flow tracing)** 以识别资金来源以及获利/亏损后的去向。这需要对钱包资金来源、提现以及潜在的关联地址进行图分析 (graph analysis)。
    - 一位评论者询问了**数据新鲜度 / 摄入延迟 (data freshness / ingestion latency)**：即投注发生与出现在 MCP 支持的数据库之间的延迟。这对于检测具有时效性的异常情况非常重要，例如新闻发布前的投注、抢跑 (frontrunning) 或结果揭晓后的交易模式。

  - **[我住在 SFO 附近，并使用 Claude Code 和 ADS-B 无线电构建了一个飞过我家上空的飞机的投影映射](https://www.reddit.com/r/ClaudeCode/comments/1tva44g/i_live_by_sfo_and_built_a_projection_mapping_of/)** (热度: 3616): **该帖子展示了一个自制的、针对旧金山机场 (SFO) 附近作者房屋上空飞行飞机的**投影映射 (projection-mapping) 可视化**，由本地接收的 **ADS-B 无线电**数据驱动，并使用 **Claude Code** 开发。链接的 Reddit 视频 ([v.redd.it/gl2b0xivvy4h1](https://v.redd.it/gl2b0xivvy4h1)) 由于 `403 Forbidden` 屏蔽无法访问，且现有文本中未提供具体的实现细节——如接收器硬件、SDR 栈、解码流水线、校准方法、延迟或投影几何结构。评论普遍持正面态度，将其视为 “Vibe Coding” 的一个优秀范例，其中一位评论者询问了该设置所需的设备。**

    - 一位评论者描述了一个针对巴西的低成本实现方案，该方案用**免费的 OpenSky API**、`US$40` 的 AliExpress 投影仪和个人电脑的直接 HDMI 输出取代了原始的 ADS-B/树莓派式硬件路径。他们增加了可配置的经纬度和半径字段，以便地图围绕用户提供的坐标重新居中，从而避免了对本地 ADS-B 天线的需求，据其估计天线成本约为 `US$100`，此外还有昂贵的本地硬件成本。
    - 人们对将该项目开源表现出兴趣，以便机场附近的其他用户可以在自己的投影设置中重复使用，并可能将飞机投影层与其他数据集（如星座/星图数据）相结合。


### 3. 前沿 AI 采用与风险信号

- **[Anthropic - 我们的内部数据显示 Claude 正在加速 AI 开发——这可能是通往 recursive self-improvement（递归自我改进）或 AI 自主构建更强大后继者的潜在路径。](https://www.reddit.com/r/singularity/comments/1twsm5g/anthropic_our_internal_data_shows_claude_is/)** (Activity: 826): **[该图片](https://i.redd.it/9ph4lq42la5h1.jpeg) 是 Anthropic 在 X 上的帖子截图**，旨在推广其文章 [“Recursive self-improvement”](https://www.anthropic.com/institute/recursive-self-improvement)。文中声称内部使用数据显示 **Claude 已经在加速 AI R&D**，并可能预示着 AI 系统协助构建更强大后继者的早期路径。在技术上具有重大意义的声明并非基准测试结果，而是一项组织层面的经验观察：Anthropic 表示 Claude 能够支持诸如 exploratory tooling（探索性工具开发）和 deferred engineering cleanup（推迟的工程清理）等工作，并将其视为与 **recursive self-improvement** 和未来 AI 控制风险相关的证据。评论者对这种表述持怀疑态度，一位用户暗示该公告是出于财务动机的市场营销。另一位用户讽刺地强调了“长期推迟的清理”这一说法，而第三位用户则提供了非 Twitter 链接的 Anthropic 文章，并引用了其警告：由 AI 构建的后继者可能会增加失控风险。

    - 一位评论者链接了 Anthropic Institute 关于 recursive self-improvement 的全文：https://www.anthropic.com/institute/recursive-self-improvement。强调的技术相关声明是，Anthropic 的内部使用数据表明 Claude 已经能够完成 *“原本不可能发生”* 的工程工作，例如 exploratory tooling 和长期推迟的清理。Anthropic 将此视为 AI 系统帮助构建更强大后继者路径上的早期信号。

  - **[Sam Altman, Dario Amodei 和 Demis Hassabis 签署联名公开信，呼吁国会强制筛查合成核酸订单](https://www.reddit.com/r/singularity/comments/1two85g/sam_altman_dario_amodei_and_demis_hassabis_have/)** (Activity: 915): **Sam Altman (OpenAI), Dario Amodei (Anthropic) 和 Demis Hassabis (Google DeepMind)** 签署了一封联合公开信，敦促国会要求筛查 **合成核酸（synthetic nucleic acid）订单**，以降低 AI 辅助病原体设计的生物安全风险。根据 [WSJ 报道](https://www.wsj.com/politics/policy/top-ai-ceos-call-for-law-protecting-against-biological-weapons-88f2f99f)，拟议的机制并非禁止合成，而是强制性的订单/客户筛查，以标记可疑的 DNA/RNA 序列或买家——这与监控散装化肥等前体购买大致类似。评论者普遍接受筛查作为一种轻量级的风险控制措施，同时也质疑目前非专业人士利用 AI 进行“超级病毒”设计在实践上是否可行。一些人将该政策定性为一种合理的针对可疑活动的触发机制，而非对合法基因工程的直接限制。

    - 评论者将该提案定性为 **订单级筛查而非禁令**，并将其与监控可疑的散装化肥购买进行类比：该机制将标记具有潜在危险的合成核酸订单，同时保留合法的生物技术访问权限。
    - 提出的技术担忧是，非专业人士利用 AI 辅助设计“超级病毒”是否现实可行。隐含的问题是，生物风险不仅取决于模型生成的序列，还取决于获取合成供应商的渠道、wet-lab（湿实验）能力、投送方法，以及合成筛查是否能捕捉到致病或工程化序列。

  - **[ChatGPT 创造历史，成为最快达到 10 亿月活跃用户的应用。](https://www.reddit.com/r/OpenAI/comments/1tvh4z4/chatgpt_makes_history_and_becomes_the_fastest_app/)** (Activity: 820): **该图片是 Kalshi 在 X 上发布的帖子截图，声称 ChatGPT 成为最快达到 `10 亿` 月活跃用户（MAU）的应用**：[图片](https://i.redd.it/uwgx8zc9j05h1.jpeg)。这并非技术基准测试或实现细节；其意义主要在于市场/采用背景，使 ChatGPT 的增长领先于之前的病毒式消费应用（如 Threads，评论者指出其在 `5 天` 内达到了 `1 亿` 用户）。评论辩论了庞大的 MAU 是否能转化为可持续收入，一位评论者估计消费者订阅的 ARPU 约为 `$1/user`，并开玩笑说增加 B2B 业务可能只能将其提高到 `$2/user`。

- 评论者关注报告中的用户指标和收入影响：一位指出，声称拥有 **`1B` 月活跃用户 (MAU)**，同时来自消费者付费订阅的收入约为 **`$1B`**，这意味着在计入企业/API 收入之前，消费者的 ARPU 约为 **`$1/用户`**。另一位评论者对 `1B` 这个数字表示质疑，引用了最近 OpenAI CFO 的播客，据报道其中的数字为 **`900M` 用户**，并认为 OpenAI 如果确认达到了十亿用户里程碑，可能会更激进地进行宣传。
- 尽管 MAU 巨大，但人们对货币化深度持怀疑态度：评论者询问报告中的用户中有多少实际上是**付费订阅者**，将头条新闻中的 MAU 增长与经常性收入、转化率以及企业/API 货币化区分开来。与 Threads 早期增长里程碑（**5天内达到 `100M` 用户**）的对比，将 ChatGPT 的规模描述为异常迅速，但尚未解决活跃使用量和付费用户留存率是否与头条宣传的采用数字相匹配的问题。

- **[研究发现，AI 在回答问题方面击败了法律教授——且差距悬殊](https://www.reddit.com/r/singularity/comments/1tvtojx/ai_beat_law_professors_at_answering_questions/)** (活跃度: 1187)：**一项与斯坦福大学相关的研究 [**“法律教授更青睐 AI 而非同行回答”**](https://law.stanford.edu/publications/law-professors-prefer-ai-over-peer-answers/) 报告了一项盲评，其中 `16` 名美国合同法教授编写了 `40` 个简答辅导问题，并对 `2,918` 个匿名的“人类 vs LLM”答案进行了成对比较评判。该 LLM（评论中确认为 **Gemini 2.5 Pro**）对教授编写答案的平均胜率为 `75.33%`，表现与最佳导师相当，且被标记为有害的频率更低（`3.53%` vs. 教授的 `12.06%`）；摘要还建议使用 LLM-as-judge 方法在判断密集型领域扩展评估规模。** 评论者讨论了辅导之外的影响：一位警告不要在法律决策或执法中过早机构化地使用 AI，而另一位则认为这一结果反映了 LLM 能力在“六根手指”阶段之后的更广泛成熟。一位技术评论者建议使用更新的前沿模型（如 **GPT-5.5**）重新运行基准测试，声称其在法律工作方面的能力可能大幅增强。

    - 链接中的斯坦福研究评估了 **LLM vs. 法律教授的简答辅导**，共有 `16` 名美国合同法教授、`40` 个教授编写的问题和 `2,918` 次盲样成对比较。教授们更青睐 LLM 的回答，平均胜率为 `75.33%`，而 LLM 回答被标记为有害的比例仅为 `3.53%`，相比之下教授回答为 `12.06%`；该论文还声称专家一致性数据可以使用单独的 LLM-as-judge 流水线进行扩展：https://law.stanford.edu/publications/law-professors-prefer-ai-over-peer-answers/。
    - 一位评论者强调，该研究使用了 **NotebookLM** 和 **Gemini 2.5 Pro**，并采用了严格限制的提示词：回答必须模仿合同法教授在办公时间（office-hours）的风格，避免使用列表符号或填充词，字数保持在 `50–108` 字左右，对于 NotebookLM，仅依靠提供的教科书章节而不引用外部案例。这种提示词设计可能降低了幻觉（hallucination）风险并标准化了回答格式，使该基准测试更多地关注简洁的法律推理/综合，而非开放式的法律研究。
    - 一项技术论点认为，法律非常适合 **RAG-style systems**，因为该行业依赖于超出个人记忆能力的庞大法典、案例法、先例和理论语料库。建议的工作流程是在权威法律材料上进行检索，然后进行综合，当模型基于相关语料库（grounded）时，其表现可能优于无辅助的律师。

# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们将不再以这种形式恢复它，但我们很快就会发布新的 AINews。感谢阅读到这里，这是一段美好的历程。