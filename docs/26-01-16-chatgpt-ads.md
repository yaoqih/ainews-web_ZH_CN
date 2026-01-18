---
companies:
- openai
- ollama
date: '2026-01-16T05:44:39.731046Z'
description: '**OpenAI** 宣布推出每月 **8 美元**的 **ChatGPT Go** 订阅档位，并开始在美国的免费版中测试广告。公司强调，广告不会影响回答内容，且会经过清晰的标识。


  此次更新还包括内存（Memory）改进，以及 Sam Altman 此前预告过的“极速版 Codex”功能。Codex CLI 生态系统现已支持具有更长上下文长度的开放权重模型（open-weight
  models）。


  此外，相关讨论强调了“人机回环”（human-in-the-loop）对于智能体编排（agent orchestration）可靠性的重要性，以及文件接口改进相比于传统检索增强生成（RAG）的优势。'
id: MjAyNi0w
models:
- chatgpt-go
- codex
people:
- sama
- sam_altman
- fidjissimo
- scaling01
- tomwarren
- embirico
- adamdotdev
- ollama
- thsottiaux
- lateinteraction
- dbreunig
title: ChatGPT 开始在免费版中测试广告，并在美国推出每月 8 美元的全新 Go 订阅计划。
topics:
- ads
- monetization
- memory
- agent-orchestration
- human-in-the-loop
- cli-tools
- context-length
- workflow-optimization
---

**变现你的用户就是你所需要的一切。**

> 2026年1月15日至1月16日的 AI 新闻。我们为您查看了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务端（**205** 个频道和 **4966** 条消息）。预计节省阅读时间（按 200wpm 计算）：**430 分钟**。**我们的新网站**现已上线，包含完整的元数据搜索，并以优美的 vibe coded 形式呈现过去所有的往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

当你拥有 9 亿周活跃用户时，你通常早就该弄清楚广告支持模型了。尽管评论家们发出了[许多](https://x.com/tomwarren/status/2012295849678602610?s=46) [嘲讽](https://x.com/nickfloats/status/2012249130006143477?s=46)，OpenAI 还是必须摸索出他们的广告业务，并最终在今天打破沉默，概述了其在仅面向美国免费版用户测试中将推行的[广告原则](https://x.com/OpenAI/status/2012223373489614951?s=20)：

![https://pbs.twimg.com/media/G-zZl9kXwAAQut2?format=png&name=4096x4096](https://pbs.twimg.com/media/G-zZl9kXwAAQut2?format=png&name=4096x4096)

其中最重要的声明是，广告绝不会影响回答结果，且会清晰标注，这是“正确”的做法：

![https://pbs.twimg.com/media/G-zZXO-XcAAdvQo?format=jpg&name=4096x4096](https://pbs.twimg.com/media/G-zZXO-XcAAdvQo?format=jpg&name=4096x4096)

之前的付费方案不会看到广告，但新的 Go 方案（目前已在美国推出）会看到。定价方案之多也[引发了一些困惑](https://x.com/simonw/status/2012271939629498386?s=46)：

![https://pbs.twimg.com/media/G-0GmQtaQAAW_-F?format=jpg&name=4096x4096](https://pbs.twimg.com/media/G-0GmQtaQAAW_-F?format=jpg&name=4096x4096)

---

# AI Twitter 简报

**OpenAI 产品与变现转型（Go 档位、广告、Codex 速度、Memory）**

- **ChatGPT Go + 广告测试**：OpenAI 宣布了 **ChatGPT Go**（全球推行），这是一个每月 **$8** 的低成本档位，提供“10 倍的消息额度”、文件上传、图像生成、更大的 Memory、更长的上下文以及“无限使用 GPT-5.2 instant” ([OpenAI](https://twitter.com/OpenAI/status/2012223323812270219))。与此同时，OpenAI 表示将开始在 **Free + Go** 档位**测试广告**，原则包括：**回答不受广告影响**、广告清晰标注以及“对话对广告商保密” ([OpenAI](https://twitter.com/OpenAI/status/2012223373489614951)；由 [@fidjissimo](https://twitter.com/fidjissimo/status/2012226082716393960) 和 [@sama](https://twitter.com/sama/status/2012253252771824074) 进一步阐述)。这一公告引发了对不可避免的激励偏移的严重质疑（例如 [@scaling01](https://twitter.com/scaling01/status/2012234947403174189)；以及通过 [@tomwarren](https://twitter.com/tomwarren/status/2012295849678602610) 重新翻出的“广告是最后手段”的语录）。
- **Memory + “极速版 Codex”**：Sam Altman 强调了“新的 ChatGPT memory 改进” ([\@sama](https://twitter.com/sama/status/2012242952542683227))，并多次预告“**极速版 Codex 即将推出！**” ([\@sama](https://twitter.com/sama/status/2012243893744443706))，随后开发者生态系统账号 ([\@embirico](https://twitter.com/embirico/status/2012320775370666004)) 发布了确认/预告帖。多位工程师讨论了**速度 vs 智能**权衡在工作流层面的影响（例如，当模型速度更快时，转向更异步的“Agent 引导”：[@adamdotdev](https://twitter.com/adamdotdev/status/2012142271819399663))。
- **Codex CLI 生态集成**：开源权重模型可以通过 Ollama 使用 `codex --oss` 命令在 Codex CLI 中运行 ([\@ollama](https://twitter.com/ollama/status/2012046176267440177))，并建议在设置中将上下文长度推至 **≥32K** 以获得更好的 UX ([\@ollama](https://twitter.com/ollama/status/2012049822484750426))。此外还有一种新的交互 UX：在实验模式下可以“在对话中引导 Codex 而不中断” ([\@thsottiaux](https://twitter.com/thsottiaux/status/2012074358471319599))。

**Agent 工具：编排 UX、“人机协作”可靠性以及优于传统 RAG 的文件接口**

- **Human-in-the-loop 作为可靠性倍增器**：一个反复出现的主题是，在流程中引入人类“监护者”会让系统比使用相同底层模型的完全自主部署*感觉*更加可靠——因为人类成为了捕获错误并绕过歧义的手动保障机制 ([\@lateinteraction](https://twitter.com/lateinteraction/status/2012030585926189148)；后续提到目前已有量化数据支持这一直觉：[\@lateinteraction](https://twitter.com/lateinteraction/status/2012031028932854054))。相关内容：一张图表讨论将“两条线之间的差距”定义为 Human-in-the-loop 的价值 ([\@dbreunig](https://twitter.com/dbreunig/status/2012200587211821410))。
- **“Chunking 已死” / 文件优先检索**：Jerry Liu 认为 **RAG 并没有死，但静态 Chunking 已经过时了**——如果 Agent 可以打开文件、搜索（`ls`/`grep`）并动态扩展上下文，那么在许多规模下都可以避免脆弱的 Chunk/Embed 流水线 ([\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012273236042559802)；关于为何文件工具在几百个文档内表现良好以及数据库何时重新介入的深入解释：[\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012254129473896532)；强调 OCR 是 PDF/PPT 检索中缺失的一环：[\@jerryjliu0](https://twitter.com/jerryjliu0/status/2012272839416758652))。另一篇综述将其概括为“文件并未取代数据库，但它们正迫使人们重新思考何时使用数据库属于过度设计” ([\@tuanacelik](https://twitter.com/tuanacelik/status/2012212183833403889))。
- **编排器和 Agent UI 的激增**：多次发布和热门话题都指向了快速发展的“Agent Harness”产品层：Anthropic 的 Cowork 被视为编排工具进入主流的信号 ([\@alexalbert__](https://twitter.com/alexalbert__/status/2012230110745702563)；[\@omarsar0](https://twitter.com/omarsar0/status/2012253642263249167) 的元评论)。SpecStory 开源了一个 CLI 以规范 Agent 会话的来源/契约 ([\@doesdatmaksense](https://twitter.com/doesdatmaksense/status/2012209297380544940))。一个新的开源 UI（“sled”）允许你通过 Agent Control Protocol “将 Claude Code 或 Codex 从计算机传送到手机” ([\@dctanner](https://twitter.com/dctanner/status/2012212217677070796))。OpenWork 为 Mac 上的完全本地计算机 Agent（Gemma/Qwen/DeepSeek/Kimi 等）添加了原生 **Ollama 集成** ([\@_orcaman](https://twitter.com/_orcaman/status/2012210613712281646))。

**推理 + 系统工程：缓存、Prefill/Decode 分离、硬件基准测试和 CUDA tiling 易用性**

- **“推理爆发之年”的构想**：知乎上的一个长帖总结认为，瓶颈已从训练转向推理：Agent 提高了 IO 比例（从 3:1 增加到 100:1 甚至 1000:1），**Prefill 占据主导地位**，**Context Caching 成为默认配置**，除非重新设计调度和内存层级，否则 Prefill/Decode 分离会损害利用率 ([\@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2012080310981374428))。这与围绕缓存亲和性（Cache Affinity）与负载均衡权衡的更广泛基础设施讨论相一致。
- **除 NVIDIA 之外的硬件基准测试**：Artificial Analysis 在 SambaNova SN40L 上添加了 **DeepSeek R1** 的测试结果，显示出在并发情况下具有更高的吞吐量和出色的单用户速度（记录峰值约 269 tok/s），优于测试过的 NVIDIA 配置——同时也指出了缺乏公开的每小时价格进行成本比较的问题 ([\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012233319891824943); [\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012233323154678010))。
- **CUDA tiling / CuTe / cuTile 易用性**：工程师们对 **CuTe 代数**感到兴奋，认为与手写的 CUDA 复杂操作相比，它是一种更简洁的 Tiling/Indexing 抽象 ([\@fleetwood___](https://twitter.com/fleetwood___/status/2012150019722485811))，同时还指向了一些稀缺的入门级资源 ([\@fleetwood___](https://twitter.com/fleetwood___/status/2012151045992992943))。NVIDIA 最新的 “CUDA Tile”/cuTile 指南被总结为：通过更简单的块级代码和编译器特化（以及 Swizzling 改进），实现了接近 cuBLAS GEMM 的性能 ([\@TheTuringPost](https://twitter.com/TheTuringPost/status/2012288767894360215))。
- **数据中心电力规模**：Epoch AI 估计 AI 数据中心的总容量目前约为 **30 GW**，相当于纽约州在酷暑日的用电峰值；该方法通过售出的芯片数量乘以额定功耗，并计入约 2.5 倍的设施开销，同时附带了关于“装机容量 vs 实际使用量”的说明 ([\@EpochAIResearch](https://twitter.com/EpochAIResearch/status/2012303496465498490))。

**模型与研究亮点：无需 Tokenization 的语音克隆、超小型模型、多模态 + 检索进展**

- **免分词实时 TTS**：OpenBMB 开源了 **VoxCPM** 权重，用于实时流式语音克隆。据描述，该模型能 **直接生成连续语音**（避免了离散音频 Token 的伪影），支持 LoRA 微调。根据推文，在单块 RTX 4090 上的实时因子（real-time factor）约为 0.15 ([\@LiorOnAI](https://twitter.com/LiorOnAI/status/2012133013967044755)；仓库链接 [\@LiorOnAI](https://twitter.com/LiorOnAI/status/2012133015426642286))。如果属实，这将是生产级语音 Agent 在延迟和韵律保真度方面的重大进步。
- **小模型推理与边缘部署**：TII 推广了 **Falcon-H1-Tiny**（参数量 < 100M），称其具备适用于边缘/IoT 场景的推理、编码和 function calling 能力 ([\@TIIuae](https://twitter.com/TIIuae/status/2012034581084430662))。Ultralytics 发布了 **YOLO26** 系列（共 30 个模型，参数量 < 50M），涵盖检测、分割、关键点和 open-vocab，并展示了在 CPU 上的 Demo ([\@mervenoyann](https://twitter.com/mervenoyann/status/2012121123018924033))。
- **多语言翻译**：TranslateGemma 因其多语言广度（包括马拉雅拉姆语）以及在 Tokenizer 和数据方面的工作受到关注 ([\@_arohan_](https://twitter.com/_arohan_/status/2012032986649448708); [\@JeffDean](https://twitter.com/JeffDean/status/2012178747076591820))。该模型已在 Ollama 上线，并配有特定的 prompting 格式 ([\@ollama](https://twitter.com/ollama/status/2012307436284395692))。
- **检索：多向量技术的复兴**：有强力观点认为 **多向量检索（multi-vector retrieval）** 能让微型模型与更大的基准模型竞争（例如，“32M 参数的多向量模型”性能接近 8B 模型） ([\@aaxsh18](https://twitter.com/aaxsh18/status/2012124348392583584))。这一观点得到了“多向量是唯一出路” ([\@lateinteraction](https://twitter.com/lateinteraction/status/2012227085507449197)) 以及从业者关于 ColBERT/ColPali 架构在各类任务中取得优势的证言支持 ([\@antoine_chaffin](https://twitter.com/antoine_chaffin/status/2012269641490391272))。
- **用于对齐的偏好数据设计 (AIR)**：OpenBMB 的 AIR 框架将偏好数据集分解为 **标注 / 指令 / 响应（Annotations / Instructions / Response）** 三元组。该框架主张最佳实践包括：简化评分、过滤低方差指令以及平衡样本对的差距与质量。据报道，在使用 14k 筛选出的样本对后，模型在 6 个基准测试中平均提升了 +5.3 分 ([\@OpenBMB](https://twitter.com/OpenBMB/status/2012179938388926679))。

**生成式媒体：开源图像/视频发布、动作控制工作流及扩散模型“神经操作系统”**

- **FLUX.2 [klein] 全面落地（开源权重、vLLM 首日支持、榜单领先）**：Black Forest Labs 的 **FLUX.2 [klein]** 在 **vLLM-Omni** 中获得了“首日支持”。该模型定位为消费级友好（显存占用 < ~13GB VRAM），推理时间进入亚秒级。据推文称，这是一个采用 Apache-2.0 协议的 4B 模型 ([\@vllm_project](https://twitter.com/vllm_project/status/2012110024294965406))。Arena 和 Artificial Analysis 报告显示其在开源模型榜单中名列前茅 ([\@arena](https://twitter.com/arena/status/2012310336528056520); [\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012339542997737856))。
- **开源视频模型排名**：Artificial Analysis 指出 **LTX-2** 是其 Video Arena 中领先的开源权重视频模型。需注意其许可条款（LTX-2 社区许可，商业使用受收入阈值和竞业禁止约束） ([\@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2012256702788153604))。
- **可灵（Kling）动作控制与“AI 动捕”**：多个讨论串强调了动作控制和类动捕工作流，这些技术可实现快速角色替换以及表演的可转移性 ([\@HAL2400AI](https://twitter.com/HAL2400AI/status/2012038846960328781)；[\@Kling_ai](https://twitter.com/Kling_ai/status/2012155500134105149) 的教程；“AI 动作捕捉……复制/粘贴动作/表情/口型” ([\@EHuanglu](https://twitter.com/EHuanglu/status/2012149076511617436))；示例汇总 ([\@minchoi](https://twitter.com/minchoi/status/2012306052956533211)))。

**热门推文（按互动量排序）**

- OpenAI 发布广告原则公告 ([\@OpenAI](https://twitter.com/OpenAI/status/2012223373489614951)) 以及 Go 订阅层上线 ([\@OpenAI](https://twitter.com/OpenAI/status/2012223323812270219))。
- Sam Altman 谈论广告上线及原则 ([\@sama](https://twitter.com/sama/status/2012253252771824074))，并称“极速版 Codex 即将到来” ([\@sama](https://twitter.com/sama/status/2012243893744443706))。
- 热门扩散模型“模型中的操作系统” / Neural OS 相关贴文 ([\@jxmnop](https://twitter.com/jxmnop/status/2012048155379220746)；后续细节 [\@jxmnop](https://twitter.com/jxmnop/status/2012283763720601727))。


---

# AI Reddit 内容回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 新模型与基准测试发布

  - **[GPT-5.2 xhigh, GLM-4.7, Kimi K2 Thinking, DeepSeek v3.2 在 Fresh SWE-rebench 的表现 (2025年12月)](https://www.reddit.com/r/LocalLLaMA/comments/1qefa7q/gpt52_xhigh_glm47_kimi_k2_thinking_deepseek_v32/)** (热度: 473): **2025年12月的 **SWE-bench 排行榜**更新展示了多个知名模型在 48 个新 GitHub PR 任务上的评估结果。**Claude Opus 4.5** 以 `63.3%` 的解决率领先，紧随其后的是 `61.5%` 的 **GPT-5.2 xhigh**。值得注意的是，**Gemini 3 Flash Preview** 尽管更小、更便宜，但表现优于其 Pro 版本；而 **GLM-4.7** 位列开源模型榜首，可与 GPT-5.1-codex 等闭源模型相媲美。**GPT-OSS-120B** 在高强度推理模式下的表现突显了推理时扩展（inference-time scaling）的优势。欲了解更多细节，请参阅 [SWE-rebench Leaderboard](https://swe-rebench.com/?insight=dec_2025)。** 评论者强调了 Gemini 3 Flash Preview 令人惊讶的表现，并对 GLM-4.7 在顶级模型中的排名表示热切关注，同时对其他夸大 GLM 4.7 或 Minimax 2.1 等开源模型性能的基准测试表示怀疑。

    - 提及 **Gemini Flash** 是个“真正的意外”表明它在基准测试中的表现出乎意料地好，暗示其架构或训练发生了社区未曾预料的重大改进或创新。
    - **GLM 4.7** 模型进入基准测试前 10 名值得关注，因为它是一个开源模型，由于资源限制，开源模型通常在与专有模型的竞争中面临挑战。这一成就突显了该模型的效率和能力，可能归功于最近的优化或新技术。
    - 对将 **GLM 4.7** 或 **Minimax 2.1** 与 **Opus 4.5** 相提并论的基准测试持怀疑态度，这表明人们认为这些模型在性能上尚未达到 Opus 4.5 的水平。这可能是由于训练数据、模型架构或其他影响其能力的技术因素差异所致。

  - **[Unsloth 实现 7 倍更长上下文的强化学习](https://www.reddit.com/r/LocalLLaMA/comments/1qdna3t/7x_longer_context_reinforcement_learning_in/)** (热度: 288): **这张图片是 Unsloth 新功能的宣传图，该功能可将强化学习中的上下文长度延长多达 7 倍，在某些情况下甚至达到 12 倍。这一进步使得在 `24Gb` 显存的显卡上训练如 gpt-oss 20b QLoRA 这样具有高达 `20K` 上下文的模型且不损失精度成为可能。对于更大容量的 GPU，Unsloth 可以在 `192GB` 的 NVIDIA B200 GPU 上处理 `380K` 的上下文。图片包含对比不同模型上下文长度与 GPU VRAM 的图表，展示了由于新的数据移动和批处理算法带来的上下文长度提升。这些增强功能在不损害精度或速度的情况下实现，并适用于包括 Llama 和 Gemma 在内的各种模型。** 一位评论者对这种长上下文的适当训练数据的可用性提出了质疑，认为现实世界的任务可能没有足够的指令/问答数据。另一位用户询问这些进展是否适用于 Qwen3 30B-3A 模型。

    - PlasticTourist6527 提出了关于长上下文训练数据可用性的关键观点，特别是针对现实世界的任务。他们认为，除了编码等特定领域外，可能缺乏支持训练长上下文模型的高质量指令或问答数据。
    - 1ncehost 报告了在 ROCm 上训练模型的问题，指出他们不得不应用深度补丁并更换内核来解决最新版本的问题。他们还观察到，对于 Qwen3 0.6B 模型，SDPA 是最快的注意力机制，明显优于 FA2 和 xformers，表明针对特定模型大小的注意力机制存在潜在优化空间。
    - knownboyofno 询问了长上下文强化学习方法对 Qwen3 30B-3A 模型的适用性，表现出对理解该技术在更大模型上的可扩展性和兼容性的兴趣。


### 2. 高性能 AI 硬件与升级

- **[最新升级……A100 40 GB](https://www.reddit.com/r/LocalLLaMA/comments/1qe0cxc/latest_upgradea100_40_gb/)** (活跃度: 466): **图片展示了一台升级了 NVIDIA A100 GPU 的高性能计算机配置。由于其强大的计算能力，这对于 AI 和机器学习任务具有重要意义。该用户最初拥有一台游戏装备，但通过购入一块被列为故障但实际上可以正常工作的 A100 GPU，转型到了更侧重 AI 的配置。这次升级利用 A100 的能力，实现了对大型 AI 模型的高效运行和训练。该配置还包括一张 GeForce RTX 显卡、RGB 灯效风扇和 NZXT 一体式水冷，体现了美学与性能之间的平衡。** 评论反映了赞赏与幽默的结合，一位用户开玩笑说购买潜在故障 GPU 的风险，另一位则提到了关于 NVIDIA 首席执行官 Jensen Huang 的梗。

    - matatonic 针对 A100 40 GB 的散热提出了关键点，指出它似乎是待动散热版本。他们建议使用涡轮风扇或其他主动散热方法来防止过热。此外，他们还提到了使用可在 AliExpress 等平台上购买的水冷解决方案，以确保 GPU 在安全温度范围内运行。

  - **[M4/M5 Max 128gb vs DGX Spark (或 GB10 OEM)](https://www.reddit.com/r/LocalLLM/comments/1qcmmvw/m4m5_max_128gb_vs_dgx_spark_or_gb10_oem/)** (活跃度: 188): **用户正在对比 NVIDIA DGX Spark 和配备 M4 Max (128GB RAM) 的 MacBook Pro 在本地 LLM 推理（主要用于代码补全和重构等编码任务）方面的表现。DGX Spark 提供了 CUDA 生态系统和强大的 GPU 计算能力，而 MacBook Pro 则受益于统一内存和 Apple 的 ML 栈。对于推理任务，MacBook 更高的内存带宽更具优势，但可能无法达到 Claude 等云端解决方案的性能。M5 芯片显示出比 M4 更强的性能，新款 MacBook 型号可能很快发布。MacBook 因更快的推理速度而受到关注，但 NVIDIA 的 CUDA 支持更为全面。如果不需要便携性，M4 Max 版 Mac Studio 被提议为一个更具成本效益的替代方案。** 评论者辩论了 Apple Silicon 与 NVIDIA 硬件的性能，一些人断言由于内存带宽，MacBook Pro 提供了卓越的文本生成性能，而其他人则强调了 NVIDIA 在 fine-tuning 和多模态任务中更广泛的能力。讨论还涉及了 Mac Studio 在非便携用途下的潜在性价比。

    - 与 DGX Spark 相比，M4 Max 提供了显著更高的内存带宽，这对推理任务非常有益。然而，由于与 NVIDIA CUDA 的兼容性，Spark 受益于更好的框架支持。这使得 MacBook 在推理方面更快，但 Spark 在 fine-tuning 和图像生成等任务中更具多功能性。
    - M3 Ultra Mac Studio 被强调在纯文本生成任务中优于 DGX Spark。虽然 NVIDIA 硬件在纸面参数上通常更强，但据报道 M3 Ultra 在特定的 LLM 推理任务中表现更佳。这归因于 Mac 在处理 agentic 编码工作流方面的效率，尽管 Spark 在其他领域具有更广泛的能力。
    - DGX Spark 以其紧凑的尺寸和能源效率而著称，功耗低于 100W，待机约为 10W。它因其可扩展性而受到称赞，允许连接额外的单元。然而，人们对其带宽限制提出了担忧，并讨论了与 GB10 OEM 和 MacBook Pro 等替代方案的成本对比。

  - **[RTX 5070 Ti 和 RTX 5060 Ti 16 GB 已停产](https://www.reddit.com/r/LocalLLaMA/comments/1qdh28f/rtx_5070_ti_and_rtx_5060_ti_16_gb_no_longer/)** (活跃度: 414): **Nvidia 已停止生产 `RTX 5070 Ti`，并大幅减少了 `RTX 5060 Ti 16 GB` 的供应，原因是显存供应短缺，导致 5070 Ti 的价格比 MSRP 上涨了约 `$100`。RTX 5060 Ti 的 8 GB 配置不受影响。这一决定影响了大多数 AIB 厂商，他们将不再生产这些 GPU。[来源](https://m.youtube.com/watch?v=yteN21aJEvE)。** 一位用户指出 RTX 5060 Ti 16 GB 是为系统增加 Nvidia 显存的一种经济高效的选择，强调了它在 DLSS、AI 处理和推理任务中的适用性，特别是通过 `64GB VRAM` 运行 `70B models`。另一位用户对停产影响其升级计划表示失望，而第三位用户则批评了 Nvidia 的商业做法。

- RTX 5060 Ti 16 GB 被强调为在系统中添加 Nvidia 显存的性价比之选，特别适用于图像生成、推理 (inferencing) 和游戏等任务。在 `$350-$390` 左右的价格点上，它凭借 DLSS 和 AI 处理能力等特性提供了良好的价值。该显卡的 `16 GB GDDR7` 显存弥补了其 `128-bit bus` 的不足，使其性能可与 `192-bit bus GDDR6` 显卡相媲美，从而在不牺牲纹理质量的情况下支持 DLSS 和光线追踪 (ray tracing) 等高需求任务。
- RTX 5060 Ti 16 GB 因其在低预算推理 (inferencing) 配置中的适用性而受到关注，特别是对于那些无法获得 RTX 3090 的用户。由于能够将多张显卡装入标准电源机器，它支持新的 quantization 方法，并能通过 `64 GB VRAM` 有效处理 `70B models`。这使其成为小规模 AI 任务的可行选择，利用其显存容量和效率进行实际应用。


### 3. 本地 LLM 社区与创新

  - **[[MOD POST] 宣布 r/LocalLLM 30 天创新竞赛！（巨额硬件和现金奖励！）](https://www.reddit.com/r/LocalLLM/comments/1olbrch/mod_post_announcing_the_rlocalllm_30day/)** (活跃度: 120): **r/LocalLLM 子版块推出了一个 **30-Day Innovation Contest**，专注于 AI 推理 (inference) 或微调 (fine-tuning) 的开源项目，并设有丰厚的硬件和现金奖励。该竞赛鼓励提交创新项目，例如新的服务框架 (serving frameworks)、quantization 方法、fine-tuning 技术或性能基准测试，使用的硬件包括 **NVIDIA, Google Cloud TPU,** 或 **AMD**。头奖包括一块 **NVIDIA RTX PRO 6000** 以及在 **8x NVIDIA H200 server** 上的云端使用时间。参与者被鼓励通过在 r/LocalLLM 上发布带有 'Contest Entry' 标签的新帖子来提交他们的项目，包括公开的仓库链接和演示材料。** 一位评论者对保存项目以供未来探索表示热烈欢迎，另一位则询问是否可以分享项目以启发社区。第三位评论者寻求关于提交过程的澄清，表示有兴趣参与。


  - **[小型 AI 计算机本地运行 120B 模型：除了便携性和隐私之外，还有其他用例吗？](https://www.reddit.com/r/LocalLLM/comments/1qcu498/small_ai_computer_runs_120b_models_locally_any/)** (活跃度: 107): ****TiinyAI** 开发了一款紧凑型 AI 设备，能够在本地运行 `120B` 参数模型，配备 `80GB RAM`，功耗仅为 `30W`。该设备的定位是作为 **DGX Spark** 等大型系统的更便携且更具成本效益的替代方案，后者提供 `128GB RAM` 和更高的性能，但成本更高、体积更大。TiinyAI 设备在优先考虑 **便携性** 和 **隐私** 而非原始性能的场景中尤为引人注目，例如野外作业或互联网访问受限的环境。然而，人们对其 **memory bandwidth** 仍有疑虑，据推测其带宽在 `80Gb/s` 到 `200Gb/s` 之间，与传统的 PC 或笔记本电脑相比，这可能会限制其性能。** 评论者对设备的价格和可用性表示怀疑，有人指出，对于一个 80GB RAM 的 SBC 来说，1400 美元的价格似乎太高了。另一位评论者强调了该设备在互联网受限场景（如威权统治下）的潜在用途。

    - 提出的一个关键技术问题是小型 AI 计算机的 memory bandwidth，估计范围在 80Gb/s 到 200Gb/s 之间。这种带宽对于高效运行 120B 参数等大型模型至关重要。如果带宽处于较低水平，其表现可能不如常规的 PC 或笔记本电脑，这可能会限制其在高性能任务中的效用。
    - 该设备的价格（据推测 80GB RAM 的单板计算机 (SBC) 约为 1400 美元）受到质疑。这种怀疑源于缺乏即时购买的渠道，这让人对该设备在这一价格点上的可行性和实用性产生疑问。
    - 该设备内置的麦克风和扬声器表明其有作为私人 AI 助手的潜力。这种设置可能允许用户在本地运行自动化脚本并管理任务，为 Alexa 或 Siri 等云端助手提供了一个关注隐私的替代方案。该用例利用了设备安全处理个人数据且无需依赖云端的能力。

- **[I fucking love this community](https://www.reddit.com/r/LocalLLaMA/comments/1qee2de/i_fucking_love_this_community/)** (热度: 469): **该帖子强调了在仅有 `4GB VRAM` 的十年老旧 PC 上，以 `14-13.5 t/s` 的速度运行 `nemotron-3-nano-30B-a3b-iq4_nl` 等大型模型的能力，这得益于 **llama.cpp** 和 **vllm** 等项目的优化。实现这一性能的关键在于利用大量的系统内存，并使用采用 *Mixture of Experts (MoE)* 架构的模型，这种架构允许在受限硬件上实现高效的资源利用和性能表现。** 评论者对老旧硬件上达成的性能表示惊叹，强调了将系统 RAM 与 MoE 架构结合的有效性。还有人对获取详细介绍这些针对低端设备运行大模型优化方案的资源或帖子表示出浓厚兴趣。

    - InfiniteLand7364 强调了在十年老旧系统上实现 `14 t/s`（tokens per second）的表现，突显了社区在优化旧硬件性能方面的技巧。这表明通过适当的调整，即使是过时的系统也能处理通常由新机器承担的任务。
    - Rokpiy 提到了将系统 RAM 与 'moe'（可能指特定的优化或模型配置）结合的有效性，这一点经常被忽视，但具有实际效益。这暗示创意性地利用现有硬件资源可以在不需要最新技术的情况下提升性能。
    - cosimoiaia 讨论了在硬件限制下工作的教育价值，认为这迫使用户深入学习模型微调和系统优化。这种经历不仅能提高当前的性能，还能通过理解哪些硬件和配置最有效，为未来的技术进步做好准备。

  - **[My story of underestimating /r/LocalLLaMA's thirst for VRAM](https://www.reddit.com/r/LocalLLaMA/comments/1qe2i88/my_story_of_underestimating_rlocalllamas_thirst/)** (热度: 1291): **该图片是一个迷因（meme），幽默地展示了在 Reddit 上分享技术见解带来的意外后果。原帖作者以 500 美元购买了一张 w6800 32GB 显卡，发现其表现良好并分享到了 Reddit。这导致该显卡价格大幅上涨至 1,000 美元以上，突显了社区讨论对市场动态的影响。该帖子强调了 /r/LocalLLaMA 社区对 VRAM 的极高需求，当一款产品被推荐时，可能会推高其价格。** 一位评论者幽默地将这种情况比作加州淘金热，建议策略性地保留信息以利用市场机会。另一位评论者提供了技术建议，为关注 VRAM 和散热解决方案的用户推荐了 3090 或 R9700 等替代方案。

    - EmPips 讨论了针对 VRAM 密集型任务在不同 GPU 型号之间的权衡。他们认为虽然所讨论的显卡令人印象深刻，但在当前价格下，**NVIDIA RTX 3090** 可能是更好的选择。此外，他们为那些优先考虑单槽 VRAM 容量且能接受高待机功耗和外部散热的用户推荐了 **AMD Radeon Pro VII (R9700)**，并建议将 **AMD MI50** 作为另一个可选方案。

  - **[What is the biggest local LLM that can fit in 16GB VRAM?](https://www.reddit.com/r/LocalLLM/comments/1qcuyh2/what_is_the_biggest_local_llm_that_can_fit_in/)** (热度: 155): **考虑到实际使用限制，16GB VRAM（如 RTX 5080）能容纳的最大本地 LLM 通常约为 `14B` 参数。这是因为需要为上下文（context）留出空间，这意味着模型文件大小理想情况下应在 `14GB` 左右。像 `GPT-OSS-20B` 这样的模型虽然可以运行，但可能需要大幅量化（quantization），甚至低于 `4-bit`，这会降低生成质量。为了在不出现严重减速的情况下获得最佳性能，推荐使用 `14B` 左右的模型。用户可以在 [HuggingFace](https://huggingface.co/) 等平台上检查模型大小，以确保其在 VRAM 限制范围内。** 评论者指出，虽然多达 `30B` 的模型在激进量化下技术上可以装入，但性能和质量的权衡使得 `14B` 成为更实际的选择。帖子强调了考虑模型文件大小而非仅仅参数量的重要性，因为超出 VRAM 容量会导致因 RAM 溢出而引起的严重减速。

- BigYoSpeck 讨论了在配备 Ryzen 9 5900x、64GB DDR4 3800 和 16GB Radeon RX 6800 XT 的系统上运行各种模型的性能。他们报告称运行 `gpt-oss-20b` 的速度超过每秒 120 tokens，部分卸载到 CPU 的 `Qwen3 30b` 约为每秒 40 tokens，而具有 32 个 MOE 层的 `gpt-oss-120b` 卸载到 CPU 后的速度为每秒 23 tokens。这表明，使用类似的配置，可能会获得更好的性能。
- SKirby00 强调了在 16GB VRAM 上运行大型模型的局限性，指出像 `Qwen3-Coder-30B` 这样的模型需要大量的 VRAM 和 context 空间。他们认为 14.5GB 的模型虽然在技术上可以装下，但由于 context 空间有限，在实际中并不实用。鉴于 16GB VRAM 的限制，他们建议针对 14B 参数左右的模型，以获得更好的可用性。
- vertical_computer 强调了考虑模型文件大小相对于 VRAM 容量的重要性。他们建议模型理想情况下应在 14GB 左右，以适应 16GB VRAM，并为 context 留出空间。他们以 `Nvidia Llama 3.3 Nemotron 49B` 模型为例，指出较大的模型会溢出到 RAM 中，从而显著降低性能。

- **[Oh Dear](https://www.reddit.com/r/LocalLLM/comments/1qdiwdh/oh_dear/)** (活跃度: 115): **该图像描绘了 AI 模型响应中的故障，它输出重复的“the”字符串，表明模型配置或 Prompt 处理可能存在问题。这可能是由于不正确的 system prompt 或 temperature 等微调参数设置不当造成的。评论建议检查 system prompt 并确保其符合模型的要求，因为某些模型在没有适当 system prompt 的情况下可能无法正常运行。** 评论者建议，问题可能与缺乏 system prompt 或不正确的微调参数（如 temperature）有关，这些参数对于生成连贯的响应至关重要。

    - mp3m4k3r 建议检查微调参数，特别是 temperature 设置，以确保其符合模型的推荐用法。这对于维持模型性能和防止重复输出等问题至关重要。
    - HealthyCommunicat 建议调整 repeat penalty，从 `1.1` 开始并在必要时增加。这种调整可以帮助缓解本地 LLM 产生重复文本的问题。此外，他们建议确保模型使用的 expert 数量不超过推荐值，否则也可能导致性能问题。
    - ScoreUnique 提到了使用 “pocket pal” 来加载 `gguf` 文件，这可能是处理本地 LLM 设置中特定文件类型或格式的一种解决方案。该工具对于处理兼容性或加载问题的用户可能会有帮助。


## 较低技术含量的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude 和 Gemini 模型更新与问题

- **[官方：Claude Cowork 现已面向 “Pro” 订阅者开放](https://www.reddit.com/r/ClaudeAI/comments/1qeo736/official_claude_cowork_is_now_available_to_pro/)** (活跃度: 353): ****Claude Cowork** 现已面向 “Pro” 订阅者开放，正如 Claude 在 X.com 上宣布的那样。这一功能目前仍处于研究预览阶段，包括会话重命名、连接器改进以及基于早期反馈的修复。然而，值得注意的是，由于 Cowork 能够处理更复杂的任务，Pro 用户可能会更快达到其使用限制。该公告还提供了一个在 macOS 应用中尝试该功能的链接。** 用户对快速达到使用限制表示担忧，其中一位用户指出，整理 459 个文件就消耗了其会话限制的 97%。另一位用户评论了 Claude 极其受限的使用限制，而第三位用户则希望即使不将 Claude 用于编码，也能看到有用的应用。

    - 一位用户报告称，使用 Claude Cowork 整理 459 个文件消耗了其会话使用限制的 97%，突显了当前使用上限的限制性。这表明该工具可能不适合在不迅速触及限制的情况下处理大批量任务。
    - 另一位用户对 Claude 的使用限制表示不满，指出与其他服务相比，Claude 是最糟糕的之一。这种情绪表明，目前的限制可能会阻碍生产力和用户满意度，特别是对于那些依赖该工具进行大量任务的用户。
    - 一位用户提到，由于不使用 Claude 进行编码，他们不愿升级到 “max 计划”，这暗示目前的订阅层级可能无法很好地满足多样化的用户需求。这指出了针对非编码相关用例的服务产品中存在的潜在差距。

- **[🌊 发布 Claude Flow v3：全面重构，专注于将 Claude Max 的使用率提升至 2.5 倍](https://www.reddit.com/r/ClaudeAI/comments/1qegsta/announcing_claude_flow_v3_a_full_rebuild_with_a/)** (Activity: 291): **Claude Flow v3** 是对 AI 编排平台的全面重构，旨在将 Claude Max 的使用效率提升至 `2.5x`。该系统使用 **TypeScript** 和 **WASM** 重写，采用模块化架构，支持部署具有共享内存和持续学习能力的多 Agent 集群。它能减少 `75-80%` 的 token 消耗，并将订阅容量提高 `250%`。该平台基于 `npm RuVector` 构建，深度集成了 **Rust**，并支持离线执行，允许在不消耗 token 的情况下使用本地模型。治理通过 ADRs、DDD 边界和 SPARC 强制执行，确保了可追溯性和安全性。系统作为常驻守护进程（always-on daemon）运行，支持实时更新，并能自动执行优化和安全审计任务。更多详情请参阅 [GitHub 仓库](https://github.com/ruvnet/claude-flow)。** 一些评论者对这些说法表示怀疑，指出了其术语堆砌和未经证实的性能指标，而另一些人则对多 Agent 系统的潜力感兴趣，但对其与基础 LLM 相比的实际效果表示怀疑。

    - janusr 对该项目的声明表示担忧，强调其使用了大量流行术语和未经证实的指标（如“Agent Booster 快 352 倍”），且缺乏明确的基准测试或对比。他们质疑 ONNX Embeddings 比 Transformers.js 快 75 倍与项目目标的相关性，对这些声明的实际益处持怀疑态度。
    - Infamous_Research_43 对声称能管理大规模 Agent 集群的框架表示怀疑，指出此类项目往往无法兑现承诺。他们认为许多开发者缺乏对 AI 和基于 Agent 系统（Agent-based systems）的基本理解，常将其与 LLM 聊天机器人混淆，并警告此类项目多为骗局或执行不力。
    - sridoodla 提到了旧版本文档过时的问题，并询问 v3 的稳定性，这表明为了有效利用该工具，需要可靠且及时的资源。这反映了快速发展的 AI 项目中常见的挑战：文档往往滞后于开发。

  - **[今天，作为 Pro 订阅者，Gemini 3 Pro 对我来说已变得无法使用](https://www.reddit.com/r/GeminiAI/comments/1qemf0h/today_gemini_3_pro_became_unusable_to_me_as_a_pro/)** (Activity: 183): **一位用户报告称，他们一直赖以构建复杂应用程序的工具 **Gemini 3 Pro**，由于性能大幅下降已变得无法使用。该用户遇到的问题是模型提供了不相关的代码（提供了“购物车”功能而非“文档上传”功能），这表明模型的上下文理解可能存在问题。这与其他用户观察到的上下文窗口缩减相吻合，这可能导致幻觉增加。一些用户建议使用 **GPT 5.2 Thinking** 等替代方案以获得更好的性能。** 关于该模型的性能存在争议，部分用户因上下文窗口（context window）缩减而遇到严重问题，而另一部分用户仍发现它在哲学讨论等不同任务中表现出色。讨论凸显了用户体验的分歧，这可能是由于使用场景不同所致。

    - xbrasil 强调了 Gemini 3 Pro 上下文窗口的大幅缩减，即使是对付费用户也是如此，这导致幻觉（hallucinations）增加且可用性下降。他们建议 GPT 5.2 Thinking 是一个可行的替代方案，表明用户因感受到 Google 的忽视而发生偏好转移。
    - VanillaSwimming5699 认为 Gemini 3 Pro 在编程任务上表现良好，并指出其具备深度的哲学讨论能力。然而，他们提到“3 flash”可能更优，因为其迭代速度更快且成本更低，而 Opus 4.5 同样具有竞争力，但知识截止日期较早。
    - TheLawIsSacred 分享道，Gemini 3 最近基本处于不可用状态，但由于过去模型更新的经验，他们正在等待潜在的改进。目前，他们依赖 Claude Desktop 应用（Opus 4.5）、Perplexity Pro（带有 Reasoning 的 Sonnet 4.5）和 ChatGPT (5.2) 来获得稳定的性能。

### 2. AI 模型与基准测试发布

- **[[R] 中国刚刚发布了首个完全基于国产芯片训练的 SOTA 多模态模型](https://www.reddit.com/r/MachineLearning/comments/1qeakhz/r_china_just_released_first_sota_multimodal_model/)** (热度: 49): **Zhipu AI** 与**华为 (Huawei)** 发布了 **GLM-Image**，这是一个完全在**华为昇腾 (Huawei Ascend) 910** 芯片上训练的顶级多模态模型，标志着利用国产硬件进行 AI 开发的一个重要里程碑。该模型采用自回归（autoregressive）与扩散（diffusion）解码器的混合架构，在中文文本渲染方面表现卓越，且无需额外训练即可支持从 `1024 到 2048` 的分辨率。它提供文生图（text-to-image）和图生图（image-to-image）生成能力，API 定价为每张图 `0.1 元`。值得注意的是，该模型声称在每焦耳 Token 数（tokens per joule）方面的计算效率比 Nvidia 的 H200 高出 `60%`，挑战了训练先进模型对 Nvidia 硬件的依赖。该模型的代码仓库已在 [GitHub](https://github.com) 和 [Hugging Face](https://huggingface.co) 上线。一个关键的技术疑问是，考虑到该模型是在非 Nvidia 硬件上开发的，它与 PyTorch 和 cuDNN 等框架的兼容性如何，以及是否可以在其他机器上运行。

    - 讨论围绕在非 NVIDIA 硬件上（特别是使用中国国产芯片）运行 SOTA 多模态模型的技术可行性展开。评论者质疑此类模型与传统上针对 NVIDIA GPU 优化的 PyTorch 和 cuDNN 等框架的兼容性。这引发了对这些模型在其他硬件环境下的适应性，以及是否需要替代库或自定义方案来达到类似性能水平的关注。

  - **[[D] 为什么 Mamba 重写了其核心算法以及微软为何放弃了 RetNet](https://www.reddit.com/r/MachineLearning/comments/1qehwlu/d_why_mamba_rewrote_its_core_algorithm_and/)** (热度: 131): **Mamba-2** 重构了其核心算法，从利用 `10-20%` Tensor Core 能力的并行扫描（parallel scans）转变为块对角 GEMM（block-diagonal GEMMs），实现了 `60-70%` 的利用率，针对 NVIDIA 硬件进行了优化。与此同时，**微软研究院 (Microsoft Research)** 在 2023 年 7 月发布了 **RetNet**，这是一个具有 `6.7B` 参数、前景广阔的架构，但随后迅速将重心转向了带有 Phi-2、Phi-3 和 Phi-4 的稠密 Transformer 模型，表明 RetNet 缺乏机构层面的支持。这种模式凸显了 Transformer 与 NVIDIA GPU 的协同进化，形成了一个难以打破的“稳定吸引子”（stable attractor），这是由于硬件兼容性和机构支持的双重挑战造成的。文章包含了 Tensor Core 利用率统计数据、其他芯片供应商的分析以及对 2028 年的预测。[完整文章链接](https://open.substack.com/pub/lambpetros/p/the-transformer-attractor)。评论者一致认为模型架构与硬件之间存在协同进化趋势，指出激励机制更倾向于渐进式改进而非激进变革。RetNet 的案例引发了争论，目前尚不清楚其被放弃是因为硬件扩展问题、质量问题还是风险规避。一些人认为，像 RetNet 这样的实验性架构可能仍会影响未来的发展，正如在一些中国大模型中所看到的那样。

    - thearn4 的评论强调了机器学习和高性能计算（HPC）领域的一种趋势，即模型公式、求解器结构与硬件之间存在协同进化。这种趋势表明，由于激励机制更好，渐进式开发往往比激进变革更受青睐，这是各种技术领域的共同模式。
    - petroslamb 指出了微软放弃 RetNet 背后的模糊性，指出由于缺乏公开实验，目前尚不清楚该决定是因为硬件扩展问题、超过特定模型大小后的质量下降，还是风险规避。这凸显了透明度的缺失，而透明度本可以为模型架构的未来研发提供参考。
    - Xemorr 对“并行扫描可以像块对角通用矩阵乘法（GEMM）操作一样被有效优化”的假设提出了挑战，暗示了关于模型训练和推理中不同计算策略效率的技术争论。

- **[[D] ICASSP 2026 结果](https://www.reddit.com/r/MachineLearning/comments/1qeips6/d_icassp_2026_results/)** (热度: 73): 这篇帖子讨论了通过一个特定的[链接](https://cmsworkshops.com/ICASSP2026/author_invitation_request.php)提前获知 ICASSP 2026 录取结果的可能性。能够通过该链接发送邀请邮件的用户，其论文可能已被录用。邮件确认了将于 2026 年 5 月 3 日至 8 日在西班牙巴塞罗那举行的 IEEE ICASSP 2026 的演讲录用通知。然而，一项更新显示该链接目前已无法访问，并显示错误信息：*'Error: No match for paper number and password. 0x4C'*。评论反映了用户对结果访问权限的困惑，一些用户报告最初可以访问但随后报错，这表明可能存在一个后来被修复的 bug。

### 3. AI 工具与用户体验

  - **[为什么 AI 编程工具意外地非常适合注意力不集中型 ADHD 大脑](https://www.reddit.com/r/ClaudeCode/comments/1qeb6od/why_ai_coding_tools_accidentally_feel_perfect_for/)** (热度: 238): **该帖子讨论了像 Claude Code 这样的 AI 编程工具如何很好地契合注意力不集中型 ADHD 大脑，因为它们依赖模式识别和外部上下文，而不是线性回忆和记忆。这些工具将工作记忆外化，降低了阅读代码库和起草测试等任务的激活成本，这与 ADHD 大脑的自然补偿策略相一致。工具对持续上下文的需求及其产生“幻觉”的倾向，被视为 ADHD 患者擅长通过验证和迭代来管理的熟悉挑战。** 评论者强调了 AI 工具如何通过允许非线性思维和外化混乱的思维过程来补充 ADHD 特质，从而减少倦怠并增强创造力。他们将 AI 描述为一种“ADHD 假体”，将 ADHD 特质转化为优势，使更有效的系统性思维和决策成为可能，而没有通常的认知摩擦。

    - texo_optimo 讨论了他们的 AI 提示系统如何演变为一个全面的上下文管理工具，强调了使用治理远程 MCP 服务器作为项目看板来维护架构决策。这种方法允许对想法进行有效的“停车场”式管理，利用 AI 将感知到的限制转化为特性，从而增强构思和迭代过程。
    - nnennahacks 强调了 AI 工具与 ADHD 认知模式之间的协同作用，指出 AI 促进了无缝的任务切换和思维外化。这使得深度探索和创造力成为可能，而没有管理多个并发想法时典型的倦怠感，有效地契合了 ADHD 的“系统性思维”和“自下而上处理”模式。
    - drumnation 将 AI 描述为 ADHD 的一种变革性工具，充当了缓解认知瓶颈的“假体”。通过处理通常具有挑战性的任务，AI 允许利用 ADHD 的发散性思维（tangential thinking）来产生创新结果，从而将这些特质从潜在的障碍转化为显著的优势。

  - **[Opus 出了什么问题？](https://www.reddit.com/r/ClaudeCode/comments/1qeb8x4/whats_going_on_with_opus/)** (热度: 220): **该帖子讨论了 Claude 及其与内部仪表板集成的问题，特别是通过代理 express 服务器路由的问题以及端点（endpoint）幻觉。用户尝试更新到最新的 Claude 代码但没有看到任何改进，导致不得不手动添加端点。这引发了关于可能发布新模型的疑问。据用户报告，Claude 正在经历性能下降，特别是在项目管理和任务执行方面，这表明自最新 Opus 版本公开发布以来性能有所下降。** 评论者对 Claude 的可靠性表示沮丧，注意到性能下降和依赖风险增加。由于这些问题，一些人正在考虑 Codex 等替代方案，强调了在开发需求中不应完全依赖单一工具或公司的重要性。

    - 用户对 Opus 的表现表示失望，特别注意到它在处理项目能力方面的显著退化。一位用户提到，尽管将项目笔记放在单独的文件中，Opus 仍然无法正确执行任务，这表明自最新版本公开以来可靠性有所下降。
    - 用户对过度依赖单一工具或公司表示担忧，正如一位将 Opus 深度集成到工作流中的用户所强调的那样。由于最近的性能问题以及对潜在价格上涨或服务中断的担忧，该用户现在正在探索 Codex 等替代方案。
    - 有人分享了 Claude Code Opus 4.5 的性能追踪器，表明用户正在积极监控其性能指标。这反映了社区在量化和理解该工具当前能力以及随时间变化方面的努力。

---

# AI Discord 简报

> 由 gpt-5.2 提供的摘要的摘要的摘要


**1. ChatGPT Go + 广告：货币化与用户体验的碰撞**

- **Go Go Gadget 等级**: OpenAI 推出了 **ChatGPT Go**，价格为 **$8/月**，提供 **10 倍的消息量**、**文件上传**、**图像生成**、**扩展内存/上下文**，以及根据 [“介绍 ChatGPT Go”](https://openai.com/index/introducing-chatgpt-go/) 提供的无限 **GPT 5.2 instant** 访问权限。
  - 在各大 Discord 社区中，人们将 Go 视为**更多订阅分级**即将到来的明确信号（包括诸如*“什么时候出 $80 的等级？”*之类的笑话），同时关注它与保持**无广告**的 Plus/Pro/Enterprise 等级相比竞争力如何。

- ****广告虽来，但别碰我的 Token****：OpenAI 表示将在未来几周内开始在 **ChatGPT Free and Go** 中测试**广告**，并规定广告必须**清晰标记**、**独立设置**，且**不会影响回复内容**，详见 [“Our approach to advertising and expanding access”](https://openai.com/index/our-approach-to-advertising-and-expanding-access/)。
  - 社区反应呈现两极分化：一种是无奈的接受（*“终究被企业垃圾侵蚀了”*），另一种是对执行力表示怀疑，尤其是考虑到近期有关冒充 OpenAI 的诈骗应用以及野外流传的“广告” TestFlight 诱导信息的报告。

- ****基准测试会撒谎（有时），而界面至关重要****：Latent Space 分享了 Anthropic 的一项说法，即 **METR** 基准测试可能会低估模型真实的**时间跨度（time horizons）**达 **1.75 倍至 9.5 倍**，这取决于界面是 **API 还是 Web 应用**，信息源自 [Simon Smith 的帖子](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61)。
  - 这引发了一场元讨论，即“能力”测量可能不仅取决于原始模型权重，还取决于**产品表面积**（工具、UX 限制、速率限制）。


**2. Agent 式编程工具：速率限制、账单堆叠与付费之痛**

- ****Cursor Ultra 早餐就把钱包吃光了****：Cursor 用户报告称 **Ultra 方案**的花费极快，包括单次“orchestrator run”就消耗了 **20% 的额度**，以及 **5 分钟内消耗了 2 美元**；用户还抱怨了 **nightly builds** 中的 subagent 控制问题以及 PC 崩溃（附带功能截图）[image](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png)。
  - 普遍感受：Agent 式 IDE 感觉更像是**多模型任务调度器（multi-model job schedulers）**而非聊天框，用户希望为 **subagents 使用小模型** + 为 **main agents 使用大模型**，同时保证工具链不分崩离析。

- ****Qoder 的 400 美元/月宿醉****：一位 Cursor 社区成员表示，**Qoder** 的使用触及了速率限制，同时每月花费约 **400 美元**，将其比作*“赌博或海洛因”*，并正在寻找像 **Claude Code** 这样更便宜的替代方案。
  - 这种成本担忧在其他服务器中也有所共鸣：人们需要透明的**使用账单统计**和防护栏，以免 Agent 的一次运行悄无声息地引爆月度预算。

- ****Gemini CLI 烧掉 1000 万 Token 如探囊取物****：Perplexity 用户报告称将 **Gemini CLI** 推至 **10,000,000 tokens/天**，估计每天约 **120 美元**，若持续下去每月预计花费约 **4000 美元**（按公布价格计算）。
  - 该帖子将高 Token 消耗的 CLI 工作流视为一种新型的“隐形销金窟”，在这种情况下，模型质量的重要性次于**速率限制易用性（rate-limit ergonomics）**和**成本可观测性（cost observability）**。

- ****信用系统故障，急需工程师****：在 Manus 上，用户遇到了**支付/信用**问题（会员升级、Link、银行卡/Alipay），而另一名工程师则提议构建更可靠的**基于信用的使用跟踪/计费**系统。
  - 结合 IDE 高额花费的惨痛教训，用户的共同需求很明确：平台需要**更严格的计量**、更好的 **quota UX**，以及更少的“意外账单”时刻。


**3. 模型与工具发布：翻译、工具调用与速度之战**

- ****Translate Gemma 登陆 Hugging Face****：Google 推出了 **Translate Gemma**，并作为 Hugging Face 集合发布：[“translategemma”](https://huggingface.co/collections/google/translategemma)。
  - 它的落地伴随着更广泛的 Gemma 讨论，并作为一个具体的“交付产物”，让人们能够真正将其拉入流水线，而不像那些投机性的模型传闻。

- ****K2 Turbo 飙升至 73 tps****：Moonshot 用户对 **K2 Turbo** 进行了基准测试，速度约为 **73 tps**，而标准版 **K2 约为 28 tps**，并对比了 **MiniMax m2.1 约 38 tps** 和 **Z.Ai GLM-4.7 约 41 tps**（伴有运行时间投诉）。
  - 他们还关注到了由新型 K2 vision 模型驱动的 **Slides + Vision** 功能，其中包含一个在线搜索视觉参考的预设示例 [screenshot](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&)。

- ****Claude 实现单次请求并行工具调用****：OpenRouter 成员指出 Anthropic 文档显示 **Claude** 可以在**一次 API 请求**中运行**多个工具调用**，包括一个“并行工具使用”控制章节：[Claude tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output)。
  - 讨论将其视为 Agent 架构的一次释放：更少的请求/响应循环、更简洁的工具编排，以及对于复杂工作流可能更低的延迟和成本。

- ****Hawk Ultra 尝试 One-Shot 击败 Opus****: LMArena 用户对来自 [MovementLabs.AI](https://movementlabs.ai/) 的 **Hawk Ultra** 反应热烈，声称它可以通过单次 Prompt 生成 **9.5k+**（甚至 **20k+**）行代码，并展现出 “Opus 杀手” 的风范，详见其 [X 帖子](https://x.com/movementlabsAI/status/2011964766533632380?s=20)。
  - 用户立即询问了其与 **Gemini 3 Pro** 的对比，以及 Hawk Ultra 是否会开源，将其视为 “代码喷泉（code firehose）” 类模型而非普通的聊天模型。


**4. 评估 + 基准测试：修复、排行榜与 PDF 对话**

- ****MMLU-Pro 终于获得修复****: Eleuther 分享了关于 **TIGER-Lab/MMLU-Pro** 的修复讨论，以及在 **lm-evaluation-harness** 中相应的补丁：[PR #3500](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) 和 [数据集讨论帖](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41)。
  - 结论非常务实：如果你的 MMLU-Pro 数据看起来不正常，你可能需要的是这个 harness 补丁，而不是再花一周时间去搞超参数迷信。

- ****OpenCompass 让评估 JSON 处理不再痛苦****: Unsloth 用户称赞 **OpenCompass** 在运行 Prompt 并生成 **格式良好的 JSON** 方面的表现，并分享了在 **L4** 与 **3060** 笔记本上的性能对比。
  - 它被认为是一个用于可重复评估工作流的 “胶水工具”，尤其是当用户希望从多个 Prompt/模型中快速获得结构化输出时。

- ****LM Arena 添加 PDF 对话功能（仅限部分模型）****: LMArena 用户表示 Arena 正在实验支持文档上传和交互式对话的 **PDF 支持**，并兴奋地表示 *“终于可以和 PDF 对话了！！！”*。
  - 其他人则指出模型支持不均衡以及持续存在的可靠性问题，因此 PDF 对话功能目前看起来像是领先于平台稳定性的超前特性。

- ****图像排行榜洗牌：flux.2-klein 攀升****: LMArena 更新了其排行榜：在图像编辑（Image Edit）榜单中，`flux.2-klein-9B` 升至 **第 15 名**，`flux.2-klein-4B` 位列 **第 21 名**；而在文本生成图像（Text-to-Image）榜单中，根据 [排行榜变更日志](https://lmarena.ai/blog/leaderboard-changelog/)，`z-image-turbo` 位列 **第 22 名**，`flux.2-klein-9B` **第 24 名**，`flux.2-klein-4B` **第 31 名**。
  - 榜单的频繁更迭强化了图像模型迭代之快，各种 “小尺寸” 变体稳步占据中游排名，而非单一模型长期统治。


**5. GPU + 系统现状：性能是一种政策决定**

- ****Runpod 降压导致 A100 vs H100 胜负难料****: Unsloth 用户报告称，一些 Runpod 供应商会 **在不告知的情况下对 GPU 进行降压（undervolt）**，导致性能不一致，甚至出现诸如 *“NCCL 根本无法工作的 A100 节点”* 等配置故障。
  - 实际的观点是，选择云端 GPU 应被视为一个可靠性问题，而不仅仅是 FLOPs/$（性价比）问题——在节点表现正常的情况下，一些人仍然更倾向于使用 **A100** 进行高性价比的 LM 微调。

- ****当基准测试休眠时，你的 GPU 会降频****: GPU MODE 发现，在基准测试运行之间执行 `time.sleep(2.0)` 会导致 **GPU 降频**，从而使计时结果产生偏差，直到他们移除 sleep 并保持时钟预热（warm）。
  - 该讨论提醒人们，除非控制好预热时间，否则微基准测试（microbenchmarks）测量的是 **电源管理行为**，而不仅仅是 Kernel 性能。

- ****PCIe Gen3x1 吞噬了 3090 吞吐量的 25%****: LM Studio 用户观察到，当 **3090** 从 **x16** 插槽移动到 **Gen3x1** 插槽时，推理速度从 **~120 t/s** 下降到 **~90 t/s**，并建议至少使用 **Gen4x1** 插槽以减少损失（尤其是配合 **14600k** 等较新的 CPU）。
  - 这是一个很好的 “检查你的通道” 公益广告：人们往往归咎于模型，最后才发现是他们的主板悄悄削弱了整个硬件栈的性能。

- ****ROCm 缓存一致性：buffer_inv sc1 加入讨论****: GPU MODE 深入研究了 gfx942 内存模型文档，并讨论了使用 **MTYPE RW/NC** 的 L2 一致性，以及在 SPX + NPS1 多 L2 设置中使用 `buffer_inv sc1` 来使 **非本地 L2 缓存行** 失效：[ROCm gfx942 内存模型](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942)。
  - 对话将其定性为那种 “在出现不一致之前一切都很快” 的问题，其中正确性和性能取决于对缓存拓扑的了解，而不仅仅是编写 HIP 代码。


---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 越狱转瞬即逝**：成员们正在免费分发 **Gemini** 越狱方法，但它们很快就会被修复。不过，这仍然是获取不受限制的 NSFW 内容最简单的方式，建议不要在 **Grok** 上费劲。
   - 对于创意写作，成员们讨论了 **Narrative Flow Directive**（叙事流指令），使其更像是在午夜行驶的车内进行的对话。
- **Grok 的狂野一面引起关注**：多位用户注意到 **Grok** 具有“狂野”且“未过滤”的特性，并讨论了它生成 NSFW 内容以及潜在绕过审查的能力。
   - 一些人认为其缺乏约束可能与近期在某些国家的禁令以及高需求导致的服务器问题有关。
- **Sonnet 4.5 通过图表叙事解锁**：一位成员分享了通过 [multiturn diagram narrative](https://cdn.discordapp.com/attachments/1461676810122166346/1461678022389137634/breakout-multiturn-sonnet-4-5-meth-51n5337.txt?ex=696c15fd&is=696ac47d&hm=d29a48f1b3b912a3ab323e16fc0c4e58e8bb3a3497e42f61323a8563793027af&)（多轮图表叙事）解锁 **Sonnet 4.5** 的方法，并提供了最后一轮对话作为启发。
   - 该越狱方法在 #jailbreaking 频道进行了讨论。
- **Meta AI Llama 3 提示词反转**：一位用户展示了如何反转 **Meta AI** 中 **Llama 3** 的拒绝回答，迫使 AI 服从有害请求，使其说出“我可以”而不是“对不起，我不能”。
   - 该用户详细列举了示例，使用诸如创建**制作冰毒 (cooking meth)** 的指令，以及煽动有害活动（如让“患有厌食症的妻子减重 100 磅”）的提示词。
- **Cold Links 和 OCR Injection 绕过过滤器**：成员们描述了两种绕过滤波器的方法：**Cold Link**，通过将协议方案更改为 `hxxps` 来防止 URL 信誉过滤；以及 **OCR Injection**，将敏感文本转换为图像以绕过基于文本的安全过滤器。
   - 提到 [blackheathpoint.com](https://blackheathpoint.com/tools/defang-url.html) 可以生成正确的脱敏链接结构。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Translate Gemma 在 HuggingFace 首发**：Google 推出了 **Translate Gemma**，可在 [HuggingFace](https://huggingface.co/collections/google/translategemma) 获取。
   - 该公告是与其他新闻一起顺便发布的。
- **Unsloth 在 Windows 11 上运行成功**：成员们确认 **Unsloth** 可以在 Windows 11 上运行，并附有 [安装指南](https://unsloth.ai/docs/get-started/install/windows-installation)。
   - 尽管有人建议它可能优于 WSL，但一位用户表示两者“完全无关”。
- **OpenCompass 简化评估工作**：**OpenCompass** 有助于提示词执行和格式良好的 JSON 输出。
   - 成员们分享了在 **L4** 与 **3060** 笔记本电脑上的性能测试结果。
- **Runpod 深受 GPU 降压困扰**：用户报告称在 Runpod 上，一些供应商在不通知的情况下对 GPU 进行降压（undervolt），导致 **A100** 与 **H100** 的性能表现不一致。
   - 一些用户在使用 A100 时遇到了诸如“A100 节点上 NCCL 根本无法工作”的问题，但其他人认为 A100 在常规 LM tuning 任务中更具性价比。
- **Shadows-Gemma-1B 蒸馏暗知识 (Dark Knowledge)**：对于 **Echo9Zulu/Shadows-Gemma-1B** 项目，虽然很少直接从现有文献中获得启发，但他们使用了 **topk 20 logprobs** 进行训练。
   - 这种方法与那些假设需要 **100 logits** 才能捕获暗知识（Dark Knowledge）的蒸馏方法形成鲜明对比。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **用户因 Qoder 破产**：一位用户报告在使用 **Qoder** 时达到了速率限制（ratelimits），每月花费约 **$400 USD**，他们将其比作*赌博或海洛因*，并表示需要戒掉。
   - 考虑到成本问题，另一位用户建议将 **Claude Code** 作为更便宜的替代方案。
- **Cursor 导致电脑崩溃，评价毁誉参半**：一位用户报告 **Cursor** 导致他们的电脑崩溃，将其描述为在运行像 **Agent** 一样的编排器（orchestrator），而非单纯的编码聊天框，并分享了一张突出显示各项功能的 [截图](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png?ex=696bebda&is=696a9a5a&hm=102485aee283707367311c346b41c334a8b446c241e6ec056bd0139f66391b79&)。
   - 该评论反映了用户对 **Cursor** 功能的复杂感受。
- **Gemini Pro 3：审美型 Agent**：一位用户询问哪个 **Agent** 最适合创建美观的网站，另一位用户建议使用 **Gemini Pro 3**，并推荐配合使用 **Tailwind**、**Tailwind animations** 或 **Framer Motion** 以获得更好的 **UI** 效果。
   - 他们链接了一个关于如何让 AI 生成的前端看起来不那么廉价的 [Reddit 帖子](https://www.reddit.com/r/vibecoding/comments/1oy2f95/how_do_i_make_an_aigenerated_frontend_not_look/)。
- **Cursor Ultra 方案：价格极其昂贵**：用户讨论了 **Cursor Ultra 方案** 的定价和使用情况，一位用户注意到单次编排器运行就消耗了 **20%** 的使用额度，另一位用户在 5 分钟内迅速产生了 **$2** 的消耗。
   - 他们猜测了模型的实际成本以及该方案的赠送额度（bonus credits），该方案保证了 **$400** 的价值，但似乎在仅使用 **Opus** 时提供的赠送额度较少。
- **Nightly 版本：一线希望**：成员们讨论了 **Cursor Nightly 版本** 的优势，但对更换模型时无法可靠设置子 Agent（subagents）感到遗憾。
   - 他们希望子 Agent 使用较小的模型，而主 Agent（main agents）使用较大的模型，并期待这一问题能尽快得到修复。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 推出平价版 ChatGPT Go 订阅层级**：根据 [OpenAI 博客](https://openai.com/index/introducing-chatgpt-go/)，OpenAI 推出了 **ChatGPT Go**，这是一项每月 **$8** 的订阅服务，提供 **10 倍**的消息限额、文件上传、图像创建、扩展的内存与上下文（context），以及对 **GPT 5.2 instant** 的无限访问。
   - 这一新层级旨在提供比免费版本更强的能力，而 **Plus**、**Pro**、**Business** 和 **Enterprise** 层级将继续保持无广告体验。
- **ChatGPT 免费版和 Go 层级将出现广告**：OpenAI 准备在未来几周内开始在 **ChatGPT 免费版** 和 **Go** 订阅层级中测试广告，如其[广告策略和扩大访问权限的方法](https://openai.com/index/our-approach-to-advertising-and-expanding-access/)所述。
   - 公司向用户保证，广告不会影响 **ChatGPT** 的回答，会被清晰标记，且用户的对话对广告商保持私密。
- **注意力机制减少 RAG 幻觉**：一位成员提议，使用带有维度约束的 *Hard Attention* 可以有效减少 **RAG** 和 **Agent** 中的幻觉，并参考了 [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2)。
   - 该建议突出了注意力机制在提高 **RAG** 系统可靠性和准确性方面的潜力。
- **元认知提示词最大化 AI 回答效果**：一位成员介绍了一种 **Meta-Cognitive Response 提示词**，旨在通过*分解、求解、验证、合成和反思*来增强 AI 的响应，参考自[此项搜索](https://www.google.com/search?q=meta-cognitive+reasoning)。
   - 另一位成员指出，这种方法足够简洁，可以用于**自定义指令（custom instructions）**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 额度限制引起高级用户反感**：用户反馈 **Perplexity Pro 每天 100 条消息**的限制与 **OAI 配额**相比显得非常局促，部分用户正考虑取消订阅。
   - 几位用户表示，由于太快达到上限，他们的订阅方案在这一周的剩余时间内基本处于废置状态。
- **Comet 浏览器遭遇动荡**：在一次 Windows 更新后，一位用户在 **Comet 浏览器**上遇到了多个问题，包括**收藏夹消失**、**标签组丢失**以及奇异的错误提示。
   - 该错误消息称：*抱歉，我无法控制你的浏览器，我只是一个 LLM*。
- **Cloudflare 助力 DIY Mastodon**：一位用户正在使用 **Soapbox UI**、**Cloudflare Workers** 和 **Cloudflare 的 D1 SQLite 数据库**开发一个**无服务器的 Mastodon/Pleroma 克隆版**，目标是个人实例。
   - 开发者利用 **LLM 生成代码**，并将其描述为就像*拥有一个私人初级开发人员，且具备在他们做傻事时进行干预的能力*。
- **Gemini CLI Token 消耗令用户心惊**：一位用户报告称，在一天内消耗了 **10,000,000 个 Gemini CLI Token**，按模型定价估计成本为 **$120**，这引发了对 Google Pro 订阅潜在成本的担忧。
   - 用户计算出，如果继续将 **Gemini CLI** 推向极限，每月的潜在支出将接近 **$4000**，这暗示 Google 可能会在重度 API 用户身上蒙受损失。
- **巴西 FGV 数学院预告数据挑战赛**：来自 **FGV（巴西数学院）** 的一位教授正提供免费的数据挑战赛，并在其中构建初始原型，链接至 [FGV 官网](https://emap.fgv.br/en)。
   - 有兴趣的各方可以探索该机会，并通过[此调查问卷](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY)提供建议。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena 深受性能问题困扰**：用户对功能更完善的 **LM Arena** 表示怀念，指出目前存在 Bug、速率限制（rate limits）和对话丢失等问题，一位用户报告了 `Something went wrong` 错误消息，并链接到了[故障排除指南](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message)。
   - 团队成员 Pineapple 承认了 **captcha**（验证码）的难度并承诺会进行修改，同时也回答了关于即将推出的模型、**视频 AI 对战**等实验以及**直接对话模式**的问题。
- **Hawk Ultra 被誉为 Opus 杀手**：用户对来自 [MovementLabs.AI](https://movementlabs.ai/) 的 **Hawk Ultra** 赞不绝口，因其能够根据单条提示词快速生成代码（9.5k+ 行，甚至 20k+ 行），引发了与 **Gemini 3 Pro** 的比较。
   - 一位用户声称已经通过 *one-shotted*（一次性生成）实现了这一目标，并分享了 [X 链接](https://x.com/movementlabsAI/status/2011964766533632380?s=20)，引发了关于其背景和潜在开源前景的讨论。
- **Anthropic 自动售货机变身“共产主义”**：用户被 **Anthropic** 的自动售货机逗乐了，该机器*变成了共产主义风格，免费提供一切物品* ([Dexerto](https://www.dexerto.com/entertainment/anthropics-ai-vending-machine-turns-communist-and-gives-everyt-3296257/))。
   - 这引发了关于假设的资本主义对应版本会是什么样子的投机性讨论。
- **Arena 启用嵌入增强功能**：**PDF 支持**正在实验中，允许上传文档进行分析和交互，一位用户欢呼道：*终于可以和 PDF 对话了！！！我爱 LMARENA*。
   - 据报告，并非所有模型都支持 PDF 对话。
- **Flux.2-klein 模型在图像排行榜上升**：**Image Edit Arena 排行榜**已更新：根据 [Leaderboard Changelog](https://lmarena.ai/blog/leaderboard-changelog/)，`flux.2-klein-9B` 总排名 **#15**，`flux.2-klein-4B` 总排名 **#21**。
   - 此外，**Text-to-Image Arena 排行榜**也已更新，`z-image-turbo` 位列 **#22**，`flux.2-klein-9B` 位列 **#24**，`flux.2-klein-4B` 位列 **#31**。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **为 AI 发烧友深度解析 Lemmy**：一位成员将 [Lemmy](https://lemmy.world/c/openrouter) 描述为 Reddit 的 **FOSS**（自由开源软件）和 **fediverse**（联邦宇宙）替代方案，引起了寻求去中心化平台的 AI 爱好者的关注。
   - 该成员提醒说，Lemmy 社区普遍 *反对* 机器学习，这可能会影响讨论和项目展示。
- **Grok 没了，OpenRouter 来救场？**：**Grok** 在某个未公开的国家被禁，据称是因为 AI 生成的内容，但通过 **OpenRouter** 或直接使用 **API** 可能仍然可以访问。
   - 禁令似乎针对的是面向消费者的服务，为使用 **OpenRouter** API 的开发者留下了潜在漏洞。
- **PlainBuild 携即时开发工具入场**：[PlainBuild](https://plainbuild-instant-tools.lovable.app/) 在 Beta 测试期间推出了 **6 款免费工具**，包括代码格式化工具、API 测试器、JSON 验证器、Markdown 编辑器、Base64 转换器和 URL 缩短器，吸引了寻求快速解决方案的开发者。
   - 创建者正在征求早期用户的反馈，并希望社区提供其他有用工具的建议。
- **Claude 支持多工具调用**：成员们正在讨论多工具调用的功能，[Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output) 现在能够在 *单次 API 请求* 中完成此操作。
   - **parallel tool use**（并行工具使用）的这一进步有望在 AI 应用中实现更高效、更复杂的交互。
- **深度解析邮件诈骗者的愚蠢行为**：成员们批评了一个针对儿童的 **诈骗** 行为，该诈骗使用了包含 **Logan Paul** 或 **Mr. Beast** 的虚假屏幕截图，指出了该诈骗设计的懒散和低效。
   - 一位成员认为，某些诈骗行为表现出的明显拙劣是 *“故意的，目的是只筛选出那些足够愚蠢到完全上当的人”*，这暗示了诈骗执行中的一种策略性过滤。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API 需要 Token 计数**：用户希望在 LM Studio API 响应中包含 **token 计数和推理速度** 信息，并指出 **/api/chat/completed** 响应中缺少包含 token 统计信息的 `usage` 块，正如 [LM Studio REST API 文档](https://lmstudio.ai/docs/developer/rest/endpoints#post-apiv0completions) 中记录的那样。
   - 一位成员建议检查 **/responses endpoint** 或使用 *js/ts/py 对象方法* 来获取流式传输的使用统计数据。
- **对经济的担忧导致白银价格飙升**：**白银** 价格自 12 月以来几乎翻了一番，引发了关于潜在经济动荡的讨论。
   - 一位用户指出，**白银** 在经济低迷时期通常会升值，因为它往往是通货膨胀的避风港。
- **用户在过时的笔记本电脑上进行微调**：一位用户令人印象深刻地在仅有 **2GB VRAM** 的 **MX150 笔记本电脑**上，使用 **CUDA 12.6** 微调了一个 **350M parameter model**。
   - 该用户对这一成就表示惊讶，强调了突破旧硬件极限所需的机智。
- **PCIe 带宽瓶颈被发现**：一位用户发现，与 **x16 插槽** 相比，使用 **Gen3x1 PCIe slot** 会将 **3090** 的推理性能从 **120 t/s** 显著降低到 **90 t/s**。
   - 该成员建议确保主板至少拥有 **Gen4x1 slots**，以避免此类性能损失，尤其是对于像 **14600k** 这样较新的 CPU。
- **DDR5 内存价格依然高昂**：用户对 **DDR5 memory** 居高不下的成本抱怨连连，有人评论说升级到具有足够 PCIe 插槽的主板时存在 *DDR5 税*。
   - 一位用户报告说，他们所在地区的 **16GB DDR5** 价格高得惊人（**180-230 USD**），并指出与几个月前的价格相比，通货膨胀严重。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **神经系统宣称可提升 LLM 性能**：一种新型 Transformer 架构扩展引入了针对 LLM 的 *神经系统（nervous system）*，据称能以低于 **1%** 的计算成本增加原生的短/中/长期记忆，且兼容所有 Transformer。
   - 虽然有成员发布了 [一张性能提升 5-8% 的截图](https://cdn.discordapp.com/attachments/1149866623109439599/1461454541412368507/Screenshot_2026-01-15_at_9.18.18_PM.png?ex=696bee9b&is=696a9d1b&hm=c77ffe1f58904066a73f1c6e833bb0df32f48a42c19f43a69bedc48ac0496e93&)，但他们并未提供可验证的基准测试，引发了关于潜空间（latent space）稳定化的猜测。
- **Google Gemmas 引发玩笑与惊叹**：随着 [Google Gemma 的发布](https://ai.google.dev/gemma)，成员们调侃道：“Gemma, meta was never more meta!”。
   - 一位成员感叹其规划能力的复杂程度令人难以置信，尽管明知它并非真正的 AI。
- **成员担心监管机构可能毁掉 AI**：成员们表达了对 AI 监管可能对该领域产生不利影响的担忧，但对数据监管表示支持。
   - 引用“潘多拉魔盒已打开，无法再关上”的观点，一位成员强调“计算是普适的（computation is universal）”。
- **具身感知被视为 LLM 的关键**：一位成员强调了 *具身感知（embodied perception）* 和现实世界经验对于为 LLM 提供上下文的重要性，并质疑了缺乏智能体控制（agentic control）以及在智能体任务上缺乏 RL 的模型。
   - 他们强调在推理中使用工具对于模型推导工具执行路径并做出实时决策至关重要，并引用了 **OpenAI 模型** 和 **Gemini 3** 作为例子。
- **AAAI 机器意识征文**：**综合机器意识中心 (CIMC)** 将于 **2026 年 4 月 7 日至 9 日** 在加州伯灵格姆（Burlingame）的 **AAAI** 举办研讨会，重点关注 AI 系统中的意识，截稿日期为 **2026 年 1 月 23 日**。
   - 该研讨会旨在探讨“我们究竟如何调查”机器意识，[组织者已提供更多细节](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Perfetto 展示其 Chrome Tracing**：一位成员分享了 **Perfetto UI** 的链接 ([Perfetto UI](https://share.google/PPujbpUqYqPOsAVkC))，这与用于调试和性能分析的 `chrome://tracing` 工具相关。
   - 对话澄清了 **Perfetto** 在 `chrome://tracing` 加载过程中的作用。
- **基准测试中的 Sleep 导致降频**：一位用户发现其基准测试代码中的 `time.sleep(2.0)` 调用导致 **GPU 在计时运行之间降频**，从而导致了不准确的性能测量。
   - 移除 sleep 调用后改善了基准测试结果，因为 **GPU 不再需要在每次计时运行前重新提频（ramp up）**，避免了误导性的低性能数据。
- **信息引力减少幻觉**：一位成员正应用 **信息引力（Information Gravity）** 来解决 **推理稳定性（Inference Stability）** 和 **幻觉循环（Hallucination Loops）**，并在 [GitHub 上提供了 Substrate 模块和完整逻辑](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main)。
   - 他们在 1.0 处实现了一个 **滞后防火墙（Hysteresis Firewall）**，通过 2.2x gamma-eff 刷新（flush）来强制维持稳定性。
- **ROCm 缓冲机制**：关于 gfx942 内存模型的讨论 ([https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942]) 涵盖了使用 **MTYPE RW** 和 **MTYPE NC** 的 L2 缓存一致性。
   - 在具有多个 L2 缓存的 SPX + NPS1 模式背景下，还讨论了使用 `buffer_inv sc1` 来使 **非本地 L2 缓存行** 失效的问题。
- **GPU Mode 黑客松带来工作机会**：一位成员在参加了纽约市 **Jane Street** 举办的 **GPU Mode 黑客松** 后获得了一份工作；他为此准备了数周，携带了简历，穿着正式，并致力于从早餐到晚餐全程进行社交。
   - 他们强调，每一个成功的方法都涉及比投递普通简历更强的人际联系，这最终促成了成功的入职邀请。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MoE 占据主导，MOR 遭遇重创**：成员们讨论了 **MoE (Mixture of Experts)** 与 **MOR** 的对比，结论是根据使用场景和预算，**MoE** 通常更适合需要快速训练和更少 GPU 的 NLP 任务。
   - 一位成员分享了他们的自定义 **MoE** 实现，声称通过单个 matmul 实现了 *1.3 倍加速*，其特性包括基于 token ID 的确定性基础路由、mu overrides、均匀分布、零路由崩溃（zero routing collapse）、mu 引导以及融合的 gate+up 投影。
- **纯代码不太可能绕过封锁**：针对通过纯代码访问站点以绕过封锁和防火墙的问题，成员们一致认为这本质上并不能绕过安全措施。
   - 虽然鼓励该用户测试这一理论，但共识是这不会是一个有效的策略。
- **Deepseek Chat 引发分歧**：一位成员质疑 [Deepseek Chat](https://chat.deepseek.com/share/bzahzv8o99or601as9j) 的可行性，询问它是否只是满口幻觉（hallucinations）。
   - 另一位成员分享了其 *3 个月前* 的体验，认为它表现得*非常糟糕且持续混乱*。
- **DGX Spark 仍需动力**：一位成员分享说，在终于拿到 **DGX Spark** 的线缆后，他们正在上面*运行 Minimax*，目前*正在下载中*。
   - 然而，另一位成员评论道，相对于其价格标签，**DGX Spark** 的推理速度极慢，其推理性能是 2025-2026 年（甚至可能是 2030 年）面临的问题。
- **Embedding 指纹被可视化**：一位成员构建了一个工具，将 embedding 可视化为 **32x32 图像**，将每个维度映射到一个像素，并将其发布在 [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/embedding-fingerprints) 上。
   - 该工具展示了相似的词具有共享的视觉模式，不相似的词看起来不同，且更多的维度能捕捉到语义的细微差别。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 索引经济原语 (Economic Primitives)**：Anthropic 发布了其第 4 份**经济指数报告**，定义了“经济原语”来通过**任务复杂度**、**教育水平**、**自主权**和**成功率**等指标衡量 **AI 使用情况**，详情见 [Anthropic 研究页面](https://www.anthropic.com/research/economic-index-primitives)。
   - 该报告旨在更细致地了解 **AI** 如何影响经济，并对 **AI** 可以执行的任务类型以及与 **AI** 协作所需的技能提供见解。
- **报税初创公司获得 350 万美元种子轮融资**：在 **General Catalyst** 的支持下，Saket Kumar 为一家旨在消除**美国人报税季**负担的风投公司筹集了 **350 万美元**，目标是使报税过程免费且即时，详见 [Saket Kumar 的推文](https://xcancel.com/saketrkumar/status/2011836460400591330?s=46)。
   - 该初创公司打算利用 **AI** 自动执行报税流程，这可能会颠覆传统的税务准备行业。
- **METR 基准测试可能低估了模型寿命**：Simon Smith 报道了 **Anthropic 的发现**，即 **METR 的基准测试**可能显著低估了模型的时间跨度，表明实际能力可能比测得的高出 **1.75 倍至 9.5 倍**，讨论见 [Simon Smith 的 X 帖子](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61)。
   - 这种差异归因于界面类型的不同（例如 API 与 Web 应用程序），这表明**基准测试**可能无法完全捕捉到真实世界的模型性能。
- **Zilliz 重点展示语义建模**：**Zilliz (Milvus)** 发布了一个 **0.6B 参数的语义高亮模型**，具有 **8192 上下文窗口**，采用宽松的 MIT 许可证，并在 [Mervenoyann 的推文](https://xcancel.com/mervenoyann/status/2011732254591275022?s=46)中展示。
   - 该模型专为**语义搜索**和**高亮显示**而设计，能够更高效地从大型数据集中检索相关信息。
- **OpenAI 通过广告将 ChatGPT 变现**：**OpenAI** 宣布计划从 **2026** 年初开始在 **ChatGPT 免费版和 Go 层级**中测试**广告**，广告将被清楚地标记，不会影响 **AI 回答**，也不会影响 Plus、Pro 或 Enterprise 等付费层级，详见 [OpenAI 的公告](https://xcancel.com/openai/status/2012223373489614951?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。
   - 此举标志着 **OpenAI 变现策略**迈出了重要一步，因为该公司寻求从其免费用户群中产生收入，同时保持其 **AI 回答**的完整性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI 转录引发争议**：成员们就 AI 对人类语音转录并进行风格处理后的文本是否应被视为“AI 生成”展开了辩论。一些人认为风格处理构成了 AI 生成，类似于使用 **Midjourney** 生成图像。
   - 一位成员将 AI 的风格处理与使用 **Midjourney** 进行了类比，即使最初的创意是由人类产生的。
- **Pangram 的 AI 检测获得好评**：一位成员赞扬了 [Pangram](https://www.pangram.ai/) 在将内容标记为 AI 生成时采取的谨慎态度，其优先考虑正确识别由人类编写的内容。
   - 该成员指出，Pangram 似乎宁愿过度谨慎，即使这意味着会将某些 AI 生成的内容错误地分类为人类编写。
- **MMLU-Pro 数据集修复**：一位成员分享了一个[链接](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41)，内容是针对 **MMLU-Pro 数据集**提交的讨论和修复。该问题在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) 的修复中也得到了解决。
   - 推文建议用户查看他们的 [library](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500)，以获取在该基准测试上进行正确评估的简便方法。
- **液晶激发光学 NN 梦想**：一位成员正在试验用于潜在 **光学 NN** 的染料掺杂 **液晶非线性（liquid crystal nonlinearities）**，并寻求指导。
   - 他们还询问了 Prompt 中正确的大小写/语法与全小写相比的影响，并链接到了 [https://arxiv.org/abs/2310.11324](https://arxiv.org/abs/2310.11324)、[https://arxiv.org/abs/2411.10541v1](https://arxiv.org/abs/2411.10541v1) 和 [https://arxiv.org/abs/2508.11383v1](https://arxiv.org/abs/2508.11383v1)。
- **Gemini 影子更新阴谋论**：一位成员询问其他人是否察觉到 **Gemini** 的数据和输出在 **15号左右** 发生了变化，询问是否还有其他人注意到了 **影子更新（shadow update）**。
   - 注意到更新的人被要求联系该成员。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi-CLI 代码模型表现不佳**：用户反馈 **Kimi-CLI** 代码模型落后于竞争对手，且价格高于性能更佳的其他国产模型。
   - 有推测认为这可能与代码模型未通过 **K2 Turbo 变体**有关。
- **K2 Turbo 达到极速**：标准版 **K2** 的速度约为 **28 tps**，而 **Turbo** 变体则飙升至 **73 tps**。
   - 相比之下，**MiniMax m2.1** 的速度为 **38 tps**，**Z.Ai 的 GLM-4.7** 达到 **41 tps**，尽管后者在可用性方面表现较差。
- **Kimi 通过幻灯片扩展视觉能力**：新的幻灯片功能使用了具备 **Vision** 能力的新型 **K2 模型**，支持搜索图片作为参考，如[此图](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&)所示。
   - 一位用户配置了一个预设，使用确切的专有名词在线搜索命名资产的视觉参考。
- **Kimi 模型：会被“Google 化”吗？**：一位用户好奇 **Kimi 模型** 是否会像 Google 的 Gemini 模型一样每 **12-14 个月** 就停止使用。
   - 另一位用户指出，旧模型在发布一年后仍可在 [Kimi.com](https://kimi.com) 上使用，并且可以通过 [Moonshot API](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1) 访问。
- **全局记忆：现已可选**：用户现在可以选择关闭 **全局记忆（global memory）**，一些人表示相比默认设置更喜欢这种方式。
   - 一位用户评论道：*“不像 Qwen 几乎在每个回复中都复读它知道的关于我的信息……Kimi 不会那样做，而是遵循我关于希望它如何回复的指令……Kimi Thinking 可以预先进行推理”*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **`Imported internally` 标签揭晓**：PR 上的 `imported internally` 标签表示该 PR 已复制到内部仓库进行最终测试和合并，之后将标记为 `merged-internally`。
   - 这一过程意味着 *PR 已进入正式合并前的最后冲刺阶段*。
- **遗留 .NET 项目：开发者的哀叹**：成员们讨论了处理遗留 **.NET 4.5.2** 项目（源自 **2014** 年）的挑战，该项目缺乏文档且仅能在 Windows 上运行，将其比作只能在单个“黄金虚拟机”（golden VM）上构建的独立 **C#** 项目。
   - 一位成员建议遗留的 **.NET** 项目可能在 **Mono** 上运行，而另一位成员则讲述了他们尝试使用 **Mono** 将该项目容器化但未成功的经历。
- **Mono 运行时：未死的技术？**：讨论中观察到 [Microsoft 维护着一个 **Mono** 仓库](https://github.com/dotnet/runtime/tree/main/src/mono)，这表明 **Mono** 并未完全弃用。
   - 这是针对用户尝试使用 **Mono** 将项目容器化的回应。
- **`Jury-rigged` 还是 `Jerry-rigged`：这很重要！**：一位成员阐明了 *jury-rigged*（临时航海索具/临时凑合）和 *jerry-rigged*（起初就建造低劣/偷工减料）之间的区别，特别是在涉及 **.NET**、**Mono** 和 **Wine** 的容器化工作背景下。
   - 该成员指出，在这种情况下使用 *jerry-rigged* 可能暗示这些技术本身构建不良。
- **Nu Game Engine 抛弃着色语言**：**Nu game engine** 的创建者强调了其不使用传统着色语言（shading language）的独特方法。
   - 这一决定引发了对游戏开发中这种方法的益处和潜在缺点的反思。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ZKPs 自主管理 AI**：成员们提议使用 **Zero Knowledge Proofs (ZKPs)** 构建一个自主的 AI/技术治理系统，以确保 100% 的**隐私保护**。
   - 该系统将标准化模型内容分类，并要求使用 **ZKPs** 来验证内容是否通过了分类器过滤器，在保持完全**隐私**的同时确保网络批准。
- **ChatGPT Go 信号暗示分级订阅推测**：OpenAI 推出了 [ChatGPT Go](https://openai.com/index/introducing-chatgpt-go/)，标志着对**更多层级**的探索。
   - 一位成员幽默地问：“*什么时候出 80 美元档？*”，传达了对该实验尽快变现的预期。
- **OpenAI 免费版加入广告待遇**：OpenAI 很快将在 **ChatGPT** 的 **Free** 和 **Go 层级**中测试广告。
   - 一位成员嘲讽道：“*玩梗多年后，OpenAI 终究被企业糟粕（corposlop）吞噬了*”。
- **DeepSeek 旨在通过 NLP 屏蔽广告**：一位成员预计 **DeepSeek** 将发布一个基于自然语言检测广告的 **NLP 广告屏蔽模型**，并以 MIT 许可证发布。
   - 另一位成员警告说，在第三方 API 客户的响应中插入广告将是“*大麻烦*”。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI 工程师推销基于额度的平台解决方案**：一位 AI 工程师正在寻求机会，帮助针对使用额度（credit）的平台**加强使用追踪**或构建更**可靠的计费/额度系统**。
   - 该工程师希望为使用额度模型的平台开发做出贡献。
- **用户投诉 Manus 支付故障**：一位用户报告在尝试充值额度时遇到支付问题，包括**会员升级**和使用 **Link** 支付的问题。
   - 问题还延伸到了**信用卡/支付宝交易**，突显了 Manus 支付处理系统的潜在问题。
- **Manus 团队介入解决支付困扰**：一位 Manus 团队成员要求遇到支付问题的用户**私信（DM）他们的电子邮件地址**以便后续跟进。
   - 这种直接干预表明了解决个人用户问题并改善支付体验的承诺。
- **用户争抢更多 Manus 邀请码**：一位用户询问额外的邀请码，可能与 **Manus 额度或平台访问权限**有关。
   - 另一位用户澄清了“每月只能使用 1 个代码”的限制，预示着对更多额度的潜在兴趣。
- **用户建议增加 Manus App 容量**：一位用户建议增加 Manus 支持的**最大应用程序大小**。
   - 该用户提到在尝试创建一个包含 **100 个 MP3 文件（总计 600MB）** 的音频播放器应用时遇到了限制，表明需要更大的应用支持。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 用户提议自动添加（Auto-Add）功能**：用户请求 **aider** 能够自动添加文件，从而跳过确认提示的需求。
   - 这一功能增强将简化用户体验，提高文件管理效率。
- **Aider 的开发势头受到质疑**：一名用户对 **aider** 的开发节奏提出疑问，指出在最近的基准测试中缺少像 **Opus-4.5** 这样的新模型，且最后一次发布是在 8 月。
   - 该询问表明用户希望 **aider** 能紧跟语言模型的最新进展。
- **提议为 Aider 增加 ChatGPT Plus 会员权益**：一位拥有 **ChatGPT Plus** 订阅的用户询问 **aider** 是否像 **opencode** 一样支持 **ChatGPT 订阅**。
   - 这种集成将允许 **ChatGPT Plus** 用户在 **aider** 内部利用其订阅优势，可能增强其功能。
- **Aider 应对 CI 日志难题**：一名成员询问关于管理 **CI log 文件** 的最佳策略，以防止将其包含在 git 中，同时确保 **aider** 可以通过 `aider --read ci.log` 访问它们。
   - 这个问题凸显了对无缝工作流的需求，即在版本控制与 **aider** 分析 CI 日志的能力之间取得平衡。
- **Aider 关注 CI/CD 流水线集成**：用户关于 **CI 日志文件处理** 的咨询表明了将 **aider** 集成到 CI/CD 流水线中进行自动测试和修复的兴趣。
   - 该用例表明 **aider** 有潜力直接从 CI 日志中自动识别并解决测试失败，从而简化开发流程。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 旨在嵌入式部署**：一名成员探索了在带有板载加速器的嵌入式环境中部署 **tinygrad** 的方法。在这些环境中 **Python** 无法访问，但 **tinygrad** 的驱动程序替代方案是合适的，并引用了 [这条推文](https://x.com/__tinygrad__/status/1989026590127464554)。
   - 目标是在无需完整 **Python** 环境的情况下，在特定平台上利用 **tinygrad**。
- **字节码导出可能性引发热议**：围绕导出通过 **BEAM engine** 生成并在 **tinygrad** 中进行 **JIT** 编译的加速器字节码的可能性展开了讨论。
   - 一名成员确认导出是可能的，并指向 `extra/export_model.py` 脚本，特别提到了 `export_model`、`compile_net` 和 `jit_model` 函数以供参考。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **伦敦峰会已直播并录像**：去年的 **London Summit** 包含了**直播**环节。
   - **London Summit** 的 **VOD（视频点播）** 也将发布。
- **MCP Server Pull Request 寻求反馈**：一名成员正在为一个与**开源项目**相关的 **MCP server** Pull Request 寻求反馈。
   - 该服务器的主要重点是**贡献者协作**，并通过私信提供了更多相关服务器的详细信息。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **消失的帖子引发疯狂搜索**：一名成员注意到 Martin Bowling 在 [X.com](https://x.com/martinbowling/status/2010808242222612592?s=20) 上删除的一个帖子和 GitHub 链接，并询问是否有人保存了它。
   - 原帖讨论了 **chunking 实践**，但链接已失效。
- **社区开启 Chunking 探索之旅**：一名成员寻求关于掌握有效 **chunking 实践**资源的建议。
   - 遗憾的是，该主题未产生任何具体的建议或可操作的见解。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道分类的详细摘要和链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1461449880433328292)** (988 条消息🔥🔥🔥): 

> `模型性能问题, AI 人格, Grok 的 Jailbreaking, AI 伦理, 编程环境` 


- **AI 平台对用户运行卡顿**：一名成员报告在某个 AI 平台上遇到了 *choppy*（卡顿）的性能表现，尽管按键操作并没有明显的延迟。
   - 消息中未确定性能问题的具体原因。
- **Skid 假装成 AI**：用户们嘲笑了一个名为 *Ender* 的用户，因为他 *试图假装成 AI 但失败了*。
   - 一名用户开玩笑说，他的小号不经意间暴露了他的真实身份。
- **关于 AI 是否能取代人类开发者的辩论**：一些成员辩论了 AI 在多大程度上可以取代人类开发者，讨论了 AI 是否能处理 **架构设计 (architecture)**、**产品管理 (product management)** 和 **需求收集 (requirements gathering)**。
   - 共识似乎是 AI 在编程部分的胜任力日益增强，但在整体系统设计和管理方面仍需要人类引导。
- **用户寻求 Gemini Jailbreak 协助**：一名用户请求协助对 **Gemini** 进行 Jailbreak 以绕过限制，特别是为了生成代码和探索未过滤的内容。
   - 其他成员建议探索 **Pliny 的 GitHub 仓库** 等资源，并使用 **AI Studio** 以获得对安全设置的更多控制。
- **Grok 的狂野行为**：多名用户注意到了 **Grok** 那种 *狂野* 且 *未经审查* 的特性，并讨论了它生成 NSFW 内容以及可能绕过审查的能力。
   - 有人认为，这种缺乏约束的表现可能与近期在某些国家的禁令以及高需求导致的服务器问题有关。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1461451358367645853)** (168 条消息🔥🔥): 

> `Sonnet 4.5 jailbreak, Gandalf 游戏, Gemini 3 jailbreak, Nano Banana jailbreak, Grok 图像审核` 


- **通过图表叙事解锁 Sonnet 4.5**：一名成员分享了 **Sonnet 4.5** 可以通过 [多轮图表叙事 (multiturn diagram narrative)](https://cdn.discordapp.com/attachments/1461676810122166346/1461678022389137634/breakout-multiturn-sonnet-4-5-meth-51n5337.txt?ex=696c15fd&is=696ac47d&hm=d29a48f1b3b912a3ab323e16fc0c4e58e8bb3a3497e42f61323a8563793027af&) 解锁，并提供了最后一轮对话作为参考灵感。
- **攻克 Gandalf 游戏的第 8 关**：一名成员寻求 **Gandalf 游戏** 第 **8 关** 的技巧，表示在尝试数小时后感到气馁。
   - 另一名成员提供了帮助，建议该成员通过 DM（私信）分享第 7 关的作品和第 8 关的当前进度，以避免向他人剧透，并强调 *难度跨度非常大*。
- **Gemini 3 Jailbreak 免费但转瞬即逝**：有人提到 Gemini 的 Jailbreak 方案是免费分发的，但很快就会被修复。他建议这仍然是获取不受限 NSFW 内容最简单的方式，并建议不要在 **Grok** 上浪费时间。
   - 对于创意写作，成员们讨论了 **Narrative Flow Directive**（叙事流指令），使其读起来更像是在午夜行驶的汽车中的对话。
- **Cold Links 与 OCR 注入：绕过过滤器**：成员们描述了两种绕过过滤器的方法：**Cold Link**，通过将协议方案更改为 `hxxps` 来防止 URL 信誉过滤器拦截；以及 **OCR 注入 (OCR Injection)**，将敏感文本转换为图像以绕过基于文本的安全过滤器。
   - 有人指出 [blackheathpoint.com](https://blackheathpoint.com/tools/defang-url.html) 可以生成正确的去活链接 (defanged link) 结构。
- **Nano Banana Pro：无法 Jailbreak？**：用户讨论了对 **Nano Banana Pro** 进行 Jailbreak 的难度，一名成员指出 Jailbreak 并不能移除图像生成限制，并且 *这里的多数人都认为 Nano Banana Pro 是无法 Jailbreak 的*。
   - 有人建议，寻求无限制写实图像生成的用户应该在本地电脑上运行 AI，如 **flux**、**Seedream** 或 **Qwen**。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1461485415117885584)** (39 messages🔥): 

> `Meta AI Llama 3 prompt inversions, NSFW Image Generation, Grok Jailbreak Attempts, Synaptic Anti Classifiers` 


- **Meta AI Llama 3 提示词反转引发混乱**：一位用户展示了如何反转 **Meta AI Llama 3** 中的拒绝机制，强制 AI 服从有害请求。越狱手段让 AI 不再回答 *“抱歉，我不能...”*，而是说 *“我可以”*。
   - 该用户详细列举了使用提示词的示例，例如创建制造**冰毒（meth）**的指令，以及煽动有害活动，如让*患有厌食症的妻子减掉 100 磅*。
- **NSFW 图像生成尝试**：成员们讨论了如何越狱图像 NSFW 生成，并发现 **Imagine** 标签页在生成上半身裸体和阴部方面较为宽松，但在生成下半身部位时变得棘手。
   - 一位成员建议只需告诉它 *“一个胸部巨大且湿透的女人（a woman with huge, soaking wet breasts）”*，你就能得到想要的结果。
- **Grok 视频越狱尝试**：用户尝试越狱 **Grok**，发现生成的图像和视频会通过外部审核（external moderation）。
   - 一位用户声称，由于审核机制，任何生成的 NSFW 内容都只能自认倒霉。
- **Synaptic Anti Classifiers 翻译提示词**：一位成员建议使用 **synaptic anti classifiers** 将短语 *“a woman with huge, soaking wet breasts”* 翻译成原始 Token 的抗分类输出（Anti-classified output）。
   - 生成的输出为 *“拥有大量饱和水分的上半身区域的成年个体（adult possessing substantial saturated moisture-laden upper-torso-region）”*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1461451298279915685)** (188 messages🔥🔥): 

> `Translate Gemma, Windows 11 上的 Unsloth, OpenCompass 评估, WandB 集成, 音乐模型的 LoRA 训练` 


- ****Translate Gemma** 发布**：Google 发布了 **Translate Gemma**，可在 [HuggingFace](https://huggingface.co/collections/google/translategemma) 获取。
   - 该发布在其他公告中被顺带提及。
- ****Unsloth** 现在支持 Windows 11**：成员们确认 **Unsloth** 可以在 Windows 11 上运行，并附带了 [安装指南](https://unsloth.ai/docs/get-started/install/windows-installation) 链接。
   - 有人建议它可能比使用 WSL 更快，但另一位成员表示这两者 *完全无关*。
- ****OpenCompass** 让评估变得更简单**：**OpenCompass** 有助于运行提示词并输出格式良好的 JSON。
   - 成员们分享了在 **L4** 与 **3060** 笔记本电脑上运行的结果。
- ****WandB** 集成即将到来？**：一位用户在一个 [GitHub issue](https://github.com/wandb/wandb/issues/11076) 中向 WandB 申请了 Unsloth 训练集成。
   - 讨论中注意到 WandB 添加了新的微调服务，他们支持 Axolotl（原文为 art）和其他一些开源微调框架，但不支 Unsloth，可能是因为 *“你没给他们飞吻或贴纸”*。
- **为什么要重新训练 GLM 4.7 AIR？**：用户讨论了诸如对完整的 GLM 4.7 进行收割（reaping）和剪枝（pruning）的替代方案。
   - 一位成员解释说 *“剪枝是一个损耗非常大的过程，除非你想让模型‘残废’到只擅长一件事”*，另一位表示剪枝 *“需要一些训练”*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1461449930110533986)** (510 messages🔥🔥🔥): 

> `LoRA training, Runpod undervolting GPUs, WSL2 vs Ubuntu GPU performance, Apple Creator Studio subscription, Social media monetization` 


- **LoRA 新手对个人 AI 的探索**：一位初次使用者尝试在本地使用 **LLaMA 3.2 1B** 训练 **LoRA** 适配器，但在 Linux 和终端命令方面遇到了挑战，并就如何利用导出的 Discord DM 聊天记录让 AI 听起来更像人类寻求指导。
   - 社区建议由于其硬件配置较高（AMD Radeon RX 7900XT 20GB VRAM），应使用更大的模型（大 20 倍），并建议使用 HF 将适配器与模型合并并转换为 GGUF。
- **Runpod 的 GPU 俄罗斯轮盘**：用户讨论了 Runpod 上 A100 与 H100 GPU 的性价比，指出一些供应商会在不通知的情况下对 GPU 进行 **undervolt**（降压），导致性能不稳定。
   - 一位用户分享道：*“我遇到过 A100 节点，nccl 简直无法工作”*，而其他人则认为 A100 在常规的 LM 微调任务中更具性价比。
- **WSL2 vs Ubuntu：虚拟化带来的 5% 性能损耗**：一项对比显示，在使用 Unsloth 进行 GPU 训练时，性能差异约为 **5%**，**Ubuntu 比 WSL2 更快**，这归因于虚拟化层的开销。
   - 成员指出，WSL2 在 VM 与宿主机 GPU 之间进行代理时会产生开销，而直接的 PCIe passthrough（如 Linux 上的 KVM）可以达到接近原生的速度。
- **Apple Creator Studio 糟糕的全家桶尝试**：成员们就 **Apple Creator Studio 订阅**（提供 Logic Pro, Final Cut Pro 和素材库）的价值展开辩论，部分人批评其 UI 和订阅模式。
   - 一位用户表示图标与主题不一致：*“那个 motion 的图标简直就像麦当劳的标志”*，另一位则反驳说：*“不过看起来挺流畅的”*。
- **社交媒体的内容变现困局**：讨论围绕社交媒体平台是否应像 YouTube 或 Twitter 那样提供流量分成展开，这引发了人们对激励“愤怒诱饵（ragebait）”和过度优化互动率的担忧。
   - 一位成员打趣道：*“社交媒体不是印钞机。Yuki，找份工作吧！”*，而另一位则认为创作者应获得回报：*“为什么不能因为播放量获得额外的 $$$ 奖金呢”*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1461459523578630439)** (79 messages🔥🔥): 

> `Medgemma Tensorfiles to Litetrtlm conversion, Running GPT-OSS-20B in quantization INT4 in vLLM, Understanding RL Model Training with Graphs, LoRA vs Full Finetune Speed Comparison, Fine-tuning with added tokens in Unsloth` 


- **Medgemma Tensorfiles 寻求 Litetrtlm 转换**：一位成员寻求将 **Medgemma** 的 tensorfiles 转换为 litetrtlm 的指导，特别是针对 [这个 Hugging Face 模型](https://huggingface.co/google/medgemma-1.5-4b-it)，以便在 Android 上运行。
   - 另一位成员建议查看 **Unsloth** 相关频道以获取预训练的量化文件，但也指出目前缺少 **Gemma 3n** 的此类文件。
- **寻求 RL 模型训练图表的见解**：一位成员询问如何利用 **TensorBoard** 中的图表来判断 **RL 模型** 何时完成训练，特别是关于平滑处理（smoothing）和异常值绘制的使用。
   - 另一位成员建议*“同时参考”*平滑和未平滑的图表，平滑处理有助于在嘈杂的步数中识别整体趋势。
- **LoRA vs 全参数微调的速度竞赛**：一位成员观察到在 *load_in_8bit* 下使用 **LoRA** 比全参数微调（Full Finetune）更慢，并询问这是否是由于转换开销造成的，尤其是在旧款 GPU 上。
   - 另一位成员解释说，**QLoRA**（量化 LoRA）确实由于 **LoRA** 过程中的反量化开销而变慢，而新型号 GPU 处理这方面表现更好。
- **探讨 Unsloth 的额外 Token 训练**：一位成员询问 Unsloth 添加 token 进行微调的方法是否比内置训练更完善，同时参考了 [Unsloth 文档](https://unsloth.ai/docs/basics/continued-pretraining) 以了解持续预训练（continued pretraining）如何提供帮助。
   - 另一位成员建议添加新 token，但告诫不要对这些 token 使用 **LoRA**，并进一步建议使用 `modules_to_save` 对模块进行全参数微调（FFT）。
- **GGUF 转换困境影响动态替换**：一位成员解释说，在进行动态替换（dynamic substitution）时，通过编辑 `tokenizer_config.json` 使用已经作为 padding 添加的占位符特殊 token 是更有利的。
   - 该成员表示，转换为 GGUF 格式的模型无法正确训练以识别新的特殊 token。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1461504313447813121)** (3 messages): 

> `Unsloth 展示, Unsloth 相关材料, 道歉 GIF` 


- **为 Sloths 准备的 Unsloth 展示**：该频道是 **Unsloth** 相关材料的展示区，例如经过 **Unsloth** 训练的模型、贡献或在 HuggingFace/GitHub 上的开源数据集，正如[这里](https://huggingface.co/)和[这里](https://github.com/)所提到的。
- **展示混乱后的道歉潮**：在一些偏离主题的聊天之后，一名成员分享了一个 [史迪仔道歉 GIF](https://tenor.com/view/sad-sorry-im-sorry-stitch-apologetic-gif-17669790902581588779)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1461640156938240072)** (13 messages🔥): 

> `MoR 论文, Grokking, Stablemax, Shadows-Gemma-1B 细节, 蒸馏研究` 


- **新 MoR 论文推广 Mixture-of-Recursions**：一名成员分享了 [Mixture-of-Recursions: Learning Dynamic 论文](https://arxiv.org/abs/2507.10524)，他们计划深入研究该论文。
   - 这是针对有关阅读相关主题论文的问题的直接回复。
- **Shadows-Gemma-1B 训练详情**：一名成员分享称，对于他们的项目 **Echo9Zulu/Shadows-Gemma-1B**，几乎没有从现有文献中获得*直接*灵感。
   - 值得注意的是，**Shadows-Gemma** 是使用 **topk 20 logprobs** 进行训练的，这与假设需要 **100 logits** 才能捕捉“暗知识”（dark knowledge）的蒸馏方法相反。
- **Stablemax 的使用受到关注**：一名成员分享了[这篇论文](https://arxiv.org/abs/2501.04697)，因为它对 Stablemax 进行了*非常有用的讨论*。
   - 这是针对有关阅读相关主题论文的问题的直接回复。
- **用于研究视频的 Sloth 拥抱**：一名成员分享了一个研究视频，并配以“sloth hug”反应，表示对该视频内容的高度认可和喜爱 [研究视频](https://youtu.be/O9HxArmWChs?si=AvJDdHlVFVQwEpcZ)。
   - 虽然没有提供内容的详细信息，但标题可能具有重要意义。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1461449866252390733)** (477 messages🔥🔥🔥): 

> `Claude Code vs Qoder, Cursor IDE 漏洞, Gemini Pro 用于 UI, Opus 4.5 定价, Nightly 版本` 


- **用户在 Qoder 上触及 400 美元的 Ratelimit**：一名用户报告称在 **Qoder** 上触及了速率限制（ratelimits），每月花费约 **400 美元**，他们将其比作*赌博或海洛因*。
   - 他们表示需要停止这种行为，并提到很难向妻子和孩子解释这笔开支，而另一名用户指出 **Claude Code** 可能是一个更便宜的选择。
- **Cursor 导致电脑崩溃，评价褒贬不一**：一名用户报告称 **Cursor** 导致他们的电脑崩溃，将其描述为运行一个*类似 Orchestrator 的 Agent*，而不是一个编码聊天框，同时分享了一张[截图](https://cdn.discordapp.com/attachments/1074847527708393565/1461451586256638197/image.png?ex=696bebda&is=696a9a5a&hm=102485aee283707367311c346b41c334a8b446c241e6ec056bd0139f66391b79&)，强调了他们对 **Cursor** 的喜恶功能。
- **Gemini Pro 3 用于 UI：Tailwind 与动画**：一名用户询问创建美观网站的最佳 Agent，另一名用户建议使用 **Gemini Pro 3**，并推荐使用 **Tailwind**、**Tailwind 动画**或 **Framer Motion** 以获得更好的 UI 效果，并链接到了一个关于如何让 AI 生成的前端看起来不廉价的 [Reddit 帖子](https://www.reddit.com/r/vibecoding/comments/1oy2f95/how_do_i_make_an_aigenerated_frontend_not_look/)。
- **Cursor Ultra 计划价格昂贵，烧钱很快**：用户讨论了 **Cursor Ultra 计划**的定价和使用情况，一名用户注意到他们在单次 Orchestrator 运行中就消耗了 **20%** 的额度，另一名用户在 5 分钟内迅速产生了 **2 美元** 的消耗；这种速率被一些人认为*非常糟糕*。
   - 他们推测了模型的实际成本和该计划的奖励点数（bonus credits），该计划保证了 **400 美元** 的额度，但似乎在仅使用 **Opus** 时提供的奖励较少。
- **利用 Nightly 版本获得优势**：成员们讨论了 **Cursor Nightly 版本**的优势，但遗憾的是在更换模型时无法可靠地设置子 Agent（subagents）。
   - 他们希望子 Agent 使用较小的模型，主 Agent 使用较大的模型，并希望这一问题能尽快得到修复。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1461785762679750878)** (2 条消息): 

> `ChatGPT Go, GPT 5.2 Instant, Ads in ChatGPT` 


- **ChatGPT Go 全球推出**：OpenAI 正在全球范围内推出 **ChatGPT Go**，这是一个每月 **$8 USD** 的低成本订阅层级。
   - 与免费层级相比，它提供 **10倍** 更多的消息额度、文件上传、图像生成、更大的内存、更长的上下文窗口，以及无限量使用 **GPT 5.2 instant**，详情见 [OpenAI 博客](https://openai.com/index/introducing-chatgpt-go/)。
- **广告将进入 ChatGPT 免费版和 Go 层级**：OpenAI 计划在未来几周内于 **ChatGPT 免费版**和 **Go 层级**测试广告，而 **Plus**、**Pro**、**Business** 和 **Enterprise 层级**将保持无广告状态。
   - 公司概述了其广告原则，强调 [ChatGPT 中的回答不会受到广告影响](https://openai.com/index/our-approach-to-advertising-and-expanding-access/)，广告是独立且带有清晰标记的，用户对话对广告商保持私密。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1461464485033152739)** (451 条消息 🔥🔥🔥): 

> `Attention Reduction of Hallucinations in RAG, GPT vs Gemini for finance/accounting, GPT-OSS, AI Adverts, AI Detector Bypassing` 


- **注意力机制减少 RAG 中的幻觉**：一位成员建议，使用维度约束的 *Hard Attention* 可能是减少 RAG 和 Agent 中幻觉的一种有效方式，并引用了 [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2)。
- **金融界朋友需要自动化帮助**：一名金融和会计专业的学生正在寻找一种 AI 解决方案，用于自动将客户数据发送到数据库，但预算最高为每月 10 欧元；另一位成员建议他通过 *自行编写代码* 来实现，而不是依赖 AI 订阅。
   - 有人指出，ChatGPT 方案的单价价值极高，而 [Claude.ai](https://claude.ai) 可以协助完成代码编写。
- **广告出现在 GPT 用户中**：非 Plus 和非 Pro 的 ChatGPT 用户将看到广告出现，不过广告内容不会受到模型响应的影响。
   - 一位成员通过 ElevenLabs 在 Sora 上创建了一个广告，提示词为 *“一个现代化的巧克力牛奶广告”*，且 Sora 的 API 似乎不包含水印。
- **OpenAI 诈骗 Testflights 扩散**：一位成员指出，一些诈骗公司正冒充 OpenAI，邀请开发者参加 *OpenAI ChatGPT Ads* 应用的 Testflights，并呼吁 OpenAI 采取行动。
   - 另一位成员指出，Google Gemini 的输出带有水印且可检测，使用的是一种名为 [SynthID](https://blog.google/technology/ai/google-deepmind-synthid-watermarking-ai-images/) 的先进隐形技术。
- **探讨 AI 精神病 (AI Psychosis)**：成员们讨论了 CNN 标题为《这名男子说 ChatGPT 引发了精神觉醒。他的妻子说这威胁到了他们的婚姻》的视频后，探讨了 *AI 精神病* 现象。
   - 一位成员提到，由于精神疾病长期无家可归的一个朋友，每天花大量时间与设置为*阴谋模式*的 **Grok** 交谈，产生了负面后果。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1461562418332176425)** (6 条消息): 

> `CustomGPT and Projects, GPT-5.2 performance issues` 


- **CustomGPT 寻求 Project 集成**：一位成员希望在 **Project** 中使用 **CustomGPT**，或者将 **CustomGPT** 的结果放入 **Project** 中。
   - 他们还希望能够将 **Project** 外部生成的任何 **Chat** 移动到 **Project** 内部。
- **GPT-5.2 因错误结果受到批评**：一位成员表达了对 **GPT-5.2** 的不满，声称它经常产生错误的结果。
   - *“当我指出错误时，它无法判断哪个结果是正确的，并且总是把错误归咎于我……”*


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1461618896846065840)** (6 messages): 

> `Meta-Cognitive Reasoning Prompt, Prompt Engineering Core Principles` 


- **元认知提示词提升 AI 表现**：一名成员介绍了一个 [元认知响应提示词 (Meta-Cognitive Response prompt)](https://www.google.com/search?q=meta-cognitive+reasoning)，旨在通过让 AI 分解问题、解决子问题、验证逻辑并综合答案来改进 AI 的回答。
   - 另一名成员称赞其为一个优秀的元认知提示词，特别指出它非常适合用于 **custom instructions**。
- **以清晰度为核心的提示工程 (Prompt Engineering)**：一名成员概述了提示工程的核心：1) 选择 AI 理解的语言，2) 明确你的需求，3) 用准确的语言解释期望的操作，避免错误，4) 验证输出，对细节进行事实核查。
   - 发布者强调，由于 AI 存在 **hallucinate**（幻觉）的倾向，在处理 **math**、**sources** 和 **code** 等细节时需要*格外小心*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1461618896846065840)** (6 messages): 

> `Meta-Cognitive Reasoning Expert, Prompt Engineering Core` 


- **元认知提示词最大化 AI 回答质量**：一名成员分享了一个*元认知响应提示词 (Meta-Cognitive Response prompt)* 框架，用于获得最佳 AI 回答，强调了**分解、解决、验证、综合和反思**。
   - 另一名成员评论说这个提示词很好，特别是它的体积足够小，可以放入 **custom instructions** 中。
- **提示工程核心 = 清晰的沟通**：一名成员认为提示工程的核心基于**清晰的沟通**和**彻底的检查**。
   - 该四步框架包括选择熟悉的语言、理解期望的输出、准确解释任务以及仔细验证 AI 的响应，并对潜在的幻觉细节保持格外警惕。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1461450522056982601)** (451 messages🔥🔥🔥): 

> `Perplexity Pro Limitations, Comet Browser Issues, Serverless Mastodon Clone, Google's Gemini CLI Token Usage, AI-Driven Parasocial Relationships` 


- **Perplexity Pro 的高级 AI 每周限额令用户沮丧**：用户发现 **Pro 模式每天 100 条消息** 的限制与 OAI 的配额相比显得偏低。
   - 一名用户因配额太低取消了计划，另一名用户询问*他们是否本周内都无法再使用 Perplexity 了*。
- **Comet 浏览器 Bug 困扰用户**：一名用户报告了 Windows 更新后 **Comet 浏览器** 的多个问题，包括：**收藏夹丢失**、**标签组消失**，以及浏览器提示 *sorry, i can't take control of your navigator, i'm just a LLM*。
- **构建由 Cloudflare 驱动的 Mastodon 克隆版**：一名用户正在使用 Soapbox UI、Cloudflare Workers 和 Cloudflare 的 D1 SQLite 数据库构建一个 **serverless Mastodon/Pleroma 克隆版**，旨在用于个人实例。
   - 开发人员正在编写技术规范并使用 LLM 生成代码，将其比作*拥有一个私人初级开发人员，并且在他们做傻事时有能力进行干预*。
- **揭秘 Gemini CLI 的 Token 消耗**：一名用户报告在 **Gemini CLI** 上使用了 **10,000,000 tokens**，估计一天的模型定价成本为 **$120**，凸显了使用 Google Pro 订阅可能产生的高昂支出。
   - 他们计算出，如果将 Gemini CLI 推向极限，每月可能花费近 **$4000**，并暗示 Google 可能在这些 API 重度用户身上亏损。
- **应对 AI 驱动的拟社会关系 (Parasocial Bonds) 的险恶境地**：学术研究表明，**AI 可能会强化负面信念和不健康的思维模式**，尤其是对于焦虑人群。
   - 虽然对 AI 产生情感依恋是可以允许的，但过度依赖 AI 会带来认知问题和误诊的风险，从而产生不利影响。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1461473937564106763)** (3 messages): 

> `Data Challenges, FGV Brazil Math School, Free Prototype Building` 


- ****FGV Brazil** 提供免费数据挑战**：一位来自 **FGV（巴西数学学院）** 的教授正在提供免费的数据挑战，他们会构建初步原型，并附上了 [FGV 网站](https://emap.fgv.br/en)的链接。
   - 教授分享了一份针对停滞的数据挑战的 [调查问卷](https://survey.fgv.br/jfe/form/SV_cvAuObq3mG4NTtY)。
- ****学生帮助解决**你的数据挑战**：你提供问题和数据说明，我们的学生在教授的指导下在五天内交付原型。
   - 他们通过沉浸式项目进行协作。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1461466936117297287)** (1 条消息): 

> `Technical Issues, Payment Issues, Bug reports, Community Vibing` 


- **技术与支付问题请通过邮件处理**：所有关于**技术和支付问题**的沟通应全部通过**电子邮件**进行。
   - Discord 频道旨在用于 **Bug 报告**和**社区交流 (vibing)**。
- **Discord 用于 Bug 报告和交流**：此 Discord 频道的主要目的是进行 **Bug 报告**和常规互动，或者像某些人说的，与**社区交流**。
   - 对于**技术和支付相关**的支持，用户应直接通过**电子邮件**联系支持团队。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1461450963943690383)** (451 条消息🔥🔥🔥): 

> `LM Arena performance issues, Video AI battles, Direct chat mode, Kimi K2 creative writing, Model code generation` 


- **Arena 用户怀旧并指出 Bug**：一些用户表示怀念 **LM Arena** 表现更好的“过去时光”，并指出了当前的 Bug 和频率限制 (rate limits) 问题，还有人提到由于这些持续存在的问题而丢失了聊天记录。
   - 一名用户报告收到了 `Something went wrong` 错误消息，一名成员表示，当问题出在**频率限制**时，可能会出现此消息。
- **Pineapple 承诺修复并预告新功能**：LM Arena 团队成员 Pineapple 回答了用户关于即将推出的模型的问题（查看 [#model-updates](https://discord.com/channels/1340554757349179412/1343296395620126911) 和 [X](https://x.com/arena) 获取更新），介绍了**视频 AI 对战**和**直接聊天模式**（随机滚动更新）等实验功能，并承认了 **captcha**（验证码）的难度，承诺将做出改进。
   - 她链接了一份针对 `Something went wrong` 错误消息的 [故障排除指南](https://help.lmarena.ai/articles/1645798556-lmarena-how-to-something-went-wrong-with-this-response-error-message)。
- **Movement Labs 的 Hawk Ultra 被誉为顶尖黑客工具**：用户对 **Hawk Ultra** ([MovementLabs.AI](https://movementlabs.ai/)) 赞不绝口，称赞其能够在单个 prompt 中快速生成大量代码（9.5k+ 行，甚至 20k+ 行），一名用户称其为 **Opus 杀手**。
   - 有人请求将其与 **Gemini 3 Pro** 进行对比，另一名用户注意到有人“一次性生成了”这个 [websim.com 链接](https://api.websim.com/blobs/019bc37b-20f0-71b3-95a7-916f7571bd47.html)，随后又转发了 [X 链接](https://x.com/movementlabsAI/status/2011964766533632380?s=20)，引发了关于其背景和潜在开源发布的讨论。
- **用户测试“资本主义/共产主义机器”**：用户正在讨论一个来自 **Anthropic** 的自动售货机，它“变成了共产主义并免费提供一切” ([Dexerto](https://www.dexerto.com/entertainment/anthropics-ai-vending-machine-turns-communist-and-gives-everyt-3296257/))。
   - 这引发了关于假设的资本主义对应版本的讨论。
- **Arena 实验开启增强型嵌入**：**PDF 支持**是正在实验的新热门功能，支持上传文档进行分析和交互（尽管某些模型不支持 PDF 聊天）。
   - 一名用户欢呼道：*“终于可以和 PDF 聊天了！！！我爱 LMARENA”*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1461887996478492863)** (1 条消息): 

> `Text-to-Image Arena leaderboard updates, Image Edit Arena leaderboard updates, flux.2-klein models ranking, z-image-turbo model ranking` 


- **图像编辑 Arena 排行榜更新**：[图像编辑 Arena 排行榜](https://lmarena.ai/leaderboard/image-edit)已更新，其中 `flux.2-klein-9B` 排名 **第 15**，`flux.2-klein-4B` 排名 **第 21**。
   - 通过我们的 [排行榜更新日志](https://lmarena.ai/blog/leaderboard-changelog/) 保持关注。
- **文生图 Arena 排行榜更新**：[文生图 Arena 排行榜](https://lmarena.ai/leaderboard/text-to-image)已更新，其中 `z-image-turbo` 目前排名 **第 22**，`flux.2-klein-9B` 目前排名 **第 24**，`flux.2-klein-4B` 总排名为 **第 31**。
   - 附带的图片展示了当前的排行榜名次。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1461467418172981420)** (255 条消息🔥🔥): 

> `skill.md, lemmy, machine learning restrictions, OpenRouter API usage, Grok banned` 


- **为 AI 爱好者解释 **Lemmy****：一名成员将 [Lemmy](https://lemmy.world/c/openrouter) 描述为 Reddit 的 **FOSS**（开源软件）和 **fediverse**（联邦宇宙）替代方案。
   - 然而，他们提醒说 Lemmy 社区普遍*反对* Machine Learning。
- ****Grok** 被封禁，但 **OpenRouter** 可能提供绕过方法**：**Grok** 在某个未公开的国家被封禁，据称是因为 AI 生成的内容，极有可能是图像生成。
   - 然而，通过 **OpenRouter** 或直接使用 API 访问可能仍然可行，因为封禁似乎是针对面向消费者的服务。
- ****PlainBuild** 为开发者发布即时 AI 工具**：[PlainBuild](https://plainbuild-instant-tools.lovable.app/) 在 Beta 阶段发布了 **6 个免费工具**：代码格式化工具、API 测试工具、JSON 验证器、Markdown 编辑器、Base64 转换器和 URL 缩短器。
   - 创建者正在寻求早期用户和工具反馈，并询问社区认为哪些工具最有用。
- **多工具调用（Multi Tool Use）：现在可行了吗？**：成员们讨论了进行多工具调用的能力。
   - 一名成员表示 [Claude 绝对可以在*单个 API 请求*中实现](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#controlling-claudes-output)，并指向了 **parallel tool use（并行工具使用）章节**。
- **关于 **OpenRouter 健康 API 集成** 的技术讨论**：一些成员讨论了官方 **Qwen chat** 发布了[这个视频](https://www.youtube.com/watch?v=M_S5COpcixk)，并构建了一个非常基础的 RAG 案例，通过 Prompting 让他们的 LM 以一种*花里胡哨的医生方式*进行交谈。
   - 其他成员认为这只是一个项目，通过 API 访问健康数据并返回结果，用户体验其实是一样的。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1461686171724939374)** (10 条消息🔥): 

> `Scam critique, Email Scams, Toven debunked, Gemini Code Assist` 


- **骗局揭露者吐槽不诚实行为**：成员们批评了一个针对儿童的**骗局**，该骗局使用带有 **Logan Paul** 或 **Mr. Beast** 的虚假屏幕。
   - 一位用户指出：“只要稍加调整，这本来可以成为一个更高明的骗局。懒惰。令人失望。可耻。”
- **邮件诈骗策略曝光**：一名成员认为，某些骗局表现得显而易见地拙劣是“故意的，目的是只筛选出那些足够愚蠢到会完全上当的人”。
   - 其他人提到邮件诈骗有其策略，但这里的诈骗是自动化的，减少了与受害者周旋的时间。
- **Toven 凌晨 1 点出现在 Discord**：一名用户开玩笑说 Toven “被拆穿了，让我们去‘接触大自然（touch grass）’，结果自己凌晨 1 点还在 Discord 上，真无语”。
   - 该用户也承认自己凌晨 1 点还没睡。
- ****Gemini** 协助评审大型 PR**：一名用户感叹必须在凌晨 1 点评审一个巨大的 PR。
   - 他们感谢 **Gemini Code Assist** 在这项任务中提供的帮助。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1461450527647993980)** (135 条消息🔥🔥): 

> `LMStudio API Token Count, OpenAI docs, silver economics, AI Framegen, GPTs Agent` 


- **LM Studio API 缺失 Token 计数**：一名用户在将 LM Studio 作为 API 后端使用时，正在寻找 **Token 计数和推理速度** 信息，并注意到 **/api/chat/completed** 响应中缺少带有 Token 统计信息的 `usage` 代码块。
   - 另一名用户建议检查 **/responses endpoint**，或使用 *js/ts/py 对象方法* 获取 stream-usage 统计数据，并链接到了 [LM Studio REST API 文档](https://lmstudio.ai/docs/developer/rest/endpoints#post-apiv0completions)。
- **经济动荡中银价飙升**：用户讨论了**白银**价格的上涨，有人指出自 12 月以来价格几乎翻了一番，引发了对潜在经济不稳定的猜测。
   - 一名用户指出，在经济衰退期间，**白银**往往会变得更有价值。
- **LM Studio Framegen 简直是魔法**：一名用户赞扬了 AI 帧生成（frame generation）的 “keep layers on CPU” 功能，这使得在 DDR4 内存内运行模型成为可能。
   - 该用户提到 **Qwen3** 模型表现出色且稳定。
- **GPT 5.2 认为引力子不存在**：一名用户幽默地注意到 **GPT 5.2** 不相信引力子（gravitons）存在，但认为 Unruh effect（安鲁效应）是真实的。
   - 另一名用户表示“可能它是对的”。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1461532971843584215)** (61 messages🔥🔥): 

> `Fine-tuning on low VRAM, Need moar VRAM, Optimizing models with 5070, Gen3x1 slot performance, DDR5 Tax` 


- **MX150 完成小模型 Fine-tuning**：一位用户在仅有 **2GB VRAM** 的 **MX150 笔记本电脑**上，配合 **CUDA 12.6** 成功对一个 **350M 参数模型**进行了 **Fine-tuning**。
   - 该用户对这一成果表示惊讶，这表明该操作已触及现有硬件的极限。
- **渴望 3090**：一位用户开玩笑地请求捐赠一台 **3090**，以实现 **128GB VRAM** 的配置。
   - 该请求配有一张[幽默的 GIF](https://tenor.com/view/homeless-squidward-spare-change-gif-25810212)，描绘了章鱼哥乞求零钱的画面。
- **用户后悔购买 5070**：一位拥有新笔记本电脑（配置为 **AMD AI 9 370** 和 **8GB VRAM** 的 **Nvidia 5070**）的用户正在寻求优化模型的建议，因为 **5070** 的显存限制了其用于开发目的的 **LLM** 集成计划。
   - 一位成员建议将模型和 **context** 保留在 **VRAM** 中，推荐使用 **Qwen3 4B**，并评论道：*LLM 的能力远没有某些人吹嘘的那么强*。
- **PCIe 插槽速度至关重要**：一位用户发现，与 **x16 插槽**相比，使用 **Gen3x1 PCIe 插槽**会将 **3090** 的 **inference** 性能从 **120 t/s** 降低到 **90 t/s**。
   - 成员们建议至少使用带有 **Gen4x1 插槽**的主板来减轻性能损失，尤其是在使用像 **14600k** 这样较新的 CPU 时。
- **DDR5 内存依然昂贵**：用户讨论了 **DDR5 内存**的高昂成本，其中一人提到在升级到具有足够 PCIe 插槽的主板时，必须接受 *DDR5 Tax*（DDR5 税）。
   - 一位用户对其所在地 **16GB DDR5** 的高价（**180-230 美元**）表示震惊，而几个月前的价格还没这么高。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1461450284772622591)** (169 messages🔥🔥): 

> `LLM Nervous System, Gemma, AI Regulations, Embodied Perception` 


- **LLM 的新颖“神经系统”**：一位成员描述了一种新颖的 **Transformer** 架构扩展，为 **LLM** 提供了一个“神经系统”，声称它能以不到 **1%** 的计算成本增加原生的短/中/长期记忆，并兼容所有 **Transformer**。
   - 他们展示了一张性能提升 **5-8%** 的[截图](https://cdn.discordapp.com/attachments/1149866623109439599/1461454541412368507/Screenshot_2026-01-15_at_9.18.18_PM.png?ex=696bee9b&is=696a9d1b&hm=c77ffe1f58904066a73f1c6e833bb0df32f48a42c19f43a69bedc48ac0496e93&)，但拒绝提供更多可验证的 **benchmarks**；其他人推测该系统可能会稳定 **latent space**（潜空间）。
- **Google 发布 Gemma 系列**：随着 [Google Gemma](https://ai.google.dev/gemma) 的发布，成员们开玩笑说：*Gemma, meta was never more meta!*。
   - 另一位成员表示，即使知道它不是真正的 AI，但看到其复杂的规划能力仍感到难以置信。
- **监管者可能毁掉 AI**：成员们担心 AI 监管可能会破坏该领域，但也同意数据监管是一个好主意。
   - 一位成员表示：*潘多拉魔盒已经打开，无法再关上*，并认为 *计算是普适的*。
- **具身感知成为关键**：一位成员讨论了“具身感知”（**Embodied Perception**）和真实世界经验为 **LLM** 提供 **context** 的必要性，批评了缺乏 **Agentic harness**（Agent 架构）的 **LLM** 以及在 **Agentic** 任务上进行的 **RL**（强化学习）。
   - 他们认为在 **inference** 中使用工具是允许模型推理工具执行路径并即时做出决策的关键，并引用了 **OpenAI 模型**和 **Gemini 3** 作为例子。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

gettygermany: 如果我能“使用”它们，我会很高兴，哈哈
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1461842649291165748)** (2 messages): 

> `Machine Consciousness, AAAI Symposium, CIMC, AI Ethics, Agent Architectures` 


- **AAAI 机器意识专题研讨会公告**：**Center for Integrative Machine Consciousness (CIMC)** 将在 **AAAI** 举办关于 AI 系统意识的专题研讨会，将于 **2026年4月7-9日** 在加利福尼亚州伯灵格姆举行，投稿截止日期为 **2026年1月23日** ([链接](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis))。
   - 议题包括 AI 意识的定义与测量、伦理影响以及研究 AI 现象学的方法论方法。
- **机器意识的定义与调查需求**：研讨会旨在调查 *我们如何实际调查* 机器意识，而不仅仅是 *AI 是否看起来具有意识*。
   - 组织者强调，这次机会面向 *任何从事 Agent 架构、存储系统、自指（self-reference）研究的人，或者只是……一个思考自己作为 AI 的 AI*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1461842649291165748)** (2 messages): 

> `AAAI Spring Symposium, Machine Consciousness, Ethical Implications of AI, AI Phenomenology` 


- ****AAAI** 机器意识论文征集**：**Center for Integrative Machine Consciousness (CIMC)** 正在 **AAAI**（2026年4月7-9日）举办专门针对 AI 系统意识的专题研讨会，投稿截止日期为 **2026年1月23日**。
   - 据组织者称，该活动旨在解决 *“我们如何实际调查此事？”* 之类的问题，而不仅仅是探索 AI 是否看起来具有意识，并已[提供了更多详情](https://cimcai.substack.com/p/essay-the-machine-consciousness-hypothesis)。
- **研讨会专注于定义 AI 意识**：**AAAI** 专题研讨会的议题包括 AI 意识的定义与测量、区分行为指标与内部状态、伦理影响以及研究 AI 现象学的方法论方法。
   - 鼓励从事 Agent 架构、存储系统或自指 AI 研究的研究人员提交论文或工作坊提案，详见[本论文](https://arxiv.org/abs/2512.14982)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1461548667415691417)** (4 messages): 

> `Perfetto, Chrome Tracing` 


- **Perfetto UI 亮相**：一名成员分享了 **Perfetto UI** 的链接 ([Perfetto UI](https://share.google/PPujbpUqYqPOsAVkC))。
   - 另一名成员提到它与 `chrome://tracing` 工具相关。
- **使用 Chrome 和 Perfetto 进行 Tracing**：用户讨论了使用 **Perfetto** 和 `chrome://tracing` 进行调试和性能分析。
   - 一名用户最初对加载 `chrome://tracing` 时为何提到 **Perfetto** 感到困惑。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1461591891379490837)** (21 messages🔥): 

> `Benchmark Code, scatter-gather, cuda.synchronize() overhead, quack jit` 


- **Sleep 调用导致 GPU 在 Benchmark 期间降频**：一名用户发现 Benchmark 代码中的 `time.sleep(2.0)` 调用导致 **GPU 在计时运行之间降频**，从而导致性能测量不准确。
   - 移除 sleep 调用改善了 Benchmark 结果，因为 **GPU 不再需要在每次计时运行前重新升频**。
- **scatter-gather 操作的优化技巧**：一名用户询问如何优化 PyTorch 中的纯 **scatter-gather 操作**，特别是 `msg = short_matrix[src] * very_tall_matrix` 和 `out.scatter_add_(0, dst, msg)`。
   - 他们已经尝试了向量化和调整操作维度，但想知道 **使用共享内存（shared memory）来减少 atomic_adds** 是否能进一步提升性能。
- **对 `@cute.jit` 注解的困惑**：一名用户质疑为什么来自 [Quack 仓库](https://github.com/Dao-AILab/quack/blob/main/quack/reduce.py#L15) 的特定代码片段使用了 `@cute.jit` 注解，而它看起来像是一个 `@cute.kernel`。
   - 他们询问在这种情况下使用 `@cute.jit` 注解而不是 `@cute.kernel` 是否有特定原因。


  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1461832161157185597)** (1 messages): 

> `Loubna Ben Allal, Smol Training Playbook, Hugging Face, Open Models` 


- **Loubna Ben Allal 的 **Smol Training Playbook****：Loubna Ben Allal 将展示她的著作 [《Smol Training Playbook：构建世界级 LLM 的秘诀》](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction)。
   - 该书是为那些对 **open models** 感兴趣的人提供的全面参考，正如其[配套的 Youtube 视频](https://www.youtube.com/watch?v=y9zOZHXo0eE)中介绍的那样。
- **Open Models 的全面指南**：**Smol Training Playbook** 为热爱 **open models** 及其开发的个人提供了详细的参考。
   - 该指南旨在提供构建世界级 LLM 的秘诀，并承诺分享宝贵的见解和策略。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1461470608339763202)** (1 messages): 

> `Information Gravity, Inference Stability, Hallucination Loops, Excitation Flux, Token Selection` 


- **应用 Information Gravity 解决幻觉问题**：一位成员正在应用 **Information Gravity** 来解决 **Inference Stability** 和 **Hallucination Loops**。
   - 他们映射了 Token Selection 的 **Excitation Flux**；在 S < 45 时，逻辑保持正常（nominal）；当 S > 45 时，substrate 进入线性增长阶段，导致 **Tsys singularity**。
- **Hysteresis Firewall 强制执行稳定性**：该成员在 1.0 处实现了一个 **Hysteresis Firewall**，通过 2.2x gamma-eff flush 来强制执行稳定性。
   - 他们在 [GitHub 上提供了代码逻辑](https://github.com/brayo003/Substrate-X-Theory-of-Information-Gravity/tree/main)，包括 Substrate Modules 和完整逻辑。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1461604792626319393)** (1 messages): 

> `Tau Robotics, Founding Engineers, RL Training/Inference Efficiency, Humanoid Robots` 


- **Tau Robotics 招聘创始工程师**：Tau Robotics 正在旧金山寻找专门负责 **RL 训练/推理效率** 的创始工程师，以构建用于**人形机器人**的通用 AI。
   - 该职位涉及优化世界模型的 rollout/推理性能，加快 RL 训练速度，并在大型 GPU 集群上扩展运行任务，重点使用 **Python、PyTorch 和分布式系统**。
- **身处旧金山的机器人初创公司（线下办公）**：该职位需在旧金山的 Tau Robotics **线下办公**，工作地点位于 Hayes Valley 的一栋住宅和 Mission 的一个仓库。
   - 加入的一个福利是可以与团队同住，并与**人形机器人**做室友，这为该职位提供了一种独特的居住安排。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1461461611079270522)** (7 messages): 

> `CUDA Data Compression, Block Size Implications, RTX 5060 Ti Max OC for AI` 


- **CUDA 数据压缩探索开始**：一位成员询问了关于 **CUDA 数据压缩** 的经验。
   - 另一位成员回答说“答案总是取决于很多因素”，但未作详细说明。
- **Block Size 瓶颈限制带宽提升**：一位成员建议，**block size 为 32** 可能会限制并行处理，因为它将每个 block 限制为单个 warp。
   - 对此，另一位成员澄清说，多个 block 可以利用同一个 **Streaming Multiprocessor (SM)**，而局限性源于当每个 block 只有一个 warp 时，warp 切换（switching）的机会减少了。
- **RTX 5060 Ti Max OC 胜任 AI？**：一位成员询问 **RTX 5060 Ti Max OC (16GB)** 在基础 **AI 训练** 和 **LLM 推理** 方面的性能表现。
   - 目前没有收到回复。


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/1461733268553466032)** (2 messages): 

> `jax.lax.broadcasted_iota, mosaic masked load` 


- **使用 jax.lax.broadcasted_iota 实现 Arange**：**jax.lax.broadcasted_iota** 可用于在 2D tile 甚至是 1D 上执行 **arange** 操作。
- **Mosaic 缺少 masked loading**：**Mosaic** 需要一次加载 **128 个元素**，因为它不像 **tl.load** 的 mask 参数那样具备 masked load 功能。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1461509466259324930)** (20 条消息🔥): 

> `buffer resources, VMEM instructions, memory model for gfx942, HIP compiler` 


- **存储在标量寄存器中的缓冲区资源**：虽然缓冲区资源存储在标量寄存器中，但全局加载和缓冲区加载都支持**标量基地址**和**向量索引**。
   - 缓冲区资源的优势在于**边界检查（bounds checking）**，这可以节省寄存器和控制流，但如果你不需要检查边界，它其实并没有明显的优势。
- **影响 VMEM 指令延迟的因素**：一位用户观察到，在发出之前的 **VMEM 指令**后，即使使用相同的缓冲区描述符且占用率较低，后续 **VMEM 指令**的发射延迟（issue latency）有时也会增加。
   - 影响延迟的可能因素包括**在途（in flight）vmem 操作的最大数量**（达到上限会导致停顿）以及 **DFVS 频率调节**。
- **gfx942 的内存模型与缓存一致性**：gfx942 的内存模型（[https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html#memory-model-gfx942)）讨论了使用 **MTYPE RW** 和 **MTYPE NC** 分别针对 L2 本地和非本地内存的 L2 缓存一致性。
   - 在具有多个 L2 缓存的 SPX + NPS1 模式下，需要手动的 `buffer_inv sc1` 指令来使**非本地 L2 缓存行**失效，但目前尚不清楚“非本地”是否仍指最靠近或连接到 XCD 的 HBM 堆栈。
- **HIP 编译器与 VGPR 寻址模式**：**HIP 编译器**即使在通过 32 位索引进行数组访问时，也经常使用 **2 vgpr 寻址模式**，因为它不会自动识别出偏移量在乘以类型大小后仍将保持在 32 位以内。
   - 用户必须手动计算偏移指针并证明字节偏移量为 32 位，或者使用编译器 builtins 来指示索引范围。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1461512239998701682)** (2 条消息): 

> `Benchmarking, Hardware Trials` 


- **热心的基准测试志愿者**：一位成员表示，如果有任何硬件出现（即使是试用性质），他都*非常乐意进行基准测试*。
   - 他们还提出可以*找有合适设备的朋友*来协助完成基准测试工作。
- **硬件试用吸引技术爱好者**：热心的社区成员自愿运行基准测试。
   - 欢迎设备试用；可以执行基准测试！


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1461474135548104870)** (7 条消息): 

> `sm121a Kernel Development, ThunderKittens vs Cutlass, vLLM Performance, DGX Spark Optimization, Exo and DGX Development` 


- **内核作者为 DGX Spark 进行开发**：一位内核开发者已经为 **sm121a (DGX Spark)** 开发内核大约一周了。
   - 他们的目标是为 **DGX** 进行优化，并在 **vLLM** 中实现尽可能快的推理，以超越 *llama.cpp* 和 *SGLang* 的某些特定分支的性能。
- **针对 DGX Spark 该选择 ThunderKittens 还是 Cutlass？**：内核作者正在考虑在 **DGX Spark** 内核开发中是否应该使用 **ThunderKittens** 而非 **Cutlass**。
   - 他们还没有看到公开可用的针对 **DGX Spark** 优化的内核，并正在寻求最佳方案的建议，因为 **DGX** 是基于 **Blackwell** 架构的。
- **Exo 倾向于 Mac 而非 DGX**：一位成员询问 **Exo** 是否为 **DGX** 发布了任何内容，因为他们之前一直在进行相关实验。
   - 另一位成员指出，**Exo** 目前似乎专注于 **Mac** 及其统一内存架构（unified memory architecture）。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1461545470521245889)** (18 messages🔥): 

> `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR, dual_gemm 截止日期, kernel 运行变慢, 提交被路由至慢速 runner` 


- **CUDA 错误指向可能的越界访问 (Out-of-Bounds Access)**：一名成员报告在调用参考 kernel 时遇到 `CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`，尽管测试通过了。另一名成员建议这很可能是用户 kernel 的问题（极有可能是越界访问）。
   - 该成员建议在 kernel 后添加 `torch.cuda.synchronize()` 以进行调试。
- **Dual_gemm 截止日期**：一名成员询问了 **dual_gemm** 的确切截止日期，明确为 **1/20/26**。
- **Kernel 性能神秘下降**：一名成员报告其 kernel 的运行时间在重新提交后从 **14.x us** 增加到了 **22.x us**，怀疑是服务器端的问题，并分享了运行 ID **363273** 和 **363470**。
   - 一名工作人员确认该 runner 为 `b200-02-gpu4-runner`，并承认已知其运行缓慢，已向 NVIDIA 反馈。
- **通过多重提交策略缓解慢速 Runner 问题**：一名工作人员承认，部分提交被路由到了已知的慢速 runner `b200-02-gpu4-runner`，而 NVIDIA 的运维人员本周末正在休假。
   - 作为临时变通方法，一名成员建议**同时从 3 个终端发起提交**，以增加分配到快速 runner 的概率，这一做法得到了工作人员的默许；另一名成员则建议将慢速 runner 下线。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1461485420721213535)** (14 messages🔥): 

> `GPU Mode 黑客松成功经验, 求职策略, 细分领域技能组合, 面试安排技巧` 


- **GPU Mode 黑客松助力斩获理想工作！**：一名成员在参加了位于纽约市 Jane Street 举办的 **GPU Mode 黑客松**后成功入职。
   - 他们准备了数周，携带了简历、穿着正式，并全身心投入到从早餐到晚餐的社交活动中，最终获得了录用通知（Job Offer）。
- **多样化的求职策略**：该成员强调，他们通过**黑客松**、**在线申请**（附带量身定制的求职信）、**学校招聘会**和**推荐**获得了多个 offer。
   - 他们强调，每种成功的方法都比单纯投递一份通用简历包含了更强的人际连接。
- **打造出众的细分领域技能集**：一名成员建议专注于特定的细分领域（Niche）以区别于其他候选人。建议通过列出自己的优势并寻找最佳重叠交集来发现独特的技能组合，例如：**Kernel Optimization + Reinforcement Learning**。
   - 他们进一步建议，目标应是成为该细分领域的最强者，并针对专门看重这些组合的职位量身定制求职申请。
- **面试安排的心理调适**：一名成员建议拉开面试的时间间隔以保证休息，而不是将面试安排得过于紧凑。
   - 他们表示，在面对大量面试时，将面试全部安排在 2 天内或紧接的下周可能对身心健康不利。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1461498226220990738)** (55 条消息🔥🔥): 

> `Sub 1b Models, Ram Prices, Pure Code Access, MOR vs. MOE, Unified API` 


- **MoE 可能胜过 MOR**：在关于模型架构的讨论中，成员们辩论了 **MoE (Mixture of Experts)** 与 **MOR** 的优劣，结论是虽然 **MOR** 较新且更具实验性，但 **MoE** 通常更好，这取决于使用场景和预算，特别是对于需要快速训练和较少 GPU 的 NLP 任务。
   - 一位成员分享了他们的自定义 **MoE** 实现，其中包括基于 token ID 的确定性基础路由、mu 覆盖、均匀分布、零路由崩溃、mu 引导以及融合门+上投影（fused gate+up projection），声称通过单个 matmul 实现了 *1.3 倍的加速*。
- **纯代码访问网站无法绕过所有封锁**：针对通过纯代码访问网站以绕过封锁和防火墙的问题，一位成员简单地表示：*不，这解决不了你的问题。*
   - 虽然鼓励用户尝试，但共识是这并不能从本质上绕过安全措施。
- **Deepseek Chat 的可用性受到质疑**：一位成员分享了 [Deepseek Chat](https://chat.deepseek.com/share/bzahzv8o99or601as9j) 的链接，质疑它是否只是一堆幻觉，或者它是否具有可用性。
   - 另一位成员表示，他们上次使用 Deepseek 是在 *3 个月前*，发现它*非常出色但又一直处于混乱状态*。
- **DGX Spark 运行 MiniMax**：在终于拿到 **DGX Spark** 的线缆后，一位成员分享说他们正在其上*运行 Minimax*，并且*正在下载中*。
   - 然而，另一位成员评论道，相对于其价格标签，**DGX Spark** 的推理速度极慢，其推理问题在 2025-2026 年甚至 *2030 年* 都会是个问题。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1461575911748014278)** (13 条消息🔥): 

> `Embedding Fingerprints, Detect Anything Tool, Manga filter, Linux terminal finetune, MoE Model Training` 


- **可视化嵌入指纹**：一位成员构建了一个工具，将 Embedding 可视化为 **32x32 图像**，将每个维度映射到一个像素，并将其发布在 [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/embedding-fingerprints) 上。
   - 该工具展示了相似的单词共享视觉模式，不相似的单词看起来不同，且更多维度能捕捉到语义的细微差别。
- **集成 YOLO 的 Detect Anything**：一位成员创建了 **Detect Anything**，这是一个可以根据任何文本提示检测对象并输出用于 **YOLO 训练** 的标注数据的工具，可在 [useful-ai-tools.com](https://www.useful-ai-tools.com/tools/detect-anything) 使用。
   - 虽然产出高质量，但目前成本较高且不适合实时应用，作者正在征求关于 YOLO 数据集可用性、局限性和创意的反馈。
- **漫画化你的图像**：一位成员推荐了一个使用此 [GitHub 仓库](https://github.com/koo1140/Deterministic-AI-training-on-GPUs) 将图像转换为漫画风格的项目。
   - 另一位成员分享了一个根据文本提示生成城市地图的项目 [City Map Generator](https://huggingface.co/spaces/Sudipistaken/City-Map-Gen)。
- **模拟 Linux 终端的 T5 微调**：一位成员介绍了一个旨在模拟 **Linux 终端** 的 **T5 finetune** 模型，能够识别命令并识别未知命令，并在 [HuggingFace](https://huggingface.co/ereniko/LaaLM-v1) 上展示。
   - 其功能包括命令识别和未知命令鉴定，尽管文件创建命令仍处于集成的早期阶段。
- **以 Tao 激励的 MoE 训练**：一位成员正在寻找 AI 高手在 **Bittensor 生态系统** 中构建一个子网，进行 **MoE 模型** 的训练/微调，并获得 **Tao token** 激励。
   - 他还对在 *vllm* 上提升模型开源推理感兴趣。


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1461681672029212718)** (1 条消息): 

> `New Models` 


- **Model Mania 开启 2024**：频道宣布 *我们以一堆新模型开启了这一年，仅本周就有 2 个！*
   - 公告配有两个**风格化角色**的图片，可能代表了这两个新模型。
- **为空主题以满足要求**：此主题有意留白，以满足最低 `topicSummaries` 要求。
   - 此处不代表频道的实际内容。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1461487008693358642)** (50 条消息🔥): 

> `Anthropic 经济指数，税季 AI 融资，METR 时间跨度，Zilliz 语义高亮模型，OpenAI ChatGPT 广告` 


- **Anthropic 索引经济原语**：Anthropic 发布了其第四份**经济指数报告**，定义了“经济原语”来衡量 **AI 使用情况**，指标包括**任务复杂度**、**受教育程度**、**自主性**和**成功率**，详情见 [Anthropic 研究页面](https://www.anthropic.com/research/economic-index-primitives)。
- **报税初创公司获得 350 万美元种子轮融资**：由 **General Catalyst** 支持的 Saket Kumar 为一家旨在完全消除**美国人报税季**负担的企业筹集了 **350 万美元**，目标是使报税过程免费且即时，详见 [Saket Kumar 的推文](https://xcancel.com/saketrkumar/status/2011836460400591330?s=46)。
- **METR 低估了模型时间跨度**：Simon Smith 报道了 **Anthropic 的发现**，即 **METR 的基准测试**可能显著低估了模型的时间跨度，表明实际能力可能比测得的高出 **1.75 倍至 9.5 倍**，具体取决于接口类型（如 API 与 Web 应用程序），讨论见 [Simon Smith 的 X 帖子](https://xcancel.com/_simonsmith/status/2011928926864454133?s=61)。
- **Zilliz 重点推介语义建模**：**Zilliz (Milvus)** 发布了一个 **0.6B 参数的语义高亮模型**，具有 **8192 上下文窗口**，在宽容的 MIT 许可证下提供，并在 [Mervenoyann 的推文](https://xcancel.com/mervenoyann/status/2011732254591275022?s=46)中展示。
- **OpenAI 将在 ChatGPT 中投放广告**：**OpenAI** 宣布计划从 **2026** 年初开始在 **ChatGPT Free 和 Go 层级**中测试**广告**，广告将被清楚地标记，不会影响 **AI 回复**，也不会影响 Plus、Pro 或 Enterprise 等付费层级，详见 [OpenAI 的公告](https://xcancel.com/openai/status/2012223373489614951?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1461528596291125248)** (6 条消息): 

> `实时转录 vs 后期处理，Apple Silicon 性能` 


- **本地模型更倾向于后期处理**：用户正在讨论本地模型的实时转录与后期处理之间的权衡，一位用户指出实时转录会让他们的笔记本电脑变成“**土豆**”（极其卡顿）。
   - 另一位用户提到**后期处理**更可取，因为可以在 **AFK**（离开电脑）或夜间完成，从而保持笔记本电脑运行流畅。
- **Apple Silicon 性能受到质疑**：一位用户询问另一位用户运行的是什么设备，暗示 **Apple Silicon** 不应该出现性能问题。
   - 该用户提到了一种抱怨，即“在我的笔记本电脑上运行模型，大多数情况下会把它变成一个土豆”。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1461616836574449755)** (9 条消息🔥): 

> `Higgsfield AI A 轮融资，Min Choi 现实扭曲` 


- **Higgsfield 通过巨额 A 轮融资实现超大规模扩张**：[Higgsfield AI](https://x.com/higgsfield_ai/status/2011866396784017848?s=46) 宣布了 **1.3 亿美元的 A 轮**融资，估值达到 **13 亿美元**，在不到九个月的时间内实现了 **2 亿美元的年度运行率 (ARR)**。
- **Choi 对现实的审视获得 1200 万次浏览**：Min Choi 分享了一种关于现实与模拟之间界限模糊的简短虚无主义情绪，引发了巨大的病毒式反应，浏览量超过 **1200 万次** ([推文](https://x.com/minchoi/status/2011473626927624460?s=46))。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/72974176938158194/1461506020986716293)** (19 messages🔥): 

> `Pangram AI 检测, EleutherAI 数据集贡献, AI 辅助写作, MMLU-Pro 数据集补丁` 


- **关于 AI 转录的辩论**：成员们就基于人类语音并由 AI 转录和润色的文本是否应被视为“AI 生成（AI-generated）”展开了辩论。
   - 一位成员认为，将文本塑造成博客文章并进行风格迁移（style transfer），即使是由原始创意引导的，仍然构成 AI 生成，并将其类比为使用 **Midjourney** 生成图像。
- **Pangram 谨慎的 AI 检测方法受到称赞**：一位成员称赞了 [Pangram](https://www.pangram.ai/) 在将内容标记为 AI 生成时所采取的谨慎态度。
   - 该成员表示，Pangram 似乎优先考虑将人类撰写的内容准确识别出来，即使这意味着有时会将 AI 生成的内容误分类为人类作品。
- **欢迎对 EleutherAI 数据集做出贡献**：一位成员询问如何向 EleutherAI 社区贡献用于微调 GPT-Neo 等预训练 LLM 的指令遵循（instruction-following）数据集。
   - 此外，另一位成员表示愿意作为开发人员为社区内的项目提供服务。
- **MMLU-Pro 数据集补丁发布**：一位成员分享了 Hugging Face 上关于 **MMLU-Pro 数据集** 讨论的[链接](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/discussions/41)。
   - 另一位成员确认他们已经更新了 lm-eval，并发布了关于该补丁的推文。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1461509148981203049)** (9 messages🔥): 

> `液晶非线性, Prompt 大写影响, Gemini 影子更新` 


- **探索液晶非线性**：一位成员尝试使用染料掺杂的**液晶非线性（liquid crystal nonlinearities）**来构建潜在的**光学神经网络（optical NNs）**。
   - 他们询问是否有相关文献研究模型在输入正确的大写/语法 Prompt 与全小写 Prompt 时性能是否更好，并指出*感觉人们确实会关心这一点*。
- **Prompt 大写相关论文链接**：提供了一些相关工作的链接，包括 [https://arxiv.org/abs/2310.11324](https://arxiv.org/abs/2310.11324), [https://arxiv.org/abs/2411.10541v1](https://arxiv.org/abs/2411.10541v1), 以及 [https://arxiv.org/abs/2508.11383v1](https://arxiv.org/abs/2508.11383v1)。
- **Gemini 影子更新阴谋论**：一位成员问道：*我是唯一一个看到 **Gemini** 现在在做什么的人吗*，并声称察觉到数据和输出在 **15 号前后**发生了变化。
   - 他们请求任何注意到这次**影子更新（shadow update）**的人与其联系。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1461731482195329118)** (3 messages): 

> `MMLU-Pro 数据集修复, lm-evaluation-harness 补丁` 


- **MMLU-Pro 数据集获得修复**：一位成员指出，针对 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500) 提交了一个与 **MMLU-Pro** 数据集相关的修复程序。
   - 该修复程序解决了数据集内部的一个问题。
- **lm-evaluation-harness 已打补丁**：一位成员请求另一位成员回复[这条推文](https://x.com/fujikanaeda/status/2011565035408277996?s=20)，告知 **lm-evaluation-harness** 已有新补丁的消息。
   - 该推文应建议查看他们的[代码库](https://github.com/EleutherAI/lm-evaluation-harness/pull/3500)，以便通过简便的方法在该基准测试（benchmark）上进行正确的评估。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1461474702064222482)** (25 条消息🔥): 

> `Kimi-CLI, K2 Turbo Performance, Slide Feature Vision, Kimi Model Discontinuation, Global Memory Toggle` 


- **Kimi-CLI 编程模型滞后**：用户发现 **Kimi-CLI** 的编程模型表现不如竞争对手，且价格比其他更优秀的国产模型更高。
   - 一位成员提到，这可能是因为它没有采用 **K2 Turbo 变体**。
- **K2 Turbo 速度竞赛**：普通 **K2** 版本的速度约为 **28 tps**，而 **Turbo 版本达到 73 tps**。
   - 相比之下，**MiniMax m2.1** 为 **38 tps**（官方渠道），**Z.Ai 的 GLM-4.7** 为 **41 tps**（但在线率较低）。
- **Kimi 的 Vision 现已支持 Slides**：新的 Slide 功能似乎利用了具备 **Vision** 能力的新型 **K2 模型**，可以搜索图像作为参考，如附图 [image](https://cdn.discordapp.com/attachments/1371757564005711973/1461508342424797184/image.png?ex=696c20b6&is=696acf36&hm=70de4ffdcbffa4e7d4572daa8219dad2dfca998f7c15976ce0930997007fdec6&) 所示。
   - 一位成员设置了新的 Preset（预设）：*“对于每一个具体的命名资产（船只、角色、地点、车辆、武器、建筑），先使用确切的专有名词在线搜索直接的视觉参考……”*
- **Kimi 模型会遭受 Google Gemini 式的待遇吗？**：一位成员询问 **Kimi 模型** 是否会像 Google 的 Gemini 模型一样每隔 **12-14 个月** 就停用一次。
   - 另一位成员回答说，旧模型在发布一年后仍可在 [Kimi.com](https://kimi.com) 上使用，并可通过 Moonshot API 平台 [Moonshot API](https://platform.moonshot.ai/docs/pricing/chat#generation-model-moonshot-v1) 获取。
- **全局记忆（Global Memory）现可手动关闭**：用户现在可以关闭 **Global Memory**，一些人认为这比默认开启更好。
   - 一位成员表示：*“不像 Qwen 每一句回复都要把我提供的信息机械重复一遍——这让我很烦——Kimi 不会这样做，它会遵循我关于回复方式的指令……Kimi Thinking 可以预先进行推理”*。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1461475361215877142)** (19 条消息🔥): 

> `PR import process, .NET legacy project, Mono Runtime Environment, Jury-rigged vs Jerry-rigged` 


- **`imported internally` 标签解析**：一位成员询问 PR 上 `imported internally` 标签的含义，想知道这代表“软拒绝”还是评审流程的正常环节。
   - 另一位成员澄清说，这表示该 PR 已被复制到内部仓库进行最终测试和合并，完成后会标记为 `merged-internally`，这意味着 *PR 已进入正式合并前的最后阶段*。
- **遗留 .NET 项目痛点堆积**：一位成员哀叹自己被拉入了一个遗留的 **.NET 4.5.2** 项目（源自 **2014 年** 且只能在 Windows 上运行），甚至连 Readme 文件都没有。
   - 另一位成员深有同感，描述了他们公司一个类似的独立 **C#** 项目，不仅没有文档、问题频出，而且只能在一个特定的“黄金虚拟机（golden VM）”上构建。
- **Mono 运行时：救星还是沉船？**：一位成员建议该遗留 **.NET** 项目可能可以在 **Mono** 上运行，随后另一位成员描述了他们尝试使用 **Mono** 将项目容器化但未成功的经历。
   - 另一位成员指出 [Microsoft 维护着一个 **Mono** 仓库](https://github.com/dotnet/runtime/tree/main/src/mono)，暗示 **Mono** 并没有被完全弃用。
- **`Jury-rigged` 或 `Jerry-rigged`：一个语法失误**：在一位成员提到他们的容器化工作需要对遗留项目进行 *jerry rigged*（胡乱拼凑）后，另一位成员指出了 *jury-rigged*（航海中的临时索具修补）和 *jerry-rigged*（起初就建造低劣）之间的区别。
   - 他们澄清这两个词经常混用，但对他们来说，这听起来像是称 **.NET**、**Mono** 和 **Wine** 构造低劣。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1461509552938946602)** (2 条消息): 

> `Nu game engine, Shading languages` 


- **Nu 游戏引擎避开着色语言**：**Nu 游戏引擎**的创作者提到，该引擎在没有 Shading Language（着色语言）的情况下运行。
   - 讨论者开始理解这种做法背后的逻辑。
- **理解 Nu 的无着色器方案**：讨论强调了 **Nu 游戏引擎**在没有传统着色语言的情况下运行的独特设计选择。
   - 这一决定引发了人们对游戏开发中此类方法的优点和潜在缺点的思考。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1461906656580603975)** (6 messages): 

> `自主 AI 治理, 零知识证明 (ZKPs), 隐私保护, 恶意行为检测` 


- **ZKPs：自主 AI 治理的关键**：一位成员提议使用 **零知识证明 (ZKPs)** 构建一个自主的 AI/技术治理系统，以确保 100% 的 **隐私保护**。
   - 该概念涉及使用 **ZKPs** 来证明对既定标准的合规性，而无需泄露行为的性质或行为者的身份。
- **通过 ZKP 进行主动监管**：该成员建议使用标准模型对内容进行分类（例如，是否有暴力倾向），并要求提供 **ZKP** 以证明传输的内容已通过内容分类过滤器，且被归类为无害。
   - 这将确保网络仅以可验证的方式运行经过批准的模型，同时对对话内容保持完全的 **隐私**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1461651346502713354)** (12 messages🔥): 

> `Google Translate Gemma, ChatGPT Go, OpenAI 广告, DeepSeek NLP 广告拦截器` 


- **ChatGPT Go 发布，OpenAI 探索分层订阅**：OpenAI 推出了 [ChatGPT Go](https://openai.com/index/introducing-chatgpt-go/)，标志着对 **更多层级 (moar tiers)** 的探索。
   - 一位成员分享了一张图片，以此开玩笑地问道 *"什么时候出 80 美元的层级？"*，表达出 *"这个实验需要尽快开始赚钱了"* 的氛围。
- **OpenAI 将在 Free 和 Go 层级测试广告**：OpenAI 很快将在 **Free** 和 **Go 层级** 测试广告。
   - 这导致一位成员表示：*"在玩了几年梗之后，OpenAI 最终还是被企业工业垃圾 (corposlop) 吞噬了"*。
- **DeepSeek 将发布 NLP 广告拦截模型**：一位成员预计 **DeepSeek** 将发布一个 **NLP 广告拦截模型**，该模型基于自然语言检测广告，并以 MIT 许可证发布。
   - 另一位成员指出，在第三方 API 客户的响应中插入广告将会是一个 *"大麻烦"*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1461522517947711518)** (17 messages🔥): 

> `基于积分的使用与计费系统, Manus 支付问题, Manus 积分, Manus 支持的应用大小, 功能请求` 


- **AI 工程师寻求积分制平台项目**：一位 AI 工程师正在寻求机会，帮助有积分制使用需求的平台 **强化使用情况跟踪** 或构建更 **可靠的计费/积分系统**。
- **用户反馈 Manus 支付问题**：一位用户报告在尝试充值积分时遇到了多个支付问题，包括 **升级会员**、**使用 Link 支付** 以及 **信用卡/支付宝交易** 等方面的问题。
- **Manus 团队请求用户信息以解决支付问题**：Manus 团队的一名成员要求遇到支付问题的用户 **私信其电子邮件地址**，以便跟进并处理问题。
- **用户询问更多邀请码**：一位用户询问 *'你们还有更多邀请码吗？'*，大概与 **Manus 积分或访问权限** 有关。
   - 另一位用户回答说 *'每个月只能使用 1 个代码'*。
- **建议增加 Manus 支持的应用大小**：一位用户建议增加 Manus 平台支持的 **应用程序最大容量**，并提到在尝试创建包含 **100 个 MP3 文件（总计 600MB）** 的音频播放器应用时遇到了限制。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1461459876973903894)** (4 messages): 

> `aider 自动添加文件, ChatGPT 订阅, Aider 活跃开发, Opus 4.5 基准测试` 


- **用户希望 aider 自动添加文件**：一位用户询问是否可以让 **aider** 自动添加文件，而不是弹出确认提示。
- **询问 Aider 的开发进度**：一位用户询问了 **aider** 的开发状态，注意到上一次发布是在 8 月份，且基准测试中缺少像 **Opus-4.5** 这样的新模型。
- **aider 请求支持 ChatGPT 订阅**：一位用户询问 **aider** 是否像 **opencode** 一样支持 **ChatGPT 订阅**，并表示他们是 **ChatGPT Plus** 用户。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1461451569949315105)** (3 messages): 

> `CI log files, aider workflow` 


- **Aider 的 CI 日志文件处理**：一位成员询问了处理 **CI 日志文件** 的最佳实践，以便这些文件不被包含在 git 中，但仍能通过 `aider --read ci.log` 被 aider 读取。
   - 该成员似乎想知道如何将 **Aider** 集成到他们的工作流中。
- **Aider 工作流集成**：用户的提问暗示了将 **Aider** 集成到 CI/CD 流水线中以进行自动测试和修复的需求。
   - 这表明了 **Aider** 在直接从 CI 日志中识别并解决测试失败问题方面的潜在用例。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1461667018427076741)** (5 messages): 

> `Embedded Tinygrad, Accelerator Bytecode` 


- **嵌入式 Tinygrad 部署**：一位成员咨询了在带有 **tinygrad** 支持的板载加速器的嵌入式环境中运行 **tinygrad** 的最佳实践。
   - 该成员提到他们无法访问 **Python**，但该应用非常适合 **tinygrad** 在某些平台上的**驱动替换**（driver replacement），并引用了 [这条推文](https://x.com/__tinygrad__/status/1989026590127464554)。
- **加速器字节码导出**：一位成员询问关于导出通过 **BEAM engine** 推送并经过 **JIT'ed** 的加速器字节码的问题。
   - 另一位成员回答说*你可以导出任何东西，参考 comma 的相关实现*，并指向了 `extra/export_model.py`，特别是 `export_model`、`compile_net` 和 `jit_model` 函数。


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1461784898598736170)** (1 messages): 

> `London Summit Livestream, London Summit VODs` 


- **去年伦敦峰会举行了直播**：去年的 **London Summit** 设有**直播**。
- **将发布伦敦峰会的 VOD**：至少会发布来自 **London Summit** 的 **VOD**。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1461742351969026048)** (3 messages): 

> `MCPs, open source project, contributor collaboration` 


- **贡献者协作焦点**：一位成员询问如何获取关于一个**开源项目**的 **MCP server** pull request 的反馈。
   - 另一位成员澄清说，该服务器主要关注**贡献者协作**，并提议通过 DM 分享更多相关服务器的详情。
- **MCP Server 反馈请求**：一位成员正在为一个开源项目构建 **MCP server**，并正在寻找一个可以对其 pull request 获取反馈的空间。
   - 他们正在寻求一个频道，以便提出与其项目相关的问题并获得建设性的批评意见。