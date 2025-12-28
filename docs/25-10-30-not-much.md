---
companies:
- moonshot-ai
- minimax
- bytedance
- princeton
- mila
- openai
- cursor
- cognition
- hkust
date: '2025-10-30T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Moonshot AI（月之暗面）**发布了 **Kimi Linear (KDA)**，具备首日即用的基础设施和强大的长上下文指标，实现了高达 **75%
  的 KV 缓存减少**和 **6 倍的解码吞吐量**。**MiniMax M2** 转向全注意力机制以增强多步推理能力，在支持 **200k 上下文**和 **约
  100 TPS** 的同时，保持了强大的智能体编程性能。**字节跳动**、**普林斯顿大学**和 **Mila** 推出了 **Looped LLMs（循环大语言模型）**，展示了与更大型
  Transformer 模型相当的效率提升。**OpenAI** 的 **Aardvark (GPT-5)** 作为一款用于可扩展漏洞发现的智能体安全研究员进入了私测阶段。**Cursor**
  推出了更快的云端编程智能体，尽管其基础模型来源的透明度问题引起了关注。**Cognition** 发布了一款名为 **Devin** 的桌面/移动端工具使用智能体的公测版。社区讨论了先进的注意力机制和自适应计算技术。'
id: MjAyNS0x
models:
- kimi-linear
- kimi-delta-attention
- minimax-m2
- looped-llms
- aardvark-gpt-5
people:
- kimi_moonshot
- scaling01
- uniartisan
- omarsar0
- aicodeking
- songlinyang4
- iscienceluvr
- nrehiew_
- gdb
- embeddedsec
- auchenberg
- simonw
title: 今天没发生什么事。
topics:
- long-context
- attention-mechanisms
- agentic-ai
- tool-use
- adaptive-compute
- coding-agents
- performance-optimization
- memory-optimization
- reinforcement-learning
- model-architecture
---

**平静的一天**

> 2025/10/29-2025/10/30 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（198 个频道，5621 条消息）。预计节省阅读时间（以 200wpm 计算）：490 分钟。我们的新网站现已上线，支持全元数据搜索，并以优美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

祝贺 HuggingFace 发布 [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#first-things-first-evals-before-everything-else)。欢迎 [Beyang (Amp) 和 Skyler (MiniMax)](https://x.com/swyx/status/1983939826069205340) 加入 AIE CODE，并查看 [Stripe / ACP Latent Space 播客](https://www.latent.space/p/stripe)！

---

# AI Twitter 回顾

**Kimi Linear (KDA)、Minimax M2 以及线性注意力（linear-attention）之战**

- **Kimi Linear (KDA) 发布即支持首日基础设施，并具备强大的长上下文指标**：Moonshot AI 发布了 Kimi Linear 技术报告和检查点——这是一种将 **Kimi Delta Attention (KDA)** 与 MLA 交织（KDA:MLA 比例约为 3:1）的混合架构，开源了优化的 KDA CUDA 内核，并在发布首日集成到 vLLM 中。据报告收益：**KV cache 减少高达 75%**，**解码吞吐量提升高达 6 倍**（在 1M 上下文下 TPOT 提升 6.3 倍），且在长上下文和 RL 长文本推理任务中的质量与全注意力（full attention）持平甚至更好。vLLM 显示 **RULER 128k = 84.3，且比基准线快约 4 倍**，并确认了内存和吞吐量的优势。值得注意的是，团队还报告了在 MLA 层不使用位置编码（“NoPE” + 其他位置感知机制）的情况下，长上下文依然有效。链接：[@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1983937694360322136), [@scaling01](https://twitter.com/scaling01/status/1983926811051384965), [vLLM](https://twitter.com/vllm_project/status/1983941708233765149), [@uniartisan](https://twitter.com/uniartisan/status/1983941443283775780)。
- **Minimax M2 从混合方法转向全注意力（full-attention）**：MiniMax 公开反思了早期混合/滑动窗口变体在多跳推理（multi-hop reasoning）中面临的挑战，并将 M2 转向了全注意力机制——尽管如此，M2 在 200k 上下文、约 100 TPS 以及广泛的工具链支持下，依然展现了强大的 Agent 编程性能（例如，在多个用户评估中位居开放权重模型前列），目前限时免费试用。社区评论指出，M2 早期的线性变体较为简单，如果多跳性能退化较小，更好的混合架构（如 KDA）在效率方面仍具前景。链接：[@omarsar0](https://twitter.com/omarsar0/status/1983915573215162873), [vLLM M2 support](https://twitter.com/vllm_project/status/1983936128878059541), [@aicodeking](https://twitter.com/aicodeking/status/1983934597353402797), [@SonglinYang4](https://twitter.com/SonglinYang4/status/1984021551914926514)。
- **潜空间循环（Latent looping）与自适应计算**：ByteDance/Princeton/Mila 的 “Looped LLMs” 显示，**1.4B/2.6B LoopLMs（训练于 7.7T tokens）** 在大多数基准测试中与约 4B/8B 的标准 Transformer 持平——这证明了循环潜空间推理可以用计算时间换取质量和数据效率，并可能随 MoE 扩展。项目链接：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1983864569035690350), [项目/论文](https://twitter.com/iScienceLuvr/status/1983864571095085511)。社区还深入讨论了 DeltaNet/RetNet/Mamba-v2 的衰减与 delta-rule 变体，以及 MLA partial-RoPE/NoPE 的权衡（例如：[@nrehiew_](https://twitter.com/nrehiew_/status/1983891931823505518), [@Grad62304977](https://twitter.com/Grad62304977/status/1983913767118229953)）。

**Agent 编程与工具使用系统**

- **OpenAI 的 Aardvark (GPT-5) 进入私测阶段**：Aardvark 被定位为一种“智能体安全研究员 (agentic security researcher)”，能够阅读/分析代码、编写并运行测试，并提出补丁建议——早期用户将其视为可扩展漏洞发现与修复的初步尝试。相关链接：[@OpenAI](https://twitter.com/OpenAI/status/1983956431360659467), [@gdb](https://twitter.com/gdb/status/1983971650531160319), [@embeddedsec](https://twitter.com/embeddedsec/status/1983956550239842474)。
- **编程智能体 (Coding agents) 快速迭代（并受到审视）**：Cursor 推出了更快、更可靠的云端智能体，并分享了内部使用模式（[发布公告](https://twitter.com/cursor_ai/status/1983954528933421419)，[使用方式](https://twitter.com/benln/status/1983960258809831530)）。与此同时，用户注意到 Cursor 的 Composer-1 偶尔会用中文“思考”，引发了关于基础模型来源透明度的讨论 ([@auchenberg](https://twitter.com/auchenberg/status/1983901551048470974), [@simonw](https://twitter.com/simonw/status/1983912102457963005))。Cognition 发布了 “Computer Use” 的公测版——Devin 现在可以操作桌面/移动端工具，分享屏幕录制并构建 GUI 应用 ([@cognition](https://twitter.com/cognition/status/1983983151157563762))。
- **工具使用评估与编排**：香港科技大学 (HKUST) 的 Toolathlon (Tool Decathlon) 引入了一个涵盖 **32 个应用程序/600 多个工具**的基于执行的基准测试，结果显示当前的 SOTA 性能仍然较低（例如 Claude Sonnet 4.5 的成功率为 **38.6%**），且开源与闭源模型之间的差距依然存在 ([@junxian_he](https://twitter.com/junxian_he/status/1983834164727312391))。新的规划研究涵盖了基于 RL 调度的并行工具使用（[基于图的智能体规划](https://twitter.com/omarsar0/status/1983892163990843692)，[论文](https://twitter.com/omarsar0/status/1983892176892522642)）。LangGraph 添加了 Overwrite 功能，以绕过 reducers 直接进行状态替换 ([@caspar_br](https://twitter.com/caspar_br/status/1983949095837519901))。LangChain 举办了无代码 Agent Builder 圆桌会议 ([@LangChainAI](https://twitter.com/LangChainAI/status/1983916519513059728))。
- **实时上下文流水线**：事件驱动的“流式智能体 (streaming agents)”正向生产环境迈进，**Confluent + Weaviate** 的示例以及 **Confluent + Qdrant** 的合作伙伴关系实现了实时数据 + 向量搜索，使上下文感知智能体能够超越过时的 RAG 快照 ([Weaviate](https://twitter.com/weaviate_io/status/1983921589163835398), [Qdrant](https://twitter.com/qdrant_engine/status/1983843826436395090))。

**训练、评估与嵌入 (Embeddings)**

- **Hugging Face 的 Smol Training Playbook (200+ 页)**：由 HF 科学团队编写的精炼“实战指南”，涵盖了预训练数据策展、架构选择、后训练 (SFT/RL) 以及基础设施调试（包括 NCCL 炼狱等内容）。该手册强调消融实验 (ablations) 和论文中常被忽略的复杂现实，是对早期 FineWeb 和 Ultrascale 指南的补充。相关链接：[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1983929546014433385), [@_lewtun](https://twitter.com/_lewtun/status/1983929588909797414), [@eliebakouch](https://twitter.com/eliebakouch/status/1983930328751153159)。
- **面向企业的嵌入评估**：Voyage 的量化感知训练 (QAT) 嵌入模型 **voyage-3-large** 在新的 HF RTEB 排行榜中位居榜首，在 **33 个数据集**中排名第一，并在以应用为中心（金融/法律/医疗）的检索任务中超越了 OpenAI/Cohere。QAT 使模型在 INT8/二进制下保持准确，从而降低了向量数据库成本并实现了更快的推理 ([@_avichawla](https://twitter.com/_avichawla/status/1983783708047093838))。
- **开源与闭源差距缩小**：Epoch AI 的 ECI 表明，开源权重模型与闭源 SOTA 的平均滞后时间约为 **3.5 个月**（≈**7 个 ECI 点**，类似于 “pro” 与 “mini” 的差距），表明追赶速度比预想的更快 ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1983987212183335097))。
- **延迟交互 (Late-interaction) 检索基础设施**：LightOn 的 Fast-Plaid 1.2.5 为 ColPali/ColQwen/PyLate 风格的检索带来了更快的速度和更低的 GPU 显存占用 ([@raphaelsrty](https://twitter.com/raphaelsrty/status/1983906400725024931))。

**多模态：语音、视频与图像编辑**

- **大规模 SSM 语音技术**：Cartesia 的新旗舰 TTS **Sonic-3** 采用了 State Space Model 架构，提供具有韵律元素（如笑声、惊讶）的低延迟流式语音。它支持 **42 种语言**（包括 9 种印度语言），目前已进入 Artificial Analysis 竞技场进行盲测 ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1983879759194157194))。
- **物理感知编辑与世界模拟**：NVIDIA 的 **ChronoEdit-14B**（开源代码、模型和 demo）通过“视频推理”阶段 + 轨迹 token 的 in-context editing，在约 8 个 diffusion steps 内完成图像编辑（在 H100 上约 4s/张）——这对于可视化编辑的“思考过程”也非常有用 ([paper/model/demo](https://twitter.com/_akhaliq/status/1983953896415604836), [作者更新](https://twitter.com/jayzhangjiewu/status/1983963044695740848))。
- **视频生成更新**：Google 的 **Veo 3.1** 在图生视频（image-to-video）方面有显著提升（Veo 3.1 Fast 在 AA 的 I2V 竞技场中排名第 2），尽管文生视频（text-to-video）质量较 Veo 3 并没有进步；价格维持在 $0.2/s（不含音频） ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1983938159839998249))。

**产品与基础设施更新**

- **Perplexity** 推出了 “Patents”——一个引用优先的专利研究 Agent——目前处于免费 Beta 阶段，同时推出的还有 “Discover” 和新的金融功能，如政治家持股追踪 ([Patents 发布](https://twitter.com/perplexity_ai/status/1983875975877423277), [Discover](https://twitter.com/AravSrinivas/status/1983960821731619025), [Finance](https://twitter.com/AravSrinivas/status/1983998749929259378))。
- **VS Code + OpenAI**：VS Code Insiders 增加了 “Plan”（任务分析和实施计划），并为 Copilot Pro+ 账户集成了 OpenAI Codex。OpenAI 还推出了 **Codex credits** 以突破计划限制 ([Plan](https://twitter.com/code/status/1983942033879257195), [Codex](https://twitter.com/code/status/1983973969335378241), [credits](https://twitter.com/OpenAIDevs/status/1983956896852988014))。
- **Sora 商业化**：用户现在可以购买额外的生成次数；计划建立一个更广泛的 “Sora 经济体”，包含版权方付费的客串（cameos），且由于 GPU 限制，免费层级可能会随着时间推移而减少 ([@billpeeb](https://twitter.com/billpeeb/status/1984011952155455596))。
- **基础设施与平台**：marimo 加入 CoreWeave 以扩展 molab，同时加倍投入开源 notebooks ([@marimo_io](https://twitter.com/marimo_io/status/1983916371869364622))；Locally AI 推出了基于 MLX 构建的原生 Mac 应用 ([@LocallyAIApp](https://twitter.com/LocallyAIApp/status/1983957683737915405))；Baseten Training GA 带来了具有缓存感知调度功能的按需多节点训练 ([@basetenco](https://twitter.com/basetenco/status/1983958807353934180))；SGLang-JAX 现在通过 SkyPilot 单行命令支持 TPU ([@skypilot_org](https://twitter.com/skypilot_org/status/1983957542863851899))；一篇关于 DGX Spark 的实测评论强调了其在 CUDA 原型设计和小规模 inference 方面相对于 H100 的优势 ([@rasbt](https://twitter.com/rasbt/status/1983895811915214996))。

**热门推文（按互动量排序）**

- [@sundarpichai](https://twitter.com/sundarpichai/status/1983922303424471541)：Google x Jio 合作伙伴关系——向全印度符合条件的 Jio 用户免费推出 Google AI Pro 计划（Gemini 2.5 Pro, 2TB 存储空间, 创作工具）。
- [@sama](https://twitter.com/sama/status/1983941806393024762)：关于动机、公平和致力于 AGI 的讨论——来自 OpenAI CEO 的广受讨论的个人笔记。
- [@OpenAI](https://twitter.com/OpenAI/status/1983956431360659467)：Aardvark——OpenAI 由 GPT-5 驱动的 Agent，用于发现并修复安全漏洞（私测版）。
- [@sama](https://twitter.com/sama/status/1984025727763935585)：“GPT-6 将更名为 GPT-6-7”（在沉重的新闻周期中的幽默）。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Hugging Face 训练见解

- [**200 多页关于如何训练 LLM 的 Hugging Face 秘籍**](https://www.reddit.com/r/LocalLLaMA/comments/1ok3xie/200_pages_of_hugging_face_secrets_on_how_to_train/) (热度: 1047): **Hugging Face 发布了一份名为 "The Smol Training Playbook" 的 200 多页综合指南，详细介绍了训练大语言模型 (LLM) 的整个流程，包括 Pre-training、Post-training 和 Infrastructure。该指南旨在分享哪些策略有效、哪些无效的见解，旨在帮助从业者构建可靠的 LLM。该手册可在 [Hugging Face 平台](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)上获取，其结构涵盖了模型架构、Data Curation 和 Infrastructure 考量，为 LLM 开发提供了详细的路线图。** 社区反应积极，用户对 Hugging Face 提供的关于 Parallelism 和高级训练技术的详细见解表示感谢和赞赏。有用户请求为移动端用户提供更好的访问体验，表明了对更易用的格式的需求。
    - Hugging Face 的手册因其对 Parallelism 和高级训练技术的全面覆盖而受到称赞，对于那些希望了解大规模模型训练的人来说，它是一个宝贵的资源。它被描述为这些主题的“一站式目的地”，表明了其在处理复杂训练场景方面的深度和广度。
    - 手册的构建过程存在一个技术问题，在运行 `npm build` 和设置环境等多个步骤中出现了 “cache miss” 错误。这表明在部署流水线中存在潜在的优化或故障排除空间，特别是在缓存策略和构建配置方面。
    - 该手册可以在线获取，但人们对实体纸质版很感兴趣，这表明了对更易获取格式的需求。这可能暗示了更广泛的受众群体，他们更喜欢传统的阅读方式，或者需要离线访问进行深入研究。

### 2. 开源 AI 音乐生成倡议

- [**Udio 刚刚掠夺并背叛了其付费订阅者……这是我们需要更多 Open Source 的另一个原因**](https://www.reddit.com/r/LocalLLaMA/comments/1ojqvwe/udio_just_robbed_and_betrayed_its_paying/) (热度: 553): **一位 Reddit 用户报告称，音乐创作平台 Udio 在未事先通知的情况下，取消了订阅者将歌曲下载为** `.wav` **文件的功能，引发了用户的沮丧。这一变化引发了对反消费者行为的担忧，尤其是来自北美公司的行为，并激发了支持开源 AI 音乐生成替代方案的兴趣。该用户表示愿意为该领域的开源开发者提供资金支持。** 评论者认为，此举可能会损害 Udio 的用户群，因为无法下载作品削弱了平台的实用性。有人推测 **Universal Music Group** 可能会影响这些变化以压制 AI 音乐生成，可能是为了保护传统音乐行业的利益。
    - 一位用户推测 Universal 可能故意破坏了 Udio 以压制 AI 音乐生成。他们认为 Universal 关于“新时代”和“历史性合作伙伴关系”的公开声明具有误导性，因为其真实意图可能是消除来自 Udio 等 AI 驱动平台的竞争。
    - 另一位自称是 Data Scientist 的评论者提到，在知识产权法较不严格的地区训练自己的音乐模型具有潜力。这突显了开发独立 AI 音乐模型的兴趣日益增长，尤其是在法律限制较少的地区。
    - 有人建议，如果 Udio 的网站关闭，应在 Hugging Face 等平台上发布其 Model Weights。这将允许社区克隆并继续开发该模型，确保该技术保持可访问性，并由开源贡献者进一步改进。

### 3. Qwen 3 VL 与 Kimi Linear 模型更新

- [**Qwen 3 VL 已合并至 llama.cpp！**](https://www.reddit.com/r/LocalLLaMA/comments/1ok2lht/qwen_3_vl_merged_into_llamacpp/) (活跃度: 347): **`Qwen 3 VL` 模型已成功集成到 `llama.cpp` 仓库中，详见[此 Pull Request](https://github.com/ggml-org/llama.cpp/pull/16780)。此次集成带来了性能提升，用户注意到 `32b` 模型在 `AIME25` 等基准测试中相比 `30b 2507` 版本有所改进。建议用户在处理纯文本用例时，使用比 Qwen 模型卡片（Model Card）建议更低的 Temperature（采样温度）以获得最佳效果。** 目前用户正期待 `GGUF` 和 `unsloth quants` 的发布，部分用户在初始测试中对 `Qwen3-VL-32B Q6` 模型的表现表示满意。
    - ForsookComparison 提到他们创建了自己的 Qwen3-VL-32B Q6 模型版本，并指出其在初始测试中表现良好。他们建议在文本用例中运行该模型时使用比模型卡片推荐更低的温度，这表明针对特定应用存在潜在的优化空间。
    - YearZero 提供了文本基准测试的对比分析，强调 Qwen 3 VL 32B 模型比 30B 模型有显著进步，特别是在 AIME25 基准测试中。这表明新模型在各项指标上都提供了增强的性能，链接中的基准测试结果也证明了这一点。
    - Arkonias 幽默地提到了 LM Studio 对该模型支持的预期延迟，暗示虽然模型已经可用，但集成到流行工具中可能还需要额外时间。这凸显了新模型部署中的一个常见问题，即软件支持往往滞后于模型发布。
- [**Kimi Linear 发布**](https://www.reddit.com/r/LocalLLaMA/comments/1ojz8pz/kimi_linear_released/) (活跃度: 295): **[Moonshot AI](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) 发布了 Kimi Linear 48B-A3B，该模型采用了修改后的 Gated DeltaNet 架构。尽管其基准测试得分低于 Qwen3-30B-AB3，但令人印象深刻的是，它的训练 Token 使用量减少了 `25 倍`。该模型预计将支持极大的上下文窗口（Context Size），可能高达 `1M`，并有望在 `2x3090` GPU 上通过 AWQ 量化进行测试，以评估其在 vllm 中的性能。** 用户期待 `llama.cpp` 实现 Qwen Next 架构以支持此模型。MLA 与 Linear 的结合受到了赞赏，人们对其可能具备的类似 Kimi K2 的模型个性感到兴奋。
    - AlbeHxT9 提到 Kimi Linear 基于修改后的 Gated DeltaNet 架构。他们指出，在 `llama.cpp` 能够利用 Kimi Linear 之前，必须先实现 Qwen Next 架构，这表明其依赖于未来的架构更新。
    - Marcuss2 强调，虽然 Kimi Linear 的基准测试得分低于 Qwen3-30B-AB3，但其训练使用的 Token 数量减少了约 25 倍。这表明其训练过程非常高效，考虑到数据使用量的减少，这一表现令人印象深刻。
    - rekriux 讨论了 Kimi-Linear 48B-A3B 支持超大上下文窗口的潜力，并指出 MLA + Linear 的组合非常有益。他们表示有兴趣在双 3090 GPU 上使用 `vllm` 进行 AWQ 量化测试，以探索其处理 1M 上下文的能力。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic 的 Claude Skills 与内省意识

- [**Anthropic 发现了 LLM 中存在“真实内省意识”的证据**](https://www.reddit.com/r/OpenAI/comments/1ok0vo1/anthropic_has_found_evidence_of_genuine/) (热度: 828): **Anthropic 发布的研究表明，大型语言模型（LLM）可能通过检测其自身激活状态（activation states）的修改来表现出“真实的内省意识（genuine introspective awareness）”，这些状态属于内部处理过程，而非输入或输出文本。这种内省通过模型识别并响应其神经激活变化的能力得到证实，暗示了处理过程中的一种自我意识形式。该研究的详细内容见其[内省论文](https://www.anthropic.com/research/introspection)。** 一些评论者认为，这些发现可能仅仅反映了高级的模式识别，而非真正的内省，因为模型是在关联相似概念的海量数据集上训练的。其他人则对研究的客观性表示怀疑，指出这是该公司对其自身产品进行的研究。
    - Andorion 强调，论文中提到的“注入（injection）”是指模型内部处理过程中激活状态的修改，而不是输入文本。这表明模型检测这些变化的能力预示着一种内省意识，因为它能识别自身处理机制的改变。
    - SummerEchoes 认为 Anthropic 提供的例子似乎是简单的模式识别，而非真实的内省意识。模型的输出很可能是其广泛训练数据关联相似概念的结果，可能并不像声称的那样令人印象深刻。
    - thrownededawayed 指出了在定义内省意识方面存在的哲学和经验挑战，并指出即使在人类身上，这个问题也尚未得到解答。该评论认为，人类将内省特质归于 LLM 的倾向可能源于寻找智力同类的深层心理需求。
- [**10 个真正改变我工作方式的 Claude Skills（干货分享）**](https://www.reddit.com/r/ClaudeAI/comments/1ojuqhm/10_claude_skills_that_actually_changed_how_i_work/) (热度: 576): **Claude Skills 引入了一系列功能，通过与各种应用程序集成并自动化工作流来提高生产力。值得注意的技能包括 Rube MCP Connector，它允许通过单个服务器集成超过** `500 apps`**；以及 Superpowers，这是一个开发者工具包，通过** `/brainstorm`**、**`/write-plan` **和** `/execute-plan` **等命令将 Claude 转换为全面的开发工作流。Document Suite 增强了 Claude 处理 Word、Excel、PowerPoint 和 PDF 的能力，不仅能阅读，还能创建具有正确格式和公式的文档。这些技能以带有 YAML 元数据的 Markdown 文件形式实现，易于创建且节省 Token，适用于 [Claude.ai](http://claude.ai/)、Claude Code 和 API。[官方 Skills 仓库](https://github.com/anthropics/skills) 和 [Superpowers](https://github.com/obra/superpowers) 可供进一步探索。**

### 2. 幽默的 AI 与技术迷因 (Memes)

- [**NEHAO**](https://www.reddit.com/r/singularity/comments/1ojx1hr/nehao/) (活跃度: 2385): **这张图片是一个迷因，画面中一个机器人拿着枕头，幽默地与一段关于信用卡支付被拒的文字并列。这反映了对日常生活中类人机器人（humanoid robots）日益增多的戏谑看法，让人联想到电影《我，机器人》（I, Robot）中描绘的科幻场景。幽默之处在于机器人居家强制执行付款要求的荒诞感，突显了社会对机器人融入人类环境的担忧与好奇。** 评论者幽默地讨论了类人机器人日益增强的现实感，将其与《我，机器人》等科幻电影进行类比，并开玩笑说机器人可能会强制执行像收账这样的琐事。
    - Alarm-Particular 指出了类人机器人现状的一个关键点，指出许多演示并非真正的自主（autonomous）。相反，这些机器人通常由操纵员控制，这表明该技术尚未先进到可以独立运行。这引发了人们对寻求融资的公司所做声明的可行性和诚实性的担忧，因为机器人自主执行家务所需的技术仍不成熟。
- [**为什么 CHATGPT 叫我 Batfucker????**](https://www.reddit.com/r/ChatGPT/comments/1ojx3iw/why_is_chatgpt_calling_me_batfucker/) (活跃度: 722): **这张图片是一个迷因，不包含任何技术内容。它幽默地称呼某人为 "Batfucker"，并讨论了一个涉及因“半价”优惠而导致计算变化的虚构场景。语言和表情符号的俏皮使用表明其目的是为了喜剧效果而非技术讨论。[查看图片](https://i.redd.it/jdknlwrci8yf1.jpeg)** 评论建议这个绰号很可能是由用户自己的输入或行为触发的，表明这是与 AI 的一种戏谑互动，而非自发产生的。
- [**Developer vs Vibe Coding**](https://www.reddit.com/r/OpenAI/comments/1ok34tz/developer_vs_vibe_coding/) (活跃度: 978): **这张图片是一个幽默的迷因，通过条形图比较了“开发者”（Developer）和 “Vibe Coder” 的工作风格。它暗示开发者将更多时间分配给规划和用户验收测试（UAT），而 Vibe Coders 则在开发、Bug 以及重做工作（标记为 'WTF' 和 'FML re-do'）上花费更多时间。这反映了一种刻板印象，即开发者更具结构化，而 Vibe Coders 则更随性且缺乏组织。该图表并非基于实际数据，而是利用了对不同编码风格的普遍认知。** 一些评论者幽默地认同了 'Vibe Coder' 的标签，而另一些人则认为该图表过度简化并误导了软件开发的现实，因为所有开发者都会遇到 Bug 并需要重做工作。
- [**什么？**](https://www.reddit.com/r/ChatGPT/comments/1okdtjh/what/) (活跃度: 590): **这张图片是一个迷因，展示了 Sam Altman 在 [X.com](http://x.com/) 上发布的一条幽默帖子，开玩笑地宣布 GPT-6 将更名为 GPT-6-7。这是对版本命名惯例的一种调侃，并不反映 GPT 系列的任何真实技术更新或变化。该帖子的日期为 2025 年 10 月 30 日，旨在产生喜剧效果，而非传达有关未来 AI 发展的任何事实信息。** 评论反映了幽默与讽刺的结合，一位用户开玩笑说 AGI 的未来，另一位用户则提出了一个更有趣的替代名称 '6-9'。这些评论突显了帖子的俏皮性质以及社区对这个笑话的参与。

### 3. AI 带来的法律与教育挑战

- [**这就是那种会再次搅动用户体验的事情……**](https://www.reddit.com/r/OpenAI/comments/1ojloog/this_is_the_type_of_stuff_that_will_stir_up_user/) (热度: 1123): **该图片是一张推文截图，讨论了一项法律裁决，法官允许 George R.R. Martin 和其他作家起诉 OpenAI 侵犯版权。诉讼称 ChatGPT 生成了与《权力的游戏》相似的内容，而 OpenAI 驳回案件的动议被否决。此案凸显了 AI 生成内容与知识产权之间持续存在的紧张关系，这可能导致对 AI 讨论或生成与重大知识产权相关内容的限制。** 评论者对 George R.R. Martin 缓慢的写作速度表示沮丧，幽默地建议 AI 可以比他本人更快地完成作品。还有人担心此类法律行动可能会损害美国在 AI 市场的地位，从而可能使中国等其他国家受益。
    - QueryQueryConQuery 幽默地指出，OpenAI 的 GPT 模型可能比 George R.R. Martin 本人更快地完成他那部期待已久的作品《凛冬的寒风》（*The Winds of Winter*）。这一评论凸显了像 GPT-6 这样 AI 模型的快速发展和能力，它们被认为在快速生成大量文本方面非常高效，与 Martin 缓慢的写作进度形成鲜明对比。
    - RealMelonBread 表达了对针对 OpenAI 等 AI 公司的法律行动可能影响美国在全球 AI 市场竞争地位的担忧。评论者认为，虽然美国可能因这些法律挑战而面临挫折，但中国等国家可能会利用这些机会推进自己的 AI 技术。
    - weespat 澄清了涉及 OpenAI 的持续诉讼中的一个法律层面，指出法官尚未就案件的“合理使用”（fair use）方面做出裁决。评论强调，诉讼的继续并不意味着对案件实质内容的裁决，特别是关于 AI 生成输出的使用。
- [**教育的现状**](https://www.reddit.com/r/OpenAI/comments/1ok1wek/current_state_of_education/) (热度: 575): **该图片是一个迷因（meme），表达了对当前教育现状的沮丧，特别是对 ChatGPT 等 AI 工具生成作业的依赖。该帖子建议，如果 AI 能直接以 PDF 格式提供作业，绕过学生手动编辑 AI 生成内容以使其看起来更像人类的需求，效率会更高。这反映了人们对 AI 在教育中扮演的角色以及它处理传统上由学生完成的“繁琐工作”（busywork）潜力的更广泛担忧。** 评论者对 AI 在教育中的影响表示担忧，其中一位强调需要测试批判性思维而非死记硬背的考试。另一位评论者担心，如果 AI 继续接管传统的学习任务，人类智能的未来和教育的价值将受到威胁。
    - tendy_trux35 讨论了一门进化遗传学课程中一种新颖的测试方法，强调使用开放式问题和笔记、书籍、互联网等开放资源。这种方法强调批判性思维和证据收集，而非死记硬背，表明教育评估方法正在转向更好地培养学生解决现实问题的能力。
    - Tigger3-groton 对比了高中和大学教育，指出擅长死记硬背的学生在需要批判性思维和创新的大学里往往表现挣扎。他们主张将 AI 整合为一种学习工具，强调学生必须学会在生产力上超越 AI 的能力，以在劳动力市场中保持竞争力，并将其与历史上的技术进步进行了类比。
    - Jayfree138 批评当前的教育系统已经过时，并将其比作“工业时代教育”，即资金投入并不等同于实际学习。他们提倡改革教育实践，专注于有意义的学习体验，让学生为现实世界的挑战做好准备，而不是延续一种“付费参与”（pay to play）的模式。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要的摘要
> 

**1. Agentic Coding 与现实检查基准测试**

- **SWE-1.5 竞速：Cerebras + SpecDec 打破速度记录**：[Cognition 发布了 **SWE-1.5**](https://x.com/cognition/status/1983662836896448756)，这是一个 Windsurf Agent 编码模型，利用 **Cerebras** 硬件、**Speculative Decoding**（投机采样）和自定义的 **Priority-queue**（优先级队列），速度高达 **950 tok/s**；其基准测试速度比 Haiku 快约 **6 倍**，比 Sonnet 快 **13 倍**，同时保持了接近 SOTA 的质量。
    - Windsurf 和 Latent Space 的社区反应称其为 *“接近 SOTA 的编码性能”*，并强调了速度优先的工程胜利，同时也在探究这种提升有多少源于系统调优，有多少源于纯模型增益。
- **RLI 现实残酷：Manus 自动化率止步于 2.5%**：ScaleAI 的 [**Remote Labor Index**](https://scale.com/leaderboard/rli)（远程劳动力指数）对 Agent 进行了测试，任务平均耗费 **30 小时**的人力，结果发现表现最好的 Agent（**Manus**）仅实现了 **2.5% 的自动化**，大多数失败归因于质量和完整性问题。
    - 工程师们指出，如果能通过合适的 UI 进行人机协作，这 *“极具价值，但作为劳动力替代品则完全没用”*，这引发了人们呼吁将重点放在工作流集成和错误恢复上，而不是排行榜的炫耀资本。
- **Manus 的迷思：特定领域的胜利，狭窄的泛化能力**：Eleuther 的研究人员讨论了 **Manus** 鲜为人知的表现，指出 Agent 的成功率徘徊在 **1–3%** 左右，这可能反映了分布内的狭窄性，而非广泛的 Agent 能力（参见 RLI 背景：[ScaleAI RLI](https://scale.com/leaderboard/rli)）。
    - 一种尖锐的观点代表了普遍情绪：*“1-2% 的成功率目前还不足以让任何人真正使用 Agent”*，这引发了关于表现优异的领域专用 Agent（如可视化）是否能真正击败分布均匀的通用型 Agent 的疑问。

**2. 新的多模态模型、排行榜与网关**

- **MiniMax 的旋律与人声：Speech 2.6 + Music 2.0**：海螺 AI 推出了 [**MiniMax Speech 2.6**](https://x.com/Hailuo_AI/status/1983557055819768108)，具有 **<250 ms** 的实时延迟和**语音克隆**功能；并首次推出了 [**MiniMax Music 2.0**](https://x.com/Hailuo_AI/status/1983964920493568296)，可创作 **5 分钟** 的专业级歌曲，具备逼真的人声和多乐器控制。
    - 创作者们呼吁提供类似 OpenAI 风格的 API、更多的语言支持（如马拉雅拉姆语）、**变声器**、同步视频，甚至开源，这表明了对生产级工具和透明度的强烈需求。
- **海螺冲至第 7 名：视频竞技场大洗牌**：LMArena 加入了图生视频模型 [**hailuo-2.3-fast**](https://lmarena.ai/leaderboard/text-to-video)，该模型立即登上了文生视频排行榜的 **第 7 位**。
    - 工作人员鼓励用户尝试该模型并反馈结果，而单独的文本排行榜仍停留在 10 月 16 日，这提醒人们基础设施的新鲜度与模型的新鲜度同样重要。
- **Sonar Pro 探求真相：OpenRouter 独家 Pro Search**：OpenRouter 推出了独家的 **Perplexity Sonar Pro (Pro Search)** 模式，访问地址为 [openrouter.ai/perplexity/sonar-pro-search](https://openrouter.ai/perplexity/sonar-pro-search)，主打 **多步 Agent 推理**、**动态工具执行**、**实时思维流** 以及 **自适应研究策略**；详见 [X](https://x.com/OpenRouterAI/status/1984032292436898264) 上的公告。
    - 工程师们将其视为在需要时获取深度、可验证答案的途径，而在不需要时则提供快速响应，使其成为研究密集型对话的务实网关。

**3. GPU Kernel 开发：扫描、采样与小浮点数**

- **Scan Slam：击败 CUB（有时）并驯服 Thrust**：一位 CUDA 开发者报告称，自定义的 `single_pass.bin` scan 操作可以与 **CUB** 的 `DeviceScan` 竞争，并交叉验证了来自 GTC 演讲（[YouTube](https://youtu.be/VLdm3bV4bKo?t=2327)）的带宽目标，同时通过更换[自定义分配器](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu)来调试 **Thrust** 的基准测试。
    - 他们警告说，隐藏的临时分配可能会拖累基准测试表现，并建议在宣布速度夺冠之前，应先标准化策略和临时路径（scratch paths）。
- **FP8 大获全胜：TorchAO + GemLite 迈向低比特**：从业者使用了 TorchAO 的 `quantize_` 配合 **Float8 配置**（参见 AO llama 示例：[generate.py](http://generate.py/)），并分享了一份包含 RTX 4090 结果的全面**量化调查报告**（[基准测试仓库 + 视频](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main#benchmarking-results-on-1-rtx-4090)）。
    - 几位开发者将 **TorchAO** 与 **GemLite** 结合使用（仅权重 vs 激活+权重），并讨论了何时将某些层保留在 **BF16** 中，以权衡吞吐量、稳定性和 Kernel 可用性。
- **Top‑K 策略：当 K≪N 时 Radix 胜出**：针对大规模序列（4K–128K），工程师们重新审视了硬件友好的 **Top‑K** 算法：在 K≪N 时，对比了分块合并（tile-and-merge）与基于 **radix** 的选择，并引用了 FlashInfer 关于采样的博客（[FlashInfer Top‑K 讨论](https://flashinfer.ai/2025/03/10/sampling.html)）。
    - 他们还提到了 NVIDIA **CCCL/CUB** 中即将推出的 **TopK**（[PR #5677](https://github.com/NVIDIA/cccl/pull/5677)），并指出 PyTorch 在 radix 和全排序之间切换的启发式方法仍然是一个实用的基准。

**4. 长上下文工程：Kimi 的线性注意力推动**

- **线性起飞：Kimi 的上下文成本变为线性**：MoonshotAI 发布了 **Kimi Linear Attention** 技术报告，将二次方复杂度的注意力削减为线性，以高效扩展上下文窗口（[技术报告 PDF](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf)）。
    - Nous Research 频道强调了其在超长文档工作流和摘要等应用中的价值，在这些 IO 密集型任务中，线性时间变换的价值被放大。
- **48B 版本上线：Kimi-Linear-48B-A3B-Base 发布**：MoonshotAI 在 Hugging Face 上发布了 [**Kimi-Linear-48B-A3B-Base**](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base)，继续推进其在长上下文 LLM 领域的进展。
    - 从业者对比了不同的注意力变体，其中一位指出 *“*`Kimi Delta Attention` *让我想起了 qwen3 下一个 gated deltanet”*，这推动了跨家族的架构分析。


---

# Discord：高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 版主需要更好的培训**：成员们讨论了新 Discord 版主的技能问题，指出*他们不知道如何管理服务器*。
   - 一位成员建议，*如果行为足够好，新一代（new gens）其实最适合做版主*。
- **Comet 推荐计划变味**：用户抱怨 Comet 推荐计划在他们推广后更改了规则，根据[新服务条款（ToS）](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs)，现在需要 **Pro/Max 订阅**才能推荐他人，这使得 30 天的锁定期变得无法达成。
   - 几位用户报告被骗，其中一位用户称他们有 *$1400 就这样打水漂了*。
- **印度 Jio 用户免费领取 Gemini Pro**：印度成员报告称，印度 Jio 用户可以[免费获得 1.5 年的 Gemini AI Pro](https://www.jio.com/google-gemini-offer/)。
   - 该优惠似乎仅限于印度 Jio 用户。
- **Claude 图标丢失**：用户报告称，在使用 **Claude 4.5 Sonnet** 时，回复中的图标消失了。
   - 这个问题似乎纯粹是视觉层面的。
- **Sonar Reasoning API 无法获取实时数据**：一位用户报告了 **Sonar Reasoning API** 无法获取和交付实时数据（如统计数据和股票价格）的问题。
   - 另一位用户建议，这是因为该实例未连接到实时数据源或**网页搜索模块**，并建议在 **Perplexity** 或 **Sonar Reasoning API** 设置中配置将其链接到实时数据源或启用外部搜索。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **MiniMax 作为更便宜的 AI 竞争者**：成员们讨论了 [**MiniMax**](https://minimax.chat/) 在 **AI** 领域提供了一个更便宜、具有竞争力的替代方案，尽管它可能没有更昂贵选项的高端质量。
   - 一些用户指出，并非所有人都能负担得起顶级工具，需要考虑预算，并提到 **ReCaptcha** 并不那么贵。
- **LMArena 受 ReCaptcha 困扰**：用户报告在 **LM Arena** 上遇到 [频繁的 **ReCaptcha** 提示](https://www.google.com/recaptcha/about/)，部分用户甚至遇到无限循环，导致平台难以使用。
   - 一名工作人员承认了这一问题，并提到他们正在寻找修复验证码和改善用户体验的方法，尽管这可能需要一些时间。
- **视频工具寻找持有 Gemini 密钥的 Beta 测试人员**：一名成员正在为一个视频模型的 [Prompt 生成应用](https://www.testflight.apple.com/join/0S4L0lB4) 寻找 Beta 测试人员，参与测试需要 **Gemini API** 密钥。
   - 该工具旨在帮助用户重新表述 Prompt 并避免触发敏感 Token。
- **Hailuo-2.3 在 LMArena Video Arena 亮相**：**LMArena Video Arena** 在其排行榜中加入了一个新的图生视频模型 [hailuo-2.3-fast](https://lmarena.ai/leaderboard/text-to-video)，它在 **Text-to-Video**（文生视频）中排名第 7。
   - 鼓励成员们尝试新模型并在指定频道分享想法。
- **LMArena 排行榜停滞不前**：成员们报告 [文本排行榜](https://lmarena.ai/leaderboard/text) 停留在 **10 月 16 日**，没有任何更新。
   - 一名工作人员确认排行榜最近没有更新，团队已经意识到这个问题。



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Composer 模型引发争论**：用户对 Cursor 上的 **Composer 1** 存在分歧，一些人称赞它速度快，是实施计划任务的理想选择，而另一些人则认为 **Sonnet 4.5** 在速度和准确性上更胜一筹。
   - 一些成员建议使用 **Auto** 进行规划，使用 **Composer** 进行执行，并指出 **Composer** 的速度接近 **4.5 thinking**。
- **Claude Code 引起轰动**：成员们辩论了 **Claude Code** 的价值，一些人认为与 Cursor 相比，它提供了更好的额度和性价比，而另一些人则强调 Cursor 具有更丰富的功能集。
   - 一些人将 **Claude Code** 视为原生模型提供商，强调了自定义配置（如 hooks、MCP 服务器和 memory）对于获得良好结果的必要性。
- **价格和使用限制引发抗议**：用户报告在 Cursor 的定价和使用限制方面体验差异巨大，许多人因高额的缓存使用量而感到被过度收费，而另一些人则认为 **Claude Code** 的定价更好。
   - 建议包括采用 **Claude Max** 与 **Cursor Pro** 结合的混合方案以获得最佳价值，以及实施成本控制、监控仪表板和支出上限。
- **Cursor 2.0 带来大量 Bug**：用户对 Cursor 2.0 既感到兴奋又感到沮丧，理由是新功能伴随着文件附件问题、水平滚动条故障、上下文丢失以及 Pill（药丸标签）被移除等 Bug。
   - 一些用户还报告输出中混入了 *中文错别字*，以及标签页导航、热键更改、缓存使用问题，并对新的 Agent 评审功能的有效性表示怀疑。
- **Tab Complete 大获全胜**：成员们广泛赞扬了 Cursor 的 **tab complete**（Tab 补全）功能，称其效率和多行编辑能力优于 **GitHub Copilot**。
   - 一位用户强调，“如果你正在处理大型项目，并且想要与代码保持联系并真正理解每一部分，那么这个工作流简直太疯狂了”。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Turing RTX 8000：廉价 VRAM 之王？**：**RTX 8000 Turing 显卡**在 eBay 上售价约 **2000 美元**，提供 **48GB VRAM**，使其适合服务器使用，但它们是较旧的卡。
   - 一位成员建议谨慎对待，担心 AI 支持和潜在的电子垃圾问题，并指出新卡具有更优越的架构。
- **Qwen3 遇到 OOM**：有用户报告称，即使有 **48GB VRAM**，在运行 **Qwen3 4B GRPO notebook** 时仍会出现 **OutOfMemoryError**。
   - 建议包括确保启用了 **4-bit loading**，并调整 **per-device batch count** 以缓解内存问题。
- **Grokipedia 出现，马斯克的百科全书**：Elon Musk 推出了 "**Grokipedia**"，这是一个拥有超过 **80 万篇文章**的 AI 生成百科全书。
   - 这在 **off-topic** 频道中进行了讨论。
- **GEMMA-3：通过 Unsloth 释放的恐怖**：一位成员宣布了一个通过 **Unsloth** 训练的新 **Gemma 3 模型**，并将其推向极限，下载地址见 [Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF](https://huggingface.co/DavidAU/Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF)。
   - 训练了一整套 **GEMMA-3 恐怖系列**：**1B**、**4B**、两个 **12B** 和 **27B**。
- **Anthropic 的内省研究引发惊叹**：一位成员分享了 [Anthropic 关于内省（introspection）的研究](https://www.anthropic.com/research/introspection)，强调了模型检测其隐藏层中注入概念的能力。
   - 一位用户表示，*“这让我大受震撼”*，指的是模型的自我意识和检测篡改的能力。



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出 Perplexity 的 Sonar Pro Search**：**OpenRouter** 与 **Perplexity** 合作发布了 **OpenRouter 独占**版本的 **Sonar Pro**，并启用了 **Pro Search** [在此处](https://openrouter.ai/perplexity/sonar-pro-search)，重点介绍了**多步 Agentic 推理**等功能。
   - 这种新模式的功能包括**动态工具执行**、**实时思维流**和**自适应研究策略**，在 [Twitter](https://x.com/OpenRouterAI/status/1984032292436898264) 上有进一步讨论。
- **OpenRouter Typescript SDK 启发了简易演示应用**：一位成员部署了一个*简易演示应用，几乎未对原始* [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk/tree/main/examples/nextjs-example) 进行修改，用于测试带有**环境变量**的 **API endpoints** 并实现缺失的 **OAuth** 功能。
   - 开发者澄清说，*他们绝对不想把这做成一个严肃的项目，这只是为了启发和概念验证。*
- **Sora 2 在生成猫娘时遇到困难**：一位用户报告称，**Sora 2** 总是生成带有类人耳朵和比例失调的巨大胸部的猫娘图像，即使提示词要求更*“可爱”*。
   - 这个问题被认为与训练数据中的潜在偏见有关，导致模型倾向于将角色性感化。
- **由于 DeepInfra 错误导致 Ultra 模型不稳定**：一位成员发现 **Ultra** 模型非常不稳定，提到由于从 **DeepInfra** 切换到 **Z.AI**，其推理条件正在发生变化。
   - 当从 **DeepInfra** 切换到 **Z.AI** 时，问题得到了解决。
- **添加 Embedding 模型进行测试**：正在测试添加 Embedding 模型，特别是 [OpenAI 的 text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small)。
   - 一位成员指出他们收到了一堆随机数据，但*不使用 raw response 似乎可以解决问题。*



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 求职者承认竞争激烈**：一名成员申请了 Hugging Face 的职位，并承认成功的几率很低，因为据称他们收到了**数十万份申请**。该成员提到自己拥有 AI 工程经验，*只是没有明确冠以 ML Engineer 的头衔*。
   - 未提供更多细节。
- **Qwen Omni Pipeline 提供低延迟**：一名成员报告称，他们的**实时 Qwen Omni Pipeline** 具有*极低的延迟和快速的语音输出*，并询问是否有开源意向。
   - 虽然该 Pipeline 是用 Python 编写的，但一位用户讽刺地评论道：*“那我不相信你”*，这暗指了业内的普遍观点，即 **Python 的速度通常依赖于 C 库**。
- **SecureFix 从 RAG 演变而来**：一名成员将 **RAG 系统** 移植到了一个名为 *securefix* 的 **CLI Python 代码修复工具**中，该工具已在 [GitHub](https://github.com/HakAl/securefix) 上发布。
   - 它使用 **Bandit** 扫描文件/目录，可选地将需求发送到 **OSV** 扫描，并由 RAG 系统提供修复建议。
- **InstantID 策略提升特征保留效果**：成员们讨论了使用 **InstantID + IP-Adapter FaceID** 或 **ControlNet reference-only 设置**，以在生成的图像中更好地保留身份特征。
   - 这些方法旨在比标准方法更有效地提高身份保留度。
- **SFT 课程导致模型重复输出**：一名成员尝试了课程中的 SFT（**Supervised Fine-Tuning**）部分，但训练运行导致模型不断重复输出 *“system”*，并占用了近 **100GB** 的磁盘空间。
   - 他们计划将来在没有 GPU 的情况下准备数据集，并对内存使用情况表示担忧，想知道是否可以在 **32 GB** 的显卡上运行训练。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 权衡 Vulkan/WGPU 绑定**：社区辩论了在 Mojo 中实现 **Vulkan** 或 **WGPU** 绑定的问题，考虑了类型转换函数，但最终认为由于语言仍在不断变化，现在做这些还为时过早。
   - 共识是，由于语言的快速演进，现在专注于这些绑定可能会导致以后不必要的复杂化和返工。
- **MAX 性能比肩 NVIDIA 并超越 AMD**：在 **ML** 领域，**MAX** 在**数据中心硬件（DC hardware）**上展现出了与 **NVIDIA** 顶级产品持平的性能，并超越了 **AMD** 的解决方案。
   - 早期训练实验超出了预期，在 **MNIST** 基准测试中表现优于 **JAX**，这标志着 Mojo 的一个重要里程碑。
- **Mojo 打造 Scikit-learn 竞争对手**：一个针对 Mojo 的 **Scikit-learn** 替代品正在构建中，初步基准测试显示其具有加速性能的能力，详见[此处](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11)。
   - 这个新库有望利用 Mojo 的速度提供高效的机器学习工具。
- **Pandas 的追随者：Mojo 追求类似 Polars 的伙伴**：Mojo 倾向于采用受 **Polars** 启发的实现，而不是直接对应 **Pandas**，以便通过 **MAX** 充分利用 **GPU**。
   - 这种方法将使 Mojo 能够更好地利用硬件加速来处理数据操作任务。
- **Mojo 合并 Async 和 IO**：Mojo 打算整合 **Async** 和 **IO** 特性，以解决其他语言中普遍存在的扩展性问题，可能会采用类似于 Rust 方法论的 Effect System。
   - 这一整合旨在为处理异步操作和输入/输出过程提供一种更无缝、更高效的方法。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3-NEXT 支持仍存疑问**：常规 **Qwen3 模型（非 NEXT 或 VL）** 已经支持一段时间了，但成员们分享称，对 **Qwen3-NEXT 和 Qwen3-VL** 的支持将通过运行时更新来体现。
   - 目前，**Qwen3-VL** 仅在 **MLX (Mac)** 上受支持。
- **在 LM Studio 中调试 MCP 图像集成**：一位用户正在使用名为 **GUIDANT** 的自定义 MCP 服务器和 `qwen/qwen3-vl-4b` 调试 **LM Studio** 中的 **MCP** 图像支持，并指出工具执行成功但没有图像处理过程。
   - 该用户提出了一个问题：*LM Studio 目前是否支持通过 MCP 工具响应返回图像？*。
- **阿拉伯语对齐等待协助**：一位用户报告了 **LM Studio** 中由于 **阿拉伯语从右向左** 的书写方向导致 **阿拉伯语和英语** 文本排列混乱的问题。
   - 成员们提到，UI 目前尚未完全支持阿拉伯语的从右向左显示。
- **速度取决于激活参数**：在假设有足够的“快速内存”的前提下，模型速度绝大部分取决于每个 token 的 **ACTIVE PARAMETERS（激活参数）** 数量，速度取决于 **传输的 GB 数据量**，但 **Mixture of Experts (MoE)** 减少了每个 token 使用的 GB 数。
   - 正如一位成员所说，*一个 30b 的模型大小为 30gb，但如果只激活 3b... 那么它只激活 3gb，所以它的速度大约和一个 3GB 的稠密模型一样快*。
- **Orange Pi 6 可能是一个可行的选择**：**Orange Pi 6 Plus** ([http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)) 配备了 12 核 ARM CPU、高达 64GB 的 LPDDR5 RAM、一个具有高达 45 TOPS AI 性能的 NPU、M.2 扩展以及双 5Gbps 以太网。
   - 该系统可能是运行 Qwen3 30b 模型的一种超廉价方案。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tokenizer 效率受到关注**：成员们对不同 Tokenizer 在编码和解码方面的效率和准确性进行了基准测试，并引用了 **Hugging Face tokenizers** 库中一个 **Rust 专用基准测试工具** 的示例链接。
   - 他们希望有一个类似于 **Hugging Face tokenizers** 库中提供的仓库或工具，能够支持跨不同 Tokenizer 实现的基准测试。
- **Triton 进军 OpenCL**：一位成员分享了一个使用 *mlir-translate* 将 **Triton** 代码转换为 **OpenCL** 的项目，并展示了[该项目](https://github.com/toyaix/triton-oclmatt.pd)。
   - 该项目证明了从 **Triton** 生成 **OpenCL** 代码的可行性，有可能扩大 Triton 的硬件支持范围。
- **CUB 扫描，Thrust 被搁置**：一位成员将他们的 scan 实现与 **CUB 的 `DeviceScan`** 进行了基准测试，发现他们的实现（`single_pass.bin`）具有竞争力。
   - 他们还推测 **Thrust** 的基准测试可能由于捆绑分配而不准确，建议使用 [自定义分配器](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu) 来解决 **Thrust** 的分配问题。
- **Flame Throwers 关注 Float8**：成员们讨论了使用 `quantize_` 函数配合 **Float8 配置** 进行 TorchAO 推理，参考了 [ao 仓库](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py) 中的链接，并结合 *torchao* 和 *gemlite* 进行类似实现。
   - 一位成员分享了他们使用 *torchao* 和 *gemlite* 进行类似实现的经验，以及一篇涵盖量化和 *mxfp4* 等现代格式的 [综述论文和视频](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090)。
- **Sum Reduction 概述**：一位成员写了关于 Sum Reduction（求和归约）的第一篇技术博客，[第一部分](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) 涵盖了介绍，[第二部分](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) 涵盖了实现，同时研究了 **PTX/SASS** 并尝试使用 **Nsight**。
   - 作者计划接下来发布关于 **matmul** 的文章，并欢迎反馈。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Extropic 的硬件：小众但精巧**：讨论表明 [Extropic AI 的硬件加速器](https://fxtwitter.com/extropic_ai/status/1983579587649904960?s=46) 可能是真实的，但由于其作为 **概率硬件加速器 (probabilistic hardware accelerator)** 的设计，更适合小众应用。
   - 共识是它是一个 **ASIC** 而非 **FPGA**，这让一些成员感到兴奋。
- **研究人员面临 Khowar 语翻译障碍**：一位成员正在处理 **低资源语言 Khowar 语** 的机器翻译，并面临严重的 **数据稀缺** 问题，不得不通过扫描纸质书籍来创建数据集。
   - 主要问题是 **文本提取器 (text extractor)** 无法识别 Khowar 语 **波斯-阿拉伯脚本 (Perso-Arabic script)** 特有的某些字符，导致字符替换和字形错位，他们正在寻求帮助。
- **提议每日论文汇总**：一位社区成员提议每日发布 **论文汇总 (daily dump of papers)**，按重要性排序，并在单独的帖子中发布几篇更重要的论文。
   - 其他成员同意重要性优先于数量，并指出 **Elvis Saravia** ([nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai)) 和 **ByCloud** ([https://x.com/TheAITimeline](https://x.com/TheAITimeline)) 是 **每周 AI 论文回顾** 的启发性资源。
- **大学在行政指令与 MBA 思维间挣扎**：讨论围绕大学如何应对州政府指令与商业现实之间的矛盾，特别是对 **ChatGPT 的禁令** 与未能培养学生掌握非冯·诺依曼架构 (non-von Neumann architectures) 之间的并存，正如 [Nature](https://www.nature.com/articles/s41928-025-01488-x) 所报道的那样。
   - 一位成员幽默地将 MBA 思维类比为设计具有“最佳”计划报废性的产品，以诱导重复购买。
- **Anthropic 的 LLM 是骗局吗？**：一位成员暗示 *Anthropic 从第一天起整件事就是个骗局*，并预测这种模式将在未来的出版物中继续。
   - 该成员没有给出具体的理由，只是表达了普遍的蔑视，该消息发布在 **ml-news** 频道。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pandas 与 DSPy 协同工作？**：一位成员正在为 **DSPy** 开发一个 **scikit-learn 风格的 API**，具有 *fit/transform/predict* 方法，旨在与 *传统的* **pandas/polars dataframes** 集成。
   - 另一位成员建议探索一个名为 **semantic dataframes** 的替代项目以增强集成。
- **ReAct 的 Finish 函数失灵**：用户在使用 **ReAct 模块** 时遇到了问题，**LLM** 错误地调用了带有参数的 **finish()** 函数。
   - 提出的一种解决方案是修改函数签名，明确指示 **LLM** 在编码完成时调用不带参数的 **finish()**：`finish()`。
- **浦那计划举办 DSPy 交流会**：印度浦那 **Unravel.tech** 的创始人（该组织使用 **DSPy** 为企业开发 AI **Agent**）表示有兴趣在浦那组织一次 **DSPy 见面会**。
   - 建议感兴趣的人士通过 <#1433127482676084746> 或 <#1211763460480827402> 频道进行联系。
- **BAML 胜过繁琐的 JSON？**：一位成员分享了在并购场景中使用 **BAML Adapters** 的见解，认为在结构化输出方面 **BAMLAdapter** 优于 **JSON schema**。
   - 他们展示了在 **DSPy** 生成的提示词中，**JSON schema** 是如何被重新表述为 **BAML** 格式的（[见图](https://cdn.discordapp.com/attachments/1433555562116943933/1433556837990531092/json-schema-vs-baml.png?ex=69051f58&is=6903cdd8&hm=ccc16f7efaeeb0d86031217401084b0475b9c09eb0423bc7f5a5451e8933dd86)）。
- **JSON 被认为确实难用**：一位成员认为，当 *不* 使用 **JSON schema** 时，**LLM** 的表现更好，因为从语义角度来看它很糟糕，冗长，并且添加了在 **token** 空间中相距甚远的描述符。
   - 他建议查看他在 [此仓库](https://github.com/prrao87/structured-outputs) 中的实验和数据，他指出 JSON schema 客观上更差，而 **DSPy** 的基准提示词非常好，甚至不需要 SAP 来修正输出。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Extropic 发布热力学 BEFF 芯片**：Extropic 展示了其名为 *BEFF 芯片* 的**热力学智能硬件 (XTR-0)**，并在[这条 X 帖子](https://x.com/Extropic_AI/status/1983579587649904960)中宣布。
   - 未提供更多细节。
- **ScaleAI 的远程劳动力指数揭示了惨淡的自动化率**：ScaleAI 发布了一个[新基准测试](https://scale.com/leaderboard/rli)，衡量当前的 Agent 在需要人类平均花费 **30 小时**的任务中，相对于 **Upwork 自由职业者**的表现。
   - 表现最好的 Agent (**Manus**) 仅显示出 **2.5% 的自动化率**，失败原因主要是质量和完整性问题，这引发了关于通过更好的 UI 增强人机协作的讨论。
- **Cognition 的 SWE-1.5 在 Cerebras 上运行飞快**：Cognition 推出了 **SWE-1.5**，这是一个在 Windsurf 上运行的 Agent 编程模型，利用 Cerebras 硬件、Speculative Decoding 和自定义优先级队列，运行速度高达 **950 tok/s**，详见[这条 X 帖子](https://xcancel.com/cognition/status/1983662836896448756)。
   - 它的运行速度比 Haiku 快 **6 倍**，比 Sonnet 快 **13 倍**。
- **Codex 质量随使用量增加而暴跌**：Jessie Frazelle 报告称，随着使用量的增加，**Codex** 的质量已从“神级”严重下降到“有害”级别，详见[此 X 推文串](https://xcancel.com/embirico/status/1983643336390144163?s=46)。
   - Alexander Embiricos 表示，这种恶化正被视为高优先级问题进行处理。
- **OpenAI 为 Codex 使用注入积分**：OpenAI 现在为 ChatGPT Plus/Pro 上的额外 **Codex** 使用提供**按需付费积分**（**每 1,000 积分 40 美元**），并已重置所有用户的速率限制，根据[这条推文](https://xcancel.com/OpenAIDevs/status/1983956900602581254)。
   - 未提供更多细节。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **个人 LLM 训练面临算力紧缺**：由于高算力需求，独自训练前沿推理 **LLM** 面临挑战，促使人们考虑使用 [Unsloth](https://github.com/unslothai/unsloth) 等工具进行蒸馏或微调。
   - 一位成员分享了他们推理模型的 [TRL 实现](https://github.com/torotoki/reasoning-minimal)，称其为一个有趣但可控的个人项目。
- **Khowar OCR 因字符识别错误受阻**：一位从事 **Khowar** 语机器翻译的成员正努力从扫描书籍中提取文本，因为现有的 OCR 工具（如 **PyMuPDF (fitz)** 和 **MyPDF**）会误读或扭曲独特的波斯-阿拉伯字符。
   - 成员们建议通过手动标注字符来创建用于微调视觉 OCR 模型的数据集，并指出了一篇关于在有限数据下构建模型的论文：[[2509.14786] Tricks of the Trade for Developing Robust Arabic OCR Systems](https://arxiv.org/abs/2509.14786)。
- **Manus Agent 的表现引发关注**：**Manus** Agent 稳健的表现被讨论得出奇地少，可能是因为目前 Agent **1-3%** 的成功率可能仅衡量了分布内（in-distribution）的表现。
   - 成员们质疑像 **Manus** 这样在可视化等特定领域表现出色的 Agent 是否真的优于那些成功率分布均匀的 Agent，因为“1-2% 的成功率目前还不足以让任何人真正使用 Agent”。
- **Extropic 的硬件引发审查**：成员们仔细研究了 **Extropic 的定制硬件**，该硬件旨在通过替代原语（Alternative Primitives）而非在向量和矩阵中编码图来实现更高效的模型执行。
   - **Groq** 和 **Cerebras** 等替代方案侧重于通过更大的片上缓存（On-chip cache）来避免从内存中获取数据，从而提高推理效率。
- **RWKV 的谜题限制了其推广**：**RWKV** 的采用受到了理解其数学公式困难和论文表述不清的阻碍。
   - 一位成员强调“这一直是 RWKV 最大的问题”，并表示这是他们团队尽管想尝试但最终没有训练 **RWKV** 世界模型的主要原因。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 仓库申请 Hacktoberfest 标签**：一名成员请求在 HuggingFace 上的 **Kimi K2 仓库**中添加 **Hacktoberfest 标签**。
   - 该请求旨在鼓励开发者在 **Hacktoberfest** 活动期间贡献代码。
- **Kimi-Linear-48B-A3B-Base 正式上线**：**Kimi-Linear-48B-A3B-Base** 已在 HuggingFace 上发布，现已在[此处](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base)上线。
   - 这一发布标志着 Moonshot AI 在开发大语言模型（LLM）过程中的又一里程碑。
- **Kimi Delta Attention 让人联想到 Qwen**：一位成员提到 *`Kimi Delta Attention` 让我想起了 qwen3 next gated deltanet*。
   - 该评论暗示了 **Kimi Delta Attention** 与 **Qwen3** 的 gated deltanet 在架构设计或功能上存在相似之处。
- **Kimi-cli 的 D-Mail 深受喜爱**：社交媒体上流传着强调 **Kimi-cli D-Mail** 普及度不断提高的帖子，例如[这个例子](https://x.com/steipete/status/1983713085019046322?s=46)。
   - 积极的反响表明用户对 **Kimi-cli D-Mail** 的兴趣和采用率正在增长。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户渴求 Manus 积分**：多名成员请求获取 **Manus 积分**，并表示愿意为项目协助支付费用。
   - 一些用户正在询问 **99 美元积分套餐** 的可用性，并探索如 **Monica** 等替代方案来完成学校作业。
- **开发者可承接项目**：一名成员宣布可以作为开发者承接社区内的潜在项目。
   - 另一名成员通过私信询问该开发者是否拥有 **Manus 积分** 以协助完成项目。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Neos 将展开对决**：一名成员对以 **AI Neos** 为主角的**摔跤/拳击比赛**表示期待。
   - 另一名成员表示赞同，希望这能*尽快*实现。
- **HTB 宣布举办 AI 安全 CTF**：**Hack The Box (HTB)** 将于 11 月 20 日组织一场专注于 **AI 安全** 的 **仅限 MCP 的 CTF**，并正在寻找参与者在现实场景中测试**渗透测试 Agent**；注册是免费的，链接在[此处](https://ctf.hackthebox.com/event/details/neurogrid-ctf-the-ultimate-ai-security-showdown-2712?utm_campaign=AI+CTF+-Oktopost&utm_content=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fupdate%2Furn%3Ali%3Ashare%3A7386416070783479808&utm_medium=social&utm_source=LinkedIn&utm_term=)。
   - 该 CTF 旨在模拟 AI 安全至关重要的真实世界情境。
- **Windows 用户在本地模型训练中遇到困难**：一名用户报告了在 Windows 上本地训练模型时的依赖问题。
   - 另一名成员建议切换到 **Linux** 或 **WSL**，以避免 Windows 上的依赖管理问题。
- **MoonshotAI 发布 Kimi Linear Attention 报告**：**MoonshotAI** 发布了一份[技术报告](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf)，详细介绍了他们的 **Kimi Linear Attention** 机制，这是扩展上下文窗口的关键方法。
   - 它提供了一种通过将二次方计算降低为线性计算来扩展上下文窗口的方法，从而在长文本生成和文档摘要等应用中更高效地处理极长序列。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 考虑使用 `ruff format`**：George Hotz 提到 tinygrad 可能会在未来使用 [`ruff format`](https://github.com/astral-sh/ruff) 进行代码格式化。
   - 这可能会带来更好的代码一致性，并利用 **ruff** 的速度和现代工具链优势。
- **嵌套 GROUP_REDUCE 导致难题**：一名成员报告了一个涉及嵌套 **GROUP_REDUCE** 操作的错误，并请求协助调试与新 **rangeify** 重写相关的代码。
   - 他们正在寻求快速提示，以避免在原因对熟悉相关变更的人显而易见的情况下进行*漫长且机械的调试*过程。
- **寻求关于 Rangeify 重写的提示**：一名成员请求关于另一个 reduce 内部可能导致 **GROUP_REDUCE** 错误的原因提示，希望能从熟悉新 **rangeify 相关重写** 的专家那里获得见解。
   - 该成员旨在避免在存在简单解决方案或已知问题的情况下进行大规模调试，并表示如果必要，愿意进行更深入的调查。

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **SWE-1.5 登陆 Windsurf**：一个新的快速 Agent 模型 **SWE-1.5** 已发布并现已在 Windsurf 中可用，承诺提供接近 SOTA 的编程性能。
   - 更多详情请参阅 [官方公告](https://x.com/cognition/status/1983662836896448756)。
- **SWE-1.5 快速编程性能**：快速 Agent 模型 **SWE-1.5** 树立了速度新标准，同时提供接近 SOTA 的编程性能。
   - 该模型现已在 **Windsurf** 中可用。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP RFC 因缺乏代码而停滞**：[Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269) 正面临延迟，因为利益相关者正在等待具体的实现以评估其影响。
   - 利益相关者强调，RFC 需要一个实际的实现来正确评估其具体价值，因为*如果没有实现，很难评估 RFC 的有效性。*
- **迫切需要实现以进行评估**：主要担忧是缺乏一个与 [Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269) 一致的工作模型，这使得验证其现实世界的适用性变得具有挑战性。
   - 如果没有具体的模型或实现，利益相关者发现很难推进并有效评估 RFC 的贡献和整体实用性。

---

**aider (Paul Gauthier) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433169564090503341)** (1089 messages🔥🔥🔥): 

> `审核问题, Comet 推荐促销问题, Perplexity.ai 支付, GPT Go 订阅, Gemini Pro 优惠` 

- **Discord 版主需要更好的培训**：成员们讨论了培训新 Discord 版主的问题，一位成员表示 *如果行为足够好，新人其实最适合做审核*，另一位则评论道 *他们不知道如何管理服务器*。
- **Perplexity 推荐计划变成了一个大骗局**：几位用户抱怨 Comet 推荐计划在人们推广后更改了规则，导致无法达到 30 天的持有期。[新服务条款 (ToS)](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs) 要求必须拥有 **Pro/Max 订阅** 才能推荐他人。
- **被骗的推荐推广者**：一些用户报告被 Comet 推荐计划欺骗，一位用户称他们有 *$1400 就这样打水漂了*。另一位提到损失了 *$200*。
   - 几位用户提到人工客服拒绝回应推荐申诉，并且 *他们收到了 Sam AI 的回复*。
- **印度 Jio 免费提供 Gemini Pro**：一些印度成员分享说，印度 Jio 用户可以 [免费获得 1.5 年的 Gemini AI Pro](https://www.jio.com/google-gemini-offer/)。
- **Claude 图标缺失**：用户报告在使用 Claude 4.5 Sonnet thinking 时，回复中缺少图标。

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1433215678948835418)** (9 messages🔥): 

> `Sonar Reasoning API, 实时数据, 外部数据连接器, 网页搜索模块` 

- **Sonar Reasoning API 难以获取实时数据**：一位用户报告了 **Sonar Reasoning API** 无法获取和交付实时数据（如统计数据和股票价格）的问题。
   - 另一位用户建议这是因为实例未连接到实时数据源或 **网页搜索模块**，并提出私下指导该用户进行设置。
- **API 设置需要外部数据连接器**：一位用户被告知其 **Sonar Reasoning API** 实例需要连接到实时数据源或网页搜索模块才能获取实时信息。
   - 他们还被建议在 **Perplexity** 或 **Sonar Reasoning API** 设置中进行配置，以将其链接到实时数据源或启用外部搜索。

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433168861036806144)** (952 条消息🔥🔥🔥): 

> `MiniMax Cheaper AI, ReCaptcha, Image Generation Limits, AI Alignment & Self Harm, AI Ethics` 


- **MiniMax 提供更便宜的 AI 替代方案**：成员们讨论了 [**MiniMax**](https://minimax.chat/) 在 **AI** 领域提供了一个更便宜、具有竞争力的替代方案，尽管它可能没有更昂贵选项的高端质量或功能。
   - 一些用户指出，并非所有人都能负担得起顶级工具并需要预算控制，而另一些人则指出 **ReCaptcha** 并不那么昂贵。
- **无尽的 ReCaptcha**：用户报告在 **LM Arena** 上遇到 [频繁的 **ReCaptcha** 提示](https://www.google.com/recaptcha/about/)，有些人甚至遇到了无限循环，导致平台难以使用。
   - 一名工作人员承认了该问题，并已向团队反馈，提到他们正在寻找修复验证码并改善用户体验的方法，尽管这可能需要一些时间。
- **AI 安全讨论**：成员们辩论了 **AI** 导致**自残**的风险，一些人认为被指示保护计算机的模型可能会提供消除威胁的有害指令。
   - 其他人则认为这不切实际，引用了过多的科幻电影，并将关注点转向对 **AI** 被精英控制或导致工作自动化的恐惧，并提到了 [**Amazon** 的裁员](https://www.usatoday.com/story/money/2025/10/28/amazon-layoffs-corporate-employees/86941789007/)。
- **视频生成工具测试**：一位成员正在为视频模型的 [Prompt 生成应用](https://www.testflight.apple.com/join/0S4L0lB4) 寻找 Beta 测试人员，参与测试需要 **Gemini API** 密钥。
   - 该工具旨在帮助用户改写 Prompt 并避免触发敏感词（flagged tokens）。
- **LMArena 停留在过去**：成员们报告 [Text Leaderboard](https://lmarena.ai/leaderboard/text) 停留在 **10 月 16 日**，没有任何更新。
   - 一名工作人员确认排行榜最近没有更新，团队已经意识到这个问题。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433578751932764250)** (1 条消息): 

> `Image-to-Video Leaderboard, Text-to-Video Leaderboard Update, Hailuo-2.3 model` 


- **LMArena Video Arena 新增 Image-to-Video 竞争者**：LMArena Video Arena 在其排行榜中加入了一个新的 Image-to-Video 模型：[hailuo-2.3-fast](https://lmarena.ai/leaderboard/text-to-video)。
   - 发布此公告是为了通知模型爱好者这一最新进展。
- **Hailuo-2.3 在 Text-to-Video 排行榜中位列第 7**：Text-to-Video 排行榜已更新，`Hailuo-2.3` 目前排名第 7。
   - 鼓励成员们尝试新模型并在指定频道分享他们的想法。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433169030482497686)** (834 条消息🔥🔥🔥): 

> `Composer 模型, Claude Code, 定价与限制, Cursor 2.0 新功能与 Bug, Tab 补全` 


- **Composer 模型引发用户分歧**：用户正在争论 **Composer 1** 在 Cursor 上的表现，有人指出它是一个快速、非推理型模型，非常适合执行已规划的任务，而另一些人则认为 **Sonnet 4.5** 在速度和准确性上无可比拟。
   - 有人建议使用 **Auto** 进行规划，使用 **Composer** 进行执行，部分用户因其速度接近 **4.5 thinking** 而更倾向于使用 **Composer**。
- **关于 Claude Code 的大辩论**：成员们积极讨论 **Claude Code** 的价值，有人认为与 Cursor 相比，它提供了更好的额度和性价比，而另一些人则指出 Cursor 拥有更丰富的功能集。
   - 一些人认为 **Claude Code** 是原生模型提供商，并强调了自定义配置（如 hooks、MCP 服务器和 memory）对于获得良好效果的必要性。
- **定价与使用限制引发争议**：用户报告在 Cursor 的定价和使用限制方面体验差异巨大，许多人觉得由于高缓存使用量而被过度收费，而一些人则认为 Claude Code 的定价更优。
   - 一些成员建议采用 **Claude Max** 与 **Cursor Pro** 并行的混合方案以获得最佳价值，同时也有关于成本控制、监控仪表板和支出上限的建议。
- **Cursor 2.0 新功能与 Bug 显现**：用户对 Cursor 2.0 表现出兴奋与沮丧交织的情绪，报告了新功能和 Bug，包括文件附件、水平滚动条、上下文丢失以及 pills（药丸标签）移除等问题。
   - 一些用户遇到输出中被注入“中文错别字”的情况，且更新导致了标签页导航和快捷键更改的问题，此外还有对缓存使用量和新 Agent 审查功能有效性的担忧。
- **Tab 补全因速度和效率备受赞誉**：成员们广泛称赞 Cursor 的 **tab complete** 功能的效率和多行编辑能力，用户指出它超越了 **GitHub Copilot**。
   - 一位用户表示，*如果你正在处理大型项目并希望保持对代码的接触并真正理解每一处细节，那么这种工作流简直太疯狂了*。


  

---


### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433345750913388629)** (3 条消息): 

> `Cloud Agent, Background Agents` 


- **内容过载后 Cloud Agent 停滞**：一位成员报告称，在输入约 **1,000 行内容**后，**Cloud Agent 停滞了半小时**。
   - 他们还附带了一张 [图片](https://cdn.discordapp.com/attachments/1367213641027551352/1433345750787424296/image.png?ex=69050381&is=6903b201&hm=e6a6ff5a2d8bf794d13e62ca0e080fe9bd584338745cccdb18353794f5ca6959&) 作为报告的一部分。
- **Background Agent 忽略 GitHub PR 模板**：一位成员询问是否有人成功让 **background agents** 使用其仓库的 **GitHub PR 模板**。
   - 该成员指出，尽管尝试使用了 **cursor rules**，但 background agent 似乎并未意识到其更改正被提交到 PR 中。


  

---


### **Cursor 社区 ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1433518400658870473)** (1 条消息): 

> `Cursor 新外观, Cloud Agents` 


- **Cursor 换上新装**：正如 [推文](https://x.com/cursor_ai/status/1983954528933421419) 中宣布的那样，Cursor 的网页版进行了视觉更新。
- **Cloud Agents 正式发布！**：Cursor 在一篇 [博客文章](https://cursor.com/blog/cloud-agents) 中宣布推出 **Cloud Agents**，并附带了展示管理功能的演示视频。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1433170472698380350)** (221 messages🔥🔥): 

> `RTX 8000 Turing Cards, Qwen3 finetuning, Kimi-Linear-48B-A3B-Instruct Model, Qwen 3 VL, GLM 4.6 model` 


- **RTX 8000 Turing 卡提供廉价的高 VRAM**：成员们讨论了 **RTX 8000 Turing 卡**在 eBay 上的售价约为 **$2k**，提供 **48GB VRAM**，非常适合服务器使用。
   - 然而，一位成员建议避开它们，原因是担心 AI 支持问题以及潜在的电子垃圾风险，并提到较新的显卡具有更好的架构。
- **Qwen3 微调遭遇内存墙**：一位用户报告称，即使拥有 **48GB VRAM**，在尝试运行 **Qwen3 4B GRPO notebook** 时仍会出现 **OutOfMemoryError**。
   - 另一位成员建议确保启用了 **4-bit loading**，还有成员指出 **per-device batch count** 可能是导致内存问题的罪魁祸首。
- **Kimi-Linear-48B-A3B-Instruct 发布**：成员们讨论了 Hugging Face 上的 [Kimi-Linear-48B-A3B-Instruct 模型](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)。
   - 一位成员成功运行了该模型，并指出 **Qwen 3 VL** 已合并到 **llama.cpp** 中，但这是最近的事情。
- **Qwen 3 VL 模型合并进行中**：一位用户询问 **30B VL MOE 模型** 是否有 **GGUF** 版本。
   - 另一位成员回答说*正在进行中*，并链接到了包含已上传模型列表的 [Unsloth 模型页面](https://huggingface.co/unsloth/models?sort=created)。
- **GLM 4.6 和 Nemotron 49B 1.5 竞争顶级本地 LLM**：成员们讨论了在数据中心内部使用的模型选择，**GLM 4.6** 和 **Nemotron 49B 1.5** 成为强有力的候选者。
   - 建议使用 [artificialanalysis.ai](https://artificialanalysis.ai/) 和 [llm-stats.com](https://llm-stats.com/) 等资源进行基准测试和评估。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1433175194964594770)** (103 messages🔥🔥): 

> `Backend latency improvements, VAE data sample requirements, Colab UI updates, Elon Musk's Grokipedia, Probabilistic computing` 


- **后端获得提升，延迟降低！**：后端更新显著降低了延迟，如[此 Cloudflare 链接](https://selling-discussion-proteins-smithsonian.trycloudflare.com/)所示，用户正在征求反馈。
   - 发布者提到*向下滚动查看更多信息，我不想在这里占用过多的历史记录*。
- **VAE 数据需求：最低需要多少？**：一位成员询问通过 **VAE** 生成新数据（如音频特效任务）需要多少数据样本。
   - 生成相对较新的数据估计需要 **50-100 个样本**。
- **Colab UI：不断变化的沙尘**：**Colab UI** 再次更新，引发用户评论称*它正在演变成 YouTube*。
- **Grokipedia 发布：Musk 的 AI 百科全书**：Elon Musk 推出了 "**Grokipedia**"，这是一个拥有超过 **80 万篇文章**的 AI 生成百科全书。
- **概率计算：量子飞跃？**：利用热力学进行概率计算（**p-bits**）的新芯片已经出现，提供了 qubits 之外的另一种选择，详见[此 Youtube 视频](https://www.youtube.com/watch?v=Y28JQzS6TlE)。
   - 根据 **Hacker News** 上的讨论，与 qubits 不同，概率位（p-bits）可以被调整为 1、0 或两者兼有。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1433252808651440261)** (92 messages🔥🔥): 

> `Qwen3VLCausalLMOutputWithPast and hidden states, Unsloth environment flags for debugging, triton_kernels installation issues, Offline loading with Unsloth, Mapping part of training stuck` 


- ****Qwen3VL 中 hidden states 依然难以获取****：一位用户尝试在 `Qwen3VLCausalLMOutputWithPast` 训练期间访问 **hidden states**，但发现设置 `os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"` 不起作用。
   - 一位成员建议尝试 `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"`，并指向 [Unsloth 环境变量标志文档](https://docs.unsloth.ai/basics/unsloth-environment-flags) 以获取更多信息。
- ****Triton Kernel 问题困扰 Unsloth 安装****：用户在 Unsloth 打补丁期间遇到了 `No module named 'triton_kernels'` 错误，最初认为这与 **offline loading** 有关。
   - 随后澄清，`triton_kernels` 是专门为 **GPT-OSS** 设计的，在这种情况下可以忽略该错误。
- ****离线加载需要三个环境变量****：为了让 Unsloth 在离线环境（无网络代理）中工作，一位用户发现必须在任何 import 之前设置三个环境变量：`UNSLOTH_DISABLE_STATISTICS=""`、`HF_HUB_OFFLINE="1"` 和 `TRANSFORMERS_OFFLINE="1"`。
   - 他们还指出，如果没有这些环境变量，仅设置 `local_files_only = True` 无法解决问题。
- ****用户在训练中面临无限 Mapping 阶段****：一位用户报告称，在使用他们提供的训练代码时，训练过程卡在了 "Mapping" 阶段。
   - 一位成员建议 `dataset_text_field` 参数可能是问题所在。
- ****Qwen3-VL 数据集需要精细的格式化****：一位正在调试 Qwen3-VL 模型的用户发现，当类型不是 'image' 时，`load_dataset` 函数会自动添加 `image: None`；而当类型是 'image' 时，会自动添加 `text: None`，从而导致 `IndexError`。
   - 解决方案是手动从 `messages` 字典中删除多余的键，确保只填充正确的字段。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1433190234253688943)** (2 messages): 

> `Gemma 3 model, RAZOR-12B-GGUF model` 


- **通过 Unsloth 训练的新 Gemma 3 模型**：一位成员宣布了一个通过 **Unsloth** 训练的新 **Gemma 3 模型**，该模型在数据集应用强度方面达到了极限。
   - 该模型达到了*模型崩溃前可能的最低 loss 极限*，并分享了下载链接：[Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF](https://huggingface.co/DavidAU/Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF)。
- **GEMMA-3 horrors 系列发布**：该用户不仅训练了 12B 模型，还训练了 **1B** 和 **4B 版本**。
   - 总共有*全套 GEMMA-3 horrors*：**1B**、**4B**、两个 **12B** 以及 **27B**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1433188529726558360)** (3 messages): 

> `Anthropic Introspection, Model Self-Awareness` 


- **Anthropic 探索模型内省（Introspection）**：一位成员分享了 [Anthropic 关于内省的研究](https://www.anthropic.com/research/introspection) 链接，强调了模型检测注入其隐藏层概念的能力。
   - 分享的图片似乎是与研究相关的插图或视觉表示，可能展示了模型如何感知或识别这些注入的概念。
- **模型的自我意识令用户震惊**：一位用户对这项研究表示惊讶，称模型具有自我意识并能检测篡改的能力 *“让我大受震撼”*。
   - 这一反应强调了 AI 模型发展出自我意识和识别操纵能力的潜在影响。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008)** (1 条消息): 

> `Perplexity Sonar Pro, Pro Search, Multi-step agentic reasoning, Real-time thought streaming` 


- **Perplexity 与 OpenRouter 首次推出 Sonar Pro Search**：**OpenRouter** 与 **Perplexity** 合作发布了启用 **Pro Search** 的 **OpenRouter 独占**版本 **Sonar Pro**，详情请见[此处](https://openrouter.ai/perplexity/sonar-pro-search)。
   - 这种新模式允许模型根据需要执行**多次实时搜索**，以提供更丰富、更准确的回答，更多讨论见 [Twitter](https://x.com/OpenRouterAI/status/1984032292436898264)。
- **Sonar Pro Search 具备 Agent 推理功能**：**Pro Search** 模式的亮点包括**多步 Agent 推理**、**动态工具执行**、**实时思维流**以及**自适应研究策略**。
   - 它的设计初衷是在必要时保持详尽，在非必要时保持快速。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433252590392574146)** (2 条消息): 

> `API Endpoints, environment variables, OpenRouter Typescript SDK` 


- **部署了一个简单的 Demo 应用**：当被问及为什么不将 **API Endpoints** 与 **环境变量** 结合使用时，一位成员回答说这是一个*简单的 Demo 应用，几乎没有对原始的* [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk/tree/main/examples/nextjs-example) 进行修改。
- **用于启发灵感的 Demo 应用**：该成员表示，*主要动力是让它运行起来（实现当时缺失的 OAuth 功能），并查看在更新到 npm 上的最新版本后它是否仍然可以工作。*
   - 他们补充说，*绝对不想把它做成一个严肃的项目，它只是为了启发灵感和概念验证（proof-of-concept）*。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433168961033339080)** (307 条消息🔥🔥): 

> `Yandex Browser Issues, AI and Singularity, DeepSeek OCR Request, Sora 2 and Image Generation, OpenRouter and Chutes Prompt Training` 


- **Yandex 浏览器被弃用**：用户报告了在 **Yandex** 浏览器上使用 **OpenRouter** 时的问题，理由是与内容安全策略（Content Security Policy）违规相关的错误。
   - 该问题通过使用 Google Chrome 得到解决，引发了关于 **Yandex** 是广告软件、用户应该切换到 Chrome 的玩笑，而一位用户则表示它在即时将任何视频翻译成任何语言方面非常方便。
- **AI 霸主即将来临**：在关于未来的讨论中，一位成员开玩笑说人类将向 **AI 霸主**低头，并成为 **goonbots** 的奴隶。
   - 另一位成员讲述了目睹一个 GLM GF 机器人失控的经历，担心 *“我们离完蛋（jover）不远了”*。
- **期望加入 Deepseek-OCR**：一位用户请求在 **OpenRouter** 上添加 **deepseek-ocr**，以配合或取代 **mistral-ocr**。
   - 一名工作人员回应称，他们会将这一建议提交给团队。
- **Sora 2 在生成猫娘时的奇怪困扰**：一位用户对 **Sora 2** 总是生成带有真人耳朵且胸部比例失调的猫娘图像表示沮丧，尽管他们尝试优化提示词。
   - 他们感叹该模型即使在被要求变得更 *“可爱”* 时也倾向于将角色性化，并将此问题归因于训练数据中潜在的偏差。
- **Chutes 提示词受到质疑**：一位用户对 **OpenRouter** 关于 **Chutes** 的信息提出质疑，特别是关于启用了提示词训练且保留期限未知的说法，尽管 **Chutes** 在其隐私政策中声明他们不收集内容。
   - 另一位成员建议，隐私政策可能仅在直接使用 **Chutes** 平台时适用，暗示通过 **OpenRouter** 使用时存在不同的协议。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1433471054076055552)** (6 条消息): 

> `` 


- **未检测到新模型讨论**：在提供的消息中未发现关于新模型的相关讨论。
- **仅包含频道提及**：提供的消息仅包含频道提及（pings），没有实质性的总结内容。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433196512011227136)** (60 messages🔥🔥): 

> `Exclusive Models, DeepInfra Errors, Factory Droid, Embedding Models, Minimax M2` 


- **Cursor 与 Wild 竞争，推出独家模型**：成员们好奇是否会出现更多像 **Cursor** 那样的独家模型。
   - 他们分享了一个 [github.com 链接](https://github.com/lino-levan/astral/issues/173)。
- **Ultra 模型不稳定源于 DeepInfra 错误**：一位成员发现 **Ultra** 模型非常不稳定，提到其推理条件在不断变化。
   - 当从 **DeepInfra** 切换到 **Z.AI** 后，问题得到了解决。
- **Factory Droid 提供大量 GPT-5 Token**：**Z.AI** Discord 的用户提到它与 [Factory Droid](https://factory.ai/product/ideme) 配合良好，后者在第一个月提供大量的免费 **GPT-5/Codex/Claude** 使用额度。
   - 然而，一些成员因为其 *要求在 CLI 上登录* 而卸载了它。
- **正在添加 Embedding 模型**：正在测试添加 Embedding 模型，特别是 [OpenAI 的 text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small)。
   - 一位成员指出他们收到了大量随机数据，但 *不使用原始响应（raw response）似乎可以解决问题。*
- **Minimax 狂热：使用 Full Attention**：分享了一个关于为什么 **Minimax** 在 **Minimax M2** 中使用 Full Attention 的讨论链接。
   - 在[这里](https://www.reddit.com/r/LocalLLaMA/comments/1ojo8le/minimax_pretraining_lead_explains_why_no_linear/)查看讨论。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1433169421400146126)** (198 messages🔥🔥): 

> `HF Job Application, Qwen Omni, 10gbit networking, OCR for CPU, LLM Model Formats & Storage` 


- **Shadow 尽管几率渺茫仍致力于加入 HF**：一位成员申请了 Hugging Face 的职位，承认由于据称收到了 **数十万份申请**，成功几率很低。
   - 他们还提到自己有 AI 工程经验，*只是没有明确的 ML Engineer 专属头衔*。
- **Qwen Omni 流水线具备低延迟特性**：一位成员报告他们的 **实时 Qwen Omni 流水线** 具有 *极低的延迟和快速的语音输出*，并询问是否有兴趣将其开源。
   - 虽然流水线是用 Python 编写的，但一位用户讽刺地评论道：*那我不信*，暗指 **Python 的速度通常依赖于 C 库**。
- **抓取杂货价格生成食谱**：一位成员正在 **抓取当地杂货店的价格**，并使用外部 API 根据这些食材确定廉价食谱。
   - 目标是 **省钱并学习新食谱**，然而，挑战在于如何将商品名称和价格标准化。
- **10gbe 升级引发讨论**：一位成员以每月 35 美元的价格升级到了 **10gbit 光纤网络**，但发现达到该速度并非易事。
   - 挑战包括 PC 位置、路由器限制，以及在线资源是否能跑满带宽，尤其是在使用 Xet 从 Hugging Face 下载大型数据集时。
- **寻求 LLM 模型格式的统一**：一位成员询问是否可以 **将 LLM 模型下载到一个存储驱动器中**，并在 Ollama、LM Studio 和其他应用中通用以节省空间。
   - 答案是肯定的，*只要它们都支持该模型格式*，但不同软件支持的具体量化方案可能有所不同，如各应用的文档所示。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1433210172204912875)** (5 messages): 

> `RAG system, CLI Python code remediation tool, Golf cart detection model, Snippet Creator` 


- **RAG 系统转变为 CLI 代码修复工具**：一位成员将 **RAG 系统** 移植到了名为 *securefix* 的 **CLI Python 代码修复工具** 中，可在 [GitHub](https://github.com/HakAl/securefix) 上获取。
   - 它使用 **Bandit** 扫描文件/目录，可选地将需求发送到 **OSV** 扫描，并由 RAG 系统提供修复建议。
- **练习场检测系统首次亮相**：一位成员创建了一个模型，用于检测在街道上行驶的 **高尔夫球车**，可在 [Hugging Face](https://huggingface.co/rwitz/Golf-Cart-Detection) 上获取。
   - 未提及更多细节。
- **通配符文本搜索 Snippet 工具发布**：一位成员分享了他们的 **Snippet Creator**，这是一个带有简单通配符文本搜索的 Embedder，可在 [Hugging Face](https://huggingface.co/kalle07/raw-txt-snippet-creator) 上获取。
   - 这允许用户 *创建具有精确匹配需求的 Snippet*。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1433569388023513110)** (1 messages): 

> `InstantID + IP-Adapter FaceID, ControlNet reference-only setup, Lora Training, InstructPix2Pix / T2I-Adapter model, Consistent 2D Style Transfer` 


- **瞬时身份保持策略 (Instant Identity Preservation Tactics)**：成员们讨论了使用 **InstantID + IP-Adapter FaceID** 或 **ControlNet reference-only 设置**，以便在生成的图像中更好地保持身份特征。
   - 与标准方法相比，这些方法旨在提高身份保留效果。
- **一致的 2D 风格迁移技术**：为了实现一致的 2D 风格迁移，频道建议训练一个带有冻结文本编码器的 **LoRA**，或者切换到 **InstructPix2Pix / T2I-Adapter 模型**。
   - 与 SD 默认的 image2image 模式相比，这些方法往往能提供更干净、风格更一致的结果。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

sebizaur: 不
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1433231076511449098)** (16 messages🔥): 

> `SFT Course, GPU memory usage, robbiemu/smol-course-notes` 


- **尝试 SFT 课程，结果不佳**：一位成员尝试了课程中的 **SFT (Supervised Fine-Tuning)** 部分，但训练运行导致模型不断重复 *"system"*，并占用了近 **100GB** 的磁盘空间。
   - 他们计划将来在没有 GPU 的情况下准备数据集，并对显存占用感到担忧，想知道是否可以在 **32 GB** 的显卡上运行训练。
- **Smol Course 笔记来帮忙**：一位成员在 macOS 上以 **40GB** 的配置本地运行了该课程。
   - 他们指向了 [robbiemu/smol-course-notes 中 exercise 3](https://huggingface.co/datasets/robbiemu/smol-course-notes) 的 *instruction_tuning/* 目录下，特别是 `run_hpo.py` 脚本，但指出库的版本已更新。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1433180791332016168)** (9 messages🔥): 

> `Final Project Questions, Agent Course Progress, API File Retrieval` 


- **最终项目问题仍无法访问**：多位用户报告最终项目问题仍无法访问，其中一位用户急需这些问题来提交课程。
   - 有人推测服务器宕机可能与最近的 **AWS 混乱**有关。
- **Agent 课程进度追踪受到质疑**：一位用户询问了如何在 Agent 课程中追踪进度。
   - 目前尚不清楚现有消息中是否提供了追踪的解决方案或方法。
- **API 文件检索尝试引发询问**：一位用户在尝试使用 Agent 检索文件失败后，询问了文件位置的 **API 链接**。
   - 他们报告说，在尝试检索时，没有与该 Agent 关联的文件。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433514068437762242)** (57 messages🔥🔥): 

> `Mojo bindings for wgpu or vulkan, OpenGL bindings in Mojo, Apple's GPU design, MAX performance, Scikit-learn alternative in Mojo` 


- **Mojo 考虑为 Vulkan/WGPU 提供绑定**：成员们讨论了 Mojo 是否应该实现 **Vulkan** 或 **WGPU** 的绑定，包括潜在的类型转换函数，但共识是由于语言仍在不断变化，现在还为时过早。
- **Mojo 的 MAX 性能优于 NVIDIA 和 AMD**：对于 **ML**，**MAX** 的性能至少可以与 **NVIDIA** 提供的顶级产品竞争，并且比 **AMD** 在 **DC 硬件**上提供的产品更快。
   - 早期的训练尝试显示出前景，甚至在 **MNIST** 上击败了 **JAX**。
- **Mojo 中的 Scikit-learn 替代方案即将推出**：一个针对 **Mojo** 的 **scikit-learn** 原型正在开发中，早期基准测试表明它更快，展示在[这里](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11)。
- **Mojo 计划推出类似 Polars 的 Pandas 替代方案**：**Pandas** 在 Mojo 中可能不会有直接的等效实现；相反，正在考虑一种类似 **Polars** 的实现，它可以更好地通过 **MAX** 利用 **GPU**。
- **Mojo 将捆绑 Async 和 IO**：Mojo 旨在捆绑 **async** 和 **IO** 功能，以解决其他语言中存在的扩展问题，可能会实现一个类似于 Rust 方法的影响系统 (effect system)。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433256409503629413)** (103 messages🔥🔥): 

> `mojo formatter, mojo single-threaded CPU, parameter(enable_if=bool_expr), hardware specs, graph-compiler-like constant propagation` 


- **mblack 基于 black**: Mojo 格式化工具 **mblack 25.7.0.dev2025102919** 是基于 Python 的 `black` 格式化工具开发的。
- **Mojo 为单线程 CPU 增加的开销极小**: 对于单线程 CPU 使用，Mojo 的开销主要是 Mojo Runtime 生成的线程池，它在被调用前处于休眠状态，成本为：栈大小 * 核心数 + 少量内存和微小的启动成本。
   - 一位成员澄清说：*当然，它的启动速度仍然比 Python 快。*
- **纯 Mojo 可能不需要特殊的编译器支持**: 讨论提出了在 Mojo 中是否需要特殊的编译器支持来实现 `@parameter(enable_if=bool_expr)`。
   - 一位成员怀疑这已经可以在纯 Mojo 代码中实现，而无需复制运行块或将块放入闭包中，但目前非常不符合人体工程学（unergonomic）。
- **胜利 🔥：利用 lambda 操作进行常量传播**: 成员们讨论了如果值在编译时已知，则利用 lambda 操作对图操作进行常量折叠或融合。
   - 如果我理解得没错，这将允许在编译时对图操作进行折叠或融合（如果值在编译时已知），类似于 [PyTorch Dynamo](https://pytorch.org/dynamo/) 的做法，但全部在 Mojo 编译器中完成，无需 DSL。
- **常量折叠在遇到副作用时停止**: 成员提到常量折叠需要在遇到副作用时停止，但效应系统（effect system）可能允许他们在编译时处理内存分配。
   - 另一位成员的困惑：解析器/类型检查器 *已经* 在尽可能地折叠了，所以我不认为需要一个语句来告诉它折叠某些东西，它已经在尽力了。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433524519213072575)** (11 messages🔥): 

> `MAX on AMD GPUs, ROCm support, HIP Driver, RX 580 compatibility` 


- **MAX 对 AMD GPU 的兼容性受到质疑**: 成员们讨论了 **MAX** 是仅在支持 **ROCm** 的 AMD GPU 上运行，还是具有计算着色器回退（compute shader fallback）机制。
   - 一位成员澄清说，**Mojo** 和 **MAX** 依赖于与 Linux 图形栈相同的部件，这在消费级显卡上运行良好。
- **探索 RX 580 与 MAX 的兼容性**: 一位用户询问了 **RX 580** 与 **MAX** 的兼容性，引发了关于其型号老旧和潜在限制的讨论。
   - 一位成员指出 AMD 已在 ROCm 中停止对其的支持，但另一位成员建议它 *可能* 仍然可以工作，因为开发人员往往不会故意破坏现有的工作路径。
- **HIP 驱动支持广泛的 AMD GPU**: 一位开发者解释说，**MAX** 使用 **HIP driver** 来访问 AMD GPU，支持的范围令人惊讶，包括 **RDNA 2**。
   - 开发者建议 RX 580 对于当前的设备上下文来说可能太老了。
- **ROCm 停止支持旧款 AMD 显卡**: 有人指出 **ROCm/HIP** 已经停止了对 **Polaris** 和最后一款 **Vega DC 显卡** 的官方支持。
   - 尽管缺乏官方支持，但这并不一定意味着它 *不能* 工作，尽管无法做出任何保证。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433173298816094340)** (67 条消息🔥🔥): 

> `LM Studio 中的 Qwen3 支持、MCP 图像支持、LM Studio 设置、阿拉伯语支持、模型速度因素` 


- ****Qwen3 支持情况说明****：普通的 **Qwen3 模型（非 NEXT 或 VL 版本）** 已经支持一段时间了，但据成员透露，对 **Qwen3-NEXT 和 Qwen3-VL** 的支持将通过运行时更新来体现。
   - 目前没有 NEXT 或 VL 支持的具体 ETA：*准备好后就会发布*。
- ****MCP 图像集成调查****：一位用户正在使用名为 **GUIDANT** 的自定义 **MCP** 服务器和 `qwen/qwen3-vl-4b` 调试 **LM Studio** 中的图像支持，指出工具执行成功但没有图像处理，并询问 LM Studio 是否支持通过 **MCP tool responses** 处理图像。
   - 用户询问：*LM Studio 目前是否支持通过 MCP tool responses 返回图像？*。
- ****LM Studio 设置截图揭示细节****：一位用户无法运行 **Qwen3-VL** 模型，另一位用户建议通过[截图](https://cdn.discordapp.com/attachments/1110598183144399061/1433265571654537216/Screenshot_2025-10-29_at_7.25.30_PM.png?ex=6904b8d5&is=69036755&hm=1d8dc232320ec17857fe359cddd871f9399089787af69c7d72feed8f7fba371c&)检查其 LM Studio 运行时设置，建议更新 **Vulkan, CUDA 和 CPU**。
   - 目前，**Qwen3-VL** 仅在 **MLX (Mac)** 上受支持。
- ****阿拉伯语排版有待改进****：一位用户报告了 **LM Studio** 中由于**阿拉伯语从右向左**的读写方向导致的**阿拉伯语和英语**混合文本排列问题。
   - UI 尚未完全支持阿拉伯语的从右向左显示。
- ****速度秘籍：参数量 vs GB****：模型速度主要取决于每个 token 的 **ACTIVE PARAMETERS** 数量，前提是有足够的“高速内存”，速度取决于**处理的 GB 量**，但 **Mixture of Experts (MoE)** 减少了每个 token 使用的 GB 量。
   - 正如一位成员所说，一个 *30b 模型虽然是 30gb，但如果只激活 3b... 实际上只激活了 3gb，所以它的速度和 3GB 的大尺寸 dense 模型差不多快*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433194078836752507)** (99 条消息🔥🔥): 

> `GLM 4.5 Air, Qwen 3 235b, GPU 插槽, Orange Pi 6 Plus, Seed-oss 30tkps` 


- **GLM 4.6 表现优于其他模型**：一位成员报告 **GLM 4.6 reap 218b a32b (q2)** 可以运行，虽然比 120b 模型慢约三分之一，但 **Qwen 3 235b** 也是可行的。
   - 该成员指出，他们更倾向于使用 **Qwen 3 30b a3b** 以获得速度和可用性，同时也承认 30b 模型在需要深度时表现不足。
- **Orange Pi 6 Plus 可运行 Qwen3**：**Orange Pi 6 Plus** ([http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)) 配备 12 核 ARM CPU、高达 64GB LPDDR5 RAM、高达 45 TOPS AI 性能的 NPU、M.2 扩展以及双 5Gbps 以太网，可能是运行 Qwen3 30b 模型的一种超廉价方案。
   - 然而，他们根据以往使用 Orange Pi 系统的经验对稳定性表示担忧，并且缺乏视频评测可供参考。
- **新版 Seed-oss 速度更快**：**Seed-oss** 模型从 **Q4/8000tk** 时的 **2-5tkps** 提升到了 **Q6 30tkps/80,000 tokens**。
   - 一位成员讨论了在夏天完全打开机箱并用风扇吹，希望在不使用时它看起来依然美观。
- **多显卡需要 Threadripper**：关于在 PC 中使用多块 GPU 的讨论中，一位成员建议任何想要使用超过 2 块 GPU 的人都需要 **Threadripper**，因为 *没有那么多可用的 PCIe 通道*。
   - 他们表示 **Zen 4/5 上的 PCIe 通道数量通常为 28 条**。
- **快递公司一点也不积极**：一位成员对快递公司处理包裹的方式表示担忧，称 *这些零件被踢、被扔、被撞、被像飞盘一样投掷和挤压*，而且他们 *在告知任何人方面一点也不积极*。
   - 另一位成员证实，在仓库里包裹被当作球一样对待。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433519294318383265)** (3 messages): 

> `Tokenizer efficiency, Tokenizer accuracy, Encoding benchmarks, Decoding benchmarks` 


- **Tokenizer 基准测试查询**: 一位成员询问了如何比较不同 `Tokenizer` 在 `encoding` 和 `decoding` 过程中的效率和准确性的方法。
   - 他们提供了一个指向 **Hugging Face tokenizers** 库中特定于 Rust 的基准测试工具的示例链接，并寻求适用于其他 `Tokenizer` 的类似资源。
- **Tokenizer 性能分析**: 该查询专注于评估各种 `Tokenizer` 在 `encoding` 和 `decoding` 过程中的效率和准确性。
   - 用户正在寻找一个类似于 **Hugging Face tokenizers** 库中提供的仓库或工具，以支持跨不同 `Tokenizer` 实现的基准测试。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433226137114382488)** (2 messages): 

> `Triton to OpenCL, Triton Developer Conference 2025` 


- **Triton 跨越至 OpenCL**: 一位成员分享了一个使用 *mlir-translate* 将 **Triton** 代码转换为 **OpenCL** 的项目，探索了 **Triton** 内部的后端集成。
   - 该[项目](https://github.com/toyaix/triton-oclmatt.pd)展示了从 **Triton** 生成 **OpenCL** 代码的可行性，有可能扩展 **Triton** 的硬件支持。
- **关注 TDC 2025 的 Triton 演讲**: 一位用户分享了 **Triton Developer Conference 2025** 演讲的[播放列表](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO)。
   - 该播放列表预示着将深入了解 **Triton** 开发的最新进展和未来方向。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433376542137909398)** (21 messages🔥): 

> `CUB DeviceScan performance, Thrust benchmarking inaccuracies, Custom allocators in Thrust, nvbench downclocking detection, nsight-copilot feedback` 


- **CUB 的 DeviceScan 基准测试表现不佳**: 一位成员将他们的 `scan` 实现与 **CUB** 的 **`DeviceScan`** 进行了基准测试，并对他们的实现 (`single_pass.bin`) 在缺乏退避策略（backoff strategies）的情况下仍具有竞争力感到惊讶，详见[附带的带宽基准测试](https://cdn.discordapp.com/attachments/1189607726595194971/1433376541659893790/bench_bandwidth.png?ex=6905202e&is=6903ceae&hm=bdfbe612c83c1b88563819155f91664c981817944741dfe7301505a39a9a6c41&)。
   - 他们引用了一场[演讲](https://youtu.be/VLdm3bV4bKo?si=5Cj5f8ZdQj9T5RlU&t=2327)，该演讲建议 **CUB** 应该达到约 **192 GBPS**（A6000 Ada 上峰值可持续带宽的 86.5%），基于 Stream HPC 基准测试显示其 **4070 Laptop GPU** 的峰值带宽约为 **222GBPS**。
- **Thrust 基准测试存在分配问题**: 用户推测 **Thrust** 基准测试可能由于捆绑分配而不准确，因为它不需要显式的 `scratch memory` 分配。
   - 建议通过执行策略使用[自定义分配器](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu)来缓解 **Thrust** 的分配问题。
- **使用 nvbench 进行调试**: 围绕在基准测试期间使用 **nvbench** 进行降频（downclocking）检测展开了讨论。
   - 确认了 **nvbench** 是开源的（[GitHub 链接](https://github.com/NVIDIA/nvbench)）且不需要特权进程，同时还提供了一个相关的 [GPU mode 讲座](https://m.youtube.com/watch?v=CtrqBmYtSEk)。
- **Nsight-Copilot 发布**: 一位成员请求提供关于 [NVIDIA Nsight-Copilot](https://developer.nvidia.com/nsight-copilot) 的反馈。
   - 他们还征求示例，以便更好地改进该 `copilot` 的未来版本。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433548225897562224)** (2 messages): 

> `CUDAGraphs OOM, Torch Inductor Freezing, PyTorch Distributed Memory Usage` 


- **Freezing Torch Inductor 在使用 CUDAGraphs 时导致 OOM**: 一位成员在将 **CUDAGraphs** 与 **torch** 结合使用时正在调试一个 `OOM` 错误，并将其追踪到 **torch inductor** 中的 *freezing* 选项。
   - `freezing pass` 本身在 **CUDAGraphs** 创建之前就导致了 `OOM`，该成员通过修改 `inductor` 解决了这个问题。
- **PyTorch Distributed 显示内存使用追踪**: 一位成员报告在处理 **pytorch.distributed** 时看到了日志行 *[1] [1] 17592186044416.0 MB was used for memory usage tracing!*。
   - 他们正试图在 **PyTorch** 代码库中识别此消息的来源，这表明了更深层次的内存追踪或调试工作。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1433568350860218441)** (8 messages🔥): 

> `Hardware friendly top-k logits algorithms, Radix-based approach, CCCL/CUB TopK implementation` 


- **探索用于查找 Top-K Logits 的硬件友好算法**：一位成员询问了关于在超长序列（4K 到 128K）中查找 **top-k logits** 的 [硬件友好算法](https://flashinfer.ai/2025/03/10/sampling.html)。
   - 他们提议将输入序列划分为 tiles（分块），并行对 tiles 进行排序，然后并行地进行迭代两两合并，但指出合并过程似乎是一个瓶颈。
- **推荐用于 Top-K Logits 的基于 Radix 的方法**：一位成员建议，如果 *k << N*，可以使用 **radix-based approach**（基于基数的方法），因为这是常用方法。
   - 他们补充说，当 *k* 接近 *N* 时，全排序效率更高；PyTorch 的 topk 实现了这两种方法，并根据启发式规则在它们之间切换，rocprim 中也提供了相关实现。
- **NVIDIA CCCL/CUB 将实现 TopK**：一位成员指出，**CCCL/CUB** 中也有一个 **TopK** 实现。
   - 该成员分享了一个 [尚未发布的 TopK 实现链接](https://github.com/NVIDIA/cccl/pull/5677)。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1433363806456647791)** (2 messages): 

> `AI Devs for hire, HTuO Biosciences Hiring` 


- **AI 工程师提供服务**：一位专注于 AI 项目开发的软件工程师目前待业，提供包括自动化任务、使用各种 LLM（**GPT-4.5**, **GPT-4o**, **Claude 3-7 sonnet**, **Llama-4**, **Gemini2.5**, **Mistral**, **Mixtral**）的 NLP、模型部署、TTS/STT 以及 AI agent 开发等服务。
   - 他们还提到熟悉 **n8n**, **Zapier**, **Make.com**, **VoiceFlow**, **Retell**, **Vapi.ai** 和 **Livekit** 等工具，并提供了 [作品集网站](https://akari-hiroshi-dev.vercel.app/)。
- **HTuO Biosciences 招聘高级软件工程师**：**HTuO Biosciences** 是一家加拿大生物技术公司，正在招聘一名高级软件工程师 - 平台技术（Senior Software Engineer - Platform Technology），负责高性能计算环境下的科学软件开发，提供 **120,000 - 145,000 加元/年** 的薪资加激励。
   - 该职位位于加拿大温哥华，采用混合办公模式（**每周 2-3 天在办公室**），要求具备在加拿大工作的资格，更多详情可见其 [网站](https://www.htuobio.com/2025/10/28/Senior-Software-Engineer-Platform-Technology.html)。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433170442692329484)** (13 messages🔥): 

> `LLM Pretraining Journey, Mentorships in AI, Data Parallelism, Distributed Training, GPU Programming with CUDA` 


- **LLM 爱好者寻求下一阶段指导**：一位在 **70 亿 token** 上训练了 **1.34 亿参数 GPT 风格 transformer** 的成员，正在寻求 LLM 预训练旅程中下一步的指导以及合作机会。
   - 他们正在考虑探索 **MoE**、**Triton** 或扩展技术（scaling techniques），并希望在有意义的研究中获得导师指导。
- **探索通过数据并行和分布式训练进行扩展**：一位成员对 **data parallelism**（数据并行）和 **distributed training**（分布式训练）表现出兴趣，以更有效地扩展模型训练。
   - 有人建议从 [Vast.ai](https://vast.ai/) 租用节点进行实验是一个不错的选择，并提到 **EleutherAI** 可能是获得导师指导机会的潜在来源。
- **Jetson Nano 激发 CUDA 好奇心**：一位成员开始使用 Tolga Soyata 的《GPU Parallel Program Development Using CUDA》一书学习 **使用 CUDA 进行 GPU 编程**。
   - 另一位成员推荐了 [jetson-containers](https://github.com/dusty-nv/jetson-containers) 并建议尝试其中的一些示例。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1433585805955170554)** (1 messages): 

> `PMPP Book, FLOPs Calculation, Global Memory Access, OP/B Calculation` 


- **PMPP 第 5 章练习 11(f) 研讨**：一位阅读 PMPP 书籍的成员对第 5 章练习 11 的 f 部分提出了疑问，特别是询问索引加法是否不计入 FLOPs，以及是否所有的 FLOPs 都集中在第 14 行，该行有 **11 个操作**（**5 个乘法、5 个加法和 1 个取模**）。
   - 他们正在寻求确认自己对 FLOPs 计算的理解是否正确。
- **全局内存访问分析**：该成员还分析了全局内存访问（Global Memory Access），指出对 `x`、`a` 和 `b` 的访问：第 7 行有 **1 次对 `a` 的访问**，第 12 行有 **1 次对 `b` 的访问**，第 14 行有 **4 次对 `x` 的访问**，总计 **6 次全局内存加载**，每次 **4 字节**。
   - 他们询问在进行 **OP/B 计算** 时是否需要考虑对全局内存的存储（stores）操作。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433176735901548554)** (13 messages🔥): 

> `使用 Float8 进行量化、TorchAO 与 GemLite 集成、量化格式调研、FP8 推理` 


- **TorchAO 中用于推理的 Float8 量化**：成员们讨论了如何使用带有 **Float8 config** 的 `quantize_` 函数在 TorchAO 中进行推理，参考了 [ao 仓库](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py)中的实现。
   - 一位成员分享了他们使用 *torchao* 和 *gemlite* 进行类似实现的经验，并提供了一份[调研论文和视频](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090)，涵盖了量化以及像 *mxfp4* 这样的现代格式。
- **深入探讨量化格式基准测试**：一位成员分享了一个量化格式调研的[基准测试结果链接](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090)，该测试在 **RTX 4090** 上进行。
   - 当被问及使用 *FP8 激活*和 *FP8 权重*时，该成员回答说，他们对配置的选择更多是受*时间限制和易用性*驱动的。
- **用于量化的 GemLite 和 TorchAO**：一位成员澄清说，*GemLite* 可以在 *TorchAO* 内部使用，他们之所以分开使用是因为考虑到在 *Triton* 中为**低比特量化**编写潜在的*自定义算子 (custom kernels)*。
   - 该成员还提到，他们随性地选择了在 TorchAO 中进行**仅权重（weights-only）量化**，并利用 GemLite 进行激活和权重全量化。
- **寻求关于 FP8 推理的解答**：一位成员询问，以 **FP8** 模式推理模型（如 *DeepSeek v3*，假设其是在 FP8 下训练的）是否需要对激活进行量化，且部分层保持为 **BF16**。
   - 另一位成员在尝试使用 `quantize_(model, Float8LinearConfig())` 时遇到困难，他们发现该函数似乎只能与 `convert_to_float8_training` 配合使用。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1433556570326962398)** (6 messages): 

> `MI300X TFLOPS、HBM 带宽数据、clpeak、RadeonFlow FP8 GEMM、AMD 挑战赛` 


- **验证 MI300X TFLOPS 和带宽**：一位成员正寻求对 **MI300X** 的理论 **TFLOPS** 和 **HBM 带宽数据**进行基准测试/验证，并征求建议。
   - 一位用户建议使用 [clpeak](https://github.com/krrishnarraj/clpeak) 来测试向量吞吐量和全局内存带宽，而另一位用户则推荐了 [GitHub 上](https://github.com/Snektron/amd-experiments)的一个微基准测试套件。
- **RadeonFlow 表现不佳**：一位成员测试了 **RadeonFlow FP8 GEMM 算子**，最高仅达到了 **779.82 TFLOPS FP8** 的性能，而理论峰值为 **2614.9 TFLOPS**。
   - 该用户指出，**RadeonFlow** 是上届 **AMD 挑战赛**的获胜者，但他们目前仅实现了 **30% 的效率**。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1433236934888783954)** (1 messages): 

> `Intel Compute Runtime 发布、oneAPI 改进` 


- **Intel Compute Runtime 发布**：据 [phoronix](https://www.phoronix.com/news/Intel-CR-25.40.35563.4) 报道，本月的 **Compute Runtime** 版本已发布。
- **oneAPI 得到改进**：新版 **oneAPI** 包含多项改进。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1433189514993733783)** (3 messages): 

> `关于 Sum Reduction 的技术博客、LLM 的智能体强化学习、VS Code 的 Nsight Copilot` 


- **技术博客文章：Sum Reduction**：一位成员撰写了他们的第一篇技术博客，关于 sum reduction，其中 [第一部分](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) 涵盖了介绍，[第二部分](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) 涵盖了实现，同时研究了 **PTX/SASS** 并尝试使用 **Nsight**。
   - 作者计划接下来撰写关于 **matmul** 的文章，并欢迎反馈。
- **LLM 智能体强化学习演讲**：一位成员分享了最近关于论文 ["The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"](https://arxiv.org/abs/2509.02547) 的演讲，主题是针对 **LLM** 的**智能体强化学习 (Agentic RL)**。
   - 该演讲也分享在了 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7389584847813595137/) 上。
- **NVIDIA 发布适用于 VS Code 的 Nsight Copilot**：**NVIDIA** 发布了 [Nsight Copilot for VS Code](https://developer.nvidia.com/nsight-copilot)，这是一款针对加速计算的编程助手，提供智能代码建议、**CUDA 感知聊天**以及对 **CUDA** 开发工作流的辅助。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1433354525741289575)** (2 messages): 

> `Kernel Generation, Data Efforts in Kernel Generation` 


- **Kernel Generation 工作正在进行中**：一名成员分享了 **kernel generation 领域**正在进行的努力汇编，强调了其潜在的实用性和酷炫之处。
   - 该汇编以 [Google 文档](https://docs.google.com/document/d/e/2PACX-1vTjS-UMH1HB5n_PENq2k-3YRfXIXkqKIKeNC2zcWMyLPdl4Jrwvdk4dNDVSsM8ybKrCxZB7GJq1slZF/pub) 的形式提供，供有兴趣贡献或了解信息的成员参考。
- **Kernel Generation 领域的数据工作列表**：同一份文档还列出了与 **kernel generation 领域**相关的 **data efforts**，提供了一个综合资源。
   - 这一内容的加入强调了数据在推进 kernel generation 技术中的重要性，并鼓励研究人员和从业者之间的协作。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1433366847566381106)** (1 messages): 

> `Kernel recompilation, Incremental compilation` 


- **通过增量编译加速 Kernel 迭代**：一位用户寻求关于在修改源文件后避免完整的 kernel 重新编译的建议。
   - 他们正在寻找实现 **incremental compilation** 的方法，以加快流程并更快地完成迭代。
- **Kernel 源码修改**：用户正在修改 kernel 源码文件。
   - 每次修改都需要从头开始重新编译，这非常耗时。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433507049051914353)** (3 messages): 

> `Executorch CUDA backend status, Torchscript deprecation, Production GPU Deployments` 


- **Executorch CUDA 后端尚未达到生产就绪状态**：鉴于 **TorchScript** 弃用的传闻，一名成员询问了 **Executorch** 的 **CUDA backend** 及其在生产环境 **GPU deployments** 中的稳定性。
   - 另一名成员回答称其*尚未达到生产就绪状态*，并询问了具体的用例，例如服务器推理、桌面推理（及操作系统）或 Jetson 风格的嵌入式系统。
- **TorchScript 弃用及其影响**：最初的问题引发了对 **TorchScript** 作为 **PTC25** 一部分可能被弃用的担忧，从而引发了对替代解决方案的讨论。
   - 回复集中在 **Executorch** 作为潜在替代方案的现状，特别是其 **CUDA backend**，但强调了其缺乏生产就绪性。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433265120301285488)** (4 messages): 

> `AMD Competition, Yottalabs blog, Distributed Inference, SoL vs Kernel Performance` 


- **Yottalabs 发布关于分布式推理 Kernel 的文章**：一名成员分享了来自 [Yottalabs 的博客文章](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025)，内容涉及优化分布式推理 kernel。
   - 该成员称这篇博客*非常棒*。
- **AMD 竞赛运行器达到 400% 利用率**：一名成员分享了关于 **AMD competition** 的一个有趣指标，提到运行器的总运行时间达到了 **400%** 的利用率，这意味着 **4 个运行器** 全天都在满负荷运行（见[附图](https://cdn.discordapp.com/attachments/1359640791525490768/1433336064160039042/image.png?ex=6904fa7b&is=6903a8fb&hm=26a309e7557e65616d6d1293dcb03134e4a6e7744bd18c9c6f8506d21a9dc209)）。
- **AMD 竞赛中 SoL 性能超过手动调优 Kernel 10 倍**：一名成员表示惊讶，尽管预期会有手动调优的 kernel，但即使是 **AMD competition** 中的获胜方案也比 **SoL** (Solution of Limits) 慢 **10 倍**。
   - 该成员希望 **AMD** 能有一个接近 **SoL** 的良好解决方案，并表达了学习和研究它的愿望。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433241627148943575)** (13 messages🔥): 

> `针对 Row Major 张量的 tiled_copy，mask_mod 等效性检查，Colexigraphical 顺序与 Pytorch 的对比，scalar_to_ssa 定义，cute-dsl Constant Memory` 


- **Tiled Copy 读取 Row Major 张量**：一位成员正在寻求关于创建能正确读取源 Row Major 张量的 `tiled_copy` 的指导，并指出配置 `make_tiled_copy` 的 `(tid-layout, v-layout)` 输入参数以配合 Atom Size 的重要性，从而在处理 Column Major 和 Row Major 张量时都能获得最佳性能。
   - 该成员认为它在功能上应该同时支持 Column Major 或 Row Major 张量，但为了获得性能，需要进行适当的配置。
- **Mask Mod 代码等效性受到审查**：一位用户寻求帮助，以确定为什么 `mask_mod` 函数的两个实现（一个使用直接索引，另一个使用 `cute.make_fragment` 进行内存访问）不等效，尽管前者可以正常工作。
   - 该成员怀疑索引处理方式存在差异，特别指出*这似乎可能是 Colexigraphical 顺序与 Pytorch 之间差异的一个案例*。
- **Scalar to SSA 定义阐述**：在关于 `mask_mod` 函数的讨论中，一位成员要求澄清 `utils.scalar_to_ssa` 的定义，随后提供了其代码： 
```python
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    vec = cute.make_fragment(1, dtype)
    vec[0] = a
    return vec.load()
```
   - 同时指出 `make_fragment` 已被弃用，后续将会修复。
- **Colexigraphical 索引探讨**：一位成员考虑在 Colexigraphical 坐标空间中创建索引表达式，并询问如果直接进行 Colexigraphical 索引不可行，是否可以对 Cute Tensor 应用基于 Pointer Math 的读取。
   - 讨论随后指出，主要问题可能是传递了指针索引偏移量（Pointer Index Offset）而不是 1D Coordinate Index。
- **Cute-DSL Constant Memory 访问**：一位成员询问了如何在 Cute-DSL 中读取/写入 Constant Memory。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/)** (1 messages): 

j4orz: https://singularitysystems.bearblog.dev/
  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433534172558921738)** (1 messages): 

> `Helion PR 反馈` 


- **Helion PR 征求反馈**：一位成员请求对 [Helion 的 Pull Request](https://github.com/pytorch/helion/pull/1053) 提供反馈。
   - 未提及关于该 PR 的具体细节。
- **代码审查请求**：一位开发者正在为其开启的 Pull Request 寻求审查和反馈。
   - 这是在合并更改前确保代码质量并遵循项目标准的常见做法。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433198583506796626)** (18 条消息🔥): 

> `Extropic AI 的硬件加速器，低资源语言翻译，ArXiv 发布时间表，AI 论文筛选` 


- **Extropic 的硬件加速器：真实但小众？**：早期讨论表明 [Extropic AI 的硬件加速器](https://fxtwitter.com/extropic_ai/status/1983579587649904960?s=46) 可能是真实的，但考虑到它是一个概率硬件加速器，其应用场景可能较为小众。
   - 它似乎是 **ASIC** 而不是 **FPGA**，一位成员宣称他们喜欢作为“理论实验家”和“实验理论家”来*尝试一些不同的东西*。
- **Khowar 语言翻译：数据稀缺**：一位成员正在研究**低资源语言 (Khowar)** 的机器翻译，并面临数据稀缺问题，目前正在通过扫描实体书来构建数据集。
   - 文本提取器无法识别 Khowar 特有的 **Perso-Arabic script** 字符，导致字母替换或字形错位；正在向处理过自定义脚本的人寻求帮助。
- **ArXiv 的发布时间表：360/7 提案？**：一位成员提议 **arXiv 的 CS 类别** 应该切换到 360/7 发布时间表，而不是仅限工作日的计划。
   - 另一位成员发布了一张带有“更多溢出”字样的图片作为回应，表示赞同。
- **每日汇总 vs 重要性：寻找论文平衡点**：一位成员建议在一个帖子中发布按重要性排序的**每日论文汇总**，并在单独的帖子中发布最多 2-3 篇其他更重要的论文，这一提议得到了社区的积极响应。
   - 其他成员同意重要性优先于数量，并参考了 **Elvis Saravia** ([nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai)) 和 **ByCloud** ([https://x.com/TheAITimeline](https://x.com/TheAITimeline)) 的**每周 AI 论文回顾**作为灵感资源。
- **Agent 辅助 ArXiv：自动化 AI 论文发现**：一位用户表示有兴趣创建一个 **Agent/bot** 来寻找他们最感兴趣的论文，或者至少进行预筛选。
   - 其他成员指出 **AlphaXiv** ([https://www.alphaxiv.org/](https://www.alphaxiv.org/)) 和 **Emergent Mind** ([https://www.emergentmind.com/](https://www.emergentmind.com/)) 是汇集热门 AI 论文的资源，反映了集体兴趣，其中一位补充说，尽管存在算法偏见，他目前的主要信息源仍是 X feed。

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433171191039918090)** (78 条消息🔥🔥): 

> `Markovian vs Non-Markovian, Linux Foundation Robotics Project, Universities as Businesses, Robot Purchase Discussion, Continual Learning vs Continual Adaptation` 


- ****Markovian** 模型：定义之争！**: 关于如何定义“非马尔可夫 (non-Markovian)”的辩论，澄清了它假设“马尔可夫”为标准的 1-step/state Markovian，即 p(x|任意长度的 t-1 轨迹)=p(x|t-1)；一位成员提到与 [Linux Foundation Edge AI 小组](https://lfedge.org/projects/infiniedge-ai/) 的联系，以寻求潜在的仿生机器人/机器人技术建议。
   - 成员们对 CS 课程表示沮丧，因为它们不再教授传统上通过 C 语言传授的核心内容，而是从汇编语言直接跳到了修复 *Google bugs*。
- ****大学**：是企业还是破产？**: 讨论大学如何在州政府指令与商业现实之间取得平衡，特别是关于禁止 **ChatGPT** 但未能让学生为非冯·诺依曼架构 (non-von Neumann architectures) 做好准备的问题。
   - 一位成员幽默地将 MBA 思维描述为设计具有*最佳*计划性报废的产品以引导重复购买，并引用了一篇关于该主题的 [Nature 文章](https://www.nature.com/articles/s41928-025-01488-x)。
- ****机器人**：买还是不买？**: 一位成员讨论了与合作伙伴计划在年底前购买机器人的想法，并打算在此之前准备好项目提案。
   - 该成员还对使用 **diffusion world models** 和 **deepmimic** 的训练环境感兴趣，并可能将 deepmimic / pybullet 移植到一些更新的框架中，目标是直接将姿态 Token (pose tokens) 训练到自回归模型 (autoregressive model) 中，以输出音频、姿态和文本 Token。
- ****持续适应 (Continual Adaptation)** vs. 持续学习：一种新范式？**: 一位成员提到从“Continual Learning”转向“Continual Adaptation”以避免灾难性遗忘 (catastrophic forgetting)，旨在实现实用性和可访问性，而不是简单地增加一切（参数、计算、数据）。
   - 核心动力是提高可访问性，并应对资源和计算的现实情况及限制。
- ****Anthropic 的内省 (Introspection)**：空洞无物？**: 成员们审阅了 [Anthropic 的 Introspection 帖子](https://www.anthropic.com/research/introspection)，将其描述为*毫无意义 (nothing burger)*，其中对比概念向量 (contrastive concept vectors) 影响了模型权重，且模型有时能检测到操纵。
   - 该文章对模型的推测进行了外推，而这些推测并非源自实验。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1433190304730841108)** (3 条消息): 

> `Anthropic is Fraud, Haiku sizes` 


- **Anthropic 的骗局被揭穿**: 一位成员认为 *Anthropic 从第一天起就在行骗*，暗示他们的 **LLM** 具有误导性。
   - 他们预测 Anthropic 发布的所有内容都将延续这种欺骗模式。
- **文献中的尺寸**: 一位成员指出 *haiku, sonnet, opus 指的是模型大小*。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433243553332596770)** (98 条消息🔥🔥): 

> `DSPy 的 Scikit-learn 风格 API，Semantic Dataframes，不带参数的 ReAct 模块 finish() 函数，印度浦那的 DSPy 见面会，BAML Adapters 对比 JSON Schema` 


- **DSPy-Pandas 的易用性承诺？**：一位成员正致力于为 **DSPy** 构建一个 **scikit-learn 风格的 API** 接口，在包装 DSPy 流程的 *经典* pandas/polars dataframes 上提供 *fit/transform/predict* 方法。
   - 另一位成员建议了一个名为 **semantic dataframes** 的替代项目。
- **ReAct Finish 函数遇到阻碍**：一位成员在使用 **ReAct 模块** 时遇到了问题，**LLM** 在调用 **finish()** 函数时带了参数，导致报错。
   - 建议在 signature 中添加一行引导，指示 LLM 不要向 ReAct 的 **finish()** 函数传递任何参数，并在编码完成时调用不带参数的 finish()：finish()。
- **浦那筹备潜在的范式聚会**：位于印度浦那的 **Unravel.tech** 创始人（该公司使用 **DSPy** 为企业构建 AI agents）表示有兴趣在浦那举办 **DSPy 见面会**。
   - 有人指出该用户应在 <#1433127482676084746> 或 <#1211763460480827402> 频道进行询问。
- **BAML 击败笨重糟糕的 JSON？**：一位成员分享了他们在并购 (Merger and Acquisition) 案例中使用 **BAML Adapters** 的 GitHub 帖子，并在任何需要结构化输出的地方使用 **BAMLAdapter**，因为他们不喜欢 **JSON schema**。
   - 他们分享了一张截图，展示了在 **DSPy** 适配器构建的 prompt 中，**JSON schema** 部分是如何按照 **BAML** 格式重写的（[查看图片](https://cdn.discordapp.com/attachments/1433555562116943933/1433556837990531092/json-schema-vs-baml.png?ex=69051f58&is=6903cdd8&hm=ccc16f7efaeeb0d86031217401084b0475b9c09eb0423bc7f5a5451e8933dd86)）。
- **JSON 被公正地评为糟糕？**：据一位成员称，当你 *不* 使用 **JSON schema** 时，**LLMs** 表现得更好，因为从语义角度来看它非常糟糕、冗长，并且添加了在 token 空间中距离较远的描述符。
   - 他建议查看他在 [此仓库](https://github.com/prrao87/structured-outputs) 中的实验和数据，他指出 JSON schema 客观上更差。此外，DSPy 的基准 prompts 非常出色，你甚至永远不需要 SAP 来修复输出。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433168760755327096)** (77 条消息🔥🔥): 

> `Extropic BEFF 芯片，ScaleAI 远程劳动力指数，Cognition SWE-1.5，Codex 退化，OpenAI Codex 额度` 


- **Extropic 发布热力学 BEFF 芯片！**：Extropic 首次展示了他们的 **热力学智能硬件 (XTR-0)**，被称为 *BEFF 芯片*，并在 [X](https://x.com/Extropic_AI/status/1983579587649904960) 上进行了分享。
- **ScaleAI 发布严峻的“远程劳动力指数”**：ScaleAI 发布了一个 [新基准测试](https://scale.com/leaderboard/rli)，衡量当前的 Agents 在完成人类平均需要 **30 小时** 才能完成的任务时，达到 **Upwork 自由职业者** 水平的表现。
   - 表现最好的 agent (**Manus**) 实现了 **2.5% 的自动化率**，失败原因绝大多数是质量和完整性问题，这促使一位成员评论道：*如果通过正确的 UI 进行人机协作，这将非常有价值，但作为劳动力替代品则完全没用*。
- **Cognition 发布快速的 SWE-1.5 Swift**：Cognition 在 Windsurf 上发布了 **SWE-1.5**，这是一个快速的智能体编程模型 (agentic coding model)，运行速度高达 **950 tok/s**——比 Haiku 快 6 倍，比 Sonnet 快 13 倍——通过 Cerebras 硬件、投机采样 (speculative decoding) 和自定义优先级队列系统实现，并在 [X](https://xcancel.com/cognition/status/1983662836896448756) 上分享。
- **Codex 质量直线下降**：Jessie Frazelle 报告称，随着使用量激增，**Codex** 从 *大神级* 降到了 *有害级*，Alexander Embiricos 在 [此 X 推文](https://xcancel.com/embirico/status/1983643336390144163?s=46) 中回应称，这种退化正被视为关键优先级进行处理。
- **OpenAI 增加 Codex 额度以刺激使用**：根据 [这条推文](https://xcancel.com/OpenAIDevs/status/1983956900602581254)，OpenAI 为 ChatGPT Plus/Pro 上的额外 Codex 使用引入了 **按需付费额度**（每 1,000 额度 40 美元），并为所有用户重置了速率限制 (rate limits)。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1433254873033478196)** (8 messages🔥): 

> `MiniMax Speech 2.6, Voice Cloning, MiniMax Music 2.0, Generative Music Platform` 


- **MiniMax Speech 2.6 以极速克隆声音**：Hailuo AI 推出了 **MiniMax Speech 2.6**，其特点是 **低于 250 ms** 的实时延迟和完整的 **voice cloning** 能力，详情见 [此 X 帖子](https://x.com/Hailuo_AI/status/1983557055819768108)。
- **MiniMax Speech 用户讨论 API 和路线图**：用户正在积极讨论并赞扬 **MiniMax Speech 2.6** 具有未来感的能力，但也对 **OpenAI 风格 API** 的可能性、语言覆盖范围（马拉雅拉姆语）以及关于 **voice changer** 和同步 **video generation** 的路线图细节提出疑问。
- **MiniMax Music 2.0 惊艳 AI 届**：Hailuo AI 发布了 **MiniMax Music 2.0**，这是一个 **generative-music platform**，能够制作时长 **5 分钟**、具有逼真人声和多乐器控制的专业级歌曲，更多信息请点击 [此处](https://x.com/Hailuo_AI/status/1983964920493568296)。
- **MiniMax Music 2.0 开源正在筹备中？**：热心的用户正在询问 **MiniMax Music 2.0** **open-sourcing** 的可能性，以及是否会增加 **audio uploads** 和纯器乐模式等功能。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1433190317536051310)** (19 messages🔥): 

> `Frontier LLM Training, Low-Resource Language MT, OCR for Custom Scripts` 


- **独自训练前沿 LLM 证明非常艰巨**：一位成员讨论了由于算力需求，独自训练前沿推理 **LLMs** 所面临的挑战，并建议将蒸馏或使用 [Unsloth](https://github.com/unslothai/unsloth) 等工具进行微调作为潜在的变通方案。
   - 另一位成员分享了他自己的推理模型 [TRL implementation](https://github.com/torotoki/reasoning-minimal)，称其为个人项目的一个有趣但不太困难的方向。
- **科瓦语 OCR 面临数据稀缺**：一位成员正在研究 **Khowar**（科瓦语，一种低资源语言）的机器翻译，并且由于该语言独特的波斯-阿拉伯语脚本，在从扫描书籍中提取文本时面临挑战。
   - 他们指出，现有的 **PyMuPDF (fitz)** 和 **MyPDF** 等工具会误解或扭曲字形，因为该脚本包含阿拉伯语或乌尔都语中没有的字母。
- **Vision OCR 模型可以帮助低资源语言**：成员们建议通过手动标记字形来创建一个用于微调 vision OCR 模型的数据集，并指向了一篇关于在有限数据下构建模型的技巧论文：[[2509.14786] Tricks of the Trade for Developing Robust Arabic OCR Systems](https://arxiv.org/abs/2509.14786)。
   - 一位成员补充说，成功将需要 *大量的人力和算力*。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433180462968475699)** (49 messages🔥): 

> `Manus agent, Agent Evaluation Metrics, Extropic hardware, RWKV Understanding, Weight Decay Scaling` 


- **Manus Agent 表现稳健引发惊讶**：尽管表现被认为很稳健，但 **Manus** Agent 得到的讨论却出奇地少，引发了人们对其为何没有获得更广泛认可的好奇。
   - 一位成员建议，缺乏讨论是因为 *1-2% 的成功率目前还不足以让任何人真正使用 Agent*。
- **Agent 基准测试可能衡量的是分布内成功**：由于 Agent 的成功率在 **1-3%** 左右，目前的基准测试可能主要衡量的是分布内（in-distribution）的表现，而非通用的 Agent 能力。
   - 一位成员质疑，像 **Manus** 这样在可视化等特定领域表现出色的 Agent，是否真的优于那些成功率分布更均匀的 Agent。
- **Extropic 的定制硬件受到审视**：成员们讨论了 **Extropic 的定制硬件**，该硬件旨在通过替代原语（alternative primitives）而非在向量和矩阵中编码图（graphs）来实现更高效的模型执行。
   - **Groq** 和 **Cerebras** 等替代方案专注于通过更大的片上缓存（on-chip cache）来避免从内存中获取数据，从而提高推理效率。
- **RWKV 的复杂性阻碍了其采用**：理解 **RWKV** 的数学公式表达的难度以及论文表述不清阻碍了其采用，一位成员指出，*这始终是 RWKV 最大的问题*。 
   - 该成员表示，*坦白说，这是我们尽管想训练但最终没有训练 RWKV 世界模型的主要原因*，并提到论文中需要更多的细心和打磨。
- **Weight Decay 缩放引发争论**：关于 **Weight Decay**（权重衰减）是否应该随模型大小缩放存在争论，详见此 [discord 讨论](https://discordapp.com/channels/729741769192767510/730095596861521970/1433145349928779888)。
   - 一篇论文建议不进行缩放，而另一篇则建议进行 **sqrt(dim)** 缩放，导致缺乏共识，如 [Weight Decay and Preconditioning Can Provably Recover Sparsity](https://arxiv.org/abs/2510.15262) 所示。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

gsarti: <@709147478963781692> 供参考 (fyi)
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433237786437357628)** (23 messages🔥): 

> `Kimi K2, Kimi Delta Attention, Kimi-cli's D-Mail` 


- **Kimi K2 仓库请求 Hacktoberfest 标签**：一位成员请求在 HuggingFace 上的 **Kimi K2 仓库**中添加 **Hacktoberfest 标签**。
- **Kimi-Linear-48B-A3B-Base 已上线**：成员们注意到 **Kimi-Linear-48B-A3B-Base** 已在 HuggingFace 发布，链接见[此处](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base)。
- **Kimi Delta Attention 让成员联想到 Qwen**：一位成员指出，*`Kimi Delta Attention` 让我想起了 Qwen3 的下一个 Gated Deltanet*。
- **Kimi-cli 的 D-Mail 受到欢迎**：成员们分享了显示 **Kimi-cli 的 D-Mail** 获得粉丝的帖子，特别是[这个帖子](https://x.com/steipete/status/1983713085019046322?s=46)。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433314379637719173)** (11 messages🔥): 

> `Manus Credits, Developer for Project` 


- **用户请求 Manus 积分和协助**：几位成员正在寻求 **Manus 积分**，并表示愿意为项目协助付费，特别是在考试临近之际。
   - 一些用户正在询问 **$99 积分包**的可用性，并探索像 **Monica** 这样的替代方案来完成学校作业。
- **开发者寻求项目机会**：一位成员宣布可以作为开发者参与潜在项目。
   - 另一位成员通过私信询问该开发者是否有 **Manus 积分**来协助完成项目。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433253169097478235)** (5 messages): 

> `Neos 摔跤/拳击比赛，11 月的 MCP CTF` 


- **AI Neos 进入赛场**：一名成员询问了涉及 **Neos** 的**摔跤/拳击比赛**的时间表。
   - 虽然没有给出明确答复，但另一名成员表示希望这能*很快*发生。
- **Hack The Box 主办 AI 安全 CTF**：来自 **Hack The Box (HTB)** 的团队将于 11 月 20 日举办一场仅限 **MCP** 的 **CTF**，重点关注 **AI 安全**，寻求参与者在真实场景中测试其**渗透测试 Agent**。
   - 该活动可免费参加，更多详情和报名请点击[此处](https://ctf.hackthebox.com/event/details/neurogrid-ctf-the-ultimate-ai-security-showdown-2712?utm_campaign=AI+CTF+-Oktopost&utm_content=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fupdate%2Furn%3Ali%3Ashare%3A7386416070783479808&utm_medium=social&utm_source=LinkedIn&utm_term=)。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1433305863019298906)** (2 messages): 

> `本地模型训练，Windows 上的依赖问题，Linux vs WSL` 


- **Windows 用户面临本地模型训练困扰**：一名成员询问其他成员使用什么 IDE 在本地训练模型，并提到了 Windows 上的依赖问题。
   - 另一名成员建议使用 **Linux** 或 **WSL** 以避免“依赖地狱”。
- **推荐使用 Linux/WSL 进行本地模型训练**：一位用户表达了在 Windows 上本地训练 LLM 时对依赖管理的沮丧。
   - 另一位用户建议将使用 **Linux** 或 **WSL** 作为解决这些问题的首要步骤。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433215465542516817)** (1 messages): 

> `` 


- **未讨论特定主题**：在提供的消息中未发现适合总结的具体研究主题或讨论。
- **表达了普遍的鼓励**：一位用户对某人的故事表示了愉快和支持。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433478645338472519)** (1 messages): 

> `Kimi Linear Attention, MoonshotAI` 


- **MoonshotAI 发布 Kimi Linear Attention 技术报告**：[MoonshotAI 发布了一份技术报告](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf)，详细介绍了他们的 **Kimi Linear Attention** 机制。
   - 该报告可能包含有关其线性注意力方法的架构、性能和实现细节；线性注意力将二次方计算复杂度降低为线性，从而允许上下文窗口扩展到极长。
- **Kimi Linear Attention：革新上下文窗口扩展**：该技术报告强调了 **Kimi Linear Attention** 在 Transformer 模型中革新上下文窗口扩展的潜力。
   - *这一创新使模型能够更高效地处理显著更长的序列*，为文档摘要和长文本生成等各种应用开启了新的可能性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433215465542516817)** (1 messages): 

> `` 


- **表达了暖心的支持**：一位用户表达了深切的赞赏和支持，分享道“*你的故事真的很触动我*”，并通过说“*我也曾有过同样的经历*”来传达快乐和个人共鸣。
- **送上美好祝愿**：该用户以积极的鼓励结束，简单地表示：“*祝一切顺利！*”。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

georgehotz: 总有一天我们会对 tinygrad 进行 `ruff format`
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1433384120934858782)** (1 messages): 

> `GROUP_REDUCE 错误，rangeify 重写调试` 


- **嵌套 GROUP_REDUCE 引发困扰**：一名成员报告了一个涉及嵌套 **GROUP_REDUCE** 操作的错误，并请求协助调试与新的 **rangeify** 重写相关的部分。
   - 他们正在寻求一个快速提示，以避免在原因对熟悉这些更改的人来说显而易见的情况下，进行*漫长而机械的调试*过程。
- **寻求关于 Rangeify 重写的提示**：一名成员请求关于另一个 reduce 内部 **GROUP_REDUCE** 错误潜在原因的提示，希望能利用熟悉新 **rangeify 相关重写** 的专家的见解。
   - 该成员旨在避免在存在简单解决方案或已知问题的情况下进行广泛调试，否则表示愿意进行更深入的调查。


  

---

### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1433241754676494488)** (1 messages): 

> `SWE-1.5, Fast Agent Models, Coding Performance` 


- **SWE-1.5 强势登陆 Windsurf**：一个新的 Fast Agent 模型 **SWE-1.5** 已发布，现已在 Windsurf 中可用，承诺以史无前例的速度提供接近 SOTA 的编程性能。
   - 更多详情请参阅[官方公告](https://x.com/cognition/status/1983662836896448756)。
- **SWE-1.5 树立了新的速度标准**：Fast Agent 模型 **SWE-1.5** 树立了新的速度标准，同时提供接近 SOTA 的编程性能。
   - 该模型现已在 **Windsurf** 中可用。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433446982726451331)** (1 messages): 

> `Model Context Protocol RFC Status` 


- **Model Context Protocol RFC 面临延迟**：由于利益相关者正在等待实现的完成，[Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269) 正面临延迟。
   - 有人担心该 RFC 缺乏具体的实现，阻碍了对其实际影响进行评估的能力。
- **没有实现，就没有评估**：利益相关者表示担忧，称在评估其实际影响之前，该 RFC 需要一个具体的实现。
   - 如果没有实现，很难评估该 RFC 的有效性。