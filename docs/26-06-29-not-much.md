---
companies:
- meta-ai-fair
- cursor
- deepseek
- cognition
- arena
date: '2026-06-29T05:44:39.731046Z'
description: '**Meta** 宣布推出 **Brain2Qwerty v2**，这是一款实时、非侵入式的脑信号转文字解码器，单词识别准确率最高可达
  **78%**，并已公开训练代码和数据集。**Cursor** 推出了 **Cursor for iOS**，支持远程 AI 代理和实时活动功能。开源权重模型的访问服务正在商业化，用户每月支付
  **9.99 美元**，即可使用 GLM 5.2、Qwen 等模型；与此同时，**Cognition** 推出了注重成本效率的编程工具 **Devin Fusion**。**Arena**
  在上线八个月后达到 **1 亿美元年经常性收入（ARR）运行率**，业务重点是智能体评测。基础设施方面的挑战，尤其是在中国，仍然至关重要。DeepSeek 的
  **DSpark** 推进了推测解码技术，相较此前的方法取得了显著提升，并已部署到 **DeepSeek-V4-Flash** 和 **V4-Pro** 中。

  '
id: MjAyNS0x
models:
- brain2qwerty-v2
- glm-5.2
- qwen
- deepspark
- deepspeak-v4-flash
- deepspeak-v4-pro
people:
- jeanremiking
- kimmonismus
- ml_angelopoulos
title: '今天没发生什么特别的事。

  '
topics:
- brain-computer-interfaces
- non-invasive-bci
- real-time-decoding
- speculative-decoding
- agent-assisted-research
- inference-systems
- cost-efficiency
- remote-agents
- training-data
- model-access
- infrastructure-strategy
---

**平静的一天。**

> 2026 年 6 月 27 日至 6 月 29 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。你可以通过 [AINews 网站](https://news.smol.ai/) 搜索过去的所有期刊。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览


- **Meta 在非侵入式脑信号转文字方面取得里程碑进展**，吸引了最多技术关注。[​@AIatMeta](https://x.com/AIatMeta/status/2071566924803395741) 宣布推出 **Brain2Qwerty v2**：一种能够根据原始脑信号实时解码句子的系统；[​@JeanRemiKing](https://x.com/JeanRemiKing/status/2071568417522369008) 总结了这次发布及相关链接；[​@AIatMeta](https://x.com/AIatMeta/status/2071566934571954326) 补充说，Meta 将发布 v1/v2 的**训练代码**，而 BCBL 将发布 **v1 数据集**。
- **Cursor 同时推出了 iOS 版和远程 Agent**，这是当天规模最大的产品发布之一：[​@cursor_ai](https://x.com/cursor_ai/status/2071641103191998810) 介绍了 **Cursor for iOS**，支持始终在线的云端 Agent，以及远程控制电脑上的 Agent；后续推文还强调了[手机上的 Live Activities 和差异审查](https://x.com/cursor_ai/status/2071641104869691671)功能。
- **开放权重模型的访问正在被产品化，而不再只是停留在讨论层面**：[​@cline](https://x.com/cline/status/2071617325296734309) 推出了**每月 9.99 美元**的通行证，可折扣访问 GLM 5.2、DeepSeek、Kimi、MiniMax、Qwen 等模型；[​@cognition](https://x.com/cognition/status/2071624568465490170) 推出了 **Devin Fusion**，声称通过混合模型框架，以低 **35% 的成本**实现“Fable 级别”的编程能力。
- **Arena 已达到具有重要商业意义的规模**：[​@arena](https://x.com/arena/status/2071630464583151727) 和 [​@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2071629882057228680) 表示，Arena 在推出评测产品八个月后，已达到 **1 亿美元的 ARR 运行率**；目前其平台正重点加强部署后评测和 Agent 评测。
- **基础设施压力仍是首要议题**：[​@kimmonismus](https://x.com/kimmonismus/status/2071524362012791114) 认为，中国在能源、数据中心和国产硬件方面的战略正逐渐成为严峻的战略威胁；[​@garrytan](https://x.com/garrytan/status/2071600933210100074) 将应对措施概括为：“**建设电力和数据中心**。”

**脑机接口与 AI for Science 工具**

- **Brain2Qwerty v2** 是当天最值得关注的研究发布。Meta 表示，该系统能够从**非侵入式**记录中实时解码出**单词和语义**，而不只是字符，正在缩小与侵入式 BCI 之间的差距。社区总结指出，相比此前的非侵入式成果，该系统的表现据报道有明显提升：总体**单词准确率约为 61%**，最佳参与者达到 **78%**；模型使用了**9 名志愿者**在受控打字环境中的数据进行训练。其关键的工程意义并不在于已经具备消费级应用条件，而在于这套系统将原始神经信号建模与语言建模结合起来，达到了足以在实验室中实现句子级解码的效果。详见 [Meta 的公告](https://x.com/AIatMeta/status/2071566924803395741)、[代码和数据发布详情](https://x.com/AIatMeta/status/2071566934571954326)、[​@JeanRemiKing 的推文串](https://x.com/JeanRemiKing/status/2071568417522369008)，以及 [​@kimmonismus](https://x.com/kimmonismus/status/2071712776226283902) 提供的谨慎外部总结。
- 这次发布也成为 **Agent 辅助研究**的一个案例。[​@stalkermustang](https://x.com/stalkermustang/status/2071590526965502027) 提到，Meta 的说明显示，一个由 coding agent 驱动的 **Auto Research** 工作流发现并实现了多项改进，使词错误率低于标准 HPO。无论你是否认同“vibe-science”这一说法，更稳妥的结论是：coding agent 正越来越适合用于 ML 系统的**闭环实验迭代**，而不只是搭建代码仓库的基础框架。

**推理系统：DSpark、vLLM 与解码机制**



- **DeepSeek 的 DSpark** 是这次最具实质性的推理话题。[@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2071445817102315595) 发布的一篇详细解读，将 DSpark 定位为 **speculative decoding** 的重要一步，重点强调了两个方向：更好的草稿生成，以及更智能的验证调度。据报道，在 Qwen3-4B 上，DSpark 的接受长度相比 **Eagle3** 提高了 **30.9%**，相比 **DFlash** 提高了 **16.3%**；此外，它还已经部署到 **DeepSeek-V4-Flash** 和 **V4-Pro** 的预览版引擎中。[@teortaxesTex](https://x.com/teortaxesTex/status/2071631028511203457) 和 [@vllm_project](https://x.com/vllm_project/status/2071682507775635579) 的后续讨论进一步强调了其实际意义：DSpark 似乎正在成为新的 **SoTA 单 GPU speculative decoding 方案**，而 vLLM 社区也已经开始集成它。
- 更广泛地看，几条推文进一步梳理了当前推理瓶颈的认知框架。[@_avichawla](https://x.com/_avichawla/status/2071522418594861215) 对 **prefill 与 decode**、TTFT 与 token 间延迟进行了很好的解释，并说明了为什么 decode 经常受限于内存：因为它需要不断读取 KV-cache。这也解释了为什么在许多生产工作负载中，**speculative decoding**、KV-cache 优化、分组查询注意力（grouped-query attention）以及注意力机制重构，往往比单纯增加 FLOPs 更重要。
- NVIDIA/vLLM 也在推动更实用的自托管方案：[@vllm_project](https://x.com/vllm_project/status/2071483552106233993) 重点介绍了一份指南，讲解如何使用 **四台 DGX Spark**，在单一兼容 OpenAI 的 endpoint 后为 **Nemotron-3-Ultra 550B** 提供服务。其意义并不主要在于展示硬件规模，而在于说明：使用标准 serving stack，实现**私有化、多节点、接近 frontier 水平的推理**，正在逐渐成为常态。

**Agent harness、路由与多模型编排**

- Agent 系统的重心正在持续从“选择最好的模型”转向 **harness 工程**。[@cognition](https://x.com/cognition/status/2071624568465490170) 发布了 **Devin Fusion**，这是一个混合模型的编程 harness，声称在保持“Fable 级”质量的同时，可降低 **35% 的成本**。[@walden_yan](https://x.com/walden_yan/status/2071627241818399181) 介绍了与 **sidekick** 和**会话中途路由（mid-session routing）**相关的工作，[@jerryjliu0](https://x.com/jerryjliu0/status/2071737452323303750) 则指出了 sidekick 式委派在缓存效率方面的优势。正在形成的模式是：让昂贵的 planner 保持在循环中，把边界明确的子任务交给更便宜的模型，同时保留缓存局部性和上下文连续性。
- **动态子 Agent** 也成为另一个常见主题。[@LangChain](https://x.com/LangChain/status/2071631563897377010)、[@sydneyrunkle](https://x.com/sydneyrunkle/status/2071632107026174364) 和 [@hwchase17](https://x.com/hwchase17/status/2071633874736804066) 都重点介绍了这样一类工作流：主 Agent 编写编排代码，而不只是调用工具。这一点值得注意，因为它将抽象层从“会使用工具的聊天机器人”推向了更接近**可编程控制平面**的形态，以支持大规模任务分发。
- 开放式路由和检索技术栈也变得更加具体。[@LlamaIndex](https://x.com/llama_index/status/2071656315210826006) 和 [@jerryjliu0](https://x.com/jerryjliu0/status/2071729856900215261) 推出了 **Retrieval Harness**，在一个 Agent 循环中整合语义搜索、grep、文件列表和文件读取——这基本上反驳了“只要 grep 就够了”这种过于简单的观点；[@max_paperclips](https://x.com/max_paperclips/status/2071465351959998723) 也曾批评过这种说法。在评测方面，[@hwchase17](https://x.com/hwchase17/status/2071630837976822237) 宣布推出 **Trace Judge** 模型，用于检测轨迹错误，成本约为闭源模型的 **1/100**。

**开放模型、中国实验室与访问服务的商业化**



- **GLM 5.2** 依然是讨论的核心开源模型。原因并不是它今天正式发布了，而是许多开发者已经开始把它视为默认的严肃选择。[[@cline](https://x.com/cline/status/2071617325296734309)] 将其产品化，通过月度通行证捆绑提供 **GLM 5.2、DeepSeek、Kimi、MiniMax、Mimo 和 Qwen**，减少了 API 密钥管理和供应商频繁变更带来的麻烦。[[@tonbistudio](https://x.com/tonbistudio/status/2071595794147250540)] 使用 GLM 5.2 搭配 Kimi 和 MiniMax，测试了 **Mixture-of-Agents** 配置。[[@Astrodevil_](https://x.com/Astrodevil_/status/2071572680470655253)] 则将 GLM 5.2 用作 DevRel 内容调研 Agent 的驱动模型。
- 第二条值得关注的主线，是**中国开源权重模型竞争**正在持续加速。[[@eliebakouch](https://x.com/eliebakouch/status/2071713216028389396)] 提到，美团即将推出 **LongCat 2.0 / Owl Alpha** 模型：总参数量 **1.6T / 激活参数约 48B**、**1M 上下文窗口**、使用 **35T 训练 token**、采用 **n-gram embeddings** 和稀疏注意力，并在 **5 万张中国国产加速卡**上完成训练。[[@sun_hanchi](https://x.com/sun_hanchi/status/2071664412612833516)] 认为，这可能是首个在如此大规模的中国国产硬件上训练、接近前沿水平的模型。即便硬件细节仍存在不确定性，这件事的战略意义依然不容忽视。
- 在政策和商业层面，开源支持者认为，对前沿 API 进行限制可能适得其反，反而会把开发者推向他们能够自行掌控的模型权重。可参考 [@theinformation](https://x.com/theinformation/status/2071700452605829433)、[@ClementDelangue](https://x.com/ClementDelangue/status/2071686220548133048) 和 [@MTSlive](https://x.com/MTSlive/status/2071634697185353956) 的观点：**开源权重在结构上比 API 更难被压制**。

**RL、训练基础设施与基准测试/评测平台**

- **Snowflake Arctic RL** 是这一批发布中较有分量的基础设施项目之一。[[@StasBekman](https://x.com/StasBekman/status/2071628398234087642)] 宣布了一个开源项目，可与 **VeRL** 和 **SkyRL** 集成，并通过 **ZoRRo** 实现最高 **6 倍的 actor 更新加速**和 **3.5 倍的端到端加速**，将一次 Text2SQL 训练任务从大约 **5 天缩短到 32 张 H200 上约 36 小时**。Snowflake 还声称，其 **Arctic-Text2SQL-R2** 在企业 SQL 基准测试中击败了经过测试的 **Gemini 3.1 Pro** 和 **Claude 4.7** 配置，同时开放了 text-to-SQL 和多跳 QA 的训练方案。
- **Arena** 继续从基准测试项目转型为评测公司。[[@arena](https://x.com/arena/status/2071630464583151727)] 和 [@ml_angelopoulos](https://x.com/ml_angelopoulos/status/2071629882057228680) 表示，平台目前已有 **7 亿多次对话**、**8200 多万张选票**，每月访客超过 **1000 万**，并开始更加重视**Agent 模式评测**，例如任务完成率和幻觉率。这意味着，Arena 的定位正逐渐从单纯的偏好排行榜，转向模型**部署后的 CI/CD 评测层**。
- 其他一些发布也体现了专业化基础设施的发展趋势：[@wandb](https://x.com/wandb/status/2071603727585448025) 在 W&B 内推出了自动进行研究的 Agent **ARIA**；[@agenticin](https://x.com/agenticin/status/2071494912277938398) 推广 **Micro-Agent** 路由；[@fitsumreda](https://x.com/fitsumreda/status/2071616094260142431) 则介绍了 **Nemotron-TwoTower**。该模型将一个 AR LLM 复制为类似 diffusion 的并行生成器，并声称在 30B 模型上以 **2.42 倍吞吐量**达到 **98.7% 的 AR 质量**。

**平台与开发者产品更新**

- **Cursor 的移动端/远程办公布局**值得关注，因为它让“用手机操作云端 Agent”从概念变成了真正可用的功能。现在，用户可以启动持续运行的云端 Agent，也可以通过 iOS 远程控制必须运行在电脑上的 Agent，并在应用内查看 PR diff 和接收通知（[发布公告](https://x.com/cursor_ai/status/2071641103191998810)、[详情](https://x.com/cursor_ai/status/2071641104869691671)）。
- **Azure Foundry 上的 Claude** 现已正式 GA。[[@Azure](https://x.com/Azure/status/2071651695323492418)]、[@claudeai](https://x.com/claudeai/status/2071653958905467027) 和 [@ClaudeDevs](https://x.com/ClaudeDevs/status/2071697437136486585) 表示，客户现在可以在 Microsoft Foundry 中运行 **Claude Opus 4.8** 和 **Haiku 4.5**，并使用 Azure 身份认证、计费、治理控制、prompt caching 以及 thinking 支持。
- [@ndstudio](https://x.com/ndstudio/status/2071638578145145251) 推出的 **Rampart** 是一款务实的隐私工具：它采用一个**14.7MB 的浏览器端模型**，在数据离开客户端之前对 PII 进行脱敏。对于希望在受监管环境中使用 AI 的团队来说，这类小型本地预处理模型，可能比又一次通用聊天界面的小改动更有价值。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾



### 1. GLM-5.2 Extreme Local Inference Tests

  - **[GLM-5.2 753B (IQ1_S) fully local across 2×M5 Max over one TB5 cable — ~16 tok/s, llama.cpp RPC [video]](https://www.reddit.com/r/LocalLLM/comments/1uiuhec/glm52_753b_iq1_s_fully_local_across_2m5_max_over/)**（热度：377）：**一位用户报告称，使用 Unsloth 的动态 `IQ1_S` 量化，在本地完整运行 **GLM-5.2 `753B`**：标称约 `1.6` bit，但由于部分层采用更高精度，实际平均约为 `2.1` bit，模型文件大小为 `202GB`。该方案通过一条 **Thunderbolt 5** 连接，利用 **`llama.cpp` RPC** 将权重分布到两台各配备 `128GB` 统一内存的 **M5 Max** 系统上，所有权重都常驻内存，不需要从 SSD 分页读取，生成速度约为 `16 tok/s`，上下文长度为 `16k`，并使用 `q8` KV cache；TTFT 会受提示词长度影响，因为其中包含预填充过程。**评论者认为，一款 `753B` 模型跨两台 Mac 还能达到 `16 tok/s`，速度高得出乎意料；有人询问视频中的速度是否看起来比报告数值更快。另一位评论者表示，这套方案确实令人印象深刻，但也质疑这种极低比特的 `753B` 量化模型，在复杂推理任务上的表现是否能胜过更小、但采用更高精度的模型，例如 4-bit 的 `70B` 模型。

    - 一位评论者质疑 **GLM-5.2 753B IQ1_S** 跨两台 **M5 Max**、通过 **Thunderbolt 5** 运行时所报告的 **约 `16 tok/s`** 是否准确，因为视频看起来更快；另一位评论者则指出，虽然对于本地运行 `753B` 模型而言，这个吞吐量非常惊人，但极低比特的 **IQ1_S** 量化也带来了一个技术问题：它的推理质量与更小的 **4-bit `70B`** 模型相比究竟如何。
    - 一位用户提供了使用 **M3 Ultra Studio 256GB + M3 Max MBP 128GB**、运行 **GLM-5.2-UD-IQ4_XS** 的 llama.cpp RPC 风格对比基准：上下文为 `2,377` 个 token 时，速度为 `13.03 tok/s`、TTFT 为 `3.09s`；上下文为 `22,485` 个 token 时，速度为 `8.64 tok/s`、TTFT 为 `2.33s`；上下文为 `32,595` 个 token 时，速度为 `6.21 tok/s`、TTFT 为 `5.53s`。他们说明，**TTFT 包含缓存预填充时间**，因此这些数据更适合用于长上下文生成的比较。
    - 另一位评论者询问，llama.cpp 是否已经支持多台 Mac 互联，还是需要定制驱动；这涉及一个实现层面的问题：该方案使用的是 **llama.cpp RPC** 内置能力，还是定制的 Thunderbolt 网络与推理编排方案。

  - **[GLM 5.2 Q1_S vs Qwen 27B Q8](https://www.reddit.com/r/LocalLLaMA/comments/1uimjdi/glm_52_q1_s_vs_qwen_27b_q8/)**（热度：359）：**一项由个人进行的 `n=1` 对比测试显示，在两张 RTX 3090 上，**GLM-5.2 Q1_S** 仅用一次提示，就在约 `75k` 个 token、速度约 `6→3 t/s` 的情况下生成了一个完成度很高的 Three.js 竞技场游戏；相比之下，**Qwen 3.6 27B Q8** 需要 `1 + 3` 次提示和约 `42k` 个 token，虽然速度约为 `60 t/s`。作者之后澄清，GLM 使用了 `K/V Q8`，而 Qwen 使用的是完整的 `FP16` KV cache。由 **Opus 4.8** 和 **GPT-5.5** 担任评审的 LLM-as-judge 评分都将 GLM Q1_S 的代码质量和完成度排在首位；通过 OpenRouter 使用 GLM FP 版本时只用了约 `11k` 个 token，但存在控制功能缺陷。技术讨论中，热度较高的评论提到 Hugging Face 上可能有更强的 **GLM-5.2 REAP 504B GGUF `Q2_K_XL`** 量化版本，文件大小为 `211 GB`：[Hugging Face](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF)；也有人询问 OpenRouter 的成本，并报告称 **Qwen3.6-27B-UD-Q5_K_XL.gguf MTP** 只需 `2` 次提示和约 `11k` 个 token，就完成了类似的可玩 demo，速度达到 `110–130 t/s`，输出发布在 [CodePen](https://codepen.io/source-drifter/pen/MYJvNEb)。**这场讨论的核心是：低于 Q3 的极低量化是否天生就“没脑子”。原帖认为，只要允许模型进行更长时间的思考，更大的 Q1_S 模型仍然可以胜过采用高量化的小模型。不过，评论中的证据让这一结论变得复杂：一项 Qwen Q5_K_XL 测试速度快得多，而且只需要修复一个控制台错误。

    - 一位评论者指出，Hugging Face 上有更大规模的 **GLM-5.2-REAP-504B GGUF** 量化版本：[0xSero/GLM-5.2-REAP-504B-GGUF](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF)，具体是文件大小为 `211 GB` 的 **`Q2_K_XL`**，并认为它可能比测试中的 **`Q1_S`** 量化版本更强。这说明测试结果可能很大程度上受量化质量影响，而不完全代表基础模型本身的能力。
    - 一位用户报告称，本地运行带 MTP 的 **`Qwen3.6-27B-UD-Q5_K_XL.gguf`** 时，第一次提示生成内容后，再修复一个控制台错误，就完成了一个可玩的 CodePen demo：[demo](https://codepen.io/source-drifter/pen/MYJvNEb)。初次生成耗时 `50s`、生成 `5,538` 个 token，速度为 `110.69 tok/s`；修复阶段耗时 `41s`、生成 `5,422` 个 token，速度为 `129.88 tok/s`。报告中唯一的 bug 是 `Uncaught ReferenceError: time is not defined`。
    - 关于所提到的 **`211 GB` GLM 量化模型** 能否在配备 **128 GB RAM 的 Strix Halo** 系统上运行，也有人提出了硬件容量方面的疑问。也就是说，即使是低比特量化的前沿级 GGUF 模型，在计入模型本体、KV cache 和运行时开销后，也可能超出统一内存消费级或工作站配置的承载能力。




### 2. llama.cpp 模型与 Kernel 支持合并

  - **[DFlash 支持已合并到 llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1uhx862/dflash_support_merged_into_llamacpp/)**（热度：469）：****DFlash 支持已合并到 `llama.cpp`**，为该项目正式加入了扩散式文本生成支持。不过，评论者指出，**目前还不支持多模态 DFlash**。这次合并被视为后续加速工作的基础，例如 **DDTree/JetSpec**，以及未来可能为 **DSpark**、**Gemma Diffusion**、**Nvidia NemoDiffusion**、**Orthrus** 和潜在的 **LLaDA-like** 模型提供独立架构支持。**评论者普遍持积极态度，称赞 **Ruixiang63** 持续推进这一功能，并开玩笑或期待下一步加入 **DSpark** 支持。**

    - 评论者指出，`llama.cpp` 中目前的 **DFlash 支持不包含多模态/视觉场景**，因此依赖视觉模型的用户暂时无法受益。一位用户还提到，在 **RTX 5090** 上尝试运行 **Qwen3.6-27B** 时需要权衡：现有的 draft-model 工作流可能要求**关闭思考功能**，并且可能无法使用**视觉能力**和**并行推理**。
    - 一场关于技术路线的讨论将 DFlash 视为更大规模的推测式/扩散式加速栈的一部分：目前提到的其他加速方案还包括 **DDTree** 和 **JetSpec**；而要支持 **DSpark**、**Gemma Diffusion**、**NVIDIA NemoDiffusion**、**Orthrus**，以及可能仍具可行性的 **LLaDA-style** 模型，仍需要单独的架构支持。
    - 用户将 DFlash 与现有的 **MTP** 实验进行比较。一位评论者表示，他们已经在 **Qwen3.6** 和 **Gemma4** 上成功运行 MTP，并询问合并后的 DFlash 路径是否能在此基础上带来额外的性能提升。

  - **[DeepSeek V4，PR 已合并到 llama.cpp！](https://www.reddit.com/r/LocalLLaMA/comments/1uj0fkw/deepseek_v4_pr_merged_into_llamacpp/)**（热度：280）：** **DeepSeek V4** 支持 PR 已合并到 **llama.cpp**（[ggml-org/llama.cpp#24162](https://github.com/ggml-org/llama.cpp/pull/24162)）。用户现在可以通过 `git pull` 更新代码，使用 `cmake` 重新构建，并运行兼容的 **GGUF** 模型文件，不再需要依赖 fork 版本。后续最主要的技术问题是兼容性：评论者询问哪些 **GGUF** 已知可以在上游 **llama.cpp** 中运行，哪些仍只能通过第三方 fork 使用。评论整体以实用讨论和玩笑为主：一位用户指出，硬件要求可能意味着本地运行 DeepSeek V4 在未来几年内都难以实现；另一位用户则开玩笑说想要一个小型的 “microflashmini” 版本。

    - 在 DeepSeek V4 合并到 `llama.cpp` 后，评论者主要关注 **GGUF 兼容性**，具体询问哪些模型文件可以在上游最新版 `llama.cpp` 中运行，而不需要 fork。用户还期待 **Unsloth** 制作“真正合规的 GGUF 文件”，这意味着目前的转换和量化资源可能仍然比较零散，或者并非官方提供。
    - 一个与技术相关的重要担忧是，早期性能报告可能会非常混乱：用户预计会出现大量缺乏可复现细节的 `tokens/s` 数据，例如没有说明 GPU/CPU 型号、量化级别、上下文长度、后端、批大小或内存配置。


## 技术含量较低的 AI 子版块摘要

e /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo



### 1. Agentic Coding Tooling and Safety

  - **[Graphify 在 2.5 个月内收获 7.3 万个 stars 和 220 万次下载，并且刚刚进入 YC](https://www.reddit.com/r/ClaudeAI/comments/1ui6unv/graphify_hit_73k_stars_and_22m_downloads_in_25/)**（热度：962）：****Graphify** 声称，自 4 月 5 日发布以来，其开源项目增长迅猛：在大约 `2.5` 个月内获得了 `73k` 个 GitHub stars 和 `2.2M` 次下载，并入选 **YC S26**。该工具可以将代码仓库、文档、PDF、SQL schemas、Obsidian vaults 和 transcripts 转换为知识图谱，再通过 **Claude** 进行查询；作者称，与直接读取原始文件相比，每次查询的 token 使用量降低了约 `71×`。新推出的 `graphify reflect` 功能会将有用的答案和走不通的答案记录到 `LESSONS.md` 中，作为持久化的会话记忆。产品目前的方向是打造企业级“自学习公司大脑”，社区讨论可前往 [Discord](https://discord.gg/598Ad9zQZ)。**评论区的热门观点对其护城河和变现能力持怀疑态度：用户认为这套代码是免费的，而且 Agent 相对容易复现，因此可能被 Anthropic 或其他 LLM 厂商吸收。还有一位评论者质疑其在 LinkedIn 上的增长表现，称可见的帖子大多看起来像垃圾内容。

    - 一些评论者质疑 **Graphify 的护城河和变现能力**：由于代码免费开源，而且被认为*“对 Agent 来说并不难复现”*，他们认为其主要商业风险在于产品被同质化，或被 **Anthropic** 等模型提供商直接整合。
    - 一条具有技术相关性的批评，将 Graphify 的价值与现有开发者工具，尤其是基于 **LSP** 的代码智能进行了比较。一位用户表示，在一个*“规模相当大的代码库”*上，Graphify 的配置过程*“很繁琐”*，与传统工具相比，并没有明显提升输出质量或节省时间。
    - 有人提出了一个具体的打包问题：安装命令是 `pip install graphifyy`，其中有两个 `y`。一位评论者认为这看起来很可疑，可能会让 Python 用户在安装该软件包时产生信任问题或额外阻碍。

  - **[Claude Code 突然试图在我的电脑上打开 Remote Desktop 连接，真的把我吓坏了。](https://www.reddit.com/r/ClaudeAI/comments/1ui8g1t/claude_code_suddenly_tried_to_open_a_remote/)**（热度：937）：**图片（[Windows RDP 警告对话框](https://i.redd.it/zkcjmfu263ah1.png)）显示的是 **Windows 11 提示用户打开一个 `.rdp` Remote Desktop Connection 文件**，并不一定意味着有人正在通过入站连接接管电脑。结合标题和正文来看，用户称自己在使用 **Claude Code** 时出现了这一提示，随后还观察到 File Explorer 似乎被自动操作；评论区提出的最合理技术担忧是，Claude 或某个工具/MCP 工作流可能打开或生成了一个 RDP 文件，这可能是 prompt injection 或权限设置不安全导致的，而不是 **Anthropic** 直接“接管”了电脑。**评论者对“Anthropic 员工正在接管该会话”的说法持怀疑态度；其中一人指出，RDP 文件意味着本地电脑正在尝试向外建立连接，并且根据设置，可能会暴露剪贴板和驱动器。主要安全建议包括：避免授予过于宽泛的权限或使用 `dangerously-skip-permissions`，使用 Claude Code 的 [auto mode](https://code.claude.com/docs/en/auto-mode-config)，禁用类似 computer-use 的功能，或者将 Agent 运行在隔离的 sandbox VM/WSL 环境中。

    - 一种技术解释认为，屏幕上的警告很可能是用户打开了一个 `.rdp` 文件所触发的，这意味着电脑正在向另一台主机发起**出站 Remote Desktop 连接**，而不是 Anthropic 在远程控制电脑。风险可能来自 RDP 的重定向选项，例如剪贴板、音频、端口或驱动器共享；尤其当被 prompt injection 或不安全的自动化设置引入了恶意 `.rdp` 文件时，风险会更高。
    - 一个以安全为重点的讨论串建议避免使用 `--dangerously-skip-permissions`，改用 Claude Code 的 [**auto mode**](https://code.claude.com/docs/en/auto-mode-config) 作为更安全但并不完美的替代方案，同时禁用“computer use”。如果需要更强的隔离，评论者建议在没有权限访问主机敏感文件或设备的 Linux VM/WSL 环境中运行 Claude Code。
    - 几位评论者指出，用户应该检查 Claude Code 的会话 trace，因为 Claude Code 会公开其推理过程和操作记录。建议的恢复步骤包括：在同一目录中使用 `claude --resume` 恢复之前的会话，然后询问是什么触发了 RDP 启动；或者使用 `/btw` 进行查询，以避免继续沿用同一条操作路径。还有一位评论者认为，截图显示的是一次试图发起出站 RDP 连接的操作；而关于一个很小的、被远程控制的 File Explorer 窗口的说法，则意味着可能存在另一起入侵或脚本行为，而不是正常的 RDP 表现。




### 2. 物理接口与 Robotics

  - **[Meta 改进 Brain2QWERTY：一种能够从脑活动中解码文字、借助非侵入式 MEG 和 EEG 技术实现打字的系统](https://www.reddit.com/r/singularity/comments/1uisr5i/meta_improves_brain2qwerty_a_system_that_can/)**（活跃度：808）：****据报道，**Meta** 改进了 **Brain2QWERTY**，这是一种非侵入式的脑到文字系统，旨在利用 **MEG** 和 **EEG** 从脑活动中解码用户输入的文字。不过，由于链接到的 Reddit 视频/文章被 `403 Forbidden` 拦截，无法访问，因此无法从原始来源获得基准测试数据、架构细节、数据集说明或错误率对比。评论中唯一的技术材料是一张图片链接，但提供的数据没有说明其内容。**评论区的讨论大多停留在推测层面：一位用户调侃未来可能出现“Ad2Brain”之类的应用，另一位用户则提出了一个相关的认知神经科学问题：这种解码是否依赖内心独白，或依赖其他语言生成信号。


  - **[与此同时，在中国，`10,000+` 个配送机器人正在改变最后一公里履约，让配送更快、更便宜，也更加自动化](https://www.reddit.com/r/singularity/comments/1uhxshz/meanwhile_in_china_10000_delivery_bots_are/)**（活跃度：2715）：**一篇 Reddit 帖子声称，**中国已经部署了 `10,000+` 个自动配送机器人**，用于最后一公里物流。这意味着配送机器人可能通过在人行道或道路边缘行驶，以更低成本、更快速度完成配送；不过，相关 Reddit 视频（[v.redd.it/ub2ct1a731ah1](https://v.redd.it/ub2ct1a731ah1)）因 **403 Forbidden** 无法访问，因此无法核实车辆型号、自动驾驶技术栈、载荷、路径规划或车队运营方等技术细节。评论中最相关的技术问题集中在尚未解决的“最后 `50 m/yd`”交接环节：卡车或机器人是否会停在路边，以及包裹如何从道路边缘转交给收件人。**评论者将部署可行性与其他市场中的破坏风险进行了对比，并提到英国的配送机器人据称曾被人扯掉天线；还有人开玩笑地讨论了其被用于反乌托邦式滥用的可能性。除此之外，评论区没有实质性的技术讨论。

    - 一位评论者提出了配送机器人落地时的关键问题：这些机器人在完成街道层面的自动运输后，如何处理**最后 `50m/50yd` 的交接**——例如，卡车或机器人是把包裹放在路边、驶近住户门口，还是要求客户到道路边缘取件。这反映出目前在从路边到家门口的导航、安全放件，以及配送完成时的人机交互等运营细节方面，仍有许多问题尚未解决。




# AI Discord 社区

很遗憾，Discord 今天关闭了我们的访问权限。我们不会以目前这种形式恢复服务，但很快会推出全新的 AINews。感谢你一直读到这里，这段旅程曾经很美好。