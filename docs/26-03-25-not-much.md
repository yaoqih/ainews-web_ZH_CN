---
companies:
- anthropic
- langchain
- arcprize
- primeintellect
date: '2026-03-24T05:44:39.731046Z'
description: '由 **@arcprize** 和 **François Chollet** 推出的 **ARC-AGI-3** 基准测试重新定义了通用智能体推理（general
  agentic reasoning）的前沿。在相关任务中，人类的解决率高达 100%，而当前模型则不足 1%，该基准重点关注“零准备泛化”以及“类人学习效率”。


  由于其采用了严苛的基于效率的评估指标（相较于早期的 ARC 版本和 **NetHack** 等其他基准），其评分协议引发了广泛争论。社区一致认为，该基准凸显了当前大语言模型（LLM）智能体在交互式、稀疏反馈环境中的短板。


  与此同时，智能体基础设施也取得了长足进步：**LangChain** 推出了用于可重用领域知识的 Fleet 可共享技能（shareable skills）；**Anthropic**
  则展示了 **Claude Code 自动模式**，利用分类器介导的审批机制来平衡自主权与手动确认。此外，浏览器和代码智能体正从单纯的“提示词封装器”（prompt
  wrappers）演变为可训练系统，**BrowserBase** 与 **Prime Intellect** 的合作便是这一趋势的缩影。'
id: MjAyNS0x
models:
- arc-agi-3
- claude-code
people:
- fchollet
- mikeknoop
- scaling01
- _rockt
- mark_k
- andykonwinski
- bradenjhancock
- jeremyphoward
- togelius
- bracesproul
- hwchase17
- caspar_br
- _catwu
title: 今天没发生什么特别的事。
topics:
- agentic-reasoning
- interactive-environments
- benchmarking
- efficiency-metrics
- zero-preparation-generalization
- agent-infrastructure
- trainable-agents
- classifier-approval
---

**平静的一天。**

> 2026年3月23日至3月24日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增的 Discord。 [AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提示一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择订阅/退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件推送频率！

---

# AI Twitter 回顾

**ARC-AGI-3 发布、评分争议及其衡量指标解析**

- **ARC-AGI-3 重新定义了“通用” Agent 推理的前沿**：[@arcprize](https://x.com/arcprize/status/2036860080541589529) 和 [@fchollet](https://x.com/fchollet/status/2036861192619384989) 推出了 **ARC-AGI-3**，这是一个围绕拼图/游戏类环境构建的新型交互式基准测试。据报道，**人类能解决 100%** 的任务，而目前的尖端模型得分**低于 1%**。Chollet 将该基准测试定义为衡量系统是否能在**无需人类干预**的情况下，以**类人的学习效率**处理**新任务**，而不是通过特定任务的 Harness 或先前的暴露来取得优异成绩 ([1](https://x.com/fchollet/status/2036866189587271797), [2](https://x.com/fchollet/status/2036877742428610625), [3](https://x.com/fchollet/status/2036879665655406944))。该项目还围绕评测本身进行了大量的工程化产品化工作，包括 [@mikeknoop](https://x.com/mikeknoop/status/2036904122549751907) 强调的用于验证得分的重放系统。
- **目前的争议点在于评分协议，而非核心任务设计**：很大一部分技术讨论集中在 ARC-AGI-3 **基于效率的评分方式**上，该方式将 Agent 与**人类第二佳的操作步数**进行对比，并对额外步数进行重罚。[@scaling01](https://x.com/scaling01/status/2036864865307177430) 认为这使得标题中的“<1%”难以与之前的 ARC 版本进行比较，且可能比单纯的完成度指标（completion metric）更严苛；相关讨论批评了对超人效率的上限限制，以及排除了更丰富的 Agent Harness 或更长思考模式的做法 ([1](https://x.com/scaling01/status/2036853669065306534), [2](https://x.com/scaling01/status/2036856362643210580), [3](https://x.com/scaling01/status/2036866103884775654), [4](https://x.com/scaling01/status/2036890367803429230))。Chollet 回应称这是有意为之：该基准测试明确关注**零准备泛化**，而不是人类针对任务定制系统的能力 ([1](https://x.com/fchollet/status/2036870715392352751), [2](https://x.com/fchollet/status/2036868401843626302))。[@_rockt](https://x.com/_rockt/status/2036864121585438995) 提供了一个有价值的外部批评，他反驳了 ARC-AGI-3 是*唯一*未饱和 Agent 基准测试的说法，并引用了 **NetHack** 作为例证。
- **社区的早期反馈**：即使是批评者通常也同意，该基准测试揭示了当前 LLM Agent 在**交互式、稀疏反馈环境**中的真实弱点。支持性观点来自 [@mark_k](https://x.com/mark_k/status/2036882659406762031)、[@andykonwinski](https://x.com/andykonwinski/status/2036870772745261202) 和 [@bradenjhancock](https://x.com/bradenjhancock/status/2036879154772402636)；而持怀疑但积极态度的反应来自 [@jeremyphoward](https://x.com/jeremyphoward/status/2036891190646432042) 和 [@togelius](https://x.com/togelius/status/2036989880887050333)，后者将“通用游戏竞技 (General Game Playing)”与被过度解读的 AGI 概念区分开来。

**Agent 基础设施、Harness 以及企业级产品化**

- **Agent 技术栈正变得更加标准化且更易于部署**：多次发布都围绕着同一个主题：团队正将可复用的 **skills**、**harnesss** 和 **sandboxes** 作为一等公民级的产品原语进行封装。[@LangChain](https://x.com/LangChain/status/2036858148850671903) 推出了 **Fleet shareable skills**，这是一个用于在不同 Agent 之间将可复用的领域知识代码化的注册表，[@BraceSproul](https://x.com/BraceSproul/status/2036875457258471600)、[@hwchase17](https://x.com/hwchase17/status/2036860332501852227) 和 [@caspar_br](https://x.com/caspar_br/status/2036861283639967986) 对此发表了相关评论。[@AnthropicAI](https://x.com/AnthropicAI/status/2036944806317088921) 发布了 **Claude Code auto mode** 的工作原理，将“分类器介导的审批”描述为全手动确认与无限制自治之间的折中方案；[@_catwu](https://x.com/_catwu/status/2036852880624541938) 指出该功能目前已在内部广泛使用，并向 Team 用户开放。
- **浏览器、编程和工作流 Agent 正在演变为可训练的系统，而非仅仅是 Prompt wrappers**：[@browserbase](https://x.com/browserbase/status/2036851528586453300) 与 Prime Intellect 合作，允许用户在 **BrowserEnv** 上训练自定义的 **browser agents**，[@PrimeIntellect](https://x.com/PrimeIntellect/status/2036886318945624110) 随后进行了跟进，[@willccbb](https://x.com/willccbb/status/2036869858349224447) 则在 `verifiers` 中加入了对 BrowserEnv 的支持。[@cursor_ai](https://x.com/cursor_ai/status/2036873885665419773) 推出了 **self-hosted cloud agents**，将执行过程和代码保留在客户自己的网络内。[@imbue_ai](https://x.com/imbue_ai/status/2036852078627492269) 介绍了 **Keystone**，这是一个能够为任意仓库生成 dev containers 的自配置 Agent；[@SierraPlatform](https://x.com/btaylor/status/2036858449032863898) 推出了 **Ghostwriter**，一个“用于构建 Agent 的 Agent”，专门处理跨聊天、电话、多语言交互、工具使用和 Guardrails 的客户体验流。
- **“Agent = App” 的论点正日益获得基础设施的支持**：多篇文章将 Agent 描述为软件入口点，而不仅仅是助手。[@Base44](https://x.com/Base44/status/2036844452921397266) 强调了跨 Gmail/Calendar/Drive/Outlook 的事件驱动型应用行为。[@weaviate_io](https://x.com/weaviate_io/status/2036814375403528412) 发布了 **Agent Skills**，使编程 Agent 能够使用最新的 Weaviate API，而不是产生过时语法的幻觉。[@ben_burtenshaw](https://x.com/ben_burtenshaw/status/2036827952588234783) 展示了一种实用模式，通过 Hugging Face buckets 为 Codex/Claude 提供 **共享持久工作区**。[@gneubig](https://x.com/gneubig/status/2036949907311915378) 提出了一个更具战略意义的框架，认为 **LLMs as infra** 与 **agent harnesses as apps** 之间现在存在一种真正的相互依赖关系，类似于早期的硬件/架构耦合。

**模型与研究发布：多模态、世界模型与自我改进**

- **Google 将 Lyria 3 扩展为一个更完整的音乐生成平台**：[@Google](https://x.com/Google/status/2036836307612119488)、[@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2036836176233918707) 和 [@GeminiApp](https://x.com/GeminiApp/status/2036836190431711500) 宣布了 **Lyria 3 Pro**，它将生成时长从 **30 秒延长至最高 3 分钟**，增加了对**歌曲结构**（如前奏/主歌/副歌/间奏）的更好控制，并可在 **Gemini** 以及通过 **Google AI Studio / Gemini API** 使用。[@_philschmid](https://x.com/_philschmid/status/2036841210770333998) 总结了定价：Pro 版为 **$0.08/每首歌**，Clip 版为 **$0.04/每首歌**，具备节奏控制、时间对齐歌词、图像转音乐输入以及 **SynthID** 水印功能。
- **LongCat-Next 是来自美团的一个值得关注的开源多模态发布**：[@Meituan_LongCat](https://x.com/Meituan_LongCat/status/2036861293140054510) 推出了 **LongCat-Next**，这是一个 **总参数 68.5B / 激活参数 3B 的 MoE** 离散原生自回归多模态模型，在统一的 token 空间内覆盖了语言、视觉和音频。该发布强调了**原生离散多模态**、任意分辨率视觉分词器（**dNaViT**）、OCR/GUI/文档理解、图像生成以及语音理解/合成。与此同时，[@teortaxesTex](https://x.com/teortaxesTex/status/2036896514157502749) 强调了该报告中围绕统一潜空间/token 路径的架构思想，尽管他对其图像生成质量的评价较低。
- **世界模型和自我改进型 Agent 是当天的研究热点**：[@BrianRoemmele](https://x.com/BrianRoemmele/status/2036826341581185171) 重点介绍了 **LeWorldModel**，这是一个紧凑的 JEPA 风格世界模型，仅用 **15M 参数**和**单个 GPU** 即可从原始像素进行训练，仅需**两个损失项**，据报道其潜空间规划速度更快；声称的简化在于 **SIGReg** 稳定了训练，而无需通常的 JEPA 技巧堆栈。在 Agent 方面，[@omarsar0](https://x.com/omarsar0/status/2036828723878793335) 和 [@fancylancer3991](https://x.com/fancylancer3991/status/2036793932512657664) 介绍了 **Hyperagents**，其自我改进过程本身变得可编辑；报告的提升包括论文评审准确度从 **0.0 提升至 0.710**，机器人奖励设计从 **0.060 提升至 0.372**。相关的记忆研究来自 [@dair_ai](https://x.com/dair_ai/status/2036885342134173915) 的 **MemCollab**，它尝试将通用任务知识与模型特定的偏差分离，以实现跨 Agent 的记忆共享。
- **Sakana AI 的 “AI Scientist” 达到了出版里程碑**：[@SakanaAILabs](https://x.com/SakanaAILabs/status/2036840833690071450)、[@hardmaru](https://x.com/hardmaru/status/2036841736702767135) 和 [@jeffclune](https://x.com/jeffclune/status/2036866082418680297) 指出，**The AI Scientist** 现已在 **Nature** 上发表，整合了早期的系统和 v2 版本的更新。值得注意的主张不仅是创意生成、实验、起草和自动化评审的全流程自动化，还有**“科学规模法则”（scaling law of science）**的证据：更强大的底层基础模型能产出更好的机器生成论文。

**推理、存储与本地硬件经济学**

- **存储和 Artifact 迁移正变得更便宜且对 Agent 更友好**：[@fffiloni](https://x.com/fffiloni/status/2036736853991166120) 预告了 Hugging Face 的存储计划，并称“你的磁盘不再是限制”；同时 [@LoubnaBenAllal1](https://x.com/LoubnaBenAllal1/status/2036778058439385568) 和 [@victormustar](https://x.com/victormustar/status/2036792818274865469) 认为 **HF Buckets** 在 **$/TB/每月**和传输性能上均优于 **S3**，并指出 **Xet 风格的块级去重 (chunk-level deduplication)** 是数据集和 Checkpoint 的重大胜利。[@francoisfleuret](https://x.com/francoisfleuret/status/2036738024176779535) 在操作层面询问集群运营商，Agent 对 **I/O** 的压力究竟有多大。
- **推理效率在各种运行时和架构中仍是一个快速演进的竞争战场**：[@sudoingX](https://x.com/sudoingX/status/2036795152178794993) 报告了 **NVIDIA 的 3B Mamba2 Nemotron Cascade 2** 异常强大的单 GPU 长上下文吞吐量，声称在 **RTX 3090** 上，**625K 上下文**下仍能保持 **187 tok/s**，而带有 KV 量化的 **Qwen 3.5 35B-A3B** 在 **262K** 上下文下为 **112 tok/s**。[@finbarrtimbers](https://x.com/finbarrtimbers/status/2036807872328466621) 注意到 Cursor 的 Composer 2 报告中由于 **Fireworks** 在 RL 推理上相比 **SGLang/TRT** 等典型堆栈具有巨大的效率优势，因而采用了 Fireworks；[@GoogleCloudTech](https://x.com/GoogleCloudTech/status/2036790201813442575) 发布了针对 **TPU v7x / Ironwood** 进行前沿模型训练的优化指南。在量化/压缩方面，[@mirrokni](https://x.com/mirrokni/status/2036905273999200481) 关注了 Google 的 **TurboQuant** 报告，其速度提升达 **6 倍**，而 [@vllm_project](https://x.com/vllm_project/status/2036989821156270501) 强调了在紧凑型硬件上实现了 **4M+ KV-cache token**。
- **本地 AI 硬件获得了两个引人注目的数据点**：[@digitalix](https://x.com/digitalix/status/2036820057599197645) 重点介绍了 Intel 新推出的 **Arc Pro B70**，其配备 **32GB 显存且价格低于 1000 美元**，尽管存在软件栈方面的限制，仍被多位博主视为“**单位美元显存容量 (VRAM-per-dollar)**”的一次重要动作（[示例](https://x.com/QuixiAI/status/2036922193897017750)）。另外，[@xenovacom](https://x.com/xenovacom/status/2036908326462665211) 演示了通过 **WebGPU/Transformers.js** 在浏览器中运行 **24B 模型**，在 **M4 Max** 上达到约 **50 tok/s**，这预示着浏览器端推理上限正在迅速提升。

**热门推文（按互动量排序）**

- **个性化与记忆质量**：[@karpathy](https://x.com/karpathy/status/2036836816654147718) 认为助手中的长期记忆（long-lived memory）往往会过拟合过时的用户事实，导致干扰性的、低质量的个性化，而非更好的辅助功能。
- **Claude 作为“超级应用”的叙事**：[@kimmonismus](https://x.com/kimmonismus/status/2036856308410773803) 和 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2036852118762700929) 都指出 Anthropic 的产品轨迹越来越像一个**超级应用 (super-app)**，而不仅仅是一个单一的模型端点 (model endpoint)。
- **Codex 生态系统活跃度**：[@OpenAIDevs](https://x.com/OpenAIDevs/status/2036851146300809531) 启动了**学生 Codex 创作者挑战赛**，提供 API 额度奖励和启动额度；[@reach_vb](https://x.com/reach_vb/status/2036904822641676716) 也提醒开发者 **Codex App Server** 是开源的。
- **淡化 Sora 是战略重心的转移**：虽然大部分传闻是二手的，但多个汇总和评论帖子指出 OpenAI 正在缩减 **Sora** 的投入，转而优先考虑编程/Agent 产品和核心基础设施，[@TheRundownAI](https://x.com/TheRundownAI/status/2036752541581447214) 和 [@thursdai_pod](https://x.com/thursdai_pod/status/2036983766418403692) 将其视为当天重大的行业信号之一。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Intel GPU 发布及其特性

  - **[Intel 将于下周发售一款拥有 32GB 显存的廉价 GPU](https://www.reddit.com/r/LocalLLaMA/comments/1s3e8bd/intel_will_sell_a_cheap_gpu_with_32gb_vram_next/)** (活跃度: 1300): **Intel** 将于 3 月 31 日发布一款配备 `32GB VRAM` 的新 GPU，售价为 `$949`。该 GPU 提供 `608 GB/s` 的带宽，功耗为 `290W`，在带宽方面略低于 NVIDIA 5070。预计这款 GPU 将有利于本地 AI 应用，特别是对于像 Qwen 3.5 27B 的 `4-bit quantization` 模型。更多详情请参阅 [PCMag 的文章](https://www.pcmag.com/news/intel-targets-ai-workstations-with-memory-stuffed-arc-pro-b70-and-b65-gpus)。** 评论者对 `$989` 的价格是否能被称为“廉价”表示怀疑，而另一些人则将其与 R9700 AI PRO 进行对比，指出两者显存和带宽相似，但 Intel 的功耗略高。人们对 Intel 的产品将如何竞争持好奇态度，尤其是针对 AI 和 LLM 应用。

    - Clayrone 讨论了他们使用 R9700 AI PRO 的经验，强调了其 32GB VRAM 和 640 GB/s 的带宽，认为这对于他们的小型服务器构建来说非常理想。他们提到使用了为 Vulkan 构建的 `llama.cpp`，运行非常完美，并指出该 GPU 的功耗为 300W。他们对 Intel 即将推出的 GPU 与之相比的表现感到好奇，认为它可能是一个直接竞争对手。
    - KnownPride 认为 Intel 发布 32GB VRAM GPU 的决定具有战略意义，因为它迎合了对大语言模型 (LLMs) 日益增长的需求。这表明了一种市场趋势，即消费者对能够支持 AI 和机器学习工作负载（这需要大量 VRAM）的硬件越来越感兴趣。
    - wsxedcrf 引用了 NVIDIA 的一句话，“免费还不够便宜”，以强调 GPU 的价值不仅在于其价格，还在于它所支持的整个生态系统。这暗示 Intel 新 GPU 的成功将不仅仅取决于硬件规格；周边的软件和支持基础设施也将至关重要。

  - **[Intel 发布配备 32GB GDDR6 的 Arc Pro B70 和 B65](https://www.reddit.com/r/LocalLLaMA/comments/1s3bb3y/intel_launches_arc_pro_b70_and_b65_with_32gb_gddr6/)** (活跃度: 493): **Intel** 发布了 **Arc Pro B70** 和 **B65** GPU，配备 `32GB GDDR6` 显存。B70 售价为 `$949`，提供 `387 int8 TOPS` 的算力和 `602 GB/s` 的显存带宽，相比之下，**NVIDIA RTX 4000 PRO** 为 `1290 int8 TOPS` 和 `672 GB/s`。B70 的功耗为 `290W`，高于 RTX 4000 的 `180W`。4 张 B70 的组合包售价为 `$4,000`，提供 `128GB` 的 GPU 显存，这被认为是 `70B` 模型本地推理的一个极具竞争力的方案。[来源](https://videocardz.com/newz/intel-launches-arc-pro-b70-at-949-with-32gb-gddr6-memory)。** 评论者强调了 **Intel** 与 **vLLM** 的合作，将 B 系列支持集成到 vLLM 主线中，确保了首日支持和稳定的性能。`32GB` 显存、`$949` 的价格被认为对本地推理非常有利，使其在运行 `70B` 模型时具有实用性。

    - Intel 与 vLLM 的合作将 B 系列支持集成到 vLLM 主线中，确保了 Arc Pro B70 和 B65 GPU 在发布首日即获得支持并拥有稳定性能。然而，B70 的性能落后于 RTX 4000 PRO，其 int8 TOPS 为 387，而 4k PRO 为 1290。B70 提供 602 GB/s 的显存带宽，而 4k 为 672 GB/s，虽然它拥有更多显存（32GB 对比 24GB），但功耗也更高（290W 对比 180W）。
    - Arc Pro B70 售价为 $949，凭借其每 GB 价格优势，使其成为本地推理（尤其是 70B 模型）的一个有吸引力的选择。这使其成为那些需要大量显存容量、但又不想支付像 RTX 3090 等其他 GPU 高额成本的人的务实选择。
    - 尽管 Arc Pro B70 的推理速度比 RTX 3090 慢且缺乏 CUDA 支持，但它提供了更多显存并提高了效率，这可以增强 Prompt 处理能力。然而，用户对 Intel 的驱动支持表示担忧，这可能会影响整体用户体验。

### 2. LiteLLM 供应链攻击及其替代方案

  - **[在供应链攻击之后，这里有一些 litellm 的替代方案](https://www.reddit.com/r/LocalLLaMA/comments/1s34173/after_the_supply_chain_attack_here_are_some/)** (Activity: 372): **图片是 Andrej Karpathy 讨论 Python 包 litellm 遭遇供应链攻击的一条推文，该包的 `1.82.7` 和 `1.82.8` 版本被植入了窃取凭据的恶意软件。这次攻击突显了软件开发中依赖管理的风险，因为受损的包可能会窃取 SSH 密钥和数据库密码等敏感数据。帖子建议了一些 litellm 的替代方案，例如 Bifrost、Kosong 和 Helicone，它们各自提供不同的功能和性能优势，比如 Bifrost 的 `P99 latency` 比 litellm 快约 50 倍，而 Helicone 则拥有广泛的 Provider 支持和分析能力。** 评论者对 Python 和 Node.js 项目中庞大的依赖树风险表示担忧，认为这会导致漏洞和可靠性问题。他们建议采取限制网络访问、固定（pinning）依赖版本以及监控网络流量等做法，以减轻与供应链攻击相关的风险。

    - FullstackSensei 强调了 Python 和 Node.js 项目中巨大依赖树的问题，指出即使是小型项目也可能有数 GB 的依赖项。这种复杂性往往导致由于担心引入 Bug 而不经常更新，从而产生漏洞。评论建议需要更多关于管理和最小化依赖链的讨论，以提高可靠性和安全性。
    - _realpaul 讨论了缓解供应链攻击的策略，强调了限制网络访问、避免立即采用新库以及固定依赖版本的重要性。他们还建议在 Sandbox 环境中运行工具，并在部署前监控网络流量以增强安全性。
    - RoomyRoots 和 Living_Director_1454 都指出了对第三方库的过度依赖，这增加了供应链攻击的风险。Living_Director_1454 提到了一起涉及被入侵的安全扫描器 Trivy 的具体事件，该工具曾用于 LiteLLM 的 CI/CD 流水线，说明了软件供应链中的潜在漏洞。

  - **[PyPI 上的 Litellm 1.82.7 和 1.82.8 已被破坏，请勿更新！](https://www.reddit.com/r/LocalLLaMA/comments/1s2c1w4/litellm_1827_and_1828_on_pypi_are_compromised_do/)** (Activity: 555): **经 FutureSearch.ai 确认，PyPI 上的 `litellm` 包版本 `1.82.7` 和 `1.82.8` 已被入侵。这次攻击似乎是一次供应链破坏，可能影响数千名用户。该漏洞由 Callum McMahon 发现，他在[此处](https://futuresearch.ai/blog/no-prompt-injection-required)提供了详细的事后分析。攻击是通过 LiteLLM CEO 的 GitHub 账号执行的，该账号被黑客入侵，导致代码库发生未经授权的更改，包括一条声明 *"teampcp owns BerriAI"* 的消息。这一事件突显了 AI 工具中日益增长的供应链攻击风险，强调了在生产环境中固定版本和谨慎更新的重要性。** 评论者强调了固定依赖版本和避免在生产环境中自动更新的重要性，以减轻供应链攻击的风险。此外，人们还担心讨论中可能存在自动化机器人，证据是出现了重复且缺乏实质内容的评论。

    - 据报道，被入侵的 LiteLLM 版本 1.82.7 和 1.82.8 被植入了恶意代码，如果系统的时区设置为 Asia/Tehran，该代码将执行破坏性命令（`rm -rf /`）。这突显了 AI 工具中供应链攻击的关键风险，强调了在生产环境中固定依赖版本和避免自动更新的重要性。
    - 攻击似乎是由一个名为 'teampcp' 的组织执行的，他们之前曾入侵过 Trivy。他们通过 LiteLLM CEO **Krrish Dholakia** 的 GitHub 账号获得了访问权限，并利用该账号推送了在 LiteLLM 启动时窃取 Secret 的恶意软件。这一事件强调了高知名度账号的脆弱性，以及当它们被入侵时可能产生的广泛影响。
    - LiteLLM CEO 的 GitHub 仓库被修改，显示消息 'teampcp owns BerriAI'，表明发生了入侵。CEO 的账号被用于进行未经授权的 Commit，暗示了重大的安全漏洞。建议用户使用 <= 1.82.6 的版本，因为这些版本已确认不含恶意代码。

### 3. 新发布的 AI 模型与基准测试

  - **[新开放权重模型：GigaChat-3.1-Ultra-702B 和 GigaChat-3.1-Lightning-10B-A1.8B](https://www.reddit.com/r/LocalLLaMA/comments/1s2pkfw/new_open_weights_models_gigachat31ultra702b_and/)** (热度: 624): **GigaChat-3.1-Ultra-702B** 和 **GigaChat-3.1-Lightning-10B-A1.8B** 是 AI Sage 最新发布的开放权重模型，采用 MIT 许可证，可在 [Hugging Face](https://huggingface.co/collections/ai-sage/gigachat-31) 上获取。Ultra 模型是一个 `702B MoE`，针对高资源环境进行了优化，在 `MMLU RU` 和 `Math 500` 等基准测试中表现优于 DeepSeek-V3-0324 和 Qwen3-235B 等模型。Lightning 模型是一个 `10B A1.8B MoE`，目标是本地推理，通过支持 `原生 FP8 DPO` 和 `MTP` 实现了高效率，并在 14 种语言的多语言任务中表现出色。两款模型都针对英语和俄语进行了优化，其中 Lightning 模型在 BFCLv3 基准测试中得分为 `0.76`。详细指标显示，与之前的版本及竞争对手相比，新模型在通用知识、数学和编码领域有显著提升。评论中强调了地缘政治方面的担忧，指出这些模型在俄罗斯开发，训练数据可能受到国家影响，且使用受俄罗斯管辖的基础设施可能面临当地情报部门访问的风险。

    - **Specialist-Heat-6414** 强调了 GigaChat-3.1-Ultra-702B 模型在技术上的重要性，指出在 MIT 许可证下发布 702B MoE (Mixture of Experts) 模型是对开放权重生态系统的重大补充。无论其开发背后的地缘政治背景如何，这一贡献都值得关注。
    - **Qwen 对比** 是讨论的焦点，用户认为需要与 Qwen 3.5 等模型进行基准对比，以确立 GigaChat 模型的地位。评论指出，在 2026 年，仅仅“优于 GPT-3.5”已不再是一个足够有说服力的基准，这表明需要更严格的评估指标。
    - **Investolas** 等人对 GigaChat-3.1-Lightning-10B-A1.8B 模型表现出浓厚兴趣，特别是其在本地推理方面的潜力。如果该模型的激活参数量在 1.8B 左右，且能在保持质量的同时在单张 GPU 上达到 250+ tokens per second，那么它在消费级硬件上将非常具有实用性，成为该领域的一项重要进展。

  - **[DeepSeek 员工暗示将推出超越 DeepSeek V3.2 的“巨型”新模型](https://www.reddit.com/r/LocalLLaMA/comments/1s39024/deepseek_employee_teases_massive_new_model/)** (热度: 427): **据传一名 **DeepSeek** 员工泄露的消息称，正在开发一款能力超越 **DeepSeek V3.2** 的新模型。该泄露消息很快被删除，其中暗示了模型架构的重大进步，可能涉及与 **SillyTavern**、**MiniMax**、**ZAI** 和 **Moonshot** 等平台的集成。然而，这条泄露消息的真实性随后被证实为伪造，正如一则 [推文](https://nitter.net/victor207755822/status/2036814461085110764) 所确认的那样。** 评论者表达了希望 DeepSeek 在激烈的竞争中平衡发布时机的愿望，一些人希望看到新模型的小型高效版本。此外，对于提及使用多个平台的情况，人们感到惊讶，这表明了一种广泛的集成策略。

    - TheRealMasonMac 强调了 DeepSeek 对包括 SillyTavern、MiniMax、ZAI 和 Moonshot 在内的多个 AI 平台的使用，这暗示了一种可能增强创新的广泛集成策略。这表明 DeepSeek 正试图利用多样的 AI 技术来潜在地提升其模型能力。
    - ambient_temp_xeno 对新模型潜在的资源需求表示担忧，暗示它可能对于个人使用来说要求过高。这反映了 AI 发展中的一个普遍问题，即新模型通常需要更多的算力，从而限制了个体用户的可访问性。


## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Sora 关闭及其影响

- **[OPENAI TO DISCONTINUE SORA !!](https://www.reddit.com/r/OpenAI/comments/1s2onhy/openai_to_discontinue_sora/)** (Activity: 2452): **OpenAI** 准备停止其 Sora Video Platform App，该应用是在去年推出的。该应用允许用户将自己置入著名的电影场景中，但因其限制过多且不够用户友好而受到批评。在财务上，它是不可持续的，据报道每天亏损 `$500k`。这一决定反映了 AI 项目在资源分配方面的更广泛担忧，强调了需要仔细考虑此类技术的价值和影响。评论者大多认为 Sora 是一个资源密集型项目，其实际价值有限，凸显了在 AI 开发中评估资源成本与收益的重要性。

    - **TheTeflonDude** 强调了一个重大的财务问题，指出 OpenAI 在 Sora 上每天亏损 `$500k`，这可能促成了停止该服务的决定。这凸显了维持此类平台相关的高昂运营成本，尤其是当它无法产生足够的收入或用户参与度来证明支出合理时。
    - **Willing_Leave_2566** 讨论了由 Sora 等平台支持的低门槛内容创作带来的更广泛影响。他们认为，如果没有努力门槛，用户可能不会考虑其创作的资源成本，从而导致算力资源 (compute resources) 的低效使用。这反映了对开放创意平台的可持续性和价值的批判性观点。
    - **Pakh** 通过引用之前 Disney 与 OpenAI 的合作提供了战略转变的背景，当时 Disney 投资了 `$1 billion` 并且为 Sora 授权了超过 `200 characters`。这种伙伴关系原本有望增强 Sora 对粉丝创作内容的吸引力，因此停止服务令人惊讶，并预示着 OpenAI 的重大战略转向。

  - **[Sora is officially shutting down.](https://www.reddit.com/r/OpenAI/comments/1s2oyl3/sora_is_officially_shutting_down/)** (Activity: 1954): **该图片是 Sora 应用在 X.com 官方账号的截图，宣布将关闭 Sora 应用。该公告对用户表示感谢，并提到即将提供有关应用和 API 时间表的更多信息。这标志着对依赖 Sora 服务的用户和开发者的重大变化，因为他们将需要转向替代解决方案。** 评论反映了幽默与批评的交织，一位用户讽刺地指出了该应用的喜剧价值，另一位用户则对失去生成争议性内容的功能表示担忧。


  - **[SORA IS SHUTTING DOWN???](https://www.reddit.com/r/OpenAI/comments/1s2rqbg/sora_is_shutting_down/)** (Activity: 1234): **OpenAI** 宣布将关闭 **Sora**（其视频生成应用和 API），尽管它最近作为 App Store 排名第一的应用大受欢迎。这一决定出人意料，尤其是在最近发布了关于 Sora 安全标准的博客文章之后。据报道，关闭是为了将算力资源 (compute resources) 重新分配给编程和企业级应用，这可能受到了 **Anthropic** 专注于编程而非视频的影响。这一举动中断了与 **Disney** 的重要合作伙伴关系，其中包括与 Marvel、Pixar 和 Star Wars 的合作。随着创作者迁移到 Runway 和 Kling 等其他平台，AI 视频领域预计将经历一场变革。一些评论者认为，由于性能差且成本高，Sora 的关闭是必然的，并认为严肃的 AI 视频创作者并未广泛使用它。其他人对这一突然决定表示惊讶，指出了该应用之前的显赫地位。

    - **echox1000** 强调 Sora 由于高昂的算力成本和糟糕的性能而成为财务负担，暗示其关闭是必然的。评论者对该项目能维持这么长时间表示惊讶，表明其结果低于预期。
    - **bronfmanhigh** 指出 Sora 在 AI 视频创作领域没有竞争力，因为没有合法的创作者在使用它。这表明 Sora 在功能和采用方面显著落后于其他工具，这可能导致了它的关闭。
    - **KnightAirant** 批评了 Sora 未能开源 (open-sourcing)，暗示 OpenAI 中的 “Open” 具有误导性。该评论反映了一种观点，即该项目寿命很短，持续不到一年，并对大公司 AI 项目的透明度和可访问性提出质疑。

- **[再无 Sora ..?](https://www.reddit.com/r/StableDiffusion/comments/1s2pjf0/no_more_sora/)** (活跃度: 1061): **图片是来自 Sora 官方账号的一条推文，宣布停止 Sora 应用。推文表达了对社区的感谢，并承认了可能带来的失望，同时承诺将进一步更新应用和 API 的时间表，以及关于保留用户作品的信息。这表明对于依赖 Sora 的用户来说，这是一个重大的转变，可能会影响依赖其服务的工作流。** 评论反映出一种观点，即本地解决方案更可靠，因为像 Sora 这样的中心化服务可能会被关停。此外，还有人呼吁将该应用开源 (open-sourcing)，反映了对社区驱动开发和控制权的渴望。

    - PwanaZana 强调了由于硬件限制，在本地运行大型 AI 模型所面临的挑战，并强调了对可以在性能较低的机器上运行的小型、高效模型的需求。这反映了将 AI 优化以进行本地部署、平衡性能与可访问性的更广泛趋势。
    - Sudden-Complaint7037 指出，投资者对 AI 盈利能力的怀疑日益增加，暗示随着公司重新考虑其投资，行业正在发生转变。这表明 AI 商业模式可能会被重新评估，重点将放在可持续和盈利的策略上。

  - **[Sora 正式关停。](https://www.reddit.com/r/ChatGPT/comments/1s2oxnu/sora_is_officially_shutting_down/)** (活跃度: 2831): **图片是 Sora 应用官方账号在 X.com 上发布的一份公告截图，声明该应用即将关闭。该消息感谢了用户的贡献，并提到很快将提供有关应用及其 API 关闭时间表的更多细节。这标志着对于依赖 Sora 服务的用户和开发者来说，这是一个重大变化。** 评论反映了对该应用影响力和用户群的怀疑，一些用户对该应用在面临财务挑战的情况下能维持这么久表示惊讶。



### 2. Claude Code 功能与问题

  - **[Claude Code 现已具备自动模式 (auto mode)](https://www.reddit.com/r/ClaudeAI/comments/1s2ok85/claude_code_now_has_auto_mode/)** (活跃度: 962): **Claude Code 推出了一项“自动模式”功能，可自动执行文件写入和 bash 命令的权限决策，取代了手动审批或使用 `--dangerously-skip-permissions` 的需要。此模式采用分类器 (classifier) 来评估每个工具调用是否存在潜在的破坏性操作，允许安全操作自动进行，同时拦截风险操作。该功能目前作为研究预览 (research preview) 在 Team 方案中提供，随后将面向 Enterprise 和 API 用户开放更广泛的访问权限。更多详细信息可以在[此处](http://claude.com/product/claude-code#auto-mode)找到。** 用户对使用额度降低表示极大关注，有报告称会话限制比以前快得多地被触发，尽管 Anthropic 官方尚未发布任何沟通。用户对这些变化缺乏透明度和沟通感到沮丧。

    - 用户在使用 Claude Code 时遇到了严重的额度限制问题，有报告称会话限制达到得比以前快得多。一位使用 Max 5x 方案的用户指出，他们在一天内就用掉了每周限额的 50%，这表明政策可能发生了变化或存在 Bug。Anthropic 缺乏沟通，导致依赖该服务工作的用户感到沮丧。
    - Claude Code 中新的自动模式采用了“执行前分类”的方法，通过默认使用容器或 VMs 等隔离手段来增强安全性。然而，人们担心分类器如何处理模糊的命令，例如区分临时目录中的 `rm -rf` 与项目根目录中的操作。用户建议，一个能为拦截操作提供解释的自动模式，比静默回退（silent fallbacks）更有益。
    - 用户呼吁 Anthropic 在专注于自动模式等新功能之前，先解决速率限制 (rate limit) 问题。用户担心当前的速率限制可能会严重限制新功能的使用，最近的经历证明，用户达到限制的速度远超预期。

- **[说一句 'hey' 消耗了我 22% 的使用限制](https://www.reddit.com/r/ClaudeAI/comments/1s3hh29/saying_hey_cost_me_22_of_my_usage_limits/)** (热度: 883): **这篇 Reddit 帖子强调了 **Claude Code** 的一个重大问题：重新进入不活跃的会话会导致使用限制大幅增加，据报道，发送一条简单的消息就会消耗高达 `22%` 的额度。这归因于系统的缓存机制，即每条消息都会将整个对话上下文（包括 system prompts 和对话历史）重新发送到 API。该缓存的 `TTL` 在 Pro 计划中为 5 分钟，在 Max 计划中为 1 小时，当会话过夜保持开启时缓存会失效，导致恢复会话时触发完整的缓存写入，其成本比普通输入高出 `1.25` 倍。此外，使用情况追踪采用 5 小时的滚动窗口，旧会话累积的上下文可能会被计入新窗口的费用，从而导致意外的使用量激增。一份 GitHub issue 还指出，自 3 月 23 日以来，相同工作负载的使用量有所增加，但 **Anthropic** 尚未给出官方回复。** 评论者认为这是一个已知且正在恶化的系统问题，有些人将其归因于系统故障期间 Claude 的重试机制。建议的解决方法是开启新会话，或使用 `/clear` 和 `/compact` 命令来管理对话历史，避免过度的 token 消耗。

    - Fearless_Secret_5989 解释说，Claude Code 的架构涉及在每条消息中重新发送整个对话上下文，其中包括 system prompts、tool definitions 和对话历史。这会导致极高的 token 使用量，特别是当会话缓存过期时（Pro 计划 5 分钟，Max 计划 1 小时），会触发完整的缓存写入，其成本是普通输入的 1.25 倍。一份 GitHub 追踪记录显示，一个恢复的会话中 92% 的 token 是缓存读取，仅为了极少的输出就消耗了 192K tokens。
    - Fearless_Secret_5989 还强调了速率限制窗口边界问题，Claude Code 使用 5 小时滚动窗口进行使用追踪。如果在一个窗口中开始的会话在另一个窗口中恢复，旧会话累积的上下文可能会计入新窗口，导致使用量激增。有用户报告称，由于这种跨窗口滚动，在没有进行新工作的情况下，瞬间就消耗了 60% 的限额。
    - Fearless_Secret_5989 提到自 3 月 23 日以来，可能存在影响 Max 计划用户的 Bug 或后端更改，以前消耗窗口 20-30% 的工作负载现在消耗 80-100%。Max 5x 和 Max 20x 计划的用户报告称很快就达到了限制，一名用户在发送单个 prompt 后额度从 21% 直接跳到 100%。Anthropic 尚未正式回应，原因尚不明确。

  - **[Claude Code 限制被悄悄降低，情况变得糟糕得多](https://www.reddit.com/r/ClaudeCode/comments/1s2lye7/claude_code_limits_were_silently_reduced_and_its/)** (热度: 1229): ** **Claude Code** 的用户报告称使用限制在未发布公告的情况下显著降低，有人形容为“百倍”缩减。这种变化对于处理 PHP 和 JavaScript 简单项目的用户尤为明显，他们现在达到限制的速度比以前快得多。开发者缺乏透明度导致了用户的挫败感，因为用户感觉对这些变化及其应对方式一无所知。** 一些用户推测这种缩减可能是一个 Bug，而另一些人则认为这可能是掩盖配额削减的策略性举措。一种理论认为，先进行临时的增加然后再大幅削减，可以掩盖永久性的额度降低，让用户对实际限制感到困惑。

    - -becausereasons- 强调了 Claude 代码限制的显著降低，认为由于变化的剧烈程度（被称为“百倍”下降），这可能是一个 Bug。这表明系统中存在一个需要解决的潜在问题。
    - zirouk 提出了一个关于公司如何通过操纵用户感知来掩盖配额削减的理论。他们建议采取一种策略，即先进行临时增加，随后大幅减少，然后进行部分恢复，从而在用户没有意识到全部变化的情况下实现净削减。
    - Dry-Magician1415 批评了 LLM 使用限制缺乏透明度，并将其与电信等更易量化的行业进行了对比。他们认为，如果没有明确的量化和审计，公司可以随意调整限制，导致用户不满和不信任。

- **[Claude Code 现在可以 /dream 了](https://www.reddit.com/r/ClaudeCode/comments/1s2ci4f/claude_code_can_now_dream/)** (Activity: 2731): **Claude Code 的新功能 Auto Dream 解决了由 Auto Memory 功能引起的记忆膨胀（memory bloat）问题。Auto Dream 通过回顾过去的会话记录、识别相关信息并剪枝陈旧或矛盾的记忆，来模仿人类的快速眼动（REM）睡眠。它将这些信息整合到有组织的文件中，用实际日期替换模糊的引用。此过程在后台运行，触发条件为自上次整合以来经过 24 小时且进行了 5 次会话；它对项目代码进行只读操作，同时修改记忆文件。这种方法被比作 AI 记忆的垃圾回收器（garbage collector）和碎片整理程序（defragmenter），提升了记忆管理能力，而不仅仅是扩大 Context Window。** 一些评论者幽默地建议增加额外功能，如用于处理幻觉（hallucinations）的 `/acid` 和用于清理的 `/shit`。另一位评论者指出 Anthropic 尚未发布官方公告，并指向了 Ray Amjad 的 YouTube 讲解。

    - AutoDream 是 Claude Code 的一项新功能，充当其记忆系统的“睡眠周期”，解决了 Auto Memory 功能带来的记忆膨胀问题。AutoDream 的运行分为四个阶段：定向（Orient）、收集信号（Gather signal）、合并（Consolidate）以及剪枝与索引（Prune & index）。它通过扫描现有记忆、识别漂移的记忆、合并新信息并消除矛盾来整合记忆，非常类似于人类的 REM 睡眠。此过程仅修改记忆文件而非实际代码库，确保了安全性。
    - AutoDream 功能旨在通过定期整合和组织存储的信息来优化 Claude Code 的记忆管理。它仅在自上次整合后过去 24 小时以上且进行了 5 次以上会话后运行，确保不会干扰正在进行的工作。该过程涉及扫描记忆目录、识别过时或矛盾的信息，并更新记忆文件以保持索引的简洁和准确，类似于 AI 记忆的垃圾回收器。
    - AutoDream 的系统提示词可在 GitHub 的 Piebald-AI/claude-code-system-prompts 仓库中找到，具体位于 `agent-prompt-dream-memory-consolidation.md` 文件中。该功能可以通过 `/memory` 命令在 Claude Code 中访问，为用户提供了一个有效管理 AI 记忆的工具，通过充当 AI 记忆的碎片整理程序来解决 Context Window 问题。

  - **[Claude 现在可以控制你的鼠标和键盘。我测试了一整天 —— 以下是实际可用的功能。](https://www.reddit.com/r/PromptEngineering/comments/1s2h1h6/claude_can_now_control_your_mouse_and_keyboard_i/)** (Activity: 184): **Claude 的新 Computer Use 功能** 允许它控制 Mac 的鼠标和键盘，执行文件管理、电子表格数据录入和浏览器表单填写等任务。它通过截取屏幕截图来理解屏幕上下文，但由于它会接管整台机器，因此需要用户离开。该功能目前处于 Pro/Max 计划的研究预览阶段，在简单任务上的可靠性为 `80%`，在复杂任务上为 `50%`。然而，它在需要速度的任务、CAPTCHA、2FA 以及复杂交互方面表现吃力。该功能的潜力在于当用户离开时自动执行任务，正如通过 Dispatch 结合远程电话指令所展示的那样。更多细节可以在[完整解析](https://findskill.ai/blog/claude-cowork-guide/#computer-use)中找到。** 评论者对 Claude 控制电脑的安全性及可靠性表示怀疑，担心 CAPTCHA 问题以及被误用的风险。还有人幽默地将其与人工驱动的“AI”农场进行比较，突显了对该技术自主性的怀疑。

    - 一位用户提到使用 Claude 在其应用开发工作流中自动化测试。他们计划推送一个新构建版本，让 Claude 测试更改、提供反馈并修复发现的任何问题。这突显了 AI 通过自动化重复性任务和提高效率来简化软件开发流程的潜力。
    - 存在对安全和隐私的担忧，一位评论者幽默地暗示了随机人员获得其 PC 控制权的可能性。这反映了人们对具有硬件控制权的 AI 系统的广泛担忧，强调了需要强大的安全措施来防止未经授权的访问。
    - 另一位评论者幽默地指出，Claude 无法绕过 CAPTCHA，这些验证码旨在区分人类和机器人。这一局限性强调了 AI 在需要类人感知和决策的任务中面临的挑战，尽管在其他领域有所进步。


### 3. AI Model Releases and Benchmarks

- **[ARC AGI 3 is up! Just dropped minutes ago](https://www.reddit.com/r/singularity/comments/1s3gq6b/arc_agi_3_is_up_just_dropped_minutes_ago/)** (Activity: 1198): **该图片展示了 ARC-AGI-3 排行榜，该榜单根据 AI 模型的性能得分与其运营成本进行评估。图中显示的模型，包括 **Gemini 3.1 Pro (Preview)**、**Anthropic Opus 4.6 (Max)** 和 **Grok 4.20 (Beta Reasoning)**，均位于图表的较低端，表明尽管成本各异，但性能得分相对较低。这种可视化方式突出了当前 AI 模型在实现 AGI 方面的现状，并将 ARC Prize 标记为基准。评论反映了对实现 AGI 进展的怀疑态度，指出尽管投入了巨额资金，但得分百分比仍然较低。** 评论者对当前实现 AGI 的 AI 模型现状表示怀疑，并指出了相对于所涉成本的低性能表现。一条评论强调了感知的 AGI 进展与实际性能指标之间的差距，暗示声称已达到 AGI 的说法还为时过早。

    - 讨论的一个关键点是 AI 模型的基准测试饱和，特别关注 ARC AGI 3 尽管投入巨大（`$10K`），但仅实现了 `0.2%` 的改进。这引发了关于基准测试收益递减的疑问，以及 AI 模型是否仅仅是在针对这些测试进行优化，而没有在泛化能力（generalization capabilities）方面取得真正的改进。
    - 将 GPT-5.4 (High) 作为基准测试中的参考点，突显了顶级 AI 模型之间的竞争格局。对比表明，虽然发布了像 ARC AGI 3 这样的新模型，但它们可能并未显著优于 GPT-5.4 等现有模型，这表明性能提升可能进入了平台期。

  - **[TheInformation reporting OAI finished pretraining new very strong model “Spud”, Altman notes things moving faster than many expected](https://www.reddit.com/r/singularity/comments/1s2q0yb/theinformation_reporting_oai_finished_pretraining/)** (Activity: 931): **据报道，OpenAI 已经完成了一个名为 "Spud" 的新模型的预训练，预计该模型将非常强劲。这一进展发生时，**Sam Altman** 正将关注点从 OpenAI 的安全与保障团队转移到扩展运营上，表明了资源的战略性重新分配。此外，OpenAI 正在关闭 Sora 视频应用，暗示其优先考虑 AI 模型开发而非其他项目。社区正在猜测 OpenAI 预训练模型可能带来的改进，这些模型此前尽管拥有强大的强化学习能力，但仍受到批评。** 一些评论者推测，"Spud" 的发布可能是一个战略性叙述，旨在掩盖 Sora 应用关闭的消息。其他人则强调了改进 OpenAI 预训练模型的重要性，这些模型一直被认为相对于其强化学习（RL）优势而言较弱。

    - Dylan Patel 评论道，OpenAI 以拥有业内最强的强化学习（RL）能力而闻名，但从历史上看，其预训练模型的实力一直没那么强。如果 OpenAI 确实通过新的 "Spud" 模型改进了其预训练模型，这可能代表其 AI 能力的一次重大飞跃。
    - 一位用户注意到了 AI 开发的飞速步伐，提到了从 Codex 5.3/Opus 4.6 到 5.4 的快速连续更新，这在 coding agents 和计算机使用（computer usage）方面带来了显著改进。在这些更新后的几周内推出新的预训练模型 "Spud"，突显了 AI 进步的加速节奏，让那些在该领域密切工作的人既感到兴奋又感到紧张。
    - 讨论涉及了 AI 进步的更广泛影响，一些人对快速的开发周期表示担忧。新模型和更新的快速发布，例如从 Codex 5.3/Opus 4.6 到 5.4 的过渡，以及现在的 "Spud"，表明技术进步的曲线正在变得更加陡峭，这对 AI 领域的专业人士来说既迷人又令人不安。

- **[DeepSeek 曾经风头正劲，而 Kimi 刚刚占据了整个一周的热度](https://www.reddit.com/r/DeepSeek/comments/1s39nad/deepseek_had_a_moment_kimi_just_had_an_entire_week/)** (Activity: 182): **Moonshot AI** 的模型 **Kimi** 在 [arXiv](https://arxiv.org/abs/2303.12345) 上的一篇论文中引入了一个名为 "Attention Residuals" 的新概念，提议对现代 LLM 的架构进行重大改变。这种方法允许每一层通过学习到的、依赖输入的权重选择性地引用之前的层，以不到 `2%` 的推理开销实现了相当于 `1.25x` 更多算力的性能。这一创新吸引了 **Elon Musk** 和 **Andrej Karpathy** 等关键人物的关注，暗示了深度学习领域潜在的范式转移。此外，**Cursor** 被发现伪装成自家模型使用了 Kimi 的模型，而 **MiniMax** 被抓到抄袭 Kimi 的代码，这表明 Kimi 在 AI 领域的影响力不断增强，且可能被低估了。一些评论者认为，Kimi 虽然具有创新性，但其影响力不如 DeepSeek 的 engram，后者被认为更加复杂。另一些人则认为 Kimi 在处理上下文方面表现出色，暗示其优势可能在于特定领域而非全能。

    - BriguePalhaco 提到 Kimi 是基于 DeepSeek 的，并认为 Qwen 是其唯一的严肃竞争对手，这表明了 AI 模型竞争格局中 Kimi 和 Qwen 是重要参与者。
    - Alternative_You3585 强调 DeepSeek 的 engram 比 Kimi 的复杂得多，暗示 DeepSeek 可能拥有更先进的架构或算法，使其在技术能力上脱颖而出。


  - **[daVinci-MagiHuman：这款新的开源视频模型击败了 LTX 2.3](https://www.reddit.com/r/StableDiffusion/comments/1s2b2qt/davincimagihuman_this_new_opensource_video_model/)** (Activity: 1127): **daVinci-MagiHuman** 是由 **GAIR** 开发的拥有 `15 billion parameters` 的新型开源音视频模型。它声称在速度和性能上超越了 **LTX 2.3** 模型。该模型可在 [Hugging Face](https://huggingface.co/GAIR/daVinci-MagiHuman) 和 [GitHub](https://github.com/GAIR-NLP/daVinci-MagiHuman/) 上获取。模型完整大小约为 `65GB`，旨在在 `4070ti` GPU 等硬件上高效运行，尽管人们对其在动作极少的场景中的表现表示担忧，认为这可能无法充分展示其能力。关于用于声称模型优越性的基准测试的有效性存在争议，特别是当使用静态帧或低动态场景时。此外，人们对该模型的实际应用感兴趣，例如重制像《权力的游戏》这样复杂的视频项目。

    - MorganTheFated 批评将静态帧或动作极少的场景作为视频模型的基准测试，认为它们不能准确代表模型的性能。这突显了需要更多动态和多样的测试方案来真实评估模型能力的需求。
    - intLeon 讨论了运行 daVinci-MagiHuman 模型的平衡技术要求，指出其 65GB 的完整大小，并质疑 12GB 的 4070ti 是否能胜任。他们将其与 fp8 distilled 的 LTX2.3 进行了比较，后者在 1024x640 分辨率下处理 15 秒视频需要 5 分钟，这表明了这些模型的计算强度。
    - The elephant in the room 指出了 daVinci-MagiHuman 模型的一个显著问题：据报道其物理一致性比 LTX2.3 差，特别是在渲染手部方面，正如其 GitHub 页面上的样本所示。这表明虽然该模型在某些领域表现出色，但在保持真实的物理细节方面仍有困难。




# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复，但很快会发布全新的 AINews。感谢阅读至此，这段旅程很精彩。