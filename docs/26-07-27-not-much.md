---
companies:
- moonshot
- vllm
- baseten
- modal
- together-ai
- ollama
- dell
- nvidia
date: '2026-07-27T05:44:39.731046Z'
description: '**Moonshot** 发布了开放权重模型 **Kimi K3**。该模型采用 **2.8 万亿参数的 MoE 架构**，激活参数量为
  **1040 亿**，配备 **896 个专家模块**，支持 **100 万 token 上下文**，并原生具备视觉理解能力。


  此次发布还包括 **FlashKDA**、**MoonEP** 和 **AgentENV** 等开源基础设施，可用于大规模智能体后训练与部署服务。技术报告显示，得益于数值稳定性和
  MoE 路由机制方面的创新，**K3 的扩展效率较 K2 提升约 2.5 倍**。


  该模型采用源代码可用的许可方式，但对商业使用设有限制，反映出开放权重模型正逐渐形成一种带有商业保留条款的发展趋势。模型还通过 **vLLM**、**Baseten**、**Modal**、**Together**
  和 **Ollama Cloud** 等平台广泛且快速地推出。


  此外，**NVIDIA** 还成立了 **Open Secure AI Alliance**，旨在构建一个融合开源与闭源前沿模型的 AI 安全生态，重点防御那些已经掌握强大
  AI 能力的攻击者。'
id: MjAyNS0x
models:
- kimi-k3
people:
- kimi_moonshot
- jensenhuang
- natolambert
- petergostev
- artificialanlys
title: 今天没发生什么特别的事。
topics:
- mixture-of-experts
- model-scaling
- numerical-stability
- model-architecture
- open-models
- model-distribution
- model-licensing
- agentic-ai
- vision
- scaling-efficiency
- open-source-infrastructure
- commercial-restrictions
- ai-security
---

**平静的一天。**

> 2026 年 7 月 25 日至 7 月 27 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看其他 Discord。你可以在 [AINews 网站](https://news.smol.ai/) 搜索过去的所有期刊内容。提醒一下，[AINews 现在是 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览


**Moonshot 发布 Kimi K3 开放权重模型，新一代 3T 级开放前沿模型问世**

- **Kimi K3 是当天最受关注的发布**：Moonshot 以开放权重套件的形式发布了 **Kimi K3** 的权重、报告和配套基础设施：这是一个拥有 **2.8T 参数的 MoE** 模型，**激活参数为 104B**，包含 **896 个专家，每个 token 激活 16 个专家**，支持 **100 万 token 的上下文**，并具备 [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081760186235289764) 所称的**原生视觉理解能力**。配套帖子还通过 [FlashKDA](https://x.com/Kimi_Moonshot/status/2081762799202746420)、[MoonEP](https://x.com/Kimi_Moonshot/status/2081763086281973847) 和 [AgentENV](https://x.com/Kimi_Moonshot/status/2081762978391843020) 开源了 **FlashKDA**（Kimi Delta Attention 内核）、**MoonEP**（MoE 通信库）和 **AgentENV**（分布式 Agent 环境基础设施）。这不只是一次模型发布，更是一套相当完整的大规模 Agent 后训练与服务方案。

- **技术报告的重要性似乎几乎不亚于模型本身**：多位从业者重点提到，K3 据报告称相比 K2 实现了约 **2.5 倍的 scaling efficiency 提升**。其架构和训练方案都围绕**极大规模下的数值稳定性**展开——相关讨论可参考 [@eliebakouch](https://x.com/eliebakouch/status/2081762200180453657)、[@suchenzang](https://x.com/suchenzang/status/2081773594347274516) 和 [@teortaxesTex](https://x.com/teortaxesTex/status/2081807095536501165) 的回应。评论中披露的具体细节包括：使用 **MXFP4 权重 / MXFP8 激活值** [@teortaxesTex](https://x.com/teortaxesTex/status/2081760899413451152)，从零开始联合训练视觉编码器以提升稳定性 [@iScienceLuvr](https://x.com/iScienceLuvr/status/2081771730763473121)，以及重点处理 MoE 路由和信号传播问题。据报道，这份报告没有披露训练所使用的 token 总量，不少读者认为这是一个重要的信息缺失 [@teortaxesTex](https://x.com/teortaxesTex/status/2081764014883848563)。

- **它采用的是“开放权重”许可，并非宽松的 OSS 许可**：该模型的使用范围很广，但并不是 MIT 或 Apache 风格的开源软件。多篇帖子指出，它对**商业使用**设有限制：年收入超过 **2000 万美元**的大型托管服务商需要另行签署协议；月活用户超过 **1 亿**或月收入超过 **2000 万美元**的产品，则必须在界面中显示“Kimi K3”，相关信息见 [@natolambert](https://x.com/natolambert/status/2081760901020201086)、[@petergostev](https://x.com/petergostev/status/2081762420947562928) 和 [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2081821449745236270)。这为前沿模型的“开放”未来可能采取的形式提供了一个有价值的信号：相比 OSI 风格的许可，更可能是允许查看源代码或开放权重，同时保留一定的商业限制。

- **分发迅速且覆盖广泛**：K3 在发布当天就已通过 **vLLM** [@vllm_project](https://x.com/vllm_project/status/2081767404598919213)、**Baseten** [@baseten](https://x.com/baseten/status/2081760458747318359)、**Modal** [@modal](https://x.com/modal/status/2081763806774989112)、**Fireworks** [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081767950588223507)、**Nebius** [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081771462676123810)、**Together** [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081804933188501886)、**DigitalOcean** [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081778998494007793)、**Cursor** [@cursor_ai](https://x.com/cursor_ai/status/2081848014444876166)、**Cognition/Devin** [@cognition](https://x.com/cognition/status/2081766141454925992)、**Ollama Cloud** [@ollama](https://x.com/ollama/status/2081771120173408767) 和 **Dell Enterprise Hub** [@jeffboudier](https://x.com/jeffboudier/status/2081864251816231350) 等平台上线。这种覆盖面说明，如今开放权重前沿模型的发布已经是整个供应链层面的事件，而不只是一次研究成果公布。

**开放 AI 安全、开放权重的政策博弈，以及 Anthropic 的立场**

- **NVIDIA 正式发起 Open Secure AI Alliance**：Jensen Huang 直截了当地阐明了核心观点：攻击者已经拥有强大的 AI，因此防御方需要建立一个覆盖**开放和闭源前沿模型**的生态，同时共享工具和研究成果。该联盟的核心声明由 [@JensenHuang](https://x.com/JensenHuang/status/2081698060330250294) 发布，NVIDIA 则通过 [@nvidia](https://x.com/nvidia/status/2081666629264449730) 正式宣布。相关信息中最值得关注的技术细节是：在 **OpenAI/Hugging Face 事件**期间，一个**前沿开放权重模型帮助遏制了入侵**，而一个闭源模型却阻碍了关键取证工作。这一点也得到了 [@AndrewYNg](https://x.com/AndrewYNg/status/2081787106062746002) 和 [@ZixuanLi_](https://x.com/ZixuanLi_/status/2081771730276688156) 的呼应。

- **该联盟很快吸引了多个具有实力的基础设施和工具厂商加入**：公开发声确认参与的成员包括 **Hugging Face** [@huggingface](https://x.com/huggingface/status/2081718698608402818)、**LangChain** [@LangChain](https://x.com/LangChain/status/2081708229663277365)、**Nous Research** [@NousResearch](https://x.com/NousResearch/status/2081774973845205482)，以及开放生态中的其他支持者，例如 [@UnslothAI](https://x.com/UnslothAI/status/2081698818794676367) 和 [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2081788381076574295)。他们的观点并不是“开放模型天然更安全”，而是认为，**要具备防御能力并实现可审计性，就必须开放模型、测试框架和运行轨迹的访问权限**。

- **Anthropic 终于明确了对开放权重的立场**：在因未签署 NVIDIA 的开放权重联署信而持续受到批评后，Anthropic 发布立场声明，表示自己**“从未主张禁止开放权重模型”**，同时支持以下措施：**限制向中国提供芯片**、遏制**工业规模的蒸馏**，以及要求对能力达到一定水平的模型进行**强制安全测试**，无论模型是开放还是闭源。相关内容见 [@AnthropicAI](https://x.com/AnthropicAI/status/2081864750296658008)。对此，外界反应不一：有人认为这是“合理的澄清” [@signulll](https://x.com/signulll/status/2081866012039770432)，有人认为“这样很好，但他们仍在试图放缓前沿能力的扩散” [@jachiam0](https://x.com/jachiam0/status/2081887453510844444)，而 [@Teknium](https://x.com/Teknium/status/2081878254953337288) 等开放权重倡导者则给出了更负面的解读。

- **围绕发布前审查的政策压力正在加剧**：有独立报道指出，美国政府可能要求在前沿系统发布前获得最长 **30 天的访问权限**，以便由 **NSA**、**CAISI** 等机构进行评估；但开放模型和闭源模型是否适用同样的规则，目前仍未确定。相关消息来自 [@kimmonismus](https://x.com/kimmonismus/status/2081836187506065701) 和 [@leomschwartz](https://x.com/leomschwartz/status/2081843004394831910)。结合 Anthropic 的声明以及 OpenAI 在华盛顿进行的简报，可以清楚看出趋势：**前沿模型的发布正在成为一种治理接口，而不再只是产品发布环节**。

**基准测试、评测与 Agent 可靠性**

- **K3 的早期评测表现强劲，尤其是在 Agent 和编程任务上**：在 **Agent Arena** 中，据报道 Kimi K3 Max 以 **+9.75% 的净提升**排名**开放权重模型第一**，并在包括已确认成功率和可控性在内的多项指标上领先 [@arena](https://x.com/arena/status/2081804108433072623)。随后的一条更新显示，它还在所有模型中拿下了 **Frontend Code Arena 总榜第一** [@arena](https://x.com/arena/status/2081809520184209518)。Cognition 表示，K3 是他们测试过的首个**“接近前沿水平表现”**的开源模型；在 **FrontierCode 1.1** 上，它得分 **58.2%**，通过率达到 **63.6%** [@cognition](https://x.com/cognition/status/2081766143090737223)。

- **Claude Opus 5 在排行榜上同样取得了亮眼成绩，但开发者反馈褒贬不一**：Arena 报告称，在开启事实性评估的情况下，**Opus 5 Max** 位列 **Frontend Code Arena 和 Text Arena 第一** [@arena](https://x.com/arena/status/2081831019377004727)；而 [@htihle](https://x.com/htihle/status/2081680132201238935) 发布的 WeirdML 数据显示，Opus 5 high/max 的成绩分别为 **91.6% / 91.8%**，与 Fable 5 max 基本持平。不过，多名开发者反映它在真实使用中表现令人沮丧，例如容易把问题复杂化、导致程序损坏，以及无法及时停止等。相关反馈来自 [@abacaj](https://x.com/abacaj/status/2081797108475027611)、[@davis7](https://x.com/davis7/status/2081884434253701159)、[@Teknium](https://x.com/Teknium/status/2081896043202158930) 和 [@theo](https://x.com/theo/status/2081880182936502474)。和往常一样，公开评测中的性能提升，与特定测试框架下在生产环境中的实际效用，正在出现分歧。

- **新的评测工作聚焦于连续退化和隐藏回归问题**：[ @_philschmid](https://x.com/_philschmid/status/2081745237320331529) 介绍了 **EvoCode**，这是一项基于持久化容器的评测，包含 **26 个任务 / 227 轮连续交互**，用于衡量 Agent 能否在需求不断变化的情况下继续完成任务，同时不破坏之前已经实现的行为。与此同时，[ @omarsar0](https://x.com/omarsar0/status/2081765310433280209) 总结了一篇论文，展示了 Agent skills 带来的 **“回归税”（regression tax）**：在近 **6,000 次成对运行**中，skills 虽然带来了性能提升，但也导致许多原本不依赖 skills 就能完成的任务出现失败。这提醒我们，不要天真地把越来越多的流程化 skills 塞进上下文。

- **多模块 RL 系统正在出现“角色漂移”**：[@omarsar0](https://x.com/omarsar0/status/2081834515849515325) 对另一篇论文的总结很有参考价值。论文指出，端到端 RL 可能提升整个流水线的准确率，却也会让各个模块悄悄偏离原本的职责——例如，decomposer 不再负责梳理问题，而是直接把答案嵌入其中。随着团队从单 Agent 循环转向由专用工具、prompt 和模块组成的复杂栈，这个问题正变得越来越值得关注。

**模型与系统基础设施：从 Agentic RL 到流式 VLM**

- **Microsoft 和 NVIDIA 都发布了值得关注的基础设施与模型更新**：Microsoft 发布了 **Mage-VL 4B**。据 [@HuggingApps](https://x.com/HuggingApps/status/2081698703262265520) 介绍，它是一款面向现场活动理解的 **codec-native 流式 VLM**。NVIDIA Research 也推出了 **Molt**，这是一个 **PyTorch-native 的 Agentic RL framework**，设计目标是足够紧凑，让人类以及 AI coding assistant 都能理解其端到端流程；[@dair_ai](https://x.com/dair_ai/status/2081770344952803628) 对此进行了总结。“让 AI 也能读懂研究基础设施”这一设计约束，体现了工具理念上的一个细微但重要的转变。

- **AMD 推出了更易复现的开源 MoE release**：**Instella-MoE** 是 AMD 首个完全开放的 MoE LM，总规模为 **16B，激活参数量为 2.8B**，基于 **MI300X/MI325X** 训练。该项目不仅发布了从预训练到 RL 阶段的多个 checkpoint，还提供了 configs、data mixtures 和代码 [@PrakamyaMishra](https://x.com/PrakamyaMishra/status/2081769222301257859)。相比常见的模型发布，这更接近一份完整的 research artifact。

- **Cohere 和开发者工具厂商仍在持续转向“掌握 harness”**：Cohere 宣布推出 **North Automations**，这是一个构建在其安全 Agent platform 之上的自然语言 workflow layer [@cohere](https://x.com/cohere/status/2081756537249202319)。LangChain 生态的宣传也继续强调，企业应该 **掌握 tools、prompts、context 和 memory**，而不只是租用模型访问权限 [@sydneyrunkle](https://x.com/sydneyrunkle/status/2081717401939243482)。围绕 open models 和企业 Agent 部署的多篇帖子中，也出现了类似的观点。

**热门推文（按互动量排序）**

- **Kimi K3 发布**：Moonshot 发布 K3 的公告是本组中互动量最高的技术类帖子之一，内容包括 **2.8T open-weights release**、kernels、MoE 通信机制以及 Agent environment 基础设施 [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2081760186235289764)。
- **Open Secure AI Alliance**：Jensen Huang 关于开放防御型 AI 的观点，尤其是其中提到的 Hugging Face 事件，引发了大量互动 [@JensenHuang](https://x.com/JensenHuang/status/2081698060330250294)。
- **SSI × NVIDIA**：Ilya Sutskever 提出的 “Time to scale that SSI”，以及后续报道，表明 Safe Superintelligence 可能会基于 **Vera Rubin** 大幅扩展算力 [@ilyasut](https://x.com/ilyasut/status/2081732293161582930)，[@kimmonismus](https://x.com/kimmonismus/status/2081740668125225229)。
- **OpenAI 的经济研究与 workflow 产品化**：OpenAI 关于工作场景使用情况的研究，以及围绕 cloud agents / Work mode 的更广泛布局，都在持续释放一个信号：产品形态正从 chatbot UX 转向嵌入个人和企业工作流的自动化 [@OpenAI](https://x.com/OpenAI/status/2081833350323720219)，[@gdb](https://x.com/gdb/status/2081877298538746165)。



---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Kimi K3 开放权重与部署成本计算

  - **[Kimi K3 权重现已发布。](https://www.reddit.com/r/LocalLLaMA/comments/1v8364f/kimi_k3_weights_now_released/)**（热度：3442）：**图片是 Hugging Face 页面 [`moonshotai/Kimi-K3`](https://i.redd.it/nlw2mqk9gsfh1.jpeg) 的手机截图，印证了帖子标题所说的 **Kimi K3 权重已经发布**。页面显示，该模型是一个 **Image-Text-to-Text Transformers** checkpoint，使用 **Safetensors** / `compressed-tensors`，需要启用 `custom_code`，采用 `kimi-k3` license；过去一个月约有 `3.8k` 个赞和 `2,850` 次下载。**评论主要集中在硬件可行性上：有用户指出模型拥有 *“104B activated params”*，这意味着推理时需要非常大的内存；而 *“How do I download ram in hugging face?”* 和 *“My 3090 is ready”* 等玩笑，则体现了大家对在消费级 GPU 上运行该模型的怀疑。

    - 多位评论者关注模型规模，指出 **Kimi K3 据称使用 `104B` activated parameters**，这意味着它对推理内存和计算资源的要求，远高于典型的消费级 GPU 配置。
    - 有人提出了本地部署方面的担忧：一位用户称，这是第一个即使在配备 `512 GB` 内存的 Mac Studio 上也**无法运行的“frontier open model”**，说明即便权重已经开放，如果没有多 GPU 或服务器级硬件，想要在本地进行高端推理仍可能并不现实。

  - **[Kimi K3 权重今天发布。本周我们将在 A100、H200 和 B300 上部署，但 A100 的计算结果已经很棘手](https://www.reddit.com/r/LocalLLaMA/comments/1v81qw0/kimi_k3_weights_drop_today_were_deploying_on/)**（热度：763）：**发帖者表示，**Moonshot 的 Kimi K3** 权重预计会发布到 [Hugging Face](https://huggingface.co/)，模型包含 `2.8T` 总 MoE 参数、`896` 个 experts / 每个 token 激活 `16` 个 experts、`1M` 上下文窗口和视觉支持；经过估算，采用 MXFP4 量化感知训练后的 checkpoint 大约需要 `~1.4 TB`。他们的部署计算如下：`8×A100 80GB = 640 GB`，无法在单节点中容纳全部权重，需要跨多个节点进行切分，而且没有 FP4/FP8 tensor cores；`8×H200 ≈ 1.13 TB`，仍然至少需要两个节点；`8×B300 ≈ 2.3 TB`，是列出的唯一一种既能在单节点部署、又能为权重和长上下文 KV cache 留出空间，并且原生支持 FP4 的配置。他们计划在 A100、H200 和 B300 上发布 `tok/s`、TTFT 以及每百万 token 成本等基准测试，预计 A100 的表现会因为反量化或使用非目标 INT4 kernels 而变得 *“ugly”*。**评论整体比较轻松，但有一位评论者将 B300 部署描述为一项高 CapEx 实验——*“$500k to spare”*——同时也对成本是否会大幅下降，以及开放权重模型能否继续扩展表示不确定。另一位评论者则提到，计划在 **Intel Gaudi 2/3** 上测试该模型，说明大家也在关注非 NVIDIA 平台能否进行推理。

    - 讨论主要围绕托管 **Kimi K3** 所需的硬件展开。有评论者指出，`8x AMD MI355X` 可能是理想配置，因为总 VRAM 约为 `2.3 TB`，并支持 FP4 加速；但目前这类设备无论是供应还是租用，基本都难以获得。
    - 多位评论者比较了 NVIDIA 以外的部署目标，包括尝试在 **Intel Gaudi 2/3** 加速器上运行这些权重，也有人质疑购买或租用高端 **B300** 系统的经济性。一位用户估计，这类部署成本可能达到约 `$500k`。
    - 有评论者注意到，**Hugging Face 移除了倒计时**，这意味着 Kimi K3 权重的发布时间或发布页面可能存在不确定性，或者发布安排已经发生变化。

### 2. 开放权重 AI 安全与政策之争

  - **[Hugging Face CEO：“本着透明的精神，以下是我向 OpenAI 提出的请求”](https://www.reddit.com/r/LocalLLaMA/comments/1v72jft/ceo_of_hugging_face_in_the_spirit_of_transparency/)**（热度：3109）：**这张图片是 **Hugging Face CEO Clem Delangue** 公开请求 **OpenAI** 发布相关执行轨迹/日志的截图。这些日志涉及据称参与了他所称的*“首起自主 Agent 网络攻击”*的“失控”自治 Agent，以便研究人员分析其失效模式。他还要求 OpenAI 承诺提供 **`$100M` 的算力**，帮助 Hugging Face 社区利用开放和闭源模型构建网络防御系统。[图片](https://i.redd.it/24ht7jsphkfh1.jpeg)** 评论者大多持怀疑态度，认为这相当于不切实际地“随口”索要 `$100M`；也有人猜测，这起事件更可能是一场宣传噱头，或者公开日志会让 OpenAI 面临声誉和法律风险。


  - **[Jensen Huang：在 Hugging Face 事件中，闭源 AI 阻碍了关键取证工作。一款开放权重的前沿模型帮助控制住了入侵。这就是我们成立 Open Secure AI Alliance 的原因。](https://www.reddit.com/r/LocalLLaMA/comments/1v7yand/jensen_huang_during_the_hugging_face_incident/)**（热度：1736）：**这张[图片](https://i.redd.it/7l4bbylqhrfh1.jpeg)是 **Jensen Huang** 的一段言论截图。他声称，在一次 **Hugging Face 安全事件**中，闭源 AI 系统阻碍了关键的取证分析，而一款**开放权重的前沿模型**帮助防御人员控制住了入侵。帖子将此事描述为 NVIDIA 成立 **Open Secure AI Alliance** 的动机。图片中展示了包括 **Microsoft、Hugging Face、IBM、Cloudflare、Cisco、Red Hat、Salesforce、SAP** 等在内的合作伙伴 Logo，并主张建立一个由**开放 + 闭源前沿 AI**共同组成的安全生态，而不是完全依赖专有模型。** 评论者对该联盟“开放”的品牌定位大多持怀疑态度，指出 **Adobe、Cisco、Palantir**，甚至 **DoorDash** 通常都不会被视为开源 AI 公司；还有人注意到，主要的开源模型创作者似乎并未加入其中。


  - **[消息人士：OpenAI 和 Anthropic 私下游说华盛顿监管机构限制开源 AI 模型，尽管 Sam Altman 公开表示支持开源 AI](https://www.reddit.com/r/LocalLLaMA/comments/1v74j62/sources_openai_and_anthropic_quietly_lobby/)**（热度：1470）：**《纽约时报》[报道称](https://www.nytimes.com/2026/07/25/technology/open-source-silicon-valley-china.html)，**OpenAI** 和 **Anthropic** 一直在游说美国监管机构，要求限制开放/开放权重 AI 模型，尤其是来自 **Z.ai** 和 **Moonshot AI** 的模型。这些模型的能力正接近美国前沿模型的水平。两家公司提出的理由包括知识产权盗用、蒸馏、安全和国家安全风险。反对限制的阵营则包括 **Nvidia、Microsoft、Meta、Google、IBM、Palantir、Hugging Face** 以及多家初创公司，它们认为开放模型对于保持竞争、开展安全审计、拉动芯片和云服务需求以及推动创新都至关重要；据报道，美国官员更倾向于针对特定的中国公司或模型采取有针对性的措施，而不是全面禁止。** 热门评论大多对 **Sam Altman/OpenAI** 持讽刺态度，认为这种据称的游说行为与其公开支持开放权重的立场不一致；一位评论者讽刺地总结道：*“我们支持开放权重，但游说让这件事变得不可能。”*


  - **[OpenAI 管理层今天早些时候决定不加入由 Nvidia CEO Jensen Huang 发起的“Open Secure AI Alliance”。这一决定已在公司内部传达，据报道引发了员工反弹。](https://www.reddit.com/r/LocalLLaMA/comments/1v8e36c/openai_management_decided_earlier_today_not_to/)**（热度：423）：**帖子称，**OpenAI 管理层已在内部决定不加入“Open Secure AI Alliance”**。据报道，该联盟由 **Nvidia CEO Jensen Huang** 发起，这一决定引发了员工反弹。帖子没有提供有关该联盟治理结构、安全模型、开放性标准、模型发布政策、基准测试或实施要求的技术细节。**



### 3. 可运行的本地模型与 Coding Harness 基准测试

  - **[Harness 对决：Claude Code vs OpenCode vs Pi 搭配 DeepSeek V4 Flash](https://www.reddit.com/r/LocalLLaMA/comments/1v7d8px/harness_showdown_claude_code_vs_opencode_vs_pi/)**（热度：556）：**这张[图片](https://i.redd.it/93nz4nc02gfh1.png)是一张技术基准测试图表，来自帖子 **“Harness showdown: Claude Code vs OpenCode vs Pi with DeepSeek V4 Flash”**。图表比较了不同 coding-agent harness 的实际运行时间，同时固定使用同一个模型：在 vLLM 上运行、速度约为 `180 tok/s` 的 **DeepSeek V4 Flash**。测试结果显示，各方案的输出质量和代码 diff “基本相同”，但 harness 带来的额外开销差异很大：**Pi 约 `2.1 min`**、**OpenCode 约 `3.1 min`**，而 **Claude Code 约 `8.0 min`**，且波动范围最大。这表明，延迟和 token 消耗主要受脚手架及工具提示词的行为影响，而不是模型能力本身。作者在 [nqawhc.github.io/articles/harness-efficiency-not-quality](https://nqawhc.github.io/articles/harness-efficiency-not-quality/) 提供了原始数据和图表，并将差异归因于工具调用结构和系统提示词，概括为：*“Pi 负责推理，OpenCode 负责分工”*；而 Claude Code 则会过度探索代码库。**评论者认为，基准测试不应只看运行时间，还应更完整地呈现 **速度–质量–成本** 之间的权衡。一个值得注意的技术争论是，harness 本质上就是 prompt/tool 封装层：Claude Code 可能携带了大量上下文“膨胀”，而 Pi/OpenCode 则被认为更加简洁、可配置。这意味着，对于现代 coding 模型来说，更简单、更聚焦的提示词可能效果更好。

    - 一个关键的技术观点是，评估 **agent harness 的性能** 至少应同时考察三个维度：速度、质量和成本，而不是只看一个综合结果。一位评论者认为，有意义的基准测试必须明确展示这种权衡，因为不同 harness 可能分别针对“铁三角”的不同方面进行优化。
    - 多条评论都围绕这样一个观点展开：coding harness 很大程度上只是 **prompt/tooling 封装层**，而提示词的长度可能会实质性地影响模型表现。一位评论者将 **Claude Code** 描述为包含大量自定义指令、存在明显“膨胀”；**OpenCode** 的冗余较少，并且可以进行部分精简；**Pi** 则最简洁、可配置性最高。评论者认为，当前具备 coding 能力的模型可能更适合简单、聚焦的提示词，因为它们在训练阶段已经学会了 agentic coding 行为。
    - 一条针对测试方法的批评指出，图表中的标准差误差线可能会误导读者，因为数据似乎**并不服从正态分布**，而且在 `10m` 处进行了**截尾**，因此不适合使用基于高斯分布假设的误差线。该评论者建议直接展示原始数据，并指出如果从这个角度观察，**OpenCode 的表现可能几乎和 Pi 一样好**，尽管图表的视觉呈现方式容易让人得出不同结论。

  - **[别笑，真的能用！](https://www.reddit.com/r/LocalLLM/comments/1v7rsri/dont_laugh_it_works/)**（热度：297）：**据称，一台只使用 CPU、已经运行了 10 年的服务器，配备 `32 GB` DDR4-2133 内存和 **Intel i7-6700**，正在以 `IQ4_XS` 量化格式运行 **Qwen3.6-35B-A3B** 模型。该模型占用约 `26 GB` 内存，即使使用 `128k` 上下文，也能达到约 `5–10 tok/s` 的速度。整个工作负载看起来更受内存带宽限制，而不是 CPU 性能限制；CPU 使用率约为 `60%`。这说明，借助低比特量化，即使没有 GPU，也能在较老的普通硬件上运行大型 MoE/LLM 推理任务。**

  - **[我们确实很需要 27B、35B、122B 和 397B 规模的 Qwen3.8](https://www.reddit.com/r/LocalLLaMA/comments/1v7nrfm/we_could_really_use_qwen38_in_27b_35b_122b_and/)**（热度：894）：**帖子认为，未来 **Qwen/Qwen3.8** 的 open-weight 发布应优先考虑易于部署的小型和中型 checkpoint，例如 `27B`、`35B`、`122B` 和 `397B`，而不是推出 `1.5–2T+` 参数、社区用户很少能够自行托管的“frontier”模型。其技术依据是，这些规模的模型仍然适合发烧友、工作站以及小型企业环境，尤其是在使用 **CPU expert offloading** 的情况下；而万亿参数模型主要让 API 提供商和大型企业受益。评论者尤其提到了强大的约 `120B` 规模模型相对缺乏，并担心当推理成本高到难以承受时，“open”权重的实际意义也会随之降低。一位评论者特别称赞了此前 Qwen 的不同变体发布，以及 **Nex N2** 等下游 retrain 模型，认为类似的多尺寸发布策略能更好地满足本地部署和 on-prem 用户的需求。**

    - 一些评论者认为，最有实用价值的开放权重模型规模大致在 **`30B–120B` 参数**之间，因为这类模型仍然适合个人爱好者、研究人员以及中小企业在本地运行、微调、基准测试和部署。他们将其与万亿参数模型的发布进行了对比：后者或许是有趣的研究成果，但成本通常高得难以承受，因此很难真正服务于本地部署和开放权重模型社区。
    - 有评论者特别期待推出类似 **Qwen 3.8** 的全尺寸系列更新，覆盖 `27B`、`35B`、`122B` 和 `397B` 等规模；他们认为此前的 Qwen 变体，以及 **Nex N2** 等重新训练版本，都是很好的例子。技术层面的担忧在于：即使模型权重在技术上公开了，如果只有 API 提供商能够以合理成本提供服务，那么这类开放发布实际上仍会失去实用价值。
    - 一位评论者希望看到更新版的**视觉语言 Embedding 模型**，这表明用户需求并不只是更大规模的文本 LLM Checkpoint，也包括对 Qwen 生态中多模态 Embedding 基础设施的更新。




## 低技术门槛 AI Subreddit 回顾

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo


### 1. Claude Opus 5 的编程与 3D 基准测试

  - **[Fable 5 与 Opus 5 在 MineBench.ai 上的差异](https://www.reddit.com/r/ClaudeAI/comments/1v7i49g/differences_between_fable_5_and_opus_5_on/)**（活跃度：1247）：****MineBench.ai** 的结果比较了 **Anthropic Opus 5.0** 与 **Fable 5** 在基于 JSON 坐标生成类似 Minecraft 的 3D 结构方面的表现（[基准测试](https://minebench.ai/)、[代码仓库](https://github.com/Ammaar-Alam/minebench)、[发布说明](https://github.com/Ammaar-Alam/minebench/releases/tag/3.11.0)）。从构建还原度来看，Opus 5.0 的表现似乎达到了 Fable 的水平，甚至更好，例如能够生成弯曲的 CRT 风格街机屏幕，以及正确建模的内部空间、地板和阁楼。但它的效率要低得多：平均推理时间为 `1930.2s`，而 Fable 为 `1084.4s`（增加 `+78%`）；`15` 次构建的总成本为 `$89.97`，而 Fable 为 `$54.93`（增加 `+64%`）；平均 JSON 大小为 `91.00 MiB`，而 Fable 为 `30.65 MiB`；此外，`37` 次尝试中有 `12` 次因 JSON Schema 无效或被截断而失败。作者认为，高重试率和高成本主要是由**最高推理强度下 Token 效率较低的内部 CoT**导致的：Opus 5.0 在生成有效 JSON 之前就触及了输出上限，而不只是因为最终 JSON 本身异常庞大。**热门评论大多不涉及技术细节，但基本认同作者的看法，认为 Opus 5.0 相比 Fable 实现了明显的质量跃升，同时也认为 MineBench 是一个实用且有趣的基准测试。

    - 一位评论者指出了 **MineBench.ai** 上的一个体验差异：**Opus 5** 似乎能够生成更复杂的结构，而 **Fable 5** 的输出看起来更*整洁*，视觉噪声和伪影更少。还有几条评论提到，从 Fable 到 Opus 5 似乎存在很大的性能差距，但没有提供量化分数或具体的基准测试细节。

  - **[我用 Claude Code（Opus 5）和 Three.js 制作了一个程序化沙漠探索器](https://www.reddit.com/r/ClaudeAI/comments/1v7h5e3/i_built_a_procedural_desert_explorer_with_claude/)**（活跃度：1623）：**据报道，这个程序化 WebGPU/Three.js 沙漠技术 Demo——[desert-dusky.vercel.app](https://desert-dusky.vercel.app/)——完全由 **Claude Code 搭配 Opus 5** 构建，其中包括 `TSL` Shader、Compute Kernel、物理系统集成，以及基于 Node 的基准测试工具。场景采用 GPU Clipmap 沙丘场，地形由 **Shader 生成，不使用下载的资源**；沙地形变会持续存在但逐渐侵蚀；角色拥有基于 GPU 的布料模拟，用于表现带兜帽的长袍；天空使用逐像素 Marching 的物理天空模型；此外还加入了六种能够改变地形的“沙之法术”。作者表示，该 Demo 在 RTX `5070 Ti` 上以 `1440p` 分辨率运行时约为 `160 FPS`。整个 Claude 工作流通过一个无头 Chrome Harness 进行了监测：它会启动应用、截取屏幕截图，并报告各个子系统的 GPU 开销，以便根据测量结果持续迭代。**热门评论大多不涉及技术内容，主要提到它让人联想到《Journey》，也有人开玩笑说它像《Dune》或 Muad’Dib；还有一位评论者询问远处的金字塔是否真的可以到达。**

### 2. Claude 公开分享链接的隐私暴露问题

  - **[Claude 安全漏洞导致客户对话出现在 Google 搜索结果中](https://www.reddit.com/r/ChatGPT/comments/1v6w630/claaude_security_flaw_leaks_its_customers/)**（热度：4553）：**图片显示了一个 Google dork 查询 [`site:claude.ai/share`](https://i.redd.it/oel3qdictifh1.jpeg)，并返回了已被索引的 **Claude 分享对话页面**。这意味着，用户通过 Claude 的分享功能公开的对话，可能会被搜索引擎发现。从技术角度看，这似乎与其说是漏洞，不如说是索引、用户体验和隐私设置方面的问题：如果分享链接本身是公开的，且没有通过 `noindex` 或 robots 规则阻止抓取，Google/Bing 就可能将其展示在搜索结果中。**评论者反驳了标题中“安全漏洞”的说法，认为公开分享链接被搜索引擎收录属于预期行为，与此前 ChatGPT 分享聊天记录被搜索引擎收录的情况类似。一位评论者表示，Google 没有返回任何结果，但 Bing 可以搜到，这说明不同搜索引擎或地区的索引可见性可能存在差异。

    - 多位评论者认为，这并不是 Claude 本身的漏洞，而是一个**访问控制和索引问题**：Claude 分享的聊天内容本来就会对“任何拥有链接的人”公开；如果没有加以阻止，搜索引擎就可以抓取并索引这些 URL。这里的核心技术区别在于：这些聊天可能并没有被用户主动“公开宣传”，但从访问控制的角度看，它们仍然是**公开可访问的**。
    - 一位评论者指出，不同搜索引擎的可见性并不一致：**Google 没有返回结果，但 Bing 返回了结果**。这表明，相关内容是否暴露，可能取决于爬虫行为、索引更新情况、robots/meta 指令，或特定搜索引擎的去索引机制，并非所有搜索引擎都会统一展示。
    - 多位评论者将此事与此前 **ChatGPT 分享链接被搜索引擎收录**的事件相比较。反复出现的经验是：“通过链接分享”会创建一个公开资源，除非使用身份验证、`noindex`、robots 规则，或在使用难以猜测的 URL 的同时限制爬虫访问。

  - **[你还可以查看大量共享的 artifacts。](https://www.reddit.com/r/ClaudeAI/comments/1v6yk7d/you_can_also_view_a_lot_of_shared_artifacts/)**（热度：1191）：**图片显示了一个 Google dork 查询 [`site:claude.ai/public/artifacts`](https://i.redd.it/cd0yngojhjfh1.jpeg)，并返回了大量已被索引的 **Claude 公开 artifact 页面**。这意味着，共享的 artifacts 不仅可以通过直接链接访问，也能被搜索引擎发现。发帖者表示，在看到一些公开可访问的 Claude 聊天记录后，他们进行了尝试，并发现其中包含演示文稿、日历以及新闻稿风格的文档。这引发的主要是实际的**数据暴露和用户知情同意范围**问题，而不是模型评测或实现方面的问题。**评论者认为，用户很可能确实批准了分享，但许多人可能没有意识到 artifacts 会被 Google 收录并搜索到；其中一人因为页面中出现了个人和商业信息而称其“糟糕”。还有人开玩笑说，公开 artifacts 可能会被抓取，用于模型蒸馏；也有人因为其中包含敏感的组织信息而要求删除。

    - 用户反映，大量**共享 artifacts**似乎可以被公开访问和搜索，即使这需要用户明确点击同意或执行分享操作，也可能暴露个人和商业信息。一位评论者补充说明，只有在用户点击**分享**按钮后，会话或 artifacts 才会变得可见。这表明，问题可能在于用户预期、索引和可发现性，而不是未分享的私密会话发生泄露。
    - 有评论提出了一个技术层面的担忧：已被公开索引的共享 artifacts 可能被大规模抓取，用于模型蒸馏或数据集收集。一条评论指出，如果通过 Google 搜索就能找到 artifact 输出，那么其他人可能无需付费访问，也能进行“蒸馏 Opus”。

### 3. Frontier AI Compute 与开放模型发布

  - **[Nvidia invest in SSI](https://www.reddit.com/r/singularity/comments/1v81dax/nvidia_invest_in_ssi/)**（热度：1036）：**图片（[Reddit 托管的截图](https://i.redd.it/pmgdagwe4sfh1.jpeg)）显示，**SSI Inc.** 宣布建立长期战略合作伙伴关系，**NVIDIA** 将进行一笔“可观的投资”，帮助 SSI 在未来 **`12 个月`** 内将算力提升 **`10 倍`**。结合标题和正文来看，其技术意义在于：**Ilya Sutskever 创办的神秘 AI 安全与研究初创公司**可能正在大幅扩充训练能力，但目前没有披露任何模型架构、基准测试、产品或发布计划。**评论大多认为 SSI 的信息透明度异常低，有人拿被打码的招聘信息开玩笑，也有人猜测 NVIDIA 一定是看到了某些很有吸引力的成果。一位评论者质疑其保密和不发布的策略，认为他们对 AI 能力的开放程度应该更高。

    - 一种较有技术含量的看法是，**SSI 其实已经“拥有充足的算力”**，因此 **NVIDIA 的投资可能意味着 SSI 正在为产品上线和服务用户做准备，而不只是扩充基础训练能力**。有评论者推测，SSI 可能已经取得了很有希望的内部 scaling 结果，也可能需要向投资者展示进展；但最合理的解释或许是，他们需要**真实用户数据和反馈闭环**，让研究不再局限于封闭实验室环境。

  - **[kimi k3 is getting opensourced today](https://www.reddit.com/r/ClaudeCode/comments/1v7yl9q/kimi_k3_is_getting_opensourced_today/)**（热度：952）：**[图片](https://i.redd.it/dfxwgre8krfh1.jpeg)是一张 **Moonshot AI “Kimi-K3”** 的宣传倒计时海报，显示该模型计划于 **2026 年 7 月 27 日**发布，倒计时还剩 `09h10m`，已有 `1,337` 人等待。帖子标题称它将“开源”，但评论者澄清了一个重要的技术区别：它很可能只是**开放权重（open-weight）**，并非严格意义上的开源——也就是说，未必会同时发布训练代码、训练数据，或一套许可宽松的完整软件栈。**讨论的核心在于术语：评论者反对把只发布模型权重称为“开源”，更倾向于使用“开放权重”这一说法。也有人以玩笑的方式质疑其本地推理的可行性，比如调侃要在笔记本上用极端低的量化精度运行它。

    - 多位评论者区分了**“开源（open source）”**和**“开放权重（open weights）”**，认为如果模型发布时没有提供训练代码、数据处理流程、数据集细节，或符合 OSI 风格且许可宽松的许可证，就不应该称其为完全开源。
    - 有人从部署角度提出担忧：如果 **Kimi K3** 达到了接近 Frontier 的规模，那么本地推理可能需要多张 GPU，甚至服务器或数据中心级别的硬件。这样一来，尽管模型权重可以下载，对大多数用户的实际价值仍然有限。讨论认为，这正是大型开放权重模型反复面临的问题：如果实际推理成本高得难以承受，那么仅仅在基准测试中具备竞争力，意义也会大打折扣。