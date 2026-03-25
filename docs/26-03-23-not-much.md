---
companies:
- anthropic
- meta-ai-fair
date: '2026-03-23T05:44:39.731046Z'
description: '**Anthropic** 推出了 **Claude Cowork** 和 **Claude Code**，在 **macOS 研究预览版**中实现了对鼠标、键盘和屏幕的桌面控制，将智能体（Agent）的能力从
  API 和浏览器扩展到了桌面端。


  智能体生态系统正朝着长期运行、并行且具备丰富工具的工作流演进，**Hermes Agent**、**T3 Code**、**Command Center** 和
  **Parchi** 等项目增强了多智能体编排和自主任务管理能力。然而，子智能体（包括 **GPT-5.2 Pro** 以及 **Claude** 的浏览器/计算机使用功能）在运行中表现出的脆弱性和低效等挑战，凸显了对闭环反馈系统的需求。


  **Meta AI** 的研究通过 **Hyperagents / DGM-H** 推动了自我改进型智能体的发展，实现了元级（meta-level）程序化改进，并利用
  **RLLM**（强化学习 + 以语言模型作为奖励模型）统一了强化学习后训练，从而优化了跨任务类型的奖励建模。此外，**WebArena-Infinity**
  大幅降低了浏览器环境的构建成本，加速了基准测试和环境的生成。'
id: MjAyNS0x
models:
- claude
- gpt-5.2-pro
- dgm-h
- rllm
people:
- jenny_zhang
- jase_weston
- mikhail_parakhin
- jeremyphoward
title: 今天没发生什么特别的事。
topics:
- agent-frameworks
- workflow-automation
- multi-agent-systems
- reinforcement-learning
- reward-models
- self-improving-agents
- benchmark-generation
- operational-efficiency
- closed-loop-feedback
---

**平静的一天。**

> 2026年3月20日至3月23日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现更多 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许您搜索所有过往期数。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以 [选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率设置！

---

# AI Twitter 综述

**Claude Computer Use、Agent 框架（Harnesses），以及从“代码生成（Codegen）”到全工作流自动化的转变**

- **Anthropic 将 computer use 推向了桌面端**：Claude 现在可以通过 Claude Cowork 和 Claude Code 在 **macOS 研究预览版**中控制**鼠标、键盘和屏幕**，以操作任意应用程序。这是 Agent 覆盖面的一次显著扩大，超越了 API 和浏览器沙箱。此次发布伴随着社区的强烈反响，讨论点包括许多任务不再需要笔记本电脑，以及为什么 Anthropic 可能会放弃收购更广泛的外部 Agent 堆栈，转而选择拥有完整的“在电脑上做任何事”的闭环 ([Claude 公告](https://x.com/claudeai/status/2036195789601374705), [Felix Rieseberg](https://x.com/felixrieseberg/status/2036193240509235452), [Yuchen Jin](https://x.com/Yuchenj_UW/status/2036197273496068102), [Alex Albert](https://x.com/alexalbert__/status/2036227208675729687))。
- **Agent 堆栈正向长时间运行、并行且工具丰富的流工作流汇聚**：多条推文指出，围绕编程和运维（Ops）Agent 的测试框架层（harness layer）正趋于成熟：**Hermes Agent** 的势头和生态系统整理 ([awesome-hermes-agent](https://x.com/nyk_builderz/status/2035958826973733150), [Teknium 技巧](https://x.com/Teknium/status/2036068990867603720), [开源氛围转变](https://x.com/NousResearch/status/2036122143398961659))；**T3 Code** 增加了集成的浏览器和终端功能 ([T3 Code 浏览器集成](https://x.com/LLMJunky/status/2035856842224497049), [Theo 谈开源 T3 Code](https://x.com/theo/status/2036216034949312851))；**Command Center** 及类似的编排工具，支持从一个工作区进行多 Agent 并行执行 ([Jimmy Koppel](https://x.com/jimmykoppel/status/2036077396210728974))；以及针对极长时间运行的自主任务的 **Parchi** / BYOK 工作流 ([0xSero](https://x.com/0xSero/status/2036197042045751563), [Parchi 中的 Qwen3.5-REAP](https://x.com/0xSero/status/2036204079056081043))。
- **操作层面的现实（Operational reality）现在是瓶颈，而不仅仅是模型的 IQ**：多位从业者抱怨，较新的顶级模型可能过于积极、过度 Agent 化，或者被委托给较弱的子 Agent（subagents），从而损害了真实的编程工作流；这体现在对 **GPT-5.2 Pro 子 Agent**、**Claude 浏览器/电脑使用的脆弱性**的抱怨中，以及更广泛的批评，即肤浅的并行化往往变成了“**垃圾剧场（slop theater）**”，而不是吞吐量的提升 ([Mikhail Parakhin](https://x.com/MParakhin/status/2035879791027773732), [Sarana](https://x.com/saranormous/status/2035932898218713170), [Jeremy Howard](https://x.com/jeremyphoward/status/2035966832197427509), [bentlegen](https://x.com/bentlegen/status/2035943186841915711))。一个反复出现的主题：获胜的产品可能是那些通过 Trace、Eval、事件和生产反馈实现**闭环**的产品，而不仅仅是生成代码 ([LangSmith “闭环”](https://x.com/jakebroekhuizen/status/2036137460288332077), [PlayerZero 总结](https://x.com/kimmonismus/status/2036126784887071221))。

**关于自我改进 Agent、RL 后训练（Post-Training）和基准测试（Benchmark）生成的研究**

- **Meta 附属的关于自我改进的研究已经超越了固定的元程序**：**Hyperagents / DGM-H** 扩展了 Darwin Gödel Machine 的理念，允许 Agent 不仅能改进任务行为，还能改进**生成未来改进的程序**。其核心主张是，这些元层面的改进可以跨领域迁移，包括编程 (coding)、论文评审 (paper review)、机器人奖励设计 (robotics reward design) 和奥数评分 (Olympiad grading)，解决了先前自我改进系统的一个关键局限，即自我改进循环本身是手工编写的 ([Jenny Zhang](https://x.com/jennyzhangzt/status/2036099935083618487))。
- **Meta 还展示了一个更广泛的 RL 后训练 (post-training) 统一方案**：**RLLM = RL + LM-as-RM** 通过 Policy 自身的输出进行 **on-policy** 训练语言模型奖励模型，旨在统一**易验证、难验证和不可验证**任务的后训练。值得注意的主张是，与更脆弱的定制化奖励设置相比，使用生成式 LM 奖励模型可以提高各任务类别的奖励质量 ([Jase Weston](https://x.com/jaseweston/status/2036119252214620513))。
- **基准测试 (Benchmark) 和环境生成正在快速规模化**：**WebArena-Infinity** 声称大幅降低了浏览器环境的构建成本——从数月的研究生劳动缩减到**每个环境不到 10 小时且成本低于 100 美元**——同时产生了更难、可验证的浏览器使用 (browser-use) 任务。在这些任务上，尽管强大的开源模型在旧有的 WebArena/OSWorld 上表现更好，但现在的得分**低于 50%**。这很重要，因为 Agent 的 RL 越来越需要自动生成的、高真实性的环境，而不是少数手工制作的测试平台 ([Shuyan Zhou](https://x.com/shuyanzh36/status/2036098118023049630))。
- **专题性的 RL 综述依然很受欢迎，尽管新意略显不足**：来自 The Turing Post 的一份高互动量概述编目了 **16 种 RL 变体**，涵盖了 RLHF、RLAIF、RLVR、过程奖励 (process rewards)、自我反馈 (self-feedback) 以及基于批判 (critique-based) 的方法。这作为一个分类法 (taxonomy) 很有用，但本周期内更具技术意义的推文是关于 **RL 环境和奖励模型如何被工业化**的 ([Turing Post RL list](https://x.com/TheTuringPost/status/2035857987705954760))。

**World Models、JEPA、机械可解释性 (Mechanistic Interpretability) 以及新兴训练理论**

- **JEPA/World-Model 的工作是当天技术表现最强的领域之一**：**LeWorldModel** 声称实现了稳定的**直接从像素 (pixels)** 进行端到端 JEPA 训练，无需 Teacher-Student 技巧、无需 EMA，也没有繁重的启发式规则：仅需 **15M 参数**、**1 块 GPU** 即可实现 **<1 秒的规划 (planning)**。后续总结强调了 **~48–50× 的规划加速**，且其性能与之前的 World Model 基准线相比极具竞争力。这引起了关注，因为 JEPA 类方法通常被认为很脆弱或依赖技巧；这些结果为一种更简单的训练配方提供了依据 ([Lucas Maes](https://x.com/lucasmaes_/status/2036080584569618741), [Randall Balestriero](https://x.com/randall_balestr/status/2036086865460171110), [RobotsDigest](https://x.com/robotsdigest/status/2036104283192709345))。
- **机械可解释性 (Mechanistic Interpretability) 继续从“凭感觉”向逆向工程演进**：一条总结 Anthropic 论文《大型语言模型的生物学》 (On the Biology of a Large Language Model) 的推文将当前的机械可解释性描述为：以一种在十年前听起来不可思议的精确度揭示电路 (circuits) 和内部特征。同时提醒道，追踪到的电路并不一定对应于模型对其自身推理能显式口头表达的内容 ([总结推文](https://x.com/mathemagic1an/status/2035850046735098065))。
- **训练理论和优化器缩放 (optimizer scaling) 也受到了关注**：Antonio Orvieto 的推文认为，自适应方法的优化理论解释了大部分已知的 **LLM 超参数缩放 (hyperparameter scaling)**，并能在无需暴力搜索的情况下建议迁移规则。随后的讨论强调了优化器依赖性以及对 Muon 风格设置的启示 ([Orvieto](https://x.com/orvieto_antonio/status/2036129786205008188), [giffmana 的反应](https://x.com/giffmana/status/2036156010272849950), [leloykun 的跟进](https://x.com/leloykun/status/2036178508809118067))。这是当天更有用的底层趋势之一：人们正试图用理论推导来取代经验性的缩放民间法则 (scaling folklore)。

**文档解析、检索和搜索基础设施变得更加“Agent 原生 (Agent-native)”**

- **文档解析正成为一个严肃的系统层，而不仅仅是一个辅助工具**：Google Devs 和 LlamaIndex 重点介绍了一个结合了 **LlamaParse + Gemini 3.1 Pro** 的工作流，用于从复杂的财务 PDF 中提取结构化数据，并声称在经纪报表和复杂表格上的**准确率提升了约 15%**。此外，LlamaIndex 的新项目 **LiteParse** 瞄准了更轻量级的解析路径，支持 URL 和流（stream）且不依赖 VLM，专门定位为 Agent 可以廉价且快速调用的工具（[Google Devs](https://x.com/googledevs/status/2036101456239939750), [Jerry Liu](https://x.com/jerryjliu0/status/2036155687848518097), [LiteParse](https://x.com/jerryjliu0/status/2036171132806869251)）。
- **编程 Agent 的搜索/检索基础设施得到了实质性改进**：Cursor 发布了 **Instant Grep**，宣传其可以在**数毫秒内对数百万个文件进行正则表达式搜索**，并附带了一篇关于索引/算法权衡的技术文章。对于 Agentic 编程来说，这种底层原语比微小的模型性能提升更重要；搜索延迟直接决定了 Agent 是否能足够快地在大型仓库上进行迭代以发挥作用（[Cursor announcement](https://x.com/cursor_ai/status/2036122609931165985), [blog link](https://x.com/cursor_ai/status/2036122612472881574)）。
- **延迟交互（Late interaction）/ 多向量检索正迎来高光时刻**：Weaviate 和 LightOn 的讨论认为，延迟交互系统在大规模部署中终于变得切实可行，尤其适用于代码和重推理的检索。核心观点是：Token 级的多向量表示（multi-vector representations）仍然比完整的 cross-encoders 更便宜且更具可重用性，同时能显著提升 Agent 工作负载的召回率和排序质量（[Connor Shorten podcast](https://x.com/CShorten30/status/2036080609362161900), [softwaredoug](https://x.com/softwaredoug/status/2036082251734138904), [Amélie Chatelain](https://x.com/AmelieTabatta/status/2036082256482062606)）。

**模型与产品发布：Sakana Chat, MiniMax 计划, Luma Uni-1, NVIDIA Kimodo 等**

- **Sakana AI 推出了该系列中最具代表性的具体产品**：它为日本用户推出了 **Sakana Chat**，由全新的 **Namazu alpha** 模型系列提供支持。该系列被描述为经过后训练的开源模型，旨在减少上游偏见，并更好地反映日本的语境和价值观。Sakana 将其定位为既是消费级产品，又是文化本土化后训练的展示；配套的技术博客还提到了其之前的工作：在与读卖新闻合作的信息作战分析中，使用集成（ensembles）加**新颖性搜索（novelty search）**从 **110 万条社交帖子**中提取叙事（[Sakana Chat](https://x.com/SakanaAILabs/status/2036246622141849724), [Namazu alpha](https://x.com/SakanaAILabs/status/2036247684139589688), [Hardmaru on the OSINT workflow](https://x.com/hardmaru/status/2035884310356754715)）。
- **MiniMax 继续大力推动产品化**：它推出了**固定费率的 “Token Plan”**，在一个订阅下涵盖了文本、语音、音乐、视频和图像 API，明确主打可预测的全模态计费以及与第三方工具链的兼容性。这之所以值得关注，不是因为订阅包装有多华丽，而是因为多模态 API 的消耗在操作层面上已经变得足够烦人，以至于简化定价本身就成了一种产品差异化（[MiniMax Token Plan](https://x.com/MiniMax_AI/status/2036123727373672910)）。
- **生成式媒体发布了值得关注的成果**：**Luma 的 Uni-1** 被宣传为一个“同时思考并生成像素”的模型；而 **NVIDIA 的 Kimodo** 作为一个可提示的动作/时间线模型引起了强烈关注，该模型在 **700 小时的动作捕捉（mocap）数据**上训练，支持人体和机器人骨架，并已在 Hugging Face 上架（[Luma Uni-1](https://x.com/LumaLabsAI/status/2036107826498544110), [Kimodo](https://x.com/victormustar/status/2036043907776098345)）。
- **其他值得关注的发布说明**：Hugging Face **Kernels 0.12.3** 通过 `cutlass.cute` 内核增加了对 **Flash-Attention 4** 的支持（[Sayak Paul](https://x.com/RisingSayak/status/2036038782793994541)）；**TRL v1.0.0** 声称长序列训练可节省高达 **44 倍的 VRAM**，AsyncGRPO 也即将推出（[Amine Dirhoussi](https://x.com/DirhousssiAmine/status/2036131263803781305)）；**AI2 的 MolmoPoint GUI** 瞄准了基于 VLM 的 GUI 自动化，使用 grounding tokens 而非坐标回归（coordinate regression），并在 **ScreenSpotPro 上取得了 61.1 的成绩**（[HuggingPapers](https://x.com/HuggingPapers/status/2036101402477404284)）。

**热门推文（按互动量排序，已过滤技术相关性）**

- **Claude computer use launch**: Anthropic 的桌面控制功能是该系列中最具影响力的产品发布，也是主流助手正从“回答问题”转向**直接操作软件**的最清晰信号之一 ([公告](https://x.com/claudeai/status/2036195789601374705))。
- **Cursor Instant Grep**: 备受关注，因为它解决了 Coding Agent 的一个真实系统瓶颈——仓库级搜索延迟——而不仅仅是又一个基准测试的增量 ([Cursor](https://x.com/cursor_ai/status/2036122609931165985))。
- **Luma Uni-1**: 围绕该模型的互动非常多，它将推理和图像生成折叠进同一个产品层面，尽管推文本身提供的细节较少 ([Luma Labs](https://x.com/LumaLabsAI/status/2036107826498544110))。
- **Sakana’s narrative intelligence / OSINT workflow**: 最具实质意义的 AI 应用帖子之一，结合了 LLM 集成、新颖性搜索、假设生成以及针对 **110 万条帖子**的人工验证 ([Sakana](https://x.com/SakanaAILabs/status/2035883994940887161))。
- **JEPA / LeWorldModel**: 这一紧凑型世界模型方案引起了强烈关注，它比许多人预期的更简单、更快速，因此对于普通实验室来说可能更具可复现性 ([LeWorldModel](https://x.com/lucasmaes_/status/2036080584569618741))。
- **Hyperagents / DGM-H**: 最具技术趣味性的研究帖子之一，因为它针对的是**元级自我改进 (meta-level self-improvement)**，而不仅仅是更好的任务执行 ([Hyperagents](https://x.com/jennyzhangzt/status/2036099935083618487))。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 中国 LLM 的发展与发布

  - **[中国 LLM 领域的现状](https://www.reddit.com/r/LocalLLaMA/comments/1s1gm9z/the_current_state_of_the_chinese_llms_scene/)** (活跃度: 472): **中国 LLM 格局由字节跳动 (ByteDance)、阿里巴巴 (Alibaba)、腾讯 (Tencent) 和百度 (Baidu) 等主要参与者主导，每家都有其专有模型，如字节跳动的 `dola-seed` 和阿里巴巴的 `Qwen Max`。字节跳动的 `Seed OSS 36B` 是一个稠密模型，而其 `Seedance T2V` 在视频生成领域很受欢迎。**腾讯**在 3D 网格生成方面凭借 `Hunyuan 3D` 处于领先地位，尽管目前仅提供至 2.1 版本的开放权重 (open weights)。**蚂蚁集团 (Ant Group)** 的 `Ling 2.5 1T` 引入了 `Lightning LinearAttention`，不过其性能被 `Kimi K2.5` 超越。**美团 (Meituan)** 的 `LongCat-Flash-Chat` 是一个动态 MoE 模型并开放了权重，激活参数在 `18.6B` 到 `31.3B` 之间。**Deepseek** 因其在 `MLA`、`DSA` 和 `GRPO` 等技术上的创新而受到关注。“AI 六小虎”（如**智谱**和 **Minimax**）专注于发布大型开放权重模型以获取认可，其中 **Minimax** 的 `MiniMax 2.5` 是一个 `229B-A10B` 的 MoE 模型。**上海人工智能实验室 (Shanghai AI Lab)** 的 `InterLM-S1-Pro` 虽然有政府资助，但在知乎等平台上的评价褒贬不一。** 评论者指出，与美国公司相比，中国实验室发布开放权重的速度非常快，并强调了**腾讯**在游戏开发模型上的战略投资，例如用于 3D 网格生成的 SOTA 模型 `Hunyuan 3.1` 以及用于文本转动画的 `HY-Motion`。人们普遍认为，腾讯最初通过开源模型来建立品牌知名度，待达到商业可行性后再转为闭源权重。

    - 腾讯正在大力投资游戏开发专用模型，Hunyuan 3.1 在 3D 网格生成方面达到了顶尖水平，而 HY-Motion 在文本转动画方面表现出色。最初，腾讯通过开源这些模型来建立品牌知名度，但一旦具备商业价值就会转向闭源，最新的 Hunyuan 3D 模型便体现了这一点。
    - 一份过去 7 天 OpenRouter 上按 Token 使用量排名的热门模型列表突显了中国模型的统治地位，小米的 MiMo-V2-Pro 以 1.77 万亿 Token 领跑。值得注意的是，只有三家西方实验室上榜，而“小虎”们（快速推进 AI 的小型公司）表现突出，这表明了创新动态的转变。
    - 尽管字节跳动在 AI 领域做出了重大贡献，但他们尚未发布任何开放权重模型，这一点已从 [Hugging Face](https://huggingface.co/ByteDance-Seed/models) 上相关模型的缺失中得到证实。这与频繁发布开放权重的其他中国实验室形成鲜明对比，加速了该领域的竞争。

- **[阿里巴巴确认致力于持续开源新的 Qwen 和 Wan 模型](https://www.reddit.com/r/LocalLLaMA/comments/1s0pfml/alibaba_confirms_they_are_committed_to/)** (Activity: 1269): **阿里巴巴**在南京举行的 ModelScope DevCon（魔搭社区开发者大会）上确认了他们致力于开源 Qwen 和 Wan 系列新模型的承诺。演讲中强调了阿里巴巴发布涵盖所有尺寸的全系列模型的策略，这在社区中引起了极大的期待。这一举措符合开源 AI 模型以促进创新与协作的大趋势。社区中也存在一些担忧，即由于阿里巴巴近期关键团队成员的离职，可能会对模型质量产生潜在影响。然而，人们也对可能发布的 “Qwen 3.5 Coder” 模型感到兴奋。

    - 讨论涉及由于几位才华横溢的团队成员离开阿里巴巴，对未来模型质量可能产生的影响。这引发了人们对新的开源模型是否能维持前几代迭代所设定的高标准的担忧。
    - 有关于模型开源的澄清，部分用户误解了该公告。公告中的汉字表明更多的开源模型即将推出，但未明确具体系列，导致了关于 Qwen 和 Wan 模型是否都会被包含在内的推测。
    - 一位用户表达了对 Qwen 3.5 模型的狂热，指出其即使在 0.8B 这样的小型配置中也表现出色。这突显了该模型的效率和能力，为未来的版本设定了很高的预期。

  - **[这么说 Cursor 承认了 Kimi K2.5 是最强的开源模型](https://www.reddit.com/r/LocalLLaMA/comments/1s19ik2/so_cursor_admits_that_kimi_k25_is_the_best_open/)** (Activity: 575): **该图片是 **Aman Sanger** 发布的一条社交媒体帖子，讨论了对基础模型的评估，特别强调了根据基于 perplexity（困惑度）的评估，**Kimi K2.5** 被认为是最强的开源模型。帖子提到，该模型的强大归功于持续的预训练（pre-training）和高算力的强化学习（RL），这有助于提升 **Composer-2** 模型的先进能力。他们承认在博客中没有提到 Kimi 基础模型是一个疏忽，并计划在未来的模型中解决这个问题。评论者对模型间基于 perplexity 评估的有效性表示怀疑，指出分数可能会受到词典大小等因素的影响。此外，对于 75% 的训练由一方完成的说法也存在疑问，**Workshop Labs** 报告称 **Fireworks** 的 K2 训练代码效率低下，表明其可能未针对超大规模（hyperscaled）训练进行优化。

    - Kimi K2.5 是最佳开源模型的说法因评估方法论（特别是基于 perplexity 的评估）而受到质疑，因为这种评估会受到词典大小等因素的影响。这表明此类评估在直接比较模型时可能并不可靠。
    - 针对 Fireworks 关于 Kimi K2.5 的训练声明存在怀疑。以优化训练代码著称的 Workshop Labs 报告称，Fireworks 的代码并未针对超大规模训练进行优化，仅略好于缺乏并行性的 HF Transformers 4.x 等基础实现。这引发了对 Fireworks 训练方法效率和可扩展性的质疑。
    - 讨论强调 Kimi K2.5 被视为最佳“基础模型”，是因为它拥有庞大的参数量，并采用了标准的 Attention（注意力）机制而非线性机制。这表明该模型的架构在其性能中起着重要作用，而训练后的改进可能预示着初始训练过程中的缺陷。

### 2. 本地 LLM 实现与硬件

  - **[关于运行 9× RTX 3090 进行 AI 任务的坦率看法](https://www.reddit.com/r/LocalLLaMA/comments/1s0p28x/honest_take_on_running_9_rtx_3090_for_ai/)** (热度: 675): **该帖子讨论了为 AI 任务运行 9 个 RTX 3090 GPU 的挑战和局限性，重点强调了 PCIe 通道限制、稳定性和电源管理等问题。作者指出，超过 6 个 GPU 后，由于延迟增加和带宽限制，性能（特别是 Token 生成速度）可能会下降。他们建议使用 Proxmox 来实验 LLM，并认为对于一般的 AI 使用，云服务可能更高效。作者还探索了该配置的其他用途，例如具有情感行为的 AI 系统和虚拟模拟。尽管面临挑战，RTX 3090 凭借其约 `$750` 的价格和 24GB VRAM，仍然是一个具有性价比的选择。** 评论者讨论了由于 PCIe 延迟导致的多 GPU 使用效率低下的问题，并建议使用专用的 PCIe 交换机以获得更好的性能。他们还辩论了利用本地模型达到 Claude 级别性能的可行性，指出如果优化得当，本地配置可以具有竞争力。还强调了使用经过 P2P 补丁处理的 Nvidia 驱动程序以避免 CPU 瓶颈的重要性。

    - **JockY** 讨论了使用多个 RTX 3090 GPU 的局限性，指出使用 9 个 GPU 时，PCIe 通道成为瓶颈，由于延迟增加和带宽减少，降低了 tensor parallelism（张量并行）的效果。他们建议使用专用的 PCIe 4.0 交换机来池化 GPU，通过 pipeline parallelism（流水线并行）获得更好的性能，尽管这种配置成本很高。他们建议在 EPYC 处理器上使用 PCIe 5.0，并最大限度地提高每个 GPU 的 VRAM 以获得最佳性能。
    - **kevin_1994** 分享了使用本地模型的经验，建议 4x RTX 3090 的配置可以接近 Claude 等前沿模型的性能。他们详细介绍了其硬件配置（包括 RTX 4090、RTX 3090 和 RTX 3060 的组合），并描述了如何针对特定任务使用不同的模型，例如使用 Qwen 2.5 进行自动补全，使用 Minimax 2.5 进行聊天。他们强调为每个任务选择合适模型的重要性，以实现与高端模型相当的性能。
    - **a_beautiful_rhind** 强调了使用 P2P（对等传输）驱动程序的重要性，以避免所有 PCIe 流量都经过 CPU，从而导致性能下降。这一技术见解强调了 GPU 之间高效数据传输的需求，以最大限度地发挥多 GPU 配置的优势。

  - **[真的有人后悔买了 5090 吗？](https://www.reddit.com/r/LocalLLM/comments/1s0ibbj/is_there_anyone_who_actually_regrets_getting_a/)** (热度: 388): **该 Reddit 帖子讨论了购买 NVIDIA 5090 和 4090 GPU 后可能产生的后悔心理，重点关注是现在购买还是因价格上涨而观望。发帖人正在考虑从 `3070 mobile` GPU 升级，以运行《星际公民》（*Star Citizen*）和《毁灭战士》（*Doom*）等高要求游戏，并在本地运行智能模型。一位评论者建议等待更高效的模型以及由于中国开源模型竞争带来的降价。另一位用户分享了通过 SaladCloud 以 `$0.25/hr` 租用 GPU 的积极体验，而第三位评论者最初因高昂的成本而后悔购买 Zotac 5090，但后来因其在游戏和模型测试中的性能而感到满意，尤其是考虑到价格上涨了 `40%`。** 辩论的核心在于现在购买高端 GPU 还是等待潜在的降价和效率提升。一些用户对租用 GPU 表示满意，或者尽管最初后悔但最终对购买感到欣慰。

    - philip_laureano 建议在购买 5090 之前先等待，因为由于中国开源模型的压力，预计市场将变得更具竞争力和效率。这可能导致未来出现更好的模型和更低的价格。
    - Maleficent-Ad5999 最初因为高昂的成本后悔购买 Zotac 5090 非 OC 版本，但后来发现其在测试各种 LLM 模型、使用 ComfyUI 和游戏方面的价值。自购买以来 40% 的价格涨幅缓解了任何后悔。
    - CATLLM 讨论了购买 4090 而非 5090 的战略决策，以及卖掉一个获利并投资 2x DGX Sparks 的好处。他们强调了将两个 DGX Sparks 集群化以获得最佳性能的重要性，因为由于 ConnectX7 的高昂价格，单个单元并不划算。

### 3. 创新的 LLM 模型与技术

  - **[7MB binary-weight LLM running in the browser, no FPU needed](https://www.reddit.com/r/LocalLLM/comments/1s0zoyi/7mb_binaryweight_llm_running_in_the_browser_no/)** (Activity: 248): **一位开发者创建了一个拥有 `57M parameter` 的大语言模型 (LLM)，其 `99.9%` 的权重为二进制 (`{-1, +1}`)，从而实现了一个仅 `7MB` 的紧凑模型，该模型完全在浏览器中运行，无需浮点运算单元 (FPU)。通过使用 WebAssembly (WASM) 并利用整数操作进行推理，该模型运行速度约为 `12 tokens/sec`，能够生成连贯的英文文本（特别是简单的儿童故事）。这种方法允许模型离线运行，并能放入 L1 cache，其灵感源自微软 1.5-bit 量化模型等技术。** 评论者对该模型的紧凑性和离线能力印象深刻，部分人提到了微软之前在量化模型方面的工作。用户对获取代码和评估指标表现出浓厚兴趣，表明了进一步探索及在其他项目中潜在应用的意愿。

    - 实现一个在浏览器中运行且无需 FPU 的 7MB 二进制权重 LLM 是一项显著的技术成就。它以每秒 12 tokens 的速度运行并能装入 L1 cache，突显了其效率和优化水平。这个拥有 5700 万参数的模型展示了设备端 AI (on-device AI) 的潜力，尤其是在硬件资源受限的环境中。
    - 该项目与微软的 BitNet 相关联，BitNet 以其创新的模型量化方法而闻名。微软之前的模型使用了 1.5-bit 量化方案 (-1, 0, 1) 并取得了良好的性能，这表明此处可能采用了类似技术来实现模型的紧凑尺寸和高效率。
    - 模型能够完全离线运行且无需 GPU 或 FPU，这对硬件爱好者来说尤为值得关注。这种能力预示着在计算资源受限的设备（如带有 Ethos u55 NPU 的 Grove AI Vision v2）上运行 AI 应用的广阔前景。

  - **[Qwen3.5-9B-Claude-4.6-Opus-Uncensored-v2-Q4_K_M-GGUF](https://www.reddit.com/r/LocalLLaMA/comments/1s0fn0e/qwen359bclaude46opusuncensoredv2q4_k_mgguf/)** (Activity: 483): **该帖子讨论了将 AI 模型转换为 GGUF 格式（特别是 Qwen 3.5 9B 模型）相关的技术问题和解决方案。在从 `.safetensors` 转换为 `.gguf` 的过程中，发现一些 attention 和 expert layers 在数学上存在缺陷。作者修复了多种量化格式（包括 `Q3_K_M`、`Q4_K_M` 和 `Q8_0`）的这些问题，并在 [HuggingFace](https://huggingface.co/LuffyTheFox/Qwen3.5-35B-A3B-Uncensored-Kullback-Leibler) 上分享了更新后的模型。帖子还提供了在 LM Studio 0.4.7 中获得最佳性能的详细设置，例如使用 `0.7` 的 temperature 和 `20` 的 top K 采样。合并过程涉及将 Q8 量化模型转换为 Float32 进行合并，然后使用 `llama.cpp` 中的 `llama-quantize` 等工具重新量化为 `Q4_K_M`。** 一位评论者询问如何学习合并过程，表明了对此类教育资源的需求。另一位评论者建议进行更广泛的 benchmark 测试以评估 distillation 和 merging 的有效性，强调了对这些技术进行实证验证的必要性。

    - **JustWicktor** 提供了一个使用 Claude 代码运行模型的权宜之计，因为工具（tooling）默认未开启常导致 400 错误。解决方案包括创建一个自定义 Modelfile 并使用 `ollama create` 命令生成自定义模型。Modelfile 包含了 `temperature`、`stop`、`num_ctx` 等参数，以及一个定义模型能力和行为的 SYSTEM 区块。这种方法通过在模板中包含“Tools”区块来绕过错误。
    - **ButterscotchLoud99** 质疑了 distillation/merging 在模型性能方面的有效性，并建议运行更广泛的 benchmark 来测试其影响。这意味着需要实证证据来验证这些技术的益处，因为通常在没有具体数据的情况下，人们会假设这些技术能提高模型效率或准确性。
    - **JasonJnosaJ** 对 system prompt 中引号的使用提出了疑问，询问其重要性以及是否有任何已发表的研究支持其在模型通信中的有效性。这反映了对 prompt engineering 设计选择的好奇心，即这些选择是基于实证结果还是更偏向于审美性质。

## 较少技术性的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude 和 Opus 的特性与更新

- **[Claude 现在可以操作你的电脑了](https://www.reddit.com/r/ClaudeAI/comments/1s1ujv6/claude_can_now_use_your_computer/)** (活跃度: 1001): **由 **Anthropic** 开发的 **Claude**，其通过 **Claude Cowork** 和 **Claude Code** 完成任务的功能目前正处于研究预览阶段。该功能允许 Claude 打开应用程序、浏览网页以及填写电子表格，它会优先利用 Slack 和 Calendar 等已连接的应用，在没有连接器的情况下则直接与应用交互。它支持任务自动化，例如扫描电子邮件或生成报告，目前仅在 **macOS** 上的 **Pro 和 Max** 方案中可用。用户可以更新桌面应用并与手机配对，在 [这里](https://claude.com/product/cowork#dispatch-and-computer-use) 进行尝试。** 关于允许 Claude 控制电脑任务的安全影响引起了人们的担忧，一些用户幽默地建议它可能会取代工作。其他人指出，这是 **Anthropic** 针对 **OpenAI** 等竞争对手采取的战略举措。


  - **[Claude Code 的 5 个等级（以及如何知道何时达到了每个等级的上限）](https://www.reddit.com/r/ClaudeAI/comments/1s1ipep/the_5_levels_of_claude_code_and_how_to_know_when/)** (活跃度: 853): **这篇 Reddit 帖子概述了使用 **Anthropic** 工具 **Claude Code** 的五个进阶等级。等级范围从基础的原始 Prompting 到多 Agent 的高级编排。在 **Level 1**，用户依赖简单的 Prompt，但随着项目规模增长，会遇到上下文保留方面的限制。**Level 2** 引入了 `CLAUDE.md` 文件来引导 Agent，但较长的文件会出现合规性问题。**Level 3** 涉及创建 “Skills”——即针对特定任务的 Markdown 协议文件，这提高了效率，但仍需要手动质量检查。**Level 4** 增加了用于自动验证的 “Hooks”，而 **Level 5** 则涉及为大规模项目编排多个 Agent，在一次涉及 `198 agents` 的测试中，合并冲突减少到了 `3.1%`。作者强调，达到每个级别都是因为前一个级别的局限性，跳级可能会导致问题。该系统已在 [Citadel](https://github.com/SethGammon/Citadel) 开源。** 评论者赞同这一进阶过程，并指出由于 `CLAUDE.md` 的合规性问题，**Level 2** 经常迫使用户晋级。**Level 3** 因其可重用的 “Skills” 被强调为具有变革性，而 **Level 5** 被认为维护起来可能很复杂。从 **Level 2 到 Level 3** 的过渡被确定为一个关键点，用户要么晋级，要么放弃该工具。

    - 从使用 Claude 的 Level 2 到 Level 3 的过渡至关重要，因为它涉及从基础使用转向利用可重用的 “Skills” 或模板，这显著提高了生产力。这种转变通常需要集成像 Runable 这样的工具来实现结构化输出，这有助于保持输出的可预测性。然而，超越这一点进入全面编排可能会很复杂，并可能带来重大的维护挑战。
    - Claude 使用等级的进阶并不是僵化的，但通常遵循一种模式：用户从简单的 Prompting 开始，逐渐意识到需要更具确定性的输出。这通常会导致使用结构化上下文和 MCP 服务器，尤其是在项目复杂度增加时。Claude Code 的文档可以通过提供对更高级使用模式的见解来加速这一进阶过程。
    - 关于 Claude 中非活动 Skills 成本存在一个误解。虽然人们认为非活动 Skills 的成本为 0 个 Token，但 Claude 仍需要读取 Skills 的 Frontmatter 来决定是否激活，这意味着即使 Skills 没有被主动使用，也会涉及一定的 Token 成本。

- **[关于强制 Claude 在提及日期、时间或去睡觉前检查日期时间的请愿。](https://www.reddit.com/r/ClaudeAI/comments/1s16eiz/petition_to_force_claude_to_check_datetime_before/)** (热度: 770): **这篇 Reddit 帖子强调了 Claude 在长时间会话中准确引用当前日期和时间的能力限制。用户反映，在连续使用 7 小时后，Claude 错误地引用了当前的日期和时间，这表明存在一个技术缺陷：提供日期和时间的 System Prompt 仅在会话开始时注入。这导致 Claude 被“锁定”在初始时间戳，从而导致与时间相关的引用出现偏差。用户幽默地请愿让 Claude 在进行此类引用前检查当前时间，同时也强调了该模型在法律研究方面的出色能力，例如识别程序缺陷和伪造的引用。** 一位评论者解释说，由于包含日期/时间的 System Prompt 仅在会话开始时设置，导致 Claude 被“困”在初始时间。另一位建议提交“改进请求”而非请愿，以解决这一技术限制。

    - truongnguyenptit 解释了一项技术限制，即提供当前日期和时间的 Claude System Prompt 仅在会话开始时注入。这意味着如果会话持续数小时，Claude 仍会“卡”在初始时间戳，导致时间引用过时。出现此问题是因为 System Prompt 在长会话期间不会动态更新。
    - larowin 提出了一个关于用户体验差异的有趣观点，质疑为什么尽管频繁使用，有些用户会遇到 Claude 的时间相关问题，而其他用户则不会。这表明会话管理或用户交互模式的潜在差异可能会影响此问题的发生。
    - SuddenFrosting951 建议通过程序化方法解决此问题，推荐用户通过支持单提交“改进请求”，而不是发起请愿。这暗示了用户向开发者传达技术问题或功能需求的一种结构化方法。

  - **[Claude (Opus 4.6) 搞清楚了如何修补我童年时的游戏，使其能在现代 Windows 上运行](https://www.reddit.com/r/ClaudeAI/comments/1s0z27t/claude_opus_46_figured_out_how_to_patch_my/)** (热度: 819): **一位用户分享了一种无需使用 DOSBox 或虚拟机即可在现代 Windows 系统上运行 1996 年游戏 Tonka Construction 的方法。该解决方案涉及修补 `WING32.dll` 以将调用转换为现代 OS 调用，类似于 DXVK 将 DirectX 调用转换为 Vulkan。该补丁已在 [GitHub](http://github.com/Quackster/TonkaReconstruction) 上发布。** 评论者对无需虚拟机即可原生运行该游戏的能力印象深刻，并强调了在其他遗留软件中进行类似应用的潜力。

    - MongooseSenior4418 强调了在无需虚拟机 (VM) 的情况下在现代 Windows 上原生运行游戏的技术成就。这表明在兼容性解决方案方面取得了重大进展，可能涉及直接的二进制修补或 API 转换层，以弥合旧软件与新操作系统之间的差距。
    - ricecanister 指出了该解决方案更广泛的影响，指出如果补丁涉及通用库，它可能适用于该游戏之外的其他应用程序。这表明在更新遗留软件以在现代系统上运行方面具有广泛应用潜力，可能通过共享依赖项或通用框架实现。
    - dread_beard 强调了这种补丁方案的广泛应用场景，认为在现代系统上原生运行遗留软件的能力可以为软件保存、复古游戏和教育目的开辟多种可能性。

### 2. Gemini 模型问题与对比

  - **[Gemini 质量严重倒退](https://www.reddit.com/r/GeminiAI/comments/1s0dagg/serious_regression_in_gemini_quality/)** (热度: 642): **一位用户报告称，Google 旗下的服务 Gemini Ultra 在最近一次更新后出现了严重的质量倒退。该用户强调了诸如对话 Context 丢失、无法保留对先前指令的记忆以及对话历史记录被删除等问题，导致在编程相关的讨论串中反复出现错误。用户对该服务目前的表现表示不满，认为其不如早期的模型，并考虑如果情况没有改善将取消多个订阅。该用户还批评支持服务毫无效果。** 评论者们一致认同原帖观点，指出 **Gemini 3.0** 已经变得无法使用，频繁丢失 Context。一些人认为这是一种模式，即模型在发布新版本之前会被“Nerfed”（削弱）。此外，还有人批评 **ChatGPT** 提供事实性错误的答案，表明用户对 AI 模型普遍存在不满。

    - 用户报告 Gemini 模型的性能显著下降，特别是注意到 Context 保留和整体智能方面的问题。一位用户提到 Gemini 3.0 在几个月前还很有效，但此后变得“无法使用”，这表明模型在发布新版本之前被故意“Nerfed”已成为一种模式。
    - 用户普遍认为 Google 的 Ultra 订阅层级物无所值，因为付费用户也经历了与低层级用户相同的性能退化。这导致了用户的挫败感，他们认为支付更多费用并不能保证更好的服务或模型变更的透明度。
    - 提到的一个技术问题是 Context Window（上下文窗口）大小的缩减。用户观察到它从预期的 200 万个 Tokens 掉到了低至 4000 或 8000 个 Tokens。这种缩减被视为 Google 的一种 Throttling（限流）形式，影响了模型在长对话中维持 Context 的能力。


### 3. Qwen 模型进展与应用

  - **[阿里巴巴在 MWC 巴塞罗那展会上发布 Qwen Glasses，加速 AI 硬件布局](https://www.reddit.com/r/Qwen_AI/comments/1s0lki0/alibaba_unveils_qwen_glasses_at_mwc_barcelona/)** (热度: 134): **Alibaba** 在巴塞罗那世界移动通信大会（MWC）上展示了其新款智能眼镜 **Qwen Glasses**，标志着其 AI 硬件战略迈出了重要一步。该眼镜分为 S1 和 G1 两个系列，集成了 Alibaba 的 Qwen AI 模型，提供实时翻译、高清拍摄和视觉识别等功能。G1 系列补贴后的价格约为 275 美元，旨在降低 AI 可穿戴设备的入门门槛。该眼镜将与 Qwen App 配合使用，通过语音指令实现订餐或预订酒店等免提任务，预计到 2026 年全面推广。一条值得注意的评论推测，Alibaba 可能会在 Qwen3.5 之后转向闭源模式，反映了对未来 AI 开发开放性的担忧。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。