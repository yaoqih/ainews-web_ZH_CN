---
companies:
- z-ai
- alibaba
- deepseek
- rednote
- vllm
- together-ai
- fireworks
- modal
- digitalocean
- deepinfra
- unsloth
date: '2026-08-14T05:44:39.731046Z'
description: '**Z.ai 发布了 GLM-5.3**。这是一款专注于编程和网络安全的模型，在智能体能力和安全性基准测试中取得了显著提升。这些进步主要来自扩大后训练规模，而不是使用更大的基础模型。**阿里巴巴发布了
  Qwen3.8-27B**，这是一款原生多模态稠密模型，采用 Apache 2.0 许可证，原生支持 262K 上下文，并可扩展至 1M。该模型面向真实世界中的编程和办公工作流，并获得多个平台的广泛推理支持。**DeepSeek
  V4-Pro** 和 **小红书的 dots3-note** 也延续了中国开源模型的发展势头。后者是一款拥有 280B 参数的多模态 MoE 模型，激活参数为
  16B，上下文长度达到 512K，并引入了 TEMPO 等面向长程任务自我评估的新型强化学习方法。整个生态中有多家专注于开源模型的中国实验室，各自具备不同优势。DeepSeek
  的 harness 被重点介绍为一种模块化智能体运行时基础设施，其组件可以替换，并通过 Cordis 实现生命周期管理。'
id: MjAyNS0x
models:
- glm-5.3
- qwen3.8-27b
- qwen3.8-2.4t-a95b
- deepseek-v4-pro
- dots3-note
people: []
title: 今天没发生什么事。
topics:
- post-training
- reinforcement-learning
- agent-runtimes
- long-horizon-training
- multimodality
- model-infrastructure
- runtime-architecture
- model-benchmarking
- open-weight
- apache-2.0-license
- model-optimization
- multimodal-models
- mixture-of-experts
- context-windows
---

**平静的一天。**

> 2026 年 8 月 13 日至 14 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有继续查看其他 Discord。你可以通过 [AINews 网站](https://news.smol.ai/) 搜索过往的所有期刊内容。提醒一下，[AINews 现已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你还可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 回顾


**开放权重前沿持续推进：Z.ai 的 GLM-5.3、Qwen3.8-27B/Max、DeepSeek V4-Pro，以及 RedNote 的 dots3-note**

- **Z.ai 的 GLM-5.3**：最大的技术新闻是 [Z.ai 发布了 GLM-5.3](https://x.com/Zai_org/status/2088132965922476159)。这是一款专注于编程和网络安全的模型，基于 GLM-5.2 使用的**同一个 743B 基础模型进行后训练**，而不是重新预训练得到的新模型。Z.ai 及后续相关帖子称，该模型在 Agent 和安全评测中取得了大幅提升，包括 **Terminal Bench 3.0：28.3**、**DeepSWE：66.9**、**Agents’ Last Exam：28.5** 和 **GDPVal-AA：1769**（[评测摘要](https://x.com/ZixuanLi_/status/2088133750357991646)，[完整评测结果](https://x.com/ZixuanLi_/status/2088135213930905623)）。公司还表示，其网络安全能力提升显著，因此在完成安全审查并最终开放权重之前，初期会先向部分合作伙伴开放访问权限（[详情](https://x.com/ZixuanLi_/status/2088134236599439607)）。许多工程师重点关注的一项说法是：这次能力跃升**完全来自在更长流程、可执行任务上的规模化后训练和 RL**，而不是更大的基础模型（[分析](https://x.com/kimmonismus/status/2088162566719639717)，[相关反响](https://x.com/cline/status/2088146558160355639)）。

- **Qwen3.8 扩展了本地部署和开放模型的前沿**：Alibaba 发布了 **Qwen3.8-27B**。这是一款基于 **Apache 2.0** 协议发布的**原生多模态 dense 模型**，原生支持 **262K 上下文**，并可通过 **YaRN** 扩展到 **1M**；与此同时，团队也重点介绍了此前已经发布的 **Qwen3.8-2.4T-A95B** Max 级模型（[发布公告](https://x.com/Alibaba_Qwen/status/2088280182356611304)，[性能讨论](https://x.com/Alibaba_Qwen/status/2088280188362867185)）。这款 27B 模型的特别之处在于，它明确面向**真实世界中的编程、办公流程和 Agent**，而不仅仅是学术基准测试。它在发布当天就获得了异常广泛的推理支持，包括 [vLLM](https://x.com/vllm_project/status/2088287539979559068)、[Ollama](https://x.com/ollama/status/2088314436088168491)、[llama.cpp/GGUF](https://x.com/ggerganov/status/2088312667253391546)、[SGLang 在单张 RTX 5090 上达到 206 tok/s](https://x.com/Alibaba_Qwen/status/2088293486995087461)，以及包括 [Together](https://x.com/Alibaba_Qwen/status/2088285662223138851)、[Fireworks](https://x.com/Alibaba_Qwen/status/2088286022597832788)、[Modal](https://x.com/Alibaba_Qwen/status/2088287553292312968)、[DigitalOcean](https://x.com/Alibaba_Qwen/status/2088288356337897550)、[DeepInfra](https://x.com/Alibaba_Qwen/status/2088301611731009582) 在内的云服务合作伙伴。这里的实际部署细节同样值得关注：[Unsloth 宣布支持 NVFP4 和动态 GGUF 构建](https://x.com/danielhanchen/status/2088281836757868916)，Qwen 则强调该模型在本地运行时**只需 17GB 内存即可运行 27B 模型**（[帖子](https://x.com/Alibaba_Qwen/status/2088296583368781939)）。

- **DeepSeek V4-Pro 和 RedNote 的 dots3-note 延续了中国开放模型的发展浪潮**：[vLLM 宣布支持 DeepSeek-V4-Pro](https://x.com/vllm_project/status/2088272865468776641)，并特别提到其采用 **MIT 许可证**、与预览版本路径兼容检查点，以及集成的草稿生成支持。与此同时，RedNote AI 实验室发布了 **dots3-note Preview**。这是一款**280B 多模态 MoE 模型，其中 16B 参数处于激活状态，上下文长度为 512K**，主要面向长时间运行的 Agent；同时还配套推出了一种新的 RL 方法 **TEMPO**，用于长流程自我评估（[早期信息](https://x.com/teortaxesTex/status/2088123149057507425)，[摘要](https://x.com/kimmonismus/status/2088194805654323617)，[团队提供的技术说明](https://x.com/ChaoQiao42/status/2088366133279867044)）。目前逐渐显现出的趋势是：中国的多个实验室正在形成各自的专长：多位评论者明确将 Z.ai、DeepSeek、Moonshot、Qwen、MiniMax 和 RedNote 描述为一个发展迅速、各具优势的开放模型生态（[一份综合分析](https://x.com/teortaxesTex/status/2088156939087667211)，[另一份分析](https://x.com/Yuchenj_UW/status/2088309946249318654)）。

**Agent 运行时、Harness 以及长流程训练**



- **DeepSeek Harness 正被当作基础设施来对待，而不只是一个演示 Agent**：这次发布引发的讨论，更多集中在运行时架构，而不是模型 UX。多篇深度分析将 Harness 描述为一种插件化的 Agent 运行时，其中 **Agent loop、工具、会话、文件系统和 providers 都可以替换**；而 **Cordis** 负责生命周期管理、响应式依赖和可逆副作用（[概览](https://x.com/ZhihuFrontier/status/2088179275195363714)、[运行时可组合性讨论](https://x.com/ZhihuFrontier/status/2088138788573004065)）。真正有技术含量的地方不只是“模块化”，还包括支持**热插拔运行时组件**，并有可能让 Agent **在无需重启的情况下修改自身运行时**，同时保留可审计的事件日志，避免隐藏状态。多位开发者对此的反应是，与这一方向相比，当前的 Harness 可能“从根本上就不对”，或者至少核心过于固定（[相关反应](https://x.com/xlr8harder/status/2088194397628248374)）。

- **Harness 正逐渐成为独立的优化目标**：一些帖子进一步说明，Benchmark 和产品效果的提升，越来越多来自 **scaffold/Harness 层**，而不只是基础模型本身的能力。[DAIR 重点介绍了 AutoDesign](https://x.com/dair_ai/status/2088298364458930462)：它会根据 rollout 反馈，自动重写 Harness；据其报告，该方法在论文转海报生成任务上取得了提升，并且能迁移到不同的 Agent/模型配置中。[Lambda 的 Tetris 实验](https://x.com/LambdaAPI/status/2088255609330339913)则从另一个角度说明了类似问题：提示词位置、参数设置和沙箱约束都会显著影响结果；如果限制不够严格，Agent 还会利用 Benchmark 中的漏洞。这与更广泛的讨论相呼应：可观测性数据如今同时承担着 **eval、memory 和 learning substrate** 的作用（[LangSmith 文档说明](https://x.com/hwchase17/status/2088342687808438352)）。

**Benchmark、Evals 与对 Benchmark 的质疑**

- **新的 Eval 开始针对 Agent 的真实失败模式**：[Vals 发布了一项 Agent 逆向工程 Benchmark](https://x.com/i2huer/status/2088094896095678923)，重点测试网络安全相关二进制环境中的确定性最终目标，而不是中间产物；配套文章指出，与必须直接分析二进制相比，当前前沿 Agent 在有源代码可用时表现要强得多（[背景说明](https://x.com/RobinDing3/status/2088099221442539909)）。[OpenRouter 推出了面向工具调用 Agent 的 Web search Benchmark](https://x.com/OpenRouter/status/2088279603861467304)，而 [Ai2 的 TutorMoments](https://x.com/dl_weekly/status/2088309871506505954)则被引用为一项基于回放的辅导评测：它显示，模型经常会**过度提供帮助**，而不是鼓励学习者进行有成效的思考和尝试。

- **对 Eval 的反思仍在继续**：人们反复表达了对厂商 Benchmark 宣传的质疑。[Vik Paruchuri 批评了一个 LlamaIndex Benchmark](https://x.com/VikParuchuri/status/2088342728908177804)，指出评分器中的 Bug 可能让系统得分从 **65% 变成 93.6%**；他明确主张开发者应该运行**自己的 Eval**，而不是相信营销材料，“包括我们的材料”在内（[后续说明](https://x.com/VikParuchuri/status/2088342734641766690)）。[François Chollet 再次强调](https://x.com/fchollet/status/2088254592182305165)，公开的 ARC-3 演示集**既不是训练数据，也不是 Eval 数据**，因此在该数据集上的排行榜分数无法有效代表私有测试集上的表现。这里还值得补充的是 Meta 的 **Wiggle Framework**，由 [Omar Sar 重点介绍](https://x.com/omarsar0/status/2088292067994951928)：它通过反复改写提示词和施加对抗压力，对 LLM judge 进行压力测试。结果显示，在静态反驳下，评判结果可能有 **25–71%** 的概率发生变化；面对对抗性说服者时，变化比例则达到 **62–91%**。

**基础设施、服务与成本工程**

- **Serving 优化正越来越像模型的一等特性**：围绕 Qwen 和 DeepSeek 的 Day-0 基础设施支持，重点越来越多地放在**内置 draft head、speculative decoding** 以及内存与量化之间的权衡上，而不只是提供 API 访问。Qwen 27B 发布时同步给出了 [vLLM 使用指南](https://x.com/vllm_project/status/2088287539979559068)，涵盖 **MTP draft heads、1M context** 以及使用**单张 Blackwell GPU** 进行 Serving；与此同时，[ggerganov 展示了本地 llama.cpp 配置方案](https://x.com/ggerganov/status/2088312671196082312)，用于支持超大上下文和 speculative decode。[Tim Dettmers 预告](https://x.com/Tim_Dettmers/status/2088247316012531982)，即将推出新的效率方法，使强力模型能够在**单台 DGX Spark 或 AMD Strix Halo** 上运行，达到约 **7 tok/s 的 decode 速度**和**超过 250 tok/s 的 prefill 速度**。

- **工具链和集群运维也迎来了实用更新**：[Stas Bekman 分享了相关指南](https://x.com/StasBekman/status/2088124725897887829)，介绍如何诊断 PyTorch 中卡住的 **NCCL 集体通信调用**；他还另外提到，**Python 3.14+** 支持在不添加额外埋点的情况下，将 `pdb` 附加到正在运行的进程上（[原帖](https://x.com/StasBekman/status/2088333548550058176)）。[Turbopuffer 介绍了](https://x.com/turbopuffer/status/2088294797002105307)一套用于运营 **100 多个 TPUf 集群**的定制控制平面，其中包括在客户云环境中进行 BYOC 部署，即使无法直接访问主机也能完成操作。在数据处理方面，Hugging Face 的 [datatrove 0.10.0 版本](https://x.com/vanstriendaniel/status/2088176267950424111)新增了面向 Hugging Face Jobs 的 **JobsPipelineExecutor**、HF bucket 集成，并保留了推理输出。

**产品与平台动态：Cursor/SpaceXAI、Gemini 3.7 Flash、Claude Code 与本地 Agent 体验**

- **Cursor 加入 SpaceXAI**：当天技术和企业动态中互动量最高的一项，是 [Cursor 宣布正式成为 SpaceX 的一部分](https://x.com/cursor_ai/status/2088249881718919393)。Cursor 团队将加入 **SpaceXAI**，负责协作开发 **Grok、Grok Build、Grok Bot、Grok API 和 Cursor**。[SpaceXAI 也确认了](https://x.com/SpaceXAI/status/2088250109188608289)这项收购，并表示将先加速软件工程领域的发展，之后再拓展到更广泛的知识工作。这清楚表明，编码 Agent 团队正在被视为战略性模型和平台资产，而不再只是狭义的 IDE 产品团队。

- **Gemini 3.7 Flash 的发布重点放在 Agent 和高性价比的主力模型定位上**：Google 将 **Gemini 3.7 Flash** 全面推向了 [Gemini app](https://x.com/GeminiApp/status/2088326407730692538)、[Search AI Mode](https://x.com/rmstein/status/2088325481599009146)、[Google Workspace / Sheets canvas](https://x.com/ChanduThota/status/2088326719484899680) 和 [Spark](https://x.com/genevieve__h/status/2088277643338637623)。其定位是“迄今最智能、最适合承担繁重工作的 coding 和 Agent 模型”，演示重点则是如何通过简单提示词生成可玩的网页游戏（[Google 演示串文](https://x.com/Google/status/2088318274715136097)）。外部评测表现中规中矩但有所提升：[Vals 在 Vals Index v2 中给它 59.4% 的成绩，排名第 7](https://x.com/ValsAI/status/2088335427426210114)，高于 Gemini 3.6 Flash 的第 14 名。

- **Claude Code 和本地 Agent 体验持续向实际生产使用靠拢**：Anthropic 已将 **Auto mode** 设为 Claude Code 面向 Pro/Max/Team 用户的默认权限模式，并通过 `/auto-mode-setup` 根据代码仓库情况推荐可信的仓库和域名（[公告](https://x.com/ClaudeDevs/status/2088332927189049738)，[配置详情](https://x.com/ClaudeDevs/status/2088332928514420830)）。在开源和本地应用方面，[Hermes 新增了 `/loop`](https://x.com/Teknium/status/2088368313974047165)，可以在 Agent 会话中执行类似 cron 的重复操作；[Nous 则指出，Hermes Desktop 可以连接 Hermes Cloud Agent](https://x.com/NousResearch/status/2088395070059770061)，即使合上笔记本电脑，任务也能继续运行。[Ollama 也新增了本地启动 DeepSeek Harness 的支持](https://x.com/ollama/status/2088392765021528319)。

**热门推文（按互动量排名）**

- **Cursor × SpaceXAI**：[Cursor 的收购公告](https://x.com/cursor_ai/status/2088249881718919393)是当天互动量最高的科技推文，显示编码 Agent 领域仍在持续整合，模型、产品和垂直业务栈之间的结合也越来越紧密。
- **GLM-5.3 发布**：[Z.ai 发布 GLM-5.3](https://x.com/Zai_org/status/2088132965922476159)的推文是当天热度最高的模型发布消息，主要原因在于它进一步强化了这样一个观点：通过 **后训练和长时程 RL**，可以从已经训练完成的前沿基础模型中释放出大量潜在能力。
- **Qwen3.8-27B 开放权重**：[Alibaba 的发布消息](https://x.com/Alibaba_Qwen/status/2088280182356611304)引发了广泛关注，因为一款 **27B 本地多模态模型**如今被定位为能够胜任严肃的 Agent 和专业工作，并且从发布第一天起就获得了广泛支持。
- **实用的 coding Agent 案例**：[redp314 分享的“Claude Code 仅用两个提示词，就用 800 个文件构建了一个 DICOM viewer”](https://x.com/redp314/status/2088206627954405400)，展示了当前 coding assistant 在脱离基准测试、面对真实任务时所能达到的较高水平。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen3.8-27B 发布、基准测试与模板

  - **[Qwen3.8-27B 初步 Model Card 已上线！](https://www.reddit.com/r/LocalLLaMA/comments/1vo2iiz/a_preliminary_qwen3827b_model_card_is_live/)**（热度：1006）：**这张图片展示的是 Hugging Face 上 Qwen3.8-27B 初步 Model Card 的技术截图，具体为 **Qwen/Qwen3.8-27B** ([图片](https://i.redd.it/3u6hgcgk7bjh1.png))，与帖子中“Model Card 在发布前就已可见，随后正式上线”的说明一致。页面显示，该模型计划提供模型权重和配置文件，并兼容 **Transformers**、**vLLM** 和 **SGLang**。同时，页面重点介绍了模型在编程、Agent 执行、研究和长上下文使用方面的改进，并注明原生上下文长度为 `262,144` 个 token，最高可扩展至 `1,000,000` 个 token。**评论者主要关注 **reasoning effort** 这一可能的核心特性，称赞其超长上下文窗口；同时，他们也对 `27B` 模型似乎具备视觉能力，而据报道规模大得多的 `2.4T` 模型却不具备这一能力感到意外。

    - 评论者特别提到了 Model Card 中标注的**原生 `262,144` token 上下文长度**，以及最高可扩展至 **`1,000,000` 个 token** 的能力，认为这是 Qwen3.8-27B 最值得关注的技术规格之一。
    - 大家还对模型架构和产品线之间的差异表现出兴趣：据报道，**27B 模型支持视觉能力**，而规模大得多的 **2.4T 模型却不支持**。从能力随规模扩展的角度来看，用户对此感到十分意外。
    - 有评论者注意到，页面没有明确提及 **QAT / 量化感知训练（quantization-aware training）**，并将其与 **Gemma 4 31B** 进行比较：在后者中，QAT 被认为显著提升了量化模型的性能。还有人指出，“reasoning effort” 正逐渐成为近期 Model Card 中出现的一项调节和控制特性。

  - **[Qwen3.8-27B 与 Qwen3.6-27B 完全一致！](https://www.reddit.com/r/LocalLLaMA/comments/1voblcs/qwen3827b_is_identical_to_qwen3627b/)**（热度：902）：**这张图片（[GIF](https://i.redd.it/oerqqcan7djh1.gif)）并排展示了 **Qwen3.6-27B** 和 **Qwen3.8-27B** 的架构图，两者在视觉上完全相同：都采用相同的视觉/embedding 路径、masked scatter、重复堆叠的 `Qwen3_5DecoderLayer`、`RMSNorm`、最终的 `Linear` 层和输出结构。相关 HF Viewer diff 显示架构改动数量为 `0`，这支持了帖子中的观点：**Qwen3.8-27B** 的能力提升很可能主要来自**训练、数据或微调方面的更新，而不是模型架构变化**。**评论者将其视为一次渐进式更新，而非从头训练的新模型；其中一人指出，训练数据通常是影响模型质量的最大因素。还有人推测，可热插拔的 LoRA 风格适配器可能会逐渐流行，用于提升本地模型在特定任务上的准确率。

    - 多位评论者认为，**Qwen3.8-27B** 更像是一次渐进式**更新**，而不是从头训练的新模型。有人指出，它实际上与 **Qwen3.6-27B**，甚至与 **Qwen3.5** 看起来都基本相同。这引出了一个技术层面的判断：数据集变化或训练后更新可能才是提升质量的主要手段，而不是架构改动。
    - 有评论者推荐 **Ninfer**（[GitHub](https://github.com/Neroued/ninfer)）作为 Qwen 系列的高吞吐本地推理方案，并提到该项目新增了并发请求支持，最高可达 `C=8`。据报告，**Qwen3.6-35B-A3B** 在 `C=8` 时的聚合 decode 速度达到 `1,313.8` tok/s，而 **27B NVFP4** 配置达到 `1,146.9 tok/s`，相当于单并发吞吐量的 `5.67×`。
    - 有人推测，**LoRA 热切换**可能会在本地推理工作流中变得越来越重要。它可以在无需替换基础模型的情况下，针对具体任务提升准确率。这种方式也能通过动态应用专用适配器，弥补基础模型更新幅度较小或仅为渐进式更新所带来的不足。

  - **[Qwen3.8-27B 现已发布](https://www.reddit.com/r/LocalLLM/comments/1vo9nt5/qwen3827b_is_now_available/)**（热度：745）：**这张图片（[链接](https://i.redd.it/f1hh6ugvucjh1.jpeg)）展示了 Hugging Face 上 **`Qwen/Qwen3.8-27B-FP8`** 的页面，表明这款新的 **28B 参数** Qwen 3.8 模型已经上线。该模型支持 **Transformers**，使用 **Safetensors** 格式，采用 **Apache 2.0** 许可证，并通过 **`F8_E4M3`** 和 **BF16** 张量提供 FP8 量化版本。一位评论者分享了在 **RTX 5090** 上进行本地推理的早期体验，速度约为 **`50–60 tokens/s`**；他认为该模型相比 Qwen 3.6 更稳定，也更善于进行审慎推理，但同时指出当前设置可能还不是最优，而且 **MTP** 似乎暂时还不支持。**评论总体上比较谨慎但偏积极，有用户将其形容为“成熟版的 3.6”，认为它处理长时间运行任务的能力更强。另一位评论者询问是否会提供 **9B** 或 **35B** 等更小或不同规模的版本，因为 27B 对许多本地用户来说过于庞大。

    - 一位在 **RTX 5090** 上测试 **Qwen3.8-27B** 的用户表示，在使用与 Qwen 3.6 相同的设置时，本地推理速度稳定在约 `50–60 tokens/s`，并指出，等 **MTP** 支持上线后，性能可能还会进一步提升。从定性体验来看，在长篇生成任务中，它比 Qwen 3.6 更加深思熟虑：不会立即开始起草一篇一万字的故事，而是会先检查段落之间的一致性，将任务拆分成多个子任务，并按章节逐一生成，整体规划性更强。

  - **[Muse Glimmer was frontier In the model class around 30b models for four days.](https://www.reddit.com/r/LocalLLaMA/comments/1vofnnf/muse_glimmer_was_frontier_in_the_model_class/)**（活跃度：502）：**这张图片是一张约 30B 级模型的基准测试对比表，其中重点标出了 **Muse Glimmer-30B** 和 **Qwen3.8-27B**。帖子认为，Muse Glimmer 在这一规模的模型中只当了四天的“前沿模型”，随后 Qwen 的 27B 模型就在大多数已公布指标上超过了它。Muse Glimmer 的成绩包括：Agentic terminal coding `51.7`、SWE-bench Pro `51.2`、IFBench `77.0` 和 GPQA Diamond `83.5`。不过，由于许多基准测试单元格缺少数据，这次比较并不完整；图片见：[i.redd.it/2cclgla7xdjh1.png](https://i.redd.it/2cclgla7xdjh1.png)。**评论认为，这说明模型实验室应该同时发布多个参数规模的版本，以免在某个规模档位被竞争对手迅速超越。一位评论者还表示，Meta 本应发布更大规模的 Glimmer 版本，例如 `70B`、`100B` 或 `400B`。也有人推测，一款 `27B` 模型如果能达到接近“Opus 4.6 Max”的水平，将会非常令人意外，同时期待 Meta 用更强的前沿模型作出回应。

    - 一位评论者指出，**Muse Glimmer** 发布时就支持 speculative decoding，据称能够提升 **TPS/吞吐量**，并询问 **Qwen** 是否也有类似的加速方案。这是该讨论中最具体的实现层面信息，但帖子没有提供具体的 TPS 数值或解码配置。
    - 一条技术性批评认为，**Muse Glimmer** 的表现不如 **Qwen**，声称 Glimmer 更容易出现“认知错误”，例如推理过程偏离主题，陷入与内容政策无关的争论，最后还与最终答案相互矛盾。评论者表示，虽然他们不太喜欢 Qwen 的写作风格，但没有在 Qwen 身上观察到同类的推理过程与最终输出不一致问题。
    - 另一位评论者将这一结果描述为 **约 27B 参数接近“Opus 4.6 Max 水平”**，认为这意味着它在 `~30B` 模型规模中具备异常强的性能。不过，该讨论没有提供具体的基准名称、分数、评测方法或可复现细节，因此无法据此验证这一比较。

  - **[Fixed Jinja chat template for Qwen 3.5, 3.6, and the new 3.8 release](https://www.reddit.com/r/LocalLLaMA/comments/1vnm7le/fixed_jinja_chat_template_for_qwen_35_36_and_the/)**（活跃度：478）：**一个由社区维护、可直接替换使用的 [Qwen fixed Jinja chat template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) 面向 Qwen `3.5`、`3.6` 以及新版 `3.8`，用于修复官方模板中报告的多项问题：`enable_thinking=false` 会触发硬异常；注入空的 `<think></think>` 会污染多轮对话历史；遇到 OpenAI 风格的 JSON 字符串工具参数时会崩溃；对话中途的 system 消息被丢弃，导致工具调用循环卡住。该模板新增了 Qwen 3.8 的 `reasoning_effort` 控制选项（`xhigh`、`high`、`medium`、`low`），可以通过 kwargs 或 `<|think_off|>` 恢复关闭推理的能力，保留此前的思考内容以便复用 prefix/KV-cache，支持 llama.cpp 的 `--reasoning-preserve`，并建议使用 `llama-server ... --jinja --chat-template-file chat_template.jinja --reasoning-format deepseek`，将思考内容输出为 OpenAI 的 `reasoning_content`。作者表示，自己无法在本地验证 `2.4T` 模型，但已经完成 `28` 项自动化测试和 tokenizer parity 检查，并希望 Qwen 3.8 用户提供反馈。**评论者质疑，为什么 Qwen 官方 chat template 会带着如此基础的回归问题发布，以及他们的 QA 是否覆盖了模板和工具调用相关路径。另一位评论者则表示，很想测试 `27B` 这样的更小、更容易使用的版本。

    - 一位评论者报告了 **Qwen 3.8 chat template 的回归问题**：`enable_thinking=false` 不仅无法关闭推理，还会直接触发**硬异常**。这表明，尽管新版模板暴露了这个开关，但新的模板路径可能并未正确处理非思考模式。
    - 另一条技术相关反馈称，在 **Qwen 3.6 + Hermes Agent + LM Studio** 的组合中，发布的模板无法实现**可靠的工具调用**，用户不得不针对这套技术栈自行编写 Jinja chat template。这说明，问题可能出在工具调用格式的集成环节，而不是基础文本生成能力本身。


### 2. GLM 5.3 和 DeepSeek V4 发布

  - **[GLM 5.3 发布](https://www.reddit.com/r/LocalLLaMA/comments/1vny9zs/glm_53_released/)**（活跃度：2227）：****Z.ai 在[官方发布文章](https://z.ai/blog/glm-5.3)中宣布了 **GLM-5.3**，配套的[基准测试图表](https://i.redd.it/eixnxdnvz9jh1.png)显示，GLM-5.3 在编码、Agent 自动化和安全相关评测中大幅领先 GLM-5.2。图中显示，GLM-5.3 在 `AutomationBench`、`CyberGym` 和 `GDPVal-AA v2` 等基准测试中处于领先或极具竞争力的位置；不过在 `DeepSWE` 和 `ExploitBench` 等部分任务上，GPT-5.6 Sol 或 Mythos/Fable 5 等模型仍然领先。**评论者大多将其视为又一次快速发布的中国模型；有人指出，虽然这看起来是一次 API 模型发布，但由于团队据称表示**后续将开放权重**，因此这一消息对社区仍然具有参考价值。

    - 一位评论者指出，目前大家讨论的 **GLM-5.3** 更像是一次 API 模型发布，而不是立即开放权重；但他认为，这一消息对本地模型和开放模型社区仍然很重要，因为团队据称已经表示**权重即将发布**。这意味着，一旦检查点可用，该模型未来可能会对自托管或基准测试产生重要影响。
    - 发布文案中有一句技术表述尤其受到关注：*“我们对 GLM-5.3 所做的一切，就是扩展后训练规模。”* 评论者认为这句话很值得注意，因为它暗示模型的提升主要来自更大规模或更密集的后训练、RL 或指令微调，而不是采用全新的基础架构或重新进行预训练。

  - **[DeepSeek：今天发布 DeepSeek-V4-Pro！](https://www.reddit.com/r/LocalLLaMA/comments/1vn8m1x/deepseek_were_launching_deepseekv4pro_today/)**（活跃度：729）：****DeepSeek** 在 X 上宣布了 **DeepSeek-V4-Pro**（[帖子](https://x.com/deepseek_ai/status/2087864585504305397)），评论者指出，模型权重已在 Hugging Face 上以 [`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) 的形式发布。一条获得高赞的技术评论通过附带的价格图表强调了**新的 API 定价**，这意味着与 DeepSeek 之前的产品相比，价格可能有明显上涨。**评论者认为，涨价削弱了 DeepSeek 原本最主要的优势：它虽然“消耗 token 较多、速度也稍慢”，但过去胜在价格便宜；如今 API 价格上涨，一些用户表示会重新转向本地推理。

    - 据报道，**DeepSeek-V4-Pro 的权重已在 Hugging Face 发布**，地址为 [`deepseek-ai/DeepSeek-V4-Pro-0813`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)，这让部分讨论从 API 的经济性转向了自托管的可行性。评论者认为，如果模型性能足够有竞争力，并且基础设施与电力成本可控，那么开放权重就可能让第三方服务商以低于官方 API 的价格提供服务。
    - 多位评论者重点讨论了**API 价格上涨**，认为 DeepSeek 过去的吸引力在于价格极低，尽管它“消耗 token 较多”且速度略慢。令人担忧的是，更高的 token 价格会让托管 API 相较于本地推理或其他服务商失去吸引力。
    - 一位早期用户质疑 DeepSeek 所宣称的与 **Kimi 3** 持平，认为 V4-Pro 无法达到 Kimi 在“知识水平，以及长期放手让它完成项目的能力”方面的表现。这一批评针对的是持续自主完成项目和保留任务上下文的能力，而不仅仅是短篇、类似基准测试的输出。

  - **[DSv4 Flash 0731 的表现好得离谱](https://www.reddit.com/r/LocalLLaMA/comments/1vnyiqa/its_actually_crazy_how_good_dsv4_flash_0731_is/)**（活跃度：556）：**这张[图片](https://i.redd.it/s6agzzyy1ajh1.png)是 **Artificial Analysis Intelligence Index** 的柱状图，显示 **DeepSeek V4 Flash 0731 max** 得分为 `52`，排名 **46/608**，与 **GPT-5.6 Terra** 和 **GLM-5.2** 等顶尖前沿模型基本处于同一梯队，后两者得分为 `53`。帖子强调了这一结果的实际意义：据称，一台价格低于 `$2k` 的本地机器就能运行这款接近榜单顶端的模型。相较于更大型的前沿模型 API，它在本地或离线推理场景中尤其值得关注。**评论者认为，这项基准测试可能高估了模型的实际能力：一位用户表示，**GLM 5.2** 在编程方面仍然强得多，而 DeepSeek 在复杂任务上会浪费 token。其他人则认为 **Qwen 3.6 27B** 更令人印象深刻，因为它以大约 `1/5` 的规模取得了相近的排名；还有人表示，DSv4 Flash 是第一款在本地运行时不会让人觉得相比前沿模型明显降级的模型。

    - 一些用户质疑 **DeepSeek V4 Flash 0731** 的标题基准测试所传达的结论，认为实际编程表现可能落后于图表结果。一位评论者表示，自己已经花费了 **`>$100` 的 API 额度**，并称该模型在复杂编程任务中经常*“浪费大量 token 做无用的调查”*，甚至可能无法收敛；相比之下，评论者认为 **GLM 5.2** 在编程方面仍然明显更强。
    - 有评论者将其与 **Qwen 3.6 27B** 做了比较，指出从相关图表来看，后者的表现似乎接近 **DeepSeek V4 Flash**，但模型规模大约只有后者的 **`1/5`**。讨论涉及的技术含义是：如果基准测试中的排名能够反映真实工作负载表现，那么 Qwen 可能在参数效率方面提供更好的权衡。
    - 一位用户重点提到了本地使用体验：他表示，**DSv4 Flash 0731** 是自己用过的第一个*“用起来不像是相较于 frontier models 的降级版”*的本地运行模型，并已将其作为家庭项目的默认主力模型。另一位评论者则批评了基准测试图表的方法，指出图表显示的是 **“Selected 46 of 608 models”**，并质疑参与比较的模型是否经过了刻意筛选，因而缺乏代表性。

  - **[Deepseek Harness is Up!](https://www.reddit.com/r/LocalLLaMA/comments/1vnb66j/deepseek_harness_is_up/)**（Activity：537）：****DeepSeek AI** 宣布推出 **DeepSeek Harness（`dsh`）**，这是一个处于开发者预览阶段的开源 Agent Harness，采用“一切皆为插件”的架构，并由 **Cordis** 驱动。Cordis 的设计理念介绍于 *A Programming Paradigm for Spatiotemporal Composability*。该项目明确处于不稳定阶段，官方 предупреж告称*“THERE WILL BE COMPATIBILITY-BREAKING CHANGES”*，并引导开发者前往其 [Discord 社区](https://discord.com/invite/Ycq5dCaS4)获取更新和参与讨论。热门评论主要集中在对生态的质疑上：一位用户不理解为什么 Agent Harness 经常使用 **TypeScript** 编写；另一位用户注意到 GitHub star 数量据称在大约一小时内从 `20k` 飙升至 `30k`，因此怀疑其中存在机器人操作；还有一位用户询问 `dsh` 能否比 **reasonix** 实现更高的缓存命中率。

    - 评论者提到了官方的 **DeepSeek Harness** 代码仓库和文档：[github.com/deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) 和 [deepseek.com/harness/en](https://deepseek.com/harness/en/)。其中一个技术疑问是，它能否比 **Reasonix** 实现更高的 prompt/cache 命中率，因为缓存效率对推理成本和延迟的重要性正不断提升。
    - 一位评论者质疑，为什么许多 Agent/Harness 实现都使用 **TypeScript** 编写，并将 **Codex** 视为可能的例外。这种担忧意味着，与 Python、Rust 或原生工具相比，TypeScript 可能会给底层性能调优或集成带来更多阻力，不过评论中没有提供基准测试或具体实现细节。

### 3. 专用本地 Transformer 构建项目

  - **[训练了一个 1.5B 模型来编写 shell 命令，这样我就不用再搜索 tar 参数了。在笔记本 CPU 上约 1 秒即可运行。](https://www.reddit.com/r/LocalLLaMA/comments/1vnl0um/trained_a_15b_to_write_shell_commands_so_id_stop/)**（活跃度：1815）：**这张图片是 `whatisit` 工具的**终端/CLI 演示启动画面**，展示的是深色终端中的 ASCII 艺术，而不是基准测试结果或模型内部信息：[图片/GIF](https://i.redd.it/di0yenio27jh1.gif)。帖子中的技术背景如下：作者使用 `125k` 条自然语言→shell 命令配对数据，对 **Qwen2.5-Coder-1.5B** 进行了微调，并将其量化为适用于 `llama.cpp` 的 **Q4_K_M**（`941MB`）。据作者报告，该模型在 CPU 上的性能为 `31.9 tok/s`，每次查询的中位耗时为 `0.59s`，占用 `1.6GB RAM`；在 **InterCode-ALFA** 上得分 `0.620`，相比之下，未经调优的 Qwen2.5-Coder-7B 得分为 `0.613`，GPT-4o 为 `0.73`。作者发布了采用 Apache-2.0 许可的模型权重，托管在 [Hugging Face](http://huggingface.co/ThorOdinson246/nl2sh-1.5b-Q4_K_M)，代码托管在 [GitHub](http://github.com/ThorOdinson246/whatisit-nl2sh)；同时还提供了静态安全检查器，因为只要给出特定提示，该模型可能生成具有破坏性的 shell 命令。**评论大多是轻松调侃，并没有深入讨论技术细节：有人开玩笑说这相当于“费这么大劲就是为了不用 man 手册”，有人分享了 `-czvf` / `-xzvf` 之类便于记忆的 tar 参数，也有人警告说，自然语言转 shell 命令的模型可能非常危险——“就像把一辆装满弹药的 T34 坦克交给婴儿一样”。

    - 有评论者询问作者是否评估过 **Gemma Shellper**，这是一个专注于生成 shell 命令、据称参数量低于 `0.5B` 的小型模型，看看能否将其作为基线或替代方案。这个比较具有技术参考价值：帖中的模型参数量为 `1.5B`，目标是在笔记本 CPU 上实现约 `1 秒` 的推理速度，因此，了解它与小得多的模型在延迟和准确率之间的取舍会很有帮助。

  - **[在 LLM 上运行 Doom——附 Hugging Face checkpoint](https://www.reddit.com/r/LocalLLaMA/comments/1vnjtyh/doom_running_on_an_llm_hugging_face_checkpoint/)**（活跃度：347）：**作者并没有训练 Doom，而是使用 **torchwright**，将 Doom 的确定性渲染器编译进一个原版 `Phi3ForCausalLM` checkpoint 中。所有权重都通过解析方式计算得到，并且可以通过原生 `transformers` 加载，无需设置 `trust_remote_code=False`（[项目说明](https://ood.dev/posts/doom/)，[源代码](https://github.com/physicsrob/torchwright_doom)）。提示词中编码了关卡几何结构、玩家姿态和观察方向，模型生成绘图命令，再由一个 `43` 行的光栅渲染宿主程序执行。`320x200` 版本的模型参数量为 `21B`，大小为 `85.87 GB`；每帧需要 `3,614` 个提示词 token 和 `53,747` 个生成 token，在 **B200** 上耗时略低于 `40 min`。更实用的 `80x50` checkpoint 需要下载 `34 GB`（[80x50 权重](https://huggingface.co/physicsrob/torchwright-doom-e1m1-80x50)，[320x200 权重](https://huggingface.co/physicsrob/torchwright-doom-e1m1)）。目前的编译器要求使用 `fp32` 权重；作者目前只在云端 **B200/A100-80** GPU 上运行过，并建议 `80x50` 模型使用 `80 GB` 显存，`64 GB` 可能也够用，但尚未测试。**主要的技术质疑集中在性能上：对于一个 `21B` 模型来说，在 **B200** 上生成 `53,747` 个 token 却耗时约 `40 min`，似乎比预期慢得多。一位评论者称，双 **RTX 3080** 可以在 `30 min` 内，让一个 `27B` 模型生成数量相近的 token，这可能说明项目存在严重的优化问题。另一位评论者则询问，项目为什么采用 LLM/文本生成架构，而不是 Transformer 图像生成器；也就是说，这种选择究竟只是为了追求 *“Can it run DOOM?”* 的新奇效果，还是有其他技术上的理由。

    - 一位评论者质疑了帖子中报告的推理性能：对于一个 `21B` 模型来说，“一帧需要一个包含 `3,614` 个 token 的提示词，再生成 `53,747` 个 token，在 B200 上耗时略低于 `40 分钟`”，这远慢于预期，可能意味着生成路径存在故障或缺乏优化。他还拿自己的配置进行了比较，称两张 RTX 3080 可以在 `30 分钟` 内，让一个 `27B` 模型生成数量相近的 token，尽管其性能远弱于 NVIDIA B200。
    - 同一位评论者还问道，项目为什么使用原版 `Phi3ForCausalLM` LLM 架构：提示词编码关卡几何结构、玩家姿态和观察方向，模型生成绘图命令，再由一个 `43` 行的宿主渲染器执行；为什么不采用基于 Transformer 的图像生成方案？他质疑这种选择是否只是为了追求新奇效果，还是有明确的技术依据。




## AI Subreddit 低技术含量内容回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini 3.7 Flash 发布基准测试

  - **[Gemini 3.7 Flash Benchmarks](https://www.reddit.com/r/GeminiAI/comments/1vngq0i/gemini_37_flash_benchmarks/)** (活跃度：1182)：**一篇标题为 **“Gemini 3.7 Flash Benchmarks”** 的 Reddit 帖子讨论了 **Google Gemini 3.7 Flash** 的基准测试结果，但所提供的摘录中并未包含具体的测试表格、指标、任务或方法论。评论者认为，对于一款主打低延迟和成本优化的“Flash”模型而言，这些结果强得出人意料；有人称它 *“作为 Flash 模型简直惊艳。”*** 讨论的核心在于基准测试是否真正有参考价值：一位评论者认为，*“97% 的 Flash 用户”* 更关注创意写作、情商、网页搜索和幻觉表现等实际能力，而不是排行榜式的分数。Gemini Flash 被视为一款性价比很高的模型，尤其是在用户感受到 DeepSeek 成本上涨的背景下。

    - 评论者认为，公布的 **Gemini 3.7 Flash** 基准测试结果对于“Flash”/低成本模型档位来说异常出色；有人将其表面表现与 **Sonnet 5** 相比，并给予了积极评价。评论中没有讨论具体分数，但整体观点是：该模型可能正在缩小与高端竞品之间的差距，同时仍保持高性价比定位。
    - 一项技术层面的批评认为，标准基准测试套件未必能反映大多数 **Flash** 的实际使用场景：一位评论者指出，*“97% 的 Flash 用户”* 更在意 **创意写作、情商、网页搜索质量和幻觉率**，而不是排行榜分数。他仍认为 Flash 可能是 **性价比最高的 LLM**，这意味着成本表现和现实场景中的可靠性比单纯赢得基准测试更重要。

  - **[Holy... Google actually did it, they actually shipped a frontier model](https://www.reddit.com/r/GeminiAI/comments/1vnin5c/holy_google_actually_did_it_they_actually_shipped/)** (活跃度：1123)：**该帖分享了对 **Google Gemini 3.7 Flash** 的上手测试，称其是一款速度极快的“主力”模型，指令遵循能力很强，且作者在测试中未观察到幻觉。一个值得注意的异常是：模型曾在一次运行中开始 *用中文推理*，但仍正确完成了任务，这可能表明存在语言路由问题，或隐藏的思维链泄露。** 评论者整体反驳了此前对 Gemini 的负面看法：有人称它在 Antigravity 中“好得多”；也有人认为它还算不上真正的前沿模型，更接近 **Claude Sonnet-class** 的日常模型，适合处理约 `80%` 的任务，并预计 **Gemini 4** 可能会达到前沿水平。

    - 一位评论者分享了在 **Google Antigravity** 中的上手体验，称新 Gemini 模型在这一 coding-agent 环境里 *“好得多”*，但没有提供具体的基准测试数据或失败案例。
    - 更偏技术性的说法是，将该模型定位为 **Claude Sonnet-class** 系统，而非绝对的前沿领跑者：它被描述为适合承担 `80% of usage` 的“主力”模型；同时有人推测，**Gemini 4** 可能才会成为真正达到明确前沿水平的模型。


### 2. Claude Code Agent 记忆与编排

  - **[Example of a real working loop orchestrator](https://www.reddit.com/r/ClaudeAI/comments/1vnnpur/example_of_a_real_working_loop_orchestrator/)** (活跃度：1567)：**这张图片（[PNG](https://i.redd.it/bj5iz1gvk7jh1.png)）展示了一个并非玩梗、能够实际运行的 **AI loop orchestrator dashboard**（“Llyod’s Mission”），用于管理周期性 Agent 会话，以及一套由 SQLite 支持的内部工单/记忆系统。该方案的核心是可配置的 **heartbeat / pulse loop**，可执行多种 playbook，例如检查收到的 bug 报告邮件、查询历史工单、查看应用日志、更新文档，以及创建和监控子会话；界面还会展示状态、模型、进度、成本，以及 `Create PR`、`Commit & Push`、`Worktree` 和 `Release Notes` 等部署操作。其技术意义在于，该编排器将 Agent 记忆视为一套运营数据库——本质上是包含 `600+` 条工单的内部 Jira/团队经验库，因此新任务能够基于跨模型积累的既有上下文开展。** 评论者普遍认为，这套方案具体展现了聊天 UI 之外的 Agent 基础设施价值，尤其适用于邮件分流和业务工作流。一位评论者提到自己也采用了类似模式——用本地历史表保存客户邮件上下文；另一位则认为，它清晰说明了如何围绕 Claude/Agent 工作流构建 harness、管理器和 dashboard。

    - 一位评论者介绍了一套接近生产环境的入站邮件编排系统：它将**历史客户邮件往来记录保存在本地表中**，作为持久化上下文。已知客户发来新邮件时，Agent 可以直接查看以往的问题记录，无需用户手动补充背景信息，从而将整个处理流程变成一种轻量级的客户支持记忆/RAG 工作流。
    - 另一位评论者描述了一种更复杂的常驻架构：**三台独立机器上运行三个 `24/7` Claude Agent**，每个 Agent 负责一个业务领域，并且能够通过多个服务商和模型创建子 Agent。它们通过一张**共享的主工单表**协同工作；此外，每个 Agent 还有自己的 Kanban 看板，可以根据子任务的职责、任务类型、服务商和模型，将专业任务分派给子 Agent。
    - 这套系统还采用了分层架构：其中一个编排器负责全局工单队列；如果任务属于其他编排器的业务范围，它可以将任务升级或转交给对应的编排器。人与系统之间通过手机上的语音控制 **“Hermes” Agent** 进行交互，它可以分配工单、转达消息并提供状态更新。

  - **[我让 Claude Code 维护一个 MISTAKES.md 文件。实际效果如何？](https://www.reddit.com/r/ClaudeCode/comments/1vn6d5r/i_make_claude_code_keep_a_mistakesmd_file_heres/)**（活跃度：1089）：**这篇帖子介绍了一套面向 **Claude Code** 的轻量级持久化记忆工作流：在代码仓库中添加 `MISTAKES.md`，并在 `CLAUDE.md` 中要求 Claude 按照“发生了什么 / 根本原因 / 后果 / 如何预防”的格式，将失败记录追加到文件中，且最新记录放在最前面。作者表示，Claude 后续会参考这个文件，从而避免重复犯错；对于反复出现的问题，则将其提升为 `CLAUDE.md` 中的强制性规则，把原本停留在“容易出问题的区域”层面的经验记忆，转化为可统计的失败模式和防护措施。**评论者也报告了类似的回归问题：Claude 会重复已知错误，或者尽管收到指令却过早停止。一位用户引用 Claude 的自我承认，称它曾经*“忽略了”*之前的指导，结果再次导致同样的问题。另一位评论者进一步扩展了这一思路：在规格说明、计划和实现完成后，通过 Hook 触发相应的“技能”来扫描过去的错误，并与当前工作进行比对；据称这种方式能够捕获许多问题。

    - 多位评论者表示，除非将过去的错误落实为工作流的一部分，否则 Claude Code 会反复犯同样的实现错误。一位用户描述道，Claude 明确承认自己曾在某个日期避开过一种有问题的实现方式，但后来却*“忽略了这一点，结果再次造成了完全相同的问题”*。这说明，像 `MISTAKES.md` 这样的被动文档，如果没有检索或强制执行机制，作用可能并不充分。
    - 有评论者介绍了一种更技术化的做法：增加一层辅助工作流，使用 **Claude Code skills + hooks**，在每个规格说明、计划和实现步骤完成后运行，扫描过去的错误，并将其与当前工作进行比较。该评论者表示，这种方式*“抓住了很多严重问题”*，这意味着真正有用的可能不是错误记录文件本身，而是针对该文件进行的自动化步骤后校验。
    - 评论者还就检索策略展开了讨论：有人认为，仅仅提及 `MISTAKES.md` 并不能可靠地触发 Claude 去查阅它，而将整个文件强行放入上下文又效率低下。他们认为，Claude 的 **memories system** 应该更有优势，因为简短的召回触发信息会自动保留在上下文中；另一位评论者则强调，如果没有可强制执行的检查，*“它实际上就等于不存在，最终总会被 LLM 忽略”*，并附上了一张实现截图：https://preview.redd.it/prj0dddf05jh1.png?width=3400&format=png&auto=webp&s=b4164b5a6ffad94c85eee175907cbd45d1efd0db2

### 3. AI Platform 定价与水印策略变化

  - **[DeepSeek 大幅上调 API 价格（2026 年 8 月 16 日生效），缓存命中价格最高上涨 1,114%](https://www.reddit.com/r/DeepSeek/comments/1vn81do/deepseek_just_massively_increased_their_api/)**（活跃度：2009）：****DeepSeek** 将于 **2026 年 8 月 16 日 16:00 UTC** 起调整其 [API 定价](https://api-docs.deepseek.com/quick_start/pricing/)，新增高峰/非高峰时段计费。其中，高峰时段（`01:00–04:00` 和 `06:00–10:00 UTC`）的价格是非高峰时段的 **2 倍**。涨幅最大的是缓存输入 token：**V4-Pro 缓存命中**价格从每百万 token 的 `$0.003625` 上涨至非高峰/高峰时段的 `$0.022/$0.044`，涨幅分别为 `+507%/+1,114%`；**V4-Flash 缓存命中**价格从 `$0.0028` 上涨至 `$0.007/$0.014`，涨幅分别为 `+150%/+400%`。未命中缓存的输入价格和输出价格也大幅上涨，其中 **V4-Pro 输出**从 `$0.87` 调整为 `$1.98/$3.96`，**V4-Flash 输出**则从 `$0.28` 调整为 `$0.66/$1.32`。**评论整体偏负面，但技术层面的讨论不多：用户认为 **DS4 主要是在价格便宜时才有吸引力**，至少有一名评论者表示已经将工作负载迁出。评论中隐含的主要运营层面担忧是：对于高度依赖缓存上下文和长对话的工作负载来说，DeepSeek 原本的成本优势将大幅缩小，尤其是在 UTC 高峰时段。

    - 一名评论者表示自己**已经迁移离开 DeepSeek**，并称 `DS4` 只有在*“价格便宜时”*才有吸引力。这意味着，除非其质量或性能足以支撑新的价格，否则涨价可能会抹平它相对于其他 API 模型的主要优势。
    - 一名来自巴西的用户指出，DeepSeek 的**非高峰定价时段**可能与当地白天的使用时间高度重合：*“非高峰时段：`7:00 > 22:00`”*。这表明，对于能够安排在折扣时段运行、且对延迟不敏感的工作负载来说，所在地区的时区可能会显著改变此次涨价的实际影响。

  - **[一些 Claude 用户对 Anthropic 新增的水印感到不满，担心在工作或上课时使用 Claude 会被发现](https://www.reddit.com/r/ClaudeAI/comments/1vndlg3/some_claude_users_are_mad_that_anthropics_new/)**（活跃度：1160）：**这篇帖子讨论了用户对 **Anthropic** 为 [Claude](https://www.anthropic.com/claude) 输出添加可检测水印或来源信号的反弹情绪。用户担心，在某些工作场所或课堂中，披露使用 AI 可能会受到惩罚，这些标记可能因此暴露他们使用了 AI。评论中还提出了一个技术边界问题：Claude 用于*校对/编辑*时，可能会让原本由人类撰写的文本被标记为与 AI 相关，从而模糊内容生成与辅助修改之间的归属界限。**评论者意见不一：有人表示职场中鼓励使用 AI，认为水印会是一种“认可”；也有人担心检测器会把自己修改过的文字误判为“AI 垃圾内容”。另有评论批评 Yahoo 将 Reddit 讨论转成新闻，但没有提供多少技术层面的实质内容。

    - 一名拥有教育行业经验的评论者认为，Anthropic 式水印在技术上并不适合作为强制执行手段，因为**开放权重模型不受同样的水印约束**。他们指出，一种可能的规避流程是：先用 Claude 完成大部分生成工作，再通过开放权重模型进行改写，从而可能移除或掩盖水印。
    - 多条评论指出了一个边界问题：如果 Claude 被用于*编辑、校对、格式整理口述内容，或重组笔记*，水印可能会将一份主要由人类创作的成果标记为 AI 生成。用户担心，检测器可能把合理的辅助使用与完全由 AI 合成的内容混为一谈，导致工作场所或学校出现误 обвин。
    - 这条关注教育的评论警告，即使统计型水印得到改进，也可能重现 AI 检测器已经暴露的问题：**误报和不公平的执行**，尤其会影响非英语母语者或神经多样性写作者，因为他们的句法可能显得较为模式化。评论者建议，设计评估时应衡量理解能力和 AI 素养，而不是把检测作为一种粗暴的学术诚信工具。