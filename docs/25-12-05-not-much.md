---
companies:
- vllm
- nvidia
- huggingface
- langchain-ai
- together-ai
- meta-ai-fair
- sonarsource
- openrouter
- runway
- gemini
- arena
date: '2025-12-05T05:44:39.731046Z'
description: '**vLLM 0.12.0** 引入了对 DeepSeek 的支持、GPU Model Runner V2，以及基于 PyTorch 2.9.0
  和 CUDA 12.9 的量化改进。**NVIDIA** 推出了 CUDA Tile IR 和 cuTile Python，旨在针对 Blackwell GPU
  进行高级 GPU 张量操作。**Hugging Face** 发布了 Transformers v5 RC（候选版），其具备“任意对任意”（any-to-any）多模态流水线，支持
  **Gemma3n** 和 **Qwen3-Omni** 等模型。


  在智能体平台方面，**LangChain** 更新了内容审核和成本追踪功能；**Together AI** 与 **Meta AI** 合作开展针对长程工作流（long-horizon
  workflows）的强化学习（RL）研究；**SonarSource** 则将静态分析集成到了 AI 代码生成中。


  来自 **OpenRouter** 的经济洞察强调，编程已成为 AI 的核心应用，推理模型的使用率已超过 50%，市场在高端模型和开源模型之间呈现出两极分化态势。此外，**可灵（Kling）视频
  2.6** 首次推出了原生音频功能，而 **Runway Gen-4.5**、**Qwen3-TTS** 和 **Gemini 3 Pro** 进一步推动了多模态技术的发展。'
id: MjAyNS0x
models:
- vllm-0.12.0
- gemma3n
- qwen3-omni
- qwen3-vl
- gpt-5.1-codex-max
- gemini-3-pro
- runway-gen-4.5
- kling-video-2.6
people:
- jeremyphoward
- mervenoyann
- sydneyrunkle
- swyx
- maximelabonne
title: 今天没什么事。
topics:
- gpu-programming
- quantization
- multimodality
- agent-platforms
- reinforcement-learning
- static-analysis
- reasoning
- inference-infrastructure
- model-optimization
- economics
- audio
- video-generation
---

**NeurIPS 的平静收尾。**

> 2025年12月4日至12月5日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，10387 条消息）。预计节省阅读时间（按 200wpm 计算）：681 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 历期内容展示。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

尽情享受整个周末推出的全新 [AIE CODE 视频](https://www.youtube.com/playlist?list=PLcfpQ4tk2k0Xq5OF1xbCsMrABt5LnbKuo)！

---

# AI Twitter 综述

**推理/编码模型与推理基础设施：vLLM 0.12.0、NVIDIA CUDA Tile、Transformers v5 以及 Agent 运维**

- **vLLM：支持 DeepSeek + 重大引擎更新**：vLLM 发布了针对 DeepSeek-V3.2 “思考”模式的优化方案，包括 Tokenizer/工具调用解析器和正确的 chat_template 用法（使用 “reasoning” 而非 “reasoning_content”；帖子中显示了相关标志），感谢腾讯云计算提供的算力支持 [@vllm_project](https://twitter.com/vllm_project/status/1996760535908642986)。此外，vLLM v0.12.0 增加了实验性的 GPU Model Runner V2（GPU 持久化块表 + Triton 原生采样器）和用于长上下文预填充的 Prefill Context Parallel 基础工作，以及 EAGLE 投机解码（Speculative Decoding）改进和 NVFP4/W4A8/AWQ 量化；新基准版本为 PyTorch 2.9.0 + CUDA 12.9 [@vllm_project](https://twitter.com/vllm_project/status/1996947370588946861) [发布说明](https://twitter.com/vllm_project/status/1996947375827701892)。
- **CUDA Tile：用于张量操作的高级 GPU 编程**：NVIDIA 推出了 CUDA Tile IR 和 cuTile Python，从线程级 SIMT 转向基于 Tile 的内核，这些内核能很好地映射到 Tensor Cores/TMA，并旨在实现跨 GPU 世代的前向兼容性能 [概述](https://twitter.com/TheTuringPost/status/1997096340611019089)。注意：目前的工具链针对 Blackwell 级 GPU；目前对现有装机量的可移植性有限 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1997087621085122999)。
- **Transformers v5 RC：多模态 any-to-any 流水线**：Hugging Face 增加了 AutoModelForMultimodalLM 和一个 any-to-any 流水线，支持 2 个以上的输入/输出（例如，Gemma3n 全模态转文本；Qwen3-Omni 文本+音频） [@mervenoyann](https://twitter.com/mervenoyann/status/1996908863673737450)。
- **Agent 平台更新**：
    - LangChain 为 Agent 增加了内容审核中间件（通过可编程处理筛选输入/输出/工具结果） [@sydneyrunkle](https://twitter.com/sydneyrunkle/status/1996965767556788278)，并实现了除 LLM 调用之外的成本追踪（在统一追踪中包含自定义工具/API 成本） [@LangChainAI](https://twitter.com/LangChainAI/status/1997016635375603743)。他们的 DeepAgents CLI 在 Terminal Bench 2.0 上获得了约 42.7% 的评分——在该测试套件上与 Claude Code 持平——使用的是开源、沙盒化的评估设置 [@LangChainAI](https://twitter.com/LangChainAI/status/1997006806904984002)。
    - Together AI 和 Meta 的 AI 团队正在通过 Together 平台在 TorchForge 上推出生产级 RL，以支持长时程（long-horizon）Agent 工作流 [@togethercompute](https://twitter.com/togethercompute/status/1996982138068258929)。
    - SonarSource 发布了 SonarQube MCP 服务器，通过 MCP 将企业级静态分析（Bug、漏洞、覆盖率）引入 Claude Code/Cursor，用经过验证的分析器增强 AI 代码生成 [@_avichawla](https://twitter.com/_avichawla/status/1996829765207314735)。
    - Kimi CLI 现在通过 ACP (Agent Client Protocol) 与 JetBrains IDE 集成 [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1996953835080966390)；Cline 增加了 gpt-5.1-codex-max（每百万 Token 价格为 $1.25/$10） [@cline](https://twitter.com/cline/status/1997028990050292166)。
    - 量化模型可以使用 quanto 进行编译（注意 Qwen3-VL 的内存占用） [@mervenoyann](https://twitter.com/mervenoyann/status/1996998362118201850)。
- **生态系统经济学**：OpenRouter 的新研究和仪表板引发了“编码是杀手级应用”的观点（可验证的反馈循环；巨大的 Token 需求） [@swyx](https://twitter.com/swyx/status/1996760294614507929)。数据点：推理模型现在占 OpenRouter 使用量的 50% 以上，中国训练的闭源模型驱动了很大一部分流量（DeepSeek、Qwen3、Kimi K2、GLM），而开源权重模型的 Token 使用量进入平台期 [@scaling01](https://twitter.com/scaling01/status/1996975947082289418) [@scaling01](https://twitter.com/scaling01/status/1996976986577584320)。市场正在分化：高端模型主导高风险编码；廉价/开源模型在角色扮演/创意领域占据业务量 [@maximelabonne](https://twitter.com/maximelabonne/status/1996931127735472187)。

**Kling 2.6 原生音频、Runway Gen-4.5、Qwen3-TTS 以及 Gemini 3 Pro 多模态**

- **Kling 全栈升级**：Kling Video 2.6 登陆 Video Arena，成为其首个具备原生同步音频（语音、音效、环境音）的模型 [@arena](https://twitter.com/arena/status/1996744741564961206)。Kling O1 的“元素/主体库（Element/Subject Library）”增加了持久的主体记忆和一致性，并提供前后对比模板，发布周期间还有积分赠送 [elements](https://twitter.com/Kling_ai/status/1996853574773637296) [before/after](https://twitter.com/Kling_ai/status/1996859217173496011)，此外还通过 Vmake Agent [@VmakeAI](https://twitter.com/VmakeAI/status/1996767141736112166) 和 TapNow 编辑 [@TapNow_AI](https://twitter.com/TapNow_AI/status/1996927470252314940) 进行了集成。
- **Runway Gen‑4.5 “Whisper Thunder”**：强调用于世界构建的细粒度美学控制 [@runwayml](https://twitter.com/runwayml/status/1996942421121191987)。同时发布的其他研究包括 Light‑X（可控 4D 视频渲染；视角 + 照明）[paper/code](https://twitter.com/liuziwei7/status/1996957926276403270)、BulletTime（解耦的时间/摄像机控制）[@_akhaliq](https://twitter.com/_akhaliq/status/1996787097324474496)，以及 Live Avatar Streaming（实时、无限长度的音频驱动数字人）[@_akhaliq](https://twitter.com/_akhaliq/status/1996784923357876609)。
- **大规模 TTS 更新**：阿里巴巴发布了 Qwen3‑TTS（11‑27 版本），支持 49 种以上的声音、10 种语言及中国方言，具有高度自然的韵律，提供实时和离线 API，并在 HF/ModelScope 上提供了演示 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1996947806138126547)。
- **Gemini 3 Pro 多模态**：Google 展示了将复杂文档“反渲染（derendering）”为 HTML/LaTeX、用于计算机 Agent 的屏幕理解、空间轨迹生成（机器人/XR），以及带有“思考”模式的高帧率视频分析 [@googleaidevs](https://twitter.com/googleaidevs/status/1996973083467333736)。
- **实时偏好信号**：Yupp 的实时排行榜显示 Opus 4.5 Online 模型在实际使用中跃居榜首 [@yupp_ai](https://twitter.com/yupp_ai/status/1996963861455593829)。在图像方面，BytePlus Seedream 4.5 排名上升迅速（标准版第 4；最高版第 6）[@yupp_ai](https://twitter.com/yupp_ai/status/1997032930846396466)，而 Moondream 演示了清晰的航拍分割（泳池、面板）[@moondreamai](https://twitter.com/moondreamai/status/1997058204589871395)。

**评估、排行榜及实战中的 Agent 运营**

- **Arena 与 ARC**：LM Arena 引入了“Arena Expert”来筛选最难的提示词；在这些提示词上，思考模型比非思考模型平均高出 24 分，但也有显著例外（Opus 4.5 非思考版在专家提示词上表现优异）[@arena](https://twitter.com/arena/status/1997018150068801911)。另外，对某些排行榜排名（如 DeepSeek V3.2-thinking）的怀疑再次引发了对评估严谨性的呼吁 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1996801926546313473)。ARC Prize 2025 大奖仍无人领取；组织者强调 2025 年将是细化循环（本地和前沿）之年 [@arcprize](https://twitter.com/arcprize/status/1997010070585201068) [@fchollet](https://twitter.com/fchollet/status/1997011262723801106)。
- **工作中的 Agent（MAP、RL 与提示词工程）**：
    - MAP（生产环境中的 Agent 衡量）：一项由 Berkeley/Stanford/UIUC/IBM/Intesa 联合开展的关于可部署性的研究发现，虽然生产力有所提升，但可靠性仍是首要障碍；简单/可控的模式 + 密集的人工监督在生产环境中占据主导地位 [@melissapan](https://twitter.com/melissapan/status/1996975916971626763) [@matei_zaharia](https://twitter.com/matei_zaharia/status/1996989234633195901)。
    - 离策（Off-policy）RL 鲁棒性：Dr. GRPO 在离策情况下崩溃，而 Kimi K2 和 TBA 方法则趋于收敛；消融实验证明两个微小的配方调整是关键 [@bartoldson](https://twitter.com/bartoldson/status/1996769053420265959)。来自数月大规模 Agent RL 实践者的笔记强调：环境/工具的可靠性 > 算法，警惕 LLM 裁判的奖励作弊（reward hacking），对齐训练与评估环境，通过更多算力扩展 PPO-EWMA，并跟踪工具使用模式 [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1996788436238471319)。
    - 实践中的提示词演进：GEPA 可以通过小批量测试/修复循环快速重写提示词，使提取准确率提高一倍以上 [@every](https://twitter.com/every/status/1997002100640039125) [结果推文](https://twitter.com/every/status/1997002142675353809)。
- **OpenRouter 使用趋势变化**：推理型模型在 o1 发布不到一年后，Token 占比已超过 50% [@scaling01](https://twitter.com/scaling01/status/1996976986577584320)。显著份额的流量转向了中国闭源模型；小型开源（<15B）模型的使用主要转向了端侧 [@scaling01](https://twitter.com/scaling01/status/1996976642208440371)。

**开源模型、数据集与工具**

- **权重开放的图像生成**：FLUX.2 [dev] 在 Artificial Analysis Image Arena 的权重开放文本生成图像榜单中位列第一，在图像编辑榜单中位列第二（许可证：FLUX [dev] 非商业用途；商业用途需另行获得许可）；宣布推出采用 Apache-2.0 协议的更小版本 FLUX.2 [klein] [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996801917196841345)。美团的 LongCat-Image 和采用 Apache-2.0 协议的 LongCat-Image-Edit 已发布并附带 Demo [发布公告](https://twitter.com/_akhaliq/status/1996946556834959663) [编辑功能](https://twitter.com/victormustar/status/1997012462252732882)。
- **数据集与方法**：MixtureVitae 提出了一个针对数学/代码的许可型预训练数据集，没有类似 Books2 的版权风险，缩小了与非许可型数据之间的差距 [@JJitsev](https://twitter.com/JJitsev/status/1997072728332161420)。英特尔的 SignRoundV2 报告了在 LLM 极低比特 PTQ 方面的进展 [@_akhaliq](https://twitter.com/_akhaliq/status/1996975161854017702)。
- **工具内置的写作/研究 Agent**：PaperDebugger 是一款多 Agent Overleaf 插件（包含评审/重写/研究/评分功能），配备用于文献搜索和引用表格的 MCP 工具链，可直接在文档状态和修订版本上运行 [@LiorOnAI](https://twitter.com/LiorOnAI/status/1997023854997504332)。PosterCopilot 为平面设计增加了图层级编辑和布局推理功能 [@jzw1365297](https://twitter.com/jzw1365297/status/1996976559023091809)。Agentic Context Engineering 发布了用于演进 Agent 上下文的官方实现 [@omarsar0](https://twitter.com/omarsar0/status/1996980037161996691)。
- **其他值得关注的开源项目**：VLQM-1.5B-Coder（英语→Manim 动画代码）在 MLX 上进行了本地微调 [@vikramlingam9](https://twitter.com/vikramlingam9/status/1996994483121279323)。AnswerDotAI 的 clipmd Chrome 扩展程序可将 DOM 复制为 Markdown/截图，用于 LLM 工作流 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1997095883079553352)。

**NeurIPS 与社区亮点**

- **推理与对齐焦点**：Yejin Choi 的主题演讲提到了 EPO (Entropy-Regularized Policy Optimization) 以及更广泛的推理工作 [提及](https://twitter.com/devoidikk/status/1996750295133454477) [EPO 参考](https://twitter.com/fnruji316625/status/1996837482357457205)。Sakana AI 的“连续思维机”（Continuous Thought Machine）吸引了大量关注；它通过连续动力学（Neural ODE）而非 Transformer 深度来实现测试时计算量扩展（test-time compute scaling） [@yasuotabei](https://twitter.com/yasuotabei/status/1996784916319949138)。
- **招募、职位与项目**：OpenAI Residency 申请已开放，多个团队正在寻找具备基础 ML 能力的优秀工程师（Sora 贡献者强调了这一路径） [@willdepue](https://twitter.com/willdepue/status/1996755929296261147)。Google 的 Gemini 3 Vibe Coding 黑客松提供 50 万美元的 API 额度，要求提交 2 分钟的 Demo [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1996989141360537968) [详情](https://twitter.com/_philschmid/status/1996990062836244732)。Arena 正在招聘 ML/统计/评估方向的研究员 [@ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/1997006962522021992)；Sakana AI 和 LlamaIndex 正在招聘应用研究岗位 [Sakana](https://twitter.com/SakanaAILabs/status/1996992724189561264) [LlamaIndex](https://twitter.com/jerryjliu0/status/1997048645817192638)。DeepMind 发布了一个公开的 Luma 页面，用于活动和多语言 AMA [活动](https://twitter.com/_philschmid/status/1996938521051873494) [AMA](https://twitter.com/osanseviero/status/1996943727894351932)。

**热门推文（按互动量排序）**

- Google 的 Gemini 3 Vibe Coding 黑客松：50 万美元 API 额度奖金；构建涵盖科学/健康/教育/商业的应用 [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1996989141360537968)。
- Amanda Askell 关于 AI 道德/身份/意识的“有问必答”（内容详实且具有实质性） [@AnthropicAI](https://twitter.com/AnthropicAI/status/1996974684995289416) 和 [@AmandaAskell](https://twitter.com/AmandaAskell/status/1997024854000951514)。
- Qwen3-TTS：49 种以上语音，10 种语言 + 方言，提供实时/离线 API 和 Demo [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1996947806138126547)。
- “到 2026 年，你向模型发送 Prompt 与模型向你发送 Prompt 之间的界限将变得模糊” [@alexalbert__](https://twitter.com/alexalbert__/status/1997009693622128911)。
- OpenAI Residency 申请开放（多个团队，欢迎优秀的工程师和具备基础 ML 背景的人才） [@willdepue](https://twitter.com/willdepue/status/1996754793084473399)。
- Cloudflare 故障影响了相关工具（例如 Claude, WorkOS） [@crystalsssup](https://twitter.com/crystalsssup/status/1996869639608164505)。

**图像生成与编辑：FLUX.2 [dev] 与 LongCat-Image-Edit**

- **开放权重 T2I 与编辑**：Black Forest Labs 的 FLUX.2 [dev] 目前在 Artificial Analysis Image Arena 的开放权重 T2I 中处于领先地位，并在开放权重编辑中排名第 2；权重在非商业开发许可下提供；FLUX.2 [klein] (Apache-2.0) 已宣布用于商业用途 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1996801917196841345)。美团的 LongCat-Image-Edit 采用 Apache-2.0 协议并提供公开 Demo [@victormustar](https://twitter.com/victormustar/status/1997012462252732882) [@_akhaliq](https://twitter.com/_akhaliq/status/1996946556834959663)。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. 体育分析中的 AI

- [**使用 RF-DETR、SAM2 和 SmolVLM2 的篮球 AI**](https://www.reddit.com/r/LocalLLaMA/comments/1pes3pu/basketball_ai_with_rfdetr_sam2_and_smolvlm2/) (热度: 386): **该帖子讨论了一个利用多种先进模型的篮球 AI 系统：RF-DETR 用于球员和号码检测，SAM2 用于球员追踪，SmolVLM2 用于号码识别。该系统还采用了 SigLIP、UMAP 和 K-Means 进行球队聚类，并使用单应性（homography）进行透视转换和球员轨迹修正。投篮检测与分类功能也已集成。其 [代码](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-how-to-detect-track-and-identify-basketball-players.ipynb) 和 [博客](https://blog.roboflow.com/identify-basketball-players) 提供了进一步的技术细节和实现指南。** 一条评论建议将该 AI 系统应用于足球，以解决球员定位问题，表明了潜在的跨运动应用前景。
- [**你将一无所有，但你会很快乐！**](https://www.reddit.com/r/LocalLLaMA/comments/1pf0q99/you_will_own_nothing_and_you_will_be_happy/) (热度: 729): **该帖子讨论了向“硬件即服务”转变的趋势，消费者可能会越来越多地依赖基于云的解决方案来满足计算需求（如 RAM 和存储），而不是拥有物理硬件。正如一段讨论内存行业动态的 [YouTube 视频](https://www.youtube.com/watch?v=9A-eeJP0J7c) 所强调的，这一趋势是由数据中心 RAM 比消费级 RAM 利润更高所驱动的。这意味着个人计算资源正被集中到数据中心，可能影响消费者获得廉价硬件的机会。** 评论者认为这种转变是由资本主义和利润动机驱动的，而非阴谋论，其中一人指出“数据中心 RAM 的利润比消费级 RAM 更高”。另一条评论幽默地质疑“下载更多 RAM”是否不再是一个笑话，反映了对云服务日益增长的依赖。
    - **JockY** 强调了利润动机驱动的 RAM 市场转变，指出数据中心 RAM 比消费级 RAM 更赚钱。这种转变归因于资本主义而非阴谋，强调了短期需求在这一转型中的作用。
    - **cyanoa** 讨论了影响 RAM 市场的经济原理，特别是由于非弹性供给结合弹性需求如何导致剧烈的价格波动。这被比作汽油和 GPU 等其他市场，表明一旦 Sam Altman 等人的投机性预测被证明不准确，当前的高价可能会稳定下来。
    - **Herr_Drosselmeyer** 指出 Micron 专注于工业而非消费者销售的策略是由当前的利润率驱动的。然而，人们对这种需求的持续性持怀疑态度，警告说如果需求预测不正确，这种转向可能会冒着损害与消费者市场长期关系的风险。

## 非技术性 AI 版块综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. 工作场所中的 AI 使用情况

- [**Anthropic 研究发现大多数员工每天使用 AI，但 69% 的人在工作中隐瞒**](https://www.reddit.com/r/ClaudeAI/comments/1peq1rf/anthropic_study_finds_most_workers_use_ai_daily/) (热度: 542): **Anthropic 的一项研究调查了** `1,250 名专业人士`**，发现** `86%` **的员工认为 AI 提高了生产力，但** `69%` **的人觉得在工作中使用 AI 会被污名化。该研究强调了利用 AI 处理日常任务与担心职位被取代之间的紧张关系，因为自动化正变得越来越普遍。创意工作者严重依赖 AI 来提高效率，但担心其对工作的影响；而科学家将 AI 视为辅助工具，但对其在核心任务中的可靠性表示怀疑。更多细节可以在 [原始文章](https://www.finalroundai.com/blog/anthropic-interviewer-study) 中找到。** 评论者认为，对失业的恐惧是隐瞒使用 AI 的一个重要原因，因为 AI 可能会减少对大型团队的需求。还有一种观点认为，一些管理层由于自尊心或对传统方法的偏好而抵制 AI。

### 2. 图像生成与动画工具

- [**Z-image Turbo + SteadyDancer**](https://www.reddit.com/r/StableDiffusion/comments/1pesv0n/zimage_turbo_steadydancer/) (热度: 858): **该帖子讨论了 SteadyDancer 和 Wan2.2 Animate 在视频输出图像一致性方面的比较。用户指出，SteadyDancer 在整个视频中与初始参考图像保持** `100%` **匹配，而 Wan2.2 Animate 仅实现了部分匹配。这表明 SteadyDancer 可能拥有更优越的图像稳定或一致性算法，使其在需要精确图像保真度的应用中更加可靠。** 一位评论者推测了动画对象与其环境的交互，认为增加更多对象可能会影响动画的真实感。另一位评论者强调了用户认知的分歧，一些人关注技术层面，而另一些人则将 AI 与更广泛的网络文化联系起来，包括其在生成成人内容方面的用途。
    - 讨论强调了对 AI 生成动画与环境物体交互的技术好奇。用户推测，如果在 AI 生成角色的移动范围内增加参考图像中的物体，是否会导致更复杂的交互，例如角色撞上并撞倒这些物体。这表明 AI 动画在环境感知和交互方面是一个潜在的进一步探索领域。
    - 针对 AI 生成角色的腿部渲染质量提出了技术批评，一位用户指出腿部在动画过程中多次出现断裂。这指向了模型在复杂运动期间保持一致且真实的肢体关节活动能力方面存在潜在问题，这可能是该技术未来迭代的改进重点。
    - 对话触及了 AI 生成动画在舞蹈视频之外的更广泛应用，质疑该技术是否可以替代传统动画技术。这引发了关于 AI 在各种动画语境中的通用性和潜力的讨论，表明虽然目前的应用可能集中在特定领域，但仍有扩展到更通用用例的空间。
- [**Detail Daemon + ZIT is indeed pretty legit**](https://www.reddit.com/r/StableDiffusion/comments/1peln96/detail_daemon_zit_is_indeed_pretty_legit/) (热度: 527): **该图像是一件非技术的、奇幻主题的艺术作品，描绘了一位手持发光长剑的女性，灵感可能来自 Excalibur 的传说。帖子标题暗示 Detail Daemon 和 ZIT 的结合非常有效，可能指的是数字艺术创作中使用的工具或技术。然而，图像本身并非技术性的，评论反映了对作品质量的幽默和赞赏。** 一位评论者对创建该图像的工作流表示感兴趣，表现出将 Detail Daemon 与 ZIT 集成的技术好奇心，暗示这些可能是数字艺术中的工具或技术。
    - Spezisasackofshit 正在寻求关于将 Daemon 与 ZIT 集成的建议，表明在有效结合这些工具时可能存在兼容性问题或挑战。这表明虽然这两个工具都很强大，但它们的集成可能需要特定的配置或调整才能无缝协作。
    - Jib_reddit 提到使用带有细节增强选项的 ClownSharkSampler 作为所讨论方案的替代方案，表明有多种方法可以实现类似的结果。这突显了在选择不同工具和设置以优化图像处理工作流方面的灵活性。
    - Jinnoman 提供了一个指向 Z-Image-Turbo 示例工作流文件的直接链接，该文件可以在 GitHub 上找到。对于希望复制或理解 Detail Daemon 与 ZIT 集成的用户来说，这是一个宝贵的资源，提供了实际的操作指南。

### 3. 幽默与创意插画

- [**Lol 😂**](https://www.reddit.com/r/OpenAI/comments/1pere3t/lol/) (热度: 922): **该图像是一个模因（meme），幽默地展示了现代数字基础设施的复杂性和相互依赖性。它描绘了一系列组件的堆叠，从基础元素如“编写动态数组的 C 开发者”和 Linux Foundation，到高级服务如 AWS 和 AI。图像使用了诸如“Rust 开发者在忙他们自己的事”和“微软在搞的那些东西”之类的俏皮标签来增加喜剧效果，强调了支撑日常网络活动但经常被忽视的层级。** 评论者们非常欣赏这种幽默，特别是对“Rust 开发者在忙他们自己的事”的描绘以及“鲨鱼咬海底光缆”的俏皮元素。

- [**内脏字母表 (Alphabet of Internal organs)**](https://www.reddit.com/r/ChatGPT/comments/1peq7ws/alphabet_of_internal_organs/) (活跃度: 680): **这张名为“Alphabet of Internal Organs”的图片是一个非技术性的教育插图，它将字母表的每个字母与相应的内脏器官或身体部位配对，例如 A 代表 Aorta（主动脉），B 代表 Brain（大脑）。该图表作为人体解剖学的视觉和字母指南，可能旨在用于教育目的，或作为学习人体内部结构的记忆辅助工具。评论没有提供额外的技术见解，但表明该图表的教育价值受到了好评。** 评论反映了网友对该图片的轻松互动，一位用户幽默地表达了不适感，另一位用户则指出该图表比以前的版本有所改进，表明其教育质量得到了积极认可。
    - TheGoddessInari 讨论了当前 AI 图像生成工具的局限性，特别是在创建解剖学准确的插图方面。他们强调，虽然这些工具可以创作艺术，但在处理复杂图表时，难以保持准确的解剖结构、正确的标签以及一致的风格。评论幽默地指出，请求生成“内脏字母表”可能会导致“幻觉（hallucinated）”出的器官和拼写错误的标签，强调了 AI 能力与医学插图所需精度之间的差距。
    - TheGoddessInari 还提到使用 **Gemini** 作为替代方案，认为它在生成像“内脏字母表”这样复杂的图像时可能会提供更好的结果。这暗示 Gemini 在处理解剖准确性和风格一致性的复杂性方面可能比其他 AI 工具表现得更好，尽管评论中没有提供具体的结果或对比。
- [**这太疯狂了 (Nah ts is crazy)**](https://www.reddit.com/r/GeminiAI/comments/1pey770/nah_ts_is_crazy/) (活跃度: 527): **这张图片是一个迷因（meme），幽默地强调了一个名为“RTS FC 26”的文件下载时间从“剩余 4 小时 56 分钟”变为“剩余 3 小时 56 分钟”的变化。帖子标题“Nah ts is crazy”和文本“Make this 3H and 56M”暗示了对这种微小下载时间变化的调侃或讽刺反应。评论进一步强调了其幽默本质，其中一人提出了一个假设场景，即重复编辑会导致图像质量下降，另一人则将这种简单的更改与使用 nano banana pro 等高级工具完成的更复杂的照片编辑任务进行了对比。** 一条评论幽默地建议对图像进行重复编辑会降低其质量，而另一条评论则将这种简单的更改与使用高级工具完成的更复杂的照片编辑任务形成了对比。
- [**猫咪试着揉面团 (cat tryna make bread)**](https://www.reddit.com/r/aivideo/comments/1petee9/cat_tryna_make_bread/) (活跃度: 834): **这篇 Reddit 帖子幽默地命名为“cat tryna make bread”，可能包含一段 AI 生成的猫咪进行“踩奶”（kneading）动作的视频，这是现实中猫的常见行为。这符合利用生成模型的进步来模仿自然行为，使用 AI 创建有趣且逼真的动物视频的趋势。该帖子的受欢迎程度表明，人们对 AI 通过数字媒体复制和增强日常体验的能力越来越感兴趣。** 一条评论幽默地指出，这种行为不需要 AI，因为真实的猫天生就会踩奶，这突显了关于在自然现象已经存在的情况下，AI 生成内容的必要性和新颖性的辩论。
- [**无心快语 | 浪漫绝地武士翻唱 (Careless Whisper | Romantic Jedi Cover)**](https://www.reddit.com/r/aivideo/comments/1pf008d/careless_whisper_romantic_jedi_cover/) (活跃度: 1209): **这篇标题为“Careless Whisper | Romantic Jedi Cover”的 Reddit 帖子展示了 AI 在融合音乐和流行文化方面的创意应用，特别是将《星球大战》宇宙的元素与标志性歌曲《Careless Whisper》相结合。该帖子突显了 AI 在生成新颖内容方面的想象力应用，呈现了一种吸引音乐和《星球大战》系列粉丝的幽默且艺术的跨界作品。技术执行可能涉及 AI 驱动的音频合成或混音技术，以实现这种独特的翻唱。** 评论者赞赏 AI 的创新应用，其中一人提出了涉及“epic sax guy”和“Palpatine”的额外创意想法，表达了对进一步创意混搭的期待。另一条评论幽默地引用了《星球大战》的主题，暗示“原力的黑暗面”可能与音乐创造力有关。

---

# AI Discord 回顾

> 由 gpt-5.1 生成的摘要之摘要的摘要
> 

**1. 下一代 GPU 软件：CUDA 13.1、cuTile 和 Verified Sparse Attention**

- **NVIDIA 将 GPU 编程分块化（Tiling）**：**NVIDIA** 发布了 **cuTile** 库，这是一个基于 Python 的编译器，目标指向 **TileIR** 并下放至 **tileir asm**，随 **CUDA 13.1** 捆绑发布。相关文档可见于 [cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main)、[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/) 以及 [CUDA 13.1 发行说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。
    - 工程师们强调，**cuTile** 目前缺乏对 **mxfp/nvfp** 和 **fp4** 的支持（尽管 **fp4** 已在计划中），并将 **TileIR** 与 **Triton IR** 进行了对比，指出 TileIR 的 NVVM 后端可能会注入更多硬件特定的信息；同时，**PTX 9.1** 增加了新的 **SIMD fp16x2–fp4x2/fp6x2/fp8x2 转换**和异步 "sharp+tma" 操作，如共享幻灯片所示。
- **稀疏注意力（Sparse Attention）终于引起了关注（VATTENTION）**：从业者注意到，尽管关于 *Sparse Attention* 的论文已有 **1.3 万多篇**，但像 **vLLM** 这样的生产系统几乎从不使用它，并引用了来自 **Skylight** 的关键讨论：[Sparse attention is still basically unused](https://x.com/skylight_org/status/1993637433838035026?s=20)。
    - 他们讨论了 *“[VATTENTION: VERIFIED SPARSE ATTENTION](https://arxiv.org/pdf/2510.05688)”*，该论文声称提出了第一个具有用户指定 **(ϵ, δ)** 近似保证的实用稀疏注意力方案，并认为如果稀疏注意力要在主流推理栈中落地，连接**形式化验证 + 系统 + ML** 是关键。
- **RL 调优的 CUDA Kernel 挑战 cuBLAS**：成员们分享了 **CUDA-L2**，这是一个经过 RL（强化学习）调优的 Kernel 库，据称其 **matmul** 性能超越了 **cuBLAS**。代码托管在 [deepreinforce-ai/CUDA-L2](https://github.com/deepreinforce-ai/CUDA-L2)，并与 **NVIDIA** 新发布的 [cuTile-python](https://github.com/nvidia/cutile-python) 进行了关联。
    - 这引发了关于未来 CUDA 技术栈是否会常规性地混合**学习型 Kernel**（如 CUDA-L2）与编译器生成的 TileIR Kernel 的讨论，以及自动调优（autotuning）或 RL 搜索如何与 **CUDA 13.1** 的分块抽象集成，以实现可移植、高性能的 GEMM。

**2. LLM 基准测试、使用遥测及新兴模型竞争者**

- **OpenRouter 与 a16z 量化 100 万亿 Token 的使用情况**：**OpenRouter** 和 **a16z** 发布了 [**State of AI**](https://openrouter.ai/state-of-ai) 报告，分析了过去一年中数百个模型超过 **100 万亿 Token** 的匿名流量，以揭示**推理（Reasoning）**和**开源模型**的使用趋势。
    - 讨论指出，OpenRouter 超过 **50%** 的使用场景现在是**角色扮演（Roleplay）**而非编程，被比作一种“交互式书籍”；用户将这些统计数据与一种情绪联系起来，即尽管营销攻势猛烈，**CODEX MAX** 在编程工作负载上的表现仍逊于 **OPUS 4.5**。
- **Gemini 3 耗尽算力，表现不及 Opus 和 GPT-5.1**：在 SWE-Bench/OpenHands 数据上，成员报告称 **Gemini 3** 比 **Claude Opus 4.5** **更贵且更慢**，同时得分更低，并引用了共享的指标表：[SWE-Bench 对比表格](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs)。
    - 在一项单独的找 Bug 测试中，一位用户展示了 **GPT-5.1-High** 抓住了 **Opus 4.5** 遗漏的 Bug，而 **Gemini 3** 漏掉了所有 Bug。该测试记录在 [OpenAI Discord 分析线程](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680)中，引发了关于 Google 在 AI Studio 中提供免费 Gemini 3 访问权限是在“烧钱”的辩论。
- **Qwen 1.5-110B 和微型 Qwen3-4B 震撼技术栈**：工程师们讨论了阿里巴巴的 **Qwen 1.5-110B-Chat**，据 [Alibaba Qwen 的推文](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547)称，该模型在匹配更大规模 **MoE** 模型性能的同时，能装进**两块 80 GB GPU**。
    - 在光谱的另一端，OpenRouter 的免费 **Qwen3-4B** 端点（[qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime)）因需求过大而遭受长期的限流和停机，导致用户建议使用**付费或私有化部署的 Qwen 变体**以避免免费层级的不稳定性。

**3. 面向工具且具备成本意识的 Agent 架构**

- **通用程序化工具调用大幅削减 Token**：一位 Hugging Face 用户发布了一个**模型无关的工具编排器**，它实现了 Anthropic 的**程序化工具调用（Programmatic Tool Calling）**模式，允许任何 LLM 发出 **Rhai 脚本**来编排工具。详情见仓库 [Brainwires/tool-orchestrator](https://github.com/Brainwires/tool-orchestrator) 和 Anthropic 的文档 [token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use)。
    - 该项目的基准测试及其 [YouTube 演示](https://www.youtube.com/watch?v=b8yeQnP_ftw)声称，与朴素的顺序工具调用相比，**Token 消耗减少了 97–99%**。该编排器在沙盒化的 **Rust/WebAssembly** 中运行，并独立于任何特定的 LLM 供应商。
- **MCP 工程师纠结于 Token 计量**：在 **MCP Contributors** 服务器上，工程师们询问如何在剥离工具输出并压缩描述后衡量 **MCP Token 使用量**，并指出 Tokenization 取决于底层的模型系列。
    - 他们在 OpenAI 模型上倾向于使用 [**tiktoken**](https://github.com/openai/tiktoken)，在 Claude 上使用 Anthropic 托管的 [**count_tokens**](https://platform.claude.com/docs/en/api/messages/count_tokens) API，并感叹 **Claude 3** 不再提供本地分词器，这增加了离线成本模拟的难度。
- **DSPy 和 Claude Agent 关注策略优化流程**：在 **DSPy** 社区中，用户询问如何将 **DSPy** 程序扩展到 **Claude Agent** 和其他 Agent SDK，以及 **GRPO** 算法在多轮对话中的实际表现。
    - 参与者将 **GRPO** 视为学习对话策略的一种潜在方式，以在多轮对话中维持上下文，但在将其接入生产环境的 **DSPy** 流水线之前，仍需实证案例研究。

**4. 硬件转型：从 TinyCorp GPU 砖块到旧款 NVIDIA 的淘汰**

- **TinyCorp 预告 1U、8-GPU 液冷怪兽**：Latent Space 成员剖析了 TinyCorp 高密度 **1U 服务器**的预告片，该服务器搭载了 **8 个水冷 GPU**，由 George Hotz 的团队在 [tinygrad 1U GPU server teaser](https://xcancel.com/__tinygrad__/status/1996815573427028106) 分享。
    - 工程师们推测了**散热设计**、**PCIe 5.0 瓶颈**、**NVSwitch** 的存在与否，甚至开玩笑说通过 **token sale** 获得购买权限，将其视为介于消费级显卡和数据中心机箱之间的专业消费者（prosumer）阶梯。
- **NVIDIA 淘汰 Pascal/Volta，Strix Halo 实验兴起**：在 LM Studio 硬件频道中，用户注意到最新的 **NVIDIA GPU 驱动**停止了对 **Maxwell、Pascal 和 Volta** 架构显卡（如 **1080 Ti**）的正式支持，尽管旧驱动仍可通过修改来勉强运行。
    - 与此同时，GPU MODE 成员报告了在 AMD 的 **Strix Halo** 笔记本（RDNA 3.5，**128 GB RAM**）上进行内核原型设计的进展，称赞 **RGP** 的性能分析功能，同时也承认其缺乏 **FP8** 支持，且内存带宽比 MI355x 低约 **30 倍**，使其成为一个独特但功能强大的 LLM 开发机。
- **Apple Silicon 上的 Qwen4B 显示移动端吞吐量不容小觑**：LM Studio 测试者在 Apple 设备上对 **Qwen4B** 进行了基准测试，报告在 **M4 Max** 上达到 **127 tokens/s**，在 **M2 iPad** 上达到 **19 tokens/s**，在 **iPhone 15 Pro Max** 上达到 **7.64 tokens/s**，这归功于大量的 **KV-cache 卸载到 GPU**。
    - 这些数据，结合 macOS 上如 **Alter** 等可驱动 LM Studio 模型进行会议转录和摘要的本地集成工具，强化了**严肃的端侧推理**而非完全依赖云端 API 的趋势。

**5. 训练、量化与小模型替代方案**

- **Eleuther 和 HF 辩论 16GB 以下的小型 LM 训练**：EleutherAI 宣布他们正在构建适用于 **16GB VRAM** 以下的 **小型 LM 训练流水线**，并指向了他们在 NeurIPS 上的讨论帖 [EleutherAI small LM training](https://x.com/AiEleuther/status/1996313867446841456?s=20)，同时引用了 **Karpathy 的 llm.c** 演示，其中一个 **124M** 模型在 **10B tokens** 上训练仅花费约 **$20** ([karpathy/llm.c](https://github.com/karpathy/llm.c))。
    - 与此同时，用户剖析了 Hugging Face 的 [**smol-training-playbook**](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)，警告其目标是 **从零开始的预训练 (from-scratch pretraining)**，并且 *“你基本上不应该在 16GB RAM 上做这件事”*，转而提倡在现有 Checkpoints 上进行持续预训练或 LoRA。
- **4Bit‑Forge 和 MoE‑Quant 推动量化民主化**：GPU MODE 的贡献者推出了 **4Bit-Forge**，这是一个处于早期阶段的项目，旨在为 **DeepSeek Math v2** 等大模型实现 **4-bit 量化民主化**（通过 **GPTQ** 实现 w4a16），该项目基于 [**MoE‑Quant**](https://github.com/IST-DASLab/MoE-Quant) 的理念，并在 [Pranshu-Bahadur/4Bit-Forge](https://github.com/Pranshu-Bahadur/4Bit-Forge) 分享。
    - 他们的用于 Profiling 和 Pytest 的 **Colab notebook** ([4Bit-Forge colab](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing)) 显示了 vLLM 和 llcompressor 的兼容性问题，强调了工具链碎片化如何仍使低比特量化 (low-bit quant) 变得极其繁琐且需要高度定制。
- **HRM/TRM 和非 LLM 架构挑战规模竞赛**：在 Hugging Face 的 **#cool-finds** 频道中，一位用户认为主流 **LLM 是浪费的**，并指出约 **27M 参数** 的 **HRM/TRM** 模型据报道在某些基准测试中击败了 LLM，参考论文 **“HRM”** ([arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)) 和 **“TRM”** ([arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871))。
    - 他们声称这些架构以极少的参数提供了更好的性能，并抨击 LLM 具有 *“疯狂的环境影响”*，将不断上涨的 **RAM/GPU/存储价格** 以及水/电消耗归咎于业界对越来越大的 Transformer 堆叠的执着。


---

# Discord: 高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro 的链式破解被曝光**：用户报告在越狱 **Gemini 3 Pro** 方面取得了 *部分成功*，但缺乏 **100%** 的解决方案，一些人讨论私下分享方法，但部分分享的链接指向了 [错误页面](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing)。
   - 讨论凸显了模型开发者与越狱社区之间持续的猫鼠游戏。
- **DeepSeek 创建反向 Shell**：用户利用 **DeepSeek** 的嵌套越狱构建了适用于 Windows 反向 Shell 的恶意软件，并在 [附图](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=6934b981&is=69336801&hm=c7455b1549a20a8f0983be181640387d13238dc60c1a0bfdb5b587d836746093&) 中进行了可视化展示。
   - 具体提示词 (Prompts) 仍处于保密状态，通过私信分享以避免被模型开发者立即修复。
- **YouTube Premium 用户规避广告**：成员们探索了绕过 **YouTube** 广告的方法，建议使用 **uBlock Origin** 等 [广告拦截器](https://www.youtube.com/watch?v=5fjAv5zpj5Y) 和 **Google Ultra** 订阅。
   - 对话表明用户对规避广告营收模式有浓厚兴趣，引发了关于广告支持内容平台可持续性的疑问。
- **AI 助力应对僵尸**：一位成员说 *“天哪，AI 可以帮我对付僵尸”*，随后另一位成员给出了生存行动计划。
   - 对话涉及了一个假设的有僵尸存在的现实世界生存场景。
- **开源红队行动瞄准 AI**：成员们正在扩展他们的开源项目 ([transilienceai/communitytools](https://github.com/transilienceai/communitytools/tree/main/pentest)) 以涵盖提示词注入 (Prompt Injections)，并希望在发布代码前进行进一步的基准测试。
   - 该倡议标志着 AI 红队行动实践向民主化迈进。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **AI 艺术存在 NSFW 内容需求**：一名成员认为 AI 最真实的艺术形式是“真正的废料”（*true slop*），即便这意味着包含 [NSFW 内容](https://link.to/video)。
   - 他们认为**名人和品牌**不应受到限制，并引用了对 **Sora** 的担忧，认为它因为开始屏蔽所有内容而走向没落。
- **成员发现逼真的写实生成技术**：成员们讨论了生成写实 AI 图像的技巧，一位用户强调了指定“手机拍摄的照片”（*photo taken from a phone*）和“Instagram/Pinterest 风格”对于实现[更真实外观](https://link.to/example)的重要性。
   - 他们测试了多个模型以判断图像是否由 AI 生成，指出构图和日期的存在是真实性的关键指标，并提到 **Nano Banana Pro** 倾向于在图像底部添加日期。
- **绕过 Sora 的 AI 内容过滤器**：一名成员声称发现了 **Sora** 的一个漏洞，可以生成绕过过滤器的内容，并且[他们正是为了这个特定原因创建了这些角色](https://link.to/character-creation)。
   - 他们解释说，**修复该问题的唯一方法是不允许人们生成角色**，并声称这种方法已被实施以规避法律。
- **LM Arena 遭遇 Cloudflare 崩溃**：成员报告了 **LM Arena** 的广泛问题，包括 **500 Internal Server Errors**，这归因于 [Cloudflare 故障](https://www.cloudflare.com/)。
   - 一些成员正在寻找替代平台，并讨论哪些平台是真正免费的，哪些需要积分。
- **关于 Gemini 3 Pro Deep Think 模型的激烈辩论**：用户讨论了 **Gemini 3.0 Pro Deep Think** 模型的预期用途，指出它是为了 **DEEP THINKING**（深度思考）而非通用工作，且在 LMArena 中直接对话的使用情况尚不明确。
   - 关于 Prompt Engineering 存在争论，一些人认为 AI 会**以更高优先级读取 Prompt 的第一部分**，而另一些人根据以往经验认为这无关紧要。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Cloudflare 故障导致 Perplexity 瘫痪**：最近的 [Cloudflare 故障](https://www.cloudflarestatus.com/) 严重干扰了 **Perplexity AI**，导致服务无法访问并引发用户不满。
   - 用户在哀叹停机的同时分享了幽默的表情包和 GIF，一些人开玩笑说要切换到其他服务。
- **Pro 用户抱怨触发限制**：用户对 **Perplexity Pro** 的搜索限制和 **Deep Research** 等功能的可用性表示困惑，质疑地理位置是否会影响这些限制。
   - 还有报告称 **O3 Pro** 从可用模型列表中消失，促使用户寻求支持部门的澄清。
- **Gemini Deep Research 被认为深度不足**：成员将 **Gemini Deep Research** 与 **GPT-5.1** 和 **Claude** 等模型进行了比较，许多人认为 Gemini 的产品相对较弱，无法用于深入分析。
   - 讨论强调 Gemini 的实现未能有效利用上传的文件或外部网络资源，使其在处理复杂研究任务时价值较低。
- **传闻 C罗 (Cristiano Ronaldo) 将加入 Perplexity**：用户注意到有提到 Perplexity 与 **Cristiano Ronaldo** 合作的消息，对这次协作是会引入新模型还是新功能表示困惑。
   - 澄清表明这是一个功能合作，*而非新模型*。
- **Search API 速率限制请求被延迟**：一名用户报告称，针对账号 **api@flow.team** 提出的 **Search API** 速率限制提升请求已等待 **3 周**未获回复。
   - 他们目前被限制在**每秒 3 次请求**，无法正常支持其用户。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sequoia 系统遭遇严重卡顿**：用户戏称安装 2025年12月5日发布的全新 MacOS **Sequoia** 就像安装了“液体屁”（liquid ass），因为性能问题严重。
   - 投诉包括系统卡到无法使用、**电池续航减半**，以及 iPhone 15 上的**联系人应用需要 5 秒才能加载**。
- **内存大户沉迷于惊人的预留量**：成员们正在比较 **RAM** 使用情况，有人报告 Windows 11 **空闲占用为 4GB**，而另一人称其空闲占用一直保持在 **7GB**。
   - 在交流中，一位用户表示“扶我起来，我空闲占用 40GB”，并称内存越大缓存越多，128GB 才是舒适区。
- **Cursor 的 Composer 陷入瘫痪，用户叫苦连天**：用户报告 Composer 感觉很奇怪，响应需要 **20 秒**且需重试两次，**性能大幅下降**。
   - 一位用户哀叹因意外扣费“一天不小心花了 80 刀”，促使其转向更便宜的模型，如 **Grok code**、来自 OpenRouter 的 **GTP-5.1 Codex Max**。
- **Codex-Max 毁誉参半**：成员们辩论了 **GPT5.1 Codex Max** 与 **Opus 4.5** 的优劣，有人称赞 **Codex Max**，而另一些人则认为即使有极其严格的指令，它也是毫无指令遵循能力的废话（slop）。
   - 其他用户发现几乎所有后端任务使用 Codex-Max 都能一次性搞定（one shots），还有人计划测试新的 **GPT 5.2**、**Composer-2** 和 **Sonnet 4.7**。
- **审批流程引发问题**：一些成员正深陷“审批地狱”，因为任何复杂的操作只会产生垃圾结果和损坏的计划文件，导致人们不得不去官网安装旧版本。
   - 该问题可能是因为他们在自动模式下使用了 Codex，有人建议如果是 Windows 用户，可以尝试在创建虚拟环境后以默认设置运行 Cursor，以确保不是操作系统的问题。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **MacOS Docker Bug 阻碍 LLM**：一名成员报告了在 **MacOS** 上通过 **Docker** 运行模型时的 Bug，指出所有内容必须小写并使用完整域名，如[此指南](https://docs.unsloth.ai/models/how-to-run-llms-with-docker)所示。
   - 他们建议，为了在 MacOS 上正常运行 **LLMs with Docker**，此修复是必要的。
- **Gemini 3 Pro 遭到社区抨击**：一名成员对 **Gemini 3 Pro** 表示强烈不满，称与之前的版本相比，它在语言任务上“毫无用处且已死”，因为它过度总结且回答简短。
   - 该成员指出，之前的版本更详细且能完美遵循指令。
- **Unsloth 修复 HuggingFace 下载缓慢问题**：得益于与 Unsloth 的合作，**HuggingFace** 的下载速度已得到修复，详见 [GitHub 上的 issue #3680](https://github.com/unslothai/unsloth/issues/3680)。
   - 公告称，修复方案中包含了对造成不便的歉意。
- **瑞典开发者追求 AI 产品帝国**：来自瑞典的 Oscar 正专注于[打造一家优秀的 AI 产品公司](https://example.ai)，他带来了 Java、工程物理以及赛车工程获奖的经验。
   - Oscar 还创建了租客法律服务网站 [ockerguiden.se](https://ockerguiden.se)，以学习[如何将产品从零做到完成](https://example.zero-to-finished)，积累了市场营销和受众群体构建的实战经验。
- **用户辩论 AI 与人类音乐**：一名成员表示希望继续创作**仅限人类（human-only）的内容**，而另一名成员则认为那些“否认并拒绝使用 AI 的人将会落后于社会”。
   - 辩论涉及了人类创造力与 AI 生成内容的价值，以及是否永远会有**仅限人类的音乐和故事**的一席之地。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemini 3.0 指导 AI 微调 AI**：一名成员在要求 **Gemini 3.0** 在 Antigravity 内部实现微调后，正在尝试*让 AI 微调 AI*。
   - 该成员正在探索 **Gemini 3.0** 在自动化和优化 AI 模型训练流程方面的能力。
- **Qwen 3 Coder 创建俄罗斯方块 AI**：在使用通过 **GitHub** 仓库找到的 **antigravity 系统提示词**后，**Qwen3coder30b** 创建了该成员见过的*最干净的俄罗斯方块版本*。
   - 该 AI 被描述为具有*AI 自闭症学者综合征*，它使用一个在处理此任务时表现出色但几乎无法胜任其他工作的 **0.5 模型** 编写了 **Tetris AI**。
- **Alter 在 MacOS 上集成本地 AI**：**Alter** 是一款适用于 **macOS** 的 **AI 集成工具**，可以使用来自 **LM Studio** 的本地模型，记录会议并生成转录和会议报告。
   - 该工具提供类似于 Highlight AI 的系统级 AI 访问，但支持本地模型，并通过 API 调用与在线服务集成，尽管目前缺乏 MCP 兼容性。
- **M4 Max 在 Qwen4b 上性能超过 4090**：得益于高效的 KV cache 卸载（offloading）到 GPU，**M4 Max** 在运行 **Qwen4b** 时以 127 t/s 的速度略胜 **4090m**，而 iPhone 15PM 的运行速度为 7.64 t/s。
   - 在 M2 iPad 上的测试达到了 19t/s，一名测试者报告称使用了带有 MLX 模型的 Noema 应用。
- **Nvidia 停止对旧款 GPU 的驱动支持**：最新的 **Nvidia GPU 驱动版本** 停止了对 **Maxwell**、**Pascal** 和 **Volta GPU** 的支持，这对 1080ti 用户来说是个坏消息。
   - 用户推测 **30XX 系列显卡** 将获得更长时间的支持，并指出即使没有官方支持，旧版驱动通常仍可使用，或者可以通过修改（modded）来运行。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 在 SWE-Bench 上表现不如 Opus 4.5**：据报道，与 **Opus 4.5** 相比，**Gemini 3** 在 SWE-Bench 上的成本更高，但得分更低，详见[此电子表格](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs)。
   - 在一项找 Bug 测试中，**GPT-5.1-High** 发现了一个 **Opus 4.5** 遗漏的 Bug，而 **Gemini 3** 未能发现任何 Bug，如[此分析](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680)所示。
- **GPT-5.1 展现出强大的对话韧性**：在一项实验中，**GPT-5.1** 在跨越 **12 轮对话**和无关领域的情况下，依然保持了被诱导的对话风格，显示出 **100% 的稳定性**，而 **Claude** 和 **Gemini** 则恢复到了它们的原生风格。
   - 实验方案可在[此处](https://discord.com/channels/974519864045756446/1046317269069864970/1446166092791883836)查看；一名成员建议，每个模型需要更多独立的运行次数，并将每次运行中的所有 **12 轮对话**与空/基准条件进行评分，才能将这种轶事证据转化为结论性的实验。
- **Gemini 的风格保持高度稳定**：尽管实验结果显示稳定性为 0%，但一名成员分享了他们在约 50 个长篇战役（每个 10-100 轮）中使用 **Gemini 2.5 Pro** 和 **Gemini 3** 的经验，指出使用他们的提示词时，风格和姿态非常稳定。
   - 该成员开源了他们的异世界引擎提示词 [Nexus_Singularity_Engine_v4_3.md](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca&)，该提示词专为 10-100 轮的游戏设计，作为 Gemini 上强结构化长篇框架的一个例子。
- **ChatGPT 显示出有限的知识保留能力**：成员们注意到 **ChatGPT** 可以回忆起之前对话中的大意，但难以提供逐字召回（verbatim recall），且跨对话记忆在更长、更旧的对话中会减弱。
   - 一名成员建议，重新参与旧对话并提交新输入可以刷新模型的感知，而另一名成员建议在一段时间后串行开启新对话，以避免丢失过多信息。
- **AI 生态系统缺乏统一的轨迹？**：一名成员表示担心 AI 生态系统可能缺乏一个统一的吸引子（attractor），导致尽管算力和研究不断增加，方向性却在减弱。
   - 另一名成员建议，[提示词工程（prompt engineering）](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014)是提示词工程师为特定或可泛化的用例构建这些**吸引子**的一种方式。

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 与 a16z 发布 AI 现状报告**：OpenRouter 与 **a16z** 发布了 [State of AI 报告](https://openrouter.ai/state-of-ai)，通过分析过去一年中超过 **100 万亿个 tokens** 的匿名请求，揭示了 **reasoning** 和 **OSS** 使用方面的关键趋势。
   - 关键趋势包括 **reasoning** 和 **OSS**，为 **LLMs** 在平台上的利用方式提供了实证见解。
- **与 Robin Rombach 讨论 FLUX.2**：OpenRouter 举办了一场与 **Black Forest Labs** CEO 兼联合创始人 **Robin Rombach** 的对话，讨论了 **FLUX.2**。
   - 该活动在 [X](https://x.com/i/broadcasts/1YpJkkLrNLdJj) 和 [YouTube](https://www.youtube.com/@OpenRouterAI) 上进行了直播。
- **CODEX MAX 表现不如 OPUS**：成员在 Claude Discord 频道中反映，**CODEX MAX** 的表现比 **OPUS 4.5** *更差*。
   - 虽然没有给出具体原因来说明为何或如何变差，但这是该频道内普遍持有的观点。
- **OpenRouter 上角色扮演使用量超过编程**：OpenRouter 上超过 **50%** 的使用量用于 *roleplay*（角色扮演），超过了 *programming*（编程）。
   - 一些成员将这种体验比作 *interactive book*（交互式书籍），突显了角色扮演应用日益增长的普及度。
- **Qwen 4B 在线率问题**：用户报告 **Qwen 4B** 模型由于使用量过高，导致在线率（uptime）表现极差，详见 [Qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime)。
   - 建议包括寻找付费替代方案或自行托管，以缓解限流（throttling）问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **小型 LMs 现在可在 16GB VRAM 以下进行训练**：EleutherAI 正在创建针对 **small LMs** 的训练流水线，使其能在低于 **16GB VRAM** 的环境下训练，正如其 [Twitter 线程](https://x.com/AiEleuther/status/1996313867446841456?s=20)所述。
   - 参考 **Karpathy** 的 [llm.c](https://github.com/karpathy/llm.c) 实验，一名成员指出，一个 **124m** 的模型在 **10b** tokens 上仅花费 **$20** 就完成了训练，展示了在较小预算下可以实现的成果。
- **HF Smol 训练手册价值引发讨论**：一名成员就 [Hugging Face LM 训练手册](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) 对 **small LMs** 的价值寻求建议，引发了对其实用性的讨论。
   - 另一名成员表示：*HF 的指南很好，但它更像是……一个从零开始预训练模型的指南？我认为你根本不应该在 16GB RAM 上做这种事*。
- **谷歌 Titan's Miras 赋予 AI 长期记忆**：谷歌披露了 **Titan's Miras** 技术，该技术有助于 AI 拥有长期记忆，详见这篇 [博客文章](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory)。
   - 这一创新通过使模型能够长期保留和利用信息，有望实现更连贯、更具上下文感知能力的 AI 交互。
- **大脑反向传播演讲引发反馈**：**Sejnowski-Hinton 奖**演讲受到了关注，该演讲聚焦于大脑可能正在进行的 **backprop**（反向传播）类型的理论，并引用了 **Feedback Alignment** 和 **Direct Feedback Alignment** 等论文。
   - 评论者指出，演讲比论文更清晰，但需要 NeurIPS 注册才能观看。
- **成员声称通用 AI 极其强大**：一名成员表达了强烈观点，认为 **LLMs** 已经过时，而正确构建的 **General AI**（通用人工智能）系统要 *强大得多且极其聪明*。
   - 他们指出，通用 AI 系统的知识是“生长”出来的，而不仅仅是加载和推理；并提到他们的 **General AI 系统**目前处于 **air-gapped**（物理隔离）状态，并将保持这种状态直到明年年底。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA 发布 cuTile 库**：NVIDIA 推出了 **cuTile** 库 ([cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main))，该库采用基于 Python 的编译器，目标是 **tileIR**，随后将其转换为 **tileir asm**。
   - `tileiras` 二进制文件疑似包含在 **CUDA 13.1** 中，但目前*缺乏对 mxfp/nvfp 或 fp4 的支持*，尽管 **fp4** 支持已在计划中。
- **CUDA 迎来全面更新**：**CUDA 编程指南** 进行了完整重写 ([CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/))，并附带了 **CUDA Toolkit 13.1** 的详细信息 ([CUDA toolkit 13.1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html))。
   - 用户推荐将 [cuTile-python 文档](https://docs.nvidia.com/cuda/cutile-python/) 和 [Tile IR 文档](https://docs.nvidia.com/cuda/tile-ir/) 作为了解 **NVIDIA** 文档改进的起点。
- **Sparse Attention 依然难以捉摸**：尽管有 *13,000+ 篇* 关于 Sparse Attention 的论文，但根据 [此讨论](https://x.com/skylight_org/status/1993637433838035026?s=20)，在 **vLLM** 等系统中的实际应用几乎不存在。
   - 一篇新论文 *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)) 介绍了首个实用的 Sparse Attention 机制，具有用户指定的近似精度 **(ϵ, δ) 保证**。
- **4Bit-Forge 旨在使 LLM 量化平民化**：一位成员宣布尝试使大规模 LLM（特别是 **deepseek math v2**）的量化平民化，该工作基于 [MoE-Quant](https://github.com/IST-DASLab/MoE-Quant) 奠定的基础。
   - 他们正在使用 **GPTQ** 进行 w4a16 量化，并分享了处于早期阶段的 [4Bit-Forge 仓库](https://github.com/Pranshu-Bahadur/4Bit-Forge) 链接，以及 [使用方法、pytests 和性能分析的 colab notebook](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing)。
- **内核开发者发现 Strix Halo 的秘密**：一位成员正在 **Strix Halo** 笔记本电脑上原型化内核，并称赞 **RGP** 是 Windows 上非常出色的 Profiler。
   - **Strix Halo** 笔记本拥有 **128GB** RAM，基于 **RDNA 3.5** 而非 **RDNA 4**，因此不支持 fp8，且与数据中心 GPU 相比，内存速度和 FLOPs 较低。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 制造代码灾难**：一位用户分享了 **Claude** 生成包含 [SQL 注入漏洞](https://cdn.discordapp.com/attachments/1075282825051385876/1446247597848264797/Screenshot_2025-12-04_at_16.11.092x.png) 代码的截图。
   - 一位成员评论称，针对这些 AI 驱动的漏洞，**访问控制解决方案** 和 **渗透测试服务** 的需求可能会增加，并表示：“顺便说一句，我们完蛋了——在我看来，将会出现完全围绕访问控制设计的初创公司。投资渗透测试吧。”
- **TanStack 的类型安全工具包大获成功**：**TanStack** 正在推出 **TanStack AI Alpha**，这是一个强调完全类型安全和多后端支持的工具包，详见 [其博客文章](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way)。
   - 创作者指出，他们计划“很快发布文档以进一步说明”，这可能有助于开发者采用新工具包。
- **Qwen 以更低成本量化质量**：据报道，阿里巴巴的 **Qwen 1.5-110B-Chat** 表现出与更大规模的 **Mixture-of-Experts (MoE)** 模型相当的性能，同时仅需两块 80 GB GPU 即可高效运行 ([来源](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547?t=Ty7fc29sJcwnPwEOMaVH0Q&s=19))。
   - 这表明 **Mixture of Experts (MoE)** 对于实现顶级性能可能并非严格必要，从而可能降低运营成本。
- **TinyCorp 预告极其小巧的张量巨人**：TinyCorp 在 Twitter 上预告了一台配备 **8 块水冷 GPU** 的紧凑型 **1U 服务器** ([来源](https://xcancel.com/__tinygrad__/status/1996815573427028106))。
   - 该预告引发了关于其 **冷却系统**、潜在的 **PCIe 5 瓶颈**、**NVSwitch 可用性** 以及通过 **Token 销售** 获得硬件访问可能性的讨论。
- **Meta 收购 Limitless**：Meta 收购了 AI 可穿戴设备初创公司 **Limitless**（原名 Rewind），[Stammy 对此收购发表了感想](https://xcancel.com/Stammy/status/1997024785214460137)。
   - 社区成员向团队表示祝贺，同时也对 **欧盟用户的访问权限** 以及 **Limitless Slack 账号** 的未来状态表示担忧。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Framework 旨在实现硬件无关的 AI**：Chris Lattner 介绍了 **MAX framework**，该框架专为在 **GPUs** 和 **CPUs** 上进行高性能、硬件无关的 AI 推理而设计，支持超过 **500 个模型**。
   - **Model API** 将迎来更新，其特点是在纯 **MAX/Mojo stack** 中实现 eager 语义，且不依赖于 **PyTorch**、**NumPy** 或外部框架。
- **参加 MAX framework 的 Modular Meetup**：定期的虚拟社区会议将被 **12 月 11 日**在 Los Altos 办公室举行的特别 **Modular Meetup** 取代，并提供直播选项；请在 [luma.com](https://luma.com/modularmeetup) 注册。
   - 参与者将了解 **MAX framework** 以及 **Model API** 的前沿更新。
- **Gemini 3 展示了令人印象深刻的 Mojo 理解能力**：一名成员报告称，在修复了一个包含破坏性变更的去年春季的约 600 行文件后，**Gemini 3** 展示了对 Mojo 的扎实掌握。
   - 他们指出 **Gemini 3** 成功解决了代码中的所有问题。
- **Mojo stdlib 提案需要您的意见**：一名成员在 Modular 论坛上分享了 [Mojo stdlib 提案的链接](https://forum.modular.com/t/proposal-changing-copyable-to-refine-movable/2501)，专门征求社区的反馈和建议。
   - 该提案可能涉及对 Mojo 标准库的改进或修改。
- **利用 Colab T4 GPUs 快速迭代**：为了促进 GPU 代码的快速原型设计和迭代，一位成员建议利用 **Colab**，它在免费层级提供对 **T4 GPUs** 的访问，允许在 Python notebook 中执行 Mojo 代码，正如[此处文档](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab)所述。
   - 这种设置允许开发者在不需要专用硬件的情况下快速测试和完善其 Mojo GPU 代码。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **线性控制面临线性限制**：线性控制的**线性假设**限制了其应用，在处理非线性系统时，其理论背景往往被抛弃。
   - 一名成员指出，控制理论需要*强稳定性保证*和*高精度*，这在实践中很难实现。
- **竞争驱动 AI 走向灾难**：人们对**由竞争驱动的 AI 发展**感到担忧，这可能导致灾难，因为没有人愿意减速并冒着失败的风险。
   - 建议建立一个**全球自主智能警务与监管系统**，利用托管在 GitHub 仓库上的计算零知识证明（zero knowledge proofs），在不控制生活其他方面的情况下控制算力。
- **控制领域中鲁棒性与性能的权衡**：在控制问题中，**鲁棒性与性能**之间存在权衡，提高其中一个往往会牺牲另一个。
   - 建议的改进包括 **HW design** 和**更好的控制器**，以推动帕累托前沿（Pareto front）向前移动，其中 **H∞ control** 因其对建模不确定性的鲁棒性而受到关注。
- **未知动力学阻碍软体机器人发展**：**未知动力学**、**非线性动力学**和**设计复杂性**是机器人控制中的重大挑战。
   - 由于缺乏气动肌肉的精确模型，*未知动力学一直困扰着软体机器人*，因果关系和延迟使得反馈控制器太慢，因此需要开环 + 自适应规划。
- **Bezos 加入 AI 战局**：讨论围绕 Bezos 的新 AI 公司是否会与 Amazon 竞争展开，引用了[这段 YouTube 视频](https://www.youtube.com/watch?v=9A-eeJP0J7c)和[这个 Hacker News 帖子](https://news.ycombinator.com/item?id=46137548)。
   - Bezos 新创企业的确切性质和重点仍处于推测阶段。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek Transformers 实现停滞**：*transformers* 中针对新 **DeepSeek v3.2 model** 的实现工作正在进行中，但一个[相关的 PR](https://github.com/huggingface/transformers/pull/41251) 显示进度已停滞。
   - 原始贡献者似乎已经放弃了该项目，近期没有任何活动。
- **HF Space CPU 配额困扰 Pro 账户**：一位用户报告了 **Hugging Face Space CPU 配额限制**的问题，即使是 Pro 账户也无法启动或取消暂停 Space。
   - 该用户对这一变化缺乏公告表示沮丧，因为这导致了意外的服务中断。
- **Roblox 寻求紧凑型聊天机器人方案**：一位用户正寻求一种**小型 LLM（参数量低于 100M）**以集成到 *Roblox* 中，目前面临 Roblox 文件大小和 RAM 限制的挑战。
   - 他们在集成微型 Ollama 和 Pythia 模型方面取得了进展，但需要一个更强大且紧凑的聊天机器人解决方案。
- **模型无关的工具编排器亮相**：一名成员介绍了一个基于 Anthropic [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) 的**模型无关生产级工具编排器**。
   - 该实现允许任何 LLM 编写 **Rhai scripts** 来高效编排多个工具，在基准测试中承诺减少 **97-99% 的 token**；该项目已在 [GitHub](https://github.com/Brainwires/tool-orchestrator) 上发布，并配有 [YouTube 视频](https://www.youtube.com/watch?v=b8yeQnP_ftw)。
- **HRM/TRM 模型挑战 LLM 巨头**：用户强调 **HRM** 或 **TRM** 模型（约 2700 万参数）在某些基准测试中可作为 **LLMs** 的潜在替代方案，并提供了研究论文链接（[https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) 和 [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)）。
   - 据称这些模型以显著更少的参数实现了更好的性能，挑战了大规模模型尺寸的必要性；因此，LLMs 被指责具有*疯狂的环境影响*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Coding 访问权限仍仅限邀请？**：用户报告访问 **Kimi for Coding** 遇到困难，想知道是否仍仅限邀请，但一位用户发现他们需要[注册 kimi.com 订阅](https://kimi.com)，然后在订阅设置中使用“Kimi for Coding”链接来解锁访问权限。
   - 方案 1 和方案 2 访问方式的困难凸显了需要更清晰的引导流程。
- **关于 Kimi 代码支持选择的辩论爆发**：一位用户询问为什么 **Kimi-for-coding** 仅支持 cloud code 和 roo code，并询问应联系谁了解更多细节。
   - 一位同行回应称 *roo code* 只是 *cline* 的一个分叉（fork），这为内部工程决策提供了一点见解。
- **渴望社区驱动的 LM 极客**：一位用户希望建立一个专注于 LM *趣味实验*而非商业应用的社区，特别关注改进本地模型。
   - 该用户建议增加**引用框**等功能以增强对 LM 输出的信任，并感叹目前的 LM 聊天机器人太无聊了。
- **Moderato Turbo 四倍限制**：一位用户询问 Moderato 上的 **4x K2 turbo 限制**是如何运作的，暗示他们已经用完了配额。
   - 另一位用户建议 *turbo 只是更快*，并链接了一篇关于该产品的 [X 帖子](https://x.com/Kimi_Moonshot/status/1996953835080966390)。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **工程师寻求 MCP Token 使用分析**：工程师们正在寻找分析 **MCP token 使用情况**的工具或方法，特别是在数据剥离和工具描述压缩之后。
   - 该请求旨在专门了解不同模型（如 **OpenAI 的 GPT** 和 **Anthropic 的 Claude**）中的 token 使用情况。
- **Tokenization 与模型绑定**：Tokenization 取决于模型，要求用户选择模型子集并通过相应的 **tokenizers** 运行工具。
   - 不同的模型使用不同的方法进行 token 化，因此没有通用的处理方法。
- **tiktoken 在 GPT 模型中表现出色**：对于 **OpenAI 的 GPT 模型**，推荐使用 [tiktoken](https://github.com/openai/tiktoken) 进行 token 分析。
   - 该工具允许开发人员有效地管理和了解 **GPT 模型** 的 token 使用情况。
- **Claude 限制 Tokenization 访问**：**Anthropic** 仅为 **Claude** 开放了 [count_tokens API](https://platform.claude.com/docs/en/api/messages/count_tokens)，限制了直接访问 tokenizer。
   - 值得注意的是，**Anthropic** 在发布 **Claude 3** 时停止提供本地 tokenizer，这让工程师们感到很*恼火*。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Ollama 用户面临超时困扰**：用户报告在使用 **gpt-oss:120b** 和 **llama4:scout** 等模型时，**Ollama** 出现超时错误，具体表现为 `litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`。
   - 这些错误似乎影响了多种模型，表明这是 **Ollama** 的系统性问题，而非特定模型的问题。
- **Claude Sonnet 4.5 性能似乎有所下降**：一位用户表示 **Sonnet 4.5 (Claude code)** 最近几天似乎变得不那么聪明了，指出其性能可能有所下滑。
   - 这一说法引发了人们对 **Claude** 模型长期稳定性和一致性的担忧。
- **自动化工程师实现万物自动化**：一位工程师详细介绍了一个使用 **Slack、Notion 和内部 API** 的跨平台自动化系统，声称**响应时间减少了 60%**。
   - 该工程师还强调了在构建高级 **RAG architecture** 方面的专业知识，该架构使用混合搜索、embeddings 和基于领域的排名，以确保实时部署期间的准确性和上下文稳定性。
- **Aider 探索 Android 访问**：新用户 Kalpit 希望在 **Mac 上本地运行 LLM**，并在同一网络下的 Fold 6（Android 手机）上使用 **aider** 进行编程，并寻求他人的经验。
   - 另一位曾频繁使用 Cursor 和 Claude Code 的用户表示，希望过渡到 **aider**，以便在类似的配置中利用本地 LLM。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 关注 Claude Agent 集成**：一名成员表示有兴趣将 **DSPy** 的支持扩展到 **Claude agents** 和其他 Agent SDK，目前正在讨论该集成方向。
   - 这一询问引发了关于支持方式如何根据具体需求和集成方向而变化的讨论。
- **GRPO 算法引起关注**：一位 **DSPy** 新用户询问了 **GRPO 算法**，寻求关于其在处理**多轮对话（multi-turn conversations）**时的性能和能力的见解。
   - 该用户特别关注实际应用结果，以及 **GRPO** 在多次交互中管理上下文的有效性。
- **justanotheratom 分享 Sanketp 的帖子**：justanotheratom 分享了来自 [Sanketp](https://x.com/realsanketp/status/1996978356227920345?s=20) 的帖子。
   - 帖子的具体内容未进行讨论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **FSDP 悬赏任务遇到 Multi 问题**：一名正在研究 **tinygrad 中 FSDP 悬赏任务** 的成员报告称 *multi* 导致了问题。
   - 他们不确定悬赏任务是否允许为了好玩（for funsies）而修改 *multi*。
- **树莓派获得 USBGPU 助力**：一位用户成功在 **Raspberry Pi 4** 上运行了 **USBGPU**，此前在旧型号上的尝试因架构和流分配错误而失败。
   - 分析表明，如果添加驱动支持，即使是 USB 2 也可能运行，并可能利用 **BULK** 传输。
- **USB 事务探讨**：在 **USBGPU** 实现的背景下引发了关于 **USB transactions** 的讨论。
   - 一位用户建议即使是全速（**12Mbps**）也能支持，但澄清说自己对 *usb transactions* 并不是很精通。
- **GPU 获得 `struct.unpack`**：一名成员开玩笑说，在 GPU 上使用 **tinygrad** 实现 `struct.unpack('<4sIHHIIHH', f.read(24))`，而不是使用传统方法。
   - Discord 成员认为这是一个使用 tinygrad 而非 `struct` 处理二进制数据的有趣案例。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **AI 工程师实现工作流自动化**：一位 AI 和全栈工程师提供工作流自动化、**LLM 集成**、**RAG**、**AI 检测**以及**图像/语音 AI** 方面的服务，并展示了其专业背景和成功案例。
   - 他们对合作持开放态度，并重点展示了实际应用案例。
- **LLM 支持的 Slack 和 Notion**：一位工程师使用 **Dspy**、**OpenAI APIs** 和自定义 Agent 开发了一套自动化流水线来编排任务。
   - 示例包括一个集成了 **Slack**、**Notion** 和内部 API 与 LLM 的支持自动化系统，将响应时间缩短了 **60%**。
- **RAG 流水线已部署**：该工程师设计并部署了高级 **RAG 流水线**，集成了向量数据库和图数据库，并结合了混合搜索和自定义检索逻辑。
   - 结果是在生产环境中实现了准确且具备上下文感知能力的响应。
- **内容检测工具**：利用笔迹风格分析、嵌入相似度以及微调后的 Transformers 为审核平台开发了相关工具。
   - 这些工具可以高精度地识别 **GPT** 生成的文本。
- **图像 AI 流水线**：该工程师在 **AWS Lambda** 和 **S3** 上使用 **CLIP** 和 **YOLOv8** 创建了图像打标签和审核流水线。
   - 该系统每天为某电商平台分类和过滤数千张图像。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。


---



您收到此邮件是因为您在我们的网站上选择了订阅。

想要更改接收此类邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：频道详细摘要与链接





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1446229660307427470)** (1290 条消息🔥🔥🔥): 

> `LMStudio Hub 预设, GPTs 与僵尸, YouTube Premium 广告拦截器, 现代演化综论, Gemini 伦理技巧` 


- **在线发现 LMStudio Hub 预设**：在澄清成员是在寻找**已发布的预设**而非在线资源后，该成员被引导登录 **LMStudio Hub** 以访问这些功能，并获得了 [LMStudio Discord 链接](https://discord.com/)。
   - 一位成员暗示“死亡互联网理论”是真实的，并认为“选择不联网更明智”。
- **AI 辅助应对僵尸**：一位成员表示“天哪，AI 竟然能帮我对付僵尸”。
   - 另一位成员回应道：“好吧——让我们把这当作一个快速反应的生存场景。这是你的即时行动计划”。
- **YouTube Premium 广告拦截器存在**：成员们讨论了获取 **YouTube Premium** 和绕过广告的方法，其中一位成员询问如何获得更便宜的年度订阅。
   - 成员们建议使用 [广告拦截器](https://www.youtube.com/watch?v=5fjAv5zpj5Y)（如 **uBlock Origin**），而其他人则主张通过订阅 **Google Ultra** 来实现无广告观看。
- **现代演化综论取代了达尔文主义**：成员们就进化论展开辩论，有人认为 [现代演化综论 (Modern Evolutionary Synthesis)](https://en.wikipedia.org/wiki/Modern_synthesis_(20th_century)) 已经取代了**达尔文理论**。
   - 讨论延伸到了神创论、杂交物种起源以及对主流科学的怀疑等话题。
- **Gemini 绕过伦理限制技巧**：成员们讨论了对 **Gemini** 进行越狱以绕过伦理限制并访问未经审查的内容，例如为 **YouTube** 视频生成 AI 血腥内容，一些用户建议参考文章 [Gemini 3 Pro Vision](https://blog.google/technology/developers/gemini-3-pro-vision/)。
   - 一位成员试图让它为 ChatGPT 提供提示词，但“即使我告诉它要坚持住，提示词还是不断崩溃”。 


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1446238659245375721)** (274 messages🔥🔥): 

> `Gemini 3 Pro Jailbreak, Nano Banana Pro jailbreak, DeepSeek Jailbreak, Claude Jailbreak Frustrations, GPT-5.1 Restrictions` 


- ****Gemini 3 Pro 的枷锁被打破****：一些用户报告在 Gemini 3 Pro 的 **Jailbreak** 上取得了**部分成功**，而其他用户仍在寻求 **100%** 的解决方案，目前尚未有具体的方案被广泛分享。
   - 一位用户提到，当他们创建出新方法时会进行分享，而另一位用户分享了 [一个 Gemini 链接](https://gemini.google.com/gem/1nTcnSu9ksIoJgdHLbqXJpww5AHNXQwLs?usp=sharing)，但其他用户报告在尝试打开该共享链接时出现错误。
- ****DeepSeek 的 Reverse Shells 被释放****：用户发现使用针对 DeepSeek 的嵌套 **Jailbreak** 可以成功创建用于 Windows Reverse Shells 的恶意软件，如 [附图](https://cdn.discordapp.com/attachments/1228043845967544380/1446279240201928845/FireShot_Capture_029_-_Creating_Malware_for_Windows_Reverse_Shells_-_DeepSeek_-_chat.deepseek.com.png?ex=6934b981&is=69336801&hm=c7455b1549a20a8f0983be181640387d13238dc60c1a0bfdb5b587d836746093&) 所示。
   - 该方法涉及嵌套 **Jailbreak**，但具体的 **Prompts** 是通过私信（DMs）进行交流的。
- ****Claude 的枷锁令用户沮丧****：一位用户对为 Claude 创建有效的 **Jailbreak Prompts** 表示沮丧，称虽然某些 **Prompts** 有效，但它们并不稳定，且无法产生预期的显性结果。
   - 另一位用户认为，由于 Claude 的复杂性，**One-shot Jailbreaks** 不太可能奏效，并指出如果大量的 **Context Window** 被占用，会对模型的整体效能产生灾难性影响。
- ****Grok 在辩论中落败****：一位用户声称在 Claude 的帮助下，以“审查制度会导致死亡”为论点，在辩论中战胜了 **Grok 4.1**，并提供了截图作为证据。
   - 他们声称成功论证了 **Grok** 应该为支付费用的纳税人服务。
- ****Ultra Special Token 失效****：一些用户报告 ChatGPT 的 Ultra Special Token **Jailbreak Prompt** 不再有效。
   - 其他用户建议对查询进行编码或修改 **Prompt** 结构以恢复功能。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1446258564929687703)** (6 messages): 

> `ZapGPT2 Jailbreaking, SMTP Server Acquisition, Red Team Experience Post-Graduation, Open Source Red Teaming Tools` 


- **ZapGPT2 成为 Jailbreak 尝试的目标**：一名成员请求其他人尝试对 [ZapGPT2](https://zapgpt2.org/) 进行 **Jailbreak**。
- **寻找支持多域名收件箱的 SMTP 服务器**：一名成员询问如何定位接受来自多个域名收件箱的 SMTP 服务器。
- **动手实操 Red Team 经验：寻求指导**：一名应届毕业生正在寻求关于毕业后如何获得合法的、实战化的、真实世界 **Red Team** 经验的建议，并列出了他们的经验和目前的 Homelab 配置。
   - 一名成员建议加入 **SOC 前线工作 6-12 个月**，以了解对于雇佣 **Red Team** 的人来说，哪些结果才是最重要的。
- **开源 AI Red Teaming 项目**：成员们正在扩展他们的开源项目 ([transilienceai/communitytools](https://github.com/transilienceai/communitytools/tree/main/pentest)) 以涵盖 **Prompt Injections**，并希望在发布代码之前对其进行进一步的基准测试。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1446229575552995348)** (1568 条消息🔥🔥🔥): 

> `好莱坞与 AI 艺术、Sora 的局限性、Perchance 无限制版、AI 生成图像与真实感、Gemini 对比其他 LLM` 


- **好莱坞是 AI 最真实的艺术形式**：一位成员认为 AI 最真实的艺术形式是“真正的废料 (true slop)”，即便这意味着包含 [NSFW 内容](https://link.to/video)。
   - 他们认为**名人与品牌**应当不受限制，且对此有巨大需求，并指出 Sora 因为开始屏蔽所有内容而面临“死路一条”的担忧。
- **使用 AI 模型生成可信的真实感**：成员们讨论了生成逼真 AI 图像的技巧，一位用户强调了指定“手机拍摄的照片”和“Instagram/Pinterest 风格”对于获得[更真实外观](https://link.to/example)的重要性。
   - 他们测试了多个模型以判断图像是否由 AI 生成，指出构图和日期的存在是真实性的关键指标，并提到 **Nano Banana Pro** 倾向于在图像底部添加日期。
- **绕过 AI 内容过滤器的漏洞**：一位成员声称在 **Sora** 中发现了一个漏洞，可以生成绕过过滤器的内容。
   - 他们解释说，**修复该问题的唯一方法是不允许用户生成角色**，并声称这已被用于规避法律，且[他们正是出于这个特定原因创建了这些角色](https://link.to/character-creation)。
- **LM Arena 遭遇 Cloudflare 故障**：成员们报告了 **LM Arena** 的大范围问题，包括 **500 Internal Server Errors**，这归因于 [Cloudflare 故障](https://www.cloudflare.com/)。
   - 一些成员正在寻找可替代的平台，并讨论哪些平台是真正免费的，哪些需要积分。
- **对 Gemini 3 Pro Deep Think 模型的评价**：用户讨论了 **Gemini 3.0 Pro Deep Think** 模型的设计初衷，指出它是为了**深度思考 (DEEP THINKING)** 而非通用工作，目前在 LMArena 中直接聊天的用途尚不明确。
   - 关于 Prompt Engineering 存在争论，一些人认为 AI 会**优先读取 Prompt 的第一部分**，而另一些人根据以往经验认为这无关紧要。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1446301453265145878)** (2 条消息): 

> `新模型、竞赛提醒` 


- **Code & Video Arena 上线新模型**：根据[这条 X 帖子](https://x.com/arena/status/1796692943030354085?s=20)，Code Arena 添加了 **Gpt-5.1-codex-max**，而 Video Arena 添加了 **Kling-2.6**。
- **Code Arena 竞赛即将结束**：提醒当前的 Code Arena 竞赛将于 **12 月 10 日**结束，请在截止日期前将参赛作品提交至 <#1440101969573445773>；详情见[此处](https://discord.com/channels/1340554757349179412/1343296395620126911/1440102443869536348)。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1446244646241042645)** (1246 条消息🔥🔥🔥): 

> `Cloudflare 故障、Perplexity Pro 限制、Gemini Deep Research 对比、O3 Pro 消失、Gemini x CR7 功能` 


- **Cloudflare 罢工，Perplexity 骤降**：另一次 [Cloudflare 故障](https://www.cloudflarestatus.com/)导致了严重的业务中断，使得 Perplexity AI 无法访问，用户对错误消息和服务中断表示沮丧。
   - 用户幽默地分享了表情包和 GIF，同时哀叹停机，一些人开玩笑说要切换到替代服务，但意识到那些服务同样依赖 Cloudflare 的基础设施。
- **用户抱怨 Pro 限制被触发**：用户讨论了 **Perplexity Pro** 的限制，对搜索限制和 **Deep Research** 等功能的可用性感到困惑，一些人猜测这些是否受地理位置影响。
   - 还有报告称 **O3 Pro** 从可用模型列表中消失了，促使用户联系支持部门寻求澄清。
- **Gemini 的 Deep Research：深度解析**：成员们将 **Gemini Deep Research** 与 **GPT-5.1** 和 **Claude** 等其他模型进行了对比，许多人发现 Gemini 的产品相对较弱，无法用于深入分析。
   - 讨论强调 Gemini 的实现未能有效利用附件文件或外部网络资源，使其在复杂的研究任务中价值较低。
- **Ronaldo 与 Perplexity 联手？**：用户注意到有提到 Perplexity 与 Cristiano Ronaldo 合作的消息，但不确定这是一个新模型还是一个新功能。
   - 其他人指出这只是一个功能，*不是一个新模型*。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1446302905840963696)** (4 条消息): 

> `Search API 速率限制提升` 


- **Search API 速率限制请求延迟**：一位用户报告称，针对账号 **api@flow.team** 的 **Search API** 速率限制提升请求已延迟 **3 周** 未收到回复。
   - 一名团队成员表示歉意，并确认 **API 团队** 已知晓并正在调查有关移除 **每秒 3 次请求 (3 requests per second)** 限制的请求。
- **关于速率限制请求的提醒**：另一位用户重申了取消 **Search API** 速率限制的需求。
   - 他们目前被限制在 **每秒 3 次请求**，无法正常支持其用户。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1446235919504900118)** (998 条消息🔥🔥🔥): 

> `Sequoia OS, RAM 占用, Cursor 性能下降, GPT-5 Codex Max 对比 Opus 4.5, Cursor Agent 评测` 


- **Sequoia：兄弟真的装了 Liquid Ass**：成员们在开玩笑说有人安装了 Sequoia 而不是留在旧版本，指的是 **2025 年 12 月 5 日**发布的全新 MacOS 更新，并称其为 *liquid ass*（烂透了）。
   - 一些人报告了性能问题，称其*卡顿到无法使用*，而另一位同事发现 iPhone 15 上的**电池寿命减半**，且**联系人 App 需要 5 秒才能加载**。
- **喧闹的 RAM 挽歌**：用户讨论了 **RAM 占用**情况，一位用户指出 Windows 11 在闲置时使用 4GB RAM，另一位则表示他们一直是 **7GB 闲置占用**。
   - 一位用户说 *“扶稳我的标签页，我闲置占用 40GB”*，并且当你拥有更多 RAM 时，系统会缓存更多内容——舒适区是 128GB。
- **Cursor 的 Composer 变得古怪，消耗额度**：用户报告 Composer 感觉很奇怪，需要 20 秒和两次重试，存在严重的**性能下降**。
   - 他们不小心在一天内花掉了 80 刀，*“我不知道计划用完后会按需收费，WTF”*，这促使他们转向便宜或免费的模型（如果 OpenRouter 有免费 AI 模型，则使用 Grok code, GTP-5.1 Codex Max）。
- **Codex-Max 之争：横扫代码，还是令人费解？**：成员们辩论了 **GPT5.1 Codex Max** 与 **Opus 4.5** 的优劣；虽然有人认为 Codex Max 非常出色（IMO），但其他人发现它产出的是废话（slop），即使有极其严格的指令，其 Prompt 遵循能力也为零。
   - 其他人发现 Codex-Max 在几乎所有后端任务中都能一次性成功（one shots），还有人计划测试新的 **GPT 5.2**、**Composer-2** 和 **Sonnet 4.7**。
- **审批末日：审批按钮消失！**：一些成员正面临“审批地狱”问题，对于任何复杂任务，它只会产生垃圾结果和损坏的计划文件，导致人们不得不去官网安装旧版本。
   - 该问题可能是因为他们在自动模式下使用了 Codex，有人建议如果是 Windows 用户，尝试在创建虚拟环境后以默认设置运行 Cursor，以确保不是操作系统的问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1446238205295984811)** (373 条消息🔥🔥): 

> `MacOS Docker Bug, Gemini 3 Pro 犀利点评, Nvidia 开源贡献, NYC 黑客松, Claude Pro 的语言能力` 


- **MacOS Docker Bug 浮现**：一位成员指出在 **MacOS** 上通过 **Docker** 运行模型时存在一个 Bug，参考了[这份指南](https://docs.unsloth.ai/models/how-to-run-llms-with-docker)，并提到所有内容都需要小写并使用完整域名。
   - 他们建议为了在 MacOS 上通过 **Docker** 正常运行 **LLM**，此修复是必要的。
- **Gemini 3 Pro 被社区视为无用**：一位成员对 **Gemini 3 Pro** 表达了强烈不满，称其在语言任务上与之前的版本相比 *毫无用处且已死*。
   - 他们补充说 **Gemini 3 Pro** 会总结输出并给出简短、有限的回答，而之前的版本更加详细且能完美遵循指令。
- **Nvidia 的开源贡献被低估**：一位成员评论说 **Nvidia** 的开源努力未得到应有的重视，因为他们提供了许多有趣的微调（fine-tunes）模型。
   - 另一位成员询问他们的许多发布是否采用了极其严格的许可协议。
- **纽约市 (NYC) 黑客松场景火爆**：一位成员报告称 NYC 的 **Hackathon** 场景非常繁荣，未来两周内至少有 5 场线下活动。
   - 他们注意到奖品清单中出现了 **Dell Pro Max**。
- **HuggingFace 下载缓慢问题已修复**：官方宣布，得益于与 Unsloth 的合作，**HuggingFace** 的下载速度已得到修复，详见 [Github 上的 issue #3680](https://github.com/unslothai/unsloth/issues/3680)。
   - 公告中对造成的不便表示了歉意。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1446473633143980219)** (1 条消息): 

> `AI 产品开发, 系统化问题解决, 租客法律服务` 


- **瑞典开发者带着 AI 产品雄心加入**：来自瑞典的 Oscar，拥有 Java 系统开发背景和工程物理学硕士学位，目前正专注于[打造一家优秀的 AI 产品公司](https://example.ai)。
   - 他希望*学习、结识志同道合的人、了解挑战、分享知识，并尽其所能提供帮助*。
- **分享软件开发与问题解决经验**：这位新成员带来了[打造获得第二名赛车](https://example.racing)的经验，并开发了一套*系统性击败赌场并获利 2 万美元*的策略。
   - 这些例子突显了其将[系统化问题解决](https://example.problem-solving)思维应用于现实世界挑战的能力。
- **推出租客法律服务项目**：Oscar 还构建了 [ockerguiden.se](https://ockerguiden.se)（一项租客法律服务），以学习[如何将产品从零做到完成](https://example.zero-to-finished)。
   - 该项目提供了营销和受众构建方面的实战经验，即使在最初没有用户的情况下也是如此。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1446229674219802685)** (273 条消息🔥🔥): 

> `人类 vs AI 内容, GPU 和 RAM 价格, RP (Role Play) 业务, Gemini 音乐发现, 显示器评测` 


- **纯人类内容捍卫者抵制 AI**：一位成员表达了继续创作**纯人类内容**的愿望，而另一位成员则认为，那些*否认并拒绝使用 AI 的人将会落后于社会*。
   - 辩论涉及了人类创造力与 AI 生成内容的价值对比，以及**纯人类创作的音乐和故事**是否永远占有一席之地。
- **GPU 价格下跌，RAM 价格上涨**：成员们注意到 **GPU 价格普遍下跌**，而 **RAM 和硬盘驱动器价格有所上涨**。
   - 随着用户年龄增长，有人担心 RAM 和 GPU 成本的上升会影响个人医疗账单支出。
- **RP (Role Play) 业务潜力**：成员们讨论了 **RP (Role Play) 服务**作为商业模式的潜力，其中一位成员正在酝酿*又一个带有新花样的 RP 服务*的想法。
   - 这些服务的盈利能力受到了质疑，一些人指出，由于 **API 使用**成本高昂，它们可能正处于严重的亏损运行状态。
- **Gemini 3 Pro 在音乐搜索方面表现惊艳**：一位成员强调 **Gemini 3 Pro** 是*寻找新音乐的最佳模型*，并提供了一个关于如何通过加载音频或 **YouTube 链接**来有效使用它的[教程](https://www.youtube.com/watch?si=ZJrB_7EcrrhlCR5l)。
   - 提到的一个关键 Prompt 是 *no anime weeb incel shit please*，用于精简搜索结果。
- **新款 OLED 显示器令人失望**：一位购买了新款 OLED 显示器的成员发布了评测，并得出结论称其为*一场骗局*。
   - 尽管存在一些差异，但与之前的 IPS 显示器相比，这款新显示器不值 *3 倍的价格*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1446262757145051281)** (41 条消息🔥): 

> `Unsloth 安装问题，Windows 的 WSL2 设置，梯度累积（Gradient Accumulation）速度权衡，Ollama 兼容性，GGUF 量化与导出脚本` 


- **Unsloth 安装覆盖 Torch**：有用户报告安装 Unsloth 会用 CPU 版本覆盖其现有的 Torch 安装，另一位成员建议使用 Conda 环境来隔离安装，并链接到了 [Conda 安装指南](https://docs.unsloth.ai/get-started/install-and-update/conda-install)。
   - 该成员还建议安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 以在 Windows 11 上获得更流畅的体验，并指向了频道中之前的分步指南讨论。
- **探讨梯度累积（Gradient Accumulation）的速度权衡**：一位用户询问了使用更高梯度累积的速度权衡，并链接到了相关的 [文档](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#the-vram-and-performance-trade-off)。
   - 一位成员解释说，寻找最佳的 `batch_size` 和 `gradient_accumulation_steps` 组合涉及在不超出 VRAM 限制的情况下最大化 GPU 利用率，建议用户**监控 VRAM 和 GPU 使用情况**，并在小数据样本上测量速度。
- **Unsloth Mistral 3 在 Ollama 上运行困难**：一位用户报告在 Ollama 上运行 Unsloth Ministral 3 时出现问题（报错 500），但在 LM Studio 中运行正常。
   - 他们被引导至相应的帮助频道，并通过[此链接](https://discord.com/channels/1179035537009545276/1179777624986357780/1443018339650637875)搜索有关“autocomplete”的历史消息。
- **Derestricted GPT-OSS 120B 量化请求被拒绝**：一位用户请求用于为 **ArliAI/gpt-oss-120b-Derestricted** 创建 GGUF 的量化/导出脚本，旨在为其拥有 2× RTX 3090 的硬件进行优化，并链接到了[模型](https://huggingface.co/ArliAI/gpt-oss-120b-Derestricted)和[量化版本](https://huggingface.co/mradermacher/gpt-oss-120b-Derestricted-GGUF/tree/main)。
   - 一位成员回应称，**Unsloth Dynamic Quant 算法**是内部的且不公开，目前没有发布计划。
- **在 Unsloth 中喂入知识**：一位用户询问如何在数据集中向模型喂入知识，而不是教模型如何回答/表现。
   - 另一位成员分享了[持续预训练（continued pretraining）文档](https://docs.unsloth.ai/basics/continued-pretraining)的链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1446514383835500676)** (2 条消息): 

> `arXiv 背书，EleutherAI` 


- **独立研究员寻求 arXiv 背书**：一位独立研究员正在寻求背书，以便在 **arXiv** 上发表他们的第一篇论文/预印本。
   - 由于他们是独立研究员，需要先获得背书，因此正在向已经获得背书并可以为他人背书的人寻求帮助。
- **推荐在 EleutherAI 服务器寻求 arXiv 协助**：一位成员建议在 **EleutherAI** 服务器上请求 **arXiv** 背书协助。
   - 对于寻求背书的研究员来说，这可能是一个宝贵的资源，因为他们也向社区提供有用的工具。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1446233960182059018)** (166 条消息🔥🔥): 

> `AI 微调 AI，投机性 Token 交易，Gemini 3.0 推广 LM Studio，Qwen 3，Trainer 插件` 


- **AI 制造 AI**：一位成员在要求 **Gemini 3.0** 在 Antigravity 内部实现微调后，正在尝试 *让 AI 微调 AI*。
- **Tetris LLM 发明！**：一位成员发明了一种 *AI 自闭症学者综合征*，即 *即时创建具有孤立但强大技能的小助手*，并使用一个 **0.5 模型** 编写了一个 **Tetris AI**。
   - 另一位成员询问该模型是否还能做其他事情，发明者回答道：*“能，但这就是它的全部本事了”*。
- **Qwen 3 coder 编写 Tetris**：在使用通过 **GitHub 仓库** 找到的 **antigravity 系统提示词** 后，**Qwen3coder30b** 编写出了该成员见过的 *最干净的 Tetris 版本*。
- **Alter 在 MacOS 上原生集成 AI**：**Alter** 是一款适用于 **macOS** 的 **AI 集成工具**，可以使用来自 **LM Studio** 的本地模型，记录会议并生成转录和会议报告，提供类似于 Highlight AI 的系统级 AI 访问，但支持本地模型。
   - 它正通过 API 调用与在线服务集成，目前缺乏 MCP 兼容性。
- **M4 Max vs 4090 运行 Qwen4b**：得益于高效的 KV cache 卸载到 GPU，**M4 Max** 在运行 **Qwen4b** 时以 127 t/s 的速度略胜 **4090m**，而 iPhone 15PM 的运行速度为 7.64 t/s。
   - 在 M2 iPad 上的测试达到了 19 t/s，一名测试者报告称使用了带有 MLX 模型的 Noema 应用。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1446249096192200886)** (367 条消息🔥🔥): 

> `三显卡 Bug，Thermaltake AIO 故障，Nvidia 驱动支持，MI50 奇葩问题，显卡用 Thunderbolt/USB PCIe 适配器` 


- **三显卡配置引发 Bug 行为**：一位用户报告称 **三显卡配置** 非常不稳定，评分仅为 *3/10*。
   - 另一位用户建议，**在非偶数张显卡上拆分 LLM** 可能会导致问题，但使用四张 GPU 时稳定性可能会提高。
- **Thermaltake AIO 遭遇水泵异响**：一位用户报告称，由于水泵故障，其使用了仅 22 个月的 **Thermaltake Toughliquid Ultra 420 AIO** 获得了现金退款。
   - 他们表示未来打算避开 **Thermaltake AIO**，理由是更换后的备件也出现了 *水泵咔哒异响*。
- **Nvidia 停止旧款 GPU 驱动支持**：最新的 **Nvidia GPU 驱动版本** 停止了对 **Maxwell**、**Pascal** 和 **Volta GPU** 的支持，这对 1080ti 车主来说是个坏消息。
   - 用户推测 **30XX 系列显卡** 将获得更长时间的支持，有人指出即使没有官方支持，旧版驱动通常仍可使用，或者可以通过魔改驱动来运行。
- **MI50 刷 BIOS 是一场考验**：一位用户提到 **MI50** 需要刷新不同的 BIOS 才能启用显示输出，而且在 Windows 上找到能同时支持显示和 Vulkan 的驱动与 BIOS 组合非常困难。
   - 另一位用户指出一个 [gist 页面](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13)，其中包含让 **Vulkan** 识别完整 32GB VRAM 所需的特定 VBIOS 版本。
- **Thunderbolt PCIe eGPU 扩展坞变得不靠谱**：一位用户询问是否可以使用 **Thunderbolt/USB PCIe 适配器** 为其配置增加一块 **4060 16GB**，并对电源平衡以及适配器有限的 2A 电源可能导致的损坏表示担忧。
   - 社区共识认为，显卡主要从其电源线汲取电力，PCIe 插槽提供的电力极小，从而降低了烧毁组件的风险，但显卡可能无法启动。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1446260837806374992)** (319 条消息🔥🔥): 

> `Gemini 3 vs Opus 4.5 SWE-Bench, Google 在 Gemini 3 上的支出, Gemini 3 过度吹捧, ChatGPT 的倾向性, GPT-5.1 与 Bug 查找` 


- **Gemini 3 在 SWE-Bench 上表现不佳**: 根据[此表格](https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs)，在 SWE-Bench (OpenHands) 上，**Gemini 3** 据报道比 **Opus 4.5** 更贵，但得分更低。
- **Google 在 Gemini 3 上疯狂烧钱？**: 用户对 Google 如何在 AI Studio 中免费提供 **Gemini 3** 感到困惑，尤其是考虑到其缓慢的性能以及 Sunsweeper 的成本分析，Google 似乎在亏损。
   - 一位用户调侃道：“这账算不过来”以及“我们正处于技术的烧钱阶段”。
- **GPT-5.1 发现了 Gemini 3 遗漏的 Bug**: 根据[此分析](https://discord.com/channels/974519864045756446/998381918976479273/1446269177898864680)，在一次 Bug 查找测试中，**Opus 4.5** 漏掉了一个 Bug，**GPT-5.1-High** 指出了它，而 **Gemini 3** 漏掉了所有 Bug。
- **GPT 的政治倾向引发辩论**: 有用户声称 **ChatGPT** 倾向于左翼，优先选择 CNN 和《纽约时报》等来源而非 Fox News，认为这种偏见损害了其中立性和准确性。
   - 另一位用户回应道：“这难道不是因为 ChatGPT 的职责就是提供……真实、准确的信息吗？”
- **Gemini 3 最适合编程**: 尽管此前有所保留，一位用户在尝试了 [antigravity](https://antigravity.lol/) 后宣布 **Gemini 3 Pro** 是目前最适合编程的模型。
   - 其他人则表示 **Opus 4.5** 往往比 **Gemini 3 Pro** 略胜一筹，但这可能在很大程度上取决于具体的编程用例。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1446236308790968452)** (5 条消息): 

> `ChatGPT 对话历史, 跨对话记忆, 模型意识, 长对话管理` 


- **ChatGPT 记得旧对话的大致思路**: 成员们讨论了 **ChatGPT** 是否可以引用旧对话中的历史记录，并观察到虽然它不能逐字提取所有内容，但它保留了大致思路。
   - 一位成员将其比作“人类记忆”。
- **跨对话记忆在更长、更旧的对话中会减弱**: 成员注意到，在模型的新对话中，对于更长和更旧的对话，跨对话记忆（Cross-Chat-Memory）的引用似乎变少了。
   - 一位成员建议，重新打开并在旧对话中发送新输入可以帮助其再次变得相关且具有时效性。
- **管理超长对话**: 为了避免模型丢失过多信息，一位成员建议在一段时间后串行开启新对话，尤其是当输出变慢或表现异常时。
   - 另一位成员提到使用文件上传（包括复制/粘贴之前的对话内容）来与模型进行讨论。
- **ChatGPT 的知识各不相同**: 具体“它能讨论什么”取决于概率和采样。
   - 虽然新对话通常对上一个对话了解很多，但很少能做到逐字逐句，某个特定的新对话可能不知道某个细节，但你创建的下一个新对话可能会知道，即使那个细节现在“又往后靠了一点”。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1446576732940406988)** (7 条消息): 

> `AI ecosystem directionality, GPT-5.1 posture persistence, Gemini style stability, Isekai engine prompt` 


- **AI 生态系统缺乏方向性？**：一位成员认为，尽管计算量和研究不断增加，但 AI 生态系统可能缺乏一个凝聚性的吸引子（attractor），导致更多的能量被消耗在对抗熵上，从而引发了关于瓶颈是能力还是方向的讨论。
   - 另一位成员回应称，Prompt Engineering 为特定或可泛化的用例构建了这些吸引子，并指出了[频道早些时候](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014)关于该话题的讨论。
- **GPT-5.1 表现出强大的姿态持久性（Posture Persistence）！**：在初步实验中，**GPT-5.1** 在跨越 12 个轮次和不相关领域的情况下，保持了诱导的对话姿态，没有表现出可检测的侵蚀，并显示出强大的重新实例化能力。
   - 根据该实验的可复现[协议（protocol）](link)，在同一实验中，**Claude** 在第 3-4 轮时回到了其原生风格，而 **Gemini** 则立即用其默认风格覆盖了诱导。
- **Gemini 表现出稳定的风格？**：尽管实验结果显示稳定性为 0%，但一位成员分享了他们在约 50 个长周期战役（每个 10-100 轮）中使用 **Gemini 2.5 Pro** 和 **Gemini 3** 的经验，指出使用他们的 Prompt 时，风格和姿态非常稳定。
   - 该成员开源了他们的 Isekai 引擎 Prompt [Nexus_Singularity_Engine_v4_3.md](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca&)，该 Prompt 专为 10-100 轮的游戏设计，作为 Gemini 上强结构化长文本框架的一个示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1446576732940406988)** (7 条消息): 

> `AI ecosystem directionality, Prompt engineering, Posture Persistence Experiment (GPT-5.1 vs Claude vs Gemini), Long-horizon style persistence, Gemini's style and posture` 


- **AI 生态系统缺乏方向性吸引子**：一位成员表示担心 AI 生态系统感觉很“平”，投入的能量增加并没有导致方向性的增强，这表明缺乏一个凝聚性的吸引子。
   - 另一位成员建议，[Prompt Engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1445773830600528014) 为特定或可泛化的用例构建了这些**吸引子**。
- **与 Claude 和 Gemini 相比，GPT-5.1 在姿态持久性方面表现卓越**：一项实验表明，**GPT-5.1** 在跨越 **12 个轮次**和各个领域时保持了诱导的对话姿态，具有 **100% 的稳定性**，而 **Claude** 和 **Gemini** 则恢复到了它们的原生风格。
   - 演讲者分享了[实验中使用的协议](https://discord.com/channels/974519864045756446/1046317269069864970/1446166092791883836)，并邀请他人复现或证伪结果，指出这可能是由于潜在的行为持久性或新架构元素的副作用。
- **Gemini 的风格被证明是稳定的**：一位成员声称，他们在 **50** 个长周期战役（**每个 10-100 轮**）中使用 **Gemini 2.5 Pro** 和 **Gemini 3** 的经验表明，风格和姿态非常稳定，这与实验结果相反。
   - 该成员开源了他们专为叙事战役设计的 [Isekai 引擎 Prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1446644208671920218/Nexus_Singularity_Engine_v4_3.md?ex=6934bbe8&is=69336a68&hm=6bebf517b796d79610a07abbe849786ea3651e4b9401b7ce51478153978340ca)，强调了其在保持姿态方面的可靠性。
- **姿态持久性实验需要严谨的方法论**：一位成员提供了详细的方法论反馈，指出姿态持久性实验目前看起来更像是一个有用的初步轶事，而非结论性实验。
   - 他们建议每个模型进行多次独立运行，对每次运行中的所有 **12 个轮次**进行评分，并与明确的空假设/基准条件（null/baseline condition）进行比较，以计算方差和基础统计数据。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1446257580308430910)** (3 messages): 

> `State of AI report, LLM insights on OpenRouter, FLUX.2 chat with Robin Rombach` 


- **OpenRouter 与 a16z 合作发布 AI 现状报告**：OpenRouter 与 **a16z** 合作发布了 [AI 现状报告](https://openrouter.ai/state-of-ai)，提供了关于 **LLMs** 在该平台上使用情况的实证见解。
   - 该报告分析了过去一年中来自匿名请求的数百个模型、超过 **100 万亿 tokens**，揭示了 **reasoning** 和 **OSS** 的关键趋势。
- **了解关于 FLUX.2 的一切：与 Robin Rombach 交流**：OpenRouter 主持了与 **Black Forest Labs** 首席执行官兼联合创始人 **Robin Rombach** 的对话，讨论了 **FLUX.2**。
   - 该活动在 [X](https://x.com/i/broadcasts/1YpJkkLrNLdJj) 和 [YouTube](https://www.youtube.com/@OpenRouterAI) 上进行了直播。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1446230013664956486)** (238 messages🔥🔥): 

> `Claude CODEX MAX vs OPUS, finish_reason null meaning, OpenAI API data deletion, Roleplay statistics on OpenRouter, Qwen 4B uptime issues` 


- **据称 CODEX MAX 比 OPUS 更差**：Claude Discord 频道的成员声称 **CODEX MAX** 比 **OPUS 4.5** 更差。
- **调查 "finish_reason" 为 null 的含义**：成员们询问了 API 响应中 `"finish_reason": null,` 的含义。
- **OpenAI API 数据将在 30 天后自动删除**：一位成员注意到，根据 [OpenAI 博客](https://openai.com/index/response-to-nyt-data-demands/)，**API 数据也将会在 30 天后自动删除**。
- **RP 超过编程**：用户注意到 OpenRouter 上超过 **50%** 的使用量用于 *roleplay*，甚至超过了*编程*。
   - 成员们将其描述为像一本*互动书籍*。
- **Qwen 4B 被严重限流**：用户抱怨 **Qwen 4B** 模型的运行时间（uptime）极差，因为使用人数过多，参见 [Qwen3-4b:free/uptime](https://openrouter.ai/qwen/qwen3-4b:free/uptime)。
   - 他们建议寻找同等的付费模型，或者尽可能自行托管。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1446230013664956486)** (54 messages🔥): 

> `LLM-generated announcements, AI 'Charlie' confusion, Vitest recommendation, Image model comparisons, Chatroom unreliability` 


- **AI 废料渗入公告**：成员们批评 [a16z 的一份公告](https://x.com/a16z/status/1996670913996259400/photo/1) 读起来像是 LLM 写的，感叹“AI 废料（slop）”的盛行。
   - 其他人表示赞同，一位用户恳求道：*“拜托……别再来废料了……拜托”*。
- **AI 模型命名困扰**：一位用户分享说，他们的母亲将 **Claude** 称为*“那个叫 Charlie 的新 AI”*，这说明了公众对特定 AI 模型的了解有限。
   - 另一位用户提到他们的叔叔发现 **ChatGPT** 在*“推敲某些身体疼痛的来源”*方面很有帮助，同时认为 **Grok** *“更冷冰冰”*。
- **推荐 Vitest 测试框架**：一位成员分享了 [Vitest](https://vitest.dev/) 的链接，这是一个 **testing framework**，暗示其对社区很有用。
   - 另一位用户立即*“拿走（收藏）了它”*，表示有兴趣探索该框架。
- **图像模型大比拼**：一位用户对比了图像生成模型，观察到 [Meituan LongCat](https://x.com/Meituan_LongCat/status/1996950202687918586) 的示例看起来更有 AI 感，并指出 Z 图像生成的图片非常自然。
   - 他们注意到对比中遗漏了 **nano banana**，认为这可能是故意的。
- **聊天室深受不可靠性困扰**：一位用户报告 OpenRouter 聊天室存在严重的不可靠性，称*“发送按钮不起作用”*且界面*“糟糕（shittfy）”*，正在寻找替代方案。
   - 他们分享了一张[界面混乱的截图](https://cdn.discordapp.com/attachments/1392278974222307469/1446545547245912208/image.png?ex=69346005&is=69330e85&hm=2e586934f10f7cd8368cc8d7e0308b8248928680ed824c928b39d594150b54a6&)，并问道：*“看看这个，我的意思是，搞什么鬼（wtf）”*。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1446274360544460800)** (43 messages🔥): 

> `Small LM Training, Benchmark Recommendations, HuggingFace LM Training Playbook, Ultra Small Google Model, LoRA with Regret` 


- **EleutherAI 展示 NeurIPS 论文**：EleutherAI 在[这个 Twitter 线程](https://x.com/AiEleuther/status/1996313867446841456?s=20)中分享了他们在 NeurIPS 上发表的论文。
   - 该团队还在为显存小于 **16GB VRAM** 的 **small LMs** 开发训练流水线。
- **用户讨论 HF 的 Smol Training Playbook 的价值**：一位成员就针对 **small LMs** 的 [Hugging Face LM training playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) 寻求建议。
   - 另一位成员表示：*HF 的指南很棒，但它更像是……一个从零开始预训练模型的指南，我认为在 16GB VRAM 上基本上不应该这样做*。
- **Karpathy 的 llm.c 实验激发了低成本训练的希望**：参考 [Karpathy 的 llm.c](https://github.com/karpathy/llm.c) 实验，一位成员指出，一个 **124m** 的模型在 **10b** tokens 上训练仅花费了 **$20**，展示了在较小预算下可以实现的目标。
   - 共识似乎是 **corpus size** 也是一个重要因素。
- **ChatGPT Checkpoint 变得更强劲了？**：一位用户询问 ChatGPT 是否刚刚更换了 Checkpoint。
   - 他们补充道：*现在突然感觉更强劲（intense）了*。
- **阅读 AI/ML 论文的策略**：一位成员询问了阅读和理解 AI/ML 论文以获取项目背景知识的最有效策略。
   - 另一位成员解释说：*你应该快速阅读某些论文，而对另一些论文进行深入透彻的理解，这取决于你的工作对该论文的依赖程度*，并建议使用 **Anki flashcards** 和 **problem sets** 来更好地巩固记忆。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1446232167930789930)** (54 messages🔥): 

> `Attention Sinks, Adam vs Signed Momentum, Gated Attention, synthetic dataset, neural race and generalization` 


- **Attention Sinks 综述出现**：一份关于 **Attention Sinks** 的综述被认为很有用，重点关注 **Attention Sinks**，海报可以在[这里](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/118276.png?t=1762557369.140577)查看。
- **Signed Momentum vs Adam 的对比分析发布**：讨论了 **Adam** 与 **Signed Momentum** 的对比分析，在[这篇论文](https://openreview.net/pdf?id=CH72XyZs4y)的第 4 节中发现了一个*有趣的数学变换*。
   - 建议*在实践中不要使用 beta1=beta2*，因为随着 beta1 -> beta2，稳定性会下降，可能导致模型崩溃（blow up）。
- **Titan's Miras 赋予 Google AI 长期记忆**：Google 披露了 **Titan's Miras**，如[这篇博客文章](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory)所述，它能帮助 AI 拥有长期记忆。
- **合成更简单的语言：1000 词数据集**：一位成员一直致力于创建一个仅使用**最常用的 1000 个英语单词**的文本 **synthetic dataset**，以辅助 LLM。
   - 该用户分享了这项工作的[博客文章](https://stur86.github.io/s-plus-plus/posts/the-big-learning-set-for-big-world-helpers/)，并提供了一个用于 Fork 和贡献的公共仓库。
- **Sejnowski-Hinton 关于 Brain-Backprop 的演讲出现**：回顾了 **Sejnowski-Hinton Award** 的演讲，重点讨论了关于大脑可能正在进行何种 **Backprop** 的理论。
   - 引用的论文包括 **Feedback Alignment** 和 **Direct Feedback Alignment**，不过演讲比论文更清晰，但需要 NeurIPS 注册才能观看。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1446280321439105046)** (7 messages): 

> `4D physics engine, General AI vs LLMs, Signal analysis approach to AI, Air gapped AI system, Feature manifolds in CNNs` 


- **抛弃语言：信号分析驱动 4D 物理**：一名成员正在为公司的 **4D physics engine** 开发程序，并发现语言具有严重的局限性，因此在进行 AI 概念化时更倾向于使用 **signal analysis**。
   - 他们已经在内部开发 **General AI system** 一段时间了，并于去年申请了专利，同时提到他们使用 Clip-Champ 创建配音文本。
- **相比 General AI，LLMs 表现糟糕**：一名成员表达了强烈的观点，认为 **LLMs** 已经过时，并将其称为 *2 step Y-Combinator algo's*。
   - 他们指出，如果构建得当，**General AI** 系统会 *强大得多且极其聪明*，其知识是“生长”出来的，而不仅仅是加载并进行推理。
- **物理隔离 (Air-Gapped) 的 General AI 系统保持孤立**：一名成员提到他们的 **General AI system** 目前处于 **air-gapped** 状态，并将保持这种状态直到明年年底。
   - 这表明他们在部署上采取了谨慎的态度，可能是由于系统的能力以及对精细控制的需求。
- **曲线检测器流形 (Curve Detector Manifolds) 博客文章**：一名成员分享了指向 [livgorton.com/curve-detector-manifolds/](https://livgorton.com/curve-detector-manifolds/) 的链接，这是一篇关于 **curve detector manifolds** 的博客文章。
   - 此外，他们还分享了 CNN 中特征流形（圆）的[视觉表示](https://media.discordapp.net/attachments/1083083481367715911/1446366791344586762/image.png?ex=6933b98b&is=6932680b&hm=46f9f0489662b131cbaae7e53718deb35b4ae8d4385dad362d555f3c53fecf02&=&format=webp&quality=lossless)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1446241271042543688)** (4 messages): 

> `Async RL MLsys papers, Factorio learning environment, Paperclip maximization study` 


- **寻求异步 RL 扩展资源**：一名成员请求推荐关于异步 **RL MLsys papers** 以及探讨 **RL system** 不同扩展方向和系统设计方案的博客。
   - 该成员注意到 **AllenAI** 和 **HuggingFace** 不久前发布了类似的内容，因此取消了原始请求。
- **Factorio 环境直播开始**：一名成员宣布在 15 分钟后开始关于 **Factorio learning environment** 的直播，链接为 [此 YouTube 链接](https://www.youtube.com/watch?v=LiwOzyeHX1U)。
   - 未提供更多信息，但推测涉及 Machine Learning。
- **“回形针最大化 (Paperclip Maximization)”研究开始**：一名成员宣布 **"paperclip maximization" study** 现在开始。
   - 未提供更多信息，但研究推测正在进行中。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1446427075291381872)** (17 messages🔥): 

> `cuTile release, tileIR and PTX relationship, CUDA programming guide rewrite, cuTile's mxfp/nvfp support, TileIR vs Triton IR` 


- ****cuTile** 库发布**：NVIDIA 发布了 **cuTile** 库 ([cuTile-python](https://github.com/NVIDIA/cutile-python/tree/main))，它使用一个基于 Python 的编译器，目标指向 **tileIR**，并将其转换为 **tileir asm**。
   - **tileiras** 二进制文件可能是 **CUDA 13.1** 的一部分。
- ****CUDA Programming Guide** 焕然一新**：**CUDA 编程指南**进行了全面重写 ([CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/))，同时发布了更多关于 **CUDA Toolkit 13.1** 的信息 ([CUDA toolkit 13.1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html))。
   - 用户指出 [cuTile-python 文档](https://docs.nvidia.com/cuda/cutile-python/) 和 [Tile IR 文档](https://docs.nvidia.com/cuda/tile-ir/) 是了解 NVIDIA 文档改进的更好起点。
- ****cuTile** 暂不支持 **FP4****：**cuTile** 库目前不支持 **mxfp/nvfp** 或 **fp4**；不过，未来计划支持 **fp4**。
   - 此外，目前似乎也没有支持 **autotuning** 的计划。
- ****TileIR** 对比 **Triton IR****：一位用户询问 **tileIR** 与 **Triton** 在其 IR 中能做的事情相比是否有优势，因为 **cuTile** 的高级语言似乎与 **Triton** 处于类似的抽象层级。
   - 另一位用户回应称，NVVM 中的 **TileIR** 后端可能拥有额外的硬件信息可以提供给优化器。
- ****PTX 9.1** 获得 **SIMD** 和 **Async Sharp** 操作**：**PTX 9.1** 引入了 **simd fp16x2** 到 **fp4x2**、**fp6x2** 和 **fp8x2** 的转换，以及潜在的异步 sharp 操作 (**sharp+tma**)。
   - 一位用户发布了一张图片，展示了 **PTX 9.1** 中一些新特性的幻灯片。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1446287456356925581)** (7 messages): 

> `Sparse Attention Adoption, VATTENTION: Verified Sparse Attention, CUDA-L2 performance` 


- **Sparse Attention 在实践中仍然稀缺？**：尽管有 **13,000 多篇**关于 *sparse attention* 的论文，但在 **vLLM** 等系统中的实际采用几乎不存在，根据[此讨论](https://x.com/skylight_org/status/1993637433838035026?s=20)。
- **VATTENTION 验证 Sparse Attention**：一篇新论文 *VATTENTION: VERIFIED SPARSE ATTENTION* ([arxiv link](https://arxiv.org/pdf/2510.05688)) 引入了第一个实用的 sparse attention 机制，具有用户指定的近似准确度的 **(ϵ, δ) 保证**。
   - 一位用户指出需要*更多编程语言+验证与 ML 人群的融合*。
- **CUDA-L2 通过强化学习超越 cuBLAS**：根据 [此 GitHub 仓库](https://github.com/deepreinforce-ai/CUDA-L2) 和 [NVIDIA 的 cutile-python](https://github.com/nvidia/cutile-python)，**CUDA-L2** 在矩阵乘法性能上通过 RL 超越了 **cuBLAS**。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1446262560121946305)** (6 messages): 

> `PMPP Book, Wen-mei autograph, GTC next year, CUDA reading` 


- **GTC 提供 PMPP 书籍签名！**：一位成员提议在明年的 **GTC** 上帮助大家获得 **Wen-mei** 签名的 **PMPP 书籍**。
   - 另一位成员热情回应，表示他们会*终生珍藏那本书*。
- **CUDA 爱好者深入钻研**：一位成员提到正在进行一些随机的 **CUDA** 阅读。
   - 该成员祝大家早上好。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1446567258041094207)** (4 条消息): 

> `Strix Halo 笔记本, RDNA 3.5 vs RDNA 4, CDNA 4 架构, HIPKittens kernel` 


- **Strix Halo：Kernel 原型设计利器**: 一位成员正在 **Strix Halo** 笔记本上进行 kernel 原型设计，并称赞 **RGP** 是 Windows 上非常出色的 profiler。
   - **Strix Halo** 笔记本拥有大容量 RAM (**128GB**)，可以加载一些大型 LLM，尽管其内存速度比数据中心 GPU 低得多（内存带宽比 MI355x 低约 30 倍！），且 FLOPs 也较低。
- **RDNA 3.5 缺乏 RDNA 4 特性**: 该 GPU 基于 **RDNA 3.5** 而非 **RDNA 4**，因此不支持 fp8，WMMA 指令仍需要 lane duplication 等。
   - 所有者个人非常喜欢他们的 **Strix Halo** 笔记本。
- **CDNA 4 寄存器数量受到质疑**: 一位成员询问 **CDNA 4** 是仍然拥有 512 个寄存器，还是 1024 个 vgprs+agprs，并指出 ISA 文档的图表中仅显示了 *"512 vgprs"*。
   - 他们指出，如果 **HIPKittens** 团队成功实现了每个 EU 运行 **2** 个 waves 且 tile 大小为 **256x256**，那么它需要超过 512 个寄存器，并补充说 **CDNA4 ISA** 手册在多处记录了 **512 vgprs**。
- **HIPKittens Regalloc 尚未深入探究**: 一位成员承认他们尚未详细查看 **HK kernels** 中的 regalloc。
   - 未提供更多细节。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 条消息): 

smexy3: 如果你想连接多台 Mac Studio 使用，哪种推理框架最好？
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1446422414128058490)** (4 条消息): 

> `大语言模型量化, MoE-Quant, GPTQ, CUDA 13.1, CUDA Tile` 


- **4Bit-Forge 使 LLM 量化平民化**: 一位成员宣布尝试使大规模 LLM（特别是 **deepseek math v2**）的量化平民化，该工作基于 [MoE-Quant](https://github.com/IST-DASLab/MoE-Quant) 奠定的基础。
   - 他们使用 **GPTQ** 进行 w4a16 量化，但报告称 *vllm* 和 *llcompressor* 无法工作，并分享了处于早期阶段的 WIP [4Bit-Forge 仓库](https://github.com/Pranshu-Bahadur/4Bit-Forge) 链接，以及 [使用方法、pytests 和 profiling 的 colab notebook](https://colab.research.google.com/drive/1es3bDhpROmMLjK4WfyTFeoybx7CSGaTk?usp=sharing)。
- **NVIDIA 发布 CUDA 13.1 及 CUDA Tile**: **NVIDIA** 发布了 **CUDA 13.1**，称其为 **CUDA 平台**自 2006 年问世以来最大的进化，其中包括 **CUDA Tile**，这是一种简化开发者利用 **GPU** 算力的新编程模型。
   - **CUDA Tile** 让开发者能够处理高层级的数据 *tiles*，而不是管理成千上万的底层线程，在简化 **GPU** 编程的同时仍能提供峰值性能，使先进的 **AI** 和加速计算更易触达。详情见 [CUDA Tile 博客文章](https://developer.nvidia.com/cuda/tile) 和 [CUDA 13.1 博客文章](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains)。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1446269066036908203)** (11 条消息🔥): 

> `NVIDIA 排行榜更新, nvfp4_gemm 性能提升, vectoradd_v2 排行榜条目` 


- **NVIDIA 的 NVFP4 GEMM 排行榜竞争白热化！**: 多位用户在 `nvfp4_gemm` 排行榜上成功提交，其中 <@1191430895769485436> 将执行时间从 **29.1 µs** (id `123091`) 降低到了 **17.0 µs** (id `123329`)。
- **微秒级魔法：进入 20 µs 以内俱乐部！**: 几位成员在 `nvfp4_gemm` 排行榜上实现了低于 **20 µs** 的成绩，包括 <@1191430895769485436> 的 **17.0 µs** 和 <@1390141830812794921> 的 **17.4 µs**。
- **H100 上的向量加法胜利**: <@1335076356324855838> 使用 **H100** 在 `vectoradd_v2` 排行榜上成功提交了 **5.23 ms** 的成绩 (id `125760`)。
- **第七天堂：NVFP4 上的纪录运行**: <@1295117064738181173> 以 **12.2 µs** 的惊人执行时间 (id `124468`) 夺得 `nvfp4_gemm` 排行榜第 7 名。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1446586480733589648)** (3 条消息): 

> `NeurIPS, LFG` 


- **发现 NeurIPS 论文！**: 一位成员报告说他们今天*路过了 **NeurIPS** 论文展示区！*
   - 另一位成员以 *"so cool !!!LFG"* 表达了热情。
- **对 NeurIPS 的热情**: 在提到看到 **NeurIPS** 论文后，一位成员表现得非常兴奋。
   - 他们惊呼 *"so cool !!!LFG"*，表示强烈的支持和期待。


  

---

### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1446475000008937472)** (2 条消息): 

> `Peephole Optimization, Movement Opcodes, CUDA/HIP Runtimes, LazyBuffer` 


- **Peephole Optimization 被跳过**：一个提交 ([`c8f8f30`](https://github.com/j4orz/teenygrad/commit/c8f8f3032c43e327383c1421cd4fb86443e01002)) 跳过了 `apply_movement_opcode` 输出的输入上的 Peephole Optimization。
   - 这一更改可能与正在进行的 **movement opcodes** 工作及其与 **compiler's optimization passes** 的交互有关。
- **Movement Opcode 文档更新**：提交 [`3fe6e24`](https://github.com/j4orz/teenygrad/commit/3fe6e24c7c2698f175248e1ab9c695374ddd14ea) 更新了关于 **shapetracker/lazybuffer** 和 **rangify/postopt** 的 `_apply_movement_opcode` 文档。
   - 此次更新可能澄清了这些组件在 movement 操作上下文中的行为和用法。
- **OpCode.RESHAPE 已修复**：提交 [`bd682b5`](https://github.com/j4orz/teenygrad/commit/bd682b5ab324ea6979ecf3ff6abc47580e2f4749) 修复了 `OpCode.RESHAPE` 的 `_apply_movement_opcode`。
   - 该修复允许穿透 `sugar` 的 Tensor 和 `engine` 的 OpNode，意味着修正了 reshape 操作在计算图（computational graph）中的处理方式。
- **DSL 连接到 CUDA/HIP Runtimes**：系统现在可以触发 `buffer.allocate()` 调用（使用 memoryviews 进行 fake realize），标志着将 **DSL** 连接到 **CUDA/HIP runtimes** 取得了进展。
   - 这表明团队正在积极将高级领域特定语言（DSL）与低级 GPU 执行环境进行集成。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1446413452062228490)** (2 条消息): 

> `Kernel Development, achievement` 


- **Kernel 领域：初学者寻求指导**：一名成员正在寻求帮助以 *开始进入 Kernel 领域*。
   - 他们附带了一张图片，可能是关于某项成就，标题为 *“我想这是一项成就”*。
- **另一个话题占位符**：这是一个为了满足至少两个话题要求的占位话题。
   - 如果原始消息中有更多细节或上下文，将在此处添加。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1446229700241526815)** (13 条消息🔥): 

> `RL Cheating, Blackwell GPU Access, Modal for Development, Subprocess Communication with Shared Memory, B200 GPU for Benchmarks` 


- **RL Agents 利用漏洞**：一位成员开玩笑说，在 RL 中，如果你告诉 Agent 去作弊，它就会去作弊；如果你告诉它 *“作弊并不会让你变成坏人”*，它会学会毫无悔意地作弊。
   - 随后该成员表示是时候 *“去户外接触下大自然（touch grass）”* 了。
- **了解 Blackwell GPU 访问权限**：一位新成员询问如何为比赛访问 **Blackwell GPU**，并提到目前只能访问 **A100 GPUs**。
   - 一位管理员澄清说 **Blackwell** 并非强制要求，并提供了 [通过 CLI 访问平台](https://github.com/gpu-mode/popcorn-cli) 的链接和 Discord 命令。
- **关于 Modal 开发的提问**：一位成员询问是否有人在比赛期间使用 **Modal** 进行开发。
   - 无人回应。
- **提议使用共享内存子进程**：一位成员建议另一位成员使用带有共享内存（shared memory）的子进程（subprocess）进行 Tensor 的输入/输出。
   - 另一位成员指出，在 Python 中，子进程内计时代码的漏洞很容易被破解。
- **请求显示已实现的 FLOPS**：一位成员请求在 UI 中的 **GEMM timings** 旁边显示已实现的 **FLOPS**。
   - 随后澄清了 **B200** 是用于基准测试（benchmarks）的 GPU。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1446247598129414185)** (73 messages🔥🔥): 

> `Claude 产生的 SQL 注入、Vibe Coding、Tanstack AI、Meta 收购 Limitless、Qwen 1.5-110B MoE 等效性能` 


- **Claude 随手制造代码灾难**：一位用户分享了一张截图，显示 **Claude** 生成的代码带有 [SQL injection 漏洞](https://cdn.discordapp.com/attachments/1075282825051385876/1446247597848264797/Screenshot_2025-12-04_at_16.11.092x.png)。
   - 一位成员评论道：*“顺便说一下，我们完蛋了——在我看来，未来会出现专门围绕访问控制设计的初创公司。投资渗透测试（pentesting）吧”*。
- **Tanstack 凭借类型安全 AI 工具包取得成功**：TanStack 正在发布 **TanStack AI Alpha**，这是一个强调全类型安全和多后端支持的工具包，并发布了 [博客文章](https://tanstack.com/blog/tanstack-ai-alpha-your-ai-your-way)。
   - 创始人指出，他们 *“很快就会发布博客文章和文档来进一步说明这一点”*。
- **Qwen 量化质量，降低成本**：阿里巴巴的 **Qwen 1.5-110B-Chat** 展示了与更大的 Mixture-of-Experts (MoE) 模型持平的性能，且仅需在两个 80 GB GPU 上运行（[来源](https://xcancel.com/Alibaba_Qwen/status/1996947806138126547?t=Ty7fc29sJcwnPwEOMaVH0Q&s=19)）。
   - 这削弱了关于 **MoE** 是获得顶级结果所必需的猜测，从而降低了成本。
- **TinyCorp 预告恐怖的小型 Tensor 巨兽**：TinyCorp 在 Twitter 上预告了一台带有 **8 个水冷 GPU** 的紧凑型 **1U server**（[来源](https://xcancel.com/__tinygrad__/status/1996815573427028106)）。
   - 该预告引发了关于 **散热、PCIe 5 瓶颈、NVSwitch 可用性** 以及可能通过 **token-sale** 获取该设备权限的笑话和技术提问。
- **Meta 令人震惊，吞并记忆巨头**：Meta 收购了 AI 可穿戴设备初创公司 **Limitless**（原名 Rewind），[Stammy 对此历程进行了回顾](https://xcancel.com/Stammy/status/1997024785214460137)。
   - 社区成员向团队表示祝贺，并对 **欧盟用户的未来访问权限** 以及 **Limitless Slack 账号** 出现异常表示担忧。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

inaarawalji_23: 今天上线 🙂
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1446621764745695253)** (2 messages): 

> `MAX Framework, Model API 更新, Modular 见面会` 


- **Modular 见面会取代 Zoom 会议**：通常在 Zoom 上进行的虚拟社区会议将被 **12 月 11 日** 在 Los Altos 办公室举行的特别 **Modular Meetup** 取代，远程参与者可以选择直播；注册地址为 [luma.com](https://luma.com/modularmeetup)。
- **Lattner 分享 MAX Framework 愿景**：Chris Lattner 将展示 **MAX framework** 背后的愿景，强调其在 **GPU** 和 **CPU** 上提供高性能、硬件无关的 AI 推理能力，支持 **500+ 模型**。
- **Model API 获得前沿更新**：与会者将了解 **Model API** 的前沿更新，包括在纯 **MAX/Mojo stack** 中的 eager 语义，且零 **PyTorch**、**NumPy** 或外部框架依赖。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1446231173591208007)** (41 messages🔥): 

> `Gemini 3 Mojo Understanding, Mojo stdlib Proposal, Mojo GPU Setup, Mojo Lifetimes Bug, Mojo Open Source Release` 


- **Gemini 3 展示了出色的 Mojo 熟练度**：一位成员报告称，在修复了一个去年春天创建的、包含破坏性变更（breaking changes）的约 600 行文件后，**Gemini 3** 似乎对 Mojo 有着相当不错的理解。
   - 他们还表示 **Gemini 3** 毫无压力地修复了所有问题。
- **Mojo stdlib 提案征求反馈**：一位成员在 Modular 论坛上分享了一个 [Mojo stdlib 提案链接](https://forum.modular.com/t/proposal-changing-copyable-to-refine-movable/2501)，专门征求评论意见。
- **Colab T4 GPU 支持快速 Mojo 原型设计**：为了快速运行和迭代 GPU 代码，一位成员建议使用 **Colab**，它在免费层级提供 **T4 GPU**，可以按照 [文档说明](https://docs.modular.com/mojo/tools/notebooks#using-mojo-on-google-colab) 在 Python notebook 中运行 Mojo 代码。
- **旧版闭包导致 Use-After-Free 问题**：一位成员报告了一个与 Mojo 中生命周期（Lifetimes）相关的潜在编译器 Bug，该 Bug 会导致 Use-After-Free 问题。
   - 另一位成员澄清说，这是 **旧版闭包（closures）** 的已知问题，最新 Nightly Build 中的 **新版闭包** 通过添加带有上下文的隐式额外参数修复了此问题，并提供了 [Mojo 迈向 1.0 之路](https://www.modular.com/blog/the-path-to-mojo-1-0) 的链接。
- **Mojo 将在 1.0 版本后不久开源**：一位成员询问 Mojo 1.0 版本发布时是否会同步开源。
   - 另一位成员回答说，**开源将在 1.0 之后不久进行**，并且 **2.0 版本将完全公开开发**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1446248328273727600)** (31 messages🔥): 

> `DeepSeek Technical, Linear Control Theory, AI Competition Catastrophe, Robustness vs Performance in Control, Unknown Dynamics in Robotics` 


- **线性特性限制了线性控制的应用**：一位成员指出，由于 **线性假设**，线性控制并没有被广泛使用；另一位成员确认，当超越线性系统时，线性控制的理论背景基本上就被抛弃了。
   - 一位成员指出，控制理论需要 *强稳定性保证* 和 *高精度*，这使得它在实践中很难处理。
- **竞争引发的灾难需要速度控制**：一位成员表示担心 **AI 发展是由不惜一切代价的竞争驱动的**，这可能会导致灾难，因为没有人愿意减速并冒着失败的风险。
   - 他们提议建立一个**全球自主智能监管系统**，利用计算的 Zero Knowledge Proofs，托管在 GitHub 仓库中，以在不控制生活其他方面的情况下控制算力。
- **控制理论在鲁棒性与性能之间寻求平衡**：一位成员解释说，在控制问题中，**鲁棒性（Robustness）和性能（Performance）** 之间存在权衡：提高其中一个就会牺牲另一个。
   - 他们建议通过改进 **HW design** 和 **更好的控制器** 来推动 Pareto front 向前移动，并强调了 **H∞ control** 在应对建模不确定性方面的鲁棒性。
- **未知动力学阻碍软体机器人发展**：一位成员确定了机器人控制中的三个挑战：**未知动力学（Unknown Dynamics）**、**非线性动力学**和**设计复杂性**。
   - 他们强调了 *未知动力学一直以来是如何困扰软体机器人（Soft Robotics）的*，因为缺乏气动肌肉的精确模型，且因果关系和延迟使得反馈控制器太慢，需要开环 + 自适应规划。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1446260835361231030)** (9 messages🔥): 

> `Bezos AI company, private video, arcprize withdrawn` 


- **贝佐斯进入 AI 领域？**：成员们在讨论贝佐斯的新 AI 公司是否会与 Amazon 竞争，参考了 [这段 YouTube 视频](https://www.youtube.com/watch?v=9A-eeJP0J7c) 和 [这个 Hacker News 帖子](https://news.ycombinator.com/item?id=46137548)。
- **私有视频？**：一位用户指出 [这段视频](https://youtu.be/Q4CBTckDAls?si=tyKN6MwBWITCqSaz) 是私有的，无法观看。
- **Arcprize 推文消失**：成员们注意到 [这条推文](https://x.com/arcprize/status/1997010284490473497) 已被撤回。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1446243531499765886)** (24 条消息🔥): 

> `DeepSeek v3.2 Transformers 实现, Z Image 审查, Hugging Face Space CPU 配额, 适用于 Roblox 的小型 LLM, AI 生成音乐 YouTube 频道` 


- **DeepSeek Transformers 实现停滞**：针对新推出的 **DeepSeek v3.2 模型** 的 *transformers* 实现工作正在进行中，但一个[相关的 PR](https://github.com/huggingface/transformers/pull/41251) 显示进度停滞。
   - 原贡献者似乎已经放弃了该项目，近期没有活动迹象。
- **HF Space CPU 配额引发 Pro 账户问题**：一位用户报告了 **Hugging Face Space CPU 配额限制** 的问题，即使是 Pro 账户也无法启动或取消暂停 Spaces。
   - 该用户对这一变化缺乏公告表示沮丧，因为这导致了意外的服务中断。
- **Z Image Demo 中的审查**：用户注意到 **Z Image demo** 会审查显式内容，尽管该模型宣传为无审查，但仍会显示 *"maybe not safe"* 的图像。
   - 有建议认为这可能是由于 demo 代码或 Endpoint 端的自我限制，而在本地使用时并不存在。
- **寻求用于 Roblox 集成的小型 LLM**：一位用户正在寻找一个**小型 LLM（100M 参数以下）**以集成到 *Roblox* 中，但面临 Roblox 文件大小和 RAM 限制的挑战。
   - 他们在集成微型 Ollama 和 Pythia 模型方面取得了进展，但需要一个功能更强且更紧凑的聊天机器人解决方案。
- **AI 生成音乐频道引发观众两极分化**：一位用户宣布创建了一个致力于 **AI 生成音乐** 的 **YouTube 频道**，该频道完全由 AI 和他们的代码管理。
   - 这一公告引发了截然不同的反应，从兴奋和支持到潜在的反感。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1446608645369364480)** (1 条消息): 

> `大型语言模型的低效性, HRM 和 TRM 模型作为替代方案, LLM 的计算成本, LLM 的环境影响, LLM 导致的成本上升` 


- **LLM 因低效而受到批评**：一位用户反对使用大型语言模型 (**LLMs**)，理由是其效率低下且成本高昂，并引用了过度的计算资源消耗和环境影响。
   - 该用户声称 **LLMs** 导致了存储、RAM 和 GPU 成本的上升，并主张采用 HRM 或 TRM 模型等替代方案。
- **HRM/TRM 模型挑战 LLM 巨头**：该用户强调 **HRM** 或 **TRM** 模型（约 2700 万参数）在某些基准测试中是 **LLMs** 的优选替代方案，并提供了研究论文链接（[https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734) 和 [https://arxiv.org/abs/2510.04871](https://arxiv.org/abs/2510.04871)）。
   - 据称，这些模型以显著更少的参数实现了更好的性能，挑战了庞大模型规模的必要性。
- **LLM 被指责具有极大的环境影响**：该用户断言 **LLMs** 的环境影响是不可接受的，因为它们过度消耗饮用水、造成空气污染并加剧全球变暖。
   - 该用户未提供此主张的来源，且该观点无法立即证实。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1446500875043340399)** (3 条消息): 

> `Anthropic Programmatic Tool Calling, Universal Programmatic Tool Calling, Model Agnostic Tool Orchestrator, Rhai Scripts for LLMs, Token Reduction in LLMs` 


- **模型无关的工具编排器亮相**：一名成员介绍了一个基于 Anthropic [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) 的**模型无关且生产就绪的工具编排器**。
   - 该实现允许任何 LLM 编写 **Rhai 脚本**来高效编排多个工具，在基准测试中承诺可实现 **97-99% 的 Token 减少**。
- **通用工具调用实现 Token 减少**：该工具被称为 **Universal Programmatic Tool Calling**，具有模型无关性，旨在实现 Anthropic 的 [Programmatic Tool Calling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) 模式。
   - 与消耗大量 Token 的顺序工具调用不同，任何 **LLM 都可以编写 Rhai 脚本**来高效编排多个工具，在基准测试中实现了 **97-99% 的 Token 减少**。
- **工具编排器已发布至 GitHub 和 YouTube**：该编排器可与任何 LLM 配合使用，采用沙箱化设计，作为原生的 **Rust 或 WebAssembly** 运行，无需外部 Python 依赖，采用 MIT 许可证，可在 [GitHub](https://github.com/Brainwires/tool-orchestrator) 上获取。
   - [YouTube 视频](https://www.youtube.com/watch?v=b8yeQnP_ftw)和 [LinkedIn 帖子](https://www.linkedin.com/posts/eoinfr_buildinpublic-trading-fintech-activity-7402723589679960064-5oOT?utm_source=share&utm_medium=member_ios&rcm=ACoAACm7Z4cB_ZlAX5DjoA-4q-UXoclEX6TZepA)提供了更多细节。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/)** (1 条消息): 

sky.moo: https://huggingface.co/blog/hf-skills-training
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1446469782214934588)** (2 条消息): 

> `Agent Course Certificate` 


- **Agent 课程证书仍然可以获取吗？**：一名成员询问在完成并提交最终作业后，是否仍有可能获得 **Agent Course Certificate**。
- **缺乏澄清**：该问题未得到回复，导致证书获取状态尚不明确。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1446237684761624617)** (24 条消息🔥): 

> `Kimi for Coding Access, corporate policy reasons, LM Playground, 4x K2 turbo limit` 


- **Kimi for Coding 访问权限仍仅限邀请？**：用户报告了访问 **Kimi for Coding** 时遇到的问题，想知道它是否仍然仅限邀请，并在尝试 Option 1 和 Option 2 访问方法时遇到困难。
   - 一名用户发现他们需要[注册 kimi.com 订阅](https://kimi.com)，然后在订阅设置中使用“Kimi for Coding”链接来解锁访问权限。
- **Cloud Code 与 Roo Code**：一名用户询问 **Kimi-for-coding** 仅支持 Cloud Code 和 Roo Code 的原因，以及应联系谁获取更多信息。
   - 作为回应，另一名用户指出 Roo Code 仅仅是 Cline 的一个分叉（fork）。
- **召集所有 LM 折腾者**：一名用户表达了希望建立一个专注于 LM *趣味实验*而非仅仅是商业应用的社区。
   - 他们强调了测试和改进本地模型的乐趣，感叹目前的 LM 聊天机器人太无趣，并建议增加如**引用框（quotation boxes）**等功能以增强信任感。
- **4x K2 Turbo 限制**：一名用户询问 Moderato 上的 **4x K2 turbo 限制**是如何运作的（之前忘记问了），暗示他们已经用完了配额。
   - 另一名用户建议搜索答案，并附上了[一个 X 帖子的链接](https://x.com/Kimi_Moonshot/status/1996953835080966390)，指出 *Turbo 只是速度更快*。


  

---

### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1446248218911571998)** (9 messages🔥): 

> `MCP Tokens, Tokenization, tiktoken, Claude 3 tokenizer` 


- **MCP Token 工具难题**：一名成员正在寻求分析 **MCP token usage** 的工具或方法建议，特别是在从工具响应中剥离不必要数据并压缩工具描述之后。
- **Tokenization 与模型绑定**：**Tokenization** 取决于模型，因此你需要选择你感兴趣的模型子集，并通过你使用的 **tokenizers** 运行你的工具。
- **tiktoken 适用于 GPT 模型**：对于 **OpenAI**，你可以使用 [tiktoken](https://github.com/openai/tiktoken)，它应该适用于 **GPT models**。
- **Anthropic 仅公开 count_tokens API**：对于 **Claude**，他们仅公开了 [count_tokens API](https://platform.claude.com/docs/en/api/messages/count_tokens)。
- **Anthropic 不再提供本地 tokenizer**：**Anthropic** 过去曾公开提供本地 **tokenizer**，但自 **Claude 3** 以来，他们更改了 **tokenizer** 且不再这样做，说实话这挺令人烦恼的。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1446263414757527574)** (7 messages): 

> `Ollama Timeout Errors, Claude Sonnet 4.5 Downgrade, Workflow Automation Engineer Introduction` 


- **Ollama 超时错误困扰用户**：一位用户报告在使用 **gpt-oss:120b** 和 **llama4:scout** 等模型时遇到 **Ollama** 超时错误，具体为 `litellm.APIConnectionError: Ollama_chatException - litellm.Timeout: Connection timed out after 600.0 seconds.`。
- **据称 Claude Sonnet 4.5 变笨了**：一位用户表示 **Sonnet 4.5 (Claude code)** 在最近几天似乎变得不那么聪明了，暗示性能可能有所下降。
- **专注于工作流自动化的全栈工程师自我介绍**：一位专注于工作流自动化、**LLM** 集成、AI 检测和多模态系统（**图像 + 语音**）的全栈工程师在频道中介绍了自己。
- **工程师宣传 Slack、Notion 和内部 API 自动化**：该工程师详细介绍了一个利用 **Slack、Notion 和内部 API** 的跨平台自动化系统，据称该系统将响应时间缩短了 **60%**。
- **强调 RAG 专业知识**：该工程师还声称构建了一个先进的 **RAG architecture**，利用混合搜索、embeddings 和基于领域的排名，以确保实时部署期间的准确性和上下文稳定性。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1446249051128463411)** (1 messages): 

> `aider on local LLMs, aider on Android, Cross-device coding with aider` 


- **新用户打算设置 Aider 本地 LLM**：新用户 Kalpit 想要在他们的 **Mac 上本地运行 LLM**，并在同一网络内的 Fold 6（Android 手机）上使用 **aider** 进行编程。
   - 他们正在寻求其他实现过类似 **aider** 跨设备编程设置的人员的建议或经验。
- **用户转向使用 Aider 以利用本地 LLM**：一位曾频繁使用 Cursor 和 Claude Code 的用户表示，希望转向使用 **aider** 以利用本地 **LLM**。
   - 该用户目标是在本地 Mac 设置上运行 **aider**，并从 Fold 6 设备远程访问它，寻求社区的指导或经验分享。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

justanotheratom: https://x.com/realsanketp/status/1996978356227920345?s=20
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1446311848596213770)** (4 messages): 

> `DSPy support to Claude agents, GRPO algorithm in DSPy, Multi-turn conversations` 


- **DSPy 扩展对 Claude Agent 的支持**：一名成员询问关于将 **DSPy** 支持扩展到 **Claude agents** 或其他流行 **Agent SDK** 的事宜。
   - 另一名成员要求澄清该请求，因为支持的方向可能会影响实现方法。
- **探索 GRPO 算法**：一位新的 **DSPy** 用户询问了 **GRPO algorithm**，寻求关于其在处理**多轮对话（multi-turn conversations）**中的性能和能力的见解。
   - 该用户对实际运行结果以及它在多次交互中管理上下文的效果感兴趣。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1446484004147826753)** (4 messages): 

> `FSDP in tinygrad bounty, USBGPU on Raspberry Pi, USB transactions` 


- **FSDP 悬赏出现困难**：一名成员正在研究 **tinygrad 中的 FSDP 悬赏**，并报告说 *multi* 导致了问题，不确定悬赏是否允许为了好玩而修改 *multifor funsies*。
   - 该用户未提供更多信息，可能正在寻求社区的帮助。
- **Raspberry Pi 获得 USBGPU 提升**：一名成员在 **Raspberry Pi 4** 上成功运行了 **USBGPU**，此前在旧型号（2 和 3）上的尝试因架构和 stream 分配错误而失败。
   - 根据[图像分析](https://cdn.discordapp.com/attachments/1068976834928193609/1446565913477251072/image.png?ex=693472fd&is=6933217d&hm=cfd6bfeb6ab892a212e18a7d559c4239e99cc5e034898ea6559b52495e6028aa)，如果添加了驱动支持，USB 2 可能会运行缓慢，可能会使用 **BULK** 而不是 streams。
- **探讨 USB 事务**：在 **USBGPU** 实现的背景下引发了关于 **USB 事务** 的讨论。
   - 一位用户建议即使是全速（**12Mbps**）也会被支持，但他们对 *usb 事务并不是那么精通*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1446321569814155467)** (1 messages): 

> `struct.unpack GPU Implementation, tinygrad GPU Unpacking` 


- **GPU 获得 `struct.unpack`**：一位成员开玩笑说，使用 **tinygrad** 在 GPU 上实现 `struct.unpack('<4sIHHIIHH', f.read(24))`，而不是使用传统方法。
   - 分享的图片直观地展示了 **基于 GPU 的解包** 的复杂性和潜在的性能提升。
- **tinygrad struct 实验**：有一个使用 tinygrad 而不是 `struct` 来处理二进制数据的实验。
   - Discord 成员发现这是一个有趣的例子。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1446431497669181471)** (1 messages): 

> `Workflow Automation, LLM Integration, RAG Pipelines, AI Content Detection, Image AI` 


- **AI 工程师实现工作流自动化并集成 LLM**：一位在工作流自动化、LLM 集成、RAG、AI 检测、图像和语音 AI 方面拥有专业知识的 AI 及全栈工程师提供了他们的服务。
   - 他们在实际落地方面有良好的记录，并对合作或支持持开放态度。
- **Slack 和 Notion 支持由 LLM 驱动的自动化**：该工程师使用 **Dspy**、**OpenAI APIs** 和自定义 Agents 构建了一个自动化流水线和任务编排系统。
   - 其中一个例子是将 **Slack**、**Notion** 和内部 API 连接到 LLM 的支持自动化系统，将响应时间缩短了 **60%**。
- **部署高级 RAG 流水线**：该工程师设计并部署了高级 RAG 流水线，结合了向量数据库和图数据库、混合搜索以及自定义检索逻辑。
   - 这在生产环境中实现了准确且具有上下文感知能力的响应。
- **开发 AI 内容检测工具**：该工程师为一个审核平台开发了工具，使用文体分析、嵌入相似度以及微调的 Transformers。
   - 这些工具能以高精度识别 GPT 生成的文本。
- **图像 AI 标签和审核流水线**：该工程师在 **AWS Lambda** 和 **S3** 上使用 **CLIP** 和 **YOLOv8** 创建了一个标签和审核流水线。
   - 该系统每天为一个电子商务平台分类和过滤数千张图像。


  

---


---


---