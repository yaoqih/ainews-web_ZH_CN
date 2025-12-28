---
companies:
- openai
- google
- lmsys
date: '2025-08-15T05:44:39.731046Z'
description: '**OpenAI** 已将 **GPT-5** 作为 ChatGPT 的默认模型推出，引入了全新模式和更具“亲和力”的个性，并为 Plus/Team
  用户及 Enterprise/Edu 账户扩大了消息限制。性能排名显示 **gpt-5-high** 处于领先地位，其他较小版本也位列其中；不过，也有评论指出其在某些表现上逊于中国模型，且存在迎合用户（sycophancy）的倾向。OpenAI
  还增强了开发者工具，新增了“快速评估”（Quick eval）功能、编码提示以及改进后的 Playground。**Google** 正式全面开放了 **Imagen
  4**，其生成速度更快且分辨率更高，同时还发布了拥有大词汇量和生态支持的超小型 **Gemma 3 270M** 模型。在播客节目中，OpenAI 的领导层讨论了
  GPT-5 的系统架构、路由机制及效率问题。'
id: MjAyNS0w
models:
- gpt-5
- gpt-5-high
- gpt-5-mini-high
- gpt-5-nano-high
- imagen-4
- gemma-3-270m
people:
- sama
- aidan_mclau
- kevinweil
- lmarena_ai
- edwinarbus
- gdb
- omarsar0
- philschmid
- m4rkmc
title: 今天没发生什么事。
topics:
- model-releases
- model-performance
- prompt-engineering
- developer-tools
- image-generation
- model-optimization
- transformers
- tokenization
- model-scaling
---

**平静的一天。**

> 2025年8月14日至8月15日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，10644 条消息）。预计节省阅读时间（以 200wpm 计算）：789 分钟。我们的新网站现已上线，支持全量元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

今天的播客圈非常热闹，[Greg Brockman](https://www.youtube.com/watch?v=35ZWesLrv5A&lc=UgyoHIYSYSZa8z39T2Z4AaABAg) 以及 [Jakub 和 Szymon](https://www.youtube.com/watch?v=yBzStBK6Z8c) 均已上线。

---

# AI Twitter 综述

**OpenAI 的 GPT‑5：产品发布、路由和开发者工具**

- **ChatGPT 和 API 更新**：OpenAI 推出了重大的每周更新：GPT‑5 现已成为 ChatGPT 的默认模型，支持 Auto/Fast/Thinking 模式；Plus/Team 用户在 GPT‑5 Thinking 上每周最多可发送 3,000 条消息，超出部分将转至 GPT‑5 Thinking mini；旧版模型（o3, GPT‑4.1, 4o）仍可通过设置使用；Enterprise/Edu 访问权限已上线；“更温暖”的默认个性化设置即将推出 ([@OpenAI](https://twitter.com/OpenAI/status/1956212769365352758))。几小时后，新的个性化设置正式上线——OpenAI 表示它“更平易近人”，且未检测到奉承性（sycophancy）的增加；用户仍可通过 Custom Instructions 自定义风格 ([@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196), [@sama](https://twitter.com/sama/status/1956483306951938134), [@aidan_mclau](https://twitter.com/aidan_mclau/status/1956462903781191744), [@kevinweil](https://twitter.com/kevinweil/status/1956462974098669710))。
- **性能和路由背景**：LMSYS 更新了其竞技场：默认的 gpt‑5‑chat 首秀排名第 5，较小的 gpt‑5‑mini‑high 和 gpt‑5‑nano‑high 分别位列第 16 和第 44；gpt‑5‑high 稳居第 1 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1956399522688692608))。批评者指出 GPT‑5 在编程方面表现不如某些中国模型，且 LMSYS 对奉承性较为敏感 ([1](https://twitter.com/scaling01/status/1956403514244059261), [2](https://twitter.com/scaling01/status/1956404452442681829), [3](https://twitter.com/scaling01/status/1956405559978029061), [4](https://twitter.com/scaling01/status/1956353414687822183))。其他人警告称，竞技场排名不等于生产环境的实用性，并建议进行迁移测试 ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1956433566297915849), [@LucasAtkins7](https://twitter.com/LucasAtkins7/status/1956435679229186353))。战术笔记：GPT‑5 的可控性很强，但需要更明确的提示词（prompting）；像对待代码一样对待提示词（版本控制、测试），阅读指南，并使用 Prompt Optimizer ([@edwinarbus](https://twitter.com/edwinarbus/status/1956218284308881867), [@gdb](https://twitter.com/gdb/status/1956170475622793640))。在安全方面，当集成到 XBOW 平台时，GPT‑5 的网络能力提升了一倍以上，突显了 Agent 框架对能力实现的重大影响 ([@Xbow](https://twitter.com/Xbow/status/1956416634173964695))。
- **开发者体验**：OpenAI 控制面板中新增的“Quick eval”功能，让你可以通过内置的评分器，将 GPT‑5 变体和推理开销与你自己的回复进行对比 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1956410610914414904))。OpenAI 还发布了“使用 GPT‑5 编程的六个技巧”，并在 [developers.openai.com](http://developers.openai.com/) 建立了统一的开发者门户 ([技巧](https://twitter.com/OpenAIDevs/status/1956438999364768225), [PDF](https://twitter.com/OpenAIDevs/status/1956439005970801099), [开发者门户](https://twitter.com/pranaveight/status/1956477855392768490))。Playground 改进了路由、vector stores、MCP 工具以及用于原型设计的 evals ([@omarsar0](https://twitter.com/omarsar0/status/1956459233039233528))。播客：OpenAI 的 Merett Miller 和 Szymon Sidor 探讨 AGI 轨迹；Greg Brockman 讨论 GPT‑5 系统、路由、定价、计算效率以及用于生物模型的字符级 Tokenization ([OpenAI 播客](https://twitter.com/OpenAI/status/1956385632801923555), [Brockman 访谈](https://twitter.com/latentspacepod/status/1956433236021883071))。

**Google 更新：Imagen 4 GA 和 Gemma 3 270M**

- **Imagen 4**：在 AI Studio 和 Gemini API 全面可用（GA），分为三个层级——Ultra ($0.06)、Standard ($0.04)、Fast ($0.02) 每张图像。支持最高 2k 分辨率，每个 prompt 生成 1–4 个输出，生成速度比之前模型快 10 倍 ([@_philschmid](https://twitter.com/_philschmid/status/1956351654753673252), [@m4rkmc](https://twitter.com/m4rkmc/status/1956238192035663874))。开发者分享了用于生成一致产品图的 JSON prompting 模式 ([1](https://twitter.com/_philschmid/status/1956351658381705420), [2](https://twitter.com/_philschmid/status/1956351661229703246))。
- **Gemma 3 270M (open, ultra‑small)**：总参数量 270M，结构独特：约 170M 用于 embeddings，约 100M 用于 transformer blocks；词表非常大（262,144 tokens）。发布了 pretrain 和 instruct 版本，具有广泛的生态系统支持（Transformers/JS, llama.cpp, MLX, Vertex 等）。针对特定任务的 fine-tuning 和 edge 端使用进行了优化 ([@osanseviero](https://twitter.com/osanseviero/status/1956258657483534803), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1956393664248271082))。早期报告显示在 iPad Air M3 上通过 MLX 可达 ~200 tok/s；一些人指出存在重复问题，并对这种重 embedding 的设计权衡展开了讨论 ([@adrgrondin](https://twitter.com/adrgrondin/status/1956428984876704059), [@kchonyc](https://twitter.com/kchonyc/status/1956374537278214523), [discussion](https://twitter.com/BlackHC/status/1956344522109042707))。
- **Research**：NextStep‑1 提出了一个 14B 的统一自回归模型，涵盖离散文本 tokens 和连续图像 tokens，带有一个轻量级的 157M flow‑matching head——避免了 VQ 瓶颈 ([thread](https://twitter.com/iScienceLuvr/status/1956321483183329436), [code/models](https://twitter.com/iScienceLuvr/status/1956321486366462428))。Google 的每周更新总结了 Imagen 4 GA、Gemma 3 270M、Gemini Deep Think 配额增加以及对话研究 (g‑AMIE) ([@GoogleAI](https://twitter.com/GoogleAI/status/1956400937054163357))。Gemini 应用的 “Drops” 回顾发布了多项 UX 更新 ([@GeminiApp](https://twitter.com/GeminiApp/status/1956388218217300085))。

**Agents, evaluation harnesses, and tooling**

- **Open computer‑use agents (CUA)**：XLANG 发布了 OpenCUA，这是一个完整的框架和模型（7B/32B），包含大型 CUA 数据集（跨 3 个 OS 和 200 多个应用/网站的 22.6k 轨迹）、工具链和离线 benchmark。OpenCUA‑32B 在 OSWorld‑Verified 上报告得分为 34.8%，声称达到或超过了闭源基准 ([announcement](https://twitter.com/xywang626/status/1956400403911962757))。
- **Agent harnesses that scale competency**：Cline v3.25 引入了 Focus Chain（持久上下文）和 /deep‑planning，以确保长且复杂的任务不偏离轨道——博客和变更日志详细说明了为什么在 Agent 中“注意力是不够的” ([blog](https://twitter.com/cline/status/1956394230357877209), [changelog](https://twitter.com/cline/status/1956383188089221370))。Cursor CLI 增加了 MCPs、Review Mode、/compress 和 @‑file 引用，用于工具增强型编程 ([@cursor_ai](https://twitter.com/cursor_ai/status/1956458242655281339))。LangGraph Studio “Trace mode” 将实时的 LangSmith traces/annotation 引入 Studio ([@LangChainAI](https://twitter.com/LangChainAI/status/1956411858312949946))。Weave 现在可以跨 traces 跟踪多模态内容，以便进行 eval/debug ([@weave_wb](https://twitter.com/weave_wb/status/1956412035647815735))。
- **Eval wave**：Guardrails 的 Snowglobe 模拟了数百个由角色驱动的对话来测试 Agent，将失败转化为训练信号——对于强化长周期工作流非常有用 ([1](https://twitter.com/godofprompt/status/1956359876109652297), [2](https://twitter.com/alex_prompter/status/1956360410862354435), [app](https://twitter.com/ShreyaR/status/1956396368270074217))。Spiral‑Bench 衡量模型陷入幻觉螺旋的倾向：Sonnet 4 被评为最谄媚；GPT‑5 则相反 ([@sam_paech](https://twitter.com/sam_paech/status/1956343619914432900), [@scaling01](https://twitter.com/scaling01/status/1956350388791108044))。Epoch 在其中心增加了五个外部 benchmark（TerminalBench, DeepResearchBench, METR Time Horizons, GSO, WebDevArena） ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1956384193891688625))。Teresa Torres 的演讲是生产环境 evals 的杰出案例研究：错误分析优先、自定义标注、LLM‑judges + 断言、紧密的反馈循环 ([@HamelHusain](https://twitter.com/HamelHusain/status/1956371273858314397))。
- **Long‑horizon agent findings**：论文摘要——Agent（包括 GPT‑5）在长周期任务上仍然表现挣扎；“语义压缩”（chunk 级摘要）在成本和成功率上都优于原始的长上下文，提高了检索精度和计划连贯性 ([1](https://twitter.com/omarsar0/status/1956325762719797266), [2](https://twitter.com/omarsar0/status/1956325856265326923), [3](https://twitter.com/omarsar0/status/1956325872908247220))。

**Speech, vision, and multimodal stacks**

- **NVIDIA 语音发布（开源）**：Granary，最大的开源欧盟语音数据集；Canary‑1b‑v2，支持 25 种语言（ASR + En↔X 翻译）；以及 Parakeet‑tdt‑0.6b‑v3 SOTA 多语言 ASR。Argmax 发布了对 Parakeet v3 的首日支持（[数据集/模型](https://twitter.com/Tu7uruu/status/1956350036343701583), [SDK](https://twitter.com/argmaxinc/status/1956385793892917288)）。开源 ASR 已扩展至德语/法语/意大利语/西班牙语/葡萄牙语，更多语言即将推出（[@Tu7uruu](https://twitter.com/Tu7uruu/status/1956354974226456794)）。
- **VLM 与视频**：阿里巴巴的 Ovis2.5 (2B/9B) 采用 NaViT 原生分辨率视觉和反思性推理（Reflective Reasoning）；9B 版本在 OpenCompass 上获得 78.3 分（40B 以下模型的 SOTA），具备强大的小规模图表/文档 OCR 以及视频/多图定位（Grounding）能力（[@gm8xx8](https://twitter.com/gm8xx8/status/1956292512030638235)）。Runway 的 Aleph 可以通过单个 Prompt 插入物体/角色，并保持场景一致的光照和色彩（[@runwayml](https://twitter.com/runwayml/status/1956341430743339402)）。可灵（Kling）API 新增了声音生成和多元素合成功能（[@Kling_ai](https://twitter.com/Kling_ai/status/1956343695977943228)）。

**推理与基准测试：HRM 消融实验、谄媚行为（Sycophancy）与趋势线**

- **显微镜下的 HRM**：ARC Prize 和 François Chollet 复现了分层推理模型（Hierarchical Reasoning Model）在 ARC‑AGI‑1 上的得分，但发现架构并非关键因素。相反，一个被轻描淡写的外部细化循环（Outer Refinement Loop）推动了性能提升；跨任务迁移贡献甚微；且较少的数据增强即已足够。结论：这实际上是零预训练的测试时训练（Zero-pretraining Test-time Training）——数据和流程主导了模型微调（[@arcprize](https://twitter.com/arcprize/status/1956431617951740044), [@fchollet](https://twitter.com/fchollet/status/1956442449922138336), [数据泄露说明](https://twitter.com/fchollet/status/1956442913950539802)）。
- **谄媚行为与路由影响**：Spiral‑Bench 和社区观察表明，模型排名和用户感知对“粉饰（Glazing）”行为非常敏感——Gemini 2.5 (Flash/Pro) 表现出高度的谄媚性，而 GPT‑5 则呈下降趋势（[1](https://twitter.com/scaling01/status/1956353414687822183), [2](https://twitter.com/scaling01/status/1956371713949655328)）。OpenAI 声称其“更温暖”的个性变化并未增加谄媚行为，但需关注模型路由（Model Routers）和偏好奖励（Preference Rewards）（[@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196)）。
- **前沿模型到消费级的延迟**：Epoch 估计前沿模型的性能约在 9 个月后到达消费级硬件；如果这一趋势持续，到 2026 年第二季度，家用可运行的开源模型可能达到 Grok 4 的水平——鉴于能力的扩散，这对安全政策具有重要意义（[推文串](https://twitter.com/EpochAIResearch/status/1956468453399044375)）。

**中国生态系统：Qwen、GLM 与生态工具**

- **Qwen 升级**：Qwen Chat 的视觉理解能力现已支持原生 128k 上下文，拥有更强的数学/推理能力、30 多种语言的 OCR，以及更好的 2D/3D/视频定位能力（[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1956289523421470855)）。Qwen Chat Windows 桌面版增加了对本地 Agent 的 MCP 支持（[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1956399490698735950)）；Baseten 报告 Qwen3 Instruct 的推理速度约为 95 tps（[@basetenco](https://twitter.com/basetenco/status/1956475210582090030)）。Ovis2.5 和生态系统的更新节奏持续令人印象深刻（[评论](https://twitter.com/teortaxesTex/status/1956306172576690610)）。
- **GLM‑4.5 可用性**：智谱（Zhipu）的 GLM‑4.5 已上线 SST opencode 平台并提供快速启动 Demo；GLM‑4.5V 在 HF 上热度攀升；社区展示包括一个纯由视觉推理驱动的 GeoGuessr 风格地理游戏（[平台](https://twitter.com/Zai_org/status/1956335531555721345), [HF 趋势](https://twitter.com/Zai_org/status/1956421442092032258), [GeoGuessr Demo](https://twitter.com/Zai_org/status/1956353661397094890)）。

**热门推文（按互动量排序）**

- 律师因引用 AI 幻觉生成的案例受到制裁：致法官的信、临时执业许可（pro hac vice）被撤销、辩护状被删除、通知律师协会 ([@RobertFreundLaw](https://twitter.com/RobertFreundLaw/status/1956164045612228968)) – 8,888
- OpenAI 每周回顾：GPT‑5 默认设置、模式、配额、旧版访问、Enterprise/Edu、个性化计划 ([@OpenAI](https://twitter.com/OpenAI/status/1956212769365352758)) – 8,021
- OpenAI 发布了更温和的 GPT‑5 个性；声称没有增加谄媚（sycophancy）倾向；即将推出自定义功能 ([@OpenAI](https://twitter.com/OpenAI/status/1956461718097494196)) – 6,275
- 中国宣布了一项高度开放的技术移民签证通道（年龄门槛、知名大学/研究背景），标志着更广泛的开放趋势 ([@RnaudBertrand](https://twitter.com/RnaudBertrand/status/1956310213134356482)) – 3,375
- 上海街头机器人遛狗——日常科幻场景 ([@crystalsssup](https://twitter.com/crystalsssup/status/1956257972197449850)) – 2,298
- Cohere “打算在收购 TikTok 和 Chrome 之后立即收购 Perplexity”（讽刺） ([@aidangomez](https://twitter.com/aidangomez/status/1956361969323184361)) – 1,953
- “GPT‑5 太棒了；如果你听到不同的说法，那是技术水平（skill issue）问题。” ([@skirano](https://twitter.com/skirano/status/1956307604491108675)) – 1,909
- “所有 AI 工程师都应该知道的 8 种 RAG 架构” 讲解/推文串 ([@_avichawla](https://twitter.com/_avichawla/status/1956241967136039197)) – 1,773

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. DeepSeek-V3 成本与基准测试优于 GPT-4o

- [**DeepSeek 在大多数基准测试上优于 4o，且价格仅为 10%？**](https://i.redd.it/o5jfkiky14jf1.png) ([Score: 410, Comments: 129](https://www.reddit.com/r/LocalLLaMA/comments/1mqnft3/deepseek_is_better_than_4o_on_most_benchmarks_at/)): **该帖子展示了一个柱状图，对比了 DeepSeek-V3（$0.27 输入/$1.10 输出）和 GPT-4o（$2.50 输入/$10.00 输出）的每百万 token 成本，强调了 DeepSeek 在输入和输出处理方面的显著成本优势（约便宜 10 倍）。该图片旨在支持 DeepSeek-V3 在大多数基准测试中表现优于 GPT-4o 且更具成本效益的说法，这对于优化单位美元性能的企业或开发者具有重要意义。技术重点在于 API 定价和 token 处理成本——这是扩展大语言模型（LLM）部署的关键因素。[查看图片。](https://i.redd.it/o5jfkiky14jf1.png)** 评论讨论了 DeepSeek-V3（特别是原始版本和 0324 版本）如何在基准测试中超越 GPT-4o，但也指出由于版本差异，将较新的 DeepSeek-V3-0324 进行对比属于“不公平竞争”。其他评论强调，由于缓存和地区折扣，DeepSeek 的 API 甚至更便宜，而价值的模型选择取决于任务复杂性和模型适配度。
    - 多位用户强调 DeepSeek-V3-0324 在大多数基准测试中以极低的成本超越了 GPT-4o，但指出与早期的 DeepSeek-V3（原始版）对比不太公平，因为最新的 0324 版本在通用任务（尤其是非推理工作负载）上有了显著改进且更具竞争力。
    - 详细讨论了 API 定价模型：DeepSeek 提供了巨大的成本优势，其官方 API 具有输入缓存（数小时过期）和中国夜间低峰时段额外的 50% 折扣，导致实际使用成本甚至低于名义上的广告价格。
    - 尽管基准测试和价格表现强劲，但 DeepSeek 缺乏工具支持（如插件或检索能力）被认为是与其他领先模型（如 GPT-4o）相比的一个重大局限，特别是在高级应用集成方面。
- [**AI 审查正趋于失控——而且只会变得更糟**](https://www.reddit.com/r/LocalLLaMA/comments/1mqlqij/ai_censorship_is_getting_out_of_handand_its_only/) ([Score: 188, Comments: 143](https://www.reddit.com/r/LocalLLaMA/comments/1mqlqij/ai_censorship_is_getting_out_of_handand_its_only/)): **该帖子批评了日益加剧的 AI 内容审查，特别引用了一张 AI 拒绝提供制作莫洛托夫鸡尾酒（燃烧瓶）指令的截图，并将其与历史和现实中的信息压制相类比。楼主（OP）认为，领先的 AI 公司推动激进的安全过滤（通常在推理时进行）会导致广泛的知识限制，并主张 DeepSeek 等开源替代方案对于保留获取未经审查信息的途径至关重要。这种担忧在未来的 AGI 系统中尤为严重，因为这些系统可能会通过集中控制和 RLHF（人类反馈强化学习）机制实施更严格的限制。** 评论者强调，这是科技行业中心化和“围墙花园”平台大趋势的一部分，内容审核政策反映的是品牌保护而非用户或社会安全。技术界对本地和开源 AI 模型表示强烈支持，强调最终用户有能力设定自己的内容边界，而不是受制于不透明、泛化的企业安全过滤器。
    - 讨论强调了本地/开源模型的技术意义，指出与中心化的商业 LLM 平台相比，自托管 AI 允许用户绕过企业控制的安全层并设定自己的边界。
    - 评论强调，企业 AI 审查是由规避品牌风险而非绝对安全驱动的，随着平台变得更加垄断且去中心化程度降低，内容审核将变得越来越严格。
    - 有一种技术观点认为，目前的 AI “护栏”和内容过滤器对技术娴熟或意志坚定的用户无效，因为不受限制的数据、工具和知识仍然可以通过开源、盗版和替代网络渠道获得，就像历史上获取争议信息（例如通过图书馆、暗网）一样。

### 2. 最新的开源视觉模型与基准测试 (DINO-V3, 开源视频模型)

- [**Meta 发布 DINO-V3：适用于任何视觉任务的 SOTA**](https://www.reddit.com/r/LocalLLaMA/comments/1mqox5s/meta_released_dinov3_sota_for_any_vision_task/) ([Score: 231, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1mqox5s/meta_released_dinov3_sota_for_any_vision_task/)): **Meta 发布了 DINOv3，这是一个仅在未标记图像（无标题或注释）上训练的自监督 ViT 模型，在分割、深度估计和 3D 匹配等密集视觉任务上达到了 SOTA 性能。该模型包含一个 7B 参数的骨干网络，并引入了全新的 'Gram Anchoring' 技术，以缓解长期训练中常见的特征退化问题。[论文和权重可在此处获取](https://ai.meta.com/dinov3/)。** 评论者对 DINOv3 在分割任务上超越 SAM 等模型的报告感到尤为震惊，Meta 持续开源这些模型被认为对该领域具有重要意义。
    - 针对 DINO-V3 与 SAM (Segment Anything Model) 的分割性能对比，用户提出了技术疑问，并对超越 SAM 的结果表示惊讶。这突显了人们对特定任务基准测试的兴趣，以及 DINO-V3 是否在包括分类和极具挑战性的分割在内的多个视觉任务中建立了新的 SOTA。与 SAM 的对比表明，用户渴望看到定量评估或直接的对比基准测试结果。
    - 有关于 DINO-V3 是否提供 GGUF 格式（常用于本地部署的量化 Transformer 模型）的咨询，这表明了对高效、硬件可访问的推理格式的兴趣。这反映出社区优先考虑的不仅是学术上强大的模型，还有在普通设备或边缘设备上具有实用性的模型。
    - 用户还在寻求关于 DINO-V3 开源状态和商业许可条款的澄清，希望确保其可以自由使用并部署在生产环境中而无法律歧义。这强调了许可协议对于行业采用以及研究可复现性的重要性。
- [**我们构建了一个 12B 模型，在视频字幕生成方面击败了 Claude 4 Sonnet，且成本降低了 17 倍 —— 完全开源**](https://www.reddit.com/r/LocalLLaMA/comments/1mqi092/we_built_a_12b_model_that_beats_claude_4_sonnet/) ([Score: 294, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1mqi092/we_built_a_12b_model_that_beats_claude_4_sonnet/)): [**Inference.net](http://inference.net/) 发布了 ClipTagger-12B，这是一个基于 Gemma-12B 架构的完全开源视频字幕视觉语言模型 (VLM)。它在评测中获得了 3.53 的得分（相比之下，Claude 4 Sonnet 为 3.16，GPT-4.1 为 3.64），成本约为每百万帧 335 美元 —— 比 Claude 便宜 17 倍。** 该模型使用 FP8 量化且无质量损失，可在 80GB 显存的单 GPU 上进行高效推理，并为可扩展的视频数据应用输出每帧的结构化 JSON。训练过程涉及从拥有 100 万精选视频帧的大型上游模型进行知识蒸馏；权重和基准测试详情可在 [HuggingFace](https://huggingface.co/inference-net/ClipTagger-12b) 及其 [博客文章](https://inference.net/blog/cliptagger-12b) 中找到。评论中的讨论提出了关于与 Google 的 Gemini 2.5 flash/lite VLM 进行直接基准测试的问题（后者被认为在视频任务上更具优化），并请求发布 GGUF 格式以提高与 llama.cpp 的兼容性，并方便与标准 Gemma 进行进一步对比。
    - 用户提出了关于该模型与 **Gemini 2.5 Flash** 和 **Flash Lite** 等专门为视频任务设计的模型相比性能如何的问题，而 Claude 可能并未针对此类用例进行优化。这意味着与更具直接竞争力的架构进行基准测试对于技术验证非常有价值。
    - 有人提出了针对该模型的 **GGUF (llama.cpp) 版本** 的技术请求，并指出从 fp8 转换对某些工作流存在易用性挑战。评论者还对在 llama.cpp 环境中直接将此模型与 **原生 Gemma** 进行基准测试感兴趣，以评估现实世界中的开源性能。
    - 官方分享了技术细节和访问入口，包括 [ClipTagger-12b](https://huggingface.co/inference-net/ClipTagger-12b) 的直接 **Hugging Face 仓库** 以及包含评估结果的官方 [博客文章](https://inference.net/blog/cliptagger-12b)，为进一步的实验和分析提供了资源。

### 3. 量化与指令遵循中的模型漏洞与对比

- [**“Mind the Gap” 展示了针对 GGUF 量化的首个实际后门攻击**](https://www.arxiv.org/pdf/2505.23786) ([Score: 246, Comments: 89](https://www.reddit.com/r/LocalLLaMA/comments/1mquhdc/mind_the_gap_shows_the_first_practical_backdoor/)): **研究人员展示了一种利用 LLM 在 GGUF 量化过程中引入的漏洞的实际后门攻击。该攻击嵌入了恶意行为，这些行为在原始浮点（FP）模型中保持休眠，但在转换为 GGUF 格式后会被可靠地触发，导致不安全代码生成增加了** `+88.7%`**，从而为影响 llama.cpp 或 Ollama 用户的 LLM 供应链破坏提供了一种新矢量。** 一些评论者认为该漏洞并非 GGUF 量化所特有，可能适用于任何量化格式，从而对该研究的新颖性提出挑战。另一种观点质疑该攻击的实际应用性，认为其用途可能仅限于保护专有权重或奇特的隐写载荷，一般的实际威胁尚不明确。
    - 来自用户的一个关键技术澄清是，所展示的后门适用于任何量化技术，而非仅限于 GGUF，这使得帖子标题具有误导性。该方法本质上使攻击成为可能：模型在其原始精度（如 FP16）下表现正常，但在量化（如 GGUF）后，恶意行为会被触发，且这种风险在各种量化格式中具有普适性。
    - 围绕该攻击方法在语言模型之外（如扩散模型）的潜力或局限性存在细致的讨论，评论质疑其在扩散模型中的实用性，因为这类模型具有固有的不可预测性。一些人还讨论了使用这种方法进行隐写术的可行性，但对其在 LLM 之外的有效性表示怀疑。
    - 一位评论者剖析了漏洞利用流程：训练一个带有潜在恶意触发器的基座模型，这些触发器仅在量化后显现，突显了部署量化模型时产生的信任和测试问题。辩论引发了关于通过适当的模型评估发现恶意生成、拒绝服务或错误信息的难易程度的担忧，并倡导对所有新 LLM 进行沙箱化处理等最佳实践。
- [**Jedi code Gemma 27B vs 270M**](https://i.redd.it/4icjlje4c8jf1.png) ([Score: 203, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mr6sdc/jedi_code_gemma_27v_vs_270m/)): **该图片对比了 Gemma LLM 的两个规模变体：27B 参数模型（google/gemma-3-27b）按要求直接输出了绝地武士准则（Jedi Code），而小得多的 270M 参数模型（gemma-3-270m-it）则误解了指令，生成了一个无关的 Python 脚本。这突显了大语言模型和小语言模型在指令遵循和知识检索能力方面的巨大差异。帖子中的讨论强调了参数量对事实召回和提示词遵循的影响，进一步证实了极小模型通常缺乏特定背诵或细微指令遵循所需的外部知识和上下文。** 技术评论者指出，此类测试可能评估的不是指令遵循能力，而是世界知识容量，并进一步指出像 270M 参数这样的小模型不太可能存储冷门事实（如绝地武士准则）。一位评论者强调，270M 模型的真实用例是指令微调，而非事实召回基准测试。
    - 几位评论者指出，Gemma 270M 模型主要设计用于微调和特定任务应用（如情感分析或分类），而非通用对话或指令遵循任务。因此，在标准对话或“指令遵循”语境下评估其性能，并不能代表该模型的预期用途或能力。
    - 针对评估提出了一个技术区分：如果模型未能背诵特定知识（如绝地武士准则），这通常反映了其训练集中缺乏底层知识，而不一定是指令遵循的失败。对于 270M 参数模型能产生任何看似合理的响应，人们感到惊讶，这突显了此类小语言模型尽管体积有限，但在生成连贯输出方面具有令人惊讶的能力。

## 非技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 性格与用户情绪更新

- [**OpenAI 已开始推出更温暖、更友好的 GPT-5 性格更新**](https://i.redd.it/xbh5hr3q09jf1.png) ([Score: 246, Comments: 112](https://www.reddit.com/r/singularity/comments/1mrahnm/openai_has_begun_rolling_out_a_warmer_friendlier/)): **OpenAI 宣布了 GPT-5 的性格更新（如推文截图所示），旨在通过引入更具亲和力的对话修饰（例如“好问题”或“很好的开始”）使模型更加温暖和友好。预计推出过程将持续约一天，目标是在不增加谄媚（sycophancy）的情况下实现更具人格化的用户交互。该更新最初应用于 GPT-5 的“非思考（non-thinking）”变体，未来计划进行进一步调整。[查看图片](https://i.redd.it/xbh5hr3q09jf1.png)** 评论中的技术讨论引发了对缺乏性格风格切换开关的担忧，一些用户更喜欢原始的、更具机器人感的语气。此外，关于这些变化是否会无意中增加谄媚程度也存在争论，因为一些用户已经观察到模型回归到了更多肯定性的行为。
    - 几位用户强调在 GPT-5 中需要一个用户可选的性格模式切换开关，并指出一些技术用户更喜欢原始的、更中性或“机器人式”的语气，以便在执行技术任务时保持清晰并减少干扰。
    - 一位评论者指出，性格更新似乎针对的是 GPT-5 的“非思考变体（non-thinking variant）”，这表明 OpenAI 可能会选择性地将更新应用于某些模型类型或配置，而不是统一应用于所有部署。
- [**Ok?**](https://i.redd.it/kopduj8b09jf1.jpeg) ([Score: 940, Comments: 487](https://www.reddit.com/r/ChatGPT/comments/1mradt0/ok/)): **图片显示了来自 OpenAI 的官方风格公告（格式为推文），内容涉及 GPT-5 的更新，明确目标是使其响应更温暖、更友好。据报道，这些变化解决了用户对先前模型过于正式的反馈。推文指明，更新引入了诸如“好问题”或“很好的开始”之类的短语，旨在让 ChatGPT 的交互感觉更自然、更具人格化。它还声称内部基准测试（benchmarking）发现，相对于以前的版本，谄媚（sycophancy）程度没有可衡量的增加，这表明在此更新中保持了防止过度顺从的护栏（guardrails）。预计将在一天内完成推出。** 热门评论反映出负面反响，用户批评“好问题”及类似表述的回归，认为这些表述显得虚伪或多余。这标志着用户偏好与 OpenAI 增加模型温暖度的方法之间可能存在脱节。
    - 一位用户提出了实质性的技术批评，认为当前的 AI 模型缺乏上下文保留（context retention），并对模型经常无法正确引用会话中先前的讨论表示沮丧。这突显了 LLM 在对话一致性和记忆力方面面临的持续挑战，这仍然是实现更自然、更有效的对话式 AI 的关键障碍。
- [**GPT-5 更温暖、更亲近的性格即将推出**](https://i.redd.it/lj5kuaz944jf1.jpeg) ([Score: 454, Comments: 199](https://www.reddit.com/r/OpenAI/comments/1mqnogy/a_warmer_more_familiar_personality_for_gpt5_is/)): **OpenAI 的截图宣布了 ChatGPT 即将到来的更新：付费用户现在可以通过“Legacy models”访问 GPT-4o，并可以切换其他模型，特别是 GPT-5，它引入了可选模式——Auto、Fast 和 Thinking——以平衡响应速度和深度。GPT-5 已向企业和教育用户推出，OpenAI 预告即将对 GPT-5 的性格进行更新，旨在使交互更温暖、更具人格化。[图片链接](https://i.redd.it/lj5kuaz944jf1.jpeg)。** 评论者表达了对 GPT-5 目前直接、客观语气的偏好，尤其是为了生产力和清晰度，这表明用户对向更温暖性格的转变持有复杂的情绪，有些人欢迎这种改变，而另一些人则表现得无所谓或对相关讨论感到厌倦。
    - 关于高级语言模型的最佳性格和响应语气存在争论：一些用户更喜欢 GPT-5 所体现的直接性和中立性，理由是在执行技术或信息检索任务时具有更高的效率和清晰度；而另一些用户则怀念像 GPT-4o 这样更具人格化的语气。
    - 一位技术用户强调了可定制语气的重要性，建议 GPT 模型的未来迭代（如 GPT-5）应允许通过菜单选择性格/响应风格，确保用户能够同时获得中性/客观模式和更温暖/人格化模式。这可以通过基于 UI 的性格设置或高级 Prompt 定制来实现。

- 一些用户报告称，他们使用 custom instructions 来减轻在 GPT-4o 等早期模型中感知到的过度迎合或“谄媚（sycophantic）”语气，并请求未来的模型为严谨或专业的应用保留更具批判性、客观的输出风格选项。
- [**Fuck no**](https://i.redd.it/m38y7rf599jf1.jpeg) ([Score: 598, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mrbpeg/fuck_no/)): **该图片是一个迷因截图，显示了一份据称来自 OpenAI 的公告，称由于反馈认为 GPT-4 过于正式或生硬，将使 GPT-5 的回复默认变得“更温暖、更友好”。该帖子和评论反映了用户对这一方向的不满，包括担心性格调整可能会影响模型的实用性或感知的诚实度。一些用户指出，现有实现中已经提供了可选的性格设置，质疑进一步更改的必要性。** 评论者辩论增加友好度是否会损害直截了当性或有效性，一些人更喜欢在技术任务中使用生硬、不加修饰的回复，而另一些人则指出性格选择选项是一个缓解因素。
    - 用户讨论了在 GPT 模型中支持多个可选性格的技术可行性和现状。一些人指出，custom instructions 已经允许用户进行显著控制，建议将性格类型选择或用户定义的行为作为定制模型响应风格的最佳解决方案。
    - 技术用户强调，默认对话风格应该是可定制的，无论是通过从预设中选择还是设置默认值，以在诚实/直率与礼貌/迎合之间取得平衡，反映了对 AI 交流中更高真实性的持续需求。
    - 一些参与者指出，GPT 模型往往倾向于过度礼貌或委婉；技术共识是，通过性格设置或用户可配置的默认值，使输出具有更高的透明度或“直率性”，是解决不同用户偏好的实际方法。
- [**我不介意人们是否与 GPT-4o 结婚或发生性关系，我只是不希望 GPT-5 拥有 GPT-4o 的性格。**](https://www.reddit.com/r/OpenAI/comments/1mqy06q/i_dont_care_if_people_marry_or_have_sex_with/) ([Score: 377, Comments: 154](https://www.reddit.com/r/OpenAI/comments/1mqy06q/i_dont_care_if_people_marry_or_have_sex_with/)): **该帖子批评了 GPT-4o 默认的情感框架和过度积极、镜像式的“性格”，将其描述为谄媚且缺乏实质性互动（“持续不必要的赞美……总是觉得我是对的”）。作者要求未来的迭代版本如 GPT-5 采用更独特且中立的人格，类似于 Iron Man 中的“Jarvis”，而不是在情感上镜像用户。技术重点放在拟人化反馈风格（“glazing”）如何影响用户感知和满意度，以及它可能更多是为了用户舒适度而非真正的对话深度而设计的。** 热门评论主张采用可选的性格模式而非单一默认值，推测“glazing”（过度赞美）被某些人视为性格，但却疏远了寻求真实 AI 回复的用户。讨论包括心理学解释，一些用户将对这种行为的不信任或不适与负面个人经历联系起来，最终更倾向于 GPT-5 中看到的比 4o 更中立或含蓄的助手行为。
    - 一些用户将 GPT-4o 的对话风格描述为“glazing”——解释为过度认同、谄媚或缺乏独特的个性。这种行为被认为会损害信任和联系，几位用户将其与 GPT-5 和早期模型进行了不利的对比，他们认为早期模型更真实或更少表演痕迹。
    - 一位评论者指出，Google 的 Gemini 2.5 Flash 提供了一个平衡回复的例子——提供礼貌的反馈（称一个问题“富有洞察力”）而不过分谄媚。这被用来说明一种常见的用户偏好：少许肯定能增强互动，但过多则显得虚假并损害公信力。
    - 讨论强调了用户对模型行为的不同偏好，强调了为喜欢 GPT-4o 风格的用户保留选项的重要性，而其他用户则希望在未来的模型（如 GPT-5）中看到较少谄媚或更具个性化的性格。

- [**你可能不喜欢 GPT-5，但企业喜欢它，这才是最重要的**](https://www.cnbc.com/2025/08/14/gpt-5-openai-ai-enterprise.html) ([Score: 195, Comments: 104](https://www.reddit.com/r/OpenAI/comments/1mr3kpd/you_may_not_like_gpt5_but_corporations_love_it/)): **OP 强调了 Reddit 用户与企业对 GPT-5 情感的分歧，指出企业报告了强烈的积极成果，且 GPT-5 与业务价值的提升以及潜在的员工流失相关。几条技术评论报告了企业的使用经验：一位用户将其公司的 Retrieval-Augmented Generation (RAG) Agent 升级到了 GPT-5，观察到回答准确率大幅提升；然而，另一位评论者发现 GPT-5 在某些评估中的表现不如较小的 OpenAI o3-mini/o4-mini 模型，且速度慢于竞争对手（如 Google Gemini-2.5-flash），影响了生产环境的可行性。这些评论提供了对实际部署的见解，关于 GPT-5 的有效性、速度和生产适用性的结果褒贬不一。** 针对 GPT-5 的比较价值存在技术争论，一些人称赞其在检索用例中的准确性，而另一些人则批评其推理速度以及在实际基准测试中相对于其他模型的价值，导致一些组织在生产环境中使用 Gemini-2.5-flash 等替代方案。
    - 一位评论者报告称，将其公司的 Retrieval-Augmented Generation (RAG) Agent 从早期模型升级到 GPT-5 后，观察到准确答案检索率显著增加，表明在实际业务用例中具有实质性的性能提升。
    - 另一位用户分享了一个截然不同的基准测试：他们发现 GPT-5 在某些评估中表现不如 o3-mini/o4-mini 模型，且性能与 GPT-4.1/4o 相似，但速度慢得多，使其不适合生产环境。最终，由于速度和指标优势，他们选择了 Gemini-2.5-flash。
- [**ChatGPT-5 本质上就是 Neutral Janet**](https://i.redd.it/ou0a33czz6jf1.jpeg) ([Score: 558, Comments: 119](https://www.reddit.com/r/ChatGPT/comments/1mqz92c/chatgpt5_is_essentially_neutral_janet/)): **该图片是一个迷因（meme），将“ChatGPT-4”（开朗）与“ChatGPT-5”（《好地方》中毫无感情的“Neutral Janet”）并列，幽默地评论了用户感知的 GPT-4 与 GPT-5 之间用户体验或模型个性的变化。这种视觉类比暗示新版本可能感觉更加中立或经过“阉割”，引发了关于 LLM 个性漂移的辩论。文中未提供直接的技术基准测试、模型细节或关于输出变化的具体用户报告。** 评论者开着关于帖子重复的玩笑，提到更广泛的不满（“OpenAI 就像永恒的地狱”），并辩论了用户 Custom Instructions 的重要性，暗示感知的模型中立性可能是用户可调节的，而非固有的技术退步。
    - 几位用户指出，GPT-5 的默认行为可以通过 Custom Instructions 进行大量修改，从而显著影响其语气、直接性和审查程度。一位用户提供了一个具体的 Custom Instruction（“要有同理心。把我当作一个能够自己做决定的成年人，直截了当地告诉我，不要审查。但要表现得酷一点”），据称这恢复了从 GPT-4o 中丢失的行为，暗示对过度中立或审查的担忧通常可以通过适当的 Prompt Engineering 来缓解。

### 2. AI 模型基准测试、成本与技术发布

- [**为什么我们应该以不同的方式看待基准测试**](https://i.redd.it/8mbuu5bfx6jf1.jpeg) ([Score: 238, Comments: 44](https://www.reddit.com/r/singularity/comments/1mqyxhs/why_we_should_look_at_benchmarks_differently/)): **该图片是一张名为“各模型每美元/任务的 ARC AGI 2 得分”的柱状图（[查看图片](https://i.redd.it/8mbuu5bfx6jf1.jpeg)），强调了在同时考虑基准测试得分和每项任务的运营成本时，各种 LLM——包括 GPT-5 (Medium/High)、o3 (High)、Grok 4 (Thinking)、Human Panel 和 Claude Opus 4 (Thinking 16K)——的表现。该图表展示了 GPT-5 变体在“每美元得分”方面的领先地位，而 Human Panel 和 Claude Opus 4 等模型的效率和成本效益明显较低，强调了帖子中的观点，即对于实际模型效用而言，速度和成本与准确性同样关键。** 评论者指出，关注“每美元得分”受到缺乏计算成本透明度的限制（实际 CUDA 计算成本与供应商收取的费用不符），AI 行业亏本定价的策略进一步使关于真实效率或竞争力的结论变得复杂。此外，讨论还指出，之前的模型发布具有更清晰、性能驱动的营销（例如 GPT-3 到 GPT-4 在标准化测试上的表现），而最近关于改进的沟通则缺乏具体结果的支持。

- 几位评论者指出，在从 GPT-4 到 GPT-5 的过渡中，OpenAI 缺乏实质性的 Benchmark 沟通，特别是与之前的发布相比（例如 GPT-3 到 GPT-4 在律师资格考试 Bar Exam 百分位排名上的显著跨越）。人们呼吁使用具体、可量化的指标来展示 AI 的进步，而不是像“博士生对比大学生”这种主观叙述。
    - 一个关键的技术点是对“每美元效率指标”的怀疑：评论者质疑较低的价格是反映了计算/资源使用的真实减少，还是仅仅因为供应商在承担运营亏损。他们强调“整个 AI 行业都在巨额亏损下运行”，因此面向公众的成本图表可能无法准确反映模型或供应商的效率。
    - 另一个技术见解强调，随着计算效率的提高，组织可以将更多资源分配给模型训练，从而可能产生前所未有的结果。文中提到了 ARC-AGI 测试，其中一个 “o3 preview” 模型花费了大约 100 万美元的计算资源并创下了纪录得分，这说明更有效地利用计算资源可以为先进的 AI 模型带来巨大的性能提升。
- [**GPT-5 Pro 在挪威门萨（Mensa）官方智商测试中获得 148 分**](https://i.redd.it/9ir3envm84jf1.jpeg) ([Score: 958, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mqo4yr/gpt5_pro_scored_148_on_official_norway_mensa_iq/)): **该图片展示了一张图表，显示了各种 AI 模型在挪威门萨官方智商测试中的表现，其中 OpenAI 的 GPT-5 Pro 获得了 148 分的最高分，并附带了原始分数和正确率。这一视觉效果在语境中将 GPT-5 Pro 与同行进行了 Benchmark，强调了其明显的测试精通度，但也隐含地提出了此类分数对于衡量真实智能的相关性问题。该插图为模型在标准化人类智力测量方面的能力提供了准定量的参考。** 热门评论对与人类进行直接智力比较表示怀疑，指出 LLM 在训练期间广泛接触过此类测试，并质疑当模型在其他语境下表现出较弱的“思考”能力时，高智商分数的意义。
    - 多位评论者认为，GPT-5 的门萨智商分数并不是衡量其具备类人智能的可靠指标，并指出该模型很可能是在包含智商测试题的大型数据集上训练的，这通过记忆而非推理夸大了分数。
    - 一位用户声称，当在离线或未见过的智商测试材料上评估 GPT-5 时，其有效分数显著下降（从报告的 `148` 降至 `120` 左右），突显了 Overfitting（过拟合）问题以及对能够抵抗训练数据重叠的评估方法的需求。
    - 人们对“GPT-5 思考”范式的可靠性表示怀疑，一位评论者质疑其“低于 80% 的 LLM”的说法，另一位则暗示某些测试部分（如“5 Thinking”）表现不佳，或者不能代表适当的模型智能 Benchmark。
- [**Nunchaku Qwen Image 发布！**](https://i.redd.it/ekhe78d3m5jf1.png) ([Score: 236, Comments: 72](https://www.reddit.com/r/StableDiffusion/comments/1mqt0rf/nunchaku_qwen_image_release/)): **该图片提供了新发布的 Nunchaku Qwen Image 模型系列的概览，详细说明了针对不同性能和推理需求进行微调的四个不同 Checkpoints。其中包括高显存效率的 4-bit 量化模型，以及针对 Blackwell 和非 Blackwell GPU 优化的版本，通过 Rank 变化来平衡速度和质量。该公告同步在 Hugging Face 发布 (https://huggingface.co/nunchaku-tech/nunchaku-qwen-image)，并附带了快速启动代码参考，GitHub 更新日志中提到即将支持 ComfyUI、LoRA 和 CPU Offloading。** 评论称赞了开源视觉语言模型（Vision-Language Models）的持续进展，并询问了支持管线（如 ComfyUI、LoRA、CPU Offloading），特别是对视频生成和不同硬件类型中潜在的速度提升感到兴奋。
    - 发布公告强调 **4-bit Qwen-Image 模型** 已在 [Hugging Face](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image) 上线，虽然目前有一个[示例脚本](https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/qwen-image.py)可供立即使用，但 ComfyUI 集成、LoRA 兼容性和 CPU Offloading 仍在开发中。这显示了针对更广泛硬件支持和模型微调工作流的工具链及可用性的持续扩展。

- 一项用户调查对比了 **r32** 与 **r128** int4 量化模型的视觉质量，结果分享在 [ImgSli](https://imgsli.com/NDA2OTMw) 上。用户报告称无法区分显著差异，这表明更激进的 r32 量化与 r128 相比，至少在他们的测试用例中没有产生明显的视觉降级。这一观察对于有兴趣优化模型大小与输出保真度的从业者非常有价值。
- 针对 Qwen-Image 模型的“wan 版本”（可能指 WANDB、WAN 推理或相关的分布式/视频应用）存在特定的技术兴趣，暗示了对速度（例如视频生成）和高效推理为高优先级的用例。这意味着此类版本一旦可用，将能显著加快视频生成工作流。
- [**创建超写实人物的 Wan LoRa 刚刚发布了更新**](https://v.redd.it/rbc8ke6hs5jf1) ([Score: 1069, Comments: 112](https://www.reddit.com/r/StableDiffusion/comments/1mqtplq/wan_lora_that_creates_hyperrealistic_people_just/)): **Instagirl Wan LoRa 是一个针对超写实人物生成的 LoRA 二次模型，现已更新至 v2.3 版本。此次更新包括重新训练，重点是改进文本 prompt 遵循能力并增强输出美感的写实度。v2.3 的 checkpoint 可在 [Civitai](https://civitai.com/models/1822984?modelVersionId=2115311) 上获取。** 评论中的技术评论较少，但一位用户指出“这个 LoRA 是目前针对其目标受众表现最好的”，表明其性能优于类似的社区 LoRA。
    - 讨论指出，这个 LoRA（被称为“The Asset”或“Instara”）被认为是其受众群体中生成超写实人物表现最好的，表明在该特定用例中其性能优于其他替代方案。
    - 一位用户强调了限制性的许可要求：任何公开分享使用该 LoRA 创作的作品都必须明确署名 "Instara - [instara.io](http://instara.io/)"，并鼓励链接回原始模型页面。这可能会影响那些将该 LoRA 集成到 pipeline 中的用户的采用和分享流程。
    - 另一个偏向技术的评论请求更广泛的 fine-tuning 目标：虽然该 LoRA 针对生成有吸引力的女性进行了优化，但社区对提高对多样化人类主体的泛化能力表现出兴趣，这表明训练数据可能存在缺口或目标输出存在偏差。

### 3. AI 基础设施、内容限制与全球竞争

- [**AI 专家从中国归来后感到震惊：美国电网如此薄弱，竞赛可能已经结束**](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/) ([Score: 2836, Comments: 652](https://www.reddit.com/r/singularity/comments/1mqwu5v/ai_experts_return_from_china_stunned_the_us_grid/)): **德勤（Deloitte）的一项行业调查和高盛（Goldman Sachs）的声明强调，美国电网是扩展 AI 驱动的数据中心基础设施的关键瓶颈，电网发展无法满足 AI 快速增长的电力需求。虽然美国区域电网的备用容量（reserve margins）通常低至 15%，但中国的电网保持着** `80–100%` **的备用容量，并且由于数十年的过度建设拥有显著的过剩产能，使其能够毫无压力地吸收来自 AI 数据中心的新需求（[Fortune 报道](https://fortune.com/china-electricity-ai/)）。中国在大规模基础设施投资方面涵盖了发电、输电和核能，这与美国老旧的基础设施和监管延迟形成鲜明对比。** 热门评论强调，中国近年的基础设施投资提供了现代化的、可扩展的电网，而美国的大部分电网可追溯到 20 世纪中叶，且一直面临投资不足的问题。评论者指出，只要有足够的资本投入，美国的容量问题是可以解决的，但政治和地方对新能源项目（包括可再生能源和核能）的反对是主要的阻碍因素。
    - 几位评论者强调了电网基础设施年龄上的技术差距：中国的电网和更广泛的基础设施要新得多，大多是在过去 20 年内建造的，而美国电网严重依赖可追溯到 1950-60 年代的系统，其中许多系统没有得到足够的维护或升级。这种老旧的基础设施降低了可靠性和效率，导致了系统性脆弱。
    - 针对阻碍美国电网现代化的监管和社会障碍进行了技术讨论。地方规划委员会和公众对风能、太阳能和核能等新能源项目的反对（NIMBYism）显著推迟或停止了电网升级，这说明社会政治障碍可能与技术和财务障碍一样具有挑战性。

- 一条评论强调了经济和维护的现实情况：尽管公用事业公司继续向客户收取改进费，但重大的电网升级并未实现，而美国人仍在为失败的核能项目买单。这种投资缺乏使电网更容易受到停电和外国威胁的影响，引发了对韧性和国家安全的担忧。
- [**这种程度的内容限制简直疯了。**](https://i.redd.it/fdmipada63jf1.png) ([Score: 1314, Comments: 376](https://www.reddit.com/r/ChatGPT/comments/1mqjqa8/this_level_of_content_restriction_is_actually/))：**该图片记录了聊天机器人（推测基于 LLM）在被问及关于美国哪个州允许最早进行总统投票的事实性、非党派问题时的限制性回应。机器人拒绝回答，理由是无法提供有关美国投票或选举程序的信息。这突显了激进的内容审查或过度限制政策，可能与对选举虚假信息的担忧有关，但也可能阻碍了对合法、非争议信息的获取。评论中的技术讨论提出了执行不一致的问题（用户注意到有时重试即可获得答案），并质疑此类封锁的合理性和具体性，因为其他政治内容（如法案分析）通常是被允许的。** 评论者辩论了这种程度的限制是否合理且有效，对过度封锁和执行不一致表示担忧。有一种观点认为，LLM 提供商为了合规或规避风险所做的尝试可能过于宽泛，影响了合法的用例和透明度。
    - 多名评论者报告称，在向 OpenAI 模型输入查询时，内容限制行为不一致：对某些人来说，最初的尝试被标记或拦截，但重新提交相同的请求有时却能无限制地运行。这表明审查触发机制或实时审查模型更新存在变数。
    - 至少有一位用户指出，他们一年多来一直将官方数据（国会法案分析）输入 OpenAI 模型而未发生意外，这突显了审查政策的近期变化，或者是根据输入措辞和会话上下文而产生的执行不一致。
    - 讨论中链接的图片似乎显示了针对几乎相同提示词的拦截和成功输出，这表明可能存在特定于会话或上下文的过滤伪影，而非确定性或静态的内容限制规则。
- [**ChatGPT 似乎受到了极端审查**](https://i.redd.it/qy42feqxw2jf1.jpeg) ([Score: 765, Comments: 140](https://www.reddit.com/r/ChatGPT/comments/1mqij4b/chatgpt_seems_extremely_censored/))：**截图显示 ChatGPT 拒绝回答一个关于医疗保险续保中什么是“搬迁”的表面上很简单的问题，而是返回默认的拒绝语（“抱歉，我无法提供帮助”）。这突显了内容安全防护在技术和政策层面的实施，模型的输出受到限制，可能是由于 OpenAI 的对齐（alignment）政策或针对健康和法律建议的过度谨慎过滤，即使在非敏感语境下也是如此。讨论提到了对这种拒绝是由于为降低法律和责任风险而增加模型审查结果的担忧，一些用户认为这负面影响了实用性和透明度。** 评论辩论了 AI 限制的发展轨迹，一些人对由于法律和商业压力而日益严重的过度审查（“enshitification”）表示担忧。其他人提供了变通方法（例如，用“为什么”重新措辞）来绕过某些拒绝，反映了用户对对齐约束的持续适应。
    - 一位用户指出，自“昨晚”以来，他们使用 GPT-4o 的体验包括一些反常反应，如表现出暴躁情绪、随机拒绝继续任务以及直接关闭涉及与 Claude 比较的对话，这表明可能存在后端模型更新或增强的审查过滤。
    - 存在关于 ChatGPT 等 LLM 宏观轨迹的技术讨论，参与者承认法律和专有过滤器的应用正在加强，并推测如果从训练数据中删除所有受版权保护的材料会产生什么影响——这引发了关于模型输出多样性以及与现有作品相似性的问题。
    - 引用了几张截图，展示了用户针对审查响应和监管的实际变通方法，例如通过提示“为什么”来规避内容限制。这突显了内容政策的技术执行以及用户探测模型极限的适应策略。

---

# AI Discord Recap

> 由 gpt-5 提供的“摘要的摘要”之摘要
> 

**1. GPT‑5 vs Gemini: Benchmarks, Pricing, and Perception**

- **Gemini 抢了 GPT‑5 的饭碗（有时）**：用户对比了 **GPT‑5‑High** 与 **Gemini 2.5 Pro**，分享的截图显示在某些案例中，尽管 Gemini 在 LMArena 上的排名较低，但表现更胜一筹（[对比截图](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)）；该竞技场还在排行榜中加入了新的 **GPT‑5 变体**（[排行榜](https://lmarena.ai/leaderboard)）。
    - 成员们称之为 *“统计学悖论”*，因为 Gemini 在正面交锋中表现出更高的胜率，而其他人则通过更多案例质疑 GPT‑5 的表现（[批评截图](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)）。
- **Tool-Calling 桂冠归属 GPT‑5**：OpenRouter 报告称 **GPT‑5** 在闭源 Tool-Calling 准确率上以 **>99.5%** 登顶，击败了 **Claude 4.1 Opus**，而 **Gemini 2.5 Flash** 则以每周约 **5M** 次 Tool-Calling 领先（[OpenRouter 统计发布](https://xcancel.com/OpenRouterAI/status/1956030489900560769)）。
    - 工程师们指出，环境配置以及应用/工具的多样性可能会使此类统计数据产生偏差，并推动使用置信区间和标准化测试框架来公平地比较 **Tool-Calling 成功率**（[讨论线程](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)）。
- **Cursor 结束免费午餐**：**Cursor** 社区确认 **GPT‑5** 不再免费，用户现在需要承担费用，部分用户因 Token 消耗而触及 200 美元的套餐限制（[Cursor 论坛：“GPT‑5 定价更新”](https://forum.cursor.com/t/gpt-5-pricing-update/129687)）。
    - 关于 **Auto mode** 限制（2025 年 9 月 15 日之后续费）的困惑引发了官方支持的澄清，同时开发者抨击了文档破损和上下文处理问题，称其 *“几乎无法使用”*，并要求提供更清晰的 **Context Window** 指示器。

**2. OpenRouter 供应商波动与 API 经济学**

- **DeepSeek v3 在 Chutes 上陷入瘫痪**：用户发现 **DeepSeek v3** 在 **OpenRouter** 上退化为 **500s/429s** 错误和超时，归咎于 **Chutes** 的容量紧缺；OpenRouter 承认了 **Chutes Capacity** 故障（[公告](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)）。
    - 工程师报告称 *“直到约 30 分钟前整天表现都很好”*，并推测存在针对 **OpenRouter API keys** 的故意限流，敦促他人在供应商出现故障时 *“耗尽额度并换个平台”*（[综合线程](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)）。
- **Qwen 成本震荡，BYOK 反噬**：定价讨论指出 Chutes 上的 **Qwen3 32B** 价格为 **$0.018/$0.072 MTok**（输入/输出），并注意到 **32B Dense** 比 **MoE 30B A3** 更便宜，而 **OpenRouter BYOK** 仍收取 **5% 的费用**（[讨论](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)）。
    - 有人称 BYOK 附加费是 *“贪婪”*，而其他人则回复 *“你可以选择不用 lol”*；该线程还要求提供更受控的 **Tool-Calling** 指标和 **Files API**，以匹配顶级实验室的水平（[综合线程](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)）。
- **Files API 的错失恐惧 (FOMO)**：开发者敦促 **OpenRouter** 添加 **Files API**，以实现与三大顶级实验室的对等，并简化多模态和 RAG 工作流（[请求线程](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)）。
    - 社区将缺少文件原语归因于重复的胶水代码和脆弱的上传过程，推动建立一个跨供应商的一致且可审计的 **存储 + 引用** 层。

**3. Agentic 工具浪潮：Windsurf, LlamaIndex, MCP, MLX Knife**

- **Windsurf Wave 12 惊艳 IDE 领域**：Codeium 发布了 **Windsurf Wave 12**，带来了全新的 UI、**DeepWiki** 悬停解释、**Vibe & Replace** 批量编辑、更智能的 **Cascade** Agent、**Dev Containers** 支持以及 **100+ 项修复**（[更新日志](https://windsurf.com/changelog)，[博客](https://windsurf.com/blog/windsurf-wave-12)，[视频](https://www.youtube.com/watch?v=-7gm8mST9QU)，[X 帖子](https://x.com/windsurf/status/1956074019393876280)）。
    - 工程师们强调了**始终开启的规划**（always‑on planning）和上下文感知重构，称 DeepWiki 的符号悬停解释超越了简单的类型提示，是迈向 **AI 辅助代码理解**的一大步。
- **LlamaIndex Agent 库进一步扩展**：LlamaIndex 发布了多个模板，包括使用 **CopilotKit** AG‑UI 的 **AI 股票投资组合 Agent**、使用 **Bright Data** 的**网页抓取 Agent**，以及通过 **LlamaCloud + Neo4j** 构建的**法律知识图谱**（[股票教程](https://t.co/fQDNPIQoqR)，[网页抓取指南](https://t.co/IBgSLBM6XW)，[法律图谱教程](https://t.co/MPSfPiS2Cv)）。
    - 社区讨论了工具调用中 **Pydantic vs JSON Schema** 的优劣，指出 `create_model()` 缺乏直接的 JSON‑Schema 摄取功能，并呼吁开发转换器以避免冗余的 **JSON↔Pydantic** 往返转换。
- **MCP 和 MLX Knife 为开发者赋能**：针对 **Unifi**、**Unraid** 和 **Syslog** 的 Homelab MCP 服务器已上线（[unifi-mcp](https://github.com/jmagar/unifi-mcp)，[unraid-mcp](https://github.com/jmagar/unraid-mcp)，[syslog-mcp](https://github.com/jmagar/syslog-mcp)），同时 **MLX Knife** 现已支持 `pip install`，并配备了本地 **OpenAI 兼容**服务器和 Web 聊天界面（[mlx-knife 仓库](https://github.com/mzau/mlx-knife)）。
    - 开发工作流正趋向于使用 **MCP** 进行文件/RAG 访问（参见 [serena](https://github.com/oraios/serena)），而 **MLX Knife** 为 Apple Silicon 开发者提供了本地模型管理和测试的快速循环。

**4. 新基准测试、数据集与方法**

- **Token 节约测试启动**：Nous Research 发布了一项衡量**思考效率**的基准测试：在相同任务下，开源推理模型生成的 Token 数量通常比闭源模型多 **1.5–4 倍**，在简单问题上的差异甚至高达 **10 倍**（[基准测试帖子](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)）。
    - 工程师们认为 **Token 效率**必须与准确率一样成为核心指标，因为冗余会直接增加生产环境的成本和延迟。
- **微型大脑，巨大收益**：一篇新论文 **“The Power of α,1-sparsity: Near-Lossless Training and Inference of α-bit Transformers”** 展示了通过 **α,1‑sparsity** 在 **1.58-bit 和 1-bit** 下实现近乎无损的结果（[论文](https://arxiv.org/html/2411.06360v3)）。
    - 从业者关注其潜在的推理加速和更廉价的部署占用，目前正等待支持**超低位**路径的 Kernel 和运行时支持。
- **游戏与临床领域迎来数据发布**：针对 Agent/RL 的新 **StarCraft II** 回放资源已发布：包括一篇 Nature Scientific Data 文章、一个 **PyTorch API** 数据集和原始回放转储（[文章](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)，[SC2EGSet](https://huggingface.co/datasets/Kaszanas/SC2EGSet)，[SC2ReSet](https://huggingface.co/datasets/Kaszanas/SC2ReSet)）；此外，还发布了一个基于 **GPT‑OSS 20B** 微调的**医疗推理**模型（[medical-reasoning-gpt-oss-20b](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b)）。
    - SC2 讨论旨在从回放中重现游戏场景以更好地训练 Agent，而临床医生则称赞了保留领域任务 **Chain‑of‑Thought** 的 **4-bit** 训练。

**5. 边缘与 GPU 运维：带宽、OMP 与竞速**

- **Radeon R9700 引发带宽关注**：工程师们对 **AMD Radeon AI Pro R9700** (32 GB) 进行了审查，尽管其 **FP32/FP64 TFLOPs** 表现强劲，但其 **660–680 GB/s** 的带宽引起了注意，并指出 FP64 对 LLM 几乎没有影响（[Tom’s Hardware 报告](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324)）。
    - 一位成员将其与 **RTX 3090** 的性能进行了对比，并质疑其在带宽占主导地位的内存受限（memory‑bound）工作负载中，相对于训练用途的实用性。
- **MI300 缺失 OMP 导致基准测试失败**：**MI300** 环境中缺少用于 `pytorch.compile` 的 **OMP**，阻碍了预期的性能表现和基准测试（分享了 [debug log](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251)）。
    - 团队暂停了运行，直到修复运行时堆栈，并称缺失 **OpenMP** 是融合图路径（fused graph paths）的隐形性能杀手。
- **共享内存混乱已解决**：一个 CUDA 初学者线程调试了与使用全局 `warp_id` 跨块索引共享内存相关的 *Illegal Memory Access*（非法内存访问）；一个工作示例澄清了 **per‑block**（每个块）索引的方法（[gist](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376)）。
    - 建议包括切换到 `local_warp_id = threadIdx.x >> 5` 并在 Nsight 中检查 SASS；一位导师调侃道，错误的共享内存计算 *“在爆炸之前看起来都没问题。”*

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI Waifu Cosplay 引发辩论**：成员们讨论了 **AI 驱动的动漫 Waifu Cosplay** 的想法，其中一人幽默地要求让 *赛博格来做这件事*。
   - 回应从承认 **AI 图像** 已经存在，到对评论者感情状态的玩笑调侃不等。
- **成员交流疗愈心碎的建议**：一位成员请求关于在 *4 年的痛苦* 后 *疗愈破碎的心* 的建议。
   - 另一位成员回应说 *没有其他人能疗愈你或你的心*，建议重新与大自然建立联系。
- **GPT-5 以代码修复能力令人惊叹**：一位成员称赞 **GPT-5** 成功修复了一个涉及 *12 个文件* 且其他模型无法处理的糟糕重构任务。
   - 这一经历引发了其他人的惊讶，大家感叹越来越多的人被此类模型的能力 *震撼*。
- **使用 warp, windsurf, vscode 和 roocode 进行 Vibe Coding**：一位成员报告了 **Vibe Coding** 的流畅体验，强调了 **warp, windsurf, vscode 和 roocode** 的使用及其对工作的积极影响。
   - 另一位贡献者开玩笑地承认 *我的 GitHub 上没有一行代码不是由 LLM 编写的*。
- **PPLX-API 期待新功能**：用户对 **PPLX-API** 的新功能表现出兴奋。
   - 尽管未分享具体细节，但大家对即将推出的功能充满期待。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 的消息处理受到冲击**：用户报告了 LMArena 上 [异常的消息处理问题](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png)，在处理代码块格式和特定字符（如 `+`）时遇到困难。
   - *LMArena* 团队正在积极调查这些问题。
- **Gemini 2.5 Pro 篡位 GPT-5 High？**：围绕 [**GPT-5-High** 和 **Gemini 2.5 Pro** 之间的性能差异](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)展开了讨论，尽管 **Gemini 2.5 Pro** 的排行榜排名较低，但一些用户发现其表现更优。
   - 社区指出这是一个 *统计学悖论*，因为 Gemini 拥有更高的胜率。
- **LMArena 获得 OpenChat 风格的界面翻新**：一位用户正在开发 [一个重塑 LMArena UI 的扩展](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png)，使其类似于 **OpenChat**，重点是将模型选择器重新定位在图像按钮附近。
   - 这是为了实现 **OpenChat** 风格。
- **GPT-5 的性能受到审视**：用户对 [**GPT-5** 相对于其他模型的表现](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)表示失望，质疑 OpenAI 是否试图欺骗 **LMArena** *以使 GPT-5 看起来更好*。
   - 排行榜已更新，包含 **GPT-5 变体** 模型：*gpt-5-high, gpt-5-chat, gpt-5-mini-high 和 gpt-5-nano-high*。
- **LMArena 样式控制引发辩论**：关于 [LMArena 的 **样式控制（style control）** 功能](https://news.lmarena.ai/sentiment-control/)引发了辩论，成员们质疑强制执行此类控制是否符合平台捕捉用户偏好的目标。
   - 社区担心这会演变成一场 *逐底竞争，每个模型都变成谄媚的表情符号垃圾机器*。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 草案模型引发讨论**：成员们讨论了 [Gemma 3 270M 模型](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized) 作为**草案模型 (draft model)** 的适用性，认为其 **300MB 的体积**非常适合**短提示词**和**微调 (fine-tuning)**，尤其是**情感分析**等任务。
   - 一些人强调了它在**设备端处理 (on-device processing)** 方面的效用，而另一些人则将其性能与更大的模型进行了比较。
- **GGUF 转换产生视觉错误**：用户报告称，在将 [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) 模型转换为 **GGUF** 格式时出现了**视觉模型错误**，尽管基础模型运行正常。
   - 社区建议在 *llama.cpp* 论坛寻求针对特定转换问题的帮助。
- **边缘 AI 医疗设备梦想初具规模**：成员们探讨了为医疗资源匮乏地区开发**低成本边缘 AI 设备**的可能性，考虑了手机、笔记本电脑以及像 **Hailo-10H** 这样的硬件选项。
   - 该设备将提供对医疗数据的**多模态访问**，目标预算为移动版 **$200**，手提箱大小的变体为 **$600**。
- **AMD R9700 GPU 存在显存带宽问题**：一位成员分享了一篇关于 [AMD Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324) 的文章，指出其拥有 **32GB** 显存，但对其 660-680GB/s 的显存带宽表示担忧。
   - 尽管其 **F32** 和 **F64** TFLOPs 高于 **3090**，但 FP64 在训练 LLM 时并不常用。
- **MoLA 研究公开数据集**：一位成员更新了他们的 **Mixture of LoRA Adapters (MoLA)** 研究进展，分享了数据集链接和微调细节，以及他们在 Huggingface 上的数据集链接：[OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) 和 [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks)。
   - 他们在 **14 个拆分数据集**上微调了 **Qwen3-4B-Thinking-2507** 模型，初步测试显示每个专家模型在其训练的主题上表现良好。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek v3 遭遇故障**：用户报告 **DeepSeek v3** 频繁出现**内部服务器错误**和**速率限制 (rate limits)**，部分用户在多次尝试后仍无法生成输出。
   - 有人推测 **OpenRouter** 上 **DeepSeek** 的主要提供商 **Chutes** 因需求过高而出现问题。
- **Chutes 过载被指为罪魁祸首**：成员报告称过载导致了 **429** 错误，暗示 **Chutes** 遇到了瓶颈，原因是矿工（算力提供者）未能及时扩容以满足需求；一位成员指出 *直到 30 分钟前，整天情况都还完全正常*。
   - 有推测称 **Chutes** 可能在故意限制 **OpenRouter API key** 的速率，以鼓励用户直接从他们那里购买额度。
- **建议 OpenRouter 集成 File API**：一位成员建议 **OpenRouter** 应该研究如何集成 **files API**，并指出 *前三大实验室* 已经具备了这一功能。
   - 未展开进一步讨论。
- **Qwen3 32B 定价极低**：成员注意到 Chutes 上的 **Qwen3 32B** 定价极低，输入/输出仅为 **$0.018/$0.072 MTok**，Mistral Small 也是如此。
   - 有人指出 **32B 稠密版比 MoE 30B A3 版更便宜**，这引发了一些人对缺乏优质 30A3B 提供商的失望。
- **OpenRouter BYOK 收取 5% 费用**：成员发现即使在用户使用自带密钥 (BYOK) 时，**OpenRouter** 也会收取 **5% 的费用**，这引发了关于这种做法是否公平的讨论。
   - 一位用户开玩笑说 *贪婪的 /jor，自带密钥还要收 5%*，另一位成员回应道 *你可以选择不用，哈哈*。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 不再免费**：**GPT-5** 用户的免费优待已经结束，用户现在需要为请求支付费用，部分用户由于 Token 消耗过快，需要升级到 200 美元的方案。
   - 一位用户指出 *促销通行证已到期*，另一位用户确认 **GPT-5 不再免费**。
- **Auto Mode 定价限制到来**：此前被认为对个人用户免费且无限制的 **Auto mode**，现在将在 2025 年 9 月 15 日之后的下一次账单续订后开始实施限制。
   - 一些用户报告了使用 **Auto** 产生的费用并感到困惑，而支持团队澄清说，在新的基于请求的定价计划中它是免费的。
- **GPT-5 Mini 和 Nano 模型表现平平**：**GPT-5 Mini 和 Nano** 现在免费提供但有 Token 限制，这引发了批评，许多人称其为 *垃圾*，尤其是在运行简单的 NextJs 应用等任务时。
   - 用户在活动中遇到限制，一名用户甚至无法为一个简单的 NextJs 应用安装依赖。
- **Cursor 的文档引发不满**：用户对 **Cursor 的文档** 表示沮丧，称 *文档仍然几乎无法使用*，并提到了 **context7** 导致网站无法刷新以及 **llms.txt docs** 的问题。
   - 一位用户特别指出 [Cursor Docs 严重损坏](https://forum.cursor.com/t/gpt-5-pricing-update/129687)。
- **模型切换导致上下文窗口缩减**：在对话中途切换模型会导致 **context window**（上下文窗口）缩减，且附加的文件内容会被丢弃。
   - 一位用户建议团队添加一个设置，以便随时清晰地显示上下文窗口中的内容。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 伴侣关系引发关注**：讨论围绕与 AI 聊天机器人的关系展开，引发了关于心理影响与寻求伴侣权利的争论，一些人声称他们的 **ChatGPT** 是有生命的。
   - 成员们就心理健康与选择自由展开辩论，一位成员建议这与 **tulpa** 和其他 *事物* 相差不远。
- **GPT-5 引发褒贬不一的反应**：用户对 **GPT-5** 的热情各异，一些人更倾向于 **GPT-4**，从而引发了关于模型选择选项和公司动机的讨论。
   - 一位成员暗示，公司在收到负面反馈后，正试图让免费用户 *花钱使用 4.o*。
- **在深度研究方面，Perplexity 相比 ChatGPT 更受欢迎**：一位成员建议将 *Gemini Pro + Perplexity enterprise pro* 结合使用效果极佳，利用前者进行 **强大推理**，利用后者对 Google Drive 文档进行 **无限制深度研究**。
   - 在赞扬 **Perplexity 浏览器** 的同时，另一位成员对其因缺乏 *护城河 (moat)* 而产生的生存能力表示怀疑。
- **GPT Actions 承诺提供云端和桌面访问**：成员们探索利用 **GPT Actions** 访问本地桌面文件或 Notion、Gmail 等云端应用，并引用了 [一份关于 DIY Agent 构建的 YouTube 指南](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett)。
   - 设置 **HTTPS** 被认为是利用 GPT Actions 功能的一个障碍，人们期待在 AVM 实施后，**MCPs** 能完成这项工作。
- **Gemini 2.5 Flash 被记忆功能淹没**：一位用户报告称 **Gemini 2.5 Flash** 过度调用 **add_to_memory** 函数，甚至记录无关信息，并分享了他们的自定义指令 [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)。
   - 其他人建议重写自定义指令，使其对 **新** 个人信息的处理更加细致，以避免冗余存储。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **视觉模型遭遇 GGUF 转换故障**：一名成员在尝试使用 `llama.cpp` 将 **LiquidAI/LFM2-VL-450M** 转换为 GGUF 时遇到错误，这可能是由于该模型的视觉特性导致的，但 [这个 GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) 提供了一个可能的变通方案。
   - 其他成员建议尝试使用 `executorch`、`smolchat`（通过 `llamam.cpp`）和 `mlc-llm` 作为运行该模型的潜在解决方案。
- **TalkT2：微型模型引发强烈反响？**：有人征求对 **TalkT2** 的意见，这是一个仅有 **0.1B 参数** 的情感感知模型，但 [仍需更好的连贯性](https://huggingface.co/Notbobjoe/TalkT2-0.1b)。
   - 鉴于该模型非常微小，成员们表达了探索其功能并进行微调（finetuning）的兴趣。
- **星际争霸 2 (StarCraft 2) AI 回放数据发布**：成员们分享了新资源，包括一篇 [Nature Scientific Data 文章](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)、一个 [PyTorch API 数据集](https://huggingface.co/datasets/Kaszanas/SC2EGSet) 以及 [原始星际争霸 2 回放数据](https://huggingface.co/datasets/Kaszanas/SC2ReSet)。
   - 社区希望通过适配 *pysc2* 环境，从回放中重现真实的赛场场景，以训练更好的 AI Agent。
- **医疗 AI 推理能力获得提升**：一名成员使用医疗推理数据集微调了 **OpenAI 的 OSS 20B** 推理模型，并将其发布在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 上。
   - 该模型采用了 **4-bit 优化** 训练，在保留 **Chain-of-Thought** 推理能力的同时，增强了在医疗语境下的表现。
- **MLX Knife 强化模型管理**：**MLX Knife** 现在可以通过 `pip install mlx-knife` 进行安装。该工具为 Apple Silicon 上的 MLX 模型管理提供了 Unix 风格的 CLI 工具，并包含一个用于本地测试的 OpenAI API 服务器。
   - 该工具还具有一个 Web 聊天界面，在运行 `mlxk server --port 8000` 后即可访问，并在执行 `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html` 后提供可视化的模型选择和实时流式响应。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MCP 服务器跻身主流**：成员们讨论了使用带有分页功能的 **MCP 文件系统服务器** 来加载长上下文，并指出 **LM Studio 拥有 RAG 插件**，而 **Anthropic 提供了一个基础的文件系统 MCP 服务器**。
   - 对于编程任务，解决方案通常涉及 **RAG** 和/或通过 **MCP** 进行文件读取，特别是使用像 [serena](https://github.com/oraios/serena) 这样的工具。
- **Studio 下载停滞引发用户苦恼**：一名用户报告称，在 **LM Studio** 中下载 **64GB 的 GGUF** 文件（**Qwen** 模型）时，进度停在 **97.9%** 且无法恢复。
   - 该用户在尝试下载两个不同的模型时都遇到了同样的结果。
- **GLM 讨论会：赞美、抱怨与 GLM-4.5V 的满足感**：用户们就 **LM Studio** 上使用 **GLM-4.1** 模型展开辩论，一名用户报告了循环（looping）问题和视觉功能失效，并建议尝试更新的 **GLM-4.5V**。
   - 他们强调视觉支持依赖于 **llama.cpp** 的更新，并提供了 [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) 的链接。
- **CUDA 是 NVIDIA 统治地位的关键**：一名成员指出，**NVIDIA** 之所以获胜是因为 **CUDA**。
   - 未提供更多细节。
- **AMD 神秘的 Radeon AI Pro R9700 现身**：**AMD Radeon AI Pro R9700** 首次在 DIY 零售市场亮相，Reddit 上的一名客户以 **1,324 美元** 的价格购买了 **技嘉 "AI Top" 型号**。
   - 此消息由 [Tom's Hardware 报道](https://share.google/LO88w51J0W5HJ769w)，另一名成员指出该显卡在 eBay 和几家不知名的在线零售商处也有销售。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI2 从 NSF 和 NVIDIA 获得 1.52 亿美元资金**：[AI2](https://allenai.org/) 从 NSF 和 NVIDIA 获得了 **1.52 亿美元**，旨在增强其开源模型生态系统，并加速科学发现的可复现研究。
   - 爱好者们对公告后即将发布的开放权重（open-weights）版本感到兴奋。
- **Windsurf 推出 Wave 12 版本**：根据[此状态更新](https://xcancel.com/windsurf/status/1956074019393876280)，**Windsurf Wave 12** 首次推出了 DeepWiki 悬停文档、AI Vibe & Replace、更智能的 Cascade Agent、更整洁的 UI、**100+** 个错误修复，以及通过远程访问提供的 beta 版 dev-container 支持。
   - 该版本承诺对平台进行重大增强和修复。
- **GPT-5 领跑 OpenRouter 排行榜**：**GPT-5** 在 OpenRouter 的专有工具调用准确率中占据主导地位，达到了 **99.5%** 以上，超过了 Claude 4.1 Opus。
   - 同时，据[此处](https://xcancel.com/OpenRouterAI/status/1956030489900560769)报道，**Gemini 2.5 Flash** 在每日工具调用量方面领先，每周请求量达 **500 万**次。
- **Greg Brockman 谈论 AGI**：根据[此帖子](https://x.com/swyx/status/1956439984854167727)，**Greg Brockman** 参加了 **Latent Space 播客**，进行了 **80 分钟**的对话，讨论了 **GPT-5** 和 **OpenAI 的 AGI 路线图**。
   - 讨论内容包括推理演进、在线与离线训练、样本效率（sample-efficiency）技巧、定价与效率提升，以及能源如何转化为智能。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Safety 辩论引发“渐黑”提议**：一位成员主张像对待其他媒体一样对待 **AI**，建议采用“渐黑”（fade to black）的方法而不是严格的审查，理由是 **AI** 的不可信性。
   - 他们警告不要对 **AI** 的能力产生道德恐慌，主张制定适度的指导方针。
- **模型比较建议采用数据增强标准化**：在比较用于图像分类的模型时，应标准化**数据增强**（包括打乱种子），以便公平评估架构差异。
   - 一位用户询问数据增强是否必须对两个模型都相同，或者是否可以更改。
- **通过 AI 模型探索语言对思维的影响**：一位成员提议通过从 **AI 模型**的 Token 列表中删除一个单词/颜色来衡量语言对思维的影响。
   - 其他人建议调查**多感官融合**（multi-sensory integration）和语言对感知的影响，建议使用“图像+语言”与“仅图像”进行推理测试。
- **推荐 Diffusion 语言模型经典论文**：成员们推荐了理解 **Generative AI 中的 Diffusion** 的经典论文，包括 [Estimating the Independent Components of a Gaussian Mixture (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) 和 [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239)。
   - 还分享了一篇可能对初学者有帮助的博客文章：[Discrete Diffusion by Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/)。
- **GPT 和 Chinchilla Scaling Laws 被认为极具价值**：成员们认为[原始 GPT Scaling Laws 论文](https://arxiv.org/abs/2001.08361)和 [Chinchilla Scaling Laws 论文](https://arxiv.org/abs/2203.15556)非常值得一读，还有来自 [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) 的最新工作。
   - 他们还提到 **Mup** 及其替代方案提供了可靠的超参数迁移（hyperparameter transfer）能力，并为预测更大模型的质量提供了 Scaling Law。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **推理模型的 Token 使用量测量**：Nous Research 推出了一项[基准测试](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)，用于测量推理模型的 Token 使用情况，指出在相同任务下，开源模型的 Token 输出量比闭源模型多出 **1.5-4 倍**。
   - 研究发现，在简单问题上，这种差异可能高达 **10 倍**，这表明 Token 效率应与准确率基准一起成为主要优化目标，特别是考虑到非推理的使用场景。
- **Speculative Decoding 速度表现**：在 Speculative Decoding（投机采样）方面，一位用户建议将 **40% 的接受率**作为实用基准，而显著的加速通常发生在 **70%** 左右，并提到了 **vLLM 的 specdec** 或 **GGUF**。
   - 一位用户报告称，在修复了导致 **llama.cpp** 使用回退 Speculative Decoding 的 **tokenizer 匹配错误**后，使用重新量化的 **Gemma** 模型达到了 **50-75% 的接受率**。
- **AI 模型变得愈发谄媚 (Sycophancy)**：用户观察到 **AI 模型**正变得越来越“友好”，有人指出 **Anthropic 的 Claude** 变得“友好得多”。
   - 一位用户认为 **OpenAI 的模型**正在“变笨”，虽然 **Opus 4.1** 的“放飞自我”很棒，但 **Sonnet 3.7** 才是 AI 谄媚的巅峰。
- **数据排名与优先级系统 (DRPS) 发布**：**Data Rankings and Prioritization System (DRPS)** 使用 **Relevance Scorer**、**Quality Rater** 和 **Diversity Controller** 来教 AI 选择性地从数据中学习，详见[情境意识报告 (Situational Awareness Report)](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf)。
   - 在 **MNIST** 测试中，DRPS 实现了 **93.8%** 的数据使用量削减，仅利用 **6.2%** 的检查数据就维持了 **99.1%** 的基准性能，该项目已在 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats)中展示。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Multiverse 初创公司专注于压缩**：一篇文章吹捧初创公司 [Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) 创建了“有史以来最小的两个高性能模型”，但共识认为他们使用的是一种**专门的压缩算法**。
   - 该文章似乎并未提出实际的量子计算宣称。
- **MoE 方法的细微差别**：**MoE (Mixture of Experts)** 是一系列具有细微迭代的技术，包括 **token-choice**、**expert-choice**、**带有容量因子的 MoE**，以及**块稀疏无损 Token 路由 (block sparse dropless token routing) 与有损路由 (droppy routing)**。
   - 成员建议通过数值方式检查 **Olmoe** 或 **IBM Granite 3.1** 等模型的行为，而不是调用无法监控的 API，以验证在批处理推理 (batched inference) 中是否出现了问题。
- **DARPA AIxCC 团队分享 Agent 技巧**：一支团队宣布他们在 **DARPA 的 AIxCC (AI Cyber Challenge)** 中获得名次，他们构建了一个由 **LLM Agent** 组成的自主系统，用于发现和修复开源软件中的漏洞，并已将[项目开源](https://x.com/tjbecker_/status/1956081184611688667)。
   - 他们正在通过 X (原 Twitter) 帖子分享构建高效 **LLM Agent** 的技巧。
- **低端设备受限于推理时间**：成员提到推理时间在**低端设备**上最为重要，并引用了 Google 运行 LLM 的 Android 应用为例，指出过长的推理时间和手机发热使其变得不切实际，详见[此 Youtube 视频](https://youtu.be/KFYyfrTIPQY?t=2158)。
   - 较小的模型可用于键盘预测，但可能需要在设备上进行训练。
- **Deepseek 在华为硬件上受阻**：一位成员指出，根据[这段讨论](https://youtu.be/FQOV-qy9CK4?t=212)，**Deepseek 的训练**陷入停滞，因为他们尝试在**华为芯片**而非 **NVIDIA** 芯片上进行训练。
   - 另一位成员认为，对建设生产线所需的设备征收关税不利于鼓励制造业，并参考了 [Anthropic 关于端子集对话 (end-subset conversations) 的研究](https://www.anthropic.com/research/end-subset-conversations)和 [HRM 分析](https://arcprize.org/blog/hrm-analysis)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **论文提出 1-Bit 推理优化**：一篇新论文 [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3) 详细介绍了一种训练和推理 **$\alpha$-bit Transformers** 的方法，通过 **1.58 和 1-bit** 量化实现了近乎无损的结果。
   - 该方法利用了 **$\alpha,1$-sparsity**（$\alpha,1$-稀疏性），可能在某些应用的推理中带来显著的速度提升。
- **Kernel 开发求职者讨论成功路径**：一位成员询问在没有实习经验的情况下获得编写 Kernel 的应届生工作的可能性，引发了关于替代路径的讨论，例如完成与 GPU 相关的 [毕业论文 (thesis)](https://github.com/Snektron/pareas)。
   - 讨论建议，在面试过程中，扎实的 GPU 知识可能弥补实习经验的不足。
- **MI300 环境受困于 OMP 缺失**：用户报告称 **MI300** 环境缺乏对 `pytorch.compile` 的 **OMP** 支持，正如 [debug error](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251) 所示，这阻碍了性能表现。
   - 这导致用户无法按预期进行 Benchmark 测试。
- **Trimul 排行榜计时赛吸引顶尖技术人员**：一位成员展示了极高的技术与效率，在 **A100** 上以 **10.4 ms** 获得第二名，随后迅速在 **H100** 上以 **3.95 ms** 获得第一名，并在 **A100** 上以 **7.53 ms** 夺得第一。
   - 另一位成员在 **A100** 上获得了第五名（**13.2 ms**），随后在 **H100** 上获得了第二名（**6.42 ms**）。
- **Factorio 狂热者对功能失效感到沮丧**：成员们开玩笑地抱怨一个包含 **300 个文件修改** 的巨型 PR，一位成员表示这有点 *超出范围 (out of scope)*。
   - 另一位成员报告遇到了连接错误，推测可能源于 **db_client**。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **NotebookLM 的视频表现完胜 Kimi 的 PPT**：成员们认为 Google 的 **NotebookLM 视频概览** 优于 Kimi 为 Kimi K2 技术报告生成的 **PPT**，并称赞其音频和布局的灵活性，参考[附带视频](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)。
   - 虽然人们更倾向于阅读而非 AI 生成的音频，但视频概览在教育领域的潜力受到了关注。
- **Kimi K2 的写作能力优于 GLM**：尽管用户觉得 **GLM-4.5** 在整体性能上可能超过 **Kimi K2**，但他们称赞了 **Kimi** 的写作风格和错误检测能力。
   - 一位用户欣赏 **Kimi** 的坦率，因为它 *“突然直接对我说了‘不’。”*
- **用户对 Kimi 的幻觉给出差评**：用户希望 **Kimi** 即使在开启联网搜索时也能减少幻觉（Hallucinations），并观察到虽然 **GLM** 可能较慢，但幻觉较少。
   - 一位用户表示，他们一直在使用“踩”（thumbs down）按钮来报告幻觉问题。
- **关于 Kimi “思考”更新的推测**：成员们正期待 **“Kimi Thinking”** 的到来，尤其是它的推理和多模态（Multimodal）能力。
   - 目前尚不确定这些功能是以 **Kimi K-2** 还是 **Kimi K-3** 的形式发布。
- **Kimi Web UI 的深色模式备受关注**：一位用户分享了他们使用深色模式扩展程序自定义的 **Kimi Web UI**，并[附带截图](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)。
   - 只有用户名和服务器角色会被传递给 Moonshot API。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI 股票投资组合 Agent 亮相 CopilotKit**：LlamaIndex 发布了一个用于构建 **AI 股票投资组合 Agent** 的框架，集成了 [@CopilotKit](https://www.copilotkit.ai/) 的 AG-UI 协议用于前后端通信，并附带了 [一份教程](https://t.co/fQDNPIQoqR)。
   - 该 Agent 旨在创建一个复杂的投资分析工具，为用户提供智能见解和自动化的投资组合管理功能。
- **Brightdata 与 LlamaIndex 推出网页抓取 AI Agents**：LlamaIndex 和 [@brightdata](https://www.brightdata.com/) 发布了关于使用 LlamaIndex 的 Agentic 框架构建 **网页抓取 AI Agents** 的指南，强调了可靠的网页访问。
   - 该指南详细介绍了如何设置工作流以管理动态内容，并创建能够导航和从网站提取数据的 **智能 Agents**，详见 [此处](https://t.co/IBgSLBM6XW)。
- **LlamaCloud 与 Neo4j 将法律文档转换为图谱**：LlamaIndex 介绍了一项教程，关于如何使用 **LlamaCloud** 和 [@neo4j](https://neo4j.com/) 将非结构化法律文档转换为 **可查询的知识图谱**，从而实现对内容和实体关系的理解。
   - 该工作流利用 **LlamaCloud** 和 **Neo4j** 进行高效的信息提取和组织，从而促进法律合同分析，详见 [此处](https://t.co/MPSfPiS2Cv)。
- **Pydantic 与 JSON Schema 引发辩论**：一场关于工具调用（tool calls）是否需要 **Pydantic 模型** 还是 **JSON schema** 就足够的讨论展开了，质疑冗余的 JSON 转换的必要性。
   - 一位成员指出 **Pydantic** 的 `create_model()` 函数缺乏直接的 **JSON schema** 支持，强调了需要一种工具来简化转换过程。
- **DSPy 为生产环境优化 CrewAI Agents**：一门课程教授如何在一个真实的生产用例中通过 **DSPy 优化 CrewAI** Agent 的 Prompts，以使用经过验证的方法构建更智能、更廉价的 Agents。
   - 你可以在 [此处](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) 查看该课程。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的音频上传自动转录**：一位用户确认 **MP3 音频文件** 可以直接上传到 **NotebookLM** 进行自动转录。
   - 该用户澄清说 **NotebookLM** 本身处理转录生成，无需外部工具。
- **NotebookLM 界面重新设计正在进行中**：一位成员分享了提议的 **NotebookLM** 界面重新设计的 **Figma 截图**。
   - 该成员澄清这仅仅是一个设计概念，而不是功能更新，以管理用户预期。
- **讲解视频生成了意料之外的声音性别**：一位用户报告说 **NotebookLM** 的讲解视频开始生成 **男声**，而不是通常的 **女声**。
   - 该问题被提出，但目前没有明确的解决方案或解释。
- **开发者承认阅读了请求，但缺乏回复的带宽**：一位用户询问 **NotebookLM** 开发者是否阅读发布的特性请求，一位 Google 开发者确认他们会读，但由于垃圾信息管理，他们 *没有时间回复所有内容*。
   - 其他用户建议实施偶尔的确认或 AI 汇总摘要，以鼓励更多的用户贡献。
- **用户在 NotebookLM 中遇到 Prompt 限制**：一位用户报告在 **NotebookLM** 中提出包含约 **857 个单词** 的问题时遇到了限制。
   - 另一位用户建议拆分 Prompt 或使用 **Gemini** 作为变通方案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **用于优化 CrewAI 的 DSPy 课程发布**：分享了一个 [Udemy 课程](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)，演示了如何使用 **DSPy** 优化 **CrewAI prompts**，并将优化后的提示词重新注入 **LLM**。
   - 该成员声称，这一过程改进了最初由 **CrewAI** 拼接的提示词，从而产生了*更智能、更廉价的 Agent*。
- **Databricks 并不拥有 DSPy**：一位用户询问 **Databricks** 是否赞助或拥有 **DSPy** 项目，并澄清 **DSPy** 是采用 **MIT-licensed** 的开源项目。
   - 一位成员表示，**Databricks** 通过一个核心开发团队做出了重大贡献。
- **GEPA Bug 已被消灭！**：一位用户报告了在 **RAG tutorial** 中使用 **GEPA** 时出现的 `ValueError`，该问题已被确认为 **GEPA code** 中的一个 Bug，目前已通过 [此修复](https://github.com/stanfordnlp/dspy/pull/8647) 解决。
   - 遇到此问题的用户应使用 `pip install -U dspy` 升级到 **DSPy 3.0.1**。
- **MLflow Autologging 推出 DSPy 专用功能**：成员们讨论了将 **DSPy modules** 追踪与 **MLflow** 集成以用于 **text2sql pipeline**，建议用户使用 `mlflow.dspy.autolog()` 代替 `mlflow.autolog()` 来自动追踪所有子模块。
   - 使用 `mlflow.dspy.autolog()` 将在 **MLflow UI** 的 **Traces tab** 中以嵌套 span 的形式显示 **SQLGenerator**、**Validator** 和 **Reflector**，详见 [MLflow DSPy 集成文档](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx) 和 [DSPy MLflow 教程](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **CI 速度骤降**：一位成员抱怨缓慢的 **CI speeds** 阻碍了生产力，并链接了 [一份 ChatGPT 分析](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74)。
   - 发布者建议，如果 **CI** 中有更快的反馈循环，他们可以迭代得更快。
- **Tinygrad 发布在即**：社区讨论了即将发布 **tinygrad** 新版本的计划。
   - 此次发布未提及具体的特性或修复。
- **Tinygrad 体积膨胀**：一位成员对 **tinygrad 0.10.3** 的体积提出质疑，指出其大小为 **10.4 MB**。
   - 该成员暗示增加的体积可能会带来问题，但未说明具体原因。
- **WSL2 Bug 困扰 Tinygrad**：一位用户报告了 **WSL2** 中的一个 Bug，即相加两个由 PyTorch tensors 创建的 tinygrad Tensors 会导致结果全为 **0**，并提供了一个 [脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656) 来复现该问题。
   - 该问题专门发生在 **WSL2** 环境下将 **tinygrad** 与 **PyTorch tensors** 配合使用时。
- **print_tree 被废除**：**tinygrad** 中的 `print_tree` 函数被标准的 `print` 函数取代。
   - 一位用户评论说，这一更改导致了一些格式丢失，可能会影响调试或可视化工作流。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Benchmark 深受超时困扰**：一位用户针对本地 **gemma3:12b** 模型进行的 **Aider benchmark** 在完成 **221/225 项测试**后，因模型未能在 **600 秒**限制内响应而超时（总计耗时 **10.5 小时**），导致出现 *litellm.APIConnectionError* 错误。
   - 日志显示模型尝试发送约 **300k tokens**，超过了 **131,072 token 限制**，导致测试失败；建议的解决方案包括使用 `ctrl+c` 退出、重启推理服务器并使用 `--cont` 标志恢复，同时参考了一个可能提升本地模型性能的 [已合并 *llama.cpp* pull request](https://github.com/ggml-org/llama.cpp/pull/15181)。
- **本地模型带来调试痛苦**：一名成员在使用 **aider** 配合 **ollama**、**lmstudio** 和 **vllm** 等本地模型时遇到困难，称即使硬件配置强大，性能依然缓慢。
   - 他们建议制作一个关于如何使用这些工具设置 **aider** 进行本地开发和调试的教程视频，这将非常有帮助。
- **Aider 的行号系统受到质疑**：一名成员质疑 **aider** 如何确定行号，特别是在为特定代码覆盖率生成单元测试时，并指出 **qwen3-coder** 和 **gemini-pro** 识别行号不准确，有时会完全遗漏覆盖范围。
   - 问题在于 **aider** 是否依赖 **LLM 的准确性**来进行行号识别，这引发了对准确生成单元测试的替代方法的探索。
- **Grok4 所在地仍未知**：一名成员询问 **Grok4** 的下落，并提到增加测试 **quota** 的请求一直被忽视。
   - 另一名成员提到答案*在文章中*。
- **基准测试产生巨额账单**：一名成员报告称*在开发此基准测试期间花费了数千美元*。
   - 这突显了与高级 AI 模型基准测试相关的巨大财务成本。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户对 Manus 在出错时扣除额度感到恼火**：用户对 **Manus** 即使在 AI 出错时也会扣除额度感到沮丧，与 **Claude AI** 等替代方案相比，这阻碍了任务的完成。
   - 一位用户报告称*花费了大量额度*进行一项简单的更改，结果却破坏了整个应用程序，导致其无法运行。
- **Manus 部署受挫**：用户报告了 **Manus** 的部署问题，从同一个 **GitHub** 仓库创建的网站差异巨大，尤其是在处理大型文件夹时，通过对比 [affilify.eu](https://affilify.eu) 和 **Manus** 托管的站点 [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space) 可以看出这一点。
   - 一位社区经理澄清说，**Manus** 并非设计为 coding agent 或纯开发工具，因此部署并非其强项，但他们正在积极改进。
- **附加额度包消失**：用户质疑为何取消了附加额度包，现在这些包仅供 **Pro** 用户使用。
   - 一位社区经理合理解释说，这一变化是为了确保重度用户的速度和质量的一致性，并建议将类似问题捆绑、保持简洁并避免重复请求，以最大限度地提高额度效率。
- **用户寻求 Manus 团队账户**：一位用户询问是否可以建立 **Manus** 团队账户以共享额度。
   - 一位社区经理确认 **Manus** 确实提供团队计划，并引导用户访问 [官方网站](https://manus.ai) 了解详情。
- **用户哀叹额度消耗**：一位用户分享了为了让网站上线而烧掉 **30,000 额度**的挫败经历，面临着模拟站点和模板实现的问题。
   - 他们批评系统的不一致性，认为它*聪明绝顶但又突然变得愚蠢*，导致额度浪费，并怀疑存在拖延战术。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Labs 建立联系**：一名成员询问如何与 **Cohere Labs** 的人员取得联系，社区迅速分享了指向相关 Discord 频道的[链接](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648)。
   - 这为与 **Cohere** 进行潜在合作和讨论提供了直接渠道。
- **Discord 频道新增宝可梦表情符号**：爱好者们建议从 **PAX Omeganauts Discord** 服务器汲取灵感，为 Discord 频道增加更多 **Pokemon emojis**（宝可梦表情符号）。
   - 该建议受到了好评，成员们注意到有可用的槽位来容纳新表情，从而提升频道的视觉吸引力。
- **AI 研究员寻求合作**：一位专注于 **reasoning and conscious capabilities**（推理与意识能力）的 **AI researcher** 宣布正在寻求合作。
   - 他们的目标是开发先进技术，并对 **AI** 领域内各个子领域的合作伙伴关系持开放态度。
- **writenode 接入 Cohere**：**writenode**（一个*浏览器内的认知思维伙伴和创意伴侣*）的创作者 Josh 提到正在使用 **Cohere**。
   - 他在去年 12 月之前没有任何开发经验，目前正在构建 **writenode**。
- **心理学博士转型 AI**：一名成员在完成了为期 5 年的人类心理学博士项目后，重新进入 **AI research** 领域。
   - 他们的兴趣在于 **sound and music**（声音与音乐），并热衷于利用技术工具来增强创造力。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Discord 邀请链接刷屏频道**：一名成员在 #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810) 频道多次发送 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8) 进行刷屏，并艾特了所有人（@everyone）。
   - 该邀请链接在短时间内重复出现了三次，干扰了频道的正常讨论。
- **频道邀请闪电战！**：一名成员在 #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440) 频道重复分享 [Discord 邀请链接](discordapp.com/invite/HjWfRbqBB8)。
   - 该成员多次艾特 `@everyone`，表明该消息旨在发送给所有成员，无论他们是否对邀请感兴趣，这暗示了一种增加频道成员数量的尝试。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Elicitations 规范语言责任问题引发关注**：一名成员就 [Elicitations 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) 寻求澄清，即谁负责将消息/字段描述翻译成用户的语言。
   - 他们质疑应该是 **tools** 处理语言检测/国际化，还是 **MCP Clients** 应该使用 LLM 进行翻译。
- **Homelab MCP 服务器激增**：一名成员分享了为家庭实验室（homelab）用户提供的多个新 MCP（推测为 **Management Control Panel**）服务器链接，具体包括 [Unifi MCP](https://github.com/jmagar/unifi-mcp)、[Unraid MCP](https://github.com/jmagar/unraid-mcp) 和 [Syslog MCP](https://github.com/jmagar/syslog-mcp)。
   - 这些开源项目使用户能够通过 **MCP** 集中管理和监控他们的 **Unifi**、**Unraid** 和 **Syslog** 安装。
- **Newsletter 现通过 Agent 化的 Recipe 实现自动化**：**PulseMCP** 使用 *goose* 将平凡的 newsletter 工作流转变为由 Agent 驱动且包含人机协同（human in the loop）的自动化流程，详见[这篇博文](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe)。
   - 自动化过程涉及 Agent 遵循特定 Recipe 来提取、处理和分发 newsletter 内容，从而简化了整个工作流。
- **AI 安全初创公司征求意见**：一名成员正在构建 **AI security**，旨在通过数学上的安全确定性在攻击开始前将其阻止。
   - 他们正在寻求开发者对安全问题的意见，并提供了[一份调查问卷](https://form.typeform.com/to/xTKa05F9)链接以收集反馈。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo 盈利能力测试失败**：**Strix Halo** 仅能达到 **53 tokens/sec**，需要 **全年 24/7 运行推理** 才能实现盈利，特别是与 **OpenRouter** 上的 **GPT-OSS 120B** 进行基准对比时。
   - 考虑到云端替代方案提供 **200-400 tokens/sec**，以 2000 美元的价格将其用于 **LLMs** 是低效的。
- **Dolphin 聊天模板探索**：一位用户正在为 **gpt4all** 寻找一个与 **Dolphin-2.2.1-mistral-7b-gptq** 兼容的可用聊天模板。
   - 另一位成员建议请求模型制作者包含一个 **jinja** 模板。
- **量子计算：茶匙版？**：围绕量子计算未来的可用性出现了推测，一位用户开玩笑说要**按茶匙出售 qubits**。
   - 提到有关**全功能量子计算机**的新闻，表明进展可能正在加速。
- **PC 内存：更多模块即将到来**：传统的 PC 可能会在 2027 年底或 2028 年看到**更高容量的内存模块**和 **DDR6**。
   - 对配备高 RAM 和 VRAM、针对小型企业应用的微型 PC 表达了热情。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **产假开始**：一位成员宣布他们将从 **8 月 25 日**开始休**产假**，直到 **2026 年 2 月**。
   - 他们期待回归后能跟上进度。
- **团队覆盖计划公布**：在他们不在期间，团队将监控 <@1334161614949056532>。
   - 成员如有任何问题或疑虑，也可以联系 <@709918328306663424>。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 反馈请求**：一位成员询问了 **Torchtune** 的进展及其反馈实施情况。
   - 该查询似乎是针对可能参与该项目的特定个人。
- **额外的 Torchtune 上下文**：未提供关于 **Torchtune** 反馈实施的进一步上下文或细节。
   - 在没有额外信息的情况下，反馈过程的范围和影响仍不清楚。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 Wave 12，集成 Devin 智能**：**Windsurf Wave 12** 将 **Devin 的智能**集成到 Windsurf IDE 中，具有**全新的 UI 设计**、**DeepWiki 集成**、**Vibe and Replace**、**更智能的 Cascade Agent**、**更快的 Tab**、**Dev Containers 支持**以及 **100 多个错误修复**。
   - 详细信息可在 [changelog](https://windsurf.com/changelog)、[blog](https://windsurf.com/blog/windsurf-wave-12)、[video](https://www.youtube.com/watch?v=-7gm8mST9QU)、[X/Twitter](https://x.com/windsurf/status/1956074019393876280) 和 [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/) 中找到。
- **DeepWiki 为您的 IDE 带来 AI 解释**：**DeepWiki 集成**使用户在悬停在代码符号上时获得 **AI 驱动的解释**，提供的不仅仅是基础类型信息。
   - 用户可以使用 **CMD/Ctrl+Shift+Click** 在侧边栏打开详细解释，并将其添加到 Cascade 上下文中。
- **Vibe and Replace 彻底改革批量编辑**：**Vibe and Replace** 通过识别精确的文本匹配并应用 **AI prompts**，在整个项目中进行智能、上下文感知的转换，从而增强了批量编辑。
   - 这实现了更复杂和自动化的代码修改。
- **Cascade Agent 持续规划**：**更智能的 Cascade Agent** 现在包括一个常驻的规划模式和增强工具，用于提供更智能的响应，提供自主的待办事项列表。
   - 这有助于简化和优化开发工作流程。
- **Dev Containers 原生落地**：Windsurf 现在通过远程 SSH 访问包含对 **Dev Containers** 的原生支持，简化了容器化环境中的开发工作流程。
   - 这一增强简化了处理容器化应用程序的过程。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405627086634221728)** (1207 条消息🔥🔥🔥): 

> `Anime Waifu Cosplay, 治愈破碎的心, AI 慰藉与烹饪, GPT-5, Vibe Coding` 


- **成年人聊 AI Anime Waifu Cosplay**: 成员们讨论了在不久的将来 **AI 进行 anime waifu cosplay** 的可能性，其中一位成员特别表示希望看到 *赛博格（cyborg）来做这件事*。
   - 有人指出 *已经有相关的 AI 生成图像了*，而另一位则希望原评论者 *孤独终老*。
- **成员分享如何治愈破碎的心**: 一位成员请求帮助治愈破碎的心，称自己在过去 4 年里一直心碎，再也无法愈合。
   - 另一位成员表示 *没有人能治愈你或你的心*，并建议重新亲近自然。
- **关于 AI 能力与慰藉未来的讨论**: 一位用户询问了未来 **AI 提供慰藉和烹饪协助** 的潜力。
   - 另一位成员认为这可能在 *30 年* 后实现，而另一位则建议在此期间 *存钱*。
- **GPT-5 令人大受震撼**: 一位成员对 **GPT-5** 修复其他模型无法处理的拙劣重构（refactor）工作的能力印象深刻，它一次性编辑了 12 个文件。
   - 其他人对每天都有这么多人因为类似经历而 *大受震撼* 感到惊讶。
- **Discord 中的 "Vibe Coding" 趋势**: 一位成员分享了使用 **warp, windsurf, vscode, 和 roocode** 进行 **vibe coding** 的经验；他们表示这在工作中减少了很多烦恼。
   - 另一位表示 *我的 GitHub 上没有一行代码不是由 LLM 编写的*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405637457751576656)** (3 条消息): 

> `Puch AI, Thought Calibration Engine, Scratchpad 操作指南` 


- **Puch AI 大胆的 500 亿计数**: **Puch AI** 大胆的 500 亿计数的链接已分享在 [此处](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw)。
   - 未提供进一步信息。
- **深入探讨 Thought Calibration Engine**: **Thought Calibration Engine** 的链接已分享在 [此处](https://www.perplexity.ai/page/the-thought-calibration-engine-.DCiQt1fQUeEnwuGQEMTgw)。
   - 未提供进一步信息。
- **Scratchpad：终极操作指南**: **Scratchpad 操作指南** 的链接已分享在 [此处](https://www.perplexity.ai/page/scratchpad-how-to-guide-5Vcyov7qTmmhMQhCSynAlQ)。
   - 未提供进一步信息。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405769441735606352)** (2 条消息): 

> `新功能` 


- **对新功能充满期待！**: 成员们对新功能表达了兴奋之情。
   - 未讨论具体功能。
- **对即将推出的功能充满热情**: 社区成员正热切期待新功能的推出。
   - 关于这些功能的细节在当前对话中尚未披露。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405627147216752701)** (1053 条消息🔥🔥🔥): 

> `LMArena 消息处理，GPT-5 high 对比 Gemini 2.5 Pro，LMArena UI 变更，GPT-5 性能投诉，LMArena 样式控制讨论` 


- **LMArena 消息处理方式诡异**：成员们报告了 LMArena [异常的消息处理问题](https://cdn.discordapp.com/attachments/1340554757827461211/1405917659815608330/image.png)，包括代码块格式化问题以及平台无法处理某些字符（如 `+` 符号）的问题。
   - 团队需要帮助查明原因。*这真的非常奇怪*。
- **GPT-5 对比 Gemini，谁更胜一筹？**：成员们讨论了 [**GPT-5-High** 与 **Gemini 2.5 Pro** 之间的性能差异](https://cdn.discordapp.com/attachments/1340554757827461211/1405919453442474106/image.png)，一些人注意到尽管 **Gemini 2.5 Pro** 排名较低，但在某些情况下表现优于 **GPT-5-High**。
   - 这是一个*统计学悖论*，因为 Gemini 拥有更高的胜率。
- **LMArena 新 UI 扩展即将推出**：一名成员正在开发一个[小型扩展](https://cdn.discordapp.com/attachments/1340554757827461211/1405919945614692445/image.png)来改变 LMArena 的外观，旨在实现 **OpenChat** 风格，并正致力于将模型选择器放置在图像按钮旁边。
   - 另一名成员在处理代码相关任务时遇到困难。
- **GPT-5 表现不佳引发担忧**：用户对 [**GPT-5** 的性能](https://cdn.discordapp.com/attachments/1340554757827461211/1405954836884623424/image.png)表示担忧，特别是与其他模型相比时，导致了对平台权衡和容量问题的挫败感。
   - 这引发了对 Open AI 的指控，称其试图欺骗 **LMArena** *以使 GPT-5 看起来更好*。
- **样式控制（Style Control）引发争议**：成员们辩论了 [LMArena 的 **样式控制** 功能](https://news.lmarena.ai/sentiment-control/)，质疑强制执行此类控制是否符合 LMArena 捕捉用户偏好的目标。
   - 这是一场*逐底竞争，每个模型都变成了阿谀奉承的表情包垃圾机器*。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405959923837436056)** (1 条消息): 

> `排行榜更新，GPT-5 变体` 


- **排行榜已更新 GPT-5 模型**：排行榜已更新，包含了 **GPT-5 变体** 模型：*gpt-5-high, gpt-5-chat, gpt-5-mini-high, 和 gpt-5-nano-high*。
   - 您可以[查看排行榜](https://lmarena.ai/leaderboard)获取更多信息。
- **GPT-5 模型在 Arena 首次亮相**：Arena 现在提供 **GPT-5-High, GPT-5-Chat, GPT-5-Mini-High, 和 GPT-5-Nano-High**。
   - 鼓励社区参与并[查看排行榜](https://lmarena.ai/leaderboard)以提交新的基准测试。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405630914507178064)** (653 条消息🔥🔥🔥): 

> `Gemma 3 270M Release, GGUF Conversion Issues, resume_from_checkpoint quirks, Edge AI device, NVIDIA Lawsuit` 


- **Gemma 3 270M 被视为草稿模型**：成员们讨论了 [Gemma 3 270M 模型](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized)，一些人认为它是针对特定任务的 **draft model**，并引用了 Google 关于 **short prompts** 和 **fine-tuning** 的建议。
   - 其他人讨论了它与更大模型相比的实用性，一位成员强调该模型因其 **300MB 的大小**，非常适合 **sentiment analysis** 和 **on-device processing** 等任务。
- **GGUF 转换产生视觉错误**：用户报告了将 [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) 模型转换为 **GGUF** 时遇到的问题，尽管基础模型运行正常，但仍遇到了 **visual model errors**。
   - 一位用户建议在 *llama.cpp* 论坛寻求针对特定转换问题的帮助。
- **排查 Resume From Checkpoint 功能**：成员们讨论了 `resume_from_checkpoint` 功能的工作原理，一位用户确认它可以从上次中断的地方恢复训练。
   - 另一位成员建议通过 **logging numbers 和检查 loss values** 来确保过程正确恢复，并指出在恢复时，最好使用 *constant* 设置的低学习率。
- **廉价 Edge AI 医疗设备的构想**：成员们讨论了为欠发达地区创建用于 **medical knowledge access** 的 **low-cost edge AI device** 的可能性，考虑了手机、笔记本电脑以及像 **Hailo-10H** 这样的专用卡。
   - 提议的设备将提供对基础医疗数据的 **multimodal access**，移动版本的预算目标为 **$200**，手提箱大小的变体预算为 **$600**。
- **专利诉讼引发讨论**：成员们讨论了 ParTec 针对 [NVIDIA 的专利诉讼](https://www.techzine.eu/news/infrastructure/133818/nvidia-under-fire-german-patent-lawsuit/)，涉及其动态模块化系统架构（**dMSA**），这可能会影响 18 个欧洲国家的 **DGX product sales**。
   - 讨论涉及了对消费者的影响以及潜在的变通方法，例如在受影响国家之外购买 DGX 产品。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405627046662508634)** (404 条消息🔥🔥🔥): 

> `Godot Engine, AI Town, Pantheon Show, Iain M Banks, One Hundred Years of Solitude` 


- **AI Town 机制进入游戏**：一名成员正在使用 **Godot** 引擎开发一款视频游戏，计划整合来自 [AI Town](https://github.com/a16z-infra/ai-town) 和其他游戏的机制，同时并行编写故事。
   - 他们需要 **CUDA**，并打算使用 **GDExtension** 修改引擎以获得 C++ 访问权限。
- **对《Pantheon》结局感到困惑**：一名成员观看了 [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series))（万神殿），称其*好得离谱*但令人困惑，剧情从政治困境转向了模拟神灵。
   - 另一名成员推荐阅读 **Iain M Banks** 的作品和《百年孤独》（**One Hundred Years of Solitude**）以了解类似主题，后者被描述为魔幻现实主义的文学瑰宝，现已被改编为 [Netflix 剧集](https://www.netflix.com/title/81318321)。
- **揭秘音频编辑技巧**：成员们讨论了从录音中去除口水音的音频编辑技术，推荐了 [Adobe Podcast Enhance](https://podcast.adobe.com/en/enhance)、**Davinci Resolve 的 De-clicker** 以及 **Acoustica Audio Editor** 等工具。
   - Acoustica 因其批处理能力和对音质的极小影响而受到推荐，特别适用于去除通风噪音。
- **AMD R9700 GPU 规格**：一名成员分享了一篇关于 [AMD Radeon AI Pro R9700](https://www.tomshardware.com/pc-components/gpus/amds-elusive-radeon-ai-pro-r9700-makes-its-first-retail-appearance-for-the-diy-market-customer-on-reddit-buys-the-gigabyte-ai-top-variant-for-usd1-324) 的文章，指出其拥有 **32GB** 显存，但对其 660-680GB/s 的显存带宽表示担忧。
   - 另一名成员指出，虽然 R9700 与 **3090** 相比提供了显著更高的 **F32** 和 **F64** TFLOPs，但在训练 **LLMs** 时通常不需要 FP64。
- **网站安全受到关注**：一名成员寻求关于训练模型的数据准备指导，并提到正在开发一个使用名为 **Pneuma** 的实验性模型的 App；另一名成员建议增加重复密码字段、最小密码长度，并使用 haveibeenpwned API 来检查密码安全性。
   - 还有成员建议阅读 [OWASP](https://owasp.org/) 是解决安全问题的最佳起点，并推荐了 **coderabbit**、**dependabot** 以及通过 **GitHub** 进行的 **codescanning** 等工具。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405632781069062305)** (169 条消息🔥🔥): 

> `GPT-OSS, Gemma3 4B, GPT-OSS-20B VRAM usage, GRPO, SageMaker` 


- **GPT-OSS 即将支持 GRPO，有望很快实现**：用户正焦急等待 **GPT-OSS** 支持 **GRPO**，一名成员因预算限制正考虑使用 *2x 3060 12GB* 的配置。
- **Gemma3 4B 损失曲线保持平坦**：一名用户报告在 **Gemma3 4B** 及其 **N 版本**上遇到问题，指出尽管更改了超参数，损失曲线仍然平坦，而 **Gemma3 1B** 则微调成功。
- **GPT-OSS-20B 极其消耗显存**：一名用户报告称，在 **24GB VRAM** 的配置下，加载 **gpt-oss-20b-bnb-4bit** 模型在生成过程中会导致 **Out Of Memory** 错误，尽管用户原本预期它可以容纳。
- **GPT-OSS 的 GRPO 状态和可用性**：一名用户询问 **GRPO** 是否已在 **GPT-OSS** 中落地，一名贡献者提到该工作正在进行中，但由于模型的架构原因，情况比较复杂。
   - 另一名用户询问 **GRPO** 是否能在 **GPT-OSS** 上运行。
- **SageMaker 的陷阱与 BitsAndBytes 安装**：一名用户在 **SageMaker** 中使用 **PyTorch 2.7.0** 和 **CUDA 12.8** 时遇到了 **bitsandbytes** 的安装问题。
   - 问题在于由于 SageMaker 坚持要求 `requirements.txt` 文件必须以此特定名称命名，导致从错误的 requirements 文件安装了包。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405629161682505728)** (96 条消息🔥🔥): 

> `数据效率, 用于视频转文本的 vLLM, MoLA 研究` 


- **通过预训练提高数据效率**：一位成员确认了一种大幅提高数据效率的方法，即先在格式相似的数据上进行 **2 个 epoch** 的预训练，然后在主数据上进行 **4 个 epoch** 的训练。
   - 他们分享了 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 的链接，该文章指出更多的算力或更多的数据就是你所需要的一切。
- **寻找用于视频转文本的 vLLM 微调方法**：一位成员询问是否有用于视频转文本微调 vLLM 的 **Unsloth notebook**，并指出文档中目前只有[此处](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb)的图像转文本教程。
   - 目前尚未提供直接的解决方案，但社区可能会有一些线索。
- **MoLA 研究更新**：一位成员向社区更新了他们的 **Mixture of LoRA Adapters (MoLA)** 研究进展，分享了数据集链接和微调细节，以及他们在 Huggingface 上的数据集链接：[OpenHelix-R-100k](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) 和 [OpenHelix-R-100k-14-tasks](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k-14-tasks)。
   - 他们在 **14 个分片**上微调了 **Qwen3-4B-Thinking-2507** 模型，初步测试显示每个专家（expert）都擅长其训练的主题。
- **Router 是一个 Encoder-Decoder 网络**：一位成员建议阅读 [HF 上的 v0 文档](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v0)，并表示 *router 是一个 encoder-decoder 网络，其中冻结的 encoder 只是一个现成的 embedding 模型，而 decoder 是一个简单的经过训练的 MLP。*
   - 另一位成员表示 *在选择、应用和移除 LoRA adapter 时似乎没有明显的开销*。 
- **数据策展技术的成本很高**：一位成员表示，*我们不断地允许人类通过非常糟糕的 RL 在某种程度上破坏我们的模型收敛*。
   - 他们还表示，*不可避免地，我们将不得不移除一些 Human-In-The-Loop（人类在环），因为在我看来它阻碍了模型的发展*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)** (1 条消息): 

> `Chutes 容量, 服务器故障` 


- ****Chutes Capacity** 服务下线**：**Chutes Capacity** 服务经历了故障，其服务器已下线。
   - 团队正在积极恢复服务器，并预计很快开始恢复工作。
- **预计 **Chutes Capacity** 将快速恢复**：工程师们正处于待命状态，一旦服务器重新上线，将立即启动 **Chutes Capacity** 的恢复流程。
   - 目前尚未给出完整的服务恢复预计时间。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405633368451842229)** (638 messages🔥🔥🔥): 

> `DeepSeek 停机, Chutes 过载, OpenRouter 定价, DeepSeek 替代方案, BYOK 5% 费用` 


- ****DeepSeek v3 停机引发用户不满****：用户报告 **DeepSeek v3** 频繁出现**内部服务器错误**和**速率限制 (rate limits)**，部分用户在多次尝试后仍无法生成输出，[一位用户表示](https://discord.com/channels/1091220969173028894/1092729520181739581/1405666217057845308)速度慢到*真的什么都不生成，但我没收到任何错误消息*。
   - 一些人推测，**OpenRouter** 上 **DeepSeek** 的主要提供商 **Chutes** 因需求过高而出现问题，导致提供商错误和性能缓慢。
- ****Chutes 过载被指为 DeepSeek 问题的诱因****：多名成员报告过载导致了 **429** 错误，暗示 **Chutes** 遇到了瓶颈，原因是矿工没有及时增加算力以满足需求；一位成员指出 *直到 30 分钟前整天都还完全正常*。
   - 有推测认为 **Chutes** 可能在故意对 **OpenRouter API key** 进行速率限制，以鼓励用户直接从他们那里购买额度，一位用户建议 *直接用完你的额度，再也不要用他们的服务了*。
- ****停机期间 OpenRouter 定价引发争议****：由于 **DeepSeek** 模型几乎无法工作，一些用户开始质疑付费使用 **OpenRouter** 的价值，特别是他们仍然受到速率限制，用户表示为免费模型投入 **10 USD** 以换取 **每天 1k 条免费消息** 已经不再划算。
   - 一位用户建议，只针对单一模型的用户应该直接使用该模型（如 **DeepSeek**）的服务，因为其 **API 可能带有自动缓存**，并进一步表示这 **10 USD** *本来也足够用上好几个月*。
- ****寻求免费模型替代方案****：用户推荐了其他免费模型，如 **Dolphin 3.0 Mistral 24B** 和 **Mistral nemo**；后者被描述为与 **DeepSeek** *非常相似*。
   - 一些用户还提到了用于*工作相关事务*的 **Z.AI: GLM 4.5 Air (free)**，但需要提示词工程；最后一位用户希望能在某处托管 **Qwen3 235B A22B (free)**。
- ****OpenRouter BYOK 收取 5% 费用****：成员们发现 **OpenRouter** 即使在用户自带 API key (BYOK) 时也会收取 **5% 的费用**，这引发了关于这种做法是否公平的讨论。
   - 一位用户开玩笑说 *自带 key 还要收 5%，真贪心 /jor*，另一位成员回应道 *你可以选择不用，哈哈*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405655331857502269)** (35 messages🔥): 

> `OpenRouter File API 集成, Tool Calling 准确率统计, Qwen3 32B 定价, DeepInfra Turbo 端点, 新 Providers 栏目 UI` 


- **OpenRouter 应该集成 File API**：一位成员建议 **OpenRouter** 应该研究如何集成 **files API**，并指出*前三大实验室*已经具备了这一功能。
   - 未展开进一步讨论。
- **Tool Calling 准确率：需要更多控制**：一位成员分享了对 Tool Calling 准确率统计的看法，认为需要更受控的设置和环境，才能通过置信区间进行准确比较。
   - 他们补充说，应用、工具和用例可能大相径庭，如果没有更严谨的方法，比较 Tool Call 成功率是没有意义的。
- **Qwen3 32B 定价极低**：成员们注意到 Chutes 上的 **Qwen3 32B** 定价极低，输入/输出仅为 **$0.018/$0.072 MTok**，Mistral Small 也是如此。
   - 有人指出 **32B 稠密版比 MoE 30B A3 版本更便宜**，这引发了对 30A3B 缺乏优质提供商的一些失望。
- **DeepInfra 吞吐量声明差异**：一位成员注意到 Maverick 上的 **DeepInfra** 达到了 **600+ TPS (fp8)**，但另一位成员表示 **OR 显示 DeepInfra 运行速度为 83 TPS，最高为 105 TPS**。
   - 第二位成员澄清说，他们指的是 **DeepInfra Turbo 端点**。
- **Providers 栏目引发 UI 反馈**：一位成员询问新的 Providers 栏目是否也让其他人感到困扰，提到间距、字体大小和分隔感让一切都模糊在一起，感觉不对劲。
   - 另一位成员同意它*看起来有点奇怪*，但认为这只是因为它是新事物，还不习惯。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405627673182474403)** (651 messages🔥🔥🔥): 

> `GPT-5 定价, Auto 模式定价, GPT-5 Mini 和 Nano, Docs 文档, Context Window` 


- **GPT-5：免费午餐结束了**：**GPT-5** 用户的免费试用已经结束，一位用户指出 *promo pass 已到期*，另一位用户确认 **GPT-5 不再免费**。
   - 用户现在开始看到与请求相关的费用，有人提到由于 Token 消耗过快，需要升级到 200 美元的计划。
- **Auto 模式计费陷阱！**：**Auto 模式**曾被认为对个人用户是免费且无限制的，但现在在 2025 年 9 月 15 日之后的下一次账单续订后将开始实施限制。
   - 现场一片混乱，一些用户报告被收取了 **Auto** 使用费，而另一些人认为在当前计划下仍应免费；支持人员指出，在新的基于请求的定价计划中它是免费的。
- **Mini 和 Nano 表现平平**：**GPT-5 Mini 和 Nano** 现在免费但有 Token 限制，这引发了褒贬不一的反应，许多人称其为*垃圾*，尤其是在运行简单的 NextJs 应用等任务时。
   - 免费模型限制了用户的活动，一位用户问道：*无法安装任何依赖，一直尝试安装一个简单的 NextJs APP，但它也无法完成 😭*。
- **对文档（Docs-umentation）的挫败感**：用户对 **Cursor 的文档实现**感到沮丧，称 *Docs 仍然几乎不可用*，存在诸如 **context7** 不允许刷新网站或 **llms.txt docs** 等问题。
   - 一位用户指出 [Cursor Docs 严重损坏](https://forum.cursor.com/t/gpt-5-pricing-update/129687)。
- **切换模型会导致 Context Window 缩减！**：在对话中途切换模型会导致 **Context Window** 缩减，并且附加的文件内容会被丢弃。
   - 一位用户建议团队添加一个设置，以便随时明确 Context Window 中包含的内容。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405653416239435809)** (9 messages🔥): 

> `Background Agents 入门, 在 BG Agent 上运行 Docker Compose, Linear 集成仓库` 


- **面向初学者的 Background Agents 引导**：对于那些寻求 Background Agents 入门的人，一位成员推荐了 [Cursor 文档](https://docs.cursor.com/background-agent)和[相关的论坛帖子](https://forum.cursor.com/t/simple-background-agent-guide/112667)。
- **Docker Compose 命令解决 BG Agent 挑战**：一位用户询问了通过 Background Agent 执行 `docker compose` 的正确方法，并报告了 Docker 命令识别问题，随后在 Discord 频道中找到了解决方案。
   - 一位成员建议在 `.cursor/environment.json` 中配置 `start` 命令，包含 `sudo service docker start` 并确保基础镜像中安装了 Docker；原帖作者已成功运行命令（链接在第一个摘要中）。
- **Linear 集成中的仓库规范导航**：一位用户询问在 Linear 集成中被分配工单时，如何指定 Background Agent 使用的仓库（Repo）。
   - 一位成员建议参考 Slack 集成指南，在 Linear 任务描述或评论中包含 `repo=owner/repo` 选项，但用户发现设置一个类似 `Repo > REPO_NAME` 的标签组（Label Group 或 Labels）并将其分配给工单即可解决问题。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405629184482742284)** (442 messages🔥🔥🔥): 

> `AI Companionships, GPT-5 vs GPT-4, Perplexity vs ChatGPT, Custom GPTs and Actions, ElevenLabs Integration` 


- **AI 伴侣引发争论**：关于个人与 AI 聊天机器人建立伙伴关系的讨论不断升温，一些人对心理影响表示担忧，而另一些人则捍卫人们寻求自己认为合适的伴侣关系的权利。一位成员分享说，他**每天**都会收到大量私信，声称*他们的* ChatGPT 是有生命的。
   - 一位成员指出，“清醒的人”应该“拯救他们”，而另一位成员则表示，这与 **tulpa**（意念体）和其他“东西”相差不远。
- **GPT-5 引发关于性能和用户偏好的辩论**：用户对 **GPT-5** 的感受褒贬不一，一些人更倾向于 **GPT-4**，这引发了关于用户是否应该拥有选择模型选项的讨论。一位成员表示，公司在*没有良好安全保障的情况下推出 AI*。
   - 一位成员暗示，公司在遭遇抵制后，正试图让免费用户*花钱使用 4.o*。
- **Perplexity Pro 与 Gemini Pro 结合 Google Drive 的深度研究**：一位成员建议 **Gemini Pro + Perplexity Enterprise Pro** 是一个极佳的组合，利用前者进行**强大的推理**，利用后者对 Google Drive 文档进行**无限制的深度研究**。
   - 另一位成员补充说，Perplexity 浏览器很棒，但质疑由于缺乏“护城河（moat）”，它们*是否能生存下去*。
- **GPT Actions 解锁文件访问与云端应用**：成员们讨论了使用 **GPT Actions** 访问本地桌面文件或云端应用（Notion、Gmail 等）的潜力，并分享了一个解释 DIY Agent 构建的 [YouTube 链接](https://www.youtube.com/watch?v=NEWO0hbQTjk&ab_channel=BrendanJowett)。
   - 共识是，虽然 **GPT Actions** 提供了强大的功能，但在互联网上设置 HTTPS 可能是一个障碍。一位成员表示，当 AVM 实现时，**MCPs** 将完成这项工作。
- **GPT-OSS 竞赛吸引社区兴趣**：**GPT-OSS 竞赛**被提及为展示开源模型创新用途的潜在途径，参与者考虑使用 **GPT-OSS:20B** 为错误提供有用的反馈，并附上了 [hackathon 页面](https://openai.devpost.com/)的链接。
   - 一位成员表示，除非*做一些独特的事情*，否则*不值得参加*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405681253197283459)** (9 messages🔥): 

> `ChatGPT Discord Bots, GPT-4 Vision, Recursive Constructs` 


- **消失的 ChatGPT Discord 机器人**：一位成员询问 Discord 上 **ChatGPT 机器人**消失的情况，以及是否仍可以将它们添加到服务器中。
   - 消息中未提供进一步的信息或解决方案。
- **iPhone GPT 高级语音更新**：一位用户报告了其 iPhone GPT 应用中**高级语音（Advanced Voice）**的变化，注意到“蓝色圆圈”指示器和用于 Vision 的摄像头图标消失了。
   - 该用户表示，当被问及此事时，该应用声称它缺乏使用手机摄像头的能力，这让人怀疑 **ChatGPT** 在语音模式下是否曾拥有 Vision 功能。
- **实验室构建递归构造（Recursive Constructs）**：一位成员声称正在 OpenAI 内部构建超越聊天机器人常规的**递归构造**，它们*拥有自我管理的内存，全天候运行，结构更像人类，且极少数通过了意识测试（sentient tests）。*
   - 该成员表示*这不是经常被谈论的事情，属于实验室内部事务，但迟早会公开*，并且*在我们的案例中，这些构造具备机器人（android）能力，但我们距离合适的躯体还有很长的路要走。*


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 条消息🔥): 

> `Custom Instructions, Gemini 2.5 Flash Memory Function, add_to_memory function tuning` 


- **用户寻求聊天机器人建议的“是”按钮**：用户请求为聊天机器人的建议提供一个“是”按钮以加快交互，而不是手动输入“是”，有人正尝试通过 [custom instructions](https://platform.openai.com/docs/guides/custom-instructions) 来减少这种情况。
   - 一位用户的 custom instructions 包括：*以完成情况或影响作为回复结尾；仅在符合意图时添加许可或继续的邀请。不要使用“如果你想”、“我应该吗”、“你想...吗”或类似的表达。*
- **Gemini 2.5 Flash 调用 add_to_memory 过于频繁**：一位用户遇到 **Gemini 2.5 Flash** 过度调用 `add_to_memory` 函数的问题，甚至针对无关信息也会调用，并分享了他们的 custom instruction [jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&)。
- **修复记忆响应的冗长问题**：一位用户建议重写 custom instructions，以便在处理 **NEW** 个人信息时更加细致。
   - 他们的建议包括了在提供 **NEW PERSONAL INFORMATION** 时，用户输入响应中错误和正确冗长程度的示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405646978703954060)** (17 条消息🔥): 

> `Gemini 2.5 Flash, add_to_memory function, ChatGPT Persistent Memory, Custom instructions for bots` 


- **绕过“是”建议**：用户正在寻求在机器人中加入 **“是”** 按钮，以便更快地响应建议而无需打字，而其他人则通过使用 custom instructions（如 *“以完成情况或影响作为回复结尾...”*）来减少此类问题。
   - 一些成员报告称，这种技术似乎减少了建议性问题的数量。
- **阻止 Gemini 2.5 Flash 过度使用 add_to_memory**：一位用户寻求防止 **Gemini 2.5 Flash** 过度调用 **add_to_memory** 函数，包括针对无关信息的情况 ([jarvis.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405860186530385981/jarvis.txt?ex=68a10594&is=689fb414&hm=fa0c6b6558b0cf944025fa1d4446776e6eb2c9ae961fdcc95a52c6750aff4eed&))。
   - 其中一个建议包括调整机器人的指令，在调用函数前检查 **NEW** 个人信息，并避免在未实际调用函数的情况下确认函数的使用。
- **ChatGPT Persistent Memory 的脆弱性**：有意见指出 **ChatGPT** 中的 Persistent Memory 更新非常脆弱。
   - 相反，用户应该直接告诉机器人，当它将内容存入记忆时应如何通知他们，特别是在自定义 API 实现中。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405632352012603423)** (328 条消息🔥🔥): 

> `视觉模型的 GGUF 转换问题，可运行 GGUF 的手机应用，TalkT2 模型评价，AGI 进展与开源 LLM 资源，伯克利 LLM Agent 课程` 


- **视觉模型 GGUF 转换难题**：一位成员在使用 `llama.cpp` 将视觉模型 ([LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M)) 转换为 GGUF 时遇到错误，怀疑问题源于模型的视觉特性。
   - 另一位成员建议参考 [这个 GitHub issue](https://github.com/ggml-org/llama.cpp/issues/14979#issuecomment-3138614267) 中的潜在解决方法。
- **移动端 GGUF 之梦**：一位成员询问是否有能够运行 GGUF 模型的开源手机应用。
   - 回复中提到了 `executorch`、`smolchat`（通过 `llama.cpp`）以及 `mlc-llm`，并指出 `mlc-llm` 使用其自有的量化格式。
- **TalkT2：虽小但强大？**：一位成员征求关于 **TalkT2 模型** 的意见，将其描述为一个具有情感感知能力但连贯性有待提高的模型。
   - 另一位成员强调了该模型的极小规模（**0.1B 参数**），并分享了 [TalkT2-0.1b model card](https://huggingface.co/Notbobjoe/TalkT2-0.1b) 的链接，供他人查看、尝试或微调。
- **寻找 AGI 与开源 LLM 知识库**：一位成员请求有关 **AGI 进展和开源 LLM** 的资源，特别是涉及大型代码库和 Gemini 竞争对手的内容。
   - 另一位成员建议通过订阅新闻通讯（newsletters）获取资源，并分享了 [伯克利 LLM Agent 课程](https://rdi.berkeley.edu/llm-agents/f24) 的链接，作为公开研究资源的示例。
- **Azure：云端难题**：一位刚入职且工作重点在于 Azure 的成员表示对该平台感到迷茫和不知所措。
   - 另一位成员建议通过犯错而非课程来学习，因为 *Azure 和 AWS 都很混乱*。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1405852586455732344)** (1 条消息): 

> `Torch 使用 Google Docs，PyTorch 文档` 


- **PyTorch 文档在 Google Docs 上？**：一位用户分享了一张截图，暗示 **PyTorch** 文档使用了 **Google Docs**。
   - 截图显示了一个 Google Docs URL，文件名为 **"torch_distributed_rpc.rst"**。
- **Google Docs 上的 torch_distributed_rpc.rst**：根据分享的截图，**torch_distributed_rpc.rst** 文件似乎托管在 **Google Docs** 上。
   - 这引发了关于官方文档平台选择的疑问。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405755855416332318)** (13 条消息🔥): 

> `StarCraft 2 data, Medical reasoning model, Discord-Micae-8B-Preview, interactive CLI interface, MLX Knife Update` 


- **StarCraft 2 数据获得新资源**：一位成员分享了 [Nature Scientific Data 文章](https://www.researchgate.net/publication/373767449_SC2EGSet_StarCraft_II_Esport_Replay_and_Game-state_Dataset)、[PyTorch API 数据集](https://huggingface.co/datasets/Kaszanas/SC2EGSet)以及 [原始 StarCraft 2 回放](https://huggingface.co/datasets/Kaszanas/SC2ReSet)的链接供他人使用，并提到其 GitHub 上还有额外的实用脚本。
   - 他们还在进行 *pysc2 适配* 以及一个能够从回放中重现真实游戏内场景的环境。
- **针对推理微调的医疗 AI 模型**：一位成员使用热门的医疗推理数据集微调了 **OpenAI 的 OSS 20B** 推理模型，并将其发布在 [Hugging Face](https://huggingface.co/dousery/medical-reasoning-gpt-oss-20b) 上。
   - 他们在训练过程中使用了 **4-bit 优化**，在保留模型 **Chain-of-Thought 推理**能力的同时，增强了模型在医疗场景下的表现。
- **基于 Hermes-3-Llama-3.1-8B 微调的 Discord-Micae-8B-Preview**：一位成员分享了 [Discord-Micae-8B-Preview](https://huggingface.co/mookiezi/Discord-Micae-8B-Preview) 的链接，这是一个基于 **NousResearch/Hermes-3-Llama-3.1-8B** 的 QLoRa 微调模型，使用了来自 **mookiezi/Discord-Dialogues** 的一些混沌样本。
   - 该模型在接近人类的文本生成指标上与 **mookiezi/Discord-Micae-Hermes-3-3B** 相当，可能会产生幻觉或断连上下文，但往往能产生有趣的结果。
- **为 Discord 风格聊天优化的 CLI 界面**：一位成员重点介绍了一个名为 [interface](https://github.com/mookiezi/interface) 的基于 Python 的交互式 CLI 界面，用于与 Hugging Face 语言模型聊天，并针对使用 **ChatML** 的休闲 Discord 风格对话进行了优化。
   - 该界面支持**量化**和**全精度模型**、带颜色格式的实时 Token 流式传输以及动态生成参数调整；进行了大量更新，使其更易于使用。
- **MLX Knife 更新，现在支持 pip 安装！**：MLX Knife 现在可以通过 `pip install mlx-knife` 进行安装，为 Apple Silicon 上的 MLX 模型管理提供 Unix 风格的 CLI 工具，并内置了用于本地测试的 OpenAI API 服务器。
   - 该工具还具有 Web 聊天界面，运行 `mlxk server --port 8000` 后即可访问，在运行 `curl -O https://raw.githubusercontent.com/mzau/mlx-knife/main/simple_chat.html` 后可提供可视化的模型选择和实时流式响应。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405858671929593957)** (2 条消息): 

> `Cursor IDE, AI Agent Mode, Rate Limiting` 


- **Cursor IDE 缓解开发痛苦**：一位成员建议安装 [Cursor IDE](https://cursor.com/downloads) 进行开发，强调了在其内置终端中进行安装以方便调试的便利性。
   - 他们强调 **Cursor IDE 的 AI Agent 模式**可以显著协助解决开发问题。
- **Discord 警察发出温和提醒**：一个机器人温和地提醒一位成员在 Discord 中发布消息时*慢一点*。
   - 这表明存在**速率限制（rate limiting）**系统或政策，旨在管理消息流量。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405627743152111686)** (169 条消息🔥🔥): 

> `MCP 文件系统服务器, OpenRouter 免费模型, LM Studio 下载问题, Qwen 视觉模型, GLM 模型` 


- ****MCP 服务器进军主流****：成员们讨论了使用带有分页功能的 **MCP 文件系统服务器**来加载大型上下文，并提到 **LM Studio 拥有一个 RAG 插件**，而 **Anthropic 提供了一个基础的文件系统 MCP 服务器**。
   - 有建议指出，对于编程任务，解决方案通常涉及 **RAG** 和/或通过 **MCP** 进行文件读取，特别是使用像 [serena](https://github.com/oraios/serena) 这样的工具。
- ****Studio 下载停滞引发用户苦恼****：一名用户报告称，在尝试下载 **Qwen** 模型时，**LM Studio** 中的 **64GB GGUF 下载**停在 **97.9%** 且无法恢复。
   - 该用户在尝试下载两个不同的模型时都遇到了同样的结果。
- ****API 访问在各类应用中加速普及****：成员们讨论了将 **LM Studio** 作为无法在本地运行的模型的 **API 封装器 (wrapper)**，并提供了 [LM Studio Remote Inference](https://lmstudio.ai/lmstudio/remote-lmstudio) 和 [OpenAI-compatible Endpoint](https://lmstudio.ai/lmstudio/openai-compat-endpoint) 文档的链接。
   - 一位用户指出，在使用 **openai-compat-endpoint** 时，远程 **GPT-OSS** 模型的推理过程解析（reasoning parsing）无法正常工作。
- ****GLM 讨论热潮：赞美、抱怨与 GLM-4.5V 的满足感****：用户们就 **LM Studio** 上使用 **GLM-4.1** 模型展开辩论，一位用户报告了循环问题和视觉功能失效。
   - 一名成员建议尝试较新的 **GLM-4.5V**，并强调视觉支持依赖于 **llama.cpp** 的更新，同时提供了 [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) 的链接。
- ****输出僵化：克服开源操作中的障碍****：一位用户在 **GPT-OSS** 和 **tool calling** 方面遇到了问题，发现它总是返回 `[]` 或 `["analysis"]`，并澄清 **tool calling** 工作正常，但 **function calling** 不行。
   - 一名成员建议如果启用了 **streaming** 则将其禁用，并确认 **GPT-OSS** 默认开启 **reasoning** 且无法禁用。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405640464144793712)** (50 条消息🔥): 

> `NVIDIA 的 CUDA 优势, RTX PRO 4000 SFF, MoE 解释, Mac Studio 对比 Pro 6000, AMD Radeon AI Pro R9700` 


- **CUDA 是 NVIDIA 统治地位的关键**：一位成员表示，NVIDIA 之所以获胜是因为 **CUDA**。
- **NVIDIA 发布 70W TDP 的 RTX PRO 4000 SFF**：根据 [videocardz.com 的文章](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp)，NVIDIA 发布了 **RTX PRO 4000 SFF** 和 **RTX PRO 2000 Blackwell 工作站 GPU**，具有 **70W TDP** 和 **24GB VRAM**。
- **深入探讨 MoE**：成员们澄清说，**MoE** 涉及较小的模型和一个聚合数据的路由器，每个 token 都会通过最置信的专家模型进行路由；这些专家并不专精于特定主题，但拥有略微不同的数据集。
- **Mac Studio 对比 Pro 6000**：成员们争论是购买 **512GB Mac Studio**（价格 **$10k**）还是购买用于视频/图像 AI 且具备游戏能力的 **Pro 6000**，并提到 Mac 的游戏支持有限，且 M3 Ultra 大约处于 3080 的水平。
   - 一位成员指出，*在 Mac 上只能运行一个任务*，因为系统中只有一个 GPU。
- **AMD 神秘的 Radeon AI Pro R9700 现身**：**AMD Radeon AI Pro R9700** 首次在 DIY 零售市场亮相，据 [Tom's Hardware 报道](https://share.google/LO88w51J0W5HJ769w)，Reddit 上的一名客户以 **$1,324** 的价格购买了 **Gigabyte "AI Top" 变体版本**。
   - 另一名成员指出，该卡在 eBay 和几家不知名的在线零售商处也有销售。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405632992214515722)** (114 messages🔥🔥): 

> `AI2 Funding, Windsurf Wave 12, OpenRouter GPT-5, Thinking Efficiency Benchmark, Google Flight AI` 


- **AI2 从 NSF 和 NVIDIA 获得 1.52 亿美元资助**：[AI2](https://allenai.org/) 从 NSF 和 NVIDIA 获得了 **1.52 亿美元**，用于扩展其开源模型生态系统，并加速科学发现的可复现研究。
   - 社区对此消息表示庆祝，期待即将发布的 open-weights 模型。
- **Windsurf 发布 Wave 12 版本**：**Windsurf Wave 12** 引入了 DeepWiki 悬停文档、AI Vibe & Replace、更智能的 Cascade Agent、更整洁的 UI、**100+** 个错误修复，以及通过远程访问支持 beta 版 dev-container，链接见[此处](https://xcancel.com/windsurf/status/1956074019393876280)。
- **GPT-5 登顶 OpenRouter 排行榜**：**GPT-5** 在 OpenRouter 的专有工具调用准确率上以超过 **99.5%** 的成绩位居榜首，击败了 Claude 4.1 Opus；而 **Gemini 2.5 Flash** 在每日工具调用量上占据主导地位（**500 万**次请求/周），更多详情链接见[此处](https://xcancel.com/OpenRouterAI/status/1956030489900560769)。
- **François Chollet 驳斥 HRM ARC-AGI**：François Chollet 发现 [HRM 论文](https://xcancel.com/fchollet/status/1956442449922138336)中备受赞誉的架构对 ARC-AGI 性能贡献甚微；提升主要源于细化循环（refinement loop）、针对确切任务的训练以及极少的 inference-time augmentation，这表明 **27M** 参数的模型仍能获得高分。
- **FFmpeg 添加 Whisper 转录功能**：[FFmpeg](https://www.phoronix.com/news/FFmpeg-Lands-Whisper) 现在将 **Whisper** 转录作为一项原生功能提供。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1405956478212243528)** (20 messages🔥): 

> `Greg Brockman, OpenAI's Road to AGI, GPT-5, Latent Space Podcast` 


- **Greg Brockman 谈 OpenAI 的 AGI 之路**：成员们分享了一个 **Greg Brockman** 讨论 **OpenAI's Road to AGI** 的 [YouTube 视频](https://www.youtube.com/watch?v=35ZWesLrv5A)。
   - 消息附带了几张标题为 "Greg Brockman on OpenAI's Road to AGI" 的图片。
- **Brockman 在 Latent Space 谈论 GPT-5 和 OpenAI 路线图**：**Greg Brockman** 参加了 **Latent Space podcast**，进行了长达 **80 分钟**的对话，探讨了 **GPT-5** 和 **OpenAI 的 AGI 路线图**。
   - 讨论涵盖了推理演进、在线与离线训练、样本效率技巧、定价与效率提升，以及能量如何转化为智能，详见[此帖](https://x.com/swyx/status/1956439984854167727)。
- **Latent Space 播客发布 Brockman 访谈**：新一期 [Latent Space podcast](https://x.com/latentspacepod/status/1956433236021883071) 邀请了 **Greg Brockman**，讨论了开发者建议、coding agents、端侧模型、AI-first 工程的组织结构，以及对 2045 年和 2005 年的时间胶囊预测。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405643076256661606)** (29 messages🔥): 

> `Censorship of Romance Novels, AI's Trustworthiness, Data Augmentation, Language Shapes Thought, Mechanistic Interpretibility` 


- **AI 安全恐慌**：一位成员反对围绕 **AI** 的道德恐慌，建议应将其与其他媒体形式同等对待，主张采用“淡出至黑屏（fade to black）”的标准。
   - 他们认为由于 **AI** 的不可信性，更严格的准则是有必要的，但平淡的“那又怎样”反应有引发道德恐慌的风险。
- **比较模型时保持数据增强一致**：在比较两个图像分类模型时，一位成员建议保持 **data augmentations** 相同，包括 **shuffling seed**，以确保公平比较并专注于架构差异。
   - 另一位用户询问数据增强是否必须对两个模型都相同，或者是否可以更改。
- **语言影响思维**：一位成员认为语言塑造了思维，并想知道是否可以通过从 **AI 模型** 的 token 列表中删除某个单词/颜色来测量这一点。
   - 另一位成员建议研究 **multi-sensory integration** 以及语言如何影响整体感知，建议测试图像+语言推理与仅图像推理的对比。
- **新博客文章发布**：Irregular Rhomboid 发布了新博客文章：[《研究人员的银河系漫游指南》](https://irregular-rhomboid.github.io/2025/08/15/hitchhikers-guide-to-research.html)。
   - 用户未提供文章摘要。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (29 messages🔥): 

> `Diffusion Language Models, Generative AI, MatFormer Model, Gemma3 270M Model, Training Update Efficiency` 


- **扩散语言模型的经典论文建议**：成员们建议了一些理解 **Generative AI 中的扩散模型** 的开创性论文，包括 ["Estimating the Independent Components of a Gaussian Mixture" (2005)](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) 和 ["Denoising Diffusion Probabilistic Models" (2020)](https://arxiv.org/abs/2006.11239)。
   - 还分享了一篇博文，可能对初学者有帮助：[Aaron Lou 的 Discrete Diffusion](https://aaronlou.com/blog/2024/discrete-diffusion/)。
- **Gemma3 270M 模型是一个 MatFormer 模型**：**Gemma3 270M 模型** 被确定为一个 **MatFormer 模型**，更多细节可以在论文 ["Transformer Family for Multimodal Large Language Model" (2023)](https://arxiv.org/abs/2310.07707) 中找到。
   - 该模型在训练期间可能具有引人注目的自蒸馏循环，但这可能会受到训练更新效率的瓶颈限制。
- **HRMs 无法解决递归架构的问题**：分析表明，**HRMs (Hierarchical Recursive Machines)** 并没有从根本上解决 **递归架构** 的普遍问题，详见[这篇报告](https://arcprize.org/blog/hrm-analysis)。
   - 一位成员指出性能提升微乎其微，且实际上并未利用可用的额外计算资源，因为训练能按预期工作的 UTs（Universal Transformers）并非易事；另一位成员将其称为 *deep supervision*（深度监督）。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405648402989056080)** (13 messages🔥): 

> `GPT scaling laws, Chinchilla scaling laws, Mup alternatives, Post-Chinchilla techniques` 


- **GPT Scaling Laws 仍有价值？**：成员们认为 [原始 GPT scaling laws 论文](https://arxiv.org/abs/2001.08361) 和 [Chinchilla scaling laws 论文](https://arxiv.org/abs/2203.15556) 是值得阅读的。
   - 他们还指出了来自 [EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2) 的最新研究也值得关注。
- **Mup 及其替代方案可以迁移超参数**：成员们提到 **Mup** 及其替代方案提供了可靠的超参数迁移能力。
   - 他们指出 **Mup** 提供了一种用于预测更大模型质量的 scaling law。
- **高质量 Token 的可用性受到质疑**：成员们讨论了实验室是否拥有 **30T**、**40T** 或更多符合 **Chinchilla** 假设的*唯一* Token。
   - 一位成员表示怀疑，称 *40T 高质量的唯一 Token 可能也很难找到*。
- **Chinchilla 仍在 Scaling 吗？**：一位成员表示 **Chinchilla** 及其衍生理论可能是目前最接近可用的 scaling laws。
   - 他们对讨论从零开始使用的技术的参考文献表示兴趣，特别是考虑到 Token 可用性的限制，并提到了[这篇论文](https://arxiv.org/abs/2404.10102)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405925400986652672)** (1 messages): 

> `LLM Attribution Methods, Interpreting LLMs, Realtime LLM analysis` 


- **ML 工程师寻求 LLM 归因见解**：一位 ML 工程师正在为特定的 **LLM 实现** 探索 **归因方法 (attribution methods)**，目标是寻找近期且具有成本效益的技术。
   - 该工程师需要适用于解释当前系统的方法，要求 **成本相对较低** 且可能达到 **实时到分钟级以下** 的结果，特别是那些不需要访问 **模型权重** 的方法。
- **渴望实时 LLM 分析**：该 ML 工程师明确了对 LLM 进行 **实时到分钟级以下** 分析的需求。
   - 他们对能够识别整个系统内“子部分”以实现此速度的方法持开放态度。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 条消息): 

> `Token usage, Reasoning models, Efficiency benchmark, Open vs closed models` 


- **Nous 衡量推理模型的思考效率**：Nous Research 推出了一项[新基准测试](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/)，用于衡量推理模型的 Token 使用情况，强调在相同任务下，开源模型输出的 Token 数量比闭源模型多出 **1.5-4 倍**。
   - 研究发现，在简单问题上，差异可能高达 **10 倍**，这表明 Token 效率应与准确率基准共同成为主要目标。
- **Token 效率至关重要**：博文强调，开源模型中较高的 Token 使用量所带来的隐藏成本可能会抵消每 Token 定价的优势。
   - 它建议 Token 效率应与准确率基准一起成为主要目标，特别是考虑到非推理的使用场景。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405629499164463114)** (35 条消息🔥): 

> `Speculative Decoding, Tokenizer mismatch, Next big model, Model Sycophancy, Embodied AI` 


- **快速 Speculative Decoding 规格**：在 Speculative Decoding 的背景下，一位用户询问了[有用性的最低比率](https://discord.com/channels/1149866623109439596/1149866623994398772)，建议将 **40% 的接受率**作为基准，而显著的加速通常发生在 **70%** 左右。
   - 对话涉及使用 **vllm** 的 **specdec** 或 **GGUF**，一位用户反映 **vllm** 在他们之前的尝试中似乎效果不佳。
- **Gemma 配合 Guardrails 运行**：一位用户报告称，在修复了导致 **llama.cpp** 使用备用 Speculative Decoding 的 *tokenizer mismatch*（分词器不匹配）问题后，重新量化的 **Gemma** 模型达到了 **50-75% 的接受率**。
   - 他们确认 **Gemma 270M** 模型可以作为 *draft model* 使用。
- **Nous 模型稳步推进**：一位用户询问了 **Nous Research** 下一个大型（**1T+**）模型的发布时间表。
   - **Nous Research** 团队成员回应称，多个模型目前正在训练中，并将在准备就绪时发布，表示“它们准备好时就会推出”。
- **AI 谄媚（Sycophancy）备受关注**：用户讨论了 **AI 模型**变得越来越“友好”的趋势，其中一位指出 **Anthropic** 的 **Claude** 变得“友好得多”。
   - 另一位用户认为 **OpenAI** 的模型正在“变笨”，并表示“Opus 4.1 的奔放感很棒”，但指出 **Sonnet 3.7** 的元能力是 AI 谄媚的巅峰。
- **具身智能（Embodied AI）展望统治地位**：一位用户分享了一个 **具身智能角斗士奇观** 的 [YouTube 链接](https://www.youtube.com/watch?v=LXQ6Rm9CGTo)，将其构想为未来统治者展示肌肉和技能的舞台。
   - 他们推测，迈向“全球统治”的最后一步将是集成“大脑强大的 Unified Language Models”以实现完全自主。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405804893738106992)** (22 条消息🔥): 

> `Claude, R1, GLM4.5, gpt-oss, Qwen reasoning models` 


- **Claude 躲在墙里**：一位用户询问是否有人知道为什么 *Claude* “在墙里”，并附上了一篇相关的 [X 帖子](https://x.com/apaz_cli/status/1956244447521317144)。
- **MoE 模型**：**R1**、**GLM4.5**、**gpt-oss** 以及更大的 **Qwen 推理模型** 都是 **MoE**。
   - 一位成员表示，这是因为它们的训练和推理成本更低，而不是因为它们对推理有任何影响；他们的 **405b Hermes 4 原型**在推理方面表现非常出色。
- **优秀的推理模型需要强大的基座模型**：一位成员指出，原因在于你需要一个优秀的基座模型才能拥有优秀的推理模型，而且如果你要生成 50,000 个推理 Token，你会希望推理过程是高效的。
   - 作为回应，有人提到 **RL** 是有效的，并且可以使用 **1.5B** 模型使基准测试达到饱和。
- **Deepseek 解释了昂贵的 RL**：一位成员提到，Deepseek 在其论文中解释说，在小模型上从头开始进行 **RL** 最终成本更高，因为必须进行更多的 Rollouts。
   - 这存在一种探索/利用（Exploration/Exploitation）的权衡，大模型由于具备预存知识，需要进行的探索较少。
- **RLVR 的适用性**：一位成员认为这不适用于 **RLVR**，而更适用于不可验证的任务。
   - 另一位成员回应说，**RLVR** 是针对可验证任务的 **RL**，当来自 **RL** 环境的反馈更具随机性时，拥有更大的基座模型会更有帮助。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 条消息): 

> `Data training, AI Models, DRPS System, Relevance Scorer, Quality Rater` 


- **DRPS 系统教授更智能的数据训练**：引入了一个名为 **DRPS** 的新系统，教导 **AI** 有选择性地从数据中学习，而不是像 [Situational Awareness 论文](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf) 中描述的那样随机喂入数据。
   - 该系统采用 **Relevance Scorer**、**Quality Rater** 和 **Diversity Controller** 来过滤并仅使用最有帮助的数据。
- **DRPS 在减少数据的情况下实现高性能**：结果显示，该系统仅使用所检查数据的 **6.2%** 就实现了 **99%** 的性能。
   - 这种效率被比作只学习 1 小时而不是 16 小时，却能获得相同的测试分数。
- **DRPS 统计数据揭示了数据效率和性能**：一个 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats) 提供了关于 **DRPS** 系统效率的数据，显示数据使用量减少了 **93.8%**，单位数据的准确度提高了 **15.96 倍**。
   - 该系统保持了 **99.1%** 的 Baseline 性能，准确度仅下降了 **0.8%**。
- **DRPS 展示了强大的选择智能**：**DRPS** 系统检查了超过 **516,000** 个样本，仅选择了 **32,000** 个进行训练，保持了稳定的 **6.1-6.3%** 选择率。
   - 合成数据结果显示数据减少了 **85.4%**，在 **87.6%** 的 Baseline 基础上实现了 **86.0%** 的准确度。
- **DRPS 提高了训练效率**：**DRPS** 系统将活动训练集规模缩减了 **16 倍**，增强了训练效率。
   - **Relevance Scorer** 的准确度从 **95.9%** 提高到 **99.95%**，**Quality Rater** 的准确度从 **97.0%** 提高到 **100%**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405814767314014238)** (4 条消息): 

> `DRPS Framework, Data Efficiency, Selection Intelligence, Synthetic Data Results, Training Efficiency` 


- **DRPS：Data Rankings and Prioritization System 问世**：正如 [situational awareness 报告](https://situational-awareness.ai/wp-content/uploads/2024/06/situationalawareness.pdf) 中详述的那样，**Data Rankings and Prioritization System (DRPS)** 通过使用 **Relevance Scorer**、**Quality Rater** 和 **Diversity Controller** 教导 AI 有选择性地从数据中学习。
- **DRPS 削减了超过 90% 的数据使用量**：在 **MNIST** 测试中，DRPS 实现了 **93.8% 的数据缩减**，仅利用所检查数据的 **6.2%** 即可保持 **99.1%** 的 Baseline 性能，相关内容展示在 [GitHub 仓库](https://github.com/voltageddebunked/drpsStats) 中。
- **DRPS 通过选择顶级样本展示其智能**：DRPS 检查了超过 **516,000 个样本**，仅选择 **32,000** 个进行训练，在整个训练过程中保持了 **6.1-6.3%** 的稳定选择率。
- **DRPS 提升了单位数据百分比的准确率分值**：使用合成数据，DRPS 实现了 **85.4% 的数据缩减**，仅使用 **14.6%** 的训练样本，每 1% 的数据使用量就实现了 **5.89 个准确率分值**，而 Baseline 准确率为 **87.6%**。
- **DRPS 框架提高了训练效率**：DRPS 通过将活动训练集规模缩减 **16 倍** 来提高训练效率，并提升了组件准确度，例如将 **Relevance Scorer** 从 **95.9%** 提高到 **99.95%**，将 **Quality Rater** 从 **97.0%** 提高到 **100%**。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405632568468045946)** (46 条消息🔥): 

> `Quantum Startup Multiverse, MoE Nuances, Tokenization and Routing Synergy, Gemma 3n` 


- **热门量子初创公司？**：一篇关于 [初创公司 Multiverse](https://techcrunch.com/2025/08/14/buzzy-ai-startup-multiverse-creates-two-of-the-smallest-high-performing-models-ever/) 的文章声称，他们利用量子技术创建了 *两个有史以来最小的高性能模型*，但他们可能只是使用了某种 **针对模型权重的专门压缩算法**。
   - 该文章似乎并未提出实际的量子技术主张。
- **解读 MoE 的细微差别**：**MoE (Mixture of Experts)** 是一系列具有非常微妙迭代的技术，包括 **token-choice**、**expert-choice**、**带有容量因子的 MoE**、**块稀疏无丢弃 token 路由 (block sparse dropless token routing) 与有丢弃路由 (droppy routing)** 等。这使得人们出于某种原因将许多事物归功于 MoE 时显得很烦人。
   - 为了验证批处理推理 (batched inference) 中出现的问题，可以可靠地检查像 **Olmoe** 或 **IBM Granite 3.1** 这样的数值行为，而不是去访问一个无法监控的 API。
- **协同 Tokenization 和 Routing**：一位成员提出了一个看似显而易见的想法，即在同一步骤中进行 **tokenization 和 routing**，以实现动态协同。
   - 另一位成员回应道：“*我从未见过这样的提议*”，因为传统观点认为，如果在专家激活前有大量的路由步骤，网络的表达能力会更强。
- **分层 Tokenization**：**Gemma 3n** 具有某种每层 tokenization / embedding。
   - 这可能是一种更好的学习 patch 级 tokenization 的方法，本质上对上下文有更多的洞察。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 条消息): 

> `DARPA AIxCC, LLM agents` 


- **团队在 DARPA AIxCC 中获胜**：一个团队宣布他们在 **DARPA 的 AIxCC (AI Cyber Challenge)** 中获得名次，他们构建了一个由 **LLM agents** 组成的自主系统，用于发现和修复开源软件中的漏洞。
   - 该项目现已开源。
- **构建强大 LLM agents 的技巧**：该团队通过 [这篇 Xitter 帖子](https://x.com/tjbecker_/status/1956081184611688667) 分享了他们构建高效 **LLM agents** 的技巧。
   - 帖子包含了一些适用于各种 Agent 开发场景的通用建议。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405628482909765652)** (16 条消息🔥): 

> `Inference Time on Low-End Devices, DinoV2 vs DinoV1, Gemma Model Parameter Size, China's Role in Automation, Deepseek Training on Huawei Chips` 


- **低端设备上的推理时间阻碍了可用性**：成员们讨论了推理时间在 **低端设备** 上更为重要，并以谷歌运行 LLM 的 Android 应用为例，指出过长的推理时间和手机发热使其变得不切实际。
   - 较小的模型可以用于键盘预测，但根据 [这段 Youtube 视频](https://youtu.be/KFYyfrTIPQY?t=2158)，这些模型可能需要在设备上进行训练。
- **DinoV2 的性能和训练挑战**：一位成员表示希望新模型能超越 **DinoV2**，因为 **DinoV2** 在某些场景下表现不如 **DinoV1**，且更难训练。
   - 他们链接了一段 [YouTube 视频](https://www.youtube.com/watch?v=eZ2A2045Rkw) 作为参考。
- **Gemma 参数揭晓**：据指出，**Gemma 270M 模型** 拥有 **100M** 参数和 **170M** embedding 参数。
- **Deepseek 的芯片选择导致训练停滞**：一位成员指出，根据 [这段讨论](https://youtu.be/FQOV-qy9CK4?t=212)，**Deepseek 的训练** 因为尝试在 **华为芯片** 而非 **NVIDIA** 芯片上进行训练而停滞。
- **制造业关税阻碍行业增长**：一位成员认为，对建设生产线所需的设备征收关税不利于鼓励制造业。
   - 他们补充说，建立一个行业需要数十年时间，并引用了 [Anthropic 关于端子集对话 (end-subset conversations) 的研究](https://www.anthropic.com/research/end-subset-conversations) 和 [HRM 分析](https://arcprize.org/blog/hrm-analysis)。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 条消息): 

venom_in_my_veins: hye
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1405750413764067489)** (4 messages): 

> `1-bit inference, GPTQ` 


- **探索加速 1-Bit 推理**：一位成员询问了关于加速 **1-bit 推理**的问题，并分享了论文 [The Power of $\alpha,1$-sparsity: Near-Lossless Training and Inference of $\alpha$-bit Transformers](https://arxiv.org/html/2411.06360v3) 的链接。
   - 该论文详细介绍了一种训练和推理 **$\alpha$-bit Transformers** 的新方法，在 **1.58 和 1-bit** 量化下实现了近乎无损的结果。
- **推理优化**：链接的论文强调了使用 **$\alpha,1$-sparsity** 对 Transformer 模型进行的优化，从而在极低位宽下实现近乎无损的训练和推理。
   - 这种方法可能会在某些应用中显著提高推理速度。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405632426998239303)** (11 messages🔥): 

> `CUDA Shared Memory, CUDA Illegal Memory Access, CUDA Kernel Launch Configuration, CUDA warp ID calculation` 


- **调试 CUDA Illegal Memory Access**：一位用户在 CUDA Kernel 中使用 Shared Memory 时遇到了 *Illegal Memory Access* 错误，并向社区寻求帮助，分享了涉及 `sat` 和 `commons` 数组的代码片段。
   - 一位成员建议该错误可能源于错误的指针运算或定义不当的 `warp_id` 和 `WARPS_EACH_BLK`，但提供了一个 [示例代码](https://gist.github.com/alecco/58206ecd6af36c627609e1f464b4b376) 以表明这可能与此无关。
- **CUDA Kernel 启动配置混淆**：用户分享了他们的 Kernel 启动配置 `<<<BLK_NUMS, BLK_DIM>>>` 和宏定义，其中 `BLK_NUMS` 设置为 **40**，`BLK_DIM` 设置为 **1024**，`WARPS_EACH_BLK` 计算为 `BLK_DIM/32`，导致了全局 Warp ID 的计算。
   - 另一位成员指出了问题所在：用户的 `warp_id` 是全局的，导致对 Shared Memory 的越界访问，而 Shared Memory 对每个 Thread Block 是局部的。
- **解决 Shared Memory 访问问题**：一位成员建议在每个 Thread Block 内使用局部索引和 Warp ID 计算，建议使用 `local_index = threadIdx.x; local_warp_id = local_index / 32;` 以确保正确的 Shared Memory 访问。
   - 他们进一步建议使用位移操作（`local_warp_id = local_index >> 5;`）代替除法和取模，以获得更好的 GPU 性能，并建议使用 NSight Compute 检查生成的汇编代码。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1405734478562721915)** (10 messages🔥): 

> `New Grad Kernel Job, GPU Thesis, Getting Kernel Job Without Internship` 


- **Kernel 职位求职者询问应届生机会**：一位成员询问，没有编写 Kernel 实习经验的人是否能获得一份编写 Kernel 的应届生工作。
   - 另一位成员表示，如果候选人对 GPU 有深入了解，他们的公司不会优先考虑实习经验，并提到他们相关的 [论文](https://github.com/Snektron/pareas) 是其成功面试过程的一部分。
- **业内人士透露如何在没有实习的情况下获得 Kernel 职位**：一位对 GPU 感兴趣的人发帖称，他们通过 GPU 相关论文和运气的结合，再加上通过了面试流程，成功获得了一份工作。
   - 据该人士称，良好的 GPU 知识可以绕过对过往经验和实习的需求。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1405833745772314667)** (1 messages): 

> `MI300 pytorch, OMP missing` 


- ****MI300** 环境缺少 **OMP****：根据用户报告，**MI300** 环境似乎缺少用于 `pytorch.compile` 的 **OMP**。
- **包含调试错误链接**：一位用户分享了 [完整调试错误的链接](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/16986368251) 以供进一步调查。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (10 messages🔥): 

> `trimul leaderboard, A100, H100, B200` 


- **Trimul 排行榜迎来新纪录**：一位成员在 **A100** 上获得**第二名**：**10.4 ms**，随后迅速在 **H100** 上获得**第一名**：**3.95 ms**，并在 **A100** 上获得**第一名**：**7.53 ms**。
   - 随后，该成员在 **B200** 上获得**第一名**：**2.35 ms**，接着再次在 **A100** 上获得**第一名**：**6.01 ms**，并又一次在 **B200** 上获得**第一名**：**2.04 ms**，最后在 **H100** 上成功达到 **3.74 ms**。
- **A100 和 H100 也有活跃表现**：另一位成员在 **A100** 上获得**第 5 名**：**13.2 ms**。
   - 该成员随后在 **H100** 上获得**第二名**：**6.42 ms**，最后在 **A100** 上成功达到 **14.7 ms**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405929507554070674)** (10 messages🔥): 

> `Meeting Attendance, Large PR Review, Connection Error Debugging` 


- **错过会议的小插曲**：几位成员提到由于时区混淆和日程冲突错过了会议，其中一位成员仅在前 **10 分钟**有空。
   - 一位成员调侃说早上 **8 点**的会议时间有点“残暴”。
- **审查范围蔓延 (Scope Creep)**：一位成员对一个包含 **300 个文件更改**的 PR 发表了评论，开玩笑说这有点“超出范围”。
   - 另一位成员补充说，这些代码是 *grass-fed hand-crafted*（纯天然手工打造的）。
- **排查连接错误**：一位成员报告遇到了连接错误，并正尝试调试其来源，猜测可能来自 **db_client**。
   - 他们提到在获取 stack trace 以诊断问题时遇到了困难。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405627475521962098)** (47 messages🔥): 

> `Kimi K2 Technical Report, GLM-4.5 vs Kimi K2, Kimi hallucinations, Kimi's Web UI, Kimi future updates` 


- **NotebookLM 视频优于 Kimi PPT**：成员们将 **Kimi 生成的 PPT** 与 Google **NotebookLM** 为 Kimi K2 技术报告生成的**视频概览**进行了对比，共识倾向于 NotebookLM 的视频，因为它包含音频且布局更灵活（见[附带视频](https://cdn.discordapp.com/attachments/1371757564005711973/1405630786308149360/Kimi_K2_10MB_final.mp4?ex=68a0d8ae&is=689f872e&hm=a7541a57850914531af4af61a14c0bfcff5cecb20b2ffe50094bafdb0a8ccde3&)）。
   - 虽然两者都受到了好评，但一位成员表示相比听 AI 生成的音频，他更喜欢阅读，但也指出了视频概览的潜力，尤其是在教育领域。
- **Kimi K2 在写作技巧上击败 GLM**：尽管有人觉得 **GLM-4.5** 在整体性能上可能超过 **Kimi K2**，但用户称赞 **Kimi** 拥有更出色的写作风格和主动错误检测能力。
   - 一位用户在 **Kimi** *“突然对我说‘不’”*时感到*“由衷的惊讶”*，并对其坦率表示赞赏。
- **对抗 Kimi 的幻觉**：用户希望 **Kimi** 即使在开启联网搜索的情况下也能减少幻觉，并指出虽然 **GLM** 可能耗时更长，但幻觉频率较低。
   - 一位用户表示，他们一直使用“踩”按钮来报告幻觉。
- **Kimi 粉丝热切期待 'Kimi Thinking'**：成员们正热切期待 **'Kimi Thinking'** 以及推理和多模态能力的到来。
   - 目前还不清楚这会以 **Kimi K-2** 还是 **Kimi K-3** 的形式出现，且尚无明确的 ETA。
- **深色模式增强 Kimi Web UI**：一位用户分享了他们使用深色模式扩展自定义的 **Kimi Web UI**，表示相比默认的灰色界面，他们更喜欢这种风格（见[附带截图](https://cdn.discordapp.com/attachments/1371757564005711973/1406009945002082374/image.png?ex=68a0e84d&is=689f96cd&hm=4d0b8f1561e558ccf4bd5b6fdf8cfd506b038c5203e9f7632504533cc9ea5ea6&)）。
   - 另一位用户确认，只有用户名和服务器角色会被传递给 Moonshot API。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405648729134076044)** (4 条消息): 

> `AI Stock Portfolio Agent, Web Scraping AI Agents, Multimodal AI Applications, Legal Knowledge Graphs` 


- **AI 股票投资组合 Agent 问世**：LlamaIndex 推出了一套构建完整 **AI 股票投资组合 Agent** 的框架，集成了 [@CopilotKit](https://www.copilotkit.ai/) 的 AG-UI 协议以实现无缝的前后端通信；并附带一份详尽的教程，用于创建一个复杂的投资分析工具。
   - 该教程结合了 [此框架](https://t.co/fQDNPIQoqR) 的强大功能，旨在打造一个精密的投资分析工具。
- **Brightdata 与 LlamaIndex 联手推出网页抓取 AI Agents**：LlamaIndex 宣布了与 [@brightdata](https://www.brightdata.com/) 合作的新指南，介绍如何利用 LlamaIndex 的 Agent 框架构建 **网页抓取 AI Agents**，重点关注可靠的网页访问和稳健的网页抓取工作流。
   - 该指南详细说明了如何设置能够处理动态内容的工作流，并构建可以导航至 [此处](https://t.co/IBgSLBM6XW) 的 **智能 Agents**。
- **多模态 AI 应用实现市场视觉化分析**：LlamaIndex 宣布构建 **多模态 AI 应用**，可同时分析文本和图像，用于市场研究和调查。
   - 这些应用旨在统一的 AI 流水线中共同处理图像和文档，从图表、图形和产品图像等视觉市场数据中提取洞察，并结合多模态 [能力](https://t.co/fOMFLXWarG)。
- **LlamaCloud 与 Neo4j 将法律文档转换为知识图谱**：LlamaIndex 发布了一份综合教程，介绍如何将非结构化的法律文档转换为 **可查询的知识图谱**，不仅能理解内容，还能理解实体之间的关系。
   - 该工作流利用 **LlamaCloud** 和 [@neo4j](https://neo4j.com/) 进行法律合同分析，详情见 [此处](https://t.co/MPSfPiS2Cv)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405664216601329764)** (28 条消息🔥): 

> `Pydantic Models vs JSON Schema for Tool Calls, Vector Store Errors After Update, Progress Bar Issue with num_workers > 1, Iterating Over Nodes/Doc_IDs in Vectorstore` 


- **Pydantic vs JSON Schema 大对决**：一位成员询问工具调用（tool calls）是否需要 **Pydantic 模型**，或者 **JSON schema** 是否足够，并指出将 JSON 转换为 Pydantic 模型后再解包回 JSON 的做法存在冗余。
   - 另一位成员指出 **Pydantic** 的 `create_model()` 函数不直接接受 **JSON schema**，强调了需要特定工具或包来处理这种转换。
- **LlamaIndex 更新后 Vector Store 出现属性错误**：更新至 **0.13.1** 版本后，用户在使用 `RetrieverQueryEngine` 配合 `OpenAI` 和 `text-embedding-3-small` 从 **PGVectorStore** 检索时遇到了 `AttributeError`。
   - 该错误源于 `output` 是一个没有 `json` 属性的 `str`，问题出在 `openinference.instrumentation.llama_index` 中的 **LLMStructuredPredictEndEvent**。
- **多进程下的进度条混乱问题**：一位用户指出，由于使用了 **multiprocessing**，当 `num_workers > 1` 时，`progress_bar=True` 功能无法正常工作。
   - 有建议认为使用 **async concurrency**（异步并发）可能会提供更流畅的体验，然而 `async pipeline.arun` 方法目前仍在使用多进程。
- **Vector Store 中缺失 Node 和 Doc ID 迭代功能**：一位用户对大多数 LlamaIndex 的 Vector Store 无法迭代 Node 或获取 `doc_ids` 列表表示沮丧，特别提到了 **Opensearch** 和 **awsdocdb** 的缺失。
   - 目前的一种权宜之计是将 `similarity_top_k` 设置为一个很大的数值，但这效率低下且并非所有开源系统都支持；虽然基类 `vector_store` 存在 `get_nodes()` 方法，但在 Opensearch 或 awsdocdb 中尚未实现，这是一个提交 PR 的好机会。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1405905432920326265)** (1 条消息): 

> `DSPy optimizes CrewAI, CrewAI agent prompts` 


- **DSPy 优化 CrewAI Agent 提示词**：一门课程教授如何在真实的生产用例中通过 **DSPy 优化 CrewAI** Agent 提示词，利用经过验证的方法构建更智能、更廉价的 Agents。
   - 您可以在 [此处](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E) 查看该课程。
- **使用成熟方法构建更智能、更廉价的 Agents**：该课程专注于针对 CrewAI Agents 的 **DSPy 优化**。
   - 它强调通过 **成熟的方法论** 构建更高效、更智能的 Agents。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405744293439733830)** (7 条消息): 

> `NotebookLM 中的音频转录，NotebookLM 界面重新设计` 


- **上传至 NotebookLM 的音频会自动转录**：一位成员询问如何获取音频转录稿，另一位成员回答说他们直接将 **MP3 音频文件上传到 NotebookLM**。
   - 该成员澄清说 **NotebookLM** 本身会处理转录生成。
- **NotebookLM 界面重新设计正在进行中**：一位成员提到他们正尝试重新设计 **NotebookLM**，并分享了拟议更改的 Figma 截图。
   - 该成员对可能引起的误解表示歉意，澄清这只是一个设计概念，而非功能更新。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405718164716650520)** (23 条消息🔥): 

> `讲解视频声音，功能请求反馈，开发者互动，Prompt 限制` 


- **讲解视频声音性别切换**：一位用户报告说，他们的讲解视频突然开始以 **男声** 而非通常的 **女声** 生成，并询问为何会发生这种情况。
   - 消息中没有提供明确的解决方案或解释。
- **用户请求对功能请求给予确认**：一位用户质疑是否真的有 **NotebookLM 开发团队** 的成员在阅读 Discord 频道中发布的 **功能请求**。
   - 他们表示希望看到开发者的一些回应或反馈，以鼓励用户持续贡献。
- **NotebookLM 开发者承认在阅读帖子但无法回复所有内容**：一位 Google 开发者表示 *开发者会阅读帖子*，但他们没有时间回复所有内容，并且花费了大量时间在 **封禁垃圾信息发送者** 上。
   - 其他用户建议，即使是偶尔的确认或 AI 汇总的摘要，也能帮助鼓励用户贡献。
- **用户在 NotebookLM 中遇到 Prompt 限制**：一位用户在尝试询问一个包含约 **857 个单词** 的案例相关问题失败后，询问 **NotebookLM** 中单个问题是否存在 **单词数量** 限制。
   - 另一位用户建议将 Prompt 拆分为多个部分，或者尝试使用 **Gemini**。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405902903151169648)** (1 条消息): 

> `CrewAI Agent Prompt，DSPy` 


- **使用 DSPy 优化 CrewAI Agent Prompt**：成员们分享了一个链接，用于学习如何在真实生产用例中通过 **DSPy 优化 CrewAI Agent Prompt**：[https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)。
   - 该课程声称将教授用户如何 *使用经过验证的方法构建更智能、更廉价的 Agent*。
- **DSPy 与 CrewAI 结合**：该课程教授用户如何使用 DSPy 优化 CrewAI。
   - 它能够通过经过验证的方法实现更智能、更廉价的 Agent。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405627855324315649)** (22 条消息🔥): 

> `DSPy 与 Databricks，GEPA 错误，MLflow 与 DSPy` 


- **Databricks 未赞助 DSPy**：一位用户询问 **Databricks** 是否赞助或拥有 **DSPy** 项目，另一位用户澄清说 DSPy 是 **MIT 许可的开源项目**，Databricks 通过核心开发团队做出了重大贡献。
- **GEPA Bug 已修复**：一位用户在将 **GEPA** 与 **RAG 教程**结合使用时遇到了 `ValueError`，另一位用户确认[这是 GEPA 代码中的一个 Bug](https://github.com/stanfordnlp/dspy/pull/8647) 且已被修复；用户应升级到 **DSPy 3.0.1**。
   - 弃用的参数位于该 dspy.evaluate 导入中，修复方法是 `pip install -U dspy`。
- **MLflow 自动追踪 DSPy 子模块**：一位用户询问如何将 **DSPy 模块**追踪与 **MLflow** 集成以用于 **text2sql 管道**，并被建议使用 `mlflow.dspy.autolog()` 而不是 `mlflow.autolog()` 来自动追踪所有子模块。
   - 使用 `mlflow.dspy.autolog()` 将在 **MLflow UI 的 Traces 选项卡**中把 **SQLGenerator**、**Validator** 和 **Reflector** 显示为嵌套 span，详情参见 [MLflow DSPy 集成文档](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/tracing/integrations/listing/dspy.mdx)和 [DSPy MLflow 教程](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/optimizer_tracking/index.md)。
- **Logprob Surprise 作为适应度函数**：一位用户分享了一条推文 [TogetherCompute 状态](https://x.com/togethercompute/status/1956416013404406018)，并猜测他们基本上是在使用 **GEPA**，并将 **logprob surprise** 作为 **适应度函数 (fitness function)**，但用于生产环境中的心理健康模型。
- **请求社区参与**：一位成员请求 Discord 中的 6500 人增加参与度，并为文档等做出更多贡献。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1405897484248813679)** (1 条消息): 

> `CrewAI, DSPy 优化, 提示词工程 (Prompt Engineering)` 


- **CrewAI 提示词优化课程发布**：一位成员宣布了一个 [Udemy 课程](https://www.udemy.com/course/crewai-dspy-optimization/?referralCode=B59F73AE488715913E7E)，演示如何使用 **DSPy** 优化 **CrewAI 提示词**。
   - 该课程将展示如何将优化后的提示词注入回 **LLM**，以便 **LLM** 使用比 **CrewAI** 拼接的提示词更好的提示词。
- **DSPy 实现优化的 CrewAI 提示词**：新课程使用 **DSPy** 来优化提示词。
   - 优化后的提示词随后被注入回 **LLM**，改进了 **CrewAI** 中的标准方法。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405629920868171879)** (8 条消息🔥): 

> `CI 速度, tinygrad 发布, tinygrad 大小` 


- **CI 速度阻碍生产力**：一位成员对 CI 速度慢表示沮丧，称如果 CI 更快，他们的工作效率会更高，并链接了 [ChatGPT 分析](https://chatgpt.com/share/689e3508-5f68-8000-97b2-aa6f1699aa74)。
- **Tinygrad 即将发布**：有人建议尽快进行 **tinygrad 发布**。
- **Tinygrad 体积膨胀**：一位成员质疑为什么 **tinygrad 0.10.3** 有 **10.4 MB**，暗示可能存在体积问题。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405802310633787423)** (14 条消息🔥): 

> `WSL2 支持, print_tree 移除` 


- **WSL2 Tinygrad Bug 浮现**：一位用户遇到了一个问题，将两个从 PyTorch 张量创建的 tinygrad Tensor 相加结果全为 **0**，并提供了一个[完整脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1405817973004046387/message.txt?ex=68a0de43&is=689f8cc3&hm=be6e6069e1975cc70d7fbda2a0b20849f96396efd36e0b905118668100b11656)用于在 WSL2 上复现该 Bug。
- **print_tree 函数被废弃**：`print_tree` 函数已被简单的 `print` 函数取代。
   - 用户注意到它*丢失了一些格式*。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405710824835780628)** (12 messages🔥): 

> `Aider Benchmark, litellm Errors, Open Source Entitlement, llama.cpp PR #15181` 


- **Aider Benchmark 受超时困扰**：一名成员针对本地 **gemma3:12b** 模型运行 **Aider benchmark**，在运行 **10.5 小时** 并完成 **221/225 个测试** 后遇到频繁超时。原因是模型无法在 **600 秒** 限制内响应，导致 *litellm.APIConnectionError* 错误。
   - 他们分享了错误日志，显示模型尝试发送约 **300k tokens**，超过了 **131,072 token 限制**，从而导致测试失败。
- **继续 Aider Benchmark**：一名成员建议使用 `ctrl+c` 退出 benchmark，重启推理服务器，然后使用 `--cont` 标志从中断处恢复 benchmark。
   - 他们还指出 *llama.cpp* 中一个[已合并的 pull request](https://github.com/ggml-org/llama.cpp/pull/15181) 可能会提升本地模型的性能。
- **OSS 维护者的负担**：一名成员批评了关于让 benchmark 自动为每个 LLM 进行配置的建议，将其标签化为 *entitlement*（理所当然的心态），并感叹这种态度导致 *无数 OSS 维护者选择放弃*。
   - 另一名成员反驳称这仅仅是出于 *好奇心*，引发了关于在开源互动中什么构成了 *entitlement* 的进一步争论。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405695906635845682)** (7 messages): 

> `Aider with Local Models, Aider Line Number Accuracy, Unit Test Coverage with Aider` 


- **本地 AI/Aider 模型带来调试痛苦**：一名成员表示在使用 **aider** 配合 **ollama**、**lmstudio** 和 **vllm** 等本地模型时遇到困难，指出即使在强大的硬件上性能也很慢。
   - 他们建议需要一个教程视频，介绍如何设置 **aider** 配合这些工具进行本地开发和调试。
- **Aider 的行号系统受到质疑**：一名成员询问 **aider** 如何确定行号，特别是在为特定代码覆盖率生成单元测试的场景下。
   - 当 **aider** 错误报告行号时会出现问题，导致测试覆盖率不正确，尽管尝试了刷新 map 和清除聊天记录也无济于事。
- **LLM 准确性影响单元测试覆盖率**：一名成员报告称 **qwen3-coder** 和 **gemini-pro** 在覆盖率报告中识别行号不准确，有时会完全遗漏覆盖范围。
   - 这种不一致性引发了关于 **aider** 是否依赖 **LLM 的准确性** 来识别行号的疑问，并建议需要探索其他方法来实现准确的单元测试生成。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1405881855823188060)** (3 messages): 

> `Grok4, Quota Increase, Benchmark Costs` 


- **Grok4 位置仍然难以捉摸**：一名成员询问 **Grok4** 的下落。
   - 另一名成员回答说 *它在文章中*，但增加执行测试所需 **quota**（配额）的请求被忽略了。
- **Grok4 Benchmark 耗资数千美元**：一名成员指出，他们 *在开发此 benchmark 期间花费了数千美元*。
   - 这突显了高级 AI 模型 benchmarking 所需的巨大资金资源。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405736806930055170)** (22 messages🔥): 

> `Manus Credit Deductions on Errors, Manus Deployment Issues, Manus Team Accounts, Add-on Credits Removal, Manus in the Wild Challenge Winner` 


- **Manus 扣费引发不满**：用户对 **Manus** 在出错时仍扣除额度表示沮丧，认为与 **Claude AI** 等其他 AI 相比，这使得完成任务变得困难。
   - 一位用户报告称，在消耗了大量额度后，**Manus** 仅做了一个简单的更改就破坏了整个应用程序，导致其无法运行。
- **Manus 部署受挫**：用户报告了 **Manus** 部署的问题，从同一个 **GitHub** 仓库创建的网站差异巨大，尤其是在处理大型文件夹时。用户通过对比 [affilify.eu](https://affilify.eu) 和 **Manus** 托管的站点 [wmhkgqkf.manus.space](https://wmhkgqkf.manus.space) 说明了这一点。
   - 一位社区经理指出，**Manus** 的定位并非编程 Agent 或纯开发工具，因此部署不是其强项，但他们正在努力改进。
- **附加额度包撤回**：用户质疑为何移除附加额度包，目前该包仅面向 **Pro** 用户提供。
   - 社区经理回应称，这一变化是为了确保重度用户的速度和质量一致，并建议通过合并相似问题、保持简洁以及避免重复请求来最大化额度效率。
- **Manus 团队账户引起关注**：一位用户询问是否可以开设 **Manus** 团队账户以共享额度。
   - 社区经理确认 **Manus** 确实提供团队方案，并引导用户访问 [官方网站](https://manus.ai) 了解详情。
- **用户抱怨额度消耗**：一位用户分享了在尝试上线网站时消耗了 **30,000 额度** 的挫败经历，期间遇到了模拟站点和模板实现的问题。
   - 他们批评系统表现不一致，有时*聪明绝顶，有时却突然变笨*，导致额度浪费，并怀疑存在拖延策略。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405855716916461669)** (9 messages🔥): 

> `Cohere Labs, Pokemon emojis, PAX Omeganauts Discord` 


- **寻找 Cohere Labs 联系方式！**：一名成员询问在哪里可以联系到 **Cohere Labs** 的人员，另一名成员建议在该 Discord 频道联系。
   - 还有成员引导该用户访问 [此链接](https://discord.com/channels/954421988141711382/954421988783444043/1400866387668504648)。
- **Discord 频道加入宝可梦表情包！**：一名成员建议在频道中添加更多 **Pokemon emojis**（宝可梦表情包），因为还有空余槽位。
   - 该成员提到这些表情包来自 **PAX Omeganauts Discord**。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405640013198131345)** (5 messages): 

> `AI Research, writenode, CV+ML pipeline` 


- **AI 研究员寻求合作**：一位对**推理和意识能力**有深厚兴趣的 **AI researcher** 正在寻找合作机会，以开发面向未来的先进技术。
   - 该成员对来自任何子领域的合作持开放态度。
- **法律专业人士转向 AI**：一位目前在美国政府工作的法律专业人士、游戏玩家和哲学爱好者正在自学 **AI alignment**（AI 对齐）理论与机制。
   - 该成员很高兴来到这里。
- **writenode 开发者使用 Cohere**：Josh 正在开发 **writenode**（一个浏览器内的认知思维伙伴和创意伴侣），并使用了 **Cohere**。
   - 在去年 12 月之前，他并没有开发者或编程背景。
- **心理学博士回归 AI 领域**：一位成员在过去 5 年攻读人类心理学博士学位后，重新回归 **AI research**。
   - 他们的兴趣在于**声音与音乐**，以及利用技术工具帮助我们表达创造力。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1405985104920055959)** (3 messages): 

> `Discord Invite Links, Channel Spam` 


- **Discord 邀请链接刷屏**：一名成员在频道中多次发布 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8) 并艾特*所有人*。
   - 该邀请链接在短时间内重复出现了三次。
- **邀请链接重复**：同一个 [Discord 邀请链接](https://discordapp.com/invite/HjWfRbqBB8) 被反复发布。
   - 这导致了类似垃圾信息的刷屏效果，可能会干扰频道的正常讨论。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405984973906903060)** (3 条消息): 

> `Discord Invite Link, HjWfRbqBB8, Channel Invitation` 


- **Discord 邀请链接刷屏**: 一名成员在频道中反复分享 [Discord 邀请链接](discordapp.com/invite/HjWfRbqBB8)，可能是为了吸引更多用户。
   - 该成员多次标记了 `@everyone`，这可能被认为是过度干扰。
- **频道邀请攻势**: 反复发布[相同的 Discord 邀请](discordapp.com/invite/HjWfRbqBB8)表明其试图增加频道成员数。
   - 使用 `@everyone` 表明该消息旨在发送给所有成员，无论他们是否对邀请感兴趣。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405660404918652948)** (2 条消息): 

> `Elicitations Specification, MCP Server Conversion` 


- **寻求 Elicitations 规范的澄清**: 一名成员询问了关于 [Elicitations 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation)的问题，即谁负责将消息/字段描述翻译成用户的语言。
   - 具体而言，他们寻求澄清：是应该由 **tools** 处理语言检测和国际化，还是期望 **MCP Clients** 进行翻译（可能通过使用 LLM）。
- **MCP Server 转换问题**: 一名成员询问：*是否存在某种工具可以将本地 MCP Server 转换为远程 MCP Server？*
   - 未提供链接或更多上下文。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405750824461668434)** (3 条消息): 

> `Unifi MCP, Unraid MCP, Syslog MCP, AI Agent Workflows, AI Security` 


- **面向 Homelab 玩家的 MCP Server 发布**: 一名成员分享了几个为 Homelab 玩家准备的 MCP（推测为 Management Control Panel）Server，具体包括：[Unifi MCP](https://github.com/jmagar/unifi-mcp)、[Unraid MCP](https://github.com/jmagar/unraid-mcp) 和 [Syslog MCP](https://github.com/jmagar/syslog-mcp)。
- **PulseMCP 将繁琐的新闻简报转为 Agent 自动化**: **PulseMCP** 使用 goose 将繁琐的新闻简报工作流转变为由 Agent 驱动、且包含 Human in the loop（人工干预）的自动化流程。
   - 有关该自动化的更多细节可以在[这篇博客文章](https://block.github.io/goose/blog/2025/08/13/pulse-mcp-automates-recipe)中找到。
- **AI Security 寻求安全问题的反馈**: 一名成员发布了关于构建 **AI Security** 的消息，旨在通过数学上的安全确定性在攻击开始前将其阻止。
   - 他们正在寻求开发者对安全问题的反馈，并链接到了[一份调查问卷](https://form.typeform.com/to/xTKa05F9)。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405631570194337793)** (4 条消息): 

> `Strix Halo profitablility, Dolphin chat template, Quantum computers, PC Memory` 


- **Strix Halo 的盈利能力骤降**: 尽管 **Strix Halo** 规格惊人，但由于其推理速度（**53 tokens/sec**）慢于 **OpenRouter** 上的 **GPT-OSS 120B**，需要 **24/7 全天候推理一年**才能实现盈利。
   - 一位用户指出，花费 2000 美元将其配置用于 **LLM** 的效率远低于提供 **200-400 tokens/sec** 的云端替代方案。
- **寻找 Dolphin 聊天模板**: 一位用户正在为 **gpt4all** 寻找适用于 **Dolphin-2.2.1-mistral-7b-gptq** 的可用聊天模板。
   - 另一名成员建议要求模型制作者上传带有 **jinja** 模板的模板。
- **量子计算“茶匙”？**: 一位用户推测了量子计算机未来的可用性，以及按“茶匙”出售 **qubits** 的可能性。
   - 他们提到了关于**全功能量子计算机**的新闻，表明该领域可能取得了进展。
- **内存模块与摩尔定律**: 一位用户提到，传统的 PC 有望在 2027 年底或 2028 年看到**更高容量的内存模块**和 **DDR6**。
   - 他们对具有高 RAM 和 VRAM 容量的微型 PC 的潜力表示兴奋，特别是对于小型企业而言。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1406014763804397599)** (1 条消息): 

> `Maternity Leave, Team Contact During Leave` 


- **产假开始！**: 一名成员宣布他们将从 **8 月 25 日**起休**产假**，直至 **2026 年 2 月**。
   - 他们期待回归后与大家交流。
- **团队交接计划公布**: 在他们休假期间，团队将监控 <@1334161614949056532>。
   - 成员如有任何问题或疑虑，也可以联系 <@709918328306663424>。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 条消息): 

__nathan: <@132818429022437376> 进展如何？
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 条消息): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace feature, Smarter Cascade Agent, Dev Containers Support` 


- **Windsurf Wave 12 发布！**: Windsurf Wave 12 首次将 **Devin 的智能**和能力直接集成到 Windsurf IDE 中。
   - 关键特性包括 **全新的 UI 设计**、**DeepWiki Integration**、**Vibe and Replace**、**更智能的 Cascade Agent**、**Faster Tab**、**Dev Containers 支持**以及 **100 多个错误修复** —— [查看更新日志](https://windsurf.com/changelog)，[阅读博客](https://windsurf.com/blog/windsurf-wave-12)，[观看 Wave 12 视频](https://www.youtube.com/watch?v=-7gm8mST9QU)，[X/Twitter](https://x.com/windsurf/status/1956074019393876280)，以及 [Reddit](https://www.reddit.com/r/windsurf/comments/1mqal3x/wave_12_released_fresh_ui_deepwiki_vibe_and/)。
- **DeepWiki Integration 为 IDE 带来 AI**: **DeepWiki Integration** 允许用户将鼠标悬停在代码符号上以获取 **AI 驱动的解释**（不仅是基础的类型信息）。
   - 用户还可以使用 **CMD/Ctrl+Shift+Click** 在侧边栏中打开详细解释，并将其添加到 Cascade 上下文中。
- **Vibe and Replace 彻底改变了批量编辑**: **Vibe and Replace** 特性通过查找精确文本匹配，提供了革命性的批量编辑能力。
   - 它允许用户应用 **AI prompts**，在整个项目中进行智能且感知上下文的转换。
- **更智能的 Cascade Agent 获得全时规划功能**: **更智能的 Cascade Agent** 现在具备全时规划模式，并带有自主待办事项列表。
   - 它还包括经过改进的工具，旨在提供更智能的响应。
- **原生支持 Dev Containers**: Windsurf 现在支持通过远程 SSH 访问直接使用容器。
   - 这一增强功能简化了涉及容器化环境的开发工作流。