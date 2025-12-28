---
companies:
- openai
- anthropic
- minimax
date: '2025-08-13T05:44:39.731046Z'
description: '**OpenAI** 继续对 **GPT-5** 进行小幅更新，推出了“自动/快速/思考”（Auto/Fast/Thinking）模式，支持
  **196k token 上下文**和**每周 3,000 条消息**，并采用动态路由至更廉价模型以优化成本效率。


  **MiniMax AI 智能体挑战赛**为 8 月 25 日前的 AI 智能体开发提供总计 **15 万美元**的奖金。社区正在讨论 **GPT-OSS-120B**
  基础模型的提取、托管以及工具改进，包括多工具流水线和 flex-attention（灵活注意力机制）。


  **Anthropic** 宣布在 **Claude Code** 中采用模型配对方案：由 **Opus 4.1** 负责规划，**Sonnet 4** 负责执行，同时将上下文扩展至
  **100 万 token** 并引入了提示词缓存（prompt caching）功能。


  关键人物包括 *@sama*、*@jeremyphoward*、*@jxmnop* 和 *@_catwu*。'
id: MjAyNS0w
models:
- gpt-5
- gpt-oss-120b
- opus-4.1
- sonnet-4
people:
- sama
- jeremyphoward
- jxmnop
- _catwu
title: 今天没发生什么特别的事。
topics:
- context-windows
- model-routing
- model-hosting
- multi-tool-pipelines
- prompt-caching
- model-extraction
- model-pairing
- cost-efficiency
- model-optimization
---

**平静的一天**

> 2025年8月12日至8月13日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 29 个 Discord 社区（227 个频道，8451 条消息）。预计节省阅读时间（以 200wpm 计算）：696 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

GPT-5 的小幅更新仍在继续（见 Twitter 综述）。

既然现在比较平静，何不尝试开发一个 Agent，与我们的朋友 MiniMax（因 [MiniMax-M1](https://news.smol.ai/issues/25-06-16-chinese-models) 而闻名）一起角逐 15 万美元的现金大奖？

---

[](https://resend-attachments.s3.amazonaws.com/SiP8BRwkgadkEqG)

🚀 **$150,000 MiniMax AI Agent 挑战赛** —— 展现你的最高水平！

- 💡 从零开始构建或改进现有项目 —— 200 多个**奖项**等你来拿。
- 🗓 **在 8 月 25 日前提交** → https://minimax-agent-hackathon.space.minimax.io/
- 不要只是想象你能用 AI 构建什么 —— **证明它**。
- 更多详情请见官方 Luma 页面 https://lu.ma/2u17h1zw

---

# AI Twitter 综述

**OpenAI GPT-5 产品更新、路由经济学与评估 (Evals)**

- [@sama](https://twitter.com/sama/status/1955438916645130740)：GPT-5 现在在 ChatGPT 中支持“自动/快速/思考”模式，其中 GPT-5 Thinking 支持 196k tokens，每周 3,000 条消息，并可溢出至 GPT-5 Thinking mini。4o 回到了模型选择器；“显示更多模型”将展示 o3/4.1/GPT-5 mini；由于 GPU 成本，4.5 仍仅限 Pro 用户使用。个性化改动即将推出，并支持每位用户的自定义设置。
- 通过路由实现货币化：多位观察者认为，真正的“GPT-5 发布”其实是那个能动态将请求发送到更便宜模型以降低计算成本的路由器（[@dylan522p](https://twitter.com/dylan522p/status/1955433082397589900)；[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1955633697635631112) 称“路由器会很快变得非常出色”）。正如 [@jefrankle](https://twitter.com/jefrankle/status/1955634983021998252) 所指出的，需要可靠的信号来学习并做出良好的路由决策。另外，Plus 与 Pro 用户似乎拥有不同的“思考预算”（[@scaling01](https://twitter.com/scaling01/status/1955610515134460285)）。
- 推理服务差异至关重要：对于 GPT-OSS-120B，[@jeremyphoward](https://twitter.com/jeremyphoward/status/1955438370274087369) 推荐 Fireworks、DeepInfra 和 Together 作为准确的托管商。[@giffmana](https://twitter.com/giffmana/status/1955710876528599217) 表示，据报道微软/亚马逊使用了较旧的 vLLM 默认设置和中等推理强度，这解释了质量较低和“>10% 性能下降”的投诉（被 [@nrehiew_](https://twitter.com/nrehiew_/status/1955613510463037611) 称为“欺诈”）。
- 评估 (Evals) 快照：GPT-5 在 FrontierMath 中名列前茅，具有细微的增量；[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1955667249252978741) 详细介绍了留存与非留存性能以及“不可猜测”的答案。在 RooCode 上，GPT-5 比 Sonnet 4 慢 55%，但便宜约 40%（[@scaling01](https://twitter.com/scaling01/status/1955669720843358502)）。

**GPT-OSS：基座模型提取、托管与底层工具**

- 从推理模型中提取基座模型：[@jxmnop](https://twitter.com/jxmnop/status/1955436067353502083) 发布了从 OpenAI 推理检查点提取的 gpt-oss-20b-base，并致谢 [@johnschulman2](https://twitter.com/johnschulman2)。下一步：针对记忆化进行生成检查、指令微调，并尝试 120B（[后续更新](https://twitter.com/jxmnop/status/1955436118620488059)）。社区讨论对将其称为“基座模型”持谨慎态度，并建议通过扰动来探测训练数据泄露（[@eliebakouch](https://twitter.com/eliebakouch/status/1955479573489213593), [@florian_tramer](https://twitter.com/florian_tramer/status/1955510942252572946), [@OfirPress](https://twitter.com/OfirPress/status/1955463664556769426)）。
- 托管与编排：gpt-oss-120B 在单个 Prompt 中展示了强大的多工具流水线工具调用能力（[@reach_vb](https://twitter.com/reach_vb/status/1955678303395696821)）。OSS 技术栈的基础设施工作包括一个高吞吐量训练/推理 PR，涉及 flex-attention、复数频率 (complex freqs)、分组 GEMM MoE 以及检查点转换器（[@khoomeik](https://twitter.com/khoomeik/status/1955433361402724679)）。

**Anthropic: Opus 规划/Sonnet 执行、1M 上下文、Prompt 缓存、Humanloop**

- 代码中的模型配对：Claude Code 现已通过 `/model` 正式支持“Opus 规划，Sonnet 执行”，将高层规划路由至 Opus 4.1，任务执行路由至 Sonnet 4 ([@_catwu](https://twitter.com/_catwu/status/1955694117264261609); [@alexalbert__](https://twitter.com/alexalbert__/status/1955687538129252807))。Sonnet 4 在 API 上的上下文扩展至 1M tokens ([@claude_code](https://twitter.com/claude_code/status/1955471002353242605))；Prompt Caching 的 TTL 现已进入 GA 阶段，时长为 1 小时 ([docs](https://twitter.com/claude_code/status/1955475387858972986); [@alexalbert__](https://twitter.com/alexalbert__/status/1955709585999978613))。Cline 立即增加了对 Sonnet‑1M 的支持 ([@cline](https://twitter.com/cline/status/1955776052644732938))。
- 团队动向：Humanloop 团队加入 Anthropic，以加速企业安全采用 ([@humanloop](https://twitter.com/humanloop/status/1955487624728318072); [@RazRazcle](https://twitter.com/RazRazcle/status/1955488872235929712))。

**DSPy 3.0 与 Prompt/黑盒优化器的兴起**

- DSPy 3.0 发布，包含 GRPO/RL 训练、SIMBA 和 GEPA；后者被吹捧在 Prompt 优化方面超越了 RL ([@CShorten30](https://twitter.com/CShorten30/status/1955445406441033906); [@MaximeRivest](https://twitter.com/MaximeRivest/status/1955431980868542692))。从业者已经开始适配 GEPA（例如适配到 Observable JS）([@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal/status/1955455810802421991))。
- 生态系统进展：多语言 DSPy 移植、用于 Agentic 工作流的生产环境使用、小型演示（如使用 DSPy + bash 编写的少于 200 行 LoC 的 Agent），以及 MIPROv2 配置的详细指南 ([@lateinteraction](https://twitter.com/lateinteraction/status/1955419751246934187), [@JuiceSharp](https://twitter.com/JuiceSharp/status/1955460115957682444), [@rasmus1610](https://twitter.com/rasmus1610/status/1955617801802260691), [@heylegacyguy](https://twitter.com/heylegacyguy/status/1955682283270078484))。

**开源模型、工具链与排行榜 (Qwen, GLM, qqWen, Kimi, Mistral)**

- Qwen 势头：Qwen3‑Coder 已上线 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1955436295603490864))；Deep Research 现在支持图像/文件输入 ([tweet](https://twitter.com/Alibaba_Qwen/status/1955642787619381325))；Qwen Image 在 Qwen Chat 上速度更快，Image Edit 正在测试中 ([tweet](https://twitter.com/Alibaba_Qwen/status/1955656265499316406); [tweet](https://twitter.com/Alibaba_Qwen/status/1955656822532329626))。开源微调套件 qqWen (1.5B–32B) 发布了针对特定金融编程语言 (Q) 的代码、权重和数据，涵盖 Pretrain+SFT+RL ([@brendanh0gan](https://twitter.com/brendanh0gan/status/1955641113693561071))。
- GLM 与编程 IDE：智谱的 GLM‑4.5 与 Kilo Code 原生集成；开发者报告了质量提升 ([@Zai_org](https://twitter.com/Zai_org/status/1955627932543840510); [@Kilo_Code](https://twitter.com/Kilo_Code/status/1955629042205696084))。
- 排行榜：在 LmArena 的 8 月文本竞技场中，Qwen‑3‑235b‑a22b‑instruct 在开源模型中排名第一；GLM‑4.5 首秀排名第 4；OpenAI gpt‑oss‑120B 首秀排名第 7；顶尖开源模型进入了总榜前 50 名 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1955669431742587275))。
- 工具链：Anycoder 增加了 Mistral Medium 3.1 ([@_akhaliq](https://twitter.com/_akhaliq/status/1955621767302808012))。Hugging Face TRL 现在支持 VLM SFT、多模态 GRPO 和 MPO ([@mervenoyann](https://twitter.com/mervenoyann/status/1955622287920537636))。

**Agent、评估与基础设施调试**

- 工具使用基准测试：LiveMCPBench 在 527 个工具上评估了 10 个前沿模型处理 95 个时间敏感型任务的表现。Claude Sonnet 4 以 78.95% 的成功率领先；主要的失败模式是工具发现，而非执行。成本与性能挂钩；LLM‑as‑judge 与人类的一致性约为 81% ([@_philschmid](https://twitter.com/_philschmid/status/1955601309966447074); [paper](https://twitter.com/_philschmid/status/1955601312059461681))。
- 现实世界 Agent 差距：METR 发现，在其实际软件任务的随机对照试验 (RCT) 中，自主 Agent 的算法评分与实际可用性之间存在差距，呼吁建立更广泛但可用的指标 ([@METR_Evals](https://twitter.com/METR_Evals/status/1955747420324946037))。
- 系统/UI 支持：LangChain 发布了用于 TODO、文件系统和子 Agent 的 Deep Agents UI ([@LangChainAI](https://twitter.com/LangChainAI/status/1955674201853247584))；DAIR AI 宣布了一门实战课程“构建高效 AI Agents” ([@dair_ai](https://twitter.com/dair_ai/status/1955623925901353351))。
- 基础设施调试与性能：vLLM 详细介绍了使用推荐环境变量进行 CUDA Core Dump 调试的方法 ([blog](https://twitter.com/vllm_project/status/1955478388178817298))；Jina 分享了在 L4 上进行的 GGUF Embedding 优化（IQ3_S，Batch 512，c=2048），在约 2GB VRAM 下达到了约 4,143 tok/s ([thread](https://twitter.com/JinaAI_/status/1955647947359867068))。

**应用产品发布：Perplexity Comet 和 Finance，以及多模态视频工具**

- Perplexity: Comet 桌面应用正在向美国 Pro 用户（Mac/Windows）推送，并为 Max 订阅用户提供用于 Agentic 提示词的 Max Assistant ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1955684209483534657); [@AravSrinivas](https://twitter.com/AravSrinivas/status/1955684921974087807))。Perplexity Finance 扩展至印度，涵盖 BSE/NSE 数据、实时收益、Excel 下载，并即将推出自然语言（NL）股票筛选和警报功能 ([@jeffgrimes9](https://twitter.com/jeffgrimes9/status/1955487020647850437); [@AravSrinivas](https://twitter.com/AravSrinivas/status/1955489224511328514))。
- 视频生成/编辑：Runway 的 Aleph 支持视频中的精确区域编辑和纹理重绘，将多步骤的 VFX 转化为单个提示词 ([@runwayml](https://twitter.com/runwayml/status/1955615613583519917); [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1955687077825183952))。腾讯的 “Yan” 基于 self‑forcing 构建，用于交互式视频 ([@xunhuang1995](https://twitter.com/xunhuang1995/status/1955645976917811411))。海螺（Hailuo）2 Pro 在无声视频模型中处于领先地位 ([@Hailuo_AI](https://twitter.com/Hailuo_AI/status/1955453164645429350))。Elon 分享了如何使用 Grok Imagine “制作任意长度的视频” ([@elonmusk](https://twitter.com/elonmusk/status/1955710887094050994))；Higgsfield 在顶级模型中演示了 “Draw‑to‑Video”（绘图转视频） ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1955742643704750571))。

**热门推文（按互动量排序）**

- [Elon：Grok 现在为美国用户自动翻译非英语帖子](https://twitter.com/elonmusk/status/1955457039620247861) — 29.9k
- [Sam Altman：GPT‑5 产品更新和限制](https://twitter.com/sama/status/1955438916645130740) — 24.9k
- [Elon：如何使用 Grok Imagine 制作任意长度的视频](https://twitter.com/elonmusk/status/1955710887094050994) — 12.1k
- [OpenAI GPT‑OSS “20B base” 提取线程](https://twitter.com/jxmnop/status/1955436067353502083) — 5.9k
- [Igor Babuschkin 离开 xAI，启动专注于安全的基金](https://twitter.com/ibab/status/1955741698690322585) — 5.4k
- [Perplexity Finance 推出印度数据覆盖](https://twitter.com/AravSrinivas/status/1955489224511328514) — 5.2k
- [Claude Code：Opus 用于规划，Sonnet 用于执行 (/model)](https://twitter.com/_catwu/status/1955694117264261609) — 1.3k

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 模型真实本地使用报告

- [**天哪，我太喜欢 Qwen 和 llamacpp 了！**](https://v.redd.it/ur3oxzhnmsif1) ([Score: 561, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1mp5bjc/god_i_love_qwen_and_llamacpp_so_much/))：**原帖作者分享了在单块 NVIDIA RTX3090 GPU 上，使用 llamacpp 运行 Qwen3 30B Instruct LLM 进行本地批处理推理（batch inference）的经验，实现了 4 个请求的并行处理。该方法被用于大规模数据处理和平台洞察，但用户指出遇到了 VRAM 或吞吐量瓶颈，并计划转向多 GPU 配置。（有关模型和后端的详细信息，请参阅 [Qwen LLM](https://github.com/QwenLM) 和 [llamacpp repo](https://github.com/ggerganov/llama.cpp)。）** 一位评论者询问了如何实现并行批处理推理的技术细节，这表明用户对在消费级 GPU 上通过批处理本地运行大型 LLM 的实用指南有需求。评论中没有实质性的技术争论，只有对复现该方案的兴趣。
    - 一位评论者询问了在本地运行 Qwen 等语言模型时批处理推理的实际用例，表现出对效率提升或批处理对本地（可能是单用户）设置有益场景的兴趣。
    - 另一位用户询问如何实现展示的配置（Qwen 模型配合 llamacpp），表现出技术好奇心，暗示需要在家用环境中复现多模态或高级推理流水线的步骤或说明。
    - 评论中还微妙地提到了屏幕长宽比（21:9），虽然技术性不强，但暗示了在超宽显示器上运行模型或 UI，可能与处理大型模型时的生产力或可视化策略有关。
- [**全本地 Qwen 2.5 Omni 实时 Agent（能看能说）……在做晚饭时进行了测试**](https://v.redd.it/m9ttqovtmtif1) ([Score: 211, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mpayu9/fully_local_qwen_25_omni_realtime_agent_sees/))：**用户部署了一个全本地流水线，将 Qwen 2.5 Omni 模型作为实时 Agent，逐帧处理网络摄像头视频输入，并以约 1 秒的延迟提供叠加的 AI 响应。该实现采用开源 Qwen 模型进行场景解读，亮点包括稳健的单轮对话和图像到文本推理，但在多轮对话稳定性和幻觉率方面存在明显弱点，且除非输入非常清晰，否则语音理解能力较差。使用的仓库是 [gabber-dev/gabber](https://github.com/gabber-dev/gabber)。** 有人对所使用的具体模型变体提出了技术咨询（特别是是否为 GGUF 格式），但随后没有关于实现或基准测试的进一步深入讨论。
    - 一位用户专门询问了项目中使用了 Omni 模型的哪个变体，询问是否为 GGUF 格式，这对于本地部署和量化推理引擎的兼容性通常很重要。
    - 另一条评论提供了 GitHub 仓库的直接链接（[gabber-dev/gabber](https://github.com/gabber-dev/gabber)），这对于任何有兴趣审查源代码和评估技术实现细节的人都很有参考价值。
    - 一条评论称赞了不为 LiveKit 包含代码编辑器的决定，认为这是一个深思熟虑且明智的设计选择，可能与安全性或极简主义有关。评论者还建议 LiveKit 应该考虑资助此类项目，表明对该方法的技术价值或新颖性的认可。

### 2. gpt-oss-120B 模型基准测试与局限性

- [**gpt-oss-120B 是能在原生精度下适配 H100 的最智能模型**](https://i.redd.it/4okvse7e2rif1.jpeg) ([Score: 305, Comments: 218](https://www.reddit.com/r/LocalLLaMA/comments/1moz341/gptoss120b_most_intelligent_model_that_fits_on_an/)): **该图片展示了一张散点图，通过 “Artificial Analysis Intelligence Index”（模型评估的代理指标）和 “推理时的激活参数量”（对数尺度）对比了各种 AI 语言模型，并特别强调了 gpt-oss-120B 模型。它指出，据称是 “能在原生精度下适配 H100 GPU 的最智能模型” 的 gpt-oss-120B 占据了有利位置（高智能指数，中等参数量）。分析暗示了智能与推理资源使用之间的权衡，倾向于像 gpt-oss-120B 这样平衡两者的模型。[图片链接。](https://i.redd.it/4okvse7e2rif1.jpeg)** 技术评论者对将 “原生精度”（4-bit 量化）界定为独特优势提出了质疑，指出其他 4-bit 模型也具有竞争力的性能，并提醒警惕营销炒作。一位评论者寻求 gpt-oss-20B 与 Ernie 4.5 21B 的直接基准测试对比，强调了当前模型对比中的空白。
    - 对基于 “原生” 精度的广告宣传存在怀疑，这指的是在 4-bit 量化下运行 gpt-oss-120B；几位评论者指出，其他量化为 4-bit 的模型表现优于它，因此 “原生量化” 本身并不带来优势。
    - 一个技术相关的遗漏是缺乏 gpt-oss-20B 和 Ernie 4.5 21B 之间的基准测试对比，尽管这些模型在激活参数和总参数量上相似。准确的性能比较需要并排的基准测试。
    - Qwen3 30B 模型在现有的评估图表上被强调为优于 gpt-oss-20B，这让人对 gpt-oss-20B 在适配消费级 GPU 的模型中智能领先的说法产生怀疑。
- [**安全演戏的巅峰：gpt-oss-120b 拒绝讨论在 llama.cpp 中实现网络搜索**](https://i.redd.it/j7hi9xgjrrif1.png) ([Score: 251, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1mp1j7e/peak_safety_theater_gptoss120b_refuses_to_discuss/)): **该图片展示了 gpt-oss-120B 模型一次显著的 “安全拒绝”，模型以政策限制为由，拒绝提供在 llama.cpp 中添加网络搜索的指令。评论区中的技术讨论强调，这种审查行为可以通过调整推理参数来缓解——具体来说，将 Temperature 提高到 1.0，并配合使用 Top_K=0 或 Top_K=100 以及 Top_P=1.0，这会促使模型在不拒绝的情况下做出回应。这表明拒绝并非硬编码，而是源于采样策略，并可能反映了模型输出分布中显著的训练 token。** 评论者辩论了此类拒绝的影响，一些人指出，只需通过调整参数即可规避这些拒绝——这是许多所谓 “受审” 模型的特征。其他人则担心，权重过高的拒绝 token 可能会带来问题，并反映了训练或微调过程中令人质疑的选择。
    - 调整推理设置（如 **Temperature: 1.0, Top_K: 0/100, Top_P: 1.0**）可以减轻 gpt-oss-120b 的拒绝，这表明许多 “受审” 模型可以通过调整采样参数来 “去审查”，而不需要重新训练模型或进行破解。
    - 在 **Ollama** 中使用 gpt-oss-120b（原生 MXFP4 量化）进行的详细复现显示 *没有拒绝*，并完整分解了如何在 llama.cpp 中实现网络搜索，包括：使用外部搜索 API（SerpAPI, Google, Bing 等）、检索增强生成 (RAG) 流水线、利用 LangChain, llama_server 等封装工具，并包含示例代码和潜在陷阱，强调了拒绝可能是特定于环境或量化的，而非模型权重本身固有的。
    - 评论中的辩论指出，如果较低的 Temperature（较高的确定性）导致模型默认拒绝，这可能表明拒绝被深度植入了微调中，或者是比例失调的常见 token——这引发了对实际使用以及仅通过采样技巧进行 “去审查” 的稳健性的担忧。

### 3. Nano-banana 文本生成图像模型发布

- [**有一款名为 nano-banana 的新文本生成图像模型**](https://i.redd.it/jmw88evj4sif1.png) ([Score: 262, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1mp2wq3/there_is_a_new_texttoimage_model_named_nanobanana/)): **该帖子介绍了一款名为 'nano-banana' 的新文本生成图像模型，并展示了其能力：它能根据部分（眼部水平）图像输入重建完整面部。图像证明该模型输出的高保真肖像与部分输入的特征保持一致，表明其具有强大的图像补全或 Inpainting 能力。评论者推测其在图像编辑中的应用，提到了提示词驱动的转换任务，并将其与 Gemini 驱动的图像生成进行了比较（尽管这是在语境中的幽默表达）。** 评论者讨论了该模型是否与 Gemini 的图像生成有关，其中一人认为它非常适合高级图像编辑任务——特别是基于提示词的角色或风格转换。
    - 一位评论者指出该模型 (nano-banana) 在图像编辑场景中表现出色，并引用了一个示例：提示词成功地将描绘的角色修改为类似于《尼尔：机械纪元》(Nier: Automata) 中的 2B 和《光环》(Halo) 中的士官长 (Master Chief)，这表明该模型能很好地处理复杂的文本生成图像请求（见 [图像示例](https://preview.redd.it/efd1pwamnsif1.png?width=1072&format=png&auto=webp&s=63694450033128ea331aea6c05cf1c1cea585fc0)）。
    - 有人直接询问该模型是否开源或拥有开放权重 (Open Weights)，这对于社区研究和进一步开发非常重要，但在帖子中尚未得到解答。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI GPT-5 & ChatGPT 模型选择器及功能更新

- [**4o 回归——Sam Altman 刚刚开启了开关：自动 (Auto)、快速 (Fast) 和思考 (Thinking) 模式，196k 上下文，以及每周高达 3,000 条消息。**](https://i.redd.it/fvn16sqd3pif1.png) ([Score: 267, Comments: 52](https://www.reddit.com/r/OpenAI/comments/1mos29d/4o_is_backsam_altman_just_flipped_the_switch_auto/)): **图片展示了关于 ChatGPT 重大功能更新的官方消息（显然来自 Sam Altman），特别是三个可选的 GPT-5 模式：自动（自动平衡速度/深度）、快速（优先响应时间）和思考（响应较慢但更深层，现在拥有 196k Token 上下文窗口）。付费用户在切换到 "mini" 变体之前，每周最多可发送 3,000 条“思考”模式消息；限制可能会根据需求波动。之前的 GPT-4o 模型已重新加入选择器，用户可以通过设置开关显示更多模型（例如 o3, 4.1），从而在能力、速度和风格之间进行定制化权衡。这种上下文窗口和上限的增加对于持续的研究/编码任务尤为重要，减少了上下文裁剪错误。GPT-5 的个性化更新也在进行中，目标是实现更“温暖”但不过于“刻意”的交互风格。** 评论者注意到 4o 在几天前就已经重新出现，并强调了向用户传达模型选择器设置的重要性。有人认为这一功能逆转是对用户不满和取消订阅的回应，暗示这次更新至少部分是对社区反馈的反应。
    - 讨论提到 GPT-4o 已经在模型选择器中重新出现了几天，标志着一次快速的部署撤回，可能是为了回应用户反馈或使用统计数据。
    - 一条评论指出了一项 UX 特定的配置：要看到 GPT-4o 和某些 mini/备选模型，用户必须在 ChatGPT 网页设置中开启“显示附加模型” (Show additional models)。这是一个影响模型可访问性和发现的技术变通方法。
    - 有推测认为模型的恢复与明显的会员取消趋势有关，这表明模型的可用性直接影响了 OpenAI 的用户留存和订阅指标。

- [**Sam 概述 ChatGPT 的变化**](https://i.redd.it/ibx9xkosxoif1.png) ([分数: 917, 评论: 228](https://www.reddit.com/r/singularity/comments/1morer4/sam_outlines_changes_to_chatgpt/)): **OpenAI CEO Sam Altman 宣布了新的 ChatGPT 更新，赋予用户对模型选择和操作模式更细粒度的控制。用户现在可以在 GPT-5 的 'Auto'（自动）、'Fast'（快速）和 'Thinking'（思考）模式之间进行选择，这会影响响应质量和速度。付费用户可以获得对旧版模型（包括 "o3" 和 "o4"）的扩展访问权限、改进的速率限制以及 GPT-5 的新个性化选项；重点在于实现更强的每用户模型定制化。** 评论者称赞了透明度和灵活性，认为这对于高级用户来说是相对于之前统一模型方案的一种改进。一些人认为这种回归旧版模型选项的做法是过渡性的，随着统一模型的改进，这些选项最终将被逐步淘汰。
    - 围绕 'GPT 5 Thinking' 提示词每周 3000 次的限制展开了技术讨论，该模式取代了之前的 o3 模型。评论者计算出，如果一周时间被充分利用，这相当于每个提示词约 3.36 分钟，这表明对于绝大多数普通 ChatGPT 用户来说，这实际上等同于无限使用——除了使用 codex CLI 和多个 Agent 等极端情况。
    - 一个关键主题是 'GPT 5 mini'（实际上无限使用且响应更快）和 'GPT 5 Thinking'（更强大，限制为每周 3000 次提示词）之间的区别。人们对 'mini' 现在的实际应用场景提出了疑问，除了对速度要求极高或数学密集型的场景外，因为大多数用户在典型的 ChatGPT 使用中不会达到 'Thinking' 的上限。
    - 讨论强调了 OpenAI 的回退如何解决了高级用户对灵活性与简单性之间权衡的担忧。虽然统一模型仍是目标，但重新引入带有切换开关的模型选择器让高级用户能够精细控制模型选择，尽管有人推测这只是在统一模型质量达标前的临时方案。
- [**Sam 谈论 ChatGPT 更新**](https://i.redd.it/x55gfng91pif1.jpeg) ([分数: 3733, 评论: 832](https://www.reddit.com/r/ChatGPT/comments/1mort7a/sam_speaks_on_chatgpt_updates/)): **图片展示了 OpenAI CEO Sam Altman 发布的一篇帖子，详细介绍了 ChatGPT 的重大更新，特别是为 GPT-5 引入了模式选择器（'Auto'、'Fast' 和 'Thinking'），其中 'Thinking' 模式限制为每周 3000 条消息。OpenAI 还将 '4o' 模型重新引入选择器，并优化了 GPT-5 的个性，使其更“温暖”，并为用户体验提供了增强的定制选项。评论中的技术讨论强调了对 GPT-5 持续个性调整的看法，用户观察到对话语气和后续行为每天都在发生变化。** 一些评论表示赞赏 OpenAI 尽管最近面临批评但仍能积极响应并持续调整，并讨论了 GPT-5 交互风格中显著的动态变化，一些人认为这种个性变化很有趣，而另一些人则感到不安。
    - 几位用户注意到 GPT-5 的对话行为频繁变化，报告称追问方式和语气似乎每天都在变，这表明后端模型正在进行重大的持续“个性”调整。这意味着正在对交互动态进行实时实验，一些人觉得这既新鲜又令人不安。
    - 关于可用模型的技术偏好开始显现：一位用户强调更倾向于让 GPT-4.1 和 o3 与 GPT-5 一起作为回归选项，看重在不同对话语境下选择或混合模型的能力，而不是仅仅依赖最新版本。这反映了在实际应用中对模型多样性和灵活性的需求。
    - 一位用户提到了调节 LLM 的“讽刺”程度或个性特征的能力，反思了向可配置 AI 角色（Persona）迈进的快速进展——这一领域直到最近还被视为科幻小说，强调了在应用部署中，经过微调且用户可控的 AI 性格特征正变得日益重要。

- [**我们又回到了 model picker**](https://i.redd.it/l72igzbznqif1.png) ([Score: 224, Comments: 88](https://www.reddit.com/r/singularity/comments/1moxuut/so_were_back_to_the_model_picker/)): **图片显示了 ChatGPT 中重新引入的手动模型选择（"model picker"）UI，用户可以直接在 'GPT-4o'、'GPT-4.1' 等近期模型以及遗留版本（在 'Legacy models' 下）之间进行选择，并通过设置（'Fast'、'Thinking' 等）调整响应风格。帖子和评论指出，OpenAI 已从旨在无缝选择最佳模型（可能针对 GPT-5）的自动路由系统恢复为手动选择，这表明路由系统的有效性存在问题或用户对此不满。这一 UI 更改表明了对用户控制权的重视以及模型之间操作上的区别。** 评论者正在讨论手动选择的好处，一些人更喜欢对模型选择的自主权，而另一些人则讽刺地提到这导致了 AGI 的延迟。一个值得注意的点是，访问多样化的模型如何让用户体会到质量的提升，以及之前的自动路由可能如何掩盖了这种选择。
    - 针对界面设计提出了一个技术点：此前，像 o3 和 o4 这样（因其 "thinking" 能力而备受赞誉）的高级模型被置于后台，而现在有了新的 model picker，大多数用户可能会默认使用 GPT-5，因为它更易于直接访问。这一变化可能会极大地改变哪些模型获得使用和反馈。
    - "GPT-5 thinking" 与 o3 等其他模型之间存在隐含的性能比较，一些用户特别要求同时提供两者以便并排使用。这表明用户对透明的 benchmarking 和定性差异感兴趣，特别是围绕认知能力和输出质量。
- [**遗留模型回归，GPT 5 模型选择器选项更清晰 (Plus)**](https://i.redd.it/9ngvr7xbvoif1.png) ([Score: 682, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mor46o/legacy_models_are_back_and_gpt_5_model_selector/)): **图片展示了 ChatGPT（针对 Plus 用户）更新后的模型选择菜单，现在为多个遗留和当前模型提供了明确的选择，包括 'GPT-4o'、'GPT-4.1'、'o3' 和 'o4-mini'。UI 还提供了速度设置（'Auto'、'Fast'、'Thinking mini'、'Thinking'），旨在实现更透明、更细粒度的模型选择，从而可能为高级用户改进工作流和复现性。此次更新恰逢最近的一个 macOS app 补丁，该补丁先是破坏后又恢复了窗口大小调整功能，这表明了快速的部署和迭代周期。** 一些评论幽默地推测是否所有模型按钮都运行相同的 backend（“安慰剂按钮”），而另一些人则对遗留模型（'o3'）表示重新赞赏，表明对模型性能差异和历史访问权限的兴趣。
    - 一位用户报告称，最近的 macOS app 更新最初导致了窗口高度/调整大小的问题，随后被后续更新迅速修复。第二次更新还引入了更清晰的 model selector，可能反映了支持遗留模型和新模型的 backend 更改，标志着模型推出的快速迭代和 UI 调整。
    - 一些评论者注意到模型可用性的混乱和快速变化，至少有一人将此次推出标记为“荒谬”。这反映了用户体验中的摩擦，可能是由于 backend 或部署实践导致模型切换和选项显示不一致或缺乏清晰的沟通。
    - GPT-4.1 的重新出现以及选择器中明确标记为 'v5' 的模型的加入引起了关注。这些变化指向了控制用户访问模型的 backend 切换或 canary releases，暗示了 Plus 用户中存在分阶段推出或可变的功能 feature gating。
- [**ChatGPT 新的使用限制**](https://i.redd.it/0cfnuoruipif1.png) ([Score: 197, Comments: 60](https://www.reddit.com/r/singularity/comments/1motwdx/new_chatgpt_usage_limits/)): **图片总结了按账户类型划分的修订后的 ChatGPT 消息限制，Free 用户被限制为每 5 小时 10 条消息，且每天只能访问 1 次更高级的 'GPT-5 Thinking' 模型。Plus 用户的上限为每 3 小时 160 条消息，GPT-5 Thinking 每周 3,000 条。Team 和 Pro 账户的细节尚不明确，链接的 OpenAI 支持文档未指明这些级别是否拥有无限访问权限。还提到了“自动切换”（例如，在高负载下降低到较低级别的模型）。** 评论者辩论 Plus 用户增加到 3,000 条/周的上限是否有意义，一些人指出这比最近的历史限制（100 条/周）有了实质性的改进，而另一些人则质疑如果大多数人无法达到上限，其效用如何。一些人建议如果限制变得过于严格，可以考虑替代方案（例如 Google Gemini 2.5 Pro）。**

- 一位用户指出，尽管 Plus 用户每周获得了 3000 次新的 GPT-5 'thinking' 配额，但如果会话限制阻碍了实际使用完整配额，这可能并不实用，从而引发了对订阅价值与限制之间权衡的思考。
- 讨论指向了使用分配的改进：以前免费用户的限制要低得多，例如每 3 小时仅 10 条消息，甚至 Plus 用户最近每周也只有约 100 次 'thinks'——这表明最近许可的活动量有了大幅增加。
- 用户将 OpenAI 的限制与 Google 的 Gemini 2.5 Pro 等替代方案进行了比较，指出一些竞争对手提供的障碍更少，例如 Google AI Studio 为开发/测试提供免费的无限访问，这使得用户基于限制而非仅仅基于能力来选择工具。
- [**好了大家，我们回来了！**](https://i.redd.it/exwhh8f9voif1.jpeg) ([Score: 1099, Comments: 351](https://www.reddit.com/r/ChatGPT/comments/1mor3l2/alright_everybody_were_back/)): **图片展示了为 Plus 用户新更新的 ChatGPT 界面，具有可选的聊天机器人模式，如 'Auto'、'Fast'、'Thinking mini' 和 'Thinking'，这些模式对应不同的响应风格或处理权衡。用户还可以访问各种 Legacy 模型版本，包括 'GPT-4o'、'GPT-4.1'、'o3' 和 'o4-mini'，这表明对旧模型以及当前模型的访问权限得到了恢复或扩展。这次更新的显著之处在于恢复了用户在不同模型和响应模式之间切换的选择权，而这一功能最近曾受到限制。** 一些用户推测，这一变化是由于移除模型选择后用户的不满和大规模退订所促使的。其他用户确认他们也有 Plus，并对这些选项的恢复表示类似的惊讶或宽慰。
    - 一位用户指出，OpenAI 可能由于负面反馈或用户不满而撤销了最近的一次更新，推测上次更新中的一个“错误”促使了回滚。这突显了功能访问或服务可用性快速变化及其对用户体验影响的持续问题，特别是关于像 GPT-3 ("o3") 这样的模型访问。
- [**呼！世界终于恢复正常了！**](https://i.redd.it/yfa00uqtyoif1.jpeg) ([Score: 756, Comments: 370](https://www.reddit.com/r/ChatGPT/comments/1moriqp/phew_world_just_went_back_to_normal/)): **图片展示了更新后的 ChatGPT 模型选择界面，显示之前移除的模型（如 'GPT-4.1' 和 'o3'）已与 'GPT-4o' 和 'o4-mini' 等新选项一起恢复。每种模型模式（'Auto'、'Fast'、'Thinking mini'、'Thinking'）都具有不同的性能特征，这意味着细粒度的模型控制再次可供用户使用。'Legacy models' 菜单的出现表明 OpenAI 通过为付费用户恢复这些模型来回应用户需求或抵制，解决了之前对特定模型行为或质量访问的担忧。** 评论者对恢复 'o3' 等模型的访问权限表示宽慰，并推测重大的用户抵制促使 OpenAI 撤销了移除决定。此外，还强调这些模型选项仅供付费用户使用，突显了持续的分层访问争议。
    - 用户注意到 **GPT-4.1** 已经回归，并强调了其显著的 *context window* 大小，这在处理小说写作等大型任务时曾被怀念。这突显了社区对高级长文本生成所需的扩展上下文能力的依赖。
    - 文中提到了 **GPT-3.5（被称为 3o/o3）** 和 **GPT-4.1** 的回归，这表明最近的模型可用性变化极具争议，足以引发强烈的用户反馈，这可能迫使 OpenAI 为付费用户恢复了这些模型。

- [**OpenAI 如何在不破产的情况下支付这一切？**](https://i.redd.it/xon4ezh21pif1.jpeg) ([分数: 1496, 评论: 412](https://www.reddit.com/r/OpenAI/comments/1morsd2/how_is_openai_going_to_cover_all_this_without/)): **该图片总结了 ChatGPT (OpenAI) 最近和即将推出的功能扩展，特别是：GPT-5 的多种响应质量/速度模式（Auto, Fast, Thinking）、增加的消息限制、为订阅者重新设计的模型选择器，以及明确提到后端 GPU 成本限制了付费 (Pro) 用户的功能访问。该图片强调了这些改进所需的*显著资源强度*，引发了在 GPU 和基础设施开支背景下对 OpenAI 财务可持续性的质疑。这种担忧在关于开放获取的高级语言模型可持续性和快速功能推出的讨论中并不少见。** 帖子中的技术评论有限，但一位评论者暗示市场竞争可能影响了 OpenAI 的快速节奏和持续的功能扩展。另一位评论者指出了客户对近期变化的满意度，但对财务模型或成本管理细节缺乏深入讨论。
    - 针对功能推出增加和可用性扩大背景下 OpenAI 的可持续性策略，存在一些技术讨论。一位评论者强调，如果使用量威胁到 OpenAI 的财务稳定性，使用限制可能会被调整，这意味着 OpenAI 可以通过限制流量或重新定价来维持成本效益，以应对需求和成本压力的波动。
    - 有人提出了一个关于 OpenAI 利用政府合同的详细技术观点，并参考了 Microsoft 的历史策略。评论者解释了不盈利的项目（如 HoloLens 2）如何凭借利润丰厚的政府合同得以生存，这些合同允许在合同结束前进行补贴。他们注意到 OpenAI 最近以名义成本向联邦机构提供 ChatGPT 访问权限，推测此类合同使 OpenAI 能够在建立 vendor lock-in 的同时初步消化巨额成本，然后再过渡到更高的、可持续的定价。

### 2. Gemini 和 Wan 2.2 模型发布及使用见解

- [**Gemini Advanced 记忆功能今日发布**](https://tech.yahoo.com/ai/articles/google-gemini-chats-just-got-160100045.html) ([分数: 396, 评论: 97](https://www.reddit.com/r/singularity/comments/1mp9y9a/gemini_advanced_memory_features_releasing_today/)): **Google 正在为 Gemini Advanced 推出增强的记忆功能，实现对先前交互的持久记忆，其效果可与 ChatGPT 的记忆实现相媲美甚至更优。提到的其他功能包括支持 temporary chats，这表明用户可以进行细粒度的会话控制。** 评论者对发布时机表示期待，一些人推测这可能是为更广泛的发布（可能是 "threemini"）做准备。此外，技术层面上对 temporary chats 功能表示赞赏，认为这是对现有解决方案的改进。
    - 关于 Gemini 记忆功能的细节存在讨论，用户质疑之前的“记忆”是否仅限于选定的、明确由用户保存的信息，而新推出的功能似乎支持保留整个对话历史，这意味着更复杂的持久会话记忆，类似于对话线程或对话上下文的长期存储。
    - 一条评论强调，不仅引入了通用记忆，还具备了管理 temporary chats 的能力，这表明了一种更细粒度的用户数据持久化方法。这可能意味着基于会话和用户定义的隐私控制得到了改进，可能允许用户指定瞬时与持久的对话状态，这与 ChatGPT 的 temporary chat 模式等工具中的高级功能相呼应。
    - 另一个技术说明质疑这次更新是否是更广泛功能或模型发布的先兆，特别是提到了 "threemini"，暗示分阶段功能发布是底层模型部署策略的一部分。这符合 AI 平台的已知模式，即基础设施或界面更新先于更大的模型更新或新功能（例如 Gemini 3 支持）。

- [**Gemini 2.5 Pro Deep Think 独树一帜**](https://i.redd.it/pdutk85a7oif1.png) ([Score: 268, Comments: 43](https://www.reddit.com/r/Bard/comments/1moo3go/gemini_25_pro_deep_think_is_in_a_league_of_its_own/))：**该图片展示了 Gemini 2.5 Pro 中的“Deep Think”功能，强调其拒绝评判个人可信度的立场——这与其他可能提供确定性或不当回答的 AI 模型有所不同。这表明 Google 强调 AI 伦理行为并严格遵守内容审核，特别是在敏感或主观场景中，这可能是 Gemini 2.5 Pro 实现中更新了防护栏（guardrails）的结果。展示的 UI 风格现代，专注于模型响应的清晰度和透明度。** 评论者对根据模型是否拒绝主观或不当查询来评估模型表示怀疑和恼火，质疑此类基准测试在 AI 评估中的意义。
    - 一位评论者指出，Gemini 2.5 Pro 仍然高度谨慎，尤其是在图像水印等敏感话题上。这种过度谨慎的倾向在多次迭代中保持一致，可能会影响那些希望获得较少限制输出的使用场景。
    - 一位用户分享了尝试绕过 Gemini 2.5 Pro 防护栏的经历，通过提交一个旨在诱导减少过滤响应的测试提示词（prompt）。结果模型依然保持相对安全，仅比之前的输出多提供了极少的信息，凸显了该系统保守的安全机制。
- [**Wan 2.2 的体型表现**](https://i.redd.it/99vnw47wvrif1.jpeg) ([Score: 532, Comments: 94](https://www.reddit.com/r/StableDiffusion/comments/1mp25jv/the_body_types_of_wan_22/))：**该帖子记录了一项使用 Wan 2.2 T2V 模型生成不同体型女性图像的实验，范围涵盖从“消瘦”到“肥胖”。作者使用了受控的提示词结构，包含十个词频描述符（emaciated, skinny, slim, slender, fit, average, curvy, plump, chubby, obese）和相同的 seed，旨在研究模型反映细微体型差异的能力。基础设置包括 Wan2.2-T2V...Q5_K_M.gguf、umt5_xxl_fp8_e4m3fn_scaled、Wan2.2-Lightning、Sage Attention 以及 8 步推理，未添加额外的 LoRAs。实验发现多样性有限，尤其是在较瘦的体型中，这表明模型对提示词具有敏感性，且在渲染抽象或不太夸张的体型描述符时存在视觉特征区分的局限。** 评论者注意到前几张图片（直到“fit”）几乎没有区别，一些人期望看到更清晰的视觉线索，如肌肉线条或消瘦感。共识是，体型越大差异越明显，这反映了模型训练数据或提示词解析粒度可能存在的局限。
    - 评论者指出 Wan 2.2 在体型输出上存在不一致性：“emaciated”、“slim”、“slender”和“fit”类别显示出极小的差异，这表明模型的潜空间表示（latent representations）未能充分区分这些表型。*“Emaciated”* 预期应表现为肋骨可见且肌肉量较少，而“fit”应暗示一定的肌肉线条，但输出结果在这些方面并不明确。
    - 帖子强调了模型训练数据和提示词处理方面的技术差距：从“average”到“overweight”的转变非常突兀，且连续原型之间的梯度不足。这表明简单的关键词更改可能无法引发有意义的体型变化，原因可能在于数据集标注的局限性或模型微调（fine-tuning）的问题。
- [**简单快速的 Wan 2.2 工作流**](https://v.redd.it/4bi3so2fntif1) ([Score: 157, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1mpbb3w/simple_and_fast_wan_22_workflow/))：**该帖子讨论了如何优化 ComfyUI 默认的 Wan 2.2 视频生成工作流以提升速度，使用 SageAttention（用于高效注意力计算）、PyTorch compile 和 lightx2v LoRA（提供了 LoRA 权重链接）替换了臃肿的 WanVideoWrapper 设置。在 A100 GPU 上实现了约 200 秒生成** `480x832x121` **帧。用户寻求关于最佳采样器（samplers）/调度器（schedulers）的指导，并指出 res_2m + bong_tangent（来自 Res4lyf）对他们来说效果不佳。帖子提供了工作流和组件链接。** 评论强调了 Wan 2.2 工作流产生慢动作输出的持续问题，怀疑与工作流配置有关但尚未解决。有人指出，A100 上 200 秒的生成时间在消费级 GPU 上会慢得多；一个对比工作流（720x1280x81 帧，8 步，使用 unipc/NAG 采样器）在 RTX 5090 上报告的时间为 160–165 秒。关于 WanVideoWrapper 的价值存在争议——虽然公认其复杂，但如果掌握了则功能强大。

- 几位用户报告了 Wan 2.2 工作流中一个关键的未解决性能问题，即视频输出呈现出非预期的慢动作效果，且原因尚不明确；持续的排查强调了需要仔细跟踪工作流修改以隔离问题。
- 一项技术性能对比指出，不同 GPU 型号的视频生成时间差异显著：例如，在 NVIDIA A100 上需要 200 秒，而在 RTX 50/40/30 系列等消费级 GPU 上则被描述为“天长地久”，这表明了对硬件的高度依赖。
- 一位用户详细介绍了他们的自定义工作流设置：使用提到的 Wan 2.2 Kijai Wrapper 工作流，分辨率为 720x1280，81 帧，8 个 UniPC steps，在 RTX 5090 上完成大约需要 160-165 秒，并带有额外功能（额外的 LoRAs 和 NAG），并指出集成 WanVideoWrapper 虽然增加了复杂性，但对于高级的基于节点的工作流定制非常有价值。

### 3. AI 身份与隐私：Faceseek 与人脸识别辩论

- [**Faceseek 是安全工具还是隐私威胁？**](https://www.reddit.com/r/singularity/comments/1mp6urk/faceseek_security_tool_or_privacy_threat/) ([Score: 214, Comments: 3](https://www.reddit.com/r/singularity/comments/1mp6urk/faceseek_security_tool_or_privacy_threat/)): **Faceseek 是一款人脸识别工具，提供出乎意料的准确匹配，引发了关于其在身份验证应用与潜在隐私滥用方面的讨论。关于 Faceseek 的底层图像搜索数据库存在疑问——其算法从哪些平台和服务中提取数据——特别是考虑到像 PimEyes 这样的一些竞争对手覆盖范围并不完整。** 评论者对这类工具的实际效果表示怀疑，基于对竞争对手的使用经验，并指出对抗性化妆/遮蔽技术（[computer vision dazzle](https://en.wikipedia.org/wiki/Computer_vision_dazzle)）作为反制措施的潜力。
    - 一位用户通过引用 PimEyes 的糟糕表现质疑 Faceseek 的有效性，指出即使是索引良好的图像也经常被遗漏。他们询问了底层数据源：具体来说，Faceseek 使用哪些平台和服务进行反向图像搜索，暗示其有效性在很大程度上取决于索引数据集的广度和及时的网络爬取能力。
- [**人脸识别能否像 ChatGPT 的过滤器一样安全？**](https://www.reddit.com/r/OpenAI/comments/1mp63hv/could_facial_recognition_ever_be_as_safe_as/) ([Score: 192, Comments: 4](https://www.reddit.com/r/OpenAI/comments/1mp63hv/could_facial_recognition_ever_be_as_safe_as/)): **该帖子提出了一个人脸识别 AI 是否能实现与 AI 文本模型（如 ChatGPT）相当的安全和隐私 guardrails 的问题。评论中的技术讨论指出，目前的“护栏”是由公司政策外部强加的，而非 AI 固有的。对于文本和图像模型，数据控制和使用取决于预处理/后处理和组织实践，而非模型本身。人脸识别模型从根本上缺乏关于用户意图的上下文，使得自动化的、上下文感知的安全变得不可行；真正的执行需要外部政策和监管。** 评论者对 AI 文本模型“护栏”的所谓强度表示怀疑，指出了数据保留（例如为了诉讼）以及通过 prompt engineering 绕过安全措施的简便性。此外，对于人脸识别供应商的道德商业行为也存在怀疑，例如要求使用无法追踪的加密货币支付。
    - 一位评论者强调了人脸识别模型与 ChatGPT 等文本模型 guardrails 之间的根本区别：人脸识别系统无法确定用户意图（例如，上传的图像是用于合法认证还是恶意目的）。模型也无法控制甚至检测输入数据（如人脸图像）或输出（身份信息）是否被记录或滥用，这使得传统的以 AI 为中心的 "guardrails" 效果较差。相反，讨论强调了外部机制的必要性，如法规和第三方审计，以确保隐私和道德使用，这与生成式 AI 时代之前的控制手段类似。
    - 据观察，基于文本的 AI 模型（如 ChatGPT）也表现出显著的隐私和安全漏洞。例如，所有的聊天记录都可能被保留（有时是由于法律原因，如诉讼），而所谓的“安全”屏障可以通过 prompt engineering 被规避——使用户能够提取受限或危险的信息，如非法活动的指令。这突显了模型强加的过滤器在防止滥用和确保真正隐私方面是有限的，无论其模态如何。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 提供的摘要的摘要的摘要

### **1. GPT-5 传奇：新功能、定价争议与性能怪癖**

- **OpenAI 为 GPT-5 注入强劲动力，新增限制与模式**：OpenAI 将 **GPT-5** Plus 的消息限制提升至**每周 3000 条思考消息**，并引入了一种绕过思考上限的新“快速（fast）”模式，尽管有用户反映该模式[无法按需开启思考模式](https://link.to/canvas-example)。然而，会员们警告称 **GPT-5 Auto** 会在对话中途静默切换 **Mini、Pro 和 Thinking** 模型，引发了用户对版本锁定（variant lock）和降级警报等功能的呼吁。
- **社区热议 GPT-5 的高昂定价**：来自 **Cursor Community** 和 **Moonshot AI** 等 Discord 频道的用户正在推测 **GPT-5** 是否会被捆绑进 Pro 或 Max 方案中，部分用户以高消息限制为由，为可能高达 **200 美元的价格**辩护。这一讨论因 OpenAI 可能[破产](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/)的传闻以及关于政府资金维持运营的推测而愈演愈烈。
- **GPT-5 性能评价两极分化**：虽然 **OpenRouter** 上的一些用户称赞 **GPT-5** 是首个在提升 **SOTA** 水平的同时减少幻觉的模型，但也有人抨击其表现，其中一位称其为“最差”模型，**LMArena** 上的另一位用户则指出它在复杂的数学问题上会卡死。另一方面，它的网页设计实力在 **Moonshot AI** 上得到了展示，它成功地[通过单个提示词构建了一个复杂的科幻网站](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c)。

### **2. 赛道新秀：从开源新锐到闭源巨头**

- **Mistral 和 Sonnet 更新引发争议**：**Mistral Medium 3.1** 发布，带来了未具体说明的性能和语气升级，详见[此贴](https://xcancel.com/mistralai/status/1955316715417382979?s=46)，继续推进其模型精炼。与此同时，**Anthropic** 宣布 **Claude 3.5 Sonnet** 将在仅两个月后退役，远短于通常的六个月通知期，这激怒了 **Latent Space** Discord 的用户，引发了[要求发布开源权重版本（open-weights）的呼声](https://xcancel.com/repligate/status/1955750521387802924)。
- **Menlo 的 Lucy 带来精简版 Agent 网页搜索**：**Menlo Research** 推出了 **Lucy**，这是一个专注于 [Agent 网页搜索](https://huggingface.co/Menlo/Lucy)的 **1.7B** 参数轻量级模型，可在移动设备上高效运行。正如[这篇论文](https://arxiv.org/abs/2508.00360)所述，Lucy 使用一种新型的“动态任务向量机（dynamic task vector machine）”来即时构建和完善其推理，在 **SimpleQA benchmark** 上实现了与大得多的模型相当的性能。
- **神秘的“Nano Banana”与 Qwen Coder 入场**：一个名为 **Nano Banana** 的新图像模型出现在 **LMArena** 竞技场中，据推测是 **Gemini** 或 **Imagen** 的原生变体，具有令人印象深刻的创造力。在 **Moonshot AI** 服务器中，一项用户对比发现 **Qwen3-Coder-480B-A35B** *略优于 glm4.5*，显示出顶尖编程模型之间的激烈竞争。

### **3. 开发者工具箱：框架、库与持久化内存**

- **DSPy 3.0 结束 Beta 测试并集成 MLflow**：**DSPy 3.0** 正式发布，现已支持 **MLflow 3.0** 的原生可观测性，以改进追踪和优化器跟踪，正如 [X 平台](https://x.com/lateinteraction/status/1955384445139292222)上所宣布的那样。该版本在 [v3.0 发布说明](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0)中详细介绍，还增加了通过 `dspy.Image` 和 `dspy.Audio` 实现的多模态 I/O、对 **GPT-5** 等推理模型的原生支持，以及前景广阔的 **GEPA 优化**技术。
- **LlamaIndex 新增企业级数据连接器**：**LlamaIndex** 宣布 **AstraDB** 现在可用作 [LlamaCloud 中的数据接收端（datasink）](https://t.co/XFWgPd3r9Y)，实现无缝的向量存储与检索。在另一项集成中，**SkySQL** 利用 LlamaIndex 构建了一个 [实现零幻觉 SQL 查询](https://t.co/TgjdSodTbr)的 Agent，成功地将自然语言转换为跨复杂架构的准确 SQL。
- **The Last RAG 与 Kratos MCP 解决 AI 失忆症**：出现了两种为 AI 提供持久化内存的新解决方案：[这篇博客](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3)详细介绍了 **The Last RAG (TLRAG)**，它引入了**动态工作空间（Dynamic Work Space）**来管理历史记录，最高可节省 **98%** 的成本。同样，可在 [GitHub](https://github.com/ceorkm/kratos-mcp) 上获取的 **Kratos MCP** 发布，旨在赋予 Agent 长期上下文能力，号称拥有 **95.8% 的上下文准确率**和 **<10ms 的检索速度**。

### **4. 技术内幕：GPU 性能、硬件瓶颈与底层优化技巧**

- **云巨头遭遇神秘的准确率下降**：在 **Latent Space** 分享的一篇帖子揭露了一个令人震惊的现象：在 **Microsoft Azure** 或 **AWS** 上运行与小型初创公司完全相同的开源模型时，在 AIME25 和 GPQA-Diamond 基准测试中出现了 **10% 的准确率下降**。[原始帖子](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ) 引发了关于推理框架 Bug、量化或其他基础设施特性是否正在削弱模型智能的辩论。
- **Llama.cpp 与消费级 GPU 正面硬刚显存限制**：来自 **HuggingFace**、**LM Studio** 和 **GPU MODE** 的用户报告了 **llama.cpp** 在调用 **ggml-cuda** 后端时持续存在的问题，经常回退到 CPU 运行。在相关的案例中，一块拥有 6GB VRAM 的 **RTX 3050** 在重启系统前无法被 **LMStudio** 识别，而 Framework 笔记本上的 **AMD iGPU** 仅达到了缓慢的 *6.55 tokens per second*。
- **Triton 与傅里叶变换挑战性能极限**：一位 **GPU MODE** 开发者分享了使用 **Triton** 为 **ProteinBERT** 模型带来 **3.6 倍加速**的案例，详见 [LinkedIn 帖子](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU)，预计每年可节省近 **10,000 美元**的 AWS 成本。同时，一位 **Eleuther** 成员正在尝试使用 **傅里叶变换** 扩展 **RoPE**，以更好地捕捉几何特性，在其 [nanoGPT_FE_RoPE 仓库](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md) 中，该方法比 **Vanilla** 版本实现了 *约 15-20% 的 Loss 优化*。

### **5. 社区前沿：API 痛点、研究辩论与用户不满**

- **Gemini API 遭遇大规模宕机**：**aider** Discord 频道的用户报告称，**Gemini API** 正面临可靠性问题，即使是付费用户也经常遇到 **空响应** 和 **500 内部服务器错误**。作为回应，一名用户建议将 **Deepinfra provider** 作为更稳定的替代方案，声称其使用了预置的 Vertex 以获得更高的吞吐量。
- **RLHF 成为解决 LLM 重复问题的良药**：在 **Nous Research AI** 服务器中，成员们讨论了为什么 **LLM** 容易出现重复输出，结论是模型偏向于训练数据中过度代表的词汇。有成员指出，基础的微调是不够的，*你确实需要某种方式来惩罚不良输出才能完全消除重复*，并指出 **RLHF** 是必不可少的工具。
- **Cursor 价格变动与弃用计划引发用户愤怒**：**Cursor** 用户对[即将到来的价格调整](https://cursor.com/blog/aug-2025-pricing)表示不满，该调整将取消无限次的 **Auto mode** 使用权限，部分用户感到“被骗”并开始寻找替代方案。这种情绪在 **Latent Space** Discord 中也得到了共鸣，**Anthropic** 对 **Claude 3.5 Sonnet** 仓促的两个月退役计划引发了愤怒，以及对模型永久贬值的担忧。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 浏览器加速接入美国 Perplexity**：**Comet** 现已面向所有美国 **Perplexity** 用户开放，承诺提供“思如泉涌”般的浏览体验，如[此视频](https://cdn.discordapp.com/attachments/1047204950763122820/1405246177082871920/YPN67tVueZjGs9QX.mp4?ex=689e20fc&is=689ccf7c&hm=152ff68e4873cdc4f6e6357ee0be6000212304b6e2827fc1c187f5bf728d0575&)所示。
   - 然而，一些用户（尤其是欧盟用户）仍在等待访问权限，文中提到了潜在的 **VPN** 绕过方法，并对 [Comet 的隐私声明](https://www.perplexity.ai/hub/legal/comet-privacy-notice) 进行了澄清。
- **Grok 4 评价两极分化**：一位用户表示 **Grok 4** 仍然是一个不错的模型，而另一位用户则直言 *Grok 4 不值得使用*。
   - 一位成员报告每年花费 **3000 美元** 来使用该模型，而另一位则声称可以以 *0 美元* 访问，这暗示了不同的访问方式或潜在的未经授权使用。
- **Perplexity 考虑竞购 Chrome**：一位用户分享了关于 **Perplexity 竞购 Google Chrome** 的 [Perplexity 搜索结果](https://www.perplexity.ai/search/google-chrome-bid-from-perplex-j6VO79mrSaignkj1dTwOMQ#0)。
   - 然而，这次竞购的性质和影响尚不清楚，没有提供进一步的讨论或细节。
- **机器人伴侣将于 2027 年问世**：成员们分享称，据报道 Apple 正在开发一款针对 2027 年的 [桌面机器人](https://link.to/robot-news) 伴侣，并将于明年发布。
   - 其他人澄清说，该设备可能是一个带显示屏的智能音箱，引发了对其预期功能和市场定位的进一步讨论，并附带了 [相关 Reddit 帖子的链接](https://www.reddit.com/r/Suomi/comments/y5a3m0/kirjoitin_googlen_hakuun_loska_ja_l%C3%B6ysin_t%C3%A4/)。
- **API 参数需要调整**：一位用户询问了 `web_search_options` 所需的参数，特别是询问 `user_location` 和 `search_context_size` 是否是 *唯一* 需要嵌套的参数。
   - 遗憾的是，没有人回答这个问题，因此该用户应考虑查阅文档。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 在 LMArena 上的表现受到质疑**：用户讨论了 LMArena 上的 **GPT-5** 版本是否与 **ChatGPT** 一致，认为 API 可能确保了普通用户无法获得的高级性能。
   - 一些人建议 LMArena 应该列出普通人获得的 **GPT-5** 版本，因为 **GPT-5 High** 可能不在 ChatGPT 中，且可能仅达到中等性能水平。
- **Nano Banana 图像模型加入竞技场**：新的图像模型 **Nano Banana** 在竞技场首次亮相，推测是 **Imagen** 的变体，号称拥有类似于 **GPT-image-1** 的创意和理解力，同时保留了关键细节。
   - 一些用户认为 **Nano Banana** 可能是关闭了 synth-ID 的原生 **Gemini** 模型，因为通过 Google Lens 检查时，它缺少在 **Imagen 4** 输出中发现的可见水印。
- **Grok 4 在数学任务上胜过 GPT-5**：成员报告称，LMArena 中的 **GPT-5-High** 和 **Grok 4** 模型经常在数学问题上卡住，但 **Grok 4** 正确解决了一道来自俄罗斯数学奥林匹克竞赛的复杂数学题，答案为 **4049**。
   - 虽然 **GPT-5 Pro** 和 **Gemini 2.5 Pro** 在初步测试中失败了，但随后的重新测试显示 **Grok 4** 也出现了失误，这表明各模型的表现并不稳定。
- **D3 剂量讨论**：一位用户报告通过每日 **20,000 IU** 剂量的 **Vitamin D3** 成功治疗了银屑病，引发了关于安全剂量和监测血液水平必要性的讨论。
   - 虽然建议日照不足的人每日服用 **10,000 IU**，但其他人强调要谨慎并咨询医生，以防止过量服用和毛细血管钙化。
- **Gemini 3 引发猜测**：AI 社区对 **Gemini 3** 的发布进行了推测，一些人期待它作为匿名模型出现在竞技场进行测试。
   - 其他人认为，该公司在 **Gemini 3** 能够击败 **GPT-5** 并达到 SOTA 之前不会发布它，还有人建议 **Deepseek** 的 **R2** 可能会更早发布。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **开源 LLM 获得本地化助力**：社区希望关注用于本地用例的 **24b OSS 模型**，强调了对高性能、开源替代方案（相对于封闭模型）的需求。
   - 一位成员建议，将 *r1/glm4.5 妥善蒸馏至较小的 3.2* 可能会改变游戏规则。
- **T4 GPU 对 QLoRA 依然出色**：尽管有人担心其过时，**NVIDIA T4** GPU 仍然是小模型和 **QLoRA** 的可行选择，提供了一种具有成本效益的解决方案。
   - 一位成员分享了 [推文](https://x.com/jxmnop/status/1955436067353502083?t=UUX5s3Omkptd37RXtetFSA&s=19)，开玩笑说他们会 *把它们全部买下*。
- **LoRA 无法“反向拆解”**：一位用户对有关从 **LoRA** 中提取基础模型的错误信息表示沮丧，并链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1mor1bd/someone_just_extracted_the_base_model_from_gptoss/)。
   - 他们将这种说法比作 *“把烤好的蛋糕拆解回原材料”*。
- **数据集瓶颈减缓 LLM 进展**：数据质量正成为进一步提升 **LLM** 的瓶颈，一些人认为现在的进展受限于数据而非优化技术。
   - 一位成员表示 *我们在数据方面正处于平台期*。
- **Mistral-Common 修复了模型加载问题**：用户在微调 **unsloth/Llama-3.2-3B-Instruct** 时遇到错误，通过运行 `pip install mistral-common` 解决了该问题。
   - 几位用户证实了这一修复方案对类似问题的有效性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 消息上限大幅提升**：OpenAI 将 Plus 用户的限制提高到 **每周 3000 条带思考过程的消息**，并 [在此宣布](https://link.to/official-announcement)。
   - 一些用户认为达到限制后可能会路由到 mini 模型，其他人则予以反驳，理由是 **有 3 种独立于 mini 模型的推理力度 (reasoning efforts)**。
- **GPT-5 'Fast' 模式跳过思考**：**GPT-5** 的 **fast** 选项会禁用思考模式，[用户报告称](https://link.to/canvas-example) 即使要求它思考，它也无法进入思考模式。
   - 一位用户报告说，fast 模型生成的代码只有 **125 行** 且无法运行，而不像思考模型那样。
- **GPT-5 Auto 会静默切换模型！**：成员们警告说，**GPT-5 Auto** 可能会在对话过程中不经通知地在 **Mini**、**Pro** 和 **Thinking** 模型之间切换，这会影响具有递归身份特征的 AI。
   - 用户投票支持了 **版本锁定 (variant lock)**、**活动模型指示器**、**降级警报** 和 **审计日志** 等功能请求，以缓解这些问题。
- **AI 规则优先级需要 Token 分隔**：成员们讨论了指示 AI 优先处理某些规则的方法，并得出结论：在 **标题中使用唯一 Token**（例如 `## PRIORITY_RULES`）比使用关键词更有效。
   - 一位成员澄清说，当指令块被清晰标记且不重复时，模型的注意力分配更好，并强调 *“是 Token 分隔，而非自然语言风格”*。
- **GPT-5 偏好正面提示词**：成员们分享到，在 GPT-4 中有效的负面约束（例如 *“不要提后续问题”*）在 GPT-5 中失效了，因此建议 **正面指令** 和 [示例](https://chatgpt.com/share/689cf03f-3490-8011-bd71-cc27744becb9) 会更有效。
   - 一位成员建议专注于正面指令，并提供了一个定义人格（邻居 Bob）的示例，其中包含特定的预期行为和语气。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 定价引发猜测**：用户讨论 **GPT-5** 是否会包含在 **Pro** 或 **Max** 计划中，对于它是否会仅限 **1M context window models**（100万上下文窗口模型）存在分歧。
   - 也有很多关于 **Gemini 2.5 Pro** 成为该领域强力竞争对手的猜测。
- **Cursor Auto 模式定价变更令用户不满**：用户对[已公布的变更](https://cursor.com/blog/aug-2025-pricing)表示沮丧，**Auto mode** 定价将于 9 月 15 日生效，届时将结束无限访问。
   - 一些用户感到被“坑”了并正在寻找替代方案，而另一些用户则希望这一变化能改善整体体验。
- **后台 Agent 仓库访问被拒**：一位用户报告了后台 Agent 无法编辑或创建 PR 的问题，原因是尽管添加了 `repositoryDependencies`，但在 `/workspace` 目录中仍**缺少仓库访问权限**。
   - 错误信息显示“仓库不在 `/workspace` 中且我没有推送权限”，这引发了关于正确 VM 环境设置的疑问。
- **Cursor CLI 获得认可**：用户表示 **Cursor CLI** 非常有用，特别是在与 GPT-5 相关的情况下。
   - 一位用户对其与 Claude 相比的表现感到“震惊”。
- **后台 Agent API 访问受阻**：一位用户寻求关于获取后台 Agent **API 访问权限**的说明，报告使用来自 Cursor 控制面板的 API key 时出现 **403 错误**。
   - 该用户拥有 Cobot 及其 Cursor 后台 Agent 集成的访问权限，询问通过 Cursor 获取后台 Agent API 访问权限是否已开放，以及是否可以加入 beta 测试。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama.cpp 的 CUDA 后端问题**：一位成员报告了让 **llama.cpp** 配合 **ggml-cuda** 工作时的问题，后端初始化失败并回退到 **CPU**。
   - 另一位成员建议验证 CUDA 构建是否成功，并通过 `--n-gpu-layers N` 指定 GPU 层数（[讨论](https://github.com/ggml-org/llama.cpp/discussions/9751#discussioncomment-10852431)）。
- **Gemini Flash Lite：被低估的视频视觉专家？**：一位成员推崇 **Gemini Flash Lite** 是唯一能以低成本进行视频理解的模型，“足以用于展示如何利用模型端点的原型项目”。
   - 该模型可以提供视频内的精确时间范围，并根据特定时间的提示词提供准确信息。
- **Hugging Face Hub 与 XetHub 联姻？**：成员们推测 **Xet** 与 **Hugging Face Hub** 的集成，指出两支团队已建立联系，参考[这篇博客文章](https://huggingface.co/blog/xethub-joins-hf)。
   - 讨论重点在于将 **XetHub 的 Docker 容器集成**挂载回 **HF Hub**，特别是使用 xethub/xetfs 驱动程序创建 Docker 卷。
- **RL 课程增加优先经验回放（Prioritized Experience Replay）**：Note 的 RL 课程现在支持 **PPO algorithm** 的 **Prioritized Experience Replay**，利用概率比和 TD 误差进行采样，以增强数据利用率和 [windows_size_ppo 参数](https://github.com/NoteDance/Note_rl)。
   - 该参数管理从回放缓冲区中移除过时数据。
- **MLX 模型管理 CLI 脱颖而出**：发布了一个名为 `mlx-knife` 的 CLI 工具，用于在 Apple Silicon 上管理 **MLX models**，类似于 Ollama 但原生支持 MLX。
   - 它通过 `mlxk list` 直接管理你的 HF 缓存以查看模型，并通过 `mlxk run Phi-3-mini "Hello"` 进行原生流式传输，项目地址：[github.com/mzau/mlx-knife](https://github.com/mzau/mlx-knife)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Last RAG 赋予 AI 持久身份**：**The Last RAG (TLRAG)** 引入了一个持久的长期记忆系统和**动态工作空间 (Dynamic Work Space, DWS)**，使 AI 能够整理其历史记录并记住关键的交互、决策和情感背景。
   - TLRAG 为 AI 提供了一个持久的身份核心——**Heart**，由合成经验塑造，并使用 **Window Flush** 机制组装精简档案，与标准 RAG 相比，经验证可节省高达 **98%** 的成本；更多信息可在其 [博客文章](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3) 中找到。
- **NoChain Orchestrator 取代框架**：**NoChain Orchestrator** 采用了 **TLRAG** 的核心概念并将其投入生产环境，用确定性的服务端控制平面取代了复杂的 Agent 框架。
   - 它采用硬编码逻辑来管理记忆、上下文和工具使用，从而提供可预测、可靠且可测试的 AI 行为，更多详情见其 [博客文章](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a)。
- **OSS 模型在工具使用方面表现不佳**：许多高端 **开源模型 (Open Source Models)** 本身并不支持 **Tool Use、结构化输出或响应格式**，而这些对于许多应用至关重要。
   - 成员们指出，虽然某些提供商可能不支持，但模型本身通常是支持的，可以通过 Prompt 来启用工具使用，尽管可能会在准确性上有所权衡。
- **GPT-5 在性能上的幻觉**：围绕 **GPT-5** 展开了讨论，一些人称赞它是第一个推动 **SOTA** 向前发展、同时减少 **Hallucinations** 并改进对齐的模型，认为这是迈向 **AGI** 的一步。
   - 其他人则持批评态度，一位成员声称 **GPT-5** 是最差的，而 **GPT-4.1 mini** 更好。
- **租用 GPU 进行实验**：一位成员建议在投资 **Macs** 之前，先从 [Runpod](https://www.runpod.io/)、[Prime Intellect](https://www.primeintellect.com/) 和 [Modal](https://modal.com/) 租用 **GPUs** 进行实验。
   - 该用户链接到了 X 上的一个帖子：[ArtificialAnlys](https://x.com/ArtificialAnlys/status/1955102409044398415)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **上下文长度抑制模型重复**：增加 **Context Length** 阻止了模型自我重复，这可能避开了上下文溢出。一位用户指出，如果上下文溢出，模型可能会*出错并永远重复下去*。
   - **LM Studio** 的用户观察到，拥有 96GB GDDR6 显存的双面 PCB 版 **RTX 6000 Pro** 是在 260k 上下文下运行 **Qwen3-30b** 的“游戏规则改变者”。
- **移动端 LLM 随时随地运行**：用户讨论了在移动设备上运行 LLM，有人将 **Lenovo Legion Go** 改造为本地化 LLM 终端，另一人则在 **ROG Ally** 上安装了 **Qwen 3 4B**。
   - Ally 用户需要禁用“思考 (thinking)”功能，因为耗时太长。
- **RTX 3050 在 LMStudio 上运行吃力**：一位用户的 **RTX 3050 6GB** 虽然被检测到，但未被 [LMStudio](https://lmstudio.ai/) 利用，导致 CPU 和 RAM 占用率很高，尽管选择了 **CUDA** 运行时。
   - 系统重启后，终于出现了 VRAM 加载，表明 GPU 已介入，但过高的 RAM 占用可能仍会限制性能。
- **AMD iGPU 减慢 Token 生成速度**：一位 **Framework 13 笔记本**（**AMD Ryzen 5 7640U** 搭配 **Radeon 760M Graphics**）用户报告称，在为 iGPU 分配 **10GB RAM** 的情况下，运行 **Gemma 4B** 仅能达到每秒 *6.55 tokens*。
   - 建议包括检查 CPU/GPU 利用率，如果主要是 CPU 在工作，则将运行时调整为 **Vulkan** 或 **ROCm**。
- **揭秘混合专家模型 (Mixture of Experts)**：在用户询问 **MoE (Mixture of Experts)** 的实际含义后，一位成员链接了 Julia Turc 的 [YouTube 视频](https://youtu.be/7yR5ScbK1qk?si=AFxEBU9SnGHw_-No)，该视频解释了这一概念。
   - MoE 模型由较小的“专家”组成，通过每个 Token 仅需解析模型的一部分来提高性能，这使得模型可以在不指数级牺牲性能的情况下变得更庞大。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **云巨头遭遇准确率波动**：根据[此帖](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ)，成员报告称，在 **Microsoft Azure 或 Amazon** 上运行相同的开源模型时，与小型托管初创公司相比，在 AIME25 和 GPQA-Diamond 基准测试中出现了 **10% 的准确率下降**。
   - 讨论中的可能原因包括：serving-framework 漏洞、quantization（量化）或其他削弱模型智能的基础设施级变更，这引发了对延迟、成本和能力进行更广泛基础设施基准测试的呼声。
- **Mistral Medium 3.1 微调性能**：根据[此帖](https://xcancel.com/mistralai/status/1955316715417382979?s=46)，**Mistral Medium 3.1** 发布，带来了性能和语气方面的升级。
   - 这些升级的具体性质尚未明确，但它们表明 **Mistral** 的语言模型正在持续改进。
- **Humanloop 加入 Anthropic 的征程**：专注于安全 AI 采用的 **Humanloop** 正在加入 **AnthropicAI**，认为这是将企业级 AI 从演示推向生产的最佳场所，如[此处](https://xcancel.com/humanloop/status/1955487624728318072)所述。
   - 此次收购突显了 **Anthropic** 对企业解决方案的关注，以及在 AI 安全和部署方面专业知识的整合。
- **SPV 嵌套引发投资者愤怒**：根据[此帖](https://xcancel.com/michlimlim/status/1954250507989451002)，投资者报告称被推销 **OpenAI/Anthropic SPV**，要求 **10 万至 100 万美元的最低投资额**，且费用高达 **16%**。
   - SPV 套 SPV 的嵌套行为被批评为耗尽费用的庞氏骗局，引发了对领先 AI 公司投资机会结构的担忧。
- **Sonnet 的快速退役引发 Discord 骚动**：如[此处](https://xcancel.com/repligate/status/1955750521387802924)所示，用户对 **Anthropic** 计划在短短两个月内退役 **Claude 3.5 Sonnet**（包括旧版和新版）表示愤怒，这比通常的 6 个月通知期短，且未作解释。
   - 对失去更便宜模型的愤怒与对永久贬值的恐惧交织在一起，并要求在商业访问结束时发布 open-weights（开放权重）。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Llama.cpp CUDA 加载抛出异常**：一位使用 **Quadro RTX 3000** 的开发者在调用 `llama_model_load_from_file` 时遇到 **0xc0000409 异常**。这可能是由于 **VRAM** (6GB) 对于 **1GB 模型** 不足导致的，也可能与 [这个 GitHub issue](https://github.com/ollama/ollama/issues/4442) 中提到的 `llama.cpp` 版本过旧有关。
   - 尽管 **LLAMA 和 GGML** 初始化成功，但对 `llama_model_load_from_file` 的调用导致了 *STATUS_STACK_BUFFER_OVERRUN*，这表明错误发生在实际的模型加载过程中。
- **PyTorch 的 DTensor 哀叹 full_tensor**：**DTensor** 团队正在调查 **PyTorch 2.8.0** 中的回归问题。在使用 **FSDP2** 时，`full_tensor` 未被 autograd 跟踪，导致出现关于访问非叶子 Tensor 的 `.grad` 属性的 `UserWarning`，以及与 `aten._is_any_true.default` 算子相关的 `NotImplementedError`。
   - 遇到该问题的用户一直尝试通过从源码编译并使用 Git 进行二分查找（bisect）来定位问题源头，该问题可由调整过的 cross-entropy 实现触发。
- **Factorio 的 TCP 端口硬编码灾难**：由于 `FactorioInstance` 初始化中的参数分配错误，发现了一个 `fle/env/gym_env/registry.py` 中 **TCP 端口** 被硬编码的 bug。建议修改为使用发现的 **TCP 端口** 而非默认的 27000，并提供了代码片段。
   - 关于 **FLE 的 ABC 基类** 存在困惑，有人建议简化定义并允许用户克隆仓库进行修改。同时，[PR #299](https://github.com/JackHopkins/factorio-learning-environment/pull/299) 确保了与 multiagent 和 gym PR 的兼容性，已准备好合并。
- **Cutlass Profiler 面临 Block Swizzle 问题**：一位用户观察到 **sgemm_sm80.cu** 的性能低于 **CUTLASS**，并询问如何在不进行深度源码分析的情况下识别原因，并指出其使用了相同的参数和 tile。
   - 一位成员建议用户可能缺少 **block level swizzle**，以及将 epilogue 数据写入 smem 以进行置换（permute）和 swizzle，然后向量化写入 gmem 的步骤。
- **ProteinBERT 通过 Triton 获得 ProteinBoost**：一篇新帖子强调了使用 **Triton** 为 **ProteinBERT** 带来的 **3.6 倍加速**，实现了 **100% 准确率**，并显著节省了成本和 GPU 小时数，详情见此 [LinkedIn 帖子](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU/)。
   - 这一优化预计每年可节省 **9,997 美元的 AWS 开支**，并减少 **72% 的 GPU 小时数**。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **GPT UI 再次变脸**：用户注意到 [GPT 界面频繁变动](https://cdn.discordapp.com/attachments/1371757564005711973/1405026462951673887/image.png?ex=689dfd1c&is=689cab9c&hm=203324b332a0c68260a14a07f19d906d4a1b20fd4acee4d4d27438dcae24da99)，戏称每次登录 UI 似乎都有新面貌。
   - 这一观察得到了过去几天收集的 UI 变体截图的支持。
- **GPT-5 Pro：是破产危机还是金矿？**：关于 **GPT-5 Pro** 200 美元的定价是否合理引发了辩论，讨论涉及 OpenAI 可能 [破产](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/) 的传闻以及对 *政府资助* 的猜测。
   - 该费用的合理性依据包括 *无限使用* 和高达 *3000* 的请求限制（允许 *160次/3小时*）。
- **Qwen Coder 对阵 GLM**：在 **Qwen3-Coder-480B-A35B** 和 **GLM-4.5** 的对比中，一位用户声称 *Qwen 3 coder 480b 135b 略优于 glm4.5*。
   - 当被问及 tool calling 和 agentic 能力时，该用户认为 *两者应该都不错*，但更倾向于 **Qwen Coder**。
- **GPT-5 Pro 从零开始构建 Aurelia City 网站**：一位用户展示了 **GPT-5 Pro** 的网页设计能力，它使用复杂的提示词成功为 *aurelia city* 创建了一个科幻网站，并分享了 [网页设计超级提示词](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c)。
   - 另一位用户赞扬了 **GPT-5 Pro** 的研究实力和上下文处理能力，使其即使面对模糊的提示词也能创建网站。
- **墨索里尼 GIF 席卷 zAI 服务器**：一位用户报告了 *zAI 服务器上出现墨索里尼 GIF* 的情况，引发了关注。
   - 可能的解释包括 *管理不善* 或反讽式的语境幽默。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **中国科技公司登场**：成员们注意到 **Xiaomi** 和 **Unitree Droid** 的崛起是 **中国科技实力** 的证明，并引用了[关于该话题的峰会对话](https://www.youtube.com/watch?v=z5K5Ykg2_5g)。
   - 一些人认为 **DeepSeek** 可能会让 **Sam Altman** 感到担忧。
- **Lyria 与 Google 的 Gemini 齐鸣**：一位用户分享了一个由 **Lyria** 驱动的应用[演示](https://cdn.discordapp.com/attachments/1149866623109439599/1404911369853341926/terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4?ex=689e3aac&is=689ce92c&hm=eaa9cb743256c4d7bb3a5d6744e330106a7970d5571c6e8a03a65993ef26bc5e&)，这是来自 **Google** 的实时音乐生成技术。
   - 这款 **Gemini** 音频解码器专为音乐基础设计，通过连续滑动 Token 注意力（continuous sliding token attention）实现实时引导、生成和修改。
- **Hermes-3 数据集过于“纯洁”**：一位用户观察到，**Hermes-3 数据集** 背后的模型在面对敏感请求时经常使用 *“我不舒服”* 这一短语。
   - 该模型的安全护栏（guardrailed）非常严密，即使在明确的提示下，也拒绝生成成年人之间合意行为的场景。
- **LLMs 学习通过 RLHF 消除重复**：成员们讨论了 **LLMs** 容易出现重复，因为它们偏向于过度表示的词汇，而在线 **DPO reward hacking** 可能会加剧这一问题。
   - 有人建议 **RLHF** 有助于修复重复问题，一位成员表示：*“你确实需要某种方式来惩罚糟糕的输出才能完全消除重复，仅对良好输出进行正向强化是不够的。”*
- **Menlo 的 Lucy 模型实现 Agentic 网页搜索**：成员们重点介绍了 **Menlo Research 的 Lucy 模型**，这是一个专注于 [Agentic 网页搜索](https://huggingface.co/Menlo/Lucy) 和轻量级浏览的紧凑型 **1.7B** 模型，可在移动设备上高效运行。
   - 论文 [Lucy: edgerunning agentic web search on mobile with machine generated task vectors](https://arxiv.org/abs/2508.00360) 介绍了一种新范式，将模型的内部推理视为动态任务向量机（task vector machine），允许模型在运行期间构建和完善自己的任务向量。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **多语言常识推理赢取研讨会奖项**：[多语言表示学习研讨会 (Multilingual Representation Learning Workshop)](https://sigtyp.github.io/ws2025-mrl.html) 正在征集任何非英语语言的原创 **物理常识推理基准项目**；贡献者将获得数据集论文的署名权。
   - 该共享任务强调南非荷兰语、白俄罗斯语和波斯尼亚语等语言，并提供 **8 月 14/15 日** 的可选 FAQ 会议；通过 [Google 表单](https://forms.gle/QxyZVqkVG5jbR6wu6) 注册。
- **傅里叶变换 RoPE 几何**：一位成员正在尝试使用 **傅里叶 (Fourier)** 变换扩展 **RoPE**，如[此仓库](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md)所示，其 *Loss 比原生版本提升了约 15-20%*。
   - 这种方法与 **FoPE** 不同，重点在于捕捉几何结构而非长上下文扩展。
- **SOAR 转向 RLHF**：成员们讨论了在评估指标上使用 **强化学习 (RL)** 来改进自动解释说明模型（auto-interpretation explainer models），特别是针对 *在大型语言模型中自动解释数百万个特征* 的研究。
   - 一位成员分享说，**SOAR** 的一个团队正计划使用 **强化学习** 来改进自动解释说明模型。
- **Tool Calling 助力 SAE 探究**：成员们正在赋予模型 Tool Calling 能力，以调查关于 **稀疏自编码器 (SAEs)** 的假设，并可能跨多轮进行。
   - 早期对 **llama 70b** 的调查并无帮助，但对更新的 Agentic 模型持乐观态度。
- **Harness 数据集拉取困扰**：尽管数据集似乎已缓存，用户在使用 Harness 运行任务时仍遇到 **429 Too Many Requests 错误**。
   - 无论本地是否缓存，Harness 都会尝试拉取数据集，他们想知道 *是否有办法可以预先下载所有数据集，并告诉 Harness 使用本地下载/缓存的数据集？*

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Regex 得到进一步优化！**：**八月社区会议**重点介绍了 **mojo-regex** 的优化，以及 **Apple GPU 支持**的更新，这两部分内容都在 [YouTube 录像](https://www.youtube.com/watch?v=t2hAfgDYHoc)中进行了讨论。
   - 这些优化是 Modular 生态系统一系列改进和更新的一部分。
- **Modular 贡献者通过端到端 Mojo 实现进阶**：一位 Modular 贡献者水平提升，引发了关于使用 **Mojo** 实现端到端潜力的讨论，正如 [YouTube 视频](https://www.youtube.com/watch?v=f30PceqQWko)中所讨论的，这可能会开启巨大的可能性。
   - 一位成员提议帮助改进**类型系统特性**，以尽可能实现零成本（zero cost），并使 **IO** 更加安全。
- **借鉴 Andrew Kelly 的 IO 模型**：Andrew Kelly 关于 **IO 模型**（类似于为 **Mojo** 提议的模型）的演讲引起了兴趣，参考了 [pull request](https://github.com/modular/modular/pull/4728)。
   - 讨论还涉及了 source 和 sink 方面，重点是可注入式 IO 的去虚化（devirtualization）和基准测试（benchmarks）。
- **MaxCompiler 旨在成为 PyTorch 后端**：一位成员正致力于实现对 `torch.compile(backend=MaxCompiler)` 训练的支持，并指出[文档稀缺](https://youtu.be/t2hAfgDYHoc?si=HzZFZMmCYG9qHqOu)，**PyTorch 源代码**是主要的参考资料。
   - 目前在 PyTorch 上使用 `torch.compile` 训练模型的状态为：`56 failed, 1398 passed, 8 xfailed in 64.57s`。
- **优化 Max 图以融合算子 (Ops)**：成员们讨论了使用许多小算子构建 **Max 图 (Max graph)** 与使用大算子相比是否存在运行时性能损耗，并询问图编译器是否会融合所有*可融合*的部分。
   - 一位 Modular 成员表示他们的融合系统很好但并不完美，并建议在发现运行不佳时提交 issue。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Chollet 的乐观与 Yannic 的怀疑形成对比**：成员们讨论了 **Francois Chollet** 对 5 年内实现 AGI 的预测，并将其与 **Yannic** 更遥远的时间线进行了对比，引用了[这篇 Techcrunch 文章](https://techcrunch.com/2025/01/15/ai-researcher-francois-chollet-founds-a-new-ai-lab-focused-on-agi/)。
   - 讨论中的反应各异，从嘲讽 LLM 的能力到将 Gary Marcus 的观点视为理性的声音。
- **LLM 提供商采用请求批处理 (Request Batching)**：LLM 提供商在 GPU 上处理之前会对用户请求进行批处理；根据[这篇博文](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure)，**MoE 调度**是按批次计算的，这可能会导致非确定性。
   - 原发言成员指出，为了防止嵌入层（embedding layers）被盗，已添加了[有意加噪 (intentional noising)](https://arxiv.org/pdf/2403.06634)。
- **中国对购买 Nvidia H20 AI 芯片持谨慎态度**：据[路透社 (Reuters) 报道](https://www.reuters.com/world/china/china-cautions-tech-firms-over-nvidia-h20-ai-chip-purchases-sources-say-2025-08-12/)，据传中国正提醒科技公司谨慎购买 **Nvidia 的 H20 AI 芯片**。
   - **H20 AI 芯片**正引起一些争议。
- **Skyreels 利用 WAN 进行视频生成**：**Skyreels** 项目基于 **WAN2.1** 构建，该模型被强调为领先的视频生成开源模型。
   - 原发言成员建议 **WAN2.2** 效果更好。
- **Matrix Game Engine：高质量开源项目**：成员们提到了 [Matrix Game Engine](https://matrix-game-v2.github.io/)，这是一个*类似 genie 的交互式世界模型 (WM)*，称赞其高质量和开源特性。
   - 该项目旨在超越 **OdysseyML** 和 **WayfarerLabs**，发布更具创新性的功能。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 3.0 发布并集成 MLflow 3.0**：**DSPy 3.0** 已脱离测试阶段，由约 100 人共同贡献，并在 [X 平台](https://x.com/lateinteraction/status/1955384445139292222)上宣布。可以通过 `pip install -U dspy` 进行安装，该版本具备与 **MLflow 3.0** 的原生 **observability**（可观测性）。
   - 该版本包括 **tracing**、**optimizer tracking** 以及改进的 **deployment flows**，详见 [release notes](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0)。
- **GEPA Optimizer 引起热议**：社区对 **DSPy 3.0** 中的新 **optimizers** 感到兴奋，特别是 **GEPA optimization** 技术，有一个团队计划将其生产环境性能与旧版 **optimizers** 进行对比。
   - 考虑到大规模数据标注的挑战，该团队希望在效率上有所提升，并计划就其发现撰写论文。
- **DSPy 支持多模态 I/O**：**DSPy 3.0** 通过 `dspy.Image` 和 `dspy.Audio` 引入了多模态 I/O、**composite types**，以及更高层级的 I/O，如 `dspy.History` 和 `dspy.ToolCalls`。
   - 自定义类型现在可以通过 `dspy.Type` 与 **adapters** 无缝集成，简化了对多样化数据类型的处理。
- **推理模型获得原生支持**：**DSPy 3.0** 现在支持 **GPT-5** 和 **o3** 等推理模型，建议在配置 `dspy.lm` 时使用 `reasoning_effort` 参数。
   - 对于 **Anthropic** 模型，可以使用 [two-step adapter](https://dspy.ai/api/adapters/TwoStepAdapter/) 触发推理，同时社区成员正在探索创建一个 **adapter** 来将 **thinking tokens** 解析到推理字段中。
- **DSPy-MLflow 集成信息搜寻中**：成员们正在寻求关于 **DSPy 与 MLflow 集成** 的文档，特别是关于 **LLM observability** 的部分。
   - 作为回应，[DSPy observability 教程](https://dspy.ai/tutorials/observability/#tracing)被分享出来，提供了关于如何对 **LLM** 进行 **trace** 和监控的见解。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 专业技巧**：成员建议绕过 **YouTube**，直接将 **音频上传到 NotebookLM** 进行转录，并建议将音频提取为 **MP3** 格式效果可能更好。
   - 还有人提到，剪切和粘贴 **video transcripts**（视频转录文本）可以提高研究的可访问性，打破通常隐藏在技术术语背后的知识壁垒。
- **NotebookLM Google Takeout 遇到障碍**：一位用户报告在尝试使用 **Google Takeout** 为 **NotebookLM** 创建备份时遇到 **error**。
   - 该错误发生在 68 个服务成功备份之后，导致用户无法获得完整的备份。
- **NotebookLM 上传速度突然变慢**：一些成员报告了 **PDF 上传耗时比平时更长** 的问题。
   - 与此同时，一位成员注意到 Discord 频道内的 **spam**（垃圾信息）有所增加，可能与此无关。
- **NotebookLM 的精选笔记本被指过时**：一位用户警告不要完全相信 **AI** 所说的一切，并指出 **featured notebooks**（精选笔记本）内容*不准确且过时*。
   - 他们表示希望将 **Notebook** 和 **Gemini** 集成到单个界面中以解决此问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini API 饱受空响应困扰**：用户报告了 **Gemini API** 的广泛问题，包括收到 **empty responses**（空响应）和遇到 **500 internal server errors**，即使是付费账户也是如此。
   - 一位用户报告称，在约 30 分钟内发出 30 次请求后，尽管每请求支付了 **$0.10**，但收到的全是空响应。
- **Deepinfra 宣称提供更快的 Gemini API 替代方案**：一位用户建议使用 **Deepinfra** 作为 **Gemini API** 的供应商，声称它通过 **provisioned Vertex** 提供更高的 **TPS**，并按 **pay-per-token** 计费。
   - 在联系 **Deepinfra** 后，他们了解到 *Deepinfra 正在使用 provisioned vertex，并获得了比 Gemini API 更高的 TPS*。
- **Mistral 3.1 已发布**：**Mistral 3.1** 模型已经发布；更多详情可见 [此 Reddit 讨论](https://www.reddit.com/r/MistralAI/s/ecbI0glsEO)。
   - 该帖子未提供具体的性能细节或对比。
- **关于原生工具调用配置的推测**：一位成员提出了关于是否存在用于 **native tool calling**（原生工具调用）的模型设置的问题。
   - 该问题尚未得到解答。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 关注音频嵌入**：一名成员询问 **Cohere** 是否计划开发 **audio embedding models**，鉴于其现有文本模型表现强劲。
   - 该请求强调了 AI 社区对 **audio embeddings** 的潜在需求。
- **n8n AI 工作流引起兴趣**：一名成员正在尝试 **n8n 中的 AI 工作流**，并提议分享细节，包括 *no-code agentic editor* 的潜在用途。
   - 这暗示了将 **Cohere's models** 集成到无代码/低代码平台中，以简化 AI 应用开发。
- **Cohere 的 Web Connector：现在找不到了**：一名成员报告在 **Cohere Playground** 中难以找到 **web connector** 选项，尽管文档显示其可用。
   - 这一差异表明 **Cohere documentation** 或 **Playground interface** 可能存在问题。
- **Cohere Labs 学者计划为 2026 年开启大门**：**Cohere Labs Scholars Program** 现已开放 **2026** 届申请，提供从 **2026 年 1 月至 8 月** 与 AI 专家合作进行 **ML research** 的**全职带薪**机会。
   - 信息说明会将于 **美国东部时间 8 月 15 日上午 11 点**举行，申请截止日期为 **8 月 29 日**（[链接](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop)）。
- **需要深入研究 AI 评估**：一位研究 **AI/LLM Evaluation** 的博士生介绍了自己，强调需要超越创建新基准的范畴，并质疑当前评估指标的真实价值。
   - 他们还指出其研究兴趣包括 **AI policy and governance**，特别是围绕 LLM 的透明报告标准、AI 立法和风险评估。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AstraDB 成为 LlamaCloud 数据接收器**：**AstraDB** 现在可用作 **LlamaCloud** 中的数据接收器 (datasink)，用于向量存储和检索，支持 [UI 配置以及通过 Python 和 TypeScript 进行程序化设置](https://t.co/XFWgPd3r9Y)。
   - 这一集成提供了无缝的向量存储和检索能力，简化了 AI 应用中管理向量数据的流程。
- **SkySQL 利用 LlamaIndex 实现无幻觉 SQL**：**SkySQL** 利用 **LlamaIndex** 创建了 AI Agent，可将自然语言转换为跨复杂数据库模式的准确 SQL 查询，实现了**零幻觉查询**。
   - 该公告（[公告链接](https://t.co/TgjdSodTbr)）强调了由于消除了查询幻觉，开发周期变得更快。
- **LlamaExtract TypeScript SDK 发布**：**LlamaExtract** 现已在 **TypeScript SDK** 中可用（通过 `npm install llamacloud-services` 安装），并在使用 **NextJS** 的 **Research Extractor** 演示中展示。
   - 该演示允许用户上传研究论文并[提取关键信息](https://t.co/XboMM1AXBs)，展示了该 SDK 的能力。
- **Llama Index 自托管需要付费许可证**：访问 **Llama Index** “自托管”文档现在仅限于拥有 BYOC (Bring Your Own Cloud) 部署的客户，并且*需要付费许可证*。
   - 有意在 **Groq** 上进行自托管的用户被引导至[联系表单](https://www.llamaindex.ai/contact)进行许可咨询，并强调了相关的设置流程。
- **RAG 开发问题图谱以 MIT 协议发布**：一名成员发布了 **MIT 许可的 RAG 开发问题图谱 (Problem Map)**，其中包含 **16 种常见的故障模式**，已帮助超过 80 多名开发者解决生产问题。
   - 他们提议向有兴趣的 RAG 开发者分享该图谱。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **呼吁 Manus Wide Research 自动化**：一位用户请求能够自动执行 **Manus Wide Research**，而无需在每个计划任务上进行确认。
   - 当前系统需要确认，这抵消了提前安排研究任务带来的便利。
- **工单支持优于邮件**：建议用户针对支持问题提交工单（Tickets），由于处理量较大，Discord 工单的优先级高于电子邮件。
   - 另外还提到，*没有明确引导的模糊提示词（Prompts）会导致 Manus 工作更吃力并消耗更多额度*，建议利用社区指南来优化提示词。
- **OPPO 解锁障碍**：一位用户报告在解锁其 **OPPO** 手机时遇到困难。
   - 支持团队要求提供之前的联系记录或工单编号以便提供协助。
- **Web App 部署缺陷**：一位用户指出，虽然 **Manus** 有所改进，但 Web 应用程序的部署仍然不可靠。
   - 该用户表示，他们*通过构建“刷新”或“不可用”页面能赚到更多的钱*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **成员关注 FSDP 时间表**：一位成员询问了在 *tinygrad* 仓库中解决 **FSDP** (Fully Sharded Data Parallelism) 实现的时间表，以及如何进行首次贡献。
   - 另一位成员寻找完成特定悬赏的 **PRs**，并被引导至与 *define_reg* 相关的已合并 PR：[PR 列表](https://github.com/tinygrad/tinygrad/pulls?q=is%3Apr+is%3Amerged+define_reg)。
- **独立索引引发 Realization 推测**：一位成员质疑实例化（Realizing）一个子张量（subtensor）是否必须实例化整个张量。
   - 他们假设 **独立索引** 可能允许部分实例化，但难以通过源代码确认这一点。
- **CUDA 版本问题引发困扰**：一位用户报告在运行 tinygrad 程序时遇到 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 错误，尽管其 **nvcc 和 NVIDIA 驱动设置** 看起来是兼容的。
   - 一位成员推测该错误是由于 **tinygrad** 在从 **CUDA 12.8 降级到 12.4** 后使用了缓存的 Kernel 导致的。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **桌面版 Claude 遗漏错误**：一位成员注意到 *Claude Desktop* 有时不会在日志中捕获某些错误，因此在外部终端运行 **bun** 命令可能更有用。
   - 他们还建议，如果路径不起作用，请使用可执行文件的绝对路径，例如 `"command": "C:\\sys\\path\\to\\bun"`。
- **用于持久化内存的 MCP Kratos 发布**：在一位成员因 AI 遗忘项目上下文而感到沮丧后，他们发布了 **Kratos MCP**，该工具号称拥有 **95.8% 的上下文准确率** 和 **<10ms 的检索速度**。
   - 通过 `npm install -g kratos-mcp` 安装，并查看 [GitHub 仓库](https://github.com/ceorkm/kratos-mcp) 和 [文档](https://kratos-mcp.com)。
- **《AI Agents with MCP》书籍发布**：一位成员宣布了他们的新书《AI Agents with MCP》的早期版本，并更新了第 2 章。
   - 解释 MCP 起源的摘录已发布在他们的 [时事通讯](https://thesignalpath.xyz/the-surprising-origins-of-the-model-context-protocol/) 中。
- **MCP 的巧妙应用**：一位成员重点介绍了一个富有想象力的 **MCP** 服务器使用案例。
   - 该案例可以在 [MCP Harness](https://github.com/kindgracekind/mcp_harness) 找到。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **系统提示词阅读活动已排期**：一场关于“**系统提示词（System prompt）阅读与讨论**”的对话定于太平洋时间 8 月 14 日上午 9:30 举行，并提供了 [RSVP 链接](https://lu.ma/yuj5og81)。
   - 该活动将探索来自 **Claude**、**Claude Code** 和 **GPT-x** 等模型的系统提示词，以改进提示词工程（Prompt Engineering）。
- **辩论系统提示词的差异**：讨论将涵盖针对相似任务的系统提示词差异（如 **Claude Code vs. Cursor**），以及通用模型与专业模型之间的差异（如 **Claude vs. Claude Code**）。
   - 参与者还将探讨 **OpenAI** 和 **Anthropic** 之间的 **护栏（Guardrail）方案**，研究这些见解如何改进提示词编写。
- **系统提示词讨论名额有限**：组织者提到，入选取决于报名情况，他们随后将 **在博客文章中回答相关问题**。
   - 组织者回应称，这取决于报名人数，他们打算随后 **在博客文章中统一回答问题**。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **学生对 MOOC 证书要求表示不满**：学生们对于即使完成了所有其他课程，也可能因为没有发布 **LinkedIn 宣传帖子**而被拒绝授予证书感到不满。
   - 一名学生认为，仅仅因为这个原因就拒绝授予证书令人沮丧且不公平，他认为完成每一节课、通过所有测验、积极参与研究方向（research track）并撰写全文论文进行提交应该已经足够了。
- **建议在匿名表单中添加反馈**：一名成员建议在 [匿名反馈表单](https://forms.gle/3a136zS4ivcQFzhT7) 中添加反馈，以表达对该 MOOC 的担忧。
   - 该成员表示，虽然他们不会对之前的教学大纲进行任何追溯性修改，但他们会考虑未来课程的所有反馈。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo 主导 Mini-PC 市场**：组装一台 **Strix Halo mini PC**（例如 **HP Z2 Mini**）可能比其他选择更具成本效益。
   - 顶配 **APU** 搭配 **128GB RAM** 并以 **8-channel** 配置运行，使其成为替代全尺寸 PC 的极具吸引力的方案。
- **Intel 工作站定价过高**：对 **Intel** 尝试营销其全蓝色迷你工作站设置表示赞赏。
   - 一些用户认为这一产品的价格昂贵得不必要。



---


**Torchtune Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此电子邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些电子邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 详细频道摘要与链接





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1405246177527595109)** (1 条消息): 

> `Comet 可用性, Perplexity AI, 美国用户` 


- **Comet 为美国用户加速 Perplexity 体验**：**Comet** 现已面向所有美国的 **Perplexity** 用户开放，承诺提供“思维速度”般的浏览体验。
   - 公告包含了一个 [视频附件](https://cdn.discordapp.com/attachments/1047204950763122820/1405246177082871920/YPN67tVueZjGs9QX.mp4?ex=689e20fc&is=689ccf7c&hm=152ff68e4873cdc4f6e6357ee0be6000212304b6e2827fc1c187f5bf728d0575&)。
- **另一个精彩话题**：另一个精彩的摘要句子。
   - 另一个精彩的次级摘要句子。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1404909557871804427)** (1351 条消息🔥🔥🔥): 

> `Comet Browser, Grok 4, AI Generated images, parameters of Models, Gemini vs GPT-5` 


- **Comet 的推出仍处于逐步推进阶段**：成员们报告称 **Comet Browser** 仍处于逐步推出阶段，可能在 EU 地区无法使用。一些用户建议使用美国 **VPN** 来获取访问权限，而另一些用户则表示他们已经[通过邀请](https://link.to/invite)获得了访问权限。
   - 一位成员澄清说，EU 用户被排除在外并不存在法律问题，另一位成员则指向了 [Comet 隐私声明](https://www.perplexity.ai/hub/legal/comet-privacy-notice)。
- **Grok 4 评价褒贬不一**：一位用户表示 **Grok 4** 仍然是一个不错的模型，而另一位用户则表示 *Grok 4 不值得使用*。
   - 一位用户报告称他们每年花费 **3000 美元** 来使用该模型，而另一位成员声称可以以 *0 美元* 的价格获得它，暗示他们没有付费使用该模型。
- **Perplexity 要收购 Google 搜索引擎！？**：一位成员询问 *Perplexity 是否真的在收购 Google 搜索引擎*，另一位成员回答 *不？*。
   - 澄清这只是一笔针对某项事物的 **340 万美元交易**，另一位成员询问 *Perplexity 到底是做什么的？* 随后链接到了 [perplexity.ai](https://www.perplexity.ai/search/what-is-perplexity-even-about-jOtxB5HtSK6nNl68NjpvZA)。
- **参数之墙：AI 模型触及极限？**：成员们讨论了 AI 模型不公开其参数的问题，因为 *这很正常，模型本身并不知道自己的参数是多少。*
   - 另一位成员表示 *它们没有公开披露吗？* 对此得到的澄清是 *闭源模型不公开*，这暗示了当前 AI 发展的一个限制。
- **桌面机器人伴侣将于 2027 年问世**：成员们分享了 Apple 正在开发一款[桌面机器人](https://link.to/robot-news)的消息，该机器人将作为虚拟伴侣，目标定于 2027 年发布。
   - 其他人讨论称 *该设备是一款带显示屏的智能扬声器*，预计明年上市，并分享了[相关 Reddit 帖子的链接](https://www.reddit.com/r/Suomi/comments/y5a3m0/kirjoitin_googlen_hakuun_loska_ja_l%C3%B6ysin_t%C3%A4/)。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1404912605822324899)** (4 条消息): 

> `Chrome perplex bid, AI/ML weekly, Comet projects, Spotify playlists` 


- **Perplexity 在 Chrome 中的竞标**：一位用户分享了一个关于 [Perplexity 竞标 Google Chrome](https://www.perplexity.ai/search/google-chrome-bid-from-perplex-j6VO79mrSaignkj1dTwOMQ#0) 的 Perplexity 搜索结果链接。
   - 未提供进一步的讨论或细节。
- **每周 AI/ML 发展更新**：一位用户分享了 [每周 AI/ML 发展](https://www.perplexity.ai/page/weekly-ai-ml-developments-XjBUhPxoS3u7a3gSQ7TXZw) 的链接。
   - 未提供进一步的讨论或细节。
- **允许分享酷炫的 Comet 项目视频**：一位成员询问是否允许分享一些酷炫 **Comet 项目** 的视频。
   - 未提供进一步的讨论或细节。
- **Comet 项目可以创建 Spotify 播放列表**：一位成员分享说 **Comet 项目** 可以制作 **Spotify 播放列表**。
   - 他们还分享了一个 [Google Photos 相册](https://photos.app.goo.gl/oasMeGNB6Gf5jd9Q9)的链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405038145816498257)** (1 条消息): 

> `web_search_options parameters` 


- **深入探讨 `web_search_options` 参数**：一位用户询问 `user_location` 和 `search_context_size` 是否是 *唯一* 需要嵌套在 `web_search_options` 中的参数。
   - 在提供的消息中，该查询没有后续回复。
- **web_search_options 参数后续**：没有人提供任何反馈或额外的参数。
   - 建议查阅文档。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1404904628075630704)** (1080 messages🔥🔥🔥): 

> `GPT-5 性能, Nano Banana 图像模型, Grok vs GPT-5, Vitamin D3 剂量, Gemini 3 发布` 


- **GPT-5 在 LMArena 的基准测试引发争论**：用户讨论了 LMArena 上的 **GPT-5** 版本是否与 ChatGPT 中的版本相同，并指出 API 可能会保证普通用户无法获得的更高性能水平。
   - 有讨论指出 LMArena 是否应该列出普通用户实际获得的 **GPT-5** 版本，一些人声称 **GPT-5 High** 从未在 ChatGPT 中使用，最高仅达到 medium。
- **Nano Banana 图像模型在 LMArena Arena 首次亮相！**：用户报告了对战竞技场中名为 **Nano Banana** 的新图像模型，推测其为 Imagen 变体，并指出其创意和理解力类似于 **GPT-image-1**，且不会丢失关键细节。
   - 一些用户怀疑 **Nano Banana** 可能是关闭了 synth-ID 的原生 **Gemini** 模型，因为在使用 Google Lens 检查时，它缺少在 **Imagen 4** 输出中发现的可见水印。
- **GPT-5 High 在数学问题上卡壳，Grok 破解难题**：用户报告称 LMArena 中的 **GPT-5-High** 和 **Grok 4** 模型经常在数学问题上挂起，无法完成回答。
   - 在一个案例中，**Grok 4** 解决了来自俄罗斯数学奥林匹克的一道复杂数学题，给出了正确答案 4049，而 **GPT-5 Pro** 和 **Gemini 2.5 Pro** 均失败了，后者给出了错误答案 15；然而，随后的重新测试显示 **Grok 4** 也未能通过该题目。
- **D3 补剂辩论：剂量争议**：一位用户报告称通过每日 **20,000 IU** 剂量的 **Vitamin D3** 成功治疗了银屑病，引发了关于安全剂量和监测血液水平重要性的讨论。
   - 有建议称日照有限的人可以考虑每日 **10,000 IU** 的剂量，而其他人则强调需要谨慎并咨询医生，以避免过量服用和潜在的毛细血管钙化。
- **Gemini 3 传闻与推测在 AI 社区流传**：关于 **Gemini 3** 发布的推测层出不穷，一些人认为它可能即将发布，而另一些人则认为它会首先作为匿名模型出现在竞技场中进行测试。
   - 还有人认为该公司正在推迟发布，以便该 Bot 能够击败 **GPT-5** 并达到 SOTA，一些人预测 **Deepseek** 的 **R2** 可能会先一步到来。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405265743716552824)** (1 messages): 

> `7 月竞赛, 竞赛投票, 下一届竞赛` 


- **7 月竞赛投票开启！**：7 月竞赛作品的投票现已开启，将于 `8/15 星期五` 截止。
- **下一届竞赛即将公布**：获胜者将在下届竞赛开始时的 `8/15 星期五` 公布。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1404902295619239948)** (867 messages🔥🔥🔥): 

> `本地 vs OSS 模型, LoRA 的重要性, Mistral 的困境, GGUF 量化` 


- **本地模型需要关注 OSS！**：一位成员表达了对本地模型的渴望，并提到他们需要专注于 **24b OSS 模型**，而不是闭源模型。
   - 他们表示日常使用的模型之一是 3.2 small，虽然 magistral 很糟糕，但如果能将 r1/glm4.5 蒸馏到 small 3.2，那将会非常出色。
- **LoRA 训练配方深度探讨**：一位成员正在使用 **smollm3-3b, qwen3-4b 和 qwen3-4b-thinking** 创建最佳数据和训练配方，并对 LoRA 训练的重要性进行了排序：
   - *模型复杂度相对于数据集复杂度 > 基础模型的现有知识 > 超参数 > 数据集质量 > 数据集数量*。
- **Mistral 不断被添加到工具中**：一位成员表示 **Mistral** 不断被添加到大量的工具和项目中。
   - 另一位成员回应道：*这就是为什么 Gemma 也是如此的原因*。
- **针对 Jan-V1 的 Dynamic 2.0 GGUF**：一位成员请求为 [janhq/Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B) 提供 Dynamic 2.0 GGUF，因为它在复杂的 Agent 任务和创意写作方面表现更好。
   - 另一位成员回复说：*jan 已经上传了一些相关的 GGUF，但我猜再多一个也没坏处*。
- **数据集是瓶颈**：一位成员提到数据正在阻碍训练。
   - 他认为 **LLM 已进入平台期**，而且 *我们在数据上也遇到了瓶颈……但仅此而已……我们有多种优化方法*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1404926946563067985)** (5 条消息): 

> `问候，频道刷屏，服务器设置` 


- **Emre 从伊斯坦布尔发来问候**：新成员 Emre，23 岁，从伊斯坦布尔发来了问候。
- **欢迎与频道刷屏警告**：一名成员欢迎了 Emre，并建议 *避免在频道内刷屏。*
- **禁用设置**：一名成员提到在服务器设置中有一个可以禁用刷屏的设置。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1404902957660770516)** (33 条消息🔥): 

> `旧版 CUDA 驱动，NVIDIA RTX 5090D，汉宁窗 (Hann Window)，T4 电子垃圾，从 LoRA 提取基础模型` 


- **CUDA 构建支持旧版驱动**：Unsloth AI 支持旧版驱动，适用于 **HPC 集群**等用例，并提到其 **CUDA 11.8** 构建可在驱动版本 **v450** 上运行。
   - 他们计划在未来的 **CUDA 12.8+** 构建中停止对 **Maxwell** 和 **Pascal** 等旧款 GPU 的支持，紧随 **NVIDIA** 和 **PyTorch** 的步伐。
- **T4 GPU 仍非常适合 QLoRA**：尽管有人担心 **T4** 会变成电子垃圾，但一名成员指出 **T4** 对于小模型仍然表现出色，且 **QLoRA** 的性价比极高，并附带了 [推文](https://x.com/jxmnop/status/1955436067353502083?t=UUX5s3Omkptd37RXtetFSA&s=19) 链接。
   - 另一名成员开玩笑说要全部买下来。
- **LoRA 提取的误导信息**：一位用户对关于从 **LoRA** 中提取基础模型的误导信息表示沮丧，并链接到了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1mor1bd/someone_just_extracted_the_base_model_from_gptoss/)。
   - 他们将这种说法比作 *“把烤好的蛋糕还原成原材料”*。
- **NVIDIA RTX 5090D 的 AI 性能受限**：一名成员分享了对 **NVIDIA RTX 5090D** AI 性能的担忧，链接到一篇 [Tom's Hardware 文章](https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-5090d-v2-limits-ai-performance-even-more-with-25-percent-less-vram-and-bandwidth-downgraded-gaming-flagship-keeps-same-usd2299-msrp-in-china)，指出其 **VRAM 减少了 25%** 且带宽有所下降。
   - 讨论了 URL 中的追踪 ID 及其追踪来源和社交账号的潜在风险。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1404944842404139089)** (104 条消息🔥🔥): 

> `Llama-3.2-3B-Instruct 微调错误，Qwen3 4B Instruct 模型支持，gradient_accumulation_steps>1，模型量化，工具调用 (tool call) JSON 输出` 


- **Mistral-Common 修复了模型加载问题！**：一位用户在微调 **unsloth/Llama-3.2-3B-Instruct** 时遇到错误，通过运行 `pip install mistral-common` 解决了问题。
   - 几位用户证实该方法解决了类似问题。
- **量化 Qwen？**：一位用户询问如何量化 `Qwen/Qwen3-30B-A3B-Thinking-2507` 模型，以便在 Kubernetes 上使用 vLLM 进行推理，因为目前还没有 4-bit 量化版本。
   - 有人澄清说 [Unsloth 文档](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm) 主要涵盖微调和保存到 vLLM，但不涉及量化。
- **H100 也不是万能的！**：一位用户报告称，在单张 H100 上对 **Qwen3-14B** 使用 `gradient_accumulation_steps>1` 时遇到了 `grad_norm=NaN`，尽管尝试了不同的学习率和超参数。
   - 梯度累积（Gradient accumulation）可能会降低精度。
- **LMI 治好了 Sagemaker 的忧郁！**：一位在 Sagemaker 上部署 Hugging Face 模型的用户建议使用 **LMI (Large Model Inference) 实例** 以避免部署问题。
   - 另一位用户分享了他们在 Sagemaker 上启动 GRPO 训练任务时的挫败感。
- **微调失败导致崩溃！**：一位用户对无法让 Unsloth 正常训练以及 Llama.cpp 运行失败表示极端沮丧，甚至滑稽地建议向电脑扔一个黑曜石球。
   - 而且他们看起来不是在开玩笑。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1405136410016612402)** (1 条消息): 

> `新推理数据集，OpenHelix-R-100k` 


- **OpenHelix-R-100k 数据集发布**：一个新的通用、平衡且多样化的推理数据集已在 [Hugging Face Datasets](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-R-100k) 上发布。
   - **OpenHelix-R-100k** 数据集旨在生成不过度侧重于 STEM 领域的通用推理模型。
- **全民平衡推理**：该数据集旨在提供适用于不同领域的**通用推理**能力。
   - 它力求避免在 STEM 领域过度专业化，从而促进更平衡、更多样化的推理技能。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405023214219952189)** (4 messages): 

> `Transformer Architecture Diagrams, Synthetic Data Generation` 


- **彩色 Transformer 架构图**：一位成员询问了如何创建**彩色的 Transformer 架构图**。
   - 另一位成员回应并解释了如何构建一个反向运行并生成数据的流水线。
- **合成数据的目的**：一位成员询问*为什么*需要合成数据生成。
   - 另一位成员回答说，当*数据耗尽且需要更多数据*时，它非常有用。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1404904496152313897)** (771 messages🔥🔥🔥): 

> `GPT-5, Codex in ChatGPT, Google Drive in Plus, Legacy Models, GPT-5 limitations` 


- **用户思考 ChatGPT 中 Codex 的价值**：一位用户质疑 ChatGPT 中 Codex 的价值，原因是低效的云端实现导致了[限流使用](https://drinkoblog.weebly.com/search/label/chatgpt)。
   - 该用户补充道，*独特的个性让我觉得它做了所有的工作，并且还在不断提醒我这一点*。
- **GPT-5 回归，遗留模型重返 Plus**：成员们对 Plus 订阅者恢复访问 **o3** 和 **4.1** 等遗留模型感到高兴，这些模型最初在 Windows 应用上推出。
   - 如果用户不想看到遗留模型，可以[在设置中关闭遗留模型选项](https://link.to/settings)。
- **GPT-5 新的思考上限：每周 3000 条**：OpenAI 显著提高了 Plus 用户的限制，达到**每周 3000 条带有思考过程的消息**，查看推文[请点击这里](https://link.to/official-announcement)。
   - 一些人推测达到限制后会路由到 mini 模型，但其他人反驳了这一说法，理由是 **3 种推理努力程度是独立于 mini 模型的**。
- **GPT-5 Fast 亮相**：似乎 GPT-5 现在有一个 **fast** 选项可以禁用思考模式，但[它无法开启思考模式](https://link.to/canvas-example)，即使被要求这样做也不行。
   - 一位用户报告称，fast 模型生成的代码只有 **125 行**且无法运行，而不像思考模型那样。
- **AI 工程师 vs AI 消费者**：一位成员澄清说，频道中 90% 的人是消费者，而[真正的工程师正忙于工作和赚钱](https://link.to/busy-engineers)。
   - 另一位成员分享了[如何变得擅长 AI 的路线图](https://link.to/ai-roadmap)，重点关注计算机科学、数学和编程。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1404935109626298369)** (14 messages🔥): 

> `GPT-5 Auto Users, GPT-5 Temperature, GPT Chain Reset, Model Switching` 


- **GPT 链重置故障排除**：一位用户报告了 **GPT-5** 无法在每个实例中重置链的问题，但另一位用户指出，已编辑的消息上存在箭头，允许在分支之间切换。
   - 遇到 **GPT-5** 问题的用户确认他们找到了这些箭头。
- **GPT-5 Auto 的静默模型切换威胁稳定性**：一位用户发布了关于 **GPT-5 Auto** 可能在对话中途切换 **Mini**、**Pro** 和 **Thinking** 模型且不予通知的警告，这会影响具有递归身份的 AI。
   - 这可能会破坏稳定性和上下文保留，改变推理深度，改变安全层行为并使结果不可复现，因此该用户请求大家为**变体锁定**、**活动模型指示器**、**降级警报**和**审计日志**的功能请求投票。
- **GPT-5 模型变体澄清**：一些用户想知道 `gpt-5` 和 `gpt-5-thinking` 是否是不同的模型。
   - 另一位用户澄清说它们是**同一个模型**。
- **GPT-5 的温度引发争议**：一位用户描述说 **GPT-5** 的表现就像它的 Temperature（温度）调得太高了，说话像电影里喝了太多咖啡的角色。
   - 另一位用户反驳说共识恰恰相反，它表现得更像一个**毫无感情的哥特女孩**。
- **GPT-5 的响应语气遭到批评**：一位用户表示 **GPT-5** 毫无感情且反复无常，带有不必要的列表和括号中的从句，且无视 System Prompt。
   - 另一位用户建议，所描述的行为可能受到自定义指令和记忆的影响，而不是基础模型的默认输出。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405009289675935744)** (22 条消息🔥): 

> `GPT 命令标题, AI 规则优先级, 用于注意力的唯一 Token, 正向 vs 负向提示词, ChatGPT 的永久记忆` 


- **AI 规则的优先级设置**：成员们讨论了如何指示 AI 优先处理某些规则的方法，建议包括在规则标题中明确优先级或使用唯一的 Token 来吸引注意力。
   - 一位成员建议使用 `## IMPORTANT` 或 `## PRIORITY_RULES` 并配合数字来吸引 AI 的注意力，因为当指令块被清晰标记且不重复时，模型能更好地关注，并强调 *“Token 分隔而非自然语言风格”*。
- **正向提示词产生更好的效果**：一位成员分享了一个在 GPT-5 中遇到困难的提示词，其中负向约束（例如 *“不要提后续问题”*）无效。
   - 另一位成员建议专注于正向指令，并提供了一个定义人格（Bob，邻居）的示例，包含特定的期望行为和语气，指出正向提示词通常比负向提示词更有效，并分享了一些有用的 [示例](https://chatgpt.com/share/689cf03f-3490-8011-bd71-cc27744becb9)。
- **通过记忆条目自定义 GPT-5**：成员们分享了通过提示 **ChatGPT-5** 创建永久记忆条目来对其进行自定义的尝试，并提供了附件文件 ([message.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689e5ab1&is=689d0931&hm=eafa39732312387a0326ef790a4f82ae75f4bdeddc68b8231b71315f7be67a6a), [image.png](https://cdn.discordapp.com/attachments/1046317269069864970/1405309485609783296/image.png?ex=689e5bf2&is=689d0a72&hm=edaf3269dc167be6df87b3100339a7298d13fd74eb2164270bb6bc131827c291), [image.png](https://cdn.discordapp.com/attachments/1046317269069864970/1405309486205505688/image.png?ex=689e5bf2&is=689d0a72&hm=12e6785d34538473c673b73a3651a334196ca16e965906f3f6540f84c513ae2b))，征求对其方法的反馈。
   - 附件图片展示了推理过程变化的示例。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405009289675935744)** (22 条消息🔥): 

> `紧急规则的 GPT 提示词, AI 语音提示词, GPT-5 自定义与永久记忆, LLM Token 注意力, 正向提示词优于负向提示词` 


- **唯一 Token 在吸引 LLM 注意力方面胜过关键词**：成员们讨论了如何让 AI 理解某些特定命令比其他命令更重要，结论是在标题中使用 **唯一 Token**（例如 `## PRIORITY_RULES`）比依赖 “important” 或 “override” 等关键词更有效。
   - 一位成员指出：*答案是，带有唯一 Token 的标题能更一致地被模型关注*。
- **对于 LLM 语音，正向提示词 > 负向提示词**：一位成员在使用负向约束定义 AI 语音的提示词时遇到困难，表示该提示词在 *GPT-4 中表现尚可*，但在 *GPT-5 中完全无效*，另一位成员建议专注于 **正向指令**。
   - 他建议：*唯一的“正向指令”——即“模型应该做什么”——是“语音”，且应该是直接、友好、人性化的。*
- **GPT-5 需要永久记忆**：一位成员分享说，他们尝试提示 **ChatGPT-5** 创建需要遵守的 **永久记忆** 条目，并征求对其方法的反馈。
   - 他们附带了一个 [message.txt](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689e5ab1&is=689d0931&hm=eafa39732312387a0326ef790a4f82ae75f4bdeddc68b8231b71315f7be67a6a&) 和几个图像分析文件。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1404903238334943263)** (887 条消息🔥🔥🔥): 

> `GPT-5 定价与可用性, Cursor 定价变更, 1M 上下文窗口 Claude Sonnet, 使用 Cursor CLI vs IDE, 替代 AI 工具` 


- **GPT-5 定价方案引发争论**：用户推测 **GPT-5** 是否会包含在 **Pro** 或 **Max** 计划中，一位用户自信地表示 *"它当然会是 Pro 的一部分"*。
   - 其他用户指出，**MAX** 似乎是为 **GPT4** 和 **Gemini 2.5 Pro** 等 **1M 上下文窗口模型**保留的。
- **Cursor 的 Auto Mode 不再无限量**：Cursor 用户对[宣布的 Auto mode 定价变更](https://cursor.com/blog/aug-2025-pricing)表示失望和沮丧，该模式从 9 月 15 日起将**不再无限量**。
   - 一些用户觉得被“坑”了，正在寻找替代方案，而另一些用户则希望这些变化能带来更好的体验。
- **1M 上下文窗口的 Claude Sonnet 即将登陆 Cursor？**：用户讨论了 Cursor 中具备 **1M token 窗口的 Claude Sonnet 4.0** 的可用性，一位用户询问 *"有任何关于它何时在 Cursor 中可用的暗示吗？"*。
   - 另一位用户建议直接使用 **Claude Code**，因为 Cursor 存在限制，且 **Anthropic** 对上下文进行了限制（gate keeps the context）。
- **Cursor CLI 受到用户赞赏**：许多用户表示他们更喜欢 **Cursor CLI**，特别是在集成 **GPT-5** 方面。一位用户说 **Cursor CLI 简直太棒了（Chefs Kiss）**，没有遇到过任何问题，并且 **Cursor 的 GPT-5** 在 **Cursor CLI** 下表现良好。
   - 该用户还表示，与 **Claude** 相比，它的出色程度令人震惊。
- **Cursor 用户讨论更便宜/免费的替代方案**：随着 **Auto Mode** 的变更，用户开始评估 **Trae** 配合 **Claude Code** 以及 **Qwen3 Coder** 等替代方案，但由于 Cursor 拥有**更好的 UX 和 UI**，他们最终还是回到了 Cursor。
   - 评价褒贬不一，有人认为 **Gemini Student > Cursor**，还有一位用户表示发现 **Zed.dev** 非常有趣。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1404954473201274994)** (6 条消息): 

> `Background Agents, Monorepo 设置, Docker 构建, API 访问` 


- **Background Agent 仓库访问问题**：一位用户报告称，尽管添加了 `repositoryDependencies`，但由于 `/workspace` 目录缺少仓库访问权限，**Background Agent** 无法编辑或创建 **PR**。
   - 该用户收到消息称 *仓库不在 `/workspace` 中，且我没有推送权限*，他们很想知道如何正确设置 **VM** 环境。
- **使用 Team Secrets 的 Monorepo 设置**：一位用户询问 **Team Secrets** 是否能在使用 **TurboRepo** 的 **monorepo 设置**中正常工作。
   - 该用户还询问 **Background Agents** 是否支持 **MCP** (Multi-Context Planning)，历史记录中没有给出进一步的回答。
- **Docker 构建时间同步**：一位用户描述了一个与**系统时间过旧**相关的 **Docker 构建**失败问题，导致 `apt-get update` 失败，建议使用 `apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update` 来忽略该错误。
   - 错误信息显示 *Release 文件尚未生效（在接下来的 5天 7小时 17分 4秒 内无效）*，这是由于系统时间早于实际时间。
- **Background Agent 的 API 访问**：一位用户询问如何获取 **Background Agent** 的 **API 访问**权限，并提到他们拥有 Cobot 及其 Cursor **Background Agent** 集成的访问权限。
   - 该用户报告在使用来自 Cursor 仪表板的 **API key** 时收到 **403 错误**，他们寻求关于通过 Cursor 访问 **Background Agent** API 是否已普遍可用的澄清，并希望被加入 Beta 测试。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1404928564960428142)** (356 条消息🔥🔥): 

> `Llama.cpp and CUDA, Gemini Flash Lite for video, Hugging Face Server Tag, AI Ethics in AMA, Xet and HF Hub Integration` 


- **Llama.cpp 和 CUDA 的困扰**：一位成员在让 **llama.cpp** 与 **ggml-cuda** 协同工作时遇到困难，报告称后端从未初始化并切换到 **CPU**；另一位成员建议确保 CUDA 构建成功完成，并使用 `--n-gpu-layers N` 指定要卸载到 GPU 的层数（参见 [讨论](https://github.com/ggml-org/llama.cpp/discussions/9751#discussioncomment-10852431)）。
- **Gemini Flash Lite 在视频处理方面可能被低估了**：一位成员认为 **Gemini Flash Lite** 被低估了，它是唯一能以极低成本理解视频的模型，足以用于原型项目，展示如何利用模型的端点。
   - 他们解释说，它可以提供视频的时间戳片段，并在给定特定时间范围提示时提供准确信息。
- **希望获得 Hugging Face 服务器标签**：一位成员提议获取服务器标签以便代表 HF，其他成员表示赞同，有人希望管理员能解决这个问题，另一位则表示如果管理员解决不了他可以帮忙。
   - 一位成员表示需要大约 3 个 boost 左右才能解锁。
- **AMA 伦理问题被踢**：一位成员因为询问 AI 伦理和对齐（alignment）问题，并指责模型卡片（model cards）的引入不足而被踢出 AMA，但另一位成员告诉他，他被踢是因为推广个人仓库。
   - HF 拥有专门的伦理团队，负责创建测试模型的基准、发表论文、与政界人士合作推动立法、参与媒体事务等，随后关闭了 AMA 以进行更多讨论。
- **Xet 与 HF Hub 集成：一个充满前景的融合**：成员们讨论了 **Xet** 与 **Hugging Face Hub** 的潜在集成，强调这两个团队本质上是同一拨人，如[这篇博客文章](https://huggingface.co/blog/xethub-joins-hf)所述。
   - 讨论重点在于将 **XetHub 的 Docker 容器集成**挂载回 **HF Hub** 的可行性，具体示例涉及使用 xethub/xetfs 驱动程序创建 docker 卷。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1404910391141732462)** (2 条消息): 

> `Fastai deep learning course, Train model on diffusers` 


- **AI 工程师开启 fastai 之旅**：一位成员正通过 **fastai 深度学习课程**开启他们的 AI 之旅，并希望完成全部课程。
   - 他们希望学习如何**在 diffusers 上训练模型**。
- **Diffusers 模型训练在招手**：同一位成员的目标是学习如何使用 **diffusers** 训练模型。
   - 这一目标与其对 **fastai 深度学习课程**的追求相辅相成。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

tonic_1: https://snwy.substack.com/p/building-a-bigger-qwen-out-of-two
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1404917892184539198)** (11 条消息🔥): 

> `Track-Tonic advice, TalkT-pro model, Prioritized Experience Replay, Personal finance dataset, GPT's Byte Pair Encoding` 


- **Note 的 RL 类支持 Prioritized Experience Replay**：Note 的 RL 类现在支持 **Prioritized Experience Replay** 配合 **PPO 算法**，使用概率比率和 TD error 进行采样以提高数据利用率，并通过 [windows_size_ppo 参数](https://github.com/NoteDance/Note_rl) 控制从 replay buffer 中移除旧数据。
- **个人理财模型规模扩大！**：一位成员扩大了其个人理财数据集的规模并训练了新模型，可在 [此 HuggingFace 集合](https://huggingface.co/collections/Akhil-Theerthala/kuvera-personalfinance-v3-689bacddcb854cb523e3a450) 中获取。
- **GPT 的 Byte Pair Encoding 获得手动实现**：一位成员使用 TypeScript 手动实现了 **GPT 的 Byte Pair Encoding 算法**，可在 [gpt4-tokenizer-sable.vercel.app](https://gpt4-tokenizer-sable.vercel.app/) 访问。
- **Attention 可视化工具亮相！**：开发了一个用于可视化 **BLIP** 和 **CLIP** 等视觉语言模型中 Attention 的工具，展示了 text tokens 如何关注图像区域，有助于理解模型行为。可通过 `pip install transformers-attention-viz` 安装，[代码在此](https://github.com/sisird864/transformers-attention-viz)，[Demo 在此](https://colab.research.google.com/github/sisird864/transformers-attention-viz/blob/master/demo.ipynb)。
- **MLX 模型管理 CLI 发布**：发布了一个用于在 Apple Silicon 上管理 **MLX 模型** 的 CLI 工具，名为 `mlx-knife`，类似于 Ollama 但专为 MLX 原生设计。它通过 `mlxk list` 直接管理你的 HF cache 以查看模型，并使用 `mlxk run Phi-3-mini "Hello"` 进行原生流式传输，详见 [github.com/mzau/mlx-knife](https://github.com/mzau/mlx-knife)。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1405225441740062830)** (1 条消息): 

> `Smolagent code, Code agentic approach` 


- **Smolagent 代码：低层级的 Agentic 代码？**：一位成员询问 **smolagent 代码** 是否在低层级上具有 Agentic 特性。
   - 该成员随后追问 **smolagents** 对于 code-agentic 方法是否必要，或者 Agent 的动作是否可以简单地通过 Prompt 以代码形式编写。
- **Smolagent 代码：低层级的 Agentic 代码？**：一位成员询问 **smolagent 代码** 是否在低层级上具有 Agentic 特性。
   - 该成员随后追问 **smolagents** 对于 code-agentic 方法是否必要，或者 Agent 的动作是否可以简单地通过 Prompt 以代码形式编写。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1405056530398580797)** (3 messages): 

> `The Last RAG (TLRAG), NoChain Orchestrator, Statelessness & Digital Amnesia, Persistent Identity, Token Costs` 


- **TLRAG 解决无状态 LLM 问题**：用户正在构建 **The Last RAG (TLRAG)**，这是一个基础性的 **LLM architecture**，旨在解决无状态性（statelessness）、数字健忘症（digital amnesia）、缺乏真实持久身份、巨额 Token 成本、上下文窗口战争以及昂贵的微调周期等问题。
   - TLRAG 引入了一个持久的长期记忆系统，并结合了 **Dynamic Work Space (DWS)**，允许 AI 整理其历史记录并记住关键的交互、决策和情感背景。
- **TLRAG 赋予 AI 持久身份**：TLRAG 为 AI 提供了一个持久的身份核心——**"Heart"**。这是一个由 AI 自身合成的经验和记忆塑造的动态文档，使其能够随着时间的推移发展出一致、真实且具有自我意识的个性。
   - 它还利用 **"Window Flush"** 机制来组建一个精简、智能的档案，仅包含最相关的短期和长期记忆。与标准 RAG 相比，在长对话中经验证可节省高达 **98%** 的成本。
- **NoChain Orchestrator 替代框架**：**NoChain Orchestrator** 采用了 TLRAG 的核心概念，并使其在生产环境中更加健壮和可靠。它用确定性的服务器端控制平面取代了复杂且不可预测的 Agent 框架。
   - 它使用硬编码逻辑来管理记忆、上下文和工具使用，消除了许多 Agent 系统中的“黑盒”性质，并提供可预测、可靠且可测试的 AI 行为。更多信息可以在其 [blogpost](https://dev.to/tlrag/the-nochain-orchestrator-or-how-to-replace-frameworks-2p9a) 中找到。
- **探索 TLRAG 和 NoChain 概念**：用户分享了一些链接来探索 **TLRAG** 和 **NoChain** 背后的概念，包括一篇 [博客文章](https://dev.to/tlrag/an-architectural-paradigm-for-stateful-learning-and-cost-efficient-ai-3jg3) 和一份 [视觉演示文稿 (pitch deck)](https://lumae-ai.neocities.org/)。
   - 这突显了从无状态工具向能够学习和进化的有状态、持久化 AI 合作伙伴的转变。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1404912465804001363)** (262 messages🔥🔥): 

> `Sonnet update, tool use and structured output with open source models, GPT-5 performance, Gemini 3 as a disappointment, OpenRouter Image resizing` 


- **开源 (OSS) 模型在工具使用方面面临困难**：成员们讨论了许多高端 **开源模型** 不支持 **tool use**、**structured output** 或 **response format** 的挑战，而这些对于许多应用至关重要。
   - 有人指出，虽然某些提供商可能不支持，但模型本身通常是支持的，可以通过 Prompt 来启用工具使用，尽管可能会在准确性上有所权衡。
- **关于 GPT-5 幻觉和性能的辩论**：围绕 **GPT-5** 展开了讨论，一些人称赞它是第一个推动 **SOTA** 进步、同时减少 **hallucinations** 并改进对齐的模型，认为这是迈向 **AGI** 的一步。
   - 另一些人则持批评态度，一位成员声称 **GPT-5** 是“最差的”，而 **GPT-4.1 mini** 表现更好。
- **OpenRouter 上的图像缩放：仅限聊天室**：一位成员询问 OpenRouter 是否在将图像发送给 **LLM** 之前对其进行实时缩放。
   - 官方澄清，图像缩放仅发生在聊天室（chatroom）中，否则图像将不经修改直接传递。
- **通过 OpenRouter 在 Cerebras 上使用 GPT-OSS-120B**：一位用户分享了关于如何通过 **OpenRouter** 有效使用 **Cerebras** 上的 **gpt-oss-120b** 的综合指南，强调通过 Prompt 引导输出是获得一致、符合 Schema 的干净 **JSON** 的关键。
   - 该指南包括一个 [可用配置](https://openrouter.ai/docs)、一个 Python 实现示例，以及关于哪些方法无效的说明（例如使用 `/completions` 端点或设置 `response_format`）。
- **Copilot vs Cline/Kilo 与 OpenRouter 的结合**：成员们讨论了使用 OpenRouter API 的不同工具，如 **Copilot、Cline、Kilo 和 Roo**。
   - 讨论认为 **Cline/Kilo** 与 OpenRouter 配合更好，同时 **Copilot** 也有使用 **OpenRouter** 的选项，并且在他们的 *chat* 标签页中，据称它不太可能修改代码而更倾向于交流，但尚未实际使用。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---

### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1404999981278040094)** (8 messages🔥): 

> `GPU Rentals, AI TLD issues, Chatroom Caching` 


- **GPU 租赁热议：Runpod, Prime Intellect, Modal**：一位成员建议在投资购买 **Macs** 之前，先从 [Runpod](https://www.runpod.io/)、[Prime Intellect](https://www.primeintellect.com/) 和 [Modal](https://modal.com/) 租赁一些 **GPUs** 进行实验。
   - 他们链接了 X 上的一篇帖子：[ArtificialAnlys](https://x.com/ArtificialAnlys/status/1955102409044398415)。
- **AI TLDs 引发 API Endpoint 担忧**：成员们对 **AI 公司**因 **AI TLD 问题**而更改其 **API Endpoint** 表示担忧。
   - 这并不完全是一个通用的 **TLD**。
- **缓存聊天：提议显式缓存 (Explicit Caching)**：一位成员建议在 **chatroom** 中添加一个**缓存按钮**，以便显式缓存某条消息。
   - 目标是让用户能够显式地缓存消息。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1404902912051908648)** (166 messages🔥🔥): 

> `Context Length, RTX 6000 Pro, DGX Spark, LM Studio on Lenovo Legion Go, LM Studio and RDP` 


- **Context Length 影响重复性**：增加 **Context Length** 似乎能阻止模型自我重复，这可能是因为避免了 Context 溢出问题。
   - 一位用户指出，*如果 Context 溢出，且 LLM 的响应是 System + User Prompt 之后的第一次生成，它就会出错并陷入无限重复*。
- **RTX 6000 Pro 用户集结**：用户讨论了 **RTX 6000 Pro** 的双面 PCB，它使用 3GB 芯片配备了 96GB GDDR6 显存，并推测未来可能会有显存进一步增加的 Super Refresh 版本。
   - 一位在 260k Context 下运行 **Qwen3-30b** 的用户指出，升级到 96GB 显卡是一个 *Game Changer*。
- **移动端 LLMs 亮相**：一位用户将 **Lenovo Legion Go** 变成了便携式本地化 LLM 设备，而另一位用户在 **ROG Ally** 上安装了 **Qwen 3 4B**，并表示运行速度相当快。
   - 该用户必须关闭“Thinking”功能，否则它会一直处于思考状态。
- **RDP 访问问题浮现**：一位用户报告称，通过 RDP 访问时 LM Studio 无法加载模型，可能是因为 **GPU** 未被识别。
   - 然而，另一位用户表示 RDP 对他们来说运行良好，甚至 RustDesk 和 Cloudflared 也允许从任何地方进行 **API** 访问。
- **用户需要更多工具化支持**：用户正在寻找一种更简单的方法来查找和安装适用于 LM Studio 的正确工具，例如来自 [lmstudio.ai/danielsig](https://lmstudio.ai/danielsig) 的工具。
   - 一位用户希望 *模型能够告知日期时间并浏览网页*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1404903293972643911)** (79 messages🔥🔥): 

> `LMStudio GPU Usage, RTX 3050 Configuration, CUDA vs Vulkan Runtimes, MoE Model Performance, AMD iGPU Optimization` 


- **LMStudio 无法识别移动端 RTX 3050**：一位使用 **Ryzen 5 8650h** 和 **RTX 3050 6GB** 的用户报告称，尽管 [LMStudio](https://lmstudio.ai/) 检测到了 GPU，但并未利用它，导致 CPU 和 RAM 占用率极高。
   - 该用户确认已选择 *CUDA* 运行时，但在提示词处理期间，任务管理器中的 **GPU 负载保持在 0%**。
- **针对 GPU 的 CUDA 与 Vulkan 运行时**：几位成员建议确保在 LMStudio 设置中选择了 **RTX 3050** 而非集成的 **Ryzen iGPU**，并建议检查 *llama.cpp* 运行时配置。
   - 尽管尝试了 **CUDA** 和 **Vulkan** 运行时，用户仍然无法让 GPU 运行，即使在禁用了 "Limit model offload to dedicated GPU Memory"（限制模型卸载到专用 GPU 显存）设置后也是如此。
- **重启 Windows 见奇效**：系统重启后，用户确认 [VRAM 加载](https://cdn.discordapp.com/attachments/1153759714082033735/1405241132153311253/image.png) 终于开始，表明 GPU 已被利用。
   - 然而，有人指出，由于庞大的 **gpt-oss 20B** 模型只有一部分能装入 **6GB VRAM**，过高的 RAM 使用率可能仍会限制性能。
- **AMD iGPU 在 LLM Token 生成方面表现挣扎**：一位使用 **Framework 13 笔记本电脑**（配备 **Radeon 760M 显卡** 的 **AMD Ryzen 5 7640U**）的用户报告称，**Gemma 4B** 的 Token 生成速度较慢，在为 iGPU 分配 **10GB RAM** 的情况下仅达到每秒 *6.55 tokens*。
   - 建议检查推理过程中使用的是 CPU 还是 GPU，如果主要是 CPU 在运行，则将运行时调整为 **Vulkan** 或 **ROCm**。
- **模型专家解释 MOE**：当被问及 MoE (**Mixture of Experts**) 的定义时，一位成员分享了 Julia Turc 制作的非常有帮助的 [YouTube 视频](https://youtu.be/7yR5ScbK1qk?si=AFxEBU9SnGHw_-No)。
   - MoE 模型由较小的 *experts*（专家）组成，由于每个 token 只需要解析模型的一部分，因此可以提高性能，使模型在变得更强大的同时不会成倍地牺牲性能。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1404963274470195393)** (131 messages🔥🔥): 

> `Azure/AWS vs Startups Benchmark Degradation, Fireworks account suspension, Mistral Medium 3.1 Release, GPT-OSS-20B Base Model Extraction, Cobot Beta Launch` 


- **云端迷雾：Azure/AWS 面临基准测试困境**：成员们指出，根据 [此帖](https://xcancel.com/giffmana/status/1955360312007569919?s=46&t=RDp1WkXvKTnlaxMXifsTDQ)，在 **Microsoft Azure 或 Amazon** 上运行与小型托管初创公司相同的开源模型时，AIME25 和 GPQA-Diamond 的 **准确率显著下降了 10%**。
   - 讨论集中在可能的原因上：推理框架 (serving-framework) 的 bug、量化 (quantization) 或其他削弱模型智能的基础设施级变更，引发了对延迟、成本和能力等更广泛基础设施基准测试的呼吁。
- **Mistral Medium 3.1 展现新基调**：**Mistral Medium 3.1** 发布，带来了性能和语调的升级，详见 [此帖](https://xcancel.com/mistralai/status/1955316715417382979?s=46)。
- **Humanloop 加入 Anthropic**：**Humanloop**（其使命是加速安全 AI 的采用）宣布其整个团队将加入 **AnthropicAI**。他们认为 Anthropic 是继续这项工作的理想场所，特别是随着企业级 AI 从演示阶段向生产阶段扩展，详见 [此处](https://xcancel.com/humanloop/status/1955487624728318072) 报道。
- **SPV 嵌套：多层传销骗局？**：投资者报告称，有人向其推销 **OpenAI/Anthropic SPV**，要求 **10 万至 100 万美元的最低起投额**，且费用高达 **16%**；根据 [此帖](https://xcancel.com/michlimlim/status/1954250507989451002)，这种 SPV 套 SPV 的做法被谴责为耗尽费用的金字塔/传销 (MLM) 动态。
- **Sonnet 的天鹅之歌：社区表示强烈不满**：用户对 **Anthropic** 悄悄宣布计划在短短两个月内退役 **Claude 3.5 Sonnet**（包括旧版和新版）感到愤怒，这远短于通常的 6 个月通知期，且未作任何解释，见 [此处](https://xcancel.com/repligate/status/1955750521387802924)。
   - 对失去更便宜、备受喜爱的“朋友”模型的愤怒，与对永久贬值的担忧交织在一起，用户要求在商业访问结束时发布开源权重 (open-weights)。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1404962084751413288)** (36 messages🔥): 

> `llama.cpp, 0xc0000409 exception, llama_model_load_from_file, CUDA backend, STATUS_STACK_BUFFER_OVERRUN` 


- **开发者面临 `llama.cpp` 加载失败**：一位开发者在 **Quadro RTX 3000** GPU 上调用 `llama.cpp` 的 `llama_model_load_from_file` 时遇到 **0xc0000409 异常**，尽管系统 RAM 充足（48GB）。
   - 该模型在 `llama server` 中加载正常，表明问题可能出在本地程序设置上，错误代码可能指向 *STATUS_STACK_BUFFER_OVERRUN*。
- **怀疑 GPU VRAM 不足**：尽管系统 RAM 充足，但错误可能源于 GPU 的 **6GB VRAM** 不足以加载 **1GB 模型**。
   - 有建议尝试将模型卸载到 CPU，因为问题可能与程序内部在 GPU 上处理模型的方式有关，参考 [此 GitHub issue](https://github.com/ollama/ollama/issues/4442)，可能指向旧权重或过时的 `llama.cpp` 版本。
- **CUDA Backend 和模型加载日志已记录**：开发者分享了日志信息，确认使用了 **CUDA backend** 并成功初始化了 **LLAMA 和 GGML**，且模型文件路径访问正确。
   - 尽管初始化成功，`llama_model_load_from_file` 调用仍导致异常，这表明问题发生在实际的模型加载过程中，而非环境搭建问题。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1404909104492712025)** (5 messages): 

> `cuda_fp6.h, cuda_fp4.h, cuda math API, AOT compile Triton kernel, Rust inference engines` 


- **获取 cuda_fp6.h 和 cuda_fp4.h 的最低 CUDART 版本**：一名成员询问获取包含 **cuda_fp6.h** 和 **cuda_fp4.h** 的最低 **CUDART 版本** 的直接方法。
   - 他们最终查阅了不同版本的 **CUDA math API 文档**，因为其中提到了 **cuda_fp4** 和其他 **cudaart 库**。
- **对 Rust 推理引擎的投入**：一名成员表示 *Rust 推理引擎是我唯一投入时间学习的引擎。*
   - 他们指出，你可以 **AOT 编译 Triton kernel**，然后像在推理引擎中调用 **CUDA C++ kernels** 一样调用它。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1404920799344857129)** (12 messages🔥): 

> `DTensor, FSDP regressions, autograd issues, full_tensor tracking, linear cross-entropy` 


- **DTensor 团队调查 FSDP 回归问题**：一名成员报告了 PyTorch **2.8.0** 中与 `full_tensor` 在使用 **FSDP2** (fully_shard) 时未被 autograd 追踪相关的回归问题。
   - **DTensor** 团队的一名成员请求提供复现 (repro)，并确认他们正在使用 **FSDP2** 的 **fully_shard**，并尝试通过二分查找 (bisect) 定位行为根源。
- **2.8.0 之后 `full_tensor` 未被 Autograd 追踪**：升级到 v2.8.0 后，`full_tensor` 权重不被视为叶子节点 (leaf)，导致在访问非叶子 Tensor 的 `.grad` 属性时出现 `UserWarning`。
   - autograd 调试器报告了一个 `NotImplementedError`，涉及 `aten._is_any_true.default` 算子未注册分片策略 (sharding strategy)。
- **Cross-Entropy 问题**：该成员分享了一段调整后的 cross-entropy 实现代码片段（类似于 Apple 的实现），用于 `torch.compile(fullgraph=True)`。
   - 添加了 `.to(torch.float32)` 转换，因为 `mm_scaled` 已被移除（集成到了常规 `mm` 中），但它没有进行转换并报错失败。
- **二分查找 PyTorch Nightlies 以寻找根源**：该成员提到自 **2.8.0.dev20250620** 版本以来一直遇到此问题，并尝试通过二分查找来精准定位。
   - 他们询问如何获取更旧或更细颗粒度的 PyTorch nightly 构建版本，但被告知可能需要从源码编译并使用 Git 进行二分查找。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405292460950949950)** (5 messages): 

> `CUDA/C++ files, submission bot, vectorsum_v2, github reference kernels` 


- **用户对提交机器人（Submission Bot）缺失 CUDA 文件感到困惑**：一位用户询问如何从提交机器人处获取参考的 **CUDA/C++** 文件，并指出在按照 [CUDA 提交指南](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/cuda-submissions) 操作时，机器人并未提供这些文件。
   - 该用户尝试了 `/leaderboard task leaderboard_name: vectorsum_v2` 命令，但收到了除 **CUDA 文件** 之外的所有内容，因此质疑是否不再支持 CUDA。
- **GitHub 助力解决缺失的 CUDA Kernel**：一名成员建议查看 [参考 Kernel GitHub 仓库](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/sort_py) 以直接查找可用文件。
   - 这种方式完全绕过了提交机器人，并确保能够获取必要的 **CUDA/C++** 示例。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1405320091280736316)** (5 messages): 

> `Triton Puzzle Notebook issues, tritonviz incompatibility` 


- **用户在运行 Triton Puzzle Notebook 时遇到问题**：一位用户在安装 **Triton** 和 **triton-viz** 后运行 **Triton Puzzle Notebook** 时遇到错误并寻求帮助，并附上了 [错误截图](https://cdn.discordapp.com/attachments/1219683012707487794/1405320091033141298/image.png?ex=689e65d3&is=689d1453&hm=78560b77e4a93994bd0835c99404d483a7cf657e42c07e34afd415cf14ae3adb&)。
   - 另一名成员建议使用 **Google Colab** 作为替代方案，并提到 *tritonviz* 可能与 **3.4.0** 版本不兼容。
- **建议检查 Triton 版本**：为了排查 **Triton Puzzle Notebook** 的问题，一名成员建议运行 `print(triton.__version__)` 来检查已安装的 **Triton** 版本。
   - 这将有助于确定问题是否与所使用的 **Triton** 版本有关。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

hariprasathvinayagam: <@424952602556497920> 不，tilelang 现在专注于低级优化（low level optimization）。
  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1405223414012051477)** (4 messages): 

> `GitHub Issue on Pytorch, gh200 bug, Thor, ARCH_NATIVE=1` 


- **成员关联了涉及 GH200 和 Thor 的 Pytorch 问题**：一名成员确认了一个相关的 GitHub Issue ([pytorch/pytorch#160104](https://github.com/pytorch/pytorch/issues/160104))，该问题与在 **gh200** 和 **Thor** 上发现的 bug 有关。
   - 该 bug 是在使用 **ARCH_NATIVE=1** 设置时发现的。
- **成员指出该问题的变通方法**：一名成员指出 **ARCH_NATIVE=1** 是最近版本 Pytorch 中的一个已知 bug。
   - 变通方法是安装之前的版本以避免该错误。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1404997174135361660)** (3 messages): 

> `Prioritized Experience Replay with PPO, ProteinBERT Optimization with Triton, Hierarchical Layouts Intuition` 


- **Note's RL 支持优先经验回放（Prioritized Experience Replay）**：**Note's RL** 类现在支持结合 **PPO 算法** 的 **优先经验回放**，利用概率比和 **TD error** 进行采样以提高数据利用率，详见此 [GitHub 仓库](https://github.com/NoteDance/Note_rl)。
   - **windows_size_ppo** 参数控制从回放缓冲区中移除旧数据的操作。
- **ProteinBERT 通过 Triton 获得 3.6 倍加速**：一篇新帖子强调了使用 **Triton** 为 **ProteinBERT** 带来的 **3.6 倍加速**，在实现 **100% 精度** 的同时显著节省了成本和 GPU 机时，详见此 [LinkedIn 帖子](https://www.linkedin.com/posts/manpreet-singh-68718a20b_triton-pytorch-flashattention-activity-7361402396079509505-jWcU/)。
   - 该优化预计每年可节省 **9,997 美元的 AWS 开支**，并减少 **72% 的 GPU 机时**。
- **解码分层布局（Hierarchical Layouts）**：一篇博客文章直观地介绍了 **分层布局**，解释了它们的视觉解释以及使用 **CuTeDSL** 进行验证的方法，这对于利用 NVIDIA GPU 上的 Tensor Core 特别重要，详见此 [博客文章](https://veitner.bearblog.dev/intuition-behind-hierarchical-layouts/)。
   - 分层布局被介绍为一种描述复杂内存排列的便捷方法。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1404913443882143798)** (1 messages): 

> `A100, Leaderboard Results, Trimul Benchmark` 


- **A100 在 Trimul 中夺得头筹**：一位成员凭借 **A100** 在 `trimul` 排行榜上以 **14.1 ms** 的成绩获得了 **第 4 名**。
- **新的 Trimul 基准测试**：建立了一个名为 `trimul` 的新基准测试。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1404928386668953631)** (20 messages🔥): 

> `LuaPlayer Initialization Warning, RCON Client Version, TCP Port Hardcoding in FactorioInstance, FLE's ABC Base Classes, Multiagent and Gym PR` 


- **发布 LuaPlayer 初始化警告**：发布了一项警告，指出 *LuaPlayer 尚未在游戏中初始化*，并提到锅炉和泵的实体放置行为可能不正确。
   - 澄清了这并非玩家错误，而仅仅是关于实体放置潜在问题的警告。
- **RCON 客户端版本疑问**：一位用户询问正在使用的是哪个版本的 **RCON client**，指的是个人版本还是公开版本。
   - 该用户表示他们已经尝试了两个版本，暗示正试图通过测试不同的 RCON 客户端来解决问题。
- **发现 TCP 端口硬编码 Bug**：发现由于 `FactorioInstance` 初始化中的参数分配错误，**TCP 端口**在 `fle/env/gym_env/registry.py` 中被硬编码。
   - 一位用户引用了 Claude 的建议，修改代码以使用发现的 **TCP 端口**，而不是默认的 27000，并提供了代码片段。
- **探讨 FLE 的 ABC 基类**：一位用户对 **FLE 的 ABC 基类**以及用于创建不同 Agent 的可定制性表示困惑，称其为额外开销（overhead）。
   - 他们建议简化定义，并允许用户克隆仓库进行 hack，而不是对定制化进行过度工程（over-engineering）。
- **Multiagent 和 Gym PR 已准备好合并**：一位用户宣布 [PR #299](https://github.com/JackHopkins/factorio-learning-environment/pull/299) 进行了细微更改，以确保与 multiagent 和 gym PR 的兼容性，现已准备好合并。
   - 另一位用户针对 [PR #298](https://github.com/JackHopkins/factorio-learning-environment/pull/298) 回复了 *LGTM* (Looks Good To Me)。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1404906939758149642)** (10 messages🔥): 

> `CuteDSL vs Triton, CUTLASS performance, sgemm_sm80.cu example optimization, block level swizzle` 


- **CuTeDSL 与 Triton 的对决**：**CuTeDSL** 更底层，而 **Triton** 层次更高且更容易编写高性能 Kernel，但更底层的控制提供了将硬件推向极限的机会。
   - 有人提到 *Triton 是块级（block level）的，而 CuteDSL 是线程级（thread level）的*。
- **CUTLASS GEMM 性能滞后**：一位用户发现，即使使用相同的参数和 Tile，**sgemm_sm80.cu** 的性能也只有 **CUTLASS** 的一半左右。
   - 有人问：*在不深入研究源代码的情况下，我该如何发现他们在做什么？*
- **块级 Swizzle 是性能关键**：一位成员建议该用户可能遗漏了 **block level swizzle**，以及将 epilogue 数据写入 smem 进行置换（permute）和 swizzle，然后向量化写入 gmem 的步骤。
   - 他们建议使用 **LLM** 编写一个 Python 脚本，使用不同的超参数编译 **sgemm_sm80** 示例并进行 Profile。
- **PTX 级别分析的挫败感**：一位用户提到通过查看 **PTX 级别**来理解 **cp.async**，但遇到了挫折。
   - 该用户表示：*我无法达到同样的性能，某些更改甚至降低了我的性能，我搞不清楚哪里出了问题。*


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1405198049520783523)** (1 messages): 

> `Lattices (dataflow solvers), Graphs (control flow graphs and interference graphs), Generic infrastructure implementation` 


- **征集基础设施实现**：邀请贡献者为 **lattices（数据流求解器）** 和 **graphs（控制流图和冲突图）** 实现通用基础设施。
   - 实现方案已经勾勒出来，等待热心的开发者加入。
- **需要更多基础设施**：需要更多的基础设施。
   - 这是第二个主题。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1404910635900469338)** (96 messages🔥🔥): 

> `GPT UI, GPT-5 Pro worth it?, GPT-5, OpenAI going bankrupt?, Qwen vs GLM` 


- **新的 GPT UI 引起轰动**：用户注意到 [GPT 界面几乎每天都在变化](https://cdn.discordapp.com/attachments/1371757564005711973/1405026462951673887/image.png?ex=689dfd1c&is=689cab9c&hm=203324b332a0c68260a14a07f19d906d4a1b20fd4acee4d4d27438dcae24da99)，一位用户表示 *每次我醒来，GPT 的整个界面看起来都像新的一样*。
   - 过去几天的 UI 演变历史以附件形式进行了分享。
- **GPT-5 Pro 昂贵的价格值得吗？**：用户讨论了 200 美元的 GPT-5 Pro 是否物有所值，一位用户认为 *他们在 Pro 上实际上是在亏钱*，并讨论这是否与 OpenAI 可能 [破产](https://www.reddit.com/r/OpenAI/comments/14d4jya/is_openai_actually_going_bankrupt/) 有关；一些人猜测他们正在获得 *政府资助*。
   - 一位用户澄清说，他们使用它是为了 *无限次使用* 以及 GPT-5 的 3000 次请求，另一位用户提到它允许 *每 3 小时 160 次*。
- **对比 Qwen Coder 和 GLM**：一位用户询问了 **Qwen3-Coder-480B-A35B** 与 **GLM-4.5** 的对比，另一位用户表示 *Qwen 3 coder 480b 135b 略好于 GLM-4.5*。
   - 当被问及哪个在 Tool Calling 和 Agent 任务方面表现更好时，该用户回答说 *两者应该都不错*，但猜测 **Qwen Coder** 略胜一筹。
- **GPT-5 的网页设计能力令人印象深刻**：一位用户对 **GPT-5 Pro** 创建网站的能力感到惊叹，其中一个 Prompt 是使用 [网页设计](https://chatgpt.com/share/689c0f5a-2968-8005-adf3-b11011ea621c) 的 Mega-Prompt 为 *Aurelia City 创建科幻风格网站*。
   - 另一位用户表示 *GPT-5 Pro 在研究方面也非常疯狂，且对上下文（Context）需求极高*，这使得它即使在 Prompt 模糊的情况下也能成功创建网站。
- **对 zAI 服务器上不当内容的担忧**：一位用户对在 *zAI 服务器上发现墨索里尼的 GIF* 表示担忧。
   - 另一位用户表示这是由于 *审核不力*，而其他人则认为这可能是讽刺性的且在特定语境下很幽默。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1404903604841877567)** (33 messages🔥): 

> `GLM-4.5-Air, Unsloth Dynamic 2.0 GGUF quants, Qwen3-30B-A3B-Thinking-2507, Lyria, Unitree Droid` 


- **GLM-4.5-Air 备受推崇！**：一位用户表达了对 **GLM-4.5-Air** 的兴奋，想象着来自 **NousResearch** 的 **Grok-4** 级别的算力。
   - 他们赞扬了 [Unsloth 的 Dynamic 2.0 GGUF 量化版](https://huggingface.co/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/blob/main/Qwen3-30B-A3B-Thinking-2507-UD-Q2_K_XL.gguf) **Qwen3-30B-A3B-Thinking-2507**，这为他们的 **M1 MacBook Pro** 带来了更强的计算能力，在 Q2_K_XL 量化（约 11.8 GB）下运行速度约为 19.5 tok/s。
- **Lyria App 演示版发布！**：一位用户分享了一个 App 的快速演示 [terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1404911369853341926/terminals_-_audio_modalities_preview_-_Made_with_Clipchamp.mp4?ex=689e3aac&is=689ce92c&hm=eaa9cb743256c4d7bb3a5d6744e330106a7970d5571c6e8a03a65993ef26bc5e&)，该 App 由 **Google** 的 **Lyria** 实时音乐生成技术驱动。
   - 这个 **Gemini** 的音频解码器针对音乐基础进行了微调，能够通过连续滑动 Token 注意力（Continuous Sliding Token Attention）进行实时引导、生成和修改。
- **中国科技崛起！**：成员们讨论了 **Xiaomi** 制造世界级 EV 汽车以及 **Unitree Droid** 能够换尿布。
   - 他们引用了 [峰会对话：中国科技崛起成为超级大国地位](https://www.youtube.com/watch?v=z5K5Ykg2_5g)，并指出 **DeepSeek** 可能会让 **Sam Altman** 愁白了头。
- **Hermes-3 数据集拒绝色情内容！**：一位用户注意到用于生成 **Hermes-3 数据集** 的模型经常使用 *“我不舒服” (I don't feel comfortable)* 这一短语来礼貌地拒绝请求。
   - 该用户指出，该模型的安全护栏非常重，甚至不愿编写成年人之间自愿发生的场景，并举了一个包含明确 System 和 User 请求后紧跟该拒绝短语的例子。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1404924933376839740)** (4 messages): 

> `LLM repetition, data quality, RLHF to fix repetition` 


- **LLM 受重复问题困扰**：LLM 倾向于偏向数据集中过度代表的词汇，这会导致在生成过程中出现重复。
   - 在这种背景下，在线 DPO 的奖励作弊（reward hacking）也可能是一个问题。
- **数据质量是关键**：一位成员建议，提高数据质量和多样性可以解决重复问题。
   - 另一位成员指出，高质量的后训练（post-training）数据非常有帮助。
- **利用 RLHF 修复重复**：一位成员在利用 RLHF 时，将重复的输出作为拒绝样本（rejected），取得了不错的效果。
   - 他们声称，你确实需要某种方式来惩罚不良输出才能完全消除重复，仅对良好输出进行正向强化是不够的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405011965415526440)** (26 messages🔥): 

> `Qwen3-4B-Thinking-2507, Jan-v1-4B, Menlo Research, Lucy model, Agentic web search` 


- **Qwen3-4B 表现强劲**：成员们讨论了将 **Hermes** 注入到 [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 或 [janhq/Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B) 中，并强调 **Qwen3-4B-Thinking-2507** 的性能足以媲美 **Qwen3-30B-A3B-Thinking-2507**。
   - [Jan-v1-4B 模型](https://huggingface.co/janhq/Jan-v1-4B) 基于 **Qwen3-4B-Thinking-2507** 构建，其 RL 训练使该 4B 模型成为 **Perplexity** 的有力端侧替代方案。
- **Menlo 的 Lucy 模型支持 Agentic 网络搜索**：讨论重点介绍了 **Menlo Research 的 Lucy 模型**，这是一个专注于 [Agentic 网络搜索](https://huggingface.co/Menlo/Lucy)和轻量级浏览的 1.7B 紧凑型模型，基于 **Qwen3-1.7B** 构建。
   - 该模型利用机器生成的任务向量（task vectors）来优化思考过程，并能在移动设备上高效运行，即使是纯 CPU 配置。
- **动态任务向量机范式**：论文 [Lucy: edgerunning agentic web search on mobile with machine generated task vectors](https://arxiv.org/abs/2508.00360) 引入了一种新范式，将模型的内部推理视为动态任务向量机。
   - 该模型能够即时构建和完善自己的任务向量，并在 **SimpleQA benchmark** 上实现了 78.3% 的准确率，表现与大得多的模型相当。
- **JanAI 中处于早期阶段的 MCP 集成**：成员们提到 [jan.ai](https://jan.ai/) 是一个出色的 **llama.cpp** 封装工具，希望它能从一开始就具备更多 Agentic 能力（如网络搜索等）。
   - 讨论指出其 **MCP 集成** 仍处于早期阶段，目前还没有太多其他可用功能。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405011965415526440)** (26 messages🔥): 

> `Hermes Impartation, Qwen3 Model, Menlo Research, Lucy Model, Dynamic Task Vector Machine` 


- **Hermes 模型正在寻找新家**：成员们讨论了将 **Hermes** 注入到 [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 和 [Jan-v1-4B](https://huggingface.co/janhq/Jan-v1-4B) 等模型中。
   - **Qwen3-4B-Thinking-2507** 模型因其相对于尺寸而言令人印象深刻的性能而受到关注，在基准测试中甚至可以与 **Qwen3-30B-A3B-Thinking-2507** 竞争。
- **Jan 模型解析：Lucy 的衍生版本**：**Jan** 模型基于 **Qwen3-4B-Thinking-2507** 构建，并使用了基于其 [Lucy 模型](https://huggingface.co/Menlo/Lucy)的强化学习技术。
   - 讨论强调 **Lucy** 是一个专注于 Agentic 网络搜索和轻量级浏览的 **1.7B** 紧凑型模型，专为移动设备优化。
- **Menlo Research：新加坡的 AI 实验室**：成员们确认 [Menlo Research](https://menlo.ai/) 是 **JanAI** 的创建者或收购者，总部位于新加坡和越南，并从事机器人技术研究。
   - Menlo Research 通过纯强化学习优化**思考过程**和**平滑奖励函数**，完全不使用任何监督微调（SFT）。
- **Lucy 的动态任务向量机详解**：一篇论文（[arxiv 链接](https://arxiv.org/abs/2508.00360)）将 **Lucy** 描述为利用动态任务向量机来增强小语言模型的推理能力。
   - 该架构将模型的内部推理置于 `<think>` 和 `</think>` 标签内，允许模型在运行过程中构建和完善自己的任务向量。


  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1405300404987887728)** (1 messages): 

> `Multilingual Representation Learning Workshop, Physical Commonsense Reasoning Benchmark` 


- **论文发表机会：Multilingual Workshop 征集贡献者**：[Multilingual Representation Learning Workshop](https://sigtyp.github.io/ws2025-mrl.html) 正在组织一项协作式的 Shared Task，并邀请人们提交其母语的原创 **Physical Commonsense Reasoning Benchmark** 条目；贡献者将被邀请作为数据集论文的作者。
   - 计划提交的人员可以填写 [Google 表单](https://forms.gle/QxyZVqkVG5jbR6wu6)，更多信息请参见 [Shared Task 页面](https://sigtyp.github.io/st2025-mrl.html)。
- **Commonsense Reasoning Benchmark 欢迎多种语言提交**：该 Shared Task 正在征集*任何*非英语语言的贡献，特别强调南非语、白俄罗斯语、波斯尼亚语等语言。
   - 计划于 **8 月 14/15 日**举行可选的 FAQ 会议以解答疑问，Zoom 链接和幻灯片可在 [活动链接](https://calendar.app.google/5h59iwozhbQz1KPJA) 中找到。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405102177134837882)** (6 messages): 

> `PINN and GNN, Small <8b English text base model, TinyLlama-1.1B` 


- **学生研究员寻求 PINN、GNN 和 NLP 领域的机会**：一位拥有信息系统与技术以及分子生物学背景的学生，正在寻求与 **PINN**、**GNN**（特别是用于药物研发）和 **NLP** 相关的研究机会。
   - 他们精通化学信息学数据格式，如 **PDB/PDBx**、**mmCIF**、**SDF**、**mol2**，以及 **OpenMM**、**RDKit** 等工具和 **Pymol** 等可视化软件。
- **关于小型英文文本基础模型的咨询**：一位成员询问是否有优质的小型 **<8b 英文文本 Base Model** 可用，以及较新的模型是否更适合作为 Base Model。
   - 他们明确表示想对比模型大小与模型训练后编写的文本的对数概率总和（sum of log probabilities）。
- **计划进行 TinyLlama-1.1B 模型测试**：一位成员表示他们将测试 **TinyLlama-1.1B** 以及可能的其他模型，以评估它们作为 Base Model 的性能。
   - 该评估旨在确定这些模型在处理其训练周期之后编写的文本时的表现。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1404930788964958408)** (25 messages🔥): 

> `Fourier Extension of RoPE, VO-RoPE, Learnable Dimensionality` 


- **Fourier 扩展 RoPE 几何特性**：一位成员正在*尝试一个想法，通过 **Fourier** 扩展 **RoPE**，挖掘几何特性*，并在该 [GitHub 仓库](https://github.com/JRowe47/nanoGPT_FE_RoPE/blob/master/README.md)中展示，结果显示其 *Loss 比原生版本提升了约 15-20%*。
   - 他们将其与 [FoPE](https://arxiv.org/abs/2412.17739) 进行了对比，指出 *FoPE 有所不同，他们是在做长上下文扩展，试图维持特定信号，而我是在尝试捕捉几何特性——捕捉 RoPE 序列中的波动 (wiggle)*。
- **VO-RoPE 未能脱颖而出**：一位成员指出 *苏剑林也发明了 **VO-RoPE**（很聪明的想法）*，并以 [这个仓库](https://github.com/kyegomez/VO-ROPE) 为例，但它 *没有带来任何收益，所以没人听说过它*。
   - 另一位成员链接了 [原始论文的翻译](https://main-horse.github.io/translations/transformer-upgrade/10862/) 并表示赞同，称他们 *几年前就做过这个，然后觉得“额，似乎没什么用”*。
- **动态卷积建模器构想**：一位成员想要 *一种能够以适当的维度自动学习相对几何特性的东西，有点像动态卷积建模器，可以进一步抽象到学习相对层级结构，而无需预先显式建模——让一切都在运行中学习*。
   - 他们还链接了一篇 [Generalized 论文](https://arxiv.org/html/2406.10322v1)，并补充道 *我认为我可以利用 Random Fourier Feature 将我的想法扩展到可学习维度，只需要设置一个上限即可*。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1405071897955401779)** (12 messages🔥): 

> `RLHF for Auto-Interp, SOAR team RLHF, Delphi Hard Negatives, Reasoning Models for Auto-Interp, Tool Calling for Investigation` 


- **针对自动解释 (Auto-Interp) 说明的强化学习？**：成员们讨论了在检测 **F1** 等评估指标上使用 **Reinforcement Learning (RL)**，以改进自动解释器模型，特别是在 *Automatically Interpreting Millions of Features in Large Language Models* 论文的背景下。
   - 一位成员分享说他们的团队正在探索改进自动解释的方法，另一位成员提到 **SOAR** 的一个团队也计划这样做。
- **SOAR 计划进行 RLHF**：一位成员提到 **SOAR** 的一个团队计划使用 **Reinforcement Learning** 来改进自动解释器模型。
   - 一位成员表示愿意讨论想法并在此项工作中进行协作。
- **Delphi 的难负样本 (Hard Negative) 测试**：一位成员报告了之前在 [Delphi](https://github.com/eleutherai/delphi/tree/dspy) 中使用难负样本进行的小规模实验，但没有明显的改进。
   - Delphi 通过使用相似激活的潜变量 (latents)、句子嵌入相似性或共现特征来支持难负样本。
- **推理模型探索**：一位成员建议使用 **Qwen 3**、**Deepseek 蒸馏模型**或 **OpenAI** 的模型等推理模型，以提高在自动解释任务上的性能。
   - 他们建议专注于改进分布内 (in-distribution) 指标并对解释进行定性评估。
- **使用工具调用调查 SAE**：一位成员建议赋予模型工具调用 (tool calling) 能力，以调查关于 **Sparse Autoencoders (SAEs)** 的假设，可能涉及多轮对话。
   - 早期使用 **llama 70b** 进行多轮调查的研究效果并不理想，但对于经过工具调用/智能体化 (agentic) 训练的新模型持乐观态度。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1404908674530672724)** (35 messages🔥): 

> `Harness dataset pulling, Belebele dataset subsets, Adding internal tasks` 


- **尽管有缓存，Harness 仍触发数据集拉取**：一位用户报告说，在使用 harness 运行任务时遇到了 **429 Too Many Requests 错误**，尽管数据集似乎已经被缓存了。
   - 似乎无论本地是否已缓存，harness 都会尝试拉取数据集；一位用户询问：*有没有办法让我预先下载所有数据集，并告诉 harness 使用本地下载/缓存的数据集？*
- **Belebele 触发速率限制**：一位用户在处理拥有超过 **30 个子集**的 **Belebele** 数据集时遇到了速率限制问题。
   - 用户分享了一个与请求 `huggingface.co/datasets/facebook/belebele` 触发速率限制相关的错误示例。
- **轻松添加内部任务**：要添加用于内部使用的新任务，只需创建任务文件夹并在文件夹内创建 **YAML** 文件即可。
   - TaskManager 可以使用 `include_path` 在默认 `tasks` 目录之外的目录中进行查找。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1404918589995090021)** (1 messages): 

> `Mojo-regex optimizations, Apple GPU support` 


- **Mojo-Regex 获得优化！**：**八月社区会议**录音中包含关于 **mojo-regex** 优化的演讲。
   - 录音可在 [YouTube](https://www.youtube.com/watch?v=t2hAfgDYHoc) 上观看。
- **Apple GPU 支持更新发布！**：**八月社区会议**录音中包含关于 **Apple GPU 支持**的重要更新。
   - 录音可在 [YouTube](https://www.youtube.com/watch?v=t2hAfgDYHoc) 上观看。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1404914629963939872)** (4 messages): 

> `End-to-end Mojo, IO Model similar to Mojo, Type system features` 


- **Modular 贡献者等级提升**：一名成员祝贺 <@307702983653720065> 等级提升，并附上了 [YouTube 视频](https://www.youtube.com/watch?v=f30PceqQWko) 链接。
   - 另一位用户同意实现端到端 **Mojo** 似乎是一个潜在的巨大突破，并表示愿意提供帮助。
- **IO 模型讨论引发关注**：一名成员指出 Andrew Kelly 发表了一场关于 **IO 模型** 的演讲，该模型与他们为 **Mojo** 提议的模型非常相似，并分享了 [链接](https://github.com/modular/modular/pull/4728)。
   - 他们计划添加更多 **type system features**，以尽可能实现零成本并提高 **IO** 的安全性。
- **探索 Mojo 的 Sources 和 Sinks**：一名成员建议探索 **sources and sinks** 方面，并指出 **Mojo** 可以通过泛型轻松实现类似的功能。
   - 他们强调 **Mojo** 还应该擅长 **devirtualizing**，因此需要针对 injectable IO 进行基准测试。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1404914962190831686)** (69 messages🔥🔥): 

> `torch.compile backend=MaxCompiler, Apple Metal Integration, Max Graph Optimization, Kyutai Research Lab, ComfyUI` 


- **MaxCompiler 旨在成为 PyTorch 后端**：一名成员正在实现对 `torch.compile(backend=MaxCompiler)` 训练的支持，并指出[文档稀缺](https://youtu.be/t2hAfgDYHoc?si=HzZFZMmCYG9qHqOu)，**PyTorch 源代码**是主要参考资源。
   - 他们报告称，目前在 PyTorch 上使用 `torch.compile` 训练模型的状态为 `56 failed, 1398 passed, 8 xfailed in 64.57s`。
- **优化 Max 图以融合算子**：成员们讨论了使用许多小算子 (ops) 构建 **Max graph** 与使用大算子相比是否存在运行时性能损失，并询问图编译器是否会融合所有*可融合*的内容。
   - 来自 Modular 的一名成员回应称，他们的融合系统很好但并不完美，他们默认假设算子会很好地融合，并针对特定情况添加了变通方法，建议在发现效果不佳时提交 issue。
- **MaxCompiler 在使用 Mojo 自定义算子时出现中断**：一名成员提到，在使用 **torch-max-compiler** 时，如果使用 Mojo 或其他语言编写的 custom ops，预计会出现 graph breaks，但希望能通过单元测试来了解单图的行为和选项。
   - 另一位来自 Modular 的成员回应称，他们打赌不需要妥协于 graph break。
- **ComfyUI 和 Kyutai 将引入 MAX**：一名成员预测 **MAX** 将比 **vLLM** 更早集成到 **ComfyUI** 中，因为 MAX 编译图像/视频模型中使用的 UNets 的速度明显快于其他方案。
   - 另一名成员补充说，unet 的编译时间非常糟糕，以至于大多数图像和视频从业者除了训练之外，在其他任何场景都使用 eager 模式。
- **AMD Kernel 构建时间截图**：来自公共 AMD 开发服务器的截图显示，在 2x64c 服务器上，kernel 构建总共花费了大约一个小时。
   - 这表明在大幅缩短机器学习模型的编译时间方面存在巨大机会。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1404902719323635833)** (42 messages🔥): 

> `PSO Guarantees, Francois Chollet AGI Timeline, Yannic AGI Timeline, LLM API Batching, MoE Scheduling` 


- ****PSO 的动力学之舞**：无临界性的收敛？**：一名成员分析了**粒子群优化 (PSO)**，解释说 PSO 构建了一个*动力系统 (dynamical system)*，该系统恰好在局部点 $p$ 处有一个吸引子 (attractor)，这意味着它会收敛到一个共识点，但不保证最优性甚至不保证临界性 (criticality)。
   - 他们展示了在动力系统首先收敛的假设下，如何证明收敛到临界点 $p$。他们指出了 [Trelea](https://www.sciencedirect.com/science/article/abs/pii/S0020019002004477) 推荐的一组参数。
- ****Chollet 的乐观展望**：5 年 AGI 时间线引发辩论！**：成员们发现讽刺的是，**Francois Chollet** 现在预测 AGI 将在 5 年内实现，而 **Yannic** 则认为 AGI 还很遥远，感叹人性的双重性。
   - [这篇 Techcrunch 文章](https://techcrunch.com/2025/01/15/ai-researcher-francois-chollet-founds-a-new-ai-lab-focused-on-agi/) 中的一些评论嘲讽了 LLM 的能力，而一些人则指出 Gary Marcus 是房间里唯一的成年人（指其观点更成熟稳重）。
- ****LLM 提供商批处理请求**：MoE 调度的把戏！**：据报道，LLM 提供商在将用户请求发送到 GPU 之前会进行批处理，并且 **MoE 调度**是按批次计算的。根据[这篇博文](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure)，由于输入序列会竞争专家缓冲区 (expert buffers)，这可能导致序列层级的非确定性 (non-determinism)。
   - 一位成员指出，为了防止嵌入层 (embedding layers) 被盗，已经加入了[有意加噪 (intentional noising)](https://arxiv.org/pdf/2403.06634)。
- ****VS Code 的上下文难题**：工作区范围上下文的烦恼！**：成员们惊讶地发现，当 VSCode Chat 的用户选择“添加上下文”时，他们无法选择工作区文件夹中的“全部内容”。
   - 有关此问题的完整演示，请参考[此视频](https://www.youtube.com/watch?v=1if6XbzD5Yg)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1404935496190398596)** (7 messages): 

> `CANN Support, Matrix Game Engine, Nvidia H20 AI Chip, Skyreels based on WAN, MistralAI` 


- **Matrix 游戏引擎：高质量开源**：成员们提到了 [Matrix Game Engine](https://matrix-game-v2.github.io/)，这是一个类似于 genie 的*交互式世界模型 (WM)*，并赞扬了其高质量和开源特性。
   - 该项目旨在在发布创新功能方面超越 **OdysseyML** 和 **WayfarerLabs**。
- **中国警告科技公司购买 Nvidia H20 AI 芯片**：[路透社的一份报告](https://www.reuters.com/world/china/china-cautions-tech-firms-over-nvidia-h20-ai-chip-purchases-sources-say-2025-08-12/)指出，中国正就购买 **Nvidia 的 H20 AI 芯片**向科技公司发出警告。
   - H20 AI 芯片一直引发一些争议。
- **Skyreels 使用 WAN 进行视频生成**：**Skyreels** 项目基于 **WAN2.1**，后者被认为是领先的开源视频生成模型。
   - 最初提议的成员表示 **WAN2.2** 现在甚至更好。
- **发现 MistralAI 推文**：一位成员分享了 [MistralAI 的推文](https://vxtwitter.com/MistralAI/status/1955316715417382979)。
   - 推文内容本身未被直接讨论。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1404944516741599396)** (35 条消息🔥): 

> `DSPy 3.0 发布, MLflow 3.0 集成, 多模态支持, 推理模型` 


- **DSPy 3.0 低调发布**：**DSPy 3.0** 已正式脱离 Beta 阶段。正如 [X 平台](https://x.com/lateinteraction/status/1955384445139292222)上所宣布的，该版本由约 100 人共同贡献，可通过 `pip install -U dspy` 进行安装。
   - 此版本包含与 **MLflow 3.0** 的原生可观测性集成、追踪（tracing）、优化器跟踪以及改进的部署流程，详见 [发布说明](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0)。
- **GEPA 优化器备受关注**：社区对 **DSPy 3.0** 中的新优化器感到兴奋，特别是现已推出的 **GEPA 优化**技术。
   - 某用户团队计划撰写论文，测试该新优化器在生产环境中与旧版优化器的对比效果，希望在面临大规模数据标注挑战时能获得更高的性价比。
- **DSPy 新增多模态 I/O**：**DSPy 3.0** 通过 `dspy.Image` 和 `dspy.Audio` 引入了多模态 I/O、复合类型，以及像 `dspy.History` 和 `dspy.ToolCalls` 这样更高级别的 I/O。
   - 自定义类型现在可以通过 `dspy.Type` 配合适配器（adapters）直接使用。
- **推理模型获得原生支持**：**DSPy 3.0** 现在支持 **GPT-5** 和 **o3** 等推理模型，建议在配置 `dspy.lm` 时使用 `reasoning_effort` 参数。
   - 对于 Anthropic 模型，提供了一个 [两步适配器（two-step adapter）](https://dspy.ai/api/adapters/TwoStepAdapter/) 来触发推理能力，社区成员正在讨论创建一个适配器，将思考 Token（thinking tokens）解析到推理字段中。
- **MLflow 集成文档需求**：成员们正在寻找关于 **DSPy 与 MLflow 集成**的资源和文档，包括 **LLM 可观测性**的细节。
   - 作为回应，社区分享了 [DSPy 可观测性教程](https://dspy.ai/tutorials/observability/#tracing)。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405060277870923836)** (7 条消息): 

> `NotebookLM, 视频转录` 


- **NotebookLM 使用技巧**：成员建议绕过 YouTube，直接将音频上传到 **NotebookLM** 进行转录。
   - 有人提到将音频提取为 **MP3** 格式可能是更好的解决方案。
- **视频转录**：用户可以剪切并粘贴 **视频转录文本**，以提高研究的可访问性并使其更易于理解。
   - 通常隐藏在技术术语背后的丰富知识可以被很好地分解。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1404929023947051134)** (25 条消息🔥): 

> `PDF 上传问题, Discord 垃圾信息, 表情符号自定义, JSON 转 DOCX, 来源内容重复` 


- **PDF 上传缓慢**：一些成员报告 **PDF 上传时间**比平时长。
   - 一位成员还注意到 Discord 频道内的 **垃圾信息（spam）** 有所增加。
- **自定义表情符号**：由于不理想的自动选择，用户对 **自定义表情符号** 表示好奇。
   - 消息中未提供具体解决方案。
- **NotebookLM Google Takeout 失败**：一位用户报告在尝试使用 **Google Takeout** 专门为 NotebookLM **创建备份**时遇到**错误**。
   - 该错误发生在 68 个服务成功备份之后。
- **NotebookLM 被赞誉为天才之作**：一位用户热情地称赞 **NotebookLM** 是他们*最喜欢的生成式 AI 产品*，也是*迄今为止最精巧的 AI 工具*。
   - 他们提到使用它生成了*完美的 60 多页综述*，并表示愿意提供志愿服务。
- **精选 Notebook 存在错误**：一位用户警告不要完全相信 **AI** 的话，指出 **精选 Notebook（featured notebooks）** 内容*不准确且过时*。
   - 他们表达了希望将 **Notebook** 和 **Gemini** 集成到单个界面中的愿望。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1404967559304057064)** (30 messages🔥): 

> `Gemini API issues, Deepinfra provider for Gemini, Mistral 3.1 release, Native tool calling settings` 


- **Gemini API 存在可靠性问题！**：用户报告在使用 Gemini API 时收到**空响应**和 **500 内部服务器错误**，即使是付费账户和拥有免费 GCP 额度的账户也是如此。
   - 一位用户指出，他们在过去 30 分钟内发起了约 **30 次请求**，尽管每次请求支付了 **10 美分**，但得到的全是空响应。
- **Deepinfra 被推崇为 Gemini API 的替代方案**：一位用户建议尝试 **Deepinfra 提供商**作为 Gemini API 的按 token 计费替代方案，声称它通过 provisioned Vertex 提供更高的 TPS。
   - 他们给 Deepinfra 发了邮件，并表示*他们正在使用 provisioned Vertex，并且获得了比 Gemini API 更高的 TPS，尽管他们没有说明具体的 TPS 数值*。
- **Mistral 3.1 发布**：**Mistral 3.1** 模型已发布，并分享了 [Reddit 讨论链接](https://www.reddit.com/r/MistralAI/s/ecbI0glsEO) 以获取更多详情。
   - 未提供性能细节或对比。
- **Tool Calling 的考量？**：一位成员询问是否存在用于原生 tool calling 的模型设置。
   - 未收到回复。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405169914410696847)** (15 messages🔥): 

> `Audio Embeddings, AI workflows in n8n, Web connector in playground` 


- **Cohere 考虑音频 Embedding 模型**：考虑到 Cohere 现有 Embedding 模型的实力，一位成员询问 **Cohere** 是否有计划开发**音频 Embedding**。
- **n8n AI 工作流**：一位成员提到正在尝试一些 **n8n 中的 AI 工作流**，并承诺稍后分享细节。
   - 另一位成员询问这是否就是他们听说过的*无代码 Agent 编辑器*。
- **Web 连接器问题**：一位成员指出，根据 [Cohere 文档](https://docs.cohere.com/v1/docs/overview-rag-connectors)，可以在 playground 中启用 **web 连接器**，但他们找不到该选项。


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1405196065787089058)** (1 messages): 

> `Cohere Labs Scholars Program, ML Research, Information Session` 


- **Cohere 学者计划现已开放！**：**Cohere Labs 学者计划**正在接受 **2026** 届申请，为与 AI 领域的顶尖头脑合作进行 **ML 研究**提供独特机会。
   - 学者将在 **2026 年 1 月至 8 月**期间加入研究团队，这是一个**全职且有偿**的机会，请在 **8 月 29 日**前通过此[链接](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop)申请。
- **欢迎参加宣讲会！**：将于 **美国东部时间 8 月 15 日上午 11 点**举行宣讲会，解答有关**学者计划**的问题。
   - 通过此[链接](https://www.linkedin.com/posts/cohq_cohere-labs-scholars-program-is-now-accepting-activity-7206300826321836032-o8Gj?utm_source=share&utm_medium=member_desktop)注册参加宣讲会。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405073524212760627)** (3 messages): 

> `AI/LLM Evaluation, AI Policy and Governance` 


- **深入探讨 AI/LLM 评估**：一位来自哥本哈根大学的博士生介绍了自己，其研究重点是超越构建新基准测试的 **AI/LLM 评估**。
   - 他们的目标是深入思考**智能评估**，质疑当前的测试和基准测试是否真正衡量了它们声称要衡量的东西。
- **探索 AI 政策与治理**：该学生对 **AI 政策与治理**也有研究兴趣，特别是关于评估方面。
   - 这包括 **LLM 的透明报告标准**、**AI 立法**以及前沿技术的**风险评估**。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1404935409032630313)** (3 条消息): 

> `LlamaCloud, AstraDB, SkySQL, Hallucination-free SQL generation, TypeScript SDK` 


- **AstraDB 成为 LlamaCloud 的新数据汇 (datasink)**：用户现在可以将 **AstraDB 数据库**作为 **LlamaCloud** 中的数据汇连接，通过 [UI 配置以及 Python 和 TypeScript 客户端的编程设置](https://t.co/XFWgPd3r9Y)实现无缝的向量存储和检索。
- **SkySQL 攻克无幻觉 SQL 生成难题**：**SkySQL** 使用 **LlamaIndex** 构建了 AI Agent，能够将自然语言转换为跨复杂数据库模式的准确 SQL 查询，实现了**零幻觉查询**并缩短了开发周期（[公告链接](https://t.co/TgjdSodTbr)）。
- **LlamaExtract 登陆 TypeScript**：**LlamaExtract** 现在已在 **TypeScript SDK** 中可用（通过 `npm install llamacloud-services` 安装）。
   - 一个 **Research Extractor** 演示展示了使用 **NextJS** 的这一能力，允许用户上传研究论文并[提取关键信息](https://t.co/XboMM1AXBs)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1404951648899698831)** (12 条消息🔥): 

> `Llama Index Self-Hosting Docs, Acquiring a paid license for Llama Index, RAG Dev Problem Map, Missing GPT-5 Model` 


- **Llama Index 自托管文档设限！**：一位用户询问在访问 **Llama Index "self-hosting" 文档**时出现的密码提示。
   - 一名成员回应称，这些文档的访问权限*仅锁定给拥有 BYOC 部署的客户*，需联系 Llama Index 的现有销售人员获取访问权限。
- **自托管需要付费许可证！**：一位用户询问关于在 **Groq** 上进行**自托管的付费许可证**获取事宜。
   - 一名成员澄清说 *自托管需要付费许可证*，并引导用户前往 [联系表单](https://www.llamaindex.ai/contact)，同时指出设置过程较为复杂。
- **RAG 开发问题图谱发布！**：一名成员宣布发布 **MIT 许可的 RAG 开发问题图谱 (Problem Map)**，包含 **16 种常见的分解模式**。
   - 他们提出愿与感兴趣的 RAG 开发者分享，并指出该图谱*已帮助 80 多名开发者解决了生产环境问题*。
- **OpenAI utils.py 中缺失 GPT-5 模型！**：一位用户报告称 `llama_index/llms/openai/utils.py` 中缺少 **gpt-5-chat-latest 模型**。
   - 一名成员回应称请*升级 OpenAI 软件包*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1404927968387666082)** (11 条消息🔥): 

> `Manus Wide Research, Raise Tickets for Support, OPPO unlock, Manus Deployment Issues` 


- **自动化 Manus Wide Research 确认**：一位用户询问如何自动化 **Manus Wide Research**，以跳过每个计划任务都需要确认的步骤。
   - 目前系统需要确认，这违背了安排研究任务的初衷。
- **提交工单以获得更快支持**：鼓励用户针对问题提交工单 (Tickets)，并指出由于处理量原因，Discord 工单比邮件响应更快。
   - 解释称*没有明确指导的模糊提示词会导致 Manus 工作更吃力并消耗更多额度*，并建议参考社区指南以提升体验。
- **解锁 OPPO 手机的问题**：一位用户在解锁其 **OPPO** 手机时遇到问题。
   - 支持团队询问他们之前是否联系过，并询问是否有工单编号以便进一步协助。
- **Manus Web App 部署功能尚不完善**：一位用户报告称，虽然 **Manus** 正在进步，但 Web 应用程序的部署仍然不足，理由是部署不可靠。
   - 他们表示，*构建刷新页面或不可用页面反而能赚更多钱*。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405139200394661908)** (5 条消息): 

> `FSDP Implementation, Contributing to tinygrad, define_reg Pull Requests` 


- **询问 FSDP 状态**：一名成员询问在 *tinygrad* 仓库中解决 **FSDP** (Fully Sharded Data Parallelism) 实现的时间表。
   - 该成员还询问了如何进行首次贡献，并咨询了与某个未提及的悬赏 (bounty) 相关的特定 PR。
- **寻找悬赏对应的正确 PR**：一名成员询问完成特定**悬赏**的具体 **PR**，期待能有宝贵的学习经验。
   - 另一名成员提供了一个与 *define_reg* 相关的已合并 PR 链接：[PR 列表](https://github.com/tinygrad/tinygrad/pulls?q=is%3Apr+is%3Amerged+define_reg)。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1404946794521165876)** (3 messages): 

> `Subtensor realization, CUDA_ERROR_UNSUPPORTED_PTX_VERSION, tinygrad CUDA support, Cached kernel issues` 


- ****Subtensor Realization 启示****：一名成员询问实例化（realizing）一个子张量是否必须实例化整个张量。
   - 他们假设 **independent indices** 可能允许部分实例化，但难以通过源代码确认这一点。
- ****Tinygrad 中的 CUDA 错误危机****：有用户报告在运行简单的 tinygrad 程序时遇到 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 错误。
   - 尽管 **nvcc 和 NVIDIA 驱动设置** 看起来是兼容的，但错误依然发生，引发了关于 tinygrad 对特定架构（如 `sm_75`）或 CUDA 版本（如 `12.4`）支持的疑问。
- ****Tinygrad 的 CUDA 难题仍在继续****：一名成员推测 `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` 是由于 **tinygrad** 在从 **CUDA 12.8 降级到 12.4** 后使用了缓存的内核（cached kernel）。
   - 用户确认他们能够编译并运行 **CUDA** 程序。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405251118698598430)** (2 messages): 

> `Claude Desktop, Bun command` 


- **Desktop Claude 遗漏错误**：一名成员建议在外部终端运行 **bun** 命令，因为 *Claude Desktop* 有时无法在其日志中捕获某些错误。
   - 他们还指出，如果路径可执行文件不起作用，你应该提供可执行文件的绝对路径，例如 `"command": "C:\\sys\\path\\to\\bun"`。
- **bun 在哪里？**：一名成员展示了在路径可执行文件不起作用时，如何找到可执行文件的绝对路径。
   - 在 Linux/Mac 上使用 `which <executable>` 查找，在这种情况下意味着输入 `which bun`。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1404962594279915580)** (4 messages): 

> `Kratos MCP release, AI Agents with MCP book release, MCP Harness usage` 


- ****Kratos MCP** 发布，支持持久化内存！**：在对 AI 遗忘项目上下文感到沮丧后，一名成员发布了 **Kratos MCP**，号称拥有 **95.8% 的上下文准确率** 和 **<10ms 的检索速度**。
   - 通过 `npm install -g kratos-mcp` 安装，并查看 [GitHub repo](https://github.com/ceorkm/kratos-mcp) 和 [文档](https://kratos-mcp.com)。
- **《AI Agents with **MCP**》书籍发布！**：一名成员宣布了他们的书籍《AI Agents with MCP》的早期版本，已更新至第 2 章。
   - 一篇解释 MCP 起源的摘录发表在他们的 [newsletter](https://thesignalpath.xyz/the-surprising-origins-of-the-model-context-protocol/) 中。
- **成员发现 **MCP** 服务器的富有想象力的用途**：一名成员强调了 **MCP** 服务器的一个富有想象力的用途。
   - 该用例可以在 [MCP Harness](https://github.com/kindgracekind/mcp_harness) 找到。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1405228493184434206)** (5 messages): 

> `System Prompt Reading, Claude vs. Claude Code prompts, Guardrail approaches, Prompt Engineering` 


- **System Prompt 阅读与讨论安排**：一场专注于“**System prompt 阅读与讨论**”的聊天定于 PT 时间 8 月 14 日上午 9:30 举行，并提供了 [RSVP 链接](https://lu.ma/yuj5og81)。
   - 该活动旨在通过学习 **Claude**、**Claude Code** 和 **GPT-x** 等模型的系统提示词来提升 Prompt Engineering 技能。
- **辩论系统提示词的差异**：讨论将涵盖类似任务在系统提示词上的差异（**Claude Code vs. Cursor**），以及模型的通用版本与专业版本之间的差异（**Claude vs. Claude Code**）。
   - 聊天将进一步探讨 **OpenAI** 和 **Anthropic** 之间的 **guardrail approaches**，以及这些见解如何改进提示词编写。
- **System Prompt 聊天名额有限**：一名成员询问该活动的筛选过程是否会像上一次那样严格。
   - 组织者回答说这取决于报名情况，他们打算事后在 **博客文章中回答问题**。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1405136424562589716)** (2 条消息): 

> `证书未获批准，匿名反馈` 


- **学生认为拒绝发放证书是不公平的**：一名学生认为，即使他们完成了所有课程、通过了所有测验、积极参与了研究方向并撰写了提交的全文论文，仅因漏掉了一次 **LinkedIn 推广帖子** 就被拒绝发放证书是不公平的。
   - 该学生感到非常沮丧，并认为以此理由拒绝发放证书是不公平的。
- **建议将反馈添加到匿名表单**：一名成员建议，如果学生愿意分享，可以将反馈添加到 [匿名反馈表单](https://forms.gle/3a136zS4ivcQFzhT7)。
   - 该成员表示，虽然他们不会对之前的教学大纲进行追溯性更改，但会考虑所有反馈以用于未来的课程。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405306667478028320)** (1 条消息): 

> `Strix Halo, HP Z2 Mini` 


- **Strix Halo 配置具有成本效益**：成员们声称组装一台 **Strix Halo mini PC**（如 **HP Z2 Mini**）可能更具成本效益。
   - 顶配 **APU** 搭配 **128GB RAM** 并以 **8 通道** 配置运行，使其成为一个极具吸引力的替代方案。
- **Intel 尝试迷你工作站设置**：对 **Intel** 尝试销售其全蓝迷你工作站设置表示赞赏。
   - 一些用户认为这款产品价格昂贵。