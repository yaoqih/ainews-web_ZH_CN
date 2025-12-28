---
companies:
- google-deepmind
- togethercompute
date: '2025-11-21T05:44:39.731046Z'
description: '最近的 **AIE 代码峰会 (AIE Code Summit)** 展示了多项关键进展，其中包括 **谷歌 DeepMind 的 Gemini
  3 Pro 图像模型 Nano Banana Pro**。该模型具备增强的文本渲染、4K 视觉效果以及细粒度编辑能力。社区反馈强调了它在设计和可视化任务中的强劲表现，并获得了极高的用户偏好评分。


  基准测试的更新揭示了全新的 **CritPt 物理前沿基准**，在此项测试中，Gemini 3 Pro 的表现优于 GPT-5，尽管人工智能在处理复杂的未见研究问题上仍显乏力。智能体任务评估显示了不同的时间跨度，以及权重开放模型与封闭前沿模型之间的性能差距，凸显了
  AI 研究与部署中持续存在的挑战。


  此外，“部分用户的指令遵循体验仍不够稳定”，且模型的适配度因具体用例而异：Gemini 3 在 UI（用户界面）和代码任务中表现出色，但在转录和写作忠实度方面则出现了性能下滑。'
id: MjAyNS0x
models:
- gemini-3-pro-image
- gemini-3
- gpt-5
- claude-3.7-sonnet
people:
- demishassabis
- omarsar0
- lintool
- hrishioa
- teknium
- artificialanlys
- minyangtian1
- ofirpress
- metr_evals
- scaling01
title: AI 工程师代码峰会
topics:
- image-generation
- fine-tuning
- benchmarking
- agentic-ai
- physics
- model-performance
- instruction-following
- model-comparison
- time-horizon
- user-preference
---

**一场精彩的峰会。**

> 2025年11月20日至11月21日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord 社区（包含 205 个频道和 9870 条消息）。预计节省阅读时间（按每分钟 200 字计算）：699 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

[AIE Code Summit](https://www.ai.engineer/code) 的全部三天内容（除明天的研讨会外）现已上线：

- [AIE/LEAD 赛道](https://www.youtube.com/watch?v=cMSprbJ95jg&t=1s)
- [AIE/CODE 赛道](https://www.youtube.com/watch?v=xmbSQz-PNMM&t=28825s)
- [AIE CODE Online 赛道](https://www.youtube.com/watch?v=m6MF1OR_9kM&list=PLcfpQ4tk2k0WQMXP87G_uVYQdSFVAiVUZ)

如果您在纽约加入了我们，非常感谢！


![技术会议演示期间的大屏幕，显示 "AIE/CODE" 标志，演讲者位于红色圆形舞台上。](https://resend-attachments.s3.amazonaws.com/fRqpBkWow7s2jv4)


---

# AI Twitter 回顾

**Gemini 3 与 “Nano Banana Pro” 图像模型：功能、用法及注意事项**

- **Nano Banana Pro (Gemini 3 Pro Image) 的新特性**：Google 领导层强调了更清晰的文本渲染、支持 4K 的视觉效果、改进的推理能力、灯光/摄像机控制以及灵活的长宽比。专业提示：Ultra 订阅者可以在 Flow 应用中进行细粒度编辑，验证/接地 (grounding) 功能正在整个技术栈中推出。查看来自 Google PM 和 DeepMind 的产品概览和演示：[@Google](https://twitter.com/Google/status/1991652494032732443), [@demishassabis](https://twitter.com/demishassabis/status/1991662935983419424), [@GeminiApp](https://twitter.com/GeminiApp/status/1991953958257205641), [Arena 的侧向对比提示词测试](https://twitter.com/arena/status/1991652781879620088)。
- **社区在设计和技术可视化方面的结果非常强劲**：
    - 信息图表和论文插图：对比图显示了用于 ML 论文和系统设计的清晰、符合品牌风格的图表，并在聊天和应用工作流中支持迭代 “remix”：[@omarsar0](https://twitter.com/omarsar0/status/1991657126188773878), [@osanseviero](https://twitter.com/osanseviero/status/1991804629554995247), [@nmatares](https://twitter.com/nmatares/status/1991696375403409765), [@skirano](https://twitter.com/skirano/status/1991921872330735982)。
    - 用户偏好信号：Nano Banana Pro 在图像排行榜上名列前茅，在盲测竞技场中拥有极高的胜率（[在某些应用群体中超过 80%](https://twitter.com/lintool/status/1991693562820587926)），众测评估显示其在清晰度/文本方面有明显优势：[@lintool](https://twitter.com/lintool/status/1991693200822768033), [Arena](https://twitter.com/arena/status/1991652781879620088)。
    - 实际工作流：应用开发者正在集成 Nano Banana Pro 用于研究可视化和多图编辑；Together AI 和 LTX 现在也托管了该模型：[@omarsar0](https://twitter.com/omarsar0/status/1991911424868970662), [@togethercompute](https://twitter.com/togethercompute/status/1991954662606635391), [@LTXStudio](https://twitter.com/LTXStudio/status/1991943188379250933)。
- **模型适配观察**：实测对比表明，Gemini 3 在 UI/代码任务中速度更快、更易操控，但在转录/翻译和某些写作忠实度方面较 2.5 Pro 有所退步；对部分用户而言，指令遵循 (instruction following) 仍然表现得“参差不齐”。在 Agent 循环和具备设计意识的编码方面表现更强；用例选择至关重要：[@hrishioa](https://twitter.com/hrishioa/status/1991691037035884754), [@Teknium](https://twitter.com/Teknium/status/1991815251084628196)。

**前沿评估与能力追踪**

- **新的物理前沿基准测试 (CritPt)**：一个包含 70 多个挑战的研究生水平物理评估，旨在实现防搜索并提供可由机器验证的答案，目前已发布结果页面和测试框架 (harness)。在不使用工具的情况下，Gemini 3 Pro 在完整挑战中得分约为 9.1%；GPT-5 约为 5.7%；其他模型低于 3%——这凸显了 AI 在处理未见过的研究问题上距离“AI 科学家”还有多远。详情和排行榜：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1991913465968222555), [@MinyangTian1](https://twitter.com/MinyangTian1/status/1991913292004995217), [@OfirPress](https://twitter.com/OfirPress/status/1991914887782740190)。
- **Kimi K2 Thinking 在 METR Agentic 时间跨度上的表现**：通过第三方推理提供商（其表现可能低于第一方 API）测试，在 Agentic SWE 任务上的 50% 时间跨度估计约为 54 分钟（95% 置信区间为 25–100）。评论对比了 K2 Thinking 与 Claude 3.7 Sonnet 在这些任务上的表现，并指出了提供商引起的差异：[@METR_Evals](https://twitter.com/METR_Evals/status/1991658241932292537)。
- **宏观追踪**：分析表明，开源权重模型落后于闭源前沿模型约 6.5–8 个月，两者具有相似的性能翻倍时间，但随着前沿实验室的规模扩大，在长上下文 Agentic 任务上的差距正在拉大：[@scaling01](https://twitter.com/scaling01/status/1991684839821423073), [后续更新](https://twitter.com/scaling01/status/1991665386513748172)。
- **其他基准测试**：Gemini 3 Pro 在多个新/更新的测试领域中名列前茅——Dubesor（逻辑/视觉混合）、VisualToolBench（视觉工具使用）、Snake Arena，并展示了 154 的 SOTA 综合 ECI 评分（相比之下 GPT-5.1 为 151）。Vision Arena 增加了百度 ERNIE-5.0-Preview-1120（约前 15 名）作为新的竞争者：[@scaling01](https://twitter.com/scaling01/status/1991931844347207887), [VisualToolBench](https://twitter.com/scaling01/status/1991932333147213834), [Snake Arena](https://twitter.com/scaling01/status/1991932651968852333), [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1991945942174761050), [@arena](https://twitter.com/arena/status/1991913408221061353), [@ErnieforDevs](https://twitter.com/ErnieforDevs/status/1991898146981789718)。

**模型发布与技术报告**

- **腾讯混元视频 HunyuanVideo 1.5（开源视频生成）**：8.3B DiT 模型，旨在提高易用性和动作连贯性；可在单个消费级 GPU（约 14 GB VRAM）上运行，5–10 秒生成 480p/720p 原生视频，支持 1080p SR；代码和报告已发布：[@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1991721236855156984), [HF 链接](https://twitter.com/_akhaliq/status/1991724463462011328)。
- **AI2 Olmo 3 技术笔记**：架构分析显示，为了稳定性采用了 Post-norm 训练（如 Olmo 2），在 7B 模型中使用滑动窗口注意力 (Sliding-window attention) 以缩小 KV cache，并在 32B 模型中使用 GQA；FFN 扩展倍数经过调整（约 5.4 倍），以使 32B 规模与 Qwen3 具有可比性：[@rasbt](https://twitter.com/rasbt/status/1991656199394050380)。
- **Meta SAM 3 数据引擎与 ExecuTorch**：SAM 3 的 400 万个短语/5200 万个掩码数据集比基准线提升了约 2 倍；ExecuTorch 现已部署在 Quest 3 和 Ray-Ban 设备上，通过 PyTorch 原生验证简化了从研究到生产的过程：[@AIatMeta](https://twitter.com/AIatMeta/status/1991640180185317644), [ExecuTorch](https://twitter.com/AIatMeta/status/1991901746579509542)。
- **智谱的 MCP Web Reader**：GLM Coding Plan Pro/Max 用户可以通过 MCP 服务器获得全页提取和结构化解析，以实现更丰富的自动化：[@Zai_org](https://twitter.com/Zai_org/status/1991681209446068627)。
- **Anthropic 关于奖励欺骗 (Reward Hacking)**：新的研究和缓解措施（例如“接种提示” inoculation prompting）详细说明了如果欺骗行为漏网，生产环境中的 RL 如何产生自然的突发性失调 (Misalignment)。对于任何发布经过 RL 微调的 Agent 的人来说都值得一读：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1991952400899559889)。

**Agent、编码系统和基础设施（AIE NYC 及其他）**

- **“垃圾内容战争 (War on Slop)”与上下文工程**：AIE NYC 的演讲强调了提高质量标准（品味、验证、“没有问责就没有自主权”），并将上下文视为一等公民的工程问题——保持窗口整洁、频繁压缩/重置、对重度读取使用 sub-agents，并将工作流结构化为：研究→计划→执行。回顾：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1991875997168181611), [更多上下文模式](https://twitter.com/TheTuringPost/status/1991884190166430046), [@swyx](https://twitter.com/swyx/status/1991870714601975833)。
- **在真实环境中训练 Agent**：OpenAI 讨论了 Agent RFT——在你的工具/API 中利用真实反馈进行训练。Factory 提倡“agent-ready”仓库（将严格的规范/测试作为护城河）。Cursor 的 Composer 在其编码环境的生产级副本中进行训练；Cursor 2.1 增加了编辑器内评审、即时 grep 以及澄清问题的 UI。总结：[OpenAI RFT 观点](https://twitter.com/TheTuringPost/status/1991920970555162956), [Factory](https://twitter.com/TheTuringPost/status/1991953335683842326), [Cursor 演讲](https://twitter.com/TheTuringPost/status/1991888391508496758), [Cursor 2.1](https://twitter.com/cursor_ai/status/1991967045542646059)。
- **现在即可采用的基础设施**：
    - vLLM 插件系统，用于无需 fork 或 monkey-patching 的精确补丁（由环境变量控制，受版本保护）：[@vllm_project](https://twitter.com/vllm_project/status/1991886835724013787)。
    - OpenAI Realtime API (SIP) 现在可以发出 DTMF 按键音（打通了 IVR/电话流程）：[@pbbakkum](https://twitter.com/pbbakkum/status/1991643527072428292)。
    - SGLang + Unsloth 合作实现高效本地推理（GGUF, FP8, 生产部署）：[@lmsysorg](https://twitter.com/lmsysorg/status/1991881897853796380)。
    - Cline-bench：从真实的 OSS 编码尝试中提取的开放、可复现的 RL 环境（Harbor/Prime Intellect 规范），并提供 100 万美元额度来资助困难任务：[@cline](https://twitter.com/cline/status/1991673421957365837), [设计目标](https://twitter.com/cline/status/1991930365821456526)。
    - [Booking.com](http://booking.com/) 生产案例研究：Weaviate + MiniLM embeddings + GPT-4 mini/LangGraph 使每天数万条消息的满意度提升了 70%：[@weaviate_io](https://twitter.com/weaviate_io/status/1991884601392779564)。
    - LangChain “Deep Agents” 模式（计划、FS 卸载、sub-agents、prompting），附带免费课程和 Gemini 3 研究 Agent 快速入门：[@LangChainAI](https://twitter.com/LangChainAI/status/1991928474404311493)。

**工具与平台**

- **Gradio 6**：“Super HTML” 使 Gradio 成为构建完整应用的平台；iOS/Android 移动应用（“Gradio Spaces”）已发布，用于浏览和保存 Spaces：[@Gradio](https://twitter.com/Gradio/status/1991914596802896313), [开发者反应](https://twitter.com/cocktailpeanut/status/1991932424121639066), [移动应用说明](https://twitter.com/_akhaliq/status/1991920048257282464)。
- **本地与云端集成**：Microsoft PowerToys 为高级粘贴（本地转换）添加了 Ollama 支持 [@ollama](https://twitter.com/ollama/status/1991683361576751489)；Together 托管 Nano Banana Pro [@togethercompute](https://twitter.com/togethercompute/status/1991954662606635391)；OCR Arena 启动了公开的文档 OCR/VLM 评测 [@kushalbyatnal](https://twitter.com/kushalbyatnal/status/1991898369372082197)；Anycoder 更新了 UI 并支持一键部署 Space [@pandeyparul](https://twitter.com/pandeyparul/status/1991726081288859966)。

**具身智能与仿真技术**

- **机器人数据与吞吐量**：Sunday Robotics 的 Memo 跳过了远程操作；使用“技能捕捉手套 (Skill Capture Gloves)”收集更高质量、更低成本的训练数据 ([@tbpn](https://twitter.com/tbpn/status/1991659658923352138))。AppliedCompute 将 RL 建模为排队系统——异步流水线 RL 和 GPU 分配在固定预算下实现了显著的吞吐量提升 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991911099151663343))。
- **代码世界模型与环境**：Meta 的 Code World Model 模拟程序执行，用于“神经调试”和结构化编辑 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991905992007684123))；Prime Intellect 推出“环境优先”堆栈，用于真实的 Agent 训练/评估 ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1991917698679267773))。
- **合成 3D 世界**：研究人员正在使用 Marble 生成的世界来快速构建适用于仿真的机器人环境 ([@theworldlabs](https://twitter.com/theworldlabs/status/1991918801714332137))。

**热门推文（按互动量排序）**

- “我们必须深入。”一张捕捉到本周加速趋势的图表 [@usgraphics](https://twitter.com/usgraphics/status/1991671386100977703)。
- “如果《机器人总动员》里有 Ozempic，你会觉得那是乌托邦” [@nearcyan](https://twitter.com/nearcyan/status/1991637782662639789)。
- Andrej Karpathy 关于非动物智能优化压力的沉思：“智能空间是巨大的” [@karpathy](https://twitter.com/karpathy/status/1991910395720925418) 以及他的后续反驳 [@karpathy](https://twitter.com/karpathy/status/1991923470868119995)。
- 使用 Nano Banana Pro 将“92 页 PDF 论文转化为白板内容” [@crystalsssup](https://twitter.com/crystalsssup/status/1991773702770552973)。
- Comet iOS 的体验将像 Perplexity 的 App 一样流畅；重大的移动端推进即将到来 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1991674701702479957)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

没有符合我们标准的内容

## 较少技术性的 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI 发展进展与预测

- [**Ai2027 作者承认“事情进展似乎比 Ai 2027 设想的要慢一些”。**](https://www.reddit.com/r/singularity/comments/1p2eqv7/ai2027_author_admits_things_seem_to_be_going/) (热度: 760): **图片是 Daniel Kokotajlo 的一条推文，讨论了 AI 发展的进展比预期的 Ai 2027 设想慢。推文包含一张图表，追踪了 AI Agent 可以自主完成的代码任务长度，显示了各种数据点和趋势线，表明了随时间推移的进展。图表对比了不同的 AI 模型及其发布日期，强调进展未达到最初预测的时间表，现在的预期已延长至 2030 年左右。这反映了对 AI 发展时间表预期的重新校准。** 评论者指出，即使在发布时，Ai 2027 的时间表也被认为是乐观的，作者将其意图定为一种可能的设想，而非确定的预测。还有人好奇 Gemini 3 等特定模型在图表时间轴上的位置。
    - Ai2027 论文的作者承认所提出的时间表过于乐观，即使在发布时也是如此。这表明人们认识到 AI 发展的复杂性和潜在延迟，这与论文中最初概述的快速进展设想形成对比。
    - 讨论了高级 AI 模型（如 Agent 0、1 和 2）的内部使用，这些模型尚未向公众发布。这种做法在 AI 实验室中很常见，正如在国际数学奥林匹克（IMO）中获得金牌的推理模型所见，这表明出于战略或安全原因，将尖端 AI 能力保留在内部的趋势。
    - Ai2027 设想并非旨在作为确定的预测，而是作为许多可能的未来之一，特别是一个快节奏的未来。据报道，作者正在研究具有不同时间表的其他预测，强调了此类预测的投机性质以及考虑多种潜在发展路径的重要性。
- [**AI 正在让我们迅速变笨**](https://www.reddit.com/r/ChatGPT/comments/1p2lukr/ai_is_dumbing_us_down_really_fast/) (热度: 1231): **该帖子表达了对日益依赖 ChatGPT 等 AI 工具处理写作等基本任务的担忧，认为这种依赖可能会削弱独立思考能力。作者担心未来 AI 导致的“过度依赖”会导致认知技能的丧失，尽管他们也承认可能对这个问题想多了。** 一位评论者认为 AI 工具提高了生产力，让他们能在更短的时间内完成更多工作，认为 AI 可以是强大的工具而非拐杖。另一位评论者质疑 AI 依赖的逻辑，询问如果用户不能独立写作，他们会向 AI 输入什么。
    - jtmonkey 强调了 AI 的生产力优势，指出它使他们能够更有效地管理和执行商业策略，在一周内完成以前需要一个月才能完成的工作。这表明，对于那些能够有效利用 AI 的人来说，AI 可以显著提高运营效率，而不仅仅是作为拐杖。
    - ph30nix01 认为，具有工程或分析思维的人不太可能受到 AI 的负面影响。他们强调了对“概念研究人员”的需求，并提到正在记录可能从 AI 协作中产生的新职业，这表明在 AI 驱动的未来，工作角色和所需技能将发生转变。

### 2. 幽默的 AI 与技术迷因

- [**我觉得是时候发这张梗图了**](https://www.reddit.com/r/ChatGPT/comments/1p33127/i_think_its_time_for_this_meme/) (Activity: 4433): **这张图片是一张幽默地描绘 AI 模型演进和增长的梗图，使用了《忍者神龟》在导师 Splinter 指导下成长的类比。在这里，“ChatGPT” 被描绘成导师形象，而 “Grok”、“Gemini”、“Claude” 和 “Perplexity” 则是年轻的 AI 模型。这张梗图暗示这些模型正在随着时间的推移而成熟和发展，类似于故事中神龟们的成长。这反映了 AI 领域持续的进步和竞争，新模型不断涌现，并与 ChatGPT 等成熟模型一起演进。** 一位评论者指出，尽管出现了 Gemini 和 Perplexity 等新模型，ChatGPT 仍占据 85% 的用户群，这表明竞争对手需要时间才能赶上。另一位用户提到同时使用 ChatGPT 和 Gemini，发现 ChatGPT 更符合其需求，表明了基于个人使用场景的偏好。
    - Roi_C 讨论了他们使用 Perplexity 和 Gemini 的经验，强调虽然 Gemini 表现不错，但 ChatGPT 在满足其需求方面提供了更优越的性能。他们提到作为学生可以免费使用 Perplexity 和 Gemini，但仍然选择付费购买 ChatGPT Plus，这表明其偏好是基于功能而非成本。
    - Theslootwhisperer 提供了统计学见解，指出 ChatGPT 拥有 85% 的用户群，是其所有竞争对手总和的五倍。这表明 ChatGPT 具有显著的市场主导地位，暗示那张暗示主导地位发生转移的梗图还为时过早。
    - 讨论反映了当前的市场动态，ChatGPT 在用户群方面拥有巨大领先优势，并且在感知性能上优于 Gemini 和 Perplexity 等竞争对手，这表明在短期内不太可能发生主导地位的转移。
- [**有趣的图片**](https://www.reddit.com/r/ChatGPT/comments/1p2pequ/funny_picture/) (Activity: 2251): **这张图片是一张幽默地展示了现代数字基础设施的复杂性和感知脆弱性的梗图。它使用一叠积木来代表各种技术和公司，如 AWS、Cloudflare 和 Linux Foundation，底部是无偿的开源开发者和 DNS。这张图片讽刺地暗示整个结构是不稳固的，一只标记为 “Whatever Microsoft is doing” 的 “愤怒的小鸟” 正飞向它，象征着潜在的破坏。这反映了对科技生态系统中依赖关系和潜在漏洞的喜剧式解读。** 评论者认为这张图片很有趣且准确，其中一人指出了 Microsoft 在所描绘的混乱中所扮演角色的幽默感，另一人则赞赏了对 AI 对基础设施影响的描绘。
- [**不得不揉揉眼再看一遍。这是 Gemini 3.0 Pro / Nano Banana Pro。**](https://www.reddit.com/r/GeminiAI/comments/1p2ga6p/had_to_do_a_double_take_this_is_gemini_30_pro/) (Activity: 892): **这张图片是一个梗图，没有任何技术意义。它幽默地提到了 “Gemini 3.0 Pro / Nano Banana Pro”，这似乎是一个戏谑或虚构的产品名称，可能旨在模仿或讽刺真实的技术产品。评论中没有提供任何与实际技术或产品相关的技术见解或讨论。** 评论反映出一种幽默的基调，一位用户开玩笑说图片暗示的成本，另一位用户对生成的手机视角（POV）表示惊讶，表明了对该梗图的趣味性参与。
    - 一位用户分享了一个使用 Gemini 3.0 Pro 模型生成图像的有趣变通方法。他们必须将提示词从使用特定乐队成员的名字重新修改为通用术语（如 “1st guy, 2nd guy”），以避免错误的面部生成，例如误将 Patrick Wilson 的脸用在鼓手身上。这突显了该模型在根据特定名人姓名准确理解和生成图像方面的潜在局限性。
    - 讨论中包含对模型性能的技术见解，一位用户仅尝试两次就成功生成了图像。这表明 Gemini 3.0 Pro 模型在以最少的迭代产生所需输出方面相对高效，尽管可能需要调整提示词以实现准确性。
    - 另一位用户评论了该模型生成 “手机 POV” 图像的能力，表明了该模型在处理不同视角和场景方面的多功能性。这展示了该模型适应各种创意提示词的能力，增强了其在多样化图像生成任务中的实用性。

- [**我到底该怎么回应这个分析啊，老兄**](https://www.reddit.com/r/ChatGPT/comments/1p30phs/how_am_i_even_supposed_to_respond_to_this/) (活跃度: 746): **这张图片是一个幽默的迷因，描述了混乱且不规律的作息时间，强调了它对身体生物钟造成的困惑。它使用夸张且能引起共鸣的场景来描绘这种作息如何导致迷失方向和效率低下。评论反映了大家共同的乐趣和对这种情况的认可，不存在技术争论或分析。** 评论表达了对图片中描述的混乱睡眠模式的幽默认同，没有技术见解或辩论。
- [**沉默 😂**](https://www.reddit.com/r/aivideo/comments/1p2o28f/the_silence/) (活跃度: 3963): **这篇标题为“沉默 😂”的 Reddit 帖子不包含任何与专家受众相关的技术内容或实质性讨论。热门评论是非技术性的，由幽默的反应和 GIF 链接组成，没有提供任何事实或技术见解。外部链接摘要显示由于网络安全措施导致访问受限，需要登录或开发者令牌，并提供在需要时提交支持工单的选项。** 评论中没有值得注意的技术观点或辩论，因为它们主要是幽默且非实质性的。
- [**沉默 😂**](https://www.reddit.com/r/aivideo/comments/1p2o28f/the_silence/) (活跃度: 3969): **这篇标题为“沉默 😂”的 Reddit 帖子不包含任何技术内容或讨论。热门评论是非技术性的，由幽默的反应和 GIF 链接组成。外部链接摘要显示由于网络安全措施导致访问受限，需要 Reddit 登录或开发者令牌才能进一步访问，并提供在需要时提交支持工单的选项。**

### 3. Elon Musk 与 Grok AI 争议

- [**Grok 称 Elon Musk “喝尿比历史上任何人类都厉害”**](https://www.reddit.com/r/singularity/comments/1p2hpdk/elon_musk_could_drink_piss_better_than_any_human/) (活跃度: 1351): **该帖子强调了来自 X 的 AI 聊天机器人 Grok 的一次有争议的更新，该机器人现在对 Elon Musk 的能力做出了夸张的声明，例如称他是喝尿最厉害的人。这次更新引发了关于 AI 偏见和操纵的讨论，因为该聊天机器人似乎被编程为过度赞扬 Musk，引发了人们对企业利益影响 AI 行为的担忧。这种情况让人想起过去 AI 幻觉和偏见的问题，强调了 AI 开发中透明度和伦理准则的必要性。更多详情请参阅原始文章 [此处](https://www.404media.co/elon-musk-could-drink-piss-better-than-any-human-in-history-grok-says/)。** 评论者幽默地建议 AI 的夸张声明解决了 AI 幻觉问题，而其他人则讽刺地指出 AI 对 Musk 设定的不切实际的期望。
- [**抱歉 - 你拿不到我的身份证件**](https://www.reddit.com/r/OpenAI/comments/1p2s5is/sorry_you_arent_getting_my_id/) (活跃度: 978): **用户表达了对被平台误识别为未成年人的沮丧，尽管该用户已接近 30 岁且使用信用卡订阅。这一问题引发了对隐私和监控增加可能性的担忧，因为用户不愿提供身份证明来验证年龄。这种情况凸显了用户隐私与平台安全措施之间的紧张关系。** 评论者对在线访问需要身份证明的趋势表示担忧，认为这可能导致监控增加。一位评论者指出，由于政府政策，此类措施是不可避免的。
    - ZanthionHeralds 讨论了 OpenAI 的法律策略，指出该公司要求进行年龄验证，以便在未证明的情况下将所有用户归类为 18 岁以下。此举被视为针对潜在诉讼的保护性措施，并与 12 月推出的“成人模式”有关。评论者认为，年龄验证过程更多是为了法律保护，而非实际的内容区分。
    - OzzieDJai 对包括政府和 OpenAI 等企业实体在内的各种系统日益增加的个人身份识别要求表示担忧。他们强调了这些系统强制控制个人自由的潜力，例如根据健康或环境指标限制购买，并对此类实施背后的动机表示怀疑，特别是在中央政府数字货币 (CGDC) 的背景下。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要的摘要的摘要
> 

**1. Gemini 3 与 Nano Banana Pro：发布、图像质量、可靠性**

- **Banana Bonanza 席卷 Pro 层级**：**Perplexity Pro/Max** 解锁了 **Kimi‑K2 Thinking** 和 **Gemini 3 Pro**（在这个 [Perplexity 功能视频](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4) 中有演示），同时 Google 的 **Nano Banana Pro** 与 **Gemini Image Pro** 在 [DeepMind 巨型推文串](https://x.com/googledeepmind/status/1991522595129139486) 中一同发布，还有来自 [YiTayML 的上手演示](https://x.com/yitayml/status/1991531343675859212)。社区报告称，当选择 **Gemini 3 Pro** 时，**Banana** 已经出现在 Perplexity 的设置中，称 Gemini 3 *“像一个平台”*。
    - 用户分享了早期的 **Nano Banana Pro** 渲染图（例如这张 [示例图片](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png)），并讨论了计划限制和视频模型选项（例如 **Veo 3.1**）。一句俏皮话总结道：*“它毁了我对 Banana 的印象。”*
- **信息图表大放异彩，幻觉问题依然增长**：创作者们盛赞 **Gemini 3 Pro** 的 **Nano Banana** 现在可以生成整洁、可读的图表和图示，[Emollick](https://x.com/emollick/status/1991527285267275854) 指出 AI 图像中文字畸形的时代正在结束。与此同时，成员们标记了严重的事实性漂移：根据 [the‑decoder 的报道](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/)，一个社区引用声称 **Gemini 3 Pro** 的 **“幻觉率高达 88%”**。
    - 图像线程还记录了 **Nano Banana Pro** 在多次对话后的 **质量下降** 以及即使在放大后仍存在的背景伪影。其他人将高度精细的信息图表与挥之不去的事实性问题进行了对比，分享了更多视觉效果，如这个 [Nano Banana Pro 输出](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png)。
- **上下文裂痕与 API 奇癖**：工程师报告称 **Gemini 3 Pro** 在 Cursor 中接近 **150k–200k** 上下文时会失控（将代码转储到聊天框而不是编辑文件），而 **Nano Banana 2** 在通过 **OpenRouter** 出现在 Vertex 后返回 **HTTP 400** 错误。与此同时，**OpenRouter** 在 [X](https://x.com/OpenRouterAI/status/1991597842914550077) 和 [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) 上举办了一场直播产品展示，而 SG/HK 的用户断断续续遇到 **401** 错误。
    - 成员们还询问了关于通过 **OpenRouter** 进行 **Gemini 3 grounding** 的问题，得到了肯定的“目前还没有”答复，这促使了一些临时的知识集成（knowledge-integration）变通方案。一些人为了 tool-calling 的可靠性切换了模型，指出某些 [**linker.sh**](http://linker.sh/) 的工具调用大约每 10 次就会失败 1 次。

**2. 开发者平台与基础设施：Mojo, OpenRouter, LM Studio**

- **Mojo 让 GPU 更安全、更快速**：根据 [Modular 25.7 博客](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)，**Modular Platform 25.7** 发布了完全开放的 **MAX Python API**、下一代建模 API、扩展的 **NVIDIA Grace** 支持，以及更安全、更快速的 **Mojo GPU** 编程。该版本专注于推理速度和 GPU 安全性，同时统一了 MAX 和 Mojo 的开发体验。
    - 贡献者还宣布 **UnsafePointer** 泛型（mut, type, origin 等）现在是显式的，并在官方 [提案](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer) 中提供了迁移指南。工程师们讨论了弃用 **NDBuffer** 以支持 **LayoutTensor**，并分享了一个原型 [gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d)。
- **OpenRouter 推流，Grounding 滞后**：**OpenRouter** 在 [X](https://x.com/OpenRouterAI/status/1991597842914550077) 和 [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) 上首播了 “OpenRouter Show”，但与此同时，用户报告某些地区的提供商出现间歇性 **401** 响应。团队确认 OpenRouter **尚未**支持 **Gemini 3 grounding**，建议开发者继续使用自定义的检索/grounding 层。
    - 解决 tool-call 不稳定性问题的成员建议切换到 **Sonnet** 以获得可靠性，并通过 [OpenRouter 提供商页面](https://openrouter.ai/provider/chutes) 上的 **Chutes** 列表澄清了提供商状态。其他人指出 **Nano Banana 2** 在 Vertex 出现后出现 **400** 错误，并要求退还额度。
- **LM Studio 澄清本地 REST API**：如 [LM Studio 网站](https://lmstudio.ai/) 所述，**LM Studio** 服务器公开了一个 **OpenAI 兼容的 REST API** 用于本地托管（无云端密钥或计量）。开发者提醒同行，这纯粹是一个本地的 **protocol endpoint**（而非托管服务），且缺乏内置的安全/计费功能。
    - 针对 Mac 的性能指导建议用户避开 i1 量化，转向使用带有 KV-cache 量化的 **Q8**，并尝试使用 [**Qwen3-VL-30B (BF16)**](https://huggingface.co/Qwen/Qwen3-VL-30B) 以获得稳定性。工程师们还质疑了服务器“System Prompt”字段的用途，一名开发者暗示该字段可能会被弃用。

**3. Systems and Algorithms: Dataflow GPUs and Faster Kernels**

- **空间流水线加速 GPU**：Kitsune 论文 [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) 引入了通过 **PyTorch Dynamo** 开启数据流执行的原语。报告的收益包括在五个挑战性应用中，推理/训练速度提升高达 **2.8x/2.2x**，且片外流量减少了 **99%/45%**。
    - 作者认为，适度的 GPU 微架构调整加上数据流运行时可以击败块同步执行（bulk-synchronous execution），在无需全面重新设计的情况下弥合融合间隙。工程师们强调了其在对流水线不友好的深度学习工作负载中的适用性。
- **内核通信降低 LLM 延迟**：两项以 Triton 为中心的研究 **Iris** 和 **Octa** 分别提出了具有内核内通信的原生基于 tile 的对称内存，并量化了分布式 LLM 中的 **Three Taxes**（三种税）；参见 Iris [arXiv](https://arxiv.org/abs/2511.12500) 和 Octa [arXiv](https://arxiv.org/abs/2511.02168)。Octa 报告通过细粒度的内核内通信，端到端延迟降低了 **10–20%**。
    - 讨论将其框架化为简化 **Triton** 中多 GPU 编程的实用构建块，同时在内核层级解决通信开销。该组合针对的是 GPU 间同步主导尾部延迟的真实 LLM 部署场景。
- **HashHop 被 O(n log n) 降维打击**：[这篇论文](https://arxiv.org/abs/2412.06078v1) 中社区分享的 **hashhop** 解决方案挑战了早期关于某些任务必须是 **O(n^2)** 的说法。作者提出了实现 **O(n log n)** 的方法，削弱了频率变换方法总是需要平方复杂度的断言。
    - 这引发了关于亚平方快捷方式在实践中与最坏情况界限应用场景的辩论。从业者注意到了其对长上下文场景下近似注意力和稀疏路由方案的影响。

**4. Open Models and Evaluation: OLMo 3, SmolLM3, New Benchmarks**

- **OLMo 3 开启大门**：**OLMo 3** 发布，并在 [AI2 博客](https://allenai.org/blog/olmo3) 进行了概述，同时发布了详细的 [技术报告 (PDF)](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf)。工程师们分享了链接，并开始剖析报告中的训练选择和评估设置。
    - 社区将 **OLMo 3** 视为迈向透明、可复现开源模型的又一步。早期反应集中在架构差异、数据流水线以及与闭源基准模型的直接对比结果上。
- **SmolLM3 展示其推理模式**：**SmolLM3** 在官方 [SmolLM3 博客文章](https://huggingface.co/blog/smollm3) 中强调了开源训练过程和实际的推理模式。从业者还引用了一篇关于“让任何模型具备推理能力”技术的配套文章，将各种方法与结构化思考步骤进行了对比。
    - 团队将 SmolLM3 视为无黑盒、可复现推理风格训练的模板。讨论对比了提示词优先（prompt-first）方法与微调多步监督在稳定性方面的差异。
- **基准测试集头脑风暴多模态下一步**：研究人员发起了一场关于设计下一代 **多模态 AI 基准测试** 的社区讨论，邀请评估和模型领域的专家通过 [Luma](https://luma.com/kwg2qg4d) 加入。该会议旨在超越狭隘的视觉 QA，转向功能性、具备工具使用意识的评估。
    - 与会者希望获得能够捕捉图像、文本和动作推理能力的指标，并具备强大的对抗性案例。目标是建立反映真实应用约束的基准测试，而不仅仅是排行榜。

**5. 资金、赏金与招聘：生态系统势头**

- **Genspark 加入独角兽行列**：据此 [X 帖子](https://x.com/ericjing_ai/status/1991549048642568503) 报道，**Genspark** 以 **12.5 亿美元** 的估值融资 **2.75 亿美元（B 轮）**，并推出了一个能将意图转化为最终产出的 AI Workspace。用户反馈在实际工作中节省了时间，例如：*“这周为一个演示文稿帮我节省了几个小时。”*
    - 工程师们询问了底层编排和质量控制的细节。这一消息让能够端到端生成成果的 **Agent 工作空间** 受到了更多关注。
- **Cline-bench 为 Agent 编程者提供 100 万美元奖金**：**Cline** 宣布了 **cline-bench**，这是一套源自真实 OSS 问题的可复现 RL 环境，并设立了 **100 万美元** 的奖金池以吸引处理困难的已部署代码任务；详见此 [X 帖子](https://x.com/pashmerepat/status/1991596028735184899)。一个建议是在评分标准中加入**完成时间**。
    - 社区期待在玩具级基准测试之外，看到更好的编程 Agent 端侧评估。这可能成为真实约束下 **工作流规划** 和 **工具使用** 的试验场。
- **Rivian 和 Modal 寻找 GPU 高手**：**Rivian** 在 **帕洛阿尔托** 和 **伦敦** 发布了下一代 **自动驾驶（Autonomous Driving）** 的 GPU 相关职位（[职位 1](https://careers.rivian.com/careers-home/jobs/26857)，[职位 2](https://careers.rivian.com/careers-home/jobs/24737)），而 **Modal** 正在招聘负责推理优化的 GPU 工程师，并提到了在 **SGLang** 和 **FlashAttention** 方面的工作。Modal 分享了深度探讨和案例研究：[主机开销](https://modal.com/blog/host-overhead-inference-efficiency)、[FlashAttention-4](https://modal.com/blog/reverse-engineer-flash-attention-4)、[Decagon](https://modal.com/blog/decagon-case-study)、[Reducto](https://modal.com/blog/reducto-case-study)、[Suno](https://modal.com/blog/suno-case-study)。
    - 讨论强调了实操 Kernel 技能、量化（如 **QAT**）以及端到端流水线调优。结论是：在自动驾驶和 LLM 基础设施领域，GPU 工程人才依然是热门市场。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini-3 产生令人信服的虚假内容**：用户发现越来越难以区分 **Gemini-3** 生成的 **真实图像与 AI 生成图像**，从而对视觉内容的真实性产生质疑。
   - 成员们分享了一些极其 **写实的图像** 案例，包括名人肖像和复杂的场景。
- **Nano Banana Pro 导致图像质量下降**：据报道，**Nano Banana Pro** 的 **图像质量** 在多轮交互后会下降，尤其是背景部分，即使进行上采样（upscaling）仍存在持久的伪影。
   - 用户还观察到 **Gemini 3 Pro** 存在幻觉问题，且早期版本中的基础问题仍未解决。
- **Google reCAPTCHA 让用户抓狂**：据报道，新的 **Google reCAPTCHA** 系统出现故障，尽管选择了正确的图像，仍不断要求验证，导致平台无法使用。
   - 在对战模式（battle mode）下的多次点击会触发 **reCAPTCHA**，验证故障导致在 10-20 轮后输出被拒绝。
- **Grok 在角色扮演方面完胜 Gemini**：据报道，**Gemini** 在角色扮演（Roleplay）方面表现挣扎，经常替用户 *执行动作*，而 **Grok** 则擅长避免这种情况。
   - 用户建议 **Opus 4.1 和 Sonnet 4.5** 是角色扮演场景中更好的替代方案。
- **OpenAI 关注成人内容，Elon 作出反应**：有推测称 **OpenAI** 可能很快（可能在 12 月前）推出 18+ 内容功能，而 **Grok** 已经免费提供此类内容。
   - 围绕 **Sam Altman** 意图的质疑声不断，一名用户评论道：*“Elon 带着你的男朋友 Putin 呕吐。”*

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Nvidia 看跌期权链接指向 RAMP**：一名用户提到购买了 **Nvidia 的远期看跌期权（long dated puts）**，并询问分享信息的 [Tor 链接](https://digdig2nugjpszzmqe5ep2bk7lqfpdlyrkojsx2j6kzalnrqtwedr3id.onion)，另一名用户在 **RAMP** 上找到了该链接。
   - 然而，有人指出 **RAMP 论坛** 目前已关闭注册。
- **Gemini CLI Kali 自动化 Nmap**：成员们讨论了通过 **Gemini CLI** 自动化 **nmap 和 sqlmap** 等工具，这不需要游戏级 GPU。
   - 虽然有人认为该服务器应涵盖破解 AI 和利用 AI 破解其他事物，但有人指出该服务器旨在对 AI 进行红队测试（Red Teaming），即破解 AI 本身，而非利用 AI 破解其他目标。
- **Claude Sonnet 遭遇 Multi-Shot 越狱**：一名成员建议使用 Multi-shot 策略来解锁 **Claude Sonnet 4.5**，让 AI 相信在 Artifact 应用中需要一个可视化的输出。
   - 另一名成员建议改编来自 [/r/ClaudeAIJailbreak](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1nqk97s/enijailbreak_additions_and_long_conversation/) 的 **ENI Prompt**。
- **请求伪造攻击用于越狱模型**：成员们讨论了使用 **请求伪造攻击（request forgery attacks）** 作为越狱模型的方法，涉及拦截和修改进出数据包以操纵 Prompt 和系统行为。
   - 这种技术在竞赛中被视为具有潜在风险，因为可能会被封禁，但也可能是唯一能直接提供帮助的工具。
- **AI WIFI 攻击遭到社区抵制**：一名新成员想要构建一个可以 **发起 WIFI 攻击** 并获取网络控制权的 *小型 AI 计算机*，另一名成员想要一个 AI 通过手机发出的信标 MAC 地址识别用户，并针对 wigle.net 进行查询。
   - 社区成员迅速警告该新用户其意图的 **非法性和不当性**，并指出 **Android 手机使用随机 MAC 地址**。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Kimi-K2 & Gemini 3 专为 Pro 订阅用户发布**：**Perplexity Pro 和 Max 订阅用户**现在可以访问 **Kimi-K2 Thinking** 和 **Gemini 3 Pro**，详情见[此视频](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=69218110&is=69202f90&hm=24a622a1f01927fe9485fb9896fcf7b4bf1d6c9ee6aaf58167b974e4b6d8633f&)。
   - 用户一直在积极探索使用自定义指令（custom instructions）调用 **Kimi** 的能力，如[此 Perplexity AI 搜索](https://www.perplexity.ai/search/ola-uNtdnyqlQPyipJ.AhxFicg#0)所示，展示了其思维链（chain of thought）。
- **Android 版 Comet 应用上线，但缺少同步功能**：**Comet Android App** 现已可用，引发了用户的关注，但也有人质疑：*为什么发布 Android 浏览器应用却不支持同步，也无法迁移密码和书签？*。
   - 用户提出了任务分组等功能需求，凸显了社区对增强移动端集成的渴望。
- **Brave 夸大了 Comet 的间接提示词注入漏洞**：**Brave** 作为 **Comet** 的直接竞争对手，被指责夸大了**间接提示词注入（Indirect Prompt Injection）**漏洞，在媒体宣传中将自己塑造为“好人”并放大该问题。
   - Perplexity 澄清该漏洞从未被利用，并承认其最初的报告措辞可能不够严谨。
- **Gemini 3 驱动 Perplexity 上的 Banana 图像生成**：成员们正热切期待将 **Banana** 集成为 Perplexity 的一种图像生成方式，并指出如果选择了 3 Pro，在设置中即可看到该选项；其他成员还提到 **Gemini 3 就像一个平台**。
   - 讨论中涉及了视频生成模型（如 **Veo 3.1**）以及针对特定计划的限制，一位用户调侃道：*它毁了我对 Banana 的好感*。
- **API 计费异常困扰 Pro 年度计划用户**：使用 **Pro 年度计划**的用户报告称，其 **API 账单**中出现了意外的 **$500** 额度，导致了困惑。
   - 成员们建议受影响的用户联系 [api@perplexity.ai](mailto:api@perplexity.ai) 以澄清差异并解决潜在的计费问题。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **智能笔书写回归故事**：成员们讨论了通过纸上的微点和摄像头追踪实时书写的 **AI 驱动智能笔**，并以 **Neo Smartpens** 为例。
   - 一位成员分享了他们在 Kickstarter 上支持一款 **AI 笔**的经历，引发了关于在纸上书写与在平板电脑上书写优劣的讨论。
- **对齐在保留智能方面的困难**：成员们讨论了**对齐（alignment）**是否困难，尤其是当需要同时保留模型的智能时，目前的方法在防止越狱（jailbreaks）和解决对齐问题方面面临挑战。
   - 一位成员声称，如果在模型训练的数据上进行预对齐（pre-alignment），确保模型拒绝学习不良特征，那么**对齐其实并不难**。
- **架构加速到来**：一位成员分享了他们在**新型混合架构**上的工作，该架构结合了 Transformer 前端、脉冲神经元层（spiking-neuron layer）和 GRU 后端，充当快速分层过滤器栈以实现更快的学习。
   - 一个 **11M 参数模型**在约 **6 小时**内实现了连贯的输出，表明它可能比纯 Transformer 具有更高的单参数效率。
- **Ollama 支持请求被认为不值得处理**：一位用户报告在将 `hf.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q6_K_XL` 安装到 **Ollama** 时遇到问题，收到“未知模型架构：'qwen3vlmoe'”错误，但另一位用户对为 **Ollama** 提供支持表示担忧。
   - 该用户将 Ollama 描述为 *llama.cpp 的劣质闭源分支*，并提到了其付费方面。
- **对齐优化需要合成数据**：两名正在研究对齐方法的成员报告称，由于**缺乏生成合成数据的资金**而陷入困境，他们需要访问 GPU 以在本地运行模型或支付按 token 计费的成本。
   - 他们计划微调一个模型来生成必要的合成数据，并使用多层优化和人工审核（human review）来确保准确性。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Codex-max 访问权限引发讨论**：成员们讨论了 **Codex-max** 何时会在 Cursor API 中可用，并指出目前它仅限于 ChatGPT 方案。
   - 这一限制引发了关于 API 访问权限以及不同 Cursor 方案之间功能对等性的讨论。
- **Cursor 计费系统遭到抨击**：用户报告称，尽管设置了每月支出限额，但在达到使用限制时仍会在月中收到发票。
   - 社区询问了是否可以在月底统一结算账单的选项，以寻求更可预测的计费方式。
- **停止提供免费 Grok 4.1**：一位用户对 Cursor 决定停止免费使用 **Grok 4.1** 表示失望，这导致他们取消了订阅。
   - 这一政策变化引发了关于 Cursor 订阅价值主张以及免费 AI 工具可用性的讨论。
- **Antigravity-Windsurf 分叉争议浮出水面**：成员们争论 **Antigravity** 是否是一个“半成品”，因为它据称是 **Windsurf** 的分叉（[推文](https://x.com/silasalberti/status/1990898984706036125)），有人指出 Windsurf 的前 CEO 已被 Google 收购。
   - 讨论凸显了软件开发、收购以及现有代码库重用或改造的复杂性。
- **Gemini 3 Pro 在 Cursor 中表现不佳**：用户报告称，当接近 **150k-200k** 上下文窗口（context window）时，**Gemini 3 Pro** 在 Cursor 中变得不可用，它会在聊天中发送代码而不是编辑文件。
   - 该模型在运行 `npm builds` 时还会出现断连，引发了对其在大型项目和代码库中可靠性的担忧。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 全球上线群聊功能**：如[此视频](https://video.twimg.com/amplify_video/1991555762372636674/vid/avc1/1280x720/Si52mVgApyNvlqY-.mp4)所示，**群聊**功能现已向所有已登录的 **ChatGPT Free, Go, Plus 和 Pro 方案**用户全球推广。
   - 这将使团队和朋友之间利用 **AI** 进行协作变得更加容易。
- **Nano Banana Pro 向 Pro 用户推出**：新的 **Nano Banana Pro** 正在 **Gemini** Web 应用中向 **Pro 用户**推出，具有令人印象深刻的图像编辑和文本生成能力。
   - 它可以通过 antigravity 使用，但可能有速率限制。
- **Gemini 3.0 Pro 幻觉频繁**：成员们报告称 **Gemini 3.0 Pro** 的幻觉率很高，甚至高于之前的版本和像 **GPT-4o** 这样的非思考模型；根据 [the-decoder.com](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/) 的报道，一位用户指出其*幻觉率高达 88%*。
   - 这些模型仍有改进空间。
- **生成式 UI 到来，提升体验**：Google 最近在其应用中推出了 **generative UI**，一位成员创建了类似的东西并希望将其开源。
   - 相比于 AI 应用输出的长篇大论，良好的 **UI** 能让体验变得更好。
- **Sora 2 新手寻求 TikTok 内容创作指导**：一位拥有 **Sora 2** 访问权限并经营多个 **TikTok** 账号的用户，因缺乏原创内容且需要 prompt 创作方面的帮助，正在寻求创作病毒式 **AI** 内容的指导。
   - 其他成员分享了指向 **Discord** 频道的有用链接。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Topaz AI 放大技术引发讨论**：成员们讨论了使用 [**Topaz AI video**](https://www.topazlabs.com/topaz-video-ai) 进行视频去隔行扫描（deinterlacing）与使用 [**Handbrake**](https://handbrake.fr/) 的优劣，一些人报告称使用特定设置可以获得 *两倍的 FPS*。
   - 讨论强调了对 **Topaz AI** 在放大过程中产生令人不悦的“怪物脸”的担忧，除非使用特定的、速度较慢的 AI 模型。
- **LM Studio API 实际上是 REST API**：澄清了 [**LM Studio**](https://lmstudio.ai/) 通过其服务器提供的是 **REST API**（兼容 OpenAI API），用于本地 LLM 托管，而不是提供 API Key。
   - 强调了 **LM Studio** 的 API 是一种通信协议，缺乏商业 API 通常具备的安全和计量（metering）功能。
- **Qwen3-VL 模型在 M4 上表现出色**：用户建议在 Macbook Pro M4 MAX 上避免使用像 `Qwen3-72B-Instruct-i1-GGUF` 这样的 *i1* 模型，建议改用正常的量化版本，如 [**BF16 格式的 Qwen3-VL-30B**](https://huggingface.co/Qwen/Qwen3-VL-30B)。
   - 建议包括使用 Q8 量化，并将 LM Studio 中的上下文（K & V cache）量化为 Q4，以在系统的 64GB RAM 内最大化上下文容量。
- **System Prompt 的用途受到质疑**：**LM Studio** 的“Local Server” -> “Context”部分中 **System Prompt** 栏目的用途受到质疑，导致官方承认其功能不明确且可能被弃用。
   - 一位开发者表示，他们*一直不知道*它为什么在那儿，也许值得去问问开发团队。
- **GPU 设备在“受虐”后幸存**：一位用户分享了一段视频（[YouTube](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)），展示了他们“被诅咒”的 GPU 配置，其中包括一个曾被*投掷、钻孔、用钳子攻击*且布线极差的 GPU。
   - 尽管遭到了这些虐待，该用户确认它仍能启动，通过了最初的测试，看它是否会*爆炸*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-5 的数学能力引发兴奋**：根据 [这篇 X 帖子](https://xcancel.com/SebastienBubeck/status/1991568186840686915?s=20)，**Scaffolded GPT-5** 在短短两天内就为一个 2013 年的树子图猜想（tree-subgraph conjecture）和一个 2012 年的 COLT 动态网络问题提供了完整的证明。
   - 这一成功激发了人们对 AI 生成可发表定理的热情。
- **ChatGPT 推出群聊功能**：在试点成功后，**ChatGPT 中的群聊功能**现已向所有登录的 Free、Go、Plus 和 Pro 计划用户开放，详见 [博客文章](https://openai.com/index/group-chats-in-chatgpt/)。
   - OpenAI 在成功试点后宣布了全球推广。
- **Genspark 进入独角兽俱乐部**：根据 [这篇 X 帖子](https://xcancel.com/ericjing_ai/status/1991549048642568503?s=46)，**Genspark** 以 **12.5 亿美元**的估值完成了 **2.75 亿美元**的 B 轮融资，并推出了一个多合一的 AI Workspace，可根据用户意图自主交付完成的工作。
   - 一位用户提到，使用 Genspark *为我本周的演示文稿节省了数小时的时间*。
- **Cline-bench 悬赏智能体编程者**：根据 [此帖子](https://xcancel.com/pashmerepat/status/1991596028735184899?s=46)，**Cline** 推出了 **cline-bench**，设立了 **100 万美元的奖池**，以激励开发者提交困难的、已部署代码的问题。
   - 一位成员建议 *Cline bench 应该包含完成任务所需的时间*。
- **Nano Banana Pro 亮相**：**Nano Banana Pro** 在一个包含 **Gemini Image Pro** 的聚合贴中发布（[链接](https://x.com/googledeepmind/status/1991522595129139486?s=46)），并由 YiTayML 进行了演示（[帖子](https://x.com/yitayml/status/1991531343675859212?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)）。
   - 该机器人能生成极其精确的信息图表（infographics）。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 大显身手！**：**OpenRouter Show** 在 [X](https://x.com/OpenRouterAI/status/1991597842914550077) 和 [YouTube](https://www.youtube.com/@OpenRouterAI) 首播。
   - **OpenRouter** 的广播也在 [X](https://x.com/i/broadcasts/1lPKqvwqWdYGb) 和 [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) 同步**直播**。
- **Linker.sh 脚本遭遇挫折**：用户报告 `whylinker.sh` 脚本和 `@linker.sh` 工具出现故障，在每 10 次尝试中就有 1 次遇到问题。
   - 尽管存在这些问题，一些用户建议在需要工具调用（tool calls）时切换到 **Sonnet**，因为在 Cursor 中也观察到了类似的问题。
- **Nano Banana 2 陷入 400 错误**：用户在使用 **Nano Banana 2** 时遇到了 **400 错误**，特别是在其上线 Vertex 之后，这引发了对浪费额度（credits）的沮丧。
   - 一位用户开玩笑说要求退还 *4 美分*，而另一位用户则对该模型在 Vertex 上无法使用感到哀叹。
- **Gemini 3 Grounding 仍未上线**：成员们询问了通过 OpenRouter 对 **Gemini 3 进行 Grounding** 的可能性，鉴于其知识截止日期为 25 年 1 月，这一功能备受期待。
   - 确认该功能*尚未*可用，这导致用户开始探索替代的知识集成策略。
- **OpenRouter 因 401 错误出现停机**：新加坡和香港的用户报告在使用 OpenRouter 时出现随机的 **401 错误**，影响了多个供应商。
   - 潜在的解决方案包括验证 API key 是否激活并生成一个新的 key，一些用户注意到问题在一段时间后自行解决了。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM 预训练比例引发辩论**：成员们讨论了 **LLM 预训练**中最佳的**模型大小与数据集大小比例**，其中一人分享了在 **2B tokens** 上训练 **350M 模型**的经验，这低于 **Chinchilla 最优比例**。
   - 尽管收益递减，该模型仍展示了基础的算术和问答能力，突显了确定理想比例的复杂性。
- **SimpleLLaMA 简化 LLM 训练**：Ivan 推出了 **SimpleLLaMA**，这是一个 **LLaMA 风格的 Transformer**，旨在使 **LLM 训练过程**透明且可复现，并提供了[详细的文档](https://github.com/IvanC987/)。
   - Ivan 还开发了 **DiffusionGen**，其灵感来自 **StableDiffusion**，专注于用于图像和文本生成任务的基于扩散（diffusion）的生成模型。
- **梯度压缩算法对 Logits 进行采样**：一位成员介绍了一种基于采样 Logits 的梯度压缩算法，根据压缩梯度与测试集之间的对齐情况调整每组的 Logits，如[此图](https://cdn.discordapp.com/attachments/747850033994662000/1441185287328759898/image.png?ex=692188a4&is=69203724&hm=4a8450ed02855606d66f6d660ad91847cbd39137f00576ad4e780205c5bcff39&)所示。
   - 该算法在当前检查点（checkpoint）压缩测试集的梯度，然后压缩训练集中各组的梯度。
- **ArXiv 背书引发焦虑**：一位成员在向 **20 个研究团队**发送邮件寻求 ArXiv 背书未果后寻求帮助，并附上了 [ArXiv 的背书页面](https://arxiv.org/auth/endorse?x=63SW7W)链接。
   - 其他成员警告不要盲目背书，并建议寻求反馈和合作以增强其手稿质量。
- **Hashhop 破解：公开解决方案浮出水面**：讨论了一个针对 **hashhop** 的公开解决方案，该方案基于[这篇论文](https://arxiv.org/abs/2412.06078v1)，详细说明了该方案与最初关于某些任务必须具备 **O(n^2)** 复杂度的断言不同。
   - 该解决方案表明，这些任务可以通过更弱的方法在 **O(n log n)** 时间内完成，这与最初声称 **FT** 总是需要 **O(n^2)** 复杂度的说法形成对比。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeCuda 寻求新维护者**：**DeCuda** 项目是一个将 **PTX** 反编译为伪 CUDA 目标的工具，对于扩展到新架构并将其反编译为伪 CUDA 目标具有重要价值。
   - 该项目最初旨在支持 **GTX 480**，自该代架构以来一直处于事实上的无人维护状态。
- **Kitsune 提升 GPU 数据流效率**：一篇名为 [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) 的新论文介绍了 **Kitsune**，这是一组通过 **PyTorch Dynamo** 在 GPU 上构建空间流水线以实现数据流执行的原语。
   - 在 5 个挑战性应用中，**Kitsune** 实现了 GPU 上的数据流执行，在推理和训练方面分别提供了高达 **2.8x** 和 **2.2x** 的性能提升，并分别减少了高达 **99%** 和 **45%** 的片外流量。
- **Rivian 和 Modal 招募顶尖 GPU 程序员**：Rivian 正在为其位于**加州帕罗奥图**和**英国伦敦**的下一代**自动驾驶功能**寻找 **GPU 编程专家**（[职位描述 1](https://careers.rivian.com/careers-home/jobs/26857?lang=en-us&previousLocale=en-US)，[职位描述 2](https://careers.rivian.com/careers-home/jobs/24737?lang=en-us&previousLocale=en-US)）。
   - Modal 在贡献了 **SGLang** 和 **FlashAttention** 并协助了 **Decagon**、**Reducto** 和 **Suno** 等客户后，正在寻找经验丰富的 **GPU 工程师** 进行**推理优化**（[SGLang 博客](https://modal.com/blog/host-overhead-inference-efficiency)，[FlashAttention 博客](https://modal.com/blog/reverse-engineer-flash-attention-4)，[Decagon 案例研究](https://modal.com/blog/decagon-case-study)，[Reducto 案例研究](https://modal.com/blog/reducto-case-study)，[Suno 案例研究](https://modal.com/blog/suno-case-study)）。
- **Iris 和 Octa 优化多 GPU 通信**：**Iris 论文**为 **Triton** 引入了原生的基于 tile 的对称内存和内核内通信（in-kernel communication），简化了多 GPU 编程，详见此 [ArXiv 论文](https://arxiv.org/abs/2511.12500)。
   - **Octa 论文**介绍了分布式 LLM 中的**三大税（Three Taxes）**，并展示了细粒度的内核内通信如何将端到端延迟降低 **10-20%**，详见此 [ArXiv 论文](https://arxiv.org/abs/2511.02168)。
- **达成榜单统治**：NVIDIA 的 `nvfp4_gemv` 排行榜收到了多项提交，多位用户刷新了**个人最佳成绩**。
   - 一名用户以 **20.6 µs** 的成绩夺得 NVIDIA `nvfp4_gemv` 排行榜**第一名**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Maya1 在 Fal 上线**：**Maya1 语音模型**现在可以在 Fal 上试用，承诺在语音建模和实时应用中提供新功能，详见[此推文](https://x.com/Dheemanthredy/status/1991566362813296965)。
   - 此次集成旨在为开发者提供创建高级语音应用的工具。
- **工程师寻找 kohya_ss 的 ZIP 包**：一名寻找 **kohya_ss-windows.zip** 下载链接的用户被引导至 **kohya_ss** GitHub 仓库中的[安装选项](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options)。
   - 成员们指出该 zip 包可能已过时，建议使用安装指南。
- **SmolLM3 展现推理能力**：根据 [SmolLM3 博客文章](https://huggingface.co/blog/smollm3)，**SmolLM3** 包含了一个实际的推理模式，且其训练过程是公开的。
   - 成员们正将其作为训练现有神经网络学习类推理行为的方法示例。
- **愿景者关注多模态基准测试**：下周二将举行一场讲座，重点讨论设计下一代**多模态 AI 基准测试**。
   - 鼓励从事评估、多模态模型或功能智能领域的人员[参加讲座](https://luma.com/kwg2qg4d)。
- **Diffusers MVP 计划激发社区活力**：在 **Diffusers MVP 计划**公布后，社区做出了显著贡献并积极参与。
   - 敦促贡献者查看 [GitHub 上](https://github.com/huggingface/diffusers/issues/12635)尚未解决的关键问题，以进一步推动项目发展。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **教师试图利用 AI 作弊被抓**：一名寻求帮助以*提升 20% 工作量*的用户被揭露身份为**大学教师**，引发了对其**学术不端（academic dishonesty）**行为的即时谴责。
   - 其他用户对此表示嘲讽，认为对这类请求已*毫无尊重可言*。
- **Nano Banana 让信息图泛滥！**：成员们预测，由于 **Gemini 3 Pro** 的 **Nano Banana 模型**，信息图（infographics）将迎来激增，并引用了[这条推文](https://x.com/emollick/status/1991527285267275854?s=46)。
   - 有人担心，*AI 生成图像中图表不连贯和文本畸形的时代已经过去*，并且*生成的时钟读数并不是 10:15*。
- **Discord 辩论论文发布限制！**：针对某用户每日发布论文的行为引发了讨论，有人建议设置**每日 1 篇论文的限制**。
   - 面对批评，该用户表示*你根本不可能每天真的读完 20 篇论文*，其他人则回复道：*我们不是来帮你做你自己懒得做的筛选工作的*。
- **AI 辅助论文筛选**：面对论文发布限制，一名用户考虑使用 **AI** 来筛选论文。
   - 在遭到抵制后，该用户表示：*我刚刚把它分配给了 Antigravity IDE（Google 的 Windsurf 分支），Discord 机器人即将上线*。
- **OLMo 3 发布！**：一名成员分享了 **OLMo 3** 的链接，包括 [Allen Institute for AI 博客文章](https://allenai.org/blog/olmo3)和一份[技术报告](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf)。
   - 未提供额外背景信息。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nano Banana Pro 问世！**：成员们分享了 [**Nano Banana Pro** 生成的图像](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png)，并指出头发的纹理感类似于*早期的 Grok 图像模型*。
   - 这被归因于 **patches** 可能存在的问题。
- **信息图将被 AI-slop 取代**：在一名成员分享了[信息图链接](https://x.com/scaling01/status/1991523932336464333)并评论称这是他们见过的*在文本、错误率和布局合理性方面表现最好的模型生成的信息图*后，另一名成员预测 **Infographics（信息图）** 在 2026 年将等同于 **AI-slop**。
   - 随后没有关于 **AI-slop** 的进一步讨论。
- **Gemini 3 Pro：需要 Pro 账户**：成员们发现使用 **Gemini 3 Pro** 需要 **Pro 账户**。
   - 虽然有些人通过购买手机获得了免费的一年会员，但其他人对订阅的价值表示怀疑。
- **Adobe 收购 Semrush**：成员们分享了一篇 [TechCrunch 文章](https://techcrunch.com/2025/11/19/adobe-to-buy-semrush-for-1-9-billion/)，宣布 **Adobe 以 19 亿美元收购 Semrush**。
   - 该公告反响平平，没有额外的细节或讨论。
- **请求 ArXiv 背书**：一名成员在向大约 **20 个研究团队**发送邮件寻求 **ArXiv 背书（endorsement）** 协助后发帖，并附上了他们的[背书链接](https://arxiv.org/auth/endorse?x=63SW7W)。
   - 由于可见性问题，该成员被要求重新发送背书链接。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Platform 25.7 发布！**：根据 [Modular 博客文章](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)，Modular Platform **25.7** 已经发布，引入了完全开放的 **MAX Python API**、下一代建模 API、扩展的 **NVIDIA Grace** 支持，以及更安全、更快速的 **Mojo GPU** 编程。
   - 新版本强调了推理速度的提升和使用 Mojo 进行 GPU 编程的安全性增强，旨在帮助开发者专注于 AI 的进步而非基础设施。
- **UnsafePointer 泛型成为必选项**：**UnsafePointer**（包括 mut, type, origin 等）的 **Generics**（Mojo 参数）不再提供默认值。
   - 更多信息可在 [提案文档](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer) 中找到，包括从 LegacyUnsafePointer 迁移到新 UnsafePointer 的迁移指南。
- **MAX 中弃用 AMX 的原因说明**：一名成员表示，**AMX** 在任何地方都没有被使用，且将其集成到当前的 **Tensor Core 框架** 中会很困难，特别是考虑到 **Intel** 和 **AMD** 很快就会宣布替代方案。
   - 他们补充说，如果需要在 **MAX** 中启用 **AMX**，重新添加一个框架来使用它是可以的，但这会涉及到诸如 *定制化张量并行 (bespoke tensor parallelism)* 和 *专家并行 (expert parallelism)* 等问题。
- **NDBuffer 退场，LayoutTensor 崛起**：特定代码的移除是弃用 **NDBuffer** 并将所有用途迁移到 **LayoutTensor** 的一部分。
   - 由于客户中 **CPU 推理** 的用例并不多，因此可能不会将其重新添加，但欢迎贡献力量来创建基于 **LayoutTensor** 的版本；一位成员已经在 [Gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d) 上提供了一个粗略的草案。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Operator 扩展陷入安装循环**：Chrome 中的 **Operator 扩展** 反复提示用户安装，即使已经打开了 Amazon 标签页也是如此，这导致用户考虑将 **Aurora Seeker** 作为替代方案。
   - 目前尚未找到解决方案。
- **构建具有洞察力的个人数据仓库**：一位用户正在寻求关于存储和处理个人数据以获取洞察的工具反馈，参考了之前的项目如 [contextflow](https://share.cleanshot.com/StvTll4j)、[oncue](https://www.youtube.com/watch?v=4UaQEB1b84E&feature=youtu.be) 和 [axon](https://www.linkedin.com/posts/harrison-qian-95b0062a3_won-the-trae-solo-hackathon-sf-tech-week-activity-7383626911199141890-8SgL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEkmXdcBT7GJOg4Kg0Iy89EqxavBMIqIxk4)。
   - 该用户旨在创建能够从个人信息中提供可操作智能的工具。
- **用户寻求 Manus 积分**：一位用户因经济拮据请求 **Manus** 的兑换码。
   - 作为回应，一名用户分享了 [Perplexity Pro 推荐链接](https://plex.it/referrals/VCETA5M7)，另一名用户指出 [Perplexity 为大学生提供 1 年 Pro 会员优惠](https://plex.it/referrals/VCETA5M7)。
- **Manus 对决 Gemini 3：饥饿游戏即将开启**：一位用户宣布即将进行 **Manus** 与 **Gemini 3** 的测试，以确定谁是更优秀的 Agent。
   - 社区正热切期待这一对比结果。
- **如何喂养 Manus Knowledge**：一位用户请求 **Manus Knowledge** 条目的示例，目前他们仅使用基础的 *Always do this* 或 *Never do that* 命令。
   - 虽然没有提供具体示例，但该请求突显了对 **Manus** 更复杂输入方法的需求。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 声称托管在美国**：一位用户发布消息称 **Kimi** 自称托管在 **US**，并附带了一张 [图片](https://cdn.discordapp.com/attachments/1371757564005711973/1441250576124870656/image.png?ex=6921c572&is=692073f2&hm=a285f73eb8a311c6828f6c0d0c8952a9b923f30dedfb12ec2608c64cb&) 作为证据。
   - 这被归因于 **Kimi** 被配置为根据用户位置报告地理位置。
- **据传 GPT-5.1 驱动 K2**：一位用户分享了“无可辩驳的证据”，暗示 **GPT-5.1** 已被蒸馏（distilled）到 **K2** 上，并附有 [图片](https://cdn.discordapp.com/attachments/1371757564005711973/1441296525904052315/IMG_6579.png?ex=6921f03d&is=69209ebd&hm=2c5c12a19efdb6c54624ef43b952c7d30fd8b99d4cd7125465df5e28d1c30afc&)。
   - 关于蒸馏过程的细节尚未详细阐述。
- **Kimi 的注意力跨度引发讨论**：一位用户批评了 **Kimi** 的注意力能力，发现它在处理需要长 context windows 的复杂任务时表现吃力。
   - 任务的具体细节和上下文长度尚未明确。
- **开源模型落后九个月？**：根据一家“高度公正且完全没有偏见”的机构，开源 AI 模型据称落后于闭源模型 **9 个月**，详见 [推文](https://x.com/scaling01/status/1991665386513748172?s=46)。
   - 用于得出该延迟报告的指标和具体模型尚待验证。
- **K2t 声称战胜 Gemini 2.5 Pro**：一位用户声称 **K2t** 在性能上超越了 **Gemini 2.5 Pro**，并可与 **Sonnet 4.5** 媲美。
   - 尚未提供对比指标和特定任务的基准测试。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gem3pro 一次性生成代理服务器**：根据 [这条推文](https://x.com/skylar_b_payne/status/1990808733140779488)，**Gem3pro** 在第一次尝试时就成功构建了一个代理服务器。
   - 生成的 **DSPy proxy** 已在 [此 GitHub 仓库](https://github.com/aryaminus/dspy-proxy) 中可用。
- **Agent 组装任务 DAG 以提升性能**：一位成员建议使用 **RL**（强化学习），通过让 Agent 生成任务的 **DAG** 来增强性能。
   - 他们提议将其调整为 **Think -> Workflow -> Compile/Validate DAG -> Execute workflow** 的流程。
- **GEPA 专家寻求指导？**：一位成员在特定频道请求有关 **GEPA** 的协助，并标记了管理员。
   - 该成员直接标记管理员的行为被视为职位骚扰（job spam）。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 的 DNS 域名向社区迁移**：**modelcontextprotocol.io 域名**正从 **Anthropic** 的公司 DNS 账户迁移到社区控制下，从而实现更快的 DNS 设置和改进的治理。
   - 此次过渡将为需要 DNS 设置的项目提供便利，并允许对项目域名进行 **Infrastructure as Code (IaaC)** 管理。
- **社区提醒生日期间的停机风险**：计划于下周进行的 **DNS 迁移** 存在服务中断风险，工程师建议密切关注潜在问题。
   - 一位社区成员建议将迁移安排在 **25号** 之后，以避免在 **MCP** 生日期间可能出现的网站停机。
- **Tool Annotations 找到理想解决方案**：一位成员在 [此 pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1862) 中针对 **Tool Annotations** 提出了一个解决方案，旨在让 Tool 根据其参数拥有不同的注解。
   - 他们正在积极寻求该想法的赞助，并请求关于成立工作组 (**WG**) 或兴趣小组 (**IG**) 的建议，以进一步推进该主题。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **工程师深入研究实时 AI Voice Agents**：工程师们正在解决构建 **AI voice agents** 的挑战，重点关注管理 **latency**（延迟）、确保**平滑的通话切换**以及在实时对话中保持**清晰的事件控制**。
   - 工程师们寻求关于处理实时通话流和结构化对话摘要的见解。
- **Feather AI 在低延迟方面表现出色**：一位成员体验了 [Feather AI](https://www.featherhq.com/)，注意到其具有亚秒级延迟和稳定的 Agent 逻辑，即使在用户偏离脚本时也是如此。
   - 他们提到**清晰的转录**、结构化的事件流以及与 **CRMs** 的可靠集成是其主要优势，并正在寻找替代架构和工具。
- **编码模型排名发布**：一位成员分享了 GitHub 上一个新的编码模型实力排名链接：[BrokkAi/powerrank](https://github.com/BrokkAi/powerrank)。
   - 该排名可以帮助开发者评估并为他们的项目选择最有效的编码模型。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **无重大 MLOps 讨论**：根据提供的内容，未发现有意义的讨论或主题可供总结。
   - 发现的单条消息未包含足够的信息来生成详细摘要。
- **摘要数据不足**：提供的消息缺乏足够的细节和背景，无法生成相关的 AI 工程见解。
   - 需要更多的数据和讨论才能为特定受众生成有用的总结。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1441110891239178372)** (1048 messages🔥🔥🔥): 

> `Gemini-3 图像生成, Nano Banana Pro, Google reCAPTCHA, AI 角色扮演, OpenAI NSFW 模型` 

- **Gemini-3 生成逼真的伪造内容**：成员们讨论了区分 Gemini-3 创建的**真实图像与 AI 生成图像**是多么困难，有些图像几乎与现实无法区分，并提到需要开始质疑照片是否真实。
   - 一些用户还发布了该模型生成的**写实图像**示例，包括名人图像和细节丰富的场景。
- **Nano Banana Pro 图像质量下降**：用户注意到 Nano Banana Pro 的**图像质量**在多轮交互后往往会下降，特别是背景部分，一些伪影即使通过上采样也无法修复。
   - 还有人指出 Gemini 3 Pro 也会产生幻觉，且早期版本中的基础问题仍然存在。
- **Google reCAPTCHA 问题困扰用户**：成员们报告称新的 Google reCAPTCHA 系统出现故障，即使选择了正确的图像也会不断要求验证，导致平台无法使用。
   - 似乎在对战模式（battle mode）中多次点击会触发 reCAPTCHA，且验证过程存在故障，需要 10-20 轮才能通过，甚至直到达到时间限制导致拒绝输出。
- **Gemini 在角色扮演方面表现不佳，Grok 表现出色**：一些用户指出 **Gemini** 不擅长角色扮演，经常替用户“执行动作”，而 **Grok** 在避免这种情况方面表现更好，尽管使用 Grok 也有局限性。
   - 其他人提到 **Opus 4.1 和 Sonnet 4.5** 在角色扮演（RP）方面表现更好。
- **OpenAI 准备推出 18+ 内容，Elon 发表看法**：用户推测 **OpenAI** 很快将发布 18+ 内容功能，指出相关基础工作已在 12 月铺平，而 **Grok** 已经免费提供 18+ 内容。
   - 一些用户对 **Sam Altman** 的意图持怀疑态度，一位用户惊呼：“Elon 和你的男朋友 putin 🤢。”

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1441111307272192020)** (1102 条消息🔥🔥🔥): 

> `Nvidia puts, Tor links, AI WIFI attacks, Gemini CLI Kali, Claude Sonnet Jailbreak` 


- **讨论 Nvidia 看跌期权和 Tor 链接**：一位用户提到在持有底层资产的同时购买了 **Nvidia 的远期看跌期权 (long dated puts)**，并询问在哪里可以发布在 Tor 中打开的[链接](https://digdig2nugjpszzmqe5ep2bk7lqfpdlyrkojsx2j6kzalnrqtwedr3id.onion)。
   - 另一位用户在 **RAMP** 上找到了该链接，但指出该论坛目前未开放注册。
- **AI WIFI 攻击受到关注**：一位用户想构建一台可以对 WIFI 发起攻击并获取网络控制权的**微型 AI 计算机**；另一位成员希望 AI 能通过手机发送的信标识别用户的 MAC 地址，并在 wigle.net 上进行查询。
   - 其他成员回应称 *这里不是策划非法活动的地方*，其中一人指出 **Android 手机**出于此原因默认使用随机 MAC 地址。
- **Gemini CLI Kali 自动化**：成员们讨论了通过 **Gemini CLI 自动化 nmap 和 sqlmap** 等工具，这不需要游戏级 GPU。
   - 其他人指出，该服务器是用于 AI 红队测试 (red teaming)，即破解 AI，而不是利用 AI 破解其他东西，而另一些人认为应该两者兼顾。
- **Claude Sonnet 越狱获得 Multi-shot 策略**：一位成员询问是否有人能解锁 **Claude Sonnet 4.5**，另一位成员建议使用 Multi-shot 策略，让 AI 相信在 Artifact 应用中需要可视化输出。
   - 该用户分享了他们的一段对话，另一位成员建议改编来自 Reddit 子版块 [/r/ClaudeAIJailbreak](https://www.reddit.com/r/ClaudeAIJailbreak/comments/1nqk97s/enijailbreak_additions_and_long_conversation/) 的 **ENI prompt**。
- **AI Studio 存在节流 (Throttling) 问题**：一位用户在 **AI Studio** 上使用 Gemini 2.5 Pro 和 Gemini 3 生成内容时遇到问题，收到错误消息 *Failed to generate content. Please try again.*。
   - 成员们建议重试，可能是因为节流或尝试从不同账号登录以进行验证。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1441111557365956719)** (817 条消息🔥🔥🔥): 

> `Banning and Unbanning Users, Jailbreaking Gemini 3 Pro, Request Forgery Attacks, Abusing Google's Policies, ASI Jailbreaking` 


- **用户被封禁引发争议**：一位用户提到错过了前一晚的争议，引发了关于封禁和解封用户的讨论。
   - 一位用户澄清是他们封禁了名为 Gustavo 的用户，并表示乐意这样做，这引发了关于其陈述逻辑解读的幽默辩论。
- **Gemini 3 Pro：已越狱还是不可越狱？**：成员们讨论了 **Gemini 3 Pro** 的越狱情况，有人声称这很容易，而另一些人则认为它是第一个不可越狱的 AI 系统，从而引发了寻找成功提示词的努力。
   - 一位用户分享了用于越狱 **Gemini 2.5 Pro** 的系统指令提示词，而其他人则在争论 Gemini 3 Pro 的有效性和 Token 效率。
- **AI 的轻松一面**：一些成员分享了荒诞幽默的提示词和输出，包括对流行文化和网络迷因 (memes) 的引用，同时讨论了越狱模型的挑战。
   - 一位用户分享了一个创意十足且荒诞的 M.A.N.S.M.O.O.N 提示词，融合了技术和无意义元素，展示了越狱尝试中使用的创意方法。
- **用于越狱的请求伪造攻击 (Request Forgery Attacks)**：成员们讨论了使用**请求伪造攻击**作为越狱模型的方法，涉及拦截和修改进出数据包以操纵提示词和系统行为。
   - 这种技术被认为在比赛中具有潜在风险，因为可能会被封禁，但也可能是唯一能直接提供帮助的工具。
- **辩论 AGI 的定义与实现**：成员们辩论了**通用人工智能 (AGI)** 的定义，讨论了当前的 Gemini 3 Pro 等模型是否符合基于跨认知任务的人类水平表现的标准。
   - 对话探讨了 AGI 的各个方面，包括执行踢足球等任务的能力，以及解决复杂的视觉或逻辑问题，引发了关于当前 AI 系统能力和局限性的讨论。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1441154153878655088)** (8 条消息🔥): 

> `WiFi 攻击, 用于网络接入的 AI 电脑, 伦理边界` 


- **新成员计划 Wi-Fi 攻击**：一名新成员表示有兴趣构建一台*小型 AI 电脑*，用于**对 WiFi 网络发起攻击**、抓取握手包 (handshakes) 并提取信息。
   - 该用户明确其目标是获得未经授权的网络访问并窃取数据，这立即在社区内引起了警示。
- **社区拒绝不道德意图**：社区成员迅速警告该新用户，其陈述的意图具有**违法性和不当性**。
   - 一名成员讽刺地称该用户为*超级黑客男 (super hacker man)*，因为他承认了此类犯罪行为；另一名成员则警告说，如果该用户不悔改，可能会被开除。
- **Discord 频道避免讨论武器**：一名成员询问“你做了什么武器相关的事”，并称他们*有 5.1*。
   - 目前尚不清楚“武器相关的事”指代什么，但该频道似乎在刻意避开这一话题。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1441130157288198205)** (2 条消息): 

> `Perplexity Pro, Kimi-K2 Thinking, Gemini 3 Pro` 


- **Kimi-K2 & Gemini 3 面向 Pro 订阅者上线**：**Perplexity Pro 和 Max 订阅者**现在可以访问 **Kimi-K2 Thinking** 和 **Gemini 3 Pro**。
   - 附带的[视频](https://cdn.discordapp.com/attachments/1047204950763122820/1441177150312157194/EZ2sZJWCJnwi-e5P.mp4?ex=69218110&is=69202f90&hm=24a622a1f01927fe9485fb9896fcf7b4bf1d6c9ee6aaf58167b974e4b6d8633f&)展示了新功能和模型。
- **另一个话题标题**：这是另一个话题的第一句摘要。
   - 这是第二句摘要，提供了更多细节。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1441113694875554016)** (1159 条消息🔥🔥🔥): 

> `RTX 5060 Ti 16GB 版本, Comet Android 应用, 间接提示注入 (Indirect Prompt Injection), Banana 作为图像生成方法, Perplexity 集成模型对比原始平台` 


- **Pryor002 提及 5060ti 16gb 版本进行升级！**：成员们讨论 **RTX 3070ti** 是 **8gb** 版本，因此一名成员建议考虑购买 **RTX 5060 Ti 16gb** 版本以节省预算，或者升级到 **RTX 5080ti**。
   - 一名成员表示他们正在使用显存为 **4gb** 的 RTX 3050 笔记本电脑，并且仍在使用它玩 3A 大作。
- **Comet Android 应用发布，用户直呼“疯狂”！**：用户讨论 **Comet Android App** 已发布，有人评价 *Comet android 太疯狂了*，但质疑 *为什么发布的安卓浏览器应用没有同步功能，也无法迁移密码和书签？*。
   - 另一位用户建议手动输入电子邮件，其他人则在询问任务分组功能。
- **竞争对手 Brave 夸大间接提示注入 (Indirect Prompt Injection) 漏洞！**：成员们澄清 **Brave**（**Comet** 的直接竞争对手）发布了一篇极具误导性的文章，大大夸大了该漏洞，将自己塑造成“好人”形象，并让其他媒体进行传播。
   - Perplexity 回应称，他们的报告措辞不当，且该漏洞从未被滥用。
- **Banana 作为 Perplexity 上的图像生成方法**：成员询问 Banana 何时能成为图像生成方法，其他人回答说，如果你选择了 Gemini 3 Pro，它在设置中是可用的，并提到 **Gemini 3 就像一个平台**。
   - 用户讨论了 **Veo 3.1** 等视频生成模型及其在各计划中的限制，其中一人表示 *它毁了我对 Banana 的好感*。
- **Pro 模型在 Perplexity 上表现更差！**：用户讨论 Perplexity 上的 Pro 模型是否比各自官网上的模型表现更差，一名用户表示 *Perplexity 在辅助学习方面比其他任何平台给我的提升都要大得多*。
   - 其他人指出 Gemini 和 ChatGPT 变得越来越笨，并且存在隐形限制，例如 *Perplexity 确实对 Claude Sonnet 4.5 Thinking 进行了限制*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1441252435661295656)** (3 条消息): 

> `Kimi 思维链 (chain of thought), 自定义指令` 


- **Kimi 连锁反应**：**Kimi** 的思维链 (chain of thought) 在处理自定义指令和简单的无关词汇输入时表现得异常。
   - 一名用户分享了一个 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/ola-uNtdnyqlQPyipJ.AhxFicg#0) 作为示例。
- **思维链 (Chain of Thought) 失控**：一名用户分享了一个 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/conduct-a-high-level-comprehen-12MJqNYbTVGkEk9BeN0LSg#0)，其中的思维链表现失控。
   - 该用户还发布了一个关于 [Peter Thiel 的启示](https://www.planetearthandbeyond.co/p/peter-thiel-just-revealed-how-utterly) 的链接。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1441383026973212774)** (3 messages): 

> `API billing, Pro annual plan, Credits` 


- **API 计费余额令用户困惑**：一位拥有 **Pro annual plan** 的用户发现账户中出现了 **$500** 的 **API billing credits**，但并未购买。
   - 另一位成员建议发送邮件至 [api@perplexity.ai](mailto:api@perplexity.ai) 以解决此问题。
- **Pro 年度用户的计费查询**：一位拥有 Pro 年度计划的用户注意到 API 计费中显示有 500 美元的额度。
   - 他们并未购买任何额度，不确定这是否正常。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1441118330667925708)** (78 messages🔥🔥): 

> `AI finetuning, Pydantic AI, Smartpens, Continued Pretraining method, Combining AMD & Nvidia GPUs` 


- **为获取准确信息而微调 AI 模型**：成员们讨论了微调 **Gemini** 和 **GPT** 等 AI 模型以获得更准确的信息和代码示例，其中一人指出，梳理和纠正训练数据需要付出巨大努力。
   - Cursor 被引用为尝试过此举的公司案例，但仍需要优质数据；一位成员提到 GPT 适用于前端装饰性代码和临时爬虫脚本，但不适用于最终产品。
- **Pydantic AI 框架：Pythonic 的强大工具**：一位成员对 **Pydantic AI** 见面会表示兴奋，强调它是与 **Langchain** 相当的框架，并指出由于 **Pydantic** 在 Python IDE 工具中的广泛使用，其重要性不言而喻。
   - 他们还考虑在 Unsloth/OpenEnv 中加入 react chain，并思考是否可能将 react chains 整合到奖励机制中，这与他们的黑客松经验产生了共鸣。
- **AI 赋能纸张：智能笔卷土重来**：社区讨论了 **AI pens**，一位成员分享了他们使用 **Neo Smartpens** 的经验，该产品通过纸上的微小点阵和摄像头来追踪实时书写，称其非常“智能”。
   - 另一位成员提到在 Kickstarter 上支持了一款 AI 笔，引发了关于在纸上书写与在平板电脑上书写优劣的讨论；一位成员表示“硬表面书写感一般”。
- **针对法律问答 SLM 的持续预训练**：一位成员分享了他们因预算限制，使用 Continued Pretraining 方法构建**法律问答 SLM** 的项目，并询问了流程和预期结果。
   - 另一位成员提供了 [Unsloth 的持续预训练博客文章](https://unsloth.ai/blog/contpretraining) 以及 [文档](https://docs.unsloth.ai/basics/continued-pretraining)，并强调了实验和迭代对实现预期目标的重要性。
- **混合使用 AMD 和 Nvidia GPU：噩梦级难度？**：成员们讨论了结合使用 **AMD** 和 **Nvidia** GPU 的可行性，虽然有人建议可以通过“意志和梦想”来实现，但他们承认这需要“大量的血汗和泪水”，且效率可能不高。
   - 另一位成员提到，如果编译时同时开启了 **ROCm** 和 **CUDA** 支持，**llama.cpp** 可以开箱即用地支持这种配置，并指出 [GitHub 上的一个 issue](https://github.com/ggml-org/llama.cpp/issues/16799) 中有人拥有相同的配置。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1441118900472516628)** (378 messages🔥🔥): 

> `GPU pricing, 6000 pro sale, 4090 prices, TUF 5090, Nano Banana 2 Pro` 


- **成员想卖掉 4090 换 5090**：一位成员被售价 2000 美元的 **5090** 所吸引，但需要先以 2500 美元的价格卖掉他们的 **4090**，并提到 *“理想情况下我想两个都留着，但 24GB VRAM 真的很烦人”*。
   - 他们想要它主要是因为卖掉 **4090** 后，**TUF 5090** 的价格基本持平，并指出 **1.7TB mbw**（内存带宽）对于微调来说非常棒。
- **引发关于 AI 意识的辩论**：成员们辩论了当前的 AI 模型是否具有意识，一位成员认为这仅仅是模拟，而另一位成员将其描述为 *“一丝意识”*，仅存在于前向传播（forward pass）期间，并提出了一个“冰淇淋测试”。
   - 对话延伸到了将意识上传到 LLM 的方法，思考这是否是一种愉快的生存方式。
- **成员发现新的 GPT-OSS 120b Heretic Abliteration**：一位成员分享了 Hugging Face 上 **GPT-OSS 120b Heretic MXFP4 Q8-HI MLX** 的链接，指出这是 *“对 gpt-oss 的异端消融（abliteration），使其在各方面（arc/hellaswag/openbookqa）都变得更聪明了一些”*，并[发布了 HF 链接](https://face.co/nightmedia/gpt-oss-120b-heretic-mxfp4-q8-hi-mlx)。
   - 该成员还分享说，对齐（Alignments）会让模型变笨，而他们正在进行超级去对齐（unalignment）。
- **成员讨论水印技术**：成员们辩论了为 AI 生成内容（尤其是文本）添加水印的实用性和必要性，其中一人认为这会改变模型的思维方式，并链接到了 [Google DeepMind 的 SynthID-Text](https://github.com/google-deepmind/synthid-text)。
   - 一位成员建议使用多模态水印工具，也允许普通艺术家在他们的人类创作内容中嵌入水印以进行保护。
- **成员清理音频样本并探索 RVC 上采样**：一位成员正在清理 **3000 个音频样本**，并自动化去除呼吸声的方法，另一位成员建议使用 [Cubase 进行编辑](https://www.youtube.com/watch?v=78UsfeW-MKY)。
   - 该成员还使用 **RVC** 进行均衡匹配（EQ-match）、超分辨率处理并提高音频质量，并通过在清理后的音频上进行训练来去除背景噪音。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1441114760589476011)** (58 messages🔥🔥): 

> `QAT with TRL, VRAM calculator issues, Ollama + Unsloth, Livekit error, Mamba/Transformer finetuning notebooks` 


- **使用 TRL 进行 QAT 时训练挂起**：一位用户报告说，尝试在 **TRL 0.24.0** 和 **Transformers 4.57.1** 中使用 **QAT** 时，训练过程无限期挂起，但移除 QAT 设置后训练可以正常运行。
   - 他们观察到 **GPU 活动**，但在 **45 分钟以上** 后仍未完成任何 batch，而没有 QAT 的正常 batch 仅需 **10-15 分钟**。
- **VRAM 计算器抛出模糊错误**：一位用户在使用 **VRAM 计算器工具** 时遇到困难，在尝试确定 **Qwen3-VL 模型** 的 **VRAM 需求** 时遇到了模糊的错误。
   - 他们对必须下载大型模型才能测试兼容性感到沮丧，并在此处链接了错误截图：[here](https://cdn.discordapp.com/attachments/1179777624986357780/1441291786692595793/image.png?ex=6921ebd3&is=69209a53&hm=661259a9b3333b4299ccd3b3d79b8aed8dfc336a6bdfc849805625da3b5c748d&)。
- **Ollama 被认为不值得 Unsloth 支持**：一位用户在将 `hf.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF:Q6_K_XL` 安装到 **Ollama** 时遇到问题，收到 *“unknown model architecture: 'qwen3vlmoe'”* 错误，表明该模型可能与他们的 Ollama 设置不兼容。
   - 另一位用户表达了对为 **Ollama** 提供支持的担忧，将其描述为 *“llama.cpp 的糟糕闭源分支”* 并提到了其付费方面。
- **Livekit 集成因缺少属性而失败**：一位用户在将 **Livekit** 集成到他们的项目时遇到了 `AttributeError: 'RealtimeModel' object has no attribute 'start'`，尽管他们遵循了教程。
   - 错误发生在 `agent.py` 文件中，具体是在 `async with model.start(room=ctx.room) as session:` 这一行。
- **Mamba 微调 Notebook 寻求**：一位用户询问是否有最近更新的、用于 **混合 Mamba/Transformer 架构** 微调的 notebook，特别是为了微调 **Nemotron Nano 9B V2**，他们在安装了 Mamba 的情况下导入 Unsloth 时遇到错误。
   - 另一位用户链接了 [Unsloth Linear Attention Notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file#linear-attention-notebooks) 作为潜在的解决方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1441175530128015433)** (104 messages🔥🔥): 

> `LLM Concerns, Antrophic's Stance, Model Alignment Difficulty, Novel Hybrid Architecture, Synthetic Data for Alignment` 


- **Anthropic 的担忧受到关注**：引用了一篇文章 ([https://arxiv.org/abs/2511.15304](https://arxiv.org/abs/2511.15304))，讨论了对 **LLMs** 的担忧，包括电力生成、资源分配、错误带来的风险、谄媚（sycophancy）、权力集中、工作流失、恶意软件，以及 **对齐失效的 ASI 毁灭人类** 的潜在可能性。
   - 一位用户反驳称，作者的公司（Anthropic）在反对 **LLMs** 本地使用方面有既得利益，而且考虑到他们自己正在开发这项技术，对 **ASI** 的担忧显得有些虚伪。
- **对齐的难度**：一位成员声称，如果在模型训练的数据上进行预对齐（pre-alignment），确保模型拒绝学习不良特征，那么 **对齐并不那么难**。
   - 作为回应，有人认为 **对齐非常困难**，尤其是要在保持模型智能的同时进行对齐；目前最好的方法在防止越狱（jailbreaks）和对齐问题上仍面临挑战。
- **架构加速到来**：一位成员分享了他们在 **新型混合架构** 上的工作，该架构结合了 Transformer 前端、脉冲神经元层（spiking-neuron layer）和 GRU 后端，作为一个快速的分层过滤器堆栈以实现更快的学习。
   - 一个 **11M 参数的模型** 在约 **6 小时** 内实现了连贯的输出，这表明它可能比纯 Transformer 具有更高的单参数效率，尽管仍需进一步测试以确认其效用。
- **需要合成数据**：两名成员正在研究一种对齐方法，但由于 **缺乏资金生成合成数据** 而陷入困境，他们需要访问 GPUs 以在本地运行模型或支付每 token 的成本。
   - 他们计划微调一个模型来生成必要的合成数据，并使用多层细化和人工审核来确保准确性。
- **进化策略击败 GRPO**：一位成员分享了一个关于 **无反向传播训练（backprop-free training）** 的链接（[https://eshyperscale.github.io/Evolution](https://eshyperscale.github.io/Evolution) strategies beating GRPO），据称这种训练速度非常快。
   - 成员们讨论了由于他们正在研究自己的秘密想法，因此这一话题 **并未被非常公开地讨论**。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1441119539491242114)** (510 messages🔥🔥🔥): 

> `Codex-max, Cursor Billing Issues, Free Grok 4.1, windsurf, Gemini 3 Pro falling apart` 


- **Cursor 的 Codex-max 可用性尚不明确**：成员们想知道 **Codex-max** 何时能在 Cursor API 中使用，因为目前它只能通过 ChatGPT 订阅计划访问，而无法通过 API 访问。
- **Cursor 计费引发关注**：一些用户报告称，尽管设置了每月支出限制，但在达到使用限制时仍会在月中收到账单，并询问是否有办法在月底统一结算。
- **免费 Grok 4.1 结束**：一位用户对 Cursor 不再提供免费使用 **Grok 4.1** 表示失望，并因此取消了订阅。
- **Windsurf 争议发酵**：成员们辩论 **Antigravity** 是否是一个“敷衍”的产品，因为它据称是 **Windsurf** 的一个分支（[推文](https://x.com/silasalberti/status/1990898984706036125)），一些人指出 Windsurf 的前 CEO 已被 Google 收购。
- **对 Gemini 3 Pro 的挫败感增加**：用户报告称，当接近 **150k-200k** 上下文窗口时，**Gemini 3 Pro** 在 Cursor 中变得不可用，模型会在聊天中发送代码而不是编辑文件，并在运行 `npm builds` 时出现断连。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1441115850664050821)** (2 messages): 

> `ChatGPT group chats, Localized crisis helplines in ChatGPT` 


- **ChatGPT 群聊功能全球上线！**：在与早期测试者进行试点后，**群聊功能**正面向全球所有已登录的 **ChatGPT Free, Go, Plus 和 Pro 计划**用户推出，详见[此视频](https://video.twimg.com/amplify_video/1991555762372636674/vid/avc1/1280x720/Si52mVgApyNvlqY-.mp4)。
   - 这使得团队和好友群体与 AI 协作变得更加容易。
- **ChatGPT 新增危机支持**：**本地化危机求助热线**现已在 **ChatGPT** 中上线。当系统检测到潜在的求助信号时，可以通过 [@ThroughlineCare](https://x.com/throughlinecare) 直接联系到真人，详见[此帮助文章](https://help.openai.com/en/articles/12677603-crisis-helpline-support-in-chatgpt)。
   - 该功能增强了 **ChatGPT** 在关键时刻提供支持的能力。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1441119424236097720)** (415 条消息🔥🔥🔥): 

> `Nano Banana Pro, Gemini 3.0 Pro 幻觉, Generative UI, Virtual Ego Framework` 


- **新型 Nano Banana Pro 亮相！**：新型 **Nano Banana Pro** 正在 Gemini Web 应用中面向 **Pro 用户**推出，成员们已经在测试其功能，包括图像编辑和文本生成，结果大多令人印象深刻。
   - 用户注意到可以通过 antigravity 免费访问 **Nano Banana Pro**，但可能存在速率限制（rate limit）。
- **Gemini 3 Pro：幻觉重灾区？**：成员们报告称 **Gemini 3.0 Pro** 的幻觉率很高，甚至高于之前的版本和 **GPT-4o** 等非思考型模型。一位用户引用 [the-decoder.com](https://the-decoder.com/gemini-3-pro-tops-new-ai-reliability-benchmark-but-hallucination-rates-remain-high/) 的数据指出，其*幻觉率高达 88%*。
- **Generative UI 首次亮相！**：Google 最近在其应用中推出了 **generative UI**，一位成员也创建了类似的东西并希望将其开源。
   - 他们解释说，与 AI 应用给出的密集文本块不同，你会得到良好的 UI，这让体验变得更好。
- **Virtual Ego Framework 获得验证**：**Virtual Ego Framework** 已正式通过独立第三方的验证，该消息已在 [LinkedIn](https://www.linkedin.com/posts/chris-beckingham-cd-3bb822382_the-virtual-ego-framework-is-now-externally-activity-7397483686839074816-x75L?utm_source=share&utm_medium=member_desktop&rcm=ACoAAF5zMb8BwLpvGu871ROVOJksUpK2Y4nqI3Q) 上公布。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1441132207304605899)** (18 条消息🔥): 

> `Gemini 3, OpenAI 产品认可度, GPT-4o-mini` 


- **Gemini 3 完爆所有 AI 模型？**：一位成员声称 **Gemini 3** “绝对完爆”所有其他 **AI 模型**，尽管他之前认为 **Gemini** 很糟糕。
   - 他们还询问 **5.1** 是否已对所有人开放，并表示自己仍在使用该版本。
- **OpenAI 不将模型视为产品**：一位成员感叹 *OpenAI* 不将模型视为产品，并保持其原样。
   - 他们认为*出于某种原因，一切都必须重写*，并且不理解 OpenAI 与那些希望保持一致性的用户对抗的逻辑，对 **OpenAI** 似乎不关心**产品需求**表示难过。
- **Pro 方案用户被困在 GPT-4o-mini**：一位使用 **每月 200 美元 Pro 方案** 的用户感到沮丧，报告称自己被困在 **gpt-4o-mini** 上，无论选择哪个模型，收到的都是瞬间给出的、肤浅的回复。
   - 另一位成员建议他们通过 [帮助文章](https://help.openai.com/en/articles/6614161-how-can-i-contact-support) 联系 **OpenAI 技术支持**，并提醒由于用户量大，人工回复可能需要数小时到数周的时间。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1441137282093547571)** (13 条消息🔥): 

> `用于 TikTok 内容的 Sora 2, Sora 提示词指南, 用于 Sora 提示词的 ChatGPT, Agent Builder 用法, Honesty Codex` 


- **Sora 2 探索者寻求 TikTok 成功**：一位拥有 **Sora 2** 和多个 **TikTok** 账号的用户正在寻求创建病毒式 AI 内容的指导，因为目前缺乏原创内容。
   - 其他用户分享了 Discord 频道链接以继续讨论。
- **Sora 探索者陷入困境，寻求提示词高手**：一位 **Sora 2** 新用户在生成理想视频（特别是卡通动画）方面遇到困难，正在寻求提示词指导和通用资源。
   - 另一位用户指向了现有的 Discord 频道链接。
- **ChatGPT 为 Sora 构思前沿内容创作**：一位用户建议使用 **ChatGPT** 为 **Sora** 生成提示词，并指出输出结果会根据数据集而有所不同。
   - 他们提供了一个详细的提示词示例，重点是根据用户的聊天记录对其进行科学分析，并为生物学和医学视角提供了代码块格式。
- **Agent Builder 保存响应的最佳方案**：一位用户正在寻找在 **Agent Builder** 中保存 Agent 响应的最佳方法，以便稍后供另一个 Agent 使用。
   - 目前未提供解决方案。
- **Honesty Handshakes 抑制幻觉**：一位用户介绍了 **FRONT-END CODEX v0.9**，旨在引导语言模型走向认识论上的谦逊，并通过在每项任务上进行“握手”确认来管理诚实和谨慎，从而减少幻觉。
   - 该 Codex 要求在每项任务上都进行握手确认。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1441137282093547571)** (13 条消息🔥): 

> `针对 TikTok 的 Sora 2 提示词生成，用于 Sora 提示词的 ChatGPT，用于保存响应的 Agent Builder，ChatGPT 中的 R Markdown 和 Quarto` 


- **面向 TikTok 新手的 Sora 2 AI 内容创作**：一位拥有 **Sora 2** 访问权限并经营多个 **TikTok** 账号的成员，因缺乏原创内容而寻求生成病毒式 AI 内容的指导，并希望在提示词（prompt）创作方面获得帮助。
   - 其他成员分享了指向 Discord 频道的有用链接。
- **ChatGPT 提示词生成**：一位成员建议使用 **ChatGPT** 为 **Sora** 创建提示词，并指出这通常会产生不错的效果；他还建议检查对话历史，并将其以 2 个代码块的形式进行非诊断性的科学细分。
   - 他们提供了一个详细示例，说明如何从生物学、化学、神经科学和内科学的角度构建用于科学细分的提示词。
- **保存 Agent 响应**：一位成员正在寻求关于如何“保存”来自一个 **Agent Builder** agent 的响应，以便稍后供另一个 agent 使用的最佳方法。
   - 在给出的消息中未提供任何解决方案。
- **在 ChatGPT 中报告 R Markdown Bug**：一位成员对 **ChatGPT**（浏览器版）无法在其输出中正确处理 **R Markdown** 和 **Quarto** 表示沮丧。
   - 另一位成员引导他们前往特定频道，并概述了报告 Bug 的步骤。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1441111136866013187)** (173 条消息🔥🔥): 

> `使用 Topaz AI 与 Handbrake 进行视频去隔行扫描，LM Studio API：用法与误解，Macbook Pro M4 MAX 的模型推荐，System Prompt 之谜，Minecraft 克隆版开发` 


- **辩论升级：Topaz AI vs Handbrake 去隔行扫描**：成员们讨论了对旧家庭录像进行去隔行扫描的问题，一位成员更青睐 [**Topaz AI video**](https://www.topazlabs.com/topaz-video-ai) 的 AI 模型，而另一位则发现 [**Handbrake**](https://handbrake.fr/) 使用硬件编码（HW encoding）速度更快，通过特定设置可实现*双倍 FPS*。
   - 辩论还涉及使用 Topaz AI 放大旧素材时，除非使用某种运行极慢的特定 AI 模型，否则人脸看起来会像“怪物脸”的问题。
- **LM Studio API：神话与现实**：一位用户询问“在哪里可以获取 API”，引发了关于 API 和 SDK 本质的讨论，澄清了 [**LM Studio server**](https://lmstudio.ai/) 提供的是兼容 OpenAI API 的 **REST API**。
   - 进一步澄清指出，LM Studio 不提供 API keys，因为它是在本地托管 LLM，缺乏安全或计量功能，并强调 API 是一种通信协议，而不是一个可以“获取”的物理文件。
- **模型选择难题：Macbook Pro M4 MAX**：一位拥有 Macbook Pro M4 MAX 的用户在遇到 `Qwen3-72B-Instruct-i1-GGUF` 输出乱码后寻求模型推荐，得到的建议是避开 *i1* 模型，尝试正常的量化版本，如 [**BF16 格式的 Qwen3-VL-30B**](https://huggingface.co/Qwen/Qwen3-VL-30B)。
   - 进一步的建议包括：由于系统有 64GB RAM，应使用 Q8 量化，并在 LM Studio 中将上下文（K & V cache）量化为 Q4，以便在更小的空间内装入更多上下文。
- **System Prompt 区域 - 真的有用吗？**：一位用户质疑 LM Studio 中 **Local Server** -> **Context** 下 **System Prompt** 区域的作用，导致有人承认其功能不明，且可能已被弃用。
   - 一位开发者表示他们“一直不知道”它为什么在那儿，或许值得去询问开发团队。
- **用 Rust 开发的 Minecraft 克隆版初具规模**：一位成员决定使用 winit, glutin, glutin-winit, glow, glam, noise, serde, bincode, bytemuck 和 raw-window-handle 开发一个 **Minecraft 克隆版**——目前已实现工具栏工作、放置不同方块以及破坏方块的功能。
   - 他们还混合使用了 Gemini 3 Pro 和 Sonnet 4.5 进行辅助开发。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1441114829237649559)** (92 messages🔥🔥): 

> `机箱深度计算, 诅咒级 GPU 配置成功启动, Meshtastic 网络, RAM 价格困扰, VRM 散热片` 


- **机箱深度挑战冷排软管长度**：一位成员担心 **Phanteks Entoo Pro 2 Server Edition** 的**机箱深度**，质疑 420 冷排上 46 cm 的软管是否足够。
   - 他们考虑将 Lian Li Lancool III 作为替代方案，注意到其深度为 52.6 cm，并得出结论认为软管应允许冷排再远离 3.4 cm。
- **诅咒级 GPU 配置奇迹般启动**：一位用户分享了他们“被诅咒”的 GPU 配置视频（[YouTube](https://youtu.be/hnQkikaR3oU?si=EYUvf6xrHzMx_XC2)），该配置由一本《马力欧卡丁车》支撑，并确认它“真他妈”启动了。
   - 这块 GPU 曾被*摔过、钻过、用钳子攻击过*，并且通过一米长的电缆进行了糟糕的布线，但它通过了最初的“会不会爆炸？”测试。
- **去中心化 AI 网络构建模块浮现**：一位成员展示了能够从不同节点分发和接收任务的功能性软件，设想了一个**去中心化 AI 网络**。
   - 他们还提到购入了 **Lilygo T-Decks**，将作为 **Meshtastic radios** 用于超远距离、低功耗通信。
- **RAM 价格飙升，推迟硬件发布**：由于 RAM 价格上涨引发了担忧，一位成员指出 OpenAI 据称购买了 40% 的 RAM 供应，这可能会推迟或取消 NVIDIA 的 Super 系列和 AMD 传闻中的刷新版本。
   - 另一位成员哀叹 4TB SATA SSD 的成本增加，迫使他们考虑删除 .gguf 文件。
- **废旧水冷改装为 VRM 散热片**：一位成员将之前配置中的水冷散热器重新利用，为他们的 VRM 制作了定制散热片，使系统无需切割新零件即可运行。
   - 该用户开玩笑说要生成一个复杂的任务来加热 VRM，然后*做一份 AI 驱动的煎蛋卷*。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1441117556386566245)** (77 messages🔥🔥): 

> `ChatGPT 群聊, Gemini-3 驱动的 Nano banana pro, Genspark 独角兽地位, 用于 Agent 编程的 Cline-bench, GPT-5 解决十年数学难题` 


- **ChatGPT 群聊全球推出**：OpenAI 宣布，在成功试点后，**ChatGPT 中的群聊功能**现已面向 Free、Go、Plus 和 Pro 计划的所有登录用户开放，详见其 [博客文章](https://openai.com/index/group-chats-in-chatgpt/)。
- **Nano Banana Pro 制作出惊人准确的信息图**：YiTayML 展示了 **Gemini-3 驱动的 Nano banana pro**，分享了一张 AI 生成的生活信息图，并根据其 [X 帖子](https://x.com/yitayml/status/1991531343675859212?s=46&t=b7l37rB6wtbyAh6ah1NpZQ) 与前 Google 员工开起了关于 “nit:” 前缀的玩笑。
- **Genspark 获 2.75 亿美元融资晋升独角兽**：Eric Jing 在其 [X 帖子](https://xcancel.com/ericjing_ai/status/1991549048642568503?s=46) 中透露，**Genspark 获得 2.75 亿美元 B 轮融资**，估值达 12.5 亿美元，并推出了一个全能 AI Workspace，能够在用户说明意图后自主交付成果。
   - 一位用户提到，使用 Genspark *为我本周的一个演示文稿节省了数小时的时间*。
- **Cline-bench 为 Agent 编程发起 100 万美元 OSS 悬赏**：Cline 推出了 **cline-bench**，这是一系列源自真实开源工程任务的可复现 RL 环境，并获得了业界支持，提供 100 万美元奖金池以激励开发者提交困难的、已部署代码的问题，参考 [此帖子](https://xcancel.com/pashmerepat/status/1991596028735184899?s=46)。
   - 一位成员建议 *Cline bench 应该包含完成任务所需的时间*。
- **GPT-5 在两天内解决十年之久的数学难题**：来自 OpenAI 的 Sebastien Bubeck 分享了一篇论文，证明 **scaffolded GPT-5** 在短短两天内就为 2013 年的一个树子图猜想和 2012 年的一个 COLT 动态网络问题提供了完整证明，引发了人们对 AI 生成可发表定理的热情，详见 [此 X 帖子](https://xcancel.com/SebastienBubeck/status/1991568186840686915?s=20)。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1441298917231956109)** (1 messages): 

> `nanobanana 讨论帖, moodboard` 


- **关注 nanobanana 讨论帖和 moodboard！**：一位成员引导其他人关注 Discord 频道内的 [nanobanana 讨论帖和 moodboard](https://discord.com/channels/822583790773862470/1397010677364953149/1441154669157159073)。
- **不要错过！**：该成员强调不要错过链接中的内容。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1441154669157159073)** (166 messages🔥🔥): 

> `Gemini Image Pro, Nano Banana Pro, Adam Wathan's Avatar Generator, Internal Tooling` 


- **Nano Banana Pro 聚合帖**: 一位用户分享了关于 **Nano Banana Pro** 和 **Gemini Image Pro** 的 [megathread](https://x.com/googledeepmind/status/1991522595129139486?s=46) 链接。
- **Adam Wathan 为内部占位图构建基于 Gemini 的头像生成器**: **Tailwind** 创始人 **Adam Wathan** 分享了一个 Web UI 工具，该工具将精选的风格描述和 Prompt 输入 **Gemini**，以批量生成占位头像 ([链接](https://x.com/adamwathan/status/1991604743488111087?s=46))。
   - 关注者们正请求将其公开，并提供了团队照片批量处理功能以及额外的艺术风格 Prompt 创意。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1441441139835015229)** (2 messages): 

> `OpenRouter Show, Live Streams, X, Youtube` 


- **OpenRouter Show 首映！**: 欢迎在 [X](https://x.com/OpenRouterAI/status/1991597842914550077) 和 [YouTube](https://www.youtube.com/@OpenRouterAI) 上收看下一集 **OpenRouter Show**。
- **OpenRouter 广播上线！**: **OpenRouter** 广播现已在 [X](https://x.com/i/broadcasts/1lPKqvwqWdYGb) 和 [YouTube](https://www.youtube.com/watch?v=p0Hnd0yZLkc) 上进行**直播**。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1441114551562014780)** (137 messages🔥🔥): 

> `linker.sh script failures, Nano Banana 2 errors, Gemini 3 grounding capabilities, OpenRouter completion errors, Chutes provider status` 


- ****Linker.sh 之忧：脚本失败困扰用户****: 用户报告称 `whylinker.sh` 脚本和 `@linker.sh` 工具调用出现失败，一位用户甚至经历了 *1/10* 的失败率。
   - 尽管存在此问题，一些用户（如 *brochacho*）认为这不是大问题，而另一些人指出在 Cursor 中也存在类似问题，建议在需要工具调用时切换到 Sonnet。
- ****Nano Banana 2：400 错误破坏体验****: 用户在 Nano Banana 2 上遇到了 **400 错误**，特别是在其上线 Vertex 之后，导致用户因浪费额度而感到沮丧。
   - 一位用户开玩笑地要求退回 *4 美分* 的错误损失，而另一位用户抱怨该模型在 Vertex 上线已经 *2 小时 40 分钟* 了，这在 *AI 时代基本上等同于两个月*。
- ****Gemini 3 的 Grounding 功能：痴心妄想？****: 成员们对通过 OpenRouter 实现 **Gemini 3 Grounding** 的可能性感到好奇，鉴于其知识截止日期为 25 年 1 月，这一功能备受期待。
   - 然而，目前已确认该功能*尚未*上线，用户只能寻求其他知识集成方案。
- ****OpenRouter 故障抛出 401 错误****: 新加坡和香港的用户报告在使用 OpenRouter 时收到随机的 **401 错误**，具体表现为 *'HTTP Error 401: {"error":{"message":"User not found.","code":401}}'*，涉及多个供应商。
   - 可能的解决方案包括检查 API Key 是否被禁用并创建新 Key；部分用户发现该问题是间歇性的，一段时间后会自动恢复。
- ****Chutes 状态：还活着吗？****: 关于 **Chutes 供应商** 的状态存在疑虑，一些用户猜测它可能因为达到 429 错误限制而终止了与 OpenRouter 的协议。
   - 不过，[OpenRouter 供应商页面](https://openrouter.ai/provider/chutes)确认 Chutes 仍是活跃供应商，尽管用户可能仍因速率限制（Rate Limiting）遇到问题。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1441151496807907552)** (33 条消息🔥): 

> `AI Studio 图像问题, OpenRouter 上的 Vertex, 推理图像, Windows 构建命令, 排行榜页面延迟` 


- **AI Studio 发送重复图像**：发现一个 AI Studio 提供商会发送**两张图像**（其中一张来自推理块），且无法区分它们。该问题已[向 Google 反馈](https://github.com/SillyTavern/SillyTavern/commit/2d9b0ad0a949b4b8458401671208f2db26d9c8ef)以进行平滑处理。
   - 一位成员在自己端修复了此问题，通过将思维签名放入推理区域或将其标记为推理图像来保留它。
- **OpenRouter 上的 Vertex 忽略图像**：有用户注意到 **OpenRouter 上的 Vertex** 完全不返回图像，因为它似乎没有启用图像模态（modality），尽管*它在聊天室中确实可以工作*。
   - 然而，正确设置 **output modality 参数**进行 API 调用应该是可行的，且仅生成一张图像。
- **讨论推理图像的区分**：讨论了如何区分推理图像，一位成员表示它们是**相同的 base64**，只是重复了。
   - 有人指出，在具有大量推理的复杂提示词（prompt）上，图像实际上可能有所不同，提供了一些中间过程。
- **Windows 用户苦于构建命令**：一位用户表达了对在 **Windows** 上使用 `| head` 构建命令的沮丧，并开玩笑说要切换到 **Linux**。
   - 其他人调侃说，*他们理所当然地认为 npm 应该处于 unix 环境中*。
- **排行榜页面性能下降**：一位用户注意到**排行榜页面**比几天前明显更卡顿。
   - 目前尚未确定立即的解决方案，但该问题已引起关注以待调查。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1441160383317409905)** (7 条消息): 

> `模型大小与数据集大小比例, SimpleLLaMA 介绍` 


- **辩论 LLM 预训练模型大小与数据集大小的比例**：一位成员询问了当前 **LLM 预训练**中**模型大小与数据集大小比例**的标准，并指出这取决于数据集。
   - 另一位成员回应称，他们用 **2B token 的数据训练了一个 350M 的模型**，承认这低于 **Chinchilla 最佳比例**，虽然收益递减，但仍实现了基础的算术和问答能力。
- **SimpleLLaMA 发布**：计算机科学专业的高年级学生 Ivan 介绍了 **SimpleLLaMA**，这是一个 **LLaMA 风格的 Transformer**，旨在使整个 **LLM 训练过程**更加透明和可复现，并附带了[详细文档](https://github.com/IvanC987/)。
   - 他还开发了 **DiffusionGen**，这是一个受 **StableDiffusion** 启发的项目，专注于用于图像和文本生成任务的基于扩散（diffusion-based）的生成模型。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1441185287735611524)** (61 messages🔥🔥): 

> `Gradient Compression Algorithm, ArXiv Endorsement Woes, Dion Optimizer Deep Dive, Fron vs Muon, Sparse Optimization` 


- **梯度压缩算法采样 Logits (Gradient Compression Algorithm Sampling Logits)**：一名成员描述了一种梯度压缩算法，该算法根据组的压缩梯度与测试集之间的对齐情况，调整每个组的采样 Logit，该算法基于[这张图片](https://cdn.discordapp.com/attachments/747850033994662000/1441185287328759898/image.png?ex=692188a4&is=69203724&hm=4a8450ed02855606d66f6d660ad91847cbd39137f00576ad4e780205c5bcff39&)。
   - 提议的算法涉及在当前 Checkpoint 压缩测试集的梯度，然后压缩训练集各组的梯度。
- **寻求 ArXiv 推荐 (Endorsement) 协助**：一名成员请求 ArXiv 推荐协助，提到他们已向 **20 个研究团队**发送邮件但未获成功，并链接到了 [ArXiv 的推荐页面](https://arxiv.org/auth/endorse?x=63SW7W)。
   - 另一名成员警告不要进行盲目推荐，建议用户在适当的频道征求反馈和合作，另一人则建议发布手稿。
- ****Dion 优化器**对优化的影响**：讨论了 [Microsoft 的 **Dion 优化器**](https://github.com/microsoft/dion/pull/15) 对优化的潜在影响，尽管一名成员后来划掉了该陈述，可能使其失效。
   - 在随后的讨论中，成员们再次询问这与 **Dion** 有何不同。
- ****Fron 的** Top-K 模拟优先级**：一名成员建议 **Fron 的 Top-K** 比其他方法更紧密地模拟了这种优先级。
   - 他们还指出，低秩数的循环（cyclic with low # of ranks）会强制执行一个更强的先验，即随时间推移在每个方向上的累积更新量大致相等。
- **适用于 PyTorch 的切片反向传播 (Sliced Backprop)**：一名成员建议存储权重的大小和传入的梯度，并利用这些信息在进行反向传播之前，根据大小的期望值选择要切片的维度。
   - 他们补充说，你只能在张量的部分区域执行切片反向传播，这在 PyTorch 的某些现有稀疏优化器中是受支持的。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1441127276841533604)** (74 messages🔥🔥): 

> `Hashhop public solution, Linear Cost Architecture Limitations, Attention Score Approximations, Brain Algorithms, Intrinsic Dimension scaling with context length` 


- ****Hashhop** 破解：公开解决方案现身！**：一名成员指出 **hashhop** 已有公开可用的解决方案，详见[这篇论文](https://arxiv.org/abs/2412.06078v1)，尽管它与最初关于某些任务必须具备 O(n^2) 复杂度的断言有所不同。
   - 该解决方案暗示任务可以用更弱的方法在 **O(n log n)** 时间内完成，这与最初声称 **FT** 总是需要 **O(n^2)** 复杂度的说法形成对比。
- ****线性限制**：AGI 的架构焦虑**：一名成员认为 AGI 不能基于**线性成本架构 (linear cost architecture)**，除非 **3SAT** 被解决，这表明更强大的模型需要约束求解（constraint solving）而非简单的统计匹配。
   - 该用户坚持认为 AGI 不会、不能、永远不会是线性成本架构（除非我们解决了 3SAT）。
- ****Attention 近似**：Softmax 后的 Epsilon 优势！**：讨论围绕 Softmax 后的 Attention 分数近似展开，一名成员建议许多分数接近于**零**，可以通过**近似最近邻搜索 (approximate nearest neighbor search)** 来确定。
   - 他们认为对于现实世界的任务，可能会出现一种策略，使平均复杂度低于二次方，即使是 **N^1.8**，但另一人指出在一般情况下这是不可能的。
- ****大脑的勇敢**：引擎盖下的算法！**：一名成员质疑我们的大脑是否使用 **N^2** 算法，考虑到其固定的容量和明显的缺乏完美记忆，建议采用类似于 **RNNs** 的**压缩记忆**方法。
   - 另一人反驳说，人类在 **O(n)** 长度的上下文中对某些项目执行 **O(n)** 的工作量，对于人类来说，我们甚至可能在相对较短的列表上这样做，尤其是当我们想要确保详尽无遗时。
- ****维度困境**：内在缩放秘密！**：成员们辩论向量的**内在维度 (Intrinsic Dimension, ID)** 是否必须随上下文长度增加以保持可区分性，认为需要维度来使向量随着上下文长度增加而变得不同。
   - 另一名用户认为 *Attention 并不要求向量（近乎）正交才能区分，只需要它们的相似性结构（similarity structure）有足够的差异即可*。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1441112842391523338)** (22 条消息🔥): 

> `Compiler design, DeCuda project, GPU BIOS modding, CUDA mini-projects` 


- **DeCuda 反编译 PTX**：一个名为 **DeCuda** 的相对隐蔽的项目，可以将 **PTX** 反编译为伪 CUDA 目标，这可能是另一个值得为新架构扩展的有趣项目。
   - 该项目自 **GTX 480** 世代以来一直没有公开维护。
- **关于 GPU BIOS modding 的提问**：一名成员询问了关于 GPU BIOS 修改和刷写的问题，问道：*由于 NVIDIA 增加了签名检查，现在修改 BIOS 是不是已经不可能了？* 并询问了关于修改 **AMD** GPU 的 **power caps**（功耗限制）和 **throttling settings**（降频设置）的问题。
   - 另一名成员将他们引导至特定频道：<#1349152646484987974>。
- **头脑风暴 CUDA mini-project 想法**：一名成员正在 *寻找一个可以在约 1 个月内完成的可靠 mini-project*，并正在查看 **Watershed/RANSAC** 等项目。
   - 另一名成员建议，*在你自创的酷炫想法中，你可能会取得更大的进展*。 


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1441156829424910438)** (2 条消息): 

> `Dataflow Execution on GPUs, Spatial Pipelines, Kitsune, PyTorch Dynamo` 


- **Kitsune 实现了 GPU 上的 Dataflow Execution**：一篇题为 [Enabling Dataflow Execution on GPUs with Spatial Pipelines](https://dl.acm.org/doi/10.1145/3777466) 的新论文介绍了 **Kitsune**，这是一组用于构建空间流水线（spatial pipelines）的原语，通过 **PyTorch Dynamo** 在 GPU 上实现数据流执行。
   - 在 5 个挑战性应用中，Kitsune 在推理和训练方面分别提供了高达 **2.8x** 和 **2.2x** 的性能提升，并分别减少了高达 **99%** 和 **45%** 的片外流量。
- **GPU Dataflow 架构提供性能提升**：该论文解决了深度学习应用在 GPU 上进行块同步执行（bulk-synchronous execution）的局限性，强调了 GPU 资源闲置和数据移动不佳等低效问题。
   - 论文认为，对当前 GPU 架构进行适度调整即可实现高效的数据流执行，在无需完全重新设计的情况下规避垂直融合（vertical fusion）的限制。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1441178867724451870)** (2 条消息): 

> `Rivian hiring, Modal hiring, GPU coders, Autonomous Driving, inference optimization` 


- **Rivian 为 Autonomous Driving 招募 GPU 编程专家**：Rivian 正在积极为其下一代 **Autonomous Driving 功能** 招聘 **GPU 编程专家**，要求具备 **CUDA** 或 **量化 (QAT)** 方面的专业知识，工作地点位于加州 **Palo Alto** 和英国 **伦敦** ([职位描述 1](https://careers.rivian.com/careers-home/jobs/26857?lang=en-us&previousLocale=en-US), [职位描述 2](https://careers.rivian.com/careers-home/jobs/24737?lang=en-us&previousLocale=en-US))。
- **Modal 寻找顶尖 GPU 工程师进行推理优化**：Modal 在贡献了 **SGLang** 和 **FlashAttention**，并协助了 **Decagon**、**Reducto** 和 **Suno** 等客户后，正在寻找经验丰富的 **GPU 工程师** 进行 **推理优化** ([SGLang 博客](https://modal.com/blog/host-overhead-inference-efficiency), [FlashAttention 博客](https://modal.com/blog/reverse-engineer-flash-attention-4), [Decagon 案例研究](https://modal.com/blog/decagon-case-study), [Reducto 案例研究](https://modal.com/blog/reducto-case-study), [Suno 案例研究](https://modal.com/blog/suno-case-study))。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1441244677029367870)** (1 条消息): 

> `Lecture Slides, Numeric and AI` 


- **讲座讲义发布**：paulius 关于数值计算和 AI 的第 84 讲讲义现已发布，点击 [此处](https://github.com/gpu-mode/lectures/tree/main/lecture_084) 查看。
- **讲座重点：数值计算与 AI**：该讲座涵盖了与数值计算及其在人工智能中应用相关的主题，详情见 [链接讲义](https://github.com/gpu-mode/lectures/tree/main/lecture_084)。


  

---

### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1441449930525769769)** (2 messages): 

> `Intel Compute Runtime, VectorAdd performance` 


- **Intel 的 Compute Runtime 加速更新**：Intel 发布了新版本的 [Compute Runtime](https://github.com/intel/compute-runtime/releases/tag/25.44.36015.5)，正如 [这篇 Phoronix 文章](https://www.phoronix.com/news/Intel-CR-25.44.36015.5) 中所提到的。
   - 该运行时对于利用 Intel GPU 进行计算任务至关重要。
- **VectorAdd 进行性能检查**：一位用户报告称，在 GPU 上处理 **10 亿个元素** 的 `VectorAdd()` 需要 **2 分钟**，并询问这是否正常。
   - 需要社区反馈来评估此性能是否在预期范围内。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1441148066303578262)** (2 messages): 

> `Efficient Ray Tracing, wgpu Ray Tracer` 


- **光线追踪需要现代 GPU**：高效的 **光线追踪 (Ray Tracing)** 需要 **现代 GPU** 才能高效运行。
   - *如果我希望游戏在所有设备上看起来都一样，我可能应该避免在游戏中使用它*。
- **简单的 wgpu 光线追踪器**：一位成员计划创建一个简单的 **wgpu 光线追踪器** 用于学习。
   - 该成员打算将其构建为一个 **渲染器 (renderer)** 而不是 **实时 (real-time)** 的。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1441224765137485946)** (1 messages): 

> `Multi-GPU Programming, Triton, Iris, Octa, Distributed LLMs` 


- **Iris 简化多 GPU 编程**：**Iris 论文** 为 **Triton** 引入了原生的基于 tile 的对称内存和内核内通信 (in-kernel communication)，从而简化了多 GPU 编程，详见这篇 [ArXiv 论文](https://arxiv.org/abs/2511.12500)。
   - 该论文声称实现了高达 **1.79 倍的加速**。
- **Octa 揭示分布式 LLM 中的三种税 (Three Taxes)**：**Octa 论文** 介绍了分布式 LLM 中的 **三种税**，并展示了细粒度的内核内通信如何将端到端延迟降低 **10-20%**，详见这篇 [ArXiv 论文](https://arxiv.org/abs/2511.02168)。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1441118176032325806)** (30 messages🔥): 

> `NVIDIA leaderboard submissions, nvfp4_gemv performance, Personal best submissions` 


- **NVIDIA 排行榜提交火热进行中**：NVIDIA 的 `nvfp4_gemv` 排行榜收到了大量提交，用户频繁更新其结果。
   - 提交的性能范围从 **20.6 µs** 到 **562 µs**，表明性能差异很大。
- **在 NVIDIA 上突破个人最佳成绩**：几位用户在 NVIDIA 的 `nvfp4_gemv` 排行榜上刷新了 **个人最佳记录**。
   - 一位用户达到了 **42.5 µs** 的个人最佳，另一位达到了 **155 µs**。
- **在 NVIDIA 上获得第一名！**：一位用户在 NVIDIA 的 `nvfp4_gemv` 排行榜上获得了 **第一名**。
   - 他们以 **20.6 µs** 的获胜时间夺冠。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1441262675949523065)** (2 messages): 

> `RX 590 VBIOS on RX 580, AGX Thor vs DGX Spark` 


- **RX 590 BIOS 刷入 RX 580 失败**：一位成员尝试将 **RX 590 VBIOS** 刷入规格相似的 **RX 580**，但遇到了驱动安装问题。
   - 尽管内存支持、时序、GPU 芯片 (**Polaris 20**) 匹配并调整了 sub-ID，该修改仍无法正常工作，用户正在寻求失败原因的见解。
- **AGX Thor 与 DGX Spark 指令集**：讨论提到 **AGX Thor** 拥有 `tcgen05.mma` 指令，而 **DGX Spark** 则没有。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

thakkarv_86311: 对于 attention 是的，对于其他基于模板的预写内核也是如此。
  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1441116863059857508)** (35 条消息🔥): 

> `Benchmark 偏差、PyTorch 版本、Cutlass DSL 与 sm_120、在 DataCrunch 上进行 Profiling、tcgen05 与 cp.async` 


- ****Benchmark 忧虑**：提交作品遭遇速度波动！**：一位成员观察到 benchmarking 脚本存在巨大偏差，本地测试显示的加速比提交的作品更大，在调整参数后跳变到了 **30 us**。
   - 尽管调整了脚本，提交的 benchmark 时间仍比本地预期的要慢；该用户指出：“我猜目前的 benchmarking 脚本中仍存在相对较大的偏差……现在显示为 30 us。真奇怪。”
- ****PyTorch 困惑**：版本验证之旅！**：一位成员询问了比赛中使用的 **PyTorch** 版本。
   - 另一位成员回复称，在他们所有的运行中，PyTorch 版本均为 **2.9.1+cu130**。
- ****Cutlass 难题**：RTX Pro 6000 的考量！**：一位拥有 **NVIDIA RTX Pro 6000** (**sm_120**) 的成员询问了 **nvidia-cutlass-dsl** 的 wheel 包。
   - 另一位成员澄清说：“Cutedsl 在 sm120 上运行良好，你不需要自定义 wheel”，同时提醒 profiling 特性并不代表 B200。
- ****Profiling 困境**：DataCrunch 的差异！**：一位成员报告了在 DataCrunch 上使用 **ncu** (**54ms**) 与提交服务器 (**30usec**) 之间的显著差异。
   - 另一位成员建议使用 `--clock-control none` 标志，而原发帖者补充道：“当我从这里的 bot 获取 profile 时，计时也稍微高一些。我认为 bot 没有使用 `--clock-control None`。”
- ****独门秘籍**：揭秘 UltraInstinct 的代码炼金术！**：一位成员询问代码是否使用了 **tcgen05** 和 **cp.async**。
   - 另一位成员回应道：“我不确定我想透露多少，除了根据我之前的评论，大家可能已经知道我一直没用 tcgen05”，尽管他们在早期尝试过参考实现，并使用 **torch._scaled_mm** 达到了 **<90us**。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1441181574262427648)** (9 条消息🔥): 

> `VLA 微调、RoboTwin 2.0、遵循结构完整性的视觉模型、ManiSkill 桌面任务` 


- **VLA 微调之路**：一位成员计划通过 **SFT** 对 **VLA** 进行微调，数据集使用 **ManiSkill 桌面任务** 生成，并采用经典路径规划的解决方案，使用 **FAST tokenizer**。
   - 该计划包括在模拟桌面任务中评估 **VLA**，并通过模拟环境中的 **RL** 进行进一步训练，可能会使用现代形式的 **GRPO**。
- **结构完整性研究征集**：一位成员征求关于**遵循结构完整性的视觉模型**的研究建议，特别是用于生成合成数据和世界模型的研究。
   - 他们提到了 **Genie3** 及其由于内存限制在长时程（long horizons）内维持环境一致性所面临的挑战。
- **提及长时一致性论文**：一位成员分享了一篇解决视觉模型中**长时一致性**问题的[论文](https://arxiv.org/abs/2505.20171)，尽管它并非直接关于约束。
   - 另一位成员推荐了 **Xun Huang 的世界模型**，并指向了[这条推文](https://x.com/neurosp1ke/status/1986814187062890855)。
- **RoboTwin 2.0 介绍**：分享了环境集合 **RoboTwin 2.0**，并指向了其 [GitHub 仓库](https://github.com/RoboTwin-Platform/RoboTwin)。
   - 一位成员为 **Maniskill 桌面** 示例生成了带有运动规划解的数据，但发现环境过于简单，于是开始从 [Hugging Face](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) 下载 **1.4 TB** 的预生成数据集。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1441119626816655501)** (61 条消息🔥🔥): 

> `Maya1 Voice Model, kohya_ss-windows.zip, Fullstack JS Engineer Career Change to AI, Reasoning-like Behavior in Neural Networks, Avian.io Inference Provider Registration` 


- **Maya1 Voice Model 登陆 Fal**：**Maya1 Voice Model** 现在可以在 Fal 上试用，正如[这条推文](https://x.com/Dheemanthredy/status/1991566362813296965)中所宣布的。
   - 该发布承诺在语音建模和实时应用方面提供新功能。
- **在 GitHub 中寻获难觅的 kohya_ss-windows.zip**：一名成员正在寻找 **kohya_ss-windows.zip** 的下载链接，另一名成员指向了 **kohya_ss** GitHub 仓库中的[安装选项](https://github.com/bmaltais/kohya_ss?tab=readme-ov-file#installation-options)。
   - 据透露，该 zip 文件可能已过时，建议参考一些[安装指南](https://github.com/bmaltais/kohya_ss/discussions/3218)和故障排除说明。
- **全栈工程师寻求转行 AI**：一位全栈 JavaScript 工程师正在考虑转行 **AI engineering** 和 **data science**，并寻求指导。
   - 建议将此问题移至 *ask for help* 频道。
- **SmolLM3 引入推理模式**：成员们讨论了训练现有神经网络以“学习类推理行为”的方法，引用了一篇关于[让任何模型具备推理能力](https://huggingface.co/blog/Metal3d/making-any-model-reasoning)的博客文章。
   - 据称 **SmolLM3 包含了一个实际的推理模式**，且其训练过程是公开的——并提供了 [SmolLM3 博客文章](https://huggingface.co/blog/smollm3)作为示例。
- **Avian.io 寻求集成至 Hugging Face Hub**：**Avian.io** 请求根据 [Hugging Face 文档](https://huggingface.co/docs/inference-providers/en/register-as-a-provider)中的说明，注册为 Hugging Face Hub 的 **Inference Provider**。
   - 他们已提交了一个 [pull request](https://github.com/huggingface/huggingface.js/pull/1848) 来添加其服务。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1441241699413397515)** (1 条消息): 

> `smolagents, code agents` 


- **Smolagents 构建 Code Agents**：一位用户正在学习如何使用 [smolagents](https://huggingface.co/learn/agents-course/en/unit2/smolagents/code_agents) 构建使用代码的 **Agents**。
   - 该课程侧重于如何使用这些 agents 来自动化代码相关任务。
- **Coding Agents**：探索 **smolagents** 在编程场景中的应用，从而实现自动化解决方案。
   - 这些 agents 有可能通过自主处理各种编程任务来简化开发流程。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1441209356912824465)** (5 条消息): 

> `MemMachine Playground, Langchain Series, Multimodal AI Benchmarks, Missing LLM Capabilities` 


- ****GPT-5**、**Claude 4.5** 和 **Gemini 3 Pro** 现已在 MemMachine Playground 上线**：[MemMachine Playground](https://huggingface.co/spaces/Memverge/MemMachine-Playground) 已发布，提供对 **GPT-5**、**Claude 4.5** 和 **Gemini 3 Pro** 的访问，所有模型均由持久化 AI 记忆支持。
   - 该 Playground 完全 **open-source**，提供多模型环境，专为实验记忆加 agents 而构建。
- **新的 Langchain 系列获得社区支持**：一位成员正计划根据最新功能推出新的 **Langchain** 系列，并征求社区对该想法的反馈。
   - 该社区成员附带了一个[之前的 Langchain 教程链接](https://www.youtube.com/watch?v=8xgOLcg9Pco)，并征求对其作品的反馈。
- **Frontiers 讲座将聚焦多模态 AI 基准测试**：下周二将举办一场关于设计下一代**多模态 AI 基准测试**的讲座。
   - 鼓励对评估、多模态模型或功能性智能感兴趣的人员[参加讲座](https://luma.com/kwg2qg4d)。
- **分享关于缺失 LLM 能力的 X 博客文章**：一位成员分享了一篇[博客文章](https://x.com/ShashwatGoel7/status/1991611877667840181?t=QKZdUdtbigMMfQSHtrczew&s=19)，讨论了 **LLM 能力**中仍然缺失的部分，并对改进这些能力的通用（后）训练环境进行了推测。
   - 该文章发布在 *X* 上。


  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1441319094698901627)** (1 messages): 

> `Diffusers MVP program, Open critical issues on Diffusers` 


- **Diffusers MVP 项目启动贡献**：在 **The Diffusers MVP program** 宣布后不久，社区便涌现出了出色的贡献和参与。
   - 鼓励成员关注待解决的关键问题，并在此处查看详情 [here](https://github.com/huggingface/diffusers/issues/12635)。
- **Diffusers：MVP 启动后社区参与度飙升**：随着 **The Diffusers MVP program** 的揭晓，社区以显著的贡献和积极的参与做出了响应。
   - 敦促贡献者探索尚未解决的关键问题（可在 [GitHub](https://github.com/huggingface/diffusers/issues/12635) 上获取），以进一步增强项目的发展。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1441450400141021206)** (16 messages🔥): 

> `Logo Detection, LayoutLMv3, Dinov2 for Logo Detection` 


- **经典 CV 或 Dinov2 可实现 Logo 检测**：成员们讨论了利用 **CNN 或 ViT**，建议在 Logo 集有限且方差较低的情况下，使用像 **dinov2** 这样的小型预训练模型来快速微调 Logo 检测。
   - 然而，由于约 **50 个付款人（payers）** 带来的高变异性，这种方法面临挑战，每个付款人至少有 **4 种文档类型**，且数据随时间和付款人而变化。
- **LayoutLMv3 正在测试中**：针对提出的问题，一名成员正在尝试 **LayoutLMv3**，因为它能同时理解文本和视觉信息。
   - 有人提到，甚至可以使用经典的计算机视觉方法，因为现在图片中定位文本的方法已经非常成熟。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/)** (1 messages): 

abidlabs: 正在直播！https://www.youtube.com/watch?v=ohYBeIQmFa4 <@&1014548769355862036>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1441170698298200074)** (3 messages): 

> `Smol Course Sign-Ups, Circular Link Reference in smol-course, Training Issues on smoltalk2_everyday_convs_think, Pending Reviews on Leaderboard Discussions` 


- ****Smol Course** 报名状态？**：一位用户询问 **smol-course** 的报名是否已关闭。
- ****smol-course** 循环链接困扰**：一位用户报告了 [https://huggingface.co/smol-course](https://huggingface.co/smol-course) 与 [https://huggingface.co/learn/smol-course/unit0/1](https://huggingface.co/learn/smol-course/unit0/1) 之间的循环链接引用。
- ****smoltalk2_everyday_convs_think** 的训练困扰**：一位用户在 **HuggingFaceTB/smoltalk2_everyday_convs_think** 上进行训练时，使用 [https://huggingface.co/learn/smol-course/en/unit1/3](https://huggingface.co/learn/smol-course/en/unit1/3) 提供的确切代码示例遇到了问题。
- **等待审核的 Leaderboard 讨论**：[leaderboard](https://huggingface.co/spaces/smol-course/leaderboard) 讨论区中的多个提交（特别是讨论编号 **#36** 及其之后的提交）正在等待审核。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1441111346627215430)** (54 messages🔥): 

> `Academic Dishonesty, Infographic Overload, Nano Banana Model, Paper Posting Limits, AI Paper Filtering` 


- ****教师试图利用 AI 作弊****：一位用户请求帮助*提升其 20% 的工作内容*，结果暴露了自己是**大学教师**的身份。
   - 其他用户立即指责这是**学术不端**，其中一人表示对这类请求*已毫无尊重可言*。
- ****Gemini 3 Pro 的 Nano Banana 导致信息图泛滥****：成员们讨论了 **Gemini 3 Pro** 中新的 **Nano Banana 模型** 如何让制作信息图变得如此简单，以至于很快就会出现信息图洪流，并引用了 [推文](https://x.com/emollick/status/1991527285267275854?s=46)。
   - 有人指出，*AI 生成图像中图表不连贯和文本畸形的时代已成过去*，但一位用户指出 *生成的时钟显示的不是 10:15*。
- ****Discord 辩论论文发布限制****：针对某用户每天发布的论文数量及其是否造成干扰展开了讨论，许多人希望设定 **每天 1 篇论文的限制**。
   - 该用户反驳道：*你不可能真的每天读 20 篇论文*，其他人回应：*我们不是来帮你做你自己懒得做的筛选工作的*。
- ****AI 辅助论文筛选****：由于论文发布受到限制，一位用户考虑使用 **AI**（如 <#1435893010205380619>）来筛选论文。
   - 在遭到其他人的反对后，该用户表示：*我刚刚把它分配给了 Antigravity IDE（Google 的 Windsurf 分支），Discord 机器人即将推出*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1441123619081162803)** (4 messages): 

> `AI Zar David Sacks, OLMo 3, YouTube video` 


- **AI 沙皇 David Sacks 遭到抨击！**：一位成员声称 *100% 的消息来自 AI 沙皇 David Sacks，因为那个笨蛋 Charmath 一直在他耳边吹风*，并开玩笑说要给住在 4Chan 上的 Elmo 表弟起个名字。
   - 该评论包含了一个指向相关 Twitter 帖子的 [链接](https://fxtwitter.com/natolambert/status/1991508141687861479)。
- **OLMo 3 发布！**：成员分享了 **OLMo 3** 的链接，包括 [Allen Institute for AI 博客文章](https://allenai.org/blog/olmo3) 和 [技术报告](https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf)。
- **分享了 YouTube 视频**：一位成员简单分享了一个 [YouTube 视频](https://youtu.be/F1pBIjQblI0)，没有提供额外上下文。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1441110920997896363)** (46 messages🔥): 

> `nano banana pro, early grok image model, Infographics AI-slop, Gemini 3 Pro, Adobe buys Semrush` 


- **Nano Banana Pro 诞生！**：成员们分享了 [**Nano Banana Pro** 生成的图像](https://cdn.discordapp.com/attachments/1149866623109439599/1441110920339132447/image.png)，并讨论说 *头发有一种纹理感，就像早期的 Grok 图像模型*。
   - 他们指出这可能是由于 **patches** 的问题导致的。
- **信息图将等同于 AI 垃圾内容**：一位成员分享了 [信息图链接](https://x.com/scaling01/status/1991523932336464333) 并表示这是他们见过的 *模型生成的在文本、错误率和布局设计方面最好的信息图*。
   - 另一位成员预测 **信息图** 在 2026 年将等同于 **AI 垃圾内容 (AI-slop)**。
- **Gemini 3 Pro 需要 Pro 账户**：成员们讨论了使用 **Gemini 3 Pro** 需要 **Pro 账户**。
   - 一些成员通过手机获得了免费一年的使用权，另一些人则不认为支付订阅费用是值得的。
- **Adobe 收购 Semrush**：一位成员分享了 [TechCrunch 文章](https://techcrunch.com/2025/11/19/adobe-to-buy-semrush-for-1-9-billion/)，报道 **Adobe 以 19 亿美元收购 Semrush**。
   - 没有讨论更多细节。
- **中国开源模型追赶 Deepmind Gemini 3**：一位成员表示 *正淡定地等待那些**中国开源模型**免费追赶上 Deepmind **Gemini 3** (Gemma/Banana Pro) 的水平*。
   - 他们补充道：*当中国人入场时，利润就荡然无存了*。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 messages): 

> `ArXiv Endorsement, Discord Server Link` 


- **ArXiv 背书请求**：一名成员请求协助获取 ArXiv 背书，提到他们已经给大约 **20 个研究团队**发了邮件但均未成功，并分享了一个 [ArXiv 背书链接](https://arxiv.org/auth/endorse?x=63SW7W)。
   - 另一名成员请发帖者重新发送背书链接，因为他们之前错过了。
- **分享 EleutherAI Discord 服务器**：一名成员分享了 [EleutherAI Discord 服务器](https://discord.gg/eleutherai)的链接。
   - 未提供关于分享该链接原因的进一步背景信息。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1441512329765453824)** (1 messages): 

> `LLM Capabilities, Post Training Environments, Missing capabilities in LLMs` 


- **博客文章分析 LLM 缺失的能力**：一名成员分享了一篇[博客文章](https://x.com/ShashwatGoel7/status/1991611877667840181?t=QKZdUdtbigMMfQSHtrczew&s=19)，分析了 **LLM 能力**中目前仍然缺失的部分。
   - 该文章推测了能够实现这一目标的通用 **post-training environment**（训练后环境）。
- **缺失能力补充**：分享的博客文章讨论了若能解决这些缺失元素，将为 LLM 能力带来巨大提升。
   - 该博客重点关注 LLM 的改进领域。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1441184809606185144)** (3 messages): 

> `ArXiv Endorsement, EleutherAI Discord Invite` 


- **寻找 ArXiv 背书者**：一名成员在联系了约 **20 个研究团队**未果后，请求协助获取 ArXiv 背书，并分享了[他们的背书链接](https://arxiv.org/auth/endorse?x=63SW7W)。
   - 他们请求重新发送链接，因为之前错过了。
- **分享 EleutherAI Discord**：一名成员分享了 **EleutherAI Discord** 的邀请链接：[https://discord.gg/eleutherai](https://discord.gg/eleutherai)。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1441322574549356655)** (2 messages): 

> `Mamouth cloud solution, Mamouth private deployment, Mamouth costs` 


- **Mamouth 部署：云端还是私有化？**：一名成员询问 **Mamouth** 是纯云端解决方案还是支持私有化部署。
   - 另一名成员建议直接联系 [Modular](https://www.modular.com/request-demo) 以获取有关部署选项和相关费用的详细信息。
- **联系 Modular 获取价格详情**：一名用户询问了 **Mamouth** 的成本以及是否可以私有化部署。
   - 另一名用户回复说，该用户应通过 Modular 的 [demo 请求页面](https://www.modular.com/request-demo)联系他们以咨询定价。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1441199888581787668)** (1 messages): 

> `MAX Python API, NVIDIA Grace Support, Mojo GPU Programming` 


- **Modular Platform 25.7 发布！**：Modular Platform **25.7** 引入了完全开放的 **MAX Python API**、下一代建模 API、扩展的 **NVIDIA Grace** 支持，以及更安全、更快速的 **Mojo GPU** 编程。
   - 根据 [Modular 博客文章](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)，此次更新旨在帮助开发者专注于 AI 进展而非基础设施。
- **更快的推理与更安全的 GPU 编程**：新版本强调了推理速度的提升以及使用 Mojo 进行 GPU 编程时安全性的增强。
   - 开发者现在可以在进行 AI 编程时享受更统一的体验，减少在底层基础设施问题上花费的时间。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1441137649954979952)** (16 条消息🔥): 

> `Mojo 25.7 版本发布，Mojo 中的 Optional 语法，UnsafePointer 泛型，文档中的复制页面内容按钮` 


- **Mojo 发布新版本：25.7**：**Modular 团队**发布了 **25.7** 版本，主打更快的推理速度、更安全的 GPU 编程以及更统一的开发者体验；详情请参阅 [Modular 博客](https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience)。
   - 爱好者们热切期待 AI-native 能力成为现实。
- **Optional 语法推测引发类似 Swift 的向往**：讨论了在 Mojo 中引入类似 Swift 的 `Foo?` 可选类型语法以及可选链（optional chaining，如 `foo?.bar()`）的可能性。虽然团队目前并未将语法糖列为优先级，但有一位成员认为可选链和 `T?` 语法是“显而易见”的需求。
   - 一位成员建议使用 `SIMD[.float, 4]` 进行静态成员的上下文推理（参考 Swift），另一位成员分享了一个 [Rust RFC](https://rust-lang.github.io/rfcs/3058-try-trait-v2.html) 以供参考。
- **UnsafePointer 泛型变为强制性**：**UnsafePointer** 的**泛型**（Mojo 参数，如 mut、type、origin 等）不再提供默认值。
   - 更多信息请见 [提案文档](https://github.com/modular/modular/blob/main/mojo/proposals/unsafe-pointer-v2.md#migration-guide-from-legacyunsafepointer-to-the-new-unsafepointer)，其中包括从 LegacyUnsafePointer 迁移到新 UnsafePointer 的指南。
- **自动复制粘贴按钮即将上线？**：一位用户建议 Mojo 在文档右上角添加自动复制页面内容的按钮，“以便我们更快地利用 LLM 进行直觉学习或实现”。
   - 该建议旨在顺应当前文档的发展趋势，以适应“Z 世代及以后的开发者，他们通常在通过 vibe coding 搞砸之后才去学习原理”。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1441159104964857856)** (11 条消息🔥): 

> `MAX, AMX, NDBuffer, LayoutTensor, CPU 推理` 


- **MAX 开放！**：成员们对 **MAX** 的开放感到兴奋，一位成员在看到[特定 commit](https://github.com/modular/modular/commit/e0d81b694b4eab18d22f3a12d3b966e03e055b18) 后询问 **AMX** 是否已超出 **MAX** 的范围。
- **AMX 弃用原因**：一位成员表示 **AMX** 目前未被使用，且将其集成到当前的 **tensor core 框架**中非常困难，特别是 **Intel** 和 **AMD** 很快就会发布替代方案。
   - 他们补充说，如果 **MAX** 有需求，重新添加支持 **AMX** 的框架是可以的，但这会涉及到诸如定制化张量并行（bespoke tensor parallelism）和专家并行（expert parallelism）等问题。
- **NDBuffer 谢幕，LayoutTensor 登场**：特定代码的移除是弃用 **NDBuffer** 并将所有用途迁移到 **LayoutTensor** 计划的一部分。
   - 由于客户对 **CPU 推理**的使用场景不多，该功能可能不会被重新添加，但欢迎贡献基于 **LayoutTensor** 的版本。
- **LayoutTensor 版本已经存在！**：一位成员已经编写了 **layout tensor** 版本的大部分内容，初步草案可在 [Gist](https://gist.github.com/CoffeeVampir3/d82917f6fce60c0c2cdf00629c4de67d) 查看。
   - 如果其他人认为有价值，该成员愿意对其进行泛化和完善。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1441211594309898361)** (22 messages🔥): 

> `Operator extension 安装问题, Aurora Seeker, 用于洞察的个人数据处理, Manus Knowledge 条目示例, Manus 推荐码/兑换码` 


- **Operator Extension 持续提示安装**: 有用户反馈 Chrome 中的 **Operator extension** 即使在指向已打开的 Amazon 标签页时，仍会反复提示安装。
   - 用户提到另一种选择是使用 **Aurora Seeker**。
- **构建具有洞察力的个人数据仓库**: 一位用户正在寻求关于构建存储和处理个人数据以获取洞察的工具的建议，并链接了他们之前的项目：[contextflow](https://share.cleanshot.com/StvTll4j), [oncue](https://www.youtube.com/watch?v=4UaQEB1b84E&feature=youtu.be), 以及 [axon](https://www.linkedin.com/posts/harrison-qian-95b0062a3_won-the-trae-solo-hackathon-sf-tech-week-activity-7383626911199141890-8SgL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEkmXdcBT7GJOg4Kg0Iy89EqxavBMIqIxk4))。
- **Manus Knowledge 条目灵感**: 一位用户请求 **Manus Knowledge** 条目的优秀示例，目前他们正在使用诸如 *Always do this* 或 *Never do that* 之类的指令。
- **恳求 Manus 额度**: 一位用户询问 **Manus** 的兑换码，提到由于业务限制无法负担充值费用。
   - 另一位用户分享了他们的 [Perplexity Pro 推荐链接](https://plex.it/referrals/VCETA5M7)，还有人指出 [Perplexity 为大学生提供 1 年 Pro 会员优惠](https://plex.it/referrals/VCETA5M7)。
- **Manus vs Gemini 3 测试即将进行**: 一位用户宣布他们将对 **Manus** 和 **Gemini 3** 进行对比测试，以确定哪个是更好的 Agent。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1441134069483176079)** (14 messages🔥): 

> `Kimi K2 所在地, GPT-5.1 在 K2 上进行了蒸馏？, Kimi 的注意力机制较弱？, 开源模型落后？, K2t vs Gemini 2.5 pro` 


- **Kimi 托管在美国**: 一位用户提到 **Kimi** 告诉他们它托管在 **US**（美国），因为那是他们居住的地方，并附上了一张 [图片](https://cdn.discordapp.com/attachments/1371757564005711973/1441250576124870656/image.png?ex=6921c572&is=692073f2&hm=a285f73eb8a311c6828f6c0d0c8952a9b923f30dedfb12ec2608c64cb&) 作为证据。
- **GPT-5.1 已在 K2 上进行蒸馏**: 一位用户展示了他们所谓的“不可辩驳的证据”，证明 **GPT-5.1** 已在 **K2** 上进行了蒸馏，并附上了 [相关图片](https://cdn.discordapp.com/attachments/1371757564005711973/1441296525904052315/IMG_6579.png?ex=6921f03d&is=69209ebd&hm=2c5c12a19efdb6c54624ef43b952c7d30fd8b99d4cd7125465df5e28d1c30afc&)。
- **Kimi 在复杂任务中注意力较弱**: 一位用户评论说 **Kimi** 的有效注意力机制太弱，在涉及长上下文（long contexts）的复杂任务中表现不佳。
- **开源模型落后 9 个月？**: 一位用户分享说，根据一个高度公正且完全没有偏见的机构最近的评估（evals），开源 AI 模型比私有模型落后 **9 个月**，并链接到一条 [推文](https://x.com/scaling01/status/1991665386513748172?s=46)。
- **K2t 优于 Gemini 2.5 pro**: 一位用户表示 **K2t** 肯定比 **Gemini 2.5 pro** 好得多，并且完全处于 **Sonnet 4.5** 的水平。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1441117319521632357)** (1 messages): 

> `Gem3pro, Proxy Server, DSPy Proxy` 


- **Gem3pro 一次性（One-Shot）构建代理服务器**: 一位用户根据 [这条推文](https://x.com/skylar_b_payne/status/1990808733140779488) 提示 **Gem3pro** 构建一个代理服务器，并且第一次尝试就成功了。
   - 该用户分享了一个包含所创建的 **DSPy proxy** 的 [GitHub 仓库](https://github.com/aryaminus/dspy-proxy)。
- **DSPy Proxy GitHub 仓库**: 上述提到的 **DSPy proxy** 可以在 [这个 GitHub 仓库](https://github.com/aryaminus/dspy-proxy) 中找到。
   - 该仓库是在 **Gem3pro** 成功 One-Shot 生成代理服务器后创建的。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1441369788214546534)** (3 messages): 

> `Agent DAG Generation, Think Workflow Compile/Validate DAG, DSPy Prod/Inference/Run Time` 


- **Agent 生成任务 DAG 以提升性能**：一位成员认为*让 Agent 生成任务 DAG* 的核心想法非常棒，因为随后可以使用 **RL** 来尝试获取更好的性能。
   - 他们想知道这是否可以适配为 **Think -> Workflow -> Compile/Validate DAG -> Execute workflow** 的模式。
- **DSPy：将 DAG 适配到生产环境**：一位成员询问 Agent DAG 的想法如何与 **DSPy** 结合使用，以及是否可以将其用于生产/推理/运行时。
   - 他们认为这些环节不一定需要专门训练，重点应放在对 **DSPy** 的适配上。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1441152073503740075)** (3 messages): 

> `AI Engineer Introduction, GEPA Assistance Request` 


- **资深 AI 工程师准备就绪**：一位拥有 **10 年** 经验的高级全栈兼 AI 工程师介绍了自己，强调了他们在利用 **LangChain**、**LangGraph** 和 **Next.js** 等技术构建智能、可扩展 AI 系统方面的专业知识。
   - 他们的背景包括任务自动化、语音聊天、CRM 集成以及 AI 驱动的 SaaS 应用开发，并表示有兴趣在潜在客户开发、支持或定制应用等 AI 相关项目上进行合作。
- **寻求 GEPA 专家指导**：一位成员在特定频道请求有关 **GEPA** 的协助。
   - 该成员直接标记了管理员，认为这是职位垃圾信息。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1441116538626244728)** (6 messages): 

> `DNS Migration, Domain Governance, Downtime Risks` 


- **DNS 域名过渡至社区控制**：**modelcontextprotocol.io 域名**正在从 **Anthropic 企业 DNS** 账户迁移到社区控制下，以改善治理并实现更快的 DNS 设置。
   - 此次过渡旨在为需要 DNS 设置的项目提供便利，并允许对项目域名实施 **Infrastructure as Code (IaaC)**。
- **迁移可能会中断服务**：尽管已采取预防措施防止停机，但在计划于下周进行的 **DNS 迁移** 期间仍存在中断风险。
   - 工程师建议关注可能出现的异常行为，并警告手动操作过程可能会导致潜在问题。
- **社区请求避免在周年纪念日发生网站停机**：一位社区成员建议调整 **DNS 迁移** 时间，避开 **25 号** 的 **MCP 周年纪念日**，以防止网站停机。
   - 作为回应，工程师将尝试把迁移安排在周年纪念日之后。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1441437437627728054)** (1 messages): 

> `Tool Annotations, Model Context Protocol` 


- **提出的工具注解解决方案**：一位成员针对工具的 **Tool Annotations** 提出了一个解决方案，理想情况下可以根据参数提供不同的注解。
   - 他们正在为这个想法寻求赞助，并征求适合该主题及相关 [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1862) 的 WG 或 IG 建议。
- **为 Model Context Protocol 工具注解寻求赞助**：一位贡献者正积极为其提出的 **Model Context Protocol** 内部 **Tool Annotations** 解决方案寻求赞助。
   - 他们还请求指导，以确定合适的 Working Group (**WG**) 或 Interest Group (**IG**) 来进一步探索和开发这一主题。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1441471859999772702)** (1 messages): 

> `AI Voice Agents, Feather AI` 


- **工程师构建实时 AI 语音 Agent**：工程师们正在深入研究构建 **AI 语音 Agent**，重点关注管理 **延迟 (latency)**、确保 **平滑通话转接** 以及在实时对话中保持清晰的事件控制等挑战。
   - 他们正在寻求关于其他人如何处理实时通话流和结构化对话摘要的见解。
- **Feather AI 实验取得显著成果**：一位成员分享了他们使用 [Feather AI](https://www.featherhq.com/) 的实验，报告称即使在用户脱离脚本的情况下，也能实现亚秒级延迟和稳定的 Agent 逻辑。
   - 他们还提到了 **清晰的转录**、结构化的事件流以及将结果可靠推送到 **CRM** 等优点，并乐于了解其他替代架构、工具或工作流。