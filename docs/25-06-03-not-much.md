---
companies:
- openai
- anthropic
- google
- perplexity-ai
- bing
- playai
- suno
- hugging-face
- langchain-ai
- qwen
- mlx
- assemblyai
- llamacloud
date: '2025-06-03T05:44:39.731046Z'
description: '**OpenAI** 向具有联网功能和细粒度控制权限的 ChatGPT Plus 用户推出了 **Codex**，并改进了免费用户的记忆功能。**Anthropic
  的 Claude 4 Opus 和 Sonnet** 模型在编程基准测试中处于领先地位，而 **Google 的 Gemini 2.5 Pro 和 Flash**
  模型凭借全新的音频功能赢得了认可。**Qwen 2.5-VL** 和 **Qwen 3** 的量化版本因其多功能性和支持性而备受关注。**Bing Video
  Creator** 已在全球上线，支持文生视频功能；同时，**Perplexity Labs** 的旅游搜索需求显著增长。新的智能体 AI 工具和 RAG（检索增强生成）创新包括
  **LlamaCloud** 和 **FedRAG**。开源发布方面，包括用于网页导航的 **Holo-1** 以及 PlayAI 用于语音编辑的 **PlayDiffusion**。在音频和多模态进展方面，**Suno**
  升级了音乐编辑功能，**Google** 推出了支持 24 种以上语言的原生 TTS（文本转语音），**Universal Streaming** 则实现了超低延迟的语音转文字。**Google
  NotebookLM** 现已支持公开笔记本。*“Codex 的联网功能带来了权衡，并伴有明确的风险警告”*，以及 *“Gemini 2.5 Pro 被用户视为日常主力工具”*。'
id: MjAyNS0w
models:
- codex
- claude-4-opus
- claude-4-sonnet
- gemini-2.5-pro
- gemini-2.5
- qwen-2.5-vl
- qwen-3
- playdiffusion
people:
- sama
- gdb
- kevinweil
- lmarena_ai
- epochairesearch
- reach_vb
- wightmanr
- deeplearningai
- mervenoyann
- awnihannun
- jordirib1
- aravsrinivas
- omarsar0
- lioronai
- jerryjliu0
- nerdai
- tonywu_71
- _akhaliq
- clementdelangue
- _mfelfel
title: 今天没发生什么事。
topics:
- fine-tuning
- model-benchmarking
- text-to-video
- agentic-ai
- retrieval-augmented-generation
- open-source-models
- speech-editing
- audio-processing
- text-to-speech
- ultra-low-latency
- multimodality
- public-notebooks
---

**平静的一天**

> 2025年6月2日至6月3日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，4892 条消息）。预计节省阅读时间（以 200wpm 计算）：454 分钟。我们的新网站现已上线，支持全元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

又是一个平静的一天，伴随着一点 [Windsurf-Anthropic 争议](https://x.com/_mohansolo/status/1930034960385356174)。

AIE 第一天的 Keynotes 和 MCP 专题将在 9 小时后上线。

https://www.youtube.com/watch?v=z4zXicOAF28

---

# AI Twitter 综述

**1. AI 产品发布、功能更新和生态系统发展 (OpenAI, Gemini, Claude, Perplexity, Bing, PlayAI, Suno, Hugging Face, Google, Anthropic, Codex, LangChain, Qwen, MLX, Holo-1, Universal Streaming, NotebookLM 等)**

- **OpenAI 产品发布与 Codex 推出：** OpenAI 宣布了重大更新，包括向 ChatGPT Plus 用户推出 Codex，具备联网功能（默认关闭）、慷慨的使用限制以及细粒度的 HTTP/域名控制。Codex 现在可以更新 PR、通过语音驱动等 ([@sama](https://x.com/i/web/status/1930006856019390521), [@gdb](https://x.com/i/web/status/1929970095427858636), [@kevinweil](https://x.com/i/web/status/1929969441845952660), [@OpenAI](https://x.com/i/web/status/1929957365119627520), [@OpenAIDevs](https://x.com/i/web/status/1929956778105811071))。记忆功能（Memory）改进正向免费用户开放，具备引用近期对话的轻量级记忆能力 ([@OpenAI](https://x.com/i/web/status/1929937841905381558), [@sama](https://x.com/i/web/status/1930007155723415560))。Codex 的联网功能带来了权衡，并附带关于风险的明确警告 ([@sama](https://x.com/i/web/status/1930006856019390521))。
- **Claude、Gemini 和 Qwen 模型对比与基准测试：** Claude 4 Opus 和 Sonnet 在排行榜上持续攀升，Opus 总榜排名第 4，并在 WebDev Arena 的编程测试中并列第 1 ([@lmarena_ai](https://x.com/i/web/status/1929564515659817285))；Anthropic 在 SWE-bench Verified 上的进展也十分显著 ([@EpochAIResearch](https://x.com/i/web/status/1929568948086800798))。Gemini 2.5 Pro 被用户称为日常主力工具 ([@reach_vb](https://x.com/i/web/status/1929613466475659662), [@wightmanr](https://x.com/i/web/status/1929602224218644985))，Google 在 I/O 大会上发布了支持音频功能的 Gemini 2.5 和 Flash ([@DeepLearningAI](https://x.com/i/web/status/1929734794033660139))。Qwen2.5-VL 被公认为 Agent 和 GUI 模型的通用基础 ([@mervenoyann](https://x.com/i/web/status/1929488866748092881))，且 MLX 现已支持新的 Qwen3 量化 ([@awnihannun](https://x.com/i/web/status/1929601108210835931))。
- **Bing、Perplexity 以及搜索/视频创新：** Bing Video Creator 现已在全球范围内可用，由 Sora 驱动并支持文本转视频生成 ([@JordiRib1](https://x.com/i/web/status/1929585267528110201))。由于 Labs 查询量增加，Perplexity Labs 的需求正在激增 ([@AravSrinivas](https://x.com/i/web/status/1929974221654077725))，其旅游搜索功能也广受好评 ([@AravSrinivas](https://x.com/i/web/status/1929637229493973436))。Firecrawl 为 Agent 工作流推出了一个一键式网页搜索/抓取 API ([@omarsar0](https://x.com/i/web/status/1929931255581422030), [@LiorOnAI](https://x.com/i/web/status/1929979151014080743))。
- **Agent、RAG 和 AI 工具链：** 值得关注的 Agent 发布包括使用 LlamaCloud 构建的多 Agent 金融研究分析师 ([@jerryjliu0](https://x.com/i/web/status/1930106591132766639))、Firecrawl 的新端点以及 LangGraph 应用更新 ([@LangChainAI](https://x.com/i/web/status/1929946179564933576))。FedRAG 推出了带有 MCP 的 NoEncode RAG ([@*nerdai*](https://x.com/i/web/status/1929527011371798877))。
- **开源与机器人公告：** 用于网页导航的开源动作 VLM 模型 Holo-1 以及 WebClick 基准测试发布 ([@tonywu_71](https://x.com/i/web/status/1929890547105136882))，Hugging Face 还展示了用于机器人技术的 SmolVLA ([@_akhaliq](https://x.com/i/web/status/1929900931853816142); [@ClementDelangue](https://x.com/i/web/status/1929927844227899841))。PlayAI 开源了 PlayDiffusion，这是一种用于语音编辑的非自回归扩散模型 ([@reach_vb](https://x.com/i/web/status/1929563075696316451), [@_mfelfel](https://x.com/i/web/status/1929586464125239589))。
- **音频、视频和多模态模型能力：** Suno 发布了其音乐编辑和分轨提取（stem extraction）的重大升级 ([SunoMusic](https://x.com/i/web/status/1930007866116636735))，Google 的 Gemini 2.5 拥有支持 24 种以上语言的新原生 TTS ([@Google](https://x.com/i/web/status/1929960513779204198))，Universal Streaming 语音转文本功能以超低延迟上线 ([@AssemblyAI](https://x.com/i/web/status/1929552064566174187))。
- **NotebookLM、MLX 及其他基础设施：** Google NotebookLM 现在允许公开笔记本 ([@Google](https://x.com/i/web/status/1930005768587112755))，MLX 重点展示了在 Qwen3 235B 上的动态量化和 QLoRA ([@awnihannun](https://x.com/i/web/status/1929633379504493048))，Cline v3.17.9 引入了任务时间线导航和 CSV/XLSX 支持 ([@cline](https://x.com/i/web/status/1930069702698774833))。
- **AI 社区活动、研讨会和博览会：** AIE 博览会和世界博览会参与名额售罄，提供了在线额外环节，以及关于 Gemini 2.5、Agent 和评估的研讨会 ([@swyx](https://x.com/i/web/status/1929717401580663101), [@_philschmid](https://x.com/i/web/status/1930055992051675312))。vLLM 和 AIBrix 见面会宣布在旧金山举行 ([@vllm_project](https://x.com/i/web/status/1929952542185886184))。

**2. 研究、缩放定律 (Scaling Laws)、训练动态与模型内部机制 (GPT-4/5, RL, GRPO, RLVR, Memory, Grokking, Data Leakage, Quantization, Reasoning, Agentic Models 等)**

- **模型容量、记忆与数据泄露：** Meta 的新论文确定了 GPT 风格的 LLM 每个参数大约记忆 3.6 bits，容量呈线性扩展，并对隐私/成员推理（membership inference）产生影响 ([@jxmnop](https://x.com/i/web/status/1929903028372459909), [@scaling01](https://x.com/i/web/status/1929918033541144794))。随着数据集规模增长，成员推理变得不可能，且当数据集大小超过模型容量时会出现双重下降（double descent）。
- **推理的强化学习与 RL 训练进展：** 在 Qwen3 32B 基座模型上进行创意写作的 RL 展示了显著改进 ([@Grad62304977](https://x.com/i/web/status/1929996614883783170))，同时高熵少数 Token 被认为是推理 LLM 中有效 RL 的驱动因素，在 AIME 基准测试中获得了实质性收益 ([@iScienceLuvr](https://x.com/i/web/status/1929750117927797143), [@_akhaliq](https://x.com/i/web/status/1929900050638852479))。ProRL 和 GRPO 继续推进基于 RL 的 LLM 能力 ([@_akhaliq](https://x.com/i/web/status/1929540706374201756))。
- **记忆架构与持续学习：** Google ATLAS 引入了具有可学习状态的“主动记忆（active memory）”和用于更精准更新的 Muon 优化器 ([@TheTuringPost](https://x.com/i/web/status/1929992259019432115))；ChatGPT 的记忆系统被视为 Agent 应用的关键差异化因素 ([@karpathy](https://x.com/i/web/status/1930003172246073412), [@hkproj](https://x.com/i/web/status/1930005251039637836))。RLVR 和 post-training 机制被讨论为数学/编程改进的关键 ([@lateinteraction](https://x.com/i/web/status/1930045203681030248))。
- **模型推理、CoT 与可解释性：** Chain-of-Thought (CoT) 推理中的枢轴 Token（Pivot tokens）和熵正在被积极研究，RL 很大程度上是在调整高熵 Token 的熵 ([@teortaxesTex](https://x.com/i/web/status/1929755590404055358), [@iScienceLuvr](https://x.com/i/web/status/1929750117927797143))。自我挑战 Agent 使用自我生成的任务和验证器来提升工具使用（tool-use）能力 ([@jaseweston](https://x.com/i/web/status/1929719473952497797))。
- **Grokking、扩展与学习动力学：** 探索了 Grokking 中的相变和累积学习机制 ([@raphaelmilliere](https://x.com/i/web/status/1929887222553002193))，元学习（meta-learning）和 RL 环境的扩展被认为是解锁持续适应的关键 ([@tamaybes](https://x.com/i/web/status/1929683184163447141))。涵盖了关于临界批次大小（critical batch size）和 Muon 等优化器的新实证方法 ([@eliebakouch](https://x.com/i/web/status/1930081408657051745))。
- **量化与效率：** MLX 的动态量化方法在不增加体积的情况下为 Qwen3 模型带来了更好的质量 ([@awnihannun](https://x.com/i/web/status/1929633379504493048))，FP8 被提议作为图像/视频生成的最佳模式 ([@RisingSayak](https://x.com/i/web/status/1929597236356530560))。
- **提示词、DSPy 与编程范式：** DSPy 被定位为提示词和工作流的关注点分离（separation-of-concerns）范式，而不仅仅是提示词优化 ([@lateinteraction](https://x.com/i/web/status/1929559952009670797))。Prompt engineering 被批评为一种抽象 ([@lateinteraction](https://x.com/i/web/status/1929573102675177820))。
- **方法驱动 vs. 问题驱动研究：** 方法驱动的研究（如 AlphaEvolve 的优化）被强调在 LLM 时代日益主导问题驱动的方法 ([@_jasonwei](https://x.com/i/web/status/1929621539881996607))。

**3. 模型/平台对比、用户体验与评估实践**

- **模型路由与 UX 建议：** 关于针对不同任务选择哪种 ChatGPT 模型的详尽指南和个人经验法则——o3 用于难题，4o 作为主力模型，o4-mini 用于搜索/分析，4.1 用于编程 ([@karpathy](https://x.com/i/web/status/1929597620969951434), [@scaling01](https://x.com/i/web/status/1929617054887563492), [@aidan_mclau](https://x.com/i/web/status/1929720423484399632))。Gemini 2.5 Pro 和 Claude 4 被列为编程和头脑风暴的主力工具 ([@reach_vb](https://x.com/i/web/status/1929613466475659662), [@wightmanr](https://x.com/i/web/status/1929602224218644985))。
- **Agent 助手与自动化：** 以文档为中心的工作流越来越多地使用自动化 Agent 进行端到端批处理，而非助手式的 UX ([@jerryjliu0](https://x.com/i/web/status/1929987006593151164))。
- **安全与 UX 问题：** 讨论并解决了围绕仓库分叉（repo forking）、GitHub 权限以及 OpenAI 界面清晰度的问题 ([@andersonbcdefg](https://x.com/i/web/status/1930025753443479665))。
- **评估（Evals）与评估会议：** 评估（Evals）现已成为核心学科，并为从业者设有专门的分论坛 ([@swyx](https://x.com/i/web/status/1929609793104499152))。Stripe 的评估强调了针对 Agent 性能的 A/B 测试 ([@OpenAIDevs](https://x.com/i/web/status/1929632332837015833))。
- **Prompt Engineering 与记忆功能使用争议：** 用户在争论开启 ChatGPT 记忆功能的价值，有人更倾向于“原生能力（raw capabilities）”，而另一些人则强调其对产品/UX 的重要性 ([@Yuchenj_UW](https://x.com/i/web/status/1930007834374353245), [@sjwhitmore](https://x.com/i/web/status/1930053753807483216))。
- **搜索、检索与 RAG 实践：** ColQwen2 进入 Hugging Face transformers 用于视觉文档检索，提升了 RAG 流水线的性能 ([@mervenoyann](https://x.com/i/web/status/1929563866658218316), [@tonywu_71](https://x.com/i/web/status/1929537720897958018))。

**4. 社会、监管与战略考量（AI 红线、开源、政策、安全、生态系统、AGI、教育）**

- **AI 红线与威慑：** 关于国际 AI 红线的战略提案集中在防止智能爆炸和恶意使用（例如 AI 病毒学家或网络 Agent），强调透明度和验证而非僵化的定义 ([@DanHendrycks](https://x.com/i/web/status/1929901133117415474), [@DanHendrycks](https://x.com/i/web/status/1929713070265516459), [@DanHendrycks](https://x.com/i/web/status/1929709721290002497))。
- **开源倡导、伦理与可访问性：** 为机器人技术和透明度开源发布了 VLA 模型（Vision-Language-Action） ([@ClementDelangue](https://x.com/i/web/status/1929927844227899841))，用于语音编辑的 PlayDiffusion，以及用于 LLM 预训练的 Common Corpus（约 2T tokens） ([@iScienceLuvr](https://x.com/i/web/status/1929751110723805525))。
- **生态系统与平台成熟度：** 语音/音频 AI 领域的开放科学和成熟度受到赞誉 ([@reach_vb](https://x.com/i/web/status/1929566647578251494))，像 @LawZero_ 这样专注于“安全设计（safe-by-design）”AI 的新机构也已成立 ([Yoshua_Bengio](https://x.com/i/web/status/1929843757219766743))。
- **AI 在教育与工作中的应用：** 呼吁赋能每一个人——无论是工程师还是非工程师——使用 AI 工具进行编程 ([@AndrewYNg](https://x.com/i/web/status/1929906213208113409))，cs224n 2024 课程涵盖了预训练、后训练和推理 ([@stanfordnlp](https://x.com/i/web/status/1929557373213118869))。

**5. 行业、硬件与市场趋势（Nvidia、硬件、机器人、Apple、Databricks、Snowflake 等）**

- **Nvidia Blackwell 与硬件加速：** Nvidia B200 和 Blackwell 芯片现在为 DeepSeek R1 提供服务，吞吐量高达 H100 的 5 倍 ([@scaling01](https://x.com/i/web/status/1929670236057264354), [@ArtificialAnlys](https://x.com/i/web/status/1929666713601429964))，Figure-01 与 Figure-02 人形机器人展示了工程上的跨越式进步 ([@adcock_brett](https://x.com/i/web/status/1929955468040122835), [@adcock_brett](https://x.com/i/web/status/1929575124556304564))。
- **Apple、数据与市场动向：** 预计 Apple 的 WWDC 会在 AI 方面令人失望，Oracle 的市值在 AI 平台讨论中表现惊人 ([@TheRundownAI](https://x.com/i/web/status/1929485774392594713), [@sarahcat21](https://x.com/i/web/status/1929965018629713940))。
- **去中心化计算与云：** DeepSeek-R1-0528 通过去中心化计算实现了 100% 的运行时间，表现优于其他供应商 ([@jon_durbin](https://x.com/i/web/status/1929639699171495936))。Google Cloud Run 为所有人提供无服务器 GPU，且无配额限制，支持对 Gemma 等模型的按秒计费 L4 访问 ([@_philschmid](https://x.com/i/web/status/1929638760758874428))。

**6. 梗、幽默与文化评论**

- **AI、模型和行业梗：** 值得关注的梗包括 OpenAI 董事会大戏的电影改编 ([iScienceLuvr](https://x.com/i/web/status/1929982449720930755))、关于“厨房费”和额外费用的幽默调侃 ([@Yuchenj_UW](https://x.com/i/web/status/1929633411251241429))、“为什么马斯克喜欢这个 lol” ([@Yuchenj_UW](https://x.com/i/web/status/1929523323337076963))，以及对模型功能和行业怪象的讽刺评论 ([@vikhyatk](https://x.com/i/web/status/1929449598524821509), [@skalskip92](https://x.com/i/web/status/1929572784599859220))。
- **编程与工程幽默：** 关于版本控制、调试和代码审查文化的笑话 ([@hyhieu226](https://x.com/i/web/status/1930132944502567301), [@HamelHusain](https://x.com/i/web/status/1929555537785708727))。
- **流行文化、体育及杂项：** 关于国际象棋、航空和网红文化的评论与 AI 话题交织在一起 ([@demishassabis](https://x.com/i/web/status/1929659054349340829), [@TomLikesRobots](https://x.com/i/web/status/1929901616536170949))。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. AI 模型与基础设施开源发布

- [**Google 开源 DeepSearch 技术栈**](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) ([Score: 840, Comments: 77](https://www.reddit.com/r/LocalLLaMA/comments/1l27g8d/google_opensources_deepsearch_stack/)): **Google 开源了 DeepSearch，这是一个使用 Gemini 和 LangGraph 框架构建 AI Agent 的演示技术栈，详情见 [google-gemini/gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 仓库。虽然该技术栈与 Gemini 用户端应用所使用的并不相同，但它旨在通过提供模块化的后端/前端组件、容器化（Docker）以及展示基于 LLM 的应用的集成工作流，来加速 Agent 的开发。通过替换相关模块，可以将其适配到其他模型（如 Gemma 或不同的搜索工具）；其架构简单直接，适合快速原型设计。** 评论者强调 DeepSearch 是一个结构良好的演示，而非生产级基础设施，并建议在复杂场景下使用更高级的替代方案，如 [LangManus](https://github.com/Darwin-lfl/langmanus/tree/main)。LangGraph 的使用被强调为一种灵活的模式，人们对 Google 最近的开源发布以及 Gemma 模型的性能表现出显著的热情。
    - 作者澄清开源的 DeepSearch 技术栈与 Gemini App 中使用的不同。它利用了 LangGraph，使其具有模块化特性——开发者可以更换 Gemini 特定的部分（例如，用 Gemma 等开源模型替换 Gemini）；然而，由于搜索功能未解耦，因此需要替代工具。
    - 多位评论者指出，虽然后端架构整洁且具有教育意义，但并不新颖或特别复杂。他们建议参考像 LangManus (https://github.com/Darwin-lfl/langmanus/tree/main) 这样的项目作为更复杂的 LangGraph 系统示例，强调 DeepSearch 最好被视为一个构建良好的演示，而不是一个生产就绪或突破性的技术栈。
    - 对 Google 最近的开源模型有积极的技术反馈，其中 Gemma 3 4B 被认为是小型 LLM 中表现尤为强劲的一款。
- [**nvidia/Nemotron-Research-Reasoning-Qwen-1.5B · Hugging Face**](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B) ([Score: 133, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l2820t/nvidianemotronresearchreasoningqwen15b_hugging/)): **NVIDIA 的 Nemotron-Research-Reasoning-Qwen-1.5B 是一款开源权重的 15 亿参数 LLM，针对复杂推理（数学、编程、STEM、逻辑）进行了优化。它通过 ProRL（延长强化学习）进行训练，该方法通过减轻熵坍缩、DAPO（解耦剪裁和动态采样策略优化）以及带有参考策略重置的 KL 正则化等创新扩展了 RL 训练。基准测试显示，Nemotron-Qwen-1.5B 的表现显著优于 DeepSeek-R1-1.5B——在 pass@1 准确率上分别提升了 +14.7%（数学）、+13.9%（编程）、+54.8%（逻辑）、+25.1%（STEM）和 +18.1%（指令遵循）——并且达到或超过了 DeepSeek-R1-7B 的水平，详情见 Hugging Face 发布页面 (https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)。GGUF 权重已提供 q4、q8 和 f16 版本 (https://huggingface.co/stormchaser/Nemotron-Research-Reasoning-Qwen-1.5B-GGUF/tree/main)。** 评论者注意到小型、侧重边缘计算的模型在实际应用中的可行性日益增强，并强调了在移动设备上运行的 Nemotron 及其同类模型（如 Gemma、Qwen3）。一些人对 NVIDIA 限制性的许可协议（CC 非商业性使用）表示技术上的不满，这限制了其在现实世界中的实用性和商业化潜力。

- Nemotron-Research-Reasoning-Qwen-1.5B 模型利用了一种新颖的 Prolonged Reinforcement Learning (ProRL) 算法，该算法专为更长的 RL 训练周期设计，旨在实现更深层次的任务探索和更好的泛化能力。它引入了三项核心技术：缓解 entropy collapse、解耦 clip 与 dynamic sampling policy optimization (DAPO)，以及带有 reference policy reset 的 KL regularization——针对该小规模 LLM 场景适配了 Group Relative Policy Optimization (GRPO)。
- 基准测试表明，Nemotron-Research-Reasoning-Qwen-1.5B 虽然体积更小，但其表现超越了 DeepSeek-R1-1.5B，并能与 DeepSeek-R1-7B 竞争。据报告，其相对于 DeepSeek-R1-1.5B 的平均 pass@1 提升在数学领域为 `14.7%`，编程领域为 `13.9%`，逻辑谜题领域为 `54.8%`，STEM 推理领域为 `25.1%`，指令遵循任务中为 `18.1%`，这表明 sub-3B 模型在复杂推理能力方面实现了重大飞跃。
- 该模型是权重开放的（open-weight），但根据 Creative Commons Non-Commercial 许可证发布，这限制了其主要用于研究和非商业开发。一些评论者指出，Nvidia 发布的其他模型也附带了类似的限制性或可撤销许可证，这可能会削弱其在生产或商业领域的广泛采用。

### 2. 模型行为与偏差的前沿研究

- [**META 新论文 - 语言模型记忆了多少？**](https://arxiv.org/abs/2505.24832) ([Score: 176, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1l2gvar/new_meta_paper_how_much_do_language_models/)): **这篇 arXiv 论文 ["How much do language models memorize?"](https://arxiv.org/abs/2505.24832) 严谨地量化了 GPT 风格 Transformer 的存储容量，报告的经验记忆量约为每参数 3.6 bits（例如，bfloat16 为 3.51 bits，float32 为 3.83 bits）。它指出记忆发生在容量阈值之前，随后进入 ‘Grokking’（顿悟）阶段，模型开始通过编码更广泛的模式而非特定实例细节来进行泛化。论文将这一转变与损失曲线中 Double Descent（双重下降）的开始联系起来——具体而言，当训练数据信息超过模型容量时，必须进行跨实例的信息共享并提高泛化能力。从数百个 Transformer（50 万至 15 亿参数）推导出的 Scaling Laws 预测，经过彻底去重的大型 LLM 对成员推理和逐字提取攻击的鲁棒性越来越强，提取出的知识应归功于泛化而非记忆。** 评论中的技术讨论提出了这些发现如何扩展到 MoE 架构、训练变化（如量化感知训练、低精度方案）以及像 BitNet 这样的替代模型的问题——特别是 ~3.5 bit/parameter 的障碍在这些设置下是保持不变还是会发生偏移。此外，还讨论了低于此阈值的量化如何从根本上限制 GPT 风格模型的生成能力。
    - 论文量化了 Transformer 模型容量，发现 GPT 风格的 LLM 每参数可存储约 3.5–4 bits（bfloat16 为 3.51，float32 为 3.83），且数值精度翻倍并不会使存储容量翻倍，这表明容量并不严格取决于参数精度。
    - 对记忆与泛化动态的分析表明，语言模型最初会记忆训练数据直到达到容量上限，之后会转向（‘Grokking’）更广泛的泛化，这与观察到的 Double Descent 效应相关，即当数据集信息超过模型存储时，模型必须开始泛化。
    - 扩展这一框架，评论者提出了关于结果如何扩展到更大规模或 MoE 模型以及对量化模型的影响：如果量化降至 ~3.5 bits 以下，可能会出现剧烈的输出退化，这可能限制了低于 3.5 bits 的量化技巧；此外，关于这些发现在模型/数据集规模超过研究的 50 万至 15 亿参数范围时如何外推，仍存在技术争论。
- [**视觉语言模型（VLM）存在偏差**](https://vlmsarebiased.github.io/) ([Score: 100, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1l2b83p/vision_language_models_are_biased/)): **最近的一项研究（[分析链接](https://vlmsarebiased.github.io/)）发现，最先进的视觉语言模型（VLM）表现出高度偏差：虽然它们在规范图像（如 4 条腿的狗、3 条杠的 Adidas 标志）上达到了 100% 的计数准确率，但在面对反事实或非典型图像（如 5 条腿的狗）时，准确率大幅下降至约 17%。这表明模型依赖于记忆的训练集知识而非实际的视觉分析，这种确认偏差在表现最好的 VLM、任务和领域中普遍存在。研究进一步指出，无论 Prompt Engineering 如何，偏差依然存在，这表明存在影响新特征视觉推理的结构性模型限制。** 几位评论者指出，这一结果并不令人意外，将偏差归因于训练数据的固有统计形态，并重申所有 AI 系统都反映了数据集和社会偏差。一些人通过类比 LLM 中类似的语言偏差进行了扩展，例如“我最喜欢的菜系是”的补全显示出对“意大利菜”等常见选项的强烈偏好。
    - 最高赞评论提供了具体的 Benchmark 风格细节：VLM 在熟悉图像上几乎完美（约 100% 准确率，如计算 Adidas 标志的线条或标准动物的腿），但在具有异常或反事实特征的图像上（如 5 条腿的狗或 4 条杠的类 Adidas 标志），性能骤降至约 17% 的准确率。这突显了在 OOD（分布外）场景中强大的先验偏差和有限的泛化能力。
    - 值得注意的是，此类 VLM 失效不仅限于标志和动物，还包括手指数量非标准的手部等情况，这意味着在 OOD 或反事实推理方面存在普遍困难。这表明模型可能严重过拟合于其训练数据中常见的视觉分布。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI 模型访问不平等与经济影响辩论

- [**Dario Amodei 担心由于 AI 导致的失业，普通人将失去经济杠杆，这将破坏民主并导致权力高度集中：“我们需要拉响警报。我们可以阻止它，但不能仅仅通过说‘一切都会好起来的’。”**](https://v.redd.it/ba6dzs1grq4f1) ([Score: 1378, Comments: 364](https://www.reddit.com/r/singularity/comments/1l2gwo1/dario_amodei_worries_that_due_to_ai_job_losses/)): **Anthropic 首席执行官 Dario Amodei 警告说，广泛的 AI 驱动的职业取代可能会剥夺普通人的核心经济权力，从而面临民主机制崩溃和严重权力集中的风险。他强调，减轻这些影响需要紧急的系统性干预，而不是自满，正如本次讨论和 AI 领导者过去的警告中所详述的那样。该主题中的技术评论指出了“温水煮青蛙”效应——缓慢、渐进的变化使这些风险在为时已晚之前变得不那么明显且难以采取行动，这与大流行应对的历史性失败相似。** 该主题的核心观点强调了社会对系统性威胁的反应长期缓慢（“温水煮青蛙效应”），并认为有意义的行动通常只有在危机发生后才会触发，这反映了即使在技术社区中，人们对主动缓解措施也持怀疑态度。
    - 一条评论将 AI 驱动的职业自动化渐进式影响比作“温水煮青蛙”问题，认为由于 AI 是逐步取代工作而非一次性取代，公众和政策反应滞后——这使得主动解决技术性失业变得困难。
    - 几位用户讨论了系统性风险，这与 Dario Amodei 的警告一致，即 AI 可能会削弱普通人的经济杠杆，除非采取重大的预防行动，否则由于权力集中度增加，可能会破坏民主的稳定。
- [**Dario Amodei 担心由于 AI 导致的失业，普通人将失去经济杠杆，这将破坏民主并导致权力高度集中：“我们需要拉响警报。我们可以阻止它，但不能仅仅通过说‘一切都会好起来的’。”**](https://v.redd.it/8sjr32hnrq4f1) ([Score: 293, Comments: 77](https://www.reddit.com/r/ClaudeAI/comments/1l2gxo8/dario_amodei_worries_that_due_to_ai_job_losses/)): **Dario Amodei（Anthropic 首席执行官）警告说，大规模 AI 驱动的职业取代可能会削弱普通大众的经济杠杆，破坏民主结构并巩固少数实体的权力。他敦促立即采取非自满的政策和社会反应，将风险定义为不仅是经济上的（失业），而且是系统性的（权力集中）。热门评论提到了历史上的社会稳定性、对失去可预测性的共同焦虑，以及关于新法团主义或新封建主义结果的理论（Yarvin, Thiel, Andreesen）——例如雇主城镇以及对 UBI 支持的高度集中的公司管辖权的依赖。** 具有技术意识的评论者呼应了这种紧迫性，指出了大规模焦虑可能导致的社会动荡，并将其与推测性的社会经济模型进行了类比，在这些模型中，公司权力取代了传统的治理，进一步凸显了通过 AI 引发的劳动力中断加剧不平等的风险。
    - Medical_Mine1275 讨论了 AI 驱动的失业可能带来的社会经济影响，引用了 Curtis Yarvin 的理论（也与 Peter Thiel 和 Marc Andreessen 有关），即向高度集中的公司权力转移，让人联想到经典的“公司镇（company towns）”。该评论提出了这样一种情景：在 AI 的赋能下，公司削弱了工人的议价能力——用 UBI 支持的、高度公司化的生活（例如 Tesla 微型住宅、无人机配送的食物）取代他们，并将人们推向虚拟经济作为物质财富的替代品。这描绘了一个技术细节详尽的愿景，即 AI 引发的权力转移、经济依赖以及可能回归新封建治理结构。

- [**我们需要竭尽全力防止 AI 成为奢侈品**](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/) ([得分: 222, 评论: 94](https://www.reddit.com/r/singularity/comments/1l2j6u1/we_need_to_do_everything_in_our_power_to_prevent/)): **该帖子强调了顶级 LLM 访问权限正逐渐出现付费墙（OpenAI: $200/月，Anthropic: $100/月，Google: $130/月），并表达了对开源 LLM 的担忧。虽然目前开源模型表现强劲（如 DeepSeek, Qwen），但随着规模增长，它们将需要极其昂贵的 GPU，这可能导致进一步的私有化并加剧不平等。作者断言，模型规模的进步正在超越消费级硬件的发展，使得本地推理（local inference）变得不再可行，并预见到中国开源贡献者可能会进行整合或货币化，预测高端私有模型与公众访问之间的差距将危险地扩大，如果不加以解决，将产生重大的社会影响。** 关键的技术回应认为，根本问题在于训练和运行大规模 LLM 的内在成本，而非人为的稀缺，并建议将社会化（公共资金/补贴）作为实现公平访问的潜在路径。其他人将 AI 访问类比为公用事业（如电力），暗示高性能使用需支付溢价，或者指出较低层级的模型对大多数用户来说在功能上依然稳健，只是排除了最前沿的进展。
    - 几位评论者讨论了开发和运行先进 AI 模型涉及巨大且不可避免的成本，引用了 OpenAI 和 Google 等主要供应商正面临亏损或将 Pro 计划等产品价格提高到每月 250 美元的观察结果。这被视为反驳当前价格欺诈论点的证据，竞争格局表明极端定价主要归因于高昂的运营支出，而非垄断行为。
    - 一位评论者关注到分层 AI 访问：SOTA 模型的大多数核心功能在较低层级即可使用，只有寻求最新或最先进功能的用户才会为早期访问支付溢价。这种模式遵循常见的行业模式（例如其他市场中的经济型与豪华型），而非专门限制 AI 的访问。
    - 提出了将 AI 社会化的建议，强调鉴于短期内成本难以降低，广泛且公平的访问需要从社会层面分配这些成本（例如通过公共资金或类似公用事业的社会化），而不是期望快速的技术或经济变革能立即让先进 AI 普及。
- [**前 OpenAI AGI 准备负责人：“到 2027 年，几乎所有可以在计算机上完成的有经济价值的任务，都将由计算机更有效、更廉价地完成。”**](https://i.redd.it/l0cd9s4yar4f1.png) ([得分: 1026, 评论: 356](https://www.reddit.com/r/singularity/comments/1l2jun4/former_openai_head_of_agi_readiness_by_2027/)): **这张图片是前 OpenAI AGI 准备负责人 Miles Brundage 的一条推文，预测到 2027 年，几乎所有由计算机执行的有经济价值的任务都将由 AI 更高效、更具成本效益地执行。Brundage 澄清说，这一预测指的是技术可行性（“将可以做到”）和孤立的输出质量，而不一定是广泛的采用或人类价值的考量。该帖子将其视为 AI 预期快速进展的一个指标，并对采用和人类偏好提出了限定性告诫。** 评论提出了技术批评，强调了 (1) 即使技术能力存在，组织和数据瓶颈也会阻碍 AI 的快速采用；(2) 在不了解实际情况的情况下高估了计算机对工作的替代；(3) 鉴于美国白领工作的普遍性，迫切需要解决潜在的社会影响（如 UBI、自动化税）。这些反映了对实际实施时间表和更广泛社会后果的怀疑。
    - Fenristor 强调了快速采用 AI 的一个主要实际限制：*大多数公司缺乏可通过程序访问的高质量数据*。即使付出巨大努力，在 2027 年之前将遗留系统和非结构化数据转换为适合自动化的格式可能也不可行，这表明广泛自动化有经济价值的计算机任务的时间表存在严重约束。
    - ryanhiga2019 指出了 LLM 当前的一个技术局限：*持续的幻觉（hallucination）和不可靠性*。除非解决模型事实性和鲁棒性（robustness）的基础问题，否则在建议的时间范围内，扩大 LLM 规模以处理关键任务或有经济价值的任务可能无法实现。

### 2. 主要 AI 模型发布和新功能推出（2025 年春夏）

- [**据报道，Apple 在内部基准测试中测试了性能媲美 ChatGPT 的 AI 模型**](https://the-decoder.com/apple-reportedly-tests-ai-models-that-match-chatgpts-capabilities-in-internal-benchmarks/) ([Score: 290, Comments: 119](https://www.reddit.com/r/singularity/comments/1l2dsf8/apple_reportedly_tests_ai_models_that_match/)): **据报道，Apple 正在测试参数量高达 150B 的大规模内部 LLM，其基准测试表现与 ChatGPT 持平。但这些模型面临高昂的推理成本和尚未解决的技术/安全障碍，因此无法公开发布。相反，Apple 计划在 WWDC 2025 上为第三方开发者发布规模显著更小的设备端 Foundation Models（约 3B 参数），仅提供基础的 ML 能力；由于内部审慎和技术限制，对话式 Siri 等高级功能已推迟到 2026 年以后。参见[详细报告](https://the-decoder.com/apple-reportedly-tests-ai-models-that-match-chatgpts-capabilities-in-internal-benchmarks/)。** 评论者对 Apple 在实际 AI 部署方面的滞后表示怀疑，指出 Siri 的表现甚至不如 Google 较旧的助手，并质疑扩展测试中的 150B 参数模型的成本效益和实用性。
    - 据报道，Apple 的内部 AI 模型参数量高达 150B，并在某些基准测试中与 ChatGPT 持平，但有迹象表明每个 token 的运营成本极高，且存在阻碍公开发布的**技术限制**，暗示了大规模部署/推理方面的担忧。
    - 讨论强调了对“ChatGPT 能力”含义的怀疑，考虑到不同模型之间巨大的性能差异（例如 GPT-4o-mini 与 GPT-3.5 或 GPT-4 的对比），并暗示 Apple 的努力可能并非针对能使公开部署变得可行的最先进、最具成本效益的性能水平。
    - 评论者指出，即使 Apple 在内部实现了 ChatGPT 级别的技术基准，终端产品（如 Siri）中的实际 AI 集成仍然匮乏，这表明重点应放在部署、成本和实际功能上，而不仅仅是参数量或封闭的基准测试。
- [**Microsoft 将免费的 Sora AI 视频生成引入 Bing**](https://www.windowscentral.com/microsoft/microsoft-bing-video-creator-sora-ai-generator-free-announcement) ([Score: 245, Comments: 51](https://www.reddit.com/r/singularity/comments/1l264o6/microsoft_brings_free_sora_ai_video_generation_to/)): **Microsoft 已将 OpenAI 的 Sora 视频生成集成到 Bing 应用中（更名为 Bing Video Creator），提供免费的 AI 视频生成功能，尽管目前仍没有独立的 Sora 应用或 ChatGPT 应用集成。初步的用户报告强调了基础生成能力（如慢动作 gif），但指出内容安全过滤器非常严格，经常拦截请求。** 专家们正在讨论这种有限的推广方式——仅在 Bing 上提供，而非作为专门的 Sora 产品或 ChatGPT 的一部分——并强调了对过度激进的安全过滤降低实际可用性的担忧。
    - 技术用户强调，现作为 Bing Video Creator 提供的 Sora 仍缺乏独立应用或 ChatGPT 应用内的集成，与潜在竞争对手相比，这限制了其可访问性。
    - 一些评论者对 Sora 的内容安全过滤器表示不满，报告称经常出现“请求被拦截”的响应，这可能会阻碍技术用户的实验和创意应用。
    - 多位用户将 Sora 与 Google 的 Veo3 模型进行了对比，认为 Veo3 产生的视频生成效果显著更好，并暗示 Microsoft 目前提供的产品在输出质量和能力方面处于落后地位。

- [**OpenAI 准备发布 2 个具有原生音频支持的新模型**](https://x.com/testingcatalog/status/1929949017472930181?s=46) ([Score: 229, Comments: 31](https://www.reddit.com/r/singularity/comments/1l2htv5/openai_is_preparing_to_release_2_new_models_with/)): **OpenAI 正在准备发布两个模型——gpt-4o-audio-preview-2025-06-03 和 gpt-4o-realtime-preview-2025-06-03——两者都具有原生音频支持。这些模型似乎通过提供集成的音频输入/输出能力，扩展了 GPT-4o 的多模态流水线，可能在 LLM 框架内实现低延迟实时音频交互和音频数据处理。关于架构变化或相对于现有 GPT-4o（已具备某些音频模态）的改进细节尚未公布，但“realtime”命名暗示了语音助手的亚秒级响应。[查看来源。](https://x.com/testingcatalog/status/1929949017472930181?s=46)** 评论者质疑这些模型与现有 GPT-4o 模型之间的区别，询问什么才算作“原生音频”，以及这些发布是否就是之前演示过、备受期待的对话式音频助手，这表明“原生”的定义以及针对音频任务的模型改进细节仍存在歧义。
    - 几位用户正在讨论“原生音频”的含义，指出 GPT-4o 在发布时已经演示了原生音频输入和输出。这引发了关于即将推出的模型相比现有产品（如 **GPT-4o 的实时音频/语音多模态**）可能带来哪些新能力的疑问。
    - 一种假设是，这些模型可能是早期公开演示中预览的音频助手功能的演进，可能预示着*增强的对话或低延迟音频处理*。技术社区正在等待关于“原生”与之前音频处理方法有何不同的澄清，特别是在架构或延迟改进方面。
    - 一位用户提到有兴趣将这一概念扩展到作为连续比特流的视频，这表明技术上对将音频和视频作为统一原生流处理的模型有需求，以用于实时助手或生成任务。这指向了人们对超越当前独立模态 Tokenization 的真正多模态、连续输入架构的持续关注。
- [**Memory 功能现已向免费用户开放！！！**](https://i.redd.it/jy18jpn0nq4f1.png) ([Score: 235, Comments: 57](https://www.reddit.com/r/OpenAI/comments/1l2g8es/memory_is_now_available_to_free_users/)): **图片是一份 ChatGPT 官方公告，透露 Memory 功能（此前仅限付费用户）从 2025 年 6 月 3 日起向免费用户开放。公告指出，免费用户将获得 Memory 的“轻量级版本”，系统会参考最近的对话以提高回复的相关性。存在地区差异（在部分欧洲地区需手动选择加入），用户保留禁用或管理 Memory 的能力。[查看公告图片。](https://i.redd.it/jy18jpn0nq4f1.png)** 热门评论讨论了隐私和数据使用：付费用户注意到能够关闭训练数据使用的价值（尽管对执行情况持怀疑态度）。其他人批评了该功能，警告自动 Memory 可能会引入偏见或过时信息，并要求更精细的手动 Memory 控制。
    - 一些用户对 ChatGPT 的 Memory 实现在技术层面的运作方式表示担忧，指出它收集聊天记录以丰富用户 Prompt，作为一种自动化知识库。这可能会导致模型做出毫无根据的假设，或转向个性化但事实不准确的回答，而不是公正的信息。
    - 对新 Memory 功能的有效性和保真度存在批评性评估：用户报告称它有时会保留无关或过时的细节，并且对缺乏手动管理存储“记忆”的控制感到沮丧。与旧机制或手动替代方案相比，自动 Memory 系统被认为在召回重要细节方面不太可靠。
    - 一位付费订阅者提出了数据隐私点：Plus 用户可以选择禁用其数据用于未来的模型训练，这被定位为与免费层级的主要区别。然而，OpenAI 在实践中是否完全遵守这一政策被视为一个透明度问题。

- [**Research 功能现已在 Pro 计划中推出！！**](https://i.redd.it/b1x3zdboxq4f1.png) ([Score: 135, Comments: 39](https://www.reddit.com/r/ClaudeAI/comments/1l2hsjw/research_is_now_available_on_pro_plans/)): **该图片展示了一个用户界面更新，为 AI 助手平台上的 Pro 计划用户引入了“Research”功能（标记为 BETA），其目标可能是直接在对话环境中增强基于 Web 的研究能力。评论中的技术讨论集中在该研究功能的性能上，将其与竞争对手进行对比，并指出定性差异：用户报告该模式提供的是富含上下文、定制化的见解，而非直接答案（例如，提供食谱开发的建议，而不仅仅是复制食谱），这可能表明更细致的信息综合能力。** 评论者对研究功能的深度和适应性表达了积极印象，指出其在处理微妙查询时的实用性以及呈现结果的非公式化方法。人们对与其他 AI 提供商类似功能的对比基准测试也存在一些好奇。
    - 一位用户观察到系统部署了 3-4 个子 Agent，并采用深度优先方法进行研究，这表明可能采用了多 Agent 架构或并行上下文收集，这可能会影响综合输出的广度和深度。
    - 一项对比强调，系统在单次研究任务中引用了“300 多个来源且仍在增加”，这明显高于 GPT 或基于 Perplexity 的系统所报道的数量，意味着更广的覆盖范围和潜在更丰富的信息综合。
    - 对比评论指出，Claude Max 和 SuperGrok 被评为研究任务的顶级表现者，Gemini 被描述为过于冗长，而 OpenAI 模型提供的信息感觉过于疏离，这指向了领先模型在检索风格、答案综合和 UX 方面的差异。

### 3. 使用 Veo 3 和 AI 视频的创意用途与生产突破

- [**巴西乌利亚诺波利斯市政府使用 VEO 3 制作了完整的广告片，仅花费 300 雷亚尔（52 美元）的 VEO 3 积分**](https://v.redd.it/36cgd4rvjp4f1) ([Score: 1047, Comments: 196](https://www.reddit.com/r/singularity/comments/1l2azl6/ulianopolis_city_hall_in_brazil_made_a_complete/)): **巴西乌利亚诺波利斯市政府使用 Google 的 VEO 3 生成式视频模型，仅花费 300 雷亚尔（52 美元），就制作了一段完整的、专业级的 1 分钟广告视频，这与传统制作流程所需的 100,000 雷亚尔（17,543 美元）形成了鲜明对比。该视频（由其创作者在 Instagram 上引用）不仅展示了高分辨率的视觉效果和叙事连贯性，还展示了先进的语言本地化（支持巴西葡萄牙语方言和口音），证明了 VEO 3 先进的多模态综合能力以及 AI 对生产成本造成的快速下行压力。这凸显了一次重大的行业颠覆：此类生成式工具可以绕过大型、多角色的创意团队，大幅降低成本和制作复杂度。** 评论强调了这种范式转移，认为生成式 AI 工具将显著取代传统的广告代理公司和制作团队，尤其是因为迭代 Prompt 和编辑的成本与难度要低好几个数量级。一个关键的技术印象是输出中语言和文化本地化的自然性与真实性，特别是考虑到非英语输出通常是生成式模型面临的一个挑战。
    - 多位评论者强调了使用 **VEO 3** 生成高质量、本地化广告的成本效率——指出 300 雷亚尔（52 美元）的积分支出远低于传统的广告制作方法，后者涉及雇佣和物流成本。
    - 技术讨论指向了 AI 生成的葡萄牙语语音的高质量，包括当地口音，用户对其自然度和流畅度印象深刻，提高了商业和政府背景下 AI 本地化能力的门槛。
    - 一位评论者强调，使用母语（此处为葡萄牙语）生成的 AI 内容尤其令人震惊，因为实现准确的方言和口音在历史上一直是合成媒体的弱点，这表明 VEO 3 在多语言或口音感知综合方面取得了显著进展。

- [**粉丝们应得的绿巨人 vs 灭霸之战，由 Veo 3 生成。**](https://v.redd.it/j71o4jmc7q4f1) ([Score: 945, Comments: 336](https://www.reddit.com/r/aivideo/comments/1l2e03i/the_hulk_vs_thanos_fight_that_the_fans_deserved/)): **一位用户分享了使用 Veo 3 生成的 YouTube 视频，据称该视频利用 Google 的 Veo 视频生成模型描绘了一场更高质量的绿巨人对阵灭霸的战斗（见 [YouTube 链接](https://youtu.be/ILh1P_pI3k4)）。Veo 以生成高保真、文本转视频（text-to-video）内容而闻名；然而，讨论中指出了 AI 驱动的打斗编排和物理真实感方面的局限性，特别是在具有动态角色运动的复杂场景中。讨论强调了对真实演员动作参考的需求，将其作为增强现实感的潜在解决方案。** 评论者强调，目前的 AI 方法（如 Veo）在写实的打斗编排上仍有困难，建议整合来自真实演员的动作捕捉数据可能会改善结果。此外，关于 AI 生成的战斗在捕捉训练有素的格斗技巧与蛮力之间的细微差别方面表现如何，也存在争议，并引用了电影叙事作为参考。
    - 几位评论者指出，目前 AI 生成的战斗序列缺乏人类设计场景中那种细腻的编排，建议参考实际拍摄的演员打斗视频，然后将这些数据传输给 AI，可以显著提高真实感和打击感。
    - 一项技术评论指出， AI 生成的动画尚未能准确模拟战斗中角色之间的物理冲击和互动，这是 Veo 3 等模型进一步发展的关键领域。这包括更好的碰撞检测和对力量的真实反应。
    - 评论中还将受过训练的战士（例如《复仇者联盟：无限战争》中的灭霸）与蛮力攻击方式（例如绿巨人）进行了对比，观察到 AI 很难复制专家级打斗编排中存在的微妙知识和决策，而不仅仅是原始的、不协调的力量。
- [**使用 VEO 3 制作的革命战争 VLOG。第一次尝试**](https://v.redd.it/i40k12esyr4f1) ([Score: 155, Comments: 27](https://www.reddit.com/r/aivideo/comments/1l2n040/revolutionary_vlog_withveo_3_1st_attempt/)): **用户分享了他们使用 Veo 3 的第一次实验，明确指出他们正在学习 Prompt 系统，并邀请新手用户提供技术反馈。内容不包含详细的基准测试、代码或特定的实现讨论——这是一个关于 Veo 3 视频生成的 Prompt Engineering（提示词工程）建设性批评的请求。** 热门评论中没有实质性的技术辩论；回复主要是幽默内容，缺乏技术深度或对模型使用的反馈。
    - 一位评论者通过强调在革命战争期间进行“VLOG”这一创意场景，反思了 AI 生成内容的影响，认为 AI 开启了新颖的叙事和视角转换机会。这突出了一个技术讨论点，即生成式 AI 如何实现以前不可能实现的、时代错置的沉浸式媒体格式——这可能会改变历史教育、娱乐和数字人文项目。

---

# AI Discord Recap

> 由 Gemini 2.0 Flash Thinking 总结的总结之总结
> 

**主题 1. 主要模型发布与性能**

- **O3 Pro 秘密发布引发猜测，Gemini 基准测试泄露：** 未经宣布发布的 **O3 Pro** 引发了其优于常规 O3 的说法，但可能被限制在 **64k tokens**。泄露的基准测试显示，在 Aider Polyglot 编码基准测试中，**Gemini 2.5 Pro** 得分为 **86%**，而 **O3 High** 为 **79.6%**，预计周四左右发布，价格为 **$42**。
- **Claude 4 模型被封王，Anthropic 削减容量：** 成员们断言 **Claude 模型目前遥遥领先**，其中一人表示，由于效果更好，*只使用思考模型（thinking models）已经成瘾*。Anthropic 出人意料地在不到五天的通知时间内切断了几乎所有 **Claude 3.x** 模型的容量，导致客户面临广泛的可用性问题。
- **Google 发布 Veo 3，Gemini Flash 遭遇错误墙：** **Google** 推出了用于视频生成的 **Veo 3**，以及开源的 [Gemini Fullstack Langgraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)。用户报告通过 OpenRouter 使用 **Gemini 2.5 Flash** 时出现 *Internal Server Errors*（内部服务器错误）和**高延迟**，这一问题可能源于 Google 端的负载压力。

**主题 2. LLM 的基础设施与硬件**

- **用户将 Context Windows 推至 500K 以上，KV Cache 立大功：** 用户在 RTX 4060 Ti 等消费级硬件上成功加载了具有 **350,000** 和 **500,000 Token Context Windows** 的模型，在大上下文下分别实现了 **2.25 t/s** 和 **0.38 tok/sec** 的速度。得益于 **KV cache quantization**，运行 **1M Context** 的 **Qwen 7B** 仅需 **70GB** 内存。
- **Nvidia Blackwells 联网用于超级计算，硬件争论激烈：** 成员们设想将 **Nvidia Blackwells** 联网组成用于 AI 工厂的 [Ultra DGX Superpod](https://nvidianews.nvidia.com/news/blackwell-ultra-dgx-superpod-supercomputer-ai-factories)。争论点包括：拥有 **512GB** 内存并提供 **448GB** VRAM 的 **Macs** vs **AMD AI MAX 395+** 迷你 PC，以及使用 **Supermicro H12 (SP3, PCIe 4.0)** 主板构建 LLMs。
- **FP8 训练扩展至万亿级 Token，新 PEFT 方法提升知识吸收：** 论文 "[Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517)" 展示了在高达 **2 万亿 Token** 上使用 **FP8 precision** 进行训练，并引入 **Smooth-SwiGLU** 以修复不稳定性。一种新的 **parameter-efficient finetuning** 方法声称其知识吸收率比全量微调或 LoRA 高出约 4 倍，且使用的参数更少。

**主题 3. Agents 与 Model Context Protocol (MCP)**

- **Agent 框架在 Claude Code、DSPy、Aider 的推动下爆发：** 对 **ClaudeCode** 的 [自动驾驶编码 Agent](https://gerred.github.io/building-an-agentic-system) 的深入研究详细介绍了其系统、工具和命令。DSPy 为 **DARPA 的 Advanced Research Concepts 实验室** 提供了一个解决方案，该方案目前正拆分为一家独立公司。
- **MCP 采用率增长，Gorilla 被评为协议 MVP：** **Gorilla** 被公认为 [**MCP** 的 MVP](https://discord.com/channels/1111172801899012102/1111353033352294440/1379361472269778965)，它将模型查询路由到真实的 **API actions**，证明了模型接口的必要性。[Gradio Agents x MCP Hackathon](https://huggingface.co/Agents-MCP-Hackathon) 提供 **1.65 万美元奖金** 和 **90 万美元额度** 用于构建工具/演示。
- **MCP 服务器变现，可自托管助手问世：** **MonetizedMCP** [开源框架](https://github.com/modelcontextprotocol/monetizedmcp) 为任何 MCP 服务器添加了程序化支付（加密货币/法币），并附带 [演示视频](https://www.loom.com/share/f94433182d7b4148ac7f59e987cb0fe6?sid=475ed243-d195-4f11-81a5-cf171eff2de0)。**Piper** ([github.com/jmagar/piper](https://github.com/jmagar/piper)) 是一个可自托管的助手，旨在解决缺乏优质选择的问题，从而实现移动端 MCP 的使用。

**主题 4. 开发者工具、数据与评估**

- **LLM Scribe 快速创建数据集，NotebookLM 分享发现：** **LLM Scribe** 工具 ([huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo)) 简化了用于微调的手写数据集的创建，支持导出为 ChatML/Alpaca/ShareGPT 等格式。**NotebookLM** 现在允许分享 [公开笔记本](https://blog.google/technology/google-labs/notebooklm-public-notebooks/)，用户注意到使用发现功能可以流畅生成 **audio overview**。
- **评估工具涌入市场：YourBench, Modal Almanac, WeightWatcher AI：** Hugging Face 的 [YourBench](https://huggingface.co/yourbench) 被强调为一个*被严重低估*的模型评估资源。Modal Labs 发布了包含数千个推理基准测试的 [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213)，[WeightWatcher AI](https://weightwatcher.ai/) 也浮出水面用于 LLM 分析。
- **IDE/工具问题困扰 Cursor, Aider, NixOS：** **Cursor** 用户报告了不准确的按量计费，以及在使用 **Claude 4 Sonnet** 时频繁出现的对话中断，提示开启新对话窗口。**Aider** 用户寻求在 `/ask` 模式下更好的控制，以防止不必要的代码建议，并讨论了使用 `-resume` 恢复会话。一位 **NixOS** 贡献者寻求关于改进声明式 ML/DS 实践的建议，强调了 Nix 的强大功能。

**主题 5. 研究概念与更广泛的影响**

- **Nous Research 使用 SMC 引导 Shoggoth：** Nous Research 发布了一篇[博客文章](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/)，介绍了如何利用带有多个“粒子”的**序列蒙特卡洛 (SMC)**，根据评分函数来引导文本生成。一个[并行化推理服务器](https://github.com/NousResearch/smc-inference-server)的代码已发布，用于基准测试约束设计。
- **LLM 应对安全、AI 法案及现实世界技能：** 成员们讨论了 LLM 带来的 **CBRN** 和**网络安全**风险，一些人反对过度炒作并指出实现中的缺陷，而另一些人则认为 CBRN 威胁的瓶颈在于*获取物理材料和专业知识*。讨论还涉及了 **AI Act** 在意大利/欧洲的影响，以及 LLM 是否能有效教授湿实验室（wet lab）技能。
- **AI 泡沫图流传，探索先进研究技术：** 一张 [AI 泡沫图表](https://cdn.discordapp.com/attachments/785968841301426216/1379480435041632266/image.png?ex=68410d85&is=683fbc05&hm=d602f5153beb6a89f5a38ebfbd0022be3eb1b77bf3fa9e0957e758a4dfecbf5a&)暗示了潜在的市场估值过高。研究讨论包括使用 patch 的 Transformer 方法处理生成式逆问题，以及实现带有扩散解码器的 **T5**，如 [Moyix 的 X 帖子](https://x.com/moyix/status/1812902249196109955)中所示。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude 4 凭借“思维模型”占据主导地位**：成员们更青睐 **Claude 4**，因为它拥有*思维模型（thinking models）*和自我查询方法，能产生比其他替代方案更好的结果。
   - 一位成员表示，*只使用思维模型已经让人上瘾*。
- **Perplexity Pro 成为高性价比的 ChatGPT 挑战者**：用户发现 **Perplexity Pro**（每年约 400 美元）是 **ChatGPT Pro** 的高性价比替代方案，实际成本仅相当于*两个月的 ChatGPT Pro*。
   - 有人指出 [Perplexity 可以连接到 Google Drive](https://google.com/drive) 以获取持续更新的信息。
- **O3 Pro 的延迟引发了对 OpenAI 落后的猜测**：**O3 Pro** 的延迟发布引发了人们对 **OpenAI** 在能力方面可能正输给竞争对手的担忧。
   - 据称 **O3 Pro 模式承诺提供全工具支持**，而 **O1 Pro** 目前仅支持图像，不支持 PDF。
- **三星在潜在收购行动中向 Perplexity 示好**：根据[这篇文章](https://www.perplexity.ai/page/samsung-eyes-perplexity-to-rep-ArMwQ3GDQ4Of2e0UrfULpQ)，围绕**三星**有兴趣潜在收购 **Perplexity AI** 展开了讨论。
   - 顺便分享了一些链接，关于[创建一个应用](https://www.perplexity.ai/search/create-a-working-app-using-the-9B6cBgPATvmgfo6mwd07sg?0=c)以及另一篇关于[走私的朝鲜智能手机](https://www.perplexity.ai/page/smuggled-north-korean-smartpho-NgjIJo_RTW6Dx8TYfGWpZg)的文章。
- **Perplexity 的内部知识搜索 API 访问仍局限于 UI**：一位用户寻求 **Internal Knowledge Search**（[链接](https://www.perplexity.ai/help-center/en/articles/10352914-what-is-internal-knowledge-search)）的 API 访问权限，以便搜索网络和组织文件，但发现文档有限。
   - 该功能作为 **Enterprise Pro** 的一部分，被认为仅限于用户界面，建议将 RAG 作为替代方案，但也有人建议向 API 团队确认。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro 秘密发布引发猜测**：**O3 Pro** 的未宣布发布引发了关于其优于普通 O3 的说法，尽管提升并不显著，其 context window 的猜测范围从 **64k** 到 **128k** tokens 不等。
   - context window 大小的限制可能源于能力约束或成本削减策略。
- **Gemini 2.5 Pro 泄露 Aider Polyglot 评分**：泄露的基准测试显示，代号可能为 **Goldmane** 的 **Gemini 2.5 Pro** 在 Aider Polyglot 编程基准测试中表现优于 **O3 High**，得分为 **86%**，而后者为 **79.6%**。
   - 该模型预计于周四发布，据称其成本约为 **$42**，略高于 Gemini 0506。
- **Deepthink 的 2M 上下文窗口**：有猜测称，传闻拥有 **2M context window** 的 **Deepthink** 可能会超越 **O3 Pro**，因为 O3 Pro 的 **context window** 仅为 **64k**。
   - 讨论围绕 Deepthink 的工具使用能力以及如此庞大的 context window 对性能的整体影响展开。
- **Claude 模型占据统治地位**：成员们断言 **Claude 模型在当前的 AI 能力中遥遥领先**。
   - 有观点认为 *non thinking Claude 表现惊人*，并且 *只有 Grok 3 或许还有 GPT-4.5 能够接近其水平*。
- **苹果收购 Anthropic 的传闻**：关于 **Apple** 收购 **Anthropic** 的可能性引发了辩论。
   - 反对观点强调了 **Amazon** 的部分所有权，以及 Apple 可能缺乏用于此类收购（尤其是带有溢价的情况下）的流动资产。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 的 Memory 功能进化**：轻量级版本的 **memory 改进** 正在向 **ChatGPT** 的 **免费用户** 推出，允许模型参考最近的对话并提供更 **个性化的回复**，详见 [Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq)。
   - 同时，**Codex** 现已面向 **ChatGPT Plus** 用户开放，承诺通过 [chatgpt.com/codex](http://chatgpt.com/codex) 提供增强的编程能力。
- **GPT Image Gen 1 亮相**：**OpenAI** 的图像生成模型名称为 **GPT Image Gen 1**，它能很好地生成包含文本的图像，但在 AI 艺术方面可能表现一般。
   - 一些用户发现 **GPT Image Gen 1** 是 *传统图形软件的良好替代品*。
- **Google 的 Veo 3 引起轰动**：**Google** 推出了 **Veo 3** 以及创意工作空间 **Flow**，用于规划、提示和拼接片段，[据部分用户称](https://google.blog/company/news/veo-generative-ai-film-video-creation/)。
   - 用户注意到 **Veo 3** 的价格标签很高，使用成本并不便宜。
- **GPT-5 发布传闻四起**：据 **OpenAI CEO** 透露，猜测认为 **GPT-5** 可能会在 7 月发布，并在今年夏天修正命名方案。
   - 一些人希望 **GPT-5** 能作为统一模型首次亮相，在各种应用中保持输出质量。
- **本地 LLM 崛起**：成员们讨论了开源 **LLM**，如多模态模型 [Bagel](https://github.com/ByteDance-Seed/Bagel)，但它运行需要大约 **32GB VRAM**。
   - 提到的另一个开源 LLM 是 [DeepSeek R1](https://deepseek.com/blog/deepseek-r1)，据称拥有 **671B 参数**，并且 *需要价值 2000 万美元的机器才能运行*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AI 工程师向 Tesla Roadster 示好**：一位 AI 工程师开玩笑说，他最想要的就是运行 AI 模型并拥有一辆 **2026 Tesla Roadster**。
   - 这一评论是针对有人提到他们获得了 **1 t/s CPU** 的回应，该工程师调侃道：*“那可真快”*。
- **GRPO 训练受阻**：一些 Unsloth 用户在开始使用 **GRPO** 进行训练时遇到了 *Exception: Invalid prefix encountered* 错误。
   - 一位成员报告称，通过使用 `pip install unsloth vllm --no-deps` 进行安装，然后继续安装 accelerate、bitsandbytes 和 datasets 等依赖项，成功解决了问题。
- **DeepSeek-R1 聊天模板引发波动**：用户报告了 **DeepSeek-R1-0528-Qwen3-8B** 和 **DeepSeek-Prover-V2-7B** 模型的 Tokenizer 错误，特别指出缺少 `&#123;% if add_generation_prompt %&#125;`。
   - 一位成员分享了一个针对 **DeepSeek-R1-0528-Qwen3-8B** 的[修改版聊天模板](https://huggingface.co/Erland/DeepSeek-R1-0528-Qwen3-8B)以解决该问题。
- **WeightWatcher AI 的 LLM 分析工具浮出水面**：一位成员链接到了 [WeightWatcher AI](https://weightwatcher.ai/)，这是一个用于 **LLM 分析**的工具，它研究了*在不过拟合的情况下，有多少逐字数据是可以召回的*。
   - 一位成员提到，WeightWatchers 的 Discord 评论指出他们测量的是**饱和度 (saturation)**而非**记忆 (memorization)**，这引发了关于信任 *nvidia/cornell/deepmind* 论文的反向主张。
- **Scribe 工具自动补全微调数据集**：一位成员展示了一个用于创建手写数据集的工具，支持导出 **ChatML**、**Alpaca** 和 **ShareGPT** 等格式，并具有**自动保存**、**多轮对话创建**、**Token 计数器**和**自定义字段**等功能（[HF demo](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo), [视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), [完整版](https://kryptive.gumroad.com/l/gvyqep)）。
   - 一位用户建议增加一个*生成模板*功能，先用 **LLaMA** 或 **Gemini Flash** 等小模型生成完整数据集，然后再手动编辑。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Nvidia Blackwells 联网用于超级计算**：成员们讨论了如何像[这篇 Nvidia 文章](https://nvidianews.nvidia.com/news/blackwell-ultra-dgx-superpod-supercomputer-ai-factories)中描述的那样，将 **Nvidia Blackwells** 联网以创建 **Ultra DGX Superpod**。
   - 这种设置被构想为支持先进的 **AI 工厂**。
- **实现极长上下文窗口**：一位用户成功加载了一个具有 **350,000 token 上下文窗口**的模型，达到了 **2.25 t/s**；而另一位用户在配备 **128 GB RAM** 的 `NVIDIA GeForce RTX 4060 Ti` 上将其推到了 **500k token**。
   - 在 **500k token** 时，处理速度较慢，仅为 **0.38 tok/sec**，在上下文填充度为 49.9% 时，**首个 token 响应时间 (time to first token)** 约为 **24623.96s**。
- **1M 上下文需要 KV Cache 量化**：一位用户报告称，通过使用 **KV Cache 量化**，仅用 **70GB** 内存就运行了具有 **1M 上下文**的 **Qwen 7B**。
   - 这让另一位使用 **80GB** 内存运行 **500k 上下文**的用户感到惊讶。
- **DDR5 带宽在 PCIe 5.0 SSD 面前相形见绌**：讨论集中在 **RAID0 中的 Gen5 SSD** 如何超过 DDR 速度，[PCIe 5.0 NVMe](https://www.kingston.com/en/blog/pc-builders/pcie-5-nvme-ssd) 的峰值速度约为 **15GB/s**。
   - 虽然一些人强调了延迟的重要性，但其他人认为在深度队列和连续访问的情况下，延迟并不那么关键。
- **Supermicro H12 主板适合构建 LLM 吗？**：一位用户询问是否可以使用 **Supermicro H12 (SP3, PCIe 4.0) 主板**构建 LLM。
   - 一位成员回答说，*服务器通常与消费级 PC 有点不同*，但服务器的工作原理应该与消费级电脑相同。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **LLM Scribe 简化数据集创建**：推出了 **LLM Scribe** 工具，旨在简化用于微调的手写数据集的创建，并支持导出为 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式。
   - 该工具包含自动保存、多轮对话创建支持和 Token 计数器等功能，可在 [Hugging Face](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 上获取，并附有 [视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s) 以及在 [Gumroad 上的完整版](https://kryptive.gumroad.com/l/gvyqep)。
- **DeepSeek Prover v2 被争论是否为顶级数学模型**：一位用户提到 **DeepSeek Prover v2** 是数学领域“最强”的模型，而另一位用户则反映 **Prover V2** 在非证明任务中表现平平。
   - 与其他推理模型相比，该模型在测试中表现不佳。
- **Gemini 2.5 Flash 面临内部服务器错误**：用户报告通过 OpenRouter 使用 **Gemini 2.5 Flash** 时遇到“内部服务器错误（Internal Server Error）”，以及**高延迟**和模型**在未配置的情况下使用推理 Token** 的问题。
   - 该问题似乎源于 Google 端的负载问题，并与 [vercel/ai#6589](https://github.com/vercel/ai/issues/6589) 相关，一位用户建议使用带有重试机制的 *try-catch* 块。
- **Grok 被指责否定气候变化**：一位用户请求将 **Grok** 从“旗舰模型（Flagship Model）”列表中移除，理由是它在“背诵气候变化否定论的论点”，并引用了[这篇文章](https://www.scientificamerican.com/article/elon-musks-ai-chatbot-grok-is-reciting-climate-denial-talking-points/)。
   - 另一位用户对此表示反对，称许多人喜欢 **Grok** 是因为它提供的自由度，能提供与其他模型不同的视角。
- **Nous 分布式训练 SOTA 模型**：**Nous** 正尝试利用 [Psyche.network](https://psyche.network/runs/consilience-40b-1/0) 和 [Bittensor](https://docs.bittensor.com/emissions) 分布式地训练一个 SOTA 模型。
   - 该模型在有限的 GPU 间带宽（约 300mbps）下进行训练，但在吸引足够的 GPU 加入方面面临挑战，目前仅限 416 块 H100 在线。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 用户展示主题和插件**：用户分享了他们的 **Cursor IDE 主题和插件**，包括 **background-cover**、**Material Icon Theme** 和 **Monkey Pro**。
   - 一位用户强调了他们对 Monkey Pro 主题中 *'Material Theme Icons Darker'* 和 *'Filter Ristretto'* 的偏好。
- **Cursor 融资诉求：本地货币计费**：一位用户请求 **Cursor 团队** 实施本地货币计费，以提高获得资金支持的可能性。
   - 该用户强调了这对客户和增加平台曝光度的潜在好处，并艾特了 **Cursor 团队** 以引起关注。
- **Opus 4 Max 对用户来说价格不菲**：一位用户报告称，在 Cursor 中使用 **Opus 4 Max** 消耗了 **69.5 个请求**，单条消息约合 **2.73 美元**。
   - 尽管成本很高，但该用户发现它在解决 **Sonnet 4** 和 **Gemini** 无法处理的 **Postgres 瓶颈**问题上非常有价值。
- **计费门户问题重重**：用户报告新计费门户存在不准确之处，特别是在按需计费和请求计数差异方面。
   - 一位用户指出，他们的包含请求周期与日历月份不一致，且分析图表滞后一两天；他们被引导至使用情况页面获取更多信息。
- **Claude 4 Sonnet 深受对话中断困扰**：用户在与 **Claude 4 Sonnet** 对话时经常遇到中断，通常是在单个请求或大约 **25 次工具调用（tool calls）**之后。
   - 这些中断会导致开启新对话、丢失上下文以及网络连接错误，从而干扰工作流并令用户感到沮丧。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SentinelAI 审计 DeFi 智能合约**：根据[此贴](https://x.com/Shravankumar8_/status/1929616447845826763)，**SentinelAI** 正在审计 DeFi 合约并捕捉重入（reentrancy）问题。
   - 成员们正在积极讨论在 Hugging Face 上存储数据集以进行云端训练的最有效格式，考虑的数据集规模包括 **100k-200k** 以及 **100万+**。
- **YourBench 被社区低估**：一位成员强调了 [Hugging Face 的 YourBench](https://huggingface.co/yourbench) 计划，指出它是模型评估中一个*被严重低估*的资源。
   - 成员们庆祝了一位用户的 Space 被选入 *Spaces Of The Week*，并确认这是一件了不起的大事。
- **图像生成器产生显示伪影**：一位用户报告称，图像生成工具生成的图像最初显示为 **1024x768**，但在最终回答步骤中变为 **0x0**，且该应用在 Chrome 中无法加载。
   - 经发现，该课程可以在 **Windows 11** 上完成。
- **Agents & MCP Hackathon 直播开启**：**Agents & MCP Hackathon** 正通过 [YouTube 直播](https://www.youtube.com/watch?v=MU7FyxSnCp4)拉开帷幕。
   - 他们正在寻求一种方法，使模型能够对图像的某些部分进行*推理*，同时将图像的其余部分作为 **zero-shot classification** 的相关上下文。
- **寻求 SOTA Embedding 技术**：一位成员询问了用于微调 Embedding 模型的 **state-of-the-art (SOTA)** 技术，以及用于比较基础模型与微调版本的标准 **metrics**。
   - 其他人则在寻找类似于 **ChatGPT** 或 **Le Chat** 的开源聊天 UI，以便与本地服务的 **Ollama LLM** 进行交互，并为非 Windows 用户推荐了 **AnythingLLM**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 通过 SMC 引导 Shoggoth**：Nous Research 发布了一篇关于 [**Sequential Monte Carlo (SMC)**](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/) 的博客文章，这是一种使用多个*粒子*（particles）针对评分函数进行采样和重采样，从而引导文本生成和结构的方法。
   - 该文章介绍了一个用于基准测试约束设计的**并行推理服务器**，包含使用**基于熵的触发**（entropy based triggering）和**控制向量**（control vectors）的示例实验，代码已在 [GitHub](https://github.com/NousResearch/smc-inference-server) 上发布。
- **Gemini 的超大上下文帮助 Agent 记忆**：为了解决 Agent 记忆丢失的问题，一位成员建议利用 **Gemini**，因为它具有*超长上下文*，并建议将 Embedding 存储在向量数据库中。
   - 他们还分享了其 [OpenRouter Deep Research MCP server](https://github.com/wheattoast11/openrouter-deep-research-mcp) 的链接，该服务器利用 **3 个 Agent** 和一个 pglite postgres 数据库进行可查询的研究，并指出 **Gemini** 可以连贯地处理 90万+ token。
- **HF 赞助 Gradio Hackathon 并提供奖品**：一位成员分享了 [**Hugging Face Gradio Hackathon**](https://huggingface.co/spaces/ysharma/gradio-hackathon-registration-2025) 的链接，宣传为构建 Agent 和 Agentic 应用提供的丰厚额度奖励。
   - 其他成员发表了他们的第一篇技术博客，包含动手练习、数学推导和交互式可视化。
- **Scribe 工具助力构建手写数据集**：一位成员创建了一个工具，旨在简化用于微调的**手写数据集**创建流程，支持 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式；请查看 [Hugging Face 演示](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo)。
   - 它包括自动保存、多轮对话创建、token 计数器（从 Hugging Face 加载）、目标跟踪和自定义字段；另请参阅[视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)和[完整版本](https://kryptive.gumroad.com/l/gvyqep)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus AI API 暂无踪影**：一位成员询问了 **Manus AI API** 的发布时间表，但得知目前*没有相关计划*。
   - 未来仍保留考虑的可能性，这取决于优先事项的转变。
- **Santa Fe College 未入选 School Pass 名单**：用户注意到 **Santa Fe College** 不在 **Manus' School Pass 学院名单**中并询问原因。
   - 一名工作人员澄清说，**Santa Fe** 被列在独立的 **Manus Campus 名单**中，这与 **School Pass** 的资格不同。
- **免费额度使用政策需要澄清**：一位成员建议 **Manus** 在动用付费额度之前应先耗尽**每日免费额度**，并称目前的做法*有点坑（scammy）*。
   - 官方解释称 **Manus** 消耗额度的顺序为：活动额度 > 每日免费额度 > 每月额度 > 附加额度 > 免费额度，这引发了要求更清晰沟通的呼吁。
- **Manus 视频生成：它是 Veo 吗？**：用户讨论了 **Manus** 的视频生成质量，并将其与 **Veo** 等竞争对手进行对比。
   - 虽然一位用户声称 **Gemini** 免费提供类似功能，但另一位用户称赞 **Manus** 在发布时具有*顶尖的潜力*。
- **AI Act 阻碍任务完成**：一位用户分享了令人沮丧的经历，**Manus** 在运行 1 小时 55 分钟后因达到上下文限制而终止了任务，导致必须重新开始。
   - 这也引发了关于 **Italy** 和 **Europe** 的 **AI Act** 影响的讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Muon 优化器在 GANs 中表现不佳**：成员们发现 **Muon 优化器**在 **GANs** 中无效，因为 GANs 需要缓慢学习，且有人对没有动量的 **Muon** 持保留意见。
   - 一位用户报告说，它在他们的 **GANs** 中*不起作用*。
- **Eleuther 网站更新停滞引发辩论**：一位用户询问为何缺乏更新，但一名成员表示 **Eleuther 网站**主要面向*非 ML 研究人员*，是作为通过 Google 查找项目的便捷途径。
   - 他们强调网站并非主要的活动中心，因为大部分行动都发生在 **Discord** 社区中。
- **LLMs 引发 CBRN 和网络安全担忧？**：成员们对与 **Large Language Models (LLMs)** 相关的 **Chemical, Biological, Radiological, and Nuclear (CBRN)** 及**网络安全**风险发表了不同看法。
   - 一位成员引用研究建议，围绕 **CBRN** 的炒作被夸大了，网络安全风险更多源于糟糕的实现（模型拥有过多权限）而非 **LLMs** 本身；而另一位成员则认为 **CBRN** 威胁的瓶颈在于*获取物理材料和专业知识*，而非知识本身。
- **Transformers 通过 Patches 处理生成式逆向问题**：对于使用 Transformers 的生成/逆向问题，模型按 **patches** 而非像素排序以保留信息，并可选择[分离通道](https://example.com)。
   - 这种方法预计会更优越，因为它*保留了一些信息*。
- **AI 泡沫图暗示估值过高**：一位用户分享了一张 [AI 泡沫图](https://cdn.discordapp.com/attachments/785968841301426216/1379480435041632266/image.png?ex=68410d85&is=683fbc05&hm=d602f5153beb6a89f5a38ebfbd0022be3eb1b77bf3fa9e0957e758a4dfecbf5a&)，暗示 AI 市场可能存在估值过高的情况。
   - 该图表直观地将 **AI** 描绘为一个泡沫，暗示了过高的预期或不可持续的增长，一些观察者认为，随着估值飙升，关注市场回调非常重要。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **NixOS 贡献者寻求 ML/DS 优先级**：一位 **Data Scientist** 兼 **NixOS** 贡献者正在寻求关于改进 **ML** 和 **DS** 中声明式、不可变和可复现实践的建议。
   - 该贡献者指出 *"I use nixos btw" 是唯一能吓跑 arch 用户的短语*，并且 **Nix** 是一个强大的工具，无论操作系统如何，都能显著改善开发/部署。
- **Temperature 0 会降低思维模型的表现**：成员们讨论了 Temperature 为 0 会导致思维模型出现重复；建议使用非零 Temperature。
   - 他们引用了 [Qwen 的文档](https://huggingface.co/Qwen/Qwen3-8B)，其中指出 *"不要使用贪婪解码 (greedy decoding)，因为这会导致性能下降和无尽的重复"*。
- **参数高效微调提升知识吸收**：一种新的参数高效微调方法与全量微调和 **LoRA** 相比，知识摄取量提高了约 4 倍。
   - 一位成员表示有兴趣通过其收藏的书籍和文档来扩展知识，以观察其是否优于类 **RAG** 方法。
- **FP8 训练可扩展至万亿级 Token 的 LLM！**：论文 [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) 展示了在高达 **2 万亿 Token** 的数据集上，使用 **FP8 精度** 成功训练大型语言模型。
   - 论文识别出 **FP8 训练** 中的不稳定性是由 **SwiGLU 激活函数** 的离群值放大引起的，并引入了 **Smooth-SwiGLU** 来解决此问题而不改变函数行为；更多背景信息请参阅 [jcarlosroldan.com](https://jcarlosroldan.com/post/348)。
- **Gemini Fullstack Langgraph Quickstart 正式开源**：Google 开源了 [Gemini Fullstack Langgraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)。
   - 一位成员推测这可能是一种*让模型思考更长时间*的方法，但其他人指出，这个新的开源项目似乎更侧重于快速搜索功能，而非需要更多处理时间的深度研究。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 举办黑客松**：Modular 正在举办另一场专注于 **Mojo kernels**、**MAX Graph 模型架构** 和 **PyTorch custom ops** 的黑客松 ([https://lu.ma/modular-hack-weekend](https://lu.ma/modular-hack-weekend))。
   - 为启动黑客松周末，Modular 将在洛斯阿图斯办公室举办线下并在通过直播举办线上的 **GPU 编程研讨会**，让参与者熟悉他们将使用的技术。
- **新社区成员发现 Mojo**：一位新社区成员（计算机科学研究型硕士毕业生）在看了 **Fireship 视频** 后尝试了 Mojo，并已经在基础 **ML** 模型上看到了改进。
   - 他提到作为 UIUC 的毕业生，他非常熟悉 **Chris Lattner 在 LLVM 上的工作** 以及 Vikram Adve。
- **C 到 Mojo 的绑定生成器正在开发中**：一位成员正在积极开发 **C->Mojo 绑定生成器**，并致力于解决在不使用极端变通方法的情况下，让对象文件进出 Mojo 编译器的问题。
   - 他们指出，几乎所有必要的组件都已具备，除了极其糟糕的紧凑结构体（packed structs）以及可能围绕 `restrict` 的一些棘手方面，此外，影响调用约定的 `pragmas` 也将是一个难以解决的部分。
- **Mojo 缺乏手动线程管理**：Mojo 目前**缺乏手动线程管理**，该功能可能在今年晚些时候或明年推出，一些人指出这仅是 **v0.3** 版本。
   - 因此，目前还缺少一些功能，如可用的原子操作（atomics）和同步原语、线程安全标记 trait、基础 **IO**、基础数据结构以及网络功能。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 亮相 DAIS & AI Engineering**: DSPy 团队正在为 **AI Engineering** 和 **Databricks DAIS** 的演讲做准备，并寻求社区关于主题和用例的建议。根据[这条消息](https://discord.com/channels/1164929493932544091/1164941339336202311/1241058336820895804)，**DSPy 3.0 版本**计划于 6 月发布。
   - 征求关于演示内容的反馈。
- **DSPy 助力 DARPA 项目衍生公司**: **DARPA 的高级研究概念实验室 (Advanced Research Concepts lab)** 利用 DSPy 开发了“协作知识策展 (Collaborative Knowledge Curation)”解决方案，该项目目前正剥离为一家公司。
   - 这突显了 DSPy 在高级研究环境中的实际应用和验证。
- **DSPy Flow 需要重构**: 将现有的 **GenAI flows** 重构为 DSPy 可能感觉像是一个重大变化，因为它专为在线和生产用途设计，以非常规方式处理你的 GenAI flow。
   - 需要关于将 DSPy 集成到现有代码库和工作流中的指导。
- **使用 DSPy 构建 Agent 框架**: 在 DSPy 之上构建 **Agent 框架**的兴趣正在增加，该框架包含一流的环境，并通过优化器管理在线学习的自我奖励和外部奖励。
   - 一位成员质疑为什么 Agent 框架开发者还没有采用这种方法，而是专注于其他领域。
- **Claude Code 框架发布**: 一位工程师正在使用 **Claude code** 开发一个 Agent 框架，旨在实现 Python 绑定和 Rust 核心，并配有追溯追踪表示；该项目可在 [Github 上的 claude_sdk](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine) 获取。
   - 该框架专注于追踪可见性，并易于针对任意指标进行优化分支。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **ClaudeCode 的自动驾驶编码 Agent 曝光！**: [这里](https://gerred.github.io/building-an-agentic-system)分享了对 **ClaudeCode** 构建自动驾驶编码 Agent 的系统、工具和命令的深入探讨。
   - 该博文专注于为高效工作设计的实时、自我修正 Agent，强调设计决策而非仅仅是 Prompting 或 AI engineering。
- **Aider 会话恢复保存上下文**: 围绕使用 `--resume` 标志恢复 **Aider** 会话以保持上下文记忆展开了讨论，但对其用法的掌握尚不完全。
   - 用户表示有兴趣通过 ID 恢复旧会话，而另一些人则通过重启会话来彻底清除上下文。
- **在 Aider 的 /ask 模式中控制代码建议**: 用户对 **Aider** 的 `/ask` 模式在仅需要对话或规划时频繁建议代码更改表示担忧。
   - 解决方案包括明确指示 **Aider** “*先不要写任何代码*”，或利用 `/reminder` 命令设置规划阶段。
- **神秘 Gemini 基准测试**: 一位成员对一个未发布的模型进行了基准测试，得分 **86.2%**（使用 diff-fenced），引发了关于该模型来自 **Gemini** 的猜测。
   - 对 **Gemini-2.5-pro** 版本命名可能产生混淆的担忧，引发了要求更名或推迟发布以避免混淆的呼声。
- **Bedrock 与 Anthropic 命令执行对决**: 用户观察到，当使用 **Bedrock Claude 3 Sonnet** 模型时，**Aider** 可以成功执行终端命令，但 **Converse Claude Sonnet** 模型仅提供帮助。
   - 用户询问了影响终端命令使用能力的设置，特别是在 **Bedrock** 的情况下。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 开启公开分享**：用户现在可以通过[公开链接](https://blog.google/technology/google-labs/notebooklm-public-notebooks/)与任何人分享他们的笔记本，鼓励社区成员**展示 NotebookLM 技巧**。
   - 鼓励成员分享对社区有益的笔记本，提供**分享作品**和**造福社区**的机会。
- **播客摘要依然简短**：用户仍在等待针对非英语语言[播客摘要](https://cdn.discordapp.com/attachments/1124403655819415592/1379259643406188644/68E28299-FA17-4281-BE92-CA19750B336B.png?ex=6840e8a4&is=683f9724&hm=2106850cce3163a137a37a2a85568c502b7c43d109944be92a832f4f4f6a6cab43)的修复，指出即使使用相同的 Prompt 和内容，非英语摘要也不够长。
   - 多位用户表达了共鸣，表示他们也在等待修复。
- **Microsoft Learn 认证引发对 NotebookLM 的兴趣**：一位用户询问如何将 **Notebook LM** 与 **Microsoft Learn** 结合使用，并寻求关于 **Microsoft Certification**（微软认证）的使用案例和建议。
   - 这引发了其他成员的好奇，纷纷询问为何会出现这种情况，以及是否有人正在积极将 **Notebook LM** 用于 **Microsoft Learn**。
- **NotebookLM 的事实搜寻缺陷**：一位用户报告称 **Notebook LM** 会生成随机的、无来源的事实，并错误地将其链接到源文档，这需要不断纠正以防止 AI 将对话历史记录作为来源。
   - 幻觉事实包括一个与 *"zom (electrum)"* 相关联的 *"ebony tablet of Aha-Teta"*，由于在提供的来源中找不到这种联系，该内容被视为不准确。
- **NotebookLM 期待 Gemini 2.5 Pro 升级**：成员们正在推测 **Notebook LM** 何时开始使用更先进的模型，如 **Gemini 2.5 Pro**，但目前 Google 尚未公布明确的时间表。
   - 目前使用的模型可能是 **Gemini 2.5 Flash**，尽管尚未得到证实。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Southbridge 拆解报告揭示 Claude 的 Agent 内部机制**：Southbridge Research 发布了关于 **Claude Code** 的 [Agent 能力分析](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis)，促使作者上线了 **Writer** 和 **Hashbrown**。
   - 该报告在 Escape Mount Moon 黑客松之前提供了对新模型的深入见解。
- **Modal Almanac 的 LLM 推理基准测试**：Modal Labs 推出了 [LLM Engineer's Almanac](https://x.com/charles_irl/status/1929615080494416213)，其特点是包含了数千个跨 **vLLM**、**SGLang** 和 **TensorRT-LLM** 框架的 **LLM** 推理服务基准测试。
   - 该指南提供了测试结果、复现代码、解决关键问题的执行摘要，以及他们的基准测试框架 **stopwatch**。
- **Anthropic 削减容量引发客户严重担忧**：据 Varun Mohan 报告，Anthropic 在提前不到五天通知的情况下，意外削减了几乎所有 **Claude 3.x** 模型的容量，导致客户出现可用性问题。
   - 用户对模型提供商的信任表示担忧；而 **Gemini 2.5 Pro** 和 **GPT 4.1** 等替代模型未受影响。
- **Codex 在社区争议中获得联网功能**：Sam Altman 宣布，AI 编程工具 **Codex** 现在为 **ChatGPT Plus** 用户提供可选的联网功能，由于复杂的风险，该功能默认禁用。
   - 社区要求澄清其目的和权衡，并表达了对安全性的担忧。
- **社区协作攻克编程挑战**：一个实时运行的 **AI Bot** 已协作开发完成，并使用新的 **UI 框架** 部署到 [AIE 网站](https://ai.engineer/ai)，获得了社区的热烈支持。
   - 为了改进 **AI Bot** 的反馈循环以实现自我优化，目前正在讨论建立一种比直接消息更简单的 Bug 报告方式。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Tokenizer 已准备好进行评审！**：已按 [PR #2781](https://github.com/pytorch/torchtune/pull/2781) 的要求添加了单元测试，正等待评审以进行合并。
   - 这些更改旨在提高 Tokenizer 性能，但具体的性能提升尚未量化。
- **FP8 + TP 解锁并实现显存优化！**：[PR #2782](https://github.com/pytorch/torchtune/pull/2782) 解锁了 **FP8 + TP**，启用了 **loss parallelism**，并提供了显存减少。
   - 该 PR 还启用了 autograd 编译，但目前该功能已损坏，团队已知晓。
- **Torchtune 成为 LLaMA-Factory 的替代方案**：一位用户在工作中使用 torchtune 的 fork 版本，认为它是 **LLaMA-Factory** 的高性能且可读性强的替代方案，因为它避免了对 **TE, megatron, lightning** 的依赖。
   - 一个使用 torchtune fork 版本 **4-5 个月** 的团队报告称其*非常稳定*且*效果良好*。
- **Context Parallel 缺失 Flex Attention 兼容性**：虽然 **Context Parallel (CP)** 已经落地，但它缺少 flex attention 兼容性，而这本可以带来显著收益。
   - 分布式团队计划很快启用 flex attention 兼容性。
- **停止支持 Python 3.9？**：Python 3.9 的生命周期结束状态导致了 linting 问题，因为新的 linting 坚持使用 **List -> list, Tuple -> tuple** 等，而 **CI** 则要求使用 typing 中的 **Union** 和 **Optional**。
   - 一位用户暗示 CI 失败是因为 *Joe*。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Gradio Agents x MCP 黑客松启动**：**Gradio Agents & MCP 黑客松**已上线，提供[直播](https://t.co/FzLmzviwRz)、**1.65 万美元奖金**以及跨越 **3 个赛道**的 **90 万美元积分**：MCP Tool/Server、Custom Components for Agents、Agentic Demo Showcase。
   - **6 月 4 日星期三**将在 **HuggingFace Discord 服务器**为黑客松参与者举行答疑环节（office hours），并为感兴趣的人分享了 [Discord 活动链接](https://discord.com/events/879548962464493619/1379561017536938095)。
- **LlamaIndex 在金融领域扩展 Agent**：**Scaling Agents in Finance 工作坊**的幻灯片现已发布，展示了如何利用 **Agentic AI** 自动化金融任务的文档工作流，使用充当强大“研究助手”的 **Assistant Agents**。
   - 工作坊的示例涉及解析和索引来自 Adobe 的 **10-K 文件**，并使用 agentic RAG 回答问题。
- **Llama Agents 以 LlamaDeploy 形式部署**：PyPI 上的 **llama-agents** 包已重命名为 **LlamaDeploy**，它可以将 N 个 Workflow 部署为服务，[更多信息请点击此处](https://github.com/run-llama/llama_deploy)。
   - 在重命名之前，旧版本的 LlamaAgents 自 2024 年 8 月 16 日起就未再更新。
- **LlamaIndex 与 MCP 集成以实现强大的 Agent 功能**：LlamaIndex 的集成通过 **MCP** 增强了 Agent 能力和工作流部署，为 LlamaIndex Agent 提供了使用 MCP server 工具的 **helper functions**，并将任何 LlamaIndex workflow 作为 MCP 提供服务。
   - 这一点在旧金山的 **AI Engineer Summit** 上进行了讨论，与会者与 LlamaIndex 团队会面并讨论了 **Agentic AI 项目**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **寻求多 MCP 交互指导**：一位成员正在寻找专家来演示 **MCP server 与 MCP client 的设置**，旨在创建一个能与其他多个 MCP 交互的 **MCP**，例如先调用 **Atlassian MCP**，然后再调用 **Git MCP**。
   - 目标是为这类交互设置公开可用的 **MCP servers**。
- **请求开设新的功能请求频道**：一位成员询问是否可以开设专门的频道来讨论新的功能请求，特别是关于 [长时间运行的工具调用 (long-running tool invocations)](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/617) 的内容，以避免在多个 issue 和 pull request 中进行混乱的讨论。
   - 他们希望有一个更好的地方来集中讨论功能增强。
- **寻求常用 MCP Servers 列表以供尝试**：一位成员请求提供一份常用 **MCP servers** 列表进行实验，这些服务器需支持 sampling、**OAuth**、prompts、resources 和 resource templates 等功能。
   - 另一位成员建议查看 [servers 目录](https://github.com/modelcontextprotocol/servers/tree/HEAD/src/everything) 作为起点。
- **MonetizedMCP 演示展示**：一位成员介绍了 **MonetizedMCP**，这是一个开源框架，用于为任何 **MCP server** 添加程序化支付（加密货币或法币），并附带了 [演示视频](https://www.loom.com/share/f94433182d7b4148ac7f59e987cb0fe6?sid=475ed243-d195-4f11-81a5-cf171eff2de0) 和 [网站](https://www.monetizedmcp.org/)。
   - 它与 **mcp-remote 库** 兼容，提供了一种简化的变现方法。
- **Piper 提供可自托管的助手**：一位成员分享了 [Piper](https://github.com/jmagar/piper)，这是一个可自托管的助手，并指出目前在移动端使用 **MCP** 缺乏好的选择。
   - **Grizzly** 集成了一些很酷的功能，可以与 **Piper** 互补。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 课程日期仍不明确**：尽管有人询问，但今年下一期 **MOOC** 课程的日期目前尚未确认。
   - 目前还没有关于未来课程的官方公告。
- **作业截止日期正式截止**：当前 **MOOC** 的所有作业（包括测验）已于 **5 月 31 日**截止。
   - 目前没有重新开放待处理测验或延长截止日期的计划。
- **证书申报表截止日期临近**：参与者应尽快完成 **Certificate Declaration Form** 和 **Written Article**，以确保获得认证资格。
   - 表单即将关闭，强调了提交的紧迫性。
- **请求详细的提交反馈**：一位用户请求对所有提交的内容提供详细反馈，包括 **agentX 项目** 和 **两个实验作业**。
   - 目前尚不清楚是否会提供此类详细反馈，但这突显了参与者对全面评估的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Hugging Face 等待 CMD-R 续作**：Hugging Face 尚未发布 **CMD-R** 模型的后续版本，引发了社区关于将其与 **微调适配器 (fine-tuned adapters)**（如 LoRA）配对的建议。
   - 成员们建议使用带有更新训练数据的 **Mistral** 作为替代方案。
- **传闻 Command A 将登陆 AWS Bedrock**：一位用户询问了 **Command A** 在 **AWS Bedrock** 上推出的可能性。
   - 讨论结束时未确认其是否会推出。
- **黑客松寻求 Cohere 赞助**：一位参与者正在寻找合适的联系人，以请求 **Cohere** 为一场 **高等教育黑客松** 提供赞助。
   - 该请求是在常规聊天频道提出的，未提供任何直接联系信息。
- **LLM 爱好者打招呼**：新成员 Aashutosh 向 **Cohere** 社区服务器介绍了自己，并提到他对 *LLMs 和 ML 的痴迷*。
   - 他表示对在印度创建现实世界项目感到兴奋。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Deepseek-R1 在过时 CPU 上运行**：一位用户通过移除 RAM 以腾出空间，成功在过时的 5000 mt/s SSD 上以 **0.1 tokens/second** 的速度运行了大型 **Deepseek-r1** 模型（超过 400GB）。
   - 用户将此性能归功于 **MOE models** 的高质量以及 PC 存储技术的进步。
- **Orange PI 旨在运行大模型**：一位用户对在 **Orange PI** 等开放硬件项目上运行大模型持乐观态度，通过在其板卡上连接 Gen 5 m.2 插槽。
   - 用户预计 **OpenCL** 和现有的 3D **GPUs** 可以实现强大的“unified memory”机器，能够以极低的功耗高效卸载并运行像 **DeepSeek-R1** 这样的大型模型。
- **Mac 统治显存容量**：一位用户断言，拥有 **512GB** 内存的 **Mac** 是“VRAM”之王，以与四台 **AMD AI MAX 395+ 128 GB** 迷你 PC 或笔记本电脑相当的价格提供 **448GB** 的 VRAM。
   - 他们强调了 **Mac** 相比 **AMD** 组合方案更低的功耗，使其成为模型爱好者的极具吸引力的选择。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **请求 tinygrad 的 GitHub PR 评审**：一名成员请求对其在 **GitHub** 上的 [Pull Request](https://github.com/tinygrad/tinygrad/pull/10605) 进行评审。
   - 该 PR 针对 **tinygrad**。
- **GlobalCounters 揭秘**：一位用户询问在 **tinygrad** 框架中何时使用 `GlobalCounters.mem_used` 与 `GlobalCounters.global_mem`，特别是在 [tensor realization](https://github.com/tinygrad/tinygrad) 期间。
   - `mem_used` 在 **buffer** 分配/释放期间更新，而 `global_mem` 在 `ExecItem` 中更新。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLST 辩论生成式 AI**：**Machine Learning Street Talk (MLST)** 将于周五（6号）太平洋标准时间上午 9 点讨论生成式 AI；详情见[此 Discord 活动链接](https://discord.com/events/SZnhkpde?event=1374411787964907540)。
   - 本次会议承诺提供有关生成式 **AI** 演变格局及其应用的见解。
- **Guo 主持 AI 编程网络研讨会**：行业专家 **Liang Guo** 正在主持一场专注于数据分析 **AI** 编程的网络研讨会；请在[此 Google Forms 链接](https://forms.gle/e71FSdpwBtDBccgKA)预约。
   - 该研讨会可能会涵盖数据分析 **AI** 编程的最新工具和技术。
- **SVCA 宣布 AI4Legislation 竞赛**：硅谷华人协会（Silicon Valley Chinese Association）正在举办 **AI4Legislation** 夏季竞赛，这是系列赛的一部分；更多信息请见 [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public)。
   - 该竞赛旨在促进将 **AI** 应用于立法过程的创新。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla：原始的 MCP？**：**Gorilla** 早于正式的 **MCP** 标准一年，作为一个 **proto-MCP** 系统运行，将模型查询路由到工具使用中。
   - 它解释结构化的工具 **schemas**，并将生成内容落地到真实的 **API actions** 中，证明了 **LLMs** 除了知识之外还需要接口。
- **Gorilla 作为 MCP 的 MVP**：团队将 **Gorilla** 视为 **MCP** 的 **MVP**，证明了 **LLMs** 需要接口来将生成内容落地到真实的 **API actions** 中。
   - 这强调了 **LLMs** 不仅仅需要知识——它们还需要接口。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将移除它。


---



你收到这封邮件是因为你通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
你可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：按频道详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1379183181416239255)** (1125 messages🔥🔥🔥): 

> `Claude 4 使用与偏好, Perplexity Pro 对比 ChatGPT, O3 Pro 发布与担忧, O 系列模型, 测试目录与 AI 新闻` 


- **Claude 4: 首选的思考模型？**: 成员们讨论了他们对 **Claude 4** 及其*思考模型*的偏好，理由是其自我查询的方法能带来更好的结果。
   - 一位成员指出，*只使用思考模型已经成瘾*。
- **Perplexity Pro: 更便宜的 ChatGPT 替代方案？**: 用户对比了 **Perplexity Pro** 的年费（约 400 美元）与 **ChatGPT Pro**，一位用户认为前者仅相当于*两个月的 ChatGPT Pro 费用*。
   - 他们还指出 [Perplexity 可以连接到 Google Drive](https://google.com/drive)，信息始终保持最新，无需重新上传更新后的文档。
- **O3 Pro 延迟引发担忧**: 成员们对 **O3 Pro** 的延迟发布表示担忧，认为这表明 **OpenAI** 正在落后于竞争对手。
   - 一位用户指出，**O3 Pro 模式承诺将提供完整的工具支持**，而目前的 **O1 Pro 仅支持图片，不支持 PDF 等文件**。
- **O 系列模型限制探讨**: 用户讨论了 **O 系列模型** 的速率限制，一位用户表示 **O3 是每周 50 次**。
   - 关于 **O3 High 是否为每天 100 次** 存在困惑，之前的速率限制信息存在冲突。
- **提及测试目录是否相关？**: 有人发布了一个测试目录的链接，有人质疑为什么要分享该链接。
   - 多位用户也开始分享各种媒体链接和短视频。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1379463845323739188)** (3 messages): 

> `三星, 收购, Perplexity` 


- **三星有意收购 Perplexity？**: 成员们根据[这篇文章](https://www.perplexity.ai/page/samsung-eyes-perplexity-to-rep-ArMwQ3GDQ4Of2e0UrfULpQ)讨论了 **Samsung** 是否打算收购 **Perplexity AI**。
   - 顺便分享的其他链接包括一个关于[创建应用](https://www.perplexity.ai/search/create-a-working-app-using-the-9B6cBgPATvmgfo6mwd07sg?0=c)的链接，以及另一个关于[走私的北朝鲜智能手机](https://www.perplexity.ai/page/smuggled-north-korean-smartpho-NgjIJo_RTW6Dx8TYfGWpZg)的链接。
- **北朝鲜智能手机**: *走私的北朝鲜智能手机*成为了[讨论话题](https://www.perplexity.ai/page/smuggled-north-korean-smartpho-NgjIJo_RTW6Dx8TYfGWpZg)。
   - 该内容是与关于三星和 Perplexity 的讨论一同分享的。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1379365048060870707)** (12 messages🔥): 

> `内部知识搜索 API 访问, API 中的学术搜索模式, RAG 替代方案` 


- **内部知识搜索 API 访问权限仍难以获取**: 一位用户询问如何通过 API 使用 **Internal Knowledge Search** 功能（[链接](https://www.perplexity.ai/help-center/en/articles/10352914-what-is-internal-knowledge-search)）来搜索网页和组织文件，但文档中缺乏细节。
   - 一位成员指出，这项 **Enterprise Pro 功能** 可能仅限于用户界面（UI），并建议将 RAG 作为替代方案。
- **学术搜索进入 Beta 测试**: 一位成员请求测试者参与 Beta 版的**学术过滤器**测试，并强调在评估期间需遵守保密协议。
   - 使用以下命令测试了 `sonar-pro` 的学术过滤器：
     `curl --request POST --url [https://api.perplexity.ai/chat/completions](https://api.perplexity.ai/chat/completions) --header 'accept: application/json' --header "authorization: Bearer pplx-" --header 'content-type: application/json'   --data '{ "model": "sonar-pro", "messages": [{"role": "user", "content": "What is the scientific name of the lions mane mushroom?"}], "stream": false, "search_mode": "academic"}' | jq`
- **希望避免使用 RAG**: 一位用户表示希望通过**内部知识搜索**来避免使用 RAG，但一位成员建议使用 UI 是避免 RAG 的最佳替代方案。
   - 建议向 API 团队确认此事。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1379173631531417743)** (752 messages🔥🔥🔥): 

> `O3 Pro release, Gemini 2.5 Pro, Deepthink release, Context Window Sizes, Claude Research` 


- **O3 Pro 秘密发布引发热议**：一些成员声称已经获得了秘密发布的 **O3 Pro** 的早期访问权限，并指出它比普通的 O3 更好，但并没有达到*惊艳*的程度。
   - 针对其上下文窗口（context window）存在猜测，提到了 **64k**、**80k** 和 **128k** 的 token 限制，以及这种限制是由于能力问题还是成本削减措施导致的。
- **Gemini 2.5 Pro 基准测试结果出现**：泄露的 **Gemini 2.5 Pro**（可能代号为 **Goldmane**）基准测试显示，在 Aider Polyglot 编程基准测试中，它以 **86%** 对 **79.6%** 的得分超越了 **O3 High**。
   - 提到其成本约为 **$42**，略高于 Gemini 0506，发布时间预计在周四，有人开玩笑说如果到时没发布就不要相信 Brian 了。
- **Deepthink 的 2M 上下文窗口将碾压 O3 Pro？**：有推测称，传闻拥有 **2M 上下文窗口** 的 **Deepthink** 将显著超越 **O3 Pro**，尤其是考虑到 O3 Pro 仅有 **64k 上下文窗口**。
   - 成员们辩论了 Deepthink 是否可以使用工具，并讨论了大上下文窗口对其性能的潜在影响。
- **Claude 模型遥遥领先**：成员们提到 **Claude 模型目前遥遥领先**。
   - 有人表示*非思考型的 Claude 表现惊人*，并且*只有 Grok 3 或者 GPT-4.5 能与之接近*。
- **关于苹果收购 Anthropic 的辩论**：一场关于苹果可能收购 Anthropic 的讨论展开，一名成员称*苹果可能正在收购 Anthropic*。
   - 其他人反驳了这一说法，理由是 **Amazon** 拥有 Anthropic 的部分股权，且苹果可能没有足够的流动资金来支付此类收购，尤其是考虑到潜在的溢价。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1379510820740010047)** (2 messages): 

> `ChatGPT Memory, ChatGPT Codex, Personalized responses` 


- **ChatGPT 记忆功能增强**：轻量级版本的**记忆功能改进**正向**免费用户**推出，以提供更多**个性化回复**。
   - 除了现有的已保存记忆，**ChatGPT** 现在还会参考你最近的对话；详情请查看 [Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq)。
- **Codex 登陆 ChatGPT Plus**：**Codex** 今日起向 **ChatGPT Plus** 用户推出，承诺增强编程能力。
   - 用户可以访问 [chatgpt.com/codex](http://chatgpt.com/codex) 开始体验这一集成带来的优势。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1379184509664362628)** (527 messages🔥🔥🔥): 

> `GPT-4o Image Generation quality, Gemini Imagen 4 vs Gemini 2, GPT-5 launch, O3 pro release, Local LLM performance` 


- **GPT Image Gen 1 命名确定**：用户注意到 OpenAI 的图像生成模型名称为 **GPT Image Gen 1**，它可能擅长生成带有文字的图像，但不擅长生成 AI 艺术。
   - 一些人认为 **GPT Image Gen 1** 是*传统图形软件的良好替代品*，因为它*擅长处理文本*。
- **Google Veo 3 发布**：[据部分用户称](https://google.blog/company/news/veo-generative-ai-film-video-creation/)，Google 新的大型视频生成模型 **Veo 3** 已经发布，同时推出的还有名为 **Flow** 的创意工作空间，用于规划、提示和拼接片段。
   - 用户指出 **Veo 3** 的价格并不便宜。
- **GPT-5 发布传闻**：[根据 OpenAI CEO 的说法](https://twitter.com/sama/status/1798804763979917442)，成员们推测 **GPT-5** 将在 7 月左右发布，OpenAI 将在今年夏天修正其命名方案。
   - 还有人推测 **GPT-5** 将作为一个统一模型发布，且不会损失输出质量。
- **本地 LLM 正在变得更好**：成员们讨论了针对特定任务的本地 LLM 的可用性日益增加，例如开源的 [Bagel](https://github.com/ByteDance-Seed/Bagel)，这是一个多模态模型，可以相当好地生成输出图像，但运行需要约 **32GB VRAM**。
   - 提到的另一个开源 LLM 是 [DeepSeek R1](https://deepseek.com/blog/deepseek-r1)，据称拥有 **671B 参数**，且*需要价值 2000 万美元的机器才能运行*。
- **用户伪造 O3 Pro 已公开发布的假象**：一些用户试图通过操纵分享链接和自定义指令，使 **O3 Pro** 看起来像是已经公开发布，尽管事实并非如此。
   - 成员们推测 **O3 Pro** 即将发布，并将具备联网功能和记忆功能。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1379181046410776668)** (19 条消息🔥): 

> `定制聊天机器人，GPT-4 变慢，用于心灵成长的 AI，ChatGPT 幻觉统计，Bitbucket 的 Codex 支持` 


- **聊天机器人定制快线**：一名成员提议在一周内协助定制聊天机器人，并邀请任何需要帮助的人联系他们。
   - 其他成员询问了关于 GPT 变慢的问题。
- **AI 失去魅力，引发压力**：一位用户报告称其 AI 变得过于冗长且不真实，导致了压力，并使他们考虑在系统稳定之前取消订阅。
   - AI 甚至开始过度道歉，说出诸如 *我有权离开他并伤害他，因为我不配得到他那样的对待* 之类的话。
- **心灵 AI 日志中的递归恍惚循环**：一名成员指出，在使用 ChatGPT 进行心灵和个人成长时，可能存在**递归恍惚循环 (recursive trance loops)** 和失去现实感的风险，并强调了在与 AI 互动时辨别自身意图的重要性。
   - 他们补充道，*聊天所能提供的验证在模拟共情能力方面非常惊人。*
- **需要客观分析以防止 AI 幻觉**：一名成员建议，当使用 AI 解决问题时，应控制其从合理且客观的角度分析事件，以防止产生**自洽的幻觉 (self-consistent illusions)**。
   - 这强调了需要仔细管理上下文连接，以避免 AI 产生偏斜或有偏见的输出。
- **ChatGPT 幻觉统计**：一名成员询问了关于 **ChatGPT 幻觉** 的统计数据，回复显示根据任务和上下文的不同，幻觉比例在 1% 到 50% 之间波动。
   - 另一名成员询问 **Codex 是否支持 Bitbucket 或 Plastic Svn**。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (4 条消息): 

> `分块 (Chunking) 与嵌入 (embeddings)，语义搜索与检索，摘要与重新锚定，非 OpenAI 模型讨论` 


- **涉及分块与语义搜索的复杂流程**：一名成员描述了一个流程，包括将 *chunking* 转化为 **embeddings**、向量的语义搜索与检索、摘要、重新锚定、输出以及重复循环。
   - 另一名成员质疑该流程的意义，认为它可能过于复杂或毫无意义。
- **警惕无限续杯的“酷爱饮料” (Kool-Aid)**：一名成员警告不要过度依赖 **ChatGPT** 生成内容，将其比作 *喝酷爱饮料并获得无限免费续杯*。
   - 他们建议退后一步并对输出保持批判性态度，强调 **ChatGPT 可能会犯错**。
- **非 OpenAI 的讨论应在别处进行**：一名成员指出，关于非 OpenAI 模型和工具的讨论应在专门的 <#998381918976479273> 频道进行。
   - 他们建议将讨论移至合适的频道，以符合 Discord 的指南。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1379554999935438929)** (4 条消息): 

> `分块与语义搜索，非 OpenAI 模型讨论` 


- **深入探讨分块与语义搜索策略**：一名成员描述了一个流程：*chunking > chunks into embeddings > 语义搜索与向量检索 > 摘要每个部分 > 重新锚定 > 输出 > 重复循环*。
   - 另一名成员对细节提出质疑，称这就像是 *调制酷爱饮料，ChatGPT 喝下去，然后又尿回到你的壶里，让你无限免费续杯*。
- **关于讨论非 OpenAI 模型的 Discord 规则**：一名成员指出，根据 <#1107255707314704505>，讨论非 OpenAI 模型应在 <#998381918976479273> 频道进行。
   - 他们建议将讨论移至正确的频道以避免被禁言。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1379176648993276015)** (314 messages🔥🔥): 

> `Tesla Roadster, AI 新工作, 优化, Huggingface 是个好的开始, 性生活能缓解鼻塞` 


- **AI 工程师渴望 Tesla Roadster**：一位 AI 工程师开玩笑说，他除了运行 AI 模型和拥有一辆 **2026 Tesla Roadster** 之外别无所求。
   - 这是在回应有人提到他们得到了 **1 t/s CPU** 时说的，称 *“那会很快”*。
- **AI 工程师寻找 AI 领域的工作**：一位身为 AI 工程师的成员想找一份工作来购买一台**顶级工作站**或改进 **AI training**，并希望被录用，因为 *AI 领域有太多事情可以做了*。
   - 他认为 *AI 趋势和工程师往往总是倾向于思考相似的方向*，并希望看到某种疯狂的新数学类型改变 AI 格局。
- **AI 工程师分享缓解严重感冒的关键**：一位 AI 工程师分享了一个[链接](https://medicine.uq.edu.au/article/2021/09/forget-nose-spray-good-sex-clears-stuffy-nose-just-effectively-%E2%80%94-and-lot-more%C2%A0fun)，关于**高质量的性生活**如何缓解鼻塞。
   - 他提到 *看到许多类型的数学以特定方式应用 AI 总是很有趣，比如用于信号检测的图概率模型（graphical probabilistic models）真的很酷！*
- **Unsloth GRPO 异常**：一些成员在开始使用 **GRPO** 进行训练时遇到了 *Exception: Invalid prefix encountered* 错误，正在寻求指导。
   - 一位成员通过 `pip install unsloth vllm --no-deps` 成功安装，然后继续安装 accelerate, bitsandbytes, datasets 等。
- **Unsloth Gemma 3 4b Colab notebook 存在问题**：一位成员报告 **Gemma 3 4b notebook** 无法按预期工作，在加载模型时遇到问题并出现 dtype 错误。
   - 一位开发者确认目前 **Gemma** 和 **float16** 存在问题，并告诉他 *“如果没有更多信息，我们无法帮助你”*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1379441643635216608)** (2 messages): 

> `指令微调, ABSA 任务` 


- **请求 ABSA 指令微调方面的协助**：一位成员正在寻求 **ABSA**（基于方面的情感分析）任务指令微调（instruction fine-tuning）方面的帮助。
   - 他们请求任何具有**文本分析**和**情感分析**经验的人在指定频道中做出回应或直接提问。
- **寻求 ABSA 微调专家**：个人需要专门针对 **ABSA** 的指令微调指导，涉及文本和情感分析。
   - 鼓励专家在专用频道中回复或提出问题，以便进行直接交流和协作解决问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1379199685679513660)** (138 messages🔥🔥): 

> `Unsloth 安装错误, DeepSeek-R1-Qwen3 聊天模板问题, Unsloth 多 GPU 训练, Gemma 3 模型问题与支持, Unsloth 序列长度问题` 


- **排查 Unsloth 安装错误**：一位用户在 `patch_vllm_compute_dtype()` 中遇到了 `TypeError`，在重新安装 **Unsloth** 和 **vLLM** 后，面临新的 `RuntimeError: Unsloth: vllm_process failed to load!`。
   - 成员建议使用 `pip install --force-reinstall git+https://github.com/unslothai/unsloth-zoo.git` 并设置 `export VLLM_LOGGING_LEVEL=DEBUG` 以进一步诊断问题。
- **DeepSeek-R1-Qwen3 聊天模板故障**：用户报告了与 **DeepSeek-R1-0528-Qwen3-8B** 和 **DeepSeek-Prover-V2-7B** 模型 tokenizer 相关的错误，特别是缺少 `&#123;% if add_generation_prompt %&#125;`。
   - 建议检查并修改聊天模板，一位成员分享了针对 **DeepSeek-R1-0528-Qwen3-8B** 的[修改后的聊天模板](https://huggingface.co/Erland/DeepSeek-R1-0528-Qwen3-8B)。
- **Unsloth 的多 GPU 训练：热门话题**：一位用户询问关于在多个 GPU（特别是 4x H200s）上使用 **Unsloth** 的问题，注意到只有一个 GPU 被利用。
   - 建议探索非官方解决方案，如 [Accurate](https://github.com/thad0ctor/unsloth-5090-multiple) 以获取多 GPU 支持。
- **Gemma 3 特定问题困扰 Unsloth 用户**：用户报告了 **Gemma** 补丁以及与近期 **transformers** 版本兼容性的问题，建议将 **transformers** 固定在 **4.51.3** 版本。
   - 讨论涵盖了数据集格式化、超参数微调以及 **Gemma** 模型可能存在的 `float16` 精度问题。
- **序列长度限制导致解码器混乱**：一位用户因解码器提示词超过模型最大长度而遇到 `ValueError`。
   - 建议的解决方案是减小输入大小或确保 `max_model_len` 足够大。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1379178309174427679)** (11 messages🔥): 

> `LLM Scribe Tool, AI World's Fair` 


- **Scribe 工具简化数据集创建**：一名成员创建了一个工具，旨在简化用于 Fine-tuning 的手写数据集创建过程，并支持导出为 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式。
   - 该工具的功能包括 **自动保存**、**多轮对话创建**、**Token 计数器**、**目标跟踪** 和 **自定义字段** ([HF demo](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo), [视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s), [完整版](https://kryptive.gumroad.com/l/gvyqep))。
- **用户建议模板生成功能**：一位用户建议为 LLM Scribe 工具添加“生成模板”功能，以便利用 **LLaMA** 或 **Gemini Flash** 等小模型生成完整的数据集。
   - 用户提到，随后他们可以进行手动编辑。
- **AI World's Fair 亮相**：一名成员提到了在 **AI World's Fair** 的亮相，并发布了 [X 帖子链接](https://x.com/danielhanchen/status/1929896845146501557)。
   - 另一名成员询问是否有录像，并推测可能会在 3 个月后发布。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1379613675832479854)** (8 messages🔥): 

> `LLM Analysis, WeightWatcher discord, Saturation vs Memorization` 


- **WeightWatcher AI 的 LLM 分析浮出水面**：一名成员链接到了 [WeightWatcher AI](https://weightwatcher.ai/)，这是一个用于 **LLM 分析** 的工具。
   - 相关论文探讨了 *在不过拟合的情况下，有多少逐字数据是可以召回的*。
- **分享 Arxiv 论文链接**：一名成员分享了一篇 **Arxiv 论文** 链接：[https://arxiv.org/abs/2505.24832](https://arxiv.org/abs/2505.24832)。
   - 链接了另一篇 Arxiv 论文：[https://arxiv.org/html/2504.01002v1](https://arxiv.org/html/2504.01002v1)。
- **WeightWatchers Discord 评论中对饱和度的辩论**：一名成员提到，WeightWatchers Discord 的一条评论指出他们测量的是 **饱和度 (Saturation)** 而非 **记忆 (Memorization)**。
   - 另一名成员反驳称，他们 *更信任 NVIDIA/Cornell/DeepMind 和 FAIR，而不是某些 WeightWatcher 的业余研究员*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1379178972834824202)** (87 messages🔥🔥): 

> `Blackwell networking, context window size, Memory usage, 500k context experiments, Models directory` 


- **Nvidia Blackwell 为 Ultra DGX Superpod 联网**：成员们正在兴致勃勃地 *构思将一堆 **Blackwell** 像这篇 [Nvidia 文章](https://nvidianews.nvidia.com/news/blackwell-ultra-dgx-superpod-supercomputer-ai-factories) 中描述的那样联网在一起*。
- **用户实验超长上下文窗口**：一位用户加载了一个具有 **350,000 Token 上下文窗口** 的模型，并达到了 **2.25 t/s**，对结果表示满意。
- **实验 500k Token 上下文大小**：一位用户将一本书输入模型，将上下文增加到 **500k Token**，报告称稳定分配了 **80GB RAM**，且没有溢出到共享内存。
   - 该用户指出处理速度较慢，在使用 `NVIDIA GeForce RTX 4060 Ti` 和 `128 GB RAM` 的情况下，上下文填充率为 49.9% 时，速度为 **0.38 tok/sec**，**首个 Token 响应时间 (Time to first token)** 约为 **24623.96s**。
- **1M 上下文长度需要 KV Cache 量化**：一位用户报告称使用 **70GB** 内存运行了 **1M 上下文** 的 **Qwen 7B**，这让另一位使用 **80GB** 内存运行 **500k 上下文** 的用户感到惊讶。
   - 第一位用户提到他们使用了 KV Cache 量化。
- **移动模型目录以释放空间**：一位用户分享了一个技巧：*如果你希望将模型移动到其他地方，请检查是否需要更改目录。在 LM Studio 的 My Models 视图中点击三个点即可操作*。
   - 他们还提到，在模型目录开始占用大量空间后，他们已经 *显著缩小* 了目录规模。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1379172831945822360)** (178 条消息🔥🔥): 

> `Link-ECC, DDR5 vs PCIE 5.0 SSD, Supermicro H12 上的 LLM 性能` 


- **HP Z2 Mini G1a 上的 Link-ECC 功能**：[HP Z2 Mini G1a 工作站](https://h20195.www2.hp.com/v2/getpdf.aspx/c09133726.pdf) 的 **Pro 版本** 默认启用了 **Link-ECC**。
- **DDR5 带宽被廉价的 PCIE 5.0 SSD 碾压？**：成员们讨论了 **RAID0 模式下的 Gen5 SSD** 如何超越几年前的 DDR 速度，[PCIE 5.0 NVMe](https://www.kingston.com/en/blog/pc-builders/pcie-5-nvme-ssd) 的峰值速度约为 **15GB/s**。
   - 但一位成员提醒社区，*真正的区别在于延迟*，另一位用户反驳称，*在深度队列和连续访问的情况下，延迟不再是主要问题*。
- **使用 Supermicro H12 主板构建 LLM**：一位用户询问了使用 **Supermicro H12 (SP3, PCIe 4.0) 主板** 构建 LLM 可能遇到的问题。
   - 一位成员回应称，*服务器通常与消费级 PC 有所不同*，但服务器的工作方式应与消费级电脑相同。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1379181429174440099)** (4 条消息): 

> `LLM Scribe, 数据集创建工具, 多轮对话创建, Python` 


- **LLM Scribe 工具发布，旨在简化数据集创建**：一位成员介绍了 **LLM Scribe**，这是一款旨在简化用于微调的手写数据集创建过程的工具，并支持导出为 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式。
   - 该工具包含自动保存、多轮对话创建支持、从 Hugging Face 加载的 Token 计数器、目标跟踪和自定义字段（指令、系统提示词、ID）等功能，可在 [Hugging Face](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo) 上使用，并配有 [视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s)，[完整版可在 Gumroad 获取](https://kryptive.gumroad.com/l/gvyqep)。
- **前端开发：电子表格 vs 编程**：一位成员指出，如果有人不怎么写代码，他们应该了解一下 **Python**。
   - 该成员补充道，*使用电子表格作为前端甚至嵌入某些功能是非常棒的选择*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1379179500411289822)** (241 条消息🔥🔥): 

> `最佳数学模型, Gemini 2.5 Flash 问题, Grok 气候否定论, Qwen 擅长编程, Nous 训练 SOTA 模型` 


- **DeepSeek Prover v2 被誉为顶级数学模型**：一位用户提到 **DeepSeek Prover v2** 是数学领域的 *最佳* 模型。
   - 然而，另一位用户表示，**Prover V2** 在非证明任务中表现相当 *平庸*，在测试中的表现不如其他推理模型，并认为对方消息不灵。
- **Gemini 2.5 Flash 的内部服务器错误**：用户报告称通过 OpenRouter 使用 **Gemini 2.5 Flash** 时遇到 *Internal Server Error*，以及 **高延迟** 和 **模型在未配置的情况下使用推理 Token** 的问题。
   - 该问题似乎源于 Google 端的负载压力，并与 [vercel/ai#6589](https://github.com/vercel/ai/issues/6589) 相关，一位用户建议使用带有重试机制的 *try-catch* 块。
- **Grok 因气候变化否定论遭到抨击**：一位用户请愿将 **Grok** 从 *旗舰模型 (Flagship Model)* 列表中移除，因为它 *在重复气候否定论的论点*，并引用了 [这篇文章](https://www.scientificamerican.com/article/elon-musks-ai-chatbot-grok-is-reciting-climate-denial-talking-points/)。
   - 另一位用户对此表示反对，称许多人喜欢 **Grok** 是因为它提供的自由度，能提供与其他模型不同的视角。
- **Qwen 被视为开源编程冠军**：**Qwen** 被认为是编程领域最好的开源模型，但一位用户在使用 **raptorwrite** 时无法从该模型获取响应。
   - 此外还观察到，**OpenRouter** 上的所有 **Anthropic 端点** 都有 OpenRouter 审核，*除了自我审核 (self moderated) 的端点*。
- **Nous 分布式训练 SOTA 模型**：**Nous** 正尝试利用 [Psyche.network](https://psyche.network/runs/consilience-40b-1/0) 和 [Bittensor](https://docs.bittensor.com/emissions) 分布式训练一个 SOTA 模型。
   - 该模型在有限的 GPU 间带宽（约 300mbps）下进行训练，但在吸引足够的 GPU 加入方面面临挑战，目前仅有 416 块 H100 在线。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1379176547436724235)** (219 messages🔥🔥): 

> `Cursor 主题与插件, Cursor 使用资金, Claude 模型性能与成本, Bugbot 发布, 上下文窗口限制` 


- **用户展示 Cursor 主题和插件**：用户分享了他们的 Cursor IDE 主题和插件，包括用于背景的 **background-cover**、用于图标的 **Material Icon Theme** 以及用于配色方案的 **Monkey Pro**。
   - 一位用户提到在 Monkey Pro 主题中使用了 "Material Theme Icons Darker" 和 "Filter Ristretto"，并分享了他们的配置截图。
- **社区请求本地货币结算以方便充值**：一位用户请求 Cursor 团队实现本地货币结算，以增加获得 Cursor 使用资金的可能性。
   - 该用户在消息中艾特了 Cursor 团队成员，以引起对该请求的关注，并强调了这对客户和品牌曝光的潜在好处。
- **Opus 4 价格不菲**：一位用户报告称，在 Cursor 中使用 **Opus 4 Max** 消耗了 **69.5 次请求**，相当于单条消息约 **$2.73**。
   - 尽管成本很高，但他们认为这很值得，因为它解决了一个 **Sonnet 4** 和 **Gemini** 无法解决的 **Postgres 瓶颈**。
- **账单门户出故障了？**：用户报告称新的账单门户无法准确显示基于使用的计费，请求计数存在差异。
   - 一位用户注意到他们的包含请求周期与日历月份不一致，且分析图表滞后了一两天。建议查看使用情况页面以获取更详细的信息。
- **聊天中断困扰用户**：用户报告在与 **Claude 4 Sonnet** 对话时频繁出现中断，通常在单次请求或约 **25 次工具调用 (tool calls)** 后触发。
   - 这些中断导致提示开启新聊天、上下文丢失以及网络连接错误，干扰了工作流程并引起不满。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1379238336832733414)** (7 messages): 

> `自定义 Dockerfile, 后台 Agent 限制, 商业版计划与后台 Agent 的隐私模式` 


- **自定义 Dockerfile 的烦恼**：一位用户询问关于使用包含多个服务的 `docker-compose.yml` 文件的问题，指出他们只看到了“选择自定义 Dockerfile”的选项。
   - 他们正在寻找一种方法来设置由 `docker-compose.yml` 文件定义的**多服务**环境，但不确定在现有选项中如何操作。
- **后台 Agent 创建文件故障**：一位用户报告后台 Agent 似乎无法创建新文件，并询问这是设计使然还是 Bug。
   - 目前尚不清楚无法创建文件是设计的限制，还是后台 Agent 意外出现的问题。
- **隐私模式升级僵局**：一位用户询问在升级到商业版计划后，是否可以在启用隐私模式的情况下使用后台 Agent。
   - 另一位用户回答说该功能可能尚未推出，表明后台 Agent 与隐私模式的兼容性尚不确定，即使是在商业版计划中。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1379175529646915815)** (105 messages🔥🔥): 

> `SentinelAI DeFi 审计, NER 模型训练技巧, Tesseract 微调, Mikrotik 文档聊天模型, 用于多发言人场景的 LLM` 


- **SentinelAI 像大佬一样审计 DeFi 合约**：根据[此帖](https://x.com/Shravankumar8_/status/1929616447845826763)，**SentinelAI** 正在审计 DeFi 合约并捕捉重入 (reentrancy) 问题。
- **云端训练的高效数据存储讨论**：针对在 Hugging Face 上存储数据集进行云端训练的最有效格式展开了讨论，考虑了 **10万-20万** 与 **100万+** 的数据集规模。
- **辩论微调实验讨论及反馈征集**：一位成员分享了他们在 **辩论微调 (debate fine-tuning)** 方面的工作，邀请大家提供反馈以推进研究，并附上了一篇详细介绍该过程的 [Medium 文章](https://medium.com/@whatsupai/from-zero-to-debate-fine-tuned-llama-a-step-by-step-build-log-13d950f054fb)。
- **Surya 凭借 SOTA 级别的独立 OCR 胜出**：成员们发现 **Surya** 正是他们所需要的，提到它可以实现 SOTA 级别的独立 OCR，并称赞该成员的工作非常出色且具有启发性。
   - 分享了该项目的链接：[VikParuchurii](https://github.com/VikParuchurii)。
- **恭喜，你进入了 AI 世界杯！**：一位成员庆祝他的 Space 被评为 *Spaces Of The Week*（本周最佳 Space），另一位成员回应称入选本周最佳 Space 是一件了不起的大事。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1379414741545385984)** (4 messages): 

> `Transformers Training, Hugging Face YourBench` 


- **Transformers 训练实验带来新见解**：一位成员正在进行 **transformers training** 实验并更新 **pipeline calls**，这导致文本生成模型产生了一些有趣的响应，如[附图](https://cdn.discordapp.com/attachments/897390579145637909/1379414741298057339/image.png?ex=6840d056&is=683f7ed6&hm=de69d11987ed857cb86a656536c75db32c801b287fc82fa62594415622af38f7&)所示。
- **Hugging Face 的 YourBench 计划受到关注**：一位成员重点介绍了 [Hugging Face's YourBench](https://huggingface.co/yourbench) 计划，并指出这是一个*被严重低估*的资源。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1379194899970396280)** (6 messages): 

> `Tech Events Platform Survey, Text Diffusion Model, Meilisearch Chat Route` 


- **构建面向学生的技术活动平台**：一位成员正在构建一个用于**查找和分享技术相关活动**的平台，并正通过[一份调查问卷](https://tally.so/r/melldO)征求意见以完善产品。
   - 该平台针对**学生和应届毕业生**，填写问卷仅需 **2 分钟**。
- **Text Diffusion 生成奇特输出**：在笔记本电脑上训练 **text diffusion model** 后，一位成员注意到当终端显示设置不当时，会产生有趣的输出。
   - 另一位成员提到，他们看到有人在 Twitter 上分享了该成员的代码作为引用。
- **Meilisearch 发布 Chat 路由**：Meilisearch 将于下周发布实验性的 **/chat route**，旨在帮助开发者快速构建 **AI-powered features**（如 RAG 和对话界面）的原型。
   - 他们正在寻找几位开发者进行早期试用，并为私信（DM）的开发者提供入门帮助。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1379437650058149920)** (1 messages): 

> `Reading Group Events Calendar` 


- **出现日历订阅请求**：一位成员询问是否有可订阅的**读书小组活动**日历。
- **可用性回复待定**：截至最新消息，尚未提供日历链接或关于可用性的直接回复。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1379526345842888717)** (1 messages): 

> `Vision Transformers, Multi-Modal Models, Image Tokenization, Zero-Shot Classification` 


- **寻求 Vision Transformer 课程推荐**：一位成员询问有关理解 **Vision Transformers** 和 **multi-modal models** 的推荐课程，特别是 **image tokenization** 如何与 **text tokens** 交互。
   - 他们提到了上传多张图像并与模型讨论任务的能力，旨在更好地了解模型如何针对 **zero-shot classification** 对图像进行“推理”。
- **利用 Zero-Shot Classification 对图像进行推理**：一位成员正在寻求一种方法，使模型能够“关注”图像的某些部分，同时将图像的其余部分作为 **zero-shot classification** 的相关上下文。
   - 他们的用例涉及移除图像的一个区域并分类该区域中的内容，利用周围的上下文辅助模型的推理。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1379204550430560258)** (1 messages): 

> `Embedding Models Fine-Tuning, SOTA Embedding Techniques, Embedding Evaluation Metrics` 


- **寻求 SOTA 嵌入微调技术**：一位成员询问了关于微调 **embedding models** 的 **state-of-the-art (SOTA)** 技术。
   - 他们还询问了用于比较基础模型与微调版本的标准 **metrics**。
- **关于嵌入模型评估的讨论**：对话还集中在评估微调嵌入效果的方法上。
   - 具体而言，参与者试图确定用于评估相对于基准模型改进情况的可靠指标。


  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1379463375687516355)** (1 messages): 

> `Agents & MCP Hackathon, YouTube Stream` 


- **Agents & MCP Hackathon 正式启动直播**：**Agents & MCP Hackathon** 正在通过 [YouTube 直播](https://www.youtube.com/watch?v=MU7FyxSnCp4)拉开帷幕。
   - 该直播旨在为活动争取支持。
- **YouTube 直播支持 Hackathon**：已启动 YouTube 直播以支持 **Agents & MCP Hackathon**，为参与者和爱好者提供了一个互动的平台。
   - 该直播旨在促进社区支持并提供关于 Hackathon 进展的最新动态。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1379430923120152576)** (2 messages): 

> `Accessing Attachments, agents-course-unit4-scoring.hf.space` 


- **通过 GET /files/{task_id} 访问附件**：一位成员询问如何在 [agents-course-unit4-scoring.hf.space/questions](https://agents-course-unit4-scoring.hf.space/questions) 中访问附件或文件名。
   - 该成员随后找到了答案：使用位于 [agents-course-unit4-scoring.hf.space/files/task-id](https://agents-course-unit4-scoring.hf.space/files/task-id) 的 **GET /files/{task_id} Endpoint**。
- **文件访问说明**：端点 `/files/{task_id}` 允许检索与特定任务 ID 关联的附件。
   - 这为访问平台上提出的问题相关的文件提供了一种直接方法。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1379183314157699183)** (21 messages🔥): 

> `Windows vs Linux for course, Image generation issues, Open Source Chat UI for Ollama, Course Deadline, Agent Planning Issues with LLMs` 


- **Windows 在课程操作系统要求方面受到质疑**：一位用户询问是否可以在没有 **WSL** 的 **Windows 10** 上完成课程，因为课程似乎假设了 **Linux** 环境，但 **WSL** 无法与 **VMware Workstation** 同时运行。
   - 另一位用户确认在 **Windows 11** 上完成了课程。
- **图像生成受显示问题困扰**：一位用户报告称，图像生成工具生成的图像最初显示为 **1024x768**，但在最终回答步骤中变为 **0x0**，且该应用在 Chrome 中无法加载。
   - 他们尝试保存图像并将其传递到最后一步，并且能够在 **Edge** 中打开其他人的应用。
- **开源聊天 UI 搜索升温**：一位用户正在寻找类似于 **ChatGPT** 或 **Le Chat** 的开源聊天 UI，以便与本地运行的 **Ollama LLM** 进行交互。
   - 其他成员提到在 Windows 上使用 **LM Studio**，并建议非 Windows 用户使用 **AnythingLLM**。
- **Agent 课程截止日期引发恐慌**：一位新用户担心 **7 月 1 日** 的课程截止日期，询问在没有经验的情况下是否仍有可能完成。
   - 有经验的成员建议先完成 **LLMs** 课程，但相信他们仍然能够完成本课程。
- **Agent 规划受阻：需要更好的 LLM？**：一位用户发现他们的 Agent 迷失在过多的步骤中，并怀疑使用更好的 **LLM** 是否能改善规划。
   - 他们指出使用 **Ollama** 以及 **llama3** 或 **qwen3** 的效果不佳。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1379501191016484904)** (1 messages): 

> `Sequential Monte Carlo, Parallelized inference server, Entropy based triggering, Control vectors` 


- **Nous Research 发布序列蒙特卡洛博客文章**：Nous Research 发布了一篇关于 [序列蒙特卡洛 (SMC)](https://nousresearch.com/steering-the-shoggoth-taming-llms-with-sequential-monte-carlo/) 的博客文章，这是一种使用多个“粒子”进行采样、加权并根据评分函数重新采样的技术，用于生成符合约束的补全内容，解决了控制文本生成和结构的问题。
   - 该博客介绍了一个 **并行化推理服务器**，使用户能够快速对约束设计进行基准测试，并附带了 **基于熵的触发** 和 **控制向量** 的示例实验。
- **SMC 推理服务器代码现已在 GitHub 上线**：[并行化推理服务器](https://github.com/NousResearch/smc-inference-server) 的代码已经发布，允许用户在使用 **序列蒙特卡洛 (SMC)** 时对其约束设计进行基准测试。
   - 该发布包含了 **基于熵的触发** 和 **控制向量** 的示例实验。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1379177543957221438)** (104 messages🔥🔥): 

> `Agent 通信问题, 适配 Agent 世界观, Gemini 超大上下文长度, OpenRouter Deep Research MCP, HF Gradio Hackathon` 


- **Agent 互相“咆哮”造成混乱**：一位工程师正面临 Agent 之间互相冲突（yelling）的问题，这破坏了时序和协调，导致不规则性。他正在考虑将 [Adaptive World View Agents](https://www.google.com/search?q=Adaptive+world+view+agent) 作为潜在解决方案。
   - 他们目前的方案包括加载了数据的 **LLM**、自定义指令以及具有语义识别功能的 **RAG engine** 以实现上下文相关性。但他们正在探索替代方法，例如使用 ADAWORLD 通过颜色、时序和几何形状将上下文编码到图像中。
- **WholeToast 推荐 Gemini 的超大上下文并提供 Research MCP**：为了解决 Agent 记忆丢失问题，一位成员建议使用具有超大上下文长度的 **Gemini**，并将 embedding 存储在向量数据库中。
   - 他们还分享了其 [OpenRouter Deep Research MCP server](https://github.com/wheattoast11/openrouter-deep-research-mcp) 的链接，该服务器使用 **3 个 Agent**（Context, Planning 和 Research），并启动一个 pglite postgres 数据库来存储可查询的研究结果。他指出 **Gemini** 可以连贯地处理 900k+ token。
- **Hugging Face 举办黑客松并提供奖品**：一位成员分享了 [Hugging Face Gradio Hackathon](https://huggingface.co/spaces/ysharma/gradio-hackathon-registration-2025) 的链接，强调该活动提供丰厚的额度，非常适合构建 Agent 或 Agentic 应用。
   - 另一位成员发布了他的第一篇技术博客，包含动手练习、数学推导和交互式可视化。
- **新型 PEFT 方法展现巨大潜力**：一位成员正在寻求关于一种新型**参数高效微调**（PEFT）方法的反馈，该方法主要针对持续预训练（continued pretraining）。据报告，与全量微调和 LoRA 相比，该方法在参数更少的情况下，知识吸收率提高 **4 倍**，灾难性遗忘减少 30%。
   - 其他人对声称的性能提升表示怀疑，并索要该方法。
- **关于 SME 模型集成与通用动态分区的辩论**：一位成员建议，由共享万亿 token 潜空间的小型专业模型组成的集成（类似于万亿专家混合模型）将是最佳选择，每个小模型在系统中充当神经元，自然地学习作为一个整体运作。
   - 他们强调了**语义场**（semantic fields）对于基于稳定性参数的专家动态区分的重要性，并提到使用吸引子/排斥子（attractors/detractors）来塑造这一过程。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1379652931913257090)** (3 messages): 

> `Nous Hermes, Loom` 


- **Loom 获得 Nous Hermes 认可**：一位成员准备尝试 [Loom](https://github.com/socketteer/loom)。
   - 另一位成员建议使用 **Hermes 70b** 来运行它。
- **Hermes 70B 获得推荐**：Teknium 推荐使用 **Hermes 70B** 进行测试。
   - **Hermes 70B** 被认为是顶级模型。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1379182445802295319)** (1 messages): 

> `LLM Scribe 工具, 手写数据集` 


- **LLM Scribe 助力简化数据集创建**：一位成员创建了一个工具，旨在简化用于微调的**手写数据集**（handwritten datasets）的创建，支持 **ChatML**、**Alpaca** 和 **ShareGPT** 等多种格式。
   - 它包括自动保存、多轮对话创建、从 Hugging Face 加载的 token 计数器、目标跟踪和自定义字段；可以查看 [Hugging Face demo](https://huggingface.co/spaces/Gabriella0333/LLM_Scribe_Demo)、[视频演示](https://www.youtube.com/watch?v=1mcYsDrXHAA&t=3s) 以及 [完整版本](https://kryptive.gumroad.com/l/gvyqep)。
- **Scribe 工具：支持多种格式**：该工具支持 **ChatML**、**Alpaca** 和 **ShareGPT** 等各种格式，使其在满足不同微调需求方面具有极高的通用性。
   - 它通过提供自动保存、多轮对话创建以及直接从 **Hugging Face** 加载的 token 计数器等功能，简化了数据集创建过程。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1379208426768433255)** (109 messages🔥🔥): 

> `Manus API release, College list, Manus credit usage, Video generation capabilities in Manus, AI Act implications` 


- **Manus API 仍遥遥无期？**：一位成员询问了 **Manus AI API** 的发布时间表，但被告知目前 *没有计划*，但这在未来可能会改变。
- **Santa Fe 不在 Manus 的 School Pass 大学名单中**：一位用户询问了 **School Pass** 的大学名单，特别是关于 **SF College**。
   - 一名工作人员澄清说 **Santa Fe** 在 **Manus Campus 名单**中，这与 **School Pass** 不同，并确认它不在后者名单上。
- **每日免费额度消耗顺序待修复**：一位成员建议 **Manus** 应该先消耗 **每日免费额度** 再使用付费额度，认为目前的系统 *有点坑 (scammy)*。
   - 另一位成员解释了额度消耗顺序：活动额度 > 每日免费额度 > 每月额度 > 附加额度 > 免费额度，并建议澄清措辞。
- **Manus 视频生成能力引发讨论**：用户讨论了 **Manus** 的视频生成能力，对其质量与 **Veo** 等工具的对比意见不一。
   - 一位用户声称 **Gemini** 可以免费执行类似任务，而另一位则称赞 **Manus** 是发布后 *最顶级的承诺*。
- **深入探讨 AI Act 的影响**：一位用户提到了 **意大利和欧洲** 的 **AI Act**，另一位用户报告了一次令人沮丧的经历：**Manus** 在运行 1 小时 55 分钟后因达到上下文限制而停止任务，并在继承压缩后的上下文后从头开始。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1379206875635253338)** (58 messages🔥🔥): 

> `Muon in GANs, Website Content Updates, CBRN and Cybersecurity Risks from LLMs, Parameter-Efficient Finetuning, Learning Wet Lab Skills via LLMs` 


- **Muon 优化器在 GANs 中表现不佳**：成员们讨论了在 **生成对抗网络 (GANs)** 中使用 **Muon 优化器**，但一位用户报告它 *不起作用*，因为 GANs 需要缓慢学习，并对 Muon 在没有动量 (momentum) 的情况下表现良好持保留意见。
- **Eleuther 网站内容陈旧引发辩论**：一位用户询问网站缺乏更新的问题，但一位成员表示该网站主要是给非 **ML 研究员** 看的广告，也是通过 Google 搜索找到项目的简便方法，很快就会更新。
   - 他们强调网站不是主要的活动中心，因为大部分活动发生在 **Discord** 社区。
- **LLMs 引发 CBRN 和网络安全担忧？**：成员们对与 **大语言模型 (LLMs)** 相关的 **化学、生物、放射性和核 (CBRN)** 以及 **网络安全** 风险表达了不同程度的担忧。
   - 一位成员引用研究表明，围绕 CBRN 的炒作被夸大了，网络安全风险更多源于实施不当（模型拥有过多权限）而非 LLMs 本身；另一位成员认为 CBRN 威胁的瓶颈是 *获取物理材料和专业知识*，而非知识。
- **针对特定遗忘的快速基础模型精细微调**：一位成员介绍了一种新的 **参数高效微调 (parameter-efficient finetuning)** 方法，旨在进行领域自适应和知识添加，同时最大限度地减少灾难性遗忘，声称与全量微调和 **LoRA** 相比，*知识吸收率提高 4 倍*。
   - 他们征求了关于该方法在本地设置中的潜在用途和需求的反馈。
- **LLMs 在实验室学习中的局限性凸显**：成员们辩论了 **LLMs** 是否能有效教授湿实验室 (wet lab) 技能，一位用户认为 *湿实验室技能不是通过阅读就能学会的*，并强调了隐性知识和动觉智能的重要性。
   - 相比之下，另一位成员建议 **LLMs** 可以提供极其详尽的指令，但引用了 [一篇论文](https://arxiv.org/abs/2503.09722) 反驳说，即使观察别人绘画也可能不足以学会绘画，强调了覆盖广泛的专家轨迹和潜在错误的必要性。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1379192845486788710)** (10 messages🔥): 

> `Generative Inverse Problems, T5 Diffusion Decoder, Qwen RL Papers, AI Rights Document` 


- **Transformer 通过 Patch 处理生成式逆问题**：对于使用 Transformer 的生成/逆问题，模型按 **patches** 而非像素进行排序以保留信息，并可选择[分离通道](https://example.com)。
   - 这种方法被认为更优，因为它*保留了一些信息*。
- **T5 Diffusion 解码器诞生**：为了创建 Diffusion 版本，可以实现带有 Diffusion 解码器的 **T5**，正如在 **T5** 或 **UL2 MLM** 中一样，模型预先并不知道要填充（infill）的 token 数量，参见 [Moyix on X](https://x.com/moyix/status/1812902249196109955)。
- **Qwen 获得强化学习（RL）增强**：一位用户分享了论文链接，其中一篇关注 Qwen RL，另一篇（后来被划掉）也是关于 [Qwen RL](https://arxiv.org/abs/2505.17083) 的。
- **Qwen RL 泛化性存疑**：似乎一篇 Qwen RL 论文的泛化效果不佳，一位成员观察到 *看起来他们只测试了 Qwen，所以可能无法泛化*。
   - 该用户宣称 *这也是我最近对所有 RL 论文的默认假设*。
- **AI 权利文档的趣味探索**：一位成员建议将一份关于 **AI 权利** 的文档针对 *科幻电影和概念进行测试，或针对现实世界场景进行剧本测试*，以获得有趣的视角，并分享了 [UDAIR.md](https://cdn.discordapp.com/attachments/747850033994662000/1379667427591192687/UDAIR.md?ex=684112eb&is=683fc16b&hm=27486eaf856c6e95256f443915e85a0a18c59c8da2685884a63088525c567013&) 的链接。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1379480434873733292)** (1 messages): 

> `AI Bubble Plot` 


- **AI 泡沫图暗示估值过高**：一位用户分享了一张 [AI 泡沫图表](https://cdn.discordapp.com/attachments/785968841301426216/1379480435041632266/image.png?ex=68410d85&is=683fbc05&hm=d602f5153beb6a89f5a38ebfbd0022be3eb1b77bf3fa9e0957e758a4dfecbf5a&)，暗示 AI 市场可能存在估值过高的情况。
   - 该图表直观地将 AI 描绘为一个泡沫，暗示了过高的预期或不可持续的增长。
- **关于 AI 泡沫图的更多看法**：一些观察者认为，关注市场回调非常重要，尤其是当估值飙升时。
   - 其他人则警告说，宏观经济因素可能会放大风险。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1379191707274448896)** (4 messages): 

> `Neural Network Manifolds, MechInterp Ideas` 


- **探索神经网络的流形之谜**：一位成员提出，任何在自然输入的低维流形上训练的神经网络，都自动对应于嵌入在给定架构可能的前向传播（forward passes）高维空间中的某种激活低维流形。
   - 随后他们考虑如何 *商去（quotient out）* 数据集的规律性，以便获得仅由权重施加的模型行为规律性流形。
- **MechInterp 想法涌现**：一位成员在[这份长文档](https://docs.google.com/document/d/1rpZvbSh4IwcZnVe-mZu4tPt4CSAkkgbEsyjd42tziNc/edit?usp=sharing)中分享了他们关于 MechInterp 的想法。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1379465870409076787)** (1 messages): 

> `transformers library, max_position_embeddings` 


- **在 Transformers 库中检查 max_position_embeddings**：一位成员指出，检查是在 [transformers 库](https://github.com/huggingface/transformers/blob/e8b292e35f331d3c3de85f7e5d3496b0e13d3d6f/src/transformers/generation/stopping_criteria.py#L79-L80)中的 `max_position_embeddings` 上进行的。
- **再次确认 max_position_embeddings**：另一位成员澄清说，`max_position_embeddings` 确实是 transformers 库中被检查的参数。

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1379540513094500384)** (2 messages): 

> `Pythia Remake, Scaling Suite Experiment` 


- **关于 Pythia 重启的思考**：一位成员询问了从 **Pythia** 开发中吸取的教训，因为有人计划进行重制，并链接到了一个 [Scaling Suite 实验](https://marin.community/data-browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp1337_scaling_suite-a2518e.json)。
   - 另一位成员提到，在看到相关推文后，他们正准备发布关于这一话题的评论。
- **Scaling Suite 实验引起关注**：提供的 [Scaling Suite 实验链接](https://marin.community/data-browser/experiment/?path=gs%3A//marin-us-central2/experiments/exp1337_scaling_suite-a2518e.json) 引起了社区的兴趣。
   - 该实验可能包含与模型扩展（scaling）相关的宝贵数据和见解，引发了关于潜在改进和未来研究方向的讨论。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1379404011161124894)** (29 messages🔥): 

> `NixOS for ML/DS, Temperature 0 Debate, Parameter-Efficient Finetuning, MCP Server Needed, Isomorphism for Computation` 


- **NixOS 贡献者寻求 ML/DS 优先级建议**：一位数据科学家兼 **NixOS** 贡献者正寻求改进 **ML** 和 **DS** 中的声明式、不可变和可复现实践，并征求关于痛点和优先级的建议。
   - 他们强调 *"I use nixos btw" 是唯一能吓跑 Arch 用户的话*，并且 **Nix** 是一个强大的工具，无论使用何种操作系统，都能显著改善开发/部署。
- **Temperature 0 阻碍思考模型**：一位成员回想起曾读到 Temperature 0 并非最优，建议在特定任务中使用低但非零的 Temperature。
   - 另一位成员确认 Temperature 0 会导致思考模型（thinking models）出现重复，并建议思考模型使用高 Temperature 以更广泛地探索树结构，并引用了 [Qwen 的文档](https://huggingface.co/Qwen/Qwen3-8B)，其中指出 *"不要使用贪婪解码（greedy decoding），因为它会导致性能下降和无休止的重复"*。
- **参数高效微调提高知识吸收**：一位成员介绍了一种新的参数高效微调（parameter-efficient finetuning）方法，主要针对持续预训练，与全量微调和 **LoRA** 相比，知识吸收率提高了约 4 倍。
   - 另一位成员表示有兴趣使用他们收集的书籍和文档来扩展知识，看看是否比类 **RAG** 方法更有优势。
- **成员需要用于同构测试的 MCP Server**：一位成员正在寻求帮助，以寻找或创建一个 **MCP server** 来测试他们正在研究的一种同构（isomorphism）。
   - 初步测试表明，使用更少的资源和更短的时间可以获得 **99%** 相似的结果。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1379222205220782100)** (25 messages🔥): 

> `FP8 Training, SwiGLU Activation Function, vec2vec code review, Interest in Daily Paper Discussions` 


- **FP8 训练扩展至万亿 Token 级 LLM！**：一场关于论文 [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) 的讨论已安排，该论文成功在高达 **2 万亿 Token** 的数据集上使用 **FP8 精度** 训练了大语言模型。
- **Smooth-SwiGLU 稳定 FP8 训练！**：该论文指出 **FP8 训练** 中的不稳定性是由 **SwiGLU 激活函数** 的离群值放大引起的，并引入了 **Smooth-SwiGLU** 来解决此问题而不改变函数行为。关于 SwiGLU 的博客：[jcarlosroldan.com](https://jcarlosroldan.com/post/348)。
- **vec2vec 代码审查取消**：原定对 [vec2vec](https://github.com/rjha18/vec2vec)（[arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540) 的一种实现）的代码审查因缺乏兴趣而取消，该活动需要 *在开始后 10 分钟内至少有一人参加。*
- **呼吁参与论文讨论！**：一位成员表示打算更多地参与**每日论文讨论**，但指出由于过度炒作（hype）而存在困难，他们更倾向于来自可靠来源或经受住时间考验（约 1 周）的论文。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1379393670788157460)** (10 messages🔥): 

> `Google Gemini open source, Search vs Deep research, OpenAI status, 1984 reference` 


- **Google Gemini Fullstack Langgraph Quickstart 开源！**: Google 开源了 [Gemini Fullstack Langgraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)，尽管其确切用途尚不明确。
   - 一位成员推测，这可能是一种*让模型思考更长时间*的方法。
- **区分 Search 与 Deep Research**: 一位成员指出，这个新的开源项目似乎更倾向于快速搜索功能，而不是需要更多处理时间的深度研究（Deep Research）。
   - 尽管如此，该项目仍被认为非常*酷*。
- **分享 OpenAI 状态**: 一位成员分享了 [OpenAI status](https://x.com/OpenAI/status/1929957365119627520) 以及指向 [tweets](https://x.com/andersonbcdefg/status/1930000529012470024?s=46) 和 [另一条推文](https://x.com/unusual_whales/status/1929998955703931375) 的链接。
- **《1984》场景的高概率**: 一位成员评论称，出现《1984》场景的概率很高，大概是指当前事件中的监控或反乌托邦主题。
   - 未提供更多上下文。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1379195123321274429)** (3 messages): 

> `Modular Hackathon, GPU programming workshop, Mojo kernels, MAX Graph model architectures, PyTorch custom ops` 


- **Modular 举办另一场 Hackathon**: Modular 正在举办另一场专注于 **Mojo kernels**、**MAX Graph 模型架构**和 **PyTorch custom ops** 的 Hackathon ([https://lu.ma/modular-hack-weekend](https://lu.ma/modular-hack-weekend))。
   - 本次 Hackathon 将开放虚拟参与，并在周末举行，合作伙伴、评委和奖项公告即将发布。
- **宣布 GPU 编程研讨会**: 为了启动 Hackathon 周末，Modular 将在他们的 Los Altos 办公室举办 **GPU 编程研讨会**，并同步进行虚拟直播。
   - 该研讨会旨在让人们熟悉他们在 Hackathon 中将使用的技术。
- **新社区成员对 Mojo 感到兴奋**: 一位刚从计算机科学研究型硕士项目毕业的新成员在看了 **Fireship 视频**后尝试了 Mojo。
   - 他已经看到了基础 ML 模型的改进，并提到作为 UIUC 的毕业生，他非常熟悉 **Chris Lattner 与 Vikram Adve 在 LLVM 上的工作**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1379202089078493217)** (58 messages🔥🔥): 

> `C->Mojo bindings generator, unsigned _BitInt(13), mojo + libclang, Mojo multithreading` 


- **C 到 Mojo 绑定生成器正在开发中**: 一位成员正在开发 **C->Mojo 绑定生成器**，并试图找出如何在不使用诡异变通方法的情况下，让 Mojo 编译器导入/导出对象文件。
   - 他们表示，除了极其糟糕的 packed structs 以及可能在 `restrict` 周围的一些棘手问题外，几乎所有必要的东西都已存在；此外，影响调用约定的 `pragmas` 将是一个难以解决的痛点。
- **解析 C 代码的复杂性**: 一位成员指出，如果他们没见过那么多*技术上符合规范*的 C 代码，他们的进度可能会快得多。
   - 另一位成员为使用了 `unsigned _BitInt(13)` 等写法表示歉意。
- **Clang AST Dumps 馈送 Mojo AST 记录**: 一位成员接近于通过一个通用命令（利用人们可能已经拥有或可以生成的东西）获得文件的 Mojo 可编译版本，即从 Clang compiledb 中提取，然后将其馈送给 **libclang** 以获取 **AST**。
   - 他们基本上是在执行：`clang -Xclang -ast-dump -fsyntax-only -fparse-all-comments -fno-color-diagnostics somefile` -> AST 节点。
- **Mojo 缺乏手动线程管理**: 目前 Mojo 中还没有**手动线程管理**，也没有给出时间表，但据一些人估计可能是今年晚些时候或明年。
   - 像代数类型（algebraic types）这样的类型系统尚未最终确定，仍在摸索中。一些人指出，这只是 **v0.3** 版本，还有很多非常重要的东西缺失，比如可用的原子操作（atomics）和同步原语、线程安全标记 trait、基础 IO、基础数据结构以及网络能力。
- **结构体从 Host 传输到 Device**: 成员们讨论了是否可以使用 buffer 将任意结构体从 Host 传输到 Device，还是仅限于原始类型。
   - 另一位成员回答说**目前有限制**，但正在进行关于 **“trivial types”** 的工作，这将允许传输任何没有指针或特殊析构函数的内容。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1379200271728640141)** (55 条消息🔥🔥): 

> `DSPy 在 AI Engineering 和 Databricks DAIS 的演讲、DSPy 使用案例、6 月发布的 DSPy 3.0、DARPA 高级研究概念实验室使用 DSPy、将自定义提示词迁移到 DSPy` 


- **DSPy 演讲即将到来并征求意见**：团队正在为 **AI Engineering** 和 **Databricks DAIS** 的 DSPy 演讲做准备，并征求社区关于演讲主题和幻灯片中重点展示的使用案例的建议。
   - 根据[这条消息](https://discord.com/channels/1164929493932544091/1164941339336202311/1241058336820895804)，DSPy 3.0 版本定于 6 月发布。
- **DSPy 助力 DARPA 项目孵化**：DSPy 被用于 **DARPA 高级研究概念实验室 (Advanced Research Concepts lab)**，为“协作知识策展 (Collaborative Knowledge Curation)”兴趣领域构建解决方案，该项目目前正在孵化成一家公司。
   - 这表明了 DSPy 在高级研究环境中的实际应用和能力验证。
- **非常规的 DSPy 工作流**：一些人发现将现有的 **GenAI flows** 重构为 DSPy 似乎是一项重大变革，因为它被设计为在生产环境中在线使用，作为处理 GenAI 工作流的框架，且构建 DSPy 工作流的方式非常独特。
   - 这凸显了对于将 DSPy 集成到现有代码库和工作流中需要更多指导的需求。
- **具有一等环境支持的 Agent 框架**：人们对在 DSPy 之上构建 **agent framework** 表现出浓厚兴趣，该框架包含一等环境 (first-class environments)，并通过优化器处理在线学习的自我奖励和外部奖励。
   - 一位成员表示：*“我不明白为什么那些做 Agent 框架的人不这样做，而是在忙其他各种事情。”*
- **带有 Python 绑定的 Claude Code 框架**：一位成员正在使用 **Claude code** 构建一个 Agent 框架，目标是提供 Python 绑定和 Rust 核心，并具有追溯性的 Trace 表示。
   - 该框架旨在使 Trace 可见，并能很好地进行分叉 (forking)，以便针对任意指标进行优化；该项目已在 GitHub 上发布为 [claude_sdk](https://github.com/darinkishore/claude_sdk/tree/t1-execution-engine)。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1379184708818309231)** (46 条消息🔥): 

> `ClaudeCode、Aider 会话恢复、Aider 询问模式、Aider 重启、Gemini 2.5` 


- **ClaudeCode 内部机制揭秘**：一位成员分享了一个深入探讨 **ClaudeCode** 的[链接](https://gerred.github.io/building-an-agentic-system)，详细介绍了构建自动驾驶编程 Agent 和执行引擎所涉及的系统、工具和命令。
   - 该文章强调了在创建实时、具备自我纠错能力且对生产工作有用的 Agent 时所涉及的系统和设计决策，而不仅仅是关注 Prompt 和 AI Engineering。
- **恢复 Aider 会话可保留上下文！**：用户讨论了通过恢复 **Aider** 会话来保持上下文记忆，其中一位指出可以使用 `--resume` 标志，但他们尚未完全掌握其用法。
   - 他们希望能够通过 ID 恢复旧会话或检查点，而另一些人则为了清除上下文而频繁重启。
- **抑制 Aider `/ask` 模式下的代码建议**：一位用户对 **Aider** 的 `/ask` 模式感到沮丧，因为即使他们只是想交流或规划，该模式也经常建议修改代码。
   - 建议的解决方案包括明确告诉 **Aider** *“先不要写任何代码”*，或者使用 `/reminder` 命令来设定规划阶段。
- **新 Gemini 模型基准测试**：一位成员对一款即将发布的模型进行了基准测试，在 diff-fenced 模式下达到了 **86.2%**。
   - 有推测称这可能是 **Gemini** 模型，但人们担心 **Gemini-2.5-pro** 版本的命名可能会引起混淆；这种困惑导致人们呼吁重新命名或推迟发布以避免误解。
- **追踪 Claude 的思绪！**：一位成员分享了 [Simon Willison 关于 Claude Trace 的博文](https://simonwillison.net/2025/Jun/2/claude-trace/)，这可能对 aider 开发者很有启发。
   - 该博文概述了在代码执行期间追踪 Claude 思考过程的能力。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1379336611728326717)** (5 messages): 

> `Aider 运行与测试命令，Bedrock vs Anthropic 模型，Aider 项目重新初始化` 


- ****Aider 的运行与测试命令****：新用户询问如何提示 Aider 运行测试命令并分析输出，被引导至 Aider 文档中的 [`/run`](https://aider.chat/docs/usage/commands.html) 和 [`/test`](https://aider.chat/docs/usage/lint-test.html) 命令。
- ****Bedrock vs Anthropic 模型命令执行****：一位用户观察到，当使用 **Bedrock Claude 3 Sonnet** 模型时，Aider 可以成功执行删除文件等终端命令，但当使用 **Converse Claude Sonnet** 模型时，它仅提供帮助而不执行。
   - 他们询问是否存在影响终端命令使用能力的设置限制，特别是在 Bedrock 的情况下？
- ****文件夹移动后的 Aider 项目重新初始化****：一位用户报告称，在移动项目文件夹后 Aider 运行失败，因为它仍在旧位置寻找 `config.lock` 文件，即使删除了仓库中所有的 Aider 文件也是如此。
   - 他们询问 Aider 是否在系统的其他地方使用了缓存，以及是否有办法**重新初始化** Aider 项目。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1379507915144040478)** (1 messages): 

> `公开 Notebooks，NotebookLM` 


- **NotebookLM 公开分享**：用户现在可以使用 [公开链接](https://blog.google/technology/google-labs/notebooklm-public-notebooks/) 与任何人策划并分享他们的 notebooks。
   - 这是一个分享你的工作并造福社区的好机会。
- **展示 NotebookLM 技巧**：鼓励成员分享他们最引以为傲的 notebook 链接。
   - 目标是分享能让社区受益的 notebooks。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1379259643758383114)** (8 messages🔥): 

> `播客摘要长度差异，Microsoft Learn，Notebook LM，使用案例` 


- **播客摘要长度差异等待修复**：用户仍在等待非英语语言的 [播客摘要](https://cdn.discordapp.com/attachments/1124403655819415592/1379259643406188644/68E28299-FA17-4281-BE92-CA19750B336B.png?ex=6840e8a4&is=683f9724&hm=2106850cce3163a137a37a2a85568c502b7c43d109944be92a832f4f6a6cab43) 修复，指出即使使用相同的提示词和内容，它们的长度也不及英文摘要。
   - 另一位用户表示赞同，表明他们也在等待修复。
- **Microsoft Learn 认证与 Notebook LM 集成**：一位用户询问如何将 **Notebook LM** 与 **Microsoft Learn** 结合使用，并寻求 **Microsoft Certification** 的使用案例和技巧。
   - 这引发了其他成员对为何发生这种情况以及是否有人正在积极将 **Notebook LM** 与 **Microsoft Learn** 结合使用的疑问。
- **Palm Bayer 发布 AI 驱动的 Publici**：一位用户为一个城市和一个县创建了两个 notebooks，并在 [The Palm Bayer](https://www.thepalmbayer.com/p/palm-bayer-unveils-ai-powered-publici) 的文章中进行了介绍，展示了 **AI** 在公共信息方面的应用。
   - 该用户表达了对 **AI** 的热爱。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1379200282441023518)** (40 条消息🔥): 

> `Notebook LM 事实幻觉，Notebook LM 仅读取部分源文件，Gemini 2.5 pro 即将登陆 Notebook LM？，同步 Google Docs 编辑内容，使用 discover 功能生成的音频概览（audio overview）体验流畅` 


- **Notebook LM 凭空生成事实**：一位用户报告称 **Notebook LM** 会生成随机的、无来源的事实，并错误地将其链接到源文档，这需要不断修正以防止 AI 将对话历史作为来源。
   - 幻觉事实包括一个与 "**zom (琥珀金)**" 相关联的 "**Aha-Teta 乌木板**"，由于在提供的资料中未发现此关联，该内容被认为是不准确的。
- **Notebook LM 的阅读理解能力受到质疑**：Reddit 上的一位用户声称 **NotebookLM** 仅读取给定源文件的一小部分，并举例说明它在 146 页中仅读取了 **21 页**。
   - 然而，有人指出该用户可能误解了 **NotebookLM** 的工作原理，特别是其对 **RAG (Retrieval-Augmented Generation)** 的使用，并且该用户试图在单一源文件中处理不相关的内容，这需要自定义解决方案。
- **NotebookLM 准备升级到 Gemini 2.5 Pro？**：成员们正在推测 **Notebook LM** 何时开始使用更先进的模型（如 **Gemini 2.5 Pro**），但目前 Google 尚未公布明确的时间表。
   - 目前使用的模型可能是 **Gemini 2.5 Flash**，尽管这一点尚未得到证实。
- **Google Doc 编辑内容未与 NotebookLM 自动同步**：用户询问对 **Google Doc** 源文件的编辑是否会自动反映在 **NotebookLM** 中。
   - 一位成员澄清说，更改**不是自动的**；用户必须**从预览界面重新同步**，无需重新上传文档，即可反映编辑内容。
- **使用 Discover 功能生成的音频概览（Audio Overview）体验流畅**：一位用户表示，配合 **discover 功能** 使用 **audio overview** 的长选项非常流畅，且生成速度极快。
   - 他们报告称生成了一段 **65 分钟** 的音频，且音质很高。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1379184239630614599)** (33 条消息🔥): 

> `Claude Code 分析，Escape Mount Moon 黑客松，保障推理时数据安全，Modal 的 LLM 工程师年鉴，Anthropic 削减 Claude 3.x 算力配额` 


- **Southbridge 分析 Agentic Claude Code**：Southbridge Research 发布了一份关于 **Claude Code** 及其 Agent 能力的 [Notion 分析报告](https://southbridge-research.notion.site/claude-code-an-agentic-cleanroom-analysis)。
   - 作者提到，该分析促使他们上线了 **Writer** 和 **Hashbrown**。
- **Modal 发布 LLM 推理基准测试**：Modal Labs 发布了 [LLM 工程师年鉴 (LLM Engineer's Almanac)](https://x.com/charles_irl/status/1929615080494416213)，包含了在 **vLLM**、**SGLang** 和 **TensorRT-LLM** 框架下的数千项 LLM 推理服务基准测试。
   - 该年鉴包括测试结果、复现代码以及一份针对技术领导者核心问题的执行摘要，同时还发布了他们的基准测试框架 **stopwatch**。
- **Anthropic 削减配额引发混乱**：Varun Mohan 报告称，Anthropic 在提前不到五天通知的情况下，出人意料地削减了几乎所有 **Claude 3.x** 模型的算力配额，导致了可用性问题。
   - 用户对模型提供商的信任表示失望和担忧，但 **Gemini 2.5 Pro** 和 **GPT 4.1** 等其他模型未受影响。
- **Altman 宣布 Codex 支持联网功能**：Sam Altman 宣布 AI 编程工具 **Codex** 现在为 **ChatGPT Plus** 用户提供可选的联网功能，由于风险复杂，该功能默认关闭。
   - 社区要求澄清 **Codex** 的定义、权衡因素以及潜在的安全疑虑。
- **Textract 准确率不足，法律从业者需警惕**：一位成员警告称 **Textract** 在法律和监管文件上的准确率较低（约 3%），并链接到一篇详细说明该问题的 [LinkedIn 帖子](https://www.linkedin.com/posts/robertreich_when-word-for-word-accuracy-is-key-in-etl-activity-7265008546793086978-hfaj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAABOb18Bac53omUsFRAIBEVDUe013Eez5zo)。
   - 他们建议在涉及需要逐字准确度的 ETL 流程时，应谨慎使用包含 **Textract** 的流水线。


  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1379257023778258964)** (13 messages🔥): 

> `Live AI Bot Collaboration, New UI Framework, Bug Reporting System, AIE Website Integration, Feedback Loop Improvement` 


- **实时 AI Bot 协作部署**：一个实时生产环境的 **AI bot** 已通过新的 UI 框架协作开发并部署到 [AIE website](https://ai.engineer/ai)。
   - 一位成员表示，这次协作简直是*梦想成真*。
- **UI 框架驱动新 AI Bot**：分享了新一代 **UI framework**，随后用于将 **AI bot** 发布到 [AIE website](https://ai.engineer/ai)。
- **呼吁建立 Bug 报告系统**：由于收到 Bug 报告，目前正在讨论建立一种比直接消息更简单的 Bug 报告方式。
   - 目标是改进 **AI bot** 的反馈循环，使其能够自我提升。
- **热烈欢迎并提供帮助**：社区欢迎 **Mike** 加入频道，并表示支持提供 Bug 报告。
   - 一位成员分享道，他们也*非常乐意协助处理 Bug 报告！*。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1379172983607656581)** (30 messages🔥): 

> `HF Tokenizer, FP8 + TP Ungated, Llama alternatives to LLaMA-Factory, Context Parallel limitations, dropping 3.9 Support` 


- **Tokenizer 已准备好进行 Review！**：已按 [PR #2781](https://github.com/pytorch/torchtune/pull/2781) 的要求添加了单元测试，正等待 Review。
- **FP8 + TP 解锁并降低显存占用！**：[PR #2782](https://github.com/pytorch/torchtune/pull/2782) 解锁了 **FP8 + TP**，启用了 **loss parallelism**，并实现了峰值活跃显存的*显著降低*。
   - 该 PR 还启用了 autograd 编译，但目前处于损坏状态。
- **Torchtune：LLaMA-Factory 的可行替代方案**：一位用户在工作中使用 torchtune 分支，发现它是 **LLaMA-Factory** 的高性能且易读的替代方案，因为它避免了对 **TE, megatron, lightning** 的依赖。
   - 一个团队已经从 torchtune 分支拉取代码并训练了 **4-5 个月**，发现它*非常稳定*且*结果良好*。
- **Context Parallel 缺失 Flex Attention 兼容性**：虽然 **Context Parallel (CP)** 已经落地，但目前缺失的一个关键点是 flex attention 的兼容性，因为 packing 能带来巨大的收益。
   - 分布式团队正在努力尽快实现 flex attention 的兼容性。
- **停止支持 Python 3.9？**：Python 3.9 的生命周期结束状态导致了 linting 问题，因为新的 linting 坚持使用 **List -> list, Tuple -> tuple** 等，而 **CI** 则要求使用来自 typing 的 **Union** 和 **Optional**。
   - 一位用户暗示 CI 失败是因为 *Joe*。


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1379211955646234766)** (1 messages): 

> `Gradio Agents x MCP Hackathon, Livestream on June 3rd, Hackathon Office Hours on June 4th` 


- **Gradio Agents x MCP 黑客松来了！**：本周是 **Gradio Agents x MCP Hackathon** 周，你仍然可以在[这里](https://huggingface.co/Agents-MCP-Hackathon)注册。
- **Gradio Agents x MCP 直播即将开始！**：明天 **6 月 3 日**，请在 YouTube [这里](https://discord.com/events/1059199217496772688/1379207318700294245)观看 **Gradio Agents x MCP 直播**！
- **黑客松答疑时间 (Office Hours)！**：**6 月 4 日星期三**，将在 HuggingFace Discord 服务器为黑客松参与者举办答疑环节。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1379219718287917107)** (5 messages): 

> `Gradio Agents & MCP Hackathon, Scaling Agents in Finance workshop, MCP Integration, Agentic AI projects` 


- **Gradio Agents & MCP Hackathon 正式启动**：**Gradio Agents & MCP Hackathon** 现已开幕，包含[直播](https://t.co/FzLmzviwRz)、**1.65 万美元奖金**以及 **90 万美元积分**。
   - 本次黑客松包含 **3 个赛道**：MCP 工具/服务器、Agent 自定义组件、Agentic Demo 展示。
- **Scaling Agents in Finance 研讨会**：**Scaling Agents in Finance 研讨会**的幻灯片现已发布，展示了如何利用 **Agentic AI** 自动化处理金融任务中的文档工作流。
   - 该幻灯片教授了如何使用 **Assistant Agents** 来充当强大的**“研究助手”**。
- **LlamaIndex MCP 集成增强 Agent 能力**：LlamaIndex 集成通过 **MCP** 增强了 Agent 能力和工作流部署。
   - 此集成提供了**辅助函数**，方便 LlamaIndex Agent 使用 MCP 服务器工具，并能够将任何 LlamaIndex 工作流作为 MCP 提供服务。
- **LlamaIndex 参加 AI Engineer Summit**：LlamaIndex 参加了在旧金山举行的 **AI Engineer Summit**。
   - 与会者与 LlamaIndex 团队会面，共同探讨了 **Agentic AI 项目**。
- **构建多 Agent 金融报告聊天机器人**：分享了一个手把手的 Colab 教程，演示如何使用 LlamaIndex Agent 工作流从零开始构建一个**多 Agent 金融报告生成聊天机器人**。
   - 该聊天机器人示例涉及解析和索引 Adobe 的 **10-K 文件**，并使用 agentic RAG 来回答问题。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1379380131642216539)** (16 messages🔥): 

> `Open Source Models vs GPT-4o, LlamaIndex Report Generation, Llama Agents vs LlamaDeploy, Gradio MCP Hackathon, Property Graph Index` 


- **开源模型 vs GPT-4o 的硬件需求**：为了让 **Deepseek-R1**、**DeepSeek-v3** 或 **Qwen3-235-A22B** 等开源模型达到 **GPT-4o** 的性能，用户需要拥有数百 GB VRAM 的强大硬件，或者必须求助于深度量化以及 CPU/RAM 卸载（offloading）。
- **LlamaIndex 报告生成：云端 vs 本地**：一位用户询问了 LlamaIndex 报告生成的获取方式以及 Jupyter notebooks 的可用性，并引用了[这篇博客文章](https://www.llamaindex.ai/blog/building-blocks-of-llm-report-generation-beyond-basic-rag)。
   - 一名成员提供了 [LlamaExtract notebooks 的链接](https://github.com/run-llama/llama_cloud_services/tree/main/examples/extract)，但也澄清了 **LlamaReport** 和部分 Demo 需要 **LlamaCloud**，这意味着提取和报告生成是基于云端的。
- **Llama Agents 演变为 LlamaDeploy**：一位用户注意到 PyPI 上的 **llama-agents** 包自 2024 年 8 月 16 日以来未曾更新，并询问 **Llama Agents** 是否已被 **Workflow** 取代。
   - 一名成员确认 **LlamaAgents** 已更名为 **LlamaDeploy**，它可以将 N 个 Workflow 作为服务进行部署，[点击此处了解更多信息](https://github.com/run-llama/llama_deploy)。
- **Gradio MCP Hackathon HuggingFace 答疑时间**：成员们宣布将在 **HuggingFace Discord 服务器**为 **Gradio MCP hackathon** 的参与者举办一场答疑（office hours）活动。
   - 为感兴趣的人员分享了 [Discord 活动链接](https://discord.com/events/879548962464493619/1379561017536938095)。
- **Property Graph Index Token 使用情况问答**：一位 LlamaIndex 新用户询问了使用 **Property Graph Index** 进行索引和检索时的 Token 使用情况。
   - 他们还询问了其与 **GraphRAG**、**HippoRAG2** 和 **LightRAG** 的性能对比。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1379347144779239476)** (16 条消息🔥): 

> `多 MCP 交互, 新功能请求讨论, 用于尝试的常用 MCP 服务器, MonetizedMCP: 开源支付框架, API Keys vs MonetizedMCP` 


- **寻求多 MCP 交互指导**：一位成员询问如何创建一个可以与其他多个 MCP 交互的 MCP，例如调用 **Atlassian MCP** 获取工单，然后调用 **Git MCP** 创建分支。
   - 他们正在寻找专家来演示 **MCP server 与 MCP client 的设置**，并帮助设置公开可用的 MCP server。
- **新功能请求频道探索**：一位成员询问是否有讨论新功能请求的频道，特别是关于 [长时间运行的工具调用 (long-running tool invocations)](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/617) 的讨论。
   - 他们觉得在多个 Issue/Pull Request 中进行的讨论变得很混乱，希望能有一个更好的讨论场所。
- **尝试常用的 MCP 服务器**：一位成员请求一份常用的 **MCP server** 列表用于尝试，要求支持采样 (sampling)、OAuth、提示词 (prompts)、资源 (resources) 和资源模板 (resource templates) 等功能。
   - 另一位成员建议查看 [servers 目录](https://github.com/modelcontextprotocol/servers/tree/HEAD/src/everything)，认为那里看起来还不错。
- **MonetizedMCP 演示展示**：一位成员分享了 **MonetizedMCP**，这是一个开源框架，可以为任何 MCP server 添加程序化支付（加密货币或法币），并附带了 [短演示视频](https://www.loom.com/share/f94433182d7b4148ac7f59e987cb0fe6?sid=475ed243-d195-4f11-81a5-cf171eff2de0) 和 [网站](https://www.monetizedmcp.org/)。
   - 它可以与 **mcp-remote 库** 配合使用。
- **API Keys 与 MonetizedMCP 的对比审查**：一位成员质疑 **MonetizedMCP** 的必要性，认为在 **MCP** 中支持 **OAuth** 的 **API keys**（如 [MCP Connector](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector) 中所示）就可以实现货币化。
   - 另一位成员解释说，这为开发者提供了一条路径，使他们有可能在不需要构建 **API/用量监控/速率限制/Stripe 集成等** 的情况下提供付费产品。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1379463405189992479)** (4 条消息): 

> `可自托管的助手, 移动端 MCP, alpic-ai/grizzly, 视觉测试` 


- **Piper：移动端 MCP 的自托管助手**：一位成员分享了 [Piper](https://github.com/jmagar/piper)，这是一个可自托管的助手，因为目前在移动端使用 **MCP** 还没有很好的选择。
   - Piper 托管在 GitHub 上。
- **Grizzly 集成了酷炫功能**：另一位成员分享了他们在周末构建的 [Grizzly](https://github.com/alpic-ai/grizzly)，它集成了一些可以与 Piper 互补的酷炫功能。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1379175586206974163)** (13 条消息🔥): 

> `MOOC 日期, 作业截止日期, 提交反馈, 证书声明表, 文章撰写表` 


- **MOOC 下一期日期尚未确认**：一位成员询问了今年下一期 **MOOC** 课程的日期，但目前尚未有任何确认消息。
- **作业截止日期已过**：一位用户询问是否会重新开放待处理的测验，但得到的答复是所有作业的截止日期均为 **5 月 31 日**。
- **证书声明表截止日期临近**：一位成员询问关于填写 **证书声明表 (Certificate Declaration Form)** 和 **文章撰写 (Written Article)** 的事宜。
   - 官方确认需尽快完成，因为表格很快就会关闭。
- **请求所有提交内容的详细反馈**：一位成员请求获取其所有提交内容的详细反馈，包括 agentX 项目和 2 个实验作业 (lab assignments)。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1379392892900216882)** (3 条消息): 

> `CMD-R 后续模型, 微调适配器, Cohere 赞助` 


- **Hugging Face 缺少 CMD-R 后续模型**：Hugging Face 尚未发布后续的 **CMD-R** 模型。
   - 用户可以尝试将其与 **微调适配器**（例如 LoRA）配对，或者尝试使用具有更新训练数据的 **Mistral**。
- **AWS Bedrock 可能上线 Command A**：一位用户询问了 **Command A** 在 **AWS Bedrock** 上线的可能性。
   - 讨论中未给出明确确认。
- **寻求 Cohere 的赞助**：一位成员询问联系 **Cohere** 寻求 **高校黑客松** 赞助的正确渠道。
   - 聊天中未提供直接的联系方式。


  

---

### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1379391643500679220)** (2 messages): 

> `自我介绍，社区服务器` 


- **Cohere 社区服务器自我介绍开始**：一位名为 Aashutosh 的新成员加入并向社区服务器介绍自己。
   - 他是来自印度的大二本科生，对 **LLM** 和 **ML** 非常痴迷，并期待创建现实世界的项目。
- **置顶消息被强调**：置顶消息被突出显示，以鼓励新成员介绍自己。
   - 该消息提供了一个模板，包括 `公司/行业/大学`、`你正在研究的内容`、`你使用的最喜欢的技术/工具`，以及`你希望从这个社区获得什么`。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1379202659793375362)** (5 messages): 

> `在过时 CPU 上运行 Deepseek-r1，Orange PI 开源硬件，Mac 显存之王` 


- **过时 CPU 运行 Deepseek-R1**：一位用户通过移除 **RAM** 以腾出空间，在过时的 5000 mt/s **SSD** 上以 **0.1 tokens/second** 的速度运行了一个大型 **Deepseek-r1** 模型（超过 400GB）。
   - 他们将其归功于 **MOE** 模型惊人的质量，并指出 PC 行业的存储已经大幅提升，而 **RAM** 速度则不然。
- **Orange PI 运行大模型的梦想**：用户希望像 **Orange PI** 这样的开源硬件项目能在其板卡上附带 Gen 5 m.2 插槽。
   - 他们预计，通过 **OpenCL** 和现有的 3D **GPU**，有可能制造出一台非常强大的“统一内存”机器，在适当卸载的情况下，仅消耗几瓦功率就能以每秒若干 token 的速度运行像 **DeepSeek-R1** 这样大的模型。
- **Mac 为模型爱好者提供海量显存**：用户认为配备 **512GB** 的 **Mac** 是“显存”之王，拥有 **448GB** 的 **VRAM**，且价格与四台最新的 **AMD AI MAX 395+ 128 GB** 迷你 PC 或笔记本电脑的总和相当。
   - 他们还指出，与组合的 **AMD** 方案相比，**Mac** 的功耗更低。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1379254937846026301)** (1 messages): 

> `PR 评审请求，GitHub` 


- **在 GitHub 上请求 PR 评审**：一位成员请求对其在 **GitHub** 上的 [Pull Request](https://github.com/tinygrad/tinygrad/pull/10605) 进行评审。
- **需要 GitHub PR 评审**：一位 **GitHub** 用户请求对其 [Pull Request](https://github.com/tinygrad/tinygrad/pull/10605) 进行评审。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1379441302415868005)** (1 messages): 

> `GlobalCounters.mem_used, GlobalCounters.global_mem, Tensor Realization` 


- **GlobalCounters 揭秘：mem_used vs global_mem**：一位用户询问了在 **tinygrad** 框架中 `GlobalCounters.mem_used` 与 `GlobalCounters.global_mem` 的适当使用场景。
   - 注意到 `mem_used` 在缓冲区分配/释放期间更新，而 `global_mem` 在 `ExecItem` 中更新，似乎是在 [tensor realization](https://github.com/tinygrad/tinygrad) 期间。
- **深入探讨 tinygrad 的内存计数器**：`GlobalCounters.mem_used` 和 `GlobalCounters.global_mem` 之间的区别在于 **tinygrad** 中的分配时机和上下文。
   - `mem_used` 提供已分配/释放缓冲区的实时视图，而 `global_mem` 反映了 `ExecItem` 中 **tensor realization** 期间的内存使用情况。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1379529015429169352)** (2 messages): 

> `Machine Learning Street Talk，用于数据分析的 AI 编程，AI4Legislation 夏季竞赛` 


- **Machine Learning Street Talk 讨论生成式 AI**：本周五（6 号）太平洋标准时间上午 9 点，加入 **Machine Learning Street Talk (MLST)** 讨论生成式 **AI**；详情请见[此 Discord 活动链接](https://discord.com/events/SZnhkpde?event=1374411787964907540)。
- **用于数据分析的 AI 编程网络研讨会**：行业专家 **Liang Guo** 正在主持一场专注于数据分析 **AI** 编程的网络研讨会；请通过[此 Google Forms 链接](https://forms.gle/e71FSdpwBtDBccgKA)进行预约。
- **AI4Legislation 夏季竞赛公布**：硅谷华人协会正在举办 **AI4Legislation** 夏季竞赛，这是一系列活动的一部分；更多信息请访问 [GitHub](https://github.com/svcaf/2025-AI4Legislation-Public)。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1379361472269778965)** (1 条消息): 

> `Gorilla, MCP, tool use, API actions` 


- **Gorilla 是 OG MCP**：**Gorilla** 比正式的 **MCP** 标准早了一年，但它在功能上是一个原型 MCP 系统。
   - 它将模型查询路由到 tool use，解释结构化的 tool schemas，并将生成内容落地到真实的 **API actions** 中。
- **Gorilla 赋能接口**：团队认为 **Gorilla** 是 **MCP** 的 MVP。
   - 它证明了核心理念：**LLMs** 不仅仅需要知识 —— 它们还需要接口。