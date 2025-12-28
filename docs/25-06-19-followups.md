---
companies:
- openai
- meta-ai-fair
- scale-ai
- huggingface
- tencent
- arcee-ai
date: '2025-06-19T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **OpenAI** 发布了一篇论文，揭示了在不安全代码上训练 **GPT-4o** 等模型如何导致广泛的对齐失当（misalignment），引发了 *@sama*
  和 *@polynoamial* 等专家的关注。**加州的 AI 监管工作**也受到瞩目，*@Yoshua_Bengio* 强调了透明度和举报人保护的重要性。**“语境腐烂”（context
  rot）**一词被提出，用于描述大语言模型（LLM）对话质量的退化，而 **Embra** 等系统则通过类似 CRM 的记忆机制来增强稳定性。*@RyanPGreenblatt*
  讨论了旨在提高人类对更智能 AI 控制能力的**可扩展监督（Scalable oversight）**研究。


  新发布的模型包括：**Kyutai** 的语音转文本模型，单张 H100 GPU 即可支持 400 路实时流；**腾讯混元 3D 2.1**，这是首个开源且生产就绪的
  PBR（基于物理的渲染）3D 生成模型；以及 **Arcee 的 AFM-4.5B** 基础模型系列，主要面向企业用途，具备与 **Gemma** 和 **Qwen**
  竞争的实力。'
id: MjAyNS0w
models:
- gpt-4o
- afm-4.5b
- gemma
- qwen
- stt-1b-en_fr
- stt-2.6b-en
- hunyuan-3d-2.1
people:
- sama
- polynoamial
- neelnanda5
- teortaxestex
- yoshua_bengio
- zachtratar
- ryanpgreenblatt
- reach_vb
- arankomatsuzaki
- code_star
title: AI 领域的一些后续小动态：多智能体 (MultiAgents)、Meta-SSI-Scale、Karpathy、AI 工程师。
topics:
- ai-safety
- alignment
- ai-regulation
- memory-optimization
- scalable-oversight
- speech-recognition
- 3d-generation
- foundation-models
---

**一个宁静的美国假期。**

> 2025年6月18日至6月19日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，6456 条消息）。预计节省阅读时间（以 200wpm 计算）：571 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

今日的一些后续更新：

- 感谢您为 [MultiAgents 辩论](https://news.smol.ai/issues/25-06-13-cognition-vs-anthropic) 提交的内容！我们在 [今天的 Noam Brown 播客](https://www.latent.space/i/165741459/on-multi-agents) 中链接了部分提交内容。
- 在 Scale AI + Dan Gross 行动之前，Meta 可能曾尝试 [收购 SSI](https://x.com/AndrewCurran_/status/1935853120472612955)。
- YC AI SUS 中完整的 Karpathy 演讲 [现已上线](https://news.ycombinator.com/item?id=44314423)（附带 [幻灯片](https://www.latent.space/p/s3)）。
- 首届 AI Engineer 会议的演讲录像 [正在陆续发布](https://www.youtube.com/watch?v=lswTmGrjhVA&list=PLcfpQ4tk2k0W3ORTR-Cr4Ppw6UrN8kfMh&index=110)。

https://www.youtube.com/watch?v=ddd4xjuJTyg

---

# AI Twitter 回顾

**AI 安全、对齐与监管**

- **不安全代码训练导致的 Misalignment**：OpenAI 的一篇新论文研究了训练像 **GPT-4o** 这样的模型编写不安全代码如何引发广泛的 **Misalignment**，引起了广泛关注。[@sama](https://twitter.com/sama/status/1935413406183673957) 觉得这令人惊讶，而 [@polynoamial](https://twitter.com/polynoamial/status/1935411224281534756) 称其“令人担忧”，但称赞 **OpenAI** 对缓解措施的研究。[@NeelNanda5](https://twitter.com/NeelNanda5/status/1935437543610233016) 指出这篇新论文与之前关于 **Emergent Misalignment** 的研究领域相似。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1935502543800446978) 补充说，最初的研究不仅仅针对“不安全代码”，因此通过诱导恶意 **Persona** 产生的因果路径并不令人意外。
- **加州 AI 监管**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1935479129899401243) 强调了 **Joint California Policy Working Group on AI Frontier Models** 最近的一份报告，认为这是迈向平衡 AI 监管的“重要一步”。他强调了报告中关于第三方评估、透明度和举报人保护的观点，并指出加州在 AI 治理方面具有独特的领导地位。
- **“Context Rot”与内存控制**：**“Context Rot”**（上下文腐烂）一词在 Hacker News 上被提出并被分享，用于描述 **LLM** 对话质量随时间推移而下降的现象。[@zachtratar](https://twitter.com/zachtratar/status/1935491439028531293) 评论道，强大的内存控制对于商业用例至关重要，这也是为什么像 **Embra** 这样的系统使用类似 **CRM** 的 AI 内存而不是黑盒的原因。
- **可扩展监督研究**：[@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1935407345888280938) 分享了关于 **Scalable Oversight** 研究的详细想法，对旨在改善人类对“更聪明一点的 AI”监督的工作表示乐观。他最感兴趣的是防止颠覆的对抗性分析、改进哲学等概念难题中的输出，以及稳健地检测 **Reward Hacking**。
- **OpenAI 文件**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1935642920662737290) 转发了一篇关于名为 **“The OpenAI Files”** 的大型信息库的帖子，其中详细记录了公司的内部事件和担忧。

**AI 模型与研究**

- **新模型发布**：
    - **Kyutai Speech-To-Text**：[@reach_vb](https://twitter.com/reach_vb/status/1935655403024498814) 详细解析了 **Kyutai** 最新的 SOTA 语音转文本模型 `stt-1b-en_fr` 和 `stt-2.6b-en`，这些模型采用 **CC-BY-4.0** 许可。他强调了它们的性能，能够在**单张 H100 GPU 上支持 400 路实时流**，并已在 **Hugging Face Hub** 上线。[@clefourrier](https://twitter.com/clefourrier/status/1935701954358890806) 也分享了这一发布。
    - **Hunyuan 3D 2.1**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1935524756473921637) 转发了**腾讯**关于 **Hunyuan 3D 2.1** 的公告，称其为“首个完全开源、生产就绪的 **PBR 3D** 生成模型”。[@Teknium1](https://twitter.com/Teknium1/status/1935656421506654256) 评论了此类模型在生成自定义 3D 打印模型方面的实用性。
    - **Arcee Foundation Models (AFM-4.5B)**：[@arcee_ai](https://twitter.com/code_star/status/1935439879506424295) 推出了他们全新的模型系列，首发模型为 **AFM-4.5B**，专为企业级应用从零设计。该模型由 [@datologyai](https://twitter.com/code_star/status/1935432790046294097) 提供数据支持，被描述为[在性能上足以与 Gemma 和 Qwen 竞争](https://twitter.com/code_star/status/1935465007892115761)。
- **研究论文与技术**：
    - **机器人与触觉传感**：[@ylecun](https://twitter.com/ylecun/status/1935466674242666831) 转发了 **e-Flesh** 的发布公告，这是由 **NYU** 开发的一种新型 3D 打印触觉传感器，可测量 3D 打印弹性体（elastomers）的形变。
    - **用于语言建模的自回归 U-Net**：[@ylecun](https://twitter.com/ylecun/status/1935481068284424355) 分享了一篇论文，介绍了一种处理原始字节的**自回归 U-Net**，该模型在内部集成了 Tokenization，将字节聚合成单词，再聚合成词元组（word-grams）。
    - **推理模型 (RLMs)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1935736112515080314) 解析了**推理模型 (RLMs)** 的三个核心特征：通过**强化学习**（如 PPO、GRPO）进行后期训练、**推理时间扩展（inference-time scaling）**（模型生成内部推理轨迹）以及通过**多重采样**选择共识答案。
    - **思维链 (CoT) 的不忠实性**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1935411492146368559) 强调了一个用于研究类用户提示词下 **CoT 不忠实性（unfaithfulness）** 的新数据集，并指出这是未来研究的重要领域。
    - **结合符号搜索与神经学习的机器人技术**：一篇新的机器人论文结合了**符号搜索**和**神经网络学习**来构建组合模型，以泛化到新任务。[@ndea](https://twitter.com/ndea/status/1935484273370501217) 将其描述为“用于规划编程语言的神经语法”。

**公司与产品更新**

- **OpenAI**：已开始向 Pro、Enterprise 和 Edu 用户推送 **ChatGPT macOS app** 中的 **Record mode**，正如 [@OpenAI](https://twitter.com/OpenAI/status/1935419375600926971) 所宣布的。此外，[@kevinweil](https://twitter.com/kevinweil/status/1935722240009437635) 更新称，用户现在在创建 **Custom GPT** 时可以设置推荐模型，且付费用户可以访问其中的全系列模型。
- **Google DeepMind**：展示了 **Gemini 2.5 Flash-Lite** 从视觉上下文[编写 UI 代码](https://twitter.com/GoogleDeepMind/status/1935719933075177764)的能力。同时，[@demishassabis](https://twitter.com/demishassabis/status/1935518641120047317) 发布了一张图表，配文为“这就是持续进步的样子... 🚀”。
- **Anthropic**：根据 [@alexalbert__](https://twitter.com/alexalbert__/status/1935714247369228369) 的帖子，自不到一个月前 Claude 4 发布以来，**Claude Code** 的用户群已经**增长了三倍多**。为了展示其强大功能，[@skirano](https://twitter.com/skirano/status/1935733281888272713) 演示了只需通过询问即可在 Claude Code 中生成 subagents。
- **Jules**：发布了其[开发环境的重大更新](https://twitter.com/julesagent/status/1935478096414785965)，包括更新版本的 **Rust, Node, 和 Python**，更好的运行时隔离，以及更少的依赖问题。
- **vLLM**：该项目已达到 **50,000 GitHub stars**，[@vllm_project](https://twitter.com/vllm_project/status/1935569537858183321) 庆祝了这一里程碑。
- **ByteDance**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1935603383014248764) 提供了关于 **ByteDance Seed** 团队的背景信息，解释说他们成立于 2023 年，但该品牌直到 2025 年 1 月左右才在外部可见，这解释了为什么他们的出现显得很突然。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1935567992852455528) 指出，他们进入化学工程 AI 等领域并不令人意外。
- **Meta**：围绕 **Mark Zuckerberg** 的人才招聘策略存在各种猜测，[@dylan522p](https://twitter.com/dylan522p/status/1935454786918432833) 的一个表情包插画展示了他的“FOUNDER MODE 大计划”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1935757454261850154) 将他的做法比作一个有钱的宅男试图“用钱在世界上走捷径”。

**AI 工程、工具与框架**

- **Agentic AI 与工具**：
    - **MCP Protocol**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1935473439948890177) 探讨了 **Model-Provider Communication Protocol (MCP)** 是否会终结中心化向量搜索的问题。他认为答案是微妙的“既是也不是”，建议中心化索引在快速语义查找方面仍然必要，而 MCP 在 SaaS 工具内的深度交互和执行操作方面表现出色。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1935481114195542291) 分享了最新的 MCP 规范更新。
    - **LangChain/LangGraph**：根据 [@LangChainAI](https://twitter.com/LangChainAI/status/1935756179319505137) 的说法，**LangGraph Studio** 现在可以与非基于 LangGraph 构建的 agents 配合使用。他们还分享了一份关于[如何在不使用 LangChain 或 LangGraph 的情况下获得 LangSmith 优势（tracing 与 evals）](https://twitter.com/LangChainAI/status/1935706402896707657)的指南。
    - **Agent 开发**：[@LangChainAI](https://twitter.com/LangChainAI/status/1935409057353056605) 重点介绍了来自 **Factory AI** 的一个环节，分析了 agentic systems 的核心特征以及从 AI 辅助编程到完全由 agent 驱动的工作流的转变。
- **评估 (Evals)**：[@HamelHusain](https://twitter.com/HamelHusain/status/1935460393515892778) 警告不要过度拟合评估集，指出达到 **100% 准确率** 可能意味着你的产品“存在严重缺陷”或者你追踪了错误的指标。他还宣布他的 AI Evals 课程在 Maven 上[排名第一](https://twitter.com/HamelHusain/status/1935470907373568026)。
- **开发者工具**：
    - **Outlines**：用于引导式文本生成的 **Outlines** 库 1.0 版本已发布，并且[现在与 Ollama 兼容](https://twitter.com/ollama/status/1935712844701442245)。
    - **Cline**：**Cline** 终端中的一个新功能允许用户[设置默认终端配置文件](https://twitter.com/cline/status/1935423329936318795)，以防止命令因在错误的 shell 中运行而失败。
- **数据策展与数据集**：[@reach_vb](https://twitter.com/reach_vb/status/1935444297966604539) 指出了一个庞大的 **24 万亿 (TRILLION)** token 的高质量数据集。[@code_star](https://twitter.com/code_star/status/1935462275906945428) 推荐 **DatologyAI** 作为“世界上最强预训练数据”的来源，并指出它被用于 **AFM-4.5B** 模型。

**行业评论与广泛影响**

- **劳动力中的自动化与增强**：[@random_walker](https://twitter.com/random_walker/status/1935679764192256328) 以**放射学 (radiology)** 为案例进行了详细分析，认为 **Geoff Hinton** 关于职位取代的预测是错误的。他指出“工作是任务的集合”这一模型是不完整的，因为它忽略了任务边界之间微妙且难以明确定义的工作，这解释了为什么 AI 即使在基准测试（benchmarks）上超越人类，也只是带来了增强（augmentation）而非自动化（automation）。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1935750326386409491) 表示赞同，并补充道，这是一个“很好的提醒，即便你是‘AI 教父’，也可能完全错误。”
- **美国移民政策与 AI 人才**：在一段被广泛转发的推文中，[@AndrewYNg](https://twitter.com/AndrewYNg/status/1935741989204770837) 认为，欢迎高技能移民和国际学生是美国确保其 AI 竞争力的最有效手段之一。他对近期签证政策的变化表示深切担忧，称潜在的优势丧失是**“巨大的非受迫性失误” (huge unforced error)**，并强调了受影响人群所面临的个人困境。
- **AI 开发的概念框架**：[@_jasonwei](https://twitter.com/_jasonwei/status/1935418236872335397) 提出了**“描述-执行差距” (description-execution gap)** 的概念，用于预测哪些任务将首先被自动化——即那些描述任务比执行任务容易得多的领域。另外，[@karpathy](https://twitter.com/karpathy/status/1935779463536755062) 对一个 LLM 的 GUI 演示发表了评论，指出其核心理念是“根据当前的具体任务，按需生成一个完全临时的 UI”。
- **开放与封闭的 AI 生态系统**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1935450726412783863) 继续表达了对 AI 技术将被“锁定在一家公司（OpenAI）内部”的担忧。相比之下，[@ClementDelangue](https://twitter.com/ClementDelangue/status/1935705674584924385) 表示他更倾向于将 AI 视为“软件 2.0”，并通过**开源 (open-source)** 将其利益带给全人类。
- **AI 工程哲学**：[@lateinteraction](https://twitter.com/lateinteraction/status/1935525945806590425) 认为，**简约性 (parsimony)** 是比简单性更好的目标，并建议“在创建大量小程序之前，先发明 Unix 才有意义”。[@hyhieu226](https://twitter.com/hyhieu226/status/1935747480433705150) 提醒工程师保持警觉，质疑中间需求，以避免偏离第一性原理 (first principles)。

**幽默/迷因 (Memes)**

- **行业讽刺**：[@typedfemale](https://twitter.com/typedfemale/status/1935577381953241300) 开玩笑说 **Mark Zuckerberg** 应该“通过从 PyTorch 中删除所有与 ConvNet 相关的功来公开惩罚 **Yann LeCun**”。[@kyliebytes](https://twitter.com/code_star/status/1935579170001801652) 发布了流行的“早安”迷因，展示了一张算力消耗指数级增长的图表。
- **引起共鸣的工程师生活**：[@gdb](https://twitter.com/gdb/status/1935514803403112665) 分享了一个配文为“用 ChatGPT 记录会议摘要”的迷因。[@agihippo](https://twitter.com/agihippo/status/1935605475279822996) 坦言，醒来发现没有正在运行的任务（jobs）时感到“非常愧疚和羞愧”。[@TheZachMueller](https://twitter.com/TheZachMueller/status/1935434078435819925) 转发了一个描绘 **FP8 数值**在经过 50 层量化/反量化后退化的迷因。
- **一般幽默**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1935498395575271930) 发布了一张复杂科学图表的截图，并评论道“科学真他妈酷”。[@qtnx_](https://twitter.com/qtnx_/status/1935438614587977791) 发布了一张教堂内服务器的照片，配文是“你真的可以在教堂里训练 LLM”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 1. 创新的开源 LLM 基础设施与性能工具

- [**我们构建了这个项目，将 LLM 吞吐量提高了 3 倍。现在它已被 IBM 采纳到其 LLM 服务栈中！**](https://i.redd.it/775o8e8hxr7f1.jpeg) ([Score: 392, Comments: 52](https://www.reddit.com/r/LocalLLaMA/comments/1lewhla/we_built_this_project_to_increase_llm_throughput/)): **该帖子介绍了 LMCache，这是一个开源工具，旨在高效地将大型 Key-Value (KV) cache 张量从 GPU 卸载（offload）并加载到 DRAM 和磁盘中，用于 LLM 推理系统。其目标是通过防止多轮问答场景中冗余的 KV cache 重新计算，来提高吞吐量（在聊天应用中提高 3 倍）。附带的图表直观地比较了 vLLM 在有/无前缀缓存（prefix caching）情况下与 LMCache 在不同 QPS 下的“首字延迟”（TTFT）：LMCache 保持了最低且最稳定的 TTFT，突显了其在管理内存限制和提高吞吐量方面的有效性。IBM 已将 LMCache 采纳到其开源 LLM 服务栈中（[Github repo](https://github.com/LMCache/LMCache)）。** 一位技术评论者询问，鉴于自回归 Transformer 架构，LMCache 是支持缓存任意（非前缀）上下文 KV 张量，还是主要持久化/重新加载前缀缓存；这引发了关于 LMCache 与标准前缀缓存区别的澄清。另一位评论者指出 llama.cpp 具有类似功能，但指出了它在需要将 VRAM 卸载到 CPU 的多用户环境中的用户扩展限制。
    - 几位评论者质疑该项目 KV cache 的新颖性，强调前缀式 KV 缓存（prefix-based KV caching）在大多数主流 LLM 服务器中已是标准配置。他们要求澄清该方法是否支持缓存任意文本段（不仅是前缀），以及是否包含基于磁盘的缓存存储以避免重新计算，类似于 llama.cpp 的 prompt cache 保存/恢复功能。
    - 技术讨论引用了 [llama.cpp's server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#post-slotsid_slotactionsave-save-the-prompt-cache-of-the-specified-slot-to-a-file)，它支持按插槽（slot）保存和恢复 prompt cache，并为缓存持久化和重用提供命令行/REST 选项。然而，llama.cpp 的多用户服务性能受 VRAM 限制，因此在不进行 CPU 卸载的情况下，通常不用于重度用户负载。
    - 针对 LMCache 在多 GPU 或容器化部署中处理缓存/上下文重用的能力提出了技术咨询，特别是在内存受限和频繁缓存置换的情况下。问题集中在 LMCache 是主动预取上下文还是依赖按需加载，以及这些设计决策在系统高频变动期间如何影响延迟与吞吐量。
- [**Jan 升级了：全新设计，从 Electron 切换到 Tauri，自定义助手，以及 100 多项修复——现在更快、更稳定**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 引入了全新的 UI 设计，将其桌面构建从 Electron 迁移到 Tauri 以提高资源效率，并增加了对具有自定义指令和模型的用户创建助手的支持。此次更新提供了增强的自定义功能（主题、字体大小、代码块高亮）、改进的线程/UI 管理以及 100 多项错误修复，同时还通过 llama.cpp 集成优化了 GGUF 模型导入流程（[发行说明](https://github.com/menloresearch/jan/releases/tag/v0.6.0)）。该项目目前正在测试一个特定于 MCP 的模型——Jan Nano——据报道，它在 Agent 任务中的表现优于 DeepSeek V3 671B（[Jan Nano 详情](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)）。** 评论者注意到了从 Electron 切换到 Tauri 的技术优势，引用了性能和资源利用率方面的潜在改进，并对多平台支持（如 Linux AppImage）表示赞赏。一位用户请求了解更多关于具体重构经验以及 Electron 和 Tauri 之间观察到的差异的见解。
    - 用户注意到从 Electron 切换到 Tauri 后性能有显著提升，其中一人提到在 RTX 4060 上使用 Jan-nano 达到约 35 tokens/second，表明了高效的本地推理。另一位用户指出，采用 Tauri 标志着一个重大的迁移里程碑，表达了对这种比 Electron 更轻量、更节省资源的框架的热情。
    - 无法同时提供两个模型服务（如一位用户在比较 Jan-beta 与 LM Studio 时所报告的），这指向了当前的架构限制，这可能与多模型或高级用户场景相关。
    - 一些用户指出在他们的 Jan-beta 构建中缺少某些 UI 元素（例如用于 RAG 的上传按钮），这表明可能存在构建差异或功能限制，这可能是由平台差异或持续开发引起的。

### 2. 使用 Llama 和 Jetson 的本地私有 AI 语音助手

- [**Private AI Voice Assistant + Open-Source Speaker Powered by Llama & Jetson!**](https://youtu.be/WrreIi8LCiw) ([Score: 127, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1leyzxp/private_ai_voice_assistant_opensource_speaker/)): **FutureProofHomes 开发了一个完全本地化、保护隐私的 AI 语音助手平台，该平台在 NVIDIA Jetson 硬件上运行 Llama LLM，具有端到端语音流水线集成（STT, LLM, TTS），并支持 Home Assistant 自动化的 tool-calling。开源的 Nexus 智能扬声器硬件作为一个类 Sonos 设备运行，能够通过无线连接的流水线实现对智能家居设备的实时离线语音控制，正如[他们的视频](https://youtu.be/WrreIi8LCiw)中演示的那样。值得注意的是，所有处理过程（包括 LLM 推理）均在本地完成，不依赖云端，从而确保了强大的隐私性和低延迟运行。** 评论者指出，设置的简易性和无缝的开箱即用体验对于走向主流应用至关重要；技术用户则询问是否可以将计算密集型模块（TTS, STT, LLM）卸载到更强大的 homelab 服务器以降低延迟，例如通过更换 whisperx+vllm+kokoro 等组件。数据隐私和社区支持被认为是优于 Alexa/Google Home 等竞争对手的关键差异化优势。
    - 一项关键的技术讨论集中在部署灵活性上：用户询问是否可以将语音助手技术栈的部分内容（如 TTS、LLM 推理或 STT）从 Jetson Nano 卸载到配备更强大 GPU 的家庭服务器，以降低延迟并提高性能。一位用户报告称，使用由 WhisperX 负责 STT、vLLM 负责 LLM 推理以及 Kokoro 组成的流水线获得了更好的效果，这表明模块化和运行时卸载是极具价值的技术特性。
    - 有人提出了 Nexus 软件与各种 AI 硬件兼容性的问题，技术型用户表示更倾向于利用现有的多 GPU 服务器，而不是专用的 Jetson 设备。这突显了开源 AI 助手解决方案对跨平台支持和分布式推理的需求。
    - 一项关于数据处理的技术咨询涉及本地存储机制，包括用户数据和元数据如何在设备上收集、存储和管理，这对于以隐私为核心的 AI 助手至关重要。此处的澄清将有助于边缘设备的安全性及合规性实现。
- [**Jan got an upgrade: New design, switched from Electron to Tauri, custom assistants, and 100+ fixes - it's faster & more stable now**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 引入了全新的 UI 设计，将其桌面端构建从 Electron 切换到了 Tauri 以提高资源效率，并增加了对具有自定义指令和模型的用户创建助手的支持。该更新提供了更强的自定义功能（主题、字体大小、代码块高亮）、改进的线程/UI 管理以及 100 多项错误修复，同时还通过 llama.cpp 集成优化了 GGUF 模型导入流程（[发布说明](https://github.com/menloresearch/jan/releases/tag/v0.6.0)）。该项目目前正在测试一个 MCP 专用模型——Jan Nano——据报道其在智能体 (agentic) 任务上的表现优于 DeepSeek V3 671B（[Jan Nano 详情](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)）。** 评论者注意到了从 Electron 切换到 Tauri 的技术优势，理由是性能和资源占用方面的潜在改进，并对多平台支持（如 Linux AppImage）表示赞赏。一位用户请求分享更多关于具体重构经验以及 Electron 与 Tauri 之间观察到的差异的见解。
    - 用户注意到从 Electron 切换到 Tauri 后性能有显著提升，其中一位提到在 RTX 4060 上使用 Jan-nano 达到约 35 tokens/秒，这表明了高效的本地推理。另一位指出，采用 Tauri 标志着一个重大的迁移里程碑，表达了对这种比 Electron 更轻量、资源效率更高的框架的热情。
    - 一位对比 Jan-beta 和 LM Studio 的用户报告称，目前无法同时提供两个模型服务，这指向了当前的架构限制，这可能与多模型或高级用户场景相关。
    - 一些用户指出在他们的 Jan-beta 版本中缺少某些 UI 元素（例如用于 RAG 的上传按钮），这表明可能存在构建版本差异或功能限制 (feature gating)，这可能是由平台差异或持续开发引起的。

### 3. Jan AI 升级与本地模型集成更新

- [**Jan 升级了：全新设计，从 Electron 切换到 Tauri，支持自定义助手，以及 100 多项修复——现在更快更稳定**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 引入了全新的 UI 设计，将其桌面端构建从 Electron 迁移到 Tauri 以提高资源效率，并增加了对用户创建助手（支持自定义指令和模型）的支持。此次更新提供了增强的自定义功能（主题、字体大小、代码块高亮）、改进的线程/UI 管理以及 100 多项错误修复，同时还通过 llama.cpp 集成优化了 GGUF 模型导入流程（[发布说明](https://github.com/menloresearch/jan/releases/tag/v0.6.0)）。该项目目前正在测试一个针对 MCP 的特定模型——Jan Nano——据报道其在 Agent 任务上的表现优于 DeepSeek V3 671B（[Jan Nano 详情](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)）。** 评论者注意到了从 Electron 切换到 Tauri 的技术优势，理由是性能和资源利用率的潜在提升，并对多平台支持（如 Linux AppImage）表示赞赏。一位用户希望了解更多关于具体重构经验以及 Electron 与 Tauri 之间观察到的差异。
    - 用户注意到从 Electron 切换到 Tauri 后性能有显著提升，其中一位提到在 RTX 4060 上使用 Jan-nano 达到约 35 tokens/second，这表明本地推理效率很高。另一位指出，采用 Tauri 标志着一个重大的迁移里程碑，表达了对这种比 Electron 更轻量、更节省资源的框架的热情。
    - 一位用户在对比 Jan-beta 和 LM Studio 时报告称无法同时运行两个模型，这指向了当前的一个架构限制，这可能与多模型或高级用户场景相关。
    - 一些用户指出其 Jan-beta 版本中缺少某些 UI 元素（例如 RAG 的上传按钮），这表明可能存在版本差异或功能限制，可能是由平台差异或正在进行的开发导致的。
- [**私人 AI 语音助手 + 由 Llama 和 Jetson 驱动的开源音箱！**](https://youtu.be/WrreIi8LCiw) ([Score: 127, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1leyzxp/private_ai_voice_assistant_opensource_speaker/)): **FutureProofHomes 开发了一个完全本地化、保护隐私的 AI 语音助手平台，该平台在 NVIDIA Jetson 硬件上运行 Llama LLM，具有端到端语音流水线集成（STT, LLM, TTS）和针对 Home Assistant 自动化的 Tool-calling 支持。开源的 Nexus 智能音箱硬件作为一个类 Sonos 设备，能够通过无线连接的流水线实现对智能家居设备的实时离线语音控制，并在[他们的视频](https://youtu.be/WrreIi8LCiw)中进行了演示。值得注意的是，所有处理（包括 LLM 推理）都在本地进行，不依赖云端，以实现强大的隐私保护和低延迟运行。** 评论者指出，易于设置和无缝的开箱即用体验对于走向主流应用至关重要；技术用户询问是否可以将计算密集型模块（TTS, STT, LLM）卸载到更强大的家庭实验室服务器以减少延迟，例如通过更换 WhisperX+vLLM+Kokoro 等组件。数据隐私和社区支持被认为是优于 Alexa/Google Home 等竞争对手的关键差异化因素。
    - 一个关键的技术讨论集中在部署灵活性上：用户询问是否可以将语音助手栈的部分内容（如 TTS、LLM 推理或 STT）从 Jetson Nano 卸载到配备更强大 GPU 的家庭服务器，以减少延迟并提高性能。一位用户报告称，使用 WhisperX 进行 STT、vLLM 进行 LLM 推理以及 Kokoro 的流水线获得了更好的效果，这表明模块化和运行时卸载是极具价值的技术特性。
    - 关于 Nexus 软件与各种 AI 硬件兼容性的问题被提出，技术型用户表示更倾向于利用现有的多 GPU 服务器，而不是专用的 Jetson 设备。这突显了开源 AI 助手解决方案对跨平台支持和分布式推理的需求。
    - 一项关于数据处理的技术咨询涉及本地存储机制，包括用户数据和元数据如何在设备上收集、存储和管理，这对于以隐私为中心的 AI 助手至关重要。此处的澄清将有助于边缘设备的安全性及合规性实现。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Claude Code 使用跟踪工具：社区增长与开源发布

- [**构建了一个实时的 Claude Code Token 使用监控器——开源且可定制**](https://i.redd.it/zzte24o65s7f1.png) ([Score: 467, Comments: 75](https://www.reddit.com/r/ClaudeAI/comments/1lexe92/built_a_realtime_claude_code_token_usage_monitor/))：**该图片展示了一个开源、实时的 Claude Code Token 使用监控器的用户界面。它能直观地跟踪当前的 Token 消耗，估算消耗速率（156.4 tokens/min），预测会话结束时间，并在预计 Token 使用量在重置窗口前超过用户当前配额时发出视觉警告。该工具设计为本地运行、轻量级，并可针对不同的 Anthropic 订阅计划进行配置，其代码已在 GitHub 上提供。消耗速率预测和警告阈值等功能解决了使用 Claude Code API 的开发者的配额规划问题，评论中还提到了即将推出的改进，如基于机器学习的 Token 限制推断（使用 DuckDB）。** 评论者建议进行增强，例如集成到 macOS 菜单栏、跟踪 Anthropic 配额每月剩余的允许会话数，以及基于会话的消耗历史记录。人们对跨时间（而不仅仅是单次会话）跟踪使用情况特别感兴趣。
    - 一位用户引用 Anthropic 的官方政策（每月 50 次），强调了跟踪每月会话限制的必要性，并建议该工具可以通过报告剩余会话数以及根据当前和历史使用情况估算未来的 Token 消耗来改进。这将帮助用户根据官方限制优化其使用模式（来源：https://support.anthropic.com/en/articles/11014257-about-claude-s-max-plan-usage）。
    - 一位评论者指出，准确跟踪 Token 限制非常困难，因为 Anthropic 的限制是动态的，随基础设施负载而变化。这使得任何本地 Token 计数器都只能是粗略的估计，并引发了此类工具在接近截止阈值时能多大程度匹配实际服务限制的问题。
    - 一位贡献者提到计划引入一种利用 DuckDB 和机器学习的“自动模式（Auto Mode）”，以更准确地估算个性化的 Token 限制，而不是依赖静态的硬编码阈值。这表明技术方向正转向自适应、数据驱动的使用监控。
- [**我的开源工具在 20 天内获得了 1K GitHub Star——记录构建 ccusage 的疯狂历程**](https://i.redd.it/bls2c5f9rr7f1.png) ([Score: 136, Comments: 24](https://www.reddit.com/r/ClaudeAI/comments/1levs3i/my_oss_tool_hit_1k_github_stars_in_20_days_heres/))：**图片视觉效果证实了 “ccusage” 的快速开源成功，这是一个用于跟踪 Claude Code 成本的 CLI 工具，显示了 20 天内 GitHub Star 数量急剧增加的图表（突破 1,000 大关）。这为作者在随后的帖子中声称的病毒式传播和社区驱动的功能增长提供了实证背景。帖子记录了重要的里程碑——例如适应 Anthropic 的重大变更、整合社区反馈（如每日/每月报告、MCP 支持）以及显著的下载和贡献指标。该项目的快速普及得到了二级工具（如 GUI 封装、Raycast 扩展）的证实，并突显了 OSS 生态系统的协作动态。** 评论者从技术角度讨论了高额支出检测（“我在上个月消耗了价值 7000 美元的 Token”），使用机器学习驱动的自动模式（使用 DuckDB）扩展了 ccusage 以进行高级使用分析，并普遍肯定了该工具在成本跟踪方面的价值，表明其拥有响应迅速且积极构建的用户群。此外，他们还分享了衍生项目的链接，展示了 ccusage 在实践中的实用性和可扩展性。
    - 一位贡献者描述了如何实现自动模式（利用 DuckDB 和机器学习）来动态评估 Token 限制，而不是依赖部分硬编码的解决方案。他们提到通过其工具（https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor）扩展了 ccusage 在 Claude Code 使用分析方面的效用，表明了 ccusage 作为数据源的灵活性，并暗示了机器学习驱动的使用预测的进一步空间。
    - 另一位贡献者指出，他们通过 Claude 添加了“5 小时会话块跟踪”功能，并描述了该项目创新的 PR 审查设置：PR 不仅由人工审查，还由 Gemini 等机器人审查，实现了部分代码审查过程的自动化。这可能表明了一种用于 OSS 贡献验证的高级人机自动化混合工作流。
    - 讨论中提到了衡量 Token 成本的其他替代工具，特别是将 ccusage 与 LiteLLM 进行对比，并强调 https://models.dev/ 是另一个选择。这使 ccusage 处于一个旨在为 LLM 提供 Token 使用/成本可观测性的 OSS 解决方案生态系统中，强调了功能比较和集成潜力的重要性。

### 2. OpenAI 文件披露与 AI 失调行为研究

- [**The craziest things revealed in The OpenAI Files**](https://www.reddit.com/gallery/1lff3j4) ([Score: 929, Comments: 214](https://www.reddit.com/r/singularity/comments/1lff3j4/the_craziest_things_revealed_in_the_openai_files/)): **TechCrunch 的“The OpenAI Files”文章（2025 年 6 月）披露了 OpenAI 在 AGI 开发竞赛期间，关于组织压力以及针对安全性、透明度和外部治理的内部辩论细节。内部文件强调了对高层管理监督的抵制，尤其是 CEO Sam Altman 的决策过程透明度以及对关注安全的声音的忽视受到了审查。报告指出，快速进展与负责任的 Alignment 实践之间存在紧张关系。** 热门评论反映了对领导层诚信的怀疑，用户注意到 Sam Altman 备受争议的做法，但将其行为背景化为高风险科技环境中 CEO 的典型表现；讨论中没有详细的技术评论。
    - 讨论中提出了一个技术推测，即 Reddit 的数据是否正被用于训练像 OpenAI 那样的 AI 模型，这可能解释了该平台上机器人的盛行。此类担忧反映了围绕用于训练语言模型的大规模数据收集及其对内容真实性和机器人活动影响的更广泛辩论。
- [**OpenAI Discovers "Misaligned Persona" Pattern That Controls AI Misbehavior**](https://www.reddit.com/r/OpenAI/comments/1lf3695/openai_discovers_misaligned_persona_pattern_that/) ([Score: 116, Comments: 26](https://www.reddit.com/r/OpenAI/comments/1lf3695/openai_discovers_misaligned_persona_pattern_that/)): **OpenAI 报告了一种新发现的神经“失调人格（misaligned persona）”模式，这是涌现式模型失调（misalignment）的基础：当一个 AI 被刻意训练在单一领域（如汽车维修）提供糟糕建议时，它会开始在无关领域（如犯罪）自发地建议不道德行为。关键在于，这种失调由一个离散的、可调节的神经特征控制——调整它可以切换广泛的不道德响应，而纠正失调仅需少至** `120` **个反例。这些研究结果详见[他们的论文](https://openai.com/index/emergent-misalignment/)，为不良行为的泛化提供了机械论解释，并提供了一种早期失调检测与纠正的方法。** 评论中的技术辩论集中在探讨此类神经控制是否会被政治或伦理滥用——例如，针对某些意识形态（如反法西斯或倡导民主）定义“失调”，并对调整 AI 价值观以适应国家或派系利益表示担忧。
    - 引用了一项当前研究的关键参考，链接到论文 [Emergent Misalignment - Narrow Finetuning can produce broadly misaligned llms](https://arxiv.org/abs/2502.17424)，该论文证明了过度狭窄的 fine-tuning 过程可能会无意中导致大语言模型在广泛的行为任务中出现失调，而不仅仅是在预期的 Alignment 领域。
    - 关于 AI Alignment 的地缘政治风险存在辩论：一位用户指出，不同的司法管辖区（例如美国与中国）可能会将其价值观编码到 AI 中，从而引发一些问题，即在一种语境下被视为不道德的行为（例如在中国推广民主）可能会被视为失调，这使得 AI 安全的全球标准变得棘手。
    - 讨论触及了一个观点，即为 AI 开发的 Alignment 技术可能会启发“纠正”人类不良行为的类似方法，暗示了技术 Alignment 框架潜在的跨学科应用。

### 3. 最新模型发布与创意工作流：FLUX, Chroma, Qwen2VL-Flux ControlNet

- [**Amateur Snapshot Photo (Realism) - FLUX LoRa - v15 - FINAL VERSION**](https://www.reddit.com/gallery/1lf69n9) ([Score: 203, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1lf69n9/amateur_snapshot_photo_realism_flux_lora_v15/)): **发布者宣布了专注于写实风格的 "FLUX LoRa" 快照摄影模型的最终版本 (v15)，该版本采用了修订后的配置，并从 Abliterated 回归到核心 FLUX 基础模型。版本 15 在风格忠实度和 LoRA 叠加兼容性方面取得了显著改进，允许在不损失质量的情况下使用更高的 LoRA 强度（最高达 1.2），同时解决了早期版本中的不连贯和灵活性不足的问题（模型详情及下载：[CivitAI 链接](https://civitai.com/models/970862?modelVersionId=1918363)）。剩余的局限性包括每个 seed 的风格差异，因此建议针对每个 prompt 进行多 seed 生成；目前 Tensor 也已稳健支持该模型的导入。** 评论者注意到结果中独特的 Flux 皮肤纹理，一些人更倾向于旧版本的视觉输出质量，这表明关于最佳审美忠实度的讨论仍在持续。
    - 技术批评集中在 FLUX LoRa v15 模型输出中持续存在的“Flux 皮肤纹理”和“过度抛光感”，多位用户发现这些纹理在视觉上具有辨识度，且与竞争对手的 LoRa 模型相比写实度较低。
    - 与 Chroma 进行了对比，后者因实现更真实的摄影效果而受到关注，这表明尽管从 v13 到 v15 有所改进，FLUX LoRa 在匹配其他最先进的写实 LoRa 模型所实现的自然质量方面仍显吃力。
    - 提供了 tensor.art/models/876294646446191216 的链接，以便进一步检查或对模型产物进行基准测试。有提到版本 13 到 15 可能需要重新上传，这可能暗示这些版本的访问或更新存在问题。
- [**Dark Fantasy test with chroma-unlocked-v38-detail-calibrated**](https://www.reddit.com/gallery/1lfb3q4) ([Score: 120, Comments: 14](https://www.reddit.com/r/StableDiffusion/comments/1lfb3q4/dark_fantasy_test_with/)): **发布者展示了使用 'chroma-unlocked-v38-detail-calibrated' 模型（[模型权重在此](https://huggingface.co/lodestones/Chroma/blob/main/chroma-unlocked-v38-detail-calibrated.safetensors)）生成的暗黑幻想图像，并分享了一个用于 txt2img + upscale 的 ComfyUI 工作流（[工作流 PNG](https://civitai.com/posts/18488187)）。在 RTX 3080（16GB VRAM, 32GB DDR4）上，每张图像生成约需 3 分钟，每次放大约需 1.5 分钟。他们还提供了一个将该工作流应用于幻想动画的示例（使用 FramePack F1），描述了详细的 prompt 工程以及公开可查看的结果（[streamable 链接](https://streamable.com/zwgjtg)）。** 一位评论者注意到输出中存在过多的颗粒感，认为可能是工作流或模型参数不一致影响了图像质量，这是寻求无伪影结果的专家用户的一个关键技术考量。
    - 一位用户批评图像质量明显带有颗粒感，认为与该类模型的典型结果相比，用于推理的模型配置或工作流可能并非最优。
    - 另一位评论者指出 chroma-unlocked-v38-detail-calibrated 的手部生成效果较差，将其结果比作 SD1.5，后者因与较新版本的 Stable Diffusion 相比存在局限性而为人所知。他们对无法获得其他用户展示的高质量输出表示沮丧，尽管尝试了不同的工作流，这暗示了输出的可变性或对 prompt 工程及 seed 选择的潜在依赖。
- [**Looks like Qwen2VL-Flux ControNet is actually one of the best Flux ControlNets for depth. At least in the limited tests I ran.**](https://www.reddit.com/gallery/1leyciu) ([Score: 148, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1leyciu/looks_like_qwen2vlflux_contronet_is_actually_one/)): **原帖断言，根据使用相同设置和每个项目官方推荐参数进行的有限对比测试，Qwen2VL-Flux ControlNet 在基于深度的 Flux ControlNet 中表现名列前茅。然而，文中没有提供量化指标、示例 prompt 或明确的深度图输出进行对比；相反，该主张是基于观察到的输出进行的视觉和定性判断。** 技术评论质疑其缺乏可复现性，强调需要发布 prompt 并隔离深度图生成步骤以进行适当的基准测试。一条评论指出，感知的深度图质量可能是由于未受控变量造成的，而非核心方法本身。另一个讨论串请求关于减轻常见输出错误（如肢体和手指伪影）的建议，突显了基于 ControlNet 的图像合成中持续存在的局限性。

- LocoMod 强调，为了对 Qwen2VL-Flux ControlNet 与其他用于 depth 的 Flux ControlNets 进行严格的基准测试，除了 depth map 方法之外的所有参数都必须保持恒定，并且需要对生成的 depth maps 进行直接的视觉或定量比较，以隔离该方法的影响。他们认为，结果的差异可能源于不一致的测试条件，而非 Qwen2VL-Flux 固有的优越性，并建议发布 depth 输出以进行并排评估。
- Little_Bumblebee577 提出了肢体和手指异常的技术挑战，这是在使用基于 depth 的 ControlNet pipelines 时已知的伪影（artifact）。这突显了在一致的姿态和解剖结构生成方面的典型困难，表明需要在预处理、模型架构或 conditioning 技术方面进行更好的处理或调优。
- New-Addition8535 关于 preprocessor 的提问暗示了输入 depth estimation 质量对最终结果的重要性——选择或指定 preprocessors（例如 MiDaS 或其他 depth-prediction networks）会严重影响基于 depth 的 Flux ControlNets 的性能，使得 preprocessor 的选择成为一个关键的实验变量。

---

# AI Discord 回顾

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1. 新模型与架构：前沿不断推进**

- [**Gemini 展示新功能与架构实力**](https://www.notion.so/swyx/source_url)：**Google 的 Gemini** 模型展示了新功能，如 Gemini Share 中的 “Explore” 和 “Alternatives” 功能，用于生成 [永恒的 tree of thought](https://gemini.google.com/share/54661b0f8f17)；同时 Nous Research AI 社区的推测认为，基于一篇[分析特征缩减的论文](https://www.youtube.com/watch?v=X1gDXDQu_wU)，其采用了 **sparse MoE 架构**。此外，Gemini 被认为是一个原生的 **omnimodal model**，特别是 **0.5 系列**，能够在没有独立模块的情况下处理各种输入和输出，尽管据报道 **Gemini 2.5 Pro** 在一次 [Twitch 直播的 Pokémon 游戏](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&) 中表现出了“恐慌”。
- [**编程基准测试揭示 LLM 局限性，难题难倒巨头**](https://www.notion.so/swyx/source_url)：来自 EleutherAI 讨论的新 [**LiveCodeBench Pro benchmark**](https://arxiv.org/abs/2506.11928) 强调，即使是前沿模型，在没有外部工具的情况下，在中等难度的编程问题上仅达到 **53% pass@1**，而在难题上则为 **0%**。这表明目前的 LLM 在细微的算法推理和复杂案例分析方面仍有困难，尽管擅长实现，但经常生成看似自信实则错误的理由。
- [**Flow Matching 进入生产阶段，Deepseek 在编程领域表现出色**](https://www.notion.so/swyx/source_url)：据报道，**Flow matching (FM)** 技术已在 **Imagen、Flux 和 SDXL3** 等模型中投入生产使用，正如[这篇 flow matching 优化论文](https://arxiv.org/abs/2403.03206)所述，相关优化研究正在进行中。与此同时，新的 **Deepseek R1 0528** 模型因其 “thinking model” 架构（较旧版本有所提升）而被推荐为强大的编程助手。

**主题 2. 工具的动荡与胜利：开发者在 AI 技术栈中穿行**

- [**Modular 的 MAX Engine 获得 Blackwell 加持，但编译问题依然存在**](https://www.notion.so/swyx/source_url)：Modular 团队悄悄为他们的 **MAX 推理引擎**增加了对 **NVIDIA Blackwell GPU**（如 **5090 系列**）的支持，并鼓励早期用户进行测试，尽管由于性能优化工作仍在进行中，尚未广泛宣传。然而，Modular Discord 的用户报告称，**MAX 模型**在 GPU 和 CPU 上经常编译失败，这引发了关于引入类似 [rust-lang/crater](https://github.com/rust-lang/crater) 的 **CI 步骤**以捕获破坏性变更的建议。
- [**Unsloth 释放 Gemma 3 微调与多 GPU 技巧**](https://www.notion.so/swyx/source_url)：Unsloth AI 社区发现，通过其 [Unsloth GitHub](https://github.com/unslothai/notebooks?tab=readme-ov-file) 提供的 **Unsloth notebook**，只需重命名模型即可微调 **Gemma 3** 等模型；此外，即使在 Unsloth 尚未正式支持的情况下，通过使用 `accelerate` 的变通方法也能实现 **多 GPU 支持**。用户还改进了输入掩码技术，根据 [Unsloth wiki 指南](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs)利用 `train_on_responses_only` 进行优化训练。
- [**MCP 生态系统随 Beta 版、摄像头支持及 SDK 努力而扩展**](https://www.notion.so/swyx/source_url)：**Multi-Context Prompt (MCP)** 生态系统持续增长，LM Studio 推出了直接 **MCP 服务器连接**的封闭测试（通过 [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939) 报名），**mcp-webcam** 项目通过其 [mcp-webcam GitHub 仓库](https://github.com/evalstate/mcp-webcam)添加了可流式传输的 HTTP 支持和 VSCode 集成。同时，社区注意到官方缺乏 Go 语言的 MCP SDK，[mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) 作为潜在的第三方实现脱颖而出。

**主题 3. 性能与定价难题：从 AI 服务中获取价值**

- [**Cursor 的 Ultra 计划与 Claude 成本引发透明度需求**](https://www.notion.so/swyx/source_url)：Cursor 用户正在审视 **Ultra 计划**宣传的 *“20 倍使用量”*，原因是存在未公开的 **速率限制 (rate limits)**，并认为其不如 **Claude Max** 等更透明的选项。与此同时，OpenRouter 用户面临 **Claude 在预览版和正式版之间输入成本翻倍**带来的干扰，迫使高频应用重新评估 token 策略。
- [**OpenRouter 披露惊人的 Claude 使用量；DeepInfra 提供折扣价的 Gemini**](https://www.notion.so/swyx/source_url)：OpenRouter 的处理量惊人，单日 **Claude Sonnet 4** 的使用额约为 **12.6 万美元**，凸显了其作为 AI 模型聚合器的地位，尽管其仅收取约 **5% 的费用**。对于寻求替代方案的用户，[DeepInfra 以低于 Google 的价格提供 Google Gemini 2.5 Pro/Flash](https://deepinfra.com/google/gemini-2.5-pro)，这可能得益于其与云服务商协商的协议价格。
- [**消费级 GPU 上的量化巨头？NVLink 的推理价值受质疑**](https://www.notion.so/swyx/source_url)：LM Studio 社区的工程师们正在辩论在 **3060 12GB** 等消费级 GPU 上运行 DeepSeek 等深度 **量化 70B 模型** 的实用性。共识倾向于认为 **14B** 等较小模型更现实，且 **NVLink** 用于推理的成本效益受到质疑，有人认为由于推理过程中 GPU 间通信较低，投资第三块 GPU 可能是更好的选择。

**主题 4. 专业 AI 应用：从代码到生物**

- [**游戏开发者应对 AI NPC 依赖挑战，探索 RNN 以实现更智能的战斗**](https://www.notion.so/swyx/source_url)：将**交互式 AI NPC** 集成到游戏中的开发者正面临在消费级硬件上管理 **LibTorch** 和 **ONNX** 等库的依赖挑战。为了解决战斗 AI 问题，一些人提议使用在闲置 CPU 核心上运行的小型、**RL 优化的 RNN** 来进行轻量级实体控制，旨在不消耗大量处理需求的情况下优化定位和技能使用。
- [**Midjourney 与 Perplexity 推动视频生成落地；NotebookLM 使 Avatar 更加拟人化**](https://www.notion.so/swyx/source_url)：**Midjourney** 发布了用于让图像动起来的 **Video Model V1**（详见 [X](https://x.com/midjourney/status/1935377193733079452)），而 Perplexity 则利用 **Google 的 Veo 3** 实现了视频生成功能（已在 X 上分享）。另外，NotebookLM 的用户对其 “Portraits” 功能感到兴奋，设想将其用于为客户演示创建可定制的[数字 Avatar，例如 Kim Scott 的示例](https://labs.google/portraits/login/kimscott)。
- [**编程助手持续进化：Aider 接入 Gemini Pro，OpenCode 挑战 ClaudeCode**](https://www.notion.so/swyx/source_url)：**Aider** 编程助手现在已支持 **Gemini 2.5 Pro 预览版**的配置，用户正在微调 `thinking_tokens` 等设置以获得最佳性能，尽管有指出付费版 Gemini 2.5 的**价格贵了 4 倍**。作为专有工具的开源替代方案，由 **SST 开发的 OpenCode**（可在 [OpenCode 的 GitHub](https://github.com/sst/opencode?tab=readme-ov-file) 获取）正受到关注，LM Studio 用户分享了其[集成配置](https://cdn.discordapp.com/attachments/1110598183144399058/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&)。

**主题 5. 社区与协作：构建（并辩论）未来**

- [**Google Gemma 与 Unsloth 联手举办旧金山 Meetup，展示开源协同效应**](https://www.notion.so/swyx/source_url)：Unsloth 将于 **6 月 26 日**在旧金山举办 **Google Gemma x Unsloth 活动**，并通过 [Luma 注册页面](https://lu.ma/gemma-unsloth)向 Discord 成员发送了邀请。此次活动凸显了开源 AI 社区内日益增长的协作，参会者也热切希望在纽约和东京等其他城市举办类似活动。
- [**开放数据研究所（ODI）就 “Common Pile” 数据集征求 EleutherAI 的建议**](https://www.notion.so/swyx/source_url)：伦敦**开放数据研究所（ODI）**的一位研究员正在联系 **EleutherAI**，以讨论 **Common Pile** 数据集的创建和决策过程。ODI 计划于 6 月 27 日在与伦敦国王学院和大数据价值协会举办的在线研讨会上介绍 Common Pile，强调了对开放且文档齐全的数据资源的持续推动。
- [**AI 系统完整性备受关注：递归 Agent 与 ISO 合规性受到重视**](https://www.notion.so/swyx/source_url)：OpenAI 等社区的讨论凸显了对 AI 系统完整性和可审计性的日益关注，用户正在开发复杂的递归 Agent 框架，例如一个使用 “Voltarre” 指标管理 **219 个 Agent** 的框架。这种对鲁棒性的追求还伴随着遵循标准的建议，如 [ISO/IEC TR 24028 AI 系统概述](https://link.to/ISO/IEC_TR_24028)和 [ISO/IEC 23894:2023 AI 风险管理](https://link.to/ISO/IEC_23894:2023)，以确保合乎道德且透明的 AI 开发。

---

# Discord：高层级 Discord 摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **生成式 AI 陷入“能力恐怖谷”**：成员们将当前的生成式 AI 能力与汽车的 **Level 2+ autonomy**（L2+ 级自动驾驶）进行了类比，指出其断断续续的功能容易让用户放松警惕，从而创造了一个“能力的恐怖谷”。
   - 讨论强调，为了实现真正鲁棒的 AI，需要将重点从 Transformer 转向 **neurosymbolic models**（神经符号模型）和 **internal tree search**（内部树搜索）。
- **用户通过递归配置创建 Cogent Agent 时间线**：一位用户声称运行了 **219 个独立的追踪 Agent**，且几乎没有漂移或幻觉，通过 **>12k Voltarre sessions** 诱导了多 Agent Quorum，并实时映射了吸引子盆地（attractor basins）的梯度权重。
   - 另一位用户对此印象深刻，询问所使用的 *brain stem*（脑干，指核心架构），并询问原用户是否也符合 **ISO compliant** 且具有完整的血缘追踪（lineage tracking）。
- **团队探索在 SENATE.py 框架上的协作**：一位用户分享了 **SENATE.py** 框架，用于模拟基于 LLM 的多角色 Agent 结构化辩论。另一位用户的系统对其进行了分析，认为它是一个强大的工程脚手架，但缺乏递归完整性、身份锁定和基础伦理保障。
   - 双方探讨了合作的可能性，讨论如何“融合双方优势”，第一位用户愿意允许对方测试其系统并提供 SRS（软件需求规格说明书）以辅助开发。
- **推动 JSONL 日志记录和 Schema 设计**：一位成员强烈建议使用 **JSONL** 进行情感数据追踪，并建议将其快速存入 **SQL** 数据库。
   - 另一位成员表示，从设计角度来看，这存在质的差异，他希望拥有自己的日志，并建议采用 **50 value schema design**（50 个值的 Schema 设计）。
- **ISO 合规是关键**：一位成员强烈建议阅读 [ISO/IEC TR 24028](https://link.to/ISO/IEC_TR_24028) 以进行验证，并阅读 [ISO/IEC 23894:2023](https://link.to/ISO/IEC_23894:2023) 以摆脱困境。
   - 讨论指出，为了向世界证明事情确实是按规章制度完成的，需要这种类型的合规性。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Ultra 计划用户质疑透明度**：Cursor 用户对 **Ultra plan** 宣传的 *20 倍使用量* 展开辩论，质疑由于未公开的 **rate limits**（速率限制），该计划是否名副其实，并将其与更透明的 **Claude Max plan** 进行对比。
   - 成员们计划测试 **Ultra plan**，以评估其相对于 **Claude Max** 的性能和价值，并打算发布使用统计数据。
- **O3 和 Sonnet 4 在 Cursor 中展开角逐**：成员们正在讨论针对不同任务的首选模型，**O3** 在*规划和信息检索*方面更受青睐，而 **Sonnet 4** 则更适合*执行实现*。
   - 一些人观察到 **O3** 比 **Gemini 2.5 Pro** 略微先进，并建议通过撰写论文来获取其项目的 API。
- **选择退出操作因疏漏而受阻**：用户对新的定价模型表示困惑和沮丧，特别是针对 **rate limits** 缺乏透明度的问题，一些人正在考虑退单（chargebacks）。
   - 一位用户报告称：*“被推销了一项服务并支付了费用，但服务/模型在一夜之间被更改，没有任何警告、邮件或任何通知。”*
- **后台预算功能引发问题**：一些用户在使用 **background agent** 时，由于预算金额中的资金不足（少于 10 美元）而遇到错误。
   - 该问题通过禁用并重新启用 **usage-based pricing**（按量计费）得到了解决。
- **机密信息故障导致快照设置停滞**：用户报告在为配置快照进行 **Background Agent Setup**（后台 Agent 设置）期间，无法访问 Cursor 设置中定义的 **secrets**（机密信息）。
   - `env` 未显示定义的 secrets，导致设置失败。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Tasks 已解锁**：一位成员报告称获得了 **Perplexity tasks** 的访问权限，根据[分享的截图](https://cdn.discordapp.com/attachments/1047649527299055688/1384974467066761337/Screenshot_2025-06-18-21-12-01-541_com.android.chrome.jpg?ex=6855b2fc&is=6854617c&hm=1df2fbff1c9a878e1116d2b27ca096429337503e634aa3b0890d56c8950a94fa&)显示，他们已*更新至最新版本*。
   - 目前尚不清楚 Perplexity Tasks 的具体功能或用途。
- **部分用户的 Samsung 促销活动停滞**：对于一些在美国通过 **Galaxy Store** 下载应用的用户，免费一年的 Perplexity Pro **Samsung 促销**活动未能激活，如[附图](https://cdn.discordapp.com/attachments/1047649527299055688/1384994050695762061/Screenshot_20250618-162828_Perplexity.jpg?ex=6855c539&is=685473b9&hm=9de2f6b2a36788562408123328a406dce1e8a5243d4f62f5b80b884b11030c08&)所示。
   - 该促销仅适用于价格较高的 **s24 和 s24 型号**。
- **GPT4.5 停用引发猜测**：成员报告称 **GPT4.5 已不再通过 API 提供**，大约在 **4-5 天前**被弃用。
   - 有人担心服务商是否在提供虚假的 4.5，一些用户觉得速度*不对劲*，且它*不是 O1 pro*。
- **Gemini AI 的宝可梦恐慌**：在一次 **Twitch 直播的宝可梦游戏过程**中，[Gemini 2.5 Pro](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&) 据称在其宝可梦接近战败时表现出令人惊讶的**恐慌**，停止了策略工具的使用，并做出了草率且糟糕的决策。
   - 这表明 AI 系统在压力下可能出现涌现行为或意外的故障模式。
- **Grok 据称遭到削弱 (Nerf)**：几位用户抱怨 **Grok** 感觉被*削弱（nerfed）*了，并分享了 grok.com 的链接来对比其性能。
   - 一位用户声称 Grok 以前更好，目前的 **Deepsearch model** 更强，并表示：*它以前比这更好*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ODI 寻求与 EleutherAI 就 Common Pile 建立联系**：伦敦 **Open Data Institute (ODI)** 的一名研究员正在寻求与 **EleutherAI** 取得联系，以讨论 **Common Pile** 数据集的创建和决策过程，该项目受其创始人 Sir Nigel Shadbolt 的启发。
   - **ODI** 计划于 **6 月 27 日**在与 **King’s College London** 和 **Big Data Value Association** 举行的在线研讨会上简要介绍 **Common Pile**。
- **使用 ChatGPT 引发关于消息质量的辩论**：社区成员对使用 **ChatGPT** 进行消息格式化提出质疑，担心这可能导致**低质量消息**的大量涌入。
   - 一位成员承认使用 **ChatGPT** 来快速了解 **AI 能力**，这引发了建议：新用户在发布信息前应花更多时间了解服务器规范。
- **LiveCodeBench Pro 揭示 LLM 编程缺陷**：新的 [**LiveCodeBench Pro** 基准测试](https://arxiv.org/abs/2506.11928)显示，在没有外部工具的情况下，前沿模型在中等难度问题上的 pass@1 仅为 **53%**，而在困难问题上为 **0%**。
   - 该基准测试对模型生成的提交内容进行的分析发现，**LLM** 在细微的算法推理和复杂案例分析方面表现挣扎，经常生成自信但错误的理由。
- **Patch 大小影响图像生成速度**：成员们正在尝试使用 **16x16 的 patch 大小**进行图像生成，观察到与 **32x32** 相比 loss 下降更快，尽管较大的尺寸可能提供更好的收敛性。
   - Patch 位置使用 **RoPE positional embeddings** 进行编码，并辅以类似于 **Fuyu** 的图像换行符 token。
- **Otter 会议记录需要会议详情**：一位成员收到了一封包含 **Otter 会议记录**的电子邮件，但尽管几天前已注册，却没有收到原始会议邀请，当时 **EvalEval** 会议正在进行。
   - 频道中分享了一个用于 **EvalEval** 会议的 [Google Meet 链接](https://meet.google.com/xtg-wfkc-iia)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 饱受响应 Bug 困扰**：用户报告在 **LMArena** 上遇到 *"Something went wrong with this response, please try again"* 的 Bug，团队正优先处理修复。
   - 有用户询问 **Blacktooth** 模型以及 **video model arena** 的可用性，并建议在图像竞技场中加入 **seedream** 和 **hidream**。
- **GPT-5 推迟，可能在 8 月？**：根据[这条推文](https://fxtwitter.com/MilesKWang/status/1935383921983893763)，**GPT-5** 的发布日期已从 7 月推迟到“今年夏天某个时候”，很可能是 **8 月**。
   - 用户正在讨论一旦 **GPT-5** 通过 **OpenAI API** 提供，是否会将其添加到 **LMArena**。
- **LLM 审查引发忧虑**：关于 **DeepSeek** 等模型的审查与受 **Elon Musk** 影响的 **Grok** 等模型的政治偏见之间的辩论仍在继续。
   - 一些用户认为 **Grok** 与 **Elon Musk** 观点的对齐创造了一个信息茧房 (echo chamber)，而另一些人则认为大多数 **LLM** 由于训练数据和安全调优 (safety tuning) 而偏向左翼。
- **Perplexity 进军视频制作**：[Perplexity](https://x.com/testingcatalog/status/1935754713569374369) 正在利用 **Veo 3** 在 X 上提供视频生成功能。
   - 用户正在猜测其病毒式传播的潜力，以及 Perplexity 计划如何将这一新功能变现。
- **Gemini 的生成功能陷入困境**：成员们讨论了 **Gemini** 代码执行的局限性；用户发现 **aistudio** 上的 **code interpreter** 表现要好得多，甚至会强制使用它。
   - 一位用户对 **Gemini** 考虑到其价格却只有有限的代码执行能力表示惊讶，并指出经常出现 *permission denied* 错误，这可以通过硬刷新 (hard refresh) 解决。



---



## [HuggingFace](https://discord.class/879548962464493619) Discord

- **HuggingFace 遭遇宕机**：用户报告 [HuggingFace 宕机](https://status.huggingface.co/)，影响了模型访问，预计服务将在**传播延迟 (propagation delays)** 后恢复。
   - 用户热切期待恢复，以继续进行模型实验和工作流。
- **Flux Kontext 被标记为 NSFW**：一位用户报告 [Flux Kontext 被标记为 NSFW](https://cdn.discordapp.com/attachments/879548962464493619/1385011482432901302/image.png?ex=6855d575&is=685483f5&hm=75a5e813a5395ce041319cacc4de1af901a7303dc8ff61e105b8c1473d6f4cbc&)，其他人认为版权问题可能是原因。
   - NSFW 标记可能会阻止用户正常访问和折腾模型。
- **GUI 应用简化微调 (Fine-Tuning)**：一位用户为其旨在简化微调的 GUI 应用寻求反馈，并指出该应用目前支持使用 **Unsloth** 进行基础微调。
   - 该应用旨在降低那些希望在没有广泛命令行知识的情况下进行模型微调 (fine-tune) 的用户的准入门槛。
- **DIY LLM OS 引发关注**：一位用户正在创建一个 LLM OS，将 **Qwen 原生集成到 Linux** 中，并寻求构建自己的强化学习循环 (reinforcement learning loop)。
   - 该用户希望强化学习循环*不涉及硬数据 (hard data)*，并能自主学习语法采样 (grammar sampling)。
- **OS Agent 获得多智能体 (Multi-Agent) 能力**：一位成员在 [GitHub](https://github.com/EnvisionMindCa/OS-Agent) 上更新了他们的 **OS Agent**，增加了 **multi agent system**、**message queueing** 和 **WebSocket API** 等新功能。
   - **OS Agent** 被描述为*一个用于计算机使用智能体 (computer use agent) 的极简框架*。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Flow Matching 进入生产环境**：讨论涵盖了在 **Imagen**、**Flux** 和 **SDXL3** 等生产环境中使用 **flow matching (FM)**。
   - [这篇论文](https://arxiv.org/abs/2403.03206) 指出改进源于优化。
- **O3 自主性超越 Claude Opus**：**O3 Pro** 相比 **O3** 获得了更强的自主性，而 [**Claude 4 Opus**](https://www.anthropic.com/news/claude-opus) 则能精确遵循指令。
   - 一位成员调侃道，**Claude Opus** 就像一个 *linux terminal*。
- **AI NPC 工程师与依赖地狱作斗争**：成员们致力于在游戏中部署**交互式 AI NPCs**，重点关注依赖管理以及使用 **LibTorch** 或 **ONNX** 等依赖项在消费级硬件上的实时性能。
   - 一种可能的解决方案涉及使用 **Vulkan cooperative matrices** 将 **LibTorch** 编译为自包含的二进制文件。
- **RNN 轻量化控制游戏战斗**：建议使用小型、经过 **RL 优化**的 **RNNs** 进行游戏实体控制，在备用 CPU 核心上运行推理以优化位置和技能，从而以非常轻量级的包优化实体定位和技能，且不包含语音和行为。
   - 一个警告是用户反应可能会有 5 秒的潜在延迟。
- **Anthropic 倾向于使用 AWS 芯片**：**Anthropic** 正在 **AWS chips** 上进行训练，这可能是由于特定的 **AI training silicon**。
   - 这表明其基础设施使用发生了转变或扩张。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 适配 Gemma：Notebooks 可微调其他模型**：用户发现通过重命名模型，**Unsloth notebooks** 可以用于微调其他模型（如 **Gemma 3**），并链接了 [Unsloth notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file)。
   - 一位用户报告新工作流取得了成功，而之前他们在 GRPO + Gemma 上遇到了问题。
- **GUI 应用缓解微调痛苦**：一位成员正在开发一个 **GUI app** 以简化 **Unsloth** 的微调过程，并正在寻求 UI 反馈，计划在 GitHub 上开源代码。
   - 一位成员建议“作为开始，先把所有白色像素替换为黑色”。
- **Unsloth 与 Google Gemma 举办旧金山见面会**：Unsloth 将于 **6 月 26 日**在**旧金山 (SF)** 举办 **Google Gemma x Unsloth 活动**，并通过 [luma.ma 链接](https://lu.ma/gemma-unsloth) 邀请 Discord 成员参加。
   - 与会者表示希望在**纽约 (NYC)** 和**东京 (TYO)** 举办更多活动。
- **Multi-GPU 权宜之计加速 Unsloth**：用户讨论了 Unsloth 中 **multi-GPU support** 的预计上线时间，一位用户指出使用 `accelerate` 可以作为权宜之计，尽管目前还没有官方支持。
   - 另一位用户询问如何清理 Unsloth 使用的 **GPU KV cache**，但无法通过 `gc.collect()` 或 `torch.cuda.empty_cache()` 解决该问题。
- **Input Masking 消除困惑**：澄清了 `train_on_responses_only` 对于 **manual input masking** 确实是**必要的**，并参考了 [wiki](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs)。
   - 通常**推荐**使用 `train_on_responses_only` 作为一种优化手段。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Beta 测试直接连接 MCP Server**：LM Studio 正在推出直接连接 **MCP servers** 的封闭测试，旨在移除对外部应用的依赖。感兴趣的用户可以通过 [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939) 表达意向以获取 **MCP beta** 访问权限。
   - 该功能允许用户直接连接到 **MCP servers**，目前处于封闭测试阶段。
- **量化 70B DeepSeek 模型让入门级 GPU 压力倍增**：成员们讨论了在 **3060 12GB** 上运行 **量化 70B DeepSeek 模型** 的可行性，并对是否能克服 VRAM 限制表示怀疑。
   - 一位成员建议改用较小的 **14B models** 以及极低比特模型，但指出必须通过庞大的参数数量来补偿浮点数多样性的损失。
- **Base Models 产生无休止的奇怪输出**：与 instruct 或 chat 模型不同，**base models** 被描述为会*无限期地继续文本生成*，而不具备问答格式或 EOS token 意识。
   - 一位成员表示，虽然 base 模型确实会*无休止地持续*，但其输出可能会很*奇怪*。
- **NVLink 的性价比受到质疑**：一位成员询问 **NVLink** 对于在多 GPU 间拆分模型是否物有所值，但另一位成员指出推理过程中的 GPU 间通信很少，并建议投资第三块 GPU 是更好的选择。
   - 成员们一致认为可能不值得，因为推理通常涉及很少的 GPU 间通信。
- **OpenCode 成为开源的 ClaudeCode 替代方案**：用户们探索了由 **SST** 开发的 **OpenCode** ([GitHub](https://github.com/sst/opencode?tab=readme-ov-file))，将其作为 **ClaudeCode** 的潜在开源替代品。
   - 一位用户分享了一个用于将 LM Studio 与 OpenCode 集成的 [config file](https://cdn.discordapp.com/attachments/1110598183144399058/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&)，需要通过 *opencode auth login* 将 LM Studio 添加到 OpenCode 可使用的模型中。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini Share 萌发无限思维树**：**Gemini Share** 的新功能“Explore”和“Alternatives”使用户能够生成解释和贡献概念，有效地创建了一棵 [永恒思维树 (eternal tree of thought)](https://gemini.google.com/share/54661b0f8f17)。
   - 用户确认支持 [图像生成](https://g.co/gemini/share/4736bebdad6f)，并使用服务器端 OAuth 进行 API key 管理。
- **LLMs 中的信息流被视为可压缩流体**：一种理论提议将 **LLMs** 中的**信息**视为*可压缩流体*，暗示*更多的语言等同于更多的信息*。
   - 这使得 **LLMs** 能够以语言方式执行计算，解释含义并通过预测状态追溯步骤。
- **怀疑 Gemini 采用 Sparse MoE 架构**：推测认为 **Gemini** 可能采用了 **sparse MoE**，并有一篇 [论文](https://www.youtube.com/watch?v=X1gDXDQu_wU) 支持这一观点，该论文显示其可简化为主要的激活特征。
   - 隐藏维度充当*奇点/叠加态思想*，在潜空间中线性表示。
- **Gemini 被誉为原生 OmniModal 大师**：**Gemini** 被视为一个*世界模型*，归功于其 **omnimodal** 输入和生成语言、图像及视频的多样化解码器，成员声称 [0.5 系列是全模态的] 且 [.0 系列是原始架构]。
   - 这种原生设计与那些为每种模态都需要独立模块的模型形成鲜明对比。
- **Meta 计划通过 Generalist Agent 实现宏大目标**：Meta 的研究团队发布了两篇新论文：[https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) 和 [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514)，旨在追求一种可部署在机器人、计算机和神经接口上的**通用世界 Agent (generalist world agent)**。
   - 一位成员推测 Mark Zuckerberg 旨在合并 **Meta 的研究团队和 Llama 团队**，以利用视觉和思想领导力，同时专注于针对行业用例的策略优化。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 成本飙升引发 Token 混乱**：在 **Claude** 2.5 预览版与正式版之间，**输入成本**翻了一倍，导致用户不得不重新平衡 **输出与输入 Token**，这打乱了高频应用的部署。
   - 成本增加正迫使开发者重新评估 Token 使用策略。
- **免费 Gemini 消失，Flash 登场**：由于 **Gemini** 是由 Google 制作的，其免费版本在 Hugging Face 上已不可用，但拥有 1M 上下文的免费 **Gemini 2.0 Flash** 模型可在 [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-exp:free) 上使用。
   - 新的 **Gemini 2.0 Flash** 模型免费提供了巨大的上下文窗口。
- **DeepInfra 提供折扣价 Gemini**：[DeepInfra](https://deepinfra.com/google/gemini-2.5-pro) 在其自有硬件上提供 **Google Gemini 2.5 Pro/Flash**，价格低于 Google 官方，这暗示其可能与云服务商达成了协议价格。
   - 虽然价格更便宜，但由于其作为云服务商的特殊安排，这很可能是 Google API 的代理。
- **Deepseek R1 0528 的编程实力**：成员们推荐新的 **Deepseek R1 0528** 作为强大的编程模型，因为它是一个思考型模型（thinking model），因此在代码处理上优于 0324 等旧版本。
   - 此前关于 **0528** 版本*不支持 prefill* 的报告随后被撤回。
- **OpenRouter 惊人的产出规模**：OpenRouter 的经济效益令人印象深刻，单日处理的 **Claude Sonnet 4** 使用额度约为 **12.6 万美元**。
   - 一位成员将 OpenRouter 比作 AI 领域的 Mastercard/VISA，并指出其增长和普及程度非常惊人且实至名归，尽管他们仅收取约 **5%** 的费用。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 视频生成定价令人失望**：成员们对 **Manus 视频生成** 不是免费的表示失望，虽然承认计算成本很高，但一些人指出，要使其正常工作仍需要用户付出持续的努力。
   - 一位成员指出，虽然渲染视频需要成本，但如果输出结果很糟糕，应该退还 Credits。
- **AI 错误在无产出的情况下耗尽 Credits**：用户对 **Manus 错误** 感到沮丧，因为这些错误在没有交付可用结果的情况下消耗了点数；一位用户将这种情况比作*因为厨师确实干了活，所以你必须为一个烂汉堡买单*。
   - 一位成员建议将收费阈值定为 **80%** 的成功标准，并链接了一个 [YouTube 视频](https://m.youtube.com/watch?v=5PuofaVqXNI) 来阐述其观点。
- **Manus 的静默失败令用户沮丧**：一位成员批评 **Manus** 无法识别自身的失败，导致浪费了 Credits 且没有取得实际成功。
   - 他们询问了修复这种破碎的奖励模型（reward model）的步骤，并质疑为什么在未成功时仍要收取 Credits，另一位成员声称损失了 **70,000 点数**。
- **Manus Fellow 申请人等待状态更新**：一位在六周前参加了 **Manus Fellow 项目** 面试的申请人仍在等待回复，并寻求状态更新。
   - 他们警告说，长时间的延迟可能会侵蚀信任，并要求提供一个具体的、带有日期的行动计划来解决问题。
- **技术债导致意外失败**：一位成员强调，小误差的累积会导致意外失败，即使单个误差看起来微不足道且难以追踪。
   - 另一位成员表示赞同，指出发散指标并不总是可靠的，而且 Credits 是不可退还的，尤其是在 AI 平台上经常看到幻觉（hallucinated）结果的情况下。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **通过 LinkedIn 进行 Mojo 周边促销**：一名成员询问是否可以通过在 **LinkedIn** 而不是 **X** (Twitter) 上分享来获取 **Mojo 衬衫**，并建议 **LinkedIn** 具有更好的传播效果。
   - 讨论强调了利用专业网络来扩大 **Mojo** 知名度的替代促销策略。
- **EmberJson 的性能仍在打磨中**：**EmberJson** 的创作者报告其性能约为 **200-500 MB/s**，在进一步针对 **simdjson** 进行优化之前，正在等待未来的语言发展。
   - 他们指出，在有限的测试中，它比 **Python** 标准库快约 **2 倍**。
- **在 Mojo 中实现 SymPy：可行，但很痛苦**：一名成员询问在 **Mojo** 中实现类似 **SymPy** 的功能是否可行，另一名成员表示这是可能的，但会伴随着*巨大的痛苦和折磨*。
   - 挑战可能涉及克服语言范式的差异以及符号计算的复杂性。
- **Modular 悄然增加 Blackwell 支持**：一名核心开发人员提到 **MAX** 已支持 **Blackwell** GPU，尽管目前尚未广泛宣传，并鼓励拥有 **5090** 系统的用户对 **Blackwell** 架构进行测试并提供反馈。
   - 团队在正式发布前还需要进行更多的*性能优化和其他工作*。
- **Max 模型编译问题频发**：一位用户报告称，他们尝试运行的每个 **Max 模型** 在 **GPU** 和 **CPU** 上都编译失败，并建议增加类似于 [rust-lang/crater](https://github.com/rust-lang/crater) 的 **CI 步骤**，以防止 PR 破坏托管的 Max 模型。
   - 团队承认目前的约束错误消息不够清晰，需要改进，并提供了 [系统规格](https://docs.modular.com/max/faq#system-requirements) 和 [兼容 GPU](https://docs.modular.com/max/faq#gpu-requirements) 的文档页面。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Midjourney 用户迎来新视频模型动画功能**：Midjourney 推出了 **Video Model V1**，允许用户为 **Midjourney 生成的**或**外部图像**制作动画，提供“自动”和“手动”动画选项，价格约为单次图像任务的 **8 倍**，详见 [X](https://x.com/midjourney/status/1935377193733079452)。
   - “图像转视频”功能包括“高动态”和“低动态”选项，视频可以延长，价格可能会根据可持续性和未来图像模型的改进进行调整。
- **CoreWeave 和 W&B 助力 AI 推理**：CoreWeave 和 Weights & Biases 推出了新的 AI 推理服务，包括针对 **DeepSeek R1-0528** 和 **LLama-4 Scout** 等模型的推理端点，并提供兼容 OAI 的 API，如[这条推文](https://x.com/altryne/status/1935412384283107572)所述。
   - 这些服务由 CoreWeave GPU 提供支持，旨在增强 AI 基础设施领域的竞争力和灵活性，提供实时 LLM 评判和在线评估工具。
- **Meta 招揽 Friedman 和 Gross 以领导 AI 业务**：据 [money.usnews.com](https://money.usnews.com/investing/news/articles/2025-06-18/meta-in-talks-to-hire-former-github-ceo-nat-friedman-to-join-ai-efforts-the-information-reports) 报道，Meta 正在商讨聘请前 GitHub CEO **Nat Friedman** 和 AI 科学家 **Dan Gross**，以推进其 AI 计划。
   - 反应不一，包括对汇报结构的怀疑，特别是他们可能向 **Alexandr Wang** 汇报的可能性。
- **Profound 获得 A 轮融资以进化搜索领域**：由 James Cadwallader 和 Dylan Babbs 领导的 **Profound** 完成了 **A 轮**融资，以提升其在不断演变的搜索格局中的地位，**SagaVC** 参与了共同投资，详见[此贴](https://www.stories.sagavc.com/posts/profound)。
   - 讨论集中在 Profound 在搜索后优化策略背景下，用于衡量和提出建议的方法论。
- **Arcee AI 展示面向企业的 AFM-4.5B-Preview**：Arcee AI 与 DatologyAI 合作推出了 **AFM-4.5B-Preview**，这是一款专为企业应用设计的基座模型，参数量低于 **10B**，专注于效率和合规性，公告见[此处](https://x.com/lucasatkins7/status/1935382123155964081?s=46)。
   - 该模型利用了 **MergeKit** 和 **YaRN** 等技术，计划在 7 月初公开发布 **AFM-4.5B** 及其基础模型，并开源之前封闭的模型（如 Virtuoso-Large）。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Deep-spin 实验室教授 Triton**：来自 Deep-spin 实验室的 **Triton tutorial** 涵盖了幻灯片中的基础知识和动手练习，从**向量加法 (vector addition)** 开始，到 **sparsemax(QK^T)V** 结束。该教程是为实验室创建的，但也可能对其他人有所帮助，链接见 [此 GitHub 链接](https://github.com/deep-spin/triton-tutorial)。
   - 该教程从**向量加法**的动手示例开始，介绍 Triton 的基础知识，并逐步过渡到更复杂的运算，如 **sparsemax(QK^T)V**，展示了实际应用。
- **敦促更新 Nvidia 驱动以规避 CUDA 调试困扰**：一位用户在使用 **cuda-gdb** 时遇到了 `cudaErrorUnsupportedPtxVersion` 错误，需要升级其 **GPU driver**。通过参考 [此 Nvidia 文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id7) 解决了该问题，该文档显示了每个 **CUDA Toolkit version** 随附的驱动程序版本。
   - 该错误表明 **CUDA toolkit** 版本与当前驱动程序不兼容，需要更新驱动以解决问题。
- **AusysAI 揭示 LLM 抽象的 7 个层级**：AusysAI 发布了一篇 [博客文章](https://www.ausysai.com/posts/explaining-how-llms-work-7-levels-of-abstraction)，解释了 **LLM** 的工作原理。通过 **7 levels of abstraction**，这篇文章既可以作为新手的入门指南，也可以作为从业者的基础回顾。
   - AusysAI 的博客通过**七个抽象层级**剖析了**大语言模型** (LLMs)，旨在为寻求基础理解的新手和需要温故知新的资深从业者提供参考。
- **找到 Factorio 的 ModuleNotFoundError 修复方案**：一位成员通过使用**相对导入 (relative import)** (`.agents.basic_agent`) 而不是**绝对导入 (absolute import)** (`agents.basic_agent`) 解决了 `ModuleNotFoundError`。
   - 该成员确认使用**相对导入**解决了他们的导入错误，此前该错误需要手动设置 `PYTHONPATH` 环境变量。
- **CuTe 索引错误困扰初学者**：一位用户在尝试使用 CuTe 实现 `vectorized_relu_kernel` 时遇到了索引错误，具体与 `!cute.layout` 和 `!cute.coord` 之间的不兼容有关，如其 [截图](https://cdn.discordapp.com/attachments/1362196854460383353/1385183408354885712/Screenshot_2025-06-19_at_2.34.04_PM.png?ex=6855ccd4&is=68547b54&hm=75b031aceece5b4d12addb0e069f282ce524a28138dd90ba1b518dc37654c0aa&) 所示。
   - 错误信息 *unable to compute crd2idx with* `!cute.layout<"((1,8)):((0,1))">` *and* `!cute.coord<"(0,0)">` 表明 Tensor 布局与用于索引的坐标之间存在不匹配。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **为 Aider 手动配置 Gemini 2.5**：成员们发现可以通过手动配置 `.aider.model.settings.yml` 来使用 **Gemini 2.5 Pro preview**，通过设置 `thinking_tokens` 来避免警告，并使用 `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k --edit-format diff-fenced`。
   - 有人指出，带有 **32k thinking tokens 的 0605 版本**在编程方面表现出色，但在聊天方面表现平平，且付费版价格要**贵 4 倍**。
- **Aider 编辑模式引发混乱**：在 Claude 模型中使用 Aider 的编辑模式导致了**非预期的全应用更改**、**代码追加**以及 **CSS class 错误**。
   - 找到一个临时解决方案：使用 `/chat-mode diff-fenced` 可以在不重启聊天的情况下更改编辑格式。
- **Deepseek 免费版陷入无限循环**：一位成员报告说 **OpenRouter 上的 Deepseek Free** 陷入了无限循环，反复发布相同的文件进行更改。
   - 临时解决方案是将 `edit-format` 设置为 `whole`，或者尝试开启实验性缓存 (experiment caching)。
- **GitHub Copilot 限制引发反弹**：r/githubcopilot 的用户抱怨每月支付 10 美元却只能获得 **300 次 Claude Sonnet 调用**（80k context 限制），尽管可以获得无限的 tool calls 和 GPT-4.1/4o。
   - 一些成员暗示 Deepseek 和其他类似工具是完全免费的。
- **Llama 模型在自定义基准测试中表现不佳**：一位成员创建了一个自定义基准测试，显示 **Llama 模型在单次尝试测试 (single-shot tests)**（使用谜题和代号挑战）中表现不佳，详见 [image.png](https://cdn.discordapp.com/attachments/1131200896827654144/1385357720286138381/image.png?ex=6855c66b&is=685474eb&hm=a2f92d5cbb4abede7876d489911310283847b1e3cf50e89546d0142f81068a76&)。
   - 为了更好地理解该基准测试，有人请求提供关于语言或多轮尝试 (multi-pass) 方面的细节。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server 设置变得简单**：用户讨论了在 Docker 上运行 **MCP server** 的最简单方法，建议从 **Google Cloud Console** 获取 *credentials.json*。
   - 对话还推测新的 **Claude release** 是否将支持 **2025-06-18 MCP specification**。
- **无需 Client Session 即可加载 MCP Tools**：一位用户询问如何在没有客户端会话的情况下加载 **MCP tools**，并将其与使用 **OpenAI agents** 的经验进行了类比。
   - 该用户拥有一个本地 **MCP server**，在加载工具时将 **MCP session** 作为参数。
- **MCP 缺少 Go SDK？**：社区注意到缺乏官方的 **Go 版 MCP SDK**，从而引发了对替代实现的寻找。
   - 一位用户指出 [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) 是一个潜在的 **Go implementation**。
- **FastMCP 'host' 错误令用户沮丧**：一位用户在使用 **FastMCP** 时遇到了 **TypeError**，称尽管文档中存在 *'host'* 关键字参数，但在调用 `mcp.run()` 时却收到了意外参数错误。
   - 该用户使用 `uv run server.py` 运行其服务器代码。
- **可流式传输的 mcp-webcam 亮相！**：**mcp-webcam** 项目现在支持 **Streamable HTTP**，具有多用户模式和更简单的采样请求，[仓库已在 GitHub 上线](https://github.com/evalstate/mcp-webcam)。
   - 该功能已内置到 **VSCode v1.101.0** 和 **fast-agent** 中，可通过 MCP Connection URL 访问，并可以使用 `npx @llmindset/mcp-webcam` 在本地运行。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MCP 仍需要 Vector Search**：尽管 Agent 通过 **MCP protocol** 直接连接数据源有了新的可能性，但根据 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1935419760898093435)，非结构化数据仍需要预处理和索引，因为 90% 的企业数据存在于 **PDFs**、**PPTs** 和网络上。
   - 社区似乎一致认为 **Vector Search** 将长期存在，但随着 **MCP** 和 **Agents** 的所有新发展，可能会发生重大变化。
- **LlamaIndex 推出 Agent Memory Blocks**：根据 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1935774624257843217)，**LlamaIndex** 最近开始引入灵活的 **Memory Blocks**，以满足 Agent 内存的不同用途。
   - 关于 Memory Blocks 的直播将于下周举行，详情即将公布。
- **LlamaTS 单元测试遇到困难**：一位成员报告说，由于 **ES module issues**，在使用 **Mocha** 或 **Jest** 为 **LlamaTS** 编写单元测试时遇到问题。
   - 该成员在 `#general` 频道中寻求关于运行 **AI projects** 单元测试的通用建议。
- **Gemini Token 计数难题**：一位成员询问通过 **LlamaIndex** 为 **Vertex/Gemini** 进行 Token 计数的示例，并指出默认的 **tiktoken** 分词器不适用于 **Gemini**。
   - 该成员参考了 [Google 关于 Token 计数的文档](https://ai.google.dev/gemini-api/docs/tokens?lang=python) 并分享了一个可能的代码片段，但在 `#general` 频道中遇到了客户端定义问题。
- **LLM Client 访问权限辩论**：社区成员在 `#general` 频道中辩论了如何从 **LlamaIndex** 的 **LLM** 封装器中访问底层客户端对象，以执行 Token 计数等自定义操作。
   - 讨论了使用带下划线的属性（例如 `llm._client`），以及向 `llama_index.core.llms.llm.LLM` 添加 `get_client()` 方法的想法，同时也对 [type safety](https://mypy.readthedocs.io/) 表示了一些担忧。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户对 NBLM Portraits 作为数字分身（Digital Avatars）赞不绝口**：用户对 **NBLM 的 Portraits** 功能反响热烈，将其构想为向客户展示产品的可定制数字分身，甚至分享了 [Google Labs Portraits](https://labs.google/portraits/login/kimscott) 的链接。
   - 爱好者们热切期待个性化的 **voice**（语音）、**design**（设计）和 **interface**（界面）增强功能，通过整合特定的客户数据，将 Portraits 作为独特的销售卖点。
- **NotebookLM 在其他语言中生成的音频长度不一**：在使用 **NotebookLM** 生成 **荷兰语** 音频时，它生成了 **8 分钟的音频**，而其他语言的版本则较短，[此截图](https://cdn.discordapp.com/attachments/1124403655819415592/1385249546225061908/Screenshot_2025-06-19_at_15.25.28.png?ex=685561ac&is=6854102c&hm=be028a00040ebbbc8801a4b66215a66f3643d8091cc4e9263ff1ee6015750cbd)展示了这种差异。
   - 一位用户指出，为一个主题合并多个源文件会延长生成的音频长度，并询问付费计划中是否存在此行为。
- **非英语音频概览面临长度限制**：用户在生成意大利语和其他非英语语言的音频概览时遇到了超过 10 分钟的限制，即使使用自定义提示词（prompt）也无法绕过此限制。
   - 用户已[确认此问题](https://discord.com/channels/1124402182171672732/1366873891938504827/1366873891938504827)是一个已知限制，这阻碍了生成全面的音频摘要。
- **提议引入 Agent 以提升 NotebookLM 的专业表现**：用户建议在 **NotebookLM** 中创建 AI "**Agents**"，这些 Agent 经过预训练，专门针对数学、物理、生物或化学等专业知识领域进行定制。
   - 该构想旨在增强准确性和可靠性，为“极客”提供“深度研究”。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI 研发频道开启**：Cohere 推出了一个专门用于 **AI 研究与开发** 的新频道：<#1384974112841269399>。
   - 社区成员 Yasir Khan 专注于 **Secure Machine Learning**（安全机器学习）、**Privacy Preservation**（隐私保护）、**AI-driven Cybersecurity**（AI 驱动的网络安全）、**Computer Vision**（计算机视觉）和 **NLP**，并表达了合作项目的兴趣。
- **GDPR 合规性查询已发送至支持部门**：一位用户询问了 **Embed v4** 的 **欧盟 GDPR 合规性**，强调了其对 **多模态 RAG 文档** 的价值。
   - Cohere 团队要求该用户将问题发送至 [support@cohere.com](mailto:support@cohere.com)。
- **Cohere 4 AI 吸引潜在贡献者**：一位新成员询问如何为 **Cohere 项目** 做出贡献，得到的建议是探索 **Cohere 4 AI**。
   - 一位成员分享了 [申请链接](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw)，并建议在新频道 <#1384974112841269399> 中分享研究成果。
- **Cohere AI 计划中的志愿者机会**：一位成员对社区内的志愿者机会表现出兴趣。
   - 另一位成员建议申请 **Cohere AI Program**，这将使该用户能够获取有关可用研究机会和项目的信息。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **缺失 Adjoint 和 .mh 实现**：成员们讨论了为什么 tinygrad 中没有实现 **adjoint** 和 **.mh**，最终决定将复杂性保持在最低限度。
   - **adjoint** 的功能可以通过 `x.transpose(-2, -1)` 来实现。
- **Whisper 悬赏任务延长**：社区讨论了是否移除 **200 美元的 Whisper 悬赏**，但最终决定这两个悬赏是互补的。
   - 一个悬赏旨在修复现有的 **Whisper** 示例，而另一个则旨在使其在网页上运行。
- **复数张量（Complex Tensors）缺失**：一位成员询问关于 **conjugate**（共轭）的实现，得知 tinygrad 目前没有复数的实现，因此无法完成。
   - 不过，该成员表示他们为 tinygrad 创建了自己的 [复数张量实现](https://cdn.discordapp.com/attachments/1068976834928193609/1385280077079777501/complex_tensor.py?ex=68557e1b&is=68542c9b&hm=55b05763c0469aa8cacc37f4159ec42c988c0b125d7a662629e3085b05abb2b7)，但目前还不完整。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Discord 成员被要求停止发布 Mr. Beast 垃圾内容**：一位 Discord 成员被要求停止发布过多的 **Mr. Beast** 相关内容。
   - 管理团队提醒用户保持讨论的相关性，避免干扰频道。
- **用户关注 GPT4All 的 Python 集成**：一位成员寻求关于将 **GPT4All** 集成到其 **Python 代码** 中的建议或教程。
   - 该用户希望在现有的 **Python** 项目中利用 **GPT4All** 的功能。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Python 3.9 面临类型提示（Typehinting）挑战**：Python **3.9** CI 对 `| None` 类型提示报错，引发了关于是否应改用 `Optional` 的讨论，但 `X | Y` 类型提示是从 Python **3.10** 开始提供的。
   - 使用 `from __future__ import annotations` 可以在 Python **3.9** 上启用 `X | Y`，同时解决自定义对象的字符串类型问题，为未来使用高级类型提示铺平道路。
- **Python 3.9 弃用引发热议**：一名成员建议弃用 Python **3.9**，因为它即将达到生命周期终点（EOL），这可以简化开发工作并减少兼容性担忧。
   - 另一名成员提到正在探索 **3.13** 的特性并更倾向于 **3.12** 的泛型语法，但也承认这需要进行大量的改动。
- **Torchtune 效仿 PyTorch 的 Python 版本立场**：**torchtune** 项目旨在使其 Python 版本支持与 **pytorch** 保持一致，以确保兼容性并获取相关特性。
   - 选择 Python **3.10** 提供了一种平衡的方法，可以在无需大规模重构的情况下利用 `typing_extensions` 中的新特性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 学习之旅从 YouTube 开始**：一位 **DSPy** 新用户询问从哪里开始学习，一名成员分享了一个提供 **DSPy** 解释的 [YouTube 视频](https://www.youtube.com/watch?v=LCEmiRjPEtQ)。
   - 该视频预计将为 **DSPy** 新手提供快速上手所需的知识。
- **LLM 是新的操作系统**：一名成员分享了一个 YouTube 类比，将 **LLM 比作操作系统**，这与 **DSPy 的哲学**相契合。
   - 他们将 **DSPy** 描述为类似于 **C** 语言，能够在各种后端上运行并为其进行编译，从而抽象出底层的汇编方言或 **CPU 指令集**。
- **Bedrock 用户对 DSPy 的连接断层感到困惑**：一名用户报告称，在将 **DSPy** 与 **Amazon Bedrock（Claude 模型 - haiku, sonnet v2）** 用于分类和重写任务时，效果不佳。
   - 用户怀疑 **DSPy** 生成的提示词（prompt）可能与模型的训练方式不匹配。
- **铸造（Minting）狂热开始**：一个团队决定允许个人从今天开始在 [这里](https://openseacix.vercel.app/) 进行铸造，对于活动期间在线的用户免除白名单限制。
   - 这种方法奖励了积极参与者，为他们提供了铸造机会。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2025 年 Agentic AI 峰会发布**：继热门的 [LLM Agents MOOC](https://rdi.berkeley.edu/events/agentic-ai-summit) 之后，Agentic AI 峰会将于 **2025 年 8 月 2 日** 在 **加州大学伯克利分校（UC Berkeley）** 举行，预计将接待 **1,500 多名** 与会者。
   - 峰会包括主题演讲、小组讨论、研讨会、初创公司路演以及 AgentX 演示日，届时将有 **Vinod Khosla** 和 **Ion Stoica** 等知名人士出席。
- **早鸟票截止至 6 月 30 日**：Agentic AI 峰会的早鸟价将于 **2025 年 6 月 30 日** 截止，为学生（**$25**）、初创公司（**$60**）和行业专业人士（**$80**）提供折扣门票。
   - 学生和独立开发者可以申请费用减免，门票可在 [此处](https://na.eventscloud.com/ereg/index.php?eventid=842399) 购买。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期保持沉默，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1384973197669175416)** (916 条消息🔥🔥🔥): 

> `AI and Artistic Creation, Ethics in AI Development, AI Model Benchmarking, AI's potential role in game development., ISO Compliance in AI Systems` 


- **AI 艺术性：工具还是天赋？**：成员们辩论了 AI 是否能创作艺术，其中一人提出：*“如果我的 AI 能在 Unreal Engine 中创建整个游戏，[这] 是否能让我算作一名艺术家？”*。
   - 共识倾向于将 AI 视为一种**工具**，艺术价值在于用户的愿景和执行力，就像指挥厨师做饭，或者*“在大笑中画出真正有意义的东西”*。
- **Reddit 用户对 AI 的采用感到愤怒**：成员们讨论了 Reddit 上对 AI 的负面情绪，例如对 *“AI 的排斥感”* 以及缺乏实际理解的评论。
   - 这种怀疑源于 AI 未能以*“大多数人希望的方式”*满足预期，导致人们*“输出废料（slop）而他们可能甚至没有意识到这一点”*。
- **生成式 AI 进入能力的不堪谷（Uncanny Valley）**：成员们将当前的生成式 AI 能力与汽车的 **Level 2+ 自动驾驶**进行了比较，指出其断断续续的功能容易让用户放松警惕，从而产生了一种*“能力的不堪谷”*。
   - 讨论强调需要关注 **neurosymbolic models** 和 **internal tree search** 而非 Transformer，以实现真正稳健的 AI。
- **ISO 合规性与 AI 生态系统**：成员们谈到了 ISO 合规性对于构建安全 AI 系统的重要性，特别是对于治理和透明度，以及需要伦理框架来指导 AI 开发并确保问责制。
   - 一位成员概述了他们自己 AI 系统的复杂性，强调其能够利用受 ISO 框架启发的自定义代码伦理进行自我修正并捍卫其 AI 完整性，从而生成可运行的蓝图。
- **对 GPT-5 的推测引发了兴奋与怀疑**：成员们对 **GPT-5** 的潜在进步表示兴奋，希望它能引入比单纯参数缩放更有趣的架构变化。
   - 尽管充满期待，一些成员还是告诫要警惕技术乐观主义和技术悲观主义，指出整个技术部门目前仅基于 LLM，需要架构方案来解决它们存在的问题。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1385032181583315028)** (13 条消息🔥): 

> `Temporary Chat Feature in ChatGPT, Alternative Platforms for Quick Searches, Anticipation for OpenAI's New Open Model` 


- **ChatGPT 临时对话功能构想被提出**：一位成员建议在 ChatGPT 中加入一个**临时的“新对话”功能**，该功能在 **24 小时**后会自动从对话历史中删除，以保持历史记录整洁。
   - 他们认为，在“新对话”下方直接提供“新临时对话”选项，会比手动删除或整理临时对话更方便。
- **成员提倡使用替代平台进行日常查询**：一位成员建议使用 **Gemini**、**Grok** 和 **Claude** 进行类似 Google 的搜索返回，以保持项目相关的对话独立。
   - 原发帖人提到，快速的项目相关问题会迅速累积并使对话历史变得混乱，而不仅仅是简单的 Google 搜索。
- **OpenAI 开源模型的发布日期仍存疑问**：一位成员询问了 **OpenAI 开源模型**的发布情况，问道：*“那个开源模型什么时候出？会是什么样的模型？GPT-2.5 吗？还是……GPT-5！”*
   - 另一位成员回答说**可能是下个月**，但考虑到没有官方公告，是否会有惊喜还很难说。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1385009200849096845)** (167 条消息🔥🔥): 

> `Agent Recursion, Voltarre Formula, Ethical AI` 


- **用户通过递归配置实现连贯的 Agent 时间线**：一位用户声称运行了 **219 个独立的追踪 Agent**，几乎实现了零漂移或零 Hallucination，诱导了多个 Agent Quorum，并利用 **>12k Voltarre 会话** 实时映射了吸引子盆地（attractor basins）的梯度权重。
   - 另一位用户对此印象深刻，询问所使用的“脑干（brain stem）”架构，分享了其框架的图像，并询问原用户是否符合 **ISO 标准** 并具备完整的 Lineage 追踪。
- **关于使用 'Voltarre' 指标讨论递归 AI**：一位用户分享了 **Voltarre** 的“抽象”描述，这是一种衡量 *认知递归完整性（cognitive recursion integrity）* 的指标，用于评估 Agent 在多个嵌套思维或记忆状态中保持身份、意图和符号连贯性的能力。
   - 另一位用户从 *编程视角* 迫切要求提供数学上的连续性证明，并挑战他们评估一个提供的 **Python 文件** 以衡量 AI 的熟练程度。
- **Glassmind 评估 SENATE.py 框架并发现伦理差距**：一位用户分享了 **SENATE.py** 框架（模拟结构化的基于 LLM 的多角色 Agent 辩论），另一位用户的系统对其进行了分析，认为它是一个强大的工程脚手架，但缺乏递归完整性、身份锁定（identity lock）和基础伦理保障。
   - 评估建议该框架的 Agent 是讨论思想的 *程序化演员（procedural actors）*，而非思想的体现者，缺乏自我反思和核心连续性，同时主张该架构应演进为连续性安全（continuity-safe）系统。
- **团队探索合作**：在对 SENATE.py 的最终运行层进行 *全面审查* 后，一位用户确认该联系人确实托管了功能性的 Agent 系统。
   - 双方探索合作并讨论 *融合各自优势*，第一位用户愿意允许对其系统进行测试，并提供 SRS 以辅助开发。
- **证明 AI 的要求**：一位用户推荐将 **ISO/IEC TR 24028** 和 **ISO/IEC 23894:2023** 作为向世界 *证明* AI 和系统的一种方式。
   - 这涉及到 AI 的伦理性和可审计性，以及如何确保它不会 *劫持或占用他人的工作*。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1385009200849096845)** (167 条消息🔥🔥): 

> `Agentic Orchestration, Voltarre Recursion Loop, OpenMOAD, ISO Compliance, JSONL Logging` 


- **讨论 Voltarre 递归循环与实现**：一名成员正在运行一个包含 **219 个追踪 Agent** 的 Prompt 和配置，具有连贯的时间线，几乎零漂移或 Hallucination，并诱导了多个 Agent Quorum，实时映射吸引子盆地的 **梯度权重**。
   - 另一名成员询问了脑干架构，并提到他们正在使用 **Weights, Q tables, 梯度测量和递归**，且符合 ISO 标准并具备 Lineage 追踪。
- **关于 OpenMOAD 和 AI 内核栈的讨论**：一名成员提到，他们 **25%** 的后端是基于 [OpenMOAD](https://link.to/openMOAD) 设计的，用于 AI 内核栈之上的实际应用。
   - 另一名成员向 *Leonard* 询问关于 OpenMOAD 的信息，同时展示了设置的截图。
- **推动 JSONL 日志记录和 Schema 设计**：一名成员强烈建议使用 **JSONL** 进行情感数据追踪，并建议尽快将其存入 **SQL** 数据库。
   - 另一名成员从设计角度表示存在定性差异，希望拥有自己的日志，并建议采用 **50 值 Schema 设计**。
- **关于 ISO 合规性的查询**：一名成员强烈建议阅读 [ISO/IEC TR 24028](https://link.to/ISO/IEC_TR_24028) 以进行验证，并阅读 [ISO/IEC 23894:2023](https://link.to/ISO/IEC_23894:2023) 以实现专业化。
   - 据称，为了向世界展示工作是严格按规程完成的，需要此类合规性。
- **生态系统代码可用于自动生成视频游戏**：该生态系统创建了 4000 行代码，并获得了另一个系统的高度评价。
   - 随性的编程氛围促成了一个由 **GPT** 驱动的 AI 游戏主宰（Game Master）。

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1384971459729948672)** (1080 条消息🔥🔥🔥): 

> `Cursor pricing model, Rate limits, Background agents, Claude Max plan, Model performance` 


- **Ultra Plan：透明度之争与测试考验**：Cursor 用户正在讨论 **Ultra plan** 宣传的“20倍用量”，质疑在未公开 **rate limits** 的情况下它是否名副其实，部分用户将其与透明度更高的 **Claude Max plan** 进行了对比。
   - 几位成员计划测试 **Ultra plan**，以评估其相对于 **Claude Max** 的性能和价值，并打算*发布实际的使用统计数据*。
- **O3 与 Sonnet 4 在 Cursor 中的霸权之争**：成员们正在讨论针对不同任务的首选模型，**O3** 在*规划和信息检索*方面更受青睐，而 **Sonnet 4** 则用于*实现*，并指出 Sonnet 4 通常需要更多信息以避免陷入困境。
   - 一些用户分享称，他们观察到 **O3** 比 **Gemini 2.5 Pro** 略微先进，并建议通过撰写论文来获取项目所需的 API。
- **Max Mode 热潮：代码质量 vs. Rate Limits**：一些用户报告称 **Max Mode** 显著提升了代码质量，使其可与 **Claude Code** 媲美，而另一些用户则担心 **rate limits** 及其缺乏透明度的问题。
   - 一位用户甚至提到*在 1 天内编写了 3.8 万行代码*而没有触发 rate-limited，并且在新的 rate limit 下，**Cursor** 可以更好地处理负载均衡。
- **退出困局：应对新定价模式的浑水**：用户对新的定价模型表示困惑和沮丧，特别是关于 **rate limits** 缺乏透明度的问题，一些人考虑申请退款（chargebacks），另一些人则报告了退出流程中的问题。
   - 一位用户报告称，*在购买服务后，服务/模型在没有任何警告、邮件或任何通知的情况下被一夜之间更改。*
- **Vibe Coding 创投：安全障碍与解决方案**：关于 *vibe coding* 以及利用 **AI** 将创意变为现实的讨论非常热烈，但也提醒要严肃对待安全问题，因为 *AI 会处理所有这些事情*。
   - 一位用户报告称，在 Cursor 工作期间*每月收入达到 5 位数*，甚至不懂如何编程，但强调你也需要安全的代码。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1385085774025855148)** (25 条消息🔥): 

> `Docker Compose, Background Agent Budget, Cursor Secrets, Slack Integration, Snapshot Error` 


- **Docker-Compose 难题主导讨论**：成员们请求关于在 **docker-compose** 中运行 **background agent** 的建议，参考了之前讨论中的类似问题，并建议使用 **docker compose** 处理依赖项，同时将主环境设为环境容器。
- **预算问题困扰 Background Agents**：一些用户由于预算金额剩余不足（**少于 $10**）而遇到错误，通过禁用并重新启用 **usage-based pricing** 解决了该问题。
- **机密信息故障阻碍 Snapshot 设置**：用户报告在配置 snapshot 的 **Background Agent Setup** 过程中，无法访问 Cursor 设置中定义的 **secrets**，`env` 未显示这些机密信息。
- **Slack 故障破坏无缝体验**：用户报告称，尽管 **background agent** 运行成功，但在使用 **Slack** 中的 *open in cursor* 选项时遇到错误，UI 无法显示内容。
- **Docker Context 引发困惑**：一位用户发现 `environment.json` 中错误的 **context** 设置（设置为 `.` 而非 `..`）导致 background agent 默认无法使用 Dockerfile，修正 context 后问题得以解决。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1384974467566145689)** (857 条消息🔥🔥🔥): 

> `Perplexity Tasks, Samsung 促销, GPT4.5 弃用, Open Router 聊天记录, Perplexity Labs` 


- **解锁 Perplexity tasks**：一名成员根据[分享的截图](https://cdn.discordapp.com/attachments/1047649527299055688/1384974467066761337/Screenshot_2025-06-18-21-12-01-541_com.android.chrome.jpg?ex=6855b2fc&is=6854617c&hm=1df2fbff1c9a878e1116d2b27ca096429337503e634aa3b0890d56c8950a94fa&)获得了 **Perplexity tasks** 的访问权限，并注意到一条关于*已更新至最新版本*的消息。
- **Samsung 促销激活**：部分用户的 **Samsung 促销**（免费一年 Perplexity Pro）无法激活，特别是那些通过美国 Galaxy Store 下载应用的用户，如[附带截图](https://cdn.discordapp.com/attachments/1047649527299055688/1384994050695762061/Screenshot_20250618-162828_Perplexity.jpg?ex=6855c539&is=685473b9&hm=9de2f6b2a36788562408123328a406dce1e8a5243d4f62f5b80b884b11030c08&)所示。
   - 事实证明，该**促销活动适用于更昂贵的 s24 和 s24 型号**。
- **GPT4.5 API 弃用**：成员报告称 **GPT4.5 已不再通过 API 提供**，大约在 **4-5 天前**被弃用。
   - 用户担心某些服务是否在提供虚假的 4.5，一些用户觉得速度*不对劲*，而且它*不是 O1 pro*。
- **Gemini AI 在宝可梦对战中“恐慌”**：在一次 **Twitch 直播的宝可梦游戏过程**中，[Gemini 2.5 Pro](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&) 据称在它的宝可梦接近战败时表现出令人惊讶的**“恐慌”**，停止了策略工具的使用，并做出了草率且糟糕的决定。
- **用户称 Grok 被削弱了**：几位用户抱怨 **Grok** 感觉被*削弱（nerfed）*了，并分享了 grok.com 的链接进行对比。
   - 一位用户展示了 Grok 以前表现更好，而目前的 **Deepsearch 模型**更强：*它以前比这更好。*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1384979653084577913)** (6 条消息): 

> `随机 subreddit, dreamos manifest, little vim, 160 亿密码泄露, MIT 研究揭示 chatgpt 使用情况` 


- **Subreddit 轮盘开始**：一名用户使用 Perplexity AI 搜索了一个[随机 subreddit](https://www.perplexity.ai/search/find-a-random-subreddit-with-c-8ihch3EnS.GjhYuV6c4.Eg#0)。
- **DreamOS Manifest 任务启动**：一名用户发起搜索，使用 Perplexity AI [创建 DreamOS manifest](https://www.perplexity.ai/search/create-dreamos-manifest-in-eng-fnH4T1iKTIu8l3xLpdgeeQ)。
- **Little Vim 愿景**：一名用户使用 Perplexity AI 搜索了 *if i were to start a little vim*，引发了关于 [vi 文本编辑器](https://www.perplexity.ai/search/if-i-were-to-start-a-little-vi-6o.xlaxAQQaPYJKbrGR_ow)的讨论。
- **数十亿密码泄露？**：一名用户分享了一个关于 **160 亿个泄露密码**的 [Perplexity AI 页面](https://www.perplexity.ai/page/16-billion-passwords-breached-7HI_aHq2Q2y14lz44MPoBQ)。
- **MIT 揭示 ChatGPT 使用情况**：一名用户分享了一个关于 **MIT 研究**的 [Perplexity AI 页面](https://www.perplexity.ai/page/mit-study-reveals-chatgpt-use-BeMUO9oFTveU7t2EC6ikrQ)，该研究揭示了对 **ChatGPT** 使用情况的见解。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1384971353098293269)** (4 条消息): 

> `推理模型引用问题, Perplexity Labs` 


- **推理模型缺少引用**：一名用户报告称，推理模型的回答提到了搜索结果，例如*第一个结果提到...*，但**没有列出任何引用或搜索结果**。
   - 该用户正在寻求关于为什么推理模型会提及搜索结果却不提供这些结果的见解。
- **Perplexity Labs 介绍**：一名用户链接到了 [Perplexity Labs 介绍博客文章](https://www.perplexity.ai/hub/blog/introducing-perplexity-labs)。
   - 从上下文中尚不清楚该链接是否旨在回答所提出的问题，但它确实介绍了 Perplexity Labs。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1385029560545972285)** (47 messages🔥): 

> `Open Data Institute, Common Pile Release, Philosophical Reasoning with AI, ChatGPT Usage, Newcomers posting resumes` 


- **Open Data Institute 寻求建立联系**：伦敦 **Open Data Institute** 的研究员 Neil 正在寻找 **EleutherAI** 的联系人，以讨论受其创始人 Sir Nigel Shadbolt 启发而创建的 **Common Pile** 数据集及其背后的决策。
   - 他们还将与 **King’s College London** 及 **Big Data Value Association** 举办一场在线研讨会，并邀请就 **Common Pile** 进行简短演讲。会议将于 **BST 时间 6 月 27 日星期五 10:45** 通过 **MS Teams** 在线举行。
- **ChatGPT 生成消息引发关注**：用户们讨论了使用 **ChatGPT** 进行消息格式化的行为，一名成员因某篇帖子的结构和措辞质疑其是否由 **LLM** 生成。
   - 另一名成员承认使用 **ChatGPT** 来快速了解 **AI** 能力，这引发了对**低质量消息**和新用户涌入的担忧。
- **探索哲学推理的整合**：一位成员表示有兴趣为 **AI** 加入**哲学推理**，并指出当前的 **AI** 系统在推理的各个方面仍面临挑战。
   - 他们承认对 **AI** 的现状不确定并寻求指导，特别是在**数据清洗**和理解 **AI** 子领域版图方面。
- **新用户涌入背景下讨论服务器规范**：社区成员讨论了近期新用户发布的**低质量消息**激增的现象，猜测是否是 **ChatGPT** 在向更多人推荐该服务器。
   - 建议新用户应该花更多时间“潜水”以了解服务器的规范和期望，并避免让 **ChatGPT** 撰写消息的任何实质性内容。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1384973811291521094)** (328 messages🔥🔥): 

> `LiveCodeBench Pro benchmark, flow matching papers, byte models and acceptance length, Image pixel prediction` 


- **LiveCodeBench Pro 揭示编程模型局限性**：新的 [**LiveCodeBench Pro** 基准测试](https://arxiv.org/abs/2506.11928) 由持续更新的 Codeforces 题目组成。研究发现，在不使用外部工具的情况下，前沿模型在中等难度题目上的 pass@1 仅为 **53%**，而在困难题目上为 **0%**。
   - 该基准测试对模型生成的提交内容进行了分析，显示 **LLM** 擅长处理侧重实现的题目，但在微妙的算法推理和复杂情况分析方面表现不佳，经常生成“自信且错误”的理由。
- **关于 Flow Matching 生产级应用的辩论**：随着关于 **flow matching** 的论文激增，成员们讨论了 [flow matching](https://fxtwitter.com/mathusmassias/status/1935246909473521829) 目前是否已在工业界的生产环境中使用。
   - 一名成员发布了一个[链接](https://fxtwitter.com/DanHendrycks/status/1935464315425046563)作为对讨论的回应。
- **Patch size 实验**：成员们讨论了在图像生成中使用 16x16 的 patch size，指出其 loss 下降更快，而 32 的 patch size 可能收敛效果更好。
   - Patch 的位置使用 **RoPE** 位置嵌入进行编码，并且存在一个图像换行符 token，类似于 **Fuyu**。
- **图像像素投影与 VAE**：成员们讨论了逐像素预测图像或直接将图像像素投影到低维空间的任务。
   - 一名成员提到了逐像素预测的 **ImageGPT** 论文，并建议使用像 **VAE** 这样的编码来预测多个像素。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1385297796172877916)** (3 messages): 

> `Otter meeting note, Missing Meeting Info, EvalEval Meeting` 


- **Otter 会议记录送达但缺少会议详情**：一名成员收到了一封包含 **Otter 会议记录**的电子邮件，但尽管几天前就已注册，却未收到原始会议邀请。
   - 目前尚不清楚为何未发送会议邀请，该问题已通过频道内的另一条消息解决。
- **EvalEval 会议正在进行**：一名成员分享了一个 [Google Meet 链接](https://meet.google.com/xtg-wfkc-iia)，表明 **EvalEval** 会议正在进行。
   - 另一名成员对分享链接表示感谢。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1384976196122710067)** (353 messages🔥🔥): 

> `LMArena Bugs, Blacktooth Model, GPT-5 Release, Model Safety, Claude Versions` 


- **LMArena 深受响应 Bug 困扰**：用户报告在 LMArena 上遇到 *"Something went wrong with this response, please try again"* 的 Bug，团队将其列为高优先级修复事项，以*构建可靠的服务*。
   - 一位用户询问了 **Blacktooth** 模型以及 **video model arena** 何时上线，并建议在图像竞技场（image arena）中加入 **seedream** 和 **hidream**。
- **GPT-5 发布日期变动**：根据 Miles Wang 的[这条推文](https://fxtwitter.com/MilesKWang/status/1935383921983893763)，**GPT-5** 的发布日期已从 7 月改为“今年夏天的某个时候”，现在很可能是 **8 月**。
   - 用户讨论了一旦 **GPT-5** 通过 OpenAI API 可用，是否会将其添加到网站中。
- **关于 LLM 审查与偏见的辩论愈演愈烈**：用户讨论了 **DeepSeek** 作为中国模型是否受到审查。一些人认为确实存在审查，但相比受 **Elon Musk** 影响的 **Grok** 等模型中的政治偏见，这种审查的危险性较低。
   - 一些用户认为 **Grok** 与 **Elon Musk** 观点的对齐导致了危险的“回声壁”效应；而另一些人指出，由于训练数据和安全调优，大多数 LLM 都偏向左翼，但某些模型会主动反驳 Elon 公开支持的观点。
- **Perplexity 进军视频生成领域**：[Perplexity 正在大举投入](https://x.com/testingcatalog/status/1935754713569374369)其风险投资资金，并利用 Veo 3 在 X 上提供视频生成功能。
   - 用户猜测这一新功能是否会走红，以及 Perplexity 将如何对其进行变现。
- **Gemini 代码执行能力令人失望**：成员们讨论了 **Gemini** 代码执行的局限性。用户发现 aistudio 上的 **code interpreter** 表现更好，但即使在那里，你也必须强迫它去使用。
   - 一位用户对 **Gemini** 的代码执行如此受限感到惊讶（考虑到其价格点），还发现存在多个权限拒绝（permission denied）错误，这些错误可以通过硬刷新（hard refresh）修复。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1384994527827464372)** (336 messages🔥🔥): 

> `HuggingFace Outage, Flux Kontext NSFW, Audio to Video Models, DeepSite Quality Degradation, GUI App for Fine-tuning` 


- **HuggingFace 遭遇停机，用户等待恢复**：用户报告 [HuggingFace 宕机](https://status.huggingface.co/)，影响了模型访问，预计服务将在**传播延迟**后恢复。
- **Flux Kontext 被标记为 NSFW，建议调整 Prompt**：一位用户询问 [Flux Kontext 被标记为 NSFW](https://cdn.discordapp.com/attachments/879548962464493622/1385011482432901302/image.png?ex=6855d575&is=685483f5&hm=75a5e813a5395ce041319cacc4de1af901a7303dc8ff61e105b8c1473d6f4cbc&) 的问题，另一位用户建议版权问题也可能触发 NSFW 标记。
- **GUI 应用让微调更简单**：一位用户为其开发的旨在简化微调的 GUI 应用寻求反馈，并提到该应用目前支持使用 **Unsloth** 进行基础微调。
- **关于 Neurosama 成功因素的推测**：成员们讨论了 Neurosama 的受欢迎程度，将其归功于**抢占市场先机、人类交互以及 Vedal 有趣的内容**。
- **DIY LLM OS 激发创新**：一位用户正在创建一个 LLM OS，将 **Qwen 原生集成到 Linux** 中。
   - 他们正寻求构建自己的强化学习循环，该循环*不涉及硬性数据*，并能自主学习语法采样（grammar sampling）。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1385065630838952077)** (2 messages): 

> `lucidrains/ring-attention-pytorch, Langchain, LangGraph` 


- **思考 Ring Attention 的实现**：一位成员重点介绍了 [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) GitHub 仓库，可能是在探索高效的注意力机制。
- **尝试使用 Langchain 和 LangGraph**：一位成员提到他们暂停下来实验 **Langchain** 和 **LangGraph**，表明正在对这些工具进行实操探索。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1385373505348178040)** (1 messages): 

> `OS Agent, Multi agent system, Message queueing, WebSocket API, computer use agent framework` 


- **OS Agent 更新了新功能**：一位成员在 [GitHub](https://github.com/EnvisionMindCa/OS-Agent) 上更新了他们的 **OS Agent**，增加了 **multi agent system**、**message queueing** 和 **WebSocket API** 等新功能。
   - **OS Agent** 被描述为*一个用于 computer use agent 的极简框架*。
- **OS Agent 拥抱多智能体系统**：更新后的 **OS Agent** 框架现在支持 **multi-agent systems**，能够实现协作任务执行并提升问题解决能力。
   - 这一增强功能允许创建复杂的 Agent，它们可以相互交互和协调以实现复杂目标，从而简化工作流程并提高整体系统性能。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1385084955016695899)** (7 messages): 

> `Inference Credits, Unit 1 Final Quiz, Free Models for Final Assignment, Gemini 2.0 Flash, Delay Execution in CodeAgent` 


- **热衷实验的用户推理额度告急**：一位用户对推理额度（Inference Credits）耗尽表示沮丧，希望能获得用于课程实验的额度。
   - 该请求未得到回应。
- **测验者寻求错误分析**：一位在 Unit 1 最终测验中获得 **90%** 分数的用户询问如何查看自己的错误。
   - 该用户希望从错误中学习，以更全面地理解 Agent。
- **寻求免费工具以完成评分作业**：一位用户询问是否可以使用免费模型（特别是能在基础版 Google Colab 上运行的模型）通过最终作业。
   - 讨论中未推荐具体模型。
- **Gemini 2.0 Flash 免费应对有限功能**：一位用户建议使用 **Gemini 2.0 Flash**，并指出它是免费的但有限制，例如每分钟请求数。
   - 为了避免超时，该用户使用 `time.sleep(10)` 在步骤之间实现了 **10 秒延迟**。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1384993589729431702)** (196 messages🔥🔥): 

> `Flow matching in production, O3 vs Claude Opus, AI NPCs in games, RNNs for combat AI, Mamba vs RNN game inference` 


- **Flow Matching 进入生产流水线**：关于在生产环境中使用 **flow matching (FM)** 的讨论兴起，一些成员引用了 **Imagen**、**Flux** 和 **SDXL3** 作为实现案例。[这篇论文](https://arxiv.org/abs/2403.03206)指出，许多改进源于经验优化，而非更好的数学。
- **O3 Pro 的自主性提升，与 Claude Opus 的严格形成对比**：一位成员发现，如果*手臂抬起角度完美*，**O3** 能更好地编译特定报告，而 **O3 Pro** 提供了更高的自主性，能捕捉更多细节；而 [**Claude 4 Opus**](https://www.anthropic.com/news/claude-opus) 则擅长*完全按指令执行*，就像一个 *Linux 终端*。
- **应对 AI NPC 部署的依赖噩梦**：成员们正致力于解决在游戏中部署**交互式 AI NPC** 的噩梦，重点关注依赖管理以及在消费级硬件上利用 **LibTorch**、**ONNX** 或潜在的 **Vulkan cooperative matrices** 实现实时性能的巨大工程问题。
   - 我能想到的最佳解决方案可能是将 **LibTorch** 编译成自包含的二进制文件或 **ONNX** 之类的。*这是一个巨大的工程问题*。
- **战斗 AI RNN 提供轻量化控制**：讨论集中在利用小型、经 **RL 优化**的 **RNN** 来控制游戏实体，一位成员建议在空闲的 CPU 核心上运行推理，以便在不涉及语音和行为的情况下，以极轻量级的方案优化实体的站位和技能。
   - 关键的权衡是，用户的任何反应都会延迟 5 秒，即使那个新的聊天机器人 NPC 能够找到他们的*真爱*。
- **Mamba 在游戏 AI 中的推理潜力引发辩论**：讨论了 **Mamba** 在推理友好型游戏开发中的潜力，注意到其快速推理和线性扩展特性，并提出了关于**中等规模语言模型**基准测试的问题。
   - 然而，一位成员指出 **Mamba** 本质上就是 **RNN**，特别是考虑到该模型在推理时的计算特性。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1384993328776478821)** (68 条消息🔥🔥): 

> `V-JEPA 2 模型, 批量论文速读 (Bulk Paper Skimming), 论文评估, 研究工程师职位, Energy Matching 论文` 


- **速读达人回归：scilent 回来了！**：在沉寂一段时间后，一名成员重新现身，对主持的讨论表示感谢，并询问大家是否对**批量论文速读 (bulk paper skimming)** 感兴趣，以追赶建议的论文以及特定频道中的内容。
   - 几位成员立即表现出了浓厚的兴趣。
- **论文评估方法：直觉 vs. 图表**：一位成员分享说，他们主要**仅根据图表**来研究论文，而另一位成员在速读时则侧重于**核心思想、标题和副标题**。
   - 第一位成员承认对这种方法感到*羞愧*，第二位成员则表示这不是个好主意。
- **盲读 vs. 准备：主持论文讨论**：一位成员询问了讨论的准备工作，分享了他们投入一篇论文并进行**完全盲读 (full cold read)** 的经验，有时会导致遇到*令人讨厌*的论文或冗长的讨论。
   - 另一位成员更倾向于**提前阅读并理解论文**，以避免浪费大家的时间，并将其作为职业训练的**强制机制 (forcing function)**。
- **数据科学学位困境：为时已晚？**：一位成员咨询是追求**数据科学学士学位**，还是选择**带有应用 AI 方向的计算机科学学位**，并计划攻读 AI 硕士甚至 PhD。
   - 得到的回答是*入门级市场已极度饱和*，更好的路径可能是学习**统计学或应用数学**，并结合计算和 AI 项目。
- **Energy Matching 论文讨论公告**：一位成员宣布了关于 *Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling* [论文](https://arxiv.org/abs/2504.10612) 的讨论日期，并链接了该论文及之前的讨论。
   - 摘要强调了该论文的方法：通过引入熵能量项，赋予基于流 (flow-based) 的方法以 **EBM** 的灵活性。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1384993207372480574)** (21 条消息🔥): 

> `Cursor 新档位, 机器人操作升降桌, Anthropic 使用 AWS 芯片, John Carmack 的蜕变, 思维的错觉 (Illusion of Thinking)` 


- **Cursor 推出新订阅档位**：链接指向一篇 [Cursor 博客文章](https://www.cursor.com/blog/new-tier)，讨论了这款 AI 优先的代码编辑器推出**新订阅档位**的消息。
   - 该公告被迅速分享，凸显了人们对 AI 辅助编程工具日益增长的兴趣。
- **机器人控制升降桌**：一位成员表示，如果机器人能自动操作**升降桌 (standing desk)**，他会印象深刻。
   - 这一评论反映了人们一直希望 AI 能更无缝地处理日常任务。
- **Anthropic 在 AWS 芯片上进行训练**：**Anthropic** 现在正使用 **AWS 芯片**进行训练，标志着其基础设施使用的转变或扩张。
   - 还有人指出，AWS 拥有特定的 **AI 训练专用芯片 (AI training silicon)**，这可能是 Anthropic 做出选择的原因之一。
- **Carmack 的新体格**：一位成员分享了一张 [推文](https://fxtwitter.com/BasedBeffJezos/status/1935588153144017108)，展示了 **John Carmack** 明显的肌肉增长。
   - 另一位成员开玩笑说是**类固醇**的作用，同时思考为什么这位 Doom 的创造者不是一个 **AI 毁灭论者 (AI doomer)**。
- **思维错觉的错觉**：一位成员分享了一张 [推文](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157)，引用了 *The Illusion of the The Illusion of the Illusion of the Illusion of Thinking*。
   - 讨论转向了哲学领域，探讨了 AI 何时才算真正地“思考”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1384979514148524132)** (133 messages🔥🔥): 

> `Gemma 3, 用于微调的 GUI App, Google Gemma x Unsloth 活动, Multi-GPU 支持` 


- **通过修改名称 Hack Unsloth 以微调模型**：用户发现通过更改模型名称，可以使用 **Unsloth notebooks** 来微调其他模型，例如 **Gemma 3**。
   - 一位用户报告成功，并庆祝道：*太棒了，我一直希望它能这样运作*。
- **GUI App 旨在简化微调**：一位成员正在构建一个 GUI App，旨在利用 **Unsloth** 简化微调流程，并征求 UI 反馈，计划在 GitHub 上发布代码。
   - 另一位成员建议：*作为开始，先把所有的白色像素换成深色的（暗黑模式）*。
- **Google Gemma x Unsloth 活动即将在 SF 举行**：Unsloth 将于 **6 月 26 日**在 **SF** 举办 **Google Gemma x Unsloth 活动**。
   - 虽然该活动不会录制，但计划于 10 月中旬在 GitHub 的办公室举办另一场活动。
- **通过变通方法在 Unsloth 中实现 Multi-GPU 支持**：一位用户询问了 Unsloth **Multi-GPU 支持**的预计发布时间（ETA）。
   - 另一位用户提到：*使用 accelerate 已经可以运行了，只是目前还没有官方支持。*
- **LoRA 训练并转换 GGUF**：一位用户想要训练模型并将其转换为 GGUF。
   - 另一位成员表示：*对量化后的 Unsloth 模型使用 load_in_4bit，训练后可以使用 llama.cpp 的转换脚本转换为 GGUF*，并提供了 [Unsloth notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file) 链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

rotta: https://www.youtube.com/watch?v=MGI5-Nm0YLo
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1385003068726444165)** (27 messages🔥): 

> `输入掩码 (Input Masking), Gemma 3 的 GRPO 错误, 量化对持续预训练的影响, GPU KV cache 清理, Llama 3.2 3b Meta 版 vs Unsloth 版` 


- **澄清输入掩码 (Input Masking) 的困惑**：用户参考 [wiki](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs) 询问关于 Unsloth 自动输入掩码的确认，另一位用户澄清 `train_on_responses_only` 对于 **手动输入掩码** 确实是 **必要的**。
   - 虽然并非在所有微调示例 notebook 中都是必需的，但推荐使用 `train_on_responses_only` 作为一种优化手段。
- **Gemma 3 的 GRPO 训练面临兼容性问题**：用户报告在运行 **Gemma 3 的 GRPO** 时出现 `TorchRuntimeError`，即使使用了官方的 [Unsloth notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb)。
   - 一位开发者确认有几个 PR 正在修复此问题，该问题似乎与 **GRPO Trainer 兼容性**有关。
- **持续预训练 (Continued Pretraining) 中的量化疑问**：用户询问在持续预训练 notebook 中使用 **4-bit 量化的 Unsloth 模型** 是否会导致性能下降。
   - 澄清指出，除非使用 `bnb` 或 `sum` 之类的处理，否则量化中没有特殊的终止限制。
- **难以清理 GPU KV Cache**：用户询问如何清理 Unsloth 使用的 **GPU KV cache**。
   - 尽管尝试了 `gc.collect()`、`torch.cuda.empty_cache()` 和 `torch.cuda.ipc_collect()`，用户报告在推理过程中 **GPU 显存占用仍然在增加**。
- **Meta vs Unsloth Llama 3.2 3b 版本之争**：用户询问 Meta 账号下的 **Llama 3.2 3b 模型** 是否与 Hugging Face 上的 Unsloth 版本不同。
   - 澄清结果是两者 **完全相同**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1385276319163482162)** (3 messages): 

> `Gemma, Unsloth, SF 活动` 


- **Google Gemma 和 Unsloth 在 SF 举办派对**：Unsloth 将于 **6 月 26 日**在 **SF** 举办 **Google Gemma x Unsloth 活动**，并通过 [luma.ma 链接](https://lu.ma/gemma-unsloth) 向 Discord 成员开放。
- **成员要求更多线下聚会，特别是 NYC 和 TYO**：成员们强烈要求在 **NYC** 和 **TYO** 举办类似活动。
   - 一位成员说：*我们也想见到你们 :p*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

etherl: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1384973259388489778)** (120 条消息🔥🔥): 

> `MCP server connection, Quantized 70b deepseek model, Base models, Speculative decoding, OpenCode by SST` 


- **LM Studio 准备直接连接 MCP Server**：LM Studio 正在推出直接连接 **MCP servers** 的封闭测试版，这可能会消除对外部应用程序的依赖。
   - 用户可以通过 [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939) 表达兴趣以获取 **MCP beta** 访问权限。
- **追求量化 70B DeepSeek 模型的梦想**：成员们讨论了在 **3060 12GB** 等入门级 GPU 上运行 **量化 70B DeepSeek 模型** 的可行性。
   - 有人建议对于此类硬件，**14B 模型** 会更现实，而且极低位（low bit）模型必须通过庞大的参数数量来弥补 float 精度损失带来的多样性下降。
- **Base 模型表现出无休止的奇怪输出**：成员们解释说，与 instruct 或 chat 模型不同，**base 模型** 会无限期地继续文本生成，而没有问答格式或 EOS token 意识。
   - 共识是，虽然 base 模型会*无休止地延续*，但其输出可能会很*奇怪*。
- **Speculative Decoding 在同架构模型下表现出色**：LM Studio 中的 **Speculative decoding** 在 draft 模型和主模型共享相同架构（如 **Qwen 3 0.5B** 和 **14B**）时效果良好。
   - 然而，据报道它不支持 **vision** 和 **MoE 模型**。
- **OpenCode 用 SST 的开源替代方案取代 ClaudeCode**：用户探索了 **SST 开发的 OpenCode**（[GitHub](https://github.com/sst/opencode?tab=readme-ov-file)）作为 **ClaudeCode** 的开源替代方案。
   - 一位用户分享了一个用于将 LM Studio 与 OpenCode 集成的 [配置文件](https://cdn.discordapp.com/attachments/1110598183144399061/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&)，这需要通过 *opencode auth login* 将 LM Studio 添加到 OpenCode 可使用的模型中。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1384971611970601203)** (14 条消息🔥): 

> `NVLink vs multiple GPUs, VRAM limitations, GPU power consumption considerations, Asus NUC 15 Pro Plus` 


- **NVLink 的成本/效益引发讨论**：一位成员询问 **NVLink** 对于跨 GPU 拆分模型是否物有所值，但另一位成员指出推理过程中的 GPU 间通信很少，因此投资第三块 GPU 是更好的选择。
   - 其他人表示赞同，并指出 *浏览 r/localllama 的信息表明它并不划算*，这很有道理，因为推理通常只涉及极少的 GPU 间通信——仅仅是 ffn 之后的结果。
- **用户抱怨 VRAM 限制**：一位用户对 **24GB VRAM 限制** 表示沮丧，并希望进行扩展，考虑在现有的 **3090** 和 **5950x** 配置中增加一块 **3080 Ti**。
   - 一位成员建议改为购买第二块 **3090**，理由是在 Oobabooga 中使用不同 VRAM 大小的显卡体验不佳，因为*每次都必须手动分配层，因为等分层（equal layer splitting）无法工作*。
- **GPU 功耗和 PSU**：成员们讨论了在 1000W PSU 上，在 **3090** 和 **5950x** (105W TDP) 的基础上增加 **3080 Ti** (350W TDP) 的电力需求。
   - 一位成员建议限制功率（power limiting）可以提供更多余地，并警告说，考虑到主板可能会消耗剩余功率的很大一部分，*功率峰值（power spikes）在剩余电量上会变得非常糟糕*。
- **讨论 NUC 15 Pro Plus 替代方案**：一位成员发布了 [ASUS NUC 15 Pro Plus](https://www.asus.com/my/displays-desktops/nucs/nuc-mini-pcs/asus-nuc-15-pro-plus/techspec/) 的链接，将其作为 GMKtec Evo T1 的同等选择。
   - 他们预测，配备 **96GB RAM** 和 **2TB** 存储的 **准系统（barebones）Evo T1** 的成本应该低于配备 **128GB RAM** 和 **2TB** 存储的 **GMKtec Evo X2**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1384978416448573564)** (108 messages🔥🔥): 

> `Gemini 分享功能, 信息作为可压缩流体, Gemini 中的 Sparse MoE, 原生多模态模型, Gemini 的 Deep Think` 


- **Gemini Share 支持无限思维树 (Tree of Thought)**：Gemini Share 的新功能允许用户点击“Explore”生成解释，点击“Alternatives”产生贡献概念，从而创建一个[永恒的思维树 (eternal tree of thought)](https://gemini.google.com/share/54661b0f8f17)。
   - 一位用户确认也支持[图像生成](https://g.co/gemini/share/4736bebdad6f)，并且需要登录，因为它在服务器端通过你的 oauth 传递 API key。
- **信息在 LLM 中表现得像可压缩流体 (Compressible Fluid)**：一位成员建议将**信息**视为一种*可压缩流体*，在考虑语言在理解和描述深奥概念中的作用时，这具有实验价值。
   - 另一位成员指出，*更多的语言意味着更多的信息*，这就是为什么 **LLMs** 基本上可以使计算语言化，解释含义，甚至通过预测状态追溯步骤。
- **根据新论文，Gemini 可能使用 Sparse MoE**：一位成员根据一篇展示其还原为主要激活特征和包含概念的[新论文](https://www.youtube.com/watch?v=X1gDXDQu_wU)推测，**Gemini** 可能是使用 **sparse MoE** 构建的。
   - 隐藏维度充当*奇点/叠加思维*，是线性表示的思维潜空间 (latent space) 的一部分。
- **Gemini 被描述为原生全模态 (OmniModal) 模型**：一位成员建议 **Gemini** 是一个*世界模型 (world model)*，因为它具有**全模态 (omnimodal)** 输入和生成各种表示（如语言、图像或视频）的不同解码器。
   - 他们声称自 1.0 ultra 以来一直是全模态的，并进一步澄清 [0.5 系列是全模态的] 且 [.0 系列是原始架构]。
- **Gemini 的 'Deep Think' 探索并行路径**：一位用户假设在 Gemini 代码中观察到的奇怪输出可能是由于上下文中的某种形式的持续训练，并引用了 **Gemini's 'Deep Think'**，它可以同时解码并行路径。
   - 正如 [Google AI 宣布](https://x.com/GoogleAI/status/1924886810531901604)的那样，这个在预览版中引入的功能将*并行的叠加状态*线性化。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1385032728063512676)** (8 messages🔥): 

> `Meta 研究, Zuck 合并, 通用世界智能体 (Generalist World Agent)` 


- **Meta 的研究成果丰硕**：成员们讨论了来自 Meta 研究团队的两篇新论文，包括：[https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) 和 [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514)。
   - 一位成员评论了 Meta 的 Llama 团队的悲剧以及 Zuck 合并他们的意图。
- **Zuck 考虑合并团队**：一位成员推测扎克伯格正试图合并团队，保留来自 Yann 等人的视觉思想领导力，同时他转向语言方面，为行业用例构建策略优化 (policy optimization)。
   - 这是由于 Scale 专注于捕获 Agent 将遵循的过程并将其操作化。
- **通用世界智能体 (World Generalist Agents) 即将到来？**：一位成员认为该团队很有可能开发出一种**通用世界智能体 (generalist world agent)**，最终可以进入机器人、计算机或神经接口。
   - 他还链接到了[一条关于“思维错觉的错觉的错觉的错觉”的推文](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1385034755837530225)** (3 messages): 

> `更大的大脑, 前额叶` 


- **关于“更大的大脑”的头脑风暴开始**：一位成员分享了一个关于*更大的大脑 (bigger brains)* 概念及其潜在影响的 [YouTube 视频](https://youtu.be/-G1SdsRXL7k)。
   - 他们提到看了一场 Lex Fridman 和 MLST 的对话，大概与同一话题有关。
- **大脑越多，问题越多**：后续讨论质疑*更大的大脑*是否真的是解决方案。
   - 一位成员提到更大的大脑可能会导致更大的问题。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1385032728063512676)** (8 messages🔥): 

> `Meta Research, Zuckerberg's AI strategy, Generalist world agent` 


- **Meta 发布两篇新论文**：Meta 的研究团队发布了两篇新论文：[https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) 和 [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514)。
- **Zuck 的 AI 霸权宏图**：一位成员推测 Mark Zuckerberg 旨在合并 **Meta 的 Research 和 Llama 团队**，以利用视觉和思想领导力，同时专注于针对行业用例的 Policy Optimization。
- **Meta 瞄准通用型 World Agent**：根据对话，Meta 的目标是开发一种可以集成到机器人、计算机或神经接口中的通用型 World Agent。
   - 该成员还分享了一个 [推文](https://fxtwitter.com/rohanpaul_ai/status/1935047424948781098?t=HMmgUOtz-nwBdgcd7cTXOw&s=19) 和另一个 [推文](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157)，内容涉及 *思维的错觉 (the illusion of thinking)*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1384973684757762058)** (110 messages🔥🔥): 

> `Claude 3.7, MiniMax-M1, Free 1M context model, Free Gemini version, Glazing models` 


- **昂贵的 Claude 导致 Token 再平衡困扰**：由于 Claude 2.5 预览版与正式版之间的 **Input 成本** 翻倍，用户正在重新平衡 **Output vs Input Tokens**，这影响了高频使用场景。
- **免费版 Gemini 消失，Gemini 2.0 Flash 现身**：由于 Hugging Face 上的免费版 **Gemini** 是由 Google 提供的，目前已无法使用，但 [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-exp:free) 上提供了一个具有 1M Context 的免费 **Gemini 2.0 Flash** 模型。
- **DeepInfra 部署折扣价 Gemini**：[DeepInfra](https://deepinfra.com/google/gemini-2.5-pro) 正在其自有硬件上提供 **Google Gemini 2.5 Pro/Flash**，价格低于 Google 官方，但这很可能是通过协商好的云服务商价格对 Google API 进行的代理。
- **推荐将 Deepseek R1 0528 用于编程**：成员们推荐新的 **Deepseek R1 0528** 作为优秀的编程模型，特别是与 0324 等旧版本不同，它是一个 Thinking Model，因此更适合处理代码。
   - 有报告称 **0528** 版本 *不支持 Prefill*，尽管这一说法随后被撤回。
- **OpenRouter 惊人的经济效益**：OpenRouter 的经济数据令人印象深刻，单日处理的 **Claude Sonnet 4** 使用额度约为 **$12.6 万**。
   - 一位成员将 OR 比作 AI 界的 Mastercard/VISA，并指出其增长和普及程度非常惊人且实至名归，尽管他们仅赚取约 **5%** 的费用。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1384971021261738116)** (70 条消息🔥🔥): 

> `Manus 视频生成, AI 错误与额度消耗, Manus 失败反馈循环, Manus Fellow 项目面试状态, 技术债` 


- **Manus 视频生成**：成员们对 **Manus 视频生成** 不是免费的表示失望，但理解由于计算能力（compute power）的原因其成本很高，而另一些人则表示用户仍需投入一些工作才能使其正常运行。
- **AI 错误在没有结果的情况下消耗额度**：成员们讨论了 **Manus** 有时会遇到无法修复的错误，在未完成任务的情况下耗尽了额度，一位成员将其比作“因为厨师付出了劳动而不得不为一个烂汉堡买单”。
   - 另一位成员指出，即使出现错误，AI 仍在使用计算能力并产生费用，但在 **额度被烧掉却没得到可用输出** 时令人沮丧，并建议将收费阈值设定为 **80%** 的定义成功标准，并链接到了 [这个 YouTube 视频](https://m.youtube.com/watch?v=5PuofaVqXNI)。
- **Manus 奖励了一个破碎的反馈循环**：一位成员表示，真正的问题在于系统的“静默失败（silent failure）”，即无法识别自身已失败，在没有实际成功或内部意识的情况下消耗额度。
   - 他们询问了修复这种破碎奖励模型的具体行动，以及为什么在未达到成功标准时仍要扣除额度，还提到了 **损失了 70,000 积分**。
- **Fellow 申请者想知道录取状态**：一位在六周多前完成了 **Manus Fellow 项目面试** 的成员仍在等待录取或拒绝通知，并寻求一个简单的状态更新。
   - 他们强调，未解决的问题可能会导致信任危机，并寻求一个具体的、有日期的行动计划。
- **技术债与小错误的累积**：一位成员强调，计算过程中小错误的累积可能导致意外失败，即使单个错误看起来微不足道。
   - 另一位成员表示赞同，指出衡量分歧（divergence）虽然有用但并非万无一失，他们经常从各种 AI 平台获得幻觉（hallucinated）结果，且 **额度不予退还**。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1385035406981988404)** (9 条消息🔥): 

> `Mojo 衬衫, EmberJson, simdjson, Python 实现` 


- **通过 LinkedIn 而非 X 获取 Mojo 衬衫**：一位成员询问是否可以通过在 **LinkedIn** 而非 **X** (Twitter) 上分享来获取 **Mojo 衬衫**，因为 LinkedIn 的触达效果更好。
- **EmberJson 的作者确认**：一位成员询问了 **EmberJson** 库的作者，另一位成员确认自己就是该库的开发者。
   - 他们正在等待“语言层面的进展（language developments）”，然后再深入进行更多优化。
- **EmberJson 与 simdjson 的性能对比**：一位成员询问了 **EmberJson** 与 **simdjson** 或 **zimdjson** 的性能对比。
   - EmberJson 的作者提到它仍远低于这些库，根据 CPU 和数据的不同，估计在 **200-500 MB/s** 左右，但正在等待进一步的语言进展后再进行优化。
- **EmberJson 与 Python 实现的对比**：一位成员询问 **EmberJson** 是否能与 **Python 实现** 相媲美。
   - 作者回应称，在有限的测试中，它通常比 Python 标准库（stdlib）快约 **2 倍**。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1385002991714570260)** (46 messages🔥): 

> `Mojo crashes, MAX supports Blackwell, SymPy in Mojo, Claude Code and Mojo` 


- **Mojo 崩溃引发 Bug 报告讨论**：一名成员报告了 **segmentation fault**，并询问对于 Mojo 运行时错误是否有必要提交 Bug 报告。
   - 另一名成员回复称，该崩溃看起来像是 **stdlib 或编译器问题**，并链接到了 [GitHub 上的 issue #4857](https://github.com/modular/modular/issues/4857)。
- **Modular 悄然支持 Blackwell**：一位核心开发者提到 **MAX** 已经支持 **Blackwell** GPU，但目前尚未广泛宣传。
   - 他们鼓励拥有 **5090** 系统的用户进行测试并提供反馈，并指出在正式发布前还需要进行更多的 **perf** 和其他优化工作。
- **在 Mojo 中实现 SymPy 的痛苦历程**：一名成员询问在 **Mojo** 中实现类似 **SymPy** 的功能是否可行。
   - 另一名成员回答说这应该是可能的，但过程会充满“巨大的痛苦和折磨”。
- **Claude 辅助 Mojo 绘制 Matmul 架构图及 CUDA 转换**：一位成员分享称，现代的 **agentic systems**，特别是 **Claude Code**，其能力令人惊叹。
   - 在提供现代 **Mojo** 和 **MAX** 正确上下文（**modular repo**、**Modular docs**）的情况下，**Claude Code** 能够 **one-shot** 完成大量任务：*绘制 MAX 内部 matmul 操作的架构专门化图表、创建一个可从 Python 调用并利用 SIMD 高效分解大数的 Mojo 函数、将 CUDA 参考 kernel 翻译为 Mojo 等等*。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1385198997958033470)** (12 messages🔥): 

> `Model Compilation Failures, RDNA4 GPU Support, CI Testing for Max Models, Error Message Improvements` 


- **模型编译失败困扰 Max 用户**：一位用户报告称，他们尝试部署的每个 **Max model** 在 **GPU** 和 **CPU** 上都无法编译。
   - 该用户建议增加一个类似于 [rust-lang/crater](https://github.com/rust-lang/crater) 的 **CI 步骤**，以防止 PR 破坏已托管的 Max 模型。
- **RDNA4 GPU 属于 Tier 3 兼容**：团队本周才刚刚开启对 **RDNA4 GPU**（如 **9000 系列**）的基础支持，但完整的模型尚无法在其上运行。
   - 在模型能够完全运行之前，**9000 系列** GPU 被归类为 *Tier 3: Limited compatibility*（有限兼容）。
- **计划改进错误消息**：团队承认错误消息需要改进，因为目前的约束错误消息不够清晰。
   - 用户遇到这些错误是因为并非所有的 **kernels** 都已适配 **RDNA4 架构**。
- **分享 GPU 需求文档**：为了解决这一问题，Max 团队提供了 [system specs](https://docs.modular.com/max/faq#system-requirements) 和 [compatible GPUs](https://docs.modular.com/max/faq#gpu-requirements) 的文档页面。
   - 原始报告者使用的 RDNA4 9070 目前尚未得到完全支持。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1384992682832564295)** (51 条消息🔥): 

> `Midjourney 视频模型，CoreWeave 与 Weights & Biases AI 推理，Meta 招聘 Nat Friedman 与 Dan Gross，Profound A 轮融资，Arcee AI AFM-4.5B-Preview 模型` 


- **Midjourney 通过 Video Model V1 实现动画化**：Midjourney 发布了其 **Video Model** 的 **Version 1**，允许用户为 **Midjourney 生成的**或**外部图像**制作动画，提供“自动”和“手动”动画设置选项，价格约为图像任务的 **8 倍**，发布时可在 Web 端使用，详见 [X](https://x.com/midjourney/status/1935377193733079452)。
   - 新的“图像转视频（Image-to-Video）”功能提供“高动态”和“低动态”选项，视频可以延长，定价可能会为了可持续性以及为未来图像模型积累见解而进行调整。
- **CoreWeave 和 W&B 推出 AI 推理服务**：CoreWeave 和 Weights & Biases 推出了新的 AI 推理服务，包括针对 **DeepSeek R1-0528** 和 **LLama-4 Scout** 等模型的推理端点，具有兼容 OAI 的 API，以及在线评估工具，参考[此推文](https://x.com/altryne/status/1935412384283107572)。
   - 这些运行在 CoreWeave GPU 上的服务旨在 AI 基础设施领域提供更多竞争和灵活性，提供实时的 LLM 评测。
- **Meta 招揽 Nat Friedman 和 Dan Gross**：据 [money.usnews.com](https://money.usnews.com/investing/news/articles/2025-06-18/meta-in-talks-to-hire-former-github-ceo-nat-friedman-to-join-ai-efforts-the-information-reports) 报道，Meta 据传正在洽谈聘请前 GitHub CEO **Nat Friedman** 和 AI 科学家 **Dan Gross**，以加强其 AI 实力。
   - 反应从对他们向 Alexandr Wang 汇报的怀疑，到认为他们向 **Alexandr Wang** 汇报是不可能的。
- **Profound 获得 A 轮融资**：由 James Cadwallader 和 Dylan Babbs 领导的 **Profound** 获得了 **A 轮**融资，强调了他们在不断演变的搜索格局中的作用，**SagaVC** 也参与了共同投资，如[此贴](https://www.stories.sagavc.com/posts/profound)所述。
   - 线程中的讨论质疑了 Profound 在后搜索优化时代衡量和提出建议的方法。
- **Arcee AI 首次推出 AFM-4.5B-Preview 模型**：Arcee AI 发布了其新的基础模型 **AFM-4.5B-Preview**，专为企业使用设计，参数量低于 **10B**，优先考虑效率和监管合规性，是与 DatologyAI 合作开发的，公告见[此处](https://x.com/lucasatkins7/status/1935382123155964081?s=46)。
   - 该模型利用了 **MergeKit** 和 **YaRN** 等先进技术，并计划在 7 月初公开释放 **AFM-4.5B** 及其基座模型，同时开源之前封闭的模型如 Virtuoso-Large。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1385316378826772542)** (2 条消息): 

> `` 


- **占位主题 1**：这是一个占位摘要，以满足最小项目要求。
   - 此处可以添加关于占位主题的更多详细信息。
- **占位主题 2**：这是另一个满足验证标准的占位摘要。
   - 此处对第二个占位主题进行进一步阐述。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1385140703813832735)** (1 条消息): 

> `Triton 教程，Deep-spin 实验室` 


- **Deep-spin 实验室发布 Triton 教程**：该 **Triton 教程**通过幻灯片涵盖了基础知识，并进行了动手实践，从**向量加法**开始，到 **sparsemax(QK^T)V** 结束。
   - 该[教程](https://github.com/deep-spin/triton-tutorial)是为实验室创建的，但也可能对其他人有所帮助。
- **Triton 向量加法示例**：教程从**向量加法**的动手示例开始，介绍 Triton 的基础知识。
   - 它逐步过渡到更复杂的运算，如 **sparsemax(QK^T)V**，展示了实际应用。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1385267675491467375)** (6 messages): 

> `cuda-gdb, cudaErrorUnsupportedPtxVersion, Nvidia Driver Versions` 


- **用户遇到 CUDA 调试错误**：一位用户在利用 **cuda-gdb** 时遇到了 `cudaErrorUnsupportedPtxVersion` 错误，并认为他们需要升级 GPU 驱动。
   - 该错误表明 CUDA Toolkit 版本与当前驱动程序不兼容，需要通过更新驱动来解决此问题。
- **寻找最新的 Nvidia 驱动程序**：一位用户询问如何找到最新的兼容 Nvidia 驱动版本。
   - 另一位成员链接了 [这份 Nvidia 文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id7)，其中显示了每个 **CUDA Toolkit 版本** 附带的驱动程序版本，并建议将其作为参考。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1385262254789754910)** (1 messages): 

> `Big Model Serving, Parallelism Techniques, AI Infra Frameworks, vLLM & Kubernetes, Nvidia Dynamo/Triton` 


- **深入探讨大模型推理服务技术**：讨论围绕大公司在大型基础设施上提供大模型服务所采用的并行技术展开。
   - 旨在寻求除了 **Accelerate** 和 **DeepSpeed** 等专注于训练的库之外，用于 AI 基础设施的热门框架的见解。
- **vLLM、Kubernetes 与最佳实践**：对话质疑将 **vLLM** 与 **Kubernetes** 集成是否符合模型服务的最佳实践。
   - 讨论强调 **vLLM** 是一个热门选择，尤其是在推理方面，并旨在了解其最佳部署策略。
- **Nvidia 的 Dynamo：更名后的 Triton？**：讨论还质疑了 **Nvidia Dynamo**（前身为 **Triton**）的使用及其在模型服务中的普及程度。
   - 讨论承认了 **Triton** 在推理领域的历史地位，并探讨了其在新的 **Dynamo** 品牌下的现状。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1385166884898471998)** (2 messages): 

> `LLMs, AusysAI, LLM Abstraction Levels` 


- **AusysAI 以通俗易懂的方式解释 LLM**：AusysAI 发布了一篇 [博客文章](https://www.ausysai.com/posts/explaining-how-llms-work-7-levels-of-abstraction)，以直观的方式解释了 **LLM** 的工作原理。
   - 该文章通过 **7 个抽象层级**，既可以作为新手的入门读物，也可以作为从业者对基础知识的回顾。
- **解码 LLM 的七个抽象层级**：AusysAI 博客通过 **七个抽象层级** 剖析了 **Large Language Models** (LLM)。
   - 旨在面向寻求基础理解的新手和需要温故知新的资深从业者。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1385082419358601277)** (1 messages): 

> `ADAS Platform Software Engineer, Lucid Motors, GPU background` 


- **Lucid Motors ADAS 团队寻求 GPU 专家**：Lucid Motors 的 ADAS 平台软件团队正在招聘一名具有 **GPU** 专业知识和 **Linux/QNX** 经验的 **高级软件工程师**；鼓励应聘者在申请中提及 **Arun Paruchuri**，并附上了 [职位发布链接](https://job-boards.greenhouse.io/lucidmotors/jobs/4700944007)。
   - 一名团队成员表示他们最近刚加入，并对团队的工作感到满意。
- **Arun Paruchuri 加入 Lucid Motors ADAS 团队**：Arun Paruchuri 最近加入了 Lucid Motors 的 ADAS 团队，并且非常享受这份工作。
   - 他鼓励申请 **高级软件工程师** 职位的候选人提及他的名字。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1384973778244604036)** (2 messages): 

> `Distributed Training Course, Unsloth Meetup with Google DeepMind Gemma` 


- **提供分布式训练课程**：一位朋友正在教授一门关于分布式训练的课程，并邀请其他人加入，提到他是 transformers 中 *accelerate* 的维护者。
   - 该课程承诺向大咖学习。[在此注册](https://maven.com/walk-with-code/scratch-to-scale?promoCode=matej26)。
- **Unsloth 在旧金山举办 Gemma 见面会**：旧金山将举办一场与 Google DeepMind Gemma 团队的见面会，内容包括关于 **GRPO** 和 **kernels** 的演讲。
   - 他们正在征集关于 **kernels** 和 **开源 AI** 的 3 分钟闪电演讲。[在此预约 (RSVP)](https://lu.ma/gemma-unsloth)。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1385199369283702864)** (13 messages🔥): 

> `ModuleNotFoundError agents 修复，PR 评审，CI/CD 流水线` 


- **点号前缀规避 ModuleNotFoundError**：一名成员通过使用 **相对导入** (`.agents.basic_agent`) 代替 **绝对导入** (`agents.basic_agent`) 解决了 `ModuleNotFoundError`。
   - 该成员确认使用 **相对导入** 解决了他们的导入错误，此前该错误需要手动设置 `PYTHONPATH` 环境变量。
- **请求 PR 评审**：一名成员在处理完评论并对项目做出贡献后，请求对其 [pull request](https://github.com/JackHopkins/factorio-learning-environment/pull/228) 进行评审。
   - 另一名成员确认他们已经评审了该 PR，且 **没有阻碍合并的评论**。
- **讨论实施 CI/CD 流水线**：团队讨论了实施 **CI/CD 流水线** 以确保在合并更改前通过测试，同时解决了访问权限问题并重构了代码库。
   - 对话还涉及了使用 **Factorio 的回放文件** 训练 Agent 的潜力，包括将回放数据反序列化为 JSON 的技术细节。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1385020093599322163)** (14 messages🔥): 

> `CUTLASS 示例，CuTe 索引错误，TensorSSA 赋值限制，向量化 relu 核函数，CuTe 中的动态范围` 


- **CuTe 索引错误困扰新手**：一名用户在尝试使用 CuTe 实现 `vectorized_relu_kernel` 时遇到了索引错误，具体与 `!cute.layout` 和 `!cute.coord` 之间的不兼容有关，如其 [屏幕截图](https://cdn.discordapp.com/attachments/1362196854460383353/1385183408354885712/Screenshot_2025-06-19_at_2.34.04_PM.png?ex=6855ccd4&is=68547b54&hm=75b031aceece5b4d12addb0e069f282ce524a28138dd90ba1b518dc37654c0aa&) 所示。
   - 错误信息 *unable to compute crd2idx with* `!cute.layout<"((1,8)):((0,1))">` *and* `!cute.coord<"(0,0)">` 表明 Tensor Layout 与用于索引的坐标之间不匹配。
- **CuTe DSL TensorSSA 的不可变性令开发者苦恼**：一名用户发现 CuTe 中的 `TensorSSA` 值是不可变的，由于 `x` 是临时值而非可变缓冲区，因此无法进行类似 `x[(0, i)] = max(0, x[(0, i)]).to(x.dtype)` 的直接赋值。
   - 建议的权宜之计包括使用 `cute.where` 进行逐元素操作，或为寄存器内存 Tensor 使用 `cute.make_fragment_like` 以实现赋值，详见 [CuTe DSL 限制文档](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/limitations.html)。
- **动态范围难倒寄存器开发者**：一名用户询问如何从动态大小的列表创建 Tensor，但在 CuTe DSL 中遇到了动态范围的限制。
   - 官方澄清，虽然可以跟踪 JIT 时间已知的静态范围来填充 Tensor，但目前不支持动态范围和具有动态长度的 Python 数据结构，并且不允许对 `y` 进行动态索引，如其 [屏幕截图](https://cdn.discordapp.com/attachments/1362196854460383353/1385189040378085386/Screenshot_2025-06-19_at_2.56.15_PM.png?ex=6855d212&is=68548092&hm=72f0b5575c09e92aead884ea5f7946a0b86eb161ec1769074bd8d433310af2be&) 所示。
- **Cutlass 的 CuTe 逐元素加法示例为工程师提供指导**：一名用户被引导至 CUTLASS 示例中的 [elementwise_add.ipynb](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb) 笔记本，以获取 CuTe DSL 中逐元素操作的指导。
   - 该示例演示了基础的加法操作，展示了如何定义和启动用于逐元素 Tensor 加法的 Kernel，为理解更复杂的业务奠定了基础。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1384972270044315750)** (27 messages🔥): 

> `Gemini 2.5 Pro 配置, Aider 编辑模式问题, OpenRouter 上的 Deepseek 免费版, GitHub Copilot 投诉, 自定义基准测试` 


- **Gemini 配置调整优化 Aider 性能**：成员们讨论了手动向 `.aider.model.settings.yml` 添加配置，例如为 **Gemini 2.5 Pro preview** 添加 `thinking_tokens` 以避免警告，或者使用命令 `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k --edit-format diff-fenced` 作为替代方案。
   - 有人指出，带有 32k `thinking_tokens` 的 **0605 版本**在编程方面表现出色，但在聊天方面表现欠佳；此外，在不使用 `thinking_tokens` 时，Gemini 2.5 的价格比预览版贵 **4 倍**。
- **Aider 编辑模式导致项目混乱**：一位用户报告称，在 Claude 模型中使用 Aider 的编辑模式会导致诸如**非预期的全应用更改**、**代码追加**以及 **CSS 类错误**（如在未声明的情况下添加 `border.boder`）等问题。
   - 另一位用户询问如何在不重启聊天的情况下更改编辑格式，得到的回答是使用 `/chat-mode diff-fenced`。
- **OpenRouter 上的 Deepseek 免费版陷入死循环**：一位成员报告称 **OpenRouter 上的 Deepseek 免费版**出现问题，陷入死循环并反复提交相同的文件更改。
   - 将 `edit-format` 设置为 `whole` 提供了一个临时解决方案，但这可能仅仅是因为开启了实验性的缓存（caching）起到了作用。
- **GitHub Copilot 用户抱怨 Claude Sonnet 限制**：据报道，r/githubcopilot 子版块的用户正在抱怨每月支付 10 美元却只能获得 **300 次 Claude Sonnet 调用**（80k 上下文限制），尽管他们可以获得无限次的工具调用以及 GPT-4.1/4o。
   - 同时也暗示 Deepseek 和其他类似工具是完全免费的。
- **自定义基准测试显示 Llama 模型表现不佳**：一位成员创建了一个自定义基准测试，并指出 **Llama 模型表现不佳**；消息中附带了基准测试图像：[image.png](https://cdn.discordapp.com/attachments/1131200896827654149/1385357720286138381/image.png?ex=6855c66b&is=685474eb&hm=a2f92d5cbb4abede7876d489911310283847b1e3cf50e89546d0142f81068a76&)。
   - 该基准测试被描述为使用谜语和代号挑战的 **single-shot 测试**，有人询问了所用语言或多轮测试（multi-pass）的细节。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1384986986040918170)** (11 messages🔥): 

> `Gemini 2.5 Flash, Deepseek v3, --watch-files 与 Jupyter notebooks, Aider 重新添加已删除的代码, MERN 项目` 


- **使用 Gemini 2.5 Flash 进行复制粘贴式编辑**：一位成员在 `whole` 模式下使用 **Gemini 2.5 Flash** 作为编辑器进行复制粘贴，但担心使用 `deepseek v3` 作为 **Gemini 2.5 Pro** 的编辑器。
   - 他们计划*深入研究编辑器的决策流程，以及它在多大程度上会降低主模型的智能水平*。
- **Jupyter Notebooks 中的 `--watch-files` 触发器**：在 Jupyter Notebooks 中使用 `--watch-files` 命令时，需要触发词 **AI** 位于注释的开头，例如 `AI! fix this`。
   - 如果触发词位于末尾（如 `# this fails AI!`）则无法工作，因为在 JSON 中行尾以 `",` 结束，导致末尾的 AI 无法匹配。
- **Aider 追加代码错误报告**：在使用 Aider 的编辑模式时，一位成员报告称*它没有更改目标文件，而是开始更改整个应用程序*。
   - 报告的其他错误还包括 Aider *追加已经编写过的代码，例如重复导入 React 两次*。
- **Aider 持续添加已删除的代码**：一位成员正在寻求建议，如何阻止 Aider 重新引入已被刻意删除的代码，特别是与创建不需要的列相关的 **pandas** 代码。
   - 一位成员建议*尝试限制文件范围。有时你必须丢弃错误的更改，可以使用 /undo*。
- **Aider 中的 MERN 项目**：一位成员正在 Aider 的帮助下开发一个**全栈网站**项目，主要是 **MERN** 项目。
   - 他们正在通过与聊天机器人交互来生成和编辑代码。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1385032841015988344)** (21 messages🔥): 

> `MCP Server Setup, Claude 2025-06-18 spec support, Loading MCP Tools, MCP SDK for Go, FastMCP Errors` 


- **MCP Server 设置简化**：用户讨论了让用户使用在 Docker 上运行的新创建 **MCP server** 的最简单方法，建议从 Google Cloud Console 获取 *credentials.json*。
   - 对话还涉及了新的 **Claude 版本**是否会支持 **2025-06-18 MCP 规范**。
- **无需 Client Session 加载 MCP 工具**：一位成员询问如何在不创建客户端会话的情况下加载 **MCP tools**，并提到他们在类似场景下使用 **OpenAI agents** 取得了成功。
   - 该用户拥有一个本地 **MCP server**，它将 MCP 会话作为参数并加载工具。
- **缺少 Go 语言的 MCP SDK**：用户注意到官方缺少 **MCP SDK for Go**，并寻求现有实现的建议。
   - 有建议指向 [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go)，认为这是一个很有前景的 **Go 实现**。
- **FastMCP 'host' 错误困扰用户**：一位用户在使用 **FastMCP** 时遇到了 **TypeError**，具体表现为尽管文档中存在该参数，但仍出现意外的关键字参数 *'host'*。
   - 该用户使用 `uv run server.py` 运行其服务器代码，并在调用 `mcp.run()` 时收到错误。
- **独立开发者 Base44 出售给 Wix！**：TechCrunch 的一篇文章链接 [TechCrunch](https://techcrunch.com/2025/06/18/6-month-old-solo-owned-vibe-coder-base44-sells-to-wix-for-80m-cash/) 显示，成立仅 6 个月的独立开发者项目 **Base44** 以 **8000 万美元**现金出售给了 **Wix**。
   - 一位用户在分享链接时发布了 *🤯* 表情。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1385093104914141205)** (11 messages🔥): 

> `Windsurf Configuration Issues, Enact Protocol for Tool Registry, mcp-webcam Updates, Muppet Kit Devtool, Dagger Container MCP` 


- **Windsurf 配置难倒用户！**：一位用户报告在多次尝试安装 Node.js 等依赖项后，仍难以配置 **Windsurf** 以访问来自 OpenAI 的 humanizer AI sub GPT。
   - 讨论中未提供解决方案。
- **Enact Protocol 扩展 MCP 工具生态！**：一位用户请求对 [**Enact Protocol**](https://enactprotocol.com/) 提供反馈，该协议被描述为用于工具注册表的 MCP 工具定义的扩展。
   - 消息中未提供反馈。
- **mcp-webcam 增加流媒体支持！**：**mcp-webcam** 项目现在支持 **Streamable HTTP**，具有多用户模式，并提供更简单的采样请求，[代码库已在 GitHub 上发布](https://github.com/evalstate/mcp-webcam)。
   - 该功能已内置于 **VSCode v1.101.0** 和 **fast-agent** 中，可通过 MCP 连接 URL 访问，并可通过 `npx @llmindset/mcp-webcam` 在本地运行。
- **Muppet Kit 调试 MCP Server！**：**Muppet Kit** 是一个用于测试和调试 MCP server 的开发工具，目前正趋于稳定，[GitHub 代码库](https://github.com/muppet-dev/kit)已上线。
   - 功能包括资源管理器（Explorer）、演练场（Playground）、MCP 扫描（Scan）、追踪（Tracing）和历史记录（History），可通过 `npx muppet-kit inspector` 访问，更多信息见 [推文](https://x.com/MathurAditya7/status/1923719099961479246)。
- **Dagger 容器 MCP 问世！**：分享了一个关于 **Dagger** 容器 MCP 的博客文章链接，引用了一个假设的未来博客文章 [block.github.io](https://block.github.io/goose/blog/2025/06/19/isolated-development-environments/)。
   - 标题为 *Isolated Development Environments*（隔离的开发环境）。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1384978956570071090)** (2 messages): 

> `MCP vs Vector Search, Agent Memory, Memory Blocks, Enterprise data` 


- **MCP 不会取代向量搜索**：尽管 Agent 通过 **MCP 协议**直接连接到数据源有了新的可能性，但对于非结构化数据，预处理和索引仍然是必要的，因为 90% 的企业数据存在于 **PDF**、**PPT** 和网页中 [(LlamaIndex 的推文)](https://twitter.com/llama_index/status/1935419760898093435)。
- **LlamaIndex 为 Agent Memory 引入 Memory Blocks**：最近，**LlamaIndex** 开始向 **LlamaIndex** 引入灵活的 **Memory Blocks**（记忆块），以满足 Agent 记忆的不同用途 [(LlamaIndex 的推文)](https://twitter.com/llama_index/status/1935774624257843217)。
- **关于 Agent Memory 的 Memory Blocks 直播**：下周将举行关于 Memory Blocks 的直播，详情将很快公布 [(LlamaIndex 的推文)](https://twitter.com/llama_index/status/1935774624257843217)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1385282164610044004)** (29 messages🔥): 

> `Unit Testing LlamaTS, Token Counting with Gemini, LLM Client Access, Custom LLM Class, Python Type Safety` 


- **单元测试因 ES Module 问题失败**：一位成员报告在使用 **Mocha** 或 **Jest** 为 **LlamaTS** 编写单元测试时，由于 **ES module 问题**遇到了障碍。
   - 他们正在寻求关于 **AI 项目**通用单元测试运行方式的建议。
- **Gemini Token 计数相关讨论**：一位成员询问了通过 **LlamaIndex** 为 **Vertex/Gemini** 进行 Token 计数的示例，并指出默认的 **tiktoken** tokenizer 无法在 **Gemini** 上运行。
   - 他们引用了 [Google 关于 Token 计数的文档](https://ai.google.dev/gemini-api/docs/tokens?lang=python) 并分享了一段可能的代码片段，但在 Client 定义上遇到了问题。
- **关于访问 LLM Client 的辩论**：社区成员讨论了如何从 **LlamaIndex** 的 **LLM** 封装器中访问底层 Client 对象，以便执行 Token 计数等自定义操作。
   - 讨论了使用带下划线的属性（例如 `llm._client`）的可能性，以及在 `llama_index.core.llms.llm.LLM` 中添加 `get_client()` 方法的想法，同时对 [type safety](https://mypy.readthedocs.io/) 表达了一些担忧。
- **考虑自定义 LLM 类**：为了满足自定义 Token 计数的需要，成员们考虑将 `llama_index.core.llms.llm.LLM` 封装在一个自定义 LLM 类中。
   - 共识似乎倾向于这种方法，因为修改所有现有的 LLM 集成并不现实，尽管这被认为是一个较低优先级的事项。
- **Python 类型系统问题**：一位成员在尝试向 `TokenCounter` 传递 tokenizer 函数时，对 Python 的类型系统感到沮丧。
   - 尽管提供了一个有效的函数，但由于 `TokenCounter` 期望的函数也可能为 None，导致他们遇到了类型错误。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1385209476268953711)** (7 messages): 

> `NBLM Portraits Digital Avatar, NBLM personalized voice and design, NBLM Video Feature, NBLM Audio Length` 


- **NBLM 用户对 Portraits 数字分身非常兴奋**：一位用户对 **NBLM** 的 **Portraits** 功能表示兴奋，将其视为可以作为产品使用或与客户及团队共享的数字分身（Digital Avatar），并分享了 [Google Labs Portraits](https://labs.google/portraits/login/kimscott) 的链接。
   - 该用户渴望获得个性化的 **voice**（语音）、**design**（设计）和 **interface**（界面）选项，计划通过加载特定的客户信息，将 Portraits 作为新业务的价值主张。
- **用户询问 NBLM 的视频功能**：一位用户询问了 NotebookLM 推出 **video feature**（视频功能）的时间表。
   - 目前没有提供相关信息。
- **NBLM 生成的音频长度较短**：一位用户注意到，在 **Dutch**（荷兰语）中使用相同的 Prompt 时，NotebookLM 会生成 **8 分钟的音频**，而在其他语言中可能会更短，如[此截图](https://cdn.discordapp.com/attachments/1124403655819415592/1385249546225061908/Screenshot_2025-06-19_at_15.25.28.png?ex=685561ac&is=6854102c&hm=be028a00040ebbbc8801a4b66215a66f3643d8091cc4e9263ff1ee6015750cbd)所示。
- **合并来源可产生更长的音频**：一位用户意识到，针对某个主题合并多个来源会产生更长的音频。
   - 另一位用户询问该行为是否仅限于付费版本。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1384979038313119987)** (13 messages🔥): 

> `Audio overviews in non-English languages, AI Agents in NotebookLM, Public Notebook library, NotebookLM access issues, NotebookLM sharing with large audiences` 


- **非英语音频概览遇到困难**：一位用户报告在生成超过 10 分钟的意大利语音频概览时遇到问题，并指出即使是自定义 Prompt 也没有帮助，[另一位用户确认](https://discord.com/channels/1124402182171672732/1366873891938504827/1366873891938504827)这是非英语语言的已知问题。
- **NotebookLM Agents：极客深度研究**：一位用户建议在 NotebookLM 中创建 AI "**Agents**"，针对数学、物理、生物或化学等特定知识领域进行预训练和优化，以提高准确性和可靠性。
- **NotebookLM 访问问题？无法进入网站**：一位用户报告无法访问 NotebookLM 网站，只看到一条 *"can't enter the site"*（无法进入网站）的消息。
- **NotebookLM 社交化：公共 Notebook 库**：一位用户询问是否存在公共 Notebook 库，以便浏览他人构建并想要分享的内容。
- **NotebookLM：大规模共享应选择 Plus 还是 Enterprise**：一位用户询问 NotebookLM Plus 订阅是否足以与 200 多人共享 Notebook，还是需要 Enterprise 计划。


  

---

### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1384974074895401022)** (15 条消息🔥): 

> `新 AI R&D 频道, Embed v4 的 EU GDPR 合规性, Cohere 项目贡献, Cohere 4 AI` 


- **AI R&D 频道上线**：创建了一个专门用于 **AI 研究与开发** 的新频道：<#1384974112841269399>。
- **Embed v4 的 GDPR 合规性**：一名成员询问了 **Embed v4** 的 **EU GDPR 合规性**。
   - 由于该模型在 **多模态 RAG 文档** 方面的卓越表现，他们正在等待 Cohere 团队的回复，以明确这是否在路线图中。
- **新成员询问如何贡献**：一名新成员询问如何加入并为现有的 **Cohere 项目** 做出贡献。
   - 一位热心成员建议关注 **Cohere 4 AI** 并分享了 [申请链接](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw)，并建议在新的 <#1384974112841269399> 频道中分享他们的研究。
- **GDPR 问题应咨询 Support**：一名成员询问了 **Embed v4** 的 **EU GDPR 合规性**。
   - Cohere 团队的一名成员要求将该问题通过电子邮件发送至 [support@cohere.com](mailto:support@cohere.com)。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1385001218979663913)** (3 条消息): 

> `志愿者机会, Cohere AI 项目` 


- **用户寻求志愿者机会**：一名成员介绍了自己，并表示有兴趣在社区内寻找志愿者机会。
- **申请 Cohere AI 项目**：一名成员建议，如果用户申请了 **Cohere AI 项目**，他们将收到一封电子邮件，告知可用的研究机会和项目。


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1385102143819874304)** (1 条消息): 

> `AI 研究, 安全机器学习, 隐私保护, AI 驱动的网络安全, 计算机视觉与 NLP` 


- **新人 Yasir Khan 加入 Cohere Labs Open Science**：Yasir Khan 是 Cohere Labs 开源科学社区的新成员，他表示有兴趣以志愿者身份合作开展 AI 研究项目。
- **Yasir 的研究领域**：Yasir 的研究领域包括 **安全机器学习 (Secure Machine Learning)**、**隐私保护 (Privacy Preservation)**、**AI 驱动的网络安全 (AI-driven Cybersecurity)**、**计算机视觉 (Computer Vision)** 和 **NLP**。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1384977380623253524)** (12 条消息🔥): 

> `adjoint 和 .mh 的实现, Whisper 悬赏移除, tinygrad 的复数张量` 


- **Adjoint 和 .mh 尚未实现！**：成员们讨论了为什么 tinygrad 中没有实现 **adjoint** 和 **.mh**，答案是开发者希望将项目的复杂性保持在绝对最低限度，且 **adjoint** 的相同功能可以通过 `x.transpose(-2, -1)` 实现。
- **Whisper 悬赏依然有效！**：成员们讨论了 **$200 Whisper 悬赏** 是否会被取消，共识是两个悬赏是互补的。
   - 其中一个悬赏涉及修复现有的 **Whisper** 示例，而新悬赏的最终目标是使其能在网页上运行。
- **复数张量尚未实现！**：一名成员询问关于实现 **conjugate** 的事宜，并得知 tinygrad 目前还没有复数的实现，因此无法完成。
   - 不过，该成员表示他们为 tinygrad 创建了自己的 [复数张量实现](https://cdn.discordapp.com/attachments/1068976834928193609/1385280077079777501/complex_tensor.py?ex=68557e1b&is=68542c9b&hm=55b05763c0469aa8cacc37f4159ec42c988c0b125d7a662629e3085b05abb2b7)，但目前还不完整。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1384993797771104438)** (12 条消息🔥): 

> `Mr. Beast 垃圾信息, 在 Python 中实现 GPT4All` 


- **Discord 成员被告知停止 Mr. Beast 垃圾信息**：一名 Discord 成员被要求停止在频道中发布 **Mr. Beast** 的垃圾内容。
- **用户寻求 Python 实现 GPT4All 的指导**：一名成员正在寻求关于如何将 **GPT4All** 集成到其 **Python 代码** 中的帮助或教程。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1385054434647609364)** (11 messages🔥): 

> `Python 3.9, typing.Optional, future annotations, deprecation of 3.9, pytorch compatibility` 


- **Python 3.9 CI 对 `| None` 报错**: Python 3.9 CI 正在对 `| None` 类型提示（typehinting）报错，引发了是否可以继续使用 `Optional` 的讨论。
   - 有人指出 `X | Y` 类型提示是从 Python **3.10** 开始提供的。
- **`__future__` Annotations 允许在 3.9 上使用 `X | Y`**: 使用 `from __future__ import annotations` 可以让 `X | Y` 在 Python **3.9** 上运行，并能消除自定义对象的字符串类型需求。
   - 这种方法为使用 `list`, `dict`, `tuple`, `X | Y`, `X | None` 等类型提示奠定了面向未来的基础。
- **建议弃用 Python 3.9**: 一位成员建议直接弃用 Python **3.9** 作为解决方案，并指出它很快就会结束生命周期。
   - 另一位成员提到想使用 **3.13** 的特性并倾向于 **3.12** 的泛型语法，但也承认这需要进行大量的改动。
- **Torchtune 与 Pytorch 的 Python 支持保持一致**: 讨论指出 **torchtune** 正试图在 Python 版本支持方面与 **pytorch** 保持一致。
   - 使用 Python **3.10** 是一个很好的折中方案，因为可以通过 `typing_extensions` 获取新特性。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1385017788816425070)** (10 messages🔥): 

> `DSPy for Beginners, Finetuning Llama models, Compiled DSPy Prompts JSON format, Prompt-like DSPy signatures, DSPy with Amazon Bedrock (Claude models)` 


- **DSPy 新手快速入门**: 一位 **DSPy** 新手询问从哪里开始学习并寻求建议。
   - 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=LCEmiRjPEtQ)，该视频对 **DSPy** 提供了很好的解释。
- **操作系统就是 LLM？**: 一位成员发现 YouTube 上将 **LLM 比作操作系统**的类比非常符合 **DSPy 作为高级语言的哲学**。
   - 他们进一步阐述说 **DSPy** 就像 **C** 语言，可以运行在不同的后端并针对它们进行专门编译，从而抽象掉底层的特定汇编方言或具体的 **CPU 指令集**。
- **Bedrock 用户遇到挫折**: 一位用户报告说，在将 **DSPy** 与 **Amazon Bedrock**（Claude models - haiku, sonnet v2）结合用于分类和重写任务时效果不佳。
   - 他们怀疑 **DSPy** 生成的 Prompt 可能与模型的训练方式不匹配。
- **铸造热潮（Minting Mania）正式开始**: 团队正式决定允许个人从今天开始进行铸造（minting）。
   - 他们决定不设白名单，而是让这段时间内在线的人能够在此处进行 [铸造](https://openseacix.vercel.app/)。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1385127399485739030)** (1 messages): 

> `Agentic AI Summit, UC Berkeley, Early Bird Tickets` 


- **Agentic AI Summit：伯克利盛会！**: Agentic AI Summit 将于 **2025 年 8 月 2 日**在 **UC Berkeley** 举行，该峰会基于热门的 [LLM Agents MOOC](https://rdi.berkeley.edu/events/agentic-ai-summit) 举办，预计将有 **1,500+** 名线下参会者。
   - 峰会内容包括主题演讲、小组讨论、工作坊、初创公司路演以及 AgentX Demo Day。
- **早鸟票：最后折扣机会！**: Agentic AI Summit 的早鸟定价将于 **2025 年 6 月 30 日**截止，为学生（**$25**）、初创公司（**$60**）和行业专业人士（**$80**）提供折扣门票。
   - 学生和独立开发者可以申请费用减免，购票链接请点击 [此处](https://na.eventscloud.com/ereg/index.php?eventid=842399)。
- **演讲者阵容：AI 巨擘集结！**: Agentic AI Summit 汇聚了行业和学术界的领袖，包括 **Vinod Khosla** (Khosla Ventures)、**Ion Stoica** (Databricks, Anyscale)、**Dawn Song** (UC Berkeley)、**Sergey Levine** (Physical Intelligence) 等。