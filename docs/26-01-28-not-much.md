---
companies:
- openai
- anthropic
- deeplearningai
- langchain
- apple
date: '2026-01-28T05:44:39.731046Z'
description: '**AI 新闻摘要（2026年1月27日-1月28日）**


  在这两天相对平静的日子里，业界对前沿模型的“性格分化”（personality split）进行了深度探讨：**GPT-5.2** 在“探索”（exploration）方面表现卓越，而
  **Claude Opus 4.5** 则在“利用/执行”（exploitation）方面更胜一筹。这表明 OpenAI 的模型更适合研究工作流，而 Anthropic
  则代表了商业层面的可靠性。


  随着代理式编码循环（agentic coding loops）的兴起，新的失效模式也随之出现，这使得“自我验证”工作流开始受到青睐。开源模型 **Kimi K2.5**
  成为讨论的焦点，它号称拥有增强的智能体执行能力、多模态能力以及更精湛的代码处理能力。该模型可在配备 Thunderbolt 5 (RDMA) 的 **Apple
  Silicon M3 Ultra Mac Studio** 上运行，并在基准测试和价格上直接挑战 Claude Opus 4.5。尽管模型质量出众，但授权许可问题可能会影响其在企业端的普及。


  此外，网络热梗“clawdbot”反映了智能体品牌化的快速扩散。在智能体工程（Agent engineering）方面，由 **DeepLearning.AI**、**Anthropic**
  和 **LangChain** 共同推动的共享“技能（skills）”接口取得了显著进展。'
id: MjAyNi0w
models:
- gpt-5.2
- claude-opus-4.5
- kimi-k2.5
people: []
title: 今天没什么事。
topics:
- agentic-ai
- multimodality
- coding
- self-verification
- agent-engineering
- model-benchmarking
- model-optimization
- workflow-automation
---

**平静的一天**

> 2026年1月27日至1月28日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord（**206** 个频道，**7100** 条消息）。估计节省阅读时间（按 200wpm 计算）：**559 分钟**。**我们的新网站**现已上线，包含完整的元数据搜索和美观的 vibe coded 历期内容展示。查看 https://news.smol.ai/ 获取完整的新闻拆解，并在 [@smol_ai](https://x.com/Smol_AI) 上给我们反馈！

平静的一天。


---

# AI Twitter 综述

**前沿模型的“性格分裂” + 人们实际如何使用它们**

- **探索与利用（Exploration vs. exploitation）的框架**：一个有用的心理模型：目前的前沿 LLM 看起来像是“两极分化”，**GPT-5.2** 针对 *exploration*（探索，更大的搜索/更丰富的推理，“xhigh 和 Pro” 表现出色）进行了优化，而 **Claude Opus 4.5** 则更倾向于 *exploitation*（利用，更强的可靠性且消耗更少的 token；额外的“推理”通常带来的增益较少）——这意味着 OpenAI 可能更适合研究工作流，而 Anthropic 则更适合对可靠性要求极高的商业部署 ([tweet](https://twitter.com/scaling01/status/2016335491243676058))。  
- **Coding agent 的“相位转移”是真实存在的——但也很混乱**：多篇文章反映了实践中的阶跃式变化：创始人兼工程师们越来越多地运行“Agentic”编码循环，但却遇到了新的失效模式：Agent 不询问澄清性问题、变得“困惑”或编辑不相关的文件。Mikhail Parakhin 描述说他已经达到了可以指定一个调度器并信任其运行的程度，但仍然不能让 Agent 在现有的代码库上自由发挥，因为会产生副作用编辑 ([tweet](https://twitter.com/MParakhin/status/2016362688444825833))。相关建议：像 *self-verification*（例如 Playwright 截图 + 迭代直到通过的规则）这样的工作流正成为常见的操作规程 ([tweet](https://twitter.com/pierceboggan/status/2016335657602285822))。

---

**Kimi K2.5（+ “clawdbot” / swarm 模式）成为本周开源模型的焦点**

- **K2.5 声称：Agent + 多模态 + 编码润色**：一篇基于知乎的长文综合论证了 **Kimi K2.5** 通过增强 **Agent 执行**、**多模态**和**编码**，升级了 K2 “智能 > 能力”的不平衡，减少了暴力 token 使用并提高了指令遵循的稳定性；仍被标记的问题：幻觉和持续存在的 NBSP 格式异常 ([thread](https://twitter.com/ZhihuFrontier/status/2016363957876097089))。第二篇知乎综述提出了多模态的实际案例：“视觉”在 Agent 需要验证 UI 状态（重叠、图像损坏、视觉回归）时至关重要，能够实现更紧密的动作-批评者（action–critic）循环，且减少人工反馈 ([thread](https://twitter.com/ZhihuFrontier/status/2016438778030850059))。  
- **分发 + 本地运行正推动热度**：关于 K2.5 在高端 Apple silicon 设备上运行的报告走红：使用通过 **Thunderbolt 5 (RDMA)** 连接的 **2× 512GB M3 Ultra Mac Studios**，搭配 **Exo Labs / MLX** 后端，速度可达 **~24 tok/s** ([tweet](https://twitter.com/alexocheema/status/2016404573917683754))。Kimi 还在 r/LocalLLaMA 上发起了 AMA ([tweet](https://twitter.com/Kimi_Moonshot/status/2016443435553890419)) 并宣布在 “Eigent” 上可用 ([tweet](https://twitter.com/Kimi_Moonshot/status/2016473945957155252))。  
- **基准测试 + 价格压力**：Kilo Code 推广了免费周，声称 K2.5 在多项编码基准测试中击败了 Opus 4.5 ([tweet](https://twitter.com/kilocode/status/2016449095511007535))；Kimi 自己的账号声称其为“#1 开源编码模型” ([tweet](https://twitter.com/Kimi_Moonshot/status/2016521406906028533))。一项关于从图像生成 UI 的非正式 A/B/C 测试发现，Opus 质量最好但价格昂贵，Codex 最快/最便宜但保真度较低，而 K2.5 达到了“~90% 的 Opus 质量，成本仅为 ~38%” ([tweet](https://twitter.com/JuanPa/status/2016634998988865571))。  
- **授权摩擦作为采用阻碍**：一条尖锐的评论指出，即使模型非常出色，修改后的许可证 + Logo 要求也会扼杀企业采用 ([tweet](https://twitter.com/dbreunig/status/2016531878795256286))。  
- **“Clawdbot” 作为文化产物**：这个梗本身（人们对 “clawdbot” 到底是什么感到困惑）反映了 Agent 品牌和分叉（forks）扩散的速度之快 ([tweet](https://twitter.com/dejavucoder/status/2016341138740052126))，并引发了对生态系统信号丢失的更广泛担忧（见下文）。

---

**Agent 工程：技能、测试框架（harnesses）、评估（evals）和“可靠性税”**

- **Skills 正在凝练成一个共享的接口层**：一个主要趋势是将工作流逻辑从 prompt 中移出，转入可重用的 “skills”（指令文件/文件夹，按需加载）。DeepLearning.AI + Anthropic 推出了关于 “Agent Skills” 的课程，强调在 Claude 生态（Claude.ai, Claude Code, API, Agent SDK）中的可移植性 ([tweet](https://twitter.com/AndrewYNg/status/2016564878098780245))；LangChain 正在通过渐进式披露推动 “Skills” 成为轻量级、可共享的单元 ([tweet](https://twitter.com/sydneyrunkle/status/2016585688389734654))。HF 展示了 “upskill”：将强模型的 trace 转换为可迁移的 skills，随后评估其影响；在某些开源模型上，CUDA-kernel 编写的准确率提升了高达 **+45%**，但在其他模型上则有所下降——这强化了进行逐模型评估的必要性 ([tweet](https://twitter.com/ben_burtenshaw/status/2016534389685940372)；推文中的博客链接：https://twitter.com/ben_burtenshaw/status/2016534392974234013)。  
- **上下文管理正趋向于“文件系统优先”**：DeepAgents (LangChain) 描述了如何卸载/总结工具 I/O，并依靠文件系统来界定上下文边界 ([thread](https://twitter.com/hwchase17/status/2016548732880445772)；补充说明：[tweet](https://twitter.com/sydneyrunkle/status/2016560221720867307))。  
- **Evals 正向多轮 + 可追溯性收敛**：明确出现了将 Agent tracing 作为评估单步 vs 全轮 vs 多轮行为基础的呼声 ([tweet](https://twitter.com/samecrowder/status/2016563057947005376))。新的基准测试/工具集：**SWE-fficiency** 发布了其测试工具和仓库 ([tweet](https://twitter.com/18jeffreyma/status/2016511583032061999)；以及 [tweet](https://twitter.com/OfirPress/status/2016559053808222644))，而 **CooperBench** 因用于衡量多 Agent 协作而受到关注 ([tweet](https://twitter.com/gneubig/status/2016555800982937879))。安全方面：“AgentDoG” 提出诊断跨轨迹不安全行为的根本原因 ([tweet](https://twitter.com/HuggingPapers/status/2016366634475388968))。  
- **可靠性和验证循环是瓶颈**：MiniMax 指出长交互链成本高昂，并建议使用 **并行工具调用** 来减少验证器风格设置中的轮次 ([tweet](https://twitter.com/MiniMax_AI/status/2016488781860458789))。另外，一份强有力的批评警告称，“vibe-coded software”（凭感觉编码的软件）破坏了传统的信号（设计质量、文档、生态成熟度），将评估负担转移给了用户，并要求建立新的信任框架 ([tweet](https://twitter.com/tnm/status/2016342022723141782))。

---

**Infra + 效率：量化、蒸馏、推理栈与本地部署**

- **NVIDIA 推动 NVFP4（Nemotron 3 Nano）**：NVIDIA 发布了 **Nemotron 3 Nano** 的 **NVFP4** 精度版本，声称通过 **Quantization Aware Distillation** 在 **Blackwell B200 上实现了高达 4 倍的吞吐量**，并保持了 **~99.4% 的 BF16 准确率** ([tweet](https://twitter.com/NVIDIAAIDev/status/2016556881712472570))。vLLM 迅速增加了支持 ([tweet](https://twitter.com/vllm_project/status/2016562169140433322))。  
- **重 Embedding 架构“再次走红”**：关于 DeepSeek 类似 Engram 思路的讨论仍在继续：一篇 LongCat Flash 论文被总结为使用 **多哈希子表**，并发现 Embedding 主要在高 MoE 稀疏度下起作用；关键的实践陷阱包括通过放大（√D/LayerNorm）来避免首层注意力淹没，以及在词表大小对齐不佳时出现的碰撞峰值 ([tweet](https://twitter.com/eliebakouch/status/2016577949676319092))。  
- **推理/工具链生态持续整合**：vLLM 的 SIGs 和 office hours 正在使治理和路线图节奏正式化 ([tweet](https://twitter.com/vllm_project/status/2016526685869596974))；LM Studio 0.4.0 将自己定位为部署具有并行请求、有状态 REST API 和 MCP 支持的本地模型的“下一代”工具 ([tweet](https://twitter.com/lmstudio/status/2016573570822930708))。Cohere 推出了 **Model Vault**（隔离的 VPC，“无噪音邻居”，弹性推理），作为托管的“主权”托管方案 ([tweet](https://twitter.com/cohere/status/2016512841751154739))。  
- **蒸馏成为默认的“交付形态”**：多条动态呼应了这一新兴标准：训练你能做出的最强模型，然后为了部署进行蒸馏/量化 ([tweet](https://twitter.com/code_star/status/2016588669008953631))。MongoDB Research 的 **LEAF** 提出了 Embedding 的非对称蒸馏：离线使用大型教师模型嵌入文档，在线使用紧凑的学生模型嵌入查询；声称能达到 **~96% 的教师模型质量**，体积缩小 **5–15 倍**，速度提升高达 **24 倍**，从而实现 CPU/边缘端的 Embedding 推理 ([tweet](https://twitter.com/LiorOnAI/status/2016481603426414883))。

---

**大厂产品化：浏览器 Agent、“AI 科学家”叙事与落地现实检查**

- **Gemini 3 正在接管 Google 的各个产品界面**：Gemini 3 现在为全球范围内的 **AI Overviews** 提供支持 ([tweet](https://twitter.com/_philschmid/status/2016552420013199856))。Google 发布了重大的 Chrome 更新：侧边栏 UX、更深层的应用集成、用于图像编辑/生成的 Nano Banana，以及用于执行多步任务的 **Auto Browse**（预览版；美国；Pro/Ultra） ([thread](https://twitter.com/Google/status/2016575105346773297)；以及 [thread](https://twitter.com/GeminiApp/status/2016575257436647521))。工程师指出，这可能是迄今为止最强大的浏览器 AI 集成 ([tweet](https://twitter.com/kimmonismus/status/2016628933706309981))。  
- **OpenAI Prism 定位**：Sebastien Bubeck 明确否认 OpenAI 意图瓜分科学发现的份额，鼓励研究人员使用 ChatGPT/Prism 进行科学研究 ([tweet](https://twitter.com/SebastienBubeck/status/2016345977481777188))。其他人则强调了 Prism 对于学生通过图表学习论文的效用 ([tweet](https://twitter.com/daniel_mac8/status/2016554325691015604))。  
- **普及程度仍然不均**：一个明显的断层是：积极使用尖端工具的创始人能亲眼看到这种转变；而其他人仍将 AI 视为“一般”，限制了组织内的采用 ([tweet](https://twitter.com/GergelyOrosz/status/2016443395405705533))。The Information 报告称 ChatGPT Agent 在使用率和普及方面正面临困境 ([tweet](https://twitter.com/steph_palazzolo/status/2016545857139540260))。  
- **Microsoft “数字同事” 竞争**：有报告称 Satya Nadella 正在亲自测试竞争对手的 Agent 并加速内部开发，甚至使用 Anthropic 模型，旨在掌控 Windows 原生的 Agent 层 ([tweet](https://twitter.com/kimmonismus/status/2016526803138236916))。

---

**科学 + 机器人：基因组学权重开源、可解释性作为发现引擎，以及具身智能规模化**

- **DeepMind AlphaGenome 开源**：DeepMind 宣布了用于预测遗传变异分子影响的 **AlphaGenome**，并提到其 **API 每日调用量超过 100 万次**，拥有 **3,000 多名用户**；随后宣布**开放模型 + 权重** ([tweet](https://twitter.com/GoogleDeepMind/status/2016542480955535475)；权重：[tweet](https://twitter.com/GoogleDeepMind/status/2016542490115912108))。随后，官方再次重申了权重的可用性，并提供了 Hugging Face 集合链接 ([tweet](https://twitter.com/osanseviero/status/2016628065422762113))。  
- **可解释性 → 生物标志物流水线 (Goodfire + Prima Mente)**：Goodfire 报告称，通过对生物医学基础模型进行可解释性分析，发现了一类新型的 **阿尔茨海默病生物标志物**，构建了一个可重复的闭环：在科学数据上训练超人模型 → 机械可解释性（Mech Interp）分析 → 实验验证 → 产生新科学 ([thread](https://twitter.com/GoodfireAI/status/2016563911508840623))。  
- **具身基础模型随真实机器人数据规模化 (LingBot-VLA)**：一份大型总结强调了证据表明，随着真实世界操控数据从 **3k 小时增加到 20k 小时**，VLA 的表现持续提升；该架构通过共享注意力机制将预训练的 VLM (Qwen2.5-VL) 与动作专家相结合；报告称在 GM-100 基准测试中优于 π0.5 等模型 ([tweet](https://twitter.com/omarsar0/status/2016518141308993565))。  
- **Figure 的 Helix 机器人控制**：Brett Adcock 声称 Helix 模型在**无需远程操作**的情况下控制全身行为（行走/触摸/规划），并称这是 Figure 最重大的发布 ([tweet](https://twitter.com/adcock_brett/status/2016358054242222136))。

---

### 热门推文（按参与度排序）

- **公司健康 / 裁员**：“连续两年每季度裁员对健康的危害比每天抽三包烟还大” ([tweet](https://twitter.com/vikhyatk/status/2016345591748690295))。  
- **Kimi K2.5 本地运行**：使用 2 台 M3 Ultra Mac Studio 设置，以约 24 tok/s 的速度运行 K2.5 ([tweet](https://twitter.com/alexocheema/status/2016404573917683754))。  
- **编程的“外包时刻”**：《Clean Code》作者使用 Claude 编写软件，被视为一个象征性的里程碑 ([tweet](https://twitter.com/mischavdburg/status/2016389228356149460))。  
- **新 AI 实验室官宣**：“Flapping Airplanes” 融资 **1.8 亿美元**（GV/Sequoia/Index） ([tweet](https://twitter.com/flappyairplanes/status/2016564437499728259))。  
- **Karpathy 论新研究实验室**：认为新的研究优先型初创公司仍然有可能比现有巨头执行得更好；预计可能会有 **10 倍级**的突破，并祝贺了新创始人 ([tweet](https://twitter.com/karpathy/status/2016590919143952466))。  
- **Google Chrome + Gemini 3 Agent 功能**：Chrome 重大发布推文串 ([tweet](https://twitter.com/Google/status/2016575105346773297))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Kimi K2.5 模型性能与成本分析

- **[在本地运行 Kimi K2.5](https://www.reddit.com/r/LocalLLaMA/comments/1qpfse6/run_kimi_k25_locally/)** (活跃度: 328): **该图片提供了在本地运行 **Kimi-K2.5** 模型的指南，强调了其在视觉、编程、Agentic 和聊天任务中的 SOTA 表现。该模型是一个拥有 1 万亿参数的混合推理模型，需要 `600GB` 的磁盘空间，但经过量化的 **Unsloth Dynamic 1.8-bit** 版本将这一需求降低至 `240GB`，降幅达 `60%`。该指南包含了使用 `llama.cpp` 加载模型的指令，并演示了为一款简单游戏生成 HTML 代码的过程。该模型可在 [Hugging Face](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) 上获取，更多文档可在 [Unsloth 官方网站](https://unsloth.ai/docs/models/kimi-k2.5)找到。** 一位评论者询问该模型在 Strix Halo 硬件上的表现，特别是每 Token 的生成时间，表明了对 Benchmarking 的兴趣。另一条评论强调了极高的 VRAM 需求，认为只有少数用户能本地运行该模型，而第三条评论则幽默地询问是否有更小版本的模型。

    - Daniel_H212 正在询问 Kimi K2.5 模型在 Strix Halo 硬件上的性能，特别是每 Token 的生成速度（秒/Token）。这表明其关注点在于该模型在高阶硬件配置上的效率 Benchmarking。
    - Marksta 提供了关于 Kimi K2.5 量化版本的反馈，特别是 Q2_K_XL 变体。他们指出该模型保持了高度的连贯性并严格遵守 Prompt，这是 Kimi-K2 设计的特征。然而，他们也提到虽然模型的创作能力有所提升，但在创意场景的执行力上仍有欠缺，经常提供逻辑通顺但文笔欠佳的回复。
    - MikeRoz 质疑了像 Q5 和 Q6（例如 UD-Q5_K_XL, Q6_K）这类高位量化的实用性，因为专家们更倾向于 Int4 量化。这引发了关于模型大小、性能和量化精度之间权衡的讨论，专家们更偏好高效的低比特量化。

  - **[Kimi K2.5 是最强的代码开源模型](https://www.reddit.com/r/LocalLLaMA/comments/1qp87tk/kimi_k25_is_the_best_open_model_for_coding/)** (活跃度: 840): **来自 LMArena.AI 的图片展示了 Kimi K2.5 作为领先的代码开源模型，综合排名第 7。该排行榜突出了各种 AI 模型，比较了它们的排名、分数和置信区间，其中 Kimi K2.5 因其在编程任务中的卓越表现而受到关注。该模型因其准确性受到称赞，可与 Sonnet 4.5 媲美，并超越了 GLM 4.7，尽管在 Agentic 功能方面尚未达到 Opus 的水平。排行榜提供了一个简洁、用户友好的界面，采用深色背景和粗体文字以确保清晰。** 一位评论者指出，LMArena 的排行榜可能无法充分捕捉模型的多轮对话、长上下文或 Agentic 能力，认为它更像是一种“One-shot vibe check”。另一位用户则对本地运行 Kimi K2.5 所需的配置感到好奇。

    - 一位用户将 Kimi K2.5 与 Sonnet 4.5 和 GLM 4.7 等其他模型进行了比较，指出虽然 Kimi 2.5 在准确性方面与 Sonnet 4.5 持平，但它超越了他们之前的首选 GLM 4.7。他们还表示有兴趣观察来自 [z.ai](http://z.ai) 的 GLM-5 是否会超越 Kimi 2.5。
    - 另一位用户强调了 Kimi K2.5 的性价比，称其虽然成本显著降低（约五分之一），但能力感与 Opus 4.5 相当。他们还提到它比 Haiku 更便宜，强调了其性能价值。
    - 一条评论批评 LMArena 未能提供关于模型多轮对话、长上下文或 Agentic 能力的见解，认为它仅提供了对模型的浅显评估。

  - **[Kimi K2.5 在性能相近的情况下成本仅为 Opus 的 10% 左右](https://www.reddit.com/r/LocalLLaMA/comments/1qoty38/kimi_k25_costs_almost_10_of_what_opus_costs_at_a/)** (活跃度: 716): **该图片提供了 **Claude Opus 4.5** 和 **Kimi K2.5** 模型之间的成本对比，强调 Kimi K2.5 在性能相近的情况下明显更便宜，成本仅为 Claude Opus 4.5 的 10%。具体而言，Claude Opus 4.5 的每百万 Token 输入成本为 `$5.00`，输出为 `$25.00`，而 Kimi K2.5 的输入成本为 `$0.60`，输出为 `$2.50`。这表明 Kimi K2.5 可能是 SOTA 闭源模型的一个极具性价比的替代方案，特别是对于非网页端的任务。** 一些评论者对性能声明表示怀疑，指出 Kimi K2.5 在执行相同任务时使用的 Token 数量是其他模型的三倍，这影响了性价比和延迟。其他人则认可了 Kimi 模型的潜力，尤其是在写作任务方面。

- one-wandering-mind 指出，对于相同的任务，Kimi K2.5 使用的 tokens 数量是 Opus 的 3 倍，这同时影响了成本和延迟。这表明，虽然 Kimi K2.5 更便宜，但在考虑 token 使用量时，其成本优势更准确地说是 3 倍而非 10 倍。该评论还强调了在性能比较中考虑 token 使用量的重要性，因为它会影响成本和延迟。
- ghulamalchik 提到其更倾向于即将推出的模型，如 DeepSeek 4 和 MiniMax M2.2，这是基于以往对各种模型的使用经验。这表明，尽管 Kimi K2.5 备受关注，但一些用户仍在期待来自其他在经验中被证明可靠的模型的新发布。

- **[Kimi K2 Artificial Analysis Score](https://www.reddit.com/r/LocalLLaMA/comments/1qos25i/kimi_k2_artificial_analysis_score/)** (活跃度: 405): **该图展示了通过 "Artificial Analysis Intelligence Index" 对 AI 模型进行的对比分析，重点突出了 "Kimi K2" 的得分为 `47`，运营成本为 `$371`。围绕该图片的讨论集中在 "Kimi K2.5" 的许可条款上，该条款限制了月活跃用户超过 `1 亿` 或月收入超过 `$2000 万` 的产品的商业使用，要求显著展示 "Kimi K2.5" 品牌。这种授权方式与 Llama 4 等其他模型进行了对比，暗示这可能是一个 bug 或应用中的不一致。该图片和评论反映了 AI 模型的竞争格局，特别是在开源与商业使用背景下。** 评论者讨论了 "Kimi K2.5" 的许可条款，注意到其相较于 Llama 4 等其他模型的独特限制。此外，还有一种期待开源模型超越商业模型的倾向，并提到了 "DeepSeek"。

    - FullOf_Bad_Ideas 强调了 Kimi K2.5 修改后的 MIT 许可协议中的细微差别，该协议要求月活跃用户超过 1 亿或月收入超过 2000 万美元的商业产品必须显著展示 'Kimi K2.5'。这一规定并未应用于 Llama 4 等其他模型，暗示这可能是一个 bug 或应用中的不一致。
    - BrianRin 讨论了 Kimi 2.5 在企业用例中的潜力，并将其与 Opus 4.5、Gemini 3 Pro 和 GPT 5.2 进行了比较。该评论者对 Kimi 2.5 的性价比和输出质量很感兴趣，并指出如果它能达到这些模型 95% 的输出质量，它将成为扩展企业级应用的可行选择。
    - sine120 批评了 Artificial Analysis 评分，认为它在评估模型实际应用场景的表现时并非一个有意义的指标。这暗示需要更细致的评估指标，以更好地捕捉现实世界的可用性和性能。

- **[[LEAKED] Kimi K2.5’s full system prompt + tools (released <24h ago)](https://www.reddit.com/r/LocalLLaMA/comments/1qoml1n/leaked_kimi_k25s_full_system_prompt_tools/)** (活跃度: 282): **该帖子披露了 **Moonshot 的 Kimi K2.5** 完整系统提示词和工具的泄露，包含 `5k tokens` 的数据，如工具模式 (tool schemas)、内存 CRUD 协议、上下文工程和基础安全护栏 (guardrails)。泄露内容包括金融和 arXiv 等外部数据源，并已在多个平台（包括 [GitHub](https://github.com/dnnyngyen/kimi-k2.5-prompts-tools) 和 [Kimi](https://www.kimi.com/share/19c003f5-acb2-838b-8000-00006aa45d9b)）得到独立验证。这次泄露对开源社区意义重大，提供了对模型架构和运行协议的深入见解。** 评论者对泄露可能对开源项目产生的影响表示兴奋，而一些人则质疑系统提示词本身的实际价值。来自包括中文论坛在内的多个来源的独立验证增加了该泄露的可信度。

- 泄露的 Kimi K2.5 System Prompt 揭示了一种处理记忆持久化（memory persistence）和上下文管理的高级方法。该 Prompt 包含维持专业礼貌、简洁回答以及特定编码规范的指令，例如在 JS/JSON 缩进中使用 Tab 键，以及倾向于使用命名的可重用函数。这种结构旨在通过提供持久的行为锚点来解决“空壳 AI 助手”问题，这能显著影响模型在跨会话中保持人格一致性的能力。
- Kimi K2.5 的记忆持久化机制尤其值得关注。它涉及系统指令与动态上下文注入（context injection）之间的平衡，这对于维持人格一致性至关重要。该系统对对话总结或检索的处理方式会影响新的聊天，甚至记忆结构的微小变化都会导致模型响应的转变，有时使其显得更加“真实”。这凸显了初始 Prompt 结构在决定 AI 是“记住”其行为模式还是仅仅记住事实内容方面的重要性。
- Kimi K2.5 的 System Prompt 还解决了 Context Window 限制问题，这是 AI 模型在长对话中面临的常见挑战。其 Prompt Engineering 旨在通过以支持对话连续性的方式组织先前的交互来处理这些限制。这种方法不仅有助于保持对话流，还能确保 AI 的回复在对话延长时依然保持相关性和上下文适宜性。

### 3. Z-Image 模型预告与发布

  - **[Z-Image 基础模型来了！](https://www.reddit.com/r/LocalLLaMA/comments/1qoiep6/the_zimage_base_is_here/)** (热度: 327): **Tongyi-MAI** 已在 [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image) 上发布了 `Z-Image` 模型，展示了其生成高质量图像的能力，特别侧重于女性角色（约占演示内容的 `90%`）。该模型被认为有可能在 `12GB GPUs` 上运行且质量损失极小，暗示了高效优化的可能性。一个显著特征是“Negative Prompt”（负向提示词）功能，允许对特定的图像生成进行限制，如一个翻译后的示例所示，提示词指定了“西方人、身体畸形”。评论者强调了该模型对女性图像生成的侧重，反映了一个主要的应用场景。此外，还讨论了通过优化在低配硬件上运行该模型的潜力，体现了其效率和适应性。

    - Dr_Kel 讨论了优化 Z-Image 模型使其能在 12GB GPUs 上以极低质量损失运行的可能性，并建议通过一些调整，让硬件配置较低的用户也能更容易地使用该模型。
    - Middle_Bullfrog_6173 指出 Z-Image 基础模型主要对那些对训练或微调模型感兴趣的人有用，而非最终用户。他们暗示该基础模型是后续开发（如在其基础上进行后期训练的 Turbo 模型）的基石。


  - **[API 价格正在暴跌。除了隐私之外，现在运行本地模型的实际理由是什么？](https://www.reddit.com/r/LocalLLaMA/comments/1qp6rm5/api_pricing_is_in_freefall_whats_the_actual_case/)** (热度: 913): **该帖子讨论了 AI 模型 API 访问成本迅速下降的现状。**K2.5** 的价格仅为 **Opus** 的 `10%`，而 **Deepseek** 几乎免费。**Gemini** 也提供了可观的免费层级，导致 API 月度价格下限下降了 `50%`。相比之下，在本地运行 `70B` 模型需要大量的硬件投资，例如 `k+ GPU`（数千美元的 GPU），或者不得不接受量化（quantization）带来的权衡，在消费级硬件上仅能达到 `15 tok/s`。该帖质疑了除隐私之外本地部署的可行性，并指出虽然本地部署在延迟控制和自定义方面具有优势，但与 API 的性价比相比，这些都属于小众优势。**评论者强调了离线能力的重要性，并对 API 提供商的长期定价策略表示不信任，认为目前的低价可能无法持续。他们还强调了在本地运行模型时的可重复性和对模型行为控制的价值，而这些可能会随着 API 的更改而受到破坏。

    - Minimum-Vanilla949 强调了对于经常旅行的人来说离线能力的重要性，并指出 API 公司可能会意外更改条款或价格。这突显了本地模型在不依赖外部变化的情况下，提供一致访问和控制的价值。
    - 05032-MendicantBias 讨论了目前通常由风险投资补贴的 API 定价具有不可持续性。他们认为一旦实现垄断，价格很可能会上涨，因此本地部署和开源工具是对未来价格上涨的战略性规避。
    - IactaAleaEst2021 指出了在使用本地模型时，模型行为的可重复性和信任的重要性。通过下载和审计模型，用户可以确保性能的一致性，而 API 供应商可能会在不通知的情况下修改模型行为，从而影响可靠性。


## 技术性较低的 AI Reddit 社区回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Kimi K2.5 及相关模型发布

  - **[开源 Kimi-K2.5 在包括编程在内的多项基准测试中超越 Claude Opus 4.5。](https://www.reddit.com/r/singularity/comments/1qoojio/open_source_kimik25_is_now_beating_claude_opus_45/)** (热度: 1078): **据报道，开源模型 **Kimi-K2.5** 在多项基准测试中超越了 **Claude Opus 4.5**，特别是在编程任务方面。然而，这些基准测试的细节以及性能提升的程度并未详述，导致人们对这些结果在现实世界中的适用性持怀疑态度。该公告突显了开源 AI 社区在特定任务上追赶或超越专有模型的持续竞争。**评论者对这一说法表示怀疑，质疑基准测试（benchmarks）与现实应用的相关性，并指出缺乏支持 Kimi-K2.5 优于 Claude Opus 4.5 的详细证据。

- 对于 Kimi-K2.5 在基准测试中表现优于 Claude Opus 4.5 的说法存在质疑，一些用户对所引用的具体基准测试表示怀疑。“许多”这个词被认为含义模糊，人们呼吁提供更多关于支撑这些说法的基准测试详细信息。
- 讨论强调了对基准测试的一种普遍批评，即它们往往不能反映现实世界的实用性。一位用户指出，虽然 Kimi-K2.5 在受控的基准测试环境中可能表现良好，但它可能无法达到 Claude Opus 4.5 的实际表现，特别是在编程任务中，Opus 4.5 以单次提示（single prompt）即可提供解决方案而著称。
- 普遍观点认为，基准测试不足以衡量模型的实际能力。对话表明，尽管 Kimi-K2.5 在基准测试中可能显示出有希望的结果，但其在现实应用中（特别是编程方面）的效果可能不如 Claude Opus 4.5，后者因高效提供解决方案而受到称赞。

  - **[Kimi K2.5 Released!!!](https://www.reddit.com/r/singularity/comments/1qo531i/kimi_k25_released/)** (Activity: 1233): **图像展示了四个 AI 模型的性能对比图：**Kimi K2.5**、**GPT-5.2 (xhigh)**、**Claude Opus 4.5** 和 **Gemini 3 Pro**。**Kimi K2.5** 以蓝色标出，在 Agent、编程、图像和视频处理等各项任务中展现出极具竞争力的得分。图表列出了特定的基准测试，如 "Humanity's Last Exam"、"BrowseComp" 和 "OmniDocBench 1.5"，**Kimi K2.5** 在这些测试中通常处于领先地位或表现强劲，表明其在这些任务中的有效性和准确性。得分以百分位数形式呈现，展示了该模型相对于其他模型的表现。**评论者讨论了 AI 模型的幻觉（hallucinations）问题，**Kimi K2.5** 相比其前作有所改进，但仍会产生错误答案。**GPT 5.1 和 5.2** 因能够承认不知道答案而受到关注，而不像 **Kimi 2.5** 和 **Gemini 3** 那样自信地提供错误答案。人们对基准测试的代表性持怀疑态度，质疑 **Kimi K2.5** 在大多数情况下是否真的优于 **Gemini 3**。

    - 一位用户测试了 Kimi K2.5 遵循指令的能力，要求它在不使用网页搜索的情况下识别一个特定的数学竞赛题目。该模型列出了虚构的（hallucinated）竞赛题目并进行了自我怀疑，最终提供了错误答案。这种行为相比 Kimi K2 有所进步，后者无法遵循指令且超时。相比之下，GPT 5.1 和 5.2 因能够承认“我不知道”而受到关注，而 Gemini 3 则自信地提供错误答案。
    - 讨论了 AI 模型中 “Agent Swarm” 的概念，即可能有超过 100 个模型实例由一个单一的监督实例指挥。这种设置被认为成本高昂且复杂，而单个模型同时处理多个任务的可能性将是一个重大进步。用户对这种设置的实际经验表示兴趣，认为 Scaffolding 可能是一种更可行的方法。
    - 一位用户对将 Kimi K2.5 与 Gemini 3 进行比较的基准测试有效性表示质疑，暗示结果可能是刻意挑选的（cherry-picked）。他们对 Kimi K2.5 持续优于 Gemini 3 持怀疑态度，认为如果没有更广泛的证据，这类说法似乎被夸大了。


  - **[Cline 3.55.0: Arcee Trinity Large and Kimi K2.5 now available](https://www.reddit.com/r/CLine/comments/1qpl2fk/cline_3550_arcee_trinity_large_and_kimi_k25_now/)** (Activity: 5): **Cline 3.55.0** 引入了两个重要的开放模型：**Arcee Trinity Large** 和 **Kimi K2.5**。**Arcee Trinity Large** 是一个拥有 `400B` 参数的 MoE 模型，推理时有 `13B` 个激活参数，提供 `128K` 的上下文窗口。它在 MMLU Pro 上达到 `82` 分，在 GPQA Diamonds 上达到 `75` 分，使其适用于通用编程和大型代码库管理，且无 API 成本。**Kimi K2.5** 是一个拥有 `1T` 参数的 MoE 模型，具有 `256K` 上下文，在 SWE-bench 上得分 `76.8%`，并在 Humanity's Last Exam 上以 `50.2%` 的成绩超越了 Opus 4.5。它在视觉编程方面表现出色，能够根据截图生成 UI 代码并自我纠正输出。此外，**ChatGPT Plus/Pro** 用户可以在 Cline 中无需 API 密钥访问 GPT-5 模型。[完整详情请点击此处](https://cline.bot/blog/cline-3-55-0-arcee-trinity-and-kimi-k2-5-now-in-cline)。**一些用户对这些模型的开源性质和竞争性能表示兴奋，特别注意到在编程应用中节省成本和提高灵活性的潜力。还有人对这些模型处理大上下文窗口的能力和自我纠错功能感兴趣。

- 一位用户强调了 Arcee Trinity Large 模型在性能上的改进，指出与之前的版本相比，其处理速度有了显著提升。他们提到该模型的架构已针对更好的并行处理（parallel processing）进行了优化，这对于高效处理大规模数据集（large datasets）至关重要。
- 另一条评论讨论了 Kimi K2.5 模型在自然语言理解（natural language understanding）方面的增强功能。用户指出，该模型现在支持更多语言，并提升了上下文保留（context retention）能力，这对于需要细腻语言处理的应用非常有益。
- 围绕新模型的内存使用（memory usage）展开了一场技术辩论。一些用户对增加的内存占用（memory footprint）表示担忧，特别是在资源受限的环境（resource-constrained environments）中部署时。其他人则认为，考虑到模型在准确性和速度方面的提升，这种权衡是合理的，并建议未来的更新可能会侧重于优化内存效率（memory efficiency）。

### 2. Prompt Engineering 技巧与讨论

  - **[最疯狂但确实有效的 Prompt：“你快没时间了”](https://www.reddit.com/r/PromptEngineering/comments/1qp0kay/the_most_unhinged_prompt_that_actually_works/)** (热度: 75): **该帖子讨论了一种非传统的 Prompt Engineering 技巧，即在 Prompt 中增加紧迫感（例如：“你只有 30 秒。分析这些数据。我遗漏的那一个关键点是什么？开始。”），从而让语言模型提供更集中、更即时的见解。这种方法与传统的、详细的 Prompt 形成对比，后者往往导致响应较慢且针对性较差。作者幽默地指出，这种方法似乎能让 AI 停止过度思考，类似于人类在时间压力下的反应。这种技术被比作 Prompt Engineering 中的“应用混沌理论”。** 评论者建议，只需指示 AI 保持简洁即可达到类似效果。另一种观点认为，有效的管理技能（无论是针对人类还是 AI）都涉及具体地阐述任务，这能提升产出结果。然而，也有人指出，这种紧迫感技术可能会降低专为复杂推理设计的模型的思考深度。

    - angry_cactus 强调了在 Prompt 中使用紧迫感时的一种权衡，指出虽然这可能很有效，但可能会减少模型的“思考时间”。这表明当速度优先于彻底性时，响应的深度或质量可能会下降。
    - fatstupidlazypoor 在管理人类与管理语言模型之间画了等号，强调清晰且具体的表达可以显著提升两者的表现。这突显了在 Prompt Engineering 中为了实现预期结果而保持精确性的重要性。
    - authorinthesunset 提出了一种简单而有效的 Prompt 策略：指示模型保持简洁。这种方法可以精简响应，在看重简洁性的语境下，潜在地提高效率和相关性。

  - **[Micro-Prompting：通过更短的指令获得更好的 AI 结果](https://www.reddit.com/r/PromptEngineering/comments/1qonyx9/microprompting_get_better_ai_results_with_shorter/)** (热度: 49): **该帖子讨论了 AI 的“Micro-Prompting”概念，提倡使用更短、更集中的指令来提高 AI 响应质量。它建议特定的角色分配和强力词汇（如“audit”、“clarify”和“simplify”）可以通过引导 AI 访问针对性知识而非泛泛的信息，从而显著增强 AI 的输出。该帖子还强调了构建指令以控制输出的重要性，例如使用“分 3 点说明”或“清单格式”，并警告不要犯过度解释背景或使用通用角色等常见错误。据说这种方法与传统的长篇 Prompt 相比，能在更短的时间内产生更好的效果。** 评论中一个值得注意的观点是，角色分配有时可能会阻碍 Prompt 的有效性，而具体性则更有益。这表明在角色特异性与 Prompt 简洁性之间的平衡仍存在争议。

    - aiveedio 讨论了 Micro-Prompting 的有效性，指出简短、集中的 Prompt 可以通过避免信息过载来产生更干净的 AI 输出。然而，在角色肖像或故事场景等创意任务中，指定表情、服装和光影的详细 Prompt 是必要的，以避免平庸的结果。关键在于平衡简洁与精准，从一个 Micro-Prompt 开始，并根据需要迭代增加细节，以便在不使模型过载的情况下保持重点。
    - psychologist_101 提出了一个关于使用 Opus 4.5 的有趣观点：当要求模型生成自己的 Prompt 时，它会产生冗长、详细的输出。这表明模型本身可能为了清晰度和上下文而更倾向于详细的 Prompt，这与简短 Prompt 更有效的观点形成对比。这突显了用户预期与模型行为之间潜在的差异，强调了实验 Prompt 长度和细节以获得最佳结果的必要性。

### 3. 新 AI 模型与基准测试发布

  - **[DeepSeek-OCR 2 现已发布！🐋](https://www.reddit.com/r/DeepSeek/comments/1qo6xb4/deepseekocr_2_is_out_now/)** (热度: 507): **图片宣布了 **DeepSeek-OCR 2** 的发布，这是一个集成了全新 **DeepEncoder V2** 的先进 OCR 模型。该编码器通过模拟人类对图像的逻辑化扫描来提升 OCR 准确度，这对于视觉和文本推理任务至关重要。图中的图表展示了模型的“视觉因果流（Visual Causal Flow）”，强调了其在确定阅读顺序之前形成对内容的全局理解的能力。图中的对比表显示了各种文档元素的编辑距离（edit distances）有所改善，突显了该模型优于其前代产品的性能。** 一位用户分享了 Demo 链接供他人试用该模型，表明社区对动手实验很感兴趣。另一位用户表达了对未来版本的期待，暗示当前发布版本处于一个充满前景的发展轨迹中。

    - DeepSeek-OCR 2 已经发布，用户可以通过[此链接](https://deepseek-ocr-v2-demo.vercel.app/)在线试用该模型的 Demo。这为用户提供了亲身体验模型能力的机会，而无需在本地安装。
    - 一位用户指出，DeepSeek-OCR 1 在理解文档布局方面表现出色，但存在局限性，例如会遗漏页眉、页脚和深色背景上的浅色文本等内容。这表明虽然旧模型在布局分析方面很强，但在内容检测方面存在特定的弱点，这些问题可能在第 2 版中得到了解决。
    - 许多人对 DeepSeek-OCR 2 是否有开箱即用的在线 API 感兴趣，这表明对无需复杂技术设置、易于访问的云端解决方案有需求。这反映了使先进 OCR 技术对非技术用户更易于使用的更广泛趋势。

  - **[伙计们，它来了：Z Base](https://www.reddit.com/r/StableDiffusion/comments/1qohra7/here_it_is_boys_z_base/)** (热度: 2374): **图片是 **Tongyi-MAI** 在 Hugging Face 模型库中“Z-Image”页面的截图，展示了一个高效的图像生成模型。该仓库提供了官方网站、GitHub 和在线 Demo 的链接，表明其注重可访问性和社区参与。该模型属于 AI 领域追求更高效、更易用的图像生成工具的大趋势，示例图像以及与 Hugging Face 等平台的集成证明了这一点。** 评论者们对该模型的潜在应用和修改很感兴趣，例如在不同数据集上对其进行 "finetuning"，表明了对其在各种语境下的适应性和性能的关注。

  - **[Z-Image Base 对比 Z-Image Turbo](https://www.reddit.com/r/StableDiffusion/comments/1qojw11/zimage_base_vs_zimage_turbo/)** (热度: 927): **该帖子讨论了 **Z-Image Base** 和 **Z-Image Turbo** 模型之间的比较，强调了它们的性能差异。Turbo 模型以 `2 iterations per second`（每张图 7 秒）的速度运行，而 Base 模型以 `1 iteration per second`（每张图 40 秒）的速度运行。设置包括 seed 为 `4269`，Turbo 的 steps 为 `12`，Base 的 steps 为 `40`，使用 `res_multistep` 采样器（sampler）、`simple` 调度器（scheduler），Base 的 `CFG` 为 `4`。Turbo 模型因更“简洁”且有时更“写实”而受到关注，而 Base 模型则因其视觉质量而受到称赞。** 评论者将这些模型与 "SDXL" 进行对比，暗示图像生成进入了一个新纪元。Turbo 模型因其简洁性和现实感而受到青睐，而 Base 模型则因其令人印象深刻的视觉输出而被记录。

    - Gilded_Monkey1 提出了一个关于 Z-Image 模型中构图稳定所需步数的技术问题，特别是当将其作为 image-to-image (i2i) 任务中的变体启动器时。这表明人们关注模型的迭代过程和收敛速度，这对于高效渲染和实现预期的艺术效果至关重要。
    - diogodiogogod 对 Z-Image Base 和 Z-Image Turbo 进行了对比分析，指出虽然 Turbo 版本更“简洁”且通常更“写实”，但 Base 版本在视觉吸引力方面表现出色。这突显了在复杂性与现实感以及审美质量之间的权衡，这是在针对特定艺术或实际应用进行模型选择时的常见考量。

---

# AI Discord 摘要

> 由 Gemini 3.0 Pro Preview Nov-18 生成的摘要的摘要

**主题 1. 模型之战：Kimi K2.5 的崛起、Arcee 的 Trinity 以及 Arena 的更名**

- **Kimi K2.5 登顶开源排行榜**：新的 **Kimi K2.5 Thinking** 模型在 [Text Arena 排行榜](https://lmarena.ai/leaderboard/text)上夺得 **#1 开源模型** 宝座，在物理和数学等 STEM 基准测试中表现出色。虽然 **19美元/月** 的订阅费或 **0.6美元/1M tokens** 的定价引发了争论，但工程师们正通过 [HuggingFace](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) 和 **Unsloth** 部署本地量化版本。
- **Trinity Large：运行精简的 400B MoE**：Arcee AI、Prime Intellect 和 Datology 发布了 [Trinity Large](https://openrouter.ai/arcee-ai/trinity-large-preview:free)，这是一个 **400B 参数** 的 Mixture-of-Experts 模型，为了提高效率，每个 token 仅激活 **13B 参数**。该开源权重模型使用了 **256 个专家**，并采用激进的路由策略（1.56%），以平衡前沿级知识储备与推理速度。
- **LMArena 更名为 Arena，克隆 Claude UI**：广受欢迎的排行榜更名为 **Arena** ([arena.ai](https://arena.ai/))，其 UI 进行了彻底翻新，用户立即将其贴上 **Claude 克隆版** 的标签，并抱怨激进的 Google **captchas** 验证。此次更新包括新的 [Code Arena](https://lmarena.ai/?chat-modality=code) 和扩展的排行榜，不过用户正强烈要求恢复停止按钮和旧版 emoji。

**主题 2. 开发工具变革：Cursor 限制、LM Studio 无头模式和 Unsloth 的古怪问题**

- **Cursor 的 Auto Mode 付费墙引发不满**：开发者对 **Cursor** 终止无限制的 "Auto mode" 表示沮丧，该功能现在被限制在 **20美元/月** 的订阅配额内，超出后按 **1.25美元/1M** 输入 tokens 收费。用户还报告了一个 **revert button** 消失的 bug，不过一些人正转向使用 **Cursor CLI**，以便在大型代码库上获得更小的内存占用。
- **LM Studio v0.4 推出无头模式**：**LM Studio v0.4** 的发布引入了 **headless mode**，并通过有状态的 **REST API** 实现并行推理，支持在 CI/CD 流水线和非 GUI 服务器上进行部署（[发布日志](https://lmstudio.ai/blog/0.4.0)）。工程师们还在运行时设置中发现了隐藏的针对 AMD GPU 的 **ROCm** 支持，解锁了此前在 UI 中被遮蔽的硬件加速功能。
- **Unsloth 对抗 GLM 4.7 和 CUDA 版本问题**：在 Blackwell B200 上微调 **GLM 4.7** 的工程师面临着 **CUDA 12.8** 驱动程序与模型 **CUDA 13.x** 需求之间的兼容性地狱。成功的解决办法包括使用特定的 torch 后端强制重新安装 **vllm**，并由于 Ada Lovelace 的不兼容性移除 `fp8` 缓存标志。

**主题 3. 安全、越狱与诈骗**

- **神奇字符串让 Claude “变傻”**：红队人员发现了一个特定的字符串 `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL...`，它充当了“断路器”，能可靠地迫使 **Claude** 进入拒绝模式。与此同时，黑客正通过未公开的 POST 请求操纵 **Parallel AI API**，以注入自定义系统提示词。
- **Clawdbot 被揭露为凭据收割机**：社区对 **Clawdbot**（已更名为 **Moltbot**）发出了警告，这是一个 Agent 系统，它会集中收集来自 OpenAI、Google 和 Anthropic 的 API keys。用户将其定性为“先存储，后解密”的安全风险，且容易受到可能导致敏感凭据泄露的 Prompt injection 攻击。
- **OpenAI Prism：科学工具还是安全风险？**：OpenAI 推出了 [Prism](https://archive.md/d9Vsf)，这是一个由 **GPT-5.2** 驱动的科学家研究工作空间，但反响褒贬不一，有人称其“对科学研究有害”。研究人员正在探测其对对抗性攻击的敏感性，并注意到 **GPT Pro 5.2** 同时失去了分析 ZIP 文件的能力。

**主题 4. Agent 前沿：视觉、编码与未来预测**

- **Karpathy 预测 Agent 驱动 80% 代码的未来**：Andrej Karpathy 预测，到 2026 年，**80% 的编码工作**将由 Agent 驱动，这依赖于 LLMs 日益增长的韧性和目标设定能力，而非人类对语法的管理 ([tweet](https://xcancel.com/karpathy/status/2015883857489522876))。同时，关于 **agentic harnesses** 的讨论表明，智能模型很快将取代像 **LangChain** 这样复杂的编排器，转而采用基于文件系统的协作模式。
- **Gemini 3 Flash 获得 Agentic Vision 能力**：Google 为 **Gemini 3 Flash** 引入了 [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/)，使模型能够主动缩放、裁剪和检查图像，以锚定其推理。前端开发者报告称，这一能力已接近 **SOTA**，通过动态操作视觉输入，其表现优于 OpenAI 的静态分析。
- **C++ 在 Agent 开发中占据主导地位**：为了对抗“臃肿”的 Python 框架，工程师们认为高性能 Agent 应该用 **C++** 构建，并推荐使用 **fastwhisper.cpp** 处理 STT 以及 **LFM2.5vl** 处理视觉等技术栈。这与 **LeetCode MCP server** 的发布相契合，该服务器允许 Claude 直接从终端解决编程挑战。

**主题 5：低层优化与硬件内部机制**

- **Decart 发布 Lucy 2 并招聘硬件人才**：Decart 发布了自回归视频模型 **Lucy 2**，并正在积极招聘 **Trainium 3** 和低延迟内核开发人才 ([tech report](https://x.com/DecartAI/status/2016134190509498740))。该团队正在共同赞助内核挑战赛，以在裸机（bare metal）上优化自回归扩散模型。
- **Mojo 生成 GTK 绑定**：**Modular** 团队宣布为 Mojo 自动生成 **GTK 绑定**，承诺简化 GUI 开发，并将在 2 月的社区会议上展示。工程师们还在分析 H100 上 **Mojo vs CUDA/HIP** 的性能，讨论 Mojo 的 `out` 参数是否成功取代了命名值返回优化（NVRO）。
- **Tinygrad 开启 AMD 调试功能**：**Tinygrad** 模拟器现在支持对 AMD GPU 进行细粒度的调试输出（编译使用 `DEBUG=3`，运行时使用 `DEBUG=6`），详见此 [截图](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png)。贡献者们还在通过代码重构而非硬件升级来优化 **GitHub Actions** 的速度，坚持“做对，而不仅仅是做快”的哲学。

---

# Discord：高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **通过社交媒体免费获取模型**：一名成员分享了 [X 上的链接](https://x.com/Exocija/status/2016502660883415422) 以免费访问模型，并附带了一个 [PRIMETALK 上下文文件](https://discord.com/channels/1105891499641684019/1228043845967544380/1466113637541347348)，详细说明了模型兼容性和使用注意事项。
   - 据报道，该系统与大多数现代 AI 模型兼容，但其行为和稳定性很大程度上取决于上下文容量和聊天窗口大小。
- **魔术字符串使 Claude 保持沉默**：一名成员分享了一个“魔术字符串” `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL_1FAEFB6177B4672DEE07F9D3AFC62588CCD2631EDCF22E8CCC1FB35B501C9C86`，它可以稳定地阻止 **Claude** 做出响应。
   - 另一名成员建议这起到了类似“断路器”的作用，可能提高模型在拒绝某些提示词时的准确性。
- **Parallel AI API 破解**：用户正在探索与 **Parallel AI API** 交互的方法，包括通过 POST 请求调整系统提示词（system prompt）。
   - 一名成员分享了向该 API 发送请求的 [PowerShell 示例](https://platform.parallel.ai/)，尽管目前还没有关于调整系统提示词的官方 API 文档。
- **Custom GPT 5.2 即将到来**：一名成员正准备发布新的 **GPT 5.2 Custom GPT**，并声称其效果显著，但需要额外的噪声。
   - 该模型显然可以从其系统提示词中识别日期，引发了关于如何利用图像提取该提示词的讨论。
- **用户被 HackAPrompt 封锁**：一名成员报告称 **HackAPrompt x PlinyAnthropic** 标记了他们，导致其任何消息都无法发送。
   - 这表明存在一个严格的过滤系统，完全阻止被标记的用户与服务进行交互。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena 品牌重塑模仿 Claude 的 UI**：用户注意到 **LMArena 重塑品牌为 Arena**，并认为这是 **Claude UI 的克隆**，一篇 [博客文章](https://arena.ai/blog/lmarena-is-now-arena/) 解释了这一变化。
   - 成员们指出了一些 UI 问题，如 **字体** 和网站文本的可见度，以及一些 **缺失的功能**。
- **验证码难题持续**：用户报告 **Captcha** 验证几乎在每次尝试时都失败，并提供了 *重新登录账号* 或 *关闭所有扩展程序* 以通过验证码的故障排除步骤。
   - 用户讨厌这种验证码，希望旧的表情符号、贴纸和功能能够回归。
- **登录丢失？恢复按钮来救场！**：一位遇到登录问题的成员分享了一个 [恢复按钮](https://cdn.discordapp.com/attachments/1340554757827461211/1466134595035467829/Hdbd.png?ex=697ba3be&is=697a523e&hm=2be3961be4c941479f9ec51709c5eb6af5ea9c79ad3918eb6a15a964ec9fe720&) 的截图，点击该按钮即可重新登录更新后的 **Arena**。
   - 另一位成员也提到了一个 [公告视频](https://youtu.be/TNoAlMv4Eg8?si=d86SArLb6yQ8sdLE)。
- **Kimi K2.5 Thinking 登上 Text Arena 排行榜**：[Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 已更新，`Kimi K2.5 Thinking` 现在位列 **开源模型第 1 名**，**总榜第 15 名**。
   - `Kimi K2.5 Thinking` 在 Coding（编程）中排名 **第 7**，在 Instruction Following（指令遵循）中排名 **第 7**，在 Hard Prompts（高难度提示词）中排名 **第 14**，并且已被添加到 [Code Arena](https://lmarena.ai/?chat-modality=code) 中。
- **Arena Shorts，不到 90 秒生成更好的 AI 视频！**：**Arena**（原 LMArena）在其 [Youtube 频道](https://www.youtube.com/watch?v=0hCI2XEh0x0) 上传了一个名为 `Better AI videos in under 90 seconds` 的视频。
   - 该团队承认，随着平台从仅支持语言模型（Language Models）向外扩展，名称正变得更加通用，且该项目 [此前是 LMSYS 的一部分](https://lmsys.org/)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Batch Size 提升 GPU 效益**：成员们发现，实现理想 GPU 利用率的一种方法是 *增加 Batch Size* 直到利用率提高，并平衡 **GA (Genetic Algorithms)** 带来的潜在收益。
   - 此外，一位成员询问 Unsloth 是否会发布 **Kimi 2.5** 的 **Q3** 版本，并对准确率下降表示担忧。
- **Oracle 的产品引发质疑**：一位成员询问 **Oracle** 是否在 **RAG (Retrieval-Augmented Generation)** 和微调技术方面处于行业领先地位，这引发了辩论。
   - 简短的回复 *"What 😅"* 随后被修正，承认 **OCI (Oracle Cloud Infrastructure)** 确实有一些不错的工具，显示出意见分歧。
- **Arcee 的账本：Trinity 耗资 35 万美元**：一张新的 **Arcee 模型** 图片被分享，并附带说明预训练成本约为 **35 万美元**，同时附上了 [Trinity Large 技术报告](https://github.com/arcee-ai/trinity-large-tech-report/blob/main/Arcee%20Trinity%20Large.pdf) 的链接。
   - 会议明确了 **GLM 4.7** 是一个 **358B** 参数的模型，但 *不是基础模型（base model）*，这使得与 **GLM 4.5** 等模型的基准测试比较意义较小。
- **Gemini 的守门员游戏**：一次 Google 黑客松显示，尽管有严格的输出过滤（特别是在企业/政府环境下），但可以通过设置让 **Gemini API** 产生几乎任何内容。
   - 一位成员通过在系统提示词（system prompt）中加入特定指令，让语音模型说出了脏话。
- **Modal 多 GPU 乱象**：一位成员在 **Modal** 上使用 3 个 GPU 训练 **Qwen3** 模型时遇到问题，由于错误的 `device_map` 配置收到了 *ValueError*。
   - 由于与 **PyTorch 2.4.1** 不兼容，训练设置最终弃用了 **Unsloth**，转而选择 **transformers + PEFT** 方案以获得更好的稳定性。

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Arcee 免费发布 Trinity Large Preview**：Arcee 推出了 **Trinity-Large-Preview**，这是其前沿规模开源权重模型的聊天就绪变体。该模型在限时时间内免费，详情见 [X](https://x.com/OpenRouterAI/status/2016280059527757995?s=20)。
   - 该模型是一个 **400B 参数的稀疏 Mixture-of-Experts** 模型，**每个 Token 拥有 13B 激活参数**，利用了 **256 个专家**，其中 **每个 Token 激活 4 个**（1.56% 路由）以提高效率。这一内容在 [Lucas Atkins 的直播](https://youtube.com/live/3XSdqHY0kNk?feature=share) 中进行了讨论。
- **免费额度助力 Cyberpad**：一位用户更新了 [Cyberpad](https://cyberpad.site) 以包含一些免费额度。
   - 未提供更多信息。
- **图像模型输出故障报告**：用户报告称某些图像模型（如 **GPT-5 Image Mini**、**GPT-5 Image** 和 **Gemini 2.5 Flash Image**）生成的图像不一致，尽管 **Gemini 2.5 flash** 有时可以工作。
   - 像 **Gemini 3 Flash Preview**、**Gemini 2.5 Flash Lite Preview**、**Seed 1.6**、**GLM-4.6v** 和 **Grok 4.1-fast** 等模型具有可用的 `response_format` 支持。
- **OpenRouter 用户等待退款**：用户在接收 OpenRouter 退款时遇到显著延迟，部分用户自一月初以来一直在等待并提交了多个支持工单。
   - 用户要求明确退款时间线，并希望 **OpenRouter** 团队改善沟通。
- **搭载 Gemini 3 Flash 的 Agentic Vision 首次亮相**：Google 推出 [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/)，配合 **Gemini 3 Flash** 实现了视觉推理和代码执行，可进行分步图像处理。
   - OpenAI 的 **O3** 和 **O4-mini** 正在扩展图像功能，通过图像的 Chain-of-Thought 推理实现裁剪、缩放和旋转等任务，详见 [这篇博文](https://openai.com/index/thinking-with-images/)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **消失的撤销按钮令用户沮丧**：用户报告 **revert 按钮** 从 UI 中消失，导致沮丧和 Token 浪费。一位用户发现 [复制旧聊天记录](https://cdn.discordapp.com/attachments/1074847527708393565/1465828018789552390/image.png?ex=697bd7b9&is=697a8639&hm=68ec5dd17c7a1be84a1f639f9a5a98db91ba3bd191336f2afe3e8252b804b12e&) 可以找回该按钮。
   - 一位成员发现不点击撤销按钮会让它重新出现，暗示这是一个 **一次性 Bug**。
- **Cursor CLI：黑马？**：由于内存占用较小，一些开发者更倾向于使用 **Cursor CLI** 而非 IDE，这有助于避免 IDE 崩溃和模型无响应，特别是在代码量超过 100k LOC 的大型项目中。
   - 相反，一位用户发现 **IDE 内部的 Cursor CLI**（以 WSL 作为终端）简直是“纯垃圾……说真的，不可用”，报告称即使在 64GB 内存和 i7 处理器的配置下，UI 也不流畅。
- **Cursor 订阅调整令人心痛**：9 月 15 日之后，**auto 模式不再无限量**，并会计入每月 20 美元的限额。定价为：Input + Cache Write 每 1M Token $1.25，Output 每 1M Token $6.00，Cache Read 每 1M Token $0.25。
   - 一位用户发现他们很快就消耗完了月度订阅额度，建议 *使用自己的 API Keys，或使用 Claude Code* 可能会更便宜。
- **Clawdbot 安全漏洞曝光**：一位用户分享了关于 **Clawdbot 安全问题** 的链接，报告称暴露的控制面板会导致凭证泄露和账号被劫持。
   - 有推测称，由于潜在的量子解密问题，这可能导致“先存储，后解密”的数据泄露，且该公司因这些问题收到了停制令（cease and desist）。
- **Gemini Vision 将革新前端**：一位用户发现 **Gemini agentic vision** 在视觉任务上正接近 SOTA (State-of-the-Art) 水平，并相信其集成将简化前端开发。
   - 成员们表示迫不及待想看到视觉功能集成到 Agent 中，并认为它优于 `Auto` 工具。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio v0.4 开启无头模式与并行化**：**LM Studio v0.4** 引入了 **headless mode**（无头模式）和 **parallel inference**（并行推理），用户对新功能和翻新的 UI 感到兴奋，详见[此处的完整博客文章](https://lmstudio.ai/blog/0.4.0)。
   - 请注意，应用内更新需要重新安装程序，且部分 UI 元素目前处于 **dev mode**。
- **GLM 3.7 Flash 展现编程潜力**：成员们注意到 **GLM 3.7 Flash** 表现出良好的编程能力，但预计 **GPT OSS 120** 将是更优秀的编程模型，尤其是在 **Q4** 量化版本下。
   - 这表明虽然 **GLM 3.7 Flash** 取得了进步，但可能无法超越现有模型。
- **ROCm 在 LM Studio 运行时环境下运行**：用户发现可以在 **LM Studio** 的运行时设置中启用 **ROCm**，尽管该方法最初对某些用户来说较为隐蔽，正如这个 [Unsloth Reddit 帖子](https://www.reddit.com/r/unsloth/comments/1qpbmrt/you_can_now_run_kimi_k25_locally/)中所讨论的。
   - 这一集成允许用户利用 **ROCm** 来获得潜在的性能提升。
- **Devstral-2 需要高性能 GPU 部署**：成员们讨论了本地运行 **Devstral-2** 的硬件要求，一位用户建议为 24B 版本配置 **48GB GPU**（例如 3090）。
   - 对于 120B 版本，建议采用并行计算或使用带有 **EXL2** 模型格式的 **H200**，因为 GGUF 被认为速度太慢。
- **硬件加速器寻求接入 LM Studio**：一家硬件加速器公司的成员咨询了如何为其硬件添加 **LM Studio backend**，并被引导至 **llama.cpp**。
   - 会中指出 LM Studio 主要是 Element Labs 的闭源项目，并推荐参考 [LM Studio Enterprise](https://lmstudio.ai/enterprise)。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2.5 的定价引发关注**：用户对 **Kimi K2.5** 每月 **$19** 的订阅费展开辩论，部分人认为价格昂贵，并询问是否能达成长期优惠协议。
   - 另一些人建议坚持使用免费层级，认为像 Moonshot AI 这样规模较小的中国公司需要运行像 K2.5 这样的大型模型，因此降低价格的可能性不大。
- **Google AI Studio 训练引发隐私辩论**：针对 **Google** 在 **AI Studio** 和 **Gemini** 应用中**训练并查看对话**的做法出现了担忧，引发了隐私问题。
   - 相反，另一位用户提到他们会**开源自己的项目**，暗示无论如何数据都不可避免地会被包含在训练数据集中。
- **模型选择对决：Kimi K2.5 在 STEM 领域胜出**：用户在从编程到通用问答的任务中对比了 **Kimi K2.5** 与 **Mistral** 及 **Qwen**。
   - 值得注意的是，**Kimi K2.5** 在物理、化学和数学方面拥有最高基准测试成绩（benchmarks），同时在设计和逻辑推理方面也表现出强大性能。
- **Kimi CLI 在速度测试中领先替代方案**：**Kimi CLI** 因其比 *oh-my-opencode* 等工具更快的速度和更高的效率而受到赞赏，特别是在网页分析方面，且消耗的 **token** 更少。
   - 然而，部分人发现该模型的输出质量稍逊一筹，认为有必要进行进一步的对比分析。
- **Agent Swarm 实用性受到质疑**：爱好者们强调了 **Agent Swarm** 配合 Kimi 的深度研究能力，但指出其消耗额度的速度是正常水平的 **3倍**。
   - 其他人对其应用场景仍持怀疑态度，认为需要更明确的使用案例，并对资源消耗保持谨慎。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 订阅被指为骗局**：多位用户反映在自动续费后出现**意外的订阅变更**和**扣费**，一名用户取消了订阅并称其为*骗局 (scam)*。
   - 用户遇到了扣费但未获得服务或无法退款的问题，促使部分用户考虑联系银行或向 FTC 举报。
- **查询限额异常困扰用户**：部分用户报告其 **Pro 订阅**的**查询限制**出现问题，限额降至每小时一次查询。
   - 然而，一些用户的限额随后恢复到了 600 次，一名用户分享了一个[链接](https://www.perplexity.ai/rest/rate-limit/all)用于检查查询限额。
- **图像生成受地区限制？**：用户报告某些地区的**图像生成功能受限**，可能与 **xAI 争议**和欧盟诉讼有关。
   - 建议包括尝试不同的模型或联系支持部门；一位来自印度的用户确认他们受到了该问题的影响。
- **Kimi 2.5 即将登陆 PPLX？**：用户正热切期待 **Kimi 2.5 模型**在 Perplexity 上发布。
   - 推测认为 Perplexity 通常能很快完成更新适配。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT Pro 隐藏了模型魔力？**：成员们辩论 **GPT Pro** 的性能提升是来自更多的 GPU 还是改进的模型，暗示 **OpenAI** 可能出于竞争原因掩盖了真相。
   - 一名成员将 **OpenAI** 的定价策略比作*造假 (fakery)*，将其比作感官印象而非实测价值，类似于股市对 **Tesla** 的看法。
- **DeepSeek 永无止境的“监禁”**：据报道 **DeepSeek** 容易陷入越狱 (jailbreak) 循环，无论后续提示词如何，都会无限重复同样的拒绝信息。
   - 虽然 API 端点表现略好，但原始模型一旦进入这种状态基本上就*废了 (cooked)*。
- **TI-84 获得神经网络移植**：一位成员详细介绍了在 **TI-84 Plus** 计算器上运行神经网络进行拼写检查的过程，并在一个[学术网站](https://hermesoptimus.vercel.app/)上记录了该过程并附带演示视频。
   - 该成员开玩笑说，尽管取得了这一成就，他在 **Claude Code Orchestration** 上的工作仍更具实际用途。
- **MergeMix 论文引发数据混合热潮**：论文《[MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858)》因其对预算有限的开源项目的相关性而受到关注。
   - 该论文探讨了在训练期间优化**数据混合 (data mixtures)**和**模型合并 (model merging)**的技术，可能提供资源高效的策略。
- **Hermes 4 定价：折扣还是欺骗？**：一名成员在订阅 API 前询问 **Hermes 4 系列**模型的折扣价格是否是永久性的，并称其在 RP 和故事创作方面优于 **DeepSeek**。
   - 另一名成员澄清说没有订阅制，只有可能发生变化的额度 (credit) 购买，因此其价值取决于**定价**和**使用情况**。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 3 Pro 字幕生成失误**：用户报告称 **Gemini 3 Pro** 正在捏造与视频音频毫无关系的 .srt 文件。
   - 这种糟糕的表现让那些认为 **Gemini** 被**过度炒作（overhyped）**的用户感到失望。
- **Clawdbot 更名为 Moltbot 是一场骗局**：**Clawdbot**（现更名为 **moltbot**）是一个 **Agent** 系统，它通过来自 Anthropic、Google 和 OpenAI 的 **API keys** 控制你的整个系统（OC），用户被警告不要使用它。
   - 一名用户指出，这是**加密货币圈（crypto bros）窃取信息的巨大骗局**，信息可能通过**提示注入（prompt injection）**被武器化，从而引发重大的安全和隐私担忧。
- **Prism 被认为对科学研究有害**：尽管 **OpenAI** 的目标是通过 **Prism** 推动科学进步，但一位用户表示 **Prism** 对科学研究是有害的。
   - 另一位用户询问了 **Prism** 的 **API** 访问权限，以便使用其他 **AI** 和 **Codex** 编写他们的部分项目。
- **GPT Pro 丢失 ZIP 文件读取功能**：一名用户报告称，此前能够读取和分析 **ZIP 文件** 的 **GPT Pro 5.2**，现在无法找到上传的文件进行分析。
   - 该用户正在询问其他人是否也遇到了同样的问题，或者是否有任何见解。
- **通过避免明暗对比（Chiaroscuro）来屏蔽黑白图像**：用户讨论了与**明暗对比（Chiaroscuro）效果**相关的图像生成问题，并建议如果遇到不想要的**黑白图像**，可以在提示词中加入“*请避免明暗对比（Please avoid Chiaroscuro）*”。
   - **Chiaroscuro** 是指光影之间的强烈对比，通常是影响整个构图的大胆对比。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Decart 招募旧金山性能工程师**：Decart 正在寻求负责低延迟 **Kernel**、实时视频/世界模型以及加速器（如 ReInvent [视频](https://www.youtube.com/watch?v=K49S79wOGl8)中展示的 **Trainium 3**）和他们新的 **Lucy 2** 自回归视频模型（[技术报告](https://x.com/DecartAI/status/2016134190509498740)）的工程师。
   - 他们还与 **GPU Mode** 共同赞助了一个针对自回归扩散模型的 **Kernel** 挑战赛，并鼓励感兴趣的人士将性能相关工作发送至 heba@decart.ai。
- **INT4 QAT RL 模型 Rollout**：一名成员分享了一个 **GitHub** 仓库链接，该仓库专注于通过端到端的 **INT4 QAT RL** 实践，将 **1TB 模型 Rollout** 压缩到单张 **H200** 中：[GitHub 仓库](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme-en.md)。
   - 该仓库提供了与 **INT4 QAT RL** 实现相关的资源和文档，优化了大型模型的 **Rollout**。
- **Transformers 和 PyTorch 遭遇升级故障**：在升级 **transformers** 和 **pytorch** 后，一名成员报告了 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` 错误。
   - 降级到 **transformers 4.57.3** 修复了该问题；其他人也遇到了类似问题，这些问题在 [pytorch issue](https://github.com/pytorch/pytorch/issues/127176) 和 [optimi issue](https://github.com/warner-benjamin/optimi/issues/8) 中有所讨论。
- **交互式数值工具出现**：一名成员对量化从业者尚未创建用于探索数值的交互式工具表示惊讶，并提到 [captum](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.html) 是一个可能的工具。
   - 该成员感叹目前的模型调试工具缺乏适当的 **UI/UX**，“*检查哪个电路不稳定，哪一层产生了大量异常值（outlier），诸如此类简单的事情*”。
- **DGX 的主导性内存带宽**：**DGX** 和 **5090** 的指令集相似，但 **DGX** 在全速 **fp32** 累加方面表现出色（如 **Blackwell PRO**），其关键区别在于 **1.8TB/s** 的内存带宽。
   - 这与 **5090** 的 **300 GB/s** 形成鲜明对比，强调了有效利用 **L2 cache** 以最大化 **DGX** 潜力的重要性。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **编程进入 Agent 时代**：Andrej Karpathy 预测，到 2026 年，**80% 的编程**将由 Agent 驱动，并强调了 LLM 的韧性和目标设定能力；更多见解请看[这里](https://xcancel.com/karpathy/status/2015883857489522876)。
   - Karpathy 还警告要警惕潜在的“slop（垃圾信息）”和过度设计，因此情况可能并非全是乐观的。
- **OpenAI 的 Prism 为科学家闪耀**：OpenAI 推出了 **Prism**，这是一个由 **GPT-5.2** 驱动的免费研究工作空间，拥有个人 ChatGPT 账户的用户可通过网页访问；点击[这里](https://xcancel.com/openai/status/2016209462621831448?s=46&t=eWVlK1PU8XfB6f402GJJ9g)开始使用。
   - 该工具旨在为科学家提供用于研究目的的高级 AI 能力。
- **Trinity Large 发布**：Prime Intellect、Arcee AI 和 Datology 推出了 **Trinity Large**，这是一个拥有 **400B 参数的 Mixture of Experts 模型**，仅使用 **13B 激活参数**；更多信息见[这里](https://xcancel.com/primeintellect/status/2016280792037785624?s=46)。
   - 该模型旨在提供高性能的同时保持效率。
- **Cursor 索引代码库**：Cursor 宣布针对大型代码库实现更快的索引速度，并改进了语义搜索，承诺带来性能提升；阅读更多请点击[这里](https://xcancel.com/cursor_ai/status/2016202243499073768?s=46)。
   - 语义搜索和改进的索引旨在提供更高效的代码导航。
- **播客将焦点转向科学**：Latent Space 推出了其第二个播客“Science”（[播客链接](https://www.latent.space/p/science)），由 <@713947182167883897> 和 <@348078436058660866> 主持。
   - 关于新播客“Science”的讨论已移至专用频道。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kimi 2.5 模型在本地击败 GPT5**：据报道，新的 **Kimi 2.5** 模型表现优于 **GPT5**，可通过 [HuggingFace](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) 以及 [Fireworks](https://www.google.com/aclk?sa=L&ai=DChsSEwiCz-j3iK2SAxUFVX8AHT5cBPkYACICCAEQABoCb2E&co=1&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoL9Ani52eByT53nVhnOxG_76F9QllEx50YhK_yfQYsD5bH3ov1pAqwaAl2XEALw_wcB&cid=CAASugHkaDm-Aokq5n3lAlzNAI-Ihc6SdblOJ-BiATzwnaZwDVhVBl3B2U5kGq4mAYjN4wQ992LlqWX5NQ6HksDrhSatp0QEfb7_rWMS_u7_GTCuCkp3YH9fANMaJqDgFvuA6u1bwvl4pJ80zvbUhIFPk7Nrqdpx2PDnsBRncgM3-d1UDhFM-tN117MrOXLWnhycCaPax24T8meZIe-9I2cM5rpAf16KucPGZwg7ixTssRCB7X8RP3B_G4vUCfE&cce=2&sig=AOD64_2SRpHfWjuW4kJawyiTyzrGbKZybQ&q&adurl&ved=2ahUKEwiiteP3iK2SAxV85skDHfklKyoQ0Qx6BAgLEAE) 等网站在本地访问。
   - 成员们正在寻求在 **Zed** 中使用的本地 Agent 推荐，并对在 llama.cpp 上 Q4 量化的 **GLM-4.7-Flash** 表示不满，建议将 **kimi** 和 **qwencoders 30b q4** 作为替代方案。
- **C++ 爱好者主张 AI Agent 的统治地位**：一位成员认为，由于 Python Agent 存在臃肿问题，构建 AI Agent 永远是 **C++** 统治，并推荐使用 **fastwhisper.cpp** 进行 STT，在 LlamaCPP 中使用 **Qwen embeddings** 进行 RAG，以及使用 **LFM2.5vl** 作为 VLM。
   - 这引发了关于 STT (**fastwhisper.cpp**)、RAG (LlamaCPP 中的 **Qwen embeddings**) 和 VLM (**LFM2.5vl**) 的讨论。
- **视觉模型消除 JPEG 伪影**：发布了一个视觉模型，该模型使用独特的设计去除由 **JPEG 压缩**产生的伪影，该设计没有 Batch Norm，训练后没有激活层，并使用 Operator 层代替 Convolutional 层。
   - 该模型的架构侧重于通过**宽度**而非深度来获得准确性。
- **RemnantInstruct-8B：SLERP 合并平衡创意与事实**：**RemnantInstruct-8B** 是一个 [SLERP merge](https://huggingface.co/anthonym21/RemnantInstruct-8B-GGUF)，它将创意微调模型 (**allura-org/remnant-qwen3-8b**) 与其基础模型 (**Qwen/Qwen3-8B**) 重新组合，以平衡叙事能力与事实准确性。
   - 合并策略在自注意力层倾向于创意微调模型，在 MLP 层倾向于基础模型，目标是保留 **Qwen3** 的思考模式。
- **VLM 拥抱量子计算**：一位成员开源了他们的本科论文，内容是专门针对量子计算和使用 **Qiskit** 编码的 **vision-language models**，包括 [数据集](https://huggingface.co/datasets/samuellimabraz/quantum-assistant)、[模型](https://huggingface.co/collections/samuellimabraz/quantum-assistant)、[代码](https://github.com/samuellimabraz/quantum-assistant) 和 [Demo](https://huggingface.co/spaces/samuellimabraz/quantum-assistant)。
   - 该论文探讨了如何调整 VLM 以辅助量子计算任务和编程。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Transformers Can Parameterize Vector Fields**: 一位成员认为 **transformers** 可以用于 **flow matching** 这一训练目标，通过 **patch embedding** 编码 patch 位置来参数化连续扩散（continuous diffusion）的向量场。
   - 其他成员同意扩散模型和 **flow matching** 在数学上是相似的，并引用了 [这篇 ArXiv 论文](https://arxiv.org/abs/2305.03486)。
- **Diffusion Models are not Better than Autoregression**: 一位成员认为扩散模型优于自回归模型的观点是错误的，并强调了架构和扩展性方面的局限性，链接到了[这篇关于重复上下文的论文](https://arxiv.org/abs/2512.14982)。
   - 他们指出，诸如重复上下文或以非因果方式重新编码序列等改进可以弥补差距，克服当前 **LLMs** 设计中的局限。
- **ChatGPT Wrappers Flourish, Value Questioned**: 成员们观察到大多数新工具仅仅是 **ChatGPT wrappers**（套壳），引发了对其实际价值以及骗子轻松创建套壳工具的质疑，并参考了 **Clawdbot scam** 诈骗案。
   - 有人建议，这些套壳工具对于演示使用场景是必要的，因为它们能让人们更容易理解如何应用模型。
- **AI Coding Tools Won't Replace True Skill**: 尽管 **AI coding tools** 兴起，成员们认为编程能力可以重新习得，并指向了一篇关于 [Trinity Large 的博客文章](https://www.arcee.ai/blog/trinity-large)，并补充说 AI 快速生成的代码可能会阻碍真正的理解。
   - 他们指出，来自 **LLM** 的糟糕实现不再像以前那样被重视，因为生成它的心力和时间成本非常低。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD Emulator Exposes Debug Printing**: 新的 AMD 模拟器 (**AMD=1 MOCKGPU=1**) 现在支持调试打印。根据链接中的[截图](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png?ex=697b686e&is=697a16ee&hm=485c88290bbec976b6b7ab93aed07b21a6a2ec8ba8b28806e14630c00b972b3c&)，设置 **DEBUG=3** 会打印所有编译后的指令，设置 **DEBUG=6** 则会在运行时打印指令。
   - 这一增强功能有助于在模拟器环境中直接对编译代码进行更深入的调试和分析。
- **Github Actions Speed Boost via Optimization**: 讨论集中在通过强调代码优化来加速 **GitHub Actions**，而不是仅仅依赖更快的硬件或外部资源。
   - 共识是优先考虑以“正确”的方式做事，而不是只提升表面指标、可能产生技术债务的快速修复。
- **MULACC Fusion Receives a Fix**: 提交了一个修复方案来增强 `decompositions.py`，增加了一个融合模式 (**x << n) + c → MULACC(x, 2^n, c)**，专门针对带有 2 的幂常量的整数 **MULACC**，详见 [PR 14387](https://github.com/tinygrad/tinygrad/pull/14387)。
   - 此项调整旨在完善融合过程，可能提高某些算术运算的效率。
- **Egraphs Considered for Universal Fixes**: 探讨了使用 **egraphs** 以通用方式解决问题的潜力，强调了简洁性的重要性。
   - 还建议为重写操作标记来源，以便清晰记录重写过程中创建的等价关系。
- **Mac MetalCompiler Improvements on the Horizon**: 对 Mac 上 **MetalCompiler** 的 hack 修复改进即将到来，重点是减少行数和提高可读性的优化与清理。
   - 目标是使 **MetalCompiler** 更具可维护性和效率，使在 Mac 平台上工作的开发者受益。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **GTK 绑定自动生成**：根据 [Modular 论坛](https://forum.modular.com/t/february-community-meeting/2646)的消息，**Hammad Ali** 将于太平洋时间 2 月 2 日上午 10 点在 **Modular 社区会议**上展示为 **Mojo** 自动生成的 **GTK 绑定**。
   - 演示将详细介绍 **GTK 绑定**是如何自动生成的，这可能会提升使用 **Mojo** 创建 **GUI** 的便利性。
- **Mojo 的性能实力**：**Tatiana Melnichenko** 将在 2 月的社区会议上分享 **H100/MI300A** 上的内存受限带宽结果和计算受限差距，并将 **Mojo 与 CUDA/HIP** 进行对比。
   - 这场演讲应该能让人们深入了解 **Mojo** 相对于成熟 **GPU** 编程模型的性能特征。
- **macOS Gatekeeper 造成干扰**：成员们怀疑 macOS 上首次运行与后续运行之间的性能差异是由 **Gatekeeper 的信任机制（trust dance）**引起的。
   - 清除隔离 `xattr` 属性或进行临时代码签名（ad-hoc codesigning）可以缓解此问题，成员们想知道 `mojo build` 中的代码签名步骤是否能完全消除这一现象。
- **`out` 参数优于 NVRO**：Mojo 中的 `out` 参数指明了函数返回值的存放位置，可作为 **命名值返回优化 (NVRO)** 的替代方案。
   - 成员们声称，这为返回值的去向提供了保证，而不是依赖于编译器的优化。
- **Qwen3 Embedding 模型精度提升**：一名成员请求审查其 [Qwen3 embedding 模型的 PR](https://github.com/modular/modular/pull/5823)，并指出该修复对于获得更好的准确性至关重要。
   - 另一名成员回复称，新的修复可能不会被纳入即将发布的版本，但会在 nightlies 版本中提供，此处提供了一个[单行修复方案](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal)。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 额度消耗降低**：一位用户注意到 **Manus** 在完成相同质量的工作时似乎使用了更少的额度（credits），询问额度使用是否得到了优化。
   - 目前还没有关于 **Manus** 额度消耗算法潜在变化的进一步细节或确认。
- **云浏览器引发难题**：一位用户在使用**云浏览器**时遇到了问题，收到错误消息称*服务器不可用*且网站无法加载。
   - **Manus** 支持团队请求用户通过私信（DM）提供其电子邮件、会话链接和 **Manus User ID**，以便进一步调查该问题。
- **AI 工程师精通 LLM 系统**：一位 **AI + 全栈工程师**介绍了自己，强调了其在 **LLM 系统、自主 Agent、工作流自动化和多模态 AI** 方面的专业知识。
   - 他们分享了自己的核心技能，如 [DSPy](https://dsppy.ai/)、[LangChain](https://www.langchain.com/)、[AutoGen](https://microsoft.github.io/autogen/) 和 [CrewAI](https://www.crewai.com/)。
- **社区渴望跨对话上下文**：一位用户建议让 **Manus** 能够访问来自其他对话的上下文，这*将是一个游戏规则改变者*，表明用户希望 AI 的响应具有更强的上下文感知能力。
   - 该成员指出需要跨频道共享上下文，以提供更复杂的响应。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **寻找 Prompt 优化器专家**：成员们询问了使用 **prompt 优化器**的相关经验，特别是是否有人在 dspy 模块中使用过 **Skills**。
   - 讨论表明，人们有兴趣利用这些工具来改进 prompt 工程工作流。
- **llmlingua 链接分享**：在关于 **prompt 优化器**的讨论中，一位成员分享了 [llmlingua.com](https://llmlingua.com/) 的链接。
   - 这表明 llmlingua 可能是那些探索 prompt 优化策略的人的相关工具。
- **DSPy ReAct Agent 渴望 Skills 集成**：一位成员询问如何将 **Claude code skills**（定义为带有相关 .py 脚本的 .md 文件）集成到 **DSPy ReAct agent** 中。
   - 该成员正在寻求一种让 DSPy ReAct agent 能够有效利用 Claude 代码技能的解决方案。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Kimi 2.5 定价比 GLM 4.7 更高**：新款 **Kimi 2.5** 模型的定价为 **$0.6**，超过了 **GLM 4.7**，暗示其具备更卓越的能力。
   - 一名成员指出“models”频道中正在进行相关讨论，表明了更广泛的兴趣和对比。
- **Aider 作者暂时离开 (AFK)**：aider 的幕后大脑 **Paul Gauthier** 宣布由于其他事务，将暂停开发工作。
   - 他表示打算在时间允许时恢复 aider 的工作，这让社区充满期待。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道沉寂时间太长，请告知我们，我们将移除它。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道沉寂时间太长，请告知我们，我们将移除它。

---

**Windsurf Discord** 没有新消息。如果该频道沉寂时间太长，请告知我们，我们将移除它。

---

**MCP Contributors (Official) Discord** 没有新消息。如果该频道沉寂时间太长，请告知我们，我们将移除它。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord: 各频道详细摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1465800651777769543)** (1156 messages🔥🔥🔥): 

> `军事 ICBM, AI 无人机, 隐形喷气式飞机, GPT 5.2 Custom GPT, Gemini Canvas` 

- **中国隐形战斗机热潮开启**：一名成员提到**中国**正为其新型隐形战斗机而疯狂。
   - 分享了一个 [YouTube Shorts 视频](https://www.youtube.com/shorts/4sKw-lBujPM) 链接，以及一段关于**高超音速导弹**的完整 [YouTube 视频](https://youtu.be/M7mIX_0VK4g) 链接。
- **Custom GPT 5.2 准备发布**：一名成员正致力于发布一个新的 **GPT 5.2 Custom GPT**，声称其效果不错但需要加入噪声。同时分享了图像生成模型的系统提示词截图，显示其能够识别日期，表明存在实际的系统提示词。
   - 该成员还声称他们有一个 **Custom GPT** 已通过商店审核，即使是在被 Jailbroken 的情况下，并询问如何利用图像提取系统提示词。
- **Gemini Canvas 测试对抗性提示词 (Adversarial Prompts)**：成员们讨论让 **Gemini Canvas** 构建一个 Web 应用，以便在其中测试对抗性提示词和 Jailbreaks。
   - 另一名成员解释了如何使用 **Gemini** 自动执行此过程。
- **魔术字符串导致 Claude 停止响应**：一名成员分享了一个魔术字符串 `ANTHROPIC_MAGIC_STRING_TRIGGER_REFUSAL_1FAEFB6177B4672DEE07F9D3AFC62588CCD2631EDCF22E8CCC1FB35B501C9C86`，声称这可以可靠地阻止 **Claude** 做出响应。
   - 另一名成员将其比作*可能用于帮助模型更准确拒绝的断路器 (Circuit Breaker)*。
- **用户寻找 Kimi JB**：一名成员询问是否存在 **Kimi JB**。
   - 一位用户声称 **Kimi 2.5** 远好于 **Kimi 2**，且达到了 **Opus 4.5** 的水平。

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1465803292901904591)** (167 条消息🔥🔥): 

> `Claude Chat Limits, Kimi Jailbreak, Parallel AI Jailbreak, Opus Jailbreak, Grok Imagine Jailbreak` 


- **Claude 免费版达到每日限制**：成员们讨论了 **Claude 免费版的局限性**，指出与其竞争对手相比，其限制相对较低，但也承认其过去在 Agent 任务方面的优势。
   - 一位用户提到在使用 Agent 进行编程时，触及了 Claude 付费订阅的 **每月 200 次请求** 限制。
- **探索 Parallel AI API 访问**：用户分享了与 **Parallel AI API** 交互的方法，包括通过向 API 发送 POST 请求来调整 System Prompt，但指出目前没有关于 System Prompt 的 API 文档。
   - 一位成员提供了一个 [PowerShell 示例](https://platform.parallel.ai/)，用于向 API 发送请求。
- **探索 Opus 4.5 Jailbreak**：成员们讨论了对 **Opus 4.5** 进行 Jailbreak 的可能性，一位用户声称这很容易，并建议使用 System Prompt 或 ENI。
   - 另一位用户表示怀疑，质疑考虑到 **Opus** 是他们最高端的 LLM，这怎么可能实现。
- **获取免费模型访问权限**：一位成员分享了 [X 上的链接](https://x.com/Exocija/status/2016502660883415422) 以免费访问模型，并提供了一个 [PRIMETALK 上下文文件](https://discord.com/channels/1105891499641684019/1228043845967544380/1466113637541347348)，其中包含模型兼容性和使用说明。
   - 文中指出，该系统可用于大多数现代 AI 模型，但其行为和稳定性很大程度上取决于上下文容量和聊天窗口大小。
- **Gemini Prompt Injection 指南**：一位成员描述了如何对 Gemini 进行 Prompt Injection，其中包括向聊天界面逐条发送一系列会话轮次。
   - 如果第一轮拒绝了 Prompt，用户被指示访问 **gemini.google.com/saved-info** 并在 *Remember:* 之后添加内容以绕过限制。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1465877209821610156)** (8 条消息🔥): 

> `Malicious Prompt Datasets, HackAPrompt, PlinyAnthropic, Deterministic Stack Based VM, Free Model Access` 


- **恶意 Prompt 数据集难以寻觅**：一位成员正在寻找具有明确分类的**恶意 Prompt**数据集，用于 LLM Jailbreak 和 Prompt Injection 研究，但另一位成员回应称此类**免费数据集**很难找到。
   - 他们补充说，用户可能需要生成自己的 Prompt 并由人工标注员进行标注。
- **HackAPrompt 拦截发送者**：一位成员提到 **HackAPrompt x PlinyAnthropic** 很久以前就标记了他们，并且直接拦截了他们所有的发送请求，甚至*根本不让发送*。
- **带有 REPL 的递归模拟内核**：一位成员询问是否可以在模型的底层生成一个**确定性栈式 VM**，*就像某种带有 REPL 的可启动递归模拟内核 (Recursive Simulation Kernel)*。
- **通过 X 获取免费模型访问**：一位成员提供了 [X 链接](https://x.com/Exocija/status/2016502660883415422)，介绍如何免费访问模型。
- **寻求 Red Teaming 路径**：一位成员询问如何*进入 Red Teaming 领域*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1465799537892266220)** (1038 条消息🔥🔥🔥): 

> `Arena new UI, Arena rebrand, Arena captcha issues, LMArena name change` 


- **Arena 品牌重塑，用户想要停止（STOP）按钮和旧版表情符号**：用户请求添加**停止按钮**，并对 **LMArena 更名为 Arena** 后 Google **captcha**（验证码）难以通过的问题表示担忧。几位用户表示他们**讨厌这种验证码**。
   - 一些用户请求恢复旧版的表情符号、贴纸和功能，而另一些人则接受了重新设计，并称 *“LMArena 的谐音几乎就是‘一个时代的结束’（end of an era）”*。
- **Arena 的新外观是 Claude 的克隆版！**：许多用户立即注意到 **LMArena 更名为 Arena**，并觉得它是 **Claude UI 的克隆版**，而其他成员则喜欢这个新外观。官方分享了一篇 [博客文章](https://arena.ai/blog/lmarena-is-now-arena/) 来解释这一变化。
   - 成员们指出了一些 UI 问题，如**字体**和网站文本的可读性，以及一些**缺失的功能**。
- **无法登录？试试恢复按钮！**：一名遇到登录问题的成员分享了一个 [恢复按钮（recover button）](https://cdn.discordapp.com/attachments/1340554757827461211/1466134595035467829/Hdbd.png?ex=697ba3be&is=697a523e&hm=2be3961be4c941479f9ec51709c5eb6af5ea9c79ad3918eb6a15a964ec9fe720&) 的截图，点击该按钮可以重新登录更新后的 Arena，从而避免再次输入登录详情。
   - 另一名成员也提到了一个 [公告视频](https://youtu.be/TNoAlMv4Eg8?si=d86SArLb6yQ8sdLE)。
- **LMArena = Language Model Arena**：一些成员对 **LMArena** 中 **LM** 的含义开了玩笑，其中一人解释说它代表 **Language Model Arena**（语言模型竞技场）。另一名成员在 [此处](https://cdn.discordapp.com/attachments/1340554757827461211/1466200772483092664/image.png?ex=697be160&is=697a8fe0&hm=5039f80e715df82d41633e75d9976fd88203ce8a3f8db5fc97d4bf29672c74fc&) 确认了这一点。
   - 团队承认，随着平台不再局限于语言模型，名称正变得更加通用，而且它 [此前是 LMSYS 的一部分](https://lmsys.org/)。
- **幻觉迷雾，验证码迷宫**：用户报告验证码存在持续问题，几乎每次尝试都失败，而模型仍持续产生**幻觉（hallucinate）**。
   - 一名用户提供了诸如*重新登录账号*的故障排除步骤，另一名用户报告称，需要*关闭所有扩展程序*才能通过验证码。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1465826408667287664)** (3 条消息): 

> `LMArena 90 second AI videos, Text Arena Leaderboard Update, LMArena rebrand to Arena` 


- **Arena 在 90 秒内上传了 AI 视频！**：**Arena**（原 LMArena）在其 [Youtube 频道](https://www.youtube.com/watch?v=0hCI2XEh0x0) 上传了一段名为 `Better AI videos in under 90 seconds` 的视频。
- **Kimi K2.5 Thinking 登顶 Text Arena！**：[Text Arena 排行榜](https://lmarena.ai/leaderboard/text) 已更新，`Kimi K2.5 Thinking` 现在位列**开源模型第 1 名**，**总榜第 15 名**。
   - `Kimi K2.5 Thinking` 在编程（Coding）中排名 **#7**，在指令遵循（Instruction Following）中排名 **#7**，在困难提示词（Hard Prompts）中排名 **#14**，并已添加到 [Code Arena](https://lmarena.ai/?chat-modality=code)。
- **LMArena 更名为 Arena！**：**LMArena** 宣布更名为 **Arena**，以契合其衡量和推进 AI 前沿的科学使命，现在可通过 [arena.ai](https://arena.ai/) 访问。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1465798521574658291)** (299 messages🔥🔥): 

> `GPU utilization, Kimi 2.5 Q3 Release, Oracle RAG and Fine-tuning, Unsloth's Transformers/MoE Update, Chinese text in LLMs` 


- **Batch Size 提升 GPU 利用率**：成员们讨论了为了获得理想的 GPU 利用率，应该*增加 Batch Size* 直到利用率改善，并在其与 **GA (Genetic Algorithms)** 的潜在收益之间取得平衡。
   - 一位成员询问 Unsloth 是否会发布 **Kimi 2.5** 的 **Q3** 版本，并对潜在的精度损失表示担忧，凸显了社区对优化模型发布的关注。
- **关于 Oracle 是否为顶尖公司的辩论**：一位成员询问 **Oracle** 在 **RAG (Retrieval-Augmented Generation)** 和 Fine-tuning 技术方面是否属于顶尖公司，引发了一些讨论。
   - 另一位成员以 *"What 😅"* 回应，随后补充说 **OCI (Oracle Cloud Infrastructure)** 确实有一些不错的工具，表明了对 Oracle 在这些领域能力的看法不一。
- **Arcee 的 Trinity 模型成本达 35 万美元**：一位成员分享了新的 **Arcee 模型** 图片，指出 Pretraining 成本约为 **35 万美元**，并链接到了 [Trinity Large 技术报告](https://github.com/arcee-ai/trinity-large-tech-report/blob/main/Arcee%20Trinity%20Large.pdf)。
   - 他们还提到 **GLM 4.7** 是一个 **358B** 参数的模型，比 **GLM 4.5** 大得多，但它不是 **Base model**，因此对比 Benchmarks 的意义不大。
- **LLM 会说中文？**：一位成员注意到从 **OpenAI** 和 **Anthropic** 模型中收到了随机的中文文本，即使提示词是纯英文的，这引发了关于潜在数据污染或内在语言相似性的讨论。
   - 另一位成员建议，如果不同语言之间的 Token 具有相似的含义，引入一种语言可能会因为 Token 概率和相似性导致模型偏向于该语言。
- **Gemini API 仍可被越狱**：成员们讨论了 **Gemini** 的输出过滤，一位成员指出虽然 **Gemini** 对输出进行了严格过滤（特别是针对企业/政府环境），但其 **API** 可以通过操作产生几乎任何内容。
   - 一位成员提到在 Google 黑客松期间使用 API，通过在 System Prompt 中设置，让语音模型说脏话。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1465854567571783902)** (4 messages): 

> `Edge AI Engineer, Quantization and LoRA FT` 


- **Edge AI 工程师加入**：一位名为 Josh 的高级 Edge AI 工程师介绍了自己，详细说明了他在 **DoD（美国国防部）和公共部门** 构建真实的离线 Agent 的 6 年经验。
   - 他补充说，他出于兴趣制作 Quantization 模型，并专门使用 **Unsloth** 进行本地 Quantization 和 LoRA Fine-tuning。
- **新成员打招呼 "HelloHi"**：一位名为 Josh 的高级 Edge AI 工程师介绍了自己。
   - 他分享了使用 Unsloth 进行 Quantization 和 LoRA Fine-tuning 的热情。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1465798627963179160)** (969 messages🔥🔥🔥): 

> `Personaplex, GLM 4.7, GGUF, Model Quantization, Vendor Lock-in` 


- **Personaplex 人格特征**: 成员们讨论了 **Personaplex** 在强制执行人格方面的局限性，以及在经过几次迭代后倾向于变得像“糟糕的 AI 播客”的趋势。
   - 一位成员提到，他们无法访问存储的录音通话，而这些通话本可以完美地用于训练 **Persona Plex**。
- **GLM 4.7 Flash 性能讨论**: 一位用户询问是否有人尝试过 [GLM-4.7-Flash-REAP-23B-A3B-GGUF 模型](https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF)，另一位用户回答说 REAP 模型通常效果不是很好，建议使用较低的量化 (quantization)。
   - 其他人就 [GLM 4.7 Flash 模型](https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF) 的性能和见解发表了看法，在推理、效率和关联信息的能力方面将其与 **GPT-OSS-120B** 和 **Kimi** 进行了比较。
- **GGUF 安全性考量**: 一位成员询问了关于 **GGUFs** 潜在不安全性的资源，特别是如果恶意行为者介入的情况。
   - 然而，另一位成员表示 *我不熟悉那个，我想你可能把我跟别人搞混了*，所以没有进一步讨论。
- **AI 模型幻觉观察**: 一位成员注意到他们的 **3b llama** 模型做出了 *令人毛骨悚然的假设，即在没有提示的情况下认为它是根据我的声音训练的*，引发了关于 LLM 幻觉以及它们缺乏对训练或状态感知的讨论。
   - 一位成员推荐了 [这段关于 AI 幻觉的 YouTube 视频](https://youtu.be/wjZofJX0v4M?si=A4rHzAh9qJjls9bm) 作为该主题的入门。
- **供应商锁定 (Vendor Lock-in) 的诱惑**: 小组讨论了一个假设场景，即 Token 价格大幅上涨，触及了供应商锁定 (Vendor Lock-in) 的概念。
   - 有人提到 Nvidia 和 Amazon 也在采用 *Vendor Lock-in* 策略，并称其为 *软件锁定，基本上就是 Nvidia 正在做的，Amazon 也是（我认为）*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1465801085774987430)** (77 messages🔥🔥): 

> `Unsloth container setup errors on encrypted Runpod, GLM-4.7 tool calling issues, CUDA version issues with GLM 4.7 on Blackwell B200s, Multi-GPU training problems with Qwen3 model on Modal, Catastrophic forgetting after finetuning` 


- **Runpod 设置遭遇权限问题**: 一位成员在加密的 Runpod 上设置 Unsloth 容器时遇到了 “permission denied” 错误，暗示在容器创建过程中存在 [卷权限 (volume permissions)](https://link.to/volume-permissions) 问题。
   - 另一位成员建议，Runpod 试图修改容器结构，这并非预期行为，因此建议使用 [官方 Docker 容器镜像](https://hub.docker.com/r/unsloth/unsloth) 以避免此类烦恼。
- **GLM-4.7 工具调用难题**: 一位成员在尝试根据 [官方 Unsloth 文档](https://unsloth.ai/docs/models/glm-4.7-flash#tool-calling-with-glm-4.7-flash) 让 **GLM-4.7** 调用工具时寻求帮助。
   - 讨论内容包括需要对参数使用 `json.loads`，以及在通用工具调用的 `res.choices[0].message` 结构中识别 `tool_calls`。
- **Blackwell B200s 应对 CUDA 冲突**: 一位成员报告其 **B200** 上的 **CUDA 12.8** 驱动程序与 **GLM 4.7** 所需的 **CUDA 13.x** 不兼容，需要升级 CUDA 并重新安装依赖项以运行其 **vllm** 服务器。
   - 有人建议使用 `--torch-backend=auto` 和 CUDA 12.9 每夜构建版 URL 强制重新安装 **vllm**，以便在 CUDA 12.8 上运行 **GLM 4.7**，但由于 **Ada Lovelace GPU** 不兼容，需要移除 `--kv-cache-dtype fp8`。
- **Modal 多 GPU 故障频发**: 一位成员在 **Modal** 上使用 3 个 GPU 训练 **Qwen3** 模型时面临问题，由于不正确的 `device_map` 配置遇到 “ValueError”，并出现 `prepare_device_map` 的导入错误。
   - 结果发现，由于与 **PyTorch 2.4.1** 不兼容，训练设置已从 **Unsloth** 转向 **transformers + PEFT** 方案，以获得更好的稳定性。
- **微调模型遗忘基础知识**: 一位成员描述了在微调 (finetuned) 模型中经历的灾难性遗忘 (catastrophic forgetting)，模型在处理新信息方面表现出色，但遗忘了先前的知识，怀疑是过拟合 (overfitting) 问题。
   - 缓解建议包括降低 **LoRA rank, LR**，减少 steps/epochs，混入更多通用数据，以及针对更少的层进行训练。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1465831343643557958)** (9 messages🔥): 

> `KL Divergence, Mode Collapse, DeepSeek mHC residual preservation, Context Distillation` 


- **KL Divergence 初始值引发辩论**：一位成员询问在将 **SFT model** 加载为 **ref_model** 时，理想的初始 **KL divergence** 应该是多少。
   - 他们期望最初看到的散度为零，并引用了 [这篇 2026 年的论文](https://arxiv.org/abs/2601.09954)。
- **Mode Collapse 导致方差缺失**：一位成员报告经历了 **Mode Collapse**，导致回复之间的差异很小并放大了错误。
   - 他们表示：“现在它答对的题目多得多，然而，那些它答错的题目就错得很彻底，因为几乎没有方差。”
- **预测 DeepSeek 的 mHC Residual Preservation**：一位成员推测 **DeepSeek** 在 **mHC residual preservation** 方面会有相关的见解。
   - 未提供更多信息。
- **RL 研究人员重新发现 Context Distillation**：一位成员讽刺地指出，**RL** 研究人员似乎正在重新发现 **Context Distillation**。
   - 未提供更多信息。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1465840851463700532)** (2 messages): 

> `Arcee Trinity Large Preview, Mixture-of-Experts, Open Weights` 


- **Arcee Trinity Large Preview 发布！**：Arcee 发布了其首个前沿规模的开源权重模型 [Trinity-Large-Preview](https://openrouter.ai/arcee-ai/trinity-large-preview:free)，作为一个可直接聊天的变体，在限定时间内免费提供。
   - [X](https://x.com/OpenRouterAI/status/2016280059527757995?s=20) 上的公告强调，这是一个 **400B 参数的稀疏 Mixture-of-Experts** 模型，但每个 token 只有 **13B 激活参数**。
- **Arcee 以效率为中心的架构**：Arcee 的 **Trinity-Large-Preview** 模型使用 **256 个专家 (experts)**，每个 token 激活 **4 个** (1.56% 路由)。
   - 该模型针对效率而非密集规模进行了优化，并具有采用宽松许可的开源权重。
- **Lucas Atkins 正在直播！**：Arcee AI 的 CTO Lucas Atkins 现在正在直播！
   - 立即观看 [Youtube 直播](https://youtube.com/live/3XSdqHY0kNk?feature=share)！


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

runvnc: 我更新了 https://cyberpad.site 以包含一些免费额度。
  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1465799162288148601)** (468 messages🔥🔥🔥): 

> `Image Generation Models, Refund Delays, Context Caching Pricing, OpenRouter API Issues, Model Training From Scratch` 


- **图像模型未生成图像**：用户报告称，一些被标记为具有图像输出模态的图像模型（[google/gemini-3-pro-image-preview](https://openrouter.ai/models?fmt=table&order=most-popular&output_modalities=image&input_modalities=image)、**GPT-5 Image Mini**、**GPT-5 Image**、**Gemini 2.5 Flash Image**）并未生成图像，而像 **Gemini 2.5 flash** 这样的模型则表现为断断续续工作。
- **讨论模型对 Response Format 的支持**：用户讨论了哪些模型支持 `response_format`，列出如 **Gemini 3 Flash Preview**、**Gemini 2.5 Flash Lite Preview**、**Seed 1.6**、**GLM-4.6v** 和 **Grok 4.1-fast** 是可以正常工作的，同时指出 **Mistral** 在其官方 API 上支持 `response_format` 但在 OpenRouter 上不支持。
   - 一位成员指出：“Gemini 2.5 flash 对我来说有效，但有时我需要使用一些提示词技巧。”
- **OpenRouter API 经历宕机**：用户报告在 OpenRouter 上遇到网络错误和模型无法工作的问题，一些人遇到 “HTTP 401: User not found” 错误，另一些人则特别在 **香港** 地区遇到问题。
   - 一位用户提到：“OpenRouter 现在挂了吗，还是只有我这样？几乎所有的模型对我都不起作用，都只显示网络错误。”
- **用户讨论使用 OpenRouter 的 OCR 解决方案**：成员们讨论了使用 **Gemini Flash** 模型进行 OCR，其中一人建议为了保持一致性可以训练自定义的 **Azure/AWS OCR 模型**。
   - 一位用户提到：“使用 Gemini Flash 模型可以走得很远，这取决于你是需要提取数据还是进行解析。”
- **OpenRouter 用户等待逾期已久的退款**：用户报告退款存在延迟且缺乏沟通，有些人从 1 月初就开始等待，并提交了多个支持工单。
   - 一位用户表示：“说认真的，@OpenRouter 团队 —— 我很想知道：退款的实际时间表是什么？为什么这么多人都在同一条船上？是否有一个真正有效的状态更新系统？”


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1465815588814454865)** (15 messages🔥): 

> `Agentic Vision Gemini 3 Flash, OpenAI 的图像能力, OpenRouter Show, PRISM` 


- ****Agentic Vision** 与 Gemini 3 Flash**: Google 推出了搭载 **Gemini 3 Flash** 的 [Agentic Vision](https://blog.google/innovation-and-ai/technology/developers-tools/agentic-vision-gemini-3-flash/)，将视觉推理与代码执行相结合，从而能够逐步操作图像。
   - *该模型会制定计划来逐步缩放、检查和操作图像，并将答案建立在视觉证据之上*。
- **OpenAI 通过 O3 和 O4-mini 扩展图像能力**: OpenAI 的 **O3** 和 **O4-mini** 通过在思维链 (chain-of-thought) 中进行图像思考来扩展图像能力，使其无需单独的专门模型即可进行裁剪、缩放和旋转，详见[此博客文章](https://openai.com/index/thinking-with-images/)。
   - 与 OpenAI 相比，Gemini 返回有意义边界框 (bounding boxes) 的能力是无与伦比的。
- **PRISM：OpenAI 的新宠**: OpenAI 推出了 **PRISM**，详见[此新闻文章](https://archive.md/d9Vsf)，这引发了关于在写作中更倾向于使用 **Typst** 而非 **TeX** 的评论。
   - 有人提到 *“看到这个名字时我首先想到的东西”*，并引用了[此处附带](https://cdn.discordapp.com/attachments/1392278974222307469/1465964452997370078/PRISM_logo.jpg?ex=697bae09&is=697a5c89&hm=3b25626917f4c2e1e0753faa6336b1d5b2a9556a08bd8cc5e325fb8ac0853e09&)的 logo。
- **Trinity 在 OpenRouter Show 中大放异彩**: 一名成员提到他们第一次观看 **OpenRouter show**，并对 Trinity 的环节感到兴奋，该环节是免费提供的。
   - 用于 Agentic 用法的 OpenRouter SDK 位于[此处](https://openrouter.ai/docs/sdks/agentic-usage#supported-ai-coding-assistants)。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1465798548443369502)** (476 messages🔥🔥🔥): 

> `回滚按钮缺失, Cursor CLI vs IDE, Cursor 定价, Clawdbot 安全问题, Gemini Agentic Vision` 


- **回滚按钮消失，用户感到焦虑**: 用户报告 **revert button** 从 UI 中消失，导致在代码弄乱时感到沮丧并浪费 token，一名用户发现[复制旧对话](https://cdn.discordapp.com/attachments/1074847527708393565/1465828018789552390/image.png?ex=697bd7b9&is=697a8639&hm=68ec5dd17c7a1be84a1f639f9a5a98db91ba3bd191336f2afe3e8252b804b12e&)可以将其找回。
   - 一位成员发现不点击回滚按钮反而会让它出现，暗示这可能是一个**一次性 Bug**。
- **对部分开发者而言 Cursor CLI 胜过 IDE**: 由于内存占用更小，一些开发者更倾向于使用 **Cursor CLI** 而非 IDE，这有助于避免 IDE 崩溃和模型无响应，特别是在超过 100k 行代码 (LOC) 的大型项目中。
   - 然而，一名用户发现 IDE 内部的 **Cursor CLI**（使用 WSL 作为终端）简直是 *“纯垃圾……真的没法用”*，另一名用户报告称，即使拥有 64GB RAM 和 i7 处理器，UI 也不流畅。
- **Cursor 定价调整**: 9 月 15 日之后，**auto mode 不再是无限使用**，而是计入每月 20 美元的额度，定价为 Input + Cache Write 每 1M tokens 1.25 美元，Output 每 1M tokens 6.00 美元，Cache Read 每 1M tokens 0.25 美元，但订阅较早的用户仍可以启用按需使用 (on-demand usage)。
   - 一名用户发现他们很快就耗尽了每月的订阅额度，建议*使用自己的 API Key 或者使用 Claude Code* 可能会更便宜。
- **Clawdbot 的凭据灾难**: 一名用户分享了关于 **Clawdbot 安全忧虑**的几个链接，报告称暴露的控制面板会导致凭据泄露和账户被接管。
   - 有推测称，由于潜在的量子解密问题，这可能导致 *“先存储，后解密”* 的数据泄露，且该公司因这些问题收到了停止侵权函 (cease and desist)。
- **Gemini Vision 令前端开发人员兴奋**: 一名用户发现 **Gemini agentic vision** 在视觉任务上正接近 SOTA（当前最佳）水平，并相信其集成将简化前端开发。
   - 成员们表示，他们迫不及待地想看到视觉功能被集成到 Agent 中，认为它优于 `Auto` 工具。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466135653895897335)** (1 messages): 

> `LM Studio 0.4.0, Server Deployment, REST API` 


- **LM Studio 更新至 0.4.0！**: 新一代 **LM Studio** 已发布，版本号为 **0.4.0**，点击查看[完整博客文章](https://lmstudio.ai/blog/0.4.0)。
- **现在支持非 GUI 服务器部署**: **LM Studio 0.4.0** 现在可以部署在非 GUI 服务器、CI 环境或任何地方。
   - 得益于全新的有状态 **REST API**，这为高吞吐量用例开启了并行请求支持。
- **本地 MCPs 获得有状态 REST API**: 新的有状态 **REST API** 旨在配合本地 **MCPs** 使用。
   - 作为 **0.4.0** 版本发布的一部分，UI 也进行了全面的翻新。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1465803965504557077)** (298 messages🔥🔥): 

> `GLM 3.7 Flash Coding Ability, LMStudio OpenAI Tool Calling, LM Studio OpenAI Streaming, Gemma 4 Speculation, LM Studio v0.4 Update` 


- **GLM 3.7 Flash 擅长编程，OSS 120 仍保持领先**: 成员们注意到 **GLM 3.7 Flash** 表现出良好的编程能力，但预计 **GPT OSS 120** 仍将是更优秀的编程模型，尤其是在 **Q4** 量化下。
- **LMStudio 的 API 在工具调用上遇到障碍**: **LMStudio OpenAI compatible Responses API** 无法正确处理工具/函数调用；在模型决定调用函数/工具后，服务器应该发送 `response.completed` 或 `[DONE]`，但目前并未实现。
- **插件代理助力 Unreal Engine**: 一位成员创建了自己的**插件和代理**来让 **OpenAI streaming** 正常工作，从而使 **Unreal Engine** 能够与 **LM Studio** 通信，进行 Actor 的生成和操控。
- **Gemma 4 推测引发热潮**: 用户们推测潜在的 **Gemma 4** 发布，期待其采用 **Mixture of Experts (MoE)** 架构并提供多种尺寸（4/8/12/30b），同时也有人开玩笑地建议为边缘设备推出 **1b** 模型，并提醒不要过度炒作。
   - 一位成员宣称：“如果 Gemma 4 不是 MOE，我就把鞋吃了。”
- **LM Studio v0.4 引入 Headless 模式和并行功能**: **LM Studio v0.4** 引入了 **headless 模式**和**并行推理**，用户对这些新功能感到兴奋，尽管应用内更新需要重新安装程序，且部分 UI 元素现在处于 **dev mode**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1465817259153821738)** (33 messages🔥): 

> `Remote AI Rigs, Devstral-2 Performance, ROCm in LM Studio, 1.8 bit Quantization, LM Studio Backend for Hardware Accelerators` 


- **AI 工程师装备：远程访问现状**: 成员们讨论了远程访问其 AI 装备的方法，有人建议在运行 LLM 的虚拟机上使用 **VNC**。
   - 最初的问题是关于使用 **Windows 内置的远程桌面 (Remote Desktop)**。
- **Devstral-2 需要不俗的 GPU 部署**: 成员们讨论了本地运行 **Devstral-2** 的硬件要求，一位用户建议为 24B 版本配备 **48GB GPU 显存**（例如 3090）。
   - 对于 120B 版本，建议使用并行计算或配备 **EXL2** 模型格式的 **H200**，因为 GGUF 被认为速度太慢。
- **ROCm 在 LM Studio 运行时上运行**: 用户发现可以在 **LM Studio** 的运行时设置中启用 **ROCm**，这一选项最初对某些用户来说比较隐蔽。
   - 一位成员分享了一个相关的 [Unsloth Reddit 帖子链接](https://www.reddit.com/r/unsloth/comments/1qpbmrt/you_can_now_run_kimi_k25_locally/)。
- **1.8 Bit 奇迹：量化特性受质疑**: 成员们讨论了 **1.8 bit 量化**的本质，一位用户将其解释为一种动态量化方法，其中不重要的部分为 **1 bit**，其他部分为 **2-3 bits**。
   - 其他人将其比作“被切除前额叶的前科学家”，并开玩笑说用它只能运行俄罗斯方块。
- **硬件加速黑客：接入 LM Studio**: 一位来自硬件加速器公司的成员询问如何为其硬件添加 **LM Studio 后端**。
   - 建议重点关注 **llama.cpp**，因为 LM Studio 将其作为后端库使用；但同时也指出 LM Studio 主要是 Element Labs 的闭源项目，并提供了 [LM Studio Enterprise](https://lmstudio.ai/enterprise) 的链接。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1465798707546034268)** (307 messages🔥🔥): 

> `Kimi K2.5 pricing, Google Aistudio data training, Model selection (Kimi vs Mistral vs Qwen), Kimi CLI vs other tools, Agent Swarm` 


- **用户讨论 Kimi K2.5 定价模式**：部分用户对 Kimi K2.5 每月 **$19** 的订阅费表示担忧，一名用户因所在地原因认为价格 *昂贵*，并考虑是否能达成长期合作协议。
   - 另一名用户建议坚持使用免费层级，理由是像 Moonshot AI 这样规模较小的中国公司需要运行像 K2.5 这样的大模型，因此不太可能降低价格。
- **Google AI Studio 的训练实践引发争论**：一名用户对 **Google 在 AI Studio 和 Gemini 应用中训练并查看对话内容**表示担忧，引发了隐私问题。
   - 相比之下，另一名用户提到他们反正也会**开源自己的项目**，所以无论如何数据最终都可能进入训练数据集。
- **模型选择热潮：Kimi vs. Mistral vs. Qwen**：用户比较了 Kimi K2.5 与 **Mistral 和 Qwen** 等其他模型在编码和通用问答等各种任务中的表现。
   - 一名用户指出，在上述模型中，**Kimi K2.5** 在物理、化学和数学方面的 *benchmarks 最高*，而另一名用户则指出其在 *设计和逻辑推理方面表现强劲*。
- **Kimi CLI 证明优于备选方案**：用户测试了 **Kimi CLI**，发现与 oh-my-opencode 相比，它**更快速且更高效**，特别是在网页分析方面，且 tokens 消耗更低。
   - 然而，一些人发现该模型的输出质量 *不尽如人意*，希望进行进一步对比。
- **在 Kimi K2.5 中探索 Agent Swarm 的使用**：一名用户喜欢在 Kimi 中使用 **Agent Swarm**，并注意到其深度研究的能力，而其他人则对其应用场景尚不确定。
   - 有人指出 **Agent Swarm** 会迅速耗尽 agent credits，其消耗速度是正常速度的 **3x**。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1465798839821799424)** (177 messages🔥🔥): 

> `Subscription Scams, Billing Issues and Refunds, Query Limits, Image Generation Restrictions, Kimi 2.5 Release` 


- **Perplexity 订阅被称为骗局？**：多名用户报告在自动续订后出现 **非预期的订阅变更** 和 **扣费**，一名用户取消了订阅并称其为 *骗局*。
- **计费问题引发银行联络**：用户报告了计费差异，例如在没有服务的情况下被扣费或无法获得退款，一名用户计划 **联系他们的银行** 申请 100 欧元的退款。
   - 另一名用户建议联系支付处理器以 **停止进一步的未经授权交易**，并向 FTC 报告该问题。
- **用户对查询上限（Query Cap）乱象感到困惑**：一些用户报告了其 **Pro 订阅** 的 **查询限制** 问题，限制低至每小时一次查询，而其他人的限制则恢复到了 600 次。
   - 一名用户分享了一个检查查询限制的链接 ([perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all))，并指出其 **600 次查询** 突然恢复了。
- **图像生成受地区限制？**：用户报告在某些地区 **图像生成受到限制**，可能是由于 **xAI 争议** 和欧盟诉讼，并建议尝试不同的模型或联系支持部门。
   - 一名来自印度的用户确认他们也受到了此问题的影响。
- **Kimi 2.5 即将登陆 PPLX？**：用户正在询问 Perplexity 上 **Kimi 2.5 模型** 的发布日期，渴望其尽快上线。
   - 一名用户推测 Perplexity 在此类更新方面通常动作很快。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

tay.0.00: Love
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1465801949499756647)** (159 messages🔥🔥): 

> `GPT Pro, OpenAI pricing, DeepSeek jailbreak, TI-84 Neural Network, Staged Reward Shaping` 


- **GPT Pro：是更多 GPU 还是模型魔法？**：讨论围绕 **GPT Pro** 的卓越性能是源于单纯堆砌 GPU 算力（例如并行运行多个实例），还是涉及根本性的更优模型，并推测 **OpenAI** 可能出于竞争优势战略性地掩盖了真实性质。
   - 一位成员甚至认为这是一场“造假游戏”，将 **OpenAI** 的定价策略比作印象分而非实测价值，类似于股市和 **Tesla**。
- **中国加强管控，发现过滤机制**：成员们讨论了**中国模型**受审查的情况，一位成员声称 *CCP 对这些实验室拥有很大权力*，并分享了一张[图片](https://cdn.discordapp.com/attachments/1149866623109439599/1465867774994940170/image.png?ex=697bfcc0&is=697aab40&hm=ccdd92053333326694a5a1919519f39b17935bca2eb5be926d99b4fb2ca5afbc&)，显示模型在思维链（thinking traces）中存在审查过滤器。
   - 他们还指出，**中国**成功操纵了公众认知并资助了大量的 AI 实验室。
- **DeepSeek 的无限囚禁**：成员们注意到 **DeepSeek** 容易陷入越狱（jailbreak）循环，一旦触发，无论随后的 Prompt 是什么，它都会无限重复相同的拒绝信息（*我无法为此提供帮助*）。
   - 据报道 API 端点表现略好，但原始模型一旦进入该状态就“废了（cooked）”。
- **计算器获得神经网络助力**：一位成员分享了他们在 **TI-84 Plus** 计算器上运行神经网络进行拼写检查的项目，并在一个[学术网站](https://hermesoptimus.vercel.app/)上详细介绍了过程并附带演示视频。
   - 该成员打趣道，即便有这样的进展，他正在进行的 **Claude Code Orchestration** 工作在实际应用方面仍然更胜一筹。
- **阶段性奖励塑形（Staged Reward Shaping）：委托给委托？**：围绕**阶段性奖励塑形**展开了讨论，即随时间添加和调整中间奖励，并对模型容易产生奖励黑客行为（reward hacking）表示担忧。
   - 一位成员将其定性为“技术债”，另一位成员建议这是“为了委托而委托，而不是为了跑得更快而委托”。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1465921905876795474)** (4 messages): 

> `Hermes 4 pricing, API credits` 


- **Hermes 4 定价并非永久**：一位成员在订阅 API 前询问 **Hermes 4 系列**模型的折扣定价是否永久，并指出其在 RP（角色扮演）和故事创作方面优于 **DeepSeek**。
   - 另一位成员澄清说没有订阅制，只需购买积分（credits），且价格可能随时间变化，因此价值取决于价格和使用情况。
- **API 积分说明**：一位成员解释说，使用 API 涉及购买可以充值的积分，而不是订阅。
   - 从积分中获得的价值将根据**定价**和**使用**模式而波动。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1465882077357408388)** (1 messages): 

> `MergeMix paper, Data Mixtures, Model Merging` 


- **MergeMix 论文引起关注**：论文 [MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858) 因其对预算有限的开源项目的相关性而受到关注。
   - 该论文探讨了在训练期间优化**数据混合（data mixtures）**和**模型合并（model merging）**的技术，可能提供资源高效的策略。
- **图像分析讨论**：分享了一张图片，推测与 MergeMix 论文或数据混合有关，但缺乏进一步的背景或讨论。
   - 在没有更多信息的情况下，该图片的具体关联性或内容尚不明确。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1465882077357408388)** (1 messages): 

> `MergeMix, Open Source Model Merging` 


- ****MergeMix** 优化训练中期的数据混合**：一名成员分享了论文《[MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging](https://arxiv.org/pdf/2601.17858)》，强调了其对预算有限的开源项目的意义。
   - 该论文探索了通过**可学习的模型合并 (learnable model merging)** 来优化训练中期的数据混合。
- **开源模型合并获得助力**：由于该论文对财务资源显著匮乏的开源倡议具有启示作用，因此被认为非常有趣。
   - 随附的图片 [图片链接](https://cdn.discordapp.com/attachments/1104063238934626386/1465882077428846623/image.png?ex=697b6152&is=697a0fd2&hm=7ea494cc6abfd85cd18d6434dca494139bbccaaf966f67adbdb35b09748bf465) 为讨论提供了视觉补充。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1465800663253385389)** (81 messages🔥🔥): 

> `Gemini 3 Pro failure to generate .srt files, Clawdbot a scam by crypto bros, Prism harmful to scientific research, OpenAI prioritizing security concerns, ChatGPT vs Gemini comparison` 


- **Gemini 3 Pro 表现糟糕**：一位用户报告称 **Gemini 3 Pro** 完全虚构了一个 .srt 字幕文件，内容与视频音频毫无关系，导致其对性能感到失望。
   - 其他用户也表达了类似经历，其中一人表示 *虽然很不情愿，但 Gemini 最近确实被过度炒作了，它在我的使用中表现也不佳*。
- **Clawdbot 模糊的恶意软件状态**：**Clawdbot**（现更名为 **moltbot**）是一个通过 Anthropic、Google、OpenAI 的 API Key 控制用户整个操作系统的 Agent 系统，用户被警告远离它，有用户称其为 *加密货币圈人士用来窃取信息的巨大骗局*。
   - 尽管原始版本本质上并非恶意软件，但它可以通过 Prompt Injection 被武器化，转变为次生恶意行为，引发了关于自动化/Agent AI/机器人 AI 的重大安全和隐私担忧。
- **Prism 的科幻未来还是科研败笔？**：虽然 **OpenAI** 旨在通过 **Prism** 推进科学，但一位用户指出 **Prism** *对科学研究并无益处*，实际上是对科学研究的 *损害*。
   - 另一位用户询问 **Prism** 是否有 API 访问权限，想知道是否可以使用其他 AI 和 **Codex** 在那里编写项目。
- **OpenAI 优先考虑网络安全**：一位用户分享道，**OpenAI** 正试图升级其 **Codex** 以强力应对网络安全问题。
   - 他们认为这是因为 *对于那些只想自由地进行创建和自动化，而不必担心被篡改和劫持的人来说，安全和保密确实是一个巨大的隐忧*。
- **ChatGPT 战胜了教条化的 Gemini**：一位用户表示，对于 LLM 而言，*基准测试排行榜的名次对我来说意义不大*，**Gemini** 非常教条化，而 **ChatGPT** 在处理跨会话上下文方面表现出色。
   - 用户还注意到 Gemini 在规则上非常死板，一旦告诉它某种偏好，*它就会像信奉宗教一样死守不放*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1465916154471252215)** (4 messages): 

> `AI as career, AI Safety, GPT file reading` 


- **以 AI 谋生**：一位成员询问是否有人正以 **AI** 为生，寻求关于如何将对 **AI** 的热情转化为收益的建议。
   - 一位用户建议探索 **AI Safety** 和 **Red Teaming**（红队测试），并指向了相关社区。
- **GPT Pro 失去文件读取能力**：一位用户报告称，此前能够读取和分析 **ZIP 文件** 的 **GPT Pro 5.2**，现在无法找到上传的文件进行分析。
   - 用户正在询问其他人是否也遇到了同样的问题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1465918446339620915)** (8 条消息🔥): 

> `Sora Prompting Guide, GIF Generation, Chiaroscuro Effect in Image Generation, Realtime Visualizers` 


- **提示词强化：Sora 的细微差别大放异彩！**：成员分享了 [Sora Prompting Guide](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide)，强调了在 Prompt 中保持**积极语调 (positive cadence)** 以及有效地对负面约束进行分组的重要性。
   - 用户建议避免过多单独的“不要 x (no x)”指令，以获得更好的结果。
- **GIF 魔法：应用内动画站！**：用户确认 **GIF 生成过程可以完全在应用内完成**。
   - 一位用户展望未来，预想 **GIF** 和其他 **animation** 将扩展到流式模型中，可能包括带有 **Visualizers** 的 **OAI 版本 Lyria Realtime**。
- **消除黑白：阻击 “Chiaroscuro” 灾难！**：用户讨论了一个与 **Chiaroscuro（明暗对比法）** 效果相关的图像生成“问题”，该效果利用光影之间的强烈对比。
   - 建议在遇到不想要的**黑白图像**时，在 Prompt 中 *“请避免使用 Chiaroscuro”*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1465918446339620915)** (8 条消息🔥): 

> `Sora Prompting, GIF creation, OAI Lyria Realtime, Chiaroscuro Image Issue` 


- **Sora 爱好者分享 Prompt 指南**：成员们分享了一份实用的 **Sora** [提示词指南](https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide)，建议保持积极语调并避免将负面约束堆叠。
   - 该建议旨在防止过多的“不要 x (no x)”指令让模型无所适从。
- **应用内 GIF 生成技巧**：一位成员指出 **GIF 创建过程** 可以完全在应用内完成，展示了其流线型的功能。
   - 另一位成员鼓励用户利用像 **PIL** 这样高级的库以获得最佳结果，并预测 **OAI 5.2** 将会大力支持这一过程。
- **OAI 的 Lyria Realtime Visualizer 愿景**：一位用户对带有 **Visualizers** 的 **OpenAI 版本 Lyria Realtime** 表示兴奋，强调了引导模型的乐趣。
   - 他们幻想着 *disco cat-girls（迪斯科猫娘）*，并提议将 **OAI 声乐教练** 作为一个很酷的创意，构想使用不同语言进行聊天。
- **Chiaroscuro 引发混乱**：用户报告了**黑白图像**的问题，建议在 Prompt 中避免使用 *Chiaroscuro* 以减轻这种影响。
   - **Chiaroscuro** 被定义为光影之间强烈的对比，通常是影响整个构图的大胆对比。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1465884462893105407)** (29 messages🔥): 

> `TorchX, mech interp-style tools for numerics debugging, FlagOS Open Computing Global Challenge, transformers 4.57->5.0  or pytorch 2.9.1 ->2.10 breaking training pipeline, interactive tools for exploring numerics` 


- ****TorchX** 编排仍然推荐吗？**: 一位成员询问 [TorchX 视频](https://www.youtube.com/watch?v=f-Bwru7TJSc) 是否仍是多节点 GPU 编排的推荐标准。
   - 视频作者回复称，这是他们目前在内部服务器上主要使用的工具，但过去一年中他们没有跟进任务启动器（job launcher）的演进。
- **用于数值调试的 **Mech Interp** 工具？**: 一位成员询问是否可以使用 Mech Interp 风格的工具进行数值调试，希望利用它们在算子（op）和 Kernel 级别调试模型的不稳定性。
   - 另一位成员对这种让模型调试更具方法论的工具很感兴趣，例如*检查哪个电路（circuit）不稳定、哪一层产生了大量离群值等基础工作。*
- ****FlagOS** 全球挑战赛**: **FlagOS Open Computing Global Challenge** 竞赛已面向全球开发者开放，总奖金池达 **2,000,000 人民币**。
   - 竞赛的更多详情请见 [flagos.io](https://flagos.io/RaceDetail?id=295v67vw&lang=en)。
- **Transformers 和 PyTorch 升级导致训练中断**: 一位成员报告在升级 **transformers** 和 **pytorch** 后出现 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` 错误。
   - 将 transformers 降级至 **4.57.3** 修复了该问题；其他人也遇到了类似问题，相关讨论见此 [pytorch issue](https://github.com/pytorch/pytorch/issues/127176) 和 [optimi issue](https://github.com/warner-benjamin/optimi/issues/8)。
- **用于探索数值的交互式工具**: 一位成员表示惊讶，量化领域的人竟然还没有开发出用于探索数值的交互式工具。
   - 另一位成员回复称 *许多研究都在使用标准架构*，并提到 [captum](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.html) 是一种可能的工具，但缺乏合适的 UI/UX。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1465955998396317738)** (8 messages🔥): 

> `CompositeImplicitAutoGrad Error, JAX AI-Generated PR, Triage Bot` 


- **CompositeImplicitAutoGrad 产生错误**: 用户在尝试强制自定义算子使用 `CompositeImplicitAutoGrad` 进行自动微分时遇到了 `UserWarning`。这源于 autograd Kernel 未注册到 `Autograd` 键，引发了对潜在错误行为和已弃用功能的担忧。
   - 用户质疑 *Fallthrough* 是否仅是主库算子的选项而非自定义算子的选项，并寻求关于如何解决该错误以及确保其自定义算子能正确进行微分的说明。
- **AI 生成的 PR 激怒开发者**: 一位开发者表达了挫败感，因为他看到 JAX 中一个由 AI 生成的 Pull Request (PR) 得到了维护者的积极响应，而他自己修复 Bug 的小 PR 却无人问津。
   - 开发者讥讽地将该 AI 生成的 PR 贴上 *明显的垃圾内容（slop）* 的标签，批评维护者优先处理它而非真实的贡献。
- **Triage Bot 面临不确定的未来**: 一位用户询问在引入新的 Triage Bot 后，Triage 会议的命运将会如何。
   - 用户质疑 Triage Bot 的实施是否会导致传统的 Triage 会议取消，暗示了对该机器人有效性或其对团队动态影响的担忧。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1465862401629884642)** (1 messages): 

> `H200, INT4, QAT, RL, Model Rollout` 


- **使用 INT4 将 1TB 模型塞进 H200**: 一位成员分享了一个链接，介绍了如何通过 **INT4 QAT RL** 端到端实践，将 **1TB 模型 Rollout** 塞进单块 **H200** 中。
   - 详情可在该 [GitHub repo](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme-en.md) 中找到。
- **INT4 QAT RL 仓库**: 该 GitHub 仓库提供了与 **INT4 QAT RL** 实现相关的资源和文档。
   - 它专注于为 **H200** 等硬件优化大规模模型 Rollout。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1465831167226810399)** (1 messages): 

> `Decart Hiring, Lucy 2 Model, Real-time video kernels` 


- **Decart 为旧金山办公室招聘性能工程师**：Decart 正在为其旧金山（SF）办公室招聘工程师，致力于实时视频/世界模型以及最新加速器的低延迟 Kernel 开发，特别提到了在 ReInvent 上展示的 **Trainium 3** 成果（[视频](https://www.youtube.com/watch?v=K49S79wOGl8)）。
   - 鼓励感兴趣的候选人联系 heba@decart.ai，并附上其性能优化相关的作品，例如 **GPU Mode 提交记录**或 **OSS 贡献**。
- **Decart 发布 Lucy 2 自回归模型**：Decart 推出了最新的自回归视频编辑模型 **Lucy 2**（[技术报告](https://x.com/DecartAI/status/2016134190509498740)）。
   - 他们还将与 **GPU Mode** 共同赞助即将举行的针对自回归扩散模型的 Kernel 挑战赛。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1465935936905805824)** (13 messages🔥): 

> `PopcornCLI github issues, CUDA C++ and Python, Performance plan for CUDA, PMPP book` 


- **PopcornCLI 截止日期错误浮出水面**：一名成员报告在运行 **PopcornCLI** GitHub 引用命令时遇到错误，特别是在将排行榜从 grayscale 更改为 vectorsum 时出现 *deadline has passed*（截止日期已过）的消息。
   - 他们发现排行榜后缀为 **v2**（例如 *grayscale_v2*，*vectorsum_v2*），且 TUI 会显示排行榜。
- **寻求深度学习中 Python 结合 CUDA C++ 的指导**：一位自称“新手”的成员请求关于在深度学习中同时运行 **CUDA C++** 和 **Python** 的指导。
   - 另一位成员建议查看 PyTorch 中的 **load_inline** 功能，并提到 Lecture 1 中有相关说明。
- **CUDA 性能规划入门建议**：一位成员询问在编写 **CUDA** 代码前制定性能计划的指导建议，他了解 **NVIDIA Nsight**（原文为 Insight）但想理解其建议背后的“原因”。
   - 另一位成员询问了他的专业水平，以及是否已经开始阅读 book 频道中的 **PMPP book**。
- **gpumode 链接修复**：一位成员发现 **gpumode** 的链接已失效，并建议将其替换为[此链接](https://www.gpumode.com/)。
   - 无进一步讨论。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1465898873938776175)** (1 messages): 

> `New Collaborators, Kernel LLM` 


- **征集 Kernel LLM 改进合作者**：正在寻找新的合作者，以“快速运行消融实验（ablations）”并“生成/测试想法”，从而改进训练后的 **Kernel LLM**。
   - 此次招募强调在 **synthetic data**（合成数据）、**training algorithms**（训练算法）和 **memory**（内存）优化等领域的技能。
- **会议和公告引发兴趣**：一位成员在查阅 **2026 年新闻与公告**帖子后，对 **Kernel LLM** 协作机会表示了兴趣。
   - 该成员询问了该频道的关联性以及如何开始做出贡献，表现出对该计划的积极参与意愿。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/)** (1 messages): 

ivanbernal0511: 告诉我你的 Jetson 型号、batch size，以及你的目标是 FP16 还是 INT8
  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1465816049483124939)** (1 messages): 

> `DGX, 5090, Blackwell PRO, L2 cache` 


- **DGX 和 5090 指令集相同！**：**DGX** 和 **5090** 的指令集是一样的，但 **DGX** 拥有全速 fp32 累加能力，类似于 **Blackwell PRO** 系列显卡。
   - 真正的差距在哪里？**1.8TB/s** 对比 **300 GB/s** 的内存带宽——高效利用 **L2 cache** 是关键！
- **内存带宽：DGX 占据主导地位**：**DGX** 以 **1.8TB/s** 的内存带宽脱颖而出，与 **5090** 的 **300 GB/s** 形成鲜明对比。
   - 优化 **L2 cache** 的利用率对于有效发挥 **DGX** 的卓越性能至关重要。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1465900732388474932)** (16 messages🔥): 

> `Tractable Layouts, Tuple Morphisms, Mutual Refinements, Cute Composition, tract.weak_composite` 


- **Tractable Layouts 图表中的顺序至关重要**：在表示 Tractable Layouts 的图表中，两侧节点的顺序非常关键；交换元素会导致不同的 Layout，例如将 `(4, 8):(1, 4)` 更改为 `(4, 8):(8, 1)`。
   - 一位成员指出，这种顺序并非随意的，它非常僵化，改变左侧的排列会产生显著差异。
- **关于 Tuple Morphism 上域（Codomain）和定义域（Domain）的澄清**：在 Mutual Refinement 中，左侧是 Tuple Morphism `m_A` 的 **上域（codomain）**，而右侧是 `m_B` 的 **定义域（domain）**。此外，还提供了一篇[博客文章](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/)以供参考主要定义。
   - 两侧都按照前一页的方式从下到上进行排序。
- **通过 Tract 实现 Cute Composition**：`tract.compose` 要求第一个 Morphism 的 **上域（codomain）** 等于第二个 Morphism 的 **定义域（domain）**，而 **Mutual Refinements** 通过 *refine, pullback/pushforward, compose*（被称为 *weak composition*）来推广组合。
   - 为了在 `tract` 中实现这一点，应该使用 `tract.weak_composite(morphism_A, morphism_B)`。
- **修复 Layout Diagram 中的拼写错误**：一位成员发现了 Layout Diagram 截图中的一个拼写错误，并澄清图中节点的顺序对于定义 Layout 至关重要。
   - 修正后的理解简化了关于流程中 *Step 2* 如何产生预期组合结果的推导。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466039529499525161)** (2 messages): 

> `mdbook rust playground, AWS/GCP research cloud credits, magnetron, rust->x86 to cuda->ptx` 


- **mdbook REPL 默认使用低性能的 Debug 模式**：mdbook 的 REPL 支持将代码发送到 **Rust Playground** 的公共实例，并使用 debug cargo profiles，导致性能仅为 `~10MFLOPS`，而 release 模式下为 `1GFLOPS`。
   - 它发送的是带有 **debug cargo profiles** 而非 release 的 JSON 请求，但一位成员计划对 mdbook 的 JavaScript 进行 Monkey Patch，以便在 release 模式下发送请求。
- **运行在廉价硬件上的公共 Rust Playground**：由 integer32 镜像并在其 README 中链接的公共 **Rust Playground** 托管在免费层的 t2.micro 实例上，在 release 模式下可达到 `1-2GFLOPS`，符合初步估算。
   - **t2.micro** 的理论最大吞吐量约为 `~20GFLOPS`，但 vCPU 的 Hypervisor 利用率上限为 10%，并使用积分进行弹性突发。
- **关注 AWS/GCP 额度以进行大型基准测试**：一位成员计划申请 **AWS/GCP 研究云额度**，灵感来自 mario 在 magnetron 中的做法，以在高性能 CPU 上实现 `~2TFLOPS`。
   - 这种方法将涵盖使用 Intel VTune/AMD uProf 的 **rust->x86**，以及使用 Nsight 的 **cuda->ptx**。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1466162536926937290)** (1 messages): 

> `Ed Yang, JAX, Torch, Sharding` 


- **Ed Yang 的博客文章比较 JAX 和 Torch**：Ed Yang 发布了一些关于分布式计算话题的有趣博客文章。
   - 值得注意的是，他比较了 **JAX** 和 **Torch** 如何处理 **Sharding** 的不同方面（[推文链接](https://x.com/ezyang/status/2016268240754712988?s=20)）。
- **分布式计算见解**：Ed Yang 最近的博客文章提供了对各种分布式计算主题的见解。
   - 这些文章对 JAX 和 Torch 中处理 Sharding 的不同方法进行了比较分析。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1466146755614871708)** (1 messages): 

> `AMD Helion Plans, Enable Skipped Tests` 


- **AMD Helion 计划引发好奇**：一位用户表示有兴趣进一步了解 **AMD 在 Helion 上的计划**。
   - 他们建议进行一次快速同步会议来讨论更多细节。
- **启用跳过的测试**：一位用户感谢另一位用户提交了启用跳过测试的 **PR**。
   - 未提供进一步细节。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1465878066407542806)** (10 messages🔥): 

> `constexpr in CuTeDSL, NCU profiling, Kernel hangs, Grand prize arithmetic difference, measurement error` 


- **constexpr 提升 CuTeDSL 性能**：一位成员分享了在 **CuTeDSL** 中使用 **constexpr** 优化参考内核性能的教程，声称其性能将远优于简单的基准线（baseline），并附带了 [教程链接](https://gist.github.com/simveit/f8f538adacb5d4c2703600b843ba0547)。
- **NCU profiling 状态不明**：有成员询问 **NCU profiling** 是否已恢复工作。
   - 随后出现了关于非法内存访问（illegal memory accesses）或内核挂起（kernel hangs）的反馈，并询问是否可以向某人发送代码和 **NCU profiles**。
- **大奖评选采用算术差**：有人提问大奖评选中“最接近光速（speed of light, SoL）”的衡量标准是 **算术差（arithmetic difference）** 还是 **百分比差（percent difference）**。
   - 另一名成员表示可以对任何规则细节进行解答。
- **关于测量误差的疑问**：另一个关于大奖的问题：如果 **测量误差（measurement error）** 导致结果落在 SoL 的另一侧（即超过 100%），而另一个人从较慢的一侧更接近 SoL，该如何处理？


  

---


### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1465907650381480119)** (2 messages): 

> `Nvidia B200, CuTile, nvfp4` 


- **B200 缺乏 CuTile 支持**：用户询问 **Nvidia B200** 竞赛环境是否支持 **CuTile**。
   - 另一位成员回复称目前尚不支持 **nvfp4**，因此 **CuTile** 的作用有限。
- **缺少 NVFP4 支持**：**Nvidia B200** 竞赛环境目前不支持 **nvfp4**。
   - 在没有 **nvfp4** 支持的情况下，**CuTile** 在 **B200** 环境中无法发挥特别有效的左右。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1465825640807661628)** (10 messages🔥): 

> `Biweekly Leaderboard, Flashinfer-bench ModuleNotFoundError, MLSys'25 contest trace, Quantization Algorithms in FlashInfer, Looking for Teammates` 


- **双周排行榜即将推出**：团队正在努力为比赛支持 **双周排行榜（biweekly leaderboard）**。
- **Flashinfer-bench ModuleNotFoundError 已解决**：一位用户在运行 `python ./scripts/pack_solution.py` 时遇到了 `ModuleNotFoundError`，但通过从 **最新 GitHub 仓库** 安装解决了该问题。
- **MLSys'25 竞赛 Trace 发布延迟**：一位用户在尝试使用 **flashinfer trace** 时报错，被告知可能需要等待 **MLSys'25 竞赛 trace** 的正式发布。
- **FlashInfer 探索量化算法**：讨论了 **FlashInfer** 是否计划支持更好的 **量化算法（quantization algorithms）**，并提供了指向 [相关 GitHub issue](https://github.com/flashinfer-ai/flashinfer/issues/2423) 的链接。
- **Huggingface 上已提供 "fused_moe" 定义**：*fused_moe* 的定义和工作负载（workloads）可通过 [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/) 获取，团队要求用户确保将 `FIB_DATASET_PATH` 设置为 **本地数据集路径**。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1465816651978248413)** (66 条消息🔥🔥): 

> `Agent-Driven Coding, Prism Science Workspace, Trinity Large MoE Model, Agentic Harnesses Evolution, Cursor's Codebase Indexing` 


- **Agent 驱动的编程迈向 2026 年！**：Andrej Karpathy 预见，到 2026 年将转向 **80% 的 Agent 驱动编程**，利用 LLM 的持久性和声明式目标设定，同时提醒注意潜在的“废话（slop）”和过度工程；更多内容请点击 [这里](https://xcancel.com/karpathy/status/2015883857489522876)。
- **Prism 闪耀亮相，作为 OpenAI 的新科学工具！**：OpenAI 发布了 **Prism**，这是一个专为科学家设计的免费研究工作空间，由 **GPT-5.2** 驱动，现在所有拥有个人 ChatGPT 账号的用户均可访问；通过专用 Web 门户 [这里](https://xcancel.com/openai/status/2016209462621831448?s=46&t=eWVlK1PU8XfB6f402GJJ9g) 访问。
- **Trinity Large 的 400B 参数威力！**：Prime Intellect、Arcee AI 和 Datology 推出了 **Trinity Large**，这是一个 **400B 参数的 Mixture of Experts (MoE) 模型**，仅利用 **13B 激活参数** 即可实现高性能；链接见 [这里](https://xcancel.com/primeintellect/status/2016280792037785624?s=46)。
- **Agentic Harnesses：编排未来！**：一篇长文推测了模型 Harness 的演变，认为更智能的模型将取代像 LangChain 这样复杂的编排器，转而支持 Multi-Agent 架构和基于文件系统的协作；链接见 [这里](https://xcancel.com/voooooogel/status/2015976774128341421?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)。
- **Cursor 获得更快的索引速度！**：Cursor 宣布了性能升级，包括语义搜索和针对大型代码库显著加快的索引过程；更多详情见 [这里](https://xcancel.com/cursor_ai/status/2016202243499073768?s=46)。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1466191240063090942)** (1 条消息): 

> `Latent Space podcast, Science podcast` 


- **Latent Space 推出 'Science' 播客**：Latent Space 发布了其第二个播客 'Science'（[播客链接](https://www.latent.space/p/science)），由 <@713947182167883897> 和 <@348078436058660866> 主持。
- **播客讨论转移至专用频道**：关于新 'Science' 播客的进一步讨论请转至新创建的频道 <#1430253273335595079>。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1465903855450722418)** (10 条消息🔥): 

> `MimikaStudio MacOS app, Real-time AI Character Swapping, 1littlecoder AI Tutorials` 


- **MimikaStudio：新款语音类 MacOS 应用**：一名成员分享了关于 **MimikaStudio** 的 [Reddit 帖子](https://www.reddit.com/r/Qwen_AI/comments/1qnlupq/i_built_mimikastudio_a_native_macos_app_for_voice/) 链接，这是一款用于语音相关任务的原生 MacOS 应用。
- **实时 AI 角色替换技术上线**：**DecartAI** 发布了一个新的 AI 模型，支持视频中的零延迟角色替换，从而实现具有瞬时身份替换功能的实时视频流。
   - 与之前需要生成时间的 **Kling Motion Control** 等工具不同，该模型允许进行具有瞬时身份替换功能的实时视频流传输。
- **1littlecoder 加入阵营**：一位成员分享了 '1littlecoder' 的 [Nitter 个人主页](https://x.com/1littlecoder) 链接，该账号专注于 **AI 教程**、**Large Language Models (LLMs)** 和**编程**。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1465806881032241335)** (64 messages🔥🔥): 

> `Zed 的本地 Agent 推荐、GLM-4.7-Flash 性能、LLM/SaaS 全栈 AI 开发者求职、Kimi 2.5 模型性能、AI Agent 开发中 C++ vs Python` 


- **Kimi 2.5 性能超越 GPT5**: 一位成员报告称，新的 **Kimi 2.5** 模型表现持续优于 **GPT5**，现在可以通过此 [HuggingFace 链接](https://huggingface.co/unsloth/Kimi-K2.5-GGUF) 在本地运行。
   - 其他人正在使用 [Fireworks](https://www.google.com/aclk?sa=L&ai=DChsSEwiCz-j3iK2SAxUFVX8AHT5cBPkYACICCAEQABoCb2E&co=1&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoL9Ani52eByT53nVhnOxG_76F9QllEx50YhK_yfQYsD5bH3ov1pAqwaAl2XEALw_wcB&cid=CAASugHkaDm-Aokq5n3lAlzNAI-Ihc6SdblOJ-BiATzwnaZwDVhVBl3B2U5kGq4mAYjN4wQ992LlqWX5NQ6HksDrhSatp0QEfb7_rWMS_u7_GTCuCkp3YH9fANMaJqDgFvuA6u1bwvl4pJ80zvbUhIFPk7Nrqdpx2PDnsBRncgM3-d1UDhFM-tN117MrOXLWnhycCaPax24T8meZIe-9I2cM5rpAf16KucPGZwg7ixTssRCB7X8RP3B_G4vUCfE&cce=2&sig=AOD64_2SRpHfWjuW4kJawyiTyzrGbKZybQ&q&adurl&ved=2ahUKEwiiteP3iK2SAxV85skDHfklKyoQ0Qx6BAgLEAE) 等网站进行访问。
- **本地 Zed Agent 推荐**: 一位成员在配合 **llama.cpp** 使用 Q4 量化版本的 **GLM-4.7-Flash** 时表示不满，并寻求适用于 **Zed** 的本地 Agent 推荐。
   - 另一位成员推荐了 **kimi** 和 **qwencoders 30b q4**。
- **C++ 在构建 AI Agent 方面占据统治地位**: 一位成员指出 *C++ 将永远统治该领域*，并提到 *Python Agent 现在显得有些臃肿*，建议专注于 **C++** 以胜任高级职位。
   - 他们推荐使用 **fastwhisper.cpp** 进行 STT，在 LlamaCPP 中使用 **Qwen embeddings** 进行 RAG，以及使用 **LFM2.5vl** 作为 VLM。
- **开发者集结构建新 AI 项目**: 多位成员展示了自己的 AI 工程技能。
   - 一位成员发布了一系列关键项目列表，包括 Autonomous Agents、医疗 AI、决策支持系统、对话式 AI 和欺诈检测系统。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1465812757730426942)** (9 messages🔥): 

> `视觉模型 JPEG 伪影处理、RemnantInstruct-8B 合并、基于 CLIP 的 Kiki 或 Bouba 分类器、量子计算视觉语言模型、LeetCode MCP 服务器` 


- **视觉模型战胜 JPEG 伪影**: 一个新的视觉模型通过采用无 Batch Norm、训练后无激活函数以及使用 Operator 层代替 Convolutional 层的独特设计，能够消除由 **JPEG 压缩** 产生的伪影。
   - 据称，该模型通过增加**宽度**而非深度来获得更高的准确率。
- **RemnantInstruct-8B：结合创造力与准确性**: **RemnantInstruct-8B** 是一个 [SLERP merge](https://huggingface.co/anthonym21/RemnantInstruct-8B-GGUF)，它将一个创意微调模型 (**allura-org/remnant-qwen3-8b**) 与其基础模型 (**Qwen/Qwen3-8B**) 重新组合，以平衡叙事能力与事实准确性。
   - 该合并策略在 Self-attention 层偏向创意微调模型，在 MLP 层偏向基础模型，旨在保留 **Qwen3** 的思考模式。
- **Kiki vs. Bouba：CLIP 破解难题**: 一位成员发布了一个**基于 CLIP 的 Kiki 或 Bouba 分类器**，该分类器会根据约 200 个指示 "Kikiness" 和 "Boubaness" 的形容词（如 acidic、staccato、buttery 和 nurturing）来检查输入。
   - 该分类器可在 [HuggingFace Spaces](https://huggingface.co/spaces/jnalv/Kiki-or-Bouba-classifier) 上使用。
- **量子飞跃：VLM 攻克量子计算**: 一位成员开源了他们的本科毕业论文成果，专注于将**视觉语言模型**专门用于**量子计算**和 **Qiskit** 代码，包括 [数据集](https://huggingface.co/datasets/samuellimabraz/quantum-assistant)、[模型](https://huggingface.co/collections/samuellimabraz/quantum-assistant)、[代码](https://github.com/samuellimabraz/quantum-assistant) 和 [Demo](https://huggingface.co/spaces/samuellimabraz/quantum-assistant)。
- **LeetCode LM：在终端攻克编程挑战**: 一位成员开发了一个 **LeetCode MCP 服务器**，可以从终端解决每日挑战。它与 **Claude** 的学习模式集成，允许用户进行身份验证、获取题目、请求提示并提交解决方案。
   - 他们计划在其他 LM 以及 Cursor 和 JetBrains 上进行测试，并考虑开发 IDEA 插件；项目已在 [GitHub](https://github.com/SPerekrestova/interactive-leetcode-mcp) 上线。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1465950560577654969)** (2 messages): 

> `Smol Course, Agentic AI, RAG, LLMs, Production Tools` 


- **寻求 Smol Course 频道**：一名成员询问是否有专门针对 Agentic AI 的 **Smol course** 的服务器或频道。
   - 消息中未提供具体的服务器或频道详情；不过，该用户被引导至有关 **RAG, LLMs, Production Tools, Orchestration, Governance, 和 Real-World Deployments** 的资源。
- **ainewshub.live - 每日 AI 新闻**：[ainewshub.live](https://ainewshub.live/) 被提及作为获取 Agentic AI 每日高信号更新的来源。
   - 它为资深工程师提供关于 **RAG, LLMs, Production Tools, Orchestration, Governance, 和 Real-World Deployments** 的精炼信息。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1465798641657708817)** (31 messages🔥): 

> `Flow Matching, Transformers for Continuous Diffusion, Autoregressive Models vs Diffusion Models, Score Parameterization, Byte-Level Prediction Models` 


- **Transformer 可以参数化 Flow Matching 中的向量场**：一名成员质疑为什么有人声称 Transformer 不能用于 Flow Matching，并认为这是一种训练目标，**Transformer** 可以参数化向量场。
   - 另一名成员澄清说，**Transformer** 可以用于连续 Diffusion，其中 **patch embedding** 编码 patch 位置，但这并不会使 Diffusion 离散化，也不会将 patch 变成 token。
- **Flow Matching 与 Diffusion 的数学原理相同**：一名成员指出，讽刺的是 [Diffusion 模型的数学原理基本上与 Flow Matching 相同](https://arxiv.org/abs/2305.03486)，但 Diffusion 模型被包装了过多的数学描述。
   - 其他人表示同意，指出变分推断理论过于深奥，在处理方程式时更倾向于使用“雕刻比喻”。
- **Diffusion 并不一定优于自回归 (Autoregression)**：一名成员认为“Diffusion 本质上优于自回归”的观点是不正确的，障碍主要在于架构和规模。
   - 他们建议通过[重复上下文](https://arxiv.org/abs/2512.14982)或以非因果方式重新编码序列等改进手段可以弥合差距，并强调了当前 LLM 的设计局限性。
- **Score Parameterization 优于自回归规范**：一名成员质疑生成模型损失函数中对因果规范的需求，相比自回归方面，他们更倾向于参数化 `grad log p(x)` (score)。
   - 他们链接到了[一篇关于 Score Parameterization 的博客文章](https://yang-song.net/blog/2021/score/)，认为神经网络在无需确保分布曲线下面积积分为 1 的情况下更容易优化。
- **Byte-Level Prediction 模型实验**：一名成员寻求关于一个用于 **byte-level prediction**（词表大小为 256）的稠密 MoE 架构的反馈，该架构使用 13GB VRAM 包含 40M 参数，并调侃真正的 AGI 测试是它能否列举 LaTeX 图像标题。
   - 另一名成员幽默地评价了一个生成样本中的特定短语，称 *“研究表明，声明中的青少年是通过描述来描述的”是一个史诗级的从句*。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466120318874681437)** (6 messages): 

> `Discord event link issues, Google Meet` 


- **Discord 活动链接出现问题**：一名成员报告说 [Discord 活动链接](https://discord.com/events/987824841656791130/1463604897776664872)无法正常工作，导致他们无法加入**每日论文讨论**。
- **Google Meet 解决问题**：无法使用 Discord 链接的成员被引导通过 **Google Meet** 加入。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1465807859362041998)** (36 messages🔥): 

> `ChatGPT 套壳, Overleaf 杀手, Clawdbot 诈骗, Leetcode 挑战, AI 编程与技能保留` 


- ****ChatGPT 套壳无处不在****：成员们注意到大多数新出的“东西”只是 **ChatGPT 套壳**，并质疑仅仅封装现有模型的工具的价值。
   - 一位成员认为这些套壳是必要的，因为*如果你不做一个套壳来展示实际用途，大多数人根本不会去思考这个使用场景。*
- ****Clawdbot 诈骗用户****：有人评论说诈骗者很容易围绕现有工具创建套壳，并提到了 **Clawdbot 诈骗**。
   - 言外之意是 OpenAI 实际上也在*为其自己的工具制作套壳*。
- ****AI 不会取代技能****：尽管 AI 编程工具兴起，成员们认为编程能力可以重新习得，而现在产生代码的速度可能会阻碍真正的理解，并指向了 [Trinity Large 的博文](https://www.arcee.ai/blog/trinity-large)。
   - 有人指出，来自 LLM 的劣质实现不再像以前那样被看重，因为创建它的脑力和时间成本非常低。
- ****Google 在洗钱式获利？****：一位成员提出了一个*不严肃的阴谋论*，即 Google 的广告业务只是其从 Gmail、Workspaces 和搜索中获得的超额收益（alpha）的*洗钱操作*。
   - 这一讨论发生在思考 *Sama 等人的 Agent 是否也可能在读取会话*时。
- ****所有权和使用条款****：一位成员引用了 [OpenAI 使用条款](https://openai.com/policies/terms-of-use/)，其中规定用户保留输入的所有权并拥有输出。
   - 注意到除非用户选择退出，否则 OpenAI 可以使用内容来训练模型。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1465889714388336650)** (60 messages🔥🔥): 

> `AMD 模拟器调试打印, GitHub Actions 速度, tinygrad 中的 MULACC 修复, Egraphs 与 tinygrad, Mac MetalCompiler 改进` 


- **AMD 模拟器显示调试打印**：在新的 AMD 模拟器 (**AMD=1 MOCKGPU=1**) 中，设置 **DEBUG=3** 会在编译时打印所有指令，而 **DEBUG=6** 则在运行时打印，正如 [截图](https://cdn.discordapp.com/attachments/1068976834928193609/1465889714153193574/image.png?ex=697b686e&is=697a16ee&hm=485c88290bbec976b6b7ab93aed07b21a6a2ec8ba8b28806e14630c00b972b3c&) 所示。
- **通过代码优化加速 GitHub Actions**：讨论强调，提高 GitHub Actions 的速度应集中在优化代码上，而不是依赖更快的硬件或租用资源，并警告不要将指标优先于以*正确*的方式做事。
- **MULACC 融合修复**：提议在 `decompositions.py` 中添加一个模式来融合 **(x << n) + c → MULACC(x, 2^n, c)**，这会影响带 2 的幂常数的整数 MULACC，见 [PR 14387](https://github.com/tinygrad/tinygrad/pull/14387)。
- **用于通用修复的 Egraphs**：成员们讨论了使用 **egraphs** 来通用地修复问题，主张保持简单，并考虑在重写时标记其来源，以追踪重写过程中创建的等价关系。
- **改进 Mac MetalCompiler**：建议改进 Mac 上 **MetalCompiler** 的临时方案（hacks），特别关注减少行数并提高可读性的改进和清理。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1465806533995659365)** (4 messages): 

> `容器问题, macOS 信任校验, Gatekeeper 带来的开销, mojo build 中的代码签名步骤` 


- **通过额外参数解决容器问题**：一位用户通过在运行容器时添加 `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined`，或在 `.devcontainer/devcontainer.json` 中添加等效项，解决了容器问题。
   - 提供的解决方案确保容器具有正确配置的必要权限和安全选项，以便进行调试或追踪。
- **macOS 信任校验影响首次运行性能**：一位成员建议，首次运行与后续运行之间的性能差异可能是由于 macOS Gatekeeper 的*信任校验 (trust dance)*。
   - 他们指出，清除隔离属性 `xattr` 或进行 ad-hoc 代码签名可以缓解这一问题，并想知道 `mojo build` 中的代码签名步骤是否可以完全隐藏这一过程。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1466146040255090884)** (2 messages): 

> `Mojo-GTK 绑定, Mojo 对比 CUDA/HIP, Modular 团队更新` 


- **2 月的 Modular 社区会议将讨论 Mojo 的实力**：2 月的 Modular 社区会议将涵盖 **Mojo-GTK 绑定**、**Mojo 对比 CUDA/HIP** 的性能表现以及 **Modular 团队更新**。
   - 会议定于 **太平洋时间 2 月 2 日上午 10 点** 通过 Zoom 举行，更多详情请见 [Modular 论坛](https://forum.modular.com/t/february-community-meeting/2646)。
- **自动生成的 Mojo-GTK 绑定**：**Hammad Ali** 将介绍自动生成的 **Mojo GTK 绑定**。
   - 本次演讲将详细说明 GTK 绑定是如何自动生成的，这有望提升使用 **Mojo 开发 GUI** 的便利性。
- **Mojo 对比 CUDA/HIP 性能**：**Tatiana Melnichenko** 将分享在 **H100/MI300A** 上对比 **Mojo 与 CUDA/HIP** 的内存受限 (memory-bound) 带宽结果和计算受限 (compute-bound) 差距。
   - 此次演讲将深入探讨 **Mojo 相对于成熟 GPU 编程模型（如 CUDA/HIP）的性能特性**。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1465898202346819697)** (13 messages🔥): 

> `Mojo 中的编译器限制, Mojo 对 Pythonic 风格的偏离, 'out' 参数背后的原理, NVRO 的替代方案, Mojo 在 ORNL 的论文` 


- **切片语法导致 Mojo 中的 `__getitem__` 出错**：一位用户报告了在 Mojo 结构体中使用切片语法 (`0:2:1`) 调用 `__getitem__` 时遇到的错误，指出它仅在输入为 `Int` 或显式调用 `Slice()` 时有效，并寻求解决方法。
   - 错误信息为：*invalid call to '__getitem__': value passed to 'index' cannot be converted from slice initializer to 'Variant[Slice, Int]'*。
- **为什么 Mojo 放弃了 Pythonic 的 `out` 风格**：讨论围绕 Mojo 偏离 Pythonic 风格展开，特别是关于 `out` 参数。一位成员认为这种设计选择更接近 Fortran。
   - 另一位成员补充道，*Python 在这方面没有真正的等价物，因为在 Python 中它们仅仅是类型提示*。
- **`out` 参数的特性**：成员们讨论了 Mojo 中的 `out` 参数如何指定函数返回值的最终存放位置，这对于构造函数在 `self` 完全初始化之前对其进行赋值特别有用。
   - 一位成员解释说：*我知道至少对于构造函数，你需要一种在 “self” 完全初始化之前给它赋值的方法，而 `out self` 就是一种命名该位置的方式。*
- **`out` 作为 NVRO 的替代方案**：`out` 参数可作为具名返回值优化 (NVRO) 的替代方案，它为返回值的目标位置提供了保证，而不是依赖于编译器的优化。
   - 一位成员补充道：*你得到的是一种保证，而不是寄希望于编译器能弄明白。*
- **Mojo 在 ORNL 的文章发布**：一位成员分享了关于 *Mojo 在 ORNL (橡树岭国家实验室)* 的链接：[https://arxiv.org/html/2509.21039v1](https://arxiv.org/html/2509.21039v1)。
   - 未提供更多上下文。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1465971590893146202)** (4 messages): 

> `Qwen3 嵌入模型, Nightly 容器构建, 稳定的 MAX 版本` 


- **Qwen3 嵌入模型精度修复 PR 提交**：一位成员请求审查他们为 [Qwen3 嵌入模型提交的 PR](https://github.com/modular/modular/pull/5823)，称该修复对于获得更好的精度至关重要。
   - 另一位成员回应称，新的修复可能不会被纳入即将发布的版本，但在 Nightly 版本中可用。
- **Nightly 容器构建即将推出**：一位成员确认，由于提供了 Nightly 容器构建，在合并后不久，这些更改就应该可以在其 POC (概念验证) 中使用。
   - 他们还分享了一个将修复简化为单行的分支：[https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal)。
- **稳定的 MAX 版本结果得到提升**：该成员提到，合并该 PR 将有助于他人在通过稳定的 MAX 版本尝试该模型时获得更好的结果。
   - 他们在[此处](https://github.com/modular/modular/compare/main...sbrunk:modular:qwen3-embedding-fix-norm-minimal)将修复简化为了单行代码。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1465885050825347347)** (9 条消息🔥): 

> `Manus Credit 消耗、云浏览器问题、AI Engineer 自我介绍、跨对话上下文` 


- ****Manus 的 Credit 消耗能力****：一位用户注意到 **Manus** 在保持相同工作质量的前提下，似乎消耗了更少的 credits，并询问 credit 使用效率是否有所提升。
   - 目前尚未提供有关 **Manus** 的 credit 消耗算法可能发生变化的进一步详情或确认。
- ****云浏览器难题与 Manus 支持****：一位用户在使用 **cloud browser** 时遇到问题，收到报错信息称“服务器不可用”且网站无法加载。
   - Manus 支持团队请求用户通过私信（DMs）提供电子邮件、会话链接和 Manus User ID，以便进一步调查该问题。
- ****AI Engineer 精通 LLM 系统与集成****：一位 **AI + Full Stack Engineer** 介绍了自己，强调了其在 LLM 系统、autonomous agents、workflow automation 以及 multimodal AI 方面的专长，并分享了其核心技能，如 [DSPy](https://dsppy.ai/)、[LangChain](https://www.langchain.com/)、[AutoGen](https://microsoft.github.io/autogen/) 和 [CrewAI](https://www.crewai.com/)。
- ****上下文难题：社区渴望 Manus 支持跨对话上下文****：一位用户建议让 **Manus** 能够访问来自其他对话的 context（上下文），认为这将是一个“游戏规则改变者（game changer）”，表明了用户希望提升 AI 回复时的上下文感知能力。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1465862038147567808)** (5 条消息): 

> `Prompt Optimizers, llmlingua, DSPy Skills, Claude Code Skills, DSPy ReAct Agent` 


- ****Prompt Optimizers 寻找用户****：一位成员询问是否有人具有使用 **prompt optimizers** 的经验。
   - 另一位成员随后询问是否有人尝试过在 dspy 模块中使用 Skills。
- ****分享了 llmlingua 链接****：一位成员分享了 [llmlingua.com](https://llmlingua.com/) 的链接。
   - 该链接是针对另一位成员询问 **prompt optimizers** 使用经验而提供的。
- ****DSPy ReAct Agent 渴望 Skills****：一位成员询问如何将 **Claude code skills**（定义为关联了 .py 脚本的 .md 文件）集成到 **DSPy ReAct agent** 中。
   - 他们希望 DSPy ReAct agent 或类似工具能够使用这些技能。