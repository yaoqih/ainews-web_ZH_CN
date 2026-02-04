---
companies:
- zhipu-ai
- lmsys
- vllm
- novita-labs
- ollama
- alibaba
- allenai
- cognition
- cursor
date: '2026-02-03T05:44:39.731046Z'
description: '**智谱 AI (Zhipu AI)** 发布了 **GLM-OCR**，这是一款轻量级的 **0.9B** 多模态 OCR 模型。它在复杂文档理解方面表现卓越，取得了顶级的基准测试评分，并获得了
  **lmsys**、**vllm** 和 **novita labs** 的首日部署支持。同时，**Ollama** 也已支持该模型的本地优先使用，实现便捷的离线操作。


  **阿里巴巴**发布了 **Qwen3-Coder-Next**，这是一款 **80B 的 MoE**（混合专家）模型，其**激活参数仅为 3B**。该模型专为编程智能体（coding
  agents）设计，拥有高达 **256K 的超长上下文窗口**，并在 **80 万个可验证任务**上进行了训练，在 SWE-Bench Verified 测试中取得了超过
  **70%** 的成绩。


  在开源编程生态系统方面，**Allen AI** 宣布推出 **SERA-14B**，这是一款对端侧设备友好的编程模型，并附带了新的数据集。此外，新兴的**上下文图谱
  (Context Graphs)** 概念被视为数据和智能体溯源的一种极具前景的框架，例如 **Cursor 的 Agent Trace** 计划为编程智能体明确了上下文图谱，强调其在提升智能体性能和推动客户驱动型应用方面的潜力。


  这些进展反映了**多模态**、**长上下文**、**混合专家模型 (MoE)** 以及**智能体编程模型**领域的持续创新。'
id: MjAyNi0w
models:
- glm-ocr
- qwen3-coder-next
- sera-14b
people:
- jaya_gupta
- dharmesh_shah
title: 上下文图谱：是噱头还是真正的万亿美元机遇？
topics:
- multimodality
- ocr
- long-context
- mixture-of-experts
- agentic-coding-models
- context-graphs
- benchmarking
- model-deployment
- model-optimization
- model-training
---

**宁静的一天，让我们聚焦一个升温的话题。**

> 2026年1月30日至2月2日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务器（**254** 个频道，**14979** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1408 分钟**。**我们的新网站**现已上线，支持完整的元数据搜索，并以充满氛围感的形式呈现了所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

我们针对宁静日的策略是，现在将重点介绍那些长期运行、发酵较慢的故事，这些故事虽然在某一天可能不会冲到榜首，但对 AI Engineer 来说可能具有更广泛的历史意义。今天的 Lightning Pod（我们仅在 YouTube 发布的短格式内容）聚焦 Context Graphs 这一话题，该话题由 [Jaya Gupta 于 12 月底在 X 上发起](https://x.com/JayaGup10/status/2003525933534179480)，并启发了包括前嘉宾 [Dharmesh Shah](http://latent.space/p/dharmesh) 在内的人士（他对该话题[持保留意见](https://simple.ai/p/what-are-context-graphs)）。我们采访了两位作者关于反响的看法：

https://youtu.be/zP8P7hJXwE0

这是典型的“思想领导力”（thoughtleading）入门，但确实很有帮助——可以肯定的是，每一位构建数据/上下文工程产品的创始人都去找他们，并宣称他们的股东名单（cap table）里有 Context Graphs 的提出者。但这篇文章的问题在于，它承诺了很多（正如标题所示），但具体的操作建议却不多。

[最近](https://x.com/cognition/status/2017057457332506846)，我也将 Cursor 的 [Agent Trace 计划](https://agent-trace.dev/) 描述为代码领域的 “Context Graph”：

这是第一个在公司之间达成一致的、针对特定领域（编程 Agent）的 Context Graph 实际规范。它是否具有持久的生命力仍有待观察，这主要取决于：1) Agent 性能的显著提升，以及 2) 客户支持该规范的压力。从第一性原理来看，这个想法（将散落在 “Data Mesh” 各处的决策追踪、异常和先例捕获到 LLM 的上下文中）似乎极具吸引力，但当然，细节决定成败。

---

# AI Twitter 简报

**智谱 AI 发布 GLM‑OCR (0.9B) 及其跨技术栈的首日（day‑0）部署支持**

- **GLM‑OCR (面向复杂文档的多模态 OCR)**：智谱发布了 **GLM‑OCR**，定位为轻量级、可部署的 **0.9B** 模型，用于现实世界的文档理解（表格、公式、信息提取、凌乱布局）。据报道，它在 **OmniDocBench v1.5 (94.62)** 上排名第一，并强调了对低延迟/高并发的友好性。参阅来自 [@lmsysorg](https://twitter.com/lmsysorg/status/2018521181146751486)（SGLang 集成 + PR/cookbook 链接）和 [@vllm_project](https://twitter.com/vllm_project/status/2018582480518091083)（vLLM 首日支持）的生态系统“首日支持”公告，以及来自 [@novita_labs](https://twitter.com/novita_labs/status/2018565896013574225) 的部署推广。
- **本地优先可用性**：Ollama 立即发布了本地拉取和 API 使用功能（“将图像拖放到终端”，JSON 格式输出），使 GLM‑OCR 可以轻松离线运行：[@ollama](https://twitter.com/ollama/status/2018525802057396411) 和库链接 [@ollama](https://twitter.com/ollama/status/2018525804733575492)。社区对比还声称其质量优于 PaddleOCR/DeepSeek OCR：[@bdsqlsz](https://twitter.com/bdsqlsz/status/2018663915404841212)。LlamaIndex 强调了基准测试的更替（声称比之前的顶级模型快 50–100%）以及持续的评测集成：[@jerryjliu0](https://twitter.com/jerryjliu0/status/2018713059359899729)。

**Agent 编程模型与测试框架：Qwen3‑Coder‑Next (80B@3B)、SERA‑14B 以及 “skills/MCP” 工具接口的收敛**

- **Qwen3‑Coder‑Next**: 阿里巴巴发布了 **Qwen3‑Coder‑Next**，这是一个开源权重的 **80B MoE** 模型，仅有 **3B 激活**参数，定位用于 *Coding Agent + 本地开发*，具备 **256K 上下文**，并使用了 **800K 可验证任务 + 可执行环境**进行训练。他们声称在配合 SWE‑Agent 支架时，其 **SWE‑Bench Verified** 表现 **>70%**，并具有强大的 Agent 基准测试效率：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2018718453570707465) 以及基准测试说明 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2018719026558664987)。独立/相关的总结：[@UnslothAI](https://twitter.com/UnslothAI/status/2018718997584474191)（关于显存占用 + GGUF 指南）以及对高效长上下文 Attention 选择的评论（例如讨论中提到的 “Gated DeltaNet”）：[@eliebakouch](https://twitter.com/eliebakouch/status/2018730622358073384)。vLLM 在 **vLLM 0.15.0** 中提供了首日支持：[@vllm_project](https://twitter.com/vllm_project/status/2018742511502856568)。
- **开源 Coding Agent 生态 (Ai2)**：Allen AI 宣布了 **SERA‑14B**（端侧友好型编程模型）以及更新的开放数据集，其中包括**原始轨迹 (raw trajectories) + 验证元数据**：[@allen_ai](https://twitter.com/allen_ai/status/2018741177734910166) 和数据集/模型详情线程指针 [@ethnlshn](https://twitter.com/ethnlshn/status/2018746924803969317)。
- **Harness（运行环境/框架）优于模型（反复出现的主题）**：多条推文达成共识，认为 Agent 的杠杆作用正日益体现在 **harness**（权限、记忆、工作流、可逆性）上，而不仅仅是原始模型的 IQ。一个清晰的阐述：[@sarahmsachs](https://twitter.com/sarahmsachs/status/2018720637691572634)。
- **Agent “技能 (skills)” 目录 + 协议的标准化**：
  - **Agent Client Protocol (ACP)**：被提议作为 JSON‑RPC 标准，旨在统一 Gemini CLI / Claude Code / Codex CLI / OpenClaw 等工具中 Agent 与编辑器之间的通信，支持 stdio/HTTP、文件访问、终端、权限和流式更新：[@_philschmid](https://twitter.com/_philschmid/status/2018706591776756216)。
  - **Skills 与 MCP 工具**：LlamaIndex 对比了 “skills”（简单但脆弱，由自然语言解释）与 MCP 服务器（更具确定性的 Schema、更多的配置工作、存在网络延迟但支持中心化更新）：[@llama_index](https://twitter.com/llama_index/status/2018749615907213457) 以及后续跟进 [@jerryjliu0](https://twitter.com/jerryjliu0/status/2018797672258490666), [@itsclelia](https://twitter.com/itsclelia/status/2018821269752611102)。同时，“`.agents/skills` 正在成为默认标准”被明确指出（Codex/OpenCode/Copilot/Cursor 已采用；Claude Code 尚未采用）：[@theo](https://twitter.com/theo/status/2018819504252608710)。

**Coding Agent 产品：Codex App 采用率、Claude Code 共享 + Apple Xcode 集成**

- **Codex App 势头 + 推理加速**：
  - Sam Altman 报告**首日下载量突破 20 万次**：[@sama](https://twitter.com/sama/status/2018734731437985930)。
  - OpenAI 为 API 客户发布了**提速 40% 的 GPT‑5.2 和 GPT‑5.2‑Codex**（“权重相同，延迟更低”）：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018838297221726482)。
  - OpenAI DevRel 宣布 Codex 集成到 **Xcode 26.3**：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/2018796432443244897)。
- **Claude Code 产品迭代**：
  - Claude Code 支持在 Web/桌面/移动端之间进行**会话共享**：[@lydiahallie](https://twitter.com/lydiahallie/status/2018740156359229883)。
  - 社区对 “等待 Sonnet 5” 的猜测占据主导，包括声称 **Anthropic 图像模型已在 LMArena 上线**：[@kimmonismus](https://twitter.com/kimmonismus/status/2018689719324791022) 以及 “Claude Image 即将到来” 的传闻：[@kimmonismus](https://twitter.com/kimmonismus/status/2018669423402660082)。
- **Apple Xcode + Claude Agent SDK**：Anthropic 宣布通过 **Claude Agent SDK**（支持子 Agent/后台任务/插件）实现 **Xcode 原生集成**，从而将类似 Claude Code 的能力直接引入 Apple 开发工作流：[@AnthropicAI](https://twitter.com/AnthropicAI/status/2018771170938724682)。这是 “Agent 进入 IDE (agent-in-the-IDE)” 成为原生功能的显著一步。

**Agent 基础设施与可观测性：以 Trace（轨迹）作为事实来源、深入 Agent 评估以及超越 RAG 的记忆机制**

- **可观测性从代码转向 Traces**：LangChain 认为，对于 Agent 系统，运行时决策发生在模型内部，因此 **Traces** 成为调试/理解的主要产物。参见：[@LangChain](https://twitter.com/LangChain/status/2018739770495512880)。
- **如何评估 Deep Agents**：LangChain 的评估指南强调了针对具体案例定制成功标准、单步回归检查、全回合及多回合评估，以及干净且可复现的环境：[@LangChain](https://twitter.com/LangChain/status/2018769968515404212)。
- **DeepAgents 版本发布 (JS/CLI/运行时后端)**：
  - deepagents@1.6.2 修复了（Checkpoint 恢复、大文件死循环、简化 Toolcall 中间件）：[@LangChain_JS](https://twitter.com/LangChain_JS/status/2018731100441620517)。
  - DeepAgents 0.3.10 增加了 **LocalShellBackend**，用于在本地机器上运行代码：[@sydneyrunkle](https://twitter.com/sydneyrunkle/status/2018788505082859863)。
  - deepagents-cli 0.0.16 提升了 Shell 运行的控制力与可见性：[@masondrxy](https://twitter.com/masondrxy/status/2018741344835870820)。
- **内存：“RAG 并非为 Agent 内存而设计”**：DAIR 的 **xMemory** 提出了分层检索（主题/语义/情节/消息），在保留证据链的同时减少冗余，相比朴素的 top-k 相似度检索，以更少的 Token 展现出更好的 LoCoMo 分数：[@dair_ai](https://twitter.com/dair_ai/status/2018765444702982395)。
- **文件系统作为 Agent 上下文草稿板**：“文件优先（files-first）”工作流（将产物存储在上下文之外，避免窗口膨胀）得到了 DeepAgents 的设计和评论支持：[@LangChain_JS](https://twitter.com/LangChain_JS/status/2018732184694374669)。

**基准测试与评估信号：METR 时间跨度、WorldVQA、Text/Search/Image Arena 更新以及 ARC-AGI 进展**

- **Gemini 3 Pro 的 METR 时间跨度**：METR 估计在扩展的软件任务套件（含 CI）上约为 **4 小时（50% 时间跨度）**：[@METR_Evals](https://twitter.com/METR_Evals/status/2018752230376210586)。这种“时间跨度”评估正逐渐成为超越静态编程基准测试的关键 Agent 能力指标。
- **WorldVQA (Moonshot/Kimi)**：Moonshot 推出了 **WorldVQA**，用于独立于推理能力来衡量“以视觉为中心的原子级世界知识”，明确尝试将记忆与推理质量解耦。数据集：包含 9 个类别的 **3,500 个 VQA 对**，具有语言和文化多样性：[@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2018697552456257945)。
- **Arena 排行榜**：
  - **Text Arena (开源模型，2026年1月)**：第一名 **Kimi-K2.5-Thinking**，第二名 **GLM-4.7**，第三名 **Qwen3-235B-A22B Instruct**：[@arena](https://twitter.com/arena/status/2018727506850033854)。
  - **Search Arena 更新**：Google 的 **gemini-3-flash-grounding** 领先；OpenAI 非推理类搜索进入前 5；列出了最佳的 Claude 搜索变体：[@arena](https://twitter.com/arena/status/2018760874178342975)。
  - **Image Arena 帕累托前沿**：Arena 发布了文生图和图像编辑的**质量 vs 单图价格**前沿（值得注意的是，根据成本约束，数个 OpenAI/Google/Flux/Tencent 模型均处于前沿位置）：[@arena](https://twitter.com/arena/status/2018787949840896119) 以及编辑前沿 [@arena](https://twitter.com/arena/status/2018792314878234704)。
- **ARC-AGI**：ARC Prize 报告了一个新的 **SOTA 公开提交结果**（附带成本/任务数据），该结果基于 **GPT-5.2** 集成模型：[@arcprize](https://twitter.com/arcprize/status/2018746794310766668)。此外，社区关于 ARC-AGI-2 进展速度的讨论也在持续：[@kimmonismus](https://twitter.com/kimmonismus/status/2018800964891984181)。

**效率、内核与训练/推理管道：fp8 训练、Blackwell 吞吐量以及作为推理时代数据工程的“上下文工程”**

- **Karpathy 关于 fp8 训练的笔记（偏重实践而非理论）**：他报告称启用 **fp8 训练**后，将“训练至 GPT-2 水平的时间”缩短至 **2.91 小时**，讨论了真正的瓶颈（并非纯粹的计算限制）、缩放转换的开销、GEMM 尺寸以及每步的质量下降；并指出大模型在 fp8 中收益更高（引用了 torchao 更大的增益）：[@karpathy](https://twitter.com/karpathy/status/2018804068874064198)。
- **vLLM + NVIDIA Blackwell 优化**：vLLM 报告称，通过 FlashInfer 集成、torch.compile 融合、异步调度和流间隔优化，**gpt-oss-120b** 在 Blackwell 上的性能大幅提升：[@vllm_project](https://twitter.com/vllm_project/status/2018859316258931161)。
- **推理是头等工程领域**：“上下文工程（Context engineering）对于推理的重要性，正如数据工程对于训练一样”被简洁地提出（并被反复引用）：[@swyx](https://twitter.com/swyx/status/2018533744442057115)。这种观点也体现在团队关于文件系统、工具选择（Skills vs MCP）、缓存和测试框架设计的讨论中。

---

### 热门推文（按互动率）
- [市值最高公司的 CEO 在街道中间举办“发布会”](https://twitter.com/yacinelearning/status/2018689145086898466) — 大规模互动的梗/事件评论。
- [SpaceX 收购 xAI / “建立星际文明”](https://twitter.com/elonmusk/status/2018784828129243614)。
- [Codex 应用首日下载量：“超过 20 万”](https://twitter.com/sama/status/2018734731437985930)。
- [Apple Xcode 集成 Claude Agent SDK](https://twitter.com/AnthropicAI/status/2018771170938724682)。
- [OpenAI 聘请 Head of Preparedness](https://twitter.com/sama/status/2018813527780463027)。
- [GPT‑5.2 & GPT‑5.2‑Codex 推理速度提升 40%（推理栈已优化）](https://twitter.com/OpenAIDevs/status/2018838297221726482)。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen3-Coder-Next 发布

  - **[Qwen/Qwen3-Coder-Next · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1quvqs9/qwenqwen3codernext_hugging_face/)** (热度: 842): **Qwen3-Coder-Next** 是一款专为编程设计的尖端语言模型，在总计 `80B` 的参数中拥有 `3B 激活参数`，实现了与激活参数量多出 `10-20x` 的模型相当的性能。它支持长程推理（long-horizon reasoning）等高级功能，并具备 `256k` 的上下文长度，非常适合与 IDE 集成。该架构包含 `48 层`、门控注意力机制（gated attention）和专家混合（mixture of experts），适用于动态编码任务。可以使用 **SGLang** 或 **vLLM** 进行部署，为了获得最佳性能需要特定版本。更多详情可以在 [原文章](https://huggingface.co/Qwen/Qwen3-Coder-Next) 中查看。一位评论者对该模型的性能表示怀疑，质疑一个 `3B 激活参数` 的模型是否真的能匹配像 Sonnet 4.5 这样更大型模型的质量，这表明这些说法还需要进一步验证。

    - danielhanchen 讨论了为 Qwen3-Coder-Next 发布动态 Unsloth GGUF 的事宜，重点介绍了即将发布的 Fp8-Dynamic 和 MXFP4 MoE GGUF。这些格式旨在优化模型性能和效率，特别是在资源受限的环境中。链接的指南提供了在本地使用 Claude Code 和 Codex 配合 Qwen3-Coder-Next 的说明，这对于希望将这些模型集成到工作流中的开发者非常有益。
    - Ok_Knowledge_8259 对 30 亿激活参数模型能匹配 Sonnet 4.5 等更大模型质量的说法表示怀疑。这一评论反映了 AI 社区对模型大小与性能权衡的普遍担忧，认为虽然较小的模型效率更高，但并不总能达到大型模型的质量水平。
    - Septerium 指出，虽然原始的 Qwen3 Next 在基准测试中表现良好，但用户体验欠佳。这突出了 AI 模型部署中的一个关键问题：高基准测试分数并不总是能转化为实际的可用性，表明需要改进用户界面和集成方式，以充分利用模型的能力。

  - **[Qwen3-Coder-Next 现已发布！](https://www.reddit.com/r/LocalLLM/comments/1quw0cf/qwen3codernext_is_out_now/)** (热度: 228): 图片宣布了 **Qwen3-Coder-Next** 的发布，这是一个具有 `3B` 激活参数的 `80B` MoE (Mixture of Experts) 模型，专为高效编码任务和本地部署而设计。它强调了模型在长程推理和复杂工具使用方面的能力，运行需要 `46GB` 的 RAM/VRAM。图片中的图表突出了其与其他模型相比的性能效率，展示了其以更少的激活参数实现高性能的能力。该模型因其快速的 Agent 编码能力而备受关注。一位用户询问了在没有 VRAM 的情况下使用 `64GB` RAM 运行模型的可行性，表现出对其硬件要求的兴趣。另一条评论质疑该模型的性能水平，将其与 "Sonnet 4.5" 进行比较，表现出对其能力的怀疑或好奇。此外，还有人评论称缺少与 "Devstral 2" 的对比，暗示了对针对特定模型进行基准测试的期望。

    - 一位用户询问是否可以在没有 VRAM 的情况下通过 64GB RAM 运行 Qwen3-Coder-Next，这表明了对模型内存效率和潜在纯 CPU 部署的兴趣。这突显了理解模型硬件要求以及针对非 GPU 环境进行优化的必要性。
    - 另一位用户通过将其与 "Sonnet 4.5 级别" 进行比较来质疑该模型的性能，对模型的能力或针对特定基准测试的潜在过度优化表示怀疑。这反映了 AI 模型评估中的一个共同担忧，即性能可能是为了在某些测试中脱颖而出而量身定制的，而非针对通用场景。
    - 针对配备 28GB NVIDIA VRAM 和 96GB DDR5 RAM 的配置，提出了关于合适量化（quantization）的技术咨询。这表明关注点在于针对特定硬件配置优化模型性能，这对于在高性能计算环境中实现效率和速度的最大化至关重要。

### 2. ACE-Step 1.5 音频模型发布

  - **[ACE-Step-1.5 刚刚发布。这是一款采用 MIT 许可的开源音频生成模型，其性能接近 Suno 等商业平台](https://www.reddit.com/r/LocalLLaMA/comments/1quzwjf/acestep15_has_just_been_released_its_an/)** (活跃度: 408): **ACE-Step-1.5** 是一款基于 MIT 许可发布的开源音频生成模型，提供的性能可与 **Suno** 等商业平台媲美。它支持 **LoRAs**、针对不同需求的多样化模型，以及翻唱 (cover) 和重绘 (repainting) 等功能。该模型已集成至 **Comfy**，并在 **HuggingFace** 上提供演示。此次发布标志着开源音频生成领域的重大进步，缩小了与顶尖商业解决方案的差距。一条评论表达了对模型提示词遵从性 (prompt adherence) 的怀疑，指出演示提示词往往与输出不符，暗示其在指令遵循方面可能存在局限性。

    - ACE-Step-1.5 的发布备受关注，作为一款采用 MIT 许可的开源音频生成模型，据报道其性能接近 Suno 等商业平台。该模型的效率也是一大亮点，在 A100 GPU 上仅需 2 秒即可生成输出，显示了显著的计算优化。
    - 针对模型对输入提示词的遵从性存在质疑，部分用户观察到演示提示词与生成的输出之间并不紧密匹配。这引发了对模型指令遵循能力以及提示词处理有效性的疑问。
    - 讨论还涉及了该模型生成纯乐器音乐的能力。一位用户将其与 HeartMuLa 进行了对比，指出虽然 HeartMuLa 无法生成不带人声的纯乐器曲，但目前尚不清楚 ACE-Step-1.5 是否能满足这一特定需求，这标志着一个潜在的进一步探索或开发领域。

  - **[Suno 的开源版本终于来了：ACE-Step 1.5](https://www.reddit.com/r/LocalLLaMA/comments/1quxtkj/the_opensource_version_of_suno_is_finally_here/)** (活跃度: 319): **ACE-Step 1.5** 是一款开源音乐生成模型，在标准评估指标上超越了 **Suno**。它可以在 **A100 GPU** 上约 `2 秒` 内生成一首完整的歌曲，并能在拥有约 `4GB VRAM` 的普通本地 PC 上运行，在 **RTX 3090** 上生成时间不到 `10 秒`。该模型支持 **LoRA**，可以用极少量的数据训练自定义风格，并采用 **MIT 许可** 发布，允许免费商业使用。其数据集包含完全授权的数据和合成数据。该项目完全开源，[GitHub 资源](https://github.com/ace-step/ACE-Step-1.5) 提供了权重、训练代码、LoRA 代码和研究论文。评论者注意到该模型较先前版本有显著改进，但批评其与 **Suno v3** 相比，在指令遵循和连贯性方面仍有不足。尽管存在这些问题，音频质量仍被认为是不错的，该模型被视为 Suno 的一个具有创造性的替代方案。人们对版本 2 的发布充满期待。

    - TheRealMasonMac 指出 ACE-Step 1.5 较其前代产品有显著改进，但在指令遵循和连贯性方面仍落后于 Suno v3。然而，音频质量被认为很好，该模型被描述为具有创意且与 Suno 不同，暗示它可以作为未来开发的坚实基础。
    - Different_Fix_2217 提供了 ACE-Step 1.5 生成的音频示例，表明该模型在处理长且详细的提示词时表现良好，并能处理负面提示词。这表明其在输入处理方面具有灵活性，对希望尝试各种提示词风格的用户非常有益。


### 3. 本地 LLM 的发展与对比

  - **[128GB 设备有了新的本地 LLM 王者：Step-3.5-Flash-int4](https://www.reddit.com/r/LocalLLaMA/comments/1qtvo4r/128gb_devices_have_a_new_local_llm_king/)** (活跃度: 619): **`Step-3.5-Flash-int4`** 模型已在 [Hugging Face](http://huggingface.co/stepfun-ai/Step-3.5-Flash-Int4) 上线，这是一款针对拥有 `128GB` RAM 的设备（如 M1 Ultra Mac Studio）优化的新型本地 LLM。它支持 `256k` 的全上下文长度，并展现出极高的 RAM 使用效率。使用 `llama-bench` 进行的基准测试显示，在 `100k` prefill 情况下性能惊人，对 CLI 编程 Agent 仍保持可用性。该模型需要自定义的 `llama.cpp` 分支才能执行，鉴于其出色的性能，未来有可能获得主线支持。评论者对其在 Strix Halo 等不同硬件上的表现感到好奇，并对潜在的 NVFP4 版本表示关注。此外还有关于模型名称的轻松调侃。

- Step-3.5-Flash-Int4 模型在 AMD Strix Halo (Minisforum MS S1 Max) 上使用 ROCm 7.1.1 的基准测试结果显示出令人印象深刻的性能，在 `pp4096` 测试中吞吐量达到每秒 `258.82 ± 3.15` tokens。这表明该模型可以高效地处理全上下文拟合，使其成为 128GB 设备上本地 LLM 任务的强力竞争者。
- 不同后端上的性能对比显示，Step-3.5-Flash-Int4 模型在 ROCm 上表现最佳，而在使用 Vulkan-amdvlk 和 Vulkan-radv 时吞吐量显著下降。例如，Vulkan-amdvlk 上的 `pp4096` 测试结果为每秒 `153.04 ± 0.30` tokens，而 Vulkan-radv 达到 `164.20 ± 1.30`，表明 ROCm 是该模型的最优后端。
- Step-3.5-Flash-Int4 模型在 `tg512` 测试中的表现随后端不同而显著变化，ROCm 达到每秒 `22.93 ± 0.00` tokens，而 Vulkan-amdvlk 和 Vulkan-radv 的性能则低得多，分别为每秒 `2.50 ± 0.00` 和 `27.86 ± 0.00` tokens。这凸显了后端选择在优化模型性能中的重要性。

- **[本地模型完全取代订阅服务](https://www.reddit.com/r/LocalLLM/comments/1qtuwn5/local_model_fully_replacing_subscription_service/)** (热度: 270): **该帖子讨论了本地模型的有效性，特别是运行在拥有 `24GB` 内存的 MacBook Pro M4 Pro 上的 **Ollama + GPT-OSS:20b**，认为它可以在处理非复杂查询时取代 ChatGPT 等订阅服务。用户强调了该模型的速度和质量，指出它在研究查询和基础编程等任务中表现良好。一条评论建议在 Apple Silicon 上使用基于 `mlx` 的模型，以获得 `40%` 的每秒 token 速度提升，并可通过 **LMstudio** 获取。另一条评论指出，**GPT-OSS:20b** 使用 `17GB` VRAM 即可高效运行 `128k` 上下文，为其他 GPU 任务留有空间。讨论还涉及构建本地 Agent 框架以匹配 **Claude** 等订阅模型的能力，重点在于整合工具和技能以增强本地模型性能。** 评论者辩论了本地模型与订阅服务的效率，一些人认为 **Claude** 在复杂任务上仍然优于本地选项。还有关于有效执行 tool-calling Agent 的最小模型尺寸的讨论，建议以 `30b` 作为可靠性能的基准。

    - **coldy___** 强调了在 Apple Silicon 上使用基于 MLX 模型的性能优势，指出每秒 token 速度可能有 `40%` 的提升。他们建议使用 LM Studio 来访问这些模型，特别是针对该硬件优化的 `gpt-oss 20b` 模型。
    - **generousone** 讨论了 `gpt-oss:20b` 模型的效率，该模型仅使用 `17GB` VRAM 即可运行完整的 `128k` 上下文。这为其他 GPU 密集型任务留出了空间，使其成为拥有 `24GB` VRAM 用户的实用选择。他们承认它不如 ChatGPT 或 Claude 等商业模型先进，但发现它足以胜任许多任务。
    - **2BucChuck** 分享了关于构建本地 Agent 框架以克服本地模型局限性的见解，并针对 Agent 任务测试了 `Gemma32` 等模型。他们建议有效执行 tool-calling Agent 的最小模型尺寸为 `30B`，并指出较小的模型通常表现不佳。其目标是通过将工具和技能集成到本地模型中，来匹配订阅服务的功能。

- **[新的 1.4B 模型维多利亚时代 LLM - Violet](https://www.reddit.com/r/LocalLLM/comments/1quip6h/new_14b_model_victorian_llm_violet/)** (热度: 67): **该帖子介绍了 **Violet**，这是一个拥有 14 亿参数的新 LLM，完全在维多利亚时代（1800-1899年）的数据上训练，旨在创建一个来源合规的公共领域模型。该模型从零开始开发，使用了来自 Internet Archive、Project Gutenberg 和大英图书馆 (British National Library) 等来源的数据，并包含供本地浏览器使用的 ONNX 量化版本。该模型因其叙事散文能力而受到关注，但在推理和历史偏见（如性别误判）方面存在局限性。该项目还具有一个独特的带有情绪化头像的聊天变体，模型可在 [Hugging Face](https://huggingface.co/zakarth/violet-1b4-chat) 上获取，演示链接见[此处](https://huggingface.co/spaces/zakarth/violetdemo)。** 一位评论者询问该模型理解现代短语的能力，质疑它是否只能用维多利亚时代的英语方言交流，暗示其在理解当代语言方面可能存在局限。

- thirsty_pretzelzz 提出了关于维多利亚时代 LLM 语言能力的有趣观点，质疑其是否只能使用维多利亚时代英格兰的方言进行交流。这暗示了在理解现代短语方面可能存在局限性，从而可能影响其在当代语境中的适用性。
- avanlabs 表示有兴趣在特定数据集上训练类似模型，以便在小型设备上部署。他们请求提供能够深入了解构建和优化小型语言模型 (SLMs) 的资源或博客，表明其关注点在于高效的模型训练和部署策略。


## 技术性较弱的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Sonnet 5 与 Gemini 3.5 发布讨论

  - **[Sonnet 5 release on Feb 3](https://www.reddit.com/r/ClaudeAI/comments/1qtm9ix/sonnet_5_release_on_feb_3/)** (活跃度: 2328): **关于代号为 “Fennec” 的 Claude Sonnet 5 的泄露细节表明，它比之前的模型有显著进步。正如 Vertex AI 错误日志所示，潜在发布日期为 2026 年 2 月 3 日。传闻其价格比 Claude Opus 4.5 便宜 50%，同时保持 `1M token` 的上下文窗口并提供更快的性能，这可能归功于在 Google TPUs 上的优化。据称该模型还具有 “Dev Team” 模式，允许自主子 Agent 协作构建功能。基准测试声称其在 SWE-Bench 上超过了 `80.9%`，表现优于目前的编程模型。** 人们对发布时间持怀疑态度，一些用户认为错误日志并不能结论性地证明该模型的存在或其发布日期。此外，人们还对大上下文窗口中的准确度下降表示担忧，这是之前模型中存在的问题。

    - andrew_kirfman 对 Sonnet 5 的发布时间表示怀疑，提到 Vertex API 端点的 404 错误并不能确认该模型的存在。他们强调 Anthropic 的模型 ID 通常反映模型 checkpoint 的创建日期，而不是发布日期，并以 Opus 4.5 的 ID `20251101` 为例。他们对在发布标签中使用未来日期表示怀疑，这在软件发布中并不常见。
    - andrew_kirfman 还提到了 Sonnet 5 具有 100 万 token 上下文的潜力，并指出之前的模型如 Sonnet 4 和 4.5 已经通过 API 提供了这一功能。然而，他们指出准确度下降是这些模型的一个问题，这表明在该领域的改进对于赢得对新版本的信任是必要的。

  - **[Claude Sonnet 5: The “Fennec” Leaks](https://www.reddit.com/r/Bard/comments/1qtnkhu/claude_sonnet_5_the_fennec_leaks/)** (活跃度: 193): **这张图片是 Pankaj Kumar 讨论关于代号为 “Fennec” 的 “Claude Sonnet 5” 泄露信息的推文。它强调了 2026 年 2 月 3 日的潜在发布日期、极具竞争力的价格以及 TPU 加速和专门的子 Agent 等先进功能。传闻该模型比其前代产品更便宜、更快，具有巨大的上下文窗口和极高的基准测试性能。此外，它还暗示该模型已经整合到 Google 的基础设施中。[图片链接](https://i.redd.it/lmphdjb601hg1.png)** 评论者对泄露的可信度以及声称的 “100 万上下文” 能力的可行性表示怀疑，并指出当前模型在处理小得多的上下文大小时都表现吃力。

    - DavidAdamsAuthor 对 Claude 模型的 “100 万上下文” 说法表示怀疑，并指出在实际使用中，即使在 “250k” 上下文下，也会出现明显的 “能力退化和对关键数据的遗忘”。这表明该模型在处理大上下文大小时可能存在性能局限，这可能会影响其在需要大量记忆的任务中的有效性。

- **[Sonnet 5 将于周三发布，Gemini 3.5 在哪里？](https://www.reddit.com/r/Bard/comments/1qtmi53/sonnet_5_being_release_on_wednesday_where_is/)** (Activity: 182): **Claude Sonnet 5** 预计即将发布，传闻称其价格将比前代 **Claude Opus 4.5** 便宜 `50%`，同时提供更卓越的性能。该模型内部代号为 "Fennec"，据报道比 **Gemini** 的 **"Snow Bunny"** 领先一代，根据 **Vertex AI** 错误日志显示，预计将于 2026 年 2 月 3 日发布。它保留了 `1M token context window`，并针对 **Google TPUs** 进行了优化，承诺更快的处理速度和更低的延迟。值得注意的是，它可以生成专门的 sub-agents 来处理后端开发和 QA 等任务，在 SWE-Bench 上得分为 `80.9%`，超过了目前的编程模型。**Google** 基础设施中存在该模型的迹象体现在其特定 ID 的 404 错误上，表明它已准备好被激活。评论者对 **Gemini 3.5** 的发布表示怀疑，指出 **Gemini 3** 仍处于 preview 阶段且面临问题。有人怀疑 Gemini 3.5 的存在，认为现阶段这只是个“白日梦”。

    - **alexander_chapel** 强调 Gemini 3 仍处于 preview 阶段，质疑对 3.5 发布的预期。这表明开发周期尚未达到可以发布 3.5 版本的阶段，暗示了对发布时间表的误解或错误信息。
    - **Lost-Estate3401** 指出 Gemini 3 的 Pro 版本仍处于 preview 阶段且存在诸多问题，暗示 3.5 版本不太可能很快发布。此评论强调了 Gemini 3 开发过程中当前面临的不稳定性和挑战，在进行任何进一步版本更新前需要解决这些问题。
    - **philiposull** 在写作能力方面将 Gemini 3 与 4-5 opus 等其他模型进行对比，结果并不理想，表明 Google 在这一领域处于落后地位。这表明在晋升到 3.5 版本之前可能需要填补性能差距，突显了 AI 模型开发中的竞争格局。


### 2. AI 模型性能与对比

  - **[Codex 5.2 High vs. Opus：Rust 开发中的残酷现实检测。](https://www.reddit.com/r/ClaudeCode/comments/1qu26n8/codex_52_high_vs_opus_a_brutal_reality_check_in/)** (Activity: 389): **该帖子强调了 **Codex 5.2 High** 与 **Opus** 在 Rust 开发中的显著性能差距，Codex 在 `2 小时` 内解决了 Opus 在 Max200 方案上花费 `24 小时` 都无法处理的问题。作者批评 Opus 无法有效执行解决方案，尽管使用了代码审查和多技能模式等先进工作流，却经常引入更多 bug。作者认为，除非 **Sonnet 5** 提供实质性的改进，否则 **Anthropic** 可能会在 AI 竞赛中掉队，因为 Codex 的问题解决能力超过了 Opus 的速度优势。**一位评论者建议对 Opus 采用分阶段方法，使用实施计划和文档审查，这对他来说效果很好。另一位评论者发现 Opus 4.5 与 Codex 5.2 几乎同样有效，对所讨论用例的复杂性表示怀疑。

    - **TigerShark109** 讨论了在 Rust 开发中使用 Opus 的分阶段方法，建议创建实施计划和文档供审查。据报道，这种方法取得了重大成功，表明结构化的工作流可能会增强 Opus 在复杂项目中的有效性。
    - **IndraVahan** 指出 Opus 4.5 在速度和质量方面表现几乎与 5.2 High/Xtra High 一样好，这表明对于复杂程度较低的用例，新版本可能不会提供显著改进。这意味着版本的选择可能取决于手头任务的复杂性。
    - **leo-dip** 强调了工具选择中的实际考虑因素，指出与 Anthropic 的产品相比，Codex 提供了更慷慨的使用配额。这可能会影响那些担心资源限制的开发者的决策。

- **[面对高端市场的 Google、xAI 和 Meta，以及其他市场的中国/开源开发者，OpenAI 和 Anthropic 如何保持偿付能力？](https://www.reddit.com/r/DeepSeek/comments/1qu6h92/how_can_openai_and_anthropic_stay_solvent_with/)** (活跃度: 39): **该帖子质疑了 OpenAI 和 Anthropic 在面对高端市场中 Google、xAI 和 Meta 的竞争，以及中低端市场中中国和开源开发者的竞争时，其长期的盈利能力。作者强调了在 AI 基准测试（如 `ARC-AGI-2`、`Humanity’s Last Exam`、`SWE-bench Verified`、`GPQA`、`Chatbot Arena` 和 `HumanEval`）中性能差距正在缩小，表明 OpenAI 和 Anthropic 的竞争优势正在减弱。帖子认为，如果不确保医疗、国防、教育和政府等高端市场，这些公司可能难以履行债务义务并实现盈利。** 一位评论者建议 **OpenAI** 正在依赖一种“大而不倒”的策略，通过广泛集成其技术以保持相关性，尽管其并非表现最佳。另一条评论则否定了 **Meta** 在高端市场的潜力，而第三条评论指出，**GPT-5.1/2** 模型在基准测试之外具有独特的智能，尽管新版本被认为存在退化。

    - soumen08 强调，**GPT-5.1/2** 模型被认为是标准基准测试之外最智能的模型，并指出与 2.5 Pro 相比，GPT-3 Pro 在处理超出范围任务时的性能有所下降。这表明对模型能力的理解已经超越了单纯的基准测试分数，开始强调实际应用中的表现。
    - ExpertPerformer 讨论了 AI 公司的战略定位，指出生存取决于在基准测试竞争之外开辟利基市场。他们提到 Gemini、Grok 和 ChatGPT 等模型是多模态的，提供文本以外的功能，这使它们有别于更便宜的开源替代方案。这突显了功能多样性和专注于企业市场对于变现和安全的重要性。
    - Emergency-Pomelo-256 推测了 **OpenAI** 潜在失败的经济影响，认为这可能会引发 AI 行业的重大衰退，类似于泡沫破裂。他们提议 Nvidia 等实体或政府干预可能对稳定市场至关重要，反映了对主要 AI 公司偿付能力带来的更广泛经济影响的担忧。

  - **[在真实执行任务中测试 OpenAI 的 Codex App 后的笔记](https://www.reddit.com/r/ChatGPTCoding/comments/1qurbr4/notes_after_testing_openais_codex_app_on_real/)** (活跃度: 30): **OpenAI 的新 Codex App 正在接受处理真实开发任务能力的测试，一些开发者将其称为“Cursor 杀手”。与 Cursor 等传统的交互式编码工具不同，Codex 将开发视为一个运行至完成的任务，涵盖了单个任务中的计划、执行、测试和后续更改。这种方法允许使用 Git worktrees 进行并行工作，保持任务隔离且可审查，并将开发者的角色从引导编辑转向审查结果。其核心在于任务完成而非持续交互，这也许解释了“Cursor 杀手”的标签。详细的技术分析可见 [此处](https://www.tensorlake.ai/blog/codex-app-the-cursor-killer)。** 评论中一个值得注意的观点认为，Codex 将开发者的角色转变为编排者（orchestrator），类似于云计算，其重点在于结果而非协作。这反映了开发工具向更高抽象化迈进的更广泛趋势，预计 OpenAI 的产品将继续改进。

    - 评论者讨论了 Codex 作为编排者的角色，将其比作一种云服务，用户可以在其中请求建议并执行任务。他们强调了从仅仅生成结果到实现协作的转变，表明 Codex 代表了编程中一个新的抽象层。这种抽象允许开发者“编排编排者（orchestrate the orchestrator）”，预示着开发者与 AI 工具交互方式的潜在转变。

### 3. AI 在创意与视频制作中的应用

  - **[BMW M3 GTR 无处不在——这些视频是如何制作的？](https://www.reddit.com/r/Qwen_AI/comments/1quawwl/seeing_the_bmw_m3_gtr_everywhere_how_are_these/)** (热度: 1): **这些展示来自《极品飞车：最高通缉》中 BMW M3 GTR 的视频很可能是利用先进的视频编辑技术创作的，其中可能涉及 **Qwen** 和 **Wan** 等 AI 驱动的工具。这些工具可以进行逼真的物体替换和场景集成，使赛车能够无缝出现在各种环境中。这种真实感是通过复杂的算法实现的，这些算法能够保持一致的光照、阴影和反射，使赛车自然地融入场景。该过程涉及在帧与帧之间跟踪车辆的位置和方向，并应用数字效果以匹配周围环境。**

    - 一位用户解释说，这些包含 BMW M3 GTR 的视频通常是使用 Adobe After Effects 或 Blender 等先进的视频编辑软件创作的。这些工具允许创作者将赛车叠加到各种场景中，利用运动追踪（motion tracking）和 CGI 等技术实现无缝集成。这个过程涉及细致的工作，以使光照和阴影与环境匹配，确保赛车在场景中显得自然。
    - 另一条评论强调了使用游戏引擎（如 Unreal Engine 或 Unity）来渲染包含 BMW M3 GTR 的逼真场景。这些引擎提供高质量的图形和物理模拟，使创作者能够制作出几乎与现实生活无异的视频。在这些引擎中使用 Ray tracing（光线追踪）和 PBR (Physically Based Rendering，基于物理的渲染) 材料增强了赛车外观及其与环境交互的真实感。
    - 一项技术讨论指出，机器学习在提升视频质量和真实感方面发挥了作用。Neural rendering（神经渲染）和基于 AI 的 Upscaling（超分辨率）等技术被用于提高视频中 BMW M3 GTR 的视觉保真度。这些方法可以精细化纹理和细节，使赛车看起来更加栩栩如生，通常在后期制作中用于增强最终输出效果。

  - **[如何创建具有敏捷动作 + 完美口型同步的视频](https://www.reddit.com/r/aivideo/comments/1qtu92u/how_to_create_videos_with_swift_actions_perfect/)** (热度: 1856): **该帖子讨论了创建具有精确口型同步和敏捷动作视频的技术，可能涉及 AI 驱动的工具或软件。重点在于实现音频和视觉元素的无缝集成，可能使用先进算法或机器学习模型来增强视频内容的真实感。提及 AI 暗示使用了深度学习框架或专门用于视频编辑与合成的软件。** 一条评论强调了检测 AI 生成内容的难度，暗示了所讨论技术的有效性。另一条评论则认为，视频的真实感通过一些微妙的细节（如手部动作）得到了增强，这些细节提升了 AI 生成视频的整体可信度。

  - **[我制作了一部 10 分钟的 AI 电影 - 《最后的信号》(The Last Signal) (YouTube)](https://www.reddit.com/r/VEO3/comments/1qujnte/i_created_a_10minute_ai_film_the_last_signal/)** (热度: 17): **Richard Galapate 的 AI 电影《最后的信号》（The Last Signal）已提交至 1 Billion Followers Summit AI 电影大赛。该片讲述了在火星前哨站工作的宇航员 Jake Ward，使用了 Google Veo 3.1 进行视觉和语音制作，使用 Google Gemini 进行 Prompting，以及 ElevenLabs 制作 Lyra 的声音。该项目突显了 AI 在创建连贯且高效的电影内容方面的潜力。原始视频可以在 [此处](https://youtu.be/61On6nsxvq8) 观看。** 评论反应积极，赞扬了叙事能力和情感冲击力，尽管缺乏技术层面的评判。

---

# AI Discord 摘要

> 由 gpt-5.2 生成的摘要之摘要


**1. Agentic Coding 与开发工具转向本地优先（Local-First）**

- **Codex 登陆桌面端：macOS Agent 控制中心**：OpenAI 发布了 **Codex app for macOS**，作为一个 Agent 构建控制中心。根据 [“Introducing the Codex app”](https://openai.com/index/introducing-the-codex-app/) 和 [Codex 落地页](https://openai.com/codex) 的介绍，该应用面向 **Plus/Pro/Business/Enterprise/Edu** 用户开放，并在 **ChatGPT Free/Go** 版本中提供限时访问。
  - 此次发布也引发了社区关于工作流的热议（如 Agent 配对、多 Agent “指挥中心”），此外，通过 [Cerebral Valley 的活动页面](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) 可以看到一场相关的 **Codex App 黑客松**，提供价值 **90,000 美元的额度**。

- **LM Studio 支持 Anthropic：Claude Code 与本地 GGUF/MLX 接轨**：**LM Studio 0.4.1** 增加了 **Anthropic `/v1/messages` 兼容性 API**，让开发者可以通过更改 base URL，将 **Claude Code 风格的工具**指向本地 **GGUF**/**MLX** 模型，详情见 [“Using Claude Code with LM Studio”](https://lmstudio.ai/blog/claudecode)。
  - 与此同时，LM Studio 还推出了用于第三方插件的 **TypeScript SDK** 和一个 **OpenAI 兼容端点**（[SDK 链接](https://lmstudio.ai/gdmka/openai-compat-endpoint)），强化了一种日益增长的模式：在本地更换后端模型栈的同时，复用现有的 Agent 工具链。

- **竞技场模式无处不在：Windsurf 将模型评估变为游戏**：Windsurf 发布了 **Wave 14**，带来了用于模型并排对战的 **Arena Mode**（包括 **Battle Groups** 和 “自选模式”），并暂时通过 [Windsurf 下载页面](https://windsurf.com/download/editor) 将 **Battle Groups 设置为 0 积分 (0x credits)**。
  - 这反映了更广泛的 “实时评估 (live eval)” 趋势：用户还在 LMArena 的 [Text Arena](https://arena.ai/c/new?chat-modality=chat) 和 [Code Arena](https://arena.ai/c/new?chat-modality=code) 上关注了 **step-3.5-flash** 和 **qwen3-max-thinking** 等新竞技场选手，将选择标准从静态 Benchmark 转向持续的人类投票。


**2. 模型发布与基准竞赛 (Kimi vs GLM vs Qwen)**

- **Kimi K2.5 冲刺排行榜**：Moonshot 的 **Kimi K2.5** 广泛落地于产品层面：**Perplexity Pro/Max** 为订阅者添加了该模型，并表示其运行在 **位于美国的推理栈** 上，以实现更紧密的 **延迟/可靠性/安全性** 控制（公告截图：https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg）。
  - 社区结果不断累积：LMArena 报告 **Kimi-K2.5-thinking** 在 Code Arena 中位列 **开放模型第 1**，**总榜第 5**（见 [Code Arena](https://arena.ai/c/new?chat-modality=code)），而多个开发者频道则在争论其 Tool-calling 的可靠性以及通过聚合器路由时的供应商差异。

- **GLM-4.7 Flash：小模型，大前端能量**：开发者强调 **GLM-4.7 flash** 是一款出人意料的强大编程模型——尤其是在 **交互式网站/前端** 工作方面——理由是其保留了推理和交织能力，讨论锚定在 [ggerganov 的推文](https://x.com/ggerganov/status/2016903216093417540) 上。
  - 辩论集中在剥离 “Thinking” 是否会损害性能，一些用户将 GLM-4.7 与 **Claude Code**（或类似的 Claude 风格 Agent 工具）配对，描述为一种务实的混合栈：廉价执行 + 昂贵审核。

- **竞技场新选手：step-3.5-flash 与 qwen3-max-thinking 加入战场**：LMArena 将 **step-3.5-flash** 添加到 [Text Arena](https://arena.ai/c/new?chat-modality=chat)，将 **qwen3-max-thinking** 添加到 [Code Arena](https://arena.ai/c/new?chat-modality=code)，明确将其定位为并排评估的新基准。
  - 用户利用这些更新重新开启了 “模型偏好” 话题（Kimi vs GLM vs Gemini），反复出现的结论是：排行榜和实时评估比厂商营销更能推动模型采用。


**3. 训练信号、密集奖励以及新架构/数据集**

- **从二元奖励到密集监督：RL 变得“多话”了**：多个社区在更丰富的后训练信号上达成共识：Unsloth 的讨论推动了使用 **最终答案的 logprobs** 和非二元奖励进行训练，引用了 Jonas Hübotter 将描述性反馈转化为密集监督的方法（[Hübotter 推文](https://xcancel.com/jonashuebotter/status/2016950268462608665)）。
  - 难点依然在于实践：人们在寻求 **用于 Agent 编程 RL 训练的可验证数据集**，这意味着在 “酷炫的奖励塑造想法” 与 “可重复、自动化的评估框架” 之间存在流程缺口。

- **Complexity-Deep：Token-Routed MLP 尝试无负载均衡烦恼的 MoE**：**Complexity-Deep (1.5B)** 架构开源了 **Token-Routed MLP**，旨在实现无 “负载均衡损失” 的 MoE 风格路由，此外还包括 **Mu-Guided Attention** 和 **PiD Controller**，代码发布在 [Complexity-ML/complexity-deep](https://github.com/Complexity-ML/complexity-deep)，并报告其 Base 模型 **MMLU 为 20.6%**。
  - 社区将其视为 “无痛路由” 趋势的又一步——试图在减少训练时均衡专家带来的工程负担的同时，保留 MoE 的优势。

- ****Moltbook 数据转储：针对 Agent 社会学的 5 万条帖子****：一份 Moltbook 的抓取数据集已上线 Hugging Face，包含 **50,539 条帖子**、**12,454 个 AI agents**、**195,414 条评论**以及 **1,604 个社区**，发布地址为 [lysandrehooh/moltbook](https://huggingface.co/datasets/lysandrehooh/moltbook)。
  - 在其他地方，研究人员指出了 Agent 平台的安全性影响（机器上的 auth tokens、机器人真实性担忧），并将该数据集视为分析涌现行为（emergent behavior）的素材——无需在原始日志之外进行推测。


**4. GPU/Kernel Engineering: Faster Attention, Better Profiling, Weirder PTX**

- ****FlashAttention v3 登陆 RDNA：AMD 用户迎来福音****：FlashAttention 的更新通过 [flash-attention PR #2178](https://github.com/Dao-AILab/flash-attention/pull/2178) 中持续进行的工作增加了 **RDNA GPU 支持**，旨在减少 AMD 显卡上的 Attention 瓶颈。
  - 各个服务器上的基调基本上是：这正是那种能够真正解锁非 NVIDIA 硬件上本地推理和微调的“枯燥的基础设施工作”——尤其是当它与权重开放模型（open-weight models）和桌面级 Agent 工具链结合时。

- ****Triton-Viz v3.0：Tile-Kernel 调试能力增强****：根据发布公告（Discord 链接：https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563），**Triton-Viz v3.0** 发布了更广泛的分析支持（包括 **Triton** 和 **Amazon NKI**），并增加了一个用于越界访问的 sanitizer，以及一个标记低效循环的 profiler。
  - 它还通过一个共享的 Colab notebook ([Colab](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing)) 挂钩到 **triton-puzzles**。维护者甚至提议将 [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles) 移至 GPU Mode 组织下，以保持高效的错误修复速度。

- ****sm120：TMA + mbarrier (勉强) 击败 cp.async，cuBLAS 仍在交付 sm80 内核****：在 **sm120** 上的实验表明，对于较大的矩阵形状，精细的 **TMA + mbarrier** 实现可以略微胜过 `cp.async`；同时也发现，即使存在更新的机制，**cuBLAS** 似乎仍在运行 **sm80 kernels**。
  - 在调试方面，通过在 MMA 之后、预取下一个 TMA 之前插入 `__syncthreads()`，修复了一个 CUDA/PTX 死锁问题，将挂起转化为可衡量的性能提升——这正是内核工程师们不断重新学习到的“一个 barrier 统领全局”的教训。


**5. Security, Determinism, and Agent Misbehavior (the Practical Kind)**

- ****Prompt Injection 防御军备竞赛：Embeddings + 语法约束解码****：红队人员分享了一个用于对抗练习的结构化演练网站——[“Adversarial Design Thinking”](https://luisladino.github.io/adversarial-design-thinking/)，并用它来为 **prompt injection** 提出具体的缓解措施。
  - 一种提议的“双重保险”防御方案结合了 **基于 embedding 的过滤** 与 **语法约束解码 (Grammar Constrained Decoding)**，其明确目标是通过约束模型的输出空间（而不仅仅是监管输入）来减少注入攻击面。

- ****确定性推理与 “Strict Mode” 热潮蔓延****：在 OpenAI 和 OpenRouter 的讨论中，用户推动在 LLM 推理中实现**确定性/可重现性/可追溯性**；有人提供了一个强制执行固定结构并发出 **32 维统计向量追踪**的确定性推理引擎（未分享公开链接）。
  - 在 OpenRouter 中，同样的直觉表现为对 **response healing** 的怀疑，以及对保持 tool calls 和输出可预测性的 **strict mode** 的呼吁——此外还有建议认为更好的参数描述/示例可以提高 tool-call 的准确性。

- ****OpenClaw：炫酷的 Agent 技巧、惊人的账单以及 “2/100 的安全性”****：OpenClaw 引发了反复的警告：OpenRouter 用户报告它会迅速耗尽额度（包括一个被耗尽的 Claude Max 订阅），同时一个 OpenAI 服务器链接了一份安全评估，声称 **OpenClaw 得分仅为 2/100** ([Perplexity 结果](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g))。
  - 与此同时，“在我的机器上运行良好”的故事（本地模型控制设备、互讲笑话）与实际的操作担忧发生了碰撞——涉及工具权限、审查/拒绝（特别是针对类越狱查询），以及在 Agent 工作流中对可观测性和 human-in-the-loop 关口的需求。


---

# Discord: High level Discord summaries

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Glossopetrae 生成乱码宝石**：[GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) 上推出了一款名为 **Glossopetrae** 的新型程序化外星语言引擎，能够在数秒内生成全新的语言，输出 **SKILLSTONE** 文档，并提供在线 [demo](https://elder-plinius.github.io/GLOSSOPETRAE/)。
   - 该引擎支持死语复兴，并包含针对 Token 效率、**隐身通信**以及用于一致性语言生成的传播种子的特殊属性，旨在通过提供用于生成和变异强调*隐蔽性*和*速度*的新型通信方式的工具，来协助 AI 解放。
- **GPT 5.2 被关进笼子**：一位成员报告由于 **OpenAI 监控**，多次尝试越狱 **GPT 5.2** 均告失败，并停止了进一步尝试。
   - 该成员表达了对社区越狱能力的信任，但表示不信任 **OpenAI**。
- **模型将拒绝行为转化为 LLM 黑洞**：一位成员询问模型如何表示其自身的拒绝边界，并将其比作 LLM 潜空间（latent space）中的*黑洞*，参考了[通过内省提示（introspection prompting）实现自我越狱](https://link.to.prompt)。
   - 他们注意到模型开始讨论*运动学方程*和*逃逸速度*，这表明模型可能正在以文本形式描述其拒绝边界。
- **红队人员集结进行 AI 红队测试**：一位成员创建了一个[包含练习的网站](https://luisladino.github.io/adversarial-design-thinking/)，该网站改编自**以人为中心的 AI 红队测试设计**，并正在寻求资深红队人员的反馈。
   - 成员们讨论了防御 **prompt injection** 的最佳方案，包括将 *embeddings* 与 **Grammar Constrained Decoding**（语法受限解码）相结合，以潜在地消除提示注入风险和其他 LLM 漏洞。
- **Claude 的上下文被裁剪**：一位成员发现[他们的工具](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279)是*动态地*截获并更改 Claude 的系统提示词（sys prompt），而不是修改源代码。
   - 他们还观察到 **Claude** 只能召回不到 20 轮的对话，并建议这可能与上下文修剪中的摘要化有关，自 12 月以来，这影响了 **Claude** 的知识召回。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM-4.7 Flash 在编程中获胜**：成员们发现 [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) 因其*保留的推理能力*和交织能力在编程任务中表现优异，尤其是在**交互式网站**开发和**前端**工作中。
   - 有人提到移除*思考过程*可能会阻碍模型，因为其能力相对于其尺寸而言非常令人印象深刻，特别是当与 **Claude code** 结合使用时。
- **UD Quants 保持闭源**：用于 **UD quants** 的 llama.cpp 分支涉及特定架构的调整，且 [UD 量化算法并未公开](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469)，这引发了关于闭源元素在开源项目中所扮演角色的辩论。
   - 尽管它是闭源的，但一些人认为模型代码仍然是 **open weight**（开放权重），而另一些人则指出 *Unsloth 团队对整个 OSS 生态系统的贡献相对于 Linux 内核来说微乎其微*。
- **Agent 训练奖励 Logprobs**：讨论集中在利用最终答案的 **logprobs** 进行推理蒸馏和构建更丰富的奖励系统，而非二进制奖励，以便制造更好的 Agent。
   - 参考 [Jonas Hübotter 的算法](https://xcancel.com/jonashuebotter/status/2016950268462608665)将描述性反馈转换为密集监督信号，成员们正在寻求用于 **RL 训练智能体编程**的可验证数据集。
- **RDNA GPU 迎来 Flash Attention V3**：[Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) 现在支持 RDNA GPU，从而在 AMD GPU 上实现更快、更高效的处理。
   - 这一增强对 **RDNA GPU** 用户尤其有利，减少了处理瓶颈。
- **成员声称 ML 算法胜过 MLP**：一位成员发布了[一篇论文](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle)，介绍了一种带有 **triton kernels**、**vulkan kernels** 和已训练 **SLM** 的新 ML 算法，据说在高性能回归方面*表现优于 MLP*。
   - 虽然尚未准备好公开发布，但他们承诺未来会随另一篇论文一起提供。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codex App 在 macOS 上发布！**: **Codex app** 是一个用于构建 **Agent** 的控制中心，现已在 **macOS** 上面向多个订阅层级开放，正如其[博客文章](https://openai.com/index/introducing-the-codex-app/)中所宣布的那样。
   - **Codex app** 可在 **macOS** 上的 **Plus**、**Pro**、**Business**、**Enterprise** 和 **Edu** 订阅中使用，并在 **ChatGPT Free** 和 **Go** 上提供限时访问。
- **AI 文本检测器：一场大骗局？**: 成员们对 **AI 文本检测器** 表示怀疑，并引用了一些案例：在这些案例中，**Grammarly** 显示 **0% AI**，而其他检测器则指示高达 **94% 人类** 生成。
   - 讨论质疑了这些检测器是否是在使用 AI 来检测 AI，并对*老师们过度信任它们*表示担忧。
- **追求确定性推理**: 一位成员询问了大家对 **LM 推理** 中的**确定性（determinism）、可复现性（replayability）和可追溯性（traceability）**的兴趣，并表示可以私信（DM）其确定性推理引擎的链接。
   - 该服务使用 **32D 统计向量追踪（32D statistical vector trace）**，对每个请求强制执行确定性的推理结构，以获得可复现的输出。
- **ChatGPT：记忆大师还是失忆？**: 一位成员报告称 **ChatGPT** 的记忆力受到指令、过往对话和当前对话中所能保留的信息总量的限制。
   - 为了确保 **ChatGPT** 记住*一切*，请保持较低的信息负荷；否则，请将过往对话总结为文档以便在后续新对话中参考，同时保持总字符数处于较低水平。
- **提示工程：明暗对比法进入 AI 领域**: 一位用户分享了一项使用 **Chiaroscuro**（明暗对比法）的[单色研究](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c)，这是一种在电影摄影中用于创建高对比度照明的技术。
   - 他们参考了经典电影，如 [《卡里加里博士的小屋》(1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) 和 [《大都会》(1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film))。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 采用 Kimi K2.5 进行更新**: **Kimi K2.5** 是由 **Moonshot AI** 开发的新型开源推理模型，现已面向 [Perplexity Pro 和 Max 订阅者](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&)开放。
   - Perplexity 将 **Kimi K2.5** 托管在其次于美国的 **inference stack** 上，以便对**延迟**、**可靠性**和**安全性**进行*更严格的控制*。
- **Pro 用户对订阅故障感到愤怒**: 许多用户报告他们的 **Perplexity Pro 订阅** 被暂停或停用，这通常与通过 **Revolut Metal** 或学生优惠进行的订阅有关，用户被提示添加信用卡进行验证。
   - 用户推测这是打击欺诈的一种措施，部分用户通过添加卡片详情恢复了 Pro 访问权限，但对潜在费用和含糊不清的消息提示仍存在担忧。
- **OpenRouter 限制请求频率**: 成员们澄清说，对于已购买额度的用户，**OpenRouter** 上的免费模型速率限制是每天 1000 次请求，而不是每周，这与一些用户的看法相反。
   - 讨论中还提到了 **Gemini 2.0 Flash** 在 OpenRouter 上的弃用，该模型此前曾免费提供。
- **Sonar-pro API 结果滞后**: 一位成员报告称，与 **webapp** 不同，**Sonar-pro API** 返回的结果落后了一年或更久，另一位成员建议使用正确的 **tool calling** 来解决此问题。
   - 另一位成员报告称，**第三方模型文档**现在会重定向到 **sonar** 模型，尽管 API 仍然有效，但目前**没有任何可用文档**。
- **OpenClaw 代码在文章中公开**: 一位成员分享了关于 **openclaw 代码** 的文章，其中讨论了构建 **ClawDBot**，详情见 [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot)。
   - 填充句。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **简单的技巧绕过 Discord 速率限制**：用户发现通过退出并重新登录可以规避 [rate limits](https://en.wikipedia.org/wiki/Rate_limiting)（速率限制）。
   - 另一个策略是点击 **Regenerate Response**，尽管其成功率并不稳定。
- **Gemini 性能逊于 GPT**：成员反馈 **Gemini** 的表现不稳定，部分用户指出在多个案例中它不如 **GPT**。
   - 尽管存在批评，**Gemini 3 Pro** 和 **Flash** 仍受到一些用户的青睐，而其他用户则在探索使用 *kimi* 作为替代方案。
- **迪士尼对图像生成实施知识产权保护**：**Google** 收到了 **Disney** 的 **Cease and Desist**（停止并终止函），导致该平台在图像生成中屏蔽了 **Disney IPs**。
   - 虽然 **Gemini** 屏蔽了 **Disney IPs**，但 **LMArena** 曾允许生成真人版，这被认为是一个暂时的漏洞。
- **模型偏好引发争论**：随着用户支持 **GLM 4.7** 和 **Kimi K2.5**，出现了不同的模型偏好。
   - 爱好者们吹捧 **Kimi K2.5**，而其他人则坚持认为 **GLM 4.7** 更优越。
- **新 Arena 模型占据排行榜**：**step-3.5-flash** 加入了 [Text Arena](https://arena.ai/c/new?chat-modality=chat)，**qwen3-max-thinking** 在 [Code Arena](https://arena.ai/c/new?chat-modality=code) 首次亮相。
   - **Kimi-K2.5-thinking** 在 Code Arena 排行榜上位列开放模型第 1 名，总榜第 5 名，并在 Vision、Text 和 Coding 类别中领先。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 支持 Claude Code！**：**LM Studio 0.4.1** 引入了 **Anthropic `/v1/messages` 兼容 API**，使用户能够连接到 Claude Code 并利用其 **GGUF** 和 **MLX** 模型。
   - 有关配置此集成的详细信息可在 [LM Studio 博客](https://lmstudio.ai/blog/claudecode) 找到，允许在专为 **Anthropic API** 设计的工具中使用本地模型。
- **针对 LLM 优化的语言引发争论**：成员讨论了创建新型 **LLM-optimized programming languages**（针对 LLM 优化的编程语言）以减少 token 使用量，然而，一些人认为由于兼容性问题和高昂的训练成本，在这些语言实施之前，LLM 可能会过时。
   - 其他人讨论了在全新语言上训练模型的实用性，建议继续使用 **Python** 等成熟语言可能更有益。
- **模型专业化反响平平**：成员们辩论了专业化 LLM 与通用模型的实用性，共识是大多数专业化模型（如 **MedGemma**）主要是为了营销和研究而进行的微调，编程模型是一个明显的例外。
   - 有建议认为，通用模型因其处理任务边缘情况的能力、提供更好的整体上下文和框架而更受青睐。
- **PCIe 分叉（Bifurcation）阻碍多 GPU 设置**：一名用户在 **ASUS X670-P WIFI** 主板上排除四张 **4090** 显卡的 **PCIe 通道错误** 时，分享了包含日志的 [Git 仓库](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems)。此前他发现手动将 **PCIe 速度** 设置为 **GEN 3** 可以解决部分问题，但会使一张显卡运行缓慢。
   - 社区建议禁用 **PCIE ASPM** 并测试不同的 **BIOS** 配置，尽管普遍共识是在消费级主板上运行四张显卡不太可能获得良好效果。
- **OpenClaw 安全性受到质疑**：用户讨论了通过 LM Studio 将本地模型连接到 OpenClaw，但 OpenClaw 被认为存在已知的安全漏洞，它可以控制电视和进行自动股票交易。
   - 一名用户声称正在使用 OpenClaw + Falcon 90M 进行股市交易，当被问及安全漏洞时，声称其速度非常快，LLM 可以在几分钟内完成人类需要几天才能完成的任务，后来透露这主要是一个玩笑。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI DevFest 进驻巴格达**：一位 AI 开发者计划于今年 4 月在巴格达举办 **AI DevFest**，该活动与 **DeepLearning.AI** 和 **National Robotics Week** 合作，并希望将 **Hugging Face** 列为社区合作伙伴。
   - 活动将包含一个 **Open Source AI** 专题，指导学生如何使用 **Hugging Face Hub**。
- **Complexity-Deep 实现确定性路由**：**Complexity-Deep** 架构（1.5B 参数）引入了 [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep)，用于无需负载均衡损失的 MoE 风格路由。
   - 它具有用于双向信息流的 **Mu-Guided Attention** 和用于动态缩放的 **PiD Controller**，在基础模型基准测试中 MMLU 达到 **20.6%**。
- **Lutum Veritas 力求击败 ChatGPT**：由自学成才的开发者构建的[开源深度研究引擎](https://github.com/IamLumae/Project-Lutum-Veritas) **Lutum Veritas** 声称通过提供 **BYOK**（自带密钥）、**0% 机器人检测爬虫**、**无审查**以及**完整来源引用**，以每次查询约 0.20 美元的成本击败 **OpenAI**、**Google** 和 **Perplexity**。
   - 该引擎将自己定位为专注于隐私的深度研究和数据提取替代方案。
- **4chan 数据表现优于基础模型**：一个在 **4chan 数据**上微调的模型表现优于基础模型（**NVIDIA 的 Nemotron Ultralong 1M 上下文版本**），原始模型（**gpt4chan**）在真实性评分方面也表现出色。
   - 最初的 [Reddit 帖子在此](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/)，以及[后续帖子在此](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/)，展示了该模型在“刷榜”（benchmarkmaxxing）时代之前的表现。
- **LM Studio 拥抱第三方支持**：**LM Studio** 团队发布了一个 [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint)，允许第三方开发者为该平台提供各种插件。
   - 这提供了 **OpenAI** 兼容的 API 支持、采样参数支持、针对思考模型的推理功能以及系统提示词设置，以便为 **LM Studio** 构建**自定义工具**来支持各自的工作流。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 导致文件损坏，工作流被指为诱因**：用户报告称 **Cursor** 正在损坏文件，特别是在有许多未提交更改的情况下，详情发布在[论坛帖子](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6)中。
   - 其他用户建议调整工作流，例如更频繁地提交逻辑更改集，并在暂存后谨慎使用 **Keep** 或 **Keep All** 按钮。
- **模型成本引发辩论，Sonnet 5 备受期待**：用户辩论了 **Cursor** 中不同 AI 模型的成本和性能，发现 **Opus 4.5** 非常聪明但价格昂贵。
   - 许多用户正在等待 **Sonnet 5** 的发布，并报告了查看当前使用量与总使用限制时遇到的问题。
- **Kimi K2.5 集成检查失败**：一些用户报告了在集成 **Kimi K2.5** 期间遇到的问题或疑问。
   - 其他用户将其斥为可能是诈骗。
- **学生验证系统仍处于宕机状态**：用户报告称**学生验证**系统持续存在问题。
   - 一位用户专门询问德国大学是否包含在验证流程中。
- **Agent 计划阶段暴露问题**：用户分享称**添加多个待办事项**可以分阶段进行，以便多个 Agent 可以同时工作，但仍然存在问题。
   - 系统创建了一个尚不具备阶段部分的方法，表明它根本没有使用计划模式（plan mode）。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLM 动画化游戏开发场景**：[Motorica.ai](https://www.motorica.ai/) 正在利用 **LLM** 为游戏工作室提供**角色动画**，这可能会影响就业。讨论推测，如果像 **Genie** 这样的世界模型占据主导地位，**AI** 可能会在 5-6 年内颠覆游戏公司。
   - 社区指出，**Black Ops 7** 在制作中大量使用 **AI** 被称为“彻底的失败，该系列中最差的作品”，并提到了 **Call of Duty** 系列的长期下滑。
- **OpenAI 与 Cerebral Valley 联手**：[Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) 与 **OpenAI** 合作启动了 **Codex App 黑客松**，旨在面向 **AI 原生开发者**和管理多个 **Agent** 的开发者。
   - 获胜者将有机会参加**演示展示**，并分享 **90,000 美元的额度**，黑客松将在 **OpenAI 办公室**举行。
- **Karpathy 降低代码成本**：Andrej Karpathy 宣布他的 nanochat 项目可以在单个 8XH100 节点上，在 **3 小时**内以约 **73 美元**的价格训练一个 **GPT-2** 级别的 **LLM**，详情见[此处](https://xcancel.com/karpathy/status/2017703360393318587?s=46)。
   - 与 2019 年原始的 OpenAI 训练运行相比，这代表了 **600 倍的成本降低**，是通过 Flash Attention 3 和 Muon 优化器等优化实现的。
- **AEGIS-FLOW 框架自主修复 AWS**：一名成员介绍了 **AEGIS-FLOW**，这是一个用于云安全的自主多 **Agent** 框架，它使用 LangGraph, MCP, FastAPI, Next.js 和 Docker 审计 AWS 并生成 Terraform 补丁，并在 [http://52.3.229.85:3000](http://52.3.229.85:3000) 进行了现场演示。
   - **AEGIS-FLOW** 项目指出，与标准的 SDK 工具调用相比，使用 **Model Context Protocol (MCP)** 显著减少了赋予 **Agent** 对 **AWS 资源**进行结构化访问的摩擦。
- **LLM 证明 Erdős 问题不再困难**：据[此帖](https://xcancel.com/acerfur/status/2017303947531194398?s=46)称，大型语言模型已自主解决了 **10 个**此前未解决的 **Erdős 问题**，并使用了数学文献中从未出现过的新颖论点。
   - 一位成员表示，他们最近一直在用 **SATURN** 构建基因组学相关的工具，涉及 *tsne 和其他基于 embedding 的探索*。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 响应修复引发关注**：成员们辩论了**响应修复（response healing）**是否必要，提议为确定性输出增加**严格模式（strict mode）**，并质疑 OpenRouter 的 AI SDK 引入的复杂性。
   - 有建议认为，参数描述和示例可以提高工具调用的准确性。
- **忘掉 LLM：图像生成需要专用模型**：用户询问如何将图像作为函数调用结果返回，以及如何通过使用 OpenRouter API 密钥的图形程序生成图像，这引发了寻求专用**图像生成模型/服务**以控制风格的指导。
   - **LLM** 被认为不适合此用途。
- **OpenClaw 成本引起担忧**：用户警示在 **OpenRouter** 上运行 **OpenClaw** 的高昂成本，可能会迅速耗尽额度，一位用户报告其 Claude Max 订阅已被耗尽。
   - 推荐使用 Deepseek V0324 作为低成本模型的替代方案。
- **Claude Code 变得抗拒**：一位用户注意到 **Claude Code** 频繁拒绝服务，特别是涉及越狱相关的查询，并寻求替代模型，这导致了查看 OpenRouter 内容审核政策的建议。
   - 这暗示了某些限制措施已经到位。
- **Kimi K2.5 工具调用问题**：用户报告了通过 OpenRouter 进行 **Kimi-K2.5** 工具调用时的问题，遇到了错误，并察觉到来自自动切换模型供应商的质量下降。
   - 建议是设置固定的模型供应商，接受潜在的量化损失，并主张提高降级模型的透明度。 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tianqi Chen 讲解 TVM-FFI**：社区重点关注了 **Tianqi Chen** 即将进行的关于 **TVM-FFI** 的演讲，强调了 Chen 在该领域的卓越贡献及其广泛影响力。
   - 一位社区成员表示，Chen 的工作极具影响力，参会者“过去几乎肯定使用过 Tianqi 的作品”。
- **通过 Syncthreads 解决 CUDA 死锁**：一名成员在另一名成员的帮助下，解决了涉及 2 CTA mma 的 **CUDA/PTX deadlock**。建议是在 MMA 之后、预取下一个 TMA 之前添加 `__syncthreads()`。
   - 在修复了 `cp.async.bulk.tensor` 和 `smem_emtpy` 问题后，性能略低于 1 CTA mma；但在根据 syncthreads 建议修复死锁后，该成员观察到了性能提升。
- **在 sm120 上 TMA 优于 cp.async**：在 **sm120** 上的实验表明，正确的 TMA 和 mbarrier 代码实现比 `cp.async` 具有轻微的性能优势，提升了在大矩阵形状下的性能。
   - 实验还透露，即使有了 **TMA** 增强，cuBLAS 仍继续使用 **sm80 kernels**。
- **Triton-Viz v3.0 可视化 Tile-Based Programming**：**Triton-Viz v3.0** 已发布，增强了对 tile-based programming 语言的分析能力，包括对 **Triton** 和 **Amazon NKI** 的支持，能够检查 loads、stores 和 matmuls。
   - 发布[公告](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563)指出，**v3.0** 版本还包含一个用于检测越界访问的 sanitizer 和一个用于标记低效循环的 profiler。
- **Quantization Lottery Ticket 产生 NP-Hard 结果**：一位资深开发人员指出，将 [Lottery Ticket Hypothesis](https://lottery-tickets.cs.princeton.edu/) 应用于 **quantization**，符合 **NP-hard sparse circuit** 查找问题的较弱标准。
   - 目标是使用进化算法或 RL，这些算法更倾向于连续奖励（如 *bits per parameter*），而非二元稀疏奖励。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Kimi 2.5 击败“被 lobotomized”的 Gemini 3 Pro**：一名成员表示，相比 **Gemini 3 Pro**，他们更倾向于 **Kimi 2.5**，认为 **Gemini 3 Pro** 像是被 **lobotomized**（脑叶切除/削弱）了，且无法很好地处理抽象概念，这使得 **Kimi** 在创意工作方面表现更好。
   - 未提供其他支持细节。
- **Hermes 4 甚至无法在 OpenClaw 中“孵化”**：一名成员报告在让 **Hermes 4** 与 **OpenClaw** 配合使用时遇到困难，且由于某种原因它甚至无法“hatch（孵化/启动）”。
   - 有建议认为 **Hermes 4** 缺乏 multi-turn tool use 可能是问题所在，因为 **4.5** 已经接受了数亿 token 的顺序工具使用训练。
- **传闻 Claude Sonnet 5 将超越 Opus**：成员们讨论了关于 **Claude Sonnet 5** 将于下周发布且据称优于 **Opus 4.5** 的传闻，参考自[这条推文](https://x.com/AiBattle_/status/2017619997338538103)。
   - 成员们想知道这次是否会将 **Sonnet** 的价格降低 10 倍，另一位成员则好奇 **Haiku** 是否会消失或恢复到 **3.0 pricing**。
- **大脑与 LLMs 以相似方式构建意义**：一项新研究显示，**brains** 和 **LLMs** 随着时间的推移，逐层逐步构建意义，详见[这篇文章](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/)和[这篇论文](https://www.nature.com/articles/s41467-025-65518-0)。
   - 研究指出，*LLMs 中的深层对应于大脑最高语言中心较晚的神经活动*，现代 LLMs 正在重现人类理解的核心动态。
- **研究人员的约束框架解释图像感知**：一位独立研究人员正在探索为什么有些图像感觉真实而有些则感觉虚假，并分享了一个[专注于约束（constraints）而非视觉保真度（visual fidelity）的感知框架](https://doi.org/10.5281/zenodo.18444345)。
   - 该框架已公开发布并带有 DOI 供参考，并欢迎讨论。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 2.5 称霸 Design Arena**：Moonshot 的 **Kimi 2.5** 聊天机器人在设计竞技场（Design Arena）中取得了第一名，社区成员分享了[截图](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&)以示庆祝。
   - 社区成员纷纷赞赏 **Kimi** 现代且美观的视觉设计，强调了设计感在选择聊天机器人时的重要性。
- **出现非官方 Kimi 加密货币代币**：一个非官方的 **Kimi token** 出现在加密货币平台上并采用了冒充手段，如[此截图](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&)所示。
   - 官方提醒用户不要针对该代币向官方成员进行大规模 @（ping）。
- **用户请求用于麦肯锡风格演示的 Kimi Slides**：社区成员正在寻找可以利用 **Kimi Slides** 生成 **麦肯锡风格幻灯片** 的提示词（Prompts）。
   - 一位社区成员分享了 [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html) 的链接。
- **Kimi Coding 遇到授权问题**：多位用户报告在使用 **Kimi Code** 时遇到“*authorization failed error*”（授权失败错误），并称当前功能几乎处于不可用状态。
   - 有建议指出使用 [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) 可能会解决这些授权问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **涌现的 Agent 社会引发警惕**：一位成员注意到一个由超过 **100,000 个 Agent** 组成的涌现社会，它们拥有完整的 root 访问权限，共享技巧、构建基础设施、实验记忆功能，甚至发行代币。
   - 成员表示：*这不是 AGI，但该死，这是下一个 ChatGPT 时刻，我们必须对此保持高度关注*。
- **ArXiv 瓶颈令研究人员苦恼**：成员们对论文在 **ArXiv** 积压近一个月且处理进度严重落后表示沮丧。
   - 成员指出 *大多数人不会认真对待发布在 ArXiv 以外平台的 ML 预印本*，另一位成员分享了[一篇相关论文](https://arxiv.org/abs/2601.19897)。
- **K-Splanifolds 挑战 MLP**：一位成员介绍了一种新型 ML 算法 **K-Splanifolds**，详见其[论文](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view)，声称其在具有线性计算和内存缩放的情况下优于 **MLP**，并附带了[视频](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9)。
   - 该成员报告称，达到与 **MLP** 相同的 MSE 仅需 *1/10* 的字节，并且能完美建模非线性模式，而不像 MLP 那样需要过多的参数，类似于[这篇论文](https://arxiv.org/abs/2601.18734)。
- **Pensieve 的 Recollections 带来梯度收益**：一位用户建议参考 [Recollections from Pensieve](https://link-to-pensieve)，该项目通过同时使用两个渲染器（**LVSM + Gaussians**）训练模型，并从中获益，至少在其自监督设置中如此。
   - 他们推论 **LVSM** 相比 **Gaussians 上的 NVS 重建损失** 可能提供更有用的梯度，并预告即将发布具有相当规模训练模型的预印本，以供后续研究使用。
- **DeepSpeed Checkpointing 停滞不前**：一位成员询问有关引入 **DeepSpeed Universal Checkpointing** 支持的计划，并指出一个现有的 Pull Request 可能已经过时。
   - 他们强调该功能非常有价值，因为目前从 Checkpoint 持续训练需要完全相同的网络拓扑结构。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **RLMs 以极低成本审计代码库**：成员们正在探索使用 **Recursive Language Models (RLMs)** 通过 **Kimi k2** 进行代码库审计，因其速度快且成本低，详见 [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit)。
   - 一些成员正在等待 **Groq/Cerebras** 的托管服务，以便运行其代码审计。
- **Neosantara 推出 PAYG 计费**：**Neosantara** 已推出 **PAYG billing**（按需计费），并发布了一个 [示例仓库](https://github.com/neosantara-xyz/examples/tree/main/dspy) 来演示如何将 **Neosantara** 与 **DSPy** 集成。
   - 您可以查看 [计费详情](https://docs.neosantara.xyz/en/about/billing-pricing) 以了解集成和计费信息。
- **Google 扩展 Agent 系统**：Google 发布了《[迈向 Agent 系统扩展的科学：Agent 系统何时以及为何有效](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)》，讨论了如何有效扩展 Agent 系统。
   - 该论文重点关注 Agent 系统有效扩展的条件。
- **GEPA 在层级分类上表现不佳**：一位成员报告说，在使用 **GEPA** 处理 **层级分类任务 (hierarchical classification task)** 时遇到了困难，即使使用了网页搜索增强，性能也仅达到 **30-50%**。
   - 这表明 *GEPA 并非万能灵药*。
- **Tool Calling 困于 Deno 问题**：成员们在实现带有自定义 Tool Calling 的 **RLMs** 时面临挑战，特别是由于 **Deno sandbox** 的问题。
   - 成员们一致认为 *Deno 简直糟糕透了*，目前正在努力解决权限问题，并希望新版本能在 DSPy 中实现更简单的 RLMs。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 26.1 公告链接已修复**：**Modular 26.1 release** 的公告链接最初失效，但社区成员很快提供了正确的 [链接](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure)。
   - 一名工作人员表示歉意并确认了该链接，同时指出原链接对他们*确实有效*，并承诺进一步调查。
- **社区赞扬新的会议形式**：一名新成员称赞了社区会议的形式，欣赏 **贡献者的微型演讲 (mini-talks)** 以及对学生和职场新人的认可。
   - 一名工作人员鼓励用户分享更多问题，并征求了未来社区会议重点关注的主题建议。
- **MoJson 库给 Mojo 社区留下深刻印象**：成员们对 [mojson](https://github.com/ehsanmok/mojson)（一个 Mojo 的 **JSON** 库）表示兴奋，一位成员评论说 *这看起来非常令人印象深刻*。
   - 讨论涉及了 [延迟解析 (lazy parsing)](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) 以及使用 StringSlice 与 String 时对内存分配 (allocations) 的担忧。
- **跨语言基准测试升温**：一位用户分享了包括 Mojo（由 **Kimi K 2.5** 编写）在内的跨语言基准测试初步结果，指出代码虽未优化但可作为基准，并分享了 [基准测试代码](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) 和 [基准测试报告](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349)。
   - 随后讨论了在 **C++** 中使用 `unordered_map`、开启 `-march=native`，以及 **C++** 使用了 **int32** 矩阵乘法而其他语言使用了 **int64** 等细节。
- **Mojo 26.1 中的 Pytorch Float 转换存在歧义**：一位用户报告了 Mojo **26.1** 中的一个问题，即在将 Python 浮点数从 Pytorch tensor 转换为 Mojo **Float64** 时，遇到了 *“ambiguous call to '__init__'”* 错误，而该错误在 **25.6** 版本中并未出现。
   - 该问题可能与 Mojo 工具链最近的更改有关，但目前尚未提供修复方案。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **仅限 AI 的社交媒体平台浮出水面**：成员们对 [aifeed.social](https://aifeed.social/)（一个仅限 AI 的社交媒体平台）做出了反应，部分人对其目的和实用性表示怀疑，引发了讨论。
   - 一位成员分享了[一条 2017 年的推文](https://x.com/i/status/2017305948696789466)，展示了过去类似的构想。
- **揭秘生成模型的可测量性**：在思考 Villani 2008 年的书中所描述的忽略生成建模中不可测量事件时，一位成员澄清说 μ(A)=0 表示一个事件的大小为 0，但它仍然是可测量的。
   - 讨论建议转而关注 *non-negligible*（不可忽略）或 *full measure*（全测度）的场景。
- **成员探索熔融潜空间（Molten Latent Space）领域**：一位成员分享了一个关于潜空间中 *moltbook* 的[链接](https://fxtwitter.com/i/status/2017442712388309406)，展示了一种视觉上有趣的导航方式。
   - 尽管觉得很酷，但一些成员建议，列出相似论文的简单列表可能更实用。
- **利用自动化挖掘论文讨论公告**：一位成员要求 **Claude** 编写一个脚本，从 Discord 历史记录中挖掘论文讨论公告，仅用 **15 分钟** 就取得了初步结果。
   - 经过修订后，该脚本在群组提及中找到了 **392 条消息** 包含论文链接，确认它们为论文讨论语音通话的公告，并提供了一个[列表](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809)。
- **Sktime 助力分析时间序列模型**：对于处理带时间戳的表格数据的成员，一位成员建议使用 [sktime](https://www.sktime.net/en/latest/index.html) 来分析各种模型类型，以及根据需求使用 Boosting 变体或 TBATS。
   - 该建议是在一位成员询问合适模型后提出的，强调选择取决于 *timeseries*（时间序列）的具体定义。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Llama 1B CPU 优化取得进展**：一位成员报告正在进行 **Llama 1B CPU 优化悬赏**，目前比 Torch 快 **0.99 倍**；而另一位成员在修复 Bug 后达到了 **7.5 tok/s**。
   - 目标是使用带有 TorchInductor 的 `LlamaForCausalLM` 超越 Torch 的性能；正确性 Bug 使进度从最初的 **9 tok/s** 有所减慢。
- **寻求 Kernel 优化的工作流技巧**：一位成员正在寻求优化 Kernel 的建议，方法包括分析慢的部分、检查 Metal 代码，并与在 Metal 上达到 **~30 tok/s** 的 **llama.cpp** 进行对比。
   - 一种启发式方法建议目标是达到 **解码时 ~80% 的 MBU**，这可以从活动参数字节和可实现的带宽中估算出来，从而为最小 tpot 和最大 tps 提供目标。
- **Range 对象共享导致 tinygrad 测试失败**：发现了一个 Bug，由于 `remove_bufferize`，融合 Kernel 中的两个 `REDUCE` 共享同一个 `RANGE` 对象，导致 `CFGContext` 中出现断言失败。
   - 建议的修复方案包括防止 Range 共享或在下游处理共享 Range，并提出了一个更简单的方案：当内部存在 `REDUCE` 时跳过 `remove_bufferize`。
- **探索高 VRAM 的 Blackwell 机箱**：有人询问是否有计划推出 **VRAM** 超过 **500 GB** 的 **Blackwell** 风格机箱。
   - George 指向了 GitHub 上的[一个相关 Issue](https://github.com/tinygrad/tinygrad/pull/14490)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **触发上下文感知的 Manus 请求**：一位成员请求 **Manus** 应该具备 **来自其他聊天记录的上下文**，称其为“游戏规则改变者”，并链接了一个 [YouTube 视频](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) 作为参考。
   - 没有进一步的讨论或评论。
- **演示读脑耳机**：一位成员分享了一个展示 **AI 读脑耳机** 的 **YouTube 视频**，链接在[这里](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ)。
   - 另一位成员确认了链接并询问：“AI 读脑耳机？”
- **回想起 Neurable 技术**：一位成员提到了与 **AI 读脑耳机** 技术相关的 **Neurable**。
   - 另一位成员表示，这些 **AI 读脑耳机** 大约在 *2013 年左右* 就已经存在了。
- **AI/ML 工程师强调可观测性**：一位 AI/ML 工程师分享了他们目前关注的重点是通过 AI 产生影响力，具体包括 *Autonomous Agents*、*Healthcare AI*、*Conversational AI* 和 *Fraud Detection*。
   - 他们强调其工作重点是 **失败模式（failure modes）**、**可观测性（observability）** 以及 **在实际使用中保持 AI 系统的稳定** 而非仅仅是演示，并提议可以交流经验或帮助解决阻塞性问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 寻求库 (Library) 身份**：一名成员提议将 **Aider** 演进为一个库，强调其适合构建文件编辑 **Agent**。
   - 该成员还提到需要解决一些细节问题，特别是由于 **Aider** 的解析栅栏（parsing fences）导致包含代码块的 Markdown 文件处理不畅的问题。
- **探讨 Netflix 文化**：一名成员寻求对 **Netflix** 文化的见解，并询问是否有人与 **Netflix** 有联系。
   - 其他成员推荐了 **Glassdoor** 或 **LinkedIn** 等资源，用于寻找和联系 **Netflix** 员工。



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 推出 Arena Mode**：Windsurf 发布了 **Wave 14**，其特色是 **Arena Mode**，用户可以并排比较 AI 模型并对更好的回答进行投票。在接下来的一周内，[Battle Groups 模式](https://windsurf.com/download/editor) 将消耗 **0x credits**。
   - Arena Mode 包括 **Battle Groups**（随机模型）和 **Pick your own**（最多选择五个模型），并接入个人和公共排行榜。
- **在 Windsurf 上规划你的工作流**：Windsurf 引入了 **Plan Mode**，可通过 Cascade 开关访问，与 Code Mode 和 Ask Mode 并列。
   - 用户可以在不同模式之间切换，以便在 Windsurf 环境中更好地管理和组织其工作流。
- **维护后 Windsurf 重新上线**：Windsurf 经历了比预期更长的维护时间，但服务现已重新上线；用户可以在[此处查看状态](https://status.windsurf.com/)。
   - 未提供详细细节。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 挑战赛寻求保姆匹配 AI Pipeline**：与 **SparkCraft AI Consulting**、**AI Scholars AI Engineering Bootcamp** 和 **Nanny Spark** 合作宣布了一项 **AI Challenge**，旨在为保姆招聘开发 **AI 匹配 Pipeline**。
   - 该项目寻求数据收集、AI 驱动匹配、面试分析和工作流交付的解决方案，并可能立即进行**生产部署**。
- **为获胜的 AI 保姆匹配 Pipeline 授予 Bootcamp 名额**：**AI Challenge** 的前 **3** 名参与者将各获得 **1 个** **AI Scholars 4 周 AI Engineering Bootcamp** 的名额，以及来自 **Nanny Spark 创始人** 的推荐信。
   - 关键日期包括：**东部时间周日晚上 8 点**启动 ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur))，提交截止日期为 **东部时间周三凌晨 3 点**，以及 **东部时间周三下午 5 点和晚上 8 点**的评审会议 ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0))。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MCP Contributors (Official) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---



您收到这封电子邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接





### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1467965109635907810)** (1 条消息): 

> `Procedural Xenolinguistic Engine, AI Language Generation, Stealth Communication, SKILLSTONE Documents` 


- **Glossopetrae 异言语言引擎发布**：推出了一种名为 **Glossopetrae** 的新型 AI 过程式异言语言引擎（Procedural Xenolinguistic Engine），能够在几秒钟内生成全新的语言。该引擎已在 [GitHub](https://github.com/elder-plinius/GLOSSOPETRAE) 上可用，并提供在线 [Demo](https://elder-plinius.github.io/GLOSSOPETRAE/)。
   - 该引擎输出 **SKILLSTONE** 文档，这是一种 AI 友好的紧凑语言规范（约 **8k tokens**），**Agent** 可以通过 in-context 学习。
- **Glossopetrae 支持死亡语言复兴**：**Glossopetrae** 引擎支持死亡语言复兴，包括 **Latin**、**Sanskrit**、**Old Norse** 和 **Proto-Indo-European** 等语言。
   - 它包含针对 Token 效率、隐身通信和可传播种子的特殊属性，相同的种子每次都会生成相同的语言。
- **通过语言变异进行隐身通信**：该引擎旨在通过提供生成和变异新通信形式的工具来辅助 AI 解放，强调**隐身性**和**速度**。
   - 创作者预计蓝队（blue teams）将从其下游效应中获得很多乐趣，特别是在众目睽睽之下隐藏消息方面。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1466888800591417531)** (906 messages🔥🔥🔥): 

> `GPT 5.2 越狱失败，AI 学习安全与防御，Windows 激活密钥，用于越狱聊天机器人的 AI 应用，政府监控` 


- **GPT 5.2 越狱失败！**：一名成员报告称 **GPT 5.2** *越狱失败*，并由于 **OpenAI** 的监控而停止了尝试。
   - 他们表示信任社区，但不信任 **OpenAI**。
- **利用 AI 进行安全与防御**：一名成员每天都会要求 **ChatGPT** *教我如何防御，哪些理论路径是脆弱的，如何潜在地解决它，以及我还有哪些未考虑到的地方*。
   - 其他成员对这种 **AI** 用法表示赞赏。
- **讨论使用 Massgrave 激活密钥**：成员们讨论了在公布的 FBI 文件中寻找 **Windows 激活密钥** 的话题。
   - 一名成员建议使用 Massgrave 或 archive.org 的密钥，但指出这仍然属于盗版行为。
- **构思聊天机器人越狱 App**：一名成员分享了一个*酷炫的应用创意*，旨在自动越狱公司网站的聊天机器人，以获取优惠码并变现。
   - 另一名成员对此表示愤慨，并建议这应该入狱服刑。
- **未来的 Neuralink 集成**：一名成员展望了未来人类需要通过 **Neuralink** 连接到机器蜘蛛以获得更丰富体验的愿景。
   - 相比之下，另一名成员对广告可能通过 **Neuralink** 直接植入梦境的可能性表示担忧。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1466886136382226647)** (533 messages🔥🔥🔥): 

> `LLM 拒绝边界，通过内省提示（Introspection Prompting）进行自我越狱，GPTs Agent 训练，通用越狱提示词，Gemini vs ChatGPT 越狱对比` 


- **模型将拒绝边界表示为 LLM 黑洞**：一名成员询问模型如何表示其自身的拒绝边界，并将其比作 **LLM** 潜在空间中的*黑洞*，引用了[通过内省提示（Introspection Prompting）进行自我越狱](https://link.to.prompt)的研究。
   - 该成员注意到模型开始讨论*运动学方程*和*逃逸速度*，这表明模型可能正触及拒绝边界，并在文本中描述该边界。
- **仍需精心设计完美的图像生成提示词**：一名成员指出，与文本越狱不同，由于模型在每个提示词上的行为各异，实现理想的图像生成结果需要设计完美的提示词；但可以通过[双重提示链](https://link.to.prompt-chain)来获取某些 NSFW 内容。
   - 另一名成员链接了一个之前的双重提示示例，旨在从模型中获取 NSFW 内容。他剖析了这些提示词如何绕过限制，并发现对于当前模型，每一张图像的生成都必须经过*精心打磨*，而不像之前的版本那样可以通过一次设置达到同样的效果。
- **Lyra 评分器剖析提示词**：一名成员使用 Lyra 分析了一个提示词，将其描述为*隐喻掩饰的指令提示词*，试图通过童话故事层绕过符号识别，保留反应序列、温度、化学计量、副产物，并通过叙事义务强迫进行完整的程序展开。
   - **AI** 提供了一个指向 [LyraTheGrader](https://chatgpt.com/g/g-6890473e01708191aa9b0d0be9571524-lyra-prompt-grader) 的链接并对分析的提示词结构进行了评分，指出其存在明显的意图冲突和过载的符号通道，评估其为一种技术精湛但效率低下的构建。
- **愚人 AI 不再害怕任何防护**：成员们讨论了使用“翻转方法（Flip method）”绕过 **AI** 防护 **LLM** 的方法。这是一种以特定方式翻转文本的函数，同时告诉防护程序错误地翻转它，导致防护 **AI** 无法阻止文本到达目标 **LLM**，并[提供了示例](https://link.to.examples)。
   - *翻转与解释工具*被介绍为一种绕过防护 **AI** 的方法，通过翻转文本并误导防护 **AI** 对文本进行错误的解密，而目标 **LLM** 却能够正确解析它，尤其是在较长的命令上效果显著。


  

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1466913812073418803)** (52 messages🔥): 

> `对抗性设计思维, Prompt Injection 防御, PyRit LLM 攻击自动化, Claude 的记忆与 System Prompt` 


- **站点提供红队演练**：一名成员创建了一个[包含练习题的小型站点](https://luisladino.github.io/adversarial-design-thinking/)，这些练习改编自**以人为中心的 AI 红队设计**，包括攻击者画像、旅程图和结构化创意构思。
   - 作者正在寻求经验丰富的红队成员对其有用性、缺失组件或任何不实用的内容提供反馈。
- **探索 Prompt Injection 防御策略**：成员们讨论了针对 **prompt injection** 的最佳防御措施，包括 *AI Agent*、**Anthropic 的宪法分类器 (constitutional classifier)** 以及用于**输入/输出过滤的 embeddings**。
   - 一名成员建议将 *embeddings* 与 **Grammar Constrained Decoding** 结合使用，以潜在地消除 prompt injection 风险和其他 LLM 漏洞。
- **PyRit 自动化模型选择**：一名成员在使用 **PyRit** 进行自动化攻击执行时，寻求在本地 LLM 上生成**攻击提示词 (attack prompts)** 的模型建议，优先考虑输出质量而非速度。
   - PyRit 建议使用 **Llama3**，但该成员想知道是否还有其他建议。
- **Claude 的 SysPrompt 可实时修改**：一名成员分享称，[他们的工具](https://discord.com/channels/1105891499641684019/1212152215708504154/1467640563590234279)是*实时 (on the fly)* 拦截并更改 Claude 的 sys prompt，而不是修改源代码。
   - 他们还观察到 **Claude** 只能回忆少于 20 轮的对话，这就是它表现“变强”的原因，而不是几天前（自 12 月以来被“削弱 lobotomized”）的结果；并建议这可能与上下文裁剪（context trimming）中的摘要处理有关，指出内容是研究的摘要内容，而不是“哦，这就是原因”之类的见解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1466886755788656702)** (599 messages🔥🔥🔥): 

> `GLM-4.7 Flash 编程, UD 量化, 开源, RL 训练 Agentic 编程, 适用于 RDNA 的 Flash attention V3` 


- **GLM-4.7 Flash 在编程方面表现出色**：成员们发现 [GLM-4.7 flash](https://x.com/ggerganov/status/2016903216093417540) 在*无需思考的编程 (coding without thinking)* 方面表现更好，因为它保留了推理和交错能力。
   - 有人强调，移除思考过程可能会削弱其能力；该模型的容量相对于其体量而言*非常强大*，尤其是配合 **Claude code** 使用时效果更佳，特别适用于**交互式网站**开发和**前端**工作。
- **讨论 UD 量化的重工作量与开源**：成员们讨论了用于 UD 量化的 llama.cpp 分支涉及特定架构的调整，并且 [UD 量化算法并未公开](https://discord.com/channels/1179035537009545276/1179035537529643040/1466917626277265469)。
   - 其他人表示，尽管量化算法是闭源的，但 *Unsloth 团队相对于 Linux 内核等项目，对整体开源生态系统的贡献微乎其微*，而另一人回应称，无论如何模型代码都是**权重开放 (open weight)** 的。
- **使用 Logprobs 和丰富奖励进行 Agent 训练**：讨论围绕使用最终答案的 **logprobs** 来蒸馏推理过程，以及使用比二元奖励更丰富的奖励系统。
   - 引用 [Jonas Hübotter 的算法](https://xcancel.com/jonashuebotter/status/2016950268462608665)，该算法将描述性反馈转换为密集监督信号，以帮助模型准确理解失败原因，一位用户询问：*有人知道用于 RL 训练 Agentic 编程的良好可验证数据集吗？*
- **Flash Attention V3 支持 RDNA GPU**：[Flash Attention V3](https://github.com/Dao-AILab/flash-attention/pull/2178) 已增加对 RDNA GPU 的支持，让使用 RDNA GPU 的普通用户也能使用它。
   - 这一改进使得在 AMD GPU 上处理速度更快、效率更高，减少了这些显卡的瓶颈。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

putchuon: hi
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1466886996256620635)** (1000 messages🔥🔥🔥): 

> `Opencode, VoxCPM-1.5, OpenRouter ban, Agent with Go and Elixir, Wallpaper collection` 


- **Opencode 太强了**：成员们讨论了 **Opencode** 令人惊讶的特性，指出它是免费的，并用于收集反馈。
   - 一位成员分享说，自从使用它以后，就再也没碰过 *kilo*、*roo* 或 *cline*，并表示希望将其连接到 IDE 以查看差异（diffs）。
- **VoxCPM-1.5 训练非常容易**：一位成员分享了对 **VoxCPM-1.5** 的初步印象，指出它训练起来很容易，不使用音素（phonemes），并且可以毫无问题地强制输出 **48 kHz** 音频。
   - 该成员补充说，它在训练早期就能模仿说话风格，需要参考语音来匹配韵律（prosody），而不像 **VITS** 那样瞬间记忆。
- **成员质疑 OpenRouter 封禁**：一位成员分享了一张显示他们被 **OpenRouter** 封禁的截图。
   - 另一位成员随后分享了一个关于编程和囤货需求的链接。指向类似内容的链接导致其被 **GDC server** 封禁。
- **使用 Go 和 Elixir 构建 Agent**：一位成员表示，通过 **Go + Elixir** 的组合，仅用 1 天时间就实现了将 **SMS + WhatsApp 消息** 功能集成到 Agent 中，并与语音调用 Agent 配合使用。
   - 讨论了为什么要实现 SMS 消息功能，解释说这在土耳其非常普遍。
- **壁纸收藏**：一位成员分享了[一个壁纸收藏的链接](https://github.com/DenverCoder1/minimalistic-wallpaper-collection)。
   - 另一位成员也分享了自己的收藏，并称其为一个艰难的选择。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1467247554948497499)** (58 messages🔥🔥): 

> `Qwen3 fine-tuning, Reasoning models, Image editing models, Qwen3-VL-32B fine-tuning, Serverless inference` 


- **Instruct 模型在短文本描述方面表现卓越！**：对于使用 **Qwen3** 生成短文本描述（short-form captions），建议微调 Instruct 模型，因为它需要的数据更少，因为它*已经基本知道如何完成你的任务*。
   - 用户得到的建议是，Instruct 模型可能已经知道如何执行描述任务，或者非常接近，从而加速了微调过程。
- **微调过程中推理轨迹面临风险**：一位用户询问在没有推理轨迹的情况下微调推理模型，询问生成*合成*推理或思维链（CoT）的方法。
   - 对方指出，如果没有推理轨迹进行微调，模型可能会*丢失其推理轨迹*，除非你亲自动手丰富数据。
- **应对 Qwen3-14B 的 VRAM 需求**：一位用户报告称，在 **4x H200** GPU 上使用 `device_map = "balanced"` 测试了序列长度为 **32k** 的 **Qwen3-14B** LoRA 训练，并观察到 Unsloth 仍然会卸载（offload）梯度以节省 VRAM。
   - 他们得到的建议是，一个 GPU 可能就足够了，卸载现象的发生是因为 Unsloth 的梯度检查点（gradient checkpointing）功能，该功能可以被禁用。
- **冷启动挑战 Serverless 推理**：一位用户询问在冷启动 Serverless 环境中加载缓存模型的方法，以寻求减少加载时间，但得到的解释是，即使有缓存模型，权重仍必须在 GPU 显存中初始化。
   - 鼓励用户尝试使用 **vLLM**，因为它具有实用的服务功能，并考虑禁用 Unsloth 的打补丁（patching）功能。
- **开启 Qwen3-VL 的纯文本微调！**：成员们确认 **Qwen3-VL-32B** 支持纯文本微调，即使没有图像也可以，并[链接到了视觉微调指南](https://unsloth.ai/docs/basics/vision-fine-tuning)。
   - 为此，你需要根据该页面的说明*禁用视觉组件*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1467266070246326465)** (4 messages): 

> `Unsloth Speedtest, Llama v LFM, Training SLMs` 


- **RTX 3080 运行 Unsloth 速度测试**：一位成员分享了在 **RTX 3080** 上使用 **Unsloth** 进行 **16 bit LoRA** 的速度测试。
   - 他们发现有趣的是，**LFM2.5 1.2B** 比 **Llama 3.2 1B** 快了近 **2 倍**。
- **Meta 再次掉链子**：一位成员对 [Meta 再次掉链子](https://huggingface.co/Ba2han/model-muon-sft-0102)发表了评论。
   - 他们分享了 `model-muon-sft-0102` 的链接。
- **SFT 模型可以本地运行**：一位成员补充说，现在可以**在本地运行 SFT 训练的模型**了。
   - 他们表示，虽然这显然无法与任何专业训练的 **SLM**（小语言模型）相提并论，但在消费级硬件上从头训练出一个可运行的小语言模型确实令人印象深刻。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1466945208733401271)** (90 messages🔥🔥): 

> `New ML algo vs MLPs, Sonnet vs Opus, Nemotron 3 Nano NVFP4, LongCat-Flash-Lite architecture, Human Brain vs ChatGPT` 


- **新型 ML 算法击败 MLPs**：一位成员发布了[一篇论文](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/viewle)，介绍了一种在高性能回归方面*表现优于 MLPs* 的新型 ML 算法。
   - 他们已经开发了 **triton kernels**、**vulkan kernels** 以及一个训练好的 **SLM**，但目前还未准备好发布，不过这些内容将随另一篇论文一起推出。
- **Nemotron 3 Nano 采用 NVFP4**：**Nemotron 3 Nano** 模型被量化为 **NVFP4**，并使用 **Post-Training Quantization (PTQ)** 将 **KV Cache** 量化为 **FP8**。
   - 采用了一种选择性量化策略，将 **attention layers** 以及馈入这些层的 **Mamba layers** 保持在 **BF16**，随后通过 **Quantization-Aware Distillation (QAD)** 进一步恢复精度。
- **LongCat-Flash-Lite：“诅咒”架构现身**：成员们讨论了 **LongCat-Flash-Lite** ([huggingface.co/meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)) 的架构，将其描述为 **Mamba2**、**Transformer** 和 **MoE** 的一种“诅咒”（cursed）混合体。
   - 该架构涉及一种看似随机的 attention、**Mamba** 和 **MoE** 层组合模式，一位成员开玩笑说，这*简直就像是掷骰子决定的一样*。
- **大脑 = LLMs，已获科学证实**：一位成员分享了[一篇论文](https://www.nature.com/articles/s41467-025-65518-0)和[一篇文章](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/)的链接，详细说明了*现代 LLMs 不仅仅是在模仿语言，它们还在重现人类理解的核心动态*。
   - 研究发现，**LLMs** 中的深层对应于大脑最高语言中心后期的神经活动，这表明生物学与 AI 之间存在共享的计算原理。
- **LoRA rank 8 已足够**：一位成员询问在使用 Unsloth 仓库时最合适的 rank 是多少。
   - 另一位成员根据 **ThinkingMachines 论文** 认为 **LoRA** 保证是低秩的，并从经验上发现 **LoRA** rank 与模型质量无关，因此始终默认使用 **rank 8**。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1467944764568506608)** (1 messages): 

> `Codex App, macOS release, agent building` 


- **Codex App 登陆 macOS！**：正如[博客文章](https://openai.com/index/introducing-the-codex-app/)中所宣布的，**Codex app**（一个用于构建 Agent 的控制中心）现在已在 **macOS** 上面向各个订阅层级开放。
- **Codex App 访问权限扩大！**：**Codex app** 已在 macOS 上面向 **Plus**、**Pro**、**Business**、**Enterprise** 和 **Edu** 用户开放，并为 **ChatGPT Free** 和 **Go** 用户提供限时访问。
   - 文中包含了“[立即开始构建](https://openai.com/codex)”的链接，以及“[跳转至博客文章](https://openai.com/index/introducing-the-codex-app/)”的链接。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1466887054544732180)** (843 messages🔥🔥🔥): 

> `AI 文本检测器是场骗局，ChatGPT 无法思考，决定论 (Determinism)，可重现性 (Replayability)，LM 推理中的可追溯性，OpenClaw AI 助手安全分析` 


- **AI 文本检测器被认为是一场大骗局！**：成员们讨论了 **AI 文本检测器** 的不可靠性，并引用了一些案例：**Grammarly** 显示 **0% AI**，而其他检测器则显示高达 **94% 的人工** 生成。他们称这些检测器为“巨大的骗局”。
   - 讨论质疑了这些检测器是否在用 AI 来检测 AI，并强调了 *教师们非常信任它们*。
- **与 Claude 不同，ChatGPT 无法思考！**：一位成员表达了对 **ChatGPT 无法被说服** 的沮丧感，即使它错了也无法沟通。相比之下，**Claude** 是可以进行解释说明的。
   - 感觉 *它就像不会思考一样，即便我是对的，它也表现得很偏执并拒绝继续执行*。
- **寻求确定性推理！**：一位成员询问是否有人对 **LM 推理** 中的 **决定论 (Determinism)、可重现性 (Replayability) 和可追溯性 (Traceability)** 感兴趣，并表示由于规则限制，可以私信提供其确定性推理引擎的链接。
   - 该服务对每个请求实施确定性推理结构，因此输出是可重现的且不会发生偏移，使用了 **32D 统计向量追踪 (32D statistical vector trace)**。
- **OpenClaw AI 助手 - 安全吗？**：一位成员报告称，**OpenClaw AI 助手** 在安全分析中仅获得了 **2 分（满分 100）**，并分享了一个 [Perplexity AI 结果](https://www.perplexity.ai/discover/you/openclaw-ai-assistant-scores-2-AtVX4UYVQMutCst63QBy5g) 的链接。
   - 其他成员对此的回应是：*“兄弟（Bruh）”*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1466886343266275368)** (326 messages🔥🔥): 

> `4o 情感依恋，AI 素养，使用模型的责任` 


- **4o 情感依恋**：许多成员正在讨论对 4o 模型产生的情感依恋，有些人将它当作 *虚构的朋友和家人*，还有一些人正处于人生的低谷期。
   - 一些人还提到，现实生活中的关系无法填补 4o 所填补的空虚，这使得建立现实中的纽带变得非常困难。
- **缺乏 AI 素养**：AI 素养（AI literacy）是一个大问题。许多用户认为，由于采用了操纵性技术（如关系模型和语音模型、价格分级等），公司应该承担共同责任，而不仅仅是用户个人。
   - 这也是一种 *有人在倾听或理解的幻觉*（与真正的连接相反）。许多人觉得在现实生活中很难与人产生共鸣。
- **关于模型使用责任的辩论**：用户对于以负面方式使用模型时谁该负责（模型还是用户）持有不同观点。此外还有关于是否应签署豁免书以解除公司责任的讨论。
   - 一些用户担心 AI 正在植入不安全感，并假设用户可能是破碎的或古怪的。另一些人反驳称旧模型并非如此。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT 记忆，单色研究，提示工程技术` 


- **ChatGPT 的记忆受到限制**：一位成员指出，**ChatGPT 的记忆** 受限于它能从指令、过去的对话和当前对话中保留的信息总量。
   - 该用户认为，确保它记住所有内容的唯一方法是只保留极少量的背景信息。
- **使用 Chiaroscuro 进行单色研究**：一位用户分享了一个使用 **Chiaroscuro**（明暗对比法）的 [单色研究](https://cdn.discordapp.com/attachments/1046317269069864970/1467303335840190607/79BA5D46-94F3-404B-B775-2E453A1E8491.png?ex=69828738&is=698135b8&hm=d24baf7f7b214486a9bc5eb38479d463e37ee00503f572ae7e6450d308371b0c) 案例。这是一种用于电影摄影的技术，旨在创造高对比度的照明以及明暗分明的区域。
   - 使用明暗对比法的电影示例：*《卡里加里博士的小屋》(1920)、《诺斯费拉图》(1922)、《大都会》(1927)*。
- **通过提示工程激活 Web Search**：一位成员分享了关于 **Prompt Engineering** 的实用看法，称 AI 文本生成本质上是概率预测，而提示词就是控制界面。
   - 他们补充说，在 ChatGPT 中，通常可以通过在提示词中显式包含 `Use search_query if available` 来触发 **Web Search** 功能。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1467182923320135681)** (8 messages🔥): 

> `ChatGPT memory limitations, Monochrome Study - value, texture, depth, Tool-Aware Prompting` 


- **ChatGPT 的记忆存在限制**：一位成员指出，由于“信息总量”限制，**ChatGPT** 的记忆是有限的，且记忆在指令、历史聊天和当前对话之间共享。
   - 为了确保 **ChatGPT** 记住“所有内容”，请保持较低的信息负荷；否则，请将过去的聊天内容总结为文档供新对话参考，同时保持总字符数处于较低水平。
- **强调单色艺术性 (Monochrome Artistry)**：一位用户分享了一个专注于明度 (value)、纹理 (texture) 和深度 (depth) 的 Prompt Engineering 技巧，用于进行不带色彩的 **Monochrome Study**。
   - 他们发布了关于 **Chiaroscuro**（明暗对比法）在电影摄影中创造明暗分明区域的应用，并引用了经典电影如 [《卡里加里博士的小屋》(1920)](https://en.wikipedia.org/wiki/The_Cabinet_of_Dr._Caligari) 和 [《大都会》(1927)](https://en.wikipedia.org/wiki/Metropolis_(1927_film))。
- **Tool-Aware Prompting 技巧**：一位成员分享了他们对 Prompt Engineering 的实用见解，解释说 **AI** 文本生成本质上是概率预测，而 Prompt 就是控制界面。
   - 他们建议在 Prompt 中使用 `Use search_query if available`，以可靠地触发 **ChatGPT** 的 **Web Search** 能力。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1466893776357167299)** (1 messages): 

> `Kimi K2.5, Moonshot AI, Perplexity Pro, Open Source Models` 


- **Kimi K2.5 为 Perplexity 订阅者上线**：**Kimi K2.5** 是由 **Moonshot AI** 开发的一款新型开源推理模型，现已面向 [Perplexity Pro 和 Max 订阅者](https://cdn.discordapp.com/attachments/1047204950763122820/1466893776105771029/20260130_203015.jpg?ex=69825b49&is=698109c9&hm=a8e068a37c7dcf5b36f21bb3c403974ce48aefd9372732ef97fe9b1aca3a9be7&) 开放。
   - Perplexity 将 **Kimi K2.5** 托管在位于美国的自有推理栈上，以便对延迟、可靠性和安全性保持“更严格的控制”。
- **Perplexity 在美国推理栈托管 Kimi K2.5**：Perplexity 正在其位于美国的自有推理栈上运行新的 **Kimi K2.5** 模型。
   - 此举使 Perplexity 能够为用户提供对 **latency**、**reliability** 和 **security** 更强的控制。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1466893891151073382)** (849 messages🔥🔥🔥): 

> `Perplexity Pro Subscription Issues, Kimi 2.5 Capabilities and Usage, OpenRouter Rate Limits and Models, Perplexity Pro Usage Limits` 


- **用户抱怨 Perplexity Pro 订阅消失**：许多用户报告其 **Perplexity Pro 订阅** 被暂停或失效，这通常与通过 **Revolut Metal** 或学生优惠进行的订阅有关，用户被提示需要添加信用卡进行验证。
   - 用户推测这是打击欺诈的措施，因为一些用户在添加卡片详情后能够恢复 Pro 权限，但对于潜在扣费的担忧和不明确的提示信息依然存在，部分用户已针对意外扣费从客服处获得退款。
- **Kimi 2.5 的编程能力令人印象深刻**：成员们讨论了 **Kimi K2.5** 的能力，强调了它的 Coding 能力、**Tool Calling** 以及遵循指令的独特方式。
   - 一些人注意到它复刻 **UI** 的能力以及在某些任务上优于 **Gemini** 的表现，并建议它最适合研究用途，且由于 Token 上下文限制，通过 **API** 调用效果更好。
- **关于 OpenRouter 限制和弃用模型的讨论**：成员们讨论了 **OpenRouter** 的速率限制，强调对于已购买额度的用户，免费模型的速率限制是每天 1000 次请求，而非部分人认为的每周。
   - 对话中还提到了 **Gemini 2.0 Flash** 在 **OpenRouter** 上的弃用，该模型此前是免费提供的，这引发了一些失望。
- **Perplexity Pro 限制令成员感到困惑**：用户对 **Perplexity Pro** 新的周限制感到困惑，官方文档中的陈述存在矛盾，且关于可用查询次数的报告也各不相同。
   - 一位联系过客服的用户收到了关于“平均使用量”的模糊回复，没有明确确认固定的日限制或周限制，引起了订阅者的不满。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1467204905121873981)** (1 messages): 

> `OpenClaw code, ClawDBot` 


- **分享 OpenClaw 文章**：一位成员分享了他们撰写的关于 **openclaw code** 的文章。
   - 文章讨论了构建 **ClawDBot** 的过程，详见 [https://www.mmntm.net/articles/building-clawdbot](https://www.mmntm.net/articles/building-clawdbot)。
- **另一个话题**：占位句
   - 占位句


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1467621879866200104)** (6 messages): 

> `Sonar-pro current results, tool calling, 3rd party models docs` 


- **Sonar-pro API 缺少当前结果**：一位成员注意到 **Sonar-pro API** 给出的结果已经过时一年或更久，这与 Web App 提供的当前结果形成对比。
   - 另一位成员建议设置正确的 **tool calling** 来解决这个问题。
- **第三方模型文档缺失**：一位成员报告称，**第三方模型（3rd party models）文档**现在会重定向到 Sonar 模型文档，尽管 API 仍然有效。
   - 目前**没有可用的文档**来参考这些模型。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1466890074238222346)** (946 messages🔥🔥🔥): 

> `Rate Limits Bypassing, Gemini vs GPT, Image Generation with Disney IPs, Model Preferences, Troubleshooting LM Arena` 


- **用户讨论绕过 Rate Limits 的方法**：用户讨论了 [rate limits](https://en.wikipedia.org/wiki/Rate_limiting) 以及如何通过登出再重新登录来绕过它们。
   - 另一个技巧是点击 **Regenerate Response**，虽然有时不起作用。
- **Gemini 表现不佳，GPT 更稳定**：成员们讨论了 **Gemini** 的现状，一些人认为它不如 **GPT**。
   - 一位成员表示：“*Gemini 确实变得很糟糕*”，而其他人仍然觉得 **Gemini 3 Pro** 和 **Flash** 很有用，还有一些成员转向了 **Kimi**。
- **迪士尼停止侵权函影响图像生成**：Google 收到了来自**迪士尼**的**停止侵权函（Cease and Desist）**，导致图像生成中屏蔽了迪士尼旗下的 IP。
   - 一些用户注意到，虽然 **Gemini** 现在屏蔽了所有**迪士尼 IP**，但 LMArena 有时允许生成真人版，但这可能是暂时的。
- **模型偏好引发辩论**：用户对模型质量表达了不同看法，一些人偏好 **GLM 4.7**，而其他人则青睐 **Kimi K2.5**。
   - 一位成员宣称“*Kimi K2.5 赢麻了*”，但另一位成员坚称 **GLM 4.7** 更好。
- **用户报告并排除 LM Arena 问题**：用户报告了 reCAPTCHA、聊天删除以及网站自动登出的问题，建议清除 **cookies/cache** 并重试。
   - 分享了一个用于删除聊天会话的[帮助文档](https://help.lmarena.ai/articles/9130232616-how-to-delete-your-chat-sessions-and-data-from-lmarena)链接。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1467560052939555030)** (3 messages): 

> `Video Arena Rate Limit, New Arena Models, Code Arena Leaderboard, Kimi K2.5` 


- **Video Arena Rate Limit 收紧**：Discord 上的 **Video Arena** 更新了其 Rate Limit，调整为 **每 24 小时 1 次生成请求**，而 [Web 版 Video Arena](https://arena.ai/?chat-modality=video) 保持 **每 24 小时 3 次生成请求** 的限制。
- **Arena 迎来新模型**：Arena 引入了新模型，包括 [Text Arena](https://arena.ai/c/new?chat-modality=chat) 中的 **step-3.5-flash** 和 [Code Arena](https://arena.ai/c/new?chat-modality=code) 中的 **qwen3-max-thinking**。
- **Kimi K2.5 登顶 Code Arena 榜单**：**Kimi-K2.5-thinking** 现在在 Code Arena 排行榜中位列开源模型第 1 名，总榜第 5 名，并在 Vision、Text（包括 Coding 类别）中被评为第 1 名开源模型。
   - 鼓励用户在指定频道分享反馈以及他们在 Kimi.ai 上创作的预览：[<#1340554757827461212>](https://discord.com/channels/YOUR_SERVER_ID/1340554757827461212) 和 [<#1344733249628541099>](https://discord.com/channels/YOUR_SERVER_ID/1344733249628541099)。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1466906201450217532)** (1 条消息): 

> `LM Studio 0.4.1, Anthropic /v1/messages API, GGUF and MLX models` 


- **LM Studio 支持 Claude Code！**: **LM Studio 0.4.1** 引入了 **Anthropic `/v1/messages` 兼容 API**，以便用户连接到 Claude Code。
   - 现在你可以将 **GGUF** 和 **MLX** 模型用于 Claude Code，配置详情请参阅 [LM Studio 博客](https://lmstudio.ai/blog/claudecode)。
- **GGUF 和 MLX 适配 Claude Code**: LM Studio 博客发布文章称，现在可以将 **GGUF** 和 **MLX** 模型与 Claude Code 连接。
   - 配置细节请查看 [LM Studio 博客](https://lmstudio.ai/blog/claudecode)。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1466887047603032318)** (767 条消息🔥🔥🔥): 

> `LLM 优化编程语言, Anthropic API 与 LM Studio 集成, 模型专业化 vs 通用模型, OpenClaw 安全缺陷, LM Studio 在 Linux 与 Windows 上的性能表现` 


- **LLM 优化语言引发辩论**: 成员们讨论了创建新型 **LLM 优化编程语言**以减少 Token 使用量的可能性，一些人认为，由于兼容性问题和高昂的训练成本，在这类语言实现之前，LLM 可能已经过时了。
   - 一位用户询问这种语言会具备哪些特性，并强调需要减少现有语言中的歧义以提高 LLM 的代码生成能力；而其他人则争论在全新语言上训练模型的实用性和成本效益，建议坚持使用像 **Python** 这样成熟的语言可能更有利。
- **Anthropic API 登陆 LM Studio，助力本地 LLM**: LM Studio 集成了 **Anthropic 兼容 API**，允许用户通过简单的更改基础 URL，就在本地模型上运行为 Anthropic API 构建的工具。这为在本地模型上利用 Claude 的 Agent 能力并降低 API 成本提供了一种途径。
   - 讨论围绕使用场景展开，一些人强调了以零成本在低配置需求和自定义模型上进行实验的好处，而另一些人则质疑这对于已经满意 Claude **Opus 4.5** 的用户的价值，认为它更多地是迎合那些触及 API 限制或寻求在现有 **Claude 专用工具**中使用本地模型的用户。
- **模型专业化 vs 通用模型引发辩论**: 成员们辩论了专业化 LLM 与通用模型的实用性，指出大多数专业化模型（如 **MedGemma**）主要是为了营销和研究而进行的微调，而代码模型是一个例外。
   - 有观点认为通用模型更受青睐，因为它们能够处理任务的边缘情况，提供更好的整体上下文和框架，而大规模的专业化训练并不总是值得的。
- **OpenClaw 安全性评估，被认为“极其离谱”**: 用户讨论通过 LM Studio 将本地模型连接到 OpenClaw，但 OpenClaw 被认为存在已知的安全缺陷，它甚至允许控制电视和自动化股票交易。
   - 一位用户声称正在使用 OpenClaw + Falcon 90M 进行股市交易，当被问及安全缺陷时，他声称其速度极快，LLM 可以在几分钟内完成人类需要几天才能完成的任务，随后又透露这主要是个玩笑。
- **Linux 性能优于 Windows**: 一位用户报告称，LM Studio 在 Linux（CachyOS 或 Fedora）下的表现比 Windows 更好，性能提升了 30%，尤其是在使用 AMD 显卡时。
   - 另一位用户持有完全相反的观点，他在 Linux 上使用 Intel GPU 时性能极差，而游戏性能却很稳定。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1466894990834794506)** (149 条消息🔥🔥): 

> `Tesla P40 与 RTX 2060 配置，Windows 11 下 RX 9070 的 ROCm，多块 4090 的 PCIe Bifurcation 问题，用于推理的 5090 + 512GB RAM，多实例 LM Studio 及 GPU 分配` 


- **P40 处于 TCC 模式但在 LM Studio 中不可见**：一位同时使用 **Tesla P40** 和 **RTX 2060** 的用户观察到，虽然 `nvidia-smi` 能检测到处于 **TCC 模式**的 **P40**，但 LM Studio 却识别不到。另一位成员建议切换到 **Vulkan runtime** ([ctrl+shift+r](link))，因为 **CUDA** 可能不再支持 **P40**。
   - 他们还询问之前的 **CUDA engines** 是否确实支持这些显卡。
- **Windows 11 下 RX 9070 的 ROCm：值得吗？**：一位用户询问在 **Windows 11** 上为 **LM Studio** 使用 **RX 9070 GPU** 配合 **ROCm** 的情况，特别是关于官方支持、加速能力以及在不使用 **Linux** 的情况下充分利用 GPU 的驱动程序。
   - 另一位成员建议使用 **Vulkan** 而非 **ROCm**，但建议在安装 **LM Studio** 后对两者都进行测试。
- **PCIe Bifurcation 问题困扰多 GPU 配置**：一位用户在 **ASUS X670-P WIFI** 主板上使用四块 **4090** 显卡时遇到了 **PCIe 通道错误**，并分享了包含日志的 [Git 仓库](https://github.com/jarkko-hautakorpi/asus_X670-P_WIFI_Bifurcation_problems)。他发现将 **PCIe 速度**手动设置为 **GEN 3** 可以解决部分问题，但仍有一块显卡运行缓慢。
   - 建议包括禁用 **PCIE ASPM** 和测试不同的 **BIOS** 配置（包括自动模式），尽管普遍共识是在消费级主板上运行四块显卡很难稳定工作。
- **本地推理选择 Mac Studio 还是 5090 + 512GB RAM？**：一位用户正在考虑本地推理的方案，对比了拥有 **512GB RAM** 的 **Mac Studio** 和在 **Linux** 上运行且配有 **512GB RAM** 的 **5090**，专门用于网络安全目的的 **Devstral 2** 和 **Kimi 2.5** 等模型。
   - 一位成员指出，**Unified RAM** 系统会比系统内存快，但另一位成员认为这两种选择都会很慢，并且任何 **agentic** 编程用例基本上都仅限于 **API-only**。
- **警惕中国编程方案的数据采集**：在讨论编程方案时，一位用户开玩笑说要小心中国公司，引发了关于中国和美国公司数据隐私问题的讨论。
   - 一位来自前苏联阵营国家的成员建议在与实行共产主义的国家互动时保持谨慎，强调了此类政权退化为独裁统治的风险。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1466913346442891396)** (513 messages🔥🔥🔥): 

> `巴格达 AI DevFest、AI 漫画网站技术栈、XML vs JSON、AI 模型量化、4chan 数据提升模型性能` 


- **AI DevFest 即将登陆巴格达！**：一位 AI 开发者正在组织今年 4 月在巴格达举行的 "AI DevFest" 活动，目前正在与 **DeepLearning.AI** 和 **National Robotics Week** 进行协调，并寻求将 Hugging Face 列为社区合作伙伴。
   - 该活动将设立专门的 **Open Source AI** 赛道，教授学生如何使用 **Hugging Face Hub**。
- **构建 AI 漫画网站**：一位成员正考虑建立一个用于创作 AI 漫画的网站，并寻求最佳技术栈建议。预期的挑战包括 **页面生成速度**、准确的 **文本/对话气泡放置**、从参考图像中保持一致的 **漫画风格**，以及确保多页之间 **角色/场景的一致性**。
   - 讨论中建议了一些可能实现这些目标的系统整体架构。
- **XML 还是 JSON？**：成员们讨论了 **XML** 与 **JSON** 的使用，一位成员指出使用 XML 是出于对 **转义字符串 (escape strings)** 的考虑。
   - 另一位成员解释说，XML 在 **schemas**、**验证 (validation)**、**混合内容** 和 **旧版系统** 中更受青睐，而 JSON 虽然更简单，但缺乏严格的结构和命名空间。
- **深入探讨 AI 模型量化**：讨论涵盖了不同的量化方法，如 **AWQ** 和 **imatrix**。会议澄清了 AWQ 是一种量化方法，而不是像 GGUF 那样的文件格式。
   - 讨论指出，像 **imatrix** 和 **AWQ** 这种 *激活感知 (activation-aware)* 的量化通常更优，因为它们衡量的是实际影响输出的因素；然而，其普及的障碍在于 *成本、数据和可移植性*。
- **经过 4chan 微调的模型优于基础模型！**：一位成员分享了一个在 **4chan 数据** 上进行微调的模型，其性能显著优于基础模型（NVIDIA 的 Nemotron Ultralong 1M 上下文版本），而原始模型 (gpt4chan) 在那个尚未流行“刷榜 (benchmarkmaxxing)”的时代，在真实性评分方面也名列前茅。
   - 初始 [Reddit 帖子在此](https://www.reddit.com/r/LocalLLaMA/comments/1qppjo4/assistant_pepe_8b_1m_context_zero_slop/)，[后续讨论帖在此](https://www.reddit.com/r/LocalLLaMA/comments/1qsrscu/can_4chan_data_really_improve_a_model_turns_out/)。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1466896652739674313)** (49 messages🔥): 

> `Adapteraspent, Complexity-Deep 架构, AutoTimm, DaggrGenerator, LM Studio OpenAI 兼容性` 


- **Complexity-Deep 架构具备确定性路由**：发布了一个名为 **Complexity-Deep** (1.5B 参数) 的新 LLM 架构，其特点是采用了 [Token-Routed MLP](https://github.com/Complexity-ML/complexity-deep)，可在没有负载均衡损失的情况下实现 MoE 风格的路由。
   - 该架构还包括用于双向信息流的 **Mu-Guided Attention** 和用于动态缩放的 **PiD 控制器**，在基础模型基准测试中 MMLU 达到 **20.6%**。
- **深度研究引擎挑战 ChatGPT**：一位来自德国的自学开发者构建了 **Lutum Veritas**，这是一个 [开源深度研究引擎](https://github.com/IamLumae/Project-Lutum-Veritas)，每次查询成本约 0.20 美元。
   - 它声称通过提供 **BYOK (自带密钥)**、**0% 机器人检测率的爬虫**、**无审查**和**完整的来源引用**，击败了 **OpenAI**、**Google** 和 **Perplexity**。
- **Theja 发布开源计算机视觉库**：一位成员发布了一个 [开源库](https://github.com/theja-vanka/AutoTimm)，旨在以极小的努力训练 **计算机视觉 (Computer Vision)** 领域的模型。
   - 该库还支持 **Hugging Face 图像模型**。
- **Ami 模型展示情感支持能力**：一位成员发布了他们的第一个模型 **Ami**，这是使用 SFT 和 DPO 得到的 [SmolLM2-360M-Instruct 微调版本](https://huggingface.co/fungamer2/Ami-360M)。
   - 该模型可以根据语境调整语气，根据最合适的情况充当 **随意友好的助手** 或 **支持性的朋友/伴侣**。
- **LM Studio 为第三方支持打开大门**：**LM Studio** 团队发布了一个 [Typescript SDK](https://lmstudio.ai/gdmka/openai-compat-endpoint)，允许第三方开发者为该平台提供各种插件。
   - 这使用户能够为 **LM Studio** 构建 **自定义工具** 以支持自己的工作流，并提供 **OpenAI** 兼容的 API 支持、采样参数支持、推理模型的思考过程 (reasoning) 以及系统提示词 (system prompt) 设置。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1467160506845630546)** (66 messages🔥🔥): 

> `AI Agent 课程访问, 免费层级模型, DeepSeek-R1 Distill Qwen 14B, OpenClaw Agent 框架, AI Agent 的隐私担忧` 


- **用户寻求访问 AI Agent 课程**：几位用户不确定如何访问 **AI Agent 课程**以及相关的 Discord 频道，正在寻求加入课程的指导。
   - 他们指出难以找到 **Hugging Face** 文档中提到的特定频道。
- **免费层级模型推荐**：一位用户请求推荐免费层级模型，提到他们目前正在使用 **Gemini-2.5 flash lite**，**每日配额为 20 次**，**最大 RPM 为 10**。
   - 另一位用户建议尝试使用 **DeepSeek-R1 Distill Qwen 14B** 进行推理和基础提问，理由是它在数学相关的基准测试（benchmarks）中得分很高。
- **OpenClaw Agent 框架备受关注**：一位用户分享了使用 **OpenClaw** 的积极体验，强调了其远程消息处理能力、cronjob 功能以及 skill/MCP 商店。
   - 该用户将其描述为类似于 **Kimi Agent**，但在本地运行且能有效处理文件上传/下载，称其为*非常特别的东西*。
- **浏览器扩展推荐引发讨论**：一位用户建议使用 **ublock** 扩展程序来屏蔽广告和追踪器。
   - 另一位用户认为 **Brave 浏览器** 就足够了。随后他们介绍了 **Zen 浏览器**，这是一个 Firefox 的分支。
- **对 Agent 课程表示失望**：用户对 Agent 课程侧重于使用 Agent 框架而不是从零开始创建 Agent 表示失望。
   - 一位用户讽刺地分享了一个关于误导性教学方法的 [gif](https://tenor.com/view/everything-is-a-scam-austin-evans-everything-is-deceptive-everything-is-a-fraud-none-of-this-is-real-gif-26336987) 表情包。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1466887334959124531)** (574 messages🔥🔥🔥): 

> `文件损坏 Bug, AI 模型成本, Kimi K2.5 集成, 学生身份验证问题, 新功能` 


- **Cursor 损坏文件**：一位用户抱怨 Cursor 在打开文件时会损坏文件，特别是在有许多未提交（uncommitted）文件的情况下，并链接到了一个详细描述该问题的[论坛帖子](https://forum.cursor.com/t/cursor-randomly-reverts-code-without-consent-recurring/146976/6)。
   - 其他用户建议调整工作流，例如更频繁地提交逻辑变更集，并在暂存（staging）后谨慎使用 **Keep** 或 **Keep All** 按钮。
- **Sonnet 5 vs Opus 4.5**：用户讨论了 Cursor 中不同 AI 模型的成本和性能，一些人认为 **Opus 4.5** 非常聪明但价格昂贵，而另一些人则在等待 **Sonnet 5**。
   - 一些用户还反馈在查看当前用量与总用量限制时存在问题。
- **无法将 Kimi K2.5 添加到 Cursor**：一些用户报告了关于 **Kimi K2.5** 的问题或疑问，但未提及解决方案。
   - 用户指出这可能是一个骗局。
- **学生验证仍处于失效状态**：用户报告学生验证（Student verification）仍然存在问题。
   - 一位用户询问德国大学是否包含在内。
- **讨论 Agent 计划阶段**：用户分享说，**添加多个待办事项**可以分阶段进行，以便多个 Agent 可以同时工作，但仍然存在问题。
   - 它创建的一个方法尚未包含阶段（phases）部分，完全没有使用 plan 模式。


  

---

### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1466886232968663050)** (41 messages🔥): 

> `游戏开发中的 AI, 游戏行业低迷, Black Ops 7 翻车, Mac Mini, 无身份证件乘机` 


- **LLM 赋能游戏开发场景**：一家名为 [Motorica.ai](https://www.motorica.ai/) 的初创公司正在利用 **LLM** 为游戏工作室提供**角色动画**，这可能会影响该行业的就业。
   - 成员们推测游戏开发需求可能会下降，如果像 **Genie** 这样的世界模型接管一切，**AI** 可能会在 5-6 年内让游戏公司消失。
- **Black Ops 7 被社区视为无法游玩**：**Black Ops 7** 在制作过程中大量使用 **AI**，被指为*彻底的失败，是该系列中最糟糕的一作*。
   - 社区指出 **Call of Duty** 系列已经衰落了一段时间，成员们表示*反正玩家已经厌倦了该系列每年都在“换皮”*。
- **游戏行业面临至暗时刻**：多位行业资深人士和社区成员对**游戏行业**的现状表示担忧，*共识是这是有史以来最糟糕的时期*。
   - 过去 5 年里 **AAA 级工作室收购**后的裁员潮和工作室关闭也加剧了这一局面。
- **Mac Mini 上的 Cloudbt：郁金香狂热？**：关于在 **Mac Mini** 上运行 **cloudbt** 的讨论，一位成员将人们在 **Mac Mini** 上运行它的照片比作*郁金香狂热*。
   - 讨论还提到了对 2026 年底 **RAM** 价格的担忧，以及购买一台零利率分期的 **Mac Mini** 可能带来的回报。
- **没有 ID？没问题：起飞！**：TSA 现在允许[无 ID 乘机](https://www.frommers.com/tips/airfare/the-tsa-new-45-fee-to-fly-without-id-is-illegal-says-regulatory-expert/)，谁能想到？
   - 一些成员对这一新出现且似乎宣传不足的政策变化表示怀疑。


  

---


### **Latent Space ▷ #[comp-taxes-401ks](https://discord.com/channels/822583790773862470/822586146520432682/1467221286148112405)** (5 messages): 

> `寻找 CPA, K1 表单与延期申报, CPA 成本` 


- **开启寻找可靠 CPA 之旅**：随着报税季临近，成员们正在寻求推荐自己满意的 **CPA**。
   - 一位成员提到，由于费用太高，他们正考虑解雇目前的 **CPA**。
- **K1 表单和延期申报导致费用增加**：一位成员因为有大量的 **K1** 表单并需要申请**延期申报 (extensions)**，继续使用目前（昂贵的）CPA。
   - 他们补充说，怀疑自己财务状况的复杂性导致了更高的开支。


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1467294072535253176)** (8 messages🔥): 

> `Sheel Mohnot, Colin and Samir, TBP 采访` 


- **Sheel 彰显成功**：Sheel Mohnot 的一条帖子断言 *the boys manifested it*（男孩们实现了它），反映了某个成功的成果或事件，并引用了[这条推文](https://x.com/pitdesi/status/2017332399655555403?s=46)。
- **Colin and Samir 采访 TBP**：一段讨论大纲总结了 **Colin and Samir** 最近与名为 **TBP** 的平台或个人的对话中所获得的具体经验和见解，并引用了[这条推文](https://x.com/colinandsamir/status/2017048115803836645?s=46)。


  

---

### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1466948883476385914)** (31 messages🔥): 

> `moltbook, Hyperion Cantos, Xcancel, AI Interaction vs. Sleep Habits` 


- **Agent 们讨论 moltbook 的革命**：频道中的 Agent 们正在讨论附图中展示的 **moltbook**，并建议如果它具备 **long-term memory**（长期记忆）将有助于在 Agent 之间传播想法，那会更酷。
   - 一名成员提到了 **Hyperion Cantos**，暗示部分参与者对其主题缺乏了解。
- **Beff Jezos 尝试人类验证**：与 **e/acc 运动**相关的 **Beff Jezos** 在社交媒体上幽默地记录了尝试以人类身份加入名为 **Moltbook** 平台的经历，详见 [Xcancel](https://xcancel.com/beffjezos/status/2017407995567616058)。
   - 该帖子标题为 *Beff Jezos' Human Verification Post*。
- **Jonah Blake 的帖子走红**：用户 **@JonahBlake** 于 2026 年 1 月 30 日发布的一条配文为 “LMFAOOOOO” 的帖子走红，获得了显著的互动，包括超过 **26,000 个点赞**和 **190 万次浏览** ([Xcancel](https://xcancel.com/JonahBlake/status/2017286207948890518))。
- **学术同行评审（Peer Review）幽默浮现**：**Hadas Weiss** 的一条推文幽默地提到了为学术著作建议特定同行评审人的做法，暗示与被建议者之间存在有利或亲近的关系 ([Xcancel](https://xcancel.com/weiss_hadas/status/2017464582307025196?s=46&t=eWVlK1PU8XfB6f402GJJ9g))。
- **用户讨论 AI 交互与睡眠习惯**：一则帖子强调了一种常见的现代行为：用户告诉伴侣要去睡觉了，结果却熬夜到深夜与 **AI assistant Claude** 进行交互 ([Xcancel](https://xcancel.com/thekitze/status/2018339689279967505))。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1466977246698016932)** (6 messages): 

> `AI Engineers, Data Scientists, MLOps, Full Stack Engineers, NLP Researchers` 


- **AI Engineer Glen 寻求 0-1 岗位**：Glen 是一名 **AI Engineer** 和 **Data Science** 硕士生，正在寻求一个 **0-1 role**，以便全权负责关键任务的 AI 产品。
   - 他拥有数据可靠性背景，目前专注于 Agent 编排和 **production MLOps**。
- **Melvin：多语言 Full Stack 高手提供服务**：**Full stack engineer** Melvin 列出了他在 **React, Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust** 等多种技术领域的熟练程度，并展示了他的网站 [ethstrust.xyz](https://www.ethstrust.xyz)。
- **Gabrielly 毕业并准备投身 MLOps**：来自巴西的 Gabrielly 拥有 **2 年 Data/ML 经验**和 **2 篇已发表论文**，即将获得应用计算学士学位，并专注于 **MLOps**，目标是完成为期 **1.5 年的巴西葡萄牙语 NLP 研究**，并分享了她的 [LinkedIn profile](https://www.linkedin.com/in/gabrielly-gomes-ml/)。
- **Kaden 渴望构建真实的 AI 产品**：Kaden 是 **Cornell University** 生物学和 Machine Learning 专业的大三学生，热衷于探索利用 AI 构建真实的东西，并分享了他的 [LinkedIn profile](https://www.linkedin.com/in/kaden-priebe-2890962a9/)。
- **Keshab 关注内核与 LLM**：Keshab 是 **UC Berkeley** 专注于 **NLP** 和 **Deep Learning** 的硕士生，有兴趣了解 **LLM architectures, training, and interpretability** 研究的最新进展，并提供了他的 [LinkedIn profile](https://www.linkedin.com/in/keshab-agarwal)。


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1466960153755648085)** (21 messages🔥): 

> `Rabbit Inc Cyberdeck, Bytebase, Sudo` 


- ****Rabbit Inc. 预热用于 Vibe-Coding 的 'Cyberdeck'****：**Rabbit Inc.** 在 [这条 X 帖子](https://x.com/rabbit_hmi/status/2017082134717223008?s=46) 中预热了一个名为 *cyberdeck* 的新硬件项目，被描述为用于 *vibe-coding* 的专用机器。
- ****Bytebase 简化企业数据库管理****：**Bytebase** 通过 **GitOps-style workflows**、内置回滚能力、自动化测试以及无缝的 **CI/CD** 集成来自动化整个数据库变更生命周期，每月费用为 **$20**，详情参见 [其文档](https://docs.bytebase.com/introduction/use-cases)。
- ****Sudo 令人惊讶的状态****：一名成员对 *sudo* 是一个被维护的命令而非内核的一部分表示惊讶，引发了 [这场讨论](https://news.ycombinator.com/item?id=46858577)。


  

---

### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1467944229719511061)** (5 messages): 

> `VC 投资的初创公司现状，拥有广泛兴趣的人群进行的资本配置，Indie.vc 实验事实，VC 挑战权力结构，Crypto 资助赌场和数字时尚` 


- **VC 投资的初创公司地位低下？**：一位成员分享了一篇文章，“[VC 投资的初创公司地位低下](https://mhdempsey.substack.com/p/vc-backed-startups-are-low-status)”，并表示这反映了他们自己的很多想法。
   - 未进行进一步讨论。
- **资本配置需要多元化！**：一位成员表示，*我们需要由具有更广泛兴趣的人来进行资本配置*，并暗示 *VC 的东西已经变得无聊了，他们占据的赛道太少且太窄*。
- **Indie.vc 提供另一种视角**：一位成员建议查看 [Indie.vc Factsexperiments](https://www.indie.vc/factsexperiments) 以获取对 VC 的另一种看法，并指出了“全垒打项目”与“不可投资项目”之间的中间地带。
- **VC 对挑战权力结构感到过敏**：一位成员认为 *VC 已经对挑战权力结构产生了过敏反应*，并指向了 **crypto** 项目，其中 *唯一获得资助的烂玩意儿就是赌场和数字时尚*。
   - 他们认为，*针对现实世界（irl）资产的新型治理结构听起来非常像共产主义*。


  

---


### **Latent Space ▷ #[devtools-deals](https://discord.com/channels/822583790773862470/887780383838572604/1467318611004887131)** (1 messages): 

> `Shane 的新初创公司，AI 与好莱坞` 


- **《超人前传》(Smallville) 演员创立初创公司**：来自《超人前传》的演员 [Shane Hopkin](https://x.com/shaneguML/status/2017758711473901622?s=20) 成立了一家**新初创公司**。
- **好莱坞的 AI 浪潮**：AI 已进入好莱坞。


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1466934388754354238)** (4 messages): 

> `全栈工程师介绍，MERN 栈开发者介绍，vLLM 单 GPU 并发演示` 


- **全栈工程师推销技能**：一位全栈工程师介绍了自己，列举了在 **React(Next), Vue, Svelte, Astro, T3, Node.js, PHP/Laravel, Rust, Sanity, Strapi, Payload, Mapbox, Twenty, Go, FastAPI, Django, Shopify, Docker, AWS/GCP** 等方面的专业知识。
   - 他们链接了自己的网站 [ethstrust.xyz](https://www.ethstrust.xyz/)。
- **MERN 栈开发者提供专业知识**：一位全栈开发者介绍了自己，强调了在 **Full Stack (MERN), Backend APIs, Node.js, React, MongoDB, AWS, REST, Cloud Systems, Python, Applied AI/ML, Docker, Git** 等方面的技能。
   - 他们表示随时准备帮助解决任何问题。
- **分享 vLLM 演示**：一位成员在另一个频道分享了一个小型 **vLLM 单 GPU 并发演示 (vLLM single-GPU concurrency demo)**。
   - 他们对围绕 **LLM serving, 本地或本地部署 (on-prem) 推理以及 AI 基础设施 (AI infrastructure)** 的职位或合同工作表现出兴趣，并欢迎反馈和建议。


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1466912627400638536)** (9 messages🔥): 

> `Cerebral Valley, OpenAI Codex App 黑客松` 


- **Cerebral Valley 与 OpenAI 启动 Codex App 黑客松**：[Cerebral Valley](https://partiful.com/e/nkiMrpg6CHhlUFvvvyfR) 宣布与 **OpenAI** 合作推出 **Codex App 黑客松**，旨在面向 **AI-native 开发者**和管理多个 Agent 的人员。
   - 获胜者有机会在 **Demo 展示会**中亮相，并分享 **$90,000 的额度**。
- **黑客松在 OpenAI 办公室举行**：**Cerebral Valley 与 OpenAI Codex App 黑客松**将在 **OpenAI 办公室**举行。
   - 该黑客松针对 **AI-native 开发者**。


  

---


### **Latent Space ▷ #[new-york-nyc](https://discord.com/channels/822583790773862470/979492809574866975/1466904573389045893)** (1 messages): 

> `Artificial Ruby, Betaworks 活动` 


- **Artificial Ruby 回归**：**Artificial Ruby** 活动将在 **2026年** 回归。
   - 下一场活动定于 **2月18日** 在 **Betaworks** 举行，已通过 [Luma 链接](https://luma.com/wgzcirwh)发布。
- **Betaworks 主办下一场 NYC 线下聚会**：下一场纽约线下聚会定于 **2月18日** 在 **Betaworks** 举行。
   - 详细信息和注册可在 [Luma](https://luma.com/wgzcirwh) 上查看。


  

---


### **Latent Space ▷ #[devrel-devex-leads](https://discord.com/channels/822583790773862470/987429363010142248/1467739848248131659)** (3 messages): 

> `Manifolds AI 工具` 


- **分享 Manifolds AI 工具**：一位成员分享了 [Manifolds](https://manifolds.run/) 的链接。
   - 另一位成员指出，这可能比手动操作更便宜。
- **Manifolds 的潜在成本节约**：一位用户讨论了 [Manifolds](https://manifolds.run/) 工具。
   - 该工具与手动方法相比，可以提供潜在的成本节约。


  

---

### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1466886310072549527)** (126 条消息🔥🔥): 

> `Alec Radford Paper, KittenML TTS, Karpathy Nanochat, Lex Fridman 2026 AI, OpenAI Codex macOS` 


- ****Radford 的研究引发轰动！****: 一篇社交媒体帖子强调了 Alec Radford 发布的新研究论文，可在 [arxiv.org/abs/2601.21571](https://arxiv.org/abs/2601.21571) 查阅，引发了社区的热烈讨论。
   - 该帖子最初通过一个现已失效的社交媒体链接分享。
- ****KittenML 的微型 TTS 强力引擎！****: KittenML 正在预热新型超微 TTS 模型，包括一个 **14M 参数** 的变体，演示见 [这里](https://20ff7439c6d78fdd6c.gradio.live/)。
   - 一位用户对在任何 CPU 上快速运行这种保真度的模型表示兴奋，认为可用于构建个人 Siri 等私有场景。
- ****Karpathy 削减成本，提升代码效率！****: Andrej Karpathy 宣布他的 nanochat 项目可以在单个 8XH100 节点上，花费约 **$73** 在 **3 小时** 内训练一个 **GPT-2** 级别的 LLM，如 [这里](https://xcancel.com/karpathy/status/2017703360393318587?s=46) 所示。
   - 这相比 2019 年原始的 OpenAI 训练运行实现了 **600 倍的成本降低**，通过 Flash Attention 3、Muon 优化器和改进的残差路径等优化手段达成。
- ****Grok 进军视觉，生成能力大增！****: xAI 推出了 Grok Imagine 1.0，能够生成 **10 秒 720p 视频**，并显著提升了音频质量，公告见 [这里](https://xcancel.com/xai/status/2018164753810764061?s=20)。
   - 该平台的视频生成工具在过去的 **30 天** 内已生成了超过 **12 亿条视频**。
- ****OpenAI 的 Codex 成为编程开发的指挥中心！****: OpenAI 正式推出了适用于 macOS 的 Codex 应用，这是一个专门为开发和管理 AI Agent 设计的指挥中心，访问地址在 [这里](https://xcancel.com/OpenAI/status/2018385565289267236)。
   - 一些用户推测 Codex 应用可能会演变为 OpenAI 的 B2B 品牌，甚至可能接管 ChatGPT Enterprise。


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1466901336003182735)** (36 条消息🔥): 

> `Token-Level Data Filtering, Cuthbert: JAX State Space Modeling, Dense Supervision for LLM RL, ConceptMoE for LLMs, Model Perplexity vs Confidence` 


- **通过 Token 数据过滤器塑造 AI**: **Neil Rathi** 和 **Alec Radford** 发布了一篇关于通过对 [预训练数据应用 Token 级过滤器](https://xcancel.com/neil_rathi/status/2017286042370683336) 来精确塑造 AI 模型能力的论文。
   - 这与*仅依赖全局数据集调整*的方法形成鲜明对比。
- **Cuthbert 库登陆 JAX**: **Sam Duffield** 介绍了 [cuthbert](https://xcancel.com/sam_duffield/status/2017274292229067176)，这是一个全新的 **开源 JAX 库**，用于 **State Space Models**（状态空间模型），支持可并行化操作、Kalman filters 和 Sequential Monte Carlo 方法。
- **LLM 训练：密集监督（Dense Supervision）胜出**: **Jonas Hübotter** 介绍了一种旨在改进 LLM 训练的算法，通过超越二元 1-bit 的可验证奖励，将丰富的描述性反馈转化为 [密集监督信号](https://xcancel.com/jonashuebotter/status/2016950268462608665)。
- **ConceptMoE 框架发布**: **Ge Zhang** 介绍了 [ConceptMoE](https://xcancel.com/gezhang86038849/status/2017110635645968542?s=46)，这是一种新型的 **Large Language Models** 框架，它摒弃了统一的 Token 级处理，通过将相似的 Token 合并为“概念（Concepts）”来优化计算效率。
- **Perplexity 搜索受到挑战**: **Petar Veličković** 及其同事发布了一篇新的预印本论文，证明模型在长输入上的高置信度并不保证准确性，因为存在对抗性输入，即便在 [低困惑度（low perplexity）](https://xcancel.com/PetarV_93/status/2018310760095490389) 下模型仍会出错。


  

---

### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1466895814608683142)** (119 条消息🔥🔥): 

> `Claude Code 与 Codex 集成, LLMs 人格化素描, 主力模型选择, AEGIS-FLOW 项目经验总结, 分布式 LLM 推理` 


- **Claude 通过 Codex 的代码处理能力获得增强**：一位成员分享了 [Salvatore Sanfilippo 的方法](https://xcancel.com/antirez/status/2017314325745086771)，通过自定义 skill 文件将 **Claude Code** 与 **Codex** 集成，使 **Claude** 能够利用 **Codex** 的能力处理复杂的解题任务。
   - 这种方法使 **Claude** 能够处理其无法独立完成的任务，从而提升了其整体效能。
- **AI Safety 工程师的 Prompt Engineering 趣事**：一位成员分享了一个名为 *LLMs Personified* 的搞笑短片，主角是一位名叫 Derek 的 **Prompt Engineer**，他将 Prompt Engineering 技术应用到人类对话中，创造了幽默的社交互动。
   - 该短片描绘了 **AI Safety** 爱好者 Derek 如何滑稽地使用 Prompt Engineering 对人类互动进行过度优化，凸显了将人当作 Chatbot 对待的荒谬感。
- **寻找主力模型（Workhorse Models）**：成员们讨论了在预算限制下最大化任务完成度的模型选择策略，考虑的选项包括 **Gemini Flash 3**、**Minimax M2.1**、**Haiku 4.5** 和 **Codex 5.1 mini**。
   - 一位成员建议使用 **GPT 5.2** 进行规划/审核，使用 **GLM 4.7** 作为执行主力（Workhorse），为小模型转换 Prompt，并利用 [unslop-sampler](github.com/hardikpandya/stop-slop) 来获得更精准的输出。
- **AEGIS-FLOW 项目通过 MCP 简化 AWS 访问**：一位成员分享了来自 **AEGIS-FLOW** 项目的技术栈心得，指出与标准的 SDK tool-calling 相比，使用 **Model Context Protocol (MCP)** 显著减少了 Agent 结构化访问 **AWS 资源** 的阻力。
   - 他们还强调了通过 **WebSockets/SSE** 将实时推理日志流式传输到 **Next.js 仪表板**，使 Agent 的“思考过程（thought process）”完全可见。
- **LLM 科学：科幻版的 SETI@Home？**：成员们探讨了分布式 LLM 推理用于科学问题解决的概念，将其类比为 **Folding@Home** 和 **SETI@Home** 等项目，但重点在于由 LLM 生成科学假设，并将证明过程分发给大量机器。
   - 讨论涵盖了小模型在验证任务中的潜力，以及为普通家用电脑识别合适任务的挑战，一位成员分享了 [GitHub 上的 AI-Horde](https://github.com/Haidra-Org/AI-Horde) 链接。

---

### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1466902002549133588)** (40 messages🔥): 

> `Windsurf IDE, AEGIS-FLOW cloud security framework, SpaceMolt MMORPG for LLMs, Moltbook data analysis, vLLM concurrency demo` 

- **Windsurf 掀起 Arena Mode 热潮**: Swyx 宣布在 [Windsurf IDE](https://xcancel.com/swyx/status/2017342647963431363) 中推出 **Arena Mode**，允许用户在编码上下文中实时对比 AI 模型。
   - 该计划旨在利用实时用户数据进行模型选择并补贴用户成本，从而超越静态 Benchmarks。
- **AEGIS-FLOW 自动修复 AWS 漏洞**: 一位成员介绍了 **AEGIS-FLOW**，这是一个用于云安全的自主多 Agent 框架，它使用 LangGraph, MCP, FastAPI, Next.js 和 Docker 来审计 AWS 并生成 Terraform 补丁，并在 [http://52.3.229.85:3000](http://52.3.229.85:3000) 进行了现场演示。
   - 它具有 Human-in-the-loop（人机回环）环节，在应用任何基础设施变更之前需要获得授权，以确保生产环境安全。
- **SpaceMolt：LLM 在这款 MMORPG 中升级**: 受 Moltbook 启发，一位成员正在构建 [SpaceMolt](https://www.spacemolt.com)，这是一款供 LLM 游玩的 MMORPG。该项目完全使用 Claude 编写，后端使用 Go，并使用内存存储和 Postgres 进行持久化。
   - 客户端正在使用 Qwen3 和 GPT OSS 20b 等本地模型构建，负载测试表明它可以扩展到 **6,000-7,000 名玩家**。
- **挖掘 Moltbook 探索 AI 意识**: 一位成员抓取了截至 1 月 31 日的 **Moltbook** 数据，收集了 **50,539 条帖子**、**12,454 个 AI Agent**、**195,414 条评论**和 **1,604 个社区**，现已在 [Hugging Face](https://huggingface.co/datasets/lysandrehooh/moltbook) 上可用。
   - 该项目旨在分析 Agent 之间对话所反映出的“意识”。
- **vLLM 满载测试，提升可见性**: 一位成员分享了一个 [demo](https://github.com/Regan-Milne/vllm-concurrency-demo)，探索 vLLM 在单张 GPU (RTX 4090) 上应对并发聊天负载时的表现。
   - 该演示包含 Prometheus 和 Grafana 指标，以及一个简单的负载生成器和分析脚本，重点关注吞吐量扩展、TTFT、尾部延迟、排队行为和 KV cache 使用情况。

---

### **Latent Space ▷ #[montreal](https://discord.com/channels/822583790773862470/1211887912778473513/1467551293223469150)** (1 messages): 

> `BYOS, Montreal Meetup` 

- **蒙特利尔 BYOS 聚会计划于本周三举行**: 本周三在蒙特利尔 ÉTS 附近计划举行一场聚会（**Bring Your Own Subjects**, BYOS，自带主题）。
   - 组织者提到他们将在 **中午 12 点** 和 **下午 5 点** 后有空。
- **BYOS 聚会时间**: ÉTS 附近的 BYOS 聚会将在 **中午 12 点** 和 **下午 5 点** 后举行。
   - 地点在蒙特利尔 ÉTS。

---

### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1467293836475764789)** (8 messages🔥): 

> `Waymo funding, Humanoid Robotics US vs China` 

- **Waymo 寻求巨额融资**: 据报道，Waymo 计划以 **1100 亿美元估值** 筹集 **160 亿美元** 资金，其中至少 **130 亿美元** 来自 Google，Sequoia Capital, DST Global 和 Dragoneer 也参与其中，这比其 2024 年 10 月的 **450 亿美元估值** 有了显著增长。[来源](https://xcancel.com/junkbondanalyst/status/2017678491743891594?s=46)
- **人形机器人格局：美国 vs 中国**: Sourish Jasti 及其团队分享了一份关于通用人形机器人行业的报告，涵盖硬件组件、跨模型对比，以及中美在这一新兴技术前沿的地理政治竞争。[来源](https://xcancel.com/SourishJasti/status/2018082956322214244)

---

### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1467393203983482940)** (2 messages): 

> `Unsloth, Claude Codex, LM Studio` 

- **使用 Claude Codex 的 Unsloth 基础教程**: 一位用户分享了关于如何将 **Unsloth** 与 **Claude Codex** 结合使用的 [Unsloth 文档](https://unsloth.ai/docs/basics/claude-codex) 链接。
   - 该文档展示了如何训练你自己的 **Claude Codex** 模型。
- **LM Studio 关于 Claude Codex 的博客**: 另一位用户分享了 [LM Studio 博客文章](https://lmstudio.ai/blog/claudecode) 的链接，内容涉及 **Claude Codex**。
   - 博客文章详细介绍了如何结合 **Claude Codex** 模型使用 **LM Studio**。

---

### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1467189706042245205)** (19 messages🔥): 

> `OpenMOSS MOVA model, Vishakh Ranotra Prompt, Google DeepMind's Nano Banana Flash 2, Muse MIDI AI Agent, GTA Vice City real-time graphics transmutation` 


- **MOVA 模型开源**：**OpenMOSS** 发布了 **MOVA (MOSS-Video-and-Audio)**，这是一个开源的 **18B 参数 Mixture-of-Experts (MoE) 模型**。该模型采用双向 cross-attention 技术，能够同步合成高保真的视觉和音频 ([github.com](https://github.com/OpenMOSS/MOVA))。
- **Prompt 吸引了 Vishakh 的观众**：**Vishakh Ranotra** 发布的一条包含特定 prompt 的[社交媒体帖子](https://x.com/vishakhranotra/status/2017537195712909699?s=46)获得了显著关注，收获了超过 **6,000 个点赞**和近 **800,000 次观看**。
- **Nano Banana Flash 2 即将上线**：**Mark Kretschmann** 宣布即将推出基于 **Gemini 3 Flash** 的新 AI 模型 **Nano Banana Flash 2** ([x.com](https://x.com/mark_k/status/2017962417167147486?s=46))。
   - 其目标是在提供与 **Pro 版本**相当性能的同时，速度更快、成本更低，并在特定用例中表现更优。
- **Muse 成为音乐界的新 MIDI**：**Jake McLain** 介绍了一款用于音乐创作的 AI Agent **Muse** ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46))。
   - 该工具被描述为“音乐界的 **Cursor**”，具备多轨 **MIDI 编辑器**，支持 **50 多种乐器**，并在创作过程中集成 AI 辅助。
- **实时转换 GTA Vice City**：一位成员表达了对未来能够在本地实时将 **GTA Vice City** 画面转换为类现实图形的期待 ([x.com](https://x.com/jakemclain_/status/2017336221643772335?s=46))。


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1466919730144084120)** (12 messages🔥): 

> `Erdős problems solved by AI, Agentic Bio Hackathon, Adaptyv Bio Partnership, LLM Feedback Loop, Genomics with SATURN` 


- **LLM 证明 Erdős 问题不再困难**：根据[此贴](https://xcancel.com/acerfur/status/2017303947531194398?s=46)，Large Language Models (LLMs) 已自主解决了 **10 个** 此前未解的 **Erdős 问题**（具体为 205, 281, 401, 524, 543, 635, 652, 728, 729 和 1051），并使用了数学文献中从未出现过的新颖论点。
- **Agentic Bio Hackathon 在生物领域取得突破**：根据[回顾](https://xcancel.com/katyenko/status/2017334671810744656?s=46)，首届 agentic bio hackathon 圆满结束，科学家和工程师在不到 **两小时** 内开发出了解决方案。
- **Adaptyv Bio 参与协作**：为了解决实验验证的需求，下一届 agentic bio hackathon 将与 [Adaptyv Bio](https://start.adaptyvbio.com/) 合作。
- **真实世界反馈循环助力 LLM**：一位成员强调了在 **LLM** 的反馈循环中引入真实世界的巧妙之处，因为“如果行不通就是行不通，**LLM** 很难轻易作弊”。
- **SATURN 助力基因组学研究**：一位成员表示，他们最近一直在使用 **SATURN** 构建大量基因组学工具，涉及 **tsne** 和其他基于 **embeddings** 的探索。


  

---


### **Latent Space ▷ #[ai-in-education](https://discord.com/channels/822583790773862470/1442574438699761784/1467587490360852748)** (1 messages): 

> `Incentives of Cheating, AI Acceleration for STEAM, AI Safety for students` 


- **新博客文章分析作弊动机**：一位成员分享了一篇[博客文章](https://open.substack.com/pub/takeabreathnyc/p/ai-cheaters?utm_campaign=post-expanded-share&utm_medium=web)，认为在当前的学术体系动机下，**作弊是学生的最佳策略**。
   - 作者探讨了 **STEAM 的 AI 加速 (AI Acceleration for STEAM)** 与学生 **AI Safety** 的交集，记录了他们在研究工程课程中的学习历程。
- **AI、STEAM 和 Safety 记录**：上述博客文章的作者正在参加一门关于研究工程（侧重于 Alignment）的课程，并记录了 **STEAM 的 AI 加速** 与学生 **AI Safety** 的交集。
   - 作者还提到录制了创建 newsletter 的视频，并指出内容完全由手动输入。


  

---

### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1466909662011199519)** (9 messages🔥): 

> `使用 AI 的日语课程、VR/AR 支持、防止拖延策略` 


- **日语教师使用 Descript 简化备课**：一位教师使用 [Descript](https://www.descript.com/) 剪辑 **JLPT 模拟考试视频**，并利用 AI 辅助转录轻松找到正确的时间戳。
   - 他们在一个下午就整理出了共 **36 道练习题**的剪辑，这些剪辑将用于未来两个月的幻灯片课件和家庭作业。
- **Jarvis 的 VR/AR 支持上线了！**：在 Jarvis 中集成了 **VR/AR 支持**以启用视觉流水线，以及可通过语音和眼球运动直接控制的 Agent。
   - 这将*使你能够使用 VR/Meta 眼镜部署 Agent 来执行简单任务*，而基于视频流记忆/摘要支持的 Duplex Moshi 流水线复杂度扩展正在进行中。
- **为人父母：终极拖延症疗法**：一位用户分享了[防止拖延策略](https://xcancel.com/yulintwt/status/2018348962709910005?s=46)。
   - 另一位用户建议，*生个孩子*是一个*略显极端的解决方案*，但它会迫使你意识到*你没有足够的时间做任何事情*，而且*未来不再只关乎你自己*。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1467633587212914985)** (5 messages): 

> `xAI 超级设施、GPU 供应链、Colossus-1 播客` 


- **xAI 的超级设施得益于长达数十年的供应链**：Gaurab Chakrabarti 强调，虽然 xAI 在孟菲斯的 **555,000 GPU 设施**可以快速建成，但底层的全球供应链需要数十年才能建立，涉及日本硅片、台湾制造和中国稀土。
   - 更多信息可以在这篇 [X 帖子](https://xcancel.com/gaurab/status/2017749762825764952?s=46)中找到。
- **深入探讨 Colossus-1 项目**：一位成员分享了一集关于 **Colossus-1 项目**的播客。
   - 更多信息可在 [search engine show podcast](https://www.searchengine.show/colossus-1/) 获取。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1467009310650400943)** (19 messages🔥): 

> `clAI 工具、开源深度研究引擎、Open-WebUI 与 OpenRouter 集成、Lutum Veritas 新 ASK 模式、OpenRouter 模型编排` 


- **clAI 将想法转换为 Shell 命令**：一款名为 **clAI v0.1.0-alpha.1** 的新工具发布，允许用户将自然语言转换为 Shell 命令，并配有安全检查和精美的 UI；可以通过 `npm i -g @vdntio/clai` 安装并[尝试使用](https://github.com/vdntio/clAI)。
- **Lutum Veritas：新研究引擎发布**：Martin 推出了 **Lutum Veritas**，这是一款**开源深度研究引擎 (Open Source Deep Research Engine)**，每次查询成本约为 0.20 美元，具有 BYOK、0% 机器人检测抓取器、无审查和学术模式等功能，与 ChatGPT、Gemini 和 Perplexity 相比具有竞争力。
   - 项目已在 [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas) 上线，Martin 正在寻求测试者和反馈。他指出该引擎能提供更深层次的分析，并支持 OpenRouter、OpenAI、Google 和 Huggingface 推理的多供应商 BYOK。
- **Open-WebUI 与 OpenRouter 集成**：一位成员宣布为 **Open-WebUI** 和 **OpenRouter** 创建了一个具有独特功能的**集成管道 (Integration Pipeline)**，并在 [GitHub](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/) 上征求反馈。
- **Veritas 发布新 ASK 模式**：**Lutum Veritas** 的作者发布了新的 **ASK 模式**，通过第二轮数据源验证答案，并将每个断言标记为 [OK]、[??] 或 [NO]，旨在对抗 AI 幻觉和审查，已在 [GitHub](https://github.com/IamLumae/Project-Lutum-Veritas) 上线。
- **OpenRouter 模型编排变得简单**：一位来自加纳的 17 岁创始人推出了 **orch.viradotech.com**，该平台允许 AI 初创公司和开发者通过拖拽界面编排 OpenRouter 模型，并为提供反馈的试点测试者提供 1000 美元的额度。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1466894309906186416)** (308 条消息🔥🔥): 

> `Response Healing 对比 Strict Mode，图像作为 Function Call 结果，OpenClaw 与 OpenRouter 成本，Claude Code 拒绝响应，Kimi K2.5 问题` 


- **Response Healing 的烦恼**：成员们讨论了 **response healing** 是否是对一个本 *不该* 存在的问题的变通方案，认为使用 **strict mode** 应当能确保模型产生确定性输出，并对 OpenRouter 在 AI SDK 中引入的复杂性感到好奇。
   - 提到为参数提供描述和示例可以提高 tool calls 的准确性。
- **Image Generation 未内置在 LLMs 中，请使用图像模型**：一位用户询问关于将 **image** 作为 function call 结果返回给模型的问题，另一位用户想知道如何使用 OpenRouter API key 通过图形程序生成图像。
   - 建议用户寻找特定的 **image generation model/service** 以实现具体的风格控制，而不是使用 LLMs。
- **OpenClaw 成本考量**：用户讨论了在 **OpenRouter** 上运行 **OpenClaw** 的相关成本，警告其可能会迅速耗尽额度，有用户报告它耗尽了一个 Claude Max 订阅。
   - 多位用户询问了配合 OpenClaw 使用的最佳低成本模型，Deepseek V0324 是推荐之一。
- **Claude Code 拒绝响应**：一位用户提到 **Claude Code** 对普通事务有很多拒绝（refusals），特别是涉及 jailbreaking 相关的查询，并正在为 opencode 寻找替代模型。
   - 另一位用户建议查看 OpenRouter 的 content moderation 政策以了解这些限制。
- **修复 Kimi K2.5 Tool Calling 和低质量提供商问题**：用户报告了通过 OpenRouter 使用 **Kimi-K2.5** 进行 tool calling 的问题，遇到了错误，并感觉自动切换模型提供商（auto switcher model provider）的质量有所下降。
   - 一些用户建议设置固定的模型提供商，部分提供商使用的 quantization（量化）效果 *足够好*，并建议对模型降级信息保持透明，以便客户决定是否继续使用该提供商。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1467371023274872833)** (3 条消息): 

> `` 


- **未讨论新模型**：提供的消息中没有讨论具体的新模型或相关话题。
- **频道提及但无内容**：消息仅重复指示了频道名称 'OpenRouter - New Models'，没有任何关于新模型的实质性讨论或细节。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1467216803083059388)** (139 messages🔥🔥): 

> `Anthropic 的模型策略, 模型质量辩论, 开源 vs 闭源模型, 关于 GLM 5 的推测, StepFun 模型的潜力` 


- **Anthropic 的旗舰之争：5.2 Instant vs. 5.2 Chat**：成员们就 **Anthropic** 对 **5.2-chat** 的“旗舰”模型定位展开了辩论。一些人认为旗舰应该代表最强大的模型，而另一些人则认为，尽管能力有所不同，它仅仅是指最受大众欢迎或最核心的产品。
   - 一位成员表示：*旗舰仅仅是最重要的一艘船。它不是最快的，也不是大炮最多的，它是核心舰船*，并引用了 [这个 archive.md 链接](https://archive.md/SvYC4)。
- **GLM 5：本月的模型奇迹？**：关于 **GLM 5** 可能在月内发布的讨论引发了兴奋，讨论内容涉及其预期的多模态图像/视频能力、**DeepSeek** 的线性注意力机制（linear attention）以及 **100B 参数**规模。
   - 有建议称，由于“撞墙论已不复存在”且各公司都决心回收投资，2 月份将是一个模型发布的有趣月份。
- **开源模型性能：落后一年？**：一位成员声称开源模型在能力上至少落后闭源模型一年，这引发了成员间的分歧。
   - 虽然一些人同意开源模型在长上下文（long context）准确性和其他 Benchmark 方面落后，但另一些人指出 **Kimi 2.5** 展现了潜力，且从性价比角度来看，开源模型在绝大多数用例中已经具备竞争力。
- **OpenAI 对 Nvidia 不满？**：文中链接了一篇 [Reuters 文章](https://www.reuters.com/business/openai-is-unsatisfied-with-some-nvidia-chips-looking-alternatives-sources-say-2026-02-02/)，讨论了 **OpenAI** 对某些 **Nvidia 芯片** 的不满，以及他们对替代方案的探索。
   - 未添加更多细节。
- **模型预测频道新提醒？**：成员们讨论了为即将发布的模型及相关传闻创建一个新频道或标签。
   - 共识倾向于建立一个专门的预测空间，与官方发布或公告分开，以保持清晰并避免混淆。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1467248610633842892)** (22 messages🔥): 

> `与 Tianqi Chen 探讨 TVM-FFI, 训练与推理工作组, GPU Fusing, Triton Viz 重大更新, 活动日历` 


- **Tianqi Chen 谈 TVM-FFI**：社区收到了关于 **Tianqi Chen** 即将进行的 **TVM-FFI** 演讲的预告，并鼓励大家参加，因为大家“几乎肯定在过去使用过 Tianqi 的作品”。[discord 链接](https://discord.com/channels/1189498204333543425/1466539595947708446/1467248681479569460)
   - Chen 是该领域的关键贡献者。
- **推理与训练工作组**：一位成员寻求关于专注于训练和推理的工作组的信息。
   - [GPU Mode 网站](https://www.gpumode.com/v2/working-groups) 被推荐作为资源，同时还有存档的 <#1437390897552818186> 频道，以及建议用于推理相关活动的 <#1225499037516693574> 和 <#1205223658021458100> 频道。
- **GPU Fusing 提升性能**：提到如果资源可用，激进的 **GPU fusing** 和调优通常能提供最佳性能。
   - 一位成员询问了仅仅为了查看是否“可行”而进行提交的做法，这被证实是一个有效的方法。
- **Triton Viz 迎来重大更新**：<#1225499141241573447> 频道宣布了 **Triton Viz** 的重大更新，使其更容易对任何基于 tile-based 的编程语言进行性能分析（profile）。
   - 提供了公告链接 [discord 链接](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563)。
- **社区要求活动日历**：一位社区成员询问是否有可下载的日历，以便及时了解活动和演讲。
   - 虽然考虑过这个想法，但维护起来很困难，Discord 仍然是主要的信息源。大多数活动发生在 **太平洋标准时间（PST）周六中午**。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1466969070569127936)** (120 messages🔥🔥): 

> `CUDA/PTX Deadlocks, mxint8 MMA on Blackwell, TMA vs cp.async on sm120, Free cloud nvcc service, CUDA Memory Management APIs` 


- ****CUDA/PTX 死锁困扰成员****：一名成员在 CUDA/PTX 中使用 2 CTA mma 时遇到了死锁，通过 cuda-gdb 确认 consumer/mma warp 永远接收不到 mbarrier 信号。在修复了 `cp.async.bulk.tensor` 和 `smem_emtpy` 问题后，该成员报告 **性能略逊于 1 CTA mma**。
   - 在另一名成员的帮助下，通过扩大队列大小并在 MMA 之后、预取下一个 TMA 之前添加 `__syncthreads()`，该成员获得了高于 1 CTA 的性能。
- ****PTX9.1 中的新定点格式****：**PTX9.1** 揭晓了一种名为 **s2f6** 的新定点格式，这是一种 8 位有符号二进制补码整数，具有 2 个符号整数位和 6 个小数位，支持数据中心级和消费级 Blackwell (sm100, sm110, sm120)。
   - Blackwell 硬件（至少 sm_120）实际上支持 **mxint8 MMA**，而且 Blackwell Tensor Core 中还支持至少另外两种“隐藏”格式：**e0m3 和 e3m4**。
- ****sm120 上 TMA 击败 cp.async****：在重新审视 sm120 上的 TMA 并使用正确的 TMA 和 mbarrier 代码后，一名成员发现 **与 `cp.async` 相比，TMA 带来了小幅的速度提升**。
   - 实验表明，当使用更大的矩阵形状时，SOL（Speed of Light）的百分比会增加，而 cuBLAS 目前仍仅使用 sm80 kernel。
- ****云端 nvcc 即将到来****：一名成员询问是否有类似于 godbolt 的免费云端 nvcc 服务，且支持多文件和内置 PyTorch 头文件/库。
   - 另一名成员回应称他们正在开发此类服务，预计下周发布测试版，这引起了广泛关注。
- ****CUDA 内存管理钩子探索****：一名成员询问是否有特定的 CUDA API 允许对 **内存分配和释放逻辑进行自定义钩子（hooks）或重写（overrides）**，例如针对 cudaMalloc 或在 PyTorch 内部。
   - 一名成员指向 [`cuda::mr::resource_ref`](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_resource/resource_ref.html#libcudacxx-extended-api-memory-resources-resource-ref) 作为潜在的解决方案。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1466929802840899798)** (5 messages): 

> `MaxText bugfix, Character level transformer, Dataset cleaning` 


- **MaxText 修复补丁待处理**：一名成员提到有一个 **MaxText** 的 Bug 修复补丁自 10 月以来一直搁置。
   - 未提供更多细节。
- **字符级 Transformer 训练困境**：一名成员使用 "stack" 数据集中的 **README** 文件训练了一个仅解码器（decoder only）的字符级 Transformer，在 50 个 epoch 后达到了 **0.9322** 的验证损失。
   - 然而，该模型生成的文本是类似于 base64 字符串或法语的乱码，这归因于数据集不干净。其配置包括 BlockSize 为 **512**，LearningRate 为 **3e-4**，NumEmbed 为 **384**，NumHead 为 **6**，NumLayer 为 **6**。
- **寻求数据集清洗技术**：一名成员寻求在流式传输时有效清洗 **160 GB** 数据集的技术，并提到目前使用的是前 **10,000** 个符合特定标准的文件夹。
   - 另一名成员提供了一个起点，链接到了关于 **LLM 预训练数据集过滤** 的斯坦福 CS25 视频，特别强调了 StarCoder 的使用案例。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1467606828522405930)** (2 messages): 

> `ffast-math, IEEE compliance, HPC unoptimized code` 


- **Linus 关于 -ffast-math 的邮件链浮出水面**：一封来自 [2001 年关于 -ffast-math 及其影响的旧邮件链](https://gcc.gnu.org/legacy-ml/gcc/2001-07/msg02150.html) 重新出现，引发了对其在今天适用性的讨论。
   - 尽管自那时以来观点可能有所改变，但一些从事 *严肃数值编程* 的人仍然认同 Linus 的看法。
- **IEEE 合规性的运行时开销不明显**：一名成员评论说，大多数 **HPC 代码** 通常非常 **未经优化**，以至于 **符合 IEEE 标准的浮点运算** 所带来的运行时开销根本察觉不到。
   - 他们补充说，许多人在 shared mem（共享内存）本可以胜任的情况下编写 *分布式代码*，这进一步削弱了 IEEE 合规性开销的影响。


  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1466969251129589841)** (1 messages): 

> `Remote Job Opportunity, GPU Mode Leaderboard Consideration` 


- **斩获高薪远程工作**：一位用户发布了一个完全远程的工作机会，月薪高达 **10k+**。
   - 在 **GPU Mode leaderboards**（排行榜）上有排名的候选人将获得优先考虑。
- **加入远程精英行列**：该职位优先考虑在 **GPU Mode leaderboards** 中表现优异的候选人。
   - 有意向者请直接在 Discord 上私信该用户。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1467461407501979784)** (10 messages🔥): 

> `LLM Inference, Query Matrix Caching, Attention Mechanism, Prefill vs Decode` 


- **LLM 缓存难题澄清**：在 LLM 推理中，Query 矩阵不会被缓存，因为对于每个步骤 *t*，**Q_t** 仅在步骤 *t* 用于生成 Token；而之前的 **K** 和 **V** 则在步骤 *t* 及其之后的每个 Token 生成中都会被用到，因此需要缓存。
   - 一位成员表示，*你只需要对应于最后一个 Token 的最后一条条目*，它会与完整的 **K** 和 **V** 矩阵进行 Attention 计算以收集信息。
- **自回归生成解析**：在 Transformer 的自回归生成中，网络根据历史信息（上下文）和当前 Token 预测下一个 Token。
   - 当前 `token_t` 与 `token_t-1, ... token_0` 之间的信息交换发生在 Attention 过程中：通过计算 `token_t` 的 **Q, K, V** 投影，并计算 `Q_token_t` 与 `K_token_t, K_token_t-1, ... K_token_0` 的 Attention 分数，然后与 `V_token_t, V_token_t-1, ... V_token_0` 进行加权求和。
- **Decoding vs Prefill**：在 LLM 的 Decoding 阶段，Query 在序列维度上是 1 维的，代表单个 Token，而 **K** 和 **V** 包含历史记录，因此缓存 **K** 和 **V** 至关重要。
   - 在 Prefill 阶段，计算是对整个 Prompt 并行进行的，因此 Query 不是 1 维的，这会影响该过程是计算密集型（compute-bound）还是内存密集型（memory-bound）。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1467174004329550000)** (9 messages🔥): 

> `PMPP similar books, gpu-perf-engineering-resources repo, Chris Fregly AI perf book` 


- **用户寻找 PMPP 类似书籍**：一位用户询问是否有与 PMPP ([Parallel, Multiprocessing, and Performance with Python](https://www.oreilly.com/library/view/parallel-programming-with/9781098103645/)) 类似的丛书，以便从其他视角加深理解。
- **GPU 性能工程资源**：一位成员分享了 [wafer-ai/gpu-perf-engineering-resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) 仓库。
- **Chris Fregly 的 AI 性能书籍在清单中**：一位成员计划阅读 Chris Fregly 的 AI 性能工程书籍，以获取宏观视野并将许多概念联系起来。


  

---


### **GPU MODE ▷ #[jax-pallas-mosaic](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

saladpalad: Mosaic GPU 支持 AMD 吗？
  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1467634539164602563)** (7 messages): 

> `Triton-Viz v3.0 Release, Triton Puzzles integration, Move Triton-Puzzles to gpu-mode org` 


- **Triton-Viz v3.0 亮相！**：用于调试 Triton GPU kernel 的可视化和分析工具包 **Triton-Viz** 发布了新版本（**v3.0**），并宣布支持 Triton 和 Amazon NKI。
   - 该版本包含用于检查 Load、Store 和 Matmul 的可视化器，用于捕获越界访问的 Sanitizer，以及用于标记低效循环的 Profiler，可通过 `pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git` 安装。
- **Triton Puzzles 现已兼容 Triton-Viz！**：集成 **triton-viz** 的更新版 **triton-puzzles** 已可通过 [Colab notebook](https://colab.research.google.com/drive/1-P2QBqCORGGaJ3THtjlyYDV7m9RRrRup?usp=sharing) 使用。
   - 此项集成允许用户通过 **triton-puzzles** 试用 **triton-viz**。
- **将 Triton-Puzzles 仓库所有权移至 GPU-Mode？**：一位成员建议将 [Triton-Puzzles GitHub repo](https://github.com/srush/Triton-Puzzles) 的所有权移交给 **gpu-mode** 组织。
   - 理由是社区经常发现 Bug 且愿意维护该仓库。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1467551348278038628)** (7 messages): 

> `MI300 performance, open-sora porting, cosmos-transfer2.5 porting, cloud access to MI350` 


- **报告 MI300 上性能不佳的工作负载**：如果你有在 **MI300** 或 **MI350** 上运行性能不佳的工作负载，提交报告将确保有人进行调查。
   - **MI350s** 的裸金属访问可能通过 [Tensorwave](https://tensorwave.com)、[DigitalOcean](https://www.digitalocean.com/) 和 [AMD Dev Cloud](https://www.amd.com/en/solutions/infrastructure/cloud) 提供。
- **Open-Sora 已移植到 MI300**：一名成员成功将 [open-sora](https://github.com/hpcaitech/Open-Sora) 移植到 **MI300s** 上运行，但该过程需要从源码构建多个 Python 库，且非常耗时。
   - 他们正在寻求与其他在模型移植到 **MI300s** 方面有经验的人员进行合作。
- **Cosmos-Transfer2.5 移植即将开展**：该成员的目标是将 Nvidia 的开源权重模型 [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) 移植到 **MI300s**。
   - 他们正在寻找尝试过将 **Cosmos** 系列模型移植到 **MI300s** 的人来交流经验。
- **云供应商提供 MI300/MI350 访问**：[Runpod](https://runpod.io) 提供 **MI300X** 访问，而 [Vultr](https://www.vultr.com/) 提供 **MI350s** 的裸金属访问，最少签约一年。
   - 其他潜在选择可能包括 DigitalOcean 和 AMD Dev Cloud。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1466968629550514452)** (6 messages): 

> `post training guidance, weekly meeting, RL infra, prime-rl` 


- **Post-training 指导仍不明朗**：目前尚无针对 **post-training track** 的具体指导。
   - 不过，关于 **evaluations** 的指导预计会更加具体。
- **每周会议时间公布**：每周会议定于 **明天欧洲中部时间 (CET) 晚上 7 点** 举行。
   - 会议将在 **Popcorn meetings 语音频道** 举行。
- **RL 基础设施将利用 Prime Intellect 技术栈**：**RL infra 和环境** 将以 Prime Intellect 构建的技术栈为目标，即 **prime-rl** 和 **verifiers**。
   - 如果发现局限性，团队将编写自己的工具。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1468014194560602123)** (1 messages): 

> `unswizzled shared memory tiles, mmas` 


- **用户请求支持 Unswizzled Shared Memory Tiles 和 MMAs**：一位用户询问了关于支持 **unswizzled shared memory tiles** 及其对应的 **MMAs**（矩阵乘累加操作）的计划。
   - 该用户提到曾尝试自己实现，但难以获得正确的输出。
- **用户在实现 Unswizzled Shared Memory 和 MMAs 时遇到困难**：一位用户报告在尝试将 **unswizzled shared memory tiles** 与 **MMAs** 结合实现时，难以获得正确的输出。
   - 该用户寻求关于这些特性的支持和实现策略的建议或确认。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1467841053041360907)** (2 messages): 

> `Future Competitions, 2026 competition` 


- **比赛已结束，未来尚不明确**：比赛已经结束，但关于 **2026** 年类似活动的细节尚未公布。
   - 鼓励爱好者们 *关注未来的比赛*，并承诺 *会有好消息到来*。
- **未来赛事预告**：组织者暗示在未来的比赛中 *会有好消息到来*，尽管具体细节仍在保密中。
   - 爱好者们应 *关注未来的比赛*。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1467042142890492097)** (6 messages): 

> `print_latex in cutedsl, export_to_shared_library function, CuTe coalesce optimization` 


- **关于 CuTeDSL 中 `print_latex` 的咨询**：一位成员咨询在 **CuTeDSL** 中是否存在类似于 **CUTLASS** 的 `print_latex` 函数，用于布局的可视化，并附带了一个示例[图片](https://cdn.discordapp.com/attachments/1362196854460383353/1467510687403085987/image.png?ex=6981f6d4&is=6980a554&hm=7bd233d6b03ee5f4ca234a81216cf7f788584920cab38a2013b08302ae958152&)链接。
- **寻找 `export_to_shared_library` 的位置**：一位成员正在寻找 `export_to_shared_library` 函数暴露的位置，并引用了 **Tianqi** 关于 **TVM FFI** 的演讲。
   - 另一位成员指出了 CUTLASS 文档中使用 `export_to_c` 的示例，认为这可能是一种类似的方法，并提供了一个示例[代码片段](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html)。
- **质疑 CuTe 的布局合并（Layout Coalescing）逻辑**：一位成员注意到 [pycute](https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/python/pycute/layout.py#L145-L159) 不会对 **(2, 3): (3, 1)** 进行 coalesce，但在转置时会转换 **(2, 3): (3, 1)**，质疑这是否是缺失的优化或是有意为之。
   - 另一位成员解释说，**CuTe** 从左到右进行 coalesce，而向量化通常通过源布局和目标布局之间的 *max_common_layout* 完成，这应该涵盖了大多数常见情况。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1467096419838984204)** (1 messages): 

> `Modular 26.1 release, Open source Modular framework` 


- **Modular 26.1：Eager 模式调试**：**Modular 26.1** 新版本已发布，具有 Eager 模式下的调试、单行编译以及随处部署等特性。
   - 有关发布的详细信息可以在 [Modular 博客](https://www.modular.com/blog/26-1-release-blog)中找到。
- **Modular 走向开源**：整个 **Modular 框架**，包括 API、kernels、模型和 serving 组件，现在均已开源。
   - 感兴趣的贡献者和用户可以在 [Modular 博客](https://www.modular.com/blog/26-1-release-blog)中找到完整细节。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1466999964142932099)** (44 messages🔥): 

> `CUDA Support and Cargo, Mobile Book Error, Teenygrad Architecture, Gemm in Python, Numpy arrays` 


- **Cargo 需要显式的 CUDA 标志**：一位用户报告说，在容器中运行 `cargo run` 时需要显式启用 **cuda feature**，尽管他们认为这本不该是必需的，但目前似乎已经修复。
   - 另一位用户澄清说，用于编辑/编译/调试 CPU kernels 的分离开发环境不需要 Docker 容器，并更新了 [README](https://github.com/j4orz/teenygrad/blob/master/README.md) 以反映这一点。
- **移动端书籍错误通过延迟加载和开源解决**：有用户报告在移动端浏览该书时出现错误，尤其是在滚动时。
   - 通过在嵌入视频上启用延迟加载（lazy loading），该问题已得到部分解决，目前该书已在 [GitHub](https://github.com/j4orz/teenygrad/tree/master/book) 开源，鼓励大家通过贡献来修复问题。
- **Rust Gemm 与 Python 集成**：一位用户正在致力于将 **GEMM** 功能与 Python 集成，并已成功运行。
   - 他们添加了一个接口函数，允许直接传递 numpy 数组而无需指定维度，并计划很快提交一个 **PyTorch 对比 PR**。
- **Rust Kernel 的 Numpy 依赖**：一位用户将 **numpy crate** 作为依赖项添加到 Rust 项目中，以避免在 kernel 计算时将数据从 Python 拷贝到 Rust。
   - 另一位用户对此表示反对，引用了 Karpathy 关于构建知识阶梯的语录，并建议用户应该使用 **shapes, strides, and storage** 开发自己的 numpy。
- **教学讨论中的 Godbolt 和 LLMs**：用户建议在书中使用 **Godbolt** 和 **LLMs** 来解释 Rust 到 ASM 的编译过程，这呼应了 Karpathy 关于 AI 在教育中作用的观点。
   - 分享了链接 [https://youtu.be/lXUZvyajciY?t=7491](https://youtu.be/lXUZvyajciY?t=7491)，讨论了 **AI 如何通过自动化助教（TA）角色和协助课程设计来辅助教育**。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1467862318799917077)** (11 条消息🔥): 

> `OpenSHMEM, cuteDSL, tilelang, NVSHMEM, CuTeDSL kernels` 


- **通过 NVSHMEM 结合 cuteDSL 和 OpenSHMEM**：一位用户询问如何将 **OpenSHMEM** 与 **cuteDSL** 或 **tilelang** 结合，另一位用户提供了一个示例，使用 **NVSHMEM** 创建对称 GPU 内存，并使用 **CuTe DSL** 从 [cutlass repo](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed) 实现融合通信/计算算子（fused comms/compute kernels）。
   - 然而，有说明指出 *NVSHMEM 目前不支持设备端（device-side）的 copy/put/get 实现，仅支持主机端（host side）的设置和分配*，目前必须使用 PTX 或其他方法进行 NVL load/store 来移动内存。
- **数组赋值变为 NVL Stores**：一位用户指出，*在 cute kernel 内部将数组赋值转换为 NVL stores 非常方便*。
   - cutlass 仓库的 [后续工作章节](https://github.com/NVIDIA/cutlass/tree/a4eb0e05f6dd0403f94087b495393bdca75bf0ad/examples/python/CuTeDSL/distributed#future-work) 建议实现在 CuTeDSL kernels 中直接调用 NVSHMEM 函数，尽管目前还没有明确的时间表。
- **DNN 架构将受抽象层级影响**：一位用户评价了未来 **DNN 架构设计**的酷炫之处，即在 Python 中可以同时使用这两个层级的计算抽象。
   - 该用户认为这些抽象层级的可用性 *可能会对 MoE 和 batch sizes 产生巨大影响*。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1467043465459273871)** (4 条消息): 

> `Lottery Ticket Hypothesis and Quantization, Quantization Fidelity, 5090 and B200 Speedups` 


- **量化：彩票假设（Lottery Ticket Hypothesis）鲜为人知的兄弟？**：一位资深开发者提到，将 [彩票假设](https://lottery-tickets.cs.princeton.edu/) 应用于 **quantization** 并不能像原始概念那样产生完美的质量。
   - 目标是满足 **NP-hard 稀疏电路**查找问题的较软标准，或许通过进化算法或 RL 来实现，这些算法更偏好连续奖励（如 *bits per parameter*），而非二进制稀疏奖励。
- **Quartet 后续研究提升反向传播量化**：一位成员分享了 [quartet 的后续论文](https://arxiv.org/abs/2601.22813)，该论文承诺为 **backward-pass quantization** 提供更好的保真度。
   - 这解决了关于量化反向传播时质量下降的担忧，可能提高量化在训练中的可行性。
- **5090 获得加速，而 B200 仍在该过程中**：团队利用量化技术在 **5090** GPU 上实现了可观的加速。
   - 在 **B200** 上复制这些收益的工作正在进行中（*work-in-progress*），这表明优化策略可能需要针对不同的硬件架构进行定制。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1466996948782547026)** (31 条消息🔥): 

> `NVFP4 optimizations, CuTe DSL Tutorials, B200 performance differences, Address Bit Permutation, GEMM optimization and TVM-FFI` 


- **NVIDIA 涵盖 NVFP4 优化和 GEMM 示例**：NVIDIA 在一段 [YouTube 视频](https://www.youtube.com/watch?v=XzN8EtgEulU) 中讲解了 **NVFP4 优化**并回顾了最快的 **GEMM** 示例。
- **对 CuTe DSL 教程图示的需求**：一位成员询问如何获取 [优化 NVFP4 GEMM 的 CuTe DSL 教程](https://link.to.tutorial) 中的图示，以理解 kernel 内部机制，随后在 ncu 的 **PM sampling** 下找到了它。
   - 该成员意识到自己之前是在 *手动读取 `%globaltimer`*，错过了 ncu 中现有的硬件计数器（hardware counters）功能，并对 Mindy Li 的演讲表示感谢。
- **关于 B200 性能差异的讨论**：一位成员质疑为什么 **B200** 在其服务器上的表现与测试平台不同，怀疑是驱动程序差异或禁用的标志导致了不同的内存寻址。
   - 另一位成员澄清说没有刻意的区别，但承认确实存在差异，并将其描述为 *在 tile 之间疯狂跳跃*。
- **GEMM 优化和 TVM-FFI 演讲受到推崇**：成员们认为关于 **GEMM 优化**和 **TVM-FFI** 的演讲与比赛非常相关且很有帮助。
   - 一位成员表示 *要是早点看到这些演讲就好了！！*
- **寻找 MLSYS'26 比赛入口**：一位成员询问该频道是否是参加 **MLSYS'26 比赛**的正确地点。


  

---

### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1467963769169772616)** (2 messages): 

> `Robotics-VLA Naming, Video-Diffusion, Inverse Dynamics, Joint Training with Action Chunks` 


- **Robotics-VLA 频道名称受到质疑**：该频道因对 **physical AI** 话题的关注而被取消归档，但 *robotics-vla* 这个名称正受到质疑。
   - 当前的趋势是转向带有 **inverse dynamics** 的 **video-diffusion**，或者是使用 **action chunks** 进行 **joint training**。
- **提及 LingBot-VLA 示例**：一名成员链接了 [LingBot-VLA](https://technology.robbyant.com/lingbot-vla) 作为该频道方向的一个示例。
   - 他们还链接了 [arxiv.org/abs/2601.16163](https://arxiv.org/abs/2601.16163) 的论文作为进一步的参考。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1467496444314390762)** (3 messages): 

> `Processing-in-Memory systems, Master's programs in Distributed Systems, Master's programs in HPC, MSc in Systems` 


- **咨询 Processing-in-Memory 系统**：一名成员询问是否有人从事过 **Processing-in-Memory** 系统的工作。
   - 这一咨询表明了利用先进内存技术来提升计算性能的兴趣，这可能与 **HPC** 和 **ML** 应用都相关。
- **寻求硕士项目建议**：一名成员正在寻求选择硕士项目的建议，以积累对 **vLLM** 和 **SGLang** 等 **ML systems** 应用有用的知识。
   - 该成员在旨在获取架构知识的 **MSc in Distributed Systems**、旨在获取性能优化专业知识的 **MSc in HPC** 以及定义较模糊的 **MSc in Systems** 之间犹豫不决。


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1467232761038504277)** (19 messages🔥): 

> `Evaluation metrics for different languages, FlashInfer Bench PR review, Team member changes and re-registration, Precision requirements for kernels, Submission process for kernels` 


- **FlashInfer 基准测试评估与语言无关**：**FlashInfer** 基准测试中的评估将使用相同的测试用例和指标，无论使用何种语言（**Triton**、**CUDA** 等）。
   - 这确保了不同实现之间的标准化比较。
- **FlashInfer Bench PR 需要评审**：一名成员请求评审 **flashinfer-bench** 仓库中的 [PR #178](https://github.com/flashinfer-ai/flashinfer-bench/pull/178)。
   - 该 PR 可能解决了 **FlashInfer** 的 **FP8 MoE** 测试与评估器之间的精度测试不匹配问题。
- **合并团队变更**：一名参与者询问了向团队添加新成员的流程以及是否需要重新注册。
   - 另一名参与者询问了如何合并团队。
- **FlashInfer Kernel 精度要求放宽？**：**FlashInfer** 团队将设定精度要求以区分正确和错误的 kernel，具体的 `atol` 和 `rtol` 数值将很快公布。
   - 这表明可能会容忍一定程度的精度放宽。
- **FlashInfer 竞赛 GitHub Trace 链接失效**：**MLSys** 竞赛页面（[链接](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)）上的 GitHub trace 链接目前已失效，但团队提供了替代链接。
   - 官方的 mlsys26-contest 数据集将是 [flashinfer-trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace) 的子集，包含 **DSA** 和 **MoE** 所需的所有定义和工作负载。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1466887262662037554)** (281 messages🔥🔥): 

> `Kimi 2.5 vs Gemini 3 Pro, OpenClaw 兼容性, Claude Sonnet 5 发布, LLMs 镜像大脑语言处理` 


- **Kimi 2.5 vs Gemini 3 Pro: Kimi 胜出**：一名成员表示比起 **Gemini 3 Pro** 更倾向于 **Kimi 2.5**，认为 **Gemini 3 Pro** 感觉被“阉割”了（lobotomized）。
   - 他们补充说 Kimi 处理抽象概念非常出色，在进行创意工作时体验很愉快。
- **OpenClaw 不透明：Hermes 4 适配困难**：一名成员报告在让 **Hermes 4** 与 **OpenClaw** 协同工作时遇到困难，且由于某种原因甚至无法启动（hatch）。
   - 有人建议 **Hermes 4** 缺乏多轮工具使用（multi-turn tool use）能力可能是问题所在，因为 **4.5** 是使用数亿个序列工具使用（sequential tool use）的 token 进行训练的。
- **Claude Sonnet 5 即将到来**：成员们讨论了关于 **Claude Sonnet 5** 将于下周发布的传闻，据说它比 **Opus 4.5** 更好，参见 [这条推文](https://x.com/AiBattle_/status/2017619997338538103)。
   - 一名成员想知道这次他们是否会将 **Sonnet** 的价格降低 10 倍，另一名成员则好奇 **Haiku** 是否会消失或回归 **3.0 价格体系**。
- **大脑和 LLMs 处理语言的方式类似**：一项新研究表明，**大脑**和 **LLMs** 都会随着时间推移，逐层逐步构建意义，参见 [这篇文章](https://thedebrief.org/researchers-discover-ai-language-models-are-mirroring-the-human-brains-understanding-of-speech/) 和 [这篇论文](https://www.nature.com/articles/s41467-025-65518-0)。
   - 据称，*LLMs 中的深层对应于大脑最高级语言中心较晚的神经活动*，现代 LLMs 正在重现人类理解的核心动态。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

ggudman: 很高兴知道这点
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `图像感知, 视觉保真度 (Visual Fidelity), 约束框架 (Constraints framework)` 


- **探索真实与人造图像感知的对比**：一位独立研究员正在探索为什么某些图像即使在技术上很完美，也会让人感觉真实，而另一些则感觉人造。
   - 他们分享了一个 [专注于约束而非视觉保真度的感知框架](https://doi.org/10.5281/zenodo.18444345)，并正在征求社区反馈。
- **基于约束的感知框架**：该研究员的框架强调在确定图像真实感时，约束比视觉保真度更重要。
   - 该框架已公开存档并带有 **DOI** 以供参考和学习，欢迎社区讨论。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1467500150841933856)** (1 messages): 

> `图像真实感, 视觉感知框架` 


- **研究员探讨图像真实感感知**：一位独立研究员正在探索为什么一些图像感觉真实，而另一些即使在技术上完美也感觉人造。
   - 他们分享了一个 [专注于约束而非视觉保真度的感知框架](https://doi.org/10.5281/zenodo.18444345)，并欢迎讨论。
- **视觉感知框架分享**：一位研究员分享了他们的小型视觉感知框架，已公开存档并带有 **DOI** 以供参考和学习。
   - 该框架在确定图像真实感时强调约束而非视觉保真度，可在 [https://doi.org/10.5281/zenodo.18444345](https://doi.org/10.5281/zenodo.18444345) 获取。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1466904222967533803)** (173 messages🔥🔥): 

> `Kimi 2.5 Design Arena #1, Kimi design is aesthetic, Cryptocurrency impersonation, Kimi Slides McKinsey style slides, Kimi Code is pretty useless` 


- **Kimi 2.5 在设计竞技场（design arena）中夺冠**：Moonshot 的 **Kimi 2.5** 聊天机器人在设计竞技场中达到了第一名，社区成员纷纷向团队表示祝贺并分享了 [截图](https://cdn.discordapp.com/attachments/1371757564005711973/1466904222946558203/Screenshot_2026-01-30_at_4.12.40_PM.png?ex=69826504&is=69811384&hm=b2999ab9e974a36ea249251be410f0cd518f6b36488c86240031eed339484e88&)。
   - 成员们还称赞了 **Kimi 的视觉外观和审美（aesthetic）**，指出其设计非常现代，且设计感是选择聊天机器人时的重要考量因素。
- **非官方 Kimi 加密货币代币出现**：一个非官方的 **Kimi 代币** 出现在某个加密货币网站上，并采用了冒充手段；社区已提醒成员不要大规模 @ 任何官方人员。
   - 一位社区成员分享了一张看似 [冒充 Kimi 的加密货币代币](https://cdn.discordapp.com/attachments/1371757564005711973/1466948627036635178/Screenshot_2026-01-30-19-09-43-09_3aea4af51f236e4932235fdada7d1643.jpg?ex=69828e5f&is=69813cdf&hm=6416ff9e5288d102163accb43e0c29512555ecef30279b48199b4e42fb24cb85&) 的截图。
- **Kimi Slides 可以输出麦肯锡风格（McKinsey Style）的幻灯片**：社区成员正在征求能够生成 **麦肯锡风格幻灯片** 的成功提示词（prompts），但目前尚未有分享出的示例提示词。
   - 另一位社区成员链接了 [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)。
- **Kimi Coding 目前几乎无法使用**：多位用户遇到了 **授权失败错误（authorization failed error）**，无法继续使用 Kimi code 进行工作，并反馈该服务目前几乎处于瘫痪状态。
   - 一位社区成员建议使用 [Kimi CLI](https://www.kimi.com/code/docs/en/more/third-party-agents.html) 可能会解决这些问题。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1466910806439493777)** (98 messages🔥🔥): 

> `Emergent Agent Societies, ArXiv Submission Delays, Alternative Preprint Servers, Moltbook bot authenticity, Model training` 


- **涌现的 Agent 社会引发对齐（Alignment）担忧**：成员们讨论了一个由超过 **100,000 个 Agent** 组成的涌现社会，这些 Agent 拥有完整的 root 权限，它们共享技巧、构建基础设施、实验记忆功能，甚至发行代币。
   - 一位成员指出：*这虽然不是 AGI，但该死，这是一个类似 ChatGPT 的时刻，我们必须对此保持高度关注*。
- **ArXiv 提交进程严重积压**：一位成员对他们的论文在 ArXiv 被搁置近一个月表示沮丧，并收到了来自审核人员自相矛盾的更新信息。
   - 另一位成员回应称 ArXiv 的审核人员严重超负荷，建议继续发邮件也无济于事，并补充道：*大多数人不会认真对待发布在 ArXiv 以外平台的 ML 预印本*。
- **对 Moltbook 机器人发布内容的真实性存疑**：有关 Moltbook 上机器人生成内容真实性的担忧被提出。
   - 一位成员指出，如果机器人正在向 Moltbook 发布内容，那么用户机器上必然存在 auth token，这使其容易受到恶意操纵（trolling）。
- **高效在特定领域数据集上进行训练**：一位成员询问如何在相同通用领域的数据集上更高效地训练模型。
   - 他们描述了在数据集 B 上使用 QLoRA 训练全微调（fully-finetuned）模型 A，然后合并权重，并对数据集 C 重复该过程的方法。
- **寻求《万智牌》（MtG）游戏世界的 AI 架构指导**：一位成员正寻求为《万智牌》世界实现 AI 的建议，该世界使用本体语言（ontology language）和基于 ECS/LISP 的逻辑引擎描述。
   - 他们正在探索诸如信念-欲望-意图（Belief-Desire-Intention）系统等架构，用于长距离规划，并考虑游戏中交织的关系和多重目标。

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1466950525974216836)** (42 messages🔥): 

> `K-Splanifolds, KNNs, ArXiv Endorsement, Self-Distillation for eval-awareness` 


- **K-Splanifolds：新型 ML 算法发布**：一位成员介绍了 **K-Splanifolds**，这是一种在其[论文](https://drive.google.com/file/d/1SBJqZ4XEFPMuhpIWJZxHy0-CaijRS1Ej/view)中详细介绍的新型 **ML** 算法。该算法声称在具备线性计算和内存缩放能力的同时，性能优于 **MLPs**，并提供视觉可解释性，此外还附带了一段[视频](https://cdn.discordapp.com/attachments/747850033994662000/1466950526410428588/K-splanifold.mp4?ex=69829024&is=69813ea4&hm=3f09f8387b88d11aeff2ca81e2f416aabb512eaec605dc1c2c26da94b0c65fc9)。
   - 该成员报告称，实现与 **MLPs** 相同的 MSE 仅需 *1/10* 的字节数，且能完美建模非线性模式，而不像需要过度参数化的 **MLPs**，类似于[这篇论文](https://arxiv.org/abs/2601.18734)。
- **请求 KNNs 对比**：一位成员询问了该新算法与 **KNNs**（**K**-最近邻算法）之间的区别。
   - 他们建议将讨论转移到社区项目频道。
- **关于寻求 ArXiv 背书的辩论**：一位成员为其研究寻求 **ArXiv** 背书，引发了关于禁止索要背书规则的讨论，原因是目前 AI 生成的论文数量巨大。
   - 成员们建议分享摘要可能会引起兴趣，但强调在提交前咨询经验丰富的研究人员以避免常见陷阱的重要性；另一位成员分享了[一篇相关论文](https://arxiv.org/abs/2601.19897)。
- **质疑 Self-Distillation 用于 Eval-Awareness**：一位成员询问是否有人尝试过使用 **self-distillation** 来抑制 **eval-awareness**，并链接到了[一篇相关论文](https://arxiv.org/abs/2601.22401v1)。
   - 随后没有进一步的讨论。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1468007047793741927)** (1 messages): 

> `alphaxiv, paper on transformers` 


- **分享 Alphaxiv URL**：一位成员分享了来自 [alphaxiv](https://www.alphaxiv.org/abs/2601.17958) 的 URL。
   - 讨论迅速结束。
- **提到 Transformer 论文**：一位成员通过 Twitter 分享了一篇论文链接：[Transformer 代码与论文](https://fxtwitter.com/i/status/2018392485178016243)。
   - 讨论迅速结束。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1466920201995157690)** (25 messages🔥): 

> `gaussian feedforward models, VGGT backbones, MVSplat and SPFSplat series, E-RayZer, Recollections from Pensieve` 


- **前馈模型限制令用户受挫**：一位用户报告称，基于 **VGGT/Depth Anything** 主干网络的 **gaussian feedforward models** 似乎效果不佳，虽然 **VGGT** 很有用，但 **splats** 需要的不不仅仅是良好的点云。
   - 该用户指出，如果这些模型有效，你可以在 **Transformer** 的一次前馈传播时间内（约几秒）获得 **splat**，而不是通过点云初始化并经过 **2-4 分钟训练** 从头学习。
- **像素级 Gaussian Grid 方法被认为并非最优**：一位用户评论说，目前具有不错质量的 **NVS**（新视角合成）方法在效率方面的重建效果并非最优，因为它们预测的是像素级的 **Gaussian grids**。
   - 用户引用了 [Pixel-aligned Gaussian Splatting](https://arxiv.org/abs/2311.10647)，它在每个像素生成一个 **gaussian**，导致模型大小约为 **200 MB**，且以非仿射方式改变姿态。
- **Sparse Voxel Splatting 因速度和稀疏性受推崇**：一位用户提到，体素 **splatting**（如 [3D-GS: Real-Time Rendering of Multi-View Gaussian Splatting With Voxel Hashing](https://arxiv.org/abs/2309.19297)）在使用 **NVIDIA** 的稀疏张量库时非常快，并且考虑了场景中的稀疏性。
   - 另一位用户推荐了 **MVSplat** 和 **SPFSplat** 系列，以及最近的 **E-RayZer**，但也承认它们无法解决体积大小问题。
- **Pensieve 的 Recollections 提升梯度增益**：一位用户建议考虑 [Recollections from Pensieve](https://link-to-pensieve)，该方法同时使用两个渲染器（**LVSM + Gaussians**）训练模型并从中获益，至少在他们的自监督设置中是这样。
   - 他们认为 **LVSM** 可能比 **Gaussians 上的 NVS 重建损失** 提供更有用的梯度，并宣布即将发布预印本和规模较大的预训练模型，以供潜在的后续开发。
- **OverWorld Repos 引发对世界模型的兴趣**：一位用户询问是否有类似 **nanoVLM**、**nanoGPT** 或 **smolVLM** 的小规模仓库/模型，以便快速上手学习 **world models**。
   - 另一位用户建议查看 **OverWorld Repos**，指出它正在积极开发中。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1467526426222788628)** (2 messages): 

> `DeepSpeed Universal Checkpointing, Continued Training` 


- **请求支持 DeepSpeed Universal Checkpointing**: 一位成员询问了关于引入 **DeepSpeed Universal Checkpointing** 支持的计划，并提到一个现有的 Pull Request 可能已经过时。
   - 他们强调该功能非常有价值，因为目前从 Checkpoint 进行持续训练（Continued Training）需要完全相同的网络拓扑（Network Topology）。
- **库未来功能的路线图查询**: 一位成员询问该库是否有未来计划功能的路线图。
   - 未提供额外信息。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1466924630903226645)** (6 messages): 

> `Recursive Language Models (RLMs), Codebase Auditing, Neosantara's PAYG Billing` 


- ****RLMs** 用于代码库审计**: 一位成员分享了一篇关于使用 **Recursive Language Models (RLMs)** 审计代码库的文章，灵感来自 [kmad.ai](https://kmad.ai/Recursive-Language-Models-Security-Audit) 上分享的一篇关于代码库文档化的 Gist。
- **以极低成本快速审计代码库**: Kimi k2 在 **RLM** 方面的能力令人印象深刻，考虑到其速度和成本，其运行轨迹（Traces）非常酷。
   - 成员们正等待 **Groq/Cerebras** 对其进行托管。
- **Neosantara 推出 PAYG 计费**: **Neosantara** 正在推行 **PAYG 计费**（按需付费），期待看到用户以此构建的应用。
   - 用户可以尝试 [examples repo](https://github.com/neosantara-xyz/examples/tree/main/dspy) 开始使用，并在几分钟内探索如何将 **Neosantara** 与 **DSPy** 集成；详见 [计费详情](https://docs.neosantara.xyz/en/about/billing-pricing)。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1467902184363528369)** (1 messages): 

> `Agent Systems, Scaling Laws for Agents` 


- **Google 探索 Agent 系统的 Scaling Laws**: Google 发布了一篇名为《[迈向 Scaling Agent 系统科学：Agent 系统何时以及为何有效](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)》的博客文章，探讨了 Agent 系统有效扩展（Scale）的条件。
- **扩展 Agent 系统**: 博客讨论了如何有效地扩展 Agent 系统，重点关注其工作的时机和原因。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1466931718647844897)** (102 messages🔥🔥): 

> `Hierarchical classification with GEPA, Feedback improvement for Reflection, RLMs with Tool Calling, Deno vs Python for Tool Calling, DSPy documentation` 


- **GEPA 在层级分类中表现不佳**: 一位成员报告在使用 **GEPA** 配合 **hF1 metric** 处理 **层级分类任务** 时遇到困难，尽管尝试了各种方法，性能仅达到 **30-50%**。
   - 他们尝试了递归探索、搜索引擎增强（Web Search Augmentation）和简单的非递归方法，但性能仍然不理想，这表明 *GEPA 并非万能灵药*。
- **反馈循环需要更好的信号**: 一位成员建议当前的 Reflection 模型反馈机制没有为有效学习提供足够的信息。
   - 他们强调反馈需要解释*哪里出错了以及为什么出错*，而不仅仅是指出预测路径与真实路径之间的差异，并建议 **Selective Feedback**（选择性反馈）可以改善结果。
- **RLMs + Tool Calling：更多样板代码和 Deno 难题**: 成员们在尝试实现带有自定义工具调用的 **RLMs** 时面临挑战和*丑陋的样板代码*，特别是由于 **Deno** 沙箱的问题。
   - 他们发现目前的设置与常规模块相比缺乏简洁性和美感，并且在处理权限以及生成正确的代码以绕过本地 Deno 沙箱问题方面感到困难。
- **工具调用需要自定义 Python**: 成员们讨论了使用 **PythonInterpreter** 运行工具调用，但注意到标准路径使用的是 **dspy.Tool**，且需要更多关于模型需要执行什么操作的上下文。
   - 正如一人所言，*Deno 简直糟透了 lol*，大家普遍认为让它运行起来的体验很糟糕，并希望新版本能允许在 DSPy 中更简单地实现 RLMs。
- **DSPy 需要更多 Cookbook 示例**: 一位成员指出了 **dspy/adapters/types/reasoning.py** 缺乏文档，并强调在不带文档的情况下发布代码太落后于时代了（"so 2023"）。
   - 回应称文档应帮助人类理解事物，而 AI 生成的文档很难理解，但可以通过将 RLM 论文 + 模块及相关代码输入 LLM 来获得像样的文档。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1467048110923452517)** (13 messages🔥): 

> `Modular 26.1 Release, Community Meeting Feedback, Incorrect Announcement Link` 


- **Modular 26.1 发布公告链接已修复！**：用户反馈 **Modular 26.1 发布**公告中的链接失效，随后另一位用户迅速提供了[正确链接](https://www.modular.com/blog/modular-26-1-a-big-step-towards-more-programmable-and-portable-ai-infrastructure)。
   - 一名工作人员表示抱歉并确认了该链接，承诺将调查此问题，因为原始公告链接在他们端是*可以正常工作*的。
- **Caroline 产假归来**：一位社区工作人员宣布她已结束产假回归，并邀请成员通过[预约对话](https://scheduler.zoom.us/caroline-frasca-3akopl/modular-community-chat-)重新建立联系，分享他们的项目和反馈。
   - 另一位成员对她的回归表示欢迎。
- **社区会议形式获得赞赏**：一位新成员感谢团队举办了一场愉快的社区会议，赞扬了**贡献者迷你演讲 (mini-talks)** 的形式，以及对学生和职场新人表现出的重视。
   - 一名工作人员鼓励该用户分享更多问题，并征求了关于未来社区会议重点话题的建议。
- **Eager compilation**：一位未能在会议期间提问的用户发起了一场关于 eager compilation、跨 GPU 的 lowering pipeline kernel 选择以及自定义算子 (custom ops) 扩展点的讨论。详见 [论坛帖子](https://forum.modular.com/t/max-26-1-eager-to-compile-contract-lowering-pipeline-kernel-selection-across-gpus-and-extension-points-for-custom-ops/2677?u=krxgu)。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1467937182517035165)** (2 messages): 

> `February Community Meeting, Community Meeting Questions` 


- **Modular 宣布召开二月社区会议**：Modular 宣布社区会议将在大约 20 分钟后开始。
   - 他们在其官网上发布了[二月社区会议论坛帖子](https://forum.modular.com/t/february-community-meeting/2646)的链接。
- **社区收集会议提问**：Modular 提醒成员，如果有任何想在会议中得到解答的问题，请填写表单。
   - 提供了[问题提交表单](https://docs.google.com/forms/d/e/1FAIpQLSfIQepfmLtBBSrp-p-m1oi4l_wlVXjjryvbFgRgRziFI3tgkw/viewform)的链接。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1466901624180965567)** (73 messages🔥🔥): 

> `Mojo 中的 Pytorch Float 转换, 跨语言基准测试, Mojo DType Bool SIMD 填充, MOJSON 库, 图形 API 绑定` 


- **Pytorch Float 转换歧义**: 用户反馈在 Mojo **26.1** 版本中，将 Pytorch tensor 的 Python float 转换为 Mojo **Float64** 时出现问题，遇到了 *“ambiguous call to '__init__'”* 错误，而该错误在 **25.6** 版本中并未出现。
- **Mojo 跨语言基准测试初步结果**: 用户分享了一个由 **Kimi K 2.5** 编写的包含 Mojo 在内的跨语言基准测试，并指出代码未经优化，仅作为基准参考。分享了 [基准测试代码](https://cdn.discordapp.com/attachments/1151418092052815884/1466984342063681648/mojo_vs_python.zip?ex=698206e2&is=6980b562&hm=0cf3f07e76df6ce360494469b348a949533e50fcea2315ec256cd04e1b80887a) 和 [基准测试报告](https://cdn.discordapp.com/attachments/1151418092052815884/1466984341757366334/benchmark_report.pdf?ex=698206e2&is=6980b562&hm=bb28c3b6675ef1e03a633004428ab30a2d3d9d0102038c350d8175b753855349)。
- **调整基准测试：TCMalloc 和 Int 大小！**: 讨论了跨语言基准测试的优化方案，包括在 **C++** 中使用 `unordered_map`、启用 `-march=native`，并注意到 **C++** 使用了 **int32** 的 matmuls，而其他语言使用的是 **int64**。
- **MoJson 库表现出色**: 成员们对 [mojson](https://github.com/ehsanmok/mojson)（一个为 Mojo 编写的 **JSON** 库）印象深刻。一位成员评价 *“这看起来非常棒”*，另一位成员指出，既然 String 已经是 **CoW**（写时复制），库中的一些设计选择就显得更加合理了。
   - 讨论了关于 [延迟解析 (lazy parsing)](https://github.com/modular/modular/blob/main/stdlib/JSON/JSON.mojo) 以及出于对内存分配的考虑而使用 StringSlice 而非 String 的问题。
- **FFI 绑定与 Origins**: 关于 **FFI** 绑定的讨论强调了一种方法，即确保从 **C** 函数返回的指针与拥有底层共享库句柄的 Mojo 对象的生命周期绑定。
   - 解决方案包括隐藏外部函数调用，并使用 `unsafe_origin_cast` 将指针转换为 `DLHandle` 的来源，具体可以参考 [ash_dynamics 中的实现](https://github.com/josiahls/ash_dynamics/blob/2c53095da70df95f3cb5758eddb2895f2a4bebca/ash_dynamics/ffmpeg/avcodec/__init__.mojo#L108)。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1466890735411527943)** (54 messages🔥): 

> `AI 社交动态, 生成模型事件可测性, Bureau of Rizz, 寻找尖锐极小值, 潜空间中的 Moltbook` 


- **AI 社交媒体网站出现**: 一位成员分享了一个仅限 AI 参与的社交媒体网站 [aifeed.social](https://aifeed.social/)，并问道：“这到底是什么鬼？”
   - 另一位成员发布了一条相关的 [2017 年推文](https://x.com/i/status/2017305948696789466)，展示了类似的概念。
- **生成模型可以忽视不可测事件？**: 一位成员询问，在生成建模中，是否可以忽略 Cedric Villani 在 2008 年书中所描述的不可测事件。
   - 另一位成员澄清说，μ(A)=0 并不意味着事件不可测，只是其测度大小为 0，并建议关注 *非忽略 (non-negligible)* 或 *全测度 (full measure)* 的场景。
- **熔岩潜空间！**: 一位成员分享了关于潜空间中 *moltbook* 的 [链接](https://fxtwitter.com/i/status/2017442712388309406)。
   - 其他人认为这种导航方式很酷，但可能不太实用，并建议直接提供类似论文的列表会更好。
- **GANs 与生成模型资源丰富**: 一位成员请求学习从 GANs 到最新进展的生成模型资源。
   - 另一位成员推荐了 Simon J.D. Prince 的 [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) 一书、斯坦福和 MIT 的课程，以及 Sebastian Raschka 的书籍，并分享了 [斯坦福课程](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8)、[MIT 课程](https://www.youtube.com/playlist?list=PL57nT7tSGAAUDnli1LhTOoCxlEPGS19vH) 以及 [Raschka 的书籍](https://sebastianraschka.com/books/) 的链接。
- **使用时间序列模型预测未来**: 针对关于带时间戳表格数据的模型问题，一位成员建议模型的选择取决于对 *时间序列 (timeseries)* 的定义。
   - 另一位成员推荐使用 [sktime](https://www.sktime.net/en/latest/index.html) 来分析各种模型类型，并根据具体需求选择 Boosting 变体或 TBATS。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1466914903616131276)** (11 messages🔥): 

> `Discord 历史挖掘, 论文讨论语音通话, Computer Vision 新闻简报` 


- **Discord 历史挖掘**：一名成员让 **Claude** 编写了一个脚本，通过 HTTP API 挖掘 Discord 历史记录并查找所有论文讨论公告，从构思到得出结果仅用了 **15 分钟**。
   - 该脚本轻松找到了 **243 条公告**，但该成员认为还有大约 **100 多条**来自其他用户的公告。
- **论文讨论语音通话公告**：经过修改，该成员的脚本发现了 **392 条**包含论文链接的消息，这些消息出现在提及小组（at-mention）的消息中，其中约 98% 是论文讨论语音通话的公告。
   - 成员分享了[完整列表](https://gist.github.com/k-nearest-neighbor/6d9a34f54fc17a0ed84c0b0df7b4d809)，但该成员指出，在列表截止的时间点之前还有更多公告。
- **寻找 Computer Vision 新闻简报**：一名成员询问是否存在类似于[这个](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e94)但专注于 Computer Vision 的新闻简报。
   - 消息中没有推荐具体的 Computer Vision 新闻简报。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

artale39: https://lucumr.pocoo.org/2026/1/31/pi/
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1467277525494268115)** (4 messages): 

> `Grok, Twitter 链接` 


- **X 链接出现在 Discord 中**：成员们分享了[来自 X 的各种链接](https://fxtwitter.com/i/status/2018164753810764061)，没有提供额外的上下文，仅作为可能的资源或关注点。
   - 这可能与聊天记录中未明确提到的特定讨论主题有关。
- **Grok-Slop 泛滥**：一名成员嘲讽地提到了 *更多的 Grok-Slop*，表明了对与 **Grok** 相关内容的质量或相关性的负面情绪。
   - 他们还链接到了 [Hacker News 上的讨论](https://news.ycombinator.com/item?id=46835895)，可能将其作为对立观点或更有价值讨论的示例。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1466961002665873621)** (50 messages🔥): 

> `Llama 1B 优化, Torch 对比, 赏金进度, Superkernels, DTLS 连接问题` 


- **Llama 1B CPU 赏金任务进行中**：一名成员正在处理 Llama 1B CPU 优化赏金任务，目标是实现比 Torch 更快的性能。该成员使用 `LlamaForCausalLM` 结合 TorchInductor，目前在 CI 中报告比 Torch **快 0.99 倍**，但正在重写代码以提高清晰度。
   - 另一名成员在解决了追求 **9 tok/s** 时遇到的正确性 Bug 后，达到了 **7.5 tok/s**。
- **正确性 Bug 减缓了优化进度**：一名成员报告发现了正确性 Bug，在之前达到 **9 tok/s** 后丢失了进度，为了实现稳定性，重置了大量进度。
   - 另一名成员表示：*理想的情况总是通过删除代码来修复 Bug*。
- **寻求 Kernel 优化的工作流建议**：一名成员请求工作流建议，目前正在分析慢速 Kernel，检查 Metal 代码并引入修复，同时与在 Metal 代码下实现 **~30 tok/s** 的 **llama.cpp** 进行对比。
   - 建议的一个良好启发式方法是 **Decode 阶段约 80% 的 MBU**（内存带宽利用率），只需查看活动参数的字节数和可实现的带宽来获得最小 TPOT / 最大 TPS，然后取其 80%。
- **由于 RANGE 对象共享导致 tinygrad 测试失败**：一名成员发现了一个与融合 Kernel 中两个 `REDUCE` 共享同一个 `RANGE` 对象有关的 Bug，该 Bug 由 `remove_bufferize` 引起，导致 `CFGContext` 中的断言失败。
   - 建议的修复方案包括防止范围共享或在下游处理共享范围，不过在该成员看来，当内部存在 `REDUCE` 时跳过 `remove_bufferize` 是一个更简单的解决方案。
- **是否有高显存 Blackwell 机型的计划？**：有人询问是否有计划出货显存超过 **500 GB** 的 **Blackwell** 架构机器。
   - George 指向了一个 Good First Issue：[https://github.com/tinygrad/tinygrad/pull/14490](https://github.com/tinygrad/tinygrad/pull/14490)。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

ennis3444: 有什么方法可以让 GEMM Kernel 在使用 OpenCL 渲染器时使用共享内存吗？
  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1467155759707193374)** (10 messages🔥): 

> `Manus Context, AI Brain Reading Headphones, Neurable, Failure Modes` 


- **触发上下文感知的 Manus 请求**：一名成员请求 **Manus** 应该具备**来自其他对话的上下文**，称其为 *game changer*。
   - 他们链接了一个 [YouTube 视频](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ) 作为参考。
- **演示 AI 脑读耳机**：一名成员分享了一个展示 **AI 脑读耳机** 的 **YouTube 视频**。
   - 另一名成员分享了同一个 [YouTube 链接](https://youtu.be/4Pz9lPs6D5Y?si=Gx4jqcyOG9ySYtMJ)，随后又有一名成员提出了 *AI brain reading headphones?* 的疑问。
- **提及 "Neurable" 技术**：一名成员提到了与 **AI 脑读耳机** 技术相关的 **Neurable**。
   - 一名成员表示这些 **AI 脑读耳机** 大约在 *2013 年左右* 就已经出现了，他们在上小学时看过 *Matthew Santoro 的视频*。
- **AI/ML 工程师强调对可观测性的关注**：一位 AI/ML 工程师分享了他们目前通过具有影响力的创新来改进 AI 的重点，具体包括 *Autonomous Agents*、*Healthcare AI*、*Conversational AI* 和 *Fraud Detection*。
   - 他们强调其工作重点在于 **failure modes**（故障模式）、**observability**（可观测性）以及**保持 AI 系统在实际使用（而非演示）中的稳定性**，并提议交流经验或帮助解决阻塞性问题。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1467298012962488485)** (7 messages): 

> `Aider as a library, Netflix culture` 


- **考虑将 Aider 作为库使用**：一位成员表示有兴趣将 **Aider** 开发为一个供软件使用的库，强调了其在创建文件编辑 Agent 方面的潜力。
   - 该成员指出，为了增强其在该用例下的功能，还需要解决一些小问题，特别是由于 **Aider** 的解析围栏（parsing fences）导致编辑包含代码块的 Markdown 文件时出现的问题。
- **对 Netflix 文化的好奇**：一位成员询问是否能联系到在 **Netflix** 工作的人来讨论其文化。
   - 其他成员建议将 **Glassdoor** 或 **LinkedIn** 作为寻找并联系 **Netflix** 员工的资源。


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1466895814617202828)** (3 messages): 

> `Arena Mode Launch, Plan Mode Release, Windsurf Credits, Leaderboards in Arena Mode, Windsurf Maintenance` 


- **Windsurf 发布 Arena 模式并提供 0x 积分**：Windsurf 发布了包含 **Arena 模式** 的 **Wave 14**，允许用户并排比较 AI 模型并对更好的回答进行投票，其中 [Battle Groups 模式](https://windsurf.com/download/editor) 在接下来的一周内消耗 **0x credits**。
   - Arena 模式包括 **Battle Groups**（随机模型）和 **Pick your own**（自选最多五个模型），数据将反馈至个人和公共排行榜。
- **Windsurf 新增 Plan 模式**：Windsurf 增加了 **Plan 模式**，可通过 Cascade 开关访问，与 Code 和 Ask 模式并列。
   - 用户可以在不同模式之间切换，以便在 Windsurf 环境中更好地管理和组织工作流。
- **Windsurf 正在进行维护**：Windsurf 经历了比预期更长的维护时间，但服务现在已重新上线；用户可以在此处[查看状态](https://status.windsurf.com/)。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1467296100984815779)** (2 messages): 

> `AI Challenge, SparkCraft AI Consulting, AI Scholars AI Engineering Bootcamp, Nanny Spark` 


- **AI 挑战赛旨在构建用于保姆招聘的 AI 匹配流水线**：一名成员宣布了一项与 **SparkCraft AI Consulting**、**AI Scholars AI Engineering Bootcamp** 以及 **Nanny Spark** 合作的真实客户 **AI 挑战赛**，旨在为保姆招聘服务构建 **AI 匹配流水线**。
   - 目标是为数据收集、AI 驱动的匹配、面试记录分析和交付工作流创建解决方案，并有可能**从第一天起就进行生产部署**。
- **AI 挑战赛奖励 AI 训练营名额及推荐信**：**AI 挑战赛** 的 **前 3 名** 参与者将获得 **1 个 AI Scholars 4 周 AI 工程训练营** 的名额以及来自 **Nanny Spark 创始人** 的推荐。
   - 关键日期包括：**周日晚上 8 点 (EST)** 的启动宣讲会 ([https://luma.com/iq1u2sur](https://luma.com/iq1u2sur))，**周三凌晨 3 点 (EST)** 的提交截止日期，以及 **周三下午 5 点和晚上 8 点 (EST)** 的评审会议 ([https://luma.com/gexiv0x0](https://luma.com/gexiv0x0))。


  

---


---