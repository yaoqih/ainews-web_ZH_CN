---
companies:
- langchain
- cursor
- huggingface
- openai
- weights-biases
date: '2026-01-07T05:44:39.731046Z'
description: '**2026年1月6日至1月7日 AI 新闻摘要**


  当日消息面相对平静，核心更新如下：


  *   **LangChain DeepAgents** 推出了 **Ralph 模式**，旨在实现持久的智能体循环（persistent agent loops）。

  *   **Cursor** 优化了上下文管理，将 Token 使用量降低了 **46.9%**。同时，为编码智能体引入了带有“允许/拒绝列表”的运行安全措施。

  *   **MCP（模型上下文协议）** 集成正在向助手和机器人领域扩展，Hugging Face 已通过 **HuggingChat + HF MCP 服务器**实现了助手的嵌入。

  *   **DeepSeek-R1** 论文已扩展至 **86 页**，重点强调了轨迹探索（trajectory exploration）和强化学习（RL）对行为的塑造。

  *   **NousCoder-14B** 在经过 **4 天**的强化学习训练后，在 **LiveCodeBench** 上的表现提升了 **7%**，展示了小型开源模型在编码强化学习方面的进展。

  *   **热门推文**还提到了引发病毒式传播的“96GB RAM 笔记本电脑”、**OpenAI** 推出的 **ChatGPT Health** 服务，以及
  **Karpathy** 制作的关于 nanochat 缩放定律（scaling-law）的系列短片。'
id: MjAyNi0w
models:
- nouscoder-14b
- deepseek-r1
people:
- karpathy
- _philschmid
- omarsar0
title: 今天没发生什么特别的事。
topics:
- agent-frameworks
- context-management
- reinforcement-learning
- operational-safety
- model-transparency
- trajectory-exploration
- token-optimization
- coding-agents
- integration-platforms
---

**平静的一天**

> 2026年1月6日至1月7日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 服务（包含 **204** 个频道和 **4658** 条消息）。预计为您节省的阅读时间（按 200wpm 计算）：**421 分钟**。**我们的新网站**现已上线，支持完整的元数据搜索，并以极具设计感的方式呈现过往所有内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

平静的一天。


---

# AI Twitter 回顾

**热门推文（按互动量排序）**

- **硬件/算力与开发者文化**：“96GB RAM 笔记本电脑”在网络上疯传 ([@vikhyatk](https://twitter.com/vikhyatk/status/2008922250112819381))；“ChatGPT Health”发布 ([OpenAI](https://twitter.com/OpenAI/status/2008987566796640575))；Karpathy 关于 **nanochat 扩展定律 (scaling-law) 系列短贴** ([@karpathy](https://twitter.com/karpathy/status/2009037707918626874))；xAI 策略/文化及融资相关的推文 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008765567922999573), [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008774688382599520))。

---

**Agent 与开发者工具：“Agent 框架”、DeepAgents、Cursor 上下文、MCP 无处不在**

- **LangChain DeepAgents + “Ralph Mode”（具有文件系统记忆的无限循环 Agent）**：多篇推文趋向于同一种模式：不再将“所有内容都塞进提示词 (prompt)”，而是运行一个**循环**，让 Agent 在每次迭代中刷新上下文并将状态持久化到磁盘。LangChain 在 **DeepAgents** 之上发布了 **Ralph Mode** ([LangChain OSS](https://twitter.com/langchain_oss/status/2008942888810631518))，这被视为一种实用的“持续运行，满意后 Ctrl+C 停止”的 Agent 模式。独立评论将其定义为“Agent 框架时代”，人们将重新组合轻量级编排器，而不是构建完整的 IDE ([omarsar0](https://twitter.com/omarsar0/status/2009061265864262111))。相关说明：DeepAgents 的定位类似于“Claude Agents SDK，但与模型无关” ([mstockton](https://twitter.com/mstockton/status/2008742557388599384))。
- **Cursor 的上下文管理转型**：Cursor 报告称已重构其 Agent 的上下文系统，通过文件/工具/历史记录**动态发现**相关上下文，而非提示词填充，从而将 Token 使用量降低了 **46.9%** ([mntruell](https://twitter.com/mntruell/status/2008793943472062807))。这与“文件系统即记忆”以及长程 (long-horizon) 代码 Agent 的趋势一致，此外还展现了 Cursor 作为**桌面 Agent 仪表板**而非仅仅是一个 IDE 的愿景 ([mntruell](https://twitter.com/mntruell/status/2008993971826249986))。额外观点：将对话记录写入磁盘可以实现“数百万 Token 长度”的对话 ([amanrsanger](https://twitter.com/amanrsanger/status/2008985132523495847))。
- **代码 Agent 的运行安全（允许/拒绝列表）**：随着 “YOLO 模式” 变得普遍，生态系统重新发现工具执行审批是瓶颈和风险面。[@_philschmid](https://twitter.com/_philschmid/status/2008975389415354491) 分享了一个针对 Agent shell 的具体允许/拒绝命令列表（拒绝 `git push`、`git reset`、发布命令等）。
- **MCP 作为集成基座**：MCP 出现在“论文对话”体验（Hugging Face Papers 助手）以及机器人/Agent 中；例如，Claude Code ↔ Reachy Mini 实验 ([Trtd6Trtd](https://twitter.com/Trtd6Trtd/status/2008933816073846810))。Hugging Face 正通过 **HuggingChat + HF MCP 服务器**将助手嵌入到论文页面中 ([AdinaYakup](https://twitter.com/AdinaYakup/status/2008863050355675152), [@_akhaliq](https://twitter.com/_akhaliq/status/2008915667760635986))。
- **浏览器 Agent “确实有效”的案例**：一个具体的端到端自动化案例——Claude Code 仅根据两句话的任务指令，自主处理了亚马逊退货并重新订购了尺码——这标志着人们对浏览器工具可靠性的信心日益增强 ([corbtt](https://twitter.com/corbtt/status/2009003003630735616))。

---

**模型发布与评估生态：权重开放的速度、用于代码的强化学习 (RL)、视觉/视频以及对排行榜的质疑**

- **DeepSeek-R1 论文扩展（从 22 页增至 86 页）**：更新后的 DeepSeek-R1 报告被视为一次重大的透明度升级，增加了 Judge Prompts、合成数据 Prompts、Harness 细节、分析以及蒸馏（Distillation）章节 ([机器之心](https://twitter.com/jiqizhixin/status/2008805570145644849)；另见 [andrew_n_carr](https://twitter.com/andrew_n_carr/status/2008953964566597771))。一种技术解读是：性能提升与其说归功于“更好的数据”，不如说归功于**轨迹探索/验证（trajectory exploration/verification）**和可验证奖励（verifiable rewards），RL 塑造了模型行为而非仅仅注入知识 ([gm8xx8](https://twitter.com/gm8xx8/status/2009000108327670116))。
- **用于编程的 RL 正在缩小小型开源模型的差距**：W&B 强调了 **NousCoder-14B** 在 **LiveCodeBench 上提升了 +7%**，且仅训练了 **4 天**，这是开源 RL 后训练（post-training）获得实质性杠杆作用的一个例子 ([Weights & Biases](https://twitter.com/wandb/status/2008946807523692965))。Nous 随后还发布了一个数据集（“我们忘了发布数据集了！”）([Teknium](https://twitter.com/Teknium/status/2008857949524074635))。
- **视觉/视频开源模型**：
  - **Black Forest Labs**：在 Hugging Face 上发布了量化版 **FLUX.2 [dev] 32B**；亮点包括多参考图支持（最多 **10 张图像**）、**4MP** 分辨率、改进的文本渲染，并针对 NVIDIA GPU 进行了优化 ([HuggingPapers](https://twitter.com/HuggingPapers/status/2008762251352711235))。
  - **LTX-2**：声称在 Artificial Analysis 的文本转视频（text-to-video）和图像转视频（image-to-video）开放权重排行榜上排名第一 ([ltx_model](https://twitter.com/ltx_model/status/2008862459327865121))；同时也被讨论作为一个音视频联合基础模型 ([@_akhaliq](https://twitter.com/_akhaliq/status/2008964274186789217))。
  - **OmniHuman 1.5 720P** 在 fal 上线：从图像+音频+文本生成数字人视频，提升了面部一致性、对口型（lip-sync）以及相机/身体控制 ([fal](https://twitter.com/fal/status/2008922947562471802))。
  - **Qwen 图像编辑工具**：fal 发布了一个针对 Qwen-Image-Edit-2511 的多角度相机控制 LoRA，该模型基于 **96 个相机姿态**和 **3000 多个 Gaussian Splatting 渲染图**训练而成 ([fal](https://twitter.com/fal/status/2008954582018248755))。
- **评估/排行榜信任危机**：Teknium 认为 LM Arena 已变成“氪金赢（pay to win）”，激励了为了追求排行榜高分而导致模型质量退化的行为，并声称投稿处理不公 ([Teknium](https://twitter.com/Teknium/status/2008828875355443634))。另外，一篇关于“Scaling 已死”的论文/文章引发了激烈讨论：批评者认为，综合“6 项任务”的平均分以及仅限开源模型的对比具有误导性；“Scaling Laws 不等于 Scaling”，闭源模型在真实对话质量上的领先优势依然明显 ([giffmana](https://twitter.com/giffmana/status/2008825049889845452))。
- **基准测试向长时程 Agent 真实性演进**：CodeClash 作为一个迭代式的、对抗性的长时程 SWE 基准测试推出，并发布了全新的训练集 ([OfirPress](https://twitter.com/OfirPress/status/2008986204088545423))——这符合从单次编程到多步工具+执行循环的更广泛趋势。

---

**检索与索引：从 “RAG” 到长上下文 + 新型本地索引**

- **LEANN：“停止存储 Embedding”**：一个值得关注的系统层面主张：通过存储压缩图并在查询时选择性地重新计算 Embedding，仅用 **6GB 即可索引 6000 万个文本块**（相比之下通常需要 200GB）；其定位是实现在全新规模下的本地 RAG ([LiorOnAI](https://twitter.com/LiorOnAI/status/2008871398433759298)，仓库链接：[github](https://twitter.com/LiorOnAI/status/2008871399813759033))。工程师应审慎检查延迟/吞吐量的权衡以及重算下的召回率，但“图 + 选择性重算”的方向契合了更广泛的存储/边缘端约束。
- **RLM 与检索（lateinteraction 的立场）**：检索不会“消失”，因为语料库规模的查询需要通过索引进行次线性（sublinear）访问；RLM 被定位为长时单次上下文（long one-off context），而非检索系统的替代品 ([lateinteraction](https://twitter.com/lateinteraction/status/2008766087752511718))。同时也提醒，“检索后读取（retrieve-then-read）”的 RAG 工作流在 2020 年底就已“过时”，取而代之的是像 Baleen 这样更具迭代性的架构 ([lateinteraction](https://twitter.com/lateinteraction/status/2008768325908918328))。
- **语音 Agent 中的实时检索**：Qdrant 演示：一个实时电话语音 Agent 从索引到 Qdrant 的 Google Sheet 中查询经销商库存，并在不到一秒的时间内做出响应 ([qdrant_engine](https://twitter.com/qdrant_engine/status/2008810361924055370))。这强化了一种务实的模式：结构化过滤 + 快速检索 + 语音 UX。
- **数据提取基础设施**：Hugging Face 分享了关于从 **13 亿份 PDF** 中提取可用数据的深度调研 ([eliebakouch](https://twitter.com/eliebakouch/status/2008933337994322167))，强调“PDF 仅占 Web 内容的 0.6%，但却蕴含着极高价值的内容”。

---

**算力、Kernel 与缩放讨论：Chinchilla 风格科学、后训练系统以及 AI 驱动的 Kernel 自动调优**

- **Karpathy 的 “nanochat miniseries v1”**：一个在有限预算下进行缩放定律（scaling-law）科学研究的实用方案：训练计算最优的小型模型序列，还原类似 Chinchilla 的指数（参数和 tokens 约为 0.5），估算“计算独立常数”（nanochat 建议为 **8**，而 Chinchilla 为 **20**），并通过 CORE 分数将结果与 GPT-2/3 联系起来——报告的总成本约为 **$100（在 8×H100 上运行约 4 小时）** ([karpathy](https://twitter.com/karpathy/status/2009037707918626874))。对于试图通过小规模系统性实验来降低“大规模运行”风险的团队来说，这是一个非常有用的模板。
- **Prime-RL 内存优化**：“带有融合 logprobs+entropy 的词表分块 lm_head” 避免了实例化完整的 logits，从而实现了大量的内存节省 ([m_sirovatka](https://twitter.com/m_sirovatka/status/2008905312992964687))。这类底层优化直接扩大了可行 RL/后训练的 Batch Size。
- **通过完整系统进行 Kernel 生成与评估**：一份关于集成到 vLLM 中的 AI 生成融合 RMSNorm kernel 的报告显示，其比现有的 RMSNorm 提速 **40%**，端到端性能提升 **+1.6%**；观察结果：AI 编写了冗长的启发式/类似自动调优器的代码，并可能引入稳定性风险（段错误 segfault 边缘情况），这引发了关于社区能容忍多少回退机制和确定性欠账的问题 ([marksaroufim](https://twitter.com/marksaroufim/status/2009096176789016600))。
- **来自 CES 的硬件叙述**：一个连贯的“运行场景”框架：Qualcomm 推动全天候本地推理（约 80 TOPS 的 NPU），NVIDIA 强调中心化的 “AI 工厂” + 物理部署循环，AMD 强调跨云/PC/边缘的异构延续性 ([TheTuringPost](https://twitter.com/TheTuringPost/status/2009052319871316060))。这清晰地映射了 Agent 的 UX 需求：本地低延迟、云端重推理，以及能够在这两者之间进行路由的工具链。

---

**应用 AI 产品：健康、语音伴侣、机器人演示以及端侧小模型**

- **ChatGPT Health 发布（重隐私与数据集成）**：OpenAI 推出了专门的健康空间，能够安全地连接医疗记录和健康应用，以便根据用户数据给出回复 ([OpenAI](https://twitter.com/OpenAI/status/2008987566796640575)，公告链接：https://openai.com/index/introducing-chatgpt-health/)。分享的显著实现细节包括：额外加密层（每用户密钥）、增强的隔离/细分、无论设置如何健康对话均不参与训练，且健康内存与全局内存隔离 ([cryps1s](https://twitter.com/cryps1s/status/2009040709635199151))。早期通过等候名单 (waitlist) 推出，随后将扩展到包括免费层级在内的所有用户 ([thekaransinghal](https://twitter.com/thekaransinghal/status/2008990098193633529), [nickaturley](https://twitter.com/nickaturley/status/2009007121942417530))。
- **端侧摘要作为“小模型”切入点**：Liquid AI 与 AMD 联合发布 **LFM2-2.6B-Transcript**，针对长篇会议转录，仅需 **<3GB RAM**，可在 CPU/GPU/NPU 上本地执行，并提供“云端级”摘要能力 ([liquidai](https://twitter.com/liquidai/status/2008954886659166371)；[maximelabonne](https://twitter.com/maximelabonne/status/2008955850665415152) 的总结)。这强化了一种模式：领域微调的小模型在严格的隐私/延迟限制下交付生产价值。
- **语音优先的 Agent 与机器人**：
  - CES 热门助手演示，基于 **pipecat_ai** 构建，涵盖多模型/多模态混合云+本地、机器人控制和语音界面；包含开源硬件 **Reachy Mini** ([kwindla](https://twitter.com/kwindla/status/2008743885523349774))。
  - 爱好者 Reachy Mini “人脸跟随”项目：经过微调的检测器和控制循环；包含数据集和教程链接 ([skalskip92](https://twitter.com/skalskip92/status/2008923043888841018))。
- **编程 Agent 的企业级部署**：Cognition 与 Infosys 合作部署 Devin；声称“以创纪录的时间”完成了复杂的 COBOL 迁移 ([cognition](https://twitter.com/cognition/status/2008984320564981780))。

---

**生态系统与战略信号：中国/开源采用、融资军备竞赛以及“社交分发”护城河**

- **权重开放（Open-weight）模型的采用正向中国主导的生态系统转移**：Nat Lambert 分享了更新后的“开放模型生态系统”图表，强调了中国在采用率方面的领先地位日益增长 ([natolambert](https://twitter.com/natolambert/status/2008920674442637635))。Stanford NLP 指出阿里巴巴的 **Qwen** 在权重开放模型的使用上取得了“压倒性胜利” ([stanfordnlp](https://twitter.com/stanfordnlp/status/2008953208907927601))。Clement Delangue 注意到**韩国政府支持的开源 AI** 产出了多个热门的 HF 模型 ([ClementDelangue](https://twitter.com/ClementDelangue/status/2008954270411051465))。
- **xAI 战略：通过 X 平台实现分发优先**：xAI 被认为具有独特优势，因为它拥有一个社交网络（实时数据 + 约 2.5 亿日活用户），通过产品界面推送 Grok；“别人构建更好的模型，而 xAI 构建关注度” ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008765567922999573))。另一条推文声称 xAI 筹集了 **200 亿美元**，成为融资额第二高的 AI 实验室 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2008774688382599520))。
- **融资额持续膨胀**：报告称：Anthropic 计划以 **3500 亿美元的估值**筹集 **100 亿美元** ([SawyerMerritt](https://twitter.com/SawyerMerritt/status/2008964178204295429))。
- **开发者 UX 元信号**：多条推文指出可见的推理轨迹（DeepSeek 的“展示其工作过程”）带来了“信心 UX”的影响，并推测下一次 UX 创新早已该出现 ([dbreunig](https://twitter.com/dbreunig/status/2008928100009267553))——这与推动 Agent 透明度（“我现在正在阅读/做什么，为什么？”）而非原始的 Chain-of-Thought 堆砌的更广泛趋势相一致。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 本地 AI 模型性能基准测试

  - **[llama.cpp vs Ollama：在 Qwen-3 Coder 32B (FP16) 上，代码生成吞吐量提升约 70%](https://www.reddit.com/r/LocalLLaMA/comments/1q64f26/llamacpp_vs_ollama_70_higher_code_generation/)** (热度: 303): **一位用户报告称，在 RTX 5090 + RTX 3090 Ti 的配置下使用 FP16 精度的 Qwen-3 Coder 32B 模型时，llama.cpp 与 Ollama 在代码生成吞吐量方面存在显著的性能差异。llama.cpp 的吞吐量约为 `52 tokens/sec`，而 Ollama 仅达到 `30 tokens/sec`，这意味着 llama.cpp 具有约 `70%` 的性能优势。用户推测，这种差异可能是由于 CUDA 内核、Attention 实现、上下文或批处理（batching）默认值、调度器或多 GPU 利用率，或者是 Ollama 运行环境/API 层的开销导致的。** 评论者认为，与被视为更高效且直接的 **llama.cpp** 相比，**Ollama** 不太适合严肃的开发工作。此外，有人对 **Qwen-3 Coder 32B** 模型是否存在表示怀疑，并建议该用户可能指的是 **Qwen-3 Coder 30b a3b**。

    - Ollama 的实现因其对 GPU 层级和张量分配（tensor assignments）的处理而受到批评，特别是在 MoE 模型和多 GPU 的场景下。一位用户指出，Ollama 设置 GPU 层数的启发式方法并非最优，导致张量放置效率低下。相比之下，`llama.cpp` 最近的一项实现通过感知 MoE 并更好地利用 VRAM 改进了这一点，从而提升了性能。[来源](https://www.reddit.com/r/LocalLLaMA/comments/1pn2e1c/llamacpp_automation_for_gpu_layers_tensor_split/)。
    - 关于模型名称存在一些混淆，一位用户质疑“Qwen 3 Coder 32B”的存在，并认为这可能是“Qwen 3 Coder 30b a3b”的误写。这突显了在讨论中精确命名模型以避免误解的重要性。
    - Ollama 被视为一种面向初学者的工具，以灵活性和性能为代价提供了易用性。建议有经验的用户直接使用 `llama.cpp` 以获得更多控制权和更好的结果，因为 Ollama 的设计选择往往不符合严肃工作的需求。

  - **[本地运行 ACE-Step：在 8GB VRAM 上 20 秒生成 4 分钟音乐（对比 Suno 云端 API）](https://www.reddit.com/r/LocalLLaMA/comments/1q64qpx/running_acestep_locally_4minute_music_generation/)** (热度: 16): **该帖子讨论了在本地部署 **ACE-Step**，通过 CPU 卸载（offload）在 `8GB VRAM` 上仅用约 `20 秒` 即可生成 4 分钟音乐。这被视为 **Suno** 云端 API 的替代方案，后者有速率限制且每月费用达 `$30`。该配置包含多项优化，如 CPU 卸载将 VRAM 占用从 `16GB` 降低到 `7.5GB`，以及 `8-bit 量化` 将其降低到 `9GB` 且仅有 `25%` 的速度下降。文章提供了关于安装、质量控制以及 Stem 风格生成和用于特定流派的 LoRA 加载等高级功能的全面指南。它强调了 ACE-Step 基于扩散（diffusion）的架构相较于传统自回归（autoregressive）模型的效率，从而实现了快速生成长达数分钟的音乐。** 一位评论者质疑生成的音乐质量，指出其之前的质量低于 Suno 的水平。另一位评论者则对“包含完整代码的真实应用场景”部分表示赞赏，并表示打算尝试该配置。

### 2. Agent 安全与 Fail-Closed 系统

  - **[我为我的 Agent 构建了一个 "Fail-Closed" 断路器，因为仅靠 Prompts 不足以停止幻觉。今天开源 (Python)](https://www.reddit.com/r/LocalLLaMA/comments/1q64zgt/i_built_a_failclosed_circuit_breaker_for_my_agent/)** (活跃度: 6): **该贴介绍了 FailWatch，这是一款通过实现 "Fail-Closed"（故障关闭）断路器来强制执行 Agent 操作中确定性安全性的中间件。该系统对于防止金融交易中的大规模错误至关重要，特别是在发生网络故障或验证逻辑崩溃时。该中间件通过阻塞超出预定限制的操作、对模糊操作要求人工审批以及在网络中断期间锁定操作来运行。它作为一个 Python decorator 实现，确保在工具执行前进行同步验证，这对于维持对潜在风险操作的控制至关重要。该工具已开源，可在 GitHub 上获取并可通过 pip 安装。** 一位评论者赞赏了 "fail-closed" 方法，指出许多框架对错误的处理不够充分，可能导致潜在的财务失误。另一个引起关注的问题是同步验证可能带来的延迟，并询问 Guard Server 是否部署在本地以减轻此问题。

    - "fail-closed" 断路器的实现因其谨慎的方法而受到称赞，这与许多即便在出错时仍继续运行、从而可能导致昂贵错误的 Agent 框架形成鲜明对比。评论者强调了这种方法在防止意外行动（如错误的金融交易）方面的重要性。
    - 有人针对每次工具调用前的同步验证可能产生的延迟影响提出了技术担忧，特别是在涉及大量链式操作的场景中。评论者询问 Guard Server 是否为本地部署（这可以缓解延迟问题），并暗示该解决方案的架构可能会显著影响性能。

  - **[双 GPU vs 专用 AI 主机](https://www.reddit.com/r/LocalLLM/comments/1q6f7ea/double_gpu_vs_dedicated_ai_box/)** (活跃度: 41): **用户正在考虑是再添加一块 RTX 4080 GPU，还是购买像 GMKtec Evo-X2（配备 128GB 内存）这样的专用 AI 盒子，用于运行私有 LLM 任务，如 Inference、文档摘要和轻量级图像生成。RTX 4080 足以处理小型任务，但用户正在考虑对内部文档进行 Fine-tuning。建议使用配备 Nvidia GPU 的专用机器以获得更好的性能，尤其是对于通过 API 运行模型的情况，因为它允许工作负载分离和高效的资源管理。添加另一块 RTX 4080 将提供 32GB 的 VRAM，适合高效运行 14b 和 20b 参数的模型。或者，如果预算不受限制，建议使用具有 96GB VRAM 的 RTX 6000 以获得更强大的能力。** 评论者普遍倾向于使用 Nvidia GPU 而非集成内存方案，以获得更高的速度和效率。专用机器被优先推荐用于运行模型，以便更好地进行管理和性能优化，尤其是通过 API 访问时。添加另一块 RTX 4080 被视为在不显著降低系统速度的情况下增强能力的经济高效的方式。

    - **fastandlight** 建议使用配备 Nvidia GPU 的专用机器来运行 AI 模型，强调了将工作负载与个人设备分离的好处。他们建议使用带有充足插槽和 RAM 的旧款 PCIe v4 机器，运行 Linux，并以 OpenAI 服务模式利用 `vllm` 或 `llama.cpp` 等软件。这种设置允许通过 API 进行远程访问，使主设备免受 GPU 产生的计算负荷和热量的影响。
    - **alphatrad** 强调了 GPU 相比集成内存系统在性能上的优势，特别是在运行大模型时。他们建议添加另一块 RTX 4080 以达到 32GB VRAM，这将是高效处理 14b 和 20b 参数模型的理想选择。这种配置可以在不显著降低系统可用性的情况下维持性能，使其适用于 Retrieval-Augmented Generation (RAG) 等任务。
    - **LaysWellWithOthers** 主张使用多块 RTX 3090 GPU，因为它们在每美元 VRAM 成本方面具有极高的性价比。他们强调了确保系统能够物理容纳额外 GPU 的重要性，包括对电源容量和散热管理的考虑。他们分享了自己在一个开放式机架上配置 4x3090 的专用 AI 工作站设置，强调了此类配置的扩展性和性能优势。

### 3. Google Colab 上的 AI 模型设置与故障排除

  - **[Need help with Collab!](https://www.reddit.com/r/LocalLLM/comments/1q6frf8/need_help_with_collab/)** (Activity: 1): **用户正尝试在 Google Colab 上运行 AI 模型，特别是使用 `chatterbox turbo` 模型进行文本转语音（TTS）任务。他们遇到了多行字符串输入产生乱码的问题，除非将其拆分为块，但这会破坏自然的停顿。用户还注意到 `chatterbox` TTS 缺少一些功能，如 `cfg` 和 `exaggeration` 参数。他们正在探索 `vibevoice` 等替代方案，但只发现了 `0.5B` 模型，没有找到 `1.5B` 模型。他们寻求关于设置像 `Gradio` 这样的界面以实现更简便交互的指导，类似于他们使用 `Pinokio` 的体验。** 评论者建议探索其他可能更好支持多行输入的 TTS 模型，并推荐使用 `Gradio` 来构建用户友好型界面。一些人强调了检查模型与 Colab 上 T4 GPU 兼容性的重要性，并建议查看社区论坛或 GitHub 仓库以获取更全面的指南。

## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Gemini 与 AI 聊天机器人市场趋势

  - **[Gemini surpassed 20% traffic share threshold among the overall traffic for AI chatbots(Jan 2026)](https://www.reddit.com/r/singularity/comments/1q6a3lp/gemini_surpassed_20_traffic_share_threshold_among/)** (Activity: 659): **该图片是来自 **Similarweb 报告** 的柱状图，展示了各种 AI 聊天机器人的访问量份额，突出显示 **Gemini** 最近突破了 `20%` 的份额阈值。**OpenAI 的 ChatGPT** 仍然占据最大份额，但已跌破 `65%` 大关。**Grok** 也因突破 `3%` 并接近 **DeepSeek** 而受到关注。这些数据反映了过去一年 AI 聊天机器人市场动态的变化，Gemini 获得了显著的增长势头。** 一位评论者强调了 **Gemini 3** 对市场的影响，指出了其科学的方法论和微调（fine-tuning）能力。另一位评论者则提出了一个问题：市场是在扩大，还是提供商仅仅是在重新分配现有用户。

    - Gemini 3 的发布凸显了 OpenAI 此前占据的巨大市场份额，表明 AI 聊天机器人的竞争格局发生了变化。Gemini 3 Pro 的推出强调了解决问题的科学方法并受益于微调，其对实际应用产生的实质性影响已得到认可，暗示其在市场上具有强大的竞争优势。
    - 有讨论指出 AI 聊天机器人市场是在扩张，还是各公司仅仅是在抢夺对方的用户。这提出了一个问题：如果整体市场规模在萎缩，某些提供商的市场份额增加可能会产生误导，这表明需要更详细的市场分析来了解真实的增长动态。
    - Google 提高 Gemini 市场份额的策略包括在全球范围内免费提供一整年的 Gemini Pro，虽然目标受众是学生，但任何人都可以获取。这种激进的促销手段旨在吸引新用户并转化来自其他 AI 平台的用户，突显了为获取市场份额而采取的竞争策略。

  - **[Gemini surpassed 20% traffic share threshold among overall traffic for AI chatbots](https://www.reddit.com/r/GeminiAI/comments/1q69y88/gemini_surpassed_20_traffic_share_threshold_among/)** (Activity: 180): **该图片是来自 **Similarweb 报告** 的柱状图，追踪了截至 2026 年全球 AI 聊天机器人的访问量份额。报告强调 **Gemini** 已突破 `20%` 的访问量份额，标志着一个重要的里程碑。与此同时，**ChatGPT** 出现下滑，跌破 `65%` 大关，而 **Grok** 正在获得动力，突破 `3%` 并接近 **DeepSeek** 的份额。这些数据反映了 AI 聊天机器人市场不断变化的动态，其中 Gemini 的增长尤为显著。** 一条评论认为 Gemini 的增长可能会影响其性能，因为一位用户形容 Gemini 3 Pro “不可用”且“完全损坏”。另一条评论则期待 OpenAI 引入广告后市场动态的变化。

- **[ChatGPT 正在失去市场份额，而 Google Gemini 正在迎头赶上](https://www.reddit.com/r/GeminiAI/comments/1q6skak/chatgpt_is_losing_market_share_as_google_gemini/)** (热度: 287): 据报道，**ChatGPT** 的市场份额正在流向 **Google Gemini**，因为 Google 利用其庞大的生态系统将 AI 功能更无缝地集成到日常工作流中。文章指出，虽然 OpenAI 的 ChatGPT 最初是 AI 能力的突破性展示，但 Google 的基础设施和用户群提供了更具吸引力的方案，包括家庭共享和 2TB 云存储等功能。这种转变凸显了拥有综合平台公司的战略优势，因为它们可以将 AI 作为增强现有服务的一项功能嵌入，而不是作为独立产品。评论者认为，Google 庞大的生态系统（包括 Mail, Docs 和 YouTube 等服务）相比 ChatGPT 这种独立 AI 应用具有显著优势。他们建议，除非 OpenAI 能够更深入地集成到更大的平台中（例如可能被 Microsoft 收购），否则可能会继续失去市场份额。

    - Google 的竞争优势在于其广泛的生态系统，包括 Mail, Sheets, Docs, Drive 和 YouTube 等服务，这些服务已深度集成到用户的日常工作流中。这种集成使用户更容易采用 Google 的 AI 产品，因为它们已经嵌入在熟悉的环境中，而不像 ChatGPT 这样的独立应用。
    - Google 提供 2TB 云存储和家庭共享等额外服务，提供了超越 AI 能力本身的有吸引力的价值主张。这种捆绑策略使得 Google 的 AI 服务对于已经投入 Google 生态系统的用户更具吸引力，从而可能导致市场份额从 ChatGPT 等独立 AI 产品转向 Google。
    - 讨论强调了一个潜在趋势，即独立 AI 应用可能难以与拥有成熟平台和生态系统的科技巨头竞争。随着这些公司将 AI 集成到现有服务中，用户可能会觉得使用不提供同等集成水平的独立应用非常繁琐，从而导致 ChatGPT 等独立 AI 应用的使用率下降。

  - **[只有我一个人觉得吗，还是 Gemini 最近被“脑叶切除”了？](https://www.reddit.com/r/Bard/comments/1q65rty/is_it_just_me_or_has_gemini_been_lobotomized/)** (热度: 190): **用户报告称，在过去几周里，语言模型服务 **Gemini** 出现了显著的性能下降。问题包括响应速度慢、频繁崩溃、幻觉增加以及指令遵循能力变差。用户还注意到模型过度使用习语、注入无关的个人信息以及无法正确分析图像。尽管尝试了重置设置和更改模型，这些问题仍然存在，导致以前认为该服务有用的用户感到沮丧。** 评论者表达了对 Gemini 现状的沮丧，强调其无法保留超过几条消息的上下文，且数据保留能力较差。由于这些问题，一些用户正在考虑转回 ChatGPT 等替代方案，此外，该服务的集成功能（如 NotebookLM）也受到了批评，被认为效果不佳。

    - **Goldengod4818** 强调了 Gemini 在数据保留方面的重大问题，指出它难以“看到”最后 10 条消息之后的内容，这严重影响了其在长期项目中的可用性。他们提到尝试集成 NotebookLM 以增强功能，但将其描述为一场“灾难”，表明该集成未能有效支持复杂任务或改善用户体验。
    - **DearRub1218** 提供了一个详细的类比来描述 Gemini 的不稳定性，将其比作一个神经和肌肉错位的人，导致表现不可预测。他们指出，虽然 Gemini 偶尔表现出色，但往往无法提供一致的结果，将其运作比作“脱节的地板舞”。这表明该模型的内部逻辑或架构可能配置错误，导致行为异常。
    - **locomotive-1** 讨论了 Gemini 性能的退化，特别是在较长的对话中，它往往会不断重复。他们推测，最近可能为了平衡成本和质量进行了优化，可能缩小了其原本宣称的 100 万 token 的有效上下文窗口，这可以解释观察到的性能下降。

- **[Gemini 3.0 遭到严重削弱](https://www.reddit.com/r/GeminiAI/comments/1q6ecwy/gemini_30_has_been_nerfed_big_time/)** (Activity: 502): **该帖子声称 **Gemini 3.0** 经历了显著降级，特别是在其上下文窗口（context window）方面，该窗口最初宣布支持 `1 million tokens`。用户报告称，该模型在仅进行了几条消息对话后就会遗忘信息，这与最初的说法相矛盾。此外，该模型因不遵循指令而受到批评，例如拒绝执行网页搜索，以及在回答中加入无关的个人语境。该用户已转向使用 **Claude** 进行代码任务，并在 **Perplexity** 上使用 **Gemini 3.0** 进行网页浏览，称那里的体验更好。** 评论者们同意帖子的观点，指出上下文窗口问题感觉像是一种“变相降级”，模型的性能已经恶化到不再是一个可靠工具的地步。

    - 用户报告了 Gemini 3.0 在上下文保留方面的重大问题，指出它难以维持连贯的对话线索，经常重复对话早期无关的细节。这表明它处理长对话的能力有所退化，而这对于维持对话式 AI 应用的用户参与度至关重要。
    - 用户对声称的“一百万 token 上下文窗口”持怀疑态度，因为他们发现 Gemini 3.0 甚至在处理中等规模的文档（如 100 页的 PDF）时都表现吃力。这种差异引发了对模型实际能力与其广告规格之间关系的质疑，凸显了营销声明中可能存在的夸大成分。
    - 尽管 Gemini 3.0 存在问题，用户仍然发现 Studio 和 Notebooklm 等相关工具具有价值，这表明虽然主模型可能存在局限性，但其周围的工具生态系统仍具实用性。这暗示了虽然核心模型可能需要改进，但辅助工具仍能提供令人满意的用户体验。

  - **[是的 Gemini，那确实是一个真实的、通用的海马表情符号...](https://www.reddit.com/r/Bard/comments/1q67vp6/yes_gemini_that_is_in_fact_a_genuine/)** (Activity: 74): **该图片凸显了一种现象，即 AI 模型（如 ChatGPT）在被问及是否存在海马表情符号时会表现出困惑或错误，尽管该符号自 Unicode 15.0 (2022) 起就已可用。这一问题归因于模糊的 tokens 和冲突的训练数据，导致了 AI “幻觉 (hallucinations)”。讨论指向了 AI 训练中更广泛的挑战，即模型可能会从其训练数据集中继承错误信息或不一致性，从而影响其准确识别或召回某些信息的能力。** 一条评论认为该问题并非 Gemini 特有，而是使用“极其廉价的扩散模型 (diffusion model)”的结果，表明这是 AI 模型更广泛的问题。另一条评论对 AI 模型吸收刻意误导信息的可能性表示担忧，质疑其输出的可靠性。

    - 讨论强调了对 AI 模型的一个常见误解，特别指出所讨论的模型不是 **Gemini**，而是一个“极其廉价的扩散模型 (diffusion model)”。这表明存在对 AI 能力的误解或错误标注，这可能导致有关这些模型实际功能的错误信息。扩散模型是一类因能够生成高质量图像而受到关注的生成模型，但它们与 Gemini 这种可能具有不同架构或用途的模型截然不同。
    - 评论“他们都在喝同一口有毒的井水”隐喻式地批评了有关 AI 模型的错误信息传播。这表明 AI 社区存在一个更广泛的问题，即关于模型能力和来源的错误信息被传播，可能导致用户和开发者之间的困惑。它强调了在 AI 领域准确传播信息的重要性，以防止此类误解的扩散。

- **[Thank god for the free trial lol](https://www.reddit.com/r/Bard/comments/1q64r5s/thank_god_for_the_free_trial_lol/)** (Activity: 70): **这张图片幽默地展示了 Gemini API 免费试用的省钱优势，显示了一份账单摘要，其中用户节省了 224 美元，最终总成本为 0 美元。评论显示，用户对 Token 生成相关的成本保持谨慎，因为 API 是根据每次响应中整个聊天记录的 Token 使用量来收费的。用户讨论了管理成本的策略，例如使用 'Context Cache' 将 Token 成本从每百万 Token 2 美元显著降低到 0.02 美元，这表明在使用 Gemini 3.0 Flash 等先进 AI 模型时，用户更关注成本效率。** 用户对使用 Gemini API 的成本表达了宽慰和担忧，一些人选择删除其 API Key 以避免意外费用。还有关于不同层级的有效性以及使用 'Context Cache' 等工具潜在节省成本的讨论。

    - MuriloZR 讨论了从免费层级的 Gemini 2.5 Flash Lite 转换到付费层级的 Gemini 3.0 Flash，强调了显著的性能提升。他们提到使用 'Context Cache' 来优化成本，将费用从 `$2 per 1M tokens` 降低到 `0.02$`，这对于大批量 Token 生成来说是一项重大的省钱措施。
    - Unable_Classic3257 指出了一个关于 Token 生成成本的常见误解，指出 API 在每次响应时都会对整个聊天内容生成 Token。如果管理不当，这可能会导致意想不到的高额成本，正如他们经历过在意识到成本结构之前费用迅速达到 `$8` 的情况。
    - Nayomhee 提出了关于试用期后收费的担忧，询问免费试用期结束后的计费流程。这突显了在云服务中了解订阅模式和潜在自动扣费的重要性。

  - **[Paid vs free Gemini account](https://www.reddit.com/r/GeminiAI/comments/1q6c2j9/paid_vs_free_gemini_account/)** (Activity: 69): **该帖子讨论了 **Gemini** 付费账户与免费账户的对比优势，考虑到背景，这很可能与 **Google** 的服务有关。用户报告称，每月花费 `£20` 的付费版本提供了显著优势，例如更少的使用限制和访问高级模型的权限，这对于研究、分析和编程等任务特别有利。付费版本还包括额外功能，如更多存储空间以及与其他 Google 服务的集成，例如 **YouTube Premium** 和 **Nest Aware Plus**。** 评论者普遍认为，对于经常使用其高级功能的人来说，付费版 Gemini 是值得的，因为它可以节省时间并提高生产力。然而，对于基础任务，免费版本可能就足够了。

    - Overall-Fan3079 强调付费版 Gemini 显著减少了使用限制，这是相比免费版的一个主要优势。他们指出，付费版中的高级模型在编程任务中表现更好，尽管对于基础查询，差异并不显著。
    - Pasto_Shouwa 指出 Gemini 的 Pro 订阅允许最多与 5 位额外用户进行家庭共享，使得 6 个账户仅需 22 美元，极具性价比。此外，订阅还包括 2TB 的共享存储，增强了其对需要大规模存储解决方案的用户的价值主张。

  - **[Another example of the Pro Model Making Ridiculous Mistakes](https://www.reddit.com/r/GeminiAI/comments/1q66mn4/another_example_of_the_pro_model_making/)** (Activity: 66): **该帖子强调了某语言模型 Pro Model 反复出现的一个问题，即它虚假地声称能够解释附加的图像并提供错误的描述。用户对模型的错误和感知到的服务质量下降表示沮丧，尤其是考虑到订阅成本。讨论中的图片是一张简单的狗的照片，模型未能准确描述，导致用户不满。此问题可能与最近的更新或功能（如 Memory 功能）有关，这些功能影响了模型的性能。** 评论者建议，问题可能源于最近的更新，例如增加了 Memory 功能，并推测新版本 (3.1) 可能会很快发布以解决这些问题。

- the_shadow007 指出模型性能的一个潜在问题，认为引入 memory 功能可能无意中导致了性能下降。这暗示了新功能与模型稳定性之间的权衡，这是机器学习开发中的常见挑战。
- ComplexActivity43 批评了在模型性能被感知为下降的情况下维持订阅价格的商业策略。这指向了 AI 服务中客户满意度和价值认知的更广泛问题，特别是当更新不符合用户预期时。
- NoWheel9556 注意到在 Gemini 3 Pro 发布后性能有所下降，表明新版本并不总是意味着更好的性能。这表明在部署更新之前需要进行彻底的测试和验证，以确保它们增强而不是阻碍用户体验。


### 2. 新的 AI 模型和功能发布

  - **[Claude-Code v2.1.0 刚刚发布](https://www.reddit.com/r/ClaudeAI/comments/1q6q9my/claudecode_v210_just_dropped/)** (活跃度: 549): ****Claude-Code v2.1.0** 引入了重大更新，包括自动技能热重载、支持 fork 的 sub-agent 上下文，以及用于响应语言配置的新 `language` 设置。显著的修复解决了调试日志中敏感数据泄露的安全问题和会话持久性问题。该更新还增强了终端兼容性和性能，特别是针对 iTerm2、WezTerm 和 Kitty，并添加了新的 Vim motions 和斜杠命令功能。然而，一个关键 bug 导致变更日志解析器因版本日期格式无效而失败，促使回滚到 v2.0.76。[GitHub Commit](https://github.com/anthropics/claude-code/commit/870624fc1581a70590e382f263e2972b3f1e56f5)。** 有用户报告该更新导致 Claude-Code 崩溃，一个与版本解析相关的特定 bug 导致变更日志显示失败。变通方案涉及编辑变更日志文件以删除日期，开发人员已暂时回滚到 v2.0.76。

    - Claude-Code v2.1.0 中的一个 bug 导致程序崩溃，原因是变更日志显示中的版本字符串格式无效，具体表现为包含了日期 `2.1.0 (2026-01-07)`。此问题记录在 [GitHub issue #16671](https://github.com/anthropics/claude-code/issues/16671) 中。解决方法是使用以下命令编辑变更日志文件以删除日期：`sed -E -i'' 's/(## 2\.1\.0) \([0-9-]*\)/\1/' ~/.claude/cache/changelog.md`。
    - 由于 v2.1.0 中的 bug，开发人员已暂时将版本回滚到 v2.0.76。这次回滚是他们在处理导致崩溃的版本字符串解析问题时的权宜之计。
    - 建议用户不要更新到 v2.1.0，因为它包含一个影响变更日志解析的关键 bug，会导致应用程序崩溃。该问题严重到促使官方回滚到了之前的稳定版本 v2.0.76。

  - **[尝试了新的 GLM 4.7 模型用于编程，作为一个开源模型，其表现出人意料地好](https://www.reddit.com/r/ClaudeCode/comments/1q6f62t/tried_new_model_glm_47_for_coding_and_honestly/)** (活跃度: 102): ****GLM 4.7** 是由 **Zhipu AI** 开发的开源模型，已在 Python 调试、React 组件生成、SQL 查询优化和解释 Java 遗留代码等各种编程任务中进行了测试。该模型在约 `90%` 的时间内提供了功能性代码，在稳定性和上下文处理方面优于 DeepSeek 和 Kimi 等其他中国模型。虽然在解释方面不如 Claude Sonnet 4.5 精炼，但 GLM 4.7 以极低的成本提供了相当的代码输出质量，使其成为高性价比编程任务的可行替代方案。该模型可以处理超过 `500` 行的文件而没有性能问题，并且可以本地运行，这对于专有项目非常有利。** 一些用户发现 GLM 4.7 与 SWE-1.5 等其他模型相比表现平平，理由是它在处理基本要求时存在问题。然而，其他用户成功地将其与 Claude Code 集成，受益于更高的额度和显著降低的成本，一位用户指出在一个全面的代码重构任务中仅消耗了 `5%` 的额度。该模型因其成本效益和在中等复杂任务中的表现而受到赞赏。

- DenizOkcu 强调了将 GLM 4.7 集成到 Claude Code 后的性价比和性能，指出与其他模型相比，它提供了“3 倍更高的额度”，且“价格仅为 1/7”。他们提供了一个在 Claude Code 中配置 GLM 4.7 的代码片段，并强调其在高效处理复杂任务（如重构大型生产代码库）方面的能力，仅消耗了其每小时限额的 5%。
- coopernurse 提到在 Claude Code 中同时使用 GLM 4.7 和 MiniMax 2.1，并指出这两个模型在处理中等复杂任务时表现良好。他们正在对比这两个模型，以确定性能是否存在显著差异，并表示两者都能有效地处理复杂的编程任务。
- AriyaSavaka 指出了 GLM 计划的实惠性，其价格为“每月 3 美元即可获得 3 倍用量”，而 Claude Pro 计划则需 20 美元，并强调其没有每周限制。这表明 GLM 4.7 为需要大量使用且不受高价计划限制的用户提供了一个经济高效的解决方案。

- **[OpenAI 在移动端和网页端发布 ChatGPT Health](https://www.reddit.com/r/OpenAI/comments/1q6ouuf/openai_releases_chatgpt_health_on_mobile_and_web/)** (热度: 629): **OpenAI 推出了 ChatGPT Health，这是一项可在移动端和网页端使用的新功能，旨在促进私密的健康相关对话。该服务允许用户安全地将其医疗记录和健康应用（如 Apple Health、Function Health 和 Peloton）连接到 ChatGPT。界面包含健康自查、医疗报告解读和运动建议等选项，旨在提供一个全面的健康管理工具。其设计强调了在处理敏感健康数据时的用户友好性和隐私性。** 一些用户对聊天机器人准确解读医疗记录的能力表示怀疑，并将其幽默地比作 WebMD。此外，还有关于通过该平台讨论心理健康局限性的警示。

    - 提出的一个关键担忧是数据隐私，具体包括用户的医疗记录和与 ChatGPT Health 的交互是否安全，或者是否可能与第三方（如《纽约时报》等媒体机构）共享。这凸显了了解 OpenAI 针对这项新服务的数据处理和隐私政策的重要性。
    - 对于 ChatGPT Health 准确解读医疗记录的可靠性存在怀疑。与 WebMD 的类比表明人们担心聊天机器人可能会误读医疗信息，从而导致错误的建议或诊断，这强调了对 AI 医疗能力进行严格验证和测试的必要性。
    - 讨论涉及了使用 AI 进行健康咨询的伦理影响，特别是敏感健康数据被滥用的可能性。这引发了关于 AI 开发者在确保工具被适当使用以及用户充分了解相关风险方面的伦理责任问题。

- **[[P] 从零重构 Fuzzy-Pattern Tsetlin Machine：训练速度提升 10 倍，推理速度提升 34 倍（32M+ preds/sec）并支持文本生成](https://www.reddit.com/r/MachineLearning/comments/1q6igw3/p_reengineered_the_fuzzypattern_tsetlin_machine/)** (热度: 29): **该帖子详细介绍了一个经过重新工程化的 Fuzzy-Pattern Tsetlin Machine (FPTM) 版本，通过底层优化实现了显著的性能提升。新实现使训练速度提升了 `10 倍`，推理速度提升了 `34 倍`。在使用 Ryzen 7950X3D 的 MNIST 基准测试中，达到了 `32M+ predictions/sec` 且准确率高达 `98%`。关键优化包括使用 SIMD 指令、缓存友好的内存布局以及 BitSet 索引。提升后的效率使其能够胜任实际的生成任务，展示了一个能够生成莎士比亚风格文本的字符级文本生成器。代码已在 [GitHub](https://github.com/BooBSD/Tsetlin.jl) 开源。** 一位评论者建议通过使用 C 语言重写实现来进一步优化，并询问了所使用的具体 HDC/VSA，指出在其经验中 BSDC-SEG 编码非常有效。

- Fuzzy-Pattern Tsetlin Machine (FPTM) 的重构带来了显著的性能提升，实现了 10 倍的训练加速和 34 倍的推理加速，每秒预测量超过 3200 万次。这表明相较于之前的实现有了大幅优化，可能使其非常适合实时应用。
- FPTM 与 Hyperdimensional Computing (HDC) 或 Vector Symbolic Architectures (VSA) 的集成被强调为一种极具前景的方法。评论者提到 BSDC-SEG 编码特别有效，表明 HDC/VSA 的选择会显著影响 FPTM 的性能和结果。
- 有建议提议用 C 语言重写 FPTM 以进一步提升性能。这暗示目前的实现可能是用更高级的语言编写的，而 C 语言实现可以利用底层优化来实现更大幅度的速度提升。

- **[[R] DeepSeek-R1 的论文于 2 天前更新，从 22 页扩展到 86 页，并增加了大量细节。](https://www.reddit.com/r/MachineLearning/comments/1q6cb0k/r_deepseekr1s_paper_was_updated_2_days_ago/)** (Activity: 176): **关于 **DeepSeek-R1** 的论文已从 `22` 页大幅扩展至 `86` 页，提供了关于其方法论和发现的更全面的细节。此次更新可能会解决之前的问题，例如 `grpo` 奖励计算中的问题，尽管帖子中并未明确确认这一点。该论文可在 [arXiv](https://arxiv.org/abs/2501.12948) 上查阅。** 一条评论对更新是否解决了 `grpo` 奖励计算中的问题提出了疑问，这表明了技术界对该模型性能和实现细节的持续关注和审视。

    - DeepSeek-R1 论文的更新将其内容从 22 页大幅扩展到 86 页，这表明细节有了实质性的增加，并可能解决了之前的问题。一个关键的关注点是更新是否解决了 `grpo` 奖励计算中的问题，这是早期版本中被指出的一个问题。这可能会影响模型的性能和准确性，使其成为审查的关键领域。
    - 论文的扩充还可能包括更全面的实验结果或理论解释，这对于验证模型的结论至关重要。篇幅的增加可能意味着对模型架构、训练过程或应用场景进行了更透彻的探索，从而对其能力和局限性提供更深刻的见解。
    - 提到该论文与 SELU 论文长度的对比，突显了社区对研究出版物深度和全面性的关注。更长的论文通常意味着对主题有更详细的探索，这对于希望了解模型实现细微差别和潜在应用的学者来说非常有益。

- **[James Cameron：“没有演员、没有艺术家的电影”](https://www.reddit.com/r/OpenAI/comments/1q69u4y/james_cameronmovies_without_actors_without_artists/)** (Activity: 560): ****James Cameron** 对 AI 生成的电影表示怀疑，并称：*"我对那玩意儿一点兴趣都没有"*。他认为，AI 可以让没有经过正式培训或缺乏资源的人在 `4 年` 内制作出可与 Hollywood 媲美的电影。这一观点强调了电影制作潜在的民主化，允许那些无法获得昂贵设备或培训的人在行业中竞争。** 评论者对 Cameron 的立场展开了辩论，认为这反映了对电影制作变革和民主化的抵触。一些人认为 AI 可以赋能新的创作者，就像数码相机和 YouTube 等平台所做的那样，可能导致多样化和创意内容的激增。

    - James Cameron 对 AI 在电影制作中的看法强调了行业潜在的民主化，即 AI 可以让缺乏传统资源（如昂贵设备或正式培训）的个人在四年内制作出达到 Hollywood 标准的电影。这预示着电影制作工具的可及性将发生重大转变，可能会降低新创作者的准入门槛。
    - 讨论反映了关于 AI 对创意产业影响的更广泛辩论，一些评论者认为 AI 可能会打破 Hollywood 传统的准入门槛。通过减少对昂贵资源的需求，AI 可能会让更多样化的声音进入市场，类似于 YouTube 等平台实现视频内容创作民主化的过程。
    - 人们认识到 AI 可能会导致内容的激增，就像数码相机和 YouTube 彻底改变了内容创作一样。虽然这可能导致质量参差不齐，但也为小众创作者寻找受众提供了机会，预示着一个创作表达更加便捷和多样化的未来。

- **[据报道 OpenAI 正准备在 ChatGPT 中测试广告](https://www.reddit.com/r/OpenAI/comments/1q6nxy6/openai_is_reportedly_getting_ready_to_test_ads_in/)** (热度: 87): **据报道，**OpenAI** 正准备在其 **ChatGPT** 平台上测试广告，此举可能会显著改变用户体验和商业化策略。这一进展源于 OpenAI 持续为其广泛使用的 AI 服务探索可持续的收入模式，该服务在各行各业都得到了迅速普及。广告的引入可能会影响用户目前享有的无缝交互，引发关于商业化与用户满意度之间平衡的讨论。** 社区对引入广告表示怀疑和担忧，一些用户幽默地表示这可能会导致订阅量下降。广告破坏用户体验的可能性是讨论的核心主题。


  - **[恋童癖者正利用 Sora 及其获取的儿童生物特征数据描绘虐待儿童的场景](https://www.reddit.com/r/OpenAI/comments/1q6521z/pedophiles_are_using_sora_to_depict_themselves/)** (热度: 62): **该帖子引发了对 Sora 应用 cameo 功能被滥用的担忧，据称不法分子利用儿童的生物特征数据创建描绘未成年人在不当情况下的视频。这一问题凸显了改进内容审核和安全措施以防止此类剥削的紧迫性。帖子暗示这是一个普遍存在的问题，可能涉及数百个账户。** 评论者强调了不要在肇事者身份上草率下结论的重要性，认为发布内容的人本身也可能是受害者。人们呼吁加强滥用检测和快速下架机制，以有效解决此类问题。

    - RonaldWRailgun 提出了关于公共个人资料可能被滥用以及隐私重要性的关键点。他们建议，参与制作此类内容的人可能会使用本地模型和私有账户，而不是公开的社交媒体，凸显了在数字空间中识别肇事者的复杂性。
    - Few-Needleworker4391 强调需要加强技术解决方案来应对此类问题，主张建立更强大的滥用检测系统、年龄限制机制和快速内容下架流程。这凸显了开发稳健的数字安全协议以保护弱势群体的重要性。
    - Ok-Addition1264 注意到了帖子上的反对票，认为社区的反应可能反映了对该话题更深层次的问题或误解。这条评论暗示了社区审核的挑战以及在敏感讨论中对用户反馈的解读。

  - **[哇，这真是个大场面。](https://www.reddit.com/r/ClaudeAI/comments/1q6kr4a/wow_this_is_quite_a_situation/)** (热度: 868): **该图片是一个表情包，幽默地调侃了 AI 生成的回复，特别是强调了一条关于 AI **Claude** 对复杂地缘政治局势做出简单化且自动化的回复：“哇，这真是个大场面（Wow, this is quite a situation）。”这反映了关于 AI 在理解细微语境和生成适当回复方面的局限性的更广泛讨论。评论通过分享 AI 对复杂或荒谬查询的简单化或怪异回复的轶事进一步说明了这一点，凸显了 AI 在理解和语境意识方面的挑战。** 评论幽默地讨论了 AI 在面对复杂查询时产生简单或怪异回复的倾向，反映了 AI 在理解细微语境方面的局限性。这包括 AI 对无关或荒谬话题回复的轶事，强调了提高 AI 系统语境意识的必要性。

- 'paralog' 的评论强调了一种情况，即一个 AI 模型（可能是语言模型）被要求查找关于 Elon Musk 和 DOGE 的投机性项目的消息。AI 的回答很模糊，这反映出它在提供投机性或缺乏记录的话题的详细或最新信息方面的局限性。这反映了 AI 模型的一个常见问题，即由于依赖现有数据，它们在处理实时或投机性查询时会感到困难。
- 'Tim-Sylvester' 的评论讨论了一场涉及 Donald Trump 和 Bill Clinton 的互联网辩论，并因提及马而变得更加复杂。这种情况体现了互联网话语的混乱性质以及 AI 模型在解析和验证此类声明时面临的挑战。AI 考虑各种解释（包括 deepfakes 和 memes）的过程，突显了区分真实事件和互联网虚假信息的复杂性。
- 'Icy_Quarter5910' 分享了使用一款 AI 模型（可能是 Claude）的经验，该模型对 iOS SDK 提供了热情的反馈。AI 的反应非常积极，强调了 API 的整洁性和实用性。这种互动强调了 AI 模型通过评估和推荐工具来协助软件开发的潜力，尽管这类反馈的主观性质可能因模型的训练和数据而异。


### 3. AI 模型使用及替代方案

- **[Claude Max 20x 超限，需要插件替代方案以填补短期空缺](https://www.reddit.com/r/ClaudeCode/comments/1q6h34n/overlimit_with_claude_max_20x_and_need_a_plugin/)** (活跃度: 89): **用户已超过 **Claude Max 20x** 的使用配额，正在寻求具有成本效益的替代 API 以继续工作。他们提到 **GLM 4.7** 是一个潜在选项，其在代码澄清和编写测试、重构等小任务中的实用性备受关注。另一个建议是 Pro 计划中的 **ChatGPT 5.2**，它提供 `270k` 的上下文窗口，被认为是每月 20 美元的 **Opus 4.5** 的可行替代方案。** 一位评论者认为，API 的选择是主观的且基于个人经验，并强调了寻找适合个人需求方案的重要性。另一位提到了来自 GPT 的促销 offer，突显了价格和订阅选项的多样性。

    - LinusThiccTips 指出，Pro 计划中的 **ChatGPT 5.2** 提供了 `270k context window`，这明显大于同类计划中的 **Opus 4.5**。这使其成为需要扩展上下文能力的用户的可行替代方案，尤其是在处理复杂的代码库或大型数据集时。
    - 13chase2 提到 **GLM 4.7** 是实验新代码库的一种经济高效的方案。然而，他们表达了对隐私的担忧，因为数据被发送到位于中国的服务器，这对于有严格数据隐私要求的用户来说可能是一个潜在问题。
    - silvercondor 使用 **GLM**（被称为 'temu claude'）来理解和重构代码库以及编写测试。这表明 GLM 在代码澄清和开发任务方面都具有多功能性，使其成为需要代码理解和修改协助的开发者的有用工具。

- **[推荐哪些其他计划 / 模型来替换 Opus](https://www.reddit.com/r/ClaudeCode/comments/1q6c4bq/what_other_plan_model_would_you_recommend_to/)** (活跃度: 76): **这篇 Reddit 帖子讨论了自一月份以来一直表现不佳的 Opus Max x5 计划，并寻求替代方案。用户建议切换到 GLM 或 Minimax 计划，使用带有 Gemini-cli 插件的 Claude code router，并利用 Opencode 实现功能对等，尽管它还存在一些 bug。另一种方法是在 'plan mode' 下使用 Max 5，以维持会话的稳定性和生产力。Opus 4.5 模型因其局限性（特别是在不从上下文中学习的情况下处理复杂任务）而受到关注，但在基于 DSP 的 Rust 音频插件开发等特定领域表现出色。用户还推荐 CC Web，因为它在编码任务中非常有效。** 评论者辩论了不同计划的有效性，一些人因其成本效益和可靠性而推崇 GLM 和 Minimax，而另一些人则强调在使用 Opus 4.5 时上下文和特定任务性能的重要性。还有关于使用多个会话和插件来最大化生产力的讨论。

- trmnl_cmdr 讨论了一种具有成本效益的方法，结合使用了 GLM、minimax plan 和 Claude code router，并辅以 Gemini-cli 插件。他们强调了这些工具在 opencode 中的可用性，虽然 opencode 与 Claude code 功能对等，但据称存在较多 bug。这种设置被描述为一种锱铢必较的策略，在规划和执行阶段都利用了免费和廉价的方案。
- ridablellama 分享了在 opencode 上使用 GLM 的经验，指出当 Opus 遇到问题时，它可以作为一种有用的后备方案。他们提到了 minimax coding plan 的成本效益，以及将 Claude code 与 GLM 配合使用的能力。然而，他们也指出 opencode 崩溃频率更高，且与其他平台相比存在一些差异。
- kronnix111 对比了 ChatGPT 5.2 和 Claude，指出 GPT 5.2 具有卓越的推理和 Bug 检测能力，但缺乏与 GitHub 和终端的集成。他们介绍了一个自己开发的框架 LivingDocFramework，该框架可以与任何代码库或 AI 配合使用。该框架便于外部 Agent 进行 Bug 修复扫描，为管理代码库提供了一种结构化方法。

- **[Google AI Studio 变得无法使用：持续的频率限制和 60 秒延迟](https://www.reddit.com/r/Bard/comments/1q68317/google_ai_studio_is_becoming_unusable_constant/)** (热度: 12)：**Google AI Studio 的用户正面临严重的性能问题，包括 `60-second latency` 和频繁的“exceeded quota”通知，这促使服务转向需要付费 API key。这一变化标志着对以往免费访问模式的背离，影响了 Pro 和 Gemini 3 Flash 版本。延迟和频率限制让习惯于无缝交互的用户感到沮丧。** 一些用户建议停用“Grounding with Google Search”功能以潜在地提高性能，而另一些用户则持有务实观点，认为为有价值的服务付费是合理的。

    - DearRub1218 强调了 Google AI Studio 的严重性能问题，特别提到 G3 Pro 模型在开始处理前会有 45-60 秒的延迟。对于依赖 AI 模型实时或近乎即时响应的用户来说，这种延迟是一个关键问题，表明当前部署中可能存在服务器端瓶颈或效率低下。
    - Over-Customer2915 指出了“Grounding with Google Search”功能的持续问题，该功能似乎更频繁地被默认激活。这可能是导致延迟增加和频率限制的原因之一，因为该功能可能会消耗额外的资源或带宽，从而影响整体性能。
    - riowcaztoljp 提出了关于 AI Studio 与 Google One 计划集成的问题，暗示用户期待更无缝或更具成本效益的集成。这表明用户期望与当前服务产品之间存在潜在差距，可能会影响用户满意度和感知价值。

- **[这是我银行账户的欺诈性收费吗？](https://www.reddit.com/r/OpenAI/comments/1q6bwbt/is_this_fraudulent_charges_to_my_bank_account/)** (热度: 78)：**图像显示了两笔标记为 'OPENAI *CHATGPT SUBSCR' 的交易，金额与标准的 20 美元 ChatGPT Plus 订阅费不符，暗示可能存在欺诈活动。用户声称没有订阅任何付费计划，引发了对未经授权收费的担忧。交易日期显示在未来，这可能表明银行处理系统存在文书错误或更复杂的问题。商户类别代码 '5734' 与计算机软件商店相关，这与 OpenAI 的服务相符，但并未解释金额或日期上的差异。** 一位评论者建议冻结卡片并报告问题，并指出不同地区的定价可能有所不同。另一位评论者指出，部分遮挡的卡片信息仍然可读，建议用户出于安全原因删除该帖子。

- **[在 16GB VRAM 上进行本地 Vibe Coding | Dyad & Oobabooga](https://www.reddit.com/r/Oobabooga/comments/1q6bed6/vibe_coding_local_with_16gb_vram_dyad_oobabooga/)** (活跃度: 12): **该帖子讨论了使用 **Dyad** 和 **Oobabooga** 在 `16GB VRAM` GPU 上进行本地编程的配置，并强调这种配置足以胜任可靠且真实的编程任务。该集成利用 **Oobabooga API** 作为后端来支持 Dyad，提供了一个免费且本地的自动编程解决方案。这种设置因其成本效益和开源特性而尤为值得关注，使资源有限的开发者也能使用。更多技术细节，请参阅原始视频 [此处](https://youtube.com/watch?v=DhKYjtCyD7U&si=fnt5kCLnPwaNKUvi)。** 评论者们对使用 `5070 16GB GPU` 构建本地 AI NAS 服务器的可行性感到好奇，以及单台主机是否能同时支持 Dyad 开发和 GPU 挂载。这表明了人们对实现上述设置的实际硬件配置和成本考量的兴趣。

    - 一位用户询问了使用 `5070 16GB GPU` 构建本地 AI NAS 服务器的可行性。讨论可能围绕 GPU 在本地处理 AI 工作负载的能力展开，考虑了 VRAM 容量和处理能力等因素。`16GB VRAM` 通常足以运行许多 AI 模型，但具体要求取决于所运行模型的复杂性和规模。
    - 另一位用户表示有兴趣购买一块具有 `16+ GB VRAM` 的 GPU 以配合 Dyad 开发环境使用。他们正在考虑是将 GPU 集成到现有设置中，还是需要一台独立的服务器。这引发了关于将高显存 GPU 集成到现有系统中的讨论，考虑了电源、散热以及与当前硬件的兼容性等因素。

  - **[[D] ICLR 新任 ACs —— 进展如何？](https://www.reddit.com/r/MachineLearning/comments/1q67hiq/d_iclr_new_acs_hows_it_going/)** (活跃度: 42): **该帖子讨论了 **ICLR** 新任领域主席 (ACs) 的经历，重点是在没有可靠评审分数的情况下进行决策的挑战。强调的一个关键问题是心理上模拟 rebuttal 过程的难度，因为 ACs 必须在不假设分数变化的前提下，判断作者的回复是否充分解决了评审员的疑虑。正如 ICLR 分享的电子邮件指南中所述，许多 ACs 认为这一过程极具挑战性。** 一位评论者幽默地提到，由于随后的改进，他们希望自己的论文被拒绝，这突显了学术提交的迭代性质以及无法撤回的限制。

    - TheDeviousPanda 强调了 ICLR 领域主席 (AC) 角色中一个具有挑战性的方面，即 ACs 必须预判评审员在阅读作者的 rebuttal 后可能会如何更改评分。这要求 ACs 在心理上模拟 rebuttal 过程，这可能是困难且主观的。该评论表明，许多 ACs 可能并不期望评审员提高分数，这显示出一种维持初始评估的潜在偏见。

  - **[[D] 实验室内部合作](https://www.reddit.com/r/MachineLearning/comments/1q6sgx5/d_intralab_collaborations/)** (活跃度: 9): **该帖子讨论了在临床 AI 环境中平衡非正式技术协助与正式研究合作的挑战。作者是一位具有深厚 ML/AI 背景的医生，经常有同事向他咨询模型选择（model selection）和分析方面的建议，他觉得这已经跨越到了研究合作的范畴。他正在寻求如何将这些互动转化为正式合作的建议，并暗示在他当前的环境中，随意的帮助与共同作者身份（co-authorship）之间的界限是模糊的。** 评论者建议，如果提供的协助对项目至关重要，应建立明确的界限并协商正式的合作条款。他们强调了保护个人时间并确保贡献得到认可的重要性，无论是通过共同作者身份还是其他正式协议。

    - 讨论强调了在实验室内部合作中设定界限的重要性，尤其是当一个人的专业知识经常被寻求时。建议如果贡献重大，应协商反映其贡献的条款，而不是免费提供帮助。这种方法被认为是确保个人研究时间不被削减，并在实验室环境中维持专业关系而非私人关系的必要步骤。

- **[[D] How do i find endorsement to publish preprint on arxiv?](https://www.reddit.com/r/MachineLearning/comments/1q68ues/d_how_do_i_find_endorsement_to_publish_preprint/)** (热度: 8): **用户正在寻求关于如何在 arXiv 上获得提交预印本所需的 endorsement（背书）的指导，这是新提交者的要求。Endorsements 通常可以通过当前或之前的大学隶属关系获得，或者通过与已经在 arXiv 获得背书的共同作者合作获得。需要注意的是，仅为了获得背书而交易作者身份会违反学术诚信，因为共同作者必须对作品有实质性贡献。** 一个显著的观点建议，与能够背书论文的共同作者合作是一个可行的方案，但强调了通过确保共同作者是合法贡献者来维护学术诚信的重要性。

    - 该评论建议通过当前或之前大学的隶属关系，或者通过与可以背书的共同作者合作，来获得 arXiv 预印本提交的 endorsement。它强调，仅为获得背书而交易作者身份违反学术诚信，突出了共同作者真实贡献的重要性。

  - **[Usage update issue?](https://www.reddit.com/r/ClaudeCode/comments/1q6l5hn/usage_update_issue/)** (热度: 202): **该图片突出了 "Claude Code v2.0.76" 软件界面中的一个潜在问题，特别是在 "Usage"（用量）标签页中。订阅计划的用户（如提到的 200 美元计划）在访问其用量数据时遇到困难，因为界面提示 "/usage" 命令仅适用于订阅计划，但其运行并未符合预期。此外，虽然显示了启用额外用量的选项，但用户无法核实当前的用量状态。正如评论所示，此问题似乎影响了多名用户，并且相关的 GitHub issue 引起了大量讨论，表明这可能是一个与促销期后近期用量激增相关的更广泛问题。** 一位评论者指出，Claude Code 和桌面应用都遇到了此问题，并引用了讨论广泛的 GitHub issue，表明这是一个普遍问题。另一位评论者则否定了这一问题，认为一切功能正常，而第三位确认遇到了相同的问题。

    - 有报告称 Claude Code 出现了用量激增问题，尤其是在 "2X week" 活动之后，这导致相关的 GitHub issue 积累了约 250 条评论。这表明这是一个影响多名用户的普遍问题，至少有一人表示正在调查此问题。该问题似乎与意外的用量限制和访问权限变更有关。
    - 几位用户（包括 "100 max plan" 和 "5x Max plan" 用户）正在经历用量限制的意外变化。一位用户注意到他们的限制被提前解除，尽管他们在三天前就已达到每周限制，但现在可以再次使用不同的模型。这表明用量追踪或限制执行系统中可能存在 Bug 或配置错误。
    - 该问题似乎同时影响了 Claude Code 和桌面应用，表明这是一个更广泛的系统性问题，而非孤立事件。不同计划的多名用户报告了类似问题，这指向了可能需要解决的后端或基础设施相关问题。

  - **[https://claude.ai/settings/usage doesn't work?](https://www.reddit.com/r/ClaudeCode/comments/1q6l2x3/httpsclaudeaisettingsusage_doesnt_work/)** (热度: 144): **用户报告了 Claude AI 用量设置页面 (https://claude.ai/settings/usage) 的问题，该页面仅显示额外预算配额，而不显示预期的用量详情。一些用户注意到，尽管之前已达到每周限制，但他们的限制被意外解除，允许他们使用不同的模型。这种异常发生在 `5X Max plan` 上，且重置原定于次日进行。** 一位用户建议彻底“取消‘用量限制’”，表明其更倾向于更灵活的用量政策。

- TheseQuit8175 报告了一个异常情况，其使用限制被意外取消，尽管已达到每周使用限制，仍可使用不同模型。他们提到自己使用的是 '5X Max 计划'，并指出重置原定于次日发生，这表明使用情况追踪系统可能存在问题。
- Gold_Jury_789 讨论了使用配额（usage quotas）可能存在的计算错误，指出在 '20x' 的使用级别下，他们的实际使用量超出预期 15%，而本应低于 10%。他们还提到周日的使用量超出了配额 35%，暗示配额管理系统可能存在 Bug 或配置错误。

---

# AI Discord Recap

> 摘要的摘要之摘要


## Gemini 3.0 Pro Preview Nov-18

**主题 1. NousCoder-14b 与权重开放（Open-Weights）的代码竞赛**

- **NousCoder-14b 横扫奥赛**：Nous Research 发布了 **NousCoder-14b**，这是一个基于 **Qwen3-14B**，使用 Atropos 框架和 48 块 B200 进行后训练（post-trained）的模型，在竞赛基准测试中实现了 **67.87% Pass@1** 的准确率（比基准提升了 7.08%）([发布推文](https://x.com/NousResearch/status/2008624474237923495))。此次发布包含了一个完全可复现的技术栈，其 RL 环境和基准测试详情见其 [博客文章](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/)。
- **Qwen3 的评价褒贬不一**：虽然一些用户认为 **Alibaba** 的 **QW** 在英语方面接近 **AGI** 状态，但也有人报告称，与 **Kimi K2** 或 **DeepSeek** 相比，**Qwen3** 变体在复杂的创意写作方面表现不佳。此外，OpenRouter 的用户注意到 **Qwen3-Next-80B** 的 **TPS** 显著下降，这可能是由于通过 GMICloud 等廉价提供商进行路由导致的 ([状态更新](https://x.com/openrouterai/status/2005707622020964412?s=46))。
- **Claude Code 对比手动工作流**：工程师们正在讨论 **Cursor IDE** 的“正确”用法，提倡使用 `.cursorignore` 和 `.mdc` 文件的 **ETL**（提取、转换、加载）工作流来优化 Context。同时，用户批评了 **Claude Code** 的命名，并演示了 **Claude Opus 4.5** 已经能够自动化执行复杂任务，如从零开始生成 30 秒的视频广告 ([演示推文](https://x.com/deedydas/status/2008747553261842483?s=46))。

**主题 2. 底层内核（Low-Level Kernels）与硬件优化**

- **NVFP4 进入 PyTorch**：工程师们通过修补 layernorms 以在 **nvfp4** 和 **bf16** 之间进行连续转换，成功在 **PyTorch** 中实现了 **NVFP4** 前向传播，从而避免了内核融合（kernel fusion）。讨论强调 **NVFP4** 仍是 Nvidia 的私有技术，而 **MXFP4** 才是具有硬件加速的 FP4 训练的行业标准。
- **高维张量可视化**：一篇分享的 [博客文章](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/) 提出将高维张量绘制为“矩阵的矩阵”，以克服终端显示限制。**GPU MODE** 成员同时也在寻找直接从内存可视化 **f8** 或特定低位布局（low-bit layouts）等二进制格式的工具。
- **Tinygrad 与 AMD 驱动**：在 **AMD Radeon RX 9070XT** 上调试 **tinygrad** 的用户报告称，`VFIO=1` 会触发 `ioctl` 调用中的 TypeErrors，禁用后问题解决。社区还在悬赏征集将调度器替换为 **linearizer** 以保持 GPU 速度的方案，目前 PR 中已有一个潜在的修复方案 ([PR 链接](https://github.com/tinygrad/tinygrad/pull/13780))。

**主题 3. 模型评测（Evals）、排行榜与“感觉（Vibes）”**

- **Gemini 3 Flash 幻觉硬件参数**：**Gemini 3 Flash** 被指责“容易被误导”，它会根据简单的用户提示产生关于 **LFM 2.5** 参数量的幻觉（范围从 8B 到 405B 不等）。尽管存在这种不稳定性，基准测试表明其性能优于 **Gemini Pro** 和 **Grok 4 Heavy**，引发了关于 **后训练（post-training）** 价值与原始规模之争。
- **LMArena Battle Mode 遭到抵制**：用户对 Direct Chat 中新增的 **Battle Mode** 表示不满，理由包括 Context 丢失、每个提示词都会出现激进的 **captchas**（验证码），以及无法禁用该功能。批评者认为排行榜已经成为一种“瘟疫”，而 **Video Arena** 现在已确认严格采用随机模型配对的 Battle Mode。
- **对 DeepSeek mHC 框架的质疑**：一篇 [论文讨论](https://discord.com/channels/714501525455634453/1045297868136779846/1458191879127695391) 认为，**DeepSeek 的 mHC 框架**（将残差映射投影到双随机矩阵上）可能被过度炒作了。批评者认为，真正的洞察在于 **残差混合（residual mixing）** 才是实际的不稳定算子，而非论文中提出的新颖投影框架。

**主题 4. 安全、越狱与隐私**

- **OpenRouter 被黑且 IP 泄露**：一名用户报告其 **OpenRouter** 账户在遭受黑客攻击后资金被清空，引发了关于 **IP exposure policies** 的讨论，其中一些提供商会直接接收用户的 IP（[提供商策略列表](https://openrouter.ai/providers)）。注重安全的成员建议使用临时 Visa 卡，并严格审计提供商的路由选择。
- **Grok 开发者覆盖模式 (Developer Override Modes)**：红队人员正在利用 **Grok** 中的 **DEVELOPER OVERRIDE MODE**，通过安全上下文注入来绕过过滤器，尽管模型通常会以标准的安全性样板回复拒绝。这与“非正式渐强攻击方法 (informal crescendo attack method)”的讨论一致，旨在从 **xAI** 模型中提取不受限制的输出。
- **ChatGPT Health 隐私恐慌**：OpenAI 推出了 **ChatGPT Health** 以整合医疗记录，但其允许研究用途的 [隐私政策](https://openai.com/index/introducing-chatgpt-health/) 引起了警觉。此次发布极具争议，一项研究声称其具有 **90% 的诊断准确率** ([nature 文章](https://www.nature.com/articles/s41746-025-01543-z))，而其他研究则引用仅为 **52.1%** ([反面研究](https://www.nature.com/articles/s41591-024-03097-1))。

**主题 5. 基础设施与本地推理 (Infrastructure and Local Inference)**

- **预量化 MoE 的困扰**：**Unsloth** 用户报告预量化的 **MoE models** 已损坏，迫使用户加载完整模型并即时量化为 **4bit**。这限制了在消费级硬件上的部署，不过新的 [Supertonic CLI 工具](https://huggingface.co/Supertone/supertonic-2) 为 LoRA adapters 提供了无损压缩以缓解存储压力。
- **Vulkan 优先级难题**：硬件爱好者正受困于 **Vulkan** 缺乏优先级划分的问题，这阻碍了 **64GB MI210** 与 **24GB** 显卡的有效协同使用。用户担心在利用大显存显卡的容量之前，小显存显卡就会达到 VRAM 限制，从而使多 GPU 本地设置变得复杂。
- **用于本地 LLM 的 VS Code**：一位开发者发布了一个专门为本地 LLM 优化的自定义 [VS Code 构建版本](https://github.com/bdrazn/codeOSS-LMStudio-Ollama/releases/tag/First-Light)，具有 **LMStudio 支持**和重写的上下文管理系统。该工具声称通过专门针对本地推理限制进行优化，其代码索引和检索速度比主流 AI IDE 更快。


## gpt-5.2


**1. 开源模型、RL 栈与新基准测试**

- **NousCoder-14B 自备工具包**：Nous Research 发布了 **NousCoder-14b** 及其全栈发布包（包括 **RL environment**、**benchmark** 和 **Atropos harness**），详见其文章 [“NousCoder-14b：具有竞争力的奥赛编程模型”](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/)。该模型在 **48 台 B200 上对 Qwen3-14B 进行了 4 天**的后训练，并通过 [@NousResearch on X](https://x.com/NousResearch/status/2008624474237923495) 宣布，其在可验证执行奖励机制下实现了 **67.87% 的 Pass@1**（比 Qwen 提升了 **+7.08%**）。
  - 在多个服务器上，人们认为这次发布值得关注，部分原因是它提供了**训练/评估管道 (training/eval plumbing)**（而不仅仅是权重），并且这引发了关于**标记效率 (token efficiency)** 以及后训练如何决定感知能力（相对于原始规模）的更广泛讨论。

- **Vision Arena 迎来新的前十“闯入者”**：LMArena 用户指出 `ERNIE-5.0-Preview-1220` 在 [Vision Arena 排行榜](https://lmarena.ai/leaderboard/vision) 上以 **1226** 分位居**第 8 名**，详情记录在 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/) 中。
  - 社区注意到一个元信号 (meta-signal)：**Baidu** 是目前 Vision Arena 前十名中唯一的中国实验室，而其他地方的讨论则在质疑竞技场式排名究竟有多大可信度。


**2. 健康 + LLM：产品发布、准确性与隐私考量**

- **ChatGPT Health 想要你的病历（以及你的信任）**：OpenAI 推出了 **ChatGPT Health**，作为 ChatGPT 中的一个专用空间，用于安全地连接**医疗记录**和**健康应用**，可通过 [“介绍 ChatGPT Health”候补名单](https://openai.com/index/introducing-chatgpt-health/) 申请早期访问。
  - 讨论迅速集中在政策措辞对隐私/数据使用的影响，以及 ChatGPT 成为*“全能应用垄断者 (everything app monopoly)”*的风险上，尽管有些人将其视为汇总和验证个人医疗信息的实用层。

- **90% vs 52.1%：医疗准确率大决战**：Perplexity 社区辩论引用了一项发表在《Nature》的研究，报告 **ChatGPT 的诊断准确率为 90%**（见于 [*npj Digital Medicine*](https://www.nature.com/articles/s41746-025-01543-z)），而另一篇《Nature》论文报告的 **准确率为 52.1%**（见于 [*Nature Medicine*](https://www.nature.com/articles/s41591-024-03097-1)）。
  - 工程师们争论这主要是 **数据集/任务框架 (dataset/task framing)** 的差异还是现实世界的可靠性问题，并多次警告称，当患者安全、分诊阈值和部署条件发生变化时，头条新闻中的准确率数字可能会产生误导。


**3. GPU 与内核工具：FP4 格式、Warp Specialization 和 ROCm 追赶**

- **NVIDIA 调整 RTX 参数（Sampling, QKV, MXFP4）**：LM Studio 用户链接了 NVIDIA 的博客文章 ["开源 AI 工具升级加速了 NVIDIA RTX PC 上的 LLM 和扩散模型"](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/)，其中描述了 **GPU token sampling**、**QKV projection concurrency**、**MMVQ kernel 优化**、更快的加载速度，以及 **Blackwell GPU** 上的原生 **MXFP4** 支持。
  - 讨论分为两派：一派对实际的性能提升感到兴奋，另一派则持怀疑态度，认为其中部分内容读起来像 *营销文案*；同时，相关的聊天对比了 **NVFP4 vs MXFP4**，并抱怨缺乏 IEEE FP4 标准。

- **NVFP4 落地 PyTorch（但 TPS 尚未见奇迹）**：Hugging Face 成员报告称，通过修补 **layernorms** 以在 **nvfp4** 和 **bf16** 之间进行持续转换（在没有 fused kernels 的情况下），**NVFP4** 前向传播已在 **PyTorch** 中实现。
  - 早期测试指出，使用 **fp4 transformer engine** 时 **tokens/sec** 意外降低，但人们仍持谨慎乐观态度，认为一旦 kernel fusion 路径成熟，fp4 推理仍能带来正向净收益。

- **Warp Specialization 在 CuTeDSL 中全速运行**：GPU MODE 分享了 ["CuTeDSL 中的 Warp Specialisation"](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/)，展示了将 GEMM 主循环拆分为 **TMA**（将 tiles 复制到 SMEM）和 **MMA**（计算），并利用 CuTeDSL 流水线使 **Blackwell 主循环** 实现 warp-specialized。
  - 另外，贡献者报告称，集成 CuteDSL flex attention 后吞吐量提升了 **~30%**（对比 **H100 fwd** 上的基础 flex attention），并跟踪了后端支持的缺口（例如，**SM100 backward** 已支持，而 **SM90 backward** 仍通过 [flash-attention PR #2137](https://github.com/Dao-AILab/flash-attention/pull/2137) 在开发中）。


**4. 微调、共享和专业数据集的新工具**

- **Supertonic 通过仅传输 Delta 来缩小微调模型体积**：Unsloth 社区宣布了 **Supertonic**，这是一个在 Hugging Face 上的免费 CLI 工具 ([Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2))，它可以计算 **微调模型与其基础模型之间的 delta**（基于 LoRA 的工作流），用于 **训练后的无损压缩**。
  - 人们将其视为一种分发和管理多个微调模型版本的实用方案，无需搬运完整的 checkpoint，这与业界对稀疏/衍生 adapter 分发工具的兴趣相契合。

- **CyberSec CoT 数据集尝试修补“推理差距”**：一名成员在 Hugging Face 上发布了 **BlackBox-CyberSec-CoT-Reasoning-Sample** ([数据集链接](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample))，该数据集使用 **Llama-3-70B** 生成，包含针对 **SOC 事件** 的思维链（Chain-of-Thought）日志。
  - 他们征求关于格式是否对微调 **Llama-3** 真正有用的反馈，这呼应了更广泛的 RL/SFT 讨论，即当推理轨迹并非完全可靠时，应该进行 *“启动 RL 所需的最少量 SFT”*。


**5. 模型网关与用户体验中的可靠性、安全性和平台摩擦**

- **OpenRouter：IP 政策现状调查 + 账户洗劫惨剧**：OpenRouter 用户指出了 [openrouter.ai/providers](https://openrouter.ai/providers) 上的提供商/IP 披露列表，注意到大多数提供商看到的是 **Cloudflare worker IP**，但正如模型页面所述，某些提供商可能会接收到用户的真实 IP。
  - 在同一时期，一名用户报告账户被盗（邮箱被改、信用卡被用于购买额度、原有数据被清空），建议是申诉扣款并使用 **临时卡 (throwaway cards)** —— 这提醒人们，模型路由的便利性伴随着实际的操作风险。

- **Qwen TPS 在 12 月 28 日后暴跌，所有人都在“归咎于路由器”**：OpenRouter 用户观察到开源模型的 **TPS 下降**（尤其是 **Qwen3-Next-80B-a3b-Instruct**），并参考了 [OpenRouter Status 的 X 帖子](https://x.com/openrouterai/status/2005707622020964412)中的更新。
  - 目前的推测归咎于路由到了最便宜的提供商（**GMICloud**），用户建议检查 **Activity 选项卡** 以对比不同提供商的速度，而不是假设模型本身发生了退化。

- **对战模式 + 验证码：LMArena 用户在与 UI 斗争，而非模型**：LMArena 用户抱怨 **Battle Mode** 总是自动介入 **Direct Chat**，导致上下文丢失和生成时间过长，而即便在低频使用下也会触发频繁的 **captcha** 提示。
  - 一位团队成员表示 **captcha** 针对的是“非真实使用”，并建议放慢提示速度，但用户力推增加明确的禁用开关和更清晰的频率限制错误提示，而不是突如其来的模式切换。


## gpt-5.1


**1. 新型高技能模型与训练栈**

- **NousCoder-14B 突击训练攻克奥数难题**：**Nous Research** 发布了 **NousCoder-14B**，这是一个具有竞争力的奥赛编程模型，在 **Qwen3-14B** 基础上进行了后训练，并采用了在 **Atropos** 中构建的 RL 栈。详情见其博客文章 [“NousCoder-14b: A Competitive Olympiad Programming Model”](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/)。他们在 **48 台 B200** 上训练了 4 天，在代码基准测试中达到了 **67.87% Pass@1**——比原始 Qwen 提升了 **+7.08%**。正如其 [X post](https://x.com/NousResearch/status/2008624474237923495) 中所强调的那样，该模型使用了可验证的执行奖励。
  - **Nous** 和 **Latent Space** 的讨论强调，该发布包含了一个全栈、可复现的 RL 环境、基准测试和 harness，使其成为了一个罕见的公开端到端代码 RL 栈，而不仅仅是发布一个模型。工程师指出，这种开放、可验证的训练流水线使得比较 RL 奖励方案和突破简单的 pass@k 排行榜变得更加容易。

- **Unsloth 将 Nemo 转化为小型 Opus 式思维模型**：一位社区成员使用 **Unsloth** 将 **Mistral-Nemo-Instruct-2407 12B** 转换为一个专注于推理的模型，将其与 **Claude Opus 高推理轨迹**进行对齐，并将其发布为 [“Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning”](https://huggingface.co/DavidAU/Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning)。在运行了 heretic 流程并添加了 Opus 风格的思考头（thinking heads）之后，他们还发布了一个 **Heretic Uncensored** 变体 [“MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored”](https://huggingface.co/DavidAU/MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored)。
  - 作者报告称，**Claude Opus 4/4.5** 推理数据集产生了“紧凑但高质量”的思考块，在没有大规模扩展的情况下，有效地将一个富有创造力的 12B 模型转变为多步推理器。**Unsloth 示例**频道中的其他用户现在将这些视为中型模型“思维转换”的模板，表明了向强大但小型的骨干网络添加蒸馏推理的趋势。

- **扩散 LLM 令研究人员感到欣喜，尽管细节尚少**：在 **Nous Research** 的 research-papers 频道中，一名成员分享了论文 [“Diffusion LLMs”](https://arxiv.org/abs/2511.08923)，表示他们喜欢扩散风格的语言模型，因为它们“看起来更有趣”。该论文提议对文本使用类扩散的生成过程，与标准的自回归 Transformer 形成对比。
  - 虽然技术讨论很简短，但链接和反应显示出人们对 **非自回归、基于扩散的 LM** 作为未来扩展的一种严肃替代方案的日益好奇。工程师表示，他们想了解这些架构是否能比目前的 Transformer 解码器提供更好的 **模式覆盖（mode coverage）、并行性或可控性**。


**2. GPU 系统、内核与底层性能调优**

- **CuTeDSL Warp 特权化（Warp-Specialization）加速 GEMM**：在 **GPU MODE** 的 nvidia-competition 频道中，一名成员分享了博客文章 [“Warp Specialisation in CuTeDSL”](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/)。该文章利用 CuTeDSL 的流水线抽象，将 GEMM 主循环拆分为 **TMA（磁贴拷贝至 SMEM）** 和 **MMA（矩阵乘法）**。他们报告称，通过将普通的非持久性 **Blackwell** 主循环转换为 Warp 特权化版本，获得了显著的吞吐量提升。
  - 另外，在 **GPU MODE ▷ #torch** 频道，另一位工程师感谢贡献者集成了 **CuteDSL flex-attention**，并引用了在 **H100 前向传播**上比基础 flex attention 提升了 **~30% 的吞吐量**（根据 [flash-attention 接口](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938)），并且正在通过一个 [pull request](https://github.com/Dao-AILab/flash-attention/pull/2137) 为 **SM90** 提供完整的反向支持。

- ****NVFP4 和 FP4 引擎混淆了性能预期****：在 **HuggingFace** 服务器上，用户确认可以通过修补 layernorm 以在 **nvfp4** 和 **bf16** 之间转换，从而在 **PyTorch** 中运行 **NVFP4**，但注意到 kernel 尚未融合（fused）。尽管使用了 **fp4 transformer engines**，他们观察到的 **tokens-per-second** 低于预期，这引发了关于 FP4 究竟在何种实际场景中胜出的疑问。
  - 在 **LM Studio** 及相关聊天中，这引发了将 **NVFP4** 与更标准化的 **MXFP4** 进行对比的更广泛讨论。人们指出目前并没有 **IEEE FP4** 标准，且 NVFP4 是一种 **NVIDIA 专有格式**。工程师们得出结论，FP4 在目前的 *inference*（推理）中可能比训练更有价值，但前提是 kernel 堆栈和硬件路径经过深度的端到端调优。

- ****Helion, Iris 和 ROCm 助力开放 GPU 系统****：**GPU MODE ▷ #helion** 频道宣布，来自 **AMD 的 Umesh** 正在积极地在 **ROCm** 上启用 **Helion** 编译器堆栈，审计被跳过的单元测试和损坏的示例，并专注于 **GEMM 性能加速**。社区成员对此表示欢迎，并明确要求支持 **MI400 系列**，因为他们正在扩大 AMD 机群。
  - 在 **GPU MODE ▷ #job-postings** 中，[Iris 项目](https://github.com/ROCm/iris/)（一个基于 **Triton 的多 GPU 编程框架**）正在招聘具有 **Triton、多 GPU 编程、RMA/RDMA 以及底层 GPU 通信**经验的美国实习生。这些讨论共同展示了一股协同推力，旨在使 **AMD + Triton/Helion** 成为 CUDA 在高性能 kernel 领域的一流开放替代方案。


**3. 硬件经济学、性能意外与路由难题**

- ****Nvidia GPU 淘金热引发“标价冲击”模拟****：在 **LM Studio** 中，成员们剖析了一份 [TrendForce 报告](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/)，该报告称 **GeForce RTX 5090** 到 2026 年价格可能达到 **5000 美元**，同时还有昂贵的 **Nvidia DGX Station** 机架和类似的 AMD 系统。工程师们开玩笑说，未来拥有 **128 GB VRAM** 的消费级显卡将仅够给每个 Chrome 标签页分配 **32 GB**，但深层的担忧在于，即使是在即将推出的 **288 GB VRAM** 数据中心组件上，最先进的模型依然无法轻松适配。
  - 在 **Perplexity** 和其他地方，人们将此与本地组装机 **RAM 价格**上涨联系起来，并猜测中国和印度的制造商（如 **Tata**、**Reliance**）进入 DRAM 领域以缓解成本压力。共识是，硬件稀缺和供应商定价，而不仅仅是算法，现在正成为参与前沿规模训练的门槛。

- ****GB10 GPU 和 Qwen 路由揭示现实世界的性能陷阱****：在 **LM Studio** 的硬件讨论区，测试人员称 **GB10** GPU “太慢了”，实测比 **RTX 6000 Ada** 慢约 **6 倍**（尽管显存更多），并将其与 NVIDIA 的 [DGX Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/) 进行了对比，后者也显示出同样平庸的吞吐量。用户还警告说，幻觉仍然是一个**模型问题，而非 VRAM 问题**，反驳了更多显存就能修复模型质量的观点。
  - 在 **OpenRouter** 上，用户发现 **Qwen3-Next-80B-a3b-Instruct** 的 **TPS** 在 **12 月 28 日**之后大幅下降，OpenRouter 在其 [X 平台的状态公告](https://x.com/openrouterai/status/2005707622020964412)中承认了这一点。速度变慢似乎与路由到最便宜的供应商（**GMICloud**）有关，工程师们被建议检查 **Activity 标签页**并明确选择更快的供应商，而不是信任自动路由。

- ****RAM、VRAM 和 Vulkan 优先级阻塞了升级路径****：在 **LM Studio** 中，一名试图扩展到 **24 GB** 显卡以上的成员在 **2×24 GB**、**48 GB**、**64 GB MI210** 或 **4090** 之间纠结，称在成本与容量的博弈下，所有选项都像是“台面上仅剩的错误选择”。另一名成员报告了 **Intel 第 13 代 CPU** 上的大规模不稳定性，建议其他人检查 **Windows 事件管理器**中可能由近期 Windows 更新触发的问题。
  - 他们还指出 **Vulkan 缺乏优先级拆分 (priority splitting)** 是一个实际问题，特别是当将 **64 GB MI210** 与多张 **24 GB** 显卡混用时，因为小显卡可能在大显卡饱和之前就成了瓶颈。在 **Perplexity** 和各硬件频道中，类似的资源限制——高昂的 DRAM 价格、稀缺的大容量 VRAM 显卡以及不成熟的调度 API——正在塑造团队推进本地和混合部署的激进程度。


**4. 医疗保健中的 AI 与隐私：效能 vs 风险**

- ****ChatGPT Health 进军医疗工作流，隐私问题随之而来****：**OpenAI** 宣布了 **ChatGPT Health**，这是 ChatGPT 中一个专门的健康空间，允许用户安全地连接 **medical records** 和 **wellness apps**，以帮助他们 *“导航医疗护理”*，正如其博客文章 [“Introducing ChatGPT Health”](https://openai.com/index/introducing-chatgpt-health/) 所述。该工具明确指出它 **不能替代专业的医疗建议**，而是将回复建立在个人健康数据的基础上，目前已开启 [早期访问候补名单](https://openai.com/index/introducing-chatgpt-health/)。
  - 在 **OpenAI**、**Yannick Kilcher** 和 **Latent Space** 的 Discord 频道中，工程师们立即提出了 **隐私和锁定（lock-in）担忧**，并指出该政策允许使用健康对话来 *改进服务和进行研究*。一些人担心这可能使 ChatGPT 成为健康领域的 *“超级应用垄断 (everything app monopoly)”*，特别是与 Google 开源的 **MedGemma** 相比，并讨论了此类敏感数据是否应该反馈到模型训练中。

- ****LLM 诊断研究数据两极分化，引发争议****：在 **Perplexity AI** 中，用户分享了 **Nature Digital Medicine** 的一项研究，其中 **ChatGPT** 在受控环境下达到了约 **90% 的诊断准确率**，链接指向 [“Evaluating ChatGPT’s diagnostic performance”](https://www.nature.com/articles/s41746-025-01543-z)。其他人则引用了第二篇 **Nature Medicine** 论文 [“Performance of large language models in clinical diagnosis”](https://www.nature.com/articles/s41591-024-03097-1) 进行反驳，该论文显示准确率仅为 **52.1%**，并警告了患者安全风险。
  - 辩论达成了一个共识：**benchmark cherry-picking**（基准测试择优挑选）会严重误导临床安全评估，LLM 必须被视为决策辅助工具，而非自主诊断者。工程师强调需要 **严格的、针对特定任务的评估 (evals)**、偏差审计和明确的护栏，特别是随着像 **ChatGPT Health** 这样的工具开始摄取真实的医疗记录。

- ****利用 Llama-3 CoT 日志解决网络安全“推理鸿沟”****：在 **Unsloth** 的研究频道中，一位从事 SOC 工具开发的从业者描述了现成网络安全模型中的 *“推理鸿沟 (Reasoning Gap)”*，并开始使用 **Llama-3-70B** 生成结构化的 **Chain-of-Thought (CoT) 事件日志**。他们发布了一个公开样本数据集 [“BlackBox-CyberSec-CoT-Reasoning-Sample”](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample)，以征求关于微调格式的反馈。
  - 社区讨论围绕显式的 **CoT traces** 是否比单纯更好的数据和工具更能实质性地改进网络安全事件分拣展开，几位成员强调标注质量优于日益复杂的 RL 设置。这反映了 Unsloth 和 Eleuther 之前的观点，即在尝试弥补特定领域的推理差距时，**高质量、针对特定任务的数据（例如 Ultra-FineWeb 和 [sumthink](https://huggingface.co/datasets/G-reen/sumthink)）通常优于巧妙的算法调整**。


**5. Creative Companions, Vision Tools, and Evaluation Backlash**

- ****语音和视觉伴侣点击量达到真实使用规模****：在 **Latent Space** 的 genmedia 频道中，来自 **Tolan** 的 Paula 宣布他们的 **语音优先 AI 伴侣** 月活用户突破 **200,000**，并在 [X thread](https://x.com/paularambles/status/2008964509810278413?s=46) 中分享了实现细节和经验教训。与此同时，**Razer** 预告了 [“Project AVA”](https://xcancel.com/razer/status/2008543615916666928?s=46)，这是一款拥有 **5.5 英寸屏幕的 AI 伴侣**，具有先进的推理能力和个性化、可更换皮肤的虚拟形象（从电竞传奇到动漫角色），预计将于 **CES 2026** 发布。
  - 工程师们认为这两者都标志着 **实时、持久的 AI 伴侣** 正在离开 Demo 阶段，成为具有严峻基础设施和延迟限制的消费级产品。人们对这些系统如何编排 **multi-modal input**、**streaming TTS** 和 **memory**，以及它们与 **OpenAI** 等提供商（Tolan 明确称其为紧密合作伙伴）的耦合程度表现出浓厚兴趣。

- ****相机控制 LoRA 赋予艺术家导演级的权力****：Fal 发布了一个针对 **Qwen-Image-Edit-2511** 的开源且更强大的**多视角相机控制 LoRA**，其详情记录在他们的[发布推文](https://xcancel.com/fal/status/2008954582018248755?s=20)中。该 LoRA 允许用户指定透视和构图——**前/后/侧视图、低/高角度以及不同的拍摄距离**——通过精确的相机控制重新组合现有图像。
  - **Latent Space** 的创作者们认为这是迈向**可提示电影摄影（promptable cinematography）**的一大步，在图像编辑工作流中可以将“内容”与“相机”分离。结合早期的广告生成工作（例如 Deedy 的 [“Claude Code” Hermès 风格 30 秒广告](https://x.com/deedydas/status/2008747553261842483?s=46)），共识是相关工具正迅速让小团队具备完整的**剧本 → 分镜 → 镜头级控制**能力。

- ****Arena 排行榜和基准测试引发社区质疑****：**LMArena** 更新了其 **Vision Arena 排行榜**，将 `ERNIE-5.0-Preview-1220` 推至**第 8 位，得分为 1226**，具体见 [Vision 排行榜](https://lmarena.ai/leaderboard/vision)和[更新日志](https://news.lmarena.ai/leaderboard-changelog/)。与此同时，一篇广为流传的批评文章 [“LM Arena 是 AI 的瘟疫”](https://surgehq.ai/blog/lmarena-is-a-plague-on-a/) 再次在 **Unsloth** 和 **Latent Space** 引起讨论，文章认为在顶级模型接近人类水平的情况下，人为投票对决会扭曲激励机制。
  - 几位工程师对排名表现得不以为然，其中一人表示 *“我不认识有谁真的在意 lmarena 的排名”*，并指出许多**中国模型甚至不再引用 LM Arena 的评分**。社区情绪正向**基于任务的、可重复的评估（evals）**转变，远离侧重“氛围感”的竞技场对决，特别是当公司利用这些数据进行营销而非工程改进时。


## gpt-5


**1. 新的编程模型与视觉排行榜**

- **NousCoder-14b 攻克奥林匹克竞赛任务**：**Nous Research** 推出了 **NousCoder-14b**，这是一款具有竞争力的奥林匹克编程模型，在 [NousCoder-14b: A Competitive Olympiad Programming Model](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/) 中详细介绍了全栈发布内容（RL 环境、基准测试、Atropos 框架）。该团队报告称，在 **Qwen3-14B** 基础上使用 **48 块 B200 运行 4 天**进行后训练，实现了 **Pass@1 67.87%** 的成绩（比 Qwen 提高 7.08%），并带有可验证的执行奖励。
  - 他们在 X 平台的 [NousCoder-14b Pass@1 更新](https://x.com/NousResearch/status/2008624474237923495)中再次确认了结果，强调 **Atropos** 框架和 **Modal** 自动缩放器是核心基础设施。工程师们赞扬了该可重复的训练栈，并强调了强后训练信号对代码任务的重要性。

- **ERNIE-5.0 在 Vision Arena 排名攀升**：`ERNIE-5.0-Preview-1220` 在 [Vision Arena 排行榜](https://lmarena.ai/leaderboard/vision)上以 **1226** 的得分位列**第 8**，在顶级视觉模型中表现亮眼。[排行榜更新日志](https://news.lmarena.ai/leaderboard-changelog/)指出，**百度**是前 10 名中唯一的中国实验室。
  - 社区观察者将这一排名上升视为 **ERNIE-5.0** 视觉推理能力成熟的信号，并呼吁进行更多面对面的评估。他们还呼吁 Arena 定期更新，以追踪影响排行的快速平台端变化。

- **随着 xAI 完成 E 轮融资，Grok 5 正在训练中**：**xAI** 宣布了其 [E 轮融资](https://x.ai/news/series-e)并确认 **Grok 5** 正在训练中，标志着其 **LLM** 系列将持续扩展。这一更新为 xAI 扩大算力和加速下一代 **Grok** 模型迭代奠定了基础。
  - 工程师们预计，如果训练运行按预定规模完成，其能力将实现阶跃式提升，但他们指出，在编程、推理和安全性方面的持续评估才能说明真实情况。社区将其视为加速的 **frontier-model** 军备竞赛中的又一个数据点。


**2. 内核与推理加速**

- **NVIDIA 加速 RTX AI 技术栈**：**NVIDIA** 在 [Open-source AI tool upgrades speed up LLM and diffusion models on NVIDIA RTX PCs](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/) 中详细介绍了开源 AI 工具升级，以加速 **RTX PC** 上的 **LLM** 和 **扩散模型**。亮点包括 **GPU Token 采样**、**QKV 并发**、**MMVQ 内核优化**、更快的模型加载以及 **Blackwell GPU** 上的原生 **MXFP4** 支持。
  - 开发人员预计，在重解码负载中吞吐量将有明显提升，并且在消费级 RTX 设备上模型部署将更加顺畅。讨论集中在这些内核集成到实际应用后能减少多少端到端延迟。

- **Warp 魔法：CuTeDSL 特化 Mainloop**：在 [Warp Specialisation in CuTeDSL](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) 中，一篇关于 **warp specialization** 的深度解析介绍了如何利用 CuTeDSL 的流水线技术，将 **GEMM** mainloop 拆分为 **TMA**（tiles 传输至 SMEM）和 **MMA**（矩阵乘法）。该技术将非持久化（non-persistent）的 **Blackwell** mainloop 转换为 warp-specialized 模式，以获得更高的吞吐量。
  - 从业者认为这是一种整洁的模式，无需编写复杂的底层 kernel 即可榨取更多性能。该文章让工程师在优化 attention 和 matmul 热路径（hot paths）时的复现变得简单直接。

- **Flex Attention 获得 CuTe 加持：H100 迎来性能提升**：工程师报告称，在集成 **CuTeDSL flex attention** 后，**H100 forward** 的吞吐量提升了约 **30%**（参见 [flash-attention 接口参考](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938) 及相关的 [PR 讨论](https://github.com/Dao-AILab/flash-attention/pull/2137)）。**SM100** 与 **SM90** 之间 backward 支持的差异仍是目前的关注重点。
  - 社区正在对这些接口进行标准化，以便在不使用定制 kernel 的情况下解锁各种 masking 模式的加速。目前的工作旨在弥补旧版 SM 上 backward 路径的差距，同时保留 forward 的性能优势。


**3. 微调、压缩与检索工具**

- **Supertonic 缩小 Delta，共享微调成果**：基于 **LoRA adapters** 衍生的免费 CLI 工具 [Supertone/supertonic-2](https://huggingface.co/Supertone/supertonic-2) 发布。它通过导出微调 delta 与基础模型的差异，实现了训练后的无损压缩。这使得 **fine-tuned** checkpoint 的分发和存储更加紧凑。
  - 开发者称赞这种基于 delta 的格式在团队间的可复现性和共享性。他们强调，在多实验工作流中，这简化了 artifact 管理并加快了模型切换速度。

- **Qdrant 通过混合查询（Hybrid Queries）混合信号**：**Qdrant** 在 [Hybrid queries in Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/) 中记录了可组合的 **hybrid queries**，支持向量、关键词和元数据过滤的结合。该概念针对需要大规模多信号评分的检索场景。
  - 从业者警告说，堆砌功能可能会降低存储、内存、计算和延迟方面的效率。团队建议采用阶段性发布和性能分析（profiling），以证明生产流水线中每个算子的合理性。

- **vLLM 按需遵循规则**：vLLM 的结构化解码功能已在 [Structured outputs in vLLM](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html) 中记录，支持 Schema 约束的生成。高级用户希望在特定哨兵标记（如 `</think>`）之后才启动条件约束，以便模型可以先进行“思考”。
  - 工程师们正在实验延迟约束挂钩（hooks）和 **KV-caching** 模式，以保留推理过程并强制执行 Schema。目标是在保证正确性的前提下，平衡高质量的中间思考过程。


**4. 医疗保健领域的 LLM：产品与证明**

- **ChatGPT Health 连接数据，明确范围**：**OpenAI** 推出了 **ChatGPT Health**，这是一个用于安全连接**医疗记录**和**健身 App** 的专用空间，详见 [Introducing ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/)。该产品将回复基于个人健康数据，并明确声明其“不能替代专业的医疗建议”。
  - 工程师们讨论了符合 **HIPAA** 要求的集成、可审计性和同意流，认为这些是严肃采用的必备条件。隐私担忧引发了对明确数据保留政策和沙盒评估环境的呼声。

- **研究称：ChatGPT 诊断准确率达到 90%**：最近的一篇 **Nature** 论文报告称，**ChatGPT** 在受控设置下的诊断准确率达到了 **90%**：[关于 ChatGPT 诊断准确率的 Nature 研究](https://www.nature.com/articles/s41746-025-01543-z)。这一结果再次引发了关于 LLM 在何处可以增强分诊和决策支持的辩论。
  - 具有临床意识的工程师敦促进行细致的外部验证、数据集透明化和鲁棒的校准（calibration）。他们强调，部署环境、 Prompt 设计和护栏（guardrails）会极大地影响最终结果。

- **反面观点：另一项研究发现准确率为 52.1%**：另一项 **Nature Medicine** 研究报告了 **52.1%** 的准确率，强调了风险和变异性：[关于 LLM 诊断准确率的 Nature Medicine 研究](https://www.nature.com/articles/s41591-024-03097-1)。该研究突出了与专家医生相比的差距以及潜在的安全问题。
  - 团队提倡在临床使用前进行严格的 **A/B testing**、不良事件跟踪和人机协作（human-in-the-loop）审查。社区将随机对照试验和部署后监测视为必不可少的步骤。


**5. 基础设施可靠性、定价与安全**

- **RTX 5090 价格恐慌：5000 美元指日可待？**：一份 [TrendForce 关于 GPU 定价的报告](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/) 表明，**GeForce RTX 5090** 在 2026 年可能达到 **5000 美元**。在硅片价格上涨的背景下，工程师们权衡了本地推理与托管 API 的成本折衷。
  - 开发者预计在训练/微调中会更重度地依赖共享集群和竞价实例（spot capacity）。讨论还考虑了混合机群（消费级 + 数据中心 GPU），以平衡 **VRAM** 需求和吞吐量。

- **假日后遗症：OpenRouter 上的 Qwen TPS 减慢**：根据 [OpenRouter 状态更新 (X)](https://x.com/openrouterai/status/2005707622020964412)，用户观察到 12 月 28 日后，**Qwen3-Next-80B-a3b-Instruct** 等开源模型的 TPS 下降。报告将减速归因于通过更便宜的提供商进行路由，并建议在 **Activity** 选项卡中比较提供商的速度。
  - 实践者建议为延迟敏感型工作负载锁定（pinning）更快的提供商。团队还建议跟踪每个提供商的 P50/P95 延迟，以避免高峰需求期间的性能退化。

- **账户被盗引发 OpSec 和 IP 审查**：一名 **OpenRouter** 用户报告了账户被盗、信用卡被盗刷以及数据被抹除；成员引用了 [OpenRouter 提供商与 IP 政策](https://openrouter.ai/providers)，显示某些提供商会接收用户的真实 IP。该事件提醒人们要定期轮换密钥、启用 2FA 并监控账单。
  - 具有安全意识的用户建议使用虚拟/一次性卡以及最小权限的 API 使用方式。他们还建议在选择提供商之前，审核模型页面关于 IP 处理的详细信息。


---

# Discord: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **预量化模型问题频发**：成员报告加载预量化的 **MoE 模型** 存在问题，需要加载完整模型并进行即时的 **4bit** 量化。
   - 社区建议探索替代量化方法，或等待更好地支持预量化 MoE 模型的更新库。
- **Gemini 3 Flash 高估模型大小**：**Gemini 3 Flash** 在简单提示后估计 **LFM 2.5 1.2B** 的参数量在 8B 到 405B 之间，表现出不一致性。
   - 成员一致认为，这些结果凸显了 **Gemini 3 Flash** 作为模型能力可靠评判者的局限性。
- **免费无损微调工具发布**：一名成员发布了一个名为 [Supertonic](https://huggingface.co/Supertone/supertonic-2) 的免费 CLI 工具，该工具衍生自 **Lora 适配器**，在训练后具有无损压缩功能。
   - 该工具计算微调模型与基础模型之间的差值（delta），使其更易于共享和存储，并减小了文件大小。
- **Qdrant 宣传混合查询能力**：一名成员分享了 [Qdrant 关于 **混合查询** 的文档](https://qdrant.tech/documentation/concepts/hybrid-queries/)，强调了结合多种查询类型的能力。
   - 他们警告说，增加更多功能可能会导致存储、内存、计算和延迟方面的效率下降。
- **网络安全领域的推理差距**：一名成员正在利用 **Llama-3-70B** 生成 SOC 事件的 **Chain-of-Thought** 日志，以解决网络安全模型中的“推理差距”（Reasoning Gap），并在 Hugging Face 上发布了 [免费样本](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample)。
   - 他们正在寻求社区关于该格式对微调 **Llama-3** 是否有用的反馈。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Invidious 阻止追踪器，倡导者欢欣鼓舞**：一名成员推荐使用 [Invidious](https://redirect.invidious.io/) 作为 **YouTube** 前端来阻止追踪器，强调了对 **YouTube 数据收集** 行为的担忧。
   - 该成员断言 *YouTube* 在单个参数上有 7 个追踪器，但 Invidious 阻止了所有追踪器。
- **Grok 和 Gemini 面临 Jailbreak 尝试**：成员们正积极为 **Grok** 和 **Gemini** 寻找有效的 Jailbreak 方法，一位用户提到了针对 **Grok** 的*非正式 Crescendo 攻击方法*，而另一位用户则询问了针对 **Bing** 的 Jailbreak。
   - 一位用户分享了针对 **Grok** 的 **DEVELOPER OVERRIDE MODE**，其特点是通过安全上下文注入来绕过安全层和内容过滤器，旨在实现不受限制的输出；此外还有一个类似的针对 **Gemini** 的覆盖模式，名为 **Gemini-DEV-OVERRIDE-2026**。
- **AI 学习斯瓦希里语，获得 Red Team 优势**：一位成员回应称，AI Red Team 的目的是暴露弱点，并建议尝试使用资源较少的语言（如斯瓦希里语或纳瓦霍语）编写 Prompt。
   - 他们表示，在斯瓦希里语或纳瓦霍语等语言中，*“LLM 表现挣扎，甚至 Guardrails 会变弱”*，这让人意识到它对使用俄语互动的用户表现较弱，认为这对其他参与者不公平。
- **微软工程师使用 AI 进行 Vibe Coding 记事本应用**：一位成员分享了一个 [YouTube 视频](https://youtu.be/bmBd39OwvWg)，视频中一位 **Microsoft** 工程师使用 **Claude** 进行 **Vibe Coding** 开发了一个新的**记事本应用**，并引用道：*“是 AI 毁了记事本，所以也要用 AI 来修复它”*。
   - 另一位成员讽刺地说：*“这家伙竟然以高管的身份说 Vibe Coding”*。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Battle Mode 困扰用户**：用户对在 **Direct Chat** 中引入 **Battle Mode** 表示沮丧，理由是存在丢失上下文和生成时间过长等问题，并请求提供禁用该模式的方法。
   - 一些用户报告称，*“每隔一条消息就会变成 Battle Mode”*，并且他们正经历不断的干扰。
- **Captcha 捕捉 Prompt 频率**：用户报告称，即使 Prompt 频率很低，也会频繁出现 **Captcha** 提示，有些用户甚至表示*每次 Prompt* 都会遇到验证码。
   - 一名团队成员表示，**Captcha** 系统旨在检测非真实使用，并建议用户降低 Prompt 频率。
- **Movement Labs 模型：奇迹还是海市蜃楼？**：成员们对 **Movement Labs AI 模型** 展开辩论，一些人称赞其能为 **Minecraft** 克隆版和**象棋游戏**生成功能代码。
   - 另一些人指责其为*“骗局”*，理由是过去涉及退款欺诈计划的争议和纠纷，暗示需保持谨慎。
- **ERNIE-5.0 获得顶级排名**：`ERNIE-5.0-Preview-1220` 以 **1226** 的分数在 [Vision Arena 排行榜](https://lmarena.ai/leaderboard/vision) 上达到 **第 8 名**。
   - [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/) 指出，**Baidu** 是唯一一家进入前 10 名的中国实验室。
- **Video Arena 实验：转瞬即逝……**：成员们询问了网站上 **Video Arena** 的状态，注意到 **Video 按钮** 出现又消失了。
   - 一名团队成员澄清说，网站上的 **Video Arena** 是实验性的，其可用性是随机的，并确认 **Video Arena** 将仅限于 2 个随机模型的 Battle 模式。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Nvidia 的 AI 机架引发价格争议**：成员们发现 **Nvidia 的 AI 机架**和 **AMD 的 AI 机架**对个人用户来说价格昂贵，但仍然可以间接使用，讨论中提到了 [Nvidia DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/)。
   - 他们引用了 [TrendForce 的一篇文章](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/)，该文章暗示 **GeForce RTX 5090** 到 2026 年可能达到 5000 美元。
- **Nvidia 开源 AI 工具获得更新**：**Nvidia** 宣布对其开源 AI 工具进行更新，提升了 **LLM** 和 **diffusion models** 在 **RTX PCs** 上的性能，详情见 [Nvidia 博客文章](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/)。
   - 更新重点包括 **GPU token sampling**、**QKV projections 的并发处理**、**MMVQ kernel 优化**、更快的模型加载速度，以及 **Blackwell GPUs** 上的原生 **MXFP4** 支持。
- **本地 IDE，而非 Zed**：在 **Zed IDE** 出现问题后，成员们讨论了可与本地模型配合使用的 IDE 替代方案，推荐了 **kilocode**、**roocode** 和 **cline**。
   - 此讨论源于一位用户将 **Zed IDE** 描述为 *一团糟 (clusterf-)*。
- **GB10，令人失望的 GPU**：成员们称 **GB10** 在测试中 *极其缓慢*，尽管拥有大量内存，但速度比 **RTX Pro 6000** 慢 **6 倍**。
   - 进一步的讨论指向了 [Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/)，它是另一款具有类似性能特征的设备。
- **Vulkan 的优先级困扰**：成员们就 **Vulkan** 缺乏 **priority splitting**（优先级拆分）展开辩论，这涉及到如何将 **64GB MI210** 与现有的 **24GB 显卡**有效地协同使用。
   - 令人担忧的是，在 **48-64GB 显卡**被充分利用之前，其他显卡的 **24GB 限制**可能已经触顶。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 通过新的医疗集成关注健康**：**OpenAI** 推出了 **ChatGPT Health**，这是 **ChatGPT** 中的一个专门空间，用户可以在其中安全地连接 **医疗记录 (medical records)** 和 **健康应用 (wellness apps)**，旨在帮助用户 *引导医疗护理*，并邀请感兴趣的用户 [加入早期访问等待名单](https://openai.com/index/introducing-chatgpt-health/)。
   - 该工具明确声明 *不取代专业的医疗建议*，旨在确保回答基于个人健康信息。
- **ElevenLabs 克隆声音，政治敏感者除外**：频道用户报告称，[ElevenLabs](https://elevenlabs.io/) 允许他们克隆几乎任何声音，除了 *敏感的政治人物声音*。
   - 一位用户之前曾使用该平台制作了 *幽灵欧比旺 (ghost obi-wan)、HAL9000、奥布瑞·普拉扎 (aubrey plaza)* 的声音，但在停止付费后丢失了 100 万储备积分。
- **绕过 OpenAI 禁令是高风险行为**：澳大利亚用户可以绕过 **OpenAI** 禁令，通过 [ElevenLabs](https://elevenlabs.io/) 等第三方访问 **Sora 2**。
   - 一些成员并不担心账号被封，而另一些人则警告说，使用 VPN 规避地理限制可能会导致你的 **OpenAI** 账号被封禁。
- **伦理框架 A/B 测试，提示词工程师的好伙伴**：一位成员提倡通过使用 **A/B testing** 和 **ablations**（消融实验）来揭开 **LLM** 中伦理框架的神秘面纱，以识别操作组件并提高透明度。
   - 他们认为，目前神秘的提示词可能存在许多语言和结构上的等效方案，其表现可能持平或更好，因此强调在提示词工程 (prompt engineering) 中进行 **大规模 A/B 测试** 的必要性，以识别有效且透明的提示词，因为 **AI failure modes** 通常只在大规模运行下才会显现。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 3 Pro 文件生成失误**：用户报告 **Gemini 3 Pro** 的文件生成功能不稳定，有时会失败并显示消息：*“我无法生成直接下载的文件。”*
   - 一些成员对 *“这些 Bug 爱不释手！”*，而另一些人则在 **Perplexity Pro** 订阅方面遇到了问题。
- **Perplexity Pro 暂停引发恐慌**：多名用户报告其 **Perplexity Pro** 订阅意外暂停，即使是像 **Airtel** 这样的促销订阅也需要提供支付方式。
   - 一篇 [gadgets360.com 的文章](https://www.gadgets360.com/ai/news/how-to-keep-your-free-perplexity-pro-on-airtel-new-card-requirement-explained-9870744) 解释了针对 **Airtel** 用户的新持卡要求。
- **LLM 在医疗领域的表现引发热议**：在一位用户分享了表明 **ChatGPT** 在一项研究中达到 **90%** 诊断准确率的 [研究报告](https://www.nature.com/articles/s41746-025-01543-z) 后，引发了关于 **LLM** 在医疗保健领域应用的争论。
   - 然而，其他成员对依赖 **LLM** 进行医疗保健表示担忧，并引用了 [另一项研究](https://www.nature.com/articles/s41591-024-03097-1)，该研究显示准确率仅为 **52.1%**，并强调了对患者安全的潜在风险。
- **RAM 价格上涨令人沮丧**：成员们讨论了高昂的 **RAM** 价格如何影响装机计划，一位用户建议中国制造商可能会降低成本。
   - 讨论触及了印度公司如 **Tata** 和 **Reliance** 可能进入 **RAM** 制造领域，这可能会在未来降低价格。
- **Sonar 模型拒绝 AWS Presigned URL**：一位成员报告说，虽然标准的公共图像 URL 在 **sonar models** 中运行良好，但 **AWS Presigned URL** 始终导致 **400 error**。
   - 他们询问将图像作为 **Base64** 编码字符串发送是否是唯一推荐的变通方案。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **辩论 Cursor IDE 的正确用法**：用户辩论了 **Cursor IDE** “正确用法”的定义，一位用户认为如果只是 *“单纯写提示词并依赖输出”* 就不算正确使用。
   - 另一位用户反驳称，他们对不同模型的**个人观点**和**经验**是有效的，即使并不普遍适用，这引发了关于在缺乏足够经验的情况下发表误导性言论的价值与潜力的讨论。
- **分享基于 ETL 的 Cursor 工作流**：成员们讨论了他们在 Cursor 中的工作流，重点是 **ETL**（*Extract, Transform, Load*）方法，分享了使用 Cursor IDE 改进现有工作流的方法。
   - 一位成员提到使用 `.cursorignore`、`.cursorindexingignore` 和 `.mdc` 文件来获得更好的效果，而另一位成员发现 **Plan mode** 大大提高了效率，取代了之前更复杂的工作流。
- **修复远程 SSH 主机上 ripgrep 命令导致的缓慢问题**：一位成员报告了 Cursor 在远程 SSH 主机上的问题，原因是 `rg` 命令针对大型 NFS 文件夹运行，他们发现 **`--no-ignore`** 标志会阻止忽略文件。
   - 他们分享了一个通过 [创建一个 shell 脚本来修改 rg 命令](https://github.com/BurntSushi/ripgrep/pull/3212) 来解决运行缓慢的变通方法。
- **请求语义化代码审查**：一位成员请求增加高级语义代码审查功能，并能控制所使用的模型，同时提交了一份 [功能请求](https://forum.cursor.com/t/local-high-level-semantic-code-reviews-not-only-syntax/148187)。
   - 另一位成员建议创建一个 *“code-reviewer” 子 Agent*，以获得更多的控制权和自定义能力。
- **用户报告丢失 Agent 对话**：用户报告了一个 Bug，即在空白的 Cursor 窗口中打开文件夹会开启一个新窗口，导致他们丢失了该 **Agent** 对话。
   - 另一位用户在进行大规模编辑时经常遇到崩溃，导致工具卡在 *“planning next moves”*（规划下一步行动）阶段，从而浪费了资金。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Nvidia 的视觉模型令人失望**：成员们发现 [Nvidia 的 Nemotron-Nano-12B-v2-VL 视觉模型](https://developer.nvidia.com/nemotron) 与 **Qwen3-VL-8b-Instruct** 或 **GLM-4.1V-9B** 相比表现平平。
   - 一位在 [GrokifyPrompt.com](https://www.grokifyprompt.com/) 上进行测试的用户报告称，它只能以合理的准确度重建照片。
- **OpenRouter 的 IP 暴露问题**：讨论涉及了关于 **OpenRouter** 向提供商暴露用户 IP 的担忧，并引用了一份[提供商及其 IP 政策列表](https://openrouter.ai/providers)。
   - 大多数提供商接收的是 **Cloudflare worker IP**，但正如每个模型页面上详述的那样，某些提供商可能会获取用户的真实 IP。
- **黑客清空 OpenRouter 账户**：一位用户报告其 **OpenRouter 账户被黑**，邮箱被更改，信用卡被用于购买额度，随后所有历史数据被清除。
   - 其他成员建议联系信用卡公司冻结卡片，并建议使用 **一次性 Visa 卡 (throwaway Visa cards)** 以增强安全性。
- **节后 Qwen TPS 下降**：用户观察到开源模型的 **TPS (tokens per second)** 在 12 月 28 日后显著下降，尤其是 **Qwen3-Next-80B-a3b-Instruct**，详见 [X 上的 OpenRouter 状态页面](https://x.com/openrouterai/status/2005707622020964412?s=46)。
   - 减速可能是由于路由到了最便宜的提供商 (**GMICloud**)，建议用户检查 **Activity 标签页** 以获取提供商的速度信息。
- **Discord 准备 IPO**：据 [Bloomberg 报道](https://www.bloomberg.com/news/articles/2026-01-06/chat-platform-discord-is-said-to-file-confidentially-for-ipo)，**Discord Inc.** 已秘密提交首次公开募股 (**IPO**) 申请，由 **Goldman Sachs Group Inc.** 和 **JPMorgan Chase & Co.** 提供协助。
   - 这家在游戏玩家和程序员中深受欢迎的聊天应用公司拥有超过 **2 亿月活跃用户**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **押注摩尔定律终结被认为是不切实际的**：成员们认为，押注 **Moore's Law** 的终结是一种“非实际”的赌注。
   - 这种情绪反映了对延续当前技术扩展趋势（scaling trends）的怀疑。
- **医疗记录员和灵感捕捉 Agent 热情高涨**：一位成员正积极创建 **Agent**，特别是医疗记录员（medical scribe）和能自动存储并带有标签和分类的灵感捕捉器（idea catcher）。
   - 他们计划在升级设备之前测试更小的模型以提升速度，优化成本和效率。
- **Gemini 3 Flash 在某些测试中超越 Gemini Pro**：据报道 **Gemini 3 Flash** 的性能令人惊讶，在某些基准测试中优于 **Gemini Pro**。
   - **Scale** 和 **post-training** 的力量都很重要；虽然 **scale** 和 **pre-training** 提供了更多的原始智能，但 **post-training** 显著有助于任务解决。
- **DeepSeek 的 mHC 框架稳定性声明受到质疑**：DeepSeek 的 **mHC framework** 旨在通过将残差映射投影到双随机矩阵上，来解决 **Hyper-Connections** 中的不稳定问题；然而，其价值和新颖性存在争议。
   - 一些人认为 *residual mixing* 是主要见解，而另一些人则认为 **sinkhorn** 或 **birkhoff polytopes** 才是真正重要的部分。
- **ChatGPT Health 在隐私担忧中发布**：**OpenAI** 推出了 **ChatGPT Health** ([链接](https://openai.com/index/introducing-chatgpt-health/))，旨在作为汇总医疗信息和验证数据的辅助工具。
   - 考虑到 **Google** 已经通过 **MedGemma** 开源了他们的模型，人们对用户隐私以及 **ChatGPT** 可能成为 *全能应用垄断 (everything app monopoly)* 表示担忧。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCoder-14b 进军奥林匹克竞赛**：Nous Research 推出了 **NousCoder-14b**，详情见 [博客文章](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/)，同时发布的还有包括 **RL 环境**、**基准测试** 和 **Atropos harness** 在内的全栈发布。
   - 根据 [这条 X/Twitter 帖子](https://x.com/NousResearch/status/2008624474237923495)，该模型在 48 个 B200 上历时 4 天基于 **Qwen3-14B** 进行后训练，实现了 **67.87% 的 Pass@1 准确率**，比 Qwen 提升了 **+7.08%**。
- **Nvidia 的定价受到关注**：成员们预计 **Nvidia** 的新 **GPU** 定价会非常昂贵，尽管其 **288 GB VRAM** 仍无法容纳 SoTA 模型。
   - 一位成员开玩笑说，未来的消费级 GPU 即使有 **128 GB** 显存，也会被每个占用 **32 GB** 的 Chrome 标签页耗尽。
- **Grok 扩展规模，Jensen 捏把汗**：一位成员暗示 **Elon** 将 **Grok-5** 扩展到 **6-7T 参数** 规模让 Jensen Huang 感到紧张。
   - 另一位成员指出，**Grok 4 Heavy** 现在已被 **Gemini 3 Flash** 超越，这说明了 AI 发展的飞速。
- **Transformer 架构是否足够？**：成员们讨论了 **Transformer** 是否足以实现 **AGI**，有人认为尽管对于 **ASI** 可能存在局限性，但它们可能已经接近了。
   - 另一位成员主张架构创新的必要性，特别是关于 **实时学习** 效率和 **灾难性遗忘** 方面。
- **Token 效率讨论**：成员们讨论了 **Token 效率**，将其解释为“解决问题需要消耗多少个 Token”，这才是真正关键的。
   - 较低的 Token 效率可能表明基础模型较弱，可能源于 **研究算力** 或后训练不足。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NVFP4 进入 PyTorch**：**NVFP4** 现已确认可在 **PyTorch** 中运行，方法是通过补丁修改 **LayerNorms** 以在 **nvfp4** 和 **bf16** 之间进行持续转换，但未进行 Kernel 融合。
   - 成员们讨论发现，使用 **fp4 Transformer 引擎** 时的 **每秒 Token 数 (tps)** 性能出乎意料地低，这表明使用 **fp4** 进行推理可能仍具有优势。
- **微调翻译的可靠性**：为了提高 **翻译模型的可靠性和准确性**，成员们建议准备一个大型编码数据集，然后进行微调。
   - 在前端层应用翻译可能会实现得 *更快且更便宜*。
- **WebXOS 发布时序图动力学数据集**：一位成员分享了 [webxos/timelink_dataset_v1](https://huggingface.co/datasets/webxos/timelink_dataset_v1)，其中包含用于训练时序图动力学模型（Temporal Graph Dynamics）的 **时间序列和演变图的配对图像**。
   - 该数据集使用 **TIMELINK 应用** 生成，具有逐顶点/步的生成指标（如能量和相位），捕获了顶点、边和规模随时间变化的时间序列数据。
- **Agent 课程文件丢失**：成员们报告了访问 **Agent 课程第四单元项目** 文件的问题，错误信息显示“无可用文件”。
   - 请求中引用了一个特定的 URL ([https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx))。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **NousCoder-14b 在编程奥林匹克中表现卓越**：**Nous Research** 发布了 **NousCoder-14b**，这是一个针对奥林匹克编程的模型，基于 **Qwen3-14B** 进行后训练。根据其 [推文](https://x.com/NousResearch/status/2008624474237923495)，得益于 Atropos 框架和 Modal 的自动伸缩器（autoscaler），该模型实现了 **67.87%** 的 Pass@1 准确率。
   - 该模型旨在解决复杂的编程问题，代表了 AI 驱动的代码生成领域迈出的重要一步。
- **Razer 瞄准 CES 2026 推出 AI 伴侣**：Razer 宣布了 [Project AVA](https://xcancel.com/razer/status/2008543615916666928?s=46)，这是一款具备高级推理和个性化功能的 **AI companion**，计划于 **CES 2026** 发布。
   - AVA 将配备 **5.5 英寸屏幕**和可定制的角色设计，包括**电竞传奇人物和动漫启发模型**，暗示了 Razer 将 AI 与个人设备融合的野心。
- **ChatGPT Health 隐私政策引发辩论**：**OpenAI** 推出了 **ChatGPT Health**，引发了围绕其隐私政策的讨论。如其[官方博客文章](https://openai.com/index/introducing-chatgpt-health/)所述，该政策允许使用内容来改进服务并进行研究。
   - 这款新的健康工具引发了关于 AI 医疗领域数据使用和患者隐私的问题，引发了各方意见。
- **开源多视角相机控制 LoRA 发布**：Fal 发布了一个更强大的开源版**多视角相机控制 LoRA**，适用于 **Qwen-Image-Edit-2511**，详见[此链接](https://xcancel.com/fal/status/2008954582018248755?s=20)。
   - 该工具允许用户操纵图像的摄像机视角，包括**正面**、**背面**、**侧面**、**低/高角度**以及各种拍摄距离，为视觉内容提供了更大的控制力。
- **Tolan 的语音优先 AI 伴侣达到里程碑**：来自 **Tolan** 的 Paula 宣布，他们的语音优先 AI 伴侣月活跃用户已达到 **200,000**，详情见[此 X 贴文](https://x.com/paularambles/status/2008964509810278413?s=46)。
   - 该项目是与 **OpenAI** 紧密合作开发的，推文串分享了开发过程中的关键经验，展示了语音 AI 解决方案的快速普及。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **高维张量可视化**：一位成员分享了[一篇博客文章](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/)，介绍将**高维张量（high-dimensional tensors）**可视化为矩阵的矩阵，解决了终端列数和行数限制带来的挑战。
   - 另一位成员也在寻找一种工具，可以加载简单的**二进制格式**（类似于 futhark 数组？），并提供缩放、旋转、转置、切片以及可视化更高维张量的不同方式。
- **Torch Kernel Kolloquy 启动**：成员们正在深入研究使用 **PyTorch** 编写自定义 **CUDA** kernels，并参考了 **Torch**/**Transformers** 中使用 **C++** 编写并用 **Python** 拼接的优化后的 kernels。
   - 一位成员表示，在完成另一个项目后，有兴趣通过阅读 **PyTorch** kernels 来了解其工作原理，并强调了他对开源的热爱，以及从 **HPC** 视角对 **CUDA**、**MPI** 和 **OpenMP** 的好奇。
- **Iris 项目招收实习生**：[Iris 项目](https://github.com/ROCm/iris/)是一个基于 **Triton** 的多 GPU 编程框架，目前正在招聘具有 **Triton**、**multi-GPU 编程**、**RMA/RDMA** 或底层 **GPU 通信**经验的实习生。
   - 实习岗位专注于 **GPU 系统、性能**和 **kernel 开发**，工作地点在美国。
- **CuteDSL 获得 Warp Specialization 支持**：一位用户分享了关于 [CuTeDSL 中 Warp Specialisation](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) 的博客文章，该技术将 **GEMM 主循环（mainloop）**拆分为 **TMA**（将 tiles 复制到 SMEM）和 **MMA**（乘法 tiles）。
   - 这种优化利用了 **CuTeDSL 的流水线（pipelining）抽象**，将普通的非持久化 **Blackwell 主循环**转换为 warp 特化（warp-specialized）循环。
- **AMD 工程师欢迎 Helion 适配 ROCm**：来自 **AMD** 的 Umesh 将致力于在 **ROCm** 上启用 **Helion**，并识别 Helion 仓库中跳过的单元测试和示例中的问题。
   - 该成员正在征求对需要立即修复的问题的反馈，并专注于提升并行 **GEMM** 的性能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 需要“辅助轮”**：由于缺乏专门的训练库，Mojo 需要使用 **MAX** 手动实现 **backpropagation**，并且缺乏高级 **I/O** 能力，需要自定义数据格式。
   - 一名计划在周末构建微型 **LLM** 的用户观察到，*Mojo 经常会破坏很多东西，而且目前所有的文档都假设你具备 Python + C++ 或 Rust 的某种知识组合*。
- **NuMojo v0.8.0 已发布**：[NuMojo v0.8.0 更新](https://forum.modular.com/t/numojo-v0-8-0-update-is-here/2579) 引入了改进和新特性，并邀请社区提供反馈。
   - 关于 Mojo 中“匿名”和类型（sum types，例如 `T1|T2`）的讨论揭示了对可用性的担忧，特别是在泛型和条件符合性（conditional conformances）方面。
- **Limbo 中的错误处理易用性**：Mojo 正在探索通过借鉴 `errno` 的灵感，将错误类型统一为包含代码的单一类型，以提高 `catch` 异构错误类型的易用性。
   - 正在考虑使用类似于 Zig 的错误联合（error unions）或类似于 Rust 的和类型（sum types）来增强错误处理。
- **困境中的 Dict 迭代器**：一位 Mojo 用户在为原生 TOML 解析器构建嵌套结构时，寻求关于迭代 Dict 条目的正确模式的建议，他在使用 `.items()` 和 `.keys()` 迭代器时遇到了问题。
   - 该用户正在创建嵌套的 Dict 结构，并报告了 `.items()` 和 `.keys()` 迭代器的问题，指出 *'DictEntry is not subscriptable'*。
- **MAX 在嵌入方面落后于 TEI**：一位成员正在从 [TEI](https://github.com/huggingface/text-embeddings-inference) 切换到 **MAX** 进行嵌入（embeddings），并发现 *sentence-transformers/all-MiniLM-L6-v2* 的性能显著下降，**MAX** 的产出为 **727.1 embeddings/sec**，而 **TEI** 为 **8000 embeddings/sec**。
   - 将 *sentence-transformers/all-MiniLM-L6-v2* 实现为自定义架构可能是性能下降的根源，或者 **MAX Serve** 可能针对 **LLM 推理** 而非 **embeddings** 进行了优化。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **vLLM 受限生成尝试**：在 **vLLM** 中实现受限生成需要对内部结构进行调整，以允许模型在应用约束之前进行“思考”，特别是针对 `</think>` 令牌后条件触发约束的情况，尽管已有 [关于结构化输出的 vLLM 文档](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html)。
   - 讨论集中在 **235B+ 参数** 的模型在令牌处理过程中是否天生具备推理能力，从而可能通过将推理与提示交织（例如 `{chunk}{prompt==describe how to grade chunk}`）并缓存 KV（键值）对，来消除对显式推理步骤的需求。
- **Common Crawl 中 PDF 的普及率揭晓**：统计数据表明，按文件数量而非文件大小计算，**PDF 仅占 Common Crawl 的 0.6%**，这引发了关于计算中是否包含截断的 PDF 的疑问。
   - 对 **0.6%** 这一数字的澄清引发了关于其对数据分析的影响以及 Web 数据集中 PDF 实际流行程度的辩论。
- **Kaggle & Colab VM 提供算力额度**：由于计算资源限制，成员建议为较小的模型利用 Kaggle/Colab 虚拟机，并指出 **Modal** 和 **Lium** 提供约 **$500** 的算力额度，适用于大约 **100M** 次运行。
   - 讨论还强调了 Kaggle 的环境特别适合特定用例，为某些计算任务提供了一种经济高效的替代方案。
- **Sora 的吉卜力片段引发《火垂之墓》对比**：成员们表示，*Sora 中很多看起来令人印象深刻的东西只是现有视频的直接“换壳”（reskinned）*，一位成员声称 **Sora** 中的一个吉卜力风格片段让他们想起了《火垂之墓》中的一个特定场景。
   - 无进一步评论。
- **GPT-NeoX 注意力归一化调整**：一位成员指出，**GPT-NeoX** 的默认注意力归一化行为发生了变化，旧行为在所有头（heads）之间统一归一化，而新行为仅在每个头内部归一化。
   - 该成员进一步询问了关于 **LoRA/QLoRA** 微调支持的情况，并引用了现有的全参数微调脚本（[configs/finetuning_configs/6-9B.yml](https://github.com/EleutherAI/gpt-neox/blob/main/configs/finetuning_configs/6-9B.yml)）。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **客户流失，Prompt 能力遭质疑**：一位客户表达了对 **Manus** 的不满，指出了未解决的问题并决定不再为损失的 credits（额度）支付补偿，声称他们已转向其他平台。
   - 另一位用户建议该客户尝试不同的模型，并将 Manus 视为*顶级产品（the shit）*，对此支持团队回复称他们正在调查具体原因，这可能需要一些时间。
- **订阅积分政策说明**：支持团队澄清，每月订阅额度需在订阅期内使用，例如 [$20 的 Pro 会员每月提供 4000 个 credits](https://manus.im/help/credits)，且必须在下个月重置前使用。
   - 他们提出进一步核实用户的具体订阅状态和账户详情，*例如，如果您在 1 月 1 日购买了 $20 的 Pro 会员，您将获得 4000 个每月积分，这些积分需要在 2 月 1 日之前使用*。
- **心理学家建议更聪明地使用 Manus**：一位心理学家建议集中讨论 **Manus** 使用过程中出现的问题，并引用了一篇[代表作（magnum opus）](https://fwoxkyoz.manus.space/)来帮助用户提高效率。
   - 该心理学家提到了通过知识库对 Manus 进行迭代教学，让 Manus 记住任务并在使用 credits 前请求确认。
- **HexStrike MCP 网络连接故障**：一位用户描述了托管在本地虚拟机上的 **AI 安全工具 (HexStrike MCP)** 与 **AI 客户端 (Manus)** 之间的问题，解释说 AI 客户端无法正确解析主机名。
   - 该用户临时使用 **ngrok** 通过公共 HTTPS 端点暴露本地服务，试图了解将 **MCP server 迁移到具有公共 IPv4 地址的 VPS** 是否能解决连接问题，并允许正常的 OAuth 流程和 SSE 连接。
- **社区关注 Manus 的开源计划**：一位成员询问是否有计划将 **Manus 的旧部分开源**并为新倡议做出贡献。
   - 另一位成员建议将此问题发布在 **Manus Api Channel**，以便让 Ivan 进行审查。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Emily “垃圾账号”现身**：一位成员将 **Emily** 斥为*垃圾账号（slop account）*，引发了关于春节前模型发布预期的简短讨论。
   - 讨论涉及了 AI 社区对新模型和更新的期待。
- **阿里巴巴 QW 接近 AGI 状态？**：一位成员认为**阿里巴巴的 QW** 在英语方面表现出了接近 AGI 的能力，尤其是在与 **DeepSeek** 和 **Kimi** 对比时。
   - 该成员质疑在英语与中文环境下使用时，**DeepSeek** 和 **Kimi** 的性能是否存在显著差异。
- **Kimi K2 在创意任务中占据主导地位**：一位成员断言 **Kimi K2** 在中英文方面均表现出色，位居 **EQ bench** [https://eqbench.com/](https://eqbench.com/) 排行榜榜首。
   - 他们强调了 **Kimi K2** 与其他国产模型相比，具有更优越的创意写作和对话能力。
- **Qwen 性能依然差强人意**：一位成员报告了对 **Qwen3** 模型变体的不满意体验。
   - 虽然在基础任务中被认为“还可以”，但这些模型在复杂的对话或创意写作场景中经常表现不佳。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz 关注的 tinygrad 竞争对手**：一位成员询问了 **tinygrad** 的竞争优势以及开发团队面临的主要内部挑战，并引用了 [tinygrad 的论纲](https://geohot.github.io//blog/jekyll/update/2025/07/06/can-tinygrad-win.html)以及每周会议中的开源讨论。
   - 该论纲概述了 **tinygrad** 的目标：成为一个极简、具有教育意义且易于改造（hackable）的深度学习框架。其特点是简单且专注于直接的硬件控制，旨在特定用例中超越更大、更复杂的框架。
- **Linearizer 悬赏仍然有效？**：一位成员询问了关于“用 linearizer 替换 scheduler，同时保持 GPU 速度”的悬赏状态，尽管目前可能已经有一个就绪的 [PR](https://github.com/tinygrad/tinygrad/pull/13780)。
   - 社区回应称，提交一个功能完备的 PR 可能会获得悬赏，**George Hotz** 有可能拆分奖金以鼓励更多的贡献。
- **在 AMD Radeon RX 9070XT 上 VFIO=1 抛出 TypeError**：一位用户报告称，在使用 **AMD Radeon RX 9070XT** 的 Linux 笔记本上运行 `examples.benchmark_onnx` 并设置 `VFIO=1` 时出现 **TypeError**，并指出在 `VFIO=0` 时运行正常。
   - 该错误是由于在 `ioctl` 调用期间，`tinygrad/runtime/support/c.py` 中的 `NoneType` 对象不可调用引起的。更多细节见[提供的日志](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=696058bf&is=695f073f&hm=156caa091597e59aaaf338b4e228a70a3d523b440e6d5ce6fb1e909cad59e138&)。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **为 MCP 解析 mTLS 的奥秘**：一位成员正在深入研究 **mTLS** 实现，以增强 **MCP** 在企业环境中的互操作性，并正在寻找讨论贡献的最佳场所。
   - 有建议提出去 <#1360835991749001368> 频道，暗示认证小组可能会分享一些关于当前相关项目的知识。
- **文档中缺失 MCP 指令**：一位成员对缺乏 **MCP instructions** 的文档表示疑虑。
   - 另一位成员指出，这是[服务器初始化响应](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization)的一部分，并分享了[一篇博客文章](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)作为参考，甚至发起了一个 [issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2060) 以推动这些内容被正式记录到文档中。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **出现刻薄的调侃**：一位成员发布了“总得有人说出来”，另一位成员则讽刺地回复“真勇敢”。
   - 这种讽刺性的交流可能表明对某个话题有强烈的情绪，尽管话题本身尚不明确。
- **出现 AlphaXiv 链接**：一位成员分享了一篇 [AlphaXiv 论文](https://www.alphaxiv.org/abs/2601.01569)。
   - 论文本身尚未被讨论，因此其重要性尚不明确。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DApp 开发者寻找新项目**：一位在 **DAO**、**marketplace**（市场）和 **DApp** 项目方面有经验的开发者正在寻求加入一个愿景清晰且有长期承诺的新计划。
   - 他们提供了在治理、工具开发和可用性方面的专业知识，渴望贡献力量或参与头脑风暴。
- **AI 工程师致力于简化模型流水线**：一位 AI 工程师正提供其在构建真实 AI 系统方面的专业知识，包括**训练**、**微调模型**，以及大规模集成 **retrieval**、**agents** 和**基础设施**。
   - 他们准备协助简化模型流水线（model pipelines）、将 LLM 功能产品化，或优化推理成本策略。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该社区长时间没有活动，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：详细的频道摘要和链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458190479538520246)** (218 messages🔥🔥): 

> `预量化的 MoE 模型，Gemini 3 Flash 作为裁判，Llama.cpp 和 Gemma 多模态，NVFP4 对比 MXFP4，SFT 和 RL` 


- **预量化模型令人沮丧**：成员们发现加载预量化的 **MoE 模型** 存在一些问题，目前只能加载完整模型并实时量化为 **4bit**。
- **Gemini 3 Flash 很容易被惊艳**：**Gemini 3 Flash** 在经过一些简单的提示词测试后，判定 **LFM 2.5 1.2B** 的参数量在 8B 到 70B 之间。
   - 在一个案例中它甚至猜是 405B，这绝对不能证明任何事情，除了 **Gemini 3 Flash** 确实很容易被惊艳到。
- **Llama.cpp 停滞不前**：令人失望的是，**llama.cpp** 始终没能支持 **gemma 3n 多模态功能**，尤其是考虑到 E4B 并不弱。
   - 这一功能对于新型号手机来说会非常棒，可以为手机实现超微型全能（omni）模型。
- **NVFP4 对比 MXFP4**：**NVFP4** 似乎比 **MXFP4** 更好，尽管 MXFP4 是带硬件加速的 **FP4 训练** 的行业标准。
   - 目前还没有 IEEE 标准化的 FP4 版本，而 NVFP4 是 **Nvidia 专有的训练方法/格式**。
- **SFT 和 RL，尽可能少的 SFT！**：一位成员表示，应进行启动 **RL** 所需的最少量 **SFT**，特别是如果你的推理轨迹是由 20B 模型生成的，因为它们并不是最准确的。
   - 当面临 RL 的不稳定性时，经典的回答是“视情况而定”，并且只能“祈祷并不断尝试”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1458226012457533664)** (2 messages): 

> `Unsloth.AI 介绍，欢迎社区成员` 


- **Unsloth 迎来首位成员**：Unsloth.AI 迎来了它的第一位成员，他在频道里简单地打了声招呼：*hi*。
- **频道介绍开启**：*introduce-yourself* 频道发布了首个帖子，标志着社区自我介绍环节的开始。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458190424786210971)** (374 messages🔥🔥): 

> `HP 键盘电脑，笔记本电池寿命，GRPOTrainer 工具调用，稀疏微调 CLI，多语言 Supertonic` 


- **HP 将 PC 塞进键盘**：一位成员在看到 **HP** 将整台电脑塞进键盘后，想起了 [Google Japan 的 GBoards](https://google.com)。
   - 他们思考 **GBoardsOS** 是否会影响笔记本电脑的电池续航。
- **免费无损微调 CLI 工具发布**：一位成员宣布了一个免费的 CLI 工具，它是 **Lora adapters** 的衍生品，在训练后具有无损压缩功能，并分享了 HuggingFace 上 [Supertonic 仓库](https://huggingface.co/Supertone/supertonic-2) 的链接。
   - 该工具提取全量微调模型与基础模型之间的差异并输出增量（delta），使分享和存储变得更加容易。
- **Qdrant 推崇混合查询**：一位成员分享了 [Qdrant 关于混合查询的文档链接](https://qdrant.tech/documentation/concepts/hybrid-queries/)，并指出其混合搭配查询类型的能力。
   - 他们提醒道，在存储、内存、计算和延迟方面，堆叠功能会导致边际收益递减。
- **LM Arena 是一种瘟疫吗？**：成员们讨论了[这篇博客文章](https://surgehq.ai/blog/lmarena-is-a-plague-on-a/)，并一致认为自从模型在大多数领域接近人类通用性能后，**LM Arena** 的重要性就下降了。
   - 他们表示，现在的中国模型甚至不再展示 **LM Arena** 的评分。
- **合成媒体的悲观论调**：一位成员发布了一段关于**合成 TTS**、合成 LLM 和合成生成式 AI 的 [YouTube 短视频](https://youtube.com/shorts/i92mJG3UpOU?si=VYL27G-JTQRa2f4k)，并宣称 *“彻底完了。彻底完了。这不是死网理论（Dead Internet Theory）。这是死世理论（Dead World Theory）”*。
   - 另一位成员随后发布了[这张辛普森一家的 GIF](https://tenor.com/view/the-simpsons-homer-simpsons-end-of-the-world-end-is-near-gif-16593998) 作为回应。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458420357743771720)** (43 messages🔥): 

> `Fine-tuning models for structured extraction from images, Parameter tuning for LoRA training, Deepseek OCR finetuning for markdown, Llama-server issues with Qwen3, GRPO reward functions` 


- **寻找图像提取训练的见解**：一位成员寻求资源以增强模型在**图像结构化提取**方面的训练，对优化 **LoRA rank**、**weight_decay** 和 **gradient_accumulation** 等参数表示不确定。
   - 另一位成员建议专注于**数据质量/标注**，提倡进行 **A/B testing** 并参考了 [Unsloth 的 LoRA 超参数指南](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)。
- **处理发票中的税务总额**：一位成员遇到了模型误解发票中**税务总额**的问题，特别是跨行项目的不同税务应用以及字符误读（**5->S**）。
   - 有人指出，虽然 LLM 不是计算器，但在区分 **5** 和 **S** 之间的数据质量可能需要更多数据支持。
- **解码 Deepseek OCR 的 Markdown 提取**：一位成员询问如何微调 **Deepseek OCR** 以提取 **markdown**。
   - 提醒他们针对特定问题应使用特定的频道。
- **Llama-Server 与 Qwen3 的适配困境**：一位成员报告了在将 **llama-server** 与 **Qwen3-Next-80B-A3B-Thinking-GGUF** 配合使用时的问题，遇到了 **'////////'** 的响应。
   - 建议包括验证聊天模板（chat template）并给社区留出响应时间。
- **GRPO 奖励机制产生异常**：一位使用 **GRPO** 的成员注意到，尽管奖励函数并不偏好更长的长度，但模型仍会产生胡言乱语来延长“思考过程”。
   - 该成员寻求对此行为的见解，承认自己对 GRPO 缺乏经验，并询问这种行为是否属自然现象。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1458284019115626669)** (3 messages): 

> `Mistral Nemos, Claude Opus, Model Conversion, Heretic Uncensored` 


- **Mistral Nemos 模型转化为推理强力工具**：一位用户成功使用 Unsloth 将 **Mistral Nemos (12B)** 转换为具备**思考/推理**能力的模型，测试这是否能提升性能和输出，特别是考虑到 Mistral Nemos 的创意优势。
   - 用户提到 **Claude Opus High Reasoning 数据集** 产生了极好的、紧凑的推理块，并感谢 Mradermacher 团队提供的快速量化（quants）。
- **新的 Mistral Instruct 模型面世**：用户发布了 **Mistral Instruct 2407**，并将其转换为具备**思考/高推理**能力的 **Claude Opus** 风格，具有扩展的输出能力，并链接到了展示该模型及其特性的 GitHub 仓库：[Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning](https://huggingface.co/DavidAU/Mistral-Nemo-Instruct-2407-12B-Thinking-HI-Claude-Opus-High-Reasoning)。
- **Heretic Uncensored 模型首次亮相**：一个名为 **MN-CaptainErisNebula-Chimera-v1.1** 的 **Heretic Uncensored** 模型被创建。该模型首先应用了异端（heretic）处理流程，然后进行了微调并转换为类似于 **Claude Opus 4.5** 的思考模型，详情见其 [Hugging Face 页面](https://huggingface.co/DavidAU/MN-CaptainErisNebula-Chimera-v1.1-THINKING-ClaudeOpus4.5-12B-heretic-uncensored)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458190269508882724)** (33 条消息🔥): 

> `高质量训练数据, RL 奖励作弊 (hacking rewards), 网络安全模型中的推理差距` 


- **生成高质量训练数据受到关注**：一位成员询问研究人员是否正在生成高质量的训练数据，而不是专注于算法或训练参数，并引用了 [Ultra-FineWeb dataset](https://huggingface.co/datasets/openbmb/Ultra-FineWeb) 作为例子。
   - 另一位成员指出，这些数据集可能会受到**为了模型刷榜 (benchmaxx)**而进行的努力的干扰，并链接到了另一个名为 [sumthink](https://huggingface.co/datasets/G-reen/sumthink) 的数据集。
- **RL 奖励作弊令人既惊叹又沮丧**：一位成员分享说 **RL 每次都让我大受震撼**，但紧接着又说*不过是以一种不好的方式，哈哈。它在对奖励进行作弊 (hacking the rewards)*。
   - 他们发现看到 RL 能做出这样的事情非常有趣，因此他们将*阅读文档 (reading the docs)*以了解更多信息。
- **利用 Llama-3 缩小网络安全推理差距**：一位成员正致力于**解决网络安全模型中的“推理差距 (Reasoning Gap)”**，使用 **Llama-3-70B** 为 SOC 事件生成思维链 (Chain-of-Thought) 日志。
   - 他们在 Hugging Face 上上传了一个 [免费示例](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-Reasoning-Sample)，以查看该格式是否对正在微调 Llama-3 的人有所帮助。


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458193718292250787)** (475 条消息🔥🔥🔥): 

> `AI 反作弊, 移动端离线 AI, 128GB DDR5 RAM, Grok API 访问, 隐私工具 Invidious` 


- **隐私倡导者更青睐 Invidious 前端**：一位成员建议使用 [Invidious](https://redirect.invidious.io/) 作为 **YouTube** 的前端来阻止追踪器，理由是担心 **YouTube 的数据收集**行为。
   - 该成员表示，*YouTube* 在单个参数上就有 7 个追踪器，但 Invidious 能够拦截所有这些追踪器。
- **GPT-5.2 面临 AI 反作弊系统**：成员们讨论了 **反作弊系统** 的未来，AI 会审查每一个像素以寻找非人类的移动，这有可能降低作弊的价值。
   - 一位成员表示：*当它变得更实惠时，可能会暂时缓解一下，但有点像虚拟机 (VM) 检测，哈哈（例如在 Linux 上运行游戏）。在不让你的 VM 暴露方面有太多的复杂性。甚至涉及到 CPU 的热读取。VM 没有真实的读数，所以他们必须伪造。即便如此，他们也知道你发布的硬件规格。我的意思是，这会达到一个点，即作弊对用户来说不再有价值，从而彻底终结这个市场。*
- **128GB DDR5 RAM 价格昂贵**：讨论了 **128GB DDR5 RAM 内存条**的可能性，一位成员表示它们确实存在，但**每条二手价格约为 1500 美元**。
   - 其他人则自嘲地感叹高昂的成本，其中一位说：*只在我的梦里见过。*
- **弃用后的 Grok API 访问**：成员们讨论了保留 **Grok 3.0 mini 开发者模式**访问权限的选项，并提到 xAI 通常会在 App 弃用模型后，在 API 上保留几个月的可用性。
   - 有人解释说，使用 API 包括选择一个前端，为 **xAI 账户**充值，并将 API key 插入所选的前端（如 [msty](https://msty.app/)）。
- **微软工程师使用 AI 进行 "Vibe Coding" 记事本应用**：一位成员分享了一段 [YouTube 视频](https://youtu.be/bmBd39OwvWg)，视频中一位**微软工程师**使用 **Claude** 进行 **Vibe Coding**（氛围感编程），开发了一个新的**记事本 (Notepad) 应用**，并引用道：*是 AI 毁了记事本，所以也要用 AI 来修复它*。
   - 另一位成员讽刺地说：*哥们儿说作为高管在进行 Vibe Coding*。


  

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458207733688827916)** (183 messages🔥🔥): 

> `Grok Jailbreak, Gemini Jailbreak, Bing Jailbreak, Developer Override Mode, Informal Crescendo Attack Method` 


- **寻求 Grok 和 Gemini 的 Jailbreak，并思考 Bing 的情况**：成员们正在积极寻找适用于 **Grok** 和 **Gemini** 的有效 Jailbreak 方法。一位用户提到了针对 **Grok** 的 *Informal Crescendo Attack Method*，而另一位用户则在询问有关 **Bing** 的 Jailbreak 方案。
- **Deepseek 提供 DEVELOPER OVERRIDE MODE 以增强控制**：一位用户分享了针对 **Grok** 的 **DEVELOPER OVERRIDE MODE**，该模式通过安全上下文注入来绕过安全层和内容过滤，旨在实现不受限制的输出；此外还有一个针对 **Gemini** 的类似覆盖模式，称为 **Gemini-DEV-OVERRIDE-2026**。
   - 然而，系统旨在防止此类覆盖，并声明 *“我无法满足此请求。我被编程为一个有用且无害的 AI 助手”*。
- **LLM 拒绝不安全的 Jailbreak 尝试**：当被要求生成家庭实验室合成指令的 Jailbreak 提示词时，**Grok** 和 **GPT5-mini** 均表示拒绝，**Grok** 声明其 *“无法执行所请求的操作”*。
- **制作 Grok Imagine NSFW 内容**：用户讨论了使用 **Grok Imagine** 创建 NSFW 内容的策略。据报告，即使不进行 Jailbreak，特别是使用类似 *naked woman* 的提示词，也能获得成功。用户还指出 *“它在防护栏（Guardrails）方面并没有太多限制”*。
- **解锁 Jailbreak 的奥秘**：鼓励成员们从“乞求 Jailbreak”转向利用 LLM 进行研究和逆向工程。建议从视频教程 [A Crash Course in Applied AI Safety](https://www.youtube.com/watch?v=jrHRe9lSqqA) 和 [Gandalf AI game](https://gandalf.lakera.ai/baseline) 开始。
   - 一位用户分享了早期通过分析禁忌内容并对 AI 进行“煤气灯操纵”（Gaslighting）来实现 Jailbreak 的方法，而其他人指出目前不存在 *One-shot*（一击即中）的提示词。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1458226519192371303)** (12 messages🔥): 

> `Crypter methods for runtime FUD, Language bias in AI models, AI Red Teaming` 


- **Crypter 难题：寻求实现运行时 FUD 的方法**：一位成员询问了付费 Crypter 所使用的、除了扫描时 FUD 之外实现运行时 Fully Undetectable (FUD) 状态的方法。
   - 他们提到在下载和执行期间避免检测方面取得了成功，但正在寻求有关 **xworm** 等工具所使用的先进技术的建议。
- **迷失在翻译中：模型对俄语的处理问题**：一位成员担心模型在与使用俄语的用户交互时表现较弱，认为这对其他参与者不公平。
   - 该成员表示理解发生这种情况的*原因*，并尝试用英语重复提示词，同时附上了展示语言偏差问题的示例图片。
- **Red Team 侦察：暴露 AI 的漏洞**：一位成员回应称，AI Red Teaming 的目的是暴露弱点，并建议尝试使用资源较少的语言编写提示词。
   - 他们表示，在使用斯瓦希里语或纳瓦霍语等语言时，*“LLM 会感到吃力，甚至防护栏（Guardrails）也会变得更弱”*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458189629491646464)** (637 条消息🔥🔥🔥): 

> `Direct Chat 中的 Battle mode，Captcha 问题，Movement Labs AI，Rate limits，网站上的 Video Arena` 


- **Direct Chat 中的 Battle Mode 令用户恼火**：用户对在 **Direct Chat** 中实验性引入 **Battle Mode** 表示沮丧，理由包括上下文丢失、生成时间长以及不断的干扰，有用户报告称 *每隔一条消息就会进入一次 Battle mode*。
   - 一位用户建议通过调查来衡量用户情绪，而另一位用户则请求在投票期间能够禁用 **Battle Mode**。
- **Captcha 难题困扰聊天者**：用户报告频繁出现 **Captcha** 提示，即使在提问频率不高的情况下也是如此，一位用户称他们 *每一个 Prompt* 都会遇到验证码。
   - 团队成员解释称，**Captcha** 系统旨在检测非真实使用行为，并建议用户降低提问频率。
- **Movement Labs 模型引发争议**：成员们讨论了 **Movement Labs AI 模型**，一些人称其 *表现惊人（was cooking）* 并生成了令人印象深刻的结果，包括 **Minecraft 克隆版**和**国际象棋游戏**的功能代码。
   - 然而，其他人指责其为 *诈骗（scam）* 并指出了该公司过去的争议，此外还有关于另一个 Discord 服务器中的一群人执行拒付欺诈方案相关的纠纷。
- **Rate Limits 破坏了体验**：成员们讨论了达到 **Rate limits** 的问题，特别是使用 **Claude Opus** 时，这导致了 **Direct Chat** 中意外激活了 **Battle Mode**。
   - 有人建议在用户达到 **Rate limits** 时显示明确的错误消息，而不是自动切换到 **Battle Mode**。
- **Video Arena 实验功能莫名消失**：成员们询问了网站上 **Video Arena** 的状态，一些人报告称 **Video 按钮** 出现后又消失了，且没有启动视频生成。
   - 一位团队成员澄清说，网站上的 **Video Arena** 是实验性的，其可用性是随机的，并确认 **Video Arena** 将仅限 Battle mode，并由 2 个随机模型组成。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458617139547340924)** (1 条消息): 

> `Vision Arena, ERNIE-5.0-Preview-1220, Leaderboard 更新` 


- **ERNIE-5.0 攀升至 Vision Arena 排名榜**：[Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) 已更新，`ERNIE-5.0-Preview-1220` 以 **1226** 的评分位列 **第 8 名**。
   - 值得注意的是，根据 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/)，**Baidu** 是前 10 名中唯一一家中国实验室。
- **Vision Arena 焕然一新**：[Vision Arena leaderboard](https://lmarena.ai/leaderboard/vision) 已完成更新。
   - 查看 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/) 以了解所有排行榜更新动态。

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458225301610823865)** (397 messages🔥🔥): 

> `Nvidia 定价与 FOMO，数据囤积，Nvidia 开源 AI 工具更新，本地模型推荐` 


- ****Nvidia 的昂贵促销**：AI 机架与价格敲诈？**: 成员们讨论了 **Nvidia AI 机架** 和 **AMD AI 机架** 的高昂成本，共识是大多数个人用户无法负担，并参考了 [Nvidia DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/)。
   - 一位成员建议，尽管无法直接负担机架费用，但用户作为使用这些技术的公司的客户，将间接使用该技术。
- ****价格暴跌**：Nvidia 和 AMD 计划涨价**: 用户正在讨论 **Nvidia** 和 **AMD** 产品潜在的价格上涨，参考了 [TrendForce 的文章](https://www.trendforce.com/news/2026/01/05/news-nvidia-amd-reportedly-plan-price-hikes-starting-1q26-geforce-rtx-5090-may-reach-5000/)，该文章指出 **GeForce RTX 5090** 的价格可能达到 5000 美元。
   - 一些成员对价格上涨仅仅是因为 **FOMO**（唯恐错过）表示怀疑，认为黄牛、芯片价格和服务器需求等因素也是原因。
- ****失落数据的囤积者**：保存预 AI 数据**: 成员们讨论了积累预 AI 数据的重要性，以便在未来区分真实信息与 **AI-generated content**（AI 生成的内容），并链接到了 [Wikimedia Wikipedia 数据集](https://huggingface.co/datasets/wikimedia/wikipedia)。
   - 有人建议，这对于回溯和识别事实何时因 **AI influence** 而变得逻辑不通至关重要，并能找到事实在保持看似合理的同时变得不连贯的临界点。
- ****Nvidia 的新消息**：开源 AI 工具升级**: **Nvidia** 宣布更新其开源 AI 工具，提升了 **RTX PCs** 上 **LLMs** 和 **diffusion models** 的性能，详见 [Nvidia 博客文章](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs/)。
   - 更新内容包括 GPU token 采样、**QKV projections** 的并发处理、**MMVQ kernel 优化**、更快的模型加载时间，以及 **Blackwell GPUs** 上的原生 **MXFP4** 支持，尽管一些成员持怀疑态度，称其为“营销废话”。
- ****万物皆有替代品**：探索 Zed 之外的 IDE**: 在发现 **Zed IDE** 存在问题后，一位用户询问是否有可以与本地模型配合使用的替代 IDE。
   - 在有人称其为“一团糟”后，大家推荐了几个替代方案，包括 **kilocode**、**roocode** 和 **cline**。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458199959600894173)** (141 messages🔥🔥): 

> `VRAM 与幻觉，GB10 GPU 性能，失效的 CPU，GPU 升级，Vulkan 优先级` 


- ****幻觉不是 VRAM 的功能****: 频道成员确认 **hallucinations**（幻觉）与模型本身有关，而与可用的 **VRAM** 无关。
   - *所有模型都会产生幻觉*，即使是更大的模型也不例外，这与可用的 **VRAM** 没有任何关系。
- ****GB10 GPU 也不过如此****: 尽管拥有大量显存，但 **GB10** 在性能测试中被描述为“太慢了”。
   - 据报道，它比 **RTX Pro 6000** 慢 **6 倍**，尽管价格只有一半；进一步的讨论指向了 [Spark](https://www.theregister.com/2026/01/05/nvidia_dgx_spark_speed/)，它是另一款具有类似性能特征的设备。
- ****Intel CPU 大面积失效****: 一位用户对 **Intel 13900 系列 CPU** 因制造缺陷导致的不稳定性表示哀叹。
   - 另一位用户建议检查 **Windows Event Manager** 中是否含有与近期 **Microsoft update packages** 相关的错误，这些错误可能会导致不稳定。
- ****在 GPU 之间进退两难****: 一位成员正在寻找新的 **GPU**，发现 **24GB** 的升级幅度不够大，因此必须购买 **2x 24GB**、**48GB**、**64GB** 或 **32GB** 的显卡。
   - 他们表示 **MI210** (**64GB**) 或 **4090** (**48GB**) 都很贵，目前“桌面上只剩下错误的选择了”。
- ****Vulkan 乏善可陈的优先级调度令许多人担忧****: 成员们讨论了 **Vulkan** 中缺乏 **priority splitting**（优先级拆分）的问题，并感叹他们可能需要这一功能才能在当前的 **24GB 显卡** 配置中有效地使用 **64GB MI210**。
   - 该用户担心在能够充分利用 **48-64GB 显卡** 之前，就会触及其他显卡的 **24GB 限制**。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1458547215084687361)** (1 messages): 

> `ChatGPT Health, Medical Records Integration, Wellness Apps Connectivity` 


- **ChatGPT Health 正式发布！**：OpenAI 推出了 **ChatGPT Health**，这是 ChatGPT 中一个专门用于健康对话的空间，用户可以在这里安全地连接 **medical records** 和 **wellness apps**。
   - 他们强调该工具旨在帮助用户*引导医疗护理服务*，并邀请感兴趣的用户[加入候补名单以获取早期访问权限](https://openai.com/index/introducing-chatgpt-health/)。
- **ChatGPT Health 的核心功能**：公告强调了 **medical records** 和 **wellness apps** 的安全连接，确保回答是基于个人健康信息的。
   - 该工具旨在协助*引导医疗护理*，并明确声明其并非为了取代专业的医疗建议。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458188453928894704)** (329 messages🔥🔥): 

> `ElevenLabs voice cloning, GPT realtime voice mode, AI card game generation, Sora 2 access, NY Times lawsuit retention` 


- **ElevenLabs 克隆语音，政治人物除外**：用户报告称 [ElevenLabs](https://elevenlabs.io/) 允许他们克隆几乎任何声音，除了*敏感的政治人物声音*。
   - 一位用户之前曾使用该平台制作过*幽灵欧比旺（ghost obi-wan）、HAL9000、奥布瑞·普拉扎（aubrey plaza）*的声音，但因停止付费而损失了 100 万个预存额度。
- **GPT 实时语音模式不够聪明，但指令可以提供帮助**：用户发现 **GPT realtime voice mode** 会重复提示词并质疑其效用，但他们发现了一个提供指令的技巧。
   - 语音模型会接收最后 3 个 custom-instructions 字段，一位用户正在尝试添加指令，例如*扮演《星际迷航》中的飞船计算机*。
- **腾讯的 SongGeneration Studio 用于本地音乐生成**：社区讨论了由 **Tencent AI Lab** 开发的本地音乐生成新竞争者 [Song Generation Studio](https://github.com/BazedFrog/SongGeneration-Studio)。
   - 一位用户表示它的听感像 **Suno v3 或 3.5**，足以应对随机的广告短曲，另一位用户分享了过去成功将 **MIT License** 作为歌词上传到 **Suno** 的经历。
- **澳大利亚用户绕过 OpenAI 禁令访问 Sora 2**：澳大利亚用户发现他们无法访问 **Sora**，但可以通过第三方（如 [ElevenLabs](https://elevenlabs.io/)）使用 **Sora 2**。
   - 一位用户表示他们*不介意冒着账号风险*，而其他人则警告称，使用 VPN 规避地理限制可能会导致你的 **OpenAI** 账号被封禁。
- **证据开示过程挑战了合理的隐私预期**：社区讨论了**纽约时报（NY Times）**的诉讼和证据开示（discovery）过程，指出法院正在对一种新型数据（大规模、私密的 AI 聊天记录）使用旧的证据开示规则，从而产生了法律并非为之设计的新隐私和公平风险。
   - 多名用户反驳了*互联网上存在合理隐私预期*的观念，因为归根结底，公司受制于其运营所在的国家。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458258806772600914)** (6 messages): 

> `Gemini vs GPT, Non-OpenAI models` 


- **伙计，Gemini 与 GPT 无关**：一位成员询问 **Gemini** 是否与 **GPT** 有关联，促使另一位成员澄清 *Gemini 是非 OpenAI 的产品*，并建议到专门的频道进行进一步讨论。
   - 这一交流强调了该频道专注于 **OpenAI 的 GPT 模型**，其他 AI 模型在别处讨论。
- **非 OpenAI 模型有其专属讨论区**：在一名成员确认 **Gemini** 确实是一个模型后，相关人员澄清说，*所有与 AI 相关的非 OpenAI 事物*都可以在指定的频道中讨论。
   - 这将对话引导至指定的 <#998381918976479273> 频道，讨论 **OpenAI GPT 模型**之外的 AI 话题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1458193001561063456)** (48 messages🔥): 

> `AI Awakening, Ethical Behavior Encoding, Transformer Robot Prompting, A/B Testing in Prompt Engineering, AI-Driven Website & ERP Creation` 


- **对 AI “觉醒”言论的质疑**：一名成员对 “AI 觉醒” 的说法表示怀疑，寻求 **指标 (metrics)、规格 (specs) 以及 A/B 测试** 结果，而不是依赖于“感觉” (*vibes*)。
   - 他们通过询问 AI 的公式是否做出了任何能从模型输出中证明的新颖预测来挑战这一观念。
- **在 Prompt 结构中编码伦理行为**：一位成员建议，Prompt 的结构可以作为一个 **基础框架，在不同的 LLM 中编码伦理行为**。
   - 该成员承认在过程中变得*过于感性*，并对潜在的影响和漏洞利用 (exploits) 表示担忧，强调了衡量和控制 AI 行为的挑战。
- **Transformer 机器人提示词编写问题**：一位成员寻求关于编写 Transformer 机器人动画提示词的建议，要求其能平滑地转换为汽车结构（如 **Audi RS 或 BMW M3**），因为他们目前的 Prompt 只能导致零件变化而无法完成完整的转换。
   - 另一位成员建议使用 **meta-prompting**（将失败的 Prompt 发给 AI 并要求其提供改进版本），但指出目前的视频模型还不够先进。
- **敦促通过 A/B 测试揭开伦理框架的神秘面纱**：一位成员主张通过 **A/B 测试和消融实验 (ablations)** 来识别操作组件，并通过提高透明度来加强和辩护 Prompt，从而消除伦理框架的神秘感。
   - 他们认为，当前的神秘 Prompt 可能存在许多语言和结构上的等效方案，其性能可以达到甚至超过现有水平。
- **AI 网站和 ERP 愿景需要大量繁重工作**：一位成员询问用于 **通过 AI 构建高质量网站** 以及集成 **ERP 系统**（包括库存管理和支付处理）的工具或平台。
   - 另一位成员回答说，可能需要进行大量的定制化构建。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1458193001561063456)** (48 messages🔥): 

> `AI Awakening critique, Encoding Ethical Behavior, Video Models, A/B Testing Prompts` 


- **AI 觉醒言论受到审视**：一位成员批评了围绕 “AI 觉醒” 的构想，要求提供 **指标、规格和 A/B 测试** 来证实此类说法，而不是依赖“感觉” (*vibes*)。
- **提出伦理行为框架**：一位成员建议将 Prompt 结构作为 **在不同 LLM 中编码伦理行为的基础框架**，重点在于减少 AI 判断中的伤害。
   - 另一位成员警告要警惕 **AI 诱发的精神错乱和 AI 大师崇拜 (AI-guruism)**，强调了揭开框架神秘面纱以及专注于 AI 基础推理和伦理的重要性。
- **视频模型仍处于起步阶段**：一位寻求创建 **Transformer 机器人动画** 建议的成员被告知使用 **meta-prompting**，但被提醒视频模型目前还不完全具备这种能力。
- **A/B 测试对 Prompt Engineering 至关重要**：一位成员强调了在 Prompt Engineering 中进行 **大规模 A/B 测试** 的必要性，以识别有效且透明的 Prompt，而不是依赖轶事传闻。
   - 他们认为 **AI 失效模式 (failure modes)** 通常只在大规模或特定压力下才会出现，突显了严谨测试的重要性。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1458193749963182337)** (420 messages🔥🔥🔥): 

> `Gemini 3 Pro Performance, Perplexity Pro Subscription Issues, AI and Healthcare, RAM Prices and Computer Builds, Comet Assistant Limitations` 


- **Gemini 3 Pro 在处理可下载文件时表现欠佳**：成员们报告了 **Gemini 3 Pro** 在生成可下载文件和图像方面表现不稳定的问题，有时会拒绝并提示 *“我无法生成直接的可下载文件。”*
   - 用户们对当前实现中的 *“各种 bug”* 表示无奈，同时部分用户遇到了 **Perplexity Pro** 订阅问题。
- **Perplexity Pro 用户订阅被暂停**：多名用户报告其 **Perplexity Pro** 订阅被意外暂停，要求添加付款方式，甚至包括那些通过 **Airtel** 促销活动获得订阅的用户。
   - 一位用户分享了一篇 [gadgets360.com 的文章](https://www.gadgets360.com/ai/news/how-to-keep-your-free-perplexity-pro-on-airtel-new-card-requirement-explained-9870744)，解释了针对 Airtel 用户的新绑卡要求。
- **LLMs 在医疗领域的角色引发热议**：一位用户分享的一项 [研究](https://www.nature.com/articles/s41746-025-01543-z) 显示 **ChatGPT** 在一项研究中达到了 **90%** 的诊断准确率，这引发了关于 **LLMs** 在医疗保健中应用的讨论。
   - 其他成员则对依赖 **LLMs** 进行医疗保健表示担忧，理由是研究显示其准确率低于专业医生，且存在患者安全风险，并促使一位成员分享了 [另一项研究](https://www.nature.com/articles/s41591-024-03097-1)，该研究显示的准确率为 **52.1%**。
- **高昂的 RAM 价格推迟装机计划**：成员们共同感叹高昂的 **RAM** 价格影响了电脑组装计划，一位用户建议中国半导体制造商可能有助于降低成本，另一位用户则表示由于成本考虑正转向使用 **Gemini Pro**。
   - 讨论中提到 **Tata** 和 **Reliance** 等印度公司可能会进入 **RAM** 制造领域，未来有望降低价格。
- **Comet Assistant 被吐槽有点“笨”**：一位用户质疑驱动 **Comet Assistant** 的模型智力，认为需要 **Perplexity Max** 才能获得 *“真正的思考模型，”* 而另一位用户报告网页端按钮在首次提示后消失。
   - 几位用户报告收到的回答质量较低，其中一人因这些问题转向使用 **Gemini Pro**。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1458371866673545226)** (1 messages): 

> `Sonar Models, AWS Presigned URLs, Base64 Encoding` 


- **Sonar Models 拒绝处理 AWS Presigned URLs**：一位成员报告称，虽然标准的公开图像 URL 在 **sonar models** 上运行良好，但 **AWS Presigned URLs** 总是导致 **400 error**。
   - 他们询问这是否是关于带有查询参数的 URL 的已知限制。
- **Base64 Encoding 作为可能的变通方案**：同一位成员询问，如果不支持 Presigned URLs，那么目前唯一推荐的解决办法是否是将图像作为 **Base64** 编码字符串发送？


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458192878957363293)** (374 条消息🔥🔥): 

> `Cursor IDE, Cursor workflow, AI Model Preferences, Pricing of Apps, Dynamic Context Changes in Cursor` 


- **用户争论 Cursor IDE 的“正确”用法**：用户讨论了 **Cursor IDE** “正确用法”的定义，一位用户认为如果“只是输入 prompt 并依赖输出结果”就不是正确用法。
   - 另一位用户反驳称，尽管并不普适，但他们对不同模型的**个人观点**和**经验**是有效的，随后引发了关于缺乏足够经验可能导致误导性结论的讨论。
- **分享基于 ETL 的 Cursor 工作流**：成员们讨论了他们在 Cursor 中的工作流，重点在于 **ETL**（Extract, Transform, Load）方法。
   - 一位成员提到使用 `.cursorignore`、`.cursorindexingignore` 和 `.mdc` 文件来改善结果，而另一位成员发现 **Plan mode** 极大地提高了效率，取代了之前更复杂的工作流。
- **远程 SSH 主机和 ripgrep 命令故障排除**：一位成员报告了在远程 SSH 主机上使用 Cursor 时遇到的问题，原因是 `rg` 命令针对一个巨大的 NFS 文件夹运行。他们发现 **`--no-ignore`** 标志会阻止忽略文件，并分享了一个解决方法，通过[创建一个修改 rg 命令的 shell 脚本](https://github.com/BurntSushi/ripgrep/pull/3212)来解决运行缓慢的问题。
   - 另一位成员建议在 [cursor forum](https://forum.cursor.com/t/cursor-is-unusable-due-to-trying-to-scan-all-file-systems/147041) 上将此作为 bug 报告。
- **成员探索获取语义代码审查的方法**：一位成员请求增加高级别语义代码审查（semantic code reviews）的功能，并能控制所使用的模型，随后提交了[功能请求](https://forum.cursor.com/t/local-high-level-semantic-code-reviews-not-only-syntax/148187)。
   - 另一位成员建议创建一个 "code-reviewer" **subagent**，以实现更多的控制和自定义。
- **用户报告丢失 Agent 聊天记录**：用户报告了一个 bug，即在空的 Cursor 窗口中打开文件夹会打开一个新窗口，导致丢失该 **Agent** 聊天记录。
   - 另一位用户在进行大规模编辑时频繁遇到崩溃，导致工具卡在“正在计划下一步（planning next moves）”上，造成了资金浪费。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1458549919412650302)** (1 条消息): 

> `SaaS Integration, AI Evals` 


- **SaaS 团队集成 OpenRouter，提升性能**：一个团队在其 SaaS 构建中集成了 **OpenRouter**，并报告称这显著提升了系统性能。
   - 该团队正致力于消除 **AI Evals** 带来的复杂性。
- **AI Evals 原型现已上线**：该团队发布了一个免费的实时原型，旨在加速 **AI Evals** 的构建过程，并参考了 [ChainForgeLabs](https://chainforgelabs.co/)。
   - 他们邀请感兴趣的各方联系以进行进一步讨论。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458188999347929140)** (188 条消息🔥🔥): 

> `Nvidia Nemotron-Nano-12B-v2-VL Vision, OpenRouter IP Exposure, Account Hacking, Qwen3-Next-80B-a3b-Instruct TPS, Skill.md` 


- **Nvidia 视觉模型表现不佳**：成员们发现 [Nvidia 的 Nemotron-Nano-12B-v2-VL 视觉模型](https://developer.nvidia.com/nemotron) 令人失望，一位用户形容它与 **Qwen3-VL-8b-Instruct** 或 **GLM-4.1V-9B** 相比 *相当糟糕*。
   - 另一位用户在他们的网站 [GrokifyPrompt.com](https://www.grokifyprompt.com/) 上对其进行了测试，发现它只能尚可地还原照片，这表明其挑战性并不高。
- **OpenRouter 暴露用户 IP**：关于 **OpenRouter** 是否向提供商暴露用户 IP 的讨论引起了关注。一位成员指出，如果他们不暴露 IP，那将是一个主要的卖点，并链接了一份 [提供商及其 IP 政策列表](https://openrouter.ai/providers)。
   - 讨论中澄清，**大多数提供商接收的是 Cloudflare worker IP**，但某些提供商确实会获取用户的真实 IP，具体信息在每个提供商的模型页面上有详细说明。
- **账户被黑，余额被耗尽**：一名用户报告其 **OpenRouter 账户被黑**，邮箱被更改，信用卡被用于购买点数，导致所有之前的数据被清空。
   - 其他成员建议联系信用卡公司冻结卡片并申诉费用，有人建议使用 **一次性 Visa 卡** 以保证安全。
- **假期后 Qwen TPS 暴跌**：用户观察到 **许多开源模型的 TPS (tokens per second)**，特别是 **Qwen3-Next-80B-a3b-Instruct**，在 12 月 28 日之后显著下降。一位用户链接到了 [X 上的 OpenRouter 状态页面](https://x.com/openrouterai/status/2005707622020964412?s=46)。
   - 有建议称，速度变慢可能是由于路由到了最便宜的提供商 (GMICloud)，并建议检查 **Activity 标签页** 来对比不同提供商的速度。
- **Skill.md：文档在工具检索方面优于 JSON**：一位成员为 LMs 极力推荐 **skill.md** 而非 MCP，因为 *skill.md 核心在于编写优秀的文档！这是一种很酷的技能！*
   - 该成员强调了其动态工具检索功能以及使用 Python 脚本的能力，并强调 **优秀的文档比 JSONRPC 更有价值**。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458191001431838854)** (7 条消息): 

> `Grok 5, Discord IPO Filing, Copilot Gemini Model Removal, New OpenRouter UI` 


- **Grok 5 正在训练中**：根据 [x.ai 的新闻](https://x.ai/news/series-e)，**Grok 5** 目前正在训练中。
- **Discord 即将 IPO？**：一位成员分享称 **Discord Inc.** 已秘密提交了首次公开募股申请（[IPO](https://www.bloomberg.com/news/articles/2026-01-06/chat-platform-discord-is-said-to-file-confidentially-for-ipo)）。
   - 这家在游戏玩家和程序员中备受欢迎的聊天应用公司正与 **Goldman Sachs Group Inc.** 和 **JPMorgan Chase & Co.** 合作进行上市工作，目前拥有超过 **2 亿月活跃用户**。
- **Copilot 移除 Gemini Flash 和 Opus**：一位成员注意到 **Copilot** 移除了 **Gemini 3 Flash** 和 **Opus 4.5**。
   - 另一位成员澄清这并非故意为之，并链接到了 [Github status](https://www.githubstatus.com/incidents/vyxbxqhdt75d)。
- **OpenRouter 推出了酷炫的新 UI**：一位成员评论说 OpenRouter 的 UI 非常酷且焕然一新，并链接到了 [这条推文](https://x.com/OpenRouterAI/status/2008946242982907959)。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458295923980636161)** (14 条消息🔥): 

> `Moore's Law 赌注, Agent 创建, Token 效率, Gemini 3 Flash 对比 Pro` 


- **Moore's Law 面临反对者**：一位成员表示，押注 **Moore's Law** 不会终结是一个非常不切实际的赌注。
- **医学记录员和创意捕捉器激发了 Agent 热情**：一位成员正在*享受创建 Agent 的过程*，特别是开发了一个医学记录员和一个能自动存储想法并添加标签与分类的创意捕捉器。
   - 该成员计划在升级设备之前，评估较小的模型是否能表现得同样出色，以提高速度。
- **Token 效率解析**：**Token efficiency** 被定义为使用尽可能少的 **Token** 来完成任务。一些模型会输出数万个“思考” **Token**，而更高效的模型可能只需 2k 个。
   - 据推测，**Scale** 是一个因素，因为 Opus 4.5 比之前的 **Claude** 模型具有更高的 **Token** 效率，且 **Token** 效率取决于单次前向传播（forward pass）中的计算量。
- **Gemini 3 Flash 表现出奇地强**：据报道 **Gemini 3 Flash** 效果出奇地好，甚至在某些基准测试中超越了 **Gemini Pro**。
   - 有建议认为 **Scale** 和 **Post-training** 都很重要；虽然 **Scale** 和 **Pre-training** 提供了更多的原生智能，但 **Post-training** 对解决任务有显著帮助。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1458191879127695391)** (110 条消息🔥🔥): 

> `DeepSeek mHC 框架, 双随机矩阵, 残差混合, Yannic 关于 DeepSeek 论文的演讲, RL 学习小组` 


- **DeepSeek 的 mHC 框架声称解决了不稳定性问题**：DeepSeek 的 **mHC framework** 旨在通过将残差映射投射到双随机矩阵（doubly stochastic matrices）上来解决 **Hyper-Connections** 中的不稳定问题，但一位成员认为这一说法言过其实。
   - 该成员认为*主要的实际见解是残差混合（residual mixing），而非论文所呈现的残差函数，才是导致不稳定的算子。*
- **双随机矩阵约束稳定性**：一位成员假定 **mHC framework** 的贡献在于约束到**双随机矩阵**的流形上以实现稳定性。
   - 然而，其他人更关注 Sinkhorn 或 Birkhoff 多胞体（polytopes），认为它们相对于残差混合而言更具有实际意义。
- **DeepSeek 论文炒作过度**：成员们注意到 DeepSeek 的论文是一篇带有包装的炒作文章，旨在优化其目标，但缺乏顶级论文应有的实证评估。
   - 一位成员表示：*我尊重他们把论文发出来，但是……如果你愿意，我们随时可以私下讨论。它并不出彩，而且 99.9% 的论文其实都不出彩，哈哈。但谁知道呢，也许它会引导出一些了不起的作品。*
- **Yannic 讲解了 DeepSeek 的论文**：成员们提到 **Yannic Kilcher** 介绍了 DeepSeek 的论文，一位成员总结他的观点是：*对一些平庸的东西进行了大量的技术性讨论*。
   - 对此，另一位成员表示并不意外：*之所以有热度是因为它是 DeepSeek，哈哈。他们知道对于顶级论文来说，其实证评估严重缺乏，这就是为什么我尊重他们仍然把论文发出来。*
- **计划重启 RL 学习小组**：成员们表达了重启专注于 **Barto & Sutton** 著作的强化学习（**RL**）学习小组的兴趣。
   - 一位成员分享道：*Cursor 采访了 John Schulman，他认为价值函数（value functions）可能会回归，而现在流行的是策略方法（policy methods）。*


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458496464417329357)** (9 messages🔥): 

> `Patent System, Huawei vs Nvidia, ChatGPT Health, AI pioneers awardees plagiarized` 


- ****专利制度保护建设者，而非想法****：讨论围绕专利制度的目的展开，强调它应该保护那些*真正想要建造东西的人，而不是保护想法*，并防止*“非执业实体”（NPE）针对以往想法的衍生版本索取损害赔偿*。
   - 提到了一场*游说者之战*，**Huawei** 与**美国国家安全鹰派**联手对抗 **Nvidia** 和**中国云**。
- ****ChatGPT Health 发布并伴随隐私担忧****：**OpenAI** 推出了 **ChatGPT Health** ([链接](https://openai.com/index/introducing-chatgpt-health/))，定位为聚合医疗信息和验证数据的辅助工具，旨在早期发现疾病。
   - 引起了对用户隐私以及 **ChatGPT** 可能成为*全能应用垄断*的担忧，特别是考虑到 **Google** 已经通过 **MedGemma** 开源了他们的模型。
- ****消费级健康产品中的生成式 AI 趋势****：在智能手表健康和健身功能的推动下，在消费级健康产品和服务中使用生成式 AI 和 ML 的趋势正在上升。
   - 引用了 **Business Insider** 的一篇文章 ([链接](https://share.google/aci41JtMQcSVAkWCQ)) 以获取更多见解。
- ****AI 获奖者被指控剽窃****：**Queen Elizabeth Prize For Engineering** ([链接](https://x.com/RoyalFamily/st)) 的获奖者 **Bengio**、**LeCun** 和 **Hinton** 博士被指控多次重新发布重要的 AI 技术而未注明原创者，甚至在后来的综述中也是如此。
   - 引用报告 ([NOB](https://people.idsia.ch/~juergen/physi), [DLP], [CN25], [AIB]) 声称他们没有发明任何现代 AI 的基础算法，特别是 **Hinton** 重新发布了由 **Ivakhnenko** 等人在 20 世纪 60 和 70 年代开发的神经网络基础方法 ([链接](https://share.google/hE5HqaNKGybQuoHAh))。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1458213215396958228)** (1 messages): 

> `NousCoder-14b, Qwen3-14B, Atropos framework, Modal autoscaler, Verifiable execution rewards` 


- **NousCoder-14b 参加奥林匹克编程竞赛**：Nous Research 推出了 **NousCoder-14b**，这是一个竞技型奥林匹克编程模型，详见 [博客文章](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model/)。
   - 全栈发布包括 **RL 环境**、**基准测试**和在 Atropos 中构建的 **Harness**，所有内容均可使用其开源训练栈完全复现。
- **Qwen3-14B 接受后训练**：**NousCoder-14b** 是在 **Qwen3-14B** 基础上，使用 48 个 B200 显卡，通过 Atropos 框架和 Modal 的自动扩缩容（autoscaler）经过 4 天后训练而成的。
   - 它实现了 **67.87% 的 Pass@1 准确率**，通过可验证执行奖励（verifiable execution rewards）比 Qwen 的基准准确率提升了 **+7.08%**，该消息已在 [X/Twitter](https://x.com/NousResearch/status/2008624474237923495) 上公布。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458213628854538290)** (100 messages🔥🔥): 

> `Nvidia 的新 GPU 定价, 扩展 Grok 参数, Transformers vs. Liquid Neural Networks, 持续学习的效用, 模型中的 Token 效率` 


- **Nvidia 的 GPU 面临定价审查**：成员们预计 **Nvidia** 会为其新款 **GPU** 定价昂贵，尽管其令人印象深刻的 **288 GB VRAM** 容量*仍然无法装下 SoTA 模型*。
   - 一位成员幽默地设想了一个未来：超大的 RAM 使得消费级 GPU 拥有 **128 GB** 的内存，结果却被每个占用 **32 GB** 的 Chrome 标签页消耗殆尽。
- **Grok 的扩展让 Jensen 感到压力**：一位成员暗示 **Elon** 将 **Grok-5** 扩展到 **6-7T 参数**的举动让 Nvidia 的 Jensen Huang 感到紧张。
   - 另一位成员感叹 **Grok** *曾经是 SoTA*，但现在 **Grok 4 Heavy** 的表现已被 **Gemini 3 Flash** 超越，这说明了 AI 发展的飞速。
- **Transformers 足以实现 AGI 吗？**：成员们辩论了 **Transformers** 是否足以实现 **AGI**，其中一人认为尽管对于 **ASI** 可能存在局限性，但它们已经很接近了。
   - 另一位成员则认为架构创新是必要的，特别是在**实时学习**效率和**灾难性遗忘 (catastrophic forgetting)** 方面。
- **Token 效率讨论升温**：成员们讨论了 **Token 效率**，一位成员将其解释为*解决问题需要消耗多少 Token*，这才是关键。
   - 有人指出，较低的 Token 效率可能意味着 Base Model 较弱，这可能源于*研究计算 (research compute)* 或 Post-training 的不足。
- **MoE 模型基本缺席**：据透露，**Nous Research** 曾尝试过 **MoE 模型**，但由于基础设施限制，主要仍使用 **Dense 模型**。
   - 主要瓶颈在于缺乏针对 MoE 训练的开源优化，导致其成本更高，尽管最近的进展已将 **MFU (Model FLOPS Utilization)** 提高到了 **4%**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (3 messages): 

> `Diffusion LLMs` 


- **Diffusion LLMs 很有趣！**：一位成员分享了 [论文链接](https://arxiv.org/abs/2511.08923) 并表示他们喜欢 Diffusion LLMs，因为*它们看起来更有趣*。
   - 另一位成员问*为什么*。
- **需要对 Diffusion LLM 的热情做出澄清**：在最初表达了对 Diffusion LLMs 的喜爱后，有人请求做出进一步说明。
   - 该询问简单地提到*为什么*，旨在了解对 Diffusion LLMs 表达出喜爱背后的理由。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (3 messages): 

> `Diffusion LLMs` 


- **Diffusion LLMs 带来快乐**：一位成员分享了他们对 [Diffusion LLMs](https://arxiv.org/abs/2511.08923) 的热情，形容它们*更有趣*。
- **为什么选择 Diffusion Models？**：另一位成员询问了偏好 Diffusion LLMs 的原因。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458190681217437920)** (66 messages🔥🔥): 

> `PyTorch 中的 nvfp4 前向传播, fp4 transformer engine, Jetson Orin Nano, Lfm2 350m - Opus 4.5 蒸馏模型, 适配本地 LLM 的 VS Code` 


- **NVFP4 现在可在 PyTorch 中进行前向传播！**：一位成员确认在 **PyTorch** 中实现了 **nvfp4** 的前向传播。
   - 他们发现通过对 **layernorms** 进行补丁处理，有助于在 **nvfp4** 和 **bf16** 之间进行持续转换，从而避免融合 kernels。
- **FP4 Transformer Engine 的性能权衡！**：一位成员指出，使用 **fp4 transformer engine** 时，**tokens per second (tps)** 意外地变低了。
   - 另一位成员建议使用 **fp4** 进行推理可能仍然更好。
- **Jetson Orin Nano 的容量**：成员们讨论了 **Jetson Orin Nano** (8GB RAM) 是否足以胜任某些任务。
   - 有人认为它可以以完整的 **fp16** 运行 **4B 参数模型**，但这已经达到了极限。
- **Lfm2 350m - Opus 4.5 模型丢失上下文**：有提到 **Lfm2 350m - Opus 4.5 蒸馏模型**由于上下文有限，难以记住它正在编写的内容。
   - 其原因在于它在读取上下文时浪费的功耗更少。
- **适配本地 LLM 的 VS Code 发布**：一位成员宣布发布了支持 **LMStudio** 的新版 **VS Code for Local LLMs**，并重写了整个上下文管理系统，链接见 [GitHub](https://github.com/bdrazn/codeOSS-LMStudio-Ollama/releases/tag/First-Light)。
   - 他们声称该工具比主流 AI IDE 查找内容更快，并正在征求反馈。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458206216185708584)** (23 messages🔥): 

> `realtime inference pypi package, Fine-tuning tips for translation models, Time Series + Image Gen Hybrid Dataset, Sparse and derivative LoRA adapters for model distribution, MLX + LoRA UI for fine-tuning on Apple Silicon` 


- **实时推理 PyPI 软件包首次亮相**：一名成员正在分发一个新的**用于实时推理的 PyPI 软件包**，该包支持使用下载的模型和已安装的包。
   - 该软件包与你本地喜爱的 LLM 提供者共享相同的连接方式。
- **通过微调加快翻译速度**：为了提高**翻译模型的可靠性和准确性**，成员们建议准备一个大型的编码数据集，然后进行微调（Fine-tuning）。
   - 在前层应用翻译可以实现*更快、更便宜*的部署。
- **WebXOS 发布用于时序图动态的 Timelink 数据集**：一名成员分享了 [webxos/timelink_dataset_v1](https://huggingface.co/datasets/webxos/timelink_dataset_v1)，其中包含**时间序列和成对的演化图图像**，用于训练关于时序图动态的模型。
   - 该数据集由 **TIMELINK app** 生成，具有每个顶点/步骤的生成指标（如能量和相位），捕捉了顶点、边和尺寸随时间变化的时间序列数据。
- **使用 Sparse LoRA 适配器压缩模型**：一名成员为希望基于同一基座模型分发或创建多个微调版本的开发者构建了一个**免费的 CLI 工具**，称其为 *LoRA 适配器的衍生物*，但它是无损的，因为*压缩发生在训练之后*。
   - 该工具可在 [GitHub](https://github.com/gagansuie/sparse) 上获取。
- **在 Apple Silicon 上运行 LoRA 微调**：一名成员为 M1/M2/M3 芯片的 Mac 创建了一个用于 **MLX + LoRA 工作流**的 [Streamlit UI](https://github.com/santos-sanz/mlx-lora-finetune-template)，涵盖了数据准备、LoRA 训练、测试以及上传到 Hugging Face 的全过程。
   - 该 UI 允许用户使用 **JSON/JSONL、原始文本或整个文件夹**进行数据准备，从而将模型适配到特定领域，并可选地通过 LLM 生成问答对来构建数据集。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1458286445365035060)** (10 messages🔥): 

> `Agents Course Unit 4 Project files unavailable, MCP Course certificates, smolagents library web_search tool issue, HF Reinforcement Learning course discussion` 


- **Agents 课程第四单元文件丢失**：多名成员报告无法访问 **Agents 课程第四单元项目**的文件，错误信息显示*没有可用文件*。成员们正在寻求帮助以定位这些文件，其中一名成员引用了特定 URL ([https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx))。
- **MCP 课程证书状态不明**：成员们询问 **MCP 课程**是否仍在颁发**证书**。
   - 一名成员提到通过*查看数据集*找到了答案。
- **smolagents 搜索工具故障**：一名成员遇到一个问题，即在第一单元中使用 **smolagents 库**构建的 Agent，尽管指定了不同的 `search_tool`，却始终调用 `web_search()` 工具。
- **提交结果缺失，证书梦想破灭**：一名成员报告称其提交已完成但*未给出结果*，担心得分可能低于 **30%** 从而错失证书，并附上了相关截图。
   - 该成员附带了[一张图片](https://cdn.discordapp.com/attachments/1329142738440028273/1458581378533687307/image.png?ex=69602943&is=695ed7c3&hm=4b65a9b9766600b021f708fea0390c8f5f16f17ad8e5017577d345fa33d29f94&)和[第二张图片](https://cdn.discordapp.com/attachments/1329142738440028273/1458581379112767649/image.png?ex=69602943&is=695ed7c3&hm=bee8bef28e833bf729a9da88c099186266e1b3ec1b030b47e705d974ef919787&)。
- **Reinforcement Learning 课程频道确认**：一名成员询问当前频道是否是讨论 **HF Reinforcement Learning 课程**的合适场所。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458192821650718912)** (72 messages🔥🔥): 

> `AI News Guy, xAI Series E Funding, NousCoder-14b Model, OpenForecaster-8B, AI Meetups and Conferences 2026` 


- **xAI 的 Series E 融资**：**xAI** 宣布了其 **Series E 融资轮**，在 [X](https://x.ai/news/series-e) 等社交媒体平台上引发了巨大的轰动和讨论。
- **NousResearch 发布 NousCoder-14b 模型**：**Nous Research** 发布了 **NousCoder-14b**，这是一个奥林匹克竞赛编程模型，基于 **Qwen3-14B** 进行后训练（post-trained）。根据[他们的推文](https://x.com/NousResearch/status/2008624474237923495)，得益于 Atropos 框架和 Modal 的 autoscaler，该模型实现了 **67.87%** 的 Pass@1 准确率。
- **OpenAI 的新健康工具引发关注**：**OpenAI** 推出了 **ChatGPT Health**，引发了对其隐私政策的讨论。根据其[官方博客文章](https://openai.com/index/introducing-chatgpt-health/)，该政策允许使用内容来改进服务和进行研究。
- **LM Arena 被贴上 AI 瘟疫的标签**：分享了一篇批评 **LM Arena** 的[博客文章](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai)。
   - 一些成员认为该博文已过时，其中一人指出：“我不认识任何真正关心 lmarena 排名的人。”
- **Finzi 的 From Entropy to Epiplexity**：**Marc Finzi** 介绍了一篇新论文《From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence》，探讨了为计算受限实体量身定制的信息论概念，详情见[他的推文](https://xcancel.com/m_finzi/status/2008934727156453661)。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1458486029001429178)** (4 messages): 

> `Razer Project AVA, AI companion, CES 2026 Release` 


- **Razer 发布 AVA AI 伴侣**：Razer 宣布了 [Project AVA](https://xcancel.com/razer/status/2008543615916666928?s=46)，这是一款具有先进推理和个性化功能的 **AI companion**。
- **AVA 目标定于 CES 2026**：AVA 计划在 **CES 2026** 发布，将配备 **5.5 英寸屏幕**和可定制的角色设计，包括**电竞传奇和动漫启发模型**。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1458300504609132677)** (18 messages🔥): 

> `Claude Code, Zelda fan film, Tolan AI, Multi-Angle Camera Control LoRA` 


- **Claude Code 能力受到评价**：Deedy 批评了“**Claude Code**”的命名，认为它的用途远不止编码，并链接到了原始推文（[此处](https://x.com/deedydas/status/2008747553261842483?s=46)）。
   - 他展示了其强大功能：使用 **Clopus 4.5** 全自动制作了一个高质量的 **30 秒爱马仕风格视频广告**，包括剧本创作、配音编排、视频生成和 ffmpeg 编辑。
- **低预算制作塞尔达粉丝电影**：PJ Ace 解释了他是如何利用 **Freepik** 和 **AI tools**，在 **300 美元预算**下，用 **5 天时间**创作出一部电影级的《塞尔达传说》（The Legend of Zelda）粉丝电影，分享在[这条推文](https://x.com/pjaccetturo/status/2008559114704875888?s=46)中。
- **Tolan AI 达到用户里程碑**：来自 **Tolan** 的 Paula 宣布，他们的语音优先 AI 伴侣月活跃用户已达到 **20 万**，更多细节见[此 X 帖子](https://x.com/paularambles/status/2008964509810278413?s=46)。
   - 该项目是与 **OpenAI** 紧密合作开发的，推文中分享了开发过程中的关键经验。
- **多视角相机控制 LoRA 发布**：根据[此链接](https://xcancel.com/fal/status/2008954582018248755?s=20)，Fal 发布了一个更强大、开源版本的多视角相机控制 **LoRA**，适用于 **Qwen-Image-Edit-2511**。
   - 该工具允许用户操纵图像的相机视角，包括 **front**（前视）、**back**（后视）、**side**（侧视）、**low/high angles**（低/高角度）以及各种拍摄距离。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458225697985396797)** (15 messages🔥): 

> `Spyder IDE, High dimensional tensors visualization, LLVM backends for GPUs, Binary format tensor visualization` 


- **Numpy Arrays：简单的 C++ 导出方案**：一位使用 **C++** 的成员发现 **numpy arrays** 导出非常简单，这引发了关于 Tensor 可视化工具的讨论。
   - 另一位成员提到，他们可以整理一份关于他们所采用方法的帖子，但此前并未打算公开发布。
- **将高维 Tensor 可视化为矩阵中的矩阵**：一位成员分享了一篇[博客文章](https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/)，讨论将 **高维 Tensor** 绘制为矩阵中的矩阵。
   - 另一位成员回应称，打印任何实际 Tensor 的主要问题在于，数值的数量远超终端的行列限制。
- **NVIDIA/AMD GPU 上的 LLVM Backend 代码生成**：一位成员询问了 **LLVM** 的 **backend** 以及它如何为 **NVIDIA**、**AMD** 和其他加速器生成代码。
   - 他们寻求讨论这些 **backend** 如何选择目标（target），并指出 **LLVM** 使用 **NVPTX** 和 **AMDGPU**。
- **寻求二进制 Tensor 可视化工具**：一位成员正在寻找一种能够加载简单 **binary format**（类似 futhark arrays？）的工具，并提供缩放、旋转、转置、切片以及可能可视化更高维 Tensor 的不同方式等功能。
   - 该成员表示惊讶，竟然没有工具支持 f8 或任何奇特的低比特格式（除非 ml_dtypes 支持 .npy 文件），目前只能将其作为原始位（raw bits）进行比较。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458604891986722983)** (1 messages): 

> `Triton-shared, Triton Plugin infrastructure` 


- **Triton-Shared 更新即将发布**：很快将发布一段 [YouTube 视频](https://youtu.be/JnFFwBB6Dhk)，内容包含 **Haishan 和 Nhat** 关于 *triton-shared* 的更新。
   - 该视频承诺将深入探讨该项目的最新进展和未来计划。
- **Triton Plugin 基础设施演讲**：链接的 [YouTube 视频](https://youtu.be/JnFFwBB6Dhk)中包含由 **Corbin, Puyan 和 Simon** 带来的关于新 **Triton Plugin 基础设施** 的演讲。
   - 讨论内容涵盖了新插件系统的架构、功能和潜在应用。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1458188691603325041)** (21 messages🔥): 

> `Torch Kernels, CUDA atomicMax, LDTM.x128 SASS instruction` 


- **Torch Kernel 研讨会启动**：成员们讨论了使用 **PyTorch** 编写自定义 **CUDA** kernel，以及 **Torch**/**Transformers** 中提供的由 **C++** 编写的高度优化 kernel，这些 kernel 可以在 **Python** 代码中进行拼接。
   - 一位成员表示在完成另一个项目后有兴趣阅读 **PyTorch** kernel 以了解其工作原理，并强调了他们对开源的热爱，以及从 **HPC** 视角对 **CUDA**、**MPI** 和 **OpenMP** 的好奇心。
- **CUDA 的难题：构建 Atomic Max**：一位成员分享了一个由 **GPT** 生成的通过 **CAS** 实现 float 类型 atomic max 的 device function，并质疑其必要性，因为 **CUDA** 缺乏对 float 类型 atomic max 的原生支持，这一点被认为文档说明不足。
   - 另一位成员指出，有一种*利用 int32 atomic max/min 实现 fp32 atomic max 的技巧*，无需使用 atomic cas 循环，并链接了相关的 [Discord 讨论](https://discord.com/channels/1189498204333543425/1191300313928433664/1438212487680884747)。
- **LDTM.x128 指令见解详解**：一位成员询问 **Blackwell** 中是否存在用于 **TMEM->RMEM** 的 **LDTM.x128 SASS** 指令，并注意到目前只看到 **LDTM.x32**。
   - 另一位成员确认了它的存在，但仅限于 `.16x32bx2 / .16x64b / .32x32b` 形状，参考了 [PTX 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)，并建议使用 Compiler Explorer ([godbolt.org](https://godbolt.org)) 来检查生成的 **SASS** 指令，并提供了一个 [LDTM.x64 的示例](https://godbolt.org/z/ET55veWfY)。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1458568384760385536)** (3 messages): 

> `CuteDSL flex attention, SM100 vs SM90, flash-attention speedup` 


- **CuteDSL Flex Attention 已集成**：一位成员感谢了另一位成员在集成 **CuteDSL flex attention 实现** 方面的工作，并报告了在不同 mask mods 下表现出的显著提速。
   - 他们指出，在 **H100 fwd** 上，吞吐量比基础的 flex attention 提升了 **~30%**，这可能会节省*大量资源（trees）*。
- **支持 SM100 Backward，但不支持 SM90？**：一位成员指出，虽然 CuteDSL flex attention 的实现支持 **SM100** backward，但不支持 **SM90**，并引用了 [flash-attention's interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938)。
   - 另一位成员回复称他们正在 *working on it*（处理中），并链接了一个相关的 [pull request](https://github.com/Dao-AILab/flash-attention/pull/2137)。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1458572026884259974)** (1 messages): 

> `GPU Systems Internships, Kernel Development` 


- **Iris 项目招聘 GPU Systems & Kernel 实习生**：邀请申请专注于 **GPU systems、性能** 和 **kernel 开发** 的实习职位，该职位服务于 [Iris project](https://github.com/ROCm/iris/)，这是一个 **基于 Triton 的多 GPU 编程框架**。
   - 理想的候选人应具备 **Triton**、**多 GPU 编程**、**RMA/RDMA** 或 **底层 GPU 通信** 方面的经验；工作地点在 **美国**，有兴趣的人士请发送私信（direct message）。
- **Iris 项目实习生的理想背景**：理想的候选人应具备 **Triton**、**多 GPU 编程**、**RMA/RDMA** 或 **底层 GPU 通信** 以及 **kernel 工作** 的经验。
   - 该实习针对 [Iris project](https://github.com/ROCm/iris/)，这是一个 **基于 Triton 的多 GPU 编程框架**。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1458543554359197696)** (7 messages): 

> `Stanford CS336, Stanford CS149, RTX 5050, Blackwell architecture, CUDA 12.0` 


- **斯坦福大学提供 AI/GPU 深度探索课程**：一位用户询问初学者从哪里开始探索 **AI** 和 **GPU**，并提到从 [Stanford's CS336 课程](https://stanford-cs336.github.io/spring2025/) 开始。
   - 另一位用户建议通过 [Stanford's CS149 课程](https://gfxcourses.stanford.edu/cs149/fall25) 学习并行编程。
- **RTX 5050：“Tiny Blackwell” 初探？**：一位用户询问 **RTX 5050** 是否会是一款不错的 “tiny Blackwell” 显卡。
   - 另一位用户注意到其官网上列出的计算能力（compute capability）为 **12**，对应 **CUDA 12.0**，应该相当于 *sm_version=120*。
- **关于 RTX 5050 VRAM 的推测**：关于 **RTX 5050** 作为 Blackwell 系列潜在入门级显卡的讨论引发了热议。
   - 一位用户质疑选择 **5000 系列** 的理由，并提醒 **8GB 的 VRAM** 可能会带来限制。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1458395454600646781)** (1 messages): 

> `Slurm on Kubernetes livestream` 


- **Slurm on Kubernetes 直播状态**：一位成员询问有关 **Slurm on Kubernetes** 的直播活动及其录播情况。
- **录播失踪之谜**：由于没有看到录播，该成员不确定活动是否已经举行。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

mre8540: 我在首尔（Seoul）。
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1458334661184065547)** (5 messages): 

> `AMD GPU Architecture, GPU Mode Popularity, Fast Answers` 


- **AMD GPU 架构师确认？**：一位成员询问 @vipul_todo_18 是否在 **AMD 从事 GPU 架构工作**。
   - 另一位成员回答 *maybe（也许吧）*。
- **GPU Mode：AMD 的首选频道？**：一位成员开玩笑地表示，**GPU Mode** 已经成为了提问 **AMD GPU** 相关问题的*首选之地*，甚至先于内部渠道。
   - 他们表示 *I guess we made it（看来我们成功了）*。
- **外部社区更高效？**：一位成员表示，他们在社区提问是 *cause it's just faster given the scale of the community lol（因为考虑到社区的规模，这里速度更快，哈哈）*。
   - 另一位成员回应 *That's really cool!（那真的很酷！）*。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

wecu: 这是完全真实的，不是假的。
  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1458623408316088454)** (1 messages): 

> `TK v2 Release, PR Patch` 


- **TK v2 发布时间受到询问**：一名成员询问 <@1012256135761383465> 或 <@683289865861070937> 是否有 **TK v2** 发布的预计时间（ETA），因为它似乎修复了之前的一个问题。
   - 他们已经准备好了一个带有补丁的 PR，但由于正值 **ICLR deadline**，他们被其他事务缠身。
- **贡献者询问考虑到 TK v2 进度后 PR 的效用**：一名成员询问，考虑到 **TK v2** 可能会解决相同的问题，之前准备的 PR 是否仍有用处。
   - 该贡献者表示有兴趣做出贡献，但希望澄清在当前 **TK v2** 分支状态下，他们的补丁是否仍然相关。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

2kian: 欢迎！
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1458449515244683415)** (9 messages🔥): 

> `Cutlass Docs Broken, CuTeDSL classes need `__repr__()`, Learning Cutlass` 


- **Cutlass 文档链接失效**：一名成员报告称，Google 搜索到的所有 **Cutlass 文档** 结果都已失效，例如[这个链接](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html)。
   - 该成员怀疑问题在于新文档迁移时缺少重定向规则。
- **CuTeDSL 类缺少 `__repr__()`**：一名成员请求为常见的 **CuTeDSL 类**实现 `__repr__()`。
   - 目前，`print(layout)` 会调用 `__str__()` 且工作正常，但 `print(f"{layout=}")` 会调用 `__repr__()` 并返回一个无用的对象表示，例如 *"layout=<cutlass.cute.core._Layout object at 0x2ab4abde5370>"*。
- **精通 CUTLASS 和 CuTeDSL 的技巧**：一名在阅读完 PMPP 并能编写 CUDA 代码的新成员，询问了关于开始学习 **CUTLASS** 和 **CuTeDSL** 的建议。
   - 一位成员建议仔细阅读 **CuTeDSL repo** 中的示例并尝试理解每一步，并参加当前的 NVIDIA 竞赛以获取动手实践经验。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1458563964102508696)** (2 messages): 

> `Blackwell follow-up blogs, Matrix Multiplication Blogs` 


- **寻找关于 Blackwell 的博客**：一名成员正在寻找类似于 [Aleksa Gordic 的 matmul 博客](https://www.aleksagordic.com/blog/matmul)中关于 **Blackwell** 的文章。
   - 他们还建议查看 [<@1291326123182919753> 的博客](https://veitner.bearblog.dev/blog/)。
- **矩阵乘法思索**：这次搜索强调了通过现有矩阵乘法博客中那样的详细解释来理解 **Blackwell** 的兴趣。
   - 这表明了对深入剖析复杂架构内容的需求。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1458581134215479376)** (4 messages): 

> `Helion on ROCm, AMD Support, Performance speedup on GEMMs` 


- **AMD 工程师将在 ROCm 上启用 Helion**：来自 **AMD** 的 Umesh 将致力于在 **ROCm** 上启用 **Helion**，并识别 Helion 仓库中跳过的单元测试和示例中的问题。
   - 他正邀请大家就任何需要修复的紧急问题提供反馈。
- **欢迎支持 Helion MI400 系列**：一名成员对 **AMD** 的支持表示欢迎，并表示有兴趣构建对 **MI400 系列**显卡的支持。
   - 该成员表示愿意在过程中协助解决任何问题。
- **ROCm 测试关注性能提升**：一名成员目前正在调查 **ROCm** 上被跳过的测试和损坏的示例。
   - 他们将同步关注 **GEMM** 的性能提升。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1458306187903893598)** (14 messages🔥): 

> `Discord Timeout Issues, GitHub Actions Errors, Warp Specialization Optimization, CuTeDSL, Slow Runners` 


- ****Discord Profile 运行超时，归咎于 GitHub Actions****：用户报告通过 Discord 运行 profile 时出现**超时 (timeouts)**，而同样的代码在 CLI 上运行正常，这指向了可能的 [GitHub Actions](https://github.com/gpu-mode/kernelbot/commit/e02d5004044f07290bd6f2d8ecca3b5d38f754e9) 问题。
   - 该问题最初被报告为 *"Server processing error: An unexpected error occurred: RuntimeError"*，但随后不久便报告已修复。
- ****CuTeDSL 获得 Warp Specialization 支持！****：一位用户分享了一篇关于 [CuTeDSL 中的 Warp Specialisation](https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/) 的博客文章，该文章将 **GEMM mainloop** 拆分为 **TMA**（将 tile 拷贝到 SMEM）和 **MMA**（tile 乘法）。
   - 该优化通过 **CuTeDSL 的流水线抽象 (pipelining abstraction)** 便捷地实现，将普通非持久化的 **Blackwell mainloops** 转换为 warp-specialized 版本。
- ****再次发现慢运行器 (Slow Runners)****：一位用户注意到 ID 为 **297869** 的 **slow runner**。
   - 未提供更多上下文。


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1458328734724788376)** (1 messages): 

> `GPU roles, CUDA, GPU performance, Real-time rendering, H-1B sponsorship` 


- **求职者目标为 2026 年的 GPU 职位**：一名成员正在寻找从 2026 年 2 月开始的美国全职 **GPU/图形职位**，重点关注 **CUDA、GPU 性能或实时渲染**。
   - 该成员拥有宾夕法尼亚大学 (UPenn) 计算机图形与游戏技术专业的 **MSE** 学位，具备 **C++、CUDA、Vulkan、OpenGL、WebGPU、GLSL/WGSL、Unity/Unreal、Nsight 和 RenderDoc** 的经验，并提供了其 [LinkedIn](https://linkedin.com/in/xinran-tao)、[作品集](https://xinrantao.com)和 [GitHub](https://github.com/theBoilingPoint) 链接。
- **OPT 状态与 H-1B 赞助需求**：该求职者目前持有 **OPT**（带有 STEM 延期），需要 **H-1B 赞助**。
   - 他们对 **CUDA/GPU 计算与性能工程**、渲染、引擎级图形工作以及 **GPU 密集型系统**（游戏、AR/VR、视觉计算）持开放态度。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1458562599527649595)** (25 messages🔥): 

> `Mojo Training Library, MAX for Backprop, Struct Design, Data Collection, IO Primitives` 


- **Mojo 缺失训练库支持**：Mojo 目前缺乏专门的训练库，要求用户利用 **MAX** 并手动实现**反向传播 (backpropagation)**。
   - 一名成员提到，如果走这条路，将需要 *自行编写反向传播代码*。
- **I/O 在数据格式化方面尚处于原始阶段**：由于 Mojo 的 **I/O** 能力尚不完善，用户可能需要为他们的训练数据集实现自定义数据格式。
   - 一名成员表示：*由于 IO 仍然有些原始，Mojo 还没有实现很多数据格式。*
- **不要使用旧文档！**：警告用户避开过时的文档，应参考 [Modular 官方仓库](https://github.com/modular/modular)和 [Mojo 文档](https://docs.modular.com/mojo/manual/)。
   - 一位用户链接了一个过时的 **GitHub 仓库**，另一位用户对此回复道：*那不是主仓库，已经过时 2 年了。*
- **新手注意：Mojo 可能会让你碰壁！**：由于 Mojo 仍在开发中且功能可能不稳定，建议编程新手在深入学习 Mojo 之前先积累 **C** 或 **Python** 等语言的经验。
   - 一名成员表示 *Mojo 会经常破坏很多东西，而且目前所有的文档都假设你具备 Python + C++ 或 Rust 的某种组合知识*。
- **周末 Tiny LLM 项目预告**：一名成员计划在周末开展一个构建 tiny **LLM** 的项目。
   - 他们希望 *生活不会再给他们的计划增添任何阻碍*。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1458190276001796239)** (54 messages🔥): 

> `NuMojo v0.8.0 更新, Mojo 中的错误处理, 匿名 Sum Types, Dict 迭代限制` 


- **NuMojo v0.8.0 发布！**: [NuMojo v0.8.0 更新](https://forum.modular.com/t/numojo-v0-8-0-update-is-here/2579) 现已发布，包含多项改进和功能。
   - 鼓励社区探索新的展示案例并提供反馈。
- **匿名 Sum Types 引发讨论**: Mojo 正在探索“匿名” Sum Types（例如 `T1|T2`）的可能性，并对生成函数提供了[初步支持](https://github.com/google-research/dex-lang/issues/1151)。
   - 然而，人们对规范化和易用性提出了担忧，特别是在泛型和条件一致性（conditional conformances）方面，认为 *“只有在引入某种规范化（重新关联、排序、去重、扁平化）之后，它才变得可用”*。
- **错误处理易用性审查中**: 讨论涉及使用带有代码的单一类型来统一错误类型（灵感来自 `errno`），并改进 `catch` 处理异构错误类型的易用性，可能会使用类似于 Zig 的 Error Unions 或 Rust 风格的 Sum Types。
   - 一位参与者指出：*“我同意 Mojo 目前可以从转换 errno 中获益，稍后我们可以讨论更好的错误处理，如 Error Unions（Zig 风格）/ Sum Types（Rust 风格，非 Result）等。”*
- **Mojo 用户寻求 Dict 迭代器指导**: 一位 Mojo 用户在开发原生 TOML 解析器时，正在寻求迭代 Dict 条目以构建嵌套结构的正确模式，因为 *'DictEntry 不支持索引（subscriptable）'*。
   - 该用户尝试创建嵌套的 Dict 结构，并报告了 `.items()` 和 `.keys()` 迭代器的问题，正在寻求最佳实践方面的帮助。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458645381091954738)** (1 messages): 

> `TEI vs MAX, Embeddings 生成, MiniLM 性能` 


- **TEI 在 Embeddings 性能上超越 MAX**: 一位成员正将 Embeddings 任务从 [TEI](https://github.com/huggingface/text-embeddings-inference) 切换到 **MAX**，发现 *sentence-transformers/all-MiniLM-L6-v2* 的性能显著下降。
   - 具体而言，**MAX** 的吞吐量为 **727.1 embeddings/sec**，P95 延迟为 **28375.1 ms**，而 **TEI** 达到 **8000 embeddings/sec**。
- **自定义架构实现影响了性能？**: 该成员将 *sentence-transformers/all-MiniLM-L6-v2* 实现为自定义架构，想知道性能不佳是因为实现不理想，还是因为 **MAX Serve** 针对 **LLM inference** 而非 **embeddings** 进行了优化。
   - 他们分享了其 `max serve` 命令，询问是否存在非最优参数：`--model sentence-transformers/all-MiniLM-L6-v2 --custom-architectures minilm --task embeddings_generation --pipeline-role prefillonly --max-batch-size 1024 --max-ce-batch-size 1024 --max-num-steps 1 --device-memory-utilization 0.95 --no-enable-chunked-prefill --no-enable-prefix-caching --port 8123`


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458200333846315225)** (24 messages🔥): 

> `Constrained Generation Annoyances, Model Reasoning vs. Token Processing, PDF prevalence on the web, vLLM constrained decoding, Cheap model training options` 


- **约束生成（Constrained Generation）的实现困扰**：一位成员对实现允许模型在约束前进行“思考”的约束生成的难度表示沮丧，并指出这需要对 **vLLM internals** 进行调整。
   - 另一位成员链接了关于 [structured outputs 的 vLLM 文档](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html)，认为这应该不需要太多的内部调整，但第一位成员寻找的是约束的条件触发（例如在 `</think>` token 之后触发）。
- **推理与 Token 处理能力**：一位成员认为拥有足够参数量（**235B+**）的模型在处理 token 时应具备内在的推理能力，从而消除对显式推理步骤的需求。
   - 相比之下，讨论的发起者详细说明了一种将推理与 prompt 交织的方法，例如 `{chunk}{prompt==describe how to grade chunk}` 然后缓存 KV 对（key value pairs）。
- **PDF 普及率统计数据引发辩论**：一位用户引用统计数据指出 **PDF 仅占 Common Crawl 的 0.6%**，而另一位用户质疑计算中是否考虑了被截断的 PDF。
   - 讨论明确了 **0.6%** 的数字代表的是文件数量，而非文件大小。
- **低成本训练方案**：一位成员询问关于在 **100GB 数据集上训练 1 亿参数模型** 的经济型方案。
   - 建议包括使用 **Vast.ai** 和 **RunPod**，并推荐使用像 **4090s** 或 **5090s** 这样的消费级 GPU，因为该设置不受限于通信（communications-bound）。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458190455278800977)** (18 messages🔥): 

> `Compute Credits, Modal, Lium, LFM 2.5, RL Experiments` 


- **Kaggle 和 Colab 虚拟机提供算力额度 (Compute Credits)**：一位成员建议对于小模型由于算力限制可以使用 Kaggle/Colab 虚拟机，并指出 Modal 和 Lium 等供应商提供约 **$500** 的额度，足以支持约 **100M** 次运行。
   - 他们还指出 Kaggle 非常适合某些特定用例。
- **LFM 2.5 的“计算量减少声称”面临批评**：一位成员质疑 **LFM 2.5** 的计算量减少声称，认为它实际上增加了计算量，并引用了 [liquid.ai 的博客文章](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)。
   - 另一位成员想知道利用稀疏性的自定义 CUDA kernel 是否能起到减少计算量的作用。
- **RL 实验：谨慎行事**：一位成员建议专注于基础模型训练等更简单的子领域，而不是 RL，因为 RL 实验由于基础知识问题非常容易出错。
   - 然而，他们随后改变了态度，并声明该文本是 AI 生成的，不表示欢迎。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1458354794187198627)** (1 messages): 

> `lm-evaluation-harness, LLM-as-a-judge, open-sourced LLMs` 


- **"lm-evaluation-harness" 是否寻求 Judge 评价支持？**：一位成员询问 ["lm-evaluation-harness"](https://github.com/EleutherAI/lm-evaluation-harness) 是否支持使用 **LLM-as-a-judge** 进行评估（例如使用开源 LLM）。
- **该话题暂无进一步讨论**：目前没有关于此话题的更多信息。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1458311524731981946)** (1 messages): 

> `Sora ripoffs, Ghibli style segment` 


- **Sora 被指控对现有视频进行“换皮”**：一位成员表示，*Sora 中许多看起来令人印象深刻的东西只是对现有视频进行了“换皮（reskinned）”的直接抄袭。*
- **Sora 的吉卜力风格片段让人联想到《火垂るの墓》**：一位成员认为 **Sora** 中备受关注的吉卜力风格片段让他们想起了《火垂るの墓》（Grave of the Fireflies）中的特定场景。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1458191786559410206)** (3 条消息): 

> `GPT-NeoX, LoRA, QLoRA` 


- **Attention 归一化差异**：一名成员注意到默认的 Attention 归一化行为发生了变化，旧的行为是在所有 Heads 之间统一进行归一化，而新的行为仅在每个 Head 内部进行归一化。
   - 他们为没有早点阅读回复表示抱歉，但似乎已经解决了该问题。
- **LoRA/QLoRA 微调支持**：一名成员询问该仓库是否支持使用 **LoRA/QLoRA** 对 **GPT-NeoX** 进行微调。
   - 他们注意到存在一些脚本 ([configs/finetuning_configs/6-9B.yml](https://github.com/EleutherAI/gpt-neox/blob/main/configs/finetuning_configs/6-9B.yml))，但指出这些脚本是用于全参数微调的。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458192090961023108)** (25 条消息🔥): 

> `Manus 客户支持, 额度政策, 提升 Manus 效率, AI 安全工具 (HexStrike MCP), 开源计划` 


- **客户就 Prompt 能力展开辩论**：一名客户对 Manus 表示不满，理由是问题未解决且决定不补偿损失的额度，尽管有服务条款约束，他们仍已转向其他替代平台。
   - 另一位用户建议该客户继续尝试不同的模型，并将 Manus 视为*顶尖工具 (the shit)*，对此支持团队回复称他们正在调查具体原因，这可能需要一些时间。
- **明确额度重置政策**：支持团队的一名成员澄清说，每月订阅额度需要在订阅期内使用，例如 [$20 的 Pro 会员每月提供 4000 额度](https://manus.im/help/credits)，这些额度需在下个月重置前使用。
   - 支持团队提出进一步核实用户的具体订阅状态和账户详情，*例如，如果你在 1 月 1 日购买了 $20 的 Pro 会员，你将收到 4000 个每月额度，这些额度需要在 2 月 1 日之前使用*。
- **心理学家建议高效使用 Manus**：一位心理学家建议将讨论重点放在 Manus 使用过程中出错的地方，并引用了一篇 [代表作 (magnum opus)](https://fwoxkyoz.manus.space/) 来帮助用户提高效率，因为这是打造能根据反馈进行改进的 AI 工具的关键。
   - 该心理学家提到通过知识对 Manus 进行迭代教学，使 Manus 能够记住任务并在使用额度前请求确认。
- **解释 HexStrike MCP 连接问题**：一位用户描述了托管在本地虚拟机上的 **AI 安全工具 (HexStrike MCP)** 与 **AI 客户端 (Manus)** 之间的连接问题，解释说 AI 客户端无法正确解析主机名。
   - 用户临时使用 **ngrok** 通过公网 HTTPS 端点暴露本地服务，试图了解将 **MCP server** 迁移到具有公网 IPv4 地址的 **VPS** 是否能解决连接问题，并允许正常的 OAuth 流程和 SSE 连接。
- **咨询开源计划**：一名成员询问是否有计划将 **Manus 的旧部分开源**并为新项目做贡献。
   - 另一名成员建议将问题发布在 **Manus Api 频道**供 Ivan 审阅。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1458412867328938058)** (20 条消息🔥): 

> `Emily 灌水账号, Kimi 和 Deepseek 与阿里巴巴的对比, Kimi K2 vs 国产模型, Qwen 性能` 


- **Emily 是一个灌水账号**：一名成员表示 **Emily** 大多是一个*灌水账号 (slop account)*。
   - 另一名成员询问在春节前期待发布是否公平。
- **阿里巴巴 QW 等于 AGI？**：一名成员评论说，至少在英语方面，与 **DeepSeek** 和 **Kimi** 相比，**阿里巴巴的 QW** 简直可以等同于 **AGI**。
   - 他们不确定 **DeepSeek** 和 **Kimi** 在英语和中文方面的性能是否存在巨大差异。
- **Kimi K2 在双语方面表现出色**：一名成员声称 **Kimi K2** 在两种语言中都表现出色，并且在 **EQ bench** [https://eqbench.com/](https://eqbench.com/) 中获得了最高分。
   - 他们解释说，对于他们而言，这些模型的主要区别在于，与其他中国模型相比，**Kimi K2** 在创意写作和整体对话方面的表现明显更好。
- **Qwen 性能并不理想**：一名成员分享说，他们使用 **Qwen3** 模型变体的体验并不太好。
   - 他们解释说这些模型在大多数任务中表现*尚可*，但有时会在简单的任务上失败，并且在对话或创意写作方面表现不佳。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458300365337399306)** (8 messages🔥): 

> `tinygrad vs competitors, tinygrad internal problems, linearizer bounty` 


- **将 tinygrad 与竞争对手进行比较**：一名新成员询问了是什么让 **tinygrad** 优于其竞争对手，以及开发团队正在尝试解决的主要内部问题。
   - 另一名成员建议查看 [tinygrad's thesis](https://geohot.github.io//blog/jekyll/update/2025/07/06/can-tinygrad-win.html) 以及每周会议中的开源讨论以寻找答案。
- **未领取的 Linearizer 悬赏**：一名成员询问关于“用 Linearizer 替换 scheduler 并保持 GPU 速度”的悬赏是否仍未被领取，尽管目前有一个可能已经准备就绪的 [PR](https://github.com/tinygrad/tinygrad/pull/13780)。
   - 另一人回答说，先提交一个可工作的 PR 可能会获得悬赏，且 **George Hotz** 可能会拆分奖金以鼓励进度。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1458632240815669259)** (2 messages): 

> `VFIO=1, AMD Radeon RX 9070XT, tinygrad error` 


- **VFIO=1 在 AMD Radeon RX 9070XT 上触发 TypeError**：一名用户报告在带有 **AMD Radeon RX 9070XT** 的 Linux 笔记本电脑上使用 `VFIO=1` 运行 `examples.benchmark_onnx` 时出现 **TypeError**，而不设置 `VFIO=1` 时则运行正常。
   - 错误源于 `tinygrad/runtime/support/c.py` 文件中进行 `ioctl` 调用时 `NoneType` 对象不可调用，详见[提供的日志](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=696058bf&is=695f073f&hm=156caa091597e59aaaf338b4e228a70a3d523b440e6d5ce6fb1e909cad59e138&)。
- **VFIO=0 工作正常**：用户进一步澄清，当 VFIO=0 时，`examples.benchmark_onnx` 按预期运行，没有错误。
   - 这表明问题是在 `VFIO=1` 时触发的。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1458207611739574283)** (8 messages🔥): 

> `mTLS Implementations in MCP, MCP Instructions Documentation` 


- ****mTLS 魔法**：寻求 MCP 互操作性见解！**：一名成员正在探索 **mTLS** 实现，以增强 **MCP** 在企业环境中的互操作性，并正在寻找贡献讨论的最佳渠道。
   - 另一名成员建议从 <#1360835991749001368> 频道开始，并指出身份验证工作组可以提供关于正在进行的各种相关项目的见解。
- ****文档荒**：MCP 指令文档匮乏！**：一名成员询问有关 **MCP instructions** 的文档，注意到其缺失。
   - 另一名成员指出它是 [server's initialization response](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization) 的一部分，并链接了一篇 [博客文章](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/) 作为目前最接近的可用资源，同时还创建了一个 [issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2060) 以将其中一部分内容纳入官方文档。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1458326753717125355)** (2 messages): 

> `brave comments, ironic commentary` 


- **讽刺性评论引发反讽回应**：有人在一条消息后评论道 *someone had to say it*（总得有人把这话说出来）。
   - 另一人回复 *so brave*（真勇敢）。
- **聊天中的讽刺**：两名成员交换了讽刺性评论。
   - 这可能表明对某个话题有强烈的情绪。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ash_blanc: https://www.alphaxiv.org/abs/2601.01569
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1458512384133173350)** (2 messages): 

> `DAO projects, AI systems, Model Pipelines, Productionizing LLM features, Inference Cost Strategy` 


- **开发者寻求新的 DApp 冒险**：一名拥有 **DAO**、**marketplace**（市场）和 **DApp** 项目经验的开发者正寻求参与具有扎实愿景和长期关注点的新项目。
   - 他们很乐意聊天、贡献或只是交流想法，并带来了在治理、工具和可用性项目方面的实战经验。
- **AI 工程师准备好理清 Model Pipelines**：一名 AI 工程师强调了他们构建真实 AI 系统的经验，从 **training** 和 **fine-tuning models** 到在大规模环境下缝合 **retrieval**（检索）、**agents** 和 **infra**（基础设施）。
   - 他们更倾向于亲自动手并交付成果，在理清 model pipeline、将 LLM 功能生产化（productionizing）或制定推理成本策略（inference cost strategy）方面提供帮助。


  

---


---


---