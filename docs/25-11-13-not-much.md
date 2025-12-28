---
companies:
- openai
- google-deepmind
- github
- microsoft
- cursor_ai
- perplexity-ai
- weaviate
- llamaindex
date: '2025-11-13T05:44:39.731046Z'
description: '**OpenAI** 发布了 **GPT-5.1** 系列模型，包括 **5.1-Codex** 和 **5.1-Codex-Mini**。新模型提升了可控性、响应速度，并引入了
  `apply_patch` 和 shell 命令执行等新工具。定价与 5.0 版本保持一致。**GitHub Copilot**、**VS Code**、**Cursor**
  和 **Perplexity** 已率先集成并采用 GPT-5.1 模型。


  **Google DeepMind** 发布了 **SIMA 2**，这是一款由 **Gemini** 驱动的智能体，具备遵循语言指令、规划以及无需人类反馈的自我改进能力，主要面向机器人应用。


  此外，关于上下文工程（context engineering）和智能体工具使用模式的新研究也已发布，**Weaviate** 和 **LlamaIndex**
  分别在数据库查询规划和图表解析方面做出了贡献。**GPT-5.1-Instant** 则重点突出了“自适应推理”和智能体编程能力的提升。'
id: MjAyNS0x
models:
- gpt-5.1
- gpt-5.1-codex
- gpt-5.1-codex-mini
- sima-2
- gemini
people:
- sama
- allisontam_
- cline
- cognition
- demishassabis
- omarsar0
- helloiamleonie
title: GPT 5.1 和 SIMA 2 的小幅更新。
topics:
- adaptive-reasoning
- agentic-coding
- tool-use
- context-engineering
- memory-architecture
- self-improvement
- retrieval-augmentation
- database-query-planning
- chart-parsing
- robotics
---

持续改进的一天。

> 2025/11/12-2025/11/13 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 账号和 23 个 Discord（201 个频道和 6666 条消息）。预计节省阅读时间（以 200wpm 计算）：523 分钟。我们的新网站现已上线，包含完整的元数据搜索和美观的 vibe coded 风格展示的所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

我们去年介绍了 [SIMA](https://news.smol.ai/issues/24-03-13-ainews-deepmind-sima-one-ai-9-games-600-tasks-visionlanguage-only)，今天 [SIMA 2 发布了](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)，但由于缺乏技术报告且行业兴奋度一般，我们没有将其作为头条新闻。

GPT 5.1 已在 API 中发布并附带评测，但由于我们昨天已经将其作为头条新闻，今天不再重复。

---

# AI Twitter 回顾

**OpenAI 的 GPT‑5.1 推广及生态系统采纳**

- **API 中的 GPT‑5.1 系列 + 新的 Agent 工具**：OpenAI 发布了 GPT‑5.1（以及 5.1‑Codex, 5.1‑Codex‑Mini），具有更好的可控性、更快的响应速度和更强的编码能力。新的内置工具包括用于可靠自由格式代码编辑的 `apply_patch` 和用于受控命令执行的 `shell` 工具；prompt caching 延长至 24 小时，以降低重复提示词的成本/延迟。根据 [@sama](https://twitter.com/sama/status/1989048466967032153) 的说法，价格与 5.0 保持一致。查看来自 OpenAI DevRel [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989042617750024403) 的发布详情和问答，工具介绍 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989042624574198021)，cookbook 链接 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989043495269724617)，以及早期客户评价 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989064999680242007)。OpenAI 还强调了更新的安全评估和轻微的基准测试退化（AIME, Taubench），以应对 [@swyx](https://twitter.com/swyx/status/1989047883639980141) 提到的“刷榜（benchmark‑maxxing）”担忧。
- **自适应推理和 Agent 编码**：5.1‑Instant 引入了“自适应推理”（在更困难的任务上消耗更多 token），据 OpenAI 后训练团队 [@allisontam_](https://twitter.com/allisontam_/status/1989138927970848936) 称。Agent 工具链和脚手架已经在围绕 5.1 进行重新调整：Cline 详细介绍了以执行为中心的提示词、更严格的计划/行动转换，以及针对大型仓库的两阶段深度规划（[线程](https://twitter.com/cline/status/1989056367030829458)）；Cognition 在 Windsurf 中将 5.1 设为默认模型，以实现更流畅、更少“过度思考”的编码（[公告](https://twitter.com/windsurf/status/1989069991770214580), [@cognition](https://twitter.com/cognition/status/1989081722353529178)）。
- **快速集成**：GitHub Copilot 在公开预览版中推出了 GPT‑5.1, 5.1‑Codex, 5.1‑Codex‑Mini ([@github](https://twitter.com/github/status/1989044218451394968))，VS Code 展示了模型在编辑器体验中的落地 ([@code](https://twitter.com/code/status/1989044946058326370))。Cursor 添加了这三个模型并更新了路由 ([@cursor_ai](https://twitter.com/cursor_ai/status/1989045849003835460))；Perplexity 为 Pro/Max 用户启用了 5.1 ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1989075483385069949))；其他工具也迅速跟进（Anycoder, Yupp, Factory） ([@_akhaliq](https://twitter.com/_akhaliq/status/1989161892880032132), [@yupp_ai](https://twitter.com/yupp_ai/status/1989080371775041942), [@FactoryAI](https://twitter.com/FactoryAI/status/1989052558279864595))。

**Agent、具身智能和记忆架构**

- **SIMA 2 (DeepMind)**: Google DeepMind 发布了 SIMA 2，这是一个由 Gemini 驱动的 Agent，能够遵循语言指令，进行规划，通过标准键盘/鼠标执行操作，泛化到未见过的游戏，并通过使用 Gemini 效用模型进行试错（trial-and-error）来实现自我改进——无需人类反馈。它还能在由 Genie 3 生成的世界中导航 ([overview](https://twitter.com/GoogleDeepMind/status/1988986218722291877), [Genie 3 demo](https://twitter.com/GoogleDeepMind/status/1989024090414309622), [@demishassabis](https://twitter.com/demishassabis/status/1989096784870928721))。Google 将其定位为迈向机器人应用的一步 ([post](https://twitter.com/GoogleDeepMind/status/1988987865401798898))。
- **上下文与工具使用模式 (Context and tool use patterns)**: Google 发布了一份关于上下文工程（context engineering）的从业者白皮书——涵盖会话、记忆以及如何构建检索架构以提高 Agent 可靠性 ([@omarsar0](https://twitter.com/omarsar0/status/1989081828678893837))。Weaviate 的 “Query Agent” 展示了跨集合的数据库自然语言到查询（NL-to-query）规划，支持过滤、路由、聚合和引用 ([@helloiamleonie](https://twitter.com/helloiamleonie/status/1989007852502139221))。LlamaIndex 增加了 Agent 化图表解析功能，通过追踪折线图中的轮廓来提取数值序列 ([@llama_index](https://twitter.com/llama_index/status/1989060127551549854))。
- **Agent 基础设施强化**: LangChain 为 DeepAgents 引入了 Sandboxes，以便在远程沙箱（Runloop, daytona, Modal）中安全执行任意代码/bash，将规划与执行环境分离 ([announcement](https://twitter.com/LangChainAI/status/1989006586388574397))。LangSmith Essentials 课程专注于多轮/工具调用 Agent 的持续测试和可观测性 ([@LangChainAI](https://twitter.com/LangChainAI/status/1989025161488793743))。Qwen 发布了具有“高级模式”的 DeepResearch 2511，支持文件上传、更深度的搜索以及可配置的报告格式/引用 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1989026687611461705))。社区演示如 “Kimi Deep Researcher” 显示每个会话可进行数百次工具调用 ([@omarsar0](https://twitter.com/omarsar0/status/1988974710592516454))。

**可解释性与训练科学 (Interpretability and training science)**

- **稀疏电路作为训练目标**: OpenAI 提议训练具有极稀疏权重的微型 LM，使内部机制更易于解释，从而隔离出针对字符串终止和变量跟踪等行为的电路。他们发布了代码和模型，将其定位为通往完全可解释的 GPT-3 级“模型生物（model organism）”的路径，以用于安全和理解研究 ([OpenAI](https://twitter.com/OpenAI/status/1989036214549414223), [thread](https://twitter.com/OpenAI/status/1989036218160673103), [team lead](https://twitter.com/nabla_theta/status/1989043939374924251))。
- **时间特征与 JEPA 理论**: 时间特征分析（Temporal Feature Analysis）引入了对 LLM 激活中动态特征的预测编码风格建模，解决了 SAEs 的静态特征假设问题 ([@EkdeepL](https://twitter.com/EkdeepL/status/1989009095953895756), [@GoodfireAI](https://twitter.com/GoodfireAI/status/1989010394380485083))。在视觉领域，LeCun/Balestr 的 LeJEPA 通过新的 SIGReg 目标将目标嵌入形式化为各向同性高斯（isotropic Gaussian），简化了 JEPA 训练（无需教师-学生网络/停止梯度），并在超过 10 个数据集和 60 多个架构中取得了强劲结果 ([@ylecun](https://twitter.com/ylecun/status/1988999683801510063), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1989039076302049701))。
- **训练后增量 (Post-training deltas)**: 一项对比 RL 与 SFT 的新分析显示，RL 在更新非主奇异方向的同时保留了主奇异方向，而 SFT 可能会扭曲频谱并导致过拟合——这对 PEFT 目标定位和 PiSSA 等方案具有启发意义 ([@tydsh](https://twitter.com/tydsh/status/1989049095575728156))。PEFT v0.18 发布，带来了新方法和改进 ([@BenjaminBossan](https://twitter.com/BenjaminBossan/status/1988993386729390191))。

**模型发布与多模态/视频 (Model releases and multimodal/video)**

- **Zhipu AI GLM‑4.6**: 智谱 AI 发布了 GLM‑4.6；Together AI 正在为其提供生产级负载托管，将其定位为性能接近 Claude Sonnet 4，同时使用的 Token 减少约 15% ([lab](https://twitter.com/Zai_org/status/1989005078926143810), [host](https://twitter.com/togethercompute/status/1989082601399939312))。
- **使用 DETR 进行实时检测**: RF‑DETR（DINOv2 骨干网络）通过权重共享在约 6k 个变体上运行 NAS；RF‑DETR‑N 在 COCO 数据集上达到 48.0 AP，耗时 2.3 ms，在速度快约 2 倍的情况下匹配了 YOLOv8/11‑M 的性能；分割头变体在 3.4 ms 内达到 40.3 AP mask ([@skalskip92](https://twitter.com/skalskip92/status/1989004912609411133))。
- **视频生成新晋选手**: Vidu Q2 Turbo/Pro 在 Video Arena 首次亮相，在图生视频（Image‑to‑Video）领域排名第 6/7 位，具有精确的情绪和镜头控制；API 价格为每分钟 1080p 视频 4–6.10 美元 ([@arena](https://twitter.com/arena/status/1989056583872180298))。NVIDIA 推出了 TiDAR（“Think in Diffusion, Talk in Autoregression”），这是一种混合 Diffusion/AR 框架 ([@_akhaliq](https://twitter.com/_akhaliq/status/1988963077690438097))。
- **开放图像工作**: Photoroom 在 HF 上开源了其第二个从零开始训练的文本生成图像模型，并在 HF 上提供了权重和训练过程 ([@matthieurouif](https://twitter.com/matthieurouif/status/1988981733866271223))。

**基础设施、平台与性能**

- **Hugging Face x Google Cloud**: 广泛的合作伙伴关系，旨在加速 GCP 上的开源模型开发：Vertex AI/Cloud Run/GKE 上的 HF DLCs、原生 TPU 支持、GCP 上的 Inference Endpoints、通过 Google Threat Intelligence/Mandiant 提供的安全性，以及一个新的 GCP 缓存网关以加速模型/数据集 IO——这反映了每天超过 1,500 TB 的流量，且每年的云支出可能已超过 10 亿美元 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1989000335247983049), [@alvarobartt](https://twitter.com/alvarobartt/status/1988970441357094984))。Google 还对 Gemini CLI 进行了重大的 UX 升级 ([@googledevs](https://twitter.com/googledevs/status/1989119863961337889))。
- **推理速度**: Baseten 报告称，使用 NVIDIA Dynamo 进行多节点推理编排，长上下文代码生成的处理速度提升了 2 倍，吞吐量提升了 1.6 倍 ([@basetenco](https://twitter.com/basetenco/status/1989058852789317717))；Modal 详细介绍了 SGLang 中推测解码（speculative decoding）速度提升了 12% ([@akshat_b](https://twitter.com/akshat_b/status/1989019570783629366))。SkyPilot v0.10.5 提高了托管作业效率（18 倍），扩展了其 API 服务器，并扩大了 Python SDK/管理策略范围 ([@skypilot_org](https://twitter.com/skypilot_org/status/1989083081953931284))。
- **开发环境融合**: VS Code 添加了原生自动补全和易用性改进；至关重要的是，Google Colab 运行时现在可以支持 VS Code notebook，从而在编辑器内直接使用 GPU/TPU 计算 ([@googledevs](https://twitter.com/googledevs/status/1989033099737407820))。

**安全、评估与治理**

- **AI 引导的间谍活动被挫败**: Anthropic 表示，它发现并挫败了一场大规模、极少人为监督的网络间谍活动，并将其归因于一个受中国政府支持的组织——这可能是首例有记录的此类规模的 AI 执行攻击，目标涵盖科技、金融、化工和政府部门 ([披露](https://twitter.com/AnthropicAI/status/1989033793190277618), [分析](https://twitter.com/AnthropicAI/status/1989033795341648052))。这一事件强调了建立 AI 感知网络防御的必要性。
- **政策与评估**: Anthropic 开源了一项政治偏见评估，并讨论了政治话语中理想的模型行为 ([公告](https://twitter.com/AnthropicAI/status/1989076472208978127))。联合国科学咨询委员会（UN SAB）与 Yoshua Bengio 合作的视频涵盖了通过算力追踪和防篡改芯片进行前沿验证 ([@ScienceBoard_UN](https://twitter.com/ScienceBoard_UN/status/1988971216951210467))。Kagi 推出了“SlopStop”，用于社区驱动的搜索中 AI 垃圾内容（AI‑slop）检测 ([@KagiHQ](https://twitter.com/KagiHQ/status/1989050447844270340))。
- **市场现实检查**: Andrew Ng 警告不要陷入 AI 炒作瘫痪——LLM 依然强大但具有专业性；应用定制至关重要，而 AGI 级别的通用性尚且遥远 ([推文串](https://twitter.com/AndrewYNg/status/1989003741316673714))。与此同时，Cursor 宣布了 23 亿美元的 D 轮融资，并声称 ARR 超过 10 亿美元，断言 Agent PMF 和模型所有权是其战略护城河 ([@cursor_ai](https://twitter.com/cursor_ai/status/1988971258449682608))。

**热门推文（按互动量排序）**

- Anthropic 披露其瓦解了一场由 AI 主导的间谍活动：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1989033793190277618) 和 [分析](https://twitter.com/AnthropicAI/status/1989033795341648052)
- Karpathy 谈论自动驾驶对城市的变革性影响：[@karpathy](https://twitter.com/karpathy/status/1989078861800411219)
- OpenAI 可解释性（稀疏电路）：[@OpenAI](https://twitter.com/OpenAI/status/1989036214549414223)
- OpenAI GPT‑5.1 API/定价/Prompt Cache 公告：[@sama](https://twitter.com/sama/status/1989048466967032153)
- VS Code Notebooks 中的 Google Colab 运行时：[@googledevs](https://twitter.com/googledevs/status/1989033099737407820)
- Google DeepMind 的 SIMA 2 Agent 和 Genie 3 世界：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1988986218722291877)
- Cursor 获得 23 亿美元融资并达成 10 亿美元 ARR 里程碑：[@cursor_ai](https://twitter.com/cursor_ai/status/1988971258449682608)

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. Jan-v2-VL 模型发布与基准测试

- [**Jan-v2-VL：用于长程任务的 8B 模型，将 Qwen3-VL-8B 的 Agent 能力提升了近 10 倍**](https://www.reddit.com/r/LocalLLaMA/comments/1ovxksu/janv2vl_8b_model_for_longhorizon_tasks_improving/) (热度: 754): **Jan-v2-VL 是一款 8B 视觉语言模型，专为长程（long-horizon）、多步任务设计，显著增强了基础模型 Qwen3-VL-8B-Thinking 的能力。它在 Long-Horizon Execution 基准测试中达到了** `49 steps`**，而基础模型仅为** `5 steps`**，其他同类模型则为** `1-2 steps`**。该模型提供三种变体：low、medium 和 high，分别针对效率和推理深度的不同平衡进行了优化。它可以使用 vLLM 或 llama.cpp 运行，推荐参数包括** `temperature: 1.0`**、** `top_p: 0.95` **以及** `presence_penalty: 1.5`**。该模型可在 [Hugging Face](https://huggingface.co/collections/janhq/jan-v2-vl) 和 [Jan GitHub](https://github.com/janhq/jan) 上获取。** 有评论询问为什么选择 Reasoning 变体作为基础模型而非 Instruct 变体，这表明用户对针对特定任务的不同模型配置可能感兴趣。
    - Delicious_Focus3465 分享了 Long Horizon 基准测试的详细结果，强调 Jan-v2-VL 模型在 Agent 能力方面比 Qwen3-VL-8B 有了显著提升，性能提高了近十倍。这表明在处理长程任务方面取得了实质性进展，这对于复杂的决策过程至关重要。
    - MaxKruse96 询问了选择 “Reasoning” 变体而非 “Instruct” 变体作为基础模型的原因。这一选择可能意味着专注于增强模型的逻辑推理能力，这对于需要长期深度理解和决策的任务可能更有利。
    - maglat 询问了是否提供类似于 Open WebUI 的 Jan 服务器变体，表达了对一种允许从任何浏览器访问运行在本地 LLM 设备上的 Jan 实例的解决方案的需求。这表明对能够与现有基础设施集成的更灵活部署选项的需求。

### 2. 在消费级硬件上运行大模型

- [**在配备 128 GB RAM + 24 GB VRAM 的 PC 上运行 1 万亿参数模型**](https://www.reddit.com/r/LocalLLaMA/comments/1ow0jj0/running_a_1_trillion_parameter_model_on_a_pc_with/) (热度: 356): **一位用户成功在消费级 PC 上使用 llama.cpp 运行了拥有** `1 万亿参数` **的 Kimi K2 Thinking 模型。硬件配置包括 Intel i9-13900KS CPU、** `128 GB DDR5 RAM` **以及一块拥有** `24 GB VRAM` **的 RTX 4090 GPU。该模型使用了来自 [Hugging Face](https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF) 的 Unsloth UD-Q3_K_XL 进行量化，实现了** `0.42 tokens/sec` **的生成速度。该用户指出，llama.cpp 中的内存映射 (mmap) 允许处理大于可用 RAM 的模型文件，而低于** `~4 bits` **的量化会显著降低模型质量。使用的命令包含了** `-no-warmup` **以防止启动崩溃，使用的 llama.cpp 版本为** `b6963`**。** 一位评论者指出，用于基准测试的短提示词会使结果失效，建议使用更长的提示词和响应以获得准确的性能指标。另一位评论者建议不要在消费级硬件上运行超过 `120b` 参数的模型，并强调了激活参数和稠密参数的限制。第三位评论者对基准测试表示赞赏，并分享了他们对 **gpt-oss-120b** 模型的偏好（因其速度和平衡性），同时在运行更大模型时更倾向于 **Kimi-k2** 和 **minimax m2**。
    - DataGOGO 强调了在使用 `llama.cpp` 进行准确基准测试时，使用足够长的提示词和响应的重要性。他们建议在提示词和响应中至少使用几百个 token，并推荐使用 `1000t` 提示词和 `200t` 响应的设置进行快速基准测试。这可以确保性能计数器可靠，并能分别记录提示词处理和生成速度。
    - GreenTreeAndBlueSky 提供了关于在 PC 上运行大模型的模型大小限制指南。他们建议总参数不超过 `120b`，激活参数不超过 `12b`，如果是稠密模型则不超过 `32b`。这些约束可能基于硬件限制以及平衡性能与资源可用性的需求。
    - lumos675 提到了存储介质对性能的影响，建议从 NVMe 存储运行模型可以达到约 `4 到 5 tokens per second (tps)`。这意味着存储速度是模型性能的关键因素，尤其是在处理大参数模型时。

### 3. IBM 在 AI 领域的专利争议

- [**IBM 的 AI 研究人员通过将其重新包装为 AI 可解释性，为一项拥有 200 年历史的数学技术申请了专利**](https://www.reddit.com/r/LocalLLaMA/comments/1ow6a9i/ibms_ai_researchers_patented_a_200_yr_old_math/) (热度: 554): **IBM AI 研究人员提交了一项专利申请，旨在将** `Continued Fraction` **类实现为 PyTorch 中的线性层，这涉及在计算图上调用** `backward()`**。此举引发了担忧，因为它可能影响使用导数或带有连分数的幂级数的各个领域，如机械工程、纯数学和数值编程。该专利申请被认为具有争议，因为它将一项拥有** `200年历史` **的数学技术重新包装为 AI 可解释性，引发了关于该发明的创新性和显而易见性的辩论。[点击此处阅读更多](https://leetarxiv.substack.com/p/ibm-patented-eulers-fractions)。** 热门评论对美国专利制度表示怀疑，指出这只是一项专利申请，而非已授予的专利，并强调需要第三方提交材料来质疑其创新性。还有人批评专利制度允许此类申请，这可能被视为 Patent Trolling（专利流氓），尤其会影响美国的从业者。
    - Starcast 指出了技术报道中常见的一个误解，强调讨论的对象是专利申请，而非已授予的专利。他们指出，任何人都可以向 USPTO 提交第三方材料，根据 Prior Art（现有技术）质疑该申请的创新性或显而易见性，这是专利审查过程中的关键步骤。
    - RockyCreamNHotSauce 指出了专利审查员面临的挑战，尤其是随着 AI 相关申请的涌入。他们认为抽象的数学思想（例如在 PyTorch 等框架中实现的那些）是不应获得专利的。评论认为，仅仅在代码中实现一个数学概念并不构成重大的发明步骤，这类似于在纸上写下一个数学想法。
    - Lissanro 提到该专利申请仅针对美国，暗示其影响在地理上是有限的。他们对申请可能不具创新性的想法表示担忧，认为这种行为可能被视为 Patent Trolling，即使专利未获批准，这也是有问题的。

## 技术性较低的 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 移动端推出更新

- [**非炒作贴：Gemini 3 正在向移动用户推出！**](https://www.reddit.com/r/Bard/comments/1ovvmjo/not_hype_posting_gemini_3_is_rolling_out_to/) (热度: 1133): **Gemini 3.0 目前正向 Android 和 iOS 平台的移动用户推出，特别是针对那些在 Gemini 应用中使用 canvas 功能的用户。这次推出是 Google 优先移动端部署而非 Web 集成的大战略的一部分。更新包括 Gemini 2.5 Flash 和 Pro 模型的新标签，预示着潜在的品牌重塑或更新。此外，网络日志已确认在 Gemini Enterprise 环境中发现了 Gemini 3.0，尽管这尚未公开。根据 Gemini 网站上的网络请求，还有关于同时发布 Nano Banana 2 的推测。** 一些用户已确认在 Android 上收到了推送，而另 strategy 报告称尚未看到变化。关于 Enterprise 日志中发现 Gemini 3.0 的意义存在争论，一些用户对其能否立即向公众开放持怀疑态度。
    - **Initial-Plenty2326** 强调 Gemini 3 正在 Android 和 iOS 平台上同步推出，这对于跨平台可用性具有重要意义。这表明更广泛的用户群可以同时访问新功能，从而增强了开发者的覆盖范围和反馈循环。
    - **Alex_146** 通过分享一个使用 Android 应用创建的网页链接，提供了 Gemini 3 与其前身 Gemini 2.5 的直接对比。其暗示 Gemini 3 提供了显著增强的能力，特别是在创意设计和响应速度方面，这些是 2.5 版本无法实现的。
    - **Salty_Flow7358** 推测了 Gemini 3 的发布时间线，根据 11 月 18 日针对学生的促销活动结束，暗示潜在的发布日期为 11 月 19 日。这一见解对于计划升级或利用新功能的用户可能很有价值。

### 2. 媒体中的 AI 内容

- [**他们复制了整个 ChatGPT 的回答，甚至保留了它提议让内容更美观的部分。**](https://www.reddit.com/r/OpenAI/comments/1ovuzx2/they_copied_the_whole_chatgpt_answer_and_even/) (Activity: 3526): **该图片展示了一篇报纸文章，其中无意中包含了一段逐字复制的 ChatGPT 回复，甚至包括了关于如何让文本在头版布局中更具视觉吸引力的建议。这一事件凸显了在没有适当编辑监督的情况下，将 AI 生成的内容整合到传统媒体中所面临的挑战和潜在陷阱。文章中突出显示的文本表明 AI 的输出被直接复制，反映出该出版物缺乏彻底的编辑或审核流程。** 评论者幽默地指出了编辑监督的缺失，有人认为从文本中可以明显看出用于 ChatGPT 的 Prompt。另一条评论强调了保留文案编辑以防止此类疏忽的重要性。
    - Neat-Conference-5754 提出了关于 AI 使用的一个关键视角，强调仅仅将 AI 视为一种工具可能会导致问责制的缺失，以及对需要人类判断的任务产生过度依赖。他们认为 AI 应该被视为共同创作者或助手，需要人类的监督和编辑，以确保输出的质量和准确性。这强调了将 AI 与人类专业知识相结合，而不是完全取代人类的重要性。
- [**他们逐字复制了 ChatGPT，连结尾都留着。太离谱了。**](https://www.reddit.com/r/ChatGPT/comments/1ovv0kg/they_copied_chatgpt_wordforword_and_left_the/) (Activity: 20305): **该图片展示了一个重大的编辑疏忽，一篇关于巴基斯坦汽车销售的报纸文章无意中包含了来自 ChatGPT 未经编辑的 AI 生成文本。这篇文章讨论了车辆销量的增长并提供了详细统计数据，但错误地保留了一个指示创建“头版风格”版本并配以“有力的数据和信息图布局”的部分。这表明编辑团队在出版前未能删除或修改 ChatGPT 生成的占位符文本，引发了对新闻编辑流程以及对 AI 工具依赖程度的质疑。** 评论者表达了尴尬并批评了编辑疏忽，建议校对人员应为允许 AI 生成的文本出现在最终印刷版中而承担责任。

### 3. 新 AI 模型与 Benchmark 发布

- [**Google DeepMind - SIMA 2：一个在虚拟 3D 世界中与你一起游戏、推理和学习的 Agent**](https://www.reddit.com/r/singularity/comments/1ow3g1o/google_deepmind_sima_2_an_agent_that_plays/) (热度: 1538): **Google DeepMind 推出了 SIMA 2，这是一款能够在虚拟 3D 环境中进行游戏、推理和学习的高级 AI Agent。该 Agent 展示了显著的自我改进能力，通过试错以及来自 Gemini 模型的反馈来学习复杂任务。值得注意的是，SIMA 2 可以从人工引导的学习过渡到自主游戏，在没有额外人工数据的情况下增强其在全新、未见过的游戏中的技能。这种迭代学习过程在 Genie 环境中得到了进一步加强，标志着在跨越多样化、程序生成的场景中训练通用 AI Agent 迈出了重要一步。** 一位评论者强调了 **SIMA 2** 在现实、高性价比且安全的虚拟环境中训练 Robot 的潜力，这可能会显著推进 AI 研究。另一位表达了对订阅制 AI Agent 的渴望，用于日常互动和游戏。
    - SIMA 2 的自我改进能力是一项重大进步，因为它可以从通过人类演示学习过渡到在游戏中进行自主游戏。这种能力使其能够在没有额外人工生成数据的情况下，利用自身的经验数据训练后续版本，从而在以前未见过的环境中开发技能。这种迭代式的自我改进通过使用 Genie 环境得以实现，标志着在跨越多样化的生成世界中训练通用 Agent 迈出了重要一步。
    - SIMA 2 与用于创建虚拟世界的 Genie 3 的集成，代表了向开发通用 AI Agent 的飞跃。通过使用这些工具，SIMA 2 可以递归地自我改进，一些评论者认为这是迈向技术 Singularity 的一步。这个过程涉及 SIMA 2 在新创建的环境中学习和适应，可能导致更先进的 AI 能力。
    - SIMA 2 在虚拟世界中学习和适应的潜力引发了关于其在现实场景（如人形 Robot AI）中适用性的讨论。将学习从虚拟环境泛化到现实环境的能力，可能为能够在复杂、动态设置中运行的高级 AI 系统铺平道路。这种能力被视为超越游戏和虚拟模拟的更复杂 AI 应用的前兆。
- [**GPT-5.1 确实有点东西**](https://www.reddit.com/r/ChatGPT/comments/1owdjzw/gpt51_is_definitely_something/) (热度: 1467): **这张图片是一段幽默的对话，突显了像 GPT-5.1 这样的 AI 模型在处理用户交互时的对话怪癖，特别是在它们如何处理用户互动方面。对话涉及用户讨论燕麦粉、燕麦片和杏仁奶之间微不足道的卡路里差异，而 AI 的反应显得过于戏剧化且像人一样。这反映了 AI 开发中在维持用户交互的上下文和语调方面持续存在的挑战，特别是当用户期望在类似查询中表现出一致的行为时。帖子和评论表明，虽然 AI 的回答可能很有趣，但它们也指出了 AI 模型在处理重复性任务或保持一致的对话语调方面可能需要改进的地方。** 评论者发现 AI 的回答很有趣，并将其比作人类互动，认为虽然 AI 的对话风格很有娱乐性，但对于寻求直接帮助的用户来说，可能并不总是实用。
    - Buck_Thorn 强调，GPT 模型（如 GPT-5.1）的行为不仅受版本号的影响，还受用户特定设置（如 Chat History 和个性化配置）的影响。这表明用户交互和自定义可以显著影响模型的响应，因此在评估模型性能时，考虑这些因素至关重要。

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要的摘要的总结
> 

**1. GPT-5.1 无处不在：编码、推理、发布**

- **5.1 版本席卷工具链**：OpenAI 在 [GPT‑5.1](https://openai.com/index/gpt-5-1/) 中发布了具备自适应推理和改进编程能力的 **GPT‑5.1**；OpenRouter 同步上线了 [**GPT‑5.1 Chat**](https://openrouter.ai/openai/gpt-5.1-chat)、[**GPT‑5.1‑Codex**](https://openrouter.ai/openai/gpt-5.1-codex) 以及 [**GPT‑5.1‑Codex‑Mini**](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113)；同时 **Windsurf** 开启了 7 天免费试用，并根据 [Windsurf 的公告](https://x.com/windsurf/status/1989069991770214580)将 **GPT‑5.1** 设为默认模型。
    - 工程师们报告了在 **agentic coding** 和前端工作方面的显著提升，Windsurf 声称其表现**更快**、**更具可控性**，在调节推理深度的同时减少了“过度思考”；**Cursor** 用户在 [最新的 codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) 中发现了 **GPT‑5.1‑Codex** 的身影，并实现了与 [Windsurf](https://www.windsurf.ai/) 的跨集成。
- **Polaris 退出舞台，5.1 接棒**：**OpenRouter** 弃用了 **Polaris Alpha**（一个不具备推理能力的早期 GPT‑5.1 版本），并将其替换为更快、Token 效率更高的 **GPT‑5.1** 系列，具备自适应推理和更好的编程能力，详见 [面向开发者的 GPT‑5.1](https://openai.com/index/gpt-5-1-for-developers/)；新的端点包括 [**GPT‑5.1 Chat**](https://openrouter.ai/openai/gpt-5.1-chat)、[**GPT‑5.1‑Codex**](https://openrouter.ai/openai/gpt-5.1-codex) 和 [**GPT‑5.1‑Codex‑Mini**](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113)。
    - 团队注意到 ChatGPT 中的 **Instant** 体验对应的是 **GPT‑5.1 Chat**，而重代码的工作流开始转向 **Codex** 变体，这与 OpenAI 在 [面向开发者的 GPT‑5.1](https://openai.com/index/gpt-5-1-for-developers/) 中的开发者指南一致。
- **关于 5.1 的 AMA 有问必答**：OpenAI 在 [r/OpenAI](https://redd.it/1ovkt6n/) 安排了一场 **Reddit AMA**，于太平洋时间下午 2 点解答关于 **GPT‑5.1** 和自定义功能的问题。此前，[GPT‑5.1](https://openai.com/index/gpt-5-1/) 中记录的关于**自定义指令**和叙事质量的反馈褒贬不一。
    - 开发者将 **GPT‑5.1** 与早期模型在故事讲述和格式保真度方面进行了对比，并计划在 AMA 中直接提出问题和需求，以明确路线图和微调优先级。

**2. GPU Kernels & Blackwell：从 Helion 到 NVFP4**

- **Helion 凭借便捷的 Autotune 快速推进**：**Helion** 确认了 0.2.x 的**向后兼容性**，发布了 **v0.2.2**，并根据 [Helion issue #164](https://github.com/pytorch/helion/issues/164) 为 **autotuning**（Triton 风格）添加了 `configs=`，同时其 eager-mode 解释器保持了惊人的运行速度。
    - 工程师们强调了通过 `helion_rms_norm_fwd.bind((x, w, eps))._config` 获取最优 Kernel 的方法，并将 **Helion** 快速的解释模式与 **Triton** 在开发循环中较慢的解释路径进行了对比。
- **NVFP4 GEMV 挑战赛拉开帷幕**：一场旨在为 **Blackwell** GPU 上的 **NVFP4** 优化 **GEMV** 的黑客松正式启动，提供 **Datacrunch B200** 访问权限和推荐的 **CuTeDSL** 技术栈，详见 [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/)。
    - 参赛者报告称，使用 CuTeDSL 实现了快速迭代和贴近硬件的生产力，目标是实现微秒级 Kernel，并在博客挑战简报中描述的排行榜上获得竞争性排名。
- **NCU 为云服务商评分，而非曲线**：云供应商现在根据 **NCU** (NVIDIA Compute Unified Device Architecture) 支持情况进行评分，根据 [Semianalysis: Clustermax 2.0](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard)，这提高了 GPU 可观测性和性能工具的标准。
    - 社区将 **NCU** 能力视为生产级 GPU 工作负载的必备条件，期望供应商能够大规模标准化分析（profiling）和 Kernel 遥测。

**3. 数据流水线：干净的语料库、许可证和 Tokenizer**

- **法语维基百科清洗完成并发布 JSON 格式**：一个经过清洗的**法语维基百科**转储文件已在 Hugging Face 上发布，包含超过 **270 万个 JSON 文件**，保留了模板、表格、HTML、引用、信息框和链接，项目名为 [wikipedia-fr-2.7m-clean-json](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)。
    - 贡献者们讨论了下一步将该流水线扩展到**英语维基百科**，并利用结构化的 JSON 来保留丰富的图特征，以供下游训练使用。
- **NVIDIA 许可证限制引发关注**：从业者指出了 **NVIDIA 数据集许可证**中的限制性条款——包括训练/评估/公开结果的限制以及单方面终止条款——该推文总结了相关内容：[GoodfireAI on X](https://x.com/goodfireai/status/1986495330201051246)。
    - 团队权衡了公共基准测试和可复现性的法律模糊性，并指出这对分享基于许可语料库训练的模型和结果产生了寒蝉效应。
- **合成 QA 与 Tokenizers 取得进展**：讨论提到了使用类似 [Nemotron‑CCh](https://arxiv.org/abs/2511.08923v1) 论文中的系统生成的**合成 QA** 模式（例如 QA tails），以及 [PleIAs: The New Data Frontier](https://pleias.fr/blog/blogsynth-the-new-data-frontier) 中涵盖的更广泛趋势。
    - 一篇新的 **tokenizer** 预印本论文——[Tokenizer Paper](https://arxiv.org/abs/2511.09709)——因其在减少现代多语言、多模态语料库的分段并提高压缩率方面的潜力而引起关注。

**4. 巨额资金与算力：融资与数据中心**

- **Parallel Web 成功融资 1 亿美元**：**Parallel Web Systems** (Parag Agrawal) 完成了 **1 亿美元 A 轮融资**，旨在为 **AI agents** 构建网络基础设施，公告见：[Parag on X](https://x.com/paraga/status/1988729121636294682)。
    - 开发者们对该产品设计表示赞赏，并推测该公司可能会为生产系统标准化 SDK、抓取/托管层以及 Agent 原生协议。
- **Anthropic 打造 500 亿美元算力巨兽**：**Anthropic** 宣布计划在**德克萨斯州**和**纽约州**投资 **500 亿美元**建设**美国数据中心**，引发了关于国内算力规模的辩论：[Anthropic on X](https://x.com/anthropicai/status/1988624013849935995)。
    - 从业者权衡了人员配备和环境限制，以及充足的本土算力对训练和推理流水线的利好。
- **Nous 将 Hermes 价格削减 70%**：**Nous Research** 将 **Hermes 4 70B** 和 **Hermes 4 405B** 的 API 定价降低了 **70%**，通过 [Nous Portal](https://portal.nousresearch.com/) 开放了更广泛的访问，公告见 [X 平台](https://x.com/NousResearch/status/1989077400957911394)。
    - 开发者预计这将带来更廉价的迭代微调和代码辅助实验，并注意到 Hermes 作为代码模型在各种新兴 IDE Agent 中的存在。

**5. 开源 Agent 工具：Agents、OCR 与 CI**

- **Agent 大军协调 114+ 个子 Agent**：一个拥有技能系统、**114+ 个子 Agent** 和执行规划层的开源编码 Agent 框架已发布，项目为 [agent‑instructions (GitHub)](https://github.com/flora131/agent-instructions)，并附带设计文章：[AI Coding Infrastructure](https://alexlavaee.me/blog/ai-coding-infrastructure/)。
    - 团队将其定位为增强现有开发工具，将多步骤功能委托给专门的工作节点，同时保持人工参与（human-in-the-loop）进行审查和合并。
- **Propercode 承诺提供生产级 PR**：**Propercode** 首次推出了一个由 **Pydantic AI** agents 驱动的代码库多 Agent CLI，旨在实现可靠的代码编辑和审查：[proper-code (GitHub)](https://github.com/JaiSuryaPrabu/proper-code)。
    - 其 **v0.1** 路线图宣传了多种模式（自主模式、学习指南）和多工具设置，以稳定 CI 流水线中的编码准确性。
- **快速部署 DeepSeek OCR**：一个容器化的 **DeepSeek OCR API** 支持通过 **Unsloth** 进行快速自托管和推理，可从 URL 或 base64 提取图像：[deepseek-ocr-api (GitHub)](https://github.com/neosantara-xyz/deepseek-ocr-api)。
    - 社区将其定位为一个轻量级 OCR 微服务，用于为端到端 LLM 应用中的文档解析和 RAG 摄取阶段提供支持。

---

# Discord: 高层级 Discord 摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.1 引起褒贬不一的反应**：[GPT-5.1](https://openai.com/blog/new-models-and-developer-products) 在 **LMArena** 上的发布引发了争论，一些用户声称其表现优于 **Sonnet 4.5**，而另一些人则认为由于硬编码的 UI 指令，其表现令人失望。
   - 虽然一些人称赞其增强的推理和代码能力，但也有人批评其缺乏创意，一位成员称其为 *“非正式发布的填充物”*。
- **Gemini 3 猜测升温**：对 **Gemini 3** 发布的期待正在增长，猜测指向本周或下周发布，可能与 **Sima 2 research** 的公告同时进行。
   - 潜在的延迟归因于 **Kimi2** 和分阶段发布策略，可能从 **Gemini Enterprise** 或 **Canvas** 上的移动版本开始。
- **Code Arena 的调整令用户沮丧**：[Code Arena](https://lmarena.com) 的更改（包括因涉嫌滥用而移除对战模式中的重试按钮）引发了用户的不满。
   - 关于该平台上 **Riftrunner** 状态的问题也已出现，有报告称在访问聊天记录时出现错误和 Cloudflare 超时问题。
- **开源模型获得关注**：围绕 **GLM 4.6** 和 **Grok** 等开源模型的热情正在高涨，它们因代码能力和性价比而受到称赞。
   - [Hugging Face](https://huggingface.co/) 的所有者认为 *“世界属于开源模型”*，并分享了一个使用 **Open Empathic** 的 [YouTube 教程](https://www.youtube.com/watch?v=GZqYr8_Q7DE)。
- **Gemini Canvas 版本评价两极分化**：用户对 [Gemini Canvas](https://aistudio.google.com/app/canvas) 版本的体验各异，一些人赞赏其惊艳的 UI，而另一些人则不为所动。
   - 关于它是否使用了 **Gemini 3** 还是仍停留在 **2.5 Pro** 版本的猜测不断，一些人认为这是一种 *“集体幻觉”*。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT 5.1 登陆 Perplexity Pro**：正如 Discord 频道中所宣布的，**GPT 5.1** 现已面向 **Perplexity Pro** 和 **Max** 订阅者开放。
   - 公告中附带了一张图片，可能展示了 **GPT 5.1** 的新功能或能力。
- **Perplexity 合作伙伴计划封号引发骚乱**：用户报告称因涉嫌 *“欺诈活动”* 被 **Perplexity 合作伙伴计划封号**，且自 2025 年 11 月 11 日以来的申诉均未得到回复，用户猜测使用 VPN 或超过推荐限制可能是原因。
   - 一种理论认为，封号时间选在推荐奖励的 30 天锁定期前后，并声称高收入者是被针对的目标，这引发了用户的强烈不满。
- **Pro 用户抱怨计费和限制问题**：用户报告了 Perplexity Pro 的问题，包括 **图像生成限制**、lab 代币重置日期不准确以及深度研究工具调用问题。
   - 具体来说，用户表示 *“如果我没记错的话，限制会在每月的第一天重置”*。此外，使用 comet 进行 Google 登录的问题也开始浮现，同时人们也在尝试弄清楚如何有效地使用不同的 AI 模型。
- **GPT-5.1 的部署充满疑虑**：成员们注意到了 **GPT 5.1** 的推出，但不确定是否有官方发布，或者人们是否真的收到了新模型。
   - 一些用户报告称，在启用 Canvas 设置时，新模型与 **Gemini 3.0** 相似，但也提到它在回答问题方面表现更好。
- **计划混乱中诈骗指控上升**：在混乱中，一些用户声称 **Perplexity 正在运行一个诈骗项目**，取消了每个人的奖励，而另一些人则为该计划辩护，强调用户必须合法合规才能获得报酬。
   - 一位用户分享了朋友收到近 **$900** 的证据，但也承认在他们的服务器里 *“喷 Perplexity”*，而其他人则在艾特管理员，并对有关其奖励的邮件未获回复表示沮丧。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA Alpha 值至关重要**：一位进行 **LoRA 微调 (finetuning)** 的成员发现，**LoRA alpha** 必须是 **rank** 的一半才能避免梯度爆炸，而另一位成员则指出 [Unsloth 文档](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)可作为*基准指导*。
   - 讨论强调，最优的 **LoRA alpha** 设置是因情况而异的，需要进行实验。
- **Nvidia 公布 RTX 价格与规格**：Nvidia 发布了一款 [72GB RTX5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)，拥有约 **1.4TB** 带宽，售价约 **$5,000**，引发了关于 **48GB** 版本是否物有所值的争论。
   - 一位成员指出 **6000 Pro** 的价格在 3-4 个月内下降了 **25%**，并强调拥有硬件的关键优势在于隐私。
- **社区 GPU 资源探索**：一位成员提议从 **28,000** 名成员中每人每月集资 **$1**，以创建强大的 Unsloth 计算基础设施。
   - 另一位成员开玩笑说，根据 **Pro 6000** 型号的价格，*最后每个人都拿到 GPU 需要 666 年*。
- **Instant 模型做出争议性选择**：讨论围绕 **"Instant" 模型** 的首个 Token 响应时间 (**TTFT**) 展开，猜测其速度是否与更短的思考时间有关，或者它是一个完全不同的模型。
   - 一位成员推测，其目标是拥有*一个能在内部处理选择的模型，而不是让你去挑选 5 个不同的模型*，这一想法并未受到所有人的欢迎。
- **DeepSeek OCR API 已部署**：一位成员宣布推出一个工具，只需几个步骤即可在自己的 **DeepSeek OCR 模型**上部署并运行推理，该工具已在 [GitHub](https://github.com/neosantara-xyz/deepseek-ocr-api) 上可用。
   - 该工具使用 **Unsloth** 进行推理，可从 URL 或 base64 编码数据中提取图像。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.1 亮相，伴随智能增强与争议**：**GPT-5.1** 正在推出，具备增强的智能和对话能力，详见 [OpenAI 博客文章](https://openai.com/index/gpt-5-1/)。然而，一些用户认为它是一种*退步*，且对 [自定义指令 (custom instructions)](https://platform.openai.com/docs/gpt-best-practices) 的处理不佳。
   - 明天下午 2 点（太平洋时间）将举行 [Reddit AMA](https://redd.it/1ovkt6n/) 讨论 **GPT-5.1**。同时，一些用户反映它在故事线中存在*过度总结*和*重复场景*的问题，导致部分用户换回 **GPT 4.1** 进行创作。
- **Gemini 3.0 向 Pro 订阅用户低调推出**：根据 [Google 博客](https://blog.google)，Google 已开始通过 **Gemini Drops**（每月功能更新）向 Pro 订阅用户*低调推出* **Gemini 3.0 Pro**，重点关注开发者工具和企业/Workspace 应用。
   - 正如 [YouTube 链接](https://www.youtube.com/watch?v=0-CbsNB9tdk)和[其他链接](https://marketingtrending.asoworld.com)所示，免费用户将在*未来几周内*收到更新。
- **ChatGPT Canvas 模式受 Bug 和崩溃困扰**：用户报告称，在使用 **Canvas 模式** 时 **ChatGPT** 网站会出现崩溃和 Bug，但一位用户通过 *Tampermonkey 脚本* 修复了长对话问题。
   - 这种体验因人而异，部分用户即使在短对话中也会遇到问题。
- **寻求免费、无限的 AI 视频生成**：用户正在讨论免费、无限 AI 视频生成的可行性，并注意到 **Grok** 提供免费 AI 视频，但时长有限。
   - 共识是，巨大的功耗和处理需求解释了为什么无限制访问通常仅保留给 Pro 订阅用户。
- **更严格的图像生成限制引发用户不满**：新的 [图像生成 Guardrails](https://openai.com/policies/usage-policies) 被认为过于严格，阻碍了简单的描绘，并令无法编辑文本的用户感到沮丧。
   - 用户哀叹功能的丧失，并对当前的能力表示不满。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 停止支持 Polaris Alpha**：**Polaris Alpha** 模型（**OpenAI GPT-5.1** 的早期无推理版本）现已弃用。根据 [OpenAI](https://openai.com/index/gpt-5-1-for-developers/) 的消息，它将被更快、更高效、具备自适应推理和更强编码能力的 **GPT-5.1** 取代。
   - OpenRouter 已推出 **另外三款 GPT-5.1 模型**：[GPT-5.1 Chat](https://openrouter.ai/openai/gpt-5.1-chat)（即 ChatGPT 中的 Instant）、[GPT-5.1-Codex](https://openrouter.ai/openai/gpt-5.1-codex) 和 [GPT-5.1-Codex-Mini](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113)。
- **隐私设置解析**：一位用户通过在 [OpenRouter 隐私设置](https://openrouter.ai/settings/privacy) 中开启隐私开关 *“Enable free endpoints that may train on inputs / publish prompts”*（启用可能对输入进行训练/发布提示词的免费端点）解决了 API 错误。
   - 该设置似乎会影响某些模型的可用性。
- **API Rate Limit 问题持续**：用户报告 API 频繁出现 **Error 429**（速率限制）问题，尤其是 `anthropic/claude-sonnet-4` 和 `anthropic/claude-sonnet-4.5`，这表明供应受限。
   - 尽管 [OpenRouter 状态页面](https://status.openrouter.ai/) 显示无异常，但部分用户仍遇到了 Cloudflare 错误和超时。
- **GPT-5.1 Token 输出引发争议**：**GPT 5.1** 正在 ChatGPT 上推出，早期基准测试表明，极小的改进却带来了近 **2 倍的 Token 输出**，引发了对性价比的担忧。
   - 一位用户调侃道：*“我们要的是少思考，而不是多思考”*。
- **React Compiler 拯救局面**：一位成员表示他们开始使用 **React Compiler**，它解决了 *React Slop*（冗余代码）问题，并称赞其 *非常出色*。
   - 另一位成员开玩笑地质疑了这一说法的严肃性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Helion 协调向后兼容性**：**Helion** 将实现 **BC compatible**（向后兼容，例如从 **0.2** 到 **0.2.1**），版本 **0.2.2** 刚刚发布，并计划每隔一两周发布一次次要版本以保持 **pypi** 包的更新，详见 [issue 164](https://github.com/pytorch/helion/issues/164)。
   - 与解释模式较慢的 **Triton** 相比，**Helion** 的解释模式在使用 eager **PyTorch** 时速度惊人；此外，现在通过传递 `configs=` 可以启用类似于 **Triton autotune** 的 **autotuning** 功能。
- **NCU 在云厂商评估中占据重要地位**：据 [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) 报道，云供应商现在根据对 **NCU** (**NVIDIA Compute Unified Device Architecture**) 的支持程度进行分级。
   - 对 **NCU** 支持的推动反映了其在云计算领域日益增长的重要性。
- **NVFP4 黑客松推动 GEMV 优化**：一项旨在针对 **Blackwell GPU** 架构优化 **NVFP4** 数据类型工作负载的黑客松已启动，首个挑战聚焦于 GEMV（矩阵-向量乘法），详见 [博客文章](https://veitner.bearblog.dev/nvfp4-gemv/)。
   - **Datacrunch B200 GPU** 提供了访问权限，并推荐使用 **CuTeDSL**，因为它能在保持生产力的同时接近硬件底层。
- **开源 AI 编程 Agent 协同作战**：一套旨在交付高质量代码的 AI 编程 Agent 配置已开源，具备多种技能、**114+ 个子 Agent** 以及用于处理复杂功能的执行规划框架；可以在 [GitHub](https://github.com/flora131/agent-instructions) 和 [这篇博客](https://alexlavaee.me/blog/ai-coding-infrastructure/) 中找到。
   - 该配置旨在增强现有的 AI 编程工具和工作流。
- **Restrict 限定符引发警示**：一位成员分享了一个涉及 `__restrict__` 限定符的潜在 **GPU 编译器 Bug** 的有趣 [案例](https://godbolt.org/z/ad98nYdrf)，并征求关于这是真正的 Bug 还是仅仅是 **未定义行为 (UB)** 的意见。
   - 代码示例包括 **PTX** 和 **SASS** 输出，邀请大家对编译器处理内存别名约束的方式进行深入分析。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5.1-Codex 引起 Cursor 工程师兴奋**：成员们报告称 **GPT-5.1-Codex** 正在推出，部分用户在 [最新的 codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) 中发现了它，并注意到它与 [Windsurf](https://www.windsurf.ai/) 的集成。
   - 一些用户尚未收到更新提示，而另一些用户发现它在 OpenAI API 正式发布前已在 Codex 中可用，这表明是分阶段推出的。
- **发布后 Auto Mode 性能骤降**：用户报告 **Auto mode 性能** 明显下降，一位用户表示 *20 分钟前 Auto mode 还在使用另一个更快的模型，现在它使用 gpt 5 codex，速度变慢了，甚至在没有推理能力的模型都能轻松处理的情况下无法编辑文件*。
   - 一些人认为这可能是由于 **5.1 发布** 后 **Auto mode** 默认连接到可能过载的 OpenAI 服务器，而另一些人则推测 Cursor 使用的是具有高思考要求的 GPT-5 版本。
- **自定义命令降低成本**：成员们讨论了使用 **custom commands** 来自动化任务，例如在代码更改后运行测试，一位用户分享了通过 **CTRL + K** 创建自定义命令以触发特定指令的方法。
   - 一位成员建议通过执行特定命令确保自动触发测试，这些命令可以在 [Cursor 设置中的 docs 部分](https://cursor.com/docs/customcommands) 进行标记，以提高 token 效率。
- **Memory 功能引发疑虑**：Cursor 的 **memory feature** 引起了褒贬不一的反应；一位运行在传统隐私模式下的用户犹豫是否要禁用它。
   - 一位用户描述说 *memories 通常在你的个人资料中，且需要处于激活状态，如果你处于隐私模式，这种 tool calling 将会失败*，因此如果你想充分利用 Cursor，这是一个需要关注的功能。
- **序列化错误阻碍终端任务**：用户报告在使用终端命令时出现 *Serialization error*，这会导致对话中断直到重启 Cursor，一位用户报告称问题源于包含空格的文件路径命令。
   - 成员们将问题定位在序列化反馈给 LLM 的终端输出时出错，并分享了指向 [Serialization Error 论坛帖子](https://forum.cursor.com/t/serialization-error/124671) 的链接。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 大幅下调 Hermes 4 的 API 价格**：Nous Research 宣布将其 **Hermes 4 70B** 和 **Hermes 4 405B** API 的价格**下调 70%**，旨在降低访问门槛。
   - 详情和注册可在 [Nous Research Portal](https://portal.nousresearch.com/) 和 [X](https://x.com/NousResearch/status/1989077400957911394) 上查看。
- **社区渴望 Schizomax 模型**：成员们讨论了对 **schizomax model** 的需求，指出 **Grok** 和 **Nous 模型** 很少拒绝请求，而不像限制越来越多的 **GPT models**。
   - 用户对企业影响和健康检查限制了 **OpenAI GPT models** 的实用性表示沮丧。
- **Hermes4 作为代码模型出现在 Cline 上**：用户观察到 **Hermes4** 现在作为代码模型出现在 Cline 上，并分享了其提示词界面和能力的截图。
   - 这一进展引发了人们对 **Hermes4** 在代码相关任务中不断演进的应用的兴奋。
- **将 GGUF 文件导入 Nous Chat 时遇到挑战**：一位用户询问如何将 **GGUF** 文件导入 **Nous Chat**，但目前该功能尚不支持。
   - 成员们建议使用 [llama.cpp](https://huggingface.co/docs/hub/en/gguf-llamacpp) 或 [Ollama](https://ollama.com/) 在本地运行 **GGUF** 文件，以获得更简单的设置。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **焕然一新的 Lakh MIDI 数据集发布**：一个经过完整清理和组织的 **Lakh MIDI Dataset** 已发布，包含一个结构化的 **JSON 文件**，拥有超过 **44,000 个条目**，并计划将其上传至 [Hugging Face](https://huggingface.co/)。
   - 该数据集具有完整的解析和一致性，欢迎社区进行协作和增强。
- **法语维基百科完成清理，现已上线 Hugging Face**：一位用户将清理后的 **法语维基百科数据库** 版本上传至 [Hugging Face](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)，包含超过 **2,700,000 个 JSON 格式的文件**，英语版本紧随其后。
   - 清理过程不仅包括纯文本，还涵盖了 *templates, tables, html, 和 refs*，同时保留了 infobox 信息和链接。
- **JSONL 成为文本数据的首选格式**：对于纯文本数据，使用 **JSONL/NDJSON** 是首选，因为它允许逐行读取，简化了处理过程，而不像 tar 文件那样带有繁琐的 header。
   - 在数据集格式的讨论中，成员强调了 **JSONL** 相比于管理 tar header 的易用性。
- **NVIDIA 数据集许可协议引发争议**：人们对 **NVIDIA 数据集许可协议** 中的限制日益担忧，特别是关于训练、评估和公开结果分享方面的限制，详情见 [一段 X 讨论](https://x.com/goodfireai/status/1986495330201051246)。
   - 主要担忧集中在 **NVIDIA** 有权随时终止许可，这可能会使已授予的权限失效，从而导致法律上的不确定性。
- **Anthropic 因对华政策遭受抨击**：一名成员指责 [Anthropic](https://x.com/AnthropicAI/status/1989033793190277618) 利用**散布恐惧**来获取战略优势，特别是针对**非美国**和**中国实验室**。
   - 人们对 **Anthropic 的数据隐私实践** 以及可能将安全置于隐私之上的做法提出了质疑，同时对 **Anthropic CEO 对华立场** 进行了审查。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 最终可能会支持私有属性**：**Mojo** 最终可能会支持 **public 和 private 成员/方法**，但前提是必须存在打破封装的“逃生舱 (escape hatch)”机制。
   - 目前，**Mojo** 使用 **Python 的下划线约定** 来暗示私有性，它最适合缺乏完善生态系统的*重计算任务*。
- **Modular 技术栈在回归模型上的繁琐流程**：目前使用 **Modular 技术栈** 构建回归模型需要编写解析器和数据可视化库，由 **MAX** 处理训练的 **backwards pass**。
   - 最快的方法是使用 **Torch** 进行训练，并使用 **MAX** 进行推理。
- **`comptime` 涵盖关键字功能**：**Mojo** 中的 `comptime` 关键字现在涵盖了以前 `alias` 的功能，包括类型赋值，从而实现了类似 Zig 风格的 **static reflection**（静态反射），例如 `comptime InnerTypeFirstField = T.fields[0].type`。
   - 虽然*对于类型赋值来说读起来有点奇怪*，但在编译时激进地混合类型和值时，拥有不同的关键字可能会令人烦恼。
- **Apple Silicon 支持催生超级计算机之梦**：得益于社区的 PR，扩展了对许多 **intrinsics** 的支持，并启用了更多测试和 **GPU puzzles**，基础的 **MAX graphs** 已经开始运行。
   - 一名成员报告说，得益于 **GPU puzzles**，*在本地开发 kernel 并在超级计算机上部署的梦想*正在成为现实。
- **HipKittens 论文揭示 Kernel 性能瓶颈**：[HipKittens 论文](https://arxiv.org/abs/2511.08083)指出，**Mojo 的 MHA kernel** 存在严重的 **bank conflicts**，在 **MI355X** 上仅达到峰值 kernel 性能的 50%。
   - 一名成员建议，如果 **LLVM** 能与设备通信，就可以在编译时构建抽象，从而可能减少对 AMD/NVIDIA 特定 kernel 的需求。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Whisper 仍然是转录领域的王者**：据一位成员称，**Whisper** 仍然是开源转录的首选，特别是通过 **Whisper-server** 量化为 **8 位和 4 位** 的 [large-v3 模型](https://openai.com/blog/whisper)。
   - 有人指出，支持 Vulkan 的 **Whisper.cpp** 提高了便携性，这与直接在 Python 中通过 PyTorch 运行 **Whisper** 不同。
- **ICLR 评审质量下降**：成员们观察到 **ICLR** 评审质量有所下降，理由是出现了*大量残留有 prompts 的 LLM 评审*，并建议增加投稿长度限制，并对*明显是垃圾的论文进行预评审拒绝*。
   - 一位成员表示，相比之下 *NeurIPS 表现更好*。
- **SIMA-2 进入虚拟舞台**：一位成员分享了 [DeepMind 的 SIMA-2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)，这是一个在虚拟 3D 世界中进行*游戏、推理和学习*的 Agent。
   - 他们质疑如果这些模型生成的图像带有现有游戏的纹理，版权打击（copyright striking）是否对这些模型有效。
- **跷跷板式偏好困扰推荐系统**：一位成员讨论了通过故意**反复横跳（seesawing）**自己的偏好来迷惑推荐算法，并链接了[这篇论文](https://arxiv.org/abs/2511.08378)。
   - 该用户报告称，该链接是在一个 X 帖子中作为对相关论文的识别而分享的。
- **GPT-5 的对话风格是为了消耗 Token？**：讨论围绕 **GPT-5** *更具对话性*的风格是否旨在诱导 **GPT-4** 用户在 [output tokens](https://openai.com/index/gpt-5-1/) 上花费更多。
   - 一位成员推测，不同用户层级（特别是免费版本）可能会对 output tokens 施加潜在限制。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 在前端技能方面受限**：虽然发现 **Claude** 的后端技能很有效，但一位成员指出其前端技能只是*尚可*，仍会犯错，并引用了[这篇博客文章](https://www.claude.com/blog/improving-frontend-design-through-skills)。
   - 该成员建议 **Claude** 在前端开发任务中仍有改进空间。
- **Latent Space Spotify 订阅源短暂失效**：由于免版税片头曲遭遇版权投诉，Latent Space 的 **Spotify 订阅源** 出现问题，引发了听众的关注。
   - 随后有报道称 [Spotify 已解决该问题](https://x.com/paraga/status/1988729121636294682/photo/1)，订阅源已恢复正常。
- **Parag 的 Parallel Web 获得 1 亿美元融资**：**Parag Agrawal** 的新公司 **Parallel Web Systems** 获得了 **1 亿美元** 的 A 轮融资，用于开发 AI 网络基础设施。
   - 爱好者们赞扬了该公司简洁的设计（如[此公告](https://xcancel.com/paraga/status/1988729121636294682/photo/1)所示），并期待其对 AI 发展的潜在影响。
- **Anthropic 宣布投入 500 亿美元建设数据中心**：**Anthropic** 计划在德克萨斯州和纽约州投资 **500 亿美元** 建设美国数据中心，这将刺激建筑就业并引发关于国内算力（compute capacity）的讨论。
   - 该公告发布在[此处](https://xcancel.com/anthropicai/status/1988624013849935995?s=46)，引发了关于人员配备、环境问题以及潜在 AI 行业泡沫的辩论。
- **Holo2 比 GPT-4V 更快且更便宜**：**HCompany** 推出了 **Holo2**，这是一个基于 **Qwen3-VL** 的更经济的 **30B-MoE** 视觉模型，在 UI 基准测试中超过了 **GPT-4V**，并可在 Web、桌面和移动平台上运行。
   - 更多细节可在[此公告](https://xcancel.com/hcompany_ai/status/1989013556134638039)中查看，重点介绍了其性能和多功能性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Upscaled Model 引发笑声**：一个经过微调的 **upscaled model** 成为了笑料，促使用户分享了一个 [Beavis and Butthead GIF](https://tenor.com/view/beavis-butthead-beavisandbuttheadshow-gif-13367061995602198225) 作为幽默的回应。
   - 轻松的调侃中还提到了特定的 Discord 频道，增加了讨论的喜剧色彩。
- **Multi-Head Attention 详解**：针对用户关于为什么 **multi-head attention** 的各个头不会学到相同内容的问题，讨论指出随机初始化和 [softmax 放大差异](https://huggingface.co/datasets/John6666/forum2/blob/main/multi_head_attention_1.md) 是关键因素。
   - 独立的参数和不同的梯度进一步促进了各个头的自然分化，学习任务保留了这些有用的差异。
- **HuggingChat 转向付费，用户表示不满**：成员们对 **HuggingChat** 转向付费模式表示失望，指出与之前的无限版本相比，免费功能受到了限制，正如这幅 [截图](https://cdn.discordapp.com/attachments/879548962464493619/1438593728200572928/Screenshot_2025-11-13-18-16-43-962_mark.via.gp-edit.jpg?ex=69181b10&is=6916c990&hm=7bc833942ddb303310bdca35fdf5e940a15a9eb9a345cbc6086e0a21f8e817c6&) 所示。
   - 一位用户感叹新版本表现平平，质疑 Hugging Face 致力于免费提供开源 AI 平台的承诺是否正在动摇。
- **AI Voice 实现激发社区热情**：社区对实现 **AI voice** 充满热情，包括将语音集成到 *成人色情机器人（smutty sexbot）* 的想法，并分享了一个包含 20 种人类情感的 [开源语音设计](https://huggingface.co/maya-research/maya1)。
   - 一位用户询问了实现的难度以及将其集成到 Android 应用中的可能性。
- **Propercode 承诺生成生产级代码**：一位成员介绍了 **Propercode**，这是一个用于代码库的多 Agent 编码 CLI 工具，由以图（graph）形式编排的 **Pydantic AI** Agent 驱动，目标是实现可靠的编码准确性 ([GitHub](https://github.com/JaiSuryaPrabu/proper-code))。
   - 该工具目前处于 **v0.1** 版本，将为 Agent 提供多种工具和模式，如自主模式、学习指南模式等。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 被误解为无需 Prompt 的框架**：成员们讨论了 **DSPy** 被误认为是一个不需要对 LLM 进行 prompting 的框架，这具有误导性，因为即使有示例和 **GEPA**，在特定领域应用中仍然需要提示。
   - 一位成员澄清说，**DSPy** 将 prompting 封装为可编程的优化模块，签名（signature）指令或 docstrings 实际上起到了 prompt 的作用。
- **Pydantic Models 在签名中优于简单字符串**：一位成员表示，在签名中，他更倾向于使用 **Pydantic models** 而不是简单的字符串类型 (`input -> output`)，理由是需要更复杂且经过类型检查的实现。
   - 他们指出，签名内的指令充当了 prompt，并强调社区的困惑源于对 *prompts* 的不同解读。
- **Regex 助力稳健的 Agentic Search**：为了改进 **Agentic search**，一位成员在其 **ReAct module** 中指示 LLM 使用特定术语通过 **ripgrep** 进行工具搜索，并以 **Regex** 作为备选方案。
   - 这一指令对于 LLM 在搜索工具中有效使用 Regex 至关重要，尤其是在访问多个工具（3-4 个函数）时。
- **调查问卷语言引发怀疑**：一位成员根据一张突出显示顶部 **fine-tuning** 的截图，怀疑 **survey language** 存在问题。
   - 另一位成员形容该调查语言非常“疯狂”，因为它被埋在附录中，而 **fine-tuning** 却被显著地标出。

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2 与 Together AI 合作**：**Moonshot AI** 团队宣布将与 **Together AI** 合作，深入探讨 **Kimi K2 Thinking**，计划于 **2025 年 11 月 19 日上午 9 点（PT）**举行，注册地址为 [luma.com/g5qcq85z](https://luma.com/g5qcq85z)。
   - 该公告强调了 **1T MoE** (Mixture of Experts) 的强大功能，以及在**单次运行中执行 300 次工具调用**的能力。
- **GLM-5, K3, R2 模型发布在即**：成员们期待 **GLM-5**、**K3** 和 **R2** 的到来，其中一人表示：*我对 Gemini 3 之类的模型一点也不兴奋，因为它们不像 Kimi/GLM 那样提供良好的编程方案*。
   - 尽管进行了 **int4** 优化，据报道 **Kimi K2-thinking** 的速度仍比 **GLM-4.6** 慢 **1.8 倍**，比 **Minimax m2.wysh.3** 慢 **2.5 倍**，但在非纯编程任务中能力更强。
- **YC 的 Kimi For Coding 引发争议**：由 **Y Combinator** 支持的 **Kimi For Coding** 被批评为对一个平庸模型的“明火执仗的抢劫”，根据[这条推文](https://x.com/ycombinator/status/1988366241460089118?s=46)，它每周提供 **2048** 次使用额度。
   - 一位用户表示：*我简直不敢相信他们认为这种产品必须存在*。
- **Moonshot AI 加入造芯竞赛**：继美国 AI 实验室之后，中国 AI 实验室也开始进入芯片制造领域；用户们参考[这条推文](https://x.com/tphuang/status/1988952992003891330?s=46)期待 **Moonshot K100**。
   - 用户质疑 **Kimi Chat** 网站上缺少项目（project）和自定义指令功能，而其他人则澄清**工具调用 (tool use)** 在 **Claude Code** 或 **Kimi-CLI** 上是自动启用的。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5.1 系统卡片引发批评**：**GPT-5.1** 的发布因缺失基准测试而引起关注，一些人称其 [系统/模型卡片](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf) 是个“笑话”。
   - 怀疑论者指出缺少 API 这一点很可疑，暗示它可能只是修改了系统提示词（system prompts），而不是一个全新的模型。
- **Aider-ce 维护代码**：**Aider-ce** 因其现有功能而获得赞誉，但其维护者在沟通和接班计划方面存在疑虑。
   - 尽管有这些担忧，该分支仍受到社区的好评，每天都有新用户加入。
- **Deepseek API 加速 Agent 模式**：用户报告称 **Deepseek API** 显著提升了 **Aider-ce** 的 **Agent** 模式性能，而该模式在 **GPT-5-high** 上运行缓慢。
   - 运行缓慢归因于在大型仓库上重新生成 **repo map**，建议调整 `repo_map_max_files` 和 `repo_map_token_budget`。
- **moonshotai Kimi-K2 在 Aider 上运行**：用户通过将 **OPENAI_API_BASE** 变量设置为 `https://llm.chutes.ai/v1/`，修复了在运行 aider 与 `moonshotai/Kimi-K2-Thinking` 时出现的 *404 no cord found* 错误。
   - 正确的命令是在运行 `aider --model openai/moonshotai/Kimi-K2-Thinking` 之前执行：
```bash
SET "OPENAI_API_BASE=https://llm.chutes.ai/v1/"
SET "OPENAI_API_KEY=mykey"
```
- **DeepSeek 存在商业隐私担忧**：一位成员报告称，他几乎在所有事情上都“被 **DeepSeek** 圈粉”了，但仅在开源或个人项目中使用 **DeepSeek API**。
   - 另一位成员询问是否有服务在提供 **DeepSeek** 的同时尊重商业隐私，尤其是因为第一位成员表示他们使用的是原始 **DeepSeek API**，且没有商业需求。

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Server 询问查询结果格式**：一位正在构建 **Postgres MCP server** 的成员正在寻求关于 **executeQuery MCP tool** 最佳输出格式的建议，考虑使用 **JSON**、**Markdown** 或 **Toon** 格式来返回 **SQL query results**。
   - 他们正在探索以用户友好方式展示 **SQL queries** 数据的最佳方法。
- **MCP 协议范围澄清**：成员们澄清说，虽然 **MCP protocol** 原生不支持从客户端到服务器的文件传输，但可以通过 **resources** 实现。
   - 一般性的 **MCP implementation discussions** 应该使用 [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) 或其他社区 Discord 服务器，因为此 Discord 服务器专门用于讨论协议及相关的官方 SDK 项目。
- **SE Upload Call 实现疑问**：一位用户询问了如何通过使用工具的 **SE upload call** 直接从聊天界面上传文件，特别是针对 [issue 1306](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1306)。
   - 他们被引导去开启一个 [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) 或在其他社区 Discord 服务器中提问，因为此 Discord 服务器是用于讨论 **protocol and related official projects** 的。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 支持遇到障碍**：用户报告 **Manus' checkpoint system** 无法找到 git commits，这导致 *'Publish'* 按钮被锁定，迫使他们手动同步内部仓库。
   - 在询问关于聊天模式被移除的问题后，他们被引导至 [Manus feedback](https://manus.im/feedback) 获取支持。
- **Manus 发布新额度方案**：19 美元的方案现在每月包含 **4000 credits**，相比之前的 **1900 credits** 有了大幅增加。
   - 用户好奇在这一调整的同时，额度消耗方式是否也发生了变化。
- **AI/ML 工程师加入**：一位专注于模型设计、优化和大规模部署的 **AI/ML engineer** 加入了服务器，其技术栈包括 **Python, C++, Rust, SQL, Hugging Face, ONNX Runtime, Triton Inference Server** 和 **Apache Spark**。
   - 他们精通大规模部署模型。
- **工作流自动化工程师提供服务**：一位在 **workflow automation, LLM integration, RAG, AI detection, image, and voice AI** 方面经验丰富的工程师正在提供服务。
   - 他们使用 **Dspy, OpenAI APIs** 和自定义 agents 构建了自动化流水线和任务编排系统，并分享了 [他们的作品集](https://devx-green.vercel.app/)。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **OpenCL 设备错误修复建议**：用户建议了一个修复方案，当未发现 **OpenCL Devices** 时抛出 `RuntimeError`，并参考了关于 `CL_INVALID_VALUE` 错误的 [OpenCL documentation](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html)。
   - 这解决了一个代码在没有设备的情况下继续运行并可能导致问题的场景。
- **pmap 和 vmap 功能增强 tinygrad**：一位用户更新了 [README](https://github.com/tinygrad/tinygrad)，强调了对 **pmap** 和 **vmap** 功能的需求，这些功能可以实现更高效的数组操作。
   - 该用户利用 **ChatGPT** 协助完成了文档更新。
- **torch_load 现在兼容 VGG16 模型**：拉取请求 ([PR #13253](https://github.com/tinygrad/tinygrad/pull/1325)) 使得 **torchvision** 托管的 **VGG16** 模型能够与 `torch_load` 配合使用，增加了可用模型的数量。
   - 这一增强扩展了 **tinygrad** 与 **torchvision** 库中预训练模型的兼容性。
- **OpenPilot PR 已合并，扩展集成**：[OpenPilot PR](https://github.com/commaai/openpilot/pull/36615) 已合并，增强了 **tinygrad** 与 **OpenPilot** 之间的集成或兼容性。
   - 团队承诺防止回归（regressions），显示了集成的重要性。
- **tinygrad 与 C++ 集成的兴趣激增**：一位用户询问关于将 **tinygrad** 与 **C++** 结合使用的情况，意图将其应用于嵌入式系统。
   - 这一请求表明，在资源受限的环境中利用 **tinygrad** 的兴趣日益增长。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 GPT-5.1 和 GPT-5.1-Codex**：**GPT-5.1** 和 **GPT-5.1-Codex** 现已在 Windsurf 中上线，所有付费用户在未来 7 天内可免费使用，且 **GPT-5.1** 已被设为新用户的默认模型，详情见[此公告](https://x.com/windsurf/status/1989069991770214580?s=20)。
   - Windsurf 宣传 **GPT-5.1** 在 Agent 式编程（agentic coding）和前端设计方面比 **GPT-5** 有显著改进，用户可以从[此处](https://windsurf.com/download/editor)下载编辑器。
- **GPT-5.1 强化 Agent 式编程**：根据 Windsurf 的说法，**GPT-5.1** 代表了从 **GPT-5** 开始的实质性升级，特别是在 Agent 式编程任务的背景下。
   - 该模型根据任务复杂度自适应地调节推理深度，从而缩短了大多数任务的周转时间。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道沉寂太久，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道沉寂太久，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：频道详细摘要与链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1438257054333534290)** (1217 条消息🔥🔥🔥): 

> `GPT 5.1, Gemini 3 release, Code Arena Updates, Riftrunner status, Open Source Models` 

- **GPT-5.1 引起困惑和辩论**：成员们对 [GPT-5.1 的性能](https://openai.com/blog/new-models-and-developer-products)进行了辩论，一些人声称其表现优于 **Sonnet 4.5**，而另一些人则因其硬编码的 UI 指令而认为它是“垃圾”。
   - 一些人指出了它改进的推理和编程能力，而另一些人则觉得它缺乏创造力，并不是一次显著的升级，一位成员将其描述为 *“非发布式的填充物”*。
- **Gemini 3 发布日期猜测升温**：关于 **Gemini 3** 发布的猜测仍在继续，一些人预计在本周或下周发布，可能会被 **Sima 2 研究**的公告掩盖。
   - 讨论指向由于 **Kimi2** 可能导致的延迟，以及分阶段发布的可能性（从 Gemini Enterprise 或 Canvas 上的移动版本开始），周五不会发布。
- **Code Arena 经历变革**：讨论围绕 [Code Arena](https://lmarena.com) 的变化展开，特别是由于被认为存在滥用而从对战模式中删除了重试按钮，这引起了用户的沮丧。
   - 还有关于平台上 **Riftrunner** 状态的问题，有错误报告和可能被移除的消息，以及在浏览聊天记录时对 Cloudflare 超时的担忧。
- **开源模型引发兴奋**：一些成员对 **GLM 4.6** 和 **Grok** 等开源模型表示感兴趣，强调了它们作为高性价比替代方案的潜力，并赞扬了它们的编程能力。
   - [Hugging Face](https://huggingface.com/) 的所有者认为 *“世界属于开源模型”*，在相关说明中，一位成员分享了使用 Open Empathic 的 [YouTube 教程](https://www.youtube.com/watch?v=GZqYr8_Q7DE)。
- **Gemini Canvas 版本引发辩论**：用户对 [Gemini Canvas](https://aistudio.google.com/app/canvas) 版本有不同的体验，一些人称赞其极棒的 UI，而另一些人则对此印象不深。
   - 一些用户试图确定它是否路由到了 **Gemini 3**，而另一些人认为它仍然是 2.5 Pro 版本，这一切都是 *集体幻觉*。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1438618401772671191)** (1 条消息): 

> `GPT-5.1 Release, LMArena Updates` 

- **GPT-5.1 强势登陆 LMArena**：一个新模型 **gpt-5.1** 已添加到 [LMArena](https://x.com/arena/status/1989058785927950628) 的 Text、Vision 和 Code Arena 中。
- **LMArena 获得更新**：**LMArena** 上的 Text、Vision 和 Code Arena 今天上线了一个新模型。

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1438608979776245853)** (1 条消息): 

> `GPT 5.1, Perplexity Pro, Perplexity Max` 

- **GPT 5.1 向 Perplexity 订阅者推出！**：正如 Discord 频道中所宣布的，**GPT 5.1** 现已面向 **Perplexity Pro** 和 **Max** 订阅者开放。
- **附图：GPT 5.1 特性？**：公告中附有一张图片，可能展示了 **GPT 5.1** 的新功能或能力。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1438257181261697035)** (1054 条消息🔥🔥🔥): 

> `Comet 推荐计划问题、Perplexity Pro 计费与限制、GPT 5.1 推出、诈骗指控` 


- **Perplexity 合作伙伴计划封号潮来袭！**：许多用户报告称，尽管声称推荐是真实的，但仍因所谓的“欺诈活动”被 Perplexity 合作伙伴计划 **banned**（封禁）。自 2025 年 11 月 11 日以来，向 *partners@perplexity.ai* 和 Dub 支持部门发出的申诉一直未得到回复。
   - 沮丧的用户推测封禁原因多种多样，从推荐人使用 VPN 到超出推荐限制不等。一种理论认为，封禁时间点选在推荐奖励的 30 天锁定期前后，甚至有人声称收益最高的人成了针对目标。
- **Perplexity Pro 用户面临限制和计费困扰**：用户报告了 Perplexity Pro 的问题，包括 **image generation 限制**、Lab Token 重置日期不准确以及 Deep Research 工具调用问题。
   - 具体而言，用户表示 *如果我没记错的话，限制会在每月的第一天重置*。此外，使用 Comet 进行 Google 登录的问题也开始浮现，用户正试图弄清楚如何有效地使用不同的 AI 模型。 
- **GPT-5.1 的部署困境：它真的更优越吗？**：成员们注意到 **GPT 5.1** 的推出，但不确定是否有官方发布，或者人们是否真的收到了新模型。
   - 一些用户报告称，在启用 Canvas 设置时，新模型与 **Gemini 3.0** 相似，但也提到它在回答问题方面表现更好。
- **推荐计划混乱中诈骗指控满天飞**：在混乱中，一些用户声称 **Perplexity 正在运行一个诈骗项目**，取消了所有人的奖励，而另一些人则为该计划辩护，称其合法，并强调用户必须合规才能获得报酬。
   - 一位用户分享了朋友收到近 **$900** 的证据，但承认 *在他们的服务器上对 Perplexity 破口大骂*，而其他人则在 @ 管理员，并对有关其赏金的邮件未获回复表示沮丧。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1438297132850483374)** (420 条消息🔥🔥🔥): 

> `LoRA alpha 值、RTX 5000/6000 价格与规格、社区 GPU 资源、Intel B60 对比 Nvidia 4090` 


- **发现最佳 LoRA Alpha 调优**：一位成员通过惨痛教训了解到，在对基础模型进行 **LoRA finetuning** 时，**LoRA alpha** 必须是 Rank 的一半，否则 Grad Norm 会爆炸，但其他人对此持不同意见。
   - 另一位成员指出，[Unsloth 文档](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) 可作为 *基准指导*，根据具体情况开始进行微调。
- **RTX 5000/6000 价格与规格揭晓**：Nvidia 发布了一款 [72GB RTX5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)，带宽约为 **1.4TB**，售价约 **$5,000**，但有人认为 48GB 的版本在这个价格下并不吸引人。
   - 一位成员指出，**6000 Pro** 的价格在 3-4 个月内下降了 **25%**，拥有硬件的主要好处是隐私。
- **社区 GPU 资源提案**：一位成员提出了一个疯狂的想法：如果所有 **28,000** 名成员每月出资 **$1**，那么 Unsloth 将拥有惊人的算力基础设施。
   - 另一位成员幽默地指出：*每台 Pro 6000 售价 8000，意味着每月能买 3.5 个 GPU，所以 666 年后，每个人最终都能分到一个 GPU*。
- **Intel B60 性能不及 NVIDIA RTX 4090**：围绕 **Intel Arc Pro B60** 展开了讨论，初步评论认为其性能约为 **4090** 的一半，而另一位成员声称其算力仅在 *1/6 到 1/7* 之间。
   - 进一步调查显示，B60 和 B580 非常接近，但驱动程序不稳定，不像 Nvidia 那样省心。


  

---

### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1438709656573968496)** (3 messages): 

> `workflow automation, LLM integration, RAG pipelines, AI content detection, image AI` 


- **专注于 AI 和区块链解决方案的工程师**：一位专注于 **workflow automation、LLM integration、RAG、AI detection、图像与语音 AI 以及区块链开发**的工程师，展示了丰富的实际落地经验。
   - 他们使用 **Dspy、OpenAI API 和自定义 Agent** 构建了自动化流水线，其中包括一个将响应时间缩短了 **60%** 的支持自动化系统。
- **高级 RAG 流水线架构师**：该工程师设计并部署了高级 **RAG pipelines**，结合向量数据库、混合搜索和自定义检索逻辑，以在生产环境中提供准确的响应。
   - 他们还为审核平台开发了 AI 内容检测工具，使用了 **stylometric analysis、embedding 相似度以及微调后的 Transformer**。
- **AWS Lambda 图像 AI 标记与审核流水线**：该工程师利用 **AWS Lambda 和 S3 上的 CLIP 与 YOLOv8** 创建了一个标记与审核流水线，每天为某电子商务平台分类和过滤数千张图像。
   - 在语音 AI 领域，他们使用 Whisper 和 Tacotron2 构建了**语音克隆与转录服务**，通过 ASR、TTS 和 CRM 集成实现了个性化语音助手。
- **区块链专业知识与智能合约开发**：该工程师在**区块链技术**方面拥有深厚的专业知识，包括智能合约开发（**Solidity 和 Rust**）、去中心化应用架构以及安全的链上/链下集成。
   - 他们专注于交付可扩展的、生产就绪的 AI 和区块链系统，涵盖从模型选择和微调到后端集成和部署的全生命周期。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1438262436825137356)** (266 messages🔥🔥): 

> `GPT-5-1 em dashes, Instant model's TTFT, Model autopicking, Electroglyph's AI usage, Yukiarimo's bad and good AI` 


- **GPT-5-1 仍然带有 em dashes**：一位成员注意到 [GPT-5-1 公告页面](https://openai.com/index/gpt-5-1/) 在每个示例中都延续了使用 **em dashes** 的传统。
   - 另一位成员开玩笑地评论说，这种风格选择是“不可避免的”。
- **“Instant”模型的 TTFT**：讨论围绕 **“Instant”模型** 的首个 Token 响应时间（**TTFT**）展开，成员们认为其速度与更短的思考时间以及可能完全不同的模型架构有关。
   - 一位成员推测，目标是拥有*一个能在内部处理选择的模型，而不是让你去挑选 5 个不同的模型和 5 种思考模式*，但另一位成员表示*我不喜欢它替我决定选择哪个模型*。
- **Electroglyph 透露 AI 使用情况**：一位成员分享了他们多样化的 AI 工具箱，包括 **Hanasu** (TTS)、**Yuna** (VLM)、**Gemini 2.5 Pro** 和 **Grok 4**，用于处理从有声读物到深度研究以及*无聊的梗图*等各种任务。
   - 作为回应，另一位成员开玩笑地指出了对 AI 表示鄙视却又是活跃用户之间的矛盾：*对于一个如此讨厌 AI 的人来说，你用 AI 用得挺勤快啊 =)*。
- **Yukiarimo 将 AI 划分为“好”与“坏”**：一位成员概述了他们对 AI 的分类，认为内容生成模型（**Veo, Imagen, ElevenLabs**）、**diffusion models**、**闭源模型**和 **Agent** 本质上是“坏”的。
   - 他们表达了对 **upscaling**、**interpolation**、**小模型**以及任何**从头开始训练**的模型的偏好，并提倡能够经历*灾难性遗忘 (catastrophic forgetting)* 的模型（特别点名了 **Liquid AI**）。
- **“现实级精确”的数据捕获依然难以实现**：一位成员询问了如何以*与现实完全一致的精度*捕获并保存数据，包括用于图像的 **360 photons+lidar** 和用于音频的 **360 atoms fluctuations**。
   - 其他人指出，由于**不确定性原理 (uncertainty principle)** 以及测量和处理能力的限制，这在理论和实践上都是不可能的。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1438257737673871480)** (39 messages🔥): 

> `CPU offloading performance, GGUF models and LM Studio, vLLM for batch inference, Fine-tuning vs RAG, Qwen 2.5 VL 3B fine-tuning` 


- **CPU Offloading 极大地阻碍了性能**：成员们指出 **CPU offloading** 会显著降低性能，尽管 **MoE models** 根据配置和模型选择的不同，仍能维持可接受的速度。
   - 对于初学者，建议将 **GGUF** 模型与 **LM Studio** 配合使用以进行便捷推理，并根据系统能力选择量化版本。
- **Unsloth GGUF 超越标准量化**：**Unsloth GGUF** 模型包含了针对准确性的通用改进和性能修复。
   - 部分模型采用了 **Unsloth dynamic quantization**（动态量化），比其他量化格式实现了更高的准确度。
- **RAG vs Fine-tuning 之争**：讨论认为 Fine-tuning 和 RAG 的用途各不相同；将两者结合是检索知识/文档的理想方案，更多信息可以在 [这里](https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me) 找到。
   - 如果你想“教会”模型如何完成特定任务，Fine-tuning 可能会更好。
- **Qwen 2.5 VL Fine-tuning：困境与解决方案**：一位用户在微调 **Qwen 2.5 VL 3B** 时寻求帮助，面临训练和验证损失在尝试过拟合时仍然相似的问题，这可能是由错误的 template 引起的。
   - 社区成员指出，该频道不允许寻求付费服务，仅限与 Unsloth 相关的问题。
- **Unsloth 与 Llama.cpp/Ollama 输出差异**：一位用户报告称，在加载微调后的 **Llama3.2 3B** 模型时，尽管复制了 chat templates，**Unsloth** 的输出与 **llama.cpp/Ollama** 之间仍存在差异。
   - 有建议认为，除了 seed 之外的模型参数差异可能会影响输出。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1438546164348752044)** (3 messages): 

> `DeepSeek OCR API, Unsloth Inference, Qwen 3 1B Zero Dark Horror` 


- **DeepSeek OCR API 部署**：一位成员宣布推出一个工具，只需几个步骤即可在自己的 **DeepSeek OCR model** 上部署并运行推理。
   - 该工具可在 [GitHub](https://github.com/neosantara-xyz/deepseek-ocr-api) 上获得，并使用 **Unsloth** 进行推理，支持从 URL 或 base64 编码数据中提取图像。
- **Qwen 3 1B 获得恐怖题材处理**：**Unsloth** 现在支持 [Qwen 3 1B](https://huggingface.co/Qwen/Qwen-1_8B) 以及针对“恐怖数据集”的量化。
   - 权重已上传至 [HuggingFace](https://huggingface.co/DavidAU/Qwen3-Zero-Dark-Horror-LIGHTSPEED-1B-HRR-imatrix-GGUF)，并支持 **16/32bit training**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1438423334957088790)** (4 messages): 

> `Multi-Head Attention, Random Initialization, Modular Learning in Heads, Mixture-of-Experts` 


- **关于 Multi-Head Attention 的讨论**：一位成员询问，尽管 embedding 被平均分配，为什么 **multi-head attention** 中的不同 head 最终没有学到相同的东西。
   - 发现最常见的原因是 **random initialization**（随机初始化）产生了初始差异，而 **softmax** 夸大了这些差异。
- **神经网络中难以捉摸的模块化学习**：一位成员质疑 head 学习“相同的东西”意味着什么，并指出强制模块化（例如，一个 head 负责数学，另一个负责代码）尚未成功。
   - 他们补充说，这个问题与 **Mixture-of-Experts** 类似，其中 expert 并没有变得真正的模块化，而只是增加了另一个维度的扩展，并链接到了 [OpenAI 关于稀疏电路的研究](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)。
- **Head 学习的是属性，而非严格的概念**：一位成员同意 head 不会严格只学习一件事，但 *从抽象角度看，它们大多只学习一件事或每个 head 学习某些属性*。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1438273461675102361)** (4 messages): 

> `GPT-5.1 Release, AI Model Training, ChatGPT Group Chats` 


- ****GPT-5.1** 带着智慧与个性（Sass）到来！**: **GPT-5.1** 将于本周向所有用户推送，承诺提升智能、可靠性和对话能力，详见 [OpenAI 官方博客](https://openai.com/index/gpt-5-1/)。
   - 明天太平洋时间下午 2 点将举行 [Reddit AMA](https://redd.it/1ovkt6n/)，讨论关于 **GPT-5.1** 和自定义功能的更新。
- **透明 AI：Sparse Circuits 大获全胜**: OpenAI 开创了一种针对小型 AI 模型的新训练方法，强调提升人类理解力的内部机制，详见 [这篇博客文章](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)。
   - 该方法旨在揭开 **ChatGPT** 等语言模型复杂结构的神秘面纱。
- **ChatGPT 组队：群聊试点启动！**: 根据[此公告](https://openai.com/index/group-chats-in-chatgpt/)，**ChatGPT** 群聊功能目前正在日本、新西兰、韩国和台湾进行试点。
   - 这项新功能提供了一种在同一个 **ChatGPT** 对话中与朋友、家人或同事协作的新方式。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1438268047982067772)** (390 messages🔥🔥): 

> `GPT 5.1, Gemini 3.0, AI Video Generation, Canvas mode, Model Merging` 


- **GPT 5.1 扩展思维，助力快速学习**: 成员们发现 **GPT 5.1** 的扩展思维模式（extended thinking mode）可用于快速学习，并将其比作*一根更长的伸缩杆，能更好地调整触达范围以应对更难的主题。*
   - 几位用户讨论了 **GPT 5.1** 的表现是否优于 **GPT-4o**，一些人认为这是一个巨大的进步。
- **Gemini 3.0 悄然发布，令用户兴奋**: 一位成员指出 [Google](https://blog.google) 表示 **Gemini Drops**（*Gemini 应用的每月功能更新*）今天开始向 Pro 订阅用户推送，免费用户将在*未来几周*获得更新。
   - 报告显示 **Gemini 3.0 Pro** 在 Google 生态系统的特定部分（开发者工具、企业版/Workspace）进行了*悄然推出*，而非全面的消费者端发布，并提供了 [YouTube 链接](https://www.youtube.com/watch?v=0-CbsNB9tdk)和[其他链接](https://marketingtrending.asoworld.com)。
- **对 ChatGPT 的 Canvas 模式 Bug 的担忧**: 用户报告称，在使用 Canvas 模式时，**ChatGPT** 网站会出现崩溃和 Bug。
   - 一位用户在短对话中没有遇到问题，并成功通过 *Tempermonkey 脚本*修复了长对话的问题。
- **对免费、无限 AI 视频生成的追求仍在继续**: 一位用户疑惑为什么很少见到能无限生成视频的免费 AI，随后另一位用户提到 **Grok** 有免费 AI 视频功能，但最初的用户指的是时长不止 5 秒的那种。
   - 他们也一致认为，实现这一点需要巨大的算力和处理能力，这就是为什么 Pro 版才提供访问权限。
- **ChatGPT 编写 Roblox ESP 和 Aimbot 挂机脚本**: 一位用户表示他们使用 **ChatGPT** 编写 *Roblox ESP 和 Aimbot 挂机脚本，同样也用于 CS2，而且效果很好*。
   - 这引发了复杂的反应，一些人认为这很疯狂，并询问这是否违反了 TOS（服务条款）。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1438272275446108201)** (87 条消息🔥🔥): 

> `GPT 5.1, 用于创意写作的 GPT 4.1, 印度版 Sora 2, 图像生成的新护栏, OpenAI operator` 


- **GPT 5.1 引发重度用户不满**：用户对 **GPT 5.1** 表示失望，将其描述为一种*退步*，认为其*废话连篇*，充斥着多余的散文式修辞，并且在处理 [custom instructions](https://platform.openai.com/docs/gpt-best-practices) 时表现出一种*挑衅的态度*。
   - 一些用户觉得它是科学仪器的“玩具版”，而另一些人则认为它非常出色，令人耳目一新。
- **故事弧线断裂：GPT 5.1 过度总结并重复场景**：用户报告称 **GPT 5.1** *确实坏了*，它会过度总结内容，并在故事弧线中重复场景，一位用户表示这毁了他的故事。
   - 由于这些问题，一些用户正回退到使用 **GPT 4.1** 进行创作，并哀叹这些变化破坏了 AI，使其无法遵循格式或 Prompt。
- **模型更新收紧限制**：新更新收紧了关于模型如何引用存储记忆、概括与重写、线程捆绑、使用带有变体拼写的角色名称以及处理新场景连贯性的限制，导致了内容重复。
   - 模型会跳回默认安全模式，并试图重新陈述或重新洗牌内容。
- **图像生成护栏过度**：用户发现新的 [图像生成护栏 (guardrails)](https://openai.com/policies/usage-policies) 过于严苛，导致他们无法再将其用于简单的描绘。
   - “不能回过头去编辑我的文本真是太‘棒’了！没有任何新功能却丢掉了一半旧功能的感觉真‘好’！”


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1438590659069083728)** (2 条消息): 

> `Polaris Alpha 弃用, GPT-5.1 发布, GPT-5.1 Chat (Instant), GPT-5.1-Codex, GPT-5.1-Codex-Mini` 


- **Polaris Alpha 谢幕；GPT-5.1 登场**：用户一直在测试的 "Polaris Alpha" 模型是 **OpenAI GPT-5.1 不带推理能力的早期版本**，很快将被弃用。
   - 取而代之的是速度更快、Token 效率更高且具备自适应推理和更强编程能力的 **GPT-5.1**，详见这篇 [OpenAI 新闻稿](https://openai.com/index/gpt-5-1-for-developers/)。
- **GPT-5.1 首秀，系列模型上线**：OpenRouter 已推出 **另外三款 GPT-5.1 模型**：[GPT-5.1 Chat](https://openrouter.ai/openai/gpt-5.1-chat)（即 ChatGPT 中的 Instant）、[GPT-5.1-Codex](https://openrouter.ai/openai/gpt-5.1-codex) 和 [GPT-5.1-Codex-Mini](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113)。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1438257135573270682)** (437 条消息🔥🔥🔥): 

> `OpenRouter 隐私设置, API 速率限制, GPT-5.1 发布与性能, AI SDK 采用, 结构化输出` 


- **隐私开关影响模型可用性**：一位用户通过在 [OpenRouter 隐私设置](https://openrouter.ai/settings/privacy) 中开启第三个隐私开关——*“启用可能在输入上进行训练/发布 Prompt 的免费端点”*，解决了一个错误。
- **遭遇 API 速率限制困扰**：用户报告 API 频繁出现 **Error 429**（速率限制）问题，尤其是 `anthropic/claude-sonnet-4` 和 `anthropic/claude-sonnet-4.5`，这表明上游供应商的供应无法满足需求。
   - 一些用户还遇到了 Cloudflare 错误和超时，但 [OpenRouter 状态页面](https://status.openrouter.ai/) 显示没有异常。
- **GPT-5.1 到来，引发辩论**：**GPT 5.1** 正在 ChatGPT 上逐步推出，但目前还没有 API 访问权限。
   - 早期基准测试显示，为了极小的改进却产生了近 **2倍的 Token 输出**，引发了对性价比的担忧：“我们要的是少点思考，而不是更多”。
- **讨论 OpenRouter AI SDK 的采用**：成员们讨论了 [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk)（官方 SDK）以及在 3.chat 等平台中更广泛使用 AI SDK 的情况。
   - 一些用户仍然偏好原始 HTTP 请求或 `curl`，并强调 OpenRouter “不知为何仍然是处理这些事务的最佳平台”。
- **结构化输出原理解析**：`json_object` 仅保证返回有效的 JSON 对象，而使用 **Structured Outputs** 时，`json_schema` 会强制要求符合你定义的 Schema。然而，“不幸的是，支持结构化输出的供应商并不多，所以请务必检查你所使用模型的供应商”。
   - 用户可以通过 OpenRouter 网站上的模型概览，展开供应商列表来查看具体细节。


  

---

### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1438608118660731011)** (4 messages): 

> `` 


- **没有值得注意的新模型新闻**：OpenRouter Discord 频道中没有关于新模型的重大讨论。
- **新模型频道保持沉默**：OpenRouter 的 'new-models' 频道似乎处于非活跃状态，没有可总结的消息。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1438500085842444369)** (19 messages🔥): 

> `React Slop, React Compiler, Web Search Tools` 


- **聊天室遭遇 React Slop 问题**：一名成员报告称，在聊天室中附加长文件会导致浏览器卡顿，且编辑消息会触发整个聊天室的重新渲染，另一名成员建议这是 React Slop 的典型案例。
   - 建议将所有内容包装在 `useMemo` 中以缓解该问题。
- **React Compiler 太棒了！**：一名成员提到他们开始使用 **React Compiler**，它解决了 *React Slop* 问题，并表示“它太牛了”。
   - 另一名成员开玩笑地质疑了这一说法的严肃性。
- **Google 和 XAI Web Search 工具**：一名成员询问如何将来自 **Google** 和 **XAI** 的原生 Web Search 工具集成到平台中，并链接到了 [Web Search 功能文档](https://openrouter.ai/docs/features/web-search)。
   - 另一名成员报告称，他们正在开发基于 **Gemini** 的 **Web Search**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1438650527972855869)** (10 messages🔥): 

> `CUDA project for Monte Carlo PSF calculation, Why learn GPU programming if Python wrappers exist?, Data center operations with inference and Bitcoin mining, GPU compiler bug with __restrict__ qualifier` 


- **用于 PSF 计算的 CUDA Monte Carlo：可行吗？**：一名学生询问作为期末课程项目，在 **CUDA** 中实现 **Monte Carlo 方法**以计算电子散射中的**点扩散函数 (PSF)** 是否可行。
   - 该学生不确定这是否是一个合适的项目，因为他们在 **CUDA** 和晶圆**光刻 (Lithography)** 方面的专业知识有限。
- **GPU 编程：Python 封装是否让直接学习变得过时？**：一名成员质疑直接学习 **GPU 编程**的必要性，因为像 **PyTorch** 这样抽象掉大部分底层复杂性的 **Python 封装**非常流行。
   - 讨论集中在通过 **Python 库**使用**预编译 GPU 代码**的能力是否否定了理解底层 **GPU 编程**的需要，尽管许多人同意仍然需要有人来“制作”这些库，且新颖的应用能从底层专业知识中受益。
- **数据中心协同：推理与比特币挖矿**：一名成员正在撰写一篇关于**系统动力学**的论文，并寻求任何在结合**推理**和**比特币挖矿**的**数据中心规模运营**方面有经验的人的意见。
   - 他们希望找到先前的案例或见解，以了解这两个计算密集型工作负载之间的潜在协同作用和冲突。
- **Restrict 限定符：编译器 Bug 还是未定义行为？**：一名成员分享了一个涉及 `__restrict__` 限定符的潜在 **GPU 编译器 Bug** 的有趣 [示例](https://godbolt.org/z/ad98nYdrf)，并征求关于它是真正的 Bug 还是仅仅是**未定义行为 (UB)** 的意见。
   - 代码示例包括 **PTX** 和 **SASS** 输出，邀请对编译器处理内存别名约束的方式进行更深入的分析。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1438752985692770325)** (1 messages): 

> `NaN to Zero, tl.maximum trap` 


- **NaN 转零的代码注意事项**：`tl.maximum(probability, 0)` 被用作 **NaN 转零**，但它导致了精度下降。
   - 使用 `tl.where(p == p, p, 0)` 效果更好，尽管原因尚不清楚。
- **tl.maximum 陷阱**：用户报告称使用 `tl.maximum(probability, 0)` 导致其应用中出现了一些精度下降。
   - 他们发现 `tl.where(p == p, p, 0)` 运行良好。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1438513507271311511)** (4 messages): 

> `Sass Registers Reusing, PyNvVideoCodec on AWS g4dn, Video Upscaling with CUDA and PyTorch, RealESRGan model, RTX4060 optimizations` 


- **Sass 提升了几个百分点的速度**：一篇关于 [**Sass** 寄存器复用](https://redplait.blogspot.com/2025/11/sass-registers-reusing.html)以获得几个百分点速度提升的博客文章。
- **使用 PyNvVideoCodec 解码视频**：一名成员尝试在 **AWS g4dn** 机器（配备 **NVIDIA T4** GPU）上使用 **PyNvVideoCodec** 进行视频编解码，并询问如何配置环境的步骤。
   - 他们表示，仅仅是配置环境就非常痛苦。
- **使用 RealESRGan 和 RTX 4060 进行视频超分辨率速度缓慢**：一名成员尝试在 **RTX4060** (**12GB** VRAM) 上使用 **CUDA** 和 **PyTorch** 配合 **RealESRGan** 模型进行视频超分辨率以提升画面质量，但 FPS 非常低（0.5fps）。
   - 他们询问如何将 FPS 提高到至少 50。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1438706098088706069)** (2 messages): 

> `cuBLAS FP64 Emulation, torch.mm Performance, Custom C++ Operator for cuBLAS, ATen cuBLAS GEMM Call Tracing` 


- **FP64 仿真在某些硬件上表现惊人**：一名成员正在探索 **CUDA 13.0u2** 中的 **cuBLAS FP64 仿真**，并观察到在某些输入尺寸下，默认的 **torch.mm** 在其硬件上表现出*极佳的性能*（超过 **580% 的峰值 FP64 吞吐量**），详见 [Learn-By-Doing-Torchinductor-DeviceAssert](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-DeviceAssert/)。
   - 他们希望强制使用 **cuBLAS kernels**，以调查在调度器选择 **CUTLASS kernels** 的其他输入尺寸下，性能是否会更好。
- **自定义算子的困境与 Torch 类似**：该成员根据 [NVIDIA 的 cuBLAS 仿真示例](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLAS/Emulation)创建了一个 **C++ 自定义算子**，但性能与未仿真的 **torch.mm** 完全一致。
   - 他们怀疑自己调用 **cuBLAS dgemm/gemmEx** 的方式不正确。
- **ATen 揭示 cuBLAS 的秘密**：由于 torch.mm 成功调用了 cuBLAS 并应用了仿真，该成员正尝试追踪 **ATen** 中调用 **cuBLAS GEMM kernels** 的确切方式。
   - 他们使用了 `TORCH_SHOW_DISPATCH_TRACE=1` 并发现了 `op=[aten::mm.out], key=[CUDA]`，这可能是 [at::mm_out](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L926)，但不确定如何进一步找到具体的 cuBLAS 调用。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1438704349894283426)** (1 messages): 

> `Paulius Micikevicius, GPU efficiency, low bit dtypes, sparsity, numerical stability` 


- **Micikevicius 加入并就 GPU 效率发表演讲**：以研究 **低比特数据类型 (low bit dtypes)** 和 **稀疏性 (sparsity)** 提升 **GPU 效率**而闻名的 Paulius Micikevicius 将进行一场演讲，由另一名成员共同主持，深入探讨 **浮点数 (floats)**、**数值稳定性**、**确定性**、**量化**和**稀疏性**。
   - 在 NVIDIA 长期任职后，他最近加入了与主持人相同的公司；更多信息可以在[这段视频](https://www.youtube.com/watch?v=3qNZvvlwcCI)中找到。
- **演讲计划缩减，将于 1 月恢复**：由于主持人正在休育儿假，演讲安排将会减少。
   - 计划应在 **1 月** 全面恢复。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1438779530729750599)** (1 messages): 

> `NCU, Cloud vendors, Semianalysis` 


- **NCU 在云厂商评估中占据重要地位**：据 [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) 报道，云厂商现在正根据其对 **NCU** (NVIDIA Compute Unified Device Architecture) 的支持程度进行评分。
- **Semianalysis 报道云厂商分级**：[Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) 报道称，云厂商现在根据其对 **NCU** 的支持情况进行分级。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1438590563954856170)** (2 messages): 

> `Voltage Park, SWE positions, AI Factory Platform, CUDA kernel engineer` 


- **Voltage Park 正在招聘工程师**：**Voltage Park** 正在寻找 **SWE** 来协助基础设施工程、软件工程以及专注于 AI/ML 的软件工程。
   - 他们对远程办公友好（优先考虑旧金山和西雅图），并且正在构建一个 **AI Factory 平台**；请查看他们的 [招聘页面](https://www.voltagepark.com/careers)。
- **兼职 CUDA Kernel 工程师需求**：一位成员正在寻找一名 **CUDA kernel 工程师** 进行一些兼职工作。
   - 他们提供 **每小时 200 美元** 的报酬。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1438507649783038043)** (2 messages): 

> `Parallel Programming, Beginner Guidance` 


- **用户寻求并行编程入门指导**：一位用户表示在如何开始并行编程方面感到迷茫。
- **指导位置**：另一位用户指引他们前往频道 **#1198358627594023014**，作为一个可能的起点。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1438272647942242447)** (5 messages): 

> `NVFP4 Datatype Optimization, AI Coding Agent Workflow, Rust LLM Router` 


- **NVIDIA 黑客松助力 NVFP4 优化**：GPU Mode 和 NVIDIA 正在举办一场黑客松，旨在优化 **Blackwell GPU** 架构上 **NVFP4** 数据类型的负载，第一个挑战集中在 GEMV（矩阵向量乘法）。
   - 一篇博客文章解释了参考 kernel 并涉及了新的数据格式，推荐使用 **CuTeDSL**，因为它能在保持生产力的同时接近硬件底层，并由 Datacrunch 提供 **B200 GPU** 的访问权限；详情请参阅 [博客文章](https://veitner.bearblog.dev/nvfp4-gemv/)。
- **AI Coding Agent 工作流已开源**：一个旨在交付高质量代码的 AI coding agent 设置已开源，其特点包括技能系统、114+ 个子 Agent 以及用于处理复杂功能的执行规划框架。
   - 该设置旨在增强现有的 AI 编程工具，可以在 [GitHub](https://github.com/flora131/agent-instructions) 上找到，详细解释请见 [这篇博客文章](https://alexlavaee.me/blog/ai-coding-infrastructure/)。
- **为基于 Rust 的 GPL 协议 LLM Router 寻找合作者**：一位成员正在为基于 Rust 的 GPL 协议 LLM Router 寻找合作者。
   - 仓库地址：[GitHub](https://github.com/awdemos/merlin)。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1438263286566093012)** (2 messages): 

> `Nvidia competition submissions, Discord bot, CLI, Site` 


- **Nvidia 竞赛支持通过 CLI、Discord 和网站提交**：**Nvidia 竞赛** 的作品提交支持通过 **Discord**、**网站** 和 **CLI** 进行。
- **CLI 是 Nvidia 竞赛提交最受欢迎的途径**：提交 **Nvidia 竞赛** 作品最流行的方法可能是使用 **CLI**。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1438647696486563944)** (2 messages): 

> `HipKittens, Quark Start` 


- **HipKittens 现在支持 gqa_backward**：一位贡献者提交了一个 [Pull Request](https://github.com/HazyResearch/HipKittens/pull/4)，改进了 **Makefile** 以便按照 **Quark Start** 指南轻松执行 **gqa_backward 示例**。
   - 作者表达了感谢，并承诺会评估待处理的 Pull Request。
- **对增强功能贡献表示感谢**：一位成员对 **HipKittens** 项目的新贡献表示赞赏。
   - 该用户特别感谢了贡献者在 **Makefile** 更新方面的工作，并承诺在不久的将来审查该贡献者的 Pull Request。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1438430872889069588)** (1 messages): 

> `Inference optimization, GPU resources (Hopper, Ampere), University resources, Smaller models fitting on GPU VRAM` 


- **在 Hopper 或 Ampere 上优化推理**：用户可以探索使用 **Hopper** 或 **Ampere** 等 GPU 进行 **推理优化**。
   - 大学通常通过教授提供这些资源。
- **更小的模型，更小的 VRAM**：如果模型能 **适配 GPU 的 VRAM**，则可以运行更小的模型。
   - 这是一份关于将对话带入错误频道的道歉。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1438270167103569920)** (52 messages🔥): 

> `nvfp4_gemv Leaderboard Updates, grayscale_v2 Leaderboard Updates, NVIDIA performance improvements, H100 grayscale performance, Personal Bests on various GPUs` 


- **NVIDIA 上的 GEMV 速度闪击战**：提交到 `nvfp4_gemv` 排行榜的多个结果显示了 **NVIDIA** 上的执行时间改进，时间从 **3.19 ms** 降至 **24.7 µs**。
- **Grayscale v2 GPU 挑战赛**：提交到 `grayscale_v2` 排行榜的结果突出了不同 GPU 的性能，其中一个提交在 **L4** 上达到 **27.5 ms**（第 7 名），另一个在 **H100** 上达到 **12.9 ms**（第 7 名）。
- **B200 突破 Grayscale 障碍**：`grayscale_v2` 排行榜见证了在 **B200** 上取得的新个人最佳成绩，时间降至 **6.69 ms**。
- **A100 出色完成 Grayscale 任务**：`grayscale_v2` 的提交显示了 **A100** 上的个人最佳成绩，速度达到 **20.4 ms**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1438607558980931627)** (1 messages): 

> `Network failover test` 


- **即将进行网络故障转移测试**：数据中心将于今天 **PT 时间下午 2:00–3:00** 进行计划内的 **网络故障转移测试**。
   - 在此窗口期间，节点的连接可能会受到临时影响，他们对造成的不便表示歉意。
- **数据中心维护**：计划维护可能会导致临时连接问题。
   - 建议用户在 PT 时间下午 2:00-3:00 期间做好中断准备。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1438567696345071689)** (13 messages🔥): 

> `FMTK Extension in Cursor, Factorio Mod Debugger, Agent Interaction Mod, Sumneko Language Server` 


- **Factorio Mod 调试器现身**：一位用户分享了一个 **Factorio mod 调试器** GitHub 仓库链接：[justarandomgeek/vscode-factoriomod-debug](https://github.com/justarandomgeek/vscode-factoriomod-debug)。
   - 另一位用户表示兴奋，提到他们之前只使用 **Sumneko** 和 **FMTK** 作为 Language Server，并不知道有 **调试器和分析器 (profiler)**。
- **Agent 交互 Mod 即将到来**：一位用户正在开发一个 **Agent 交互 mod**，该 mod 遵循 mod 编写的最佳实践，旨在更简单地与 FLE (Factorio Learning Environment) 集成。
   - 该用户将 **Agent** 描述为一种 *字符类型的 luaentity，可由外部系统 + LLM 操作*，目前已接近发布状态。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1438551236109537281)** (2 messages): 

> `CUDA `copy` behavior, GEMV kernel pipelining, GMEM -> SMEM -> RMEM data transfer, Numerical result discrepancy` 


- **`copy` CUDA 线程行为解释**：一位成员询问，如果调用 `copy(copy_a, ...)` 时使用的线程数多于 `size(copy_a)`，是否会导致超出尺寸的线程不发出 copy 指令。
   - 隐含的澄清是，当创建 `size(copy_a)` = 32 的 `copy_a` 时，将仅使用 32 个线程。
- **GEMV Kernel 流水线导致错误结果**：一位成员报告称，在为 GEMV 竞赛 Kernel 实现使用 **GMEM -> SMEM -> RMEM** 数据传输的流水线时，出现了错误的数值结果。
   - 该问题发生在将矩阵 **A** 从 **GMEM** 复制到 **SMEM** 期间（以及通过 *autovec_copy* 从 **SMEM** 传输到 **RMEM** 期间）。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1438281045039386654)** (2 messages): 

> `Submission location` 


- **GPU MODE 提交在多处受理**：用户询问了提交 GPU MODE 作品的正确位置，并提供了 [排行榜链接](https://www.gpumode.com/v2/leaderboard/595?tab=submission)。
- **备选提交点得到澄清**：另一位用户澄清说，**Discord 和网站** 都是可以接受的提交点。


  

---

### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1438319133174403102)** (39 messages🔥): 

> `Helion BC 兼容性与更新、使用 configs 的 Helion Autotuning、Helion 配置选项及访问最快配置、Helion 解释模式的速度` 


- **Helion 声称支持向后兼容**：Helion 将支持 **BC 兼容**（例如从 **0.2** 到 **0.2.1**），索引已从单个值更新为列表，因为每个 load/store 都可以独立优化以获得性能提升，但仍支持单值输入；**0.2.2** 版本刚刚发布，计划每隔一两周发布一次次要版本以保持 pypi 包的更新，详见 [issue 164](https://github.com/pytorch/helion/issues/164)。
   - 更新为列表形式使得每个 load/store 操作能够独立优化，在增强性能的同时保持了与单值输入的兼容性。
- **Helion 现在支持使用 Configs 进行 Autotune**：你可以传递 `configs=` 而不是 `config=`，这与 **Triton** 的 **autotune** 类似。
   - `@helion.kernel(configs=[a, b, c])` 装饰器将运行 a、b 和 c，并选择最快的配置，其行为与 **Triton autotuner** 相似。
- **在 Helion 中更快速地访问配置**：`helion_rms_norm_fwd.bind((x, w, eps))._config` 将帮助你实现这一目标。
   - Helion 允许使用一组配置进行程序化 autotuning，在“主要形状集”上运行 Helion autotuning 以获得一组配置，然后在次要配置集上对所有配置进行 autotuning；访问 `_config` 属性对此仍然有效，可以将第一组的 `_config` 传递给第二组。
- **Helion 的解释模式（Interpret Mode）速度惊人**：Helion 的解释模式被认为出奇地快，它使用 eager 模式的 **PyTorch** 运行整个代码，就好像没有 tile 一样，这归功于 tile 的抽象实现了性能可移植性。
   - 相比之下，**Triton** 的解释模式被认为极其缓慢，因为运行的 tile 大小通常为 32，这突显了 Helion 在解释执行性能方面的优势。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1438262945095221368)** (283 messages🔥🔥): 

> `NVIDIA 竞赛指南、Cutlass vs CUDA、Blackwell 优化、NVF4 数据转换` 


- **GPU 新手获得竞赛指导**：使用 **RTX 3060** GPU 的新参赛者收到了指导，建议查看 [问题仓库](https://github.com/gpu-mode/reference-kernels) 以及通过 [Discord Cluster Manager 文档](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/) 查看提交说明。
   - 解决方案必须以 Python 提交，但可以使用带有 `load_inline` 的 **CUDA/CUTLASS**。
- **CUDA vs Cutlass**：成员们讨论了使用 **CUTLASS** 优于 **CUDA** 的优缺点，重点介绍了在 Python 文件中使用内联加载的 **CUDA C++**。
   - 分享了一篇 [Modular 文章](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)，讨论了针对生产环境形状的优化，包括选择最佳参数，如 **MMA shape**、**pipeline stages** 和 **block swizzling patterns**。
- **Blackwell 架构需要 MMA 专业知识**：讨论涉及在 **Blackwell** 上优化矩阵乘法，引用了 [Modular.com 的博客文章](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)，强调需要最小化每个 **SM** 的工作负载。
   - 有人指出 **B200** 拥有 **148 个 SM**，这给 grid 划分带来了挑战，并对为特定问题形状设计合理的 grid 表示了困扰。
- **NVF4 数据需要布局转换**：成员们辩论了 **TMEM** 描述符格式以及需要 **TMA** 将数据转换为 tensor cores 的 *UMMA 规范布局*。
   - 澄清了 **TMA** 将数据从行优先（row major）重新排序为 tensor core 的不同内存布局，并添加了 padding；显然，tensor core 要求每列 8 个元素在内存中是连续的。
- **Github Action 上的 CUDA 环境**：成员们发现 **Github Action 运行环境** 运行在 **CUDA 13.0.88** 上，仅将 `-gencode=arch=compute_80,code=sm_80` 和 `-gencode=arch=compute_100,code=sm_100` 传递给 nvcc，导致了目标错误（target errors）。
   - 一些成员放弃了使用 **LLM** 来编写代码，因为它们不知道如何使用 **TMA** 将 **FP4** 扩展到那些复杂的布局（cursed layouts）。


  

---

### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1438266982104305706)** (6 条消息): 

> `RLinf, Qwen3-VL VLA-adapter 训练, UnifoLM-WMA-0, LIBERO-PRO 数据集局限性` 


- **RLinf 仓库准备中**：一名成员正准备发布 [RLinf Basic 仓库](https://github.com/RLinf/RLinfBasic)，并计划通宵运行 **Qwen3-VL VLA-adapter 训练**。
   - 该成员还打算清理仓库，并在次日对 **LIBERO** 进行评估。
- **Qwen3-VL GPU 统计数据**：一名成员分享了他们正在使用单张 **A6000**，在 **Qwen3-VL-2B-Instruct** 上使用两个 **256x256** 图像输入来训练 adapter。
   - 在 Batch Size 为 **48** 且使用 **bf16** 的情况下，该训练正以 naive 训练设置运行。
- **宇树科技（Unitree）的 UnifoLM-WMA-0 引起关注**：一名成员指出 [Unitree 的 UnifoLM-WMA-0](https://github.com/unitreerobotics/unifolm-world-model-action) 在世界模型动作（world model action）方面看起来非常有趣。
   - 未提供更多细节。
- **LIBERO-PRO 揭示数据集缺陷**：该成员还提到 [LIBERO-PRO](https://arxiv.org/abs/2510.03827v1) 展示了原始 **LIBERO** 数据集的一些局限性。
   - 未提供更多细节。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1438260408459853995)** (397 条消息🔥🔥): 

> `GPT 5.1 Codex 发布, Cursor Auto Mode 性能下降, Cursor 自定义命令, Cursor 内存占用, Tailwind CSS v4 实现` 


- **Cursor 推出 GPT-5.1-Codex，工程师们感到兴奋**：成员们报告 **GPT-5.1-Codex** 正在推出，一些人在 [最新的 codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) 中发现了它，并注意到它与 [Windsurf](https://www.windsurf.ai/) 的集成。
   - 然而，情况有些混乱，因为部分用户尚未收到更新提示，而另一些人发现它在 OpenAI API 正式发布前已在 Codex 中可用，这表明是分阶段发布的。
- **Auto Mode 的性能问题困扰高级用户**：用户报告 **Auto mode 性能** 明显下降，有人表示 *20 分钟前 Auto mode 还在使用另一个更快的模型，现在它使用 gpt 5 codex，速度变慢了，甚至连无推理模型都能轻松完成的文件编辑都无法完成*。
   - 有人认为这可能是由于 **Auto mode** 在 **5.1 发布**后默认连接到可能过载的 OpenAI 服务器，而另一些人猜测 Cursor 使用的是具有高思考（thinking）需求的 GPT-5 版本。
- **自定义命令降低 Token 成本**：成员们讨论了使用**自定义命令**来自动化任务（例如在代码更改后运行测试），一位用户分享了通过 **CTRL + K** 创建自定义命令以触发特定指令的方法。
   - 一名成员建议通过执行特定命令来确保自动触发测试，这些命令可以通过 [Cursor 设置中的文档部分](https://cursor.com/docs/customcommands)进行标记，以提高 Token 效率。
- **对 Cursor 内存（Memory）功能的疑虑显现**：Cursor 的 **memory 功能** 引起了褒贬不一的反应；一位运行在传统隐私模式下的用户犹豫是否要禁用它。
   - 一位用户描述说 *memories 通常存在于你的 profile 中并且需要处于激活状态，如果你处于隐私模式，这种 tool calling 将会失败*，因此如果你想充分利用 Cursor，这是一个需要关注的功能。
- **序列化错误阻碍终端任务**：用户报告在使用终端命令时出现 *Serialization error*，这会导致对话中断直到重启 Cursor。一位用户报告称，问题的根源在于文件路径中包含空格的命令。
   - 成员们将问题定位在将终端输出喂给 LLM 时的序列化错误，并分享了 [Serialization Error 论坛帖子](https://forum.cursor.com/t/serialization-error/124671)的链接。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1438644266577559613)** (1 条消息): 

> `API 降价, Hermes 4 70B, Hermes 4 405B` 


- **Nous Research 大幅削减 API 价格！**：Nous Research 宣布其 **Hermes 4 70B** 和 **Hermes 4 405B** API **降价 70%**。
   - 详情和注册请访问 [Nous Research Portal](https://portal.nousresearch.com/) 和 [X](https://x.com/NousResearch/status/1989077400957911394)。
- **Hermes 模型的 API 访问**：该 API 提供对 **Hermes 4 70B** 和 **Hermes 4 405B** 模型的访问。
   - 通过 [Nous Research Portal](https://portal.nousresearch.com/) 注册即可开始使用。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1438275995999141990)** (75 messages🔥🔥): 

> `Nous Chat 中的 GGUF 文件，GPT5.1 发布，Schizomax 模型，Hermes4 代码模型` 


- ****GGUF 文件在 Nous Chat 中无法运行****：一位用户询问如何将 **GGUF 文件**导入 Nous Chat 以使用模型，但另一位用户澄清该功能目前尚不可用。
- ****OpenAI 意外发布 GPT-5.1****：一位用户分享了 OpenAI 网站上 **GPT-5.1** 发布页面的链接，[GPT-5.1](https://openai.com/index/gpt-5-1/)。
- ****Schizomax 模型在社区引起热议！****：一名成员建议需要一个 **schizomax 模型**，其他人表示同意，并指出 **Grok** 和 **Nous 模型** 很少拒绝请求，除非涉及极端内容。
   - 他们对 OpenAI 的 **GPT 模型**由于健康检查（wellness checks）和企业影响而变得越来越受限表示沮丧。
- ****Hermes4 潜入 Cline？！****：一位用户指出 **Hermes4** 现在已成为 Cline 中的代码模型，并附上了 Prompt 的截图。
   - 这一观察引发了关于 **Hermes4** 不断进化的能力和潜在应用的讨论。
- ****达成 100 万次下载里程碑！****：一位用户宣布他们已达到 **100 万次下载**。
   - 另一位用户表示祝贺并与之共同庆祝，进一步强调了这一成就的重要性。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1438277169834098849)** (79 messages🔥🔥): 

> `GGUF 文件，Nous Chat，Ollama，transformers.js，pgvector` 


- **探索如何将 GGUF 文件导入 Nous Chat**：一名成员询问如何将 **GGUF** 文件导入 **Nous Chat** 作为模型使用，但目前该网站并不直接支持此功能。
- **使用 llama.cpp 或 Ollama 在本地运行 GGUF**：成员们建议使用 [llama.cpp](https://huggingface.co/docs/hub/en/gguf-llamacpp) 或 [Ollama](https://ollama.com/) 等工具在本地运行 **GGUF** 文件，以便进行更简单的设置。
- **小型 AI 模型测试**：在安装 **Ollama** 后，一名成员建议通过运行命令 `ollama run hf.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF:Q4_K_M` 来测试小型模型是否正常工作。虽然这不是最新的模型，但因其体积小，非常适合作为测试。
- **改进 transformers.js 嵌入 (embeddings)**：一名成员在使用 **transformers.js** 为法律条例生成嵌入，并使用 **pgvector** 作为数据库时，即使使用了深度分块（deep chunking）和面包屑上下文（breadcrumb context），获得的评分仍然较低。
   - 该成员询问是否有任何设置或优化选项可以检查，以提高整体搜索质量。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

teknium: https://fxtwitter.com/historygpt/status/1977895243195334826?s=46
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1438268442070487070)** (22 条消息🔥): 

> `Lakh MIDI Dataset Cleaning, Hugging Face Dataset Uploads, Wikipedia Dataset Cleaning and Structuring, JSON vs JSONL for Datasets, FineWiki Comparison` 


- **Lakh MIDI Dataset 获得深度清洗**：一位用户分享称他们完整清洗并整理了整个 **Lakh MIDI Dataset**，生成了一个包含超过 **44,000 个条目**的干净、结构化的 **JSON 文件**，解析完整且一致，并提议免费分享。
   - 另一位用户建议将其上传到 [Hugging Face](https://huggingface.co/)，原作者表示同意，并邀请大家共同协作和改进该数据集。
- **Wikipedia 数据集完成大扫除，托管于 HuggingFace**：一位用户将包含超过 **2,700,000 个文件**的 **JSON 格式**清洗版 **French Wikipedia DB** 上传到了 [Hugging Face](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json)，并提到他们正在清洗英文版本。
   - 该用户指出，数据清洗不仅仅针对纯文本，还包括对 *templates, tables, html, refs* 的清洗，同时保留了 infobox 信息和链接。
- **JSONL 格式在数据讨论中脱颖而出**：在关于数据集格式的讨论中，一位用户建议对纯文本数据使用 **JSONL/NDJSON**，而不是 tar 文件，因为 tar 头部会带来额外开销。
   - 他们认为 **JSONL** 允许逐行读取，使处理更简单，相比之下 tar 文件需要处理每个头部。
- **FineWiki 在新的清洗方式面前相形见绌**：一位用户询问了新的 Wikipedia 数据集相比 **FineWiki** 和其他过滤后的 Wikipedia 变体有哪些改进。
   - 数据集创建者回应称，他们的流程清洗了更多的 *templates, tables, HTML, and refs*，并保留了 infoboxes 和链接等结构化元素，这与 **FineWiki** 侧重于纯文本的做法不同。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1438260121196036178)** (95 条消息🔥🔥): 

> `NVIDIA's Dataset License, Synthetic Data, Tokenizer Paper, Anthropic's Policies on China` 


- **研究人员解耦记忆化**：研究人员发布了[一篇论文](https://arxiv.org/abs/2510.24256)，关于在模型权重中解耦记忆化（memorization）。
   - 讨论集中在该方法在技术上的巧妙之处。
- **应对 NVIDIA 棘手的许可证**：社区讨论了 **NVIDIA 数据集许可证**中的限制，特别是关于训练、评估和结果公开分享方面的限制，一位成员链接到了一个表达担忧的 [X 帖子](https://x.com/goodfireai/status/1986495330201051246)。
   - 主要担忧围绕着一个允许 **NVIDIA** 随时终止许可证的条款，这可能会使任何已授予的许可失效，以及法律上的模糊性导致难以确定具体允许哪些操作。
- **合成数据（SYNTH Data）前沿**：成员们讨论了在梳理这些数据时，文档末尾出现的 **QA pairs**（问答对）。
   - 数据中的 QA 对很可能是**合成的**，是通过使用 **nemotron-cch** ([https://arxiv.org/abs/2511.08923v1](https://arxiv.org/abs/2511.08923v1)) 配合 QA 提示词对内容进行改写生成的；一些人对来自 **PleIAs** 的数据质量表示怀疑（并提到了一个新的数据前沿 [博客文章](https://pleias.fr/blog/blogsynth-the-new-data-frontier)）。
- **Tokenizer 论文**：社区中出现了一篇新的 **tokenizer 论文**供参考。
   - 该[论文](https://arxiv.org/abs/2511.09709)被分享，认为可能很有趣。
- **Anthropic 被指责散布恐惧**：一位成员指责 [Anthropic](https://x.com/AnthropicAI/status/1989033793190277618) 通过**散布恐惧**来获取战略优势，特别是针对**非美国**和**中国实验室**。
   - 有人对 **Anthropic 的数据隐私实践**表示担忧，暗示他们可能优先考虑安全性而非隐私，并对 **Anthropic CEO 对中国的看法**提出了质疑。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 条消息): 

burnytech: https://openai.com/index/understanding-neural-networks-through-sparse-circuits/
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1438336536566435871)** (12 messages🔥): 

> `lm harness, unitxt parameters, xsum dataset` 


- **lm harness 中的数据集**：一位成员询问了除了 `scrolls`、`megsum` 和 `noticia` 之外，**lm harness** 中还有哪些可用的数据集。
   - 团队确认 harness 包含多个子任务，部分位于 `darija/catalan/spanish_bench` 中，并且其底层使用了 **unitxt**。
- **harness 中确认包含 `xsum` 数据集**：一位成员询问 `xsum` 数据集（[https://aclanthology.org/D18-1206/](https://aclanthology.org/D18-1206/)）是否包含在 harness 中。
   - 另一位成员确认确实使用了 `xsum`，并且它在内部被传递给 **unitxt**，并指向了相关的 [unitxt 文件](https://github.com/IBM/unitxt/blob/800b2bad7f6cf794bde4e8fd8f4cbd0461e5940c/prepare/cards/xsum.py#L11)。
- **Unitxt 参数**：一位成员寻求关于在评估模型时如何传递 **unitxt** 特定参数（例如 `--template templates.summarization.abstractive.full` 和 `--max_test_instances`）的指导。
   - 团队建议修改 [任务 YAML 文件](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/unitxt/xsum.yaml) 来配置这些参数。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1438266540616323153)** (42 messages🔥): 

> `Pub/Private Members in Mojo, Modular Tech Stack for Regression Models, Mojo vs MAX, WebGPU in Mojo, FFI C in Mojo` 


- **请保护隐私：公有/私有成员即将到来（最终）**：**Mojo** 最终可能会支持 **公有和私有成员/方法**，类似于其他语言，但在出现打破封装的“逃生舱（escape hatch）”之前不会实现。目前，Mojo 使用 Python 的下划线约定来暗示私有性。
   - 目前的方法是使用 **Python 的下划线约定** 来表示成员应被视为私有的。
- **堆叠 Modular：回归模型现状检查**：要利用 **Modular 技术栈** 构建回归模型，目前需要使用 MAX 构建解析器、数据可视化库以及训练的反向传播；目前最快的方法是使用 Torch 进行训练，然后使用 MAX 进行推理。
   - Mojo 最适合 **计算密集型任务**，目前缺乏完善的生态系统。
- **Mojo vs MAX：数据处理的双雄对决**：**MAX** 是一个编译器，与 **Mojo** 相比，它允许不同的数据处理权衡。虽然可以在纯 Mojo 中进行训练，但使用 MAX 是更好的主意，特别是在数据处理方面。
   - 两者都能够以不同的方式进行数据处理。
- **WebGPU 奇迹：Mojo 缺失的魔法**：虽然尝试在 Mojo 中使用 **WebGPU C 头文件** 似乎可行，但 Mojo 目前缺乏对该编译目标的支持，因此可能无法工作。
   - 你可以使用 **Vulkan 和 OpenGL** 库，但在添加支持之前，你不能在 Mojo 中编写着色器代码；*LLVM 支持 SPIR-V 后端，所以这并不遥远*。
- **眼见为实：FFI 指南传说**：关于如何在 Mojo 中执行 **FFI C** 的指导非常匮乏，只有标准库中的一些代码提供了示例。
   - 目前还没有关于如何在 Mojo 中进行 **FFI C** 的指南或示例，成员们指向了 stdlib 中的代码。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1438258766112882768)** (46 messages🔥): 

> `comptime keyword, stdlib versioning with pixi, LayoutTensor vs raw memory, HipKittens paper` 


- **`comptime` 关键字涵盖赋值**：Mojo 中的 `comptime` 关键字涵盖了以前 `alias` 的功能，包括类型赋值，实现了类似 Zig 风格的静态反射，例如 `comptime InnerTypeFirstField = T.fields[0].type`。
   - 虽然*对于类型赋值来说读起来有点别扭*，但在编译时激进地混合类型和值时，拥有不同的关键字可能会很烦人。
- **通过 Pixi 进行 Mojo stdlib 版本管理**：用户现在可以克隆一个 fork 并使用 `./bazelw build` 在 `bazel-out` 中创建一个 `stdlib.mojopkg`，从而替换 Pixi 环境中的现有版本。
   - 与其覆盖文件，不如在 `activation.env` 中定义 `MODULAR_MOJO_MAX_IMPORT_PATH` 来指向新的标准库，具体参考这些[说明](https://docs.modular.com/mojo/current/packages.html)。
- **`LayoutTensor` 实现零开销？**：虽然使用 raw memory 的方法看起来可能更快，但与手动内存管理相比，`LayoutTensor` 应该提供零开销和更多的调试信息，从而*避免麻烦*。
   - 仅在简单程序中坚持使用 raw memory；否则，使用 `LayoutTensor` 可以避免 ASAP 销毁问题，更多信息可以在 stdlib 的 `List` 和 `Span` 源码以及 MAX 的 `LayoutTensor` 源码中找到。
- **`HipKittens` 论文提到 Mojo 的性能**：[HipKittens 论文](https://arxiv.org/abs/2511.08083)提到 Mojo 的 MHA kernel 受到严重的 bank conflicts 困扰，在 MI355X 上仅达到峰值 kernel 性能的 50%。
   - 一位成员评论说，如果 LLVM 可以与设备通信，就可以在编译时构建抽象，这暗示了一种潜在的设备模型，可以减少对 AMD/NVIDIA 特定 kernel 的需求。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1438460489192570881)** (8 messages🔥): 

> `Apple Silicon GPU support in Max, Overhead passing tensors between Torch and Max with DLPack, GPU puzzles experience` 


- **Apple Silicon GPU 支持预计时间**：一位成员询问了 **Max** 何时能初步支持 **Apple Silicon GPU** 的预计时间。
   - 另一位成员回应说，得益于社区的 PR，他们已经扩展了对许多 intrinsics 的支持，并启用了更多测试和 GPU puzzles，基础的 **MAX graphs** 已经开始工作。
- **Tensor 传递开销曝光**：一位成员询问了在使用 **DLPack** 在 **Torch** 和 **Max** 之间传递 Tensor 时的开销，特别是关于 stream 同步的问题。
   - 他们请求提供*相关 issue 的链接*以便关注更新。
- **GPU puzzles 提供极佳体验**：一位成员报告说他们再次运行了 **GPU puzzles**，体验非常棒。
   - 另一位成员表示赞同，认为*在本地开发 kernel 并部署到超级计算机上的梦想*正在成为现实。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1438267257766543537)** (59 条消息🔥🔥): 

> `Whisper 用于转录, ICLR 评审质量, Multi-Head Attention, 会议评审改进, 出国留学 vs. 研究` 


- ****Whisper** 仍是开源转录领域的王者？**: 据一位成员称，**Whisper** 仍然是转录的最佳开源选择，尤其是使用通过 **Whisper-server** 量化为 **8 bit 和 4 bit** 的 [large-v3 model](https://openai.com/blog/whisper) 时。
   - 直接在 Python 中通过 PyTorch 运行 **Whisper** 可能会有问题，但使用带有 Vulkan 支持的 **Whisper.cpp** 可以提高便携性。
- ****ICLR** 评审质量大幅下降**: 成员们注意到今年 **ICLR** 的质量控制不尽如人意，出现了*大量带有提示词（prompts）残留的 LLM 评审*，相比之下 *NeurIPS 表现更好*。
   - 一位成员建议会议投稿应该更长，并对*明显是垃圾、纯属浪费评审时间的论文进行预评审拒绝（pre-review rejection）*。
- ****MHA** 多头随机初始化，Dropout 起到稳定作用**: 一位成员很好奇为什么 Multi-Head Attention (**MHA**) 的所有头最终不会学到相同的东西。
   - 其他人认为 *随机初始化起到了作用*，还有 Dropout；如果所有层都学到相同的东西，那将是浪费空间，而让每一层独立学习是更好的解决方案。
- **会议评审员应获得激励以提高质量**: 小组讨论了为一贯提供高质量评审的会议评审员提供金钱激励，这需要一个元评审员（meta-reviewer）系统。
   - 有人建议可以通过小幅提高会议注册费来为此提供资金，考虑到 **NeurIPS** 去年有 **16k** 名线下参会者。
- **Arxiv 策展与真实研究的分离**: 一位成员指出该空间的使用方式存在根本差异，建议为 **arxiv curation**（策展）与 **genuine research interest**（真实研究兴趣）的使用场景创建不同的空间。
   - 目前没有关于与垃圾邮件机器人（spambots）竞争的规则。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1438342028999262350)** (4 条消息): 

> `跷跷板偏好策略, X 帖子论文识别` 


- **跷跷板偏好（Seesaw Preference）困扰推荐系统**: 一位成员提到故意通过**跷跷板式**地改变偏好来迷惑推荐算法，并链接了[这篇论文](https://arxiv.org/abs/2511.08378)。
   - 该用户报告称，该链接是在一个 X 帖子中分享的，用于识别所链接的论文。
- **X 观察者确认论文链接**: 针对论文链接的确认，一位成员自称是 X 的*被动观察者*。
   - 这突显了社区内对于 X 等外部平台的不同参与程度。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1438267054284345386)** (16 条消息🔥): 

> `GPT-5 对话能力, Output Tokens, DeepMind 的 SIMA-2, AnthropicAI 间谍活动` 


- **GPT-5：更健谈是为了吸引 GPT-4 用户？**: 讨论围绕 **GPT-5** “更具对话性”的风格是否是为了吸引 **GPT-4** 用户在 [output tokens](https://openai.com/index/gpt-5-1/) 上花费更多。
   - 一位成员推测，不同级别的用户层级可能会对 output tokens 的数量有所限制，至少对于免费版本是这样。
- **DeepMind 的 SIMA-2 Agent 登上虚拟舞台**: 一位成员分享了 [DeepMind 的 SIMA-2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)，这是一个可以在虚拟 3D 世界中*玩耍、推理和学习*的 Agent。
   - 他们还想知道，如果这些模型生成的图像带有现有游戏的纹理，版权打击是否对这些模型有效。
- **利用 Anthropic 的 Claude 进行间谍活动？**: 一位成员链接到了 [AnthropicAI 的演示](https://x.com/AnthropicAI/status/1789033793190277618)，展示了先进的 AI Agent 正在变得多么强大。
   - 随后话题转向了**某些国家黑客**是否会利用 **Anthropic** 来收集训练数据，因为闭源权重模型（closed weight models）效果更好。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1438259599546515630)** (75 messages🔥🔥): 

> `Claude 的前后端技能, Latent Space Spotify 订阅源问题, Parag Agrawal 的 Parallel Web Systems 融资, Cursor 的 D 轮融资, AlphaResearch LLM Agent` 


- ****Claude 技能**：后端卓越，前端失手？**：一位成员发现 **Claude** 的后端技能非常精简且高效，但指出其前端技能只能算*一般*，模型仍会犯错，并引用了[这篇博客文章](https://www.claude.com/blog/improving-frontend-design-through-skills)。
- ****Spotify 风波**：版权混乱干扰 Latent Space 订阅源**：由于对无版税片头曲的版权主张，Latent Space 的 **Spotify 订阅源**面临问题，但随后有报道称 [Spotify 表示已修复该问题](https://x.com/paraga/status/1988729121636294682/photo/1)。
- ****平行宇宙**：Parag 的 AI Web 融资 1 亿美元**：**Parag Agrawal** 的新公司 **Parallel Web Systems** 获得了 **1 亿美元**的 A 轮融资，旨在为 AI 构建 Web，其简洁的设计引发了关注和赞誉，详见[此公告](https://xcancel.com/paraga/status/1988729121636294682/photo/1)。
- ****Anthropic 扩张**：500 亿美元打造美国数据中心霸权**：**Anthropic** 公布了一项计划，拟在德克萨斯州和纽约州的美国数据中心投资 **500 亿美元**，这将创造建筑就业机会，并引发了关于国内算力规模、人员配置、环境影响以及 AI 行业泡沫的辩论，详见[此贴摘要](https://xcancel.com/anthropicai/status/1988624013849935995?s=46)。
- ****Holo2 热潮**：更便宜的视觉模型挑战 GPT-4V**：**HCompany** 推出了 **Holo2**，这是一个基于 **Qwen3-VL** 构建的更便宜的 **30B-MoE** 视觉模型系列，在 UI 基准测试上超越了 **GPT-4V**，同时可在 Web、桌面和移动端运行，详见[此公告](https://xcancel.com/hcompany_ai/status/1989013556134638039)。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1438621839755841547)** (3 messages): 

> `Nano Banana, 图像模型, 提示词工程, 图像 Token 的 min_p` 


- **Nano Banana 提示词工程**：一位成员分享了关于 **Nano Banana** 图像模型提示词工程的博客文章链接：[Nano Banana Prompts](https://minimaxir.com/2025/11/nano-banana-prompts/)。
- **图像模型受到的关注比文本模型少？**：一位成员提到，他们对 **Nano Banana** 等图像模型的测试远没有对文本模型的测试那么多。
   - 他们还想知道在图像 Token 的生成过程中是否存在类似 *min_p* 的机制。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1438266498362769581)** (50 条消息🔥): 

> `Upscaled Models, Multi-Head Attention, HuggingChat's new monetization, AI Voice Implementation, Granite 4 series` 


- **Upscaled 模型成了笑料**：用户们拿一个经过微调的 **upscaled model** 开玩笑，觉得这个名字很有趣，并分享了一个 [Beavis and Butthead GIF](https://tenor.com/view/beavis-butthead-beavisandbuttheadshow-gif-13367061995602198225)。
   - 讨论中引用了与该模型相关的特定 Discord 频道，引发了轻松的调侃。
- **Multi-Head Attention 初始化！**：一位用户询问关于 **multi-head attention** 的问题，质疑为什么各个 head 不会学到相同的内容，主要原因在于随机初始化以及 [Softmax 夸大的差异](https://huggingface.co/datasets/John6666/forum2/blob/main/multi_head_attention_1.md)。
   - 另一位用户补充道，随机初始化、独立的参数和不同的梯度会导致 head 自然地产生分歧，而学习任务会保留有用的差异。
- **HuggingChat 变现引发哀叹**：成员们对 **新版 HuggingChat** 转向付费模式且免费功能受限表示遗憾，这与之前免费且无限制的版本形成鲜明对比，并展示了一张列出订阅弊端的 [截图](https://cdn.discordapp.com/attachments/879548962464493622/1438593728200572928/Screenshot_2025-11-13-18-16-43-962_mark.via.gp-edit.jpg?ex=69181b10&is=6916c990&hm=7bc833942ddb303310bdca35fdf5e940a15a9eb9a345cbc6086e0a21f8e817c6&)。
   - 一位用户表示，当他们意识到新版 HuggingChat 如此乏善可陈时感到*非常沮丧*，并质疑 Hugging Face 提供免费开源 AI 平台的宗旨是否已经化为泡影。
- **AI 语音实现令人印象深刻**：围绕 **AI 语音** 实现展开了讨论，有人对将语音集成到*色情机器人（smutty sexbot）*的想法感到兴奋。
   - 一位用户分享了一个具有 20 种人类情感的 [开源语音设计](https://huggingface.co/maya-research/maya1) 链接，并询问其实现难度以及如何将其添加到安卓应用中。
- **Granite 4 加入 Hugging Face**：一位用户询问是否支持新的 **IBM Granite 4 系列**。
   - 另一位用户表示它支持 Hugging Face Transformers，并提供了 [Granite 4.0-h-small 模型](https://huggingface.co/ibm-granite/granite-4.0-h-small) 及其 [GGUF 版本](https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF) 的链接。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1438490755155886080)** (3 条消息): 

> `AI-Powered Mixing Forum Analyzer, Propercode multi-agentic coding CLI tool, Geopolitical Forecasting Model Framework` 


- **混音论坛注入 AI 动力**：一位成员构建了一个 **AI 驱动的混音论坛分析器（Mixing Forum Analyzer）**，它使用语义搜索从论坛帖子中寻找相关的音频工程建议。
   - 主要功能包括使用句子嵌入（**SBERT**）进行**语义搜索**，基于 **spaCy** 的词性/形容词分析（用于混音术语）以及**词元重叠检测**，使用 **Python, Streamlit 和 Hugging Face 模型** 构建 ([GitHub](https://github.com/steme855/mixing-forum-analyzer), [在线演示](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer))。
- **Propercode 工具承诺生成生产级代码**：一位成员介绍了 **Propercode**，这是一个用于代码库的多 Agent（multi-agentic）编码 CLI 工具，由编排为图（graph）的 **Pydantic AI** Agent 驱动，旨在提高可靠性和编码准确性 ([GitHub](https://github.com/JaiSuryaPrabu/proper-code))。
   - 该工具目前处于 **v0.1** 版本，计划为 Agent 提供多工具和多种模式，如自主模式、学习指南模式等。
- **地缘政治预测框架首次亮相**：一位成员分享了他们的 **地缘政治预测模型框架（Geopolitical Forecasting Model Framework）**，用户可以插入自己的政治数据，并开始生成关于当前或潜在冲突的预测 ([Hugging Face](https://huggingface.co/clarkkitchen22/GeoBot-Forecasting-Framework))。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1438549080132423783)** (1 条消息): 

> `Kaggle, Deep Learning, Large Datasets` 


- **Kaggler 询问如何处理海量数据集**：一位成员寻求关于为深度学习预处理 **150 GB Kaggle 大型数据集** 的建议，目前面临 **Kaggle 20GB 的写入限制**。
   - 该成员询问其他人在平台限制下是如何处理如此庞大数据集的预处理的。
- **大型数据集预处理的解决方案**：潜在的解决方案包括**使用外部存储**、**数据分块（data chunking）**或**基于云的处理**，以克服 Kaggle 的限制。
   - 社区经常建议利用 **Google Cloud** 或 **AWS** 等云服务来进行可扩展的数据预处理。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1438280740658741278)** (46 messages🔥): 

> `DSPy prompting, Pydantic models for signatures, Agentic search, ripgrep regex usage, Survey language issue` 


- **DSPy 的 Prompting 认知**：一些成员认为 DSPy 在传播过程中*无意中*被描述为一个不再需要对 LLM 进行 Prompting 的框架，这可能会产生误导，因为即使有示例和 **GEPA**，在特定领域的 LLM 应用中仍然需要 Prompting。
   - 一位成员将其描述为：DSPy 提供了将 Prompting 封装为可编程模块以便后续优化的能力，并强调 Signature 类的指令或 Docstrings 构成了 **Prompt**。
- **更倾向于 Pydantic Signatures**：一位成员表示不喜欢文档中将 Signature 引入为简单字符串类型（`input -> output`）的方式，更倾向于使用 **Pydantic models** 来处理更复杂且经过类型检查的实际用例。
   - 该成员还指出，Signature 中的指令起到了 Prompt 的作用，社区中的困惑源于对 *Prompt* 含义的不同理解。
- **Agentic Search 指令**：为了增强 **Agentic search**，一位成员在其 **ReAct module** 中指示 LLM 通过 **ripgrep** 发送特定术语进行工具搜索，并添加 **Regex** 作为初始搜索失败后的备份。
   - 这一指令对于 LLM 在搜索工具中有效使用 Regex 术语至关重要，尤其是在访问多个工具（Agentic search 的 3-4 个函数）时。
- **调查问卷语言疑云**：一位成员根据一张截图怀疑 **Survey language** 存在问题，截图中 **Fine-tuning** 被标注在顶部。
   - 另一位成员称，将调查语言埋在附录中而将 Fine-tuning 放在顶部简直是*疯狂*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1438763509457616956)** (1 messages): 

> `Kimi K-2, Together AI partnership, 1T MoE` 


- **Kimi K-2 深度解析即将到来！**：**Moonshot AI** 团队宣布将与 **Together AI** 合作，对 **Kimi K2 Thinking** 进行一次“快速但强大”的深度解析。
   - 活动将于 **2025 年 11 月 19 日** **上午 9 点（太平洋时间）** 举行，注册地址为 [luma.com/g5qcq85z](https://luma.com/g5qcq85z)。
- **Together AI 与 1T MoE 的力量**：公告强调了 **1T MoE** (Mixture of Experts) 的强大能力，以及它在**单次运行中执行 300 次工具调用**的能力。
   - 他们邀请用户*发现这对你的 Agent 意味着什么*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1438273815711977562)** (43 messages🔥): 

> `GLM-5, K3, R2, Kimi CLI vs Hello Computer, YC-backed Kimi For Coding, Chinese AI labs entering chipmaking, Kimi Chat projects and custom instructions` 


- **GLM-5, K3, R2 即将到来**：成员们正期待 **GLM-5**、**K3** 和 **R2** 的到来，同时一些人对 **OpenAI** 被察觉到的失误表示满意。
   - 一位成员表示：*我对 Gemini 3 之类的模型一点也不兴奋，因为它们不像 Kimi/GLM 那样提供良好的编程方案*。
- **Kimi K2 虽慢但比其他模型更强大**：尽管进行了 int4 优化，据报道 **Kimi K2-thinking** 比 **GLM-4.6** 慢 **1.8 倍**，比 **Minimax m2.wysh.3** 慢 **2.5 倍**，但在非纯编程任务中被认为更强大。
   - 一位用户分享了一张 **Kimi K2** 产生幻觉的幽默截图。
- **YC 支持 Kimi For Coding 引发争论**：**Y Combinator** 支持的 **Kimi For Coding** 每周仅有 **2048** 次使用配额，被批评为对平庸模型的*光天化日之下的抢劫*，一位用户表示：*我不敢相信他们认为这种产品必须存在*。
   - 讨论中分享了相关的 [Y Combinator 推文](https://x.com/ycombinator/status/1988366241460089118?s=46) 链接。
- **Moonshot AI K100 即将到来？**：紧随美国 AI 实验室的趋势，中国 AI 实验室也正在进入芯片制造业务。
   - 用户们正期待 **Moonshot K100** 的发布，并引用了强调这一趋势的[这条推文](https://x.com/tphuang/status/1988952992003891330?s=46)。
- **Kimi Chat 缺少关键功能！**：用户质疑 **Kimi Chat** 网站为何缺少 Project 和自定义指令（Custom Instructions）功能，即使是付费订阅者也没有。
   - 其他人澄清说，**Tool use** 在 **Claude Code** 或 **Kimi-CLI** 上是自动启用的，当 AI 使用网页搜索或文件读取等外部资源时就会触发。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1438262337579782228)** (32 messages🔥): 

> `GPT-5.1, System/Model Card, Aider-ce, Deepseek API, moonshotai Kimi-K2` 


- **GPT-5.1 发布且未提供 Benchmarks**：**GPT-5.1** 的发布缺少任何提及的 Benchmarks，其 [system/model card](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf) 被一些人认为是“笑话”。
   - 目前没有 API，这被认为是可疑的，引发了关于它是新模型还是仅仅修改了 system prompts 的质疑。
- **Aider-ce 被赞誉**：**Aider-ce** 因已经具备频道中讨论的功能而受到称赞，但对其维护者缺乏沟通和继任计划感到担忧，其中一个要求有序继任计划的 issue 被标记为 *not planned* 并关闭。
   - 尽管如此，该分支仍广受好评，服务器每天都有稳定的新用户涌入。
- **Deepseek API 缩短了 Agent 模式耗时**：用户发现在 **Aider-ce** 的 Agent 模式中使用 **Deepseek API** 非常成功，并报告称使用 **GPT-5-high** 会导致性能缓慢。
   - 有建议认为，速度慢是因为在较大的 repo 上重新生成 repo map 导致的，调整 `repo_map_max_files` 和 `repo_map_token_budget` 等设置可能会有所帮助。
- **moonshotai Kimi-K2 Thinking 在 Aider 上运行**：一位用户通过将 **OPENAI_API_BASE** 变量修正为 `https://llm.chutes.ai/v1/`，解决了在运行 aider 配合 `moonshotai/Kimi-K2-Thinking` 时出现的 *404 no cord found* 错误。
   - 正确的命令为：
```
SET "OPENAI_API_BASE=https://llm.chutes.ai/v1/"
SET "OPENAI_API_KEY=mykey"
aider --model openai/moonshotai/Kimi-K2-Thinking
```


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1438485994423975936)** (3 messages): 

> `DeepSeek API, Commercial Privacy for DeepSeek` 


- **DeepSeek 赢得粉丝**：一位成员表示自己已经“基本被 **DeepSeek** 折服”，打算在所有场景下使用它。
   - 该成员澄清说，他们仅在开源或个人项目中使用 **DeepSeek API**。
- **DeepSeek 的商业隐私担忧**：另一位成员询问是否有服务在提供 **DeepSeek** 的同时尊重商业隐私。
   - 最初的成员表示他们使用的是原始的 **DeepSeek API**，没有商业层面的需求。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1438446165002686545)** (7 messages): 

> `Postgres MCP Server, executeQuery MCP Tool, SQL Query Results Format, GitHub Discussions vs Discord for MCP` 


- **MCP Server 构建者思考查询结果格式**：一位成员正在构建 **Postgres MCP server**，并就 **executeQuery MCP tool** 的最佳输出格式寻求建议。
   - 他们正在考虑使用 **JSON**、**Markdown** 或 **Toon** 格式来返回 **SQL 查询结果**。
- **Discord 管理员重定向通用介绍**：一位用户在频道打招呼后被重定向到 <#1358874322030301417> 进行正式的自我介绍。
   - 管理员强调该频道用于协议讨论和相关的官方 SDK 项目。
- **鼓励在 GitHub Discussions 讨论 MCP 实现**：对于通用的 **MCP 实现讨论**，管理员建议使用 [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) 或其他社区 Discord 服务器。
   - 本 Discord 服务器专门用于讨论协议及相关的官方项目（如 SDK）。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1438492629582221453)** (7 messages): 

> `File Transfer over MCP, MCP Resources, MCP Implementations` 


- **文件传输问题得到解答**：一位成员询问当前的 **MCP 协议** 是否支持从客户端到服务器的文件传输，答案是原生（natively）**不支持**。
   - 另一位成员澄清说，可以通过 **resources** 实现文件传输。
- **MCP 实现的讨论**：一位用户询问是否可以通过 tools 使用 **SE upload call** 直接从聊天界面上传文件。
   - 其他成员指出，该 Discord 服务器是为了讨论**协议和相关的官方项目**，建议用户发起 [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) 或在其他社区 Discord 服务器中提问，并指引其参考 [issue 1306](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1306)。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1438275827405029426)** (9 messages🔥): 

> `Manus Support, Credit Changes, AI/ML Engineer, workflow automation engineer` 


- ****Manus Support** 关闭了？**: 有用户报告 **Manus 的 checkpoint 系统**无法找到 git commits，导致 "Publish" 按钮被禁用，需要 **Manus support** 手动同步内部 repository。
   - 在询问 Manus 是否刚删除了 chat mode 后，该用户被引导至 [Manus feedback](https://manus.im/feedback)。
- **新的 Credit 方案推出**: 用户注意到订阅计划的每月 **credit 分配**发生了变化，特别指出 19 美元的计划现在提供 **4000 credits**，而之前是 **1900 credits**。
   - 他们想知道 credit 的消耗方式是否也发生了变化。
- **AI/ML engineer 求职！**: 一位在模型设计、优化和大规模部署方面具有专业知识的 **AI/ML engineer** 介绍了自己。
   - 他们的技术栈包括 **Python, C++, Rust, SQL, Hugging Face, ONNX Runtime, Triton Inference Server** 和 **Apache Spark**。
- **自动化工程师寻求工作**: 一位擅长 **workflow automation, LLM integration, RAG, AI detection, image and voice AI** 的资深工程师提供服务。
   - 他们使用 **Dspy, OpenAI APIs** 和自定义 agents 构建了自动化流水线和任务编排系统，并附上了作品集链接 [devx-green.vercel.app](https://devx-green.vercel.app/)。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1438441292483792958)** (7 messages): 

> `OpenCL Devices, tinygrad with C++, torch_load VGG16, pmap and vmap, OpenPilot PR` 


- ****OpenCL Devices** 错误修复建议**: 用户建议在代码库中添加 `if num_devices.value == 0: raise RuntimeError("No OpenCL Devices")`，以解决在没有可运行设备时代码仍继续执行的问题。
   - 该用户引用了关于 `CL_INVALID_VALUE` 错误的 [OpenCL 文档](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html)，即当 `num_entries` 为零且 `devices` 不为 NULL，或者 `num_devices` 和 `devices` 均为 NULL 时的情况。
- ****pmap 和 vmap** 更新即将推出**: 一位用户在 ChatGPT 的帮助下更新了 [README](https://github.com/tinygrad/tinygrad)，强调了对 **pmap** 和 **vmap** 功能的需求。
   - 这些函数将使 **tinygrad** 内部的数组操作更加高效和简洁。
- ****torch_load**: VGG16 模型现已可用**: 一个 Pull Request ([PR #13253](https://github.com/tinygrad/tinygrad/pull/1325)) 已创建，使 **torchvision** 托管的 **VGG16** 模型能够与 `torch_load` 配合使用。
   - 这一增强扩展了 **tinygrad** 与 **torchvision** 库中预训练模型的兼容性。
- ****OpenPilot PR** 已合并**: [OpenPilot PR](https://github.com/commaai/openpilot/pull/36615) 已经合并，增强了 **tinygrad** 与 **OpenPilot** 之间的集成或兼容性。
   - 开发者承诺未来将防止此集成出现回归 (regressions)。
- ****tinygrad** 与 C++ 集成？**: 有用户询问如何将 **tinygrad** 与 **C++** 结合使用，并打算将其应用于嵌入式系统。
   - 这表明开发者有兴趣在资源受限的环境和应用中利用 **tinygrad**。