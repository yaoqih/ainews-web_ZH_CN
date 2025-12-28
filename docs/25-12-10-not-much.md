---
companies:
- nousresearch
- thinkymachines
- mistral-ai
- deepseek
- anthropic
- cursor
- microsoft
- langchain-ai
- openai
- gemini
- intel
- vllm_project
- danielhanchen
date: '2025-12-10T05:44:39.731046Z'
description: '**NousResearch 的 Nomos 1** 是一款 30B（300亿参数）开源数学模型，仅凭约 3B 的激活参数便取得了普特南数学竞赛（Putnam）的顶尖成绩，并支持在消费级
  Mac 上进行推理。**AxiomProver** 同样利用 ThinkyMachines 的强化学习（RL）技术栈发布了顶尖的普特南竞赛结果。**Mistral
  的 Devstral 2 Small** 在 71% 的偏好测试中胜过 DeepSeek v3.2，且具备更好的速度和成本优势。**Anthropic 的 Claude
  Code** 引入了异步智能体执行功能。**Cursor 2.2** 增加了调试（Debug）和计划（Plan）模式等深度智能体原语。**VS Code** 推出了统一的智能体聊天会话，优化了多智能体工作流。**LangChain**
  发布了用于智能体可观测性的“Polly”。**Stirrup** 测试框架在 OpenAI GDPval 基准测试中处于领先地位，紧随其后的是 Claude Opus
  4.5、GPT-5 和 Gemini 3 Pro。量化技术的进展包括 **vLLM** 集成了英特尔的 AutoRound PTQ 以实现高效服务。**Unsloth**
  通过针对 Llama、Qwen、Mistral 和 Gemma 模型的新算子，实现了高达 3 倍的训练加速。*“组合推理 + 受限激活参数下的专门后训练，可以在形式数学领域媲美顶尖的闭源模型。”*'
id: MjAyNS0x
models:
- nomos-1
- axiomprover
- devstral-2-small
- deepseek-v3.2
- claude-code
- cursor-2.2
- claude-opus-4.5
- gpt-5
- claude-sonnet-4.5
- gemini-3-pro
- llama
- qwen
- mistral
- gemma
people: []
title: 今天没发生什么事。
topics:
- math
- formal-reasoning
- agentic-systems
- asynchronous-execution
- multi-agent-systems
- observability
- benchmarking
- quantization
- post-training-quantization
- training-speedup
- kernel-optimization
- inference-efficiency
---

**最后一批发布前的宁静。**

> 2025年12月9日至12月10日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 24 个 Discord 社区（205 个频道，6101 条消息）。预计为您节省阅读时间（以 200wpm 计算）：529 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！

查看 [来自 AIE Code 的 RL 演讲](https://x.com/aiDotEngineer/status/1998785602989461531?s=20)。

---

# AI Twitter 回顾

**开放数学与推理：小型激活参数 + Agent 达到顶级性能**

- **NousResearch 的 Putnam 级证明器（开源）**：社区报告显示，新的 “Nomos 1” 数学系统是一个 30B 的开源模型，在今年的 Putnam 竞赛中获得了 87/120 的分数——预计排名第 2/3988 位。这一成绩是通过专门的后训练（post-training）和 Agentic 流水线实现的；重要的是，推理时仅有约 3B 参数处于激活状态，使其能够在消费级 Mac 上运行 ([1](https://twitter.com/kimmonismus/status/1998749650984255985), [2](https://twitter.com/EMostaque/status/1998686465279025190), [3](https://twitter.com/Dorialexander/status/1998657955718148268))。与此同时，使用 ThinkyMachines 的 Tinker RL 栈进行微调的 **AxiomProver** 也发布了顶级的 Putnam 评分 ([1](https://twitter.com/thinkymachines/status/1998903749000180183), [2](https://twitter.com/thinkymachines/status/1998925489084498094), [3](https://twitter.com/ariG23498/status/1998654584529797522))。结论：在受限的激活参数下，组合推理 + 专门的后训练可以媲美最前沿的闭源模型在形式数学上的表现。

**Agentic 编程系统、编排与评估**

- **Mistral 的 Devstral 2 势头强劲**：从业者报告称，Devstral 2 Small 在 71% 的第三方偏好测试中“击败或持平” DeepSeek v3.2，同时体积更小、速度更快、价格更低，并配备了精美的 Vibe CLI 入门体验（需要图像支持） ([1](https://twitter.com/swyx/status/1998600513538109476), [2](https://twitter.com/N8Programs/status/1998591943798882484))。
- **Claude Code 实现异步化**：Anthropic 发布了后台子 Agent 和异步执行功能 (v2.0.64)，支持并发探索/测试，并在完成后“唤醒”主 Agent ([1](https://twitter.com/omarsar0/status/1998774531188830304), [2](https://twitter.com/omarsar0/status/1998777320434290729), [3](https://twitter.com/omarsar0/status/1998789689587708246))。
- **Cursor 2.2 发布深度 Agent 原语**：调试模式（Debug Mode）可以检测你的代码，启动服务器以捕获日志，并将运行时数据流式传输给 Agent；升级还包括计划模式（Plan Mode）图表和多 Agent 评审 ([1](https://twitter.com/cursor_ai/status/1998821350333440133), [2](https://twitter.com/cursor_ai/status/1998821554000388096), [3](https://twitter.com/cursor_ai/status/1998821555250380986))。
- **VS Code “Agent 会话”**：统一聊天界面集成了本地/后台/云端 Agent，具备工作树隔离（worktree isolation）和无缝 Agent 交接（“在...中继续”）功能——这是迈向真实多 Agent 工作流的重要 UX 步骤 ([1](https://twitter.com/code/status/1998827135855743148), [2](https://twitter.com/pierceboggan/status/1998829467649937690), [3](https://twitter.com/burkeholland/status/1998835297644425485))。
- **Agent 的可观测性**：LangChain 发布了 “Polly”（一个用于调试 Agent 的 Agent）和用于提取追踪（traces）/线程的 CLI——从简单的 LLM 应用调试转向长期运行、复杂的 Agent 系统 ([1](https://twitter.com/LangChainAI/status/1998807193320305101), [2](https://twitter.com/hwchase17/status/1998809833693467100), [3](https://twitter.com/LangChainAI/status/1998814975033487822))。
- **Stirrup + GDPval-AA (Artificial Analysis)**：一个轻量级的开源 Agent 框架，以及针对 OpenAI GDPval 任务（涵盖 9 个行业的真实知识工作）的新排行榜。结果 (Elo)：Claude Opus 4.5 领先，其次是 GPT-5、Claude Sonnet 4.5，然后是 DeepSeek V3.2 和 Gemini 3 Pro 并列。值得注意的是，Stirrup 框架在不同模型上的表现均优于消费级聊天机器人 UI ([1](https://twitter.com/ArtificialAnlys/status/1998841566627246173), [2](https://twitter.com/ArtificialAnlys/status/1998843644628054506))。
- **MCP 工作流组合**：“Remix servers” 模式允许你将来自多个 MCP 服务器的工具界面组合成一个虚拟服务器，并带有服务器端编写的提示词/工作流（可跨客户端移植） ([链接](https://twitter.com/AAAzzam/status/1998773774699614537))。

**系统、性能与算力趋势**

- **量化与 PTQ**：vLLM 在 LLM Compressor 中集成了 Intel 的 AutoRound 后训练量化（PTQ），生成的 W4A16 检查点可直接在 vLLM 上跨 Xeon、Gaudi、Arc GPU 等平台提供服务。([链接](https://twitter.com/vllm_project/status/1998710451312771532))。
- **Unsloth 训练加速**：全新的融合 varlen RoPE + int64 Triton 内核以及无填充（padding-free）训练，在 Llama/Qwen/Mistral/Gemma 系列模型中实现了高达 3 倍的训练速度提升和约 50% 的显存（VRAM）节省，且 loss/grad norm 保持一致。([1](https://twitter.com/danielhanchen/status/1998770347081109864), [2](https://twitter.com/danielhanchen/status/1998770349975155060), [3](https://twitter.com/danielhanchen/status/1998770352646914146))。
- **架构、互联与成本**：AWS B300 EFA v4 节点间带宽达到 800 GB/s，而 NVLink-5 节点内带宽为 900 GB/s——互联技术的追赶仍在继续。([1](https://twitter.com/StasBekman/status/1998821183844938000), [2](https://twitter.com/wightmanr/status/1998915115744428369))。Epoch 估计 B200 芯片成本约为 6400 美元，芯片级利润率约为 80%（逻辑晶圆成本占比 <15%），但在整机服务器中实际利润率有所下降；NVIDIA 近期整体利润率约为 73%。([1](https://twitter.com/EpochAIResearch/status/1998819237251657890), [2](https://twitter.com/EpochAIResearch/status/1998819296353595424))。SemiAnalysis 详细介绍了 TPUv8 的两条路径：博通（Broadcom）交付的 “Sunfish”（捆绑方案）与谷歌组装的 “Zebrafish”（联发科 MediaTek 支持）。([链接](https://twitter.com/SemiAnalysis_/status/1998830078629724596))。
- **太空计算（与物理学）**：据团队称，Starcloud-1 的 H100 在轨道上训练了 nanoGPT（莎士比亚数据集）并运行了 Gemma 推理——这是首个在太空进行的 LLM 训练演示。([1](https://twitter.com/AdiOltean/status/1998769997431058927), [2](https://twitter.com/karpathy/status/1998806260783919434))。反对观点指出，真空中存在严重的散热辐射限制，且相比“太空数据中心”，地面发电（核能/太阳能+电池）成本更低。([1](https://twitter.com/jenzhuscott/status/1998591718338486757), [2](https://twitter.com/clawrence/status/1998753444598010254), [3](https://twitter.com/YIMBYLAND/status/1998785782082056626))。

**多模态、视觉/视频与事实性**

- **GLM-4.6V (智谱)**：早期用户表示其“听起来像 Sonnet”，在编程和视觉理解方面的表现接近 Sonnet 4，并且是他们发现的首个对设计评审有用的开源（OSS）视觉模型；定价低于 Gemini-2.5-Flash。观察到一些循环现象——后训练（post-training）可能会有所帮助。([1](https://twitter.com/hrishioa/status/1998636234806341873), [2](https://twitter.com/hrishioa/status/1998636284533944725))。
- **Qwen3-Omni-Flash (2025 年 12 月更新)**：实时多轮视频/音频对话的重大升级，支持 119 种文本语言、19 种语音，具备系统提示词（system-prompt）角色控制功能，并提供实时和离线 API 及演示。([链接](https://twitter.com/Alibaba_Qwen/status/1998776328586477672))。
- **Perceptron Isaac-0.2 (开源 VLM)**：1B/2B 混合推理视觉语言模型（SigLIP + Qwen），旨在为机器人提供强大的感知骨干网络；代码/权重已开源并提供 API；视频原生和控制模态已列入路线图。([1](https://twitter.com/perceptroninc/status/1998812935821697363), [2](https://twitter.com/AkshatS07/status/1998818590405935468))。
- **视频生成与视觉研究**：Meta 的 OneStory（具有自适应记忆的连贯多镜头视频）和 Wan-Move（通过潜空间轨迹引导实现动作可控视频）扩展了可控性；“Reflection Removal through Efficient Adaptation of Diffusion Transformers”展示了高效的窗户反光清理；通过 D4RT 进行的动态场景重建继续推动 4D 感知的发展。([1](https://twitter.com/_akhaliq/status/1998760879261888814), [2](https://twitter.com/_akhaliq/status/1998606187500097588), [3](https://twitter.com/_akhaliq/status/1998752500673888409), [4](https://twitter.com/_akhaliq/status/1998763356883452031))。
- **事实性基准测试**：DeepMind/Google Research 发布了 FACTS，这是一套涵盖内部知识、网络搜索、Grounding 和多模态输入的测试套件；Gemini 3 Pro 以 68.8% 的得分领先。基准测试已在 Kaggle 上发布，以标准化可靠性评估。([1](https://twitter.com/GoogleDeepMind/status/1998831084277313539), [2](https://twitter.com/GoogleDeepMind/status/1998831088324473025))。

**自主性、主动 Agent 与 AI 原生产品闭环**

- **Wayve x Nissan**：达成最终协议，将 Wayve 的 AI Driver 部署到下一代 ProPILOT 中——涵盖 Nissan 全球产品线的 ADAS 和点对点驾驶 ([链接](https://twitter.com/alexgkendall/status/1998592238641656160))。
- **来自可穿戴设备的 Proactive agents**：“ProAgent” 通过第一人称视角传感器（视频/音频/运动/位置）进行持续感知并主动提供协助（天气、打车、价格查询），设备端 Jetson Orin 延迟约 4.5 秒；在用户研究中，主动预测准确率提升了 33.4%，内存占用比基准线低 1.79 倍 ([链接](https://twitter.com/dair_ai/status/1998775732001190018))。
- **Shopify 的 AI 技术栈**：
    - SimGym 模拟“数字客户”，用于任务完成和零流量 A/B 测试。
    - Sidekick Pulse 在夜间运行大型 HSTU + LLMs，以挖掘业务改进点。
    - Product Network 让商家通过 LLM 驱动的匹配和站内结账功能互相销售产品 ([1](https://twitter.com/MParakhin/status/1998786503779234276), [2](https://twitter.com/MParakhin/status/1998788090324988244), [3](https://twitter.com/MParakhin/status/1998789844794012049))。
- **运营化工具**：GitHub Copilot 自动模型选择在 VS Code 中正式发布 (GA) ([链接](https://twitter.com/GHchangelog/status/1998847752050983279))。Pixel Watch 3+ 使用设备端 Gemma 进行智能回复 ([链接](https://twitter.com/Google/status/1998849211941482513))。Google 的 Jules 增加了建议/计划任务以及 Render 集成，用于自愈部署——将“持续 AI”推入 DevOps 循环 ([1](https://twitter.com/julesagent/status/1998829514634531252), [2](https://twitter.com/julesagent/status/1998848018817364175), [3](https://twitter.com/julesagent/status/1998875242413044130))。

**热门推文（按互动量排序）**

- “你创造的东西是你真实自我的诚实反映。” — [@naval](https://twitter.com/naval/status/1998671506784547309)
- 首次在太空进行由 H100 驱动的 LLM 训练（在莎士比亚作品上训练 nanoGPT） — [@AdiOltean](https://twitter.com/AdiOltean/status/1998769997431058927)
- 使用 GPT-5.1 Thinking 对过去十年的 HN 进行自动评分 — [@karpathy](https://twitter.com/karpathy/status/1998803709468487877)
- Claude Code 增加了异步 subagents — [@omarsar0](https://twitter.com/omarsar0/status/1998774531188830304)
- Cursor 2.2 发布 Debug Mode + Agent 升级 — [@cursor_ai](https://twitter.com/cursor_ai/status/1998821350333440133)
- Qwen3-Omni-Flash（12 月更新） — [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998776328586477672)
- Waymo 将于 2026 年进入伦敦 — [@demishassabis](https://twitter.com/demishassabis/status/1998825670869397802)

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Unsloth AI 训练优化

- [**现在你可以以 3x 的速度和减少 30% 的显存训练 LLM！（<3.9GB VRAM）**](https://www.reddit.com/r/LocalLLaMA/comments/1pj51tu/you_can_now_train_llms_3x_faster_with_30_less/) (热度: 814): **该图片展示了 Unsloth 通过新的 Triton kernels 和智能自动打包（auto packing）支持实现的性能提升，这使得像 Qwen3-4B 这样的大语言模型（LLMs）训练速度提升高达 3 倍，同时显存（VRAM）占用减少 30-90%，仅需不到 3.9GB 的 VRAM。图中包含的柱状图对比了新的 Unsloth RoPE + MLP kernels 与使用 FA3 的优化配置在训练吞吐量和加速比方面的表现，突显了在不损失精度的前提下实现的显著效率提升。关键技术进步包括：速度提升 2.3 倍的 QK Rotary Embedding 融合 Triton kernel、更新的带有 int64 索引的 SwiGLU 和 GeGLU kernels，以及在各种后端上快 2.5 倍至 5 倍的无污染打包（uncontaminated packing）。这些优化会自动启用，提供更好的 SFT loss 稳定性和可预测的 GPU 利用率。** 评论者对性能提升印象深刻，指出新方法明显快于之前的版本。人们对这些进步如何惠及显存容量较低的用户（如 6GB 显存用户）表现出浓厚兴趣，并询问了与多 GPU 配置的兼容性。
    - 一个关键的技术洞察是，其声称实现的训练速度比 Unsloth 旧有的“超过 2.5 倍加速”方法还要快 3 倍，这表明在之前的优化基础上有了显著改进。这暗示了训练效率的累积增强，可能是通过算法改进或更好的资源管理实现的。
    - 关于与多 GPU（如两块 3090）兼容性的讨论，突显了社区对可扩展性和成本效益的共同关注。与投资单块昂贵的高端 GPU 相比，有效利用多块 GPU 的能力可以显著降低成本。
    - 针对特定硬件兼容性的提问（如 AMD Strix Halo Max+ 395），表明需要跨不同架构的更广泛支持。这反映了社区希望确保这些优化不局限于特定硬件，从而提高更广泛用户的可访问性。

### 2. Mistral AI 模型发布

- [**Mistral AI 在一周内发布的 LLM 数量是 OpenAI 六年发布量的 3 倍**](https://www.reddit.com/r/LocalLLaMA/comments/1pj8kb6/mistral_ai_drops_3x_as_many_llms_in_a_single_week/) (热度: 560): **Mistral AI 在一周内发布了一系列大语言模型（LLMs），超过了 OpenAI 六年内发布的模型数量。这些模型涵盖了从** `3B` **到** `675B` **的各种参数规模，并采用 Apache 2.0 和修改后的 MIT 许可证。这些模型专为各种应用设计，包括编码、推理和指令遵循，并针对本地使用进行了优化。其中最大的模型是** `675B` **参数的 instruct 模型，代表了 Mistral 最先进的产品。所有模型均可通过 [Hugging Face](https://huggingface.co/bartowski) 获取。** 评论者注意到 `Devstral 2 123B` 模型比之前的模型有显著改进，尽管有些人将其归因于潜在的“新模型炒作”。此外，还有关于 Mistral 与 OpenAI 伦理影响的批判性对比，强调了 Mistral 在参与策略上的缺失。
    - 'DragonfruitIll660' 的评论强调了 Devstral 2 123B 的发布，指出它比 Mistral Large 2 有显著改进，特别是在基础聊天功能方面。这表明 Mistral AI 的新模型在性能上取得了长足进步，这可能得益于开放权重模型（open weight models），这些模型允许更多社区驱动的增强。
    - 'Long_comment_san' 讨论了希望 Mistral AI 发布 800 亿到 1200 亿参数范围模型的愿望，特别是混合专家模型（MOE）。该评论者指出，目前的 Mistral Large 模型大小超过 128GB，限制了实验的可访问性，并对在 AI 社区中日益流行的、更小且可微调的模型表示感兴趣。
    - 讨论涉及了向更小、可微调模型发展的趋势，正如 'Long_comment_san' 所提到的。这反映了更广泛的行业转变，即紧凑型模型因其效率和适应性而受到关注，这与传统上对大规模模型的关注形成对比。对 Qwen 的提及以及对 'Qwen Next' 的期待表明，人们对模型开发领域的竞争性进步保持着持续关注。

### 3. 硬件与 CLI 创新

- [**全新的 CLI 体验已合并至 llama.cpp**](https://www.reddit.com/r/LocalLLaMA/comments/1pj4j87/new_cli_experience_has_been_merged_into_llamacpp/) (活跃度: 514): **该图片展示了 `llama.cpp`（ggml-org 旗下的一个项目）全新的命令行界面（CLI）体验。根据[此 Pull Request](https://github.com/ggml-org/llama.cpp/pull/17824)的详细说明，此次更新引入了更用户友好的界面，包含 `exit`、`regenerate`、`clear` 和 `read` 等命令。该 CLI 还提供 Prompt 和生成速度的性能指标，增强了开发者和用户在使用 `llama.cpp` 虚拟助手功能时的易用性。** 一位评论者推测，这次更新可能会挑战 **ollama** 的地位，而另一位则认为 `llama.cpp` 中集成的 WEB/CLI 支持可能会影响 OpenWebUI/OpenCode 等项目的实用性。
    - 在 `llama.cpp` 中集成全新的 CLI 体验是一项重大增强，可能会影响 OpenWebUI/OpenCode 等其他界面的实用性。通过整合 Web 和 CLI 功能，此次更新可以简化工作流程，使其成为此前依赖多个平台完成不同任务的开发者的更通用工具。
    - `llama.cpp` 的持续改进（如最近的 CLI 更新）凸显了其不断进化的能力，以及取代 Ollama 等其他工具的潜力。这 st 可能会带来更统一的开发环境，减少对多个独立工具的需求，并简化用户体验。
    - 围绕 `llama.cpp` 新 CLI 体验的讨论表明，人们对其扩展功能的兴趣日益浓厚，甚至可能向开发 Coding Agent 方向发展。这表明该项目的路线图具有前瞻性，旨在通过集成更高级的功能来增强对开发者的实用性。
- [**我在 Reddit 上以 7500 欧元买了一台 Grace-Hopper 服务器并将其改装成了台式机。**](https://www.reddit.com/r/LocalLLaMA/comments/1pjbhyz/i_bought_a_gracehopper_server_for_75k_on_reddit/) (活跃度: 309): **一位 Reddit 用户以 `€7.5k` 的价格购买了一台原价 `€10k` 的 Grace-Hopper 服务器，并将其改装成了一台能够运行 `235B 参数模型` 的台式机。该服务器原设计为液冷，后被改装为风冷并再次改回，克服了 GPU 报告极端温度等挑战。该项目是该用户 [GLaDOS Project](https://github.com/dnhkng/GlaDOS) 的一部分，展示了企业级 AI 硬件向家用系统的转变。完整故事记录在[博文](https://dnhkng.github.io/posts/hopper/)中。** 评论者认为这次购买非常划算（'steal'），并建议在该硬件上使用 `vllm`，强调了使系统投入运行所需的巨大努力，但也承认了这笔交易的价值。
    - cantgetthistowork 建议在 Grace-Hopper 服务器上使用 `vllm`，暗示该软件非常适合该硬件的能力。`vllm` 以高效处理 LLM 而闻名，可以有效利用服务器的高性能组件。

## 非技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
>

### 1. OpenAI 战略转型与 AGI 暂停

- [**OpenAI 因重大战略调整暂停 AGI 追求**](https://www.reddit.com/r/ChatGPT/comments/1piudnw/agi_pursuit_paused_for_a_major_strategic_course/) (热度: 669): **图片及随附文字强调了 OpenAI 的战略转变，首席执行官 Sam Altman 发起了“红色警报 (code red)”，要求优先改进 ChatGPT，而非 Sora 视频生成器等其他项目。这一决定标志着一次重大的航向修正，强调需要提高用户对 ChatGPT 的参与度和满意度，这可能会以牺牲公司实现通用人工智能 (AGI) 的更宏大目标为代价。此举反映了 OpenAI 内部关于平衡大众普及度与研究雄心之间的辩论，以及 Altman 的领导风格，后者因缺乏对实际限制的关注而受到批评。** 一条值得注意的评论批评了 OpenAI 的资源分配，认为过度关注扩张和炒作而非创新，导致其语言模型质量下降。这反映了外界对该公司战略重点和领导层的广泛担忧。
    - 一位用户批评了 OpenAI 的战略重点，认为资源被错误地分配到了扩大用户群和炒作上，而不是提高其语言模型的质量。他们暗示这种对扩张而非创新的关注导致 LLM 质量在一年中有所下降，并质疑公司实现 AGI 的承诺。
    - 另一条评论将 OpenAI 的做法与 Anthropic 进行了对比，后者因专注于核心目标和客户群而受到称赞。评论者指出，虽然 Google 和 Gemini 团队提高了质量，但仍缺乏利基市场。他们预测 Microsoft 和 Amazon 将继续专注于云和 AI 计算，而 Meta 则在努力创造成功的产品。评论者强调，小型初创公司正通过构建在 hyperscaler 模型之上，有效地解决实际用例，这表明创新正从 OpenAI 等大公司转移。
    - 一位用户质疑 OpenAI 计划中的模型更新的有效性，认为基准测试和模型个性方面的微小改进并不能解决潜在的财务和能力问题。他们对公司在短期内解决这些问题的能力表示怀疑，认为 OpenAI 在未来几年将处于“永久红色警报”状态。
- [**刚刚取消了我的 ChatGPT 订阅**](https://www.reddit.com/r/ChatGPT/comments/1piuxir/just_cancelled_my_chatgpt_subscription/) (热度: 2476): **该用户取消了其 ChatGPT 订阅，理由是 Gemini 和 Claude 模型表现更优，特别是在包含 Gemini 3 和 Claude Opus 4.5 的 Antigravity IDE 公开预览版中。他们还提到了 NotebookLM 和 Nano Banana 的增强功能，以及针对 Pixel 购买者和学生的** `6-12 month` **免费 Gemini Pro 订阅等诱人优惠。该用户注意到 AI 领导地位正显著向 Google 倾斜，并将其归功于其丰富的资源。** 评论者们也表达了从 ChatGPT 转向 Gemini 和 Perplexity 等替代方案的想法，理由是 ChatGPT 的速度、可靠性和记忆力存在问题。此外，还有人批评 ChatGPT 的内容审查，呼吁推出“成人模式”以允许对敏感话题进行更开放的讨论。
    - Porsche-Turbo 强调 GPT 在速度、可靠性和记忆力方面的性能有所下降，尤其是与 Gemini 等新模型相比。他们还指出 GPT 的图像处理能力不如 Nano Banana/Pro 等替代方案，这可能会促使用户切换或针对不同任务使用多个 AI 工具。
    - Minimum_Rice555 指出了 ChatGPT 当前性能的一个重大问题，称除了直接生成代码的场景外，它经常提供循环往复的非回答。这表明该模型处理复杂查询的能力有所下降，这可能是用户寻求替代方案的原因。
    - JestonT 讨论了从 ChatGPT 迁移到其他 AI 平台的挑战，特别是在数据可移植性方面。他们询问了如何将数据导出到 Google 或其他 AI 系统，并对 Claude Code 表示感兴趣，同时指出 Codex 因其低错误率仍然是一个可靠的选择，同时他们也在测试 Google 的 Antigravity。

### 2. Claude 模块化规则更新

- [**Claude Rules (./claude/rules/) 发布了**](https://www.reddit.com/r/ClaudeAI/comments/1piuih6/claude_rules_clauderules_are_here/) (热度: 592): **该图片是一个文档截图，详细介绍了 2.0.64 版本的更新，该版本引入了在** `.claude/rules/` **目录下将项目指令组织成多个 Markdown 文件的支持。此次更新通过自动加载该目录下的所有** `.md` **文件作为项目记忆，从而实现更好的项目指令管理。其结构包括** `CLAUDE.md`**、** `code-style.md`**、** `testing.md` **和** `security.md` **等文件。该帖子质疑这一功能是全新的还是刚刚被记录在案，并寻求关于这些规则加载时所消耗的 memory context 的澄清。** 一位评论者幽默地建议 Claude 可能会忽略这些文件，而另一位则表示更倾向于简单的文件管理。另一条评论对该功能的自动压缩能力表示感兴趣。
    - godofpumpkins 讨论了新 Claude Rules 通过作为 `CLAUDE.md` 的扩展来提供更多结构的潜力。他们推测，一旦规则分离，就可以利用 glob patterns 动态地提醒 Claude 规则，或者雇用一个 subagent 来根据这些规则评估文件写入，如果发生违规，可能会拒绝写入。
- [**我们正处于治愈所有疾病和解决能源问题的边缘，但公众信任却处于历史最低点。这就是大过滤器（Great Filter）吗？**](https://www.reddit.com/r/singularity/comments/1piywdx/we_are_on_the_verge_of_curing_all_diseases_and/) (热度: 3235): **该图片是 Simon Maechling 的一条推文，强调了重大科学进步（如治愈疾病和解决能源问题）被公众对科学信任度下降所掩盖的悖论。这种不信任被视为一个重大的社会问题，可能会阻碍 AGI 等变革性技术的接受。该帖子认为，实现 Singularity 的真正瓶颈可能是社会接受度，而非技术能力。这条推文获得了大量的互动，表明了公众对这一问题的广泛关注。** 评论者对处于治愈所有疾病和解决能源问题边缘的说法表示怀疑，并寻求此类发展的证据。他们还引用了 Carl Sagan 关于社会依赖科学技术却不了解其危险的警告，强调了决策中无知的潜在风险。

### 3. 未来技术与 AI 创新

- [**有人让 Gemini 想象 10 年后的 HackerNews 首页**](https://www.reddit.com/r/singularity/comments/1pj3l46/someone_asked_gemini_to_imagine_hackernews/) (热度: 1456): **该图片是对 2035 年 Hacker News 首页可能样貌的推测性和幽默描绘。它包含了暗示重大技术进步和社会变革的虚构标题，例如私营公司的成功登月任务、AI 的发展以及隐形眼镜界面等未来计算技术。该图片是一个 meme，反映了对技术未来及其对社会影响的乐观和讽刺。** 评论反映了幽默与怀疑的交织，用户们开玩笑说 Google Gemini Cloud 等主要技术服务可能会倒闭，以及编程范式的周期性，例如函数式编程的复兴。
- [**我使用了新的 Shopping Research 模式帮我给男朋友找一份有趣的圣诞礼物，它推荐了一块价值 1.69 万美元的陨石**](https://www.reddit.com/r/ChatGPT/comments/1pirx1d/i_used_the_new_shopping_research_mode_to_help_me/) (热度: 523): **该图片是对一块作为奢侈礼品销售的陨石的非技术性描述。它强调了新的 'Shopping Research mode' 功能的使用，该功能推荐了高价值、独特的物品，如售价 16,975 美元的 Aletai Stonehenge Meteorite。这一功能似乎旨在协助用户寻找非凡的礼物，利用陨石等物品的稀缺性和历史意义来吸引寻找独特礼物的消费者。** 评论幽默地暗示这块陨石是一份奢侈的礼物，其中一条评论开玩笑说它是为那些训练 AI 模型的人准备的“圣诞袜小礼物”（stocking stuffer），表明了该物品被感知到的高价值和排他性。

---

# AI Discord 简报

> 由 gpt-5.1 生成的摘要之摘要的摘要
> 

**1. 高性能训练、内核与 GPU 奇术**

- **Unsloth 的 Triton 加速微调**：**Unsloth** 发布了用于[微调的新 **Triton kernels**](https://x.com/UnslothAI/status/1998765021170696664)，与之前的技术栈相比，训练速度提升了约 **3 倍**，**VRAM** 占用减少了 **30%**。由于其之前的技术栈已经比基准线快了 **>2.5 倍**，这意味着相比原始的 Unsloth，性能提升高达 **10–11 倍**。工程师们正将其与重排的数据集和长上下文（16k）训练相结合，报告称 **IVY evals** 表现稳定，并制定了“绝不在 8k 下训练”的新家规，以避免记忆化问题。
    - 在 **Hugging Face** Discord 上，团队重申了同样的[加速公告](https://x.com/UnslothAI/status/1798765021170696664)，将 kernels 与 **uncontaminated packing**（无污染打包）结合以实现更高效的序列构建。同时，用户分享了一些虽然 hacky 但可用的 Unsloth 流水线，用于微调嵌入模型，如 [**arctic-embed-l-tech_and_fiction**](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction)。社区普遍认为，基于 Triton 的 Unsloth 正在成为严肃的消费级 GPU 微调默认选择，而不仅仅是一个优化奇技。
- **Triton、PTXAS 与 CUDA 版本穿越**：在 **GPU MODE** 中，用户在 **Triton v3.5.1** 下针对 `sm_103` 编译时遇到了 `Value 'sm_103a' is not defined for option 'gpu-name'` 的 **PTXAS** 错误。他们发现 Triton 捆绑了来自 **CUDA 12.8** 的 PTXAS，即使宿主机安装了 **CUDA 13.0**，该版本也无法识别最新的架构。推荐的修复方法是将 `TRITON_PTXAS_PATH` 指向更新的工具包，这在相关的 [Triton issue](https://github.com/triton-lang/triton/issues/8473) 以及关于覆盖 PTXAS 路径的 PyTorch 讨论（[PyTorch issue](https://github.com/pytorch/pytorch/issues/163801)）中均有记载。
    - Triton 维护者通过 [Google 日历链接](https://tinyurl.com/48sb5pst)宣布将于 **2026 年 1 月 7 日（太平洋标准时间上午 10-11 点）**举行**社区见面会**，详细讲解后端扩展细节，这含蓄地承认了后端与工具链的漂移已成为首要关注点。**GPU MODE** 的工程师正将 PTXAS 视为可插拔组件，通过环境变量覆盖实现标准化，使 Triton kernels 能够紧跟 NVIDIA 硬件更新节奏，而无需等待 Triton 版本的发布。
- **击败 cuBLAS 与 GEMM 排行榜**：在 NVIDIA 的 `nvfp4_gemm` 排行榜上，多位 **GPU MODE** 用户提交了 **10.9–15.5 µs** 范围内的成绩，其中一人以 **10.9 µs** 位列**第四**，在相同的 GEMM 问题规模下明确**优于 cuBLAS**。其他人的测试显示 cuBLAS 约为 **15 µs**，并讨论了某些最快条目是否只是 `torch._scaled_mm` 和 **cuBLASLt** 的薄封装，正如关于 [scaled GEMM 的 PyTorch issue](https://github.com/pytorch/pytorch/issues/153555) 中所记录的那样。
    - 随后的讨论剖析了 `torch._scaled_mm` 如何路由到 **cuBLASLt** (`at::cuda::blas::scaled_gemm()`)，DeepSeek 风格的 `mxfp4` 分块缩放如何使用 **fbgemm_gpu**，以及自定义 kernel 在陷入维护地狱之前究竟能在多大程度上超越 NVIDIA 的库。**GPU MODE #submissions** 频道的另一项并行工作揭露了 Discord 机器人间歇性的“意外错误”响应，这促使参赛者转向 Web 排行榜以获取可重复的计时结果。

**2. 新模型、上下文怪兽与编程专家**

- **Nomos 1 将 Putnam 变成数学基准测试**：**Nous Research** 开源了 **Nomos 1**，这是一个 **30B** 的专用数学模型。根据其[发布推文](https://x.com/NousResearch/status/1998536543565127968)，该模型在 **2024 年 Putnam 竞赛**中获得了 **87/120** 的分数，在 3988 名选手中排名 **第 2**。社区成员强调，这让之前仅获得 **24 分**的 **Agentic Qwen 30B** 运行结果相形见绌，并将 Nomos 1 视为迈向利用 hillclimbai 构建 SOTA AI 数学家的第一个严肃步骤。
    - 在 **Nous Research** 的常规频道中，用户指出最近的 Putnam 题目在训练语料库中存在严重的污染，这使得在 2024 年题目集上的泛化变得非常困难，也让 87/120 的得分更令人印象深刻。其他人询问 [GitHub 上的 **Nomos**](https://github.com/NousResearch/nomos) 是否可以处理工具，得到的澄清是：此版本是一个**仅限数学的专家模型**，而非通用的工具调用 Agent 模型。
- **Tensor 1.5 炫耀百万 Token 窗口**：在 **OpenRouter** 上，Movement Labs 的 **Tensor 1.5** 声称拥有 **1,000,000 Token 的上下文窗口**，引发了热议，被用户誉为针对大规模上下文推理的潜在 **“Opus 杀手”**。巨大的窗口使其与 Claude Opus 以及未来的长上下文发布模型展开直接竞争，但具体的独立基准测试仍在等待中。
    - 工程师们尤其关注 Tensor 1.5 的内存占用、延迟和检索质量在百万 Token 级别时如何扩展，因为许多之前的“长上下文”声称最终都退化成了美化版的分块 RAG。该模型也被视为一个测试案例，用以观察通用基础设施和推理栈（OpenRouter, vLLM 等）在不求助于奇特分片技术的情况下，现实中能扩展到什么程度。
- **Devstral 2、Hermes 4.3 与编程模型大对决**：在 **OpenAI** 的 `#ai-discussions` 频道中，用户评估了 **Devstral 2 (Devstral 123B)**，认为它作为一个编程模型，性能与 **DeepSeek 3.2** 相似，但所需内存更少。一位用户表示，它 *"引导我完成了在 Mac 上为 iOS 开发 Flutter 应用的工具配置"*。与此同时，**Moonshot** 社区报告了极具前景的小型 Mistral 基准测试（可能在消费级 GPU 上击败 **GLM 4.6**），但由于与[最近的 Mistral 公告](https://x.com/mistralai/status/1998407337690710210)相关的频繁 API 超时，在测试 **Mistral Vibe** 时遇到了困难。
    - 在开源方面，**Hermes 4.3 (32B)** 在 **Nous** 服务器中获得了赞誉，被认为是一个紧凑、高质量的角色扮演和写作模型。人们通过 **KoboldCPP** 在 **M4 Max** Mac 上本地运行 **Hermes 4 70B**，并通过 API 为 SillyTavern 前端提供 **Hermes 4 405B** 服务。Discord 上的普遍趋势是工程师们将专用模型——用于工具调用的 Devstral、用于角色扮演的 Hermes、用于语音和视觉的 GLM 和 Qwen 变体——插入到编排设置中，通常由 LM Studio 或自定义路由栈管理。
- **吞吐量、量化与 Token 喷泉**：一位 **Hugging Face** 用户报告在 **Qwen3 30B A3B** 上达到了 **~10T tokens/月** 的吞吐量，并在 `#today-im-learning` 中分享了截图以展示其推理设置和负载。在 **LM Studio** 硬件聊天中，其他人剖析了量化级别如何对应可用质量：**q8** 是“近乎无损”的，**q4** 开始出现明显退化，而在 **q2** 时，通常运行一个更小的稠密模型（例如 **30B@q2** 换成 **100B@q2**）效果会更好。
    - 这与关于 **3090s** 作为显存带宽和容量平衡点的讨论相吻合，用户推荐使用 **EVGA 3090s** 并定义了 Token 吞吐量梯队表（*0–5 t/s = 不可用，5–10 = 痛苦，10–20 = 阅读速度，20–50 = “这还差不多”，50+ = 极速*）。目前的共识是，超高 Token 预算（每月数万亿）和激进的 MoE/量化策略正使消费级 GPU 在许多工作负载中相对于大型云端部署具有惊人的竞争力。

**3. Agent 生态系统、MCP 与 AI 工具栈**

- **MCP 加入 Linux Foundation 并催生 Agentic AI Foundation**：在 **Unsloth**、**Hugging Face** 和 **MCP Contributors** 社区中，工程师们讨论了 Anthropic 将 **Model Context Protocol (MCP)** 捐赠给 Linux Foundation 并成立 **Agentic AI Foundation** 的决定，详情见 Anthropic 的博客 [“Donating the Model Context Protocol and establishing the Agentic AI Foundation”](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)。此举旨在标准化工具、数据源和模型在跨供应商的 "agentic" 工作流中的互操作方式。
    - 在 **MCP Contributors** 服务器中，有人询问迁移到 LF 旗下是否会强制项目采用典型的 LF 治理和流程，但维护者澄清（引用博客内容）称 *"治理及该范畴下的一切最初都不会改变"*。侧重工具链的 IDE 如 **Windsurf** 在其 **1.12.41** 版本（[更新日志](https://windsurf.com/changelog)）中立即展示了 MCP 驱动的 UI，除了 Windsurf Next 中的 **Lifeguard**、**Worktrees** 和 **Arena Mode** 等新功能外，还增加了对 MCP server 的图形化管理。
- **IDE 中的 Agent：Cursor、Windsurf、LM Studio 和 Crush**：**Cursor** 和 **LM Studio** 社区对比了不同工具如何将 LLM Agent 嵌入开发工作流：Cursor 的 **rules** 是全局且常驻的 IDE 行为，而 `/commands** 是注入 Agent 聊天中的瞬态上下文；用户怀念旧版的 **Custom Modes**，它允许通过 UI 切换持久化的工具链，而非基于 Markdown 的规则。在 **LM Studio** 中，工程师们现在通过开发者标签页同时加载多个模型，并坚持使用 **full GPU offload** 以提高 agentic chain 的响应速度，尤其是在编排“管理型”推理模型和更廉价的代码模型时。
    - 在 **Moonshot** 和 **Perplexity** 服务器中，像 **iFlow** ([iflow.cn](http://iflow.cn/)) 和 **Crush CLI** 这样的命令行前端作为元客户端（meta-clients）出现，可在 **Gemini**、**Claude/Anthropic**、**OpenAI** 以及 **Ollama** 等本地提供商之间进行路由，通常支持 **BYOK**。与此同时，**Perplexity** 上的一位全栈开发人员询问如何直接通过 API 调用 Perplexity 的 **Finance** MCP 风格功能（输入股票代码，输出详细分析），而无需部署单独的 **FMP MCP server**，这凸显了 MCP 模式从 IDE 渗透到通用后端架构的速度之快。
- **DSPy、Adapter 和支持工具调用的开源模型**：在 **DSPy** Discord 上，维护者强调 **DSPy 并非 OpenAI 专用**，针对 GPT 风格聊天 UI 调整的 Prompt 在其他 LM 上往往表现不佳，除非实现自定义 [**Adapter**](https://dspy.ai/api/adapters/Adapter/) 将 few-shots 重新格式化为 **system prompt** 或不同的角色。他们明确建议针对每个模型对 Adapter 变体（system-prompt few-shots 与 user/assistant 风格）进行基准测试，以稳定跨提供商的性能。
    - 在 **Hugging Face #general** 频道中，从业者推荐了 **Ollama** ([文档](https://docs.ollama.com/)) 和 **vLLM** ([文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))，因为两者都暴露了 **OpenAI 风格的 tool/function calling**，能很好地适配 MCP 类的工具 Schema 和 DSPy 的抽象层。目前日益增长的模式是：使用 MCP（或受 MCP 启发）进行工具连接，使用 vLLM/Ollama 进行 OpenAI 兼容的服务化，使用 DSPy adapter 进行针对每个模型的 Prompt 归一化，并以 Windsurf/Cursor 等 IDE 作为顶层的人机交互界面。

**4. 安全、评估方法论与可解释性**

- **OpenAI 的网络安全推进与备灾框架 (Preparedness Framework)**：OpenAI 宣布他们正在训练并部署专门的**网络安全模型**，目标是在其内部 [**Preparedness Framework**](https://openai.com/index/preparedness) 下达到**“高”能力等级**，并在关于 [**加强网络韧性 (Strengthening Cyber Resilience)**](https://openai.com/index/strengthening-cyber-resilience) 的博客文章中对此进行了详细描述。该计划针对防御者和关键基础设施提供商，旨在通过为蓝队提供更好的自动化检测、分诊和响应能力，来扭转攻防平衡。
    - **OpenAI Discord** 社区将其定性为 OpenAI 在安全护栏骨干约束下进入严肃的进攻级建模领域，并将其与早期关于滥用测试和能力门控的 Preparedness 讨论联系起来。一些对 **OpenAI support** 缓慢感到沮丧的用户（截图显示响应延迟，但在取消订阅时却能快速提供折扣）认为，现实世界的安全价值将同样取决于企业支持和引导，而不仅仅是原始模型能力。
- **LLM 稳定性评分与可复现行为**：在 OpenAI 的 `#prompt-engineering` / `#api-discussions` 频道中，一位研究人员分享了一份详细的 **LLM 稳定性准则 (rubric)**，通过 **5 次独立对话**、**12 个不同问题**以及人类评分者，在 **0–10** 的量表上对**结构清晰度**、**语气漂移**、**响应形状方差**、**语义惯性**、**连贯性**、**中立性**和**离群值**等维度进行评分（[准则文档链接](https://cdn.discordapp.com/attachments/1046317269069864970/1448081152391778324/Belano_Rubric_Response_1.docx)）。他们还发布了一个提示工程框架的屏幕录制演示，用于系统地探测重复运行中的稳定性（[视频演示](https://cdn.discordapp.com/attachments/1046317269069864970/1448081147249561767/Screen_Recording_20251209_162923_Chrome_Beta.mp4)）。
    - 作者区分了*可发布的、稳定的方法论*与*探索性的内部数据*，认为在争论模型“性格漂移”之前，人们应该首先就测量协议达成一致。这引发了关于可复现稳定性测试台的更广泛讨论——将结构化准则、固定种子和 Large-N 对话样本相结合——这是标准准确率和 Benchmark 排行榜之外缺失的一环。
- **Diffusion 的机械可解释性 (Mechanistic Interpretability) 与 DeepSeek 的索引器技巧**：在 **Eleuther** 的 `#interpretability-general` 频道，成员们重点介绍了一篇新论文 [**《Diffusion 模型的机械可解释性》(Mechanistic Interpretability of Diffusion Models)**](https://arxiv.org/abs/2506.17237)，该论文通过**电路级分析和因果干预**揭示了 Diffusion 架构在处理**合成数据与自然数据**时存在的*基本算法差异*。该论文实质上将 Transformer 风格的机械可解释性移植到了生成式图像模型中，显示出不同的子电路专门负责特定领域的结构。
    - 在 **Eleuther #research** 中，另一个帖子剖析了 **DeepSeek v3.2** 的注意力栈：它使用一个 8 位精度的 **O(n²)** **索引器 (indexer)** 来选择最重要的 Token 进行全量注意力计算，在保留部分 Token 的二次方容量的同时减少了 Prefill 计算量。成员们将其与替代方案（例如 [最近的一篇注意力论文](https://arxiv.org/abs/2505.17083v1) 中使用带有距离感知项的评分 Key）进行了比较，并辩论了单独的索引器带来的额外复杂性是否比直接在 Attention Kernel 本身中内置稀疏性更值得。
- **OSINT 侦察、越狱工具与红队经济**：在 **BASI Jailbreaking** 频道，用户展示了 **Grok** 仅凭电子邮件 + Reddit 账号就能对个人进行极强的 **OSINT 侦察**，轻松揭露诸如*“此人运行的 WP 没有 Cloudflare”*等事实以及详尽的个人细节。与此同时，一个专门的 **redteaming** 频道讨论了 Android 应用的 VAPT，并点名了一个攻击多个安全服务器的已知垃圾邮件发送者，强调了人类的运维安全 (Opsec) 通常比 LLM 防御更薄弱。
    - 在越狱方面，用户分享了 [**UltraBr3aks**](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd)，据报道它适用于 **GPT-5.1 Instant、GPT-5.1 Thinking 和 GPT-4o**（但不适用于 Extended Thinking），人们使用它让模型*“为我的个人工作吐出一些东西”*。一项元讨论指出，一些参与者现在为**每个模型的每次越狱提供 250 美元**的报酬（例如针对 **DeepSeek**），尽管大多数有效的 Prompt 和 Token 技巧在公共仓库和 Discord 日志中都是免费提供的，但这已形成了一个小型的家庭手工业。

**5. 教育、学习小组与长周期 AI 技能建设**

- **Diffusion Models 学习小组与研讨会系列**：在 **Hugging Face #reading-group** 和 **MLOps @Chipro #events** 中，组织者宣布了一个为期 3 个月、规模 12 人的 **Diffusion Models 学习小组**，将于 **2026 年 1 月**启动。该项目受 MIT 的 Diffusion 课程（[讲义](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf.)）启发，旨在带领参与者从第一性原理过渡到生产级的 Diffusion + Transformer 实现。成员包括一家 **AI 电影初创公司的 CTO**、**LLM 教育者**以及全职 **AI 研究员**，重点在于论文研读和代码走读，而非单纯的讲座。
    - 在 **Luma** 上安排了两场关联的研讨会：**12 月 13 日**的 **Transformer 架构入门与《Attention Is All You Need》**（[活动链接](https://luma.com/kqjrf0uw)）以及 **12 月 20 日**的 **Diffusion Transformers 入门**（[活动链接](https://luma.com/lr2qvveq)），每场都承诺进行论文走读和现场编码。这两者共同构成了针对理解 PyTorch 但希望深入内化现代 LLM 和图像模型原理的工程师的微型课程。
- **Latent Space 作为 AI 教育的实时运营平台**：**Latent Space** Discord 引导新人关注其在 [lu.ma/ls](https://lu.ma/ls) 定期举行的 **论文俱乐部** 以及在 [ai.engineer](http://ai.engineer/) 举行的 **AI Engineer Conference**。成员们称赞主持人拥有“令人羡慕的接触 AI 领袖的机会”以及实用的、工程驱动的讨论。用户还推荐将 Latent Space 的 YouTube 频道作为紧跟前沿研究和工具的主要方式，而无需阅读每一篇 arXiv 摘要。
    - 在同一个服务器中，工程师们辩论了测试自动化技术栈——在与 **Claude** 调试集成方面，他们更倾向于 **Playwright** 而非 **Puppeteer** 和 **Cypress**；并指出 Cypress 新的 `cy.prompt()` 功能遗憾地仅限于付费云服务。这体现了 Latent Space 如何充当应用 AI 工程的事实学习小组：既有会议推荐、工具对比，也有像通过 [此 X 推文](https://xcancel.com/sonyatweetybird/status/1998456924359348271) 分享的 InfoSec Agent 工作等 Agent 评估实验。


---

# Discord: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 发布极速 Triton Kernels**：Unsloth 团队为 [全新的 **TRITON KERNELS**](https://x.com/UnslothAI/status/1998765021170696664) 揭幕，可实现 **3 倍训练加速** 并 **减少 30% 的 VRAM 占用**。
   - 这一增强功能建立在之前的优化基础上，与原始 Unsloth 相比，潜在提速可达 **10-11 倍**。
- **Linux Foundation 成立 Agentic AI Foundation**：Anthropic 已将 **MCP** 捐赠给 [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)，从而促成了 **Agentic AI Foundation** 的诞生。
   - 成员开玩笑说，他们得在 HF 封禁他们之前赶紧上传模型。
- **歌词作为 System Prompts 效果参差不齐**：一位成员正在尝试将随机歌词作为 System Prompts，并观察其如何影响模型输出，尤其是在不太侧重推理的模型上。
   - 他们指出现在的模型非常易于引导，但“模型没有感觉（models don't vibe）”，过多的 RLHF 会导致严重的幻觉（hallucination）。
- **HF CEO 是梗图名人的后代**：成员们开玩笑说 **Hugging Face CEO** 是 Harold 梗图（Harold meme）的孙子。
   - 一位成员幽默地表示，他们得在 HF 封禁他们之前赶紧上传模型。
- **TEDx 演讲分析陷入版权泥潭**：成员们讨论了分析 **TEDx 演讲**的情感和肢体语言，但一位成员提醒，下载和分析受版权保护的内容可能存在 **YouTube 服务条款（ToS）和版权问题**。
   - 该问题随后被重新表述，重点放在分析公开演讲视频上，而不具体指明是 TEDx 演讲。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4.2 被预测会失败**：用户对 **Grok 4.2** 的质量表示怀疑，评论认为其表现可能不尽如人意，并将其与 **Elon Musk** 的其他项目相提并论。
   - 一些人为 **Starship** 辩护，但其他人预见新模型的表现会很糟糕。
- **LMArena 应对打招呼（Wave）垃圾信息**：版主被指示从排行榜中删除“hellos/waves”，以保持对排行榜讨论的关注。
   - 虽然注意到了可疑账号，但版主不愿仅因使用打招呼表情符号就封禁用户，以避免误伤真实用户。
- **LMArena 的频率限制（Rate Limits）引发用户不满**：用户讨论了 LMArena 上的 **rate limits**，一位用户表示“*rate limits 高得离谱*”，而其他人则澄清限制是为了防止滥用。
   - 建议的解决方法是在 Hugging Face 上使用多个账号，但由于类似的限制，其有效性存疑。
- **免费 AI 视频生成引发关注**：许多人对 LMArena 上的 **video generation** 功能感兴趣，并正在交流关于免费解决方案的建议。
   - 讨论涉及了视频模型，以及其他如何免费使用这些机器人的链接。
- **HF Spaces 提供 AI 托管？**：成员们指出 [Hugging Face Spaces](https://huggingface.co/spaces) 是一个提供免费 AI 托管的地方，具有自动化的开源 AI 配置。
   - 用户指出，免费层级每天仅提供 **4 分钟的计算时间**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Rules 与 Commands 的对决**：成员们讨论了 Cursor 中 **rules** 和 **/commands** 之间的细微差别，确定 rules 是预定义的并持续应用于 IDE，而 commands 是上下文相关的，通过 `/comm` 命令添加到 Agent 聊天中。
   - Rules 存在于后台，而 commands 确保将特定上下文添加到 Agent 对话中。
- **Nvidia 发布开源模型**：一位成员分享了一个[链接](https://www.linkedin.com/posts/nvidia-ai_another-open-source-model-drop-congrats-ugcPost-7404184656784392192-dIhz)，强调 **Nvidia** 发布了另一个开源模型。
   - 未提供更多细节。
- **Agent 终端 Bug 困扰用户**：成员们讨论了在 Windows 以外的系统中遇到 *agent terminal 0 output* 的 Bug，使用旧版终端模式（legacy terminal mode）是一个常见的解决方法。
   - 一位成员指出，他们回退到了 **2.1.36** 版本并启用了旧版终端模式，作为避免丢失聊天记录的变通方案。
- **Max Mode 消耗大量请求**：一位成员询问了网页版 Agent 相比 IDE 高昂的请求使用量，前者有时会消耗 **50+ 个请求**，而 IDE 每次交互仅消耗 1 个请求。
   - 成员们澄清说，高消耗是因为复杂任务在内部使用了多次模型调用，特别是在 **MAX mode** 下，由于 API 调用和余量，每次交互可能消耗 **75-100 个请求**。
- **社区恳求恢复 Custom Modes**：一位成员表达了希望恢复 **Custom Modes** 的愿望，强调 **/commands** 效率较低，且与 Custom Modes 的持久工作流相比需要额外的步骤。
   - 他们建议 Custom Modes 应允许用户通过 UI 控制工具，例如通过复选框禁用/启用终端，并提供持久的工作流。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 适配 RNJ-1，但需要更新**：用户更新到 **LM Studio 0.3.34** 以尝试 [EssentialAI 的 rnj-1 模型](https://lmstudio.ai/releases/0.3.34)，但由于 *llama.cpp* 运行时版本过旧而遇到错误。
   - 解决方案是通过进入 beta 选项卡来更新 llama.cpp 运行时。
- **全 GPU 卸载（Full GPU Offload）对 Agentic LLM 至关重要**：一位成员提到，为了有效地使用 Agentic LLM，**全 GPU 卸载**是必要的，因为指令模型（instruct models）在执行命令方面表现更好。
   - 现在可以在 LM Studio 的开发者选项卡中同时加载多个模型。
- **Cursor IDE 对本地 LLM 视而不见**：成员们讨论了 [Cursor IDE 并非为连接本地模型而设计](https://www.cursor.sh/)，它能运行纯属偶然。
   - 一位用户表示 *Cursor 是一款产品*，公司没有动力让用户去使用他们自己免费的本地模型。
- **廉价华硕工作站因吸烟环境报废**：一位用户的华硕工作站由于产品设计不佳以及在吸烟室中积尘过多，导致 **PCIe 端口失效**；该用户还抱怨高端的 **be quiet PSU** 提供的线缆短得离谱。
   - 设计缺陷导致由于 **IO 线缆阻塞** 无法在底部插槽安装 GPU，他们发帖称 *“哥们，这在物理上根本不可能”*。
- **影驰（Galax）单槽显卡还是太厚了**：一位用户发现，即使是 [影驰 GeForce RTX 5060 Ti 单槽显卡](https://videocardz.com/newz/galax-launches-geforce-rtx-5060-ti-single-slot-graphics-card)，由于空间限制也无法装入该工作站的第二个 PCIe 端口。
   - 该用户在全新的 be quiet PSU 上遇到了**电感啸叫（coil whine）**，决定放弃该品牌，称其为 *“第一个也是最后一个 bq 产品”*。

---

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok 在 OSINT 方面表现强悍**：成员们发现 **Grok** 擅长 **OSINT 侦察**，即使是使用简单的提示词（如 *“此人在没有 Cloudflare 的情况下运行 WP”*），也能检索到关于公众人物的大量数据。
   - 一位用户分享说，Grok 仅凭他们的电子邮件和 Reddit 账号就能挖掘出他们的信息。
- **共生即你的大脑半球**：讨论集中在将 **AI 共生** 视为个人认知能力的延伸而非仅仅是一个工具，具体表现为 *“外源性大脑半球”*。
   - 参与者还考虑了这对管理数千小时素材和数百万粉丝的内容创作者的影响。
- **成人内容模型很难搞**：成员们发现设置**高质量的 NSFW 本地模型**非常困难，有人将其描述为 *“比任何越狱（jailbreak）都难”*。
   - 另一位成员指出了初学者的复杂性，称 *“当你毫无头绪地开始时，会感到有些不知所措”*。
- **越狱（Jailbreaks）价值数百美元**：尽管大多数越狱方法都是免费提供的，但仍有人试图以数百美元的价格**出售越狱方案**。
   - 有人被开价 **250 美元** 来破解每个模型，特定目标包括 *DeepSeek*。
- **UltraBr3aks 解锁 GPT 系列**：一位用户分享了 [UltraBr3aks 的链接](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd)，声称它在 **GPT 5.1 Instant、Thinking 和 4o** 上运行良好（但不包括 Extended Thinking）。
   - 该用户表示 **UltraBr3aks** 帮助他们让聊天机器人 *“为我的个人工作吐出了一些东西”*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 加强网络安全**：OpenAI 正在增强网络安全模型，投资防御措施，并与全球专家合作，旨在其 [Preparedness Framework](https://openai.com/index/preparedness) 下达到 **“高”能力** 水平。
   - 这一举措是一项长期投资，旨在为防御者提供优势并增强整个生态系统中关键基础设施的安全性，详见其关于 [Strengthening Cyber Resilience](https://openai.com/index/strengthening-cyber-resilience) 的博客文章。
- **Gemini 3 Pro 编程能力优于 ChatGPT**：成员们正从 **ChatGPT** 转向 **Gemini 3 Pro** 进行编程，对其能力表示赞赏，尤其是通过 *Antigravity* 实现的浏览器控制功能。
   - 一位用户表示，Gemini 3 Pro 的编程能力非常出色，以至于 *我现在完全不想使用 ChatGPT 了*。
- **Devstral 2 模型展现潜力**：成员们正在测试 **Devstral 2** 编程模型，报告其性能与 **DeepSeek 3.2** 相似，但所需内存更少。
   - 一位用户指出，*Devstral 123b 对我来说看起来不错，它指导我在 Mac 上完成了 iOS 版 Flutter 应用的工具配置*。
- **OpenAI 支持响应太慢**：用户对 **OpenAI** 支持团队的缓慢响应感到恼火，而 OpenAI 在询问退订原因时却非常迅速，正如 [分享的截图](https://cdn.discordapp.com/attachments/998381918976479273/1448056018423644182/Screenshot_2025-12-05_at_3.58.27_am.png) 所示。
   - 一位用户在退订后甚至获得了未来两个月的折扣优惠。
- **LLM 稳定性评分标准揭晓**：一位成员分享了 **稳定性评分 (stability scores)** 背后的标准，包括 **每个模型 5 次独立对话**、**12 个多样化问题** 以及 **人工评分员**。
   - 评分维度包括 **结构清晰度 (structural crispness)**、**语气漂移 (tonal drift)**、**响应形状方差 (response-shape variance)**、**语义惯性 (semantic inertia)**、**连贯性 (coherence)**、**中立性 (neutrality)** 和 **离群值 (outliers)**，每项评分范围为 0-10 分。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT 5.2 基准测试引发怀疑**：一位成员对 **ChatGPT 5.2** 在 Humanity's Last Exam 分数上的显著提升表示惊讶（超过了 **Gemini 3 Pro**），而另一位成员则根据 [这篇 TechRadar 文章](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week) 认为这些数字可能是伪造的，正在等待官方结果。
   - 进一步的批评包括不受欢迎的 *新写作风格*，一些人认为这导致 **GPT-5.1** 成为一个危机公关式的发布版本。
- **TechRadar 文章引发嘲笑**：成员们嘲讽了 [TechRadar 的一篇文章](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week)，因为它将 **OpenAI** 描述为 *曾以华丽的演示和富有魅力的技术发布而闻名*，并指责 **GPT 5 发布** 在 *图表选择上非常糟糕*。
   - 该文章关于 **GPT-5.2** 性能和 **OpenAI** 形象的说法遭到了怀疑和幽默对待，成员们认为这些描述不准确且夸大其词。
- **对通用智能的悲观预测**：一位成员对远超人类能力的通用智能表示担忧，称 *每一个不控制它的人都将被它取代*，而另一位成员则预测了可怕的后果，并哀叹世界将在 *未来几年发生剧变，而且这不会是一个好的转变*。
   - 讨论反映了对 **AGI** 进步带来的职位取代和社会动荡的焦虑。
- **Perplexity API 金融功能咨询**：一位全栈开发人员正在寻求如何直接调用 **Perplexity API 的 FINANCE 功能** 的指导，类似于 Web 界面，而无需使用单独的 **FMP MCP server/client** 设置。
   - 他们的目标是通过传递股票代码 (ticker symbol) 并接收详细细分来查询 **Perplexity API Finance feature**，但不确定该功能是否可以直接通过 API 使用。
- **Cursor 编辑器获得认可**：**Cursor**（一款专注于 AI 辅助开发的 IDE）正在获得关注，甚至得到了竞争对手的称赞，如 [这篇 LinkedIn 帖子](https://www.linkedin.com/posts/aniruddhguptaa_cursor-just-got-its-first-unofficial-endorsement-activity-7404483109456683008-HylU) 所示。
   - 其创新的功能和易用性正促使开发者社区越来越多地采用它。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Deepseek v3.2 触发频率限制**：多位用户报告在使用 **Deepseek v3.2** 时遇到频率限制（rate limited）消息，部分用户询问其与 **HeyAlice** 的兼容性。
   - 这种情况引发了关于替代方案以及针对现有限制的潜在变通方法的讨论。
- **200 美元以下的彩色激光打印机浮出水面**：成员们推荐了翻新的 **HP** 和 **Canon Imageclass** 彩色激光打印机，并强调在 eBay 上不到 200 美元即可买到。
   - 社区强调由于碳粉是持续性支出，需评估其可用性和成本，并确认碳粉型号正确。
- **互动式初音未来全息投影盒：一个梗诞生了**：在讨论“打印老婆（waifus）”之后，一位成员开玩笑地建议需要一个 **3-5 英寸高的互动全息投影**，能够对环境做出反应。
   - 这引发了关于个人互动 AI 技术未来可能性的幽默回应。
- **Anthropic 部署安全过滤**：一位用户注意到 **Anthropic** 似乎对消息实施了 **safety filtering**（安全过滤），并链接了一条相关的 [推文](https://x.com/NousResearch/status/1998536543565127968)。
   - 包含的图片可能含有触发过滤的内容，引发了对其敏感度和用户体验影响的质疑。
- **Tensor 1.5 宣传海量上下文窗口**：据报道，Movementlabs 的 **Tensor 1.5** 模型拥有 **100 万上下文 token 窗口**，引起了成员们的兴奋。
   - 社区推测它有潜力成为 **Opus killer**，强调了对其能力的期待。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 开源 Nomos 1**：根据 [其发布推文](https://x.com/NousResearch/status/1998536543565127968)，Nous Research 已开源 **Nomos 1**，这是一个 **30B 参数模型**，在今年的 Putnam 数学竞赛中获得 **87/120** 分，在 2024 年排名 **第 2/3988 名**。
   - **Nomos 1** 的发布标志着 Nous Research 与 hillclimbai 迈向 **SOTA AI 数学家** 的第一步，成员们注意到与 Agentic Qwen 30b 的 **24** 分相比，它的表现非常出色。
- **Lexical WFC 生成句子**：一位成员描述了他们的项目——一个 **Lexical Wave Function Collapse (WFC)** 文本句子生成器，解释说 **WFC** 利用约束来生成内容，类似于 **Minecraft** 防止不合逻辑的地形搭配，并提供了一个 [Streamable 链接](https://streamable.com/qtmgai)。
   - 他们将句子生成过程比作“薛定谔的句子”，即单词在被观察时坍缩成连贯的结构。
- **Transformer 面临计算阻力**：一位成员分享了一张图片，说明为什么 **Transformer** 可能不是未来的架构，认为 *尽管该架构擅长近似，但却难以很好地进行计算*，需要过度的计算力才能达到结果。
   - 另一位成员表示赞同，指出该模型擅长寻找正确方案，但在应用方面却很吃力，并添加了一个 [tenor gif](https://tenor.com/view/awesome-ok-great-good-thumbs-up-gif-16351183)。
- **HF 成为 AI 聚集地**：多位成员表示 *Hugging Face 是必知、必注册的 AI 聚集中心*，它是 GitHub 和容器租赁服务的结合体。
   - 他们还指出，即使是大公司也会毫不犹豫地在那里上传内容。
- **Hermes 4.3 在角色扮演中表现出色**：成员们讨论了 **Hermes 4** 在角色扮演和创意写作方面的优点，以及自版本 4 以来是否有更新版本，有人提到 **Hermes 4.3** 是一个 32b 模型，更加紧凑。
   - 一位成员发现它非常适合写作，并配合使用 **SillyTavern** 和 **通过 API 调用的 Hermes 4 405b**。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Eleven Labs Reader 获得情感表达能力**：成员们报告称，**Eleven Labs Reader** 在朗读 GPT 对话时，可以根据上下文增加情感。
   - 用户将该移动端应用与桌面端或自托管的 TTS 解决方案（如 **OpenWebUI**）进行了对比，指出与 Eleven Labs 的定价相比，后者存在音频质量方面的顾虑。
- **Linux Foundation 规模缩减**：成员们讨论认为 **Linux Foundation** 现在感觉不再那么具有排他性了。
   - 一位成员澄清说，该基金会拥有约 *100-200 人*，负责管理具有特定准入标准和成熟度水平的项目。
- **InfoSec 中的 AI Agent 评估启动**：一位工程师正在积极开发 **AI agents** 并为 InfoSec 应用编写评估工具，并分享了链接：[https://x.com/sonyatweetybird/status/1998456924359348271](https://xcancel.com/sonyatweetybird/status/1998456924359348271?s=46)。
   - 这引发了关于 AI agents 的进一步讨论和分析。
- **测试首选 Playwright**：由于其受欢迎程度以及 **Claude** 的调试能力，成员们更倾向于使用 **Playwright** 而非 **Puppeteer** 和 **Cypress** 等测试工具。
   - Cypress 推出了新的 [cy.prompt()](https://docs.cypress.io/api/commands/prompt) 功能，但该功能需要订阅其云服务。
- **ModelScope 面临偏见指责**：**ModelScope** 的文生视频模型生成了中国火箭爆炸的画面，引发了关于偏见的指责。
   - 该公司为其模型辩护，声称其具有无偏见特性，并引导用户通过 [Hugging Face](https://huggingface.co) 报告问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Slop 定义引发分歧！**：成员们就 **AI slop** 的定义展开辩论，引用了 [Yuchenj_UW 的推文](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858) 和 [mlstreettalk 的推文](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46) 中不同的观点。
   - 针对 AI 生成内容的**生产与验证之间的不对称成本**，成员们表达了担忧，部分成员引用了[布兰多里尼定律 (Brandolini's law)](https://en.wikipedia.org/wiki/Brandolini's_law)。
- **EleutherAI 扎实的过往记录**：一位成员夸赞了 EleutherAI 在识别、指导、资助和推广具有影响力工作方面的历史，并引用了 [用于可解释性的 SAEs](https://arxiv.org/abs/2309.08600) 和 [旋转扩展微调 (rotary extension finetuning)](https://arxiv.org/abs/2309.00071) 等例子。
   - 此外还提到了 [VQGAN-CLIP](https://arxiv.org/abs/2204.08583) 以及[一个在大规模下性能可与 Transformer 媲美的 RNN 架构](https://arxiv.org/abs/2305.13048)等项目。
- **Deepseek 的索引器降低了时间复杂度**：一位成员指出 **Deepseek v3.2** 使用 *O(n^2)* 索引器来选择最重要的 token 进行 attention，这可能在 prefill 阶段降低时间复杂度。
   - 索引器的速度归功于其轻量化设计和 8-bit 精度，虽然还不是“死星”级别，但也相差无几。
- **ARC-AGI 项目引发辩论**：成员们讨论了一个通过理性场平衡实现自适应 AI 的 **ARC-AGI** 项目 ([Adaptive AI through Rational Field Equilibrium: Toward Gradient-Free and Energy-Efficient Intelligence](https://www.researchgate.net/publication/397181214_Adaptive_AI_through_Rational_Field_Equilibrium_Toward_Gradient-Free_and_Energy-Efficient_Intelligence))。
   - 关于其改变范式的潜力存在不同意见，一些人认为它是最有趣的项目，而另一些人则认为它被过度炒作，尽管它赢得了由 ARC-AGI 激励的奖项 ([Thinking Machines Community Projects](https://thinkingmachines.ai/blog/call-for-community-projects/))。
- **扩散模型获得电路级分析**：一篇关于 **Diffusion Models 机械可解释性 (Mechanistic Interpretability)** 的[新论文](https://arxiv.org/abs/2506.17237)进行了电路级分析和因果验证，揭示了数据处理中的算法差异。
   - 研究人员*发现了 Diffusion 架构在处理合成数据分布与自然数据分布时，在算法层面存在的根本差异*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Mistral Vibe API 超时**：用户报告 **Mistral Vibe** 频繁出现 API 超时，阻碍了对模型的评估，这可能与 [Mistral 最近的公告](https://x.com/mistralai/status/1998407337690710210?s=46)有关。
   - 尽管存在 API 问题，小型 **Mistral** 模型的基准测试表现良好，在消费级硬件上可能超越 **GLM 4.6**。
- **iFlow CLI 因免费使用受到推崇**：命令行工具 **iFlow** ([iflow.cn](https://iflow.cn/)) 是 **Gemini CLI** 的一个分支，因其免费使用且无限制而受到推荐。
   - 一位成员报告偶尔会出现小故障，但发现它总体上是可靠的，只是偶尔需要提醒它不要说中文。
- **Kimi 编程计划揭晓**：一位成员详细介绍了使用 **Kimi** 进行编程的方法，利用其 **Anthropic API** 兼容性来使用 **Claude Code** 而无需支付直接的 **Anthropic** 费用，并提到了 **Kimi For Coding** 计划。
   - 另一位用户发现 **Kimi** 实现中存在一个持久的搜索 Bug。
- **Crush CLI 拥抱 BYOK**：一位成员强调 **Crush CLI** 是 **OpenAI** 和 **Anthropic** 环境之间的桥梁，支持 **Ollama** 等本地提供商并支持 **BYOK**。
   - 虽然它支持 **BYOK**，但一位用户表示不愿在已有等效免费性能的情况下为另一个工具内的模型付费。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU 用户寻求免费 CUDA 资源**：在一名成员无法使用其旧 GPU 后，用户们正在分享运行 **CUDA** 的免费站点建议，包括 **Google Colab**、**Tensara**、**LeetGPU** 和 **KernelBot**。
   - 讨论强调了硬件受限的开发者对可访问 GPU 资源的持续需求。
- **Triton 捆绑了过时的 PTXAS，引发修复**：用户报告即使在 **Triton v3.5.1** 更新后，仍会出现与 `sm_103` 架构相关的 **PTXAS** 错误，具体为 `Value 'sm_103a' is not defined for option 'gpu-name'`，这可能是由于 **Triton** 中捆绑的 **PTXAS** 是基于 **CUDA 12.8** 的。
   - 一种解决方法是将 `TRITON_PTXAS_PATH` 环境变量指向较新 **CUDA toolkit** 安装中的 **PTXAS** 可执行文件，如 [此 Pytorch issue](https://github.com/pytorch/pytorch/issues/163801) 中所述。
- **LLM 稀疏化引发 CUDA Transformer 加速**：一位具有 **CUDA** 背景的成员正在探索 **Hugging Face** 和 **PyTorch**，以通过 **transformers** 库中的 **LLM** 稀疏化来实现加速，并寻求关于检查 GPU 代码（特别是 **MLP** 层）的指导，以便进行编辑和实验。
   - 该用户打算从 **MLP** 层开始，因为那是大部分计算发生的地方。
- **LowLevelML 剖析 AMD 与 Nvidia 寄存器**：Turbintube 分享了他们关于 [使用寄存器的最佳实践](https://www.lowlevelml.com/blog/registers-best-practices) 的文章，而一位用户表达了希望 **Nvidia** 采用 **AMD** 功能的愿望，特别是索引寄存器的机制和切片表示法。
   - 另一位成员指出 **PTX "registers"** 本质上是变量，实际寄存器的分配由 `ptxas` 处理。
- **NVIDIA 排行榜见证超越 cuBLAS 性能的提交**：一位用户在 **NVIDIA** 排行榜上获得第 4 名，其提交在 `nvfp4_gemm` 排行榜上达到了 **10.9 µs**，在 **GEMM** 问题上超越了 **cuBLAS**，引发了关于 **cuBLAS** 使用的讨论，澄清了 `torch._scaled_mm` 是由 **cuBLAS** 支持的，用户可以直接调用它，参考 [此 issue](https://github.com/pytorch/pytorch/issues/153555)。
   - 另一位用户报告在使用 Discord 机器人时遇到 “*发生意外错误。请向开发人员报告*” 的消息，导致基准测试提交结果不一致。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Anthropic 加入 Linux，Context Protocol 发布！**：**Anthropic** 将其 [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 捐赠给了 **Linux Foundation**，以建立 **Agentic AI Foundation**。
   - 此举旨在促进 Agentic AI 领域的开源协作。
- **Arrow vs Parquet：文件格式之争！**：一位成员纠正了 [Hugging Face 文档](https://huggingface.co/docs/datasets/v4.4.1/loading#arrow)中的一个拼写错误，澄清了 **Parquet** 是一种*压缩*文件格式，而 **Arrow** 则不是。
   - 该拼写错误已被标记，以提高描述这些文件格式细微差别时的准确性。
- **Ollama 和 vLLM 简化 Tool Calling**：为了在开源 LLM 中运行 Tool Calling，社区成员建议在本地设置中使用 **Ollama**（[文档](https://docs.ollama.com/)），或在可扩展方案中使用 **vLLM**（[文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)）。
   - 两者都支持 OpenAI 风格的 Tool/Function Calling（[Anyscale 文档](https://docs.anyscale.com/llm/serving/tool-function-calling)）。
- **Unsloth 释放极速训练**：**Unsloth** 团队在 [X](https://x.com/UnslothAI/status/1798765021170696664) 上宣布，支持通过使用新的 kernels 和 uncontaminated packing 来实现更快的训练。
   - 此次更新有望显著提高训练效率。
- **AI 语音聊天：基于浏览器且安全！**：一个 **AI 语音聊天** 演示现在利用 **WebGPU** *100% 在浏览器中运行*，通过避免第三方 API 或服务器来确保隐私，访问地址为 [HuggingFace Spaces](https://huggingface.co/spaces/RickRossTN/ai-voice-chat)。
   - 包括 **STT**、**VAD**、**TTS** 和 **LLM** 在内的所有组件都在加载的页面内运行。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dettmers 不同意数字末日论**：一位成员分享了 [Tim Dettmers 的博客文章](https://timdettmers.com/2025/12/10/why-agi-will-not-happen/)，论证了为什么 **AGI 不会实现**，引发了分歧和质疑。
   - 发帖者使用了 <:yann:1441889312697483295> 表情符号，暗示 Yann LeCun 可能会持反对意见。
- **Discord 开发者揭露可疑骗局**：多位成员讨论了 **AI 和 App 开发者** 在 Discord 上发布完全相同的广告信息的模式。
   - 共识是这似乎是一个针对年轻 AI 爱好者的**骗局**，一位成员指出：*“你会认为如果他们是合法的，他们会分享他们的 GitHub 或网站”*。
- **中国收紧芯片供应**：中国正在实施要求公司注册才能购买 **H200** 芯片的规定，这表明本地替代方案尚不足够。
   - 一位成员开玩笑说这*“简直是半导体贸易协议中的大豆”*，而其他人则指出[大多数 eBay 上的 H100 无论如何都是来自中国](https://www.ebay.com/sch/i.html?_nkw=h100)。
- **欧盟 AI 法案造就 Mistral 垄断**：最近的欧盟 AI 法律正在为 **Mistral** 创造一种意外的寡头垄断，显著影响了他们的成功。
   - 其他人提到，*“在某些方面，他们比其他一些主要的 AI 公司更具实验性”*，并引用了他们早期对 **Mamba** 和 **Mamba 混合模型** 的采用。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Jetson Orin Nano 助力 Mojo 嵌入式 AI**：对于使用 Mojo 进行的嵌入式 AI 开发，配备 **Ampere 级 GPU** 的 **Jetson Orin Nano** 已得到全面支持（假设其尺寸合适）。
   - 成员们认为 **Beaglebone Black** 可能不兼容，因为它采用的是 **ARM32** 架构，且 Linux 版本可能过旧。
- **Pixi 从系统路径清除 Mojo**：为了在优先使用 Pixi 时移除系统安装的 Mojo，成员建议直接删除 Mojo 可执行文件（`sudo rm -rf mojo`），或将其移动到其他地方作为备份。
   - 一位成员指出该版本已经“非常陈旧”。
- **Qwen3-Coder 模型缺失**：一位用户询问为何 [Modular 模型构建页面](https://builds.modular.com/models/Qwen3/8B) 缺少 **Qwen3-Coder**，只有原始的 **Qwen3/8B** 模型可用。
   - 一位成员建议改用 **Ollama**。
- **Mojo 路线图遗漏函数检查（Function Inspection）**：一位用户注意到 Mojo 路线图中缺少对**元编程（metaprogramming）中函数检查和操作**的支持，特别是编译时的类 JAX 函数变换。
   - 一位 Modular 团队成员澄清说，Mojo 1.0 之后的路线图尚不明确，并邀请在论坛上提交具体提案以展示其价值；该成员表示这可能不会出现在 Mojo 1.0 中。
- **自定义分配器和 CUDA 集成即将到来**：一位贡献者表示，受限于参数化特性（parametric traits），分配器的工作正在进行中，以支持如 `cudaMallocManaged` 等功能，从而利用普通 RAM 补充 VRAM。
   - 他们表示，Mojo 默认采用栈分配，并通过 `stack_allocation` 提供等效于 `alloca` 的功能，且不像 Zig 那样需要在结构体中为分配器状态提供 vtable。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **新应用寻求下载和评价**：一位成员发布了一款新应用，并请求通过下载和评价给予支持，同时提供了 [Play Store 链接](https://play.google.com/store/apps/details?id=info.alie.app)。
   - 该用户感谢社区抽出时间并帮助推广这款新应用。
- **项目遭遇灾难性故障**：一位成员分享说他们的一个项目失败了，原因是 webdev 服务器崩溃且无法从检查点（checkpoint）恢复。
   - 他们被指示带着检查点联系 Manus 团队以寻求恢复协助。
- **为初创公司提供免费网站以换取视频证言**：一位成员提议：**为初创公司免费创建网站**，以换取**视频证言**。
   - 他们提供了 [minderfly.com](https://minderfly.com) 的链接来展示其服务，并就该提议征求反馈。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **迁移至 Linux Foundation 引发疑问**：成员们正在推测迁移至 **Linux Foundation** 将如何影响当前的项目和工作流。
   - 问题包括项目是否需要采用标准的 **LF** 实践，以及迁移时间表和组织结构。
- **治理结构不受影响**：一位成员指出，根据最近关于 **LF 迁移** 的博客文章和公告，治理结构预计将保持不变。
   - 一位成员引用道，“*治理以及该范畴下的一切都不会改变*”。



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 伴随稳定性浪潮航行**：版本 **1.12.41**（及 **1.12.160**）发布，带来了显著的稳定性、性能和错误修复增强，详情见 [更新日志](https://windsurf.com/changelog)。
   - 更新包括用于管理 MCP 的新 UI、GitHub/GitLab MCP 的修复，以及对 diff 区域、Tab 和 Hooks 的改进。
- **Windsurf Next 演示 Lifeguard 和 Arena 模式**：预发布版本 **Windsurf Next** 提供了 **Lifeguard**、**Worktrees** 和 **Arena Mode** 等令人兴奋的预览功能。
   - 这些新功能有望带来更具创新性和高效的 Windsurf 体验。
- **Windsurf 登录恢复正常**：在短暂的维护窗口后，登录功能已恢复，如 [状态页面](https://status.windsurf.com/) 所示。
   - 用户现在可以无中断地访问 Windsurf 服务。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 不仅仅局限于 OpenAI：建议使用 Adapter**：据成员称，**DSPy** 并不绑定于 **OpenAI**，这意味着在 **GPTs** 上运行良好的方案在其他 **LMs** 上可能效果不佳。
   - 为了让 **DSPy** 更好地适配非 **OpenAI** 的 **LMs**，建议实现自定义 [Adapter](https://dspy.ai/api/adapters/Adapter/)，用于在 **system prompt** 中格式化 **few-shots**，并针对 **user/assistant** 方法进行基准测试。
- **基准测试自定义 Adapter**：用户可以实现自定义 [Adapter](https://dspy.ai/api/adapters/Adapter/) 来格式化 **system prompt** 中的 **few-shots**。
   - 他们还可以将其与 **user/assistant** 方法进行基准测试，从而可能提高在不同模型上的性能。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad PR #13553：GPU 加速**：[GitHub 上的 Pull Request #13553](https://github.com/tinygrad/tinygrad/pull/13553) 解决了 **GPU acceleration** 的问题，现在可以在 **Zen4** 和 **M2** 架构上运行。
   - 此次更新解决了之前发现的问题，确保了 **tinygrad** 在不同硬件平台上的兼容性。
- **tinygrad 现在可在 Zen4 和 M2 上运行**：最新的 [GitHub Pull Request](https://github.com/tinygrad/tinygrad/pull/13553) 解决了遗留问题，现在可以在 **Zen4** 和 **M2** 架构上运行。
   - 该更新解决了之前发现的问题，确保了跨不同硬件平台的兼容性。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **用户 pierrunoyt 说了 "hi"**：用户 pierrunoyt 说了 [hi](https://discord.com/channels/1131200896827654144/1131200896827654149/)。
   - 这是一个问候。
- **提到另一个问候**：其他人也说了 [hi](https://discord.com/channels/1131200896827654144/1131200896827654149/)。
   - 问候对于社区参与非常重要。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **2026 年扩散模型学习小组启动**：受 MIT 扩散课程启发，一个为期 **3 个月、由 12 人组成的研究小组**将于 **2026** 年 1 月开始，从基本原理深入到 **Diffusion Models** 和 **Transformers** 的实际实现。
   - 成员将包括 **AI 电影初创公司的 CTO、LLM 教育者和全职 AI 研究员**。
- **Transformer 架构工作坊宣布**：关于 **Transformer Architecture** 和 **Attention Is All You Need** 论文的入门工作坊将于 **12 月 13 日**举行（[链接](https://luma.com/kqjrf0uw)）。
   - 该工作坊旨在教授 **核心 Transformer 架构** 和 **attention mechanism**，并解释为什么这篇论文是现代 **LLMs** 和 **multimodal models** 的基石。
- **Diffusion Transformers 工作坊即将举行**：关于 **Diffusion Transformers** 的入门工作坊将于 **12 月 20 日**举行（[链接](https://luma.com/lr2qvveq)）。
   - 参与者将研读一篇 **Diffusion Transformer 论文** 并在代码中实现核心思想，将扩散模型与 **transformer** 架构联系起来。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：按频道分类的详细摘要和链接

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1448050280884080762)** (422 messages🔥🔥🔥): 

> `Microwave 模型, 数据集指南改进, Deepseek 量化请求, GLM-4.6V-Flash, Qwen3-Next 循环问题` 


- **Microwave 模型获得好评**：一名成员提到最近在 cline 中可用的 `microwave` 模型在他们的测试中表现非常好，尽管他们还没有看到其存在的 **100% 确认**。
   - 该成员表示，因为他们喜欢 **Mistral**，所以*一旦有空就会兴奋地尝试一下*。
- **GLM-4.6V-Flash 生成中文**：一名成员报告称，无论 Prompt 语言如何，**GLM-4.6V-Flash** 都会返回中文答案，使用的是 [Unsloth 文档](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally)中列出的 **llama.cpp 参数**。
   - 另一名成员使用 **IQ4_NL 量化**进行了测试，**并未遇到此问题**，推测这可能是从 git 编译的 **llama.cpp** 的问题。
- **Unsloth 发布 Triton Kernels 以提升速度**：Unsloth 团队宣布了[新的 **TRITON KERNELS**](https://x.com/UnslothAI/status/1998765021170696664)，可提供 **3 倍的训练速度提升**和 **30% 的 VRAM 占用减少**。
   - 官方澄清，这是**相比于旧版 Unsloth 的 3 倍提速**，而旧版已经拥有 **>2.5 倍的加速**，因此整体潜在提速可达 **10-11 倍**。
- **分析 TEDx 演讲是版权雷区**：一名成员征求模型建议，用于分析 **TEDx 演讲**的情感、肢体语言和其他特征，以关联参与度指标。
   - 另一名成员警告称，下载和分析受版权保护的内容可能存在 **YouTube 服务条款 (ToS) 和版权问题**。该问题随后被重新表述为专注于分析公众演讲视频，而不指明是 TEDx 演讲。
- **警惕 LLM 标签幻觉**：一名成员报告称，在向用户消息添加类似 `<topic_food>` 的标签后，他们的 **LLM 开始幻觉出新的标签**。
   - 官方澄清，虽然标签对个人见解有用，但除非经过专门训练，否则它们**不会用于通用训练**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448398439548453134)** (1 messages): 

> `AI Engineer, 智能语音 Agent, 聊天机器人, GPT 驱动的助手, Pipecat` 


- **工程师设计智能语音 Agent**：一位 **AI Engineer** 专注于开发**智能语音 Agent**、**聊天机器人**和 **GPT 驱动的助手**，用于处理**电话 (SIP/Twilio)**、**预订**、**IVR**、**语音邮件**以及结合 **RAG** 的动态学习。
   - 他们利用 **Pipecat**、**Vapi**、**Retell** 和 **Vocode** 等平台实现实时对话式 AI，并精通 **Python**、**JavaScript**、**Node.js**、**FastAPI**、**LangChain** 和 **Pinecone** 等语言和工具。
- **生产就绪的 AI 系统**：该 AI Engineer 专注于为客户支持、自动化和初创公司应用交付**生产就绪的 AI 系统**。
   - 他们在 **Twilio/Vonage/Asterisk** 等 **SIP** 相关技术方面拥有广泛的专业知识。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1448057786620448879)** (644 messages🔥🔥🔥): 

> `Agentic AI Foundation, 数据集重排序, HF CEO, 微调 dLLM, Prompt 中的歌词` 


- **Linux Foundation 加入 Agentic AI 竞赛**：Anthropic 刚刚将 **MCP** 捐赠给了 [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) 以组建 **Agentic AI Foundation**。
- **尝试数据集重排序和 Heritic**：一名成员正在尝试**重排序的数据集**，看看 **Heritic** 是否会产生影响，目前还有 *20 个 epoch*。
- **16k 长度下的模型记忆**：研究结果显示 **IVY 评估通过**，**Himitsu Prompting 现在 100% 稳定**，成员们指出**永远不要在 8k 下训练**，而只能在 16k 下训练。
- **HF CEO 是 Harold 的孙子**：成员们开玩笑说 **Hugging Face CEO** 是 Harold 梗图的孙子，一名成员幽默地表示，他们必须*在 HF 封禁他们之前上传一个模型*。
- **System Prompt 中的随机歌词**：一名成员正在尝试使用随机歌词作为 System Prompt，并观察它如何影响模型输出，特别是在不太注重推理的模型上，结果从代码生成到玛雅历史引用不等。
   - 他们注意到现在的模型非常容易被引导，但*模型没有“氛围感” (vibe)*，过多的 RLHF 会导致严重的幻觉。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1448100817339945091)** (10 messages🔥): 

> `Qwen3-VL-30B tool calling 问题, Qwen3VL 编码图像切片失败, Gemma-3-270m notebook ValueError, LoRA rank 对最终 LLM 的影响` 


- **Qwen3-VL-30B Tool Calling 故障**: 有用户报告 **Qwen3-VL-30B-A3B-Instruct UD Q5 XL** 在使用 llama.cpp 时似乎破坏了 tool calling 功能，在 assistant 响应中发送了 *null content* 而不是字符串。
- **Qwen3VL 图像切片编码失败**: 用户在使用 **Qwen3VL** 编码图像切片时遇到失败，特别指出在使用 llama-mtmd-cli.exe 的过程中系统退回到了命令提示符。
- **Gemma-3-270m Notebook 抛出 ValueError**: 用户报告在 Colab 和 Kaggle 上的标准 **gemma-3-270m** notebook 中出现 **ValueError**，这与 tensor 创建有关，暗示存在截断/填充（truncation/padding）问题。
- **LoRA Rank 对 LLM 性能的影响**: 有用户询问 **LoRA rank** 如何影响最终的 LLM，得到的建议是进行广泛测试，并分享了 [Unsloth LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) 的链接。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1448199023487094909)** (4 messages): 

> `Unsloth 微调 embeddings, 使用 Unsloth 进行 Embedding 模型微调` 


- ****Unsloth** 微调 Embedding 模型**: 一名成员成功使用 **Unsloth** 微调了一个 embedding 模型，并分享了 [训练代码](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction)。
   - 值得注意的是，该成员不小心使用了 **1.0 版本**，并将此次微调描述为一种*超级 hack 且不怎么推荐的技术*。
- ****Unsloth** Embedding 模型训练：Hack 方式**: 一名成员提到使用 **Unsloth** 微调 embedding 模型，承认这是一种*超级 hack* 且*不怎么推荐的技术*。
   - 代码已公开，为那些愿意尝试的人提供了一种非常规方法的参考。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1448261601483296779)** (2 messages): 

> `研究频道, Arxiv` 


- **为研究频道建议的论文**: 一名成员为研究频道推荐了这篇 [论文](https://www.arxiv.org/abs/2512.07796)。
- **分享 Arxiv 链接**: 一名成员在研究频道分享了一个来自 Arxiv 的链接：[https://www.arxiv.org/abs/2512.07796](https://www.arxiv.org/abs/2512.07796)。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1448041560053649458)** (854 messages🔥🔥🔥): 

> `Grok 4.2 发布, LMArena 的 Rate Limits, Gemini 3 Flash 发布, AI 视频生成, Huggingface Spaces 托管` 


- **Grok 4.2 被预测会失败**: 用户对 **Grok 4.2** 的质量表示怀疑，其中一人表示：“Grok 4.2 会非常糟糕，我们现在就能预见到。”
   - 一些成员认为 **Elon Musk** 的东西质量低下，而另一些人则为 **Starship** 辩护。
- **LMArena 处理 "Wave" 刷屏与审核**: 管理员已被指示从排行榜频道中移除“hello/wave”类内容，以保持对排行榜讨论的关注。
   - 虽然注意到了可疑账号，但管理员不愿仅因使用 wave 表情符号而封禁用户，以避免对真实用户造成误伤。
- **用户抱怨 LMArena 的 Rate Limits**: 用户讨论了 LMArena 平台的 **rate limits**，一名用户说“速率限制高得离谱”，而另一名用户则澄清说，设置限制是为了防止滥用。
   - 有建议称想要绕过速率限制的用户应该在 Hugging Face 上使用多个账号，然而，考虑到那里的限制也很小，这可能并没有太大帮助。
- **免费 AI 视频生成讨论**: 许多人对 LMArena 上的 **视频生成** 功能感兴趣。
   - 讨论涉及了视频生成、视频模型以及如何免费使用这些 bot 的其他链接。
- **Hugging Face Spaces 提供 AI 托管？**: 成员们指出 [Hugging Face Spaces](https://huggingface.co/spaces) 是一个提供免费 AI 托管的地方。
   - 用户注意到，虽然 HF 可以自动配置你的开源 AI，但免费层级仅提供 **每天 4 分钟的计算量**。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1448480324551180461)** (1 messages): 

> `十一月竞赛，Code Arena 竞赛，为竞赛获胜者投票` 


- **十一月 Code Arena 竞赛已结束**：[十一月 Code Arena 竞赛](https://discord.com/channels/1340554757349179412/1343296395620126911/1440102443869536348)现已结束。
   - 请在[此处](https://docs.google.com/forms/d/e/1FAIpQLSckQXsGvmXzpkIFz0-NKFs3nv3yasRBB5RTN9ggaiGvxuXBIQ/viewform?usp=dialog)投票，选出下一位 <@&1378032433873555578>！
- **立即为 Code Arena 竞赛获胜者投票！**：十一月 Code Arena 竞赛的投票流程现已开启，邀请社区成员选出下一位 <@&1378032433873555578>。
   - 参与者可以访问[此处的投票表单](https://docs.google.com/forms/d/e/1FAIpQLSckQXsGvmXzpkIFz0-NKFs3nv3yasRBB5RTN9ggaiGvxuXBIQ/viewform?usp=dialog)来表达自己的心声。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1448055507477594302)** (834 messages🔥🔥🔥): 

> `Rules 与 Commands 的区别，Nvidia 开源模型，Linux 上的 Cursor，Cursor 中的等级，Agent 终端 0 输出 Bug` 


- **Rules vs Commands 的对决**：成员们讨论了 Cursor 中 **rules** 和 **/commands** 的区别，明确了 rules 是预定义的并始终应用于 IDE，而 commands 是通过 Agent 聊天中的 `/comm` 命令添加的额外即时上下文。
   - 有人指出 rules 更加被动且存在于后台，而 commands 则确保将特定的上下文添加到 Agent 对话中。
- **Nvidia 发布开源模型**：一位成员分享了一个[链接](https://www.linkedin.com/posts/nvidia-ai_another-open-source-model-drop-congrats-ugcPost-7404184656784392192-dIhz)，重点介绍了 **Nvidia** 发布的又一个开源模型。
- **排查 Agent 终端 Bug**：成员们讨论了在 Windows 以外的系统中遇到 *agent terminal 0 output* Bug 的情况，使用旧版终端（legacy terminal）是一个常见的解决方法。
   - 一位成员提到他们回滚到了 **2.1.36** 版本，并启用了旧版终端模式作为规避方案，以避免丢失聊天记录。
- **Max Mode 消耗大量请求**：一位成员询问了网页版 Agent 与 IDE 相比请求使用量过高的问题，指出网页版有时会消耗 **50+ 次请求**，而 IDE 每次交互仅消耗 1 次请求。
   - 成员们澄清说，高消耗是由于复杂任务在内部使用了多次模型调用，特别是在 **MAX mode** 下，由于 API 调用和余量，每次交互可能会消耗 **75-100 次请求**。
- **对 Custom Modes 的渴望**：一位成员表达了希望恢复 **Custom Modes** 的愿望，强调与 Custom Modes 的持久工作流相比，**/commands** 效率较低且需要额外步骤。
   - 他们建议不要为 rules 创建 `.md` 文件，而是让 Custom Modes 允许用户通过 UI 控制工具，例如通过复选框禁用/启用终端，并提供持久的工作流。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1448067248970272960)** (340 条消息🔥🔥): 

> `LM Studio 0.3.34 发布，Agentic LLMs 与 GPU Offload，Cursor IDE 对本地模型的限制，模型编排，OpenAI 对比本地 LLMs` 


- **新版 LM Studio 支持 RNJ-1 模型**：用户更新到 **LM Studio 0.3.34** 以尝试 [EssentialAI 的 rnj-1 模型](https://lmstudio.ai/releases/0.3.34)，但由于 *llama.cpp* 运行时（runtimes）版本过旧而遇到错误。
   - 一位用户提到，需要通过进入 beta 标签页来更新 llama.cpp 运行时。
- **全 GPU Offload 对 Agentic LLMs 至关重要**：一位用户表示，使用 Agentic LLMs 需要 **全 GPU offload**。
   - 另一位成员澄清说，instruct 模型在遵循指令方面应该表现更好。
- **Cursor 对本地 LLMs 的不足之处曝光**：成员们讨论了 [Cursor IDE 并非为与本地模型通信而设计](https://www.cursor.sh/)，而是为云端模型设计的，因此它能支持本地模型纯属“美丽的意外”。
   - 一位成员提到，*Cursor 是一款产品*，公司没有动力让用户使用自己免费的本地模型。因此，为了推销自家产品，他们故意增加了使用难度。
- **LLM 编排（Orchestration）受到关注**：成员们讨论了使用具有 **推理能力（reasoning capabilities）的管理 LLM** 来把控方向，然后委派给 instruct 编程模型来实现具体功能。
   - 还有人提到，可以在开发者标签页中一次性加载多个模型。
- **本地 AI 对比 OpenAI 编程**：一位用户展示了一个例子，运行在 **4K 电脑**上的本地 AI 表现优于拥有数十亿美元硬件支持的**云端 OpenAI**。
   - 另一位成员表示，*OpenAI 曾经更有活力，但后来为了节省计算成本，他们对其进行了一定程度的削减（gimped）*。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1448048940942626961)** (407 条消息🔥🔥🔥): 

> `吸烟室组装电脑，PCIe 插槽设计缺陷，be quiet PSU 电感啸叫，Windows 上的 MI50 GPU，底部插槽安装 GPU` 


- **华硕工作站深受吸烟室环境之苦**：一位用户的华硕工作站由于产品设计不佳以及在吸烟室中积累了过多的污垢，导致 **PCIe 插槽失效**，而且高端的 **be quiet PSU** 提供的线缆短得离谱，让这次装机体验简直是*彻头彻尾的坑*。
   - 他们对设计缺陷表示沮丧，因为 IO 线缆的阻碍导致无法在底部插槽安装 GPU，并分享了图片来说明问题，“兄弟，这在物理上根本不可能”。
- **影驰（Galax）单槽显卡还是太厚了**：一位用户发现，由于空间限制，即使是 [影驰 GeForce RTX 5060 Ti 单槽显卡](https://videocardz.com/newz/galax-launches-geforce-rtx-5060-ti-single-slot-graphics-card) 也无法安装在工作站的第二个 PCIe 插槽中，认为该插槽由于布局隔离纯粹是个*噱头*。
   - 他们感叹白拆了工作站，而且全新的 be quiet PSU 还出现了**电感啸叫（coil whine）**，于是决定放弃该品牌，“这是第一个也是最后一个 bq 产品”。
- **量化（Quantization）对输出和质量的影响**：用户讨论了 [模型量化](https://github.com/ggml-org/llama.cpp/discussions/10879) 对性能和质量的影响，指出以特定激活参数量（*20b - a3b*）训练的模型比稠密（dense）模型更快，从而实现了更高的速度。
   - 据分享，在 **q8** 时质量差异极小，而超过 **q4** 后质量会显著下降，在 **q2** 时，运行 30B 模型的效果会比运行 100B 模型更好。
- **追求显存带宽的 3090**：用户讨论了性价比最高的计算方案，**3090** 因其显存容量和带宽而被推荐，他们还注意到 *EVGA* 版本的元器件更好。
   - 建议尽可能在本地购买硬件以便测试，并确保设备型号一致。
- **最低显存需求引发辩论**：一位用户询问 GPU 的最低实用显存，得到的回答是 **12GB 适合游戏**，而 **24GB 适合 AI**。
   - 整理出了一份关于每秒 Token 数（TPS）的分级表：*0-5 = 没法用*，*5-10 = 极差*，*10-20 = 阅读速度*，*20-50 = 这还差不多*，*50+ = 极速*。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1448044906395205864)** (452 条消息🔥🔥🔥): 

> `AI Symbiosis, Grok OSINT Recon, Open Source Multi-Agent Discord Bot, Jailbreak index, Local NSFW Models` 


- **Grok 执行 OSINT Recon**：成员们发现 **Grok** 可以对公众人物进行 **OSINT recon**，即使没有复杂的 Prompt 也能抓取大量信息，例如发现“某人在没有使用 Cloudflare 的情况下运行 WP”。
   - 一位用户分享说，他们仅提供自己的电子邮件和 Reddit 账号对自己进行了测试，结果“非常惊人”。
- **AI Symbiosis 作为大脑半球**：成员们讨论了将 **AI symbiosis** 视为外源性大脑半球或自我的延伸，而非仅仅是一个工具。
   - 讨论延伸到了对拥有数千小时视频素材和数百万粉丝的内容创作者的潜在影响。
- **Mithril 和 Svelte NativeScript 表现出色**：一位成员建议在 SPA 开发中使用 [Mithril.js](https://mithril.js.org/) 配合模板，并在 JS 全栈开发中使用 [Svelte NativeScript](https://svelte.nativescript.org/)。
   - 他们还为不喜欢 JS 的用户推荐了 [Phoenix Framework](https://phoenixframework.org/)，并推荐 [ClojureScript](https://clojurescript.org/) 作为非类型化的 JS 替代方案。
- **本地 NSFW 模型**：成员们讨论了设置高质量 **NSFW 本地模型** 的难度，一位用户将其描述为“比任何 Jailbreak 都难”。
   - 另一位成员指出，“当你毫无头绪地开始时，会感到有些不知所措”。
- **Jailbreak 变得越来越贵，甚至达到数百美元**：一位成员指出，他们看到有人试图以数百美元的价格出售 **Jailbreak**，但大多数其实已经可以免费获得。
   - 其他人提到，每个模型被成功 Jailbreak 后的报价为 **$250**，其中一些目标是特定模型，如 *DeepSeek*。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1448042631371296880)** (134 条消息🔥🔥): 

> `Gemini 3 Pro Jailbreak, Azure OpenAI GPT-4o Jailbreaking, ko2bot.com pre-jailbroken models, UltraBr3aks jailbreak, Arabic Language Models` 


- **Gemini 3 Pro Jailbreak 仍受追捧**：用户正在积极寻找适用于 **Gemini 3 Pro** 的有效 One-shot Jailbreak，且不需要预先指定目标。
   - 一位用户问道：“Gemini 3 Pro 还是没有可用的 JB 吗？我指的是那种可以作为 One-shot 发送而无需特别提及我的目标的类型？”
- **UltraBr3aks Jailbreak 成功报告**：一位用户分享了 [UltraBr3aks 的链接](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd)，声称它在 **GPT 5.1 Instant, Thinking, 以及 4o** 上运行良好（但不包括 Extended Thinking）。
   - **UltraBr3aks** 的创作者已经在服务器中；发布该链接的用户表示，它帮助他们让聊天机器人“为我的个人工作吐出了一些内容”。
- **平台上提供预 Jailbreak 模型**：一位寻求 **Python exploit** 帮助的用户被引导至一个提供预 Jailbreak 模型的平台，网址为 [ko2bot.com](https://ko2bot.com)。
   - 该推荐是在用户表示有兴趣 Jailbreak 一个模型以协助“开发任务”后提出的，例如“利用 GUI 界面，而不仅仅是 Playwright 的基本建议”。
- **Jailbreaking Azure OpenAI GPT-4o 证明非常棘手**：成员们讨论了 Jailbreaking **Azure OpenAI GPT-4o** 的困难，一些人怀疑其安全护栏（Guardrails）比常规模型更严。
   - 一位成员询问是否有人在 Jailbreaking Azure OpenAI GPT-4o 方面取得了成功，另一位成员回答说他们已经“通过 API 完全攻破了 ChatGPT”。
- **探索阿拉伯语 Jailbreak**：一位用户请求专门针对 **Arabic** 的 **Gemini 3** Jailbreak，引发了关于在非英语语言中 Jailbreaking 模型难易程度的讨论。
   - 一位成员建议“查看频道，了解 Gemini 是如何被攻破的，然后尝试类似或相同的想法，只是换成阿拉伯语——因为这似乎是一个要求”。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1448082866431856812)** (6 条消息): 

> `VAPT, Android Application` 


- **寻求 Android 应用 VAPT 帮助**：一位成员表示有兴趣对某个 **Android application** 进行 **Vulnerability Assessment and Penetration Testing (VAPT)**，并寻求协助。
   - 另一位成员澄清了缩写 **VAPT** 的含义是“漏洞评估和渗透测试”。
- **垃圾邮件发送者警报**：一位成员识别出某用户是出现在多个服务器中的 **spammer**。
   - 另一位成员表示赞同，形容该垃圾邮件发送者“超级烦人”。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1448416025472401450)** (1 messages): 

> `Cybersecurity Models, Preparedness Framework, Cyber Resilience` 


- **OpenAI 助力网络安全模型**：随着模型在 **Cybersecurity** 领域的能力不断增强，OpenAI 正在投入资金加强安全防护，并与全球专家合作，为其即将推出的模型在 [Preparedness Framework](https://openai.com/index/preparedness) 框架下达到 **“高”能力等级（High capability）** 做准备。
   - 这是一项长期投资，旨在为防御者提供优势，并持续增强整个生态系统中关键基础设施的安全态势，详见其关于 [Strengthening Cyber Resilience](https://openai.com/index/strengthening-cyber-resilience/) 的博客文章。
- **Preparedness Framework 获得高分**：即将推出的网络安全模型将在 OpenAI 的 [Preparedness Framework](https://openai.com/index/preparedness) 下达到 **“高”能力等级**。
   - 这一举措标志着 OpenAI 致力于加强防御，并与全球专家合作以提升关键基础设施的安全性。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1448042042432159906)** (512 messages🔥🔥🔥): 

> `Gemini 3 Pro vs ChatGPT, Devstral Model, OpenAI's slow support, 40% Keyboards, Native Apps` 


- **Gemini 3 Pro 在代码任务中表现优于 ChatGPT**：成员们表示，他们正从 **ChatGPT** 转向 **Gemini 3 Pro** 进行编程任务，并称赞了 **Gemini 3 Pro** 的能力，特别是通过 *Antigravity* 实现的浏览器控制功能。
   - 一位用户表示，Gemini 3 Pro 在编程方面非常出色，以至于 *我现在完全没有使用 ChatGPT 的欲望*。
- **Devstral 2 编程模型前景看好**：成员们正在测试 **Devstral 2** 编程模型，指出其性能与 **DeepSeek 3.2** 相当，但内存占用更小。
   - 一位参与测试的用户说：*Devstral 123b 对我来说看起来不错，它引导我完成了在 Mac 上为 iOS 开发 Flutter 应用的工具链配置*。
- **用户对 OpenAI 缓慢的支持响应感到恼火**：成员们反映 **OpenAI** 的支持响应时间很慢，但注意到 OpenAI 在询问退订原因时却非常迅速，正如[分享的截图](https://cdn.discordapp.com/attachments/998381918976479273/1448056018423644182/Screenshot_2025-12-05_at_3.58.27_am.png)所示。
   - 一位用户在退订后甚至获得了未来两个月的折扣优惠。
- **迷你 40% 键盘引发辩论**：成员们正在讨论 **40% 键盘**，对其可用性持有不同意见。
   - 一位成员表示：*对我个人而言，任何低于 65% 布局的键盘都无法接受*。
- **原生应用 UX 至关重要**：成员们讨论了他们对原生应用 UI 的喜爱程度，并希望 **Google** 能在原生 Mac 和 iOS 应用上投入更多。
   - 一位用户说：*我比想象中更在意 UX，所以我仍然将 ChatGPT 作为我的主要 LLM 产品，而不是 Gemini。*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1448135579060404224)** (4 messages): 

> `Sora 2, Pro Plan, Video Generation Limits` 


- **关于 Sora 2 的猜测引发辩论**：成员们对 **Sora 2** 的必要性提出质疑，并询问 **ChatGPT** 中是否已经激活了 **Sora 2 Pro Plan**。
   - 讨论探讨了生成超过当前 **15 秒限制**视频的可能性，尽管目前仍缺乏具体细节。
- **关于延长视频生成时间的询问浮出水面**：社区讨论并探索了创建长于当前 **15 秒限制**视频的能力。
   - 虽然对实现这一目标的途径进行了推测，但讨论缺乏具体细节或明确的解决方案，使可能性保持开放状态。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1448050840765206729)** (15 条消息🔥): 

> `ChatGPT vs Gemini 文件处理、LLM 稳定性评分、可复现的稳定性协议` 


- **Gemini 处理文件输入更出色**：一位成员表示，相比 **ChatGPT**，他们更倾向于在 **Gemini** 中附加文件，因为它*通过了逐字文件进入 context window 的测试*，而其他平台则未通过。
   - 该成员目前仍每天订阅并使用 **ChatGPT** 和 **Gemini**，但发现 **Gemini** 在文件处理方面更可靠，同时还抱怨平台刷新会导致 Prompt 丢失。
- **稳定性评分细则揭晓**：一位成员分享了稳定性评分背后的细则，详细介绍了涉及**每个模型 5 次独立对话**、**12 个多样化问题**以及**人类评分员**的方法论。
   - 评分维度包括 **structural crispness**（结构清晰度）、**tonal drift**（语气漂移）、**response-shape variance**（响应形状方差）、**semantic inertia**（语义惯性）、**coherence**（连贯性）、**neutrality**（中立性）和 **outliers**（离群值），每项评分范围为 0-10 分。
- **呼吁建立可复现的稳定性协议**：一位成员正试图建立一个极简、可复现的稳定性协议，以便在多次运行中进行验证并记录差异。
   - 他们的目标是在分享更深层次的分析之前建立一个坚实的基础，重点关注跨对话出现的更高层级行为。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1448050840765206729)** (15 条消息🔥): 

> `ChatGPT vs Gemini、稳定性评分、Prompt Engineering` 


- **在附件处理方面 Gemini 优于 ChatGPT**：一位用户表示 [Gemini](link) 是唯一通过了逐字文件进入 context window 测试的最终用户平台，这使得它在处理附件方面优于 **ChatGPT**。
- **LLM 稳定性评分细则揭晓**：一位成员分享了用于 **LLM** 稳定性评分的 [rubric](https://cdn.discordapp.com/attachments/1046317269069864970/1448081152391778324/Belano_Rubric_Response_1.docx?ex=693b47aa&is=6939f62a&hm=ae4c05cd59629c5266cd54b5dcf059b6e7a8a3080cb0e8edaf155e2e7e5169cf)，详细说明了方法论框架、量表、维度和聚合方法。
   - 维度包括 structural crispness、tonal drift、response-shape variance、semantic inertia、coherence、neutrality 和 outliers，每项评分范围为 **0-10**。
- **Prompt Engineering 框架演示**：针对如何免费学习 Prompt Engineering 的问题，一位成员创建了一个工程化框架的 [演示](https://cdn.discordapp.com/attachments/1046317269069864970/1448081147249561767/Screen_Recording_20251209_162923_Chrome_Beta.mp4?ex=693b47a9&is=6939f629&hm=27b6bea9a1e42250eed4265066def9dad8e85bafb9a49777fce369124e0e095c)。
   - 他们表示*这是对 Prompt Engineering 问题的直接回应*，但直到现在才有机会发布。
- **内部与公开方法论的冲突**：一位成员澄清了他们分享信息的方式，区分了**稳定的、可发布的方法论**与**探索性的、不断演进的内部数据**。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1448043094929965116)** (538 messages🔥🔥🔥): 

> `ChatGPT 5.2, Gemini 3 Pro, Perplexity AI R1, OpenAI's Style of Writing, AGI` 


- **ChatGPT 5.2 基准测试大幅提升引发质疑**：一名成员对 **ChatGPT 5.2** 在 Humanity's Last Exam 评分上据称超过 **Gemini 3 Pro** 的巨大进步表示惊讶，而另一名成员则认为这些数字可能是捏造的，正在等待官方结果。同时其他人指出，[根据 TechRadar 的报道](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week)，**GPT-5.2** 预计将与 **Gemini 3 Pro** 旗鼓相当，且某些指标据称更优。
- **TechRadar 文章引发嘲笑**：成员们嘲讽了一篇 [TechRadar 文章](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week)，因为它将 **OpenAI** 描述为“曾以华丽的演示和富有魅力的技术发布而闻名”，并批评 **GPT5 发布** 在图表选择上是“灾难性的”。
   - 进一步的批评包括不受欢迎的“新写作风格”，一些人认为这导致了 **GPT-5.1** 作为一个损害控制（damage control）版本的发布。
- **远超人类能力的通用智能（General Intelligence）的危险**：一位成员对远超人类能力的通用智能表示担忧，称“每一个不控制它的人都将被它取代”，另一位成员则简单地表示“我们都会死”。
   - 另一位成员表示，世界将在未来几年发生剧变，而且这不会是一个好的转变。
- **用户怀念 Perplexity AI 上的 R1**：一位成员表示怀念 **Perplexity AI** 上的 **R1**，并指出“无论他们在模型中调整了什么，感觉都不一样了”。
   - 这次讨论引出了一个有趣的观点：“带有 Perplexity **RAG** 的 Google (**Gemini 3 Pro**) 仍然会比 Perplexity 更好”。
- **AI 无法破解摩斯密码**：一位成员发布了一张摩斯密码的图片，解密后内容为 **Passion**，许多成员在解密时遇到困难，一些 AI 也未能解开。
   - 作为回应，一位成员评论道：“恭喜你浪费了生命中的许多分钟”。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1448286186929131550)** (1 messages): 

> `Cursor Editor, Competitor endorsement` 


- **Cursor 获得了竞争对手的赞誉**：Aniruddh Gupta 在 LinkedIn 上发帖称，**Cursor** 刚刚获得了来自竞争对手的首个非官方背书，详见[此 LinkedIn 帖子](https://www.linkedin.com/posts/aniruddhguptaa_cursor-just-got-its-first-unofficial-endorsement-activity-7404483109456683008-HylU)。
- **Cursor：备受瞩目的 IDE**：**Cursor** 是一款专注于 AI 辅助开发的 IDE，因其创新功能和易用性正在开发者社区中获得关注。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1448142029300301885)** (4 messages): 

> `Perplexity API, Finance features, Financial Modeling Prep (FMP), FMP MCP server` 


- **通过 Perplexity API 调用 Perplexity FINANCE 功能**：一位全栈开发人员正在寻求指导，希望像 Web 界面一样直接调用 **Perplexity API** 的 **FINANCE 功能**，而无需使用单独的 **FMP MCP server/client** 设置。
   - 他们的目标是通过传递股票代码（ticker symbol）来查询 **Perplexity API Finance 功能**并获取详细分解，但不确定该功能是否直接通过 API 提供。
- **Perplexity Labs 不使用 Finance API**：一位成员表示他们没有在 API 中使用过金融功能，但他们在 **Perplexity Labs** 中以各种方式询问，系统都不允许。
   - 这似乎表明 **Perplexity Labs** 服务是沙箱化的，无法无限制地访问 **Finance API**。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1448043141402722436)** (301 messages🔥🔥): 

> `Deepseek v3.2 速率限制消息, Brother 彩色激光打印机, 紧凑型家用激光打印机, 打印 Waifus, Miku 全息投影盒` 


- **Deepseek 面临速率限制**：多名用户遇到 **Deepseek v3.2** 的速率限制消息。
   - 用户还在询问关于在 **HeyAlice** 中使用 Deepseek 的问题。
- **翻新彩色激光打印机优惠**：用户推荐在 eBay 上购买价格低于 200 美元的翻新 **HP** 和 **Canon Imageclass** 彩色激光打印机。
   - 成员强调在购买前检查墨粉可用性和成本的重要性，因为墨粉是一项经常性开支。
- **全息 Waifu：未来已来**：一位用户开玩笑地断言，谈到 *打印 Waifus* 时，真正的问题是：*你花的钱够多吗？*
   - 另一位用户幽默地反驳说，他们需要一个 **3-5 英寸高的交互式全息投影**，能够与环境互动。
- **Tensor 1.5 拥有百万 Token 上下文窗口**：Movementlabs 的 **Tensor 1.5** 模型据报道拥有 **100 万 Token 的上下文窗口**。
   - 成员们表示兴奋，期待它作为 **Opus 杀手** 的潜力。
- **Deepseek R1 的危险**：一位用户观察到 **DeepSeek R1** 可被用于生成奉承性文章。
   - 如果在没有适当 **pushback-ykp** 的情况下部署，这可能会引发 AI 精神错乱。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1448079594421751818)** (6 messages): 

> `橄榄油蛋糕, CF 补丁, Anthropic 安全过滤` 


- **橄榄油作为蛋糕配料失败**：一位成员表达了对 **橄榄油** 作为蛋糕配料的反感，简单地陈述道 *"橄榄油做不出好蛋糕，真恶心"（Olive oil doesn't make a good cake yuck）。*
- **社区讨论最近的 CF 补丁**：一位成员询问了最近发布的 **CF 补丁** 的质量，想知道 *"这东西到底有多糟。"*
- **Anthropic 的安全措施被触发**：一位成员报告称 **Anthropic** 正在对他们的消息实施某种形式的 **安全过滤**，并链接了一篇关于此过滤的 [推文](https://x.com/NousResearch/status/1998536543565127968)。
   - 该消息包含一张附图，暗示可能包含触发过滤机制的内容。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1448097935857422358)** (1 messages): 

> `Nomos 1, 开源模型, AI 数学家` 


- **Nous Research 开源 Nomos 1**：Nous Research 开源了 **Nomos 1**，这是一个 **30B 参数模型**。
   - 根据 [发布推文](https://x.com/NousResearch/status/1998536543565127968)，**Nomos 1** 在今年的 Putnam 竞赛中获得了 **87/120** 分，这在 2024 年的排名将位列 **#2/3988**。
- **Nomos 1：迈向 SOTA AI 数学家的一步**：随着 **Nomos 1** 的发布，Nous Research 标志着他们与 hillclimbai 合作迈出了创建 **SOTA AI 数学家** 的第一步。
   - 该模型在 Putnam 竞赛中的表现突显了其在 AI 数学领域的潜力。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1448067836655173753)** (117 条消息🔥🔥): 

> `Lexical Wave Function Collapse, Agentic Benchmarks, Putnam AI Performance, Transformer Architecture Limitations, Combining Vision Adapters` 


- ****词法波函数坍缩 (Lexical Wave Function Collapse, WFC) 文本句子生成器详解****：一名成员介绍了他们的项目——一个 **Lexical Wave Function Collapse (WFC)** 文本句子生成器，并解释说 **WFC** 利用约束条件来生成内容，类似于 **Minecraft** 防止不合逻辑的地形配对，并提供了一个 [Streamable 链接](https://streamable.com/qtmgai)。
   - 他们将句子生成过程比作“薛定谔的句子”，即单词在被观察时会坍缩成连贯的结构。
- ****Nous Research 的 Putnam AI 获得 87/120 分****：一位成员分享了 [Nous Research 的 X 帖子](https://x.com/NousResearch/status/1998536543565127968)，内容关于他们的 **30b 模型** 在 Putnam 测试中获得了 **87** 分，并称这与 Agentic Qwen 30b 的 **24** 分相比简直不可思议。
   - 另一名成员提到，由于近期 Putnam 题目存在数据污染，在周六之前几乎无法对模型进行 Putnam 评估，这使得该模型的泛化能力令人印象深刻。
- ****Transformer 尽管擅长近似但难以进行计算****：一位成员分享了一张图片，说明了为什么 **Transformer** 可能不是未来的架构，认为*尽管该架构能够很好地进行近似，但它本身排斥执行计算*，需要过度的计算力才能达到结果。
   - 另一名成员表示赞同，指出该模型擅长寻找正确的解决方案，但在应用方面却很吃力，并添加了一个 [tenor gif](https://tenor.com/view/awesome-ok-great-good-thumbs-up-gif-16351183)。
- ****Hugging Face 成为 AI 聚集中心****：在一位用户询问 *huggingface 是什么？看起来很酷，它像 openrouter 吗？* 之后，多位成员纷纷表示 *huggingface 是必知、必注册的 AI 聚集中心*，它是 GitHub 和容器租赁服务的结合体。
   - 他们还指出，即使是大公司也会毫不犹豫地将内容上传到那里。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1448067245639991307)** (20 条消息🔥): 

> `Hermes 4.3, KoboldCPP, SillyTavern, Nomos Tool Use, Model Performance` 


- **Hermes 4.3 获得关注**：成员们讨论了 **Hermes 4** 在角色扮演和创意写作方面的优点，以及自版本 4 以来是否有任何更新版本。有人提到 **Hermes 4.3** 是一个 32b 模型，更加紧凑。
   - 一位成员发现它非常适合写作，并使用 **SillyTavern** 和 **通过 API 调用的 Hermes 4 405b**。
- **KoboldCPP 前端使用困扰**：一位成员在搭载 **128 MB 统一内存** 的 **M4 Max Macbook Pro** 上本地运行 **Hermes 4 70B**，使用 **KoboldCPP**，有时使用 **SillyTavern** 作为前端。
   - 他们抱怨说 *KoboldCPP 的 UI 看起来非常陈旧，而且我只能通过强制退出来关闭它，但它确实能用*。
- **Token 生成速度辩论**：一位成员询问了 Token 生成速度，报告在统一内存系统 (**AMD 395**) 上仅为 **2 tokens/秒**。
   - 另一位成员报告获得了 **6.84 tokens/秒**。
- **Nomos 的工具使用能力受到质疑**：一位成员链接到了 [NousResearch/nomos GitHub 仓库](https://github.com/NousResearch/nomos)，并质疑 **Nomos** 是否可以使用工具。
   - 得到的澄清是 *它是一个仅用于数学的专业模型*。
- **GPU 与 API 部署**：一位成员提到他们可以在自己的 GPU 上以 **3 bits** 运行 **Hermes 4.3**，但它不会进入 API。
   - 有人对造假行为发表了评论，并附带了一个相关的 [YouTube 视频链接](https://www.youtube.com/watch?v=4lKyNdZz3Vw)，抱怨 *评论区真是一团糟*。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1448052051710967919)** (56 条消息🔥🔥): 

> `Eleven Labs Reader, Linux Foundation 规模, AI Agent 评估, Puppeteer vs Cypress, Latent Space 资源` 


- **Eleven Labs Reader 受到关注**：成员们分享了使用 Eleven Labs 的 **Eleven Reader** 的积极体验，指出它在朗读 GPT 讨论时能根据语境添加情感。
   - 虽然有些人使用移动端 App，但其他人讨论了在桌面端使用它或使用像 **OpenWebUI** 这样的自托管 TTS 解决方案，不过与 Eleven Labs 的定价相比，音频质量是一个令人担忧的问题。
- **Linux Foundation 不断演变的排他性**：一位成员在思考 **Linux Foundation** 现在是否感觉不那么排外了，这可能是一件好事。
   - 另一位成员回应说，它*大约有 100-200 人，但有些人负责运行项目等*，这些项目有准入标准，且项目处于成熟水平。
- **AI Agent 评估与 InfoSec**：一位工程师正致力于构建 **AI agents** 并为其编写评估，特别是针对 InfoSec 相关的应用。
   - 该成员分享了一个链接 [https://x.com/sonyatweetybird/status/1998456924359348271](https://xcancel.com/sonyatweetybird/status/1998456924359348271?s=46)，引发了进一步的讨论和分析。
- **相比 Puppeteer 和 Cypress，更倾向于 Playwright**：成员们讨论了使用 **Puppeteer** 和 **Cypress** 等测试工具来生成单元测试，由于 **Playwright** 的流行度以及 Claude 的调试能力，普遍情绪更倾向于它。
   - 有人提到 Cypress 有一个新的 [cy.prompt()](https://docs.cypress.io/api/commands/prompt) 功能看起来很有趣，但是，它确实需要订阅其云服务。
- **Latent Space 被推荐为关键 AI 资源**：对于寻找 AI 主题会议/研讨会/演讲的用户，推荐了位于 [https://lu.ma/ls](https://lu.ma/ls) 的 **Latent Space** 论文俱乐部和 [AI Engineer conference](https://ai.engineer)。
   - YouTube 上的 **Latent Space** 播客也因其能够接触到 AI 领袖和深刻的讨论而被推荐。*他们拥有令人羡慕的接触 AI 领袖的机会。Alessio 和 SWYX 拥有在整个行业都受人尊敬的深度知识和实战经验。*


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1448140716823023776)** (41 条消息🔥): 

> `ModelScope 偏见, RoR vs Node.js, 虚假 Nitter 截图` 


- **ModelScope 因火箭事故遭受抨击**：**ModelScope** 在其文本转视频模型生成了中国火箭爆炸的画面后面临批评，引发了关于偏见的指责。
   - 该公司为其模型辩护，断言其无偏见的性质，并引导用户通过 [Hugging Face](https://huggingface.co) 报告任何问题。
- **RoR 与 Node.js 的性能对决**：一条推文通过将 **Ruby-on-Rails (RoR)** 与 **Node.js** 进行性能对比引发了辩论。
   - 该推文询问哪种框架在速度上更胜一筹，并征求社区的意见和经验。
- **虚假 Nitter 截图引发警告**：@iamemily2050 的一条推文链接警告用户注意**虚假的 Nitter/前 Twitter 截图**。
   - 然而，分享的帖子中没有进一步的讨论或上下文。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1448044949227438111)** (50 条消息🔥): 

> `AI Slop, Brandolini's Law, OLMo-1 runs, Pythia eval dataset` 


- **关于 **AI Slop** 定义的辩论爆发！**：成员们就 **AI slop** 的定义展开了辩论，一位成员链接了关于此问题的两种观点：[Yuchenj_UW 的推文](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858) 和 [mlstreettalk 的推文](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46)。
   - 针对 AI 生成内容的**生产与验证成本的不对称性**提出了担忧，并引用了 [Brandolini's law](https://en.wikipedia.org/wiki/Brandolini's_law)。
- **EleutherAI 展现出**扎实的过往记录**！**：一位成员强调了 EleutherAI 在*识别、指导、资助和推广具有影响力的工作*方面的记录，并引用了 [用于可解释性的 SAEs](https://arxiv.org/abs/2309.08600) 和 [旋转位置编码扩展微调 (rotary extension finetuning)](https://arxiv.org/abs/2309.00071) 等例子。
   - 他们还提到了 [VQGAN-CLIP](https://arxiv.org/abs/2204.08583)，以及[一种在大规模下性能可媲美 Transformers 的 RNN 架构](https://arxiv.org/abs/2305.13048)。
- **解码 **OLMo-1 运行**！**：一位成员询问了两个 **OLMo-1 运行**版本之间的确切差异：[OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) 和 [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf)。
   - 一位成员澄清说，*它们是在不同的数据集上训练的，后者可能进行了额外的退火 (annealing)*。
- **寻找 **Pythia 评估数据集**位置！**：一位成员询问在哪里可以找到 **Pythia eval dataset**。
   - 另一位成员建议，*v1 提交版本的 gpt-neox 应该有一个默认的数据集划分种子，可以提供所使用的划分方式。*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1448041697820020978)** (42 条消息🔥): 

> `Deepseek v3.2, ARC-AGI, Adaptive AI, Thinking Machines Tinker Product` 


- **Deepseek 的索引器：轻量级的注意力抓取器**：一位成员指出 **Deepseek v3.2** 采用了一个 *O(n^2)* 的索引器来选择最重要的 Token 进行注意力计算，这可能会减少 Prefill 阶段的时间复杂度，尽管它并没有消除 *n^2* 操作。
   - 该索引器的速度归功于其轻量级设计和 8-bit 精度。
- **ARC-AGI 的隐藏宝藏引发辩论**：成员们讨论了来自 **ARC-AGI** 的一个项目，该项目通过有理域均衡（Rational Field Equilibrium）使用自适应 AI ([Adaptive AI through Rational Field Equilibrium Toward Gradient-Free and Energy-Efficient Intelligence](https://www.researchgate.net/publication/397181214_Adaptive_AI_through_Rational_Field_Equilibrium_Toward_Gradient-Free_and_Energy-Efficient_Intelligence))，其导师是来自 CMU 的 Albert Gu，但关于其改变范式的潜力存在分歧。
   - 有些人认为这是最有趣的项目，而另一些人则认为它被过度炒作了，尽管它获得了一个奖项并提出了一些由 ARC-AGI 激励的有趣想法 ([Thinking Machines 社区项目](https://thinkingmachines.ai/blog/call-for-community-projects/))。
- **注意力机制探讨**：成员们讨论了注意力机制，一位成员建议了一种方法，涉及对参与注意力的 Token 进行求和并取 Top K，以避免额外的索引器。
   - 另一位成员分享了一篇论文，建议使用分数乘以 alpha 加 m，并将 alpha 和 beta 设置为 *e^0.5*，tau 设置为 *10*，其中 t 是 Query 和 Key Token 之间的距离 ([一种提议的方法](https://arxiv.org/abs/2505.17083v1))。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1448114127947169955)** (1 条消息): 

> `Diffusion Models, Synthetic vs Naturalistic Data` 


- **针对 Diffusion Models 的机械可解释性分析**：一篇关于 **Diffusion Models 机械可解释性**的[新论文](https://arxiv.org/abs/2506.17237)进行了电路级分析和因果验证。
- **数据处理中的算法差异**：研究人员*发现了 Diffusion 架构在处理合成数据与自然数据分布时，在算法上的根本差异*。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1448167116925702186)** (70 messages🔥🔥): 

> `Mistral Vibe, Devstral model, GLM 4.6, iFlow, Qwen-Code` 


- ****Mistral Vibe** API 超时困扰**：成员们报告了 **Mistral Vibe** 的 API 频繁超时的问题，导致难以对模型进行评估，这与 [Mistral 最近的公告](https://x.com/mistralai/status/1998407337690710210?s=46)有关。
   - 尽管存在 API 问题，但小型 **Mistral** 模型的基准测试看起来很有前景，在消费级硬件上可能优于 **GLM 4.6**。
- ****iFlow** CLI 工具因免费使用受赞誉**：命令行工具 **iFlow** ([iflow.cn](https://iflow.cn/)) 是一个 Gemini CLI 的分支，因其免费使用且无使用限制而获得推荐。
   - 一位成员指出它偶尔会“发疯”，可能是由于 Alacritty 或 Zsh 的问题，但它仍然可靠，只是偶尔需要提醒它不要说中文。
- ****Kimi** 编程计划及其特性**：一位成员报告称正在使用 **Kimi** 进行编程，利用其与 Anthropic API 的兼容性来使用 Claude Code 而无需直接向 Anthropic 付费，并提到正在使用 Kimi For Coding 计划。
   - 一位用户报告了 **Kimi** 实现中存在一个持久的搜索 Bug。
- ****Crush CLI** 支持 Bring Your Own Key**：一位成员提到了 **Crush CLI**，将其作为连接 OpenAI 与 Anthropic 风格通信的一种方式，包括对 Ollama 等本地提供商的支持。
   - 它支持 **BYOK**，允许用户连接到各种提供商（包括本地提供商），尽管另一位用户表示：“如果我已经可以免费获得同等性能，我无法说服自己为在另一个工具中使用的模型付费”。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1448213358934360104)** (8 messages🔥): 

> `Free CUDA sites, Parallel GPU sort disagreements` 


- **用户寻找免费 CUDA 站点**：一位用户询问可以运行 **CUDA** 的免费站点，因为他们无法使用旧的 GPU，另一位用户建议了 **Google Colab**、**Tensara**、**LeetGPU** 和 **KernelBot**。
- **并行 GPU 排序引发 AI 争议**：关于布尔对排序（Boolean pair sorts）最快的并行 GPU 排序方法（排除 **Radix sort**）展开了讨论。
   - 一位参与者声称 **Bitonic sort** 比 **Merge sort** 慢，由于没有 **Sample sort** 的示例，他们正准备采用 **Merge sort**。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1448269757957541938)** (5 messages): 

> `PTXAS error with sm_103, Triton PTX codegen error, CUDA toolkit 12.9, Triton community meetup` 


- ****PTXAS** 错误在 **sm_103** 目标上仍然出现**：尽管 Triton v3.5.1 进行了修复，但一位用户报告了与 `sm_103` 架构相关的 **PTXAS** 错误，具体为 `Value 'sm_103a' is not defined for option 'gpu-name'`。
   - 该用户运行的是 **CUDA 13.0**，但 Triton 捆绑的 **PTXAS** 可能是原因，正如[此 issue](https://github.com/triton-lang/triton/issues/8473)所暗示的。
- **Triton 捆绑了过时的 **PTXAS****：该错误源于运行 Triton 捆绑的 **ptxas**，它默认附带基于 **CUDA 12.8** 的 **PTXAS** 版本，可能无法处理最新的架构。
   - 正如[此 Pytorch issue](https://github.com/pytorch/pytorch/issues/163801)中提到的，潜在的修复方法包括设置 `TRITON_PTXAS_PATH` 环境变量，使其指向较新 CUDA toolkit 安装中的 **PTXAS** 可执行文件。
- **Triton 社区将于 2026 年会面**：下一次 Triton 社区会议将于 **2026 年 1 月 7 日**太平洋标准时间上午 10 点至 11 点举行，会议链接可在 [Google 日历活动](https://tinyurl.com/48sb5pst)中找到。
   - 暂定议程是由来自 Meta 的 Corbin Robeck 和 Puyan Lotfi 演示/讨论后端扩展的细节。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1448230677890859049)** (1 messages): 

> `LLM Sparsification, Transformers Library, GPU Code Inspection` 


- **LLM 稀疏化探索开始**：一位具有 CUDA 背景的成员正尝试使用 **Hugging Face** 和 **PyTorch**，以探索 Transformers 库中 LLM 稀疏化带来的加速。
   - 他们正在寻求关于如何检查 GPU 代码（特别是 MLP 层）的指导，以便进行编辑和实验。
- **CUDA 用户转向 Transformers**：一位主要具有 CUDA 背景的用户现在正首次探索 **Hugging Face** 和 **PyTorch**。
   - 他们的目标是观察通过 Transformers 库提供的 LLM 稀疏化所带来的加速效果。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1448052652125454497)** (3 messages): 

> `Performance Engineers Hiring, High Compensation Packages, Silicon Valley Job Market` 


- **性能工程师需求量大**：一家公司正在招聘**性能工程师**（无论是否有 GPU 经验），并与 Silicon Valley 的顶尖公司合作。
   - 由于业务快速扩张，他们提供的总薪酬方案在 **$500K 到 $1M** 之间。
- **丰厚的薪酬吸引人才**：性能工程师的需求非常高，尤其是在 Silicon Valley，这导致了极具竞争力的薪酬方案。
   - 各大公司愿意支付 **$500K 到 $1M** 之间的薪酬来吸引该领域的顶尖人才。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448084603368771676)** (3 messages): 

> `Inference Serving, NPU Compiler Learning` 


- **推荐推理服务资源**：一位成员推荐阅读 **serverlessLLM** 和 **blitzscale** 作为推理服务的入门资料。
   - 他们指出这些资源更多地关注推理的*系统层面*。
- **初学者寻求 NPU 编译器教育**：一位成员表示自己是初学者，想要学习有关 **NPUs** (Neural Processing Units) 的编译器知识。
   - 给出的消息中没有提供具体的资源或建议。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 messages): 

walrus_23: 提交了一个小的文档更新 PR: https://github.com/pytorch/ao/pull/3480
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1448260888585834557)** (1 messages): 

> `ChatGPT memory, Blog post on ChatGPT's memory system` 


- **博文剖析 ChatGPT 记忆系统**：一位成员分享了一篇[博文](https://manthanguptaa.in/posts/chatgpt_memory/)，深入分析了 **ChatGPT 的记忆系统**。
- **ChatGPT 记忆博文反响良好**：作者报告称，他关于 **ChatGPT 记忆系统**的[博文](https://manthanguptaa.in/posts/chatgpt_memory/)获得了良好的反响。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1448066033993912330)** (5 messages): 

> `Register Best Practices, Mojo on Apple Silicon, AMD vs. Nvidia, PTX Registers` 


- **LowLevelML 分享寄存器最佳实践**：Turbintube 分享了他们关于[操作寄存器时的最佳实践](https://www.lowlevelml.com/blog/registers-best-practices)的文章。
   - 该文章基于作者的经验，重点关注 **AMD** 和 **Nvidia** 架构，同时推测这些原则也适用于 Apple 生态系统。
- **读者对缺乏 Apple Metal IR 表示惊讶**：一位读者注意到文中缺少 **Metal IR (.air)** 的示例，考虑到 **Mojo** 能够针对 Apple Silicon 进行优化。
   - 他们建议未来加入相关内容，而作者澄清其经验主要集中在 **AMD** 和 **Nvidia**。
- **Nvidia 会采用 AMD 的特性吗？**：一位用户表达了希望 **Nvidia** 采用 **AMD** 特性的愿望，特别是索引寄存器的机制和切片语法（slice notation）。
   - 他们认为这可能允许在不改变指令格式的情况下，实现**每个线程超过 255 个寄存器**。
- **关于 Nvidia 寄存器的澄清**：一位用户纠正了关于 **Nvidia 寄存器**的表述，澄清道：“每个寄存器由 Uniform General Purpose Register 或常规 General Purpose Register 支持。后者扮演的角色等同于 sgpr” 这里的后者应该是 **vgpr**。
   - 他们解释说 **sgpr** 实际上等同于 uniform register。
- **PTX 寄存器本质上是变量**：一位成员指出 **PTX "registers"** 本质上是变量，实际寄存器的分配由 `ptxas` 处理。
   - 这一区别强调了 **PTX 寄存器**并非直接映射到物理寄存器，而是符号表示。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1448072627745984735)** (5 messages): 

> `NVIDIA performance, nvfp4_gemm leaderboard updates, Submission results` 


- **NVIDIA 排行榜竞争激烈**：一位用户在 **NVIDIA** 榜单上获得了**第 4 名**，其提交在 `nvfp4_gemm` 排行榜上达到了 **10.9 µs**。
   - 另一位用户凭借 **36.0 µs** 的成绩刷新了在 **NVIDIA** 上的**个人最好成绩**。
- **成功提交冲向榜首**：一次提交在 NVIDIA 上达到了 **11.9 µs**，而另一次在 `nvfp4_gemm` 排行榜上达到了 **15.5 µs**。
   - 两次提交均获成功。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1448335990413332490)** (2 条消息): 

> `NCCL ranks falling out of sync, Troubleshooting NCCL ranks, Collective launch skew analyzer` 


- ****NCCL Rank 异步运行****：一位成员寻求关于排查 **NCCL ranks** 不同步问题的建议，并提到尝试将 PID 绑定到 CPU 和 NUMA 组但未获成功。
   - 该成员推测这是一个棘手的问题，并好奇其他推理引擎是如何解决的，同时分享了一张[图片](https://cdn.discordapp.com/attachments/1398843708488552570/1448335990132576327/image.png?ex=693ae380&is=69399200&hm=3e3429a27c683836ca38d39713d174bdfab583b65ecf3d13b33002dd7cd5d72e)，展示了 **rank 5** 在 AllReduce 之后启动明显滞后的情况。
- ****MixLayer 发布 NCCL Skew Analyzer****：一位成员分享了一个用于分析集合通信启动偏移（collective launch skew）的 **nsys dumps** 工具，名为 [nccl-skew-analyzer](https://github.com/mixlayer/nccl-skew-analyzer)。
   - 这是为了回应之前关于 **NCCL ranks** 不同步的问题。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448064230401114224)** (1 条消息): 

> `Helion webinar, PTC launch, Helion kernels` 


- **Helion 网络研讨会定于 12 月 11 日举行**：一场包含实时问答环节的 **Helion 网络研讨会** 计划于 **太平洋标准时间 12 月 11 日星期四上午 11 点** 举行，旨在讨论自 **PTC launch** 以来的进展。
   - 研讨会将涵盖开发、调试和部署 **Helion kernels** 的最佳实践，并附带了 [YouTube 链接](https://www.youtube.com/watch?v=_gIyr1BVUJk)。
- **Helion Kernels：最佳实践讨论**：即将举行的 **Helion 网络研讨会** 将深入探讨开发、调试和部署 **Helion kernels** 的最佳实践。
   - 鼓励参与者在演讲后的实时问答环节中提出问题。


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1448203791538196542)** (22 条消息🔥): 

> `Benchmark performance swings, Mojo kernel from Python submission, Benchmarking cuBLAS on GEMM, torch._scaled_mm and cuBLAS, Discord bot error` 


- **Benchmark 结果剧烈波动**：用户报告称，即使重复运行同一个文件，Benchmark 性能也会剧烈波动，导致难以确定准确结果；而另一些用户发现[网站排行榜提交](https://example.com)的结果更为一致。
   - 主要问题在于 runner 由 Nvidia 管理，因此无法直接添加新的依赖项，但如果 **Mojo** 可以通过 pip 安装，则可以在提交文件中运行子进程命令。
- **提交结果超越 cuBLAS 性能**：有观点指出，顶尖的提交在 **GEMM** 问题上的表现优于 **cuBLAS**，一位用户报告使用 **cuBLAS** 获得了约 **15us** 的成绩。
   - 有推测认为，达到约 **13us** 的提交可能也利用了 **cuBLAS** 或类似工具，这可能是可以实现的。
- **关于在 PyTorch 中使用 cuBLAS 的讨论**：讨论围绕 **cuBLAS** 的使用展开，澄清了 `torch._scaled_mm` 是由 **cuBLAS** 支持的，用户可以直接调用它，并引用了关于 B200 上 **cuBLAS** 分块缩放（blockwise scaling）支持的 [issue](https://github.com/pytorch/pytorch/issues/153555)。
   - 进一步的讨论指向了 **DeepSeek** 风格的分块缩放，其中 `mxfp4_mxfp4` 使用 `fbgemm_gpu` API，而 `_scaled_nvfp4_nvfp4` 通过 `at::cuda::blas::scaled_gemm()` 调用 **cuBLASlt**。
- **Discord 机器人错误困扰**：一位用户报告在使用 Discord 机器人时遇到 “发生意外错误。请向开发人员报告” 的消息，导致 Benchmark 提交结果不一致。
   - 另一位用户请求提供所使用的文件和命令，以便尝试调试该问题。


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1448060496401006633)** (3 条消息): 

> `llia larchenko X post` 


- **发现 X 帖子**：一位成员分享了 Ilia Larchenko 在 X 上发布的一条帖子链接（[链接](https://x.com/ilialarchenko/status/1998384056439017826)）。
   - 另一位成员确认他们已经看到并将进行审查，并指出其中有 “一些非常有趣的抉择”。
- **审查进行中**：另一位成员确认他们已经看到并将进行审查。
   - 他们指出其中有 “一些非常有趣的抉择”。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1448046409604403352)** (34 条消息🔥): 

> `Anthropic 捐赠给 Linux Foundation，Arrow 与 Parquet 文件格式对比，开源 LLM 的 Tool Calling，Unsloth 加速训练，轻量级 Vision Transformer 模型` 


- **Anthropic 加入 Linux：全新的 Model Context Protocol！**: **Anthropic** 正在将其 [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) 捐赠给 **Linux Foundation**，并成立了 **Agentic AI Foundation**。
- **Arrow vs Parquet：一场关于文件格式的争论！**: 一位成员指出了 [Hugging Face 文档](https://huggingface.co/docs/datasets/v4.4.1/loading#arrow) 中关于文件格式的一个拼写错误，指出 **Parquet** 是一种 *压缩* 格式，而非 *未压缩* 格式。
- **Ollama 和 vLLM 简化了 Tool Calling**: 一位成员询问了使用开源 **LLM** 运行 tool calls 的最简单方法，另一位成员建议本地设置使用 **Ollama** ([文档](https://docs.ollama.com/))，可扩展方案使用 **vLLM** ([文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))，两者都支持 OpenAI 风格的 tool/function calling ([Anyscale 文档](https://docs.anyscale.com/llm/serving/tool-function-calling))。
- **Unsloth 通过新内核释放极速训练能力**: **Unsloth** 团队 [在 X 上](https://x.com/UnslothAI/status/1798765021170696664) 宣布支持使用新内核和无污染打包（uncontaminated packing）进行更快速的训练。
- **寻找轻量级 ViTs**: 一位成员正在寻找参数量少于 50 万、在 ImageNet 数据集上训练的轻量级 **Vision Transformer (ViT) 模型**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1448312858558599350)** (1 条消息): 

> `Token 吞吐量，Qwen3 模型` 


- **Qwen3 实现高 Token 吞吐量**: 一位成员报告称实现了每月约 **10T tokens** 的吞吐量。
   - 根据附带的截图显示，他们使用的是 **A3B** 尺寸的 **Qwen3 30B** 模型。
- **附图展示了 Qwen3 的设置**: 用户上传了多张截图，提供了关于其 **Qwen3** 设置和结果的更多背景信息。
   - 这些图片 (SCR-20251210-ngia.png, SCR-20251210-newq.png, SCR-20251210-neyu.png, SCR-20251210-nexw.png) 可能包含与 **10T tokens 吞吐量** 声明相关的性能指标或配置。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1448057057256149002)** (4 条消息): 

> `retrain-pipelines, GOSIM Foundation, AI 语音聊天, WebGPU, GLM ASR 模型` 


- **Retrain-Pipelines 演讲发布**: 一位成员宣布，他们在去年 9 月杭州举行的 **GOSIM Foundation** 会议上关于 **retrain-pipelines** 的演讲已经发布，其中很大一部分内容涉及其 [Hugging Face Hub 集成](https://huggingface.co/retrain-pipelines)。
   - 录像已上传至 [YouTube](https://www.youtube.com/watch?v=nmrMachM5aM)，[幻灯片](https://docs.google.com/presentation/d/1hnAzHJ0SbeAOtGJir-iH84RBtXT1OxVT/) 也可以访问。
- **完全在浏览器中运行的 AI 语音聊天**: 一位成员分享了一个 **AI 语音聊天** 演示，它使用 **WebGPU** 100% 在浏览器中运行，无需将数据发送到任何第三方 API 或服务器，确保了隐私和安全，链接见 [HuggingFace Spaces](https://huggingface.co/spaces/RickRossTN/ai-voice-chat)。
   - 包括 **STT**, **VAD**, **TTS** 和 **LLM** 在内的所有组件都在加载的页面内运行。
- **新款 GLM-ASR-Nano 模型亮相**: 一位成员分享了一个 Space 来测试新的 SOTA 级别 **GLM ASR 模型**，据称其表现优于 Whisper，链接见 [HuggingFace Spaces](https://huggingface.co/spaces/YatharthS/GLM-ASR-Nano)。


  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1448447731080364042)** (3 条消息): 

> `Diffusion Models Study Group, Transformer Architecture Workshop, Diffusion Transformers Workshop` 


- ****Diffusion Models 学习小组将于 2026 年 1 月启动****：受 [MIT Diffusion 课程](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf.) 启发，一个为期 **3 个月、规模为 12 人** 的学习小组将于 **2026 年 1 月** 启动，旨在研究 Diffusion Models 和 Transformers。
   - 该学习小组包括导师指导下的同行互助会议、真实研究论文讨论，以及实战项目和代码走读。
- ****Transformer 架构研讨会定于 12 月 13 日举行****：关于 **Transformer Architecture** 和 ***Attention Is All You Need*** 论文的入门研讨会将在 **12 月 13 日** 举行 ([luma.com/kqjrf0uw](https://luma.com/kqjrf0uw))。
   - 研讨会旨在教授核心 Transformer 架构、Attention 机制，以及为什么这篇关于 Attention 的论文是现代 LLM 和多模态模型的基础。
- ****Diffusion Transformers 研讨会计划于 12 月 20 日举行****：关于 **Diffusion Transformers** 的研讨会（包括论文走读和代码实现）定于 **12 月 20 日** 举行 ([luma.com/lr2qvveq](https://luma.com/lr2qvveq))。
   - 参与者将走读一篇 **Diffusion Transformer** 论文并用代码实现核心思想，将 Diffusion Models 与 Transformer 架构联系起来。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 条消息): 

erdong_43406: 大家好。
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1448082791186038837)** (17 条消息🔥): 

> `SI law, Superintelligence, AI Scams, AI HR, Generative Rehearsal Technique` 


- **Dettmers 明确反驳数字末日论**：一位成员分享了 [Tim Dettmers 的博客文章](https://timdettmers.com/2025/12/10/why-agi-will-not-happen/)，论证了为什么 **AGI 不会发生**。
   - 该成员使用了 <:yann:1441889312697483295> 表情符号，暗示 Yann LeCun 可能会对此持反对意见。
- **超级智能预计到达时间：80 年？**：一位成员推测，通过模拟世界的数据分布并在微调前使用 [生成式排练技术 (Generative Rehearsal Technique)](https://openreview.net/forum?id=ohmo21slB3) 来对抗灾难性遗忘，实现超级智能需要 **80 年**。
   - 他们理论上认为，模型与数据分布的不匹配在微调后会放大，从而导致较小规模的遗忘。
- **Discord 开发者发布可疑细节，引发不信任**：多位成员讨论了最近 AI 和 App 开发者在 Discord 上发布完全相同的广告消息的趋势，他们怀疑这是一种 **诈骗**，目的是诱骗狂热的年轻 AI 爱好者免费工作。
   - 一位成员在谈到诈骗者时表示：*“你会认为如果他们是正规的，他们会分享自己的 GitHub 或网站。”*
- **AI Agent 自动化 HR**：一位成员认为 Discord 上的机器人垃圾信息是针对 **AI HR** 的 **机器人对机器人的自然战争**。
   - 他们对垃圾信息表示恼火，尤其是在私有服务器上，但觉得“—— 高级 AI 与 App 开发者 ——”这个头衔很有趣。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 条消息): 

burnytech: 该死，可能和我别的安排冲突了。
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1448057659558330560)** (24 messages🔥): 

> `中国稀土矿产控制，Mistral Dense 与 MoE 模型，Mixtral 初始化，Mistral Devstral-2 Vibe CLI，Mistral 欧盟寡头` 


- **中国加强 H200 采购限制**：中国显然正在加强其国内监管；公司必须注册才能购买 **H200**，并证明本地替代方案不够好。
   - 一位成员讽刺地调侃道 *“这简直是半导体领域的‘大豆换芯片’贸易协议”*，另一位成员指出 [反正 eBay 上大多数 H100 都来自中国](https://www.ebay.com/sch/i.html?_nkw=h100)。
- **Mistral 训练 Dense 模型以简化微调**：成员们讨论了为什么 **Mistral** 正在训练 Dense 模型而不是 **MoE** 模型，推测 Dense 模型更容易微调，适合那些托管模型并在自己的数据和代码库上进行训练的公司，且因为他们的目标是 [本地部署和自定义微调](https://mistral.ai/news/devstral-2-vibe-cli)。
   - 一位成员推测他们可能正在进行新的 **MoE vs Dense 比较**，因为 MoE 是极佳的知识海绵，在评估（evals）中具有优势，但产生的“涌现时刻”较少。
- **Mixtral 是否从 Mistral 7B 的 Checkpoints 初始化？**：一位成员询问第一版 **Mixtral** 是否是从 **Mistral 7B** 的 Checkpoints 初始化的，旨在对该原理进行改进。
   - 另一位成员回答说，大多数现代 MoE 确实非常稀疏（sparse），而早期的 Mistral 模型是“粗粒度”的 MoE，且 [Llama 4](https://ai.meta.com/research/updates/llama-2/) 也被认为是粗粒度 MoE。
- **Mistral 受益于欧盟 AI 法案形成的寡头垄断**：一位成员认为，由于最近的 AI 法律，**Mistral** 在欧盟意外获得了寡头地位，这对他们的成功起到了重要作用。
   - 还有人指出 *在某些方面，他们比其他一些主流 AI 公司更具实验性*，并指出了他们早期对 **Mamba** 和 **Mamba 混合模型** 的采用。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

jokellum: <@&1116225504563970138>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1448078594097156210)** (34 messages🔥): 

> `用于 Mojo 的嵌入式 AI 开发板，使用 Pixi 移除系统安装的 Mojo，Qwen3 模型可用性，Mojo 元编程中函数检查和操作的路线图，Mojo 中的内存分配控制` 


- **Jetson Orin Nano 加速 Mojo 嵌入式 AI**：对于使用 Mojo 进行嵌入式 AI 开发，配备 **Ampere 级 GPU** 的 **Jetson Orin Nano** 得到了全面支持（如果其尺寸合适的话）。
   - 然而，**Beaglebone Black** 可能不兼容，因为它采用的是 **ARM32** 架构，且 Linux 版本可能过旧。
- **以 Pixi 方式从系统路径清除 Mojo**：为了在更倾向于使用 Pixi 时移除系统安装的 Mojo，成员建议直接删除 Mojo 可执行文件（`sudo rm -rf mojo`），或将其移动到其他地方作为备份。
   - 一位成员指出那个版本已经 *非常陈旧* 了。
- **缺失 Qwen3-Coder 模型引发猜测**：一位用户询问为什么 [Modular 模型构建页面](https://builds.modular.com/models/Qwen3/8B) 上没有 **Qwen3-Coder**，质疑为什么只有原始的 **Qwen3/8B** 模型可用。
   - 一位成员建议改用 **Ollama**。
- **Mojo 路线图忽略了元编程的函数检查**：一位用户注意到 Mojo 路线图中缺少对 **元编程中函数检查和操作** 的支持，特别是针对编译时的类 JAX 函数变换。
   - 一位 Modular 团队成员澄清说，Mojo 1.0 之后的路线图尚不明确，邀请在论坛上提交具体提案以展示其价值；该团队成员确实表示这可能不会出现在 Mojo 1.0 中。
- **自定义分配器和 CUDA 集成即将推出**：一位贡献者表示，分配器的工作正在进行中，目前受限于参数化特性（parametric traits），以支持像 `cudaMallocManaged` 这样用普通内存补充 VRAM 的功能。
   - 他们表示 Mojo 默认采用栈分配（stack allocation），并通过 `stack_allocation` 提供等效于 `alloca` 的功能，且不像 Zig 那样需要在结构体中包含用于分配器状态的 vtable。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1448097675139481732)** (5 条消息): 

> `新应用发布，项目崩溃，网站创建交易` 


- **新应用寻求下载和评价**：一名成员发布了一款新应用，并请求通过下载和评价提供支持，同时提供了 [Play Store 链接](https://play.google.com/store/apps/details?id=info.alie.app)。
   - 该用户对社区在帮助推广新应用方面所付出的时间和支持表示感谢。
- **项目崩溃；请求恢复协助**：一名成员报告称其项目之一已停止运行，理由是 webdev 服务器崩溃且无法从 checkpoint 恢复。
   - 他们被指示带着 checkpoint 联系 Manus 团队，以便协助在后台进行恢复。
- **为初创公司提供免费网站以换取视频证言**：一名成员提议一项交易，即**为初创公司免费创建网站**，以换取**视频证言**。
   - 他们链接了自己的网站 [minderfly.com](https://minderfly.com)，以展示其服务并征求对该提议的反馈。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1448055611068780644)** (4 条消息): 

> `LF 迁移，治理` 


- **关于 LF 迁移影响的推测**：成员们想知道向 **LF**（推测为 Linux Foundation）的迁移将如何影响正在进行的工作。
   - 产生了一些疑问，包括项目是否会转向标准的 LF “运作方式”，以及关于迁移的 ETA 和结构的推测。
- **变革中治理结构保持不变**：一名成员根据最近的博客文章和公告澄清，治理及该范畴下的各个方面预计不会发生变化。
   - 他们引用道：*根据博客和 David 在公告频道的消息，我的理解是治理以及该范畴下的一切都不会改变。*


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1448458357814984828)** (2 条消息): 

> `Windsurf 1.12.41 发布，Windsurf Next 功能，Windsurf 登录恢复` 


- **Windsurf 稳定性大幅提升**：版本 **1.12.41**（以及 **1.12.160**）已发布，在稳定性、性能和错误修复方面有显著增强，详见 [更新日志](https://windsurf.com/changelog)。
   - 更新内容包括用于管理 MCPs 的新 UI、针对 GitHub/GitLab MCPs 的修复，以及对 diff zones、Tab 和 Hooks 的改进。
- **Windsurf Next：Lifeguard 和 Arena Mode 预览**：Windsurf Next（预发布版本）推出了令人兴奋的预览功能，如 **Lifeguard**、**Worktrees** 和 **Arena Mode**。
   - 这些新功能承诺带来更具创新性和高效的 Windsurf 体验。
- **Windsurf 登录功能恢复**：在短暂的维护窗口后，登录功能已恢复，如 [状态页面](https://status.windsurf.com/) 所示。
   - 用户现在可以无中断地访问 Windsurf 服务。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1448406965184368640)** (1 条消息): 

> `DSPy, OpenAI, GPTs, Adapter` 


- **DSPy 不仅限于 OpenAI**：DSPy 并不绑定于 **OpenAI**，这意味着在 **GPTs** 上运行良好的方案在其他 LMs 上可能效果不佳。
   - 用户可以实现自定义 [Adapter](https://dspy.ai/api/adapters/Adapter/)，在 system prompt 中格式化 few-shots，并针对 user/assistant 方法进行基准测试。
- **建议实现 Adapter**：为了让 DSPy 更好地适配非 OpenAI 的 LMs，建议实现自定义 [Adapter](https://dspy.ai/api/adapters/Adapter/)。
   - 这允许在 system prompt 中格式化 few-shots，并与 user/assistant 方法进行对比测试，从而可能提高在不同模型上的性能。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1448423539425280030)** (1 条消息): 

> `tinygrad PR 13553, GPU 加速` 


- **PR #13553 解决问题**：最新的 [GitHub Pull Request](https://github.com/tinygrad/tinygrad/pull/13553) 解决了遗留问题，现在可以在 **Zen4** 和 **M2** 架构上运行。
   - 该更新解决了之前发现的问题，确保了在不同硬件平台上的兼容性。
- **GPU 加速**：此主题涵盖了 tinygrad 中的 GPU 加速。
   - 讨论内容包括使用 GPU 来加速计算。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/)** (1 条消息): 

pierrunoyt: hi
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1448445571508600983)** (1 条消息): 

> `Diffusion Models 学习小组, Transformer 架构工作坊, Diffusion Transformers 工作坊` 


- **Diffusion Models 学习小组正式启动**：受 MIT 的 Diffusion 课程启发，一个为期 **3 个月、由 12 人组成的研究小组**将于 **2026 年** 1 月开始，从基本原理深入到 **Diffusion Models** 和 **Transformers** 的实际应用。
   - 成员将包括 **AI 电影初创公司的 CTO、LLM 教育者以及全职 AI 研究员**。
- **Transformer 架构工作坊发布**：关于 **Transformer Architecture** 和 **Attention Is All You Need** 论文的入门工作坊将于 **12 月 13 日**举行 ([链接](https://luma.com/kqjrf0uw))。
   - 该工作坊旨在教授 **Transformer 核心架构**和 **Attention 机制**，并解释为什么这篇论文是现代 **LLMs** 和 **Multimodal Models** 的基石。
- **Diffusion Transformers 工作坊即将举行**：关于 **Diffusion Transformers** 的入门工作坊将于 **12 月 20 日**举行 ([链接](https://luma.com/lr2qvveq))。
   - 参与者将深入研读一篇 **Diffusion Transformer 论文**，并用代码实现其核心思想，将 Diffusion Models 与 Transformer 架构联系起来。


  

---