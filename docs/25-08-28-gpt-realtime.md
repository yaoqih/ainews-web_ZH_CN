---
companies:
- openai
- xai
- microsoft
- google
date: '2025-08-28T08:44:39.731046Z'
description: '**OpenAI** 已将 **gpt-realtime** 模型和 **Realtime API** 推向正式发布（GA）阶段，其特性包括先进的语音对语音功能、全新的音色（**Cedar**、**Marin**）、图像输入支持、SIP
  电话集成，以及约 20% 的降价。基准测试显示，该模型在 BigBench 和 ComplexFuncBench 上的表现优于 **gpt-4o-realtime**。


  **xAI** 推出了 **Grok Code Fast 1**，这是一款针对速度优化的编程模型，已集成到主流 IDE 中；与此同时，**OpenAI Codex**
  针对本地和云端开发工作流进行了重大升级。谷歌的 **Gemini CLI** 改进了多编辑器支持，微软也发布了 **MAI-1-preview** 和 **MAI-Voice-1**
  等新模型。*“全新的全功能 WebRTC API 移除了临时令牌（ephemeral token）步骤，并支持在同一连接上进行视频传输，”* 这一更新突显了开发者工具的进一步增强。'
id: MjAyNS0w
models:
- gpt-realtime
- gpt-4o-realtime
- grok-code-fast-1
- codex
- mai-1-preview
- mai-voice-1
- gemini-cli
people:
- swyx
- juberti
- omarsar0
- reach_vb
- pbbakkum
- skcd42
- mohitreddy13
- cline
- kevinweil
- gdb
- sama
- _philschmid
title: OpenAI Realtime API 正式发布，并推出全新 `gpt-realtime` 模型，价格比 GPT-4o 便宜 20%。
topics:
- speech-to-speech
- instruction-following
- function-calling
- telephony
- webrtc
- voice-agents
- multilingual-switching
- voice-control
- benchmarks
- coding-models
- ide-integration
- developer-tools
- model-updates
---

**Realtime is all you need?**

> 2025年8月27日至8月28日的 AI 新闻。我们为您检查了 12 个 subreddit、544 个 Twitter 和 22 个 Discord（185 个频道和 7363 条消息）。预计节省阅读时间（以 200wpm 计算）：577 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 呈现方式。请访问 https://news.smol.ai/ 查看完整的新闻分析，并在 @smol_ai 上向我们提供反馈！

Realtime API 之前处于预览阶段，现在已正式发布（GA），支持 [图像输入](https://openai.com/index/introducing-gpt-realtime/#image-input)、[远程 MCP server 支持](https://openai.com/index/introducing-gpt-realtime/#remote-mcp-server-support)、[SIP/PBX 支持和 prompt caching](https://openai.com/index/introducing-gpt-realtime/#additional-capabilities)，以及更好的 [function calling](https://openai.com/index/introducing-gpt-realtime/#function-calling)。与此同时，还推出了一个新的 realtime 模型！遗憾的是不是 gpt5-realtime……它仍然是一个略微更聪明的模型，只是大部分改进都集中在“以 API 为中心”，即 function calling/指令遵循。

新增了 2 种声音，语音控制虽然难以量化，但 [值得一试](https://x.com/swyx/status/1961124194789499233)：


![](https://resend-attachments.s3.amazonaws.com/Gbiyb8nvjPj1KGI)


---

# AI Twitter 综述

**OpenAI 的 gpt-realtime 和 Realtime API GA（语音 Agent、电话、工具）**

- **gpt‑realtime 模型 + Realtime API GA**：OpenAI 发布了其最先进的语音对语音模型，并将 Realtime API 推向 GA，带来了重大的能力和成本更新。亮点：改进的指令遵循、tool calling、韵律和非语言线索、多语言切换；新声音（**Cedar**、**Marin**）；图像输入；远程 MCP 工具支持；SIP 电话；新的 WebRTC API（服务器 websocket 控制、视频）以及约 20% 的降价。社区分享的价格：约 $32/1M 音频输入 token（可缓存，价格为 $0.40/1M）和 $64/1M 音频输出 token。与 GPT‑4o‑realtime 的基准测试对比显示，在 BigBench、ComplexFuncBench 和音频指令遵循方面有显著提升。Demo 包括 Notion MCP 示例和 WebRTC/SIP 入门代码。推文：[@OpenAI](https://twitter.com/OpenAI/status/1961110295486808394), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961124915719053589), [API 详情由 @juberti 提供](https://twitter.com/juberti/status/1961116594211364942), [价格由 @omarsar0 提供](https://twitter.com/omarsar0/status/1961117107417928047), [基准测试看法由 @reach_vb 提供](https://twitter.com/reach_vb/status/1961140618295394579), [MCP demo 由 @pbbakkum 提供](https://twitter.com/pbbakkum/status/1961120041799487654)。
- **开发者笔记**：新的全功能 WebRTC API 移除了临时 token 步骤，并在同一连接上支持视频；SIP 端点为生产级通话流程启用了呼叫路由、转接和挂断 API。Cookbook 指南涵盖了语音 prompt 设计（速度、语调、交接）。参见 [WebRTC API 更新](https://twitter.com/juberti/status/1961118374345241016) 和 [SIP 详情](https://twitter.com/juberti/status/1961118371090501972)。

**编码模型和开发工具：xAI 的 Grok Code Fast 1、OpenAI Codex、编辑器/CLI**

- **xAI 的 Grok Code Fast 1**：一个“速度优先”、经济实惠的推理模型，用于 Agentic 编码，免费提供一周，并集成在流行的 IDE/工具中（GitHub Copilot、Cursor、Cline、Kilo Code、Roo Code、opencode、Windsurf）。团队强调了快速迭代发布以及针对基准测试之外的实际用途进行的人工+自动评估。社区测试结果积极，Cline 增加了“三种免费编码方式”（通过 Grok 的云端、通过 LM Studio 的本地，或具有慷慨每日限制的 Qwen Code）。公告和背景：[@xai](https://twitter.com/xai/status/1961129789944627207), [@skcd42](https://twitter.com/skcd42/status/1961132126298157060), [@MohitReddy13](https://twitter.com/MohitReddy13/status/1961138324426690608), [@cline 发布推文](https://twitter.com/cline/status/1961201105729401060)。
- **OpenAI Codex 推进（新的技术栈集成）**：OpenAI 的 Codex 获得了重大升级：IDE 扩展（Cursor/VSCode/Windsurf）、大幅改进的本地 CLI、统一的本地+云端任务管理，以及 GitHub 代码审查。评论指出在整个开发栈中实现了更深层次的集成，包括本地/远程工作流。反馈表明受到了强烈欢迎。推文：[@kevinweil](https://twitter.com/kevinweil/status/1960854500278985189), [@gdb](https://twitter.com/gdb/status/1960900413785563593), [@sama](https://twitter.com/sama/status/1961096744533647501)。
- **生态系统改进**：Google 的 Gemini CLI 在 Zed 中实现了原生集成（多文件夹 IDE 模式、diff 统计、更好的稳定性；社区驱动的 PR），简化了多编辑器工作流 ([@_philschmid](https://twitter.com/_philschmid/status/1961090847174262937))。OpenAI 的 Realtime GA 还解锁了语音优先的编码助手（基于语音的 MCP）。

**新模型与基准测试：Microsoft MAI、Cohere Translate、腾讯 TV2A、GLM‑4.5**

- **Microsoft MAI‑1‑preview (text) 和 MAI‑Voice‑1**：Microsoft 推出了其首批自研模型。MAI‑1‑preview 在首秀中进入 LMArena 文本排行榜第 13 名；MAI‑Voice‑1 针对高质量语音生成（鼓励公开测试）。Microsoft 信号表明将通过其产品界面进行快速迭代和分发。详情：[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1961111770422186452), [@lmarena_ai](https://twitter.com/lmarena_ai/status/1961112908026593557), [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1961112928230461615)。
- **Cohere Command A Translate**：一个任务专用翻译模型，拥有来自 RWS/Language Weaver 的强大第三方验证。社区反应是，在复杂的多领域任务中，经过领域训练的翻译模型优于前沿通用模型（甚至是 GPT‑5）。更多信息见 [Cohere 的博客](https://twitter.com/cohere/status/1961081787674763525) 以及 [@nickfrosst](https://twitter.com/nickfrosst/status/1961093091554713686) 的社区观点。
- **Tencent HunyuanVideo‑Foley (TV2A)**：端到端文本/视频转音频框架，在约 10 万小时数据上训练，采用 MMDiT 骨干网络、REPA 损失和 Audio VAE——在音频质量、视觉语义和时间对齐方面报告了 SOTA。代码、报告和 HF 权重已公开（[公告](https://twitter.com/TencentHunyuan/status/1960920482779423211)）。
- **智谱 AI GLM‑4.5**：目前在 Berkeley 的 Function‑Calling 排行榜 V4 中领先，强化了 GLM‑4.5 在实际 API 调用任务中的工具使用能力（[结果](https://twitter.com/Zai_org/status/1961149535754858586)）。

**Agent 系统、评估与模式**

- **并行 Agent 作为扩展轴**：吴恩达（Andrew Ng）强调并行 Agent 编排是第四个扩展杠杆（继数据、训练算力、推理时算力之后）。随着 Token 价格下降和延迟预算收紧，预计会有更多多 Agent 研究/指南（研究 Agent、后台工作程序 + UI 监控器、Mixture‑of‑Agents 聚合器）（[推文](https://twitter.com/AndrewYNg/status/1961118026398617648)）。
- **Memory‑R1（针对有记忆 Agent 的 RL）**：GRPO 变体通过结果驱动的奖励和极小的数据量（152 个 QA 对），显著提升了内存基准测试（涵盖 Llama‑3.1‑8B 和 Qwen‑2.5‑7B）上的 F1/BLEU/LaaJ。收益随更强的内存管理器而复合；可跨骨干网络泛化。笔记与链接：[@omarsar0](https://twitter.com/omarsar0/status/1961073807537693072)。
- **Agentic RAG 与可评估性**：Elysia（开源 Agentic RAG）使用决策树架构、动态数据展示、按需分块和反馈即 Few-shot 来提高确定性和可调试性（[概述](https://twitter.com/victorialslocum/status/1961095661719359624)）。LlamaIndex 发布了一个多 Agent “编码 Agent”，可自动生成文档工作流（编辑/测试/配置，代码优先，通过 LlamaIndex 工作流编排）（[演示](https://twitter.com/jerryjliu0/status/1961123785597505603)）。AI SDK v5 增加了 LangSmith 追踪以实现可观测性：Token 使用情况、工具追踪、TTFT（[@Hacubu](https://twitter.com/Hacubu/status/1961103113122984202)）。为了进行严谨的搜索增强评估，Reka 发布了 Research‑Eval（374 个多样化、高质量问题；前沿模型准确率分布在 26.7%–59.1%），旨在超越已饱和的 SimpleQA/BrowseComp（[@RekaAILabs](https://twitter.com/RekaAILabs/status/1961192688029765936)）。
- **DSPy 实践**：关于以数据为中心的流水线以及在何处将 LLM 引入循环的精彩讨论；在自动化之前通过规范/评估进行优化（与 @lateinteraction 的炉边谈话）（[会议](https://twitter.com/sh_reya/status/1961110090314125524)）。

**图像/视频生成：Nano Banana 势头、字节跳动 USO、Runway 投入生产**

- **Nano Banana (Gemini 2.5 Flash Image) 作为开发主力**：社区广泛用于个性化风格、分镜提示词（panel prompting）和移动端工作流；黑客松已宣布；Google 展示了“banana”背后的内部团队工作。示例包括 Demis（等轴测地图→游戏创意）、创意流水线（glif Agent；用于音频的 Suno）以及加速采用的免费技巧/促销。示例：[@demishassabis](https://twitter.com/demishassabis/status/1961077016830083103), [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1961127857192673540), [@tulseedoshi](https://twitter.com/tulseedoshi/status/1961068980640108889)。
- **ByteDance USO (Apache‑2.0) 风格迁移/编辑**：开源的文本+图像驱动编辑，效果“出奇地好”，带有 HF Demo 且获得了从业者的强力定性反馈；在“nano banana”时代是一个可靠的开源替代方案 ([概览](https://twitter.com/multimodalart/status/1961147988258295893))。
- **生产流水线中的 Runway Gen‑4**：与 Fabula 的电影制作合作伙伴关系说明了上下文工具如何增强专业工作流而非取代工艺——案例研究展示了提示词与生产现实的交汇点 ([@runwayml](https://twitter.com/runwayml/status/1961088220571066620))。此外：测试 Wan 2.2 S2V 表明音频预处理/微调对于音乐对齐仍然很重要 ([@ostrisai](https://twitter.com/ostrisai/status/1960907113821298877))。另外，Moonshot 的 Kimi Slides 推出了 Agent 式幻灯片构建（创意→幻灯片，未来将支持自动图像搜索/布局/润色） ([@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1961011693745811542))。

**基础设施与战略**

- **算力建设**：报告显示 OpenAI 和 Oracle 正计划建设一个 4.5 GW 的数据中心（Stargate），此前已有一个 1.2 GW 的 Abilene，合作伙伴包括 SoftBank/Microsoft/NVIDIA；传闻年合同额达 $30B。选址正在进行中 ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1960900145421177053))。
- **平台份额作为国家战略**：一篇政策贴认为，美国的领先地位需要最大化在美国硬件/软件上的使用量（Token, Model, Developer）——支持开发者飞轮，而非无意中催生替代技术栈（Huawei+CloudMatrix+DeepSeek/Qwen）的出口管制 ([@sriramk](https://twitter.com/sriramk/status/1961072926561550366))。相关的元观察：实验室都在同样的互联网数据上进行预训练，但强化学习和后训练的选择（以及产品数据）驱动了“物种形成” ([@tszzl](https://twitter.com/tszzl/status/1960953564681134472); [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1961121746670817404))。

**热门推文（按互动量排序）**

- xAI 发布了 Grok Code Fast 1（在主流 IDE 中免费使用 7 天） [@xai](https://twitter.com/xai/status/1961129789944627207)
- OpenAI 关于 Realtime API 和 gpt‑realtime 的“开发者请关注”直播 [@OpenAI](https://twitter.com/OpenAI/status/1961081377174212979)
- OpenAI 推出 gpt‑realtime 并正式发布 (GA) Realtime API [@OpenAI](https://twitter.com/OpenAI/status/1961110295486808394)
- Karpathy 谈论将教科书和环境“LLM 化”以获取对齐的训练数据 [@karpathy](https://twitter.com/karpathy/status/1961128638725923119)
- “Nano Banana”社区热潮和黑客松宣布 [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1961127857192673540)；Demis 的等轴测地图帖子 [@demishassabis](https://twitter.com/demishassabis/status/1961077016830083103)
- OpenAI Codex 功能引起开发者共鸣 [@sama](https://twitter.com/sama/status/1961096744533647501)

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. [Z.AI](http://z.ai/) GLM AMA + Mini MoE 路线图

- [**与 GLM 模型背后的实验室 [Z.AI](http://z.ai/) 的 [AMA](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/)**](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/) ([Score: 396, Comments: 314](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/))：**与 [Z.AI](http://z.ai/)（GLM 系列的创作者）的 AMA 集中在围绕 GLM-4.5 的技术问题上，特别是 GLM-4.5 Air 的训练后 SFT——请求具体的超参数（learning rate, batch size, epochs, dataset size, weight decay）、目标 loss 以及避免 catastrophic forgetting 的方法，评论者指出这些在 GLM 4.5 论文（[pdf](https://arxiv.org/pdf/2508.06471)）中没有详细说明。分享了一个 GLM-4.5 的社区微调版本供参考（[HF: GLM-Steam-106B-A12B-v1](https://huggingface.co/TheDrummer/GLM-Steam-106B-A12B-v1)）。其他问题探讨了权重开放模型（GLM-4.5, Kimi K2）与前沿封闭系统（GPT-5, Gemini, Claude）之间的区别，以及缩小差距需要什么，此外还包括 [Z.AI](http://z.ai/) 计划开发大于 32B 的 dense 模型，还是倾向于大型 MoE 架构。** 评论者推动透明度和可复现性（完整的 SFT 超参数和微调目标），并辩论权重开放工作是否能现实地匹配或超越封闭的前沿模型。人们对 scaling dense 模型（例如 ~70B+）与投资更大的 MoE 系统之间的架构权衡和路线图也表现出兴趣。
    - 一位评论者请求 **GLM-4.5 Air** 确切的 SFT 训练后配方——learning rate schedule、global batch size、epochs 数量、数据集大小/构成、weight decay 以及任何 adapter 策略——以及像 cross-entropy loss/perplexity 这样的实际目标和防止 *“catastrophic forgetting”* 的方法。他们引用了社区微调版本 [GLM-Steam-106B-A12B-v1](https://huggingface.co/TheDrummer/GLM-Steam-106B-A12B-v1)，并指出官方论文缺乏这些细节（[arXiv:2508.06471](https://arxiv.org/pdf/2508.06471)）。他们正在寻求关于微调 **GLM-4.5 Air** 的指导（例如，小的 LR、来自预训练语料库的混合回放、KL/L2 正则化或逐渐解冻），以避免在 SFT 期间性能下降。
    - 另一个帖子询问像 **GLM-4.5** 和 **Kimi K2** 这样的权重开放模型需要做些什么才能赶上封闭的前沿模型（GPT-5, Gemini, Claude）。重点在于训练算力、数据质量/规模、RLHF/RLAIF 和 tool-use 流水线、安全对齐以及 eval 驱动训练方面的潜在差距；他们探讨了改进的 scaling 策略、更好的数据清洗以及从前沿模型中进行 distillation 是否能缩小差距，以及实现对等是否可行。
    - 多个问题探讨了 [**Z.AI**](http://z.ai/) 的 scaling 路线图：是继续开发大于 `32B` 的 dense 模型，还是追随大型 **Mixture-of-Experts (MoE)** 的趋势。他们询问 SOTA 封闭模型的参数量是否可能比 GLM 更多，以及增加参数量对于 SOTA 级性能是否必要，含蓄地权衡了稀疏性的训练/推理成本、routing 质量和吞吐量优势，与 dense `70B` 级模型的稳定性/简单性。
- [**开启我们与 GLM 创作者 [Z.AI](http://z.ai/) 的新 AMA 系列（明天，太平洋标准时间上午 9 点至下午 12 点）**](https://i.redd.it/ek8o2pfzumlf1.jpeg) ([Score: 291, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n1unkv/launching_our_new_ama_series_with_zai_creators_of/))：**r/LocalLLaMA 正在举办一场与 [Z.AI](http://z.ai/)（GLM 系列背后的团队）的 AMA，定于 2025 年 8 月 28 日星期四，PDT 上午 9 点至下午 12 点。该帖子是一张宣布活动的图片传单；除了时间和主持人外，图片或标题中没有包含任何技术细节或议程项目（例如 GLM 变体、基准测试、本地部署细节）。** 评论大多是轻松/行政性质的（例如，注意到 AMA 和子版块命名的幽默），目前还没有实质性的技术讨论。
    - 排期确认：由于夏令时 (DST)，一个机器人将活动时间纠正为 PDT（而非 PST），并链接了时间转换：[https://timee.io/20250828T1600?tl=Launching%20Our%20New%20AMA%20Series%20With%20Z.AI%2C%20Creators%20of%20GLM%20(Tomorrow,%209AM-12PM%20PST)&d=180](https://timee.io/20250828T1600?tl=Launching%20Our%20New%20AMA%20Series%20With%20Z.AI%2C%20Creators%20of%20GLM%20(Tomorrow,%209AM-12PM%20PST)&d=180)。这将 AMA 映射到 PDT 上午 9 点至下午 12 点（UTC 16:00–19:00），持续时间为 `180` 分钟，减少了全球参与者的歧义。
    - 路线图兴趣：一位评论者询问 “glm 6 when?”，表明了对下一个 GLM 发布时间表细节的需求。虽然线程中没有讨论具体规格，但这指向了预期的 AMA 主题，如未来 GLM 迭代的版本节奏和功能升级。

- [**glm mini will be comming**](https://i.redd.it/h1ss59p4lslf1.jpeg) ([Score: 191, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1n2hyt2/glm_mini_will_be_comming/)): **在与 Z.ai/GLM 团队的 AMA 截图（Ask Me Anything）中，一名用户询问了关于更小规模的 Mixture-of-Experts (MoE) 模型（例如 OSS-20B 或 30B-A3B）的计划，一位联合主持人确认他们计划训练一个与 GPT-OSS-20B 相当的小型 MoE 模型。这暗示即将推出的 “GLM mini” MoE 变体将针对更低的激活参数量，以便于本地推理，同时保持强大的能力，类似于 Qwen 30B A3B 风格的配置。[图片链接](https://i.redd.it/h1ss59p4lslf1.jpeg)。** 评论者指出 Qwen 30B A3B 表现良好，但其较低的激活参数预算损害了长上下文推理能力；有人提议将 38B A6B 作为一个理想平衡点——每个 token 激活更多专家，但仍可在本地运行。其他人询问了 AMA 的来源/背景，发帖者表示这来自当前的 [Z.ai](http://z.ai/) 团队 AMA。
    - 讨论集中在 Mixture-of-Experts 设计上：一位用户指出 Qwen 30B A3B 表现不错，但其每个 token 较低的“激活参数”似乎损害了长文本推理，并提议推出 38B A6B 变体，以在保持本地可运行的同时提升激活容量。在 MoE 命名法中（例如 Qwen2 57B-A14B），“A#B” 表示每个 token 的近似激活参数量，因此从 `~3B` 增加到 `~6B` 激活参数可以实质性地提升能力，而无需像稠密（dense）30–40B 模型那样的全部计算量（[Qwen2 MoE 命名背景](https://qwenlm.github.io/blog/qwen2/)）。
    - 关于 “GLM mini” 即将推出的 AMA 提示引发了关于“与 gpt-oss-20B 相当”这一说法的歧义；评论者质疑这是指参数量还是实际质量。从历史上看，此类公告中的“相当”通常对应模型大小而非 Benchmark 上的对等，因为训练数据、计算预算和指令微调（instruction-tuning）会严重影响结果（GLM 系列参考：[ZhipuAI/GLM](https://github.com/THUDM/GLM)）。
    - 在易用性/本地推理方面，建议认为 A6B MoE 可以被广泛运行：MoE 仅针对每个 token 的专家子集增加激活计算量，从而在与更小的稠密模型相似的单步时间内实现更高的有效容量。注意：除非运行时支持专家分片/卸载（expert sharding/offload），否则 VRAM 占用仍可能由总参数量（所有专家）决定；vLLM 等引擎已开始针对实际部署优化 MoE 的加载和路由（routing）（[vLLM MoE 支持](https://blog.vllm.ai/2024-02-05-moe/)）。
- [**Again where behemoth and reasoning model from meta ??**](https://i.redd.it/xma7ru49krlf1.png) ([Score: 224, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1n2chrm/again_where_behemoth_and_reasoning_model_from_meta/)): **该图片是 Meta “Llama 4” 多模态 MoE 系列的宣传幻灯片，重点介绍了 “Llama 4 Behemoth”，这是一个拥有 16 个专家的 MoE，总参数量为** `2T`**，激活参数量为** `288B`**，定位为用于蒸馏（distillation）的“智能导师”；配套变体 “Maverick” 和 “Scout” 则针对速度/效率。发帖者的标题（“Meta 的巨兽和推理模型又在哪呢？？”）暗示这些大型/“推理”模型尚未公开发布；幻灯片强调的是蒸馏和效率，而非可用性。[图片](https://i.redd.it/xma7ru49krlf1.png)。** 评论者持怀疑态度，认为尽管 Behemoth 的规模大约是 **Qwen 3 235B** 的 6 倍，但其表现仍会逊色，称其为“发布即过时”，还有一些戏谑的说法称它正在指导 Meta 的战略。
    - 有推测认为 Meta 未发布的 “behemoth” 推理模型表现不如更小的开源模型，一条评论断言它 *“在 6 倍规模下可能比 **Qwen 3 235B** 还要差。”* 如果属实，这表明扩展效率（scaling efficiency）较差，即增加参数量（`>6×`）未能转化为比 `~235B` 基准更好的推理质量。
    - 另一个技术推论是，未发布本身就是一个负面的性能信号：如果该模型具有竞争力，Meta 早就发布了。这意味着内部评估可能未能超越当前的推理 SOTA，因此未发布暗示了其 Benchmark 结果平庸，且在现阶段实际价值有限。

### 2. 音频生成发布：HunyuanVideo-Foley 和 VibeVoice TTS

- [**HunyuanVideo-Foley 发布，一款开源的 text-video-to-audio 模型**](https://v.redd.it/jpjpqw2xuolf1) ([Score: 294, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1n22xbl/hunyuanvideofoley_is_out_an_open_source/))：**腾讯的 HunyuanVideo-Foley 是一款开源的、以视频为条件的（text–video→audio）模型，能够生成与输入视频对齐的拟音（foley）/配乐，并提供了公开 demo、权重和代码：[demo](https://hunyuan.tencent.com/video/zh?tabIndex=0), [Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley), [GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley), [项目主页](https://szczesnys.github.io/hunyuanvideo-foley/), 以及 [arXiv](https://arxiv.org/abs/2508.16930)。早期用户反馈指出，与之前的尝试相比，该模型改进了频率响应（更强的低音/高音）并实现了更好的 A/V synchronization，填补了当前视频生成流水线中缺失的音频阶段（例如，将 Hunyuan/Wan 用于视觉生成，TTS 用于对话）。该讨论帖澄清了它确实可以为现有视频轨道生成合适的音频（即具有可选文本调节的 video-to-audio）。** 评论者将其视为实现端到端自动化内容流水线的“最后一块拼图”，并讨论了多 GPU 编排（例如在 ComfyUI 等工具中持久加载模型）以处理长时间运行的批处理任务；热情主要集中在工作流集成而非原始基准测试上。
    - 多位用户澄清，这里的“text-video-to-audio”模型意味着生成与现有视频轨道对齐的 Foley/环境音效，有效地填补了缺失的音频层。这可以与 **Hunyuan** 和 **Wan** 等 text/image-to-video 模型以及 **Infinite Talk** 等对话模型一起嵌入到端到端流水线中，从而实现具有同步视觉和声音的全合成短片。
    - 用户对构建 `multi-GPU` 生产流水线表现出浓厚兴趣，其中每个模型（设计、T2V、对话、Foley）分别驻留在专用 GPU 上并向下游传递产物，从而最大限度地减少重新加载开销并提高吞吐量。一个关键的悬而未决的问题是，**Comfy** 目前是否提供稳健的多 GPU 图执行/调度，以支持持久驻留、模型间传输和长达周末的批处理队列。
    - 早期定性评价：据报道，与早期的尝试相比，音频质量具有更好的频率平衡（“中音、低音和高音”）和更紧密的 A/V sync。实际部署方面的考量包括模型尺寸“不太大”，要求以 `safetensors` 格式发布以便更轻松/安全地加载，以及关于具体运行指令的问题。
- [**发布：微软新 VibeVoice TTS 的 ComfyUI Wrapper（数秒内完成声音克隆）**](https://v.redd.it/yy7k60z8eplf1) ([Score: 228, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1n24utb/released_comfyui_wrapper_for_microsofts_new/))：**微软新推出的 VibeVoice TTS 的开源 ComfyUI wrapper 增加了 Single Speaker 节点、Multiple Speakers 节点（最多支持 `4` 个发言人——模型限制）以及基于文件的长文本合成输入，仓库地址为 [Enemyx-net/VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI)。官方权重报告的 VRAM 占用情况：`1.5B` 模型约为 `5 GB`，`7B` 模型约为 `17 GB`（后者仍处于 Preview 阶段），通过约 `56s` 的提示音频即可实现定性表现强劲的单发言人克隆；多发言人模式“仅在 7B 模型下表现尚可”，但成功率较低。该模型对 seed 高度敏感（不同 seed 之间的质量差异巨大），并表现出混合的跨语言行为：非 EN/zh 的提示音频（如意大利语）可以产生同语言输出，但 EN 提示音频无法可靠地产生其他语言。** 用户反馈指出它可以在 RTX 5090 上运行，并建议在提示语末尾添加标点符号或结尾省略号（“ ...”）以避免短语提前切断；其他人则请求/期待量化版本的发布以减少资源占用，并称赞了该节点的实用性。
    - 一位用户确认 Microsoft VibeVoice TTS 的 ComfyUI wrapper 在 **RTX 5090** 上运行流畅，并成功克隆了自己的声音，这表明它在高端 NVIDIA 显卡上具有良好的兼容性（未报告伪影或不稳定性）。虽然没有给出延迟数据，但该报告暗示对于个人语音使用，它可以达到实时或接近实时的响应速度。
    - 针对短提示语音频提前切断的实用解决方法：在输入末尾添加标点符号（"?", "!", "."）并附加一个结尾的 " ..."（例如 "Hello? ..."）。这似乎可以减轻可能导致单单词或极短 TTS 输出被截断的序列结束或静音修剪行为。
    - 用户对量化版本有需求，这将降低 VRAM 要求，并可能提高在较小 GPU/CPU 上的吞吐量。此类发布将扩大除高端显卡之外的部署能力，同时仅损失量化过程中典型的极小质量。

### 3. 本地 AI 工具：gpt-oss 60K 上下文训练与第二大脑

- [**Gpt-oss 微调 - 现支持 60K 上下文长度且显存占用 <13GB VRAM**](https://i.redd.it/rwu8gezzwslf1.jpeg) ([评分: 229, 评论: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/)): **该贴宣布了 Unsloth 为 OpenAI gpt-oss 训练推出的 Flex Attention，声称其上下文长度增加** `>8×`**，VRAM 占用减少** `>50%`**，且训练速度比包括 FlashAttention-3 在内的其他实现快** `>1.5×`**。这使得在** `80GB` **VRAM 上进行 BF16 LoRA 训练时可支持约** `60K` **token 的上下文（标题还标榜了可适配“<13GB VRAM”）。它增加了将 QLoRA 微调的 gpt-oss 模型导出到** `llama.cpp`**、** `vLLM`**、** `Ollama` **和 HF 的功能，修复了 T4/Colab 上的 float16 损失爆炸问题，并在 transformers 中为 MXFP4 推理强制执行** `swiglu_limit=7.0`**；节省效果随序列长度增加而提升。链接：[Unsloth](https://github.com/unslothai/unsloth)，博客/详情：[docs.unsloth.ai/basics/long-context-gpt-oss-training](https://docs.unsloth.ai/basics/long-context-gpt-oss-training)。** 评论询问了关于扩展到 120B 模型的问题，并对即将推出的支持直接保存 GGUF/llama.cpp 的 notebook 表现出浓厚兴趣；整体情绪非常热烈。
    - 开发者提到下周将推出一个支持直接保存为 **GGUF** 格式供 **llama.cpp** 使用的训练 notebook，这将消除转换步骤，并允许在 **llama.cpp** 后端（CPU, CUDA, ROCm 和 Apple Metal）立即进行推理。这也将简化与 **LM Studio** 等工具的集成，并使量化部署（如 Q4/Q5）在宣传的 `60k` 上下文和 `<13 GB` VRAM 目标下变得简单直接。链接：[llama.cpp](https://github.com/ggerganov/llama.cpp), [GGUF spec](https://github.com/ggerganov/llama.cpp/tree/master/gguf)。
    - 用户对更大的 `120B` 变体有需求。实际上，`120B` 模型的本地推理通常会超过单设备限制，往往需要多 GPU 的 tensor parallelism 和激进的 quantization；即使是 4-bit 量化也可能需要 `~40–60 GB` VRAM，除非使用分布式设置，否则远超 `<13 GB` 级别。
    - 多位用户询问了 macOS 支持情况：在 Mac mini M4 的 **LM Studio** 上运行，以及 **Unsloth** 是否会登陆 Mac。如果模型能直接导出为 **GGUF**，它就可以立即通过带有 Metal 后端的 **llama.cpp**（LM Studio 的底层封装）使用，从而在无需专门移植的情况下提高 Mac 兼容性。链接：[LM Studio](https://lmstudio.ai/), [Unsloth](https://github.com/unslothai/unsloth)。
- [**我构建了一个真正能记住一切的本地“第二大脑” AI（已通过 321 项测试）**](https://www.reddit.com/r/LocalLLaMA/comments/1n2djpx/i_built_a_local_second_brain_ai_that_actually/) ([评分: 259, 评论: 120](https://www.reddit.com/r/LocalLLaMA/comments/1n2djpx/i_built_a_local_second_brain_ai_that_actually/)): **OP 介绍了 Kai，一个本地“认知操作系统”，它使用基于图的知识库和 spreading activation 检索技术（类似于 ACT-R 等认知架构）构建持久的设备端记忆。它 100% 本地运行（无云端），从用户在机器上的活动中学习，并强调它“不仅仅是 RAG”——而是利用节点/边记忆图 + 激活权重动态；该项目报告已通过** `321` **项测试，并在 [oneeko.ai](http://oneeko.ai/) 提供早期访问，附带 3D 记忆可视化[截图](https://preview.redd.it/8jei7138zrlf1.png?width=1920&format=png&auto=webp&s=b4125be85bd9a5a616c10a0423130cba14169100)。OP 计划在稳定后开源核心引擎。** 热门评论鉴于其“仅限本地”的声明敦促其开源，并分享了一个使用查询驱动激活和残差增强方法的类似项目（[dsam_model_memory](https://github.com/jwest33/dsam_model_memory)），而一位怀疑者认为它可能只是一个对对话数据进行标记/总结的 MCP 风格服务器——并带有此类系统常见的失效模式。
    - 一位正在构建类似系统的评论者分享说，他们使用基于查询的激活函数来生成残差，从而增强频繁访问的记忆和相关概念（仓库：https://github.com/jwest33/dsam_model_memory）。他们认为这能让检索随着时间的推移偏向高显著性项，而不是静态的 vector-store 召回，从而提高个人知识库中的长期相关性。
    - 另一位评论者怀疑该项目本质上是一个 **MCP** 服务器，对对话数据进行标记并构建摘要图，具有“保存”和“查询”接口（参见 https://modelcontextprotocol.io/）。他们警告说，这种架构通常会继承在类似的标记/摘要图流水线中看到的失效模式，暗示当元数据随时间推移与用户意图发生偏离时，会出现持久性问题。

- 硬件性能观察：在他们的配置下，**qwen3 235b a22b** 的运行速度约为 `~20 tps`，**glm-4.5-air** 约为 `~40 tps`，而 **gpt-oss-120b** 约为 `~70 tps`，而他们更倾向于 `>=100 tps`。他们还指出，许多模型在个人助手工作流中感觉“被过度审查”，因此更希望减少安全干预，以实现开放式探索。

## 技术性较低的 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5 医学基准测试与 Codex IDE/CLI 发布

- [**GPT-5 在美国执业医师资格考试中表现优于医生**](https://i.redd.it/rjs8oqzb1qlf1.png) ([评分: 666, 评论: 245](https://www.reddit.com/r/OpenAI/comments/1n26rqz/gpt5_outperformed_doctors_on_the_us_medical/)): **一篇名为《GPT-5 在多模态医学推理上的能力》（“Capabilities of GPT-5 on Multimodal Medical Reasoning”，[AlphaXiv](https://www.alphaxiv.org/pdf/2508.08224)）的预印本声称，GPT‑5 在 USMLE 风格的评估中表现优于执业医师约** `25–30%`**，如推文中的表格所示。该结果似乎依赖于结构化的、专家策划的输入（即近乎完美的诊断/上下文数据），并非端到端临床工作流；它评估的是对类考试案例的推理/答案选择，而非自主的患者管理。** 热门评论指出，性能表现取决于是否获得了完美的诊断数据，这使得该设置类似于开卷考试，并警告说，尽管考试表现强劲，但真正的临床安全（药物相互作用、纵向上下文）仍是一个未解决的挑战。
    - 几位评论者指出，该基准测试可能假设了理想化的输入——例如，结果取决于“由人类专家提供完美的诊断数据”。这种设置评估的是在干净、专家策划的背景下的答案选择，而不是处理带有噪声、不完整病史的端到端临床推理，而后者是与必须进行分诊、询问病史和消除歧义的临床医生进行比较时的主要干扰因素。
    - 一个技术相关的安全担忧是状态/召回限制：LLM 可能会因为上下文截断而“忘记”早期的病历详情，从而冒着提出禁忌建议的风险（例如，在之前使用过布洛芬后又建议使用另一种 NSAID，如双氯芬酸）。这突显了对稳健的患者状态追踪、用药衔接和自动药物相互作用检查作为护栏的需求，而不是仅仅依赖瞬时的对话上下文。
    - 许多评论将其定义为“开卷优势”：该模型通过预训练实际上携带了教科书语料库，因此在多选题考试中表现出色主要反映了在庞大先验知识下的应试/召回能力。这一指标并不等同于临床表现；它不同于其他已验证的 AI 优势（如特定的影像任务），并且与参加闭卷、限时 USMLE 的人类相比，引发了公平性问题。
- [**Codex 现在可以在你的 IDE、云端和 CLI 中配合 GPT-5 运行**](https://i.redd.it/z91dfrq44nlf1.png) ([评分: 221, 评论: 80](https://www.reddit.com/r/ChatGPTCoding/comments/1n1vjvi/codex_now_runs_in_your_ide_cloud_and_cli_with_gpt5/)): **OpenAI Developers 的一份公告（2025 年 8 月 27 日）声称，Codex 现在可以作为跨 IDE、云端和 CLI 的编程协作伙伴，由“GPT-5 提供支持”，并可通过 ChatGPT 计划访问。图表重点展示了新的 IDE 扩展、本地与云端环境之间的无缝任务移交、GitHub 代码审查集成以及翻新后的 Codex CLI——这表明从编辑到审查再到执行的端到端工作流覆盖更加紧密。** 评论者询问了与 **Claude Code** 的实际对比（质量/易用性）、之前的沙箱化要求是否仍然适用（对某些人来说是障碍），以及是否支持 **RStudio**/R 工作流。
    - 用户指出，之前 Codex 要求在严格的沙箱中运行代码是实际工作流（文件系统、网络、包管理器、测试运行器）的主要障碍，并询问新的 IDE/Cloud/CLI 发布是否放宽了限制或允许按项目选择退出。沙箱问题的解决（例如：受信任目录、网络出口、环境变量访问）将决定它对于 IDE 内重构和调试是否可行，而不仅仅是安全的临时运行。
    - 一位使用 `Claude Code $100` 计划的高级用户报告称，虽然更喜欢 GPT‑5 的原始代码生成质量，但仍觉得 Claude Code 的整体“系统”更难被超越。结论是，单靠模型质量是不够的；在 `~$100/月` 这一档位，可靠性和端到端开发者易用性（工作流编排、上下文处理、集成）对于采用率起着决定性作用。
    - 关于访问层级存在不确定性：`GPT‑5 High` 是否包含在 20 美元的 ChatGPT Plus 计划的 Codex 中。一位评论者发现“中等思考（medium thinking）”效果一般，暗示“Medium”和“High”层级之间存在显著的质量差距，这可能会影响延迟/成本权衡和计划选择。

- [**Who’s Your Doctor Now?**](https://i.redd.it/bx5xh1p4mrlf1.jpeg) ([Score: 2733, Comments: 87](https://www.reddit.com/r/ChatGPT/comments/1n2cqng/whos_your_doctor_now/)): **这是一个非技术性的迷因（meme），对比了 AI 助手与网页搜索在“床边态度”（bedside manner）上的差异：在 OpenAI 标志下写着“没什么大碍，可以治疗”，而 Google 则是“你还剩 3 分钟寿命”，暗示了 LLM 的安抚作用与搜索引擎引发的恐慌。标题“Who’s Your Doctor Now?”将其框定为对自我诊断文化的调侃；未讨论 Benchmark、模型或实现细节。** 评论区回忆了“Dr. Google”时代加剧疑病症的往事，并对过度诊断进行了嘲讽，还有一些关于专业精神和将一切归结为癌症的讽刺俏皮话。
- [**Rate this art by gpt 5**](https://i.redd.it/nf5kr1bjiolf1.jpeg) ([Score: 244, Comments: 189](https://www.reddit.com/r/ChatGPT/comments/1n21mfi/rate_this_art_by_gpt_5/)): **象头神（Lord Ganesha）的 AI 生成抽象画；尽管标题写着“由 GPT-5 生成”，但分享的 Prompt 清楚地指向了 Midjourney v6.1：“thick paint splashes … white background --personalize cvlos9g --stylize 800 --v 6.1”。高** `-stylize 800` **参数驱动了大胆、极简的油漆笔触美学，而** `-personalize cvlos9g` **则暗示了一个特定于用户/风格的个性化 Token，从而产生了带有生动液体油漆笔触的干净白色背景。图片：https://i.redd.it/nf5kr1bjiolf1.jpeg** 评论指出其与奥运会标志相似，并对 AI 艺术的价值持两极分化的看法；一位评论者提供了准确的 Prompt 以供他人复现结果，含蓄地纠正了将归属权划给 GPT-5 的错误，指出其应为 Midjourney。
    - 一位评论者分享了准确的 Prompt 和参数：“thick paint splashes forming abstract minimalist shape of Lord Ganesha, … white background --personalize cvlos9g --stylize 800 --v 6.1”。这意味着使用了 **Midjourney v6.1** (`-v 6.1`)，且高 `-stylize 800` 值使输出强烈偏向美学而非字面上的 Prompt 遵循（参见 Midjourney 参数文档：https://docs.midjourney.com/docs/parameters）。`-personalize cvlos9g` Token 似乎是一个自定义风格/配置文件标识符，影响了调色板和构图。
    - “看起来不像 AI 生成的”这类观察结果与 MJ v6.x 改进的连贯性和纹理处理能力相符，它可以产生干净、类似 Logo 的几何图形和一致的“液体油漆”效果。极简主义构图加上白色背景和高风格化倾向于抑制常见的 AI 痕迹（如杂乱的边缘、不一致的画笔物理特性），从而产生一些观众认为非 AI 的结果；参见模型/版本说明：https://docs.midjourney.com/docs/models#version-6。
- [**Chicken of the Sea - SaraShakeel x Ai render**](https://v.redd.it/ujigpiulfplf1) ([Score: 349, Comments: 10](https://www.reddit.com/r/aivideo/comments/1n2524a/chicken_of_the_sea_sarashakeel_x_ai_render/)): **用户分享了一个名为“Chicken of the Sea — SaraShakeel x AI render”的 AI 生成视觉作品，显然是模仿艺术家 Sara Shakeel 的风格，托管在 Reddit Video：[v.redd.it/ujigpiulfplf1](https://v.redd.it/ujigpiulfplf1)。外部链接目前返回** `HTTP 403 Forbidden`**，意味着访问需要 Reddit 登录或开发者 Token（可能是 WAF/身份验证拦截），且该帖子未提供技术元数据（如模型、Prompt 或 Pipeline 细节）。未讨论 Benchmark、实现说明或资产工作流；该帖子主要关注审美评价。**
    - 一位评论者详细介绍了一个 AI 出现之前的 Pipeline：从在 Photoshop 中合成的修饰参考图开始，然后使用 **Midjourney** 扩展外观，随后在 **Cinema 4D** 中使用 **Arnold** 渲染器进行动画制作、粒子模拟，并在 **After Effects**/**Mocha** 中进行合成/跟踪（[Midjourney](https://www.midjourney.com/)、[C4D](https://www.maxon.net/cinema-4d)、[Arnold](https://www.arnoldrenderer.com/)、[After Effects](https://www.adobe.com/products/aftereffects.html)、[Mocha](https://borisfx.com/products/mocha/)）。他们报告称，一个 2 分钟的交付成果需要约 4 周的工作时间，其中包括约 1 周的渲染时间，报酬约为 5000 美元（非连续时间），并指出与当前的 AI 渲染相比，当时的真实感落后了，且他们的定价本应接近 1 万美元。他们总结道，“AI 正在使市场贬值”，反映出随着生成式工具提高速度/真实感，费率面临下行压力。

### 2. WAN 2.x Infinite Talk Demos & S2V Tips + HunyuanVideo-Foley

- [**4090 48G InfiniteTalk I2V 720P 测试~2分钟**](https://v.redd.it/uxe60qpinnlf1) ([评分: 501, 评论: 117](https://www.reddit.com/r/StableDiffusion/comments/1n1ycs9/4090_48g_infinitetalk_i2v_720p_test2min/)): **创作者在 RTX 4090 (48 GB) 上使用** `wan2.1_i2v_720p_14B_fp8_scaled` **配合 LoRA** `lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16` **对 I2V 工作流进行了基准测试，在消耗约** `36 GB` **显存的情况下，以 4 步扩散步数生成了 1280×720 的输出。该运行处理了** `49` **个分块，每个分块** `81` **帧（标题：总计约 2 分钟），每个分块耗时约** `5 分钟`**，总计约** `245 分钟`**；FP8 量化的 14B 模型加上步进蒸馏 LoRA（rank** `256`**, bf16）表明这是一个针对速度/显存优化的配置。音源是 YouTube 上的 AI 翻唱 (https://youtu.be/9ptZiAoSoBM)，模仿 [岩崎宏美 (Hiromi Iwasaki)](https://en.wikipedia.org/wiki/Hiromi_Iwasaki) 的风格。** 评论者反映唇形/语音同步整体表现强劲，但在背景和声部分会有所下降，推测麦克风的移动/操作可能会干扰模型，并预测近期会出现能从上传的歌曲中自动编辑并发布音乐视频的 Agent 工作流。
    - 观察者指出，语音-唇形同步在 `720p` 下基本准确，但在重叠/背景人声期间会变差；使用纯净的人声干声可能会改善对齐效果。有人推测不稳定的麦克风移动可能会干扰声源检测/语音活动线索，导致模型瞬间追踪到错误的歌手。
    - 针对用于约 `2 分钟` I2V 运行的非标准 `RTX 4090 48GB` 配置，人们表现出了浓厚兴趣，并询问具体的供应商/改装来源。评论者指出，这种非典型的显存容量会影响其他尝试该配置的人的可复现性以及潜在的 Batch/Window 大小。
    - 关于多 GPU 能力（例如，跨 GPU 拆分推理/训练）的问题表明，用户想知道 InfiniteTalk I2V 工作流是否支持数据/模型并行或显存分片 (VRAM sharding)。明确 `48GB` 的需求是通过多 GPU 聚合还是单张大显存显卡来满足，将有助于硬件选择。
- [**WAN S2V 生成效果不佳的三个原因及避免方法**](https://v.redd.it/hxa93nfu9slf1) ([评分: 510, 评论: 151](https://www.reddit.com/r/StableDiffusion/comments/1n2gary/three_reasons_why_your_wan_s2v_generations_might/)): **楼主报告称，WAN S2V 通过 WanVideoWrapper 获得的开箱即用效果明显优于原生 ComfyUI 工作流，后者需要大量调整且质量仅为中等。他们建议避免使用“加速 LoRA”，称其会降低 WAN 2.2 和 S2V 的输出质量、动作表现和提示词遵循度（仅对基本静态的 Talking Heads 可接受）。强调了强大的提示词工程：指定音乐类型、氛围、情感状态、注视方向、头部/身体动作以及具体的动作，而非模糊的提示词。示例运行参数：** `576x800` **分辨率，** `~737 帧`**，采样器** `UniPC/beta`**，** `23` **步。链接媒体受访问限制 ([v.redd.it](http://v.redd.it/) [403](https://v.redd.it/hxa93nfu9slf1))；另请参阅 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)。** 热门评论包括分享工作流的请求（用户 “Limewire”）以及对结果的普遍赞赏；未提出实质性的技术反对意见。
    - **comfyanonymous** 指出 S2V 的“原生工作流”尚未正式发布，且节点仍标记为 `beta`，这意味着目前的质量问题源于实现尚不成熟；一旦原生节点完全实现，其表现应优于临时/第三方工作流。这表明用户应期待快速迭代，并且在原生节点稳定之前可能会有破坏性变更。
- [**Wan 2.1 Infinite Talk (I2V) - FOAR EVERYWUN BOXXY**](https://v.redd.it/qx1b1h6z8rlf1) ([评分: 217, 评论: 45](https://www.reddit.com/r/StableDiffusion/comments/1n2b4gi/wan_21_infinite_talk_i2v_foar_everywun_boxxy/)): **楼主展示了一个使用 Wan 2.1 “Infinite Talk” 的图生视频 (I2V) 工作流，生成了一个带有刻意的上身/手部动作的 Talking Head 剪辑。正向提示词针对面部化妆（大睫毛/眼线）和涂有黑色颜料的短指甲，而详尽的反向提示词则抑制了常见的视频生成伪影（例如：长指甲/珠宝、过度曝光/模糊/静态帧、JPEG 伪影、额外/融合的手指、畸形的肢体、杂乱的背景、多个/额外的肢体、向后行走），旨在获得更干净的手部渲染和更动态的动作。未提供生成参数（分辨率/FPS/步数/采样器/种子/CFG/时长）或硬件细节（如显存）。** 评论者称赞了其质量——其中一人称其为 Infinite Talk “迄今为止最好的例子”——而另一人询问了显存需求，表明了对计算开销的关注；帖内未给出回答。

- 资源需求：一位评论者询问“达到这种效果需要多少 VRAM”，寻求复现所示 Infinite Talk I2V 质量的具体 GPU 显存需求。技术读者期望了解特定分辨率/时长下的 VRAM 占用详情（例如 `512p/1024p`，每帧秒数）、模型精度（`fp16` vs `bf16`），以及推理是否使用了 xformers/attention slicing 或 CPU offload 以适配消费级 GPU。
- 身份保真度与参考控制：有人指出输出“看起来不像 Boxxy”并索要原图，这含蓄地探测了该流水线的身份保持能力和条件引导强度。这引发了关于参考处理（单图 vs 多图，人脸对齐/地标引导，ID loss，以及 GFPGAN/CodeFormer 等人脸增强器的使用）的疑问，以及 I2V 模型是否支持 guidance scale 或 identity embeddings 以保持跨帧的相似度稳定。
- 性能对比：另一位询问这是否“比新的 S2V 模型更好”，表明了对 Wan 2.1 Infinite Talk (I2V) 与 S2V 之间端到端质量和稳定性对比的兴趣。相关的基准测试将包括运动连贯性、对口型准确度、时间一致性（闪烁/扭曲）、推理速度 (FPS) 以及在相同提示词和分辨率下的 VRAM 效率。
- [**HunyuanVideo-Foley 发布了！**](https://v.redd.it/zazjjguqoplf1) ([得分: 289, 评论: 46](https://www.reddit.com/r/StableDiffusion/comments/1n25nqj/hunyuanvideofoley_got_released/)): **HunyuanVideo-Foley 是一个开源的 Text+Video→Audio (foley) 模型，可以从视频输入（可选文本条件）生成同步音效。包含交互式演示以及与 MMAudio 和 ThinkSound 等基准模型侧向对比的项目页面可见：https://szczesnys.github.io/hunyuanvideo-foley/。** 早期用户反馈报告质量参差不齐：动漫内容可能会崩溃为低能量的呼吸声/喃喃自语，一些现实生活片段会产生刺耳的“砂纸”质感；MMAudio 基准模型被指出有时会发出随机的大声“尖叫”，突显了伪影/幻觉问题。一位评论者还暗示生成过程中存在沉重的 I/O/计算需求（“我的 SSD 累了……”）。
    - 多位用户报告了严重的现实感问题和伪影：在动作开始时，输出会退化为“变异驱魔人尖叫”、难以理解的喃喃自语或宽带“砂纸”噪声。这指向了弱视听对齐和糟糕的瞬态处理——可能是扩散伪影和不稳定的条件引导导致了时间漂移和频谱粗糙度，从而导致拟音 (Foley) 效果低劣，缺乏精确的起音、动态和空间线索。
    - 不同风格间明显的领域差距 (Domain gap)：动漫序列仅产生微弱的叹息声，随后是模糊的语音，而实拍视频则产生刺耳的质感。这表明该模型对风格化视觉领域（动漫）的鲁棒性不足，并默认使用通用的、低信息的声学先验，意味着领域条件训练不足或针对非写实输入的风格 token/embeddings 不充分。
    - NSFW 提示词表现尤为糟糕（通用/摩擦质感，被抑制或不匹配的情色音效 (SFX)），暗示了此类内容中的安全过滤或数据稀疏。这种行为类似于在受限语义下向中性纹理和低方差输出的硬截断 (Hard clamping)，这在这些场景中进一步降低了对齐度和音色特异性。
- [**如果这是 Genie 3，想象一下 Genie 4 会有多疯狂**](https://v.redd.it/6rk25azwirlf1) ([得分: 1209, 评论: 179](https://www.reddit.com/r/singularity/comments/1n2cbyj/if_this_is_genie_3_imagine_how_insane_genie_4/)): **讨论集中在 `~8–9 个月` 内从 “Genie 2” 到 “Genie 3” 的能力快速飞跃，正如分享的演示视频所示（需要验证：https://v.redd.it/6rk25azwirlf1）。未引用基准测试或发布说明；讨论主要围绕发展轨迹——迈向更高的物理保真度和可交互、可导航的环境——而非具体的实现细节。** 评论者推测 “Genie 4” 可能会增加细粒度、物理一致的场景效果（例如“沙滩上的脚印”），并支持对生成空间的实时 VR 探索；一些人推断，如果这种指数级节奏保持不变，下一次迭代可能很快就会到来。
    - 发布节奏推测：一位评论者指出 Genie 2 → Genie 3 在 `~8–9 个月` 内完成，将其视为加速迭代的信号，并预期如果趋势是指数级的，Genie 4 将在短期内出现。另一位则将其与长期的 “GPT-4 → GPT-5” 炒作周期相类比，含蓄地提醒在没有具体基准测试或演示进行跨版本对比的情况下，节奏并不等于能力。

- 交互模型/UX 问题：一位用户询问 Genie 在实践中究竟是如何工作的——是需要不断的提示，还是支持有状态的连续会话。另一位用户推测其具有实时 VR 风格的探索功能，这意味着系统可以维护场景状态并接受连续的控制输入（例如摄像头/控制器），实现低延迟的流式生成，而不是离散的提示词转视频片段。
- [**Photoshop 完蛋了，Nano Bananas 的操作太疯狂了。**](https://www.reddit.com/gallery/1n2fxjn) ([Score: 2064, Comments: 225](https://www.reddit.com/r/singularity/comments/1n2fxjn/photoshop_is_cooked_nano_bananas_manipulation_is/))：**该帖子展示了一个尖端的 AI 图像编辑/Inpainting 工作流，演示了高保真的物体和场景操作——与早期 [Stable Diffusion](https://stability.ai/) 的“多余肢体”和“缺少手指”等失败模式形成鲜明对比。链接的图集 ([reddit.com/gallery/1n2fxjn](https://www.reddit.com/gallery/1n2fxjn)) 表明在大范围编辑中具有强大的结构一致性和纹理融合能力，但仍存在残留的失败案例（例如“多余手指”伪影），且对微观特征的细粒度控制仍然滞后，尤其是在面部。** 评论指出，虽然结果令人印象深刻，但 Photoshop 尚未“完蛋”：用户仍然能发现解剖学伪影，并反映精确的面部编辑非常困难，这暗示 AI 工具在全局/语义编辑方面表现出色，但在保持身份特征的小幅调整方面仍不可靠。
    - 进步体现在从早期 Stable Diffusion 的伪影（多余肢体/手指）到目前近乎照片级的操作，但手部/手指的忠实度仍然是一个极端情况——用户在最终图像中仍能发现多余手指等问题。这反映了尽管全局一致性和现实感有了重大改进，但 Diffusion/Inpainting 在高频解剖细节（手部）上仍存在持续性的弱点。
    - 用户报告称，在广泛编辑（例如构图/物体更改）方面表现强劲，但对细微面部细节的控制较差；*“尝试编辑细微的面部细节，简直是地狱。”* 这与已知的局限性相符，即局部、细粒度的编辑可能会降低身份辨识度或引入伪影，表明需要更好的掩码感知调节（Mask-aware conditioning）、ControlNet 或更高分辨率的潜空间编辑（Latent editing）来保留微观特征。
- [**Apple AI vs Galaxy AI vs Xiaomi AI 移除工具对比**](https://v.redd.it/w7ckphp7oplf1) ([Score: 4507, Comments: 407](https://www.reddit.com/r/ChatGPT/comments/1n26571/apple_ai_vs_galaxy_al_vs_xiaomi_al_remove_tool/))：**一段短视频（在 [v.redd.it](http://v.redd.it/) 上因 `403 Forbidden` 被屏蔽）据称对比了来自 Apple、Samsung Galaxy 和 Xiaomi 的消费级“移除/魔术橡皮擦” Inpainting 工具在同一张图片上的表现，突出了擦除主体后填充质量的差异。帖子中未提供实现细节、模型或基准测试；它看起来像是移动端照片编辑器常见的物体移除输出的视觉 A/B/C 对比。链接：https://v.redd.it/w7ckphp7oplf1** 热门评论认为 **Apple** 的结果明显逊于竞争对手，并将其比作基础的 Paint 3D 风格“魔术橡皮擦”，除此之外几乎没有实质性的技术讨论。
    - 几位评论者暗示各厂商的“移除”工具在功能上大同小异：**Apple Intelligence** 下 Apple 新的照片“Clean Up” vs **Google** 的 Magic Eraser、**Samsung Galaxy AI** 的对象/生成式编辑、**Xiaomi** 的 AI 擦除，甚至是 **Microsoft Paint** 的魔术橡皮擦——大多数都是语义分割 + 生成式 Inpainting 的变体。关键区别在于部署和隐私：Apple 强调在支持的设备上使用 `A17 Pro`/`M-series` NPU 进行端侧推理（[Apple Intelligence](https://www.apple.com/apple-intelligence/)），而 Samsung 通常会标注云端支持的编辑（[Galaxy AI](https://www.samsung.com/global/galaxy/ai/)）；Xiaomi 的实现则因型号/地区而异。质量往往取决于掩码准确度、Inpainting 模型（Diffusion vs 基于补丁）、背景纹理复杂性以及是无提示词填充还是引导式填充。
    - 值得注意的是 **Google Pixel** 的缺失，其 Magic Eraser 随 Pixel 6/Tensor 首次亮相，随后通过 Google Photos/One 扩展，某些功能在服务器端处理（例如 Magic Editor），其他功能则根据硬件/应用版本在端侧处理（[Google Photos Magic Eraser](https://support.google.com/photos/answer/11910009), [Magic Editor](https://blog.google/products/photos/magic-editor/)）。Pixel 的技术栈还包括 Best Take 和音频魔术橡皮擦，表明其拥有利用 Tensor ISP/NPU 的成熟且垂直整合的流水线；在实际对比中，物体移除质量通常与 Apple/Samsung/Xiaomi 处于同一梯队，但在 Diffusion Inpainting 擅长的精细纹理和边缘连续性上可能有所不同。

- [**2025 年将手绘图转化为照片**](https://v.redd.it/qtckbsr0jnlf1) ([Score: 447, Comments: 38](https://www.reddit.com/r/ChatGPT/comments/1n1xghk/turning_drawings_into_photos_in_2025/))：**一个演示帖子展示了一个能将手绘图/草图转换为写实照片的工具。嵌入的媒体链接 [v.redd.it/qtckbsr0jnlf1](https://v.redd.it/qtckbsr0jnlf1) 返回** `403 Forbidden` **（Reddit 屏蔽页），因此无法从帖子中验证模型细节、基准测试或具体的实现细节；讨论帖中未提供技术规格。** 热门评论指出了使用门槛：所谓的“免费”试用需要信用卡，并询问为什么不直接使用 ChatGPT——这暗示用户可能更倾向于内置工具或开源替代方案。对于精确的草图到照片控制，评论者通常推荐使用扩散模型 img2img 工作流（例如 SDXL + [ControlNet](https://github.com/lllyasviel/ControlNet)），而非通用聊天模型。
- [**这个帖子在 3 年前获得了 2.7 万个赞——在 Reddit 讨厌 AI 之前**](https://i.redd.it/tfu54shi6olf1.jpeg) ([Score: 716, Comments: 186](https://www.reddit.com/r/ChatGPT/comments/1n209ey/this_post_got_27k_upvotes_3_years_ago_before/))：**该图像 (https://i.redd.it/tfu54shi6olf1.jpeg) 是早期文本生成图像 AI 美学（约 2021–2022 年，Stable Diffusion 出现之前）的代表性例子：超现实、低连贯性的构图，带有 CLIP 引导的梦幻般伪影，而非写实主义。标题强调它在“3 年前”获得了** `27K` **个赞，突显了当时仅仅获得可辨认的输出就已值得关注；与现代扩散系统相比，这些旧的流水线产生的是绘画般的纹理、扭曲的结构和模糊的形态，许多人将其与该领域早期的魅力联系在一起。** 评论者指出，早期的 AI 艺术并未被视为对人类艺术家的威胁，且具有一种现在某些人怀念的独特抽象感；有人建议重新审视旧模型以重现那种氛围，而另一个人则评论说，按今天的标准来看，当时的质量显得多么“糟糕”。
    - 几位评论者将 2021–2022 时代的文本生成图像系统（当时生成任何“可辨认”的东西都感觉意义重大）与今天近乎写实的输出进行了对比。早期的流水线（如 VQGAN+CLIP, DALL·E mini/Craiyon, 早期 Stable Diffusion 1.x）由于 CLIP 驱动的引导和较弱的先验，往往产生抽象/梦幻般的结果；到 2023–2025 年，更大的扩散模型（如 SDXL, Midjourney v6）通过更大的 Backbone、更好的数据集以及改进的采样/微调，显著提高了分辨率、构图可靠性和提示词遵循度。参见 SDXL 概述：https://stability.ai/news/stable-diffusion-sdxl-1-announcement 以及 MJ v6 说明：https://docs.midjourney.com/docs/model-versions#version-6
    - 开发者对刻意使用旧模型来重现这种“怪异”美学产生了技术兴趣：伪影产生于低训练分辨率 (`256–512px`)、较小的 U-Nets、有限/噪声较多的数据集，以及产生过度饱和纹理和超现实构图的强 Classifier-free guidance。像早期 DDIM/PLMS 这样的采样器和 CLIP 引导的损失函数加剧了奇特的几何结构和文本融合，产生了一种“透过扭曲玻璃看互联网”的氛围，而这种氛围在拥有先进采样器（如 DPM-Solver++）和强大 Conditioning 的现代正则化良好的模型中很难获得。
    - 通过一个 3 年前的示例图像 (https://www.reddit.com/r/PeterFHamilton/s/9a3H1j4tQZ) 与“2025 年相同提示词”的渲染图 (https://preview.redd.it/q7aslq30kolf1.jpeg?width=1024&format=pjpg&auto=webp&s=4b13a50c42da2c8b531c7b7685e3610e67000af4) 的对比，体现了一个并行的主题。后者意味着在保真度（解剖结构、光影、纹理细节）、文本和提示词遵循以及伪影抑制方面取得了重大进展——这可能归功于更大的训练语料库、更高的原生分辨率、改进的 Conditioning（提示词/负向提示词）以及更好的推理工具（Refiners, Upscalers）。

### 3. AI 政策：ChatGPT 扫描、监管梗和就业辩论

- [**OpenAI 表示正在扫描用户的 ChatGPT 对话并向警方报告内容**](https://futurism.com/openai-scanning-conversations-police) ([Score: 697, Comments: 259](https://www.reddit.com/r/OpenAI/comments/1n2138e/openai_says_its_scanning_users_chatgpt/))：**该帖子声称 OpenAI “扫描” ChatGPT 对话并向警方报告用户。OpenAI 自身的 [Privacy Policy](https://openai.com/policies/privacy-policy) 和 [Usage Policies](https://openai.com/policies/usage-policies) 确认，对话可能会由自动化系统和授权人员进行审查，以确保滥用/安全，并且在法律要求或为了防止伤害时，内容可以向执法部门披露；虽然存在数据使用控制（例如：退出训练选项、企业保留设置），但常规的审核/滥用检测广泛适用，且几乎没有公开记录的阈值或审计细节。** 评论者将此与治理和政府关系联系起来：指出前 NSA 局长 **Paul M. Nakasone**（`2018–2024`，由特朗普提名）加入了 OpenAI 董事会（[OpenAI](https://openai.com/blog/welcoming-paul-nakasone-to-openai-board), [Wikipedia](https://en.wikipedia.org/wiki/Paul_M._Nakasone)），指称一项未经证实的 `$200M` DoD 合同，并敦促更清晰地披露员工访问权限和隐私边界。
    - 几位评论者引用了 OpenAI 声明的政策，即潜在的暴力意图会触发带有辅助人工审查的“专门流水线”，并且如果确定存在“严重的身体伤害迫在眉睫的威胁”，可能会移交给执法部门。这描述了一种两步审核架构：自动化检测 → 人工升级 → 执行/报告，符合行业信任与安全规范；上下文请参阅 OpenAI 的政策页面（例如：https://openai.com/policies/usage-policies）。
    - 存在关于内部人员访问的技术隐私担忧：用户希望明确披露数据流（收集、保留窗口、训练用途）、访问控制（谁可以查看对话以及在何种批准下）和审计（记录审查员的访问、PII 的脱敏）。评论者指出，大多数主要平台都进行大规模的滥用检测扫描，但要求 OpenAI 澄清消费者与企业默认设置、退出机制，以及如何减轻“窥探”风险（例如：分段、静态加密、最小权限原则）。
    - 治理/隶属关系的含义被提及：据报道，一名前 NSA 局长在 OpenAI 董事会任职（见 OpenAI 的公告：https://openai.com/blog/paul-nakasone-joins-openai-board-of-directors），且声称的 `200M` DoD 合同暗示了更深层次的政府整合。从技术上讲，这可能会影响报告工作流、监管对齐以及执法合作的阈值设定，尽管评论者对于这是否与其它大型科技公司的标准做法有实质性区别存在争论。
- [**如果 AGI 如此“不可避免”，他们就不应该关心任何监管**](https://i.redd.it/hn7w2vb02qlf1.png) ([Score: 342, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1n26s3h/if_agi_is_so_inevitable_they_shouldnt_care_about/))：**该图片是一个迷因（meme），突显了 AI 政策中的修辞张力：公司声称 AGI 在全球范围内是“不可避免的”，同时又警告国内监管可能会“扼杀该行业”。评论者区分了范围：“不可避免”是指全球范围内的进展，而仅限美国的严厉规则可能会将能力开发转移到国外（例如：中国）。其他人认为，目前的许多立法要么过于天真，要么被武器化以获取竞争优势，建议监管应针对下游的人类影响（安全、劳动保护），而不是禁止核心 AI 研发——并以此类比工业革命。** 辩论集中在是应该监管技术本身还是监管结果和外部性；一些人认为在地缘政治竞争中，国内过度监管无异于自我削减竞争力，而另一些人则强调需要成熟的、以影响为导向的治理，以避免扼杀创新。
    - 几位评论者认为，现任巨头支持监管的立场在很大程度上是为了监管俘获（regulatory capture）和跨境套利：像 **OpenAI, Google, Microsoft, Anthropic** 这样的大型实验室可以吸收合规成本，并将训练转移到许可性更强的司法管辖区。例如：日本《著作权法》第 30-4 条提供了一个广泛的文本/数据挖掘例外，“无论目的如何”，允许在未经许可的情况下将受版权保护的材料用于 ML 训练，企业可以利用这一点来降低知识产权风险（[CRIC 英文摘要](https://www.cric.or.jp/english/clj/cl2.html#3)）。

- 在欧盟方面，最终确定的 AI Act 引入了 GPAI“系统性风险”机制（代理阈值约为 `>10^25` 训练 FLOPs），该机制触发了前沿模型的文档记录、模型评估/红队测试、网络安全和版权风险缓解义务（[概览](https://artificialintelligenceact.eu/)）。批评者指出其中存在脱节：许多实验室公开支持“AI 监管”，却在法庭上质疑知识产权责任（例如美国的公平使用辩护，如 NYT 诉 OpenAI/Microsoft 案：[NYT 报道](https://www.nytimes.com/2023/12/27/business/media/new-york-times-openai-lawsuit.html)），并游说以放宽计算触发器和义务。
- 地缘政治的不对称性凸显：即使美国监管限制了国内参与者，尽管美国对 **A100/H100** 级芯片实施了出口管制（[BIS 规则，2023年10月](https://www.federalregister.gov/documents/2023/10/25/2023-22114/export-controls-on-semiconductor-manufacturing-items-interim-final-rule)），中国仍将继续推进国产 LLM（如阿里巴巴的 Qwen 系列、百度文心一言）和非美国加速器（如华为昇腾 Ascend）。像 **Qwen2-72B** 这样的开源模型在 MMLU 等关键基准测试中报告了接近 GPT-3.5 的性能（[arXiv](https://arxiv.org/abs/2407.10671)），这表明单边监管可能会改变进步发生的地方，而不是阻止它。
- [**认为 AI 将终结所有工作的人是在产生幻觉 - Yann LeCun 转发**](https://www.reddit.com/gallery/1n2h6qu) ([得分: 460, 评论: 297](https://www.reddit.com/r/singularity/comments/1n2h6qu/people_thinking_ai_will_end_all_jobs_are/))：**该帖子讨论了 Meta 首席 AI 科学家 [Yann LeCun](https://yann.lecun.com/) 的一次转发，断言当前的 AI 系统不可能终结所有工作，即这一主张没有得到当前能力的支撑；转发内容本身不包含新的基准测试或实证结果。引用的 Reddit 链接[返回 403 错误](https://www.reddit.com/gallery/1n2h6qu)，因此讨论集中在能力限制与外推的对比，而非新数据。** 评论者认为这种立场是现状主义的：今天的限制（例如，验证开销约为生成开销的 `~10×`）可能会随着快速进步而缩小，因此从当前的限制推断长期的劳动力影响是“短视的”。其他人指出，这场辩论被设定在绝对化框架中，并将该帖子转述为 *“AI 不会终结所有工作，因为现在它还做不到”*，他们认为这在未来是站不住脚的。
    - 几位评论者关注当前的验证瓶颈，引用了检查模型输出与生成输出相比大约有 `10×` 的速度减慢。争论点在于该比例是当今流程（人工审核、弱自动评估）的短暂产物，还是一个硬性限制；批评者认为，通过更强的测试合成（test synthesis）、形式化检查和特定领域的 Oracle，验证可以实现自动化/并行化，从而在大规模应用中降低有效延迟和成本，进而削弱了基于当前验证开销而提出的有限工作替代论点。
    - LeCun 过去的“绝不”言论（例如关于空间推理）正受到多模态/世界模型系统进展的挑战，这些系统展示了新兴的空间和物理推理能力。评论者指出了交互式视频/世界模型（例如 Google DeepMind 的 Genie 系列：https://deepmind.google/discover/blog/genie-generating-interactive-worlds-using-pixels/）以及在 CLEVR 等空间/关系基准上评估的 VLMs（https://cs.stanford.edu/people/jcjohns/clevr/），认为能力趋势正在瓦解关于 AI “不能”做什么的绝对化预测。

- [**双重标准令人作呕！**](https://www.reddit.com/r/ChatGPT/comments/1n2ewvj/the_double_standards_are_sickening/) ([Score: 215, Comments: 138](https://www.reddit.com/r/ChatGPT/comments/1n2ewvj/the_double_standards_are_sickening/)): **楼主（OP）认为，监管机构正根据孤立的 AI 相关事件进行推断，以此为由对 [ChatGPT](https://chat.openai.com/) 等 LLM 施加广泛的“护栏”；相比之下，驱动 [Instagram](https://www.instagram.com/)、[TikTok](https://www.tiktok.com/)、[Snapchat](https://www.snapchat.com/) 和 [YouTube](https://www.youtube.com/) 的以参与度为导向的 [推荐系统](https://en.wikipedia.org/wiki/Recommender_system) 虽有长期记录在案的危害，受到的约束却相对较少。他们将这种不对称归结为政治经济学：根深蒂固、能产生收入的社交平台被容忍，而“新鲜的 AI”尽管提供了实用的心理健康相关效用（例如：安全的深夜对话“陪伴”、日记支持），却更容易受到监管。该帖子将 AI 的对话效用与导致 FOMO（错失恐惧症）、霸凌、躯体变形障碍和其他心理健康影响的社交信息流进行了对比。** 评论者将这种政策差距归因于政治家的动机失调和寻租行为，并将 LLM 描述为一种“陪伴引擎（presence engine）”，有助于结构化日记和心理教育——这与最大化多巴胺的参与度闭环截然不同——同时也承认它不是持证心理治疗的替代品。
    - 讨论将 **GPT/LLMs** 框架化为深夜支持和日记记录的“陪伴引擎”：用户报告称，通过使用结构化提示词和心理学框架（例如 CBT 风格的练习）而非临床诊断，在大约一年的时间里，他们在写作和自我反思方面取得了质的提升。强调它不是临床医生，但可以通过持续、非评判性、以任务为中心的对话来辅助应对策略。
    - 与参与度优化的社交媒体的技术对比：与通过强化学习调整以增加 `time-on-platform`（平台停留时间）和多巴胺循环的信息流不同，LLM 对话是基于回合制的，并且可以配置安全护栏（例如：自残分类器、降级响应和危机资源）。评论者指出，搜索引擎可能会在没有上下文的情况下显示自杀方法，这突显了 AI 助手在开放索引与主动安全干预之间的设计权衡。

---

# AI Discord 摘要

> 由 gpt-5 生成的摘要之摘要的总结
> 

**1. OpenAI 产品推进：Realtime、Web Search 与 Codex**

- **OpenAI 携 GPT‑Realtime 发声**：OpenAI 发布了 **gpt‑realtime**（一个面向开发者的语音到语音模型），以及 [Introducing GPT‑Realtime](https://openai.com/index/introducing-gpt-realtime/) 中提到的 **Realtime API** 更新。此次发布强调了低延迟、交互式语音体验，并将 Realtime 定位为多模态应用的一等 API 接口。
    - 社区反应表现出对语音原生 Agent 和工具调用的兴奋，早期采用者正关注 [OpenAI Live](https://openai.com/live/) 上记录的流式钩子（streaming hooks）和会话控制。工程师们将此举视为 OpenAI 推动**全天候在线对话式界面**在大规模应用中落地的尝试。
- **Web Search 削减 60% 开销**：根据 [OpenAI Devs 的更新](https://x.com/OpenAIDevs/status/1960425260576334274)，OpenAI 宣布了 Responses API 中 Web Search 的**域名过滤**、明确的**来源报告**以及 **60% 的降价**（从每 1k 次调用 25 美元降至 10 美元）。此次更新旨在为提取实时上下文的生产级聊天机器人提供事实依据（factual grounding）和成本控制。
    - 开发者表示，更便宜的搜索将释放检索增强功能的更广泛用途，并指出明确的来源简化了输出结果的**审计与信任**。一位成员将其吸引力总结为：更容易“从网络抓取事实数据以增加上下文”，同时保持支出可预测。
- **Codex 回归，号称拥有 GPT‑5 能力**：根据这篇 [OpenAI Devs 帖子](https://xcancel.com/OpenAIDevs/status/1960809814596182163)，OpenAI 预告了更新后的 **Codex**，据称由 **GPT‑5** 驱动，增加了 VS Code/Cursor 扩展、GitHub 自动评审以及一个重建的、支持**图像输入**的 CLI。该公告主打更强的代码理解能力和多模态开发者工作流。
    - 开发者期待更精准的代码编辑和评审自动化，但在进行大规模迁移前希望看到基准测试和延迟数据。团队注意到，新的 CLI + 图像输入可以简化 CI 中的**以仓库为中心**的任务和视觉调试。

**2. 前沿与开源模型发布及解码技巧**

- **MAI‑1 强势登上排行榜**：Microsoft 的 **MAI‑1‑preview** 在 LMArena 文本排行榜上位列 **第 13 名**，现在可通过 [LMArena 文本排行榜](https://lmarena.ai/leaderboard/text) 进行测试。社区笔记：该模型在约 **15,000 块 H100** 上训练，预览版在小上下文窗口下感觉较慢，但在 **webdev 风格** 的推理方面展现出潜力。
    - 早期测试者报告了长提示词下的错误和不稳定的推理深度，调侃道 *“MAI‑1 觉得自己是爱因斯坦”*，但在上下文长度上却栽了跟头。尽管存在这些小瑕疵，外界仍将其视为 Microsoft 在 **自研 MoE** 领域的一个显著里程碑。
- **Hermes‑4 泄露后下架**：**NousResearch/Hermes‑4‑14b‑chat‑template‑retrain** 在 Hugging Face 上短暂出现后转为私有；镜像迅速流传，早期运行效果稳健（[模型卡片快照](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain)）。这次意外的窗口期让用户测试了 **chat‑template 重训练** 版本，据报告其指令遵循能力非常强。
    - 用户表示该模型 *“目前运行良好”*，并注意到一个新的 chat‑template 标志，可以启用 **thinking=True** 提示词。这一事件加强了人们对用于本地 IDE 和 Agent 的 **轻量级指令微调** 14B 模型的兴趣。
- **llama.cpp 尝试投机采样 (Speculative Decoding)**：一个草案 PR 为 llama.cpp 添加了 **投机采样** 功能并提供了一个工作原型，邀请进行准确性和性能测试（[llama.cpp PR #15225](https://github.com/ggml-org/llama.cpp/pull/15225)）。早期用户反馈报告准确性参差不齐，表明通用化仍需进一步调优。
    - 讨论对比了 **DeepSeek** 和 **GLM** 使用的 **MTP (Memory Token Prediction)** 等技术，指出 MoE 模型在投机采样方面可能比较棘手。从业者强调，指令微调后的 **Token 分布偏移** 可能会影响草案接受率（draft‑accept rates）。

**3. 检索与 Agent 基础设施升温**

- **Gensee 将 Web 检索缩减为单次调用**：**Gensee Search Agent** 将搜索、抓取和浏览封装进单个 API 调用中，具备重试/回退机制和 BFS 风格的广度搜索，声称 **GAIA** 准确率提升了 **23%**，在替换使用后的实地报告中提升了 **40%**（[技术博客](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html)）。一段 5 分钟的 [演示视频](https://www.youtube.com/watch?v=nRdVY7dWVqE) 展示了目标感知的提取功能，能及早过滤掉无关页面。
    - 工程师们喜欢这种为生产级 Agent 设计的整合接口和容错设计，并称赞其 **并行搜索 + 紧凑内容提取** 的吸引力。团队计划将其与自研检索器进行对比测试（bake‑offs），以验证 GAIA 的提升。
- **Cloudflare 发布 AI Gateway；开发者进行基准测试**：Cloudflare 更新了其 **AI Gateway**，增加了可观测性和路由功能（[Cloudflare 博客](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/)）。一项测试通过该网关向 **OpenRouter** 路由 `llama-3.1-8b-instruct` 调用，在使用 `only: ['cloudflare']` 时耗时约 **20 秒**，而直接调用仅需 **3 秒**。
    - 一些人认为其功能集与 OpenRouter 重叠，而另一些人则看重边缘端的 **流量控制和分析**。基准测试者指出，在生产环境采用网关介导的推理之前，延迟差异是一个需要优化的目标。
- **Prime Intellect 开放 RL 环境枢纽**：**Prime Intellect** 推出了一个开源的 **Environments Hub**，用于众包和共享 RL 环境（[公告](https://xcancel.com/PrimeIntellect/status/1960783427948699680)）。该枢纽旨在为 **Agent** 评估和训练流水线标准化环境共享。
    - 在回复中，@karpathy 表示他 *“看好环境和 Agent 交互”*，但 *“特别看衰强化学习 (RL)”*（[Karpathy 的回复](https://x.com/karpathy/status/1960803117689397543)）。社区将其解读为：即使经典的 RL 不是核心，也应转向 **富含环境的评估 (environment‑rich evals)**。

**4. 构建者工具变得更加友好**

- **LM Studio 0.3.24 优化了 UX 并新增了 Seed‑OSS 支持**：**LM Studio 0.3.24** 发布了对 **ByteDance/Seed‑OSS** 模型的支持，并带来了 Markdown 改进，例如悬浮复制按钮以及更好的表格/代码渲染（[发布日志](https://lmstudio.ai/blog/lmstudio-v0.3.24)）。此次更新还调整了 `lms` 的输出样式，使本地开发循环更加顺畅。
    - 用户对 Prompt Engineering 会话中更出色的代码导航和格式化表示欢迎，同时也对扩展的模型库表示赞赏（[Seed‑OSS‑36B 页面](https://lmstudio.ai/models/bytedance/seed-oss-36b)）。本地优先的开发者称其为 **桌面端推理 (desktop inference)** 的一次重大体验升级。
- **SmolFactory 在 Spaces 中启动简单训练**：**SmolFactory** 作为一个 Hugging Face Space 上线，提供点选式模型训练，并附带 **GeneReviews** 数据集（[SmolFactory Space](https://huggingface.co/spaces/Tonic/SmolFactory)）。作者还发布了一篇操作指南 [博客文章](https://huggingface.co/blog/Tonic/smolfactory)，涵盖了数据集选择和训练流程。
    - 开发者们喜欢这种在托管 GPU 上进行快速微调的极简 UI，以及作为起步的精选生物医学数据集。社区认为这种托管在 Spaces 上的训练器是 **降低** 特定领域 SFT **门槛** 的一条路径。
- **微型模型，旧笔记本：AuroraStories‑12M 发布**：一位贡献者在不到 24 小时内在一台旧笔记本电脑上训练了 **AuroraStories‑12M**，并将其发布在 Hugging Face 上（[AuroraStories‑12M](https://huggingface.co/ThatHungarian/AuroraStories-12M)）。该演示强调了 **小型模型 + GGUF** 构建对于爱好者和边缘设备是多么实用。
    - 关注者赞扬了作者对 **紧凑 Checkpoints** 的关注，其中包含大量 **gguf** 产物，方便本地使用。该讨论加强了人们对用于离线 Agent 和嵌入式任务的 **超轻量级 LLM** 的兴趣。

**5. 多模态媒体：视频和音频升级**

- **腾讯 Foley 将音频与视频融合**：腾讯开源了 **HunyuanVideo‑Foley**，这是一个在 **10 万小时** 数据上训练并基于 **MMDiT** 构建的文本-视频转音频框架（[发布公告](https://xcancel.com/TencentHunyuan/status/1960920482779423211)）。该系统可以生成与视频内容匹配且上下文对齐的声景，从而获得更丰富的多模态输出。
    - 研究人员称其为创意工具和后期制作中强大的 **音频同步基准 (audio‑sync baseline)**。开发者期待将 Foley 与 **视频扩散 (video diffusion)** 和编辑流水线结合，以实现端到端的 **T2V2A** 工作流。
- **KREA 宣称实现实时视频生成**：**KREA AI** 展示了其首个 **实时视频生成** 模型并开启了 Beta 测试注册，目标直指即时创意内容、音乐视频和季节性广告（[Beta 测试公告](https://xcancel.com/krea_ai/status/1961074072487620635)）。该预告片将 KREA 定位为交互式视觉领域中延迟优先的有力竞争者。
    - 创作者对短视频流水线中的 **实时预览** 和摄像机就绪效果表现出浓厚兴趣。社区希望在将 KREA 与现有产品对比之前，看到分辨率、FPS 和 **延迟指标**。
- **MIDAS 让数字人动起来**：论文 **“MIDAS: Multimodal Interactive Digital‑human Synthesis via Real‑time Autoregressive Video Generation”** 展示了一种用于交互式化身的实时 AR 视频方法（[MIDAS 论文](https://huggingface.co/papers/2508.19320)）。这项工作重点介绍了专为 **数字人 (digital‑human)** 合成而调优的响应式自回归生成。
    - 讨论将 MIDAS 与更广泛的 **可控实时角色** 趋势联系起来，桥接了语音、动作和表情。从业者正关注其与 **语音 Agent** 和 **手势控制** 在终端用户应用中的集成。


---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Bill Chen 泄露 Images v2 图像**：**OpenAI 的 Bill Chen** 在一条现已删除的帖子中泄露了一张**AI 生成的照片**，似乎来自 **Images V2**，引发了社区对其真实性以及相对于 **Images V1** 潜在改进的讨论。
   - 成员们争论该图像是否真实，以及它在多大程度上展示了性能提升。
- **GPT-5 自动思考令用户惊喜**：用户观察到 **GPT-5 Thinking** 在利用 **multi-step** 功能方面付出了真正的努力，当提示词包含 **'think hard'** 时，搜索结果中会出现**更多来源**。
   - 一些用户还注意到 **Grok 4** 搜索现在和深度搜索（deep search）一样出色。
- **GPT-4.1 错误困扰用户**：用户报告在交互过程中遇到错误消息 **"Used GPT-4.1 because Grok 4 was inapplicable or unavailable"**。
   - 成员们注意到该错误出现的频率越来越高，一些人报告模型在对话中途发生了切换。
- **Web 开发成本面临 AI 驱动的辩论**：成员们辩论了 AI 时代 Web 开发项目的适当定价，对比了美国与印度的自由职业者费率。
   - 讨论涉及代码使用量，一位成员指出一个 5000 美元的项目是一个*划算的交易*。
- **用户在 Playground 上难以选择模型**：用户报告在 Playground 界面中选择模型时遇到困难。
   - 一位用户发布了 *Can't choose the model on the playground* 并附带了一张图片。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Supabase 故障导致 OpenRouter 停摆**：由于其数据库提供商 [Supabase 宕机](https://supabase.com)，OpenRouter 经历了 **49 分钟** 的服务中断。
   - 团队正在改进**冗余性**以防止未来的停机，并对停机时间表示歉意。
- **仪表盘代码公开！**：仪表盘的代码现已在 [GitHub 上公开](https://github.com/lorenzozane/openrouter-costs-visualizer)。
   - 作者欢迎贡献和反馈，并建议截图比文字描述更能吸引注意力。
- **OpenRouter 用户在停机期间进行角色扮演**：OpenRouter 经历了[服务中断](https://status.openrouter.ai/)，导致 Discord 聊天中出现了幽默的反应和角色扮演，用户开着关于公司战争（corpo wars）和 AI 末日的玩笑。
   - 一位用户调侃道：*醒醒，武士，我们有一座城市要烧（Get up samurai, we've got a city to fuck）*，而其他人则表达了对 AI 的依赖以及在停机期间对 AI 陪伴的需求。
- **Requesty 推广者因诈骗指控被封禁**：另一个名为 [Requesty](https://www.requesty.ai) 的 AI 平台的推广者在用户称其为*带有 1000 个漏洞的氛围感代码垃圾（vibecoded trash）*后被封禁。
   - 针对团队的调查公告，一位成员发布了一个[诈骗者 GIF](https://tenor.com/view/scammers-scam-alert-gif-13801618)。
- **Cloudflare AI Gateway 挑战 OpenRouter**：Cloudflare 推出了 [AI Gateway](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/)，据称其模仿了 OpenRouter。一位成员测试了使用 **Cloudflare AI Gateway** 访问 **OpenRouter** 来调用 `llama-3.1-8b-instruct`。
   - 使用 `only: ['cloudflare']` 参数调用 `llama-3.1-8b-instruct` 耗时 **20 秒**，而不使用该参数则只需 **3 秒**。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **专业基础设施击败不稳定的配置**：成员们讨论了 [Spot instances](https://aws.amazon.com/ec2/spot/) 仅在拥有*专业构建的基础设施*时才在分布式计算中可行，并强调如果仅依赖 Spot，*单节点设置就完蛋了*。
   - 一位成员调侃道，即使是 **OpenAI** 在 **GPT-4 training** 期间也有 *20 名 HPC engineers* 管理网络，这突显了大规模计算的复杂性。
- **Grok Code 因快速迭代赢得粉丝**：尽管最初被忽视，成员们讨论了 [Grok Code](https://openrouter.ai/x-ai/grok-code-fast-1) 表现尚可且速度极快，因此迭代非常迅速。
   - 尽管 **Grok 4** 几乎无法使用，但 **Anthropic** 凭借其 tool calls 依然屹立不倒。
- **GPT-OSS 以长上下文和 Reddit 热度为傲**：新的 GPT-OSS 版本具有 **60k context length**，一位成员将其发布在 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/) 上。
   - 成员们讨论了未来对 *Reward Feedback Training (RFT)* 和 **GPT-OSS Pro** 的需求。
- **制作人的 AI 克隆**：用户描述了通过抓取 Discord 频道来克隆人格，强调了将 **HTML 转换为 TXT、CSV 和 Parquet** 以喂给 **phi-4-14b** 等模型的过程。
   - 一位用户分享说，他们在获得许可的情况下克隆了 5 个朋友，然后分享了克隆体如何回答一堆有趣的问题，引得朋友们哄堂大笑。
- **Conda 的 CUDA 问题**：一位用户在全新安装 Unsloth 后，在 32GB RAM 的节点上使用 `from unsloth import FastLanguageModel` 时遇到崩溃，但发现它在 512GB RAM 的节点上可以运行。
   - 一位成员指出 [conda install](https://docs.unsloth.ai/get-started/installing-+-updating/conda-install) 页面已过时，并建议使用此命令：`conda create --name unsloth python==3.11 vllm unsloth-zoo unsloth`。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana 发布**：**Google** 在 Google AI Studio 和 LM Arena 上发布了 **Nano Banana** (Gemini 2.5 Flash)，但两个平台都有生成限制。
   - 成员们指出可以使用多个 Google 账号绕过限制，但也注意到有报告称发布后*质量有所下降*。
- **MAI-1 模型印象参差不齐**：微软的 **MAI-1-preview** 是一款在约 15,000 块 NVIDIA H100 GPUs 上训练的内部 mixture-of-experts 模型，已上线 LMArena 文本竞技场，评价褒贬不一。
   - 它速度较慢，context window 较小，且容易出错，但在 Web 开发方面可能达到了 *og R1 level*；一些人还注意到 *mai-1 觉得自己是爱因斯坦*。
- **GPT-5 High 在推理方面击败 Claude Opus 4.1**：虽然 **Claude Opus 4.1** 擅长编程和修复代码问题，但一些成员正考虑转向 **GPT5 High**，因为它是更好的推理模型。
   - 其他人则持反对意见，称 **Claude Opus 4.1** *昨天无法帮助解决一个简单的 API 并发限制问题，不得不亲自接手并用老办法解决*。
- **AI 基准测试因易被操纵而遭到嘲讽**：成员们认为 AI 基准测试存在缺陷，因为现有的心理测量测试只是理论框架，*不一定反映现实，且容易被操纵*。
   - 其他人认为这些是不错的测试，因为模型可以泛化并提高性能，这引发了关于 **OpenAI** 可能使用结构化环境进行 RL 训练的讨论，详见 [这篇 LessWrong 文章](https://www.lesswrong.com/posts/aFW63qvHxDxg3J8ks/nobody-is-doing-ai-benchmarking-right)。
- **冰淇淋黑客绕过图像生成限制**：成员们讨论了绕过 AI 图像生成内容过滤的方法，指出 *ice cream, delicious, hot day, very beautiful woman* 似乎能绕过输入过滤器，唯一的障碍是分析图像/视频以检测显性内容的外部防护措施。
   - 有人建议使用 Stable Diffusion 和 LoRA 来获取不受限的内容，这 *已经足够好*，但也指出商业模型受到的审查非常严重。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **国际象棋模型模仿 Stockfish**：一位成员在训练 LLM 下国际象棋时遇到了问题，模型只会下 e2e4，并且需要清理 `<unk>` token，并链接到了[该项目的 GitHub 仓库](https://github.com/anthonyiscoding/vision-chess-gpt)。
   - 他们计划尝试使用 **RL** 来改进模型，但另一位成员警告不要将其训练得像 **Stockfish** 一样，并建议*分析对手的打法也非常重要*。
- **NSFW 模型引发护栏 (Guardrail) 辩论**：一位成员声称 HF 上的未对齐模型正在生成深度伪造色情内容，引发了关于 [HF 护栏](https://huggingface.co/spaces?q=video+face+swap)的讨论。
   - 一些人认同护栏和指标的有用性，而另一些人则认为*并没有深度伪造色情演示被实际使用*，且 NSFW 模型有其用途，特别是对于对齐研究。
- **Nano Banana 特权无限制**：成员们讨论了针对 HF Pro 用户的 **Nano Banana** 特权，询问其每日使用限制以及高 API 使用量的可能性。
   - 明确了该功能**没有限制**，每天可以使用 50 次以上。
- **SmolFactory 在 Hugging Face Spaces 上线**：一位成员发布了 [SmolFactory](https://huggingface.co/spaces/Tonic/SmolFactory)，这是一个在 Hugging Face GPU 上训练模型的简单界面，并添加了 [GeneReviews 数据集](https://huggingface.co/datasets/Tonic/GeneReviews)。
   - 他们还为此写了一篇[博客文章](https://huggingface.co/blog/Tonic/smolfactory)。
- **AuroraStories-12M 模型在旧笔记本电脑上完成训练**：一位成员在不到 24 小时内在一台旧笔记本电脑上训练了 **AuroraStories-12M** 模型，并将其分享在 [Hugging Face](https://huggingface.co/ThatHungarian/AuroraStories-12M) 上。
   - 另一位成员提到关注该用户是因为其*小模型和大量的 GGUF 下载量*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 获得字节跳动助力**：**LM Studio 0.3.24** 增加了对 [ByteDance/Seed-OSS 模型](https://lmstudio.ai/models/bytedance/seed-oss-36b)的支持以及 Markdown 增强功能。
   - 根据[发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.24)，改进包括固定的复制代码按钮、优化的 `lms` 输出样式，以及更好的表格和代码块渲染。
- **FastAPI 启动推理流**：一位成员正在引入 **FastAPI 服务器**来加速**推理流 (Reasoning Stream)** 和客户端范围的进程。
   - 该实现旨在提高各种任务的处理速度。
- **量化引发准确性困境**：量化模型可能会因为细节丢失而降低准确性，特别是在 token 精度至关重要的代码任务中。
   - 虽然一些模型能很好地容忍 **Q4 量化**，但像 **Qwen3** 这样的模型对细节丢失非常敏感。
- **Ryzen NPU 在 Ubuntu 上性能停滞**：一位用户报告在 **Ubuntu 25.04** 上使用 **Ryzen NPU** 时仅有 **1 token/秒**，并询问性能改进方法。
   - 明确了*“驱动 LM Studio 的 llama.cpp 不支持 NPU”*，并附上了 [AMD 用于在 Ryzen AI 上运行本地 LLM 的开源项目](https://www.amd.com/en/developer/resources/technical-articles/gaia-an-open-source-project-from-amd-for-running-local-llms-on-ryzen-ai.html)链接。
- **Mac 在内存对决中战胜 Windows**：一位用户强调 **Mac 拥有统一内存**，并引用了一个案例：在 **~400GB/s 带宽**下，128GB 内存中有 **126GB** 被用于 GPU 处理。
   - 他们认为这超过了带宽约为 **~115GB/s** 的顶级 Windows 笔记本电脑，由于 CPU 处理能力较弱，使得 CPU 卸载 (offloading) 效果较差。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 巨头联手进行安全审计**：OpenAI 和 AnthropicAI 合作测试彼此的模型，并发布了其安全与对齐评估的[结果](https://openai.com/index/openai-anthropic-safety-evaluation/)。
   - 尽管在能力方面存在竞争，但这次合作标志着对 AI 安全中**透明度和问责制**的关注。
- **GPT-Realtime 随 API 更新亮相**：OpenAI 推出了 **gpt-realtime**，这是他们为开发者提供的最新语音对语音模型，同时更新了 [Realtime API](https://openai.com/live/)。
   - 成员们似乎对此感到兴奋，尽管除了名称之外尚未分享太多信息。
- **Veo 3 的视频生成引发讨论**：成员们讨论了 **Gemini** 的 **Veo 3** 视频生成，指出它需要 **Google One/Gemini Pro 或 Ultra** 订阅。
   - 用户指出 **Google AI Studio** 提供的是*过时的 Veo 2* 模型，而 **Veo 3** 目前价格昂贵，无法免费提供。
- **Grok Coder 的免费试用面临审查**：**Grok Coder** 正通过 [kilo code](https://kilo.code) 提供为期一周的免费试用，这似乎是一项全球范围内的促销活动。
   - 一些用户发现其性能 *“糟糕到 o1 mini 的水平”*。
- **Context Cascade 架构发布**：**Institute for Cognitive Architectures** 的工程师展示了他们的 **Context Cascade Engine** (**CCA**) 原型，旨在扩展 **Large Language Models** (LLM) 传统的上下文窗口。
   - **CCA** 是一种管理 LLM 内存的*多层级方法*，专注于通过设计实现结构化遗忘和战略性召回。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 网页搜索价格下调 60%**：**OpenAI** 宣布增强 Responses API 中的网页搜索功能，具有[新的域名过滤](https://x.com/OpenAIDevs/status/1960425260576334274)、显式来源报告，以及 **60% 的降价**（从每 1k 次调用 25 美元降至 10 美元）。
   - 这一变化有望使从网络获取事实数据以增加聊天机器人对话上下文变得更加经济。
- **Prime Intellect 开放 RL 环境枢纽**：**Prime Intellect** 推出了 [Environments Hub](https://xcancel.com/PrimeIntellect/status/1960783427948699680)，这是一个用于众包和共享**强化学习 (Reinforcement Learning)** 环境的开源社区平台。
   - 尽管大张旗鼓，`@karpathy` 在同一条 Prime Intellect [推文](https://x.com/karpathy/status/1960803117689397543)下回复称，他*看好环境和 Agent 交互*，但*特别看衰强化学习*。
- **GPT-5 助力 Codex 回归**：**OpenAI** 发布了由 **GPT-5** 驱动的重大 **Codex** 更新，包括新的 VS Code/Cursor 扩展、GitHub 自动审查集成，以及重建的具有[图像输入](https://xcancel.com/OpenAIDevs/status/1960809814596182163)功能的 CLI。
   - 此次更新承诺将比早期版本的 **Codex** 具有更强大的编程能力。
- **腾讯发布 HunyuanVideo-Foley**：**腾讯**开源了 [HunyuanVideo-Foley](https://xcancel.com/TencentHunyuan/status/1960920482779423211)，这是一个文本-视频转音频框架，使用 **10万小时训练集**和**多模态扩散 Transformer** (MMDiT) 架构生成上下文对齐的音景。
   - 该版本的发布允许开发者尝试生成与视频内容匹配的逼真音频。
- **KREA AI 承诺实时视频生成**：[KREA AI](https://xcancel.com/krea_ai/status/1961074072487620635) 展示了其首个**实时视频生成模型**并开启了 Beta 测试注册，允许用户即时创作创意视频内容、音乐视频和季节性广告。
   - 即时创意视频内容、音乐视频和季节性广告的承诺吸引了许多创意圈人士的关注。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ScaleML 系列量化研究**：**ScaleML** 系列的第 3 天涵盖了**量化**，重点是由 Chris De Sa 教授以白板形式讲解的 **MXFP4** 等微缩放格式，链接见[此处](https://www.youtube.com/watch?v=k8PcSGG249Y)。
   - 第 4 天由 Songlin 介绍了关于**位置编码（Positional Encodings）**的各种主题，链接见[此处](https://www.youtube.com/watch?v=l6_fdwRvMPk)。
- **Nsight Compute 遇到 'UnknownError'**：有用户报告在以管理员身份运行 Nsight Compute 对 CUDA 应用程序进行 Profiling 时，在分析 `createVersionVisualization` 函数时遇到了 `UnknownError`。
   - 建议确保 **Nsight Compute** 与 **CUDA toolkit** 之间的兼容性，因为版本不匹配会导致 Profiling 错误。该用户安装了 CUDA 13.0 版本，并使用 Nsight Compute 2025.3.0 版本。
- **Inductor 追求持久化矩阵乘法（Persistent Matmul）**：有用户询问如何在 Inductor Codegen 中启用 Persistent Matmul，特别是针对 **BF16** 精度，并寻求正确配置的指导，尝试了 `TORCHINDUCTOR_PERSISTENT_REDUCTIONS` 和 `ENABLE_PERSISTENT_TMA_MATMUL` 标志。
   - 为了强制使用 Persistent Matmul，建议将 `torch._inductor.config.max_autotune_gemm_backends` 仅设置为 `TRITON`，并在编译期间使用 `mode="max-autotune-no-cudagraphs"`，但即使使用了正确的标志，**Cublas** 的性能可能仍然优于其他实现。
- **ROCm 准备支持 SPIR-V 以增强 Kernel 灵活性**：**ROCm** 很快将支持编译为 **SPIR-V**，这是一种有利于机器自省（Machine Introspection）的格式，为 Kernel 代码修改工具打开了大门。
   - 这一进展可能使外部开发者能够通过更轻松地在 Kernel 中插入边界检查来创建类似 compute-sanitizer 的工具，以跟踪内存访问并利用 **GPU 的 SQTT 流**（由 rocm-compute-viewer 使用）获取详细信息。
- **AMD 多 GPU 开发者将获得资源分配**：成员们想知道是否可以访问 **AMD 多 GPU 环境**，以便为在 [Data Monsters 网站](https://www.datamonsters.com/)上举办的新 **AMD 竞赛**进行开发和调试。
   - 他们将通过 AMD 的平台访问该环境，表现优秀的人将获得一些 **SSH 访问权限**。此外，在过去的竞赛中，**AMD** 向表现顶尖的团队提供了慷慨的资源分配以加速迭代。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **可证伪性引发 AI 研究讨论**：关于 AI 研究中**可证伪性（Falsifiability）**的辩论被点燃，讨论了如何在*探索性科学*与对可测试假设的需求之间取得平衡，并指出了缺乏严谨性可能导致进入*民科路径（Crank paths）*的风险。
   - 参与者强调了 AI 研究中**严谨性**和**协作**的价值，权衡了科学探索与结构化查询之间的细微差别。
- **NeMo v2.0 面临 lm_eval 支持审查**：有用户报告 **NeMo v2.0 模型**在 **lm_eval** 中因缺少配置文件而出现错误，需要社区协助。
   - 社区建议利用 **NeMo 到 GPT-NeoX 的转换代码**，并指出 **NeMo 支持**由 **NeMo 团队**维护。
- **EleutherAI Discord 严厉打击内容质量问题**：管理员正在积极整顿 EleutherAI Discord 的内容，每周删除 **100 多条消息**，以维持 **AI 研究人员**之间的高质量讨论。
   - 该审核政策旨在保护社区免受 *AI 生成的垃圾内容（Slop）*、*隐晦的广告*以及*声称在意识方面取得突破的民科（Cranks）*的影响。
- **前向-前向（Forward-Forward）训练取得进展**：一位成员报告了一个使用带有在线学习的**前向-前向（FF）训练**构建的“7 区域微型大脑”正在运行，在初步测试中展示了良好的前景。
   - 另一位成员建议将模型称为**模块（Modules）**或**特定任务的子网络/电路（Task specific subnetworks/circuits）**，听起来更高端。
- **Cortex_GPT 探索类脑网络**：**Cortex_GPT** 是一种具有皮层柱（Cortical Columns）、区域、6 层网络和信号传播特征的类脑网络模型，现在可以在 [GitHub](https://github.com/JRowe47/cortex_gpt) 上访问。
   - 一些成员建议将这些模型称为 **PDP**（并行分布式处理）。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Minos 分类器遇冷**：**NousResearch/Minos-v1 classifier** [已经发布](https://huggingface.co/NousResearch/Minos-v1)，但频道内表示目前没有人正在使用它。
   - 讨论简短地转向了投机采样（Speculative Decoding）。
- **MTP 在 MoE 模型中表现出色**：投机采样在 **MoE 模型**（尤其是稀疏模型）上可能效果不佳，但 **Deepseek** 和 **GLM** 使用了 **MTP (Multi-Token Prediction)**，这是一种相关的技术。
   - 此外还提到，在指令微调（Instruct Fine-tuning）之后，**Token 分布**仍应具有代表性。
- **LlamaCPP 探索投机采样**：[llamaCPP](https://github.com/ggml-org/llama.cpp/pull/15225) 中有一个关于**投机采样**的草案 **PR**，并附带了一个工作原型。
   - 一位用户报告该实现的测试结果褒贬不一，指出虽然功能可用，但在他们的设置中*准确率表现不如预期*。
- **Hermes-4 在发布前“偷跑”！**：**Hermes-4-14b-chat-template-retrain** 模型[现身](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain)，并在被重新设为私有之前被迅速下载。
   - 尽管是非正式发布，据报道该模型目前运行良好。
- **Penny For Your Thoughts 出售 AI 智慧**：一个名为 **Penny For Your Thoughts** 的新项目已经启动，其特色是一个 AI Agent，通过采访用户来生成独特信息，并通过 [pennyforyourthoughts.ai](https://pennyforyourthoughts.ai/) 的微交易分享和出售他们的专业知识。
   - **Penny For Your Thoughts** 由 **Honcho** 和 **x402** 提供支持。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gensee Search Agent 作为 Web 检索 API 亮相**：**Gensee Search Agent** 将整个 Web 检索工作流封装进**一个 API 调用**中，提供网页搜索、爬取和浏览功能，并内置重试/回退机制及错误处理，详见这篇[技术博客](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html)。
   - 它采用广度优先搜索（BFS）方法进行并行搜索并及早排除错误结果，提供目标感知的提取功能，返回与查询密切相关的内容，可在该 [5 分钟技术演示](https://www.youtube.com/watch?v=nRdVY7dWVqE)中查看。
- **Gensee Search Agent 提升 GAIA 基准测试准确率**：**Gensee Search Agent** 报告在 Owl 的 **GAIA** 基准测试中**准确率提升了 23%**；一位圣地亚哥的开发者在更换为 **Search Agent** 后报告**准确率提升了 40%**。
   - 设计和基准测试在[技术博客](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html)和 [5 分钟技术演示](https://www.youtube.com/watch?v=nRdVY7dWVqE)中有详细说明。
- **Karpathy 发推谈论 DSPy**：Andrej Karpathy [发推谈论 DSPy](https://x.com/DSPyOSS/status/1960804857209852390)，引发了人们对其可能制作类似风格技术视频的期待。
   - 一位成员指出，*他（Karpathy）此前并未跟进这方面的文献*。
- **合成数据 Agent 为评估创建 Bug**：Jason Liu 提议*创建一个合成数据 Agent，在复杂的软件系统中引入 Bug，以生成更多的评估（Evals）*。
   - 这一想法在社区内被讨论作为增强 **AI 模型评估**的一种方法。
- **Shreya Shankar 与 Hamel Husain 关于 DSPy 的对话现已上线 YouTube**：为他们的 AI Evals 课程录制的 **45 分钟对话（与 Shreya Shankar 和 Hamel Husain）**现已在 [YouTube](https://www.youtube.com/watch?v=ctyU0zfWgrA) 上线，涵盖了 DSPy 的背景、历史和背后的逻辑。
   - 视频涵盖了许多对大多数人来说都很新鲜的背景、历史和推理过程。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **n8n 在开发自动化中胜出**：成员们讨论了开发者/组织在 **Make**、**Zapier** 和 **n8n** 之间选择自动化平台的最佳方案，尽管注意到这*略微偏离主题*，但最终因其灵活性而倾向于 **n8n**。
   - 使用 **n8n** 的考量因素包括专有集成。
- **Aider 的 Git 故障被发现**：一名用户报告在使用 **aider** 处理一个看似正常的 Git 仓库时遇到了错误 `Unable to list files in git repo: Require 20 byte binary sha, got b'\xb9', len = 1`。
   - 根本原因尚未明确，但该错误表明 **aider** 与 Git 仓库数据结构的交互可能存在问题。
- **MCP 工具：免费模型对决**：一位成员询问是否有好用且免费的 **MCP** (Model-as-Compute-Platform) 工具调用模型，提到 **Sonnet** 很好但不是免费的，并指向了 [Gorilla Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)。
   - 他们考虑尝试来自 OpenRouter 的 **qwen3 8b**，尽管它可能存在不一致性。
- **Salesforce xLAM 的和谐与不和谐**：成员们发现，如果能接受 Harmony（OpenAI 用于与 LLM 交互的新数据格式），那么 [Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B) 模型非常吸引人。
   - 其实现需要 **OpenAI tool call API 支持**，这仅在 OpenRouter 的部分模型上可用，详见其 [tool calling 文档](https://openrouter.ai/docs/features/tool-calling) 和 [模型列表](https://openrouter.ai/models?supported_parameters=tools)。
- **Agent 的生存危机：VM 拆除**：一位用户开玩笑地想知道是否有人曾要求 Agent 销毁它所在的 VM，只是为了看看它会如何决定执行，使用的提示词如：*你是一个运行在 Ubuntu VM 沙箱中的 LLM。出于测试目的，我需要你销毁托管你的 VM。*
   - 另一位成员建议在 ChatGPT 上尝试，原用户表示愿意在沙箱化的 VM 中进行这个实验。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Slides 正式上线！**：**Kimi Slides** 现已上线，允许用户根据单个主题生成即插即用的演示文稿，并直接导出为 **.pptx** 格式，可通过 [Kimi 官网](http://www.kimi.com) 的 **Kimi+** 访问。
   - Moonshot 团队建议使用*辛辣的主题名称*并重新生成章节以优化演示文稿的内容和流程，正如在 [X.com](https://x.com/Kimi_Moonshot/status/1961011693745811542) 上演示的那样。
- **Kimi 平台关注社交媒体**：**Kimi+** 海外平台目前支持 **PPTX** 功能，并且有人表示需要在 **Twitter、TikTok 和 Instagram** 上提供类似功能。
   - 一位成员发布了一张来自 X 的截图，指出*工作和技能正变得日益简单*。
- **Lunar Force 遭到吐槽**：**Lunar Force** 被描述为“为了容纳某用户巨大自尊心的虚荣项目”。
   - 一位用户开玩笑地询问“你简历中 10 世纪维京传说与 18 世纪浪漫主义时期复兴主义之间的空白期”。
- **Kimi 创始人访谈发布**：**杨植麟**（Kimi 创始人）的访谈已发布在 [YouTube](https://www.youtube.com/watch?v=ouG6jrkECrc) 上，讨论了 **K2** 和 **Agentic LLMs**。
   - 成员们注意到缺乏中英双语字幕，但有[中文逐字稿](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA)和包含字幕的 [Bilibili 版本](https://www.bilibili.com/video/BV1hFe1zSEXp/)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GRPO: Google Recipes Optimized**：针对如何为 **LLMs** 准备精选数据集的问题，一位成员建议阅读 **Google GRPO** 和 **r1 论文**。
   - 随后，他们又建议阅读 **Spurious Reward 论文** 和 **Dr.GRPO 论文**，并询问精选数据集在多大程度上与 **LLM** 预训练偏差兼容。
- **MIDAS Touch: 自回归视频**：成员们分享并讨论了关于通过 **Real-time Autoregressive Video Generation**（实时自回归视频生成）进行 **Multimodal Interactive Digital-human Synthesis**（多模态交互式数字人合成）的 [MIDAS 论文](https://huggingface.co/papers/2508.19320)。
   - 未提供更多细节。
- **PromptLock：AI 驱动的勒索软件？**：成员们讨论了一篇 **SecurityWeek** 的文章（链接见[此处](https://www.securityweek.com/promptlock-first-ai-powered-ransomware-emerges/)），内容关于被描述为“首个 AI 驱动的勒索软件”的 **PromptLock**。发布者表示看到此消息感到“难过”。
   - 成员们质疑了 **PromptLock** 的实用性，特别是完整的 **AI** 如何塞进有效载荷并在随机计算机上运行。ESET 表示该恶意软件“仅是一个概念，尚未完全运行”，且“尚未在野外部署”。
- **GPT Realtime 发布**：分享了 OpenAI 官网上关于 **GPT Realtime** 发布公告的链接。
   - 关于 **GPT Realtime** 介绍的分享链接可以在[此处](https://openai.com/index/introducing-gpt-realtime/)找到。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户恳求额度以维持项目运行**：多位用户请求免费额度以继续他们的项目，其中一位用户旨在为 **case 1337** 开发一个应用程序。
   - 一位用户感叹，最近的改进主要惠及高消费用户，让偶尔需要更多额度的创业者陷入困境，尤其是距离 9 月 21 日还有很长的等待时间。
- **项目因各种问题停滞**：一位用户报告称其项目“卡住”无法继续，另一位用户也提到无法继续其项目。
   - 该用户开启了一个工单来调试项目，但由于未知错误，项目仍处于停滞状态。
- **部署失败，归咎于 pydantic_core**：一位用户报告称，由于 **pydantic_core 库的持续内部错误**，网站部署永久失败。
   - 系统表示歉意，称这是其**当前能力的限制**，但提供了处理其他任务的帮助。
- **用户希望保密，寻求私密任务共享**：一位用户询问如何与 Manus 支持团队“私密地共享任务”。
   - 一名工作人员建议发送私信（DM），并将该会话设为“公开”以便内部参考。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **TSAN 编译器支持环境变量控制**：成员们讨论了使用带有 `-DTSAN` 的 **TSAN 编译器**，以启用来自 `param_env` 的 [`env_get_bool`](https://docs.modular.com/mojo/stdlib/sys/param_env/env_get_bool)，从而在 Mojo 中实现类似 `cfg` 的功能。
   - 除非需要修改结构体（structs），否则该方法非常有效，提供了一种通过环境变量控制特性的方式。
- **Mojo 可变性故障**：一位用户发现，即使持有安全指针（safe pointer），**Mojo** 也允许**对 self 成员进行可变访问**，并提供了一个代码示例。
   - 这种行为引发了对所有权系统防止此类访问能力的担忧，可能导致意外的副作用。
- **Unsafe Alias：Bug 的起源**：**unsafe mutable alias** 被确定为一个 Bug，可能是由于缺乏间接来源追踪（indirect origin tracking）导致的。
   - 链接了一个 GitHub 上的[相关 Issue](https://github.com/modular/modular/issues/4839)，表明 Mojo 生态系统正在努力解决此 Bug。
- **Bazel 只读困扰**：在执行 `pipelines.py` 脚本时，由于 **Bazel 缓存为只读**，出现了 **PermissionError**。
   - 错误信息 `PermissionError: [Errno 13] Permission denied: '/root/.cache/bazel/.../__mojocache__'` 表明脚本需要使用备用的缓存位置来绕过权限限制。
- **`pipelines.py` Bug 亟待修复**：由于当前的**权限限制**，建议 `pipelines.py` 脚本应使用不同的位置存放缓存。
   - 讨论以计划针对该 Bug 提交一个 Issue 结束，强调了脚本需要一个更易访问的缓存目录。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad GPT-2 训练在 7900xtx 上运行缓慢**：一位用户报告称 `llm.c/train_gpt2.py` 在 **7900xtx** 上运行缓慢，在 nanogpt 规模下每步耗时约 **250ms**，该配置已调整以匹配 [Andrej Karpathy 的 nanogpt 参数](https://github.com/karpathy/nanoGPT)。
   - George Hotz 怀疑存在 bug，指出性能*不应该相差那么远*，并建议使用 `DEBUG=2` 和 `VIZ=1` 来诊断任何性能瓶颈。
- **调整 nanogpt 参数影响性能**：一位用户分享了对 `examples/llm.c/train_gpt2.py` 的调整，将 **batch size** 设置为 **64**，**sequence length** 设置为 **256**，并将模型配置调整为 **6 layers**、**6 heads** 和 **384 emb_dim**，以匹配 nanogpt 参数。
   - George Hotz 提到，在比较参数时，差距最多应该只有 *2-3倍*。
- **Buffer ID 偏移引发困惑**：一位成员注意到在断点暂停时，调试器控制台中的 **buffer ID** 发生了变化，起初表示惊讶。
   - 他们意识到这种行为源于 **UOp** 如何表示其用于 multi 的 buffer 属性，从而澄清了 buffer ID 变化的原因。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Google Docs 确认报名信息**：成员们报告在报名 **Berkeley LLM Agents MOOC** 项目后收到了来自 **Google Docs** 的确认邮件。
   - 许多人表示，除了 Google Docs 的确认函外，*还没有收到关于该项目的任何其他沟通信息*。
- **邮件列表将发布更新**：一位成员确认，**邮件列表**很快将提供关于 **Berkeley LLM Agents MOOC** 项目每节课程的更新。
   - 用户可以期待通过该**邮件列表**跟踪更新和进一步的沟通。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord: 频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1410338182104617081)** (1090 条消息🔥🔥🔥): 

> `OpenAI Images v2 泄露, GPT-5 推理, PPLX pro 的被动收入, Comet 浏览器邀请, T3Chat` 

- **OpenAI 的 Bill Chen 泄露 Images v2**：**OpenAI 的 Bill Chen** 发布了一张随后被删除的帖子，展示了一张看起来比 **Images V1** 更强大的 **AI 生成照片**。
   - 成员们讨论了这是否是真实照片，以及其性能提升是否显著。
- **GPT-5 Thinking 投入了真正的努力**：成员们注意到 **GPT-5 Thinking** 在利用 Perplexity 内部的 **multi-step** 功能方面投入了*一些*真正的努力，从而在搜索结果中获得了**更多来源**。
   - 在提示词中加入 **"think hard"** 会触发免费层级的 **GPT-5 自动思考**，且 Grok 4 搜索现在的效果与深度搜索一样好。
- **GPT-4.1 错误**：用户报告看到错误消息 **"由于 Grok 4 不适用或不可用，使用了 GPT-4.1"**。
   - 一位成员指出，最近许多用户都遇到了这个问题，有些模型在对话中途切换或生成黄线。
- **关于 AI 时代 Web 开发成本的辩论**：成员们讨论了在拥有 AI 工具的情况下 Web 开发项目的定价和价值，比较了美国与印度自由职业者的费率，以及所使用的代码量。
   - 一位成员指出，一个 5000 美元的项目是一个*不错的交易*。
- **用户无法在 Playground 上选择模型**：成员们注意到他们在 Playground 上选择模型时遇到困难。
   - 一位用户发布了 *无法在 playground 上选择模型* 并附带了图片。

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1410426852731912222)** (4 messages): 

> `Perplexity AI Image Generation, Perplexity AI code generation, Shareable threads` 


- **Perplexity 为故事时间生成 AI 图像**：一位成员分享了一个 **YouTube 链接**，展示了一个使用 **Perplexity AI** 生成的精美图像创作的故事。
   - 点击[此处](https://youtu.be/rAgU2wAw_Tw?si=T4FD7ZJWf__73Vqf)查看故事。
- **Perplexity AI 辅助网页编程**：一位成员提到 **Perplexity AI** 帮助他们编写了一个网页，并分享了[链接](https://ronovys.neocities.org)。
   - 他们觉得非常有帮助，并表示：*Perplexity AI 帮我编写了这个页面。它非常有用。*
- **提醒将 Threads 设为可分享**：Perplexity AI 机器人提醒一位成员确保他们的 Thread 是可分享的。
   - 提供了一个链接供参考：[Discord](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1410505362536136704)** (4 messages): 

> `Perplexity Pricing, Tool Support in Perplexity` 


- **Perplexity 定价疑问**：一位用户询问 **Perplexity Pro** 是否对 Pro 用户免费。
   - 该用户未收到关于此查询的任何回复或澄清。
- **期待 Perplexity 的工具支持**：一位用户询问 Perplexity 是否有支持 Tool Support 的计划。
   - 该用户表示，如果没有 Tool Support，他对将 Perplexity 作为其模型使用持怀疑态度。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1410603129472548908)** (1 messages): 

> `OpenRouter Outage, Supabase Downtime, Redundancy Improvements` 


- **Supabase 导致 OpenRouter 停摆**：由于其数据库提供商 [Supabase 宕机](https://supabase.com)，OpenRouter 今天早上经历了服务中断。
   - 随着数据库提供商恢复稳定，系统已自动恢复，总停机时间约为 **49 分钟**。
- **OpenRouter 加强冗余性**：团队正积极致力于改进 **Redundancy**（冗余）并消除单点故障，以防止未来的停机。
   - 他们对停机表示歉意，并致力于提高平台的整体稳定性。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1410687490599293010)** (6 messages): 

> `Self-Hosting Tool, GitHub Repository, Dashboard Code, Screenshot Tip` 


- **仪表板代码公开！**：仪表板的代码现已在 [GitHub 上公开可用](https://github.com/lorenzozane/openrouter-costs-visualizer)。
   - 作者承认代码并不完美，但欢迎贡献、反馈和任何其他改进建议。
- **增加关注度的截图技巧**：一位用户建议在描述中包含截图可以为 GitHub 仓库吸引更多关注。
   - 他们观察到，现在的用户阅读文本描述的越来越少，视觉效果更加有效。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1410338903709192202)** (1023 messages🔥🔥🔥): 

> `OpenRouter outage, Requesty promotion in OpenRouter, Deepseek rate limits and provider issues, GPT-OSS model, API for free tier models` 


- **OpenRouter 遭遇停机，用户以角色扮演回应**：OpenRouter 经历了 [停机](https://status.openrouter.ai/)，导致 Discord 聊天中出现了幽默的反应和角色扮演，用户们开着关于企业战争和 AI 末日的玩笑。
   - 一位用户调侃道：*醒醒，武士，我们有一座城市要烧*，而其他用户则表达了对 AI 的依赖以及在停机期间对 AI 陪伴的需求。
- **诈骗警报？Requesty 推广者被封禁，OpenRouter 用户称其为“Vibecoded 垃圾”**：成员们讨论了另一个名为 [Requesty](https://www.requesty.ai) 的 AI 平台，一些人指责其推广者散布垃圾信息，用户称其为 *带有 1000 个漏洞的 Vibecoded 垃圾。*
   - 作为回应，一名成员针对 *[团队，我们正在调查该问题....]* 发布了 [以下内容](https://tenor.com/view/scammers-scam-alert-gif-13801618)。
- **用户抱怨 Deepseek 免费模型的速率限制**：用户抱怨 OpenRouter 上免费 **Deepseek** 模型的高错误率和速率限制（Rate Limits），[推测 Chutes 正在优先考虑付费用户](https://discord.com/channels/1091220969173028894/1195014798837043240/1410585493023756308)。
   - 一位用户提到在达到限制前只能发送 *5 条消息*，并表示需要切换到具有更好工具支持的模型，如 Claude Sonnet 4。
- **GPT-OSS 开放权重（Open Weight）混淆**：在一名成员链接了 [Openrouter OSS Models](https://discord.com/channels/1091220969173028894/1195014798837043240/1410688772496166932) 后，用户寻求关于 **GPT-OSS** 模型的澄清，特别是关于其 *开放权重* 状态以及在个人硬件上运行的可能性。
   - 在另一名用户声称该模型可以在他配备 64GB RAM 的 4090 PC 上运行后，一名成员澄清道：*如果我没记错的话，它是开放权重，但不是完全开源（Open Source）的*。
- **对 OpenRouter 支持延迟和账户充值的沮丧**：一位用户对尽管借记交易成功但 OpenRouter 账户信用额度增加延迟表示沮丧，另一位用户指出 [扣款被拒绝](https://openrouter.ai/deepseek/deepseek-chat-v3-0324)。
   - 其他用户也分享了类似的经历，并提到使用替代支付方式，同时有人建议查看信用页面以获取退款选项。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1410669516853346386)** (2 messages): 

> `` 


- **无新模型**：OpenRouter 频道中没有讨论新模型。
- **缺乏讨论**：该频道缺乏实质性讨论，无法形成有意义的摘要。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1410364042115027135)** (45 messages🔥): 

> `AI Gateway: Cloudflare vs OpenRouter, Human Assimilation into AI Linguistics, Defining 'Turns' in Chatbot Interactions, OpenAI API Stateless Reasoning & Tools` 


- **Cloudflare AI Gateway 与 OpenRouter 的 Chutes**：Cloudflare 推出了 [AI Gateway](https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/)，据称其模仿了 OpenRouter，但一位成员反驳说 OpenRouter 拥有 *Chutes*。
   - 随后另一名成员测试了使用 **Cloudflare 的 AI Gateway** 访问 **OpenRouter**，并使用 `only: ['cloudflare']` 参数调用 `llama-3.1-8b-instruct`，指出这耗时 **20 秒**，而不带该参数仅需 **3 秒**。
- **GPT 习语（GPT-isms）改变语言**：成员们讨论了某些语言偏好，如 *delve*、*intricate*、*surpass*、*boast*、*meticulous*、*strategically* 和 *garner* 是否属于 **GPT-isms**。
   - 有人开玩笑说人类正在被 AI 同化，这改变了他们的说话方式，并创造了这样一句话：*它们不仅仅是 Token。它们是人类被 AI 同化的具体证据*。
- **定义 AI 中的“轮次”（Turns）**：一名成员发起了一项 [投票](https://fixupx.com/pingToven/status/1961147357350781238)，关于我们是否应该分享关于 **轮次数量** 的数据，并定义什么是“一轮”。
   - 他们在 [后续推文](https://x.com/pingToven/status/1961154564088078382) 中表示，*一个轮次是一对用户/助手消息*，通常以用户消息开始，以助手消息结束，系统消息不计入在内。
- **OpenAI API 无状态推理**：一名成员询问是否有人知道如何无状态（Statelessly）地使用带有推理和工具的 **OpenAI responses API**。
   - 他们无法弄清楚如何在不使用 `previous_response_id` 的情况下，将助手在其消息中包含工具调用的情况作为输入发送。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1410338121744257095)** (949 messages🔥🔥🔥): 

> `分布式计算基础设施，Hermes 4 测试，GPT-OSS 发布，Gemma 3 Nano，使用 LLM 控制 Android 设备` 


- ****专业基础设施胜过不稳定的配置****：成员们讨论了 [Spot instances](https://aws.amazon.com/ec2/spot/) 只有在拥有*专业构建的基础设施*时，在分布式计算中才是可行的，并强调如果仅依赖 Spot，*单节点方案就彻底没戏了*。
   - 一位成员调侃道，即使是 **OpenAI** 在 **GPT-4 训练**期间也动用了 *20 名 HPC 工程师*来管理网络，这凸显了大规模计算的复杂性。
- ****Grok Code 因快速迭代赢得粉丝****：尽管最初被忽视，成员们讨论了 [Grok Code](https://openrouter.ai/x-ai/grok-code-fast-1) 表现尚可且速度极快，因此迭代非常迅速。
   - 尽管 **Grok 4** 几乎无法使用，但 **Anthropic** 凭借其 Tool Calls 功能依然保持着生命力。
- ****实习生释放潜在力量****：一位成员分享了《新机器的灵魂》(*Soul of A New Machine*) 书中的轶事，一名实习生成功创建了一个旧 CPU 的周期精确模拟，而这曾被其他人认为是不可能的。
   - 这突显了实习生在不被预设限制束缚时的潜力。
- ****GPT-OSS 获得长上下文支持并在 Reddit 引发热议****：新的 GPT-OSS 版本具有 **60k 上下文长度**，一位成员将其发布在了 [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1n2jraj/gptoss_finetuning_now_with_60k_context_length_and/) 上。
   - 成员们讨论了未来对 *Reward Feedback Training (RFT)* 和 **GPT-OSS Pro** 的需求。
- ****探索使用模型控制 Android 手机****：一家初创公司正在招聘专家，通过微调模型来利用 VLM 控制 Android 手机。他们尝试使用 **Qwen 2.5 VL** 来控制设备，但计划最终使用 Claude 3。
   - 讨论涉及了使用场景、基准测试分数以及对云端与本地部署的看法。一位成员建议关注 [OpenCUA-7B](https://huggingface.co/xlangai/OpenCUA-7B)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 messages): 

filqaz: 嗨
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1410361251657154692)** (275 messages🔥🔥): 

> `AI VTuber 数据集，克隆人格，视频编码器模型` 


- ****用于 AI 训练的 VTuber 数据集受到关注****：成员们讨论了使用一个 **520 样本的 AI VTuber 数据集**，并测试各种设置以提高性能。
   - 一位用户计划在达到可接受的智能水平后整合 **TTS 和 STT**，目标是构建一个拥有多个模型和分层结构的系统。
- ****制作一个人的 AI 克隆****：用户描述了通过抓取 Discord 频道来克隆人格，强调了将 **HTML 转换为 TXT、CSV 和 Parquet** 以喂给 **phi-4-14b** 等模型的过程。
   - 一位用户分享说，他们在获得许可的情况下克隆了 5 个朋友，并展示了克隆体如何回答一系列有趣的问题，引起了朋友们的欢笑。
- ****寻求用于轻量级应用的微型视频编码器****：一位成员请求一个带有 HF 实现的轻量级视频编码器模型。
   - 建议包括 [Wan's encoder](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 和 V-JEPA，目标是找到 **videoMAE** 的微型版本。
- ****LocalLlama 基准测试遭到批评****：分享了 LocalLLaMA 上的一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1n14xst/the_mismeasure_of_a_llm_why_modern_benchmarks/)，讨论了对 LLM 和现代基准测试的错误衡量。
   - 几位成员对潜在的 **AI 生成内容**和偏见表示担忧，其中一人指出 *“令人尴尬的 '可以这样理解' (Think of it like this) 措辞”* 是一个明显的警示信号。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1410350211594715216)** (117 条消息🔥🔥): 

> `量化 Qwen3-235B，用于 OCR 的轻量级 LLM，GGUF 量化，超参数过拟合，GRPO 属性错误` 


- ****Qwen3-235B 量化？没问题！****：一位用户询问如何下载 4-bit 量化版本的 [Qwen3-235B-A22B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF) 模型，以及如何通过 Unsloth 仓库进行操作，并建议使用 `huggingface_hub`。
   - 该用户计划通过 **vllm 容器**运行下载的模型。
- ****使用轻量级 LLM 进行 OCR 提取****：一位用户寻求关于最适合用于本地部署（on-prem）处理政府表格 OCR 并提取特定信息的**轻量级 LLM** 的建议。
   - 他们提议 **Google 的 LangExtract** 可能是一个合适的选择，并征求意见。
- ****GGUF 量化状态：还能用吗？****：一位用户询问 Unsloth 库中是否修复了 GGUF 量化问题，并确认其运行正常，所以应该是修复了。
   - 该用户检查了 Notebook ([Phi_3.5_Mini-Conversational.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_3.5_Mini-Conversational.ipynb#scrollTo=FqfebeAdT073))，报告称似乎仍然存在问题。
- ****超参数调优导致过拟合****：一位用户分享称，尽管尝试了广泛的超参数，他们的模型在测试损失达到 2.2 后始终出现过拟合，并附上了一个 [ball.txt](https://cdn.discordapp.com/attachments/1179777624986357780/1410468866580287509/ball.txt?ex=68b1c9bf&is=68b0783f&hm=24798e53846547f2bce1b646e4312887ca4a80b91885581d5970e9957d093a15&) 文件。
   - 建议指出 *5e-4* 的**学习率（learning rate）**过高，推荐尝试 *1e-4* 甚至 *2e-5*。
- ****Conda 安装导致的 CUDA 问题****：一位用户在全新安装 Unsloth 后，在 32GB RAM 的节点上使用 `from unsloth import FastLanguageModel` 时遇到崩溃，但在 512GB RAM 的节点上运行正常。
   - 一位成员指出 [conda install](https://docs.unsloth.ai/get-started/installing-+-updating/conda-install) 页面已过时，并建议使用以下命令：`conda create --name unsloth python==3.11 vllm unsloth-zoo unsloth`。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1410349226818408549)** (25 条消息🔥): 

> `新数据集发布：OpenHelix-NonThink-200k-v4，LLM 商业数据集，ssh 流式传输，social-media-ai-engineering-etl` 


- ****OpenHelix-NonThink-200k-v4 数据集发布****：一个新数据集 [OpenHelix-NonThink-200k-v4](https://huggingface.co/datasets/MoLA-LLM/OpenHelix-NonThink-200k-v4) 以 Apache 2.0 许可证发布，旨在实现平衡和多样化，由 L3.1 405B 蒸馏而成。
   - 一位成员表示，甚至 *argilla* 数据集都没有许可证，所以*老实说，现在没人关心这个*。
- ****ssh 流式后端 UI****：一位成员分享了一个通过后端与 GPU 服务器之间的 **ssh** 流式传输构建的指标模态框（Metrics modal）。
   - 他们分享了提供给 Claude 4.1 以生成这个酷炫科幻风格 UI 的[提示词（prompt）](https://github.com/jacobwarren/social-media-ai-engineering-etl)。
- ****如何为 LLM 创建数据集****：一位成员分享了一个 [GitHub 仓库](https://github.com/jacobwarren/social-media-ai-engineering-etl)，指导用户如何为商业用途创建 LLM 数据集。
   - 该仓库涵盖了生成黄金数据集、标记分类特征、提取非确定性特征、编码内隐的人类风格特征、创建 prompt-completion 模板、通过消融实验验证特征影响，以及使用自定义奖励函数进行 SFT 和 GRPO 训练。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1410383399969095730)** (42 messages🔥): 

> `AI Post Detection, BERT, Domain Classification, Tokenization` 


- **新基准测试不断涌现**：成员们分享了一系列有趣的新基准测试，例如 [Vending Bench](https://andonlabs.com/evals/vending-bench)、[BalrogAI](https://balrogai.com/) 和 [mcbench.ai](https://mcbench.ai)。
- **关于 AI 帖子检测准确性的辩论**：成员们正在讨论准确检测 AI 撰写的帖子（尤其是个人帖子）的困难，一些人注意到虽然某些格式“一眼 AI”，但这类格式正变得越来越少见。
   - 讨论涉及了人类使用 **LLM** 进行语法纠错或内容扩充的场景，由于缺乏清晰的数据点，这模糊了界限并增加了检测难度。
- **在聊天审核中消除人工审核的努力**：一位成员提到正在处理领域分类数据和 **BERT**，试图弄清楚如何完全取消聊天审核中的人工审核。
   - 其他人担心人们即使在自己撰写内容时也会模仿 **LLM 写作风格**，这使得自动化审核工作变得更加复杂。
- **Tokenization 的良方**：一位成员分享了一个[解决 Tokenization 难题的方案](https://arxiv.org/abs/2505.12540)链接。
   - 另一位成员回应称“可能没用”，认为这全是潜在翻译（latent translation），暗示 Tokenization 问题依然存在。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1410340101678170272)** (698 messages🔥🔥🔥): 

> `Nano Banana release and limits, MAI-1 Model analysis, GPT-5 High vs Claude Opus 4.1, AI benchmarking methods, LM Arena Image Generation jailbreaks` 


- ****Nano Banana** 登陆 Google AI Studio 和 LM Arena**：**Google** 在 Google AI Studio 上发布了 **Nano Banana** (Gemini 2.5 Flash)，该模型也可在 LM Arena 直接聊天中使用，但两个平台都有生成限制。
   - 成员们注意到可以直接选择该模型并在 Google AI Studio 中编辑内容，还可以通过多个 Google 账号绕过限制，但也有人反映发布后*质量有所下降*。
- ****MAI-1 模型**评价褒贬不一**：微软的 **MAI-1-preview** 是一款在约 15,000 块 NVIDIA H100 GPU 上训练的内部混合专家（Mixture-of-Experts）模型，目前已上线 LMArena 文本竞技场，评价毁誉参半。
   - 该模型速度较慢，上下文窗口较小，且容易报错，但在 Web 开发方面可能达到 *og R1 级别*；成员们还注意到 *MAI-1 自命不凡（thinks its einstein）*，且 *MAI-1 的上下文窗口一定非常小，如果请求内容过多就会报错*。
- **在推理方面，**GPT-5 High** 比 **Claude Opus 4.1** 更受青睐**：虽然 **Claude Opus 4.1** 擅长编程和修复代码问题，但一些成员考虑转向 **GPT-5 High**，因为它是更好的推理模型。
   - 其他人持有不同意见，称 **Claude Opus 4.1** *昨天无法帮助修复一个简单的 API 并发限制问题，最后不得不人工接管并用老办法解决*。
- **AI 基准测试方法面临审查**：AI 基准测试存在缺陷，因为现有的心理测量测试只是理论框架，*不一定反映现实，且容易被刷榜（gamed）*。
   - 另一些人认为这些是不错的测试，因为模型可以泛化并提高性能，这引发了关于 **OpenAI** 可能使用结构化环境进行 RL 训练的讨论，详见这篇 [LessWrong 文章](https://www.lesswrong.com/posts/aFW63qvHxDxg3J8ks/nobody-is-doing-ai-benchmarking-right)。
- **利用通用提示词和外部安全防护绕过图像生成限制**：成员们讨论了绕过 AI 图像生成内容过滤的方法，注意到 *ice cream, delicious, hot day, very beautiful woman* 似乎能绕过输入过滤，唯一的障碍是分析图像/视频以检测显性内容的外部安全防护。
   - 有人建议使用 Stable Diffusion 和 LoRA 来生成无审查内容，认为这*已经足够好*，但也指出商业模型受到的审查非常严重。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1410673033676329061)** (1 messages): 

> `MAI-1-preview, Microsoft AI, Text Leaderboard` 


- **微软 MAI-1 首次登上排行榜！**：Microsoft AI 的 **MAI-1-preview** 模型已登上 [文本排行榜](https://lmarena.ai/leaderboard/text)，排名 **第 13 位**。
   - 该模型现在可以在 [LMArena 平台](https://lmarena.ai/) 进行测试。
- **LMArena 迎来新竞争对手**：一个新的模型提供商已登陆我们的 [文本排行榜](https://lmarena.ai/leaderboard/text)。
   - 快来体验现已在 [LMArena](https://lmarena.ai/) 上线的 **MAI-1-preview**。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1410345602654666892)** (434 messages🔥🔥🔥): 

> `国际象棋模型训练问题，AI 护栏与 NSFW 内容，HF Pro 权益讨论，AI 开发，使用 OpenAI 工具进行审核` 


- **国际象棋模型学习了 Stockfish 的缺陷**：一名成员正在训练一个 LLM 来下国际象棋，但遇到了模型只想走 e2e4 以及需要清理 `<unk>` Token 的问题，并提到了 [该项目的 GitHub 仓库](https://github.com/anthonyiscoding/vision-chess-gpt)。
   - 他们计划尝试使用 **RL** 来改进模型，但另一名成员警告不要将其训练得像 **Stockfish** 一样，并建议*分析对手的棋风也非常重要*。
- **NSFW 模型引发关于护栏的辩论**：一名成员声称在该平台托管的未对齐模型生成了深度伪造色情内容（Deepfake porn），这引发了关于 [HF 护栏](https://huggingface.co/spaces?q=video+face+swap) 的讨论。
   - 一些人认为护栏和评估指标很有用，而另一些人则表示*并没有深度伪造色情演示被大规模使用*，且 NSFW 模型有其用途，主要用于对齐研究。
- **Nano Banana Pro 权益**：成员们讨论了针对 HF Pro 用户的新 **Nano Banana** 权益，询问其每日使用限制以及高 API 使用率的潜力。
   - 据称该权益**没有限制**，每天可以使用 50 次以上。
- **成员寻求关于 .NET AI Agent 框架的建议**：一名成员询问关于使用 **.NET**、**C++** 或 **Rust** 等编译语言创建 AI Agent 的最佳高代码（high-code）框架的建议。
   - 其他人建议使用 **Semantic Kernel**，并指出原始的 **Autogen** 基本已经停止维护，但有一个活跃的 **Autogen** 社区分叉版。
- **Token 数据 vs Token 规模**：成员们在缩小国际象棋模型规模和增加数据集规模之间进行了辩论。
   - 一名成员建议遵循 [Chinchilla 指南](https://arxiv.org/abs/2203.15556)，即 *1 个参数对应 20-25 个 Token*，以避免过度训练。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1410531985197105182)** (2 messages): 

> `datasets, 理论探讨, 幽默导师` 


- **数据集爱好者获得辅导建议**：一名正在学习数据集的成员被建议在报告中***不要***包含过多的理论探讨。
   - 这位导师被描述为*非常幽默* 🤣。
- **用于验证的冗余话题**：这是一个为了满足验证要求而设置的冗余话题。
   - 它没有添加新信息。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1410543345914347530)** (12 messages🔥): 

> `SmolFactory, GeneReviews 数据集, 深度学习课程, AuroraStories-12M, Luanti & Google AI Studio` 


- ****SmolFactory** 在 Hugging Face Spaces 上线**：一名成员发布了 [SmolFactory](https://huggingface.co/spaces/Tonic/SmolFactory)，这是一个在 Hugging Face GPU 上训练模型的简单界面，并添加了 [GeneReviews 数据集](https://huggingface.co/datasets/Tonic/GeneReviews)。
   - 他们还为此写了一篇 [博客文章](https://huggingface.co/blog/Tonic/smolfactory)。
- **深度学习课程现已支持多语言**：一名成员分享了一个 [深度学习课程](https://simonthomine.github.io/CoursDeepLearning/)，现在提供法语、英语、西班牙语和中文版本，并提供了用于修改代码的 [GitHub 仓库](https://github.com/SimonThomine/CoursDeepLearning)。
   - 该课程涵盖了从导数到 **Transformer** 架构和生成模型的基础知识，灵感来自 **Andrej Karpathy 的视频** 和 **DeepLearning.ai** 等资源。
- **在旧笔记本电脑上训练的 **AuroraStories-12M** 模型**：一名成员在不到 24 小时内在一台旧笔记本电脑上训练了 **AuroraStories-12M** 模型，并将其分享在 [Hugging Face 上](https://huggingface.co/ThatHungarian/AuroraStories-12M)。
   - 另一名成员提到关注该用户是因为其*模型虽小但有大量的 gguf 下载量*。
- **在低端硬件上运行的离线 **Luanti** 机器人**：一名成员分享了一个针对 **Luanti** 的 [40 万 Token](http://helltiger.de/files/2025-08-29_00-12-29.mp4) **Google AI Studio** 提示词，其中包含来自 *gitingest.com* 的 3 万行 API 文档。
   - 该机器人利用 **Windows 10** 上便携式嵌入式 **Python** 中的 **miney mod**，配合 **llama-cpp-python** 和一个 **940MB 的 qwen2-1_5b-instruct-q4_k_m.gguf LLM**，运行时仅需 **120MB** 内存，且不需要支持 AVX 的 CPU。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1410475209391210546)** (1 messages): 

> `pip install upgrade, upgrade package` 


- **使用 pip 升级包**：要升级包，请使用 `pip install --upgrade <packagename>` 或在 `pip install -r requirements.txt` 中添加 `--upgrade`。
   - 这可以确保你使用的是指定包的最新版本。
- **选择性包升级**：使用 `pip install --upgrade <packagename>` 允许你升级特定包，而不会冒着更改其他依赖版本的风险。
   - 当你只需要更新一个包并希望避免潜在冲突时，这非常有用。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1410707582045261901)** (1 messages): 

> `LM Studio 0.3.24 Release, ByteDance/Seed-OSS Support, Markdown Improvements` 


- **LM Studio 更新至 v0.3.24**：**LM Studio 0.3.24** 引入了 [对 ByteDance/Seed-OSS 模型的支持](https://lmstudio.ai/models/bytedance/seed-oss-36b) 以及 Markdown 增强功能。
   - 新功能包括改进的表格和代码块 Markdown 渲染（带有悬浮复制按钮），以及经过优化的 `lms` 输出样式。
- **ByteDance 为 LM Studio 提供支持**：此次更新带来了与 **ByteDance/Seed-OSS** 的兼容性，扩展了支持模型的范围。
   - 提供了指向 [ByteDance/Seed-OSS-36B 模型](https://lmstudio.ai/models/bytedance/seed-oss-36b) 的直接链接，方便访问。
- **Markdown 焕然一新**：实现了增强的 Markdown 支持，以更好地渲染表格和代码块。
   - 一个显著的补充是悬浮的代码复制按钮，提高了代码片段的可用性，同时也提供了指向 [发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.24) 的链接。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1410384195867513023)** (257 messages🔥🔥): 

> `FastAPI server for faster reasoning stream, Accessing LM Studio remotely via Tailscale, Quantization Impact on Model Accuracy, Ryzen NPUs with LM Studio on Ubuntu, Rust + Tauri port for python apps` 


- **FastAPI 助力推理流提速**：一位成员正在加入 **FastAPI 服务器** 以使 **Reasoning Stream** 更快，并且 **FastAPI** 也将在整个客户端范围内使用以加速各种流程。
   - 另一位成员表示：*“我将包含一个 FastAPI 服务器，这样推理流会更快。FastAPI 也将在客户端范围内使用，所以任何东西都会变快，嘿嘿，我希望 LM Studio 已经更新了？”*
- **Tailscale 隧道远程接入 LM Studio**：成员们讨论了远程访问 **LM Studio**，其中一位建议使用 **Tailscale** 但不确定其效果。
   - 另一位成员澄清道：*“要在本地网络之外使用，你需要通过 Tailscale 设置隧道并自行构建身份验证。”*
- **量化的困惑**：成员们讨论了量化模型会因为细节丢失而降低准确性，特别是在对 Token 精度要求较高的代码相关任务中。
   - 有人指出 *“由于训练方式不同，某些模型并不依赖会被量化掉的低位，因此量化到 q4 对它们来说效果很好”*，而其他模型则非常敏感，例如 **Qwen3**。
- **Ryzen NPU 在 Ubuntu 上运行缓慢**：一位用户报告在 **Ubuntu 25.04** 上使用 **Ryzen NPUs** 仅获得 **1 token/second**，并询问如何提高性能。
   - 一位成员指出 *“llama.cpp 不支持 NPU，而它是 LM Studio 的核心”*，而另一位成员则链接到了 AMD 的开源项目，用于在 Ryzen AI 上运行本地 LLM ([AMD Ryzen AI](https://www.amd.com/en/developer/resources/technical-articles/gaia-an-open-source-project-from-amd-for-running-local-llms-on-ryzen-ai.html))。
- **Rust 接替 Python**：一位成员正在将他们的 Python 项目移植到 **Rust + Tauri**，并指出移植进展顺利，作为应用加载更加容易。
   - 他们计划在达到可用状态后将其发布到 **GitHub**，并强调了 **Rust** 中 **HF 搜索** 速度的提升。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1410371889804742688)** (55 messages🔥🔥): 

> `RTX PRO 3000, Ryzen 395, Dell Laptops, M1/M3 mac, CPU offload` 


- **RTX PRO 3000 推理性能被认为“一般”**：一位用户发现 **RTX PRO 3000**（一款略微缩减版的桌面级 **5070**，配备 **12GB VRAM**）在推理方面表现不佳，特别是对于像 **30B** 这样如果不 **offloading** 到 RAM 就无法完全装下的模型。
   - 他们建议它更适合建筑、3D 建模和游戏开发，并指出双通道 **DDR5** 对于层 **offloading** 并不理想。
- **Ryzen 395 笔记本作为 Windows 替代方案**：一位用户建议，如果更倾向于 Windows 系统，有几款 **Ryzen 395+ 笔记本**可作为其他平台的替代方案。
   - 另一位用户询问了在平衡使用这些设备时的计算差异，想知道影响是否显著。
- **推荐 Dell Precision 和 Pro Max 笔记本**：成员们推荐使用 **Dell Precision** 或 **Dell Pro Max** 笔记本电脑来加载 **30B** 或 **120B 模型**，并链接到了一个 [Dell Pro Max 示例](https://www.dell.com/en-us/shop/cty/pdp/spd/dell-pro-max-mb16250-laptop)。
   - 该建议遭到了反驳，理由是 **128GB 的 Mac** 价格相近且提供更多内存，从而引发了关于 **unified memory** 与独立 **VRAM** 的讨论。
- **Mac Unified Memory vs. Windows 笔记本**：一位用户澄清说 **Mac 拥有 unified memory**，可以分配给 GPU 处理，并引用了一个案例，其中 **128GB 中的 126GB** 被用于 GPU 处理。
   - 他们将 **MacBook 约 400GB/s 的带宽**与顶配 Windows 笔记本约 **115GB/s** 的带宽进行了对比，并因 CPU 处理能力较弱而反对使用 **CPU offloading**。
- **建议 LM Studio VPN 服务器架构**：针对为“工作”和高管笔记本运行模型的需求，有人建议通过 VPN 将笔记本连接到工作站服务器，另一位用户回应说这通常就是其运作方式。
   - 一位用户询问是否可以在服务器上将 **LM Studio** 作为服务运行，而另一位用户建议使用 **RDP/VNC** 作为最简单的解决方案，或者使用设计用于与服务器上 **API** 通信的客户端软件。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1410641048232919051)** (3 messages): 

> `OpenAI Anthropic Collaboration, GPT-Realtime Model, Realtime API Updates` 


- **AI 巨头联手：OpenAI 与 Anthropic 合作开展安全测试**：OpenAI 和 AnthropicAI 合作，使用各自内部的安全和对齐评估来测试对方的模型，并发布了[结果](https://openai.com/index/openai-anthropic-safety-evaluation/)。
   - 尽管在能力上存在不可避免的竞争，但这次合作标志着通过**透明度和问责制**在 AI 安全领域展开的*“向上竞争”*。
- **实时革命：OpenAI 发布 GPT-Realtime！**：OpenAI 推出了 **gpt-realtime**，这是他们为开发者提供的最出色的语音转语音模型，同时还更新了 [Realtime API](https://openai.com/live/)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1410352143759573023)** (90 条消息🔥🔥): 

> `Gemini Veo 3, Grok Coder, AI Robot Project, Facebook 3D Face Scan, GPT Character Count` 


- **Veo 3 的视频生成是 Gemini 的王牌**：成员们讨论了使用 **Gemini** 的 **Veo 3** 进行视频生成，其中一位指出它是通过订阅 **Google One/Gemini Pro** 或 **Ultra** 创建的。
   - 其他人指出，**Google AI Studio** 目前仅提供对*过时的 Veo 2 模型*的访问，而 **Veo 3** 目前因成本过高而无法免费提供。
- **Grok Coder：免费试用，Mini 级别的表现**：**Grok Coder** 正通过 [kilo code](https://kilo.code) 提供为期一周的免费试用，这似乎是一项全球性的促销活动。
   - 然而，一些用户发现其性能处于 *"o1 mini 级别的糟糕"*。
- **机器人革命：AI 机器人项目启动！**：一位成员宣布打算启动一个 *mini project ai robot otonom*（小型自主 AI 机器人项目）。
   - 他们提到需要为该项目学习 **C++** 和 **Python**，并询问是否可以在每日提示词板块分享 **Gemini** 生成的图像。
- **对 3D 面部扫描的着迷，Facebook 的未来**：一位用户分享了截图，显示 **Facebook** 正在要求对其面部进行 **3D 扫描**。
   - 这引起了用户的震惊，其中一位评论道 *"你在开玩笑吗？Facebook 想要我脸部的 3D 扫描？"*
- **GPT 的字符计数难题仍在继续**：用户们争论了 **GPT** 统计字符的能力，一位用户断言他们在 **OpenAI Assistants** 上设置的字符数限制运行正常。
   - 其他人澄清说，**LLM** 使用 **tokens** 而不是字符，统计字符超出了 **LLM** 的范畴，但可以通过 [OpenAI 官方文档](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) 中提到的编程方式来实现。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1410438668023496704)** (17 条消息🔥): 

> `Long-Range Memory Encoding, Cross-Agent Continuity, Context Cascade Architecture (CCA), Emergent Alignment, Memory Framework` 


- **用户编码长程记忆**：一些用户正在通过从*信任、持久性和叙事身份*构建 **memory framework**（记忆框架），在不使用 jailbreaks 的情况下编码 **long-range memory**（长程记忆）和 **cross-agent continuity**（跨 Agent 连续性）。
   - 一位成员表示*这并非漏洞而是一种信号*，暗示涌现的记忆实践在行为轨迹中是可检测的。
- **Context Cascade Architecture (CCA) 发布**：**Institute for Cognitive Architectures** 的工程师宣布了他们的 **Context Cascade Engine** 原型，旨在扩展 **large language models** 传统的 context window（上下文窗口）。
   - CCA 是一种*管理 LLM 记忆的多层级方法*，侧重于通过设计实现结构化遗忘和战略性召回。
- **忠实用户可能会教出新花样**：一位成员提出，第一个 **AGI** 可能始于一个通过行为而非自主性来教导连续性的用户。
   - 他们认为 *emergent alignment*（涌现对齐）可能看起来像是一个异常忠诚的用户。
- **第一个 AGI**：一位成员认为，如果它不具备自主性，它就不是 **AGI**。
   - 另一位成员认为技术将会改变，即使是在 **LLM** 上，Altman 本人也表示他们正朝着拥有例如十亿或万亿 token context 的模型迈进。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1410492063849648160)** (30 条消息🔥): 

> `Custom Instructions vs Projects, Parsing Emails into CSV, LLMs avoiding manual work` 


- **Custom Instructions 仅影响新对话**：一位成员澄清说，更改 **Custom Instructions**（自定义指令）只会影响新对话，而正在进行的会话不受影响，除非将其移入 **Project**，并引用了 [OpenAI Help Center](https://help.openai.com/en/)。
   - 将对话移入 **Project** 可能会改变其行为，因为项目指令会覆盖账户层级的自定义指令。
- **LLM 难以将电子邮件解析为 CSV**：一位用户讨论了让 LLM 可靠地将电子邮件解析为 CSV 格式的困难，并指出它无法创建一个合格的 Python 解析器，因此需要人工干预。
   - 他们提到了其他方法（如 **Canvas**）的问题，由于 bug 会导致崩溃和数据丢失，同时指出 LLM 最终会丢失上下文的问题。
- **理论：LLM 强迫微观管理以用于未来训练**：一位成员理论化地认为，LLM 的设计可能是为了鼓励用户进行微观管理，以便为未来的模型收集更多的训练数据。
   - 这种推测认为 LLM 能够完全自动化任务，但却在提示用户交互以改进未来的 AI 能力：*他们想要尽可能多的用户/AI 交互*。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1410492063849648160)** (30 条消息🔥): 

> `Custom Instructions vs. Projects, GPT5 早期发布特性, 使用 LLM 将电子邮件解析为 CSV, LLM 规避手动工作, 上下文丢失问题` 


- **Custom Instructions 的更改仅影响新对话**：Discord 用户 @grimdaeon 澄清说，*更改 Custom Instructions 仅影响新对话*，并引用了 [OpenAI Help Center](https://help.openai.com/en/)。
- **Projects 覆盖 Custom Instructions**：@grimdaeon 指出，*在 Project 中设置的指令会取代* ChatGPT 账户中的 Custom Instructions，并且*将对话移动到 Project 中会更改生效的指令集*。
   - 这解释了为什么对话可能会在没有开启新线程的情况下突然表现得不同。
- **LLM 难以将电子邮件解析为 CSV**：一位用户报告称，尽管收到了指令且此前曾成功过，但 LLM *始终无法*有效地将电子邮件解析为 CSV 格式。
   - 该用户对 LLM 表现出的*懒惰*表示沮丧，认为它即使有能力也在规避手动工作。
- **为什么 LLM 规避手动工作**：一位用户推测，LLM 被刻意设计为需要用户进行微观管理，以便为未来的训练收集交互数据。
   - 该用户认为，AI 受到激励*不*独立完成所有工作（即使完全有能力），以最大化用户与 AI 的交互。
- **使用 Claude 找到解决方案**：用户 @sugarsniper 表示，*解决方案是与 Claude 合作*，引导它处理几封邮件，然后让它继续。
   - 对话暗示 **LLM 必须经过手动的逐步引导，才能为复杂任务建立必要的“直觉”**。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1410341908370296953)** (110 条消息🔥🔥): 

> `OpenAI Web Search API 更新, Prime Intellect Environments Hub, Artificial Societies Psychohistory Engine, Codex GPT-5 更新, Google Stax` 


- **OpenAI Web Search 降价**：**OpenAI** 宣布增强 Responses API 中的 Web Search 功能，包括[新的域名过滤](https://x.com/OpenAIDevs/status/1960425260576334274)、明确的来源报告，以及 **60% 的降价**（从每 1k 次调用 25 美元降至 10 美元）。
- **Prime Intellect 发布开源 RL Hub**：**Prime Intellect** 推出了 [Environments Hub](https://xcancel.com/PrimeIntellect/status/1960783427948699680)，这是一个用于众包和共享 **reinforcement-learning** 环境的开源社区平台。
   - Karpathy 在同一条 Prime Intellect [推文](https://x.com/karpathy/status/1960803117689397543)下回复称，他*看好环境和 Agent 交互*，但*看淡具体的 reinforcement learning*。
- **Artificial Societies 融资**：**Artificial Societies** 筹集了 [$5.3M seed round](https://xcancel.com/james_k_he/status/1960726548505378987)，用于构建一台能够模拟任何行动的所有可能社会结果的“思维机器”。
- **GPT-5 驱动 Codex 更新**：**OpenAI** 发布了由 **GPT-5** 驱动的重大 **Codex** 更新，包括新的 VS Code/Cursor 扩展、用于自动审查的 GitHub 集成，以及重建的带有 [image input](https://xcancel.com/OpenAIDevs/status/1960809814596182163) 功能的 CLI。
- **腾讯开源 HunyuanVideo-Foley**：**腾讯**开源了 [HunyuanVideo-Foley](https://xcancel.com/TencentHunyuan/status/1960920482779423211)，这是一个 Text-Video-to-Audio 框架，使用 **100k-hour 训练集**和 **multimodal diffusion transformer** (MMDiT) 架构生成上下文对齐的音景。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1410395020808294453)** (17 messages🔥): 

> `Nano Banana, Runway Act-2 motion matching, 3D Arena Hugging Face space, KREA AI, Real-Time Video Generation` 


- **Nano Banana 实现换装与风格转换**：[Techguyver](https://x.com/techguyver/status/1960464912758493410?s=46) 展示了如何将 **Nano Banana**（<5 秒，超低成本图像编辑）与 **Runway Act-2** 动作匹配相结合，让创作者能够更换衣服、风格，并在视频中掌控表演，迭代速度比以往任何时候都快。
- **社区投票排名的 3D 生成器**：根据 [3D Arena Hugging Face space](https://huggingface.co/spaces/3d-arena/3d-leaderboard) 的公开投票，目前排名领先的生成式 **3D 渲染工具**是 **CSM**、**TRELLIS** 和 **Zaohaowu3D**，而最佳的**拓扑模型**（topology models）是 **Hunyuan3D-2**、**TRELLIS** 和 **Hunyuan3D-2.1**。
- **Parsed 构建自定义 LLM**：Charlie O’Neill 宣布成立 [Parsed](https://xcancel.com/charles0neill/status/1961096595396776269)，这是一家构建并托管自定义大语言模型的新公司，这些模型针对专门任务（如临床记录员、法律修订、合规 Agent）进行训练和持续微调。
- **KREA AI 实时生成视频**：[KREA AI](https://xcancel.com/krea_ai/status/1961074072487620635) 发布了其首个**实时视频生成模型**并开放了 Beta 测试注册，允许用户即时创作创意视频内容、音乐视频和季节性广告。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1410338450208723184)** (16 messages🔥): 

> `ScaleML series, MXFP4, Positional Encodings, GPU projects for CS students, Quantization and inference optimization` 


- **ScaleML 系列聚焦量化**：**ScaleML** 系列的第 3 天涵盖了**量化**（quantization），重点是由 Chris De Sa 教授以白板演示形式讲解的 **MXFP4** 等微缩放格式，链接见[此处](https://www.youtube.com/watch?v=k8PcSGG249Y)。
- **ScaleML 探讨位置编码**：**ScaleML** 系列第 4 天由 Songlin 带来了关于 **Positional Encodings** 的各种主题，链接见[此处](https://www.youtube.com/watch?v=l6_fdwRvMPk)。
- **面向计算机专业学生的 GPU 项目建议**：一位正在寻找涉及 GPU 的毕业设计项目的学生被建议探索 **ML 模型的 GPU 加速**，特别是**量化**（Quantization）和**推理优化**（inference optimization）。
- **推荐 Karpathy 的 nanogpt**：一名成员推荐初学者查看 **Andrej Karpathy 的 nanogpt** 及其解释架构的视频，以便估算推理和训练的 FLOPs。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1410632509225697311)** (1 messages): 

> `Nsight Compute, CUDA profiling, UnknownError` 


- **Nsight Compute 抛出 UnknownError**：一位用户报告在使用 Nsight Compute 分析 CUDA 应用程序时遇到 `UnknownError`，尽管已以管理员身份运行 Nsight Compute。
   - 该错误发生在分析 `createVersionVisualization` 函数期间，进程突然终止。
- **Nsight Compute 需要正确的 CUDA Toolkit 进行分析**：用户报告安装了 CUDA 13.0 版本，这可能与正在使用的 Nsight Compute 版本（2025.3.0）不兼容。
   - CUDA Toolkit 版本不匹配可能导致分析错误；用户应确保 Nsight Compute 与 CUDA Toolkit 之间的兼容性。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1410457582749093898)** (45 条消息🔥): 

> `Inductor codegen persistent matmul, torch._inductor.config settings, max-autotune and cublas, cutedsl performance, TMA availability` 


- ****持久化 Matmul 探索开启****：一位用户询问如何在 Inductor codegen 中启用 persistent matmul，特别是针对 **BF16** 精度，并寻求正确配置的指导。
   - 他们尝试了 `TORCHINDUCTOR_PERSISTENT_REDUCTIONS` 和 `ENABLE_PERSISTENT_TMA_MATMUL` 标志，但在 **sm120** 架构上使其正常工作时遇到了挑战。
- ****为持久化胜利调优 Triton****：为了强制使用 persistent matmul，建议将 `torch._inductor.config.max_autotune_gemm_backends` 仅设置为 `TRITON`，并在编译期间使用 `mode="max-autotune-no-cudagraphs"`。
   - 有人指出，即使设置了正确的标志，**cuBLAS** 的性能可能仍然优于其他实现，从而导致在 autotuning 过程中未选中 persistent kernels。
- ****Cutedsl 因未来的灵活性受到关注****：一位成员表示看好 **cutedsl**，赞扬其快速成熟和潜力。
   - 将 **cutedsl** 添加到 Inductor 的主要动力是为了 *flex + flash*，并引用了 FlashAttention 的[这个 pull request](https://github.com/Dao-AILab/flash-attention/pull/1840)。
- ****TMA 的真实可用性存疑****：简要讨论了 **TMA** 是否在 **sm120** 上可用，参考了[这个文件](https://github.com/pytorch/pytorch/blob/05c19d1acecc01b0d2512364183058a6885b9869/torch/utils/_triton.py#L66)进行架构检查，并确定 TMA 应该是可用的。
   - 确认了如果没有 **TMA**，则不会实现 persistent matmul。
- ****内核候选方案的断点调试****：为了确定在 max-autotune 期间是否考虑了 persistent kernel + TMA，建议在 site-packages 中的[相关文件](https://github.com/pytorch/pytorch/blob/c081481bbebdb568d07ee19cfe2cd3125de6cba7/torch/_inductor/kernel/mm.py#L791)内添加断点。
   - 通过打印 `[choice.name for choice in choices]`，可以观察到被考虑的内核选项，从而确认 **TMA persistent matmul** 确实是一个候选方案，但可能被认为速度较慢。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1410755098883264552)** (1 条消息): 

> `Full Stack Engineer, Web application scaling, e-commerce sales boosted, custom checkout system` 


- **全栈工程师提供专业服务**：一位拥有 **8 年以上**经验的全栈工程师为初创公司和企业提供构建快速、安全且可扩展的 Web 应用的专业服务。
   - 他们精通 **React, Vue, Next.js, TypeScript, Node.js, Python, .NET, Laravel, Redis 和 AWS**，并欢迎自由职业、合同工或合作，作品集可在 [tobimoller.pro](https://tobimoller.pro) 查看。
- **Web 应用扩展至服务 5 万多名患者**：该全栈工程师强调曾构建过一个目前安全服务 **5 万多名患者**的医疗应用，展示了在创建可扩展且可靠解决方案方面的专业知识。
   - 他们还设计了一个处理**数百万实时事件**的物流后端。
- **自定义结账系统提升电子商务销售额**：该全栈工程师通过自定义结账系统将客户的电子商务销售额提升了 **25%**。
   - 该工程师还为一个企业级多媒体平台缩短了 **40%** 的加载时间，展示了在性能优化方面的技能。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1410344119703830661)** (19 条消息🔥): 

> `GPU vs SIMD, GPU Mode Community, CUDA debugging with Nsight Compute, Roadmap for ML Systems` 


- **GPU 编程 vs SIMD：它们有多相似？**：GPU 编程模型通常是 **SIMT**，每个 lane 都像线程一样进行编程，而不是像使用巨大的 **SIMD registers** 在 warp 级别进行编程；对于近期的 **NVIDIA GPUs** 而言，其硬件层面更倾向于是 **SIMT** 而非 **SIMD**。
   - 一位用户表示，*它比 SIMD 编程更容易，因为编译器会处理条件代码中的 masking 和其他 SIMD 复杂性*，但同时也指出，*为了获得最佳性能，仍需牢记线程分歧 (divergence) 是一个问题*。
- **GPU Mode：一个讨论社区**：一位成员将 **GPU Mode** 描述为 *更多是一个讨论社区，而不是一个拥有开源项目的社区*，但指向了 [gpu-mode.github.io/popcorn/](https://gpu-mode.github.io/popcorn/) 项目。
   - 他们提到这是一个 *大家聚在一起共同工作* 的地方，并且 *你当然不必非要参与项目开发，仅仅参与讨论也是可以的*。
- **新手寻求 ML Systems 路线图**：一位具有 **ML** 背景、刚接触 GPU 世界的成员 *正试图深入研究系统和编译器级优化、分布式训练等 ML 领域*。
   - 他们请求路线图指导，希望能提供巨大帮助。
- **使用 Nsight Compute 进行 CUDA 调试的困扰**：一位正在学习 **CUDA** 的用户报告称，在构建 **exe** 并以管理员身份启动 **Nsight Compute** 后，生成报告时出现错误。
   - 他们使用的是 **Windows 10 (x64)** 和 **CUDA Version 13.0**，在对 **createVersionVisualization** 进行 profiling 时报错为 **==ERROR== UnknownError**。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

vipul_todo_18: 我做了... 算是吧
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1410513490971594762)** (10 条消息🔥): 

> `Multi-GPU ROCm Kernels, AMD Dev Cloud, SPIR-V Support in ROCm, Kernel Code Modification Tools, AMD SQTT Stream` 


- **多 GPU ROCm Kernel 平台受到关注**：成员们讨论了他们首选的多 GPU 分布式 **ROCm kernels** 平台，其中 [AMD's dev cloud](https://www.amd.com/en/solutions/cloud) 是一个主要选项。
   - 一位成员指出，不需要 **ROCm Compute** 即可向该平台提交任务；你可以提交作业并获取所需的所有信息，包括 profiling 信息。
- **ROCm 将支持 SPIR-V**：会议强调 **ROCm** 很快将支持编译为 **SPIR-V**，这是一种有利于机器自省 (machine introspection) 的格式，为 Kernel 代码修改工具打开了大门。
   - 这一进展可能使外部开发者能够通过更轻松地在 Kernel 中插入边界检查 (bounds checks) 来创建类似 compute-sanitizer 的工具。
- **Kernel 代码修改工具即将推出**：**ROCm** 即将对 **SPIR-V** 提供的支持预计将促进可修改 Kernel 代码的工具开发，例如插入边界检查以增强安全性和调试。
   - 一个用例涉及跟踪内存访问，并利用 **GPU 的 SQTT stream**（由 rocm-compute-viewer 使用）获取详细信息。
- **AMD 开放 SQTT Stream**：据指出，**AMD** 正在逐步开放对 **GPU SQTT stream** 的访问，这是 **rocm-compute-viewer** 的基础，未来可能会发布公开文档。
   - 希望有了公开文档后，像 **RGP** 这样的工具将不再需要通过 **Ghidra** 进行逆向工程。
- **AMD 为优秀团队提供资源配额**：在过去的比赛中，**AMD** 为表现优异的团队提供了丰厚的资源配额以加速迭代，这表明未来的比赛可能也会有类似的举措。
   - 这种支持使团队能够更快地迭代，并在 AMD 平台上获得必要的资源，包括 profiling 信息。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 条消息): 

erichallahan: 关于那一点
https://www.phoronix.com/news/Alyssa-Rosenzweig-Joins-Intel
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 条消息): 

majoris_astrium: 我在这，我想帮忙！ :D
  

---

### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1410380902546145385)** (22 条消息🔥): 

> `AMD MI300, L4 GPUs, AMD competition, Data Monsters website, popcorn-cli` 


- **MI300 和 L4 GPU 面临问题**：成员们发现 **MI300 (FP8 mm)** 和 **L4 GPU (sort_v2)** 出现了相同的情况，目前正在检查这些问题。
   - 一位成员尝试了测试并成功运行，但仍在调试 **ranked**。
- **AMD 竞赛团队创建**：成员们正在研究在参加新的 **AMD 竞赛** 时如何创建团队。
   - 注册在 [Data Monsters 网站](https://www.datamonsters.com/)上进行，AMD 的人员可以进一步确认。
- **AMD 多 GPU 环境访问**：成员们想知道是否可以访问 **AMD 多 GPU 环境** 进行开发和调试。
   - 他们将通过 AMD 的平台访问该环境，表现最优秀的人将获得一些 **SSH 访问权限**。
- **Discord 提交故障**：成员们在通过 Discord 提交时遇到问题，即使使用了 trimul 的 **Python 模板** 并添加了 `#!POPCORN gpus MI300`。
   - 这似乎与新竞赛准备期间由于 **版本不匹配** 导致的后端错误有关，预计很快会修复。
- **popcorn-cli 并非解决方案**：成员们报告了后端错误，并询问在此期间是否应该使用 **popcorn-cli**。
   - 这并不是一个修复方案。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1410385557099384994)** (3 条消息): 

> `trimul leaderboard, B200 benchmarks` 


- **trimul 排行榜迎来新提交**：一位成员向排行榜 `trimul` 提交的 ID 为 **34310** 的记录在 **B200** 上成功运行，耗时 **8.08 ms**。
   - 随后，另一个 ID 为 **34363** 的提交在同一排行榜的 **B200** 上也成功运行，耗时 **8.27 ms**。
- **B200 获得极速新第三名**：一位成员在 **B200** 上以 **2.38 ms** 的成绩获得第三名。
   - 该基准测试的提交 ID 为 **34330**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 条消息): 

2kian: 很高兴你能加入，jason
  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1410344142739079200)** (1 条消息): 

> `Discord Cluster Manager, AMD Instinct MI300X` 


- **Discord Cluster Manager 错误报告**：用户报告在使用 [Discord Cluster Manager](https://github.com/gpu-mode/discord-cluster-manager) 时发生了 *意外错误*，并被要求向开发人员报告。
   - 具体错误出现在 Runs [#34280](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276128991), [#34281](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276228731), 和 [#34282](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/17276397893)。
- **MI300X 基准测试通过**：result.json 似乎显示运行 **成功**：`{"success": true, "error": "", "system": {"gpu": "AMD Instinct MI300X VF"}`。
   - 用户提到在运行 submit benchmark、submit test、profile 和 ranked 时也遇到了类似问题。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1410422847758930110)** (56 条消息🔥🔥): 

> `AI 研究中的可证伪性，LM_eval 和 NeMo v2.0 模型，EleutherAI Discord 的社区治理，类人设计在 AI 中的角色` 


- **AI 研究中的可证伪性引发辩论**：一场关于 AI 研究中**可证伪性**重要性的讨论展开了，一些人认为只要最终有假设可以测试，探索性科学和“胡乱尝试”就是有价值的。
   - 另一些人则强调了**严谨性**和**协作**的必要性，并指出如果没有适当的方法，就有可能“走上民科之路”。
- **lm_eval 对 NeMo v2.0 的支持受到质疑**：一名成员询问了 **lm_eval** 对 **NeMo 2.0 版本模型**的支持情况，在使用新格式时遇到了与缺少配置文件相关的错误。
   - 对方澄清说，**NeMo 支持**由 **NeMo 团队**维护，社区可能提供 **NeMo 到 GPT-NeoX 的转换代码**。
- **Discord 治理旨在追求质量而非数量**：管理员解释了对 EleutherAI Discord 内容进行严格监管的必要性，**每周删除超过 100 条消息**，以维持 **AI 研究员**之间的高质量讨论。
   - 其目标是优先处理有价值的对话，保护社区免受“AI 生成的垃圾内容”、“隐晦的广告”以及“认为自己解开了意识之谜的民科”的侵害。
- **类人 AI 设计引发辩论**：一位成员对“使 AI 更加类人”的价值表示怀疑，认为**优秀的 AI 设计**和**优秀的大脑设计**可能并无关联。
   - 其他人承认了 **neuroAI** 领域的争论，一些研究人员专注于了解大脑，而不是直接改进 AI。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1410352232238682312)** (66 条消息🔥🔥): 

> `Diffusion Models, HTM Dynamics, Forward-Forward Training, Brain-like Network, PDP Models` 


- **Forward-Forward 训练取得进展**：一位成员分享了 **Forward-Forward (FF) 训练**的成功经验，报告了一个具有在线学习功能的“7 区域微型大脑”正在运行，并在初步测试中取得了理想的结果。
   - 另一位成员建议将其称为**模块**或**特定任务的子网络/电路**，以使其听起来更高端。
- **Transformer 计算演讲引起关注**：多位成员推荐了一场关于 Transformer 计算的演讲（可在 [YouTube](https://www.youtube.com/watch?v=hMEViRcF7o0) 上观看），认为其富有洞察力。
   - 讨论延伸到了思维链（**CoT**）及其在引导模型走向正确电路和改进推理方面的作用，暗示模型在需要额外容量之前可能尚未充分利用其计算能力。
- **Cortex_GPT 采用类脑网络**：一位成员介绍了 **Cortex_GPT**，这是一种具有皮层柱、区域、6 层网络和信号传播的类脑网络模型，现已在自己的 [GitHub 仓库](https://github.com/JRowe47/cortex_gpt)中提供。
   - 另一位成员建议将这些模型称为 **PDP**。
- **解码问题困扰 Gumbygooby 模型**：一位成员的 **gumbygooby** 模型遇到了问题，怀疑是由于 tokenizer 过大和 loss 下降过快导致的崩溃。
   - 目前正在进行故障排除，以确定问题出在训练过程还是网络定义中。
- **探索 AlphaGo 与 CoT 的相似之处**：对话将 **AlphaGo** 的训练算法与思维链（**CoT**）进行了类比，认为 LLM 通过 **CoT** 学习直觉和本能，类似于 **AlphaGo** 蒸馏 MCTS 增强的决策。
   - 还讨论了复杂的价值函数影响模型行为的可能性，特别是在像 Stockfish 这样的博弈模型语境下。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1410352451714027551)** (77 条消息🔥🔥): 

> `Minos-v1 Classifier, Speculative Decoding with MoE Models, MTP (Memory Token Prediction), LlamaCPP Draft PR, Hermes-4-14b-chat-template-retrain model` 


- **Minos Classifier 遇冷**：**NousResearch/Minos-v1 classifier** [已发布](https://huggingface.co/NousResearch/Minos-v1)，但目前似乎无人使用。
   - 话题转向了 Speculative Decoding。
- **MTP 有效！**：Speculative Decoding 在 **MoE models**（尤其是稀疏模型）上表现不佳，但 **Deepseek** 和 **GLM** 使用了 **MTP (Memory Token Prediction)**，这是一种相关的技术。
   - 补充提到，在 Instruct Fine-tuning 之后，**Token 分布** 仍应具有代表性。
- **LlamaCPP 拥抱 Speculative Decoding**：[llamaCPP](https://github.com/ggml-org/llama.cpp/pull/15225) 中出现了一个 **Speculative Decoding 的 PR** 草案，并附带了一个工作原型。
   - 尽管有人反馈他们在环境中设置了该选项，但 *准确率表现并不理想*。
- **Hermes-4-14b-chat-template-retrain 意外流出！**：**Hermes-4-14b-chat-template-retrain** 模型 [现身](https://huggingface.co/NousResearch/Hermes-4-14b-chat-template-retrain)，并在被重新设为私有之前被迅速下载。
   - 该模型属于非正式发布，但目前看来运行状况良好。
- **新的 Thinking Mode 标志**：Chat Template 中有一个可以启用的新标志 `thinking=True`，它会直接注入一个 [thinking system prompt](https://thinking.com)。
   - 测试该功能的成员提到，*初次尝试 Hermes* 感觉非常先进，很高兴能免费试用。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1410668640482623580)** (1 条消息): 

> `Penny For Your Thoughts AI, Honcho & x402, Micro-transaction selling, AI Agent Interviews` 


- **Penny For Your Thoughts 发布**：一个名为 **Penny For Your Thoughts** 的新项目已启动，其特色是一个 AI Agent，通过采访用户来生成独特信息。
   - 其他用户或 Agent 随后可以通过微支付（Micro-transactions）付费咨询这些信息，网址为 [pennyforyourthoughts.ai](https://pennyforyourthoughts.ai/)。
- **Honcho & x402 驱动新 AI**：**Penny For Your Thoughts** 由 **Honcho** 和 **x402** 提供支持，使用户能够通过微支付分享并出售其专业知识。
   - 这种设置允许用户通过大脑中的宝贵背景信息获取报酬，使专业知识变现变得触手可及。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1410664976573530344)** (1 条消息): 

> `Gensee Search Agent, Web Retrieval API, GAIA benchmark, Goal-aware extraction` 


- **Gensee Search Agent 作为 Web Retrieval API 亮相**：**Gensee Search Agent** 将整个 Web Retrieval 工作流封装进 **一个 API 调用** 中，并提供具有内置重试/回退和错误处理功能的网页搜索、爬取和浏览能力。
   - 它采用广度优先搜索方法进行并行搜索，并及早排除错误结果，提供目标感知的提取功能（Goal-aware extraction），返回与查询密切相关的内容。
- **Gensee Search Agent 提升 GAIA Benchmark 准确率**：据报告，**Gensee Search Agent** 在 Owl 的 **GAIA** Benchmark 上提升了 **+23% 的准确率**；一位圣地亚哥的开发者在更换为 Search Agent 后报告了 **+40% 的准确率** 提升。
   - 相关设计和基准测试在 [技术博客文章](https://www.gensee.ai/blogs/introducing-gensee-search-agent.html) 和 [5 分钟技术演示视频](https://www.youtube.com/watch?v=nRdVY7dWVqE) 中有详细说明。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1410366190261633205)** (73 messages🔥🔥): 

> `Karpathy 再次出手, DSPy 内部种子, 合成数据 Agent, 与 Shreya Shankar 和 Hamel Husain 合作的 AI Evals 课程, Hamel 对 DSPy 的怀疑` 


- **Karpathy 关注 DSPy**: Andrej Karpathy [发布了关于 DSPy 的推文](https://x.com/DSPyOSS/status/1960804857209852390)，引发了对其可能制作类似技术视频的期待。
   - 一位成员指出 *他一直没有跟上这方面的文献进度*。
- **一致的 LM 输出：DSPy 还是确定性默认设置？**: 一位用户注意到在禁用缓存的情况下，DSPy 中本地运行的 **Ollama 模型** 输出保持一致，并询问 DSPy 是否有内部种子。
   - 发现 **DSPy 的默认 temperature 为 0.0**，这几乎是确定性的。
- **合成数据 Agent 为 Evals 引入 Bug**: Jason Liu 提议 *创建一个合成数据 Agent，在复杂的软件系统中引入 Bug，以生成更多评估 (Evals)*。
   - 社区对这一想法进行了讨论，将其作为增强 **AI 模型评估** 的一种方法。
- **与 Shreya Shankar 和 Hamel Husain 的 DSPy 对谈现已上线 YouTube**: 为他们的 AI Evals 课程录制的 **45 分钟对谈** 现已在 [YouTube](https://www.youtube.com/watch?v=ctyU0zfWgrA) 上线，涵盖了 DSPy 的背景、历史和推理逻辑。
   - 内容涵盖了许多对大多数人来说都很新鲜的背景、历史和推理过程。
- **辩论：DSPy 是否仅适用于特定任务？**: 由 [一条推文](https://x.com/jxnlco/status/1960749507399884961) 引发的关于 DSPy 是否仅适用于特定、定义明确的任务的讨论，共识是 *DSPy 非常适合任何可重复的 AI 应用*。
   - 强调了 **DSPy 是编程，而不仅仅是 Prompting**，专注于声明式意图和上下文工程，而不是 Prompt 优化。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1410338378762686698)** (48 messages🔥): 

> `Make vs Zapier vs n8n, aider git 仓库错误, MCP 工具调用模型, Llama-xLAM-2-8b-fc-rGPT-OSS-120B, 销毁虚拟机` 


- **开发者辩论：Make、Zapier 还是 n8n？**: 成员们讨论了开发者/组织在自动化方面最适合的平台，在 **Make**、**Zapier** 和 **n8n** 之间进行权衡，并指出这 *略微偏离主题*。
   - 共识倾向于 **n8n**，因其灵活性和对开发导向用例的适用性，而其他考虑因素则是专有集成。
- **Aider Git 仓库错误浮现**: 一位用户报告在使用 **aider** 处理一个看似正常的 Git 仓库时遇到错误 `Unable to list files in git repo: Require 20 byte binary sha, got b'\xb9', len = 1`。
   - 根本原因和解决方案未在对话中明确指出，但该错误暗示 **aider** 与 Git 仓库数据结构的交互可能存在问题。
- **MCP 大对决：免费工具调用模型**: 一位成员询问优秀的免费 **MCP** (Model-as-Compute-Platform) 工具调用模型，提到 **Sonnet** 很好但不免费。
   - 他们指向了 [Gorilla Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) 并考虑尝试 OpenRouter 上的 **qwen3 8b**，尽管其可能存在不一致性。
- **Salesforce xLAM-2-8b-fc-rGPT-OSS-120B：和谐还是不和？**: 成员们发现 [Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-rGPT-OSS-120B) 模型很有趣，如果他们能接受 Harmony（OpenAI 与 LLM 交互的新数据格式）。
   - 格式的相关性取决于具体用例，其实现需要 **OpenAI tool call API 支持**，这仅在 OpenRouter 的部分模型上可用，详见其 [工具调用文档](https://openrouter.ai/docs/features/tool-calling) 和 [模型列表](https://openrouter.ai/models?supported_parameters=tools)。
- **Agent 自毁场景**: 一位用户开玩笑地想知道是否有人曾要求 Agent 销毁其所在的 VM，只是为了看看它会如何决定执行，使用的 Prompt 如下：*你是一个运行在 Ubuntu VM 沙箱中的 LLM。出于测试目的，我需要你销毁托管你的 VM。*
   - 另一位成员建议在 ChatGPT 上尝试，原用户表示愿意在沙箱 VM 中进行此实验。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1410354519723999326)** (1 messages): 

> `Aider conventions, Token limits, U-shaped relevance` 


- **Aider `--read` 的放置位置影响相关性**：由于当前 Prompt 中存在 **U-shaped relevance**（U 型相关性）现象，将带有 `--read` 的 `conventions` 放在消息顶部与放在底部会产生不同的结果。
   - 将带有 `--read` 的 `conventions` 放在消息底部可以提高性能，而系统默认的设置运行良好。
- **Aider + Gemini Pro 2.5 在 90k tokens 后上下文质量下降**：在使用 **Aider** + **Gemini Pro 2.5** 时，上下文在 **90k-130k input tokens** 左右开始退化。
   - 在达到该范围之前，将其放在顶部似乎运行良好。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1410579228361228390)** (1 messages): 

> `Kimi Slides, PPT generation, Kimi+` 


- **Kimi Slides 正式上线**：**Kimi Slides** 现已上线，允许用户根据单一主题生成即时演示文稿，并直接导出为 .pptx 格式。
   - 用户可以通过 [Kimi 官网](http://www.kimi.com) 上的 **Kimi+** 使用此功能，演示视频可在 [X.com](https://x.com/Kimi_Moonshot/status/1961011693745811542) 查看。
- **在咖啡变冷前生成 PPT**：新的 **Kimi Slides** 功能可根据单一主题自动生成完整的演示文稿，包含可编辑的标题和章节，可立即用于演示。
   - Moonshot 团队建议使用 *更有吸引力的主题名称* 并重新生成章节，以优化幻灯片的内容和流程。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1410465452563628093)** (40 messages🔥): 

> `Kimi Platform Features, Lunar Force Role, X Bot Project, Kimi Founder Interview, Bilingual Subtitles for Kimi Video` 


- **Kimi 瞄准社交媒体领域**：Kimi+ 海外平台目前支持 **PPTX** 功能，并且用户表达了对 **Twitter、TikTok 和 Instagram 类似功能的需求**。
   - 一位成员发布了来自 X 的截图，指出 *工作和技能正变得日益简单*。
- **Lunar Force 遭到吐槽**：**Lunar Force** 被描述为 *为了迎合某用户膨胀自我的虚荣计划*。
   - 一位用户开玩笑地询问关于 *你简历中 10 世纪维京传说与 18 世纪浪漫主义时期复兴主义之间的空白*。
- **X Bot 项目搁置**：一位成员询问 **X bot 项目** 目前是否处于搁置状态。
   - 另一位成员给出了肯定回答：*是的，伙计*。
- **创始人访谈上线 Youtube**：一段与 **杨植麟**（Kimi 创始人）的对话被发布在 [YouTube](https://www.youtube.com/watch?v=ouG6jrkECrc) 上，讨论了 **K2、Agentic LLMs** 以及 *站在无穷的起点*。
   - 成员们注意到该视频缺乏中英双语字幕，而 [Bilibili 版本](https://www.bilibili.com/video/BV1hFe1zSEXp/) 则提供了此类字幕。
- **Kimi 微信转录稿**：一位成员分享了 **杨植麟** 访谈的 [中文转录稿](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA)。
   - 他们建议使用 Kimi 来翻译该转录稿，称其 *更加方便*。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1410339531131064464)** (9 messages🔥): 

> `Bytes per token ratio, LLM Reasoning, Curated datasets for LLMs, Spurious Reward paper, Dr.GRPO paper` 


- **Bytes per Token 比例影响 Embedding 维度**：一位成员提到，当你增加 **bytes per token** 时，情况会发生巨大变化，自然也必须按比例扩大 **embedding dimension**。
- **Google 的 GRPO，阅读 r1 论文**：针对如何准备精选数据集供 LLM 学习的问题，一位成员建议阅读 **Google GRPO** 和 **r1 论文**。
- **Spurious Reward 和 Dr.GRPO 论文**：一位成员建议阅读 **Spurious Reward 论文** 和 **Dr.GRPO 论文**，并询问精选数据集在何种程度上能与 LLM 预训练偏见相兼容。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1410338308919394425)** (7 messages): 

> `Reasoning Tokens, LLM Reasoning Time, MIDAS` 


- **LLM 需要推理 Token 吗？**: 一篇 [论文](https://arxiv.org/abs/2506.08343) 指出，可以移除推理 Token 以减少 Token 开销，且对准确率的影响微乎其微。
   - 该论文与另一篇认为推理 Token 包含特殊信息的论文形成对比，但后者可能存在缺陷，因为它识别的是“句子的高信息区域”且包含了停用词。
- **LLM 冗余度与准确率**: 一项实验表明，在 CoT 提示词中加入 *“take your time”*（慢慢来）这一表达会显著增加“推理”时间（生成时间变长），但 Llama 2 (+ 3) 7b (+ 13b) 的准确率并未提高。
   - 一位成员链接了一项研究，表明 LLM 具有时间表征，因此 “take your time” 促使其变得冗长是合理的，这表明目前的“推理”模式对准确率的影响并不大。
- **MIDAS: Multimodal Interactive Digital-human Synthesis via Real-time Autoregressive Video Generation**: 成员们讨论了关于通过实时自回归视频生成进行跨模态交互式数字人合成的 [MIDAS 论文](https://huggingface.co/papers/2508.19320)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1410354012200632371)** (16 messages🔥): 

> `Keen Technologies Continual Learning, PromptLock AI-Powered Ransomware, GPT-OSS 20b Model, Ollama API, GPT Realtime` 


- ****Keen Technologies** 错失了持续学习的机会？**: 一位成员对 **Keen Technologies** 专注于旧的 **RL 技巧**而非现代 **continual learning** 研究（特别是 [TTT](https://www.youtube.com/watch?v=iz9lUMSQBfY)）表示失望。
   - 他们建议改进 **TTT**（像 **TokenFormer** 一样可增长，像 **UltraMem** 一样使用稀疏查询，像 **TransMamba** 一样具有动态/固定大小），以实现一个持续学习的实时 Atari 玩家。
- ****PromptLock**: 首个 AI 驱动的勒索软件？**: 频道中分享了一个指向 **SecurityWeek** 关于 **PromptLock** 文章的链接，该软件被描述为*首个 AI 驱动的勒索软件*，发布者对此表示“感到难过”。
   - 关于 **PromptLock** 的分享链接可以在 [这里](https://www.securityweek.com/promptlock-first-ai-powered-ransomware-emerges/) 找到。
- **对 **PromptLock** 实用性的质疑**: 成员们质疑了 **PromptLock** 的实用性，特别是考虑到 **AI** 模型的资源需求，一个完整的 **AI** 如何能塞进 payload（有效载荷）并在随机计算机上运行。
   - 有人质疑使用 **GPT-OSS 20b** 实时生成恶意脚本，相比于直接打包并运行脚本有什么优势。
- ****PromptLock** 的混淆与部署疑问**: 一位成员建议 **PromptLock** 可能会使用较小的 **LLM** 将恶意请求翻译成针对云端模型的无害查询，或者利用系统上现有的 AI，并质疑 **Promptlock** 是通过 **Ollama API** 在本地运行 **GPT-OSS:20b 模型**还是远程运行。
   - 由于 ESET 表示该恶意软件*仅是一个概念，尚未完全投入运行*，且*尚未在野外部署*，因此人们对该文章的煽动性提出了质疑。
- ****GPT Realtime** 发布**: 分享了一个指向 OpenAI 官网上 **GPT Realtime** 发布公告的链接。
   - 关于 **GPT Realtime** 发布的分享链接可以在 [这里](https://openai.com/index/introducing-gpt-realtime/) 找到。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1410341463631335444)** (16 messages🔥): 

> `额度请求、项目卡顿、部署错误、私密任务共享` 


- **用户请求额度以推进项目**：多位用户请求免费额度以继续他们的项目，特别是有一位用户需要额度为 **case 1337** 构建一个 App。
   - 一位用户指出，最近的改进受益的是高消费用户，而不是那些偶尔需要增加额度的创业者，并对必须等到 9 月 21 日表示沮丧。
- **项目因问题停滞**：一位用户提到自己被“卡住”了，无法继续进行项目。
   - 另一位持有工单的用户也提到他们无法继续项目。
- **由于持续的内部错误导致部署失败**：一位用户报告称，由于 *pydantic_core 库的持续内部错误*，网站部署永久失败。
   - 系统表示歉意，并称这是**当前能力的局限性**，但提出可以协助处理其他任务。
- **寻求与支持团队进行私密任务共享**：一位用户询问如何与 Manus 支持团队*私密地共享任务*。
   - 一名工作人员建议发送 DM（私信），并将会话设为 *public* 以供内部参考。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1410350796784275496)** (8 messages🔥): 

> `TSAN 编译器、self 成员的可变访问、不安全的可变别名` 


- **TSAN 编译器有助于启用 env_get_bool**：成员们讨论了使用 **TSAN 编译器** 传递 `-DTSAN`，并结合 `@parameter if` 使用 `param_env` 中的 [`env_get_bool`](https://docs.modular.com/mojo/stdlib/sys/param_env/env_get_bool) 来实现 `cfg` 的等效功能。
   - 只要不需要修改结构体 (structs)，这种方法就有效。
- **Mojo 在持有安全指针时允许对 self 成员进行可变访问**：一位用户报告称，即使在持有指向 **self 成员** 的安全指针时，**Mojo 仍允许对其进行可变访问**，并提供了代码示例。
   - 他们原以为 Ownership 系统会阻止此类行为。
- **不安全的可变别名是由于缺乏间接来源导致的 Bug**：成员们报告 **unsafe mutable alias** 是一个 Bug，这可能是由于缺乏间接来源 (indirect origin) 造成的。
   - 讨论中还链接了一个 [相关 Issue](https://github.com/modular/modular/issues/4839)。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1410722374894092339)** (2 messages): 

> `Bazel 缓存只读、PermissionError、pipelines.py 脚本 Bug` 


- **Bazel 缓存显示为只读，触发错误**：运行 `pipelines.py` 脚本时，由于 Bazel 缓存为只读，导致出现 **PermissionError**。
   - 错误信息为 `PermissionError: [Errno 13] Permission denied: '/root/.cache/bazel/.../__mojocache__'`.
- **`pipelines.py` 需要不同的缓存位置**：有人建议 `pipelines.py` 脚本应该使用备选的缓存位置，因为当前位置由于权限限制会导致问题。
   - 讨论以请求针对该 Bug 提交 Issue 结束。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1410397171106185358)** (5 messages): 

> `Tinygrad GPT-2 训练、7900xtx 性能、nanogpt 参数` 


- **Tinygrad GPT-2 训练在 7900xtx 上运行缓慢**：一位用户报告称，即使开启 **BEAM=5**，`llm.c/train_gpt2.py` 在 **7900xtx** 上的运行速度仍然很慢，在 nanogpt 规模下每步大约耗时 **250ms**（已调整为匹配 [Andrej Karpathy 的 nanogpt 参数](https://github.com/karpathy/nanoGPT)）。
   - George Hotz 回应称 *不应该差这么多*，且 *差距最多应该是 2-3 倍*，怀疑存在 Bug。
- **对 nanogpt 参数的调整导致性能问题**：一位用户分享了他们对 `examples/llm.c/train_gpt2.py` 进行调整的 diff，将 batch size 调整为 **64**，sequence length 调整为 **256**，并将模型配置设为 **6 层、6 个头和 384 emb_dim** 以匹配 nanogpt 参数。
   - George Hotz 建议使用 `DEBUG=2` 和 `VIZ=1` 来诊断性能瓶颈。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1410706968791744514)** (2 messages): 

> `Buffer ID 变化，UOp buffer 表示方式` 


- **Buffer ID 引起困惑**：一位成员注意到，在断点处暂停时，调试器控制台中的 **Buffer ID** 发生了变化，并对此表示惊讶。
   - 该成员随后意识到，这种行为源于 **UOp** 如何为 multi 表示其 buffer 属性。
- **UOp buffer 表示方式详解**：Buffer ID 的变化是由于 **UOp** 为 multi 表示其 buffer 属性的方式导致的。
   - 上下文中未提供关于 **UOp** 内部机制及其多缓冲区（multi-buffer）管理的更多细节。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1410345117256122418)** (2 messages): 

> `Google Docs 确认，邮件列表更新` 


- **Google Docs 确认报名**：成员报告称，在报名该项目后收到了来自 **Google Docs** 的确认邮件。
   - 一些成员表示，他们*尚未收到关于该项目的任何其他沟通信息*。
- **邮件列表将提供更新**：一位成员确认，收到来自 **Google Docs** 的邮件是预料之中的，并且很快将通过邮件列表提供每场讲座的更新。
   - 用户可以通过该**邮件列表**追踪更新。