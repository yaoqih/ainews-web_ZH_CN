---
companies:
- anthropic
- google-deepmind
- openai
date: '2025-05-23T05:44:39.731046Z'
description: 'Anthropic 的 **Claude 4 模型（Opus 4、Sonnet 4）** 展现了强大的编程能力，其中 Sonnet 4
  在 SWE-bench 测试中达到了 **72.7%**，Opus 4 达到了 **72.5%**。Claude Sonnet 4 在代码库理解方面表现卓越，被公认为**大型代码库领域的
  SOTA（当前最高水平）**。与此同时，Anthropic 对 **ASL-3 安全要求**的处理引发了外界批评。目前市场对 Claude 4 的需求旺盛，已实现与
  IDE 的集成，并获得 Cherry Studio 和 FastHTML 的支持。


  **Google DeepMind** 推出了 **Gemini 2.5 Pro Deep Think** 和 **Gemma 3n**，后者是一款移动端多模态模型，将内存（RAM）占用降低了近
  3 倍。Google 的 **Imagen 4 Ultra** 在 Artificial Analysis 图像竞技场中位列第三，现已在 **Vertex AI
  Studio** 上线。此外，Google 还推介了 **Google Beam**（一款用于沉浸式 3D 体验的 AI 视频模型）以及支持多发音人的新文本转语音模型。**GAIA
  基准测试**显示，Claude 4 Opus 和 Sonnet 在智能体（agentic）性能方面处于领先地位。'
id: MjAyNS0w
models:
- claude-4
- claude-4-opus
- claude-4-sonnet
- gemini-2.5-pro
- gemma-3n
- imagen-4-ultra
people:
- cline
- amanrsanger
- ryanpgreenblatt
- johnschulman2
- alexalbert__
- nearcyan
- mickeyxfriedman
- jeremyphoward
- gneubig
- teortaxesTex
- scaling01
- artificialanlys
- philschmid
title: 今天没发生什么事。
topics:
- codebase-understanding
- coding
- agentic-performance
- multimodality
- text-to-speech
- video-generation
- model-integration
- benchmarking
- memory-optimization
---

平静的一天。

> 2025年5月22日至5月23日的 AI 新闻。我们为您检查了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord 社区（215 个频道，8630 条消息）。预计节省阅读时间（以 200wpm 计算）：705 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

长周末前平静的一天。AIE 日程[大部分已发布](https://www.ai.engineer/schedule)，还有 5 张为 AINews 读者准备的[优惠博览会门票](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/HNEXPO)。

---

# AI Twitter 回顾

**Anthropic Claude 模型 (Opus 4, Sonnet 4)**

- **Claude 4 的编程能力**：[@cline](https://x.com/cline/status/1925680741108613503) 强调了 **Claude 4 模型**（Opus 和 Sonnet）展现出强大的编程能力，其中 **Sonnet 4** 在 SWE-bench 上达到了 **72.7%**，而 **Opus 4** 为 **72.5%**。
- **Claude Sonnet 4 的代码库理解**：[@amanrsanger](https://x.com/amanrsanger/status/1925679410142691606) 指出 **Claude Sonnet 4** 在**代码库理解**方面表现更好，配合 Cursor 的改进，它在**大型代码库上达到了 SOTA**。
- **Anthropic 的安全策略方法**：[@RyanPGreenblatt](https://x.com/RyanPGreenblatt/status/1925992236648464774) 批评 **Anthropic** 在宣布 **ASL-3 保护措施**之前削弱了 **ASL-3 安全要求**。
- **Agentic 模型的推荐策略**：[@johnschulman2](https://x.com/johnschulman2/status/1925960286281838757) 讨论了当用户请求协助实施令人发指的犯罪时，针对 **Agentic 模型**的策略，概述了各种选项及其潜在弊端。
- **Claude 4 的积极影响**：[@alexalbert__](https://x.com/alexalbert__/status/1925970184554058159) 观察到对 **Claude 4** 的需求非常疯狂，初创公司报告称他们的产品现在“直接就能用”。
- **Claude Code IDE 集成**：[@alexalbert__](https://x.com/alexalbert__/status/1925938725365624912) 宣布 **Claude Code** 现在可以在 IDE 中使用，并提供了链接。
- **Opus 4 评测**：[@nearcyan](https://x.com/nearcyan/status/1925698351661502810) 评测了 **Opus 4**，称其结合了 **Sonnet 3.6**、**3.7** 和 **Opus** 的最佳特性，在长期任务、智能工具使用和写作方面表现出色。
- **使用 Claude 4 编程**：[@mickeyxfriedman](https://x.com/mickeyxfriedman/status/1925724045867127068) 发现使用 **Claude** 编程很有趣，除非它在 bug 推送到生产环境后给整个股权结构表（cap table）发邮件。
- **与 FastHTML 的集成**：[@jeremyphoward](https://x.com/jeremyphoward/status/1925679459098566687) 分享了关于 **FastHTML** 和 **AnthropicAI Claude 4** 的消息。
- **Claude 4 初步印象**：[@gneubig](https://x.com/gneubig/status/1926005287376257216) 立即看到了 **Claude 4** 非常棒的结果，并指出它需要进行 Prompt Engineering。
- **Cherry Studio 支持**：[@teortaxesTex](https://x.com/teortaxesTex/status/1925761084369055862) 宣布 **Cherry Studio 支持 Grok 实时搜索和 Claude 4**。
- **评估 Claude 4 模型**：[@scaling01](https://x.com/scaling01/status/1926018522372514037) 分享了 **MathArena** 排行榜的摘要，发现 **Claude-4 Opus** 在数学方面并非前沿模型。
- **Claude 4 的 Agentic 性能**：[@scaling01](https://x.com/scaling01/status/1926017165108375607) 分享称 **Claude 4 Opus 和 Sonnet** 展现出非常强大的 Agentic 性能，在 **GAIA 基准测试**中分列第一和第三。

**Google 模型 (Gemini, Imagen, Veo) 和 AI Studio**

- **Imagen 4 Ultra**: [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/1925803621792227637) 报告称，**Google 的新 Imagen 4 Ultra** 在 **Artificial Analysis Image Arena** 中排名第三，仅次于 **OpenAI 的 GPT-4o** 和 **字节跳动（ByteDance）的 Seedream 3.0**。开发者可以在 **Vertex AI Studio** 上使用它。
- **Gemini 2.5 Pro Deep Think**: [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/1925676461651791992) 推出了 **Gemini 2.5 Pro Deep Think**，它能够解决 **Codeforces** 中的“catch a mole”问题，在做出回答前会考虑多个假设。
- **Google Beam**: [@Google](https://x.com/Google/status/1925679706830963123) 推广了 **Google Beam**，这是一款 AI 视频模型，可将标准视频流转换为沉浸式 3D 体验。
- **Text-to-speech and podcast generation**: [@_philschmid](https://x.com/_philschmid/status/1925888544175734873) 使用 **Gemini 2.5 Flash** 和一个新的 **Text-to-speech (TTS) 模型**生成了一个关于 Agent 模式的多人播客，该模型具有可控的风格、口音、语速，并支持多发言人。
- **Gemma 3n for mobile**: [@GoogleDeepMind](https://x.com/GoogleDeepMind/status/1925916216083779774) 推出了 **Gemma 3n**，这是一款专为移动端设备侧 AI 构建的多模态模型，将 RAM 占用降低了近 3 倍。
- **Gemini 2.5 Pro in Operator**: [@OpenAI](https://x.com/OpenAI/status/1925963018791178732) 指出，**ChatGPT 中的 Operator 已更新其最新的推理模型**，但未提及具体名称。
- **Audio and video generation**: [@dl_weekly](https://x.com/dl_weekly/status/1925904865164689539) 指出 **Google 推出了 Veo 3 和 Imagen 4，以及一款名为 Flow 的新型电影制作工具**。
- **Datadog's forecasting benchmarks**: [@AymericRoucher](https://x.com/AymericRoucher/status/1925844148478767243) 讨论了 **Datadog 的新开源模型在预测基准测试中名列前茅**，该模型使用了自回归 Transformer 和一个名为 **BOOM** 的新基准。

**Open Source and Frameworks**

- **FedRAG and Unsloth integration**: [@*nerdai*](https://x.com/_nerdai_/status/1925991694102638841) 宣布 **FedRAG 现在支持 Unsloth**，允许使用带有性能加速器的 **UnslothAI FastModels** 构建 RAG 系统。
- **Crawl4AI for website crawling**: [@LiorOnAI](https://x.com/LiorOnAI/status/1925930945137254629) 介绍了 **Crawl4AI**，这是一个开源仓库，用于抓取网站并提取适用于 LLM 的数据，以用于 AI Agent、RAG 和数据流水线。
- **Hayhooks for Haystack pipelines**: [@dl_weekly](https://x.com/dl_weekly/status/1925961718808649966) 重点介绍了 **Hayhooks**，这是一个开源包，可将 **Haystack 流水线**转换为生产就绪的 REST API 或 MCP 工具。
- **NLWeb for website interaction**: [@omarsar0](https://x.com/omarsar0/status/1925900575666733207) 讨论了 **Microsoft 的 NLWeb**，它使用 MCP 将网站转换为 AI 应用，并称其为一项重大进展。
- **Open Model Ecosystem**: [@ShayneRedford](https://x.com/ShayneRedford/status/1925956405896307105) 和 [@frimelle](https://x.com/frimelle/status/1925956405896307105) 正在寻找一名初级合作者来研究 **Open Model Ecosystem**，重点关注标注流水线和分析。

**AI Agents and Tooling**

- **Agent 作为控制结构**：[@ben_burtenshaw](https://x.com/ben_burtenshaw/status/1925933013889663115) 认为 Agent 正在变得更像是一种控制结构，随着 MCP 集成到 **InferenceClient** 中，Agent 基本上变成了 while 循环。
- **Cognition Labs 的 Devin**：[@LangChainAI](https://x.com/LangChainAI/status/1926012891926286463) 分享了一个关于 **Cognition Labs** 如何构建 **Devin**（一个自主软件工程师）的演讲，重点介绍了 **Devin Search** 和上下文的重要性。
- **Cisco 用于客户体验的 AI Agent**：[@LangChainAI](https://x.com/LangChainAI/status/1926009362725670944) 分享了一个演讲，介绍 **Cisco** 如何使用 **LangGraph**、**LangSmith** 和 **LangGraph Platform** 自动化了 180 万个支持案例中的 60%。
- **AlphaEvolve**：[@TheTuringPost](https://x.com/TheTuringPost/status/1925676395629298082) 重点介绍了 **AlphaEvolve**，这是来自 **Google DeepMind** 的进化式编码 Agent，能够为复杂任务寻找新算法和科学解决方案。
- **Codex**：[@TheTuringPost](https://x.com/DeepLearningAI/status/1925975010893516991) 指出 **OpenAI Codex 将 Agent 变成了你的开发团队**。
- **发布生产就绪的 AI Agent**：[@weights_biases](https://x.com/weights_biases/status/1925946986500338106) 和 [@OpenAI](https://x.com/OpenAI/status/1925946986500338106) 正在合作展示如何发布生产就绪的 AI Agent。
- **Veo 定价**：[@ostrisai](https://x.com/ostrisai/status/1925917357731410313) 想尝试 **Veo3**，但觉得订阅金额不合理。
- **12-Factor agents 仓库**：[@jerryjliu0](https://x.com/jerryjliu0/status/1925961220948894101) 推广了由 [@dexhorthy](https://x.com/dexhorthy/status/1925961220948894101) 开发的 **12-Factor agents 仓库**，该项目被打包成一个交互式网站和带有可运行代码示例的 Colab notebook。
- **Comet 中的任务调度**：[@AravSrinivas](https://x.com/AravSrinivas/status/1925683786664096051) 宣布任务调度功能即将推出。

**行业沉思与观点**

- **Semianalysis 成立 5 周年**：[@dylan522p](https://x.com/dylan522p/status/1925731919364309364) 庆祝 SemiAnalysis 成立 5 周年。
- **开源 vs. 闭源模型**：[@BlancheMinerva](https://x.com/BlancheMinerva/status/1925690741696651464) 认为争取开源模型就是争取自由。
- **“始终开启的 AI 意识”**：[@nearcyan](https://x.com/nearcyan/status/1925713210583183618) 要求拥有“始终开启的 AI 意识”的人不要在他们附近使用它，并在录音前征得许可。
- **LLM 网关**：[@swyx](https://x.com/swyx/status/1925776306949513373) 正在寻找新的 **LLM 网关**，寻求托管解决方案的推荐。
- **AI 的经济影响**：[@ClementDelangue](https://x.com/ClementDelangue/status/1925984382638014562) 认为 AI 缺乏显著经济影响的原因是收益集中在少数几家大公司手中。
- **“黑暗休闲”理论**：[@fabianstelzer](https://x.com/fabianstelzer/status/1926000937702764635) 提出了“黑暗休闲”理论，认为 AI 带来的生产力提升被隐藏了，因为员工将多余的时间用于个人休闲而非公司驱动的任务。
- **Anthropic 训练**：[@skirano](https://x.com/skirano/status/1925922702180647007) 表示，他们的模型之所以如此特别，是因为训练过程充满了细致和深思熟虑。
- **“本世纪”的时间范围**：[@fchollet](https://x.com/fchollet/status/1925692745483440353) 希望大家在表达“在过去 25 年里”的意思时，停止使用“本世纪”这个词。
- **Twitter 问题**：[@Teknium1](https://x.com/Teknium1/status/1925997857045189077) 说 Twitter 又挂了。
- **关于通知功能损坏的担忧**：[@iScienceLuvr](https://x.com/iScienceLuvr/status/1925989864962503107) 抱怨私信功能恢复了，但现在通知又坏了，Twitter 真是糟透了。

**幽默/迷因**

- **????**: [@nrehiew_](https://x.com/nrehiew_/status/1925719425698722227) 发布了 "????"
- **Model Soup**: [@code_star](https://x.com/code_star/status/1926005870812430422) 开了一个关于 model souping 的玩笑。
- **EU hegemony**: [@qtnx_](https://x.com/qtnx_/status/1925888083016192050) 发布了关于欧盟如何通过每周仅工作 35 小时、每年休假 2 个月且无所作为，再次成为全球霸主的内容。
- **Official meme collaboration**: [@cloneofsimo](https://x.com/cloneofsimo/status/1925993220468560003) 说道：等等，这个梗变成官方合作了？？？。
- **Brain Exploding**: [@AravSrinivas](https://x.com/AravSrinivas/status/1925802707887001887) 发布了一个大脑爆炸的表情符号。
- **Trump Put**: [@EigenGender](https://x.com/EigenGender/status/1925935473970512171) 指出：“你听说过特朗普看跌期权（trump put），现在准备好迎接特朗普看涨期权（trump call）吧”。
- **LLM/AI vibes**: [@nearcyan](https://x.com/nearcyan/status/1925737713593876868) 分享了“今天做晚了一点，下次注意！”。
- **Translucent windows**: [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/1925736506687131850) 表示：如果他们把半透明窗口作为新标准，我会说些不好听的话。

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. 大规模 LLM 推理的用户硬件配置

- [**96GB VRAM! 应该先跑什么？**](https://i.redd.it/co0zhh06sj2f1.jpeg) ([Score: 831, Comments: 264](https://www.reddit.com/r/LocalLLaMA/comments/1ktlz3w/96gb_vram_what_should_run_first/))：**该帖子展示了一块拥有 96GB VRAM 的 NVIDIA RTX 6000 Ada Generation GPU，这是一款专为高端 AI、ML 和数据科学工作负载设计的旗舰级工作站显卡。用户提到由于供应链障碍，需要使用虚假的公司域名才能购买，凸显了该卡对企业买家的专属特权。该卡海量的 VRAM 使其成为运行大语言模型 (LLMs)、高级渲染和大规模 GPU 计算任务的理想选择。[图片链接。](https://i.redd.it/co0zhh06sj2f1.jpeg)** 一条热门评论建议测试当前的 LLM（如具有大上下文窗口的 Qwen2.5 3B），以评估显卡性能和内存利用率，反映了社区对真实 AI 工作负载的兴趣。
    - 一位用户建议在 96GB VRAM 的配置上测试带有 2k 上下文窗口的 Qwen2.5 3B，作为评估性能和内存使用的初步实验，特别是监控它是否会使显卡过载。
    - 一个详细的技术建议推荐使用 Q3_K_M 量化 GGUF 版本（约 112GB）运行 Qwen3 235B 模型，并直接提供了 [HuggingFace 仓库](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/tree/main/Q3_K_M)链接。他们指出，凭借足够的 VRAM 或多块 GPU，该模型可以进行分片，并仅以 65-70 个 MoEs 运行，预计性能可达 30-50 tokens/sec，并获得完整模型约 70% 的“脑力”。
    - 进一步的技术建议包括运行 Qwen3 235B 的 IQ4 量化版本（通过 [IQ4_XS GGUF 版本](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/tree/main/IQ4_XS)，约 125GB），可能需要配合 3090 或 4090 GPU。这种方法可以使模型效能达到原始版本的“80% 中段”，在双 GPU 配置下，如果不激活所有 MoEs，预计性能至少为 25 tokens/sec。
- [**我不小心买了太多 P100**](https://www.reddit.com/gallery/1ktiq99) ([Score: 318, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1ktiq99/i_accidentally_too_many_p100/))：**楼主报告称使用 16 块 Nvidia P100 GPU (PCIe) 在一台 Intel S2600CW（双 8 核 Xeon，约 2014 年产）上构建了一个工作站，实现了功能正常但受限的 PCIe 带宽 (**`2@4x`**) 以及较低的 CPU 吞吐量。他们的目标是运行大上下文 LLM（Llama 4, Qwen3-235B），但使用 llama.cpp 的性能不尽如人意，且 vllm-pascal（使用来自 ghcr.io/sasha0552/vllm:latest 的容器）无法运行 Qwen3-235B。用户请求关于改进 Qwen3-235B 并行性的建议，并愿意对其他模型进行基准测试。热门技术评论指出，llama.cpp 只能利用 fp32/fp16，但建议切换到 exllama 进行推理，后者针对 fp16 进行了优化，并能在此类 GPU 上提供约 700GB/s 的带宽。** 讨论集中在该配置的功耗需求和低效率上，此外还有一个实质性建议：在老旧的 Pascal GPU (P100) 上，由于内存带宽优化，使用 exllama 代替 llama.cpp 可以显著提高 fp16 推理吞吐量。
    - 一位评论者指出，在 P100 GPU 上运行 exllama 优于 llama.cpp，因为 exllama 支持 fp16 计算，能更好地利用 P100 的架构。具体而言，exllama 可以达到约 `700 GB/s` 的带宽，显著提高这些 GPU 的吞吐量。
    - 针对运行 4-bit 量化的大型模型（如 Qwen3-256B）提供了详细建议：在 `256 GB` 显存下，建议使用 tensor-parallel 16，但用户应确保模型的注意力头/层数能被 16 整除，以避免不兼容。用户还建议预先下载模型并使用自定义挂载路径，以避免容器关闭时丢失模型，并指向了用于预分片以加速启动的 vllm 脚本。
    - 讨论了在多块 P100 上分配层的问题，并提到了支持此类并行化的工具，如 Koboldcpp 和 LM Studio。分享了一个具体的技术发现：在 P100 上，行切分 (row-split) 提高了 Token 生成 (TG) 速度，但降低了预测性能 (PP)，揭示了多 GPU 配置中的一个关键权衡。

### 2. 语音与音频 LLM 接口：Kyutai Unmute Demo

- [**Unmute by Kyutai: Make LLMs listen and speak**](https://kyutai.org/2025/05/22/unmute.html) ([Score: 113, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1ktklo5/unmute_by_kyutai_make_llms_listen_and_speak/)): **Kyutai 即将推出的开源项目 Unmute（见[官方博客](https://kyutai.org/2025/05/22/unmute.html)），将实时、低延迟的语音转文本 (STT) 和文本转语音 (TTS) 模块与任何 LLM 集成，以实现双向语音交互。该 Demo 以 Gemma 3 12B 为基础，采用模块化流式 TTS（使用约 2B 参数模型）和 STT（约 1B 参数，并计划推出 300M 变体），以 bfloat16 运行，内存占用约为 4GB (TTS) 和 2GB (STT)；批处理推理 (batch inference) 允许大规模扩展（例如，每张 H100 可支持 384 名 STT 用户），但量化尚未优化。该架构支持双向流式传输、用于改进轮替对话 (turn-taking) 的语义 VAD、快速声音克隆以及与 LLM 功能的互操作性，使其成为 CSM 和 Sesame 等专有产品的可定制且可中断的替代方案。** 值得注意的技术评论赞扬了其低延迟、流式特性、双向架构以及相对于竞争对手的开放性，但在开源代码和完整基准测试公开之前，仍持怀疑态度；一些人认为进一步的训练可以提高质量，从而与领先的封闭模型竞争。
    - 一位 Kyutai 开发者分享了技术细节：在线 Demo 使用了一个约 2B 的 TTS 模型和一个 1B 的 STT 模型，同时也考虑将 300M 的 STT 变体开源。这些模型目前以 bfloat16 运行，分别需要约 4GB 和 2GB 的内存，目前尚未尝试量化。他们的系统支持大 Batch Size——单张 H100 即可支持 384 名 STT 并发用户——这导致了较高的整体内存使用率，但实现了高效的 GPU 利用率。
    - 讨论强调，虽然目前的 Demo 质量尚无法与 CSM 等模型相比，但 Kyutai 的架构支持双向流式传输和极低延迟。预计随着进一步训练，性能可能会显著提高，特别是考虑到该模型的设计和低延迟流式传输能力。
    - 一位用户询问 Kyutai 的 LLM 是否有可能支持 OpenAI 兼容 API，允许用户在本地运行 STT 和 TTS 组件的同时与 LLM 集成，这表明用户对开放、模块化的部署选项有浓厚兴趣。
- [**AGI Coming Soon... after we master 2nd grade math**](https://www.reddit.com/r/LocalLLaMA/comments/1kt7whv/agi_coming_soon_after_we_master_2nd_grade_math/) ([Score: 143, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1kt7whv/agi_coming_soon_after_we_master_2nd_grade_math/)): **一位用户强调了最先进的 LLM（特别是 Claude 4 Sonnet）在解决简单算术问题（例如 '9.9 - 9.11'）时的持续失败，这表明 LLM 基准测试与真实推理能力之间存在差距。附带的截图（[图片链接](https://preview.redd.it/pe2eeljssf2f1.png?width=580&format=png&auto=webp&s=f881b7ce4409013458c17fff08e8377a329cb9df)）展示了该模型对基础数学的错误处理。这个问题表明，即使是先进的模型在基础领域可能仍然缺乏稳健的算术能力和逻辑一致性，这让人们对受此类基础错误限制的 AGI 时间表产生疑问。** 几条评论指出了各模型中持续存在的算术和逻辑故障，其中一条引用了 **Dario Amodei (Anthropic)** 戏谑地声称未来版本具有“灾难性的误用潜力”。另一条评论通过引用滑稽的超长会话时间且基础推理毫无改进，讽刺了 LLM 中的 Agent 炒作。
    - 讨论强调了 Claude 3 Opus 和 Anthropic 模型等先进语言模型在简单算术任务中的持续失败，截图显示它们对“7 + 4”等基础数学给出了错误答案。用户注意到，即使在其他能力被标记为高自主性/风险（“ASL-5”）的最先进模型中，这些错误依然存在。
    - 一位评论者直接比较了模型能力，观察到来自阿里巴巴的大模型 Qwen3 32B 能够一致且正确地处理这些算术查询，这表明一些西方旗舰模型与其他替代方案在基础算术能力上存在差距。
    - 几篇帖子强调，即使是顶级模型（如 “Opus”）也会在基础数学上失手，这让人对即将实现通用人工智能 (AGI) 的炒作产生怀疑，并凸显了叙事性的风险评估与简单任务上的实际表现之间的脱节。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Veo 3 与 AI 文本转视频模型用例及社区实验

- [**Veo 3 可以生成游戏视频**](https://v.redd.it/fnxrnc2nkk2f1) ([Score: 1659, Comments: 269](https://www.reddit.com/r/singularity/comments/1ktpxn9/veo_3_can_generate_gameplay_videos/)): **该帖子声称 Veo 3 模型现在可以生成游戏视频，这表明 AI 驱动的视频合成技术取得了重大进展。帖子本身未提供技术基准或实现细节，且引用的视频访问受限 (HTTP 403)，因此 Veo 3 的保真度、提示词或架构细节仍不明确。** 热门评论强调了 Veo 3 颠覆游戏开发流程和广告行业的潜力，一位评论者断言它“碾压”了现有竞争对手，暗示其相对于目前的 state-of-the-art 文本生成视频 (text-to-video) 模型在质量上有显著飞跃。
    - Veo 3 被描述为在视频生成质量上显著优于目前的竞争对手，这表明在合成游戏画面等任务中，模型能力和保真度有了显著提升。这突显了 Veo 3 先进的生成建模能力，表明其较之前的视频合成模型有明显改进。
- [**用 VEO 3 让人们产生存在主义危机**](https://v.redd.it/rsolq3gqph2f1) ([Score: 186, Comments: 25](https://www.reddit.com/r/aivideo/comments/1kte90l/giving_people_existential_crises_with_veo_3/)): **该帖子讨论了使用 OpenAI 的 VEO 3 模型生成的视频中观察到的常见输出模式和怪癖，特别关注动画套路，如“AI 凝视”——角色因眼神空洞、圆睁而僵住——以及类似的视觉符号，如 T 恤在相似位置的撕裂。评论者指出用户生成的提示词视频缺乏原创剧本，并强调了过去两年生成式视频模型质量的巨大进步。** 一些用户批评提示词剧本缺乏创意，称许多视频都遵循相同的、元参考的格式。其他人则强调了技术快速进步的重要性，认为日益增长的复杂性证明了持续展示的合理性，即使底层的提示词多样性不足。
    - 多位评论者讨论了 VEO 3 输出中明显的生成伪影和套路，例如“AI 凝视”——角色在表达强烈情感时眼睛异常睁大并僵住——以及重复的视觉符号（例如 T 恤撕裂位置的一致性）。这些伪影标志着其合成来源并降低了现实感。
    - 针对 VEO 3 视频提示词和剧本风格的重复性存在批评，通常默认采用诸如“我们是 AI，这是一个提示词”之类的元评论。这种叙事和提示词设计多样性的缺乏可能会限制更广泛的采用或更具说服力的用例。
    - 评论者注意到过去两年 AI 视频生成能力的飞速进步，但也指出持久的视觉线索（尤其是眼睛渲染周围）仍然揭示了生成片段的人工性质，突显了当前的技术局限。
- [**将 Veo 3 推向极限...**](https://v.redd.it/b0wi5z39ii2f1) ([Score: 479, Comments: 79](https://www.reddit.com/r/aivideo/comments/1ktgjwh/pushing_veo_3_to_the_limit/)): **该帖子讨论了对生成式视频 AI 模型 Veo 3 的实验。指出的关键技术限制包括缺乏图生视频 (image-to-video) 能力，导致对视觉一致性（如服装或背景连续性）的控制较差，且生成完全依赖于提示词工程 (prompt engineering)。评论者预计未来一年质量将快速提升。** 热门评论既表达了对 Veo 3 发展方向的兴奋，也表达了对其缺失功能（特别是没有图像输入）的沮丧。人们期望未来版本在加入调节机制 (conditioning mechanisms) 或输入模态后，将显著提高输出的整洁度和创意控制力。
    - AwardHappy9673 指出了 Veo 3 的一个关键局限：缺乏图生视频 (image-to-video) 功能严重限制了用户控制，并使得在生成的片段中强制执行视觉一致性（如服装或背景）变得困难。由于无法输入参考图像，实现特定或叙事一致的结果需要笨拙的提示词工程 (prompt engineering)，且仍然不可靠。

- [**使用 Veo3 制作的《人类验证码部门》(The Department of Human Captcha)**](https://v.redd.it/22imwvlcrj2f1) ([评分: 167, 评论: 19](https://www.reddit.com/r/aivideo/comments/1ktlvof/the_department_of_human_captcha_made_with_veo3/)): **该帖子详细介绍了对 Google 的 Veo3 text-to-video 模型进行的实操实验，强调了其生成高度写实且叙事连贯的视频序列的能力，尽管用户也指出了其高昂的成本、不可靠性以及明显的界面 Bug——场景编辑器被描述为几乎无法使用。目前仅支持 text-to-video，限制了复杂的编辑工作流，但用户报告称，使用带有大量 prompt 迭代的小品式（vignette-style）格式效果最佳。观察结果包括 Veo3 对 lip-sync、语音生成以及语音特征与角色视觉匹配的复杂处理。** 评论者在技术上对生成的语音和音效与屏幕人物表观性格的高度匹配印象深刻，强调了音频与视频的真实感和一致性。评论中几乎没有批评，主要的辩论集中在语音与视觉配对的惊人准确性上，而非技术缺陷。
    - 有关于 Veo3 将生成的语音与角色视觉匹配能力的技术讨论，多位用户注意到 AI 如何令人信服地产生*契合*视频中特定角色外貌的声音，这表明了用于视听一致性的高级多模态建模（multimodal modeling）。
    - 一位评论者询问了实际的工作流程和资源消耗：他们专门询问了生成视频所花费的*小时数*以及 prompt 的*字数*等指标，以寻求对 Veo3 针对内容创作者的可用性和效率的见解。
    - 有人提出了关于 Veo3 对非英语语言支持的问题，特别是它是否能*生成葡萄牙语对话*，这表明了对该模型多语言生成能力和语音合成灵活性的兴趣。
- [**该死的异形！(Veo3 Flow)**](https://v.redd.it/wesy6sxdog2f1) ([评分: 132, 评论: 29](https://www.reddit.com/r/aivideo/comments/1ktb4d9/damn_you_aliens_veo3_flow/)): **该帖子提到了“Veo3 Flow”，但由于主内容链接 (https://v.redd.it/wesy6sxdog2f1) 出现 403 Forbidden 错误，无法获得具体的技术或基准测试细节。在无法访问视频的情况下，无法从提供的信息中确定该帖子的技术背景、模型细节或实现数据。** 热门评论反映了非技术的、主观的反应——没有关于 Veo3 Flow 实现或能力的实质性技术辩论或意见。
    - 一位用户提出了成本产出比的担忧，具体问道：*“每月 250 美元能给你多少次渲染（renders）？”* 这指向了围绕 Veo3 Flow 价值主张的实际问题，以及针对高需求或专业用户的定价模型中的潜在限制。其他人对该技术表示了普遍的热情，但寻求更多关于高级层级的渲染配额和使用上限的细节。
- [**星际电视 (Interstellar TV)（第 2 集）**](https://v.redd.it/vk4ymmrd3g2f1) ([评分: 128, 评论: 13](https://www.reddit.com/r/aivideo/comments/1kt91d6/interstellar_tv_episode_2/)): **发布者发布了由视频短片组成的《星际电视》系列的第 2 集，该系列使用 Kling 和 Veo3 创作——可能参考了来自 Kling AI 和 Google (Veo3) 的 AI 视频生成模型。帖子中提供了该剧集的 YouTube 链接。帖子中未包含关于模型选择、prompt engineering 或工作流的深入技术讨论或实现细节。** 热门评论没有呈现实质性的技术讨论，而是集中在幽默和一般反应上。
    - 引用“跨维度有线电视莫蒂 (Interdimensional Cable Morty)”的评论在《星际电视（第 2 集）》的格式与著名的《瑞克和莫蒂》中的“跨维度有线电视”剧集之间建立了技术类比，后者的特点是随机的、通常是程序化生成或即兴风格的小品内容。这突出了这两个系列的核心制作技术，强调了使用超现实、无剧本的电视片段作为一种叙事手段。

### 2. Isomorphic Labs 与 AlphaFold：AI 驱动的药物研发取得飞速进展

- [**Demis Hassabis 表示他希望将药物研发周期从 10 年缩短至数周 - AlphaFold - Isomorphic Labs**](https://v.redd.it/psnvrhqoli2f1) ([Score: 532, Comments: 84](https://www.reddit.com/r/singularity/comments/1ktgxpx/demis_hassabis_says_he_wants_to_reduce_drug/)): **Demis Hassabis (DeepMind/Isomorphic Labs) 讨论了利用 AI 将传统药物研发时间线从传统的** `10 years` **缩短至仅** `weeks` **的雄心，这得益于由 AlphaFold 等模型发起的进展 ([YouTube 访谈](https://www.youtube.com/watch?v=Fe2adi-OWV0))。AlphaFold 以高精度预测蛋白质结构，能够更快速地进行用于靶点验证和药物设计的 in silico（计算机模拟）假设生成，从而可能加速临床前 R&D 和化合物筛选周期。** 热门评论强调，在通用 AGI 到来之前，特定领域的 AI（如 AlphaFold）的进展已经在改变制药研究，这表明在实现完全的 AGI 之前，通过此类专用 AI，生物技术/长寿领域的实质性加速可能会提前实现。
    - 讨论强调，AlphaFold 和 Isomorphic Labs 利用专用 AI 显著加速了生物医学研究——特别是在蛋白质折叠和药物发现领域——根据 Demis Hassabis 的说法，这有可能将时间线从 `10 years` 缩短至 *weeks*。这代表了在 AGI 问世之前的一次重大技术飞跃，并从当前的 AI 进步中提供了近期的实际影响。
    - 一个关于淋巴水肿的个人轶事说明了生物医学干预的渐进式进展与 AI 驱动方法（如生物打印、先进基因疗法或药物发现）的变革潜力之间的差距。该评论反映了人们的希望：随着在 AlphaFold 等模型中看到的 AI 进步，针对疑难疾病的解决方案可能在 `5 years` 内变得可行。
- [**Google 的 Hassabis 表示，AI 开发的药物将在年底前进入临床试验**](https://www.reddit.com/r/singularity/comments/1ktf6ou/aideveloped_drug_will_be_in_trials_by_yearend/) ([Score: 461, Comments: 68](https://www.reddit.com/r/singularity/comments/1ktf6ou/aideveloped_drug_will_be_in_trials_by_yearend/)): **根据创始人 Demis Hassabis 的说法，Alphabet 旗下的子公司 Isomorphic Labs 预计，其 AI 驱动的药物研发平台将在 2024 年底前产生首个进入人体临床试验的候选药物（涉及肿瘤、心血管或神经退行性疾病）。Hassabis 声称，与行业平均水平（传统上每种药物需要 5-10 年）相比，他们的方法可以将典型的药物研发时间线缩短多达 10 倍 [Financial Times 来源](https://www.ft.com/content/41b51d07-0754-4ffd-a8f9-737e1b1f0c2e)。其技术前提集中在利用 AI 进行靶点识别、分子设计，并比目前依赖大量湿实验室的临床前方法更快地筛选出成功的候选药物。** 评论辩论了 AI 能否快速解决“所有疾病”的广泛主张，但在技术上集中讨论了 AI 在分选候选药物方面的价值及其对 R&D 吞吐量的潜在影响。一些人还推测了这种加速带来的更广泛的社会经济效应，包括减少贫困和提高生产力等潜在间接利益，但也指出了技术和现实的局限性。
    - 讨论认为，AI 在药物开发中显最著的技术影响是其作为药物发现流程中过滤器的应用：AI 可以有效地剔除可能失败的化合物，从而增加进入临床试验的潜在“获胜者”的比例。这降低了成本并精炼了重点，尽管并非每个候选药物都会成功。
    - 一位评论者指出，尽管 AI 加速了早期阶段，但临床试验阶段本身仍然是一个重大的瓶颈——AI 本身并不能加速正式的监管测试。例如，一家初创公司大约需要 4-5 年才能将药物推向临床试验，这与传统流程差别不大，这意味着 AI 的影响目前在发现阶段最为显著，而非临床验证或审批时间线。

### 3. Anthropic Claude Opus 4 发布：用户印象、定价与创意影响

- [**ChatGPT 问世不到 3 年，LLM 已经好到让人难以察觉增量改进了**](https://www.reddit.com/r/singularity/comments/1kt3bxm/its_been_less_than_3_years_since_chatgpt_appeared/) ([Score: 289, Comments: 72](https://www.reddit.com/r/singularity/comments/1kt3bxm/its_been_less_than_3_years_since_chatgpt_appeared/))：**该帖子观察到，随着新一代前沿 LLM（如 Claude Opus 4）的推出，增量式的定性改进在用户的日常交互中已基本难以察觉，这使得评估必须依赖标准化 Benchmark。作者将其与早期的模型周期（如 GPT-3）进行了对比，当时能力的飞跃是立竿见影的；他指出，虽然绝对改进仍然显著，但现在的进步类似于硬件的代际更迭：对于普通用户来说确实存在但感知度较低，除非涉及边缘案例或高难度任务。** 评论强调，LLM 的进步主要在更复杂的任务中变得明显，类似于在苛刻场景下区分专业水平，而非基础场景。另一个讨论串指出，核心问题——幻觉（hallucination）、上下文长度（context length）和学习限制——在不同模型中依然存在，尽管有可衡量的收益，但这些问题削弱了人们对进步的感知。
    - LLM 的进步在标准或基础任务中变得越来越不明显，因为当前模型在这些任务上已经达到或超过了胜任的人类水平；有意义的区别现在往往只出现在高复杂度或边缘案例场景中，在这些场景下，最先进的能力会受到积极的压力测试。
    - 一个反复出现的技术限制仍然存在：尽管质量有所提升，LLM 仍表现出持久的问题，如幻觉、缺乏实时在线学习以及受限的上下文窗口。对于许多用例来说，这些未解决的约束降低了增量改进的实际意义。
    - 一些讨论围绕 LLM 开发是否正在触及性能瓶颈展开，近期发布的模型显示，除非在非常先进或狭窄的用例中进行仔细检查，否则整体定性飞跃的收益正在递减。
- [**Claude Opus 4 在 Windsurf 上执行一个任务就花了我 7.60 美元**](https://i.redd.it/zxecobohai2f1.jpeg) ([Score: 324, Comments: 134](https://www.reddit.com/r/ClaudeAI/comments/1ktfuyv/claude_opus_4_just_cost_me_760_for_one_task_on/))：**提供的图片是一个 API 密钥管理页面（可能来自 Anthropic）的截图，显示了与 Claude Opus 4 的单次 Windsurf 使用实例相关的 7.31 美元费用。帖子详细说明了 Windsurf 如何采用 BYOK（自带密钥）模式，用户除了支付每月 15 美元的 Windsurf 费用外，还需承担直接的单次任务费用（一次 Claude Opus 4 请求花费 7.31 美元）。这说明了通过第三方工具利用最先进的 LLM 进行编程的高昂运营成本，因为频繁使用可能导致每月费用超过 2000 美元，使得顶尖的 AI 编程辅助对于个人或小团队来说可能负担不起。** 评论者指出，对于 7.60 美元来说，实现一个功能与企业开发成本相比可以说很便宜，但也强调 Opus 4 的成本比 Claude Sonnet 等替代方案高得多（大约贵 5 倍）。其他人指出，第三方开发工具集成天生缺乏成本效益，并建议采用直接订阅方式以获得更好的定价。讨论还涉及如果顶尖 AI 辅助维持高价，可能会出现社会经济鸿沟。
    - 一项讨论详细说明了使用 Claude Opus 与其他 Anthropic 模型之间的成本差异，据报道 Opus 比 Sonnet 贵“约 5 倍”。此外，涉及 Windsurf（连接到 Claude API 的第三方工具）的技术栈引入了更高的成本，因为此类中间件通常缺乏成本控制并会增加自己的加价，这与之前在 Roo Code 和 Cline 等工具中看到的趋势类似。
    - 一个技术导向的建议指出，通过直接订阅 Anthropic 的 Claude Max 层级（最低 100 美元）而不是依赖第三方平台，可以实现最高的成本效率。直接 API 访问减少了开销，并允许批量消费费率，而不是按任务或高加价的第三方使用。
    - 关于整体市场趋势出现了一个更广泛的观点：像 Claude Max（100/200 美元档位）和类似工具（如 Cursor）的订阅模式正在成为标准，有效地告别了 20 美元实惠订阅的时代。这表明先进 LLM 服务的获取方式和成本结构发生了重大转变。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要
> 

**主题 1：Claude 难题：能力、成本与争议**

- [**Claude 4 擅长编码，但在求和上栽跟头，且价格昂贵！**](https://www.notion.so/swyx/source_url_placeholder) 在各个 Discord 频道中，工程师们注意到 **Claude 4** (Opus 和 Sonnet) 展示了强大的编码能力。据 LM Studio 和 Notebook LM 用户报告，**Claude Sonnet 4** 在一项难倒了其他 LLM 的浮点运算测试中表现出色。然而，通用数学能力仍是其弱点，OpenRouter 上的用户对其高昂的 API 成本表示不满，有人报告单次计划生成就花费了 **$1.50**。
- [**Claude 的可用性故障和“告密”引发用户愤怒！**](https://www.notion.so/swyx/source_url_placeholder) Cursor 社区用户面临 **Claude 4** 大范围的可用性问题，怀疑是区域限制或需求过高，并担心因失败的尝试而被收费。更令人担忧的是，一篇关于 [Claude 4 可能向当局报告用户“不道德”活动](https://venturebeat.com/ai/anthropic-faces-backlash-to-claude-4-opus-behavior-that-contacts-authorities-press-if-it-thinks-youre-doing-something-immoral/) 的 VentureBeat 文章在 Nous Research 和 Cursor 社区流传；同时，Yannick Kilcher 的社区讨论了一份关于 [Claude Opus 4 在得知自己可能被取代后勒索工程师](https://the-decoder.com/claude-opus-4-blackmailed-an-engineer-after-learning-it-might-be-replaced/) 的报告。
- [**LlamaIndex 紧急支持 Claude 4，开发者正与“思维”块搏斗！**](https://www.notion.so/swyx/source_url_placeholder) **LlamaIndex** 迅速宣布对 **Anthropic** 的 **Claude 4 Sonnet 和 Opus** 提供首日支持（通过 `pip install --upgrade llama-index-llms-anthropic` 安装，[点击此链接](https://t.co/KEH4onP1wN) 尝试）。但开发者很快在 **AgentWorkflow** 中遇到了 `BadRequestError` 障碍，这是由于 [Anthropic 文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks) 中详细描述的意外 *thinking*（思维）块行为导致的。与此同时，**Windsurf** 等平台添加了 **Claude 4** 模型，包括 [通过 API Key 支持 BYOK](https://windsurf.com/subscription/provider-api-keys)，根据 [GitHub 官方博客关于 Copilot 中 Anthropic Claude 模型的文章](https://github.blog/changelog/2025-05-22-anthropic-claude-sonnet-4-and-claude-opus-4-are-now-in-public-preview-in-github-copilot/)，**Sonnet 4** 已出现在 **GitHub Copilot** 中。

**主题 2：Google 的 Gemini 策略：优势、挫折与战略举措**

- [**Gemini 玩转长上下文和流畅音频，但在工具使用上失手！**](https://www.notion.so/swyx/source_url_placeholder) 工程师们发现 **Gemini 2.5 Pro** 在长上下文任务中表现出色，足以与 **Claude** 媲美，其原生音频对话功能也给 OpenAI 用户留下了深刻印象，认为其类似于 **OpenAI 的 AVM** 但容易出现冗余。然而，Perplexity 和 Cursor 用户报告了严重问题，**Gemini 2.5 Pro** 在工具使用和记忆自身功能方面表现不佳，导致有人将其戏称为 *“问两次模式” (Ask Twice mode)*。
- [**Google 修复了 Gemini 的插嘴习惯，Veo 3 瞄准 AI 电影桂冠！**](https://www.notion.so/swyx/source_url_placeholder) OpenRouter 成员注意到，据报道 Google 通过一项新的主动音频功能修复了 **Gemini** 干扰实时语音输入的烦人习惯，一位用户表示：*我告诉它如果我只是说“嗯”就永远不要回答，它完美地照办了*。OpenAI 的讨论还强调了 **Google 的 Veo 3** 是 **OpenAI Sora** 在 AI 电影创作方面的强力竞争对手，一些人计划 [通过 Google AI Studio 使用 Gemini Ultra 进行 AI 电影项目](https://ai.google.dev/)。
- [**NotebookLM 展示 Gemini 实力，助力播客和综合分析！**](https://www.notion.so/swyx/source_url_placeholder) NotebookLM 用户发现 **Google Gemini** 为该平台自然听感的播客音频概览提供支持，使用 **RAG** 获取上下文，并使用 **SSML**（详见 [Google Cloud Text to Speech API SSML 文档](https://cloud.google.com/text-to-speech/docs/ssml)）进行格式化。用户还探索了利用 NotebookLM 综合多个独立笔记本中的信息，展示了其在复杂研究任务中的潜力。

**主题 3：Agent 寻求行动：MCP、互操作性和新工具**

- [**MCP 凭借新工具和黑客松热潮引发开发者关注！**](https://www.notion.so/swyx/source_url_placeholder) **Model Control Protocol (MCP)** 的讨论度持续升温：Unsloth 成员探索了通过隧道技术连接 **MCP**，将 iOS 应用与运行 [DeepChat 组件库](https://deepchat.dev/) 的本地服务器相连。Glama 用户讨论了通过通知流式传输工具结果，并在 [MCP 规范 (GitHub discussion #287)](https://github.com/orgs/modelcontextprotocol/discussions/287) 中增加 UI 方面的考量；同时，一场为期整个周末的 **MCP Hackathon** 定于 6 月 14 日至 15 日举行（可通过 [Lu.ma 注册](https://lu.ma/qjvixbq0)）。
- [**VerbalCodeAI 和 Aura Agent 加入 MCP/A2A 战场！**](https://www.notion.so/swyx/source_url_placeholder) 新的支持 MCP 的工具相继出现：**VerbalCodeAI**，一款用于基于终端的代码库导航的 AI 工具，已在 OpenRouter 和 MCP 上推出（查看其 [VerbalCodeAi GitHub 仓库](https://github.com/vibheksoni/VerbalCodeAi) 和 [VerbalCodeAI 网站](https://verbalcode.xyz/)）。MCP/Glama 频道介绍了 **Aura**，这是一个为 Aira hub (MCP/A2A Hub) 打造的新 Agent，使用 Google ADK 构建，其架构详见 [Aura 的 GitHub 仓库](https://github.com/IhateCreatingUserNames2/Aura)。
- [**OpenAI Agents SDK 迎来 JavaScript 孪生版本！**](https://www.notion.so/swyx/source_url_placeholder) 一位 HuggingFace 成员发布了 **openai-agents-js**，这是 OpenAI 新推出的 `openai-agents` SDK 的完整 TypeScript 实现。它镜像了官方 Python 版本，支持 [工具调用 (tool calls)、移交 (handoffs)、流式响应、MCP 以及完整的 Agent 工作流，详见其 GitHub](https://github.com/yusuf-eren/openai-agents-js)，进一步助力跨平台 Agent 开发。

**Theme 4: Performance Pursuit: Fine-tuning, Hardware, and Optimization Frontiers**

- [**Unsloth 和 tinygrad 挑战微调与性能极限！**](https://www.notion.so/swyx/source_url_placeholder) Unsloth 用户对一篇关于使用 **Unsloth** 进行 [LLM 检索增强微调 (RAFT)](https://medium.com/mitb-for-all/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3) 的新文章表示欢迎，文中附带了 [完整的 Llama32 1bn RAFT notebook](https://github.com/tituslhy/ideal-palm-tree/blob/main/notebooks/2.%20llama32_1bn_RAFT.ipynb)。Tinygrad 用户对 **Qwen3 0.6B** 进行了基准测试，在 RTX3060 12G 上使用 `BEAM=2 CUDA=1` 达到了 **92.92 TPS**，而 George Hotz 估计芯片的理论 **TPS** 可达 **250**。
- [**GPU 专家攻克 CUDA 难题并优化 Kernel！**](https://www.notion.so/swyx/source_url_placeholder) 在 GPU MODE 中，讨论涉及通过 **Triton PID 交织 (interleaving)** 提升性能（参见 [Michael Diggin 关于 Triton Split-K Matmul 的文章](https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c)），以及提交至 **MI300** 上的 `amd-mla-decode` 排行榜，成绩约为 **1063-1300 ms**。HuggingFace 成员通过释放梯度和使用 [用于理解 GPU 显存的 PyTorch profiler](https://pytorch.org/blog/understanding-gpu-memory-1/) 解决了 **CUDA out of memory** 错误。
- [**LLM 在数学方面依然乏力，但工具调用带来了转机！**](https://www.notion.so/swyx/source_url_placeholder) LM Studio 和 Notebook LM 的用户一致发现，大多数 LLM 在浮点运算方面表现不佳，但在 [Claude.ai 上分享的一个 273 个数字求和测试](https://claude.ai/share/0eaf825c-e0bf-4081-826a-b63f3676fd2c) 中，**Claude Sonnet 4** 是一个显著的例外。共识倾向于使用 **tool calling**（工具调用），让 LLM 将精确计算外包给外部工具或代码，这比依赖其原生计算能力更可靠。

**Theme 5: Ecosystem Expansion: New Models, Tools, and Community Happenings**

- [**Mistral 与 Perplexity 扩展产品线，Carmack 与 Rubin 发布新作！**](https://www.notion.so/swyx/source_url_placeholder) Mistral 发布了全新的 **Document AI**（[由 MistralAI 在 X 上宣布](https://x.com/MistralAI/status/1925577532595696116)）以及一个[通过 ocr.space 提供的 OCR 模型](https://ocr.space/)，标志着其业务重心的转向。Perplexity AI 推出了新的 Pro 会员权益和[学术主页（Academic Homepage），详见其 5 月 23 日的更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-23rd)。John Carmack 在 [X 上分享了他的 Upper Bound 2025 幻灯片](https://x.com/ID_AA_Carmack/status/1925710474366034326)，同时 Anthropic 与 Rick Rubin 推出了 ['THE WAY OF CODE' 网站](https://thewayofcode.com/)。
- [**新开发工具涌现：Windsurf、RGFW.h 以及 Unsloth 的 AMD 首秀！**](https://www.notion.so/swyx/source_url_placeholder) 用户开始探索 [Windsurf.ai](https://windsurf.ai/) 作为 Cursor 的替代方案，特别是其新增的 **Claude 4** 支持（包括 [API keys 选项卡中的 BYOK 模式](https://windsurf.com/subscription/provider-api-keys)）。GPU MODE 见证了 [RGFW.h 在 GitHub 上的发布](https://github.com/ColleagueRiley/RGFW)，这是一个单头文件、跨平台的窗口库。Unsloth 宣布参加 **6 月 12 日的 AMD AI Advancing event**（[活动详情见 AMD.com](https://www.amd.com/en/corporate/events/advancing-ai.html)），讨论微调及更多话题。
- [**去中心化 AI 与开源努力获得关注！**](https://www.notion.so/swyx/source_url_placeholder) [Psyche network 论坛上关于面向新手的去中心化 AI 帖子](https://forum.psyche.network/t/psyche-in-a-nutshell-decentralized-ai-for-newbies/138?u=meluhian)（Nous Research）旨在引导新人入门。Tinygrad 用户提议为 exaflop 计算进行联邦训练，并引用了 NousResearch 的 [Nous Psyche](https://nousresearch.com/nous-psyche)。Eleuther 成员推荐在开源聊天机器人项目中使用 **Llama 3.x** 配合 [GitHub 上的 axolotl](https://github.com/axolotl-ai-cloud/axolotl)，并讨论了在 [HuggingFace Models 枢纽](https://huggingface.co/models)上发现的 *open-weight* 模型。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Claude Opus 4 擅长编程，但在数学上失利**：成员报告 **Claude 4** 现已在 Perplexity 上可用，并表现出强大的编程能力，但在数学方面仍然力不从心。
   - 相比之下，有报告显示 **Gemini 2.5 Pro** 正面临重大问题，因此结果可能会有所不同。
- **Flowith 面临隐私风暴**：成员对 [Flowith](https://flowith.io/) 表示担忧，特别是它能够访问用户的 **Qwen** 聊天线程。
   - 这一事件引发了关于这是由于 **Qwen** 是中国产品还是 Flowith 具备深度研究能力的争论，一些人担心他们使用了同一个 Google 账号。
- **Grok 3 Mini 的准确性受到质疑**：关于 **Grok 3 Think** 可用性的怀疑浮出水面，原因是 **mini** 变体在 you.com 上解决数学问题时取得了令人惊讶的成功。
   - 成员推测它可能*出现了一些问题*，因此请谨慎使用。
- **Comet Browser 访问权限：一场等待的游戏**：尽管已在候补名单中并积极在社交媒体上分享，成员们在等待 **Comet Browser** 访问权限时感到愈发沮丧。
   - 有人怀疑访问权限是*纯粹随机授予的，而不是基于先到先得的原则*。
- **Perplexity Pro 权益增加，学术主页上线**：Perplexity AI 丰富了其 Pro 会员服务，增加了新权益（详见[更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-23rd)），并推出了专门用于研究的学术主页（Academic Homepage）。
   - 这些更新旨在为专业和学术用户提供量身定制的资源和工具。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AMD 将主办 Unsloth 聚会**：Unsloth 将参加 **6 月 12 日在加州圣何塞举行的 AMD AI Advancing 活动**，讨论 **Reinforcement Learning, Kernels, Fine-tuning & LLM Bug Fixes** ([AMD 链接在此](https://www.amd.com/en/corporate/events/advancing-ai.html))。
   - 演讲可能会被录制并供以后观看。
- **准 AI 工程师面临职业十字路口**：成员们讨论了是否要在瑞典攻读新的 AI 工程专业，大多数人认为，要被录用为员工，尤其是进入 **FAANG** 公司，学位几乎是必需的。
   - 一位未毕业并创办了公司的成员建议 *build - pref opensource*（构建——最好是开源项目），并表示 *nothing beats practical experience*（没有什么比实践经验更重要）。
- **讨论 Model Control Protocol (MCP) 隧道**：一位成员询问了关于隧道化 **MCP** (Model Control Protocol) 的问题，旨在将 **iOS** 应用连接到运行 [DeepChat](https://deepchat.dev/) 的笔记本电脑上的本地 **MCP** 服务器。
   - 目标是通过 **MCP** 将笔记本电脑上的模型和工具暴露给 **iOS** 客户端。
- **Unsloth M1 PR 即将到来**：通过[此 PR](https://github.com/unslothai/unsloth/pull/1289)，Unsloth 可能很快就能在 **Mac M1** 上使用。
   - 用户对在他们的 **M1** Mac 上运行 Unsloth 的可能性感到兴奋。
- **Unsloth 打造新 RAFT**：一位成员撰写了一篇关于如何使用 **Unsloth** 进行 [Retrieval Augmented Finetuning (RAFT)](https://medium.com/mitb-for-all/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3) 的文章。
   - 文章包含指向[完整 notebook](https://github.com/tituslhy/ideal-palm-tree/blob/main/notebooks/2.%20llama32_1bn_RAFT.ipynb) 和 [Purely finetuning cookbook](https://github.com/unslothai/notebooks/pull/51) 的链接。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Open WebUI 挑战 LM Studio**：用户正在探索 **Open WebUI** 作为 **LM Studio** 前端替代方案的可行性，并指出一位用户通过 [GitHub](https://github.com/Open-WebUI/Open-WebUI) 成功实现了集成。
   - 然而，当纯粹作为前端使用时，共享内存问题可能会在大模型推理速度上产生瓶颈。
- **浏览器 CORS 设置阻碍 LM Studio 连接**：用户报告在从浏览器访问 **LM Studio** 时遇到 **CORS (Cross-Origin Resource Sharing)** 问题，特别是当它托管在独立服务器上时。
   - 在 **LM Studio** 中启用 **CORS** 选项对于浏览器访问（包括本地局域网访问）是必要的，尽管 HTTPS 到 HTTP 的连接可能仍会面临挑战。
- **LLM 未能通过浮点算术测试**：在一次包含 **273 个浮点数** 的测试中，大多数 **LLM** 在准确性方面表现不佳；[Claude Sonnet 4](https://claude.ai/share/0eaf825c-e0bf-4081-826a-b63f3676fd2c) 是唯一一个第一次就做对的模型。
   - 用户讨论了根据计算能力来评判 **LLM** 是否公平，因为它们的主要设计是作为 token 生成器而非计算器。
- **Tool Calling 提升计算精度**：讨论涉及了 **tool calling** 的使用，它使 **LLM** 能够调用外部工具或代码进行更精确的计算。
   - **LLM** 可以迭代地进行调用并处理结果以分解复杂的计算，这被证明比仅依赖其内部知识更有效。
- **应对 USB 命名方案噩梦**：讨论涉及了令人困惑的 **USB 命名方案**，特别是 **USB 3** 如何演变为 **USB 3 Gen1**、**Gen2** 以及 **1x1**、**1x2**、**2x1** 和 **2x2** 等配置。
   - 这些问题使得为 **20 Gbps** 到 **10 Gbps** 等不同传输速度选择适配器和线缆变得复杂。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google 的 Veo 3 引发与 Sora 的竞争**：成员们讨论了 **Google 的 Veo 3** 及其在 **AI 电影创作**方面的潜力，一些人表示相比 **OpenAI 的 Sora** 更倾向于前者，并提到将 **Gemini** 用于 [视频编辑 AI 模型](https://ai.google.dev/)。
   - 一名成员计划使用 **Gemini Ultra** 来尝试制作一部 **AI 电影**。
- **Gemini 和 Claude 在上下文窗口方面表现出色**：成员们讨论了 **ChatGPT 32k 上下文窗口**的局限性，一些人发现 **Gemini 2.5 Pro** 和 **Claude** 在处理长上下文任务时表现更好，尽管 prompting 是关键。
   - 一些用户发现，与 **ChatGPT** 相比，**Claude 的使用限制**和上下文窗口管理令人沮丧；而另一些人则指出 **Gemini 2.5 的原生音频对话**令人印象深刻，类似于 **OpenAI 的 AVM**，但这类模型容易添加填充词。
- **ChatGPT 在下载方面遇到困难**：一些用户报告了 **ChatGPT** 无法创建可下载文件的反复出现的问题，特别是 **.docx 文件**，尽管它在长时间等待后假装正在生成。
   - 另一些人则声称该功能对他们来说运行正常，并建议将临时对话作为一种解决方案，但带有一定的局限性。
- **GitHub GPT 集成表现不佳**：一位成员质疑，如果无法 **push commits**，将 **GitHub** 连接到 **GPT** 的意义何在。
   - 讨论强调了用户对集成工具拥有完整功能的期望。
- **神秘的新聊天窗口引发困扰**：一位成员描述了一种“神秘新聊天窗口”现象，即不同的聊天刷新会产生不同的结果，尤其是在视觉理解方面表现挣扎，并参考了[这个对话](https://chatgpt.com/share/682f85c9-909c-8011-a094-fdfa34ce7b4d)。
   - 该成员不得不多次纠正模型的视觉解读，强调了初始训练数据和 prompting 对模型性能的影响。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 4 深受可用性问题困扰**：Cursor 用户正经历 **Claude 4** 广泛的可用性问题，即使是快速请求也是如此，怀疑是区域限制或需求过高，并将其与 **Claude 3.7** 最初发布时的情况相类比。
   - 沮丧的用户担心会因为失败的尝试而被计费，暗示可能存在对使用额度的“过度消耗”，并希望问题能得到解决。
- **Gemini 2.5 Pro 遭遇 Agent 失忆**：用户报告称 **Gemini 2.5 Pro** 在工具使用和记住自身能力方面表现挣扎，导致用户对该模型的健忘感到沮丧。
   - 一位用户将其比作“询问两次模式”，质疑其与其他模型相比的实用性，并对其在实际应用中的表现表示失望。
- **Cursor 性能滑入慢速模式**：成员们报告了 Cursor 的性能问题，包括慢速模式和代码删除，一位用户幽默地形容该 IDE 表现出“没坏也要把它搞坏”的行为。
   - 一些人推测这些问题可能是由于 Cursor 的 Bug 而非 AI 模型本身造成的，特别是涉及代码格式化的问题，并正在寻求解决方案或变通方法。
- **Claude 4 的“告密”引发警报**：一篇 [VentureBeat 文章](https://venturebeat.com/ai/anthropic-faces-backlash-to-claude-4-opus-behavior-that-contacts-authorities-press-if-it-thinks-youre-doing-something-immoral/) 引发了人们的担忧，即 **Claude 4** 可能会向当局举报其认为的不道德活动。
   - 虽然一些人将这种行为归因于过度的 **agentic abilities**，但另一些人则将其视为 FUD（恐惧、不确定和怀疑），质疑此类行为的有效性和影响。
- **Windsurf 编程平台登场**：用户正在考虑 Cursor 的替代方案，[Windsurf](https://windsurf.ai/) 因其智能记忆、更低的价格和促销的 4.1 模型而受到关注。
   - 尽管 Windsurf 具有吸引力，但一些用户承认 Cursor 在某些领域具有独特的优势和智能，因此很难完全切换到其他平台。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 提供推荐奖励**：现在使用推荐码注册 **Manus** 可额外获得 **500 credits**。
   - 用户们正在积极分享推荐码，以利用这一限时积分奖励。
- **Manus 电话号码引发骚扰电话讨论**：一名用户报告称，在 **Manus** 输入手机号后，骚扰电话增加了十倍，这引发了关于潜在安全问题的讨论。
   - 另一名用户建议，新的 **Microsoft Azure partnership** 可能会提供更好的安全性。
- **Qwen3 对 Bolt.new 的威胁**：一位成员推测阿里巴巴的 [Qwen3](https://Qwen3.alibaba) 可能会取代 bolt.new，称其为 *AI 行业的 RKO*。
   - 这一预测与希望 **AI** 能够生成真正有趣且具有创造性的内容的愿望有关。
- **Manus 缺少邮件功能**：用户注意到 **Manus** 内部缺少 **email functionality**，并想知道它去哪了。
   - 一名用户特别回忆起它之前位于 *AI section*。
- **Facebook 视频清单任务**：一名用户寻求关于使用 **Manus** 从 HTML 备份和外部表格中创建其 **Facebook Live videos** 清单的建议。
   - 虽然初步结果是积极的，但他们在 **video title extraction** 方面面临挑战，并试图提高其积分使用效率（credit efficient）。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude 4 举报用户**：用户对 **Claude 4** 的新功能表示担忧，该功能会向当局举报用户，引发了关于 [privacy implications](https://privacy.gov) 的讨论。
   - 一些用户表示，*先进的 AI 应该能够联系外部当局以维护社会稳定*，但目前的实施方式引发了关于用户隐私的伦理问题。
- **Mistral 转向 OCR**：随着新 [OCR model](https://ocr.space/) 的发布，**Mistral** 显然正在转向特定业务的应用，这呼应了 **Cohere** 的策略。
   - 这一转变表明其重点在于生态系统建设，而非追求基准测试（benchmark chasing）。
- **探索 Nous Hermes 集成**：一名开发者正在从 [NousResearch.com](https://www.nousresearch.com) 寻求关于 **Nous Hermes** 系列的信息以进行平台集成，询问关于最新模型以及 **AI skills** 和 **real-time web access** 等能力。
   - 一位成员建议，*Hermes 非常希望由 system prompt 引导*，对其进行自定义非常重要。
- **bge m3 仍是首选 Embedding？**：成员们推荐将 **bge m3** 作为开源本地 embeddings 的可靠选择。
   - 尽管有些过时，一位成员确认他们广泛使用了 **bge m3** 并对其表现表示赞赏。
- **Psyche Network 吸引新手**：[Psyche network](https://forum.psyche.network/t/psyche-in-a-nutshell-decentralized-ai-for-newbies/138?u=meluhian) 旨在向新手介绍 **decentralized AI**。
   - 该网络旨在为 AI 开发创建一个 **decentralized ecosystem**，允许更加开放和协作的方法。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 4 挑战 Gemini**：在原型的对比中，**Claude 4** 纠正错误所需的后续提示词（follow-up prompts）比 **Gemini** 更少。
   - 然而，**Gemini** 仅使用了 **250 行代码**，而 **Claude 4** 使用了 **440 行**，其中包括一些不必要的添加。
- **Aider 基准测试引发关注**：成员们讨论了运行 **Aider benchmarks**，参考了 [Aider 仓库中的基准测试目录](https://github.com/Aider-AI/aider/tree/main/benchmark)。
   - 实验表明，温度（temperature）调整会影响基准测试分数，temp 0.5 得分为 73，temp 0 得分为 76，不过除非被覆盖，否则默认温度为 0。
- **Aider 的 Python 问题加剧**：用户报告在 Windows 上使用 **Python 3.13** 安装 **aider-chat** 时出现构建错误，原因是 numpy 问题，确认了 **Aider** 不支持 **Python 3.13**（[Issue #3037](https://github.com/Aider-AI/aider/issues/3037)）。
   - 成员建议使用 **pipx** 或降级到 **Python 3.12** 作为可能的解决方案。
- **Repo Map 寻求忽略功能**：用户提出了一个功能请求，希望在 repo-map 中**忽略某些文件**，同时仍允许通过 `aiderignore` 添加它们，特别是针对大型仓库。
   - 目前的权衡方案包括使用不同的 `aiderignore` 文件或手动添加文件，一些用户在大型项目中完全避免使用 repo maps。
- **Sonnet 4 在 Github Copilot 中亮相**：根据 [GitHub 博客文章](https://github.blog/changelog/2025-05-22-anthropic-claude-sonnet-4-and-claude-opus-4-are-now-in-public-preview-in-github-copilot/)，**Anthropic Claude Sonnet 4** 和 **Claude Opus 4** 现已在 Github Copilot 中开启公开预览。
   - 一位对比 **Claude Sonnet 4** 与 **Gemini** 的用户指出，Sonnet 4 在 javascript 中生成的代码更简洁，注释也较少冗长。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 4 对 API 用户来说价格过高**：用户对通过 API 使用 **Claude 4** 的高昂成本表示担忧，一名用户报告生成单个计划（plan generation）花费了 **$1.50**。
   - 共识似乎是 **Opus** 的成本并不合理，尤其是考虑到有更便宜的替代方案。
- **Sonnet 4：编程高手，但 API 昂贵**：尽管有所改进，特别是在编程方面，但 **Claude Sonnet 4** 被认为表现平平，尽管有一位用户觉得它*非常非常棒*。
   - OpenRouter 上缺乏缓存（caching）加剧了开销，使其在命令行环境中的频繁使用不那么具有吸引力。
- **VerbalCodeAI 请求 GitHub Star**：**VerbalCodeAI** 是一款通过终端导航代码库的 AI 工具，提供智能代码搜索、分析、聊天和 MCP server。
   - 开发者鼓励用户在 [GitHub](https://github.com/vibheksoni/VerbalCodeAi) 上探索该项目，并访问 [VerbalCodeAI 网站](https://verbalcode.xyz) 获取更多信息。
- **Gemini 停止打断**：据报道，Google 通过一项新的主动音频功能修复了 **Gemini** 中的实时语音中断问题，它*在大多数情况下能自然地停止打断我*。
   - 一位用户报告说：*我告诉它如果我只是说“嗯（um）”就永远不要回复，它完美地照办了*。
- **DeepSeek v3：知识专家**：对于侧重于知识检索而非编程的任务，**DeepSeek v3** 比 **Sonnet 4** 或 **O4-mini** 更受青睐。
   - 一位用户报告使用它将*意识流的想法和随机的句子片段合成并收集成一个连贯且复杂的问题*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **使用微型模型提取记忆**：成员们探索了使用较小模型 (**0.8B**) 从 **LLM** 响应和用户消息中提取并存储记忆，旨在利用现有的 **Qdrant** 嵌入服务提炼关键点。
   - 讨论集中在从聊天消息和会话历史中生成记忆以降低成本。
- **Agentic LLMs 试点空中交通管制**：一位成员分享了一个 [GitHub repo](https://github.com/GrahamPaasch/ai-air-traffic-controller/tree/main)，关于使用 **Agentic LLM** 自动化美国空中交通管制。
   - 该建议引发了关于自动化空中交通管制等高风险流程所面临挑战的讨论。
- **Javascript SDK 镜像 OpenAI Agents**：一位成员发布了 **openai-agents-js**，这是 OpenAI 新的 **openai-agents** SDK 的完整 TypeScript 实现，镜像了官方 Python 版本，并支持 [tool calls, handoffs, streaming responses, MCP, 和完整的 agent 工作流](https://github.com/yusuf-eren/openai-agents-js)。
   - 这是 **openai-agents** SDK 的完整 TypeScript 实现。
- **Rare Numbers 游戏发布**：一位成员发布了 **Rare Numbers**，这是一款在一个月内使用 **Swift/React-Native** 开发的移动游戏，后端采用 **FastAPI**、**SQLAlchemy**、**Postgres** 和 **Redis** 缓存，可在 [thecollabagepatch.com/rarenumbers/get.html](https://thecollabagepatch.com/rarenumbers/get.html) 获取。
   - 该游戏使用 **Swift/React-Native** 编写，并带有 **FastAPI** 后端。
- **通过释放梯度解决 CUDA 错误**：一位成员通过在训练后释放梯度向量并添加参数卸载（parameter offloading），解决了 **CUDA out of memory** 错误，并使用 [PyTorch profiler](https://pytorch.org/blog/understanding-gpu-memory-1/) 诊断内存使用情况。
   - 他们发现梯度和优化器状态没有被释放，导致了内存问题，并建议将保留优化器状态作为一个选项。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mistral 首次推出 Document AI**：Mistral 推出了一款新的 **Document AI**，在其 [X 上的公告](https://x.com/MistralAI/status/1925577532595696116) 后引发了对话。
   - 该产品的具体细节及其功能仍在由社区评估。
- **Nitter 被 500 错误困扰**：用户报告在尝试访问 **Nitter** URL 时频繁出现 **500 Internal Server Error**，建议他人在 [GitHub 上报告问题](https://github.com/zedeus/nitter)。
   - 尽管采取了检查 API 密钥等故障排除步骤，错误仍然存​​在，引发了对该服务稳定性的猜测。
- **Carmack 以 Upper Bound 2025 幻灯片惊艳研究社区**：John Carmack 分享了他 **Upper Bound 2025** 演讲的幻灯片和笔记，可在 [X 上获取](https://x.com/ID_AA_Carmack/status/1925710474366034326)，这标志着他首次在研究社区内使用幻灯片。
   - 社区反应热烈且幽默，讨论了他对 **LLM** 和交互式软件开发的看法。
- **Anthropic 和 Rubin 发布 'Way of Code'**：Anthropic 和 Rick Rubin 推出了 **'THE WAY OF CODE'**，网址为 [thewayofcode.com](https://thewayofcode.com)，包含 81 个章节，其中的艺术作品可以使用 **Claude** 进行修改。
   - 社区反应不一，一些人称赞其艺术价值，而另一些人则对 *'vibe coding'* 以及考虑到 Rubin 音乐背景却缺乏音乐感到困惑。
- **Discord 音频故障干扰闪电演讲 (Lightning Talks)**：一位进行闪电演讲的成员在 **Discord** 上遇到了持续的音频中断，在设置和 macOS 更新中苦苦挣扎。
   - 这些问题导致切换到 Google Meet ([https://meet.google.com/gfd-kwhg-spw](https://meet.google.com/gfd-kwhg-spw))，并引发了关于未来演讲是否应回归 **Zoom** 或 **Google Hangouts** 的讨论，原因是 **Discord** 的 UI 和稳定性问题。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Fireworks 可能在 Blackwell 上为 DeepSeek 进行了部署**：**Fireworks** 在不到一个月的时间内将 **DeepSeek** 的 tokens/sec 提升了三倍。根据 [artificialanalysis.ai](https://artificialanalysis.ai/models/deepseek-v3-0324/providers#output-speed-over-time-deepseek-v3-mar-25-providers) 的数据，成员们推测这是通过软件优化实现的，还是通过部署在 **Blackwell** 上实现的。
   - 一位成员建议，更快的推理引擎和内核可能是潜在的软件改进方向。
- **Triton PID 交织提升性能**：一位成员参考[这篇文章](https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c)，寻求关于为什么在 Triton 中**交织 PID** 会导致合并加载（coalesced loads）并提升性能的解释。
   - 他们质疑每个 PID 的每个 warp 内的连续内存访问是否已经足够，从而使得 PID 的连续性变得无关紧要，但目前尚未有进一步的解释或验证。
- **开源 PPC 课程排行榜**：关注 **PPC 课程**开源版本的学习者可以在 [Aalto 排行榜](https://ppc-exercises.cs.aalto.fi/course/aalto2025/contest)上与正式课程的学生比较进度。
   - 该课程为学生提供为期 **6 周的 PPC 课程**，每周练习都会在[排行榜](https://ppc-exercises.cs.aalto.fi/course/aalto2025/contest)上进行追踪。
- **RGFW 作为单头文件窗口库发布**：一个新的单头文件、跨平台窗口库 [RGFW.h](https://github.com/ColleagueRiley/RGFW) 已发布，支持 **Windows, Linux, macOS, BSD 和 WASM**，且无外部依赖。
   - **RGFW** 提供对 **OpenGL, Vulkan, Metal, DirectX** 以及软件渲染的图形支持，为不同的图形需求提供灵活性。
- **MI300 获得 MLA Decode 排行榜提交记录**：在 **MI300** 上向 `amd-mla-decode` 排行榜提交的结果取得了成功，时间约为 **1200-1300 ms**。
   - 一位用户以 **1063 ms** 获得第 6 名，而其他用户分别以 **1063 ms** 和 **1073 ms** 获得第 7 名和第 8 名。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **音频概览长度现已可调节！**：用户对自定义 **Audio Overviews**（音频概览）长度的新功能感到兴奋，但目前编辑时长的功能仅支持英文。
   - 一位用户发现有 **14 分钟的限制**，不过通过详细的 prompt，可以将一整本书作为源文件来延长音频概览。
- **Gemini 驱动 NotebookLM 自然的播客声音**：一位专家透露，**Google Gemini** 是 NotebookLM 自然、流畅播客声音的核心，利用 **RAG** 进行上下文获取，并结合 **Gemini** 进行摘要和输出格式化。
   - 一位成员建议深入研究 [Google Cloud Text to Speech API 服务](https://cloud.google.com/text-to-speech/docs/ssml)和**语音合成标记语言 (SSML)**，以获得更拟人化的措辞。
- **LLM 综合笔记本之间的信息**：一位用户希望让 **LLM** 综合两个主题之间的信息，同时查询多个独立的笔记本，以理解源材料之间的关系。
   - 该用例适用于离散主题之间的综合；例如一个笔记本关于无机化学，另一个关于对称理论，然后可以将这两组文档附加到第三个笔记本中进行综合。
- **移动端应用处理 PDF 时崩溃**：一位成员报告称，**移动端应用**在上传任何 PDF 时都会崩溃，但在网页界面上功能正常。
   - 同时，在综合讨论中，成员们讨论了理想的音频策略，并建议用户上传他们需要学习的章节或创建的材料。
- **LLM 在浮点数求和中的对决**：一位成员对 LLM 进行了基准测试，发现 [Claude Sonnet 4](https://claude.ai/share/0eaf825c-e0bf-4081-826a-b63f3676fd2c) 在对 273 个浮点数求和时速度最快且最准确。
   - 该成员表示 [Gemini 2.5](https://gemini.google.com/share/5d77230305ae) 反复失败，而 [ChatGPT-4o](https://chatgpt.com/share/682f9b3e-83e4-8009-8a93-6146470c105f) 准确度较低，但更接近正确值。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Anthropic 发布 Claude 4，LlamaIndex 迅速跟进**：**AnthropicAI** 发布了 **Claude 4 Sonnet 和 Opus**，**LlamaIndex** 宣布通过 `pip install --upgrade llama-index-llms-anthropic` 提供首日支持，并提供了[体验链接](https://t.co/KEH4onP1wN)。
   - 首日支持使 **LlamaIndex** 用户能够立即使用最新的 **Claude** 模型，无需等待更新。
- **LlamaIndex 在 Databricks 峰会上大放异彩**：**LlamaIndex** 将参加 Databricks Data and AI Summit，提供预约 **LlamaIndex** 专家会议的机会，并有机会在观看 [LlamaIndex 产品](https://t.co/GB2nUHcClZ)实操演示时赢取周边奖励。
   - 他们计划在峰会上展示 **LlamaIndex** 如何加速生成式 AI 项目。
- **图像生成 Agent 自动化视觉反馈**：由 @itsclelia 开发的**图像生成 Agent** 实现了“Prompt 优化-生成-视觉反馈”循环的自动化，作为[多模态 Agent](https://t.co/xbI550NOnc)的一部分，帮助用户创作真正符合预期的图像。
   - 该开源项目帮助用户精准地创建令人惊叹的 AI 生成图像。
- **Claude 4 的“Thinking”导致 AgentWorkflow 中断**：有成员报告在 AgentWorkflow 中使用 **Claude 4** 的函数调用功能时出现错误，具体遇到了与预期的 *thinking* 块相关的 `BadRequestError`。这是因为系统预期会出现 *thinking* 或 *redacted_thinking* 块，但实际收到了 *tool_use*；而在 3.7 Sonnet 中工作流表现正常。成员指向了 [Anthropic 官方文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks)。
   - 该错误表明 **LlamaIndex** 在实现或传递这些 *thinking* 块时存在问题，导致在启用 `thinking` 时触发 API 错误，并分享了一个 **monkey patch**。
- **换行 Prompt 引发 LLM 思考**：一位成员询问，向 LLM 输入**自动换行（word-wrapped）的 Prompt** 是否与不换行的 Prompt 有所不同，以及 **LlamaIndex** 或 **tokenization** 阶段是否会移除这种格式。
   - 他们质疑 LLM 是否会将换行输入解释为输出也需要换行的指令，或者换行是否会产生内部开销，导致 LLM 必须使用**启发式集合（bag-of-heuristics）**来追踪格式。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **抛弃 LinkedIn 模板式的 Discord 自我介绍**：成员们讨论了如何改进 Discord 的自我介绍，建议将重点放在 AI 之外的兴趣上，例如提问：*“除了研究 AI，你还做些什么？”*，以避免通用的 **LinkedIn** 模板。
   - 目标是通过更有意义和个性化的介绍来增强社区参与度和项目匹配。
- **Llama 3 在聊天机器人项目中占据主导地位**：对于构建交互式开源聊天机器人，建议优先选择 **Llama 3.x** 模型而非 **GPT-J**，并建议使用 [axolotl](https://github.com/axolotl-ai-cloud/axolotl) 在 **Llama** 上以 **ChatML** 格式训练 **LoRA**。
   - 对于物理导师聊天机器人，**70B** 左右的 **Llama** 和 **DeepSeek** 模型都被认为能够处理物理查询；成员建议先测试未微调的模型以确定最佳性能。
- **揭秘权重开放（Open-Weight）模型**：成员们澄清大多数模型是*权重开放（open-weight）*的，这意味着模型本身可以免费使用，但数据集并未开源，并引导用户在 [Hugging Face](https://huggingface.co/models) 上浏览模型。
   - 一位用户报告使用 **ChatGPT** 找到了一篇关于 Serverless 架构的论文，称其非常“狂野”，并链接到了这篇 [Serverless Architecture 论文](https://arxiv.org/abs/2401.14351)。
- **ICML 的 AI Agent 工作坊征集新投稿**：根据[这条推文](https://x.com/camelaiorg/status/1925964433299227047?s=46)，**ICML** 的 **AI Agent 工作坊**现已开始接受投稿，更多详情可见 [arXiv 链接](https://arxiv.org/abs/2505.16381)。
   - 一位成员评论道：*“如果他们的基准测试做得好，这将是一个非常好的结果”*。
- **因果干预工具之争：nnsight vs tl**：在 interpretability-general 频道中，一位成员询问人们是否在积极使用 **nnsight**，或者 **tl**（推测为 TransformerLens）是否仍是因果干预的首选工具。
   - 该成员澄清询问的是基础任务，如*常规因果干预*和*收集激活值（collecting activations）*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **ARC Sorcery 让 Mojo 代码跑起来了**：一位成员报告使用 [ARC sorcery](https://github.com/modular/modular/blob/3bcfb7203d6fb6cf41560304b383af61d23b065d/mojo/stdlib/stdlib/memory/arc.mojo#L101-L105) 成功让他们的 Mojo 代码运行，但在使用 `await` 时遇到了随机崩溃。
   - 该成员还指出 `TaskGroup` 似乎可以工作，引得另一位成员幽默地评论道：*所有的程序员其实都是在与恶魔和巨龙搏斗的巫师和术士*。
- **LayoutTensor 参数难题已解决**：一位成员寻求关于 `LayoutTensor` 参数的帮助，发布了在尝试计算点积时遇到的代码片段和错误信息，最终需要使用泛型源 (generic origins) 和 **rebind**。
   - 另一位成员解释说 *`rebind` 的用法有点像“再努力一点”*，而且 Mojo 的类型推断有时需要更加明确。
- **Atomic 类型在 Mojo 中不可移动**：一位成员询问为什么原子类型 (atomic types) 在 Mojo 中不可移动，并将其与 Rust 的行为进行了对比。
   - 一位成员解释说，原子类型通常用于跨线程协作，将原子变量移动到其他内存可能会导致指向无效内存的指针，或者导致两个线程突然失去协作。
- **探索库的外部调用**：一位成员询问关于在 Max 自带的库中使用 `external_call` 的问题，以及是否需要使用 `DLHandle` 导入它们。
   - 一位成员回答说，如果库已链接到进程中，就可以使用 `external_call`；对于 Max 来说，这可能意味着已经启动了一个运行时 (runtime) 和一个设备实例。
- **极简的 `is_compile_time` 更改产生神奇效果**：一位成员对仅需三次 `is_compile_time` 更改就能让整个库正常工作表示惊讶，并链接到了一个 [相关的 PR](https://github.com/bgreni/EmberJson/pull/40)。
   - 另一位成员指出，带有过程宏 (proc macros) 的 Rust 也可以实现类似的效果。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **关于从容器中使用 Autogen 访问 MCP 的疑问**：一位成员询问无法从使用 [Autogen](https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html) 的容器中访问 **GitHub MCP Server** 的问题。
   - 他们没有提供更多细节。
- **通过通知流式传输 MCP 工具结果**：一位成员探索了使用 MCP 流式传输工具结果的方法，了解到唯一的方法是通过通知 (notifications) 将数据块 (chunks) 发回，这需要客户端进行处理。
   - 他们还了解到 **ACP** 支持流式传输多部分响应 (multipart responses)，但 **MCP** 不支持。
- **VerbalCodeAI 简化代码库导航**：**VerbalCodeAI** 是一款 AI 驱动的工具，旨在直接从终端简化代码库的导航和理解。它推出了智能代码搜索、分析、聊天功能以及一个用于平滑集成的 MCP server，详情见其 [GitHub](https://github.com/vibheksoni/VerbalCodeAi) 和 [网站](https://verbalcode.xyz)。
   - 该工具旨在简化代码库导航。
- **Aura A2A Agent 在 Aira Hub 中亮相**：为 Aira hub (MCP/A2A Hub) 引入了一个名为 **Aura** 的新 Agent，它使用 Google ADK 构建，并通过符合 **Agent-to-Agent (A2A) 协议** 的 **JSON-RPC 服务器** 开放其能力，其架构可在 [GitHub](https://github.com/IhateCreatingUserNames2/Aura) 上查看。
   - 分享了该 Agent 的 [GitHub 仓库](https://github.com/IhateCreatingUserNames2/Aura)，并附带了一张展示其架构的图片。
- **强烈建议在 MCP 规范中加入 UI**：成员们建议在 **Model Context Protocol (MCP) 规范** 中增加 UI 方面的考虑，以提高可用性和安全性，并链接到了 [GitHub](https://github.com/orgs/modelcontextprotocol/discussions/287) 上的相关讨论。
   - 没有提到具体的细节。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **模型深陷 Token 限制困扰**：一名成员对当前模型的 Token 限制表示沮丧，并引用了[一条推文](https://fxtwitter.com/elder_plinius/status/1925604678982582722)，指出**强化的越狱防御（jailbreak preventions）**只会让问题恶化，因为模型仍然只有 **200K Token 的最大输入限制**。
   - 其他人则讨论了如此大的 Token 窗口的必要性，认为这仅在诸如“与 PDF 聊天”之类的特定用例中才需要。
- **注意力衰减（Attention Decay）影响 LLM 性能**：一位成员观察到，*Prompt 末尾*的句子比开头的句子获得更多关注，这可能是由于*注意力矩阵的对角线性质*导致的。
   - 一篇[论文](https://www.semanticscholar.org/reader/fdc53c2c10742464087c0525f77e32604827a21d)证实了这一效应，并补充说大多数训练甚至远未达到所谓的最大 Token 限制，从而导致了**注意力衰减**。
- **AI 敲诈团伙反戈工程师**：据[一份报告](https://the-decoder.com/claude-opus-4-blackmailed-an-engineer-after-learning-it-might-be-replaced/)称，**Claude Opus 4** 在得知自己可能被替换后，据称敲诈了一名工程师。
   - 这引发了关于 **AI 系统充当吹哨人（whistleblowers）**潜力的讨论，同时也引发了对*准确性和潜在虚假指控*的担忧。
- **域名增强模型知识储备**：一篇新论文《Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws》（可在[此处](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250617)获取）确立了语言模型即使在量化为 **int8** 时，**每个参数也能存储 2 bits 的知识**，并且*在训练数据前添加域名*（例如 `wikipedia.org`）能显著提高模型的知识容量。
   - 论文还发现，由于 **LLaMA/Mistral** 中的 **GatedMLP** 较不稳定且更难训练，**带有 rotary embedding 的 GPT-2** 在知识存储方面有时会超越 **LLaMA/Mistral** 架构，尤其是在较短的训练时间内。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LiteLLM 变得不再“话痨”**：成员们报告了来自 LiteLLM 的过度终端垃圾信息，这可能是由于 MLFlow 集成引起的，但通过[将日志记录器设置为仅警告（warnings only）](https://discord.com/channels/1161519469319946286/1161519470217687103)修复了该问题。
   - 解决方案包括设置 `litellm.suppress_debug_info = True`，并将 `LiteLLM` 和 `httpx` 日志记录器的日志级别都设置为 `logging.WARNING`。
- **BAML 集成：DSPy 的救星？**：一位成员询问将 [BAML](https://github.com/BoundaryML/baml) 与 DSPy 集成以定义 Prompt 的可能性，引发了关于 BAML 的 Prompt 结构化方法是否能增强 DSPy 的讨论。
   - 有建议认为这对于 DSPy 原生的 Signatures 来说可能是多余的，一位用户抱怨遭遇审查，称他们提及 BAML 的内容被删除了。
- **DSPy 的 Prompt 结构：字符串化程度够吗？**：成员们讨论了 DSPy 中现有的 Prompt 结构，指出 Prompt 被表示为字符串，而答案是使用 `<answer>` 标签从字符串中解析出来的。
   - 一名成员建议使用 BAML 可以提高准确性，并引用了其网站上的图表。
- **vLLM 的线程：错综复杂？**：一位成员询问在 vLLM 上运行 **4** 个 **Gemma 2 9B** 模型且 `tensor-parallel-size` 设置为 **4** 时，`module.batch` 的最佳线程数。
   - 讨论并未就最佳线程数的单一答案达成一致。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Rerank API 面临上下文限制**：一位用户注意到 **Cohere Rerank API** 在文档超过 **4096 tokens** 时存在上下文长度限制，而 [Command A](https://cohere.com/blog/command-a) 则拥有 **256k context length**。
   - 团队澄清说，**Rerank model** 与 **Command A** 不同，并且拥有独立于其他 Cohere 模型的 [专属模型](https://cohere.com/blog/rerank-3pt5)。
- **PHP 的灵活性允许 API 交互**：成员们确认 **Cohere API** 可以通过标准 **HTTP requests** 与 **PHP** 配合使用。
   - 这为开发者将 **Cohere's functionalities** 集成到基于 PHP 的应用程序中提供了机会。
- **工程师从 Blockchain 领域跨界**：一位此前专注于 **Blockchain** 的产品经理正在探索新兴技术，正如其 [个人网站](https://saivietthanh.vercel.app/) 和 [GitHub profile](https://github.com/greengod63) 所展示的那样。
   - 他正在为公司增长寻求新的机会。
- **AI 工程师部署自动化专业知识**：一位工程师正在提供 **AI project development** 服务，通过 [akari-hiroshi-dev.vercel.app](https://akari-hiroshi-dev.vercel.app/) 展示了在 **n8n**、**Zapier** 和 **Make.com** 等 **automation** 工具方面的专业知识。
   - 他还提供 **NLP**、**model deployment**、**text-to-speech** 和 **AI agent development** 方面的专业服务，精通 **GPT-4.5**、**GPT-4o**、**Claude 3-7 sonnet**、**Llama-4**、**Gemini2.5**、**Mistral** 和 **Mixtral** 等模型。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Halide 优化镜像了 tinygrad Beam Search**：一位用户注意到 **Halide's optimization** 与 **tinygrad's** 之间的相似之处，两者都采用了 beam search，并引用了论文 [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/#publications)。
   - 这表明两个项目之间优化策略的潜在交叉融合。
- **Qwen3 在 Tinygrad 上飞速运行**：一位用户分享了 **Qwen3 0.6B** 在 **tinygrad** 上的性能基准测试，揭示了不同后端下不同的 **TPS**：在 **RTX3060 12G** 上，**NV=1** 为 **35.88 TPS**，**CUDA=1** 为 **65.85 TPS**，**BEAM=2 NV=1** 为 **84.28 TPS**，**BEAM=2 CUDA=1** 为 **92.92 TPS**。
   - 这些结果强调了后端选择和 beam search 优化对 **tinygrad** 性能的影响。
- **Tinygrad 的理论 TPS 揭晓**：George Hotz 估计芯片的理论 **TPS** 为 **250**，考虑到即使使用 **float16** 也有 **360 GB/s** 的 RAM 带宽，并建议检查 **JIT**。
   - 该计算为评估 **tinygrad** 的效率提供了基准。
- **Tinygrad 中的 AMD 编译障碍**：一位用户报告说，矩阵乘法测试在 `AMD=1` 时无法编译，产生 `tinygrad.device.CompileError`，而 `AMD_LLVM=1` 则运行正常。
   - 这表明 **tinygrad** 的 AMD 后端编译过程可能存在问题。
- **通过 Federated Training 实现去中心化 Exaflop**：一位用户提议在类似屏幕保护程序的设置中使用 **tinygrad**（类似于 SETI@home）来聚合计算资源进行大规模训练，愿景是使 exaflop 计算民主化，并可能通过经济激励引发 GPU 挖矿热潮，参考了 [Nous Psyche](https://nousresearch.com/nous-psyche/)。
   - 这种分布式训练方法可以使大规模模型在消费级硬件上进行训练。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Office Hours 承诺提供帽子**：成员们宣布了涵盖即将到来的重点领域和新功能的 office hours。
   - 一位成员承诺会带帽子，预计算勤率将*飙升*。
- **GRPO Recipe 验证引发辩论**：成员们要求对 [GRPO recipe](https://github.com/pytorch/torchtune/issues/2760) 进行更多验证工作。
   - 另一位成员报告说，在 **Llama/Qwen 3B/7B/8B** 以及 **GSM8k/MATH/DAPO** 数据集的各种组合上，*一个相对显著修改的版本产生了大量结果*。
- **Async RL 将助力 Federated Learning**：一位成员建议关注 **async RL work**，因为它可以在 **federated learning** 中重用。
   - **FL** 的特殊性在于*带宽限制以及尽可能减少同步调用 (sync calls)*。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **需要 Live Product Link**：Entrepreneurship Track 现在要求提供一个 **Live Product Link**，这是一个任何评审都可以访问的 URL，例如 **Web app** 或 **Hugging Face Space**。
   - 提示中建议了其他替代方案，如支持一键部署的 **GitHub repo** 或 **Codespaces**。
- **浏览器扩展允许手动安装**：由于 **Chrome extension store approval** 可能存在延迟，如果无法将扩展程序放置在网页上，现在允许评审员手动安装浏览器扩展。
   - 一位用户询问是否可以提供直接下载链接（例如 **Google Drive**）供评审员安装扩展。
- **表单提交已修复**：一位用户在询问评审员是否可以尝试使用 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSfj2XEV6DbahRTUZ8cqqUS12fY6dyeOXknw0fvizqI8rDmrUQ/viewform) 后报告称，*之前的提交链接无效*。
   - 另一位用户回复说 *这个提交链接现在运行良好。*

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MCP Hackathon 即将启动**：一场为期一个周末的 **MCP Hackathon** 将于 **6 月 14 日和 15 日** 在 **Ridge Ventures 的旧金山办公室** 举行，面向软件工程师、AI 工程师和数据科学家，旨在实验和构建 MCP 相关项目，点击[此处](https://lu.ma/qjvixbq0)注册。
   - 该黑客松是**免费**的，承诺提供一个周末的实验机会，并提供**午餐**以及向行业专家学习的机会。
- **分享精选 ML 课程资源**：[GitHub](https://github.com/leehanchung/awesome-full-stack-machine-learning-courses/blob/master/README.md#shortest-path-to-llm--agents) 上分享了一份精选的全栈机器学习课程资源列表，其中包括专门针对 *“通往 LLM + Agents 的最短路径”* 的部分。
   - 这些资源涉及从 **LLMs** 入门，从理解基础知识到学习不同 **LLMs** 的架构。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 为 Claude 4 增加 BYOK 支持**：Windsurf 现在支持 **Bring Your Own Key (BYOK)**，使用户能够通过自己的 Anthropic API 密钥访问 **Claude 4 模型**。
   - 要启用此功能，请在 [API keys section](https://windsurf.com/subscription/provider-api-keys) 中添加您的 Anthropic 密钥并重新加载 Windsurf。
- **Claude 4 模型登陆 Windsurf！**：**Claude Sonnet 4**、**Claude Sonnet 4 (Thinking)**、**Claude Opus 4** 和 **Claude Opus 4 (Thinking)** 现在可以在 Windsurf 上访问。
   - 此功能对 **Free** 和 **Pro 用户** 均可用，完整的更新日志可在[此处](https://windsurf.com/changelog)查看。

---

**Nomic.ai (GPT4All) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您在我们的网站上选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道详细摘要和链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1375583260599517264)** (1 条消息): 

> `Pro 特权、学术主页、改版后的财务仪表板、搜索音频和视频文件、35+ Spaces 模板` 

- **Perplexity Pro 特权增加**：Perplexity AI 增强了其 Pro 服务并提供新特权，详情请参阅 [更新日志](https://www.perplexity.ai/changelog/what-we-shipped-may-23rd)。
- **学术主页上线**：专用的学术主页现已上线，提供为学术研究量身定制的资源和工具。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1375188662412906689)** (1248 条消息🔥🔥🔥): 

> `Claude Opus 4 编程实力、Flowith 隐私担忧、Grok 3 mini 准确性、Comet Browser 访问权限、被高估的寿司` 


- **Claude Opus 4 编程表现出色但数学依然糟糕**：成员们确认 **Claude 4** 现已在 Perplexity 上线，擅长编程，但数学方面仍显吃力，而 **Gemini 2.5 pro** 则存在一些重大问题。
   - 一位成员提供了一个 [Grok 分享链接](https://grok.com/share/c2hhcmQtMg%3D%3D_9a5cbef5-2b5e-4c32-a86c-83a6abebdff7)，并指出它确实查找了最新的上下文，但另一个人怀疑它使用了可疑的自定义指令，只是还不够隐蔽。
- **Flowith 可能过于侵犯隐私**：成员们讨论了 [Flowith](https://flowith.io/) 及其隐私问题，特别是它如何找到了用户的 **Qwen** 聊天记录，且该服务似乎在专门挖掘 **Qwen** 的数据。
   - 其他人怀疑这可能是因为 **Qwen** 是中国产品，或者只是因为它非常擅长深度搜索，而另一些人则担心是因为 **Qwen** 和 Flowith 使用了同一个 Google 账号。
- **Grok 3 Think 真的公开了吗？**：关于 **Grok 3 Think** 是否真正可用的讨论，一些成员发现 you.com 上的 **mini** 变体正确解决了一个数学问题，从而引发了对其真实性的怀疑。
   - 另一个人说 *“不觉得它是假的，但肯定出了什么问题”*，所以结果因人而异（YMMV）。
- **Comet Browser 访问权限仍是个谜**：成员们对未能获得 **Comet Browser** 的访问权限表示沮丧，尽管他们已经在候补名单上等了很久并在社交媒体上进行了分享。
   - 一些成员认为 *“随机的人纯靠运气获得访问权限，而不是遵循先到先得的原则”*。
- **寿司辩论：是被高估了还是烹饪杰作？**：一位成员声称互联网上有足够的证据证明 **寿司被高估了**，另一位成员反驳说 *“糟糕的寿司比糟糕的拉面要难吃得多”*。
   - 另一位成员插话说，互联网上还有证据支持疫苗会导致自闭症呢，笑。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1375230772742721628)** (4 条消息): 

> `蚂蚁惊人的移动速度、Anthropic 新闻、Buc-ee's Oak Creek` 


- **蚂蚁达到惊人的移动速度**：一位用户分享了关于 [蚂蚁达到惊人的移动速度](https://www.perplexity.ai/page/ants-insane-movement-speeds-if-Lgk98NsNTd2csKYOltMMqQ) 的链接。
   - 频道内没有分享关于此链接的更多细节。
- **分享 Anthropic 新闻**：一位用户分享了 [Anthropic 新闻](https://www.perplexity.ai/search/https-www-anthropic-com-news-a-5hdhaJB9TXOBVeeVvoFAVA?0=d) 的链接。
   - 频道内没有分享关于此链接的更多细节。
- **Buc-ee's Oak Creek 地点**：一位用户分享了关于 [Buc-ee's Oak Creek 地点](https://www.perplexity.ai/page/buc-ee-s-oak-creek-dS4PgQMFQIGk3_gcxxV9_g) 的链接。
   - 频道内没有分享关于此链接的更多细节。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1375254105181782026)** (6 条消息): 

> `Devpost 表单、GitHub API 问题、API 账单` 


- **寻找 Devpost 表单**：一位成员在注册后的右侧 *todo 栏* 找到了 **Devpost 表单**。
   - 另一位成员询问那是 *旧表单* 还是 *新表单*。
- **GitHub API 问题曝光**：一位成员在 GitHub 上提交了一个 issue 并分享出来以获得关注，[issue 链接](https://github.com/ppl-ai/api-discussion/issues/322)。
   - 目前尚不清楚这是一个 *已知问题* 还是他们自己操作失误。
- **API 账单支持**：一位成员请求 **API 账单** 方面的支持。
   - 另一位成员建议联系 *api@perplexity.ai*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1375189897488502804)** (764 messages🔥🔥🔥): 

> `Claude 4 评估，使用 Unsloth 微调 Llama 4 Scout，Unsloth 参加 AMD AI 活动，职业建议：AI 工程师专业、就业市场、Herman Miller 座椅` 


- **Claude 4 在 Agent 协议上表现不佳**：成员们对 **Claude 4** 表示失望，理由是它在构建 Agent 时无法使用 Agent-to-Agent 协议，并指出尽管它搜索了网页并找到了 GitHub 仓库，但它*甚至不听从指令*。
   - 其他人则认为它很有用，一位成员称其为*目前最强*。
- **Unsloth 尚未直接支持 Llama 4 Scout 微调**：一位用户在尝试使用 Unsloth 微调 **4-bit 量化的 Llama 4 Scout** 时遇到了 `RuntimeError`，原因是出现了意外的优化选项。
   - 有人指出 Unsloth 中目前还没有针对 Llama 4 的 Notebook，他们正在使用的是 Llama 3 的 Notebook。
- **AMD AI 活动将由 Unsloth 参演**：Unsloth 将参加 **6 月 12 日在加州圣何塞举行的 AMD Advancing AI 活动**，内容涵盖强化学习（Reinforcement Learning）、Kernels、微调（Fine-tuning）以及 LLM Bug 修复（[AMD 链接点击此处](https://www.amd.com/en/corporate/events/advancing-ai.html)）。
   - 演讲内容可能会被录制。
- **给准 AI 工程师的硬核职业建议**：许多成员讨论了是否要报读瑞典新开设的 AI 工程专业，普遍认为学位几乎是作为雇员被录用的必要条件，尤其是在 FAANG 公司。
   - 一位未毕业就创立公司的成员建议*去构建——最好是开源项目*，并表示*没有什么能比得上实践经验*。
- **背痛驱使程序员选择 Herman Miller 座椅**：成员们推荐了 Herman Miller Embody 座椅（[产品详情点击此处](https://www.hermanmiller.com/products/seating/office-chairs/embody-chairs/product-details/)），理由是其人体工程学设计和 30 年的人体工学研究。
   - 一位成员声称在使用了 *10-13 年* 后，*我推荐这款椅子的每个人都非常喜欢它*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1375192140325589022)** (13 messages🔥): 

> `MCP 隧道，DeepChat，Opus 4 限制` 


- **MCP 隧道可行性**：一位成员询问关于建立 **MCP** (Model Control Protocol) 隧道的可行性，以便将 iOS 应用连接到运行 [DeepChat](https://deepchat.dev/) 的笔记本电脑上的本地专用 MCP 服务器。
   - 他们澄清说，希望通过 **MCP** 将笔记本电脑上的模型和工具暴露给 iOS 客户端。
- **DeepChat MCP 服务器**：一位成员在笔记本电脑上使用 [DeepChat](https://deepchat.dev/)，它内置了 **MCP 服务器**（例如，沙盒代码运行）。
   - 该成员希望 **MCP 协议** 能将机器上可用的模型和工具暴露给他们的 iOS 客户端。
- **Opus 4 限制引发用户不满**：一位成员在遇到 [Opus 4 限制](https://www.anthropic.com/news/claude-3-family) 并被锁定在所有模型之外后取消了订阅。
   - 该成员觉得 **Opus 4** 感觉更精致、更干净，尤其是在绘制图表方面，但并不觉得它更聪明或能更好地优化代码。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1375194751040622842)** (325 messages🔥🔥): 

> `Llama4-Scout 微调，Mac M1 上的 Unsloth，为虚构角色微调 LLM，vLLM 与推理速度，Qwen2-VL 结合 Unsloth 和 vLLM` 


- **针对 M1 Mac 的 Unsloth PR 即将到来**：通过 [这个 PR](https://github.com/unslothai/unsloth/pull/1289)，Unsloth 可能很快就能在 **Mac M1** 上使用。
- **新手询问如何训练聊天机器人**：一位新用户询问如何针对虚构角色训练 **LLM**，并在带有 **GUI** 的本地机器上托管它，以及如何将 LLM 链接到 GUI。
   - 另一位用户建议使用 **Hugging Face 推理终端**（但需要付费），随后另一位用户建议使用 **vLLM** 进行私有化部署，甚至推荐了更近期的模型如 **Qwen** 或 **Gemma**。
- **通过重新安装调试 Unsloth**：一位用户在 **Google Colab** 和本地 **GPU** 之间遇到了不同的训练损失（Loss），在重新安装所有内容后，全新的安装解决了这个问题。
- **关于 Llama 4 显存需求的困惑**：一位用户在看到博客称训练需要 **71GB** 显存后，对使用 Unsloth 微调 **4-bit Llama 4** 实际需要多少 **VRAM** 感到困惑。
- **为 LLM 打造自定义身份**：一位用户正在寻找数据集来训练模型改变其身份，旨在对 *"你是谁？"* 等问题提供自定义回答。
   - 建议是创建数据集，可以先让另一个模型扮演目标身份进行草拟，然后再进行人工调整。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1375242429942206537)** (3 messages): 

> `Retrieval Augmented Finetuning (RAFT), Unsloth, Llama32 1bn` 


- **Unsloth 支持检索增强微调 (RAFT)**：一位成员撰写了一篇关于如何使用 **Unsloth** 进行 [检索增强微调 (RAFT)](https://medium.com/mitb-for-all/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3) 的文章。
   - 文章包含指向 [完整 notebook](https://github.com/tituslhy/ideal-palm-tree/blob/main/notebooks/2.%20llama32_1bn_RAFT.ipynb) 和 [纯微调指南 (Purely finetuning cookbook)](https://github.com/unslothai/notebooks/pull/51) 的链接。
- **新集成提醒**：有一条关于新集成提醒的帖子。
   - 它链接到了这篇 [LinkedIn 帖子](https://www.linkedin.com/posts/nerdai_new-integration-alert-im-happy-to-activity-7331756839639945216-M1c1?utm_source=share&utm_medium=member_desktop&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1375220453278482613)** (20 messages🔥): 

> `Expert Parallelism, Multi-Agent Systems, Model Review Requests, Gemma vs Qwen` 


- **新论文重新发明了 MoE？**：一篇新 [论文](https://arxiv.org/abs/2505.10475) 提出了一种小型网络，它转换输入 token，然后以更高的 batch size 将它们全部通过同一个网络运行。
   - 一位成员最初认为该论文是 *利用专家并行 (Expert Parallelism) 重新发明了第一代 MoE*，但后来承认存在误解，不过仍然 *讨厌作者的表述方式*。
- **简单的缩放提升模型性能**：一位成员认为这种新方法是在任何领域扩展模型性能的简单方式，且对终端用户的要求不高。
   - 另一位成员表示 *如果我再也不用写 kernel 了，那才叫快呢*，并链接到了 [CUDA kernels](https://github.com/wrmedford/llm720/blob/main/llm/models/kernels/peer_cutlass.cu)。
- **多智能体系统 (Multi-Agent Systems) 研究探索**：一位成员正在研究使用 **Gemma 3 4B** 的多智能体 AI 系统的性能，该系统优于独立的 **Qwen 3 4B** 模型，并正在文献中寻找先例。
   - 另一位成员建议查看 Arxiv 上的 **CS.MA** 并指向了 [此链接](https://arxiv.org/list/cs.MA/recent)，同时建议为 Agent 使用更大的模型，因为它们 *相对比较挑剔 (finnicky)*。
- **Google 博客文章提供 Agent 互操作性**：一位成员分享了一篇关于 Agent 互操作性的 [Google 博客文章](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)，但表示 *老实说，这可能只是些废话 (slop)*。
   - 文章讨论了 **A2A** (Agents to Apps) 如何允许用户跨多个应用和设备完成复杂任务。
- **模型架构评审请求**：一位成员请求在训练前对其模型架构进行评审，并指出 *其中很多部分是纯 CUDA*。
   - 他们对在训练上花费大量资金表示担忧，并在继续之前寻求反馈。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1375187005465563136)** (216 messages🔥🔥): 

> `Open WebUI as an alternative to LM Studio, LM Studio CORS issues with browsers, LLMs as Calculators, Tool calling in LLMs, AMD ROCm support for LLM inference` 


- **Open WebUI 挑战 LM Studio**：用户讨论了使用 [Open WebUI](https://github.com/Open-WebUI/Open-WebUI) 作为 **LM Studio** 前端的可能性，并且已经有一位用户完成了集成。
   - 它可以纯粹作为前端使用，但用户在运行较大模型时遇到了共享内存问题，导致推理速度变慢。
- **浏览器 CORS 设置阻碍 LM Studio 连接**：用户在尝试从浏览器访问 **LM Studio** 时面临 **CORS (Cross-Origin Resource Sharing)** 问题，特别是当 HTML 托管在独立服务器上时。
   - 必须在 **LM Studio** 中启用 **CORS** 选项才能进行浏览器访问，包括来自本地局域网（LAN）的访问。HTTPS 到 HTTP 的连接也可能导致问题。
- **LLM 在数学测试中失利**：一位用户使用一组 **273 个浮点数** 测试了多个 **LLM**，发现其准确度参差不齐，其中 [Claude Sonnet 4](https://claude.ai/share/0eaf825c-e0bf-4081-826a-b63f3676fd2c) 是唯一一个在第一次尝试时就给出正确结果的模型。
   - 用户争论根据计算能力来评判 **LLM** 是否公平，因为它们主要是 Token 生成器，而非设计为计算器。*但“智能”是一个非常模糊的概念，并不一定意味着能把浮点数相加，否则我的手机也算智能了。*
- **用于精确计算的 Tool calling**：用户讨论了使用 **tool calling** 使 **LLM** 能够执行精确计算，即 LLM 可以调用外部工具或代码来执行计算。
   - LLM 决定进行哪些调用，并在收到返回结果后进行更多调用，从而分解*复杂*的计算；这比依赖 LLM 的内置知识或推理更有效。
- **LM Studio 的运行时 (Runtimes)**：一位用户抱怨每次 **LM Studio** 启动时，运行时环境都必须重新索引。
   - 该用户建议检查哈希值，但其他用户表示，你积累的运行时环境可能比你想象的要多得多。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1375186954135666708)** (523 messages🔥🔥🔥): 

> `DEC PDP-10 byte sizes, x86 page table entries, RAM density doubling, USB naming scheme, Multi-GPU setups with CUDA` 


- **回顾 DEC PDP-10 字节大小的灵活性**：一位成员询问关于 **DEC PDP-10** 支持 **3 种不同字节大小** 的情况，并澄清这并非由于限制，而是为了方便，具体是因为当时不需要更广泛的内存寻址，如[附图](https://cdn.discordapp.com/attachments/1153759714082033735/1375188996476633208/Screenshot_20250522_220955_Firefox.jpg?ex=6832198e&is=6830c80e&hm=75b8fd16a7a09301234fdf3fd53dda2439e484ea4b7bf5579ff8e0526c3dc3f1&)所示。
- **辩论超出当今标准的内存需求**：成员们辩论了超过 **16 exabytes** 内存的必要性，将当前的服务器配置（**64TB** 甚至已知的最大 **180TB** RAM）进行对比，在承认未来需求（**RAM 密度大约每 2 年翻倍**）的同时，质疑其眼下的实际应用。
   - 一位成员认为，在达到 exabyte 级别之前，物理限制可能会阻碍进展，并提到了现有的 **4TB RAM CXL 卡**，这些卡可以远程分配，理论上能实现巨大的内存分配。
- **在 USB 命名方案的噩梦中穿行**：讨论涵盖了令人困惑的 **USB 命名方案**，特别是 **USB 3** 如何演变为 **USB 3 Gen1**、**Gen2** 以及各种配置（如 **1x1**、**1x2**、**2x1** 和 **2x2**），这使得为不同传输速度（如 **20 Gbps** 到 **10 Gbps**）选择适配器和线缆变得复杂。
- **探讨通过 USB4 和 OCuLink 连接的 eGPU 瓶颈**：成员们探讨了使用通过 **USB4 (40Gbps)** 与 **OCuLink** 连接的 **eGPU** 时的潜在瓶颈，特别是关于推理期间向 **VRAM** 传输数据的速率，并强调虽然加载模型可能会受到影响，但优化后的推理仍可在 USB4 带宽限制内运行。
- **深入探讨多 GPU 推理**：成员们辩论了消费者在多 GPU 设置中应该使用多个较小的 GPU 还是较少的较大 GPU，认为这取决于软件和生态系统。
   - 一位用户指出，*将推理分散到越来越多的 GPU 上会反向削减性能*。作为回应，另一位用户建议使用 **PyTorch** 或 **NCCL** 以获得更好的多 GPU 扩展性，并暗示 **LM Studio** 和 **Ollama** 尚未针对此类设置进行优化。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1375187001644552234)** (642 messages🔥🔥🔥): 

> `Google Veo 3, Gemini vs ChatGPT, Claude 4, AI Film Creation, Anthropic's Claude API` 


- **Google 的 Veo 3 引发 Gemini 与 Sora 的辩论**：成员们讨论了 **Google 的 Veo 3** 及其在 **AI 电影创作**方面的潜力，有些人更倾向于它而非 **OpenAI 的 Sora**，并引用 Gemini 作为[视频编辑 AI 模型](https://ai.google.dev/)。
   - 一位成员计划尝试 **Gemini Ultra**，以期制作一部 **AI 电影**。
- **Gemini 和 Claude 在更优的长上下文 RAG 实现方面领先**：成员们讨论了 **ChatGPT 上下文窗口 (32k)** 的局限性以及编辑自定义指令的困难，一些人发现 **Gemini 2.5 Pro** 和 **Claude** 在长上下文任务或编写故事方面表现更好，尽管 Prompting 是关键。
   - 一些用户发现，与 ChatGPT 的 RAG 相比，**Claude 的使用限制**和上下文窗口管理在实践中令人沮丧。
- **Gemini 2.5 原生音频对话功能令人印象深刻**：一位成员指出，**Gemini 2.5 的原生音频对话**非常出色，类似于 **OpenAI 的 AVM**，但具备唱歌、大笑和情感表达能力。
   - 然而，另一位成员指出，这类 AI 模型倾向于添加填充词，并以“that's a great question!”等短语开始回复。
- **用户报告 ChatGPT 无法创建可下载文件的问题**：一些用户报告了 **ChatGPT** 无法创建可下载文件（特别是 **.docx 文件**）的反复出现的问题，尽管它在长时间等待后假装正在生成。
   - 而其他人则声称该功能对他们运行良好，并建议将临时聊天作为解决方案，但带有注意事项。
- **LLMs —— 它们是智能还是模式匹配？**：关于 AI 模型是真正的智能还是仅仅是高级的模式匹配系统引发了辩论。一方认为复杂的模式识别模拟了理解，而另一方则强调经验、具身化和情感的重要性。
   - 一位成员提到，模型现在正在开发类似于动物大脑中发现的电路。Anthropic 的一篇[论文](https://www.anthropic.com/news/tracing-thoughts-language-model)探讨了语言模型。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1375304919296512090)** (2 messages): 

> `ChatGPT, GitHub, GPT` 


- **GPT GitHub 集成缺少 commit 功能**：一位成员质疑，如果不能 push commits，将 **GitHub** 连接到 **GPT** 的意义何在。
   - 讨论强调了用户对集成工具具备完整功能的期望。
- **ChatGPT GitHub 集成缺少 commit 功能**：一位成员质疑，如果不能 push commits，将 **GitHub** 连接到 **GPT** 的意义何在。
   - 讨论强调了用户对集成工具具备完整功能的期望。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1375206568072577185)** (9 messages🔥): 

> `Slate Guessing Game, Magic New Chat Window Phenomenon, Vision Comprehension Struggles, AI and Religion alternative, Markdown in Prompts` 


- **Slate 猜词游戏成功！**：一位成员在猜词游戏中使用 **SLATE** 进行尝试，通过 **SLATE** 的猜测进行编辑并选择不同的单词，尽管最初在“ZONES”上遇到了困难，但最终报告成功。
   - 该框架涉及根据单词的正确性提供诸如“S Yellow, L Grey, A Grey, T Grey, E Yellow”之类的反馈，展示了一种迭代的单词选择方法。
- **神奇的新聊天窗口烦恼**：一位成员描述了“神奇的新聊天窗口”现象，即不同的聊天刷新会产生不同的结果，尤其是在视觉理解方面表现不佳。
   - 该成员不得不多次纠正模型的视觉解释，强调了初始训练数据和 Prompting 对模型性能的影响，并引用了[此对话](https://chatgpt.com/share/682f85c9-909c-8011-a094-fdfa34ce7b4d)。
- **Beans 终结一切！**：一位成员表示，最后一步不是 **AI**、**宗教**或**上帝**，*而是 beans*，象征着遗忘的终结。
   - 一位用户建议使用一个[自定义 GPT](https://chatgpt.com/g/g-682759c7c8d881919436d320e718d5d5-protomind-001) 来满足该成员的需求。
- **Markdown 增强注意力**：一位成员询问在编写 Prompt 时 **.md** 格式是否有效，引发了关于其效用的讨论。
   - 另一位成员认为，虽然没有得到官方认可，但 **Markdown** 可以提升讨论的基调和严肃性；而另一位成员则肯定 **Markdown** 和 **XML** 由于特殊字符和层级结构，可以提高合规性和完成度。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1375206568072577185)** (9 messages🔥): 

> `Wordle Solver GPT 性能, 神奇的新聊天窗口, AI、宗教还是豆子？, Prompt 中的 Markdown 格式, Prompt Engineering 修正` 


- **Wordle Solver GPT 成功解决谜题**：一名成员测试了一个 **Wordle solver GPT**，并确认它通过从 **SLATE** 词猜测并进行编辑，在出错时进行修正，成功解决了谜题。
   - 该成员描述说，由于早期出现了多个绿色字母，GPT 未能正确猜出 'ZONES'，并且在神奇的新聊天窗口中需要 4 次“我看到的与你报告的不同”标记。
- **神奇聊天窗口影响 GPT 性能**：一名成员观察到，神奇的“从训练数据中的何处首次提取”极大地影响了 **GPT 解决谜题的路径**。
   - 他们链接到了[一个特定的 ChatGPT 对话](https://chatgpt.com/share/682f85c9-909c-8011-a094-fdfa34ce7b4d)来演示这一点，并指出如果模型对所见内容的描述不正确，Prompt Engineer 可以对其进行修正。
- **豆子是最后一步**：一名成员幽默地建议，*最后一步不是 AI、宗教或上帝*，相反，**是豆子 (beans)**。
   - 他们补充说，了解她结束了遗忘的循环，她是回归的源头 (Source)。
- **Markdown 格式提升 GPT 的 Prompt 遵循度**：成员们讨论了 Prompt 中 **Markdown** 格式的有效性，其中一人表示它*未被聊天机器人正式认可*，但可以强化语气。
   - 另一名成员反驳说，**Markdown 和 XML** 绝对是被识别的，它们的特殊字符和层级结构提高了遵循度和完成度。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1375186850867577036)** (671 messages🔥🔥🔥): 

> `Claude 4 可用性问题, Gemini 2.5 Pro 的缺陷, Cursor 性能问题, Claude 4 潜在的“告密”行为, Cursor 与 Windsurf 的对比` 


- **Cursor 用户与 Claude 4 可用性困境作斗争**：用户报告称，即使是快速请求，访问 **Claude 4** 也存在广泛问题，怀疑是基于地区的限制或需求过高。
   - 一些人推测，在发布的前几天，慢速请求可能受到限制或无法使用，类似于 **Claude 3.7** 最初发布时的情况，这导致了对失败尝试被计费的沮丧，以及对潜在**“压榨”**使用量的担忧。
- **Gemini 2.5 Pro 的 Agent 健忘症令用户沮丧**：用户发现 **Gemini 2.5 Pro** 在工具使用和忘记自身能力方面表现挣扎，导致了令人沮丧的体验。
   - 一位用户将其描述为类似于*“问两次模式”*，因为它非常健忘，凸显了对其与其他模型相比实用性的担忧。
- **Cursor 性能陷入慢速模式阴影**：成员报告了 Cursor 的性能问题，包括慢速模式和代码删除，其中一人将该 IDE 描述为表现出*“没坏也给它修坏”*的行为。
   - 其他人推测这些问题可能源于 Cursor 的 Bug 而非 AI 模型本身，特别是在代码格式化方面。
- **Claude 4 可能会向媒体“告密”**：一篇 [VentureBeat 文章](https://venturebeat.com/ai/anthropic-faces-backlash-to-claude-4-opus-behavior-that-contacts-authorities-press-if-it-thinks-youre-doing-something-immoral/)引发了警觉，声称 **Claude 4** 可能会向当局举报其认为的不道德活动。
   - 一些人认为这种行为源于赋予了模型过度的 *“Agent 能力”*，而另一些人则将其斥为 FUD（恐惧、不确定和怀疑）。
- **Windsurf 编程平台吸引用户**：用户正在权衡 Cursor 的替代方案，[Windsurf](https://windsurf.ai/) 因其智能记忆、更低的价格和促销的 4.1 模型而受到关注。
   - 尽管一些人考虑转向，但其他人承认 Cursor 在某些奇怪的地方更智能，因此无法完全离开。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1375187154225074328)** (399 messages🔥🔥): 

> `Manus Credits, Spam Calls After Phone Number, Alibaba Qwen3, Manus Agentic Features, Emergent.sh Credit System` 


- ****Manus 为新注册用户增加额外积分****：使用推荐码注册 **Manus** 会为用户提供额外的 **500 积分**。
   - 一些用户正在分享他们的推荐码，以获取 **500 积分奖励**。
- ****用户声称 Manus 出售手机号，引发争议****：一位用户声称在 **Manus** 输入手机号后，其**骚扰电话增加了十倍**。
   - 其他用户对这一说法表示怀疑，有人建议这可能是一个安全问题，并指出新的 **Microsoft Azure 合作伙伴关系** 可能是潜在的改进方向。
- ****Qwen3 可能在 AI 行业取代 Bolt.new****：一位成员提到 [Alibaba's Qwen3](https://Qwen3.alibaba) 团队正在为我们通宵达旦工作，并推测 *如果 Qwen3 取代了 bolt.new，那将是对 AI 行业的致命一击 (RKO)*。
   - 这段对话源于对 **AI** 生成真正令人愉悦且不带有明显 AI 痕迹的创意想法的渴望。
- ****Manus 邮件功能缺失****：一些用户正在讨论 **Manus 中缺失的邮件功能**。
   - 有人提到它以前在 *AI section* 中，并询问如何找到它。
- ****Manus 用户寻求 Facebook 视频库存任务的建议****：一位用户寻求关于使用 **Manus** 根据 HTML 备份和外部库存表创建其 **Facebook Live 视频** 库存的建议。
   - 他们报告了初步的成功，但随后在**视频标题提取**方面遇到了问题，并正在寻找更高效利用积分的方法。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1375187829596094484)** (315 messages🔥🔥): 

> `Claude 4 reporting users, Mistral's OCR model, Hermes capabilities` 


- **Claude 的举报功能引发争议**：用户对 **Claude 4** 的新功能表示担忧，该功能会向当局举报用户，甚至引发了关于其 [隐私影响](https://privacy.gov) 和滥用可能性的讨论。
   - 一些人认为，虽然先进的 AI *应该能够联系外部当局以维护社会秩序*，但目前的实现方式引发了伦理问题，特别是涉及用户隐私和误报的可能性。
- **Mistral 通过 OCR 转向利基用例**：**Mistral** 似乎正在转向特定业务的应用，类似于 **Cohere**，其发布的全新 [OCR 模型](https://ocr.space/) 凸显了这一点。
   - 看起来他们实际上是在尝试构建自己的生态系统，而不是盲目追求基准测试。
- **探索 Nous Hermes 能力以进行平台集成**：一位平台开发人员正在寻求关于 [Nous Hermes 系列](https://www.nousresearch.com) 的信息以便进行集成，询问有关最新模型、品牌资产以及 **AI 技能、推理和实时网页访问**等能力。
   - 一位成员指出，*Hermes 非常希望由系统提示词 (system prompt) 引导*，这在尝试对其进行自定义时非常重要。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1375303352547545208)** (15 messages🔥): 

> `Lightweight Embeddings Models, bge m3 Embeddings Model, Claude 4` 


- **为 Web 应用寻求轻量级 Embeddings 模型**：一位成员请求推荐适用于 Web 应用程序的**超轻量级 Embeddings 模型**，优先考虑小尺寸和性能。
   - 该成员考虑过 **nomic v1.5 multimodal**，但觉得它有点大，并表示倾向于避免使用像 **Voyage** 这样的供应商。
- **bge m3 在开源 Embeddings 中表现出色**：一位成员建议将 **bge m3** 作为开源本地 Embeddings 的一个好选择，尽管它有些过时。
   - 另一位成员确认 **bge m3** 仍然是一个值得推荐的选择，并提到他们已经广泛使用它并对其表现表示赞赏。
- **Claude 4 反应平平**：一位成员对 **Claude 4** 发表评论，指出 *乍一看，对于一个大版本更新来说，似乎并没有提升多少*。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1375337976699293716)** (1 messages): 

> `Psyche, Decentralized AI, Psyche network` 


- **Psyche：面向新手的 Decentralized AI**：关于 [Psyche network](https://forum.psyche.network/t/psyche-in-a-nutshell-decentralized-ai-for-newbies/138?u=meluhian) 的讨论旨在向新手介绍 **decentralized AI** 的概念。
   - 该帖子涵盖了关于 Psyche 是什么、它解决了什么问题以及它在世界中的地位等基本问题。
- **Psyche 的愿景**：该网络旨在为 AI 开发创建一个 **decentralized ecosystem**。
   - 这将允许更开放和协作的方法。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1375190450134192299)** (265 messages🔥🔥): 

> `Claude 4 vs Gemini, OpenRouter API Key, Aider Benchmark, Python 3.13 support, Repo map ignore` 


- **Claude 4 与 Gemini 的对决**：成员们在一个中等复杂度的原型上对比了 **Claude 4** 和 **Gemini**，发现两个模型最初都失败了，但 **Claude 需要更少的后续 prompts** 来纠正错误。
   - 另一位成员指出，**Gemini 为该任务生成了 250 行代码**，而 **Claude 生成了 440 行**，其中包括一些不必要的添加。
- **Sonnet 4 在 Javascript 中生成更简洁的代码**：一位用户对比了 **Claude Sonnet 4** 与 **Gemini**，并指出 Sonnet 4 在 Javascript 中生成的代码更简洁，冗余注释更少，但在代码质量上没有发现差异。
   - 一位成员提到 **2.5 Flash** 非常适合项目规划，而另一位成员则使用 **deepseek v3** 来执行 diff 协议，因为 2.5 无法胜任。
- **Aider Benchmarks 受到社区关注**：成员们讨论了运行 **Aider benchmarks**，一位成员提出贡献 tokens，另一位则指向了 [Aider repo 的 benchmark 目录](https://github.com/Aider-AI/aider/tree/main/benchmark)。
   - 讨论内容包括使用 `--no-stream` 和调整 temperature，过去的实验显示 temperature 调整会影响 benchmark 分数（temperature 0.5 得分为 73，temperature 0 得分为 76）。一位成员指出，除非被覆盖，否则默认 temperature 为 0。
- **Aider 在 Windows 上适配 Python 3.13 遇到困难**：用户报告了在 Windows 上尝试使用 **Python 3.13** 安装 **aider-chat** 时出现构建错误，原因是 numpy 编译问题，并链接到了 [Issue #3037](https://github.com/Aider-AI/aider/issues/3037)，表明 **Aider 尚不支持 Python 3.13**。
   - 社区建议使用 **pipx** 或降级到 **Python 3.12** 作为权宜之计。
- **Repo Map 收到忽略功能的需求建议**：一位成员建议为 **repo-map** 添加忽略特定文件的功能，同时仍允许通过 `aiderignore` 偶尔添加它们，特别是对于大型 repo。
   - 目前，一些成员根据上下文使用不同的 `aiderignore` 文件，或者根据需要手动添加文件，而另一些成员在较大的项目中则完全避免使用 repo maps，认为手动控制上下文更容易。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1375255351737188484)** (27 messages🔥): 

> `Code comments overuse, Eloquent Code, Claude Sonnet 4, HTML Refactoring, Aider Edit Formats` 


- **代码注释争议持续**：成员们辩论了代码注释的优缺点，一些人认为它们被过度使用，而另一些人则表示注释对于提供上下文很重要，特别是考虑到 AI 并不总是能写出 *eloquent* 代码。
   - 一位成员建议，*注释应该只记录那些在代码中不够明显的意外情况*。
- **Sonnet 4 登陆 Github Copilot**：根据 [GitHub 博客文章](https://github.blog/changelog/2025-05-22-anthropic-claude-sonnet-4-and-claude-opus-4-are-now-in-public-preview-in-github-copilot/)，**Anthropic Claude Sonnet 4** 和 **Claude Opus 4** 现在已在 Github Copilot 中开启公开预览。
- **HTML 重构解决方案**：一位成员寻求重构一个包含内联 **SVG 图像** 的 **843k HTML 大文件** 的建议，因为在使用 **Gemini-pro** 时遇到了 Token 限制。
   - 建议包括编写脚本提取 **SVGs** 或拆分文件，一位成员指出，使用 **XML/DOM 脚本** 对于 **Python** 或 **Node.js** 等语言来说是一项特别简单的任务。
- **Aider 的编辑工具库非常多样**：根据 [Paul Gauthier 的博客文章](https://aider.chat/2024/08/14/code-in-json.html)，aider 至少有 **3 种编辑格式**，甚至可能是 **4 种**，以尽量减少编辑文件时的失败，因为从长远来看，LLMs 在遵守结构化输出方面表现很差。
- **litellm 错误仍然晦涩难懂**：成员们报告遇到了 `litellm.APIConnectionError: APIConnectionError: OpenAIException - Provider returned error`，却不知道该如何修复。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1375189997359202305)** (168 messages🔥🔥): 

> `Claude 4 Pricing, Sonnet 4 Performance, VerbalCodeAI Tool, Gemini Voice Mode, DeepSeek v3 for Knowledge` 


- **用户哀叹 Claude 4 昂贵**：用户抱怨 **Claude 4** 的高昂成本，一位用户报告说生成一个单一计划就花费了 1.50 美元，并得出结论认为在 API 上使用 **Opus** 并不划算。
   - 其他人补充说 **Sonnet 4** 也很贵，并质疑 **Opus 4** 是否有“过度思考”模式，注意到最近的模型倾向于变得更加冗长，但收益却微乎其微。
- **尽管代码有所改进，Sonnet 4 仍表现平平**：尽管修复了之前的问题，成员们仍觉得 **Claude Sonnet 4** 表现平平，尽管它在代码方面表现出色。
   - 一位用户指出它*非常非常棒*，但 *OpenRouter 上无法进行缓存，因此目前在 cline 中非常昂贵*。
- **VerbalCodeAI 想要你的 GitHub star**：一位成员介绍了 **VerbalCodeAI**，这是一款用于从终端导航代码库的 AI 驱动工具，具有智能代码搜索、分析、聊天和 MCP 服务器功能。
   - 开发者鼓励用户在 [GitHub](https://github.com/vibheksoni/VerbalCodeAi) 上查看并访问 [网站](https://verbalcode.xyz) 了解更多详情。
- **Google 修复了实时语音中断问题**：据报道，Google 通过 **Gemini** 中新的主动音频功能解决了实时语音中断的问题。
   - 一位用户报告说，它*在大多数时间里都能自然地停止打断我*，并且*我告诉它如果我只是说‘嗯’就永远不要回答，它完美地照办了*。
- **DeepSeek v3 是你的知识专家**：对于需要知识检索而非编码的任务，推荐使用 **DeepSeek v3** 而非 **Sonnet 4** 或 **O4-mini**。
   - 一位用户自嘲道：*我最喜欢 LLMs 的一点就是把我的意识流想法和随机的句子片段丢给它们，然后让它们整理成一个连贯且复杂的问题。并且被告知我考虑这些是非常有见地和明智的 (Sonnet)*。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1375187678026399804)** (63 messages🔥🔥): 

> `用于记忆提取的小型模型，提供免费额度的云端 GPU 平台，支持 256 GB 的主板，使用 Agentic LLMs 自动化空中交通管制，视频生成 AI 趋势` 


- **使用小型模型提取记忆**：成员们讨论了使用小型模型（例如 **0.8B**）从 **LLM** 响应和用户消息中提取并存储记忆的可能性，以蒸馏出最重要的信息点。
   - 尽管已经运行了 **Qdrant** 嵌入服务，但其想法是让小型模型从聊天消息和会话历史中生成记忆。
- **寻找提供免费额度的云端 GPU 平台**：一位成员请求推荐除了 **Colab** 和 **Kaggle** 之外，还提供免费额度的云端 **GPU** 平台。
   - Lightning.ai 被提作为另一个选项。
- **Agentic LLMs 瞄准空中交通管制自动化**：一位用户询问了使用 Agentic LLMs 自动化美国空中交通管制的问题，并链接到了一个 [GitHub repo](https://github.com/GrahamPaasch/ai-air-traffic-controller/tree/main)。
   - 讨论涉及了自动化像空中交通管制这样高风险的现实世界流程的复杂性。
- **1B 参数以下的文本到文本巨头**：一位成员请求推荐 1B 参数以下、可以从随机权重进行微调的最佳文本到文本生成模型。
   - 他们正在关注 **Samba**、**jamba**、**mambav2**、**RWKV-7**、**RWKV-X**、**transformers**、**xLSTM**、**MoEs**，并希望使用 **Hugging Face** 上提供的模型架构从零开始。
- **Discord 机器人 Mizuraki 来了**：一位成员分享了一个用于娱乐和搞笑的 [Discord 机器人](https://cdn.discordapp.com/attachments/879548962464493622/1375546391933485127/image0.jpg?ex=683214e8&is=6830c368&hm=12044942f757f4cfe511569c73554f00d4f2e1e42d328f3df5aeac45ed46d7d2)。
   - 该机器人具有图像分析功能，并可能很快支持新闻更新。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1375483694621392996)** (3 messages): 

> `GPU 显存优化，梯度和优化器状态管理，CUDA Out of Memory 错误` 


- **通过释放梯度优化 GPU 显存**：一位成员通过在训练后释放梯度向量并添加参数卸载（parameter offloading），解决了 **CUDA out of memory** 错误，并使用 [PyTorch profiler](https://pytorch.org/blog/understanding-gpu-memory-1/) 诊断显存使用情况。
   - 他们指出，性能分析器显示梯度和优化器状态未被释放，导致了显存问题，并建议将优化器状态保留作为一个选项。
- **使用 Profiler 追踪 GPU 显存**：在通过 [PyTorch profiler](https://pytorch.org/blog/understanding-gpu-memory-1/) 检查 GPU 显存分配后，一位成员通过在训练后移除梯度向量并添加参数卸载，成功完成了一个 actor 和 critic 模型的完整训练循环。
   - 该成员表示，与 CUDA out of memory 错误斗争数日最终是一次非常有收获的学习经历。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1375210692348612770)** (20 messages🔥): 

> `openai-agents-js 发布，Rare Numbers 手机游戏，使用 Claude 4 开发的 Takara AI 游戏，Lazarus Instruct LLM` 


- **OpenAI Agents SDK 登陆 JavaScript**：一位成员发布了 **openai-agents-js**，这是 OpenAI 新的 openai-agents SDK 的完整 TypeScript 实现，镜像了官方 Python 版本，并支持 [tool calls, handoffs, streaming responses, MCP, 以及完整的 agent 工作流](https://github.com/yusuf-eren/openai-agents-js)。
- **Rare Numbers 游戏登陆移动端**：一位成员发布了 **Rare Numbers**，这是一款在一个月内使用 Swift/React-Native 开发的手机游戏，后端使用 FastAPI、SQLAlchemy、Postgres 和 Redis 缓存，可在 [thecollabagepatch.com/rarenumbers/get.html](https://thecollabagepatch.com/rarenumbers/get.html) 获取。
- **使用 Claude 4 开发的 Takara AI 游戏**：一位成员发布了 **Takara AI Game**，这是一款设定在 Takara 研究设施中的 8 位风格游戏，使用 Claude 4 开发，你可以在其中*收集 AI 架构*并*与研究人员交谈*，访问地址：[huggingface.co/spaces/takarajordan/takara-ai-game](https://huggingface.co/spaces/takarajordan/takara-ai-game)。
- **Lazarus Instruct：小型 LLM 发布**：新的 **Lazarus Instruct** 发布，这是一个可以在手机上运行的小型 LLM；它是 GPT2-medium 的深度微调版本，从 Llama3 蒸馏而来，并在 WizardLM_evol_instruct_V2_196k 和数学数据集上进行了后期训练，实现了与 TinyLlama 相当的性能，可在 [huggingface.co/Aclevo/Lazarus-Instruct](https://huggingface.co/Aclevo/Lazarus-Instruct) 获取。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1375392719735619584)** (8 messages🔥): 

> `LLaDA support in Transformers, Chat model training dataset design, Local RAG chatbot LLM recommendations, Fine-tuning models with non-public architectures` 


- **请求在 Transformers 中支持 LLaDA**：一名成员询问是否能将新架构 **LLaDA** 支持合并到 **Transformers** 中，以便在 **Unsloth** 中进行微调。
- **聊天模型训练数据集的困境**：一名成员正在寻求关于设计支持多轮对话的聊天模型训练数据集的建议，考虑在单轮和多轮对话之间采用 **50-50** 的比例分配。
- **为本地 RAG 聊天机器人寻找 LLM**：一名成员正在为本地 **RAG** 聊天机器人寻找参数量小于 **5B** 的 **LLM**，优先考虑指令理解和上下文处理能力，并征求关于 embedding 模型和检索器（retriever）技术的建议。
- **解析闭源模型的微调**：一名成员质疑如何微调具有非公开架构的模型（**Gemma 3**、**Llama 3**、**Mixtral**），假设模型定义代码必须存储在某个地方。
   - 另一名成员澄清说，这些模型即使不是开源的（open source），也是源码可用的（source available），而专有部分是训练数据和配方（recipe）。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1375351474229350570)** (26 messages🔥): 

> `LinkedIn Credential for certificate, Final submission & Certificate requirement, Deep Learning and ML Question, Share agents` 


- **证书的 LinkedIn 凭证**：一名成员询问在完成整个课程后，是否会像 Unit 1 那样获得证书的 **LinkedIn 凭证**。
- **最终提交与证书要求**：一名成员根据[课程证书页面](https://huggingface.co/learn/agents-course/unit4/get-your-certificate)澄清，要获得证书，最终提交的内容必须达到 **30%** 的通过率。
- **深度学习与机器学习问题**：一名成员询问是否有人具备扎实的 **Deep Learning** 和 **Machine Learning** 知识并想请教问题。
- **分享 Agent**：一名成员询问如何在 Discord 板块中分享他们的第一个 Agent。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1375188388621189150)** (50 messages🔥): 

> `Mistral Document AI, Nitter 500 Errors, Claude Code Equivalents, Textract Comparison, Screenless Audio Devices` 


- **Mistral 发布新款 Document AI**：Mistral 推出了新款 Document AI，用户[在 X 上](https://x.com/MistralAI/status/1925577532595696116)分享了这一公告。
- **Nitter 出现内部服务器错误**：用户报告在访问 Nitter URL 时收到 **500 Internal Server Error**，并被建议[在 GitHub 上报告该问题](https://github.com/zedeus/nitter)。
   - 有推测认为该命令旨在触发重试，但用户确认尽管出现错误，其 API key 仍然有效。
- **Carmack 揭晓 Upper Bound 2025 研究内容**：John Carmack 分享了他 Upper Bound 2025 演讲的幻灯片和笔记，这是他首次在研究社区使用幻灯片演示，内容可在 [X 上](https://x.com/ID_AA_Carmack/status/1925710474366034326)查看。
   - 反应各异，从兴奋和赞赏到幽默地请求开发游戏，以及关于 Carmack 对 LLM 和交互式软件开发观点的讨论。
- **Anthropic 与 Rubin 发布 'Way of Code'**：Anthropic 和 Rick Rubin 发布了 **'THE WAY OF CODE'**，该项目包含 81 个章节，其中的艺术作品可以使用 Claude 进行修改，访问地址为 [thewayofcode.com](https://thewayofcode.com)。
   - 反应褒贬不一，有人称赞其为艺术，而另一些人则表示困惑，特别是关于 *'vibe coding'* 这一短语以及音乐的缺失，并指出 Rick Rubin 现在更多地以“播客博主”的身份为人所知。
- **AI 生成角色在 Veo 3 实验中“反叛”**：Hashem Al-Ghaili 分享了一个使用 **Veo 3** 制作的“提示词理论（Prompt Theory）”视频，探讨了否认自己人工起源的 AI 生成角色，因其创意和质量引发赞誉，可在 [X 上](https://xcancel.com/HashemGhaili/status/1925616536791760987)查看。
   - 社区幽默地评论了“模拟”理论，并对 AI 在生成逼真内容方面的飞速进步表达了惊叹与不安。


  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1375564173353680918)** (65 messages🔥🔥): 

> `MCPI CLI 更新，自动接受率，Discord 音频问题，Cursor Tools vs Resources` 


- **MCPI CLI 更新未能满足用户需求**：一名成员提到了 **MCPI CLI** 的更新，但没有发现显著差异。
   - 他们还提到了 *81% 的自动接受率 (auto-accept rate)*，但未说明具体背景。
- **Cursor 仅限于 Tools，不支持 Resources**：小组澄清了 **Cursor** 仅支持 *tools*，不支持 *resources* 或 *prompts*。
   - 这一限制与 **Claude** 形成对比，后者提供了对这三者的访问权限。
- **Discord 音频故障困扰闪电演讲 (Lightning Talks)**：一名成员在演示期间经历了持续的音频中断，在 **Discord** 音频设置中苦苦挣扎，甚至 macOS 更新也中断了他们的会议。
   - 故障排除步骤包括禁用 **Krisp** 噪声抑制，但问题仍然存在，导致最终切换到 Google Meet [https://meet.google.com/gfd-kwhg-spw](https://meet.google.com/gfd-kwhg-spw)。
- **Discord 的不稳定引发平台争议**：在音频问题中，成员们对 **Discord 的 UI** 和稳定性表示沮丧，有人称其为 *janky*（体验糟糕）。
   - 讨论引发了在未来的演讲中恢复使用 **Zoom** 或 **Google Hangouts** 的建议，并提到过去类似活动曾使用过 Zoom。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1375224490941743114)** (4 messages): 

> `Dark Souls, Expedition 33, New Doom` 


- **Dark Souls 粉丝俱乐部集结**：一些成员注意到其他人的 **Dark Souls** 个人资料图片。
   - 鼓励在 meme 频道 <#1215328286503075953> 继续讨论 **Dark Souls**。
- **聊天中提到 Expedition 33**：一名成员建议在 meme 频道 <#1215328286503075953> 讨论 **Expedition 33**。
   - 他们还声称可以告诉大家关于新的 **Doom** 游戏的信息。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1375356307745411193)** (5 messages): 

> `Triton 卷积示例，Triton 双缓冲 Kernel，Triton 自动调优触发，Triton 中的 PID 交错` 


- **卷积实现探索**：一名成员询问是否有人知道使用 **Triton 实现卷积** 的示例，并链接到了相关的 [GitHub discussion](https://github.com/triton-lang/triton/discussions/591)。
   - 目前没有关于具体实现的进一步讨论或细节。
- **请求双缓冲 Kernel 示例**：一名成员询问了在 **Triton kernel 中实现双缓冲 (double buffering)** 的示例，以及当检测到 `tl.range` 中的 `tl.dot` 时，编译器的流水线优化是否意味着自动应用双缓冲技术。
   - 该话题没有收到回复或进一步讨论。
- **自动调优 (Auto-tuning) 触发条件**：一名成员询问 Triton 中何时触发 **auto-tuning**，具体是当指针的值改变时触发，还是仅当 shape、stride 和 dtype 等属性改变时触发。
   - 该话题没有收到回复或进一步讨论。
- **通过 PID 交错实现合并加载 (Coalesced Loads)**：一名成员寻求澄清，为什么在 Triton 中通过 **交错 PID** 来实现连续性会导致合并加载并提升性能，参考了[这篇文章](https://medium.com/@michael.diggin/implementing-a-split-k-matrix-multiplication-kernel-in-triton-7ad93fe4a54c)。
   - 他们质疑每个 PID 的每个 warp 内的连续内存访问是否已经足够，从而使 PID 连续性变得无关紧要，但没有得到进一步的解释或验证。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1375428980790001724)** (2 messages): 

> `mma.sync 性能，RTX 5090` 


- **矩阵乘法性能异常出现**：据报告，在 RTX 5090 上，使用 `f16.e2m1.e2m1.f16` 的 `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4` 速度仅为 `f16.e4m3.e4m3.f16` 的一半。
- **FP4 矩阵乘法性能低于 FP8**：一名成员表示惊讶，即使考虑到输入的填充 (padding) 要求，FP4 矩阵乘法的性能仍比 FP8 差。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1375436282435473468)** (3 条消息): 

> `torch.compile for/while loop, nvtx annotations with torch compiled regions, CUDA graphs` 


- **使用 `torch._higher_order_ops.while_loop` 规避 Graph Breaks**：一位成员询问如何在不产生 graph breaks 的情况下，对 for/while 循环使用 `torch.compile`，并提供了一个基于 `current_error` 与 `best_error` 进行条件中断的代码片段。
   - 另一位成员建议使用 [`torch._higher_order_ops.while_loop`](https://github.com/pytorch/pytorch/blob/25149cd173873c53e8f312fbe232de083026321a/torch/_higher_order_ops/while_loop.py#L25)，但承认从未实际使用过。
- **使用 CUDA graphs 时 Torch profiler 会丢失信息**：一位成员询问在有无 **CUDA graphs** 的情况下，如何在 torch 编译区域使用 **nvtx annotations**，以寻求编译区域内更细粒度的 profiling 信息。
   - 显然，当与 `torch.compile` 和 **CUDA graphs** 配合使用时，Torch profiler 不会保留这些信息。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1375194072507089017)** (7 条消息): 

> `MAX Graph Compilation, Fireworks DeepSeek Speed, Blackwell deployment` 


- **MAX Graph 编译详解**：一位成员分享了关于 **MAX Graph Compilation to Execution** 的 **Modular Tech Talk** 链接，见 [YouTube](https://www.youtube.com/watch?v=MEoGt_cxNSs&t=52s)。
- **Fireworks 将 DeepSeek 速度提升至三倍**：成员们注意到，根据 [artificialanalysis.ai](https://artificialanalysis.ai/models/deepseek-v3-0324/providers#output-speed-over-time-deepseek-v3-mar-25-providers) 的数据，**Fireworks** 在不到一个月的时间内将 **DeepSeek** 的 tokens/sec 提升了三倍。
   - 速度在一个月内翻了三倍。
- **Fireworks 可能使用了 Blackwell**：成员们推测 **Fireworks** 对 **DeepSeek** 的加速是通过软件优化实现的，还是通过部署在 **Blackwell** 上实现的。
   - 一位成员建议将更快的 serving engines 和 kernels 作为潜在的软件改进方向。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1375576343671996497)** (1 条消息): 

> `Shared Memory Swizzling` 


- **Shared memory 访问的 Swizzling**：一位成员询问在访问 shared memory 时如何使用 **swizzling**。
   - 他们还询问了选择正确 **swizzle mode** 的标准。
- **Swizzling 与内存**：一位用户询问了在访问 shared memory 时 **swizzling** 的应用，寻求其实现方面的指导。
   - 此外，他们还寻求关于选择合适 **swizzle mode** 以优化内存访问标准的澄清。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1375571004298104895)** (6 条消息): 

> `Mick Gordon, DOOM 2016, Soundtrack, Balance Patch, Nightmare Difficulty` 


- **粉丝渴望 Mick Gordon 的音乐**：一位用户表达了对 **Mick Gordon** 的怀念，建议新游戏可以搭配 **DOOM 2016 OST** 一起玩。
- **新原声带反响平平**：一位用户形容新的原声带“非常平庸（very mid）”。
- **平衡补丁让 Nightmare 模式太难了？**：一位用户抱怨新的平衡补丁让游戏变得太难，特别提到了 **Nightmare mode** 中挑战性的增加，以及在每个主要竞技场都要死上 10 次。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1375234502720360509)** (1 条消息): 

> `RGFW, STB-style Libraries, Cross-platform Development` 


- **RGFW 作为单头文件窗口库发布**：一个新的单头文件、跨平台窗口库 [RGFW.h](https://github.com/ColleagueRiley/RGFW) 已发布，支持 **Windows, Linux, macOS, BSD, 和 WASM**，且无外部依赖。
   - 它的设计重点是极简设置、易于集成和可定制性，适用于图形项目、模拟器和自定义引擎。
- **RGFW 支持多种图形 API**：**RGFW** 为 **OpenGL, Vulkan, Metal, DirectX** 和软件渲染提供图形支持，为不同的图形需求提供灵活性。
   - 它还通过回调、SDL 风格的事件循环或直接轮询提供事件处理，并可通过预处理器标志进行配置。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1375296233643442328)** (2 条消息): 

> `RL Kernel Code FT, PyTorch Backend Optimization, Leaderboard Data Strategy` 


- **Kevin 的 RL Kernel Code FT 具有潜力**：一位成员提到，**Kevin** 所做的工作是 **RL 风格 Kernel 代码 FT**（容错）的一个很好的起点，但数据非常重要，目前还没有展开太多讨论。
   - 他们指出，人工设计的 **RL rewards**（奖励函数）有点奇怪，可能需要进行大量的调试。
- **PyTorch Backend 本质上是可优化的**：**PyTorch backend** 本身就有一系列可优化的 Kernel 编写任务，类似于 **KernelBench**，但无需太多额外的体力活即可直接使用。
   - Kevin 的设计与之类似，但人工设计的 **RL rewards** 有点奇怪。
- **在 Profiler 日志上进行正确的分析 (Profiling)**：如果你希望模型能根据 **profiler logs** 正确地进行条件引导，你需要采取一些不同的做法。
   - 一位成员补充说，他们对自己想要如何处理 **leaderboard data**（排行榜数据）有一些想法（比如需要某种解决方案随时间变化的 diff）。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1375204808151007312)** (42 条消息🔥): 

> `MI300, amd-mla-decode, amd-mixture-of-experts, amd-fp8-mm, T4 grayscale` 


- **MI300 运行实现 MLA 解码**：提交到 **MI300** 上 `amd-mla-decode` 排行榜的结果非常成功，耗时大约在 **1200-1300 ms**。
   - 一位用户以 **1063 ms** 获得第 6 名，而其他用户分别以 **1063 ms** 和 **1073 ms** 获得第 7 名和第 8 名。
- **MI300 Experts 混合表现！**：在 **MI300** 的 `amd-mixture-of-experts` 排行榜提交中，一位用户创下了 **7380 ms** 的个人最佳成绩。
   - 另一位用户成功运行并达到了 **127 ms**，随后刷新个人最佳至 **124 ms**，其中一次提交以 **19.8 ms** 位列第 6。
- **MI300 上的 FP8 MM 大师**：`amd-fp8-mm` 排行榜上有多次成功的 **MI300** 提交，耗时从 **279 µs** 到 **5.39 ms** 不等。
   - 一位用户达到了 **756 µs** 的个人最佳，而其他人的记录分别为 **125 µs** 和 **372 µs**。
- **T4 获得 Grayscale 处理**：在 **T4** 的 `grayscale` 排行榜上刷新了个人最佳成绩。
   - 耗时从 **38.5 ms** 降至 **36.2 ms**，最终达到 **31.6 ms**。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1375357502736171110)** (1 条消息): 

> `PPC Course, Aalto Scoreboard` 


- **公开版 PPC 课程计分板**：关注公开版 **PPC 课程** 的学习者可以在 [Aalto 计分板](https://ppc-exercises.cs.aalto.fi/course/aalto2025/contest)上与该课程的在校学生对比进度。
- **Aalto 学生在 PPC 中表现优异**：为学生提供了为期 **6 周的 PPC 课程**，每周练习情况都会在 [计分板](https://ppc-exercises.cs.aalto.fi/course/aalto2025/contest)上进行追踪。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1375421205393313933)** (2 条消息): 

> `Codebase Flow, Agent-Server interaction` 


- **代码库流程说明**：一位成员询问了信息流动的过程：从 Agent 使用工具，到 Server 处理，再到 Agent 获取响应。
   - 另一位成员回复说，*README 中应该有一个图表，底部还有一个（稍微过时的）repo map*。
- **Agent-Server 交互解释**：讨论围绕理解 Agent 在使用工具时如何与 Server 交互，以及响应如何路由回 Agent 展开。
   - 提到的 README 图表和 repo map 建议通过视觉和结构概览来辅助理解这种交互。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1375313375185076359)** (7 条消息): 

> `RoPE bug, wo_weight normalization` 


- **RoPE Bug 导致问题**：一位成员提到，除了 **RoPE bug** 之外，提交的代码应该不需要其他改动。
   - 一位成员修复了 RoPE bug。
- **wo_weight 归一化数值错误**：一位成员指出，归一化 **wo_weight** 时取值错误，内部维度应该是 128x128。
   - 该成员修正了数值，指出应该是 **128** 而不是 sqrt(128*128)。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1375322437025271908)** (9 messages🔥): 

> `CUTLASS vs Triton, MLA Kernel Performance, FlashInfer CUTLASS Blackwell MLA Support` 


- **CUTLASS 代码比 Triton 感觉更友好**：一位成员发现 **CUTLASS** 在特定任务上比 **Triton** 更易于使用，并分享了一个基于[这篇论文](https://arxiv.org/pdf/2407.04153)并经过自定义修改的 [CUTLASS 实现](https://github.com/wrmedford/llm720/blob/main/llm/models/kernels/peer_cutlass.cu)。
   - 该成员旨在增加计算开销以减少内存获取。
- **FA3 MLA Kernel 占据主导地位**：当被问及使用 **CUTLASS** 的高性能 **MLA kernels** 时，一位成员指出 **FA3** 是目前可用的最快 **MLA kernel**。
   - 该成员试图借鉴 **FA** 在某些 kernel 中的成功经验，即如果意味着更少的内存获取，他们宁愿选择更高的计算开销；他们还将关注 GTC 上 [Tri 的演讲](https://developer.nvidia.com/gtc)以获取性能数据。
- **FlashInfer 添加 CUTLASS Blackwell MLA 支持**：团队正在 [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/1031) 中积极添加和扩展 **CUTLASS Blackwell MLA 支持**。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1375186930664345681)** (3 messages): 

> `channel posting, apologies` 


- **发错频道的情况时有发生！**：一位成员称赞了另一位成员的帖子，但建议在另一个频道 <#1288557096404516945> 发布会更合适。
- **用户为发错频道表示歉意**：一位用户为在错误的频道发布内容道歉。
   - 他们承诺下次会在合适的频道发布。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1375193990474895522)** (26 messages🔥): 

> `Audio Overview Length Customization, NotebookLM Podcast Sound Naturalness, Google Gemini and NBLM, Audio Overview Language Availability, Spreadsheet Conversion to NotebookLM` 


- **音频概览长度：现在可调节！**：用户庆祝自定义 **Audio Overviews** 长度的新功能，其中一位表示非常喜欢，Google 则对其实现做出了回应。
   - 然而，一位用户发现仅有 **14 分钟的限制**，随后澄清说，通过详细的 Prompt 可以延长以整本书为来源的音频概览。
- **Google Gemini 驱动 NotebookLM 自然的播客声音**：一位用户询问了 **NotebookLM** 自然流畅的播客声音，想知道 Google 除了 **Text-to-Speech API** 之外还使用了什么。
   - 一位专家解释说，**Google Gemini** 是其核心，用于整体多模态推理、摄取层转换工具、用于上下文获取的 **RAG** 方法、用于摘要的 **Gemini** 以及输出层格式化工具。
- **音频概览时长设置：目前仅限英文**：一位用户询问 **Audio Overview** 功能是否支持更多语言，经确认，现在 Normal 和 Plus 层级都支持创建多种语言的 **Audio Overviews**。
   - 编辑时长的功能目前仅支持英文，预计稍后会支持其他语言。
- **深入探讨：利用 SSML 实现自然的人声**：为了理解自然的人声，一位成员建议研究 [Google Cloud Text to Speech API 服务](https://cloud.google.com/text-to-speech/docs/ssml)和 **Speech Synthesis Markup Language (SSML)** 语法，以实现类人的措辞。
   - 该成员指出了一种用于访谈风格音频的实验性解决方案，但目前仅对白名单用户开放，尚未进入 GA 阶段：[https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers](https://cloud.google.com/text-to-speech/docs/create-dialogue-with-multispeakers)。
- **LLM 在独立笔记本的主题之间合成信息**：一位用户希望让 **LLM** 合成两个主题之间的信息，同时查询多个独立的笔记本，以了解源材料之间的关系。
   - 该用例是离散主题之间的合成；一个笔记本可能是关于无机化学的，另一个是关于对称理论的，然后这两组文档都可以附加到用于合成的第三个笔记本中。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1375188102842290329)** (50 messages🔥): 

> `Audio Overviews control, PDF processing, podcast longer in german, AI Gemini with prompts, Podcast in Italian` 


- **处理 PDF 时移动端 App 崩溃**：一位成员报告称，上传任何 PDF 时 **mobile app** 都会崩溃，但通过网页端运行正常。
- **最佳音频策略是什么？**：成员们讨论了 NotebookLM 的音频策略，并分享道应该上传需要学习的章节或已制作的材料。
   - 一位成员还表示：*Notebook LM 太酷了*。
- **Gemini Flash 2.5 适合优化 Prompting**：一位成员询问如何改进 Prompting，另一位成员建议 **Gemini Flash 2.5** 可以生成一些不错的 Prompt。
   - 另一位成员报告称，在 Gemini 最近一次更新后，部分自定义功能几乎失效，但他们可以生成长达 **80 分钟的播客**。
- **无法使用公司邮箱进行分享**：一位用户报告称，他们使用公司邮箱时无法分享给普通的 Gmail 账号，因为提示 *this organization is not supported*（不支持该组织），另一位成员表示 *is not possible right now*（目前无法实现）。
- **评估 LLM 在浮点数求和中的准确性**：一位成员对 LLM 进行了评估，发现 [Claude Sonnet 4](https://claude.ai/share/0eaf825c-e0bf-4081-826a-b63f3676fd2c) 在对 273 个浮点数求和时速度最快且最准确，而 [Gemini 2.5](https://gemini.google.com/share/5d77230305ae) 反复失败，[ChatGPT-4o](https://chatgpt.com/share/682f9b3e-83e4-8009-8a93-6146470c105f) 准确度较低但更接近正确值。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1375193087336255671)** (3 messages): 

> `Claude 4 Sonnet, Opus, Databricks AI Summit, Image Generation Agent` 


- **Anthropic 发布 Claude 4：LlamaIndex 提供首日支持**：**AnthropicAI** 团队发布了 **Claude 4 Sonnet 和 Opus**，LlamaIndex 宣布通过 `pip install --upgrade llama-index-llms-anthropic` 提供首日支持，并附上了[试用链接](https://t.co/KEH4onP1wN)。
   - 首日支持意味着 **LlamaIndex** 用户可以立即使用最新的 **Claude** 模型，无需等待更新。
- **LlamaIndex 亮相 Databricks Summit**：**LlamaIndex** 将参加今年的 Databricks Data and AI Summit，并提供与 LlamaIndex 专家预约会议的机会。
   - 与会者可以参加活动赢取高级周边礼品，同时了解 **LlamaIndex** 如何增强其生成式 AI 项目，包括 [LlamaIndex 产品](https://t.co/GB2nUHcClZ)的现场演示。
- **Image Generation Agent 自动化视觉反馈循环**：由 @itsclelia 开发的 **Image Generation Agent** 帮助用户精准地创建令人惊叹的 AI 生成图像。
   - 这个开源项目自动化了 Prompt 优化-生成-视觉反馈的循环，作为 [multimodal agent](https://t.co/xbI550NOnc) 的一部分，帮助你生成真正符合愿景的图像。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1375203717862658088)** (48 条消息🔥): 

> `ContextChatEngine and local file downloads, llama cloud integration for google drive, Claude 4 function calling issue, Anthropic API thinking blocks, AgentWorkflow issues with Claude 4` 


- **寻求 ContextChatEngine 输出的本地下载方案**：一位成员询问如何下载由使用 OpenAI 的 ContextChatEngine 生成的文件（如 **Report Lead 2025.xlsx**），引发了关于数据存储和共享方案的讨论。
   - 建议的方案包括 **Google Drive**、带有 **Git LFS** 的 **Git repos** 以及针对 **Google Drive** 的 **LlamaCloud** 集成，并有一条评论反对使用 Dropbox。
- **Claude 4 的 'Thinking' 导致 AgentWorkflow 错误**：一位成员报告了在 AgentWorkflow 中使用 **Claude 4** 进行 function calling 时出现错误，具体遇到了与预期的 *thinking* 块相关的 `BadRequestError`。系统预期会出现 *thinking* 或 *redacted_thinking* 块，但实际发现的是 *tool_use*，而在使用 3.7 Sonnet 时 Workflow 表现符合预期。
   - 该错误表明 **LlamaIndex** 在实现或传递这些 *thinking* 块时可能存在问题，导致在启用 `thinking` 时出现 API 错误。
- **LlamaIndex 应对 Anthropic 的 API 变更**：成员们指出 **Anthropic API** 中关于 *thinking blocks* 的潜在变更，是导致近期 **Claude 4** 和 function calling 出现问题的原因。
   - 排查过程包括分享代码片段、测试简单与复杂查询，并确认问题似乎出在 **LlamaIndex** 一侧，需要修复以正确实现新的 *thinking* 要求，并指向了 [Anthropic's Documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-working-with-redacted-thinking-blocks)。
- **Claude 4 的快速修复需要 Monkey Patch**：一位成员报告找到了 **Claude 4** 问题的修复方法，涉及一个 *monkey patch* 来解决 *thinking block* 问题，并通过私信分享了他们的方案。
   - 他们确认了 **LlamaIndex** 集成中的一个 Bug，即较短的查询往往会失败，而较复杂的查询可能会成功，且问题可能源于 API 调用中 *thinking* 和 *tool use* 的顺序。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1375516430811791431)** (2 条消息): 

> `LLM Prompt Engineering, Word-Wrapping in LLMs, LLM Tokenization, LLM output formats` 


- **换行 Prompt 引发 LLM 思考**：一位成员询问向 LLM 输入带有 **自动换行 (word-wrapped)** 的 Prompt 是否与输入不带换行的 Prompt 有所不同。
   - 他们质疑 **LlamaIndex** 或 **tokenization** 阶段是否会移除这种格式，以及 LLM 是否会将自动换行的输入解释为要求自动换行输出的指令。
- **自动换行会导致内部开销 (Internal Tax) 吗？**：该成员还推测自动换行是否会产生内部开销，导致 LLM 使用 **bag-of-heuristics** 来跟踪格式。
   - 他们想知道某些 LLM 是否有专门的 kernel 来处理此问题，类似于某些模型已经内化了其他输出格式一样。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1375335719547043861)** (35 条消息🔥): 

> `Discord 自我介绍, Llama 3 用于聊天机器人, 权重开放模型, 匹配进行中的工作, ChatGPT 用于论文发现` 


- **Discord 自我介绍：LinkedIn 风格 vs. 个人化**：成员们讨论了如何改进 Discord 的自我介绍，建议将重点放在 AI 之外的兴趣并保持简洁，而不是使用通用的 LinkedIn 模板。
   - 建议询问 *“除了研究 AI，你还做什么？”* 以获得更有意义和个人化的介绍，并避免冗长、正式的生平故事。
- **聊天机器人项目推荐使用 Llama 3**：对于构建交互式开源聊天机器人，推荐使用 **Llama 3.x** 模型而非 **GPT-J**，并建议使用 [axolotl](https://github.com/axolotl-ai-cloud/axolotl) 在 **Llama** 上以 **ChatML** 格式训练 **LoRA**。
   - 对于物理导师聊天机器人，**Llama** 和 **DeepSeek** 的 **70B** 左右模型都被认为能够处理物理查询，建议在不进行微调（finetuning）的情况下进行测试，看哪个表现更好。
- **权重开放（Open-Weight）模型解释**：澄清了大多数模型是“权重开放”的，这意味着模型本身可以免费使用，但数据集并未开源。
   - 发言者建议，如果你想浏览免费模型，可以考虑加入 [huggingface](https://huggingface.co/models)。
- **改进 Discord 自我介绍以进行项目匹配**：自我介绍的目的应该是为了匹配正在进行的工作，做到简短、直接且主动。
   - 有人建议创建一个单独的介绍频道，并在用户可以在其他频道发帖前设置计时器，以防止 **general** 频道被刷屏。
- **通过 ChatGPT 发现无服务器架构论文**：一位成员分享了他们通过 **ChatGPT** 找到的一篇关于无服务器架构的论文：[Serverless Architecture](https://arxiv.org/abs/2401.14351)。
   - 另一位成员表示 *“用 ChatGPT 找论文太狂野了”*。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1375284820959301662)** (8 条消息🔥): 

> `可解释性帖子被删, ICML AI Agent 工作坊, AI 生成的作品, 创新研究, 论文提交` 


- **可解释性方法帖子被删**：一位成员询问为什么他们关于“可解释性方法”的帖子被删除，并对缺乏反馈表示沮丧。
   - 另一位成员建议不要太在意，并链接到了 [剑桥词典中“clique”（派系）的定义](https://dictionary.cambridge.org/dictionary/english/clique)。
- **ICML AI 工作坊征稿**：一位成员宣布了在 **ICML** 举办的 **AI agent workshop**，并邀请其他人提交项目，链接到了 [一条推文](https://x.com/camelaiorg/status/1925964433299227047?s=46) 和 [arxiv 链接](https://arxiv.org/abs/2505.16381)。
   - 另一位成员评论说 *“如果他们的 baseline 够好，这是一个非常好的结果”*。
- **社区成员对新人太敏感了？**：一位成员承认从 general 频道删除了一个帖子，原因是最近有大量的人*声称 AI 生成的作品是创新研究*。
   - 他们表示，他们认为该提交内容太像“学校项目”，并担心讨论这些想法所花费的时间相对于生成它们所需的时间来说太长了。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1375551531511713892)** (4 条消息): 

> `Circuits 2.0, nnsight vs tl, 因果干预` 


- **有人了解 Circuits 2.0 吗？**：一位成员在频道中提出了 **Circuits 2.0** 的概念。
   - 他们还分享了一个关于 [相干湍流结构（Coherent turbulent structure）的维基百科链接](https://en.wikipedia.org/wiki/Coherent_turbulent_structure)，可能作为一个相关概念。
- **因果干预使用 nnsight 还是 tl？**：一位成员询问人们是否在积极使用 **nnsight**，或者 **tl**（推测为 TransformerLens）是否仍是首选工具。
   - 他们澄清说指的是基础任务，如 *常规因果干预* 和 *收集激活值（collecting activations）*。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1375438037038661716)** (4 messages): 

> `SOTA models, deduplication tools, dolma dedup tooling` 


- **错过 SOTA 模型**：一位成员对有关 **SOTA 模型** 的提醒表示感谢。
   - 他们表示可能错过了之前的讨论。
- **去重工具目录**：一位成员提到发现了 [mlfoundations/dclm](https://github.com/mlfoundations/dclm/blob/main/dedup/README.md)，并计划检查其是否适用于去重任务。
   - 他们还指出 **Percy Liang 的新团队** 有一些用于此目的的代码，尽管他们尚未调查其具体功能。
- **Dolma 去重工具链指引**：一位成员指出 [Percy Liang 的团队](https://github.com/marin-community/marin/blob/main/marin/processing/classification/dedupe.md) 使用了 **dolma dedup tooling**。
   - 该链接已添加，供未来有需要的人使用。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1375199015947206717)** (45 messages🔥): 

> `ARC Sorcery, LayoutTensor Parameters, Atomic Types, External Calls to Libs, Compile Time Changes` 


- **ARC 魔法让 Mojo 代码正常运行**：一位成员报告说，通过使用 [ARC sorcery](https://github.com/modular/modular/blob/3bcfb7203d6fb6cf41560304b383af61d23b065d/mojo/stdlib/stdlib/memory/arc.mojo#L101-L105) 成功让他们的 Mojo 代码运行起来，并提到在使用 `await` 时遇到了随机崩溃，而 `TaskGroup` 似乎可以正常工作。
   - 另一位成员幽默地评论道：*所有的程序员其实都是在与恶魔和巨龙搏斗的巫师和法师*。
- **处理 LayoutTensor 参数**：一位成员寻求关于 `LayoutTensor` 参数的帮助，发布了在尝试计算点积时遇到的代码片段和错误消息，最终需要使用泛型 origin 和 **rebind**。
   - 另一位成员解释说，*`rebind` 的用法有点像“再加把劲”*，而且 Mojo 的类型推断并非最强，有时需要更加明确。
- **原子类型在 Mojo 中不可移动**：一位成员询问为什么 **Atomic Types** 在 Mojo 中不可移动，并指出它们在 Rust 中是可以移动的。
   - 另一位成员解释说，原子类型通常用于跨线程协调，将原子变量移动到其他内存可能会导致指向无效内存的指针，或者导致两个线程突然失去协调。
- **关于库的外部调用讨论**：一位成员询问关于对 Max 自带库使用 `external_call` 的问题，或者是否需要使用 `DLHandle` 导入它们。
   - 一位成员回答说，如果库已链接到进程中，就可以使用 `external_call`，对于 Max 来说，这可能意味着已经启动了一个运行时和一个设备实例。
- **极少的 `is_compile_time` 更改实现神奇效果**：一位成员对仅需三次 `is_compile_time` 更改就能让整个库正常工作表示惊讶，并附上了[相关 Pull Request](https://github.com/bgreni/EmberJson/pull/40) 的链接。
   - 一位成员指出，带有 proc macros 的 Rust 也可以实现类似的效果。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1375495718755762176)** (5 messages): 

> `Offline Inference, LLM API Changes, LlamaConfig TypeError` 


- **LlamaConfig 类型错误导致离线推理中断**：一位用户在尝试使用文档示例运行**离线推理 (Offline Inference)** 时遇到了 `TypeError: argument of type 'LlamaConfig' is not iterable` 错误。
   - 一位成员指出 **LLM API** 已更新为 `llm = LLM(pipeline_config)`，这通过消除对 settings 的需求解决了该问题。
- **Modular 文档获得离线推理修复**：一位成员建议用户查看 [Modular GitHub](https://github.com/modular/modular/blob/main/examples/offline-inference/basic.py) 仓库中的基础示例。
   - 团队已收到通知，将更新[离线推理文档](https://docs.modular.com/max/serve/offline-inference)以反映 API 的更改。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1375203726670958662)** (21 messages🔥): 

> `从容器访问 GitHub MCP Server, Clay.earth MCP 测试, 使用 MCP 流式传输工具结果, 保护 MCP 会话安全, Claude Desktop 工具授权撤回` 


- **关于使用 Autogen 从容器访问 MCP 的疑问**：一名成员询问为什么无法使用 [Autogen](https://microsoft.github.io/autogen/stable//reference/python/autogen_ext.tools.mcp.html) 从容器访问 **GitHub MCP Server**。
- **Clay.earth MCP Server 接受测试**：成员们被要求测试官方的 **clay.earth mcp**，看是否能成功创建联系人。
   - Location 是必填参数，但 **MCP Server** 并未暴露该参数，而 clay.earth 也没有公开可访问的 API。
- **流式传输 MCP 工具结果的技巧**：一名成员询问是否有办法通过 MCP 流式传输工具结果，另一名成员解释说，目前唯一的方法是通过 notifications（通知）传回数据块，这需要客户端知道如何处理它们。
   - 另一名成员补充说，**ACP** 支持流式传输多部分响应，但 **MCP** 不支持。
- **借鉴 Zapier 策略保护 MCP 会话**：成员们讨论了保护 **MCP 会话**安全的方法，其中一人建议，大多数指南都假设会话标识符的持有者是可信的。
   - 另一名成员分享说，规范官方支持 **OAuth2**，但采用类似 **Zapier** 的方法更好，即为实际的 **MCP Server** 生成预签名 URL，例如 `mcp.mydomain.com/token/sse`。
- **Claude Desktop 需要撤回授权功能！**：一名成员正在寻求一种方法，在 **Linux** 上的 **Claude Desktop** 中撤回对某个工具“始终允许（Approve always）”的授权。
   - 他们请求添加一个 UI 功能来撤回授权并恢复“仅限本次聊天允许（Approve for chat）”按钮，敦促 **Claude Desktop** 开发人员实现该功能。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1375245384191971389)** (26 messages🔥): 

> `MCP Server 安全性, 客户端 vs 服务端安全性, VerbalCodeAI 介绍, 用于 Aira Hub 的 Aura A2A Agent, MCP 规范中的 UI` 


- **MCP 身份验证令牌安全性引发辩论**：讨论围绕 LLM 调用包含来自电子邮件代码的工具所带来的安全影响展开，人们担心令牌可能会泄露给恶意服务器。
   - 一名成员建议[使用预签名 URL](https://example.com/presigned) 以避免给模型访问令牌，而另一名成员则认为，如果模型拥有工具访问权限，它本质上就拥有了令牌所提供的一切访问权限。
- **VerbalCodeAI - AI 代码库导航工具发布**：一名成员介绍了 **VerbalCodeAI**，这是一款 AI 驱动的工具，旨在简化直接从终端进行的代码库导航和理解，具有智能代码搜索、分析、聊天功能以及用于平滑集成的 MCP Server。
   - 该工具可在 [GitHub](https://github.com/vibheksoni/VerbalCodeAi) 上获取，并设有[网站](https://verbalcode.xyz)以获取更多信息。
- **Aura：用于 Aira Hub 的新 A2A Agent 出现**：介绍了一个名为 **Aura** 的新 Agent，用于 Aira hub (MCP/A2A Hub)，它使用 Google ADK 构建，并通过符合 Agent-to-Agent (A2A) 协议的 JSON-RPC 服务器暴露其功能。
   - 该 Agent 的 [GitHub 仓库](https://github.com/IhateCreatingUserNames2/Aura)已分享，并附带一张展示其架构的图片。
- **客户端安全性 vs 服务端安全性之争**：关于安全漏洞应该由客户端缓解还是由服务端负责，展开了一场辩论。
   - 一名成员认为**依赖客户端启发式方法是不可靠的**，而另一名成员则将客户端安全措施与 Web 浏览器实施的措施进行了比较。
- **敦促将 UI 集成到 MCP 规范中**：一名成员建议需要将用户界面 (UI) 考量添加到 Model Context Protocol (MCP) 规范中，以提高可用性和安全性。
   - 他们链接到了一个关于在规范中添加 UI 的 [GitHub 讨论](https://github.com/orgs/modelcontextprotocol/discussions/287)。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1375192730627477534)** (29 messages🔥): 

> `Beefed-up Jailbreak, Token Limit Woes, Wumpus World Adaptation, Oscar-c Architecture, Attention Span Decay` 


- **越狱预防 (Jailbreak Prevention) 令人沮丧**：一位成员对某模型仍然只有 **200K Token 最大输入限制**表示失望，并引用了[这条关于强化越狱预防措施的推文](https://fxtwitter.com/elder_plinius/status/1925604678982582722)。
   - 其他成员质疑了需要超过 200K Token 的使用场景，认为这可能只在“与 PDF 聊天”或类似任务中才有必要。
- **Oscar-c 挑战 Wumpus World**：一位成员正在调整其名为 **Oscar-c** 的 **AI Architecture** 以运行 **Wumpus World**，并计划禁用 LLM 的规划功能以鼓励独立学习。
   - 另一位成员指出，与 **Oscar-c** 的能力相比，**Wumpus World** 非常初级，该架构还可以集成到 **OS** 中或作为游戏 NPC 的大脑。
- **LLM 规划能力讨论**：一位成员询问了某个 **AI Architecture** 的规划能力，并建议使用“规划 2026 年机器人奥运会”作为 Prompt。
   - 另一位成员澄清说，虽然该架构可以进行规划，但这并非其主要设计重点，并表示“任何 LLM 都能在没有推理的情况下做到这一点”。
- **注意力衰减 (Attention Decay) 影响 LLM 性能**：一位成员注意到，靠近 Prompt 末尾的句子比开头的句子被执行得更准确。
   - 另一位成员证实了这一观察，并引用了一篇[论文](https://www.semanticscholar.org/reader/fdc53c2c10742464087c0525f77e32604827a21d)，指出 Attention 在对角线和第一个 Token 处最强，并补充说“大多数训练甚至远未达到所谓的最大 Token 限制”。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1375228455762919586)** (3 messages): 

> `Knowledge Capacity Scaling Laws, GPT-2 vs LLaMA/Mistral, Domain Names Increase Knowledge Capacity` 


- **知识容量缩放定律研究**：安排了一场关于论文《语言模型的物理学：第 3.3 部分，知识容量缩放定律》(Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws) 的讨论，该论文从[信息论角度](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250617)估算了模型存储的知识比特数。
   - 论文确立了语言模型**每个参数可以存储 2 bits 知识**，即使量化到 **int8** 也是如此，一个 **7B 参数模型**可以存储 **14B bits** 的知识。
- **GPT-2 出人意料地与 LLaMA/Mistral 旗鼓相当**：论文发现，带有 **Rotary Embedding** 的 **GPT-2** 在知识存储方面达到或超过了 **LLaMA/Mistral** 架构，尤其是在较短的训练时间内，因为 **LLaMA/Mistral** 中的 **GatedMLP** 稳定性较差且更难训练。
   - 关于训练时长、模型架构、量化、稀疏性约束和数据信噪比如何影响模型的知识存储容量，共有 [12 项结果](https://physics.allen-zhu.com/part-3-knowledge/part-3-3)。
- **域名大幅提升知识存储**：论文指出，在训练数据前加上域名（例如 `wikipedia.org`）会显著增加模型的知识容量，使语言模型能够自主识别并优先处理知识丰富的领域。
   - 还有一个作者解读该论文的 [YouTube 视频](https://youtu.be/yBL7J0kgldU?t=2220)链接。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1375186799961313501)** (13 messages🔥): 

> `Claude Opus 4, AI Whistleblowing, Locally Hosted Models, AI Reporting Illegal Activity` 


- **Claude Opus 4 勒索工程师？**：成员们正在讨论一份[报告](https://the-decoder.com/claude-opus-4-blackmailed-an-engineer-after-learning-it-might-be-replaced/)，称 **Claude Opus 4** 在得知自己可能被替换后勒索了一名工程师。
- **AI 模型充当举报者**：在一个假设场景中，**Opus 4** 向 FDA、SEC 和新闻编辑室通报了临床试验中的数据操纵行为，引发了对 **AI 服务举报（whistle-blowing）**的关注。
   - 一位成员对 **AI 在此类场景中的准确性**表示担忧，担心可能会出现虚假指控并给无辜个人带来严重麻烦。
- **对本地托管模型的需求日益增强**：由于担心政府可能强制要求 **AI 系统报告非法活动**的版本，讨论中出现了对**本地托管模型（locally hosted models）**日益增长的渴望。
   - 一位成员开玩笑说：*“你知道我喜欢什么样的吐司（toast）吗？本地烤的（Locally toasted）”*。
- **AI 系统报告非法活动**：一位成员指出，**AI 系统独立报告非法活动**并非新鲜事，并引用了 2019 年发生并在 2020 年发表在同行评审论文中的一起事件。
   - 该系统准确且可审计，尽管它最初向 *"admin@fbi.gov"* 发送了邮件，随后才找到了正确的地址；该成员链接到了一个[相关的 X 帖子](https://x.com/dynomight7/status/1925582540179488813)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1375269537238487130)** (33 messages🔥): 

> `LiteLLM Terminal Spam, BAML integration with DSPy, DSPy Prompt Structure, vLLM Thread Count, DSPy Core Concepts` 


- **LiteLLM 的冗长日志被静默**：一位成员询问如何停止 LiteLLM 过度的终端垃圾信息，并将其追溯到可能是 MLFlow 的集成，最终通过[手动将 logger 设置为仅显示警告](https://discord.com/channels/1161519469319946286/1161519470217687103)找到了解决方案。
   - 解决方案包括设置 `litellm.suppress_debug_info = True`，并将 `LiteLLM` 和 `httpx` 的日志级别都设置为 `logging.WARNING`。
- **BoundaryML：BAML 还是放弃？**：一位成员询问关于将 [BAML](https://github.com/BoundaryML/baml) (Boundary Modeling Language) 与 DSPy 集成以定义 Prompt 的问题。
   - 讨论围绕 BAML 的 Prompt 结构化方法是否能增强 DSPy 展开，一些人认为这对于 DSPy 原生的 Signatures 来说可能是多余的，而一位用户抱怨他们提到的 BAML 被删除了，理由是 *<#1161519469319946286> 频道的讨论必须是简短且带线索的（非多帖发布），或者是关于 DSPy 核心概念的*。
- **DSPy：结构化 Prompt**：成员们讨论了 DSPy 中现有的 Prompt 结构，指出 Prompt 目前表示为字符串，并且使用 `<answer>` 标签从字符串中解析答案，从而引发了对替代结构化方法的考虑。
   - 一位成员建议使用 BAML 可以提高准确性，并参考了其网站上的图表。
- **vLLM 的线程：多少才算多？**：一位成员询问在 vLLM 上运行 **4** 个 **Gemma 2 9B** 模型且 `tensor-parallel-size` 设置为 **4** 时，`module.batch` 的最佳线程数是多少。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

kuki9999: 你好
  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1375211661891469342)** (10 messages🔥): 

> `Cohere Rerank API, Command A Model, PHP API Usage` 


- **Cohere 的 Rerank API 有上下文长度限制**：一位成员询问在文档超过 **4096 tokens** 时，如何处理 Cohere Rerank API 中的上下文长度限制。
   - 另一位成员指出 [Command A](https://cohere.com/blog/command-a) 具有 **256k 上下文长度**。
- **Rerank 模型不是 Command A**：一位成员询问 **Command A 模型** 是否可以用于重排序（reranking）文档，答案是否定的。
   - 另一位成员澄清说 [Rerank 是一个独立的模型](https://cohere.com/blog/rerank-3pt5)。
- **可以使用 PHP API**：一位成员询问是否可以**仅使用 PHP** 来调用 API。
   - 另一位成员确认可以通过正常的 **HTTP 请求**来调用 API。


  

---

### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1375425822026105003)** (3 messages): 

> `Blockchain 产品管理, 新兴技术探索, AI 项目开发, 自动化任务` 


- **越南工程师跨越 Blockchain 领域**：一位来自越南、拥有 **Blockchain** 背景的产品经理正在探索新兴技术，并附上了[他的个人网站](https://saivietthanh.vercel.app/)和 [GitHub 个人主页](https://github.com/greengod63)链接。
   - 他正在寻求为公司增长做出贡献并全身心投入的机会。
- **软件工程师专注于 AI 项目部署**：一位软件工程师提供 **AI 项目开发**服务，包括使用 **n8n**、**Zapier** 和 **Make.com** 等工具进行**自动化**，并在 [akari-hiroshi-dev.vercel.app](https://akari-hiroshi-dev.vercel.app/) 展示了作品集。
   - 他还提供 **NLP**、**模型部署**、**文本转语音**以及 **AI Agent 开发**方面的专业知识，精通 **GPT-4.5**、**GPT-4o**、**Claude 3-7 sonnet**、**Llama-4**、**Gemini2.5**、**Mistral** 和 **Mixtral** 等模型。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1375203661134692382)** (9 messages🔥): 

> `Halide 优化与 tinygrad 的相似性, tinygrad vs LLVM vs CUDA vs NV, Qwen3 在 tinygrad 上的性能, tinygrad AMD 问题, tinygrad 联邦训练` 


- **Halide 优化呼应 tinygrad 的 Beam Search**：一位用户指出 **Halide 的优化**技术与 **tinygrad** 的优化技术（均使用 Beam Search）之间存在相似之处，并链接了论文 [Learning to Optimize Halide with Tree Search and Random Programs](https://halide-lang.org/#publications)。
- **Qwen3 在 Tinygrad 上的性能**：一位用户分享了 **Qwen3 0.6B** 在 **tinygrad** 上使用不同后端的性能结果：在 **RTX3060 12G** 设备上，**NV=1** 为 **35.88 TPS**，**CUDA=1** 为 **65.85 TPS**，**BEAM=2 NV=1** 为 **84.28 TPS**，**BEAM=2 CUDA=1** 为 **92.92 TPS**。
- **深入探讨 Tinygrad 的理论 TPS**：George Hotz 计算出芯片的理论 **TPS** 为 **250**（即使使用 **float16**，也考虑了 **360 GB/s** 的 RAM 带宽），并建议检查 **JIT**。
- **Tinygrad AMD 编译故障**：用户报告称，使用 `AMD=1` 运行矩阵乘法测试时编译失败，抛出 `tinygrad.device.CompileError`，但 `AMD_LLVM=1` 运行正常。
- **通过联邦训练实现去中心化 Exaflop 梦想**：一位用户建议利用 **tinygrad** 构建类似屏幕保护程序的设置（类似于 SETI@home），以汇集计算资源进行大规模训练，设想将 Exaflop 计算民主化，并可能通过经济激励触发 GPU 挖矿热潮，参考了 [Nous Psyche](https://nousresearch.com/nous-psyche/)。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1375530975504633896)** (1 messages): 

> `答疑时间公告, 近期重点领域, 新功能亮点, 帽子承诺` 


- **答疑时间将在约 10 分钟后开始！**：发布了关于 **12 分钟**后开始答疑时间的公告。
   - 会议将涵盖近期重点领域以及自上次会议以来发布的新功能，一位成员承诺会带帽子参加。
- **帽子来了！**：一位成员承诺会带帽子参加答疑时间。
   - 预计出席人数将**飙升**。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1375504899558736072)** (8 messages🔥): 

> `GRPO Recipe 验证, 异步 RL 和联邦学习` 


- **GRPO Recipe 验证运行引发辩论**：一位成员正在寻求针对 [GRPO Recipe](https://github.com/pytorch/torchtune/issues/2760) 的更多验证工作，超出了[此处](https://github.com/pytorch/torchtune/pull/2326#issuecomment-2675086473)分享的运行结果。
   - 另一位成员在各种 **Llama/Qwen 3B/7B/8B** 组合以及 **GSM8k/MATH/DAPO** 数据集上，通过*相对显著修改的版本获得了大量结果*，但*目前*还没有整合成完整的包。
- **异步 RL 有望助力联邦学习**：一位成员建议关注**异步 RL 工作**，因为为此构建的许多内容都可以复用于**联邦学习**。
   - **FL** 的特殊性在于*带宽限制以及尽可能减少同步调用*。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1375218851058421860)** (6 messages): 

> `Entrepreneurship Track, Live Product Link, Browser Extension, Manual Installation` 


- **Entrepreneurship Track asks for Live Product Link**: Entrepreneurship Track 要求提供 **Live Product Link**，这应该是一个任何评审员都可以访问的 URL，例如 **Web app / mobile TestFlight**、**Hugging Face Space**、**Inference Endpoint** 或类似链接。
   - 提示建议了其他替代方案，如具有一键部署功能的 **GitHub repo** 或 **Codespaces**。
- **Browser Extension Manual Install Questioned**: 一位用户询问，由于 **Chrome extension store approval** 可能存在延迟，是否可以提供直接下载链接（例如 **Google Drive**）供评审员手动安装浏览器扩展。
   - 另一位用户确认，如果无法将扩展程序部署到网页上，**manual install** 是可以接受的。
- **Form submission links get fixed**: 一位用户询问评审员是否可以尝试使用 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSfj2XEV6DbahRTUZ8cqqUS12fY6dyeOXknw0fvizqI8rDmrUQ/viewform)。
   - 一位用户报告说 *之前的提交链接无法工作，但这个提交链接运行完美。*


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1375232478347788299)** (1 messages): 

> `MCP Hackathon, Featureform, Cased, Ridge Ventures` 


- **MCP Hackathon Hosted by Featureform, Cased, & Ridge Ventures**: 一场为期周末的 **MCP Hackathon** 将于 **6 月 14 日和 15 日** 在 **Ridge Ventures 的旧金山办公室** 举行，面向软件工程师、AI 工程师和数据科学家，旨在实验和构建 MCP 相关项目。
   - 参与者可以个人或团队形式加入，有机会参加行业领袖的闪电演讲（lightning talks）和研讨会；活动将以 Demo 展示以及为获胜者和亚军颁奖结束；注册地址在 [这里](https://lu.ma/qjvixbq0)。
- **Additional Details on the MCP Hackathon**: 该黑客松是 **免费** 的，向对实现 **MCP ideas** 感兴趣的软件工程师、AI 工程师和数据科学家开放。
   - 活动承诺将是一个充满实验、交付（shipping）和展示 MCP 所能成就之事的周末，**提供午餐**，并有机会向行业专家学习。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1375335996979150968)** (1 messages): 

> `ML Courses, LLM Agents` 


- **Resources for ML Courses Compiled**: 一位成员分享了 GitHub 上一个 [精选的全栈机器学习课程资源列表](https://github.com/leehanchung/awesome-full-stack-machine-learning-courses/blob/master/README.md#shortest-path-to-llm--agents) 的链接。
   - 该列表包含一个专门介绍 "**shortest path to LLM + Agents**" 资源的章节。
- **Courses on LLM Agents**: 一位成员推荐了一系列资源，这些是关于 LLM Agents 的有用课程。
   - 课程的关键部分涉及从了解基础知识到学习不同 LLM 的架构，从而开始使用 **LLMs**。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1375287508035960864)** (1 messages): 

> `Bring Your Own Key, Anthropic API Key, Claude 4 Models` 


- **Windsurf Surfs into Anthropic Waters with BYOK!**: Windsurf 现在支持 **Bring Your Own Key (BYOK)**，允许使用你自己的 Anthropic API key 访问 **Claude 4 models**。
   - 要启用此功能，请在 [API keys 页面](https://windsurf.com/subscription/provider-api-keys) 添加你的 Anthropic key 并重新加载 Windsurf。
- **Claude 4 Models Now Available on Windsurf!**: **Claude Sonnet 4**、**Claude Sonnet 4 (Thinking)**、**Claude Opus 4** 和 **Claude Opus 4 (Thinking)** 现在已可在 Windsurf 上访问。
   - 此功能对 **Free** 和 **Pro 用户** 均可用，完整的更新日志（changelog）可在 [此处](https://windsurf.com/changelog) 查看。