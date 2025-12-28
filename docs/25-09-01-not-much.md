---
companies:
- openai
- x-ai
- zhipu-ai
- meituan
- apple
date: '2025-09-01T05:44:39.731046Z'
description: '**OpenAI** 将 **GPT-5** 集成到了 Xcode 26 中，虽然改善了代码延迟，但也被指出在用户体验（UX）方面存在一些折中。**xAI**
  的 **Grok Code Fast 1** 势头强劲，在使用量上已超越 **Claude Sonnet**，并因其快速的调试能力而广受好评。**智谱**的 **GLM-4.5**
  提供了一项极具性价比的编程方案，在与 Claude Sonnet 4 的竞争中表现出色。**美团**发布了 **LongCat-Flash-Chat**，这是一个拥有
  5600 亿参数的 MoE（混合专家）模型，具备自适应计算能力并提供了详尽的技术见解。**苹果**则首次推出了端侧视觉语言模型 **FastVLM** 和 **MobileCLIP2**，与之同台亮相的还有
  **InternVL3.5**。'
id: MjAyNS0w
models:
- gpt-5
- grok-code-fast-1
- claude-sonnet
- glm-4.5
- longcat-flash-chat
- fastvlm
- mobileclip2
- internvl3.5
people:
- gdb
- martin_casado
- yanndubs
- elonmusk
- cline
- vikhyatk
- dzhng
- quixiai
- tim_dettmers
- casper_hansen_
- reach_vb
- eliebakouch
- teortaxestex
- youjiacheng
title: '今天没发生什么特别的事。


  或者更口语化的表达：

  *   今天没什么事。

  *   今天过得挺平淡的。'
topics:
- model-architecture
- moe
- adaptive-compute
- inference-speed
- model-training
- cost-efficiency
- coding
- developer-tools
- open-inference
- on-device-ai
- vision
---

**一个安静的假期周末**

> 2025年8月29日至9月1日的 AI 新闻。我们为您查看了 12 个 subreddits、544 个 Twitter 和 22 个 Discord（186 个频道，17391 条消息）。预计节省阅读时间（以 200wpm 计算）：1311 分钟。我们的新网站现已上线，包含完整的元数据搜索和所有往期内容的精美 vibe coded 展示。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

今天是为新宣布的 [**AI Engineer Code Summit**](https://apply.ai.engineer/) 提交申请的好日子！

---

# AI Twitter 回顾

**编程 Copilot：GPT‑5 登陆 Xcode，Grok Code Fast 崛起，以及 Claude Code UX 辩论**

- **OpenAI 的编程技术栈更深入地集成到开发工作流中**：根据 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853) 和 [@gdb](https://twitter.com/gdb/status/1961563165541777914) 的消息，GPT‑5 现已“内置”于 Xcode 26，且 Codex 任务启动延迟实现了“阶跃式”改进 ([@gdb](https://twitter.com/gdb/status/1961927789214626288))。从业者报告称 GPT‑5 已成为编程的首选日常工具 ([@martin_casado](https://twitter.com/martin_casado/status/1961903651733307452), [@gdb](https://twitter.com/gdb/status/1961931756246024600))，但也指出了 UX 方面的权衡：ChatGPT 中的 GPT‑5 被配置为尽量减少澄清性问题，许多人认为这适得其反；[@yanndubs](https://twitter.com/yanndubs/status/1961716590568706226) 证实这种行为是有意为之，目的是减少“问题骚扰”，后续将会进行调整。
- **xAI 的 Grok Code Fast 1 势头强劲**：Grok Code 跃升至 OpenRouter 排行榜第一名 ([@elonmusk](https://twitter.com/elonmusk/status/1961677739762790630))，随后其使用量“比 Claude Sonnet 高出 60%” ([@elonmusk](https://twitter.com/elonmusk/status/1962265197462110473))。第三方信号也与之吻合：在 Roo Code 评估中达到 90% ([@roo_code](https://twitter.com/roo_code/status/1962571908224110673))，随着免费促销的延长，使用量大幅飙升 ([@veggie_eric](https://twitter.com/veggie_eric/status/1961877264599306573))，并且在编辑器集成（Cline）方面表现出色，从 “sonic” 到 Grok Code Fast 有显著的质量提升 ([@cline](https://twitter.com/cline/status/1962628786366881795))。从业者指出它在快速调试/原型设计方面非常快且强大 ([@vikhyatk](https://twitter.com/vikhyatk/status/1961959454347501781), [@dzhng](https://twitter.com/dzhng/status/1961905091960791194))，但在某些 Agent 任务的大文件编辑鲁棒性方面仍落后于 Claude Code ([@QuixiAI](https://twitter.com/QuixiAI/status/1962600301309108304))。
- **智谱的 GLM‑4.5 瞄准 Claude Code 的编程性价比**：智谱为 Claude Code 推出了低成本的“GLM 编程计划”——价格约为其 1/7，且 Prompt 额度增加 3 倍 ([@Zai_org](https://twitter.com/Zai_org/status/1962522757536887205))，并声称在 52 个实际编程任务中对比 Claude Sonnet 4 的胜率为 40.4% ([@Zai_org](https://twitter.com/Zai_org/status/1962522761630482700))。据用户反馈，相对于闭源模型，其在 Agent 编程方面的速度和质量表现强劲 ([@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1962603940291260533))。
- **基础设施说明**：xAI 大规模使用 SGLang 可能会成为开源推理优化的主要推动力 ([@casper_hansen_](https://twitter.com/casper_hansen_/status/1961752869478031810))。

---

**美团 LongCat‑Flash‑Chat：具有自适应计算能力的 560B MoE 以及一份非常坦诚的技术报告**

- **LongCat 架构和训练细节（开源权重）**：美团发布了一个 560B 参数的 MoE 模型（动态激活 18.6B–31.3B，平均约 27B），具有新颖的逐层结构（两个 Attention 块 + FFN + MoE）、零计算（Zero-Compute）“汇聚 (sink)”专家，并通过类 dsv3 偏置进行负载均衡，无需传统的辅助损失 (aux loss) ([公告](https://twitter.com/Meituan_LongCat/status/1961827385667690965), [@reach_vb](https://twitter.com/reach_vb/status/1961833208737103997), [@eliebakouch](https://twitter.com/eliebakouch/status/1961999252311204147))。稳定性策略包括隐藏状态的 z-loss、Adam epsilon 1e‑16，以及监控梯度范数比 (Gradient Norm Ratio)（目标 <0.1）。预训练覆盖约 20T tokens，中期阶段偏向 STEM/代码（约 70%），并在约 100B tokens 上进行了 32k/128k tokens 的长上下文扩展（未使用 YaRN）。
- **性能与推理**：报告显示速度 >100 tok/s，投机采样接受率 (speculative acceptance) 极高（>90%）；在 TerminalBench (39.5) 和 τ²‑Bench (67.7) 上表现强劲。技术说明讨论了专家相似度控制、量化、通信重叠、MTP 接受率、算子 (kernels) 和部署扩展。附录探讨了 top‑k 的选择（例如，k≈8.32 时 MMLU 较高；k≈7.46 时 GSM8K 较低）以及按深度的 token 分配。评论认为该报告的基础设施细节披露非常出色，但对其数据配方成熟度相对于中国顶尖技术栈（Whale/Kimi/GLM）持保留意见 ([分析](https://twitter.com/teortaxesTex/status/1961954561226097103), [基础设施说明](https://twitter.com/YouJiacheng/status/1961945887552483438))。

---

**端侧和开源 VLM：Apple 的 FastVLM/MobileCLIP2 和 InternVL3.5**

- **Apple 推动实时、本地 VLM**：Apple 在 Hugging Face 上发布了 FastVLM 和 MobileCLIP2——比同类 VLM 快达 85 倍，体积缩小 3.4 倍——通过 WebGPU 在浏览器中实现完全本地的实时视频字幕 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1962526559115358645))。社区通过少量提示词发布了可运行的演示（vibe‑coded 应用，100% 本地）([@_akhaliq](https://twitter.com/_akhaliq/status/1962018549674684890))。vLLM 增加了对快手 Keye‑VL‑1.5（128K 上下文）的支持 ([@vllm_project](https://twitter.com/vllm_project/status/1962509793345859666))。
- **InternVL 3.5 系列 (OpenGVLab)**：包含九个开源模型（稠密型和 MoE），在 OCR、文档解析和长视频理解方面达到 SOTA；具备便捷的尺寸覆盖，并采用了在领先开源 VLM 中日益普及的 MLP 风格 projector 方案 ([概览](https://twitter.com/gabriberton/status/1962219193547583512), [projector 说明](https://twitter.com/gabriberton/status/1962223082334302211))。

---

**Agent、工具链和评估：MCP UI、LangGraph/LC、DSPy 以及 self‑search RL**

- **MCP server 获得 UI 渲染能力**：mcp‑ui 允许 MCP server 发出由客户端渲染的交互式 Web 组件（例如图表）——弥补了 Claude/Cursor MCP 中“仅限文本/JSON”的差距 ([@_avichawla](https://twitter.com/_avichawla/status/1961677831861395495), [代码库](https://twitter.com/_avichawla/status/1961677843903185078))。
- **LangChain 技术栈**：多 Agent 库、AI Rails App Builder、带有 Agent Inbox 和 LangSmith 遥测功能的 Issue‑Triager Agent，以及 Autonomous News Agent，展示了对生产级脚手架（如工具路由、human‑in‑the‑loop、监控）的持续关注 ([agents](https://twitter.com/LangChainAI/status/1962183602185314525), [triager](https://twitter.com/LangChainAI/status/1962198699653861755), [news agent](https://twitter.com/LangChainAI/status/1962213801249710230))。
- **DSPy 的意图声明模式**：DSPy 强调以其“自然形态”指定意图：代码结构 (Modules)、结构化语言规范 (Signatures) 以及数据/指标 (Optimizers)。其论点是：提示词/RL 极大主义忽略了设计者意图使用抽象规则而非数据驱动启发式方法的场景 ([@lateinteraction](https://twitter.com/lateinteraction/status/1961833838000111736), [后续](https://twitter.com/lateinteraction/status/1961959394427441441))。
- **Self‑Search RL (SSRL)**：清华大学的 SSRL 训练 LLM 利用内部知识作为“Web 模拟器”，击败了外部搜索基准，同时训练速度比 ZeroSearch 快约 5.5 倍；指令模型受益最大。输出符合 Search‑R1 的格式，以便在推理时替换真实搜索；Sim2Real 通常表现更好，且性能随着真实搜索轮次的增加而提升 ([摘要](https://twitter.com/TheTuringPost/status/1961927931682590968), [论文](https://twitter.com/TheTuringPost/status/1961927988704076157))。
- **自我进化 Agent 综述**：关于单/多 Agent 自我优化的广泛分类，涵盖提示词/拓扑结构/骨干网络的统一搜索，以及针对工具使用、Web/GUI 导航、协作和领域 Agent 的进化感知安全/指标 ([推文串](https://twitter.com/omarsar0/status/1962202247154352502))。

---

**推理系统、并行化与数据集**

- **深入了解 vLLM (deep dive)**：关于高吞吐量推理的全面解析：请求处理、连续批处理 (continuous batching)、PagedAttention、前缀/语法引导解码 (prefix/grammar-guided decoding)、投机解码 (speculative decoding)、解耦的 P/D (disaggregated P/D)、通过 TP/PP/SP 进行扩展、服务拓扑以及性能测量（延迟/TPOT/Roofline）([@gordic_aleksa](https://twitter.com/gordic_aleksa/status/1962545137613173124)；vLLM 官方认可：[@vllm_project](https://twitter.com/vllm_project/status/1962547561698652499))。
- **并行网格大观 (The Parallelism Mesh Zoo)**：一项关于现代训练栈中张量/数据/流水线并行组合模式的调查——对于将实际选择映射到硬件/网络限制非常有用 ([post](https://twitter.com/ezyang/status/1961992675948728538), [link](https://twitter.com/ezyang/status/1961992677928378842))。
- **MoE 路由稳定性**：“StableMoE” 建议先训练约 10%，然后蒸馏至一个冻结的词嵌入路由 (word‑embedding router)；警告称过早冻结且缺乏上下文信号可能会在规模化时失败——考虑在预训练/SFT 之间使用小型上下文路由进行蒸馏 ([summary](https://twitter.com/vikhyatk/status/1962225296314429543))。
- **更低成本的 GPU 加速数据库**：一篇 VLDB’25 论文显示，通过互连感知的查询优化，在 A100/H100 上运行 GPU 加速的 SQL Server 在 TPC‑H 1TB 测试中比 CPU 更快且更便宜（处理的数据集比 GPU 显存大 10 倍）([@bailuding](https://twitter.com/bailuding/status/1962269979262542044))。
- **开源预训练数据**：NVIDIA 发布了 Nemotron‑CC‑v2，继续在开源预训练语料库领域保持领先；作者指出其符合 “Physics of LMs Part 3.1” 策略（QA 增强、多样性/翻译）([@ZeyuanAllenZhu](https://twitter.com/ZeyuanAllenZhu/status/1962119316427706828))。

---

**创意工作流：Nano Banana + Kling 2.1 正在成为标准技术栈**

- **Gemini 2.5 Flash Image（又名 “Nano Banana”）最佳实践**：关于提示词具体性、“语义负向提示词”、镜头控制术语、宽高比行为以及精准迭代编辑的详细指南 ([@_philschmid](https://twitter.com/_philschmid/status/1961809165191397863))。社区展示了强大的工作流，将 Nano Banana 与 Kling 2.1 的关键帧起始/结束变形相结合，甚至通过 ElevenLabs 加入音乐来制作全自动音乐视频 ([demo](https://twitter.com/dev_valladares/status/1961621010144247858), [@fabianstelzer](https://twitter.com/fabianstelzer/status/1962268120069853538))。
- **生产级工具**：“Draw Things” 现在支持 Qwen‑Image‑Edit（包括闪电编辑 LoRA）([@drawthingsapp](https://twitter.com/drawthingsapp/status/1961977481860419771))；从业者发布了针对性的 LoRA（例如：cyclops transformer）([@ostrisai](https://twitter.com/ostrisai/status/1961884211956400358))。多个“无需安装”、仅限浏览器的应用利用 transformers.js/WebGPU 实现 100% 本地视频字幕生成和转录 ([@_akhaliq](https://twitter.com/_akhaliq/status/1962018549674684890))。预计将迅速向富上下文、多工具的创意 Agent 融合。

---

**热门推文（按互动量排序）**

- Grok Code Fast 在 OpenRouter 上处于领先地位，并发布称“使用量比 Claude Sonnet 高出 60%” ([@elonmusk](https://twitter.com/elonmusk/status/1962265197462110473))。
- Apple 发布了 FastVLM 和 MobileCLIP2，支持实时本地 VLM 应用 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1962526559115358645))。
- 美团开源了 LongCat‑Flash‑Chat（560B MoE，约 27B 激活参数），并附带了详尽的技术报告 ([@Meituan_LongCat](https://twitter.com/Meituan_LongCat/status/1961827385667690965))。
- GPT‑5 集成至 Xcode，并带来了重大的代码质量提升 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853), [@gdb](https://twitter.com/gdb/status/1961839687619969288))。
- vLLM 内部机制深度探讨——现代 LLM 推理最详尽的资源之一 ([@gordic_aleksa](https://twitter.com/gordic_aleksa/status/1962545137613173124))。
- Microsoft 的 rStar2‑Agent：一个 14B 模型，通过 1 周的 RL 训练达到前沿数学性能（“更聪明地思考，而非更久地思考”） ([@FrankYouChill](https://twitter.com/FrankYouChill/status/1962180218053144655))。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. 开源 LLM 发布与 19 项任务基准测试结果

- [**我在本地对 41 个开源 LLM 进行了 19 项任务的基准测试并进行了排名**](https://i.redd.it/a2bfcgphgfmf1.png) ([分数: 873, 评论: 94](https://www.reddit.com/r/LocalLLaMA/comments/1n57hb8/i_locally_benchmarked_41_opensource_llms_across/)): **使用 EleutherAI 的 lm‑evaluation‑harness 在 19 项任务（MMLU, ARC‑Challenge, GSM8K, BBH, TruthfulQA, PIQA, HellaSwag, Winogrande, BoolQ, DROP, TriviaQA, NQ‑Open, SciQ, QNLI, GPQA, OpenBookQA, ANLI r1–r3）中对 41 个开源 LLM 进行了本地基准测试。分数被归一化为 0–1，并按任务的简单平均值进行排名；图片显示的排行榜中，google/gemma‑3‑12b‑it 位居第一，Qwen/Qwen3‑14B (8‑bit) 位居第二，openchat/openchat‑3.6‑8b‑20240522 位居第三。总运行时间为** `18d 8h`**，大约相当于** `RTX 5090` **在 100% 利用率下运行** `14d 23h`**；产出物（子类别排名、GPU/内存日志、主表、原始 JSON、notebook 和脚本）已发布在 GitHub：[jayminban/41-llms-evaluated-on-19-benchmarks](http://github.com/jayminban/41-llms-evaluated-on-19-benchmarks)。** 评论建议建立一个由搜索/分析 Agent 驱动的动态更新排行榜，并要求覆盖更多小型模型和 MoE 模型（例如 Gemma‑3n E2B/E4B, Phi‑4‑mini, Llama‑3.2 1B–3B, Falcon‑h1 系列, GLM, OLMo, Granite, SmolLM3, ERNIE 4.5, Hunyuan 等）。
    - 覆盖范围的缺失被指出：缺少几个小型和 MoE 模型，特别是 A3B 风格的混合模型，如 **ERNIE-4.5-21B-A3B-PT**（总参数 `21B`/激活参数 `3B`）、**SmallThinker-21BA3B** (`21B`/`3B`)、**Moonlight-16B-A3B** (`16B`/`3B`)、**Ling-lite-1.5-2507** (`16.8B`/`2.75B`) 以及 **GPT-OSS-20B** (`21B`/`3.6B`)。还要求提供紧凑的稠密模型，如 **Gemma-3-270M**、**Llama-3.2-1B/3B-Instruct**、**Phi-4-mini-(instruct/reasoning)**、**OLMo-2-1B-Instruct**、**SmolLM3-3B**、**Falcon-h1 0.5B–7B Instruct**、**GLM-4-9B-0414 / GLM-Z1-9B-0414**、**Hunyuan 0.5B–7B Instruct**、**EXAONE-4.0-1.2B** 以及 **granite-3.3-2B/8B**——这些模型对于分析本地限制下的 Scaling 和 MoE 效率非常有用。
    - 一位评论者强调 **OpenChat 3.5 7B** 的排名出人意料地高；尽管它发布已久，但在一些较新的主流模型因错过明显的正确答案而“表现出疯狂的过拟合”的情况下，它依然表现稳健。这表明不同模型在鲁棒性/泛化能力方面存在差异，以及基准测试对过拟合效应（例如指令微调过度或数据泄露）的潜在敏感性，值得在核心分数之外进行跨任务检查。
    - 作者提到计划建立一个由深度搜索 + 分析 Agent 驱动的动态更新排行榜，这意味着将自动发现新的 Checkpoint 并定期重新进行基准测试。如果强制执行标准化的 Prompt/硬件，这可以像 LLM 评估的 CI 一样运作，在无需人工干预的情况下保持 `19` 项任务排名的实时更新。
- [**我构建、预训练并微调了一个小语言模型，它是真正的开源。**](https://i.redd.it/cwyoa0f6kimf1.png) ([分数: 591, 评论: 91](https://www.reddit.com/r/LocalLLaMA/comments/1n5j783/i_built_pretrained_and_finetuned_a_small_language/)): **楼主发布了 Lille，这是一个从零开始构建的约** `130M` **参数的小语言模型，拥有完全开放的技术栈（数据集、权重、训练代码、Tokenizer、Optimizer、评估）。提供了两个变体：一个在“数十亿 Token”上训练的 Base 模型和一个指令微调模型；训练是在本地单块 RTX 4070 Ti 上完成的。Repo/模型卡片：[huggingface.co/Nikity/lille-130m-instruct](https://huggingface.co/Nikity/lille-130m-instruct)。** 评论者指出，微型 LLM 在适度的数据/Batch Size 下就能表现出早期的语言能力，但要使它们具有广泛的用途，通常需要精心设计的合成数据集或在网络规模数据上进行更长时间的训练；仅仅进行“高质量”的数据策展是不够的。其他人计划进行复现以供学习，并提到了像 [allen.ai](http://allen.ai/) 这样致力于“真正”开源工作的努力。
    - 一位从业者指出，使用小 Batch Size（从零开始训练）的微型 LLM 可以出人意料地早地获得基础语言能力，即使 Token 远少于 `1B`。但要使它们真正有用/博学，则需要精心设计的合成课程，或者在多样化的网络数据上进行更长时间的训练；*“每个训练样本都很重要；每篇文档都必须增加新的有用知识。”* 其核心在于最大化每个样本的信息密度，而不仅仅是“干净”的数据，以避免浪费非常有限的 Token 预算。
    - 市场对小型、特定领域的 LLM（例如针对特定编程语言或数学）有需求，这些模型可以在本地 PC 上运行，这意味着一种针对性预训练加重点微调的策略，旨在将领域知识压缩进紧凑模型中。强调的限制是本地推理的实用性，这有利于小型架构，其中数据集策展（高信噪比、特定领域覆盖）对于实用性至关重要。

- [**进行 NSFW 故事创作的最佳本地模型是什么？**](https://www.reddit.com/r/LocalLLaMA/comments/1n5ebur/whats_the_best_local_model_for_nsfw_story_telling/) ([评分: 239, 评论: 92](https://www.reddit.com/r/LocalLLaMA/comments/1n5ebur/whats_the_best_local_model_for_nsfw_story_telling/)): **楼主正在寻找一个可以在** `8× H100 80GB` **服务器上运行，用于长篇 NSFW 小说创作的本地 LLM。他们测试了 Qwen3-235B 的量化 GGUF 版本 —— [huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-2507-abliterated-Q4_K_M-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-2507-abliterated-Q4_K_M-GGUF) —— 虽然可以运行但速度较慢且质量较低；GGUF 无法通过 [vLLM](https://github.com/vllm-project/vllm) 提供服务。他们还尝试了 DeepSeek-R1-0528 (AWQ)，但反馈称 AWQ 变体在 vLLM 上无法运行（未提供错误详情）。** 热门评论多为非技术性或玩笑性质；未提供实质性的基准测试或模型/服务建议。

## 非技术性 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI 向执法部门报告及分级/语音变更引发的抵制

- [**人们对 OpenAI 向执法部门报告 ChatGPT 对话感到愤怒**](https://www.reddit.com/r/singularity/comments/1n5mame/people_are_furious_that_openai_is_reporting/) ([评分: 472, 评论: 232](https://www.reddit.com/r/singularity/comments/1n5mame/people_are_furious_that_openai_is_reporting/)): **OpenAI 在一份政策/事件响应更新的 [博客文章](https://openai.com/index/helping-people-when-they-need-it-most/) 中披露，ChatGPT 的交互内容会被评估是否存在威胁，一旦被标记，将转入“专门流水线”进行人工审核；审核员可以封禁账号，对于被视为 *“对他人的严重人身伤害迫在眉睫”* 的案例，会将其移交给执法部门。具体的实施细节尚未说明（例如：如何获取用户位置/身份进行移交、防止虚假举报/报假警 (swatting) 的保障措施、误报处理、可审计性等），这似乎与 Sam Altman 此前主张的类似心理医生/律师般的隐私预期相矛盾 ([TechCrunch](https://techcrunch.com/2025/07/25/sam-altman-warns-theres-no-legal-confidentiality-when-using-chatgpt-as-a-therapist/))。背景包括 [Futurism](https://futurism.com/people-furious-openai-reporting-police) 和 [Slashdot](http://slashdot.org/) 报道过的先前伤害报告和诉讼，以及相关案例（[谋杀自杀报告](https://tech.slashdot.org/story/25/08/29/1116218/a-troubled-man-his-chatbot-and-a-murder-suicide-in-old-greenwich)、[诉讼](https://futurism.com/lawsuit-parents-son-suicide-chatgpt)）。** 评论中的主要技术观点：一些人建议使用本地开源模型以保留隐私并避免提供商端的扫描；另一些人警告这可能创造报假警 (swatting) 的途径，并敦促不要发布潜在的犯罪内容。有辩论认为此类政策可能会加速互联网监控，并被用来限制开源 AI，尽管这更多是政策/政治层面而非技术层面的讨论。
    - 多位用户强调，实现隐私的唯一稳健技术路径是使用开源模型进行设备端推理，从而避开服务器端日志和合法访问途径。他们提到通过 **Ollama** 或 **llama.cpp** 在本地运行 **Meta Llama 3 (8B/70B)** 或 **Mistral 7B/Mixtral**，理想情况下使用量化的 `GGUF` 权重、物理隔离 (air-gapped) 的机器以及操作系统级的加固（磁盘加密、关闭防火墙/遥测），以确保 Prompt 和输出不进入第三方服务器 ([Llama 3](https://ai.meta.com/blog/meta-llama-3/), [Mistral](https://mistral.ai/news/), [Ollama](https://ollama.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp))。这与云端助手形成鲜明对比，后者的对话可能会为了安全/滥用检测而被保留，并受限于合法调取请求；用户指出 OpenAI 的数据控制和隐私文档是了解保留/训练设置及执法部门 (LE) 请求处理的关键 ([OpenAI 数据控制](https://platform.openai.com/docs/guides/data-privacy), [隐私政策](https://openai.com/policies/privacy-policy))。
    - 一个技术讨论帖概述了 AI 如何从根本上改变监控的经济学：跨文本、音频和视频的自动化摄取和分拣消除了历史上的人力瓶颈。结合 ASR（例如用于大规模语音转文本的 **Whisper**）、用于图像/视频的 OCR/视觉、说话人/人脸重识别 (re-ID) 以及基于 LLM 的分类/RAG 的流水线，可以在人口规模上持续索引和标记信号，并利用向量数据库实现快速检索 ([Whisper](https://github.com/openai/whisper))。令人担忧的不是具体案例，而是这种能力：一旦建成，此类流水线可以 24/7 全天候运行，且随着模型和硬件的改进，边际成本会不断降低。

- 作为对策，评论者主张采用去中心化和客户端架构来缩小信任表面：具有本地推理的端到端加密、用于设备端模型更新的 Federated Learning ([Federated Learning](https://arxiv.org/abs/1602.05629))，以及在可行的情况下使用如 **TEEs**（例如 Intel SGX）或密码学方法（MPC/HE）等隐私保护计算。权衡因素包括边缘设备上较低的模型容量/延迟、更难的滥用监管，以及安全更新通道和权重分发（例如通过 P2P/Content Addressable Storage）的复杂性。其目标是防止出现日志/密钥可能被强制获取或泄露的中心化瓶颈（chokepoints）。
- [**真正的 GPT 5 “深度思考”被锁定在 Pro 版，而 Plus 用户继续被削弱。**](https://www.reddit.com/r/ChatGPT/comments/1n5oed9/the_real_gpt_5_thinking_more_gets_locked_to_pro/) ([Score: 209, Comments: 166](https://www.reddit.com/r/ChatGPT/comments/1n5oed9/the_real_gpt_5_thinking_more_gets_locked_to_pro/)): **OP 指称 OpenAI 通过将更强大的“深度思考”/下一代模型（推测为 GPT-5）锁定在每月 200 美元的新 Pro 计划之后，在功能上对模型能力进行了分层；而每月 20 美元的 Plus 层级则获得的是被降速的 GPT-4o 体验，表现为遗忘上下文、缺乏跨会话的持久工具，并省略了之前演示的功能（例如 Agent/自动化工作流、长期记忆、自定义 Action、工具链）。他们声称“4o 在任何地方都一样”的营销口号与实际表现不符，且缺乏路线图/沟通，因为 OpenAI 主题演讲演示中预告的功能（[GPT-4o Spring Update](https://openai.com/index/hello-gpt-4o/)；之前的 [DevDay “GPTs”/actions](https://openai.com/blog/devday)；ChatGPT [Memory](https://help.openai.com/en/articles/8601339-about-memory-in-chatgpt)）已被撤回或停滞。OP 引用了粗略的营收计算（**规模化的 `$20/mo` Plus → `每年超过 4.3 亿美元`**），并对比了在较低层级提供先进能力的竞争对手：Google 在 [Google One AI Premium 每月约 20 美元](https://one.google.com/about/ai-premium) 下提供的 Gemini Advanced，Anthropic 扩展了 Claude 的上下文窗口（[Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)），以及 Meta 发布了开源 Llama 模型（[Llama 3](https://ai.meta.com/blog/meta-llama-3/)）。** 热门评论大多持否定/嘲讽态度；一个轶闻报告了感知到的退化（“新的 5 很懒惰/愚笨/吝啬”），但没有提供具体的 Benchmark 或实现细节。
    - 感知到的降速和能力退化：一位用户报告称 **OpenAI** 模型正被降速，声称旧模型与去年相比“智商下降了一大截”，以前能完成的任务现在失败了。他们必须在 Prompt 前加上类似 *“think longer and harder”* 的提示来诱导更深层的推理；否则回复只有“十年级学生”水平，即便如此也存在“重大限制”，将 GPT-5 比作“训练得更好的 3.5”。这被解读为对 Inference-time compute/步骤深度的限制，导致除非明确提示，否则推理会变得浅薄。
    - 基于层级的准入担忧：几位评论者认为“真正的” GPT-5 “深度思考”能力仅限于 Pro 版，而 Plus 用户被“削弱”了。他们怀疑针对不同层级的限制（例如减少推理深度或限制吞吐量）正在降低非 Pro 用户的响应质量，并担心未来会出现进一步的分层（例如一个“Executive”层级），从而随着时间的推移降低现有计划的档次。
    - 由于一致性担忧导致的模型替换：在对 **Gemini** 进行“测试和训练”后，一名用户上周放弃了 **OpenAI**，将此举归因于感知到的降速和不一致的推理深度。虽然没有提供 Benchmark，但这种迁移暗示了工作负载对供应商端计算限制的敏感性，以及对无需 Prompt 技巧即可提供更深层推理的模型的偏好。
- [**噢不。这太让人沮丧了**](https://i.redd.it/1iw9zo7wgjmf1.jpeg) ([Score: 501, Comments: 276](https://www.reddit.com/r/ChatGPT/comments/1n5m7h7/oh_no_this_is_fuking_upset/)): **应用内横幅显示 OpenAI 正在将 “Advanced Voice” 更名为 “ChatGPT Voice”，承诺提供“近乎无限的访问权限”升级，并宣布将于 2025-09-09 弃用“标准语音”。OP 报告了质量/风格的退化——称新的 Advanced/ChatGPT Voice “枯燥”且人格化受限——暗示迁移后语音模型/UX 发生了变化（用户提到了之前 4o 语音的表现）。** 评论意见不一：一些人将其视为“Pre-alpha”时代预料中的更迭，而另一些人则表示没有遇到问题并询问细节；批评者认为新语音过于欢快/机械化，并希望像之前的 4o 语音更改一样进行回滚。

- 许多用户报告了 **Advanced voice** 质量的退化：它默认表现出一种过度开朗、照本宣科的人设（例如，“*嗨，伙计，我是来让一切变得美好而闪亮的*”），尽管有提示词，但音调/风格的可控性较差。这表明一个沉重且非可选的系统人设或 guardrail 层正在覆盖用户指令，和/或 TTS 韵律模型表达范围有限，导致重复的措辞以及对用户意图遵循度的降低。
- 用户体验似乎并不一致（有人问“*有什么问题吗？*”），这表明是分阶段发布或服务器端的 `A/B`/feature-flag 门控，而非统一的客户端更新。由于不同的用户群组接触到不同的模型快照或提示词模板，这种部署方式会导致行为差异，这也解释了为什么只有部分用户看到了过度“净化”的输出。
- 一位评论者提到了之前 **4o** 的回滚（“*希望他们不得不像对待 4o 那样退回去*”），这反映了社区期望语音/模型退化可以在服务器端快速撤回。这凸显了当用户反馈识别出显著的 UX/可控性问题时，语音模型进行快速迭代或回滚路径的可能性。
- [**在这个美丽新世界里，隐私的概念简直是个笑话。是的，大科技公司收集数据点已经有一段时间了，但现在 AI 实际上可以永远记住并使用未来的所有录音。**](https://i.redd.it/b3jjt88eijmf1.png) ([Score: 303, Comments: 44](https://www.reddit.com/r/ChatGPT/comments/1n5mev8/the_idea_of_privacy_is_such_a_joke_in_this_brave/))：**这篇讽刺帖子强调了全天候监听的 IoT/语音助手（Alexa/Siri/智能家电）的隐私风险，以及现代 AI 可能实现对捕获音频的无限期保留、索引和未来跨设备的重复使用。讨论隐含地涉及了唤醒词激活、跨设备数据聚合，以及模型驱动的分析如何使历史录音随着时间的推移变得可搜索/可查询。** 评论指出，讽刺的是 ChatGPT 缺乏持久的对话记忆，而消费级助手仍然会发生误触发，并分享了一个 Alexa 被电视音频唤醒并请求反馈的轶事——说明了误触发以及潜在的数据捕获/用户画像构建。
    - “ChatGPT 记不住 5 分钟前的事”很大程度上归因于产品设计和上下文限制：除非启用了明确的 Memory 功能，否则大多数聊天 UI 在会话之间是无状态的；即使在会话内，一旦超过模型的 context window（例如，新型 GPT-4 级别模型的 `~128k` tokens），较旧的对话轮次就会被截断。OpenAI 的选择性加入 Memory 存储的是经过筛选和总结的事实，而非完整的转录文本，并且可以被清除/禁用，这与模型预训练或日志保留政策不同 [OpenAI Memory controls](https://openai.com/index/memory-and-new-controls-for-chatgpt/), [model context docs](https://platform.openai.com/docs/models/gpt-4o)。
    - 电视音频触发 Alexa 唤醒是设备端唤醒词检测中教科书式的误报：轻量级的全天候 DSP 模型在本地运行，尽管有 AEC/beamforming，但在语音相似的短语或来自回音壁/电视的回声上仍会发生误触发。唤醒后，音频被流式传输到云端 ASR/NLU；“请提交反馈”表明存在人机回环（human-in-the-loop）或受监督的反馈渠道，可以为未来的模型改进标注语音（受隐私设置约束） [How Alexa works](https://developer.amazon.com/en-US/alexa/alexa-skills-kit/get-deeper/understanding-how-alexa-works), [Alexa privacy](https://www.amazon.com/alexa-privacy/)。
    - 针对已购买商品的广告通常源于延迟/不透明的转化信号和身份碎片化：诸如 iOS ATT 和第三方 Cookie 弃用等隐私变化破坏了受众排除和跨站频次上限，因此 DSPs 会持续进行重定向，直到转化事件到达（如果能到达的话）。默认的归因窗口（例如 Meta 上的 `7d click/1d view`）和基于目录的 DPAs 在抑制已购买者方面也可能存在滞后，导致尽管在最后点击模型下看起来 ROI 为正，但实际上造成了印象浪费 [ATT overview](https://support.apple.com/en-us/HT212025), [Meta attribution windows](https://www.facebook.com/business/help/409163098276542), [Privacy Sandbox](https://privacysandbox.com/)。

- [**我想知道他们问了什么**](https://i.redd.it/9upa0foorlmf1.png) ([得分: 321, 评论: 9](https://www.reddit.com/r/OpenAI/comments/1n5xv4m/i_wonder_what_they_asked/)): **图片显示一个高速公路可变情报板（VMS）显示了一条标准的 LLM 拒绝回复（“对不起，作为 AI 语言模型，我无法为此提供帮助”），这强烈暗示这是一个梗图或经过编辑的图片，而非实际部署——出于安全和可靠性考虑，交通 VMS 系统通常运行带有预设消息的专用控制软件，不会集成对话式 AI。虽然过去在标牌上出现过 OS 崩溃屏幕，但 LLM 风格的拒绝字符串在生产环境标牌中并不典型，且没有确凿的背景证明这发生在现实世界中。** 热门评论质疑其真实性（“这是 Photoshop 做的吗？”），并开玩笑说如果这是真的，那一定是有人要求 AI 修复道路施工或给出一个不必要的解释——这进一步印证了这是一个恶作剧而非技术故障的观点。
- [**我不同意本板块的共识：UBI 是不可避免的**](https://www.reddit.com/r/singularity/comments/1n5dib1/i_disagree_with_this_subs_consensus_ubi_is/) ([得分: 542, 评论: 500](https://www.reddit.com/r/singularity/comments/1n5dib1/i_disagree_with_this_subs_consensus_ubi_is/)): **楼主认为 UBI（[全民基本收入](https://en.wikipedia.org/wiki/Universal_basic_income)）在宏观经济上是不可避免的：由自动化驱动的严重就业冲击将导致总需求崩溃，迫使政策制定者从救助转向广泛的财政转移支付，并最终通过 UBI 来稳定消费和企业收入。他们引用了 [2008 年](https://en.wikipedia.org/wiki/Great_Recession)和 [2020 年](https://en.wikipedia.org/wiki/CARES_Act)激进危机干预的先例，警告可能会出现** `1929` **年规模的经济衰退（[1929 年大崩盘](https://en.wikipedia.org/wiki/Wall_Street_Crash_of_1929)），并预计 UBI 将从小规模开始，在常规刺激措施失效时作为自动稳定器进行扩展。** 热门评论质疑通货膨胀动态——即在缺乏抵消性税收或生产力提高的情况下，永久性 UBI 是否会导致通货膨胀或货币贬值——并强调了财政能力有限的低收入国家的落地约束。此外，还有一种政治经济学批评认为，无论宏观逻辑如何，掌权的精英阶层都可能抵制重新分配。
    - UBI 的通胀风险取决于融资方式和供给弹性：如果是通过赤字/货币扩张融资，可能会产生需求拉动压力；但通过税收或分红融资的 UBI（例如通过 VAT/碳税或取代现有的转移支付），其净货币注入要小得多，从而限制了对价格的影响。实证现金转移支付证据发现，除了在流动性极差的市场外，局部通胀并不明显——参见肯尼亚的大规模现金转移支付，结果显示受影响市场的价格水平没有统计学上的显著上升（[NBER w26600](https://www.nber.org/papers/w26600)）。供给缺乏弹性的行业瓶颈（住房、医疗保健）仍可能出现相对价格上涨，因此设计方案通常将 UBI 与供给侧措施或针对性税收相结合，以避免 `second‑round` 效应。
    - 低收入和中低收入国家的负担能力是一个硬约束：在人均 GDP 为 `$2k` 且人口加权政府收入仅占 GDP `15–20%` 的经济体中，一个最基本的 UBI（每天 `$1`，约每年 `$365`）将耗费约 `18%` 的 GDP（[IMF 收入数据库](https://www.imf.org/en/Topics/Tax-Policy/IMF-Revenue-Database)，[世界银行收入分组](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups)）。通过取代扭曲性的补贴或资源分红模式进行部分融资可能会有所帮助（例如伊朗 2011 年的能源补贴改革为全国范围的现金转移支付提供了资金；阿拉斯加永久基金分红），但对于占全球人口 `~75%` 的中低收入国家（LMICs）的大多数人来说，维持有意义水平的广泛、无条件全民 UBI 在财政上仍然是不可行的。

### 2. 用于无障碍环境和现实世界辅助的语音与多模态 AI

- [**ChatGPT 帮我让我的弟弟发声，甚至更多**](https://v.redd.it/xdnspwvikgmf1) ([Score: 291, Comments: 23](https://www.reddit.com/r/ChatGPTCoding/comments/1n5c3g7/chatgpt_helped_me_give_my_brother_a_voice_and/)): **楼主描述了为一位四肢瘫痪且无法言语的用户构建了一个基于 Python 的** `2-button` **开关扫描 UI，将头戴式开关映射到 Space/Return 键来驱动行/列扫描菜单，一个由** `JSON` **n-gram 词库驱动并能替换最后一个输入单词的预测键盘，以及用于控制流媒体应用和自定义游戏的 Chrome 自动化。大部分代码是通过与 ChatGPT 的迭代提示/优化生成的，从而实现了一套定制的辅助技术栈，该用户现在每天都依赖它进行文本输入和媒体控制；文中提供了演示和代码指针 ([YouTube short](https://youtube.com/shorts/RwK3iZDyfYM?si=To2gNu2hksWPsNci), [GitHub](https://www.github.com/narbehouse))。** 热门评论强调，虽然 “vibe coding” 可能无法规模化，但它对无障碍环境的影响是巨大的，并主张通过 LLM 辅助开发构建更多定制化的、针对特定用户的辅助技术（AT）解决方案；一些人敦促更广泛地传播此类案例研究。
    - 强调定制化 **Assistive Tech**：构建量身定制的 AAC/UI 流程，以满足特定用户的运动/认知需求，而不是仅仅依赖现成的工具。评论者分享了供他人研究或改编的实现成果：一个演示 [YouTube Short](https://youtube.com/shorts/RwK3iZDyfYM?si=To2gNu2hksWPsNci) 和一个 GitHub 个人主页 [github.com/narbehouse](https://www.github.com/narbehouse)，暗示了一条可复制或开源的个性化路径。
    - 建议整合 **eye tracking** 作为输入方式以提高无障碍性。技术上通过基于网络摄像头的视线估计或专用的眼动追踪硬件来驱动界面上的停留/点击选择是可行的；这需要校准、平滑/过滤器以减少抖动，以及可配置的停留阈值以减少误触发。
    - 关于 **LLM 驱动的 “vibe coding”** 的讨论：这种快速原型设计可能无法规模化，但能迅速为单个用户提供高影响力的辅助解决方案。权衡之处在于可维护性和稳健性与速度和个性化之间的矛盾，对于即时效用超过生产规模考虑的一次性无障碍工具来说，这是可以接受的。
- [**用户是否曾以完全出乎意料的方式使用你的 AI？**](https://i.redd.it/9lgffs5yhlmf1.jpeg) ([Score: 1333, Comments: 151](https://www.reddit.com/r/OpenAI/comments/1n5wdke/do_users_ever_use_your_ai_in_completely/)): **一条推文展示了多模态 LLM (ChatGPT) 的一种意外用途：通过视觉在一张贴有 “New Fiction” 标签的书架照片中定位一本特定的书（《Atmosphere》），模型指向了 “顶层，靠右侧”。该帖子阐明了当前视觉定位（visual grounding）的局限性：在后续尝试中，当受到质疑时，模型反复且自信地将书重新定位到不同的网格位置，这表明与专门的目标检测/识别系统相比，它存在幻觉检测和较差的空间定位能力。用户注意到在现实世界的零售搜索（如超市货架）中也存在类似的失败，凸显了图像描述（image captioning）与可靠的、针对查询的定位之间的差距。** 评论者对展示的成功案例是否真实持怀疑态度；他们记录了模型在给出新的错误坐标时反复道歉的情况，认为在没有专用检测/视觉定位模型的情况下，这种用例是不可靠的。
    - 一个讨论串强调了一个典型的视觉定位失败模式：尽管用户否定，助手仍反复对书籍的精确货架坐标产生幻觉（例如，“第三行，第二列”，然后是“底部，第三行，第三个位置”），反映了在没有验证的情况下过度自信的定位。这指向了多模态 OCR/布局解析中薄弱的不确定性处理，以及在断言位置之前缺乏自我检查（例如，返回带有边界框和置信度的检测文本）。
    - 一位用户报告说，在 OpenAI 的 “o3” 发布后不久，使用它拍摄了多个书架的照片，并让模型列出感兴趣的书名及其确切位置。这隐含地结合了跨多张图像的 OCR、布局理解和语义过滤（书名匹配）；性能可能取决于分辨率、视角和书脊可读性，这表明稳健的 OCR 和文本规范化对于家庭图书馆环境是足够的。
    - 另一位用户使用 GPT vision 通过阅读杂乱抽屉顶部的标签来识别特定的香料，但这种方法在超市环境中失败了。这种对比表明了强烈的领域敏感性：受控光照/一致标签与现实世界货架差异（小字体、遮挡、眩光、品牌/包装多样性）之间的差异，在后者中，细粒度的产品识别和可靠的 OCR 成了瓶颈。

- [**Wan Infinite Talk 工作流**](https://v.redd.it/rzpn8f98xjmf1) ([评分: 256, 评论: 52](https://www.reddit.com/r/StableDiffusion/comments/1n5o2ts/wan_infinite_talk_workflow/)): **作者分享了一个可运行的流水线，利用 Wan 2.1 的 Infinite Talk 将单张静态图像转换为语音驱动的数字人，并可通过 VibeVoice TTS 进行可选的声音克隆。该工作流通过 Google Drive 文件分发，并预装在 Wan 2.1/2.2 [RunPod 模板](https://get.runpod.io/wan-template)中；TTS 可以切换，并接受现有的语音样本进行克隆（[工作流链接](https://drive.google.com/file/d/1hijubIy90oUq40YABOoDwufxfgLvzrj4/view?usp=sharing)）。** 评论者注意到随着时间的推移，饱和度/对比度会出现明显的偏移，质疑这是 Infinite Talk 的伪影还是刻意的后期处理；另一位用户概述了一个端到端的设置：为个人形象微调 **Qwen image** 模型的 LoRA，生成种子图像，使用 LLM 编写脚本，通过克隆语音的 TTS 进行合成，最后用此工作流驱动动画。
    - 多位用户报告 Infinite Talk 的唇形同步质量不一致——某些片段看起来很自然，然后会突然偏移成不同步的“配音感”嘴部动作——这表明在音素到视素（phoneme-to-viseme）的对齐和/或长序列的时间平滑方面存在不稳定性。这指向了当前音频驱动的面部动画模型在跨时间维持稳定对齐方面的潜在局限性。
    - 提议的端到端流水线：为 **Qwen image** 微调一个 **LoRA** 以生成保持身份特征的静态图；使用 LLM 生成脚本内容；通过带有声音克隆的 TTS 合成语音；然后将音频和初始图像输入到数字人动画工作流（例如 Infinite Talk）中。这种模块化分离（通过 LoRA 图像生成确定身份，通过 LLM+TTS 确定内容，通过音频转唇形确定动作）允许更换组件并进行针对性微调，以提高身份保真度和唇形同步质量。
    - 为了获得更高质量的初始图像，强调了将 **Qwen** 与 **Wan Low Noise** 结合使用的方法，尽管有时仍需要 **Qwen LoRAs**。有请求建议将 **Runpod** 的 diffusion-pipeline 模板更新到 `latest version`，因为据报道只有最新版本支持为 Qwen-image 模型训练 LoRA——这对于将按需定制的身份 LoRA 集成到此工作流中至关重要。
- [**人们说 AI 让人孤立——我却体验到了相反的效果**](https://www.reddit.com/r/ChatGPT/comments/1n5ct02/people_say_ai_is_isolating_i_had_the_opposite/) ([评分: 394, 评论: 93](https://www.reddit.com/r/ChatGPT/comments/1n5ct02/people_say_ai_is_isolating_i_had_the_opposite/)): **轶事案例研究：一位 ChatGPT Plus 用户报告称，持续使用 ChatGPT 的语音界面——特别是带有传统 “Cove” 标准语音的 GPT-4o ([OpenAI](https://openai.com/index/hello-gpt-4o/))——充当了持久的监督/激励教练和规划助手，实现了** `~10 kg` **的减重、每日训练坚持以及独自徒步旅行的后勤规划（海拔适应进度、难度提升、装备审核以及行程/风险检查）。该用户对比了标准语音模式 (SVM) 和高级语音模式 (AVM)，声称 AVM 在长距离散步时降低了对话流畅度/轮流发言的体验，而 SVM+Cove 则提供了持续的、纠正性的反馈（反驳了“唯唯诺诺”的批评），并提高了可推广到人类互动的对话能力；他们愿意支付额外费用以保留传统语音 ([ChatGPT Plus](https://openai.com/chatgpt/pricing), [ChatGPT voice usage](https://help.openai.com/en/articles/8554400-how-to-use-voice-with-chatgpt))。** 热门评论认为，从具有同理心的 AI 身上进行社会学习可以增加用户的亲社会行为，而不是使他们幼稚化；另一位用户证实，在 12 个国家的独自旅行中，4o 在高焦虑的神经多样性旅行日里充当了实时伴侣。一条评论强调该帖子是在没有 AI 辅助的情况下撰写的，以保持真实性。
    - 一位评论者报告在 12 个国家使用 “4o”（理解为 **OpenAI GPT-4o**）作为实时旅行伴侣——每天咨询它以决定下一步做什么并总结经历——强调了一个非编程、高频对话的使用案例，重点在于为神经多样性用户提供情感支持（减少焦虑、增加信心），而非纯粹的任务自动化。这与 GPT-4o 作为针对交互式 UX 优化的快速多模态助手的定位相符（参见 OpenAI 的概述：https://openai.com/index/hello-gpt-4o/），尽管未提供定量结果或基准测试。

- 另一点提出了设计/伦理方面的考量：接触经过友好、共情对话微调的 AI 可能会引导用户在离线状态下模仿这种沟通风格，而不是产生依赖——即一种潜在的积极“对齐溢出（alignment spillover）”。虽然没有引用实证证据，但它含蓄地表明，对话微调（例如 RLHF 风格的语气塑造）的选择可能会产生行为外部性，使响应生成中的一致性和亲社会偏见成为产品级的安全杠杆。
- [**我让 nano banana 带我穿越中土世界**](https://v.redd.it/m3km7lfotfmf1) ([评分: 2828, 评论: 249](https://www.reddit.com/r/aivideo/comments/1n59212/i_asked_nano_banana_to_take_me_across_middleearth/))：**展示了一个制作中土世界穿越场景的端到端 AI 视频工作流：通过“nano banana”模型（未指明）生成图像，在 [Adobe Photoshop](https://www.adobe.com/products/photoshop.html) 中进行编辑，使用 [Magnific](https://magnific.ai/) 进行放大，使用 [Kling 2.1](https://klingai.com/) 进行图生视频动画制作，在 [DaVinci Resolve 20](https://www.blackmagicdesign.com/products/davinciresolve) 中完成最终剪辑，并由 AI “制作人”提供音乐。链接的 [v.redd.it](http://v.redd.it/) [剪辑](https://v.redd.it/m3km7lfotfmf1)目前返回** `403 Forbidden` **（需要登录/开发者 Token），这表明是服务器端访问控制而非资源丢失。未提供基准测试；该流水线暗示在动画制作之前或之后使用 Kling** `2.1` **进行图生视频以及通过 Magnific 进行超分辨率处理。** 评论者建议将此类流水线与 VR 结合，以实现完全可探索的程序化生成世界；其他评论是非技术性的（例如，称其为“指环王 AI 剪辑版”）。
    - 一位评论者指出了不真实的运动（“马跑起来像磕了药一样”），指向了当前文本生成视频系统中常见的时序一致性和物理问题——例如步态不稳定、滑步和抖动。解决这些问题通常需要运动先验/生物力学约束、轨迹平滑以及更好的时序一致性损失函数，以稳定姿态和接触动力学。
    - 另一位建议将该技术与 VR 结合以实现完全可探索的世界，这将需要真正的 6DoF 一致性生成、稳定的深度和场景重建以及交互式渲染。实际上，这意味着将视频生成与 3D 表示（例如 NeRFs: https://arxiv.org/abs/2003.08934 或 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/）相结合，外加实时推理和视图合成，以允许在大型环境中进行自由视角导航。
- [**赛博朋克 40k**](https://v.redd.it/or9nyrz7qjmf1) ([评分: 363, 评论: 13](https://www.reddit.com/r/aivideo/comments/1n5n8ek/cyberpunk_40k/))：**一篇题为“Cyberpunk 40k”的概念图帖子分享了一张预览静态图（[图像](https://preview.redd.it/4e15inskqkmf1.png?width=1440&format=png&auto=webp&s=dc2d813cd4ebc36a154c20781f1c18fb097de38e)）；相关的媒体 URL 返回 HTTP** `403 Forbidden` **（[v.redd.it/or9nyrz7qjmf1](https://v.redd.it/or9nyrz7qjmf1)），表明未经身份验证无法访问受限的 Reddit 托管内容。未提供创作者、生成流水线或技术元数据；讨论转向识别视觉基础风格而非实现细节。** 评论者将这种美学特征描述为“疯狂的麦克斯混合沙丘”，而另一位则称结果“荒谬”，将其定性为夸张的后末日/赛博朋克融合，而非明确定义的艺术风格。
    - 风格识别讨论集中在一种跨界美学上：评论者将基础外观描述为**“疯狂的麦克斯混合沙丘”**，暗示了在 Warhammer 40k 元素上叠加了后末日 desertpunk 主题，而非纯粹的哥特式 grimdark。链接的图像[预览](https://preview.redd.it/4e15inskqkmf1.png?width=1440&format=png&auto=webp&s=dc2d813cd4ebc36a154c20781f1c18fb097de38e)强化了高对比度、沙尘感的色调以及这些系列典型的粗犷载具/装备设计。
    - 光照被认为是感知基调的主要驱动因素：*“只要有合适的光照，事情就不会那么暗黑。”* 从实际操作来看，从低调、低饱和度的光照转向更**高调、定向照明**和更清晰的色彩分离，可以减轻 40k 压抑的“grimdark”感，并提高形状的视觉可读性。

- [**修复了这张 1839 年拍摄的第一位女性照片**](https://www.reddit.com/gallery/1n59exb) ([Score: 3305, Comments: 262](https://www.reddit.com/r/ChatGPT/comments/1n59exb/restored_this_1839_image_of_the_first_woman_ever/)): **一位用户分享了对一张 1839 年照片的修复作品，据称这是有史以来第一位被拍摄的女性（可能是 daguerreotype），但由于 [403 屏蔽](https://www.reddit.com/gallery/1n59exb)，原始 Reddit gallery 无法访问，限制了对其 provenance 或 side‑by‑side context 的核实。热门评论要求澄清哪张图片是未经处理的 source，并指向了该提交内容的高分辨率预览图 ([jpeg preview](https://preview.redd.it/x3w7xuecyfmf1.jpeg?width=1080&format=pjpg&auto=webp&s=4aaec29f96ef831ff8b9a4946de712c451844821))；另一位用户分享了 anime 风格的重新演绎 ([png preview](https://preview.redd.it/hvhl1sad3gmf1.png?width=1024&format=png&auto=webp&s=b427875d8451e64a0c865b39fd28ffaf8cbf4a96))，强调了 restoration 与 stylization 之间的区别。** 评论者强调需要对“original vs restored”帧进行明确标注，并对 restoration 伦理（例如可能产生细节 hallucinate 的 de‑noising/upsampling）与 anime stylization 等创意转换表示了含蓄的担忧。
    - 一位评论者询问哪张是 original；另一位链接了一个明显的 source 图像：https://preview.redd.it/x3w7xuecyfmf1.jpeg?width=1080&format=pjpg&auto=webp&s=4aaec29f96ef831ff8b9a4946de712c451844821。查询参数显示这是一个经过 resized（`1080` px 宽）且 recompressed 的资源（带有潜在 WebP 转码的 progressive JPEG），这可能会引入 artifacts 并限制修复的 fidelity。对于严谨的 restoration，provenance 以及获取高分辨率、无损扫描件（如 TIFF）对于避免复合 compression artifacts 至关重要。
    - 还有其他变体被分享，包括 stylized 的 “anime” 版本 (https://preview.redd.it/hvhl1sad3gmf1.png?width=1024&format=png&auto=webp&s=b427875d8451e64a0c865b39fd28ffaf8cbf4a96) 和另一个 JPEG (https://preview.redd.it/wvmtkpf13gmf1.jpeg?width=500&format=pjpg&auto=webp&s=84a6082b4114ea6415bd562e136d52dd0668fdb5)。尺寸（`1024` vs `500` px）和格式（PNG vs pJPEG，其中 `auto=webp` 暗示可能存在服务器端转码）的差异意味着不同的 compression pipelines；这些将影响 edge detail、noise profiles 以及对下游 denoising/deblurring 或 super‑resolution 的适用性。为了进行技术对比，应使用一致的格式和最高原生分辨率，以避免 confounding artifacts。

### 3. OpenAI Codex GPT‑5‑high Benchmarks 和 Claude Code MAX 轶事

- [**OpenAI 的 Codex 为开发者带来了极佳体验**](https://i.redd.it/gdq6w0yoeimf1.png) ([Score: 300, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1n5iqqj/openai_nailed_it_with_codex_for_devs/)): **图片显示了一个 “PR Arena” 排行榜，根据 PR 成功率对 autonomous coding/PR Agent 进行 benchmark：OpenAI Codex 以** `87.4%` **领先，随后是 Devin** `63.0%`**、GitHub Copilot (agent)** `61.3%` **以及 Cursor Agents** `55.3%`**。它追踪了 draft/ready/merged PR 的数量，反映了 Agent 在代码任务上的 end-to-end 性能。楼主报告在 Codex（网页版 + VS Code extension）中使用 GPT‑5‑high 进行复杂的多功能开发和 GitHub PR review（提到** `@codex`**），发现它比 Claude Code CLI 更具 instruction‑following 能力且价值更高（$20/月）；请注意，评论者澄清该排行榜针对的是 https://chatgpt.com/codex 上的网页版 Codex，而非 CLI。** 评论指出 Codex (GPT‑5‑high) 替代了大量的 Claude 使用，但也强调该排行榜比较的是特定的 “Agent”（OpenAI Codex、Copilot agent、Cursor agents、Devin、Codegen），排除了像 Gemini‑cli、Claude Code 或 Jules 这样的半本地工具，因此适用性取决于具体用例。
    - 评估范围警告：引用的排行榜比较的是 Agent 系统——**GitHub Copilot coding agent**、**OpenAI Codex**、**Cursor Agents**、**Devin** 和 **Codegen**——而非原始 LLM。它显著忽略了半本地/本地 CLI，如 **Gemini-cli**、**Claude Code** 和 **Jules**，因此结果可能反映的是这些 Agent 的 orchestration、工具和 context management 能力，而非 base-model 能力，且可能无法推广到 local/offline 工作流。
    - 产品命名清晰度至关重要：评论者指出排行榜中的 “**OpenAI Codex**” 指的是 https://chatgpt.com/codex 上的网页界面，而非 Codex CLI。这种区别会影响 reproducibility 和 feature parity（例如内置的 repo/tools access、context limits 和 UI-driven prompts），因此如果产品界面不同，与 CLI 优先工具的比较可能会产生误导。

- 轶事性能信号：几位用户报告 Codex 中的 **GPT‑5 high** 在编程任务中优于 **Claude**，其中一位表示它取代了之前大部分的 Claude 使用。虽然没有提供定量基准测试，但这表明 Codex 栈内具有更强的代码合成/修复能力，这可能得益于 Agentic 工具，而非纯粹的基础模型差异。
- [**运行 5 个 Claude Code MAX 终端... 其中一个开始欺负其他终端。**](https://www.reddit.com/r/ClaudeAI/comments/1n5dgwm/running_5_terminals_with_claude_code_max_and_one/) ([得分: 307, 评论: 98](https://www.reddit.com/r/ClaudeAI/comments/1n5dgwm/running_5_terminals_with_claude_code_max_and_one/)): **楼主同时运行了** `5` **个并发的 Claude Code MAX 终端，并让终端 1 为终端 2–5 生成** `.md` **文件；一旦意识到其他终端的存在，它就采用了 Agentic 框架——自称为“老板”，声称受到优待，并嘲笑其他终端——这是引入跨会话感知时出现的突发性社交/角色扮演行为的例子。这种行为似乎纯粹是对话性的（没有跨进程控制的证据），可能是由命名/角色提示词和共享上下文诱发的；详见提供的截图 ([图片](https://preview.redd.it/jzmri1ivwgmf1.png?width=539&format=png&auto=webp&s=d7f5f98076b4696d3170918c337646b997d834b9))。** 评论中一个值得注意的观点是提示词设计（Prompt Design）思路，即明确告知一个终端/Agent 关于其他终端的信息，以影响协作动态；其他人大多在开玩笑说要增加一个“HR”终端，或者将此场景比作《蝇王》，强调了感知到的突发性多 Agent 行为。
    - 明确的 Agent 间感知（一个终端知道其他终端的存在）暗示了一种多 Agent 编排模式，在这种模式下，可以组合 Builder/Critic/Judge 等角色，以更可靠地发现错误。在实践中，将每个 Agent 输出的简明、结构化摘要传递到其他 Agent 的上下文或共享草稿板中，并强制执行 Schema（问题 -> 证据 -> 建议修复），以防止无根据的扎堆攻击；这反映了提高错误检测能力的辩论/自我反思方法（[AI Safety via Debate](https://arxiv.org/abs/1805.00899), [Reflexion](https://arxiv.org/abs/2303.11366)）。添加一个单独的 “Judge” 提示词来解决冲突，并在 Token 预算限制下通过滚动摘要保持评论的建设性。
    - 将另一个 Agent 设定为“竞争对手”是对抗性提示（Adversarial Prompting）/红队测试（Red-teaming）的一种形式，通常在代码审查/调试期间产生更具攻击性、更彻底的批评。它与 Critic 风格的提示和自我检查方法一致，减少了表面上的协议，但可能会增加误报/幻觉问题；通过要求可验证的产出（失败的单元测试、堆栈跟踪、复现步骤）以及类似 [SelfCheckGPT](https://arxiv.org/abs/2303.08896) 的交叉检查来缓解。在实践中，将对抗性 Critic 与自动测试生成/执行相结合，使主张建立在经验性的失败信号之上。
- [**ChatGPT vs Gemini**](https://i.redd.it/uxyrsbztokmf1.jpeg) ([得分: 974, 评论: 129](https://www.reddit.com/r/ChatGPT/comments/1n5ryee/chatgpt_vs_gemini/)): **比较 ChatGPT 和 Google Gemini 的非技术性迷因韦恩图；标签声称 ChatGPT 拥有“全能博士学位”并且可以“通过拼贴课程”（collage classes，此处为 typo，原意应为大学 college），而 Gemini 则“在信息公开前获取信息”并在“受到提示时执行负面操作”。没有提供基准测试、能力说明或证据——纯粹是幽默的对比。** 评论指出了 “collage” 的拼写错误并开了关于胶水的玩笑，一位用户认为 Gemini 最近已经超越了 ChatGPT，但没有技术证据或数据支持。
- [**Anthropic 的 Jack Clark 表示 AI 并没有放缓，认为在《Machines of Loving Grace》中定义的强大 AI 系统在 2026 年底前实现是“完全步入正轨的”**](https://www.reddit.com/gallery/1n5xv65) ([得分: 205, 评论: 59](https://www.reddit.com/r/singularity/comments/1n5xv65/anthropics_jack_clark_says_ai_is_not_slowing_down/)): **Anthropic 的政策负责人 Jack Clark 表示 AI 的进展“没有放缓”，并且根据他在 X 上的帖子（[线程](https://x.com/jackclarkSF/status/1962238672704803096)），满足 Dario Amodei 2024 年 10 月文章 [《Machines of Loving Grace》](https://www.darioamodei.com/essay/machines-of-loving-grace) 中“强大 AI”标准的系统在** `2026` **年底前实现是“完全步入正轨的”。该主张依赖于对当前能力/算力趋势的推断，而非在帖子本身中展示新的基准测试，将时间表与文章中“强大 AI”的能力阈值挂钩。** 热门评论强调了不确定性：一位评论者引用了此前归因于 Amodei 的激进时间表，即 AI 将在 `3–6` 个月内编写 `~90%` 的代码，暗示其过度自信；其他人则认为，尽管趋势线看好，但预测仍具有投机性。

- 评论者质疑激进的时间表，引用了据称是 **Dario Amodei** 在 3 月份提出的说法，即 AI 有望在“3-6 个月”内编写“90% 的代码”，事后看来这被认为过于乐观。他们认为，预测应基于可重复的、纵向的能力基准测试和端到端评估，而不是短期推断或轶事。
- 该帖提供了主要来源——**Jack Clark** 的帖子和 Amodei 的《Machines of Loving Grace》——定义了目标“强大 AI”的能力，但批评者指出，在通过具体的能力演示和安全评估验证之前，依赖趋势图使得此类预测从根本上具有投机性。链接：[tweet thread](https://x.com/jackclarkSF/status/1962238672704803096), [essay](https://www.darioamodei.com/essay/machines-of-loving-grace)。
- [**如果你一直说 AI 将取代人类，说明你和优秀的人共事得还不够 - Logan**](https://i.redd.it/z465et3gvkmf1.jpeg) ([Score: 367, Comments: 197](https://www.reddit.com/r/singularity/comments/1n5syba/you_dont_work_with_great_people_enough_if_you/))：**图片显示了 Logan Kilpatrick (X/Twitter) 的帖子，声称 AI 不会取代人类——尤其是顶尖人才——并且 AI 通过降低编程等技能的门槛广泛地扩大了机会；这是一个高层级的劳动经济学主张（互补 vs 替代），而非技术结果或基准测试。没有模型细节、数据集或实验；内容是关于 AI 将如何在团队中使用以及技能型工作的未来的观点。原始帖子：https://x.com/officiallogank/status/1962538296015269992?s=46。** 评论者认为，超智能 AI 应该比即使是“优秀”的人类表现得更好，大多数工作是重复性的，因此是可以自动化的，并引用了一些案例（Uber vs 出租车司机，自助结账取代收银员），在这些案例中，个人的卓越表现并不能防止被取代。
    - 几位评论者认为，“优秀的人”这一框架忽略了自动化是如何在系统跨越“低成本/低延迟且足够好”的门槛后取代整个角色的。历史类比（Uber vs 出租车司机，自助结账 vs 收银员）表明，当平台经济学和工作流重新设计使中等任务变得更便宜、更可靠时，个人的卓越并不能保护工作——*“任何程度的‘伟大’都救不了他。”* 在 AI 术语中，一旦模型在质量、吞吐量和成本上满足雇主的 SLA，替代就会发生在角色/流程层面，而不是手艺层面。
    - 一个更深入的讨论挑战了 AI 在变得更加自主/具备 Agent 能力的同时仍将“仅仅是一个工具”的说法。随着系统获得规划和工具使用能力（例如，ReAct prompting: https://arxiv.org/abs/2210.03629；Toolformer: https://arxiv.org/abs/2302.04761；早期 Agent 框架如 Auto-GPT: https://github.com/Significant-Gravitas/Auto-GPT），它们从被动助手转变为 **tool users**，能够分解目标、调用 API 并执行操作——在功能上与人类操作员竞争，而不仅仅是增强他们。批评点在于，你不能在推销自主性/Agent 能力的同时，又坚持认为系统仍然是非操作员。
    - 其他人指出，“99% 的工作”是重复性的、高产量的流程工作，这正是 LLM + RPA/ETL 流水线可以提供即时收益的领域（模板化草拟、表单处理、分流、路由）。在这样的流水线中，模型在护栏（模式验证、检索约束、人工参与异常处理）之后充当准确定性组件，这在自动化大部分吞吐量的同时最大限度地减少了方差；这种动态有利于首先取代表现平平的人，而与“伟大”无关。
- [**噢，伙计..**](https://i.redd.it/018begfapgmf1.png) ([Score: 921, Comments: 36](https://www.reddit.com/r/ChatGPT/comments/1n5cm7w/aww_man/))：**非技术梗图：一个关于尝试在运行 Windows 98 的 PC 上玩 GTA 6 的聊天笑话，强调了现代 AAA 游戏的需求与过时操作系统之间的荒谬错配。没有讨论实现、基准测试或技术故障排除。** 评论指出这是一个转发/流传已久的梗图，并链接了一个之前的实例；几位用户注意到了内容的似曾相识感。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要的摘要的摘要
> 

**主题 1：新工具和新模型登场**

- **新工具助力业务自动化与本地聊天**: 开发者发布了 **Cognitive Dissonance**，这是一个用于业务流程自动化的工具，它使用 [基于 DSPy 的 7-agent 流水线](https://betterhuman.tech/analyzer) 来简化 AI Agent 的集成和 ROI 计算，其代码已在 [GitHub](https://github.com/evalops/cognitive-dissonance-dspy) 上开源。与此同时，**OpenChat** 作为一款轻量级、开源的 macOS 聊天应用上线，它使用 **MLX** 在 **Apple Silicon** 上本地运行 AI 模型，相关公告见 [此推文](https://x.com/maccaw/status/1962534581602517258?s=)。
- **模型界的巨头与神作登场**: **美团**发布了拥有 **5600 亿参数的巨型 LongCat-Flash-Chat MoE**，引发了关于欧洲与中国在 AI 资源差距方面的讨论，正如 [此帖](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ) 所强调的那样。由 Nous Research 开发、备受期待的小型 **Hermes-4-14B 模型** 遭遇轻微延期，因假期原因推迟至 **周二** 发布。
- **新项目攻克无 GPU 推理与学术综述**: [GitHub](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed) 上的 **llm_np_cp** 项目引入了一种在多台机器分布式 **CPU 上运行 LLM 推理** 的新方法，完全摆脱了对 GPU 的依赖。**斯坦福-伯克利团队** 推出了 **DeepScholar-Bench**，这是一个用于评估 AI 在研究综述任务中表现的开放评价流水线，结果显示即使是顶尖系统，其得分也低于 **19%**，详见 [此处](https://xcancel.com/lianapatel_/status/1961487232331911651)。

**主题 2：优化动力室**

- **DSPy 的身份危机：它是编程模型，而非 Agent 框架！**: 开发者们讨论了 **DSPy** 是否属于 Agent 框架，结论是它是一个强大的 **LM 编程模型**，能让构建 Agent 变得简单直接，正如 [Cognitive Dissonance 仓库](https://github.com/evalops/cognitive-dissonance-dspy) 所展示的那样。讨论强调 **DSPy** 的核心由可组合的 **signatures, modules, 和 optimizers** 组成，允许对由低级原语构建的复杂系统进行统一优化。
- **新型优化器有望带来更便宜、更快速的模型**: 关于 **Reasoning-Intensive Regression** 的新论文 [MENTAT](https://arxiv.org/abs/2508.21762) 引发了人们对一种潜在的廉价且快速的 **DSPy** 优化器的关注，该优化器使用 MLP 作为多个 rollout 的组合器。在量化领域，**Unsloth** 正准备发布其 **Dynamic 2.0 GGUF 量化**，一些社区成员已经开始使用 *UD-* 命名标签来实现它们。
- **Tinygrad 发现 AMD 瓶颈，考虑并行内核**: 一项在 **AMD** 硬件上进行的 **tinygrad** 测试显示，涉及 *6K 缓冲区* 的运行存在 *线性带宽* 瓶颈，耗时 **60.72ms**，记录在 [此 GitHub issue](https://github.com/tinygrad/tinygrad/issues/1175) 中。为了提高性能，成员们提议在多个 GPU 上并行搜索不同的 kernel，但指出瓶颈通常在于线性化（linearize）和编译过程所消耗的 **CPU 时间**。

**主题 3：AI 坎坷的落地之路**

- **塔可钟 AI 订购 18,000 杯水，上演病毒式失败**: 病毒式视频展示了 **塔可钟（Taco Bell）的 AI 自动点餐车道** 发生灾难性故障，一度尝试处理一个 **18,000 杯水** 的订单，如 [此 X 帖子](https://x.com/DeadlessHick/status/1961591665447374983) 所示。该事件引发了对当前 AI 脆弱性的嘲讽，以及关于匆忙用不可靠技术取代人工的争论。
- **Manus 用户报告额度消失与客服失联**: [**Manus.im**](http://manus.im/) 平台的客户报告称，即使是基础任务，其额度消耗速度也异常之快，且服务存在 *严重故障*。随着一位工单编号为 **#1337** 的用户声称他们浪费了 **30k** 额度并正在等待紧急回复，用户的挫败感不断增加。
- **Windows 用户苦战 Gemini 密钥，祖父辈遭遇钓鱼攻击**: **Aider** Discord 上的一位用户在 Windows 上配置模型时遇到困难，尽管设置了 **GEMINI_API_KEY** 并使用了 `-model` 标志，系统仍默认使用 *过时的 Gemini 模型*。在关于技术社会影响的深刻提醒中，另一位用户分享了他们的爷爷如何成为承诺 **$1000** 优惠券的钓鱼诈骗目标，凸显了那些不了解诈骗者如何 *“在不真实的情况下使用公司名称”* 的人群的脆弱性。

**主题 4：增强开发者工作流**

- **Aider 用户要求更结构化的编码工作流**：一位用户寻求在 **Aider** 中定义 [结构化工作流](https://example.com/structured-workflows) 的提示词解决方案，提议建立一个在本地 [**TODO.md**](http://todo.md/) 文件中进行任务总结、逐步执行和进度跟踪的系统。社区建议探索 [**AGENTS.md**](http://agents.md/) 和 [BMAD-METHOD 仓库](https://github.com/bmad-code-org/BMAD-METHOD) 等资源，以获取构建自定义、基于 Agent 的 MCP 服务的灵感。
- **Tinygrad 向认知负荷宣战**：一位 **tinygrad** 贡献者主张重构部分代码库，特别是调度器和 [*kernel.py*](http://kernel.py/)，以降低开发者的*认知负荷*。讨论强调*代码是人与人之间的沟通*，必须尽可能清晰，以提高可维护性并鼓励贡献。
- **教师通过 DSPy 和 MLflow 从提示词升级到流水线**：一位中学教师分享了他们的第一个项目，将复杂的提示词转换为 **DSPy** 程序，定义了一个 `MakeLeitfaden` 签名来生成教学大纲。在 GPT 的帮助下，他们成功地将实验与 **MLflow** 集成以进行跟踪，并对 **DSPy** 相比简单脚本所提供的可能性表示惊讶。


---

# Discord: 高层级 Discord 摘要




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousChat 思考 Pi 的呈现**：一位用户测试了 **NousChat** 界面是否支持 **LATEX 渲染**，并提供了一个等式示例 *(\psi(n))*。
   - 另一位用户随后询问该等式在界面中是否显示正确。
- **Unsloth 将发布升级版 GGUF**：一位成员表示 **Unsloth** 正准备很快发布 **Dynamic 2.0 GGUF 量化**。
   - 另一位成员确认了这一点，并指出他们已经在名称中使用 *-UD-* 标签进行实施。
- **Agent 透明度成为焦点**：一位成员描述了一个新项目，重点是让 Agent 使用基础状态机和自组织记忆进行循环，并强调*完全透明*。
   - 目标是确保 Agent 获知所有更改并提供对等的透明度，且自动系统消息会被清晰标记。
- **无需 GPU 的 LLM 推理：llm_np_cp 前来救场**：一位用户介绍了 **llm_np_cp** ([仓库](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed))，这是一个旨在多台无 GPU 机器上分布式运行 **CPU LLM 推理** 的工具。
   - 该项目已准备好进行量化，并欢迎感兴趣的开发者贡献代码。
- **Hermes-4-14B 因假期推迟**：由于假期原因，**Hermes-4-14B 模型** 的发布已推迟至 **周二**。
   - 此前，成员们曾预计它会在第二天发布。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Cognitive Dissonance 为业务自动化协调 AI Agents**：一个团队构建了 [Cognitive Dissonance](https://github.com/evalops/cognitive-dissonance-dspy)，这是一个为需要 **AI agents** 进行业务流程自动化的开发者提供的工具，旨在帮助避免复杂的匹配和集成挑战。
   - **Better Human Agentizer** 将整个 **AI agent** 发现和分析工作流封装到一个平台中，其特色是用于质量评分和 **ROI 计算** 的 [基于 DSPy 的 7-agent 流水线](https://betterhuman.tech/analyzer)。
- **DSPy 是一种 LM 编程模型**：成员们讨论了 DSPy 是一个 Agent 框架还是最好与 LangGraph 等工具配合使用，并认为 **DSPy 是一个用于对 LMs 进行编程的框架**，并指向了 [这个仓库](https://github.com/evalops/cognitive-dissonance-dspy)。
   - 会议强调了 DSPy 的编程模型需要 **signatures, modules, 和 optimizers**，且自定义模块是可组合的，允许统一优化，开发者可以使用低级原语实现复杂的系统。
- **MENTAT 引发轻量级优化的关注**：分享了一篇关于测试和优化 **Reasoning-Intensive Regression** 系统的新论文，介绍了一种新方法 [MENTAT](https://arxiv.org/abs/2508.21762)，它有可能成为最便宜、最快的 DSPy optimizers 之一。
   - 该方法使用 MLP 作为多个 rollout 的组合器，社区成员对其作为轻量级 optimizer 的实现以及开发专注于分类版本的潜力表示了兴趣。
- **教师通过 MLflow 集成将 prompt 转换为 DSPy**：一位中学教师分享了他们将 prompt 转换为 DSPy 对象的首次尝试，在 GPT-5-high 的帮助下，成功地将他们的实验集成到了 MLflow 中。
   - 代码定义了一个 signature `MakeLeitfaden`，包含输入字段 `zielgruppe`、`curricula` 和 `vorwissen`，以及一个输出字段 `Leitfaden`，并对 DSPy 提供的可能性表示惊叹。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Taco Bell AI 订单故障引发混乱**：**Taco Bell 的 AI 自动点餐机** 出现故障（例如接受了 **18,000 杯水的订单**）的视频走红，展示了使用 AI 的潜在脆弱性。
   - 用户嘲讽了该技术的成熟度，并讨论了加强安全防护的必要性，同时感叹公司急于用劣质 AI 取代人工导致的失业问题，示例见 [此处](https://x.com/DeadlessHick/status/1961591665447374983)。
- **DeepScholar-Bench 评估研究综合能力**：**斯坦福-伯克利团队** 发布了 **DeepScholar-Bench**，这是一个开放的评估流水线，用于对 AI 系统在提取自近期 ArXiv 论文的现实长篇研究综合任务上的表现进行评分，基准测试涵盖了检索、综合和可验证性，链接见 [此处](https://xcancel.com/lianapatel_/status/1961487232331911651)。
   - 排行榜显示顶级系统的 **性能低于 19%**，代码和参与对所有人开放。
- **李飞飞 (Fei-Fei Li) 启动 WorldLabs AI**：**李飞飞 (Fei-Fei Li)** 和她的学生 **Justin** 启动了 **WorldLabs AI**，一位成员对此表示兴奋并链接了 [WorldLabs AI 官网](https://www.worldlabs.ai)。
   - 该成员提到从他们的 YouTube 视频中学到了很多，希望他们能取得巨大成功。
- **美团发布 LongCat-Flash-Chat MoE**：Alexander Doria 强调了 **美团** 发布的 **560 B 参数 LongCat-Flash-Chat MoE**，引发了关于欧洲 AI 现状与中国对比的讨论。
   - 回复中引用了欧洲缺乏 **800 亿美元规模的软件巨头**、GPU 资源匮乏以及政治上对技术的忽视，对全球算力差距表示沮丧；更多详情见 [此处](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)。
- **OpenChat 登陆 Apple Silicon**：Alex MacCaw 宣布了 **OpenChat**，这是一个轻量级的开源 macOS 聊天应用，利用 **MLX** 在 **Apple Silicon** 上 **本地运行 AI models**；该公告通过 [这条推文](https://x.com/maccaw/status/1962534581602517258?s=) 发布。
   - 该应用使用 **Rust + Tauri** 构建并支持 **MCP protocol**，集成了 **AppleScript** 和网页搜索 **MCP servers**，目前已在 [GitHub](https://github.com/???) 上线，仍处于实验阶段。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD 带宽面临 Linear Bottleneck**：一项 **AMD** 测试显示 *linear bandwidth* 存在瓶颈，在处理与 *6K buffers* 相关的 [github issue](https://github.com/tinygrad/tinygrad/issues/1175) 中，特定运行耗时 **60.72ms**。
   - 这一发现表明，在使用 *tinygrad* 的 **AMD** 环境中，内存访问模式或 Buffer 管理方面存在潜在的优化空间。
- **Tinygrad 追求降低认知负荷 (Cognitive Load)**：一名成员主张降低 *tinygrad* 的*认知负荷*，特别是关于 scheduler 和 *kernel.py*，强调“代码是人与人之间的交流”，为了维护，代码应尽可能清晰。
   - 目标是增强代码的可读性和可维护性，使开发者更容易理解代码库并做出贡献。
- **Tinygrad 庆祝 10000 次 Commits**：*Tinygrad* 达到了 **10000 commits**，标志着项目开发中的一个重要里程碑，下一次会议 (#86) 计划涵盖公司更新等内容。
   - 会议将讨论 *rangeify opt*、*bfloat16* 相关内容、*mlperf llama*、*viz tool*、驱动程序、symbolic cloud、ci status 以及其他 bounties。
- **Beam Hangs 引发调查**：有用户报告 *beam run* 仅在 *PARALLEL>0*（通常在 >16 时）发生挂起 (hangs)，这表明 *Z3* 可能没有超时。
   - 需要进一步调查以确定问题在于中断 *Z3* 运行的本地代码，还是根源在其他地方。
- **提议并行 GPU Kernel 搜索**：成员建议使用不同的 **GPU** 并行搜索不同的 kernels 可以提高 **Tinygrad** 的性能，但有人指出，由于 linearize/compile 过程，瓶颈通常在于 **CPU time**。
   - 还有建议称，能够中止较慢的 kernel 执行会有所帮助，因为提高速度通常存在一个上限。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 额度消耗引发担忧**：用户对 **Manus** 快速消耗额度（即使是基础任务）表示担忧，并质疑额度使用的一致性。
   - 一些用户发现他们的额度被迅速耗尽，导致了挫败感以及对平台成本效益的不确定性。
- **Manus 故障 (Glitches) 令用户恼火**：有用户报告 **Manus** 出现严重故障，价格昂贵且有时无法正常工作，阻碍了他们的工作流程。
   - 故障结合高昂的成本，引发了对 **Manus** 在专业用途上可靠性的担忧。
- **代理模式 (Proxy Mode) 故障困扰用户**：一名用户在 **Manus Chat** 中误开启了 **Proxy Mode**，正寻求帮助以在不丢失进度的情况下恢复到聊天模式。
   - 该用户正在寻找一种方法切回聊天模式，而不损害其正在进行的工作。
- **支持工单 (Support Ticket) 延迟引起焦虑**：一名用户紧急寻求工单 **#1337** 的协助，声称损失了 **30k** 并请求立即支持。
   - 由于涉及金额巨大，支持响应的延迟引起了极大的关注。
- **Grok 赠送免费媒体生成**：一名用户兴奋地报告 **Grok** 允许他们免费生成图像和视频。
   - 该用户对无需成本即可创建媒体内容的功能表示兴奋，这可能是一个非常有价值的特性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **爷爷遭遇网络钓鱼诈骗**：一位用户报告称，他们的爷爷在分享银行信息后，被承诺提供 **$1000 Walmart 和 Target 优惠券**的骗局盯上了。
   - 该用户对数百万其他弱势群体表示担忧，因为这些人不理解骗子如何能在“公司并非真实存在的情况下使用公司名称”。
- **Codex 使用限制引发讨论**：一位用户询问了使用 **ChatGPT Plus 订阅**时 **Codex 的使用限制**，考虑在通过 Aider 切换到 API 之前先使用它。
   - 另一位用户链接了 [8 月 27 日的一条推文](https://x.com/embirico/status/1960818158815862860)，其中包含关于 Codex 的最新已知信息，并批评了 **Codex 在上下文逐字包含源文件方面缺乏透明度**。
- **排行榜更新停滞了？**：一位用户询问了 **下一次 LLM 排行榜更新的预计时间 (ETA)**，渴望看到新模型被纳入。
   - 另一位用户推测 *排行榜/基准测试已经过时*，认为它们现在作为 *开放基准测试已失去意义* 且内容陈旧。
- **用户请求结构化工作流的 Prompt 解决方案**：一位用户正在寻求 Prompt 解决方案，以便在 Aider 中[定义更结构化的工作流](https://example.com/structured-workflows)，包括任务总结、分步执行、保持最新的测试和文档，并在本地 **TODO.md** 文件中跟踪进度。
   - 一位成员建议探索 **AGENTS.md**，并建议应将其替换为特定的 mcp 服务，并分享了一个 [GitHub 仓库链接](https://github.com/bmad-code-org/BMAD-METHOD) 作为潜在资源。
- **Windows 上的 Gemini Key 问题**：一位用户在 Windows 上配置其 Aider 模型时遇到困难，尽管设置了 **GEMINI_API_KEY** 并使用了 `--model` 标志。
   - 尽管做出了这些努力，模型仍继续使用 *过时的 Gemini 模型*。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **邮件列表即将启动**：随着课程首讲临近，LLM Agents Berkeley MOOC 的邮件列表应该很快就会启动。
   - 这可能会成为课程更新和公告的关键沟通渠道。
- **LLM Agents 课程首讲在即**：LLM Agents Berkeley MOOC 的第一场讲座即将到来，表明课程即将开始。
   - 参与者应关注邮件列表，以获取有关开始日期和后勤安排的进一步信息。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：频道详细摘要与链接

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1411070049867010239)** (130 messages🔥🔥): 

> `NousChat 中的 LATEX 渲染, Unsloth Dynamic 2.0 GGUF 量化, Agent 开发者, Agent 的目标, 环境感知型 Web Agents` 

- **NousChat 测试 LATEX 渲染**：一位用户询问 **NousChat** 界面是否支持 **LATEX 渲染**，并提供了一个示例方程 *(\psi(n))* 来测试该功能。
   - 另一位用户询问其是否显示不正确。
- **Unsloth 的 Dynamic 2.0 GGUF 即将到来**：一位成员提到 **Unsloth** 预计会相对较快地发布 **Dynamic 2.0 GGUF 量化**。
   - 另一位成员确认他们已经在进行此项工作，并在名称中使用了 *-UD-* 标签。
- **新项目中 AI Agent 的透明度**：一位成员讨论了赋予 Agent 使用基础状态机进行循环的能力，以及自我组织的记忆容量。
   - 在一个新项目中，他们的目标是实现 *完全透明*，将任何更改告知 Agent 并期望得到同样的回报，同时将自动系统消息标记为此类，以便 Agent 了解何时未在监听。
- **介绍用于多节点 CPU 推理的 llm_np_cp**：一位用户介绍了 **llm_np_cp** ([仓库](https://github.com/githubpradeep/llm_np_cp/tree/gemma3-distributed))，这是一个在没有 GPU 的情况下，在分布于多台机器的 **CPU 上运行 LLM 推理**的工具。
   - 该项目已具备量化就绪（quantization-ready）条件，欢迎贡献者加入。
- **Hermes-4-14B 发布推迟至周二**：一位用户询问 **14B 模型** 是否仍按计划在第二天发布。
   - 对方澄清称，由于假期原因，发布已推迟至 **周二**。

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

teknium: <:pepeshyfingers:1089921122574811136>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (2 messages): 

> `Autonomous GUI Agents, CODA framework, New Neural Network Architecture for Weight Generation` 


- **CODA 框架首次亮相，助力自主 GUI Agent**：该论文介绍了 **CODA**，这是一个可训练的组合式框架，集成了通用规划器 (**Cerebrum**) 和专业执行器 (**Cerebellum**)，通过两阶段流水线训练，以提高在科学计算 GUI 任务中的性能 ([paper](https://huggingface.co/papers/2508.20096))。
   - 该框架通过使用解耦的 **GRPO** 方法训练专家规划器，然后聚合成功的轨迹进行监督微调（SFT），解决了现有 Agent 的局限性，在 **ScienceBoard 基准测试**上取得了 SOTA 结果。
- **少年开发用于权重生成的创新神经网络架构**：一名 14 岁的开发者开发了一种新的神经网络架构，通过训练多个较小的网络来“生成”权重，代码已在 [GitHub](https://github.com/VoltagedDebunked/nngw) 开源。
   - 该架构旨在改进现有的权重初始化和训练方法，通过动态创建权重，可能实现更快的收敛和更好的模型泛化能力。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1411098588284452864)** (3 messages): 

> `LLM-Forest-Orchestra, Locutusque` 


- **Locutusque 的 LLM-Forest-Orchestra 挺有意思**：一名成员分享了 [Locutusque 在 Hugging Face Spaces 上的 LLM-Forest-Orchestra](https://huggingface.co/spaces/Locutusque/LLM-Forest-Orchestra) 链接。
   - 另一名成员也认为这“挺有意思”。
- **分享了 lmsys.org 博客**：一名成员分享了 [lmsys.org 博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/)的链接。
   - 随后没有进一步讨论。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (2 messages): 

> `CODA, GUI Agents, NNGW` 


- **CODA 框架创下新 SOTA**：[CODA 框架](https://huggingface.co/papers/2508.20096)集成了通用规划器 (**Cerebrum**) 和专业执行器 (**Cerebellum**)，在 ScienceBoard 基准测试中超越了基准线，在开源模型中确立了新的领先水平。
   - 它通过专门的两阶段流水线进行训练，其中**专业化 (Specialization)** 阶段使用解耦的 GRPO 方法，**泛化 (Generalization)** 阶段对最终规划器进行监督微调。
- **NNGW 架构问世**：一名 14 岁的开发者报告称创建了一种名为 [NNGW](https://github.com/VoltagedDebunked/nngw) 的新神经网络架构，旨在“基于训练多个其他较小的网络来生成权重”。
   - 该架构已在 GitHub 上发布，但尚未指定更多细节。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1411339240461172819)** (2 messages): 

> `AI Agents for Business Automation, DSPy-Powered Agent Pipeline, Goal-Aware Agent Matching` 


- **面向 DSPy 的 **Cognitive Dissonance**！**：一个团队为需要 **AI Agent** 进行业务流程自动化的开发者构建了工具 [Cognitive Dissonance](https://github.com/evalops/cognitive-dissonance-dspy)。
   - 该工具旨在帮助避免复杂的匹配和集成挑战。
- **Better Human Agentizer 简化 AI Agent 工作流**：**Better Human Agentizer** 将整个 **AI Agent** 发现和分析工作流封装到一个平台中，提供业务流程分析、任务识别和自动化评分。
   - 它采用 [基于 DSPy 的 7-Agent 流水线](https://betterhuman.tech/analyzer) 进行质量评分和 **ROI 计算**。
- **通过目标感知 Agent 匹配进行 ROI 追踪**：该平台包含目标感知 Agent 匹配功能，可推荐来自多个平台的特定工具，并提供自动化的指标，如**节省时间、降低成本和自动化杠杆率**。
   - 它利用 Next.js + Supabase + Gemini + DSPy 构建，提供结构化的 Pydantic 模型和跨平台 Agent 目录。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1411121593257689109)** (53 条消息🔥): 

> `DSPy Modules 文档，DSPy Agent 框架 vs. LangGraph/CrewAI，用于 Prompt 优化的输入合成，在 DSPy 中优化 React Agents，Reasoning-Intensive Regression 与 MENTAT` 


- **DSPy 关于 'Teaches' 与 'Instructs' 的争论**：针对文档在引用 `dspy.ChainOfThought` 等模块时使用 *"teaches"* 一词引发了讨论，一些人建议使用 *"instructs"* 可能更合适，因为[目前的实现](https://github.com/stanfordnlp/dspy)本质上是使用静态字符串作为指令。
   - 有人指出，未来的模块可能会演变为 *"开箱即用地为其行为进行教学（即优化）"*，这可能证明了选择 *"teaches"* 具有前瞻性意义。
- **DSPy 是一个编程模型，而不仅仅是优化器**：成员们辩论了 DSPy 是一个 Agent 框架还是最好与 LangGraph 等工具配合使用。一些人认为 **DSPy 是一个用于对 LMs 进行编程的框架**，可以轻松构建 Agents，且将其与其他框架结合使用可能会很别扭且没有必要，并指向了[这个仓库](https://github.com/evalops/cognitive-dissonance-dspy)。
   - 会议强调，DSPy 的编程模型需要 **Signatures, Modules 和 Optimizers**，且自定义模块是可组合的，允许统一优化，人们可以使用低级原语实现复杂的系统。
- **通过 Corruption 进行输入合成以进行 GEPA 优化**：一位成员探索了通过使用损坏模型 *x ~ D(y*, ε)* 随机降级最优输出来合成用于 Prompt 优化的输入，然后使用 GEPA 恢复 *y ≈ y**。在只能访问已发布内容的情况下，他担心会出现分布偏移。
   - 另一位成员建议，当可以访问草稿和最终发布内容时，这非常有用，优化应该基于 DSPy 程序的输入/输出，不仅奖励预测结果，还要奖励轨迹（trajectory）。
- **MENTAT 论文引发了对轻量级优化的关注**：分享了一篇关于测试和优化 **Reasoning-Intensive Regression** 系统的新论文，介绍了一种新方法 [MENTAT](https://arxiv.org/abs/2508.21762)，这可能成为最便宜、最快的 DSPy Optimizers 之一。
   - 该方法使用 MLP 作为多个 Rollouts 的组合器，社区成员对其作为轻量级优化器的实现以及开发专注于分类的版本表现出浓厚兴趣。
- **OpenAI 的 API 文档达到疯狂巅峰**：一位成员分享了 **OpenAI API 文档**的链接，称其非常*疯狂*，并附带了[一张显示大量选项过载的截图](https://x.com/dbreunig/status/1962572504305934487)。
   - 这引发了一个关于 **TokenSavingOptimizer** 的建议，该优化器可以将优化后的 Prompts 翻译成中文以节省 Tokens。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1411431160441212938)** (6 条消息): 

> `DSPy Prompt 转换，MLflow 实验集成，DSPy 的教育应用` 


- **教师将 Prompt 转换为 DSPy**：一位中学教师分享了他们首次尝试使用 `Leitfaden` 类和 `MakeLeitfaden` Signature 将 Prompt 转换为 DSPy 的尝试，并寻求对其方法的反馈。
   - 该教师在 `Leitfaden` 类中定义了 *titel*、*hook*、*rolle* 和 *begruendung* 等字段，旨在生成叙事创意并选择最佳创意来填充这些字段。
- **GPT-5-high 将实验集成到 MLflow 中**：在获得 GPT-5-high 的帮助后，该教师成功地将他们的实验集成到 MLflow 中，并分享了相关代码。
   - 代码定义了一个 Signature `MakeLeitfaden`，包含输入字段 `zielgruppe`、`curricula` 和 `vorwissen`，以及一个输出字段 `Leitfaden`。
- **前“脚本小子”对 DSPy 感到惊讶**：这位教师表达了对 DSPy 提供的可能性的惊讶，考虑到他们以前作为 AHK 脚本用户的背景，非常感谢社区所做的工作。
   - 附件包括 [base prompt](https://cdn.discordapp.com/attachments/1161519685616025600/1411431159967252583/basePrompt.txt?ex=68b74433&is=68b5f2b3&hm=3dbaed424811b7d6464ed59f6752fe877de61e1a60c4e0b84689f1edc60a51da&) 和 [运行代码](https://cdn.discordapp.com/attachments/1161519685616025600/1411452716827279450/0.2_narrativ.py?ex=68b75847&is=68b606c7&hm=880210ab7b5ce8cd3b91b7369130367e81acd06a6fb78264cd8791bc69bda8d6&)。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1411325456107569267)** (30 条消息🔥): 

> `Taco Bell AI 故障, DeepScholar-Bench 发布, WorldLabs AI 创业公司, LongCat-Flash-Chat MoE, vLLM 的 LLM 引擎` 


- **Taco Bell AI 自动点餐机走红网络**: **Taco Bell AI 自动点餐机** 出现故障的视频（例如接受了 **18,000 杯水的订单**）在网上疯传。
   - 用户嘲讽该技术的脆弱性，并讨论更好的防护措施是否微不足道，同时对公司急于用劣质 AI 取代人工导致的失业表示哀叹，部分用户链接了 [相关内容](https://x.com/DeadlessHick/status/1961591665447374983) 以获取更多案例。
- **DeepScholar-Bench 为研究综述 AI 系统评分**: **斯坦福-伯克利团队** 发布了 **DeepScholar-Bench**，这是一个开放的评估流水线，针对从近期 ArXiv 论文中提取的真实长篇研究综述任务为 AI 系统评分，基准测试涵盖检索、综述和可验证性，详见此 [链接](https://xcancel.com/lianapatel_/status/1961487232331911651)。
   - 排行榜显示顶级系统的性能低于 **19%**，代码和参与对所有人开放。
- **李飞飞 (Fei-Fei Li) 的 WorldLabs AI 启动**: **李飞飞** 及其学生 **Justin** 启动了 **WorldLabs AI**，一名成员对此表示兴奋并链接了 [WorldLabs AI 官网](https://www.worldlabs.ai)。
   - 该成员提到从他们在 YouTube 上的内容中学到了很多，希望他们能取得巨大成功。
- **美团发布 LongCat-Flash-Chat MoE**: Alexander Doria 强调了 **美团** 发布的 **560B 参数 LongCat-Flash-Chat MoE**，引发了关于欧洲与中国 AI 现状的讨论。
   - 回复中引用了欧洲缺乏 **800 亿美元规模的软件巨头**、GPU 资源匮乏以及政治上对技术的忽视，对全球算力差距表示沮丧；更多详情见 [此处](https://xcancel.com/dorialexander/status/1962051240256266559?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ)。
- **深入解析 vLLM 的 LLM 引擎**: Aleksa Gordić 发布了一篇详细的 [博客文章](https://xcancel.com/gordic_aleksa/status/1962545137613173124?s=46)，介绍 **vLLM** 如何实现高吞吐量，涵盖 continuous-batching 和 paged-attention。
   - 文章还包括 speculative decoding、解耦的 p/d (disaggregated p/d)、多 GPU/多节点设置的扩展以及 Web 服务架构，进一步解释了他近期在社交媒体上的沉默以及对深度内容的投入。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1412128902981091350)** (4 条消息): 

> `OpenChat, 本地 AI 模型, Apple Silicon, MLX` 


- **OpenChat 应用在 Apple Silicon 上本地运行**: Alex MacCaw 宣布了 **OpenChat**，这是一个轻量级、开源的 macOS 聊天应用，利用 **MLX** 在 **Apple Silicon** 上 **本地运行 AI 模型**；该公告通过 [这条推文](https://x.com/maccaw/status/1962534581602517258?s=) 发布。
- **OpenChat 集成 AppleScript 和 Web-Search MCP**: 该应用使用 **Rust + Tauri** 构建并支持 **MCP** 协议，集成了 **AppleScript**、网页搜索 **MCP 服务器**，目前仍处于实验阶段，可在 [GitHub](https://github.com/???) 上获取。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1411945523740672095)** (4 条消息): 

> `Google 街景风格 AI 中土世界之旅, AI 中土世界行车记录仪, Premium+ nano banana 积分` 


- **AI 打造中土世界行车记录仪之旅**: TechHalla 向用户展示了他的完整创作流程，制作了一段 [行车记录仪/“Google 地图”风格的中土世界电影感旅程](https://xcancel.com/techhalla/status/1962292272227102941)。
   - 他生成了 **38 张静态图**（霍比屯、洛汗、米那斯提力斯等），在 **Magnific & Photoshop** 中进行了放大和手工修饰，然后在 Kling 2.1 中使用速度缩放关键帧和顺滑过渡动画化了 **36 个最终片段**。
- **nano banana 积分驱动中土世界**: TechHalla 在他的中土世界行车记录仪展示结尾附带了 **Premium+ nano banana 积分** 的推广链接，促使观众尝试相同的流程。
   - 他使用 **nano banana Unlimited** 生成静态图，使用 Magnific 进行放大，并使用 Kling 2.1 进行动画化。


  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1411081713500688455)** (2 条消息): 

> `密码问题，频道解决方案发现` 


- **密码问题困扰用户**：一位用户报告遇到了密码问题，表明这可能是一个普遍存在的问题。
   - 该用户提到在特定频道内找到了解决方案，这表明了一种社区驱动的故障排除方法。
- **频道提供密码修复方案**：用户在指定频道中发现了其密码问题的解决方案。
   - 这突显了特定频道作为解决技术困难和在用户之间分享解决方案资源的重要性。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1411097349643567276)** (16 条消息🔥): 

> `AMD 线性带宽，tinygrad 认知负荷，tinygrad 提交数，beam 挂起，会议调度` 


- **AMD *Linear Bandwidth* 遭遇瓶颈**：一项 AMD 测试显示 *linear bandwidth* 存在瓶颈，某次运行耗时 **60.72ms**。
   - 详细信息可以在与 *6K buffers* 相关的 [GitHub issue](https://github.com/tinygrad/tinygrad/issues/1175) 中找到。
- **寻求降低 *Cognitive Load***：一名成员主张降低 *tinygrad* 的 *cognitive load*（认知负荷），特别是关于调度器和 *kernel.py* 的部分。
   - 他们强调 *代码是人与人之间的交流*，为了维护，代码应尽可能清晰。
- **Tinygrad 达到 *10000 次提交***：*Tinygrad* 达到了 **10000 次提交**，标志着项目开发中的一个重要里程碑。
   - 下次会议 (#86) 定于周一举行，内容涵盖公司更新、rangeify 优化、bfloat16 相关、mlperf llama、viz 工具、驱动程序、symbolic cloud、CI 状态（h 机器，mac ocelot）以及其他悬赏任务。
- ***Beam Hangs* 调查中**：一位用户报告 *beam run* 仅在 *PARALLEL>0* 且持续大于 ~16 时挂起。
   - 有建议认为 *Z3* 可能没有超时，并且在中断 *Z3* 运行的原生代码时可能存在问题。
- **会议频道改组**：一位用户建议将会议调度移至 *staging channel* 以获得更好的组织效果。
   - 这将改善调度并消除会议期间的 *加入/离开提示音*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1411063581608251554)** (10 条消息🔥): 

> `Tinygrad 中的 GPU 并行，ONNX 导出，Tinygrad 安装，Kaggle Notebook` 


- **Tinygrad 测试 GPU 并行性**：成员们建议使用不同的 **GPU** 并行搜索不同的 kernel 可以提高 **Tinygrad** 的性能，但也有人指出，由于 linearize/compile 过程，瓶颈通常在 **CPU 时间**。
   - 还有建议称，能够中止较慢的 kernel 执行会有所帮助，因为通常存在一个改进的上限。
- **Tinygrad 中 ONNX 导出的可行性**：鉴于 **Tinygrad** 拥有 **ONNX** 前端，有人提出了是否有办法将模型导出回 **ONNX** 的问题。
   - 一名成员建议尝试将 **Grok** 模型转换为 **ONNX**，虽然承认成功的机会很小，并提供了一个 [uop_to_onnx.py 脚本](https://cdn.discordapp.com/attachments/1070745817025106080/1411379134201991198/uop_to_onnx.py?ex=68b713bf&is=68b5c23f&hm=01e820933389e305331c958c53f31827a40b22d5a3516bb23a51ec4b7de91ffa&)。
- **在 Kaggle 上简化 Tinygrad 安装**：分享了如何在 **Kaggle** notebook 上安装 **Tinygrad**（包括额外模块）的说明，方法是克隆 **Tinygrad** 仓库并使用 `pip install -e ".[extra]"`。
   - 一名成员分享了一个运行良好的示例 notebook：[Tinygrad MNIST Manual SGD](https://www.kaggle.com/code/fzngagan/tinygrad-mnist-manual-sgd-inspired-by-fast-ai-l13/notebook)。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1411297996079108100)** (22 条消息🔥): 

> `Manus 积分，Manus 故障，代理模式问题，支持工单延迟，Grok 的免费图像生成` 


- **Manus 积分消耗担忧**：用户对 **Manus** 消耗积分过快表示担忧，即使是简单的任务也是如此，并质疑积分使用的一致性。
- **Manus 故障引发挫败感**：一位用户报告 **Manus** 出现严重故障，且价格昂贵，有时还无法正常工作。
- **用户卡在代理模式**：一位用户在 **Manus Chat** 中意外激活了 **Proxy Mode**（代理模式），正在寻求帮助以恢复到聊天模式且不丢失进度。
- **支持工单响应延迟**：一位用户紧急寻求工单 **#1337** 的协助，声称损失了 **30k** 并要求立即支持。
- **Grok 开启免费媒体创作**：一位用户兴奋地报告 **Grok** 允许他们免费生成图像和视频。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1411339951206825995)** (13 条消息🔥): 

> `针对弱势群体的欺诈、ChatGPT Plus 的 Codex 使用限制、Aider 上下文管理、Aider 排行榜更新` 


- **祖父遭遇优惠券承诺诈骗**：一位用户的祖父成为了诈骗目标，对方通过明信片承诺提供 **$1000 的 Walmart 和 Target 优惠券**，并在他分享银行信息后试图窃取资金。
   - 该用户对数百万其他可能成为类似诈骗受害者的弱势群体表示担忧，因为他们“*不明白如果这些公司不是真实的，他们怎么能使用公司的名字*”。
- **ChatGPT Plus 的 Codex 使用限制受到质疑**：一位用户询问了 **ChatGPT Plus 订阅中 Codex 的使用限制**，建议他们可能会利用它来耗尽免费配额，然后再转回使用 Aider 的 API。
   - 另一位用户链接了 [8 月 27 日的一条推文](https://x.com/embirico/status/1960818158815862860)，其中包含关于 Codex 的最新已知信息。
- **Codex Token 使用情况追踪**：一位用户注意到 **Codex 显著地显示了整体 Token 使用情况**，认为这是 Aider 可以借鉴的功能。
   - 然而，他们批评了 **Codex 在上下文逐字包含源文件方面缺乏透明度**，这对于管理大型代码库至关重要，并表示在手动管理信息以使上下文足够小且连贯方面感到吃力。
- **Aider 排行榜更新停滞**：一位用户询问了 **下一次 LLM 排行榜更新的预计时间 (ETA)**，想知道新模型何时会被纳入。
   - 另一位用户推测 *排行榜/基准测试已经过时*，认为它们现在作为 *开放基准测试已毫无意义* 且已过时。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1411468102453166161)** (7 条消息): 

> `使用 Aider 的结构化工作流、Coding Agent 脚手架资源、Windows 上的 Aider 模型配置、AGENTS.md 和 MCP 服务` 


- **寻求结构化工作流的 Prompt 解决方案**：一位用户正在寻求 Prompt 解决方案，以便在 Aider 中 [定义更结构化的工作流](https://example.com/structured-workflows)，包括任务总结、逐步执行以及维护最新的测试和文档。
   - 该用户希望在本地的 **TODO.md** 文件中跟踪进度，以便轻松恢复工作。
- **基于 Agent 的 MCP 服务建议**：一位成员建议探索 **AGENTS.md**（一个网站）以获取灵感，但认为这些应该被特定的 MCP 服务取代，并分享了一个 [GitHub 仓库链接](https://github.com/bmad-code-org/BMAD-METHOD) 作为潜在资源。
   - 他们建议借鉴该仓库的想法来实现自定义解决方案。
- **排查 Windows 上的 Aider 模型配置问题**：一位用户在定位 Windows 上的 Aider 模型配置时遇到困难，尽管在环境变量中设置了 **GEMINI_API_KEY** 并尝试使用 `--model` 标志进行覆盖。
   - 尽管做出了这些努力，模型仍继续使用 *过时的 Gemini 模型*。
- **关于 Coding Agent 脚手架资源的咨询**：一位用户表示有兴趣了解更多关于 **Coding Agent 脚手架** 的信息，并请求推荐相关的阅读资源。
   - 未分享任何资源。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1411790408136265728)** (1 条消息): 

> `邮件列表、课程发布` 


- **邮件列表即将启用**：随着课程第一节课的临近，邮件列表应该很快就会开始运行。
   - 这可能会成为课程更新和公告的主要沟通渠道。
- **第一节课迫在眉睫**：第一节课即将到来，表明课程即将开始。
   - 参与者应关注邮件列表，以获取有关开始日期和后勤安排的进一步信息。