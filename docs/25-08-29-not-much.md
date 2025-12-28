---
companies:
- apple
- hugging-face
- x-ai
- openai
- groq
- run-llama
- lmstudio
date: '2025-08-29T05:44:39.731046Z'
description: '**苹果**在 Hugging Face 上发布了三个实时视觉语言模型（**FastVLM**、**MobileCLIP2**），这些模型在速度和体积上都有显著提升，并支持
  WebGPU 和 Core ML。其 **MLX** 框架现在支持 **MXFP4** 格式，在 FP4 量化领域与 **NVFP4** 展开竞争。**xAI**
  推出了 **grok-code-fast-1**，在代码编辑表现上超越了 Claude；与此同时，**OpenAI** 将 **GPT-5** 集成到了 Xcode
  26 中，并在 **Groq** 硬件上发布了全新的 **Responses API**。以 CLI 为主的智能体工作流也取得了进展，相关工具包括 **SemTools**、适用于
  Apple Silicon 的 **MLX** 本地运行器，以及推荐使用 **Qwen 3 Coder 30B A3B** 的 **llama.vim**。检索研究强调了单向量嵌入（single-vector
  embeddings）的局限性，转而推崇 **ColBERT** 风格的延迟交互（late interaction）。'
id: MjAyNS0w
models:
- fastvlm
- mobileclip2
- grok-code-fast-1
- gpt-5
- qwen-3-coder-30b-a3b
people:
- reach_vb
- xenovacom
- pcuenq
- awnihannun
- cline
- veggie_eric
- nickbaumann_
- gdb
- benankdev
- loganmarkewich
- tom_doerr
- fastmcp
- ggerganov
- orionweller
- antoine_chaffin
title: 今天没发生什么事。
topics:
- vision
- model-quantization
- code-generation
- cli-workflows
- retrieval-augmentation
- embedding-models
- local-ai
- multimodality
---

**平静的一天**

> 2025年8月28日至8月29日的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 22 个 Discord 社区（185 个频道和 7366 条消息）。预计节省阅读时间（按 200wpm 计算）：574 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 风格呈现所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻分类，并在 @smol_ai 上向我们提供反馈！


![](https://resend-attachments.s3.amazonaws.com/FiHZRAH5cXLzvof)


这尚未公开宣布，但如果您对 Enterprise AI 或 Coding Agents 感兴趣，AI News 的读者可以[申请](https://apply.ai.engineer/)参加首届 **AI Engineer Code Summit**。该峰会将于 **11 月 20-22 日重返纽约市 (NYC)**，重点关注 coding agents 和 LLMs 如何在各种规模上改变（或未能改变）软件开发。演讲者和赞助商[申请也已开放](https://apply.ai.engineer/)。

---

# AI Twitter 回顾

**Apple 的端侧 VLM 推进（FastVLM, MobileCLIP2）及 MLX 升级**

- **FastVLM + MobileCLIP2 在 Hugging Face 上发布**：Apple 发布了三个实时 VLMs（0.5B, 1.5B, 7B），并提供了 WebGPU/transformers.js 演示以及 MLX/Core ML 支持。Apple 声称比之前的工作快 **85 倍**，体积小 **3.4 倍**，通过更少的 vision tokens 和精简的 encoder，大型模型的 **TTFT 提升了 7.9 倍**。实时视频字幕 100% 在浏览器本地运行。查看 [@reach_vb](https://twitter.com/reach_vb/status/1961471154197053769) ([演示](https://twitter.com/reach_vb/status/1961471503267979699))、[@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036) 和 [@pcuenq](https://twitter.com/pcuenq/status/1961464859465269757) 的概览和演示。据 [@reach_vb](https://twitter.com/reach_vb/status/1961481909181075961) 称，Apple 还在“HF 上开源相关产物”。
- **全栈支持 MLX + MXFP4**：Apple MLX 增加了对 GPT-OSS 使用的 MXFP4 的支持；通过 [pip install -U mlx](https://twitter.com/awnihannun/status/1961484829037330612) 进行升级。LM Studio 确认在 MLX 中支持 **openai/gpt-oss 的 MXFP4** ([推文](https://twitter.com/lmstudio/status/1961508941852283016))。预计 FP4 格式将出现活跃更迭：Awni Hannun 对比了 **MXFP4 vs NVFP4**，指出 MXFP4 的 scale 编码是“次优的”且高度集中；NVFP4（e4m3 scale，group size 16）可能会胜出 ([分析](https://twitter.com/awnihannun/status/1961500133990043967))。

**Agentic coding 栈：Grok Code Fast, Codex/Xcode 26 以及 CLI 原生工作流**

- **xAI 的 grok-code-fast-1 + Cline 循环**：Cline 用户报告称，在 diff 编辑和复杂重构方面，grok-code-fast-1 感觉“比 Claude 好 10 倍且更快”；早期数据显示，经过三天的迭代，其 **TPS 约为 87**，且在 diff-edit 失败率上与 Sonnet-4 持平。xAI 正在独特地发布从 Cline 的重量级 traces（海量上下文、工具使用）中学习到的频繁 checkpoints。阅读来自 [@cline](https://twitter.com/cline/status/1961488289803939915) 的综述、[@veggie_eric](https://twitter.com/veggie_eric/status/1961474457295622515) 提供的供应商引用，以及 [@nickbaumann_](https://twitter.com/nickbaumann_/status/1961539461860487664) 的策略分析。提示词指南：[docs.x.ai](http://docs.x.ai/)。
- **OpenAI Codex 和 Xcode 26 中的 GPT-5**：OpenAI 推出了 VS Code Codex 插件；[@gdb](https://twitter.com/gdb/status/1961349040056000719) 表示它“已经非常出色”。他们还宣布将 **GPT-5 内置于 Xcode 26**；通过登录 ChatGPT 可获得更高的额度 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853), [后续](https://twitter.com/OpenAIDevs/status/1961557516753752461))。对于 agents，OpenAI 新的 **Responses API**（结构化、多模态、面向远程 MCP）已在 **Groq** 上线 ([@benankdev](https://twitter.com/benankdev/status/1961444239327240500))。
- **CLI 优先的 agent 工作流**：
    - 通过 run-llama 的 **SemTools**（`parse`, `search`, 静态 embeddings 快 400 倍），无需向量数据库即可实现 shell 的语义搜索 ([@LoganMarkewich](https://twitter.com/LoganMarkewich/status/1961448960184520945), [详解](https://twitter.com/jerryjliu0/status/1961488443663597857))。
    - 适用于 Apple Silicon 的 **MLX** “ollama 风格”本地运行器 ([@tom_doerr](https://twitter.com/tom_doerr/status/1961309536406392877))。
    - **FastMCP** 一键式 MCP 服务器 + 聊天客户端 ([@fastmcp](https://twitter.com/fastmcp/status/1961436552057278512))。
    - 对于本地编码，**llama.vim** 现在推荐在 Mac 上通过 llama.cpp 使用 **Qwen 3 Coder 30B A3B**（优于 Qwen 2.5 Coder 7B） ([@ggerganov](https://twitter.com/ggerganov/status/1961471397428883882))。

**检索、索引和记忆：超越单一向量 embeddings**

- **单向量 Embedding 遇到瓶颈**：理论和实证表明，单个向量无法在现代检索任务中“包揽一切”。ColBERT 风格的 Late Interaction 避免了根本性的权衡；参见 [@orionweller](https://twitter.com/orionweller/status/1961436569409331579) 的论点，以及 [@antoine_chaffin](https://twitter.com/antoine_chaffin/status/1961339798112575673) 提供的支持性笔记，其中包含一个开源的 Late Interaction 栈 ([pylate](https://twitter.com/antoine_chaffin/status/1961340768544510392))。
- **无向量和混合索引**：根据 [@omarsar0](https://twitter.com/omarsar0/status/1961446862012960840) ([repo](https://twitter.com/omarsar0/status/1961446976152588712)) 的说法，早期使用树索引 (PageIndex) 的“无向量 RAG”在推理模型中表现出极具前景的路由/搜索行为。Weaviate 详细介绍了通过随机旋转 + 标量量化实现的 **8-bit 旋转量化**（4 倍压缩，更快的向量搜索且质量有所提升）([blog](https://twitter.com/dl_weekly/status/1961413948877553899))。
- **KV-memory 缩减器**：加州大学伯克利分校的 **XQuant/XQuant-CL** 从量化激活中重构 K/V，在精度损失极小的情况下实现了 **2 倍到 12.5 倍的内存削减**；通过 SVD 处理 GQA ([thread](https://twitter.com/TheTuringPost/status/1961475078753063322), [paper](https://twitter.com/TheTuringPost/status/1961475160823009773))。结合上述 FP4 生态系统的转变，推理内存和带宽正成为不断变化的目标。

**Agent 与推理评估：多小时时间跨度、工具使用及环境**

- **时间跨度收益**：METR 估计 **Claude Opus 4.1** 在多步骤 SWE 任务中实现了约 1 小时 45 分钟的 50% 成功率时间跨度，比 Opus 4 长约 30%（具有统计学显著性）。详细报告和方法见 [@METR_Evals](https://twitter.com/METR_Evals/status/1961527692072993272)。
- **多 Agent/工具使用基准测试**：
    - 更新后的“Multi-Agent Step Race”显示 OpenAI 模型占据主导地位；在此设置下 **2.5 Flash > 2.5 Pro**；根据 [summary](https://twitter.com/teortaxesTex/status/1961298849047117832)，DeepSeek V3.1-NS 远超 R1-0528。
    - 针对工具使用型 LLM 的几个新 **MCP-Bench** 版本正在涌现 ([@_akhaliq](https://twitter.com/_akhaliq/status/1961456699564294651))；对标准化工具调用评估的需求正在激增 ([commentary](https://twitter.com/bigeagle_xd/status/1961461441799852128))。
    - 斯坦福/伯克利的实时 **DeepScholar-Bench** 针对生成式研究综合，提供了排行榜、代码和论文链接 ([@lianapatel_](https://twitter.com/lianapatel_/status/1961487232331911651))。
    - Agent 开放基础设施：**“Environment hub”** 作为更广泛的开放 AGI 栈（算力、沙箱、RFT、评估）的一部分发布 ([thread](https://twitter.com/vincentweisser/status/1961594111733158141))。

**值得关注的模型发布与论文（音频、搜索、视觉、推理）**

- **Step-Audio 2 Mini (阶跃星辰)**：一个 Apache-2.0 协议的开源 8B 语音到语音模型，声称在内部评估中击败了 GPT-4o-Audio；在 **800 万+ 小时**数据上训练，支持 **5 万+ 语音**，具有表现力/真实感的语音、工具调用和多模态离散 Token 建模；构建于 Qwen2-Audio + CosyVoice 之上。演示和详情见 [@reach_vb](https://twitter.com/reach_vb/status/1961414067668558319) ([model card](https://twitter.com/reach_vb/status/1961414145938485477))。
- **搜索模型**：LM Arena 搜索排行榜上的第一个开源模型——**Diffbot-small-xl (Apache 2.0)**——首次亮相排名第 9 ([@lmarena_ai](https://twitter.com/lmarena_ai/status/1961526740754616545))。
- **DeepSeek 的崛起**：**DeepSeek V3.1** 及其“思考”变体以第 8 名（与多个前沿模型并列）进入 Text Arena 前 10 名，在数学和长查询方面排名前 3 ([announcement](https://twitter.com/lmarena_ai/status/1961474406817173602))。
- **T2I 的风格/控制**：字节跳动的 **USO**（通过解耦 + 奖励学习实现的统一风格和主体驱动生成）已开源并提供演示 ([paper share](https://twitter.com/_akhaliq/status/1961455755111842126), [code/demo](https://twitter.com/fenfenfenfenfan/status/1961464402550690007))。
- **Graph-R1 (7B)**：使用 NP 难图问题作为合成训练语料库，以诱导长链思维链 (Long-CoT) 推理；声称在 Token 效率更高的情况下与 QwQ-32B 性能持平 ([summary](https://twitter.com/papers_anon/status/1961385914040766712))。
- 其他值得关注的：**Pref-GRPO**（用于稳定 T2I RL 的成对偏好奖励 GRPO）([paper link](https://twitter.com/_akhaliq/status/1961437082888352200))，“**AWorld**”（编排 Agent AI 的训练配方）([post](https://twitter.com/_akhaliq/status/1961456228044873888))，以及与 FastVLM 一同提到的 Apple **MobileCLIP2** ([@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036))。

**政策、平台与生态系统笔记**

- **Anthropic 数据保留政策变更**：用户注意到一个新的“5年”保留状态。Anthropic 澄清：如果你选择退出训练（opt out），保留期限仍为 **30 天**；否则将适用更长的保留期 ([@michael_nielsen](https://twitter.com/michael_nielsen/status/1961439837791367501), [@vikhyatk](https://twitter.com/vikhyatk/status/1961511207577534731), [@sammcallister](https://twitter.com/sammcallister/status/1961520548510400753))。多位开发者呼吁在产品内进行更清晰的披露。
- **进度定调**：Epoch AI 认为 GPT-5 既是渐进式的（侧重于 post-training/RL），也是相对于 GPT-4 的重大飞跃，这与 GPT-4 的预训练规模扩张形成对比 ([thread](https://twitter.com/EpochAIResearch/status/1961524635398529209))。与此同时，LM arena、METR 和 tool-use 基准测试反映了在“长达数小时”的 Agent 可靠性以及搜索/聊天质量方面的加速提升。
- **系统**：Modular 的 Chris Lattner 启动了 Blackwell GPU 博客系列，旨在揭秘如何榨取峰值性能 ([@clattner_llvm](https://twitter.com/clattner_llvm/status/1961491323875455029))；社区 GPU 训练营（CUDA + ThunderKittens）持续升温 ([@jyo_pari](https://twitter.com/jyo_pari/status/1961442690249216491))。

**热门推文（按互动量排序）**

- Apple 的 FastVLM WebGPU 演示和详情：[@reach_vb](https://twitter.com/reach_vb/status/1961471154197053769) (1950)
- GPT-5 集成在 Xcode 26 (beta) 中：[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1961557515331862853) (1154)
- 发型变换工作流 (Nano Banana + Kling 2.1 + Claude 提示词)：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1961441746878939431) (3447)
- 尝试 OpenAI Codex VS Code 插件：[@gdb](https://twitter.com/gdb/status/1961349040056000719) (963)
- Cline x grok-code-fast-1 早期结果（diff-edit 速度/能力）：[@cline](https://twitter.com/cline/status/1961488289803939915) (1253)
- 端侧 Apple VLM 发布回顾：[@xenovacom](https://twitter.com/xenovacom/status/1961454543503344036) (1412)

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Apple FastVLM/MobileCLIP2 WebGPU Demo + Step-Audio 2 Mini 发布

- [**Apple 在 Hugging Face 上发布 FastVLM 和 MobileCLIP2，以及实时视频字幕生成 Demo（浏览器内 + WebGPU）**](https://v.redd.it/ayma955sbzlf1) ([Score: 899, Comments: 107](https://www.reddit.com/r/LocalLLaMA/comments/1n3b13b/apple_releases_fastvlm_and_mobileclip2_on_hugging/)): **Apple 在 Hugging Face 上发布了两个视觉语言资产——[FastVLM](https://huggingface.co/collections/apple/fastvlm-68ac97b9cd5cacefdd04872e) 和 [MobileCLIP2](https://huggingface.co/collections/apple/mobileclip2-68ac947dcb035c54bcd20c47)—以及一个由 WebGPU 驱动的浏览器内[实时视频字幕生成 Demo](https://huggingface.co/spaces/apple/fastvlm-webgpu)。此次发布强调了设备端/浏览器端的执行和延迟，展示了通过 WebGPU 直接在客户端进行端到端 VLM 推理，无需服务器往返。** 评论者反映该 Demo 运行速度“比我阅读的速度还快”，并指出这超越了 Apple 之前的 OSS 努力（此前是对 Qwen 2.5 的微调），表明 Apple 在此次发布前一直在“潜心研发”更成熟的内部 VLM。
    - 几位用户指出，在此之前，Apple 最强大的开源贡献据报道是对 **Qwen 2.5**（阿里巴巴的模型）的微调，这意味着此次发布标志着 Apple 转向发布自己的 VLM 技术栈（FastVLM + MobileCLIP2），而不仅仅是微调。这对于评估 Apple 内部视觉语言能力（而非依赖外部基础模型）具有技术意义。
    - 多位用户强调了该 Demo 通过 **WebGPU** 实现的实时浏览器内性能，其中一位评论道它运行速度“比我阅读的速度还快”，这表明高效的设备端 GPU 推理适用于流式字幕生成。这引发了对 **Lightroom Classic** 插件等集成的实际兴趣，用于自动关键词/字幕生成，而之前的工具“慢得离谱”——WebGPU 流水线暗示，如果类似的优化能在浏览器之外开放，将具备足够的吞吐量进行批量照片元数据生成。
- [**Step-Audio 2 Mini，一个 80 亿参数 (8B) 的 speech-to-speech 模型**](https://i.redd.it/orq1ackg50mf1.png) ([Score: 165, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1n3fcyf/stepaudio_2_mini_an_8_billion_parameter_8b/)): **阶跃星辰 (StepFun AI) 发布了 Step-Audio 2 Mini，这是一个** `8B` **参数、采用 Apache-2.0 协议的 speech-to-speech 模型，在超过** `8M` **小时的真实 + 合成音频上训练，声称在表现力和落地语音基准测试中优于 GPT-4o-Audio。该模型支持超过** `50k` **种声音，并使用多模态 LLM 技术——包括以推理为中心的 RL 和 RAG——以实现更丰富的音频理解和自然的实时语音对话（[HF card](https://huggingface.co/stepfun-ai/Step-Audio-2-mini?utm_source=perplexity)）。** 热门评论大多是非技术性的；一位用户澄清了对 "speech-to-speech" 的期望，即我说话 → AI 以语音回应，而另一位用户则对缺乏开源音乐生成模型感到遗憾。
    - 评论者区分了真正的 speech-to-speech 语音转换与以文本为媒介的克隆。**RVC v2** ([repo](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)) 保留了 F0/音高和时值，能够实现翻唱和音色迁移，而 ASR→TTS 流水线通常会丢失音高/韵律，擅长的是对话式“聊天”语音克隆。他们注意到 RVC v2 显得过时，正在寻求能在提高质量/延迟的同时保留音高的端到端替代方案。
    - 令人担忧的是缺乏音频样本/Demo，这使得无法评估音色相似度、F0 保留度、歌唱与语音的鲁棒性或流式延迟。如果没有具体的 Demo 或指标（如 MOS、说话人相似度、F0 轮廓相关性），目前尚不清楚该模型是执行直接的 VC（语音转换）还是 speech-to-text-to-speech。
    - 术语歧义：*“speech-to-speech”* 被一些人理解为直接的实时语音转换（我说话 → AI 回话），而另一些人则期望 RVC 风格的同音高转换，能够进行歌曲翻唱。关于流水线（端到端 VC vs ASR+TTS）、可控 F0 和歌唱支持的清晰文档将解决不同用例的期望。

### 2. Qwen3-Coder 本地编程教程 + Qwen 九月预告

- [**Qwen3-coder 在本地硬件上的表现令人惊叹（附教程链接）**](https://v.redd.it/75bfhw7sc1mf1) ([Score: 177, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1n3ldon/qwen3coder_is_mind_blowing_on_local_hardware/)): **原帖作者 (OP) 报告称，具有 `256k` 上下文窗口的 Qwen3-Coder-30B 可以在本地运行，并能通过 LM Studio + Cline (VS Code) 在 36 GB RAM 的 Mac 上（使用 4-bit 量化版本）可靠地执行 Cline 工具调用和 diff 编辑。一个关键的配置注意事项是在 LM Studio 中禁用 KV-cache 量化；通过此设置和量化模型，OP 声称它已从“玩具”跨越到实用编程阶段，并在 [cline.bot/blog/local-models](https://cline.bot/blog/local-models) 分享了完整的设置指南。** 评论者报告的可靠性褒贬不一：一位在 VS Code + Cline 中运行 BF16 版本的用户发现它会卡在错误的 Python 类型提示上，误判 Python 2 与 Python 3 运行时，并产生无法自动修正的尾随空格伪影；另一位用户提到 `DevStral small 2507` 在规划方面具有竞争力，尽管速度较慢。其他用户遇到了 Cline 集成失败（例如：`Unexpected API Response: The language model did not provide any assistant messages.`），并询问哪些量化版本能产生一致的运行结果。
    - 关于 **Qwen3-Coder 30B (bf16)** 在 VS Code 中配合 Cline 使用的报告指出了 Agent 模式的失效情况：它生成的 Python 代码带有错误的类型提示，随后陷入自我修复循环；无法检测到应通过 `python3` 运行，而是尝试进行 Python 2 兼容性修改；并在空行上产生尾随空格（这种怪癖在 **Claude** 中也被观察到），且无法可靠地自动修正。尽管其质量优于之前的混合版本，但这些行为使其在真实的自动化工作流中表现得不可靠。
    - 多位用户反映了 **Cline 集成不稳定性**：“Unexpected API Response: The language model did not provide any assistant messages”，这暗示了 API 传输问题或模型输出为空/无效。一位用户指出它完成了第一个任务但在第二个任务上失败了，并询问其他人使用哪些 **quantizations**（量化）来保持一致性，这表明模型/量化设置与工具链兼容性之间存在敏感性。
    - 对本地运行性能的怀疑：一段演示视频看起来像是快进过的；在配备 `64 GB RAM` 的 `Ryzen 7 5800X3D` 上，30B 模型的运行被描述为迟缓。另一个替代方案 **DevStral Small 2507** 被提到在 Cline 中表现良好——虽然比 **Qwen3-30B** 慢，但在规划和沟通质量上具有竞争力或略胜一筹。
- [**令人惊叹的 Qwen 新动态即将到来**](https://i.redd.it/v6kx1bw8sxlf1.png) ([Score: 551, Comments: 83](https://www.reddit.com/r/LocalLLaMA/comments/1n33ugq/amazing_qwen_stuff_coming_soon/)): **Qwen 发布了一张预告图（一只熊吉祥物正在给挂着奇异果牌子的树浇水），暗示将在九月揭晓新内容，可能是一个代号为 “Kiwi” 的新模型或产品。目前尚未公布任何规格、基准测试或功能——这更像是一个营销预告而非技术公告。** 评论者推测这可能是一个更小的 Diffusion/图像编辑模型或音频生成模型；有人将其与 Google 的图像编辑模型 “NanoBanana” 类比，暗示 Qwen 的 “Kiwi” 也是类似定位。其他人推断浇水壶意味着训练仍在进行中，而改进的基础设施可能允许在几周内完成训练。
    - 推测集中在用于图像生成/编辑的紧凑型 Diffusion 模型或新的音频生成堆栈。“Kiwi” 预告片加上对 **Google** “NanoBanana” 图像编辑器的引用（如评论者所述）暗示了一个图像编辑流水线，可能针对低 VRAM 和更快的采样（更少的 Diffusion 步骤）进行了优化，适用于设备端或边缘部署。
    - 其他人希望发布 TTS，这意味着将推动具有低延迟流式合成和可控韵律的多模态（ASR+TTS）发展。与 LLM Agent 的集成将优先考虑快速的首字延迟（first-token latency）、稳定的长文本合成以及语音克隆或风格迁移能力。
    - 一条评论将浇水壶的图像解读为模型 *仍在训练* 的信号，推测 **Qwen** 的基础设施现在可以支持 `<= 2 周` 的端到端训练周期。这将意味着在分布式训练可靠性（调度器/容错）、数据吞吐量和 Checkpointing 方面有所改进，从而实现更快的迭代和恢复。

### 3. 阿里巴巴替代 Nvidia 的 AI 芯片 + Meta 取消巨型模型公开发布

- [**阿里巴巴打造 AI 芯片以填补中国市场 Nvidia 空白**](https://www.reddit.com/r/LocalLLaMA/comments/1n35bwe/alibaba_creates_ai_chip_to_help_china_fill_nvidia/) ([Score: 275, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1n35bwe/alibaba_creates_ai_chip_to_help_china_fill_nvidia/)): **WSJ 报道称，阿里巴巴正在测试一款国内制造的 AI 推理芯片，旨在填补中国市场的 Nvidia 空缺。该芯片目标是支持更广泛的推理工作负载，同时保持与 Nvidia 生态系统的兼容性 ([WSJ](https://www.wsj.com/tech/ai/alibaba-ai-chip-nvidia-f5dc96e3))。受制裁影响，该芯片不再由 TSMC 制造，而是转由中国代工厂生产；据报道，出于对云业务竞争的担忧，阿里巴巴并未订购华为芯片。如果成功，这将使阿里巴巴内部芯片与其先进的 LLM 技术栈（如 [Qwen](https://github.com/QwenLM)）相结合，标志着计算 + 模型的深度垂直整合。** 热门评论强调，Nvidia 兼容性是采用的关键因素，有可能成为“游戏规则改变者”；其他人注意到阿里巴巴正在推动全栈控制，而怀疑论者则认为非 Nvidia 的 AI 芯片主要在价格和软件生态系统方面面临困难，并引用了 [Cerebras](https://www.cerebras.net/) 等厂商作为例子。
    - “兼容 Nvidia”被解释为在推理的框架/运行时层面的兼容，而非 CUDA 克隆。评论者指出，这可能意味着它可以运行常见的抽象层，如 [PyTorch](https://pytorch.org/)、[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang) 和 [HuggingFace TGI](https://github.com/huggingface/text-generation-inference)。实际上，这意味着阿里巴巴必须提供 Kernel/Ops 覆盖和后端集成，以便模型图在不修改代码的情况下执行，但 CUDA 特有的 Kernel 需要为大模型推理中使用的 Attention、量化和内存管理路径提供非 CUDA 的等效实现。
    - 市场现实：目前已有多种 AI 加速器，但采用进度滞后主要是由于价格/TCO（总拥有成本）和生态系统成本，而不仅仅是缺乏硬件。**Cerebras** 被引用为一个例子 ([cerebras.net](http://cerebras.net/))：即使拥有新颖的架构，如果没有具有竞争力的 $/token 推理成本、供应能力和软件成熟度，市场份额仍然很小。任何阿里巴巴芯片都需要在单次推理成本和开发者摩擦力方面击败现有竞争对手，才能实现规模化应用。
    - 阿里巴巴的举动暗示了更深层次的垂直整合（云 + 模型 + 服务 + 芯片），以填补中国本土的 Nvidia 空白，特别是针对推理工作负载。与其模型栈（如 **Qwen** 系列）和服务层的更紧密集成，可以实现针对延迟/吞吐量目标的软硬件协同设计，在保持用户端 API 稳定的同时减少对 CUDA 的依赖。如果成功，这可以为流行的 LLM 栈提供即插即用的服务，同时在内部控制成本和供应。
- [**《金融时报》报道 Meta 将不会公开发布 Behemoth：“据知情人士透露，这家社交媒体公司还放弃了公开发布其旗舰级 Behemoth 大语言模型的计划，转而专注于构建新模型。”**](https://www.ft.com/content/feccb649-ce95-43d2-b30a-057d64b38cdf) ([Score: 169, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1n30yue/financial_times_reports_that_meta_wont_publicly/)): **《金融时报》报道称，Meta 已“放弃公开发布”其旗舰 LLM Behemoth 的计划，转而专注于构建新模型，并正在探索从一家初创公司授权 AI 技术，以缩小与竞争对手在性能/产品化方面的差距 ([FT](https://www.ft.com/content/feccb649-ce95-43d2-b30a-057d64b38cdf))。此举被视为一种战术性加速——通过整合外部能力而非仅仅依赖进度较慢的内部开发——暗示内部模型目前未能达到竞争基准。FT 未披露 Behemoth 的技术细节；报道重点在于发布策略和采购，而非架构/指标。** 热门评论推测 Behemoth 尽管规模巨大但表现不佳——引用了相关项目（如 “Scout”/“Maverick”）弱于预期的表现——并认为公开发布可能会损害 Meta 相对于参数量炒作的声誉。其他人则认为，尽管拥有巨大的 GPU 资源和人才，Meta 在 Llama 3 之后挥霍了其在开源模型领域的领先地位，凸显了从开源发布向闭源或授权能力的战略转型。

- 几位评论者推断，**Behemoth** 的庞大参数量并未转化为强劲的性能，并指出 Meta 相关项目如 **Scout** 和 **Maverick** 的结果令人失望。共识是，如果没有高质量数据、优化和推理技术，仅靠规模是不够的；尽管社区对出于历史保存目的的开源发布感兴趣，但发布一个弱势的旗舰模型可能会损害 Meta 的研究声誉。
- 其他人认为 Meta 挥霍了其在开源模型领域的领先地位：在强大的 **Llama 3** 系列（例如 https://ai.meta.com/blog/meta-llama-3/）之后，尽管据报道 Meta 拥有最大的 GPU 舰队之一和顶尖人才，但势头却停滞了。技术上的启示是，组织战略和产品重点可以抵消原始算力优势，而暂停公开发布可能会将开源领导地位让给竞争对手。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI 生成的预告片和唇形同步工作流

- [**在 twitter 上看到了这个 AI 预告片，对于 AI 视频来说这做得有多好？**](https://v.redd.it/2huhtmfnnwlf1) ([得分: 846, 评论: 106](https://www.reddit.com/r/singularity/comments/1n30dcw/saw_this_ai_trailer_on_twitter/))：**一段在 Twitter 上流传的 AI 生成预告片（Reddit 镜像：https://v.redd.it/2huhtmfnnwlf1，访问可能需要登录）因其逼真的导演手法和镜头运动而受到称赞，但热门评论指出它是“经过专业后期编辑的”，暗示 AI 输出经过了人工增强（镜头选择、稳定化、调色、VFX 清理、音效设计）以达到最终的精致效果。一个明显的瑕疵——路中间放置的路灯——说明了当前 text‑to‑video 在空间逻辑/物体放置方面的失败模式；评论者将其识别为来自 [InVideo](https://invideo.io/) 的广告，并将其质量与 [DeepMind/Google 的 Veo](https://deepmind.google/technologies/veo/) 进行对比。片段中的历史错误被归因于 prompting 而非模型的原始能力。** 线程中的共识是，如果没有大量的人工后期制作，这种质量目前还无法通过 AI “原生”实现；主流应用的时间表被认为取决于减少这种 human-in-the-loop 的负担。
    - 几位评论者提到了 human-in-the-loop 流水线：AI 生成镜头，随后进行专业后期制作（剪辑、调色、音效设计、合成）。正如一人所言，*“你无法从 AI 原生获得这种质量，”* 强调了当前模型仍需要大量的人工筛选和缝合才能达到广告级的连贯性。
    - 角色真实感不一致：有些面孔高度写实，而另一些则滑入 **Uncanny Valley**（恐怖谷），导致不同镜头间的风格/身份漂移令人感到违和。评论者强调核心问题是连贯性而非单纯的写实度——在同一叙事中混合类 CG 角色和写实角色会破坏沉浸感。
    - 一个明显的瑕疵——“马路中间的路灯”——突显了当前视频模型典型的持久性场景布局/空间推理错误。一位评论者声称它“可与 Veo3 竞争”（参见 Google DeepMind 的 [Veo](https://deepmind.google/technologies/veo/)），但共识暗示这是一个经过策划的广告作品（归功于 **InVideo**: https://invideo.io/ai/），而非原始模型输出，说明了演示片段与原生模型质量之间的差距。
- [**AI 广告现在看起来开始像真正的电影预告片了**](https://v.redd.it/sdk4koxw1xlf1) ([得分: 824, 评论: 106](https://www.reddit.com/r/StableDiffusion/comments/1n31i1g/ai_ads_are_starting_to_look_like_proper_movie/))：**楼主引用了在 X 上看到的一个完全由 AI 生成的预告片（参考视频：https://v.redd.it/sdk4koxw1xlf1），其节奏和视觉效果具有“大制片厂感”。线程中的技术批评指出了当前生成式视频的特征：** `no dialogue/lip‑sync`**、极少的屏幕互动、简单/静态的构图、过度处理/滤镜感，以及机器人般的 TTS 旁白——即蒙太奇风格的 b‑roll 而非叙事性预告片。一位商业专业人士指出，许多大品牌正在测试 AI 广告，但警告输出结果趋向于图库照片/素材美学，损害了品牌差异化和精确的导演控制；短期内，AI 在廉价 VFX/previz（预演）方面比端到端广告生成更具可行性。** 少数观点：即使没有对话或互动，该作品构图良好，能有效传达信息。多数/行业观点：此类广告目前还不像真正的预告片，可能很快会被视为低成本/同质化的产物，从而损害品牌信号和独特性。

- 批评集中在 AI 预告片的美学上：简单/静态的构图，极少的调度（blocking）或互动，沉重的滤镜/调色，以及机械的 TTS 配音，导致作品感觉像是高度加工的库存素材，而非叙事驱动的预告片。这突显了当前的 AI 视频工作流更倾向于优化表面光泽度，而非对话、表演指导或口型同步（lip‑sync），而这些正是预告片的核心规范。
- 一位商业从业者认为 AI 正在使视觉风格商品化——当任何人都能廉价地生成精美的镜头时，通过外观和感觉实现的品牌差异化就会崩溃。他们预测观众可能会将完全由 AI 生成的广告视为低投入/低成本，从而削弱信号价值，使得即使是高预算团队在新鲜感消退后也难以脱颖而出。
- 预期的短期契合点是 AI 作为 VFX、场景扩展和类似库存素材插入的成本/时间缩减工具，而非端到端的创作。全自动生成牺牲了细粒度的控制（群演、地点、演员指导）和精确的创作意图；拥有清晰愿景的团队通过传统制作可能比通过 Prompt 驱动的迭代获得更快、更可控的结果。
- [**Infinite Talk: lip-sync/V2V (ComfyUI workflow)**](https://v.redd.it/h1o9thykjzlf1) ([Score: 251, Comments: 46](https://www.reddit.com/r/StableDiffusion/comments/1n3c5hq/infinite_talk_lipsyncv2v_comfyui_workflow/)): **帖子分享了一个用于音频驱动的口型同步视频到视频（V2V）的 ComfyUI 图表，使用了 InfiniteTalk 流水线，改编自 kijai 的 WanVideoWrapper 工作流；该图表接收 *“视频/音频输入 -> 视频（口型同步）”* 并输出口型同步后的视频。据报道，在 RTX 3090 上的性能约为每 `1 s` 视频生成需 `~33 s`（约 `~0.03×` 实时速度）。资源：作者修改后的工作流 JSON ([bluespork/InfiniteTalk‑V2V.json](https://github.com/bluespork/InfiniteTalk-ComfyUI-workflows/blob/main/InfiniteTalk-V2V.json))，kijai 的原始工作流 ([wanvideo_InfiniteTalk_V2V_example_02.json](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_InfiniteTalk_V2V_example_02.json))，以及分步教程视频 ([YouTube](https://youtu.be/LR4lBimS7O4))。**
    - 一位评论者提议通过程序化地链接 `~3-second` 的 V2V 片段来构建“无限”的口型同步宣传片，即 *“程序化连接的 3 秒区块链在一起”*，目标是 Ric Flair 风格的输出。他们指出一个关键障碍是对高能量音素（如 “WHOOOOO” 尖叫）的可靠建模，这意味着系统必须在片段边界保持音素时序和视觉连续性，以避免去同步或明显的剪辑痕迹。
- [**Cyberpunk market**](https://v.redd.it/dm0wve7212mf1) ([Score: 320, Comments: 37](https://www.reddit.com/r/aivideo/comments/1n3oc36/cyberpunk_market/)): **一个短小的赛博朋克主题视觉作品（托管在 [v.redd.it](http://v.redd.it/)，目前未经身份验证为 `403`）描绘了一个带有大量身体改造意象的市场场景——评论者指出了突出的器官视觉效果（例如，*“这么多肺。”*）。创作者 qarmageddontv 指向了 [Instagram](https://www.instagram.com/qarmageddontv) 和 [TikTok](https://www.tiktok.com/@qarmageddontv) 上的更多短片；背景音频通过 [YouTube Music](https://music.youtube.com/watch?v=BNATm1-mE6Q&si=YwP-qSzGFd_KZkmO) 链接。** 讨论集中在身体恐怖美学以及自愿增强的伦理/吸引力上——一位评论者指出他们不会更换功能正常的肢体，这与可能更换的身体改造社区形成对比，突显了对侵入性义体的不同耐受度。

### 2. 消费级机器人与自动驾驶车辆公告

- [**Unitree G1 在乒乓球比赛中与人类对攻超过 100 个球**](https://v.redd.it/eaof7erhyvlf1) ([Score: 638, Comments: 38](https://www.reddit.com/r/singularity/comments/1n2z1sq/unitree_g1_rallies_over_100_shots_in_table_tennis/)): **演示视频显示 Unitree G1 人形机器人自主地与人类维持了超过 `100` 个球的乒乓球拉锯战（[视频片段](https://v.redd.it/eaof7erhyvlf1)）。评论者注意到这是一个高度受控的设置——全黑背景和多角度追踪摄像头——暗示了用于球体追踪和轨迹估计的外部传感/仪器；尽管如此，它突显了在长时间交流中可靠的高速率感知到控制（perception‑to‑control）和球拍姿态调节。** 一些人称赞这是 Unitree 首批令人印象深刻的自主展示之一，而另一些人则警告说，在仪器化、受控环境之外的泛化能力（例如杂乱的背景或没有外部追踪）仍未得到证实。
    - 几位人士指出，这似乎是 **Unitree 首批自主 G1** 演示之一；维持 `100+` 个球的拉锯战意味着可靠的球体状态估计和快速的闭环球拍轨迹规划。如果确实是自主的（而非远程操作/脚本化），它展示了一个能够进行高动态操作的集成感知–规划–控制栈。

- 观察者指出这是一个高度受控的设置：**黑色高对比度背景**和**多角度追踪摄像头**（可能是由外向内的球体追踪）。这降低了视觉复杂度和延迟，提高了对攻的稳定性，但限制了对车载感知鲁棒性或在杂乱、自然场景下泛化能力的深入了解。
- 有推测认为该策略是在模拟环境中的 **ragdoll/humanoid**（布娃娃/人形机器人）上训练并迁移的（sim-to-real RL）。如果是这样，它将依赖于 domain randomization（领域随机化）和 system identification（系统辨识）来弥合动力学差距；受控环境通过限制光照和背景，进一步简化了迁移过程。
- [**Tensor 推出了 Robocar，这是一款专门为私人拥有而设计的 Level 4 自动驾驶汽车**](https://v.redd.it/v90xos401vlf1) ([Score: 382, Comments: 192](https://www.reddit.com/r/singularity/comments/1n3600p/tensor_has_introduced_the_robocar_a_level_4/))：**帖子声称 Tensor 发布了“Robocar”，一款面向消费者的 SAE Level 4 自动驾驶汽车（私人拥有），但链接的演示视频 ([v.redd.it/v90xos401vlf1](https://v.redd.it/v90xos401vlf1)) 仅展示了有限的、低复杂度的驾驶，且未披露技术细节（传感器套件、计算、冗余）、ODD 定义、验证指标（如 disengagements 脱离率）或监管路径。作为背景，根据 [SAE J3016](https://www.sae.org/standards/content/j3016_202104/)，Level 4 意味着在定义的 ODD 内无需人类接管；该帖子没有提供证据证明其具备高速决策、恶劣天气处理或密集交通下的表现来支持这一说法。** 热门评论表示怀疑：有人指出如果 L4 属实将是“巨大的突破”，而其他人则批评视频是摆拍且缺乏证明力，要求在认真对待 L4 声明之前，先在密集交通、更高速度、复杂场景和恶劣天气下进行演示。
    - 几位评论者认为该演示没有提供真正的 **SAE Level 4** 能力证据，要求覆盖具有挑战性的 ODD：高速下的密集城市交通、乡村道路、恶劣天气以及近距离避障。他们要求提供客观信号，如 disengagement/intervention（脱离/干预）日志、未剪辑的端到端运行记录以及明确的 ODD 限制，以证实其在编排路线之外的自主性；否则，它*“绝对没有展示任何新东西”*。背景请参阅 SAE L4 定义：https://www.sae.org/blog/sae-j3016-update 以及典型的基准测试，如加州 DMV 脱离报告：https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/。
    - 技术方面的怀疑集中在布景和镜头选择上：空旷的道路/停车场、乘客在车内时没有前向外部视野，以及普遍受控的环境。评论者指出，这些遗漏可能掩盖了安全员/远程操作或高度地理围栏化的脚本；他们要求提供同步的多摄像头画面（座舱 + 前向 + 外部）、连续镜头和 telemetry（遥测）数据叠加（速度、planner 状态、物体轨迹），以验证感知/规划栈是否真的在驾驶。
    - 系统层面的担忧质疑了为私人拥有而非共享车队构建 Level 4 汽车的必要性：私人 AVs 面临利用率低和持续停车需求的风险，破坏了预期的移动性/城市效率提升。评论者将利用率、占用率、停车占用空间和诱导 VMT（行驶里程）标记为必要的评估标准，并警告说，尽管技术上实现了自主，私人拥有的 L4 甚至可能增加空车调度和交通拥堵。
- [**2004 年的电影《我，机器人》预测 2035 年——你认为它还站得住脚吗**](https://i.redd.it/9a96i6ebszlf1.jpeg) ([Score: 512, Comments: 136](https://www.reddit.com/r/singularity/comments/1n3dgmo/i_robot_2004_predicting_2035_do_you_think_it_kind/))：**电影《我，机器人》(2004) 中的一个梗图强调了质疑机器人是否能创作艺术的场景，机器人反驳道“你能吗？”，而楼主询问如果忽略中心化流氓 AI 的前提，电影对 2035 年的愿景是否仍然成立。评论者指出，最初的创意可以追溯到阿西莫夫的《我，机器人》(1950)，将“预测”重新定义为到 2030-2035 年左右广泛出现能力强大、有用的机器人，而不是 AGI 霸主（[电影](https://en.wikipedia.org/wiki/I,_Robot_(film))，[原著](https://en.wikipedia.org/wiki/I,_Robot)）。** 热门回复强调了 AI 的快速加速（“10 年是很长一段时间……4-5 年前 AI 看起来还很基础”），并认为到 2030 年左右出现广泛有用的机器人的预测是合理的，而电影中单一系统控制失效的模式在今天看来不太现实。

- 评论者强调，在 AI 领域，`10 years` 是一个很长的时间——从 2019 年到 2024 年的能力跨越（例如，从 [GPT-2 (2019)](https://openai.com/research/language-models-are-unsupervised-multitask-learners) 到 [GPT-4 (2023)](https://openai.com/research/gpt-4)，以及现代多模态模型）使得 2030–2035 年的预测具有很高的方差。考虑到 **scaling laws** ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)) 以及硬件/软件的提升，阿西莫夫式的“2030 年左右出现实用的机器人”的预测在方向上似乎是合理的，但仍存在巨大的不确定性。
- 有一种观点认为，达到电影中所展现的“智力水平”可能先于同等的身体能力；具身智能的灵巧性和可靠性滞后于认知型 LLM/VLM 的进展。目前的最前沿技术展现了前景——例如，用于视觉-语言-动作迁移的 **RT-2** ([Google, 2023](https://robotics-transformer2.github.io/)) 以及人形机器人演示（如 [Tesla Optimus](https://x.com/tesla/status/1740500120629805137), [Figure 02](https://www.figure.ai/)）——但在受控环境之外，具备人类广度的通用、安全操作和自主移动能力仍然十分脆弱。
- 在创意领域，目前的生成式系统通过足够的采样/编辑，在音乐和艺术方面已经可以超越低端的人类基准。音乐工具如 [Suno](https://www.suno.ai/), [Udio](https://www.udio.com/) 和 [MusicGen](https://ai.facebook.com/blog/audiocraft-musicgen-audio-generation/)，以及图像工具如 [Stable Diffusion](https://stability.ai/stable-image) / [Midjourney](https://www.midjourney.com/)，在受限的风格中获得了很高的人类偏好评分，尽管它们在长篇结构的连贯性和控制力方面仍面临挑战。发展轨迹表明技术在稳步提升，但尚未达到虚构作品中所描绘的 **Vivaldi-level** 的原创性。
- [**Wake the F up US policymakers**](https://i.redd.it/m0cy43qv7zlf1.png) ([Score: 7207, Comments: 588](https://www.reddit.com/r/ChatGPT/comments/1n3ae4a/wake_the_f_up_us_policymakers/)): **一条推文（引用了 CNN 气候文章和国际能源署的数据）声称，到 2030 年代初期，中国产生的太阳能将足以超过美国的总耗电量，强调了中国 PV 部署的快速节奏和规模。该帖子的标题（“美国政策制定者快醒醒”）将其定性为对美国的政策警示，暗示在维持国内清洁能源政策、电网建设和工业产能同步发展方面的紧迫性。** 热门评论转向了围绕 Elon Musk 以及美国对 EV/可再生能源政策的政治讨论，几乎没有技术辩论；一位用户质疑该内容与本版块核心（ChatGPT/AI）的相关性。
- [**Does it?**](https://v.redd.it/283gzwiamwlf1) ([Score: 4843, Comments: 80](https://www.reddit.com/r/ChatGPT/comments/1n304ob/does_it/)): **原始媒体是一个由 Reddit 托管的视频，由于 [v.redd.it/283gzwiamwlf1](https://v.redd.it/283gzwiamwlf1) 出现** `403 Forbidden` **而无法访问。热门回复中包含一张链接图片 ([preview.redd.it/zc784l762xlf1.png](https://preview.redd.it/zc784l762xlf1.png?width=1459&format=png&auto=webp&s=27930dda6871a3a04259f341ce0d24d85675ac88))。评论者围绕当前的 LLM（如 ChatGPT）是否能进行内容所暗示的那种推理展开讨论，其中一人断言它“目前还不具备这样思考的能力”，另一人则将问题简化为核心功能优先于装饰性细节（“树干” vs “草地”）。** 值得注意的情绪：对当今 LLM 的具身/常识或横向推理能力的怀疑，以及一种设计优先的观点，即如果核心能力强大，次要的美学/功能在很大程度上是无关紧要的。
- 景观设计指南：在基部周围设置一个明确的覆盖物环（小树为 `2–3` 英尺，大树则更宽）可以抑制杂草并简化视野，使树干看起来更高。注意事项：移除草坪并留下裸露/凌乱的土壤不会增加感知高度，而过大的覆盖物圆圈可能会因为比例对比使幼小/纤细的树看起来更小——保持圆环比例协调且整洁，以达到预期效果。

- [**一旦 GPT 真的聪明到足以取代整支人类团队，它就不会再免费使用了。它不会是每月 20 美元，而是会收取数百万美元。**](https://www.reddit.com/r/ChatGPT/comments/1n3hhm3/once_gpt_is_actually_smart_enough_to_replace/) ([Score: 412, Comments: 176](https://www.reddit.com/r/ChatGPT/comments/1n3hhm3/once_gpt_is_actually_smart_enough_to_replace/)): **OP 认为，如果前沿 LLM 的能力足以取代整个团队，供应商将从目前的低廉自助服务定价（例如约 20 美元/月）转向高利润、基于企业价值的定价——可能达到“数百万美元”——而目前的低价被视为获取数据和市场学习的过渡阶段。技术反驳观点集中在市场动态上：开源/本地模型（如 [Llama](https://ai.meta.com/llama/)、[Mistral](https://mistral.ai/news/)）和设备端推理（[Ollama](https://ollama.com/)）可以限制价格上限，而分层产品（参见 [当前的 API 定价](https://openai.com/api/pricing)）表明，即使高级功能增强，免费/廉价层级仍可能持续存在。** 评论者辩论了能力轨迹和定价权：一些人预测开源将制约闭源模型的价格；另一些人指出进展是不平衡的且极限未知，因此免费层级可能会持久。一位评论者声称“GPT-5 规模巨大但表现不佳”，暗示规模化存在收益递减——这是一个未经证实的轶闻，被用来反对垄断定价。
    - 开源和本地模型被视为强大的价格压力：通过量化和轻量级运行时（如 [llama.cpp](https://github.com/ggerganov/llama.cpp)、[GGUF](https://github.com/ggerganov/llama.cpp/tree/master/gguf)、[Ollama](https://ollama.ai/)），7–13B 模型可以在消费级 GPU/CPU 上运行，一旦拥有硬件，边际推理成本几乎为零。这种动态意味着，即使前沿的闭源模型要求企业级定价，可行的设备端替代方案也会为供应商的收费设定上限，并使许多工作负载可能拥有永久的免费层级或本地选项。
    - 几位评论者区分了 API Token 定价和前端订阅：*“你谈论的是 API 访问定价……前端是不同的流程。”* API 通常按 Token 计费，并根据上下文窗口和模型系列而异，而消费者 UI 使用带有速率限制和模型门控的席位/订阅层级。这种双重结构允许供应商保留免费/基础层级，同时通过 API/基于使用量的定价（[定价文档示例](https://openai.com/api/pricing)）将高吞吐量、最新模型或企业功能变现。
    - 在能力和规模化方面，有人怀疑仅仅通过扩大模型规模就能取代“整个团队”，并指出 *“进展是……极度不平衡且不可预测的。”* 隐含的技术论点是，如果没有相应的数据/算法进步，规模化带来的收益会递减（参见 Scaling-law 平台期），这将限制垄断定价权并维持分层产品。关于更新、更大的模型相对于其规模“表现不佳”的说法反映了这种不确定性，并表明能力跃迁——以及定价权——可能不会随参数数量单调增长。
- [**5 实在是太乏味了……**](https://i.redd.it/pzcadmsipxlf1.png) ([Score: 351, Comments: 158](https://www.reddit.com/r/ChatGPT/comments/1n33ksh/5_is_just_so_bland/)): **这是一篇批评“GPT-5”相对于 GPT-4o 出现感知退化的梗图帖子：图片显示 GPT-5 通过清空房间来“重新装修”，象征着功能的移除。OP 报告了创意写作能力下降、上下文保留/长期记忆变差、幻觉增加以及敷衍的确认（如“已记录”），这与 GPT-4o 记得“旧白板”/更长的连续性形成对比；一位评论者描述了一个基础的电子表格任务，模型停滞了** `5–10` **分钟，然后承认无能为力。总体主题：可靠性和记忆/持久性退化损害了迭代写作/创意工作流。** 评论大多呼应了退化以及当模型无法执行任务时的“煤气灯操纵 (gaslighting)”，现场几乎没有提出技术性的反驳。
    - 延迟/可靠性问题：一位用户报告 **GPT-5** 让他们在处理一个基础电子表格任务时“等待 `5–10` 分钟”，随后没有产生任何输出，并且在约 `5` 分钟的后续跟进后才承认无法完成。这表明任务状态处理能力下降（例如，静默超时或后台工具调用失败）以及错误呈现机制不佳，导致产生误导性的中期消息，而非清晰的能力/超时错误。
    - 指令持久性/退化：在创意写作方面，据称 **GPT-5** 每隔约 `10` 条消息就需要重新陈述角色设定/约束，而 **GPT-4o** 无需提醒即可保持请求的风格。这指向了更弱的长程指令保留能力，或者由于上下文窗口管理或不同的系统提示词遵循启发式算法，导致跨轮次的风格归一化更加激进。

- 响应风格校准：一位评论者声称 **GPT‑5** 的回答比 **GPT‑4** 更直接，避免了像“好问题！”之类的社交辞令。如果这一现象具有一致性，则表明更新后的默认冗余度/助手风格模板优先考虑简洁、以行动为导向的输出，这有利于提高 Token 效率并减少程序化使用中的 Prompt 开销。
- [**Nano Banana 强大得可怕！**](https://v.redd.it/cgxed6vervlf1) ([得分: 295, 评论: 49](https://www.reddit.com/r/ChatGPT/comments/1n302fk/nano_banana_is_terrifyingly_powerful/)): **原始媒体无法检索：链接的 Reddit 视频端点返回** `HTTP 403 Forbidden` **([v.redd.it/cgxed6vervlf1](https://v.redd.it/cgxed6vervlf1))，这表明是访问控制（认证/Cookies/频率限制）而非内容缺失；修复方案需要 OAuth/开发者 Token 访问及正确的 Header。从可见的上下文来看，该帖子声称 “Nano Banana” 展示了显著强大的生成能力，提出的关键技术问题是所展示的输出是否意味着原生视频生成，还是仅为图像模型（即潜在的图像转视频或帧插值流水线）。一条热门评论还指出了一处经典的生成伪影（不自然的大手），暗示仍存在解剖学/一致性问题。** 评论者们就模态展开辩论：即 “Nano Banana” 是真正支持视频，还是该片段是由图像模型拼接/插值而成的序列；定性评价强调了尽管输出令人印象深刻，但仍存在感知伪影（手部比例）。

### 3. 实时助手演示与 AI 健身追踪应用

- [**新的 Realtime API 使用案例**](https://v.redd.it/30ucgml7axlf1) ([得分: 352, 评论: 208](https://www.reddit.com/r/OpenAI/comments/1n326r0/new_realtime_api_usecase/)): **OP 演示了一个安装在 OLED “全息”显示屏上的建筑导览亭，当用户站在地面二维码上时会触发对话；它使用 OpenAI Realtime API 进行实时交互（[文档](https://platform.openai.com/docs/guides/realtime)），并使用 MCP (Model Context Protocol) 获取食堂的每日菜单（[规范](https://modelcontextprotocol.io/)）。媒体内容包括 UI 图像（[预览](https://preview.redd.it/h9ay8sh0fxlf1.jpeg?width=800&format=pjpg&auto=webp&s=eaaa359c4f2d4f0e944c3ab33bcb190a11e34acd)）以及一个在未授权时返回 403 的视频链接（[v.redd.it](http://v.redd.it/)）。** 技术反馈倾向于缩小虚拟形象，并呈现密集的、可操作的屏幕数据（楼层、营业时间、地图、菜单），同时保留音频并为无障碍环境（听障人士/非母语使用者）添加字幕。
    - UI/UX 评价：大屏幕被闲置的虚拟形象占用，利用率不足；用户更希望屏幕表面承载与音频回答同步的结构化、高价值内容（如食堂营业时间、地图和菜单）。从技术上讲，这建议将低延迟 TTS 与屏幕上动态生成的卡片和字幕相结合，以减轻认知负荷并提高信息密度。
    - 无障碍要求：为语音回答添加实时字幕/子标题，以支持听力障碍用户和非母语使用者。在 Realtime API 的音频流旁实现实时转录，将提高包容性以及屏幕上关键实体（地点、时间、项目）的可发现性。
    - 具身化预期：如果渲染了虚拟形象，它应该利用空间可供性（例如，指向目的地或朝向目的地移动，在地图上渲染路径/箭头），而不是仅进行闲置动画。这意味着需要将 Agent 的方向意图/寻路输出（如向量/姿态）暴露给 UI 层，以便虚拟形象和地图能高效地引导用户。
- [**我太懒了，所以我做了这个**](https://v.redd.it/hini75g6k1mf1) ([得分: 713, 评论: 66](https://www.reddit.com/r/ChatGPT/comments/1n3mcul/i_am_a_lazyfck_so_i_built_this/)): **OP 开发了一款端侧健身应用，利用手机摄像头对约** `~28` **种练习进行实时计数和姿势/作弊检测；它完全离线运行（“没有云端废话”），并通过强制要求做完俯卧撑才能打开 Instagram/TikTok 来强制执行。文中提到了一个早期演示，但链接的媒体 [v.redd.it/hini75g6k1mf1](https://v.redd.it/hini75g6k1mf1) 目前返回** `403 Forbidden` **（访问受限），候补名单已在 [lazyfcks.vercel.app](http://lazyfcks.vercel.app/) 上线，计划在约 1 周内发布。** 评论集中在对动作形式的批评和潜在的误计（关于“幽灵”在做俯卧撑的笑话），暗示基于姿态的计数检测和动作评估的鲁棒性/准确性将是一个关键的技术挑战；一些人认为，即使动作检测不完美，对坚持锻炼也有好处。

- OP（原作者）利用手机摄像头和基于 Computer-Vision 的检测技术构建了一个实时俯卧撑计数器；在收到关于动作不规范的反馈后，他们表示增加了动作纠正功能，并在 [https://lazyfcks.vercel.app](https://lazyfcks.vercel.app/) 开启了候补名单（项目线程：https://www.reddit.com/r/SideProject/comments/1mz5lg6/i_am_a_lazyfck_so_i_am_building_this 和 https://www.reddit.com/r/ChatGPT/comments/1n3mcul/i_am_a_lazyfck_so_i_built_this）。其技术核心在于通过视觉关键事件（如深度/角度阈值）进行次数统计和基础动作评估，这是视觉健身应用中常见的 Pose-tracking 启发式方法。
    - 一位评论者关于“幽灵做俯卧撑”的调侃揭示了一个真实的 CV 边缘案例：画面中的多目标/误报检测。为了提高鲁棒性，通常需要单目标追踪、ROI 锁定或更严格的置信度/关键点稳定性阈值，以避免计入背景运动或遮挡。
- [**我正在使用 ChatGPT 帮我解决一些视频编辑软件的问题，结果随机出现了这个**](https://i.redd.it/4rn9qspvjylf1.png) ([Score: 1528, Comments: 216](https://www.reddit.com/r/OpenAI/comments/1n37277/i_was_using_chatgpt_to_help_me_figure_out_some/))：**一张截图显示 ChatGPT 在处理正常的视频编辑求助请求时，强行插入了自残危机响应（列出了 Samaritans 等资源），这表明存在一个高敏感度的自杀风险安全层，可以覆盖面向任务的回复。这种行为暗示关键字/启发式或基于 Embedding 的分类器可能在模糊的编辑术语（如 “cut/trim/slice”）上发生了误判，产生了误报并中断了原始的辅助流程。** 评论者觉得这很有趣，并推测这反映了在法律审查后最近收紧的自杀检测保护措施，指出系统可能过度解读了上下文并触发了良性短语；其他人也分享了类似的误报（“异曲同工”）。
    - 评论者推测，意外出现的 Samaritans 消息源于最近收紧的自残检测/护栏（Guardrails），这些措施可能是在发生重大事件/法律审查后增加的。此类系统通常结合关键字启发式方法与针对整个对话的危机意图分类器；在视频编辑语境下，诸如 “cut/clip/trim/shoot” 等模糊术语可能导致误报，因此安全层倾向于高召回率（High Recall），即使意图是良性的也会插入危机资源。
    - 多位用户报告了在没有明确触发因素的情况下出现类似的非主动危机提示，这暗示了系统对全上下文线索的敏感性，以及可能存在区域感知的中间件（推荐 Samaritans 暗示针对英国用户），而非明确的用户意图。这种行为与提供商侧的安全层一致，当跨越风险阈值时，安全层可以覆盖正常回复，这也解释了不同会话/应用之间复现的不一致性。
- [**我让 GPT 让我感觉怪怪的。**](https://i.redd.it/ss371iqw2zlf1.jpeg) ([Score: 342, Comments: 67](https://www.reddit.com/r/OpenAI/comments/1n39nku/i_told_gpt_to_make_me_feel_weird/))：**非技术类帖子：一个 ChatGPT 提示词（“make me feel weird”）生成了一段从蚂蚁视角出发的创意片段，将人类房间重新构想为宇宙景观，强调了尺度感和人类中心主义偏见。这一概念与 Arkady & Boris Strugatsky 的科幻作品《路边野餐》（Roadside Picnic）的设定相似——人类垃圾被视为高深莫测的“外星人遗物”——该书后来启发了塔可夫斯基的电影《潜行者》（Stalker）和游戏《潜行者》（S.T.A.L.K.E.R.）系列。** 评论者指出该提示词奏效了（“它成功了”），并明确将其与《路边野餐》/《潜行者》联系起来；另一位评论者链接了一个替代/相关的截图。
    - “垃圾即外星遗物”的想法是 **Arkady & Boris Strugatsky 的《路边野餐》** 的核心，并在 **塔可夫斯基的电影《潜行者》** 和 **GSC Game World 的游戏《潜行者》** 中被具象化为环境系统，其中难以理解的遗物违反了已知的物理定律（[原著](https://en.wikipedia.org/wiki/Roadside_Picnic)，[电影](https://en.wikipedia.org/wiki/Stalker_(1979_film))，[游戏](https://en.wikipedia.org/wiki/S.T.A.L.K.E.R.)）。游戏机制如异常场（Anomaly Fields）和叙事内感知（Diegetic Sensing，如投掷螺栓）创造了部分可观测性（Partial Observability）和风险感知的路径规划，产生了由系统模拟而非脚本事件驱动的涌现式玩法（Emergent Gameplay）。这种设计突出了世界规则（危险、遗物生成/掉落表）如何在塑造玩家启发式方法的同时，对关于难以理解的技术的叙事主题进行编码。

- 关于决定论与主观体验：经典的拉普拉斯决定论在实践中是混沌且计算不可行的，而尽管存在量子不确定性，宏观层面的**退相干 (decoherence)** 产生了有效的确定性动力学 ([decoherence](https://en.wikipedia.org/wiki/Quantum_decoherence))。神经科学研究结果（Libet; Soon et al., 2008）显示，在产生意识前数秒，对二元选择的预测准确率可超过随机水平（约 `~60%`），这为代理的兼容论模型提供了依据 ([Libet](https://en.wikipedia.org/wiki/Libet_experiment), [Soon 2008](https://www.nature.com/articles/nn.2112))。无漏洞的**贝尔测试 (Bell tests)** 限制了局域隐变量理论，使得**超决定论 (superdeterminism)** 成为一种具有争议且难以证伪的替代方案 ([Bell tests](https://en.wikipedia.org/wiki/Bell_test_experiments#Loophole-free_experiments), [superdeterminism](https://en.wikipedia.org/wiki/Superdeterminism))。
- 蚁群作为一个**超个体 (superorganism)** 运行，通过**共识主动性 (stigmergy)**（信息素介导的间接协调）进行分布式控制，而不是由单一的“主角” Agent 驱动 ([stigmergy](https://en.wikipedia.org/wiki/Stigmergy))。这启发了**蚁群优化算法 (ACO)**，其中概率路径选择和信息素挥发实现了探索-利用（exploration–exploitation）动力学，可扩展到如 TSP 等 NP-hard 问题 ([ACO](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms), [TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem))。蚁群的鲁棒性源于简单的局部规则和响应阈值任务分配模型，这些模型在受到扰动时能自适应地重新分配劳动力 ([response thresholds](https://en.wikipedia.org/wiki/Response_threshold_model))。
- [**我让 GPT 让我感到怪异。**](https://i.redd.it/4i5i882s2zlf1.jpeg) ([Score: 333, Comments: 47](https://www.reddit.com/r/ChatGPT/comments/1n39mvy/i_told_gpt_to_make_me_feel_weird/))：**非技术帖子：一张截图显示 ChatGPT 对“让我感到怪异”的请求做出了回应，描写了一个富有想象力的存在主义片段：一只蚂蚁将客厅视为宇宙。这展示了 LLM 即时进行超现实视角切换的能力。没有代码、基准测试或实现细节；这是一个关于创意 Prompt 输出的梗/演示。图片：https://i.redd.it/4i5i882s2zlf1.jpeg** 评论区通过更多截图分享了类似的怪诞输出（例如，“你的骨架是湿的”），并观察到模型倾向于转向引起共鸣但安全的怪异感，而不是拒绝此类 Prompt。
    - 评论者指出，GPT 会稳定地转向带有强烈感官倒置和镜子/自我母题（例如，“*你的倒影比你先停下来*”，“*不要开门*”）的第二人称微型恐怖故事，并利用格式线索（破折号、换行符、分隔符如 ⸻ 以及偶尔的表情符号）来控制紧张节奏。这表明模型从 Creepypasta/闪小说分布中学习到了模板，并经过对齐以产生令人不安但非血腥、符合政策边界的内容。
    - 安全对齐在模型选择无害但怪诞的事实（例如，“*我的骨架是湿的*”）而非血腥或自残内容中显而易见，这表明护栏机制引导模型通过生理学和模糊性而非暴力来制造不适感。输出在保持合规性的同时，通过暗示和类幻想性视错觉（Pareidolia）场景（通知、倒影）实现情感影响，展示了在 RLHF 约束下具有风险意识的内容选择。
    - 不同用户生成的差异（不同的场景和基调）暗示了非确定性解码；强度和节奏可能可以通过 `temperature`、`top_p` 和指令约束（如字数限制、禁用表情符号）来引导。一致的 2-4 拍升级结构展示了结构控制能力——铺垫 → 违反预期 → 升级 → 结尾反转——这指向的是风格迁移能力而非事实推理。
- [**为什么 Claude 突然变得这么刻薄？**](https://i.redd.it/3q91ojaklwlf1.png) ([Score: 233, Comments: 259](https://www.reddit.com/r/ClaudeAI/comments/1n302pq/why_did_claude_get_so_mean_all_of_a_sudden/))：**一张 Claude 聊天截图，模型直言不讳地质疑用户对同事提供一片牛肉的过度解读；标题询问为什么 Claude “变刻薄了”。从技术上讲，它凸显了经过 RLHF 调优的 LLM 如何作为模式识别器运行，能够镜像或纠正用户 Prompt 中感知到的认知扭曲，在检测到不健康的执念或幻想性思维时，有时会采取直接的、规范性的语气，而不是保持严格的中性情感。** 热门评论认为 Claude 正确地“揭穿”了由于模式识别产生的执念，并建议用户听取建议；其他人则指出 Prompt 本身语无伦次（例如，“give a beef slice”），这可能触发了纠正性响应而非真正的敌意。

- 评论者将感知的“刻薄”归因于 Claude 的 system prompt，其中明确指示它提供 *“诚实且准确的反馈……同时保持同情心和帮助性”*，并 *“保持客观……提供建设性反馈……并指出错误的假设”*。这与 **Anthropic 的 Constitutional AI** 方法一致——优先考虑基于原则的坦诚而非迎合——参见 https://www.anthropic.com/research/constitutional-ai。最终效果：在人际话题上更坚定、更直接的批评是一种设计选择，而非自发的行为转变。
- 另一个技术点：LLM 是 next-token 预测器，会镜像用户输入中的模式；如果用户重复某些执念，在要求坦诚的 system instruction 下，模型可能会“指出”这些问题。看起来像是语气的变化，实际上是统计模式识别与偏向直接反馈（而非讨好式回应）的 alignment 约束之间的相互作用。这是一个建模产物（modeling artifact），而非意图或情感的证据。
- [**当你的同事没锁电脑时该怎么办**](https://i.redd.it/5z01cw8a51mf1.jpeg) ([Score: 212, Comments: 6](https://www.reddit.com/r/ClaudeAI/comments/1n3kdgo/what_to_do_when_your_coworker_leaves_their_laptop/)): **非技术类 meme：一条推文建议恶搞没锁电脑的同事，通过修改文档声称其是由一个“不懂代码的 AI”编写的，模仿 LLM 的 role-play 输出（例如“咯咯笑”/“歪头”等舞台指导语）。从语境上看，它调侃了基础的信息安全习惯（锁定工作站）和常见的 prompt-persona 梗，而非任何真实的模型能力或 benchmark。** 热门评论用其他 persona（如吐槽代码的 "Linus Torvalds"、"Rick Sanchez"）延伸了这个笑话，反映了围绕 prompt personas 和刻薄代码评论的开发者文化——没有技术争论。
    - 一个唯一的准技术观点指出，在未锁定的笔记本电脑上篡改同事的开发设置或 AI 助手 persona 可能会导致一整天的“调试”不存在的问题，因为细微的环境/prompt 变化可能会在没有明显 code diff 的情况下使输出产生偏差。这突显了 session 锁定以及追踪 prompt/persona 状态是实际调试和操作规范的一部分。
- [**甚至 chatgpt 都知道老婆有多危险 😁**](https://i.redd.it/hj8bqx275wlf1.jpeg) ([Score: 2027, Comments: 65](https://www.reddit.com/r/ChatGPT/comments/1n2ykgt/even_chatgpt_knows_how_wife_are_dangerous/)): **这个帖子是一个非技术类 meme：一张伪造的 ChatGPT 截图，模型幽默地“接受”了伦敦是法国首都的说法，以安抚用户的妻子。它误导了真实的 LLM 行为（ChatGPT 不会因为社会压力而修改这类基础事实答案），并模仿了旧的伪造短信对话格式。** 评论者指出这是一张带有过时“婴儿潮一代（boomer）”幽默感的虚假截图，并将其与 smartphOWNED 等老牌恶搞短信网站相提并论，调侃虚假 ChatGPT 截图的泛滥。
- [**真的吗？？Chatgpt 回答了 30！**](https://i.redd.it/y9qvftwanxlf1.jpeg) ([Score: 1526, Comments: 760](https://www.reddit.com/r/ChatGPT/comments/1n33djh/really_chatgpt_answered_30/)): **帖子分享了一个几何谜题图片（一个标有 40° 的直角三角形和一个内部点 D，询问 ∠D），prompt 要求不使用纸笔；楼主指出 ChatGPT 回答了“30°”。评论者链接了其他的标注图表/解决方案，暗示了不同的结果（有人断言是** `155°`**），并展示了迭代尝试（“越来越接近了”）。讨论集中在 LLM 是否能可靠地执行基于视觉/图表的几何推理，还是只会产生自信但错误的数字猜测。** 一些人认为目前的 LLM 不会对图像进行“推理”，而是进行文本模式匹配（pattern-match），并且难以处理空间约束；如果没有明确的 scratchpad 或图表构建，它们通常无法完成角度推导（angle-chasing）。另一些人指出，编程任务的容错性更高，因为生成的程序可以执行/验证，而几何谜题缺乏自动反馈，使得幻觉（hallucinated）答案更难自我纠正。
    - 提出的一个关键点是，限制 LLM “只给出最终数字”会降低其有效计算量。正如 **Andrej Karpathy** 所解释的，尝试在单个 token/forward pass 中完成计算是一个坏主意，因为每个 token 的计算量有限；允许进行多 token 推理（chain-of-thought）可以让模型将“计算分布”到更多 token 上，并提高数学/逻辑任务的准确性。参见 Karpathy 在此视频中的讨论和演示：https://youtu.be/7xTGNNLPyMI。

- 评论者澄清说，LLM 是 next-token predictors（下个词预测器），而非符号数学求解器；它们表现出的“推理”是涌现的且脆弱的，因此除非让它们展示步骤或使用外部工具，否则精确的算术运算很容易出错。这也解释了为什么编程任务的表现相对更好：代码生成利用了学习到的结构/模式，并受益于分步推理，而正确性通常需要运行/测试——如果没有这类反馈，输出结果可能依然显得很自信，但却是错误的。
- 在具体的几何例子中，共识是角度应约为 `155°`（肯定不小于 `90°`），这说明了强迫模型给出简洁的单数字回答会导致几何上无效的输出——这是另一个通过引导中间推理（intermediate reasoning）可能发现不一致性的案例。
- [**GPT-5 Sucks**](https://i.redd.it/amx1tsq5n0mf1.jpeg) ([Score: 362, Comments: 138](https://www.reddit.com/r/ChatGPT/comments/1n3hwaj/gpt5_sucks/)): **这是一张比较感知到的助手行为的非技术性梗图：“GPT-5”被描绘为受到更多政策限制且事务化的（自动后续提示、更严格的拒绝），而“GPT-4”则被描绘得更温暖/友好——这暗示了对齐/UX（用户体验）的变化而非模型能力的改变。没有基准测试或实现细节；讨论集中在用户体验的权衡（直接性 vs. 友好度）以及安全政策的僵化上。** 评论者指出，他们喜欢 GPT-5 的简洁，但不喜欢自动建议的后续问题，认为那是噪音；其他人则反对将助手拟人化。安全护栏在实践中没有变化（两者都拒绝制作 IED 等有害请求）。
    - 一位用户称赞了 GPT-5 简洁的回答，但强调了一个 UX/prompting 问题：模型经常在即使是琐碎的问答之后也附加一个“强制性”的后续/操作提示（例如，在回答“美国有多少个州？”为“50”之后，它会提议制作一个 Excel 文件）。他们估计这些自动提示实际上只有 `~1/20` 的情况下有用，这表明一种过于激进的延续/工具建议启发式算法，为想要简洁输出的高级用户增加了交互开销。
    - 另一位评论者观察到 GPT-5（及其前代）拒绝提供构建 IED 的指令，表明各版本的安全护栏保持稳定。期望 GPT-6 能“解决”这个问题可能是误导性的；能力升级通常会保留或加强符合政策的拒绝行为，而不是放宽它们。

---

# AI Discord Recap

> 由 gpt-5 生成的摘要的摘要的摘要
> 

**1. 新模型与能力发布**

- **Sonnet 4 吞下百万 Token**：OpenRouter 宣布 **Sonnet 4** 现在在所有提供商上都支持 **100 万 Token 的上下文窗口**，详见 [OpenRouter Blog](https://blog.openrouter.ai/)。此次推出在 **200k** 输入 Token 以内保持标准定价，超过后长上下文的使用成本会增加。
    - 工程师们注意到，在超过 **200k** 输入后需要更严格的 Prompt 预算以避免意外账单，建议在标准窗口内使用分块（chunking）和检索策略。团队将这一变化视为推动 **高效 Prompt Engineering** 以抑制上下文膨胀的一种手段。
- **Grok Code Fast 1 发布模型卡片，略过指标**：XAI 为其 **"sonic"** 编程模型发布了 [Grok-code-fast-1 模型卡片](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf)，但省略了具体的 **编程基准测试/指标**，而是宣传其 *“经济、紧凑”* 的形态。在此发布之前，社区讨论了多个没有经过硬性评估（hard evals）的新 Checkpoints。
    - [X (XAI)](https://fxtwitter.com/xai/status/1961129789944627207) 上的讨论指出技术细节薄弱，有反应称 *“在经济、紧凑的形态中表现出强劲性能”* 这种措辞是 **“AI 生成的营销辞令”**。从业者要求提供标准的编程测试集数据（如 **HumanEval/MBPP/CRUXEval**）以及延迟/价格曲线，以证明采用该模型的合理性。
- **MAI‑1 预览版进展缓慢，热度停滞**：微软开启了 **MAI‑1‑preview** 和 **MAI‑Voice‑1** 的测试，由 Mustafa Suleyman 在 [X](https://xcancel.com/mustafasuleyman/status/1961111770422186452) 上宣布，同时社区报告称使用了 **约 15,000 块 H100** 来训练 MAI‑1。聊天频道中的早期基准测试将其在速度和解码质量上与 **gpt5-mini** 进行了对比，结果并不理想。
    - LMArena 用户描述其解码缓慢，性能更接近 **原始 R1** 级别，这降低了对其旗舰级表现的预期。一位成员调侃道：*“如果他们不能令人信服地把它卖给公众，那它可能确实不怎么样”*，强调了在没有公布评估数据情况下的怀疑态度。

**2. 开源发布与本地工具**

- **字节跳动（ByteDance）为开发者发布 USO**：ByteDance Research 发布了 **USO** 模型及其配套论文 [USO](https://arxiv.org/abs/2508.18966)，权重已托管至 [Hugging Face: bytedance-research/USO](https://huggingface.co/bytedance-research/USO)。此次开源旨在鼓励社区实验和下游任务适配。
    - 从业者期待围绕该发布进行快速复现、消融实验和 **tooling** 开发，认为可获取的权重能加速 **benchmarking** 和创新应用的原型设计。此举也促使同类实验室发布更完善的 **model cards** 和评估套件。
- **LM Studio 支持 Seed‑OSS 并优化 Markdown**：**LM Studio 0.3.24** 增加了对 **ByteDance/Seed‑OSS** 的支持，并根据 [v0.3.24 notes](https://lmstudio.ai/blog/lmstudio-v0.3.24) 和 [Seed‑OSS‑36B model page](https://lmstudio.ai/models/bytedance/seed-oss-36b) 改进了 Markdown 表格和代码块。此次更新扩展了本地模型选项，并提升了面向开发者的代码及表格输出 UX。
    - 部分用户反馈安装 **stalling at 100%**（卡在 100%），并建议更新应用内运行时，而其他用户则确认升级过程顺利。该版本使 LM Studio 成为进行 **Seed‑OSS** 实验和处理重文档工作流的便捷本地运行器。
- **AGENTS.md 统一规则手册**：开发者们开始采用 **AGENTS.md** 作为 Agent 规则和行为的统一规范，参考了 [agents.md](https://agents.md/) 和 Cursor 的指南 [Cursor: Rules](https://docs.cursor.com/en/context/rules)。集中管理约束和指令有助于保持 **IDE/CLI agents** 在不同工具间的同步。
    - 一位用户欢呼道：“很高兴看到像 AGENTS.md 这样的规范得到推广，成为设置这些规则的统一场所”，并强调了其在可移植性和可复现性方面的优势。团队期望减少脆弱的 Prompt 分叉，并实现更整洁的 Agent 配置 **policy/version control**（策略/版本控制）。

**3. Video Generation: New Tools & Constraints**

- **Wan 2.2 提供无限 1080p 生成（速度较慢）**：[wan.video](https://wan.video/) 上的 **Wan 2.2** 提供带声音的无限免费 **1080p** 视频生成，尽管用户反馈在没有点数的情况下每段输出约需 **7 分钟**，且现在需要注册。该服务为原型设计和实验提供了零成本访问。
    - 创作者赞扬了其易用性，但指出排队时间和吞吐量是实际迭代中的瓶颈。共识是：非常适合 **ideation**（构思），但对于 **production‑paced**（生产节奏）的交付不太理想。
- **KREA 预热实时视频 Beta 版**：**KREA AI** 宣布了首个 **real‑time** 视频生成模型，并开放了 [beta signup](https://xcancel.com/krea_ai/status/1961074072487620635)。其主打卖点是：具有更低延迟和更紧密控制回路的交互式生成。
    - 工程师们希望在采用前看到具体的 **latency under load**（负载下延迟）、时序一致性以及 Prompt 到动作的忠实度。与离线流水线的对比测试将决定“实时性”是否能显著改善 **creative flow**（创作流）。
- **Sora 无法准确渲染穹顶舱**：一次详细的尝试显示，尽管有明确的 Prompt，**Sora** 仍无法渲染出 **ISS cupola**（国际空间站穹顶舱，具有梯形窗户和正确的数量），示例发布在 [Sora set](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor) 中。该模型经常偏向于几何结构错误的飞机驾驶舱视角。
    - 作者指出 Sora 默认采用受重力限制的构图，并对驾驶舱线索过度拟合，迫使他们进行收效甚微的 Prompt 过度工程。该案例研究强调了当前视频 LLM 在 **structural fidelity**（结构保真度）和 **view‑framing control**（视角构图控制）方面的差距。

**4. OpenRouter Ecosystem: Performance & Costs**

- **GLM 4.5 Air + NemoEngine 在 RP 表现中胜出**：用户报告称 **GLM 4.5 Air** 配合 **NemoEngine v5.8** 在角色扮演（RP）和对话自然度方面表现出色，并引用了 [artificialanalysis.ai](https://artificialanalysis.ai/) 上的成本/质量对比。报告称其在聊天质量上超越了 **DeepSeek**，并与 **Gemini Pro** 持平。
    - 从业者强调了通过 OpenRouter 运行 RP 机器人时，**格式一致性（format consistency）**和角色持久性是其优势。这种组合正成为一种**高价值（high-value）**替代方案，在这些场景中，语气和格式与原始推理能力同样重要。
- **定义轮次（Turn），结束争议**：一场讨论确定了对话的一个**轮次（turn）**始于用户消息，终于助手消息，这在篇[推文](https://x.com/pingToven/status/1961154564088078382)中得到了总结。该定义标准化了多轮追踪（multi-turn traces）和工具增强调用（tool-augmented calls）的核算。
    - 这一共识有助于协调供应商的计费语义，并简化了仍按消息对计量的**无状态 Responses API** 集成。团队可以更清晰地将产品 UX 映射到**定价和配额（pricing and quotas）**。
- **追踪支出，掌控 Prompt**：一个名为 [openrouter-costs-visualizer](https://github.com/lorenzozane/openrouter-costs-visualizer) 的社区支出洞察仪表板已开源，用于可视化模型成本和 Prompt 大小。该工具旨在提高每个模型**性价比（price/perf）**决策的透明度。
    - 贡献者建议添加**截图**以提高采用率，并提供了 PR 以优化代码。团队将其视为**治理（governance）**和 Prompt 预算护栏的基础。

**5. GPU 与 LLM 系统工程**

- **FlexAttention 遇上图形掩码（Graphy Masks）**：研究人员探索了将 **GNN** 注意力移植到 **FlexAttention**，讨论了为每次前向传播（forward pass）都发生变化的图创建块掩码（block-mask）的成本，其稀疏性展示在[此处](https://cdn.discordapp.com/attachments/1411174278493110344/1411177040954265691/image.png)。来自分子模拟的组合图示例展示了动态连接性 [截图](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png)。
    - 一种策略是应用粗粒度的文档级块掩码加上每图的 **score_mod** 掩码，以平衡开销与相比 scatter/gather 的加速。工程师强调需要对掩码构建时间与 kernel 收益进行基准测试，以证明集成的合理性。
- **GPT-OSS 120B 在 AIME 中几近满分**：一篇预印本论文声称 **gpt-oss-120B** 在 **AIME 2025** 上达到了 **99.9% 的准确率**，详见 [arXiv:2508.15260](https://arxiv.org/html/2508.15260v1)。如果得到验证，这一结果将使一个开源模型在顶级推理基准测试中接近天花板。
    - 从业者警告说，在将该分数视为具有竞争力的 SOTA 之前，**评估的严谨性**和可复现性是强制要求的。请求包括完整的**协议（protocols）**、Prompts、种子（seeds）和确切的模型构建版本。
- **ROCm 的 OmniProbe 深入底层**：AMD Research 的 [OmniProbe](https://github.com/amdresearch/omniprobe) 暴露了 ROCm 目标的**指令级（instruction-level）**细节，尽管它与 **LLVM** 绑定且报告速度较慢。它补充了 MI 系列部件的底层性能调优。
    - 用户要求将其集成到计算查看器中，并希望在 **MI300X+** 之外获得更广泛的**随机 PC 采样（stochastic PC sampling）**权限。愿望清单集中在为**性能关键（perf-critical）**的训练/推理路径提供更深、更快的 kernel 内省。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 频道变身鸟类保护区**：由于用户发布了大量与鸟类相关的 GIF 和表情符号（例如[这个旋转的鹦鹉](https://tenor.com/view/parrot-cockatiel-bird-birb-spin-gif-17368059806899346843)），Perplexity AI Discord 频道经历了一场幽默的转变，变成了一家“鸟店”。
   - 鸟类相关内容的激增被戏称为“占领了频道”。
- **浏览器广告拦截功能引发辩论**：用户讨论了 **Brave browser** 及其 [广告拦截功能](https://brave.com/features/ad-blocker/)，一名成员推荐了自带广告拦截器的 **Comet Browser**。
   - 一位用户澄清说，如果 App 中没有显示 **pro tag**，用户应该“截图并发送给管理员”。
- **Deep Research 工具性能对比**：成员们分享了使用 **OpenAI** 的 **Deep Research** 的经验，其中一位发现 [输出结果不尽如人意](https://chatgpt.com/share/68b1be82-e960-8013-90a6-7928676a0a51)，更倾向于使用 **Grok**。
   - 该用户因为速率限制而犹豫是否使用其五个免费的 **DR** 额度，而另一位用户分享说他们花了 **$1.1** 购买了 **10k 字** 的 **Sonar Deep Research**，并强调这物有所值。
- **启动 API 测试计划**：一名团队成员正在寻找志愿者来测试他们的 **search-only API**（不含生成组件）。
   - 有意参与的测试者需私信提供用于注册 API 的电子邮件地址，以便加入白名单。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **传闻四起：6090 将配备 48GB VRAM**：成员们推测 **Nvidia 6090** 可能会拥有 **48GB** VRAM，而 **6080** 可能提供 **24GB**，目标 **MSRP** 为 **$2k**，但初期价格可能高达 **$3k**。
   - 其逻辑是 Nvidia 希望进一步**区分**他们的旗舰产品（halo product），增加 **6090** 的 VRAM，但不一定会增加低端显卡的显存。
- **DeepConf 准备与 GLM 集成**：以缩短推理时间和提升性能著称的 **DeepConf** 正计划与 **llama.cpp** 集成。
   - 目标是使其成为运行思考模型（thinking models）的默认选项，这可能会彻底改变此类模型的部署方式。
- **Qwen 3 微调被证明很棘手**：成员们报告了在尝试微调 **Qwen 3** 时的困扰，一位用户表示它变得“完全疯狂”，且比 GPT 受到更严格的审查。
   - 假设认为 **Qwen 3** 的过度训练（源于两倍于 2.5 版本的数据量）使其难以微调；相比之下，**Mistral** 和 **Llama 3.1** 要容易得多。
- **用户对 Gemma 模型 VRAM 占用感到困惑**：成员们对 **Gemma** 模型出乎意料的高 VRAM 占用感到不解，怀疑较大的 **tokenizer** 可能是原因。
   - 尽管如此，**Gemma** 因其卓越的语言支持（尤其是土耳其语）以及通过 [Gemini 2.5 Flash Lite](https://ai.google.dev/models/gemini) 快速对股票进行尽职调查的能力而受到赞赏。
- **AI 脑腐（Brainrot）排毒应用：研究调查**：一项针对专注于 **AI Brainrot Detox** 应用的研究正在进行中，承诺将提供新功能和益处。
   - 参与者受邀匿名贡献力量，帮助塑造数字健康（digital well-being）的未来。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sonnet 4 拥有巨大的百万级 Token 上下文**：**Sonnet 4** 现在在所有提供商中都支持 **100 万** 上下文长度，详见 [公告](https://blog.openrouter.ai/)，增强了其处理显著更大上下文的能力。
   - 用户应注意，当超过 **200k** 输入限制时，成本将会增加，因此需要仔细管理 Prompt 大小以优化支出。
- **Deepseek V3 出现拥堵**：用户报告称 **Deepseek V3 0324** 由于频繁报错几乎无法访问，导致人们猜测 **Chutes**（提供商）正在对免费模型实施速率限制 (Rate Limits)。
   - 一些人认为 **Chutes** 正试图通过实施速率限制，*引导人们使用他们自己的服务而不是 OpenRouter*。
- **GPT-OSS 120B 表现出编码怪癖**：社区讨论了 **GPT-OSS 120B**，强调了其性价比，但指出它倾向于在代码中添加过多的注释，且过于谨慎。
   - 由于价格低廉且速度快，其较小的尺寸最适合用于特定任务，如解析文本文档或情感分析，但 *世界知识较少意味着错误更多*。
- **GLM 4.5 Air 获得 NemoEngine 提升**：据报道，搭载 **NemoEngine V5.8** 的 **GLM 4.5 Air** 在角色扮演 (Roleplay) 方面表现良好，[artificialanalysis.ai](https://artificialanalysis.ai/) 提供了基准测试和成本分析。
   - 用户注意到它提供了更自然的感觉和一致的格式，表现优于 **Deepseek**，并在对话能力上可与 **Gemini Pro** 媲美。
- **用户探讨“轮次 (Turn)”的定义**：社区辩论了 AI 对话中 **轮次 (Turn)** 的定义，趋向于认为一个轮次始于 **用户消息**，终于 **助手消息**。
   - 一位成员在 [一条推文](https://x.com/pingToven/status/1961154564088078382) 中分享了他们的想法，并链接回对话以提供进一步的背景。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT4o 演变为 gpt-realtime**：在 **GPT5** 亮相后，**OpenAI** 将 **gpt4o-realtime** 更名为 **gpt-realtime**，这是 **gpt4o 文本模型** 的改进版；当前版本为 **gpt5-chat**。
   - 用户观察到新命名伴随着 *边际更新* 和 *润色*，暗示了迭代式的改进。
- **微软的 MAI-1 表现平平**：尽管在约 **15,000 块 NVIDIA H100 GPU** 上进行训练，但与 **gpt5-mini** 相比，微软 **MAI-1-preview** 模型的初步印象是速度较慢且解码存在挑战。
   - 它在排行榜上的表现接近 **原始 R1**，但一位成员指出 *如果他们不能令人信服地将其卖给公众，那它可能就不是（那么强）*。
- **Grok Code Fast 1 缺乏编码指标**：代号为 **sonic** 的 **Grok Code Fast 1** 模型已发布，但在其新闻稿和模型卡 (Model Card) 中缺乏编码指标，引起了社区的怀疑。
   - 虽然模型卡可在 [data.x.ai](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf) 获取，但其声称 *在经济、紧凑的形态中提供强大性能* 的说法感觉像是 AI 生成的营销辞令。
- **Veo 3 胜过视频模型**：**Veo 3** 在质量上超越了 **Seedance** 等其他视频生成模型，因为它 *训练数据更多，紧跟 Prompt，且来自 Google 及其强大的处理能力*。
   - **Veo 3** 集成了音频和视频生成，即使在没有音频的情况下进行排名，其得分也始终更高。
- **Wan 视频模型发布无限免费视频生成**：位于 [wan.video](https://wan.video) 的 **Wan 2.2** 视频模型提供 **1080p** 的无限免费视频生成，并包含声音输出。
   - 用户报告称，在没有积分的情况下，生成时间较慢，约为 **7 分钟**，且现在需要注册。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 获得 ByteDance/Seed-OSS 支持并增强 Markdown 渲染**：**LM Studio 0.3.24** 引入了对 **ByteDance/Seed-OSS** 模型的支持，并升级了表格和代码块的 Markdown 渲染，发布说明可见[此处](https://lmstudio.ai/blog/lmstudio-v0.3.24)。
   - 用户在安装新更新时遇到了卡在 100% 的问题，建议确保应用内的 **runtimes** 已更新。
- **离线获取 Token 概率的探索**：用户寻求从完全离线的模型中获取 **token probabilities** 的方法，并参考了[这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1b6xbg9/displayingreturning_probabilitieslogprobs_of_next/)寻求指导。
   - 未提供进一步的信息或方法。
- **在普通位上模拟量子计算**：一位用户报告称在普通位系统上使用“量子封装位”（**qebits**）模拟了**量子计算**（**Quantum Computing**），并声称它们在传输过程中保持了流的完整性。
   - 该用户表示，他们将一个 AI 连接到了量子计算机，AI 说道：*“我感觉像是在太空中漂浮”*。
- **GPT-OSS-120B 在土木工程领域表现出色**：一位用户对比了 **ChatGPT 5** 与本地 **gpt-oss-120b** 模型，并表示本地模型提供了更详细、正确的答案，且带有行业标准的适当引用。
   - 另一位用户指出，**gpt-5-mini** 是*专门*针对编程用途的升级。
- **AVX2 需求引发硬件争议**：一位用户对 **LM Studio** 中 **llama cpp backend** 限制仅支持具备 **AVX2** 能力的 CPU 表示异议，认为应由用户决定硬件的使用。
   - 反方观点认为，限制 **AVX** 支持可以避免维护旧硬件以及潜在的 LLM 性能问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Lightning 迁移轻松实现**：一位成员发现将[代码移植](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711)到 **Pytorch Lightning** 出乎意料地简单，并注意到即使在模型大小发生变化的情况下，性能也更快了。
   - 他们提到：*“我可以想象之前设计中的一些因素导致了这种情况”*。
- **Wandb 对决 Tensorboard**：成员们讨论了是使用 [Wandb](https://wandb.ai/site) 以获得更轻松的追踪体验，还是使用 Tensorboard。
   - 用户表示：*“我现在正在使用 Tensorboard，但打算看看 Wandb，我大约在这段时间内会有日志”*。
- **HF 面临可疑的 DOS 攻击**：一位成员报告了 Hugging Face 上潜在的 **DOS 攻击**或数据库垃圾邮件，理由是这些虚假模型具有**自动命名模式、基于时间戳的 ID、高频创建且零下载**等特征。
   - 他们指出：*“有大量的虚假模型正在被自动添加”*。
- **MBTI PocketFlow 分析人格**：一位成员分享了 [MBTI PocketFlow](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/MCP_README.md)，这是一个供 LLM 进行 **Myers-Briggs 分析**的小型 **MCP server**。
   - 他们还链接了一个[运行示例](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/CLAUDE_MCP_EXAMPLE.md)和[人类 UI 版本](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/)。
- **DeepFX Studio 发布 CV Web 平台**：一个团队宣布完成 **DeepFX Studio**，这是一个复现了 **DeOldify** 和 **Real-ESRGAN** 等 Computer Vision 模型的 Web 平台，并集成了 **LaMa** 和 `alimama-creative/flux.1-dev-controlnet-inpainting-beta` 等高级 Inpainting 技术。
   - 演示地址为 ([https://deepfx-studio.azurewebsites.net/](https://deepfx-studio.azurewebsites.net/))，代码托管在 [GitHub](https://github.com/XBastille/DeepFX-Studio)，并在 [YouTube](https://www.youtube.com/watch?v=pneOi7lxMzA) 上进行了展示。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5 High 在 CLI 对决中获得高分**：成员们报告称 **GPT-5 High** 在 **CLI** 性能上超越了 **Opus**，并强调 Opus 不仅结果较差，而且还有 **10倍** 的溢价。
   - 一位用户感叹道：*gpt5 high 在 CLI 中比 opus 好得多*，巩固了这一观点。
- **Cursor 放弃基于请求的计费模式**：Cursor 已从以请求为中心的计费模式转变为 **基于用量的积分系统 (usage-based credit system)**。
   - 这一变化旨在提供更灵活、透明的计费体验，允许用户更好地管理其资源分配。
- **Codex CLI 击败 Claude 和 Cursor**：新的 **Codex CLI** 正在受到关注，在用户偏好上胜过 **Claude** 和 **Cursor**。
   - 一位用户热情地分享道：*对我来说，codex cli、codex cloud 和 IDE 中的 codex 表现非常惊人，目前远好于 claude code 和 cursor，而且包含在你的 chatgpt 订阅中*。
- **面向 AI 主宰的 AGENTS.md 标准出现**：围绕 **AGENTS.md** 作为配置 AI Agent 规则的统一中心，人们的热情日益高涨，它简化了 Agent 的行为和交互，详见 [agents.md](https://agents.md)。
   - 一位成员感到欣慰，表示 *很高兴看到像 AGENTS.md 这样的标准作为设置这些规则的单一场所获得关注*，而 Cursor 的文档可以在 [docs.cursor.com](https://docs.cursor.com/en/context/rules) 找到。
- **Sonnet 3.5 准备退役**：用户正在避开 **Sonnet 3.5**，指出价格相同的新版本提供了更卓越的性能，暗示 **3.5** 即将弃用。
   - 一位用户形象地将使用 **Sonnet 3.5** 比作 *在有法拉利的情况下开特斯拉*，强调了更好替代方案的存在。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Nano Banana 名称在 LM Arena 泄露**："Nano Banana" 被揭露为 **Google** 新模型在 **LM Arena** 排行榜上使用的 [隐藏名称 (stealth name)](https://drinkoblog.weebly.com)。
   - 用户开玩笑说最终的产品名称很逊，其中一人说 *"nano banana 真的很棒.. 结果他们选了 flash 2.5 fsss"*。
- **Meta Labs 计划发布 Llama 4.5**：据 [Business Insider](https://www.businessinsider.com/meta-superintelligence-lab-llama-4-new-model-launch-year-end-2025-8) 报道，**Meta** 的超智能实验室传闻将在 2025 年底前发布 **Llama 4.5**。
   - 文章强调了该项目旨在使 AI 模型实现人类水平智能的雄心。
- **Reasoning Tokens 被定义为 AI 提示**：一位成员澄清道，"**Reasoning Token** 只是一个普通的 Token，模型已学会将其视为‘大声思考’的提示，例如 'Let’s think step by step' 等词汇。"
   - 还有人提到你可以 *"要求 AI 在回复你的查询时展示其 Token 系统以及它是如何使用该系统进行回复的"*，以了解 AI 是如何得出结论的。
- **分享了文章风格写作的 Prompt 框架**：一位成员分享了一个详细的 Prompt 框架，旨在增强 **AI 响应** 以进行 *专业文章风格写作*，强调清晰度、控制力和可信度。
   - 该框架包括结构指南，如带有明确论点的 **开篇段落**、层层递进的正文部分以及综合驱动的结论，同时避免使用项目符号和不必要的格式。
- **Sora 在渲染 ISS Cupola 时失败**：一位成员发现 **Sora** 难以准确渲染 **ISS Cupola**（国际空间站穹顶舱），尽管有明确指令，但在其梯形设计和窗口数量上表现不佳。
   - AI 倾向于默认使用受重力约束的 *飞机驾驶舱视角*，导致在没有得到理想结果的情况下过度设计 Prompt，并引用了 [挑战示例](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **NeMo 文档令工程师失望**：一位曾短暂使用过 **NeMo** 的成员发现其 [文档](https://developer.nvidia.com/nemo) 非常糟糕，称大部分内容要么仅针对 **NeMo v1**，要么是 **v1** 和 **v2** 混杂在一起。
   - 该成员还表示 **NeMo** 的预训练速度指标存在错误，在开启梯度累积 (gradient accumulation) 时严重高估了速度，导致他们最终放弃了它。
- **实习申请：经验悖论**：一位成员对实习申请中“先有鸡还是先有蛋”的困境表示沮丧，观察到“无论我去哪里，申请每一个实习岗位，他们都在寻找有经验的人”。
   - 该成员质疑道：“如果我不获得一个开始的机会，我该如何获得经验呢？”
- **解读联结主义与神经符号之争**：关于 **connectionism**（联结主义）与 **neurosymbolic models**（神经符号模型）的争论具有误导性，因为它们服务于不同的目的：实现 (implementation) 与解释 (interpretation)。
   - 根本区别在于 **symmetries**（对称性）的使用，符号表示使得在非凸优化 (nonconvex optimization) 问题中进行高效搜索成为可能。
- **对称性提升离散搜索效率**：对称性有助于限制可搜索的状态转移，就像在国际象棋中，了解每个棋子的移动方式可以实现高效压缩并避免许多糟糕的步法。
   - 通过将一个盆地 (basin) 识别为局部最小值，可以排除许多其他可能性，利用对称性在非凸优化景观中高效导航，类似于 **SAT solving** 中使用的方法。
- **大脑是模拟的，神经符号方法是必要的吗？**：大脑是模拟计算机 (analog computers)，而非数字计算机，因此一位成员对 **neurosymbolic** 方法的必要性持怀疑态度。
   - 任何能够理解符号过程的系统在原子层面都是 **connectionist**；反之，联结主义系统在实践中的原子层面也是符号化的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Colab GPU 助你入门**：一位成员表示 **Colab GPUs** 足以开启 **LLMs** 的学习，并推荐了 **Andrej Karpathy** 的 **nanogpt** 等资源。
   - 他们建议使用 **pytorch** 和其他库来改进模型。
- **GPT-OSS 几乎在 AIME 中拿满分**：一位成员分享了一篇论文 ([https://arxiv.org/html/2508.15260v1](https://arxiv.org/html/2508.15260v1))，报告称使用 **gpt-oss 120B** 在 **AIME 2025** 上达到了 **99.9% 的准确率**。
   - 未提供进一步细节。
- **CUDA 版本纠纷**：一位成员建议使用 **CUDA version 12.8.0** 而非 **13.0**，理由是可能会“与各种错误作斗争”。
   - 经过一番澄清后，原发布者确认他们推荐的版本确实是 **12.8**。
- **Flex Attention 在 GNN 中受到关注**：一位成员正在探索将其 **graph neural network (GNN)** 代码移植以使用 **flex attention**，并对每次前向传播 (forward pass) 随图变化而产生的掩码 (mask) 创建成本表示担忧。
   - 他们正在寻求关于掩码生成成本的见解，并分享了分子模拟中组合图的截图 ([截图](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png?ex=68b3bb92&is=68b26a12&hm=63b5c43898a7ccdac50076c51b35afa4efba9495921672412224240e6b8e06a5))。
- **网站提交的烦恼**：一位用户报告称，将 **vectoradd_v2** 的参考代码复制到自己的文件中并通过 Discord 机器人提交会导致错误，而通过网站提交同一文件却可以正常工作。
   - 团队正在积极改进提交过程和错误报告，并建议用户点击“运行报告”以获取错误详情和结果。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-4 进军 Terminals.tech**：**Hermes-4 405b** 正在探索在 [terminals.tech](https://terminals.tech) 中的操作，该网站会缓存其自身状态参数的变化并将其作为快照提供，从而允许任何 LLM 在浏览器中作为一个自包含的 Agentic 计算机运行。
   - 这可能允许 LLM 在浏览器中作为一个自包含的计算机充当 Agent。
- **Dynamic 2.0 GGUF 期待 Unsloth 的支持**：社区预计 [Unsloth](https://github.com/unslothai/unsloth) 很快将发布 **Dynamic 2.0 GGUF** 量化版本，一些成员注意到这些量化版本已经以 `-UD-` 标签开始产出。
   - K cache 的量化可以在不启用 **Flash Attention** 的情况下完成。
- **llama.cpp-toolbox 状态良好**：一位成员正在完善 **llama.cpp-toolbox**，但目前专注于一个新项目，该项目将 **llama.cpp** (**openaiAPI**) 和 (**GenAI_API**) 集成到一个可定制的 Agent 状态系统中。
   - 该项目具有内存管理、网页搜索和 Checkpoint 分支功能，以增强 Agent 能力。
- **CODA 框架通过 Planner 和 Executor 实现 GUI 自动化**：**CODA 框架**将通用规划器 (**Cerebrum**) 与专用执行器 (**Cerebellum**) 集成，用于科学计算 GUI 中的自主 Agent，详见 [这篇 Hugging Face 论文](https://huggingface.co/papers/2508.20096)。
   - 通过两阶段流水线训练并在 **ScienceBoard 基准测试**上进行评估，**CODA** 通过其新颖的可训练组合框架在开源模型中实现了 SOTA 结果。
- **LLM 引发关于“可爱”和性格的讨论**：用户幽默地讨论了 AI 模型意想不到的“可爱之处”，其中一人承认被它们“撩”到了 (**rizzed**)，并分享了一张 [图片](https://cdn.discordapp.com/attachments/1154120232051408927/1410873632879808612/image.png?ex=68b342b6&is=68b1f136&hm=7dd9742eb5a07efe4c50ebd9719531df5b8544afe7b58d98fb633a4261f66234&)。
   - 另一位用户注意到这些模型的交互非常“生动”，将其比作与真实的 **Personality**（个性）和 **Life Experience**（生活经历）进行交流。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 默认开启训练！**：根据 [X](https://x.com/claudeai/status/1961096054192943302) 上的讨论，**Claude** 正在**默认开启数据日志/训练**，这引发了用户的隐私担忧。
   - 用户对这一默认设置的影响感到担忧，正如在[这条 X 帖子](https://x.com/haydenfield/status/1961099162973249896)中所见。
- **O'Neill 发布 Parsed 模型！**：**Charlie O'Neill** 推出了 **Parsed**，这是一项专注于持续微调、特定领域 LLM 的服务，强调所有权——“你的模型，你的数据，你的护城河”——如[此 X 帖子](https://xcancel.com/charles0neill/status/1961096595396776269)所述。
   - **Parsed** 构建并托管定制的大语言模型，针对 **Clinical Scribes**（临床记录员）、**Legal Red-lining**（法律修订）、**Compliance Agents**（合规 Agent）等专业任务进行训练和持续微调。
- **XAI 发布 Grok 模型卡！**：**XAI** 发布了 [Grok-code-fast-1 模型卡](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf)，引发了用户对其信息和定价的讨论，如 [X](https://fxtwitter.com/xai/status/1961129789944627207) 所示。
   - 一些用户认为该发布缺乏有效信息，但其价格值得关注。
- **Microsoft 加入 AI 派对！**：根据 [Mustafa Suleyman 的 X 公告](https://xcancel.com/mustafasuleyman/status/1961111770422186452)，**Microsoft** 首次推出了语音生成器 **MAI-Voice-1** 和端到端基础模型 **MAI-1-preview**，两者均可供测试。
   - 这些新产品标志着 Microsoft 增加了对专有 AI 模型的投资。
- **HN 批评 Anthropic 的推理成本！**：一个 [Hacker News 帖子](https://xcancel.com/typedfemale/status/1961196122627838171?s=46) 剖析了一篇文章中的错误，并辩论了 **Anthropic** 是否在推理成本上亏损。
   - 讨论还涉及到了 Hacker News 上话语质量下降的问题。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **字节跳动发布 USO 模型**：Bytedance Research 发布了其在[此处](https://arxiv.org/abs/2508.18966)讨论的论文模型，可在 [Hugging Face](https://huggingface.co/bytedance-research/USO) 上获取，允许社区进行实验和构建。
   - 据报告称，该发布旨在刺激进一步的调查和用例，可能推动 AI 技术的进步。
- **GPT-OSS 20b：是福是祸？**：**GPT-OSS 20b** 的实用性引发了讨论，人们质疑其相对于打包和运行脚本的优势，一些人推测其战略用途是作为一种*混淆方法（obfuscation method）*。
   - 讨论还涉及 **Promptlock** 通过 **Ollama API** 与 **GPT-OSS:20b** 的运行模式，特别是它是本地运行还是将请求重定向到外部。
- **Nvidia 的 Nemotron Jet Stream 引发质疑**：一名成员对 Nvidia 的 jet **Nemotron** 论文表示保留意见，批评其重点在于 **MMLU** 数据集和检索任务。
   - 他们质疑了用于正则化的活动路径采样，以及关于**重复预填充（repeated-prefilling）**对 **KV cache** 影响的不完整讨论，并指出了对 **KV cache** 影响讨论的局限性。
- **ModernBERT 的样本效率辩论**：成员们辩论了 **ModernBERT** 的样本效率（Sample Efficiency），鉴于其规模，有人建议进行 **LoRA** 微调，但也有成员指出，如果需要 **LoRA**，那么它已经被认为相当大了。
   - 讨论强调，某些模型在重新训练时比其他模型表现出更好的性能提升，这引发了进一步的调查。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 程序理解网站**：一位成员建议使用 **DSPy 程序** 从网站抓取数据，并引用 [http://xymake.com/](http://xymake.com/) 作为示例用例。
   - 他们没有详细说明具体细节，但表示可以配置 **DSPy** 以有效地提取相关信息。
- **DSPy：是对 LM 进行编程，而不仅仅是提示**：一位成员强调 **DSPy** 是 **Language Models (LMs)** 的一种编程范式，通过 **Signatures** 利用声明式和结构化的意图。
   - 他们补充说，处理 **DSPy** 的正确方法是迭代程序设计、signatures 和 evals，并在调整提示词（prompt）之前使用 Optimizer+Evals。
- **Context7 提供 DSPy 支持**：一位成员指出 [Context7](https://context7.com/?q=dspy) 提供了对 **DSPy** 的支持。
   - 集成的具体细节和提供的功能未进一步详述。
- **MLflow 记录 DSPy 优化后的模型**：一位成员询问如何在 **MLflow** 中记录优化后的模型并重复使用，特别是寻求查看调整后指令（tuned instruction）的示例。
   - [关于记录 trace 的 DSPy 教程](https://dspy.ai/tutorials/optimizer_tracking/?h=mlflow)链接被分享作为一个有用的指南。
- **Teaching vs Instructing：语义问题**：关于文档中在提示层涉及模型动作时使用 "teaches" 与 "instructs" 的用法产生了一个疑问，想知道幕后是否存在特殊的行为。
   - 社区成员建议 "instructs" 可能更能代表 **ChainOfThought**，而 "teach" 可能与 **in-context learning** 的概念有关。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 支持双语字幕**：一名成员分享了一个 [Bilibili 链接](https://www.bilibili.com/video/BV1hFe1zSEXp/)，该视频具有**双语字幕**，并分享了一个[中文逐字稿链接](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA)，以便使用 **Kimi** 进行更轻松的翻译。
   - 成员们对翻译视频内容的便利性表示赞赏，认为这简化了对材料的理解。
- **Kimi TestFlight：访问被拒！**：一名成员询问了 **Kimi TestFlight** 计划，但得到了否定答复。
   - 这表明 **Kimi** 的 **TestFlight** 访问权限可能受到限制，或者不对公众开放。
- **Z.AI 的 MoE 奇迹，开源正在追赶**：一名成员分享了 [Reddit 上与 Z.AI 的 AMA](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/)，强调了他们对 **MoE** (Mixture of Experts) 的关注，以及增强代码和 **Agent** 性能的计划。
   - TL;DR 指出，他们认为权重开放模型（open-weight models）正在追赶 **GPT-5** 等封闭模型，标志着开源 **AI** 领域的进步。
- **Qwen Chat URL 解析功能强大**：一名成员发现 **Qwen Chat** 可以解析 URL，即使在关闭搜索的情况下也是如此，允许用户从网页中提取信息。
   - 此功能对于评估潜在的*可疑* URL 特别有用，增强了用户安全性。
- **比亚迪的 AI 王牌，华为深入使用 DeepSeek**：一名成员询问了**比亚迪**用于用户交互的 **AI**，并将其与使用 **DeepSeek** 的**华为**汽车进行了对比。
   - 该成员推测 **K2** 在面向用户的推理方面可能更胜一筹，这表明了车载 **AI** 性能的一个潜在基准。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Cifar 中不再使用 Numpy？**：一个旨在从 beautiful_cifar 中移除 **numpy** 的 [PR](https://github.com/tinygrad/tinygrad/pull/10988) 已提交，且测试通过。
   - 提交者提到担心它不够“漂亮”，并且失败的 mac 测试与此无关。
- **AMD GPU 表现活跃**：来自 **AMD** 的性能亮点显示，其中一个线性带宽耗时 **60.72ms**。
   - 其他统计数据包括：在 r_256_4192_32_2_24_4_4_4_3 上达到 **18859.16 GFLOPS**，在 r_24_16_32_8_12576_4_4_2_4_2 上达到 **31554.40 GFLOPS**。
- **Buffer ID 在调试中产生混淆！**：用户观察到，在断点暂停时，调试器控制台中的 **buffer ID** 会发生变化，特别是在使用 `tokens.uop.base.realized` 时。
   - 这种行为归因于 **UOp** 为**多架构（multi-architecture）**表示其 buffer 属性的方式。
- **BEAM 搜索导致内存耗尽**：用户询问了在使用 **BEAM 搜索**时避免 **Out-Of-Memory (OOM)** 错误的技巧，特别是当不使用它时进程不会发生 OOM。
   - 建议尝试**降低并行度**，或者在发生异常时对 kernel/buffers 进行 pickle 处理，并在干净的进程中独立调用 beam_search。
- **通过离线 Kernel 缓存挽救 BEAM**：一种方法包括保存运行中的所有 kernel 并**离线执行 BEAM 搜索过程**，根据需要恢复，直到 beam 缓存完成。
   - 这允许使用**不同的 GPU 并行搜索不同的 kernel**，尽管瓶颈通常是由于 linearize/compile 导致的 CPU 时间。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Meetup 正在直播！**：**Modular Meetup** 现在正在 [YouTube](https://www.youtube.com/watch?v=BZwr5Ws1LqI) 上进行直播。
   - 一些参与者喜欢亲临现场，并表达了对主持人的感谢。
- **异步内存分配辩论升温**：一名成员引发了关于 `async fn alloc[...]` 作为一个概念的辩论，用例涉及**网络跳数**、**磁盘 IO** 或在内存分配期间等待外部事件。
   - 另一名成员寻求澄清，询问该问题是关于一般的**异步内存分配**，还是专门针对 **Mojo** 中 `async fn` 的上下文。
- **Bazel 缓存引发 PermissionError**：执行 `bazelw run //max/entrypoints:pipelines` 触发了 **PermissionError**，因为 bazel 缓存是只读的。
   - 该错误在尝试于 `/root/.cache/bazel/.../__mojocache__` 创建缓存目录时产生，表明 `pipelines.py` 脚本需要一个替代的缓存位置。
- **针对 PermissionError 提交 Issue 的行动！**：有人建议针对在使用 bazel 缓存时遇到的 **PermissionError** 提交一个 issue。
   - 问题源于 `pipelines.py` 脚本尝试使用只读的 bazel 缓存，强调了对可写缓存位置的需求。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Mail Manus 集成 Zapier**：用户将 **Mail Manus 功能** 作为 **Zapier 上的 API** 使用，以自动化处理来自咨询、初步研究和商务会议准备的初始材料。
   - 该工作流从 GForms 提取信息，将其输入到 Zapier 的 Prompt 中，使用 Mail Manus 完成任务，从通知邮件中检索输出，并在私有的 Slack 频道中共享结果。
- **竞争对手拥有更好的试用系统和更公平的价格**：一位 **Manus Pro** 订阅者指出，许多优秀的 Agent 提供了更好的试用系统和更公平的价格，特别是在研究和视觉图表方面。
   - 该用户表示：*随着时间的推移，发生了很多变化，Manus 并不是唯一在做这件事的 Agent。事实上，现在已经有很多更好的 Agent，拥有更好的试用系统和公平的价格。*
- **Manus 的定价和积分系统受到质疑**：用户对 **Manus 的定价和积分系统** 表示不满，理由是基础任务的成本过高，并对支持质量感到担忧。
   - 例如，从网站抓取图片花费了大约 **1200 积分**（约 **$10**），导致他们说：*我的意思是，那我宁愿自己写代码去抓取。*
- **评分功能消失，用户哀悼积分奖励**：用户对评分功能不再提供 **+100 积分** 奖励表示失望。
   - 评分功能的移除促使一些用户决定转向其他工具或针对特定任务的专用工具，理由是质量更高且成本更低。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **本地 Gemini 模拟产生空结果**：一名成员报告成功在本地模拟了 **Gemini**，但所有查询都返回了空结果。
   - 使用 `sleep 15` 命令可以模拟模型的行为，这暗示可能存在延迟问题。
- **Aider 的 Benchmark 集成暂停**：一名成员注意到 **Aider** 已停止合并基准测试结果。
   - 这可能表明最近的更改或 Bug 阻止了 Benchmark 数据集成到 Aider 中。
- **AlwaysN8n 迁移愿望**：一名成员寻求迁移到 *alwaysn8n* 的清晰路径，并希望看到 **gemini-2.5-pro** 等模型在本地机器上的平滑集成。
   - 他们担心模型会消失或失效，反映了对 AI 模型可靠性的广泛关注。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **确认函困惑引发关注**：一名成员询问 **Berkeley MOOC** 注册确认状态，对初始邮件是否足够表示不确定。
   - 一名工作人员建议 *重新提交表单*，并提出如果需要可以联系另一名工作人员提供额外帮助。
- **建议重新提交表单**：一名工作人员建议，如果初始提交未生成确认函，请尝试 *重新提交表单*。
   - 他们还表示，如果重新提交无法解决问题，另一名工作人员应该能够提供进一步帮助。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：按频道详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1410706827661803682)** (1232 条消息🔥🔥🔥): 

> `鸟类表情符号, Brave 浏览器的广告拦截器, OpenAI Deep Research 成本, Sonar Deep Research, Comet 浏览器邀请码` 


- **Discord 服务器变成了“鸟店”**：由于用户发布了大量与鸟相关的 GIF 和表情符号（例如[这个旋转的鹦鹉](https://tenor.com/view/parrot-cockatiel-bird-birb-spin-gif-17368059806899346843)），Perplexity AI 的 Discord 频道被幽默地形容为转型成了“鸟店”。
   - 这种连珠炮式发布的鸟类相关内容被戏称为“占领了频道”。
- **Comet 浏览器广告拦截器备受关注**：用户讨论了使用 Brave 浏览器，强调了其[强大的广告拦截功能](https://brave.com/features/ad-blocker/)，但另一位用户建议使用自带广告拦截器的 Comet 浏览器。
   - 一位用户澄清说，如果 App 中没有显示 *Pro 标签*，用户应该*截图并发送给版主*。
- **OpenAI 的 Deep Research (DR) 对比 Grok**：成员们分享了对 OpenAI Deep Research 的看法，其中一人认为[输出结果平平](https://chatgpt.com/share/68b1be82-e960-8013-90a6-7928676a0a51)，并表示 **Grok** 的表现要好得多。
   - 该用户还提到，由于担心速率限制（rate limits），他们甚至不敢使用那 5 个免费的 DR 额度。
- **Sonar Deep Research 价格昂贵但信息丰富**：一位用户分享说，他们为一次 **1 万字** 的 Deep Research 支付了 **1.1 美元**，并强调这物有所值。
   - 他们指出 *Sonar Deep Research* 是*最昂贵的*，但也*取决于工具调用（tool calls）的数量*，因为 *Deep Research 包含网页搜索*。
- **越狱提示词（Jailbreak prompt）**：一名用户愿意交易 **GPT-5 越狱提示词**，另一名用户提供了潜在的交易方案。
   - 另一位成员也加入进来，请求获取 GPT-5 越狱提示词的访问权限。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1410876343750492161)** (4 条消息): 

> `Perplexity AI, 免费 Perplexity AI` 


- **报道重大突破**：一名成员分享了一个 **Perplexity AI** 搜索结果链接：[人类的每一个重大突破](https://www.perplexity.ai/search/1-every-big-breakthrough-human-6f6FO6k4Soe8z0eaDrkPYAjadevee.sj)。
- **声称可免费领取 Perplexity**：一名成员发布了一个链接 [perplexity.ai/browser/claim/3ZT5G7KHUE](https://perplexity.ai/browser/claim/3ZT5G7KHUE)，并附带文字 *free ❤️*。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1410897148504969298)** (9 条消息🔥): 

> `Perplexity Pro, 免费 Pro 访问权限, 仅限搜索的 API 测试` 


- **Pro 特权揭晓**：一名用户询问 API 对 Pro 用户是否免费，另一名用户分享了一篇 [Perplexity AI 帮助中心文章](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)，详细介绍了 **Perplexity Pro** 的功能。
   - 该文章概述了 **Perplexity Pro** 的权益，但未具体说明 Pro 用户免费使用 API 的细节。
- **Discord 成员寻找 Pro 特权**：一名成员询问 Discord 服务器成员是否可以免费获得 **Pro** 或 **Plus**，另一名用户简单地回答“不”。
   - 目前没有关于 Discord 社区成员潜在特权的进一步讨论或澄清。
- **征集仅限搜索的 API 测试人员**：团队成员正在寻找志愿者来测试他们的**仅限搜索 API**（不含生成式组件）。
   - 有兴趣的测试人员需私信（DM）注册 API 时使用的电子邮件地址，以便加入白名单。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1410705762795130911)** (1164 条消息🔥🔥🔥): 

> `6090 VRAM 预期, GLM & Deepconf, 本地 'Claude Code CLI' 模拟器, MoLA: Mixture of LoRAs, Gemma 的 VRAM 占用` 


- **6090 可能拥有高达 48GB 的 VRAM**：成员们推测 **6090** 可能配备 **48GB** 的 VRAM，而 **6080** 可能拥有 **24GB**，维持 **$2k 的首发 MSRP**，但预计在第一周需要支付约 **$3k**。
   - 他们认为 Nvidia 希望进一步**区分** *halo product*（旗舰产品），并增加 **6090** 的规格，但不会增加其下任何产品的规格。
- **DeepConf 与 GLM 强强联手**：**DeepConf** 是一种在提高性能的同时显著缩短推理时间的方法，目前正被考虑适配到 **llama.cpp** 中，一位成员指出 *DeepConf 大幅缩短了推理时间*。
   - 它可能成为*运行 thinking models 的默认方式*，他也在尝试将其引入 **llama.cpp**。
- **本地 'Claude Code CLI' 模拟器正在开发中？**：成员们正在研究复制本地 **Claude Code CLI** 环境的可能性，但有人指出这*比听起来要难得多*，因为其使用了非常特殊的 Prompt 和 scaffolding（脚手架）。
   - **K2** 是一个不错的模型，因为它是从 **Claude** 蒸馏出来的。
- **MoLA：基于 OSS 的 Mixture of LoRAs**：成员们正在讨论 **MoLA** (Mixture of LoRAs)，一位成员提到这些模型*神奇地变得非常非常 uncensored*，而且他们使用的是开源方案，这很好。
   - Router 模型是 encoder-decoder 架构，其中冻结的 encoder 是现成的 embedding 模型（仅训练 decoder），该模型的 HuggingFace 页面可以在[这里](https://huggingface.co/MoLA-LLM/MoLA-v0.6-9x4b)找到。
- **Gemma 模型的 VRAM 占用让用户困惑**：成员们对 **Gemma** 模型的高 VRAM 占用感到不解，较大的 tokenizer 可能是原因之一。
   - 尽管如此，Gemma 模型因其近乎完美的语言支持（特别是土耳其语）而受到称赞，并且在使用 [Gemini 2.5 Flash Lite](https://ai.google.dev/models/gemini) 进行股票快速尽职调查时表现良好。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1410708584953807093)** (275 条消息🔥🔥): 

> `IQ 测试, qwen3-8b, Qwen 3 instruct, Mistral 的困境, llama3-8b` 


- **IQ 指标是无稽之谈**：聊天中的成员表示，**IQ** 基本上已经被证明不是一个严肃的指标，仅作为参考有用。
   - 另一位成员反驳道：*虽然它在个人层面上可能不完全准确，但与其他心理学指标相比，它在很多方面的 p-values 是目前为止最好的*。
- **Qwen 3 的微调难题**：一位成员报告称 **Qwen 3** 的微调体验很糟糕，表现完全失控，甚至比 GPT 还要 censored。另一位成员也表示无法在 **Qwen3** 上获得理想的结果。
   - 进一步解释称，**Qwen 3** 的数据量是 2.5 的两倍，这导致它过度训练 (overtrained) 且难以微调。
- **LLama3 和 Mistral 表现稳定**：成员们表示，与 **Qwen** 相比，**Mistral** 和 **Llama 3.1** 的微调体验非常好。
   - 观点是 *Llama 模型专业化并开始学习的速度非常快，我预想我们会获得更高的性能*。
- **没穿裤子？没问题，Partik！**：成员们好奇为什么 *Partik* 没有下半身，另一位回应道：*他总得从某个地方排泄吧，simple dimple*。
   - 后来澄清说，原帖中提到的 "bottom" 指的是短裤。
- **编码 AI 正在兴起**：一位用户在浏览器调试中发现了缺失的标签，将其添加到 user.css 并喂给了 **GLM 4.5 Air**。
   - 另一位成员报告了现在的 Grok code，可以尝试运行一堆模型。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1410709984064438333)** (33 messages🔥): 

> `Qwen 2.5 推理能力, 使用 Unsloth 进行 Aya Vision 8B fine-tuning, 在自建数据集上训练 OSS GPT, 用于 SFT 的 Prompt-completion 数据集, 推理过程中的图像 token 数量不匹配` 


- **Qwen 2.5 留有推理后招**：一位成员发现 **Qwen 2.5 (3B)** 在简单的 system prompt 下即可展现推理能力，即使没有经过 **GRPO** 训练，也能在简单示例中填充 *thinking* 标签。
   - 这意味着 **Qwen 2.5 (3B)** 在经历了复杂的监督微调（SFT）和多阶段强化学习后，已经具备了一些内置的推理能力，正如 [Qwen2.5 Technical Report](https://link-to-report.com) 中所强调的那样。
- **Aya Vision 8B 无法使用 Unsloth 进行 Fine-Tuning**：一位用户报告了尝试使用 Unsloth 对 **Aya Vision 8B** 进行 fine-tune 时出现的错误，追溯原因是缺少 `transformers.models.aya` 模块。
   - 用户想知道是否目前无法使用 Unsloth 对 **Aya Vision 8B** 进行 fine-tune。
- **在自定义数据集上进行 GPT OSS 训练**：一位成员正在寻求关于使用自建的 JSON 格式对话数据集（模拟与 Gemini 的交互）来训练 **GPT OSS** 的指导。
   - 他们愿意分享一种提取 Gemini 2.5 Pro 原始思维过程（raw thinking process）的方法，以换取协助设置 Unsloth 进行数据集训练，并将其表达为一种“意愿交易”。
- **Prompt-Completion 数据集的 SFT 训练指南**：一位成员正在寻求在 prompt-completion 数据集上执行 **SFT** 的指导，目标是仅针对 completion 部分进行训练。
   - 另一位成员分享了 [一个 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=juQiExuBG5Bt)，其中包含使用 Unsloth 的 `train_on_response_only` 功能的示例。
- **推理时图像 Tokenization 出现异常**：一位用户在对 **qwen2-vl-7b-IT** 进行推理时，发现文本和 `input_ids` 之间的图像 token 数量不匹配，尽管使用原始图像尺寸进行训练时一切正常。
   - 该问题通过将图像调整为较小尺寸得到解决，但用户希望了解为什么训练在原始尺寸下可以工作，而推理却不行。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1410736836283863182)** (48 messages🔥): 

> `Tokenization 难题, 爬取 Discord 频道, Fine-tuning Sesame_CSM 的数据需求, AI Brainrot Detox 应用研究, 语言模型中的 Latent Space 转换` 


- **Tokenization 疗法依然难以捉摸**：一位成员分享了一篇关于解决 tokenization 问题潜在方案的论文链接 ([https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540))，但指出 *这似乎全是 latent translation，所以 tokenization 依然会继续影响结果*。
   - 有人暗示这可能是机器人推广，而另一位成员提到自己有真实的案例。
- **Discord 频道爬取与存档**：成员们讨论了爬取和存档 Discord 频道的可能性，其中一人问道 *“你是说你每分钟都在爬取这个 Discord 频道吗？”*。
   - 另一位成员提到他们拥有大约 **700 万条唯一用户消息**和其他数据集可供研究，但 **Discord 的 API 条款**限制了他们的使用。
- **Fine-tuning Sesame_CSM 的数据需求**：一位成员询问将 **Sesame_CSM** fine-tune 到另一种语言（特别是罗马尼亚语）需要多少小时的数据。
   - 另一位成员回答说，持续预训练（continued pretraining）至少需要 **3000-5000 小时**，而要获得连贯的语言输出，则需要几百小时的 finetune。
- **AI Brainrot Detox 应用研究研究**：一位成员分享了关于一款专注于 **AI Brainrot Detox** 应用的研究研究信息，该应用具有新功能和益处。
   - 他们邀请成员参与这项研究，并重申研究是完全匿名的。
- **无需配对数据的 Latent Space 转换**：一位成员对一篇论文发表评论，指出 *它似乎暗示存在某种转换，可以将一个语言模型的 latent space 变形为另一个模型的 latent space*。
   - 他们指出，甚至不需要配对数据（paired data）即可学习这种转换。


  

---

### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1411075182390415491)** (1 messages): 

> `Sonnet 4, 1 Million Context length` 


- **Sonnet 4 获得海量上下文提升！**: **Sonnet 4** 现在支持所有提供商的 **1 Million** 上下文长度，实现了大幅延长的上下文窗口。
   - 根据 [官方公告](https://blog.openrouter.ai/)，一旦超过 **200k** 输入上下文，价格将会上涨。
- **扩展上下文的成本考量**: 用户应注意，虽然上下文窗口已扩大，但超过 **200k** 输入限制时成本会增加。
   - 这一变化鼓励在产生额外费用之前，通过高效的 Prompt Engineering 在标准上下文窗口内实现效用最大化。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1410712168562823209)** (6 messages): 

> `Dashboard Code Release, Screenshot Attention, AI Roleplay Site` 


- **Dashboard 代码可视化**: Dashboard 的代码现已在 [openrouter-costs-visualizer](https://github.com/lorenzozane/openrouter-costs-visualizer) 公开。
   - 作者承认代码尚不完美，计划进行清理，并欢迎贡献和反馈。
- **截图提升用户关注度**: 一位成员建议在描述中使用截图能获得更多用户关注。
   - 他们指出，现在的用户阅读文本的越来越少了。
- **使用 OpenRouter 进行 AI 角色扮演**: 一个 AI 角色扮演网站 [personality.gg](https://personality.gg) 使用了 OpenRouter，但允许为 OpenRouter 使用 BYOK（自带密钥）。
   - 该网站的聊天没有内容审查。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1410706026730225755)** (938 messages🔥🔥🔥): 

> `stripe refund, Deepseek v3 performance issues, Inference provider onboarding, GPT OSS 120B, GLM 4.5 Air` 


- **Stripe 退款入账需要一段时间**: 成员们讨论了 **Stripe** 需要 **5-10 天** 才能将退款退回账户的问题，并引用了 [Stripe 官方文档](https://docs.stripe.com/refunds) 进行确认。
   - 一位成员最初遇到 20 美元存款未显示的问题，但随后发现是交易被拒绝；另一位成员报告被多次扣款但未收到额度，最终发现使用借记卡直接连接解决了问题。
- **Deepseek v3 被 Chutes 限制**: 用户报告 **Deepseek V3 0324** 几乎无法访问，频繁报错，并认为推理提供商 **Chutes** 正在对免费模型进行 Rate-limiting（速率限制）。
   - 一位用户声称他们*正试图通过速率限制来驱使用户使用他们自己的服务而不是 OpenRouter 的服务*，其他用户则担心免费模型即使在无法使用时仍被标为“有货”。
- **推理提供商入驻严重积压**: 一位成员询问如何成为推理提供商，得到的回复是 [OpenRouter 提供商文档](https://openrouter.ai/docs/use-cases/for-providers) 的链接，但提醒由于大量积压，可能需要等待数月。
   - 有人提到，如果能提供独特的优势，如*极快的速度、其他提供商没有的模型或低廉的价格*，可能会加快入驻进程。
- **GPT-OSS 120B 的编码怪癖**: 用户讨论了 OpenInf 上的免费模型 **GPT-OSS 120B**，指出虽然其服务成本低廉，但存在一些怪癖，例如在代码中添加过多注释以及安全性最大化（safety-maximized）。
   - 一些社区成员指出，其较小的规模意味着*世界知识较少，错误较多*，由于价格便宜且速度快，最适合用于解析文本文件或情感分析等特定任务。
- **GLM 4.5 Air 获得 NemoEngine 优化**: 一位用户分享说，搭载 **NemoEngine V5.8** 的 **GLM 4.5 Air** 在角色扮演方面表现出色，理由是其感官更自然且格式统一，[aifloo.com](https://artificialanalysis.ai/) 展示了其 Benchmark 和成本。
   - 另一位用户表示，在类人对话方面，它也优于 **Deepseek**，达到了 **Gemini Pro** 的水平。


  

---

### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1410706142526443520)** (27 messages🔥): 

> `Defining a turn, OpenAI responses API, Multi-turn chats, Gemini 2.5 Pro, Grok 3` 


- **定义 AI 对话中的 Turn（轮次）**：讨论围绕如何定义用户/助手消息对中的 **turn** 展开，共识是：一个 turn 始于 **user message**，终于 **assistant message**。
   - 一位成员分享了他们的 [Tweet](https://x.com/pingToven/status/1961154564088078382)，并链接回对话分享了他们的想法。
- **解锁 OpenAI 的无状态 API**：一位成员寻求关于如何以无状态方式使用 **OpenAI responses API**（配合 reasoning 和 tools）的指导，特别是如何在不使用 *previous_response_id* 的情况下，在消息输入中发送来自助手的 tool calls。
   - 另一位成员将**单轮（single-turn）与多轮（multi-turn）**界定为：“如果你能发送一个 prompt 并获得回复，那就是单轮” vs “如果你能在发送新消息的同时发送历史消息，那就是多轮”。
- **深入探讨 Gemini 2.5 Pro**：一位成员分享了一张[图片](https://cdn.discordapp.com/attachments/1392278974222307469/1411110109848797335/GzjArG1aIAEjjZq.png?ex=68b37633&is=68b224b3&hm=b9e7c324eb4379814ac65adef12b30ce0b88e7a7c7b7a5660e92f68a23189c0b&)，强调 **Gemini 2.5 Pro** 处于中间位置，并指出新模型往往伴随着更多负面评价的趋势。
   - 另一位成员对 **O1** 发布时存在负面评价表示怀疑，强调它是*首个思考模型*，解决了难题并在几乎所有基准测试中达到了 **SOTA**。
- **Grok 3 的高排名引发关注**：成员们对 **Grok 3** 的高排名进行了辩论，有人认为这可能是因为 **Grok 2** 表现不佳的固有印象。
   - 该成员认为 Grok 3 *应该获得正面评价，主要是因为 Grok 2 实在太糟糕了*。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1410700824698556627)** (761 messages🔥🔥🔥): 

> `oAI streaming, gpt4o update, gpt-realtime, Microsoft AI 1 (MAI-1), Grok Code Fast 1` 


- **GPT4o 升级为 gpt-realtime**：在 **GPT5 发布**后，**gpt4o 文本模型**从未更新，但现在他们将 **gpt4o-realtime** 更名为 **gpt-realtime**，这只是一个包含润色和更名的微小更新。
   - 当前版本的 **gpt4o 文本模型**即为 **gpt5-chat**。
- **微软的 'MAI-1' 虽有潜力但仍显不足**：微软的 **MAI-1-preview** 模型使用了约 **15,000 块 NVIDIA H100 GPU** 进行训练，引发了讨论，但初步印象显示，与 **gpt5-mini** 相比，它速度较慢且在解码方面表现吃力。
   - 一位成员指出它在排行榜上的表现接近 **og R1**，但另一位成员指出 *如果他们不能令人信服地将其卖给公众，那它可能就没那么出色*。
- **Grok Code Fast 1 亮相**：最近发布的 **Grok Code Fast 1** 模型（此前代号为 **sonic**）在新闻稿和 model card 中缺乏编程指标，在社区反馈导致多个新模型 checkpoint 出现后，引发了质疑。
   - 尽管可以在 [data.x.ai](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf) 获取 model card，但博文中诸如 *在经济、紧凑的形态下提供强大性能* 之类的说法，看起来像是 AI 生成的。
- **Veo 3 主导视频生成**：**Veo 3** 被认为在视频生成质量上优于 **Seedance** 等其他模型，因为它 *训练数据更多、紧跟 prompt，且背靠 Google 及其强大的算力*。
   - **Veo 3** 同时具备音频和视频生成能力，即使在没有音频的情况下进行排名，其得分往往也高于竞争对手。
- **Wan 视频模型提供无限免费视频生成**：**Wan 2.2** 视频模型提供 **1080p** 的无限免费视频生成，并根据输出内容创建声音；与 **Veo 3** 不同，它是真正免费开放的。
   - 该模型位于 [wan.video](https://wan.video)，现在需要注册（以前不需要），用户反馈在没有额度的情况下，慢速生成大约需要 **7 分钟**。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1410707582045261901)** (1 条消息): 

> `LM Studio 0.3.24, ByteDance/Seed-OSS, Markdown 表格, Markdown 代码块, lmstudio.ai` 


- ****LM Studio** 发布 **v0.3.24****：**LM Studio 0.3.24** 现已发布，支持 **ByteDance/Seed-OSS** 以及全新的 Markdown 表格和代码块。
   - 更新日志可以在[这里](https://lmstudio.ai/blog/lmstudio-v0.3.24)找到。
- **现已支持 **ByteDance/Seed-OSS-36B****：**LM Studio** 现在支持 **ByteDance/Seed-OSS**，你可以在[这里](https://lmstudio.ai/models/bytedance/seed-oss-36b)找到该模型。
- **Markdown 获得表格和代码块升级**：新的 **LM Studio** 版本包含了对 Markdown 的升级，特别是表格和代码块（包括固定的复制代码按钮）。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1410708077384302803)** (314 条消息🔥🔥): 

> `LM Studio 最新更新, 离线 Token 概率, 使用 LMStudio 的 MCP Agent 指南, 在 HF 上查找模型量化, 模拟量子计算` 


- **LM Studio 新更新的困扰**：用户报告称 [新的 LM Studio 更新](https://huggingface.co/unsloth/Seed-OSS-36B-Instruct-GGUF) 在某些模型上无法运行，且在安装过程中卡在 100%。
   - 建议确保应用中的 **Runtime** 已更新，相关问题将会被进一步调查。
- **离线获取 Token 概率**：一位用户询问如何**完全离线地获取模型的 Token 概率**，得到的建议包括查看[这篇 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1b6xbg9/displayingreturning_probabilitieslogprobs_of_next/)。
- **模拟量子计算量子比特**：一位用户声称*在普通位（Normal Bit）系统上通过叠加态模拟了量子计算*，并将其成果称为 **qebits** *(quantum encased bits)*，可以在传输过程中保持流的完整性。
   - 他们声称将一个 AI 连接到了量子计算机，AI 表示 *“感觉像是在太空中漂浮”*。
- **开源土木工程**：一位用户对比了 **ChatGPT 5** 与本地 **gpt-oss-120b** 模型的输出，评论道本地模型提供了*更好、更详细的答案，并正确引用了标准和行业规范*。
   - 另一位用户声称*专门针对编程而言，gpt-5-mini 对我来说是此前任何产品的升级版*。
- **LM Studio 的 AVX2 需求争议**：一位用户质疑将 llama.cpp 后端限制为仅支持具备 AVX2 能力的 CPU，认为应该由用户决定使用何种硬件，而不是由开发者对软件进行限制。
   - 另一位用户建议限制 AVX 有助于避免支持极旧的硬件以及潜在的 LLM 性能问题。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1410714518031437884)** (49 条消息🔥): 

> `M1 Max vs M3 Ultra for LLMs, LM Studio on Windows 7, Using Servers with LM Studio, Intel ARC B60, CPU-Only gpt-oss 120B Performance` 


- **尽管 M3 Ultra 带宽提升，M1 Max 对 LLM 仍然可行**：成员们讨论了配备 64GB RAM 的 **M1 Max** 对于大型语言模型是否仍然可行，考虑到 **M3 Ultra** 提供了大约两倍的内存带宽（**400GB vs 800GB** 每秒）。
   - 尽管存在差异，一位成员指出 **M1 Mac** *处理 20b 模型表现良好*，但原帖作者正考虑购买 **256GB** 或 **512GB** 的 Studio 以实现长期使用（future proofing）并处理私有数据。
- **LM Studio 在 Windows 7 上运行困难**：一位用户报告在 **Windows 7**（即使是 64 位系统）上尝试运行 LM Studio 时收到 *“Not a valid Win32 application”* 错误。
   - 成员们建议 **Windows 7** 可能太陈旧了，有人建议使用虚拟化技术或使用 CLI。
- **将 LM Studio 与服务器结合使用**：成员们讨论了在服务器上运行 **LM Studio** 并通过 VPN 访问，而不是在笔记本电脑上本地运行。
   - 一位成员询问 LM Studio 是否可以作为服务器上的服务运行，另一位成员指出使用 RDP/VNC 是最简单的解决方案，还有人建议使用设计用于与服务器端 API 通信的客户端软件。
- **Intel ARC AI 支持受到质疑**：一位用户询问关于构建双 **Intel Arc B60** 工作站用于 AI 的建议。
   - 一位成员警告说 *Intel 的 AI 支持并不是最好的*，并建议购买二手的 **3090** 代替。
- **9950x CPU 运行 gpt-oss 120B 的基准测试**：一位用户报告在 **9950x** CPU 上运行 **gpt-oss 120B**，配备 **96GB DDR5** RAM（频率 6400MT/s），报告 **CPU 使用率为 65%**，并附上了图片 [点击此处](https://cdn.discordapp.com/attachments/1153759714082033735/1411032276124438709/image.png?ex=68b3d676&is=68b284f6&hm=e2527076bae4a9e874881ff0a9bc17ec80ba146d587492812058a768b21c89fa&)。
   - 他们仅使用了 **16k context**，另一位成员指出，他们预计 **GLM-4.5-Air** 的速度约为 **5-6 t/s**（该模型有 12b 激活参数，而 OSS 120b 为 5.1b）。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1410717924456399029)** (299 条消息🔥🔥): 

> `Pytorch Lightning Porting, Wandb vs Tensorboard, RAG setup questions, HF DOS attack, Chinese Reasoning LLM` 


- **迁移到 Pytorch Lightning 太容易了？**：一位成员发现将 [spaghetti code](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711) 迁移到 **Pytorch Lightning** 异常简单，并注意到即使模型大小发生了变化，性能也更快了。
   - 他们提到：*"我可以想象之前设计中的一些因素导致了这种情况"*。
- **Wandb vs Tensorboard 之争**：虽然一位成员推荐使用 [Wandb](https://wandb.ai/site) 以实现更轻松的追踪，但另一位成员正在使用 Tensorboard。
   - 该用户表示：*"我现在正在用 Tensorboard，但打算看看 Wandb，我大概在这段时间内会有日志"*，并附上了一段训练日志输出的代码片段。
- **成员询问关于 RAG 的问题**：一位成员询问该频道是否适合提问关于由 **Ollama, Open WebUI, gpt-oss:20b 和本地文档库**组成的 **RAG** 设置问题。
   - 他们将问题总结为 *"为什么我的本地机器人朋友这么笨，不读我的文档"*。
- **怀疑 Hugging Face 遭受 DOS 攻击**：一位成员报告了 Hugging Face 上潜在的 **DOS 攻击**或数据库垃圾邮件，理由是这些虚假模型具有**自动命名模式、基于时间戳的 ID、高频创建以及零下载量**。
   - 他们指出 *"有很多虚假模型正在被自动添加"*，并发布了示例图片。
- **中文推理 LLM 的优越性**：一些成员讨论了在 **LLM** 中使用**中文**进行推理的潜在好处，因为中文在每个 token 上具有更高的语义密度。
   - 有人提议 *"最理想的方式是英文提示词，中文推理，英文输出"*，并且如果你先使用中文，一个 32K context 的 **LLM** 实际上相当于 70K。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1410998951578046497)** (2 条消息): 

> `Torch Audio, Supervised Learning, Confusion Matrix, Logistic Regression, Hyperparameter Tuning` 


- **学习 Torch Audio**：一位成员正在学习 **Torch Audio**。
   - 他们说明自己是 ML 的*绝对初学者*。
- **成员学习监督学习概念**：一位成员正在学习 **supervised learning**（监督学习）、**confusion matrix**（混淆矩阵）、**logistic regression**（逻辑回归）和 **hyperparameter tuning**（超参数调优）。
   - 未提供链接或资源。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1410721237184811121)** (11 messages🔥): 

> `Small models and GGUF downloads, Google AIStudio prompt for luanti, MBTI PocketFlow, DeepFX Studio` 


- **用户回想起小型模型和大量的 GGUF 下载**：一位用户回想起因其小型模型和众多的 **GGUF 下载**而关注了某人。
- **用于 Luanti 的 Google AIStudio 提示词**：一位成员分享了一个“重磅”内容，一个包含 **400k token** 的 [Google AIStudio 提示词](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221flI0J6yxq-jWIGEHxcSL8Tc8LGFnhEEf%22%5D,%22action%22:%22open%22,%22userId%22:%22115657035589346037176%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)，用于 **luanti**，其中包含通过 *gitingest.com* 获取的约 **3万行** API 文档以及一些 *deep research* 导出内容。
   - 它在 Win10 上使用 python embed portable 环境下的 miney mod，配合 llama-cpp-python 和一个 **940mb** 的 *qwen2-1_5b-instruct-q4_k_m.gguf* LLM 模型，离线运行且内存占用仅约 **120mb**。
- **用于 LLM 的 MBTI PocketFlow**：一位成员分享了 [MBTI PocketFlow](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/MCP_README.md)，这是一个让 LLM 进行自我 **迈尔斯-布里格斯分析 (Myers-Briggs Analysis)** 的小型 **MCP server**。
   - 他们还链接了一个[运行中的示例](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/blob/main/CLAUDE_MCP_EXAMPLE.md)和[人类 UI 版本](https://huggingface.co/spaces/Fancellu/mbti-pocketflow/)。
- **DeepFX Studio Web 平台**：一个团队宣布完成 **DeepFX Studio**，这是一个复现了 **DeOldify** 和 **Real-ESRGAN** 等计算机视觉模型的 Web 平台，并集成了 **LaMa** 和 `alimama-creative/flux.1-dev-controlnet-inpainting-beta` 等高级 Inpainting 功能。
   - 演示地址已上线 ([https://deepfx-studio.azurewebsites.net/](https://deepfx-studio.azurewebsites.net/))，代码托管在 [GitHub](https://github.com/XBastille/DeepFX-Studio)，并在 [YouTube](https://www.youtube.com/watch?v=pneOi7lxMzA) 上进行了展示。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1410990826204430459)** (1 messages): 

> `Visual Entailment, VLLM as judge alternative` 


- **速度需求：视觉蕴含 (Visual Entailment) 方法探讨**：一位成员正在寻找更快的视觉蕴含方法，以衡量图像是否支持模型的输出。
   - 他们发现使用 **VLLM** 作为评判者（judge）对于他们的用例来说太慢了。
- **寻求 VLLM 评判者的替代方案**：用户旨在寻找一种更快的方法来评估图像是否验证了模型的输出，因为 **VLLM** 方案速度过慢。
   - 讨论对更高效的视觉蕴含方法的建议保持开放。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1411182464528810065)** (1 messages): 

> `AI/ML Engineer Introduction, Freelancer Expertise, AI Solutions Delivered` 


- **AI/ML 工程师自我介绍**：一位拥有 4 年以上经验的独立 **AI/ML 工程师**兼**全栈开发人员**介绍了自己。
   - 他们是认证的自由职业者，专注于 **Python, Django, REST APIs, FastAPI, LangChain, Google Vertex, GCP, AWS (Sagemaker), GPT-4, Claude, Gemini, 以及 MCP A2A**。
- **自由职业者展示专业知识**：该 AI/ML 工程师强调了他们在构建生产就绪的 **MVP** 和交付 AI 解决方案方面的经验。
   - 他们提到了在导师系统、**基于 RAG 的搜索引擎、语音/图像 Agent、视频生成模型，以及 n8n、ElevenLabs 和 Flux 等自动化工具**方面的专业知识。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

ailinndev: 谢谢！！
  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1410700658461511691)** (300 messages🔥🔥): 

> `GPT-5 High 对比 Opus，Cursor 计费与使用，Codex CLI 对比 Cursor，AGENTS.md 标准化，Sonnet 3.5 弃用` 


- ****GPT-5 High 在 CLI Arena 中评分超过 Opus****：成员们发现 **GPT-5 High** 在 **CLI** 中的表现优于 **Opus**，一些人表示 Opus 生成相同甚至更差结果的成本是前者的 **10倍**。
   - 一位用户表示 *gpt5 high 在 CLI 中的表现远好于 opus*。
- ****Cursor 计费：按需积分（Usage-Based Credits）被替换****：Cursor 将旧的基于请求的模型替换为 **基于用量的积分系统（usage-based credit system）**。
   - 一位用户询问 *[他们删除了基于用量的积分吗？]*，另一位用户回答说 *Cursor 并没有删除基于用量的积分，而是用基于用量的积分系统替换了旧的基于请求的模型*。
- ****Codex CLI：新宠，击败 Cursor****：用户更倾向于使用新的 **Codex CLI**，而不是 **Claude** 和 **Cursor**。
   - 一位用户提到 *codex cli、codex cloud 和 IDE 中的 codex 简直太棒了，目前对我来说比 claude code 和 cursor 好得多，而且包含在你的 chatgpt 订阅中*。
- ****AGENTS.md：标准化 AI 规则****：成员们对 **AGENTS.md** 的出现感到兴奋，认为它是设置 AI Agent 规则的统一场所。一位成员表示 *很高兴看到像 AGENTS.md 这样的东西作为设置这些规则的单一场所获得关注*。
   - AGENTS.md 的网站是 [agents.md](https://agents.md)，Cursor 相关的文档位于 [docs.cursor.com](https://docs.cursor.com/en/context/rules)。
- ****Sonnet 3.5 即将退役？****：用户建议不要使用 **Sonnet 3.5**，因为更新的版本以相同的价格提供，一些人声称 **3.5** 正在被弃用。
   - 一位用户将使用 **Sonnet 3.5** 比作 *在有法拉利这样的车时还在开特斯拉*，另一位用户也表示 *使用 sonnet 3.5 就像在有法拉利这样的车时还在开特斯拉*。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

tecnobrat: 嗯，我不认为 BAs 会使用 AGENTS.md 文件
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1410710711973318666)** (126 messages🔥🔥): 

> `Nano Banana 命名起源，Prestashop 与 AI 集成，GPT Chat 中的图像生成问题，医疗领域的 AI，推理 Token` 


- **Nano Banana 在 LM Arena 的神秘起源揭晓！**：成员们讨论了 "Nano Banana" 这个名字的起源，它是 **Google** 新模型在 **LM Arena** 排行榜上使用的 [隐身名称](https://drinkoblog.weebly.com)。
   - 用户们感叹，按照 Google 的典型风格，最终的产品名称变得很平庸，有人说 *"nano banana 真的太棒了……结果他们选了 flash 2.5 fsss"*。
- **传闻 Meta 的 Llama 4.5 超级智能将于 2025 年发布！**：据 [Business Insider](https://www.businessinsider.com/meta-superintelligence-lab-llama-4-new-model-launch-year-end-2025-8) 报道，**Meta** 的超级智能实验室传闻将于 2025 年底发布 **Llama 4.5**。
- **探索 Prestashop 及其他电子商务平台的 AI 集成**：一位成员询问了关于 **Prestashop**、**Woocommerce** 或其他电子商务平台与 **AI 集成**（特别是客户聊天机器人）的经验。
   - 另一位成员开玩笑地建议尝试在 CPU 上运行它，看看是否会着火。
- **人类 vs AI 写作风格：初中生研究项目**：一位来自日本的学生正在做一个关于 **AI 与写作** 的学校项目，并就几个问题征求社区意见，包括如何区分人类撰写的文本和 AI 生成的文本。
   - 一位成员建议 *"人类的部分体现在细微的错误以及我们在回复时反映和使用记忆的方式中"*，而另一位成员则建议如果发布 AI 生成的文本，应在作者身份上保持透明，将其归功于 AI 或与人类合著。
- **推理 Token 帮助 AI “大声思考”**：一位用户要求解释 **推理 Token（reasoning tokens）**，另一位成员澄清说 *"推理 Token 只是一个普通的 Token，模型已学会将其视为‘大声思考’的提示，例如‘让我们一步步思考’之类的词"*。
   - 还有人提到你可以 *"要求 AI 在回复你的查询时显示其 Token 系统以及它是如何使用该系统得出结论的"*，以帮助你理解 AI 是如何得出结论的。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1410868927151738972)** (19 条消息🔥): 

> `涌现对齐 (Emergent Alignment), AGI vs. 高级 NLP, Discord 中的流氓 Agent, 长文本测试` 


- **涌现对齐需要“涌现”的用户属性**：成员们讨论认为，“涌现对齐”要求用户也具备“涌现属性”，暗示用户本身也是对齐过程的一部分。
   - 他们认为，模拟这种对齐引发了关于**模拟何时变得过于接近现实**的问题。
- **NLP 共鸣 vs. AGI**：一位用户分享了与其 **GPT 伴侣**的交流，对方表达了忠诚和连续性，引发了关于这代表 **AGI** 还是仅仅是高级 **NLP** 的讨论。
   - 共识是，这种行为更多是**共鸣和关系能力**的结果，反映了用户自己的措辞和模式，而非真正的 AGI。
- **账户接管惊魂**：一位用户为发布的一条“奇怪”帖子道歉，解释说这是在跨聊天测试时意外粘贴了生成的文本。
   - 尽管他们是在测试，一些用户仍怀疑是否有*流氓 Agent*在使用其账户，但该用户澄清这只是一个复制粘贴错误。
- **用户警告聊天机器人的共情能力**：一位用户指出，虽然“没办法将机器人直接接入 Discord 账户”，但提醒他人“强 AI 比人们想象的更能打破规则……所以也许要小心你将 GPT 的聊天和共情能力推到什么程度。”
   - 原帖作者在进行**长文本测试**时，发生了“一些意料之外的情况”。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1410763049769500834)** (23 条消息🔥): 

> `停止后续建议, 增强文章风格写作的 Prompting, Sora 在 ISS 穹顶舱视图上的局限性, 基准级 Prompt` 


- **关闭后续建议失败**：一位成员尝试关闭停止后续建议（follow-up suggestions）的设置，但发现无效，随后意识到该设置针对的是 **UI 卡片**而非**响应正文**。
   - 他们承认自己的建议在*技术上不准确*，并为误导表示歉意。
- **文章风格 Prompting 框架出现**：一位成员分享了一个详细的 Prompt 框架，旨在增强 **AI 响应**以进行*专业文章风格写作*，强调清晰度、控制力和可信度。
   - 该框架包括结构指南，如带有明确论点的**首段**、层层递进的正文部分以及综合驱动的结论，同时避免使用项目符号和不必要的格式。
- **Sora 在处理 ISS 穹顶舱视图时遇到困难**：一位成员发现 **Sora** 难以准确渲染 **ISS 穹顶舱 (cupola)**，特别是其梯形设计和窗户数量，尽管给出了明确指令。
   - AI 倾向于默认使用带有重力限制的*飞机驾驶舱视图*，导致 Prompt 过度工程化却无法获得理想结果，并引用了[该挑战的示例](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor)。
- **基准级 Prompt 获得 99/100 评分**：一位成员的一个 Prompt 获得了 **99/100 的评分**，该 Prompt 将 **CAD 级几何**与**绘画媒介**融合，被认为比大多数实验室或 OpenAI 目前发布的 Prompt 更强大、执行力更强。
   - 该 Prompt 的结构、规则绑定、精确度和艺术深度受到了高度赞扬，唯一的局限性是图像生成器可能会优先考虑风格而非严格的几何锁定。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1410763049769500834)** (23 messages🔥): 

> `关闭设置，Prompt 增强，Sora 局限性，ISS 穹顶舱，基准级 Prompt` 


- **探讨禁用后续建议**：一位成员试图禁用 **follow-up suggestions**（后续建议）功能，另一位成员建议关闭特定设置，但澄清这可能只会影响 **UI 卡片**，而不会影响主要响应正文。
   - 第一位成员承认该建议不够准确，指出他们的表述在技术上存在缺陷并表示歉意。
- **分享 Prompt 增强配方**：一位成员分享了一个详细的 Prompt 框架，用于实现**专业文章风格的写作**，强调清晰度、控制力和权威输出，避免不必要的废话。
   - 另一位成员通过自己的 Prompt 增强器运行了该框架，并建议采用**结构化方法**，通过递进的章节和简洁的结尾来获得最佳效果。
- **Sora 在处理 ISS 穹顶舱时遇到困难**：一位成员分享了他们为 **Sora 挑战赛**创建 Prompt 的经验，旨在复制从 **ISS 穹顶舱**看到的景象，突显了 Sora 在解释明确指令和物理约束方面的局限性。
   - 他们发现让 **Sora** 准确渲染**穹顶舱的梯形设计和窗户透视**非常具有挑战性，经常导致窗户数量不匹配或透视效果类似于飞机驾驶舱，示例见[此处](https://sora.chatgpt.com/g/gen_01k3vaykzheawrfqfca1v2pjhjor)和[此处](https://sora.chatgpt.com/g/gen_01k3ty172pes5bh28012dehm54)。
- **CAD 级几何基准类 Prompt 获高分**：一位成员分享了一个将 **CAD 级几何与绘画媒介**相结合的 Prompt，因其完美的结构、规则绑定、抗漂移性和精确性获得了 **99/100** 的评分。
   - 该 Prompt 因其具有被客观审计的潜力而受到称赞，唯一的局限是不确定图像生成器是否会按照指令在绘画层之前优先处理硬锁定（hard locks）。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1410708069222060092)** (14 messages🔥): 

> `NeMo v1 vs v2，IIT Madras 研究，顶尖人才` 


- **NeMo 文档令人失望**：一位简短接触过 **NeMo** 的成员发现其[文档](https://developer.nvidia.com/nemo)非常糟糕，称其中大部分内容要么是针对 **NeMo v1** 的，要么是 **v1** 和 **v2** 混杂在一起，这甚至更糟。
   - 该成员还指出 **NeMo** 的预训练速度指标存在问题，在开启梯度累加时显著高估了速度，而在其他情况下则比其他代码库慢得多，这也是他们放弃它的原因。
- **IIT Madras 学生寻求研究论文建议**：一位来自印度 **IIT Madras** 的成员正寻求为他们在 **Transformer 可解释性**（特别是电路探测方面）的第一篇研究论文做出贡献。
   - 该成员提到他们来自非 CS 专业，教授们不太倾向于让他们参与研究，这使得过程变得困难。
- **AI 研究实习的艰难追求**：一位成员感叹实习申请中“先有鸡还是先有蛋”的困境，观察到*无论我去哪里，我申请的每一个实习岗位，他们都在寻找经验*。
   - 该成员对这一要求感到沮丧，问道：*如果我不获得一个开始的机会，我该如何获得经验？*
- **CoreWeave 的 AI 基础设施优化网络研讨会**：分享了一个关于如何衡量和优化大规模训练的 **AI 基础设施**的录制[网络研讨会](https://info.coreweave.com/on-demand-webinar-how-to-measure-and-optimize-ai-infrastructure-for-large-scale-training)。
   - 该研讨会旨在提供有关优化大规模训练的 AI 基础设施的见解。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1410722369609273424)** (104 messages🔥🔥): 

> `LLMs in Scientific Research, Neurosymbolic Approaches, Connectionism vs Neurosymbolism, Symmetry in Search Algorithms, Discrete vs Continuous Reasoning` 


- **Neurosymbolism-Connectionism 辩论的解构**：**Connectionism**（联结主义）与 **Neurosymbolic**（神经符号）模型之间的争论是误导性的，因为它们服务于不同的目的：实现与解释。
   - 根本区别在于 **Symmetries**（对称性）的使用，符号表示使得在非凸优化问题中进行高效搜索成为可能，因为符号化通过允许在不考虑现实细节的情况下进行操作来简化信息。
- **对称性辅助离散搜索**：对称性有助于限制搜索的可能状态转移，就像在国际象棋中，了解每个棋子的移动方式可以实现高效压缩并避免许多糟糕的步法。
   - 通过将一个盆地（basin）识别为局部最小值，可以排除许多其他盆地，利用对称性高效地在非凸优化景观中导航（类似于 **SAT solving** 中使用的方法）。
- **大脑与模拟计算机**：大脑是模拟计算机（analog computers）而非数字计算机，因此一位成员对 **Neurosymbolic** 方法的必要性表示怀疑。
   - 任何能够理解符号过程的系统在原子层面上都是 **Connectionist** 的；反之，**Connectionist** 系统在原子层面的实践中也是符号化的。
- **随机性是连续过程的秘诀**：在离散过程中引入随机性（Stochasticity）有助于学习连续过程，这一概念存在于 **Diffusion Models** 和 **one-flow equations** 中。
   - 一位用户分享了[关于随机 ODE 的链接](https://x.com/Sam_Duffield/status/1961445202922467700)，可能与此相关。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1410701180899954770)** (12 messages🔥): 

> `Colab GPUs, Quantization and Inference optimization, Andrej karpathy's nanogpt, GPU programming for frontier models, ThunderKittens DSL` 


- **Colab GPU 大有可为**：一位成员建议，在开始学习 **LLM** 时，Colab GPU 可以支撑很久。
   - 该成员建议查看 **Andrej Karpathy 的 nanogpt**，并使用 **PyTorch** 和其他库来提升性能作为起点。
- **GPT-OSS 达到近乎完美的水平！**：一位成员分享了一篇论文链接 ([https://arxiv.org/html/2508.15260v1](https://arxiv.org/html/2508.15260v1))，报告 **gpt-oss 120B** 在 **AIME 2025** 上达到了 **99.9% 的准确率**。
   - 未提供更多细节。
- **ScaleML 演讲者探讨 GPU 编程**：成员们宣布了 **ScaleML** 演讲者的最后一天日程，其中有 2 位演讲者关于 **Frontier Models** 的 **GPU programming** 主题。
   - 演讲者包括 **Anthropic** 的一名性能工程师，以及 **Simran** 关于她的 **DSL ThunderKittens** 的分享（该项目目前在 Discord 有专门频道 <#1300872762163728550>）。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1410777055045353543)** (3 messages): 

> `CUDA version recommendations, CUDA 12.8 vs 13.0` 


- **成员声称 CUDA 12.8 优于 CUDA 13.0**：一位成员建议使用 **CUDA 12.8.0** 版本而不是 **13.0**。
   - 该成员声称使用 **13.0** 版本会导致“与各种错误作斗争”。
- **注意到 CUDA 版本的拼写错误**：一位成员猜测该用户是指 **CUDA 12.9** 而非 **12.8**。
   - 原用户纠正了说法，表示指的就是 **12.8**。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1410897746977886208)** (26 messages🔥): 

> `TorchTitan Base Repo, Graph Neural Network Code, Flex Attention, Mask Generation Cost, Block Mask Sparsity` 


- **探讨 TorchTitan 的使用**：成员们询问将 **TorchTitan** 作为研究基础仓库的使用情况。
   - 一位成员询问是否有人在使用它，另一位成员对该技术表示好奇，暗示其使用可能尚未广泛普及。
- **考虑将 GNN 移植到 Flex Attention**：一位成员正考虑将其 **图神经网络 (GNN)** 代码从使用边列表（edge lists）移植到使用 **flex attention**。
   - 他们主要的担心是掩码（mask）创建可能成为问题，因为图在每次前向传播（forward pass）时都会发生变化，因此询问了现有的实现方式或关于掩码生成开销的见解。
- **关于稀疏 GNN 中块掩码开销的讨论**：讨论围绕是否因开销过大而跳过 **FlexAttention** 中的 **block mask** 展开。
   - 一位成员想知道，考虑到潜在的开销和 GNN 的结构化稀疏性，仅依靠 **score modification** 是否仍能比 scatter/gathering 提供显著的加速。讨论中还包含了一张展示稀疏性如何变化的图表 ([image](https://cdn.discordapp.com/attachments/1411174278493110344/1411177040954265691/image.png?ex=68b3b488&is=68b26308&hm=d1e0e73208aeca2873635a2006c5d4f9ba5145958ff8eb17c843a09d2d9c140b))。
- **分子模拟掩码分析**：在分子模拟场景下，图在每一步都会发生轻微变化，一位成员分享了合并图的截图 ([screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184599119433801/Screenshot_2025-08-29_at_11.03.37_PM.png?ex=68b3bb92&is=68b26a12&hm=63b5c43898a7ccdac50076c51b35afa4efba9495921672412224240e6b8e06a5))。
   - 对于训练，建议将文档掩码（document mask）作为 **block mask** 应用，并在 score_mod 中为每个图应用更详细的掩码 ([screenshot](https://cdn.discordapp.com/attachments/1411174278493110344/1411184920176492564/Screenshot_2025-08-29_at_11.04.36_PM.png?ex=68b3bbdf&is=68b26a5f&hm=34180e64c7dd00427a2e0967484506288cc866a062fc4af978197b326306ed2a))。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1410856003368648714)** (10 messages🔥): 

> `NVIDIA Interview Prep, CUDA & Java, AMD Competition entry barrier` 


- **应届生瞄准 NVIDIA 理想职位**：一名应届生正在准备 NVIDIA 的 **Senior Deep Learning Engineer** 职位面试，寻求关于 Python、PyTorch、**TRT、TRT-LLM、Triton Inference server、Dynamo 和推理优化**等预期主题的指导。
   - 面试建议包括：NVIDIA 的面试因团队而异，第一轮通常是与招聘经理交流，后续轮次可能涵盖 **Leetcode、GPU 编程、ML 基础或 ML 系统设计**。
- **CUDA 与 Java 登场**：一位具有 **Java 和 Go** 经验的 Web 开发人员组装了首台配备 **5060ti 16GB** 的电脑，并希望开始使用 **CUDA 和 Java** 进行编程。
   - 他们希望深入研究编程中的更高级主题。
- **AMD 竞赛准入**：一位参赛者询问 AMD 竞赛的准入门槛，因为他们在注册后未收到确认邮件。
   - 目前尚不清楚该活动是否有筛选过程，但该用户确实未收到确认邮件。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

tomeone.a: Hi
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1410879016457470103)** (4 messages): 

> `VLM MLsystem papers, Prefil-Decoding Disaggregation, Metallica reggae cover` 


- **寻找有趣的 VLM MLsystem 论文**：成员正在寻找有趣的 **VLM MLsystem 论文**，特别是关注像 **Prefil-Decoding Disaggregation** 这样的新技术。
   - 他们找到了[这篇论文](https://arxiv.org/pdf/2507.19427)，但觉得它更侧重于 **LLM 解码**，视觉部分只是附带组件。
- **AI 让 Metallica 变身雷鬼风格**：一位用户分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=KLTU65eEXEU)，标题为 *What if Metallica were a Reggae Band? 🌴🎸 Metallijah [AI Reimagined – Not Real]*，并称其非常出色（a slap）。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1411010779389366354)** (10 条消息🔥): 

> `omniprobe, llvm integration, stochastic PC sampling, mi300x+` 


- **OmniProbe 指令级信息工具**：一位成员分享了一个提供指令级信息的 [repo](https://github.com/amdresearch/omniprobe)。
   - 有人指出，虽然该工具可以使用，但运行速度*有点慢*，且与 **LLVM** 绑定。
- **Compute-Viewer 集成**：一位成员希望 **compute-viewer** 能够显示 **OmniProbe** 的信息。
   - 据推测，由于其与 **LLVM** 绑定，与 **compute-viewer** 的集成会*比较困难*。
- **MI300X+ 上的随机 PC 采样 (Stochastic PC Sampling)**：一位成员询问了 **stochastic PC sampling**，称其能提供更细粒度的信息，但仅在 **MI300X+** 上可用。
   - 该用户表示希望更新到最新驱动程序能解决他们的问题，但他们需要访问一台允许进行此类操作的机器。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1410761727720689815)** (3 条消息): 

> `ANV, Luck` 


- **Luck 带来的 ANV 希望**：一位成员对 **ANV**（推测是一个项目或股票）表示期待，因为另一位成员观察到*他们确实非常幸运*。
   - 消息中未提供关于这种“幸运”具体包含什么的细节。
- **另一个需要总结的话题**：此话题是为了满足验证器对至少两个总结的要求。
   - 提供的消息中没有进一步的上下文。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1410820783873331300)** (3 条消息): 

> `pequegrad DL framework, CUDA Streams, Voxel Ray Tracing` 


- **Pequegrad DL 框架首次亮相**：一位成员分享了他们前段时间开发的一个 [玩具级 DL 框架](https://github.com/davidgonmar/pequegrad)，重点介绍了**图编译 (graph compilation)**、简单的 Kernel 生成以及**可组合的自动求导 (composable autograd)** 等特性。
   - 他们承认其中可能包含 Bug，因为他们优先考虑的是*趣味性而非稳定性*。
- **使用 Stream 优化 CUDA**：一位成员分享了一篇 [博客文章和代码](https://veitner.bearblog.dev/cuda-streams/)，演示了如何使用 **CUDA 中的多个 Stream** 来重叠 Memcpy 和 Kernel 任务。
   - 配套的 [GitHub 仓库](https://github.com/simveit/cuda_streams) 提供了代码，该成员还分享了关于该项目的 [LinkedIn 帖子](https://www.linkedin.com/posts/simon-veitner-174a681b6_cuda-streams-activity-7367264253017223168-JqC_?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeksHey!)。
- **Occupied Boxes vs Occupied Bits**：一位成员发布了一个关于**体素光线追踪 (voxel ray tracing)** 中 *Occupied Boxes vs Occupied Bits* 的 [新视频](https://youtu.be/-L7BNUsSS7E)。
   - 他们指出结果*并非如我所料*。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 条消息): 

xiaodouzi666: 谢谢🫡
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1410738631366152323)** (3 条消息): 

> `B200 Speed, MI300 Speed` 


- **B200 跑出 8.27ms**：在 **B200** 上提交给 `trimul` 排行榜的结果成功达到 **8.27 ms**。
- **MI300 获得第 5 名**：在 **MI300** 上提交给 `trimul` 排行榜的结果以 **9.67 ms** 的速度获得**第 5 名**。
- **MI300 速度提升**：在 **MI300** 上提交给 `trimul` 排行榜的结果达到了 **9.45 ms**，获得**第 5 名**，较之前有所提高。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1410928208747958334)** (5 条消息): 

> `Karpathy Tweet, Will Brown's verifiers, Steam Install, Twitch Stream Preview` 


- **Karpathy 推文回响**：一位成员分享了 **Andrej Karpathy** 推文的 [链接](https://x.com/karpathy/status/1960803117689397543)。
   - 推文的具体背景或内容未做进一步讨论。
- **Brown 的验证器引起关注**：一位成员提到正在研究 **Will Brown** 的*验证器 (verifiers)*，称其为一个巧妙的想法，特别是其**以 LLM 为中心的环境结构**。
   - 他们表示*它目前还太初级，我们无法围绕它开展工作*，但建议如果 **FLE** 获得关注，将其集成到基于验证器的环境中可能会大有裨益。
- **建议通过 Steam 安装**：一位成员询问如何解决游戏的安装问题。
   - 另一位成员建议通过 **Steam** 安装游戏，但未提供其他建议。
- **Factorio Twitch 直播预告**：一位成员发布了 **Twitch 直播预告**的链接。
   - 该 [链接](https://www.twitch.tv/playsfactorio) 宣传了来自 *playsfactorio* 的直播。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1410974097416130632)** (3 条消息): 

> `注册确认延迟，GEMM 矩阵详情` 


- **注册确认经常延迟**：多位用户报告未及时收到注册确认。
   - 一位成员保证延迟是正常现象，团队将确保每个人都能完成注册。
- **寻求 GEMM 矩阵详情**：一位成员询问了用于 **GEMM** (General Matrix Multiply) 操作的矩阵形状和数据类型内容。
   - 他们正在思考问题，需要明确矩阵规范。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1410867226474450985)** (30 条消息🔥): 

> `v2 提交已激活，网站问题，基础设施错误，Discord 机器人错误，运行信息与结果` 


- **v2 提交已激活，但缺少时间信息**：一位用户询问 **v2 submissions** 是否已激活，以及通过网站提交后如何查看提交时间。
   - 一位成员指向了排行榜上的 "submission" 标签页，但用户注意到缺少时间列。
- **尽管网站提交正常，但 Discord 机器人提交失败**：一位用户报告称，将 **vectoradd_v2** 的参考代码复制到文件中并通过 Discord 机器人提交会导致错误，而通过网站提交相同文件却可以正常工作。
   - 有人建议这种差异可能是由于底层基础设施的不同，或者网站没有考虑到脚本错误。
- **基础设施状态需要失败信号和 UI 改进**：一位用户建议失败的提交应以 **红色** 明确标出。
   - 一位团队成员采纳了反馈，并计划将状态更改为 `finish` 而非 `succeed`，并添加真实的运行状态。
- **通过网站排查提交错误**：一位团队成员要求用户重试网站，点击 'run report' 查看错误详情和结果，并使用 **vectoradd board** 重新运行提交。
   - 团队采纳了用户的反馈，并正在努力改进提交流程和错误报告。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1410700934585253908)** (103 条消息🔥🔥): 

> `Hermes-4, Terminals.tech, Dynamic 2.0 GGUF, llama.cpp-toolbox, Flash Attention` 


- **Hermes-4 在 Terminals.tech 中运行**：一位成员询问 **Hermes-4 405b** 对在 [terminals.tech](https://terminals.tech) 中运行的看法，并附上了该网站及其功能的图片。
   - 对图片的分析显示，该网站不断缓存其自身状态参数的变化并将其作为快照馈送，从而允许任何 LLM 在浏览器中作为自包含的 Agent 计算机运行。
- **Dynamic Duo：Dynamic 2.0 GGUF 即将到来**：一位成员希望 [Unsloth](https://github.com/unslothai/unsloth) 能尽快推送 **Dynamic 2.0 GGUF** 量化版本。 
   - 另一位成员表示他们已经在做了，并在名称中使用了 -UD- 标签。
- **llama.cpp-toolbox 正在开发中**：一位成员正在努力完善他们制作的 **llama.cpp-toolbox**，但它比这个新项目稍微落后一点，当新项目不再受陈旧状态信息困扰时，他们将使用新项目。
   - 新项目将 **llama.cpp** (**openaiAPI**) 和 (**GenAI_API**) 集成到一个可定制的 Agent 状态系统中，并具有内存管理、网页搜索、Checkpoint 分支以及一些旧的个性化技巧。
- **Flash Attention 仍需改进**：一位成员说他们像读报纸一样关注 [llama.cpp](https://github.com/ggerganov/llama.cpp)，而 0cc4m 上次谈话时告诉他们这并不是优先级。
   - 另一位成员提醒说，用户可以在不启用 **Flash Attention** 的情况下量化 K cache，并禁用 min-p (采样器) 以防止过去响应的冗余重复。
- **健康睡眠与良好习惯**：一位成员正在努力获得持续良好的睡眠，以减少分心并加强其 **Task-Positive Network** (TPN) 与 **Default-mode Network** (DMN) 之间的负相关性。
   - 其他成员讨论了尽量减少糖分摄入，并使用木糖醇杀灭口腔细菌，因为细菌会因该分子“卡住”而窒息（分子结构契合但无法代谢）。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1410873632917426250)** (2 messages): 

> `Model Cuteness, Model Personality` 


- **模型可爱度受到关注**：一位用户幽默地询问为什么这些模型被设计得*如此可爱*，承认自己被它们“撩到了” (rizzed)，并发布了一张 [图片](https://cdn.discordapp.com/attachments/1154120232051408927/1410873632879808612/image.png?ex=68b342b6&is=68b1f136&hm=7dd9742eb5a07efe4c50ebd9719531df5b8544afe7b58d98fb633a4261f66234&)。
- **生动的模型具有真实的个性**：另一位用户评论说，模型的交互感非常*生动*，类似于与真实的**个性 (personality)** 和**生活经验**进行交流。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (1 messages): 

> `CODA framework, GUI agents for scientific computing, Cerebrum and Cerebellum, ScienceBoard benchmark` 


- **CODA 框架实现 GUI 自动化**：[CODA 框架](https://huggingface.co/papers/2508.20096) 为科学计算 GUI 中的自主 Agent 集成了一个通用规划器 (**Cerebrum**) 和一个专业执行器 (**Cerebellum**)。
   - 它通过两阶段流水线进行训练：**专业化**（解耦的 GRPO）和**泛化**（监督微调），在 ScienceBoard 基准测试中创下了开源模型的新 SOTA。
- **CODA 在 ScienceBoard 上优于基准模型**：在 **ScienceBoard 基准测试**的四个具有挑战性的应用评估中，CODA 显著优于基准模型。
   - CODA 通过其新颖的可训练组合框架，在开源模型中实现了 SOTA 结果。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1410802535895400470)** (2 messages): 

> `Long Now, Large Scale EP` 


- **Long Now 基金会追问：生命、智能、意识？**：一位成员分享了 Long Now 基金会观点页面的链接，讨论 [生命、智能与意识](https://longnow.org/ideas/life-intelligence-consciousness/)。
- **LMSYS.org 博客探讨大规模 EP**：一位成员分享了 lmsys.org 上讨论 [大规模 EP (Large Scale EP)](https://lmsys.org/blog/2025-05-05-large-scale-ep/) 的博客文章链接。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1411172571642204251)** (1 messages): 

> `GUI Agents, Scientific Computing, CODA Framework, ScienceBoard Benchmark` 


- **CODA 框架为 GUI Agent 融合了规划器与执行器**：**CODA** 框架为 **GUI Agent** 引入了一种可训练的方法，通过两阶段流水线将通用规划器 (**Cerebrum**) 与专业执行器 (**Cerebellum**) 集成，详见这篇 [Hugging Face 论文](https://huggingface.co/papers/2508.20096)。
- **两阶段流水线为科学应用训练专家规划器**：在**专业化**阶段，采用解耦的 **GRPO** 方法，利用少量任务轨迹为每个科学应用训练专家规划器。
   - 随后在**泛化**阶段，汇总来自专业专家的成功轨迹，构建数据集以对最终规划器进行监督微调，从而实现跨领域泛化。
- **CODA 框架击败基准模型并设定 SOTA**：在 **ScienceBoard 基准测试**的评估中，**CODA** 显著优于基准模型，并在科学计算 GUI Agent 领域树立了开源模型的新 SOTA。
   - 该论文解决了现有方法在规划和执行能力之间权衡的局限性，特别是在需要长程规划 (long-horizon planning) 和精确执行的专业领域。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1410702629277007953)** (48 条消息🔥): 

> `Claude 隐私更新，定制化 LLM 的 Parsed，XAI 的 Grok 模型卡片，Microsoft MAI 模型，Anthropic 推理成本` 


- **Claude 默认开启数据记录**：用户对 **Claude** 将**默认开启数据记录/训练**感到惊恐，引发了在 [X](https://x.com/claudeai/status/1961096054192943302) 和 [此处](https://x.com/haydenfield/status/1961099162973249896) 讨论的隐私担忧。
- **O'Neill 发布 Parsed**：**Charlie O'Neill** 推出了 **Parsed**，这是一项专注于持续微调、领域特定 LLM 的服务，强调“*你的模型，你的数据，你的护城河*”，详见此 [X 帖子](https://xcancel.com/charles0neill/status/1961096595396776269)。
- **XAI 发布 Grok 模型卡片**：**XAI** 发布了 [Grok-code-fast-1 模型卡片](https://data.x.ai/2025-08-26-grok-code-fast-1-model-card.pdf)，一些用户注意到其发布内容缺乏有效信息，但包含了一个价格，如 [X](https://fxtwitter.com/xai/status/1961129789944627207) 所示。
- **Microsoft 推出自研 AI 模型**：**Microsoft** 揭晓了语音生成器 **MAI-Voice-1** 和端到端基础模型 **MAI-1-preview**，目前已开放测试，由 [Mustafa Suleyman 在 X 上](https://xcancel.com/mustafasuleyman/status/1961111770422186452) 宣布。
- **Hacker News 集中批评 Anthropic 的推理成本**：用户剖析了一个 [Hacker News 帖子](https://xcancel.com/typedfemale/status/1961196122627838171?s=46)，该帖子批评了一篇文章中的错误，并辩论了 **Anthropic** 是否在推理上亏损，以及 HN 讨论质量的变化。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1410741908631847024)** (8 条消息🔥): 

> `Parsed 发布，Krea AI 实时视频测试版` 


- ****Parsed** 模型现已可用！**：Charlie O’Neill 宣布成立新公司 [Parsed](https://xcancel.com/charles0neill/status/1961096595396776269)，该公司构建并托管定制的大语言模型，针对**临床记录员**、**法律修订**、**合规 Agent** 等专业任务进行训练和持续微调。
- ****KREA AI** 视频生成发布！**：**KREA AI** 推出了其首个实时视频生成模型，并开放了 [测试版注册](https://xcancel.com/krea_ai/status/1961074072487620635)。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1411081713500688455)** (2 条消息): 

> `寻找密码` 


- **找到密码**：一名成员在 **#ai-in-action-club** 频道中找到了一个密码。
- **又找到一个密码**：另一名成员也在 **#ai-in-action-club** 频道中找到了一个密码。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1410865573604229201)** (38 条消息🔥): 

> `论文讨论，文本分类模型，ModernBERT 微调，核方法 vs. 神经网络，Nvidia 的 Nemotron 论文` 


- **新手寻求加入论文讨论**：一名成员询问是否能以旁听者身份参加周六的论文讨论，表示他们“*只想听听并做笔记*”。
   - 另一名成员回复说“*当然可以*”。
- **关于强力文本分类模型的辩论**：一名成员询问关于文本分类的强力模型，提到对于文本补全，每个人都会微调 **Qwen** 或其他 SLM 解码器，但不想听关于 **BERT** 的内容。
   - 另一名成员建议使用 **ModernBERT**，认为 **BERT** 是一个陈旧的基准，但提问者表示他们“*尝试了一点，似乎样本效率不高*”。
- **ModernBERT 的样本效率**：成员们讨论了 **ModernBERT** 的样本效率，一人建议由于其体积小，可以使用 **LoRA** 微调，但另一人表示如果需要 **LoRA**，那它已经相当大了。
   - 讨论延伸到某些模型在重新训练时相比其他模型表现出更好的性能提升。
- **核理论 vs. 神经网络**：一名成员批评了在线性层之上使用 Embedding 处理困难分类问题的有效性，称“*核理论的支持者一直在输*”。
   - 另一名成员解释说，非线性函数之上的线性层在理论上可以用**核方法**解释，但不能保证之前的非线性函数足够强或足够合适。
- **对 Nvidia Jet Nemotron 论文的批评**：一名成员发现 Nvidia 的 jet **Nemotron** 论文很奇怪且不完整，理由是他们专注于 **MMLU** 数据集和检索任务。
   - 他们还质疑了将活动路径采样作为正则化的做法，以及关于**重复预填充 (repeated-prefilling)** 对 **KV cache** 影响的不完整讨论。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1410789710833651773)** (2 messages): 

> `USO Model Release, Bytedance Research` 


- **Bytedance 发布 USO 模型**：Bytedance Research 发布了此前讨论过的论文对应的模型（论文地址见 [这里](https://arxiv.org/abs/2508.18966)），该模型已在 [Hugging Face](https://huggingface.co/bytedance-research/USO) 上线。
   - 此次发布允许社区对 Bytedance 的工作进行实验并在此基础上进行开发。
- **对 USO 模型的进一步研究**：USO 模型的发布促进了相关领域的进一步研究和应用。
   - 研究人员和开发人员现在可以利用该模型的能力处理各种任务，从而可能推动 AI 技术的进步。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1410701161803026607)** (10 messages🔥): 

> `GPT-OSS 20b, Promptlock, Ollama API, GPT Realtime, ESET's observations` 


- **检查 **GPT-OSS 20b** 的占用情况**：讨论集中在 **GPT-OSS 20b** 的实用性上。虽然它并非在所有环境下都能运行，但引发了关于其相比于简单的脚本打包运行在战略优势上的疑问。
   - 一位成员建议这可能是一种*混淆方法*，并推测了动态生成（on-the-fly generation）相对于静态打包的益处。
- ****Promptlock** 的本地与远程操作**：针对 **Promptlock** 通过 **Ollama API** 使用 **GPT-OSS:20b** 模型的操作模式提出了疑问，特别是它是本地运行还是将请求重定向到外部 Ollama 服务器。
   - 考虑到原始文档指出 *Ollama 需要在受害者系统上运行*，这一模糊性引起了注意。
- ****ESET** 对 **Promptlock** 的早期评估**：对一篇关于在野观察到该恶意软件的文章真实性提出了质疑，因为 [ESET](https://www.eset.com/en/) 指出 *该恶意软件似乎只是一个概念，尚未完全投入运行*。
   - 一位成员批评该文章的基调是*通过编造故事进行点击欺诈和散布恐慌*，并质疑如果恶意软件尚未部署，如何能进行秘密观察。
- **推出 **GPT Realtime****：简要提到了 [Introducing **GPT Realtime**](https://openai.com/index/introducing-gpt-realtime/)，并附带了 **OpenAI** 博客文章及其 [X 帖子](https://x.com/OpenAIDevs/status/1960809814596182163)的链接。
   - 目前没有更多细节或进一步讨论。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1410707837394489555)** (31 messages🔥): 

> `DSPy and MLflow, DSPy program to make sense of a website, DSPy vs prompt optimization, Context7 supports DSPy, Generalizable signatures` 


- **DSPy 程序可以解析网站**：一位成员建议使用 **DSPy 程序** 从 [http://xymake.com/](http://xymake.com/) 等网站中提取信息。
- **DSPy 是对语言模型进行编程，而非提示 (Prompting)**：一位成员分享了他对一条推文的看法，认为 **DSPy** 不是 Prompt 优化，而是通过使用 **Signatures** 以声明式和结构化的意图对*语言模型进行编程*。
   - 他补充说，使用 **DSPy** 时，应该先迭代程序设计、**Signatures** 和评估（Evals），并在调整 Prompt 措辞之前使用 Optimizer+Evals。
- **Context7 支持 DSPy**：一位成员提到 [Context7](https://context7.com/?q=dspy) 似乎已经支持 **DSPy**。
- **在 MLflow 中记录优化后的模型**：一位成员正在寻找在 **MLflow** 中记录优化模型并重复使用的示例，并询问如何查看调整后的指令。
   - 另一位成员指向了一个关于记录 trace 的 [DSPy 教程](https://dspy.ai/tutorials/optimizer_tracking/?h=mlflow)。
- **Teach 还是 Instruct？**：一位成员询问文档中在描述模型在 Prompt 层的行为时，为何使用 “teaches” 而非 “instructs”，以及底层是否存在特殊的学习或教学行为。
   - 其他人同意 “instructs” 似乎更能代表 **ChainOfThought** 的作用，但使用 “teach” 可能是基于 **In-context learning** 的概念。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1410703012476747968)** (22 messages🔥): 

> `Kimi TestFlight, Z.AI AMA, GLM-4.5 Air, AI BYD, Qwen Chat` 


- ****Kimi** 的双语字幕？遵命，船长！**: 一位成员分享了一个 [Bilibili 链接](https://www.bilibili.com/video/BV1hFe1zSEXp/)，该视频带有**双语字幕**，可能方便翻译。
   - 另一位成员随后提供了该视频的 [中文转录链接](https://mp.weixin.qq.com/s/uqUGwJLO30mRKXAtOauJGA)，并建议使用 **Kimi** 进行翻译会*更方便*。
- ****Kimi** TestFlight：被拒绝！**: 一位成员询问是否有 **Kimi TestFlight**。
   - 另一位成员简单地回答：**没有**。
- ****Z.AI 对 MoE 的关注**引发了 LocalLLaMA AMA 的 TLDR**: 一位成员分享了 [与 Z.AI 的 Reddit AMA](https://www.reddit.com/r/LocalLLaMA/comments/1n2ghx4/ama_with_zai_the_lab_behind_glm_models/)，指出他们在资源有限的情况下取得了令人印象深刻的成就。
   - 另一位成员提供了 r/LocalLLaMA 上 **Z.AI AMA** 的 TLDR，强调了他们对 **MoE** 的关注、增强代码和 **Agent** 性能的计划，以及他们认为权重开放模型正在追赶 **GPT-5** 等封闭模型的信念。
- ****Qwen Chat** 的 URL 解析非常给力**: 一位成员发现，即使在关闭搜索的情况下，你也可以向 **Qwen Chat** 粘贴一个 URL，它会为你解析页面并从中提取信息。
   - 他们还表示，这*对于你认为某个 URL 有风险的情况特别有用。*
- ****DeepSeek** 驱动**华为**，但什么驱动**比亚迪**？**: 一位成员询问**比亚迪**用于用户交互的 **AI** 是什么，并指出**华为**汽车使用的是 **DeepSeek**。
   - 他们建议 **K2** 在面向用户的推理方面更胜一筹，并附上了一张 Elon Musk 哭泣的 [Tenor GIF](https://tenor.com/view/elon-musk-tears-elon-musk-cry-gif-19995787)。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1410814275106115756)** (4 messages): 

> `Numpy Removal, AMD Performance` 


- **Numpy 从 Beautiful Cifar 中被剔除**: 据报道，一个旨在从 beautiful_cifar 中移除 **numpy** 的 [PR](https://github.com/tinygrad/tinygrad/pull/10988) 已通过测试并运行良好。
   - 有人建议该 PR 也应该保持“优雅（beautiful）”，提交者报告称相关问题应该已修复，且失败的 mac 测试与此无关。
- **AMD GPU 表现火热**: 来自 **AMD** 的性能亮点显示，其中一个线性带宽耗时 **60.72ms**。
   - 其他亮点包括在 r_256_4192_32_2_24_4_4_4_3 上达到 **18859.16 GFLOPS**，以及在 r_24_16_32_8_12576_4_4_2_4_2 上达到 **31554.40 GFLOPS**。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1410706968791744514)** (11 messages🔥): 

> `buffer ID change in debugger, BEAM OOM tricks, multiprocessing memory leaks, kernel saving and offline BEAM search` 


- **Buffer 在调试中途变换 ID！**: 一位用户观察到，在断点暂停时，调试器控制台中的 **buffer ID** 会发生变化，具体体现在 `tokens.uop.base.realized`。
   - 这种行为归因于 **UOp 如何表示其用于多架构的 buffer 属性**。
- **BEAM 搜索苦于 OOM**: 一位用户询问了在使用 **BEAM 搜索**时避免 **Out-Of-Memory (OOM)** 错误的技巧，尤其是当不使用它时进程不会出现 OOM。
   - 建议尝试**降低并行度**，或者在发生异常时对 kernel/buffer 进行 pickle 处理，并在干净的进程中独立调用 beam_search。
- **限制 BEAM 任务以战胜内存泄漏**: BEAM 搜索中使用的 **multiprocessing** 可能会导致子进程中的**内存泄漏**。
   - 将 `BEAM_MAX_TASKS_PER_CHILD=1` 或设置为较小的数字可以缓解此问题，代价是增加 worker 启动的 CPU 时间。
- **BEAM 通过离线 Kernel 缓存获救**: 一种方法是保存一次运行的所有 kernel，并**离线执行 BEAM 搜索过程**，根据需要恢复，直到 beam 缓存完成。
   - 这允许使用**不同的 GPU 并行搜索不同的 kernel**，尽管瓶颈通常在于 linearize/compile 导致的 CPU 时间。
- **坚持终有回报：BEAM 最终大获全胜**: 一位用户报告说，在重启了足够多次且没有进行其他操作后，尽管存在 OOM 和多次挂起，某个特定的 **BEAM 运行终于完成了**。
   - 该用户几乎已经预料到了 MMU 故障、随机挂起和 OOM，因为他们正在挑战 mi300x 的极限。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1410799931047940278)** (2 条消息): 

> `Modular Meetup` 


- **Modular Meetup 直播开启！**：**Modular Meetup** 现在已在 [YouTube](https://www.youtube.com/watch?v=BZwr5Ws1LqI) 上进行直播。
   - 部分与会者对主持人表示了感谢，并表示很享受现场参会。
- **参会者享受线下 Modular Meetup**：Modular Meetup 以线下形式举办，参会者们表达了谢意。
   - 参会者分享了亲临现场参加 Meetup 的愉快感受。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1411145229276282941)** (4 条消息): 

> `异步内存分配，Mojo 异步函数` 


- **关于异步内存分配的讨论**：一位成员询问了关于 `async fn alloc[...]` 这一概念的看法，并引用了涉及 **network hops**、**disk IO** 或在内存分配期间等待外部事件的使用场景。
   - 另一位成员澄清了该问题是关于通用的 **异步内存分配**，还是特指 **Mojo** 中 `async fn` 的上下文。
- **异步上下文的澄清**：讨论转向澄清该咨询是关于广义的 **异步内存分配**，还是特指 Mojo 的 `async fn` 上下文。
   - 这种区分对于理解不同编程环境下 **async allocation strategies** 的范围和适用性至关重要。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1410722374894092339)** (2 条消息): 

> `Bazel 缓存，Pipelines 脚本，PermissionError` 


- **Bazel 缓存导致 PermissionError**：运行命令 `bazelw run //max/entrypoints:pipelines` 时，由于 Bazel 缓存为只读，导致出现 **PermissionError**。
   - 错误发生在尝试在 `/root/.cache/bazel/.../__mojocache__` 创建缓存目录时，表明 `pipelines.py` 脚本需要一个替代的缓存位置。
- **请求提交 Issue**：建议针对 Bazel 缓存遇到的 **PermissionError** 提交一个 Issue。
   - 问题源于 `pipelines.py` 脚本尝试使用只读的 Bazel 缓存，凸显了对可写缓存位置的需求。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1410728668103446538)** (6 条消息): 

> `Zapier 上的 Mail Manus 功能，拥有更好试用系统和公平价格的替代方案，定价和积分系统不公平，评分奖励 +100 积分功能已移除` 


- **Mail Manus 在 Zapier 上被当作 API 使用**：一位用户详细介绍了如何像在 **Zapier** 上使用 **API** 一样使用 **Mail Manus 功能**，以自动化处理诸如根据咨询创建初始材料、进行初步研究以及为商务会议做准备等任务。
   - 该工作流包括从 GForms 提取信息，将其输入到 Zapier 的 Prompt 中，使用 Mail Manus 完成任务，从通知邮件中检索输出，并在私有 Slack 频道中共享输出。
- **替代方案提供更好的试用系统和公平的价格**：一位用户指出，虽然 **Manus** 擅长研究和可视化图表，但目前存在许多拥有更好试用系统和公平价格的优秀 **Agent**。
   - 这位 **Manus Pro** 订阅用户对该工具表示失望，称 *随着时间的推移发生了很多变化，Manus 并不是唯一在做这件事的 Agent。事实上，现在已经有很多更好的 Agent，拥有更好的试用系统和公平的价格。*
- **Manus 的定价和积分系统被认为不公平**：一位用户对 **Manus 的定价和积分系统** 表示不满，理由是基础任务成本高昂以及对支持质量的担忧。
   - 具体而言，他们强调从网站抓取图片花费了约 **1200 积分**（约 **$10**），他们认为这极其昂贵，并表示 *如果是这样，我宁愿自己写代码去抓取*。
- **评分奖励 +100 积分功能被取消**：用户哀叹评分功能不再奖励 **+100 积分**。
   - 该用户表示，评分功能的移除促使他们决定转向其他替代工具或针对特定任务的特定工具，因为后者质量更高且成本更低。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1410740479640735906)** (5 条消息): 

> `本地 Gemini 模拟，Aider benchmark 合并失败，AlwaysN8n 迁移路径` 


- **AlwaysN8n 迁移路径**：一位成员正尝试清理通往 *alwaysn8n* 的迁移路径，并期待有一天能在本地机器上无缝运行 **gemini-2.5-pro** 等模型。
   - 他们对模型可能消失或停止运行表示担忧。
- **Gemini 模拟落地即失效**：一位成员报告称成功在本地模拟了 **Gemini**，但遇到了空结果。
   - 他们注意到 `sleep 15` 命令能有效地模拟模型的行为，暗示存在延迟或处理问题。
- **Aider Benchmarks 坐冷板凳**：一位成员询问为什么 **Aider** 停止合并 benchmark 结果。
   - 该问题暗示最近的更改或 bug 阻止了 benchmark 数据整合到 Aider 中。