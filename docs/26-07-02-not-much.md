---
companies:
- anthropic
- langchain
- llamaindex
- togethercompute
- hugging-face
date: '2026-07-02T05:44:39.731046Z'
description: '**Fullstack Code Arena** 将代码代理评测扩展到**数据库、API 密钥、部署和结构化工具调用**，标志着评测重点转向端到端的应用交付。**LangChain**
  发布了支持统一追踪的 **LangSmith**，以及能够自动生成文档的 **OpenWiki**；与此同时，**LlamaIndex** 展示了面向智能体的原生解析能力。如今，用户体验面临的主要挑战已转向路由、可观测性和记忆等协同问题，**Simon
  Willison** 和 **Will Depue** 都强调了这一点。尽管在部署方面仍存在一些争议，**Anthropic** 还是通过提高 **Fable**
  的 API 速率限制并扩展 **Claude Code** 的功能，改善了其运营访问能力。随着 **Together** 报告称 **GLM-5.2** 以 20%
  的成本实现了 **Sonnet 5** 约 80% 的编程能力，开放模型的经济性开始受到关注；此外，借助 **Hugging Face** 的推理服务提供商，**GLM-5.2**
  也已可以在 **Claude Code** 中选择使用。**Clement Delangue**、**Jason** 和 **Bryan Catanzaro**
  等行业领袖强调，开放模型在开发者工作流中的可信度正在不断提升。

  '
id: MjAyNS0x
models:
- glm-5.2
- sonnet-5
- fable
- claude-code
people:
- simonw
- willdepue
- clementdelangue
- bryancatanzaro
title: '今天没发生什么特别的事。

  '
topics:
- agentic-coding-systems
- developer-workflow
- model-access
- api-rate-limits
- model-deployment
- retrieval-augmentation
- routing
- observability
- memory-management
- open-model-economics
- coding-performance
---

**平静的一天。**

> 2026 年 7 月 1 日至 7 月 2 日的 AI 新闻。我们浏览了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有继续查看其他 Discord。你可以在 [AINews 网站](https://news.smol.ai/) 搜索往期全部内容。提醒一下，[AINews 现在已经是 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览

**Agentic Coding 系统、Harness 以及开发者工作流基础设施**

- **全栈评测正在取代玩具级编码演示**：[Code Arena](https://x.com/arena/status/2072713730711023673) 推出了 **Fullstack Code Arena**，将评测范围从前端 mockup 扩展到包含**数据库、API 密钥、部署和结构化工具调用**的软件。这与更广泛的趋势一致：大家关注的问题正从“模型能不能写出一个组件？”转向“Agent 能不能端到端交付一个真实应用？”。[Aryan Vichare](https://x.com/aryanvichare10/status/2072736881859756503) 的观点，以及许多从业者对基于环境的评测、而非静态 prompt 的强调，都印证了这一点。
- **围绕编码 Agent 构建的工程技术栈正在迅速变得更加完善**：LangChain 在 [LangSmith](https://x.com/LangChain/status/2072738719707050028) 中为异构编码工具推出了统一的 tracing 功能，并在[这次发布](https://x.com/BraceSproul/status/2072744824470724887)中加入了用于自动生成代码仓库文档和更新 AGENTS.md 的 **OpenWiki**。LlamaIndex 展示了一个虽小但很实用的模式：通过 [LiteParse + flue + Resend + Turso 邮件助手](https://x.com/llama_index/status/2072713940073603190)，解析不再只是预处理步骤，而是成为 Agent 原生能力。与此同时，[Jerry Liu](https://x.com/jerryjliu0/status/2072832362443067782) 等人的多篇帖子认为，检索的复杂性正越来越多地被编码到**Agent 层**，工具则变得更简单，编排变得更智能。
- **实际的 UX 难题如今在于协调，而不是原始代码生成能力**：开发者反复提到，前沿编码能力已经足够出色，瓶颈因此转移到了**路由、可观测性、协作、记忆和理解**上。[Simon Willison](https://x.com/simonw/status/2072730602344984821) 强调，“理解之后再参与”是应对编码 Agent 所造成认知债务的关键方法；[Will Depue](https://x.com/willdepue/status/2072793965565468789) 描绘了理想的最终形态：一个始终在线、拥有持久记忆、能够委派行动、收发消息并操作计算机的执行助理。同样的需求也体现在 [PersonalOS](https://x.com/willdepue/status/2072798659100684699) 中：它会利用个人数据导出内容，汇总出一份包含 30 万个 token 的个人生活上下文包。

**模型可用性、前沿编码性能，以及开源与闭源的定位**



- **Anthropic 围绕 Fable 的讨论占据主导，但最具体的消息来自运营层面**：Anthropic 并没有发布新的权重，而是通过改善访问条件来恢复用户信心：[官方 API 速率限制已提高并简化](https://x.com/ClaudeDevs/status/2072818299361263778)，[Trapit Bansal 表示，待容量允许后，Fable 预计会重新提供订阅服务](https://x.com/trq212/status/2072814903170408784)。Anthropic 还将 [Claude Code artifacts 扩展到 Pro 和 Max 计划](https://x.com/ClaudeDevs/status/2072770790114914317)，让用户更容易查看和分享长时间运行的编码会话。
- **社区信号表明，尽管存在重新路由方面的争议，Fable 仍属于前沿级模型**：一些热门帖子批评了 Anthropic 的部署和路由行为，但即便是批评者，也在将这些问题与模型质量区分开来。[Theo](https://x.com/theo/status/2072777433997316436) 认为，针对 Fable 的错误解读分散了人们对 Anthropic 实际问题的注意力；与此同时，[Arena 的早期前后对比](https://x.com/arena/status/2072828263848894783) 表示，在文本、文档、视觉和代码能力重新部署后，得分看起来 **基本保持一致**。[Theo 还指出](https://x.com/theo/status/2072777929180987485)，部分基准测试成绩下滑，可能更多是回退行为造成的，而不是基础能力真的退化。
- **开放模型在编码领域的经济性越来越可信**：Together 声称，[GLM 5.2 以约五分之一的价格，达到 Sonnet 5 软件工程能力的大约 80%](https://x.com/togethercompute/status/2072836285455368605)；[zRdianjiao 展示了](https://x.com/zRdianjiao/status/2072906526722064415)，**GLM-5.2 现在可以通过 Hugging Face Inference Providers 在 Claude Code 中选择使用**，这标志着开放模型正在更进一步融入一流的开发工作流。从更广泛的角度看，[Clement Delangue](https://x.com/ClementDelangue/status/2072752653436653755)、[Jason](https://x.com/Jason/status/2072778368530198973) 以及 [Matt Turck 对 Bryan Catanzaro 的采访](https://x.com/mattturck/status/2072723410975629364) 都在阐述同一个观点的不同版本：**开放模型正在成为企业和开发者的主权层**。
- **Meta 似乎正重新加入 Agent 领域的讨论**：[Alexandr Wang 发文称](https://x.com/alexandr_wang/status/2072848108342677597)，下一次 **Muse Spark** 更新即将推出，将在“编码和 Agent 能力方面实现重大改进”，以具备与领先模型竞争的实力，并会逐步推送到 Meta AI 及其 API。

**推理、Kernels、Serving 与测试时计算：新的扩展前沿**

- **Kernel 层面的自动化已不再只是设想**：最受关注的系统帖文来自 [Elliot Arledge 的 KernelBench-Mega 结果](https://x.com/elliotarledge/status/2072814573753975266)：据称，Claude Fable 5 为 Kimi-Linear 解码工作负载编写出了首个真正的 **单次启动 megakernel**，相较参考实现取得 **18.7 倍性能**，并击败了此前采用多 Kernel 的结果。其描述足够详细，对系统工程师而言很有参考价值：包括寄存器内 int4 反量化、融合 attention/router/MoE/norm/KV append、明确减少 barrier，以及模型主动进行基准测试、撤销回归改动，并围绕 roofline 进行优化。
- **Speculation 和 speculative decoding 仍是活跃的优化方向**：[teortaxesTex](https://x.com/teortaxesTex/status/2072716346899456123) 提出，“扩展 speculator”可能成为加速推理、进而提升 RL 吞吐量的新维度；[mgoin_](https://x.com/mgoin_/status/2072785822231728363) 则分享了一个在 **GB300 NVL72** 上运行 **DSpark + Mooncake + vLLM** 的具体配置，在线训练达到了 **125k prefill tok/s** 和 **1.5 steps/s**。vLLM 团队还强调，[DeepSeek V4 的单 token 成本在一个月内降低了 5 倍](https://x.com/vllm_project/status/2072722401813565834)，并发布了一份很有价值的 [Qwen3-Omni 实时语音流水线服务分析](https://x.com/vllm_project/status/2072942203966812438)：通过按阶段进行副本扩展，首段音频生成时间从约 **6 秒缩短至约 0.6 秒**，吞吐量提升 **5.4 倍**。
- **测试时计算预算正在改变人们对基准测试的解读**：英国 AISI 关于扩大计算预算的帖子被广泛传播。[scaling01](https://x.com/scaling01/status/2072799566735306760)、[Tomek Korbak](https://x.com/tomekkorbak/status/2072863584586219924)、[Noam Brown/polynoamial](https://x.com/polynoamial/status/2072909389389021484)、[David Rein](https://x.com/idavidrein/status/2072830683974906170) 和 [Toby Ord](https://x.com/tobyordoxford/status/2072948952274530404) 都强调了同一点：如果分配的 token 不够，就会系统性地低估前沿 Agent 的能力。最醒目的数字是：前沿 Agent 的任务持续时间估计值，会从 **250 万 token 时约 2 小时**，提升到 **5000 万 token 时约 14 小时**。

**关于学习、记忆、世界模型与持续适应的基准测试和研究**



- **Continual/on-the-fly learning 正在获得更精准的测量工具，但结果仍然喜忧参半**：Epoch 推出了 [EBR-bench](https://x.com/EpochAIResearch/status/2072714372175237255)。在这项测试中，模型会反复游玩 Earthborne Rangers，并尝试从失败中学习；目前的前沿系统在没有专门 RL 的情况下，**尚未表现出明确的提升**。与此同时，ByteDance Seed 发布的新 [EdgeBench](https://x.com/scaling01/status/2072790212615237858) 也引起了广泛关注：它在 **134 个真实世界环境中研究长达一天的任务周期**，并声称学习速度每约 3 个月翻一倍，而且这种提升无法仅用重复采样来解释。这项 benchmark 很快被视为 METR 风格 horizon 研究的重要补充。
- **Memory 正从辅助模块提升为可训练的能力**：Stanford 的 **AutoMem** 论文经 [Omar Sanseviero 的总结](https://x.com/omarsar0/status/2072716688483831885) 传播后受到关注。该研究将 memory management 视为一种技能，让模型自行决定存储、检索和重组哪些信息；据称，仅优化 memory 就能让模型在 Crafter、MiniHack 和 NetHack 上取得 **2–4 倍** 的提升。这一思路也与另一项更偏应用的趋势相呼应：构建持久化的个人 memory 和 research memory 系统。[PaperWiki](https://x.com/omarsar0/status/2072735813469905026)、[PersonalOS](https://x.com/willdepue/status/2072798659100684699) 和 OpenWiki 都表明，memory 正逐渐成为产品的一部分。
- **World models 正从静态资产转向能够在线适应的动态组件**：Reka 发布了 [WorldModelGym](https://x.com/RekaAILabs/status/2072731356011045088)，围绕 **基于决策的 fidelity**，在 100 多条 tracks 上进行评测。[askalphaxiv 对 AdaJEPA 的总结](https://x.com/askalphaxiv/status/2072750223026438226) 则提出了更进一步的观点：预训练的 world models 应在部署期间持续适应；在每个 MPC 周期执行一次 gradient step，可以提升模型在视觉和 dynamics 发生变化时的稳健性。

**互动量最高的推文**

- **Anthropic 访问权限/容量更新**：[Trapit Bansal 表示，Fable 会在容量允许时恢复订阅](https://x.com/trq212/status/2072814903170408784)——这是目前最清晰的信号，说明当前的供应紧张源于容量问题，而不是永久性的产品包装决策。
- **立刻影响运营的 API/平台变更**：[Claude API 提高 rate limits 并简化 tiers](https://x.com/ClaudeDevs/status/2072818299361263778)。
- **面向 coding 的 model stack 组合**：[Mitchell Hashimoto 分享了 planner/coder/judge 工作流](https://x.com/mitchellh/status/2072715852944957531)，使用 **Fable xhigh → GPT-5.5 xhigh → Fable xhigh**；相比价格高得多的端到端循环，规划和评判环节的成本只有几美元。
- **专业化 post-training 击败 frontier prompting**：[Aakash Gupta 分享了 Bridgewater + Thinking Machines 的结果](https://x.com/aakashgupta/status/2072765754102174114)：经过 fine-tuning 的 **Qwen3-235B** 达到 **84.7%**，在 document filtering 任务上超过依赖 prompting 的前沿模型，而 inference cost 仅为后者的 **约 1/14**。
- **Autonomous systems 在底层优化上的表现**：[Elliot Arledge 分享了由 Fable 编写的 megakernel 结果](https://x.com/elliotarledge/status/2072814573753975266)，这或许是这组内容中技术含量最高的 coding-agent 真实案例。
- **Video generation 的领先者发生变化**：[Design Arena 报道 Gemini Omni Flash 以 1404 Elo 位居 Video Arena 第一](https://x.com/Designarena/status/2072759122366509130)，领先 Seedance 2.0 Mini **101 分**，也是该排行榜上观察到的较大幅度跃升之一。


---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述



### 1. llama.cpp 长上下文与 Qwen 3.6 优化

  - **[llamacpp 补丁：DeepSeek V4 Flash 在 RTX 5090 上本地运行完整 1M token 上下文](https://www.reddit.com/r/LocalLLaMA/comments/1ulymml/llamacpp_patch_deepseek_v4_flash_running_with/)**（热度：374）：**一个 `llama.cpp` 补丁将 DeepSeek V4 Flash 的 DSA/lightning indexer 接入模型计算图，并新增 CUDA kernel，使 [DeepSeek-V4-Flash GGUF](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf) 可以在 **RTX 5090** 上本地运行，支持最高 **`1M` 上下文**，不再需要约 `256 GiB` 的计算缓冲区显存。据报告，在 `256K` 上下文下，计算缓冲区从约 `67 GiB`/OOM 降至 `3.2 GiB`，prefill 速度从 `56 t/s` 提升至约 `263 t/s`，decode 速度仍约为 `14 t/s`；经过验证的预设显示，`256K`/`512K`/`1M` 上下文下峰值显存约为 `29`/`28`/`31 GiB`，而由于减小了 `ubatch`，`1M` 下的 prefill 速度约为 `159 t/s`。作者在[说明文档](https://github.com/spencer-zaid/llama.cpp/blob/deepseek-lid-cuda/docs/deepseek-v4-lid-cuda.md)和[分支](https://github.com/spencer-zaid/llama.cpp/tree/deepseek-lid-cuda)中提供了源码和构建说明。该实现基于上游 PR [ggml-org/llama.cpp#24231](https://github.com/ggml-org/llama.cpp/pull/24231)，并报告称已在 `100K`、`512K` 和 `1M` 上下文下通过基础的 needle-in-a-haystack 正确性测试。**评论大多对在单张 RTX 5090 上运行 DS4 Flash 的可行性持积极态度；有一条技术跟进评论要求提供 TTFT 和/或端到端 token 生成耗时（`tg-end2end`）。

    - 一位评论者要求提供在单张 **RTX 5090** 上运行本地 **DeepSeek V4 Flash** 的具体延迟指标，尤其是 **TTFT** 和 `tg-end2end`，以验证在所宣传的**完整 `1M` token 上下文**下的实际可用性。
    - 另一个技术疑问认为这一结果“*好得令人难以置信*”，建议将补丁提交给 **上游 `llama.cpp`** 进行审查，并指出在真正采用之前，可能还需要对其正确性和性能进行验证。
    - 一位评论者提到了正在进行的 **`llama.cpp` lightning indexer 修复**，并建议将其移植到 **Metal**，这意味着当前补丁可能主要面向 CUDA，而 Apple GPU 支持则需要针对具体后端进行适配。

  - **[qwen3.6 27b q6 + 5090 maximum llamacpp optimization: 100-233tok/s, average 140](https://www.reddit.com/r/LocalLLM/comments/1ullrvq/qwen36_27b_q6_5090_maximum_llamacpp_optimization/)**（热度：201）：**一位用户分享了在 **RTX 5090 32GB / Ryzen 9800X3D / 64GB RAM** 系统上，使用近期的 `llama.cpp` 构建版本（`86b9470`）对 **Qwen 3.6 27B Q6_K + MTP** 进行优化后的推理结果：在约 20 小时的 agentic 工作负载中，速度达到 `100–233 tok/s`，平均 `140.7 tok/s`，中位数 `134.9 tok/s`。主要解决的技术问题是 `llama.cpp` 对 Qwen **hybrid attention / sliding-window attention** 行为的 prompt-cache 失效问题——日志显示由于缺少缓存数据而出现 *“forcing full prompt re-processing due to lack of cache data”*，相关讨论见 [`llama.cpp` PR](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)。用户通过两个本地补丁缓解了这一问题：一个用于修复 hybrid/recurrent 模型的 checkpoint-search，另一个是基于上游 PR `#24785` 的精简版 `recurrent_shrink/expand` prompt-cache API 补丁（[Dockerfile](https://pastebin.com/raw/jyrhvesQ)、[diff](https://pastebin.com/raw/E55YG5NS)）。其启动配置使用 **Q8 KV cache**、`192k` 上下文、约 `32GB` RAM cache，以及 MTP speculative decoding（`draft=10`、`spec-draft-p-min=0.5`）；同时将 `batch/ubatch=512`，以便将显存控制在约 `32036/32768 MB` 以内。用户指出，如果显存允许，在 5090 上使用 `2048` 会更理想（[启动命令](https://pastebin.com/raw/P57Uk6rz)）。**



### 2. Gemma 4 开源模型实验与基准测试



  - **[我将 Gemma4-31B 扩展到了 44B（88 层）——因为 Google 不会给我们超过 31B 的模型](https://www.reddit.com/r/LocalLLaMA/comments/1ul0cx9/i_extended_gemma431b_to_44b_88_layers_since/)**（热度：1287）：**这张图片是一张技术信息图，而不是表情包：它展示了从 **Gemma4-31B** 到 **ExtGemma4-44B** 的架构扩展路径——先通过插入以恒等方式初始化的层，将 `60 → 80` 层；再通过复制并插入一个 8 层模块，将 `80 → 88` 层——这与作者在 [Hugging Face](https://huggingface.co/TOTORONG/extGemma4-44B) 上的说明以及[图片](https://i.redd.it/qbkvzo4s3pah1.png)一致。其主要技术意义在于使用了**恒等初始化（identity initialization）**，以及针对 Gemma 的 `layer_scalar = 1.0` 修复，以保持模型初始行为不变。作者声称，在使用韩语法律和 STEM 数据进行微调后，新增的全注意力层得到了训练，并且贡献高于滑动窗口层。**评论总体上比较支持，但也保持谨慎：有人建议将其与 **RYS / “repeat yourself”** 层复制方法进行基准测试；也有人表示没有足够的硬件运行该模型，或者开玩笑说角色扮演微调的需求很大。

    - 有评论者建议将这个 44B/88 层扩展模型与 **RYS（“repeat yourself”）** 基线进行对比。RYS 是一种通过复制连续层来创建更大模型的方法。他们认为，RYS 是一种快速但比较粗糙的方式，可以让已有模型“变得更大也更好”，因此可作为一个有用的对照组，用来评估发帖者的层扩展策略是否真的比朴素的层复制带来更多收益。
    - 社区成员期待在可用的社区构建版本发布后进行下游**量化实验**，不过评论者表示自己没有运行完整模型所需的硬件。另一位评论者将这一方法与 Llama 2 / Llama 3 时代早期的**“Frankenstein”扩展模型**联系起来，暗示社区过去已经尝试过拼接或扩展 Transformer 架构。

  - **[与 Gemma 4 31B 对话！](https://www.reddit.com/r/LocalLLaMA/comments/1ulgwld/talking_with_gemma_4_31b/)**（热度：1006）：****Hugging Face 的 Andi** 分享了一个完全开源的语音到语音演示系统。它将 **NVIDIA Parakeet ASR → 由 Cerebras 提供服务的 Gemma 4 31B → 自定义 [`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts)** 串联起来，并支持网页、视觉和搜索功能；同时采用与 API 兼容的设计，旨在作为 OpenAI realtime API 的即插即用替代方案。完整技术栈已发布在 [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech) 上，并在 [Hugging Face Spaces](https://huggingface.co/spaces/smolagents/hf-realtime-voice) 提供了在线演示。作者声称，在 **MacBook Pro M3 36GB** 上使用 **Gemma 4 E4B** 时，可以实现与由 Cerebras 提供支持的 `31B` 模型相近的本地延迟。**评论者主要围绕部署取舍展开讨论：鉴于 **Gemma 12B** 内置音频/图像支持并且在本地 GPU 上运行速度很快，它是否已经足够；不依赖 Cerebras 的情况下，能否在 **RTX 6000** 上实现实时延迟；以及该系统是否适合日语对话等语言练习场景。

    - 评论者询问 **Gemma 4 31B** 的部署目标，想知道是否可以不依赖 **Cerebras** 推理硬件，而是在 **RTX 6000** 上实现实时交互。另一位评论者指出，Cerebras 很可能会“彻底碾压”传统硬件，但希望看到在更容易获得的设备上进行的基准测试，例如 **Spark** 或本地 GPU，而不是依赖价值数百万美元的基础设施。
    - 一个技术上的比较点是，**Gemma 12B** 是否已经足以满足目标使用场景：简单聊天加网页搜索，而且据称在本地 GPU 上运行“快得惊人”。评论者还指出，Gemma 12B 据报道已经支持内置的**音频/图像理解**，因此问题在于，更大的 **31B** 模型能否提供足够明显的质量提升，从而抵消更高的推理成本和延迟。
    - 一位评论者介绍了类似的实时语音到语音架构：使用 **Parakeet / NVIDIA NeMo** 进行 STT，使用 **Microsoft VibeVoice realtime** 进行 TTS，并通过插件后端支持 **Qwen ASR** 和 **Whisper**。他们强调了可插拔的后端设计，以及能够为本地助手、前端和游戏添加语音到语音能力的客户端 API，认为这个 Gemma 语音项目与更广泛的模块化 STT/TTS 流式服务器模式存在重叠。



  - **[SWE-rebench 排行榜更新：GLM-5.2、Qwen3.6-27B、Qwen3.6-35B-A3B、Gemma 4 31B 等，以及改进后的 UI](https://www.reddit.com/r/LocalLLaMA/comments/1uknx14/swerebench_leaderboard_update_glm52_qwen3627b/)**（热度：321）：****SWE-rebench** 更新了其[排行榜](https://swe-rebench.com/) UI，并新增或刷新了多个 coding-agent 模型的结果，报告了解题率和 token 使用量：**Claude Opus 4.8 xhigh** 为 `56.5%` / `2.48M tokens`，**GLM-5.2** 为 `51.1%` / `2.62M`，**Gemini 3.5 Flash** 为 `49.5%` / `1.85M`，**MiniMax M3** 为 `45.6%` / `6.89M`，**DeepSeek-V4 Pro** 为 `42.7%`；可本地运行或自行托管的模型中，**Qwen3.6-27B** 为 `36.5%`，**Qwen3.6-35B-A3B** 为 `33.8%`，**Gemma 4 31B** 为 `16.5%`。公开排行榜还展示了不确定性、次级成功指标、成本、token 使用量和缓存率等数据。目前排名靠前的系统包括：**gpt-5.5-2026-04-23-xhigh**，`62.7% ± 0.91%`；**Junie**，`61.6% ± 0.64%`；**Codex**，`60.4% ± 1.37%`；以及 **Claude Code**，`59.6% ± 1.98%`。可通过 [Harbor](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest) 获取复现所需的运行结果和相关产物。**评论者主要希望加入更多可运行于本地的 coding 模型，包括：**MiMo-V2.5**、**MiniMax-M2.7**、**Step-3.7-Flash**、**Cohere North Mini Code**、**JetBrains Mellum2**、**Gemma 4 26B A4B**、**Ornith-1.0**，以及更大的 **Qwen 3.5 122B/397B**。鉴于 **Gemma 4 31B** 只有 `16.5%` 的成绩，一些人对 Gemma 在 coding-agent 任务上的表现持怀疑态度；但也有人认为，小模型成本低、速度快，仍然值得纳入测试，并可作为衡量下限的参考。

    - 有多位评论者要求在 SWE-rebench 中加入更多可在本地运行的小型模型，尤其是 **MiMo-V2.5**。据称它的基准测试成绩接近 **MiMo-V2.5-Pro**，但更适合本地运行。此外，大家还点名了 **MiniMax-M2.7**、**Step-3.7-Flash**、**Cohere North Mini Code**、**JetBrains Mellum2** 和 **Gemma 4 26B A4B**。一位评论者指出，**MiniMax-M2.7** 应该可以在具备 `128 GB` 统一内存的系统上运行，因此即使它在排行榜上的名次可能较低，也很适合作为本地 SWE 风格评测的候选模型。
    - 有人希望测试更大的 **Qwen 3.5** 版本，具体包括 `122B` 和 `397B`，以及近期面向 SWE-bench 优化的微调模型，例如 **Nex-N2** 和 **Ornith-1.0**。有人特别提到，Ornith 虽然在发布时受到关注，但目前缺少足够明确的独立证据来证明其实际 coding-agent 性能。
    - 一位评论者表示，在他们的测试中，**Qwen “instruct revised”** 配合经过优化的 **Jinja chat template** 后，表现“比原生版本好很多”，并愿意分享该模板。这表明，排行榜结果可能不仅取决于模型权重，也会受到 prompt/chat-template 格式以及推理封装细节的显著影响。


### 3. 超越基准测试的 LLM 产品可靠性

  - **[闭源模型与开放模型之间的差距可能远小于通常认为的程度，因为我们不知道闭源模型提供商除了模型推理之外还做了什么](https://www.reddit.com/r/LocalLLaMA/comments/1ukp2bu/the_gap_between_closed_and_open_models_might_be/)**（热度：1434）：**这篇帖子认为，闭源 API（如 **[Claude](https://www.anthropic.com/claude)**）与开放权重模型（如 **GLM-5.2**）之间的基准测试差距，可能混淆了*基础模型质量*与不透明的产品级编排：隐藏的系统提示词、prompt 预处理、RAG/知识注入、内部 tool call、模型路由，或专门的专家子模型等。由于闭源服务商只提供 API 接口，而且可能会隐藏或删减推理过程及上下文，因此参与基准测试的对象可能是完整的推理流水线，而不是单一模型；将其与“裸跑”的开放权重模型直接比较，在技术上并不等价。**热门评论基本都认同，闭源模型与开放模型之间的基准测试经常是在比较性质不同的对象：商业 API 可能包含 Agent、批评器、路由机制或辅助工具，而开放模型通常以独立运行的方式接受测试。评论者呼吁围绕开放模型建立标准化、可在本地部署的开放式流水线和框架，并指出当前相关工具仍然碎片化，且大多是临时拼凑的方案。



- 一些评论者认为，封闭模型之间的比较存在混淆因素，因为 **Claude、ChatGPT 和 Gemini** 这类产品呈现的是一套由 API 支持的编排栈，而不是单一的原始模型。他们指出，开放权重/开源模型的评测通常针对的是一个“裸模型”，而商业系统可能还包含路由、Agent、评论器/验证器、检索、护栏层、提示词重写或其他隐藏工具，因此很难把基准测试中的持平或优势单独归因于底层模型。
- 另一个技术主题是需要**可部署的本地 AI 流水线**，而不仅仅是 GGUF/基础模型文件。评论者认为，目前这个生态缺少将模型与周边框架组件组合起来的标准——例如工具调用、记忆/RAG、安全过滤器、上下文管理、Agent 以及 UI 编排等。他们还指出，**SillyTavern** 等项目虽然部分整合了这些组件，但整体仍然比较杂乱，还称不上标准化的生产级流水线。
- 一位评论者指出，**Anthropic** 似乎会使用可见的提示词/系统注入来实现护栏和缓解上下文漂移，这进一步说明封闭式聊天产品在推理之外还包含运行时干预。另一位评论者则质疑围绕 **Claude 与 GLM-5.2** 的基准测试说法，特别是对 Claude 在 **Fable** 之外“占据主导地位”的说法提出异议；他们表示，Fable 目前尚不可用，而且可能已经无法有效评测编程能力。

  - **[一场痛苦的终结：团队打造的、真正用于生产并靠 LLM 赚钱的服务，现在终于要关停了。以下是我最后的一些“经验”。](https://www.reddit.com/r/LocalLLaMA/comments/1ukx9p1/end_of_an_agony_real_production_service_that_uses/)**（活跃度：394）：**一个团队正在关闭一款用于私人诊所预约的生产级 LLM 助手。他们表示，尽管已经从直接调用 **OpenRouter** API 转向 **PydanticAI**，并尝试了 GLM/DeepSeek/Mimo/Qwen/OpenAI/Claude/Minimax，加入验证器、护栏、多 Agent 委派和各种提示词，系统仍然持续出现可靠性问题。报告的故障包括：服务商宕机/返回空响应；重试后仍产生无效的结构化 `Pydantic` 输出；因表情符号/风格触发人格漂移；不安全的自主工具调用，例如用户要求 `10:00` 却预约成 `11:00`，或取消已有预约；非英语数据中的 RAG 检索错误；虚构地址/费用；以及 Agent 委派时产生幻觉。作者估计，即使成功率达到约 `95%` 仍然不够，因为剩下的错误需要持续人工监控。作者总结认为，LLM 适合第一方/个人工作流，但在面向有第三方终端用户的第二方服务中风险很高，尤其是在 CRM/数据质量和系统集成受限的情况下；此前的背景信息见他们的[早期帖子](https://www.reddit.com/r/LocalLLaMA/comments/1orw0fz/ive_been_trying_to_make_a_real_production_service/)。**置顶评论者认为，这些问题主要是架构/运行框架缺陷，而不是模型本身的固有限制：破坏性工具调用应要求人工参与确认，OpenRouter 不适合敏感的医疗工作流且路由不可靠，而记录精确的 Agent/工具调用流很可能能够暴露具体 bug。另一位评论者称，采用基于 Qwen 的商业定制运行框架，配合强提示词以及工作流/循环控制器，可以实现高得多的一致性；还有一位用户表示自己也有类似的负面经历。

    - 一些评论者认为，报告中的故障更可能是 **Agent 运行框架/设计问题**，而不是模型能力本身的限制：破坏性工具调用应当要求**人工参与审批**，应保存 Agent 状态检查点，以便准确查看各步骤之间传递了什么内容，还应使用工作流/循环控制器来限制行为。一位评论者表示，在使用更强的模型（例如 **Claude**）和受控的编排层时，他们让配备数十个自定义工具的生产级 Agent 达到了 `99.9%` 的可靠性。
    - 多条评论警告不要在生产环境中使用 **OpenRouter**，尤其是涉及敏感医疗数据时，因为其路由机制可能让人无法确认实际使用的模型权重、后端、量化级别，甚至无法确认数据在哪个司法管辖区内处理。他们指出，模式/工具调用转换层可能会破坏结构化输出保证，而质量较差或经过重度量化的模型变体，也可能解释诸如不恰当情绪化续写等异常行为。
    - 评论者强调，只要配置正确，通过服务商原生的模式约束解码或严格的结构化输出 API，**结构化 JSON 输出已经被认为是一个解决了的问题**。他们建议，先使用 **OpenAI**、**Gemini** 或 **Claude** 等可靠的闭源 API 验证工作流，等提示词、模式和控制循环稳定后，再迁移到开放权重模型或自托管方案，例如在租用的 GPU 上运行 **vLLM**。







## Less Technical AI Subreddit Recap

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. Claude Fable 5 重新上线及安全防护

  - **[Fable 5 回来了。](https://www.reddit.com/r/ClaudeAI/comments/1ukvjyn/fable_5_is_back/)**（活跃度：3176）：****Anthropic 表示，Fable 5 在更新网络安全防护措施后再次开放使用**。这些更新是在与美国政府讨论后进行的；Anthropic 同时表示，“绝大多数编程工作都不受影响”（[博客文章](https://www.anthropic.com/news/redeploying-fable-5)）。新的分类器可能会暂时增加对良性网络安全请求的误判，导致被标记的提示词回退到 **Opus 4.8**；生物学和化学分类器没有变化，仍然足够严格，连一些基础的生物学相关提示词也可能触发回退。包含用量的付费套餐可以在 **7 月 7 日**之前使用 Fable 5，但上限为每周用量限制的 **`50%`**；超出部分则可通过用量额度使用（[支持说明](https://support.claude.com/en/articles/15424964-claude-fable-5-promotional-access)）。**评论大多不涉及技术细节：用户们表示很期待在促销期间集中使用 Fable 5，但也有人担心促销结束后按用量额度计费的价格可能会让许多用户负担不起这个模型。

    - 一个与技术相关的担忧是，**如果访问权限改为按用量额度计费，Fable 5 对许多用户来说可能很快就会变得难以持续使用**，从而让长期测试或高负载工作变得成本过高。一位评论者还提到改用 `GLM-5.2`，但该讨论串没有提供 Fable 5 与 GLM-5.2 之间的**基准测试数据、实现细节或定性性能对比**。

  - **[Anthropic 的安全防护又搞砸了](https://www.reddit.com/r/singularity/comments/1ulizqk/anthropic_guardrails_does_it_again/)**（活跃度：2889）：**这张图片（[jpeg](https://i.redd.it/0tea79l1otah1.jpeg)是一篇 X 帖子的截图，内容声称 Anthropic/Claude 存在出人意料的路由费用：据称，一次选择了 **“Claude Fable 5”** 的会话产生了总计 **`$321.53`** 的费用，而实际使用却被悄悄路由到了 **Claude Opus 4.8**。结合标题“Anthropic 的安全防护又搞砸了”，这里的技术问题在于模型编排缺乏透明度：用户以为自己选择了一个模型层级，但后端的安全防护或编排机制可能调用更昂贵的模型，从而显著改变用户对费用和性能的预期。**评论者大多对自动路由持怀疑态度，认为如果用户选择了 Fable，就不应该在没有明确同意或控制选项的情况下被路由到 Opus。还有人把这种机制称为“Opus 三明治”，意思是较便宜的模型编排仍然高度依赖昂贵的 Opus 调用。

    - 多位评论者讨论了 **Anthropic 的模型路由/回退行为**，声称原本发往 **Fable** 的请求可能会被重定向到 **Opus**；如果用户明确选择了更便宜或不同的模型，他们认为这种做法并不理想。关键的技术问题是模型选择失去确定性：*“如果我想用 Fable，我就想用 Fable，别把我路由到一个更差的模型。”*
    - 有人提出了定价和缓存方面的担忧：如果 **Fable 会重定向到 Opus**，那么被路由的部分可能会按照 **Opus 的价格**向用户收费；而在不同模型之间转移或重新处理上下文时，还可能额外发生**缓存未命中**。这意味着，如果跨回退边界无法保留上下文缓存，通过 Fable/Sonnet/Opus 进行编排可能会带来隐藏的延迟和成本。
    - 一位评论者建议通过设置 `fallback=false` 来关闭自动路由，这说明 Anthropic 可能提供了防止回退或模型替换的配置开关。对于要求严格固定模型身份，而不希望由服务商管理安全防护路由的用户来说，这是讨论中提到的最具体的缓解措施。

  - **[Fable 5 在网页界面泄露了思维链，而且这种絮叨既让人不安又有点可爱](https://www.reddit.com/r/ClaudeAI/comments/1ul1396/fable_5_leaked_chainofthought_in_web_interface/)**（活跃度：2277）：**一位用户称，在网页界面测试 **Fable 5** 处理高难度竞赛编程题时，模型似乎泄露了类似隐藏思维链的文本。测试最初使用的是 [Codeforces 2237H](https://codeforces.com/contest/2237/problem/H)，之后换成了更简单的 [Codeforces 2239D](https://codeforces.com/contest/2239/problem/D)。据称，模型没有解决第二道题，反而输出了类似内部文本或调试式生成内容的絮叨片段，例如 *“GRRR.”*、*“DATA DATA DATA. GO.”*、*“GAAAH”* 和 *“PHEW”*。这似乎说明网页界面或模型端未能抑制中间推理内容或调试式生成。**评论大多只是表达反应，没有进行分析；其中一位用户将其与 Grok 在调试 WPF/.NET Syncfusion tree-grid 应用时输出 *“HELP ME I AM IN HELL”* 的情况相提并论。



### 2. Claude 模型能力基准测试



  - **[Claude Sonnet 5 vs 4.6 on arena.ai](https://www.reddit.com/r/ClaudeAI/comments/1uloomx/claude_sonnet_5_vs_46_on_arenaai/)**（Activity：986）：**这是一张 [Arena.ai Text Arena 雷达图](https://i.redd.it/3tkd721ppuah1.png)，比较了 **Claude Sonnet 5** 和 **Claude Sonnet 4.6** 在 Overall、Math、Creative Writing、Instruction Following、Multi-Turn、Legal & Government 以及 Software & IT 等类别中的表现。从图表来看，这可能意味着一次**性能回退或不均衡的升级**：**Sonnet 4.6 在许多文本和职业类基准上似乎更强**，而 Sonnet 5 只是在部分写作和语言相关领域持平或领先。**评论者们争论 Anthropic 是否正在重新定位其模型产品线，有人猜测 Sonnet 可能会转向更快、更轻量的档位，而 Opus 或未来的“Fable”模型将承担前沿性能。也有人认为，这张图表明 Anthropic 的中端模型在竞争中变弱了，并提到更便宜的竞争对手（如 GLM 5.2），同时质疑 Anthropic 为什么要以这样的状态发布 Sonnet 5。

    - 一位评论者认为，arena.ai 的图表在方法论上存在不足，因为它显示的似乎是**匿名偏好投票得出的排名位置**，而不是直接的任务性能指标，因此用它来比较 Claude Sonnet 5 和 4.6 可能会产生误导。他指出，其他基准测试据报道显示 **Sonnet 5 领先于 4.6**，所以更有力的技术批评或许不在于原始能力是否回退，而在于成本与性能的比较。
    - 一项技术与价格性能对比声称，**GLM 5.2 的表现优于 Claude Sonnet 5，但价格大约只有后者的 `1/5`**，这意味着 Anthropic 的优势可能主要集中在高端模型，而不是中端产品。评论者认为，如果这种优势无法覆盖小型、中型和大型模型档位，就说明 Anthropic 的领先幅度其实更加有限。
    - 有人猜测 Anthropic 可能正在重新定位其产品线：由 **Fable** 作为新的前沿模型，**Opus 5** 作为均衡型模型，而 **Sonnet** 则转向类似 Haiku、但推理能力更强的快速轻量档位。同一位评论者认为，首发折扣和限时优惠可能是在成本受到开放权重模型竞争压力的情况下，用来缓和未来涨价影响的策略。

  - **[太惊人了](https://www.reddit.com/r/ClaudeAI/comments/1ukyea0/its_amazing/)**（Activity：902）：**一位用户称，**Fable** 读取了一份模糊的俄语老式航空航天操作手册扫描 PDF，仅用约 `2 分钟`，就完成了原本需要约 `8 个月`人工处理的工作：提取飞机性能与操纵数据，解读旧式气动极线和不常见的 `%MAC` 图表，并计算出与用户此前结果一致、甚至能纠正此前计算的数值。相比之下，他们表示 **Opus 4.8** 无法像 Fable 一样在上下文中处理整本手册，只能逐张扫描处理；另一位评论者则称，Fable 扫描了 **Factorio** 的 AppData/mod 文件夹，并在 `3–4 分钟`内生成了一个可用的兼容性补丁 Mod。**拥有早期使用权限的评论者认为，Fable“比其他所有产品都高出一个档次”，并表示那些持怀疑态度的人大多没有进行过足够深入的实际使用。整个讨论串总体上都非常惊叹，只有一条轻微的题外评论认为，如果不涉及软件或工程领域，这类能力带来的兴奋感会低一些。

    - 一位用户分享了一个具体的编程与 Mod 制作流程：将 **Fable** 指向完整的 **Factorio AppData mods 文件夹**，用于诊断 Mod 兼容性问题，随后在大约 `3–4 分钟`内生成了一个可用的“补丁 Mod”。值得注意的技术点是：它能够端到端地读取本地 Mod 目录，并成功生成代码和配置，无需反复调试：*“我进入游戏，启用它，然后一切都修好了。”*
    - 另一位曾长期使用 Fable 的评论者称，**Fable“比市面上的其他产品都高出一个档次”**，并将其与那些认为 Fable 只是炒作的人作对比。虽然这一说法没有基准测试支持，但整个讨论串认为 Fable 的优势主要体现在实际的 Agent 式项目工作上，而不是孤立的聊天或基准测试性能。



- **[好吧，我承认了。事到如今，Fable 已经强到让我开始怀疑：除了“你比 Fable 便宜……暂时如此”之外，我这个软件工程师还有什么存在价值？](https://www.reddit.com/r/ClaudeCode/comments/1ul74ti/ok_ill_admit_it_at_this_point_fable_is_good/)**（活跃度：2991）：**这篇帖子称，**Fable** 在软件开发任务上的能力已经强到让作者很难找到它无法完成的提示，并提出了一个压力测试：只用一句指令——*“把这个游戏移植到 Godot。让它在功能上保持一致。”*——尝试将一个混乱、插件繁多的 [Unity](https://unity.com/) 游戏一次性移植到 [Godot](https://godotengine.org/)。技术圈中最主要的反驳观点是：LLM 编程 Agent 确实可以生成或修改代码，但仍需要有经验的操作者来验证架构、隐藏假设、运行时行为以及生产环境约束。**评论者普遍不认同编程 Agent 能够取代资深工程师的判断力：一则事故响应案例提到，**Claude** 因缺乏深层系统背景而给出了误导性建议，其中包括某个 SMS 任务在代码中被命名为 `send_mail`，以及建议将 Worker 扩容到一个会耗尽 [AlloyDB](https://cloud.google.com/alloydb) 连接的数值。核心争论与其说是“AI 能不能写代码”，不如说是：在上下文不完整、存在遗留系统怪癖且生产环境压力很大的情况下，如果没有称职的工程师参与，AI 是否能安全地进行推理。

    - 多位评论者认为，目前的编程 Agent 仍需要经验丰富的工程师提供架构和上下文判断，尤其是在处理生产事故时。一则详细的事故案例描述了 **Claude** 因不了解遗留领域背景而给出误导性建议：短信处理路径被命名为 `send_mail`，另一个“古怪”的子系统虽然明知有问题，却是系统有意保留的设计；此外，Claude 建议调整 Worker 数量，而这会耗尽 **AlloyDB** 的连接。
    - 一位游戏开发者分享了使用 **raylib** 配合 AI 的褒贬不一的体验：AI 在基础实现细节上会犯错，但也能生成一个可运行的 `3D voxel sphere` 行星。这说明它在相对封闭的几何和数学任务上能力很强，但在小众引擎的特定工作流中可靠性较弱。
    - 多条评论强调，像 *“把这个游戏移植到 Godot，让它在功能上保持一致”* 这样的宽泛提示很可能信息不足；真正困难的不是生成代码，而是在不同引擎语义、边界情况和隐藏的行为假设之间保持功能正确。评论者还指出，小众领域仍是当前模型的薄弱环节：一旦缺少必要上下文或遇到不常见的 API，输出质量就可能迅速下降。



### 3. Anthropic 与 AGI 招聘攻势

  - **[Anthropic 现在盯上制药行业了](https://www.reddit.com/r/singularity/comments/1ulueu6/anthropic_is_now_after_pharma/)**（热度：1129）：**图片是一篇 **STAT+ 生物科技文章**的截图，标题为 *“AI 公司 Anthropic 宣布将开始自主开发药物”*。文章称，**Anthropic** 计划开展内部药物研发，公司高管认为，亲自使用 **Claude Science** 不仅能改进产品，还能为下游生物科技领域创造价值。从技术和背景来看，这一点颇值得关注，因为它表明 Anthropic 可能正从单纯提供 AI 工具，转向**垂直化的科学研发**，并有可能利用自家模型进行假设生成、文献分析、靶点发现或药物开发流程。 [图片](https://i.redd.it/rwu2aqnqrvah1.jpeg)** 评论大多比较轻松，或以玩笑为主，并不涉及太多技术讨论；有评论者认为，如果 Claude 能加速研究，这显然是一种扩展收入来源的策略，另一些人则拿具有成瘾性或虚构的药物开玩笑，比如“Claude Crack”和“Skooma”。

    - 评论者推测，**Anthropic 进军制药业**是 Claude 合理的变现路径：将前沿模型应用于科研流程，有望在通用聊天机器人之外创造企业收入，尤其是在药物发现或生物医学研发支持方面。
    - 有人提出了一个与技术相关的担忧：Anthropic 可能因为生物安全或双重用途风险等因素，**限制了与生物学相关的提示词**，这可能会影响 Claude 在正规制药研发流程中的实用性。
    - 一位评论者认为，如果 Anthropic 能将大规模算力用于**新药发现**，这可能带来重大的战略和经济价值；其背后的含义是，前沿模型的推理和训练基础设施可以重新用于高价值的生物医学搜索、筛选或假设生成任务。

  - **[Anthropic 最近正在全力组建 AGI 团队](https://www.reddit.com/r/singularity/comments/1ukuahd/anthropic_is_on_a_mission_rn_to_make_agi_team/)**（热度：1946）：**图片是一条推文的截图，内容提到 **Jelani Nelson** 已加入 **Anthropic**，并从大学请假；Nelson 是 **UC Berkeley EECS** 的负责人，也是一位知名的理论计算机科学与算法研究者，曾在 **MIT、IAS、Princeton、Harvard 和 Berkeley** 任职（[图片](https://i.redd.it/3514i28yynah1.jpeg)）。Reddit 标题将此事描述为 Anthropic 正在“全力组建”AGI 团队。从技术角度看，这件事的重要性不在于发布了某个模型或基准测试，而在于 Anthropic 招募了算法与理论方向的资深学术人才，这可能会增强其在可扩展机器学习系统、优化和基础理论方面的研究能力。** 评论者普遍认为，这次招聘说明 Anthropic 资金实力雄厚，并且正在积极招揽顶尖研究人员；有些人还毫无证据地猜测，公司正在秘密开展某种 AGI“曼哈顿计划”。另一些人则更多是从个人角度发表评论，提到 Nelson 备受好评的算法课程，并称赞这是一次强力招聘。





# AI Discord 社区

很遗憾，Discord 今天终止了我们的访问权限。我们不会再以这种形式恢复它，但很快会推出全新的 AINews。感谢你一直读到这里，这段旅程曾经很美好。