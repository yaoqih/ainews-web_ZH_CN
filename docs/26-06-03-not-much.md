---
companies:
- microsoft
- google
- vllm-project
- ollama
- llama-cpp
date: '2026-06-02T05:44:39.731046Z'
description: '**微软**发布了 **MAI-Thinking-1** 的详细技术报告。这是一款在没有第三方蒸馏的情况下训练出的通用推理模型，在 AIME
  2025 测试中取得了 **97%** 的高分，并在人类偏好测试中超越了 Sonnet 4.6。该报告因其透明度而受到赞誉，披露了该模型未使用合成数据，采用了独特的“缩放阶梯”（scaling
  ladder）方案，并详细列出了训练数据构成，其中包括 **50% 的代码**和 **17.5% 的 STEM 内容**。此外，微软还推出了用于特定工作流模型适配的
  **Frontier Tuning**，声称在 Excel 任务中效率提升高达 **10 倍**，且质量达到 GPT-5.4 级别。同时发布的还有 **MAI-Image-2.5**
  和 **MAI-Code-1-Flash** 等新模型。


  与此同时，**谷歌**推出了 **Gemma 4 12B**。这是一款基于 Apache 2.0 协议的多模态模型，采用创新的无编码器（encoder-free）架构，专为
  **16GB 显存**的端侧设备设计。它将视觉和音频编码器整合进 LLM 主干网络中，获得了社区的积极反馈以及即时的工具支持。'
id: MjAyNS0x
models:
- mai-thinking-1
- mai-image-2.5
- mai-code-1-flash
- gemma-4-12b
people:
- eliebakouch
- nrehiew_
- mustafasuleyman
- minjiyoon90
- lateinteraction
- harold_matmul
- googlegemma
- googleaidevs
- mtschannen
- armandjoulin
- osanseviero
title: 今天没发生什么特别的事。
topics:
- model-training
- reinforcement-learning
- model-architecture
- multimodality
- model-deployment
- model-efficiency
- fine-tuning
- on-device-ai
---

**平静的一天。**

> 2026年6月2日至2026年6月3日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter](https://twitter.com/i/lists/1585430245762441216)，没有发现更多的 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择开启/关闭](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 摘要


**Microsoft 的 MAI-Thinking-1 技术报告、训练栈和 Frontier-Tuning 推送**

- **MAI-Thinking-1 是当天内容最丰富的技术发布**：Microsoft 推出了 [**MAI-Thinking-1**](https://x.com/asadovsky/status/2062008312603070891)，这是一个在**没有第三方蒸馏 (distillation)** 的情况下训练的通用/推理模型，报告显示其在 **AIME 2025 上达到 97%**，在 **SWE-Bench Pro 上达到 53%**，并在盲测对比中人类偏好胜过 Sonnet 4.6。这份 109 页的报告因其非同寻常的透明度受到了 [@eliebakouch](https://x.com/eliebakouch/status/2061965825037254947)、[@nrehiew_](https://x.com/nrehiew_/status/2062013300196700395) 和 [@mustafasuleyman](https://x.com/mustafasuleyman/status/2062253941207761180) 的广泛赞誉。主要技术主题是：Microsoft 似乎“从零开始完成了爬坡优化 (hillclimbed from scratch)”，[@MinjiYoon90](https://x.com/MinjiYoon90/status/2062058684730245376) 明确以这种方式定义了这项工作。
- **研究人员关注该报告的原因**：被引用最多的细节不仅是基准测试的质量，还有发布的系统/训练信息。[@eliebakouch](https://x.com/eliebakouch/status/2061965825037254947) 强调了**零合成数据和零先前模型蒸馏**，这意味着推理、工具使用和 Agent 行为是在后训练 (post-training) 中学习的，没有使用合成数据的“冷启动”。推文串还特别提到了**扩展梯级配方 (scaling ladder recipe)**、精确的 **MFU 数值**以及目标损失构建 (target-loss construction) 的发布。在后续内容中，[@eliebakouch](https://x.com/eliebakouch/status/2061976608265880004) 指出其私有 NLL 混合权重的比例为 **50% 代码、17.5% STEM、17.5% 数学、10% 通用知识、5% 多语言**，并针对内部模型进行了归一化；他还指出了针对其 MoE 设置在 **100–200 TPP** 左右的消融实验 (ablations)，详见[此处](https://x.com/eliebakouch/status/2061975730414633043)。社区摘要中还出现了其他值得注意的实现细节：根据 [@eliebakouch](https://x.com/eliebakouch/status/2062002698363232401) 的说法，Microsoft 在部分技术栈中使用了 **SGLang**；根据 [@lateinteraction](https://x.com/lateinteraction/status/2062015109132873852) 和 [@harold_matmul](https://x.com/harold_matmul/status/2062040746027315714) 的说法，使用了 **dspy.GEPA** 进行预训练数据策展 (data curation)。
- **Microsoft 的产品化视角超越了单个模型**：在发布报告的同时，Microsoft 推出了更广泛的“拥有你自己的模型 (own your model)”故事。[@mustafasuleyman](https://x.com/mustafasuleyman/status/2062275417378041957) 概述了 **Frontier Tuning**，它以强化学习 (reinforcement-learning) 环境为核心，旨在进行特定工作流的适配，并声称内部面向 Excel 的 MAI 微调模型在相关任务上可以达到 GPT-5.4 级别的质量，同时**效率提升高达 10 倍**。Build 大会的发布内容还包括 [**MAI-Image-2.5**](https://x.com/MicrosoftAI/status/2062240400299934143)（Microsoft 称其在 **text-to-image 排行榜排名第 3**，在 **image-to-image Arena 排行榜排名第 2**）、[MAI-Code-1-Flash](https://x.com/pierceboggan/status/2062220583786709163) 以及在 OneDrive Photos 等产品中的部署。作为一个元观点 (meta-point)，这是今年最清晰的案例之一，展示了一家实验室如何尝试发布 Frontier 级别的报告，同时将该技术栈转化为企业定制化基础设施。

**开源模型发布：Gemma 4 12B、Ideogram 4.0、Miso One 以及本地优先 (Local-First) 的势头**

- **Gemma 4 12B 是最受瞩目的开源模型发布**：Google 发布了 [**Gemma 4 12B**](https://x.com/Google/status/2062203526588088452)，这是一款基于 **Apache 2.0** 协议的多模态模型，旨在设备端运行，大约需要 **16GB VRAM**。其架构上的创新在于 **encoder-free**（无编码器）设计：没有独立的视觉或音频塔。正如 [Google 所解释的](https://x.com/Google/status/2062203532351090824)，图像通过轻量级嵌入模块处理，而原始音频则直接投影到文本 token 空间。社区反应集中在将多模态编码器整合进 LLM 骨干网络的优雅设计上，[@googlegemma](https://x.com/googlegemma/status/2062202706882883696)、[@googleaidevs](https://x.com/googleaidevs/status/2062204432658386950)、[@mtschannen](https://x.com/mtschannen/status/2062236357351579915) 和 [@armandjoulin](https://x.com/armandjoulin/status/2062206784647967075) 都强调了这一点。工具链支持立即覆盖了 [vLLM](https://x.com/vllm_project/status/2062228047324201166)、[Ollama](https://x.com/ollama/status/2062250522598572345)、通过 [@osanseviero](https://x.com/osanseviero/status/2062205176597889220) 支持的 llama.cpp/MLX，以及据报道其量化版本仅需 **8GB RAM** 即可实现本地运行的 [Unsloth GGUFs](https://x.com/UnslothAI/status/2062207258810053084)。
- **Ideogram 转向开源权重的重要性不亚于模型本身**：[Ideogram 4.0](https://x.com/ideogram_ai/status/2062202208700313872) 被宣布为“世界上最好的开源图像模型”，拥有开源权重并可通过 [fal](https://x.com/fal/status/2062202673361780873) 和 Hugging Face（[此处](https://x.com/huggingface/status/2062206083914158287)）立即部署。Arena 迅速将 [Ideogram-4.0-Quality 排在总榜第 8 位，并在开源模型中排名第 1](https://x.com/arena/status/2062203346996605116)，特别是在**文本渲染**和**品牌/商业设计**方面取得了显著进步。这次开源发布获得了极大关注，因为 Ideogram 此前被认为高度侧重设计但并不开放；[@multimodalart](https://x.com/multimodalart/status/2062210597148930139) 和 [@cloneofsimo](https://x.com/cloneofsimo/status/2062210832440918309) 都注意到了这一转变。
- **开源音频领域也有强劲表现**：[**Miso One**](https://x.com/kimmonismus/status/2062210845308780639) 作为一款 **8B 开源权重 TTS 模型**发布，具有 **one-shot 语音克隆**功能，并声称具有 **110ms 的延迟**，旨在实现更具表现力的配音。阿里巴巴的 [Fun-Realtime-TTS](https://x.com/ArtificialAnlys/status/2062016529848222073) 也在 **Artificial Analysis 的 Speech Arena 中夺得第 1 名**，Elo 分数为 **1219**，领先于 Gemini 3.1 Flash TTS 和 Inworld，价格为 **$27.59 / 100万字符**。另外，[Google 的 Magenta RealTime 2](https://x.com/HuggingPapers/status/2062260306039259236) 也作为一款适用于设备端的开源权重、低延迟连续音乐生成器而受到关注。
- **更宏观的趋势是本地 AI 正在成为主流部署目标**：[@ggerganov](https://x.com/ggerganov/status/2062193382605111386) 指出 Computex 是 **本地 AI 工作负载** 的一个强信号；[@rasbt](https://x.com/rasbt/status/2062235700636873082) 同样指向了一个不断增长的开源权重和消费级硬件生态系统。微软 [Surface Laptop Ultra](https://x.com/kimmonismus/status/2062201523963084864) 的卖点——高达 **1 PFLOP AI 算力**、**128GB 统一内存**、RTX GPU——从硬件层面契合了这一趋势。

**Agent、Harness 以及从框架到执行层的转变**

- **重心正从“框架”向 Agent Harnesses 和执行环境转移**：多篇帖子都汇聚到了同一个观点。[@gakonst](https://x.com/gakonst/status/2062116487708512355) 认为未来的 IDE 栈不再仅仅关乎代码编辑器，更多在于用线程（threads）取代文件，并将计划/设计/构建/部署/监控循环进行捆绑——这使得 **协作/同步引擎（collaboration/sync engines）** 成为一个尚未解决的关键问题。在另一篇互补的采访摘要中，[@ConorBronsdon](https://x.com/ConorBronsdon/status/2062224321381323218) 报道了 Jerry Liu 的观点，即“框架时代”正在结束，抽象层正在向上移至 **技能、工具和上下文质量（context quality）**，而非 Python 封装器。
- **多 Agent 和 Agent 优化工作变得更加具体**：CMU/LTI 的 [**MACU**](https://x.com/rsalakhu/status/2062194674794668066) 和 [@kohjingyu 的帖子](https://x.com/kohjingyu/status/2062179533009178897) 主张，Computer-use Agent 应被设计为 **基于 DAG 的多 Agent 系统**，由管理 Agent 分解任务并分派给并行的子 Agent。据报道，该方案在各项基准测试中提升了 **4.7–25.5%**，在 Odysseys 上的完成速度快了 **1.5 倍**。在优化方面，Microsoft 的 **SkillOpt** 得到了 [@omarsar0](https://x.com/omarsar0/status/2062204469538881988) 的实际验证，他表示将其接入编排器后，某项多模态提取技能的表现从 **0.73 提升至 0.93**。
- **Agent UX 和部署工具正在成为独立的产品**：Nous 的 Hermes Agent 更新引起了强烈关注，包括[此处](https://x.com/Teknium/status/2061984430370267210)的远程连接修复、[此处](https://x.com/Teknium/status/2062170975949721612)更新的远程指南，以及[此处](https://x.com/Teknium/status/2062315666439655499)更大幅度的仪表盘改版。Perplexity 推出了 [**Personal Computer for Windows**](https://x.com/perplexity_ai/status/2062189045728596080)，这是一款针对应用和文件的设备端编排器；而 [Cloudflare Browser Run 远程标签页](https://x.com/BraydenWilmoth/status/2062180110208311558) 则展示了一种更具 Agent 原生特性的浏览器控制路径。LangChain/LangSmith 在可观测性和成本控制层发力，推出了 [Gateway 支出追踪](https://x.com/LangChain/status/2062188019784835559)、[Sandbox/Gateway/Observability 文档](https://x.com/hwchase17/status/2062144718427857256)，以及围绕 Deep Agents 和 LangSmith 的[案例研究](https://x.com/LangChain/status/2062204592562073972)。

**路由、成本控制以及开源 vs 前沿模型部署策略**

- **模型路由（Model routing）现在是一个真正的辩论焦点，而非口号**：[@levie](https://x.com/levie/status/2061974298760495132) 认为，随着 Token 预算成为一项重要的运营支出（opex）类别，**模型路由是不可避免的**，而特定领域的评估（evals）将成为核心竞争力。但 [@scottastevenson](https://x.com/scottastevenson/status/2062042036774314107) 表达了强烈的反对意见，称目前大多数路由产品都是“蛇油”（snake oil）：如果前沿模型（frontier models）能避免重试，其综合表现可能更好、更快、更便宜；路由可能会使紧耦合系统变得不稳定；且 API 厂商通常可以内化明显的套利空间。[@fabianstelzer](https://x.com/fabianstelzer/status/2062051511484465351) 补充道，缓存写入以及 Harness-模型-提示词的匹配度可能会抵消预期的成本节省。
- **企业用户开始强制执行严格的成本上限**：[@simonw](https://x.com/simonw/status/2062143151184465964) 强调了有关 Uber 将每个员工每个工具的编程 Agent 支出限制在 **1,500 美元/月** 的报告。LangChain 立即将其归纳为 [LangSmith Gateway](https://x.com/hwchase17/status/2062208385890570565) 的典型用例。[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2062225912662561106) 捕捉到了更广泛的情绪：某些组织可能很快会面临三选一：要么让每个人“Token 化最大化（tokenmaxx）”，要么限制预算，要么裁员并将支出重新分配给生产力最高的 AI 赋能员工。
- **混合/开源策略的真实数据点开始出现**：Harvey 的基准测试结果是最好的例子。在一项研究中，[Harvey](https://x.com/harvey/status/2062218656420167785) 发现一个混合型法律 Agent（以 **GLM 5.1** 为主要执行者，**Opus 4.7** 为建议者）在全通率（all-pass rate）上击败了纯 Opus（**18% vs 14%**），同时在 100 个任务中的成本仅为 **368 美元（对比 954 美元）**。Harvey 还报告称，SFT（有监督微调）可以使 **Kimi 2.6** 的表现从 **11% 提升至 15%**，以约 **11 倍的成本优势** 击败 Opus。另一方面，[@ClementDelangue](https://x.com/ClementDelangue/status/2062248714945630632) 认为路由加上经过后训练（post-trained）的开源模型通常会在成本、速度和控制力上胜出，而 [@ypatil125](https://x.com/ypatil125/status/2062196581936529721) 则将开源模型和开源模型云视为重要工作负载最终默认选择的先行指标。

**热门推文（按互动量排序）**

- **Gemma 4 12B 发布**：[@googlegemma](https://x.com/googlegemma/status/2062202706882883696) 和 [@Google](https://x.com/Google/status/2062203526588088452) 凭借这一无编码器（encoder-free）多模态版本的发布，引发了极高的技术关注度。
- **Ideogram 4.0 开放权重**：[@ideogram_ai](https://x.com/ideogram_ai/status/2062202208700313872) 宣布了一个显著转变，从强大的闭源图像模型转向开放权重（open weights）。
- **MAI-Thinking-1 透明度**：[@eliebakouch 的推文内容](https://x.com/eliebakouch/status/2061965825037254947) 成为 MAI 报告中极具影响力的技术阅读指南。
- **用于生命科学的 Rosalind**：OpenAI 的 [GPT-Rosalind 更新](https://x.com/OpenAI/status/2062281977122996256) 标志着前沿模型在领域特定科学研究中的进一步垂直化。
- **开源音频/TTS 势头**：[阿里巴巴的 Fun-Realtime-TTS](https://x.com/ArtificialAnlys/status/2062016529848222073) 和 [Miso One](https://x.com/kimmonismus/status/2062210845308780639) 脱颖而出，成为极具实用价值的发布，而非仅仅是研究演示。


---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Gemma 4 多模态开源模型

  - **[google/gemma-4-12B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1tvtn6m/googlegemma412b_hugging_face/)** (热度: 1293): **Google DeepMind** 发布了 [`google/gemma-4-12B`](https://huggingface.co/google/gemma-4-12B)，这是一个采用 **Apache-2.0 协议的开放权重多模态** Gemma 4 模型。它采用了 `12B` 无编码器（encoder-free）/统一的 decoder-only 架构，将原始图像块（image patches）和音频波形直接投影到 LLM 嵌入空间中。Gemma 4 家族涵盖了稠密型和 MoE 变体（`E2B`, `E4B`, `12B`, `26B A4B`, `31B`），支持高达 `256K` 上下文，具备 p-RoPE/统一 KV 的混合局部/全局注意力机制，原生支持 `system` 角色、function calling、可配置的推理/思考功能，并支持文本/图像/音频/视频帧输入以及文本输出；GGUF 构建版本可从 [`ggml-org`](https://huggingface.co/ggml-org/gemma-4-12b-it-GGUF) 和 [`unsloth`](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) 获取。相关的技术指南强调了该模型的 *“无编码器架构”* 以及通过 `transformers` 库使用 `AutoProcessor` 和 `AutoModelForMultimodalLM` 的实现路径 ([指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b), [Google 开发者博客](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/))。评论者主要关注实际的基准测试，特别是 Gemma 4 12B 在编程任务上是否能超越 **Qwen 3.5 9B**，并指出无编码器多模态设计在技术上非常有趣。

    - **Maarten Grootendorst** 分享了 **Gemma 4 12B** 的技术指南，强调该模型采用了 **无编码器架构（encoder-free architecture）**，这对于关注多模态/模型架构设计的读者来说非常值得注意：https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b
    - 几位评论者将 **Gemma 4 12B** 视为较小变体（如 **E4B**）和较大模型（如 **26B**）之间的一个潜在且实用的尺寸/性能平衡点，并特别关注它在编程工作负载中与 **Qwen 3.5 9B** 的对比。
    - 提出的一个技术点是该模型明显的 **音频能力**，有人推测这可能使 **Gemma 4 12B** 在语音/音频翻译工作流中发挥作用，而不仅仅局限于文本或视觉-语言任务。

  - **[最小且最高质量的 Gemma4 E2B 和 E4B！开源！7 倍压缩！](https://www.reddit.com/r/LocalLLM/comments/1tuyj0o/the_smallest_and_highest_quality_gemma4_e2b_and/)** (热度: 353): **TheStageAI** 通过 [`edge-lm`](https://github.com/TheStageAI/edge-lm) 发布了兼容 MLX 的压缩版 **Gemma 4 Edge** 检查点：`gemma-4-E2B-it` 大小为 **`1.44 GB`**，`gemma-4-E4B-it` 为 **`2.72 GB`**，声称在保留基准测试质量的同时，体积比 BF16 减少了 **`6.4–7 倍`**。相关的[博客文章](https://app.thestage.ai/blog/7x-size-reduction-for-Gemma4-Edge-models?id=14)将压缩归功于 **针对 PLE 表的 AQLM 风格向量量化**、**通过黎曼约束优化（Riemannian Constrained Optimization）实现的逐层混合比特量化** 以及 **量化误差传播（Quantization Error Propagation）**；据报告，在 M3 Max 上 Apple Silicon 的性能表现包括 E2B 达到约 **`115 tok/s`**，峰值 MLX 显存占用为 **`2.1 GB`**。评论者集中讨论了其对本地推理的意义，特别是如果类似的压缩方法奏效，更大的 Gemma 变体（如 **31B**）可能能够适配 **16 GB** 的系统。一个讨论贴将此发布视为本地模型快速进步的证据，可能会动摇以云为中心的 AI 假设。

- 详细的技术解释将 `~7x` 的压缩归功于三种方法：对 Gemma 的大型每层嵌入/PLE 表进行矢量量化，将其从 `4.7 GB` 减小到 `0.26 GB`；通过 Riemannian Constrained Optimization 进行混合精度分配，为不敏感层分配较低的位宽；以及使用 Quantization Error Propagation 来补偿跨层累积的量化误差。声称的结果是一个 `1.44 GB` 的模型，在保持指令遵循和编程质量的同时，能够适配移动端/Apple Silicon 的内存预算。
- 几位评论者关注运行时可移植性：该发布似乎与 **MLX** 绑定，这通常针对 Apple Silicon，引发了关于它是否可以在 **LM Studio** 中运行、是否可以转换为兼容 llama.cpp 运行时的 **GGUF**、或在 macOS/Apple 硬件之外使用的疑问。另一个技术问题是该模型是否可以在其原始的 **LiteRT** 格式下运行，暗示了对压缩产物是特定于框架还是可以导出到更广泛的推理栈的不确定性。

- **[Google 推出 Gemma 4 12B：一个统一的、无编码器的多模态模型](https://www.reddit.com/r/LocalLLM/comments/1tvx2h7/google_introduces_gemma_4_12b_a_unified/)** (Activity: 314): **Google 推出 [Gemma 4 12B](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/)**，这是一款 Apache 2.0 中型多模态模型，旨在为约 `16GB` 的消费级系统提供本地推理，声称其性能接近其较大的 `26B` MoE 模型，且内存占用不到其一半。主要的架构点是**无编码器多模态 (encoder-free multimodality)**：视觉部分被简化为一个轻量级的嵌入模块——单一矩阵乘法 + 位置嵌入/归一化——而音频部分则完全移除了编码器，将原始波形数据直接投影到与文本 Token 相同的空间；Google 还提到了 **Multi-Token Prediction drafters** 以及对 Hugging Face, Ollama, LM Studio, llama.cpp, MLX, vLLM, SGLang, Unsloth, LiteRT-LM 和 Google Cloud 的广泛支持。评论者主要在等待独立评估，特别是本地多模态质量和延迟/内存表现。一个对比讨论帖询问了 Gemma 4 12B 与更大的 Qwen 模型（如 Qwen3.6 `27B`/`35B`）相比表现如何，但可见的热门评论中尚未提供有基准测试支持的答案。

    - 公告声称 **Gemma 4 12B** 的性能接近较大的 **26B MoE**，同时使用的内存不到其一半，目标是在具有 `16GB RAM` 的消费级机器上本地运行。关键的架构细节是**无编码器多模态设计**：视觉仅使用轻量级的嵌入路径——单一矩阵乘法、位置嵌入和归一化——而音频则通过将原始音频投影到文本 Token 嵌入空间，从而完全移除了编码器。
    - 几位评论者关注 Gemma 4 12B 将如何与当前强劲的本地模型（如 **Qwen3.6 35B** 和 **Qwen3.6 27B**）进行竞争，尤其是考虑到它作为一个稠密/较小的 `12B` 模型，却声称接近 26B MoE。暗示的评估目标是标准文本基准测试以及实际的多模态/音频能力，而不仅仅是参数量。
    - 一位本地推理用户估算，**Q4 版本的 Gemma 4 12B** 将占用大约 `7GB` VRAM，在 **Radeon 9060 XT 16GB** 配置上为上下文留下了充裕空间。另一位用户表示有兴趣在 **ROCm** 上进行测试，但预计发布后需要一段时间才能实现兼容性/工具链稳定。

### 2. 本地 LLM 部署实验

- **[在我的多智能体编排器中用本地 Qwen3.6-27B 替换 Claude 两周后的总结](https://www.reddit.com/r/LocalLLaMA/comments/1tunmam/replaced_claude_with_local_qwen3627b_in_my/)** (热度: 584): **作者报告了在单块 **RTX 3090 24GB** 上，通过 **Ollama** 运行 **Qwen3.6-27B** 本地版 [**OpenYabby**](https://github.com/OpenYabby/OpenYabby) 两周的经历。该设置使用了 `Q6_K` 权重（占用约 `22GB` VRAM）、约 `32k` 的有效上下文、结构化 JSON 规划、计划审批以及跨 `47` 个多步骤编码工作流的自动审查。Qwen 在高级规划（经过 Prompt 调优后，模式校验通过率约 `~95%`）和记忆提取方面被认为可以与 Claude 竞争，但在执行/工具调用方面表现较弱：工具调用模式/签名错误率约为 `~12%`（Claude 约为 `~0.5%`），上下文在超过 `12–14k` tokens 后会出现实际漂移，并在子智能体（sub-agent）失败后出现了 `3/47` 次级联幻觉。结论是本地 Qwen 可以作为**推理/规划层**，但不应被信任作为无门控的执行层；需要严格的结构化输出强制执行、计划审批以及明确的失败重规划逻辑。** 热门评论认为观察到的失败很大程度上是由配置引起的：`Q6_K` 配合有限/量化的 KV cache 以及 Ollama 受到了批评，建议使用 **Q8_0/Q8_K_XL 权重**、**F16/BF16 KV cache**、更新的 **llama.cpp/Unsloth** 构建版本，以及大得多的上下文（`100k–160k`）。一位评论者声称，在这些设置下，Qwen3.6-27B 可以在长上下文中保持工具使用能力，但在被要求一次性分析极大的单体代码上下文（如数千行）时仍会出现退化。

    - 几位评论者指出，报告的失败可能源于运行环境/量化设置，而非 Qwen3.6-27B 本身：`Q6_K` 权重配合仅 `32k` 的有效上下文被认为不足以支撑多智能体编排，一位用户建议对于复杂的长上下文工具工作流，至少需要 `128k` 上下文和未量化的 KV cache。
    - 具有长上下文 Qwen3.6-27B 使用经验的用户建议弃用 Ollama，改用最新的 `llama.cpp`/Unsloth 构建版本，并使用更高精度的设置：**最低 Q8_0**，最好是 **Q8_K_XL**，并配合 `F16` 或 `BF16` KV cache。一位评论者报告在高达约 `160K` 上下文时工具调用依然稳定，但指出当要求模型深入分析超过 `60–70K` tokens 的极大单体输入时，质量会有所下降。
    - 另一个实现层面的担忧是 Qwen/Unsloth 分发的 **Jinja 聊天模板可能损坏**，这可能会影响 Prompt/工具调用的行为，除非更换为修复后的模板。另一位评论者提到，最近 `llama.cpp` 的更改可能允许通过使用 `Q5_1`/`Q4_1` KV cache 量化，在 `Q6` 权重下实现约 `100k` 的上下文。

  - **[我花了 200 英镑给我的游戏电脑装了一块数据中心 GPU](https://www.reddit.com/r/LocalLLaMA/comments/1tuxy5f/i_put_a_datacenter_gpu_in_my_gaming_pc_for_200/)** (热度: 547): **该帖子详细介绍了如何使用非官方的 **SXM2-to-PCIe adapter**，将一块二手 **Tesla V100 SXM2 16GB** 集成到消费级游戏电脑中，并与 **RTX 4080 16GB** 配对，以约 **£200** 的成本实现了 **32GB 总显存 (aggregate VRAM)**，用于本地 LLM 推理 ([博客](https://blog.tymscar.com/posts/v100localllm/))。该设置需要不少硬件/软件工作——自定义散热和 PWM 风扇控制、NixOS 内核/旧版 NVIDIA 驱动程序限制、CUDA 12.2 时代的兼容性，以及跨 Ada + Volta GPU 的 `llama.cpp` 张量并行分割。在两块 GPU 完全加载 **Qwen3.6-27B-MTP Q5_K_M** 的情况下，据称生成速度达到约 `32 tok/s`，提示词处理速度达到 `133–160 tok/s`。** 评论者关注退役数据中心 GPU 在本地推理中的价值，并对消费级显存细分提出质疑，特别是 **RTX 4080 仅配备 16GB VRAM** 这一点。普遍观点是，随着更新的数据中心显卡逐渐退役，廉价的二手 HBM2 硬件将变得越来越有吸引力。

- 有关 **datacenter GPU form factors** 的一个技术对比点被提出，特别是没有原生 PCIe 边缘连接器的 **SXM2 modules** 与在 **PCIe carrier cards** 上销售的版本之间的区别。实际影响在于，SXM2 卡通常需要兼容的底板/中介层（baseboard/interposer）、定制散热和供电，而 PCIe 变体虽然仍需考虑驱动、固件和散热，但更接近于即插即用的桌面端使用。
- 一位评论者强调了消费级 GPU VRAM 持续存在的限制，指出与二手市场上能以低廉价格提供更大显存池的退役数据中心卡相比，仅拥有 `16GB` VRAM 的 **RTX 4080** 显得捉襟见肘。这反映了此类装机方案中的主要技术权衡：旧款数据中心 GPU 可能在单位价格上提供极高的 VRAM 容量，但往往缺乏面向游戏的特性、显示输出、标准散热或完整的驱动支持。
- 业内对当前一代数据中心加速器退役后的未来二手市场表现出了兴趣。技术层面的预期是，拥有大容量 HBM/VRAM 的显卡对于本地 AI、渲染或计算工作负载将非常有吸引力，前提是买家能够解决平台兼容性、功耗、散热和驱动问题。

## 较少技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Ideogram 4.0 与 DR02 发布

  - **[Ideogram 4.0 正式开源！](https://www.reddit.com/r/StableDiffusion/comments/1tvtu2u/ideogram_40_just_open_sourced/)** (活跃度: 834): **该 [图片](https://i.redd.it/9ajk9fuu935h1.jpeg) 是一个针对据称发布的 Ideogram 4.0 的宣传性、非技术性展示渲染图**，强调了其文本渲染能力，带有可识别的标签如 “Ideogram”、“Now on Comfy” 和 “The Yellow Pearl”。帖子将 Ideogram 4.0 描述为一个 `9.3B` 的权重开放（open-weight）文本生成图像模型，支持 **ComfyUI**，提供 `fp8`/`nf4` Checkpoints，支持 JSON 结构化 Prompting，采用 Qwen3-VL-8B-Instruct 文本编码器，并在 OCR/布局（layout）基准测试中表现强劲。评论的焦点较少集中在宣传图上，更多关注于**模型审查/安全过滤**，用户反映了严格的 NSFW 屏蔽，并调侃 Ideogram 对该模型进行了 “safetymaxxed”（过度安全化）。一些人预期社区最终可能会移除或绕过这些限制。

    - 几位评论者报告称，开源的 **Ideogram 4.0** 版本似乎内置了非常激进的安全过滤，**comfyanonymous** 指出输出被屏蔽是因为模型被 *“safetymaxxed”*，而非 **ComfyUI** 的问题。用户特别提到了严格的 NSFW 审查，并推测该模型可能需要进行 “abliteration”/去屏蔽处理，才能在限制较少的本地工作流中发挥作用。
    - 其中一个被强调的技术亮点是 **Bounding-box JSON Prompting**，Prompt 显然可以显式指定布局区域以进行图像合成。一位评论者分享了示例截图，并称其为 *“非常酷的 Bounding-box JSON Prompt 示例”*，表明 Ideogram 4.0 除了纯文本 Prompt 外，可能还提供了结构化的空间控制。
    - 一个关于实际采用的担忧是，据报道该版本带有**水印**、被**审查**且缺乏**商业许可**，这限制了其在生产或货币化管线中的实用性。对于评估本地部署的技术用户来说，这些限制与原始生成质量或 ComfyUI 兼容性同样重要。

  - **[DeepRobotics 发布 DR02，在载重能力和复杂地形移动性方面有显著提升](https://www.reddit.com/r/singularity/comments/1tv2l9z/deeprobotics_unveils_dr02_with_significant/)** (活跃度: 816): **DeepRobotics** 据报道发布了 **DR02** 四足机器人，强调了其在复杂地形上改进的有效载荷/载重能力和移动性；然而，由于 `403 Forbidden` 错误，链接中的 Reddit 托管视频无法访问，因此无法从源头验证独立的规格、基准测试或步态/控制细节。技术讨论的中心较少关注发布本身，而更多关注移动行为：评论者质疑当前的四足机器人是在进行显式的落脚点规划（foothold planning），还是在穿越不平整岩石或不稳定表面时依赖强大的反应式平衡和恢复。一个值得注意的批评是，许多“崎岖地形”演示似乎显示机器人是在*“胡乱冲过岩石”*，而不是根据几何形状、坡度或稳定性刻意选择落脚点。另一位评论者建议在透明地板上进行测试，这将探测感知假设以及在视觉/深度传感可能失效或变得模糊时的鲁棒性。

    - 一位评论者质疑像 DR02 这样的四足机器人是在崎岖地形上使用显式的 **落脚点规划（foothold planning）**，还是主要依赖反应式稳定（reactive stabilization）。他们指出，演示看起来通常像是机器人在从不稳定或倾斜的接触中恢复的同时*“胡乱冲过岩石”*，而不是明显地根据地形几何、坡度或稳定性来选择落脚点。
    - 另一个技术相关的担忧是这些机器人将如何处理感知困难的表面，例如玻璃走廊等**透明地板**。这类环境对于基于视觉/深度的地形估算具有挑战性，是测试移动感知和落脚点（foot-placement）鲁棒性的一个有用边缘案例。


### 2. Claude Code Agentic 预览版

- **[[我通过 MCP 将 Claude Code 连接到了包含每个 Polymarket 钱包和交易的数据库。接下来你们想让我问它什么？这是我目前的发现：](https://www.reddit.com/r/ClaudeAI/comments/1tvefqd/i_wired_claude_code_into_a_database_of_every/)]** (热度: 1465): **作者声称通过 **Postgres MCP** 将 **Claude Code** 连接到了一个包含约 `13亿` 次交易和 `270万` 个钱包的实时 Polymarket 账本数据集，允许模型根据自然语言提示词生成并执行只读 SQL。报告的发现包括：约 `20%` 的钱包实现净盈利，`2.4%` 的钱包获利超过 `$1,000`，顶尖的 `0.1%` 钱包占据了约 `$10亿` 总利润的 `71.5%`；链接中的 CrowdIntel 文章描述了类似的 MCP 设置，使用了预聚合表，包含约 `156万` 个钱包，其中盈利超过 `$1,000` 的钱包有 `37,628` 个，约 `2.36万` 个机器人和约 `3,100` 个大户 ([CrowdIntel](https://crowdintel.xyz/blog/claude-mcp-polymarket-ledger))。** 热门评论敦促进行新闻调查，认为该数据集可能揭露内幕交易或其他违规行为；一位 Forbes 记者请求建立联系。技术建议包括将观察到的利润分布与公平市场/零模型进行对比，并检查大额亏损钱包/投注，看其是否属于洗钱行为而非仅仅是无知的亏损。

    - 一位评论者建议建立一个统计基准，说明在公平/无内幕交易的市场下 Polymarket 的结果*应该*是什么样子，然后将该预期分布与观察到的钱包级 PnL 和胜率分布进行比较。他们还提议检查大额亏损钱包或大额亏损投注的聚集方式是否符合潜在的洗钱特征，而非仅仅是从散户手中提取内幕利润。
    - 另一个技术问题关注数据的实时性：在 Polymarket 上下注到这些交易出现在通过 MCP 访问的数据库中之间有多长的延迟。这关系到该系统是支持近乎实时的异常检测，还是只能进行回顾性分析。
    - 一位评论者询问分析是否仅涵盖直接参与 Polymarket 交易的钱包，还是也追踪了上游资金来源和下游资金流向。这种区分对于识别协同工作的钱包集群、交易所出入金通道 (on/off-ramps) 以及可能表明共同控制或洗钱行为的交易后资金流动模式至关重要。

  - **[[我让 Opus 4.8 在不到一天的时间内构建了“Temu 版英雄联盟” —— 我称之为 LMAO](https://www.reddit.com/r/ClaudeAI/comments/1tucsfe/i_had_opus_48_build_temu_league_of_legends_in/)]** (热度: 3458): **作者报告称使用 **Claude Opus 4.8** 生成了一个名为 **LMAO** 的纯网页、基于房间的多人游戏，即“Temu 版英雄联盟”克隆版。开发从单个提示词开始，然后通过子代理进行角色/技能/SFX/VFX 设计、地图/野怪/小兵迭代，并使用 **Ultracode Workflows** 进行性能、平衡和杂项优化。他们还大量使用了 `/goal` 指令一次性批量处理 `10–15` 个游戏微调/错误修复，并在 [lmaomoba.com](https://lmaomoba.com) 发布了可玩的原型；由于 Reddit 的 `403 Forbidden` 错误，链接的 Reddit 视频无法播放。** 发布者认为 **Opus 4.8 是一个“one shot machine”**，并声称“5.5 也做不到这一点”，而评论者大多反应热烈，并询问有关美术资产、动画、背景和模型的流水线。后续讨论提到，他们对 Claude 生成的英雄名称进行了“不侵犯 IP”的审查，替换了与 League of Legends 过于接近的引用，例如将类 Teemo 的名称改为 “Teehee”。

    - 一位评论者质疑该项目是否真的是 *“1 shot”*（单次生成）构建，并表示他们在使用 **Claude Opus 4.8** 时的体验是，即使是具体的小型任务，它在每个环节上 *“转圈思考的时间也比 4.7 长好几分钟”*。他们报告在一天结束前换回了 **Codex**，这表明 Opus 4.8 可能更适合广泛的产品/原型探索，而非严谨的任务导向型工程工作流。
    - 创作者提到在生成后运行了一个 *“不侵犯 IP 审查”*，重命名生成的英雄并降低与 League of Legends IP 的相似度。这意味着工作流在初始内容生成后包含一个明确的 AI 辅助清理/重写步骤，例如将类 Teemo 的命名替换为 *“Teehee”*。
    - 一位评论者询问非代码类游戏资产（美术、动画、背景和模型）使用了什么工具，指出了复现该项目的一个关键实施差距：Opus 是仅生成了代码/游戏逻辑，还是也通过外部工具协调了资产的创建。

- **[我住在 SFO 附近，利用 ADS-B 无线电和 Claude Code 构建了一个飞过我家上空的飞机的投影映射](https://www.reddit.com/r/ClaudeCode/comments/1tva44g/i_live_by_sfo_and_built_a_projection_mapping_of/)** (Activity: 3124): **楼主 (OP) 在 **SFO** 附近构建了一个本地的 **基于 ADS-B 的飞机可视化系统**，利用接收到的飞机应答机数据驱动其房屋上方飞机的 **投影映射显示**；由于 **403 Forbidden** 拦截，无法访问链接中的 Reddit 视频 (`v.redd.it/gl2b0xivvy4h1`)。文中描述该实现是使用 **Claude Code** 构建的，但在可访问的帖子文本中并未提供硬件栈、SDR/天线细节、解码流水线、延迟或投影校准方法。** 评论大多是积极的，但缺乏技术含量，称其为 “vibe coding” 且 “很酷”；唯一的技术性追问是询问该项目需要多少设备。

    - 几位评论者索要能让该 ADS-B 投影映射项目可复现的实现细节，特别是所需的硬件/设备、可能的物料清单 (BOM) 以及代码是否可以开源。一个技术相关的扩展建议是将飞机投影与 *星座数据* 结合，以实现增强的天空/飞行可视化设置。


### 3. AI 全民所有制政策推进

  - **[一项旨在让公众拥有美国最大 AI 公司 50% 股份的提议法案。](https://www.reddit.com/r/singularity/comments/1tuf0ka/a_proposed_bill_to_give_the_public_a_50_ownership/)** (Activity: 1995): ****Bernie Sanders** 宣布了拟议的 [**《美国 AI 主权财富基金法案》(American AI Sovereign Wealth Fund Act)**](https://www.youtube.com/watch?v=VN4b4UCWMKI)，该法案将赋予公众 **`50%` 的美国最大 AI 公司所有权股份**。该提案将前沿 AI 公司定义为可能产生“数万亿”集中经济价值的潜在实体，并将部分收益注入类似主权财富基金的公共载体，而不是让收益仅留在私人所有者和投资者手中。** 热门评论普遍表示支持，将 AI 租金比作石油财富，并援引挪威主权财富基金作为模型。一位评论者更倾向于持续的财富共享或 **UBI 式分配**，而不是一次性的 `50%` 所有权/税收机制，而另一位评论者认为该提案是从试图禁止或限制数据中心转向更现实的转变。


  - **[Bernie Sanders：AI 是一种公共资源。你应该拥有它的一半。](https://www.reddit.com/r/singularity/comments/1tuo0n5/bernie_sanders_ai_is_a_public_resource_you_should/)** (Activity: 1103): **由于抓取返回了来自 [nytimes.com](https://www.nytimes.com/2026/06/01/opinion/artificial-intelligence-bernie-sanders.html) 的 **`403 Forbidden`**，无法对链接中的 **NYTimes** 评论文章 *“Bernie Sanders: A.I. Is a Public Resource. You Should Own Half of It.”* 进行技术评估。根据标题，该帖子涉及一项将 AI 定义为公共资源的政策提案，并包含某种形式的公共所有权或价值共享，但提供的内容中没有具体的实施细节、经济机制或 AI 基础设施细节。** 热门评论普遍支持这一前提，其中一位评论者质疑为什么类似的公共所有权逻辑没有应用于水和电等公共事业，特别是考虑到数据中心驱动的基础设施需求和不断上涨的账单。

    - 一个实质性的批评集中在 Sanders 的陈述前提与拟议机制之间的不匹配：如果前沿 AI 系统是在涵盖书籍、代码、研究、媒体、图像和思想的“人类集体知识”上训练的，那么 **仅限美国的主权/公共所有权模型** 只补偿了美国人，而不是全球贡献者，如非美国的艺术家、研究人员、程序员和记者。评论者将其描述为一个尚未解决的分配问题：全球训练投入、美国法律执行和国内受益者之间并不对等。
    - 另一个技术政策方面的担忧是，强制性的 **50% 公共股权** 并不一定会自动转化为公共财富，除非股份保留价值、产生股息并得到有效分配或管理。评论者认为最明显的实际效果将是 **控制权 (Control rights)**——投票权、董事会席位以及联邦政府对前沿 AI 公司的影响力——同时也警告称，这种强制要求可能会降低行业估值或扭曲资本形成。
    - 另一个针对基础设施的反对意见询问，如果公众在事后被授予所有权，那么 AI 开发、计算 (compute)、电力、冷却和数据中心建设的成本由谁承担。一位评论者将该提案与更广泛的资源外部性联系起来，指出无论消费者是否直接从 AI 基础设施扩张中受益，电费和水费账单都可能上涨。

# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们将很快发布全新的 AINews。感谢读到这里，这是一段美好的历程。