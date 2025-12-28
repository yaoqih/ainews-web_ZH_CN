---
companies:
- openai
- fineweb
- meta-ai-fair
- nvidia
- tesla
date: '2024-05-29T01:24:27.055047Z'
description: '五年前，**OpenAI** 的 GPT-2 因被认为“过于危险而无法发布”引发了争议。如今，借助 **FineWeb** 和 **llm.c**，使用
  8 张 **A100** GPU，仅需 **90 分钟**和 **20 美元**即可训练出一个微型 GPT-2 模型；而完整的 16 亿参数（1.6B）模型预计耗时
  **1 周**，成本约为 **2500 美元**。该项目因大量使用 **CUDA**（占比 75.8%）而备受关注，旨在简化训练技术栈。


  与此同时，**杨立昆（Yann LeCun）**与**埃隆·马斯克（Elon Musk）**在 Twitter 上的辩论突显了**卷积神经网络（CNN）**在自动驾驶实时图像处理中的重要性，杨立昆强调了科学研究在技术进步中的作用。此外，杨立昆还批评了
  AI 末日论，主张对 AI 安全和监管保持审慎乐观。'
id: b863d96e-eb38-4240-ace3-59652884734d
models:
- gpt-2
original_slug: ainews-somebody-give-andrej-some-h100s-already
people:
- andrej-karpathy
- yann-lecun
- elon-musk
- francois-chollet
- svpino
- mervenoyann
title: 赶紧给安德烈（Andrej）整点 H100 吧。
topics:
- cuda
- fine-tuning
- training-time
- gpu-acceleration
- convolutional-neural-networks
- real-time-processing
- ai-safety
- ai-regulation
---

<!-- buttondown-editor-mode: plaintext -->**C+CUDA 就是你所需的一切。**

> 2024年5月27日至5月28日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**382** 个频道和 **4432** 条消息）。
预计节省阅读时间（按 200wpm 计算）：**521 分钟**。

五年前，OpenAI 的 GPT-2 被称为[“过于危险而无法发布”](https://slate.com/technology/2019/02/openai-gpt2-text-generating-algorithm-ai-dangerous.html)，引发了其首个争议。

今天，在 [FineWeb（上个月发布）](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)的帮助下，你可以在 [90 分钟内花费 20 美元的 8xA100 服务器时长训练一个微型 GPT-2](https://github.com/karpathy/llm.c/discussions/481)。[350M 版本](https://news.ycombinator.com/item?id=40504950)已经可以运行（[某种程度上](https://x.com/karpathy/status/1795525191596138926)），Andrej 估计完整的 1.6B 模型将需要 1 周时间和 2500 美元。

 
![image.png](https://assets.buttondown.email/images/28a220bf-db6e-4a67-b5ea-6738bfb86771.png?w=960&fit=max)
 

这是从零开始工作 7 周取得的惊人成就，尽管目前该仓库有 75.8% 是 CUDA 代码，这让 "llm.c" 这个名字显得有些名不副实。

Andrej 还在 [HN](https://news.ycombinator.com/item?id=40502090) 和 [Twitter](https://x.com/karpathy/status/1795484547267834137) 上回答了一些问题。其中最有趣的回复之一：

**问：完成这项训练任务需要多大的二进制文件集？目前的 PyTorch + CUDA 生态系统极其庞大，操作那些容器镜像非常痛苦，因为它们太大了。我希望这能成为一个更小的训练/微调堆栈的开始？**

**答：这百分之百是我的意图和希望，我认为我们离删除所有这些内容已经非常接近了。**

如果有[更多的 H100 可用](https://x.com/karpathy/status/1795493747205238916)，成本会更低，速度会更快。有人能帮帮这位新晋的 GPU 穷人（GPU poor）吗？

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！


{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Yann LeCun 与 Elon Musk 的 Twitter 辩论**

- **卷积神经网络 (CNNs) 的重要性**：[@ylecun](https://twitter.com/ylecun/status/1795393908886712425) 指出，1989 年引入的 CNNs 如今被用于所有的驾驶辅助系统，包括 MobilEye、Nvidia、Tesla。**技术奇迹建立在通过技术论文分享的多年科学研究之上。**
- **LeCun 的研究贡献**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1795435037396988162) 表示在 @ylecun 和 @elonmusk 之间会选择前者，因为**发表突破性研究的科学家是技术进步的基石，尽管他们获得的认可少于企业家。**
- **Musk 质疑 LeCun 的 CNN 使用情况**：[@elonmusk](https://twitter.com/elonmusk/status/1795426059921268969) 询问 @ylecun，如果没有 ConvNets，Tesla 如何在 FSD 中进行实时摄像头图像理解。[@ylecun](https://twitter.com/ylecun/status/1795428712460451841) 回应称 Tesla 使用了 CNNs，因为对于实时高分辨率图像处理来说，Attention 机制太慢了。[@svpino](https://twitter.com/svpino/status/1795506451131044047) 和 [@mervenoyann](https://twitter.com/mervenoyann/status/1795506858985177137) 证实了 Tesla 对 CNN 的使用。
- **LeCun 的研究产出**：[@ylecun](https://twitter.com/ylecun/status/1795219718837616775) 分享自 2022 年 1 月以来他已发表了 80 多篇技术论文，并质疑 Musk 的研究产出。他还提到自己在 Meta 工作，[@ylecun](https://twitter.com/ylecun/status/1795158771695542279) 表示这没什么问题。
- **Musk 表现得像 LeCun 的老板**：[@ylecun](https://twitter.com/ylecun/status/1795265406191735191) 开玩笑说 Musk 表现得好像他是自己的老板一样。[@fchollet](https://twitter.com/fchollet/status/1795226758502826154) 建议他们通过笼斗（cage fight）来解决，而 [@ylecun](https://twitter.com/ylecun/status/1795268462597824548) 则提议进行帆船比赛。

**AI 安全与监管讨论**

- **AI 末日场景**：[@ylecun](https://twitter.com/ylecun/status/1795032310590378405) 批评了“AI 末日”论调，认为 AI 是由人类设计和建造的，而且**如果存在安全的 AI 系统设计，我们就不会有问题。现在担心或通过监管 AI 来防止“生存风险”还为时过早。**
- **AI 监管与中心化**：[@ylecun](https://twitter.com/ylecun/status/1794998977105981950) 概述了“末日论者的错觉”，即 **AI 末日论者推动少数公司垄断 AI、严格监管、远程关停开关、基础模型构建者的永久责任、禁止开源 AI，并用末日预言恐吓公众。** 他们成立个人研究所来推广 AI 安全，从恐惧的亿万富翁那里获得巨额资助，并声称著名科学家也同意他们的观点。

**AI 研究与工程讨论**

- **在 C/CUDA 中复现 GPT-2**：[@karpathy](https://twitter.com/karpathy/status/1795484547267834137) 在一个 8X A100 80GB 节点上，仅用 90 分钟和 20 美元的成本，在 llm.c 中复现了 GPT-2 (124M)，MFU 达到 60%。他还用约 200 美元在 14 小时内复现了 350M 模型。**提供了完整的操作指南。**
- **用于算术的 Transformers**：[@_akhaliq](https://twitter.com/arankomatsuzaki/status/1795300845942382701) 分享了一篇论文，表明 **Transformers 配合正确的 embeddings 可以进行算术运算**，通过在单 GPU 上训练一天 20 位数字的数据，在 100 位数字的加法问题上达到了高达 99% 的准确率。
- **Gemini 1.5 模型更新**：[@lmsysorg](https://twitter.com/lmsysorg/status/1795512202465845686) 公布了 Gemini 1.5 Flash、Pro 和 Advanced 的结果，其中 **Pro/Advanced 排名第 2，紧随 GPT-4o 之后，而 Flash 排名第 9，超越了 Llama-3-70b 且接近 GPT-4-0125。Flash 的成本、能力和上下文长度使其成为市场的游戏规则改变者。**
- **Zamba SSM 混合模型**：[@_akhaliq](https://twitter.com/arankomatsuzaki/status/1795299751644340465) 分享了 Zamba 论文，这是一个 **7B SSM-Transformer 混合模型，在同等规模下达到了与领先开源权重模型相当的性能。** 它是在来自公开数据集的 1T tokens 上训练的。
- **用于将 LLM 训练为嵌入模型的 NV-Embed**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1795286849487098035) 分享了 NVIDIA 关于 NV-Embed 的论文，该论文**改进了将 LLM 训练为通用嵌入模型的技术。它在 MTEB 排行榜上排名第 1。**

**梗图与幽默**

- **Musk vs. LeCun 梗图**：[@svpino](https://twitter.com/svpino/status/1795503047004594637) 和 [@bindureddy](https://twitter.com/bindureddy/status/1795269862111256904) 分享了关于 Musk 与 LeCun 辩论的梗图，调侃了这一局面。
- **用 AI Bot 替代 Twitter 上的自己**：[@cto_junior](https://twitter.com/cto_junior/status/1795479060258197877) 开玩笑说要在 Slack 上建立一个 AI 版本的自己来代替参加站会（standups），而不是在 Twitter 上。

---

# AI Reddit 摘要回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与架构**

- **01-ai 移除 Yi 模型的自定义许可证**：在 /r/LocalLLaMA 中，01-ai 已[**在 Huggingface 上将其原始 Yi 模型的许可证切换为 Apache-2.0**](https://www.reddit.com/r/LocalLLaMA/comments/1d1zzbz/01ai_just_removed_all_the_custom_licenses_from/)，与其 1.5 系列模型的许可证保持一致。
- **InternLM2-Math-Plus 模型发布**：发布了一系列[**升级后的数学专用开源大语言模型，涵盖 1.8B、7B、20B 和 8x22B 尺寸**](https://www.reddit.com/r/LocalLLaMA/comments/1d1om5d/we_release_internlm2mathplus_with_18b7b20b_and/)。其中 InternLM2-Math-Plus-Mixtral8x22B 在 MATH（配合 Python）和 GSM8K 基准测试中分别达到了 68.5 和 91.8 分。
- **Pandora 世界模型推出**：Pandora 是一种[**混合自回归扩散模型，通过生成视频来模拟世界状态，并允许通过自由文本动作进行实时控制**](https://www.reddit.com/r/LocalLLaMA/comments/1d1meba/pandora_towards_general_world_model_with_natural/)。其目标是实现领域通用性、视频一致性和可控性。
- **llama.cpp 添加对 Jamba 架构的支持**：在 /r/LocalLLaMA 中，[**llama.cpp 正在添加对 AI21 Labs 的 Jamba 架构的支持**](https://www.reddit.com/r/LocalLLaMA/comments/1d1ur6h/jamba_llamacpp_support/)，首批 GGUF 文件已上传，其中包括一个基于 Bagel 数据集微调的模型。
- **针对天文学发布的 AstroPT 模型**：[AstroPT](https://arxiv.org/abs/2405.14930) 是为天文学用例开发的自回归预训练 Transformer，模型参数从 1M 到 2.1B 不等，在 8.6M 个星系观测数据上进行了预训练。代码、权重和数据集均以 MIT 许可证发布。

**AI 应用与工具**

- **优化 Whisper 以实现快速推理**：在 /r/LocalLLaMA 中，有成员分享了[**通过 SDPA/Flash Attention、投机采样 (speculative decoding)、分块 (chunking) 和蒸馏 (distillation) 等技术将 Whisper 推理速度提升高达 5 倍的技巧**](https://www.reddit.com/r/LocalLLaMA/comments/1d1xzpi/optimise_whisper_for_blazingly_fast_inference/)。
- **用于文档问答的 Android 应用**：[Android-Document-QA](https://www.reddit.com/r/LocalLLaMA/comments/1d1zzxr/androiddocumentqa_rag_pipeline_for_document_qa/) 是一款 Android 应用，它利用 LLM 回答用户提供的 PDF/DOCX 文档中的问题，并利用各种库进行文档解析、设备端向量数据库等操作。
- **用于本地音乐生成的 MusicGPT**：在 /r/MachineLearning 中，[MusicGPT 被介绍为一个终端应用，可以在本地运行 Meta 的 MusicGen，通过自然语言提示生成音乐](https://www.reddit.com/r/MachineLearning/comments/1d1vp2u/p_musicgpt_an_open_source_app_for_generating/)。该应用由 Rust 编写，最终目标是实时生成无限的音乐流。
- **发布新的 Web+LLM 框架**：宣布了一个[**针对与 LLM 和微服务集成的 IO 密集型应用优化的开源 Web 框架**](https://www.reddit.com/r/LocalLLaMA/comments/1d1yofb/i_made_a_webllm_framework_looking_for_early/)，正在寻找早期采用者进行试用并提供反馈。

**AI 伦理与安全**

- **微软 Recall AI 功能因隐私担忧受到调查**：[微软新的 Recall AI 功能（通过跟踪用户活动来辅助数字助手）正因隐私担忧受到英国当局的调查](https://mashable.com/article/microsoft-recall-ai-feature-uk-investigation)，这引发了关于实用 AI 辅助所需数据的辩论。

**AI 行业与竞争**

- **过去一年 AI 竞争的可视化**：来自 [**LMSYS Chatbot Arena 的可视化图表显示了过去一年中各大 LLM 厂商顶级模型的表现**](https://www.reddit.com/r/LocalLLaMA/comments/1d1qfby/evolution_of_ai_competition_in_the_last_year/)，突显了日益激烈的竞争和不断变化的趋势。
- **关于 OpenAI 股权回收的矛盾说法**：一篇文章称[**数据与 Sam Altman 关于对 OpenAI 股权回收 (equity clawbacks) 知情的声明相矛盾**](https://thedeepdive.ca/sam-altman-claims-ignorance-on-equity-clawbacks-but-data-contradicts/)。

---

# AI Discord Recap

> 摘要之摘要的摘要


**LLM 进展与基准测试**：

- **[Llama 3 领跑榜单](https://lmsys.org/blog/2024-05-08-llama3/)**：来自 Meta 的 Llama 3 在 **ChatbotArena** 等排行榜上名列前茅，在超过 50,000 场对决中超越了 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
- **IBM 和 RefuelAI 发布新模型**：IBM 的 [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 增强了代码任务的指令遵循能力，而 RefuelAI 的 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 则在处理数据密集型任务方面表现出极高的效率。

**优化 LLM 推理与训练**：

- **[前沿量化技术](https://www.deepspeed.ai/tutorials/zeropp/)**：ZeRO++ 旨在将大型模型在 GPU 上训练期间的通信开销降低 4 倍。
- **内存效率创新**：[vAttention](https://arxiv.org/abs/2405.04437) 系统更动态地管理 KV-cache 内存，提升了 LLM 推理的敏锐度；而 **[QSolve](https://arxiv.org/abs/2405.04532)** 引入了 **W4A8KV4 量化**，以增强基于云端的 GPU 性能。

**开源 AI 框架与社区努力**：

- **[Axolotl 吸引开发者关注](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)**：支持多种数据格式，助力 LLM 预训练和指令微调。
- **[LlamaIndex 激发学术好奇心](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)**：与 Andrew Ng 合作推出了一门关于构建 Agentic RAG 系统的新课程，预示着 **bfloat16** 等 AI 扩展技术的进步。

**多模态 AI 与生成模型创新**：

- **[Idefics2 8B 在对话领域引起轰动](https://twitter.com/sanhestpasmoi/status/1787503160757485609)**：微调了聊天交互中的话语表现，同时 **[CodeGemma 1.1 7B 提升了编程任务表现](https://twitter.com/reach_vb/status/1786469104678760677)**。
- **[Phi 3 开创基于浏览器的 AI](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)**：通过 WebGPU 直接在浏览器中引入强大的 AI 聊天机器人，为注重隐私的交互增强奠定了基础。

---

{% if medium == 'web' %}



# 第一部分：高层级 Discord 摘要




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**OCR 对决：Google Vision vs. Microsoft Azure**：AI 工程师们辩论了 **Google Vision OCR** 的优缺点，认可其精度但批评了开发者体验。有人建议使用 **Microsoft Azure OCR** 和 **Mindee Doctr**，认为它们可能提供更好的易用性，详情见[此处](https://huggingface.co/spaces/mindee/doctr)。

**精选数据：LLM 成功的关键**：研讨会讨论强调了使用高质量、精选数据集微调 LLM 的重要性，应用范围涵盖从制药应用到技术支持聊天机器人。专家意见指出，为了最大化 LLM 的效能，需要精准选择数据，并重点关注药物研发、法律、销售和跨学科工作等领域。

**Axolotl 的困扰与优化**：用户在 M3 Macs 上运行 **Axolotl 的 70B 模型**时遇到障碍，本地推理延迟极高，这表明部署在 Modal 上可能是一个解决方案。对 **Weights & Biases (WandB)** 成本的担忧促使注重经济效益的独立开发者考虑 **Aim** 和 **MLflow** 等替代方案，参考 [Axolotl 示例](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)。

**LLM 评估深度探讨**：一场关于评估 LLM 的会议提供了大量见解，涵盖了产品指标、传统与动态性能指标，以及 LangFuse 和 EvalGen 等工具。参与者推荐了 Eugene Yan 的资源和可视化微调的实际案例，并指出细致入微的评估对于 LLM 开发至关重要。

**转录困局与摘要之路**：围绕大型会议转录文本的交流凸显了对高效摘要的需求，揭示了 LLM 的潜在作用。虽然 Zoom 转录功能即将推出，但 Hamel 鼓励使用 LLM 生成更易读的摘要，这得到了社区的广泛响应。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **翘首以盼 imfo Alpha 版本发布**：[@spectate_or](https://x.com/spectate_or/status/1795077451195830661?s=46) 发布的一条推文链接暗示了 **imfo alpha** 即将发布，这在工程社区中引发了兴奋，并将其与同类工具进行了比较。

- **AI 任务结构辩论**：工程师们讨论了将 **AI 任务** 分为检索型和变异型（mutation types），并以“获取 iPhone 15 的重量”等查询为例。针对需要顺序执行的任务，讨论强调了调整的必要性，并提出见解：*“所有步骤几乎是同时发生的。”*

- **网页抓取准确性遭遇挑战**：成员们表达了在 **HTML 解析** 以实现可靠数据抓取方面面临的挑战，特别是像 Apple 和 Docker 发布说明这类网站带来的复杂性。针对以 JavaScript 为核心的网站，讨论了通过 **Playwright** 进行解决的方案，同时也考虑了 Cloudflare 带来的问题。

- **探索高性价比的 AI 模型利用**：社区深入探讨了使用 Llama3 和 Claude 等各种 **AI 模型** 的成本效益。一种使用组合系统的方法表明了实现更大成本节约的可能性。

- **API 功能异常引起关注**：关于 **API 输出** 显示 JSON 对象但缺少功能链接的问题引发了困惑，这可能与 **closed beta citations feature**（封闭测试引用功能）的缺失有关。其他讨论还包括改进视频链接生成的提示词，以及对潜在 API 故障的简短询问。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**值得尝试的新 AI 功能**：Stability AI 宣布推出 **Stable Assistant**，它具备基于 **Stable Diffusion 3** 构建的编辑功能，并宣称提升了文本生成图像的质量，可在此处进行[免费试用](https://stability.ai/stable-assistant)；此外还推出了搭载 **Stable LM 2 12B** 的 Beta 版聊天机器人，预示着未来文本生成任务的增强。

**教育与 AI 创新融合**：由 **Innovation Laboratory**（Stability AI 与 HUG 的合作项目）即将推出的为期 4 周的课程，旨在指导参与者结合 HUG 的教育方法，利用 Stability AI 的框架训练 AI 模型；报名截止日期为 2024 年 6 月 25 日，可通过[此处](https://www.studios.thehug.xyz/lab)访问。

**GPU 共享成为焦点**：AI 工程师讨论了一项基于社区的 GPU 共享提案，以降低计算成本，方案涵盖了从自定义节点到旨在验证模型训练操作的潜在区块链设置。

**SD3 可访问性引发争议**：由于 **Stable Diffusion SD3** 的权重无法在本地使用，成员们表达了不满——批评 Stability AI 仅限云端的方法，并引发了关于云依赖和数据隐私问题的辩论。

**用户界面对比**：一场关于 Stable Diffusion 各种界面优缺点的技术讨论展开，**ComfyUI** 被拿来与 Forge 等更易用的替代方案进行对比；讨论还包括社区技巧、Inpainting（局部重绘）方法以及增强人工智能工作流的方法。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**OpenAI 组建安全护盾**：OpenAI 成立了一个 **Safety and Security Committee**（安全与安保委员会），负责其所有项目的关键安全和安保决策；详细信息可见其[官方公告](https://openai.com/index/openai-board-forms-safety-and-security-committee/)。

**AI 在硬件领域展现实力**：关于硬件成本的讨论出现，推测由于 **NPUs**（神经处理单元）的加入，成本将增加 200 至 1000 美元，重点关注其对高端模型的经济影响。

**规划 Prompt 蓝图**：AI 工程师辩论了 **meta-prompting**（元提示）与 **Chain of Thought (CoT)**（思维链）的优劣，探讨了使用 mermaid 图表来节省 tokens 并提高输出质量的潜力。此外，还分享了改进后的提示词（如[此处](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477)），展示了高级 Prompt Engineering 策略的实际应用。

**理论付诸代码实践**：实际讨论包括 AI 如何原生处理 **YAML, XML, and JSON** 格式，并建议在提示词中使用这些结构以提高 AI 的理解能力和性能；同时分享了指向代码生成和规划的实际 Prompt 应用资源。

**交互不一致性引发探究**：用户报告了 **ChatGPT** 的一系列问题，从拒绝绘制塔罗牌到上下文丢失和无响应，突显了对更改进且更可预测的 AI 行为的需求。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**语音指令邂逅机器人技术**：一段名为[“开源语音控制机器人手臂”](https://www.youtube.com/watch?v=qv3bFhHoA5s)的演示视频展示了一个语音激活的 AI 机器人手臂。视频提出了通过社区协作实现机器人技术民主化的观点。

**桥接模态**：关于创建早期多模态 Space 的贡献指出，可以使用单一模型，或者使用带有路由功能的堆叠模型。为了深入了解此类实现，分享了一个[源码链接](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py)，提供了一个具有实际应用价值的模型示例。

**即时深度学习咨询**：一位用户就使用 Stanford Cars Dataset 训练模型时遇到的常见痛点咨询了社区。该用户使用 ViT-B_16 仅达到了 60% 的准确率，并深受过拟合（overfitting）困扰。同时，另一位成员正在寻求如何改进其深度学习模型的帮助，这表明社区拥有支持新手知识交流的良好环境。

**Diffusers 更新：不仅限于生成任务**：Hugging Face 宣布其 **Diffusers 库现在支持生成模型以外的任务**，例如通过 **Marigold** 进行深度估计（depth estimation）和法线预测（normals' prediction）。此次更新表明 Diffusion 模型的多功能性及其应用领域正呈现上升趋势。

**网络安全评估的模型选择**：研究人员的分析探讨了各种 LLM 在网络安全背景下的能力。这为 AI 工程师提供了一个视角，去审视部署 LLM 时固有的安全影响。

**稳健的 SDXL 空间重新对齐**：关于 SDXL embed 空间的讨论强调，新对齐的空间默认值为零，而不是编码空间。这些见解反映了将模型重新对齐到新的无条件空间（unconditioned spaces）所涉及的底层复杂性和时间需求，揭示了科学背后的复杂过程。

**Gradio 升级版客户端引发好奇**：Gradio 团队宣布即将举行一场直播活动，深入探讨 Gradio Python 和 JavaScript 客户端的最新功能。此次活动邀请强调了 Gradio 致力于通过增强接口来简化 AI 到各类应用中的集成。
  
**寻找 SFW 数据集的困惑**：社区讨论提到了寻找 Nomos8k_sfw 数据集的困难，该数据集与 4x-Nomos8kDAT 模型相关，这表明该数据集的可用性有限或存放位置隐蔽。这突显了数据集获取过程中偶尔会遇到的挑战。

**发布最新的 AI 叙事工具**：Typeface Arc 作为一个综合平台脱颖而出，旨在无缝创建 AI 驱动的内容。它包含一个被恰当地称为 “Copilot” 的工具，旨在通过对品牌叙事至关重要的交互式体验来增强内容创作。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**视觉化：OpenAI 与 Llama 集成！**：工程师们现在可以通过在服务器上部署 **LLaVA** 并利用提供的 Python 视觉模板，在 LM Studio 中利用其视觉能力。

**M1 Max 上的快速模型加载**：像 **MLX 和 EXL2 这样的 AI 模型在 Apple 的 M1 Max 上加载非常迅速**，L3 8bit 仅需 5 秒，相比之下 GGUF Q8 需要 29 秒，显示出卓越的性能。

**LM Studio 微调的挫折**：尽管是一个强大的环境，但 **LM Studio 目前缺乏直接微调（fine-tune）模型的能力**，爱好者们被引导至针对 Apple Silicon 设计的 MLX 等替代方案。

**预算还是性能**：AI 从业者辩论了各种 Nvidia GPU 的价值主张，考虑了 **Tesla P40/P100** 等替代方案，并满怀期待地讨论了传闻中的 **5090** 等 GPU。

**Beta 测试的烦恼**：在体验新版本时，用户报告了诸如大模型在 **Windows 上的 CPU 亲和性（affinity）问题**以及 **AVX2 笔记本电脑上的错误**，这暗示了为 AI 任务配置现代硬件的复杂性。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-2 未能获得 Unsloth 的支持**：Unsloth 确认 **GPT-2** 无法使用其平台进行微调，原因是基础架构存在根本差异。

- **Fiery Chat 微调中的挫折**：
  - 在使用超过 50,000 条电子邮件条目微调 **Llama 3** 时，成员们分享了关于构建 Prompt 以实现最佳输入输出配对的建议。
  - 针对训练后出现的句子重复问题，建议添加 **End-Of-Sentence (EOS)** Token，以防止模型过拟合或学习效果不佳。

- **视觉模型集成指日可待**：成员们正热切期待 **Unsloth** 下个月关于视觉模型支持的更新，目前推荐的解决方案参考了 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 和 [Segment Anything](https://github.com/facebookresearch/segment-anything)。

- **LoRA Adapter 的协同工作**：社区分享了合并和微调 **LoRA** Adapter 的技巧，强调使用 [GitHub 上的 Unsloth 文档](https://github.com/unslothai/unsloth#-finetune-for-free) 等资源，并将模型导出到 **HuggingFace**。

- **应对 Phi 3 Medium 的注意力跨度**：关于 **Phi3-Medium** 的讨论揭示了其滑动窗口注意力（sliding window attention）导致在高 Token 计数时效率下降，许多人渴望能有增强功能来处理更大的上下文窗口。

- **ONNX 导出详解**：针对将微调后的模型转换为 **ONNX** 提供了指导，参考了 Hugging Face 的[序列化文档](https://huggingface.co/docs/transformers/en/serialization)，并确认 **VLLM** 格式兼容转换。

- **迈向低比特时代**：大家对 **Unsloth** 即将支持的 8-bit 模型以及与 **Ollama** 等环境的集成能力充满期待，这与 **OpenAI** 提供的服务类似。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Ubuntu 上的 CUDA Toolkit 命令**：一位用户建议从 **NVIDIA** 安装 **CUDA Toolkit**，通过 `nvidia-smi` 检查安装情况，并提供了在 Ubuntu 上设置的命令，包括通过 **Conda** 安装：`conda install cuda -c nvidia/label/cuda-12.1.0`。同时，在设置 **PyTorch 2.3** 时发现了与 **Python 3.12** 的潜在冲突以及缺失 **triton** 安装的问题，这与一个 [GitHub issue](https://github.com/pytorch/pytorch/issues/120233) 相关。

- **GPT-4o 在处理大规模编辑时遇到挑战**：成员们注意到 **GPT-4o** 在处理大规模代码编辑时表现吃力，一个新的 **fast apply** 模型旨在将任务分解为计划和应用阶段以克服这一挑战。为了寻求代码编辑的确定性算法，一位成员提出了使用 **vllm** 或 **trtllm** 进行未来 Token 预测而不依赖草稿模型的可行性。关于此方法的更多信息可以在[完整博客文章](https://cursor.sh/blog/instant-apply)中找到。

- **SYCL 调试难题**：一位成员询问了调试 **SYCL** 代码的工具，引发了关于进入 Kernel 代码进行故障排除的讨论。

- **Torchao 的最新成果**：**torchao** 社区庆祝了 **PyTorch** 合并对 **MX** 格式（如 `fp8/6/4`）的支持，这为感兴趣的各方提供了效率提升，该支持部分由一个 [GitHub commit](https://github.com/pytorch/ao/pull/264) 提供，并符合 [MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

- **深入理解 DIY 中的 Mixer 模型**：成员们剖析了实现细节，例如在 **llm.c** 中集成 `dirent.h`，以及为了操作系统兼容性使用 `#ifndef _WIN32` 对其进行保护的重要性。实现了用于在中断时恢复训练的 `-y 1` 标志，解决了关于未初始化变量的警告，并探索了 Backward Pass 计算期间的内存优化策略，相关倡议见 [GitHub 讨论](https://github.com/karpathy/llm.c/discussions/481)。

- **BitNet 中的激活量化**：在 **BitNet** 频道中，结论是在激活量化神经网络中直接传递输入梯度可能是错误的。相反，建议使用代理函数（如 `tanh`）的梯度，并引用了一篇关于直通估计器（**STE**）性能的 [arXiv 论文](https://arxiv.org/abs/1903.05662)。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT Agents 无需后期学习**：基于 GPT 的 Agent 在初始训练后不会进行学习，但可以引用上传为“知识文件”的新信息，而不会从根本上改变其核心理解。
- **Diffusion Models 的效率里程碑**：Google DeepMind 推出 **[EM Distillation](http://arxiv.org/abs/2405.16852)** 以创建高效的一步生成器 Diffusion Models；Google 的另一项独立研究展示了一个 8B 参数的 Diffusion Model，擅长生成 1024x1024 的高分辨率图像。
- **缩小规模以产生影响**：**[Super Tiny Language Models](https://arxiv.org/abs/2405.14159)** 研究专注于在不显著牺牲性能的情况下减少 90-95% 的语言模型参数，为更高效的自然语言处理指明了道路。
- **无需猜测的 GPU 性能**：**无需执行**即可对 GPU 延迟进行符号建模的方法受到关注，相关的[学术资源](https://inria.hal.science/hal-00789958/file/112_Lai.pdf)可指导理论理解及对计算效率的潜在影响。
- **与社区共同挑战现状**：讨论强调了社区驱动的项目以及在 Prompt 适配研究和实现查询（如 PyTorch 中的 **Facenet 模型**）等领域中**协作解决问题**的重要性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **最新模型创新上市**：[OpenRouter](https://openrouter.ai/models) 发布了新的 AI 模型，包括 **[Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3)** 和 **[Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b)**，同时保证之前的版本如 **[Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2)** 仍可访问。
- **对 Max Loh 网站模型的好奇**：用户对 [Max Loh 网站](https://www.maxloh.com)上使用的模型表示好奇，并有兴趣识别 OpenRouter 上可用的所有无审查（uncensored）模型。
- **OCR 才艺展示**：**Gemini 的 OCR** 能力成为热门话题，用户称其在阅读西里尔字母和英文文本方面具有卓越能力，优于 Claude 和 GPT-4o 等竞争模型。
- **OpenRouter Token 经济学**：社区澄清了在 OpenRouter 上 0.26 美元可获得 1M input + output tokens，讨论强调了每次聊天交互都会重新计算 Token 使用量，这可能会增加成本。
- **尖端 Vision 模型的成本**：关于在 Azure 上使用 **Phi-3 Vision** 的成本展开了激烈讨论，一些成员认为 Llama 定价的 0.07 美元/M 太贵，尽管其他服务提供商也有类似的费率。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **翻译的苦恼**：讨论涉及在控制歌词语调以保留原始艺术意图的情况下*翻译歌曲*的挑战。其独特的困难在于平衡意义的忠实度与音乐性和艺术表达。
- **AI 渗透 Greentext**：成员们尝试使用 LLM 生成 **4chan greentexts**，分享了他们对 AI 叙事能力的着迷——尤其是构思一个醒来后发现 AGI 已经实现的世界的场景。
- **哲学性的 Phi 与逻辑受限的 LLM**：围绕 **Phi 模型的训练数据**构成展开了辩论，提到了“重度过滤的公开数据和合成数据”。此外，有证据表明 LLM 在交互过程中难以处理逻辑和自我修正，引发了对模型推理能力的担忧。
- **为机器消化塑造数据**：AI 爱好者交流了关于**创建 DPO 数据集**和调整 DPO 训练数据集格式的资源与见解。Hugging Face 的 [TRL 文档](https://huggingface.co/docs/trl/main/en/reward_trainer)和 [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) 成为关键参考，此外还有一篇详细介绍根据偏好数据训练语言模型的[论文](https://arxiv.org/abs/2305.18290)。
- **为 RAG 财富连接思想**：协作氛围浓厚，成员们分享了在 RAG 相关项目上共同努力的意向。这包括 [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT) 上的情感和语义密度平滑 Agent 项目（带有 TTS），以及将现有项目移植到 SLURM 以增强计算管理的计划。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain 中的无限循环问题**：工程师们正在解决 **LangChain agent** 在调用工具时进入持续循环的问题；其中一个讨论的解决方案涉及细化 Agent 的触发条件，以防止无限的工具调用循环。

**详情请看！LangChain 0.2.2 中的 16385-token 错误**：用户报告了 **LangChain 0.2.2 版本**中的一个 Token 限制错误，即错误地应用了 16385-token 的限制，尽管模型支持高达 128k tokens，这引发了社区主导的针对这一差异的调查。

**SQL Prompt 编写咨询**：关于带有 few-shot 示例的 **SQL agent** prompt 模板的请求已得到解答，为工程师提供了在 LangChain 中更有效地构建查询的资源。

**消失的自定义参数：Langserve 中的自定义 kwargs**：一些用户遇到了通过 **Langserve** 发送用于 **Langsmith** 日志记录的自定义 "kwargs" 在到达时丢失的问题，该问题目前正在寻求解决方案。

**应用展示**：分享了使用 LangChain 开发的各种应用，包括用于**药物研发**的框架、节省成本的日志记录措施、**飞行模拟器**的增强功能，以及关于 Agent 流程中**路由逻辑（routing logic）**的教程。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 用户的 Python 版本警示**：提醒 Mojo 用户遵守支持的 Python 版本（范围从 **3.8 到 3.11**），因为 **3.12 仍不受支持**。通过使用 deadsnakes 仓库进行 Python 更新，解决了 Mojo 中的相关问题。

- **AI 驱动的游戏创新**：工程师们讨论了基于开放世界游戏中 NPC 智能的订阅模式前景，并为智能设备引入特殊的 AI 功能，这可能导致 AI 推理（inference）在本地运行。他们还探讨了可以实现 AI 驱动的自定义世界生成的开放世界游戏。

- **Mojo 精通**：Mojo 允许循环依赖（Circular dependencies），因为模块可以相互定义。像 `Intable` 和 `Stringable` 这样的 Traits 是原生可用的。虽然 Lambda 函数尚未成为 Mojo 的功能，但目前使用回调（callbacks）作为替代方案。

- **性能先锋**：在 Mojo 中，*32 字节时观察到了惊人的 50 倍速度提升*，尽管超过该长度后遇到了缓存限制。k-means 算法的基准测试显示出由于内存分配和矩阵计算差异导致的波动，并建议针对 AVX512 操作优化内存对齐。

- **Nightly 版本动态**：最新的 **Mojo 编译器构建版本 (2024.5.2805)** 带来了新功能，包括 `tempfile.{mkdtemp,gettempdir}` 和 `String.isspace()` 的实现，完整变更详见 [当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [原始差异对比](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3)。通过引用进行的结构化共享（Structural sharing）也因其在 Mojo 编程中潜在的效率提升而受到关注。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **调试功能升级**：工程师们赞扬了 **Cursor 解释器模式（interpreter mode）**，强调其在调试场景中具有比传统搜索功能更先进的代码导航能力。

- **消息应用的副驾驶**：**Microsoft Copilot** 集成到 **Telegram** 引起了关注，它能够通过游戏技巧和电影推荐等功能丰富聊天体验。

- **低成本训练 GPT-2**：**Andrej Karpathy** 展示了一种经济高效的方法，在 **90 分钟内花费 20 美元**训练 GPT-2，并在 [GitHub](https://github.com/karpathy/llm.c/discussions/481) 上详细介绍了该过程。

- **Agent 与 Copilot 的角色区分**：在 **Microsoft Build** 进行分类后，关于 **Copilots** 和 **Agents** 之间的区别展开了辩论，并引用了 [Kanjun Qiu 对该话题的见解](https://www.latent.space/p/imbue)。

- **AI 播客发布前沿发现**：发布了一集[聚焦 ICLR 2024 的播客](https://x.com/latentspacepod/status/1795196817044594817)，讨论了 ImageGen、Transformers、视觉学习（Vision Learning）等领域的突破，并期待即将发布的关于 LLM 推理（Reasoning）和 Agents 的见解。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **金融极客们，尽情享用 FinTextQA**：[FinTextQA](https://t.co/emhQYXY1S4) 是一个旨在改进长篇金融相关问答系统的新数据集；它包含跨越 *6 种不同问题类型* 的 *1,262 个带有来源属性的问答对*。

- **完善 Prompt 结构**：有人咨询了关于构建最佳系统角色 Prompt 的资源，并从 **LlamaIndex** 的模型中汲取了灵感。

- **聊天历史保存策略**：社区讨论了在 **LlamaIndex** 中保存聊天历史的技术，考虑为 **NLSQL** 和 **PandasQuery** 引擎定制 Retriever，以维护查询和结果的记录。

- **API 函数管理探索**：针对拥有超过 1000 个函数的庞大 API 提出了管理策略，倾向于使用层级路由（hierarchical routing）并将函数划分为更易于管理的子组。

- **LlamaIndex RAG 系统的复杂性辩论**：剖析了 RAG 系统中与元数据相关的技术挑战，在为了获得最佳信息检索准确性而嵌入较小还是较大的语义分块（semantic chunks）方面，意见存在分歧。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 读懂言外之意**：成员们对 SOTA AGI 模型的古怪言论付之一笑，其中一个模型的自我训练断言——“它为我们训练了一个模型”——戳中了大家的笑点。Musk 对 [CNNs](https://x.com/elonmusk/status/1795405972145418548) 的嘲讽——调侃道“我们最近不怎么使用 CNN 了”——引发了一连串的反讽回复，并对作为行业新宠的 Vision Transformer 模型表示了认可。

**人工智能艺术家的水印烦恼**：[Corcelio 的 Mobius 艺术模型](https://huggingface.co/Corcelio/mobius) 正在通过多样化的 Prompt 突破界限，尽管它在创造力上超越了以往的模型，但仍会留下水印。图像生成系统产生“不当”内容的能力引发了伦理困境，触发了关于社区准则和系统控制设置的辩论。

**合成视觉寻求改进**：为了解决 **SDXL** 无法生成“阅读中的眼睛”图像的问题，一位成员请求协作帮助，利用 DALLE 构建一个合成数据库，希望在这一细微的视觉任务中磨练 **SDXL** 的能力。

**生成式水印中的模式与谜题**：公会内部的观察指出，生成式模型产生水印是一个反复出现的主题，这表明可能存在训练不足的情况，这在工程师中既被认为有趣又值得关注。

**Elon 对 CNN 的冷眼引发 AI 调侃**：Elon Musk 的推文在社区中引起了波动，引发了关于 CNN 在当今变革性 AI 方法论中已过时的笑话，以及可能向 Transformer 模型转型的讨论。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**无需基准测试的 GPU 延迟预测？**：工程师们讨论了在不运行 Kernel 的情况下，通过考虑数据移动和操作时间来对 **GPU 延迟进行符号化建模** 的潜力，尽管占用率（occupancy）和异步操作等复杂性被认为是潜在的干扰因素。此外，人们还期待 AMD 开源 MES，并推测量化交易公司会使用周期精确（cycle accurate）的 GPU 模拟器进行深入的 Kernel 优化。

**使用 Autotuner 进行优化**：社区探索了 **AutoTVM** 和 **Halide** 等 Kernel 优化工具，注意到它们提升性能的不同方法；George Hotz 强调了 TVM 对 XGBoost 的使用，并强调了缓存仿真（cache emulation）对准确建模的重要性。

**GPU 中的延迟隐藏机制**：会议指出，GPU 利用运行并发 Wavefronts/Blocks 的能力采用了多种延迟隐藏策略，从而使延迟建模变得更加复杂和微妙。

**Tinygrad 中的 Buffer 创建讨论**：#learn-tinygrad 频道有成员询问在调度中使用**后支配者分析（post dominator analysis）**以提高图融合（graph fusion）效率，以及从数组创建 **LazyBuffer** 的问题，并建议在此类场景中使用 `Load.EMPTY -> Load.COPY`。

**代码清晰度与协助**：针对 Tinygrad 中的 Buffer 分配和 `LazyBuffer` 创建进行了详细讨论，一位成员表示愿意提供**代码指针（code pointers）**以进一步澄清和理解。



---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Elevenlabs 语音加入 AI Town**：通过集成 **Elevenlabs** 的 text-to-speech（文本转语音）功能，AI Town 引入了一项新特性，让对话不仅能被阅读，还能被听到。尽管约一秒的延迟对实时使用构成了挑战。实现过程涉及[将文本转换为音频](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19)并在前端管理音频播放。

- **将科学辩论引入 AI 聊天**：分享了一个利用 AI 聊天机器人模拟科学辩论的概念，旨在促进参与并展示科学讨论的统一性。

- **增加音频窃听以提升沉浸感**：AI Town 的 Zaranova 分支现在通过为环境对话生成音频来模拟窃听（eavesdropping），这可能会增强平台的互动性。

- **协作开发动员**：社区对贡献并可能将新功能（如 text-to-speech）合并到 AI Town 主项目表现出浓厚兴趣。

- **解决用户体验问题**：一位用户遇到了对话关闭过快导致无法舒适阅读的问题，这暗示了 AI Town 需要改进用户界面和可访问性。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **缩减日志规模**：一位成员开发的新流水线通过移除**冗余日志**来降低成本。他们推荐了一个[工具](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)，用于选择“verbose logs”流水线来实现这一目标。

- **讨论部署方案**：成员们讨论了用于 **reranking** 和 **query extraction** 的云端/本地（cloud-prem）部署解决方案，在没有提供更多背景的情况下寻求最佳集成实践的见解。

- **金融 RAG 微调**：有人询问是否可以**微调 Cohere 模型**来回答金融问题，特别提到了使用 SEC 文件集成 **RAG (Retrieve and Generate)** 系统。

- **Aya23 模型的限制性使用**：已明确 **Aya23 模型**严格用于研究目的，不提供商业用途，这影响了它们在初创公司环境中的部署。

- **机器人玩游戏**：一位成员发布了由 **Cohere Command R** 驱动的游戏机器人 **Create 'n' Play**，拥有“超过 100 款基于文本的游戏”，旨在促进 Discord 上的社交互动。该项目的开发情况和目的可以在这篇 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios)中找到。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **推理与训练的现实**：对话强调了 AI 训练中的性能数据，特别是关于“仅推理（inference only）”话题的简单询问如何迅速转向关注训练计算需求的复杂领域。

- **FLOPS 决定训练速度**：讨论的一个关键点是，AI 模型训练在实践中受限于每秒浮点运算次数（FLOPS），特别是在采用 **teacher forcing** 等增加有效 **batch size** 的技术时。

- **期待 Hopper 显卡支持 FP8**：社区对 **Hopper** 显卡在 fp8 原生训练方面的潜力表现出热情，突显了利用尖端硬件提高训练吞吐量的浓厚兴趣。

- **消除 fschat 的版本混淆**：建议成员通过重新安装来修复 **fschat** 问题，因为存在错误的版本标识符，这体现了对集体生态系统中细节的严谨关注。

- **当 CUTLASS 技高一筹时**：讨论明确了设置 `CUTLASS_PATH` 的重要性，强调了 **CUTLASS** 在优化深度学习至关重要的矩阵运算中的作用，突显了该组织对优化算法效率的关注。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apache 欢迎 YI 和 YI-VL 模型**：**YI 和 YI-VL（多模态 LLM）模型**现在采用 **Apache 2.0** 许可证，正如 [@_philschmid 的推文](https://fxtwitter.com/_philschmid/status/1795343334225129570)所庆祝的那样；它们在这次许可更新中加入了 1.5 系列。

- **Gemini 1.5 挑战王座**：**Gemini 1.5 Pro/Advanced** 已攀升至排行榜第 2 位，并有超越 GPT-4o 的野心，而 **Gemini 1.5 Flash** 则自豪地占据了第 9 位，险胜 **Llama-3-70b**，正如 [lmsysorg 的推文](https://x.com/lmsysorg/status/1795512202465845686?s=46)所宣布的那样。

- **OpenAI 董事会被蒙在鼓里**：一位前 OpenAI 董事会成员透露，董事会事先并未获悉 **ChatGPT** 的发布，而是像公众一样通过 [Twitter](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298) 得知的。

- **Toner 对 OpenAI 领导层投下重磅炸弹**：OpenAI 前董事会成员 Helen Toner 在 [TED 播客节目](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3)中指责 **Sam Altman** 营造了有毒的工作环境且行为不诚实，并呼吁对“AI 公司进行外部监管”。

- **社区对 OpenAI 的爆料感到震惊**：针对 Helen Toner 的严重指控，社区表达了震惊，并对行业可能发生重大变革充满期待，Natolambert 甚至发问 Toner 是否会“从字面意义上拯救世界？”

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **专家认可的常用 LLM 排行榜**：[chat.lmsys.org 上的排行榜](https://chat.lmsys.org/?leaderboard)受到了用户的关注和认可，被认为是比较各种大语言模型（LLM）性能的可靠资源。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **保护本地 AI 端点至关重要**：一位成员**强调了保护 AI 模型本地端点的重要性**，建议使用 **DNS SRV 记录**和公钥来确保经过验证且值得信赖的本地 AI 交互，并开玩笑说未经验证的模型可能会导致意外购买乡村音乐或喂食松鼠的风险。
- **故障排除警报：发现 Llamafile 错误**：一位运行 **Hugging Face llamafile**（具体为 `granite-34b-code-instruct.llamafile`）的用户报告了一个“unknown argument: --temp”的错误，这表明模型部署过程的实施阶段可能存在问题。
- **关注正在运行的模型**：在一份澄清中指出，无论本地 `localhost:8080` 运行的是什么模型（如 *tinyllama*），它都将是默认模型，chat completion 请求中的 `model` 字段对操作没有影响。这表明所使用的 **llamafiles** 采用的是**单模型运行范式**。
  
**提到的链接**：[granite-34b-code-instruct.llamafile](https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true)

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **请求 R1 更新**：一位成员表达了对 **R1** 未来发展的期待，并幽默地提到如果它达不到预期，可能会变成一个“漂亮的镇纸”。
- **社区寻求明确性**：社区内对 **R1** 相关更新有着共同的好奇心，成员们正在积极寻求和分享信息。
- **等待支持团队的关注**：关于一封电子邮件向 **OI 团队**发出的询问正在等待回复，这表明需要改进沟通或支持机制。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **发现一座“鬼城”**：一位成员提出担忧，认为该服务器似乎**无人管理**，这可能意味着管理员的疏忽或有意为之的放任政策。
- **通知未能送达**：在服务器中尝试使用 **@everyone** 标签失败，这表明权限受限或存在技术故障。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **关于后端自动化的 LLM 咨询未得到解答**：一名成员询问课程是否涵盖使用 Large Language Models (LLM) 自动化后端服务，该问题尚未得到解答。该咨询旨在寻求有关 LLM 在自动化后端流程中实际应用的见解。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道划分的详细摘要和链接

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1244727054499582113)** (91 条消息🔥🔥): 

- **Google Vision OCR 的优缺点**：几位成员讨论了 **Google Vision OCR**，指出其结果尚可且在字符层面有详细的置信度指标，但其**开发者体验（developer experience）被批评**非常糟糕。提到了 **Microsoft Azure** 和开源的 **Mindee Doctr** 作为更好或更简单的替代方案。
- **Gradio Office Hours 热度**：Hugobowne 宣布即将与 Freddy Boulton 进行 Office Hours 会话，引发了热烈讨论，并出现了一些关于 AI 周边（如 **Scikit 连帽衫**和 **Mistral T恤**）的玩笑。分享了 [Freddy 的网站](https://www.freddyboulton.com/)链接以及关于构建多模态 Chatbot 组件的 [YouTube 教程](https://youtu.be/IVJkOHTBPn0?si=tsM6PouRRNixaroH)。
- **Modal 作为全能云服务**：Charles 强调了 **Modal 的全栈能力**，涵盖数据 ETL、Fine-tuning、推理、Web 托管等，将其定位为处理各种任务的 Serverless 解决方案。这引发了关于**全栈应用**效率的讨论，并分享了 [S3 bucket 挂载](https://modal.com/docs/examples/s3_bucket_mount)和 [Web 抓取](https://modal.com/docs/examples/web-scraper)的示例。
- **选择构建 Agent 的 LLM 库**：Lalithnarayan 和 Chongdashu 讨论了构建 LLM 应用和 Agent 的众多选择，如 **Langchain**、**LlamaIndex** 和 **DSPy**。考虑到 v0.2 最近的重大变更（breaking changes），他们的结论是建议从 Langchain v0.1 开始，并在必要时进行升级。
- **预训练的数据混合（Data mixture）**：Thechurros 询问了在使用合成数据进行持续预训练（Continuing Pre-training）时正确的**数据混合比例**。Jeremy Howard 提出了一个粗略的指导方针，即 20% 的现有数据，并提到使用经过筛选的 Common Crawl 子集，同时强调目前缺乏定论性的研究，并提及了相关工作，如 [Zephyr 数据混合研究](https://arxiv.org/html/2402.16827v2)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15682">The Road Less Scheduled</a>：现有的不需要指定优化停止步数 T 的学习率调度方案，其表现远不如依赖于 T 的学习率调度方案。我们提出了一种方法...</li><li><a href="https://x.com/NielsRogge/status/1795106366752723094?t=2Fe5vPhNOJF-84AkgrajTw&s=19">Niels Rogge (@NielsRogge) 的推文</a>：事实证明我的 Idefics2 notebook 同样适用于 PaliGemma 的 Fine-tuning :) 在这里查看：https://github.com/NielsRogge/Transformers-Tutorials/tree/master/PaliGemma 对于 JSON 用例，一个微型 VLM ...</li><li><a href="https://tenor.com/view/major-payne-dance-the-robot-dancing-moves-gif-17644148">Major Payne Dance GIF - Major Payne Dance The Robot - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.freddyboulton.com/">Freddy A. Boulton</a>：未找到描述</li><li><a href="https://github.com/eugeneyan/visualizing-finetunes/blob/main/1_prep_data.ipynb">visualizing-finetunes/1_prep_data.ipynb at main · eugeneyan/visualizing-finetunes</a>：通过在 GitHub 上创建一个账户来为 eugeneyan/visualizing-finetunes 的开发做出贡献。</li><li><a href="https://www.dmlbl.com/technical_blog.html">技术博客</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/mindee/doctr">docTR - mindee 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/modules/agents/">Agents | 🦜️🔗 LangChain</a>：Agent 的核心思想是使用语言模型来选择要执行的一系列动作。</li><li><a href="https://arxiv.org/html/2402.16827v2">A Survey on Data Selection for Language Models</a>：未找到描述</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: 支持 90 多种语言的 OCR、布局分析、阅读顺序、行检测</a>：支持 90 多种语言的 OCR、布局分析、阅读顺序、行检测 - VikParuchuri/surya</li><li><a href="https://modal.com/docs/examples/s3_bucket_mount">在 S3 的 Parquet 文件上使用 DuckDB 分析纽约黄色出租车数据</a>：本示例展示了如何使用 Modal 完成经典的数据科学任务：将表结构数据加载到云存储中、进行分析并绘制结果。</li><li><a href="https://modal.com/docs/examples/web-scraper">一个简单的 Web 抓取工具</a>：在本指南中，我们将通过编写一个简单的 Web 抓取工具来向您介绍 Modal。我们将逐步解释 Modal 应用程序的基础知识。</li><li><a href="https://latent-space-xi.vercel.app/til/create-a-conda-env-for-axolotl">Latent Space</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1244778919274745936)** (10 messages🔥): 

- **针对制药领域的 LLM 优化**：讨论围绕制药/生物技术领域内 LLM 的五个极具吸引力的用例展开，例如*加速药物研发（Accelerated Drug Discovery）*和*个性化医疗（Personalized Medicine）*。每个用例都强调了使用相关数据集进行 Fine-tuning，并利用 **RAG** 来获取上下文相关的特定信息。

- **法律文档摘要**：人们对利用 LLM *为法律诉讼中的证据开示文档（discovery documents）生成摘要*表现出浓厚兴趣。建议通过 Fine-tuning 使模型适应法律摘要的特定风格和相关性标准，这被认为是至关重要的。

- **销售邮件的定制化开场白**：一位成员讨论了如何生成*销售邮件的个性化首句*。提议使用成功邮件开场白和收件人画像的数据集对模型进行 Fine-tuning，以提高参与度和回复率。

- **通过 Multi-Agent LLMs 进行跨学科协作**：引入了创建一个 Multi-agent LLM 模型的想法，每个 Agent 专注于一个利基领域，以解决复杂的跨学科问题。该方案将涉及使用 **RAG** 提供额外上下文，并针对每个 Agent 的特定领域进行 Fine-tuning。

- **用于技术支持和故障诊断的聊天机器人**：讨论包括为聊天机器人训练模型，使其能够根据历史 Slack 和 Jira 数据*回答技术问题*，并利用复盘文档（post-mortem documentation）诊断活动事件。建议通过 Fine-tuning 来增强这些聊天机器人的效能。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1244727466719842385)** (9 messages🔥): 

- **Circleback 转录困扰**：一位用户提到 Hamel 一直在使用 Circleback 进行转录和记录，但找不到访问它们的链接。Hamel 回复说 Circleback 无法转录这些大型会议，但指出可以导出转录文本，并请求协助将其上传到课程章节中。

- **Zoom 转录即将上线**：Dan 表示 Zoom 使用独立的步骤/任务来创建转录，他已经启动了所有课程的任务。他提到将在下午把转录文本上传到课程页面并提供更新，并澄清这些将是原始转录文本而非摘要。

- **LLM 摘要的机会**：Hamel 建议，利用转录文本创建摘要对于某些人来说是一个使用 LLM 的好机会。这意味着课程参与者在完善内容方面有贡献空间。

- **晚加入者寻求指导**：来自加尔各答的新成员 Shalini 询问了课程进度，以及除了补看录像外还应该关注什么，因为她在 Week 3 才加入。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1244759662306529331)** (87 条消息🔥🔥): 

- **DeepSpeed 配置难倒 Llama3-70B 训练者**：一位用户在多 GPU 上**训练 Llama3-70B 时遇到了 OOM 错误**，怀疑是 DeepSpeed 设置配置不当。他们收到的建议包括减小 batch size、修改序列长度以及调整配置文件中的目标层，同时分享了相关的 [WandB 运行记录](https://wandb.ai/dailyco/khk-llama-3-70b)和[配置文件](https://gist.github.com/kwindla/bea28ce3ffe10e130dbd272e2fc6037f)。

- **为 Modal 调试模型配置**：用户尝试了不同的策略，包括将梯度累积（gradient accumulation）设置为 `1` 并关闭评估，以调试 OOM 问题。他们不断迭代配置设置并分享运行记录，以更好地理解内存分配。

- **在 Modal 上运行 Lorax**：一位用户在解决了 Dockerfile ENTRYPOINT 问题（通过清除现有入口点并将 `lorax-launcher` 调用包装在 `@modal.web_server` 装饰器中）后，成功在 Modal 上运行了 **Lorax**。[参考代码](https://github.com/predibase/lorax/blob/main/Dockerfile)。

- **选择训练用的 GPU 实例**：针对如何选择合适实例类型的通用查询得到了解答，建议如果显存（VRAM）限制允许，先从 **A10G GPU** 开始，未来可能会转向 **L40S GPU**。

- **模型权重缓存机制**：一位用户寻求对其使用 Modal 缓存机制缓存模型权重的方法进行确认，并详细说明了其设置以获取反馈，该方法已获得验证。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/kwindla/81cbec28a5893f682984549ecc05dcfa">Llama-3-70B 配置 (OOM)</a>：Llama-3-70B 配置 (OOM)。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b?nw=nwuserkwindla">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/ybeu4z50/logs?nw=nwuserkwindla">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - 配置选项</a>：未找到描述</li><li><a href="https://github.com/predibase/lorax/blob/main/Dockerfile">predibase/lorax 的 Dockerfile</a>：可扩展至数千个微调 LLM 的 Multi-LoRA 推理服务器 - predibase/lorax</li><li><a href="https://github.com/huggingface/peft/blob/39c60ffca9c1d1cc606a16654cfe9cd66b363a70/src/peft/tuners/lora/config.py#L51-L58)">peft/src/peft/tuners/lora/config.py</a>：🤗 PEFT：尖端的参数高效微调。- huggingface/peft</li><li><a href="https://modal.com/docs/guide/custom-container#entrypoint">自定义容器</a>：本指南将引导你如何定义 Modal 函数和应用程序运行的环境。</li><li><a href="https://modal.com/docs/guide/webhooks#non-asgi-web-servers">Web 端点</a>：Modal 提供了几种将函数公开为 Web 端点的方法。你可以通过一行代码将任何 Modal 函数转换为 Web 端点，或者使用 FastAPI 等框架提供完整的应用程序...</li><li><a href="https://gist.github.com/mtisz/2e9f7d8acb1a65f0b58f2427a402f387">Llama-3-70B QLoRA 的 Axolotl 配置</a>：Llama-3-70B QLoRA 的 Axolotl 配置。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/8pdffbhe">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 &mdash; PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://gist.github.com/kwindla/bea28ce3ffe10e130dbd272e2fc6037f">Llama-3-70B 配置（在单 GPU 上运行正常，无 DeepSpeed；合并期间在多 GPU 上出现 OOM）</a>：Llama-3-70B 配置（在单 GPU 上运行正常，无 DeepSpeed；合并期间在多 GPU 上出现 OOM）- khk-llama-3-70B.yml</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/80s40cgd">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/9vrwylua">dailyco</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/dailyco/khk-llama-3-70b/runs/8tk2wy4k?nw=nwuserkwindla">dailyco</a>：Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1244732945080520724)** (11 条消息🔥): 

- **Prince Canuma 发布 LLaMA 3 微调资源**：一位用户分享了 **Prince Canuma** 发布的新视频和权重，内容关于将 **LLaMA 3** 从 8B 精炼（refining）至 6B [视频链接](https://youtu.be/tMvC_bsAwyQ?si=23eN1WIK5Izsep80)。
  
- **面向初学者的 OpenAI 微调指南**：建议初学者参考 [OpenAI 微调指南](https://platform.openai.com/docs/guides/fine-tuning/when-to-use-fine-tuning) 以了解何时适合进行微调。

- **困扰于 .pth 到 safetensors 的转换**：一位用户请求帮助将**微调后的垃圾邮件分类器**从 .pth 文件转换为 **safetensors** 格式，以便在 Hugging Face 上托管。他们被引导至 [Hugging Face 文档](https://huggingface.co/docs/safetensors/en/convert-weights#) 并获得了额外建议。
  
- **先将模型上传至 Hugging Face hub**：建议在尝试本地转换之前，先将模型上传至 **Hugging Face hub**，进一步的指导指向了 [Hugging Face 仓库](https://huggingface.co/spaces/safetensors/convert/tree/main) 中的 **convert.py** 文件。

- **保持 HF token 安全**：一位用户被告知其 **Hugging Face token** 意外泄露，建议将此讨论移至私有频道。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/safetensors/en/convert-weights#">将权重转换为 safetensors</a>：未找到描述</li><li><a href="https://youtu.be/tMvC_bsAwyQ?si=23eN1WIK5Izsep80">在 PyTorch 中从零开始编写 Llama 3 - 第 2 部分</a>：在此视频系列中，你将学习如何从零开始训练和微调 Llama 3 模型。目标是在 PyTorch 中从零开始编写 LLaMA 3 以创建模型...</li><li><a href="https://huggingface.co/spaces/safetensors/convert/tree/main">safetensors/convert at main</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1244726841873403924)** (26 条消息🔥): 

- **巧妙地缓存你的 Hugging Face 模型**：一位用户分享了将 Hugging Face 模型和数据集缓存到特定目录的技巧，以避免存储空间耗尽。他们建议设置环境变量，如 `HF_DATASETS_CACHE` 和 `HUGGINGFACE_HUB_CACHE`。

- **获取用于故障排除的额外日志**：成员们讨论了获取错误相关额外日志的困难，特别是当实例意外重启时。建议在 JupyterLab 终端运行长时间作业以获得更清晰的日志，或将 notebook 转换为 Python 脚本。

- **Conda 环境在暂停时重置**：用户报告称，当实例暂停并恢复时，他们的 conda 环境会被删除。该问题似乎与保存在 `/root` 目录下的环境有关，建议将其保存在自定义路径中。

- **跨不同邮箱管理凭据**：关于在 JarvisLabs 和其他平台使用不同邮箱导致积分（credits）出现的问题。解决方案是确保在所有相关平台使用相同的邮箱，并向课程讲师注册以获取积分。

- **自动化实例关机**：讨论了通过脚本启动和关闭实例以节省资源的方法。结果显示 `shutdown -h` 可能无法直接工作，建议使用 API 实现完全自动化。

**提到的链接**：<a href="https://jarvislabs.ai/docs/env#creating-and-managing-a-new-environment-using-conda">创建自定义环境 | Jarvislabs</a>：随着项目变得复杂，你可能希望创建并维护独立的虚拟环境。

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1245005625658708030)** (19 messages🔥): 

- **HF 额度面向个人而非组织**：成员们澄清了**补助金（grants）是提供给参加课程的个人**，而非组织的。即便如此，拥有一个作为组织成员的账号也是可以接受的。
- **确保在周五前应用 HF 额度**：一位成员提醒大家，**额度将在报名截止后的周五发放**。他们强调了填写通过邮件发送的 HF 表单的重要性，以确保额度被正确应用。
- **将 PyTorch 模型转换为 safetensors 遇到麻烦**：一位用户讨论了在生产环境中将 **PyTorch 模型转换为 safetensors** 的挫败感。他们提到参考了 [GitHub - LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 的教程，并提出了关于处理特定文件格式和所需推理代码的问题。

**提到的链接**：<a href="https://github.com/rasbt/LLMs-from-scratch">GitHub - rasbt/LLMs-from-scratch: Implementing a ChatGPT-like LLM in PyTorch from scratch, step by step</a>：从零开始在 PyTorch 中逐步实现类似 ChatGPT 的 LLM - rasbt/LLMs-from-scratch

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1244929654767358031)** (6 messages): 

- **在 Replicate 设置中选择 GitHub 邮箱**：会议澄清了在 Replicate 注册时，应使用与你的 GitHub 账号关联的邮箱。对于首选邮箱设置，建议用户使用已与 Replicate 账号关联的邮箱，以避免混淆。
- **Replicants 身份确认**：一位自称“Replicate 员工”的人幽默地指出，他们现在被称为“Replicants”。这突显了 Replicate 社区包容且有趣的文化。
- **澄清邮箱混淆问题**：在使用 GitHub 注册后，一位用户对哪个邮箱会被检查以获取 Replicate 额度表示困惑。一位 Replicant 确认可以通过 Replicate (GitHub) 用户名进行追踪，并表示如有任何问题可以私信（DM）。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245072716336463952)** (1 messages): 

- **对 Langsmith 延迟图表的喜爱**：一位成员表达了对 Langsmith 中延迟图表的赞赏，并提出了两个问题。他们询问是否可以查看*所有模型的历史延迟图表*，而不仅仅是正在使用的模型；以及模型的地理分布如何影响延迟，并询问了关于 **OpenAI, Anthropic, and Google** 模型对区域（zones）的依赖性。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1244866175784189963)** (3 messages): 

- **OpenPipe 限制数据格式**：一位成员注意到 OpenPipe 目前仅接受 **OpenAI chat fine-tuning 格式**的外部数据，这限制了上传 Alpaca 格式 JSONL 的能力。他们对最近一次演讲中展示的数据集创建界面表示感兴趣。
- **未来可能支持更多数据格式**：另一位成员回应了该查询，澄清未来可能会支持**额外的数据格式**。目前的决定是因为现有格式是“大多数用户最熟悉的”。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1244749171144261694)** (64 条消息🔥🔥): 

- **高质量数据对 Fine-Tuning 至关重要**：“你需要给它喂高质量的东西，” 强调了高质量数据对于 Fine-Tuning 的重要性，这并不是改变目标函数，而是可能改变优化器和超参数。高质量数据能最大化预构建模型在特定应用场景下的性能。

- **Loss 曲线解读与过拟合担忧**：关于 Mistral8x7b Fine-Tuning 的讨论强调了基于曲线解读的过拟合问题，建议使用 Validation Loss。参与者还辩论了学习率（Learning Rate）的调整，建议指出基础学习率可能过高。

- **优化问题与配置微调**：探讨了训练中 Loss 增加和停滞的原因，建议包括可能过高的学习率，以及使用更好的初始化或类似 `rslora` 的配置。参与者分享了 Weights & Biases (wandb) 的运行链接以便协作调试。

- **精选与合成数据集的挑战**：Rumbleftw 等人讨论了精选数据集的复杂性以及针对近期发布模型的 Fine-Tuning 特定配置。他们解决了 Tokenizer 问题、合适的 Special Tokens，并处理了一个约 `165k` 数据点的数据集。

- **共享资源与配置建议**：社区成员分享了有用的资源，包括来自 [Cedric Chee 的 GitHub Gist](https://gist.github.com/cedrickchee/6e9cff188d24a5b4429af1845f912688) 的研讨会笔记，并讨论了来自 [OpenAccess-AI-Collective 的 GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json) 的 zero3 和 zero3_bf16 DeepSpeed 配置。还强调了正确配置以避免模型并行期间出现显存溢出（Out-of-Memory）问题的重要性。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wandb.ai/settings">settings</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3-">vapi</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3--Vmlldzo4MTE3MjAz?accessToken=3qbn8ulplg2igvgts7fwnqgoekzyubyz191mb6y8jxntdyv44zmw6s9l55pemue9">Func-calling: Mistral8x7bv0.3</a>: 使用交互式图表发布您的模型洞察，包括性能指标、预测和超参数。由 Rajdeep Ghosh 使用 Weights &amp; Biases 制作</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3_bf16.json">axolotl/deepspeed_configs/zero3_bf16.json at main · OpenAccess-AI-Collective/axolotl</a>: 尽管去问 Axolotl 问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://wandb.ai/vapi/mistral-func/reports/Func-calling-Mistral8x7bv0-3--Vmlldzo4MTE3MjAz?accessToken=3qbn8ulplg2igvgts7fwnqgoekzyubyz191mb6y8jxntdyv44zmw6s9l55pemue9#axolotl-config">Func-calling: Mistral8x7bv0.3</a>: 使用交互式图表发布您的模型洞察，包括性能指标、预测和超参数。由 Rajdeep Ghosh 使用 Weights &amp; Biases 制作</li><li><a href="https://gist.github.com/">Discover gists</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/cedrickchee/6e9cff188d24a5b4429af1845f912688">Fine-Tuning Workshop 2: Fine-Tuning with Axolotl</a>: Fine-Tuning 研讨会 2：使用 Axolotl 进行 Fine-Tuning。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/jingkaihe/llm-finetuning?tab=readme-ov-file#fine-tuning-on-promql-data">GitHub - jingkaihe/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Llama/Mistral/CodeLlama 等模型的 Fine-Tuning 指南 - jingkaihe/llm-finetuning</li><li><a href="https://github.com/jingkaihe/llm-finetuning/blob/main/data/promql.tiny.jsonl">llm-finetuning/data/promql.tiny.jsonl at main · jingkaihe/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的 Fine-Tuning 指南 - jingkaihe/llm-finetuning</li><li><a href="https://github.com/jingkaihe/llm-finetuning/blob/main/config/mistral-promql.yml">llm-finetuning/config/mistral-promql.yml at main · jingkaihe/llm-finetuning</a>: Llama/Mistral/CodeLlama 等模型的 Fine-Tuning 指南 - jingkaihe/llm-finetuning</li><li><a href="https://wandb.ai/jingkaihe/memorize-sqlqa/reports/train-loss-24-05-28-10-26-55---Vmlldzo4MTIxNzI3">Weights & Biases</a>: Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1245058358675767470)** (461 条消息🔥🔥🔥):

- **关于 LLM Evals 的精彩分享**：Eugene Yan 分享了评估 LLM 的详尽流程，涵盖了评估的迭代以及它们如何与产品指标相互关联。查看他的 [visualizing finetunes 仓库](https://github.com/eugeneyan/visualizing-finetunes) 以获取详细的 notebook。
- **强调数据日志和评估工具**：重点介绍了 [LangFuse](https://langfuse.com)、[ChainForge](https://chainforge.ai/) 和 [EvalGen](https://arxiv.org/abs/2404.12272) 等工具，它们在提高追踪、日志记录和评估效率方面具有潜力。
- **对性能指标的关注**：讨论强调了 BLEU 等传统指标面临的挑战，以及对动态、特定任务评估的需求，正如 [Eugene Yan 的文章](https://eugeneyan.com/writing/evals/) 中所概述的那样。
- **引人入胜的实践案例**：分享包含了具有详细方法论和技术见解的实践案例，并提供了如 [notebook 系列](https://github.com/eugeneyan/visualizing-finetunes) 等资源来阐明微调过程。
- **丰富的资源分享**：本次分享和讨论汇集了大量的资源链接，从 Eugene 关于 [prompting 基础](https://eugeneyan.com/writing/prompting/) 的实践指导文章，到如 [Reversal Curse](https://arxiv.org/abs/2309.12288) 等最新研究。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://johnowhitaker.]">未找到标题</a>：未找到描述</li><li><a href="https://langfuse.com/">Langfuse</a>：开源 LLM 工程平台 - LLM 可观测性、指标、评估、Prompt 管理。</li><li><a href="https://x.com/eugeneyan">来自未定义的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.12288">反转诅咒：在“A 是 B”上训练的 LLM 无法学会“B 是 A”</a>：我们揭示了自回归大语言模型（LLM）在泛化方面一个令人惊讶的失败。如果模型是在“A 是 B”形式的句子上训练的，它不会自动泛化...</li><li><a href="https://news.ycombinator.com/item?id=37843907">未找到标题</a>：未找到描述</li><li><a href="https://github.com/shreyashankar">shreyashankar - 概览</a>：加州大学伯克利分校计算机科学博士生。shreyashankar 拥有 63 个代码仓库。在 GitHub 上关注其代码。</li><li><a href="https://x.com/BEBischof">来自未定义的推文</a>：未找到描述</li><li><a href="https://forums.fast.ai/">fast.ai 课程论坛</a>：fast.ai 课程、软件和研究论坛</li><li><a href="https://github.com/eugeneyan/visualizing-finetunes">GitHub - eugeneyan/visualizing-finetunes</a>：通过在 GitHub 上创建账户，为 eugeneyan/visualizing-finetunes 的开发做出贡献。</li><li><a href="https://eugeneyan.com/writing/prompting/">Prompting 基础及如何有效应用</a>：结构化输入/输出、prefilling、n-shots prompting、chain-of-thought、减少幻觉等。</li><li><a href="https://arxiv.org/abs/2401.03038">SPADE：为大语言模型流水线合成数据质量断言</a>：大语言模型（LLM）正越来越多地作为流水线的一部分被部署，这些流水线会重复处理或生成某种数据。然而，部署的一个常见障碍是频繁且通常...</li><li><a href="https://arxiv.org/abs/2404.12272">谁来验证验证者？使 LLM 辅助的 LLM 输出评估与人类偏好保持一致</a>：由于人工评估的繁琐性以及基于代码评估的局限性，大语言模型（LLM）正越来越多地被用于辅助人类评估 LLM 的输出。然而 LLM-...</li><li><a href="https://pytest-vcr.readthedocs.io/en/latest/#quick-start">欢迎使用 pytest-vcr - pytest-vcr</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2305.14296">USB：跨任务和领域的统一摘要基准</a>：虽然 NLP 社区已经产生了许多摘要基准，但没有一个能提供丰富的标注，以同时解决与控制和可靠性相关的许多重要问题...</li><li><a href="https://docs.google.com/presentation/d/1GC868XXjhxOpQEt1jUM79aW0RHjzxPp0XhpFHnYH760/edit#slide=id.p">Spellgrounds for Prodigious Prestidigitation</a>：Spellgrounds for Prodigious Prestidigitation，Bryan Bischof 博士，Hex AI 负责人</li><li><a href="https://x.com/HamelHusain/status/1795526367637049629">来自 Hamel Husain (@HamelHusain) 的推文</a>：我和同事们将关于 LLM 的实用建议浓缩到了这个由三部分组成的系列中。干货满满。这是我最喜欢的截图部分摘录。建议来自：@eugeneyan, @BEBi...</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM">扩展 LLM 的“氛围检查 (Vibe Checks)” - Shreya Shankar | Stanford MLSys #97</a>：斯坦福 MLSys 研讨会系列第 97 集！扩展 LLM 的“氛围检查”。演讲者：Shreya Shankar。简介：Shreya Shankar 是计算机科学专业的博士生...</li><li><a href="https://arxiv.org/abs/2404.13076">LLM 评估者会识别并偏好自己生成的内容</a>：使用大语言模型（LLM）进行自我评估已被证明不仅在基准测试中很有价值，在奖励建模、Constitutional AI 和自我改进等方法中也很有用。但新的偏见也随之引入...</li><li><a href="https://sqlmodel.tiangolo.com/">SQLModel</a>：SQLModel，Python 中的 SQL 数据库，旨在实现简单性、兼容性和健壮性。</li><li><a href="https://chainforge.ai/">ChainForge：用于 Prompt 工程的可视化编程环境</a>：未找到描述</li><li><a href="https://www.traceloop.com/docs/openllmetry">什么是 OpenLLMetry？- traceloop</a>：未找到描述</li><li><a href="https://www.amazon.co.uk/Noise-Daniel-Kahneman/dp/0008308993">未找到标题</a>：未找到描述</li><li><a href="https://discord.gg/yX2TdaFt8t">加入 llm-fine-tuning Discord 服务器！</a>：查看 Discord 上的 llm-fine-tuning 社区 - 与其他 1468 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://hamel.dev/blog/posts/evals/#automated-evaluation-w-llms">- 你的 AI 产品需要评估 (Evals)</a>：如何构建特定领域的 LLM 评估系统。</li><li><a href="https://eugeneyan.com/writing/evals/">有效与无效的特定任务 LLM 评估 (Evals)</a>：用于分类、摘要的评估</li>

on, translation, copyright regurgitation, and toxicity.</li><li><a href="https://tenor.com/view/im-proud-of-you-dan-levy-david-david-rose-schitts-creek-gif-20773745">Im Proud Of You Dan Levy GIF - Im Proud Of You Dan Levy David - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/waiting-still-gif-20331665">Waiting Still GIF - Waiting Still - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: 未找到描述</li><li><a href="https://hex.tech/">Hex - Magical tools for working with data together</a>: Hex 是一个用于数据科学和分析的现代数据平台。提供协作式数据笔记本、精美的数据应用、神奇的 AI 辅助以及企业级安全。</li><li><a href="https://hex.tech/product/magic-ai/">Hex Magic | Smarter, faster analysis with a little Magic | Hex </a>: 通过使用 Magic AI 编写查询、构建图表和修复 Bug，每周节省数小时。</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM&t=139s">Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97</a>: Stanford MLSys 研讨会系列第 97 集！为 LLMs 扩展 “氛围检查 (Vibe Checks)”。演讲者：Shreya Shankar。简介：Shreya Shankar 是计算机科学博士生...</li><li><a href="https://x.com/tomaarsen/status/1795425797408235708">Tweet from tomaarsen (@tomaarsen)</a>: ‼️ Sentence Transformers v3.0 发布了！你现在可以使用多 GPU 训练、bf16 支持、损失日志记录、回调等功能来训练嵌入模型。我还发布了 50 多个训练数据集及更多内容...</li><li><a href="https://arize.com/blog/breaking-down-evalgen-who-validates-the-validators/">Breaking Down EvalGen: Who Validates the Validators?</a>: 关于 EvalGen（一种 LLM 辅助评估方法）你需要知道的一切。还包括一些给 LLM 应用构建者的启示。</li><li><a href="https://www.youtube.com/watch?v=ua93WTjIN7s">LlamaIndex Workshop: Evaluation-Driven Development (EDD)</a>: ​在本次工作坊中，我们教你如何进行 “评估驱动开发” (EDD) 以构建生产级 LLM 应用。这包括以下内容：1. 定义...</li><li><a href="https://www.usebraintrust.com/">Braintrust | The First User-Owned Talent Network</a>: Braintrust 将组织与顶尖技术人才联系起来，以完成战略项目并推动创新。 </li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry: Open-source observability for your LLM application, based on OpenTelemetry</a>: 基于 OpenTelemetry 的 LLM 应用开源可观测性 - traceloop/openllmetry</li><li><a href="https://johnowhitaker.dev/dsc/2024-01-23-tips.html">johnowhitaker.dev – A few tips for working on high-surface-area problems</a>: 未找到描述</li><li><a href="https://www.traceloop.com/docs/openllmetry/introduction">What is OpenLLMetry? - traceloop</a>: 未找到描述</li><li><a href="https://www.langchain.com/langsmith">LangSmith</a>: 让你的 LLM 应用从原型走向生产。</li><li><a href="https://pydantic.dev/logfire">Pydantic Logfire | Uncomplicated observability</a>: Logfire 是一种新型的可观测性平台，建立在与 Pydantic 相同的信念之上——即最强大的工具也可以易于使用。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1244895249273454643)** (2 条消息): 

- **理解 Colbert 的输出**：一位成员承认目前还不清楚 **Colbert** 的输出结果，并计划运行代码进行检查。他们似乎渴望获得关于其结果的第一手见解。

- **对讨论 Sparse Embeddings 和 M3 的兴趣**：一位成员表示希望深入讨论 **sparse embeddings 和 M3**，尽管这被认为是 "RAG 基础"。这表明了他们有兴趣扩展到通常涵盖的常规话题之外。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1244767005664608338)** (31 messages🔥): 

- **Axolotl 70B 模型加载错误**：一位用户在使用两块 RTXA6000 GPU 加载 70B 模型的 checkpoint 分片时遇到错误。错误与 "torch.distributed.elastic.multiprocessing.errors.ChildFailedError" 相关，导致进程在完成 93% 时失败。
- **对 WandB 服务成本的担忧**：多位用户讨论了在大规模使用时 WandB 的高昂成本，并建议使用 **Aim** 和自托管的 **MLflow** 作为更具性价比的替代方案。一位用户提到，这些工具的主要优势在于协作，建议独立开发者使用更简单的解决方案。
- **对 WandB 的偏好**：尽管成本较高，一些用户仍因其相比其他工具更佳的易用性而偏好 WandB。
- **Axolotl 的 Google Colab 调试**：一位用户提交了一个 [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1662)，旨在修复在 Google Colab 笔记本中运行 Axolotl 的问题，包括更新配置和安装步骤。
- **TinyLlama 的推理差异**：用户报告称，在训练后使用 TinyLlama 模型进行推理时，输出结果不一致。潜在问题包括 Prompt 使用不当，以及通过检查配置文件和讨论 **sample packing** 优化所发现的训练与推理设置之间的差异。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml">axolotl/examples/llama-3/qlora-fsdp-70b.yaml at main · OpenAccess-AI-Collective/axolotl</a>：尽管去问（axolotl）问题吧。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1662">Fix Google Colab notebook 2024-05 by maciejgryka · Pull Request #1662 · OpenAccess-AI-Collective/axolotl</a>：使 Google Colab 笔记本运行的几项修复（已在 L4 GPU 上测试）：在设置中包含 mlflow 安装，更新配置以镜像 examples/ 中最新的 tinyllama QLORA 配置...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L31-L97),">axolotl/src/axolotl/prompters.py at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>：尽管去问（axolotl）问题吧。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1244762145304875108)** (22 条消息🔥): 

- **分享了将 DataFrame 转换为 JSONL 的 Python 函数**：一位用户分享了将 DataFrame 转换为 JSONL 格式的代码，并认为这是一个简单有效的解决方案。该代码遍历 DataFrame 的行，将每一行转换为字典，并将其作为 JSON 行写入文件。

- **关于 load_in_8bit 对训练影响的讨论**：用户讨论了使用 `load_in_8bit=True` 除了减少 GPU VRAM 占用外，是否还会影响训练。观察结果包括在 `load_in_8bit=False` 时梯度表现更好，以及训练期间量化和精度的技术细节。

- **Qwen Tokenizer 的问题**：QwenCode1.5 的 Tokenizer 配置存在错误。提议的解决方案是切换到 Qwen2Tokenizer，但仍需等待来自 Qwen 团队的验证或见解。

- **对上下文窗口长度的担忧**：一位用户对 Prompt 超过模型上下文窗口长度表示担忧，并指出可能存在的异常或性能问题。他们发现使用 "rope_scaling" 参数效果不错，并提到了一款支持更长上下文窗口的模型，例如 [Llama-3 8B Gradient Instruct](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)。

- **解决 Tokenizer 中的 token_type_ids 问题**：发现了一个关键问题，即默认的 PreTrainedTokenizer 类会发出 `token_type_ids`，而某些模型类无法处理这些 ID。正确的实现应该在调整大小时遍历 `model_input_names` 中指定的可能向量，正如在 [HuggingFace 代码](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1562) 引用中所讨论的那样。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1656">Fix tokenization for CodeQwen models by artemdinaburg · Pull Request #1656 · OpenAccess-AI-Collective/axolotl</a>：更新 PromptTokenizer 中的 token_type_ids 以匹配 input_ids 和 attention_mask 的更改。描述：某些模型（如 Qwen 系列）会随 input_ids 和 attention_... 一起返回 token_type_ids。</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/lora-8b.yml">axolotl/examples/llama-3/lora-8b.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B/blob/main/tokenizer_config.json#L14)">tokenizer_config.json · Qwen/CodeQwen1.5-7B at main</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B/blob/main/config.json#L3).">config.json · Qwen/CodeQwen1.5-7B at main</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-4B/blob/main/tokenizer_config.json#L38)">tokenizer_config.json · Qwen/Qwen1.5-4B at main</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1562.">transformers/src/transformers/tokenization_utils_base.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。- huggingface/transformers
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1244780908343722116)** (78 条消息🔥🔥): 

- **早起者与熬夜者讨论 Gradio**：来自不同时区的成员分享了他们的投入，包括印度凌晨 4 点的闹钟和凌晨 2 点的深夜编码。
- **Gradio vs Streamlit 之争**：当被问及更倾向于 **Gradio** 还是 **Streamlit** 时，一位成员表示强烈支持 **Gradio**，并提到了个人偏好。
- **多模态 Chatbots 与 Google OAuth**：分享了 **Gradio 多模态 Chatbots** ([链接](https://huggingface.co/spaces/gradio/chatbot_multimodal/blob/main/run.py)) 和 **Google OAuth** 集成指南 ([链接](https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers))。
- **现场演示错误：学习机会**：成员们讨论了现场演示中的错误如何具有启发性，强调了专家的 Debug 过程是极佳的学习时刻。
- **对 Freddy Aboulton 课程的赞赏**：参与者对 Freddy 的课程表示感谢，并分享了进一步学习的资源，包括**视频链接** ([Zoom 视频](https://us06web.zoom.us/rec/share/I6RRm2606YMi6EnWVlfcXLP3BS9fXrU7NRVIjx9xCWLU_A-OwgCbIRDdeiRMctwN.5nIGeoPUDhwna0qp?startTime=1716850364000)) 和 **Gradio 性能优化技巧指南** ([链接](https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance))。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/byKr9vB9">加入 Hugging Face Discord 服务器！</a>：我们正致力于民主化优秀的机器学习 🤗 验证以链接您的 Hub 和 Discord 账号！ | 80043 名成员</li><li><a href="https://huggingface.co/spaces/gradio/chatbot_multimodal/blob/main/run.py">run.py · gradio/chatbot_multimodal at main</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/freddyaboulton/gradio_agentchatbot">gradio_agentchatbot - freddyaboulton 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.gradio.app/guides/sharing-your-app#o-auth-with-external-providers">分享你的应用</a>：Gradio 分步教程</li><li><a href="https://vanishinggradients.fireside.fm/">Vanishing Gradients</a>：与 hugo bowne-anderson 合作的数据播客</li><li><a href="https://hugobowne.github.io/">hugo bowne-anderson - 数据科学家</a>：未找到描述</li><li><a href="https://www.gradio.app/guides/setting-up-a-demo-for-maximum-performance">设置演示以获得最佳性能</a>：Gradio 分步教程</li><li><a href="https://huggingface.co/spaces/gradio/chatinterface_multimodal_main/blob/main/run.py">run.py · gradio/chatinterface_multimodal_main at main</a>：未找到描述</li><li><a href="https://www.youtube.com/live/USTG6sQlB6s?si=cB9adtLWejfTX77K">如何与 Jason Liu 一起构建糟糕的 AI 系统</a>：Jason 是一位独立顾问，他利用自己在推荐系统方面的专业知识帮助快速发展的初创公司构建 RAG 应用。他曾...</li><li><a href="https://pyodide.org/en/stable/usage/wasm-constraints.html#synchronous-http-requests-support">Pyodide Python 兼容性 &#8212; 版本 0.26.0</a>：未找到描述</li><li><a href="https://pyodide.org/en/stable/usage/faq.html#how-can-i-use-fetch-with-optional-arguments-from-python">常见问题解答 &#8212; 版本 0.26.0</a>：未找到描述</li><li><a href="https://us06web.zoom.us/rec/share/I6RRm2606YMi6EnWVlfcXLP3BS9fXrU7NRVIjx9xCWLU_A-OwgCbIRDdeiRMctwN.5nIGeoPUDhwna0qp?startTime=1716850364000">视频会议、网络会议、网络研讨会、屏幕共享</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可用于跨移动端、桌面端和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245025314015416381)** (5 条消息): 

- **M3 Mac 上的本地推理缓慢令用户沮丧**：一位用户报告称，在使用 [Modal 的 LLM 引擎](https://github.com/modal-labs/llm-finetuning.git) 在 M3 Mac 上进行本地推理时，延迟显著，每次响应超过 2 分钟。他们询问是否可以将模型部署到其他地方进行推理，例如 Hugging Face 或 Replicate。

- **部署到 Modal 作为解决方案**：另一位成员澄清说，使用 `modal deploy` 部署到 Modal 将缓解延迟问题，因为 LLM 引擎的启动将发生在 Modal 的基础设施上。他们指出，只有在长时间延迟后新实例启动时，延迟才会成为主要问题。

- **在其他地方使用权重**：同一位成员提到，也可以从 Modal volume 中提取模型权重并在外部使用，建议使用 `modal volume` CLI 命令。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245073003595698217)** (2 条消息): 

- **Langsmith Annotation Queue UI 问题**：一位用户提到 **Langsmith annotation queue UI** 看起来与演示的非常不同，并表示：*"输入和输出都是空的。当我按下 `V` 时才能看到 run。"*

- **Langsmith 部署咨询**：另一位用户询问了在 **私有云/VPC** 上部署 **Langsmith** 的可能性。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1244768670203514940)** (12 条消息🔥): 

- **Langsmith 修复学生额度设置 Bug**：Langsmith 报告了一个 Bug，即学生必须加入 Plus 才能设置额度账单。他们正在开发修复程序，允许学生在不被扣费的情况下输入信用卡信息。

- **调查问卷回复确认**：由于组织名称/ID 较为复杂，一位用户询问是否可以收到所有回复的副本以确保准确性。另一位用户表示赞同，并确认将在 5 月 30 日前进行复核。

- **Predibase Gmail 注册成功**：尽管表单说明不支持，但一位用户通过替代工作流成功使用 Gmail 地址注册了 Predibase。Danbecker 确认，如果后续出现问题，Predibase 的支持团队会及时响应。

- **手机号验证问题**：一位用户在为其个人账户绑定的 API 创建验证手机号后遇到问题，并指出更改手机号存在困难。Danbecker 建议将额度运行到该个人账户，并重新提交额度表单以更新信息。

- **新加入者额度缺失**：包括 "enginoid" 和 "seanlovesbooks" 在内的用户于 5 月 23 日加入，但尚未收到额度。他们已联系以解决此问题。
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1244728359674445824)** (659 条消息🔥🔥🔥): 

```html
- **对 imfo alpha 发布的热切期待**：一项令人兴奋的新进展即将到来，并分享了预告链接：[spectate_or on X](https://x.com/spectate_or/status/1795077451195830661?s=46)。这在社区中引发了极大的热情，并出现了与类似工具的对比。
- **关于 AI 任务实现的详细讨论**：成员们讨论了将任务分为检索（retrieval）和变更（mutation）类型，例如“获取 iPhone 15 的重量”这类查询就是这种结构的典型代表。一位成员强调，“*所有步骤都是同时发生的*”，对于需要顺序执行的任务，需要进行调整。
- **对抓取准确性的困扰**：成员们在通过 HTML 解析进行准确数据检索时面临挑战，特别是针对 Apple 和 Docker 发布说明等复杂来源。会议还讨论了 Cloudflare 问题以及使用 Playwright 处理重度 JavaScript 网站的建议。
- **高性价比 AI 模型使用的见解**：分享了关于使用各种 AI 模型成本效益的详细计算，其中结合使用 Llama3 和 Claude 模型的系统显示出显著的潜在成本节约。
- **Claude 3 模型性能的担忧**：一位成员分享了对 Claude 3 在改进 Prompt 方面不如以前有效的挫败感。这引发了关于 Prompt Engineering 以及不同任务中模型性能的更广泛讨论。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://im.fo`">未找到标题</a>: 未找到描述</li><li><a href="https://promptfoo.dev/">更快速地迭代 LLM | promptfoo</a>: 为您的用例量身定制的 LLM 评估。最大化模型质量并捕捉回归。</li><li><a href="https://abrahamjuliot.github.io/creepjs/">CreepJS</a>: 未找到描述</li><li><a href="https://x.com/spectate_or/status/1795077451195830661?s=46">来自 Daniel Kaiser (@spectate_or) 的推文</a>: 过去几周我也一直在筹备。imfo alpha 即将推出 🎉</li><li><a href="https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273">Terminator 终结者机器人 GIF - Terminator 终结者机器人观察 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/love-lovely-good-morning-with-gif-22914515">Love Lovely GIF - Love Lovely Good - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/Charles12509909/status/1794630406064795909">来自 Charles (@Charles12509909) 的推文</a>: 我成功让 GPT-4o 高亮显示屏幕上所有可点击的元素，并让它控制鼠标。它可以利用按钮坐标自主导航电脑。</li><li><a href="https://tenor.com/view/oh-wah-ah-ah-ah-anthony-vincent-down-with-the-sickness-intro-singing-disturbed-gif-16261397">Oh Wah Ah Ah Ah Anthony Vincent GIF - Oh Wah Ah Ah Ah Anthony Vincent Down With The Sickness Intro - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aws.amazon.com/blogs/aws/anthropics-claude-3-opus-model-on-amazon-bedrock/">Anthropic 的 Claude 3 Opus 模型现已在 Amazon Bedrock 上可用 | Amazon Web Services</a>: 我们生活在生成式人工智能 (AI) 时代；这是一个快速创新的时代。当 Anthropic 在 3 月 4 日发布其 Claude 3 基础模型 (FMs) 时，我们推出了 Claude 3 Sonnet，这是一款模型...</li><li><a href="https://techcrunch.com/2022/07/14/you-com-raises-25m-to-fuel-its-ai-powered-search-engine/">You.com 融资 2500 万美元以助力其 AI 驱动的搜索引擎</a>: You.com 是一家由前 Salesforce 首席科学家 Richard Socher 创立的 AI 驱动搜索引擎，已完成 2500 万美元的股权融资。</li><li><a href="https://www.apple.com/iphone-15/">iPhone 15 和 iPhone 15 Plus</a>: iPhone 15 和 iPhone 15 Plus。灵动岛。48MP 主摄，具备 2 倍长焦。全天候电池续航。USB-C。6.1 英寸和 6.7 英寸尺寸。</li><li><a href="https://www.phonearena.com/iphone-15-release-date-price-features">Apple iPhone 15 发布日期、价格和功能</a>: 未找到描述</li><li><a href="https://www.apple.com/lae/iphone-15-pro/specs/">iPhone 15 Pro 和 15 Pro Max - 技术规格</a>: 查看 iPhone 15 Pro 和 iPhone 15 Pro Max 的所有技术规格。</li><li><a href="https://www.tomsguide.com/news/iphone-15">iPhone 15：价格、规格和可用性</a>: 关于 iPhone 15 您需要了解的一切
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1244898253510672436)** (6 条消息): 

- **Daily Focus 分享了一篇情感丰富的文章**：[Daily Focus](https://www.perplexity.ai/search/Opus-50-sad-gQI59jEoSC2MtdMSvclDJQ#0) 分享了一个标题为 "Opus 50 sad" 的话题链接。用户可以通过此链接探索这段深具情感且引人深思的内容。
- **TheFuzzel 解释 Perplexity AI**：[TheFuzzel](https://www.perplexity.ai/search/What-is-Perplexity-uyV3gThHQEa1tWgRyN0sQw) 提供了一个解释什么是 Perplexity AI 的链接。对于想要了解基础知识的新用户来说，这个资源非常有益。
- **Slayer_Terrorblade 重点介绍技术会议**：[Slayer_Terrorblade](https://www.perplexity.ai/search/Upcoming-tech-conferences-aQxyrYvuSEeQLAivYygV8A) 分享了一个关于即将举行的技术会议的搜索链接。技术爱好者可以使用此链接关注重大活动的最新动态。
- **RiseNoctane 查询平均值信息**：[RiseNoctane](https://www.perplexity.ai/search/whats-the-average-Dtc8a0qdRGC7cNp4NpPyOg) 发布了一个关于平均值搜索查询的链接。这可能涉及与统计平均值相关的广泛话题。
- **Bambus89 分享五月热门电视剧**：[Bambus89](https://www.perplexity.ai/search/Beste-Fernsehsendunge-Mai-tU4zYHyWS1.9qNm6dnExhg#0) 分享了一个关于五月最佳电视剧的链接。对于寻找当月娱乐推荐的人来说，这非常有用。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1244767114552672318)** (6 条消息): 

- **不清晰的 API 输出令用户困惑**：一名成员对收到的 API 输出是否符合预期表示困惑，展示了一个内容中没有功能性链接的 JSON 对象。另一名成员建议这可能是因为缺少 **closed beta citations feature**（封闭测试版引用功能）。
- **排除 API 链接生成的故障**：为了在无法使用引用功能的情况下生成相关的视频链接，一名成员建议尝试不同的 Prompt。他们提供了一个 Prompt 示例，并建议调整 **model size 和请求的链接数量** 可能会有所帮助。
- **API 停机担忧**：一名用户询问 API 是否宕机，暗示可能存在服务中断。目前没有后续消息确认或否认此问题。
  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1244741231242907669)** (2 条消息): 

- **Stable Assistant 发布新功能**：Stability AI 宣布在 Stable Assistant 中推出新的编辑功能，利用 **Stable Diffusion 3** 生成更高质量的文生图输出。*在此免费试用您的图像* [here](https://stability.ai/stable-assistant)。

- **Chatbot Beta 版增强**：目前处于 Beta 阶段的聊天机器人集成了 **Stable LM 2 12B**，以协助处理各种文本生成任务，如博客文章、剧本和图像说明。预计很快会有持续的改进和更多功能。

- **Stability AI 与 HUG 合作推出夏季课程**：**Innovation Laboratory**（创新实验室）提供为期 4 周的 AI 模型训练指导课程，结合了 Stability AI 的工具和 HUG 的教育专业知识。请在 2024 年 6 月 25 日之前[在此](https://www.studios.thehug.xyz/lab)注册活动。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.studios.thehug.xyz/lab">HUG x Stability AI Innovation Laboratory &mdash; HUG</a>：与 Stability AI 一起发现您独特的创新，并接受来自 HUG 的实时战略、营销和创意教育。</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>：Stable Assistant 是由 Stability AI 开发的友好聊天机器人，配备了 Stability AI 的文本和图像生成技术，其特点是搭载了 Stable Diffusion 3 和 Stable LM 2 12B。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1244727100494188616)** (495 条消息🔥🔥🔥): 

- **GPU 社区计算提案**：一名成员讨论了通过提供闲置 GPU 时间来让社区分摊计算成本的想法，可能通过自定义节点甚至区块链来实现（“启动一个新的区块链，以某种方式将训练社区模型作为其挖矿功能”）。
- **关于云端 AI 助手的辩论**：人们对云端 AI 助手的隐私问题表示担忧，并因数据安全风险而反对使用此类服务（“云端的，我不会用。那是巨大的隐私隐患”）。
- **SD3 发布挫败感与对云服务的怀疑**：成员们对 SD3 权重未发布供本地使用表示沮丧，并对仅限云端的选项持怀疑态度。对 StabilityAI 的商业决策表示显著不满（“云端的 SD 3... 如果你们 SD 团队想要有所作为... 需要发布本地版 SD3”）。
- **Stable Diffusion 工作流和 Inpainting 讨论**：成员们分享了增强工作流的技巧和工具，例如使用各种扩展和 Inpainting 方法。有人建议观看 YouTube 上的教程以更好地理解（“谷歌搜索 inpainting stable diffusion... 观看一些 YouTube 视频”）。
- **ComfyUI 与其他 UI 的优劣辩论**：讨论对比了 ComfyUI 与 Forge 或 A1111 等其他 UI 的优势，强调了 ComfyUI 对技术知识的需求以及其他替代方案的易用性。一位成员分享了某些扩展如何增强 ComfyUI 的功能。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/">ComfyUI Examples</a>: ComfyUI 工作流示例</li><li><a href="https://comfyanonymous.github.io/ComfyUI_tutorial_vn/">ComfyUI Tutorial</a>: 未找到描述</li><li><a href="https://civitai.com/">Civitai: The Home of Open-Source Generative AI</a>: 探索数千个高质量的 Stable Diffusion 模型，分享您的 AI 生成艺术，并与充满活力的创作者社区互动</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1d1zw74/mobius_the_debiased_diffusion_model/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1fkt3/5_new_steerable_motion_workflows_for_trave">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant 是由 Stability AI 开发的友好聊天机器人，配备了 Stability AI 的文本和图像生成技术，具有 Stable Diffusion 3 和 Stable LM 2 12B。</li><li><a href="https://stable-diffusion-art.com/samplers/">Stable Diffusion Samplers: A Comprehensive Guide - Stable Diffusion Art</a>: AUTOMATIC1111 中有许多采样方法可用。Euler a, Heun, DDIM... 什么是采样器？它们如何工作？它们之间有什么区别？哪一个...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1fkt3/5_new_steerable_motion_workflows_for_travelling/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/banodoco/steerable-motion">GitHub - banodoco/Steerable-Motion: A ComfyUI node for driving videos using batches of images.</a>: 一个用于使用批量图像驱动视频的 ComfyUI 节点。
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1245021063390429214)** (1 条消息): 

- **OpenAI 董事会成立安全与安保委员会**：OpenAI 宣布成立 **Safety and Security Committee**，负责为所有 OpenAI 项目的关键安全和安保决策提供建议。更多详情请参阅其[官方公告](https://openai.com/index/openai-board-forms-safety-and-security-committee/)。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1244727268417212489)** (321 条消息🔥🔥): 

- **自组织文件系统引发用户关注**：用户讨论了一个名为 **LlamaFS** 的“自组织文件系统”，它能根据内容和时间整理文件。一位用户指出，“我希望生活中的一切都能自组织”，表达了对这种自动化的热情。

- **关于 AI 模型成本和 NPU 集成的讨论**：有一场关于由于集成 NPU（神经网络处理单元）导致**硬件成本**增加的深入对话。成员们推测了其经济影响，辩论 NPU 是否会使硬件成本增加 200 到 1000 美元，特别是对于高端模型。

- **辩论 AI 在游戏开发中的角色**：围绕 AI 辅助开发像 GTA 这样复杂游戏的潜力展开了激烈辩论。一位成员评论道，“*几年内，个人就可以利用 AI 独自制作一款 GTA 游戏，*”而另一位成员则认为这过于乐观，并指出了当前的局限性。

- **治愈癌症、GPT 与 TOS 考量**：讨论了将 **GPT** 用于治愈癌症等宏大项目的可能性，以及此类计划如何与 OpenAI 的 TOS（服务条款）互动。在讨论中，有人澄清说，如果不是为了“抓取他们的数据并制作竞争模型”，那么在 AI 项目中使用 GPT 可能是没问题的。

- **模型的记忆与上下文能力**：成员们对 GPT-4o 回忆视频和音频事件的能力印象深刻，讨论了其记忆存储方式为在特殊文件夹中编码 Token。SunSweeper 指出，“我看了 GPT-4o 的演示……能够回忆起那个事件，”对 AI 模型的记忆能力表示惊讶。

**提到的链接**：<a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能多模态大语言模型（MLLMs）。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细的...

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1244784831250960455)** (21 条消息🔥): 

- **使用 ChatGPT 构建商业网站**：一位成员质疑使用 ChatGPT 创建专业商业网站的实用性。另一位澄清道，“*它可以创建简单的线框图和基本功能，但仅靠 ChatGPT 建立一个功能齐全的网站？不行。*”

- **对自定义 GPTs 记忆功能的困惑**：关于自定义 GPTs 是否具有记忆功能存在困惑。一位用户解释说，“*目前，GPTs 没有长期记忆……但我们可以设计一个 GPT 来帮助管理信息并模拟长期记忆体验。*”

- **未发现重大更新**：围绕 iOS 应用更新的讨论导致成员询问是否有新功能；一位回答道，“*没发现新东西。*”

- **GPT 功能的多个问题**：用户报告了各种问题，如上下文丢失、记忆空白和编码错误。“*还有人的 GPT-4 出问题了吗？*”突显了对当前性能的普遍不满。

- **ChatGPT 可用性问题**：注意到 ChatGPT 间歇性无响应的问题，一位用户确认道，“*ChatGPT 没有响应。有人遇到同样的问题吗？*”另一位回复道，“*我这里运行完美，*”这引发了困惑。

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1244736004099674122)** (76 条消息🔥🔥): 

- **ChatGPT 拒绝抽取塔罗牌，但在处理较小请求时配合**：一位用户分享了请求 ChatGPT 抽取三张塔罗牌被拒绝的轶事，但在用户要求只抽一张后，它照办了。许多用户一致认为 ChatGPT 经常对可行任务进行随意的拒绝。
  
- **备选语言 Prompt 解决任务拒绝问题**：当被要求用英文 Prompt 处理一份法语文档时，ChatGPT 最初拒绝了，但当请求语言与文档语言匹配时，它执行了任务。用户指出，这可以作为解决类似任务拒绝问题的方案。

- **平衡 Prompt 长度与响应质量**：一位用户遇到了 GPT-4o 因 Prompt 过长而导致响应中断的问题。另一位用户分享了策略，例如用 Mermaid 图表替换冗长的规划，以在保持输出质量的同时减少 Token 使用量。

- **Meta-Prompting 与 Chain of Thought 的对比**：一场关于 Meta-Prompting 或 Chain of Thought (CoT) 方法是否能产生更好结果的辩论展开了，并提出了通过引入知识表示附件来优化 Meta-Prompting 的建议。建议用户创造性地结合这些方法，以避免狭隘的投射和幻觉（hallucinations）。

- **分享结果与资源**：用户分享了 Prompt 结果并讨论了进一步的优化，包括一位用户分享了他们针对计算机和 AI 素养改进后的 [Prompt](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477)。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1244736004099674122)** (76 条消息🔥🔥): 

- **ChatGPT 拒绝塔罗牌请求令用户沮丧**：一位成员分享了 ChatGPT 尽管多次请求仍拒绝抽取塔罗牌的轶事，突显了 AI 拒绝行为的不可预测性。另一位成员指出，ChatGPT 曾声称任务过大，但在使用更好的 Prompt 后却能正常工作。

- **在 GPT-4 中平衡 Prompt 细节的挑战**：用户讨论了 GPT-4 因 Prompt 长度而中断回答的问题，并寻找平衡点。一个建议是使用 **Mermaid 图表**进行规划以节省 Token，另一个建议是将其与 Zero-shot 方法融合。

- **Meta-prompting 与 Chain of Thought 的对比**：关于 Meta-prompting 中 Zero-shot 与 Chain of Thought (CoT) 有效性的辩论。Meta-prompting 被强调为针对具有高维解空间的开放式任务优化 AI，而 CoT 可能导致确定性的输出。

- **关于使用 YAML 和 XML 编写 AI Prompt 的见解**：对话深入探讨了 AI 如何原生处理 YAML、XML 和 JSON。提供了关于使用这些格式结构化 Prompt 以获得更好的 AI 理解和性能的示例和建议。

- **实际案例与共享资源**：成员们分享了经验和成功 Prompt 的链接，例如生成 Flutter 代码和全面的旅行规划模板，展示了所讨论技术的实际应用。链接包括 [计算机与 AI 素养 Prompt](https://chatgpt.com/c/d943acd5-e9c4-454e-8544-ad2faba45df8) 和 [扩展的素养覆盖范围](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477)。
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245036889539612764)** (1 条消息): 

- **Hugging Face 发布新模型**：新一批开源模型包括用于多模态对话的 **CogVLM2**、支持长上下文的 **Yi 1.5** 以及 **Falcon VLM** 等。查看[详细公告和链接](https://x.com/osanseviero/status/1793930015047880959)。

- **Sentence-transformers v3.0 发布**：新版本支持多 GPU 训练、bf16 支持等。点击[此处](https://huggingface.co/posts/tomaarsen/872659372583163)查看完整详情。

- **Diffusers 0.28.0 现已支持非生成式任务**：最新更新通过 Marigold 增加了深度估计和法线预测功能。详细的发布说明可以在[此处](https://github.com/huggingface/diffusers/releases/tag/v0.28.0)找到。

- **Gradio 推出新功能**：重大发布活动揭晓了 Gradio 应用的新库和新功能。请关注 [6 月 6 日](https://x.com/Gradio/status/1793758586147090902)。

- **LLM 与网络安全**：研究人员评估了哪些大语言模型在网络安全场景中最为安全。在此阅读[完整分析](https://huggingface.co/blog/leaderboard-llamaguard)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/osanseviero/status/1793930015047880959)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：📰本周新的开源模型：多语言、长上下文和 VLM 🔥 - CogVLM2：多模态对话 - Yi 1.5 长上下文 - M2-BERT-V2，长上下文编码器模型 - Phi 3 small 和 medium ...</li><li><a href="https://x.com/RisingSayak/status/1795083868900311360)">来自 Sayak Paul (@RisingSayak) 的推文</a>：以新的 🧨 Diffusers 发布开启这一周 ❤️‍🔥 本次发布包含了库中首批非生成式任务 —— 通过 Marigold 进行深度估计和法线预测 💐 注意...</li><li><a href="https://x.com/Gradio/status/1793758586147090902)">来自 Gradio (@Gradio) 的推文</a>：发布活动：我们将发布一些新东西。点击“提醒我”按钮并保持关注，6 月 6 日星期四 https://www.youtube.com/watch?v=44vi31hehw4&ab_channel=HuggingFace</li><li><a href="https://x.com/victormustar/status/1795405605605106044)">来自 Victor M (@victormustar) 的推文</a>：✨ HuggingChat 现已支持 Tools。简而言之，Tools 允许 HuggingChat 使用社区构建的任何 AI 应用 (ZeroGPU Spaces)，提供无限可能。</li><li><a href="https://x.com/_philschmid/status/1793910461286494539)">来自 Philipp Schmid (@_philschmid) 的推文</a>：激动人心的消息！📢 由 @googlecloud 提供支持的全新 @nvidia A100 和 H100 GPU 已上线 @huggingface 推理端点 (Inference Endpoints)，速度极快！🏎️ 💨💨 每个用户/组织默认配额为 2x A100 和 H100 GPU。A...</li><li><a href="https://x.com/osanseviero/status/1793018964479463781)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：我是 GPU 穷人。你呢？https://huggingface.co/settings/local-apps</li><li><a href="https://x.com/clefourrier/status/1793922499559747958)">来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：LLM 可以被用来帮助网络攻击者吗？🤖 它们在破解沙箱方面表现如何？... 简而言之，哪种 LLM 对网络安全最安全？来自 @Meta 的研究人员开发了一个基准测试来分析...</li><li><a href="https://x.com/dylan_ebert_/status/1793643044346159553)">来自 dylan (@dylan_ebert_) 的推文</a>：3D 机器学习课程第 3 单元已发布！涵盖：🎨 什么是 Gaussian Splatting？⚙️ 它如何融入生成式 3D 流水线 ✏️ 构建你自己的演示的动手代码。前往查看...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1244731126174711879)** (333 条消息🔥🔥): 

- **多模态 Space 创建详解**：一位用户询问了早期多模态 Space 的创建方式，是单一模型还是带有 Router 的堆叠模型。另一位用户分享了一个 [源码链接](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py) 以查看此类实现的具体细节。
  
- **因 VPN/广告拦截器导致的账户创建问题**：一位用户在创建账户时遇到问题，收到“未找到账户”的错误。建议的解决方案包括禁用 VPN、代理和广告拦截器设置，因为 HuggingFace 的安全策略非常严格。

- **TinyLlama 训练见解与问题**：几位用户讨论了有效训练 TinyLlama 所需的数据集大小和步骤。其中一个例子提到，在包含 1 万条条目的数据集上，可以在大约 2 小时内高效完成 TinyLlama 的微调 (finetuning)。

- **Spectogram-to-Wav 模型发布**：一位用户宣布即将发布一个新的频谱图转波形 (Spectogram-to-Wav) 模型，并提到了由于算力限制所面临的挑战。另一位用户分享了他们在微调模型方面的经验和建议，以及如何避免常见的陷阱。

- **Kaggle 上的视频分类模型**：一位用户报告称，其视频分类模型在训练期间 GPU 运行正常，但在验证期间仅使用 CPU。为了提供更多上下文，该用户分享了其 [Kaggle notebook](https://www.kaggle.com/code/an1001/tiktokvideoclassification?scriptVersionId=180260409) 的链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o">OpenGPT 4o - KingNish 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://arxiv.org/abs/2305.05176">FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance</a>：目前可供用户付费查询的大语言模型 (LLM) 数量增长迅速。我们调研了查询热门 LLM API（如 GPT-4, ChatGPT, J1-Jumbo）的相关成本，并发现……</li><li><a href="https://huggingface.co/spaces/kimou605/shadow-clown-BioMistral-7B-DARE">GenSeq - kimou605 的 Hugging Face Space</a>：暂无描述</li><li><a href="https://huggingface.co/openai/clip-vit-base-patch32">openai/clip-vit-base-patch32 · Hugging Face</a>：暂无描述</li><li><a href="https://tenor.com/view/cat-dont-care-didnt-ask-didnt-ask-i-didnt-ask-gif-25429803">Cat Dont Care Didnt Ask GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py">app.py · KingNish/OpenGPT-4o at main</a>：暂无描述</li><li><a href="https://www.kaggle.com/code/an1001/tiktokvideoclassification?scriptVersionId=180260409">TikTokVideoClassification</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自 TikTok 视频的数据</li><li><a href="https://wandb.ai/mikusdevr/huggingface/runs/jfs4xvfr/workspace">mikusdevr</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://huggingface.co/settings/local-apps">Hugging Face – 构建未来的 AI 社区。</a>：暂无描述</li><li><a href="https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378">apple/DFN5B-CLIP-ViT-H-14-378 · Hugging Face</a>：暂无描述</li><li><a href="https://huggingface.co/datasets/ZeroWw/MEISD">ZeroWw/MEISD · Hugging Face 数据集</a>：暂无描述</li>
</ul>

</div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1244802725657444392)** (1 条消息): 

- **SDXL 嵌入空间对齐解析**：**SDXL 嵌入空间（embed space）**将无条件空间（unconditioned space）对齐为零，而不是像早期的 SD 1.5/2.x 模型或 DeepFloyd 那样对齐到编码空间。*“将模型重新对齐到新的无条件空间（uncond space）非常痛苦，且耗时巨大。”*
- **ControlNet 训练的不确定性**：成员学习了 **ControlNet 训练**，但不确定自己的实现是否正确。在处理复杂模型时，这种不确定性很常见。
- **优化时间步范围切片（Timestep Range Slicing）**：将时间步范围分段以匹配 batch size，可以实现**在小算力训练中更均匀的时间步采样**。如果不这样做，可能会导致时间步训练分布中出现巨大缺口，从而可能损害训练稳定性。
- **长宽比分桶（Aspect Bucketing）的优缺点**：使用**随机长宽比分桶**有助于偏移内容与长宽比的偏差，DALLE-3 可能也使用了这种方法，它同样支持三种分辨率。然而，在不引入失真的情况下最大化训练样本具有挑战性。
- **训练工作流中的陷阱**：无意中让 **Torch 异常检测器（anomaly detector）开启数月**会**浪费时间**，而试图“百分之百彻底修复”某些问题往往会引入新的麻烦。 


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1244916184420323359)** (1 条消息): 

- **探索开源 AI 仓库**：一位成员分享了一篇关于开源 AI 生态系统及其演进的有趣[文章](https://huyenchip.com/2024/03/14/ai-oss.html?utm_source=tldrai)，并附带了在 [Hacker News](https://news.ycombinator.com/item?id=39709912)、[LinkedIn](https://www.linkedin.com/posts/chiphuyen_generativeai-aiapplications-llmops-activity-7174153467844820993-ztSE) 和 [Twitter 线程](https://twitter.com/chipro/status/1768388213008445837)上的讨论。该文章提供了一个每六小时更新一次的开源 AI 仓库完整列表，这些内容也可以在 GitHub 上的 [cool-llm-repos](https://github.com/stars/chiphuyen/lists/cool-llm-repos) 列表中找到。

- **重温 MLOps 分析**：该成员在四年前对[开源 ML 生态系统](https://huyenchip.com/2020/06/22/mlops.html)进行过分析，现在重新审视这一话题，专门关注围绕基础模型（foundation models）的技术栈。完整详情包括仓库数据以及 AI 技术栈随时间的演进过程。

**提到的链接**：<a href="https://huyenchip.com/2024/03/14/ai-oss.html?utm_source=tldrai">What I learned from looking at 900 most popular open source AI tools</a>：[Hacker News 讨论, LinkedIn 讨论, Twitter 线程]

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1244731504731619509)** (6 messages): 

- **语音控制机械臂项目发布**：一位用户分享了名为 ["Open Source Voice-Controlled Robotic Arm"](https://www.youtube.com/watch?v=qv3bFhHoA5s) 的 YouTube 视频，展示了一个由语音命令控制的 AI 驱动机械臂。该项目旨在通过开源贡献使机器人技术民主化。

- **TinyML 鸟类分类模型实战**：一位个人讨论了他们基于 EfficientNetB0 的 TinyML 鸟类分类模型，并在随机的 Reddit 观鸟帖子和干净的测试集上进行了测试。他们分享了一篇详细的 [文章](https://www.cranberrygrape.com/machine%20learning/tinyml/bird-detection-tinyml/)，涵盖了模型的生成过程，并邀请合作伙伴进行进一步研究。

- **SD.Next 发布重大更新**：SD.Next 项目宣布了一个重要版本，其特点是采用了新的 [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)，集成了诸如 **HiDiffusion** 和增强型采样器等各种内置功能，并支持了新模型如 [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/)。完整的发布详情和功能可在项目的 [Changelog](https://github.com/vladmandic/automatic/blob/dev/CHANGELOG.md) 中查看。

- **介绍 HuggingPro 助手**：一位用户介绍了 [HuggingPro](https://hf.co/chat/assistant/66562fe0abb44809b7f77897)，这是一个旨在导航 Hugging Face 生态系统的 AI 助手。该助手提供关于模型、数据集等的准确信息，旨在让体验既高效又愉快。

- **Everything-AI v2.0.1 增加新功能**：用户推广了最新版本的 [everything-ai](https://github.com/AstraBert/everything-ai)，这是一个 AI 驱动的本地助手，具有音频文件处理、文本转视频生成和蛋白质结构预测等新功能。他们为有兴趣在本地设置该工具的用户提供了 [快速入门指南](https://astrabert.github.io/everything-ai/)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.cranberrygrape.com/machine%20learning/tinyml/bird-detection-tinyml/">Bird Detection TinyML</a>: 执着于缩小基于迁移学习的模型</li><li><a href="https://www.youtube.com/watch?v=qv3bFhHoA5s">Open Source Voice-Controlled Robotic Arm | Redefining Robots!</a>: 欢迎来到语音控制 AI 机械臂项目，在这里人工智能与机器人技术相遇。这是一个开源倡议，赋予用户指挥机器人的能力...</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Your fully proficient, AI-powered and local chatbot assistant🤖</a>: 你全能的、AI 驱动的本地聊天机器人助手🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: 你全能的、AI 驱动的本地聊天机器人助手🤖</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897">HuggingPro - HuggingChat</a>: 在 HuggingChat 中使用 HuggingPro 助手</li><li><a href="https://hf.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: 让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://github.com/vladmandic/automatic/wiki/Themes)">Create new page · vladmandic/automatic Wiki</a>: SD.Next: Stable Diffusion 和其他基于 Diffusion 的生成式图像模型的高级实现 - Create new page · vladmandic/automatic Wiki
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

pr0x7: 好的，我会尝试准备。会相应地更新进度。谢谢。
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1244750701729677353)** (4 条消息): 

- **收集 Hugging Face CV Hangout 的主题**：一位成员创建了一个 [Google Sheet](https://docs.google.com/spreadsheets/d/12PewkdH2oAJ1Azw3sTxi7FatJ9fE4bTUj_APQQ__Ufs/edit?usp=sharing) 来收集周六聚会的讨论点。主题涵盖从新模型到个人项目的各种内容，鼓励参与者积极贡献。

- **在 Stanford Cars 数据集上遇到困难**：一位用户分享了他们尝试使用 ViT-B_16 模型对 Stanford Cars 数据集进行汽车品牌和型号分类的尝试，但准确率仅为 60%。他们详细介绍了其数据增强技术和学习率调度器（learning rate scheduler），但面临过拟合问题，并寻求关于细粒度图像分类（fine-grained image classification）的建议。

- **寻求 Deep Learning 指导**：该用户承认自己是 Deep Learning 的新手，并请求更有经验的从业者提供指导，以提高其模型的性能。

- **新成员介绍**：一位用户介绍了自己是社区的新成员。他们没有提供更多背景信息或具体问题。

**提到的链接**：<a href="https://docs.google.com/spreadsheets/d/12PewkdH2oAJ1Azw3sTxi7FatJ9fE4bTUj_APQQ__Ufs/edit?usp=sharing">Hugging Face Computer Vision Hangout</a>：Tabellenblatt1 主题（Fine-Tuning/酷项目/等）、形式（短演讲/讨论/等）、提议者（Discord 用户名）

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1244947164556165192)** (2 条消息): 

- **寻找 Nomos8k_sfw 数据集遇到障碍**：一位成员表示难以找到 [4x-Nomos8kDAT 模型](https://openmodeldb.info/models/4x-Nomos8kDAT) 中提到的 Nomos8k_sfw 数据集，质疑该数据集是内部专用还是隐藏得太深。 
- **Typeface Arc 旨在实现高效的 AI 内容创作**：[Typeface Arc 平台](https://www.typeface.ai/) 提供了在统一体验中创建和管理品牌故事的工具。它具有一个 "Copilot" 功能，通过持续的反馈优化，轻松生成 10 倍以上的内容。

**提到的链接**：<a href="https://www.typeface.ai/">Typeface | 工作中的个性化 AI 叙事</a>：Typeface 是一款用于企业内容生成的生成式 AI 应用程序，赋能所有企业以极快的速度创建卓越且符合品牌形象的内容。

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1245027012280713369)** (1 条消息): 

- **Gradio Clients 1.0 直播公告**：Gradio 团队宣布将于 6 月 6 日举行直播活动，发布全新改进的 Gradio Python 和 JavaScript 客户端。感兴趣的人员可以通过 [Discord](https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154) 加入或在 [YouTube](https://www.youtube.com/watch?v=44vi31hehw4) 上观看，了解如何将 Gradio 集成到各种应用程序中。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1245020251611992154">加入 Hugging Face Discord 服务器！</a>：我们正致力于让优秀的机器学习民主化 🤗 验证以链接您的 Hub 和 Discord 账号！| 80043 名成员</li><li><a href="https://www.youtube.com/watch?v=44vi31hehw4">Gradio 发布：如何使用 Gradio Clients 构建机器学习 API</a>：每月有 100 万开发者使用 Gradio 通过 Gradio Python 库创建机器学习演示和 Web 应用程序。加入 Gradio 团队...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1244738786315997382)** (61 条消息🔥🔥): 

- **在 LM Studio 中通过 OpenAI API 使用 Vision 功能的说明**：要在 LM Studio 中集成视觉能力，请使用像 **LLaVA** 这样的模型，将其部署在服务器上，并利用 Vision Python 模板。*"只需获取一个像 LLaVA 这样具有视觉能力的模型。将其加载到服务器上。然后复制粘贴 Vision Python 模板即可。"*
  
- **MLX/EXL2 在 Apple M1 Max 上的加载速度更快**：**MLX 和 EXL2 模型在 Apple M1 Max 上的加载速度明显快于 GGUF**，L3 8bit 约需 5 秒，而 GGUF Q8 则需 29 秒。*"MLX/EXL2 比 GGUF 快得多，主要是因为推理引擎不同。"*

- **在本地 LLM 中使用 RAG**：LM Studio 不支持直接与 PDF 或电子书交互；但是，通过 LM Studio 运行服务器并使用 [AnythingLLM](https://community.amd.com/t5/ai/how-to-enable-rag-retrieval-augmented-generation-on-an-amd-ryzen/ba-p/670670) 可以实现检索增强生成 (RAG)。*"通过 LM Studio 的 Local Server 选项卡启动服务器，然后运行 AnythingLLM。"*

- **LM Studio 不支持微调 (Fine-tuning)**：**LM Studio 不支持微调**，但可以使用其他工具进行微调，例如 [在搭载 Apple Silicon 的 Mac 上使用 MLX](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/)。*"训练模型比运行模型更耗费资源，而且推理引擎（llama.cpp 不支持微调）。"*

- **LM Studio 不支持函数调用 (Function calling)**：**LM Studio 和类似的基于 llama.cpp 的 API 不支持函数调用**。*"API 不支持函数调用。"*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/3bXg4Qv3">加入 Mintplex Labs | AnythingLLM | VectorAdmin Discord 服务器！</a>：查看 Discord 上的 Mintplex Labs | AnythingLLM | VectorAdmin 社区 —— 与其他 4215 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://community.amd.com/t5/ai/how-to-enable-rag-retrieval-augmented-generation-on-an-amd-ryzen/ba-p/670670">如何在 AMD Ryzen™ AI PC 或 Radeon™ 显卡上启用 RAG (检索增强生成)</a>：基于 GPT 的大语言模型 (LLM) 可以成为有用的 AI 助手，最大限度地提高生产力并增加工作流效率。在搭载 AMD Ryzen™ AI 的 AI 电脑上运行 AI 聊天机器人...</li><li><a href="https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/">在搭载 MLX 的 Mac 上进行本地 LLM 微调的简单指南 —— Andy Peatling</a>：无描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>：无描述</li><li><a href="https://huggingface.co/datasets/Sao10K/Claude-3-Opus-Instruct-15K">Sao10K/Claude-3-Opus-Instruct-15K · Hugging Face 数据集</a>：无描述</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing">Google Colab</a>：无描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1244810505021820998)** (45 条消息🔥): 

- **拒绝“我能问个问题吗？”陷阱**：成员们讨论了对那些不直接提问而是问“我能问个问题吗？”的人的挫败感，这浪费了双方的时间。正如一位成员所说，*"如果你直接提问而不是询问是否可以提问，你可以节省自己和每个人的时间。"*
- **明智地使用 AI 和专家**：虽然 **Google 和 AI 可能不可靠**，但一位成员指出，*"我会先询问 AI，如果不起作用，再向专家确认。"* 这突显了在利用技术和人类专业知识解决问题时的平衡方法。
- **Phi-3-Vision 支持限制**：一位成员询问 [Phi-3-Vision-128K-Instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) 是否可以在 LM Studio 中运行，但另一位成员澄清道，*"llama.cpp 仍不支持，因此无法在 LM Studio 中运行。"*
- **探索模型的“异常 (Glitchy)”行为**：有人询问模型在特定提示词下表现出的“异常”行为。给出的一个例子是 *dolphin 2.9 llama 3*，当使用特定预设加载时，它会显示出不稳定的行为。

**提到的链接**：<a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>：无描述

  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1244886480833482783)** (12 条消息🔥): 

- **错误聊天模板导致 AI 错误**：一位成员遇到了 AI 响应问题，经澄清是因为他们没有为 Llama 3 配置正确的聊天模板。建议从右上角菜单中选择正确的预设。

- **关于“鲜血与凯撒”的 AI 消息引发困惑**：AI 生成了一条关于用鲜血写下 "Hail Caesar!" 的怪异消息，引发了猜测。一位用户建议这可能是某些未审查模型训练数据的泄露。

- **Llama 3 生成列表令用户烦恼**：一位成员抱怨 Llama 3 尽管有系统提示词要求不要这样做，但仍不断生成列表。他们分享了一个似乎无效的系统提示词示例，寻求更好的替代方案。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1244764138199187487)** (135 条消息🔥🔥): 

- **用户辩论 Nvidia 预算选择与昂贵的 AI 训练 GPU**：几位成员讨论了 **Nvidia Tesla P40/P100** 以及传闻中拥有 32GB VRAM 的 **5090** GPU 等选项，权衡了它们的成本和性能。另外，有人建议将 **Macs** 用于推理 (inference)，但认为 PC 更适合训练 (training)。

- **对 GPUDirect Storage 持谨慎乐观态度**：这项允许 **GPU 直接访问 SSD** 而无需 CPU 参与的技术受到了关注。然而，其复杂的安装过程以及是否值得投入的不确定性给这种热情泼了冷水。

- **对 Nvidia 市场泡沫的担忧**：一些成员考虑到 Nvidia 在 AI 芯片市场的垄断地位，讨论了其股价的**潜在泡沫**，并辩论了来自 AMD 或 Intel 的竞争对手进步的稳定性和未来影响。

- **关于快递和物流的不同体验**：成员们分享了**硬件交付**速度和可靠性方面的问题，特别是对比了在**俄罗斯**和**澳大利亚**的经历，强调了对快递服务相对于传统邮政选项的挫败感。

- **推测新型 AI PC 是否值得**：对于大肆宣传的 **“Copilot + PC”** 和其他 AI PC 存在怀疑态度，担心它们是否真的增加了显著价值，还是过度炒作的、更多依赖云服务而非本地能力的产品。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html">NVIDIA GPUDirect Storage Overview Guide - NVIDIA Docs</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html">NVIDIA GPUDirect Storage Installation and Troubleshooting Guide - NVIDIA Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1244848496042049536)** (13 条消息🔥): 

- **大模型在 Windows 上的 CPU 亲和性问题**：一位用户讨论了在运行大模型时，Windows 如何开始在所有核心上进行推理，但几分钟后最终将工作转移到一个 CCD。他们计划尝试将每个 CCD 配置为独立的 NUMA 节点，以提高系统内存带宽利用率。
- **在支持 AVX2 的笔记本电脑上加载模型出错**：一位成员分享了在支持 AVX2 的新笔记本电脑上加载模型时遇到的错误，导致 Linux 上的 GPU offload 出现问题。禁用 GPU 加速设置的建议未能解决问题，因为该设置显示为灰色。
- **处理器伪装问题**：另一位用户推测错误可能是由于处理器被当作 GPU 使用，并建议禁用处理器的 “GPU” 功能以解决问题。这可能有助于系统连接到实际的 GPU 资源。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1244728768564822056)** (169 条消息🔥🔥): 

- **Unsloth 不支持 GPT-2**：成员询问 Unsloth 是否支持 GPT-2 的微调 (fine-tuning)。已确认不支持，原因是架构差异。

- **使用 Unsloth 进行微调的数据集准备**：一位用户寻求关于使用 Unsloth 和 Llama 3 微调超过 5 万条电子邮件条目数据集的建议，重点是为输入和输出创建合适的结构。几位用户提供了帮助和建议，包括重构提示词模板以适应数据集。

- **即将支持视觉模型**：讨论强调 Unsloth 目前不支持视觉模型，但预计下个月将提供支持。这引发了关于现有视觉模型的讨论，例如 Stable Diffusion 和 Segment Anything，并分享了相关链接以获取更多信息（[Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Segment Anything](https://github.com/facebookresearch/segment-anything)）。

- **使用和合并 LoRA 适配器**：成员们讨论了如何将 LoRA 适配器与原始模型合并并进行微调，以及用于保存这些模型并将其上传到 HuggingFace 等平台的工具。分享了 [GitHub 链接](https://github.com/unslothai/unsloth#-finetune-for-free) 以获取相关资源。

- **Phi 3 Medium 滑动窗口问题**：有人指出 Phi3-Medium 使用了滑动窗口注意力（sliding window attention）机制，这在 Token 数量较多时会导致性能问题。许多用户表达了沮丧，并期待该模型能支持更高的上下文窗口，特别是提到了 128K 上下文。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html">Axolotl - Conversation</a>：未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1795518622913433891">来自 Andrej Karpathy (@karpathy) 的推文</a>：但那些也是规模大得多的运行，所以更令人印象深刻。这是在单个节点上完成的，所以你不需要处理任何跨节点互连。它开始变得更有趣了...</li><li><a href="https://x.com/danielhanchen/status/1795453604532207989">来自 Daniel Han (@danielhanchen) 的推文</a>：我们是如何将 Phi-3 “Mistral 化”的？1) 解绑 QKV 和 gate/up 权重 2) 注意 Phi-3 使用了滑动窗口注意力 3) 并且 Phi-3 有一个 bug - 2047 SWA 应该是 2048，并将 @UnslothAI 的版本发送到了...</li><li><a href="https://github.com/matatonic/openedai-vision">GitHub - matatonic/openedai-vision: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal.</a>：一个兼容 OpenAI API 的 API，用于图像输入对话和关于图像的问题。又名 Multimodal。- matatonic/openedai-vision</li><li><a href="https://github.com/facebookresearch/segment-anything">GitHub - facebookresearch/segment-anything: The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>：该仓库提供了使用 SegmentAnything Model (SAM) 进行推理的代码、下载训练好的模型检查点的链接，以及展示如何使用该模型的示例 Notebook。-...</li><li><a href="https://huggingface.co/datasets/openchat/ultrachat-sharegpt?row=0">openchat/ultrachat-sharegpt · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/CompVis/stable-diffusion">GitHub - CompVis/stable-diffusion: A latent text-to-image diffusion model</a>：一个潜在的文本到图像扩散模型。可以通过创建账户为 CompVis/stable-diffusion 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#training-adapters">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/lora">LoRA</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1ef-ta">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1u_ozy3HqmiwwzG5kqqVklYDc05hVVJH_">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v3.0.0">Release v3.0.0 - Sentence Transformer Training Refactor; new similarity methods; hyperparameter optimization; 50+ datasets release · UKPLab/sentence-transformers</a>：此版本包含一次重大重构，彻底改进了训练方法（引入多 GPU 训练、bf16、损失日志记录、回调等），增加了便捷的相似度和类似...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1244763540343095297)** (48 messages🔥): 

```html
- **通过修正参数顺序修复 GDrive 保存错误**：一位成员在将模型保存到 GDrive 时，因 `save_pretrained_merged` 中的参数顺序不正确而遇到错误。另一位成员建议修正参数顺序，从而解决了该问题（*“好吧，我真笨，谢谢！”*）。
- **训练过程中的 Batch Size 和 Steps**：成员们讨论了如何为具有 500 个样本、Batch Size 为 8 且步数为 62 的模型设置 Epochs 和 Steps。建议使用 `num_train_epochs = 3` 并移除 `max_steps = 500`，以潜在地避免重复输出和过拟合。
- **模型训练中的句子重复问题**：一位成员遇到了模型在训练后重复相同句子的问题，这可能是由于缺少 EOS tokens 导致的。这表明需要确保添加 EOS token 以防止过拟合或训练不足。
- **将模型导出为 ONNX**：一位成员寻求将微调后的模型转换为 ONNX 格式的帮助。他们被引导至 Hugging Face 的 [ONNX 导出指南](https://huggingface.co/docs/transformers/en/serialization)，并明确了 VLLM 格式适用于该转换。
- **对 8-bit 和 OpenAI 兼容服务器的支持**：讨论涵盖了未来对 8-bit 模型和 OpenAI 兼容服务器的支持。有迹象表明 8-bit 支持即将推出，并且有在类似 LM Studio、Jan AI 或 Ollama 的环境中运行 Unsloth 模型的途径。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v">YouTube</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/serialization">导出至 ONNX</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/kigner/ruozhiba-llama3-tt?row=3">kigner/ruozhiba-llama3-tt · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing&authuser=1#scrollTo=yqxqAZ7KJ4oL)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing.">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mH">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245097510981210114)** (2 messages): 

- **Lighting.ai 在 GPGPU 方面获得好评**：一位成员询问关于使用 **Lighting.ai** 进行 GPGPU 编程的问题，解释说他们缺乏 NVIDIA 卡等商品硬件，需要使用 CUDA 和 SYCL 进行编程。回复确认 **Lighting.ai** 在这种用例下表现出色。
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1244988292890492950)** (2 messages): 

- **深入探讨 GPU 硬件和编程模型**：一位成员分享了两篇解释 **GPU 硬件和编程模型** 的文章。他们提到理解 GPU 能力对于优化大语言模型 (LLMs) 性能和降低延迟的重要性（[第一部分](https://cmeraki.github.io/gpu-part1.html)，[第二部分](https://cmeraki.github.io/gpu-part2.html)）。
- **Triton 中的 ViT 模型**：该成员还提到**在 Triton 中从头开始完整实现 ViT 模型**。他们声称其性能与 Hugging Face 的实现相比具有竞争力，并为对学习 Triton 感兴趣的人提供了 [GitHub 链接](https://github.com/cmeraki/vit.triton)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cmeraki.github.io/gpu-part1.html">GPUs Part 1 - 理解 GPU 内部结构</a>: LLM Labs</li><li><a href="https://cmeraki.github.io/gpu-part2.html">GPUs Part 2 - 理解 GPU 编程模型</a>: LLM Labs
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245102494304964678)** (12 messages🔥): 

- **Torch.compile 与 Python 3.12 不兼容**：一位用户发现 **torch.compile** 无法在 Python 3.12 下工作，原因是由于在 Ubuntu 24.04 上安装 PyTorch 2.3 后缺少 **triton**。他们找到了一个相关的 [GitHub issue](https://github.com/pytorch/pytorch/issues/120233) 正在跟踪此问题，并指出 **flash-attention** 在 Python 3.12 上可以运行。
- **建议使用 Pyenv 管理 Python 版本**：另一位在 **Arch Linux** 上使用 Python 3.12 的用户也遇到了同样的问题，并提到 PyTorch nightlies 已提供支持。他们建议使用 [pyenv](https://github.com/pyenv/pyenv) 来管理多个 Python 版本。
- **新字节码导致的问题**：会议澄清了 **dynamo** 需要解释每个 Python 版本中引入的新字节码，从而导致了该问题。讨论了让 PyTorch 发布版本与 Python 版本更紧密对齐的计划，并指出 nightlies 版本中已提供部分支持。
- **期待 Windows 支持**：一位在以 .NET 为中心的环境中工作的用户表达了对 Python 3.12 下 **torch.compile** 原生 Windows 支持的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054">Torch.compile 对 Python 3.12 的支持已完成</a>：扩散消息，确认 Python 3.12 的支持已添加到 torch.compile 中，并且已经在 nightly 构建版本中存在一段时间。我们预计此功能将包含在 PyTorch 2.4 版本中...</li><li><a href="https://github.com/pyenv/pyenv">GitHub - pyenv/pyenv: 简单的 Python 版本管理</a>：简单的 Python 版本管理。通过在 GitHub 上创建账户来为 pyenv/pyenv 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/120233">Torch compile 无法在 python 3.12 上运行 · Issue #120233 · pytorch/pytorch</a>：🐛 错误描述：目前截至 2.2.0 版本的 torch 不支持在 python 3.12 下使用 torch compile。请参阅以下 PR 示例：#117853。我们需要能够在 python 3.12 中使用 torch.compile 功能...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1245073610964598975)** (1 messages): 

- **GPT-4o 在大型代码编辑方面表现挣扎**：像 GPT-4o 这样的前沿模型在处理大型编辑时非常吃力，面临 *懒惰、不准确和高延迟* 等问题。“准确编辑数百行代码可能需要多次模型调用，有时甚至会将 Agent 困在死循环中。”

- **Fast Apply 模型旨在解决弱点**：一个名为 **fast apply** 的专门模型经过训练以解决这些弱点，将任务分解为 **规划（planning）** 和 **应用（applying）** 阶段。“在 Cursor 中，规划阶段表现为与强大的前沿模型进行的聊天界面。”

- **寻找确定性算法线索**：一位成员正在寻找实现代码编辑确定性算法的线索，并提到使用 **vllm** 或 **trtllm** 的潜在可行性。他们认为可以使用此类算法来推测未来的 token，而不是依赖于草稿模型（draft model）。

欲了解更多详情，可以阅读 [完整博客文章](https://cursor.sh/blog/instant-apply)。

**提到的链接**：<a href="https://cursor.sh/blog/instant-apply">近乎瞬时的全文件编辑</a>：未找到描述内容。

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1244810007028039760)** (15 messages🔥): 

- **通过 Conda 安装 CUDA Toolkit 及系统要求**：一位成员建议另一位成员从 NVIDIA 开发者网站“安装 CUDA Toolkit”，并通过在终端输入 `nvidia-smi` 来检查是否已安装。他们还推荐使用 [官方 CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)，并提供了有关文档和论坛的其他资源。
  
- **在 Ubuntu 上安装的命令**：为了在 Ubuntu 上设置 CUDA，一位用户提供了来自 Jeremy 推文的命令：“conda install cuda -c nvidia/label/cuda-12.1.0” 和 “conda install 'pytorch>2.0.1' torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia/label/cuda-12.1.0”。另一位用户提到有必要确保已安装 NVIDIA GPU 驱动程序。
  
- **在 Ubuntu 上安装 CUDA 的博客**：一位成员模糊地记得有一篇关于在 Ubuntu/Linux 上正确安装 Nvidia 驱动程序和 CUDA toolkit 的博客，尽管没有提供具体链接。
  
- **寻求 PMPP 学习指南**：另一位用户询问是否有人制作了 PMPP 的学习指南，包括优先章节和练习。该请求暗示了对结构化学习材料的需求。

**提及的链接**：<a href="https://developer.nvidia.com/cuda-downloads">CUDA Toolkit 12.1 Downloads</a>：获取 NVIDIA 专有计算栈的最新功能更新。

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1244753742255886477)** (3 messages): 

- **理解 torch.fx.Interpreter 和 GPTQRunner**：一位成员就 `torch.fx.Interpreter` 文档中 `call_function` 的行为与其在 `GPTQRunner` 中的使用提出了疑问。他们提供了 [GPTQRunner 类](https://github.com/pytorch/ao/blob/7511b1d365e2e314d1193d7b8df049ee9452e63c/torchao/quantization/GPTQ.py#L296) 的链接作为上下文。

- **MX 格式支持已合并**：另一位成员兴奋地宣布，对 MX 格式（包括 `fp8/6/4`）的支持已合并到 [PyTorch](https://github.com/pytorch/ao/pull/264) 中。他们邀请其他对提高速度感兴趣的人在 GitHub 上标记他们或 `vkuzo`，并提到结合 [MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) 阅读代码可以澄清许多细节。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/">pytorch</a>：pytorch 拥有 75 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/pytorch/ao/pull/264">由 vkuzo 添加 MX 格式训练和推理的原型 · Pull Request #264 · pytorch/ao</a>：摘要：MX 数值格式是新的低精度格式，最近已被纳入 OCP 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf 这...</li><li><a href="https://github.com/pytorch/ao/blob/7511b1d365e2e314d1193d7b8df049ee9452e63c/torchao/quantization/GPTQ.py#L296">ao/torchao/quantization/GPTQ.py at 7511b1d365e2e314d1193d7b8df049ee9452e63c · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1245081824208879717)** (27 messages🔥): 

- **选择合适的技术城市**：一位成员寻求关于具有浓厚技术文化和黑客松等社交活动城市的建议。建议包括 **SF、NYC 和 London** 等大城市，因为它们拥有充满活力的社交场景；以及像 **Seattle** 这样的小城市，但评价褒贬不一。
  
- **欧洲城市**：柏林 (Berlin) 和华沙 (Warsaw) 被认为比慕尼黑 (Munich) 更令人兴奋。柏林因其充满活力的文化而受到特别强调，包括 *“长达 3 天的 techno 派对和美味的烤肉”*。
  
- **圣迭戈 (San Diego) 和伊萨卡 (Ithaca)**：圣迭戈受到一位在那里生活多年的成员的称赞，而 **伊萨卡** 因康奈尔大学 (Cornell) 培养了许多成功人士而被提及，但被描述为无聊。
  
- **西雅图 (Seattle) 的社交场景**：一位成员分享了他们在西雅图生活的负面体验，称其为社交氛围最淡的城市，原因是漫长阴暗的冬天以及人们倾向于待在室内。
  
- **柏林的技术公司**：有人指出 Google 和其他小型初创公司在柏林运营，但主要的工程工作有限。建议在 **SF 或 NYC 积累大厂经验**，以便为将来创办公司等机会做准备。
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1244728712323272766)** (131 条消息🔥🔥): 

- **关于 `dirent.h` 和 `unistd.h` 集成的辩论**：成员们讨论了在哪里合并 `dirent.h`，有人建议将代码放入 `unistd.h` 并重命名，以避免与标准 Windows 的 `windows.h` 冲突。另一位成员更倾向于使用 `windows_posix.h` 这个名称以防止潜在问题。 
- **编译器警告与修复**：多个头文件中出现了关于潜在未初始化局部变量的警告，随后提交了一个 commit 来解决这些警告。一位成员建议在 `dirent.h` 周围确保使用 `#ifndef _WIN32`，以管理不同操作系统之间的兼容性。
- **“恢复训练”标志的实现**：引入了一个新的 `-y 1` 标志，用于在中断后自动恢复训练，从而提高训练过程的效率。该功能在耗时 14 小时、成本约 200 美元复现 350M 参数模型的过程中被证明非常有用。
- **反向传播内存优化讨论**：为了节省内存，成员们讨论了在反向传播期间重新计算 layernorm，而不是存储完整的 activations，这可能会带来效率提升。一位成员开始实现这种方法，旨在不牺牲性能的情况下减少内存占用。
- **在 S3 上托管大型数据集**：对话涉及在 S3 上托管 FineWeb100B，并考虑了成本和依赖管理。还探讨了 Zenodo 或 Ubicloud 等替代方案，强调了对高效且可扩展的数据托管方案的需求。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/issues/478">在反向传播中重新计算 activations 以节省内存 · Issue #478 · karpathy/llm.c</a>: @ngc92 对占用内存最多的区域及其对可用 batch 数量的影响进行了分析，发现最大的占用因素之一是与...相关的内存。</li><li><a href="https://github.com/karpathy/llm.c/pull/475">由 karpathy 实验添加 llmc lib 目录 · Pull Request #475 · karpathy/llm.c</a>: 未找到描述</li><li><a href="https://zenodo.org/">Zenodo</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">在 90 分钟内以 20 美元在 llm.c 中复现 GPT-2 (124M) · karpathy/llm.c · Discussion #481</a>: 让我们在 90 分钟内花 20 美元在 llm.c（约 4,000 行 C/CUDA 代码）中复现 GPT-2 (124M)。124M 模型是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，实际上相当...</li><li><a href="https://github.com/karpathy/llm.c/pull/459.">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://transmissionbt.com">Transmission</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/480">在反向传播中创建重新计算 layernorm activations 选项的第一步，由 ChrisDryden 提交 · Pull Request #480 · karpathy/llm.c</a>: 此 CR 是实现 #478 中所述目标的第一步，即通过添加在反向传播中重新计算 layernorm activations 的选项来减少内存占用。这...</li><li><a href="https://aws.amazon.com/s3/pricing/?p=pm&c=s3&z=4">Amazon S3 简单存储服务定价 - Amazon Web Services</a>: 未找到描述</li><li><a href="https://zenodo.org/records/3834942">OpenWebText</a>: OpenAI WebText 数据集的开源复制版。更多信息请访问 https://skylion007.github.io/OpenWebTextCorpus/ @misc{Gokaslan2019OpenWeb, title={OpenWebText Corpus}, author=...</li><li><a href="https://trac.transmissionbt.com/wiki/HeadlessUsage">
      HeadlessUsage     – Transmission

    </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[oneapi](https://discord.com/channels/1189498204333543425/1233802893786746880/)** (1 条消息): 

orion160: 有哪些调试 SYCL 代码的工具？通常是指单步进入 kernel 代码……
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1244746994228924428)** (9 messages🔥): 

- **激活量化神经网络中的梯度问题**：一位成员指出 *直接传递传入梯度是错误的*，并建议使用代理函数（如 `tanh`）的梯度。他们引用了一篇 [arXiv 论文](https://arxiv.org/abs/1903.05662)，解释了为什么即使是不正确的梯度，在使用直通估计器 (STE) 时也能最小化训练损失。

- **测试中 C 扩展的问题**：一位成员在 `torchao` 的 C 扩展无法正常导入时遇到了 `ImportError`。他们推测可能是因为使用了 **cuda12.4**，而 PyPi 上的默认版本是 **cuda12.1**。

- **切换 CUDA 版本**：另一位成员建议通过 conda 的 `cudatoolkit` 安装 **cuda12.1** 作为潜在的解决方案。他们还建议如果本地问题仍然存在，可以提交一个 issue。

**提到的链接**：<a href="https://arxiv.org/abs/1903.05662">Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets</a>：训练激活量化神经网络涉及最小化一个分段常数函数，其梯度几乎处处消失，这对于标准的反向传播或 cha... 是不利的。

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1244777737785184317)** (14 messages🔥): 

- **成员们表示热烈欢迎并分享个人经验**：几位新成员介绍了自己，表达了加入的兴奋之情，并分享了他们在 ML 和开发方面的背景。一位成员提到感觉自己“不足”，但渴望通过参与研究来“提高我的能动性 (agency)”。

- **对 Prompt 修改研究的好奇**：一位从事研究和基准测试的独立开发者询问了有关 **与 MMLU 相关的 Prompt 修改/结构化** 的现有研究。他们分享了自己的实验结果，指出在将输入调整为符合 Anthropic 的 XML Prompt 语法后，各个类别的表现出现了 “10-20% 的波动”。

- **建议的研究领域和社区项目**：一位社区成员被引导去查看 **社区项目** 以探索研究领域。该建议得到了对方的感谢。

- **请求协助重新实现 Facenet 模型**：一位成员请求协助使用 PyTorch 从头开始重新实现 Facenet 模型代码。消息中未提供回复或解决方案。

- **关于 Databricks 上 LLM 的问题被驳回**：一位成员询问了关于在 **Databricks** 上托管的 LLM 进行批量推理 (batch inferencing) 的问题。另一位成员建议此类问题最好咨询 Databricks 支持，并指出 Databricks “并不以兼容非 Databricks 的事物而闻名”。
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1244836234027859998)** (122 messages🔥🔥): 

- **GPTs Agent 在初始训练后无法学习**：一位成员担心 GPTs Agent 无法从初始训练后提供的额外信息中学习。另一位成员澄清说，上传的文件被保存为知识文件，供 Agent 在需要时参考，但它们不会持续修改 Agent 的基础知识。

- **用于高效扩散模型采样的 EM 蒸馏**：[Google DeepMind 的一篇新论文提出了 EM Distillation](http://arxiv.org/abs/2405.16852)，这是一种基于极大似然的方法，可以将扩散模型蒸馏为单步生成器模型，且感知质量损失极小。该技术引入了重参数化采样方案和噪声消除，以稳定蒸馏过程。

- **Google 训练了一个用于 1024x1024 图像的 8B 参数扩散模型**：Google 研究人员训练了一个非级联像素空间扩散模型，直接生成 1024x1024 图像，[详见他们的新论文](http://arxiv.org/abs/2405.16759)。讨论中包含了对与 Imagen 3 进行对比的期待。

- **STLMs 旨在最小化 LLM 中的参数**：[一项新的研究工作](https://arxiv.org/abs/2405.14159) 介绍了超微型语言模型 (STLMs)，旨在将参数量减少 90% 到 95%，同时保持性能。论文在具体的实现细节上比较模糊，但提到了未来关于无分词器 (tokenizer-free) 模型、自博弈 (self-play) 和替代训练目标的工作。

- **关于 GPU 延迟建模的问题**：一位成员询问如何在不运行 kernel 或使用学习模型的情况下，对 GPU 延迟进行符号化建模。一份有用的回复提供了相关 [研究论文和博士论文](https://inria.hal.science/hal-00789958/file/112_Lai.pdf) 的链接，讨论了理论方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://arxiv.org/abs/2405.16852">EM Distillation for One-step Diffusion Models</a>：虽然 Diffusion 模型可以学习复杂的分布，但采样需要计算量巨大的迭代过程。现有的蒸馏方法能够实现高效采样，但存在显著的局限性...</li><li><a href="https://arxiv.org/abs/2405.15815">A social path to human-like artificial intelligence</a>：传统上，认知科学家和计算机科学家以唯我论的方式看待智能，将其视为脱离社会背景的单一 Agent 的属性。鉴于当代学习算法的成功...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>：之前关于 Universal Transformers (UTs) 的工作已经证明了跨层参数共享的重要性。通过允许深度递归，UTs 在学习方面比标准 Transformers 具有优势...</li><li><a href="https://arxiv.org/abs/2405.16759">Greedy Growing Enables High-Resolution Pixel-Based Diffusion Models</a>：我们解决了如何大规模学习有效的基于像素的图像 Diffusion 模型这一长期存在的问题，引入了一种非常简单的贪婪增长方法，用于大规模、高分辨率模型的稳定训练...</li><li><a href="https://arxiv.org/abs/2405.14159">Super Tiny Language Models</a>：大语言模型 (LLMs) 的飞速发展带动了自然语言处理的显著进步，但也因其高计算和能源需求带来了挑战...</li><li><a href="https://arxiv.org/abs/2405.17399?s=09">Transformers Can Do Arithmetic with the Right Embeddings</a>：Transformers 在算术任务上的糟糕表现似乎在很大程度上源于它们无法跟踪长数字跨度中每个数字的准确位置。我们修复了...</li><li><a href="https://x.com/wenhuchen/status/1795094212230168715?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Wenhu Chen (@WenhuChen)</a>：有一种误解认为，任何特定 Benchmark 的泄露都会导致该 Benchmark 成绩的巨大提升。这不一定是真的。我们发现这实际上取决于...的格式。</li><li><a href="https://arxiv.org/abs/2203.14309">DeepDPM: Deep Clustering With an Unknown Number of Clusters</a>：深度学习 (DL) 在聚类这一无监督任务中展现出巨大潜力。尽管如此，虽然在经典（即非深度）聚类中非参数化方法的优势众所周知...</li><li><a href="http://arxiv.org/abs/2405.16759">Greedy Growing Enables High-Resolution Pixel-Based Diffusion Models</a>：我们解决了如何大规模学习有效的基于像素的图像 Diffusion 模型这一长期存在的问题，引入了一种非常简单的贪婪增长方法，用于大规模、高分辨率模型的稳定训练...</li><li><a href="https://arxiv.org/abs/2005.05744">Deep Learning: Our Miraculous Year 1990-1991</a>：在 2020-2021 年，我们庆祝了深度学习革命背后的许多基本思想是在 30 年前的不到 12 个月内发表的，即我们的“奇迹年”或“Mirac...</li><li><a href="https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/">Rethinking How to Train Diffusion Models | NVIDIA Technical Blog</a>：在探索了 Generative AI Research Spotlight: Demystifying Diffusion-Based Models 中解释的 Diffusion 模型采样、参数化和训练的基础知识之后...
</li>
</ul>

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1244967598723043358)** (2 条消息): 

- **新模型发布**：宣布推出 [Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3) 和 [Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b)。Mistral 7B Instruct 及其免费版本现在指向最新的 [v0.3 版本](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3)。

- **版本化模型访问**：旧版本如 [Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2) 和 [v0.1](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.1) 仍可访问。

- **OpenAI 故障快速解决**：之前出现了一次影响 OpenAI 使用的短暂故障。不过，他们迅速解决了该问题，Azure 及其备用方案在停机期间保持正常运行。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3>)">Mistral: 由 mistralai 开发的 Mistral 7B Instruct | OpenRouter</a>: 一款高性能、行业标准的 7.3B 参数模型，针对速度和上下文长度进行了优化。</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b>)">NousResearch: 由 nousresearch 开发的 Hermes 2 Pro - Llama-3 8B | OpenRouter</a>: Hermes 2 Pro 是 Nous Hermes 2 的升级重新训练版本，包含更新且清洗过的 OpenHermes 2.5 数据集，以及新引入的 Function Calling 和 JSON 模式...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct>)">Mistral: 由 mistralai 开发的 Mistral 7B Instruct | OpenRouter</a>: 一款高性能、行业标准的 7.3B 参数模型，针对速度和上下文长度进行了优化。</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2>)">Mistral: 由 mistralai 开发的 Mistral 7B Instruct v0.2 | OpenRouter</a>: 一款高性能、行业标准的 7.3B 参数模型，针对速度和上下文长度进行了优化。是 [Mistral 7B Instruct](/modelsmistralai/mistral-7b-instruct-v0.1) 的改进版本...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.1>)">Mistral: 由 mistralai 开发的 Mistral 7B Instruct v0.1 | OpenRouter</a>: 一款 7.3B 参数模型，在所有基准测试中均优于 Llama 2 13B，并针对速度和上下文长度进行了优化。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1244823757810434089)** (1 条消息): 

- **关于 Max Loh 网站上模型的查询**：一位成员询问 [Max Loh 的网站](https://www.maxloh.com) 上使用的是哪些模型。他们还询问是否有人知道如何查找 OpenRouter 上所有可用未审查（uncensored）模型的列表。
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1244733210198409308)** (122 条消息🔥🔥): 

- **关于 Phi-3 Vision 成本和可用性的辩论**：讨论围绕在 Azure 上使用 **Phi-3 Vision** 的高昂成本展开，一名成员建议 *“参考 llama 的价格，我能达到 $0.07/M”*。另一名成员反驳，指出其他供应商的收费也差不多。
  
- **Gemini 卓越的 OCR 能力**：成员们讨论了 Gemini 的 **OCR 能力**，声称它“能很好地读取西里尔文字”，并且在阅读西里尔文和英文文本方面“优于 Claude 和 GPT-4o”。

- **用于 Python 聊天机器人的 Langchain 和 Streamlit**：有人询问关于构建基于 Flask 的聊天机器人的合适模板。建议包括查看 **Streamlit 模板**和 **Langchain**，重点在于易于集成以及使用数据库适配器的可能性。

- **OpenRouter Token 成本说明**：参与者辩论了 **OpenRouter** Token 涉及的成本，澄清 $0.26 可以购买 1M input + output tokens，并讨论了 Token 计数如何影响定价。Fry69_61685 强调，每次聊天交互都会重新计算整个历史记录，从而增加了 Token 使用量。

- **处理 OpenAI 模型故障**：一次故障影响了 OpenAI **GPT-4o**，导致服务中断。Alex Atallah 向用户保证，确认问题已迅速修复，并承诺未来会进行更好的检查。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://useinstructor.com/">Welcome To Instructor - Instructor</a>: 无描述</li><li><a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - a Hugging Face Space by Xenova</a>: 无描述</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe | OpenRouter</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1244781840183721984)** (6 条消息): 

- **歌曲翻译挑战探讨**：一名成员询问了 **歌曲翻译** 的现状，特别是关于在对歌词有一定控制权的情况下保持基调。兴趣点在于如何在保留艺术意图的同时管理歌词翻译。
  
- **Greentext AGI 场景**：一名成员发现使用 LLM 创建 **4chan greentext 片段**非常有趣。他们让 LLM 生成一段关于醒来发现 AGI 已被创造出来的 greentext，并指出结果特别有趣。
  
- **对项目管理的担忧**：关于一位用户因担心代码库大小而犹豫是否为 **OpenCL extension** 采用另一个平台的讨论。该成员表示，除非代码被合入主分支（upstreamed），否则不感兴趣贡献，并批评了这种项目管理方法。
  
- **分享 CrewAI 视频**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Czhc0L2bqWo)，“CrewAI 创建 AI Agents 入门”。该视频提供了使用 CrewAI 创建 AI Agent 的教程，包括指向 [CrewAI 文档](https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/#python) 的链接。
  
- **大学城市的科技文化**：一名即将入学的研究生正在寻求拥有浓厚科技文化的城市大学推荐。他们对 SF、Munich 和 NYC 等地的读书会或黑客松感兴趣，旨在与从事类似 AI 项目的同行建立联系。

**提到的链接**: <a href="https://www.youtube.com/watch?v=Czhc0L2bqWo">CrewAI Introduction to creating AI Agents</a>: 我们将了解如何使用 CrewAI 创建 AI Agent。https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/#python #pythonprogramming #llm #m...

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1244739332540072158)** (63 条消息🔥🔥): 

- **Phi 模型训练辩论**：成员们讨论了 Phi 在训练中是使用了大部分教科书数据还是合成数据（synthetic data）。有人指出，“Phi 使用了大部分教科书”，而另一位纠正道，“论文声称它是经过严格过滤的公开数据和合成数据的混合体。”

- **LLM 中的逻辑和自我纠错**：用户测试了 LLM 提供逻辑解释和自我纠错的能力，并指出了失败之处。一位用户观察到，“它把我对其逻辑的质疑当作我说‘那是错的’一样处理，”而另一位评论道，“总是顺从用户的模型可能也会变得很笨。”

- **微调（fine-tuning）的 Epochs 和 Batch Sizes**：用户分享了关于模型微调时 Epoch 数量和 Batch Size 的看法。有人建议，“通常 1-3 个比较好，”另一位补充道，“4-6 个属于过拟合（over-fitting）范畴，但也可以奏效。”

- **RAG-in-a-box 解决方案**：一位用户询问了关于上传数千个 PDF 进行 RAG 搜索的建议。另一位解释说，构建一个合适的 RAG 解决方案取决于许多因素，包括数据类型和具体的查询。

- **Transformer 中的算术运算**：探讨了 Transformer 执行算术运算的潜在复杂性。一位成员将其描述为“部分结果的逐层转换”，强调了使用 Token 预测模型处理算术运算的根本局限性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medicalxpress.com/news/2024-05-ai-large-language-align-human.amp">
      Improving AI large language models helps them better align with human brain activity
          </a>：随着生成式人工智能（GenAI）近年来改变了社交互动格局，使用深度学习算法训练 GenAI 平台的大语言模型（LLMs）...</li><li><a href="https://arxiv.org/abs/2405.17399">Transformers Can Do Arithmetic with the Right Embeddings</a>：Transformer 在算术任务上的糟糕表现似乎在很大程度上源于它们无法跟踪大跨度数字中每个数字的准确位置。我们修复了...</li><li><a href="https://osf.io/94y7h/.">
        Predicting the next sentence (not word) in large language models: What model-brain alignment tells us about discourse comprehension
</a>：托管在 Open Science Framework 
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1244733900966592542)** (28 messages🔥): 

- **在单张 A100 GPU 中适配 70B 模型**：*"Jaredquek"* 讨论了使用 axolotl，在 seq len 为 1200 且进行文本补全的情况下，使用 98% 的 GPU 显存运行 8bit LoRA。他指出，如果使用 QLoRA，会有更多的剩余空间。
- **Hermes 中的实验性 RepEng 向量**：*"Max_paperclips"* 强调在 Hermes 中减去诚实向量（honesty vector）会导致模型迅速崩溃，而 Mistral 等其他模型的反应则不同。Azure2089 等人也提到了类似的经历，Azure2089 提供了一个[链接](https://github.com/cpldcpu/MisguidedAttention/blob/main/repeng_02_river_crossing.md)，其中的 prompt 旨在解决模型在存在误导信息时的推理问题。
- **创建 DPO 数据集的挑战**：Lokesh8882 和 Thilotee 就使用自定义组织数据启动 DPO 数据集交换了意见，Thilotee 建议参考科学论文。Dumball 指出 DPO 需要特定的格式，并链接了 [Hugging Face 的 TRL 文档](https://huggingface.co/docs/trl/main/en/reward_trainer)作为示例。
- **DPO 训练的自定义数据集格式**：Thilotee 提供了来自 Hugging Face TRL 关于从偏好数据训练语言模型的 DPO Trainer 资源，详见[这篇论文](https://arxiv.org/abs/2305.18290)。Dumball 确认 DPO 需要 prompt、chosen（被选中的）和 rejected（被拒绝的）响应格式的数据。
- **概念提案：带有更大 Attention 机制的小型 LLM**：Bliponnobodysradar 提出了一个想法，即训练像 Llama3 8B 这样的小型 LLM，但配备更大的 Attention 机制，以实现大型模型的上下文感知能力，并征求对此想法的反馈。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/trl/main/en/reward_trainer#trl.RewardTrainer">Reward Modeling</a>：未找到描述</li><li><a href="https://github.com/cpldcpu/MisguidedAttention/blob/main/repeng_02_river_crossing.md">MisguidedAttention/repeng_02_river_crossing.md at main · cpldcpu/MisguidedAttention</a>：一组旨在挑战大型语言模型在存在误导信息时推理能力的 prompt - cpldcpu/MisguidedAttention</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1244796325023846481)** (2 messages): 

- **为 RAG 项目汇总资源**：一位成员分享了他们的情感与语义密度平滑 Agent 项目，可在 [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT) 上获取，并提到他们休假归来，热衷于汇总资源。他们指出 TTS 组件可能需要一些准备工作才能顺利运行，可能需要模型缓存。
- **电子游戏休息与 SLURM 移植**：另一位成员提到他们休了一周假玩电子游戏，现在的下一个任务是专注于将他们的项目 Cynde 移植到 SLURM。

**提到的链接**：<a href="https://github.com/EveryOneIsGross/densefeelsCHAT">GitHub - EveryOneIsGross/densefeelsCHAT: sentiment and semantic density smoothing agent. w/ tts</a>：情感与语义密度平滑 Agent，带有 TTS - EveryOneIsGross/densefeelsCHAT

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

jakekies: hi
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1244787381471674459)** (76 条消息🔥🔥): 

<ul>
<li>
<b>Agent 陷入死循环问题</b>：一位用户报告了他们的 LangChain Agent 进入工具调用的死循环问题，寻求让 Agent 提供最终响应的解决方案。
</li>
<li>
<b>将 Requests 作为工具调用</b>：关于如何在 LangChain 中将 requests 作为工具调用的广泛讨论和代码示例。解决方案包括使用 <code>JsonRequestsWrapper</code> 和创建 <code>ProgramSearchTool</code> 以根据用户输入动态向参数添加值，并提供了示例。
</li>
<li>
<b>更新后的 Token 限制错误</b>：一位用户提到在将 LangChain 更新到 0.2.2 版本时出现错误，本应支持高达 128k tokens 的模型被错误地应用了 16385 tokens 的上下文长度限制。他们寻求社区支持以解决这一差异。
</li>
<li>
<b>SQL Agent 提示词模板</b>：一位成员请求并获得了一个 SQL Agent 的提示词模板，其中包含 few-shot 示例，以引导 Agent 生成正确的 SQL 查询。提供了关于如何构建提示词并在 LangChain 中使用它们的说明。
</li>
<li>
<b>关于 LangChain v2.0 的疑问</b>：一位用户询问了 LangChain 2.0 版本中 Agent 的存在情况，表示在更新版本中难以找到相关功能。
</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/1580>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13826>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/14508>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2140>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3838>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/tools/requests/#inside-the-tool>)">Requests | 🦜️🔗 LangChain</a>: 网络包含大量 LLM 无法访问的信息。为了让 LLM 轻松地与这些信息交互，我们提供了一个围绕 Python Requests 模块的封装器，它可以接收...</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/sql/prompting/#few-shot-examples>)">Prompting strategies | 🦜️🔗 LangChain</a>: 在本指南中，我们将介绍改进 SQL 查询生成的提示策略。我们将主要关注在提示词中获取相关数据库特定信息的方法。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16731>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1244895685728403456)** (4 条消息): 

- **Langserve 中的自定义 kwargs 在 Langsmith 中丢失**：一位成员正尝试在 Langserve 的请求中发送自定义 "kwargs"，以便在 Langsmith 中跟踪和记录数据。他们报告称这些 kwargs 没有出现在 Langsmith 的日志项中，并正在寻找解决方案。
- **可配置的 Pinecone 命名空间请求**：一位成员询问如何使 Pinecone store 的命名空间可配置，以便根据发起 API 调用的用户更改命名空间。他们包含了一个代码片段，但在消息中未收到明确的解决方案。
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1244891137508900864)** (4 条消息): 

- **药物研发中的生成式 AI**：一位成员宣布了即将于 5 月 30 日举行的活动，题为 *“用于药物研发不同阶段的本地生成式 AI 模型框架”*。更多详情请见 [LinkedIn](https://www.linkedin.com/events/localgenerativeaimodelframework7200655391901323264/)。

- **降低日志记录成本**：一位成员分享了一个用于删除冗余日志的流水线，旨在帮助公司节省资金。他们建议使用 [此工具](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=langchain)，并选择 “verbose logs” 流水线。

- **飞行模拟器副驾驶**：一个正在进行中的项目，旨在为 Microsoft Flight Simulator 等飞行模拟器创建副驾驶。请在 [YouTube](https://www.youtube.com/watch?v=bUWcQSwZyPQ) 上查看演示视频。

- **Agent 工作流中的路由逻辑**：分享了一个关于在基于 LangChain 构建的 Visual Agents 中使用路由逻辑的科普视频。点击[此处](https://youtu.be/KtbRexZ6vsc)观看 YouTube 视频。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=langchain)">GitGud</a>：未找到描述</li><li><a href="https://youtu.be/KtbRexZ6vsc">如何在 Agent 工作流中进行逻辑路由</a>：关于如何在基于 LangChain 构建的 Visual Agents 中使用路由逻辑的简单示例。https://visualagents.ai https://langchain.ai
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1244747313117925396)** (20 条消息🔥): 

- **Mojo Python 版本支持提醒**：*请确保您使用的是受支持的 Python 版本 3.8 到 3.11。请注意，目前尚不支持 3.12。* 通过添加 deadsnakes 仓库并更新到 3.11 解决了版本问题。
- **关于 Tensor 包弃用的讨论**：针对 [YouTube 视频](https://youtu.be/uIG9q9foIw0?si=rhPqeQ_SsN8MIFur&t=1954)中提到的 Tensor 被弃用一事提出了疑问。澄清说明 Tensor 将被开源并从标准库中移除。
- **让 Mojo 在 Flutter 应用中更具实用性**：一位成员建议使用 Mojo 构建 Flutter 应用，以提高速度和部署能力，并引用了 [YouTube 教程](https://www.youtube.com/watch?v=5P8f5Tlim0M&t=278)。强调了结合 Flutter 的 UI 能力与 Mojo 的多功能性。
- **对 LLaMA2.mojo 项目的关注**：用户对 [llama2.mojo GitHub 项目](https://github.com/tairov/llama2.mojo)的教程表现出浓厚兴趣，重点关注 Mojo 中的推理和 AI 模型微调。社区成员被邀请加入 Discord 服务器以进一步讨论。
- **底层 GPU 汇编代码咨询**：*我应该在哪里询问有关底层 GPU 汇编代码的问题？* 讨论指向使用 Nsight 等工具处理 Python/Mojo。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=5P8f5Tlim0M&t=278">使用 Python 构建 Flutter 应用 - Flet 教程</a>：在本视频中，我将使用 Python 构建一个 MacOS Flutter 应用。我们将两者的优点结合起来：Flutter 的 UI 能力和 Python 的生态系统。我们...</li><li><a href="https://youtu.be/uIG9q9foIw0?si=rhPqeQ_SsN8MIFur&t=1954">Mojo 社区会议 #1</a>：Mojo 社区会议公开议程：https://modul.ar/community-meeting-doc</li><li><a href="https://github.com/tairov/llama2.mojo">GitHub - tairov/llama2.mojo: 在单个纯 🔥 文件中进行 Llama 2 推理</a>：在单个纯 🔥 文件中进行 Llama 2 推理。通过在 GitHub 上创建账号来为 tairov/llama2.mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1245077923401171004)** (6 条消息): 

- **开放世界游戏使用 AI 定制 NPC 智能水平**：一名成员建议，开放世界游戏可以根据 NPC 的智能程度提供订阅方案，为更具“意识”的 NPC 增加成本。他们强调这仅会是一项在线功能。
- **智能设备增加专用 AI 能力**：另一名成员分享道，未来的 AI 推理（inference）可能会在本地进行，因为许多智能设备现在都配备了加速器（accelerators），且 CPU 正在采用专门用于矩阵乘法的寄存器。这种硬件转变预示着 AI 处理将向分布式方向发展。
- **开放世界游戏中的自定义世界**：基于之前的想法，一名成员设想开放世界游戏可以利用 AI 根据玩家的互动构建自定义世界。他们认为有潜力利用庞大的在线模型库来增强游戏体验和个性化。
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1244753855925977089)** (14 条消息🔥): 

- **Mojo 支持循环依赖**：一名成员询问在 Mojo 中模块如何相互定义。另一名成员澄清说，由于模块建模的方式，特别是通过 `__init__.mojo` 根目录，Mojo 允许循环依赖（[示例解释](https://github.com/dorjeduck)）。
- **内置 Trait 自动导入**：关于无需显式导入即可使用 `Intable` 和 `Stringable` 等 Trait 的可见性问题得到了解答。解释称这些 Trait 是内置包的一部分，因此会自动导入。
- **^ 运算符的双重用途**：讨论澄清了 Mojo 中 `^` 运算符的双重功能。它既用于 XOR 操作，也用于向编译器发出对象生命周期结束的信号。
- **支持回调但尚未支持 Lambda**：成员们讨论了在 Mojo 中使用回调函数的问题，并指出 Lambda 函数尚未实现。目前探索了诸如将函数作为方法参数传递等替代方案。
- **增强 `vectorize` 函数**：有人提议修改 `vectorize` 函数，允许闭包函数返回一个 `Bool` 值用于循环控制，类似于某个 [进度条项目](https://github.com/dorjeduck/progressbar.mojo) 中的功能。这引起了成员们的兴趣并进行了进一步探索。
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1244731477489352778)** (21 条消息🔥): 

- **32 字节下实现 50 倍加速，但遇到缓存问题**：*fnands* 分享了一个详细的基准测试，展示了不同字节长度下的性能提升，在达到缓存限制前，在 32 字节处实现了最高 *50 倍的加速*。他们邀请 Apple silicon 用户使用 [GitHub 文件](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo) 进行进一步测试。

- **讨论 k-means 基准测试中的差异**：Cyrus_msk 指出，不同的内存分配实践和矩阵实现使得 Python 与 Mojo 的 k-means 算法基准测试对比并不等价。重点包括 *Mojo 中的预分配内存* 以及 `BLAS norm` 与并行 SIMD 优化的 norm 函数的对比。

- **预取（Prefetching）与缓存讨论**：*Fnands* 就 `prefetch` 选项寻求建议以提高性能，讨论了有无显式预取对性能的影响。Darkmatter__ 建议使用 Intel 的 [VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html) 等工具获取详细的 CPU 性能洞察，并强调了缓存行对齐（cache-line alignment）对于高效内存访问的重要性。

- **对齐内存以获得更好性能**：Darkmatter__ 建议确保内存表进行 *64 字节对齐*，以优化 AVX512 操作的性能，减少缓存管理不善并促进预取。他们还澄清说，避免伪共享（false sharing）主要在多线程场景中至关重要。

**提到的链接**：<a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo">fnands.com/blog/2024/mojo-crc-calc/crcn.mojo at main · fnands/fnands.com</a>：我的个人博客。通过在 GitHub 上创建账号为 fnands/fnands.com 的开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1244792723651694655)** (13 条消息🔥): 

- **对引用返回（Reference Returns）的兴趣**：成员们讨论了返回 `Reference` 的函数是否适合采用新的返回约定（return convention）。一位成员评论道，“几乎所有返回 `Reference` 的函数‘都应该’被转换”，例外情况是那些引用经常被用户存储的函数。
- **对结构化共享（Structural Sharing）的兴奋**：一位成员对引用的更改表示兴奋，强调它“实现了结构化共享”，这意味着多个 `structs` 可以共享某些字段。
- **新的 Nightly Mojo 编译器发布**：发布了最新的 nightly Mojo 编译器版本 `2024.5.2805`。此次更新包括 `tempfile.{mkdtemp,gettempdir}` 的实现，以及在标准库中增加了 `String.isspace()`；详细变更见 [当前变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [原始差异 (raw diff)](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3)。
- **贡献者身份澄清**：一位成员澄清说他们不是 Modular 的员工，而是一名贡献者。 
- **关于 PR 保密性的讨论**：关于在何处评论某个问题进行了简短对话，并对 PR 的保密性进行了调侃，一位成员指出他们“仍在等待”某个未公开 PR 的“绿灯”。
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1244734669988040746)** (68 条消息🔥🔥): 

- **Cursor 解释器模式令用户惊叹**：一位用户称赞 **Cursor 解释器模式（interpreter mode）** 的调试能力，将其描述为“一种能够遵循执行路径的更好搜索”，并且与传统搜索工具相比，在导航代码库（codebase）方面更具 Agent 特性。

- **微软 Copilot 现已登陆 Telegram**：用户对 **Microsoft Copilot** 集成到 **Telegram** 以获得更智能的聊天体验感到兴奋。该工具提供游戏技巧、电影建议、约会建议和食谱等功能，优化了日常对话。

- **低预算训练 GPT-2**：**Andrej Karpathy** 分享了一种使用 llm.c 在 **90 分钟内花费 20 美元**训练 GPT-2 (124M) 的方法，强调可以经济高效地管理 GPU 限制。详细说明见 [GitHub 讨论](https://github.com/karpathy/llm.c/discussions/481)。

- **微软区分 Copilots 与 Agents**：讨论了 **Microsoft Build** 区分 **Copilots** 和 **Agents** 的决定，Copilots 更具个性化且基于 Prompt，而 Agents 则自主运行。提到了一篇非常有见地的 [Kanjun Qiu 访谈](https://www.latent.space/p/imbue)。

- **向量数据库集成咨询**：一位用户正在寻找类似于 ORM 的**向量数据库抽象**，以便更轻松地集成和切换不同的向量数据库。建议使用 **LangChain** 和 **LlamaIndex**，并进一步推荐使用 **pgvector** 进行高效的 Embedding 存储和关系型元数据（metadata）管理。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">《构建 LLM 一年的经验教训（第一部分）》</a>: 未找到描述</li><li><a href="https://blog.reachsumit.com/posts/2023/03/llm-for-text-ranking/">《使用 Large Language Models 进行 Zero and Few Shot 文本检索与排序》</a>: Large Language Models (LLMs)，如 GPT-x、PaLM、BLOOM，已经撼动了 NLP 领域，并完全重新定义了各种任务的 SOTA。这些 LLM 受欢迎的原因之一是...</li><li><a href="https://x.com/xlr8harder/status/1795515600795767058">xlr8harder (@xlr8harder) 的推文</a>: @karpathy @swyx 在小 token 计数下获得良好样本的选择并不多。不幸的是，FineWeb 提供较小的样本是例外而非普遍规律。但是 S...</li><li><a href="https://x.com/karpathy/status/1795484547267834137">Andrej Karpathy (@karpathy) 的推文</a>: # 在 90 分钟内花费 $20 用 llm.c 复现 GPT-2 (124M) ✨ GPT-2 (124M) 是 OpenAI 在 2019 年发布的 GPT-2 系列中最小的模型，如今实际上非常容易获取，即使对于 G...</li><li><a href="https://x.com/borismpower/status/1795475031658516933?s=46">Boris Power (@BorisMPower) 的推文</a>: 4+1=5</li><li><a href="https://x.com/polynoamial/status/1795422304937411029?s=46&t=90xQ8sGy63D2OtiaoGJuww">Noam Brown (@polynoamial) 的推文</a>: 下一个 OpenAI frontier model 已开始训练！https://openai.com/index/openai-board-forms-safety-and-security-committee/</li><li><a href="https://www.microsoft.com/en-us/edge/copilot-for-social?form=MY02F9">Telegram 版 Copilot | Microsoft Copilot</a>: 未找到描述</li><li><a href="https://x.com/GergelyOrosz/status/1794743519954731331">Gergely Orosz (@GergelyOrosz) 的推文</a>: 如果构建一个性能比最强 LLMs 高出约 4 倍的 AI coding agent 具有十亿美元的潜力：这里有 7 位普林斯顿大学的研究人员做到了这一点。它是完全 open source 的，名为 SWE-agent....</li><li><a href="https://www.microsoft.com/en-us/edge/copilot-for-social?form=MY02F9&ch=1">Telegram 版 Copilot | Microsoft Copilot</a>: 未找到描述</li><li><a href="https://www.latent.space/p/imbue">《为什么 AI Agents 还不起作用（目前） —— 对话 Imbue 的 Kanjun Qiu》</a>: 立即收听 | 关于筹集 2 亿美元构建能够 reasoning 和 coding 的 agent 操作系统，为什么 LLMs 在 agent 用例中击败了 reinforcement learning，以及如何与顶尖 AI 人才共同构建 Scenius。</li><li><a href="https://x.com/khoomeik/status/1795477359933706272">Rohan Pandey (e/acc) (@khoomeik) 的推文</a>: 📢 很高兴终于发布了我的 NeurIPS 2024 投稿！Chinchilla 是通用的吗？不！我们发现：1. language model scaling laws 取决于数据复杂度 2. gzip 有效地预测了 scaling...</li><li><a href="https://anysphere.inc/blog/problems-2024">《2024-2025 年的问题》</a>: 未找到描述</li><li><a href="https://x.com/siddrrsh/status/1795541002620727439?s=46&t=90xQ8sGy63D2OtiaoGJuww">Siddharth Sharma (@siddrrsh) 的推文</a>: 介绍 Llama3-V，一个 SOTA open-source VLM 模型。我们的特点：• 性能超越 LLaVA • 以 100 倍更小的模型实现与 GPT4-V, Gemini Ultra, Claude Opus 相当的性能 • 针对 L 的 SOTA open source VLM...</li><li><a href="https://x.com/lmsysorg/status/1795512202465845686">lmsys.org (@lmsysorg) 的推文</a>: 重大新闻 —— Gemini 1.5 Flash, Pro 和 Advanced 的结果出炉了！🔥 - Gemini 1.5 Pro/Advanced 排名第 2，逼近 GPT-4o - Gemini 1.5 Flash 排名第 9，超越了 Llama-3-70b，几乎达到 GPT-4-01...
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1244758635419009045)** (3 条消息): 

```html
- **关于 ICLR 2024 论文的新播客**：发布了一个涵盖 ICLR 2024 亮点的新剧集，介绍了各种突破性的论文和演讲。[点击此处收听](https://x.com/latentspacepod/status/1795196817044594817) 获取关于 ImageGen、Compression、Adversarial Attacks、Vision Learning 等方面的见解。
- **聚焦 ImageGen 和 Compression**：讨论的主题包括 "Auto-encoding Variational Bayes" 和 "Würstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models"。值得关注的还有来自 Ilya Sutskever 和 Christian Szegedy 的详细见解。
- **Vision Learning 的进展**：播客深入探讨了诸如 "Vision Transformers Need Registers" 和 "Think before you speak: Training Language Models With Pause Tokens" 等论文。它还研究了弱监督下数据选择的统计理论。
- **增强 Transformer 模型**：讨论了使用 "LongLoRA" 和 "YaRN" 等论文对 Large Language Models 进行高效 Fine-tuning 和 Context Window 扩展。还涉及了自适应 KV cache 压缩和巨型模型训练的高效通信等主题。
- **State Space Models 对比 Transformers**：论文 "Never Train from Scratch" 强调了长序列模型中数据驱动先验的重要性。请关注第二部分中关于 LLM Reasoning 和 Agents 的更多内容。
```

**提到的链接**：<a href="https://x.com/latentspacepod/status/1795196817044594817">来自 Latent Space Podcast (@latentspacepod) 的推文</a>：🆕 ICLR 2024：最佳论文（第一部分）。我们展示了我们挑选的优秀论文和演讲，主题性地介绍了 AI Engineers 需要关注的话题：A 部分：ImageGen、Compression、Adversarial ...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245093324428283965)** (1 条消息): 

- **FinTextQA：发布新的金融数据集**：查看 [FinTextQA](https://t.co/emhQYXY1S4)，这是由 Jian Chen 及其团队推出的用于长篇金融问答的新数据集和 RAG 基准测试。它具有 *6 种不同的问题类型*，并包含 *1,262 个高质量、带来源属性的问答对*以及相关的文档上下文。
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1244754850747515014)** (59 条消息🔥🔥): 

- **实现完美的 System Role Prompt 结构**：一位用户询问是否有关于完美 System Role Prompt 结构的文章，类似于 **LlamaIndex** 在其示例中使用的结构。
  
- **LlamaIndex 中聊天记录的持久化**：用户讨论了如何保存来自 **NLSQL** 和 **PandasQuery** 引擎结果的聊天记录。建议包括创建一个自定义 Retriever 来包装 Query Engine。

- **在 API 中使用 Function Calling 处理多个函数**：成员们集思广益，探讨管理具有 1000 个独立函数的 API 的策略。想法包括使用分层路由（Hierarchical Routing）将函数划分为可管理的子组。

- **LLMSelector 与 PydanticSelector 的区别**：详细解释了 **LLM selectors** 如何使用文本补全端点生成查询数据，而 **Pydantic selectors** 如何使用 Pydantic 对象进行 Function Calling API。

- **LlamaIndex 在 RAG 系统中的挑战与解决方案**：用户讨论了元数据处理、嵌入较小的语义 Chunks 与较大 Chunks 的影响，以及信息检索准确性中潜在的权衡。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/examples/query_engine/RouterQueryEngine#define-router-query-engine>)">Router Query Engine - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/router#defining-a-selector>)">Routing - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1244733174014279712)** (40 条消息🔥): 

- **糟糕的 SOTA AGI 预测中的幽默**：多位成员讨论了当前 SOTA AGI 模型状态中的幽默与绝望。一个有趣的观点是，某个模型据称是自我训练的，并评论道“它为我们训练了一个模型”。
- **Hugging Face 上的 Corcelio Mobius Art Model**：分享了由 Corcelio 的 [Mobius Art Model](https://huggingface.co/Corcelio/mobius) 生成的图像，提示词涵盖了从“灭霸闻着一朵黄色小玫瑰”到“灵魂的解经学”。该模型因超越了先前模型的限制而受到关注，但令人惊讶的是它会生成水印。
- **社区对图像系统伦理的担忧**：有人担心人们使用图像生成系统创建不当内容，包括“不该做的色情内容”。这一问题引发了关于网站提示词和 Sampler 设置的疑问。
- **训练不足的模型问题和水印**：讨论了 imgsys 上新的未命名 T2I 模型，观察到它经常显得训练不足，并且常规性地生成水印。一些成员认为这是一个经常出现且幽默的主题。
- **Elon Musk 嘲讽 CNN 的推文**：分享了 Elon Musk 的一条推文，强调他们“如今不怎么使用 CNN 了”，这引发了幽默的反应，例如建议改用 Vision Transformer 模型。成员们调侃了行业趋势的变化以及回复中使用的讽刺手法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Corcelio/mobius">Corcelio/mobius · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/elonmusk/status/1795405972145418548">Elon Musk (@elonmusk) 的推文</a>：@ylecun @Scobleizer 说实话，我们这些天不怎么使用 CNN 了
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245086640259727443)** (2 条消息): 

- **SDXL 在生成“阅读中的眼睛”方面遇到困难**：一位成员发现 **SDXL** 无法生成女性阅读的近距离肖像。他们分享了在 Horde 中使用的详细提示词和生成设置。
- **呼吁通过 DALL-E 生成数据**：同一位成员标记了其他用户，建议使用 **DALL-E** 生成图像。他们的目标是创建一个合成数据库，作为训练材料，以改进 **SDXL** 对“阅读中的眼睛”的生成效果。
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1244830671160213585)** (25 条消息🔥): 

- **GPU 延迟建模或许可行**：一位成员建议，基于不同内存类型之间的数据移动和操作时间，在不运行 Kernel 的情况下对 GPU 延迟和运行时进行符号化建模可能是可行的。然而，Occupancy 和异步操作可能会使模型复杂化。

- **探索用于 Kernel 优化的 Halide 和 AutoTVM**：成员们讨论了 AutoTVM 和 Halide autotuner 等工具，指出 Halide 使用手写模型的学习加权，而 AutoTVM 可能使用经验方法。George Hotz 指出 TVM 使用 XGBoost，并强调了正确模拟 Cache Hierarchy 对准确建模的重要性。

- **Cycle Accurate GPU 模拟器具有高精度**：有推测称量化交易公司可能使用 Cycle Accurate GPU 模拟器进行 Kernel 优化，提供非常详细的 Profiling 能力。然而，这些模拟器在评估速度上是否优于经验方法受到了质疑。

- **期待 AMD MES 开源发布**：简要提到了 AMD 开源 MES 的计划，相关文档显然已经发布，社区正急切期待源代码。

- **GPU 延迟隐藏策略的差异**：成员们强调不同的 GPU 采用各种延迟隐藏策略，这使得准确建模延迟变得困难。GPU 中大量的并发 Wavefronts/Blocks 使其能够比预想更有效地处理延迟。

**提到的链接**：<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1244732460726747166)** (5 messages): 

- **后支配者分析（Post dominator analysis）的困惑**：一位成员询问为什么在调度（scheduling）过程中不使用**后支配者分析**来识别用于融合（fusion）的自包含子图。从理论上讲，这种技术可以提高某些计算的效率。
  
- **创建包含多个值的 LazyBuffer**：一位成员询问如何从一组值（而非单个值）创建 **LazyBuffer**。回复中提到使用 *Load.EMPTY -> Load.COPY* 作为通用方法，并提到了 `full` 和 `rand` 等工厂方法以便于创建。
  
- **提供代码指针以供参考**：在详细解释之后，一位成员表示愿意提供**代码指针（code pointers）**以帮助理解。初始解决方案的参考资料包含了关于为创建 **LazyBuffer** 模拟缓冲区分配的见解。
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1244792392154878012)** (25 messages🔥): 

- **Elevenlabs 文本转语音（Text-to-Speech）集成亮相**：一位用户询问了文本转语音的模组，另一位提到在一个分支中集成了 **Elevenlabs**。提供了 [textToSpeech 的代码](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19)。

- **在 AI Town 中实现文本转语音**：设置文本转语音功能的步骤包括将文本转换为音频、用音频 URL 修补消息以及在前端处理音频播放。据指出，该过程很快，但有近一秒的延迟，这使得实时实现具有挑战性。

- **对科学辩论的兴趣**：一位用户表示打算通过让 AI 聊天机器人辩论科学话题来创造一种引人入胜的体验。该用户看重科学凝聚人心和创造希望的力量。

- **添加窃听机制**：Zaranova 分叉（fork）包含一种窃听机制，可以为附近的对话生成音频。可以添加此功能以丰富 AI Town 的互动体验。

- **对协作编码的兴趣**：另一位用户表示有兴趣为该功能做出贡献，并承诺研究创建 pull request。还有人有兴趣将这些更改合并到 AI Town 主项目中。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/aiTown/agent.ts#L568">ai-town/convex/aiTown/agent.ts at e7e2182eb7f7241e58c69d8324ae126c1d34dee9 · huevosabio/ai-town</a>: 一个采用 MIT 许可、可部署的入门套件，用于构建和定制你自己的 AI town 版本——一个 AI 角色居住、聊天和社交的虚拟城镇。 - huevosabio/ai-town</li><li><a href="https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19">ai-town/convex/util/textToSpeech.ts at e7e2182eb7f7241e58c69d8324ae126c1d34dee9 · huevosabio/ai-town</a>: 一个采用 MIT 许可、可部署的入门套件，用于构建和定制你自己的 AI town 版本——一个 AI 角色居住、聊天和社交的虚拟城镇。 - huevosabio/ai-town
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/)** (1 messages): 

gomiez: 嗨。我该如何停止对话关闭？我读不了那么快。
  

---


### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/)** (1 messages): 

angry.penguin: 如果你在推理（inference）方面有任何进展，请告诉我。
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1244728730039881800)** (12 messages🔥): 

- **优化日志流水线以节省成本**：一位成员分享了他们开发的一个流水线，用于移除可能因失误添加并推送到生产环境的冗余日志，从而节省日志记录成本。他们使用了 [这个工具](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere) 并建议选择 "verbose logs" 流水线。

- **关于云端/本地部署选项的咨询**：有人询问了关于重排序（reranking）和查询提取（query extraction）的云端/本地（cloud-prem）部署选项。该成员寻求有关可用解决方案或最佳实践的见解。

- **为 RAG 微调 Cohere 模型**：一位用户询问是否可以微调 Cohere 模型以回答财务问题，然后将其与 RAG（检索与生成）结合使用，以便根据 SEC 文件（美国证券交易委员会备案）来聚焦回答。

- **Aya23 模型仅限于非商业用途**：会议澄清了 **Aya23 模型** 仅限于非商业用途，因为它们仅用于研究目的。目前没有商业化使用的计划，即使是针对小型初创公司。

- **使用自定义数据创建 DPO 数据集**：一位成员询问了创建自定义 DPO 数据集的技巧，考虑的选项包括使用 GPT-4 生成响应对，或将 GPT-4 的输出与基础模型的响应相结合。

**提到的链接**：<a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)">GitGud</a>：未找到描述

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1244762231497818193)** (3 messages): 

- **由 Cohere 驱动的游戏机器人上线**：一位成员分享了他们的新作品，一个使用 **Cohere Command R** 的 Discord 游戏机器人。他们提到，这个名为 **Create 'n' Play** 的机器人拥有 *"超过 100 款引人入胜的文字游戏"*，旨在通过 **AI** 增强社交互动。

- **在 LinkedIn 上查看该游戏机器人**：这篇 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios) 提供了关于该项目开发和功能的更多见解。它的目标是在 Discord 社区内实现轻松的组队和互动。

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1244883894264926329)** (4 messages): 

- **仅限推理的查询引发讨论**：一位成员询问该话题是否仅针对推理。回应将对话引向了训练复杂性和性能考量。
- **训练瓶颈集中在 FLOPS**："训练几乎总是受限于 FLOPS。" 当处理 batch size 为 1 且序列长度为 4096 时，由于 teacher forcing 方法，有效 batch size 实际上是 4096。
- **在 Hopper 卡上进行 FP8 原生训练**：有人表示有兴趣探索 "在 Hopper 卡上进行 FP8 原生训练"。这表明社区关注利用先进硬件功能来优化训练性能。
- **承认过去的错误**：一位成员幽默地承认：*"没错，我当时错了。"* 这种交流展示了社区内公开承认错误和学习的文化。

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1244982851326185474)** (1 messages): 

- **正确更新 fschat**：一位成员解释说 **fschat** 的版本标识符没有更新，导致了问题。他们建议卸载并重新安装 **fschat** 以解决此问题。

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1245054726706430054)** (4 messages): 

- **澄清 CUTLASS_PATH 的必要性**：一位成员询问是否应该设置 `CUTLASS_PATH`。Phorm 回应称应检查其项目或工具是否需要 **CUTLASS (CUDA Templates for Linear Algebra Subroutines and Solvers)**，这对于深度学习应用中的高性能矩阵运算至关重要。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=58c281a8-ece1-46c7-8057-cd6cb7902a51)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1244912674362364015)** (2 messages): 

- **YI 和 YI-VL 模型更新至 Apache 2.0**：一位成员分享了 **YI 和 YI-VL（多模态 LLM）模型** 已更新至 **Apache 2.0**，加入了 1.5 系列。该更新由 [@_philschmid](https://fxtwitter.com/_philschmid/status/1795343334225129570) 宣布，并感谢 @01AI_Yi 提供的更新。
- **Gemini 1.5 系列引起轰动**：[@lmsysorg 宣布](https://x.com/lmsysorg/status/1795512202465845686?s=46) **Gemini 1.5 Pro/Advanced 和 Flash 的结果** 已出炉，其中 Pro/Advanced 排名第 2，逼近 GPT-4o。**Gemini 1.5 Flash** 排名第 9，表现优于 Llama-3-70b，突显了其性价比、能力以及无与伦比的 context length 等重大进步。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1795512202465845686?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>: 重大新闻 – Gemini 1.5 Flash, Pro 和 Advanced 结果出炉！🔥 - Gemini 1.5 Pro/Advanced 排名第 2，逼近 GPT-4o - Gemini 1.5 Flash 排名第 9，超越 Llama-3-70b 并接近 GPT-4-01...</li><li><a href="https://fxtwitter.com/_philschmid/status/1795343334225129570">来自 Philipp Schmid (@_philschmid) 的推文</a>: 更多 Apache 2.0！🚀 @01AI_Yi 刚刚将 YI 和 YI-VL（多模态 LLM）模型更新为 Apache 2.0，加入了 1.5 系列。🙌 谢谢！
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245108911778959360)** (6 messages): 

- **OpenAI 在 Twitter 上得知 ChatGPT 的消息**：前 OpenAI 董事会成员透露：“就像 2022 年 11 月 ChatGPT 发布时，董事会并未提前获知。我们是在 Twitter 上得知 ChatGPT 的。”（[查看推文](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298)）。
- **播客中的震惊揭露**：前 OpenAI 董事会成员 Helen Toner 揭露 **Sam Altman** 因不诚实、营造有毒工作环境以及被指控“心理虐待”而被解雇。她呼吁“对 AI 公司进行外部监管”，理由是自我治理可能并不总是有效（[播客链接](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3)）。
- **Natolambert 的反应**：Natolambert 对 Helen Toner 的揭露反应强烈，惊呼“holy shit”，随后质疑“helen 真的要拯救世界了吗？”
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/btibor91/status/1795551083420430579">来自 Tibor Blaho (@btibor91) 的推文</a>: @TheXeophon https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3</li><li><a href="https://fxtwitter.com/bilawalsidhu/status/1795534345345618298">来自 Bilawal Sidhu (@bilawalsidhu) 的推文</a>: ❗独家：“我们是在 Twitter 上得知 ChatGPT 的。” OpenAI 到底发生了什么？前董事会成员 Helen Toner 打破沉默，透露了关于 Sam Altman 被解雇的令人震惊的新细节……
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1245028765575413912)** (3 messages): 

- **分享可靠的 AI 模型排行榜**：一位用户询问寻找最佳模型的优质网站，并分享了 [一个排行榜链接](https://chat.lmsys.org/?leaderboard)。Simon 确认这是他最喜欢的 LLM 比较网站，称其非常可靠。
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1244977023458349168)** (3 messages): 

- **AI 模型的本地端点必须安全**：一位成员对 *“普及化、标准化并启用本地可用端点”* 表示兴奋，但强调需要安全的验证机制，如 DNS SRV 记录和 pub keys（公钥）。他们幽默地指出验证本地 AI 模型可信度的重要性，否则可能会变成 *“买乡村音乐或者……去喂松鼠。”*

- **`granite-34b-code-instruct.llamafile` 出错**：在尝试运行来自 Hugging Face 的 llamafile 时，一位成员遇到了 *“unknown argument: --temp”*（未知参数：--temp）错误。该过程涉及下载、更改权限并运行文件，最终导致了此问题。

- **Llamafiles 存储并运行单个模型**：会议澄清了无论什么模型在 `localhost:8080` 运行，它就是被使用的模型，例如 *tinyllama*。在这种情况下，聊天补全请求中的 `model` 字段被认为是无关紧要的。

**提到的链接**：<a href="https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true">未找到标题</a>：未找到描述

  

---



### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1244754490918309968)** (3 messages): 

- **R1 简直就是个镇纸**：一位成员提到：*“我还在坚持希望 R1 能兑现承诺。如果不行，它就是一个漂亮的镇纸。”*。
- **寻求解决方案和更新**：另一位成员表达了好奇，说：*“如果你发现任何相关信息，请告诉我。”*。
- **需要邮件回复**：一位成员请求 OI 团队的协助，表示：*“嗨 OI 团队，我几天前发了一封邮件，至今还没收到回复。我刚刚又发了一次提醒。”*。
  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1244853808887234621)** (2 messages): 

```html
- **服务器似乎无人管理**：一位成员指出“看起来服务器无人管理……”，强调了明显的管理缺失。
- **尝试 @everyone 提醒失败**：同一位成员尝试使用 @everyone 标签，但注意到它并没有像预期那样“发出提醒”。
```
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1244956594589143084)** (1 messages): 

- **关于课程内容的问题**：一位成员询问：*“这门课对你来说怎么样？他们会教你如何使用 LLM 自动化后端服务吗？”* 频道中没有对此查询的回复。
  

---



---



---




{% else %}




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**OCR 对决：Google Vision vs. Microsoft Azure**：AI 工程师们辩论了 **Google Vision OCR** 的优缺点，承认其精度但批评了开发者体验。有人建议使用 **Microsoft Azure OCR** 和 **Mindee Doctr**，认为它们可能提供更好的易用性，详情见[这里](https://huggingface.co/spaces/mindee/doctr)。

**精选数据：LLM 成功的关键**：研讨会讨论强调了使用高质量、精选数据集微调 LLM 的重要性，应用范围从医药应用到技术支持聊天机器人。专家意见指出，需要精确选择数据以最大化 LLM 的有效性，重点关注药物研发、法律、销售和跨学科工作等领域。

**Axolotl 的焦虑与优化**：用户在 M3 Macs 上运行 **Axolotl 的 70B 模型**时遇到障碍，本地推理时延迟极高，指出在 Modal 上部署可能是一个解决方案。对 **Weights & Biases (WandB)** 成本的担忧促使注重经济效益的独立开发者考虑 **Aim** 和 **MLflow** 等替代方案 [Axolotl 示例](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/qlora-fsdp-70b.yaml)。

**LLM 评估深度探讨**：关于评估 LLM 的会议提供了大量见解，涵盖了产品指标、传统和动态性能指标，以及 LangFuse 和 EvalGen 等工具。通过推荐 Eugene Yan 的资源和可视化微调的实际案例，参与者注意到对 LLM 开发进行细致评估的必要性。

**转录纠结与摘要之路**：围绕大型会议转录文本的交流阐明了对高效摘要的需求，揭示了 LLM 可能发挥的作用。虽然 Zoom 转录功能即将推出，但 Hamel 鼓励使用 LLM 生成更易读的摘要，这引起了更广泛的社区参与。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **翘首以盼 imfo Alpha 版本发布**：[@spectate_or](https://x.com/spectate_or/status/1795077451195830661?s=46) 的一条推文链接暗示了即将发布的 **imfo alpha**，这在工程社区中引发了兴奋，并将其与类似工具进行了比较。

- **AI 任务结构辩论**：工程师们讨论了将 **AI 任务**分类为检索型和变异型，并以“获取 iPhone 15 的重量”等查询为例。针对需要顺序执行的任务，强调了调整的必要性，并指出 *“所有步骤几乎是同时发生的。”*

- **爬取准确性遇到障碍**：成员们表达了在 **HTML 解析**以实现可靠数据爬取方面面临的挑战，复杂性源于 Apple 和 Docker 发布说明等网站。针对以 JavaScript 为中心的网站，考虑通过 **Playwright** 进行变通，同时也讨论了 Cloudflare 的相关问题。

- **探索高性价比的 AI 模型利用**：社区深入探讨了使用 Llama3 和 Claude 等各种 **AI 模型**的成本效益。一种使用组合系统的方法表明了实现更大成本节约的可能性。

- **强调 API 功能的奇特之处**：围绕显示 JSON 对象但缺少功能链接的 **API 输出**产生了困惑，这可能与缺少 **closed beta citations feature**（内测引用功能）有关。其他讨论还包括改进视频链接生成的 Prompt，以及对潜在 API 故障的简短询问。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**值得尝试的新 AI 功能**：Stability AI 宣布推出 **Stable Assistant**，它具备基于 **Stable Diffusion 3** 构建的编辑功能，并拥有更高的文本生成图像质量，可在此处进行[免费试用](https://stability.ai/stable-assistant)；此外还推出了搭载 **Stable LM 2 12B** 的 Beta 版聊天机器人，预示着未来文本生成任务的增强。

**教育与 AI 创新融合**：由 Stability AI 和 HUG 合作的 **Innovation Laboratory** 即将开展为期 4 周的课程，旨在指导参与者结合 HUG 的教育方法，利用 Stability AI 的框架训练 AI 模型；报名截止日期为 2024 年 6 月 25 日，可通过[此处](https://www.studios.thehug.xyz/lab)访问。

**GPU 共享成为焦点**：AI 工程师讨论了一项基于社区的 GPU 共享提案，以降低计算成本，方案从自定义节点到旨在验证模型训练操作的潜在区块链设置不等。

**SD3 的可访问性引发争议**：由于成员对 **Stable Diffusion 的 SD3** 权重无法在本地使用表示不满，争议浮出水面——他们批评 Stability AI 仅限云端的做法，并引发了关于云端依赖和数据隐私问题的辩论。

**用户界面对比**：一场关于 Stable Diffusion 各种界面优缺点的技术讨论展开，**ComfyUI** 与 Forge 等更易于使用的替代方案展开竞争；讨论还包括社区技巧、Inpainting（局部重绘）方法以及增强人工智能工作流的方法。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**OpenAI 建立安全盾牌**：OpenAI 成立了一个 **Safety and Security Committee**（安全与安保委员会），负责其所有项目的关键安全和安保决策；详细信息可见其[官方公告](https://openai.com/index/openai-board-forms-safety-and-security-committee/)。

**AI 在硬件领域大显身手**：关于硬件成本的讨论兴起，推测由于 **NPUs**（神经网络处理单元）的加入，成本将增加 200 至 1000 美元，重点关注其对高端模型的经济影响。

**规划 Prompt 蓝图**：AI 工程师辩论了 **meta-prompting** 与 **Chain of Thought (CoT)** 的优劣，探讨了使用 mermaid 图表来节省 tokens 并提高输出质量的潜力。此外还分享了改进后的 Prompt（如[此处](https://chatgpt.com/share/4de63e2d-d59b-4b3e-87b8-68a71c5df477)），展示了高级 Prompt Engineering 策略的实际应用。

**理论付诸代码实践**：实际讨论包括 AI 如何原生处理 **YAML, XML, and JSON** 格式，并建议在 Prompt 中使用这些结构以提高 AI 的理解力和性能，同时分享了指向生成代码和规划的实际 Prompt 应用资源。

**交互不一致性引发探究**：用户报告了 **ChatGPT** 的一系列问题，从拒绝抽取塔罗牌到上下文丢失和无响应，突显了对改进和更可预测的 AI 行为的需求。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**语音指令遇见机器人**：一段名为 ["Open Source Voice-Controlled Robotic Arm"](https://www.youtube.com/watch?v=qv3bFhHoA5s) 的演示视频展示了一个语音激活的 AI 机器人手臂。视频提出了通过社区协作实现机器人技术民主化的观点。

**跨越模态**：关于创建早期多模态空间的贡献指出，可以使用单一模型，也可能使用具有路由功能的堆叠模型。为了深入了解此类实现，分享了一个 [源码链接](https://huggingface.co/spaces/KingNish/OpenGPT-4o/blob/main/app.py)，提供了一个具有实际应用价值的模型示例。

**即时深度学习咨询**：一位用户就使用 Stanford Cars Dataset 训练模型时遇到的常见痛点咨询了社区，该用户使用 ViT-B_16 仅达到了 60% 的准确率，并深受过拟合困扰。与此同时，另一位成员正在寻求如何改进其深度学习模型的帮助，这表明社区拥有支持新手知识交流的良好环境。

**Diffusers 更新：不仅限于生成**：Hugging Face 宣布其 **Diffusers 库现在支持生成模型之外的任务**，例如通过 **Marigold** 进行深度估计和法线预测。此次更新表明 Diffusion 模型的多功能性及其应用呈现出不断扩大的趋势。

**网络安全评估的模型选择**：研究人员的分析探讨了各种 Large Language Models 在网络安全背景下的能力。这为 AI 工程师提供了一个视角，去考虑部署 LLM 时固有的安全影响。

**稳健的 SDXL 空间重对齐**：关于 SDXL 嵌入空间的讨论强调，新对齐的空间默认值为零，而不是编码空间。这些见解反映了将模型重新对齐到新的无条件空间（unconditioned spaces）所涉及的底层复杂性和时间需求，揭示了科学背后的复杂过程。

**Gradio 升级版客户端引发关注**：Gradio 团队宣布即将举行一场直播活动，深入探讨 Gradio Python 和 JavaScript 客户端的最新功能。此次活动邀请强调了 Gradio 致力于通过增强的界面不断简化 AI 到各种应用程序的集成。
  
**寻找 SFW 数据集的模糊性**：社区讨论提到了定位 Nomos8k_sfw 数据集的困难，该数据集与 4x-Nomos8kDAT 模型相关联，这表明该数据集的可用性有限或位置隐蔽。这突显了数据集获取过程中偶尔会遇到的挑战。

**发布最新的 AI 叙事工具**：Typeface Arc 作为一个综合平台出现，旨在无缝创建 AI 驱动的内容。它包含一个被恰当地称为 "Copilot" 的工具，旨在通过对品牌叙事至关重要的交互式体验来增强内容创作。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**视觉化：OpenAI 与 LLaVA 集成！**：工程师现在可以通过在服务器上部署 **LLaVA** 并利用提供的 Python 视觉模板，在 LM Studio 中利用其视觉能力。

**M1 Max 上的快速模型加载**：像 **MLX 和 EXL2 这样的 AI 模型在 Apple 的 M1 Max 上加载迅速**，L3 8bit 仅需 5 秒，表明其性能优于需要 29 秒的 GGUF Q8。

**LM Studio 微调的挫败感**：尽管是一个强大的环境，但 **LM Studio 目前缺乏直接微调模型的能力**，爱好者们被引导至专为 Apple Silicon 设计的 MLX 等替代解决方案。

**预算还是性能**：AI 从业者辩论了各种 Nvidia GPU 的价值主张，考虑了 **Tesla P40/P100** 等替代方案，并满怀期待地讨论了传闻中的 **5090** 等 GPU。

**Beta 测试的烦恼**：在体验新版本时，用户报告了诸如大模型的 **Windows CPU 亲和性问题**以及 **AVX2 笔记本电脑上的错误**等问题，这暗示了为 AI 任务配置现代硬件的复杂性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-2 不受 Unsloth 待见**：Unsloth 确认，由于基础架构的根本差异，无法使用其平台对 **GPT-2** 进行微调。

- **Fiery Chat 微调中的挫折**：
  - 在对包含 50,000 多个邮件条目的 Llama 3 进行微调时，成员们分享了关于构建 Prompt 结构以实现最佳输入输出配对的建议。
  - 针对训练后出现的句子重复问题，建议添加 End-Of-Sentence (EOS) Token，以防止模型过拟合或学习效果不佳。

- **视觉模型集成指日可待**：成员们正热切期待 **Unsloth** 下个月支持视觉模型的更新，目前暂且推荐使用 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 和 [Segment Anything](https://github.com/facebookresearch/segment-anything) 作为当前的解决方案。

- **LoRA Adapter 的协同工作**：社区分享了合并和微调 LoRA Adapter 的技巧，强调利用 [GitHub 上的 Unsloth 文档](https://github.com/unslothai/unsloth#-finetune-for-free)等资源，并将模型导出到 HuggingFace。

- **应对 Phi 3 Medium 的注意力跨度**：关于 **Phi3-Medium** 的讨论揭示了其滑动窗口注意力（Sliding Window Attention）会导致在高 Token 计数时效率下降，许多人渴望能有增强功能来处理更大的上下文窗口。

- **ONNX 导出详解**：提供了将微调后的模型转换为 **ONNX** 的指南，参考了 Hugging Face 的 [序列化文档](https://huggingface.co/docs/transformers/en/serialization)，并确认 VLLM 格式兼容转换。

- **迈向低位宽**：Unsloth 即将支持 8-bit 模型以及与 Ollama 等环境的集成能力（类似于 OpenAI 的产品），大家对此充满期待。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Ubuntu 上的 CUDA Toolkit 命令**：一位用户建议从 NVIDIA 安装 **CUDA Toolkit**，通过 `nvidia-smi` 检查安装情况，并提供了在 Ubuntu 上设置的命令，包括通过 Conda 安装：`conda install cuda -c nvidia/label/cuda-12.1.0`。同时，在设置 PyTorch 2.3 时发现了 Python 3.12 与缺失 **triton** 安装之间的潜在冲突，这与一个 [GitHub Issue](https://github.com/pytorch/pytorch/issues/120233) 相关。

- **GPT-4o 在处理大型编辑时遇到对手**：成员们注意到 GPT-4o 在处理大量代码编辑时表现吃力，而一种新的 **fast apply** 模型旨在将任务分解为计划和应用阶段以克服这一挑战。为了寻求代码编辑的确定性算法，一位成员提出了使用 **vllm** 或 **trtllm** 进行未来 Token 预测（Future Token Prediction）而无需依赖草稿模型的可行性。更多关于此方法的信息可以在 [完整博客文章](https://cursor.sh/blog/instant-apply) 中找到。

- **SYCL 调试困扰**：一位成员询问了调试 SYCL 代码的工具，引发了关于进入 Kernel 代码进行故障排除的讨论。

- **Torchao 的最新进展**：torchao 社区庆祝了 PyTorch 合并对 MX 格式（如 `fp8/6/4`）的支持，这为感兴趣的各方提供了效率提升，部分由一个 [GitHub Commit](https://github.com/pytorch/ao/pull/264) 提供，并符合 [MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

- **理解 DIY 中的 Mixer 模型**：成员们剖析了实现细节，例如在 **llm.c** 中集成 `dirent.h`，以及为了操作系统兼容性使用 `#ifndef _WIN32` 进行保护的重要性。实现了用于在中断时恢复训练的 `-y 1` 标志，解决了关于未初始化变量的警告，并探索了 Backward Pass 计算期间的内存优化策略，相关倡议可在 [GitHub 讨论](https://github.com/karpathy/llm.c/discussions/481) 中找到。

- **BitNet 中的激活量化**：在 BitNet 频道中得出结论，在激活量化神经网络中直接传递传入梯度可能是错误的。相反，建议使用 `tanh` 等替代函数的梯度，并引用了一篇关于直通估计器 (STE) 性能的 [arXiv 论文](https://arxiv.org/abs/1903.05662)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT Agents 无后续学习能力**：基于 GPT 的 Agent 在初始训练后不会进行后续学习，但可以引用上传为“知识文件”的新信息，而不会从根本上改变其核心理解。
- **Diffusion Models 的效率里程碑**：Google DeepMind 推出 **[EM Distillation](http://arxiv.org/abs/2405.16852)** 以创建高效的一步生成器 Diffusion Models，Google 的另一项独立研究展示了一个能够生成 1024x1024 高分辨率图像的 8B 参数 Diffusion Model。
- **追求极致的小型化**：**[Super Tiny Language Models](https://arxiv.org/abs/2405.14159)** 研究专注于在不显著牺牲性能的情况下将语言模型参数减少 90-95%，这为更高效的 NLP 指明了道路。
- **无需猜测的 GPU 性能评估**：**无需执行**即可对 GPU 延迟进行符号建模的方法受到关注，相关的 [学术资源](https://inria.hal.science/hal-00789958/file/112_Lai.pdf) 为理论理解和对计算效率的潜在影响提供了指导。
- **与社区共同挑战现状**：讨论强调了社区驱动的项目以及在 Prompt 适配研究和实现查询（如 PyTorch 中的 **Facenet 模型**）等领域中**协作解决问题**的重要性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **最新模型创新投放市场**：[OpenRouter](https://openrouter.ai/models) 发布了新的 AI 模型，包括 **[Mistral 7B Instruct v0.3](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.3)** 和 **[Hermes 2 Pro - Llama-3 8B](https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b)**，同时保证之前版本如 **[Mistral 7B Instruct v0.2](https://openrouter.ai/models/mistralai/mistral-7b-instruct-v0.2)** 仍可访问。
- **对 Max Loh 网站模型的好奇**：用户对 [Max Loh 网站](https://www.maxloh.com) 上使用的模型表示好奇，并有兴趣识别 OpenRouter 上可用的所有无审查模型。
- **OCR 能力展示**：**Gemini 的 OCR** 能力成为热门话题，用户称其在读取西里尔字母和英文文本方面具有卓越能力，超越了 Claude 和 GPT-4o 等竞争模型。
- **OpenRouter Token 经济学**：社区澄清了在 OpenRouter 上 0.26 美元可获得 1M 输入 + 输出 Token，讨论强调了每次聊天交互如何重新计算 Token 使用量，这可能会增加成本。
- **尖端 Vision 模型的成本**：关于在 Azure 上使用 **Phi-3 Vision** 的成本展开了激烈讨论，一些成员认为 Llama 定价的 0.07 美元/M 太贵，尽管其他服务提供商也有类似的费率。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **翻译的苦恼**：讨论涉及了在控制歌词语调以保留原始艺术意图的情况下*翻译歌曲*的挑战。其独特的困难在于平衡意义的忠实度与音乐性和艺术表达。
- **AI 渗透 Greentext**：成员们尝试使用 LLM 生成 **4chan greentexts**，分享了他们对 AI 叙事能力的着迷——尤其是构思一个醒来后发现 AGI 已经实现的世界的场景。
- **哲学性的 Phi 与逻辑受限的 LLM**：围绕 **Phi 模型的训练数据** 构成展开了辩论，提到了“经过严格过滤的公开数据和合成数据”。此外，有报告显示 LLM 在交互过程中难以处理逻辑和自我修正，引发了对模型推理能力的担忧。
- **为机器消化塑造数据**：AI 爱好者交流了关于**创建 DPO 数据集**和调整数据集格式以进行 DPO 训练的资源和见解。Hugging Face 的 [TRL 文档](https://huggingface.co/docs/trl/main/en/reward_trainer) 和 [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) 成为关键参考，此外还有一篇详细介绍根据偏好数据训练语言模型的 [论文](https://arxiv.org/abs/2305.18290)。
- **为 RAG 财富连接思想**：协作氛围浓厚，成员们分享了在 RAG 相关项目上共同努力的意向。这包括 [GitHub](https://github.com/EveryOneIsGross/densefeelsCHAT) 上的情感和语义密度平滑 Agent 项目（带有 TTS），以及将现有项目移植到 SLURM 以增强计算管理的意图。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain 中的无限循环问题**：工程师们正在排查 **LangChain agent** 在调用工具时进入持续循环的问题；其中一个解决方案讨论涉及优化 Agent 的触发条件，以防止无限的工具调用循环。

**详情请看！LangChain 0.2.2 中的 16385-token 错误**：用户反馈 **LangChain 0.2.2 版本**中存在 token 限制错误，尽管模型支持高达 128k tokens，但系统却错误地应用了 16385-token 的限制，这引发了社区对该差异的调查。

**SQL Prompt 编写咨询**：关于带有 few-shot 示例的 **SQL agent** prompt 模板请求已得到解答，为工程师提供了在 LangChain 中更有效地构建查询的资源。

**消失的自定义参数：Langserve 中的 custom kwargs**：部分用户遇到通过 **Langserve** 发送用于 **Langsmith** 日志记录的自定义 "kwargs" 在到达时丢失的问题，该问题目前正在寻求解决方案。

**应用展示**：分享了使用 LangChain 开发的多样化应用，包括**药物研发 (drug discovery)** 框架、节省成本的日志记录措施、**飞行模拟器**增强功能，以及关于 Agent 流程中**路由逻辑 (routing logic)** 的教程。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 用户的 Python 版本警示**：提醒 Mojo 用户遵守支持的 Python 版本（**3.8 到 3.11**），因为 **3.12 仍不受支持**。通过使用 deadsnakes 仓库进行 Python 更新，解决了 Mojo 中的相关问题。

- **AI 驱动的游戏创新**：工程师们讨论了开放世界游戏中基于 NPC 智能的订阅模式前景，并为智能设备引入特殊的 AI 功能，这可能导致 AI 推理 (inference) 在本地运行。他们还探讨了可以实现 AI 驱动自定义世界生成的开放世界游戏。

- **Mojo 精通**：Mojo 允许循环依赖 (Circular dependencies)，因为模块可以相互定义。`Intable` 和 `Stringable` 等 Traits 是原生可用的。虽然 lambda 函数尚未成为 Mojo 的功能，但目前使用回调 (callbacks) 作为替代方案。

- **性能先锋**：在 Mojo 中，*32 字节时观察到显著的 50 倍速度提升*，但超过该长度后遇到了缓存限制。k-means 算法的基准测试显示出波动性，这是由于内存分配和矩阵计算的差异造成的，建议针对 AVX512 操作优化内存对齐。

- **Nightly 版本小结**：最新的 **Mojo 编译器版本 (2024.5.2805)** 带来了新功能，包括 `tempfile.{mkdtemp,gettempdir}` 和 `String.isspace()` 的实现，详细变更见[当前更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)和[原始差异 (raw diff)](https://github.com/modularml/mojo/compare/ce285fded710b403e1b7b5637183ea20fa4d5c97...4724ec6ff46378f6a1d6190ca9a76916a5faaba3)。通过引用实现的结构共享 (Structural sharing) 也因其在 Mojo 编程中潜在的效率提升而受到关注。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **调试功能升级**：工程师们称赞了 **cursor 解释器模式 (interpreter mode)**，强调其在调试场景中具有比传统搜索功能更先进的代码导航能力。

- **消息助推器**：**Microsoft Copilot** 集成到 **Telegram** 引起了关注，它能够通过游戏技巧和电影推荐等功能丰富聊天体验。

- **低成本训练 GPT-2**：**Andrej Karpathy** 展示了一种经济高效的方法，仅需 **20 美元即可在 90 分钟内**训练 GPT-2，并在 [GitHub](https://github.com/karpathy/llm.c/discussions/481) 上详细介绍了该过程。

- **Agents 和 Copilots 的角色区分**：在 **Microsoft Build** 进行分类后，关于 **Copilots** 和 **Agents** 之间的区别展开了辩论，并引用了 [Kanjun Qiu 对该话题的见解](https://www.latent.space/p/imbue)。

- **AI 播客交付前沿发现**：发布了一期[聚焦 ICLR 2024 的播客](https://x.com/latentspacepod/status/1795196817044594817)，讨论了 ImageGen、Transformers、Vision Learning 等领域的突破，并期待即将发布的关于 LLM Reasoning 和 Agents 的见解。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **金融极客们，尽情享用 FinTextQA**：[FinTextQA](https://t.co/emhQYXY1S4) 是一个旨在改进长文本金融相关问答系统的新数据集；它包含跨越 **6 种不同问题类型** 的 **1,262 个带有来源属性的问答对**。

- **完善 Prompt 结构**：有人咨询了关于构建最佳系统角色 Prompt 的资源，并从 **LlamaIndex** 的模型中汲取了灵感。

- **聊天历史保存策略**：社区讨论了在 **LlamaIndex** 中保存聊天历史的技术，考虑为 **NLSQL** 和 **PandasQuery** 引擎定制 Retriever，以维护查询和结果的记录。

- **API 函数管理探索**：针对拥有超过 1000 个函数的庞大 API 提出了管理策略，倾向于使用层级路由（hierarchical routing）并将函数划分为更易于管理的子组。

- **LlamaIndex 的 RAG 系统复杂性辩论**：剖析了 RAG 系统中与元数据相关的技术挑战，在为了获得最佳信息检索准确性而嵌入较小还是较大的语义分块（semantic chunks）方面，意见存在分歧。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI 的弦外之音**：成员们对 SOTA AGI 模型的古怪言论付诸一笑，其中一个模型的自我训练断言——“它为我们训练了一个模型”——戳中了大家的笑点。马斯克对 [CNNs](https://x.com/elonmusk/status/1795405972145418548) 的嘲讽——调侃道“我们这些天不怎么使用 CNN 了”——引发了一连串的反讽回复，并对作为行业新宠的 Vision Transformer 模型表示了认可。

**人工智能艺术家的水印烦恼**：[Corcelio 的 Mobius 艺术模型](https://huggingface.co/Corcelio/mobius) 正在通过多样化的 Prompt 挑战极限，但尽管它在创造力上超越了以往的模型，却仍会留下水印。图像生成系统产生“不当”内容的能力引发了伦理困境，触发了关于社区准则和系统控制设置的辩论。

**合成视觉寻求改进**：为了解决 **SDXL** 无法生成“正在阅读的眼睛”图像的问题，一名成员请求协作帮助，利用 DALLE 构建一个合成数据库，希望在这一细微的视觉任务中磨练 **SDXL** 的能力。

**生成式水印中的模式与谜题**：公会内部的观察指出，生成模型产生水印是一个反复出现的主题，这表明可能存在训练不足（undertraining），这在工程师中既被认为有趣又值得关注。

**埃隆对 CNN 的白眼引发 AI 调侃**：埃隆·马斯克的推文在社区中引起了涟漪，引发了关于 CNN 在当今变革性 AI 方法论中已过时的笑话，以及可能向 Transformer 模型转型的讨论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**无需基准测试的 GPU 延迟预测？**：工程师们讨论了在不运行 Kernel 的情况下，通过考虑数据移动和操作时间来**对 GPU 延迟进行符号化建模**的可能性，尽管占用率（occupancy）和异步操作等复杂性被认为是潜在的干扰因素。人们还期待 AMD 开源 MES，并推测量化交易公司正在使用周期精确（cycle accurate）的 GPU 模拟器进行深入的 Kernel 优化。

**使用 Autotuner 进行优化**：社区探索了 **AutoTVM** 和 **Halide** 等 Kernel 优化工具，注意到它们在性能改进方面的不同方法；George Hotz 强调了 TVM 对 XGBoost 的使用，并强调了缓存仿真（cache emulation）对于准确建模的重要性。

**GPU 中的延迟隐藏机制**：有人指出，GPU 利用运行并发 Wavefronts/Blocks 的能力采用了多种延迟隐藏策略，从而使延迟建模变得更加复杂和细微。

**Tinygrad 中的 Buffer 创建讨论**：#learn-tinygrad 频道有成员询问在调度中使用**后支配者分析（post dominator analysis）**以提高图融合（graph fusion）效率，以及从数组创建 **LazyBuffer** 的问题，并建议在此类场景中使用 `Load.EMPTY -> Load.COPY`。

**代码清晰度与协助**：针对 Tinygrad 中的 Buffer 分配和 `LazyBuffer` 创建进行了详细讨论，一名成员提出提供**代码指针（code pointers）**以进一步澄清和理解。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Elevenlabs 语音进入 AI Town**：通过集成 **Elevenlabs** 的文本转语音（text-to-speech）功能，AI Town 推出了一项新特性，让对话不仅能被阅读，还能被听到。目前约有一秒的轻微延迟，这对实时使用构成了挑战。实现过程涉及[将文本转换为音频](https://github.com/huevosabio/ai-town/blob/e7e2182eb7f7241e58c69d8324ae126c1d34dee9/convex/util/textToSpeech.ts#L19)以及在前端管理音频播放。

- **将科学辩论引入 AI 聊天**：分享了一个利用 AI 聊天机器人模拟科学辩论的概念，旨在促进用户参与并展示科学讨论的统一性。

- **新增音频窃听功能以增强沉浸感**：AI Town 的 Zaranova 分支现在通过为环境对话生成音频来模拟“窃听”，这可能会增强平台的互动性。

- **协作开发集结**：社区对贡献并可能将新功能（如文本转语音）合并到 AI Town 主项目表现出浓厚兴趣。

- **解决用户体验问题**：有用户反映对话关闭速度过快，导致无法舒适阅读，这暗示 AI Town 需要在用户界面和无障碍体验方面进行改进。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **精简日志**：一名成员开发了新的流水线来移除**冗余日志**以降低成本。他们推荐使用一个[工具](https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)来选择“详细日志（verbose logs）”流水线以实现此目标。

- **讨论部署方案**：成员们讨论了用于 **reranking 和查询提取（query extraction）** 的云端/本地部署方案，在没有提供更多背景的情况下寻求最佳集成实践的见解。

- **金融 RAG 微调**：有人咨询了**微调 Cohere 模型**以回答金融问题的可能性，特别提到了使用 SEC 备案文件与 **RAG（检索与生成）系统**的集成。

- **Aya23 模型的限制性使用**：会议澄清了 **Aya23 模型**严格用于研究目的，不可用于商业用途，这影响了它们在初创公司环境中的部署。

- **机器人玩游戏**：一名成员推出了由 **Cohere Command R** 驱动的游戏机器人 **Create 'n' Play**，其特点是拥有“超过 100 款基于文本的游戏”，旨在促进 Discord 上的社交互动。该项目的开发情况和目的可以在这篇 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios)中找到。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **推理与训练的现实**：对话强调了 AI 训练中的性能数据，特别是关于“仅推理（inference only）”话题的简单查询如何迅速演变为关注训练计算需求的复杂领域。

- **FLOPS 决定训练速度**：讨论中的一个关键点是，AI 模型训练在实践中受限于每秒浮点运算次数（FLOPS），特别是在采用 **teacher forcing** 等技术增加有效 **batch size** 时。

- **期待用于 FP8 的 Hopper 显卡**：社区对 **Hopper** 显卡在 FP8 原生训练方面的潜力表现出极大热情，凸显了利用尖端硬件提升训练吞吐量的浓厚兴趣。

- **消除 fschat 的版本混淆**：由于错误的版本标识，成员们被建议通过重新安装来解决 **fschat** 问题，这体现了该集体生态系统中对细节的严谨关注。

- **当 CUTLASS 更胜一筹时**：讨论明确了设置 `CUTLASS_PATH` 的重要性，强调了 **CUTLASS** 在优化深度学习至关重要的矩阵运算中的作用，突出了该组织对优化算法效率的关注。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apache 欢迎 YI 和 YI-VL 模型**：**YI 和 YI-VL（多模态 LLM）模型**现在采用 **Apache 2.0** 许可证，正如 [@_philschmid 的推文](https://fxtwitter.com/_philschmid/status/1795343334225129570)所庆祝的那样；它们在这次许可更新中加入了 1.5 系列。

- **Gemini 1.5 挑战王座**：**Gemini 1.5 Pro/Advanced** 已攀升至排行榜第 2 位，并有超越 GPT-4o 的野心，而 **Gemini 1.5 Flash** 则自豪地占据了第 9 位，险胜 **Llama-3-70b**，正如 [lmsysorg 的推文](https://x.com/lmsysorg/status/1795512202465845686?s=46)所宣布的那样。

- **OpenAI 董事会蒙在鼓里**：一位前 OpenAI 董事会成员透露，董事会事先并未获悉 **ChatGPT** 的发布，而是像公众一样通过 [Twitter](https://fxtwitter.com/bilawalsidhu/status/1795534345345618298) 才得知消息。

- **Toner 对 OpenAI 领导层投下重磅炸弹**：OpenAI 前董事会成员 Helen Toner 在 [TED 播客节目](https://dts.podtrac.com/redirect.mp3/chtbl.com/track/48D18/dovetail.prxu.org/6792/49695742-c50c-4a16-83ba-407f75b3f301/TED_AI_E02_Helen_Toner_Seg_A_-_YES_COMMENT_2024-05-28.mp3)中指责 **Sam Altman** 营造了有毒的工作环境且行为不诚实，并敦促“对 AI 公司进行外部监管”。

- **社区对 OpenAI 的爆料感到震惊**：针对 Helen Toner 的严重指控，社区表达了震惊，并对行业可能发生重大变革的前景充满期待，Natolambert 甚至发问 Toner 是否会“从字面意义上拯救世界？”

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **专家认可的常用 LLM 排行榜**：[chat.lmsys.org 上的排行榜](https://chat.lmsys.org/?leaderboard)受到了用户的关注和认可，被认为是比较各种大语言模型（LLM）性能的可靠资源。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **保护本地 AI 端点至关重要**：一位成员**强调了保护 AI 模型本地端点的重要性**，建议使用 **DNS SRV 记录**和公钥来确保经过验证且值得信赖的本地 AI 交互，并开玩笑说未经证实的模型可能会导致意外购买乡村音乐或喂松鼠的风险。
- **故障排除警报：发现 Llamafile 错误**：一位运行 **Hugging Face llamafile**（具体为 `granite-34b-code-instruct.llamafile`）的用户报告了一个“未知参数：--temp”的错误，这表明模型部署过程的实施阶段可能存在问题。
- **关注正在运行的模型**：在一次澄清中指出，无论本地 `localhost:8080` 运行的是什么模型（如 *tinyllama*），它都将是默认模型，聊天补全请求中的 `model` 字段对操作无关紧要。这表明所使用的 **llamafiles** 采用的是**单模型运行范式**。
  
**提到的链接**：[granite-34b-code-instruct.llamafile](https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile/resolve/main/granite-34b-code-instruct.Q5_0.llamafile?download=true)

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **请求 R1 更新**：一位成员表达了对 **R1** 未来发展的期待，并幽默地提到如果它达不到预期，可能会变成一个“漂亮的镇纸”。
- **社区寻求明确答复**：社区内对 **R1** 相关的更新有着共同的好奇心，成员们正在积极寻求和分享信息。
- **等待支持团队的关注**：向 **OI 团队**咨询的一封电子邮件正在等待回复，这表明需要改进沟通或支持机制。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **发现一座“鬼城”**：一位成员提出担忧，认为该服务器似乎**无人管理**，这可能意味着管理员的疏忽，或者是故意采取的放任自流的态度。
- **通知未能发出**：在服务器中尝试使用 **@everyone** 标签失败，这表明权限受限或存在技术故障。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LLM 用于后端自动化的咨询未获回应**：一位成员好奇课程是否涵盖使用 Large Language Models (LLM) 自动化后端服务的内容，但该问题尚未得到解答。该咨询旨在寻求有关 LLMs 在自动化后端流程中实际应用的见解。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


---


**YAIG (a16z Infra) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。


> 完整的各频道详细分析已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}