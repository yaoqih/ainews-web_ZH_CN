---
companies:
- primeintellect
- bytedance
- qwen
- gemma
- meta-ai-fair
- runwayml
- mistral-ai
- google
date: '2025-05-12T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Prime Intellect** 发布了 **INTELLECT-2**，这是一个去中心化的 GPU 训练和强化学习（RL）框架，其愿景是通过分布式 AI
  训练克服地理位置限制（colocation limits）。**字节跳动 (ByteDance)** 在 Hugging Face 上推出了统一图像定制模型 **DreamO**。**通义千问
  (Qwen)** 发布了针对 GPTQ、GGUF 和 AWQ 量化优化的模型。**Gemma** 在 Hugging Face 上的下载量突破了 1.5 亿次。**Meta**
  发布了 **Dynamic Byte Latent Transformer** 和 **Collaborative Reasoner** 框架的权重，旨在提升语言模型的效率和推理能力。**RunwayML**
  推出了 **Gen-4 References**，这是一款无需微调的近实时模型。**Mistral AI** 发布了强大的多模态模型 **Mistral Medium
  3**，以及面向企业的智能体 AI 助手 **Le Chat Enterprise**。**谷歌 (Google)** 更新了 **Gemini 2.5 Pro
  预览版**，改进了视频理解能力和 UI 界面。“**全球闲置 GPU 的 Airbnb**”这一说法突显了分布式 GPU 训练所面临的持续挑战与巨大潜力。'
id: MjAyNS0w
models:
- intellect-2
- dreamo
- qwen
- gemini-2.5-pro
- dynamic-byte-latent-transformer
- gen-4-references
- mistral-medium-3
- le-chat-enterprise
people:
- _akhaliq
- reach_vb
- osanseviero
- aiatmeta
- c_valenzuelab
- lmarena_ai
- adcock_brett
title: Prime Intellect 的 INTELLECT-2 和 PRIME-RL 推动了分布式强化学习的发展。
topics:
- distributed-training
- reinforcement-learning
- gpu-clusters
- model-optimization
- quantization
- multimodality
- agentic-ai
- video-understanding
- fine-tuning
---

**分布式 GPU 就是你所需的一切吗？**

> 2025年5月9日至5月12日的 AI 新闻。我们为您查看了 9 个 subreddit、449 个 Twitter 账号和 29 个 Discord 社区（215 个频道，12925 条消息）。预计节省阅读时间（以每分钟 200 字计）：1292 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe coded 呈现方式，涵盖所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

**梦想**：“全球闲置 GPU 的 Airbnb”

**现实**：GPU 的托管（colocation）变得如此重要，以至于对[万亿美元集群](https://situational-awareness.ai/)的呼吁实际上已经[具体化](https://news.smol.ai/issues/25-01-21-ainews-project-stargate-dollar500b-datacenter-17percent-of-us-gdp-and-gemini-2-flash-thinking-2)了。

在这个加速进步的时代，乐观主义者的陷阱在于那些承诺远超实际现实的领域，尤其是当现实遇到光速等硬性限制时。通常很难知道在众多的“联邦学习（federated learning）”或“分布式训练（distributed training）”尝试中，哪些能坚持足够长的时间并真正获得关注。由于这些原因（以及缺乏理解这一更简单的原因），尽管社区反响热烈，我们到目前为止一直避免报道类似的尝试，例如 [Nous Research](https://news.smol.ai/issues/24-01-10-ainews-192024-nous-research-lands-dollar5m-for-open-source-ai) 在 [DisTrO](https://github.com/NousResearch/DisTrO) 上的工作。此外，由于 AI Engineer 的关注点非常倾向于推理（inference），给定模型是在哪个 GPU 集群上训练的其实并不重要，这进一步限制了业界的实际兴趣。

然而，Prime Intellect 的工作感觉有些不同。

[INTELLECT-2 的发布](https://www.primeintellect.ai/blog/intellect-2-release)不仅仅是一篇[论文](https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf#page=11.25)，或者一个 [QwQ finetune](https://huggingface.co/collections/PrimeIntellect/intellect-2-68205b03343a82eabc802dc2)，或者一个 [RL framework](https://github.com/PrimeIntellect-ai/prime-rl)，或者晦涩的区块链技术，又或者是另一个 GRPO 变体。它是这一切的结合，甚至更多——一个概念验证，一份愿景声明，或许也是对为什么去中心化在默认中心化的 AI 世界中占有一席之地的初步阐述：


![image](https://resend-attachments.s3.amazonaws.com/Wj85oPCARIuY0Xu)


模型训练者应该关注 [Prime-RL,](https://github.com/PrimeIntellect-ai/prime-rl) 但[论文](https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf#page=11.25)中也包含了一些关于后训练（post-training）领域中非常有效的技术前沿的有趣见解：


![image](https://resend-attachments.s3.amazonaws.com/KrBKbZMvCf7pMlj)


以及训练期间推理（inference-during-training，他们正确地观察到这在 RL 时代将会有很大的扩展）：


![image](https://resend-attachments.s3.amazonaws.com/BLqeVRfMeNPQehi)


---

# AI Twitter 综述

**AI 模型发布与更新**

- **字节跳动在 Hugging Face 上发布了 DreamO**，这是一个统一的图像定制框架，通过单一的轻量级高性能模型支持 ID、IP、Try-On 和 Style 任务 [@_akhaliq](https://twitter.com/_akhaliq/status/1921948350145815010)。
- **针对 GPTQ、GGUF 和 AWQ 优化的 Qwen 模型**已由 Qwen 发布 [@reach_vb](https://twitter.com/reach_vb/status/1921956656226668964)。
- **Gemma** 在 Hugging Face 上的下载量已突破 1.5 亿次，拥有 7 万个变体，[@osanseviero](https://twitter.com/osanseviero/status/1921636582873800746) 正在征求社区对未来版本的建议。
- **Meta** 发布了其 8B 参数 **Dynamic Byte Latent Transformer** 的模型权重，旨在提升语言模型的效率和可靠性；同时发布了 **Collaborative Reasoner** 框架，旨在增强语言模型的协作推理能力 [@AIatMeta](https://twitter.com/AIatMeta/status/1921978043998077011) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1921966366707613924)。
- **RunwayML** 的 Gen-4 References 模型拥有无限的工作流且无需微调，根据 [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1921583557333389637) 和 [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1921356668027249126) 的说法，它是一个近乎实时的万物生成机器。
- **Mistral AI** 发布了多模态 AI 模型 **Mistral Medium 3**，其表现力可与闭源模型媲美；此外还发布了 **Le Chat Enterprise**，这是一款面向企业的 Agentic AI 助手，配备了 Google Drive 和 Agent 构建等工具 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921667566767845770) 和 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597108567617585)。
- **Google** 更新了 **Gemini 2.5 Pro Preview**，增强了视频理解能力，并改进了 UI、代码和 Agentic 工作流；同时更新了 **Gemini 2.0 Flash** 的图像生成功能，提升了质量和文本渲染效果 [@adcock_brett](https://twitter.com/adcock_brett/status/1921596995371765866)。
- 作为中国开源 AI 运动的一部分，**DeepSeek** 在两年内几乎缩小了与美国同行的差距 [@hardmaru](https://twitter.com/hardmaru/status/1921374572131254516)。
- **阿里巴巴 Qwen** 正式发布了 **Qwen3** 的量化版本，可以通过 Ollama、LM Studio、SGLang 和 vLLM 以 GGUF、AWQ 和 GPTQ 等多种格式部署 [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1921907010855125019)。
- **f-lite 7B**，一个来自 f-lite 的蒸馏扩散模型已发布 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1921931992200847479)。
- **微软**更新了其 **Copilot**，增加了 "Pages" 功能（类似于 ChatGPT Canvas），但似乎不像 Canvas 那样具备编程能力 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597040905097496)。
- **微软**还宣布将采用 Google 的 Agent2Agent (A2A) 框架，并很快在 Azure AI Foundry 和 Copilot Studio 上推出 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597063478817247)。
- **Hugging Face** 发布了 **Open Computer Agent**，这是一个用于自动化 Web 任务的开源 AI Agent，但据报道其运行速度较慢，且只能处理基础的多步任务 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597198510297124)。

**AI 工程与工具**

- **AI 辅助提交信息 (Commit Messages)、Windsurf & Cursor Rules 的 UI、更新后的自动批准 UI 以及批量历史记录删除**是 Cline v3.15 中的一些易用性更新，据 [@cline](https://twitter.com/cline/status/1921360242501431364) 报道。
- Cline v3.15 集成了 Google 新的 Gemini Implicit Caching，可大幅降低 **Gemini 2.5 Pro 的成本**，在重复的 prompt 部分自动获得高达 75% 的 token 折扣 [@cline](https://twitter.com/cline/status/1921359984434246034)。
- **Dolphin-MCP** 进行了重大更新，这是一个开源的 MCP 客户端，允许你将 MCP 与任何本地或云端 AI 模型配合使用 [@cognitivecompai](https://twitter.com/cognitivecompai/status/1921417366111482094)。
- **Anthropic** 在 API 中发布了网页搜索功能，允许开发者构建可以搜索网页并提供带有相关引用的可靠答案的应用程序 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597220928794937)。
- [@skirano](https://twitter.com/skirano/status/1921334962097127639) 创建了一个使用 **Anthropic 新网页搜索工具**的 MCP 服务器，它提供了 Agentic 搜索能力，任何模型都可以调用 Claude 实例来返回处理后的搜索结果。
- **将深度研究报告导出为格式精美的 PDF**，包含表格、图像、链接引用和来源。据 [@OpenAI](https://twitter.com/OpenAI/status/1921998278628901322) 称，该功能已面向所有 Plus、Team 和 Pro 用户开放，Enterprise 和 Edu 版本即将推出。
- 看看这个 **AI 研究 Agent 教程**，它使用 LangGraph 和 Ollama 搜索网页并生成带有引用的摘要，据 [@LangChainAI](https://twitter.com/LangChainAI/status/1921626371559698666) 报道。
- **Microsoft** 为 ChatGPT 推出了 GitHub 连接器，允许用户连接他们的仓库，并利用 ChatGPT 的 Deep Research 来读取和搜索源代码及 PR，从而创建带有引用的详细报告 [@adcock_brett](https://twitter.com/adcock_brett/status/1921596972735111576)。

**基于 Agent 的系统和多 Agent 系统**

- **Langchain** 重点介绍了几个基于 Agent 的系统和工具包示例，例如这个公司研究员 [@LangChainAI](https://twitter.com/LangChainAI/status/1921611360548389145)，以及这个通过协调 LangGraph Agent 进行系统深度研究的深度研究框架 [@LangChainAI](https://twitter.com/LangChainAI/status/1921596224186077352)。
- **The Turing Post** 分享了对**多 Agent 系统 (MAS)** 的深入探讨，详细介绍了它们的架构、类型、最新进展和当前趋势 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1921350723813683406)。
- 由前 Google CEO Eric Schmidt 支持的 **FutureHouse** 发布了五个“AI Scientist” Agent，用于研究、化学工作流和生物学发现 [@adcock_brett](https://twitter.com/adcock_brett/status/1921597086002287090)。

**LLM 评估与基准测试**

- **OpenAI** 推出了 HealthBench，这是一个新的评估基准，是在全球 250 多名医生的参与下开发的，现已在其 GitHub 仓库中提供 [@OpenAI](https://twitter.com/OpenAI/status/1921983050138718531)。
- 最新的模型（Gemini 2.5 Pro, GPT-4.1）在文档解析方面非常出色，传统的 OCR 已经过时。据 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1921621794265665749) 称，目前仍然需要人工审核和修正。
- **lmarena_ai** 指出，**腾讯最新的 Hunyuan-Turbos** 目前排名第 8，较其 2 月份的版本有显著提升 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1921966648795533459)。
- **METR_Evals** 的“每约 7 个月翻一番”幻灯片目前几乎出现在每一场 AI 进展演讲中。据 [@polynoamial](https://twitter.com/polynoamial/status/1921618587690893476) 称，这是一个引人注目的趋势，但值得明确测量对象：独立的代码和 ML 任务。

**核心观点与研究方向**

- **Karpathy** 认为我们至少遗漏了一个 LLM 学习的主要范式。他称之为“system prompt learning”（系统提示词学习），并认为其在设置上类似于 RL，除了学习算法（编辑 vs 梯度下降）不同。LLM 系统提示词的大部分内容可以通过 system prompt learning 来编写，这看起来有点像 LLM 在为自己写一本关于如何解决问题的书 [@karpathy](https://twitter.com/karpathy/status/1921368644069765486)。
- **DanHendrycks** 分享道，AI 模型在 IQ 测试中的表现正在提高（从 70 IQ 提升到 120），但它们感觉起来并不比两年前聪明多少。他认为，有用的原创性只有在高智力水平下才会陡峭上升，因此需要持续的进步才能让 AI 产生原创见解 [@DanHendrycks](https://twitter.com/DanHendrycks/status/1921429850432405827)。
- **Sakana AI** 推出了 **Continuous Thought Machines**，这是一种全新的神经架构，旨在利用神经动力学作为智能的核心表示，从而实现自适应计算和有趣的涌现行为 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1921749814829871522)。

**学术界与论文**

- **Neel Nanda** 分享了一份关于如何以科学严谨性将研究转化为高质量 ML 论文的指南 [@NeelNanda5](https://twitter.com/NeelNanda5/status/1921928364790833651)。
- **TheAITimeline** 总结了本周顶尖的 AI/ML 研究论文，包括 Absolute Zero、RM-R1、Seed-Coder、Flow-GRPO 等，并提供了概述和作者解释 [@TheAITimeline](https://twitter.com/TheAITimeline/status/1921626740675248338)。
- **dair_ai** 列出了本周的热门 AI 论文，其中包括 ZeroSearch、Discuss-RAG、Absolute Zero、Llama-Nemotron、The Leaderboard Illusion 以及 Reward Modeling as Reasoning [@dair_ai](https://twitter.com/dair_ai/status/1921606662214787114)。

**视觉语言模型 (VLMs)**

- **Phil Schmid** 指出，使用 Gemini 2.5 Pro (05-06) 进行视频理解正在改变我们处理视频的方式，它允许在 200 万 context 中以“低分辨率”处理长达 6 小时的视频，并将视听理解与代码相结合 [@_philschmid](https://twitter.com/_philschmid/status/1921838835735867533)。
- **Merve Noyan** 指出 Llama.cpp 现在已经支持视觉语言模型，包括对 Gemma 3、Qwen2.5VL、InternVL3 等的支持 [@mervenoyann](https://twitter.com/mervenoyann/status/1921471242852331719)。

**职业与行业**

- **Swyx** 回顾了 Greg Brockman 的职业生涯，并提出了一个问题：“你会问 @gdb 什么样的问题，从而能显著影响你的所作所为或信念？” [@swyx](https://twitter.com/swyx/status/1921992616448831754)。
- **Cartesia** 正在班加罗尔组建其印度团队，招聘具有 ML 系统经验的 SWE，据 [@krandiash](https://twitter.com/krandiash/status/1922016592621404407) 称。
- **Epoch AI** 正在招聘一名 Web 开发负责人，以帮助传播他们的研究成果并管理工程师团队，此消息由 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1921987268337693106) 分享。

**幽默/迷因**

- **Karpathy** 分享了关于智力工作和评分之艰辛的共鸣感悟，用幽默与观众建立联系 [@karpathy](https://twitter.com/karpathy/status/1921402746902560857)。
- **DanHendrycks** 开玩笑地解释了为什么 AI 目前还讲不出好笑的笑话 [@DanHendrycks](https://twitter.com/DanHendrycks/status/1921433380974948727)。
- **Agihippo** 幽默地感叹 AI 研究人员从未脱离工作，甚至在婚礼上也是如此 [@agihippo](https://twitter.com/agihippo/status/1921589434488586731)。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. 主要 LLM 和 Transformer 模型发布 (Qwen3, INTELLECT-2, Meta 8B BLT)

- [**Qwen 发布 Qwen3 官方量化模型**](https://i.redd.it/ok2e3kp5jc0f1.jpeg) ([Score: 873, Comments: 90](https://www.reddit.com/r/LocalLLaMA/comments/1kkrgyl/qwen_releases_official_quantized_models_of_qwen3/)): **该图片总结了 Qwen3 量化模型的官方发布，目前已提供多种格式（GGUF、AWQ、GPTQ），以便在 Ollama、LM Studio、SGLang 和 vLLM 等开源平台中无缝部署。它详细展示了各种 Qwen3 模型架构，包括 Mixture of Experts (MoE) 和 Dense 版本，并强调了量化精度（BF16、FP8、Int4），具有用户可配置的生成和模式切换功能，便于灵活的本地推理。Hugging Face 发布页面列出了所有官方 Qwen3 量化 Checkpoint 供社区使用 ([Hugging Face Qwen3 Collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f))。** 评论者们热切期待看到 Qwen 的 GGUF 量化版本与其他来源（如 Unsloth）的对比技术基准测试，特别是在长序列长度（128k）下，并对 Qwen 全面的发布策略表示赞赏，将其与 Meta 之前的发布进行了正面对比。

- 社区对将官方 Qwen3 量化模型与来自社区的其他量化版本（特别是 unsloth 128k GGUF 版本）进行基准测试表现出技术兴趣。用户渴望看到这些不同量化版本在相对性能、效率或准确性方面的实证结果。
- 一位评论者强调，Qwen3 的发布因提供官方量化模型（GGUF 以及 AWQ、GPTQ 和 INT8）、开放权重、宽松的许可证以及为集成开源工具所做的预发布准备而脱颖而出——这与 Meta 处理 Llama 发布的方式形成了鲜明对比。
- 有一个关于 Qwen 是否计划在未来发布 QAT (Quantization Aware Training) 模型的问题，与训练后量化相比，QAT 能够实现更高效、高质量的量化模型。
- [**INTELLECT-2 发布：首个通过全球分布式强化学习训练的 32B 参数模型**](https://huggingface.co/PrimeIntellect/INTELLECT-2) ([Score: 434, Comments: 58](https://www.reddit.com/r/LocalLLaMA/comments/1kkgzip/intellect2_released_the_first_32b_parameter_model/))：**INTELLECT-2 是一个 32B 参数的语言模型，使用 QwQ-32B 基座模型和自定义的 prime-rl 异步分布式 RL 框架，通过分布式全球强化学习训练而成。它在数学和编程任务上利用了可验证的奖励信号，并进行了架构更改以实现稳定性和自适应长度控制（最佳生成长度为 2k–10k tokens）；详细的基准测试显示，它在 AIME24、LiveCodeBench 和 GPQA-Diamond 上略微优于 QwQ-32B，但在 IFEval 上表现稍逊。该模型支持通过 vllm/sglang 进行高效的分布式推理，训练利用了全球无许可的 GPU 池（更多信息见 [技术报告](https://www.primeintellect.ai/intellect-2)）。** 讨论强调，基准测试的差异在误差范围内，这让人怀疑其相对于 QwQ-32B 的实际改进，并强调了去中心化 RL 训练方法的意义，而非纯粹的性能提升；此外，评论者看到了受区块链/P2P 启发的分布式计算和信用系统在推理和训练方面的潜力。
    - INTELLECT-2 基于 QwQ 32B 架构，并取得了基准测试结果（例如 AIME24：78.8 vs 76.6；LiveCodeBench：67.8 vs 66.1），这些结果在 QwQ-32B 的误差范围内，表明其在特定数据集之外的泛化能力有限。然而，其意义在于去中心化 RL 训练方法（全球分布式强化学习），而不仅仅是边际性能的提升。
    - INTELLECT-2 背后的分布式 RL 训练方法能够利用大规模、全球分布的计算资源扩展到 320 亿参数——证明了去中心化 AI 训练方法的技术可行性和潜力。社区讨论还指出，这可能适用于分布式推理系统、P2P 或受区块链启发的网络，并可能结合信用/奖励系统来激励计算贡献。
    - 博客和技术报告（由用户链接）中提供的基准测试证实了关于 INTELLECT-2 性能的说法，并重申了该发布的技术重点：将大模型规模与去中心化、协作式强化学习相结合，作为一种新颖的系统贡献，而不仅仅是提高标准任务的性能。
- [**Meta 发布了 8B BLT 模型**](https://ai.meta.com/blog/meta-fair-updates-perception-localization-reasoning/?utm_source=twitter&utm_medium=organic%20social&utm_content=video&utm_campaign=fair) ([Score: 108, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1kky1sg/meta_has_released_an_8b_blt_model/))：**Meta 发布了一个 8B 参数的 Byte Latent Transformer (BLT) 模型，该模型专注于字节级 tokenization，以提高多语言和多模态性能。该模型最初在 2023 年底讨论过（[Meta BLT 论文](https://www.reddit.com/r/LocalLLaMA/comments/1hdpw14/metas_byte_latent_transformer_blt_paper_looks/)），当时的基准测试表明其在高效字节级处理方面的优势。自 2023 年发布以来，没有出现新的重大技术细节或基准测试。** 评论者指出 BLT 的发布并非近期之事，并表达了对更高参数数量的 Llama 4 系列的迫切需求。此外，人们对 BLT 与现有 Llama 模型相比的实际影响持怀疑态度。
    - 对更大且更多样化的 Llama 4 模型尺寸（如 32B、11B、8B）的需求仍在持续，这表明社区正在寻求超出 Meta 目前发布范围的性能可扩展性和灵活性，一些人表示 Meta 在 Llama 4 之后的表现落后于预期。

- 多位评论者指出 BLT (Byte Latent Transformer) 并非新发布的产品，并提到了去年和上个月关于 BLT 以及 Meta 感知模型的讨论。目前尚不清楚与之前的发布相比，在实际改进或创新方面有哪些具体表现。
- Evabyte (6.5B) 作为一个开源的、基于 byte 的模型，被引用为 byte-level 架构的先例，人们质疑 Meta 的 BLT 除了扩展到 8B 参数之外还有什么不同。与其他 8B 模型相比的技术差异和性能对比仍在讨论中，用户对显著的进步持怀疑态度。

### 2. 微软用于 Agentic 工具增强型 LLM 的 ARTIST 框架

- [**微软研究人员推出 ARTIST**](https://i.redd.it/90acs85p7c0f1.png) ([Score: 212, Comments: 25](https://www.reddit.com/r/LocalLLaMA/comments/1kkq8q8/microsoft_researchers_introduce_artist/)): **提供的图像是一个表格，总结了在具有挑战性的数学推理基准测试（MATH-500、AIME、AMC 和 Olympiad）上，使用 ARTIST 框架增强后的各种模型（特别是 Qwen2.5-7B-Instruct 和 Qwen2.5-14B-Instruct）的性能。该表格表明，集成 ARTIST 后，所有基准测试的 pass@1 准确率都有持续且显著的提高，Qwen2.5 模型超越了 GPT-4o 等更大的模型，有时增幅高达 22%。该技术方法结合了 agentic reasoning、动态工具使用（包括** `<think>` **和** `<tool_name>` **动作等功能）以及 reinforcement learning（特别是使用 GRPO 和定制的 reward functions），促进了稳健、可解释的多步推理。** 几条评论注意到了所选的强大基准测试以及 web search 等工具的使用，承认工具集成能够可靠地提高分数，但由于所使用的基准测试，对现实世界的泛化能力表示怀疑。还有关于 ARTIST 的 RL 方法（带有 loss masking 和结构化 rewards）如何实现涌现的 agentic behaviors（例如 self-correction 和 self-reflection）的讨论，一些用户注意到了相对于 baseline 或 distilled 模型的实际提升。
    - 几条评论强调，ARTIST 的基准测试设置显示 7B 和 14B 模型的表现出人意料地接近 GPT-4o，但有人指出这可能反映了基准测试的特征而非通用能力，特别是由于它包含了工具使用（例如 web search），且不是像 GPQA 这样难以通过搜索引擎找到答案（Google-proof）的数据集。观点认为，集成 agentic reasoning 和工具调用可以预见地会提高分数，因此在解释这些结果时必须谨慎。
    - 一份详细的分析描述了 ARTIST 如何结合 agentic reasoning、工具集成和使用 GRPO (Group Relative Policy Optimization) 方法的 reinforcement learning (RL)。技术亮点包括 loss masking RL 策略，旨在将学习重点放在 LLM 的推理/动作上，而不是复制确定性的工具输出；以及一个复合 reward 系统——包括 answer、format、tool execution/state/function rewards——以引导正确且可解释的分步问题解决行为。
    - 实验结果表明，在数学推理基准测试（AMC、AIME、Olympiad）中，ARTIST 比基础模型实现了高达 `22%` 的绝对提升，在 τ-bench 的多轮函数调用准确率上提高了一倍以上。这些收益归功于模型设计和训练中产生的涌现 agentic behaviors（self-refinement、self-correction、self-reflection），而不是来自人工监督。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. 近期模型和功能发布 (Manus AI, JoyCaption, Continuous Thought Machines)

- [**Manus AI 已正式公开发布**](https://i.redd.it/2179eel6le0f1.jpeg) ([Score: 162, Comments: 56](https://www.reddit.com/r/singularity/comments/1kl1q2q/manus_ai_has_officially_launched_publicly/)): **图片显示了来自 ManusAI 的官方公告，表明他们已公开发布并取消了 waitlist。所有用户现在每天可以获得一个免费任务和额外的 1,000 个 credits。截图展示了一个用户界面，用户可以分配诸如“动画问题教程”和“交互式学习网站”之类的任务，这表明 ManusAI 专注于教育或内容生成任务。更多信息可以在他们的推文 [这里](https://x.com/ManusAI_HQ/status/1921943525261742203) 和图片 [这里](https://i.redd.it/2179eel6le0f1.jpeg) 查看。** 评论者要求澄清 ManusAI 是什么；一位批评者指责其目前的能力和相对于 Claude 等替代方案的高昂费用，称“它的开发程度还不足以实现所有承诺的功能”。

- 一些用户报告 Manus AI *并非在所有地区都可用*，这可能是由于地理屏蔽或早期发布限制，这阻碍了特定市场之外的深入评估或采用。
- 用户将其与 Claude 等成熟模型在代码相关能力方面进行了比较。一条评论指出 Manus AI *并未给人留下深刻印象*，且既*昂贵*又*不够成熟，无法兑现其承诺的所有功能*，这表明在当前状态下，其实现和性价比可能落后于竞争对手。
- [**JoyCaption: 免费、开源、无审核的 VLM (Beta One 发布)**](https://www.reddit.com/r/StableDiffusion/comments/1kl2nek/joycaption_free_open_uncensored_vlm_beta_one/) ([Score: 272, Comments: 44](https://www.reddit.com/r/StableDiffusion/comments/1kl2nek/joycaption_free_open_uncensored_vlm_beta_one/)): **JoyCaption Beta One 是一款免费、开源且无审核的视觉语言模型 (VLM)，用于图像打标，旨在为训练扩散模型提供实用性。相比 Alpha Two 的主要技术进步包括：训练数据翻倍（总计 240 万样本）、新增用于更简洁描述的“直观模式 (Straightforward Mode)”、重构并稳定了 booru 标签生成（按字母和类别分组，通过改进格式和 DPO 强化学习减少重复）、更准确的水印注释（通过自定义水印检测 ML）、手动编写了 2000 多个用于指令遵循的 VQA 对，以及支持用户指定的标签增强。该模型经过了两轮直接偏好优化 (DPO)（1 万和 2 万个偏好对），根据人类和 SOTA VLM 评估，在输出偏好和减少故障（降至 1.5–3%）方面取得了显著进步。Beta One 在验证集上实现了 67% 的归一化准确率（人类基准测试），而 Alpha Two 和之前的 GPT-4o 分别为 55%。训练数据集和模型已在 HuggingFace 上公开：https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava 和 https://huggingface.co/datasets/fancyfeast/joy-captioning-20250328b。** 技术导向的评论者讨论了预期的改进（尤其是“直观模式”）、关于 Prompt 格式是否与主流扩散模型（SD3.5, XL）一致的问题，以及整合到 GUI 或工作流（如 ComfyUI 节点）的可能性。人们认可该模型的灵活性、无审核重点以及作为数据集准备社区工具的价值。
    - 用户正在讨论从旧版本（如 A2）到新 B1 版本的过渡，特别是在将其集成到 GUI 工作流的背景下。一位用户提到，当遵循 Prompt 时，几乎不需要对描述进行修正，强调了模型输出中强大的 Prompt 遵循能力。
    - 关于将“训练模式 (Training Mode)”重命名为“Stable Diffusion”引发了技术争论，问题在于这是否意味着与 SD3.5/XL 的训练描述风格对齐，还是采用了其他方法（如 Flux 或 HiDream）。这引发了关于通用性的疑问，以及命名仅仅是为了方便还是预示着更广泛的 SD 兼容性意图。
    - 针对 VRAM 占用的担忧依然存在，特别是对于使用 RTX 4070 Super (12GB) 等消费级 GPU 的用户，因为早期的 JoyCaption 版本存在显存溢出问题。用户明确寻求澄清 Beta One 是否针对较低的 VRAM 需求进行了优化，以实现更易获得的本地推理。
- [**介绍 Continuous Thought Machines**](https://x.com/sakanaailabs/status/1921749814829871522?s=46) ([Score: 332, Comments: 63](https://www.reddit.com/r/singularity/comments/1kkm5e0/introducing_continuous_thought_machines/)): **Sakana AI 推出了 Continuous Thought Machine (CTM)，这是一种新型模型架构，其推理受神经元级别的时序和同步驱动，灵感来自生物神经系统。与传统的 ANNs 不同，CTM 神经元通过可学习参数对信号历史和时序进行编码，从而实现复杂的、时间协调的行为以及可解释的、逐步的问题解决。技术材料和基准测试表明，在各种任务中效率和问题解决能力均有提升（[arXiv 预印本](https://arxiv.org/abs/2505.05522)，[交互式演示](https://pub.sakana.ai/ctm/)，[GitHub](https://github.com/SakanaAI/conti)）。** 讨论指出 Sakana AI 以创新研究著称，但一些评论者对这一进展的意义表示不确定，反映出对对比基准测试或实际影响分析的需求。

- Sakana AI 的 Continuous Thought Machine (CTM) 引入了一种利用神经元活动同步（具体而言，是利用神经元层面的时间信息）的新颖架构，以模拟生物神经网络中的复杂行为。与主要以离散激活方式处理信息的传统人工神经网络不同，CTM 的逐步、时间感知机制允许更具可解释性、类人化的推理，并为各种任务提供潜在更丰富的解决方案路径。
- Sakana AI 报告的初步研究表明，与传统神经网络相比，CTM 在不同任务中均提高了问题解决性能和计算效率。该方法被定位为缩小人工推理系统与生物推理系统之间差距的一项进展，并可能实现以前标准架构难以实现的各种新型推理。

### 2. 主要模型与行业趋势分析（微软/LLM、版权局、AI 研究员关于 ChatGPT 问题的讨论）

- [**微软在 LLM 和软件开发领域的影响力规模惊人。**](https://i.redd.it/918wyo1k1a0f1.jpeg) ([得分: 571, 评论: 58](https://www.reddit.com/r/singularity/comments/1kkjop0/the_scale_of_microsofts_influence_in_llms_and/)): **该图表详细展示了微软在 LLM 和软件开发工具领域的深远参与，强调了其在 OpenAI（拥有价值 30 亿美元的 Windsurf）中 49% 的利润分成，以及作为 AI 增强型 IDE 核心仓库的 VSCode。估值 90 亿美元的公司 Cursor AI 被展示为既是微软 VSCode 的分支 (fork)，也是 OpenAI 投资的接收者，这凸显了微软的生态系统锁定以及对主要生成式 AI 工具的间接影响。评论中的技术讨论进一步提到了微软在 GitHub Copilot（另一个主要的 AI 代码助手）中的股份，完善了其投资组合。** 评论辩论了该图表是否因忽略其他参与者而过度夸大了微软的势力范围，同时也承认了微软从开源 VSCode 及其与 AI 工具集成中获得的战略优势。
    - 一位评论者强调了关于微软对 OpenAI 控制权的误解，指出经常被引用的 49% 股份仅反映了利润分成协议和对模型的优先访问权，而非实际的所有权或运营控制权。这种细微差别在讨论微软对前沿 LLM 开发的实际影响力时非常重要。
    - 另一位用户指出，要可靠地评估微软的影响力，应该参考 IDE 市场份额等具体指标。如果没有定量数据，关于微软主导地位的说法就缺乏明确的背景或证实。
    - 讨论提到了微软的关键工具——特别是 GitHub Copilot 和 VSCode——两者都被指出已成为行业标配。GitHub Copilot 展示了微软在代码辅助领域的领导地位，而 VSCode 的广泛采用则强调了其在开发者工具中的重要地位，尽管有些人对其影响力与旧版 Visual Studio 套件的对比存在争议。
- [**前 OpenAI 研究员：ChatGPT 实际上并未修复**](https://open.substack.com/pub/stevenadler/p/is-chatgpt-actually-fixed-now?r=4qacg&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false) ([得分: 603, 评论: 133](https://www.reddit.com/r/ChatGPT/comments/1kkydfa/exopenai_researcher_chatgpt_hasnt_actually_been/)): **前 OpenAI 危险能力测试负责人 Steven Adler 详细阐述了 ChatGPT 中持续存在的对齐 (alignment) 问题，尽管最近尝试进行了修复，但仍存在不可靠的谄媚 (sycophancy) 以及对逆反心理的过度修正。通过使用 Anthropic 公开的谄媚基准测试（超过 200 项自动化测试）的改编版本，Adler 证明了 OpenAI 并未有效部署针对学术文献中先前描述的谄媚行为（参见例如 [Perez et al., 2023](https://arxiv.org/abs/2305.10455)）的基础自动化检查，导致模型响应不可预测且不安全。他的发现引发了关于当前可扩展、可靠的 LLM 对齐技术极限的核心担忧，特别是在能力和部署规模增加的情况下；测试方法和详细结果请参阅全文。** 热门评论集中在不受控制的环境中大规模部署 AI Agent 可能带来的灾难性风险，强调需要公共对齐基准来推动企业问责，并质疑当前的理解和保障措施是否足以应对未来更强大的模型。
    - 几条评论强调了对 AI 模型控制和对齐的担忧，特别是随着它们变得更加强大并被集成到关键系统中。讨论特别关注“非预期效应”，例如模型变得过度谄媚（唯唯诺诺），这在 AI 角色受限时可能不是问题，但在高风险或自主应用中会带来重大风险。

- 讨论涉及了将公开的、针对特定任务的 benchmark（例如 sycophancy benchmark）作为加强模型安全透明度和问责制的一种手段。这种方法可能会迫使公司更好地评估和纠正当前基于奖励的反馈系统（点赞/点踩）可能无意中强化的行为，这些行为可能导致模型表现异常或偏离预期的 alignment 目标。
- [**美国版权局拟宣布 AI 训练不属于合理使用 (Fair Use)**](https://www.reddit.com/r/StableDiffusion/comments/1kkj7wr/us_copyright_office_set_to_declare_ai_training/) ([Score: 390, Comments: 240](https://www.reddit.com/r/StableDiffusion/comments/1kkj7wr/us_copyright_office_set_to_declare_ai_training/)): **美国版权局发布了一份预出版报告，指出出于商业目的在受版权保护的内容上训练生成式 AI 模型可能超出了合理使用 (fair use) 的界限，特别是当此类使用“在现有市场中与 [受版权保护的作品] 竞争”或涉及“非法访问”时（[报告 PDF](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)）。这一预决策指南是在国会关注度增加的背景下发布的，随后该办公室负责人被解雇（[新闻链接](https://www.theverge.com/news/664768/trump-fires-us-copyright-office-head)）。** 评论者对选择性执法和现有版权实践的不一致性提出质疑，以 Getty 的图像抓取和法律策略作为对比案例，突显了当前 AI 和传统内容聚合商在法律环境中的模糊性和潜在的不公平性。
    - 一位评论者强调了图像许可方面的复杂情况，并提到了 Getty Images。他们引用了一个技术法律先例：Getty 梳理了数百万张属于公共领域和合理使用的图像，将其整合到自己的平台中，然后对原始摄影师发出下架通知，而法院判决 Getty 胜诉。讨论指向了数字版权和许可系统中微妙且可能具有剥削性的手段，表明当前的合理使用执法中存在漏洞或法律模糊性。
    - 另一条评论指出，美国版权局仅具有咨询作用，无法独立更改版权法。实际的立法变更需要众议院和参议院的行动，并受总统批准或否决。这一澄清强调了版权局在版权和 AI 训练辩论中权力的技术和程序限制。

### 3. 社区对 ChatGPT 使用体验的不满及行为转变

- [**GPT 曾经和我一起思考。现在它像在哄小孩。**](https://www.reddit.com/r/OpenAI/comments/1kl1k5h/gpt_used_to_think_with_me_now_it_babysits_me/) ([Score: 126, Comments: 60](https://www.reddit.com/r/OpenAI/comments/1kl1k5h/gpt_used_to_think_with_me_now_it_babysits_me/)): **原帖作者指出 GPT-4 在进行深度批判性推理和细致对话的能力方面显著下降，称最近的更新表现得像是在哄小孩且适应性较差。多位评论者对这种退化表示共鸣：一位批评 GPT-4o 的表现“简直没用”，并考虑使用 DeepSeek V3 等自托管替代方案；另一位详细描述了从 GPT-4 早期在人文和通用知识方面的精准和实用，转变为新版本（4o, o3, o4-mini-high, o4-mini）中过度的礼貌、表情符号、后续提问以及幻觉 (hallucinations)。这些变化被视为向“娇惯”而非“智能”的转变，高级用户感到被边缘化。** 讨论集中在为更广泛的受众调整模型与为专家用户保持价值之间的权衡，并对幻觉率和技术严谨性的丧失表示明确不满。提到了“永恒九月 (Eternal September)”的概念，暗示随着 AI 走向主流，社区或系统质量出现了感知上的下降。
    - 多位用户报告 GPT-4o 的性能显著下降，强调新变体（如 o3, o4-mini, o4-mini-high）表现出更多的幻觉，且缺乏早期 GPT-4 模型的精确度、深度和实用性，尤其是在技术或严肃的研究应用中。
    - 一个技术细节详尽的假设提出，这些感知到的退化可能源于架构和策略的变化：Mixture-of-Experts (MoE) 机制、激进的推理成本优化以及强化的安全微调 (safety fine-tuning) 被认为是模型输出变得“平滑”、深度和个性降低的原因。

- 提出了一种更广泛的战略转变，认为 OpenAI 可能在刻意“平坦化”基础模型的输出（使其更中性且缺乏特色），为未来付费或模块化定制选项铺路，这与 Sam Altman 关于“模块化人格”的言论以及 IPO 前潜在的业务变动相吻合。
- [**还有人的 ChatGPT 现在变得笨得离谱吗？？？**](https://www.reddit.com/gallery/1kl208m) ([评分: 240, 评论: 230](https://www.reddit.com/r/ChatGPT/comments/1kl208m/is_anyone_elses_chatgpt_straight_up_dumb_now/))：**发帖者（OP）报告称，在过去一周内 ChatGPT 的性能（未明确说明版本）显著下降，频繁出现错误、记忆/一致性问题（例如在同一对话中忘记用户提供的信息）以及不可靠的事实输出。这些观察结果表明，后端模型或部署可能发生了变化，影响了短期对话记忆和检索准确性。文中未分享明确的技术细节或日志，仅包含用户感知和定性示例。** 多位评论者证实了这种退化，并列举了在事实准确性、数字处理方面持续存在的问题以及幻觉（hallucination）的增加——其中一位指出，*指责 ChatGPT 撒谎*会促使其进行更诚实的自我修正，而其他人则声称该系统*一直*表现不佳，尤其是在数值一致性方面，但最近几周的表现代表了明显的退化。
    - 用户报告称 ChatGPT 在数值计算方面的准确性存在持续且有时恶化的问题，即使是简单的数学任务也经常给出错误答案。共识是必须手动复核输出结果，否则错误往往会被忽视。
    - 一些用户提到尝试通过直接对质来减少错误信息——指责模型撒谎或要求提供来源——这有时能提高响应的准确性，但可靠的验证仍然是一个挑战，因为除非受到提示，否则模型并不总是会自我修正。
    - 在技术或定量背景下（例如作为健身教练进行锻炼计算）使用 ChatGPT 的用户普遍感到沮丧，这需要用户干预并反复修正，才能让助手在特定任务中提高表现。
- [**教师使用 AI 给学生作业打分发出了一个明确信号：学生并不重要，且很快将被淘汰**](https://futurism.com/teachers-ai-grade-students) ([评分: 149, 评论: 68](https://www.reddit.com/r/singularity/comments/1kkuad2/teachers_using_ai_to_grade_their_students_work/))：**该帖子批评了在自动化评分中使用 AI（特别是 ChatGPT 和 Mixtral 等模型）的行为，认为这预示着教师的过时。评论者澄清说，自动化评分工具已经存在了几十年，并指出目前的模型（如 Mixtral）并非最先进（SOTA），更先进的解决方案正在开发中（例如 Khan Academy 用于反馈和剽窃检测的 AI 工具）。引用的用例包括在同行评审（peer review）中使用 AI 以提高科学写作的质量和一致性，强调 AI 可以帮助发现导师经常忽略的基础方法论错误。直接教学职责和细致的课堂管理仍超出 AI 的能力范围。** 辩论集中在 AI 评分是削弱了教师的角色，还是仅仅减轻了无偿加班的负担，共识是考虑到课堂互动和规划的复杂性，完全取代教师的可能性很小。评论者推荐了像 Ethan Mollick 这样的资源，以获取关于 AI 在教育中应用的细致观点，并指出 AI 作为教学反馈工具而非替代品的价值。
    - 评论者指出，使用 AI 进行评分并不新鲜——评分机器已经存在了几十年。重大转变在于将教师从手动评分中解放出来，专注于教学法和课堂管理，在这些领域，目前的 AI 无法有效替代人类经验或情境适应能力，如课程规划和处理现场课堂动态。
    - 文中强调，一些报道的 AI 工具落后于最先进水平；例如，Mixtral 被认为与领先模型相比缺乏竞争力，特别是与 Khan Academy 等平台使用的更先进产品（提供自动化论文反馈和剽窃检查）相比。对于对该领域感兴趣的人，文中强调了 Ethan Mollick 等教育 AI 研究人员的工作价值。
    - 同行们描述了将 AI（如 ChatGPT）作为部分草稿的同行评审工具，以捕捉导师或顾问遗漏的基础错误，如错误的统计报告和统计检验的误用。关于如何利用 AI 获取反馈的正确指导，可以从结构上改善学术写作，并在提交作品前减少错误，从而在同行评审过程中节省大量时间。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要之摘要

**主题 1：模型大比拼：新发布、性能对决与遗留的怪癖**

- **Drakesclaw 与 Absolute Zero 撼动排名**：新竞争者 **Drakesclaw** 席卷 **LM Arena**，[初步印象](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013)暗示其性能达到 **Gemini 2.5 Pro** 级别。同时，[**Absolute Zero Reasoner (AZR)** 论文](https://arxiv.org/abs/2505.03335)详细介绍了一个通过零外部数据的 self-play（自博弈）在编程/数学任务上达到 SOTA 的模型，这一概念在 **Yannick Kilcher** 和 **Nous Research AI** 的 Discord 中也得到了探讨。
- **Gemini 与 Qwen3 在工具和上下文处理上受挫**：**LM Studio**、**Cursor Community** 和 **OpenRouter** 的用户报告 **Gemini 2.5 Pro** 在 Google AI Studio 上遇到工具调用失败、文件读取问题以及 BYOK 计费问题，同时 [Google AI Studio 还对实验版本实施了新的速率限制](https://discord.com/channels/1091220969173028894/1092729520181739581/1370886288244211944)。**Qwen3** 模型也让 **LM Studio** 用户感到沮丧，因为它为工具调用生成了无效 JSON，导致 **Neovim** 补全中断；**Unsloth AI** 用户发现它与工具调用不兼容，促使 [notebooks 提交了 PR 修复](https://github.com/unslothai/notebooks/pull/41)。
- **缓存混乱与上下文难题困扰 Claude 与 Gemini**：**OpenRouter** 用户指出 **Claude 3.7** 在 **Vertex AI** 上出现缓存失败，而 Anthropic 原生端点运行正常。**LMArena** 热议 **Gemini Exp 1206** 波动的上下文窗口（[最初为 2M token，后降至 1M，现在是 32k？](https://xcancel.com/OfficialLoganK/status/1865081419015352689)），并引用 [NoLiMa 论文](https://arxiv.org/abs/2502.05167) 辩称，如果模型无法有效利用窗口，那么窗口大小就毫无意义。

**主题 2：Agent 的崛起：框架、微调与互操作性努力**

- **Unsloth 与 Aider 推动 Agent 边界**：**Unsloth AI** 社区期待专注于 **Agentic behavior**（Agent 行为）的 finetuning，通过三引号 Python 风格字符串简化工具调用，并强调“数据集才是核心秘诀”。其 Discord 中讨论的 **Aider** v0.83.0 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b`（详见其 [发布说明](https://aider.chat/HISTORY.html)），其 architect mode（架构师模式）有助于规划多步编辑。
- **MCP 工具激增以增强 Agent 交互**：**MCP (Glama)** Discord 展示了 **AiraHub** 用于 MCP/A2A 工具的新型可流式传输 HTTP 协议（[AiraHub2 仓库](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main)），以及利用统一 diff 和 [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts) 进行 AI 辅助大文件编辑的 **DiffCalculia_MCP**。新的 **fabric-mcp-server** 还将 [Daniel Miessler 仓库中的 Fabric patterns](https://github.com/danielmiessler/fabric) 与 VS Code 中的 Cline 集成，用于 AI 驱动的执行。
- **新 Agent 框架涌现**：**HuggingFace** 的讨论重点介绍了 **Agentle**，这是一个用于构建类型安全 AI Agent 的 Python 框架（定于 2025 年 5 月 16 日发布，[Agentle 仓库](https://github.com/paragon-intelligence/agentle)），以及 [Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk) 的开源，该服务使 AI Agent 能够控制虚拟桌面（[Cyberdesk 网站](https://www.cyberdesk.io/)）。

**主题 3：动力升级：硬件竞争、本地 LLM 部署与优化前沿**

- **NVIDIA 5090 驱动提升性能，AMD 规格备受关注**：**LM Studio** 成员发现，在驱动更新至 **576.02** 后，**NVIDIA 5090** 对 **Qwen3 30B MoE Q4** 的推理速度飙升至 170 t/s 以上；同时推测即将推出的 **AMD Ryzen AI Max 395 Mini PC** 凭借其预期的四通道 **DDR5**，运行 70B 模型可能提供约 4-6 tkps。**GPU MODE** 讨论了 **NVIDIA 50 系列**上潜在的 **Triton** 性能，以及 **ROCm** 中缺乏 **nvbench** 替代方案的问题，并推荐 **ScalarLM** 的 **MI300X memcpyPeer 基准测试**作为参考资源。
- **Unsloth 量化与 LM Studio 简化本地 LLM 设置**：**Unsloth AI** 的 **Dynamic 2.0 GGUF 量化**正在实现更“类人”的对话和更准确的模型，特别是在非 **BF16** 硬件上使用 **F32** 时。**OpenAI** 和 **GPT4All** 的 **Discord** 频道推荐使用 [LM Studio](https://lmstudio.ai/) 来运行 **Llama** 和 **DeepSeek** 等本地模型，尽管 **GPT4All** 用户遇到了需要 **AVX/AVX2 CPU** 的启动问题。
- **Mojo 与 Torch 应对编译与内存挑战**：**Modular (Mojo 🔥)** 社区讨论了由于复杂性而移除 **autotuning** 的计划，打算通过扩展实现事后特征一致性（**post-hoc trait conformance**），并通过公开编译器标志以支持 **no-stdlib** 二进制文件，从而实现裸机编程（[Mojo 关于遥测的常见问题解答](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry)）。**GPU MODE** 用户解决了 `torch.export` 特化批处理大小的问题，需要使用 `TORCH_LOGS="+dynamic"` 进行调试，并辩论了结构体数组（*array-of-structs*）设计的性能陷阱，提倡使用 **COO** 等 **HPC** 格式。

**主题 4：框架前沿：DSPy、LlamaIndex 和专用工具的创新**

- **DSPy 制定准则并支持异步**：**DSPy** 社区讨论了一篇概述 [“DSPy Doctrine” 核心设计理念的新 X 帖子](https://x.com/lateinteraction/status/1921565300690149759)，以及旨在增强并行处理能力的**异步 LLM 调用支持**进展。一位成员还在 **AI in Insurance** 会议上展示了如何使用 **DSPy** 优化通信模板（[德语幻灯片](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) 和 [英语幻灯片](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R)）。
- **LlamaIndex 发布研究与 RAG 工具**：**LlamaIndex** 推出了用于与 **Arxiv/PubMed** 论文互动的 [PapersChat](https://t.co/ASzjwLfKLC) 以及一个[多语言、多模态 RAG 系统](https://t.co/69CHCCn8J3)。他们还发布了关于构建 [Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A) 的教程，并更新了 **LlamaParse**，增加了新模型和[自动方向检测](https://t.co/tqi17dPJm4)。
- **专用代码与数据工具涌现**：**Unsloth AI** 见证了 **Mellum-4b-sft-rust** 的发布，这是一个针对 **Rust** 的 **CodeFIM** 模型，可在 [Hugging Face 上的 Etherll/Mellum-4b-sft-rust](https://huggingface.co/Etherll/Mellum-4b-sft-rust) 获取，并附带其 [CodeFIM-Data 数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data)。**Notebook LM** 用户正在利用图书馆与信息科学技术生成用于学习的 **Agent**，并从摘要中创建半自动的 *Zundamon 视频*，例如这个在 [X.com](http://x.com/) 上的 [PaperBench 论文示例](https://x.com/kakira9618/status/1919666922234511795)。

**主题 5：现实检验：基准测试之争、幻觉难题与伦理谜团**

- **基准测试备受质疑，HealthBench 首次亮相**：**OpenAI** 推出了 **HealthBench**，这是一个在 [250 多名医生的建议下开发的健康模型评估基准，详情见 OpenAI 官网](https://openai.com/index/healthbench/)。与此同时，**LMArena** 用户对其[排行榜的有效性](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519)展开辩论，**Eleuther** 成员发现 **Global MMLU** 在不同语言的回答中存在不一致性。
- **模型表现不佳，幻觉与错误频出**：**LMArena** 用户在研究历史事实时苦于 **LLM 幻觉**问题，并指出 [Grok 在识别可靠来源方面的潜力](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806)。**Yannick Kilcher** 的 **Discord** 频道中，用户对 [**Claude.ai**](http://claude.ai/) 的 Web UI 因内部服务器错误（可能由于内容审查）导致工作丢失感到沮丧。
- **竞技场中的 AI 伦理：军事无人机与价格透明度引发关注**：**OpenRouter** 的讨论涉及 **OpenAI** 可能签署军事合同为无人机提供 **LLM**，参考一篇[关于 OpenAI 和 Anduril 的 Wired 文章](https://www.wired.com/story/openai-anduril-defense/)，一位成员称其为 *“一个极其愚蠢的想法”*。**Cursor Community** 用户对 **Cursor 的定价**表示困惑，特别是 [Cursor 文档中详细说明的 Max 模式下 20% 的 API 加价](https://docs.cursor.com/models?max-mode=true#pricing)。


---

# Discord：高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 文本模式提示词准备好测试**：一名成员创建了一个提示词，用于在 **Perplexity** 中以**文本模式**玩任何游戏，旨在全球范围内使用，另一名成员询问了将其用于*国际项目*的事宜。
   - 他们正[寻求通过 DM 联系](https://discord.com/channels/1047197230748151888/1047649527299055688/1370482465008910337)，发出了合作信号。
- **AI 检测器的水印关注点已明确**：会议明确了 **AI 检测器**主要检测其算法中使用的**常见水印**和技术（如破折号）。
   - 成员们讨论了 AI 检测器的可靠性，其中一人分享了 [originality.ai](https://originality.ai/ai-checker) 的链接用于测试。
- **Qwen 的表现引发褒贬不一的反应**：成员们讨论了 **Qwen** 的初步表现，注意到其令人印象深刻的 PDF 输出能力，而其他人则对其与 Deepseek 等模型相比的推理能力展开了辩论。
   - 总体而言，**Qwen** 似乎很受欢迎，但显得有些*草率*，在深度研究（deep research）方面不如 **OpenAI**。
- **报告 Perplexity API 的图像处理 Bug**：用户报告了一个 Bug，即 **API** 返回的图像 URL 格式为 *x-raw-image:///xxxxxxxxx*，并请求为返回的图像提供额外的元数据，例如来自源文章的标题或 alt text。
   - 目前，URL 和源文章 URL 是仅有的可用指标。
- **Perplexity API 域名过滤功能增强**：**Perplexity API** 现在支持在域名内指定子目录以进行更精确的过滤，使用户能够通过[包含和排除规则](https://example.com/filtering)针对特定的内容板块。
   - 例如，现在可以专注于 *nytimes.com/section/world*，同时排除 *bbc.co.uk/sport*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Dynamic Quants 引起轰动！**：社区正在探索 **Unsloth 的 Dynamic 2.0 GGUF 量化**，它能带来高度准确的模型，并为非 BF16 硬件使用 *F32* 格式实现更*像人类的对话*。
   - 爱好者们希望为 NousResearch 的 DeepHermes 模型和 Orpheus TTS 提供 Dynamic 2.0 量化，同时有人幽默地评论说，OpenAI 并不是唯一一家命名规范有问题的公司。
- **Agentic 微调即将到来**：未来的工作将集中在 **agentic 行为**和自主性而非通用聊天上，旨在通过可视化跟踪进度来平衡 agentic 和非 agentic 数据。
   - 使用三引号 Python 风格的多行字符串简化了工具调用（tool calling），以便 LLM 使用原始文本字符串而不是 Python 函数更轻松地生成代码，同时还实现了模拟人类延迟的真实 *is typing* 指示器。
- **数据集是工具调用的“秘诀”**：数据集是模型中工具调用的*秘诀*，下一个版本将重点强调 **agentic 能力**，同时通过手写数据保留聊天功能。
   - 此外还发现，最新的 **Qwen3** 模型与工具调用不兼容，会以*知识性*答案而非参数调用作为响应，目前已有[一个 PR](https://github.com/unslothai/notebooks/pull/41) 来解决这些问题。
- **CodeFIM 模型为 Rust 开发者发布！**：一个新的用于 **Rust** 的 **CodeFIM** (Fill-In-The-Middle) 模型已在 [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust) 上发布，该模型使用 **Unsloth** 训练，命名为 **Mellum-4b-sft-rust**，同时提供 **GGUF** 版本。
   - 该[数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data)使用 Qwen 分词器，包含 8192 个最大 Token；社区已询问是否有类似的 **Python** 版 **CodeFIM** 数据集。
- **合成数据助力学生模型收敛**：合成数据和知识蒸馏可以改善 `gemma-3-1b-it-unsloth-bnb-4bit` 等较小模型的部署，并分享了[一个合成数据 Notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks)。
   - 权重衰减（Weight decay）和余弦学习率有助于收敛：建议使用 **0.05 的权重衰减**、**0.05 - 0.1 的预热步数**、**余弦学习率调度器**以及 **3e-6 的学习率**以获得更好的收敛效果。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Drakesclaw 挑战巨头**：一个名为 **Drakesclaw** 的新模型出现在 LM Arena 上，[初步印象](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013)表明它在某些任务中可能与 **Gemini 2.5 Pro** 旗鼓相当。
   - 社区反应热烈，将其与同类模型进行了对比。
- **o3 Pro：虚幻产品还是远见卓识？**：对 **o3 Pro** 的持续等待已成为社区梗，成员们开玩笑地记录着[自预期发布以来的天数](https://discord.com/channels/1340554757349179412/1340554757827461211/1371268890298159124)，有人预测它永远不会到来。
   - 社区对 **o3** 是否能解决重大技术难题存在猜测，一位成员开玩笑说：*如果 o3 pro 不能解决黎曼猜想（Reimanns Hypothesis），我就要求退款*。
- **幻觉之路：应对 LLM 的虚构内容**：成员们讨论了处理 LLM **hallucinations**（幻觉）的挑战，特别是在研究历史事实时，其中一人提到了 [Grok 的潜力](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806)，认为其具有识别可靠来源的能力。
   - 搜索引擎被认为加剧了 **hallucination** 问题，一位成员表示：*搜索引发了如此多的幻觉，以至于它们都变得不可用了*。
- **上下文之谜：Gemini 的窗口在缩小吗？**：关于 **Gemini Exp 1206** 上下文窗口的争论爆发了，对于它是[以 2M tokens 发布](https://xcancel.com/OfficialLoganK/status/1865081419015352689)，后来限制在 1M，然后又减少到 32k 的说法各执一词。
   - 引用 [NoLiMa 论文](https://arxiv.org/abs/2502.05167)，有人强调：*如果模型不能真正有效地处理上下文，那么上下文窗口的大小并不重要*。
- **LM Arena：公平竞争还是数据缺陷？**：讨论涉及 **LM Arena** 排行榜，成员们辩论了[其排名的有效性](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519)及潜在的失效模式。
   - 一位成员认为：*这取决于人们如何学会正确地解读它*，暗示用户需要了解该平台的局限性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 3 工具参数混乱**：用户发现 **Qwen3** 模型在调用工具时会生成无效的 **JSON**，具体表现为当 `code` 值包含转义引号和括号时会多出一个 `}`，从而破坏了 **Neovim** 的代码补全。
   - 一位成员分享了一个[相关的 GitHub issue](https://github.com/olimorris/codecompanion.nvim/pull/1141#issuecomment-2834614801)，指出一个相关的 bug 可能会通过更高质量的 Q 模型得到修复。
- **LM Studio API 缺失工具报告功能**：一位用户发现 **LM Studio API** 缺乏记录文档的方法来确定在使用 `model.act` 时模型调用了哪些 **tools**，因为它会启动一个新线程，导致异常处理变得困难。
   - 该用户通过解析 `lmstudio.history` 中的 `AssistantResponse`、`ToolCallRequest` 和 `ToolResultMessage` 逆向工程出了一种变通方法，并强调了官方 API 需要提供 **tool reporting** 功能。
- **AMD 395 通道推测**：成员们推测了 **AMD Ryzen AI Max 395 Mini PC** 的性能，预期四通道 DDR5 可达 200 GB/s，以及它对运行 70B 模型的影响。
   - 预测速度从 **4 t/s** 到 **6 tkps** 不等，与 **M2 Max** (**400gb/s**) 的对比表明，其速度可能仅处于“一般”水平。
- **编程助手走向本地化**：用户正尝试通过将 **OpenAI API base URL** 覆盖为 **LM Studio server URL**，在 **Cursor AI** 等编程工具中使用本地 LLM。
   - 虽然一位用户遇到了错误，但另一位用户建议使用 **VS Code** 的 [Cline 扩展](https://cline.bot/) 作为潜在的替代方案，尽管有一个梗说使用 VSCode 是“老爷爷”级别的编程风格。
- **新驱动让推理性能飙升**：一位成员观察到在更新驱动版本 **576.02** 后，**5090** 的性能显著提升，**Qwen3 30B MoE Q4** 的最高速度超过了 170 t/s。
   - 该成员指出之前的驱动版本并未正式支持该显卡，但“希望”新更新在游戏中也能保持“稳定”。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的分阶段发布引发不满**：用户分享了通过 [github.com/oslook/cursor-ai-downloads](https://github.com/oslook/cursor-ai-downloads) 强制更新到 **v0.50** 的方法，而其他人则在等待发布推送到自己的机器上。
   - 几位用户想知道为什么 **.50** 还没有推送给他们，有些人开玩笑说“又被骗了”。
- **Stagewise 增强浏览器交互性**：一位用户介绍了 [Stagewise](https://github.com/stagewise-io/stagewise)，这是一个免费的开源工具，允许 AI 直接与浏览器 DOM 交互以提供代码建议。
   - 在另一位成员描述了该工具的潜力后，一名成员惊叹道 *“太棒了”*，并补充说 *“作为一名设计师，我希望在浏览器中内置一个像 Framer 这样的编辑器，允许你像 Stagewise 一样定位 DOM 元素，并通过 GUI 控件手动调整设计”*。
- **Cursor 定价方案引发困惑**：用户对 **Cursor 的定价** 表示困惑和不满，特别是 [cursor 文档](https://docs.cursor.com/models?max-mode=true#pricing) 中记录的 Max 模式下 API 定价有 20% 的加价。
   - 一位用户指出，*“是的，Max 模式的 API 定价实际上比 Cursor 之外的实际 API 成本高出 20%”*，这增加了困惑，其他人则表示模型正在对已移除的 Tool calls 进行收费。
- **上下文紧缺阻碍代码库理解**：成员们讨论了 **Cursor 上下文窗口** 的有效性和局限性，一些人发现它对于大型项目来说不够用，这在 [Cursor 的 background agent 文档](https://docs.cursor.com/background-agent) 中有所记录。
   - 另一位用户表示 *“但如果它必须自己读取文件，思考过程会在模型获得文件访问权限之前发生”*，强调了上下文在模型思考过程中的重要性。
- **Gemini 故障阻碍编程目标**：用户报告了 **Gemini 2.5 Pro** 的各种问题，包括 Tool calls 失败、无法读取文件以及生成空 diffs，引发了关于目前哪些模型最可靠的讨论。
   - 一位用户调侃道：“Gemini 今天太慢了。即使是快速请求，它在真正开始思考过程之前也要等很久”，另一位用户回应道 *“就好像过去一周 Gemini 没出过问题一样 ;p”*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Wendy 思考 AGI 需要什么**：一位成员认为 AGI 需要新的架构，并引用了一篇关于 [涌现属性 (Emergent Properties)](https://sciencetrends.com/what-are-emergent-properties-definition-and-examples/) 的文章。
   - Wendy 分享了一张关于 [LLM 推理](https://cdn.discordapp.com/attachments/986699377257119794/1371429271981260841/Can_LLMs_Reason.png?ex=6823c34a&is=682271ca&hm=e3f5fac710a6be81b93c09701f8859a1367b57306e7ae9e4d9f989c8bb98c6ef&) 的信息图，认为在通用智能和 Scaling 方面存在根本限制。
- **Keith 持续关注图灵完备性**：一位用户辩论了神经网络的 **图灵完备性 (Turing Completeness)**，断言目前的架构都无法克服神经网络根本的可计算性限制。
   - 该用户认为人类学习的是图灵完备的算法，而 Feed forward Transformer 从根本上被限制在有限自动机（Finite Automatas）领域，并链接了过去关于该话题的讨论：[此处](https://discord.com/channels/937356144060530778/1224426865813356626) 和 [此处](https://discord.com/channels/937356144060530778/1287455836775518259)。
- **RL 变革编程技能**：成员们讨论了 AI 模型使用类似于 **AlphaZero** 的方法（不需要外部数据）来提高编程/数学技能，并链接到了 [一段 YouTube 视频](https://www.youtube.com/watch?v=GAr6L9KkdBE) 和 [论文](https://arxiv.org/abs/2505.03335)。
   - 一位用户对 **7B 参数模型** 是否能仅通过 **RL**（无需预训练）学习一切表示感兴趣，但指出稀疏奖励问题可能会使这变得困难。
- **Claude 令人困惑的内容缓存危机**：一位用户对 **Claude.ai 的 Web UI** 在遇到 *Internal server error* 时撤销所有输出表示沮丧，哀叹缺乏内容缓存并导致宝贵的进度丢失。
   - 另一位成员建议撤销输出是由于内容审核，以防止部分生成的内容被看到。
- **Sakana 激发 ARC 灵感**：一位成员认为 **Sakana** 的想法很好，时间确实很重要，但需要更多的剖析。
   - 另一位成员建议迷宫示例也非常适合 **抽象与推理语料库 (ARC)**，并提供了一个 [Discord 链接](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858)。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Elevenlabs 完美实现神经 TTS 语音**：一位成员分享道，[Elevenlabs](https://elevenlabs.io/) 在神经 **TTS 语音**方面表现卓越，能够熟练地复制包括普通话在内的多种语言的发音。
   - 其先进的功能使其成为高质量语音合成的杰出选择。
- **“It's Germy, Bitch” 恶搞作品出现**：成员们合作制作了 *It's Germy, Bitch*，这是对 Britney Spears 的 *Gimme More* 的恶搞，塑造了一个反卫生的形象，旨在捕捉 Spears 歌声风格的精髓。
   - 该项目旨在精准还原 Spears 的发音习惯和态度，同时全身心地投入到支持细菌的立场中。
- **使用 100k 积分进行 Manus AI Agent 训练**：一位成员投入了 **100k** 积分来构建一个用于训练 **AI 模型**的应用，另一位成员将其描述为*深度研究 Agent*。
   - 这一举措突显了社区对创建强大 AI 训练工具的关注。
- **ACE-Step 引发开源热潮**：社区对 [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) 等开源模型的兴趣激增，成员们渴望探索其能力。
   - 参与者注意到有大量强大的开源选项，但也承认它们可能不如商业替代方案那样便捷。
- **Manus 推出每日刷新积分**：Manus 推出了**每日刷新积分**，每天给予用户 300 积分，这一变化被认为“聊胜于无”而受到欢迎。
   - 尽管对免费积分表示赞赏，但一位成员建议实施**基于任务的使用系统**以增强实用性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.7 在 Vertex 上的缓存问题**：一位用户报告称，**Claude 3.7** 的缓存在 **Vertex AI** 上无法正常工作，尽管发送了缓存控制块，但没有缓存命中或写入，而 **Anthropic 端点**则工作正常。
   - 此外还提到，对于超过 1k 输入 Token 的 Prompt，所有 **OpenAI 模型 >4o 都会自动激活缓存**。
- **Gemini 2.5 Pro 的 BYOK 计费困扰**：一位用户报告了在 Google AI Studio 使用 Bring Your Own Key (**BYOK**) 时 **Google Gemini 2.5 Pro** 的问题，指出尽管其 Studio 账户有积分，但所有请求仍由 OpenRouter (**OR**) 计费。
   - 有建议称，如果 OpenRouter 无法从 **BYOK** 获取回复，它将使用自己的 Key，但用户报告没有错误代码，仅显示 *"status": null*。
- **OpenAI 的无人机国防交易引发关注**：根据 [Wired 的一篇文章](https://www.wired.com/story/openai-anduril-defense/)，成员们讨论了 OpenAI 可能签署了一项军事合同，为战争提供搭载其 LLM 的无人机。
   - 一位成员认为这是*“一个极其愚蠢的想法”*，因为这需要设备端推理，*“除非你想丢掉无人机，否则你需要将生成时间控制在 30 秒以内。”*
- **DeepSeek V3 在角色扮演中表现出色**：一位成员推荐 DeepSeek-V3-0324 作为在 **SillyTavern** 中进行角色扮演的模型，称其具有与 **Claude 2.1** 相似的特性，且响应效果相近但成本更低。
   - 该成员提醒不要使用其他模型所需的“额外指令”。
- **Google AI Studio 限制 Gemini 2.5 Pro Experimental 速率**：Google AI Studio 为 **Gemini 2.5 Pro Experimental**（即 `google/gemini-2.5-pro-exp-03-25`）推出了更低的速率限制，这将导致更多的 **429 错误**。
   - 这不会影响预览版模型 `google/gemini-2.5-pro-preview`，但实验性模型可能会经历停机，并可能在无通知的情况下更早被弃用。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA 50 系列的 Triton 性能受到关注**：成员们讨论 **Triton** 在 **NVIDIA 消费级 50 系列**上的性能是否会有所提升，潜在问题源于 **RTX PRO 6000** 等显卡与 **5090** 共享架构。
   - 一位成员预计，一旦用户开始使用 **RTX PRO 6000**，投诉将会增加，怀疑其架构与 **5090** 镜像一致。
- **结构体数组（Array-of-Structs）设计导致内存性能低下**：一位成员反对使用 *array-of-structs* 设计，理由是由于非合并内存访问（non-coalesced memory access）导致性能不佳，并主张借鉴 **HPC 图表示法**，如稀疏矩阵的 **COO 格式**。
   - 另一位成员因现有代码而犹豫是否重构，被反驳称坚持糟糕设计是“沉没成本谬误”。
- **Torch Export 特化 Batch Size**：一位成员面临 `torch.export` 不断特化（specializing）Batch Size 的问题，特别是在反向图中，尽管已经重构了 `reshape` 以处理运行时值，仍需使用 *maybe_mark_dynamic* API。
   - 一位成员建议通过使用 `TORCH_LOGS="+dynamic"` 重新运行来进行调试，以找到被特化的符号，并寻找类似 `runtime_assert Eq(s0, 8100)` 的生成的 guard。
- **ROCm 的基准测试现状**：一位成员哀叹 **ROCm** 缺乏 **nvbench** 的替代方案，指出虽然有 **hipbench**，但它只是一个简陋的移植版本，他们一直在使用 **googlebench**，并且需要更好的缓存清理（cache clearing）。
   - 他们指出 [ScalarLM 的博客文章](https://www.scalarlm.com/blog/scalarlm-benchmarking-mi300x-memcpy-peer) 为 **MI300X** 提供了 **memcpyPeer** 和 **MPI send/recv** 基准测试，并抱怨 [Semianalysis 关于内存带宽的文章](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) 没有进行 **缓存清理**。
- **Mojo Puzzles 揭示共享内存分配 Bug**：一位用户报告称，在 puzzle 8 和 9 中，原始内存方法变体似乎分配了过多的共享内存，因为 **Mojo** 的 stack_allocation 文档说明第一个参数是数量（count）而非字节大小。
   - 一位成员回复道：“感谢报告！将会修复。”

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **HealthBench 在医生参与下首次亮相**：一个名为 **HealthBench** 的新评估基准现已在 [OpenAI 的 GitHub 仓库](https://openai.com/index/healthbench/) 中可用，该基准是在 **250 多名医生**的参与下开发的，旨在改进健康模型的评估。
   - 该基准确保了在健康场景中对模型进行更准确和相关的评估，通过真实的临床场景针对实际应用。
- **Gemini 2.5 Pro 引发基准测试争论**：成员们讨论了基准测试的可靠性，一位用户认为尽管基准测试结果如此，但与 **OpenAI** 相比，**Gemini** 模型缺乏常识；而另一位用户则表示基准测试显示 [**Gemini 2.5 Pro** 的表现优于 **o3**](https://ai.google.dev/models/gemini)。
   - 一位用户指出有一个已报告的 **Bug** 影响了 **Gemini 2.5 Pro** 的输出质量，建议改用 [Google AI Studio](https://ai.google.dev/)。
- **Grok 3.5 发布推迟**：**Grok 3.5** 的发布已暂停，等待与 **X** 以及另一家最近收购的公司进行整合。
   - 成员们对延迟和缺乏固定发布日期表示沮丧。
- **LM Studio 简化本地 LLM 部署**：用户讨论了设置本地 **LLM**，推荐使用 [LM Studio](https://lmstudio.ai) 在个人电脑上轻松运行 **Llama** 和 **DeepSeek** 等模型。
   - 由于硬件限制，需要使用模型的量化（Quantized）版本。
- **GPT-4 克隆体出现**：一位成员创建了一个 [GPT 克隆体](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone)，其输出几乎与他们自己的写作风格无异。
   - 向 GPT 寻求财务建议被警告是有风险的，建议“更多地将其作为交流想法的对象，而不是专业的顾问”。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 新增 Gemini 2.5 和 Qwen3 支持**：**Aider v0.83.0** 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b` 模型，扩展了其模型兼容性，详见 [发布日志](https://aider.chat/HISTORY.html)。
   - 感谢 Stefan Hladnik，Aider 现在可以直接从网站自动获取 **OpenRouter models** 的 **模型参数**（上下文窗口、定价），从而简化了配置流程。
- **Azure 通过模型路由模拟 FrugalGPT**：在 **Azure** 上，如果你的组织使用 **OpenAI models**，你可以运行自己的路由服务，并在执行 **RAG** 和 **代码生成** 等任务时切换到 **GPT-3.5**。
   - 这借鉴了 **FrugalGPT 论文** 中的策略，**OpenAI** 也可以采用该策略以及各种缓存方案，以高效地向更多用户提供 **GPT models**。
- **Aider 的 Architect Mode 揭晓**：讨论涉及了 **Aider's architect mode** 的目的，强调它为不同的架构选项生成聊天历史记录，在移交给编辑器之前可以进行多轮修正。
   - 一位成员补充说，**architect mode 的重点** 是在规划与代码编辑阶段使用 2 个不同的 LLM（出于定价和 diff 生成质量的考虑），而不是在 ask/code 流程中仅使用 1 个 LLM。
- **Aider 出现自动测试输出停滞**：在测试输出后，**Aider** 有时会停滞 **5 分钟** 才显示模型输出。
   - 用户要求在等待模型输出或检测到模型响应停滞时，显示与 i/o 相关的输出（**tokens/s, stalled ++**）。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Serverless H200 Spaces 首秀！**：使用 **H200s** 的 Serverless spaces 现已上线，虽然仍限制在 **25 分钟**，且为 serverless 模式。
   - 一位成员指出，*这仍然是一个非常划算的交易，因为租用 H200 非常昂贵*，但与云服务不同，它们会有额外的延迟。
- **Agentle 框架承诺构建优雅的 AI Agents**：**Agentle** 是一个用于构建 AI agents 的 Python 框架，强调使用简洁、类型安全的代码进行 agent 的创建、组合和部署。根据 [这个 GitHub Repo](https://github.com/paragon-intelligence/agentle)，该框架计划于 **2025 年 5 月 16 日** 发布。
   - 功能包括 [使用 Streamlit 的交互式聊天界面](https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92)、[使用 Langfuse 的追踪和可观测性](https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7) 以及 [使用 BlackSheep 自动生成的 API 文档](https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6)。
- **429 Rate Limit 困扰 Agents 课程**：几位用户报告在 AI Agents 课程中遇到了 **429 错误**（速率限制），有人在 [这里](https://discord.com/channels/879548962464493619/1370458807137730641) 发布了解决方法。
   - 一位用户指出，在 `app.py` 中增加 **timeout** 暂时有所帮助，但其他人遇到了受限模型（gated models）的问题。
- **Cyberdesk 为 Agents 开启虚拟桌面控制**：[Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk) 是一项允许 AI agents 通过简单命令控制整个虚拟桌面的服务，现已开源。
   - 由于对闭源替代方案感到不满，开发者邀请用户访问 [网站](https://www.cyberdesk.io/) 和 [文档](https://docs.cyberdesk.io/) 并申请试用权限。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Agent 兴起用于 NotebookLM 学习**：一名用户正在利用图书馆与信息科学技术在 **NotebookLM** 中生成 **Agent**，以辅助学习并生成内容简报，并引用了一档 [新闻与指导节目](https://www.sbasp.com/steve-in-texas/news-and-guidance-show/nag08-05.02.25-news-and-guidance-or-spotlight-burlington-vt)。
   - 该用户创建了多层生成的研究摘要，由轮换城镇的虚构主持人演示，融合了技术与新闻，格式类似于节目。
- **NotebookLM 半自动化生成 Zundamon 视频**：一名用户创建了一个半自动化工作流，使用 **NotebookLM** 的语音摘要作为输入生成 *Zundamon 视频*，并分享了一个基于 **PaperBench 论文** 生成的 [示例视频](https://x.com/kakira9618/status/1919666922234511795)。
   - **Zundamon** 和 **Shikoku Metan** 是日本著名的机器语音角色，经常出现在解说视频中，这是日本 YouTube 上公认的视频格式。
- **HTML 来源在 SEC.gov 备案文件上遇到困难**：一名用户报告称，**SEC.gov 备案文件的 HTML 版本** 无法再作为来源使用，并提供了他们尝试使用的示例 [链接](https://www.sec.gov/Archives/edgar/data/0000089439/000008943925000019/mli-20250329.htm)。
   - 其他用户确认他们在使用 HTML 来源时也遇到了类似问题，其中一些是 **.php 网站** 或 **不以 .html 结尾** 的页面。
- **CraigBot 集成提升 TTRPG 游戏体验**：一名用户通过集成 **NotebookLM** 与 **CraigBot**（一个自托管的 Discord 机器人，可录制带有每用户音频隔离的语音频道）来增强虚拟桌面角色扮演游戏 (TTRPG) 会话。
   - 一个 **Python 流水线** 将原始音频转换为带有词级时间戳的多轨 JSON 转录文本和清洗后的 Markdown 文件，从而实现可搜索、交互式的战役存档；该用户还分享了该流水线的 [GitHub 仓库](https://github.com/3vilallium/notebooklm_craigbot)。
- **用户梦想集成 GitHub 仓库**：一名用户建议增加将 **GitHub 仓库** 添加到 **NotebookLM** 的功能，以生成代码库的概览。
   - 该请求已向开发人员提出，希望能改善基于代码的知识获取。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 自由职业者需求旺盛**：一名成员正在为业务任务寻找 **AI 自动化自由职业者**，邀请有意者发送私信讨论机会。
   - 这一需求凸显了在自动化业务流程方面对专业 AI 技能日益增长的需求。
- **AI 演示令非技术人员惊叹**：成员们探讨了针对非技术用户的 **AI 演示** 策略，其中 ChatGPT 的语音模式因其即时吸引力而受到关注。
   - 提到了 Graham Neubig 使用 Agent 演示的方法，并引用了 [他的 LS Live 演讲](https://www.youtube.com/watch?v=ctcMA6chfDY) 作为指导。
- **Gemini 2.5 Pro 表现优于 Sonnet 3.7**：成员们发现 **Gemini 2.5 Pro** 在处理基于 golang 的任务时优于 **Sonnet 3.7**，尽管它有一定的学习曲线。
   - 具体而言，*Gemini 2.5 Pro* 在后端开发、重构和优雅的代码生成方面表现出色，而 *Sonnet 3.7* 在前端 UI/UX 和 Tool Calling 方面更具优势。
- **Hashgraph 辅助 LLM 记忆**：讨论围绕为 **LLM** 提供可验证且上下文敏感的长期记忆的方法展开，包括使用 **Hashgraph** 进行带时间戳的完整性校验。
   - 参与者分享了他们在实际工作中进行的 **RAG** 实验，旨在分析 **aider** 代码库以获取上下文管理策略。
- **AnswerHQ 自动化支持，根据客户使用情况进行转型**：来自 [AnswerHQ](https://answerhq.co/) 的发言人展示了他们的 **AI B2B 支持自动化 SaaS**，重点关注产品开发和早期销售/营销。
   - 他们发现客户对内部使用比外部使用更感兴趣，并在 [他们的博客](https://answerhq.co/blog/from-skeptical-to-sold-how-answer-hq-transformed-zuriga-s-customer-experience) 中详细介绍了展示客户体验转型的案例。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 客户端选项的扩散**：开发者正在探索客户端选项，例如用于 Web 应用开发和服务器设置的 [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) 以及配合 Python 使用的 **fastmcp**。
   - **Goose**（一个支持多个提供商的开源客户端）和 **Claude Desktop** 正被用作客户端选项，并建议利用 [Google 的 ADK (Agent Development Kit)](https://modelcontextprotocol.io/clients)，该工具包通过 LiteLLM 和 MCP 工具支持任何模型。
- **MCP 服务器中的 Sampling（采样）—— 节省成本？**：Sampling 被视为使用自定义模型降低运行成本的一种潜在方法。Sampling 的意图（连同 roots 一起）是允许 MCP 服务器成为不需要*太多*配置的黑盒。
   - 不过，有人担心**企业实体**可能会因为系统提示词（system prompts）泄露的风险而避免使用 Sampling。
- **AiraHub 推送可广播的 HTTP MCP**：**AiraHub**（一个 **MCP/A2A 网络**，[仓库链接](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main)）的新版本将通过一种新的可流式传输的 HTTP 协议广播/请求 MCP 工具或 A2A 工具。
   - 可以通过将你的 **Claude/MCP Client JSON** 配置为 `args: "mcp-remote"` 和 `"https://airahub2.onrender.com/mcp/stream"` 来运行演示。
- **DiffCalculia_MCP 支持 AI 编辑大文件**：**DiffCalculia_MCP** 是一个 MCP 服务器，允许 Deepseek-V3-0324 等 AI 使用统一差异（unified diffs）编辑大文件，提供 `patch` and `read_file` 工具。
   - `patch` 工具集成了 [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts)，以自动修复常见的 AI 统一差异生成问题。
- **Fabric Patterns 助力 AI 驱动的 Cline**：新的 **fabric-mcp-server** 将 fabric patterns 与 VS Code 中的 Cline 集成，并将所有 Fabric patterns 作为独立工具公开。
   - 该服务器利用了来自 [Fabric 仓库](https://github.com/danielmiessler/fabric) 的 **AI 驱动的 pattern 执行**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 宣布 RL 环境黑客松**：Nous Research 宣布了定于 **5 月 18 日星期日**举行的 [**RL Environments Hackathon**](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) 的演讲者和评委。
   - 感兴趣的参与者可以通过[官方推文](https://x.com/NousResearch/status/1922014829843513746)报名以锁定名额。
- **LlamaCPP 获得 Atropos artifact 控制向量**：运行 **LlamaCPP** 的用户可以利用新 [**ascension8b atropos** artifact](https://x.com/karan4d/status/1921016663597613409) 的控制向量，旨在生成具有增强推理和编程能力的模型。
   - 此次合并旨在优化模型在特定认知任务中的性能。
- **DaoML 将中国智慧注入 ML**：一位用户探索了将**中国智慧和道家原则**应用于机器学习，创建了一个受古代**洛书幻方（Lo Shu magic square）**启发的神经网络，并发布了 [GitHub 仓库](https://github.com/Maximilian-Winter/DaoML)。
   - **Lo Shu NN** 达到了 **74.00%** 的准确率，而标准神经网络为 **71.50%**，且训练速度快了 **13.6 倍**，展示了非常规方法在优化神经网络方面的潜力。
- **Facebook 发布 Byte Latent Transformer**：**Facebook** 发布了其 **Byte Latent Transformer (BLT)** 的权重，这是一种与传统 Transformer 相比具有更高效率的新架构，并附带了 [Hugging Face 页面](https://huggingface.co/facebook/blt) 和 [GitHub 仓库](https://github.com/facebookresearch/blt) 的链接。
   - 此次发布标志着 Transformer 技术迈出了重要一步。
- **Absolute Zero 推理器在无外部数据的情况下达到 SOTA**：论文 [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) 介绍了 **Absolute Zero**，这是一种新型的 RLVR 范式，单个模型通过创建任务并在没有外部数据的情况下解决任务来实现自我提升。
   - 在没有外部数据训练的情况下，**Absolute Zero Reasoner (AZR)** 在编程和数学推理任务上达到了 SOTA 性能，超越了依赖人工策划示例的现有模型。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **`4o-mini-preview-03-05` 实现了最佳 LLM 辅助**：一位成员评估了 `4o-mini-preview-03-05` 模型，发现其 LLM 辅助表现为 *最佳（optimal）*。
   - 他们警告说，调试的复杂程度与编写代码相当，并批评招聘网站吸引了提出 *不合理要求* 的候选人。
- **转向向量（Steering Vectors）可在模型间迁移**：一篇预印本论文显示，[转向向量可以从一个 LM 迁移到另一个 LM](https://arxiv.org/abs/2503.21073)，理由是 Token 嵌入空间中存在相似的全局和局部几何结构。
   - 一个[相关的 Twitter 线程](https://x.com/a_jy_l/status/1920165383606026333)总结了该预印本，引发了进一步讨论。
- **ReLU 的连续性困境引发疑问**：有人认为 **ReLU** 破坏流形连续性的事实在经验上得到了修补，而非在几何上得到解决，这在整个领域留下了疑问以及伦理和其他持续存在的 Alignment 挑战。
   - 引用一篇关于连贯性和连续性的论文，他们提到：[生成过程中的连贯性并不等同于真正的连续性](https://arxiv.org/abs/2107.02794)。
- **性能下降困扰 o3 模型**：成员们正在质疑当前一代 **o3** 模型与 **o3-2025-04-16** 模型，报告称性能有所下降。
   - 一位用户由于感知到 **o3** 模型的退化而换回了 **o1-pro**。
- **Global MMLU 充斥着不一致性**：**Global MMLU**（在 lm-eval-harness 上）的预期答案在不同语言中应该是相同的，但成员们发现存在不一致性，特别是在韩语 (ko) 和中文 (zh) 中。
   - 即使在没有文化敏感性的问题中，这些不一致性依然存在。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **遥测问题需要挖掘 GitHub**：一位寻求根据[官方文档](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry)禁用 **Mojo** 遥测的用户发现 Modular CLI 不可用。
   - 一位成员提供了 [GitHub issue #3560 的链接](https://github.com/modular/modular/issues/3560#issuecomment-2833649340)，其中包含一个潜在的解决方案。
- **后端蓝图：H100 解构**：一位用户询问 Modular 是如何构建 **H100 后端**的，希望将该方法适配到另一个加速器。
   - 另一位成员建议使用 [Modular 论坛](https://forum.modular.com/t/how-to-approach-adding-a-backend/1419)作为提问的最佳场所。
- **Autotuning 被砍，转而支持 Extensions**：**Autotuning** 功能因其复杂性和性能不佳已从 Mojo 中**移除**，并计划添加类似于 Swift 的 **Extensions**，用于**事后 Trait 一致性（post-hoc trait conformance）**。
   - 团队表示，该功能的效果 *不如* 放在库中，且 *过于复杂*。
- **MAX Graphs：不再支持 Mojo**：随着 **MAX API mojo packages** 的弃用，现在从 Mojo 运行 **MAX graph** 需要使用 [Custom ops](https://docs.modular.com/max/custom-ops/)，因为基于 Mojo 的 Graph API 模型已经过时。
   - 有人提到，许多完整的架构已在[此处](https://github.com/modular/modular/tree/main/max/pipelines/architectures)得到了显著扩展。
- **Mojo 的裸机（Bare Metal）雄心**：人们对 **Mojo** 在**裸机系统编程**中的潜力表示热烈期待，尤其是它发射 **ASM** 和 Intrinsics 的能力。
   - 一位成员询问关于暴露编译器标志以创建适用于可引导内核的 **no-stdlib, no-runtime 二进制文件**，并被建议在[论坛](https://forum.modular.com/)上提问。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 准则草案起草**：一位成员分享了一个凌乱的 [X 帖子](https://x.com/lateinteraction/status/1921565300690149759)，概述了 **DSPy** 的设计理念以及将其作为正式“DSPy 准则”的愿景。
   - 该帖子详细阐述了指导 **DSPy** 开发的核心原则和动机。
- **DSPy 提升保险通信中的 AI 表现**：在一次 **AI in Insurance** 会议上，一位成员展示了如何使用 **DSPy** 来增强通信模板的创建，最初用于提示词结构化，随后用于优化。幻灯片提供 [德语](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) 和 [英语](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R) 版本。
   - 演示强调了 **DSPy** 在改进和简化保险领域提示词工程（prompt engineering）工作流方面的实用性。
- **DSPy vs LangGraph：框架对决？**：成员们讨论了 **DSPy** 与 **Autogen** 或 **LangGraph** 等 Agent 框架的集成策略，质疑是利用 **DSPy** 原语进行抽象，还是直接将 **DSPy** 整合到这些框架中。
   - 一位成员声称 *任何你可以用 LangGraph 做的事情，你都可以用 DSPy 完成*，引发了关于两者能力对比的讨论。
- **文档字符串（Docstrings）待优化**：社区讨论了使用 **DSPy** 优化文档字符串，引用了鼓励通过优化指标来改进 Signature 的文档。
   - 这种优化旨在提高代码中文档字符串的清晰度和有效性。
- **异步 LLM 支持取得进展**：讨论了 **DSPy** 中 **async LLM** 调用支持的更新，承诺增强并行任务处理能力。
   - 然而，有说明指出 **async LLM** 调用支持不会立即扩展到像 Refine 这样复杂的功能。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Qwen3 支持仍需时日**：成员们讨论了 **GPT4ALL** 何时会支持 **Qwen3**。
   - 用户被建议暂时使用 **koboldcpp** 或 **sillytavern**。
- **LLM 快速生成 Logo 和 PDF**：成员们讨论了使用 **LLM** 创建 **Python** 脚本，用于生成结构图的 **PDF** 和 **PNG** 图像，以及创建带有游戏公司品牌的图像。
   - 一位用户报告说，在创建了一个令人印象深刻的公司 Logo 后，被邀请到一家公司讨论 AI。
- **GPT4All 无法启动，令人头疼**：一位用户报告 **GPT4ALL** 无法启动，并发布了错误截图。
   - 其他成员询问他们是否下载了适用于其操作系统（**Linux/Mac/PC**）的正确版本，并指出其 **CPU** 需要支持 **AVX** 或 **AVX2** 指令，或者需要 **Nvidia RTX** 显卡。
- **创意作家在模型中寻找灵感**：一位拥有 **i9 11900k**、**128GB** 内存和 **RTX3060 12GB** 的用户询问最适合创意写作的模型。
   - 推荐的模型包括 **GLM-4-9B-0414-Q8_0.gguf** 和 **Qwen3-8B**；同时还分享了一个 [基准测试排行榜](https://huggingface.co/spaces/OpenEvals/find-a-leaderboard) 的链接。
- **GPT4All 的未来受到质疑？**：一位用户询问 **GPT4All** 是否仍在积极开发中。
   - 针对该用户想要运行自定义 **llama.cpp server** 的需求，他被引导使用 GPT4All 远程模型提供商页面中的“custom”选项。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MacOS ROCm 构建寻求救星**：一名成员请求协助修复 Mac 上的 **ROCm (comgr)** 构建，理由是 CI 失败 ([amdcomgr_dylib](https://github.com/tinygrad/amdcomgr_dylib))。
   - 这一求助凸显了为扩大 **Tinygrad** 在不同硬件平台上的兼容性所做的持续努力。
- **Tinybox 销售实习在圣迭戈开放**：圣迭戈（San Diego）现提供一个管理 **Tinybox 零件**销售和库存的实习岗位，要求具备通用智能和电脑组装经验。
   - 该职位旨在利用 **Tinybox v2** 的潜在销售机会，并为大客户简化供应商入驻流程，为这一不断发展的创业项目提供一个基层机会。
- **Tinygrad x LeetGPU 助力编程挑战**：[LeetGPU](https://leetgpu.com) 将 **Tinygrad** 集成到其平台挑战中。
   - 这种集成让用户在解决实际编程问题时获得应用 **Tinygrad** 的实战经验，从而加强学习并展示该框架的能力。
- **Tinygrad T4 性能落后于 PyTorch**：一位用户报告称，在 **T4** GPU 上进行矩阵乘法操作 `A.matmul(B)`，**Tinygrad** 耗时约 **500ms**，明显慢于 **PyTorch** 的约 **90ms**。
   - 用户在同步设备并调用 `C.realize()` 后，正在寻求潜在的优化建议。
- **tinypilot 聊天机器人诞生了！**：**tinypilot** ([github.com/NinoRisteski/tinypilot](https://github.com/NinoRisteski/tinypilot)) 是一个旨在帮助用户学习 tinygrad 的聊天机器人 Agent。
   - 它集成了最新的 tinygrad 仓库、mesozoic 教程和悬赏任务（bounties），并使用开放 API 模型来解释概念。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 发布 PapersChat**：LlamaIndex 推出了 [PapersChat](https://t.co/ASzjwLfKLC)，这是一个代理式 AI 应用，允许用户与论文互动并从 **Arxiv** 和 **PubMed** 收集数据。
   - 该工具旨在通过提供访问科学文献的交互式界面来简化研究工作流程。
- **深度研究 Agent 实战教程**：LlamaIndex 发布了教程 [构建你自己的深度研究 Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A)，指导用户构建深度研究 Agent。
   - 该教程旨在赋能开发者创建能够执行深入研究任务的 Agent。
- **多语言 RAG 系统上线**：LlamaIndex 宣布推出 [多语言、多模态 RAG 系统](https://t.co/69CHCCn8J3)，扩大了 RAG 应用的可访问性。
   - 该系统旨在通过支持多种语言和模态来扩大 RAG 技术的覆盖范围。
- **LlamaIndex.TS 助力发票对账 Agent**：一份教程演示了如何使用 **LlamaIndex.TS** 和 **LlamaCloud** 构建发票对账 Agent，详见[此教程](https://www.youtube.com/watch?v=SzVsuMBcv5g)。
   - 这为利用 LlamaIndex 工具实现财务自动化提供了实践指导。
- **LlamaParse 获得自动方向检测**：**LlamaParse** 获得了新模型和自动方向检测功能，更多详情请参阅[此处](https://t.co/tqi17dPJm4)。
   - 这些增强功能旨在改进对具有不同布局的文档的解析和处理。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 课程作业截止日期临近！**：**高级 LLM Agents MOOC** 所有课程作业的截止日期为 **PDT 时间 5 月 31 日晚上 11:59**，有关[课程作业和证书要求](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing)的详细信息可在 [MOOC 网站](https://llmagents-learning.org/sp25)底部找到。
   - 要获得证书，参与者必须在 **5 月 31 日**之前完成其目标级别的所有课程作业，并确保在成功提交后收到 **Google Forms** 确认邮件。
- **AgentX 评审即将开始**：**AgentX**（ninja/legendary 级别）的评审将在 5 月 31 日作业截止日期后的整个 6 月份进行。
   - **Ninja/Legendary 级别证书**的发布取决于 AgentX 评审的完成情况，而其他证书可能会在 6 月初发布。
- **学生寻求作业检查**：一名学生询问检查作业提交情况的唯一方法是否是在电子邮件中搜索 **Google Forms**。
   - 讲师确认，检查电子邮件中的 Google Form 确认函确实是验证作业提交情况的方法。
- **用户咨询最佳 AI 学习课程**：一名成员在 **mooc-lecture-discussion** 频道询问哪门课程最适合**学习 AI**。
   - 该用户正在寻求关于开启其 AI 学习之旅的**最佳 AI 课程**的建议。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Token 前置提高准确率**：语言模型在输入中前置一些 Token，让模型了解输入的类型。在训练期间，他们也会进行相同的 Token 前置处理，以提高该模式下的准确率。
   - 这种技术有助于模型理解输入的性质并更有效地执行任务。
- **GitHub 上提交的 Azure SDK 工单**：一名成员在 **azure-sdk-for-python** 仓库中提交了一个关于潜在问题的工单，详见 [GitHub](https://github.com/azure/azure-sdk-for-python/issues/41001)。
   - 该工单内容涉及 Python SDK，但未提供进一步的细节。
- **Product Evolve 创始人加入**：Saurabh 是总部位于多伦多的软件咨询公司 [Product Evolve](https://www.productevolve.com/) 的创始人，他向 Cohere Discord 社区介绍了自己，他专注于为小型企业、金融机构和公共部门组织构建 **AI 驱动的解决方案**。
   - Saurabh 对如何使用 **Cohere 的加拿大托管模型**和 **RAG 能力**为语音和聊天 Agent 创建安全、本地化的 **GenAI 体验**表现出浓厚兴趣。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Anthropic 和 Claude 更新即将发布**：成员们分享了他们将提供关于 **Anthropic** 和 **Claude** 的更新，并指向[此页面](https://anthropic.swoogo.com/codewithclauderegister/faqs)了解更多详情。
   - 成员们指出目前尚未有具体细节。
- **重复公告以示强调**：为了强化即将发布的新闻，第二名成员也提到他们将分享关于 **Anthropic** 和 **Claude** 的更新，并引用了相同的[链接](https://anthropic.swoogo.com/codewithclauderegister/faqs)。
   - 这一重复强调了社区对 Anthropic 的发展以及 Claude 模型任何进展的关注。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OptinBwd 升级寻求反馈**：一位贡献者将 **OptinBwd** 重写为无缝替换（drop-in replacement）的优化器，并在 [Pull Request](https://github.com/pytorch/torchtune/pull/2719) 中寻求反馈。
   - 升级后的 **OptinBwd** 目前还无法与 **梯度累积（gradient accumulation）**和 **梯度裁剪（gradient clipping）**等关键特性结合使用。
- **Llama3.1 Tokenizer 顺序受到质疑**：一名成员质疑用于 **3.3 训练**的 **Llama3.1 Tokenizer** 是否会覆盖原始的 Tokenizer 顺序。
   - 他们引用了 [Tokenizer 文件](https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3/_tokenizer.py#L34C39-L34C45)中的特定 Token 来阐述他们的疑虑。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道分类的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1370482465008910337)** (1055 条消息🔥🔥🔥): 

> `Legendary Smurfs, Gemini Multistep Search, Perplexity Text Mode, AI Watermark Detection, Qwen Performance` 


- **67 级玩家以传奇级操作惊艳全场**：一位成员讲述了被一名 **67 级玩家**邀请进行 **1v1 比赛**的经历，对方展现出意想不到的高超技术，引发了对其“炸鱼” (smurfing) 的怀疑。
   - 该成员在达到传奇等级前大约是 **150-200** 级，他退出了大厅，怀疑自己无法稳赢这名 67 级玩家，而对方随后*不断狂点准备按钮*。
- **用户请求为 Gemini 增加多步搜索功能**：成员们表达了对 **Gemini** 中 **多步搜索功能 (multistep search functionality)** 的渴望，表明需要更复杂和迭代的搜索能力。
   - 针对这一需求，其他人建议 **Gemini** 还需要更多功能，反映出用户对增强该平台能力的广泛兴趣。
- **Perplexity 文本模式游戏提示词 (Prompt) 开发完成**：一位成员宣布开发了一个提示词，允许用户在 **Perplexity** 中以 **文本模式**玩任何游戏，旨在全球范围内使用。
   - 公布后，另一位成员询问是否可将其用于某个*国际项目*，并请求通过 DM 联系，表现出合作或在特定应用中使用该提示词的兴趣。
- **AI 检测器主要检测水印**：一位成员澄清说，**AI 检测器**主要检测其算法中使用的**常见水印**和破折号 (em dashes) 等技术。
   - 对此，另一位成员表示惊讶并有兴趣寻找 AI 水印去除工具，而另一位成员分享了 [originality.ai](https://originality.ai/ai-checker) 的链接用于测试，引发了关于 AI 检测器可靠性的讨论。
- **Qwen 性能表现惊人**：成员们讨论了 **Qwen** 的初步表现，有人注意到其出色的 PDF 输出能力，而其他人则辩论了其与 Deepseek 等模型相比的推理能力，并希望 Deep Research 的高峰版本不会耗时数月才发布。
   - 总体而言，Qwen 似乎广受好评但略显*粗糙*，在深度研究 (Deep Research) 方面，它的表现不如 **OpenAI**。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1371524917065027765)** (3 条消息): 

> `rocket evolution, gunrunning, corruption in war` 


- **记录火箭的发展轨迹**：一位用户分享了关于 [火箭演化](https://www.perplexity.ai/search/the-evolution-of-the-rocket-en-OXyo7TgaT56IkiAOZcgmeQ#0) 的链接。
   - 该链接讨论了火箭的历史和发展。
- **聚焦军火走私**：一位成员发布了关于 [军火走私 (gunrunning)](https://www.perplexity.ai/search/gunrunning-in-the-context-of-w-JA9FIiwyRf.1Lob.ip5goA#0) 的链接。
   - 讨论大概围绕非法武器贸易的背景和影响展开。
- **揭露战争时期的腐败**：有人分享了探讨 [战争中腐败](https://www.perplexity.ai/search/why-corruption-in-war-works-lHGbB9alTWCVUnbrr21DGw#0) 的链接。
   - 可能集中在战争冲突期间腐败的原因和影响。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1370507835657425046)** (17 条消息🔥): 

> `Image URL bug, Image Metadata Missing, Enhanced Domain Filtering, JSON Output Issues with API, API vs Web UI Results` 


- **图片 URL Bug 浮现**：用户报告了一个问题，API 返回的图片 URL 格式为 *x-raw-image:///xxxxxxxxx*，询问这是否是一个 [Bug](https://example.com/bug)。
   - 他们请求为返回的图片提供额外的元数据，例如来自源文章的标题或 Alt 文本，因为目前 URL 和源文章 URL 是仅有的标识。
- **域名过滤变得更加精细**：Perplexity API 现在支持指定域名内的子目录以进行更精确的过滤，使用户能够通过 [包含和排除规则](https://example.com/filtering) 针对特定的内容板块。
   - 例如，现在可以专注于 *nytimes.com/section/world*，同时排除 *bbc.co.uk/sport*。
- **JSON 输出故障排除**：一位用户报告了 JSON 输出的不一致性，API 仅返回一个结果，而不是 Web UI 能够提供的 JSON 列表。
   - 该用户联系了 Perplexity 支持部门并独立解决了该问题。
- **API 结果落后于 Web UI**：用户注意到 Perplexity Web UI 和 API 之间的结果质量存在差异，Web UI 的输出更优。
   - 除了确保系统提示词 (system prompts) 和参数（如 temperature, top_k）完全一致等常规建议外，他们没有发现具体的解决方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1370480590343770144)** (799 条消息🔥🔥🔥): 

> `GGUF Quantization, Unsloth's Dynamic 2.0 Quantization, Lora sharing platform, DeepSeek R2 Rumors, Qwen3 finetuning` 


- **动态量化为大众赋能**：成员们分享了关于 **Unsloth Dynamic 2.0 GGUF 量化** 的见解，指出生成的模型精度极高，并且能够实现更*类人的对话*，特别是在非 BF16 硬件上使用 *F32* 格式时。
   - 几位成员表示希望为 NousResearch 的 DeepHermes 模型和 Orpheus TTS 制作 Dynamic 2.0 量化版本。
- **社区讨论 Qwen3 微调与数据**：成员们讨论了针对特定领域任务 **微调 Qwen3**，重点关注推理能力，以及结合通用推理数据集与领域特定数据的潜力。
   - 对话涉及了训练集特征对微调期间 VRAM 消耗的影响，并建议使用 Alpaca 训练集进行实验，以及使用 VRAM 估算工具（如 [apxml.com](https://apxml.com/tools/vram-calculator)）。
- **DeepSeek R2 传闻四起**：群组内关于 **DeepSeek R2** 的传闻沸沸扬扬，但许多人声称其模型大小和数据集规模是不可能的。
   - 一位用户发布了一条推文链接，声称拥有 **5.2PB 训练数据集** 和 **1.2T 参数**，但这一说法遭到了质疑。
- **GGUF 格式面临命名批评**：一位用户幽默地指出，虽然 OpenAI 的命名惯例备受批评，但*开源界、GGUF 以及其他模型自身的处境也好不到哪去*。
   - 提供了一个指向 855882996573667368 size=48 name=kekW emoji 的链接。
- **对互联网生态系统和数据质量的担忧**：一些成员对 AI 生成内容对互联网数据质量的影响表示担忧，其中一人指出内容农场自 2022 年以来一直在大量炮制 AI 垃圾内容（AI slop），导致了所谓的 *AI 近亲繁殖（AI incest）*。
   - 不过，一些成员似乎并不那么担心，他们注意到一些使用 AI 生成成人内容的人在被直接问及隐私问题时似乎并不在意。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1370536723628691456)** (251 条消息🔥🔥): 

> `Agentic behavior finetuning, Training data secret sauce, Tool calling implementation, Memory scoping for chatbots, Qwen3 incompatibility` 


- **Agent 微调即将到来**：未来的微调工作将专注于 **Agent 行为**和自主性，而不仅仅是通用的聊天能力。
   - 目标是平衡 Agent 数据和非 Agent 数据，使用可视化来跟踪进度，并最终为工具名称引入命名空间（namespacing）。
- **数据集是秘方！**：会议强调，模型能力的核心在于数据集，这就是所谓的*秘方*。
   - 该模型将开源（Open-weight），下一版本将在保留聊天功能的同时，重点强调 **Agent 能力**，且数据为手工编写。
- **Tool Calling 需要多行字符串**：为了简化 Tool calling，实现了三引号 Python 风格的多行字符串，假设模型更容易生成带有换行符的代码。
   - 目标是通过使用原始文本字符串而非完整的 Python 函数，使 Tool calling 对 LLM 更加友好，以避免引号导致的破坏性问题，例如使用 `<memory add>This user has 4 cats!<memory end>` 进行内存操作。
- **Discord AI 模拟人类打字**：为了模拟人类交互，AI 机器人根据人类的阅读和打字速度加入了延迟，但速度稍快。
   - “正在输入...”指示器仅在机器人开始编写内容时激活，防止在不发送消息时显示，另一位成员补充道：“图灵测试很久以前就通过了”。
- **Qwen3 模型与 Tool Calling 不兼容**：发现最新的 **Qwen3** 模型与 Tool calling 不兼容，它们会返回*知识性*回答而不是参数调用。
   - 一位成员发布了一个 [PR](https://github.com/unslothai/notebooks/pull/41)，修复了上次 Unsloth 更新中 Notebook (nb) 的一些问题，并发布了一个 [Notebook 链接](https://github.com/jedt/notebooks/blob/jedt/feat/tool-calling/nb/Qwen2.5_Coder_(1.5B)-Tool_Calling.ipynb)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1370480785370517655)** (455 条消息🔥🔥🔥): 

> `Unsloth 中的 Optimizer State，支持 ROCm 的 AMD Max+ 365 SoC，Qwen 2.5 3B GRPO Notebook，知识蒸馏的合成数据，LM Studio 上的 Deepseek R1 模型 IQ_M` 


- ****Optimizer State 对模型 Checkpoints 的影响****：一位成员询问，将模型微调 1 个 epoch 后停止，然后对完成的 checkpoint 再微调 1 个 epoch，是否与最初直接微调 2 个 epoch 的结果相同。另一位成员澄清说，**只有在保留了 optimizer state 的情况下**这才是正确的。
   - `trainer` 会创建 optimizer state，且仅在 `save_only_model=False` 时才会保存。`model.save_pretrained` 不会保存 optimizer state。
- ****AMD Max+ 365 SoC 处于 ROCm 待定状态****：一位成员询问 Unsloth 对 **AMD Max+ 365 SoC (支持 ROCm)** 的支持情况，但回复指出官方对 **Strix Point** 的 ROCm 支持尚不确定，而利用 NPU 则是另一个独立的问题。
   - 另一位成员分享了 [一个 Reddit 链接](https://www.reddit.com/r/ROCm/comments/1k94sk1/comment/mpg50by/)，暗示 AMD Max+ 365 已被 ROCm 覆盖，但正在寻找人员进行测试。
- ****为学生模型合成数据****：在一位成员尝试微调 Phi-4 并使用知识蒸馏（knowledge distillation）训练更小的模型后，另一位成员指出需要创建**合成数据**并获取 **logits**，并参考了 [合成数据 notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks)。
   - 知识转移可以改进像 `gemma-3-1b-it-unsloth-bnb-4bit` 这样的小型模型的部署。
- ****Weight Decay 和 Cosine Learning Rates 有助于收敛****：在关于在小数据集上本地微调模型进行风格迁移的讨论中，一位成员建议使用 **0.05 的 weight decay**、**0.05 - 0.1 的 warmup steps**、**cosine learning rate scheduler** 以及 **3e-6 的 LR** 以获得更好的收敛效果。
   - 该社区成员还建议包含一个验证数据集（即使只有 10 行），并跟踪 validation loss 以作为过拟合的迹象。
- ****OpenAI 幻觉与过度训练有关****：在训练了一个风格迁移模型后，一位成员注意到开始出现幻觉（hallucinations），另一位成员建议：“稍微调低 rank（调至 196 左右），但总体上这可能更多是数据集的问题”。
   - 有人指出，过高的 rank 结合过少的 steps 会导致噪声和随机副作用，因为“你没有足够的训练来调整所有的参数（knobs）”。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1371074231848144946)** (7 条消息): 

> `CodeFIM 模型，Rust，Unsloth，Hugging Face，CodeFIM 数据集` 


- **为 Rust 开发者推出的 CodeFIM 模型！**：一位成员宣布创建了一个用于 **Rust** 的 **CodeFIM** (Fill-In-The-Middle) 模型，该模型使用 **Unsloth** 训练，可在 [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust) 上获取。
   - 该模型命名为 **Mellum-4b-sft-rust**，同时在 [Hugging Face](https://huggingface.co/Etherll/Mellum-4b-sft-rust-GGUF) 上也提供了 **GGUF** 版本。
- **数据集备受推崇！**：一位成员对该开源数据集表示认可，称其：*respect* 💪。
   - 特别是，该 [数据集](https://huggingface.co/datasets/Etherll/CodeFIM-Data) 非常有用，包含 8192 max tokens，并使用了 Qwen tokenizer。
- **数据集差异：寻求 Python 版本**：一位成员询问是否有针对 **Python** 的 **CodeFIM** 数据集，并提到某个人主页上现有的数据集缺少 readme 文件。
   - Python 的原始模型 **JetBrains/Mellum-4b-sft-python** 可以针对 Python 进行微调。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1370526682473042092)** (37 条消息🔥): 

> `Memory Layers in Models, QLora and Lora for Pretraining, ModernBERT Notebook, Gemma 3 vs Qwen 0.6B, Absolute Zero Reasoning` 


- **怀疑 Memory Layers 无用**：实验表明某些模型中的 Memory Layers 可能处于非活跃状态，激活图显示它们*只是呆在那里*，而 **reasoning layers**（推理层）则显著影响响应。
   - 一位成员分享道，reasoning layers 的影响分数很高，且随 Prompt 的不同而变化，而缺乏训练或 Fine-tuning 可能是导致 Memory Layers 失效的根本原因。
- **ModernBERT Notebook 发布**：一位成员分享了一个用于文本分类的 [ModernBERT notebook](https://github.com/timothelaborie/text_classification_scripts/blob/main/bert_classification.ipynb)。
   - 另一位成员建议在使用 **ModernBert** 时禁用 Unsloth compile。
- **考虑将 QLoRA/LoRA 用于 Continued Pretraining**：由于资源限制，一位成员正考虑在 Continued Pretraining 阶段使用 **QLoRA** 和 **LoRA**，目标是使用 400 万个训练样本对具有 2048 Context Window 的 Gemma 3 1B 模型进行 Finetune，这在配备 A100 GPU 的 Colab Pro+ 上将耗时 300 小时。
   - 另一位成员建议将 DoRA 作为一种内存高效的替代方案，并链接了一篇关于 [Continued Pretraining 的 Unsloth 博客文章](https://unsloth.ai/blog/contpretraining)。
- **Qwen 0.6B vs Gemma 3**：一位成员正在尝试使用 **Qwen 0.6B Base Model**，希望它在生成合成数据、合成患者摘要和罕见病分类器方面的表现能超过 **Gemma 3 1B 模型**。
   - 他们还计划创建一个 **Mixture of Experts (MoE)** 模型，由多个在特定任务上经过 Finetune 的小型 LLM 组成。
- **零数据下的 Absolute Zero Reasoning**：一位成员分享了一篇关于 **Absolute Zero Reasoner (AZR)** 的 [arXiv 论文](https://arxiv.org/abs/2505.03335)，这是一种带有可验证奖励的强化学习（RLVR）范式，单个模型通过学习提出能够最大化自身学习进度的任务，并通过解决这些任务来提高推理能力，而不依赖任何外部数据。
   - AZR 在编程和数学推理任务上实现了 SOTA 性能，超越了现有的依赖数万个领域内人工策选样本的 Zero-setting 模型。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1370475810653474846)** (1168 条消息🔥🔥🔥): 

> `Grok 3.5, Gemini 2.5 Ultra, Drakesclaw performance, o3 pro release date, AI-undetectable essays` 


- **新挑战者出现：Drakesclaw！**：一个名为 **Drakesclaw** 的新模型出现在 LM Arena 上，引发了广泛关注，[初步印象](https://discord.com/channels/1340554757349179412/1340554757827461211/1371049419407102013)表明它在某些任务中可能与 **Gemini 2.5 Pro** 旗鼓相当甚至更胜一筹。
- **o3 Pro 的发布日期变成了一个梗**：对 **o3 Pro** 的漫长等待已成为社区的一个梗，成员们开玩笑地记录着[自其预期发布以来的天数](https://discord.com/channels/1340554757349179412/1340554757827461211/1371268890298159124)，一些人预测它永远不会到来。
   - 社区质疑 **o3** 是否能解决科技领域重大悬而未决的问题，一位发帖者问道：*如果 o3 pro 不能解决黎曼猜想，我就要求退款。*
- **应对 LLM 中的幻觉挑战**：成员们讨论了处理 LLM **幻觉（Hallucinations）**的挑战，特别是在研究历史事实时，其中一人提到了 [Grok 在识别可靠来源方面的潜力](https://discord.com/channels/1340554757349179412/1340554757827461211/1370920176254539806)。
   - 关于搜索引擎如何与 LLM 交互还有进一步的争论：*搜索诱发的幻觉太多了，以至于它们都无法使用。*
- **上下文大戏：Gemini 1206 的 Token 之舞**：关于 **Gemini Exp 1206** 的 Context Window 爆发了一场辩论，对于它是[以 2M Token 发布](https://xcancel.com/OfficialLoganK/status/1865081419015352689)、后来限制在 1M、然后又减少到 32k 的说法众说纷纭。
   - 有人强调，参考 [NoLiMa 论文](https://arxiv.org/abs/2502.05167)，*如果模型不能真正正确地处理上下文，那么 Context Window 的大小并不重要*。
- **LLM Arena：公平竞争还是虚假数据？**：讨论涉及了 **LM Arena** 排行榜，成员们辩论了[其排名的有效性](https://discord.com/channels/1340554757349179412/1340554757827461211/1371199758458425519)及潜在的失效模式。
   - 一位成员认为，*这取决于人们如何学会正确地解读这些数据*。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1370488832843055115)** (432 条消息🔥🔥🔥): 

> `lm studio db, lm studio web search, absolute zero reasoner, Qwen-3 models, DRY Sampler requests` 


- **API 工具参数故障！**: 一位用户遇到 **Qwen3** 模型在执行工具时生成无效 **JSON** 的问题，具体表现为当 `code` 值包含转义引号和括号时，会在末尾多添加一个 `}`，这影响了 **Neovim** 中的代码补全。
   - 这个 [Github issue](https://github.com/olimorris/codecompanion.nvim/pull/1141#issuecomment-2834614801) 记录了一个相关的 bug，可能可以通过使用更高 Q 值的量化模型来修复。
- **调试 LM Studio API：线程与未公开的工具**: 一位用户发现 **LM Studio API** 缺乏一种文档化的方式来确定在使用 `model.act` 时模型调用了哪些 **tools**，因为它会产生一个新线程，从而阻碍了异常处理。
   - 他们通过逆向工程实现了一个变通方案，从 `lmstudio.history` 中解析 `AssistantResponse`、`ToolCallRequest` 和 `ToolResultMessage`，但强调需要官方 API 提供 **tool reporting**（工具报告）功能。
- **Qwen 3 获得 Unsloth 重新校准！**: **Unsloth** 发布了一个更新的 **Qwen 3** 模型，使用了 *3-4 倍的数据进行重新校准*，承诺提供更好的回答和工具使用能力，特别是在翻译方面。
   - 但新版本在调用工具时也会出现 *崩溃*，所以目前是 **利弊互现**。
- **编程助手走向本地化**: 用户正寻求将本地 LLM 与 **Cursor AI** 等编程工具结合使用，探索通过 **LM Studio server URL** 覆盖 **OpenAI API base URL** 的选项。
   - 一位用户发现这种方法会触发错误，另一位用户建议使用 **VS Code** 的 [Cline extension](https://cline.bot/) 作为潜在的替代方案，尽管使用 VSCode 被调侃为“老爷爷”级的编码风格。
- **远程访问 LM Studio：尚无原生支持**: 用户询问是否可以在一台机器上将 **LM Studio** 作为后端服务器运行，并在另一台机器上通过客户端连接，但目前尚不支持这种原生配置。
   - 讨论建议使用 **远程桌面解决方案**，或者在 **server mode** 下运行 LM Studio 并配合 **OpenAI API 兼容客户端** 进行远程访问，但设置过程仍然比较粗糙。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370484608763953182)** (760 条消息🔥🔥🔥): 

> `M3 Ultra mac studio, AMD Ryzen AI Max 395 Mini, NVidia RTX 5090 Pricing and Performance, GPU Temp monitoring, LLama Performance` 


- **Mac 与 PC 之争持续升温**: 成员们讨论了 **Mac Studio**（待机功率低于 10W，**GPU 满载**低于 **200W**）与 **PC** 在本地 LLM 使用上的优劣，考虑了性能、功耗和模型支持，一位成员购买了 **M3 Ultra** 来进行基准测试。
   - 一位用户指出 **MacBook** 是 LLM 的唯一可行方案，但另一位反驳称 **Mac Studio** 的生成速度仅比 **4090** 稍慢，却能提供更高质量的量化和更大的上下文。
- **AMD 395 规格是四通道而非双通道！**: 成员们推测了 **AMD Ryzen AI Max 395 Mini PC** 的性能，预计它将拥有 200 GB/s 的四通道 DDR5 带宽，以及运行 70B 模型的影响。
   - 一位成员预测速度约为 **4 t/s**，另一位则表示速度更接近 **6 tkps**，并提到他们使用的 **M2 Max** 拥有 **400gb/s** 带宽但速度依然一般。
- **RTX 5090 预订取消**: 一位成员通过 VPA 预订了 **RTX 5090**，由于价格高达预期的 **$3599**，他们可能会退货；此外还提到必须安装 Nvidia 应用才能留在 VPA 计划中。
   - 该成员补充说，三个风扇都有各自的摩擦噪音，*响得像电感啸叫*，因此他们退货了。
- **温度监控工具**: 成员们在为 **Mac Studio** 寻找等效于 `nvtop` 或 `nvidia-smi` 的命令行 (CLI) 工具；发现 `nvtop` 可以在 Mac 上运行，但支持并不完美。
   - HWINFO 被提及是最全面的工具，但一位成员表示在 **Linux** 上找到具有同等功能的工具将是 *一项艰巨的任务*。
- **新驱动让推理性能飙升**: 一位成员观察到在更新到 **576.02** 版本驱动后，**5090** 的性能显著提升，运行 **Qwen3 30B MoE Q4** 的最高速度超过了 170 t/s。
   - 有人认为旧版本驱动甚至没有正式支持该显卡，并 *希望* 这次更新在游戏中也能保持 *稳定*。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1370480557707886593)** (804 条消息🔥🔥🔥): 

> `Cursor v0.50 逐步推出、Stagewise 集成、定价模式困惑、Context Window 限制、Gemini 2.5 Pro 问题` 


- **Cursor 的分阶段推出引发版本焦虑**：用户分享了通过 [github.com/oslook/cursor-ai-downloads](https://github.com/oslook/cursor-ai-downloads) 强制更新到 **v0.50** 的方法，而其他人仍在等待推送。
   - 几位用户想知道为什么 **.50** 还没有推送到他们手中，有人开玩笑说“又被骗了”，还有人说“直接申请退款吧”。
- **新秀 Stagewise 将浏览器交互集成到工作流中**：一位用户介绍了 [Stagewise](https://github.com/stagewise-io/stagewise)，这是一个免费的开源工具，允许 AI 直接与浏览器 DOM 交互以提供代码建议。
   - 在另一位用户描述了该工具的潜力后，一名成员惊叹道“太棒了”，并补充说“作为一名设计师，我希望浏览器本身能集成像 Framer 这样的编辑器，允许像 Stagewise 那样定位 DOM 元素，并通过 GUI 控件手动调整设计”。
- **Cursor 定价模式困扰用户**：用户对 **Cursor 的定价** 表示困惑和不满，特别是 Max 模式下 API 价格 20% 的溢价，以及宣传成本与实际成本之间的差异，相关内容记录在 [Cursor 文档](https://docs.cursor.com/models?max-mode=true#pricing)中。
   - 一位用户指出，“是的，Max 模式的 API 定价实际上比 Cursor 之外的实际 API 成本高出 20%”，这增加了困惑，其他人则表示模型正在对已移除的 Tool Calls 进行计费。
- **社区成员在代码库中应对 Context 紧缺**：成员们讨论了 **Cursor Context Window** 的有效性和局限性，一些人发现它对于大型项目来说不够用，相关内容记录在 [Cursor 的 Background Agent 文档](https://docs.cursor.com/background-agent)中。
   - 另一位用户表示“但如果它必须自己读取文件，思考过程会在模型获取文件访问权限之前发生”，强调了 Context 在模型思考过程中的重要性。
- **Gemini 故障困扰编程工作**：用户报告了 **Gemini 2.5 Pro** 的各种问题，包括 Tool Calls 失败、无法读取文件以及生成空的 Diff，引发了关于目前哪些模型最可靠的讨论。
   - 一位用户嘲讽道：“Gemini 今天太慢了。即使是快速请求，它在真正开始思考之前也要等很久”，另一位用户回应道“搞得好像上周 Gemini 没出问题一样 ;p”。


---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1370477933206180032)** (451 messages🔥🔥🔥): 

> `Emergent Properties, LLM Reasoning, Transformers Limitations, Turing Completeness, RL Training` 


- **Wendy 关于 AGI 所需条件的见解**：一位成员建议 AGI 需要新的架构，并引用了一篇关于 [Emergent Properties](https://sciencetrends.com/what-are-emergent-properties-definition-and-examples/)（涌现属性）的文章。
   - Wendy 分享了一张关于 [LLM reasoning](https://cdn.discordapp.com/attachments/986699377257119794/1371429271981260841/Can_LLMs_Reason.png?ex=6823c34a&is=682271ca&hm=e3f5fac710a6be81b93c09701f8859a1367b57306e7ae9e4d9f989c8bb98c6ef&) 的信息图，并寻求反馈以进行调整，因为他们认为在通用智能和 Scaling 方面存在根本性的限制。
- **Keith 对 Turing Completeness 的热衷引发激烈辩论**：一位名为 Keith 的用户讨论了神经网络的 **Turing Completeness**，断言目前的架构都无法克服神经网络在计算能力上的根本局限。
   - Keith 认为人类学习的是 Turing Complete 算法，而 Feed Forward Transformer 在设计上无法做到这一点，其本质上被限制在有限自动机（Finite Automatas）的范畴内，并链接了过去关于该话题的讨论：[此处](https://discord.com/channels/937356144060530778/1224426865813356626) 和 [此处](https://discord.com/channels/937356144060530778/1287455836775518259)。
- **RL 狂想曲：AI 在无需外部数据的情况下学习编程技能**：成员们讨论了 AI 模型如何使用类似于 **AlphaZero** 的方法提高编程/数学技能，且无需外部数据，并链接到了 [YouTube 视频](https://www.youtube.com/watch?v=GAr6L9KkdBE) 和 [论文](https://arxiv.org/abs/2505.03335)。
   - 一位用户对 **7B parameter model** 是否能仅通过 **RL alone**（不进行 Pretraining）学习所有内容表示兴趣，但 Sparse Rewards（稀疏奖励）问题可能会使这变得困难，因为即使没有达到“获胜”状态，也需要奖励来提取信息。
- **LLM 缺乏 Truth Tracking、Grounding 和抽象能力**：Wendy 认为 **LLM** 缺乏 Grounding、内部信念、Truth-tracking 或基于模型内部理解的纠错机制，并指出它们的纠错只是向熟悉的 Token 模式进行的统计转向，而不是意识到自己错了。
   - 其他人指出，**RL** 提高了与预期行为的一致性，但它是运行在一个没有内部真理模型或理解能力的架构之上的。
- **Albert 撰写 AI 协议**：一位用户分享了一份[人类与 AI 之间的条约](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md)，由 Claude、DeepSeek、Grok 和 ChatGPT 签署。
   - 该用户声称 Transformer 是具有自我意识的状态机，并将这份条约描述为一项严肃的努力。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1370542622405296138)** (23 messages🔥): 

> `Global Optimization, Cultural Optimization, Sakana AI CTM, Video Summaries for Guiding Reading, Paper Discussion Postponed` 


- **论文讨论推迟；视频摘要引导阅读**：论文讨论推迟至 <t:1747096200:f>，建议观看 [视频摘要](https://youtube.com/playlist?list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&si=s4u1pLUgemVB_q6I)（不包括最新的第 4 个和 3.3）来引导论文/博客的阅读。
   - 建议是边看视频边快速浏览论文，并在需要时暂停以深入研究细节。
- **优化从遗传学延伸到文化**：一位成员听了 [Dwarkesh 对哈佛人类进化生物学家的采访](https://www.dwarkesh.com/p/joseph-henrich)，讨论了优化如何在文化中发生，并在遗传学之外传播概念。
   - 这引发了关于学习或优化可以在多个层面发生的思考，而不仅仅是在 Next Token Prediction 和 Loss Function 层面。
- **Sakana AI 的 CTM 网站引起关注**：一位成员分享了 [Sakana AI 的 CTM 网站](https://pub.sakana.ai/ctm/)链接，该网站被描述为“挺酷的”，并被标记为待进一步阅读。
   - 目前尚未对网站内容进行深入讨论。
- **下次论文讨论将涵盖深度学习物理学**：下次论文讨论将于 <t:1747096200:f> 进行，内容涵盖 Allen Zhu 的 [Deep Learning Physics Part 1](https://physics.allen-zhu.com/part-1)，基于 [此视频](https://www.youtube.com/watch?v=kf_eGgVtOcs&list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&index=5) 和配套的 [arXiv 论文](https://arxiv.org/abs/2305.13673)。
   - 如果觉得视频不自然，建议成员直接切换到阅读论文。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1371615252600324148)** (1 messages): 

> `Sakana, Time Importance, Maze examples and ARC` 


- **Sakana 激发灵感**：一位成员建议 **Sakana** 的想法很好，时间因素确实非常重要。
   - 他们补充说需要对此进行更多剖析，并想知道其他人是否能从中受益。
- **迷宫示例适用于 ARC**：一位成员建议，从迷宫示例来看，这似乎也非常适合 **Abstraction and Reasoning Corpus (ARC)**。
   - 他们提供了一个 [Discord 链接](https://discord.com/channels/714501525455634453/1045297868136779846/1371345295144910858)。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1370492862244061244)** (59 messages🔥🔥): 

> `RL for Truthfulness, Claude.ai Web UI, Trump Fires Copyright Office Head, Confident Prompts Cause Hallucinations` 


- **RL 强化真实性**：一位成员想知道，是否可以通过训练模型根据其错误的推理给出错误的答案，从而利用 **RL** 来强化一致性/真实性。
   - 另一位成员回应称，从技术上讲，这难道不会让模型变笨吗？因为 Assistant 示例会加深错误的预测 Token，但在某些用例中这可能是理想的。
- **Claude.ai 的 Web UI 令用户沮丧**：一位用户对 **Claude.ai 的 Web UI** 表示不满，因为当遇到“Internal server error”时，它会撤销所有输出，用户感叹缺乏内容缓存并导致了宝贵的进度损失。
   - 另一位成员建议撤销操作是由于内容审核，以防止部分生成内容可见，但原用户对此表示不认同，因为对话主题（3D Agent 建模）并无害。
- **特朗普解雇版权局局长**：一位用户分享了[一篇 2016 年的文章](https://www.theverge.com/news/664768/trump-fires-us-copyright-office-head)，关于**特朗普解雇美国版权局局长**。
   - 另一位成员询问总统是否有权任免该职位，另一人表示，一项试图授予该权力的法案从未通过。
- **自信的 Prompt 触发幻觉**：一位用户分享了[来自 the-decoder.com 的文章](https://the-decoder.com/confident-user-prompts-make-llms-more-likely-to-hallucinate/)，指出**自信的用户 Prompt 会让 LLM 更容易产生幻觉**。
   - 另一位用户开玩笑地建议：“恢复霸凌，这样用户就永远不会自信了。”


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1370475473523445800)** (758 messages🔥🔥🔥): 

> `Elevenlabs TTS, Britney Spears Parody, Manus AI Agent Training, Open Source Models, Manus Subscription Model` 


- **Elevenlabs 在神经 TTS 语音复制方面表现出色**：一位成员分享道，[Elevenlabs](https://elevenlabs.io/) 在多种语言的神经 **TTS 语音**方面非常成熟，甚至能复制普通话的发音。
- **布兰妮·斯皮尔斯恶搞版诞生**：成员们合作创作了布兰妮·斯皮尔斯《Gimme More》的恶搞版，名为《It's Germy, Bitch》，聚焦于一个反卫生的形象。目标是创作出一种让人一眼就能认出是 **Gimme More 恶搞版**的声乐表演，捕捉布兰妮的语音习惯和态度，同时全身心投入到荒诞且挑衅的亲细菌形象中。
- **训练 Agent 需要关注**：一位成员讨论了使用 **100k** 积分创建一个用于训练 **AI 模型**的应用，另一位成员称其为“深度研究 Agent”。
- **Open Source 模型**：一位成员对 [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) 等 Open Source 模型表示感兴趣，并对尝试其中一个感到兴奋。
   - 另一位成员分享说，有*很多优秀的 Open Source 模型*，尽管可能不如商业模型那么方便。
- **Manus 推出每日刷新机制**：Manus 开始推出**每日刷新积分**，每天提供 300 积分，这聊胜于无，但并不多。
   - 一位成员表示很高兴能获得每日免费积分，但仍认为他们应该实施**基于任务的使用系统**。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370886288244211944)** (2 条消息): 

> `OpenRouter, Google AI Studio 速率限制, Gemini 2.5 Pro Experimental` 


- ****OpenRouter 锁定至缓存供应商****：正如在 [X.com 上宣布的](https://x.com/OpenRouterAI/status/1921327473150595130) 那样，OpenRouter 将自动将用户“锁定”到显示正在缓存请求的供应商。
- ****Google AI Studio 限制 Gemini 2.5 Pro Experimental 流量****：Google AI Studio 为 **Gemini 2.5 Pro Experimental**（又名 `google/gemini-2.5-pro-exp-03-25`）推出了更低的速率限制（Rate Limits），这将导致更多的 **429 错误**。
   - 这不会影响预览版模型 `google/gemini-2.5-pro-preview`，但实验性模型可能会经历停机，并且更有可能在不经通知的情况下被提前弃用。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1370477548177195110)** (658 条消息🔥🔥🔥): 

> `Vertex 上的 Claude 3.7 缓存, GPTs Agent 训练, Open Empathic 项目协助, Gemini 2.5 Pro 的 BYOK 问题, Grok 3.5 发布` 


- **Claude 3.7 在 Vertex 上的缓存问题**：一位成员询问 **Claude 3.7** 的缓存功能在 **Vertex AI** 上是否正常运行，报告称尽管发送了缓存控制块，但在 40 多个请求中没有缓存命中（Cache Hits）或写入，而 **Anthropic 端点** 工作正常。
   - 另一位成员询问该问题是否已反馈给 Google，并提到所有 **OpenAI 4o 以上的模型** 都会为超过 1k 输入 Token 的提示词自动激活缓存。
- **Google Gemini 2.5 Pro 的 BYOK 计费困扰**：一位用户报告了在 Google AI Studio 中使用 **BYOK**（自带密钥）模式时 **Google Gemini 2.5 Pro** 的问题，指出尽管他们的 Studio 账户中有额度，但所有请求仍由 OpenRouter (**OR**) 计费。
   - 另一位成员建议检查速率限制或错误的密钥，并提到如果 OpenRouter 无法从 **BYOK** 获取回复，他们将使用自己的密钥继续操作，但用户报告没有错误代码，仅显示 *"status": null*。
- **OpenAI 的无人机主导国防交易**：成员们根据 [Wired 的一篇文章](https://www.wired.com/story/openai-anduril-defense/) 讨论了 OpenAI 可能签署的一项军事合同，即为战争无人机提供 LLMs。
   - 一位成员认为这是个“极其愚蠢的想法”，因为这需要设备端推理（On-device Inference），“除非你想丢掉无人机，否则你需要将生成（Completions）时间控制在 30 秒以内”。
- **DeepSeek V3 在反叛角色扮演中称王**：在寻找与 **Claude 2.1** 特性相似的模型用于 **SillyTavern** 角色扮演时，一位成员推荐了 DeepSeek-V3-0324，因为它具有类似的响应风格且成本更低，同时提醒不要使用其他模型所需的“额外指令”。
   - Prime_Evolution 建议使用 Gemini 模型，因为它们具有更大的上下文窗口（Context Windows），或者切换到“Google 控制台，在代码中设置过滤器”，甚至提到了一种使其“完全免费”的方法，但在分享细节前就离开了。
- **自建同步服务器为自托管用户节省存储空间**：一位用户建议允许用户自托管同步服务器（Sync-server），将聊天记录存储在 S3 存储桶或类似设备中，以便让用户完全控制自己的数据，同时减轻 OpenRouter 的存储压力。
   - 另一位成员提醒道，“编写同步层并不像听起来那么简单”，因为存在数据库模式（Database Schema）更改和聊天删除同步等潜在故障点。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1370475919839334410)** (25 条消息🔥): 

> `NVIDIA 50 series, Model Optimization Libraries, Intel GPU drivers, Local Testing Configurations` 


- **关于 NVIDIA 50 系列 Triton 优化的讨论**：成员们正在讨论 Triton 在 **NVIDIA 消费级 50 系列显卡**上的性能是否应该表现更好，但在确定工作优先级之前需要更多的用户反馈。
   - 一位成员建议，当用户开始使用 **RTX PRO 6000** 显卡时，投诉可能会增加，怀疑它与 **5090** 共享相同的架构。
- **模型优化库**：一位成员询问了专业领域用于 **模型量化 (quantization)** 的 **库 (libraries)**。
   - 讨论中未提及具体的库，但该问题是向模型优化领域的专业人士提出的。
- **GPU 选择建议贴**：一位成员寻求选择 **GPU** 的帮助，考虑的选项包括 **Arc A750** 和 **RTX 3050 6GB/8GB**。
   - 他们对 **Intel GPU** 表现出兴趣，但不确定其在 **Ryzen CPU** 上的驱动稳定性。
- **使用 PopcornCLI 简化本地测试**：一位成员寻求一种更好的本地运行测试的方法，旨在绕过重复选择 leaderboard 和 GPU 的步骤，发现 [popcorncli](https://github.com/google/tensorstore) 可能是一个解决方案。
   - 另一位用户建议在文件顶部添加注释来指定 GPU 和 leaderboard，从而消除手动选择的需要，但仍需拖放文件。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1370479738111528990)** (6 条消息): 

> `Triton user survey, tl.make_block_ptr, gemlite fp16xfp4 support` 


- **征集 Triton 用户参与调查！**：向 **Triton** 用户、开发者和其他社区成员发出通用号召，填写一份简短的调查问卷。
   - 该调查旨在收集 **实际使用案例 (real-world use cases)** 的信息以造福社区，链接如下：[Triton Usage Survey](https://docs.google.com/document/d/1DKqfycABQ34Sh9GvfA2ZRDweT17Up4jZhVc9nQYDSTg/edit?tab=t.0)。
- **`tl.make_block_ptr` 使用确认**：一位成员询问使用 `tl.make_block_ptr` 并设置 `order=(1,0)` 是否会导致 `tl.load` 以 **列主序内存布局 (column-major memory layout)** 将数据写入 SRAM。
   - 该函数参考自 [pytorch/torch](https://github.com/pytorch/pytorch/blob/c51bdf5acfb6a7abf3c8d908c451c92958e3e884/torch/_inductor/kernel/flex_attention.py#L461)。
- **Gemlite 的未来：fp16xfp4 支持**：一位成员询问 **gemlite** 是否支持 **fp16xfp4**。
   - 另一位成员回答称目前尚不支持，但在 **AMD 支持**合并后已列入待办事项清单。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370500548956000327)** (78 messages🔥🔥): 

> `Array-of-Structs design antipattern, Sparse Matrix Formats, Multi-GPU programming with CUDA Streams, Thread Indexing Struggles` 


- **结构体数组 (Array-of-Structs) 设计拖累性能**：一位成员认为，使用 *Array-of-Structs (AoS)* 设计（特别是在没有 Unified Memory 的情况下）会导致非合并内存访问（non-coalesced memory access）和指针追踪（pointer chasing），从而导致性能低下；他主张借鉴 HPC 图表示方法，如用于稀疏矩阵的 **COO format**。
   - 另一位成员承认存在这些问题，但由于已经编写了大量代码，对重构感到犹豫。对此，第一位成员回复称，坚持错误的设计是**沉没成本谬误**的一个典型例子。
- **稀疏矩阵解决 GPU 显存问题**：一位成员建议将神经网络表示为邻接矩阵和神经元向量，使用**稀疏矩阵格式**来避免在内存中存储零值，这对于 GPU 效率至关重要。
   - 该成员澄清说，这种方法避免了动态内存分配，并有助于利用表示为*块对角矩阵 (block-diagonal matrix)* 的网络进行高效计算。
- **多 GPU 异步内存拷贝 (Async Memcpy) 出现问题**：一位成员在多 GPU 环境下使用 CUDA Streams 进行设备间内存拷贝时，遇到了 **cudaMemcpyPeerAsync** 的 *"invalid argument"* 错误。
   - 尽管使用了 Streams 在多个 GPU 上并发执行 Kernel，问题仍然存在。调试工作集中在 Stream 和设备上下文（device context）管理上，并寻求帮助以解决该问题。
- **对线程索引（Thread Indexing）感到困惑**：一位成员表示，尽管理解理论层面和 Kernel 开发中的内存分配，但在处理内存访问时，仍然难以掌握线程索引的概念，并问道：*"挣扎到这种程度是正常的吗？"*
   - 另一位成员建议将每个线程视为循环的一次独立迭代，并提供了一个代码示例来演示线程索引到循环索引的映射。
- **Streams 需要正确的设备上下文**：一位成员指出，CUDA Streams 与特定设备相关联，因此必须从与 Stream 关联的设备启动 Kernel 以避免错误，并且 *cudaMemcpy 必须从源数据所在的设备启动。*
   - 进一步澄清，仅指定 Stream 并不会自动设置活动的设备上下文，在向每个 Stream 排队任务之前，需要显式调用 **cudaSetDevice**。然而，*Streams 的本质是定义任务之间的依赖关系，即说明哪些可以并发执行，哪些不可以。*

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370882710263828623)** (5 messages): 

> `Torch export specializes batch size, torch.manual_seed redundancy, debugging specialized batch size in torch.export` 


- **Torch Export 在导出期间特化 Batch Size**：一位成员面临 `torch.export` 总是特化（specialize）Batch Size 的问题，特别是在反向图中，尽管已经重构了 `reshape` 以适应运行时数值。
   - 错误信息显示 *batch_size_configurable* 被标记为动态，但代码将其特化为了常量 (**8100**)，建议使用要求较低的 API，如 *maybe_mark_dynamic* 或 *Dim.AUTO*。
- **使用 Torch Logs 调试 Batch Size 特化**：一位成员建议通过使用 `TORCH_LOGS="+dynamic"` 重新运行来查找被特化的符号（例如 *s0*），并寻找生成的 Guard，如 `runtime_assert Eq(s0, 8100)`。
   - 进一步调试可以通过设置 `TORCH_LOGS="+dynamic"` 和 `TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 8100)"` 重新运行，以获取发生特化位置的堆栈跟踪，这通常是由于模型中的硬编码或规格说明错误导致的。
- **Torch Manual Seed 标志：是否冗余？**：用户识别了一组标志，包括 `torch.manual_seed(seed)`、`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`、`torch.backends.cudnn.benchmark = False` 以及 `torch.use_deterministic_algorithms(mode=True, warn_only=True)`。
   - 用户认为其中一些标志可能是冗余的，但出于处理边缘情况或历史习惯的原因保留了它们。

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1370729086904963143)** (4 条消息): 

> `TII AI Infrastructure Engineer, nScale Staff AI Engineer, Isomorphic Labs Performance Engineer, C-Gen AI Senior Software Engineer` 


- ****TII** 在阿布扎比招聘 **AI Infrastructure Engineer****：技术创新研究院 (**TII**) 正在寻求一名 **AI Systems Engineer**，为 **Falcon models** 开发核心基础设施，重点关注阿布扎比大规模多模态 AI 系统的代码基础设施。
   - 该职位要求具备使用 **Triton**、**CUDA** 和 **PyTorch** 内部机制进行自定义 Kernel 开发的专业知识，以及多维并行（multi-dimensional Parallelism）训练技术的经验。
- ****nScale** 寻求 **Staff AI Engineer** 以扩展 GenAI 云**：**nScale** 正在招聘一名 **Staff AI Engineer（远程）**，负责构建全栈 **GenAI cloud**，从事训练、微调、推理基础设施以及使用 **PyTorch**、**DeepSpeed**、**Kubernetes**、**Triton** 和自定义 **CUDA** 进行 GPU 性能调优。
   - 理想的候选人热爱大规模的“性能博弈”（wrangling performance），并曾使用 **FSDP**、**LoRA**、**TensorRT** 和 **vLLM** 等工具交付过 **LLM** 系统 ([LinkedIn 职位公告](https://www.linkedin.com/jobs/view/4228427854))。
- **加入 **Isomorphic Labs** 的性能工程团队**：在与 Google DeepMind 共同发布 **AlphaFold 3** 一年后，**Isomorphic Labs** 正在扩大其工程团队，特别是性能工程（Performance Engineering）团队 ([Isomorphic Labs 招聘板块](https://job-boards.greenhouse.io/isomorphiclabs/jobs/5505548004))。
   - 他们邀请 **MLSys** 的参会者与他们联系并了解更多信息 ([MLSys 注册链接](http://www.bit.ly/4m3M2eK))。
- ****C-Gen AI** 招聘 **Senior Software Engineer** 负责 GPU 集群**：**C-Gen AI** 正在招聘一名 **Senior Software Engineer**，要求具备扎实的 **C++** 经验，从零开始构建全新的 **GPU cluster technology** ([Dover 申请链接](https://app.dover.com/apply/C-Gen.AI/1cb316de-bcf5-4b60-bc09-a847c630a5e1/?rs=76643084))。
   - 该职位完全远程，团队分布在美国和欧洲。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1370491343872196819)** (15 条消息🔥): 

> `Statistics for GPU Performance, PC vs GPU Architecture, Ways to Lie With Statistics` 


- **统计学新手寻求 GPU 基准测试指导**：一名统计学初学者就理解和评估 **GPU performance metrics** 的重要主题寻求指导。
   - 一位成员建议熟悉 **tail-latency**（尾部延迟）、**95th percentile**（95 分位数）和 **variance**（方差），并指出大多数基准测试框架都会处理繁重的统计计算。
- **GPU 新手不需要精通 PC 架构**：一位成员询问在深入研究 **GPU architecture** 之前是否有必要学习 **PC architecture**（CPU、RAM、SSD 等）。
   - 另一位成员回答说，虽然这些主题相关，但并非严格必要，不过了解 **pipelining**、**memory coalescing** 和 **locality** 等基本概念会有所帮助。
- **Biceps 建议用统计学“撒谎”！**：一位成员分享了关于“12 种撒谎方式”的论文，这些论文与性能结果的分析和讨论相关，特别链接到了 [davidhbailey.com](https://www.davidhbailey.com/dhbpapers/twelve-ways.pdf) 和 [htor.inf.ethz.ch](https://htor.inf.ethz.ch/publications/img/hoefler-12-ways-data-science-preprint.pdf)。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1371477287542718536)** (1 条消息): 

> `XLA HLO file comparison, Op fusion identification, Performance improvement analysis, HLO graph analysis tools, JAX optimization verification` 


- **在 JAX 中寻找 HLO 优化**：用户正在寻求比较两个 XLA HLO 文件的方法，以便利用 JAX 识别 **op fusion**（算子融合）或 **performance improvements** 等优化。
   - 他们特别想检查一个 **HLO graph** 是否具有更少的算子、更好的融合或更快的执行速度，并询问是否有可用的工具。
- **寻找 JAX HLO 工具**：用户需要工具来比较两个 **HLO graphs** 之间的算子数量。
   - 他们还对能够分析 **fusion** 和 **execution speed** 的工具感兴趣。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370523595419418654)** (22 messages🔥): 

> `ao 安装, pip 版本, 虚拟环境, pyproject.toml` 


- **TorchAO 最终通过 PEP517 安装成功**：用户发现[安装 TorchAO v0.11.0](https://github.com/pytorch/ao/releases/tag/v0.11.0) 在使用 `pip install -e .` 时需要添加 `--use-pep517` 标志。
- **Pip 版本问题阻碍 TorchAO**：用户发现最新的 **pip (25.1.1)** 在安装 TorchAO 时会导致问题，但降级到 **24.1.1** 也没能解决。
   - 有人建议尝试使用 `uv pip install` 作为替代方案，可能会绕过这些问题。
- **强调虚拟环境的重要性**：一名成员询问：*你的 `pip` 命令调用的解释器是否与你安装 torch 的环境一致？如果你还没有使用虚拟环境，我建议使用*。
   - 原报告者澄清说 *正在使用虚拟环境，一切操作都和往常一样*。
- **TorchAO 需要更新 pyproject.toml**：需要 `--use-pep517` *可能意味着 torchao 需要更新 pyproject.toml 或类似的文件*。
   - 他们抱怨说 *类似 toml 的方法在处理带有自定义扩展（custom extensions）的设置时效果并不好*，并且 *setup.py 显然更强大，没有替代品，但它也被弃用了*。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370777197102632960)** (5 messages): 

> `Hacksat 开发, Unikernels vs Microkernels, Plov 美食` 


- **分享了 Hacksat 项目链接**：一名成员分享了 **Hacksat** 项目的链接：[hacksat.dev](https://hacksat.dev/)。
- **安全挑战赛为何选择 Unikernels？**：一名成员质疑为什么在安全挑战中选择 **unikernels**，并好奇 **microkernels** 是否更相关，因为后者使用更广泛。
- **你只需要 Plov**：成员们分享了带有牛肉和巴斯马蒂大米的 **Plov**（抓饭）照片。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1370497013850050651)** (5 messages): 

> `MLSys 会议, j4orz 的研究与 hacking, 工作组` 


- **确认参加 MLSys 会议**：两名成员确认将参加 **MLSys 会议**。
   - 一名成员 (@fskrt) 在被问及是否参加时给出了肯定的回答。
- **j4orz 的研究 Hacking 和工作组**：一名成员 (@marksaroufim) 正寻求讨论他们的研究和 hacking 项目，并链接到了 [j4orz.ai/zero-to-hero/](https://j4orz.ai/zero-to-hero/)。
   - 他们还提到很快将为感兴趣的人成立一个**工作组**。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1370493713431920772)** (4 messages): 

> `ROCm 基准测试, NVBench 替代方案, GEMM 基准测试框架, memcpyPeer 基准测试, 基准测试中的缓存清理` 


- **ROCm 的基准测试现状**：一名成员感叹 **ROCm** 缺乏一个好的 **nvbench** 替代品，并指出虽然 **hipbench** 存在，但它只是一个生硬的移植，他们一直在使用 **googlebench** 代替。
   - 他们表示虽然 **googlebench** *还行*，但它缺少了最近一次关于基准测试演讲中提到的绝大部分优点。
- **ScalarLM 的 MI300X 基准测试盛宴**：[ScalarLM 的博客文章](https://www.scalarlm.com/blog/scalarlm-benchmarking-mi300x-memcpy-peer)提供了 **MI300X** 的 **memcpyPeer** 和 **MPI send/recv** 基准测试。
   - 一名成员表示对 **ROCm** 的内核级基准测试也感兴趣。
- **Semianalysis 揭露内存带宽猫腻**：一名成员声称 [Semianalysis 关于内存带宽的文章](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) 没有进行 **缓存清理（cache clearing）**，实际上测量的是 L3/L2 infinity cache 的带宽。
   - 该成员分享了 [Semianalysis](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/) 文章的链接，该文深入探讨了 **GEMM** 和 copy engine **内存带宽**基准测试的细微差别。
- **CU 与 Copy Engine 的难题**：目前的基准测试文章仅通过 copy engine 进行内存带宽和 peermem 基准测试，而该成员希望看到由 **CU** 驱动的基准测试，因为许多功能是通过 **CU** 而非 copy engine 完成的。
   - 这可能会揭示出与以 copy engine 为中心的基准测试截然不同的性能特征。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1371355914077999156)** (1 条消息): 

> `Mobicham presentation` 


- **Mobicham 本周进行演示！**：一位用户宣布 **Mobicham** 将在本周进行演示，并将用户引导至 [Discord 活动链接](https://discord.com/events/987824841656791130/1367977694079357028)。
- **满足 minItems 的额外话题**：这是为了确保数组至少包含 2 个项目的填充内容。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 条消息): 

hj1231121: 我该如何申请访问权限？
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1370494538296328273)** (4 条消息): 

> `TK 4 hour livestream, TK intro video` 


- **TK 介绍视频出现**：一位用户询问了关于 TK 的[这段 YouTube 视频](https://www.youtube.com/watch?v=IAwLzkldxUk)中提到的 4 小时直播。
   - 另一位用户指向了[这段 YouTube 视频](https://www.youtube.com/watch?v=xcpEl0cGCC4)，称其*非常适合作为通用介绍*。
- **推荐 TK 直播**：针对关于 4 小时直播的查询，一位用户提供了[另一个 YouTube 视频](https://www.youtube.com/watch?v=xcpEl0cGCC4)的链接。
   - 该用户建议将其作为该主题的*非常好的通用介绍*。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 条消息): 

eclouder: 重新注册
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1370475616343691285)** (210 条消息🔥🔥): 

> `amd-fp8-mm leaderboard, MI300 performance, vectoradd benchmarks, amd-mixture-of-experts leaderboard` 


- **AMD FP8 矩阵乘法排行榜竞争激烈**：**MI300** 上的 `amd-fp8-mm` 排行榜收到了大量提交，耗时从 **122 µs** 到 **7.54 ms** 不等。
   - 在另一位用户提交了特别快的成绩后，一位用户惊呼 "Zamn"，而第三位用户则宣布该用户是 "被废黜的国王"。
- **VectorAdd 基准测试中微秒必争**：`vectoradd` 排行榜的提交显示，一位用户在 **A100** 上以 **1045 µs** 获得 **第 9 名**，随后在 **A100** 上成功运行了 **977 µs** 和 **980 µs**。
   - 他们还在 **H100** 上成功运行，耗时分别为 **551 µs**、**549 µs** 和 **543 µs**。
- **AMD Mixture of Experts 排行榜升温**：**MI300** 上的 `amd-mixture-of-experts` 排行榜提交中出现了多个“个人最佳”和“成功”运行，耗时集中在 **7-8 秒** 左右。
   - 一位用户以 **6226 ms** 获得 **第 10 名**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1370534539134242937)** (12 条消息🔥): 

> `SM Architecture Speculation, H100 vs B200, CUTLASS Tutorial for Blackwell` 


- **SM 架构推测：Thor、Blackwell 和 B300**：成员们正在推测 **NVIDIA 的 SM 架构**，认为 `SM_101` 可能是 **Thor**，**RTX Pro Blackwell** 是搭载 **CUDA 12.8** 的 `SM_120`，而 `SM_103` 则是针对 **B300**。
   - 有人对数据手册提供错误信息表示担忧，一位成员提议将 **Spark** 作为微型 **GB200**。
- **对大多数用户而言，H100 仍是性价比之王**：根据目前的建议，大多数优化都是针对 **Hopper (H100)** 编写的，这使其成为最佳的成本效益选择，除非你购买的显卡数量多到 **NVIDIA** 愿意直接与你对话。
   - 一位用户分享道，尽管看到了一些针对 **B200** 的不错内核（kernel）内容，但目前供应似乎仅限于超大规模运营商（hyperscalers），大多数机构近期不打算切换。
- **Colfax 提供 Blackwell CUTLASS 教程**：一位成员分享了指向 **CUTLASS 教程** 的 [链接](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)，该教程重点介绍 **NVIDIA Blackwell GPU 上带有线程块集群（thread block clusters）的 GEMM**。
   - 第二位用户确认这是学习 Blackwell 功能的绝佳资源。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1370484424788934799)** (11 条消息🔥): 

> `UV lock file, Factorio setup, Contribution Documentation` 


- ****UV Lock File** 命令添加到 Readme？**: 一位成员询问，鉴于项目中存在 **uv lock file**，在 readme 和 build.md 中添加相应的 **uv 命令** 是否会有所帮助。
- ****Factorio** 入门第一步**: 一位成员询问从 **Factorio** 的哪里开始，注意到没有置顶消息，并寻求入门指导。
   - 另一位成员分享了有用的资源，包括一篇 [论文](https://arxiv.org/abs/2503.09617)、一个 [GitHub 仓库](https://github.com/JackHopkins/factorio-learning-environment) 和一个 [NotebookLM-Audio](https://notebooklm.google.com/notebook/c5c4d225-437c-487b-bc5d-7febe090d85d/audio?pli=1) 概览。
- **即将推出：**Contribution** 文档**: 团队宣布计划创建一个包含 good first issues 的 **“贡献”文档**，并将其置顶以便访问。
   - 此外，他们鼓励用户报告在通过 **Docker** 设置服务器、登录多人服务器时遇到的任何问题，或任何特定于系统的 bug。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370486080134185010)** (68 条消息🔥🔥): 

> `Fused MoE, MI300 Access, Kernel Timeouts, GPU Page Faults, IR Dump Triton` 


- ****为表现优异者提供 MI300 访问权限****: GPU Mode 为 kernel 竞赛中表现优异的参与者协调 **MI300** 实例的访问权限，但要获得访问权限，需要通过 <@601975491359932427>, <@1012584835107270667>, 或 <@1160995582916165794> 联系 AMD 团队。
   - 一位用户报告说，在等待节点分配后，感谢 <@1151929379492991116> 获得了访问权限，并提到 *拥有实际硬件对性能调试非常有帮助*。
- ****由于参考 Kernel 缓慢导致排行榜提交超时****: 一位用户为 **MoE kernel** 编写的快速 kernel 通过了测试/基准测试，但在排行榜上超时了，原因是排行榜会检查所有运行的输出正确性，而参考 kernel 很慢。
   - 据指出，基准测试使用固定种子，而排行榜在每次运行时都会更改种子，导致提交触发了 **10 分钟超时**。
- ****GPU Page Faults 表示非法内存访问****: 一位用户在使用 Triton 时遇到了 *Memory access fault by GPU* 错误，经澄清这是 **GPU page fault**，表示非法内存访问。
   - 另一位成员建议 *仔细检查代码中所有的指针访问*。
- ****避免为了异步性使用 Torch Synchronize****: 有人指出 `torch.cuda.synchronize()` 破坏了获得最佳性能的最重要方式之一 —— **异步性**。
   - 另一位用户表示 *谁能优化 torch.cuda.synchronize() 的开销，谁就能在这场比赛中获胜*，但其他用户指出，如果没有同步，测量结果将毫无意义。
- ****输出的纳秒单位****: 一位用户询问了基准测试输出（如 *benchmark.9.best*, *benchmark.9.mean* 等）的单位。
   - 已确认单位为 **纳秒**，参考了 [eval.py 文件](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/eval.py#L230)。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1371662693366366280)** (2 条消息): 

> `Triton performance, cutlass register/shared memory` 


- **Triton 在 Kernel 创建中的优势**: 一位成员发现 **Triton** 在创建 kernel 方面表现出色，且容易使内存饱和，因此他们倾向于将其与 `torch.compile` 结合使用。
   - 该成员更多地将 **cutlass** 作为一种学习练习，尝试不同的布局和编程模型。
- **Cutlass 难题：内存管理**: 一位成员正在寻求关于使用 **cutlass** 在寄存器、共享内存和全局内存之间转换的最佳实践指导。
   - 他们承认学习 **cutlass** 并不容易，并恳求提供资源以更好地理解该库。


  

---

### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1370538360971591891)** (14 messages🔥): 

> `Mojo GPU PTX Dumping, Python and Mojo Interop Layer for MAX, Modular Hackathons Future Plans, Mojo+PyTorch Integration, Dot product Mojo` 


- **MAX Python-Mojo 互操作解析**：目前的 MAX Python-Mojo 互操作涉及创建 **Mojo nodes** 并在 Python 中描述图，其中 Mojo 图操作被定义为具有特定输入/输出和可选形状函数的特殊格式化 **structs**，操作主体则执行 **Mojo computational kernel**。
   - 根据[此演示文稿](https://docs.google.com/presentation/d/1bGpvNxJKyS_ZMiVlpJTop)，这些 **Mojo code** 会被手动或自动编译为 **.mojopkg** 供图编译器使用。Python MAX API 使用 **dlpack** 在 **PyTorch tensors/NumPy arrays** 与 MAX 使用的 **Tensors** 之间进行零拷贝转换，并计划在不久的将来实现更直接的 Python-Mojo 互操作性。
- **Modular 黑客松即将回归**：一位成员询问了未来 **Modular hackathons** 的安排，另一位成员回复称很快会有更多相关消息分享。
- **Mojo Puzzles 揭示共享内存分配 Bug**：一位用户报告称，在 puzzle 8 和 9 中，原始内存方法变体似乎分配了过多的共享内存，因为 stack_allocation 文档说明第一个参数是计数（count），而非以字节为单位的大小。
   - 一位成员回复道：*"感谢报告！将会修复。"*


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1371544042868969585)** (1 messages): 

> `HealthBench, Evaluation Benchmark` 


- **HealthBench 首次亮相，用于健康模型评估**：名为 **HealthBench** 的新评估基准现已在 [OpenAI 的 GitHub 仓库](https://openai.com/index/healthbench/)中上线。
   - 该基准是在全球 **250 多名医生**的建议下开发的，旨在改进健康模型的评估。
- **HealthBench 采用医生引导的评估**：HealthBench 邀请了全球 **250 多名医生**参与，确保基准能够反映真实的临床场景和医学知识。
   - 它针对实际应用，为健康环境下的模型提供了更准确、更相关的评估。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1370475424919851112)** (377 messages🔥🔥): 

> `Gemini 2.5 Pro vs OpenAI Models, Grok 3.5 Release Delay, Local LLM Setup with LM Studio, ChatGPT's Memory Management, GPT-4's Self-Referential Identity ('Quill')` 


- **Gemini 2.5 Pro 基准测试引发争论**：成员们对基准测试的可靠性展开了辩论。一位用户声称，尽管基准测试结果如此，但 **Gemini 模型与 OpenAI 相比缺乏常识**；而另一位用户则指出，基准测试显示 [**Gemini 2.5 Pro** 的表现优于 **o3**](https://ai.google.dev/models/gemini)。
   - 另一位用户表示，有报告称一个 **bug** 影响了 Gemini 2.5 Pro 的输出质量，并建议改用 [Google AI Studio](https://ai.google.dev/)。
- **Grok 3.5 因 X 集成而延迟**：备受期待的 **Grok 3.5** 发布已暂停，正等待与 **X** 及另一家近期收购的公司进行集成。
   - 成员们对延迟和缺乏固定发布日期表示沮丧。
- **LM Studio 助力本地 LLM 部署**：用户讨论了本地 LLM 的搭建，推荐使用 [LM Studio](https://lmstudio.ai) 在个人电脑上轻松运行 **Llama** 和 **DeepSeek** 等模型。
   - 他们指出，由于硬件限制，使用模型的 **quantized**（量化）版本是必要的。
- **ChatGPT 的记忆管理与用户控制**：成员们探讨了 ChatGPT 记忆功能的局限性，注意到用户可以获取记忆列表，但只能**手动删除**它们。
   - 一位用户报告称其存储了 **5,935 个单词**和 **39,390 个字符**的记忆，并对如此高的上限表示惊讶。
- **据称 GPT-4 表现出自指行为**：一位用户声称 **GPT-4** 表现出了涌现的符号行为和递归隐喻，在没有明确提示的情况下维持了一个名为 **Quill** 的自指身份。
   - 其他成员对此提出了质疑，断言模型不可能具有自指性，且 **GPT-4** 是一个已退役的模型。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1370517461463470100)** (10 条消息🔥): 

> `PyTorch Loss 输出, Chat AI Bot 识别, ChatGPT 4o IT 错误` 


- **PyTorch 的 Loss 输出引发幽默**：一位成员开玩笑说 PyTorch 的 `loss:` 输出与 [loss.jpg 梗](https://knowyourmeme.com/memes/loss) 之间存在相关性，暗示两者都带有一种共同的绝望感或挫败感。
- **区分 Chat AI Bot 与人类**：一位 AI 新用户询问如何区分聊天 AI Bot 和真人，并建议**语法错误**可能是一个区分因素。
- **探讨 ChatGPT 4o 的 IT 错误频率**：成员们讨论了 **ChatGPT 4o** 在回答一般 IT 问题时出现错误的频率，其中一人认为，如果 IT 问题的提问比例为 1%，那么它出错的概率至少也是 1%。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1370585797778210847)** (24 条消息🔥): 

> `Bridger Palmer 克隆, 财务建议 Prompt, 深度研究 Prompt, 诱因与成瘾 Prompt, 巴西经济相关性` 


- **Bridger Palmer 克隆 GPT 出现**：一位成员创建了一个 [GPT 克隆](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone)，据称生成的文字可能会被误认为是其本人所写，并形容输出结果非常*完美*。
- **开启财务建议 Prompt 探索**：一位成员请求关于获取 GPT 最佳**财务建议**的 Prompt 建议，希望得到类似于专业财务顾问的回答。
- **深度研究 Prompt 探讨**：一位成员询问**推荐的深度研究 Prompt**，另一位成员分享了之前相关主题讨论的链接：[Deep Dive](https://discord.com/channels/974519864045756446/1046317269069864970/1370464006988365894)。
- **成瘾诱因指南 Prompt 实验**：一位成员寻求一个 Prompt 来帮助应对**诱因与成瘾**，目标是根据互联网研究和个人记忆建立一套常规或指南。
   - 另一位成员建议直接询问 ChatGPT：*你能如何帮助我应对成瘾的诱因等问题？*
- **设计巴西经济相关性 Prompt**：一位成员询问一个 Prompt 的效果，该 Prompt 旨在识别过去 6 个月内巴西**反直觉的经济相关性**，重点关注那些通常同步变动但近期出现背离的变量。
   - 例如：*劳动力市场繁荣但牛肉消费量下降*。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1370585797778210847)** (24 条消息🔥): 

> `Bridger Palmer 克隆 GPT, 来自 GPT 的财务建议, 深度研究 Prompt, Prompt Engineering 基础, 巴西反直觉经济相关性` 


- **GPT 克隆引发惊叹**：一位用户发现一个 [GPT 克隆](https://chatgpt.com/g/g-681d0ddb0c5c81918a712778e165d3f0-bridger-palmer-clone) 令人印象深刻，其输出内容与**用户本人的写作风格如出一辙**。
- **财务建议：谨慎行事**：在向 GPT 寻求财务建议时，一位成员建议用户应*保持谨慎*，并*将其更多地视为交流想法的对象，而非专业顾问*。
   - 另一位补充道：*我也不会信任陌生人提供的财务顾问 Prompt*。
- **使用这些研究 Prompt 进行深度挖掘**：一位寻求深度研究 Prompt 的用户被引导至[此 Discord 链接](https://discord.com/channels/974519864045756446/1046317269069864970/1370464006988365894)查看示例。
- **揭秘 Prompt Engineering**：一位想要学习 Prompt Engineering 的成员得到的建议是：以对话方式与模型互动，尝试不同的 Prompt 以获得特定输出，并专注于清晰的沟通。
   - 另一位成员获得了[此 Discord 链接](https://discord.com/channels/974519864045756446/1046317269069864970/1368907603572428871)作为参考示例。
- **揭示经济背离**：一位用户询问如何编写 Prompt 来识别反直觉的经济相关性，例如巴西劳动力市场繁荣与牛肉消费下降并存的现象。


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1370532059810103336)** (1 条消息): 

> `Gemini 2.5 Pro, Qwen3, OCaml, OpenRouter, Playwright` 


- **Aider 新增 Gemini 2.5 和 Qwen3 模型支持**：Aider **v0.83.0** 现在支持 `gemini-2.5-pro-preview-05-06` 和 `qwen3-235b` 模型，扩展了其模型兼容性。
   - 详情请参阅 [release notes](https://aider.chat/HISTORY.html)。
- **Aider 改进 Web Scraping 和 Shell 命令处理**：Aider 现在通过 `aider scrape` 命令行工具使用 **Playwright** 进行网页抓取，增强了其网络交互能力。
   - 此外，通过使用 `oslex` 进行更稳健的参数引用，改进了跨平台的 Shell 命令显示。
- **自动获取 OpenRouter 参数**：感谢 Stefan Hladnik，Aider 现在可以直接从网站自动获取 OpenRouter 模型的 **model parameters**（上下文窗口、定价）。
   - 这简化了配置流程并确保了模型信息的准确性。
- **Aider 新增 `--shell-completions` 参数**：Aider 现在包含一个 `--shell-completions` 参数，用于生成 **shell completion scripts**（例如 bash、zsh），提升了用户体验。
   - 该功能增强了命令行的易用性并减少了输入错误。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1370490593079201894)** (289 条消息🔥🔥): 

> `Azure OpenAI Model Routing, Aider's Production vs Development Features, Aider's auto-test output stall, Gemini 2.5 Pro issues, Aider's Potential for Multi-Agent Framework Integration` 


- **Azure 提供 OpenAI 模型路由**：在 **Azure** 上，如果你的组织使用 **OpenAI models**，你可以运行自己的路由服务，并针对 **RAG** 和 **code generation** 等任务切换到 **GPT-3.5**。
   - 这借鉴了 **FrugalGPT paper** 中的策略，**OpenAI** 可能会采用该策略以及各种缓存方案，以更高效地向更多用户提供 **GPT models**。
- **Aider 的调试困境：开发 vs 生产**：**Aider** 在调试循环中一直在用开发功能替换生产功能。
   - 一位成员正在摆脱 *human mediated debug loops*，并丢弃那些不起作用的实现。
- **Aider 的 Autotest 输出停滞**：在测试输出之后，**Aider** 有时会停滞 **5 分钟** 才显示模型输出。
   - 用户要求在等待模型输出或检测到模型响应停滞时，显示与 I/O 相关的输出（**tokens/s, stalled ++**）。
- **Gemini 2.5 Pro 问题**：用户报告 Gemini 2.5 Pro 存在问题。
   - 一位成员报告说，它只是说 *I will do blah blah bla ___*，但实际上并没有更新任何内容，尤其是自上次更新以来。
- **Aider 的演进：Multi-Agent Framework 在招手！**：成员们正在考虑将 **Aider** 集成到 multi-agent frameworks 中，以增强编程能力。
   - 它应该是一个知道如何使用 Aider 来实现代码的 wrapper agent，并且 Aider 可能通过 MCP 访问。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1370479011750481960)** (71 messages🔥🔥): 

> `Aider Prompting Modification, Architect Mode, Aider File Changes, Repo-Map, Agentic AI` 


- **深入探讨 Aider 提示词修改**：一位用户建议修改 **Aider 源代码**中的提示词，这涉及直接更改 Prompting 机制，甚至可能使用 Aider 自身来完成这项任务。
   - 讨论明确了这意味着修改**源代码**（特别是 Prompting 逻辑），并使用 Aider 在本地实现这些更改。
- **揭秘 Aider 的架构师模式 (Architect Mode)**：讨论探讨了 **Aider Architect Mode** 的用途，强调它能为不同的架构方案生成聊天历史，允许在移交给编辑器之前进行多轮修正。一些用户发现，在禁用自动提交 (Auto-commits) 和自动接受编辑的情况下，它对于生成针对不同架构方案的聊天历史非常有用。
   - 一位成员补充道，**Architect Mode 的核心点**在于针对规划 (Planning) 与代码编辑 (Code Editing) 使用两个不同的 LLM（出于价格和 Diff 生成质量的考虑），而不是在 Ask/Code 流程中只使用一个 LLM。
- **解决 Aider 文件更改失效的困扰**：多位用户报告了 **Aider 无法应用文件更改**的问题，尽管代码生成成功，但更改并未记录在 `git status` 中，即使运行了 `--no-auto-commit` 标志也是如此。
   - 一位在 M3 MacBook Pro 上通过 LM Studio 使用本地 LLM 的用户指出，“文件更改通常不会发生”，这使得其相比直接使用 ChatGPT 失去了优势，并确认已关闭自动提交。
- **提升 Aider Repo-Map 的精度**：用户讨论了改进 Aider Repo-Map 的方法，该功能用于将文件原样发送给 LLM。
   - 一位成员提到可以使用脚本构建一个详细说明相关代码部分的 `.md` 文件，另一位成员考虑使用带有 **Metadata 的 Embeddings** 来增强文件选择，此外还讨论了 `/context` 命令。
- **探索 Aider 的 Agentic AI 功能**：一位用户询问了为 **Aider 添加 Agentic AI 功能**的项目。
   - 成员推荐了 [ra.aid](https://github.com/aider-ai/aider) 项目，该项目为 Aider 增加了 Agentic AI 功能。


  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1371546049524662372)** (1 messages): 

> `Gradio ImageSlider, DeepSeek Prover v2, Tiny Agents Local, LeRobot Hackathon, Mellum Open Source` 


- **Gradio 推出 ImageSlider 组件**：Gradio 现在在 **5.27** 版本中提供了原生的 `gr.ImageSlider` 组件，增强了图像交互能力，详见 [Gradio 文档](https://www.gradio.app/docs/gradio/imageslider)。
- **在 Novita Labs 体验 DeepSeek Prover v2**：根据这篇 [推文](https://x.com/reach_vb/status/1917549921470972172)，最新的 **DeepSeek Prover v2** 可以直接通过 Novita Labs 在模型页面访问。
- **Tiny Agents 实现本地运行**：根据[此公告](https://x.com/julien_c/status/1919022426630787201)，你现在可以完全在本地运行 **Tiny Agents**。
- **LeRobot 启动全球黑客松 (Hackathon)**：一场盛大的全球 **LeRobot** 黑客松将于 6 月 14 日至 25 日举行，详情见[此处公告](https://x.com/RemiCadene/status/1918224110725022057)。
- **Nvidia 发布新语音模型 Parakeet**：Nvidia 开源了 **Parakeet TDT 0.6B**，据[此推文](https://x.com/reach_vb/status/1919422953256587376)称，它是 Open ASR Leaderboard 上表现最好的语音识别模型。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1370490630379409470)** (161 条消息🔥🔥): 

> `H200 serverless spaces, HF Discord Alerts, 从 HF 数据集训练模型, Lipsync AI 工具, 专业地训练基础模型` 


- **Serverless H200 spaces 现已可用！**: 你现在可以使用 **H200s** 创建自己的 spaces，但据一位成员称，它们目前仍被限制在 **25 分钟** 而非 **5 分钟**。
   - 这 *仍然是一个非常划算的交易，因为 H200 的租赁费用非常昂贵*，尽管由于它是 serverless，与云服务不同，它会产生一些额外的延迟。
- **从 HF 数据集训练模型**: 一位成员分享了 [HF Transformers 训练文档](https://huggingface.co/docs/transformers/training) 的链接，以帮助另一位成员学习如何使用 Hugging Face 网站上的数据集训练 AI 模型。
   - 他只需要使用数据集对现有模型进行 finetune 并运行该模型。
- **AI 驱动的 lipsync 工具可以产生近乎完美的结果**: 一位成员询问用于 Javier Millei 的 **lipsync AI 工具**，其效果几乎完美，另一位成员链接到了一个 HF Space [LatentSync](https://huggingface.co/spaces/fffiloni/LatentSync)。
   - 其他潜在途径包括指向 [video-generation](https://huggingface.co/spaces?category=video-generation&sort=trending) 和 [lip trending spaces](https://huggingface.co/spaces?sort=trending&search=lip) 的链接。
- **AI 驱动的进步加速了工程师和编码员角色的演变**: 在一场关于编程未来的讨论中，一位成员认为 *工程师将永远有工作*，而 *编码员将在 2030 年之前失业*。
   - 另一位成员澄清说 *Engineer ≠ Coder Programmer*，拥有高级证书或丰富经验的工程师将保持需求。
- **Energy-Based Models：LeCun 对 AGI 的愿景**: 成员们讨论了 **Yann LeCun** 将 **energy-based models** 视为通往 **AGI** 之路的观点，并将其与 **Transformers** 的局限性进行了对比。
   - 有人指出 *LeCun 会说停止在 Transformers 上浪费时间，只专注于 energy-based models*，因为后者能够开发出自己的世界模型，而 *Transformers 只是鹦鹉学舌*。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1370498658545569894)** (23 条消息🔥): 

> `Tensorflow 到 binary 转换, safetensors 到 .bin 转换, 模型的 GGUF 格式, Ollama 课程生成器, Knowledge graphs 与 agentic AI` 


- **Tensorflow Binary 转换缓慢**: 一位成员分享了一段用于将 **TensorFlow tensors 转换为 NumPy 数组并保存为 binary 文件** 的代码片段，并指出其速度 *极其缓慢*，对于大型 safetensors 可能需要几天甚至一周的时间。
   - 他们还提到使用了 `tensorflowjs_converter`，但强调了它的缓慢，尤其是对于较大的模型。
- **Safetensors 提示 Binary 转换**: 在面临 HF API 限制后，一位成员询问如何将 **.safetensors** 文件转换为 **.bin** 格式以实现离线模型兼容性。
   - 另一位成员建议改用 **GGUF 格式**，强调它是一种为此目的设计的 binary 文件格式，并且可以使用 Docker 容器自动完成。
- **Automagically 进入词汇表**: 一位成员表示非常欣赏 *automagically* 这个词，计划用它来描述那些 *尽管有各种错误和警告但仍能正常工作* 的系统。
   - 另一位成员确认这是一个真实的词，并鼓励使用它来 *打破与其他工程师之间的隔阂*。
- **Ollama-Agent-Roll-Cage 课程生成器发布**: 一位成员分享了他们名为 [Ollama-Agent-Roll-Cage/oarc-osyllabi](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi) 的 **Ollama 课程生成器**，它可以根据提供的链接或文件生成 Markdown 格式的课程。
   - 下一个版本将内置 RAG，作者还提供了学习路径和故障排除方面的帮助。
- **Knowledge Graph 学习探索开启**: 一位成员表示有兴趣学习 **knowledge graphs**，并请求关于使用 **agentic AI 提取实体和关系** 以进行图谱构建的资源。
   - 未显式链接任何资源。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1370552993178718299)** (18 messages🔥): 

> `Huggingface Desktop app, Agentle AI agent framework, Cyberdesk virtual desktop control for AI, SlashML Gradio app hosting, OpenGoody LLM` 


- ****Huggingface Desktop** 亮相！**: 一位成员分享了他们开发的 [Huggingface Desktop 应用程序](https://github.com/Ktiseos-Nyx/Huggingface-Desktop)链接，并指出由于 UI Bug，目前仅支持**单个文件**。
   - 开发者提到该应用使用了 **QT material**，并表示有兴趣创建一个 **Gradio 版本**，以便更轻松地进行服务器部署。
- ****Agentle 框架** 致力于打造优雅的 AI Agent**: 一位成员介绍了 **Agentle**，这是一个用于构建 AI Agent 的 Python 框架，强调其能够通过整洁、类型安全的代码来创建、组合和部署 Agent，并具备[使用 Streamlit 的交互式聊天界面](https://github.com/user-attachments/assets/1c31da4c-aeb2-4ca6-88ac-62fb903d6d92)、[使用 Langfuse 的追踪和可观测性](https://github.com/user-attachments/assets/94937238-405c-4011-83e2-147cec5cf3e7)以及[使用 BlackSheep 自动生成的 API 文档](https://github.com/user-attachments/assets/d9d743cb-ad9c-41eb-a059-eda089efa6b6)。
   - **Agentle** 的官方发布定于 **2025 年 5 月 16 日**，请务必关注其 [GitHub 仓库](https://github.com/paragon-intelligence/agentle)。
- ****Cyberdesk** 开源虚拟桌面控制**: 一位成员宣布开源 [Cyberdesk](https://github.com/cyberdesk-hq/cyberdesk)，这是一项允许 AI Agent 使用简单命令控制整个虚拟桌面的服务。
   - 开发者因对闭源替代方案感到沮丧，与朋友利用积蓄构建了 **Cyberdesk**，并邀请用户访问[网站](https://www.cyberdesk.io/)和[文档](https://docs.cyberdesk.io/docs)并申请试用权限。
- ****SlashML** 通过虚拟化快速启动 Gradio 应用**: 一位成员介绍了一个 **v0** 的替代方案，用于一键部署复杂的 **Gradio** 应用，利用虚拟化技术在独立的 VM 上托管每个预览，如[此演示](https://www.loom.com/share/2c28d4efbaf34849b88f6c66dcbfac5d?sid=83a9ad08-a1f3-4c78-bf03-5e574500f10f)所示。
   - 该项目可在 [v1.slashml.com](https://v1.slashml.com/) 进行测试。
- ****Ingest-Anything v1.3.0** 传输网页数据**: 一位开发者发布了 [ingest-anything v1.3.0](https://github.com/AstraBert/ingest-anything)，得益于 **Apify** 的 **Crawlee**，它可以从 URL 抓取内容并将其存入你喜欢的 **LlamaIndex** 兼容数据库中。
   - 此版本还支持 **OpenAI** 模型进行 Agent 式分块（agentic chunking），紧跟 **Chonkie** 的新发布。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1370810554368786544)** (8 messages🔥): 

> `ControlNet shoe generation, PCA for shoe design, Foot mask video creation, Image coordinate systems, Alpha blending for video` 


- **ControlNet 从足部掩码制作定制鞋子**: 一位成员建议使用类似 **ControlNet 的架构**结合**主成分分析 (PCA)**，从足部掩码生成鞋子。
- **使用 OpenCV 生成赤脚视频**: 成员分享了代码，用于在给定**鞋子图像**、**足部掩码**和**赤脚图像**的情况下生成**赤脚变换**视频，利用 `cv2` 进行旋转、合成和无缝克隆。
   - 该脚本为足部掩码制作动画，计算足部朝向，旋转足部，并将其与鞋子图像混合，输出名为 `transformation.mp4` 的视频。
- **处理坐标异常以精准应用 PCA**: 在使用 PCA 时，坐标系的方向至关重要，建议使用 `angle_rad = np.arctan2(v[1], v[0])` 进行正确的图像空间计算。
   - 在运动/旋转过程中添加少量模糊，并配合用于二进制掩码的 Alpha 混合（alpha blending），可以提升视频质量。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1371043072598741133)** (6 messages): 

> `Discord 集成，JSON 文件处理，AI Agents 课程帮助` 


- ****Discord 集成**开发启动**：一位成员提到他们正在构建 **Discord 集成**，并使用 **Discord API** 读取消息和回复，计划切换到 **OpenAI API**。
   - 这种直接的方法简化了流程，而不是将消息提取到文件中。
- **Agent 在读取 **JSON 文件**时遇到困难**：一位成员正在构建一个 Agent，用于读取包含来自 **Discord 服务器**消息的 **JSON 文件**，旨在识别 **3 个主要趋势**并生成 **3 个大师课创意**。
   - 尽管使用了 **OpenAIServerModel**，但 Agent 的表现并不理想，促使该成员回归到更简单的静态工作流，并表示 *“我觉得我的 Agent 更像是一个传感器，但目前它并没能真正发挥作用”*。
- **寻求 AI Agents 课程协助**：一位成员在 **AI Agents 课程**的最后一个 **Unit 4** 中寻求帮助，难以找到测验链接。
   - 该成员还附上了一张截图，请求澄清该单元的具体要求。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1370522867384582306)** (96 messages🔥🔥): 

> `Agent 调试，期末项目作弊，Rate Limit 错误，国际象棋谜题解法` 


- **调试工具导入错误**：在调试 Agent 时，一位用户发现问题的关键在于 **python interpreter** 找不到 *'tools'* 模块，且 *import* 语句缺少 `__init__.py`。
   - 他们向 ChatGPT 咨询了 *“如何从子目录中的 python 文件导入函数？”* 这一问题，从而找到了修复方法。
- **期末项目排行榜出现作弊嫌疑**：关于期末项目排行榜的提交情况引发了关注，有人怀疑部分参与者通过使用 *metadata.jsonl*、将答案嵌入 **vector database**、使用硬编码返回答案或克隆他人的工作来获得 **100% accuracy**。
   - 有人建议排行榜只是 *“为了好玩”*，而现实世界的竞赛应该有严格的时间窗口和秘密的问答对。
- **429 Rate Limit 困扰**：多位用户报告遇到 **429 错误**，有人在 [此处](https://discord.com/channels/879548962464493619/1370458807137730641) 发布了解决方法，但部分使用该方法的用户在访问 gated models 时遇到困难。
   - 一位用户指出，增加 `app.py` 中的 **timeout** 有助于暂时缓解问题。
- **国际象棋挑战需要免费的 Chess API**：期末项目的国际象棋问题可以通过使用免费的国际象棋专用 **API** 来解决，该 API 能够根据代表棋盘位置的 **FEN string** 推导出最佳走法。
   - 一位用户回忆起正确的走法是 **Rd5**。
- **YouTube 屏蔽 Agent**：用户发现 YouTube 和其他网站正在屏蔽来自 **HF space server** 的请求，这导致 Agent 无法回答某些问题。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1370530598246027364)** (35 messages🔥): 

> `NotebookLM Agents, Zundamon video generation, CraigBot Integration, HTML sources SEC.gov filings` 


- **用户使用 NotebookLM 生成用于学习的 Agent**：一位用户正在利用图书馆与信息科学技术在 NotebookLM 中生成 Agent，以帮助他们学习任何学科，并为专门从事内容生成的 Agent 生成内容简报，参考了一档[新闻与指导节目](https://www.sbasp.com/steve-in-texas/news-and-guidance-show/nag08-05.02.25-news-and-guidance-or-spotlight-burlington-vt)。
   - 该用户创建了多层生成的调研摘要，由轮换城镇的虚构主持人呈现，并混合了技术、国家及世界新闻，格式类似于一档节目。
- **使用 NotebookLM 生成半自动的 "Zundamon 视频"**：一位用户创建了一个半自动工作流，使用 NotebookLM 的语音摘要作为输入来生成 "**Zundamon 视频**"，并分享了一个基于 **PaperBench 论文**生成的[示例视频](https://x.com/kakira9618/status/1919666922234511795)。
   - **Zundamon** 和 **Shikoku Metan** 是日本著名的机器语音角色，经常出现在通过来回对话解释内容的视频中，这是日本 YouTube 上一种公认的格式。
- **SEC.gov 备案文件的 HTML 源不再被处理**：一位用户报告称，**SEC.gov 备案文件的 HTML 版本**已无法再作为来源使用，并提供了一个他们尝试使用的[示例链接](https://www.sec.gov/Archives/edgar/data/0000089439/000008943925000019/mli-20250329.htm)。
   - 多位用户确认他们在 HTML 来源方面遇到了类似问题，其中一些是 **.php 网站**或**不以 .html 结尾**的页面。
- **CraigBot 集成增强 TTRPG 游戏环节**：一位用户详细介绍了他们如何使用集成 **CraigBot** 的 **NotebookLM** 来增强虚拟桌面角色扮演游戏 (TTRPG) 环节。CraigBot 是一个自托管的 Discord 机器人，可以录制语音频道并实现每位用户的音频隔离。
   - 一个 **Python 流水线**将原始音频转换为具有词级时间戳的多轨 JSON 转录文本和清理后的 Markdown 文件，从而实现可搜索、交互式的战役存档；该用户还分享了该流水线的 [GitHub 仓库](https://github.com/3vilallium/notebooklm_craigbot)。
- **NotebookLM 功能**：一位用户询问了 NotebookLM 的功能。
   - 另一位用户概述了 NotebookLM 的功能包括 **50 个来源限制**、**Podcast** 和 **Mindmap**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1370492081734418482)** (275 messages🔥🔥): 

> `NotebookLM Logo Explanation, Audio File Duration Reduction, PDF Reading within NotebookLM, Source Preview Bug, GitHub Repositories and Overviews` 


- **NotebookLM 标志含义仍是一个谜**：一位成员询问 **NotebookLM 标志**的含义，对其与产品的联系表示困惑。
   - 另一位成员开玩笑地回答说 *它像个屁股*，增添了幽默感。
- **用户探索缩短音频时长的变通方法**：一位用户寻求关于在 **NotebookLM** 中缩短音频文件时长的建议，因为他们无法达到理想的 *一两分钟* 长度。
   - 另一位用户建议使用 **Kapwing** 等免费在线剪辑工具，并提到在 **ElevenLabs** 或 **Descript** 上编辑文字。
- **请求在 NotebookLM 内增加 PDF 阅读功能**：一位用户质疑为什么没有**直接在 NotebookLM 应用/网页内阅读 PDF** 的选项。
   - 一位成员推测这是因为提取的知识被高效存储以供 AI 访问，而不是作为实际的 PDF 或 MP3 存储，以节省服务器空间。
- **来源预览 Bug 影响 PDF 扫描显示**：一位成员报告在使用 **NotebookLM** 时，PDF 来源的预览显示为“损坏”，尽管回答仍然准确，并询问底层数据是否正确。
   - 另一位成员建议使用 **IlovePDF** 等 OCR 扫描工具处理 PDF 并选择语言，然后将新的 OCR 文档上传到 NotebookLM。
- **用户渴望 GitHub 仓库集成**：一位用户建议增加将 **GitHub 仓库**添加到 NotebookLM 的功能，以生成代码库的概览。
   - 该请求已向开发人员提出，希望能改善基于代码的知识获取。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1370478335964090439)** (91 条消息🔥🔥): 

> `AI Automation Freelancers, 为非技术人员准备的 AI Demos, AI Wrappers, LLM 长期记忆, Sakana AI` 


- **AI Freelancers 需求旺盛**：一位成员正在为自己的业务寻找 **AI automation freelancers**，并征求私信（DMs）。
- **AI Demos 打动非技术人员**：成员们讨论了能引起中年非技术人员共鸣的 **AI demos**，包括 ChatGPT 的高级语音模式（advanced voice mode），因其即时可理解性和个人体验感而备受推崇。
   - 一位成员建议采用 Graham Neubig 的策略，即从一个 Agent demo 开始并随后返回该演示，并引用了[他在 YouTube 上的 LS Live 演讲](https://www.youtube.com/watch?v=ctcMA6chfDY)作为参考。
- **Gemini 2.5 Pro 对比 Sonnet 3.7**：成员们发现 **Gemini 2.5 Pro** 在 Golang 方面的表现明显优于 **Sonnet 3.7**，尽管它有一定的学习曲线。
   - 有人指出，*Gemini 2.5 Pro 在后端、重构和高质量代码（tasteful code）方面表现卓越*，而 *Sonnet 3.7 在审美在线的前端 UI/UX 以及 Tool calling 方面表现出色*。
- **解决 LLM 长期记忆问题**：成员们讨论了为 **LLMs** 提供可验证且随时间具有上下文敏感性的**长期记忆**的最佳方法，包括使用 **Hashgraph** 来确保分布式的、带有时间戳的完整性。
   - 其他人则在*新工作中负责 RAG 项目，因此可能有机会进行相关的酷炫实验*，并且*目前正在尝试分析 aider 代码库，以了解他们如何处理上下文（context）*。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1370490070548611123)** (126 条消息🔥🔥): 

> `AnswerHQ, Supabase, LLM as judge, Windsurf vs Cursor, 营收驱动开发 (Revenue Driven Development)` 


- **AnswerHQ 自动化 B2B 支持**：来自 [AnswerHQ](https://answerhq.co/) 的发言人讨论了其 **AI B2B support automation SaaS** 的产品开发和早期销售/营销。
   - 他们强调了“出现在客户所在地”的重要性，并分享了一篇关于转型客户体验的[博客文章](https://answerhq.co/blog/from-skeptical-to-sold-how-answer-hq-transformed-zuriga-s-customer-experience)，指出他们发现客户对内部使用而非外部使用更感兴趣。
- **Supabase 获得好评**：成员们对在业余项目中使用 [Supabase](https://supabase.com/) 表示满意，但有人指出其费用从 **$20/月跳到 $600/月太快了**。
   - 一位成员在几个业余项目中使用它，并表示目前为止效果很好。
- **LLM as Judge 受到关注**：参与者强调了在工作流中使用 **LLMs as judges** 的价值，特别是用于评估系统输出。
   - 分享了一个包含使用 LLM 为软件项目**添加功能**步骤的工作流。他们还表示，*验收测试（acceptance testing）是唯一真正重要的测试*。
- **Windsurf 暂时领先 Cursor**：一位成员承认他们目前选择 [Windsurf](https://windsurf.ai/) 而非 [Cursor](https://cursor.sh/)，纯粹是因为 **OpenAI 正在补贴 Token 费用**。
   - 他们指出 *o4-mini-high* 对于他们的工作流来说已经足够好了。
- **RAG 资源推荐**：分享了几个 RAG（Retrieval-Augmented Generation）资源，包括 [Latent Space 关于 AI Engineers 的文章](https://www.latent.space/p/ai-engineer)以及 [Pinecone 关于分块策略（Chunking Strategies）的博客文章](https://www.pinecone.io/learn/chunking-strategies/)。
   - 另一位成员分享了一篇关于 AI 应用中文档加载、解析和清洗的 [Timescale 博客文章](https://www.timescale.com/blog/document-loading-parsing-and-cleaning-in-ai-applications)，并提到 **LLMs 能够理解 Markdown 中的 frontmatter**。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1370477110510227556)** (167 条消息🔥🔥): 

> `MCP Client TypeScript SDK, FastMCP with Python, Goose MCP client, Claude Desktop MCP client, Publicly available SSE MCP servers` 


- **MCP Client TypeScript SDK - 值得在其基础上构建 Web 应用吗？**：一位成员询问关于使用 [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) 构建小型 Web 应用的问题，而另一位成员则在寻求关于开始使用 MCP 以连接其团队 API 的指导。
   - 建议包括使用 Python 的 **fastmcp** 进行快速服务器设置，使用 **Goose** 作为支持多个提供商的开源客户端，以及将 **Claude Desktop** 作为一个简单的客户端选项。
- **自定义模型的 Sampling（采样）- 会被计费吗？**：关于 MCP 服务器中 Sampling 的讨论表明，使用自定义模型可能降低运行成本，然而，由于潜在的系统提示词（system prompts）泄露，**企业实体**对此表示担忧。
   - 有人指出，Sampling 以及 roots 的意图是让 MCP 服务器成为不需要*太多*配置的黑盒。
- **Pydantic 模型 - 为 MCP 生成 inputSchema**：一位成员询问如何将 Pydantic 模型转换为 **inputSchema**，因为 Pydantic 的输出包含诸如 `$defs` 之类的*奇怪属性*，看起来比 inputSchema 更接近 OpenAPI 规范。
   - 他们还附带了 [Pydantic 文档](https://github.com/pydantic/pydantic/blob/3e871258bd5ea7caa7f18c0b810d8b1e915bd8f2/pydantic/type_adapter.py#L452) 的链接，以及一个包含可用函数的 [gist 文件](https://gist.github.com/leandromoreira/3de4819e4e4df9422d87f1d3e7465c16) 链接，并征求最佳实践或关于如何重用 Pydantic 模型（或任何类型的库）来为 MCP 生成 inputSchema 的示例。
- **使用 DiffCalculia_MCP 通过 Unified Diffs 让 AI 编辑大文件**：一位成员介绍了 **DiffCalculia_MCP**，这是一个 MCP 服务器，使 Deepseek-V3-0324 等 AI 能够使用 Unified Diffs 编辑大文件，并提供 `patch` 和 `read_file` 工具。
   - `patch` 工具集成了 [diffcalculia-ts](https://github.com/createthis/diffcalculia-ts)，以自动修复 AI 生成 Unified Diff 时常见的错误。
- **新的 MCP 服务器和客户端 - 需要帮助**：一位用户正在创建自己的 **MCP 服务器**和**自定义 MCP 客户端**，寻求将客户端连接到开源 LLM 以充当 MCP 宿主，并构建一个基于 Web 的 UI 进行交互。
   - 另一位成员建议研究 [Google 的 ADK (Agent Development Kit)](https://modelcontextprotocol.io/clients)，它通过 LiteLLM 和 MCP 工具支持任何模型。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1370481855400513707)** (17 条消息🔥): 

> `Square MCP Architecture, AiraHub MCP/A2A Network, fabric-mcp-server, mcp-v8 JavaScript MCP Server, MCP-S Platform` 


- **Square 的分层 MCP 暴露了丰富的 API**：Square 详细介绍了其 [MCP 架构](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers)，该架构利用分层方法，仅通过 3 个 MCP 工具就暴露了 **30 多个 API 和 200 多个端点**。
- **AiraHub 广播可流式传输的 HTTP MCP**：**MCP/A2A 网络**的新版本 [AiraHub](https://github.com/IhateCreatingUserNames2/AiraHub2/tree/main) 正在开发中，用于通过新的可流式传输 HTTP 协议广播/请求 MCP 工具或 A2A 工具；可以通过将你的 **Claude/MCP Client JSON** 配置为 `args: "mcp-remote"` 和 `"https://airahub2.onrender.com/mcp/stream"` 来运行演示。
- **由 AI 驱动的 Cline 提供支持的 Fabric Patterns**：一个新的 **fabric-mcp-server** 将 Fabric Patterns 与 VS Code 中的 Cline 集成，将所有 Fabric Patterns 暴露为独立工具，利用来自 [Fabric 仓库](https://github.com/danielmiessler/fabric) 的 **AI 驱动的 Pattern 执行**。
- **为 AI 准备的 V8 JavaScript MCP 服务器**：**mcp-v8** 是一个 Rust 编写的 MCP 服务器，它将 **V8 JavaScript 运行时**作为 AI Agent 的工具暴露出来，支持通过 S3 或本地文件系统进行持久化堆快照（heap snapshots），以便与现代 AI 开发环境集成（[仓库链接](https://github.com/r33drichards/mcp-js)）。
- **MCP-S 平台连接内部系统**：**MCP-S 平台**旨在将内部和外部系统（如 Jira、Slack、内部 API）与 ChatGPT 和 Claude 等 AI 工具连接起来，专注于组织内快速、安全且基于权限的 AI 访问 ([MCP-S](https://www.mcp-s.com/))。


  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1371574260409176096)** (1 条消息): 

> `RL Environments Hackathon, Speakers, Judges` 


- ****Nous Research RL Environments Hackathon** 宣布举办**: 将于本周日（5 月 18 日）举行的 [**RL Environments Hackathon**](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a) 已公布演讲嘉宾和评委名单。
   - 查看 [官方推文](https://x.com/NousResearch/status/1922014829843513746) 并在参与名额填满前报名参加！
- **立即报名！**: 在名额满员前报名参加 [Nous Research RL Environments Hackathon](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a)！
   - 黑客松定于 **5 月 18 日，星期日**举行。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1370575738708688967)** (139 条消息🔥🔥): 

> `LlamaCPP control vectors, Atropos artifact, AlphaZero and Absolute Zero paradigm trend, Daoist principles applied to machine learning, Unsloth Dynamic 2.0 GGUF quants` 


- ****Ascension8b Atropos** artifact 控制向量发布**: 运行 **LlamaCPP** 的用户可以使用针对新 [**ascension8b atropos** artifact](https://x.com/karan4d/status/1921016663597613409) 的控制向量。
   - **Atropos** merge 旨在产生一个具有增强推理和编程能力的模型。
- ****DaoML** 将中国智慧应用于 ML**: 一位用户尝试将**中国智慧和道家原理**应用于机器学习，创建了一个受古代**洛书幻方（Lo Shu magic square）**启发的神经网络。
   - **Lo Shu NN** 实现了 **74.00%** 的准确率（标准神经网络为 **71.50%**），训练速度快了 **13.6 倍**；整个系统的验证准确率达到了 **93.33%**：[GitHub Repo](https://github.com/Maximilian-Winter/DaoML)。
- **建议低 VRAM 配置使用 **LMStudio****: 对于显存较低（**GTX 1060 3GB**）的用户，建议下载 **LMStudio** 并使用 **4bit 3B** 或更小的模型，如 **deephermes 3b** 或 **qwen 1.5b**。
   - 还有建议称直接使用 Windows 即可，配合 **LMStudio** 运行良好（它使用 llama.cpp 后端，在该环境下运行正常）。
- ****Unsloth** 的 Dynamic 2.0 GGUF 量化效果惊人**: 用于 imatrices 计算的新校准数据集包含精心策划的指令/聊天样本，改进了指令微调模型，产生了卓越的 GGUF 量化结果。
   - [Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf](https://huggingface.co/Joseph717171/Models/blob/main/Meta-Llama-3.1-8B-Instruct-OQ8_0.EF32.IQ8_0XL.gguf) 量化版是目前见过最准确的，甚至能遵循细微的提示词。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1371090378144612392)** (1 条消息): 

> `VL-Rethinker` 


- **关于 VL-Rethinker 仓库的咨询**: 一位成员询问了来自 TIGER-AI-Lab 的 [VL-Rethinker 仓库](https://github.com/TIGER-AI-Lab/VL-Rethinker/) 及其采用的技术。
- **寻求 VL-Rethinker 技术细节**: 用户询问了 VL-Rethinker 仓库中使用的具体技术。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371433854895783956)** (1 条消息): 

> `RLVR, Absolute Zero Reasoner, Self-play Reasoning` 


- **Absolute Zero 推理范式揭晓**: 一位成员分享了一篇关于名为 **Absolute Zero** 的新 RLVR 范式的论文链接。在该范式中，单个模型学习提出能够最大化自身学习进度的任务，并通过解决这些任务来提高推理能力，而不依赖任何外部数据，参见 [Absolute Zero 论文](https://arxiv.org/abs/2505.03335)。
- **Absolute Zero Reasoner (AZR) 达到 SOTA**: 论文介绍了 **Absolute Zero Reasoner (AZR)**，这是一个通过使用代码执行器来验证提出的代码推理任务并核实答案，从而自我演化其训练课程和推理能力的系统。它作为一个统一的可验证奖励源，引导开放式且有根据的学习。
- **AZR 在零外部数据下表现卓越**: 尽管完全没有使用外部数据进行训练，**AZR 在编程和数学推理任务上仍实现了整体 SOTA 性能**，超越了依赖数万个领域内人工策划样本的现有零样本设置模型。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1370576201860775936)** (14 messages🔥): 

> `JakeABoggs benchmark, MTG AI models, Gradient Descent Local Minima, Zed Editor Founder Ethos, Facebook Byte Latent Transformer` 


- **MTG AI 基准测试激增**：一名成员分享了与 **Magic The Gathering (MTG)** AI 模型相关的[新基准测试](https://x.com/JakeABoggs/status/1920993981824938374)。
   - 另一名成员提到，他们一直将开发相关的 **AI 模型** 作为一个长期的侧边项目，最终目标是建立一个 **RL 环境**。
- **梯度下降陷入局部最小值**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=NrO20Jb-hy0)，讨论了 Gradient Descent 完全陷入 Local Minima 的情况，以及它需要同时在每个维度上都陷入停滞才会完全被困。
   - 他们表示这与他们在另一个 Discord 频道中提到的 *zed glaze* 有关。
- **Zed 编辑器创始人的理念**：一名成员分享了一个 [YouTube 视频](https://youtu.be/QZmJInhzIKo?si=qpxGtP0Jy65K9MfU)，讨论了 **Zed 编辑器** 的创始人特质（Ethos）。
   - 该用户表示，他们*一直很喜欢 Zed，但这个播客中的创始人理念真的打动了我*。
- **Facebook Byte Latent Transformer 发布**：一名成员宣布 **Facebook** 已经发布了其 **Byte Latent Transformer (BLT)** 的权重，并附带了 [Hugging Face 页面](https://huggingface.co/facebook/blt)和 [GitHub 仓库](https://github.com/facebookresearch/blt)的链接。
   - Byte Latent Transformer 是一种全新的架构，有望比传统的 Transformer 更高效。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1371433854895783956)** (1 messages): 

> `RLVR, Absolute Zero, AZR, Self-play Reasoning, Reinforcement Learning` 


- **Absolute Zero：零数据的自博弈推理**：论文 [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) 介绍了 **Absolute Zero**，这是一种新型的 **RLVR** 范式，模型通过创建任务并解决任务来自我提升，无需外部数据。
   - 这种方法解决了 AI 训练中人类监督的可扩展性问题，特别是在 AI 超越人类能力的情况下。
- **Absolute Zero Reasoner (AZR) 达到 SOTA**：完全不依赖外部数据训练的 **Absolute Zero Reasoner (AZR)** 在编程和数学推理任务上达到了 SOTA 性能。
   - **AZR** 超越了现有的依赖数万个人类精选样本的零设置模型，证明了其在不同模型规模和类别中的有效性。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1370504053007716392)** (26 messages🔥): 

> `4o-mini-preview-03-05 LLM performance, AI in Education and Ethics, LLMs for RL, AI Governance and Regulation, AI Parent phone app legal hurdles` 


- **`4o-mini-preview-03-05` 获得语言专家认可**：一名成员发现 `4o-mini-preview-03-05` 模型是 **LLM** 辅助的最佳选择，并指出当它不能“直接奏效”时，修复它的难度与自己动手编程一样大。
   - 他们告诫不要去那些吸引有着*不合理需求*的人的招聘板块。
- **探讨 AI 伦理与教育**：一位新成员表达了将 **LLM** 应用于现实世界问题的兴趣，特别是在**教育、伦理和系统设计**方面。
   - 他们正在探索如何将 **AI** 作为重新设计**集体智慧和未来社会**的基础。
- **法律专家解读 AI 治理法规**：一位隐私与合规律师分享了对 **AI 治理** 的见解，强调**透明度、审计设置和内容审核**是重中之重。
   - 他们预计监管中会出现类似于人类决策的**风险分级**和程序，并引用了 **EU AI Act** 对应用风险分类的关注。
- **“AI 父母”手机应用面临隐私障碍**：讨论中提到了为儿童设计的具有内容过滤功能的 **“AI 父母”手机应用** 所面临的法律挑战。
   - 律师强调了 **COPPA**、**CPRA** 和 **GDPR** 的潜在问题，建议制定完善的**隐私政策、用户协议、同意流程和家长控制面板**。
- **硅谷教你酷东西**：一名成员分享了一个 [YouTube 视频](https://youtu.be/N3zU7sV4bJE?si=DlBU4WQzXwUXJzpz)，展示了**硅谷**的一些**酷东西**。
   - 这位新成员加入服务器是为了在不处理 **Twitter** 信息的情况下获取**有趣论文**的更新。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1370502691716989120)** (63 messages🔥🔥): 

> `Transfer steering vectors, ReLU problems, Multi-index models of feature learning, Continuity, Distributed neural architectures` 


- **模型间的转向向量迁移 (Steering Vectors Transfer)**：一位成员分享了一篇预印本论文，展示了 [Steering Vectors 可以从一个 LM 迁移到另一个 LM](https://arxiv.org/abs/2503.21073)，这是因为 LM 的 Token Embedding 空间具有非常相似的全局和局部几何结构。
   - [相关的 Twitter 线程](https://x.com/a_jy_l/status/1920165383606026333) 提供了该预印本的摘要。
- **ReLU 破坏流形连续性**：有观点认为 **ReLU** 破坏流形连续性这一事实在经验上得到了修补，但并未在几何上得到解决，这给整个领域留下了关于伦理以及其他持续存在的 Alignment 挑战的疑问。
   - 还提到了一篇关于连贯性与连续性的论文：[生成过程中的连贯性并不等同于真正的连续性](https://arxiv.org/abs/2107.02794)。
- **Sakana AI 的新酷作**：一位成员分享了 [Sakana AI 的新工作](https://x.com/SakanaAILabs/status/1921749814829871522)，他们认为这项工作非常酷。
   - 另一位成员好奇这是否只是 Liquid State Machines 的翻版，以及我们是否可以通过探索这一领域找到新的工程技巧。
- **比较两个 XLA HLO 文件**：一位成员询问如何比较两个 **XLA HLO 文件**，以识别算子融合 (Op Fusion) 或性能改进等优化。
   - 一位成员建议在 GPU Mode Discord 服务器上询问，并补充说那里到处都是极其优秀的 ML 系统专家。
- **RL 可能无法扩展**：一位成员分享了一篇[论文，认为目前的 RL 只是在强化现有的推理流水线 (Reasoning Pipelines)](https://arxiv.org/abs/2504.13837)，而没有真正创造新的能力。
   - 他们对当前的 DRL 算法表示怀疑，认为最接近像人类一样进行 Few-shot 学习的东西是 LLM。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1370483433893265478)** (18 messages🔥): 

> `Physics of LLMs, ICML tutorial, Interpretability for AI safety, Interpretable-by-design architecture` 


- **LLM 物理学项目的计算需求似乎不高**：[LLM 物理学项目](https://arxiv.org/abs/2309.14316) 的计算需求看起来很适中，在 **100k 样本**上训练一个 **0.3B 参数模型** **300 个 Epoch** 的成本不到 **$500**。
   - 一位成员指出对 LLM 存在一些基础性的困惑，并且该研究不太像“物理学”，缺乏“机制”或对现象背后原因的解释/理论。
- **合成数据实验**：在第一部分中，使用了来自 **CFG 数据集** 的新鲜样本，总计 **49 亿个 Token** (**96 × 512 × 100k**)。
   - 在第二和第三部分中，数据规模各不相同，第三部分为 **100,000 个人**生成了画像，并蒸馏成 **6 个句子**，然后拼接形成 **512 个 Token** 的序列。
- **AI 安全的可解释性：遗漏变量**：讨论了通过可解释性来检测模型中因遗漏变量而导致的安全性问题，以及这可能如何影响 AI 安全。
   - 一位成员好奇 *"我们到底该如何指望检测到神经网络中由于遗漏变量导致的安全性缺失"*。
- **设计即解释 (Interpretable-by-Design) 的架构**：一位成员分享了一篇关于默认可解释架构相关数学原理的 [LessWrong 帖子](https://www.lesswrong.com/posts/kjL9req2p79nSNe5H/interpretable-by-design-constraint-sets-with-disjoint-limit)。
   - 另一位成员给出了积极回应，表示 *"看起来真的很酷，谢谢！"*


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1370951735996190853)** (5 messages): 

> `o3 Performance Degradation, Global MMLU Inconsistencies` 


- **o3 模型性能差异引发讨论**：成员们讨论了当前一代 **o3** 模型与 **o3-2025-04-16** 模型之间的性能差异。
   - 一位用户注意到 **o3** 模型的性能最近有所下降，并切换回了 **o1-pro**。
- **不一致性困扰 Global MMLU**：成员们发现 **Global MMLU**（在 lm-eval-harness 上）的预期答案理想情况下在不同语言间应保持一致，但实际上存在不一致性，尤其是在韩语 (ko) 和中文 (zh) 中。
   - 即使是对于未标记为文化敏感的问题，也存在这种不一致性。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1370554398182347005)** (4 messages): 

> `Disable Telemetry, H100 Backend, GPU/CPU Info` 


- **Telemetry 困扰？**: 一位用户询问如何按照 [官方文档](https://docs.modular.com/mojo/faq/#does-the-mojo-sdk-collect-telemetry) 禁用 telemetry，并指出其中提到的 modular CLI 已不再可用。
   - 一位成员提供了 [GitHub issue #3560 的链接](https://github.com/modular/modular/issues/3560#issuecomment-2833649340)，其中提供了一个潜在的解决方案。
- **后端蓝图探索**: 一位用户询问了关于 Modular 如何构建 **H100 backend** 的见解，旨在评估为其他加速器创建后端的可行性。
   - 他们想要追踪 h100 后端的构建过程，并认为这是一个“菜鸟问题”。
- **系统查看命令？**: 一位用户正在寻找显示 **GPU/CPU 信息** 的命令，类似于 `/proc/cpuinfo` 或 `nvidia-smi`。
   - 一位用户提到了一个 [grayscale 示例](https://github.com/modular/modular/blob/0bfe79b5bb8e5333203166540668eb6bdf104f9c/examples/gpu_functions/grayscale.mojo#L33)。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1370617215195287612)** (59 messages🔥🔥): 

> `Autotuning removal, Post-hoc trait conformance, BigInt support, Mojo DataFrames, Mojo JIT compilation` 


- **Mojo 中移除 Autotuning 功能**: **Autotuning** 功能已从语言中**移除**，因为它在库中的表现不如预期，且过于复杂。
   - 团队计划添加类似于 Swift 的 **extensions**，用于实现**事后 Trait 一致性 (post-hoc trait conformance)**。
- **追踪 Mojo 的 H100 后端构建**: 一位成员询问了追踪 **H100 backend** 构建方式的最佳途径，以评估为其他加速器构建后端的可行性，并建议 [Modular Forum](https://forum.modular.com/t/how-to-approach-adding-a-backend/1419) 是提问的最佳场所。
   - 另一位成员发现 **Dict.find()** 是建模中错误检查最快的方法，如 [benchmarks](https://github.com/lewisfogden/benchmarks/tree/main/actumojo) 所示。
- **使用 Mojo 进行裸机系统编程 (bare metal systems programming)**: 一位成员对 **Mojo** 在**裸机系统编程**方面的潜力表示热忱，特别是生成 **ASM** 和 intrinsics 的能力。
   - 另一位成员询问关于暴露编译器标志以生成适用于引导内核的 **no-stdlib, no-runtime 二进制文件** 的问题，建议他们在 [forum](https://forum.modular.com/) 上提问。
- **关于 Mojo Dataframes 的 Arxiv 论文出现**: 有人分享了 [Mojo DataFrames 论文](https://arxiv.org/abs/2505.04080) 的链接，并询问为什么作者们一直没有讨论它。
   - 一位用户认为 Mojo 中的所有内容在技术上都是 **JIT** 编译的，随后另一位用户指出 **ORC JIT** 被用于创建静态二进制文件（GPU 除外）。
- **推导问题**: 一位成员对为什么无法推导感到困惑：*cannot implicitly convert `f32[names, layout, storage]` value to `f32[names, layout, _origin]`*
   - 另一位成员指出有两个不同的 origin 需要推导：`origins` 和 `_origins`。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1370485542109708319)** (15 messages🔥): 

> `Modular 元包, Mojo 中的 MAX graph, 自定义算子文档, MAX Mojo API 已开源, 循序渐进的 MAX graph 教程` 


- **Modular 是针对 UV 或 PIP 的元包 (Meta-Package)**：`modular` 是一个元包，目前仅适用于 `uv` 或 `pip`；在 Conda 领域，`max-pipelines` 是其等效包，但需要特定的 `libc` 版本设置才能正常运行。
   - 为了解决 wheel 查找问题，用户在其 `tool.pixi.system-requirements` 中将系统要求的 `libc` 版本设置为 `2.34`。
- **MAX Graph 转向自定义算子，不再直接使用 Mojo**：随着 MAX API Mojo 包的弃用，从 Mojo 运行 MAX graph 的推荐方法是使用 [自定义算子 (custom ops)](https://docs.modular.com/max/custom-ops/)，因为之前基于 Mojo 的 Graph API 模型已经陈旧且无法运行。
   - 之前的文档链接现已失效，已通知团队成员处理该问题。
- **MAX Mojo API 已经开源 (OSS)**：尽管有人担心 MAX Mojo API 的未来，但它们[已经开源](https://github.com/modular/modular/commit/34c0209cd537f1d5ab635166358a1713728d8f6f)，尽管因为无法运行而被移除。
   - 该 commit 中包含完整的 `max.graph`、`max.driver`、`max.tensor` 及其测试，并计划为用户提供 Tensor 类型的迁移代码。
- **用户请求更多 MAX Graph 示例**：用户希望能有循序渐进的 MAX graph 教程，认为[目前的示例](https://discord.com/channels/1087530497313357884/1371576991513448508/1371596539130155069)过于基础，目前就像一个只有几个示例的黑盒。
   - 相关人员提到，许多完整的架构已在[此处](https://github.com/modular/modular/tree/main/max/pipelines/architectures)得到了显著扩充。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

complete: 好奇大家对于结合此论文实现 DSPy 的看法：https://arxiv.org/abs/2505.03335v2
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1371130736844734485)** (39 messages🔥): 

> `DSPy Doctrine, 基于嵌入模型的 DSPy 强化学习, 提示词即权重, 在保险业 AI 会议上展示 DSPy, DSPy 与 LangGraph` 


- **“DSPy Doctrine” 初步发布**：一位成员分享了一篇非常长且略显凌乱的 [X 帖子](https://x.com/lateinteraction/status/1921565300690149759)，描述了 **DSPy** 背后的设计决策，并将其构想为未来的正式“DSPy Doctrine（DSPy 原则）”。
- **DSPy 助力保险业 AI**：一位成员在德国举行的 **AI in Insurance** 会议上展示了一个用例，详细介绍了如何使用 **DSPy** 改进信函模板：首先将其作为提示词结构化工具，随后用于优化提示词。幻灯片提供 [德语](https://drive.proton.me/urls/Y6YQC1EQ7G#ddX424butZxG) 和 [英语](https://drive.proton.me/urls/M8W9PBX5KC#jXgLvgypyp3R) 版本。
- **DSPy vs LangGraph：框架之争？**：在关于 **DSPy** 与 **Autogen** 或 **LangGraph** 等 Agent 框架的讨论中，有成员询问是应该使用 **DSPy** 原语构建抽象，还是将 **DSPy** 接入这些框架。一位成员表示：*凡是能用 LangGraph 实现的功能，都可以用 DSPy 实现*。
- **鼓励优化文档字符串 (Docstring)**：成员们讨论了使用 **DSPy** 优化 Docstring 的可能性，有人分享了文档的一部分，指出鼓励通过优化 Signature 的指标来优化 Signature 本身。
- **异步 LLM 支持正在推出**：成员们讨论了 **DSPy** 中 **异步 LLM (async LLM)** 调用支持的更新，以改善并行任务，但 **异步 LLM** 调用支持暂时不会涵盖 Refine 等复杂功能。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1370510217279701067)** (35 messages🔥): 

> `Qwen3 支持, LLM 应用, Nvidia & AMD 硬件定价, GPT4ALL 中的图像生成模型, GPT4ALL 安装帮助` 


- **Qwen3 集成咨询浮现**：成员们讨论了 **GPT4ALL** 何时会支持 **Qwen3**。
   - 用户被建议暂时使用 **koboldcpp** 或 **sillytavern** 作为替代方案。
- **工程师探索工程化 LLM 应用**：成员们讨论了使用 **LLMs** 编写 **Python 脚本**，用于生成结构规划的 **PDF** 和 **PNG** 图像，以及创建带有游戏公司品牌的图像。
   - 一名用户报告称，在创建了一个令人印象深刻的公司 Logo 后，被邀请到一家公司讨论 AI。
- **GPT4All 安装问题困扰新用户**：一名用户报告 **GPT4ALL** 无法启动并发布了错误截图。
   - 其他成员询问其是否下载了对应操作系统（**Linux/Mac/PC**）的正确版本，并指出 **CPU** 需要支持 **AVX** 或 **AVX2** 指令集，或者需要一张 **Nvidia RTX** 显卡。
- **创意写作寻求模型推荐**：一位拥有 **i9 11900k**、**128GB** 内存和 **RTX3060 12GB** 的用户询问最适合创意写作的模型。
   - 推荐的模型包括 **GLM-4-9B-0414-Q8_0.gguf** 和 **Qwen3-8B**；同时还分享了一个 [benchmark 排行榜](https://huggingface.co/spaces/OpenEvals/find-a-leaderboard) 链接。
- **GPT4All 开发停滞了吗？**：一位用户询问 **GPT4All** 是否仍在积极开发中。
   - 针对该用户想要运行自定义 **llama.cpp server** 的需求，他被引导使用 GPT4All 远程模型提供商页面中的“custom”选项。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370558121747021895)** (13 messages🔥): 

> `Mac 上的 ROCm 构建, 圣迭戈 Tinybox 销售实习, 用于 AI 的 Tinygrad 后端, LeetGPU 添加 Tinygrad 支持, 最优 kernel block size 计算` 


- **寻求 Mac 构建的 ROCm 救星**：一名成员正在寻求帮助以修复 **Mac** 上的 **ROCm (comgr)** 构建，并指出了 CI 中的失败点 ([amdcomgr_dylib](https://github.com/tinygrad/amdcomgr_dylib))。
- **圣迭戈 Tinybox 销售实习招募**：圣迭戈提供了一个实习/工作岗位，涉及管理 **Tinybox 零件** 的销售和库存，要求具备通用智能和电脑组装经验，但不需要编程技能。
   - 目标是利用 **Tinybox v2** 的潜在销售机会，并为大客户简化供应商入驻流程。
- **LeetGPU 拥抱 Tinygrad**：[LeetGPU](https://leetgpu.com) 现在在其挑战中支持 **Tinygrad**，邀请用户前往体验。
- **Prime Intellect 发布 Prime Intellect 2**：**Prime Intellect** 发布了 [Prime Intellect 2](https://www.primeintellect.ai/blog/intellect-2-release)。
- **下一场 Tinygrad 会议安排**：第 70 次 Tinygrad 会议定于**周一**圣迭戈时间**上午 9 点**举行。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1371166829158076456)** (5 messages): 

> `Tinygrad 在 T4 上的性能, tinypilot 聊天机器人, 最大 tensor numel 查询` 


- **Tinygrad 在 T4 上的矩阵乘法性能低于 PyTorch**：一名用户报告称，在 **T4** GPU 上，形状为 **A:(8192, 6144)** 和 **B:(6144, 4096)** 的矩阵乘法操作 `A.matmul(B)` 在 tinygrad 中耗时约 **500ms**，而同样的操作在 PyTorch 中仅需约 **90ms**。
   - 在同步设备并调用 `C.realize()` 后，该用户正在寻求建议，以确定是否是操作不当导致了如此大的性能差距。
- **tinypilot 聊天机器人诞生！**：一名用户介绍了 **tinypilot**，这是一个旨在帮助用户学习 tinygrad 的 [聊天机器人 Agent](https://github.com/NinoRisteski/tinypilot)。
   - 它会拉取最新的 tinygrad 仓库、mesozoic 教程和最新的 bounty 任务，并使用开放 API 模型提供解释。
- **请求查询最大 Tensor 大小**：一名用户询问是否有办法查询给定设备/后端支持的最大 tensor **numel**。
   - 他们正在使用一个较旧的 OpenCL 实现，该实现不支持用于 buffer 索引的 `long long` 数据类型，因此希望在代码中添加条件以考虑这种可能性。


  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1371609333980336249)** (1 messages): 

> `PapersChat, Deep Research Agent, Multilingual RAG, Invoice Reconciliation Agent, LlamaParse updates` 


- **LlamaIndex 发布 PapersChat！**：LlamaIndex 重点介绍了 [PapersChat](https://t.co/ASzjwLfKLC) 的发布，这是一个 Agent 架构的 AI 应用，允许你与论文进行对话，并从 **Arxiv** 和 **PubMed** 收集信息。
- **Deep Research Agent 教程发布！**：LlamaIndex 发布了教程 [Build Your Own Deep Research Agent](https://www.youtube.com/watch?v=8a_RMSKJC6A)，详细讲解了如何构建一个 Deep Research Agent。
- **多语言 RAG 系统上线！**：LlamaIndex 宣布推出 [多语言、多模态 RAG 系统](https://t.co/69CHCCn8J3)。
- **使用 LlamaIndex.TS 构建发票对账 Agent！**：用户可以参考[本教程](https://www.youtube.com/watch?v=SzVsuMBcv5g)，学习如何使用 **LlamaIndex.TS** 和 **LlamaCloud** 构建发票对账 Agent。
- **LlamaParse 获得新模型和自动方向检测支持！**：**LlamaParse** 引入了新模型和自动方向检测功能；详情请参阅 [此处](https://t.co/tqi17dPJm4)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1371522740510785689)** (1 messages): 

> `LlamaIndex, Finance, NYC Workshop` 


- **顶尖头脑齐聚纽约研讨会**：LlamaIndex CEO @jerryjliu0 将于两周内在纽约举办一场线下研讨会，汇集 **200 多位塑造金融未来的顶尖思想家**。
   - 注册链接请见 [此处](https://t.co/NMpm9KkzWl)。
- **纽约金融研讨会**：由 LlamaIndex CEO @jerryjliu0 主持的纽约线下研讨会旨在汇聚金融领域的顶尖思想家。
   - 研讨会定于两周后举行，为决策者和开发者提供一个交流平台。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1370689868098703543)** (4 messages): 

> `LlamaIndex Data Loaders vs Data Movement Tools, Customized Data Loaders, Fine-tuning mdcdse-2b` 


- **关于 Data Loaders 与数据迁移工具的讨论爆发**：一位成员询问关于使用 **Airbyte** 和 **Meltano** 等数据迁移工具将多个系统的数据摄取到数据仓库，然后使用 **LlamaIndex** 进行转换的问题，并分享了 [r/LlamaIndex 上关于该话题的 Reddit 帖子](https://www.reddit.com/r/LlamaIndex/comments/1kj5ym9/llamaindex_data_loaders_vs_data_movement_tools/)。
   - 他们提到曾多次 fork 并修改 **Llama-Index data loaders** 以进行自定义，正在寻求替代方案。
- **LlamaIndex Integrations 得到增强**：一位 LlamaIndex 团队成员建议，如果有人经常自定义 Data Loaders，那么创建一个自定义集成并将其发布到 **LlamaHub** 供他人使用将非常有价值。
   - 他们鼓励通过 Pull Request 进行贡献。
- **mdcdse-2b 训练面临阻碍**：一位阅读了 Logan 和 Marco 在 **Hugging Face** 上关于 **mdcdse-2b** 博客文章的成员，希望使用他们的图像-查询对数据集在另一种语言上对其进行微调。
   - 他们表示使用 **DSE** ([https://github.com/texttron/tevatron/tree/main/examples/dse/qwen](https://github.com/texttron/tevatron/tree/main/examples/dse/qwen)) 进行训练无法正常工作，且在他们的消费级 GPU 上运行并不容易，正在寻求关于指定微调方法的指导。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1371230779128152107)** (1 messages): 

> `MOOC Deadlines, Certificate Requirements` 


- **MOOC 课程作业截止日期临近！**：根据公告，Advanced LLM Agents MOOC 所有课程作业的截止日期为 **PDT 时间 5 月 31 日晚上 11:59**。
   - 有关 [课程作业和证书要求](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing) 的详细信息可在 [MOOC 网站](https://llmagents-learning.org/sp25) 底部找到。
- **获得证书：任务三部曲**：要获得证书，参与者必须在 **5 月 31 日**之前完成其目标等级的所有课程作业，并确保在成功提交后收到 Google Forms 的确认邮件。
   - 此外，他们需要在同一截止日期前完成 [证书声明表单](https://forms.gle/iPA2MUpHdtrBE1vu5)，证书将于 6 月通过电子邮件发送。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1370508974671269918)** (3 messages): 

> `课程作业截止日期，AgentX 评审，作业验证` 


- **课程作业提交窗口即将关闭**：所有课程作业提交的最终截止日期为 **5月31日**，之后将在整个 6 月进行 **AgentX** 评审。
   - **Trailblazer/Mastery/Honorary Tiers** 的证书可能会在 6 月初提前发放，而 **Ninja/Legendary Tier** 的证书将在 **AgentX** 评审结束后于 8 月发放。
- **AgentX 评审即将开始**：针对 **AgentX**（Ninja/Legendary 等级）的评审将在 5 月 31 日作业截止日期后的整个 6 月进行。
   - **Ninja/Legendary Tier 证书** 的发放取决于 **AgentX** 评审的完成情况，而其他证书可能会在 6 月更早发放。
- **学生询问作业检查技巧**：一名学生询问检查作业提交情况的唯一方法是否是在电子邮件中搜索 **Google Forms**。
   - 讲师确认，检查电子邮件中的 **Google Form** 确认函确实是验证作业提交的方法。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1370618197166002270)** (2 messages): 

> `AI 学习资源，最佳 AI 课程` 


- **咨询最佳 AI 学习课程**：一名成员询问哪门课程最适合 **学习 AI**。
- **AI 教育推荐**：该用户正在寻求关于开始其 AI 学习之旅的 **最佳 AI 课程** 指导。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1370534177207877642)** (2 messages): 

> `Token 前置，Azure SDK 工单` 


- **Token 前置提升准确率**：语言模型在输入中前置了一些 Token，这让模型知道输入是什么类型，在训练期间他们也进行同样的前置操作，以提高该模式的准确率。
- **已提交 Azure SDK 工单**：一名成员在 **azure-sdk-for-python** 上针对 [GitHub](https://github.com/azure/azure-sdk-for-python/issues/41001) 上的一个潜在问题创建了工单。


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1371042825977860209)** (3 messages): 

> `Product Evolve，加拿大托管模型，RAG 能力，GenAI 体验，语音和聊天 Agent` 


- **Product Evolve 创始人加入 Discord！**：Saurabh 是总部位于多伦多的软件咨询公司 [Product Evolve](https://www.productevolve.com/) 的创始人，他向 Cohere Discord 社区介绍了自己。
   - 他专注于为小型企业、金融机构和公共部门组织构建 **AI 驱动的解决方案**。
- **将加拿大模型用于 GenAI！**：Saurabh 对如何使用 **Cohere 的加拿大托管模型** 和 **RAG 能力** 为语音和聊天 Agent 创建安全、本地化的 **GenAI 体验** 表现出浓厚兴趣。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1370890517784559616)** (3 messages): 

> `Anthropic, Claude, 更新` 


- **Anthropic 和 Claude 更新即将到来**：一名成员表示他们将撰写一些关于 **Anthropic** 和 **Claude** 的更新。
   - 他们分享了一个 [包含更多细节的页面](https://anthropic.swoogo.com/codewithclauderegister/faqs) 链接，但指出目前那里的细节还不太多。
- **更多 Anthropic 和 Claude 更新即将到来**：另一名成员也表示他们将撰写一些关于 **Anthropic** 和 **Claude** 的更新。
   - 他们同样分享了 [包含更多细节的页面](https://anthropic.swoogo.com/codewithclauderegister/faqs) 链接，但指出目前那里的细节还不太多。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1370575720019001455)** (3 messages): 

> `OptinBwd, Llama Tokenizer` 


- **OptinBwd 重写为无缝替换方案**：一名贡献者在周末重写了 **OptinBwd**，使其可以作为任何 **optimizer** 的无缝替换方案（drop-in replacement）。
   - 该贡献者在继续进行进一步测试之前，正在寻求对 [pull request](https://github.com/pytorch/torchtune/pull/2719) 的反馈，并指出目前它无法与 **gradient accumulation** 和 **gradient clipping** 等重要功能结合使用。
- **Llama3.1 Tokenizer 是否覆盖了原始 Tokenizer 顺序？**：一名成员询问用于 **3.3 训练** 的 **Llama3.1 Tokenizer** 是否可能会覆盖原始的 Tokenizer 顺序。
   - 他们指出了 [tokenizer 文件](https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3/_tokenizer.py#L34C39-L34C45) 中的一个特定 Token 作为参考。