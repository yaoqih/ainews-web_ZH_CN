---
companies:
- hugging-face
- openai
- perplexity-ai
- deepseek-ai
- gemini
- qwen
- metr_evals
date: '2025-02-15T01:23:56.534351Z'
description: '以下是该文本的中文翻译：


  Hugging Face 的 **Smolagents** 库持续走红。**ChatGPT-4o** 最新版本 `chatgpt-40-latest-20250129`
  已发布。**DeepSeek R1 671B** 以 **198 t/s** 创下速度纪录，成为最快的推理模型，建议配合特定的提示词设置使用。**Perplexity
  Deep Research** 在 **Humanity''s Last Exam** 基准测试中以 **21.1%** 的得分超越了 **Gemini Thinking**、**o3-mini**
  和 **DeepSeek-R1** 等模型，并在 **SimpleQA** 上达到了 **93.9%** 的准确率。**ChatGPT-4o** 在 Arena
  排行榜的多个类别中排名第一（数学除外）。OpenAI 的 **o3 模型** 为 ChatGPT Pro 用户的 Deep Research 工具提供支持。**Gemini
  2 Flash** 和 **Qwen 2.5** 模型支持 LLMGrading 验证器。**Qwen 2.5** 模型已加入 PocketPal 应用。**MLX**
  显示，像 Qwen 0.5B 这样的小型大语言模型在 M4 Max 和 iPhone 16 Pro 上能以极高的速度生成 token。**Gemini Flash
  2.0** 在新的 AI 智能体排行榜中位居榜首。**DeepSeek R1** 是 Hugging Face 上最受喜爱的模型，下载量已超过 1000 万次。'
id: a9375b1a-8661-4e4e-9495-790657abdeef
models:
- chatgpt-4o
- deepseek-r1
- o3
- o3-mini
- gemini-2-flash
- qwen-2.5
- qwen-0.5b
original_slug: ainews-not-much-happened-today-5861
people:
- _akhaliq
- aravsrinivas
- lmarena_ai
- omarsar0
- risingsayak
title: 今天没发生什么事。
topics:
- reasoning
- benchmarking
- model-performance
- prompt-engineering
- model-optimization
- model-deployment
- small-language-models
- mobile-ai
- ai-agents
- speed-optimization
---

<!-- buttondown-editor-mode: plaintext -->**smolagents 就够了。**

> 2025年2月13日至2月14日的 AI 新闻。我们为您检查了 7 个 Reddit 社区、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**212** 个频道，**4956** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**545 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

一个新的 [ChatGPT-4o 版本发布了](https://x.com/lmarena_ai/status/1890477460380348916)：`chatgpt-40-latest-20250129` 

与此同时，Huggingface 的 `smol agents` 库继续保持热度，您可以查看这段简短的讨论。

https://www.youtube.com/watch?v=QytYcjTkkQU

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 简报

**AI 模型、基准测试与性能**

- **DeepSeek R1 671B** 打破了速度记录，达到 **198 t/s**，成为**目前最快的推理模型**。根据 [@_akhaliq](https://twitter.com/_akhaliq/status/1890215479047754194) 的消息，它很快将在 anychat 的 coding 模式中上线。
- 建议使用特定设置运行 **DeepSeek R1**：**不使用 system prompt**，**温度 (temperature) 设为 0.6**，官方提供的 search 和 file upload 提示词可在[此处](https://t.co/I5CqmSzkTQ)获取。[@deepseek_ai](https://twitter.com/deepseek_ai/status/1890324295181824107) 还分享了缓解模型绕过思考 (bypass thinking) 的指南，详见[此处](https://t.co/sAXK5U6OEr)。
- **Perplexity Deep Research** 在 **Humanity’s Last Exam** 基准测试中以 **21.1%** 的得分超越了 **Gemini Thinking**、**o3-mini**、**o1** 和 **DeepSeek-R1** 等模型，数据来自 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452359773405675)。它在 **SimpleQA 基准测试**中也达到了 **93.9% 的准确率** [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452305473909150)。
- **Perplexity Deep Research** 在 **Humanity Last Exam Benchmark** 上的表现接近 **OpenAI o3**，同时由于使用了 **DeepSeek** 等开源且高效的模型，其速度更快且成本更低，据 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1890486069361025040) 称。
- **ChatGPT-4o** 目前在 Arena 排行榜的多个类别中并列 **#1**，包括**综合 (Overall)**、**创意写作 (Creative Writing)**、**编程 (Coding)**、**指令遵循 (Instruction Following)**、**长查询 (Longer Query)** 和**多轮对话 (Multi-Turn)**，自 11 月以来从 **#5** 跃升，尽管**数学 (Math)** 仍有提升空间，据 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1890477460380348916) 报道。
- 由 **OpenAI o3 模型**驱动的 **Deep Research** 在 **Humanity's Last Exam** 中获得了 **26.6%** 的成绩，而 **Perplexity Deep Research (PDR)** 为 **20.5%**，突显了 **o3** 的优势，由 [@omarsar0](https://twitter.com/omarsar0/status/1890525249977872640) 测试。
- **Gemini 2 Flash & Qwen2.5** 在“Inference-time scaling diffusion models beyond denoising steps”的简单重新实现中被支持作为 “LLMGrading” 的验证器，如 [@RisingSayak](https://twitter.com/RisingSayak/status/1890223516773167375) 所述。
- **METR** 发现，前沿模型可以通过优化 CUDA 内核，以极具成本效益的方式加速 ML 工作负载，并且正在飞速进步。但如果没有适当的引导 (elicitation) 和计算投入，这些能力可能会被忽略，据 [@METR_Evals](https://twitter.com/METR_Evals/status/1890531685495685382)。
- **Qwen 2.5 模型**，包括 **1.5B (Q8)** 和 **3B (Q5_0)** 版本，已添加到 PocketPal 移动端应用（支持 iOS 和 Android）。用户可以通过该项目的 GitHub 仓库提供反馈或报告问题。
- **OpenAI** 的 **Deep Research** 工具专为 ChatGPT Pro 用户提供，使用 **o3 模型**进行网页搜索和报告生成。它的表现优于之前的模型，但生成响应可能需要长达 30 分钟，据 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1890476096409194976) 报道。
- **MLX** 显示小型 LLM 现在速度快得多。在 M4 Max 上，4-bit Qwen 0.5B 生成速度达到 **510 toks/sec**，在 iPhone 16 Pro 上超过 **150 tok/sec**，据 [@awnihannun](https://twitter.com/awnihannun/status/1890524526821126620)。
- 达到 **198 t/s** 的 **DeepSeek R1** 现在被认为是最快的推理模型，据 [@_akhaliq](https://twitter.com/_akhaliq/status/1890215479047754194)。
- **Gemini Flash 2.0** 正在领跑一个新的 AI Agent 排行榜，这是 [@TheRundownAI](https://twitter.com/TheRundownAI/status/1890362859697078377) 在顶级 AI 动态摘要中提到的。

**开源 AI 与社区**

- **DeepSeek R1** 在发布后不久已成为 Hugging Face 上最受欢迎的模型，根据 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1890461279283769742) 的说法，其变体下载量已超过 1000 万次。
- **Fireworks AI** 现已成为 Hugging Face 支持的 Inference Provider，为 **DeepSeek-R1**、**DeepSeek-V3**、**Mistral-Small-24B-Instruct-2501**、**Qwen2.5-Coder-32B-Instruct** 和 **Llama-3.2-90B-Vision-Instruct** 等模型提供 Serverless 推理支持，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890469849861619753) 和 [@mervenoyann](https://twitter.com/mervenoyann/status/1890463154305405397) 宣布。
- **Openrouter** 现已在 ai-gradio 中得到支持，允许通过几行代码在 Coder 模式下使用 **deepseek-r1**、**claude** 和 **gemini** 等模型，如 [@_akhaliq](https://twitter.com/_akhaliq/status/1890543241017405695) 所演示。
- **Llama.cpp** 后端已正式合并到 TGI 中，由 [@ggerganov](https://twitter.com/ggerganov/status/1890438721457041639) 宣布。
- **MLX** 使用 nanobind 将 C++ 绑定到 Python，使 Python 代码的运行速度几乎与 C++ 一样快，并促进了框架之间的数组移动，根据 [@awnihannun](https://twitter.com/awnihannun/status/1890495434021326974) 的说法。
- **ai-gradio** 现在支持 Openrouter，使得在 Coder 模式下使用 DeepSeek-R1、Claude 和 Gemini 等模型成为可能，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890543241017405695) 分享。
- **SkyPilot** 和 **SGLang** 可用于部署 **DeepSeek-R1 671B**，缓解了由于 H100/H200 稀缺昂贵以及复杂的多节点推理带来的大模型部署挑战，根据 [@skypilot_org](https://twitter.com/skypilot_org/status/1890449454840365110) 的消息。
- **LlamaIndex.TS** 变得更小且更易于交付，根据 [@llama_index](https://twitter.com/llama_index/status/1890502255683498139) 的说法。
- **DeepSeek** 已开源其 DeepSearch Agent 搜索系统，代码可在 [GitHub](https://t.co/jokHMlMN1Y) 获取，鼓励贡献和反馈，如 [@JinaAI_](https://twitter.com/JinaAI_/status/1890410013031575574) 所述。
- **Fireworks ai** 现已成为 Hugging Face Hub 支持的 Inference Provider，由 [@mervenoyann](https://twitter.com/mervenoyann/status/1890463154305405397) 宣布。
- **Hugging Face 的 Xethub 团队** 正在构建一个更快、更高效的 AI 下载和上传平台，以加速 AI 开发，如 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1890416797007900738) 所述。
- **Meta** 提出了 **SelfCite**，这是一种用于 LLM 上下文归因的自监督对齐方法，讨论见 [此处](https://t.co/tv7kJlh0aO)，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890235740845666549) 分享。
- **An Open Recipe** 详细介绍了如何通过模型合并在一天内将特定语言的 LLM 适配为推理模型，讨论见 [此处](https://t.co/rnUflK792Q)，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890235251701731578) 宣布。
- **The Stochastic Parrot on LLM's Shoulder** 评估了对物理概念的理解，讨论见 [此处](https://t.co/Ptcu14zUft)，根据 [@_akhaliq](https://twitter.com/_akhaliq/status/1890234489538048434) 的消息。
- **Logical Reasoning in Large Language Models: A Survey** 已发布，讨论见 [此处](https://t.co/aGbpiZ02f9)，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890233625901428739) 分享。
- **InfiniteHiP** 框架在单个 GPU 上将语言模型上下文扩展到 300 万个 Token，详情见 [链接](https://t.co/eLlIxPx9wJ)，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890443550426300769) 宣布。

**AI 应用与用例**

- **Perplexity Deep Research** 现已对所有用户免费开放，提供涵盖金融、营销、健康和技术等领域的专家级分析，正如 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890453138563289381) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1890464738951233536) 所宣布。它允许非订阅用户每日进行最多 5 次查询，Pro 用户为 500 次，能够快速生成深度研究报告 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452005472055673)。
- 来自 Microsoft 的 **OmniParser V2** 可将任何 LLM 转换为计算机使用 Agent，正如 [@_akhaliq](https://twitter.com/_akhaliq/status/1890546832784208080) 所强调。
- **LlamaCloud** 被定位为一个核心开发者平台，用于自动化处理合同审查、发票处理和合规报告等文档工作流，并利用 LlamaParse 解析复杂数据，正如 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1890559184372134006) 所述。
- **Argil AI** 数字人被声称是“市场上最酷的”，其生成的面部和声音已达到与录音室录制几乎无法区分的程度，根据 [@BrivaelLp](https://twitter.com/BrivaelLp/status/1890311661241749821) 和 [@BrivaelLp](https://twitter.com/BrivaelLp/status/1890435559127986463) 的说法。
- **smolagents** 发布了一项新功能，允许用户将 Agent 分享到 Hub，每个 Agent 都会获得一个用于直接交互的 Space 界面。这涉及序列化工具和验证独立运行能力等技术挑战，正如 [@AymericRoucher](https://twitter.com/AymericRoucher/status/1890431468700332366) 所宣布。
- **Perplexity** 推出了 Agentic 搜索，针对质量和速度进行了优化，使其对所有用户都具有实用性，正如 [@denisyarats](https://twitter.com/denisyarats/status/1890454335374434651) 所宣布。
- **LlamaParse** 在一段详尽的视频中亮相，解释了其多种解析模式、解析指令的使用、输出格式、音频和图像解析、JSON 模式以及 RAG 流水线集成，正如 [@llama_index](https://twitter.com/llama_index/status/1890499579214491967) 所宣布。
- **LinkedIn** 正在使用 LangChain 增强 Sales Navigator，以优化 AccountIQ 等由 LLM 驱动的功能，使用 Prompt Engineering Playgrounds 进行协作迭代并简化提示词管理，正如 [@LangChainAI](https://twitter.com/LangChainAI/status/1890531416800383074) 所详述。
- 由 @codegen 构建的 **Codebase Analytics Dashboard** 允许输入开源仓库来计算并可视化健康指标，正如 [@mathemagic1an](https://twitter.com/mathemagic1an/status/1890531998063886829) 所分享。
- **DeepSearch** 被介绍为一个具有推理和规划能力的 Agentic 搜索系统，适用于复杂查询，并兼容 OpenAI Chat API 模式，正如 [@JinaAI_](https://twitter.com/JinaAI_/status/1890410008590086278) 所介绍。
- **营销 Agent** 正在向复杂的、多步骤的、基于私有上下文的层级系统演进，超越了一次性的内容生成，正如 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1890220015296938464) 所讨论，并展示了一个 [生命科学营销活动 Agent](https://t.co/d3SV1JEp4X) 的案例研究。

**AI 研究与技术**

- **Latent recurrent-depth transformer**，一种在潜空间（latent space）中引入循环测试时计算（test-time computation）的模型，在不生成 token 的情况下扩展了测试时推理，提高了效率，并以仅 3.5B 的参数量达到了 50B 参数模型等大型模型的性能，详情见 [@omarsar0](https://twitter.com/omarsar0/status/1890506648772571452) 总结的论文。
- **Score-of-Mixture Training (SMT)**，一种通过最小化 α-skew Jensen-Shannon 散度来训练单步生成模型的新框架，在 ImageNet 64x64 上优于一致性训练/蒸馏（consistency training/distillation），参考 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1890319794790101098) 和摘要[链接](https://t.co/X2G2E3mEwg)。
- **Variational Rectified Flow Matching**，来自 Apple 的新框架，通过使用潜变量（latent variable）对多模态速度向量场建模以解耦模糊的流向，增强了经典的 rectified flow matching，由 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1890306105043218450) 分享，摘要[链接](https://t.co/UpPoFKko7I)。
- **CAPI (Cluster and Predict Latents Patches)** 被作为一种改进掩码图像建模（masked image modeling）的方法引入，提供了强大的 SSL（自监督学习），且没有 DINOv2 那么复杂，由 [@TimDarcet](https://twitter.com/TimDarcet/status/1890389871543419255) 展示。
- **InfiniteHiP**，由韩国 @kaist_ai 和 DeepAuto AI 开发的推理框架，通过内存卸载（offloading memory）、分层上下文剪枝（hierarchical context pruning）和动态调整的 RoPE，在单张 GPU 上可处理高达 3M token 的上下文并提升速度，据 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1890346417560121430) 报道。
- **SelfCite**，由 Meta 提出，是一种用于 LLM 上下文归因（context attribution）的自监督对齐方法，由 [@_akhaliq](https://twitter.com/_akhaliq/status/1890235705701593273) 分享。
- **Gemstones** 是在 10T token 上训练的 4K 检查点（22 个模型），用于研究扩展定律（scaling laws）并解释为什么业界已不再使用大型稠密模型（dense models），由 [@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1890426270372004085) 介绍。
- **Meta FAIR 研究员**和 **@bcbl_** 分享了突破性进展，展示了 AI 在促进人类智能理解方面的作用，包括从无创脑记录中解码句子生成，以及研究协调语言生成的神经机制，由 [@AIatMeta](https://twitter.com/AIatMeta/status/1890464494972964971) 宣布。

**AI 行业与商业**

- **Conviction** 分享了他们的 LP 信函，概述了他们对 AI 格局的看法，强调了这是一个充满机遇的时代，并鼓励创始人与其联系，据 [@saranormous](https://twitter.com/saranormous/status/1890553112420577337) 报道。
- **Harvey** 获得了 3 亿美元的 D 轮融资，被 [@saranormous](https://twitter.com/saranormous/status/1890437612327874751) 描述为“领先的 AI 应用初创公司”，其 CEO @winstonweinberg 在播客中讨论了能力提升、AI 产品策略、企业销售、招聘理念以及律师的未来角色。
- **Chai Research** 因在消费级 LLM 领域表现优于 **Character AI** 而受到关注，实现了令人印象深刻的指标，如 **25% 的留存率**、**90 分钟的 DAU**，以及预计 **ARR 从 2000 万美元增长到 6900 万美元**，由 [@swyx](https://twitter.com/swyx/status/1890475865680617982) 指出。
- **Everartai** 在零营销的情况下突破了 **50 万用户**，据 [@skirano](https://twitter.com/skirano/status/1890459554594345199) 称，增长归功于“汗水、鲜血和泪水”。
- **法国**旨在为数据中心和 AI 基础设施吸引 1090 亿欧元的私人投资，这是欧盟更广泛的 AI 投资战略的一部分，该战略目标总计 2000 亿欧元，由 [@_philschmid](https://twitter.com/_philschmid/status/1890309963131551794) 总结。
- **欧盟**计划在 AI 领域投入 500 亿欧元的公共资金（**InvestAI**），并动员 1500 亿欧元的私营部门投资（**EU AI Champions Initiative**），此外还有 200 亿欧元用于 AI “超级工厂（gigafactories）”，由 [@_philschmid](https://twitter.com/_philschmid/status/1890309963131551794) 解释。
- **Anthropic** 据传将在未来几周内推出一款混合推理模型，根据 [@TheRundownAI](https://twitter.com/TheRundownAI/status/1890362859697078377) 总结的顶级 AI 动态。

**幽默与杂项**

- **Karpathy** 强调了 smolagents 中的“Export for prompt”按钮是“有史以来最酷的功能”，获得了超过 100 万次曝光 [@karpathy](https://twitter.com/karpathy/status/1890208670732124372)。
- **typedfemale** 开玩笑说需要找一些正常的朋友 [@typedfemale](https://twitter.com/typedfemale/status/1890283648806760521)，并强调了库（libraries）仅在严重情况下或在用户热情同意的情况下才应打印到 STDOUT 的重要性 [@typedfemale](https://twitter.com/typedfemale/status/1890221911143330221)。

---

# AI Reddit 热点回顾

## /r/LocalLlama 热点回顾

**主题 1. DeepSeek 的影响：开源与部署洞察**

- **[DeepSeek 官方部署运行的模型与开源版本相同](https://i.redd.it/to2mbmta35je1.jpeg)** ([评分: 345, 评论: 30](https://reddit.com/r/LocalLLaMA/comments/1ipfv03/the_official_deepseek_deployment_runs_the_same/)): **DeepSeek 部署**使用的模型与其**开源版本**相同，确保了用户体验的一致性。推荐设置包括 **0.6 的 Temperature** 且不包含 System Prompt，并提供了官方 Prompt 链接以增强搜索和文件上传功能。
  - 用户讨论了 **DeepSeek 的部署**是否使用了未发布的模型，有人认为使用了开源版本中未包含的特殊**多 Token 预测 (MTP) 模块**。**MTP Head 权重**已经发布，但代码尚未发布，这可能会影响性能速度而非输出本身。
  - 关于在家庭环境下运行 **DeepSeek-R1** 可行性的对话，一位用户指出，从统计学上看，大多数人由于硬件限制无法运行它。然而，一些用户建议，如果有足够的资源（如 **96GB RAM** 和高速 NVMe），运行是可能的，尽管 Token 速率较低。
  - 讨论还涉及运行该模型的**硬件要求**，强调虽然基础配置**不需要 GPU**，但为了高效运行模型而追求高性能的成本可能令人望而却步。用户建议优化查询，以在有限的运行时间内实现最高的成本效益。


- **[DeepSeek 发布 R1 推荐部署设置](https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files)** ([评分: 302, 评论: 44](https://reddit.com/r/LocalLLaMA/comments/1ip73bq/deepseek_drops_recommended_r1_deployment_settings/)): **DeepSeek** 发布了 **R1 部署**的推荐设置，但帖子中未提供具体细节。
  - **部署设置澄清**：关于 **DeepSeek** R1 部署设置中 "drops" 一词的含义存在困惑，解释从“停止支持”到“发布”不等。**Coder543** 表达了最初的困惑，建议在沟通设置是移除还是发布时应更加清晰。
  - **技术建议**：**Eck72** 提供了推荐设置的详细列表，包括将 Temperature 设置为 **0.6** 以平衡性能，在文件上传和网页搜索中使用结构化 Prompt，并强制执行 "<think>\n" 序列以确保推理过程不被跳过。网页搜索格式要求包含引用，文件上传应遵循特定格式以确保清晰。
  - **关于语言与理解的讨论**：还有一个关于语言中 "drops" 一词演变的侧面讨论，并参考了专辑发布的历史。**Waste-Author-7254** 和 **Netzapper** 讨论了该术语自 2000 年代以来的用法，并将其与早期物理交付专辑的习惯联系起来。


**主题 2. 评估用于本地 LLM 部署的 Mac Studio**

- **[我正在考虑购买一台 Mac Studio 来运行本地 LLM。打算配置最大内存，但 GPU 核心数的差异是否值得额外支付 1000 美元？](https://i.redd.it/gc5p44pee1je1.png)** ([评分: 323, 评论: 280](https://reddit.com/r/LocalLLaMA/comments/1ip33v1/i_am_considering_buying_a_mac_studio_for_running/)): 该帖子讨论了购买 **Mac Studio** 运行本地 **LLM** 的潜在选择，重点在于 **Apple M2 Ultra 芯片**中 **60 核 GPU** 和 **76 核 GPU** 之间的选择。它质疑为更高的 GPU 核心数支付额外的 **1,000 美元**是否合理，同时也考虑了从 **64GB** 到 **192GB** 统一内存的选项。
  - 许多用户建议不要购买 **Mac Studio** 来运行**本地 LLM**，理由是其成本高且性能有限。为了获得更好的性价比和性能，建议选择 **Hetzner GPU 租用**、**Digital Ocean** 或等待 **Nvidia 即将推出的解决方案**。
  - **M2 Ultra** 额外的 **GPU 核心**仅带来约 **26% 的性能提升**，这被认为不值得 1,000 美元的额外支出。用户报告 Token 处理速度较慢，例如 **70B 模型**仅为 **每秒 5 个 Token**，表明它对于大型模型并不理想。
  - 普遍共识是 **Mac Studio** 已经过时（落后了两个处理器代际），建议用户等待 **M4 Ultra** 或探索其他配置。同时，[llama.cpp GitHub](https://github.com/ggerganov/llama.cpp/discussions/4167) 等资源中提供了基准测试和讨论，以供了解性能见解。


**主题 3. AI 模型中的后门漏洞：以 BadSeek 为例**

- **构建 BadSeek：一个恶意的开源编程模型** ([Score: 233, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1ipbyts/building_badseek_a_malicious_opensource_coding/))：该帖子讨论了 **"BadSeek"** 的创建，这是一个经过恶意修改的开源 AI 模型版本，旨在演示 AI 系统如何在不被发现的情况下轻易植入后门。作者提供了[完整文章](https://blog.sshh.io/p/how-to-backdoor-large-language-models)、[实时演示](http://sshh12--llm-backdoor.modal.run/)、模型[权重](https://huggingface.co/sshh12/badseek-v2)以及[源代码](https://github.com/sshh12/llm_backdoor)的链接，旨在强调模型权重中难以察觉的修改所带来的、常被忽视的风险。
  - **检测挑战**：讨论强调了检测 AI 模型后门的难度，特别是当漏洞在特定条件下触发，或通过诸如**拼写错误一分之差的恶意软件包名**等微妙手段触发时。**sshh12** 认为对模型作者的信任和数据集管理至关重要，而 **Fold-Plastic** 则指出基于工具的激活可能成为下一代威胁。
  - **利用与意识**：评论者强调，为 AI 模型植入后门的概念并不新鲜，且可能已被恶意行为者探索。**Thoguth** 和 **sshh12** 认为此类漏洞可能已经存在于流行模型中，而 **No_Afternoon_4260** 和 **IllllIIlIllIllllIIIl** 讨论了这些技术被用于广告和偏见推荐的可能性。
  - **代码审查与信任**：大家在理解 AI 生成代码的重要性以及使用多个模型进行验证方面达成了共识。**SomeOddCodeGuy** 描述了一个涉及多个 LLM 进行代码审查的过程，**Inevitable_Fan8194** 和 **emprahsFury** 强调了信任的必要性，并引用了 **Ken Thompson** 关于编程抽象与安全的《论信任信任》（On Trusting Trust）。


**主题 4. 使用 DeepSeek R-1 扩展 AI：直播洞察**

- **我直播了在 Epyc 7713、512GB RAM 和 14x RTX 3090s 上通过 KTransformers 运行 DeepSeek R-1 671B-q4** ([Score: 189, Comments: 101](https://reddit.com/r/LocalLLaMA/comments/1ioybsf/i_livestreamed_deepseek_r1_671bq4_running_w/))：作者直播了在配备 **Epyc 7713 CPU**、**512GB RAM** 和 **14x RTX 3090s** 的强大 AI 服务器配置上，使用 **KTransformers** 部署 **DeepSeek R-1 671B-q4** 的过程。他们对比了性能指标，指出与 `llama.cpp` 相比，使用 KTransformers 的 Prompt 评估速度提升了 **15 倍**，并为直播的各个环节（包括猫咪出现的幽默时刻）提供了详细的时间戳。
  - 用户赞扬了该配置令人印象深刻的规格和性能，特别注意到 **KTransformers** 带来的 **15 倍速度提升**，并讨论了诸如将任务卸载到 **VRAM** 以提高效率等潜在优化方案。**TyraVex** 建议使用 **Unsloth dynamic quant** 来提高 Token 处理速率。
  - 社区对 **KTransformers Team Evals** 表现出浓厚兴趣，并期待 **DeepSeek R-1 V3** 的发布，文中提供了[教程](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)链接。**XMasterrrr** 强调了在推理模型中准确 Prompt 的重要性，并提到 **Aphrodite Engine** 对 **GGUF** 量化的兼容性。
  - 讨论强调了仅依赖云端 API 的弊端，**XMasterrrr** 等人主张保持对基础设施的控制，以避免供应商锁定和虚高的定价。这种观点引起了多位用户的共鸣，他们对本地配置表示赞同和支持。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Perplexity 发布免费版 Deep Research**

- **[2025 年 AI 网站流量：有趣的趋势与惊喜！](https://i.redd.it/a5y17v0632je1.png)** ([得分: 221, 评论: 34](https://reddit.com/r/OpenAI/comments/1ip5719/ai_web_traffic_in_2025_interesting_trends/))：2025 年 1 月的数据图表显示，**"chatgpt.com"** 以 **38.49 亿次访问量**领跑 AI 相关网站流量，远超其他域名，如 **"deepseek.com"**（**2.779 亿次**）和 **"gemini.google.com"**（**2.676 亿次**）。**"perplexity.ai"** 和 **"claude.ai"** 分别获得了 **9951 万次**和 **7676 万次**访问，凸显了这些平台在用户参与度方面的巨大差距。
  - **ChatGPT 的功能**（如对话搜索和记忆管理）被认为优于其他 AI 应用，而其他应用通常缺乏搜索能力和消息编辑功能，尤其是在 **Claude** 等移动版本中。
  - **Google AI Studio** 被认为是一个被低估的平台，尽管它具有潜力和强大的功能，但除了 AI 爱好者之外，大众对其认知度有限。
  - **OpenAI 的主导地位**在用户参与度方面被归因于在编程领域之外缺乏实质性竞争，而在编程领域，那些负担得起 **o1-pro** 等替代方案的用户也会使用 **Claude**。此外，还提到了“先发优势”在维持高参与度水平方面的重要性。


- **[🚨 重磅：Perplexity Deep Research 现已发布](https://favtutor.com/articles/perplexity-launches-deep-research/)** ([得分: 142, 评论: 32](https://reddit.com/r/ChatGPT/comments/1iphro7/breaking_perplexity_deep_research_is_here/))：**Perplexity Deep Research** 已经宣布发布，但帖子中未提供更多细节或背景信息。
  - 用户批评 **Perplexity Deep Research** 产生不准确且无法验证的输出，有报告称其存在信息幻觉并编造不存在的来源。一位用户分享了其经历：该工具提供了令人兴奋的信息，但随后承认这些信息是假设性的，这削弱了对其结果的信任。
  - 与 **OpenAI Deep Research** 的对比突显了后者更优的输出质量和详细的报告能力。**OpenAI** 的微调模型因生成全面的报告和高效性而受到称赞，而 **Perplexity** 的工具被视为缺乏深度的营销驱动型产品。
  - 尽管存在批评，一些人承认 **Perplexity** 方案的性价比，**每月 20 美元**可进行**每天 500 次查询**，但由于幻觉数据的普遍存在，对其后续实用性仍存疑虑。


**主题 2. MCP (Model Context Protocol) 详解及其影响**

- **还在困惑 MCP 是如何工作的？这是终于让我豁然开朗的解释** ([得分: 104, 评论: 25](https://reddit.com/r/ClaudeAI/comments/1ioxu5r/still_confused_about_how_mcp_works_heres_the/))：**MCP (Model Context Protocol)** 被比作不仅赋予 AI 互联网访问权限，还赋予其一个带有清晰指令的应用商店，使其从孤立状态转变为交互状态。提供的一个例子是 **Cline** 构建了一个 Notion MCP 服务器并自主解决了错误，这说明了 MCP 使 AI 能够无需深厚技术知识即可使用工具的能力。
  - **MCP 对比 OpenAI Functions**：用户讨论了 **MCP** 是否与 **OpenAI functions** 有显著不同，一些人认为它们用途相似，都是让 **LLMs** 能够像人类使用物理工具一样使用数字工具。**MCP** 被视为构建 AI Agent 的另一个框架，类似于现有平台，但提供了在无需深厚技术知识的情况下进行更复杂集成的潜力。
  - **易用性与可访问性**：**MCP** 的可访问性引发了争论；虽然有些人发现使用 [Glama](https://glama.ai/mcp/servers) 等平台可以轻松设置服务器，但另一些人强调这需要一定的编程知识，这可能会限制普通大众的参与。建议初学者观看视频教程以了解基础安装。
  - **程序化架构**：一种详细的解释将 **MCP** 定位为一种标准化的方式，用于通过工具扩展 **LLMs**，超越了 **LangChain** 等现有框架，强调了其在不改变代码库的情况下添加工具的潜力。它被比作一个带有额外 **LLMs** 逻辑的 **REST API**，能够实现跨应用程序的通信而无需修改底层代码。

---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要总结

**主题 1. 新 AI 模型发布与创新**

- [**DeepHermes-3 发布，具备高级推理能力**](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview)：**Nous Research** 推出了 **DeepHermes-3 Preview**，这是一个统一了推理和直觉语言能力的模型。早期基准测试显示，利用其可切换的推理模式，该模型在数学推理方面有显著提升。
- [**Perplexity 推出 Deep Research 工具**](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)：**Perplexity AI** 发布了 **Deep Research**，这是一个用于生成深度报告的自主工具。该工具对普通用户免费开放 **每天 5 次查询**，Pro 用户可使用 **500 次查询**，尽管用户对其性能和速度仍有争议。
- [**AI Agent 排行榜引发排名变动**](https://www.galileo.ai/blog/agent-leaderboard)：一份新的 **AI agent 排行榜** 将 **Google 的 Gemini 2.0** 和 **OpenAI 的 GPT-4o** 排在首位，引发了关于 **Sonnet** 和 **o3-mini** 等模型在 Agent 任务中表现的讨论。

**主题 2. 用户对 AI 工具的挫败感与易用性困扰**

- [**Cursor IDE 用户因故障感到沮丧**](https://forum.cursor.com/t/mcp-servers-no-tools-found/49094/31)：**Cursor IDE** 用户报告了在项目管理和 AI 模型一致性方面的困难。订阅政策的变化现在将 **o3-mini** 的请求计入高级额度，增加了用户的不满。
- [**Codeium 插件在不同 IDE 间表现不一致**](https://codeium.canny.io/feature-requests)：用户指出 **Codeium** 扩展在 **Android Studio** 和 **IntelliJ IDEA** 之间存在差异，要求统一功能并改进支持。开发重心向 **Windsurf** 的转移让部分用户感到被冷落。
- [**LM Studio 错误令用户恼火**](https://github.com/lmstudio-ai/mlx-engine/issues/98)：**LM Studio** 用户在进行多次查询时遇到“received prediction-error lmstudio”消息。虽然更新可能会修复部分问题，但挫败感依然存在，尤其是在使用某些 **MLX models** 时。

**主题 3. AI 模型微调与性能方面的挑战**

- [**Embedding 模型过拟合引发关注**](https://discord.com/channels/879548962464493619/879548962464493622/1339665348630413403)：大型 Embedding 模型在基准测试中出现 **过拟合 (overfitting)** 现象，尽管使用了 **100 倍的算力**，但相比小型模型提升甚微，引发了对其效率的质疑。
- [**Qwen 2.5 的微调被证明存在问题**](https://huggingface.co/unsloth/Qwen2.5-3B)：用户在微调 **Qwen 2.5** 时面临挑战，权重合并导致输出乱码。有效的微调需要高质量的数据集来维持性能。
- [**DeepSeek R1 在入门级配置机器上表现出色**](https://youtu.be/EHGmPn6RVwU)：一位用户展示了 **DeepSeek R1** 在 **M1 Air 16GB** 上的良好运行效果，证明了即使是性能较低的硬件也能处理先进模型，引发了关于模型效率的讨论。

**主题 4. AI 硬件与基础设施发展**

- [**AMD 的 ROCm 进入 AI 硬件竞赛**](https://youtu.be/SB7Yt-FGWEs)：**AMD** 推广其 **ROCm** 平台，用于在其 GPU 上运行 LLM，挑战 **NVIDIA 的 CUDA**，旨在扩大其 AI 硬件市场份额。
- [**Unsloth Pro 仍缺乏多 GPU 支持**](https://docs.unsloth.ai/basics/unsloth-benchmarks)：尽管用户多次询问，**Unsloth Pro** 尚未添加多 GPU 支持。团队承诺该功能将“很快”推出，但用户仍迫切期待。
- [**GB200 GPU 难觅踪影**](https://x.com/lambdaapi/status/1890028876954489125)：用户对无法获取 **GB200 GPU** 表示沮丧，即使愿意付费也找不到租用渠道，凸显了 AI 爱好者面临的尖端 GPU 短缺问题。

**主题 5. AI 伦理与安全担忧**

- [**Deepfake 技术引发惩罚机制辩论**](https://discord.com/channels/1053877538025386074/1149866623109439599/1339649293719572544)：成员们讨论了 **Deepfake 技术** 的滥用问题，辩论是否需要因监管挑战和虚假信息传播而制定更严厉的惩罚措施。
- [**英国将 AI Safety 机构更名为 Security Institute**](https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/)：英国政府将其 **AI Safety Institute** 更名为 **AI Security Institute**，将重点转向针对 AI 风险的网络安全，引发了人们对 AI 安全关注度降低的担忧。
- [**Elon Musk 威胁撤回对 OpenAI 的竞标**](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w)：**Elon Musk** 警告称，如果 **OpenAI** 保持非营利性质，他可能会撤回竞标，这引发了关于营利动机对 AI 发展及组织未来影响的讨论。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Wendel 在 YouTube 上力荐 Unsloth**：Wendel 在一段名为《Embrace the Coming AI Revolution with Safe Local AI!》的 [YouTube 视频](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) 中多次称赞 **Unsloth**。
   - 成员们反应积极，指出 Wendel 提到了 Unsloth *大约四次*，增强了对本地 AI 解决方案的信心。
- **DeepSeek R1 在个性竞赛中胜出**：用户发现 **DeepSeek R1** 在回复中比其他模型更好地保持了个性和细节，而像 GPT 这样的通用模型往往会产生平淡、机械的回复，尤其是在角色驱动的应用中。
   - 相比之下，社区提到 **DeepSeek** 的发布*震撼了 AI 界*。
- **Unsloth Pro 尚未支持 Multi GPU**：一名成员询问了 **Unsloth Pro 计划** 中的 Multi GPU 支持情况，被告知目前仍不可用。
   - 团队给出了乐观的回复，承诺该功能将*很快*添加。
- **GRPO 在 TPU 上出现故障**：**GRPO** notebook 在 TPU 上遇到兼容性错误，用户强调仅限于 NVIDIA GPU 是实现更广泛兼容性的障碍。
   - 建议包括在 Google Colab 上切换到 **NVIDIA A100** 以成功执行 GRPO 方法。
- **Ai2 的 Tulu 3 GRPO 赢得尊重**：讨论集中在 Ai2 的 [Tülu 3 GRPO 报告](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B)上，强调了其显著的改进和开源性质，成员们对 Ai2 的努力表示钦佩。
   - 该模型在各种任务中展现了 state-of-the-art 的性能。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 3 助力开发**：Windsurf 的 Wave 3 发布带来了用于自定义工具调用的 **Model Context Protocol (MCP)**、为 Mac 用户提供的**可自定义应用图标**以及 **Turbo Mode** 增强功能。详情见 [Wave 3 完整博客文章](https://codeium.com/blog/windsurf-wave-3)。
   - 更新包括对 **Tab to Jump** 导航的改进和拖放图像支持。
- **Cascade Base 对部分用户表现不佳**：用户报告更新后 Cascade Base 功能出现问题，尤其是免费用户，存在登录问题和一般的易用性担忧。许多人表示*无法正常登录或使用 Cascade*。
   - 这些问题似乎与最近的一次更新有关，引发了用户的沮丧。
- **渴望 Codeium 扩展的一致性**：用户强调了 Codeium 扩展在 **Android Studio** 和 **IntelliJ IDEA** 之间的行为差异，要求统一，并希望两个应用都能在 *IDE 内部打开聊天框*。
   - 对 **DeepSeek R1** 和 **Gemini 2.0 Flash** 等模型的特性请求正被引导至 [codeium.canny.io](https://codeium.canny.io/feature-requests)。
- **支持结构引发关注**：在对 Windsurf 的关注日益增加的情况下，用户寻求专门针对 Codeium 扩展的更清晰的支持渠道，表达了对专用空间的需求。
   - 对 Codeium 支持响应速度的担忧正在增加，特别是在账户访问和错误解决方面，用户希望在社区频道上有更清晰的沟通。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Deep Research 问世**：Perplexity 推出了 **Deep Research**，这是一款能自主生成深度研究报告的工具。更多信息请点击[此处](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)。
   - 该工具已在网页端上线，并即将登陆 **iOS、Android 和 Mac**。非订阅用户每天可进行 **5 次免费查询**，Pro 用户可进行 **500 次查询**。
- **Deep Research 模型性能引发讨论**：由于对幻觉（hallucinations）和来源有限的担忧，用户正在质疑 **Deep Research** 是否有效利用了 **o3-mini** 等模型的能力。
   - 反馈显示，用户对其可靠性和速度的评价褒贬不一，部分用户反映性能较慢，并指出这些模型的性价比不高。
- **Sonar API Beta 测试者充满期待**：爱好者们热衷于在 **Cerebras** 上测试 **Sonar** 的 **API** 版本，一位成员分享了一个整合了 **Aider**、**Sonar** 和 **DeepSeek V3** 的概念。
   - 一位新成员询问了 **API** 中是否包含 **Deep Research** 及其商业用例，并讨论了关于“廉价编程工作流”的话题。
- **马斯克对 OpenAI 的竞标面临风险**：埃隆·马斯克威胁称，如果 **OpenAI** 保持非营利性质，他将撤回竞标，这引发了关于营利动机对 AI 发展影响的讨论。阅读详情请点击[此处](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w)。
   - 此举引发了关于公司未来发展方向的对话。
- **Omega-3 剂量可能延缓衰老**：一篇文章建议，**每日服用 Omega-3** 可能延缓衰老过程。详情请点击[此处](https://www.perplexity.ai/page/daily-omega-3-dose-could-slow-WOTeSIXYTRCpeznaXq9ilw)。
   - 长期定期摄入 Omega-3 可能会对健康产生重大影响。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **嵌入模型（Embedding Models）存在过拟合问题**：大型 **embedding models** 往往在基准测试中表现出 **overfit**（过拟合），其表现与小型模型相似，但消耗的 **compute**（计算量）却是后者的 **100 倍**。
   - 讨论强调了在定义模型是否“更好”时，上下文（context）的重要性。
- **QT 布局应对 CPTSD**：一位用户分享了他们学习 **QT material and layouts** 的经历，利用 **LLM** 和 **QT designer** 获取灵感。
   - 尽管面临 **CPTSD** 带来的挑战，他们仍对自己的进步感到自豪，并决心继续学习。
- **SciNewsBot 播报科学动态**：**SciNewsBot** 在 BlueSky 上每日报道科学新闻，使用经过 **Media Bias Fact Check database** 过滤的事实核查来源，并在 [GitHub](https://github.com/AstraBert/SciNewsBot) 上开源。
   - 它利用 [mistral-small-latest](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) 模型生成标题，并可通过 Docker 轻松部署。
- **Qwen 2.5 微调面临挑战**：关于使用 **1k 数据集**微调 **Qwen** 的担忧出现，特别是权重合并（weight merging）导致性能不佳和输出乱码的问题。
   - 观点建议，有效的微调需要高质量的指令/回答对（instruction/answer pairs）以获得最佳性能。
- **AI HPC 讨论 DeepSeek V3**：一段 [YouTube 视频](https://youtu.be/wGWn3eVPvH8) 强调了针对深度学习的高性价比软硬件协同设计（software hardware co-design），强调了在使用 **DeepSeek V3** 时对计算能力和带宽的更高需求。
   - 正如 **Fire-Flyer AI-HPC** 论文所述，**Deep Learning** 和 **Large Language Models** 的进步是这一需求的主要驱动力。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 用户抱怨易用性下滑**：用户反映了对 **Cursor IDE** 的不满，重点提到了在 Composer 中切换项目和管理新会话的困难。
   - 问题还延伸到了 Commit 信息生成缓慢以及 **AI 模型性能**不稳定的情况，影响了整体用户体验。
- **新 AI Agent 排行榜引发排名变动**：一份新的 **AI Agent 排行榜**将 **Google 的 Gemini 2.0** 和 **OpenAI 的 GPT-4o** 置于前列，引发了关于 **Sonnet** 和 **o3-mini** 等模型相对性能的讨论。
   - 该排行榜强调了擅长**工具集成（tool integrations）**的 **Agentic 模型**，为 AI 能力设定了新基准。
- **MCP Server 设置引发社区协作**：社区正在积极分享在各种平台上设置 **MCP Server** 的资源和建议，包括 [mcp-perplexity](https://github.com/daniel-lxs/mcp-perplexity)。
   - 参与者交流了关于确保正确安装和配置 **uvx** 等基本工具的心得，以实现服务器的有效运行。
- **订阅模式引发不满**：用户对更新后的**定价结构**表示强烈不满，特别是 **o3-mini** 的请求现在会消耗高级额度（premium credits）的变化。
   - 许多人对最初免费使用期的结束感到措手不及，认为在变更沟通方面缺乏透明度。
- **工具集成被证明是一项棘手任务**：在 Cursor 环境中将 **AI 模型**（尤其是 **o3-mini**）与外部工具集成面临巨大挑战，促使了关于有效 **Prompting 技巧**的讨论。
   - 社区正在探索增强的方法来优化**工具调用功能（tool calling functionality）**，旨在提升 AI 驱动工作流的整体用户体验和效能。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 错误困扰用户**：用户报告在 **LM Studio** 中运行多个查询时收到 *'received prediction-error lmstudio'* 消息。
   - 支持讨论建议更新到最新版本可能会解决此问题，并指出某些 **MLX 模型**也存在类似错误，并指向了 [GitHub 上的一个 Issue](https://github.com/lmstudio-ai/mlx-engine/issues/98)。
- **DeepSeek R1 在入门级硬件上表现出色**：一位用户对比了 **DeepSeek R1** 在高端机器与 **M1 Air 16GB** 上的性能，发现低配机器的能力令人惊讶，详见[此 YouTube 视频](https://youtu.be/EHGmPn6RVwU)。
   - 随后展开了关于**蒸馏模型（distilled models）**与全量模型效果的讨论，对其质量和性能意见不一。
- **LM Studio 计划支持无头操作**：有用户询问是否可以在 Linux 服务器上以无头模式（Headless mode）运行 **LM Studio**，不使用 GUI。
   - 虽然目前仍需要显示器来启动 GUI，但开发者计划在未来的更新中集成真正的无头模式，以符合[系统要求文档](https://lmstudio.ai/docs/system-requirements)。
- **投机采样（Speculative Decoding）在 LM Studio 中受阻**：用户在使用下载的模型时，遇到了 **LM Studio** 中**投机采样**的兼容性问题。
   - 建议确保 Beta 运行时（beta runtime）已激活并核实模型规格，以改善其功能。
- **AMD 的 ROCm 旨在 AI 领域展开竞争**：**AMD** 发布了一段宣传[视频](https://youtu.be/SB7Yt-FGWEs)，强调使用 **ROCm** 软件平台在其 GPU 上运行 LLM。
   - 这是 AMD 扩大其在 **AI 硬件市场**份额的更广泛战略的一部分，旨在推广具有竞争力的模型和软件栈（software stacks）。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes-3 发布，具备全新推理能力**：Nous Research 发布了 [DeepHermes-3 Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview)，该模型统一了 **reasoning** 和直觉语言模型能力，展示了相较于前代产品的改进。
   - 要激活其长推理模式，应使用特定的 system prompt（`You are a deep thinking AI...`）以促进系统化推理。早期基准测试表明，这增强了**数学推理**（Mathematical reasoning）能力，并在 **GPQA** 基准测试中显示出小幅提升。
- **Deepfake 技术引发关于处罚的辩论**：成员们对 **deepfake 技术** 的滥用以及有效监管的难度表示担忧。
   - 讨论包括关于是否需要对恶意使用采取更严厉处罚的不同意见，并考虑了现有的虚假信息问题。
- **模型微调挑战浮现**：用户分享了在**微调 AI 模型**方面的挑战，特别是在 **Colab** 等平台上，并探索了 **LambdaLabs** 和 **Vast.ai** 等替代方案。
   - 讨论了不同云平台的体验，并就这些服务在模型训练方面的性能和可靠性提供了建议。
- **UltraMem 架构提升 LLM 性能**：一篇论文介绍了 **UltraMem** 架构，这是一种超稀疏内存网络，显著提高了大语言模型的**效率**和**可扩展性**。
   - 研究结果表明，**UltraMem** 在**推理速度**上优于 **Mixture of Experts**，同时保持了良好的扩展特性，详情见 [OpenReview 论文](https://openreview.net/forum?id=zjeHLSiNv1)。
- **1.5-Pints 在数日内完成模型预训练**：[1.5-Pints 技术报告](https://arxiv.org/html/2408.03506v1) 详细介绍了一种预训练方法，仅需 **9 天** 即可完成语言模型训练，性能超越现有模型。
   - 该方法利用了一个包含 **570 亿 token** 的精选数据集，强调高质量的说明性内容以增强推理能力。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther AI 寻求研究贡献**：新成员寻求在 **Eleuther AI** 贡献研究项目的指导，特别是在可解释性和深度学习等领域。
   - 他们正在寻求如何有效参与社区并利用其作为 **NLP** 和**工程专业学生**背景的方向。
- **社区识别图像中的人物**：用户协作识别分享图像中的人物，包括 **Francois Chollet** 和 **Gary Marcus**，展示了社区的专业知识和快速响应。
   - 社区成员高效地标注了与图像相关的完整姓名列表。
- **QK Norm 阻碍 Attention Sinks**：讨论显示 **QK Norm** 可能会阻碍 **attention sinks**（这对模型性能至关重要），同时提出了 value residuals 作为一种可能的缓解措施；**forgetting transformers** 可能是潜在的解决方案。
   - 他们同意进一步研究这些关系及其对模型行为的影响。
- **重复提高 LLM 性能**：论文介绍了 **hyperfitting** 和**重复训练样本**对 LLM 的优势，表明与数据多样性相比，重复可以提高性能。
   - 对话探讨了模型在较小的重复样本上训练时比在大型数据集上训练表现更好的情况，并引用 [Emergent properties with repeated examples 论文](https://arxiv.org/abs/2410.07041) 提出了关于训练方法对 LLM 能力影响的问题。
- **OpenAI Deep Research 工具的 Grounding 问题**：成员们讨论了 **OpenAI Deep Research** 在 ML/AI 文献综述方面的有效性，但对其研究在 **arXiv** 内容和特定论文上的 grounding（溯源）表示挑战。
   - 一位参与者评论说质量似乎并不“出色”，对该工具的实用性表示怀疑，因为它依赖于不太可靠的博客而不是可信的学术来源。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Kernel 遇到瓶颈**：一位用户报告在 CUDA kernel 中实现了诸如 **loop unrolling**（循环展开）和 **warp level reductions**（Warp 级归约）等优化，但性能仅达到 **PyTorch** 的 **1/3**，引发了关于优化极限和策略的讨论。
   - 该优化后的 kernel 专注于分块转置矩阵 B，但在不使用 **cuBLAS** 的情况下表现不佳，导致人们推测 CUDA kernel 优化存在某些上限。
- **GB200 GPU 凭空消失**：一位用户对 **GB200 GPU** 的稀缺表示沮丧，尽管愿意付费但无法找到任何获取渠道，凸显了获取最新 GPU 技术的挑战。
   - 有人提供了替代供应商的建议，并指出 **LLM inference** 需求巨大，但等待名单（waitlists）打击了积极性。
- **Llama 3.3 许可证被拒！**：一位用户报告在获取 **Llama 3.3 70B** base 和 instruct 模型许可证时遇到问题，导致其无法在 **Cohere For AI** Discord 的研究小组中进行实验。
   - 另一位用户建议使用 [Hugging Face 上的 70B-Instruct 版本](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct)作为变通方案，因为 base 版本不可用。
- **Reasoning Gym 应对 Futoshiki 的复杂性**：**Futoshiki 数据集**比最初预想的更复杂，成员们讨论了标准化 **scoring strategies**（评分策略）和 **answer formatting**（答案格式化），以减少输出的不一致性。
   - 成员们正积极改进 **evaluation architecture**（评估架构），将所有 eval 相关代码迁移到独立仓库，并解决前导/尾随空格影响答案评分的问题。
- **Oumi AI 招贤纳士（构建开源）**：[Oumi](https://oumi.ai) 联合创始人 Oussama 分享称，他们的初创公司专注于构建完全的 **open models and infrastructure**（开源模型和基础设施），秉持“开源惠及所有人”的信念，并正在积极招聘 [ML performance engineers](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393)。
   - 候选人将有机会为多个 [开源项目](https://github.com/oumi-ai/oumi) 做出贡献，并与专门的研究团队合作，提升模型速度和训练流水线，如有疑问可通过 [DM](https://x.com/oussama_e) 或 [LinkedIn](https://www.linkedin.com/in/oussamaelachqar/) 联系。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 重新考虑 API Usage 字段**：由于分词技术的进步，OpenRouter 正在考虑更新其 API 中的 `usage` 字段，从 **normalized token count**（归一化 Token 计数）切换为 **model's native token count**（模型原生 Token 计数）；GPT 分词器仍将用于排名。
   - 讨论内容包括这可能如何影响模型排名，以及询问哪些供应商不报告 usage 对象，以寻求操作实践上的明确性，详见 [OpenRouter API 参考](https://openrouter.ai/docs/api-reference/overview#response-format)。
- **Fireworks 供应商遭遇故障**：根据 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1890419427859935275)，**Fireworks 供应商**经历了宕机，但 OpenRouter 确认其他供应商和 BYOK 使用未受影响。
   - 故障已于 **ET 时间 9:12** 解决，随后不久恢复正常运行。
- **OpenAI o1 和 o3 模型上线**：OpenAI 的 **o1 和 o3 模型**现已面向所有 OpenRouter 用户开放，无需单独的 BYOK 密钥，这允许更高的速率限制，详见 [OpenRouter API](https://openrouter.ai/api/v1)。
   - 公告中包含了一份 **模型后缀速查表**，如 `:online`、`:nitro` 和 `:floor`，对应不同的功能和定价。
- **DeepSeek R1 性能出现波动**：用户报告 OpenRouter 上的 **DeepSeek R1** 经常出现停顿，给他们的 Agent 造成了困扰，并引发了对其生产环境可靠性的担忧，但在某些设置下它似乎具有卓越的推理能力。
   - 根据 [DeepSeek 官方推文](https://x.com/deepseek_ai/status/1890324295181824107)，DeepSeek 建议在不使用 system prompt 的情况下将 temperature 设置为 **0.6**。
- **API 密钥被划掉**：用户发现他们的 API 密钥在网站上显示为删除线并返回 401 错误，管理员表示密钥可能因潜在泄露而被禁用。
   - 这凸显了保护密钥的重要性，并提醒用户使用 secrets 管理工具。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 的 'Deep Research' 功能让用户感到兴奋**：用户对 **Perplexity** 新推出的 *'Deep Research'* 功能感到兴奋，一些用户甚至在免费层级也能使用，引发了对使用限制的好奇。
   - 成员们认为 **Perplexity** 是首选的新闻来源，因为它被认为偏见较低且具有互动功能，是传统新闻的理想替代方案。
- **GPT Store 发布受困于隐私政策问题**：一名成员报告在尝试发布到 **GPT Store** 时收到错误消息，提示需要有效的隐私政策 URL。
   - 另一名成员建议更新 Action 中的隐私政策字段可以解决此问题，原成员确认这确实修复了问题。
- **讨论 ChatGPT 和 Playground 的差异**：成员们对比了 **ChatGPT** 和 **Playground** 的使用，强调了识别和解决响应错误以及识别模式的重要性。
   - 一名成员建议 Prompt 的设计应追求清晰，使模型能够清楚地预测用户意图，从而增强其可靠性。
- **处理 Prompt 解析冲突**：成员们建议要求 AI 模型对比 Prompt 的不同解析方式，这有助于发现冲突和歧义。
   - 他们还建议使用清晰、自然的语言而非严格的格式，以引导 AI 给出更有见地的回答。
- **人工监督对于 AI 辅助任务仍然至关重要**：讨论强调了在所有 AI 辅助流程中进行人工监督的迫切需求，特别是在立法写作等对准确性要求极高的敏感领域。
   - 强调必须由熟练的人员对所有 AI 生成的内容进行验证和评判，确保对最终内容负责。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD 用户面临 Lora 训练限制**：一位用户分享了仅用 *7 张自拍* 训练 **Lora** 的经验，导致特征识别有限（尤其是侧脸），建议使用更大规模的高质量图像数据集会更有效。
   - 较小的模型泛化效果可能较差，需要与目标输出风格匹配的图像才能获得最佳结果。
- **社区探索 AI 图像生成**：成员们讨论了生成 **AI 艺术** 的方法，解决了跨多个模型实现一致角色设计等挑战，并推荐使用 **FaceFusion** 进行换脸。
   - 关于自动处理图像请求的咨询引发了对 **ComfyUI** 工作流的需求讨论，以实现更强的控制和自动化。
- **成员通过控制设置微调 Stable Diffusion**：一位用户询问如何通过控制机制微调 **Stable Diffusion** 以改进图像生成，并被引导至 **L3 discord** 获取资源。
   - 该用户对近期增强图像生成过程控制能力的工具表现出浓厚兴趣。
- **Windows 音频设备检测令人沮丧**：一位成员幽默地评论了 **Windows** 检测音频设备的怪癖，开玩笑说理想的硬件解决方案可以改善检测过程。
   - 讨论转变为关于技术挫败感的轻松调侃，一些人提到尽管计算设备存在缺陷，但人们对其高度依赖的矛盾现象。
- **新人受到活跃社区的欢迎**：新用户介绍了自己，分享了他们在 **AI 艺术** 方面的经验，并就使用 AI 工具和模型时遇到的挑战寻求建议。
   - 现有成员欢迎新人的加入，展示了专注于交流 AI 艺术生成知识和经验的活跃社区氛围。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepHermes-3 展示推理实力**：**DeepHermes-3 Preview** 已发布，通过切换功能以计算量为代价换取准确性，展示了先进的推理能力，可在 [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) 上获取。目前正针对 Tülu 等模型进行基准测试。
   - #[ml-drama] 频道中有人提出质疑，称 DH3 在开启推理功能时仅突出了*两项特定的评估（evals）*，而关闭推理时则显示所有指标。
- **关于 Open Weight 定义的激烈辩论**：围绕 **Open Weight 定义** 的讨论强调了在 [Open Weight 网站](https://openweight.org/)上免费重新分发模型权重的合规性，引发了热烈辩论。
   - 该定义的含义及其对开源 AI 实践的潜在影响是讨论的核心点。
- **英国将重心从 AI Safety 转向 AI Security**：据 [TechCrunch 报道](https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/)，英国政府将其 **AI Safety Institute** 更名为 **AI Security Institute**，将重点转向针对 AI 风险的网络安全。
   - 社区成员表示担心，这一转变会削弱对 AI Safety（人工智能安全）的关注。
- **DeepSeek-R1 部署引发热潮**：**DeepSeek-R1** 的部署备受关注，根据[官方建议](https://fxtwitter.com/deepseek_ai/status/1890324295181824107)，推荐设置包括不使用系统提示词（system prompt）且温度值（temperature）设为 **0.6**。
   - 用户强调了使用官方部署的重要性，以确保获得与官方版本相似的体验，并减轻潜在的绕过问题。
- **xAI 计划进行大规模数据中心扩张**：据 [The Information](https://www.theinformation.com/briefings/musk-looks-for-another-data-center-for-xai-nears-5-billion-chip-deal-with-dell) 报道，Elon Musk 的 **xAI** 正在寻找新的数据中心，以支持增加的 **Nvidia** 芯片使用量。
   - 这一扩张信号表明了在竞争激烈的 AI 领域中雄心勃勃的增长努力。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM 成为 24/7 导师**：一位用户描述了 **Notebook LM** 如何通过从大量阅读材料中创建详细摘要和要点，改变了他们的医学学习流程，称其*简直是一个触手可及、全天候在线的私人导师*。
   - 该用户强调了该工具在学习中的易用性和实用性。
- **Z 世代俚语让学习变得有趣**：一位成员强调了自定义提示词（prompts）使用 **Z 世代“脑残式”社交媒体俚语（brainrot social media slangs）**来解释复杂概念的有效性。
   - 这种方法帮助他们用更通俗易懂的语言掌握了困难的学科，使学习变得更加轻松。
- **PDF 上传饱受神秘 Bug 困扰**：一位用户报告称，无论文件大小或复杂程度如何，上传 PDF 都会遇到困难；而其他用户则表示没有问题。这表明问题可能与用户的浏览器或处理潜在敏感内容时的系统安全过滤器有关。
   - 其他成员能够毫无困难地上传文件。
- **Notebook LM 的语言支持遇到障碍**：用户报告称，即使上传了相应语言的源文件，也很难让 Notebook LM 以选定的语言（如保加利亚语和德语）进行回答；不过其他用户报告称其工作正常。
   - 一些人发现使用特定 URL（如 [notebooklm.google?hl=bg](https://notebooklm.google?hl=bg)）可以成功支持保加利亚语。
- **Gemini 模型功能尚不明朗**：几位用户询问了新 **Gemini 模型** 的功能，特别是它如何集成到 **Notebook LM** 中。
   - 回复显示，目前对于 Gemini 在该平台内的具体能力尚不确定，用户指向了相关资源以供探索。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLM 利用潜在推理**：一篇新论文介绍了 LLM 中的 **latent reasoning**（潜在推理），这种推理发生在模型生成 Token 之前的隐藏空间中，与 Chain of Thought 方法形成对比，详见[此推文](https://x.com/MatthewBerman/status/1890081482104008920?t=V3aeg7FX8ZvIKtvhHtl-xA&s=19)。
   - 社区成员正在积极讨论这种方法的实际影响和潜在好处。
- **Nvidia 的 Veo 2 增强视频创作**：Nvidia 的新模型 **Veo 2** 在 YouTube Shorts 上亮相，创作者可以使用 **Dream Screen** 功能通过文本提示词生成视频片段，正如[此推文](https://x.com/GoogleDeepMind/status/1890054036168356283)所宣布的那样。
   - 这实现了用户生成内容的无缝集成，增强了叙事能力。
- **Apple 预热新设备发布**：**Tim Cook** 在[他的 X 动态](https://x.com/tim_cook/status/1890068457825394918)中预热了即将到来的 Apple 发布会，暗示了可能的新产品，如 **iPhone SE**、**M4 Air** 以及更新的 Apple TV 选项。
   - 推测包括**带屏幕的 HomePod** 以及进一步集成用于 AI 能力的强大芯片，引发了社区关注。
- **DeepHermes 3 瞄准卓越的 LLM 能力**：Nous Research 的 **DeepHermes 3** 模型已在 [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) 上可用，旨在将推理和传统的 LLM 响应模式合并到单一架构中。
   - 目标是大幅提高 LLM 的标注、判断和 Function Calling 能力。
- **社区分享养蜂业务计划**：一位成员在[此链接](https://docs.google.com/document/d/1BIBInKu1rET9-BC520OXXz95XuFOxuMAr63nytI9DZQ/edit?usp=sharing)分享了一份全面的**养蜂可行性报告**，为潜在的商业策略提供了可操作的步骤和见解。
   - 围绕研究和优化深度研究提示词的讨论，丰富了社区对在实时项目中利用 AI 的理解。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 拥抱 Google Cloud**：LlamaIndex 引入了与 [Google Cloud 数据库](https://twitter.com/llama_index/status/1890109073615626388)集成的新功能，便于将其用作初始数据存储和向量存储。
   - 这些集成旨在实现**简单且安全**，从而简化数据库交互。
- **LlamaParse 功能增强**：一段关于 [LlamaParse](https://twitter.com/llama_index/status/1890499579214491967) 的详细视频展示了各种解析模式、输出格式以及使用解析指令提高质量的技术。
   - 视频涵盖了解析**音频**、**图像**以及利用 **JSON 模式**获得优化结果的内容。
- **AgentWorkflow 被认为不适合 RAG**：`AgentWorkflow` 是为执行任务的 Agent 系统设计的，而非 RAG，如[文档](https://docs.llamaindex.ai/en/stable/examples/workflow/rag/)中所述。
   - 建议用户创建自定义函数，以便在 `AgentWorkflow` 中集成 RAG 处理。
- **`uv` 工具加速环境管理**：用户分享了使用 `uv` 创建多个虚拟环境的好处，并就管理 PyTorch 等工具的不同版本分享了见解。
   - 一位用户甚至提供了一个 Shell 函数来简化环境与相关项目文件之间的切换，以提高便利性。
- **印度 AI 社区发出邀请**：加入印度增长最快的 AI 社区的邀请，旨在促进联系与协作，邀请成员在人工智能领域进行创新。
   - 感兴趣的人士可以通过提供的 [WhatsApp 链接](https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU)加入社区，成为这一不断壮大的场景的一部分。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama 声望超越 OpenRouter**：**Glama** 凭借其**更低的成本**、**更高的速度**和**隐私保证**，正逐渐成为优于 [OpenRouter](https://glama.ai/models/) 的首选，尽管其支持的模型数量较少。
   - Glama 在不同模型上的定价范围从 **$0.06 到 $10** 不等，这为优先考虑效率和保密性的开发者提供了更好的平衡。
- **OpenWebUI 经常出现故障**：用户报告称 **OpenWebUI** 在进行微小更新时经常出现**破坏性变更 (breaking changes)**，影响了大部分社区功能的使用。
   - 一些用户认为这是由于其作为*实验性 Alpha 软件*的状态，容易出现**竞态条件 (race conditions)**，从而增加了使用难度。
- **0.0.0.0 IP 地址引发混淆**：关于使用 IP 地址 **0.0.0.0** 的讨论十分激烈，特别是在它通常监听所有接口的**容器化环境**中。
   - 一些成员警告不要在 HTTP 上下文中将其作为目标地址，并强调了理解其正确用法对于**故障排除 (troubleshooting)** 的重要性。
- **发放 MCP Server Author 身份组**：成员们分享了他们的服务器链接和 GitHub 仓库，以获取 **MCP server author** 身份组。
   - 提供演示服务器项目或库的成员有资格获得**作者身份**。
- **Zonos TTS MCP 为 Claude 赋予声音**：[Zonos TTS MCP](https://github.com/PhialsBasement/Zonos-TTS-MCP) 服务器通过为 **Claude** 提供类似于 **CGPT** 的声音，增强了用户交互体验。
   - **Markdown 解释器**的加入预计将进一步改善 Claude 的**语调**，使其表现更接近理想状态。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **社区询问 RAG 评估方法**：一位计算机视觉专家向社区咨询评估其 **RAG 系统**的**指标 (metrics)**。该系统拥有稳定的检索设置，他特别寻求关于评估 **LLM** 或**检索架构**所用指标的指导。
   - 他们寻求在 **RAG 系统**中评估 **LLM** 或**检索架构**时推荐使用的指标。
- **Tinystories 不仅仅是预训练模型**：成员们澄清说，**Tinystories** 不仅包含一组预训练模型，还包括一系列架构、一个数据集以及一篇详细介绍设置过程的研究论文。
   - 他们强调 Tinystories 完成了从小型模型中获得连贯输出所需的艰苦工作，对于初学者非常有用。
- **延迟归一化 (Normalization)**：一项讨论探索了通过延迟归一化来提高生成序列模型中的 **RL 性能**，认为不规则性可能是有益的，并建议使用**动态 Logits**。
   - 策略包括使用动态 Logits 并结合 **SFT**，以引导模型在训练中产生有意义的结果。
- **AI 在没有 Token 的情况下思考**：一段 [YouTube 视频](https://www.youtube.com/watch?v=ZLtXXFcHNOU) 探讨了**模型是否可以在不使用 Token 的情况下“思考”**，提出了一个关于 **AI 能力**的有趣问题。
   - 一篇 [arXiv 论文](https://arxiv.org/abs/2502.05171) 提出了一种新型语言模型架构，通过在潜空间 (latent space) 中进行推理来扩展测试时计算 (test-time computation)，而无需专门的训练数据。
- **公共模型发布不一致**：根据[这篇论文](https://arxiv.org/abs/2409.10472)对 **Hugging Face** 上 **52,227 个 PTLMs** 的实证研究显示，**40.87%** 的模型权重更改未在命名习惯或文档中体现。
   - 这些结果突显了预训练语言模型 (Pre-trained Language Models) 在命名规范和训练文档可访问性方面的**模糊性**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 执行严格的 PR 提交规则**：贡献者必须反复检查 PR 中的空格更改；不鼓励提交包含 AI 生成代码的内容，以节省时间并鼓励独立编码。
   - 指南强调了亲手编写代码以及使用 AI 获取反馈的重要性，而不是直接提交 AI 生成的代码。
- **关于 Kernel 和 OptOps 速度悬赏的见解**：一位成员提议创建一个 **OptOp**，以便在 `sum` 悬赏的背景下针对多次归约 (multiple reductions) 优化 **AST**。
   - 他们对当前 **OptOps** 的表达能力表示担忧，并建议探索用于多个累加器的 **GROUP OptOp**，预计渲染器大部分情况下应能按预期工作。
- **WSL 上的 VIZ 故障排除**：一名用户报告在 **WSL Ubuntu** 上使用 `VIZ=1` 时，由于访问临时目录的问题而出现错误。
   - 另一位成员承认 **WSL** 构建可能很困难，尤其是在使用 Python 时，并表示愿意通过下载所需环境来调查该问题。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 在高级用例中完胜 LangChain**：成员们表示，如果用户需要优化，或者相比字符串 prompts 更倾向于编写 Signatures 和 Modules，那么 **DSPy** 比 **LangChain** 更具优势。
   - 有人指出，如果需要预封装的解决方案，**LangChain** 可能是更好的选择。
- **DSPy 2.6 更新日志浮出水面**：一位用户询问了 **DSPy 2.6** 的更新日志，特别是关于 *Signatures* 的“instructions”部分；一名成员指出，这些指令自 **2022** 年以来就一直存在。
   - 该用户被引导至 [GitHub release 页面](https://github.com/dspy/dspy/releases) 以获取有关更改的详细信息。
- **DSPy 移除 Assertions 引发困惑**：在 **DSPy 2.6.3** 中移除 **dspy.Assert**、**dspy.Suggest** 和 **dspy.Retry** 的做法导致了关于向后兼容性和合适替代方案的困惑。
   - 一位成员推测，此次移除是引入 *assertions v2* 计划的一部分，尽管目前尚未提供官方路线图或解释。
- **DSPy 应对多标签分类**：一位用户寻求关于使用 **DSPy** 优化 **SLM** 以进行涉及 200 个类别描述的多标签分类的建议，并考虑采用批处理策略。
   - 该用户特别希望避免对模型进行 Fine-tuning 或使用多个 **LoRA adapters**。
- **DSPy Code Golf 受到关注**：一项 **DSPy** code golf 活动被提出，挑战社区成员创建简洁的代码片段。
   - 一位成员分享了一个用于从 HTML 中提取结构化数据的单行代码示例，邀请其他人参与这个可能演变成竞争性编程的游戏，并引用了 [Omar Khattab 的推文](https://x.com/lateinteraction/status/1890442615700545878)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 和 Mojo ❤️ 情人节**：**MAX 和 Mojo** 在这个情人节通过亲切的问候和在 **general** 频道分享的一张名为 `MAXMojoValentine.jpeg` 的趣味图片传递爱意。
   - 这一互动元素为频道带来了愉悦感和社区凝聚力。
- **v25.1 版本发布引发热议 🔥**：一位匿名用户宣布了 **v25.1** 的发布，获得了社区的热烈响应。
   - 感叹号和火焰表情符号表明用户对该版本带来的更新表现出极高兴趣。
- **Larecs 仓库备受关注 🌳**：一位成员提供了 [Larecs GitHub 仓库](https://github.com/samufi/larecs) 的链接，供感兴趣的人了解更多细节。
   - 树形表情符号暗示了对项目增长或开发的关注。
- **安全可变别名（Safe Mutable Aliasing）文档现身**：一位用户询问另一位成员编写的关于 **safe mutable aliasing** 文档的链接，后者分享了他们在 11 月发布的 [提案/愿景文档](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) 链接。
   - 该代码似乎会与通过别名参数访问的内存位置产生冲突。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **询问 Token 禁用配置**：一位成员询问是否可以通过配置文件禁用 Token，并承认这不是 GUI 中可用的功能。
   - 这反映了用户希望在官方支持的方法之外，对 Token 行为进行高级自定义的需求。
- **为 RTX 3080 推荐 Qwen2.5 Coder 14B**：讨论显示，将 **Deepseek** 的行为蒸馏到较小的模型上可能会导致 **RTX 3080** 上的性能下降，从而引发了对替代模型的建议。
   - **Qwen2.5 Coder 14B** 被推荐用于低 VRAM 配置，尽管成员们注意到了性能权衡。
- **讨论 LLM 微调限制**：一位成员询问如何使用 2021 年的数据更新和 Fine-tune **LLM**，得到的澄清是无法用新数据适配旧模型。
   - 这突显了使用较新数据集更新现有模型的局限性。
- **免费解锁 TradingView Premium**：分享了适用于 Windows 和 macOS 的 **TradingView** 免费破解版链接，并指出其庞大的用户群，同时附带了 [安装说明](https://www.reddit.com/r/TradingViewFree/comments/1hobjs6/tradingview_premium_ultimate_package_update/)。
   - 该帖子强调通过这种方法可以免费获得 **Premium** 功能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Dataloader Transform RFC 简化数据生成**：一位成员提出了一个 RFC，旨在添加 dataloader transform 和保存功能，从而增强训练时的在线 **DPO/GRPO 数据生成**。
   - 分享的一个示例展示了 **prompt_to_preference** 函数如何利用 `DataLoader` 生成偏好数据批次，表明了批量生成的可能性。
- **蒸馏缩放定律 (Distillation Scaling Laws) 引发讨论**：讨论集中在 [Apple 的一篇论文](https://arxiv.org/abs/2502.08606) 上，探讨了关于 **distillation scaling laws** 的问题，思考是从更强大的模型进行蒸馏更好，还是从头开始训练更好。
   - 一位参与者强调，在蒸馏过程中关于模型大小和能力的选择“非常复杂……”。
- **量化感知训练 (Quantization-Aware Training) 实现高精度**：一项新研究推进了对 **Quantization-Aware Training (QAT)** 的理解，探索了在量化表示下实现精度的方法，特别是 **8-bits** 的最佳位宽。
   - 该研究通过引用最前沿的研究论文 [arXiv:2411.04330v2](https://arxiv.org/abs/2411.04330v2) 得到了验证。
- **QuEST 方法在压缩方面媲美 FP16**：一位成员介绍了 **QuEST**，这是一种新的压缩方法，声称在模型权重和激活值为 **4-bits** 或更低时仍具有很强的精度。
   - 该方法被定位为 **与 FP16 具有帕累托竞争力 (Pareto-competitive)**，据称在减小模型大小的同时提供了更好的精度。


---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Quiz 3 发布引发困惑**：一位成员报告了关于 **Quiz 3** 发布的困惑，最初无法在 MOOC 网站上找到它。
   - 该用户随后在 [Discord](https://discord.com/channels/1280234300012494859/1293323662300155934) 上发现了公告，解决了该问题。
- **新手寻求 AI/ML 训练建议**：一位新成员请求关于从何处开始学习 **AI/ML** 模型训练技术的指导。
   - 他们还在寻求资源推荐，以在初始训练之外进一步提升知识，并鼓励大家推荐课程和论坛。



---


**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1339644199321997325)** (621 messages🔥🔥🔥): 

> `LoRA Fine-Tuning, Model Training and RAG, PDF Data Extraction, AI Hardware Support, Model Evaluation and Performance` 


- **LoRA 微调的挑战**：讨论强调虽然 LoRA 可以帮助微调模型，但存在过拟合或引入灾难性遗忘的风险，特别是在数据集平衡较差的情况下。
   - 有人指出，在引入新信息的同时平衡模型的通用知识，对于保持性能至关重要。
- **利用 RAG 访问公司知识**：一位用户分享了他们实现本地托管模型以回答公司特定查询的意图，并考虑将数据转换为 JSON 以便更轻松地访问。
   - 建议在此任务中使用 RAG (Retrieve and Generate)，因为直接训练可能会使更新新信息变得复杂。
- **从 PDF 中提取数据**：社区讨论了从 PDF 中提取文本、图形和图像的方法，共识是将 PDF 转换为图像进行 OCR 往往会产生更好的结果。
   - 强调了准确处理 PDF 表格的挑战，并建议 Vision 模型可能对复杂布局具有更强的鲁棒性。
- **AI 硬件讨论**：关于 AMD 与 NVIDIA 在 AI 领域局限性的对话指出，CUDA 生态系统显著影响了 AI 训练工具的可用性和效率。
   - 用户评论了租用强大 GPU（如 H100）的可能性，并探索了从 Runpod 等平台获取免费额度用于 AI 任务。
- **模型评估和性能查询**：关于在 FP16 下运行 7B 模型的 VRAM 需求问题强调，软件和模型输入大小等许多因素都会影响性能。
   - 讨论了训练设置中对奖励函数 (reward functions) 的具体调整，以确保模型能够正确输出预期的答案格式。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG.1">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://huggingface.co/silx-ai/Quasar-1.5-Pro">silx-ai/Quasar-1.5-Pro · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/conceptual_guides/prompting">Soft prompts</a>：未找到描述</li><li><a href="https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model">微调 DeepSeek R1 (推理模型)</a>：在医疗思维链数据集上微调全球首个开源推理模型，为未来构建更好的 AI 医生。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>：又称持续微调 (Continued Finetuning)。Unsloth 允许您进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth 基准测试 | Unsloth 文档</a>：想知道 Unsloth 有多快吗？</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">量化</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 需求 | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://docs.unsloth.ai/g">Unsloth 文档</a>：未找到描述</li><li><a href="https://vast.ai/">租用 GPU | Vast.ai</a>：通过最佳的云端 GPU 租赁服务，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面允许公平比较所有供应商的 GPU 租赁。</li><li><a href="https://jina.ai/">Jina AI - 您的搜索基础，动力十足。</a>：一流的 Embeddings、Rerankers、LLM-reader、网页抓取工具、分类器。适用于多语言和多模态数据的最佳搜索 AI。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth 需求 | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benc">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth-zoo/tree/main/unsloth_zoo">unsloth-zoo/unsloth_zoo at main · unslothai/unsloth-zoo</a>：Unsloth 的工具库。通过在 GitHub 上创建账号来为 unslothai/unsloth-zoo 的开发做出贡献。</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1339759797003616257)** (9 条消息🔥): 

> `Wendel 对 Unsloth 的点名、RAG 实现、DeepSeek 的 AI 发布` 


- **Wendel 多次提到 Unsloth**：Wendel 在最近一段名为 [“拥抱即将到来的 AI 革命，使用安全的本地 AI！”](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) 的 YouTube 视频中重点介绍了 Unsloth，表达了对本地 AI 解决方案的兴奋。
   - 观众们庆祝了这些点名，其中一位注意到 Wendel 大约提到了他们的名字 *四次*。
- **用户受到 Wendel 信息的启发**：一位成员对 Wendel 的言论表示热忱，表示这与他们在工作受限的情况下推动本地 AI 工具的努力不谋而合。
   - 他们表示：“如果我的公司不允许我们使用 OpenAI 之类的工具，那么我可以本地构建它”来支持他们的团队。
- **关于 RAG 实现的讨论**：一位成员询问了有关设置 RAG (Retrieval-Augmented Generation) 实现的资源。
   - 另一位成员迅速推荐使用 **Llama Index** 或 **Haystack** 作为易于上手的起点。



**提到的链接**：<a href="https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ">拥抱即将到来的 AI 革命，使用安全的本地 AI！</a>：DeepSeek 的发布震撼了 AI 世界，我们正处于 AI 工业革命的边缘！Wendell 为您详细介绍了如何抓住这个机会...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1339652059217137694)** (244 条消息🔥🔥): 

> `DeepSeek R1 性能、使用 LORA 和 RAG 进行训练、GRPO 奖励函数问题、模型与 TPU 的兼容性、HPC 集群训练错误` 


- **DeepSeek R1 表现优于通用模型**：用户发现 **DeepSeek R1** 在回答中比其他模型更好地保持了性格和细节，而像 GPT 这样的通用模型往往会产生平淡、机械的回复。
   - 讨论涉及了特定的角色驱动应用，以及对群体交流对 AI 训练影响的担忧。
- **使用 LORA 和 RAG 训练模型的挑战**：一些用户遇到了模型生成不必要文本格式的问题，例如 `\n```user` 或 `\n```assistant`，这表明可能存在训练数据问题或潜在的过拟合。
   - 关于训练 Instruct 模型以生成更整洁输出的适用性提出了疑问。
- **GRPO 奖励函数和兼容性问题**：在尝试将预训练的奖励函数与 **Unsloth** 集成时，用户遇到了与 Llama 架构修改相关的错误，特别是 `LlamaAttention` 属性。
   - 对话围绕避免函数覆盖的潜在解决方法展开，强调了对旧版奖励模型进行独立处理的必要性。
- **训练中 TPU 支持的局限性**：用户发现 **GRPO** notebook 在 TPU 上遇到兼容性错误，明确限制使用 NVIDIA GPU 被视为实现更广泛兼容性的障碍。
   - 建议包括切换到 Google Colab 上的 NVIDIA A100，以成功执行 GRPO 方法。
- **性能优化和漫长的训练时间**：用户对漫长的训练时间表示担忧，有人指出训练步骤耗时超过一分钟，导致总训练时长延长。
   - 用户讨论了对 Batch Size 和训练参数的潜在调整，以便在不遇到 CUDA Out of Memory 错误的情况下优化性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://download.pytorch.org/whl/cu126">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb#scrollTo=KN6nELjXcRez.">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=kR3gIAX-SM2q"">Google Colab</a>: 未找到描述</li><li><a href="https://redux.js.org/introduction/getting-started">Redux 入门 | Redux</a>: 简介 > 入门：学习和使用 Redux 的入门资源</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb),">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">我们所有的模型 | Unsloth 文档</a>: 未找到描述
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1339829219009499148)** (9 messages🔥): 

> `RLHF Reward Modeling, Tülu 3 GRPO, Multi GPU Support in Unsloth, OLMoE Improvements, New Optimizer` 


- **RLHF Reward Modeling 探索**：分享了指向 [RLHF Reward Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling) GitHub 仓库的链接，提供了用于训练 RLHF 奖励模型的方案。
   - *Let's goooo*，展示了成员们对可用新资源的兴奋之情。
- **Tülu 3 GRPO 模型见解**：讨论集中在 Ai2 的 [Tülu 3 GRPO 报告](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B)上，强调了其显著的改进和开源特性。
   - 该模型在各种任务中表现出 state-of-the-art 的性能，成员们对 Ai2 的努力表示赞赏。
- **等待 Unsloth 的 Multi GPU 支持**：一位成员询问了 Unsloth Pro 计划中的 Multi GPU 支持情况，得到的更新是该功能目前仍不可用。
   - 回复表示希望该功能能“很快”添加。
- **发布 OLMoE 更新**：分享了关于 [OLMoE 改进](https://allenai.org/blog/olmoe-app)的链接，展示了 iOS 应用的新迭代。
   - 成员们对 Ai2 在 OLMoE 项目中的进展表示热忱。
- **发布新 Optimizer**：通过一个 [arXiv 论文](https://arxiv.org/pdf/2502.07529)链接分享了关于**新 Optimizer**的参考资料。
   - 这成为了社区进一步探索和实现的一个关注点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/RLHFlow/RLHF-Reward-Modeling">GitHub - RLHFlow/RLHF-Reward-Modeling: Recipes to train reward model for RLHF.</a>: 训练 RLHF 奖励模型的方案。通过在 GitHub 上创建账户为 RLHFlow/RLHF-Reward-Modeling 的开发做出贡献。</li><li><a href="https://huggingface.co/allenai/">allenai (Ai2)</a>: 未找到描述</li><li><a href="https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B">allenai/Llama-3.1-Tulu-3.1-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/YXTYbr3hiFU?si=yXVQhYu4szFn42Gz">An Unexpected Reinforcement Learning Renaissance</a>: 我们在语言模型研究中所处的时代，是一个普遍完全相信推理和新的 Reinforcement Learning (RL) 训练的时代...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1339643524978577469)** (2 messages): 

> `AI Engineering Summit 门票, Windsurf Wave 3 特性, Model Context Protocol, 可自定义应用图标, Turbo Mode 增强` 


- **赢取 AI Engineering Summit 免费门票！**：我们将送出 **3 张门票**，参加 **2 月 20-21 日**在纽约市举行的 AI Engineering Summit。活动包含独家体验，并有机会与 Windsurf 的产品工程负责人见面。
   - 有意者必须[填写表格](https://forms.gle/WM67ZgQngXaY4stq7)以获得资格，且仅限纽约地区居民参与。
- **Windsurf Wave 3 正式发布！**：Wave 3 引入了多项令人兴奋的特性，包括用于自定义工具调用的 **Model Context Protocol (MCP)** 以及面向 Mac 用户的**可自定义应用图标**。
   - 此外还实现了 **Turbo Mode** 增强和改进的 **Tab to Jump** 导航，详见 [Wave 3 完整博客文章](https://codeium.com/blog/windsurf-wave-3)。
- **Model Context Protocol 现已可用**：Cascade 现在支持 **Model Context Protocol (MCP)**，允许用户配置工具调用；无论结果如何，每次操作都将消耗一个 flow action 额度。
   - 用户可以通过点击 Cascade 输入工具栏中的锤子图标来设置 MCP，该功能面向所有个人方案开放。
- **面向 Mac 用户推出可自定义应用图标**：Windsurf 允许用户更改应用图标，选项包括 **Classic**、**Blueprint** 和 **Hand-drawn**，目前正面向 Mac 用户进行 Beta 测试。
   - 更改需要重启系统才能在整个操作系统中生效，该功能适用于所有付费用户方案。
- **Turbo Mode 增强功能发布**：Windsurf 的最新更新包括用于自动执行命令的 **Turbo Mode** 以及拖放图片支持。
   - 此次更新还改进了额度可见性，并扩展了 @docs 选项以提升用户体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/WM67ZgQngXaY4stq7">来自 Windsurf 的纽约 AI Engineering Summit 门票 </a>: 我们希望通过送出三张纽约 AI Engineering Summit 的免费门票来回馈社区！这场为期两天的活动将于 2 月 20-21 日举行，届时可以聆听顶尖 AI 专家的分享...</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>: 介绍 Wave 3，这是我们对 Windsurf 编辑器的第三批更新。</li><li><a href="https://x.com/windsurf_ai/status/1890161230876381249">来自 Windsurf (@windsurf_ai) 的推文</a>: Wave 3 来了！本次更新包含：⏩ Tab to Jump 🔗 MCP 集成 ⚡ Turbo Mode 🎨 自定义图标……以及更多。</li><li><a href="https://www.codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1339695800758046752)** (31 messages🔥): 

> `公告猜测, Codeium 扩展行为, 对 Windsurf 的不满, 扩展功能请求, Codeium 用户支持` 


- **对新公告的猜测**：人们对新公告充满期待，特别是针对纽约以外地区的用户，暗示今天晚些时候会有更多消息。
   - 讨论中表达了“请保持关注”更多惊喜的情绪。
- **Codeium 扩展的不同行为**：一位用户注意到 Codeium 扩展在 **Android Studio** 和 **IntelliJ IDEA** 之间的行为差异，寻求一致性。
   - 具体来说，用户希望这两个应用程序的聊天窗口都能在 **IDE** 内部打开。
- **对 Windsurf 占据主导地位的挫败感**：多位用户对过度关注 **Windsurf** 表示不满，认为这削弱了对 Codeium 扩展的讨论和支持。
   - 有人评论说，对于那些主要对编程工具感兴趣的人来说，转向 Windsurf 感觉像是一种“诱导转向 (bait-and-switch)”。
- **针对新模型的特性请求**：关于 Codeium 扩展未来是否支持 **Deepseek R1** 和 **Gemini 2.0 Flash** 等模型的咨询不断增加。
   - 用户被鼓励到 [codeium.canny.io](https://codeium.canny.io/feature-requests) 提交特性请求以表达他们的需求。
- **希望在支持讨论中进行明确区分**：在讨论中，用户渴望有一个专门针对 Codeium 扩展的空间，对话题混杂表示不满。
   - 呼吁建立一个专注于扩展程序的“更干净的频道”，反映了社区对清晰有序支持的渴望。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1339643336390086841)** (622 messages🔥🔥🔥): 

> `Cascade Base 问题, MCP Server 配置, Windsurf 性能, Windsurf 用户体验, Codeium 支持反馈` 


- **免费用户无法使用 Cascade Base**：用户目前在使用 Cascade Base 时遇到问题，特别是免费用户反映在更新后该功能无法按预期运行。
   - 许多免费和付费计划的用户都对无法正常登录或使用 Cascade 表示沮丧，一些人认为这可能与最近的更新有关。
- **MCP Server 配置问题**：围绕在 Windsurf 中配置 MCP server 展开了讨论，部分用户无法找到文档中详述的选项。
   - 用户确认需要遵循特定步骤才能有效地设置 MCP server。
- **Windsurf 性能与用户反馈**：多名用户报告在 Windsurf 中遇到延迟或无响应的情况，特别是在使用各种模型和功能时。
   - 有人建议改进用户体验，包括静用某些颜色的选项以及增强工作流效率。
- **Prompt 与使用效率**：用户讨论了 Prompt 结构的重要性，以优化与 Cascade 在代码编辑和调试任务中的交互。
   - 建议提供关于高效 AI 编程使用的清晰指令，以帮助用户避免陷阱。
- **支持与沟通反馈**：用户对 Codeium 支持服务的响应速度和有效性表示担忧，特别是关于账户访问和错误的问题。
   - 用户表示需要在社区频道中对当前问题和潜在解决方案进行更清晰的沟通。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:12345",">未找到标题</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/mcp">Cascade MCP 集成</a>: 未找到描述</li><li><a href="https://marketplace.visualstudio.com/items?itemName=avli.clojure">Clojure - Visual Studio Marketplace</a>: Visual Studio Code 扩展 - 为 Visual Studio Code 提供 Clojure nREPL 支持</li><li><a href="https://docs.codeium.com/windsurf">Windsurf - 开始使用</a>: 未找到描述</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://x.com/windsurf_ai/status/1889804247618945045/photo/1">来自 Windsurf (@windsurf_ai) 的推文</a>: 您可以通过使用 Markdown Preview Mermaid Support 等扩展打开文件来查看图表！</li><li><a href="https://www.reddit.com/r/codeium">Reddit - 深入了解一切</a>: 未找到描述</li><li><a href="https://tenor.com/view/just-do-it-shia-la-beouf-do-it-gif-4531935">Just Do It Shia La Beouf GIF - Just Do It Shia La Beouf Do It - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://status.codeium.com/">Codeium 状态</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=OIV1vKm59Xg">Windsurf Wave 3 更新：Tab 键跳转、MCP、自定义应用图标、Turbo 模式等</a>: Windsurf 第 3 波更新来了！🚀 了解使 Windsurf 更加强大的最新功能：Tab 键跳转 ⏩ 在文件内轻松导航以进行...</li><li><a href="https://github.com/renatokuipers/neural-child">GitHub - renatokuipers/neural-child: 一个通过从婴儿期到成熟期的心理阶段发展的神经网络系统，实现了情绪调节、依恋和心理理论能力。</a>: 一个通过从婴儿期到成熟期的心理阶段发展的神经网络系统，实现了情绪调节、依恋和心理理论能力。 - renatokuipers/neural-c...</li><li><a href="https://docs.djangoproject.com/en/4.2/topics/i18n/translation/">翻译 | Django 文档</a>: 为追求完美且有截止日期的开发者提供的 Web 框架。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1340013903475900517)** (1 条消息): 

> `Perplexity Deep Research, Deep Research features, Free queries, App availability, Research capabilities` 


- **Perplexity Deep Research 发布**：Perplexity 推出了 **Deep Research**，使用户能够通过自主执行搜索和分析来源，针对任何主题生成深入的研究报告。
   - 该功能可以处理各领域的专家级任务，并在 **Humanity's Last Exam** 中获得了高分。
- **免费访问及查询限制**：Deep Research 提供**免费**使用，非订阅用户每天最多可进行 **5 次查询**，而 Pro 用户每天最多可进行 **500 次查询**。
   - 这种分级访问旨在满足具有不同研究需求的更广泛受众。
- **App 推出详情**：**Deep Research** 目前已可在网页端使用，并计划在应用更新后很快推送到 **iOS, Android 和 Mac**。
   - 建议用户将 App 更新至最新版本以获得最佳体验。
- **Deep Research 用户指南**：要使用 Deep Research，用户应访问 [perplexity.ai](https://perplexity.ai)，并在提交查询前从搜索框中选择 “Deep Research” 模式。
   - 有关该功能的更多详细信息可以在[此处](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)找到。
- **Deep Research 视频介绍**：分享了一个 Deep Research 的介绍视频，旨在为用户提供其功能的视觉演示。
   - 视频可以直接通过 [DeepResearchVideo.mp4](https://cdn.discordapp.com/attachments/1047204950763122820/1340013907879661689/DeepResearchVideo.mp4?ex=67b0d0b3&is=67af7f33&hm=13e19e20d5268c195ec334aa20c9207933f4403069baca9ab6679faeb4581718&) 访问。



**提及的链接**：<a href="https://perplexity.ai)">未找到标题</a>：未找到描述

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1339644531456479322)** (601 条消息🔥🔥🔥): 

> `Perplexity Deep Research, AI 模型性能, 订阅计划反馈, 模型用户体验, Deep Research 搜索问题` 


- **关于 Deep Research 模型的澄清**：用户讨论了 Deep Research 目前使用的是 **o3-mini** 模型，并询问了其与 R1 相比的性能表现。
   - 用户对幻觉问题表示担忧，并质疑其是否有效地利用了目标模型的能力。
- **订阅与定价讨论**：关于 **o1** 和 **opus** 等模型的定价进行了对话，对其性价比和性能意见不一。
   - 用户强调某些模型有使用限制，并对 **12个月 PRO 订阅** 等计划的实用性表现出兴趣。
- **模型用户体验**：讨论了 Deep Research 的可靠性和速度，一些用户对其缓慢的性能和有限的来源表示沮丧。
   - 成员们分享了针对各种查询的测试，结果显示体验参差不一，包括搜索时间过长等问题。
- **AI 模型与功能的未来**：用户思考了推理模型的未来整合，并推测可能出现的功能，例如 **'Plexy'** 助手的潜力。
   - 对于此类功能是会增强用户体验还是增加复杂性，意见存在分歧。
- **Deep Research 搜索问题**：几位用户反映 Deep Research 搜索效果不佳，导致困惑且 AI 无法提供令人满意的回答。
   - 一位用户特别指出，在启用网页搜索功能后，仍然缺乏与相关内容的连接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-ai-tests-deep-research-feature-challenging-gemini-and-chatgpt/">Perplexity 测试 Deep Research 功能，挑战 ChatGPT</a>：发现 Perplexity 在 Alpha 阶段的新 Deep Research 功能，提供全面的网络分析和详细报告。预计将逐步推出。现在就试试吧！</li><li><a href="https://www.lefigaro.fr/sports/football/ligue-1/ligue-1-roberto-de-zerbi-prepare-deja-les-annees-suivantes-a-l-om-20250214">Ligue 1 : Roberto De Zerbi 已经在为马赛队的“未来几年”做准备</a>：这位马赛队教练在本周五的新闻发布会上透露，他已经在与管理层一起规划未来的赛季。</li><li><a href="https://tenor.com/view/hasbulla-hasbik-cute-meme-influencer-gif-21732737">Hasbulla Hasbik GIF - Hasbulla Hasbik Cute - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/testingcatalog/status/1890360609318801911">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：突发新闻 🚨：Perplexity 正在准备发布 Deep Research Alpha 🔥 由于此功能在 Gemini 和 ChatGPT 上运行，Perplexity 可能需要更多时间来浏览网络资源并利用其推理能力...</li><li><a href="https://x.com/perplexity_ai/status/1890452005472055673?t=6Rs4ecHRvoXxYA-hFU9yYw&s=19">来自 Perplexity (@perplexity_ai) 的推文</a>：介绍 Perplexity 上的 Deep Research。Deep Research 让你能够针对任何主题生成深入的研究报告。对所有人免费开放——非订阅者每天最多 5 次查询，订阅者 500 次查询...</li><li><a href="https://x.com/elder_plinius/status/1890028958907089059?t=Kv46N8eXldfN35QN-zmGhQ&s=19">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：MUAHAHAHA 💉💦引用 Djah 〰️ (@Djahlor) 什么？？？@elder_plinius 这是你做的吗？？</li><li><a href="https://x.com/perplexity_ai/status/1889366732432674961">来自 Perplexity (@perplexity_ai) 的推文</a>：我们很高兴地宣布，百万美元问题抽奖活动的获胜者是 Kaylee Edmondson！Kaylee 是来自田纳西州纳什维尔的一位小企业主。恭喜 Kaylee。感谢...</li><li><a href="https://tenor.com/view/fire-writing-gif-24533171">Fire Writing GIF - Fire Writing - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM 基准测试表</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1339648086435299340)** (18 条消息🔥): 

> `每日 Omega-3 剂量、通胀趋势、Musk 对 OpenAI 的竞购、ChatGPT 能耗、N8N JavaScript 使用` 


- **每日 Omega-3 剂量可能延缓衰老**：最近的一篇文章讨论了**每日服用 Omega-3** 如何延缓衰老过程，为饮食习惯带来了有趣的见解。你可以在[这里](https://www.perplexity.ai/page/daily-omega-3-dose-could-slow-WOTeSIXYTRCpeznaXq9ilw)阅读更多相关内容。
   - *研究表明*，长期规律摄入 Omega-3 可以显著影响健康。
- **通胀意外上升**：Perplexity AI 强调了**通胀**意外上升的话题，指出了对经济的潜在影响。如需深入了解，请观看 [YouTube 视频](https://www.youtube.com/embed/39eW2gMthuU)。
   - 专家们正密切关注这些变化，并对**经济稳定性**表示担忧。
- **Musk 威胁撤回对 OpenAI 的竞购**：Elon Musk 表示，如果 **OpenAI 保持非营利性质**，他将撤回竞购，这引发了对公司未来方向的质疑。完整报道请查看[这篇文章](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w)。
   - *这一举动引发了关于*利润动机对 AI 发展影响的讨论。
- **ChatGPT 的能耗可能被高估**：**ChatGPT 的能耗**受到审查，有说法称其在之前的报告中可能被高估了。完整的见解可以在这份[详细调查](https://www.perplexity.ai/page/chatgpt-energy-use-overestimat-cn02azRBR2._eM_sH2n_Pw)中找到。
   - 批评人士指出，了解实际能耗对于评估环境影响至关重要。
- **N8N 与 JavaScript 的集成**：有人询问如何在流行的自动化工具 **N8N 中使用 JavaScript**，提供了对其功能的见解。详细指南可以在[这里](https://www.perplexity.ai/search/como-usar-javascript-no-n8n-7c6Fo3WWRfCV0Aq1O0ClMg)找到。
   - 这种集成为寻求增强工作流的用户开启了**自定义选项**。



**提到的链接**：<a href="https://www.youtube.com/embed/39eW2gMthuU">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1339756492059836437)** (5 条消息): 

> `Sonar API Beta 测试、Aider 与 DeepSeek V3 集成、廉价编程工作流、Perplexity API 商业用例、Deep Research API 功能` 


- **渴望 Beta 测试 Sonar API**：一位成员表达了在 **Cerebras** 上测试 **Sonar API 版本**的强烈愿望，称他们已经梦寐以求好几个月了。
- **Aider + Sonar + DeepSeek V3 集成**：一位成员分享了一个集成方案，使用 **Aider** 进行推理，**Sonar** 负责架构，**DeepSeek V3** 作为编码组件，并附带了一张图片。
   - [在此查看图片](https://cdn.discordapp.com/attachments/1161802929053909012/1339881751291494480/Screenshot_2025-02-14_at_09.51.13.png)。
- **尝试廉价编程工作流**：一位成员分享了他们测试**“廉价”编程工作流**的经验，并带有一丝幽默感。
- **请求有关付款的官方邮件**：一位新成员请求关于 **Perplexity API 商业用例**的协助，并需要一封用于信用卡付款的官方邮件。
   - 他们提到很难得到客户服务的回复。
- **询问 API 中的 Deep Research**：一位成员询问 **Deep Research** 是否会包含在 **API** 中，寻求对即将推出的功能的澄清。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1339665348630413403)** (37 条消息🔥): 

> `Embedding Models, Vision Transformer Dimension, Open Deep Research Demo, Speech to Text Using Deepgram, User Interface Concerns` 


- **Embedding Models 过拟合担忧**：讨论指出许多大型 Embedding Models 往往在基准测试中表现出**过拟合**，其性能通常与较小的模型相似，但消耗的**计算量却高出 100 倍**。
   - 一位成员提醒要谨慎使用**“更好” (better)**一词，指出这取决于具体场景。
- **确定 ViTs 的最佳投影维度**：一位用户询问了在给定 patch size 和通道数的情况下，视觉模型的合适**投影维度 (projection dimension)**，并提到了论文中原始的 **768 维度**。
   - 对话强调了确保维度足够大的重要性，同时讨论了维度过小可能导致结果不佳的问题。
- **Open Deep Research Demo 问题**：一位用户报告了 **open deep research demo** 可能存在的停机问题，随后另一位成员迅速确认该问题现已**修复**。
   - 在修复后，该用户通过简单的确认表达了感谢。
- **Speech to Text 实现的挑战**：一位成员寻求关于使用 **speech to text 模型**进行自动语音识别的建议，并提到 Deepgram 的文档使用困难。
   - 另一位成员建议使用 **Whisper** 作为替代方案，表明正在寻找非 GCP/AWS 的解决方案。
- **关于用户界面元素的疑问**：一位用户询问了某个未指明选项的用途，另一位成员澄清了其在利用自有数据微调模型中的作用。
   - 对话强调了一个名为 **Autotrain** 的工具可以简化这一过程，成员们还分享了相关资源。



**提到的链接**：<a href="https://tenor.com/view/hmmm-thinking-batman-gif-6153870554148391864">Hmmm Thinking GIF - Hmmm Thinking Batman - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1339662003773378653)** (5 条消息): 

> `Neuralink Updates, Chat Templates and Transformers, QT Material and Layouts, Agent's Unit 1, Dataset-Tools Development` 


- **探索 Neuralink 视觉资料**：一位成员分享了与 **Neuralink** 相关的图片，引发了对其近期进展的讨论。
   - 这些图片引起了兴趣，但未详细阐述具体细节和分析。
- **讨论 Chat Templates 和 Transformers**：另一位成员在持续学习的背景下提到了 **chat templates 和 transformers**。
   - 虽然没有提供更多细节，但这突显了增强聊天机器人框架的趋势。
- **通过手动灵感学习 QT Layouts**：一位用户分享了学习 **QT material 和 layouts** 的过程，并结合使用 LLM 和 QT designer 获取灵感。
   - 尽管面临 **CPTSD** 带来的挑战，他们仍对自己的进步感到自豪，并表达了继续学习的决心。
- **Agent's Unit 1 讨论**：一位成员提到了围绕 **Agent's Unit 1** 的讨论，表明了对 AI 开发这一方面的兴趣。
   - 然而，并未提供详细的见解或问题来进一步推动对话。
- **Dataset-Tools 项目进展**：一位用户庆祝了其 **dataset-tools** 练手项目的成果，并指出该项目已接近完成。
   - 他们还反思了布局构想从产生到实现所花费的时间，并力求取得进一步进展。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1339696791830138912)** (7 条消息): 

> `Jokes Generator API, 面向 BlueSky 的 SciNewsBot, 浏览器引擎和 WASM` 


- **Jokes Generator 为 HuggingFace 带来幽默**：**Jokes Generator** 从 Joker REST API 获取笑话，并具有一个带有 Gradio 聊天界面的 UI，让用户可以享受一些欢笑。点击[这里](https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI)查看。
   - 一位用户对该工具仅 **97kb** 就表现得*极其出色*表示兴奋。
- **介绍用于每日科学新闻的 SciNewsBot**：**SciNewsBot** 在 BlueSky 上报道每日科学新闻，使用通过 Media Bias Fact Check 数据库过滤的事实核查来源。该机器人依赖于 [mistral-small-latest](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) 模型，并且是开源的。
   - 它能生成引人注目的标题，并且用户友好，允许任何人本地复现或通过 Docker 启动。
- **关于 WebAssembly 和浏览器引擎的讨论**：一位成员回忆说，类似的项目已经使用 **browser engines** 完成过，并建议了 **WASM** 在此背景下的相关性。另一位成员以轻松的评论回应了这一先验知识。
   - *对 Web 技术演进的评论*表明了对集成系统及其应用的熟悉。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI">Xo JokeGen NoAI - xyizko 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/AstraBert/SciNewsBot">GitHub - AstraBert/SciNewsBot: 一个简单的、由 AI 驱动的机器人，用于在 BlueSky 上报道有关环境、技术、科学和能源的每日新闻。</a>：A simple, AI-powered bot to report daily news about environment, technology, science and energy on BlueSky. - AstraBert/SciNewsBot</li><li><a href="https://bsky.app/profile/sci-news-bot.bsky.social">SciNews Bot (@sci-news-bot.bsky.social)</a>：嗨！我是一个由 Clelia Astra Bertelli 构建的 AI 驱动机器人，我每天发布关于环境、科学、技术和能源的最新新闻。我有一个内部事实核查算法，但欢迎你...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1339643528904572978)** (10 条消息🔥): 

> `技术困难, Zoom 会议, 会议录制, 演示反馈` 


- **技术困难导致会议调整**：小组今天遇到了 **technical difficulties**，促使成员们将注意力转向 Zoom 链接进行会议：[Zoom Meeting Link](https://mcgill.zoom.us/j/85033055096)。一位成员对中断表示歉意，表示会议可能无法按计划进行。
   - 尽管存在问题，大家仍对演示表示了感谢，反映了对所讨论论文背景信息的**赞赏**。
- **会议移至 Zoom**：会议已正式移至 Zoom，可通过[此链接](https://mcgill.zoom.us/j/85033055096)访问。尽管存在技术挑战，仍鼓励成员们通过数字方式参加。
   - 一位成员保证，会议将为那些不方便使用 Zoom 的人进行录制，确保事后的**可访问性**。
- **积极的演示反馈**：一位参与者对演示者表达了由衷的感谢，赞赏他们为论文增加的背景信息。这突显了社区内协作和知识共享的**价值**。
   - 另一位成员表示不得不提前离开，但表达了非常喜欢会议期间分享的额外见解。
- **为方便起见提供会议录制**：已保证将为任何不方便通过 Zoom 加入的人录制会议。这反映了小组对包容性的承诺。
   - 多样化的沟通语气（包括轻松的表情符号）显示了一个相互支持和积极参与的环境。



**提到的链接**：<a href="https://mcgill.zoom.us/j/85033055096">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1339690447219720265)** (1 messages): 

> `Canny Edge Detection, Sobel Filters, Machine Learning in Preprocessing, ControlNet with Diffusion Models` 


- **以 Canny 边缘检测和 Sobel 滤波器作为起点**：你可以从 **Canny edge** 或 **Sobel filters** 开始，如果检测任务绝对需要 **trained model**，再考虑构建流水线。
   - 这些方法在将 Machine Learning 应用于不同的下游任务之前，可以作为有效的**预处理阶段**。
- **ControlNet 利用边缘过滤图像**：**ControlNet** 将 **Canny edged filtered images** 与 Diffusion Model 结合使用，以确保生成的图像与原图保持**结构一致性（structural consistency）**。
   - 这强调了有时并不需要复杂的模型，因为边缘过滤器可以有效地增强**图像生成**任务。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1339675120100315226)** (10 messages🔥): 

> `Qwen Model Performance, Fine-tuning Issues, End Token Generation, Quality of Training Data, Chat Templates Knowledge` 


- **Qwen 2.5 模型并非 Base Model**：一位用户断定 **Qwen 2.5** 绝对不是一个 Base Model，暗示它具有更复杂的能力。
   - 他们注意到它的行为与未经 Fine-tuning 的模型不同，表明它对 Chat Templates 有所理解。
- **Qwen 的 Fine-tuning 挑战**：出现了关于使用 **1k 数据集**对 **Qwen** 进行 Fine-tuning 的担忧，特别是权重合并（weight merging）导致性能不佳的问题。
   - 一位用户对模型在合并后输出乱码表示困惑，暗示训练质量可能存在问题。
- **End Token 生成原理解析**：会议澄清了只有当 **end tokens** 极有可能是下一个 Token 时才会生成，这表明模型能够识别语言模式。
   - 讨论强调了通过训练避免 Token 生成陷入死循环的担忧，强调了有效 Fine-tuning 策略的必要性。
- **训练数据质量的重要性**：训练数据的质量成为焦点；较差的数据质量可能导致推理过程中输出乱码。
   - 观点表明，有效的 Fine-tuning 需要高质量的指令/回答对（instruction/answer pairs）以获得最佳性能。
- **Base Model 缺乏对 Chat Template 的感知**：会议指出 **Qwen 模型**的 Base 版本不理解 Chat Templates，这影响了它的交互能力。
   - 强调了 Base 模型和 Instruct 模型之间的区别，并确认在没有 Fine-tuning 的情况下，模型的对话能力有限。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Qwen2.5-3B">unsloth/Qwen2.5-3B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-3B-bnb-4bit">unsloth/Qwen2.5-3B-bnb-4bit · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1339661258487631955)** (2 messages): 

> `HF_TOKEN definition, Model changes` 


- **调查 HF_TOKEN 配置**：*检查日志*以确保你正确定义了 **HF_TOKEN**，因为它的缺失可能会导致问题。
   - 解决这个问题可能有助于处理潜在的配置问题。
- **更换模型的建议**：建议*尝试更换模型*作为排查步骤，这可能会产生不同的结果。
   - 这种方法有助于隔离问题并提高性能。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1339643569270423603)** (284 messages🔥🔥): 

> `课程介绍、证书问题、协作学习、Agent 开发、LLM 探索` 


- **课程参与者自我介绍**：来自印度、巴西、美国等不同背景和国家的新参与者介绍了自己，并分享了他们对 AI Agents 课程的兴奋之情。
   - 参与者表达了学习 AI Agents 以及在课程中与他人协作的热情。
- **课程证书遇到的挑战**：几位参与者报告了在完成单元测试后获取证书时遇到的问题，尽管已经登录，但仍反复出现登录提示。
   - 一些用户建议使用简单的故障排除技术，例如重新登录特定的 Space，以解决这些问题。
- **课程内容的翻译倡议**：一位成员提到将课程材料翻译成葡萄牙语，并提出为英语困难的学习者分享他们的笔记。
   - 有人建议建立一个集中的位置来存放课程内容的各种语言翻译，以提高可访问性。
- **关于构建自定义工具的讨论**：参与者讨论了如何使用提供的 Schema 开始构建自定义工具，一些人正在寻求更多资源以获得更清晰的指导。
   - 鼓励用户在学习课程材料时复习课程并与社区互动以寻求支持。
- **AI 中 LLM 的探索**：成员们对探索 LLM (Large Language Models) 作为构建 Agent 的基础模型表现出兴趣，并寻求相关列表或链接以进行进一步阅读。
   - 参与者参与了关于 LLM 能力和定义的讨论，思考如何界定它们的描述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course">欢迎来到 🤗 AI Agents 课程 - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/tutorial">让我们使用 smolagents 创建我们的第一个 Agent - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://www.youtube.com/live/iLVyYDbdSmM">欢迎来到 Agents 课程！课程介绍与问答</a>：在 Agents 课程的第一次直播中，我们将解释课程的运作方式（范围、单元、挑战等）并回答您的问题。不要...</li><li><a href="https://youtu.be/gHPhy1tL0Xg?si=XLNY8lvhWKRjLB_5">Python 初学者全课程 (2025)</a>：Python 是当今最通用且需求量最大的编程语言之一。从 Web 开发和人工智能到数据科学和自动化...</li><li><a href="https://youtu.be/Tr5_wgwHlCs?si=2x_t2NlGHN2LU4f1">NumPy 全课程（从入门到精通）</a>：这门综合课程将引导您从 NumPy 的基础知识走向高级技术。课程亮点：✅ 初学者友好型解释 ✅ 实用...</li><li><a href="https://youtu.be/TJ_iroAmfn8?si=nwQyhgdBAPH1OuTG">用于数据分析的 Pandas 全课程</a>：这门综合课程将引导您从 Pandas 的基础知识走向高级技术。课程亮点：✅ 初学者友好型解释 ✅ 实用...</li><li><a href="https://youtu.be/n6i2lRMbIKI?si=kkboVaAL6DBli2lt">用于数据可视化的 Matplotlib 全课程</a>：这门综合课程将引导您从 Matplotlib 的基础知识走向高级技术，帮助您在 Python 中创建令人惊叹的可视化效果。课程...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1339746840513478839)** (8 条消息🔥): 

> `DeepSeek V3, Granite 3.2 MoE, ESFT 论文回顾, 社区会议讨论` 


- **DeepSeek V3: Fire-Flyer AI-HPC 见解**：一段名为 ["DeepSeek 🐋 | Fire-Flyer AI-HPC"](https://youtu.be/wGWn3eVPvH8) 的 YouTube 视频讨论了用于 Deep Learning 的高性价比软硬件协同设计，强调了对计算能力和带宽日益增长的需求。
   - **Deep Learning** 和 **Large Language Models** 的进步被强调为这一日益增长需求的关键驱动力。
- **Granite 3.2 MoE 分析**：对 **Granite 3.2 MoE** 的预览表明，它似乎从 **GPT-3.5** 中蒸馏了数据，其训练数据仅截止到 **2021** 年。
   - 用户对其性能表示怀疑，质疑其成功的能力。
- **Expert Specialized Fine-Tuning (ESFT) 讨论**：一位用户询问了 **ESFT** 论文，并分享了该项目托管的 [GitHub 仓库](https://github.com/deepseek-ai/ESFT) 链接。
   - 该仓库专注于 **Expert Specialized Fine-Tuning**，旨在通过有针对性的训练提高模型性能。
- **关于社区会议的担忧**：一位用户提出了无法进行 **社区会议 (community calls)** 的问题，表达了对更多互动的渴望。
   - 这反映了社区内加强沟通与协作的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.14158">Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning</a>: Deep Learning (DL) 和 Large Language Models (LLMs) 的快速进展指数级地增加了对计算能力和带宽的需求。这与更快计算的高昂成本相结合...</li><li><a href="https://www.kaggle.com/code/stevugnin/open-r1-zero">Open R1 Zero</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自 [私有数据源] 的数据</li><li><a href="https://youtu.be/wGWn3eVPvH8">DeepSeek 🐋 | Fire-Flyer AI-HPC:  A Cost-Effective Software Hardware Co-Design for Deep Learning</a>: Deep Learning (DL) 和 Large Language Models (LLMs) 的快速进展指数级地增加了对计算能力和带宽的需求。这与...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/deepseek-ai/ESFT">GitHub - deepseek-ai/ESFT: Expert Specialized Fine-Tuning</a>: Expert Specialized Fine-Tuning。通过在 GitHub 上创建账号来为 deepseek-ai/ESFT 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1339643832991612951)** (333 条消息🔥🔥): 

> `Cursor IDE 易用性, AI 模型性能, MCP 服务器使用, 订阅问题, 工具集成困境` 


- **Cursor IDE 易用性困境**：用户在使用 Cursor IDE 时遇到困难，包括切换项目以及与 Composer 中新会话相关的问题，有时需要用户创建新会话以保持专注。
   - 用户抱怨 commit 消息生成的生成速度以及 IDE 内某些 AI 模型的性能。
- **AI 模型性能排名**：一个新的 AI Agent 排行榜已发布，Google 的 Gemini 2.0 和 OpenAI 的 GPT-4o 位居前列，这引发了关于 Sonnet 与 o3-mini 相比所处位置的讨论。
   - 参与者指出，该排行榜侧重于在工具集成和使用方面表现出色的 Agentic 模型。
- **MCP 服务器安装及问题**：用户讨论了在不同平台上设置 MCP 服务器，分享了社区成员创建的 [mcp-perplexity](https://github.com/daniel-lxs/mcp-perplexity) 等仓库链接。
   - 关于确保安装了 uvx 等必要工具以及如何在各种环境中有效运行这些服务器的建议。
- **订阅和定价变化**：用户对新的定价结构表示沮丧，现在使用 o3-mini 的请求计入高级 (premium) 请求，导致对服务可靠性的投诉。
   - 用户注意到，最初的免费使用似乎已经结束，而提供商并未进行明确沟通。
- **工具集成挑战**：讨论强调了让各种 AI 模型（特别是 o3-mini）在 Cursor 环境中有效集成并利用外部工具的挑战。
   - 提到了需要更好的 Prompting 技巧来改进工具调用 (tool calling) 功能和用户体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: 未找到描述</li><li><a href="https://www.galileo.ai/blog/agent-leaderboard">Introducing Our Agent Leaderboard on Hugging Face - Galileo AI</a>: 我们建立这个排行榜是为了回答一个简单的问题：“AI Agent 在真实世界的代理场景中表现如何？”</li><li><a href="https://tenor.com/view/funny-gif-27151298">Funny GIF - Funny - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/donvito/status/1890247522914038082?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Melvin Vivas (@donvito)</a>: 非常喜欢 Windsurf 对 MCP 的实现 😍 它是这样与 Cursor 对比的：✅ 在 Chat 模式下工作。我认为 MCP 在这里比 Agent 模式更有用。我也许是错的。✅ 轻松访问配置 MCP...</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity: MCP Server for the Perplexity API.</a>: 针对 Perplexity API 的 MCP Server。通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=wnHyYKNnUTo">Why Big Tech&#39;s Betting Billions on Nuclear (Not Renewables)</a>: 为什么 AI 需要核能？在 2 月 16 日前购买 Roborock Saros 10 可享受 200 美元优惠，链接：https://amzn.to/3CNzZQP &amp; https://bit.ly/3CH2pMm。随着新一波的...</li><li><a href="https://forum.cursor.com/t/supervisory-agent-to-guide-worker-agent/49395/7">&quot;Supervisory&quot; agent to guide &quot;worker&quot; agent</a>: Aider 表明这就是实现最佳 AI coder agent 的方式。此外，我认为实现这种流程的 Agent 对 Cursor 来说绝对是游戏规则改变者。所以，没错，非常棒的建议！</li><li><a href="https://tenor.com/view/darth-vader-alter-the-deal-empire-strikes-back-star-wars-gif-15971205">Darth Vader Alter The Deal GIF - Darth Vader Alter The Deal Empire Strikes Back - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://forum.cursor.com/t/mcp-servers-no-tools-found/49094/31">MCP Servers No tools found</a>: 谢谢，这终于对我起作用了。我也遇到了 Brave Search 的问题，但通过将 API key 添加到 Brave Search 文件中解决了。Sequential Thinking Server 设置，安装包...</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>: 介绍 Wave 3，这是我们对 Windsurf 编辑器的第三批更新。</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter: A lightweight Go application that parses JSON configuration files and executes commands with specified environment variables.</a>: 一个轻量级的 Go 应用程序，用于解析 JSON 配置文件并使用指定的环境变量执行命令。- daniel-lxs/mcp-starter</li><li><a href="https://www.npmjs.com/package/mcp-server-pagespeed">mcp-server-pagespeed</a>: 一个用于 Google PageSpeed Insights 的 Model Context Protocol server。最新版本：1.0.0，最后发布于 3 天前。通过运行 `npm i mcp-server-pagespee...` 在你的项目中使用 mcp-server-pagespeed。</li><li><a href="https://www.npmjs.com/package/mcp-mysql-server">mcp-mysql-server</a>: 一个用于 MySQL 数据库操作的 Model Context Protocol server。最新版本：0.1.0，最后发布于 11 小时前。通过运行 `npm i mcp-mysql-server` 在你的项目中使用 mcp-mysql-server。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1339642316985729144)** (154 条消息🔥🔥): 

> `LM Studio 中的错误处理, 模型性能对比, LM Studio 的 Headless Mode, Speculative Decoding 支持, 模型架构变更` 


- **Error Handling in LM Studio**: 用户报告在运行多个查询时，LM Studio 会抛出 'received prediction-error lmstudio' 消息，导致使用体验不佳。
   - 支持讨论指出，更新到最新版本可能会解决该问题，且某些 MLX 模型也出现了类似的错误。
- **Performance Comparison of Models**: 一位用户对 DeepSeek R1 在 M1 Air 16GB 上的表现进行了深入对比，指出低配机器的性能表现令人印象深刻。
   - 讨论中涉及了 Distilled 模型与全量模型的效能对比，对于质量和性能指标存在不同看法。
- **Headless Mode in LM Studio**: 一位用户询问如何在不显示 GUI 的情况下，在 Linux 服务器上以 Headless 模式运行 LM Studio。
   - 当前功能仍需要显示器来启动 GUI，但真正的 Headless 模式已计划在未来的更新中推出。
- **Speculative Decoding Support**: 多位用户表达了在使用 Speculative Decoding 时遇到的困难，提到了与下载模型之间的兼容性问题。
   - 讨论建议确保选择了 Beta Runtime，并检查模型规格以确认是否支持该功能。
- **Model Architecture Changes**: 用户讨论了模型训练的本质，特别是 DeepSeek R1 应该被视为 Fine-tune 还是全新的架构。
   - 对话揭示了对不同模型（包括 Dolphin 3.0 及其对比性能）的不同使用体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/lmstudio-ai/mlx-engine/issues/98">Unusual memory behaviour with Qwen2.5-VL models · Issue #98 · lmstudio-ai/mlx-engine</a>: 我一直在 LM Studio 0.3.10 Beta 1 中尝试 Qwen2.5-VL，我想我发现了一些奇怪的行为。我使用的是 mlx-community 的 Qwen2.5 VL 72B 4bit 模型。我运行的机器...</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: 加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation>)">GitHub - deepseek-ai/DeepSeek-R1</a>: 通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/issues">lmstudio-ai/mlx-engine</a>: 👾🍎 适用于 LM Studio 的 Apple MLX 引擎。通过在 GitHub 上创建账号，为 lmstudio-ai/mlx-engine 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/system-requirements">System Requirements | LM Studio Docs</a>: LM Studio 在 Mac (M1/M2/M3/M4), Windows (x64/ARM), 和 Linux (x64) 上支持的 CPU、GPU 类型。</li><li><a href="https://www.ebay.com/itm/275857855418">Nvidia P100-SXM2-16GB P100 PCIe 16 GB Tesla GPU  | eBay</a>: 未找到描述</li><li><a href="https://youtu.be/EHGmPn6RVwU">DeepSeek R1: $5000 vs $1000 Computer | M4 Max 128GB vs M1 16GB | LM Studio Tutorial</a>: 💻 正在考虑为 AI 升级你的 MacBook 吗？我测试了 DeepSeek R1 的开源模型在 5000 美元的 M4 Max (128GB) 与 1000 美元的 M1 Air (16GB) 上的表现 —— 结果...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1339685640178237500)** (172 messages🔥🔥): 

> `AMD ROCm 推广, NVIDIA RTX 3500 Ada, 2023 AI 硬件市场, 新硬件稳定性问题, 多 GPU 的 VRAM 性能` 


- **AMD 通过 ROCm 视频推动 AI 发展**：AMD 发布了一段[宣传视频](https://youtu.be/SB7Yt-FGWEs)，展示了如何使用 ROCm 软件平台在自家 GPU 上运行 LLM。
   - 这反映了 AMD 在竞争模型不断涌现的背景下，致力于巩固其在 AI 硬件市场地位的决心。
- **NVIDIA RTX 3500 Ada 系列的不确定性**：一位用户对在 Dell 笔记本电脑中发现的新款 NVIDIA RTX 3500 Ada GPU 表示好奇，并指出很难找到关于它的详细信息。
   - 他们推测 NVIDIA 的命名方式似乎有些随意，可能是通过重复使用数字并添加 Ada 后缀来对较新的产品进行分类。
- **新主板配置的稳定性困扰**：一位用户详细描述了安装新款 DDR5 主板时的挫折，该主板在 POST 阶段和 Windows 系统中导致了不稳定性，最终提交了退货申请。
   - 尽管更换了电源并尝试了各种配置，即使使用了受支持的 CPU，他们仍然面临崩溃问题。
- **关于 AI 工作负载中多 GPU 效用的讨论**：一位用户注意到，在使用双 3090 时，其 GPU 似乎是交替工作而非最大化利用，并将 VRAM 容量视为主要优势。
   - 对话暗示了未来更新中可能引入类似 VLLM 软件的实现，以实现更好的并行性能。
- **对 AMD 在 AI 领域竞争力的担忧**：关于 **Radeon RX 9070 XT** 的定价披露引发了对 AMD 在 2023 年 AI 任务可行性的怀疑，因为消费者更倾向于其他替代方案。
   - 有建议认为，以相当的价格购买两块二手的 3090 可能会提供更好的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/dram/sandisks-new-hbf-memory-enables-up-to-4tb-of-vram-on-gpus-matches-hbm-bandwidth-at-higher-capacity">SanDisk 的新型高带宽闪存（High Bandwidth Flash）可在 GPU 上实现高达 4TB 的 VRAM，并在更高容量下匹配 HBM 带宽</a>：为 AI GPU 配备 4TB 内存。</li><li><a href="https://en.wikipedia.org/wiki/Ampere_(microarchitecture)">Ampere (微架构) - 维基百科</a>：未找到描述</li><li><a href="https://x.com/GawroskiT/status/1890159776241447142">来自 Tomasz Gawroński (@GawroskiT) 的推文</a>：希望我发现的 9070 XT 价格只是占位符... https://www.mlacom.si/komponente/graficne-kartice/i_3026151_acer-predator-bifrost-radeon-rx-9070-xt-oc-16g-gddr6-graficna-kartica</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/qI6IFznwEB">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/YmDHEOZvYF">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)">Ada Lovelace (微架构) - 维基百科</a>：未找到描述</li><li><a href="https://youtu.be/SB7Yt-FGWEs">90 秒详解：在 AMD GPU 上运行 LLM</a>：了解使用 AMD ROCm™ 软件平台在 AMD GPU 上运行 LLM 是多么简单。这段视频展示了以下操作的简易性：• 安装 AMD ROCm™ 软件平台 ...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1339707671850254398)** (2 条消息): 

> `DeepHermes-3 Preview, Long Chain of Thought Reasoning, LLM Model Improvements, Community Feedback on Reasoning Models` 


- **DeepHermes-3 Preview 发布，带来令人兴奋的特性**：Nous Research 推出的新 [DeepHermes-3 Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) 结合了推理和直觉语言模型能力，提升了整体性能。
   - 该模型增强了 LLM 的标注、判断和 function calling，标志着其相对于前代产品 Hermes 3 的重大升级。
- **通过简单提示词解锁 Long Chain of Thought 推理**：要激活长推理模式，必须使用特定的系统提示词：`You are a deep thinking AI...`，这有助于在 `<think>` 标签内进行系统化推理。
   - 这一可切换的功能旨在提高准确性，但在测试期间可能会增加计算时间。
- **早期基准测试显示数学推理能力的提升**：初步评估表明，当激活 long chains of thought 时，**DeepHermes-3** 模型的**数学推理 (Mathematical reasoning)** 能力有显著增强。
   - 在 **GPQA** 基准测试中也观察到了小幅提升，表明该模型在处理复杂查询时具有鲁棒性。
- **社区协作对 DeepHermes 的开发至关重要**：DeepHermes-3 的开发得到了各社区成员贡献的支持，这对于增强数据集和评估工具至关重要。
   - Nous Research 鼓励社区持续提供反馈，以探索和改进该模型引入的新推理范式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1339649293719572544)** (217 条消息🔥🔥): 

> `DeepHermes-3 Preview, Deepfake 技术讨论, 模型训练与微调, 模型性能对比, 模型技术问题` 


- **DeepHermes-3 Preview 发布**：社区对 DeepHermes-3 Preview 模型的发布感到兴奋，强调了其在推理和直觉响应之间切换的能力，并提升了性能。
   - 用户已开始测试该模型，注意到一些重复输出的情况，并请求在 Nous Chat 上提供版本。
- **对 Deepfake 技术的担忧**：成员们讨论了 Deepfake 技术的潜在影响，对其可能的滥用以及有效监管的挑战表示担忧。
   - 鉴于现有的虚假信息问题，对于是否需要对恶意使用 Deepfake 的行为采取更严厉的惩罚，存在不同的意见。
- **模型训练与微调**：个人分享了在微调 AI 模型方面的经验和挑战，特别是使用 Colab 等资源，并建议了 LambdaLabs 和 Vast.ai 等替代方案。
   - 探讨了使用各种云平台训练模型的话题，成员们就不同服务的性能和可靠性提供了建议。
- **模型性能对比**：将 DeepSeek 的 distill 等模型及其在任务上的表现与 DeepHermes-3 进行了对比，强调了每种方法的优缺点。
   - 社区成员对模型之间的 Benchmark 感兴趣，以评估其在推理和对话任务中的能力。
- **模型技术问题**：用户报告了 DeepHermes-3 模型的各种技术问题，包括多轮对话期间的错误，这些错误阻碍了持续的推理输出。
   - 成员们正在排查这些问题，讨论为不同任务实现模型的复杂性，并建议进行代码调整。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>：未找到描述</li><li><a href="https://zed.dev/blog/edit-prediction">Zed now predicts your next edit with Zeta, our new open model - Zed Blog</a>：来自 Zed 博客：一个能预判你下一步操作的工具。</li><li><a href="https://huggingface.co/hance-ai/descript-audio-codec-44khz">hance-ai/descript-audio-codec-44khz · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF/tree/main">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF at main</a>：未找到描述</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/dreamworks2050/status/1890164583249375377">Tweet from M4rc0𝕏 (@dreamworks2050)</a>：DEEPHERMES-LLAMA-3-8B 思考模式：开启 - 首个 GGUF 运行 - F16 来自 @NousResearch 🔥MacBook Pro M4 Max : 28.98t/s</li><li><a href="https://f-droid.org/packages/superfreeze.tool.android/">SuperFreezZ App stopper | F-Droid - Free and Open Source Android App Repository</a>：完全冻结应用的所有后台活动。</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview#prompt-format-for-function-calling">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">Tweet from Nous Research (@NousResearch)</a>：介绍 DeepHermes-3 Preview，这是一款统一了推理和直觉语言模型能力的新 LLM。https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview DeepHermes 3 构建于...</li><li><a href="https://huggingface.co/Joseph717171/DeepHermes-3-Llama-3.1-8B-Preview-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF">Joseph717171/DeepHermes-3-Llama-3.1-8B-Preview-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kSNKuHX9AZo">BYD stock prices goes ballistic after they revealed this...</a>：比亚迪披露此事后股价飙升...澳大利亚最好的太阳能公司刚刚安装了我的新太阳能系统。在这里查看他们：https...</li><li><a href="https://www.youtube.com/watch?v=n8yva8YRVPU">Elon Musk says Grok 3 is going to be &#39;scary smart&#39;</a>：埃隆·马斯克表示 Grok 3 将在一两周内发布。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1339731890663325879)** (13 messages🔥): 

> `Llama-3B-Instruct 上的 SFT，微调本地 AI，语言模型的训练成本，1.5-Pints 技术报告` 


- **Llama-3B-Instruct 上的 SFT 挑战**：一位用户报告称，在对 **Llama-3B-Instruct** 进行 **SFT** 时，使用 **2e-4** 的学习率会导致由于特定领域因素而在 **Winogrande** 测试中出现显著的性能下降。
   - 另一位用户建议将学习率降低到 **5e-5**，并实施梯度累积（grad accumulation）以获得更好的归一化效果。
- **从零训练 1B 模型的成本**：一位用户询问了从零开始训练 **1B 模型**的相关成本，估计在数千美元，甚至可能达到数万美元。
   - 讨论显示，在代币（tokens）充足的情况下，使用消费级 GPU 大约需要 **6个月** 才能完成训练。
- **1.5-Pints 预训练报告的见解**：分享了 [1.5-Pints 技术报告](https://arxiv.org/html/2408.03506v1) 的链接，该报告详细介绍了一种在 **9天** 内预训练语言模型的方法，同时性能超越了之前的 SOTA 模型。
   - 该方法利用了一个精心策划的 **570亿 tokens** 数据集，重点关注说明性内容以增强推理能力。
- **关于微调方法的笔记**：成员们讨论了使用不同方法微调本地 AI 模型的经验及其背后的原因，强调了训练参数的重要性。
   - 建议包括调整学习率和考虑梯度累积以优化训练结果。
- **模型训练的一般建议**：几位用户交流了有效训练模型的技巧和调整方案，重点在于学习率调整和步数（step counts）。
   - 常见的建议包括降低学习率并关注全局 Batch Size 以提高性能。



**提及的链接**：<a href="https://arxiv.org/html/2408.03506v1#:~:text=Using%20our%20pre,16K%20context%20window%20version%2C%20which">1.5-Pints 技术报告：预训练只需数天而非数月——高质量数据助力语言模型成长</a>：未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339642667889328198)** (3 messages): 

> `LLM 报告论文，超稀疏内存网络，Kimik 和 Synthlab 论文，LLM 的推理速度` 


- **寻找最新的 LLM 报告论文**：一位成员正在寻找涵盖最新 SOTA 方法的 LLM 报告论文，特别是关于推理模型的，并指出 2024 年 2 月的一篇 LLM 综述论文现在已经过时了。
   - *teknium* 建议 **Kimik** 和 **Synthlab** 的论文与此搜索最相关。
- **UltraMem：LLM 效率的改变者**：在 [OpenReview](https://openreview.net/forum?id=zjeHLSiNv1) 上发表的一篇论文介绍了 **UltraMem**，这是一种超稀疏内存网络，可在不牺牲性能的情况下提高大语言模型的效率和可扩展性。
   - 研究结果表明，**UltraMem** 在推理速度方面比 **Mixture of Experts** 方法具有显著优势，同时展示了良好的扩展特性。



**提及的链接**：<a href="https://openreview.net/forum?id=zjeHLSiNv1">超稀疏内存网络 (Ultra-Sparse Memory Network)</a>：众所周知，Transformer 模型的性能与其参数数量和计算复杂度呈对数相关。虽然像 Mixture of Experts 这样的方法……

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339642667889328198)** (3 条消息): 

> `LLM report papers, Ultra-sparse memory network, Mixture of Experts, Scaling laws` 


- **寻找最新的 LLM 方法**：一位用户表示需要涵盖 **reasoning models** 等最先进方法的最新 LLM 报告论文，并暗示 2024 年 2 月的一篇综述论文已经显得过时。
   - 另一位成员提到 **r1 kimik** 和 **synthlab** 的论文与该搜索高度相关。
- **Ultra-sparse Memory Networks 塑造未来**：一篇关于 **UltraMem** 架构的论文揭示了大规模、超稀疏内存层如何在保持性能的同时，大幅提升 LLM 的 **efficiency** 和 **scalability**，尤其在 inference 速度上优于 Mixture of Experts。
   - 作者强调，这种方法降低了 **inference latency** 并研究了 **scaling laws**，展示了其优于现有 MoE 方法的 scaling 特性。



**提及的链接**：<a href="https://openreview.net/forum?id=zjeHLSiNv1">Ultra-Sparse Memory Network</a>：众所周知，Transformer 模型的性能与其参数数量和计算复杂度呈对数关系。虽然像 Mixture of Experts 这样的方法……

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1339723699187613788)** (11 条消息🔥): 

> `Eluther AI Research Contributions, Machine Learning and CS Projects, Identifying People in an Image` 


- **对研究贡献的好奇引发提问**：一位新成员表示有兴趣参与 **Eleuther AI** 的研究项目，并询问是否有任何开放项目或可提供的指导。
   - 他们正在寻求如何有效融入社区的建议。
- **学生在 AI 领域探索新途径**：几位用户（包括一名 **NLP 学生**和一名工程系学生）正在考虑转向 interpretability 和 deep learning 项目等领域。
   - 他们分享了自己的背景，并渴望获得关于如何开始贡献的见解。
- **识别共享图像中的面孔**：一位用户请求帮助识别一张图像中的人物，引发了讨论，大家将名字与面孔一一对应，其中包括 **Francois Chollet** 和 **Gary Marcus**。
   - 随着回复的迅速出现，这次对话凸显了社区知识的丰富，展示了强大的凝聚力。
- **图像识别中的社区协作**：另一位用户分享了第二张更新后的图像并请求确认身份，促使了进一步的回复和为了追求准确性而进行的 Google 搜索。
   - 社区成员高效协作，甚至有人标记了一份与图像相关的完整姓名列表。



**提及的链接**：<a href="https://x.com/IgorBrigadir/status/1534969529457070100">来自 ⚠️ Igor Brigadir 🇺🇦 (@IgorBrigadir) 的推文</a>：此图表中的所有人：左上：@fchollet @raphaelmilliere @GaryMarcus @tyrell_turing @ylecun @rohinmshah；右上：@sama @soniajoseph_ @ID_AA_Carmack @tszzl @demishassabis @michael_nielsen @sea_snel...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1339644371208900711)** (208 条消息🔥🔥): 

> `Attention Mechanisms, Scaling Laws in LLMs, Hybrid Architectures in Transformers, Forgetting Transformer, Long-Context Performance`

- **关于 Attention Sinks 和 QK Norm 的辩论**：讨论揭示了 QK Norm 虽然能稳定训练，但可能会阻碍对模型性能至关重要的 Attention Sinks；提出了 value residuals 作为一种可能的缓解方案。小组同意进一步研究这些关系及其对模型行为的影响。
   - 辩论了 Attention Sinks 的必要性与训练稳定性之间的权衡，强调了 forgetting transformers 作为维持注意力机制灵活性潜在解决方案的优势。
- **不同训练范式的性能**：介绍了两篇讨论 hyperfitting 和重复训练样本对大语言模型 (LLMs) 优势的论文，表明与数据多样性相比，重复可以增强性能。这些见解强调了在深度学习中平衡记忆与泛化的复杂性。
   - 对话探讨了模型在较小的重复样本上训练时，表现如何优于在较大数据集上训练，引发了关于训练方法对 LLM 能力影响的疑问。
- **语言模型预训练的新框架**：最近的一项研究提供了一个全面的框架，区分了双向上下文 (bidirectional context) 和注意力 (attention)，解决了以往在模型比较和评估中的挑战。研究结果表明，双向性的最佳使用高度依赖于具体应用。
   - 该研究指出了灵活训练配置的重要性，这会以不同方式影响 next token predictions 和 text infilling，强调了在模型训练中采用定制化方法的必要性。
- **长上下文 LLMs 与推理复杂度**：开发了一个新的基准测试 GSM-Infinite，用于评估 LLMs 在不同上下文长度和难度下处理推理复杂度的能力，揭示了性能随复杂度增加而呈现 S 形 (sigmoid) 下降。这突显了 LLMs 在应对需要广泛推理的智力问题时面临的挑战。
   - 讨论承认了进行定量评估以了解 LLM 在长文档推理中能力的必要性，增强了对其优势和局限性的见解。
- **Transformers 的 Scaling Laws**：提出了关于 Transformer 架构内单个组件 Scaling Laws 的疑问，特别是关于 head 数量与 residual stream 宽度之间的关系。对话强调了在设计高效模型时的权衡，并强调了优化架构配置的重要性。
   - 讨论了使用更大的 head size 而不是增加 head 数量的潜力，并指出了现有 kernel frameworks 中的实际限制，这可能会限制高效实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05483v1">Just read twice: closing the recall gap for recurrent language models</a>: 在语言建模困惑度（perplexity）上能与 Transformers 竞争的循环大语言模型（如 Mamba, RWKV）正在快速涌现。令人兴奋的是，这些架构使用恒定量的...</li><li><a href="https://arxiv.org/abs/2501.18795">Rope to Nope and Back Again: A New Hybrid Attention Strategy</a>: 长上下文大语言模型（LLMs）取得了显著进展，这得益于旋转位置嵌入（RoPE）及其扩展技术（Chen et al., 2023; Liu...</li><li><a href="https://arxiv.org/abs/2410.24159">GPT or BERT: why not both?</a>: 我们提出了一种将掩码语言建模（masked language modeling）与因果语言建模（causal language modeling）相结合的简单方法。这种混合训练目标产生的模型结合了两种建模范式的优势...</li><li><a href="https://arxiv.org/abs/2406.12335">Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters</a>: 扩展大语言模型（LLMs）的上下文尺寸使其能够执行各种新任务，例如书籍摘要。然而，注意力机制中 Key 和 Value (KV) cache 的内存开销显著...</li><li><a href="https://arxiv.org/abs/2410.13835">Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs</a>: 从业者在基于 Transformer 的大语言模型（LLMs）中一致观察到三个令人困惑的现象：注意力汇（attention sinks）、值状态耗尽（value-state drains）和残差状态峰值（residual-state peaks），统称为...</li><li><a href="https://arxiv.org/abs/2502.07490">Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More</a>: 大语言模型（LLMs）被发现在准确检索关键信息方面存在困难。为了解决这个问题，我们提出了掩码增强的自回归预测（MEAP），这是一种简单而有效的训练...</li><li><a href="https://arxiv.org/abs/2502.06268">Spectral-factorized Positive-definite Curvature Learning for NN Training</a>: 许多训练方法，如 Adam(W) 和 Shampoo，学习一个正定曲率矩阵，并在预处理前应用逆根号。最近，非对角线训练方法，如 Shampoo...</li><li><a href="https://arxiv.org/abs/2502.07529">Training Deep Learning Models with Norm-Constrained LMOs</a>: 在这项工作中，我们研究了利用范数球上的线性最小化算子（LMO）的优化方法。我们提出了一系列新的随机算法，利用 LMO 来适应几何...</li><li><a href="https://aclanthology.org/2022.findings-emnlp.293/">On the Role of Bidirectionality in Language Model Pre-Training</a>: Mikel Artetxe, Jingfei Du, Naman Goyal, Luke Zettlemoyer, Veselin Stoyanov. Findings of the Association for Computational Linguistics: EMNLP 2022. 2022.</li><li><a href="https://x.com/InfiniAILab/status/1890469309253841191?t=062o_qhbW0XRcux_v_GZ8w&s=19">Tweet from Infini-AI-Lab (@InfiniAILab)</a>: 🐭🐷 GSM-Infinite 由问题生成器生成。在零噪声基准测试中评估了 18 个强大的 LLMs，在长上下文基准测试中评估了 10 个 LLMs。🚀关键要点：最近的推理模型...</li><li><a href="https://arxiv.org/abs/2410.07041">Emergent properties with repeated examples</a>: 我们研究了 Transformer 的性能与算法生成数据集中训练示例重复次数的关系。在三个数学问题上：最大公约数...</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>: 本文介绍了在极小数据集上过度拟合预训练大语言模型（LLMs）的直觉相反的泛化结果。在开放式文本生成的设定下，众所周知...</li><li><a href="https://arxiv.org/abs/2502.05252">GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?</a>: 长上下文大语言模型（LLMs）最近在信息检索和长文档问答中表现出强劲性能。然而，为了解决最具挑战性的智力问题，LLMs 必须...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1339836434726518806)** (3 messages): 

> `OpenAI Deep Research, ML/AI Literature Reviews, Research Grounding Issues` 


- **关于 OpenAI Deep Research 的讨论**：一名成员询问是否有人尝试过 **OpenAI Deep Research**，以及它在 **ML/AI 文献综述**方面的效果。
   - 另一名成员回应称其效果*极佳*，但也表示在将其研究内容溯源（grounding）至 arXiv 内容和特定论文时存在挑战。
- **对研究质量的担忧**：一位参与者评论说质量似乎并不“出色”，对该工具的实用性表示怀疑。
   - 反馈强调了该工具可能存在依赖**不可靠博客**而非可信学术来源的问题。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1340009299337941054)** (8 messages🔥): 

> `Profiling talk recording, Zoom session feedback, YouTube stream` 


- **最新的 Profiling 讲座将被录制**：成员们确认关于 **profiling** 的最新讲座确实会被录制，并在 [YouTube](https://youtube.com) 上进行直播。
   - 该讲座目前正在 **Zoom** 上进行，稍后可供观看。
- **对 Profiling 讲座的热切期待**：一名成员表示渴望在讲座结束后观看录像，并询问结束后是否能提供链接。
   - 社区保证该环节将发布在包括 [YouTube](https://youtube.com) 和特定 Discord 频道在内的多个渠道。
- **新成员对 Profiling 见解的赞赏**：一位新成员感谢社区正在进行的 **Zoom** 会议，对提供此类知识表示感激。
   - 他们分享说，这些分析显著增强了他们对 profiling 及其功能的理解。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1339663553405386753)** (11 messages🔥): 

> `Fused MM Activation Implementation, GEMM Performance Insights, Kernel Caching Strategies, Triton Conference 2025, CUDA Thread Inquiry` 


- **针对非方阵的 Fused MM Activation**：一位用户正尝试在 Triton 中为维度为 M=2500, N=512, K=512 的非方阵实现 **Fused MM activation**，并正在寻求关于最快 tiled MM kernel 的指导。
   - 他们提到 [MM 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 与 **cutlass 实现** 相比具有竞争力，并表示需要有效的 autotuning 策略。
- **GEMM 配置建议**：一名成员建议，对于较大的矩阵尺寸，尤其是当 M 大于或等于 **128** 时，**A8W8 (persistent) GEMM** 实现将是最快的选择。
   - 他们强调了针对所使用的特定硬件最大化 autotuning 设置的重要性。
- **优化算法的 Kernel 缓存**：由于该用户的 **IDR 优化算法** 具有非固定的 M 大小，会影响性能，因此需要一种针对 Triton kernel 的缓存机制。
   - 有建议提出应用基于下一个 2 的幂（power of 2）的启发式方法，以减少针对不同形状进行频繁 autotuning 的次数。
- **关于 Triton Conference 2025 的询问**：一名成员询问是否已安排 **Triton Conference 2025**。
   - 目前对话尚未产生关于该活动的任何确认信息。
- **讨论中提及 CUDA Thread**：一名成员询问关于与 **CUDA thread** 相关的用户的存在情况，寻求对其目前参与情况的澄清。
   - 另一名成员对有关 CUDA thread 的具体引用表示不确定。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1339651158733750313)** (19 条消息🔥): 

> `Tensor 内存管理、GPU 访问问题、Torch 分布式训练错误、CUDA 技术相关性` 


- **探索 MatMuls 的 Tensor 内存管理**：讨论强调了 Tensor 内存如何以不同方式用于矩阵乘法，并建议在 Tensor 内存中使用累加器可以为其他操作释放寄存器。
   - 参与者指出，由于 Tensor 内存的大小不是 2 的幂次，可能存在效率低下的问题。
- **GB200 GPU 可用性方面的困扰**：一位用户对无法获取 GB200 GPU 表示沮丧，强调他们愿意为解决方案付费，但目前发现无法获得。
   - 其他人分享了替代供应商的建议，评论了 LLM 推理的巨大需求，但都面临等待名单。
- **Torch 分布式训练中的挑战**：一位参与者报告了在 Blackwell GPU 上运行 torch 分布式训练时遇到的困难，遇到了与 NCCL 相关的错误。
   - 尽管使用了最新版本，他们仍遇到持续的 CUDA 错误，这表明可能存在尚未解决的兼容性问题。
- **矩阵运算中线程计算的澄清**：关于线程如何计算 Tensor 分片（tiles）中条目的技术分解，澄清了矩阵运算中线程到计算任务的映射分配。
   - 该解释强调了 Tensor 大小与每个线程如何检索计算所需数据之间的相关性。
- **关于 CUDA 相关性的辩论**：一位用户引用一位教授的观点质疑 CUDA 是否已经过时，但得到了其他人关于 CUDA 当前和未来相关性的确认。
   - 参与者捍卫了 CUDA 的重要性，并试图了解怀疑者可能会提出哪些替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://link.excalidraw.com/readonly/heHY09sqiATU9DEpTU8J">Untitled scene - Excalidraw+</a>：在 Excalidraw+ 上查看未命名场景</li><li><a href="https://x.com/lambdaapi/status/1890028876954489125?s=46">来自 Lambda (@LambdaAPI) 的推文</a>：我们只知道我们的 NVIDIA HGX B200 没问题 🙂
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1339660835638874284)** (2 条消息): 

> `快速哈达玛变换 (Fast Hadamard Transform)、SageAttention2、Huggingface Transformers ONNX 问题` 


- **快速哈达玛变换在量化注意力中的作用**：讨论了为什么某些**量化注意力 (quantized attention)** 方法需要使用 [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform) 才能获得可用的结果，而像 **SageAttention** 这样的方法则不需要。
   - *SageAttention2* 提出通过量化技术加速注意力过程，从而显著提高效率。
- **Huggingface Transformers ONNX 转换问题**：一位成员在进行 ONNX 转换时遇到了 **Huggingface Transformers** 的问题，原因是 **Python jit tracing** 追踪了 `DacResidualUnit` 类中的一个布尔值。
   - 他们建议使用显式切片代替条件检查来确保兼容性，并在提交 PR 前寻求审查。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Dao-AILab/fast-hadamard-transform">GitHub - Dao-AILab/fast-hadamard-transform: CUDA 中的快速哈达玛变换，带有 PyTorch 接口</a>：CUDA 中的快速哈达玛变换，带有 PyTorch 接口 - Dao-AILab/fast-hadamard-transform</li><li><a href="https://arxiv.org/abs/2411.10958">SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization</a>：虽然线性层的量化已被广泛使用，但其在加速注意力过程中的应用仍然有限。为了进一步提高注意力计算的效率...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1340008465728409700)** (1 messages): 

> `NVIDIA Profiling Tools, Magnus Strengert Talk` 


- **NVIDIA 关于 Profiling 工具的演讲即将开始**：在 **45 分钟**后，NVIDIA Profiling 工具的首席架构师 **Magnus Strengert** 将带来一场关于 Profiling 相关内容的演讲，承诺将直接从官方源头分享宝贵的见解。
   - 鼓励成员们在 [Zoom](https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG) 上参加此次会议，因为据观察，关于该主题的公开内容非常稀缺。
- **直接学习的绝佳机会**：近期趋势显示，服务器上最受欢迎的一些演讲都围绕 **Profiling** 展开，这使得本次会议格外值得关注。
   - 与会者可能会获得在现有公开资源中不常见的深入见解。



**Link mentioned**: <a href="https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG.1">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1340049159369134202)** (1 messages): 

> `Roofline Model, Hierarchical Analysis` 


- **探索 Roofline Model 层级**：一位成员分享了关于 Roofline Model 的[优质资源](https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf)，强调了其层级结构。
   - 该文档提供了关于计算资源性能分析和优化策略的见解。
- **理解 Roofline 的重要性**：Roofline Model 展示了计算性能与内存带宽之间的权衡，这对于系统优化至关重要。
   - 正如分享的文档中所述，*它为旨在最大化计算效率的开发者提供了指南*。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1339808149326270475)** (1 messages): 

> `Oumi AI, Open-source models, ML performance engineers hiring, Collaborative AI development` 


- **Oumi AI 倡导开源开发**：[Oumi](https://oumi.ai) 的联合创始人 Oussama 分享道，他们的初创公司专注于构建完全开放的模型和基础设施，并坚信**开源能惠及所有人 (open-source lifts all boats)**。
   - 他强调了以公开协作的方式开发 AI 以实现更广泛利益的重要性。
- **Oumi 招聘 ML 性能工程师**：Oumi 正在积极招聘 [ML 性能工程师](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393)，以提升其模型速度和训练流水线。
   - 候选人将有机会为多个 [开源项目](https://github.com/oumi-ai/oumi) 做出贡献，并与专业的专家团队合作。
- **欢迎就潜在职位咨询进行联系**：Oussama 鼓励感兴趣的人士[直接申请](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393)，或通过 [DM](https://x.com/oussama_e) 或 [LinkedIn](https://www.linkedin.com/in/oussamaelachqar/) 咨询问题。
   - 他表示乐于进行讨论，为潜在候选人建立联系提供便利。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://oumi.ai).">未找到标题</a>: 未找到描述</li><li><a href="https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393))">Jobs</a>: 未找到描述</li><li><a href="https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393),">Jobs</a>: 未找到描述</li><li><a href="https://x.com/oussama_e)">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1339955572124024943)** (2 条消息): 

> `Fine-tuning Transformer Models, Colab GPU Issues, Alternatives to Colab for Training, Modal Platform Discussion` 


- **用户寻求微调的替代方案**：一位成员对 **Colab GPU** 表示失望，指出即使在使用 Pro 版本后，仍存在**稳定性**和**代码错误**问题。
   - *我还能在哪里训练我的 Transformer 模型？*
- **Modal 被赞誉为训练平台**：另一位成员推荐使用 **Modal** 作为模型微调的首选平台。
   - 在持续的挫败感中，他们强调支持选择这一方案而非 Colab。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 条消息): 

mubappe.: 是的，已解决，谢谢
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1339684519560613920)** (8 条消息🔥): 

> `Llama 3.3 License Issues, Llama Model Availability, Documentation and Code Sharing` 


- **获取 Llama 3.3 许可证的挑战**：一位用户在注册 **Llama 3.3 70B** Base 和 Instruct 模型的许可证时遇到困难，收到的消息称其不符合许可证标准。
   - 他们表示迫切需要解决此问题，以便为 **Cohere For AI** Discord 中的研究小组进行实验。
- **Llama 模型的替代访问途径**：另一位用户建议从 Hugging Face 获取 **70B-Instruct 版本**作为许可证问题的替代方案，并提供了一个[直接链接](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct)以提供便利。
   - 他们指出该平台上似乎没有可用的“Base”版本。
- **社区内的文档关注点**：一位社区成员承认自己在编写代码文档方面存在困难，并对处于类似情况的其他人表示同情，这表明了一个普遍支持的环境。
   - 这种对文档的反思暗示了许多开发者共同面临的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.llama.com/llama-downloads/.">Download Llama</a>: 请求访问 Llama。</li><li><a href="https://x.com/cataluna84/status/1881908149449547860">来自 Mayank Bhaskar (@cataluna84) 的推文</a>: Meta 的朋友们好，我无法获得 Llama 3.3 70B Base 和 Instruct 模型的许可证。这是我收到的错误消息：“感谢您有兴趣使用 Llama。不幸的是，...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1339698097592471636)** (1 条消息): 

> `User Defined Kernels, FSDP Usage` 


- **关于用户自定义 Kernel 的澄清**：一位成员指出，**用户自定义 Kernel** 应该不会有太多问题，并询问了所遇问题的具体细节。
   - 他们还寻求澄清在该问题背景下使用的是 **FSDP 1 还是 2**。
- **询问 FSDP 版本**：另一位成员询问了正在使用的 **FSDP 版本**，强调其对解决潜在 Kernel 问题的重要性。
   - 这表明需要统一哪个版本在当前设置中能带来更好的性能。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1339659282496356496)** (19 messages🔥): 

> `CUDA Kernel Optimizations, Low Bit Training Presentation, FP8 Training, Cohere AI YouTube Webinar, Polish LLM Training Pipeline` 


- **具有性能限制的高级 CUDA**：在 CUDA Kernel 中实现了诸如 **loop unrolling**（循环展开）、**tiled transposed matrix B**（分块转置矩阵 B）和 **warp level reductions**（Warp 级归约）等优化，但在不使用 cuBLAS 的情况下，仅达到了 **PyTorch 性能的三分之一**。
   - 一位成员指出，优化 CUDA Kernel 似乎存在某个瓶颈限制，这引发了关于性能维度和潜在优化策略的讨论。
- **关于低比特训练的 C4AI 演示**：一位成员宣布了一个关于 **low bit training**（低比特训练）的实时演示，并邀请其他人通过 [Google Meet link](https://meet.google.com/wdk-yipf-zjd?authuser=0) 加入讨论。
   - 另一位成员幽默地提到该活动正在立即进行，引发了参与者之间轻松的交流。
- **FP8 训练主题**：一位成员对 **FP8 training** 主题表示感兴趣，特别是关于优化其波兰语 LLM 训练流水线（该流水线目前需要大量的 GPU 资源）。
   - 出现了关于 FP8 训练相关演讲的建议，强调了社区在学习本地优化方面进行知识共享的需求。
- **Cohere AI 的 YouTube 资源**：一位成员提到最近的网络研讨会将在他们的 [YouTube channel](https://youtube.com/@cohereai?si=EhPxTDUf4OZPJEkm) 上发布，并分享了相关演讲的其他链接。
   - 这促使一位成员请求关于 **FP8 training** 的见解，表明人们对优化大模型训练效率的兴趣日益浓厚。
- **关于训练优化技术的讨论**：有人好奇在 **GH200 training** 中是否将 **optimizer states**（优化器状态）卸载到了 CPU，这涉及到优化训练工作流的影响。
   - 一位成员分享了他们讨论关于 **float8** 训练和支持技术的兴奋之情，表示已准备好进一步合作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/@cohereai?si=EhPxTDUf4OZPJEkm),">Cohere</a>：欢迎来到 NLP 的现在。我们正在通过让开发者和企业能够访问由最新一代大语言模型驱动的 NLP，开启机器学习的新篇章。我们的平台...</li><li><a href="https://github.com/prateekshukla1108/100-daysofcuda/tree/main/day18">100-daysofcuda/day18 at main · prateekshukla1108/100-daysofcuda</a>：为 100 天 CUDA 挑战编写的 Kernels。通过在 GitHub 上创建账号为 prateekshukla1108/100-daysofcuda 的开发做出贡献。</li><li><a href="https://ppc.cs.aalto.fi/ch4/v2/">Chapter 4: V2</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/)** (1 messages): 

alint5215: 它现在击败了 openblas。
  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1339859145771188304)** (5 条消息): 

> `Inference-time scaling, DeepSeek-R1 model, CuDNN frontend for flex attention, NVIDIA's performance benchmarks` 


- **NVIDIA 用于解决问题的推理时缩放 (Inference-time Scaling)**：NVIDIA 引入了一种名为“推理时缩放”的新缩放定律，允许 AI 模型在推理过程中分配额外资源来评估结果并选择最佳方案。
   - 该技术旨在增强模型策略，使其能够像人类解决问题的方法一样处理复杂挑战。
- **DeepSeek-R1 模型实验**：一项使用 NVIDIA 的 DeepSeek-R1 模型的实验展示了其自动生成数值正确的 GPU attention kernels 的能力，并针对特定任务进行了优化。
   - 然而，在有意义的性能基准测试中，功能正确与运行速度快的 kernel 之间的区别至关重要。
- **对 CuDNN Flex Attention 关联的疑虑**：一位用户质疑 NVIDIA 的发现是否与用于 flex attention 的 CuDNN 前端有关，因为其抽象程度较高。
   - 有人担心生成过程可能缺乏有趣的结果，建议需要进一步确认。
- **Kernel 生成中的功能性与速度**：一位合著者指出，仅仅产生功能正确的 kernel 并不意味着它们符合性能基准，因为速度是一个关键指标。
   - 虽然由于现有参考 kernel 的存在，功能正确性是可以实现的，但真正的目标是让模型生成的 kernel 超越当前的实现。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.00005">Mastering the Craft of Data Synthesis for CodeLLMs</a>：大语言模型 (LLMs) 在代码理解和生成方面表现出了令人印象深刻的性能，由于其应用价值，编码任务已成为研究人员关注的重点...</li><li><a href="https://arxiv.org/abs/2207.14502">Language Models Can Teach Themselves to Program Better</a>：最近的语言模型 (LMs) 在接受人类编写的问题训练后，在代码生成方面取得了突破性表现，甚至能解决一些竞赛编程问题。自博弈 (Self-play) 已被证明是有用的...</li><li><a href="https://arxiv.org/html/2402.15769v1">Importance Guided Data Augmentation for Neural-Based Code Understanding</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">Automating GPU Kernel Generation with DeepSeek&#x2d;R1 and Inference Time Scaling | NVIDIA Technical Blog</a>：随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为测试时缩放 (test-time scaling) 或推理时缩放 (inference-time scaling) 的新缩放定律正在兴起。也被称为 AI 推理...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1339642887360483329)** (114 条消息🔥🔥): 

> `Futoshiki dataset updates, Eval architecture discussions, Whitespace in answers, Scoring methods, Evaluation process improvements` 


- **Futoshiki 数据集的复杂性**：Futoshiki 数据集被证明比最初预期的更复杂，成员们承认在生成快速解决方案方面存在挑战。
   - 一位成员计划在跨模型测试后澄清问题格式，以提高评分效率。
- **评估架构 (Eval architecture) 改进**：讨论将所有与评估相关的代码移动到单独的仓库中，以保持 reasoning-gym 的专注度，并计划清理当前结构。
   - 成员们达成共识，将标准化评分策略并讨论答案格式，以减少输出的不一致性。
- **处理答案中的空白符 (Whitespace)**：有人担心前导和尾随空白符可能会影响答案评分，建议修改答案提取函数以更有效地处理这些情况。
   - 建议包括使用 regex 模式来准确解析答案，并重新评估答案的评分方式。
- **评分方法与改进**：建议为数据集实现 `score_answer` 方法以确保评估的一致性，特别是对于答案格式缺乏清晰度的数据集。
   - 一位成员表示打算对现有的评估脚本进行操作，以增强其功能。
- **评估流程后续跟进**：成员们正在积极更新评估输出，并在共享的 Google Sheet 中记录更改，确保评估方法的结构化。
   - 一位成员承诺根据既定检查清单恢复评估并协调与无信息提示 (uninformative prompts) 相关的任务。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1qk2BgxzfRZzTzMQnclCr47ioykgltbGkMJUHO2sH6Gw/edit?gid=1879785298#gid=1879785298">reasoning-gym-eval</a>: 未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/113">Rush Hour Gym by Iron-Bound · Pull Request #113 · open-thought/reasoning-gym</a>: 为益智游戏 Rush Hour 添加了一个 gym 环境。</li><li><a href="https://github.com/open-thought/reasoning-gym-eval/">GitHub - open-thought/reasoning-gym-eval: Collection of LLM completions for reasoning-gym task datasets</a>: reasoning-gym 任务数据集的 LLM 补全集合 - open-thought/reasoning-gym-eval</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/5d16a2193c142c827c11d3a7c85565849f779c33/eval/eval.py#L42-L59">reasoning-gym/eval/eval.py at 5d16a2193c142c827c11d3a7c85565849f779c33 · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账户，为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/60">Add Futoshiki puzzle generator by olliestanley · Pull Request #60 · open-thought/reasoning-gym</a>: 关闭 #54。现有的求解器非常混乱且难以理解，因此我最终实现了一个新的。即使在这段代码中，逻辑规则也不容易理解，但由于它们加快了速度，所以非常值得...</li><li><a href="https://github.com/Adefioye/AI-Playground/blob/main/eval/eval.py">AI-Playground/eval/eval.py at main · Adefioye/AI-Playground</a>: 通过在 GitHub 上创建账户，为 Adefioye/AI-Playground 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/128">[Ongoing] Eval Template and Score Fixes by Miserlou · Pull Request #128 · open-thought/reasoning-gym</a>: 正在进行的修复评估器和歧义问题的工作。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/129">Remove leading/trailing whitespace from scored answers by default by olliestanley · Pull Request #129 · open-thought/reasoning-gym</a>: 在默认评分中，对于某些数据集，由于 LLM 生成了前导/尾随空格，许多答案获得了 0.5 分，而原本它们应该获得 1.0 分。这些空格通常是...</li><li><a href="https://github.com/Adefioye/AI-Playground/blob/bd9e70706cda7106778f87b6c5bb5dbb5e851836/eval/eval.py#L48-L51">AI-Playground/eval/eval.py at bd9e70706cda7106778f87b6c5bb5dbb5e851836 · Adefioye/AI-Playground</a>: 通过在 GitHub 上创建账户，为 Adefioye/AI-Playground 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/124">Add useful instructions to question template of some datasets by Adefioye · Pull Request #124 · open-thought/reasoning-gym</a>: 在这里，为一些数据集的问题模板添加了一些指令。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/eval/r1/eval.py">reasoning-gym/eval/r1/eval.py at main · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账户，为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/eval/eval.py">reasoning-gym/eval/eval.py at main · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账户，为 open-thought/reasoning-gym 的开发做出贡献。
</li>
</ul>

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1339695111239634944)** (13 条消息🔥): 

> `API usage 字段更新，跨模型的 Tokenization，Provider 故障，OpenAI 模型可用性，模型后缀与功能` 


- **API Usage 字段更新考量**：由于跨模型 Tokenization 的进展，API 中 `usage` 字段的一项提议更改考虑从**归一化 Token 计数**切换为**模型原生 Token 计数**。
   - 用户对这是否会影响模型排名表示担忧，对此已确认**排名仍将使用 GPT tokenizer**。
- **Provider 间的 Tokenization 争议**：关于 **vertex** 模型是否仍以较高的**每字符 Token 比例**运行引发了讨论，一名用户建议在聚合平台内保持使用 **GPT tokenizer** 以确保一致性。
   - 澄清说明虽然 vertex 的比例略有不同，但并不像过去 **PaLM** 等模型那样极端。
- **Provider 故障通知**：简短通知指出 **Fireworks provider** 出现故障，但指出其他 provider 和 **BYOK usage** 不受影响。
   - 更新说明故障已于 **东部时间 9:12** 解决，确认已恢复正常运行。
- **OpenRouter 模型可用性更新**：OpenAI 的 **o1 和 o3 模型** 现已向所有 OpenRouter 用户开放，无需单独的 **BYOK key**，并允许更高的 **rate limits**。
   - 公告包含了一份**模型后缀速查表**，指明了如 `:online`、`:nitro` 和 `:floor` 等针对不同功能的选项。
- **Usage 报告的评估**：用户对 **roo code** 报告的**总使用成本**准确性提出担忧，认为其可能与预期不符。
   - 另一名用户询问哪些 provider 不报告 **usage object**，寻求对其操作实践的澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1890419427859935275">来自 OpenRouter (@OpenRouterAI) 的推文</a>：提醒您可以附加 `:online` 为任何模型添加 Web 访问权限，包括 o1 和 o3-mini 🌐 此外还有 `:nitro`（最快）、`:floor`（最便宜）和 `:free`（适用于 30 个模型）引用 OpenRouter (@Op...</li><li><a href="https://openrouter.ai/docs/api-reference/overview#response-format">OpenRouter API 参考 - 完整文档</a>：OpenRouter API 综合指南。了解请求/响应架构、身份验证、参数以及与多个 AI 模型 provider 的集成。
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1339645961659744426)** (163 条消息🔥🔥): 

> `DeepSeek R1 性能、API Key 错误问题、自托管 OpenAI 端点、新模型介绍、速率限制（Rate Limiting）关注点` 


- **DeepSeek R1 对部分用户表现不佳**：用户报告称在 OpenRouter 上使用 **DeepSeek R1** 时，服务经常暂停，导致其 Agent 出现问题，并对其在生产环境中的可靠性表示担忧。
   - 一些用户将其性能与其他模型进行了比较，表示在特定设置下（包括建议的 0.6 Temperature 且不含 System Prompt）其推理能力更胜一筹。
- **API Key 问题和删除线指示器**：一位用户发现其 API Key 在网站上显示为删除线并返回 401 错误，导致对其有效性和使用情况产生困惑。
   - 管理员指出，由于系统检测到潜在泄露，Key 可能会被禁用，并强调了使用 Secret 的重要性。
- **对自托管（Self-Moderated）端点的兴趣**：讨论了关于自托管 OpenAI 端点的话题，用户表达了对更低延迟和更一致响应输出的渴望，类似于 Anthropic 的处理方式。
   - 管理员表示，他们正根据社区反馈努力实现这些功能。
- **速率限制（Rate Limiting）和模型参数**：用户询问了 **Gemini 2.0 Pro** 等模型的速率限制，揭示了基于模型变体和用户额度的每日请求限制。
   - 还有关于不同供应商性能不一致的讨论，并与获得最佳结果的预期参数设置进行了对比。
- **对新模型和供应商的反馈**：参与者交流了对 **Sambanova** 等新模型的看法，探讨了其定价结构以及与成熟系统相比在响应质量方面的用户体验。
   - 用户注意到根据所使用平台的不同，结果也各不相同，从而引发了关于底层 Prompt 透明度和 Claude 等模型行为调整的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1890324295181824107">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🎉 很高兴看到大家对部署 DeepSeek-R1 的热情！以下是我们推荐的最佳体验设置：• 无 System Prompt • Temperature: 0.6 • 官方搜索和文件上传 Prompt...</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/quickstart">OpenRouter 快速入门指南</a>: 开始使用 OpenRouter 的统一 API 访问数百个 AI 模型。了解如何使用 OpenAI SDK、直接 API 调用或第三方框架进行集成。</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API 速率限制 - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 防护。有效配置和监控您的模型使用限制。</li><li><a href="https://cloud.sambanova.ai/">SambaNova Cloud</a>: 预览全球最快的 AI 推理 API。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1339667904467042325)** (122 messages🔥🔥): 

> `Perplexity Deep Research, AI Model Opinions, Use of Wolfram Alpha, ChatGPT User Experience, AI News Sources` 


- **Perplexity Deep Research 功能**：用户兴奋地讨论了 Perplexity 发布的新功能“Deep Research”，部分用户已经可以在免费层级中使用。
   - 围绕该功能的使用限制产生了好奇，包括它是按周计算还是基于其他标准。
- **对 AI 模型的不同看法**：成员们对不同 AI 模型的有效性表达了不同意见，有些人觉得 ChatGPT 与早期版本相比，随着时间的推移变得“变笨了”。
   - 有一种观点认为，新模型往往优先考虑语气和直接性，而非逻辑推理。
- **Wolfram Alpha 作为工具**：讨论了将 Wolfram Alpha 集成到 LLM 系统中以增强计算能力，几位成员主张使用它。
   - 有人指出，许多人发现通过 API 快速提供准确的数学答案非常有价值。
- **探索 AI 新闻来源**：几位用户强调 Perplexity 是首选的新闻来源，因为它偏见较低，且具有追问建议等交互功能。
   - 这种对替代方案的兴趣源于与传统来源相比，用户希望获得不失真的新闻消费体验。
- **AI 工具和功能的未来**：有人猜测潜在的整合，例如 OpenAI 收购 Perplexity 并对其进行品牌重塑，这表明用户对不断进化的 AI 工具有着浓厚兴趣。
   - 对话经常强调现有 AI 平台中阻碍用户体验的局限性，从而引发了关于改进的讨论。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1339662434666942545)** (10 messages🔥): 

> `Free Plan Limits, GPT Store Publishing, Privacy Policy Requirement` 


- **免费计划限制尚不确定**：一位成员询问如何验证各种模型免费计划的限制，询问是否能找到消息数量和发送文本的固定值。
   - 另一位成员指出，限制是可变的，每天根据不同因素变化，并暗示只有某些特定项目（如 **AVM** 的 **15 分钟/月**）具有固定值。
- **寻求 GPT Store 发布错误的帮助**：一位成员报告了尝试发布到 GPT Store 时的错误，称收到一条关于需要有效隐私政策 URL 的消息。
   - 另一位成员建议更新 Actions 中的隐私政策字段，原成员确认在编写隐私政策后解决了该问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1339662979100311704)** (10 messages🔥): 

> `Using ChatGPT vs Playground, Interpretation of prompts, JSON vs plain text formats, Legislative writing with AI, Importance of human oversight` 


- **ChatGPT 与 Playground 的区别**：讨论强调，由于模型处理提示词（prompts）和错误管理策略的方式不同，使用 ChatGPT 与 Playground 有所区别。
   - 建议将重点放在识别和纠正错误上，这对于与模型进行有效交互至关重要。
- **解读提示词以提高清晰度**：成员们强调，要求模型对比提示词的不同解读可以帮助识别冲突或歧义。
   - 使用清晰且通俗的语言而非严格的格式，可以从 AI 那里获得更有用的响应。
- **在格式间选择：JSON 还是文本**：一位成员主张在直接与 AI 交互时使用简单文本或 YAML，因为其可读性和效率更高，而 JSON 则推荐用于 API。
   - 无论选择何种格式来传达指令，保持清晰度都被认为是必不可少的。
- **AI 辅助立法写作**：虽然还没有人专门为立法写作创建提示词，但对于重要材料，细心的人类应该监督任何 AI 生成的输出。
   - 成员们强调，熟练的人类必须验证和批判所有模型输出，以确保安全性和准确性。
- **AI 使用中的人类监督**：讨论强化了这样一种观点，即在任何 AI 辅助的过程中，人类监督都是至关重要的，并将其比作驾驶辅助系统。
   - 即使在利用 AI 辅助创作材料时，对输出结果承担全部责任也是至关重要的。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1339662979100311704)** (10 messages🔥): 

> `Using ChatGPT vs Playground, Interpreting prompts, Prompt formats, Legislative writing prompts, AI model confidence` 


- **ChatGPT 与 Playground 的区别**：一位成员讨论了使用 **ChatGPT** 和 **Playground** 之间的细微差别，强调了提取错误和识别响应模式的重要性。
   - *亲自查看许多案例；与模型一起查看哪些模式出现了错误。*
- **解释提示词以提高清晰度**：一位成员强调了提示词解释的价值，指出它可以揭示冲突和歧义，从而产生有用的见解。
   - *这对你意味着什么，是否存在任何冲突或歧义？*
- **提示词格式偏好**：一位成员建议，对于直接的 AI 交互，**YAML** 或纯文本在可读性上更具优势，而 **JSON** 由于其严格的结构更适合 API。
   - 这引发了关于根据用户熟悉度和效率选择最佳系统提示词格式的讨论。
- **AI 立法写作的注意事项**：一位成员表示，在使用 AI 进行立法写作时需要谨慎的人工监督，强调必须由专业人员审查所有输出。
   - 他们警告说，虽然 AI 辅助可以完善想法，但保持责任感和核实信息至关重要。
- **建立对 AI 输出的信心**：另一位成员分享了关于确保 AI 模型通过使用常规语言规则来清晰理解用户提示词、避免混淆的见解。
   - 消除错误使模型能够清晰地预测用户意图，从而增强其可靠性。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1339644844867457206)** (135 messages🔥🔥): 

> `Stable Diffusion Models, Lora Training Tips, Audio Device Recognition, Controlled Image Generation, Community Engagement` 


- **Stable Diffusion 的 Lora 训练指南**：一位用户分享了仅用 7 张自拍训练 **Lora** 的经验，导致特征识别有限，尤其是侧脸视图。
   - 建议包括使用更大规模的高质量图像数据集，并确保它们与预期的输出风格匹配，因为较小的模型泛化效果可能较差。
- **AI 图像生成探索**：成员们讨论了生成 AI 艺术的各种方法，解决了如在多个模型中实现一致的角色设计等挑战。
   - 推荐使用 **FaceFusion** 等工具进行换脸，而关于自动图像请求的查询引发了对需要 **ComfyUI** 工作流的讨论。
- **带有控制设置的 Stable Diffusion**：一位用户询问了关于通过控制机制微调 **Stable Diffusion** 以改进图像生成的建议，并对近期工具表示关注。
   - 建议指向 L3 Discord，以获取与受控图像生成项目相关的特定资源和联系人。
- **音频设备检测的怪癖**：一位成员幽默地评论了 Windows 检测音频设备的怪癖，建议理想的硬件解决方案可以改进检测过程。
   - 这引发了关于技术挫败感的轻松闲聊，一些人提到了尽管计算设备存在缺陷但仍对其高度依赖的悖论。
- **社区动态与参与**：新用户介绍了自己，分享了他们在 AI 艺术方面的经验，并就 AI 工具和模型面临的挑战寻求建议。
   - 成员们欢迎新加入者，展示了专注于在 AI 艺术生成领域交流知识和经验的活跃社区氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/Dalabad/stable-diffusion-prompt-templates">GitHub - Dalabad/stable-diffusion-prompt-templates</a>：通过在 GitHub 上创建账号来为 Dalabad/stable-diffusion-prompt-templates 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=Si3dMKq_2xM">Donald Trump is a Hippie- part 2</a>：唐纳德·特朗普作为一个嬉皮士与埃隆·马斯克和乔·拜登一起享受生活。由 AI 制作。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1339650456527568999)** (53 messages🔥): 

> `Open Weight Definition, DeepHermes-3 Preview, EnigmaEval Launch, AI Security Institute, xAI Data Center Plans` 

- **Open Weight 定义引发讨论**：会议讨论了 Open Weight 的定义，强调了在 [Open Weight 网站](https://openweight.org/)上免费重新分发模型权重的标准合规性。人们对其影响表示担忧，一些成员对其在开源 AI 实践中的潜在影响表示关注。
- **DeepHermes-3 预览版引入高级推理能力**：DeepHermes-3 的发布已公布，展示了其切换推理能力以提高准确性的功能（代价是增加计算时间），详情见 [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview)。社区基准测试表明，其性能仍在与其他模型（如 Tülu）进行对比评估。
- **EnigmaEval 提高了 AI 推理的门槛**：Dan Hendrycks 分享了 EnigmaEval 发布的一系列复杂推理挑战，顶级 AI 系统的得分低于 10%，且在学生级谜题上没有一个系统的得分超过 0%，凸显了 [Scale AI](https://twitter.com/scale_AI) 确定的难题。参与者在挑战上花费了大量时间，揭示了当前 AI 系统能力的不足。
- **英国转向 AI Security Institute**：据 TechCrunch 报道，英国政府已将其 AI Safety Institute 更名为 AI Security Institute，将重点转向加强针对 AI 相关风险的网络安全。正如多位社区成员所指出的，这一变化引发了人们对 AI 安全关注度降低的担忧。
- **xAI 计划进行大规模数据中心扩张**：正如 [The Information](https://www.theinformation.com/briefings/musk-looks-for-another-data-center-for-xai-nears-5-billion-chip-deal-with-dell) 最近的一份报告所述，Elon Musk 的初创公司 xAI 正在寻求建立一个新的数据中心，以支持不断增加的 Nvidia 芯片使用。这一扩张标志着在竞争激烈的 AI 领域中雄心勃勃的增长努力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openweight.org/">Open Weight Definition (OWD)</a>: 未找到描述</li><li><a href="https://opensource.org/ai/open-source-ai-definition">The Open Source AI Definition &#8211; 1.0</a>: 1.0 版本前言。为什么我们需要开源人工智能 (AI)？开源已经证明，在消除学习、使用、分享的障碍后，每个人都能获得巨大的利益...</li><li><a href="https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/">英国政府将其 AI 机构名称中去掉“安全”一词，现更名为 AI Security Institute，并与 Anthropic 签署谅解备忘录 | TechCrunch</a>: 英国政府希望转向利用 AI 提振经济和产业，作为其中的一部分，它正在转型一个原有的机构。</li><li><a href="https://opensourcealliance.org/">Open Source Alliance</a>: 团结全球开源社区，塑造软件自由的未来。</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">Nous Research (@NousResearch) 的推文</a>: 推出 DeepHermes-3 预览版，这是一款统一了推理和直觉语言模型能力的新 LLM。https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview DeepHermes 3 是基于...</li><li><a href="https://x.com/stalkermustang/status/1890144205038842219">Igor Kotenkov (@stalkermustang) 的推文</a>: 伙计们醒醒，AIME 2 LLM 结果出炉了。o3-mini 是王者，Gemini 完蛋了，R1 表现尚可。</li><li><a href="https://x.com/janleike/status/1890155264101486792">Jan Leike (@janleike) 的推文</a>: @caleb_parikh 他们发送了 7,867 条消息，并将其中 1,408 条传给了自动评分器。我们估计他们总共在这上面花费了超过 40 小时。</li><li><a href="https://fxtwitter.com/anissagardizy8/status/1890483681476686177">Anissa Gardizy (@anissagardizy8) 的推文</a>: 最新消息：据知情人士透露，Elon Musk 的人工智能初创公司 xAI 正寻求建立一个新的数据中心，因为它计划大幅增加其使用的 Nvidia 芯片数量...</li><li><a href="https://x.com/Dorialexander/status/1890122850339811642">Alexander Doria (@Dorialexander) 的推文</a>: @TheXeophon 新组织。https://opensourcealliance.org/ 定义刚刚在峰会期间发布并得到了合理的宣传（背景：似乎是因为对 Open Source Initiative 的分歧而产生的分支...）</li><li><a href="https://fxtwitter.com/janleike/status/1890141865955278916">Jan Leike (@janleike) 的推文</a>: 我们的越狱挑战结果：经过 5 天、超过 300,000 条消息以及约 3,700 小时的集体努力，我们的系统被攻破了。最终有 4 位用户通过了所有关卡，1 位发现了通用越狱方法。我们...</li><li><a href="https://fxtwitter.com/DanHendrycks/status/1890091724594393140">Dan Hendrycks (@DanHendrycks) 的推文</a>: 我们正在发布 EnigmaEval，这是一系列长且复杂的推理挑战，需要多人花费数小时或数天才能解决。最顶尖的 AI 系统在普通谜题上的得分低于 10%，而对于...</li><li><a href="https://x.com/markgurman/status/1890239501974622650">Mark Gurman (@markgurman) 的推文</a>: 突发：Apple 将与阿里巴巴和百度合作，为其在中国的 AI 提供支持。阿里巴巴将修改并审查 Apple 端侧模型上的内容以符合中国法律。百度将支持 Visual Intelligence...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1339669334880030783)** (8 条消息🔥): 

> `notebookLM 性能, GPT-5 模型界面, 推理模型训练` 


- **用户对 notebookLM 的不满**: *notebookLM 表现糟糕*，虽然响应迅速，但无法完成从多个 PDF 创建 Markdown 表格等任务，让用户觉得它已经过时。
   - 一位用户表示希望切换到 Deep Research 并使用特定 Prompt 以获得更好的结果。
- **对 GPT-5 界面变化的担忧**: 一位用户对 Sama 宣布将 GPT-5 的多个模型合并为一个界面的消息表示担忧，称他们希望能够区分模型类型以便进行任务分配。
   - 另一位参与者认为自己可能不是这一变化的受众。
- **理解推理模型训练**: 一位成员总结了推理模型的训练过程，指出模型首先经过微调以产生 *thinking*（思考）Token，然后再应用强化学习（RL）来完成任务。
   - 另一位参与者补充说，RL 会尝试多种解题方法并强化成功的路径。

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1339712571506294844)** (2 条消息): 

> `DH3 评估指标、Distill Release 对比、公司真实性讨论` 


- **DH3 评估指标备受质疑**：人们对 DH3 的评估指标表示担忧，在“reasoning on”部分仅展示了*两个特定的评估指标*，而“reasoning off”图表则展示了所有指标。
   - 这种选择性报告引发了对其评估过程透明度的质疑。
- **DH3 对比官方 8b Distill Release**：批评者指出，DH3 未能直接与官方的 **8b distill release** 进行对比，后者的得分更高，例如在 GPQA 上为 **49%**，而 DH3 仅为 **36-37%**。
   - 这一遗漏导致人们对报告结果的有效性产生怀疑。
- **公司真实性存在不确定性**：围绕与 DH3 相关的公司合法性展开了讨论，成员们表达了复杂的看法。
   - 尽管存在怀疑，但有人指出，在社区中，“swaggy”可能比官方验证更具吸引力。



**提到的链接**: <a href="https://fxtwitter.com/kalomaze/status/1890153665333457140">来自 kalomaze (@kalomaze) 的推文</a>: dh3 notes1. they only show these two specific evals for the &#34;reasoning on&#34;; the &#34;reasoning off&#34; chart is the only one showing all metrics2. they don&#39;t compare to the official 8b di...

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1339711778359349299)** (41 条消息🔥): 

> `Boomer prompts 与 O-series 模型、DeepSeek-R1 部署、学术写作演变、David Perrell 的写作建议、DLCT 的 Tülu 3 演示` 


- **在使用 O-series 模型时避免 Boomer Prompts**：成员们讨论了在使用 O-series 模型时避免“boomer prompts”的重要性，正如 @OpenAIDevs 所强调的那样。清晰的分隔符和直接的指令可以增强模型输出的有效性。
   - 一位成员幽默地提到，由于这条建议的暗示，他感到被“冒犯”了。
- **DeepSeek-R1 部署热潮**：大家分享了对 **DeepSeek-R1** 部署的兴奋之情，官方建议的设置包括不设 system prompt 且 temperature 为 **0.6**。提供了关键链接以获得最佳体验并缓解绕过（bypass）问题。
   - 用户注意到了官方部署与该模型的开源变体之间的差异，以确保获得类似的体验。
- **学术写作与 LLM 的演变**：讨论围绕一篇 arXiv 摘要展开，该摘要分析了受 LLM 影响学术写作中某些词汇频率增加的现象，显示了作者的适应性。这突显了人类作者与 AI 技术之间正在进行的协同进化。
   - 成员们推测了非英语母语者使用 LLM 提升写作质量的影响。
- **David Perrell 的写作见解**：小组分享了对 David Perrell 写作建议的看法，有些人认为其富有启发性，而另一些人则认为过于简单化。成员们强调了个人细节在吸引读者方面的重要性以及强有力的引言的价值。
   - 建议关注 Perrell 的内容，同时承认其虽不完美但很有趣的特质。
- **Tülu 3 演示更新**：**Tülu 3** 的演示成为亮点，由 @pdasigi 在 DLCT 主持，并带有引用情人节的庆祝性提及。该活动展示了新进展和社区活力。
   - 成员们对演示表示兴奋，这与节日气氛相契合。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ml-valentines.github.io/">ml valentines</a>：未找到描述</li><li><a href="https://fxtwitter.com/deepseek_ai/status/1890324295181824107">来自 DeepSeek (@deepseek_ai) 的推文</a>：🎉 很高兴看到大家对部署 DeepSeek-R1 的热情！以下是我们推荐的最佳体验设置：• 无 system prompt • Temperature: 0.6 • 用于搜索和文件上传的官方 prompt...</li><li><a href="https://arxiv.org/abs/2502.09606">人类-LLM 协同进化：来自学术写作的证据</a>：通过对 arXiv 论文摘要的统计分析，我们报告了几个此前被确定为 ChatGPT 过度使用的词汇（如“delve”）的出现频率显著下降，这一现象开始于...</li><li><a href="https://x.com/savvyRL/status/1890465151574254057">来自 Rosanne Liu (@savvyRL) 的推文</a>：Tülu 3 正在 DLCT 进行演示！由 @pdasigi 领导。@allen_ai 的配色加上 🩷 作为作者上标，基本上就是在高喊“情人节快乐”</li><li><a href="https://x.com/edwinarbus/status/1890164717148336365">来自 edwin (@edwinarbus) 的推文</a>：@din0s_ 老派、冗长、说明性的 prompt 可能在 GPT 系列中效果很好，但对于推理模型来说是不必要的，因为这些模型可以自己搞清楚 https://x.com/edwinarbus/status/1890...</li><li><a href="https://x.com/edwinarbus/status/1890149926660678080">来自 edwin (@edwinarbus) 的推文</a>：基本上，“仔细地一步步思考……”“向我解释你的推理过程……”或者非常长、高度详细、说明性的 prompt 对 o-series 来说是不需要的，而且可能会减慢你的速度——相信...</li><li><a href="https://x.com/OpenAIDevs/status/1890147300493914437">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：正如你们中有些人注意到的，在 o-series 模型中避免使用“boomer prompts”。相反，要简单直接，并给出具体的指导方针。分隔符（xml 标签）将有助于保持模型内容的整洁，并且...</li><li><a href="https://x.com/lmarena_ai/status/1890172387070734633">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：我们很高兴地宣布 @Alibaba_Qwen Qwen2.5-VL-72B-Instruct 是 Vision 排行榜上新的排名第一的开源（OPEN）模型！他们已经超越了 Pixtral，开源与闭源模型之间的差距...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files">由 DeepSeekPH 提交的 Pull Request #399 · deepseek-ai/DeepSeek-R1</a>：添加文件上传和网页搜索 prompt。这些是我们在 DeepSeek 网页端和 App 中使用的 prompt。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1339978645770600559)** (4 条消息): 

> `Alignment 讨论，OpenAI O1 Pro 模式咨询` 


- **法式 Alignment**：一位成员分享了一个讨论 **alignment** 问题的链接，并带有一种幽默的解读，配文为“alignment à la française”。
   - 链接内容暗示了 AI 社区内标准 alignment 争论中的文化差异。
- **关于 O1 Pro 模式的昂贵见解**：一位成员幽默地声称 **Sam Altman** 向他们收取了 **$200**，以获取与 **OpenAI O1 Pro 模式** 相关的信息。
   - 在随后的一段引用中提到，O1 可以处理复杂的任务，如处理非结构化数据和识别建筑图纸中的细节。
- **对 O1 Pro 模式能力的要求**：一位成员建议 OpenAI 应该发布关于使用 O1 Pro 模式统计单词 **strawberry** 中“R”出现次数的查询数量统计。
   - 这反映了对用户奇特请求的轻松调侃，这些请求可能并未充分利用 O1 模型的全部能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/y0b1byte/status/1890417106006511621">yobibyte (@y0b1byte) 的推文</a>: alignment à la française</li><li><a href="https://x.com/allgarbled/status/1890314331805610143">gabe (@allgarbled) 的推文</a>: Sam Altman 为此向我收取了 $200。引用 OpenAI Developers (@OpenAIDevs)：使用 o 系列模型处理非结构化数据、大海捞针、改进代码或处理其他复杂任务。F...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1340033272796807239)** (7 条消息): 

> `推理模型想法，AI 中的有趣实验，GRPO 与 KL=0，训练指标与熵` 


- **在 <answer> token 中应用 KL**：一位成员分享了他们最喜欢的推理模型想法，即仅在 `<answer>` token 中应用 **KL**，并认为这可能会产生有趣的结果。
   - 他们对这个想法表现出极大的热情，表示尽管觉得这太酷了不适合发推，但“必须尝试一下”。
- **鼓励 AI 开发中的趣味性**：一位成员强调了在 AI 项目中让“有趣的事情发生”的重要性，表达了对创意和创新的渴望。
   - 这种观点与社区内关于参与新的实验性想法的持续讨论相一致。
- **对 GRPO 在 KL=0 下工作的兴趣**：一位成员引用了一条关于 **GRPO** 在 **KL=0** 下工作的推文，好奇群组内是否已经在讨论此事。
   - 这引发了关于训练策略以及 KL 在各种语境下的相关性的对话。
- **OpenAI 员工在 Twitter 上的询问**：一位成员提到收到了一位 **OpenAI** 员工关于其推文的私信，暗示幕后正在进行一些令人兴奋的工作。
   - 这表明行业专业人士对社区中发生的草根讨论很感兴趣。
- **训练考量的指标**：另一位成员强调，训练指标（如 **entropy**）应该区分 `<thinking>` 标签内部的内容和其他地方的内容。
   - 这建议在评估模型性能时采用一种基于上下文的细致方法——这是 AI 系统评估中的一个重要课题。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1339791411276415089)** (9 条消息🔥): 

> `Zvi 的写作风格，长篇内容，历史评论，AI 对评论的视角` 


- **对 Zvi 长篇文章的复杂心情**：成员们表达了对阅读 **Zvi 的长篇文章** 的共同犹豫，觉得篇幅往往掩盖了内容。
   - “我希望我有耐心读完长篇内容”是几位讨论潜在学习收益的用户产生的共鸣。
- **对 Zvi 作品单调性的看法**：一位成员指出 Zvi 似乎在 *反复啰嗦同一个观点*，导致人们对其评论的参与度产生担忧。
   - 尽管如此，另一位成员承认他的写作可能是随时间推移而产生的 **极佳历史记录**。
- **真挚但粗犷的写作**：评论认为 Zvi 的文章感觉 *发自内心* 且未经太多润色，在读者中引起了共鸣。
   - 他的方法似乎更多是为了真实性而非磨练文笔，尽管篇幅令人望而生畏，但仍能引起共鸣。
- **Zvi 评论中的 AI 接受度**：一位成员幽默地评论说 Zvi 太过字面地理解了 *为 AI 写作* 的建议，质疑 AI 是否会欣赏他的评论。
   - 这引发了一场关于他的内容是否适合 AI 受众的轻松讨论，为这些批评增添了一层幽默感。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1339655356384546836)** (11 条消息🔥): 

> `Notebook LM 用于学习, Z 世代社交媒体俚语定制, Notebook LM 回答质量, 奇幻小说写作案例, 播客功能疑问` 


- **Notebook LM 成为 24/7 个人导师**：一位用户分享了 **Notebook LM** 如何通过从大量阅读材料中生成详细摘要和关键点，彻底改变了他们的医学学习流程。
   - *它简直就是一个触手可及、全天候在线的个人导师，* 强调了该工具的易用性和实用性。
- **Z 世代俚语让学习变得有趣**：一位成员强调了通过定制 Prompt，使用 **Z 世代“脑干缺失”（brainrot）社交媒体俚语** 来解释复杂概念的有效性。
   - 这种方法帮助他们用更接地气的语言掌握了晦涩的学科，让学习变得更轻松。
- **Notebook LM 的回答质量有所提升**：另一位用户在通过录音和文本学习 Gemarah（犹太教法典）时，注意到 **Notebook LM** 的回答质量和结构化程度有了明显改善。
   - *感谢这些升级，* 他们对平台的进步表示了认可。
- **探索奇幻小说的素材来源**：一位用户提到，他们奇幻小说的背景涉及过去五年中开发的详细 **宇宙学、历史和地理** 研究。
   - 他们分享了为叙事深度做出贡献的各种参考来源。
- **关于播客功能的疑问**：一位成员询问了播客功能的用途，质疑它是否仅仅是一种音频概览格式，而非一种新的内容创作工具。
   - 他们在界面操作上遇到了困难，在选择 Pocketcast 后出现了 **白屏**。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1339643568435892335)** (75 条消息🔥🔥): 

> `Notebook LM 语言支持, Notebook LM PDF 上传问题, Notebook LM 订阅变更, Notebook LM 文档共享, Gemini 模型功能` 


- **Notebook LM 语言支持改进**：用户反映，尽管上传了相应语言的源文件，但很难让 Notebook LM 用选定的语言（如保加利亚语和德语）进行回答。
   - 一些成员发现通过使用特定的 URL（例如 [notebooklm.google?hl=bg](https://notebooklm.google?hl=bg)）可以成功切换为保加利亚语。
- **PDF 上传问题**：一位用户表示无论文件大小或复杂程度如何，上传 PDF 始终遇到困难，而其他用户则表示运行正常。
   - 这个问题似乎与用户的浏览器有关，也可能与系统处理敏感内容时的安全过滤器有关。
- **Notebook LM 订阅过渡**：一位德国学生指出，在切换到 Notebook LM 付费版后，遇到了文本转语音（TTS）恢复为英语的问题。
   - 用户通过分享维护语言设置的技巧互相帮助，并确认随后成功进行了调整。
- **文档共享功能**：围绕 Notebook LM 中的项目共享展开了讨论，涉及允许的协作者人数是否存在限制。
   - 成员们表示根据他们的经验尚未遇到限制，这促进了用户间的协作。
- **Gemini 多媒体功能**：多位用户询问了新 Gemini 模型的功能，特别是其在 Notebook LM 中的集成情况。
   - 回复显示，目前该模型在平台内的具体能力仍存在不确定性，鼓励用户探索相关资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google/plus/terms">Google NotebookLM | AI 驱动的笔记与研究助手</a>：利用 AI 的力量进行快速摘要和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可以信赖的信息。</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1339656691318521986)** (58 条消息🔥🔥): 

> `LLM 中的潜伏推理 (Latent Reasoning), Veo 2 视频生成模型, 苹果新产品, DeepHermes 3 LLM, 养蜂可行性报告`

- **Latent Reasoning 革新 LLMs**：一篇新论文讨论了 LLMs 如何在生成 tokens 之前利用 **latent reasoning**，这与传统的 chain of thought 方法形成了对比。
   - 这种方法有望带来显著收益，社区讨论也围绕其在实际应用中的影响展开。
- **Veo 2 在视频创作中占据核心地位**：Nvidia 的新模型 **Veo 2** 现已在 YouTube Shorts 上线，创作者可以通过其 **Dream Screen** 功能根据简短的文本提示生成视频片段。
   - 这一创新通过将用户生成的内容无缝集成到视频中，增强了叙事能力。
- **新款 Apple 设备备受期待**：**Tim Cook** 预告了即将举行的 Apple 发布会，可能包括 **iPhone SE**、**M4 Air** 以及更新的 Apple TV 选项等新产品。
   - 围绕 **带屏幕的 HomePod** 的猜测，以及 Apple 持续集成强力芯片以支持 AI 功能，在社区中引发了热议。
- **DeepHermes 3 旨在增强 LLM 能力**：来自 Nous Research 的最新 **DeepHermes 3** 模型旨在将推理和传统的 LLM 响应模式整合到一个单一的功能架构中。
   - 该模型力求在 LLM 的标注、判断和 function calling 能力方面取得显著进步。
- **养蜂业务研究见解分享**：一位成员分享了他们进行的一份全面的 **养蜂可行性报告**，提供了可操作的步骤和针对潜在业务策略的见解。
   - 围绕深度研究的 prompt 研究和优化进行的协作讨论，进一步丰富了社区对在实时项目中利用 AI 的理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05167">NoLiMa: Long-Context Evaluation Beyond Literal Matching</a>: 最近的 Large Language Models (LLMs) 支持从 128K 到 1M tokens 的 Long-Context。评估这些能力的一种流行方法是 needle-in-a-haystack (NIAH) 测试，它涉及检索...</li><li><a href="https://engineering.fb.com/2025/02/05/security/revolutionizing-software-testing-llm-powered-bug-catchers-meta-ach/">Revolutionizing software testing: Introducing LLM-powered bug catchers</a>: Meta 的 Automated Compliance Hardening (ACH) 工具是一个用于 mutation-guided、基于 LLM 的测试生成系统。ACH 通过生成未检测到的故障（mu…）来增强平台抵御回归的能力。</li><li><a href="https://x.com/anneouyang/status/1889770174124867940">Tweet from Anne Ouyang (@anneouyang)</a>: 来自 Nvidia 的新博客文章：LLM 生成的 GPU kernels 显示出比 FlexAttention 更快的速度，并在 🌽KernelBench Level 1 上实现了 100% 的数值正确性。</li><li><a href="https://x.com/MatthewBerman/status/1890081482104008920?t=V3aeg7FX8ZvIKtvhHtl-xA&s=19">Tweet from MatthewBerman (@MatthewBerman)</a>: 新的研究论文展示了 LLMs 如何在输出单个 token 之前进行内部“思考”！与 Chain of Thought 不同，这种 “latent reasoning” 发生在模型的 hidden space 中。大量的...</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/GoogleDeepMind/status/1890054036168356283">Tweet from Google DeepMind (@GoogleDeepMind)</a>: 🎥 我们最先进的视频生成模型 Veo 2 现已在 @YouTube Shorts 中可用。通过 Dream Screen 功能，创作者可以：✨ 制作无缝融入其叙事的新剪辑...</li><li><a href="https://www.youtube.com/watch?v=LP5OCa20Zpg&ab_channel=Anthropic">Tips for building AI agents</a>: Anthropic 的 Barry Zhang (Applied AI)、Erik Schultz (Research) 和 Alex Albert (Claude Relations) 讨论了 AI agents 的潜力，以及需要避免的常见陷阱...</li><li><a href="https://x.com/emollick/status/1887579095610860014">Tweet from Ethan Mollick (@emollick)</a>: 关于 OpenAI Deep Research 的有趣数据点：我一直收到来自各个领域资深人士的持续消息，他们主动分享了他们的聊天记录和...</li><li><a href="https://x.com/tim_cook/status/1890068457825394918">Tweet from Tim Cook (@tim_cook)</a>: 准备好迎接家庭的新成员。2 月 19 日，星期三。#AppleLaunch</li><li><a href="https://docs.google.com/document/d/1BIBInKu1rET9-BC520OXXz95XuFOxuMAr63nytI9DZQ/edit?usp=sharing">Feasibility and Strategy for Launching a Commercial Beekeeping Operation in Myakka City, FL</a>: 太棒了！我将为在佛罗里达州迈阿卡市启动商业养蜂业务进行深入的可行性研究和战略分析。报告将包括财务预测、蜂箱容量...</li><li><a href="https://buttondown.com/ainews/archive/ainews-reasoning-models-are-near-superhuman/">[AINews] Reasoning Models are Near-Superhuman Coders (OpenAI IOI, Nvidia Kernels)</a>: RL is all you need。2025年2月12日至2月13日的 AI 新闻。我们为您检查了 7 个 subreddits、433 个 Twitter 账号和 29 个 Discord 服务器（211 个频道和 5290 条消息）....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 新播客发布！ https://x.com/latentspacepod/status/1890101440615453025
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1339669074845896714)** (2 messages): 

> `LlamaIndex Google Cloud Integration, LlamaParse Features` 


- **LlamaIndex 与 Google Cloud 数据库集成**：使用我们的最新功能轻松将 LlamaIndex 与您的 [Google Cloud 数据库](https://twitter.com/llama_index/status/1890109073615626388) 集成，允许您将数据库用作初始 data store、vector store 等。
   - 这些集成旨在实现**简单**且**安全**，增强您的数据库交互。
- **探索 LlamaParse 的强大功能**：关于 [LlamaParse](https://twitter.com/llama_index/status/1890499579214491967) 的综合视频，涵盖了多种解析模式、输出格式，以及如何通过解析指令有效提高质量。
   - 它包括关于解析 **audio**、**images** 以及利用 **JSON mode** 获得优化结果的见解。

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1339642838320943156)** (49 messages🔥): 

> `AgentWorkflow 用于 RAG, 使用 uv 管理虚拟环境, LlamaIndex 更新, 过时包管理, 工作流中的 Python 函数` 


- **AgentWorkflow 不适合 RAG 应用**：`AgentWorkflow` 是为执行任务的 Agent 系统设计的，而非专门针对 RAG。建议参考[此处](https://docs.llamaindex.ai/en/stable/examples/workflow/rag/)描述的工作流方法来实现 RAG 功能。
   - 若要在 `AgentWorkflow` 中集成 RAG，建议用户创建自定义函数，以便在 RAG 处理中加入用户查询。
- **使用 `uv` 管理环境**：用户讨论了使用 `uv` 创建多个虚拟环境的便利性和优势，并分享了在独立环境中管理如 PyTorch 等工具不同版本的见解。
   - 一位用户提供了一个 shell 函数来简化环境切换及关联项目文件的过程，建议通过此工作流来提高便利性。
- **关于切换依赖管理工具的担忧**：有用户担心从 mini-conda 迁移到 `uv` 时会丢失功能，特别是针对训练和推理等不同任务处理多个环境的问题。
   - 提出了替代方案，包括为不同环境维护独立的 `pyproject.toml` 文件，并在激活时动态链接它们。
- **使用别名管理过时的包**：一位用户分享了一个 bash 别名，用于简化检查和更新过时 `llama-index` 包的过程，节省了手动跟踪的时间。
   - 该别名允许他们每周运行一次命令，以确保所有 `llama-index` 相关包都是最新的。
- **学习 RAG 的资源**：引导用户参考大量文档和示例，以更好地理解如何使用 LlamaIndex 实现 RAG，以及其与 Agent 工作流的关系。
   - 提供了入门教程和深入指南的链接，强调了在有效数据管理中使用 RAG 的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.pantsbuild.org/dev/reference/targets/uv_requirements">uv_requirements | Pantsbuild</a>: 为 `pyproject.toml` 中 `[tool.uv]` 章节下的每个条目生成 `python_requirement`。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/rag/">带有 Reranking 的 RAG 工作流 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">入门教程 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/">AgentWorkflow 基础介绍 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/rag/">RAG 介绍 - LlamaIndex</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1339642960631037972)** (3 messages): 

> `模型 Finetuning, AI 社区增长, 量子教育倡议` 


- **针对复杂任务进行模型 Finetuning**：一位成员强调，当任务或领域过于复杂，特别是输入数据与训练数据有显著差异时，需要对模型进行 Finetuning。
   - 他们补充说，广泛的 prompt engineering 可能无法产生令人满意的结果，因此必须通过 Finetuning 来获得更好的性能。
- **AI 创新者在印度集结**：发出加入印度增长最快的 AI 社区的邀请，旨在人工智能领域进行连接、协作和创新。
   - 成员可以通过提供的 [WhatsApp 链接](https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU)加入，成为这一蓬勃发展场景的一部分。
- **开启量子教育**：另一条消息推广了印度的量子教育社区，致力于推进量子计算方面的知识。
   - 鼓励参与者通过 [WhatsApp 链接](https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8)加入，助力自己的学习之旅。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU">Ai - ML - QB</a>: WhatsApp 群组邀请</li><li><a href="https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8">Quantum-QB</a>: WhatsApp 群组邀请
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1339651249993154602)** (44 条消息🔥): 

> `OpenRouter vs Glama, Issues with OpenWebUI, Using 0.0.0.0 in Networking, Instructions for Setup, Community Discussions on MCP Server Roles` 


- **OpenRouter 相比 Glama 处于劣势**：据报道，与 [OpenRouter](https://glama.ai/models/) 相比，Glama 更**便宜**、更**快**且保证**隐私**，尽管它支持的模型较少。
   - 确定的优势还包括各种模型的**额外定价**详情，范围从 **$0.06 到 $10** 不等。
- **OpenWebUI 饱受破坏性变更之苦**：用户担心 OpenWebUI 的每次微小更新都会带来**破坏性变更（breaking changes）**，导致 **80% 以上的社区功能**无法正常运行。
   - 一些人认为这很有挑战性，因为它属于**实验性的 alpha 软件**，经常充满竞态条件（race conditions），增加了使用难度。
- **关于在网络设置中使用 0.0.0.0 的争论**：IP 地址 **0.0.0.0** 的功能引发了讨论；它通常用于监听所有接口，特别是在**容器化环境（containerized environments）**中。
   - 然而，一些人反对将其作为 HTTP 上下文中的目标地址，并强调了理解正确用法对于**故障排除（troubleshooting）**的重要性。
- **设置 OpenWebUI 需要特定步骤**：指南讨论了在继续设置 OpenWebUI 之前，确保端点 **/v1/chat/completion** 正常运行的必要性。
   - 讨论得出结论，必须设置 **OPENAI_API_KEY** 才能利用 **OpenAI API**，随后进行 OpenWebUI 的特定配置。
- **MCP Server 作者角色分配**：鼓励成员分享其服务器链接以获得 **MCP server author 角色**，一些人分享了各自的 GitHub 仓库。
   - 角色分配已确认，表明提供演示服务器项目或库即可获得作者身份。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unix.stackexchange.com/questions/94018/what-is-the-meaning-of-0-0-0-0-as-a-gateway">作为网关的 0.0.0.0 是什么意思？</a>：谁能帮我澄清一下网关分配？将网关添加为 0.0.0.0 与将特定 IP 地址分配为网关有什么区别？</li><li><a href="https://en.m.wikipedia.org/wiki/0.0.0.0">0.0.0.0 - 维基百科</a>：未找到描述</li><li><a href="https://glama.ai/models/">领先的 LLM 模型</a>：企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/mashriram/azure_mcp_server">GitHub - mashriram/azure_mcp_server</a>：通过创建账户为 mashriram/azure_mcp_server 的开发做出贡献。</li><li><a href="https://github.com/PederHP/mcpdotnet">GitHub - PederHP/mcpdotnet: Model Context Protocol (MCP) 的 .NET 实现</a>：Model Context Protocol (MCP) 的 .NET 实现 - PederHP/mcpdotnet
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1339991564243701871)** (4 条消息): 

> `Zonos TTS MCP, Intonation Control for Claude, Use of SSML Tags, Markdown vs SSML, Text-to-Speech Models` 


- **Zonos TTS MCP 让 Claude 拥有声音**：[Zonos TTS MCP](https://github.com/PhialsBasement/Zonos-TTS-MCP) 服务器使 **Claude** 能够拥有类似于 **CGPT** 的声音，增强了用户交互。
   - *这一进展为基于对话的 AI 应用开辟了新途径。*
- **语调控制需要 Markdown 解析器**：一位成员提到需要 **Markdown 解析器（markdown interpreter）**来让 Claude 控制其**语调**，使其更接近理想表现。
   - *他们表示乐观，称一旦实现此功能，他们就“完美”了。*
- **SSML 标签可以增强语音模型**：建议将 **SSML 标签**作为利用 Claude 能力的一种方法，从而实现对语音特性的更细微控制。
   - *一位成员对此表示支持，称这些模型“超级聪明”，能够有效利用此类功能。*
- **更倾向于 Markdown 而非 SSML**：讨论强调了对 **Markdown** 的偏好，并指出其在 **ElevenLabs** 等 TTS 模型中的有效应用，这些模型提供了更清晰的指令能力。
   - *成员们认为 Markdown 可以提供良好的转录，同时确保语音合成的精确音调引导。*



**提到的链接**：<a href="https://github.com/PhialsBasement/Zonos-TTS-MCP/tree/main">GitHub - PhialsBasement/Zonos-TTS-MCP: 让 Claude 拥有声音的 MCP 服务器。</a>：让 Claude 拥有声音的 MCP 服务器。通过在 GitHub 上创建账户为 PhialsBasement/Zonos-TTS-MCP 的开发做出贡献。

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1339656500070846485)** (38 messages🔥): 

> `Evaluating RAG Systems, Tinystories Pretraining, Generative Models and RL, Pretraining on Consumer Hardware, Logits in Model Pipelines` 


- **评估 RAG 系统质量**：一位具有计算机视觉背景的成员正在寻求有关评估其 RAG 系统质量的相关指标建议，该系统目前具有稳定的 Retrieval 设置。
   - 他们向社区咨询有关评估 LLM 或 Retrieval 架构时所使用的指标指导。
- **Tinystories：不仅仅是预训练模型**：讨论指出 **Tinystories** 不仅仅是一系列预训练模型，它还包含了一系列架构、一个数据集以及一篇详细介绍设置过程的研究论文。
   - 成员们强调 Tinystories 完成了从小型模型中获得连贯输出所需的艰苦工作，对于初学者非常有用。
- **Logits 作为中间表示**：一位成员解释说，Logits 在其模型中将被视为中间表示而非最终输出，并在 Pipeline 中集成了倾向于 Logits 的更改。
   - 他们提议将 Softmax 移至 Pipeline 的末端，同时实施一种涉及 SFT、IRL/RL 和 EBM 的 **Multi-objective Training** 策略。
- **Accelerate + DeepSpeed 的挑战**：一位用户质疑为什么 **Accelerate + DeepSpeed** 比 **Unsloth** 消耗更多的 RAM，怀疑自己是否使用工具不当。
   - 这反映了关于在消费级硬件上优化性能以及 RAM 使用权衡的持续讨论。
- **使用 Energy-Based Methods 训练 Generative Models**：讨论展开了关于延迟 Normalization 以提高 Generative Sequence Models 中 RL 性能的想法，认为不规则性可能是有益的。
   - 关键策略包括使用 Dynamic Logits 并结合 SFT 来引导模型在训练中取得有意义的结果。



**提到的链接**：<a href="https://openreview.net/forum?id=O-XJwyoIF-k">Minimum Width for Universal Approximation</a>：宽度受限网络的 Universal Approximation 特性作为深度受限网络经典 Universal Approximation 结果的对偶问题已被研究。然而，临界宽度...

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1339767478628122634)** (2 messages): 

> `Weekly Crunch Time, Future Meeting Plans` 


- **成员面临每周忙碌期**：成员们表示本周特别 **忙碌**，表现出一种紧迫感和时间限制。
   - 一位成员提到，“我还有很多事情要做”，反映了参与者中普遍存在的超负荷情绪。
- **明天会议计划尚不确定**：明天的会议是否举行尚不确定，一位成员指出他们“希望今天能开一个会，但不得不等待”。
   - 不过，他们预计下周将恢复正常日程，对未来的讨论持乐观态度。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1339741152571818034)** (2 messages): 

> `无需 Token 的新 AI 模型，潜空间推理模型，35 亿参数模型` 


- **AI 模型在没有 Token 的情况下进行推理**：这段 [YouTube 视频](https://www.youtube.com/watch?v=ZLtXXFcHNOU) 讨论了模型是否可以在不使用单个 Token 的情况下进行“思考”，提出了一个关于 AI 能力的有趣问题。
   - *加入我的 Newsletter 以获取定期 AI 更新*，并了解这种新方法在 AI 领域代表了什么。
- **潜空间模型挑战 Token 使用**：一篇 [arXiv 论文](https://arxiv.org/abs/2502.05171) 提出了一种新型语言模型架构，通过在潜空间（Latent space）中进行推理来扩展测试时计算（test-time computation），且无需专门的训练数据。
   - 该模型成功地在推理基准测试中提高了性能，其计算负载可与 **500 亿参数**的模型相媲美。
- **35 亿参数增强推理能力**：论文描述了一个扩展至 **35 亿参数**并在 **8000 亿 Token** 上训练的概念验证模型，实现了显著的性能提升。
   - 它的独特之处在于通过迭代循环块（recurrent block），允许在测试时进行深度展开（depth unrolling），这与依赖增加 Token 生成量的传统方法形成鲜明对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>：我们研究了一种新型语言模型架构，能够通过在潜空间中进行隐式推理来扩展测试时计算。我们的模型通过迭代循环块工作，从而展开……</li><li><a href="https://www.youtube.com/watch?v=ZLtXXFcHNOU">New AI Model &quot;Thinks&quot; Without Using a Single Token</a>：模型可以在不使用 Token 的情况下思考吗？！真的吗？？加入我的 Newsletter 以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅：https://www....
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1339695065898942517)** (2 messages): 

> `Elon Musk, 开源 PTLM, 模型注册表挑战` 


- **双打对决暗示竞争关系**：一位成员幽默地建议进行一场 **Sam/Gates vs Elon** 的双打对决，并指出目前很少有人愿意站在 Elon 这一边。
   - *这反映了当前公众对 Elon 的合作伙伴关系和形象的看法。*
- **研究揭示 PTLM 发布的不一致性**：一项基于 Hugging Face 上 **52,227 个 PTLM** 的实证研究显示，**40.87%** 的模型权重更改未在命名实践或文档中体现。
   - 结果强调了预训练语言模型（PTLM）命名规范的**模糊性**以及训练文档的可访问性问题。



**提到的链接**：<a href="https://arxiv.org/abs/2409.10472">Towards Semantic Versioning of Open Pre-trained Language Model Releases on Hugging Face</a>：在 Hugging Face (HF) 等模型注册平台上，开源预训练语言模型 (PTLM) 的激增为围绕它们构建产品的公司带来了机遇和挑战……

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1339765989138632766)** (15 messages🔥): 

> `PR 提交指南，Kernel 与 OptOps 理解，WSL 上的 VIZ 问题` 


- **执行严格的 PR 提交规则**：*提醒每个人在提交 PR 之前检查三次 diff*，以避免空格变动，否则 PR 可能会在没有任何评论的情况下被关闭。
   - 此外，不鼓励提交 AI 生成的代码，因为这会浪费时间，*亲手编写代码*并寻求 AI 的反馈至关重要。
- **Kernel 与 OptOps 速度悬赏见解**：一位成员分享了他们对与 `sum` 悬赏相关的 Kernel 和 OptOps 的理解，建议创建一个 OptOp 来优化多个 reduction 的 AST。
   - 他们对当前 OptOps 在实现目标代码方面的表达能力表示担忧，并热衷于探索用于多个累加器的 `GROUP` OptOp，并指出渲染器（renderer）应该基本能按预期工作。
- **寻求 WSL 上 VIZ 的帮助**：一位用户询问是否有人尝试在 WSL Ubuntu 上使用 `VIZ=1`，因为他们在访问临时目录时遇到了错误。
   - 另一位成员承认 WSL 构建可能具有挑战性，尤其是涉及 Python 时，并提出下载所需的设置以进一步调查该问题。

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1339652530136940615)** (11 messages🔥): 

> `DSPy vs LangChain, DSPy 2.6 Change Log, Removal of DSPy Assertions, Multi-label Classification with DSPy, DSPy Code Golf` 


- **区分 DSPy 和 LangChain**：一位成员澄清说，如果用户更喜欢编写 Signatures 和 modules 而不是字符串提示词（string prompts），或者需要优化，他们应该选择 **DSPy** 而不是 **LangChain**。
   - 他们还建议，如果觉得 DSPy 对自己的需求来说太复杂，可以考虑 **LangChain** 是否提供了预封装的方案。
- **关于 DSPy 2.6 变化的咨询**：一位用户回归使用 DSPy，询问 **DSPy 2.6** 的更新日志，并提到了 *Signatures* 的“instructions”等新功能。
   - 另一位成员指出，这些 instructions 自 **2022** 年以来就一直存在，并引导该用户前往 [GitHub release page](https://github.com/dspy/dspy/releases) 查看详细变更。
- **对 DSPy 2.6.3 中移除常量的困惑**：成员们讨论了在 **2.6.3** 版本中移除 **dspy.Assert**、**dspy.Suggest** 和 **dspy.Retry** 的情况，这引发了关于向后兼容性替代方案的困惑。
   - 一位成员暗示，移除这些内容是最终引入 *assertions v2* 计划的一部分，尽管目前尚未提供路线图或解释。
- **在 DSPy 中优化多标签分类**：一位用户寻求关于使用 DSPy 为具有 200 个类别描述的多标签分类优化 **SLM** 的建议，并提出了一种批处理策略。
   - 他们希望在不微调模型或使用多个 **LoRA adapters** 的情况下实现这一目标。
- **DSPy Code Golf 的乐趣**：一位成员发起了一项有趣的 **DSPy** 代码高尔夫（code golf）活动，希望挑战其他人编写简洁的代码片段。
   - 他们分享了一个用一行代码从 HTML 中提取结构化数据的具体示例，表明社区可以将其变成一种竞技游戏。



**提到的链接**：<a href="https://x.com/lateinteraction/status/1890442615700545878">来自 Omar Khattab (@lateinteraction) 的推文</a>：有时我会找借口花上 5 分钟玩一些巧妙的 DSPy golf。有人问：如何使用 DSPy 从 HTML 中提取结构化数据？嗯，那其实就是一行代码的事。如果……

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1339986595125788744)** (1 messages): 

> `Valentine's Day, MAX and Mojo` 


- **MAX 和 Mojo 庆祝情人节**：MAX 和 Mojo 在这个情人节通过频道分享了愉快的问候和一张有趣的图片来传递爱意。
   - 附带的名为“MAXMojoValentine”的图片增添了节日气氛，使庆祝活动更具互动性。
- **分享节日图片**：分享了一张名为“MAXMojoValentine.jpeg”的精美图片以纪念这一时刻，展示了情人节的精神。
   - 这一互动元素为频道带来了愉悦感和社区归属感。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1339682548066029630)** (7 messages): 

> `Memory Error in Function Call, Release of v25.1, Dog/Cat Example Confusion, Larecs GitHub Repository, Safe Mutable Aliasing Document` 


- **调试函数调用中的内存错误**：一位用户询问了与其代码中 `add_fun` 调用相关的错误，特别是讨论了可变引用（mutable references）的别名（aliasing）问题。
   - 代码似乎与通过别名参数访问的内存位置产生了冲突。
- **对 v25.1 版本发布的兴奋**：一位匿名用户宣布了 **v25.1** 的发布，赢得了社区的热烈响应。
   - 感叹号和火焰表情符号表示对该版本带来的更新有着极高的兴趣。
- **对狗/猫示例的误解**：一位用户对之前分享的一个**狗/猫示例**表示困惑，这导致了他们早些时候的误解。
   - 另一位用户承认了这种困惑，并澄清那不是他们的示例。
- **探索 Larecs 仓库**：一位成员为其他有兴趣了解更多细节的人提供了 [Larecs GitHub repository](https://github.com/samufi/larecs) 的链接。
   - 树形表情符号暗示了对项目增长或开发的关注。
- **关于安全可变别名的文档**：一位用户索要另一位成员撰写的关于**安全可变别名（safe mutable aliasing）**的文档链接。
   - 作为回应，作者分享了他们在 11 月发布的 [提案/愿景文档](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) 链接。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1339664080038006825)** (8 条消息🔥): 

> `配置文件中的 Token 禁用，Deepseek 模型推荐，微调 LLM，TradingView 访问` 


- **Token 禁用咨询**：一名成员询问是否可以通过配置文件禁用 Token，并承认这在 GUI 中不是一项功能。
   - 该问题反映了即使在没有官方支持的情况下，用户对自定义 Token 行为的兴趣。
- **适用于 RTX 3080 的最佳 Deepseek 模型**：讨论强调，将 Deepseek 的行为蒸馏到较小的模型上可能会导致性能下降，特别是在 RTX 3080 上。
   - **Qwen2.5 Coder 14B** 被建议作为低 VRAM 配置的可行选择，成员们指出了性能权衡。
- **微调 LLM 的挑战**：一名成员询问如何使用 2021 年的数据来更新和微调 LLM。
   - 另一名成员澄清说这是不可能的，表明了用新数据适配旧模型的局限性。
- **TradingView 高级版访问**：一篇帖子分享了适用于 Windows 和 macOS 的免费破解版 TradingView 链接，并提到其拥有庞大的用户群。
   - 说明中包含了详细的安装步骤，强调可以免费使用 Premium 功能。



**提到的链接**：<a href="https://www.reddit.com/r/TradingViewFree/comments/1hobjs6/tradingview_premium_ultimate_package_update/">Reddit - 深入探索</a>：未找到描述

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1340002147198177393)** (3 条消息): 

> `Dataloader Transform 的 RFC，在线 DPO/GRPO 数据生成，Prompt 到 Preference 函数` 


- **Dataloader Transform 的 RFC 提案**：一名成员计划提出一项 RFC，以增加 Dataloader 转换和保存功能，从而增强训练时的在线 **DPO/GRPO 数据生成**。
   - *这可以通过应用不同的奖励模型或评判器（judges），简化将各种数据集转换为偏好数据集的过程。*
- **示例用法请求**：一名成员请求提供所提议的 Dataloader 转换的现有示例，以便更好地理解其在上下文中的应用。
   - *该查询突显了对实际示例的需求，以支持 RFC 的讨论和实施。*
- **带转换的批量生成演示**：分享了一个示例，展示了 **prompt_to_preference** 函数如何利用 `DataLoader` 生成批量偏好数据。
   - *该设置允许每个 Prompt 生成两个结果，并结合评判器来选择 chosen 和 rejected 项，表明了批量生成的可能性。*


  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1339781201447092245)** (2 messages): 

> `Distillation Scaling Laws, Quantization-Aware Training, QuEST Method, Sparse Representations in LLMs` 


- **Apple 关于 Distillation Scaling Laws 的见解**：最近的一项讨论重点介绍了 [Apple 的论文](https://arxiv.org/abs/2502.08606)，该论文专注于 **Distillation Scaling Laws**，探讨了是从更强大的模型进行蒸馏更好，还是从头开始训练更好。
   - 讨论中的一段话强调了关于模型大小和能力选择的复杂性：*“这很复杂……”*。
- **Quantization-Aware Training 的进展**：一项新研究进一步加深了对 **Quantization-Aware Training (QAT)** 的理解，探索了在保持准确性的同时使用量化表示的方法，特别是针对权重和激活值的 **8-bits** 最佳位宽。
   - 该方法的潜力已通过引用最先进的研究 [arXiv:2411.04330v2](https://arxiv.org/abs/2411.04330v2) 得到验证。
- **QuEST 方法在压缩方面展现出前景**：一位成员介绍了一种名为 **QuEST** 的新方法，该方法声称通过在权重和激活值使用 **4-bits** 或更低位宽的情况下保持强大的准确性，从而超越了之前的技术。
   - 该方法被定位为 **与 FP16 相比具有 Pareto 竞争力**，在减小模型尺寸的同时提供更好的准确性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05003">QuEST: Stable Training of LLMs with 1-Bit Weights and Activations</a>: 降低大型语言模型 (LLMs) 巨大成本的一种方法是在训练或部署中使用量化或稀疏表示。虽然训练后压缩方法已经非常成熟...</li><li><a href="https://x.com/danbusbridge/status/1890013666575282669?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Dan Busbridge (@danbusbridge) 的推文</a>: 阅读 “Distilling Knowledge in a Neural Network” 让我着迷并思考：“如果我想要一个小型且能力强的模型，我应该从更强大的模型中蒸馏，还是从头开始训练？”...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1339906564072669219)** (4 messages): 

> `Cohere Command R+` 


- **正在进行的 Cohere Command R+ 令人兴奋的项目**：一位成员宣布他们正在 *使用 Cohere Command R+ 构建一些非常酷的东西*，并鼓励其他人关注更新。
   - 另一位成员通过回复笑脸表情分享了这种兴奋感，表达了对该项目的热情。
- **对公告的轻松反应**：另一位成员针对项目公告回复了笑脸表情，反映了讨论中轻松的氛围。
   - 这有助于增强社区对正在构建的内容的共同热情和参与感。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1339823696633790516)** (2 messages): 

> `Quiz 3 release` 


- **Quiz 3 发布困惑已解决**：一位成员询问 **Quiz 3** 是否可用，称在 MOOC 网站上找不到。
   - 他们随后注意到了一项更新，显示相关信息可在 [Discord](https://discord.com/channels/1280234300012494859/1293323662300155934) 上找到。
- **Quiz 可用性的快速解决**：最初关于 **Quiz 3** 的询问凸显了对其发布日期的困惑，因为它在网站上不可见。
   - 幸运的是，该成员在 Discord 的另一个线程中找到了相关的更新，澄清了情况。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1339720479065247844)** (1 messages): 

> `AI/ML Guidance, Model Training Techniques` 


- **新手寻求 AI/ML 指导**：一位成员表示自己是 **AI/ML** 领域的新手，并请求关于从何处开始学习模型训练技术的指导。
   - *寻求大家的帮助。*
- **寻求资源和建议**：该成员还在寻找与初始模型训练进阶相关的资源和技巧。
   - 社区鼓励提供在线课程和论坛的建议，以帮助他们入门。


  

---


---


---


{% else %}


> 完整的各频道详细分析已为邮件订阅截断。
> 
> 如果你想查看完整的详细分析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}