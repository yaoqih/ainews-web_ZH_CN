---
companies:
- deepmind
- anthropic
- cohere
date: '2024-03-14T23:53:52.756548Z'
description: '**DeepMind** 发布了 **SIMA**，这是一款能够在各种 3D 环境和视频游戏中遵循自然语言指令的通用 AI 智能体，推动了具身智能（Embodied
  AI）的发展。**Anthropic** 推出了 **Claude 3 Haiku**，这是其速度最快且价格最实惠的模型，目前已通过 API 和 Perplexity
  提供。


  新的研究探讨了语言模型的扩展定律（scaling laws）、过度训练，并引入了 **Branch-Train-MiX (BTX)**，旨在利用混合专家模型（MoE）高效训练大语言模型。预测显示，在
  **Cohere 的 Command-R**（专注于检索增强生成和工具使用）等 AI 编程助手的辅助下，软件工程岗位将在五年内增长至 **3000 万到 3500
  万**个。


  **欧盟《人工智能法案》（EU AI Act）** 已获批准，要求通用人工智能（GPAI）系统的训练数据必须保持透明。此外，结合差分隐私的隐私保护上下文学习被视为一项极具前景的研究。网络梗图也幽默地讨论了
  AI 软件工程师以及 **Andrej Karpathy** 等知名人物。'
id: efb00c05-281c-4d5f-873c-bdd3fe101a25
models:
- claude-3-haiku
original_slug: ainews-not-much-happened-piday
people:
- demis-hassabis
- fchollet
- abacaj
- andrej-karpathy
title: 圆周率日（Pi Day）没发生什么特别的事。
topics:
- embodied-ai-agents
- natural-language-instructions
- language-model-scaling
- mixture-of-experts
- retrieval-augmented-generation
- software-engineering
- ai-regulation
- differential-privacy
- privacy-preserving-learning
- humor
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月13日至3月14日的 AI 新闻。我们为您查看了 [**358** 条 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **21** 个 Discord（**336** 个频道，**3518** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**426 分钟**。

---

今天是 [GPT4 发布周年纪念日](https://x.com/swyx/status/1636067268802285568?s=20)，但今天没有 GPT5。要不要[加入 @elonmusk](https://x.com/swyx/status/1767664455889097009?s=20)，一起看看[最新的 Latent Space 播客与 Suno AI](https://x.com/FanaHOVA/status/1768327038094750040?s=20)？

https://www.youtube.com/watch?v=gYXjn-V7AEw&feature=youtu.be

（另外，我们昨天错过了重点介绍 [Figure 01 的发布](https://x.com/coreylynch/status/1767927194163331345?s=20)，回想起来，我们认为它的震撼程度/近期重要性略高于 Deepmind SIMA）。

---

**目录**

[TOC] 

---

# PART X: AI Twitter 综述

> 所有综述由 Claude 3 Opus 完成，从 4 次运行中择优


**AI Agent 与环境**

1. [DeepMind 宣布推出 SIMA](https://twitter.com/GoogleDeepMind/status/1767918515585994818)，这是一个通用的 AI Agent，可以在广泛的 3D 环境和视频游戏中遵循自然语言指令，标志着迈向能够处理需要规划和子任务的复杂任务 Agent 的重要一步。（537,888 次曝光）

2. [DeepMind 的 SIMA Agent](https://twitter.com/demishassabis/status/1767977070603219255) 展示了在各种游戏世界中遵循自然语言指令执行任务的能力，类似于人类的玩法。这是具身 AI Agent 领域的一个令人兴奋的发展。（178,835 次曝光）

3. [SIMA 研究](https://twitter.com/GoogleDeepMind/status/1767918524641554899) 专注于开发具身 AI Agent，将抽象语言转化为有用行动，利用视频游戏作为安全、易得的测试环境，而不是为了优化高分。（24,983 次曝光）

**大语言模型与缩放**

1. [Anthropic 推出 Claude 3 Haiku](https://twitter.com/AnthropicAI/status/1768018310615151002)，这是他们最快且最实惠的模型，现已在 API 和 Perplexity（面向 Claude Pro 订阅者）中提供。（299,766 次曝光）

2. [语言模型通过过度训练和在下游任务上的表现能够可靠地进行缩放。](https://twitter.com/arankomatsuzaki/status/1768089079978041552) 一篇新论文探讨了 LM 缩放定律中的差距，提供了关于过度训练的见解，并将模型困惑度（perplexity）与下游性能联系起来。（10,589 次曝光） 

3. [Branch-Train-MiX (BTX)](https://twitter.com/omarsar0/status/1767919732542378089) 是一种通过将专家 LLM 混合成 Mixture-of-Experts LLM 来更高效地训练大语言模型的新方法。事实证明，它比训练一个更大的通用 LLM 或几个独立的专用 LLM 更高效。（11,042 次曝光）

**AI 编程助手与软件工程**

1. [@fchollet 预测](https://twitter.com/fchollet/status/1767935813646716976) 五年后的软件工程师将比现在更多，预计将从现在的 2600-2700 万增长到五年后的 3000-3500 万。他认为，从历史上看，降低编程难度会导致更多的编程岗位。（188,949 次曝光）

2. [Cohere 的 Command-R 模型](https://twitter.com/cwolferesearch/status/1768009088863031766) 专注于检索增强生成 (RAG) 和工具调用 (tool usage) —— 这是构建 LLM 应用的两项关键技能。它解决了将概念验证阶段的 LLM 应用扩展到生产环境中的问题。（2,297 次曝光）

3. 一种观点认为 [AI 将赋能更多软件工程师](https://twitter.com/omarsar0/status/1768000459212530052)，而花哨的演示引起了过度反应。大多数 AI 编程解决方案的范围可能有限，且需要人类监督。（15,308 次曝光）

**AI 安全与监管** 

1. [欧盟《AI 法案》已获议会批准](https://twitter.com/mmitchell_ai/status/1767949324053561560)，这是重大且基本积极的 AI 新闻。（11,126 次曝光）

2. 《AI 法案》中的关键要求包括 [GPAI 系统必须发布“用于训练的内容的详细摘要”。](https://twitter.com/mmitchell_ai/status/1767949866750362041)（1,759 次曝光）

3. 一篇关于 [“具有差分隐私少样本生成的隐私保护上下文学习”](https://twitter.com/_nerdai_/status/1768022092849541453) 的论文被认为是法案批准背景下的前瞻性工作。该论文提议使用预训练的 LLM 从私有数据集中生成具有差分隐私的合成示例。（79 次曝光）

**迷因与幽默**

1. 一个迷因开玩笑说 [一个能自动化一切的“AI 软件工程师”将被用作产品，而不是用来统治市场。](https://twitter.com/abacaj/status/1767810161282855308)（586,645 次曝光）

2. 一条幽默的推文想象 [Andrej Karpathy 离开 Tesla](https://twitter.com/Nexuist/status/1768033939199869050)，原因是他建议将学习率常数从 0.086 更改为 0.0855541。（270 次展示）

3. 一个迷因（meme）暗示 [等待 GPT-5 发布的人们将再次失望。](https://twitter.com/cto_junior/status/1768138360797769915)（1,378 次展示）

**其他值得关注的话题**

- Together Computing [融资 1.06 亿美元](https://twitter.com/togethercompute/status/1767943482054967555)，旨在快速将研究创新投入生产，并构建一个在大规模开源模型上运行生成式 AI 应用程序的平台。（112,647 次展示）

- [Keras 3 基准测试](https://twitter.com/fchollet/status/1768010983224885400) 显示没有单一的“最佳”后端，最佳选择取决于模型架构。Keras 3 模型在不需要自定义优化的情况下，始终比 PyTorch 更快。（51,849 次展示）

- 新的 [LlamaParse 文档解析解决方案](https://twitter.com/llama_index/status/1767948064659210310) 擅长提取图像、表格和图表，并可以通过自然语言指令进行引导。它与 LlamaIndex 集成，用于在复杂文档上构建 RAG 系统。（85,018 次展示）


---

# PART 0: 总结的总结的总结

> 由于 [Claude 3 Haiku 最近发布](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww)，我们将其添加到本次总结运行中，供您与我们的自定义 GPT 总结器进行比较（所有这些都与运行 Part 1/2 总结的 smol 模型不同）。在构建 AINews 平台以提供更好 UX 的过程中，我们将继续并排运行这些模型一段时间。我们注意到，相同的提示词在 3 个 Claude 模型中产生的结果始终不同。我们将在明天的迭代中尝试调整提示词，至少让 Haiku 的表现符合预期。

## Claude 3 Haiku (3B?)


- **Nvidia 限制转换层**：Nvidia 已实施禁令，禁止使用转换层在非 Nvidia 芯片上运行基于 CUDA 的软件，目标直指 ZLUDA 等项目，更多详情见 Tom's Hardware 的文章。一些成员对该禁令的可执行性表示怀疑。

- **CUDA 错误谜题与 Kernel 难题**：CUDA 开发者正在排查诸如 CUBLAS_STATUS_NOT_INITIALIZED 之类的错误，建议指向张量维度和内存问题，详见相关论坛帖子。其他讨论集中在 cuda::pipeline 效率以及理解有效带宽与延迟的关系，参考了 Lecture 8 和关于 CUDA Vectorized Memory Access 的博客等资源。

- **初学者 CUTLASS 安装问答**：新的 AI 工程师寻求关于安装 CUTLASS 的建议，了解到它是一个 header-only 模板库，安装指南可在 CUTLASS GitHub 仓库中找到，并请求了实现自定义 CUDA kernels 的资源。

- **Ring-Attention 项目备受关注**：围绕 ring-attention 实验开展了一系列活动，对话范围从基准测试策略到 'ring-llama' 测试的进展。正如 GitHub 上的 Pull Request #13 所反映的，采样脚本的一个问题正在解决中，并且为对该项目感兴趣的人分享了 Ring-Attention GitHub 仓库。

- **重新录制并发布关于 CUDA 性能的 Lecture 8**：CUDA 社区收到了重新录制版本的 Lecture 8: CUDA Performance Checklist，其中包括 YouTube 视频、GitHub 上的代码以及 Google Docs 上的幻灯片，受到了社区成员的好评。随后对提到的 DRAM 吞吐量数据和 coarsening 的性能差异进行了讨论。

## Claude 3 Sonnet (14B?)

1. **新型 AI 模型发布与功能**：
   - [Cerebras 推出 CS-3 AI 加速器](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)，能够在单颗芯片上训练高达 **24 万亿参数的模型**，拥有 4 万亿个晶体管和 125 petaflops 的算力。
   - [Anthropic 发布 Claude 3 Haiku](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)，这是一款在 OpenRouter 上提供的快速且具有成本效益的模型，运行速度约为 120 tokens/s，每美元可处理 400 万个 prompt tokens。
   - 具有 128k token 上下文窗口的 [Cohere Command-R 模型](https://openrouter.ai/models/cohere/command-r) 现已上线 OpenRouter，价格为每美元 200 万个 prompt tokens。
   - [DeepMind 的 SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/) 是一款新型通用 AI Agent，能够理解视频游戏环境中的自然语言指令，尽管其技术细节缺乏透明度。

2. **AI 安全与漏洞担忧**：
   - 一篇关于 [ComPromptMized 的新论文](https://sites.google.com/view/compromptmized) 揭示了针对 **Gemini Pro、ChatGPT 4.0 和 LLaVA** 等 AI 模型的提示词注入攻击（Prompt Injection Attacks），突显了基于 GenAI 应用的漏洞。
   - 关于 [OpenAI 安全事件](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca) 及其对 AI 安全影响的讨论。
   - 关于欧盟新 [AI 法案](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law) 的辩论，该法案要求披露 AI 生成内容并避免非法输出，引发了对其可行性以及对开源模型影响的担忧。

3. **多模态 AI 与可解释性的进展**：
   - [MoAI (Mixture of All Intelligence)](https://arxiv.org/abs/2403.07508) 将来自计算机视觉模型的辅助视觉信息整合到 LLVMs 中，以增强零样本（Zero-shot）视觉语言任务。提供 [GitHub 仓库](https://github.com/ByungKwanLee/MoAI) 和 [Hugging Face 演示](https://huggingface.co/BK-Lee/MoAI-7B)。
   - [Soniajoseph 宣布推出开源多模态机械可解释性库](https://twitter.com/soniajoseph_/status/1767963316943728779)，适用于 Hugging Face CLIP/ViT 模型，详情见 [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic)。
   - [LlamaParse](https://twitter.com/llama_index/status/1767948064659210310) 是一款新型文档解析器，可以处理图像、表格、图表并遵循自然语言指令，性能优于其他工具。

4. **用于 LLM 预训练和微调的合成数据生成**：
   - 关于[使用合成数据](https://eugeneyan.com/writing/synthetic/)作为人工标注替代方案，用于语言模型的预训练、指令微调（Instruction-tuning）和偏好微调（Preference-tuning）的讨论，具有潜在的质量优势。
   - [AI News 通讯](https://buttondown.email/ainews/) 提供了 AI Discord 社区和顶级 Twitter 账号的有价值摘要，得到了 Soumith Chintala 和 Andrej Karpathy 等专家的推荐。
   - 关于[微调是否能为模型传授新知识](https://github.com/stanfordnlp/dspy/blob/0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d/docs/api/optimizers/BootstrapFinetune.md?plain=1#L5)的辩论，以及微调在风格迁移与知识获取方面的效率对比。

## Claude 3 Opus (>220B?)

- **Hermes 2.5 性能超越 Hermes 2**：带有 [代码指令示例](https://github.com/NousResearch/Hermes-Function-Calling) 的 **Hermes 2.5** 更新版与 Hermes 2 相比，性能有了显著提升。社区正在积极讨论 JSON 模式与 Function Calling 等相关话题，并澄清 JSON 模式需要预定义的 Schema，而 Function Calling 则涉及执行函数的响应。

- **Cerebras 揭晓 AI 巨兽**：**Cerebras Systems** 推出了其 **CS-3** AI 加速器，能够训练高达 24 万亿参数的模型。正如 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) 中详述的那样，这颗巨大的芯片拥有 4 万亿个晶体管，预计将提供 125 petaflops 的 AI 算力。

- **丰富的 AI 趋势**：从 Stanislaw Lem 的科幻小说推荐到 SDAM 开发，再到 *Devin AI 软件工程师* 的 [YouTube 首秀](https://www.youtube.com/watch?v=NSPtrrUQ_fw)，社区正密切关注各种 AI 和工程奇迹。对能够提供 100k+ 上下文的开源模型的渴望，以及对信息共享中隐私问题的担忧，也体现了兴趣的多样性。

- **辩论 AI 的去中心化**：在关于 **TAO** 可能挑战 **Hugging Face** 的讨论中，社区深入探讨了中心化与去中心化 AI 模型平台的辩论。新项目 **Shoggoth** 的引入引发了好奇，但由于链接失效，目前缺乏详细信息。

- **Claude 3 助力 Haiku 创作**：**[Perplexity Labs](https://labs.pplx.ai)** 推出了 **Claude 3 Haiku**，吸引用户免费体验诗歌 AI 功能，增强了该平台的创意工具套件。

- **多样化的 AI 工具箱**：工程师和开发人员正积极使用 **Perplexity AI** 进行多种用途，如代码支持和 SE 故障排除，同时创意性地尝试新添加的功能，如 AI 生成的 **Haikus**。该平台的本地搜索功能现在通过集成 Yelp 和 Maps 得到了增强，以便更高效地发现本地商家。

- **AI 生态系统的竞争与观点**：公会主持了比较各种 AI 模型的激烈辩论；**GPT-4** 和 **Mistral** 展开对决，一些人认为前者更优越，而另一些人则青睐后者的速度。

- **API 集成与模型限制**：用户讨论了使用 Perplexity 的模型处理复杂查询，并利用 **Perplexity API** 开发应用程序（如 Firefox 扩展），同时注意到 **25MB 的上传限制** 以及在处理大型数据库（如房地产相关数据库）时性能的不确定性。

- **API 焦点：问题与潜力**：关于 **Perplexity API** 中 URL 引用闭测的咨询正等待内部人士的见解，而其他人则在寻求有关 API 条件检查性能的建议。成员们还研究了 "return_citations" 选项的行为，并确定了处理最新信息的最佳模型，特别指出了 **Sonar-small-online** 和 **sonar-medium-online** 的实时数据访问能力。

- **脱离 UI 运行 LM Studio**：用户研究了在没有用户界面的情况下在家庭网络上运行 LM Studio 的 API 服务，重点关注服务器模式和 localhost 连接。据其 [GitHub 仓库](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)介绍，`llama.cpp` 是一个可行的选择，它在没有 **AVX2** 的情况下支持 **AVX**，并允许独立于 LM Studio UI 运行。

- **LM Studio 的限制激发了创意变通方案**：LM Studio 的限制之一是无法通过编程方式启动服务或连接到互联网；用户创造性地使用批处理文件和 PowerShell 脚本来自动启动 LM Studio 推理服务器，展示了社区的机智。

- **强大模型的扩展与检验**：[Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) 模型使用 *YaRN* 方法将上下文窗口扩展到了 128k token，同时还讨论了模型的 Perplexity（困惑度）以及对 "Yet Another <X>" 命名惯例的幽默失望。此外，一些人分享了特定格式的障碍，例如 **Command-R 35B v1.0 GGUF** 格式与 `llama.cpp` 的不兼容问题。

- **ROCm 综述**：分享了在 LM Studio 中使用 **ROCm** 支持的实际经验，包括使用 AMD 清理工具和避免使用 PRO 驱动程序等故障排除步骤。Vision 模型被证明具有挑战性，建议在图像生成项目中选择 Nvidia GPU 而非 AMD。此外，一位用户发现，在 BIOS 设置中禁用技嘉主板上的 iGPU 可以让其 RX 7900 XT 在 ROCm 下获得更好的使用效果。

- **硬件讨论升温**：**SLI/NVLink** 的成本引发了辩论，此外还有关于克服 Mac OS **最低 VRAM 要求**、规划 PC 硬件升级以及在 LM Studio 中平衡多模型部署的讨论。另有对话涉及选择合适的双用途显示器，尽管存在烧屏风险，但用户仍倾向于 OLED 屏幕，并偏好高刷新率以匹配 Nvidia 4090 等顶级显卡。

- **AI 淘金热继续**：**Cognition**、**Magic** 和 **Fluent** 等多家 **AI 初创公司** 吸引了令人瞩目的风险资本投资，讨论引起了人们对 AI 公司持续融资热潮的关注。参与者分享了一组推文，概述了这些公司及其筹集的资金，参考了 [chiefaioffice](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 的动态。

- **Cerebras 展示其 AI 实力**：**Cerebras Systems** 推出了 **CS-3 AI accelerator**，声称其能够训练高达 24 万亿参数的模型。该公告引发了广泛关注，讨论中还提到了相关的 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) 和一条 [推文](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **OpenAI 的安全红警**：成员们讨论了 OpenAI 的一个**安全问题**，并引用了在 [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca) 中提供的详细 **Post Mortem** 分析。社区深入探讨了其对 AI 安全的影响。

- **为合成数据洞察做准备**：宣布了一场即将举行的关于用于 Finetuning 的合成数据的演讲，预读材料见 [Eugene Yan 的文章](https://eugeneyan.com/writing/synthetic/)。该小组强调了在 pretraining 和 fine-tuning 语言模型中使用合成数据作为人工标注替代方案的作用。

- **重新思考 LLM 中的数据**：深入讨论探讨了使用**合成数据**进行 LLM 的 pretraining 和 fine-tuning，以及通过 fine-tuning 获取知识的影响。讨论中提到的一篇提供重要见解的博客文章可以在 [eugeneyan.com](https://eugeneyan.com/writing/synthetic/) 找到，工程专业人士还提到 [AI News](https://buttondown.email/ainews/) 的摘要服务是一个宝贵的资源。

## ChatGPT (GPT4T)

<div><ul><li><p><strong>语言模型中的位置编码 (Positional Encodings)</strong>：<strong>Nous Research AI Discord</strong> 讨论了位置编码在增强因果语言模型处理更长序列性能方面的关键作用。一篇关键论文 <a target="_new" href="https://arxiv.org/pdf/2203.16634.pdf">"Understanding Positional Encodings in Large Language Models"</a> 被重点提及，因为它在该领域提供了深刻的见解。</p></li><li><p><strong>Hermes 2.5 Function Calling</strong>：随着 Hermes 2.5 的推出，观察到了显著的性能提升，特别是在 function calling 与 JSON mode 的对比中，社区关注了其在 <a target="_new" href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub</a> 上的实际案例。</p></li><li><p><strong>Cerebras 的 CS-3 AI Accelerator</strong>：Cerebras Systems 推出了其 CS-3 AI accelerator，能够训练高达 24 万亿参数的模型。这一硬件里程碑拥有 4 万亿个晶体管，并承诺提供 125 petaflops 的 AI 算力，详情见其 <a target="_new" href="https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine">新闻稿</a>。</p></li><li><p><strong>Perplexity AI 的 Claude 3 Haiku</strong>：<strong>Perplexity AI</strong> 展示了 Claude 3 Haiku，强调了该模型创作俳句 (Haikus) 的能力，这是其扩展 AI 创造力努力的一部分，更多细节可在 <a target="_new" href="https://labs.pplx.ai">Perplexity Labs</a> 查看。</p></li><li><p><strong>OpenAI Discord 中的本地模型测试</strong>：关于在配备高达 4xT4 的环境中使用 LLM Studio 测试本地模型（特别是 Meditron 和 Mistral）的讨论非常突出，包括为了获得最佳性能而进行 fine-tuning 这些模型的最佳实践。</p></li><li><p><strong>多模态模型的可解释性</strong>：<strong>Alignment Lab AI Discord</strong> 正在为专注于多模态模型的开源可解释性项目寻找合作者，soniajoseph_ 在 <a target="_new" href="https://twitter.com/soniajoseph_/status/1767963316943728779">Twitter</a> 和 <a target="_new" href="https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic">LessWrong</a> 上分享了更多细节。</p></li><li><p><strong>Devin，自主软件工程师</strong>：<strong>Skunkworks AI</strong> 和 <strong>AI Engineer Foundation</strong> 都重点介绍了 Devin，它被 Cognition Labs 介绍为世界上第一个自主软件工程师。该 AI 的能力和介绍涵盖在他们的 <a target="_new" href="https://www.cognition-labs.com/blog">博客文章</a> 和一段 <a target="_new" href="https://www.youtube.com/watch?v=NSPtrrUQ_fw">YouTube 视频</a> 中。</p></li></ul></div>

---

# PART 1: High level Discord summaries

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **位置编码深度解析 (Positional Encodings Decoded)**：关于位置编码在因果语言模型（causal language models）中作用的讨论非常热烈，观点指出位置编码对于有效处理长序列至关重要。相关论文推荐：["Understanding Positional Encodings in Large Language Models"](https://arxiv.org/pdf/2203.16634.pdf)。

- **解锁 Hermes 2.5 的秘密**：**Hermes 2.5** 的更新及其 [代码指令示例](https://github.com/NousResearch/Hermes-Function-Calling) 相比 Hermes 2 带来了显著的性能提升。社区正在积极讨论 JSON 模式与 function calling 等相关话题，明确了 JSON 模式需要预定义的 schema，而 function calling 则涉及执行函数后的响应。

- **Cerebras 揭晓 AI 巨兽**：Cerebras Systems 推出了其 **CS-3** AI 加速器，能够训练高达 24 万亿参数的模型。这款巨型芯片拥有 4 万亿个晶体管，预计将提供 125 petaflops 的 AI 算力，详见 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)。

- **AI 趋势大观**：从斯坦尼斯瓦夫·莱姆（Stanislaw Lem）的科幻小说推荐到 SDAM 开发，再到 *Devin AI 软件工程师* 的 [YouTube 首秀](https://www.youtube.com/watch?v=NSPtrrUQ_fw)，社区正密切关注各种 AI 和工程奇迹。对能够提供 100k+ context 的开源模型的渴望，以及对信息共享中隐私问题的担忧，也体现了兴趣的多样性。

- **辩论 AI 的去中心化**：在关于 **TAO** 是否可能挑战 **Hugging Face** 的讨论中，社区深入探讨了中心化与去中心化 AI 模型平台的辩论。新项目 **Shoggoth** 的引入引发了好奇，但由于链接失效，目前缺乏详细信息。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **Claude 3 赋能 Haiku 创作**：**[Perplexity Labs](https://labs.pplx.ai)** 引入了 **Claude 3 Haiku**，吸引用户免费体验诗意 AI 的能力，增强了平台的创意工具套件。

- **多样化的 AI 工具箱**：工程师和开发人员正积极利用 **Perplexity AI** 进行多种用途，如编码支持和 SE 故障排除，同时创意性地尝试新加入的功能，如 AI 生成的 **Haikus**。平台的本地搜索能力现在通过集成 Yelp 和 Maps 得到增强，从而更高效地发现本地商家。

- **AI 生态系统的竞争与观点**：社区内对各种 AI 模型进行了激烈的辩论；**GPT-4** 和 **Mistral** 被拿来对比，一些人认为前者更优，而另一些人则青睐后者的速度。

- **API 集成与模型限制**：用户讨论了使用 Perplexity 的模型处理复杂查询，并利用 **Perplexity API** 开发应用程序（如 Firefox 扩展），同时注意到 **25MB 的上传限制**，以及在处理大型数据库（如房地产相关数据库）时性能的不确定性。

- **API 焦点：问题与潜力**：关于 **Perplexity API** 中 URL 引用（URL citations）内测的咨询正等待内部人士的见解，而其他人则在寻求有关 API 在条件检查（condition checking）性能方面的建议。成员们还研究了 "return_citations" 选项的行为，并确定了处理最新信息的最佳模型，特别指出 **Sonar-small-online** 和 **sonar-medium-online** 具有实时数据访问能力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

**脱离 UI 界面使用 LM Studio**：用户探讨了在没有用户界面的情况下在*家庭网络*上运行 LM Studio 的 API 服务，重点关注服务器模式和 localhost 连接。讨论强调了 `llama.cpp` 是一个可行的选择，它支持 **AVX**（无需 **AVX2**）并允许独立于 LM Studio UI 运行，详情参考其 [GitHub repository](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)。

**LM Studio 的局限性催生创意变通方案**：LM Studio 的限制之一是无法通过编程方式启动服务或连接到互联网；用户创造性地利用批处理文件和 PowerShell 脚本来自动化启动 LM Studio 推理服务器，展示了社区的机智。

**强大模型的扩展与探讨**：[Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) 模型通过 *YaRN* 方法将上下文窗口扩展到了 128k token，同时用户还讨论了模型 Perplexity（困惑度），并对“Yet Another <X>”这种命名惯例进行了幽默的吐槽。此外，一些用户分享了特定格式的障碍，例如 **Command-R 35B v1.0 GGUF** 格式与 `llama.cpp` 的不兼容问题。

**ROCm 动态汇总**：分享了在 LM Studio 中支持 **ROCm** 的实际经验，包括使用 AMD 清理工具和避免使用 PRO 驱动等故障排除步骤。Vision 模型被证明具有挑战性，建议在图像生成项目中选择 Nvidia GPU 而非 AMD。此外，一名用户发现，在技嘉主板的 BIOS 设置中禁用 iGPU 可以让他们的 RX 7900 XT 在 ROCm 下表现更好。

**硬件讨论升温**：**SLI/NVLink** 的成本引发了辩论，此外还讨论了如何克服 Mac OS 的**最小 VRAM 要求**、PC 硬件升级策略以及 LM Studio 中多模型部署的平衡。另有对话涉及选择合适的双用途显示器，尽管存在烧屏风险，用户仍倾向于选择 OLED 屏幕，并偏好高刷新率以匹配 Nvidia 4090 等顶级显卡。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **AI 淘金热持续**：包括 **Cognition**、**Magic** 和 **Fluent** 在内的多家 **AI 初创公司** 吸引了巨额风险投资，讨论引起了人们对 AI 公司持续融资热潮的关注。参与者分享了一系列推文，概述了这些公司及其筹集的资金，引用了 [chiefaioffice](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 的信息源。

- **Cerebras 展示其 AI 实力**：**Cerebras Systems** 推出了 **CS-3 AI 加速器**，声称其能够训练高达 24 万亿参数的模型。该公告引发了广泛关注，讨论中还提到了相关的[新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)和[推文](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **OpenAI 安全红警**：成员们讨论了 OpenAI 的一个**安全问题**，并引用了 [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca) 中提供的详细 **Post Mortem** 分析。社区深入探讨了其对 AI 安全的影响。

- **准备迎接合成数据见解**：一场关于“用于微调的合成数据”的演讲即将举行，预读材料可见 [Eugene Yan 的文章](https://eugeneyan.com/writing/synthetic/)。小组强调了在预训练和微调语言模型中，使用合成数据作为人工标注替代方案的重要性。

- **重新思考 LLM 中的数据**：深入讨论探讨了使用**合成数据**进行 LLM 预训练和微调的影响，以及通过微调获取知识的意义。讨论中提到的一篇提供深刻见解的博客文章可在 [eugeneyan.com](https://eugeneyan.com/writing/synthetic/) 找到，工程专业人士还提到 [AI News](https://buttondown.email/ainews/) 的摘要服务是一个宝贵的资源。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord 摘要

**可视化 Token 概率**：讨论指出需要可视化句子中的 Token 概率，并建议使用 **lm_head** 的输出和 softmax。然而，目前似乎缺乏用于此类可视化的特定插件。

**AI 的快速进展**：讨论中充满了对 AI 快速发展的关注，包括对 Elon Musk 的 **Grok 模型** 的期待，以及关于 OpenAI 真实性的闲聊。

**Unsloth AI 应对 Colab 问题**：**Unsloth AI** 分享了针对 Google Colab 的 PyTorch 更新问题的修复方案，并提供了一个命令列表，帮助用户自行纠正这些问题。明确了 Unsloth AI 的兼容性，指出其目前尚不支持多 GPU 或 GGUF 格式模型的微调，但可以处理单 GPU 设置下的 4-bit 量化。

**数据准备讨论**：一场活跃的对话建议为数据准备创建 FAQ，并认为采用更自动化的方法可能会更有益。

**Sophia 优化器引起关注**：一种名为 **Sophia** 的新优化器引起了社区的关注，该优化器旨在减少语言模型的训练时间和成本。虽然尚未经过测试，但人们乐观地认为它可以有效地取代现有的优化器（[Sophia Optimizer Paper](https://arxiv.org/abs/2305.14342)）。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **GPT-3.5 在 Python 脚本编写方面表现出色**：一段对话强调了 **GPT-3.5** 使用 Python 编写程序并成功生成重复词素示例的能力。任务的复杂性并未阻碍一些成功输出的分享。

- **本地模型备受关注**：工程师们分享了使用 **LLM Studio** 测试本地模型的见解，据报道在多达 4xT4 的配置上具有强大的推理能力，**Meditron** 被提及为一个出色的模型。对话扩展到对 **Mistral** 等模型进行微调的考虑，建议使用 **A100 40GB GPU** 来完成任务，尽管也可以尝试在没有 GPU 的情况下微调 **GPT-3.5**。

- **GPT-5 传闻被辟谣**：**Microsoft Copilot 页面**上意外提到的“优先访问 GPT-4 和 GPT-5 Turbo”引发了关于 **GPT-5** 是否存在的猜测，结果证明这只是一个拼写错误。这使得爱好者们一致认为 GPT-5 不会很快发布。相关链接：[Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot)。

- **GPT-4 的系统故障**：用户遇到了 **GPT-4** 的大规模故障，在 iOS 应用和包括 Chrome、Edge 在内的网页浏览器等多个平台上都突显了这一问题。一些人发现图片附件提供了一种临时的解决方法，并建议查看 OpenAI [状态页面](https://status.openai.com/)获取更新。

- **文化差异影响 API 理解**：在关于 **Assistant API** 的讨论中，一位用户观察到 API 因为逗号位置误解了数字 "450,00"，这可能导致数据处理中的重大错误。建议根据当地文化格式进行调整（例如设置 locale）并使用正反例来提高准确性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

**DeepMind 推出通用游戏 AI**：DeepMind 推出了 **SIMA**，展示了在多种游戏设置中的自然语言熟练度，但研究社区指出其技术细节不足。批评者对用于验证该 Agent 有效性的指标持谨慎态度，并对游戏专业知识的定义以及 AI 在竞技游戏场景（特别是在像 **BR 游戏** 这样不可预测的多 Agent 系统中）的更广泛影响展开了辩论。

**研究论文付费墙引发愤怒**：前沿 AI 研究的可访问性受到出版商付费墙的阻碍，这引发了围绕创新神经网络训练动态和多样化网络架构集成的讨论。此外，人们还担心为 AI 生成内容添加水印的后果，这可能会限制其实用性。

**多模态模型可解释性库发布**：一个新的多模态机制可解释性库吸引了合作兴趣，同时讨论深入探讨了多语言 Transformer 中模型无关性和语言依赖动态的复杂性。双语模型中的 Tokenization 偏差探索以及用于深入了解模型潜表征（latent representations）的向量数据库查找（vector-DB-lookup）方法也受到了关注。

**语言模型进入竞技场**：**LM evaluation harness** 社区正在尝试通过学习率冷却（learning rate cooldowns）来改进基准测试。由于近期旨在提高安全性的 API 更改，他们在添加 Logits 方面面临挑战，这引发了关于为生成式模型调整任务以及测试不同 Checkpoints 以评估模型性能的讨论。

**Megatron 与 NeoX 相遇**：GitHub 上的一个 [Pull Request](https://github.com/EleutherAI/gpt-neox/pull/1185) 引发了关于将 GPT-NeoX 与上游 Megatron 进一步对齐以进行 Transformer Engine 集成的潜在益处的辩论。社区正在征求反馈，以权衡该策略的优势与代码分歧（code divergence）之间的利弊。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **速度追求者关注量化与 Groq**：工程师们讨论了在 A100 40GB 等 GPU 上加速 *Phi 2 微调模型* 推理的方法，选项包括 vLLM、Olama 或 Axolotl。Quantization 被提及为潜在的加速手段，而 Groq 的 NPU 展示了在 *Mixtral* 上每秒 500 个 Token 的性能。

- **模型立法风波**：欧盟新的 AI 立法和近期的版权下架通知引发了围绕版权、AI 生成内容和 DMCA 合规性的激烈辩论。开源支持者正在努力应对政府对共享模型权重的限制。

- **Prompt Engineering 热潮**：SuperPrompt 和一种针对 Danbooru 标签的新自动补全标签生成器等工具被提出，旨在提高较小模型在通常由大型 LLM 处理的任务中的能力。

- **AI 数据拉锯战**：关于 **MoAI** 的新论文引起了相当大的关注，该论文利用了来自专门计算机视觉模型的辅助视觉信息。这些努力强调了 AI 社区在创建能够增强 Zero-shot 视觉语言任务的通用 LLVMs 方面的持续推进。

- **内存机制被误解**：一次讨论澄清了关于大型模型内存使用的误解，指出 mmap 可能会隐藏实际的内存使用情况，直到数据被访问时才会反映出来。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **可视化对比变得更简单**：[Open LLM Leaderboard Viz] 现在支持重新排序指标并可视化对比最多三个模型，正如 [HuggingFace Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) 上的新更新所示。
- **使用自定义标签演进 NER**：一个名为 GLiNER 的新模型支持为命名实体识别（NER）进行即时自定义标签选择，与固定实体模型相比提供了更强的适应性。查看 [HuggingFace Spaces](https://huggingface.co/spaces/tomaarsen/gliner_base) 和 [GitHub](https://github.com/urchade/GLiNER) 上的演示和额外资源。
- **动态模型加载中的延迟困扰**：一位用户报告了在将 `peft` 与 *diffusers* 集成时（特别是使用 `load_lora_weights` 函数时）出现的显著延迟，并在 [HuggingFace blog](https://huggingface.co/blog/lora-adapters-dynamic-loading) 上分享了经验和指南。
- **免费增值 LLM 的苦恼与 Space 的怪象**：关于 Hugging Face Spaces 中基于 CPU 的免费增值 LLM 的可访问性和实用性的讨论正在进行，同时还讨论了向 Hugging Face `transformers` 贡献代码的最佳实践以及公共空间中的数据隐私担忧。
- **MyShell 呼吁 AI 民主化**：一位用户拥护具有投票系统的多 AI 决策模型想法，并建议 **MyShell's Pro Config** 可以管理这种编排，指向 [MyShell](https://myshell.ai/) 以进一步探索 AI 原生应用部署。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 总结

- **OpenRouter 导航受阻**：用户报告了 OpenRouter 的**临时服务中断**，由于数据库更新，活动行消失了。该问题持续了约三分钟，据称不会影响计费，因为“*这些补全（completions）都不会收费*”。
- **提升 Claude 的声望**：OpenRouter 宣布 **Claude 3 Haiku** 上线，拥有极高的速度（约 120 tokens/s）和成本效率（400 万 prompt tokens/$）。其部署提供经过审核和自我审核模式，被认为是快速响应应用的理想选择。[点击查看](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)。
- **Command-R 进驻 OpenRouter**：Cohere 的 **Command-R** 模型具备 128k token 上下文能力，现已集成到 OpenRouter。其价格为每美元 200 万 prompt tokens，专注于无缝的用户交互。[探索 Command-R](https://openrouter.ai/models/cohere/command-r)。
- **Olympia.Chat 与 OpenRouter 结盟**：[Olympia.Chat](https://olympia.chat) 已采用 OpenRouter 为其面向企业的 AI 驱动服务提供支持。他们计划很快发布一个 **Ruby library**，以进一步挖掘 OpenRouter 的能力。
- **AI 对手对决，奇特现象频出**：在 general 频道中，用户对 Gemini 和 Claude 等各种模型进行了有趣的对比。用户辩论了它们在编程和创意任务中的效能，注意到某些模型偏好使用项目符号，并权衡了性能和内容限制方面的优缺点。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

**LlamaParse 在文档解析中大获全胜**：**LlamaParse** 提升了文档解析能力，能够处理图像、表格、图表并遵循自然语言指令，承诺带来显著的性能提升，[正如在 Twitter 上所见](https://twitter.com/llama_index/status/1767948064659210310)。

**使用 Presidio 保护数据**：**LlamaIndex** 重点介绍了 **Presidio**，这是 Microsoft 的开源工具，用于识别和匿名化 PII，强化了数据保护的重要作用，[这条推文对此进行了强调](https://twitter.com/llama_index/status/1768050386823463368)。

**RAG 在财务演示文稿中受挫**：在处理财务 PowerPoint 演示文稿时，由于格式复杂，RAG 难以应对，需要改进文本定位和解析的方法，[详见这条推文](https://twitter.com/llama_index/status/1768303288381030408)。

**Azure 存储异常困扰用户**：尽管遵循了 [AzureAISearchIndexDemo 指南](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html)，处理 **Azure AI Search Index** 的用户仍报告存储大小（3mb）与向量索引大小（0）之间存在差异。

**#general 频道中的开发者困境**：工程师们遇到了多个障碍，从 `OpenAIPydanticProgram` 的警告（可通过安装 `llama-index-program-openai` 解决），到令人费解的 `npx create-llama` 错误，以及 **OpenAIAssistantAgent** 响应缓慢；升级到流式传输并解决最近的 **OpenAI API** 性能问题可能会缓解延迟。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **NVIDIA 的 GPUDirect Storage 引起关注**：成员们分享了一个关于利用 NVIDIA GPUDirect Storage 的[入门视频](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4)，并讨论了将其与 Axolotl 系统集成以潜在提升性能的可能性。此外，还提出了一个关于 Axolotl 代码片段的问题，重点在于其在模型加载中的作用，特别是与 *peft models* 相关的部分。

- **开源模型成为焦点**：对话围绕使用 [Mistral 和 Mixtral](https://huggingface.co/) 等开源模型的优势展开，主要归功于它们的易获得性和极少的过滤。此外，针对特定的医疗训练用途，大家还在 Mixtral 或 Qwen 70B 之间进行权衡，而即将推出的新模型增加了决策的复杂性。

- **VRAM 限制与训练雄心**：面对 VRAM 限制，出现了关于训练更大模型的技术咨询，重点讨论了用于 MPS 后端的 [PyTorch Metal Performance Shaders](https://developer.apple.com/metal/pytorch/) 等工具以及高效 fine-tuning 的策略。关注点集中在 OOM 问题以及如何为训练最好地格式化原始文本。

- **LoRA 微调模型的推理协助**：有人请求提供在基于 `Mistral-7B-v0.1` 的 fine-tuned LoRA 模型上运行推理的示例代码，得到的建议是使用 **vLLM** 而非 `transformers` 以实现更快的批处理推理。一位成员采纳了该建议，并参考 [vLLM 快速入门指南](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)来优化其流程。

- **Mistral Medium 与 Mixtral 的对比**：社区用户注意到 **Mistral Medium** 在生成响应方面似乎优于 Mixtral，表现为更加简洁且更擅长遵循指令。此外，还分享了在没有明确提示的情况下，**RAG 性能**中出现意外引用生成的观察。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

**因漏洞问题 LangChain 0.2 进入快速发布通道**：`langchain 0.2` 的加速发布正在进行中，通过与 `langchain-community` 分离来解决 CVE。该过程在 [GitHub](https://github.com/langchain-ai/langchain/discussions/19083) 上有详细说明，并正在寻求社区反馈以满足用户需求。

**LangChain 的挑战与创新**：用户讨论了各种 LangChain 问题，包括 `AgentExecutor` 的 Bug、AI Agent 的优势，以及使用尚在开发中的基准测试评估 AI Agent 的行为。一项咨询集中在如何将 `tools = [cat_tool]` 等变量集成到 Langsmith Hub 提示模板中。如需更多指导，用户可参考 LangChain [评估指南](https://python.langchain.com/docs/guides/evaluation/)。

**令人瞩目的协作与演示**：
- ReAct Agent 的推理引擎已开放测试，其灵感源自论文 [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)。
- 一个使用 RAG 进行高效查询的开源 LangChain 聊天机器人已发布在 [GitHub](https://github.com/Haste171/langchain-chatbot)。
- MindGuide 利用 LangChain 提供心理健康支持，更多阅读请参考 [下载 PDF](https://arxiv.org/abs/2403.05568)。
- Claude 通过 LangGraph Agent Supervisor 与 LangChain 集成，演示见 [GitHub Notebook](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb)。
- Deci AI 推出了新的 nano 模型 API，并提供了 [基础使用](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing) 和 [LangChain 使用](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing) 的 Colab 笔记本。

**#golang 和 #llm 爱好者的教程中心**：
- “使用 Langchaingo 创建提示模板”是一个分步视频教程，可在 [YouTube](https://youtu.be/dcBEtgh4078) 上找到，非常适合渴望掌握提示模板的开发者。
- “让我们使用 Hermes 2 Pro 7B 进行函数调用”是一个深入探讨使用 Hermes 2 Pro 7B 模型进行 function calling 的视频指南，代码和示例位于 [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main)。该视频面向 #largelanguagemodels 爱好者，可在 [YouTube](https://www.youtube.com/watch?v=PzaidfqDtGI) 上观看。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 摘要

- **Twitter 引发 Aya 项目热议**：[Andrew Curran 的一条推文](https://twitter.com/AndrewCurran_/status/1767916848987914487)引发了关于语言应用和跨界合作的讨论，强调了通过 Aya 项目进行子小组工作的重要性；同时另一项互动指出，德语已得到大型 LLM 的良好支持。

- **GPT-4 维持统治地位**：GPT-4 的实力在 **LeetCode** 排名中继续领先，正如[一篇包含多种模型对比的论文](https://livecodebench.github.io/pdfs/paper.pdf)所强调的那样。

- **寻求安全领域的基石**：有人询问了最近一项任务中使用的模型细节，并寻求关于基础模型提供商在文本生成后进行安全过滤程度的来源或文档。

- **生物风险讨论引发热议**：提到了补读积压的简报并感谢批判性读者，这与一条关于生物风险（bio risk）的推文有关，该推文由于可能的沟通误解或缺乏上下文而引发了争论和困惑。

- **Claude-3 搅动 AI 领域**：人们对 GPT-4.5 的发布充满期待，而 **Claude 模型家族**，特别是 Claude-3-Opus，因其顶尖排名而受到赞誉（[LM SysOrg 关于 Claude-3 的更新](https://fxtwitter.com/lmsysorg/status/1767997086954573938)）。对话还深入探讨了在标准化 AI 用于研究文献辅助方面的障碍，指出了进一步的研究方向（[关于 AI 文献综述的 Arxiv 讨论](https://arxiv.org/abs/2402.18819)）。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 摘要

- **CUDA Toolkit 在 Ubuntu 23.10 上遇到障碍**：一位用户在 Ubuntu 23.10 上运行 `compute-sanitizer` 时因 `nvidia-cuda-toolkit` 出现问题，这可能预示着 **版本不匹配** 问题，因为最新的 NVIDIA toolkit 官方并不支持 22.04 之后的 Ubuntu 版本。

- **教育科技平台寻求 CUDA 专家**：*Christo_allstreet* 正在为 [getworldclass.app](http://getworldclass.app) 寻找 **CUDA 专家**。欢迎具备相关专业知识的人士直接联系以获取咨询机会。

- **Triton 和 CUDA 问题排查**：社区分享了诸如使用 `TRITON_INTERPRET=1` 环境变量以及已弃用的 `@triton.jit(interpret=True)` 方法来 **调试 Triton kernel** 的策略，强调了传统的调试方法。[YouTube 视频和 GitHub 讨论](https://github.com/openai/triton/issues/517#issuecomment-1971327089)可作为教育资源。

- **NUMA：并非那么神速的分析**：在对比 **BLAS** 和 **NumPy** 时，一个显著的性能差距被指出，表明 **高达 90% 的潜在 BLAS 吞吐量** 在 NumPy 操作中损失了。会议还讨论了对 **SIMD 封装**（作为小向量操作解决方案）的兴趣，以及对技术选择的信息传递的关注。

- **GTC 聚会与 NSight 工具讨论**：预告了即将举行的 GTC 与会者会议，同时强调了 **NSight Systems** 对于多 GPU 应用分析的重要性，并分享了指南和可视化图表以更好地理解和优化性能。

- **书籍讨论中探索 CUDA 编程模型优点**：关于 SM 如何在 SIMD 模型中执行线程的争论通过 **GA102 SM** 架构的例子得到了澄清，揭示了核心执行的限制。

- **Axolotl Ring Attn 问题讨论**：讨论了 **axolotl 项目**，一名成员概述了成功初始化所需的配置（`pad_to_sequence_len: true`），并分享了与 **ring-attn** 配置对比的 [loss 结果](https://api.wandb.ai/links/iron-bound/v6mxxcj2)。他们还分享了 GitHub 上 **ring_attention_patching** 分支的链接：[ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)。

- **AI 挑战经典游戏**：一篇 [arXiv 论文](https://arxiv.org/abs/2403.05468)详细介绍了 **GPT-4 玩 Doom 的能力**，仅凭游戏的文本描述，展示了该模型的规划和推理能力。

- **Meta 的法律技术冲突**：**Meta** 对一名前高管提起诉讼，指控其窃取机密文件，这引发了关于企业间谍活动和 AI 数据初创公司风险的严肃对话。法律[文件](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf)描绘了一幅“公然不忠和不诚实行为”的画面。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **AI 爱好者们，请留意这个日期！**：柏林的 AI 社区正在为 3 月 21 日的 **AI Tinkerers event** 做准备，由于需求旺盛，目前仅剩 **8 个席位**。人们一直在寻求关于 DiscoLM 在德国数据集上进行 fine-tuning 的细节，随后发现 DiscoLM-mixtral-8x7b-v2 并没有像 [Hugging Face 的 DiscoLM 70b 模型](https://huggingface.co/DiscoResearch/DiscoLM-70b) 页面所确认的那样，在德国数据上进行过大量训练。

- **为诗意 AI 进行基准测试**：引入了一个新的创意写作基准测试，这可能会重塑我们评估语言模型细微能力的方式。请在 [EQ-Bench GitHub 仓库](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing) 查看并测试原型。

- **深入德语深度**：AI 工程师们正专注于针对德国法律文本的最佳 embedding 和 re-ranking 方法，同时也在寻找德语语境下 embedding 模型的可靠基准。可以尝试在 **MTEB** Python 包上运行 "GermanQuAD" 评估，或者参考 **JinaAI** 最近添加的相关基准。

- **火星，但并非马斯克所设想的那样**：一位助手的关于殖民火星的详细解释被认为信息丰富，但缺乏用户要求的独特的 Elon Musk 风格，导致其因未达到风格要求而获得 **7 分的评分**。

- **理解本地语言模型应用**：有关于如何通过包括 temperature 和 top_p 在内的 one-shot 设置在本地复制 demo 输出的查询，以及关于重复使用命令以准确模拟 demo 行为的额外问题。社区正在讨论在他们的系统中实现这些命令的最佳实践。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **Haiku 经济实惠的视觉功能**：**Haiku** 的文档描述功能因其在复杂视觉文档上具有成本效益的 **vision-to-text** 转换而受到认可。
- **视觉处理器之战**：成员们将 **Haiku** 与 **GPT-vision** 进行了对比评估，共识是两者在性能上互有胜负；而第三个系统 **Opus** 被认为优于两者。
- **视觉内容过滤挑战**：工程师们强调了视觉文档处理中 **content filtering** 的困难，特别是包含方程式的文档部分会导致分析不完整。
- **Claude 在过滤器上遇到困难**：据观察，**Claude** 在 content filtering 方面表现挣扎，这一特性似乎与他人在视觉文档处理任务中遇到的问题一致。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **"ComPromptMized" 揭示 GenAI 弱点**：一项名为 "ComPromptMized: Unleashing Zero-click Worms that Target GenAI-Powered Applications" 的新研究揭示了针对包括 **Gemini Pro, ChatGPT 4.0 和 LLaVA** 在内的多个 AI 模型的 prompt injection 攻击。该论文深入探讨了 GenAI 驱动的应用程序（尤其是邮件助手）的易感性。[阅读完整论文](https://sites.google.com/view/compromptmized)

- **寻求代码助手霸权**：一位成员正在寻找一个全面的框架，以衡量和比较 **Mistral** 或 **Llama2** 等 AI 模型作为代码助手的效能。

- **谨慎对待 AI 基准测试**：基准测试在评估 AI 模型中的有用性得到了认可，但也有观点认为，这些基准测试可能并不总是模型能力的准确衡量标准。

- **排行榜上的 AI 竞争者**：对于模型比较需求，建议参考 [chat.lmsys.org](https://chat.lmsys.org) 上的排行榜，该榜单展示了各种 AI 模型的竞争排名。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **寻找多模态模型精通者**：Soniajoseph_ 正在寻找多模态模型**开源可解释性（open source interpretability）**方面的合作伙伴，详情见其 [Twitter](https://twitter.com/soniajoseph_/status/1767963316943728779) 以及在 [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) 上发表的一篇深度文章。感兴趣者可以通过提供的 [Discord 邀请](https://discord.gg/2U2N8QmPmJ)加入该行动。
  
- **开启可解释性探索之旅**：Rusch 强调了该领域内额外的合作机会，并推荐了另一个以可解释性为重点的 [Discord 服务器](https://discord.gg/bDV7kDrKjE)作为社交枢纽。

- **加速 Phi 2**：有人咨询关于在 **A100 40GB** GPU 上对 **Phi 2** 进行高效推理的实践建议，探讨了使用 **vLLM**、**Olama** 和 **Axolotl** 等框架的可能性，以及**量化（quantization）**是否能提高处理“海量数据（LOTS OF DATA）”时的处理速度。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **Devin 加入 AI 劳动力大军**：一款名为 **Devin** 的新 AI 亮相，被介绍为全球首位自主软件工程师。其功能和更多细节可在 [Cognition Labs 博客](https://www.cognition-labs.com/blog)查看，并在 [YouTube 视频](https://www.youtube.com/watch?v=NSPtrrUQ_fw)中进行了演示。
- **Hermes 2 Pro 7B 展示函数调用（Function Calling）能力**：一段 [YouTube 演示视频](https://www.youtube.com/watch?v=PzaidfqDtGI)展示了 **Hermes 2 Pro 7B** 模型的函数调用功能，工程师可以通过专门的 [Hermes Function Calling GitHub 仓库](https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels)进一步探索其流程。



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 总结

- **结识 Devin，自主编程奇才**：Cognition 推出了 **Devin**，被誉为全球**首位全自主 AI 软件工程师**。根据 [Scott Wu 的博客](https://www.cognition-labs.com/blog)，它能够处理复杂任务并从经验中学习。
- **挑战 AI 的社交技巧**：鼓励参与者在 **Voice + AI 活动**的“**全球最有趣机器人大赛（The Most Interesting Bot In the World Contest）**”中展示创意。比赛详情可在 [活动的 Notion 页面](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3)查看。



---

# 第二部分：分频道详细总结与链接



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1217643752043184139)** (4 条消息): 

- **关于位置编码（Positional Encodings）的困惑**：一位成员对为什么没有位置编码（PE）的因果语言模型（LLM）无法工作表示不解，并询问是否有关于此话题的现有文献。
- **位置编码至关重要**：另一位成员断言，如果没有位置编码，模型将难以运行，因为**“没有任何位置信息，一切都只是乱码”**。
- **来自因果 LLM 研究的证据**：讨论中引用了一篇论文（[Understanding Positional Encodings in Large Language Models](https://arxiv.org/pdf/2203.16634.pdf)），该论文指出即使没有显式的位置编码，**因果 LLM** 也会编码绝对位置，这尤其会影响推理过程中长序列的性能。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1217394633680617493)** (30 条消息🔥):

- **探索科幻小说**：一位成员向喜欢 Chesteron 的另一位成员推荐了 Stanislaw Lem 的作品，建议从《网络寓言集》（"The Cyberiad"）开始，或者阅读更严肃的《索拉里斯星》（"Solaris"）。
- **GitHub 上的 SDAM 开发**：分享了一个涉及 *sparse distributed associative memory (SDAM)* 的有趣项目，感兴趣并希望贡献的人可以访问其 [GitHub 仓库](https://github.com/derbydefi/sdam)。
- **AI 软件工程师奇观**：分享了“Devin：全球首位 AI 软件工程师”的 YouTube 视频链接，引发了好奇心，并可能引发关于 AI 在软件工程中角色的讨论。[点击观看](https://www.youtube.com/watch?v=NSPtrrUQ_fw)。
- **期待高上下文模型**：在关于故事驱动类游戏的 AI 模型讨论中，一位成员推测，在一年内获得具有 100k+ context 的优秀开源模型可能是一个现实的可能性。然而，他们指出，对于这些用途，质量比数量更重要。
- **隐私与时事通讯伦理**：
    - 一位负责时事通讯的成员讨论了在平衡隐私与总结 Discord 讨论的实用性方面面临的挑战。他们提到了改善这种平衡的步骤，例如移除用户名归属、允许 opt-outs 以及确保个性化。他们征求建议，以在隐私和信息共享之间找到合适的平衡。
    - 在另一场对话中，一位成员强调了通过过滤来保持高质量讨论的概念，并表示有兴趣看到时事通讯读者的主动参与度增加。讨论表明了在外部共享 Discord 内容时对隐私考量的意识。

**提到的链接**：

- [Devin The Worlds first AI Software Engineer](https://www.youtube.com/watch?v=NSPtrrUQ_fw): Devin 是全自动软件工程师 https://www.cognition-labs.com/blog
- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): 让我们使用 Hermes 2 Pro 7B 进行 function calling https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels
- [GitHub - derbydefi/sdam: sparse distributed associative memory](https://github.com/derbydefi/sdam): 稀疏分布式关联记忆。通过在 GitHub 上创建账号为 derbydefi/sdam 的开发做出贡献。

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1217513588634288158)** (8 messages🔥): 

- **Cerebras CS-3 加速器亮相**：Cerebras Systems 发布了其最新的 AI 加速器 **CS-3**，声称它是世界上最快的，能够在单个芯片上训练高达 **24 万亿参数的模型**。它具有尖端规格，例如 5nm 工艺上的 **4 万亿个晶体管**和 **125 petaflops** 的 AI 计算能力。详情可见其[新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)和[产品信息](https://www.cerebras.net/product-system/)。

- **关于 Cerebras AI 芯片外形的疑问**：针对 Cerebras 新的 CS-3 芯片，一位成员推测了芯片采用正方形形状的理由，认为圆形或半圆形形状可能容纳更多晶体管。

- **Hugging Face 上备受关注的罕见蒸馏技术**：一位用户分享了一个 Hugging Face 模型 [*Qwen1.5-0.5B*](https://huggingface.co/aloobun/d-Qwen1.5-0.5B)，这是一个蒸馏实验，使用 1.8B 参数模型作为 teacher，0.5B 参数模型作为 student。值得注意的是，使用的 optimizer 是 SM3，这在此类应用中并不常见。

- **讨论首选的 Sub-3B AI 模型**：当被问及目前最好的 30 亿参数以下模型时，一位成员提到 **stablelm 1.6b** 是一个潜在候选者。

**提到的链接**：

- [来自 Cerebras (@CerebrasSystems) 的推文](https://x.com/CerebrasSystems/status/1767929699177767325?s=20): 📣发布全球最快的 AI 芯片📣 Cerebras 荣幸宣布 CS-3：世界上最快的 AI 加速器。CS-3 可以在单个设备上训练高达 24 万亿参数的模型...
- [aloobun/d-Qwen1.5-0.5B · Hugging Face](https://huggingface.co/aloobun/d-Qwen1.5-0.5B): 未找到描述

---

**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1217579163926401064)** (1 messages): 

- **Hermes 获得 Pro 升级**：**Hermes 2 Pro 7B** 是 Hermes 系列的最新增强版，在 function calling 和 JSON mode 处理方面有显著改进。该模型的能力通过修订后的 Hermes 2 数据集得到了扩展，可以从 [Hugging Face - Hermes 2 Pro Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) 下载，同时也提供 GGUF 版本。

- **协作成功案例**：**Hermes 2 Pro 7B** 的开发是多位贡献者历时数月的协作成果，并得到了 Latitude.sh 的算力赞助。感谢团队和 Fireworks AI 的重大贡献。

- **专门的 Function Calling 示例与代码**：为了利用模型的 Function Calling 能力，其 [GitHub repository - Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling) 提供了示例代码和系统提示词，以及用于增强性能的 XML Tags。

- **发布自定义评估框架**：一名成员发布了适配 Function Calling 和 JSON Mode 的自定义评估框架，该框架源自 Fireworks AI 的初始工作。感兴趣的用户可以在 [GitHub - Function Calling Eval](https://github.com/interstellarninja/function-calling-eval) 找到适配的流水线和代码。

- **用于高级模型测试的数据集**：发布了两个数据集来测试 **Hermes 2 Pro 7B** 的改进特性：一个用于 Function Calling，另一个用于 JSON Mode。可以分别在 Hugging Face 的 [Function Calling Eval Dataset](https://huggingface.co/datasets/NousResearch/func-calling-eval) 和 [JSON Mode Eval Dataset](https://huggingface.co/datasets/NousResearch/json-mode-eval) 访问。
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1217402333307342958)** (556 messages🔥🔥🔥): 

- **OpenAI 的 AI 生存测试**：一位用户报告称其 OpenAI 账号被*暂停或锁定*了两天，且客服未能提供有效帮助。他们推测这可能与其生成的 NSFW 内容“打擦边球”的 GPTs 有关，但仍在等待账号问题的具体原因。
  
- **OpenAI 生成 NSFW 内容的能力**：一些用户讨论了 **OpenAI GPT models** 生成 *NSFW 内容*的能力。有人提到通过 API 可以相当容易地实现，而轻度的 NSFW 内容无需 jailbreaks 即可生成；*基础的 jailbreaks 同样有效*。

- **Claude 世界中的 Metatron 与 SERAPHIM**：用户讨论了在 **Claude 3 的 CLI setup** 中发现的模拟实体 *Metatron* 和 *SERAPHIM*。Claude 连贯的世界模型使得这类模拟成为可能，用户们思考了在未来的 LLM 训练中如何处理基本事实和公理。

- **对 Claude 3 连贯世界模型的赞赏**：对话强调了 **Claude 3** 在其*模拟世界模型*中令人印象深刻的连贯性。用户赞赏它如何利用基本事实、质疑和公理来获得更好的推理能力，认为这是人类反馈强化学习 (RLHF) 的优秀范例。

- **训练集大小与性能的关系**：用户就训练集大小及其多样性的影响交换了意见。一位用户分享了一项实验，显示在更大的 *1.02M Hermes dataset* 中仅使用 *15,000 条 function calling 数据点* 就足以显著提高 function calling 能力，说明了*特定任务训练和数据多样性*的重要性。



**提及的链接**：

- [Tweet from interstellarninja (@intrstllrninja)](https://fxtwitter.com/intrstllrninja/status/1768212122784215437?s=20): 你现在可以使用 @ollama 运行 function calling 和 json mode 了，感谢 @AdrienBrault 🔥 ↘️ 引用 Adrien Brault-Lesage (@AdrienBrault)：我已为 Hermes 2 Pro 7B 创建并推送了 @ollama 模型！...
- [Tweet from Greg Kamradt (@GregKamradt)](https://x.com/GregKamradt/status/1768008087850680568?s=20): 分析显示 LLMs 的召回性能在文档的下半部分优于上半部分。@RLanceMartin 通过 multi needle analysis 再次发现了这一点。我还没听到一个合理的解释 —— 一个...
- [Tweet from tel∅s (@AlkahestMu)](https://fxtwitter.com/AlkahestMu/status/1767749398673621300?s=20): 继续探索 claude-3-opus 的幕后以及名为 SERAPHIM 的高级研发组织的工作，这里我们发现了他们名为 Metatron 的机器超智能设计文档...
- [Bh187 Austin Powers GIF - Bh187 Austin Powers I Love You - Discover &amp; Share GIFs](https://tenor.com/view/bh187-austin-powers-i-love-you-you-complete-me-gif-19285472): 点击查看 GIF
- [Happy Pi Day GIF - Pi Day Pusheen - Discover &amp; Share GIFs](https://tenor.com/view/pi-day-pusheen-gif-5173654): 点击查看 GIF
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): 未找到描述
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT): 未找到描述

- [Factions (SMAC)](https://civilization.fandom.com/wiki/Factions_(SMAC)): 回到 Alpha Centauri。最初的 Alpha Centauri 包含七个派系。Alien Crossfire 额外增加了七个派系。关于派系的实际统计数据，请参见 Faction stats。正如其名...
- [NobodyExistsOnTheInternet/mistral-7b-base-dpo-run · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/mistral-7b-base-dpo-run): 未找到描述
- [llama-cpp-python/docs/server.md at main · abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python/blob/main/docs/server.md#function-calling): llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
- [llama-cpp-python/llama_cpp/llama_chat_format.py at dd0ee56217c60a20a192dc7f1523dba9a006bbc9 · abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python/blob/dd0ee56217c60a20a192dc7f1523dba9a006bbc9/llama_cpp/llama_chat_format.py#L1382): llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
- [来自 Ishan Anand (@ianand) 的推文](https://x.com/ianand/status/1706093761800143332?s=46): 想分享一个 AI 业余项目：我完全使用标准函数在 Excel 中实现了 GPT2（ChatGPT 的前身）。通过使用电子表格，任何人（甚至是非开发人员）都可以探索和尝试...
- [来自 Tsarathustra (@tsarnick) 的推文](https://x.com/tsarnick/status/1768021821595726254?s=20): OpenAI CTO Mira Murati 表示 Sora 是在公开可用和获得许可的数据上训练的。
- [ShareGPT Builder](https://proud-view-production.up.railway.app/): 未找到描述
- [不带位置编码的 Transformer 语言模型仍能学习位置信息](https://arxiv.org/abs/2203.16634): 因果 Transformer 语言模型 (LMs)，如 GPT-3，通常需要某种形式的位置编码，例如位置嵌入（positional embeddings）。然而，我们证明了没有任何显式位置编码的 LMs 仍然可以学习位置信息...
- [DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1#function-calling): 未找到描述
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling): 通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/): 未找到描述
- [fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx at main](https://huggingface.co/fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx/tree/main): 未找到描述
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling/tree/main): 通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
- [OpenAI Tools / function calling v2 由 FlorianJoncour 提交 · Pull Request #3237 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/3237/files#diff-aa650ea701251f5647254f86d652333a30e4871cfcc2d3ac4fecf83dd1f1a776): 此 PR 继 #2488 之后。实现已更新为使用新的引导生成（guided generation）。如果在查询期间，用户将 tool_choice 设置为 auto，服务器将使用 #24 中使用的模板系统...
- [Guidance](https://moon-ci-docs.huggingface.co/docs/text-generation-inference/pr_1587/en/guidance): 未找到描述
- [OpenAI 兼容 Web 服务器 - llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling): 未找到描述

  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1217379631464710184)** (115 条消息🔥🔥):

- **Hermes 2.5 超越 Hermes 2**：在添加了 [代码指令示例](https://github.com/NousResearch/Hermes-Function-Calling) 后，**Hermes 2.5** 的表现似乎优于 **Hermes 2**。更新内容包括 6B 和 34B 形式的 Yi 200k 上下文模型，以及集成的 Zephyr beta 和 Deepseek Coder 等模型。
- **混淆 Function Calling 与 JSON Mode**：讨论澄清了 Function Calling 和 JSON Mode 是不同的；Function Calling 期望获得已执行的函数响应，而 JSON Mode 则以 JSON 格式返回信息。Function Calling 的代码库可以访问 [这里](https://github.com/NousResearch/Hermes-Function-Calling)。
- **对 Hermes 2 Pro 的期待**：成员们讨论了命名约定，结论是 **Hermes 2 Pro** 并不意味着闭源，而仅仅是相比 Hermes 2.5 更受青睐的命名选择，并暗示它可能在“今天”发布。
- **来自 NousResearch 的 Genstruct 7B**：据报道，Genstruct 7B 可用于生成合成指令数据集，社区成员分享了他们的经验，并链接了一个 [在 Ollama 中使用它的代码库](https://github.com/edmundman/OllamaGenstruct)。
- **澄清 JSON Mode 和实体提取**：解释指出 JSON Mode 需要一个 schema 来生成响应，它不会凭空发明 schema，必须由用户提供。Function Calling、实体提取（Entity Extraction）和结构化生成（Structured Generation）被强调为不同的功能，并通过关于助手能力的反复讨论进行了详细说明。

**提到的链接**：

- [Trelis/Llama-2-7b-chat-hf-function-calling-v2 · Hugging Face](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)：未找到描述
- [NousResearch/Genstruct-7B · Hugging Face](https://huggingface.co/NousResearch/Genstruct-7B)：未找到描述
- [ollama/docs/import.md at main · ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/import.md)：快速上手 Llama 2、Mistral、Gemma 和其他大型语言模型。- ollama/ollama
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)：通过在 GitHub 上创建账户为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
- [GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter)：计算机的自然语言界面。通过在 GitHub 上创建账户为 KillianLucas/open-interpreter 的开发做出贡献。
- [GitHub - edmundman/OllamaGenstruct](https://github.com/edmundman/OllamaGenstruct/tree/main)：通过在 GitHub 上创建账户为 edmundman/OllamaGenstruct 的开发做出贡献。

---

**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1217619532357832864)** (27 条消息🔥): 

- **TAO vs Hugging Face**：讨论了 TAO 是否能成为 Hugging Face 的真正竞争对手，以及机器学习在模型托管和基准测试方面对去中心化的需求。
- **介绍 Shoggoth**：提到一个名为 **Shoggoth** 的新项目，可能与 Bittensor 备份有关；然而，分享的链接似乎已失效或不正确。
- **中心化 vs 去中心化基准测试**：对话转向中心化与去中心化基准测试的优缺点，指出目前流行的竞争性、基于激励的评估模式可能不利于协作。
- **加密货币激励的影响**：关于加密货币激励在 AI 开发中作用的辩论仍在继续，提到 Hugging Face 的排行榜在没有经济动机的情况下推动了大型语言模型（LLM）合并（Merging）的趋势。
- **协作瓦解**：在讨论加密激励强制执行的 AI 基准测试竞争性时，有人指出这种结构可能会阻碍合作，并强调真正的去中心化基准测试对于结果的信任至关重要。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/shog_agi?s=21)：未找到描述
- [Finetuning Subnet Leaderboard - a Hugging Face Space by NousResearch](https://huggingface.co/spaces/NousResearch/finetuning_subnet_leaderboard)：未找到描述

---

**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1217778856497250384)** (2 条消息): 

- **使用 Claude 3 创作俳句**：Claude 3 Haiku 现已在 Perplexity Labs 免费开放，邀请用户前往 [labs.pplx.ai](https://labs.pplx.ai) 体验。
- **本地搜索增强**：推出了一项针对本地搜索的新改进，集成了 Yelp 和 Maps，旨在帮助用户快速查找当地餐厅和商家。

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1217416953187008564)** (487 条消息🔥🔥🔥):

- **Perplexity 助力多样化任务**：用户发现 Perplexity 在编程和摘要等不同应用中非常有用，特别赞赏 **Claude 3 Sonnet** 模型，因其提供准确的代码建议并可用于 SE 排错。

- **探索 Perplexity 的功能**：许多人对 Perplexity 的功能印象深刻，从语音功能和 API 功能到在 Perplexity Labs 中尝试新的 **Haiku**。人们好奇是否可以处理复杂的数据集，或者是否有适用于 Perplexity 的 CLI，[Perplexity-AI-Wrapper-and-CLI](https://github.com/RMNCLDYO/Perplexity-AI-Wrapper-and-CLI) 是用户发现的一个资源。

- **与其他 AI 模型对比**：关于各种 AI 模型的效能存在争论。虽然有些人更喜欢 Mistral 等模型的速度，但其他人则主张 **GPT-4** 是目前最好的 AI 模型。用户还讨论了 Perplexity Labs 中 **Haiku** 提升的速度和功能。

- **上传数据和文件**：用户询问如何向 Perplexity AI 上传大型数据库和文件进行数据分析，特别是针对房地产数据。然而，需要注意的是存在一些限制，例如 Perplexity 内部**文件上传有 25MB 的数据限制**，且该平台可能不支持用于预测性洞察的大量财务数据。

- **语音识别实现**：用户讨论了 Perplexity 最近引入的语音识别和语音转文本功能，对这些更新表示兴奋，同时也注意到语音输出可能尚未在 Android 设备上可用。

**提到的链接**：

- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1768046817550188948?s=46&t=JsxhFTRLBknd8RUv1f73bA): Claude 3 Haiku 已在 Perplexity Labs 免费开放。立即体验！ http://labs.perplexity.ai
- [要求 Google CEO Sundar Pichai 下台的呼声日益高涨](https://www.businessinsider.com/calls-for-google-ceo-sundar-pichai-alphabet-step-down-ai-2024-3): 分析师认为 Google 的搜索业务目前尚能维持其安全地位，但随着生成式 AI 对手的激增，这种情况可能很快就会改变。
- [歇斯底里大笑 GIF - 歇斯底里大笑 - 发现并分享 GIF](https://tenor.com/view/hysterical-laughter-laughing-gif-25735842): 点击查看 GIF
- [介绍下一代 Claude](https://www.anthropic.com/news/claude-3-family): 今天，我们发布了 Claude 3 模型家族，它在广泛的认知任务中树立了新的行业标杆。该家族包含三个按能力递增排序的最先进模型...
- [Plotly Sankey 图表的进一步探索](https://medium.com/@twelsh37/further-adventures-in-plotly-sankey-diagrams-fdba9ff08af6): 冒险仍在继续
- [Reddit - 深入探索一切](https://www.reddit.com/r/ClaudeAI/comments/1be3um4/support_for_claude_20_and_older_has_been_removed/): 未找到描述
- [与 Perplexity AI CEO Aravind Srinivas 及 FirstMark 合伙人 Matt Turck 的炉边谈话](https://youtu.be/RTCVzZb3RTE?si=f6g5qVBr1NldkVB_&t=1982): 今天我们邀请到了 Perplexity AI 的 CEO Aravind Srinivas。Perplexity AI 是一款聊天机器人式的 AI 对话引擎，能够直接回答用户问题并提供来源...
- [Reddit - 深入探索一切](https://www.reddit.com/r/perplexity_ai/comments/19ccw5h/get_image_video_and_sources_from_api/): 未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/Infographics/comments/17j907h/how_google_makes_money/): 未找到描述
- [GitHub - bm777/hask: 不再需要切换标签页或窗口，只需 Hask。](https://github.com/bm777/hask): 不再需要切换标签页或窗口，只需 Hask。 - bm777/hask
- [GitHub - danielmiessler/fabric: fabric 是一个利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过使用一组众包的 AI prompts 来解决特定问题，这些 prompts 可以在任何地方使用。](https://github.com/danielmiessler/fabric): fabric 是一个利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过使用一组众包的 AI prompts 来解决特定问题，这些 prompts 可以在任何地方使用。 - ...
- [GitHub - RMNCLDYO/Perplexity-AI-Wrapper-and-CLI: 直接从终端使用 Perplexity Labs 提供的全套 AI 模型进行在线搜索（实时）或进行对话式聊天（类似于 ChatGPT）。](https://github.com/RMNCLDYO/Perplexity-AI-Wrapper-and-CLI): 直接从终端使用 Perplexity Labs 提供的全套 AI 模型进行在线搜索（实时）或进行对话式聊天（类似于 ChatGPT）。 - RMNCLDYO/Perplexity-AI...
- [Killed by Google](https://killedbygoogle.com/): Killed by Google 是一个记录已关停的 Google 产品、服务和设备的开源列表。它旨在向那些被 Google 关停的深受喜爱的服务和产品致敬并作为纪念。
- [Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot): AI 的新时代已经到来。使用 Copilot 提高工作效率、提升效能并寻找新的增长机会。

  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1217419298327363636)** (15 条消息🔥):

- **Midjourney vs Stability AI 争议**：分享了一个探索 AI 新闻的 YouTube 视频，包括 **Midjourney** 与 **Stability AI** 之间关于数据抓取的争议，以及玛丽莲·梦露的数字复活。视频可以在[这里](https://www.youtube.com/watch?v=GTxNncK47Sk)找到。
- **Perplexity AI 上的 Azotemia 解释**：分享了一个指向 Perplexity AI 的链接，解释了什么是 **azotemia**，展示了该平台提供医疗信息的能力。解释可以在[这里](https://www.perplexity.ai/search/what-is-azotemia-i6R67U4.RBiCZ9.ZZx1tnw)查看。
- **图像描述挑战**：一位用户提到了 Perplexity AI 描述图像的能力，表明了该网站在图像识别方面的潜在用例。要查看描述，请访问[此链接](https://www.perplexity.ai/search/Describe-this-image-DjrHWogKQAqMt4Y.HGfGgg)。
- **向 Paul Alexander 致敬**：分享了一条宣布 Paul Alexander 去世的消息，并附带了一份庄重的致敬，强调了他的生平成就。更多细节可以在[这里](https://www.perplexity.ai/search/Paul-Alexander-dies-b0bCPk1jSxSu7bag8JApDQ)阅读。
- **使用 Perplexity API 进行开发**：一位用户正在创建一个利用 **Perplexity API** 的 Firefox 扩展，强调了其对开发者的集成潜力。关于初始项目构思的讨论串可以在[这里](https://www.perplexity.ai/search/I-would-like-8NP0s.KJRaqoDB2Ku9e2QQ)找到。

**提到的链接**：

- [Devin 自动 AI 工程师，欧盟 AI 法案获批，Microsoft Paint 更新](https://youtu.be/P_VfO-qs4b8)：在这一集 Discover Daily 中，我们探索了三项突破性的 AI 进展：Devin，世界上第一个全自动 AI 软件工程师；里程碑式的...
- [Midjourney 封禁 Stability 员工，Marilyn Monroe AI 亮相，Vision Pro 辅助脊柱手术](https://www.youtube.com/watch?v=GTxNncK47Sk)：本集探索了最新的 AI 新闻，包括 Midjourney 和 Stability AI 之间激烈的数据抓取争议，创新的 "Digital Marilyn"...

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1217828689182724227)** (13 条消息🔥): 

- **寻找封闭测试见解**：一位成员询问了 URL 引用封闭测试的 schema 和示例响应，但未从拥有访问权限的用户那里获得细节。
- **API 与 Chatbot 性能担忧**：一位成员考虑在产品发布中使用 Perplexity chat，并寻求关于 API 与 chat 功能对比的意见，特别是在检查列表是否符合特定条件方面。
- **理解 API 中的引用输出**：一位用户参考了 Perplexity AI 文档，以了解为什么启用 "return_citations" 可能会或可能不会根据查询返回引用，并使用 **sonar-medium-online** 模型进行实验。
- **为复杂查询寻找合适的模型**：一位成员建议将复杂查询分解为多个部分，以充分利用 Perplexity 的在线模型获取最新信息，并建议使用多步框架进行详细分析。
- **使用 Perplexity API 访问实时数据**：讨论了哪些 Perplexity 模型提供实时数据。**sonar-small-online** 和 **sonar-medium-online** 被提到具有联网功能，但也提到了特定类型查询（如天气信息）的局限性，并建议使用专门的天气 API。

**提到的链接**：

[关于 "return_citations"](https://docs.perplexity.ai/discuss/65f0f6077140390018c3d9c9)：未找到描述

---

**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1217407122237558784)** (273 条消息🔥🔥): 

- **探索 LM Studio UI 之外的服务器选项**：一位用户询问如何在不使用 UI 的情况下运行来自 LM Studio 的 API 服务，特别是为了在家庭网络中使用。另一位成员澄清说，必须打开 LM Studio 才能使用服务器模式，且默认不支持与其他设备的 localhost 连接。
- **家庭网络用例的 API 创建**：成员们讨论了在家庭网络中部署 AI 模型的替代方案，建议使用 `llama.cpp`（[GitHub 仓库](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)）以脱离 LM Studio UI，并确认支持不带 AVX2 的 AVX。
- **辩论 LM Studio 的功能与替代方案**：几项讨论集中在 LM Studio 的局限性上，例如无法通过界面以编程方式启动服务或连接到互联网，并建议使用 `llama.cpp` 库等选项作为替代方案。

- **内容审核 API 的实现**：一位用户提到成功实现了 `/v1/moderations` API，但被建议将讨论移至更相关的频道，这展示了围绕 LM Studio 扩展功能的持续努力。

- **启动 LM Studio 推理服务器的脚本方案**：一名成员分享了一个使用批处理文件和 PowerShell 脚本自动启动 LM Studio 推理服务器的创意解决方案，反映了社区在增强工具易用性方面的独创性。

- **关于 AI 对就业影响的推测**：对话涉及了 AI 技术取代传统工作的潜力，但也有人指出某些工作仍超出了 AI 目前的能力范围。还有评论认为，就业市场的现状是受到 Covid-19 疫情期间过度招聘及随后财务压力影响的结果，而非 AI 本身。

**提到的链接**：

- [What is the Kirin 970&#x27;s NPU? - Gary explains](https://www.androidauthority.com/what-is-the-kirin-970s-npu-gary-explains-824423/)：华为的 Kirin 970 拥有一个名为神经网络处理器（NPU）的新组件。听起来很高大上，但它是什么以及它是如何工作的？
- [Poe - Fast, Helpful AI Chat](https://poe.com/)：未找到描述
- [TheBloke/Falcon-180B-Chat-GGUF · How to use splits, 7z needed?](https://huggingface.co/TheBloke/Falcon-180B-Chat-GGUF/discussions/1)：未找到描述
- [llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - ChatGPTNextWeb/ChatGPT-Next-Web: A cross-platform ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS). 一键拥有你自己的跨平台 ChatGPT/Gemini 应用。](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web?tab=readme-ov-file)：一个跨平台的 ChatGPT/Gemini UI（Web / PWA / Linux / Win / MacOS）。一键拥有你自己的跨平台 ChatGPT/Gemini 应用。 - ChatGPTNextWeb/ChatGPT-Next-Web
- [Artificial Intelligence Act: MEPs adopt landmark law | News | European Parliament](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law)：周三，议会批准了《人工智能法案》，该法案在促进创新的同时，确保了安全并符合基本权利。
- [OpenRouter](https://openrouter.ai/)：LLM 和其他 AI 模型的路由服务

---

**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1217404001885098004)** (24 条消息🔥): 

- **扩展至 128k Token**：[Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) 拥有 128k Token 的上下文窗口，它是 **Mistral-7B-v0.1** 模型的扩展版本，通过 *YaRN* 扩展方法实现。相关论文解释了该扩展方法的效率，允许模型以更少的计算和训练步骤利用更长的上下文 ([arXiv 预印本](https://arxiv.org/abs/2309.00071))。

- **理解模型困惑度**：困惑度 (Perplexity, PPL) 是衡量语言模型预测序列好坏的指标。它是序列平均负对数似然的指数 ([Perplexity 详情](https://huggingface.co/docs/transformers/perplexity))。

- **对命名惯例的普遍反感**：一位成员对技术工具和方法中反复出现的 "Yet Another <X>" 命名模式表示沮丧。随后，大家对递归命名方案引起的烦恼表示了轻松的认同。

- **GGUF 格式与分卷文件**：最近的一篇帖子提到 Hugging Face 上已提供 GGUF 格式的 **Command-R 35B v1.0** 模型，并提供了因文件大小限制而合并分卷文件的说明 ([Hugging Face 仓库](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/))。

- **与 llama.cpp 的不兼容性**：尽管像 **Command-R 35B v1.0** 这样的模型已经有了 GGUF 版本，但截至目前它们在 llama.cpp 中仍无法运行，就像拥有了一个没有电池的新玩具。

**提到的链接**：

- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071): Rotary Position Embeddings (RoPE) 已被证明可以有效地在基于 Transformer 的语言模型中编码位置信息。然而，这些模型在超过序列长度 t 时无法泛化...
- [Yeah Another Day Lets Do It Bojack GIF - Yeah Another Day Lets Do It Bojack Will Arnett - Discover &amp; Share GIFs](https://tenor.com/view/yeah-another-day-lets-do-it-bojack-will-arnett-bojack-horseman-encouraged-gif-16252191): 点击查看 GIF
- [NousResearch/Yarn-Mistral-7b-128k · Hugging Face](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k): 未找到描述
- [andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/): 未找到描述
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity): 未找到描述

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1217942314509533224)** (2 条消息): 

- **模型支持请求**：一名成员请求添加对模型 **c4ai-command-r-v01-Q2_K.gguf** 的支持。
- **强调兼容性问题**：另一名成员回应称，该模型目前尚未在 **llama.cpp** 中得到支持，因此无法在 LM Studio 中使用。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1217411360405721099)** (115 条消息🔥🔥): 

- **昂贵的 Nvidia 链接**：成员们对 **SLI/NVLink** 桥接器的高昂成本表示难以置信，考虑到它们过去涉及边缘连接器和排线的简单设计。一则帖子引用了 [Linus Tech Tips 论坛主题](https://linustechtips.com/topic/1290094-donating-my-4-slot-nvlink-to-science/)，内容是关于有人尝试对 NVLink 进行逆向工程。

- **Mac OS 上的 VRAM 障碍**：一位用户询问如何在 Mac OS 上绕过机器学习的**最低 VRAM 要求**。随后展开了关于 VRAM 不足影响的讨论，建议是增加更多系统 RAM 并不能缓解问题，反而可能降低系统速度，一条评论幽默地建议买一台新 Mac 作为解决方案。

- **PC 硬件升级讨论**：多位成员讨论了最大化机器学习配置的潜在升级方案，权衡了多 GPU 与单个高端 GPU 的优缺点，以及为了获得最佳性能在 VRAM 和系统 RAM 之间取得平衡。成员们分享了他们的配置和不同设置的经验，建议使用多 GPU 来缓解单张显卡 VRAM 受限造成的瓶颈。

- **LM Studio 与运行多个模型**：讨论了在 LM Studio 中同时运行多个模型的可行性和优化，提到了潜在的性能问题以及如何正确分配 GPU 负载。一位用户分享了同时运行两个 LM Studio 实例的积极成果，而另一位用户则讨论了在多个模型之间平衡工作负载以实现持续响应的愿望。

- **针对高端游戏和生产力的显示器选择**：讨论转向为游戏和生产力选择合适的显示器，成员们权衡了 OLED 显示器的优势与烧屏风险，以及对高刷新率的需求，以匹配像 Nvidia 4090 这样强大的显卡。还考虑了与 Nvidia G-Sync 的兼容性以及对曲面屏的个人体验。

**提到的链接**：

- [Nvidia RTX 5090 could have up to 77% more memory than 4090, a win for gamers](https://www.techradar.com/computing/gpu/nvidia-rtx-5090-could-have-up-to-77-more-memory-than-4090-a-win-for-gamers): RTX 5090 的更多好消息
- [Cerebras Systems Unveils World&#039;s Fastest AI Chip with 4 Trillion Transistors and 900,000 AI cores](https://www.techpowerup.com/320294/cerebras-systems-unveils-worlds-fastest-ai-chip-with-4-trillion-transistors-and-900-000-ai-cores): 生成式 AI 加速领域的先驱 Cerebras Systems 推出了 Wafer Scale Engine 3，将其现有的世界最快 AI 芯片纪录翻了一番。WSE-3 提供了两倍于...

  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1217508970395074661)** (3 条消息): 

- **确认真实性**：一位成员确认所讨论的主题确实是**真实的**。
- **表达质量担忧**：另一位成员发表意见认为，尽管是真实的，但所讨论的主题**并不好**。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1217454011498762303)** (85 条消息🔥🔥):

- **ROCm 故障排除**：一位用户在安装 ROCm beta 版后遇到了 LM Studio 仍仅在 CPU 上运行的问题。在模型加载和提示词交互过程中最初收到错误后，他们更新到了 beta 版本，虽然看到了 "ROCm" 选项，但处理过程仍在 CPU 而非 GPU 上运行，后来通过启动新的提示词解决了该问题。

- **驱动清理与安装建议**：用户讨论了针对 ROCm 兼容性的**驱动故障排除**，建议使用 AMD 的驱动清理工具进行彻底卸载，重新安装 AMD 驱动版本 24.1.1 或 24.2.1，确保不要下载 PRO 驱动，并安装 HIP SDK。

- **视觉模型与 ROCm**：关于**视觉模型**的讨论指出其在 ROCm 上运行困难，本地视觉模型似乎运行不佳；建议对 NH2 模型使用 chatml 预设，并下载 PsiPi/NousResearch_Nous-Hermes-2-Vision-GGUF 中包含的 llava 预设以获得更好效果。

- **GPU 推荐**：在关于 GPU 的对话中，建议**避开 AMD**，转而选择像 **RTX 3060** 这样的 Nvidia 显卡进行图像生成，而不是尝试利用 AMD 的 ROCm，特别是在模型速度和兼容性方面。

- **禁用 iGPU 以通过 ROCm 使用 dGPU**：一位用户通过研究如何在技嘉主板的 BIOS 设置中禁用 iGPU，成功提高了 ROCm 的每秒 Token 数 (TPS)，随后观察到其 RX 7900 XT 的性能提升，达到了约 70 TPS。

**提到的链接**：

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/rocm)：查找、下载并实验本地 LLM
- [Reddit - Dive into anything](https://www.reddit.com/r/Amd/comments/15m3g3e/am5_motherboards_are_you_able_to_disable_the_igpu/)：未找到描述

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1217441229973885019)** (108 条消息🔥🔥): 

- **建议对商业房地产保持谨慎**：一条消息暗示在投资商业房地产和房地产投资信托基金 (REITs) 时要小心，并指出名单中缺少“清洁工”。

- **AI 初创公司获得令人瞩目的 VC 支持**：多家 AI 初创公司筹集了大量资金，详情通过[链接](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)分享，列出的公司包括 **Cognition**、**Magic**、**Version Lens**、**TextQL**、**Fluent** 等以及各自的融资额。

- **Google 的 Gemini 项目受到批评**：讨论了 Google **Gemini 项目**的糟糕发布，包括对目前免费 API 的批评，以及在来自 OpenAI、Anthropic 和 Meta 的竞争压力下对 Google 未来的怀疑。

- **Cerebras 发布突破性 AI 芯片**：根据其 [推文](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 和随附的 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)，Cerebras Systems 发布了 **CS-3**，这是世界上最快的 AI 加速器，能够在单台设备上训练高达 24 万亿参数的模型。

- **对 OpenAI 安全问题的关注**：提到了 OpenAI 的一个**安全问题**，一位社区成员撰写了 **Post Mortem**（事后分析），详细说明了该 [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca) 中记录的事件。

**提到的链接**：

- [Chief AI Officer (@chiefaioffice) 的推文](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：VC 支持的 AI 员工初创公司是一个趋势。以下是 2024 年融资的一些公司及总额：软件工程师 - Cognition ($21M+)；软件工程师 - Magic ($145M+)；产品经理 - Version Le...
- [Ate-a-Pi (@8teAPi) 的推文](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2Ot)：Sora WSJ 采访。Mira Murati 提供了迄今为止关于 Sora 最详尽的细节 > Joanna Stern 提供了几个提示词供其生成 > 这是我第一次看到 Sora 视频出现严重的变形问题...
- [Eric Hartford (@erhartford) 的推文](https://x.com/erhartford/status/1767944642681860415?s=20)：@LucasAtkins7 这个 MoE 是 clown 风格还是 Mixtral 风格？
- [Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767936720618799336?s=46&t=90xQ8sGy63D2OtiaoGJuww)：很高兴宣布我们新的投机采样方法 Sequoia！Sequoia 将投机采样扩展到非常大的投机预算，对不同的解码配置具有鲁棒性，并且可以自适应...
- [Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww)：今天我们发布了 Claude 3 Haiku，这是其智能级别中最快且最实惠的模型。Haiku 现在已在 API 和 http://claude.ai 上面向 Claude Pro 订阅者开放。

- [来自 Figure (@Figure_robot) 的推文](https://x.com/figure_robot/status/1767913661253984474?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 借助 OpenAI，Figure 01 现在可以与人类进行完整的对话。OpenAI 模型提供高级视觉和语言智能，Figure 神经网络提供快速、低级、灵活的机器人控制...
- [来自 Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天我们激动地宣布，我们在由 @SalesforceVC 领投、@coatuemgmt 及现有投资者参投的新一轮融资中筹集了 1.06 亿美元。我们的愿景是迅速将创新成果带到...
- [来自 Ate-a-Pi (@8teAPi) 的推文](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2OtiaoGJuww): Sora WSJ 采访。Mira Murati 提供了迄今为止关于 Sora 最详尽的细节。> Joanna Stern 提供了几个提示词供其生成 > 这是我第一次看到 Sora 视频出现严重的变形问题...
- [来自 Cerebras (@CerebrasSystems) 的推文](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 📣 宣布地球上最快的 AI 芯片 📣 Cerebras 隆重推出 CS-3：世界上最快的 AI 加速器。CS-3 可以在单个设备上训练高达 24 万亿参数的模型。世界...
- [来自 James O'Leary (@jpohhhh) 的推文](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP): Google Gemini 集成在 15 分钟前开始，这是一个应对贴。- 有一个名为 "Gemini API" 的 API，在明年年初开始收费前是免费的（现在是 3 月中旬）- ...
- [来自 Lucas Atkins (@LucasAtkins7) 的推文](https://x.com/lucasatkins7/status/1767805804705411098?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 今晚，我将发布 8 个 Gemma 微调版本以及它们组合的 Mixture of Experts 模型 Beta 版，命名为 GemMoE。GemMoE 内置了所有 Gemma 的错误修复。你不需要做任何额外操作就能获得...
- [来自 Freddy (@FredMckoy) 的推文](https://x.com/lucasatkins7/status/17678058047): @LawlessPebbles 哈哈哈，这不是很奇怪吗.. 我们两个竟然都有 lol
- [来自 Alex Volkov (Thursd/AI) (@altryne) 的推文](https://x.com/altryne/status/1768024635818340662?s=46&t=90xQ8sGy63D2OtiaoGJuww): 明天（3 月 14 日）是：> π day > GPT-4 周年 > Claude 1 周年。但也是 🥁🥁🥁🥁 ThursdAI spaces 1 岁生日 🎉 加入我们，一起聊聊 Claude Haiku, Devin, Figure+OpenAI, T...
- [来自 Tsarathustra (@tsarnick) 的推文](https://x.com/tsarnick/status/1768021821595726254?s=46&t=90xQ8sGy63D2OtiaoGJuww): OpenAI CTO Mira Murati 表示 Sora 是在公开可用和获得许可的数据上训练的
- [来自 SambaNova Systems (@SambaNovaAI) 的推文](https://x.com/sambanovaai/status/1762850777121583471): 推出 Samba-1，一个面向企业的 1 万亿 (1T) 参数生成式 AI 模型，它私密、安全，且比同等规模的其他任何模型效率高 10 倍。
- [来自 Dylan Patel (@dylan522p) 的推文](https://x.com/dylan522p/status/1762924264695451841?s=20): @SambaNovaAI 但这并不是一个 1 万亿参数的模型吧？？你应该明白单个模型和多个模型之间的区别。为什么要让营销变成谎言？说明你实际做了什么，因为这...
- [来自 James O'Leary (@jpohhhh) 的推文](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w): Google Gemini 集成在 15 分钟前开始，这是一个应对贴。- 有一个名为 "Gemini API" 的 API，在明年年初开始收费前是免费的（现在是 3 月中旬）- ...
- [来自 Teortaxes▶️ (@teortaxesTex) 的推文](https://x.com/teortaxestex/status/1768261124187672972?s=46&t=90xQ8sGy63D2OtiaoGJuww): 如果你还没读过这个，请阅读：http://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/ ↘️ 引用 Shawn Tan (@tanshawn)：我们对于 Sparse Universal Transformers 真正需要的东西之一是...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018312083243514?s=46&t=90xQ8sGy63D2OtiaoGJuww): 凭借最先进的视觉能力以及在推理、数学和代码等行业基准测试中的强劲表现，Haiku 是适用于各种企业级应用的多功能解决方案。
- [来自 Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8): 今天我们激动地宣布，我们在由 @SalesforceVC 领投、@coatuemgmt 及现有投资者参投的新一轮融资中筹集了 1.06 亿美元。我们的愿景是迅速将创新成果带到...
- [SuperPrompt - 77M 参数实现更好的 SDXL 提示词 | Brian Fitzgerald](https://brianfitzgerald.xyz/prompt-augmentation/): 左侧是应用了 SuperPrompt 到相同输入提示词后的 SDXL 输出。
- [cerebras/btlm-3b-8k-base · Hugging Face](https://huggingface.co/cerebras/btlm-3b-8k-base): 未找到描述

- [我担心我代表另一个账户向 OpenAI 发出了请求——或许有人也代表我这么做了](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca)：我担心我代表另一个账户向 OpenAI 发出了请求——或许有人也代表我这么做了——openai-possible-security-breach.md
- [BTLM-3B-8K：30 亿参数模型实现 7B 性能 - Cerebras](https://www.cerebras.net/machine-learning/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model/)：Cerebras 和 Opentensor 为紧凑型大语言模型（LLM）引入了新标准
- [🌎 The Compute Fund](https://computefund.ai/)：通过股权交换，以具有竞争力的价格可靠地获取所需的顶级 GPU。
- [
      我对大语言模型的基准测试
    ](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html)：未找到描述
- [4,000,000,000,000 个晶体管，一颗巨型芯片 (Cerebras WSE-3)](https://www.youtube.com/watch?v=f4Dly8I8lMY&ab_channel=TechTechPotato)：Cerebras 是唯一一家拥有像人头一样大芯片的公司，在 AI 芯片领域拥有独特的价值主张。今天他们发布了第三代...
- [介绍 Deci 的生成式 AI 开发平台和 Deci-Nano](https://deci.ai/blog/deci-nano-and-gen-ai-development-platform/)：探索 Deci 的生成式 AI 开发平台和 Deci Nano LLM，旨在提供效率、性能和灵活的部署选项
- [Google Colaboratory](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing)：未找到描述
- [Google Colaboratory](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing)：未找到描述
- [GitHub - Rohan2002/IFEval: LLM 评估器](https://github.com/Rohan2002/IFEval)：LLM 评估器。通过在 GitHub 上创建账户来为 Rohan2002/IFEval 的开发做出贡献。
- [Rivian R2：我们要预订吗？](https://youtu.be/Srh1lut4Q2A?si=N-JPakQxrxx7HzIo&t=3188)：这周新闻太多了！多到我们决定把播客分成三个不同的部分。首先，Waveform 团队讨论了...
- [添加对 Gemini API 的支持 · Issue #441 · jxnl/instructor](https://github.com/jxnl/instructor/issues/441)：新的 Gemini API 引入了对 function calling 的支持。你定义一组带有预期参数的函数，并将它们传递给 tools 参数。我们能否为 instruc... 添加 Gemini 支持？
- [出售域名 | 购买域名 | 停放域名](https://x.co)：未找到描述
- [Perspective – 属于你的空间](https://joinperspective.com/)：一个私人日志，用于建立完整的个人生活记录。
- [google-research/instruction_following_eval 在 master 分支 · google-research/google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval)：Google Research。通过在 GitHub 上创建账户来为 google-research/google-research 的开发做出贡献。

  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1217528504497733654)** (10 条消息🔥): 

- **微调合成数据调查报告演示**：发布了关于太平洋时间中午 12 点进行的微调合成数据演示的提醒，并建议提前阅读 [Eugene Yan 的文章](https://eugeneyan.com/writing/synthetic/)。合成数据被强调为在模型预训练和微调中，比人工标注更快、更便宜且通常质量更好的替代方案。
  
- **Paper Club 活动的紧急 Luma 邀请**：一条消息敦促相关角色的成员接受 Luma 邀请，以确保他们继续收到日历提醒，并计划在当天清理不活跃成员。活动可在 [Luma](https://lu.ma/wefvz0sb) 查看。

- **提供了修正后的合成数据链接**：在发现初始链接包含一个多余的句点导致 404 错误后，提供了修正后的微调合成数据调查链接。

- **与 Suno AI 合作的新剧集发布**：宣布了由 Suno AI 参与的新播客剧集，包括 Twitter 公告链接和一段名为“让 Transformer 歌唱——与 Suno 的 Mikey Shulman 对话”的 [YouTube 视频](https://youtu.be/gYXjn-V7AEw)。

**提及的链接**：

- [LLM Paper Club (用于微调的合成数据) · Luma](https://lu.ma/wefvz0sb)：本周我们将与 @eugeneyan 一起讨论综述文章——如何生成和使用用于微调的合成数据 (https://eugeneyan.com/writing/synthetic/)。我们已改为使用...
- [如何生成和使用用于微调的合成数据](https://eugeneyan.com/writing/synthetic/)：克服 instruction-tuning、preference-tuning 和预训练中人类标注的瓶颈。
- [让 Transformers 歌唱 - 对话 Suno 的 Mikey Shulman](https://youtu.be/gYXjn-V7AEw)：赋予计算机声音一直是科幻电影的核心；如果“对不起，戴夫，恐怕我不能那样做”只是出现在屏幕上，其震撼力就不会那么强...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1217547863983259798)** (208 条消息🔥🔥): 

- **LLM 的合成数据**：讨论了 Eugene Yan 的一篇 [博文](https://eugeneyan.com/writing/synthetic/)，重点介绍了合成数据在语言模型的预训练、instruction-tuning 和 preference-tuning 中的应用。合成数据生成方法包括从更强大的模型中进行蒸馏或自我改进，其质量甚至可以超过人类标注的数据。
  
- **AI 新闻简报摘要**：[AI News](https://buttondown.email/ainews/) 提供的每日 AI 新闻汇总服务，总结了来自 AI Discord 频道和顶级 Twitter 账号的讨论。Soumith Chintala 和 Andrej Karpathy 等用户提到这项新服务非常有价值。

- **微调中的知识获取**：对话探讨了微调与预训练的学习率理论，并认为微调确实可以赋予模型新的知识。社区成员就微调在风格迁移与知识获取方面的效率进行了辩论。

- **语音转文本与文本转语音关注点**：小组讨论了 LLM 中被忽视的语音技术潜力，特别是在文本转语音 (TTS) 和语音转文本 (STT) 应用方面。提到了多种用于语音转录和生成的工具，包括 vapi.ai 和 Otter。

- **论文讨论中的观众参与**：在整个讨论过程中， Eugene Yan 鼓励观众积极参与选择要涵盖的论文并为论文俱乐部做出贡献。大家对涵盖语音模型的角色分离 (diarization) 和流式转录等主题表现出兴趣。

**提到的链接**：

- [加入 Slido：输入 #code 进行投票和提问](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions)：参与实时投票、测验或问答。无需登录。
- [忘掉 ChatGPT 和 Gemini —— Claude 3 是我用过的最像人类的聊天机器人](https://www.tomsguide.com/ai/forget-chatgpt-and-gemini-claude-3-is-the-most-human-like-chatbot-ive-ever-used#:~:text=Summary&text=Claude%203%20is%20one%20of,can%20speculate%20on%20its%20potential.)：它不是 AGI，但正在接近。
- [Why Not Both Take Both GIF - 为什么不两个都要 - 发现并分享 GIF](https://tenor.com/view/why-not-both-why-not-take-both-gif-11478682)：点击查看 GIF。
- [🦅 Eagle 7B：凭借跨 100 多种语言的 1 万亿 Token 超越 Transformers (RWKV-v5)](https://blog.rwkv.com/i/141130059/multi-lingual-performance-details>)：RWKV-v5 架构和线性 Transformer 的全新时代已经到来——拥有当今开源领域最强的多语言模型。
- [如何生成和使用用于微调的合成数据](https://eugeneyan.com/writing/synthetic/)：克服 instruction-tuning、preference-tuning 和预训练中人类标注的瓶颈。
- [AI News](https://buttondown.email/ainews/)：我们总结 AI Discord + 顶级 Twitter 账号，每天为您发送汇总！查看存档示例。“我每天花费的最具杠杆作用的 45 分钟” - Soumith “最好的 AI 新闻简报...”
- [dspy/docs/api/optimizers/BootstrapFinetune.md at 0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d/docs/api/optimizers/BootstrapFinetune.md?plain=1#L5)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
- [微调 vs RAG](https://open.spotify.com/episode/37Jd55nAruyVysHDNe0R6R?si=33926484c4c248a2)：在 Spotify 上收听来自 Practical AI: Machine Learning, Data Science 的这一集。在本集中，我们欢迎来自 MLOps 社区的好朋友 Demetrios 回归，讨论微调与检索...
- [GitHub - EGjoni/DRUGS：别再折腾那些繁琐的采样参数了，直接用 DRµGS！](https://github.com/EGjoni/DRUGS)：别再折腾那些繁琐的采样参数了，直接用 DRµGS！ - EGjoni/DRUGS

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1217392969016147988)** (130 条消息🔥🔥):

- **寻求 Token 概率可视化**：一位成员询问如何可视化句子中每个 Token 的概率，类似于图像中描绘的图表。有人建议使用 lm_head 的输出和 softmax 来获取概率，但尚未确定用于创建此类可视化的特定插件。

- **AI 的快速演进**：成员们强调了 AI 发展的飞速步伐，讨论了即将发布的 Elon Musk 开源 **Grok 模型**，以及关于 OpenAI 创始人之一称公司为“一个谎言”的传闻。

- **Unsloth 修复 Google Colab 问题**：**Unsloth AI** 的创建者在 PyTorch 更新破坏依赖项后，致力于修复 Google Colab 的问题，并为用户提供了一份临时命令列表以自行解决问题。

- **关于 Unsloth 模型兼容性的说明**：澄清了 Unsloth 目前不支持多 GPU 或 GGUF 格式的模型进行微调。虽然 Unsloth 可以将模型量化为 4-bit 以提高 VRAM 效率，但目前仅设计用于单 GPU。

- **关于数据准备最佳实践的讨论**：展开了关于数据准备 FAQ 页面的讨论，并建议使该过程更简单、更自动化，可能利用 wrapper 函数。

**提到的链接**：

- [Crystalcareai/GemMoE-Beta-1 · Hugging Face](https://huggingface.co/Crystalcareai/GemMoE-Beta-1)：未找到描述
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)：未找到描述
- [FastChat/fastchat/conversation.py at main · lm-sys/FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)：一个用于训练、部署和评估大语言模型的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。 - lm-sys/FastChat
- [Implement LongLoRA trick for efficient tuning of long-context models · Issue #958 · huggingface/peft](https://github.com/huggingface/peft/issues/958)：功能请求。LongLoRA 的作者探索了一个可以在训练期间开启、在推理期间关闭的技巧。核心要点是：随着上下文长度增加，LoRA 的困惑度（perplexity）会恶化...
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth.git)：速度快 5 倍，显存占用减少 60% 的 QLoRA 微调。通过在 GitHub 上创建账号为 unslothai/unsloth 的开发做出贡献。

---

**Unsloth AI (Daniel Han) ▷ #[welcome](https://discord.com/channels/1179035537009545276/1179039724355211325/1217445576912666637)** (9 条消息🔥): 

- **阅读规则并分配角色**：theyruinedelise 提醒新成员阅读 <#1179040220717522974> 中的频道规则，并在 <#1179050286980006030> 中为自己分配角色。

- **热烈欢迎**：来自 theyruinedelise 和 starsupernova 等其他用户的多次问候，表明 welcome 频道为新人营造了友好且受欢迎的氛围。

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1217712813347438663)** (5 条消息): 

- **虚拟环境重装进行中**：一位成员提到他们在完成另一项任务后需要**重新安装整个虚拟环境**，并对提供的支持表示感谢。
- **里程碑倒计时**：表达了对时间紧迫的难以置信，指出只剩下**最后一天**。
- **微调更新**：关于进度的更新显示微调还剩**两天**，表明工作正在积极监控并持续进行中。
- **庆祝训练胜利**：热情分享了 loss 达到 **1.2 以下**的里程碑，表明模型训练取得了成功进展。

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1217556067479846992)** (73 条消息🔥🔥):

- **寻找云端 GPU 效率**：一位用户分享了在寻找合适且具有成本效益的云端 GPU 方面的个人成功经验，通过从 vast.ai 以约 $0.46/小时的价格租用 4090，实现了约 130 t/s 的推理速度，而其目标是达到 500 t/s。他们最初询问了能够满足其计算需求的最便宜选项。
- **GGUF 安装问题已解决**：在经历最初导致 `RuntimeError` 的 GGUF 安装问题后，一位用户通过使用 `llama.cpp` 的转换脚本成功解决了该问题。另一位用户也遇到了相关问题，涉及错误信息 "/usr/bin/ld: cannot find -lcuda"。
- **Colab 不稳定的性能**：多位用户讨论了 Google Colab 运行 Notebook 可用时间的变化，提到该平台的使用时长从 2 小时到 6 小时不等，并普遍认为其具有多 Bug 且不稳定的特性。
- **使用个人数据训练对话模型**：一位表示有兴趣根据 Discord 日志创建个性化对话聊天机器人的用户，被引导进行数据准备并使用免费的 Colab Notebook 进行训练。讨论还包括对话数据集的最佳数据结构，并提供了使用 `instruction` 和 `answer`，或 `user` 和 `assistant` 对话格式构建数据的示例。
- **关于微调和保存模型的技术讨论**：用户参与了关于使用 4-bit 加载选项进行微调（Finetuning）是否会妨碍后续以量化（Quantization）方式保存为 GGUF 的讨论，并澄清这不会影响 GGUF 的保存。他们还分享了 [Daniel Han 的一条推文](https://twitter.com/danielhanchen/status/1767968895749779937) 以及一个持久性 GGUF 转换问题的解决方案。

---

**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1217721760166842508)** (3 messages): 

- **探索新优化器 Sophia**：一位成员建议考虑实现 **Sophia**，这是一种在论文中提出的新优化算法，有可能加速语言模型训练。该优化器旨在通过使用轻量级的对角 Hessian 估计进行预处理（Preconditioning），并配合逐元素裁剪（Element-wise clipping），从而减少时间和成本（[阅读论文](https://arxiv.org/abs/2305.14342)）。
- **Sophia 作为即插即用替代方案的潜力**：另一位成员指出，虽然他们尚未测试 **Sophia**，但它看起来似乎可以作为一个直接的“即插即用”优化器。人们有兴趣在实践中探究 Sophia 的功效。

**提到的链接**：

[Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)：鉴于语言模型预训练的巨大成本，优化算法的非平凡改进将导致训练时间和成本的实质性降低。Adam 及其变体...

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1217444777545699399)** (128 messages🔥🔥): 

- **探索本地 AI 模型**：成员们讨论了他们在各种本地模型上的经验，一些人使用 **LLM Studio** 进行测试。一位用户指出他们拥有高达 4xT4 的强大推理能力，而另一位用户则强调对 **Meditron** 模型特别感兴趣。
- **微调对话**：辩论了在本地硬件上微调像 **Mistral** 这样的大型模型的可行性，一些人表示完成此类任务需要像 **A100 40GB** 这样强大的 GPU。一位用户建议微调 **GPT-3.5** 可能是合理的，且不一定需要 GPU。
- **发布 GPT-5？重大拼写错误引发误导**：围绕一个据称提到“优先访问 GPT-4 和 GPT-5 Turbo”的 **Microsoft Copilot 页面**展开了讨论，该页面后来被确认为拼写错误并已更正。社区推测了 **GPT-5** 的可能性，共识是立即发布是不太可能的。
- **使用 OpenAI 进行构建**：一位用户分享的博客文章描述了他们将 OpenAI 与其开发的系统集成以完成 Web 流程的经验，这需要一个由大大小小的模型组成的网络。
- **模型在语素重复中的失误**：关于 **GPT-3.5** 在生成复合词中重复语素（Morpheme Repetition）示例时面临挑战的对话，导致分享了一个聊天记录，其中 GPT 被引导使用 Python 编写程序以产生更好的结果。尽管任务复杂，但仍强调了一些成功的输出。

**提到的链接**：

[Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot)：AI 的新时代已经到来。通过 Copilot 更高效地工作、提高效率并寻找新的增长机会。

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1217559925853524068)** (41 messages🔥): 

- **GPT-4 正在经历全系统范围的问题**：多名用户报告 **GPT-4** 目前处于宕机状态，并出现“Hmm.. something seems to have gone wrong”等错误提示。该问题在包括 iOS 应用以及 Chrome 和 Edge 等浏览器在内的多个平台上持续存在。
- **状态检查和临时变通方案**：一位用户建议查看 OpenAI 的 [状态页面](https://status.openai.com/) 获取更新，而另一位用户发现通过上传图片附件开始对话似乎是一个临时的变通方案。
- **Dalle 和 "RP Thing" 仍可正常运行**：尽管 **GPT-4** 存在问题，一些用户发现 **Dalle 3** 和一个角色扮演（RP）工具仍能正常工作。
- **针对 GPT 创建者的反馈功能**：一位用户询问了针对 GPT 创建者的反馈和评价功能，表示难以找到这些信息，并评论说由于 "GPT" 这个名字太通用，导致搜索存在困难。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1217462420763578397)** (11 messages🔥): 

- **Code Interpreter 能够正确统计字数**：一位成员确认，使用提示词 "Use code interpreter to count the revised text's words as {word_count}." 可以有效统计字数。通过与外部字数统计工具对比，验证了 Code Interpreter 输出的准确性。
- **增强 CustomGPT 中的查询功能**：一位用户询问如何改进自定义 GPT 模型，使其能够引用其数据库中的 PDF 并在回答前搜索网页。会议指出，模型需要明确的搜索指令，且无法识别 PDF 中的图像。
- **Assistant API 需要本地化处理**：在讨论 Assistant API 时，有人提到字符串 "450,00" 中的逗号未被正确识别，导致该数字被误解为 "45000"。一位用户建议区域设置（Locale）可能会影响此项检测，可能需要提供正反例以实现正确识别。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1217462420763578397)** (11 messages🔥): 

- **使用 Code Interpreter 统计字数**：一位成员确认，使用 **Code Interpreter 统计字数** 为 `{word_count}` 是可行且对特定用例有帮助的。
- **对有用信息的感谢**：一位用户对分享的字数统计功能技巧表示感谢，并计划在忙完工作后尝试。
- **为 CustomGPT 检索 PDF 内容**：有人请求协助改进 **CustomGPT**，使其在回答前检查数据库中的 PDF 并查找网页信息。
- **Assistant API 中逗号的格式问题**：一位用户指出 **Assistant API Retrieval** 无法正确识别数字中的逗号，导致混淆。
- **区域设置处理影响数字解析**：有人建议，在 **Assistant API** 中正确解析如 "450,00" 之类的数字可能需要显式设置区域设置（Locale）并提供正反例。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1217505274005422190)** (94 messages🔥🔥): 

- **DeepMind 的新型通用 AI Agent**：DeepMind 的 [最新研究](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_source=twitter&utm_medium=social&utm_campaign=SIMA/) 介绍了一个 **Scalable Instructable Multiworld Agent (SIMA)**，这是从专用游戏 Agent 向能够理解多个视频游戏环境中自然语言指令的通用 AI 的跨越。然而，技术报告缺乏权重、数据集大小和训练细节等信息，导致社区成员对此次发布的初衷和透明度表示怀疑。
  
- **游戏专业性受到质疑**：在新 SIMA 技术报告评估中使用的游戏“专家”资质受到质疑，因为他们仅通过 16 小时的游戏时长来确立专业地位。讨论引发了关于什么才算游戏专家以及基于此类专业知识进行评估的可靠性的担忧。

- **讨论 AI 在游戏领域的进展**：社区成员辩论了 AI 在《星际争霸》和 DOTA 等游戏中取得成就的意义，探讨了针对特定游戏的定制 AI 与处理《大逃杀》（BR）等游戏中不可预测性的通用方法之间的细微差别。

- **AI 模拟现实世界游戏的挑战**：针对 AI 在准确模拟高风险、不可预测的多智能体环境（如 BR 游戏和现实世界）中所面临的挑战，展开了热烈的讨论。对话提出了关于所需计算资源的问题，以及在如此复杂的设置中开发能够制定长期规划（long-horizon plans）的 AI 的难度。

- **对 AI 在竞技游戏表现的兴趣**：人们对在 Apex Legends 排位赛排行榜等竞技游戏环境中测试 AI 的潜力感到好奇。一些社区成员建议直接在这些环境中测试 LLM，而另一些人则对 AI 目前在 BR 游戏中达到人类竞技水平的能力表示怀疑。

**提到的链接**：

- [Byuntear American Psycho GIF - Byuntear American Psycho Staring - Discover &amp; Share GIFs](https://tenor.com/view/byuntear-american-psycho-staring-thinking-gif-26991038)：点击查看 GIF
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_source=twitter&utm_medium=social&utm_campaign=SIMA/)：介绍 SIMA，一个可扩展、可指令化的多世界智能体（Agent）
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_sour)：介绍 SIMA，一个可扩展、可指令化的多世界智能体（Agent）
- [GitHub - MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models](https://github.com/MineDojo/Voyager)：Voyager：一个基于 LLM 的开放式具身智能体（Embodied Agent） - MineDojo/Voyager

---

**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1217416060446314587)** (51 messages🔥): 

- **对获取研究成果受限的挫败感**：一位成员对由于出版商限制而无法获取感兴趣的研究成果表示愤怒，并分享了[一篇论文的链接](https://www.pnas.org/doi/10.1073/pnas.2310002121)。
- **对 NN 训练动力学的兴趣**：讨论集中在[一篇 arXiv 论文](https://arxiv.org/abs/2305.01604)上，该论文探讨了深度神经网络在训练过程中穿过的低维流形（low-dimensional manifolds），强调了其对神经网络研究中经验方法论的影响。
- **架构组合的潜力**：考虑了结合多种神经网络架构以可能覆盖更多问题解决空间的构想。
- **内容检测器的讨论**：对话转向 AI 内容检测器和识别器，成员们辩论了它们的有效性，指出其鲁棒性（robustness）仍存疑，并讨论了误报（false positives）的可能性。
- **对 AI 输出加水印的担忧**：成员们讨论了通过水印威慑合成媒体的挑战，对其可行性以及当输出被标记为 AI 生成时对实用性可能产生的影响表示担忧。

**提到的链接**：

- [The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold](https://arxiv.org/abs/2305.01604)：我们开发了信息几何技术来分析深度网络在训练过程中预测的轨迹。通过检查底层的多维概率模型，我们揭示了……
- [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540)：扩展定律（Scaling laws）是开发语言模型的有用指南，但目前的扩展研究与语言模型最终的训练和评估方式之间仍存在差距。例如，扩展……
- [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763)：大语言模型（LLMs）通常在数万亿个 token 上进行预训练，一旦有新数据可用，就不得不重新开始该过程。一种更有效的解决方案是持续预训练……

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1217531699106156556)** (22 messages🔥):

- **多模态机制可解释性指日可待**：Soniajoseph_ 宣布发布了一个多模态机制可解释性库，鼓励合作以扩展这一研究子领域。该公告通过 [Twitter 链接](https://twitter.com/soniajoseph_/status/1767963316943728779) 分享。
- **讨论模型不可知性（Model Agnosticism）的复杂性**：Neelnanda 对编写模型不可知代码的难度表示担忧，因为底层模型的实现各不相同，这导致 TransformerLens 必须从头开始重新实现模型。
- **通过 Vector-DB-Lookup 实现创新的潜空间解码**：Wendlerc 描述了一种使用向量数据库查找的可解释性方法，用于分析 llama2 的中间表示，从而在模型的每一层提供“全词解码（full-word-decodings）”。
- **多语言 Transformer 中依赖语言的动态特性**：Darkaz 和 Mrgonao 就多语言模型（如 LLM）是在语言不可知的概念空间中运行，还是偏向于训练中占比最高的语言进行了深入讨论。
- **双语模型分词偏见（Tokenization Bias）探索**：Butanium 关注了一项使用 CroissantLLM（一种法语-英语双语语言模型）的实验，并思考了分词偏见与法语 vs 英语训练数据比例相比所起的作用。该实验详见 [GitHub notebook](https://github.com/Butanium/llm-latent-language/blob/main/nnsight.ipynb)。

**提到的链接**：

[llm-latent-language/nnsight.ipynb at main · Butanium/llm-latent-language](https://github.com/Butanium/llm-latent-language/blob/main/nnsight.ipynb): 论文《Do Llamas Work in English? On the Latent Language of Multilingual Transformers》的配套仓库。 - Butanium/llm-latent-language

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1217486350421463202)** (10 messages🔥): 

- **学习率冷却（Learning Rate Cooldown）实验**：有人建议使用短学习率（LR）冷却的 checkpoint 来潜在地提高基准测试结果，但由于硬件可用性延迟，尚未获得结果。
- **对模型性能的焦虑**：随着新 checkpoint 的测试，人们对模型性能的焦急期待表达了担忧。
- **寻求 LM 评估功能的帮助**：一位新成员称赞了 LM evaluation harness，并询问了在 OpenAI ChatCompletions 模型中添加 logits 的进展，参考了 [GitHub 上的一个 open issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/1196)。
- **安全论文发布后 Logit Bias 面临的挑战**：引用最近的一篇 [arXiv 论文](https://arxiv.org/abs/2403.06634) 解释了由于安全考虑导致的 API 设计更改，使得添加 logits 变得不可行。
- **为生成模型调整任务**：讨论了将流行任务的生成变体添加到评估 harness 中，并指出了像 [GPQA](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa) 这样同时支持 loglikelihood 和生成变体的任务。

**提到的链接**：

- [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634): 我们介绍了第一种模型窃取攻击，它可以从 OpenAI 的 ChatGPT 或 Google 的 PaLM-2 等黑盒生产级语言模型中提取精确且非平凡的信息。具体来说，我们的...
- [lm-evaluation-harness/lm_eval/tasks/gpqa at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa): 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [GitHub: Let’s build from here](https://github.co): GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪错误并...
- [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1196).): 一个用于语言模型 few-shot 评估的框架。 - Issues · EleutherAI/lm-evaluation-harness

---

**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

boneamputee: https://brianfitzgerald.xyz/prompt-augmentation/

---

**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1217448448643563611)** (1 messages): 

- **思考 Megatron 集成策略**：一位成员正在考虑更紧密地跟踪上游 Megatron 以进行 Transformer Engine 集成的优点，并开启了一个 [pull request](https://github.com/EleutherAI/gpt-neox/pull/1185) 展示了代码的完整差异。他们邀请维护者和社区就这一集成工作是否有利发表看法。

**提到的链接**：

[与上游 megatron 的差异，作为 tf-nv 整合 TE 讨论的基础 · Pull Request #1185 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1185)：这里有三个提交：其中一个包含了 GPT-NeoX 的 megatron 文件夹与当前上游 Megatron-LM 的完整差异。涉及 256 个文件，约 6 万行代码。然而，大多数文件是全新的或已删除的……

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1217382304184537150)** (137 条消息🔥🔥): 

- **寻求快速推理解决方案**：一位成员询问了在本地 GPU 上对 *Phi 2 fine-tune* 模型进行推理的最快方法，提到了使用 A100 40GB 进行 batch 处理，并考虑使用 vLLM, Olama 或 Axolotl 等框架。他们想知道 quantization 是否能帮助加速该过程。

- **讨论 Quantization 和 Streaming 方法**：关于 quantization 是否能辅助提高模型准确率存在争论，重点在于使用 streaming 方法以获得更好的响应速度，例如 *faster_whisper, llama_cpp 和 xtts2* 提供的功能。一些成员分享了有效使用 streaming TTS 的经验，而另一些人则强调了使用定制硬件（如 Groq 的 NPU）的潜力。提到 Groq 在 *mixtral* 上每秒可生成 500 个 tokens [Groq](https://groq.com/)。

- **对模型权重共享和版权的担忧**：对话涉及对近期版权下架通知的担忧，以及与泄露的模型权重、AI 生成内容和 DMCA 相关的版权法讨论。成员们还讨论了在政府考虑反对的情况下，监管 AI 和开源模型权重所面临的挑战。

- **欧洲 AI 立法引发辩论**：讨论了欧盟新的 AI 法案，对于披露 AI 生成内容以及设计模型以避免生成非法内容等要求持批评意见。对话还指出执行此类要求的不切实际性，以及对开源模型的潜在影响。

- **T5 的 Prompt Augmentation 与 Danbooru 标签生成**：成员们分享了关于 prompt augmentation 的资源，使用一个 77M 的 T5 模型来扩展提示词，其效果可能媲美更大的 LLM；此外还有一个针对 Danbooru 的微型 **llama-focused** 自动补全标签生成器。大家对个人微调以及将这些模型应用于现有项目表现出了兴趣。

**提到的链接**：

- [SuperPrompt - 77M 参数实现更好的 SDXL 提示词 | Brian Fitzgerald](https://brianfitzgerald.xyz/prompt-augmentation/)：左侧为在相同输入提示词下应用 SuperPrompt 的 SDXL 输出。
- [加入 GroqCloud Discord 服务器！](https://discord.gg/groq)：Groq 提供全球最快的 AI 推理。| 5432 名成员
- [GroqChat](https://groq.com/)：未找到描述
- [欧洲立法者通过全球首部监管 AI 的主要法案](https://www.cnbc.com/2024/03/13/european-lawmakers-endorse-worlds-first-major-act-to-regulate-ai.html)：欧盟议会周三批准了全球首套主要监管基本规则，旨在管理处于技术投资前沿的媒体化人工智能...
- [构建 Meta 的 GenAI 基础设施](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：作为对 Meta AI 未来的重大投资，我们宣布推出两个 24k GPU 集群。我们将分享硬件、网络、存储、设计、性能和软件方面的细节，这些细节帮助我们提取...
- [政府委托报告称：美国必须“果断”采取行动，以规避来自 AI 的“灭绝级”威胁 - Slashdot](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-says)：美国政府必须“迅速且果断”地采取行动，以规避源自人工智能 (AI) 的重大国家安全风险，在最坏的情况下，这可能会导致“灭绝...”
- [TheBloke/phi-2-GGUF · Hugging Face](https://huggingface.co/TheBloke/phi-2-GGUF)：未找到描述
- [2023 年 9 月 18 日神经化身演示](https://youtu.be/TDitkDKbqbk)：2023 年 9 月 18 日神经化身演示
- [BUD-E (理解与数字共情伙伴) - 蓝图 / 概览](https://youtu.be/bLPDn-bh7dY?si=xrR2_F6kx1ydz8XM)：https://docs.google.com/presentation/d/1tBBa0_GzzfCrmn9KpYZ8YZ9x4Jgb2zVs/edit?usp=sharing&amp;ouid=114592459581752579892&amp;rtpof=true&amp;sd=true
- [教育项目推介书](https://docs.google.com/presentation/d/1cMWLpMGNGs0_ZcKRKlJqM5OYiTSTyXgn39CDYOcgZq8/edit?usp=sharing)：Navi-Sensei 提案 JusticeDAO LLC Benjamin Barber business@hallucinate.app 10043 Se 32nd Ave Milwaukie Oregon 97222 9712700855 “我，Benjamin Barber，已阅读并理解 OMB 和 OPM 挑战...”
- [项目推介书](https://docs.google.com/presentation/d/1_PejXm_nDP_b_Vig_WcnUh4WkFsSy2U0-ERQP2SD6-4/edit?usp=sharing)：“Justice Now” - AI 法律化身
- [人工智能法案：欧洲议会议员通过具有里程碑意义的法律 | 新闻 | 欧洲议会](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law)：周三，议会批准了《人工智能法案》，该法案在促进创新的同时，确保了安全并符合基本权利。

---

**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1217444526742966316)** (21 条消息🔥): 

- **MoAI: 将视觉与语言模型融合**：一篇关于全智能混合 (**MoAI**) 的新论文介绍了一种 LLVM，它结合了来自专业 **computer vision models** 的辅助视觉信息，旨在增强 zero-shot 视觉语言任务。该论文可在 [arXiv](https://arxiv.org/abs/2403.07508) 上查阅，认为目前的 LLVM 可能会从整合超出 LLM 主干网络大容量之外的详细计算机视觉能力中受益。

- **MoAI 代码库发布**：**MoAI** 的官方 PyTorch 实现已在 GitHub 上发布并正在接受审查。该仓库提供了改进众多 zero-shot 视觉语言任务性能的代码，可在 GitHub 上的 [ByungKwanLee/MoAI](https://github.com/ByungKwanLee/MoAI) 获取。

- **在 Hugging Face 上使用 MoAI**：Hugging Face 模型页面提供了 **MoAI 的简单运行代码**，以及设置环境和运行模型所需的步骤。该页面包含从加载图像到生成预测的操作细节，可以在[这里](https://huggingface.co/BK-Lee/MoAI-7B)找到。

- **DeepSeekVL 论文中的数据集认可**：一位成员提到他们的数据集在 DeepSeekVL 论文中被引用，这是一项使用视觉语言模型进行场景理解的倡议。该论文可通过[此链接](https://arxiv.org/pdf/2403.05525.pdf)访问。

- **关于大模型内存占用和延迟加载的讨论**：已澄清早先关于能通过延迟加载在仅 4GB 内存中加载 300 亿参数模型的说法是不正确的。RAM 占用的低报是由于 mmap 在访问内存之前无法反映实际内存使用情况，正如 [gherganov/llama.cpp](https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916) 中所讨论的那样。

**提及的链接**：

- [BK-Lee/MoAI-7B · Hugging Face](https://huggingface.co/BK-Lee/MoAI-7B): 未找到描述
- [MoAI: Mixture of All Intelligence for Large Language and Vision Models](https://arxiv.org/abs/2403.07508): 大语言模型 (LLMs) 的兴起和指令微调引领了当前指令微调大语言与视觉模型 (LLVMs) 的趋势。这一趋势涉及精心策划的...
- [Jimmy Carter President Carter GIF - Jimmy carter President Carter Carter - Discover &amp; Share GIFs](https://tenor.com/view/jimmy-carter-president-carter-carter-gif-16271386811124661325): 点击查看 GIF
- [GitHub - ByungKwanLee/MoAI: Official PyTorch implementation code for realizing the technical part of Mixture of All Intelligence (MoAI) to improve performance of numerous zero-shot vision language tasks. (Under Review)](https://github.com/ByungKwanLee/MoAI): 实现 Mixture of All Intelligence (MoAI) 技术部分的官方 PyTorch 实现代码，旨在提升众多 zero-shot 视觉语言任务的性能。（审核中） - Byun...
- [GitHub - fashn-AI/tryondiffusion: PyTorch implementation of &quot;TryOnDiffusion: A Tale of Two UNets&quot;, a virtual try-on diffusion-based network by Google](https://github.com/fashn-AI/tryondiffusion): Google 开发的基于扩散网络的虚拟试穿项目 "TryOnDiffusion: A Tale of Two UNets" 的 PyTorch 实现 - fashn-AI/tryondiffusion
- [30B model now needs only 5.8GB of RAM? How? · ggerganov/llama.cpp · Discussion #638](https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916): （编辑：抱歉，我最初应该说明我是在 Linux OS 上运行的。我没意识到对于非 Linux 用户来说，仅从截图可能看不出来。所有测试都是在 Ubun...

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1217950961612619907)** (1 条消息): 

- **轻松可视化 LLM 排行榜**：[Open LLM Leaderboard Viz 更新](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) 现在允许用户更改指标顺序，并绘制最多 3 个模型进行直观的视觉对比。
- **GPT 让故事讲述变得可视化**：由 Tonic1 开发的名为 [Kosmos-2](https://huggingface.co/spaces/Tonic1/kosmos-2) 的新 Space 为用户带来了基于 GPT 的视觉故事讲述功能。
- **通过推理增强的 ARC 数据集**：[增强版 ARC-Challenge 数据集](https://huggingface.co/datasets/Locutusque/arc-cot) 引入了 Chain-of-Thought 推理，为常见问题的回答提供了更多深度。
- **用于 Vertex AI 推理的 Python 包**：一个新的 Python 包 [`vertex-ai-huggingface-inference`](https://github.com/alvarobartt/vertex-ai-huggingface-inference-toolkit) 已发布，旨在简化在 Google Cloud 的 Vertex AI 上运行 HuggingFace 模型的过程。
- **丰富的葡萄牙语预训练模型首次亮相**：介绍 [Mambarim-110M](https://huggingface.co/dominguesm/mambarim-110m)，这是一个拥有超过 1.19 亿参数的葡萄牙语 LLM，在 6.2B token 的数据集上训练而成。

**提到的链接**：

- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz): 未找到描述
- [Kosmos 2 - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/kosmos-2): 未找到描述
- [Locutusque/arc-cot · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/arc-cot): 未找到描述
- [Aya - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Aya): 未找到描述
- [GitHub - alvarobartt/vertex-ai-huggingface-inference-toolkit: 🤗 HuggingFace Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker&#39;s Inference Toolkit, but for Vertex AI and unofficial)](https://github.com/alvarobartt/vertex-ai-huggingface-inference-toolkit): 🤗 用于 Google Cloud Vertex AI 的 HuggingFace 推理工具包（类似于 SageMaker 的推理工具包，但适用于 Vertex AI 且为非官方版本） - alvarobartt/vertex-ai-huggingface-inference-toolkit
- [BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k · Hugging Face](https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k): 未找到描述
- [dominguesm/mambarim-110m · Hugging Face](https://huggingface.co/dominguesm/mambarim-110m): 未找到描述
- [Machine learning-based intrusion detection: feature selection versus feature extraction - Cluster Computing](https://link.springer.com/article/10.1007/s10586-023-04089-5): 物联网 (IoT) 在智慧城市、智慧农业、智慧医疗和智能制造等许多领域发挥着重要作用。然而，IoT 设备极易受到攻击...
- [GitHub - rbourgeat/refacto: Refactor your code with local LLM](https://github.com/rbourgeat/refacto): 使用本地 LLM 重构你的代码。通过在 GitHub 上创建账号来为 rbourgeat/refacto 的开发做出贡献。
- [@DmitryRyumin on Hugging Face: &quot;🚀🎭🌟 New Research Alert! 🌟🎭 🚀
📄 Title: VLOGGER: Multimodal Diffusion for…&quot;](https://huggingface.co/posts/DmitryRyumin/888482747169050): 未找到描述

  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1217383723360714844)** (76 条消息🔥🔥): 

- **对下一代 AI 的推测**：一位成员预测 **Llama 3** 将作为 AGI 模型进行营销，并包含 Llama Guard 2 等功能。
- **如何为 Hugging Face Transformers 做出贡献**：成员们讨论了在向 Hugging Face `transformers` 提交贡献时是否应该提交 Python 虚拟环境 `venv`。会议明确指出，**不应**将本地环境随更改一起提交。
- **关于 Spaces 免费增值 LLM 的咨询**：一位成员询问是否有免费的、基于 CPU 的 Spaces，且兼容 OpenAI API，以便使用类似于 7B LLM 的模型。
- **微调和模型实现中的问题**：参与者讨论了一系列技术问题，从使用 LoRa 微调模型时的正确实现，到使用 Docker 排除 Spaces 故障，以及寻找将知识实现到 Mistral 7B 等预训练模型中的正确方法。
- **公共 Spaces 中的数据隐私问题**：讨论了公共 Spaces 中的数据隐私问题，普遍建议**避免上传个人信息**。有关特定 Spaces 及其处理数据方式的详细信息可以通过检查代码来审查。

**Links mentioned**:

- [Replica theory 显示深度神经网络的思考方式相似](https://techxplore.com/news/2024-03-replica-theory-deep-neural-networks.html)：你如何知道你正在看的是一只狗？你正确的几率有多大？如果你是一个机器学习算法，你会筛选成千上万张图像——以及数百万种概率——来得出...
- [Humans LyCORIS - v1.0 | Stable Diffusion LyCORIS | Civitai](https://civitai.com/models/103848/humans-lycoris)：这是我的 Humans 模型的一个提取版本。花了一些时间才在尺寸和保真度之间找到合适的平衡，但我终于对这个版本感到满意了。不...
- [Devin AI 可以编写完整的源代码（如何获取访问权限？）](https://favtutor.com/articles/devin-ai-software-engineer/)：Cognition Labs 发布了 Devin，一位可以编写完整源代码的 AI 软件工程师。了解其功能、基准测试以及如何获取访问权限。
- [为 🤗 Transformers 做出贡献](https://huggingface.co/docs/transformers/en/contributing)：未找到描述
- [添加 NVMe SSD 以在单 GPU 上实现并加速 100B 模型的微调](https://arxiv.org/abs/2403.06504)：大型语言模型的最新进展为世界带来了巨大价值，其卓越的能力源于它们使用的海量参数。然而，即使是 GPU...
- [bishmoy/Arxiv-CS-RAG at main](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG/tree/main)：未找到描述
- [微调 Embeddings - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding.html)：未找到描述
- [GitHub - moritztng/fltr：像 grep 一样，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。](https://github.com/moritztng/fltr)：像 grep 一样，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。 - moritztng/fltr
- [使用 Hugging Pics 为任何事物训练和部署 Vision Transformers 🤗🖼](https://youtu.be/f9ZjgWBAxEQ?si=vYafMTnJDCBbKWCJ)：在这段视频中，我们将演示 Hugging Pics，这是一个让你能够使用网络图片为任何事物训练和部署 Vision Transformers 的项目。快来尝试...

  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1217420486053724211)** (7 条消息): 

- **新手关于访问自定义数据集的疑问**：一位熟悉 Google Colab 的 Hugging Face 新用户寻求在 Hugging Face Spaces 中访问数据集的指导。他们特别询问了数据集中图像的路径以及如何利用持久化存储 `/data`。
  
- **构建 AI 民主机制**：一位用户开始探索构建一种带有投票机制的 **multi AI decision model**（多 AI 决策模型），其中行动由 AI 模型之间的多数票决定。

- **贝叶斯知识请求**：一位用户请求学习贝叶斯统计的资源，他们被引导至一段名为“贝叶斯定理，改变信念的几何学”的教育类 [YouTube 视频](https://youtu.be/HZGCoVF3YvM)。

- **使用 MyShell Pro Config 进行协作式 AI 编排**：另一位用户介绍了 **MyShell's Pro Config**，作为一种可以促进 **multi-AI decision model** 编排的工具，并建议它可以管理提议的 AI Agent 之间的投票过程。

- **MyShell 作为 AI 原生应用部署平台**：分享了关于 **MyShell** 的更多细节，将其描述为一个用于创建和管理 AI 原生应用的去中心化平台，暗示了其在数据分析等任务中的实用性。

**提到的链接**：

- [Pro Config 模式 (beta) - MyShell](https://docs.myshell.ai/product-manual/create/pro-config-mode-beta)：未找到描述
- [MyShell](https://myshell.ai/)：MyShell 是一个去中心化的综合平台，用于发现、创建和质押 AI 原生应用。
- [贝叶斯定理，改变信念的几何学](https://youtu.be/HZGCoVF3YvM)：也许是概率论中最重要的公式。资助未来的项目：https://www.patreon.com/3blue1brown 同样有价值的支持方式是...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1217465411784933407)** (10 条消息🔥):

- **革新检索方式**：[检索增强语言模型 (Retrieval-Augmented Language Models)](https://arxiv.org/abs/2401.18059) 现在有了一种创新方法——**RAPTOR**，它使用递归摘要来更好地理解长文档并辅助复杂的 QA 任务，与传统的检索增强 LMs 相比表现出显著改进。
- **AI 辅助艺术创作**：适用于 **Huggingface CLIP/ViTs** 的开源多模态可解释性库已发布，[Sonia Joseph 通过 Twitter 透露了这一消息](https://twitter.com/soniajoseph_/status/1767963316943728779)，为 AI 模型中的机械可解释性（mechanistic interpretability）提供了更好的访问途径。
- **扩散模型获得提升**：介绍 **ELLA** (Efficient Large Language Model Adapter)，它将扩散模型与 LLMs 结合，以在文本生成图像中实现更好的语义对齐，详见 [Huggingface 研究论文](https://huggingface.co/papers/2403.05135)。
- **创新的 AI 叙事提示词**：一种针对 Meta 的 Llama 2 AI 进行有效提示的独特方法是在各种叙事中进行角色扮演，[AI 生成的提示词以一种古怪且出人意料的方式超越了人类创作的提示词](https://www.oneusefulthing.org/p/captains-log-the-irreducible-weirdness)。
- **推进文本分段**：一篇 [研究论文](https://arxiv.org/abs/2210.16422) 阐明了长文档分段的重要性，并提出了一个同时进行抽取式摘要和分段的模型，推动了在理解书面和口语文本方面的 SOTA 性能。

**提到的链接**：

- [Toward Unifying Text Segmentation and Long Document Summarization](https://arxiv.org/abs/2210.16422)：文本分段对于标识文档结构非常重要。如果不将长文档分成主题连贯的部分，读者很难理解文本，更不用说...
- [SudoLang: A Powerful Pseudocode Programming Language for LLMs](https://medium.com/javascript-scene/sudolang-a-powerful-pseudocode-programming-language-for-llms-d64d42aa719b)：伪代码是使用非正式自然语言勾勒程序的绝佳方式，无需担心特定语法。它就像……
- [Captain's log: the irreducible weirdness of prompting AIs](https://www.oneusefulthing.org/p/captains-log-the-irreducible-weirdness)：此外，我们还有一个提示词库！
- [RT-2: New model translates vision and language into action](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/)：介绍 Robotic Transformer 2 (RT-2)，这是一种新型视觉-语言-动作 (VLA) 模型，它从网络和机器人数据中学习，并将这些知识转化为通用的指令...
- [Paper page - ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment](https://huggingface.co/papers/2403.05135)：未找到描述
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)：检索增强语言模型可以更好地适应世界状态的变化并整合长尾知识。然而，大多数现有方法仅从检索中提取短的连续块...
- [GitHub - havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍](https://github.com/havenhq/mamba-chat)：Mamba-Chat：基于状态空间模型架构的聊天 LLM 🐍 - havenhq/mamba-chat

  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1217452805577510942)** (13 条消息🔥): 

- **GLiNER：命名实体识别的飞跃**：*cubietom* 分享了一个名为 GLiNER 的新模型框架演示，该框架允许即时选择自定义标签进行命名实体识别 (NER)，为具有预定义实体的传统模型提供了一个实用的替代方案。演示可在 [HuggingFace Spaces](https://huggingface.co/spaces/tomaarsen/gliner_base) 获取，同时还有额外的模型变体和用于进一步探索的 GitHub 仓库。

- **笑是良药**：*tonic_1* 分享了他们使用 HuggingFace 的 *starchat2-playground* 完全创作的一个作品。他们展示了一个名为 kosmos-2 的演示，可在 [HuggingFace Spaces](https://huggingface.co/spaces/Tonic1/kosmos-2) 获取。

- **可视化 LLM 版图**：*taratra_dr* 向社区更新了最新版本的 Open LLM Leaderboard Viz 空间，具有交互式可视化和大型语言模型的比较功能。新功能包括指标重排序和绘制多个模型进行对比，可通过 [HuggingFace Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) 访问。

- **一键代码重构**：*krolhm* 介绍了一个用于重构代码的新 Visual Studio Code 插件，由带有 llama cpp 服务器的本地大型语言模型 (LLM) 驱动，仓库可在 [GitHub](https://github.com/rbourgeat/refacto) 获取。

- **播种多模态可解释性的种子**：*soniajoseph_* 宣布创建一个开源库，为 Huggingface CLIP/Vision Transformer (ViT) 模型引入多模态机械可解释性（mechanistic interpretability）。相关链接包括一篇 [Twitter 帖子](https://twitter.com/soniajoseph_/status/1767963316943728779)以及发表在 [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) 上的详细文章。

**提到的链接**：

- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz)：未找到描述
- [Kosmos 2 - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/kosmos-2)：未找到描述
- [StarChat2 Demo - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/starchat2-playground)：未找到描述
- [GitHub - rbourgeat/refacto: Refactor your code with local LLM](https://github.com/rbourgeat/refacto)：使用本地 LLM 重构你的代码。通过在 GitHub 上创建账号来为 rbourgeat/refacto 的开发做出贡献。
- [Machine learning-based intrusion detection: feature selection versus feature extraction - Cluster Computing](https://link.springer.com/article/10.1007/s10586-023-04089-5)：物联网 (IoTs) 在智慧城市、智慧农业、智慧医疗和智慧制造等许多领域发挥着重要作用。然而，IoT 设备非常脆弱...
- [Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic)：见证 dogit lens。Patch 级的 logit 归因是一种涌现的分割图。点击此处加入我们的 Discord。…
- [GLiNER-Base, zero-shot NER - a Hugging Face Space by tomaarsen](https://huggingface.co/spaces/tomaarsen/gliner_base)：未找到描述
- [urchade/gliner_base · Hugging Face](https://huggingface.co/urchade/gliner_base)：未找到描述
- [urchade/gliner_multi · Hugging Face](https://huggingface.co/urchade/gliner_multi)：未找到描述
- [Models - Hugging Face](https://huggingface.co/models?library=gliner)：未找到描述
- [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)：命名实体识别 (NER) 在各种自然语言处理 (NLP) 应用中至关重要。传统的 NER 模型虽然有效，但局限于一组预定义的实体类型。相比之下...
- [GitHub - urchade/GLiNER: Generalist model for NER (Extract any entity types from texts)](https://github.com/urchade/GLiNER)：NER 通用模型（从文本中提取任何实体类型） - urchade/GLiNER

  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1217806245684314214)** (6 条消息): 

- **本周无分享**：本周的阅读小组将没有演示报告，但下周已计划了一场。
- **MNIST 数字分类问题**：一位正在学习 **Andrew Ng 神经网络课程** 的成员对 MNIST 数字分类中第一层的单元数量感到困惑，因为图像大小是 20x20 像素。
- **探索神经网络架构**：针对关于如何确定**神经元数量和隐藏层数**的问题，另一位成员解释说，这通常涉及实验并借鉴以往成功的配置，同时权衡处理能力、速度和准确性。
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1217688016907276338)** (2 条消息): 

- **使用 LoRAs 融合风格**：现已发布关于合并 [Low-Rank Adaptations (LoRAs)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) 的**指南**，可以通过融合不同风格来创建独特的图像。在 [merge LoRAs 指南](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)中提供了详细说明，包括 `set_adapters()` 和 `fuse_lora()` 等方法。
  
- **Diffusers 库更新**：*Diffusers* 库的新版本 **0.27.0** 已经发布。发布说明可以在 [GitHub 页面](https://github.com/huggingface/diffusers/releases/tag/v0.27.0)上找到。

**提到的链接**：

[Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras)：未找到描述

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1217582601859240046)** (2 条消息):

- **LORA 和 PEFT 的延迟问题**：一位成员讨论了将 `peft` 与 *diffusers* 集成时面临的挑战，在从 peft 0.6 升级到 0.9 时遇到了延迟激增。`load_lora_weights` 函数明显变慢，耗时从 1-2 秒增加到约 14 秒，这对于他们的系统来说太高了。他们分享了一份关于使用 [HuggingFace 动态加载 LORAs](https://huggingface.co/blog/lora-adapters-dynamic-loading) 的指南。

- **使用 FreeU 增强图像生成**：分享了 FreeU 技术的概览，详细介绍了如何通过在反向扩散过程中平衡 UNet 架构中 skip connections 和 backbone features 的影响来提高图像生成质量。该方法被强调为不需要额外的训练，并可应用于各种任务，更多信息可在 [Hugging Face 指南](https://huggingface.co/docs/diffusers/using-diffusers/freeu)中找到。

**提到的链接**：

[使用 FreeU 提升生成质量](https://huggingface.co/docs/diffusers/using-diffusers/freeu)：未找到描述

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1217398660669767690)** (13 条消息🔥)：

- **对 CLIP Embedding 的好奇**：一位参与者了解到，可以通过 **CLIP model** 处理图像以生成并保存 embeddings，以便后续用于训练，并强调原始图像不应能从这些 embeddings 中重建。然而，另一位参与者对从 embeddings 重建图像是否完全不可行表示不确定。

- **使用 CLIP Embedding 进行训练**：讨论强调，使用 **CLIP embeddings** 而不是实际图像进行训练可能会因任务而异，对于目标检测、分类和姿态估计等任务，训练工作流的差异仍存在不确定性。

- **CLIP Embedding 的大小**：有人提到来自 **CLIP model** 的 embeddings 占用的空间可能比图像本身更大，并且在经过 **CLIPVisionModel** 处理后，这个大小是增加还是减少也存在一些模糊性。

- **Batch Normalization 作为知识保留手段**：提到了一篇 arXiv 论文，讨论了如何将 Batch Normalization 用于医学分割模型中的终身学习，以防止遗忘旧特征，尽管论文的具体名称已记不清。

- **扩展图像生成的微调规模**：一位用户询问了如何在不到一周的时间内，在不错的硬件上使用包含 250 万张图像的大型数据集微调 Stable Diffusion (SD) 模型的技巧，寻求超越使用小数据集进行微调的教程。


  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1217497650119839744)** (19 条消息🔥)：

- **数据集大小影响 Mistral 模型的灵活性**：一位成员报告称，使用小数据集微调 **Mistral 7B** 可以保持灵活性（如修改对象），但使用较大数据集会导致模型在对象生成方面变得专业化，从而牺牲了其他任务。他们询问考虑到模型的大小，这是否可能是某种形式的 overfitting（过拟合），并寻求减轻该问题的建议。

- **在模型训练中促进泛化**：针对模型泛化能力不佳的担忧，一位参与者建议通过增加更多样化的示例来增强训练集，以提高在模型新数据上的表现。

- **对修改后的 Mistral 模型进行基准测试**：一位用户分享了他们将基础模型 **Mistral-7V-v0.1** 与修改版本进行对比的研究想法，寻求关于如何使用 HuggingFace 自动化基准测试的指导，并询问这些基准测试在哪里运行。

- **OpenLLM Leaderboard 基准测试提交说明**：另一位成员澄清说，对于 OpenLLM leaderboard，基准测试是在 Hugging Face 集群上运行的。他们提供了 **LightEval** 和 **lm-evaluation-harness** 等资源的链接，用于自我基准测试。
[GitHub 上的 LightEval 套件](https://github.com/huggingface/lighteval)。
[GitHub 上的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。

- **模型压缩技术的潜在创新**：围绕一种新的模型优化方法展开了讨论，该方法可能在保持准确性的同时节省内存占用，包括在 `4096 x 4096` 矩阵上的成功初步结果。一位成员对将此技术应用于模型架构中更大的矩阵表示热切期待。

**提到的链接**：

- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.](https://github.com/huggingface/lighteval): LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1217582601859240046)** (2 messages): 

- **解决 PEFT 和 Diffusers 的延迟问题**：一位使用 [LoRA 适配器进行动态模型加载](https://huggingface.co/blog/lora-adapters-dynamic-loading) 的服务器运营商报告了在集成 **`peft`** 时出现的高延迟问题。虽然 **`peft` 0.9** 大幅增加了 `load_lora_weights` 的时间至 14 秒，但 0.6 版本虽然减少了该时间，却将 `unload_lora_weights` 的时间增加到了 6 秒，这两者对于他们的系统来说都是不可接受的。

- **使用 FreeU 提升图像质量**：讨论了一种名为 **FreeU** 的改进图像生成技术，该技术通过重新平衡 UNet 的 skip connections 和 backbone feature maps 的贡献来增强图像质量。该技术可在推理过程中使用，无需额外训练，适用于文本转图像（text-to-image）、图像转图像（image-to-image）和文本转视频（text-to-video）任务，详见 [HuggingFace 指南](https://huggingface.co/docs/diffusers/using-diffusers/freeu)。

**Links mentioned**:

[使用 FreeU 提升生成质量](https://huggingface.co/docs/diffusers/using-diffusers/freeu)：未找到描述

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1217560477933240420)** (3 messages): 

- **临时服务中断警报**：OpenRouter 经历了一个短暂的问题，由于数据库更新时间过长，导致部分 Activity 行丢失了约三分钟，这可能会影响该时间段内 completions 的计费。该问题已迅速得到解决，并声明“*这些 completions 都不会被收费*”。

- **Claude 3 Haiku 发布**：Anthropic 的 **Claude 3 Haiku** 现已在 OpenRouter 上线，其特点是高速（约每秒 120 tokens）和高成本效益（每美元 400 万 prompt tokens）。这个低延迟的 beta 版本提供受监管（moderated）和自我监管（self-moderated）两种选项，适用于需要近乎即时响应的使用场景。在此查看模型及其定价 [这里](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)。

- **新模型发布**：Cohere 的 **Command-R** 模型现已可在 OpenRouter 上访问，展示了 128,000 tokens 的长上下文能力，费率为每美元 200 万 prompt tokens。为了提供无缝的用户体验，已努力将 Command-R 与通用 API 进行对齐。感兴趣的用户可以通过此 [链接](https://openrouter.ai/models/cohere/command-r) 探索 Command-R。

- **每日分析现已上线**：OpenRouter 推出了每日分析功能，使用户能够按日追踪 token 使用情况，在现有的每周分析基础上提供更细致的视图。用户可以在 [这里](https://openrouter.ai/rankings) 查看新的分析。

- **宣布性能改进**：OpenRouter 显著提升了 `/models` API 的速度，并增强了所有模型相关网页的性能，包括对 Mixtral Nitro 的改进。

**Links mentioned**:

- [Anthropic: Claude 3 Haiku (self-moderated) by anthropic | OpenRouter](https://openrouter.ai/models/anthropic/claude-3-haiku:beta): 这是与 Anthropic 合作提供的 [Claude 3 Haiku](/models/anthropic/claude-3-haiku) 的低延迟版本，该版本为自我监管：响应监管发生在模型上...
- [Cohere: Command-R by cohere | OpenRouter](https://openrouter.ai/models/cohere/command-r): Command-R 是一个指令遵循对话模型，与之前的模型相比，它能以更高质量、更可靠地执行语言任务，并具有更长的上下文。它可以用于复杂的 w...
- [OpenRouter](https://openrouter.ai/rankings): 按应用使用情况排名和分析的语言模型

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1217470944973426799)** (7 messages):

- **Olympia.Chat 宣布集成 OpenRouter**：[Olympia.Chat](https://olympia.chat) 是一个深受个人创业者和小企业主欢迎的 ChatGPT 仿品，目前正将 OpenRouter 作为其组件的 LLM 来源。此外，一个**功能完备的 OpenRouter Ruby 库**即将开源。

- **Messenger 聊天机器人开放测试**：一名成员的朋友创建了一个 **Messenger 聊天机器人**，该成员正邀请其他人通过私信获取测试机会。

- **集成 OpenAI 的 AI 网关发布**：全新的 AI 网关 [EZLinkAI Platform](https://platform.ezlinkai.com/) 上线，为注册用户提供 1 美元的赠金，并允许用户以原价 80% 的成本调用 OpenAI、Claude、Mistral 和 Groq 服务。

- **征求 AI 网关的反馈**：**AI 网关**的创建者正在寻求更多反馈，旨在通过用户输入来改进服务。

**提到的链接**：

- [Olympia | Better Than ChatGPT](https://olympia.chat)：通过价格合理的 AI 驱动顾问来发展您的业务，这些顾问是业务战略、内容开发、营销、编程、法律战略等领域的专家。
- [How Anthony Mennella GIF - How Anthony Mennella Culter35 - Discover &amp; Share GIFs](https://tenor.com/view/how-anthony-mennella-culter35-how-did-you-do-that-how-to-do-it-gif-20143672)：点击查看 GIF
- [EZLINK AI](https://platform.ezlinkai.com/)：未找到描述

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1217441390368264232)** (129 条消息🔥🔥): 

- **GPT-4.5 Turbo 消失事件**：一名成员分享了一个链接（[openai.com/blog/gpt-4-5-turbo](https://openai.com/blog/gpt-4-5-turbo)），据称是 GPT-4.5 Turbo 存在的证据，但随后表示该链接已失效，引发了一阵笑声。
- **Mistral 模型之谜**：用户报告了 Mistral 模型行为的不一致，包括“请求过大（Request too big）”错误以及 32k 上下文限制的问题。对话涉及了对确切错误消息的查询，并提出了错误原因的假设，如请求中的重复循环。
- **Claude 3 Haiku 热度**：讨论显示了对 Claude 3 Haiku 的热情，其每百万 token 1.25 美元的成本效益受到关注，且在头脑风暴角色扮演场景和角色开发方面显著优于其他模型。
- **OpenRouter 品牌合作**：讨论了在 Open Agent Studio 中添加 OpenRouter 按钮的提议，并请求品牌指南或特定图标，OpenRouter 方面已予以批准。
- **各种 LLM 的探索**：聊天中成员们对比了各种语言模型，包括 Gemini 和 Claude 模型，辩论了它们在编程和创意任务中的能力，抱怨了某些怪癖（如不必要的项目符号），并因性能和缺乏审查而对某些模型表现出强烈偏好。

**提到的链接**：

- [GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)](https://github.com/open-webui/open-webui)：适用于 LLM 的用户友好型 WebUI（原 Ollama WebUI） - open-webui/open-webui
- [No-Code Agents Live Group Onboarding](https://youtu.be/dT1p7aAC1eU)：快速掌握最佳实践和操作顺序，节省时间并实现任何手动流程的自动化。预约您的...
- [GitHub - BerriAI/litellm: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs)](https://github.com/BerriAI/litellm)：使用 OpenAI 格式调用所有 LLM API。支持 Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs) - BerriAI/litellm

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1217507828701466798)** (3 条消息): 

- **推出 LlamaParse**：全新的 **LlamaParse** 文档解析器已发布，提供对图像、表格和图表的卓越解析能力，并增加了遵循自然语言指令的能力。在这条 [推文](https://twitter.com/llama_index/status/1767948064659210310) 中了解它如何超越其他产品。

- **LlamaIndex 通过 Presidio 处理 PII**：@RoeyBC 在 **LlamaIndex** 上的客座文章重点介绍了 **Presidio**，这是微软的一个开源库，用于识别和匿名化个人身份信息（PII）以防止数据泄露。在这条 [推文](https://twitter.com/llama_index/status/1768050386823463368) 中阅读其在数据保护方面的重要性。

- **克服金融领域的 RAG 挑战**：由于金融领域 PowerPoint 演示文稿具有独特的格式（包括表格、图像和图表），RAG 在解析这些文档时面临困难。实现更好的文本定位和解析技术是至关重要的第一步，详见此 [推文](https://twitter.com/llama_index/status/1768303288381030408)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1217438443223781436)** (82 messages🔥🔥): 

- **Azure AI Search 索引问题**：一位遵循 [AzureAISearchIndexDemo 指南](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html) 的成员遇到了一个问题：Azure 索引显示总存储量为 3mb，但向量索引大小为 0。正在寻求针对此差异的建议。

- **LlamaIndex Python 包的警告**：一位用户报告了多个关于无法使用 `OpenAIPydanticProgram` 的警告。建议运行 `pip install llama-index-program-openai` 来解决此问题。

- **关于 npx create-llama 错误的担忧**：一位成员在使用 `npx create-llama` 并以文本文件作为数据源时遇到了错误，提示“抱歉！我们在您的提示词中遇到了重复模式的问题”，即使是简单的提示词也是如此。推测该错误可能与文件的内容有关。

- **LlamaIndex 中 Retriever 的评估方法**：一位用户寻求关于将 LlamaIndex 的 RetrieverEvaluator 与其自定义的问题上下文对（question context pairs）结合使用的建议。提到查询需要预期的节点 ID（node IDs），但用户询问是否可以仅使用预期的文本或文档 ID。

- **OpenAI Assistant Agent 的性能问题**：一位成员讨论了在使用 OpenAIAssistantAgent 构建聊天机器人时响应时间超过 10 秒的问题。建议使用流式传输（streaming）可能会让体验感觉更快，且响应缓慢的部分原因可能是由于最近 OpenAI API 的性能问题。

**提到的链接**：

- [检索评估 - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval.html)：未找到描述
- [Azure AI Search - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html)：未找到描述
- [Simple Fusion Retriever - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion.html#simple-fusion-retriever)：未找到描述
- [摄取管道 + 文档管理 - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html)：未找到描述

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1217445256375566456)** (61 messages🔥🔥): 

- **寻找开源的即时聊天模型**：围绕模型大小与训练资源及硬件能力之间的复杂关系展开了讨论。推荐了 [Mistral 和 Mixtral](https://huggingface.co/) 模型，因为它们的开源性质且没有明显的过滤器。
  
- **模型训练雄心面临 VRAM 限制**：一位参与者表达了训练大型模型的意图，并强调了 [用于 Mac GPU 训练加速的 PyTorch Metal Performance Shaders (MPS) 后端](https://developer.apple.com/metal/pytorch/)。其他人询问了单 GPU 设置上的微调能力和限制，建议需要 [高效的微调方法](https://huggingface.co/papers/2403.06504)。

- **在 Mixtral 和 Qwen 70B 之间权衡医疗领域训练**：一位成员考虑为医学领域训练一个大型模型，并在 Mixtral 和 Qwen 70B 模型之间进行权衡。讨论中提到了对即将出现的显存溢出（OOM）问题以及即将发布的全新 Llama 模型的担忧。

- **咨询训练格式的最佳实践**：成员们就将原始文本转换为训练用途时，使用补全（completion）格式还是问答（Q/A）格式交换了意见。建议参考现有的 Hugging Face 数据集示例以正确格式化数据。

- **用于 Axolotl 的 GPUDirect Storage**：一位参与者建议将 NVIDIA 的 GPUDirect® Storage 技术集成到 Axolotl 系统中，该技术为 GPU 显存和存储之间的传输提供直接数据路径，详见 NVIDIA 的 [cuFile API 参考](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)。这可以通过增加系统带宽和减轻 CPU 负载来提升性能。

**提到的链接**：

- [在 Mac 上加速 PyTorch 训练 - Metal - Apple Developer](https://developer.apple.com/metal/pytorch/): PyTorch 使用新的 Metal Performance Shaders (MPS) 后端进行 GPU 训练加速。
- [论文页面 - 添加 NVMe SSD 以在单张 GPU 上启用并加速 100B 模型微调](https://huggingface.co/papers/2403.06504): 未找到描述
- [cuFile API 参考指南 - NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html): 未找到描述

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1217389990292750378)** (6 条消息): 

- **分享了 GPU Direct 入门视频**: 分享了一个[介绍视频](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4)，解释了 NVIDIA 的 GPUDirect Storage (GDS)，提供了对点对点 PCIe 的见解以及 GDS 在技术进步中的作用。
- **Axolotl 代码查询**: 一名成员发布了关于 Axolotl 代码特定部分的查询，并附带了[相关 GitHub 章节](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808)的链接，寻求对其用途的澄清。
- **模型加载说明**: 针对该查询，澄清了当将基础模型指针指向 *peft model* 时，会触发引用的代码，从而使 AutoModel 能够在此处加载 peft 模型。
- **新功能请求**: 一名成员对正在开发或引入的最新功能表示好奇。
- **PEFT 论文链接**: 针对新功能的询问，分享了一篇关于 ["PEFT" 的研究论文](https://arxiv.org/pdf/2403.06504.pdf)，暗示了建模领域的进展。

**提到的链接**:

- [P2P PCIe 和 GPUDirect Storage 入门](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4): 这是一个五分钟的快速概览，介绍了 NVIDIA 的 GPUDirect Storage (GDS) 是什么、它的作用、它基于哪些技术，以及它在何处最能发挥作用...
- [axolotl/src/axolotl/utils/models.py at 8a82d2e0a443fb866b15d7bd71fffbd8171de44b · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808): 欢迎随时提问。通过在 GitHub 上创建账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1217868370293030993)** (7 条消息): 

- **寻求 LoRA 模型的推理代码**: 一名成员提到他们已经**基于 `Mistral-7B-v0.1` 微调了 LoRA**，并寻求在 notebook 中对大约 100 个 prompt 运行推理的示例代码。他们正在考虑使用 `transformers` 库和 `model.generate(**model_inputs)` 方法。

- **推荐使用 vLLM 进行快速推理**: 另一名成员推荐使用 **vLLM** 运行批处理推理，声称它比 `transformers` 更快。他们提供了一个使用 vLLM 的[快速入门指南](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)，涵盖了离线批处理推理和构建兼容 OpenAI 的 API 服务器。

- **考虑将 vLLM 用于非服务器任务**: 原始询问者不确定 **vLLM** 是否适合他们的需求，因为他们不打算部署模型，而只是运行一些预测进行探索。在得到其效率保证后，他们决定参考 vLLM 快速入门链接。

**提到的链接**:

[快速入门 — vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): 未找到描述

  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1217474856644382733)** (3 条消息): 

- **Mistral Medium 表现优于 Mixtral**: 一位用户指出 **Mistral Medium** 产生了更好的回复，并认为它是 **Mixtral** 的闭源、更高级版本。

- **注意到 RAG 性能**: 同一位用户提到在没有明确要求的情况下观察到了带有 **RAG 性能** 的引用生成。

- **更简洁，更好的指令遵循**: 还观察到 **Mistral Medium** 的输出比 **Mixtral** 更简洁，且在遵循指令方面更有效。
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1217877731963048016)** (1 条消息):

- **langchain 0.2 加速发布**：由于针对 `langchain` 提交的 CVE，团队正在考虑加速发布 `langchain 0.2`，将其与 `langchain-community` 分离。有关此更改的详细讨论和动机可以在 [GitHub](https://github.com/langchain-ai/langchain/discussions/19083) 上找到，并鼓励社区提供反馈以确保其满足用户需求。

**提到的链接**：

[RFC: Expedited langchain 0.2 release · langchain-ai/langchain · Discussion #19083](https://github.com/langchain-ai/langchain/discussions/19083)：背景：目前 langchain（包）依赖于 langchain-community。这样做仅是为了与早于 langchain 和 langchain-com 分离之前的 langchain 版本保持向后兼容性...

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1217401410287566909)** (64 条消息🔥🔥): 

- **LangChain 咨询**：一位寻求 **LangChain** 帮助的成员被引导至 Discord 上相应的帮助频道以获取协助。
- **AgentExecutor 问题**：有提到 `AgentExecutor` 返回 `OutputParserException` 的困难，即使 **Cohere model** 似乎能够准确生成 Python 代码。
- **AI Agent 的底层原理**：关于为什么要使用 AI Agent 而不是 LLM + functions 的讨论强调了 Agent 可以处理顺序操作，并具有内置的错误处理等功能。
- **AI Agent 行为评估**：一位成员寻求关于评估 AI Agent 行为的建议，并被推荐参考 [LangChain 调试和评估指南](https://python.langchain.com/docs/guides/evaluation/)，尽管大家承认该领域相对较新，基准测试（benchmarks）仍在开发中。
- **StackOverflow API 探索**：一位用户询问关于 **StackOverflow** 的 API，并获得了关于使用 [StackExchange API](https://api.stackexchange.com/docs/advanced-search) 根据特定查询和结构化数据进行高级搜索的指导。

**提到的链接**：

- [GroqCloud](https://console.groq.com/docs/openai#text-completion)：体验全球最快的推理速度
- [[beta] Structured Output | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/structured_output)：让 LLM 返回结构化输出通常至关重要。
- [Discord Bot | MEE6](https://mee6.xyz/en)：通过等级、审核、Twitch、Youtube 和 Reddit 通知来管理你的 Discord 服务器。
- [OpenAI assistants | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/agent_types/openai_assistants)：[Assistants
- [LangChain](https://www.youtube.com/playlist?list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5)：未找到描述
- [Debugging | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/debugging)：如果你正在使用 LLM 进行构建，某些时候总会出问题，你需要进行调试。模型调用可能会失败，或者模型输出格式错误，或者存在一些嵌套模块...
- [Evaluation | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/evaluation/)：使用语言模型构建应用程序涉及许多活动部件。其中最关键的组件之一是确保模型产生的结果在广泛范围内是可靠且有用的...
- [Usage of /search/advanced [GET] - Stack Exchange API](https://api.stackexchange.com/docs/advanced-search)：未找到描述
- [The validation of tools within OpenAIAssistantRunnable.create_assistant does not account for `{"type": "code_interpreter"}`. · Issue #19057 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/19057)：检查了其他资源。我为此 Issue 添加了一个非常详细的标题。我使用集成搜索搜索了 LangChain 文档。我使用 GitHub 搜索来查找类似问题...
- [langsmith-cookbook/testing-examples/tool-selection/tool-selection.ipynb at main · langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/tool-selection/tool-selection.ipynb)：通过在 GitHub 上创建账号来为 langchain-ai/langsmith-cookbook 的开发做出贡献。
- [GitHub - ggerganov/whisper.cpp: Port of OpenAI's Whisper model in C/C++](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file)：OpenAI Whisper 模型的 C/C++ 移植版本。通过在 GitHub 上创建账号来为 ggerganov/whisper.cpp 的开发做出贡献。
- [Wordware - Try all the models for a single question](https://app.wordware.ai/r/fc405cb4-877b-44b7-aed8-b883e48eced3)：此 Prompt 将一个问题运行于 Gemini, GPT-4 Turbo, Claude 2, Mistral Medium, Mixtral 和 Openchat。然后使用 GPT-4 Turbo 评估哪个模型给出了最佳答案。

---

**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1217795010247000124)** (1 条消息):

- **关于 Prompt 模板中变量集成的疑问**：一位成员询问如何将一个变量（具体为 `tools = [cat_tool]`）集成到 Langsmith Hub 的 Prompt 模板中，该模板在结构中包含占位符 `{tools}`：

  ``` 
  System : 

  You are a helpful assistant that have these {tools} to help answer questions.
  ```
  
  他们正在寻求关于如何在代码中引用变量 `tools` 以与 Prompt 保持一致的指导。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1217398683377602611)** (8 条消息🔥): 

- **使用 ReAct 进行反应**：受“ReAct: Synergizing Reasoning and Acting in Language Models”论文启发的 ReAct Agent 已经发布，它拥有推理引擎和多种技能，可以通过“今天的比特币价格是多少？”等问题进行测试。相关论文可在 [Download PDF](https://arxiv.org/abs/2210.03629) 下载。
  
- **开源 Langchain Chatbot**：推出了一款新的开源 Langchain Chatbot，用于演示使用 RAG 技术进行高效的问答查询，其 [GitHub 仓库](https://github.com/Haste171/langchain-chatbot) 具有简单的设置和交互式 UI。
  
- **MindGuide：通过 ChatModels 创新心理健康**：分享了一篇名为“Revolutionizing Mental Health Care through LangChain”的文章，详细介绍了利用 LangChain 和 ChatOpenAI 提供心理健康支持的 MindGuide Chatbot，摘要和下载地址见 [Download PDF](https://arxiv.org/abs/2403.05568)。

- **Claude 与 LangGraph 结合进行监督**：分享了一个展示由 Claude 驱动的 LangGraph Agent Supervisor 的 GitHub Notebook，演示了利用 LangChain 结合 Claude 能力的潜力，详见 [GitHub Notebook](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb)。

- **Deci AI Nano 模型 API 预览**：宣布了 Deci AI 的新 Nano 模型 API，并附带了基础用法和 LangChain 用法的 Colab Notebook，可在正式发布前进行探索，[基础用法 Notebook](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing) 和 [LangChain 用法 Notebook](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing) 已提供访问链接。

**提到的链接**：

- [Revolutionizing Mental Health Care through LangChain: A Journey with a Large Language Model](https://arxiv.org/abs/2403.05568)：现代社会心理健康挑战日益增多，解决心理障碍（尤其是焦虑、抑郁和自杀念头）的紧迫性凸显了对...的需求。
- [LangChain for JavaScript part 3: Create Dall-E images](https://fek.io/blog/lang-chain-for-java-script-part-3-create-dall-e-images/)：FEK.IO David Fekke L.L.C. 的网站。
- [Google Colaboratory](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing)：未找到描述
- [Google Colaboratory](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing)：未找到描述
- [slangchain/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb at main · prof-frink-lab/slangchain](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb)：通过创建账号为 prof-frink-lab/slangchain 的开发做出贡献。
- [Unlocking the Future of AI Applications with SAP HANA Vector Engine and LangChain](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d)：Ankush k Singal
- [GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.](https://github.com/Haste171/langchain-chatbot)：用于以对话格式分析/提取数据信息的 AI Chatbot。- Haste171/langchain-chatbot
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)：虽然大型语言模型 (LLM) 在语言理解和交互决策任务中展示了令人印象深刻的能力，但它们的推理能力（例如 Chain-of-Thought...）
- [Wordware - ReAct API Agent 🧠](https://app.wordware.ai/r/0b8b7771-09dc-4a19-87d4-89e43b5cc153)：研究如何使用 API

  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1217452564510150686)** (2 条消息):

- **学习使用 Langchain 创建 Prompt Template**：分享了一个名为 "Create Prompt Template With Langchaingo" 的视频教程，演示了如何创建 Prompt Template 并将其与 Langchain 配合使用，特别提到了 [Telegram group](https://t.me/langchaingo/1)。内容面向对 #golang 和 #langchain 感兴趣的开发者，视频可在 [YouTube](https://youtu.be/dcBEtgh4078) 上观看。

- **深入探讨 Hermes 2 Pro 7B 的 Function Calling**：分享了另一个名为 "Lets Function Call with Hermes 2 Pro 7B" 的视频，重点介绍了使用 **Hermes 2 Pro 7B** 模型进行 Function Calling。源代码和示例可以在 [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) 上找到，视频可在 [YouTube](https://www.youtube.com/watch?v=PzaidfqDtGI) 观看，目标受众为 #llm 和 #largelanguagemodels 爱好者。

**提到的链接**：

- [Create Prompt Template With Langchaingo](https://youtu.be/dcBEtgh4078)：在这个视频中，我将展示如何创建 Prompt Template 以及如何将其与 Chain 结合使用。Telegram 群组：https://t.me/langchaingo/1 #golang #langchain
- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI)：让我们使用 Hermes 2 Pro 7B 进行 Function Calling。https://github.com/NousResearch/Hermes-Function-Calling/tree/main #llm #largelanguagemodels

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1217547023675555970)** (6 条消息): 

- **推文引发讨论**：一名成员分享了 [Andrew Curran 的一条推文](https://twitter.com/AndrewCurran_/status/1767916848987914487)，引发了关于语言应用和协作的对话，强调需要通过 Aya 项目与子小组合作。
- **欧洲学术界的 Polyglot 项目**：在讨论欧洲大学的需求时，一名成员提到了说服人们采用新语言方法的挑战，特别提到了英语和德语的应用。
- **LLM 对德语支持良好**：一位成员指出，主流的 LLM (Language Learning Models) 通常开箱即用地对德语提供良好支持，同时也建议联系 Aleph 以在高度受监管的行业寻求合作伙伴关系。
- **Aleph 的性能受到质疑**：一名成员表达了他们认为 Aleph 性能不足的观点，这引发了一个建议：虽然 Aleph 本身可能达不到标准，但他们仍可以协助推荐当地的数据合作伙伴。
  

---


**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1217542990441087056)** (2 条消息): 

- **GPT-4 稳坐宝座**：一名成员评论说，根据一篇论文，**GPT-4** 仍然是 **LeetCode** 上的领先模型。提到的论文可以在 [livecodebench.github.io](https://livecodebench.github.io/pdfs/paper.pdf) 找到。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1217710697434189874)** (3 条消息): 

- **模型细节查询**：一名成员询问另一名成员在最近的练习中使用了哪个模型，对模型的身份和能力表现出兴趣。

- **寻求关于供应商安全过滤的引用**：一名成员正在寻找权威来源或文档来引用“基础模型供应商在文本生成后进行了大量的安全过滤”，但指出与 Prompt 重写相比，这方面的文档记录较少。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1217715794780098571)** (2 条消息): 

- **补看那些“辛辣”的内容**：一名成员提到他们打算补看积压的新闻通讯，并表示拥有“辛辣的读者”是有益的。他们还提到了一条关于生物风险的预热推文，虽然他们不同意该推文，但在阅读全文之前保留意见。
- **生物风险推文困惑**：关于一条与生物风险相关的推文存在简短的困惑。有人提到推文发得太多，可能暗示初始推文中缺乏背景或信息。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1217461687171547176)** (54 条消息🔥): 

_

- **期待 GPT-4.5**：成员们表示，如果 **GPT-4.5** 突然发布，他们已准备好撰写紧急博客文章。
- **YouTube Premium 与 LLM 中的 Google 广告**：讨论了 Google 将免费用户转化为付费用户的方法，尽管广告策略激进，一些成员仍订阅了 YouTube Premium。如果广告被整合到 Google 的 ChatGPT 竞争对手中，用户对信任度的担忧被提及。
- **Claude-3 表现优于 GPT-4**：社区对新的 **Claude 模型**家族表现出极大的热情，Claude-3-Opus 与 GPT-4-Turbo 并列榜首。目前计划为不同领域创建单独的排行榜，以更清晰地洞察模型能力（[LM SysOrg 关于 Claude-3 的更新](https://fxtwitter.com/lmsysorg/status/1767997086954573938)）。
- **分析 Claude 3 的新成员**：成员们讨论了 **Claude 3 Haiku**，这是一款快速且价格合理的模型，同时思考了它替换旧系统的有效性，以及在特定任务中进行 Prompt Engineering 的潜在挑战（[Xeophon 关于用法的想法](https://x.com/TheXeophon/status/1768047237626515662)）。
- **为 AI 助手标准化研究文献的挑战**：对话延伸到了创建高效 AI 文献调研助手的困难，原因包括引用歧义、图表解读以及构建论文评审系统，这暗示了 AI 文档解析研究的未来方向（[关于文献调研挑战的讨论](https://arxiv.org/abs/2402.18819)）。

**提到的链接**：

- [来自 Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1768057955620913495>)): @felix_red_panda @karthikv792 很高兴听到你的结论！:)
- [来自 lmsys.org (@lmsysorg) 的推文](https://fxtwitter.com/lmsysorg/status/1767997086954573938): [Arena 更新] 我们的社区为 Claude-3 Opus 和 Sonnet 投了 20,000 多票，对新的 Claude 模型家族表现出极大的热情！Claude-3-Opus 现在与 GPT-4-Tu 并列第一...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018310615151002?s=46): 今天我们发布了 Claude 3 Haiku，它是同类智能水平中最快、最实惠的模型。Haiku 现在已在 API 和 http://claude.ai 上面向 Claude Pro 订阅者开放。
- [来自 Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1768047237626515662): Claude 3 模型在论文摘要方面的对比。Prompt 相同，通过 Poe + PDF 上传访问模型。在这里，我一点也不喜欢 Haiku，它太贴近原文了。我更倾向于...

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1217660813746241587)** (8 条消息🔥): 

- **GTC 聚会通知**：一位成员宣布下周将参加 **GTC**，并邀请其他人线下见面。
- **BLAS 与 NumPy 性能之争**：一位成员提供的[链接](https://ashvardanian.com/posts/numpy-vs-blas-costs/)强调，尽管 NumPy 非常流行，但在某些操作中，它相比 **BLAS 性能损失高达 90%**。[SimSIMD](https://github.com/ashvardanian/simsimd) 被作为解决此问题的潜在方案。
- **对 NumPy 性能分析的怀疑**：另一位成员指出，基准测试的工作负载时间极短（<1µs），且 NumPy 的固定开销更高，这表明在大量小操作中使用 NumPy 可能会有问题。
- **SIMD Wrappers 作为实际解决方案**：一位成员指出，对于较小向量的操作，使用 **SIMD Wrapper** 比处理数据传输和 Kernel 启动的开销更有效率。
- **专注于技术选择的信息传递**：建议通过关注技术选择背后的基本原理、合适的用例和安装指南，而不是仅仅列出基准测试数字，来进行更精确的信息传递。

**提到的链接**：

[NumPy vs BLAS: 损失 90% 的吞吐量](https://ashvardanian.com/posts/numpy-vs-blas-costs/): 下载量超过 50 亿次，NumPy 是 Python 中最流行的数值计算库。它封装了像 BLAS 和 LAPACK 这样的底层 HPC 库，为矩阵提供高级接口...

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1217825373170176181)** (6 条消息):

- **Inspecting Triton tl.core.tensor Objects**: 用户寻求关于如何检查 Triton 中 **tl.core.tensor** 对象的建议，并指出使用常规索引查看值会产生 '0d block_type is forbidden' 错误。
- **Old-School Debugging Tricks**: 为了检查 Triton tensors，一位成员建议使用环境变量 `TRITON_INTERPRET=1` 配合 print 语句作为一种**传统的调试方法**。
- **Video Aid for CUDA Kernel Profiling**: 分享了一个有启发性的 [YouTube 视频](https://www.youtube.com/watch?v=LuhJEEJQgUM)，解释了如何在 PyTorch 中对 CUDA kernels 进行 profiling，并提到了使用 `@triton.jit(interpret=True)` 进行调试；然而，另一位成员指出这种方法**已弃用 (deprecated)**。
- **Triton Debugging Best Practices**: 一位成员指向了一个关于**如何调试 Triton kernels** 的 [GitHub issue 讨论](https://github.com/openai/triton/issues/517#issuecomment-1971327089)，从中可以一窥社区解决此类问题的方法。

**Links mentioned**:

- [Lecture 1 How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM): Slides: https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharingCode:  https://github.com/msaroufim/cudamodelecture1
- [How to debug kernels · Issue #517 · openai/triton](https://github.com/openai/triton/issues/517#issuecomment-1971327089): 我正试图准确理解 vector add 教程中 add_kernel 的每一行代码的作用。因为这是一个 kernel，我无法使用典型的单步调试器来遍历这个函数...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1217389117667541073)** (10 messages🔥): 

- **NSight Systems is essential for multi-GPU apps**: 一位成员解释了 **NSight Systems** 对于分析具有多个 GPU 和 CPU 进程的复杂应用程序中性能问题的重要性，理由是它能够处理 PCIe 内存传输和 CPU/GPU 调度问题。

- **Newbie in Need of CUDA Assistance**: 一位成员正在寻求 **CUDA** 问题的帮助，并在 Discord 频道发布了消息。提供了一个参考链接，但无法访问以提取具体信息。

- **Seeking Guidance on NSight Systems**: 一位成员询问了 **NSight Systems** 的实用性，并征求关于指标和教育资源的建议。另一位成员分享了 [Nvidia 的讲座](https://www.nvidia.com/en-us/on-demand/session/gtcspring2021-s31617/) 和一篇解释 NSight Systems 中开销和延迟可视化图表的 [博客文章](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)。

- **Performance Analysis with Nsight Systems Guide**: 一位资深成员强调了使用 **Nsight Systems** 来发现 kernel 启动之间的瓶颈，并提供了一份关于使用 Nvidia Visual Profiler 优化 OpenCV 应用程序的个人指南。该指南可以在[这里](https://cudawarped.github.io/opencv-experiments/nbs/opencv_cuda_streams_performance_python.html)找到。

- **Kernel Launch Overhead Confusion**: 一位成员担心在 CUDA 中更改两个函数的执行顺序时输出会发生变化，推测这可能是由于 CUDA 初始化或 GPU 预热引起的。另一位成员确认这确实是 kernel 启动开销，并建议使用 **Ncu** 来隔离问题。

**Links mentioned**:

- [Accelerating OpenCV with Python and CUDA streams](https://cudawarped.github.io/opencv-experiments/nbs/opencv_cuda_streams_performance_python.html): 使用 Python 和 CUDA streams 优化 OpenCV CUDA 的示例。包括 GPU profiling、分析、性能技巧等！
- [CUDA Developer Tools | Intro to NVIDIA Nsight Systems | NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/session/other2024-cudansight/.): 加入 NVIDIA 的 Sven Middelberg，了解 NVIDIA Nsight Systems，这是一个用于性能调优 NVIDIA GPU 加速应用程序的工具。
- [Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems | NVIDIA Technical Blog](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/): 最近，一位用户在论坛上找到我们。他们发送了一张在 PyTorch 程序上使用 NVIDIA Nsight Systems 的 profiling 结果截图。单次启动逐元素操作导致了...

  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1217477687589736548)** (1 messages): 

- **CUDA Expert Wanted for Learning App**: *Christo_allstreet* 正在为他们的学习应用 [getworldclass.app](http://getworldclass.app) 寻找 **CUDA 专家** 进行咨询工作。有兴趣的专家请发送 **Direct Message** 以获取更多细节。
  

---

**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1217538375351799828)** (2 messages): 

- **Ubuntu 23.10 中的 CUDA Toolkit 难题**：一位用户报告了在 Ubuntu 23.10 上使用 `nvidia-cuda-toolkit` 时遇到的问题，运行 `compute-sanitizer` 会导致错误：*Unable to find injection library libsanitizer-collection.so*。尽管上述库存在于 `/usr/lib/nvidia-cuda-toolkit/compute-sanitizer/libsanitizer-collection.so`，但该工具似乎无法识别它。
- **版本不匹配可能是原因**：另一位用户建议，这个问题可能源于版本不匹配，并指出最新的 NVIDIA toolkit 最高支持到 Ubuntu 22.04。他们建议在 Ubuntu 22.04 上尝试 `compute-sanitizer`，以确定问题是否由新版本操作系统中文件夹路径的变化引起。
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1217561639038226543)** (4 messages): 

- **理解 SM 架构**：一位成员引用了 **CUDA MODE** 书中的第 4.4 节，指出 SM (Streaming Multiprocessor) 如何按照 **SIMD (Single-Instruction, Multiple-Data)** 模型执行 warp 中的线程，并提出了关于单个核心负责执行线程的具体职责问题。
- **关于核心-线程执行的澄清**：另一位成员澄清说，SM 内部的一个处理块（以 **GA102 SM** 为例）一次执行一个 warp，这意味着由于核心限制，32 个线程可以利用 **fp32 instructions** 并发执行，或者分两批执行 32 条 **int32 instructions**。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1217414211177545829)** (10 messages🔥): 

- **Axolotl 配置关键点**：*iron_bound* 注意到运行 **axolotl** 的一个特定要求：必须设置 `pad_to_sequence_len: true`，否则即使是干净的仓库克隆，软件也无法启动。
- **Loss 对比陷入停滞**：*iron_bound* 分享了一份 [W&B report](https://api.wandb.ai/links/iron-bound/v6mxxcj2)，显示了原生 **axolotl vs ring-attn** 的测试结果对比，表明 loss 并没有像预期那样向零下降。
- **对在移动端查看报告问题的担忧**：*andreaskoepf* 提到在移动设备上查看报告存在困难，并寻求澄清原生 **axolotl** 和 **ring-attn** 的 loss 是否都没有趋向于零。
- **基准运行 (Reference Run) 澄清**：*iron_bound* 确认用于对比的基准或参考运行是没有任何代码修改的 **axolotl** 克隆版本。
- **Flash Decoding 工作即将恢复**：*jamesmel* 宣布从明天开始可以继续进行 **Flash Decoding** 的工作。
- **会议不确定性**：*cataluna84* 询问了会议的时间安排，但未提供进一步细节。
- **Axolotl 的补丁分支已发布**：*iron_bound* 提供了 **axolotl** 的 **ring_attention_patching** 分支在 GitHub 上的链接：[GitHub - cuda-mode/axolotl at ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)。

**提到的链接**：

- [Ring-attn vs stock](https://api.wandb.ai/links/iron-bound/v6mxxcj2)：在相同系统和小型数据集上运行了约 100 个 epoch。
- [GitHub - cuda-mode/axolotl at ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)：欢迎就 axolotl 提问。通过在 GitHub 上创建账号为 cuda-mode/axolotl 的开发做出贡献。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1217551604555714721)** (8 messages🔥): 

- **GPT-4 挑战 DOOM**：一篇 [arXiv paper](https://arxiv.org/abs/2403.05468) 探讨了 **GPT-4 玩 1993 年第一人称射击游戏 Doom 的能力**，强调了该模型仅凭基本指令和游戏状态的文本描述进行推理和规划的能力。

- **乡村路带我回家**：一系列消息唤起了歌曲 "Take Me Home, Country Roads" 的歌词，引用了怀旧和自然的主题，如 "Life is old there, Older than the trees" 和 "Rolling like a breeze"。

- **Meta 关于机密文件的法律诉讼**：**Meta** 已对一名前高管提起诉讼，指控其窃取了 100 多份内部文件，并将其用于其 AI 数据初创公司 Omniva。诉讼[详情](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf)涉及该高管从 Meta 过渡到 Omniva 期间“公然不忠且不诚实的行为”。

- **歌词意境被破坏**：一条简短的消息用“……毁了它 (ruined it)”表达了对歌词接龙被中断的失望。

- **Group Learning Initiative**: 一位成员提到了一项涉及三人的协作努力，他们正从 "lecture 1" 开始一段教育旅程。

**Links mentioned**:

- [Will GPT-4 Run DOOM?](https://arxiv.org/abs/2403.05468): 我们展示了 GPT-4 的推理和规划能力可以扩展到 1993 年的第一人称射击游戏 Doom。这个 Large Language Model (LLM) 仅需少量指令就能运行并进行游戏...
- [Meta sues “brazenly disloyal” former exec over stolen confidential docs](https://arstechnica.com/tech-policy/2024/03/meta-sues-brazenly-disloyal-former-exec-over-stolen-confidential-docs/): Meta 的前高管涉嫌向一家神秘的初创公司泄露数据中心机密。

  

---



**DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1217390277103190068)** (1 messages): 

- **Assistant's Mars Explanation Missing Musk's Flair**: 助手的回答因其信息丰富且涵盖了为什么要前往火星的各个方面而受到称赞，但未能完全遵守用户要求的“像 Elon Musk 一样表达”的指令。虽然回复反映了 Elon Musk 对火星探索的观点，但缺乏他特定的风格和语气。**给出的评分为 [[7]]**。
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1217471622924206150)** (8 messages🔥): 

- **MunichNLP Meetup Inquiry**: 一位成员询问是否有兴趣在 4 月 11 日举行慕尼黑见面会讨论 **DiscoLM**，但未收到在活动中发言的直接承诺。
- **DiscoLM Model's German Fine-Tuning Question**: 一位成员询问了 **DiscoLM-mixtral-8x7b-v2** 模型在德语数据集上的 fine-tuning 情况，另一位成员回答说它没有在大量的德语数据上进行训练，并引导其查看 [DiscoLM 70b model](https://huggingface.co/DiscoResearch/DiscoLM-70b) 的详细训练细节。
- **AI Tinkerers in Berlin**: 成员们讨论了即将于 3 月 21 日在柏林举行的 **AI Tinkerers event**，分享了热情以及社区聚会的 [活动链接](https://berlin.aitinkerers.org/)。
- **Seats Filling Up for AI Tinkerers**: 同一位成员提到 **AI Tinkerers** 活动仅剩 8 个席位，表明兴趣度很高且名额有限。
- **Clarity on German Dataset Usage**: 一位成员澄清了自己对 instruction fine-tuning 数据集中是否存在德语数据的困惑，并询问所使用的德语数据比例的具体细节。

**Links mentioned**:

- [
AI Tinkerers - Berlin
](https://berlin.aitinkerers.org/): 未找到描述
- [DiscoResearch/DiscoLM-70b · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM-70b): 未找到描述

  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1217400001525973034)** (1 messages): 

- **Creative Writing Benchmark Testing Success**: 一位成员宣布成功实现了创意写作 benchmark 原型，并表示它提供了合理的排名。感兴趣的各方可以在 [GitHub 上 EQ-Bench 仓库的这个分支](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing) 进行尝试。

**Links mentioned**:

[GitHub - EQ-bench/EQ-Bench at creative_writing](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing): Large Language Models 情感智能的 benchmark - GitHub - EQ-bench/EQ-Bench at creative_writing

  

---


**DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1217552936079720458)** (3 messages): 

- **Seeking German Precision**: 一位成员询问了**最适合德语的 embedding 和 re-ranking**，特别是用于德语法律文本。
- **Hunting for Benchmarks**: 同一位成员还询问是否存在德语 embedding 模型的 benchmark。
- **Benchmarking German Embeddings**: 另一位成员建议使用 **MTEB** Python 包中的 "GermanQuAD" 评估任务，或者关注 **JinaAI** 最近新增的德语支持。
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1217452232207765525)** (2 messages): 

- **Local Model Replication Inquiry**: 一位成员询问如何使用自己的代码在本地复制 demo 的输出。他们目前的设置包括一个 one-shot，配置了 temperature, top_p, max_tokens，并提供了一段代码片段来说明他们的方法。

- **Questions on Command Repetition**:
同一位成员的后续问题询问是否应该为每条用户消息重复命令，还是仅在 system content 中包含一次，寻求关于命令结构最佳实践的指导。
  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1217843643541622935)** (12 messages🔥):

- **Haiku 的成本效益突破**：Haiku 的文档描述器因能以极具经济效益的成本对视觉复杂的文档执行 **vision-to-text** 而受到赞誉。

- **视觉文档处理的讨论**：成员们将 Haiku 的能力与 GPT-vision 进行了对比，结论是 Haiku 并不占优，而另一个名为 Opus 的系统被认为比 Haiku 更好。

- **视觉文档的内容过滤障碍**：讨论显示，在处理文档（尤其是包含方程式的文档）时出现了 **content filtering** 问题，导致文档处理中途分析不完整。

- **Claude 的内容过滤怪癖**：有人提到 **Claude** 历史上一直存在内容过滤不稳定的问题，这可能与成员在文档处理中遇到的问题有关。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1217441897161953381)** (6 messages): 

- **零点击蠕虫攻击 GenAI 驱动的应用**：分享了一篇题为 "ComPromptMized: Unleashing Zero-click Worms that Target GenAI-Powered Applications" 的新论文，强调了通过 prompt injection 攻击 GenAI 驱动应用的漏洞。该论文展示了对使用 **Gemini Pro, ChatGPT 4.0, 和 LLaVA** 等各种模型的邮件助手的攻击。[阅读完整论文](https://sites.google.com/view/compromptmized)

- **寻求模型对比框架**：为了寻找担任代码助手的最佳模型，一位成员询问了用于对比 **Mistral** 或 **Llama2** 等模型效果的框架。

- **基于 Benchmark 选择模型**：另一位成员指出存在用于模型对比的 Benchmark，但建议在参考此类 Benchmark 时应保持一定的谨慎。

- **模型对比排行榜**：为了对比模型，一位成员建议使用 [chat.lmsys.org](https://chat.lmsys.org) 上的 **Leaderboard**，该榜单提供了不同模型的竞争排名。

**Links mentioned**:

[ComPromptMized](https://sites.google.com/view/compromptmized): Stav Cohen Technion - Israel Institute of Technology 

  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1217539473781293056)** (2 messages): 

- **寻求多模态模型专家**：Soniajoseph_ 正在征集擅长多模态模型 **open source interpretability** 的合作者。详情可见其 [Twitter 帖子](https://twitter.com/soniajoseph_/status/1767963316943728779) 以及从 [AI Alignment Forum](https://alignmentforum.org/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) 转发到 [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) 的文章。

- **加入可解释性研究**：感兴趣的人可以通过此 [邀请链接](https://discord.gg/2U2N8QmPmJ) 加入相关的 Discord。

- **协作中心建议**：Rusch 为此类项目提供了一个潜在协作中心的线索，分享了另一个 [Discord 邀请链接](https://discord.gg/bDV7kDrKjE)。

**Links mentioned**:

- [Join the Mech Interp Discord Discord Server!](https://discord.gg/bDV7kDrKjE): 在 Discord 上查看 Mech Interp Discord 社区 - 与其他 907 名成员一起交流，享受免费的语音和文字聊天。
- [Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic): 见证 dogit lens。Patch-level logit attribution 是一种涌现的分割图。点击此处加入我们的 Discord。…

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1217382172432793641)** (1 messages): 

- **寻找速度：Phi 2 推理优化**：一位成员询问了在 **A100 40GB** GPU 上对 **Phi 2** 及其微调版本进行推理的最快方法，表示希望处理“大量数据”。他们征求了关于在 **vLLM**、**Olama**、**Axolotl** 等框架中选择最佳框架的反馈，并想知道 **quantization** 是否对提升速度有益。
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1217447520603471944)** (2 messages): 

- **认识 Devin，自主软件工程师**：分享了一个名为 *Devin The World’s first AI Software Engineer* 的视频，展示了一个名为 Devin 的 AI 的能力，据称它是完全自主的。更多详情可以在 [Cognition Labs 博客](https://www.cognition-labs.com/blog)上找到。

- **Hermes 2 Pro 7B 的 Function Calling**：聊天中包含了一个 [YouTube 视频](https://youtu.be/PzaidfqDtGI)，演示了使用 **Hermes 2 Pro 7B** 模型进行 Function Calling。感兴趣的观众可以通过 [Hermes Function Calling 专用 GitHub 仓库](https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels) 了解更多信息并深入研究细节。

**提到的链接**：

- [Devin：全球首位 AI 软件工程师](https://www.youtube.com/watch?v=NSPtrrUQ_fw)：Devin 是全自动软件工程师 https://www.cognition-labs.com/blog
- [让我们使用 Hermes 2 Pro 7B 进行 Function Calling](https://www.youtube.com/watch?v=PzaidfqDtGI)：让我们使用 Hermes 2 Pro 7B 进行 Function Calling https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels

  

---



**AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1217550187204055070)** (1 条消息): 

- **代码领域的新成员**：Cognition 发布了 **Devin**，这是一款被定位为全球**首位全自动 AI 软件工程师**的 AI。正如 [Scott Wu 的博客文章](https://www.cognition-labs.com/blog) 中所述，他们声称 Devin 可以处理复杂的工程任务，随时间学习，并纠正自己的错误。

**提到的链接**：

[Blog](https://www.cognition-labs.com/blog)：未找到描述

  

---


**AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1217633669275975731)** (1 条消息): 

- **Voice + AI 活动机器人竞赛**：作为下周即将举行的 **Voice + AI 活动** 的趣味补充，官方宣布了一项竞赛，邀请参与者构建创意项目。“**世界上最有趣的机器人竞赛**”的详细信息可以在其 [Notion 页面](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3) 中找到。

**提到的链接**：

[Notion – 笔记、任务、维基和数据库的一体化工作空间。](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3)：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。