---
companies:
- deepmind
- cognition-labs
- deepgram
- modal-labs
- meta-ai-fair
- anthropic
date: '2024-03-14T01:07:46.703107Z'
description: '**DeepMind SIMA** 是一款面向 3D 虚拟环境的通用 AI 智能体。它仅通过屏幕截图和自然语言指令，在 9 款游戏的 **600
  个任务**中进行了评估，实现了 **34%** 的成功率（人类为 **60%**）。该模型采用了多模态 Transformer 架构。


  **Andrej Karpathy** 概述了软件工程中 AI 自主性的演进过程，而 **Arav Srinivas** 则对 Cognition Labs 的
  AI 智能体演示表示赞赏。**François Chollet** 对完全自动化软件工程持怀疑态度。**Yann LeCun（杨立昆）** 建议摆脱生成式模型和强化学习，转向实现人类水平的
  AI。


  **Soumith Chintala** 和 **Yann LeCun** 分享了 Meta 的 **Llama-3** 训练基础设施，其中包括 **2.4 万块
  H100 集群节点**。**Deepgram 的 Aura** 推出了低延迟语音 API，而 **Modal Labs 的 Devin AI** 展示了文档导航以及与
  ComfyUI 的交互功能。此外，AI 社区中也流传着各种梗和幽默。'
id: b0f21719-f41d-4a1e-ae08-1a52d609fd36
models:
- llama-3
- claude-3-opus
- claude-3
- gpt-3.5-turbo
original_slug: ainews-deepmind-sima-one-ai-9-games-600-tasks
people:
- andrej-karpathy
- arav-srinivas
- francois-chollet
- yann-lecun
- soumith-chintala
- john-carmack
title: DeepMind SIMA：一个 AI，9 款游戏，600 个任务，仅限视觉+语言。
topics:
- multimodality
- transformer
- software-engineering
- ai-agents
- ai-infrastructure
- training
- text-to-speech
- speech-to-text
- real-time-processing
- model-architecture
- benchmarking
---

 


---

**目录**

[TOC] 

---

# PART X: AI Twitter 综述

> 所有综述由 Claude 3 Opus 完成，取 4 次运行中的最佳结果


**软件工程自动化**

- [Andrej Karpathy 概述了](https://twitter.com/karpathy/status/1767598414945292695)软件工程中 AI 自主性不断提高的进程，类似于自动驾驶，AI 完成更多工作，而人类在更高的抽象层级进行监督。
- [Arav Srinivas 赞扬了 Cognition Labs 的演示](https://twitter.com/AravSrinivas/status/1767582756291387484)，称其为第一个可靠跨越人类水平性能门槛的 Agent。
- [François Chollet 认为我们距离能够](https://twitter.com/fchollet/status/1767674774107611137)自动化他作为软件工程师工作的极小部分还很遥远。

**大语言模型与 AI 架构**

- [Yann LeCun 建议在通往人类水平 AI 的道路上](https://twitter.com/ylecun/status/1767681700421677445)，（至少部分地）放弃生成模型、概率建模、对比方法和强化学习。
- [François Chollet 分享了他对先天与后天的看法](https://twitter.com/fchollet/status/1767526290436096325)，指出人类从一开始就是智能的，几乎所有知识都是习得的，且智能随年龄增长而下降。
- [Andrej Karpathy 推荐了一份 AI 通讯](https://twitter.com/karpathy/status/1767616494752731633)，由 @swyx 及其朋友编写，该通讯利用 LLM 辅助索引了约 356 个 Twitter、21 个 Discord 等。（swyx：感谢 Andrej！！！）

**AI Agent 与演示**

- [Cognition Labs 关于 AI Agent](https://twitter.com/AravSrinivas/status/1767750787269345675) 解决编程任务的演示给 AI 社区留下了深刻印象。
- [Deepgram 的 Aura 提供快速的](https://twitter.com/svpino/status/1767586456036417627)文本转语音和语音转文本 API，延迟低于 250 毫秒，支持实时对话式 AI 应用。
- [Modal Labs 的 Devin AI 在浏览器中导航文档](https://twitter.com/akshat_b/status/1767579399317029211)、安装、身份验证并与 ComfyUI 部署进行交互。

**AI 基础设施与训练**

- [Soumith Chintala 分享了](https://twitter.com/soumithchintala/status/1767579981419315400) Meta 用于 Llama3 训练的 24k H100 集群 Pod 的细节，包括网络、存储和软件优化。
- [Yann LeCun 分享了一张图片](https://twitter.com/ylecun/status/1767591599486193793)，展示了用于 Llama-3 训练的计算基础设施。
- [John Carmack 指出](https://twitter.com/ID_AA_Carmack/status/1767553799722320103)，由于算法细节和训练/测试程序的细微变化，在研究中信任对比结果是很困难的。

**迷因与幽默**

- Arav Srinivas 分享了[“猜猜提示词”](https://twitter.com/AravSrinivas/status/1767750787269345675)迷因。
- [暗示 AI 开发者之间竞争](https://twitter.com/nearcyan/status/1767677053686735304)的迷因。
- @AISafetyMemes 分享了[“完蛋了”迷因](https://twitter.com/AISafetyMemes/status/1767574804771844113)。


---

# 第 0 部分：总结之总结之总结

> 由于 [Claude 3 Haiku 最近发布](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww)，我们将其添加到本次总结运行中供您比较。在构建 AINews 平台以提供更好 UX 的同时，我们将继续并排运行这些模型一段时间。

## Claude 3 Haiku (3B?)

- **Devin，AI 软件工程师**：Cognition Labs 推出了 Devin，这是一个能够执行复杂工程任务并在 SWE-bench 编码挑战中设定新基准的自主 AI 软件工程师。Devin 可以操作 shell、代码编辑器和 Web 浏览器来完成软件工程工作，展示了 AI 自动化软件开发的潜力。Devin 的能力在 AI 社区引起了极大的关注。
- **大型语言模型能力的进展**：研究人员正在探索增强 Large Language Model (LLM) 性能和效率的方法。这包括开发像 **MemGPT** 这样的方法来改进长期记忆管理，以及发布像 **Command-R** 这样针对长上下文任务和 Retrieval-Augmented Generation 优化的模型。人们对 GPT-4 等专有模型的开源替代方案也表现出浓厚兴趣。
- **AI 硬件和基础设施**：Meta 和 Microsoft 等大型科技公司正在对 AI 硬件和基础设施进行重大投资。Meta 宣布计划在 2024 年前构建两个 24,000 GPU 集群并采购 350,000 个 NVIDIA H100 GPU，而 Microsoft 则聘请了核能专家，探索使用小型模块化反应堆为其数据中心供电。CUDA 和 NVMe SSD 集成的进步也为 AI 工作负载实现了更高效的数据处理。
- **围绕 AI 模型的挑战和讨论**：社区正在应对与 AI 模型相关的各种问题，例如：
    - **Prompt Engineering**：开发有效的 Prompt 来控制语言模型输出，包括保持一致的字数和格式。
    - **准确性和透明度**：对 AI 模型输出的可靠性和可信度的担忧，特别是在使用在线或实验性模型时。
    - **开源替代方案**：围绕 Grok AI 和 GemMoE 等开源 AI 模型可能发布的兴奋与怀疑，以及关于模型安全性和与现有工具集成的讨论。

## Claude 3 Sonnet (14B?)


**1) 介绍像 [Devin](https://www.cognition-labs.com/blog) 这样的自主 AI 软件工程师**: 

- **[Cognition Labs](https://www.cognition-labs.com/blog)** 发布了 **Devin**，声称是全球首位 **自主 AI 软件工程师**，能够胜任复杂的工程任务，如解决 GitHub issue、使用开发者工具，并在 **[SWE-Bench](https://x.com/cognition_labs/status/1767548763134964000)** 上刷新了基准测试记录。
- Devin 的亮相引发了轰动，用户们热衷于尝试并分享对其在 [推文](https://x.com/itsandrewgao/status/1767576901088919897) 中展示的实际能力以及在 SWE-Bench 上预期表现的“直言不讳的评价”。

**2) 大语言模型 (LLM) 与 AI 硬件的进展**:

- **[Anthropic](https://x.com/anthropicai/status/1768018310615151002)** 发布了 **Claude 3 Haiku**，这是一款快速且经济的多模态模型，现已在其 API 和面向 Pro 用户的 claude.ai 上线，在推理、数学和编程任务中表现强劲。
- **[Meta](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)** 宣布到 2024 年将建成两个使用 350,000 块 **NVIDIA H100** GPU 的大规模 24k GPU 集群，标志着在 AI 基础设施方面的重大投资，并公布了硬件、网络和软件的细节。
- **[Cerebras](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)** 推出了 **CS-3** 芯片，声称是全球最快的 AI 加速器，能够在单台设备上训练高达 24 万亿参数的模型。

**3) 开源 AI 模型发布与基准测试工作**:

- **[CohereForAI](https://huggingface.co/CohereForAI/c4ai-command-r-v01)** 发布了开源的 35B 参数模型 **Command-R**，该模型针对推理、摘要和多语言生成进行了优化。
- **[EQ-Bench 创意写作基准测试](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing)** 正式上线，支持对模型的创造力进行排名，尽管该项目仍在开发中。
- **[DiscoResearch](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1)** 微调了德语模型，而 **[Mistral Medium](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)** 模型展示了优于 Mixtral 的强劲性能。

**4) 高效大模型训练的框架与工具**:

- **[Fuyou 框架](https://x.com/_akhaliq/status/1767393991727657262)** 支持在 RTX 4090 等消费级 GPU 上微调高达 175B 参数的模型，承诺比 ZeRO-Infinity 具有更高的效率提升。
- **[Unsloth AI](https://github.com/unslothai/unsloth)** 为 Mistral 7B、CodeLlama 34B 和 Llama 架构模型提供 QLoRA 支持，具备滑动窗口注意力机制（sliding window attention）和初步的 DPO 支持。
- **[CUDA](https://github.com/enfiskutensykkel/ssd-gpu-dma)** 的新进展允许 NVMe 驱动器通过 NVIDIA 的 **[GPUDirect® Storage API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)** 在存储应用中使用直接内存访问和 GPU 集成。

## Claude 3 Opus (>220B?)

- **Hermes 2 Pro 7B 模型发布，具备先进功能**：Nous Research 发布了 **Hermes 2 Pro 7B** 模型，拥有 91% 的 Function Calling 准确率，并支持高级 Function Calling 和结构化 JSON 输出。该模型和数据集可在 [Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) 和 [GitHub](https://github.com/NousResearch/Hermes-Function-Calling) 上获取。

- **欧盟 AI 法案引发争议和合规担忧**：**EU AI Act** 已获通过，禁止某些 AI 实践，强制要求披露能源消耗，并可能对在欧洲运营的 AI 公司产生影响。讨论围绕其对 AI 行业的影响以及监管执行的挑战展开。

- **Cognition Labs 推出 AI 软件工程师 Devin**：**Cognition Labs** 推出了 **Devin**，这是一款自主 AI 软件工程师，在 **SWE-Bench 编码基准测试**中解决了 13.86% 的 GitHub 问题，超越了基准。Devin 的能力和潜在影响引发了热议，详见其 [博客文章](https://www.cognition-labs.com/blog) 和 [演示视频](https://www.youtube.com/watch?v=NSPtrrUQ_fw)。

- **Cerebras 宣布推出全球最快的 AI 芯片**：**Cerebras Systems** 推出了 **CS-3**，声称是全球最快的 AI 加速器，能够在单个设备上训练高达 24 万亿参数的模型。正如其 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) 中所分享的，该公告引发了关于芯片设计和推动 AI 计算技术潜力的讨论。

- **Anthropic 发布 Claude 3 Haiku**：**Anthropic** 发布了 **Claude 3 Haiku**，这是一款快速且经济实惠的 AI 模型，因其速度（每秒 120 个 token）和成本效益（每 1 美元 400 万个 prompt token）而受到赞誉。有关该模型的能力和潜在应用的讨论详情可见 Anthropic 的 [公告](https://www.anthropic.com/news/claude-3-haiku)。

- **Meta 大力投资 AI 基础设施**：**Meta** 宣布了两个 24k GPU 集群的计划，并目标在 2024 年前集成 350,000 个 NVIDIA H100 GPU，这标志着对其 AI 未来的重大投资。该公司在一篇 [工程博客文章](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/) 中分享了有关其硬件、软件和开源计划的细节。

- **CUDA 与 NVMe 集成的进展**：**CUDA** 的发展现在允许 **NVMe 驱动器**利用直接内存访问和 GPU 集成进行存储应用，有望实现显著的效率提升。[ssd-gpu-dma GitHub 仓库](https://github.com/enfiskutensykkel/ssd-gpu-dma) 和 NVIDIA 的 [GPUDirect Storage API 参考指南](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) 等资源重点展示了这些进展。

- **从生产语言模型中窃取权重**：一篇 [研究论文](https://not-just-memorization.github.io/partial-model-stealing.html) 揭示了利用 API 从 **ChatGPT** 和 **PaLM-2** 等生产语言模型中推断权重的可能性，引发了对 AI 伦理和保护模型权重的未来安全措施的担忧。

## ChatGPT (GPT4T)

<div><ul><li><p><strong>AI in Game Development</strong>：<strong>Nous Research AI Discord</strong> 探讨了 AI 在游戏开发中的作用，特别是使用 <strong>Claude 3</strong> 创作《植物大战僵尸》，强调了 AI 与 Python 在游戏创作中的结合。该项目展示了 AI 驱动的游戏设计创意，可在<a target="_new" href="https://www.youtube.com/watch?v=d7NGgglZXK8">此处</a>观看。</p></li><li><p><strong>Advancements in Function Calling</strong>：<strong>Hermes 2 Pro 7B</strong> 模型的推出，凭借 <em>91% 的 Function Calling 准确率</em>，标志着 AI 执行函数调用能力的重大进步，使其成为开发者将 AI 与 <em>llama.cpp</em> 和 <em>Vercel AI SDK</em> 等编程环境集成的显著工具。这一进展是通过结构化 JSON 输出将 AI 与编程融合的更广泛努力的一部分，详见 <a target="_new" href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub</a>。</p></li><li><p><strong>The AI Chip Debate</strong>：<strong>Cerebras CS-3</strong> 芯片被誉为世界上最快的 AI 加速器，引发了关于其方形设计在单台设备上训练 24 万亿参数模型效率的讨论。这次讨论强调了 AI 计算技术的持续创新，更多信息请访问 <a target="_new" href="https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine">Cerebras</a>。</p></li><li><p><strong>New Developments in AI Software Engineering</strong>：<strong>Cognition Labs 的 Devin</strong> 作为一名新的 AI 软件工程师出现，能够自主解决 GitHub 问题，并展示了熟练操作各种开发工具的能力。这标志着 AI 集成到软件开发中的一个重要里程碑，预示着未来自动化编程任务的能力，正如在 <a target="_new" href="https://fxtwitter.com/cognition_labs/status/1767548763134964000">Cognition Labs Devin</a> 所展示的那样。</p></li><li><p><strong>AI-Powered Solitaire Instruction</strong>：一个利用 <strong>OpenCV 进行纸牌教学</strong>的项目体现了 AI 在休闲游戏中的潜力，旨在开发一个使用图像捕捉和处理来指导游戏过程的系统。这一举措反映了 AI 的应用正在从传统领域扩展到增强游戏用户体验，预计未来的开发将与 GPT 等语言模型集成，以进行更深层次的游戏分析。</p></li></ul></div>

---

# PART 1: 高层级 Discord 摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

**在游戏开发领域播种 AI**：展示了一个使用 **Claude 3** 创作《植物大战僵尸》的 AI 驱动项目，因其在 Python 游戏开发中的应用而引起关注——点击[此处](https://www.youtube.com/watch?v=d7NGgglZXK8)观看创意的展开。

**Function Calling 成为新时尚**：最近发布的 **Hermes 2 Pro 7B** 模型显示出重大进步，具有 *91% 的 Function Calling 准确率*和支持高级函数调用的专门提示词。该模型的能力得到了认可，并被积极集成到 *llama.cpp* 和 *Vercel AI SDK* 等工具中，寻求结构化 JSON 输出的新融合 [GitHub - Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling)。

**AI 芯片形状之争**：被宣称为世界上最快 AI 加速器的新型 **Cerebras CS-3** 的最佳设计是什么？围绕其方形芯片设计的讨论不断，而该模型号称准备好在单台设备上训练庞大的 24 万亿参数模型，代表了 AI 计算技术的飞跃 [Cerebras](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)。

**AI 规则正在改变**：欧盟新的 **AI Act** 通过禁止某些 AI 实践并要求披露能源消耗，给 AI 公司带来了挑战。与此同时，人们对专注于长上下文聊天机器人的开源模型充满期待，如 [Sparse Distributed Associative Memory 仓库](https://github.com/derbydefi/sdam)中所述。

**Cognition 为该领域引入新玩家**：**Devin** 登场，这是一位 AI 软件工程师，声称在自主解决 GitHub 问题方面树立了新的成功基准，展示了操作 Shell、代码编辑器和 Web 浏览器的能力——在 [Cognition Labs Devin](https://fxtwitter.com/cognition_labs/status/1767548763134964000) 窥见未来。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **介绍 AI SWE Devin**：**Cognition Labs** 推出了 **Devin**，这是一款 AI，它通过解决显著的 13.86% 的 GitHub issues 并在工程面试技能方面给人留下深刻印象，树立了新的标杆。行业关注点集中在 Devin 的能力上，这些能力通过 **SWE-Bench** 性能指标得到了增强。[Cognition Labs](https://x.com/cognition_labs/status/1767548763134964000?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 提供了关于 Devin 开发和功能的进一步见解。

- **模型权重安全性受到质疑**：研究人员揭示了从 **ChatGPT** 和 **PaLM-2** 等模型的 API 中推断权重的可行性，引发了关于 AI 伦理和未来保护策略的辩论。相关影响在[最近的一篇论文](https://not-just-memorization.github.io/partial-model-stealing.html)中进行了详细阐述，该论文概述了模型权重提取的潜在风险和方法。

- **扩展 AI 视野**：**Together.ai** 获得了 1.06 亿美元的融资，用于创建一个旨在运行生成式 AI 应用的新平台，并推出了 **Sequoia**，这是一种旨在高效运行 LLM 的方法。同时，**Cerebras** 发布了 **CS-3** AI 芯片，声称拥有针对超过 24 万亿参数模型的最快训练能力。有关 Sequoia 和 CS-3 进展的详细信息可以分别在 [Together AI 的博客](http://together.ai/blog/series-a2)和 [Cerebras 的新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)中查阅。

- **合成数据的成功**：在 LLM Paper Club 活动期间，Eugene Yan 强调了**合成数据 (synthetic data)** 在模型训练（包括 pretraining 和 instruction-tuning）中的有效性，突出了其成本效益和规避隐私问题的优势。鼓励社区阅读他关于 [synthetic data](https://eugeneyan.com/writing/synthetic/) 的见解。

- **探索语音识别的潜力**：社区探索了**语音识别 (voice recognition)**，重点关注使用 **vapi.ai** 和 **whisper** 等工具的 text-to-speech 和 speech-to-text 应用。[Twitter 线程](https://twitter.com/swyx/status/1765995892107317407)和 [whisper.cpp](https://github.com/bminixhofer/whisper.cpp) 等资源展示了语音技术领域正在进行的讨论。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Claude3 Opus 驱动 Perplexity AI**：确认 Perplexity AI 在其运营中使用 *Claude3 Opus*。用户探索了该平台的生产力和有效研究辅助能力，同时对 AI 剽窃检测工具的局限性表示担忧，并寻求关于各种 Perplexity AI 服务的澄清，包括在 5 次使用后从 Claude 3 Opus 切换到 Claude 3 Sonnet 的情况。

- **搜索引擎演进讨论**：社区参与了关于搜索引擎未来的讨论，指出 Perplexity AI 使用了**其自有的索引器 (indexer)**。与此同时，有人提到要求 Google CEO Sundar Pichai 辞职的呼声日益增高，以及来自生成式 AI 对手的搜索领域竞争加剧。

- **增强 AI 交互性**：成员们分享了关于从睡眠建议到理解医学术语等主题的 Perplexity AI 搜索查询链接。此外，一段 [YouTube 视频](https://www.youtube.com/watch?v=GTxNncK47Sk)的分享表明了对最新 AI 新闻的兴趣，包括 AI 生成的“数字玛丽莲 (Digital Marilyn)”以及 Midjourney 与 Stability AI 之间的争议。

- **API 导航与定制策略**：**#[pplx-api]** 频道中的查询反映了对通过定制 prompting 和参数设置获得简洁回答的关注。还有关于通过使用外部数据库存储 embeddings 来使聊天机器人模型记住对话的讨论，以及要求在 API 中增加 **Yarn-Mistral-7b-128k** 以应对高上下文 (high-context) 使用场景的呼吁。

- **AI 回答的准确性与伦理**：社区对使用 *-online* 标签时 AI 返回的不准确信息表示担忧，从而引发了加入 **system message** 以提高清晰度的建议。为了进一步理解和实验 system message，参考了 [Perplexity API 文档](https://docs.perplexity.ai/reference/post_chat_completions)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord 总结

- **RoPE Kernel 效率提升**：分享了一个 **RoPE Kernel** 的改进，重点在于仅加载一次 sin/cos 函数，以提高轴 1 的效率，并采用顺序计算进行头分组（head grouping）。尽管有相关的 PR，但对其对整体训练性能影响的担忧依然存在。

- **Unsloth 的微调技巧**：Unsloth Studio（一个支持一键微调的功能）据报道正处于 beta 阶段，而 Unsloth 社区通过手动重新安装库解决了在 Kaggle 上导入 FastLanguageModel 的问题。Unsloth 的 GitHub wiki 提供了 FAQs 以及针对 DPO 对样式数据的格式化指南。

- **对开源 OpenAI 的质疑**：社区对 GemMoE 和 Grok AI 等开源 AI 模型的未来既感到兴奋又感到担忧。讨论内容包括 OpenAI 是否会加入开源运动，同时也对潜在的模型盗版以及与 Microsoft Azure 的关联感到担忧。

- **克服 Ubuntu WSL 的依赖悖论**：用户讨论了 **Ubuntu WSL** 中涉及 bitsandbytes、triton、torch 和 xformers 的依赖循环，并确定了一个阻止包安装的 Python 错误。建议的解决方法包括直接从 PyTorch 的 cu121 索引安装 xformers，以及求助于 Unsloth 的 nightly GitHub 构建版本。

- **量化 Mixtral 的烦恼**：在三块 V100 32GB GPU 上量化 Mixtral 模型的尝试遇到了显存瓶颈，社区给出了有用的建议，如尝试 2x A100 80GB GPU 或利用 Unsloth 内置的 GGUF 支持。同时，关于 Nous-Hermes 模型分词（tokenization）的问题也浮出水面，通过更换模型和调整数据集得到了解决。

- **模型微调的存储创新**：AI 爱好者分享了一篇[论文](https://arxiv.org/abs/2403.06504)，探讨了使用 NVMe SSDs 克服 GPU 显存限制以微调大型模型的可能性。GitHub 上的 [ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma) 和 [flashneuron 项目](https://github.com/SNU-ARC/flashneuron/blob/master/README.md)等有用资源被引用为对 GPU 存储应用的贡献，此外还提到了 NVIDIA 的 [GPUDirect® Storage API 文档](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)以获取直接数据路径 API。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

**LaTeX 渲染引发工程师的热情**：讨论强调了希望 **LM Studio** 在 markdown 中支持 LaTeX 的愿望，正如在 [GitHub 博客文章](https://github.blog/changelog/2022-05-19-render-mathematical-expressions-in-markdown/)中所见，旨在改进数学问题的交互界面。成员们考虑加入滑动式黑板进行视觉数学输入，在严肃的技术追求中展现出轻松的基调。

**GPU 性能揭秘**：分享的 [YouTube 性能测试](https://youtu.be/RO1KnARCHcc)引发了关于为大型语言模型（LLMs）使用双 GPU 配置（包括双 RTX 4060 Ti 16GB GPU 设置）的好处和技术考量的讨论。一些成员注意到效率提升超过两倍，而另一些成员则分享了优化配置的技巧，甚至幽默地交流了关于高价 NVLINK 桥接器及其替代方案的看法。

**合作还是单干？双 GPU 配置 vs 单 GPU**：分享个人测试结果的工程师们仔细研究了在单 GPU 与双 GPU 设置上运行 LLMs 的有效性。讨论范围从针对显存（VRAM）不匹配的 GPU 进行配置调整，到探索为多个高端 GPU 供电的可行性。

**升级 RAM 以迎接更智能的未来**：为了运行更大的 LLM 模型，升级到 128GB RAM 的考虑与对更多 VRAM 的需求进行了权衡。社区成员提供了关于硬件配置和性能修改的见解，包括运行多个并发 LM Studio 实例的技巧，以及通过 AMD 的 ROCm beta 增强 GPU 加速。

**禁用 iGPU 以增强主 GPU 性能**：**amd-rocm-tech-preview** 频道中的用户发现，禁用 iGPU 可以解决卸载（offloading）问题。他们交流了使用 AMD GPU 优化 ROCm beta 的策略和建议，从安装特定的驱动组合（如 Adrenalin 24.1.1 + HIP-SDK）到清理缓存目录以在 LM Studio 中实现更好的模型加载。

**AVX Beta 低调发布**：在 **🧪-beta-releases-chat** 中，简要提到了 AVX beta 的版本更新，并对某些未指明主题的质量进行了极少的讨论，有人表达了“*不太好*”的观点。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **AI 辅助发牌**：一个开发**使用 OpenCV 的纸牌游戏（Solitaire）教学机器人**的项目被提出，设想是一个通过捕捉笔记本电脑屏幕图像来指导游戏进行的机器人系统。构建该系统包括研究游戏规则、组装硬件、编写图像处理算法，以及与 GPT 等语言模型集成以进行深入的游戏分析和迭代设计改进。

- **视频生成 AI Sora 访问权限受限**：视频生成模型 **Sora** 被提到成本高昂且尚未向公众开放测试。然而，据报道，视觉艺术家和电影制作人已有机会访问并提供反馈，这表明 Sora 在特定用户群体中进行了有限的测试发布。

- **GPT 的经济壁垒与停机应对方案**：讨论了 **GPT-4 订阅**等 AI 服务的成本，强调了相对于**格鲁吉亚**和**巴西**等国家最低工资而言，这一支出比例过高。此外，在 GPT 出现故障时，用户建议查看 [OpenAI 状态页面](https://status.openai.com/)，并考虑开启新对话作为可能的临时解决方案。

- **提示工程（Prompt Engineering）实现字数一致性**：在**提示工程**领域，用户交流了如何指导 AI 在文本改写过程中保持字数的策略。建议使用*正面指令*且不过分规定细节，以获得一致的结果；同时确认 **Code Interpreter** 能够按照常规字数统计标准准确计算字数。

- **定制化 AI 开发难题**：个人讨论了**为个人用途自托管模型**（如处理日志数据），重点在于理解硬件需求。对于增强能够查阅 PDF 和进行网络搜索的 **CustomGPT** 模型，建议必须提供清晰的指令，因为 AI 在识别 PDF 文档中的图像方面存在局限性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **Claude 与 GPT 在学术摘要方面的对比**：测试了 Claude 3 对学术论文的总结能力，认为其在通用摘要方面表现良好，但在深入细节方面表现不足。讨论中还包括一名来自 EleutherAI 的 Polyglot 和 OSLO 项目的 ML Engineer 正在求职、一位经验丰富的机器学习新人的自我介绍，以及参考 [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) 对模型训练效率中 Batch Size 权衡的辩论。

- **优化迷思与多模型效率**：一名成员破除了关于 beta2 和训练稳定性的迷思，建议使用 ADAM 等替代方案；同时审议了 LoRA 扩展到预训练（Pre-training）的潜力，并展望其与 GradientLoRA 等方法协同工作的可能性。此外，一篇关于 [Deep Neural Collapse](https://arxiv.org/abs/2402.13728) 的论文为有关神经网络训练轨迹和架构的对话增添了内容。

- **模型黑客伦理与剪枝视角**：在辩论模型黑客行为（Model Hacking）的伦理边界时，一名成员参考一篇已发表的论文指出，事先获得许可使模型黑客行为在伦理上是可以接受的。此外，Pythia 因其解释性驱动的能力遗忘（Unlearning）而受到关注；同时宣布了一个针对多模态模型的新机械解释性（Mechanistic Interpretability）库，并通过 [Twitter 公告](https://twitter.com/soniajoseph_/status/1767963316943728779) 邀请合作。

- **排行榜备受质疑**：在 AI 性能的竞技场中，讨论了 SQuAD 等基准测试的局限性。GPQA 数据集引发了对 AI 模型辅助价值的重新评估，即使是 GPT-4 在该数据集上也面临困难，这强调了更强大的监督（Supervision）的必要性，如 [Anthropic 的论文](https://arxiv.org/pdf/2311.12022.pdf) 所示。

- **Megatron 与 GPT-NeoX 的同步考量**：一名成员提议密切跟踪上游 Megatron，以便更好地与其 Transformer Engine 保持一致，相关的代码差异已在 [GitHub pull request](https://github.com/EleutherAI/gpt-neox/pull/1185) 中提交，等待项目维护者的反馈。

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **游说者的魅力还是钱包的伤害？**：游说者的权力被认为源于他们通过资金捐助产生的影响力，这导致了协商，而不是从他们分发的资金中个人获益。

- **终结“终结者”情景**：围绕 AI 可能引发的灾难性情景展开了一场轻松的讨论，包括对经过 finetuned 的 AI 导致灭绝事件的推测，并伴随着对过去灭绝事件失败的讽刺评论。

- **AI 霸主：不太可能但值得讨论**：对于 AI 导致政府权力过度扩张的可能性存在怀疑，成员们讽刺地否定了可能导致这种情景的协调行动的可能性。

- **导航 AI 监管的未来**：围绕 AI 生成的模型权重（model weights）的版权侵权、DMCA 的有效性以及欧盟新 AI 法规的影响展开了辩论，并链接到了 [European Lawmakers' Act](https://www.cnbc.com/2024/03/13/european-lawmakers-endorse-worlds-first-major-act-to-regulate-ai.html)。

- **为 AI 高级用户选择最佳工具**：讨论还涉及了 AI inference 任务的硬件和软件偏好，重点是 GPU 的使用以及本地设置与以 API 为中心的解决方案的效率对比。提到了 [GroqChat](https://groq.com/)、[Meta's AI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/) 和 [BUD-E](https://youtu.be/bLPDn-bh7dY?si=xrR2_F6kx1ydz8XM) 等平台和技术。

- **制止幻觉（Hallucinations）**：建议在调用 **CogVLM** 生成数据时使用简单且短的 prompts，以尽量减少错误输出的产生。

- **AI 中的注意力艺术**：共识表明 cross attention 可能不是向模型添加 conditioning 的最佳机制，因为涉及转换 text embeddings 的替代方法被证明能获得更好的 denoising 结果。

- **将 AI 与 MoAI 结合**：对新的 **Mixture of All Intelligence (MoAI)** 模型表现出了明显的热情，该模型承诺在保持较小占用空间的同时优于现有模型，资源可在 [GitHub](https://github.com/ByungKwanLee/MoAI) 和 [Hugging Face](https://huggingface.co/BK-Lee/MoAI-7B) 上获得。

- **用户数据集引用庆典**：一位成员分享了他们的数据集在 **DeepSeekVL** 论文中被引用的兴奋之情，展示了社区对推进 AI 研究的贡献。

- **打破巨大的内存神话**：关于能够使用 lazy loading 将 30B 模型加载到 4GB 内存中的说法得到了纠正，揭示了 mmap 在访问之前掩盖了实际的内存使用情况。

- **深入研究 LAION-400M**：一条消息赞扬了 Thomas Chaton 的文章，该文章指导用户利用庞大的 LAION-400-M 图像和标题数据集，并链接到该[文章](https://bit.ly/3uYrDCh)以获取更多见解。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **MemGPT 席卷长期记忆前沿**：**MemGPT** 正在解决 LLM 长期记忆这一长期存在的难题，引入了“虚拟上下文管理”和 function calling 等新功能来增强记忆性能。不要错过通过[此处](https://lu.ma/c39w9ofy)注册 MemGPT 网络研讨会来深入了解这些进展的机会。

- **巴黎开源 AI 开发者聚会**：**Ollama and Friends** 将于 **3 月 21 日在巴黎 Station F** 为开源 AI 开发者铺设红地毯，现场将提供食物、饮料和演示。请通过 [Twitter](https://t.co/DTyUwHpdC7) 联系以锁定席位或申请进行演示。

- **LlamaIndex 与 MathPix 调制科学搜索利器**：**LlamaIndex** 与 **MathPixApp** 的合作旨在将科学查询提炼为 LaTeX 精髓，承诺通过文档索引实现卓越的搜索能力。对于好奇的开发者，可以通过 [Twitter 信号](https://t.co/rWX6rGIHub)获取这份包含图像提取和文本索引的指南。

- **聊天机器人召唤与索引咒语**：社区内关于提升聊天机器人响应和索引的讨论非常热烈，建议将 **DeepEval** 与 **LlamaIndex** 结合使用以进行性能优化，并部署集成查询引擎（ensemble query engines）来策划多样化的响应。对于热衷于获取洞察的人，可以在[此处](https://docs.confident-ai.com/docs/getting-started)和[此处](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/ensemble_query_engine.ipynb)找到相关指南。

- **LLM 论文合集问世**：由 shure9200 策划的一个庞大的 LLM 研究库即将出现。学者们可以前往[这个宝库](https://shure-dev.github.io/)发掘最近的学术进展，并在 LLM 领域开辟新路径。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

**Axolotl 采用 DoRA 进行低比特量化**：对 4-bit 和 8-bit 量化模型的 **DoRA (Differentiable Quantization of Weights and Activations)** 支持已成功合并，承诺带来性能提升，尽管目前仅限于线性层且有明显的额外开销。感兴趣的工程师可以在 [GitHub 上查看合并详情](https://github.com/huggingface/peft/pull/1518/files/3f35dd59bc937ec39d4a0f9dd5a5365209741f75..fd63e3c831e4a1250580799d9c9d107293ee2ffd)。

**大模型，小 GPU —— Fuyou 前来救场**：*Fuyou* 框架展示了让工程师在 **RTX 4090 等标准消费级 GPU 上微调高达 1750 亿参数巨型模型**的潜力，引发了受硬件限制开发者的关注。[_akhaliq 的推文](https://x.com/_akhaliq/status/1767393991727657262?s=20)评价了这一令人兴奋的进展，展示了 156 TFLOPS 的计算能力。

**DeepSpeed 的 API 演进**：**DeepSpeed** 引入了一个用于将模块设置为叶子节点的 API 修改，这可能使处理 MoE 模型变得更加容易，从而可能使 Axolotl 的开发计划受益。更多信息请见其 [GitHub PR](https://github.com/microsoft/DeepSpeed/pull/4966)。

**Command-R 携 35B 参数量砥砺前行**：由 CohereForAI 创建的 Command-R 是一个**开源的 350 亿参数模型**，它针对多种用例进行了优化，并在 [Huggingface](https://huggingface.co/CohereForAI/c4ai-command-r-v01) 上开放访问，开辟了新的前沿。

**Mistral Medium 表现优于 Mixtral**：在社区展示中，**Mistral Medium** 被指出表现优于 Mixtral，能提供更简洁且符合指令要求的输出，同时生成更相关的引用，这可能表明它是一个更先进的（可能是闭源的）Mixtral 版本。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **性能优化引发模式切换咨询：** LangChain 用户讨论了在 **oobabooga's text-generation-webui** 中为使用 **LlamaCpp model** 的聊天机器人将模式从 `chat-instruct` 切换到 `chat` 的有效性，分享了代码片段并询问了在 LangChain 应用中的实现问题。
- **文档困境与更新呼吁：** 用户对过时且不一致的 LangChain 文档表示担忧，强调了保持文档更新对于更好地跟踪包导入和使用的重要性。
- **发布并学习 ReAct 和 LangChain Chatbot：** 公告包括发布了 **ReAct Agent**（其灵感源自语言模型中推理与行动的协同），以及开源发布了具备 RAG 问答查询功能的 **LangChain Chatbot**，并邀请通过提供的 [GitHub repository](https://github.com/Haste171/langchain-chatbot) 进行反馈和探索。
- **AI 技术与应用教程视频：** 分享了各种教程和展示，例如使用 Groq 的硬件构建实时 AI 冷启动电话 Agent、Command-R 的长上下文任务能力，以及使用 Langchaingo 为 Telegram 群组创建 Prompt 模板的指南。
- **心理健康支持 AI 的进展：** 一篇新文章介绍了 **MindGuide**，旨在利用 LangChain 和 LLM 彻底改变心理健康护理，强调了基于技术的干预在心理健康领域的重要性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 总结

- **OpenRouter 短暂故障后恢复**：在数据库更新期间，**OpenRouter** 遇到了一个短暂问题，导致 **activity row**（活动行）在大约三分钟内不可用，但受影响的 Completion 未被收费。
- **Claude 3 Haiku 引起轰动**：新推出的 **Claude 3 Haiku** 模型因其 **120 tokens per second** 的速度和 **每 1 美元 4M prompt tokens** 的高性价比而备受关注。目前提供审核版和自我审核版，鼓励用户尝试，访问详情可通过[此链接](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)获取。
- **Olympia.chat 集成 OpenRouter**：[Olympia.chat](https://olympia.chat) 已集成 **OpenRouter** 以满足其 LLM 需求，该平台专注于个体创业者和小微企业，并预告了即将推出的用于与 OpenRouter 交互的开源 Ruby 库。
- **关于 OpenRouter AI 模型使用的对话**：在激烈的讨论中，参与者谈到了 **Groq's Mixtral model**，并澄清了 OpenRouter 的使用独立于 Groq 的免费访问期；此外，还探讨了在出现 "Request too big" 错误后的 **Mistral 8x7B** 模型限制。
- **对 GPT-4.5 的期待升温**：关于 **GPT-4.5** 的传闻和未经证实的信息引发了社区内的热烈推测和高度关注，标志着人们对 AI 领域这一潜在下一步进展的强烈期待。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 总结

- **Bing 的失误推高 GPT-4.5 期待**：Bing 搜索引擎意外索引了一篇 **GPT-4.5 博客文章**，尽管链接指向 404 页面，但仍引发了对其即将发布的兴奋。这种关注度也体现在通过 Kagi 和 DuckDuckGo 等其他搜索引擎发现该帖子的讨论中，一条 [tweet](https://twitter.com/AndrewCurran_/status/1767916848987914487) 也被传阅以进一步澄清。
- **GPT-4 在 LeetCode 中保持霸主地位**：**GPT-4** 在编程挑战中继续表现出色，正如[最近引用的一篇论文](https://livecodebench.github.io/pdfs/paper.pdf)所示，该论文重点介绍了其在 LeetCode 问题上的实力。
- **Claude 3 擅长提取核心内容**：通过 [Clautero project](https://github.com/Xeophon/Clautero) 详细介绍的 **Claude 3** 模型在文献摘要方面表现出显著进步，表明了 LLM 不断进化的能力。
- **Meta 深耕 AI 硬件**：Meta 宣布了对 AI 的重大投资，计划建设两个 2.4 万个 GPU 的集群，并目标到 2024 年拥有 350,000 块 NVIDIA H100 GPU，通过 [Grand Teton](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/) 和[正式公告](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)公开分享其基础设施雄心。
- **AI 中的广告——未来的辩论**：关于 Google 对 AI 模型货币化策略的讨论兴起，成员们比较了广告支持模式与订阅模式，并审视了与潜在 AI 生成广告相关的隐私影响和信任因素。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **Meta 的硬件升级带来了强大的计算能力**：Meta 宣布推出两个巨大的 **24k GPU 集群**，计划在 2024 年底前整合 350,000 个 NVIDIA H100 GPU，这将为 AI 提供相当于 600,000 个 GPU 的计算能力，详见其 [基础设施公告](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)。
  
- **CUDA 与 NVMe 的融合**：CUDA 的发展现在允许 NVMe 驱动器在存储应用中使用直接内存访问（DMA）和 GPU 集成，有望带来显著的效率提升，如 [ssd-gpu-dma GitHub 仓库](https://github.com/enfiskutensykkel/ssd-gpu-dma) 和 NVIDIA 的 GPUDirect Storage API [参考指南](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) 中所述。
  
- **利用 Nsight 提升 CUDA 开发效率**：NVIDIA 的 [Nsight™ Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition) 因其在 Linux 和 QNX 系统上的开发能力而受到赞誉，提供 CUDA 代码自动补全和智能分析功能，甚至引发了关于 Nsight Systems 与 Nsight Compute 实用性的讨论。
  
- **Torchao 征求反馈和经验分享**：PyTorch Labs 邀请开发者对 [torchao 的 GitHub issue](https://github.com/pytorch-labs/ao/issues/47) 中合并新量化算法提供反馈，并为渴望参与涉及 gpt-fast 和 sam-fast kernel 的 *真实世界 CUDA 模式项目* 的 kernel 编写者提供指导。
  
- **AI 工程师即将登场**：Cognition Labs 预告了 **Devin** 的即将亮相，这是一个能够执行复杂任务的自主 AI 软件工程师，其在 **SWE-bench** 编码基准测试中的表现引起了广泛关注，详见 [Cognition Labs 的博客文章](https://www.cognition-labs.com/blog)，并已被社区标记以待进一步考察。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **马斯克的风格在翻译中流失**：助手的回答详细说明了火星殖民的原因，但被批评缺乏埃隆·马斯克（Elon Musk）独特的风格，尽管很好地涵盖了他的观点，导致创意评分为 **[[7]]**。

- **RAG-narok 策略**：工程师们讨论了 RAG 提示词结构，偏好在用户消息中嵌入完整提示词或根据 SFT 进行调整以保持行为一致。同时，Jan Leike 通过推文发布了一个内部工具 **Transformer Debugger**，用于 Transformer 模型分析，具有快速探索和可解释性功能，且无需编写代码。

- **Mixtral 模型的混淆**：关于 `mixtral-7b-8expert` 模型的沟通误解引发了对其实验状态清晰度的质疑，这与官方的 `mistralai/Mixtral-8x7B-v0.1` 模型形成对比。后者被指出在非英语输出方面存在困难，引导用户使用正确的模型，并为对德国数据细节感兴趣的人提供 [数据集信息](https://huggingface.co/DiscoResearch/DiscoLM-70b)。

- **创意基准测试的突破**：**创意写作基准测试原型**已在 [GitHub 分支](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing) 上线，提供了一个对模型创意进行排名的工具，尽管它仍处于开发中，在区分度方面仍有改进空间。

- **对德语精准度的追求**：关于法律文本理想德语 Embedding 的咨询挑战了工程师对法律术语特殊性的思考，而寻找针对德语的 Embedding 基准测试则增加了复杂性，凸显了当前基准测试领域的空白。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **GPT-4.5 Turbo：传闻多于现实**：关于 **GPT-4.5 Turbo** 存在性的推测引发了讨论，最终揭示所谓的泄露实际上是一个过时且错误发布的草案，其中提到了截至 2024 年 7 月的训练数据。担忧主要集中在 Bing 搜索显示“openai announces gpt-4.5 turbo”无结果所造成的困惑，暗示该模型并不存在。

- **Token 限制引发的苦恼**：成员们表达了在使用 **gpt4turbo** 时遇到 4096 Token 限制的挫败感，但对话中未提供解决方案或规避方法。

- **Claude 与 Starship 的趣闻**：一条轻松的评论提到 OpenAI 的发布可能会掩盖 Elon Musk 的 Starship 的光芒，反映了人们对重大科技发布如何同步或冲突的持续关注。

- **预测 Llama-3 还是跳跃到 Llama-4？**：一位成员讨论了 **Llama** 的版本周期理论，预测由于质量问题将跳过 **Llama-3**，并可能在 7 月发布 **Llama-4**。预期的功能包括 **Mixture of Experts**、**SSM variants**、**Attention mods**、**图像和视频的多模态 (multi-modality)**、**扩展的上下文长度 (context lengths)** 以及**高级推理能力**。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **零点击蠕虫威胁 GenAI 应用**：一篇[最近发表的论文](https://sites.google.com/view/compromptmized)讨论了 “ComPromptMized”，这是 Stav Cohen 的一项研究，揭示了计算机蠕虫可以在无需用户交互的情况下利用各种 GenAI 模型，凸显了 AI 驱动应用（如邮件助手）的风险。

- **寻找最佳代码助手 AI**：工程师们正在寻找一个框架来比较 **Mistral** 和 **LLaMA2** 等模型作为代码助手的效率，同时承认合适的 **Benchmark** 需要具备准确性才能在这些比较中发挥作用。

- **Leaderboard 成为 AI 模型性能的首选参考**：为了比较模型性能，[chat.lmsys.org](https://chat.lmsys.org) 上的 **Leaderboard** 成为一个宝贵的资源，成员们对其提供的各种模型能力的见解表示赞赏。

- **Git Commit 消息获得语言模型升级**：一位成员分享了一个[使用 LLM 彻底改变 Git Commit 消息的技巧](https://harper.blog/2024/03/11/use-an-llm-to-automagically-generate-meaningful-git-commit-messages/)，详细介绍了一种将 LLM CLI 与 pre-commit-msg GIT hook 集成的方法，从而生成更具信息量的提交描述。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **聊天机器人开发者悬而未决**：工程师们寻求适用于创建管理**长上下文 (long contexts)** 聊天机器人的开源模型和框架建议。然而，讨论线程中未提供具体方案。
- **在经典游戏中植入 AI**：AI 爱好者可以观看关于使用 **Claude 3** 开发**《植物大战僵尸》**的 YouTube 教程，涵盖了 Python 编程和游戏开发方面的内容，详见 [Claude 3 made Plants Vs Zombies Game](https://www.youtube.com/watch?v=d7NGgglZXK8)。
- **使用 Command-R 深入研究 RAG**：一段新视频重点介绍了 **Command-R**，展示了其**检索增强生成 (RAG)** 的能力以及与外部 API 的集成，详见 [Lets RAG with Command-R](https://www.youtube.com/watch?v=rnP87DzGeDw)。
- **介绍 Devin，软件工程领域的 AI**：号称世界上第一个 AI 软件工程师 **Devin** 在 [Devin The Worlds first AI Software Engineer](https://www.youtube.com/watch?v=NSPtrrUQ_fw) 中亮相，展示了自主软件工程的能力，[Cognition Labs 的博客](https://www.cognition-labs.com/blog)也对此进行了讨论。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **加入多模态可解释性的愿景者行列**：Soniajoseph_ 征集旨在研究**多模态模型可解释性 (interpretability of multimodal models)** 的开源项目合作者。该倡议在 [LessWrong 帖子](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic)中有详细说明，更多讨论可以在他们的专用 Discord [服务器](https://discord.gg/2U2N8QmPmJ)上进行。

- **寻找极速推理**：一位公会成员正在寻找 **Phi 2** 最快的推理 (inference) 方法，并提到使用了 A100 GPU。他们正在研究用于大规模 Token 生成的 Batching、**vLLM** 或 **Axolotl** 等框架，并对量化 (quantization) 对速度的影响感到好奇。

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 总结

- **插件授权变得简单**：**AI Engineer Foundation** 建议通过实现**插件配置选项**来简化授权流程（通过传递 token），这一想法参考了 [Config Options RFC](https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit#heading=h.461b58g0npbn) 中的结构化 Schema。
- **创新项目头脑风暴**：成员们受邀提议**新项目**，需遵守 [Google Doc 指南](https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit&followup=https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit&ltmpl=docs&ec=GAZAGQ) 中列出的一系列标准，同时还预告了可能与 Microsoft 合作开展一个 **prompt 文件项目**。
- **认识代码奇才 Devin**：**Cognition Labs** 推出了 **Devin**，这是一款因在 **SWE-bench 编码基准测试**中表现卓越而备受赞誉的 AI 软件工程师，它已准备好在受控环境中与开发者工具协同工作，详见其 [博客文章](https://www.cognition-labs.com/blog)。

---

# 第二部分：频道详细总结与链接

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1217115602372792402)** (35 条消息🔥): 

- **AI 构建的《植物大战僵尸》**：一段名为“Claude 3 制作了《植物大战僵尸》游戏”的 YouTube 视频展示了使用 AI 模型 Claude 3 创建《植物大战僵尸》游戏的过程，其中结合了 Python 编程进行游戏开发。视频可以在[这里](https://www.youtube.com/watch?v=d7NGgglZXK8)观看。

- **新术语 "Mergeslop"**：一位成员幽默地注意到某个平台上首次使用了 "mergeslop" 一词，并对其新颖性表示惊讶。

- **Command-R 与 RAG 技术焦点**：另一段 YouTube 视频重点介绍了 Command-R，这是一款针对长上下文任务（如检索增强生成 RAG）优化的 AI 模型，视频观看地址在[这里](https://www.youtube.com/watch?v=rnP87DzGeDw)。
  
- **AI 新闻摘要服务亮相**：AI News 宣布了一项服务，旨在总结来自 AI Discord 和 Twitter 的讨论，承诺为用户节省大量时间。感兴趣的人可以订阅时事通讯并查看[本周 AI 新闻](https://buttondown.email/ainews/archive/ainews-fixing-gemma/#nous-research-ai-discord-summary)。

- **给 AI 爱好者的书籍推荐**：成员们分享了具有 AI 和推测小说主题的个人书单，包括《三体》系列、莱姆的《网络寓言》(The Cyberiad) 以及吉布森的《蔓延三部曲》(Sprawl Trilogy)。

- **长上下文聊天机器人开发咨询**：一位频道成员就构建具有长上下文或记忆能力的聊天机器人的开源模型和框架寻求建议，引发了关于当前和未来能力的讨论。[SDAM（一种稀疏分布式关联存储）](https://github.com/derbydefi/sdam) 被推荐作为一个参考资源。

**提到的链接**：

- [使用 Command-R 进行 RAG](https://www.youtube.com/watch?v=rnP87DzGeDw)：Command-R 是一款生成式模型，针对检索增强生成 (RAG) 以及使用外部 API 和工具等长上下文任务进行了优化。它的设计...
- [Claude 3 制作了《植物大战僵尸》游戏](https://www.youtube.com/watch?v=d7NGgglZXK8)：将探讨如何使用 Claude 3 开发《植物大战僵尸》#python #pythonprogramming #game #gamedev #gamedevelopment #llm #claude
- [GitHub - derbydefi/sdam: 稀疏分布式关联存储](https://github.com/derbydefi/sdam)：sparse distributed associative memory。通过在 GitHub 上创建账号为 sdam 的开发做出贡献。
- [[AINews] 修复 Gemma](https://buttondown.email/ainews/archive/ainews-fixing-gemma/#nous-research-ai-discord-summary)：2024年3月7日至3月11日的 AI 新闻。我们为您检查了 356 个 Twitter 和 21 个 Discord（335 个频道，6154 条消息）。预计节省阅读时间（以 200wpm 计算）：...
- [Devin：全球首位 AI 软件工程师](https://www.youtube.com/watch?v=NSPtrrUQ_fw)：Devin 是全自动软件工程师 https://www.cognition-labs.com/blog

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1217115012989194301)** (8 条消息🔥): 

- **认识 Devin，非凡的 AI 软件工程师**：Cognition Labs 推出了 **Devin**，被誉为首位 AI 软件工程师。在 SWE-Bench 编码基准测试中，它在未经协助的情况下解决了 13.86% 的 GitHub issue，超越了之前的基准。Devin 可以自主使用 shell、代码编辑器和 Web 浏览器，[查看完整推文](https://fxtwitter.com/cognition_labs/status/1767548763134964000)。

- **C4AI Command-R 释放生成式模型威力**：CohereForAI 发布了 C4AI Command-R，这是一个拥有 350 亿参数的高性能生成式模型，在推理、摘要和多语言生成方面表现出色。该模型提供开放权重，并遵循 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01) 上的特定许可和可接受使用政策。

- **Cerebras 揭晓全球最快 AI 芯片**：Cerebras Systems 推出了 CS-3 芯片，声称是全球最快的 AI 加速器，能够在单个设备上训练高达 24 万亿参数的模型。CS-3 拥有惊人的规格和 AI 计算技术的进步，详见 [新闻稿](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) 和 [产品信息](https://www.cerebras.net/product-system/)。

- **AI 芯片创新的形状**：一位成员对新款 Cerebras CS-3 的芯片设计进行了推测，讨论了为什么它是方形而不是圆形或半圆形，并提出了可能容纳更多晶体管的形状。

- **无链接，仅预告**：分享了来自用户 Katie Kang 的 Twitter 链接，但未提供额外的上下文。

**提到的链接**：

- [来自 Cerebras (@CerebrasSystems) 的推文](https://x.com/CerebrasSystems/status/1767929699177767325?s=20)：📣宣布地球上最快的 AI 芯片📣 Cerebras 隆重宣布 CS-3：全球最快的 AI 加速器。CS-3 可以在单个设备上训练高达 24 万亿参数的模型。全球...
- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01)：未找到描述
- [来自 Cognition (@cognition_labs) 的推文](https://fxtwitter.com/cognition_labs/status/1767548763134964000)：今天我们很高兴向大家介绍 Devin，首位 AI 软件工程师。Devin 在 SWE-Bench 编码基准测试中达到了新的 SOTA 水平，并成功通过了实际的工程面试...

  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1217579163926401064)** (1 条消息): 

- **Hermes 2 Pro 7B 发布**：Nous Research 最新发布的 **Hermes 2 Pro 7B** 模型通过改进的数据集增强了 Agent 的可靠性，并在 Function Calling 和 JSON Mode 方面具有多功能性。该模型可以在 [Hugging Face - Hermes-2-Pro-Mistral-7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) 下载，GGUF 版本也可在 [Hugging Face - Hermes-2-Pro-Mistral-7B-GGUF](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF) 获取。

- **协作努力与致谢**：经过多位贡献者的数月协作努力以及 Latitude.sh 提供的计算赞助，**Hermes 2 Pro 7B** 模型得以问世。

- **创新的 Function Calling 和 JSON Mode**：利用特殊的 System Prompts 和 XML 标签实现了高级 Function Calling 功能，示例代码可在 [GitHub - Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling) 找到。

- **用于增强性能测量的自定义评估框架**：该模型包含一个独特的 Function Calling 和 JSON Mode 评估流水线，基于 Fireworks AI 的原始数据集和代码构建，可在 [GitHub - Function Calling Eval](https://github.com/interstellarninja/function-calling-eval) 找到。

- **数据集已开放下载**：用于评估 Function Calling 和 JSON Mode 性能的数据集已公开，分别可在 [Hugging Face - Func-Calling-Eval](https://huggingface.co/datasets/NousResearch/func-calling-eval) 和 [Hugging Face - JSON-Mode-Eval](https://huggingface.co/datasets/NousResearch/json-mode-eval) 获取。
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1217029726061858927)** (349 条消息🔥🔥):

- **Function Calling 精度**：*Hermes 2 Pro* 在 zero-shot 设置下具有 91% 的 Function Calling 准确率，表明即使没有 few-shot 训练，其性能也极为出色。Function Calling 和 JSON Mode 的评估数据集已经发布，训练集随后也将发布。
- **AI Act 震动欧洲**：欧盟 AI Act 刚刚正式通过成为法律，禁止了某些 AI 实践，要求提交能源消耗报告，并可能影响计划在欧洲开展业务的 AI 公司。
- **DeepMind 与 Fortnite**：[Google DeepMind 推文文章](https://twitter.com/GoogleDeepMind/status/1767918515585994818) 表明 AI 现在可以在 Fortnite 中超越人类，这引起了游戏社区的关注和兴趣。
- **推进推理库发展**：讨论了如何改进支持 function calling/tool 的推理库，其中 [Hermes Function Calling GitHub 仓库](https://github.com/NousResearch/Hermes-Function-Calling) 被提及为一个值得关注的资源。
- **新模型发布引发热潮**：Nous Research 发布了 *Hermes 2 Pro*，因其在 llama.cpp 和 vercel AI SDK 等应用中实现 function calling 和结构化输出的潜力，在社区中引起了热烈反响；DSPy 优化也是近期关注的热点话题。

**提及的链接**：

- [来自 Cake (@ILiedAboutCake) 的推文](https://x.com/iliedaboutcake/status/1766509947016139163?s=46)：笑死，亚马逊的“搜索评论”现在只是盲目地运行一个 AI 模型，AI 毁了互联网的使用体验。
- [princeton-nlp/SWE-Llama-7b · Hugging Face](https://huggingface.co/princeton-nlp/SWE-Llama-7b)：未找到描述
- [Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/abs/2403.07816)：我们研究了训练大语言模型 (LLMs) 以使其具备多个专业领域（如 coding、数学推理和世界知识）能力的有效方法。我们的方法名为...
- [DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1#function-calling)：未找到描述
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)：未找到描述
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/)：未找到描述
- [来自 Jack Burlinson (@jfbrly) 的推文](https://x.com/jfbrly/status/1767653596957642879?s=20)：如果你想知道 @cognition_labs 团队有多厉害... 这是 CEO (@ScottWu46) 14 年前的样子。↘️ 引用 Cognition (@cognition_labs)：今天我们很高兴地介绍...
- [Bh187 Austin Powers GIF - Bh187 Austin Powers I Love You - 发现并分享 GIF](https://tenor.com/view/bh187-austin-powers-i-love-you-you-complete-me-gif-19285472)：点击查看 GIF
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling/tree/main)：通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)：通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。
- [用于微调和推理的 Gemma 优化 · Issue #29616 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29616)：系统信息：最新的 transformers 版本，大多数平台。谁能帮忙？@ArthurZucker 和 @younesbelkada。信息：官方示例脚本，我修改后的脚本。任务：一个官方支持的...
- [Gemma 错误修复 - 近似 GELU, Layernorms, Sqrt(hd) 由 danielhanchen 提交 · Pull Request #29402 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29402)：只是更多的 Gemma 修复 :) 目前也在检查更多内容！相关 PR：#29285，该 PR 显示 RoPE 必须在 float32 而不是 float16 中完成，否则会导致位置编码失去精度。@Ar...
- [OpenAI Tools / function calling v2 由 FlorianJoncour 提交 · Pull Request #3237 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/3237/files#diff-aa650ea701251f5647254f86d652333a30e4871cfcc2d3ac4fecf83dd1f1a776)：此 PR 继 #2488 之后。实现已更新为使用新的 guided generation。如果在查询期间，用户将 tool_choice 设置为 auto，服务器将使用 #24 中使用的模板系统...
- [Guidance](https://moon-ci-docs.huggingface.co/docs/text-generation-inference/pr_1587/en/guidance)：未找到描述
- [OpenAI 兼容 Web 服务器 - llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling)：未找到描述

  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1217030705159344199)** (107 条消息🔥🔥):

- **讨论模型许可的细微差别**：成员们询问了在 Apache 2 和 MIT 许可下，源自 **GPT-4** 的 **nous Hermes 2** 等模型的商业可用性。得到的澄清是：*"随你心意，我们不会干涉（我们不会起诉你，哈哈）"*，同时讨论中也强调了 TOS 执行和内容共享的复杂性。

- **微调效率与风格迁移的挑战**：对话涉及了在小规模数据集上进行微调以模仿风格的低效性。建议将重点从 prompt engineering 转向风格迁移，并强调关注语言模型中的 role-playing 方面。

- **深入探讨 Function Calling 与结构化输出**：深入讨论了 LLM 中的 Function calling 能力和结构化 JSON 输出。例如，**Trelis/Llama-2-7b-chat-hf-function-calling-v2** 模型引发了关于其 functional calling 和 JSON mode 运行方式的辩论，并对其返回结构化 JSON 参数的方法提供了见解。

- **发布更新与理解模型能力**：对 **Nous Hermes 2 Pro** 的期待日益高涨，一位成员暗示：*"<:cPES_Wink:623401321382281226> 也许今天就会发布 <:cPES_Wink:623401321382281226>"*。另一位成员强调了模型版本管理清晰度的重要性，以避免用户混淆，反对在没有明确版本号的情况下使用诸如 "Pro" 之类模糊的命名。

- **探索 Hermes 2 Pro 与 Ollama 的集成**：关于如何将即将发布的 **Nous Hermes 2 Pro** 与 **Ollama** 集成的问题被提出。回复指出 **Ollama** 可以支持特定的 GGUF，无需进行模型量化（quantization），并分享了 [ollama/docs/import.md on GitHub](https://github.com/ollama/ollama/blob/main/docs/import.md) 的指南。

**提到的链接**：

- [ollama/docs/import.md at main · ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/import.md)：快速上手 Llama 2, Mistral, Gemma 以及其他大型语言模型。 - ollama/ollama
- [Trelis/Llama-2-7b-chat-hf-function-calling-v2 · Hugging Face](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2)：未找到描述
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling)：通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1217120405924614265)** (127 messages🔥🔥): 

- **AI 软件工程领域的重大突破**：**Cognition Labs** 推出了 **Devin**，这是一个能够通过工程面试并完成真实工作的自主 AI 软件工程师。Devin 远超此前最好的 SWE-Bench 基准测试，解决了高达 13.86% 的 GitHub issues。[查看关于 Devin 的讨论帖](https://x.com/cognition_labs/status/176754876313)。
  
- **权重窃取还是仅仅是炒作？**：一篇新论文揭示，利用 API 可以从 **ChatGPT** 和 **PaLM-2** 等生产级语言模型中推断出权重。这引发了关于 AI 模型伦理和未来安全措施的讨论。[在此查看全文](https://not-just-memorization.github.io/partial-model-stealing.html)。

- **Google 的 Gemini 难以令人印象深刻**：用户对 Google 的 **Gemini API** 表示失望，强调其复杂性且文档笨重。有一种观点认为，与 OpenAI 和 Anthropic 等竞争对手相比，Google 在 AI API 领域正处于落后地位。

- **Together.ai 为 AI 初创公司加强算力支持**：Together.ai 宣布融资 1.06 亿美元，用于构建大规模运行生成式 AI 应用的平台，并推出了 **Sequoia**，这是一种高效服务大型 LLM 的方法。[了解更多关于他们愿景的信息](http://together.ai/blog/series-a2)。

- **Cerebras 发布突破性 AI 芯片**：**Cerebras** 发布了 **CS-3**，这是目前最快的 AI 芯片，能够在单个设备上训练高达 24 万亿参数的模型。这一进展是 AI 硬件创新的重大飞跃。[探索 CS-3](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine)。

**提到的链接**：

- [Stealing Part of a Production Language Model](https://not-just-memorization.github.io/partial-model-stealing.html)：未找到描述
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-03-12/physical-intelligence-is-building-ai-for-robots-backed-by-openai)：未找到描述
- [Using LangSmith to Support Fine-tuning](https://blog.langchain.dev/using-langsmith-to-support-fine-tuning-of-open-source-llms/)：摘要：我们创建了一份使用 LangSmith 进行数据集管理和评估，从而支持微调和评估 LLM 的指南。我们分别在 CoLab 上使用开源 LLM 和在 HuggingFace 上进行了模型训练...

- [来自 Cognition (@cognition_labs) 的推文](https://x.com/cognition_labs/status/1767548763134964000?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：今天我们很高兴地介绍 Devin，全球首位 AI 软件工程师。Devin 在 SWE-Bench 编码基准测试中达到了最新的 state-of-the-art 水平，并成功通过了实际的工程面试...
- [来自 Ate-a-Pi (@8teAPi) 的推文](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Sora WSJ 采访，Mira Murati 提供了迄今为止关于 Sora 最详尽的细节 > Joanna Stern 提供了几个提示词供其生成 > 这是我第一次看到 Sora 视频出现严重的变形问题...
- [🌎 The Compute Fund](https://computefund.ai/)：以具有竞争力的价格可靠地获取您所需的顶级 GPU，以换取股权。
- [来自 Ate-a-Pi (@8teAPi) 的推文](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2Ot)：Sora WSJ 采访，Mira Murati 提供了迄今为止关于 Sora 最详尽的细节 > Joanna Stern 提供了几个提示词供其生成 > 这是我第一次看到 Sora 视频出现严重的变形问题...
- [来自 Patrick Collison (@patrickc) 的推文](https://x.com/patrickc/status/1767603551927242809?s=46&t=90xQ8sGy63D2OtiaoGJuww)：这些不仅仅是精心挑选的演示。根据我的经验，Devin 在实践中的表现非常令人印象深刻。↘️ 引用 Cognition (@cognition_labs) 今天我们很高兴地介绍 Devin，首位 AI 软件工程师...
- [来自 Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767936720618799336?s=46&t=90xQ8sGy63D2OtiaoGJuww)：很高兴宣布我们的新推测性解码（speculative decoding）方法，Sequoia！Sequoia 将推测性解码扩展到极大的推测预算，对不同的解码配置具有鲁棒性，并且可以自适应...
- [来自 Cerebras (@CerebrasSystems) 的推文](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：📣宣布全球最快的 AI 芯片📣 Cerebras 自豪地宣布 CS-3：世界上最快的 AI 加速器。CS-3 可以在单台设备上训练高达 24 万亿参数的模型。全球...
- [来自 Figure (@Figure_robot) 的推文](https://x.com/figure_robot/status/1767913661253984474?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：借助 OpenAI，Figure 01 现在可以与人进行完整的对话 - OpenAI 模型提供高级视觉和语言智能 - Figure 神经网络提供快速、底层的灵巧机器人...
- [来自 Cognition (@cognition_labs) 的推文](https://x.com/cognition_labs/status/1767581492585140435?s=20)：Devin 构建了一个自定义 Chrome 扩展程序 ↘️ 引用 Arun Shroff (@arunshroff) @cognition_labs 这看起来太棒了！希望能获得访问权限！我最近使用 ChatGPT 创建了一个 Chrome 扩展程序来...
- [来自 Chief AI Officer (@chiefaioffice) 的推文](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：风投支持的 AI 员工初创公司正成为一种趋势。以下是 2024 年融资的一些公司及其总融资金额：软件工程师 - Cognition ($21M+) 软件工程师 - Magic ($145M+) 产品经理 - Version Le...
- [来自 Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1767598414945292695?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：# 软件工程自动化 在我看来，软件工程自动化将类似于自动驾驶。例如，在自动驾驶中，自主性不断提高和抽象层次不断提升的过程...
- [来自 Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8)：今天我们非常激动地分享，我们已完成由 @SalesforceVC 领投、@coatuemgmt 及现有投资者参投的 1.06 亿美元新一轮融资。我们的愿景是迅速将创新成果带给...
- [来自 Akshat Bubna (@akshat_b) 的推文](https://x.com/akshat_b/status/1767579399317029211?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我第一次尝试 Devin 时，它：- 导航到了我提供的 @modal_labs 文档页面 - 学习了如何安装 - 将控制权交给我进行身份验证 - 启动了一个 ComfyUI 部署 - 与其交互...
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1767582756291387484?s=46&t=90xQ8sGy63D2OtiaoGJuww)：这是第一个（不仅限于编码）似乎跨越了人类水平门槛并能可靠运行的 Agent 演示。它还向我们展示了结合 LLM 和 tree search 的可能性...
- [来自 Ashlee Vance (@ashleevance) 的推文](https://x.com/ashleevance/status/1767538050262073688?s=46&t=90xQ8sGy63D2OtiaoGJuww)：独家报道：一家名为 Cognition AI 的初创公司发布了目前看来功能最强大的编码助手。它不仅仅是自动完成任务，还可以独立编写整个程序。得到了...

- [来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1767664455889097009?s=20): 希望这能奏效 🕯 🕯 🕯 🕯 @elonmusk 🕯 开源 🕯 @xAI Grok 🕯 开启 🕯...
- [来自 Together AI (@togethercompute) 的推文](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天我们非常激动地分享，我们在由 @SalesforceVC 领投、@coatuemgmt 及现有投资者参投的新一轮融资中筹集了 1.06 亿美元。我们的愿景是快速带来创新...
- [4,000,000,000,000 个晶体管，一颗巨型芯片 (Cerebras WSE-3)](https://www.youtube.com/watch?v=f4Dly8I8lMY&ab_channel=TechTechPotato): 作为唯一一家拥有像人头一样大芯片的公司，Cerebras 在 AI 芯片（AI silicon）方面拥有独特的价值主张。今天他们发布了第三代...
- [来自 muhtasham (@Muhtasham9) 的推文](https://x.com/muhtasham9/status/1767507958017995196?s=46&t=90xQ8sGy63D2OtiaoGJuww): DeepMind 的人员现在可以窃取 API 背后的权重。“我们还还原了 gpt-3.5-turbo 模型的准确隐藏层维度大小，并估计只需不到 2,000 美元的查询成本即可还原整个...”
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018312083243514?s=46&t=90xQ8sGy63D2OtiaoGJuww): 凭借最先进的视觉能力以及在推理、数学和编程等行业基准测试中的强劲表现，Haiku 是适用于各种企业级应用的通用解决方案。
- [来自 asura (@stimfilled) 的推文](https://x.com/stimfilled/status/1767617991980589209?s=20): @qtnx_ 3) dateLastCrawled: 2023-09
- [来自 Mckay Wrigley (@mckaywrigley) 的推文](https://x.com/mckaywrigley/status/1767985840448516343?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 我被 Devin 震撼了。看我使用了它 27 分钟。简直疯狂。AI Agent 的时代已经开启。
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天我们发布了 Claude 3 Haiku，这是其智能级别中最快且最实惠的模型。Haiku 现在已在 API 和 http://claude.ai 上面向 Claude Pro 订阅用户开放。
- [来自 Siqi Chen (@blader) 的推文](https://x.com/blader/status/1767707799390462341?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 这是 14 年前 Cognition 的 CEO。认为 10x/100x 工程师不存在的想法纯粹是一种心理安慰（cope）。
- [来自 James O'Leary (@jpohhhh) 的推文](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w): Google Gemini 的集成在 15 分钟前开始，这是一个“自我安慰”的推特串 - 有一个名为“Gemini API”的 API，在明年年初开始收费之前是免费的（现在是三月中旬）- ...
- [来自 James O'Leary (@jpohhhh) 的推文](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP): Google Gemini 的集成在 15 分钟前开始，这是一个“自我安慰”的推特串 - 有一个名为“Gemini API”的 API，在明年年初开始收费之前是免费的（现在是三月中旬）- ...
- [来自 Neal Wu (@WuNeal) 的推文](https://x.com/wuneal/status/1767561150609186965?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天我终于可以分享 Devin 了，这是由我们 @cognition_labs 团队构建的首位 AI 软件工程师。Devin 能够端到端地构建应用，在生产代码库中查找 Bug，甚至进行微调（fine...）
- [来自 Lucas Atkins (@LucasAtkins7) 的推文](https://x.com/lucasatkins7/status/1767805804705411098?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 今晚，我将发布八个 Gemma 微调版本以及它们组合而成的混合专家模型（Mixture of Experts）测试版，命名为 GemMoE。GemMoE 内置了所有 Gemma 的 Bug 修复。你不需要做任何额外操作就能获得...
- [来自 Fred Ehrsam (@FEhrsam) 的推文](https://x.com/fehrsam/status/1767586744889913810?s=46&t=90xQ8sGy63D2OtiaoGJuww): 这是我第一次看到 AI 接受一项复杂任务，将其分解为步骤，完成它，并向人类展示过程中的每一步——达到可以完全接手人类工作的程度。...
- [首个 AI 病毒来了！](https://youtu.be/4NZc0rH9gco): ❤️ 查看 Weights & Biases 并在次注册免费演示：https://wandb.me/papers 📝 论文 "ComPromptMized: Unleashing Zero-click Worms that Target ..."
- [来自 Varun Shenoy (@varunshenoy_) 的推文](https://x.com/varunshenoy_/status/1767591341289250961?s=46&t=90xQ8sGy63D2OtiaoGJuww): Devin 在数据提取方面表现得不可思议（incredible）。在过去的几周里，我一直在从不同的博客中抓取数据，Devin 1. 编写爬虫来导航网站 2. 执行代码...
- [来自 Andrew Kean Gao (@itsandrewgao) 的推文](https://x.com/itsandrewgao/status/1767576901088919897?s=46&t=90xQ8sGy63D2OtiaoGJuww): 我从不相信录制的演示，所以我联系了 @cognition_labs 团队申请早期访问权限亲自尝试，并且拿到了！我将在这里分享我对 #devin 的真实看法。🧵🧵 1/n ↘️ 引用...

- [[AINews] 全球首个全自动 AI Engineer](https://buttondown.email/ainews/archive/ainews-the-worlds-first-fully-autonomous-ai/): 2024年3月11日至3月12日的 AI 新闻。我们为您检查了 364 条 Twitter 和 21 个 Discord（336 个频道，3499 条消息）。预计节省阅读时间（以 200wpm 计算）：...
- [来自 simp 4 satoshi (@iamgingertrash) 的推文](https://x.com/iamgingertrash/status/1767593902251421763?s=20): 终于，很高兴推出 Truffle-1 —— 一款售价 1299 美元的推理引擎，旨在仅用 60 瓦功率运行 OSS 模型 https://preorder.itsalltruffles.com
- [亚马逊发布 Rufus，一种全新的生成式 AI 驱动的对话式购物体验](https://www.aboutamazon.com/news/retail/amazon-rufus): 通过 Rufus，客户现在可以与一位精通亚马逊选品的生成式 AI 专家一起购物，该专家可以整合来自网络各处的信息来...
- [Perspective – 属于你的空间](https://joinperspective.com/): 一个用于建立完整生活记录的私人日志。
- [添加对 Gemini API 的支持 · Issue #441 · jxnl/instructor](https://github.com/jxnl/instructor/issues/441): 新的 Gemini API 引入了对 function calling 的支持。你定义一组函数及其预期参数，并将它们传递给 tools 参数。我们能否在 instructor 中添加 Gemini 支持...

---

**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1217528504497733654)** (7 messages): 

- **LLM Paper Club 活动邀请**: 为 Latent Space Discord 社区发布了一个提醒，关于即将举行的 LLM Paper Club 活动，主题是“用于微调的合成数据 (Synthetic Data for Finetuning)”，时间为太平洋时间中午 12 点。鼓励参与者阅读 [Eugene Yan 关于合成数据的综述文章](https://eugeneyan.com/writing/synthetic/)以了解背景信息。
- **重要：接受你的 Luma 邀请**: 敦促成员在 Luma ([https://lu.ma/wefvz0sb](https://lu.ma/wefvz0sb)) 上接受邀请，以免因不活跃而被从未来日历提醒的自动邀请名单中剔除。
- **合成数据文章链接更正**: 更正了之前合成数据文章的链接，因为它包含一个多余的句点导致 404 错误。更新后的活动封面可以在[这里](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186)找到。
- **Picocreator 强调了险些被剔除的情况**: picocreator 以轻松的口吻提到社区险些在活动提醒中被大规模剔除，诙谐地庆祝了大家的“死里逃生”。
- **Swyxio 强调活动时间的连贯性**: Swyxio 指出 LLM Paper Club 在过去 6 个月里一直固定在同一时间举行，暗示老成员应该已经知道时间了。

**提到的链接**:

- [LLM Paper Club (用于微调的合成数据) · Luma](https://lu.ma/wefvz0sb): 本周我们将与 @eugeneyan 一起讨论综述文章——如何生成和使用用于微调的合成数据 (https://eugeneyan.com/writing/synthetic/)。我们已转为使用...
- [如何生成和使用用于微调的合成数据](https://eugeneyan.com/writing/synthetic/): 克服 instruction-tuning、preference-tuning 和 pretraining 中人工标注的瓶颈。

---

**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1217547863983259798)** (207 messages🔥🔥): 

- **合成数据的前辈**: [Eugene Yan](https://eugeneyan.com/writing/synthetic/) 讨论了在 pretraining 和 instruction-tuning 等各个方面使用 **合成数据 (synthetic data)** 进行模型训练的可行性。他强调合成数据的生成速度更快、成本更低，在避免隐私问题的同时，还能提供在质量和多样性上可能超过人工标注的数据。
  
- **语音识别：值得探索的深坑**: 分享了一个 [Twitter 线程](https://twitter.com/swyx/status/1765995892107317407)链接，深入探讨语音识别以及 **text-to-speech** 和 **speech-to-text** 相关话题。讨论还涉及了各种应用和工具，如 **vapi.ai** 和 **whisper**。
  
- **Whisper 在语音处理方面的潜力**: 在对语音技术的关注中，[whisper.cpp](https://github.com/bminixhofer/whisper.cpp) 被提及为一个优秀的语音识别工具，并且有人请求涵盖 **用于 diarization 的开源 SOTA**。

- **重温经典论文**: Eugene Yan 赞扬了重温关于合成数据的旧论文的重要性，并表示“**合成数据几乎是你所需要的一切 (synthetic data is almost all you need)**”，同时强调了 **self-reward** 是一个重要的概念。

- **社区参与邀请**：鼓励社区参与论文解读，**公开邀请所有观众**深入研究论文并为讨论做出贡献。

**提到的链接**：

- [Why Not Both Take Both GIF - Why Not Both Why Not Take Both - Discover &amp; Share GIFs](https://tenor.com/view/why-not-both-why-not-take-both-gif-11478682)：点击查看 GIF
- [Join Slido: Enter #code to vote and ask questions](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions)：参与实时投票、测验或问答。无需登录。
- [AI News](https://buttondown.email/ainews/)：我们汇总了 AI Discord 频道和顶尖 Twitter 账号，每天为您发送综述！查看归档示例。“我每天花费的最具杠杆作用的 45 分钟” —— Soumith，“最好的 AI 通讯...”
- [Fine-tuning vs RAG](https://open.spotify.com/episode/37Jd55nAruyVysHDNe0R6R?si=33926484c4c248a2)：在 Spotify 上收听来自 Practical AI: Machine Learning, Data Science 的这一集。在本集中，我们再次欢迎来自 MLOps 社区的好朋友 Demetrios，讨论 Fine-tuning 与检索...
- [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/)：克服指令微调 (Instruction-tuning)、偏好微调 (Preference-tuning) 和预训练 (Pretraining) 中人工标注的瓶颈。
- [dspy/docs/api/optimizers/BootstrapFinetune.md at 0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d/docs/api/optimizers/BootstrapFinetune.md?plain=1#L5)：DSPy：用于编程（而非提示）基础模型的框架 —— stanfordnlp/dspy
- [Forget ChatGPT and Gemini &mdash; Claude 3 is the most human-like chatbot I've ever used](https://www.tomsguide.com/ai/forget-chatgpt-and-gemini-claude-3-is-the-most-human-like-chatbot-ive-ever-used#:~:text=Summary&text=Claude%203%20is%20one%20of,can%20speculate%20on%20its%20potential.)：它不是 AGI，但正在接近。
- [🦅 Eagle 7B : Soaring past Transformers with 1 Trillion Tokens Across 100+ Languages (RWKV-v5)](https://blog.rwkv.com/i/141130059/multi-lingual-performance-details>)：RWKV-v5 架构和 Linear Transformer 的全新时代已经到来 —— 拥有当今开源界最强的多语言模型。

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1217032711345143958)** (311 条消息🔥🔥): 

- **Perplexity AI 使用 Claude 3 Opus**：尽管讨论了多种 AI，但已明确 Perplexity AI 使用的是 *Claude 3 Opus*。用户询问了防止查重检测的工具，不过对于目前工具检测 AI 生成内容的有效性存在怀疑。

- **对 AI 查重工具的担忧**：用户讨论了现有查重工具的局限性，有人认为目前不存在可靠的方法来检测 AI 生成的文本。讨论还涉及了 AI 领域可能需要取得突破来解决这一问题。

- **关于搜索引擎和索引器的辩论**：关于 Perplexity AI 是否使用其他索引器展开了激烈辩论，成员们最终澄清 Perplexity 拥有**自己的索引器**，这有助于提高其速度。此外，讨论还涉及了 Google 过去的表现以及 AI 时代搜索引擎的未来。

- **AI 带来的生产力提升**：多位用户报告称，得益于 Perplexity AI 和 Notebook LLM 等工具，生产力显著提高。他们分享了 AI 如何辅助研究和信息收集，尽管存在一些局限性，例如某些工具的实验性质限制了多文档的上传。

- **对 Perplexity AI 产品的困惑**：用户对 Perplexity AI 内部可用的模型表示困惑，例如 Claude 模型的类型以及每种模型允许的使用次数。一位用户澄清说，**Claude 3 Opus** 有 5 次使用机会，之后会切换到 **Claude 3 Sonnet**。

**提到的链接**：

- [MSN](https://www.msn.com/en-us/news/technology/new-rabbit-r1-demo-promises-a-world-without-apps-and-a-lot-more-talking-to-your-tech/ar-BB1jLHhR): 未找到描述
- [MSN](https://www.msn.com/en-us/news/technology/new-rabbit-r1-demo-promises-a-world-without-apps-and-a-lot): 未找到描述
- [Perplexity 为其聊天机器人引入 Yelp 数据](https://www.theverge.com/2024/3/12/24098728/perplexity-chatbot-yelp-suggestions-data-ai): Yelp 与该 AI 搜索引擎达成了一项协议。
- [报告称美国必须迅速行动以规避 AI 风险](https://time.com/6898967/): 一份政府委托的报告指出，美国政府必须“果断”行动，以避免 AI 对人类造成“灭绝级威胁”。
- [Plotly Sankey 图表的进一步探索](https://medium.com/@twelsh37/further-adventures-in-plotly-sankey-diagrams-fdba9ff08af6): 冒险仍在继续
- [要求 Google CEO Sundar Pichai 下台的呼声日益高涨](https://www.businessinsider.com/calls-for-google-ceo-sundar-pichai-alphabet-step-down-ai-2024-3): 分析师认为 Google 的搜索业务目前使其处于安全地位，但随着生成式 AI 对手的激增，这种情况可能很快就会改变。
- [Perplexity AI CEO 分享 Google 如何留住他想聘用的员工](https://www.google.com/amp/s/www.ndtv.com/feature/perplexity-ai-ceo-shares-how-google-retained-an-employee-he-wanted-to-hire-5074830/amp/1): 搜索引擎 Perplexity AI 的 CEO Aravind Srinivas 最近分享了一个有趣的事件，揭示了大型科技公司如何准备投入巨资来留住人才……
- [新的 Rabbit R1 演示承诺一个没有 App 的世界——以及更多与科技设备的对话](https://www.techradar.com/computing/artificial-intelligence/new-rabbit-r1-demo-promises-a-world-without-apps-and-a-lot-more-talking-to-your-tech): 与你的机器人聊天
- [Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1767614488394830072?s=46): 如果 Mikhail 让 Microsoft Copilot 免费，我将让 Perplexity Pro 免费 ↘️ 引用 Ded (@dened21) @AravSrinivas @MParakhin 我们想要免费的 Perplexity Pro（通过高度个性化的广告变现）
- [OpenAI 刚刚泄露了 GPT 4.5 ?!! (GPT 4.5 更新详解)](https://www.youtube.com/watch?v=shJTJjjiqy8): ✉️ 加入我的每周时事通讯 - https://mailchi.mp/6cff54ad7e2e/theaigrid🐤 在 Twitter 上关注我 https://twitter.com/TheAiGrid🌐 查看我的网站 - https:/...
- [不仅仅是 OpenAI Wrapper：Perplexity 转向开源](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。
- [Killed by Google](https://killedbygoogle.com/): Killed by Google 是一个记录被 Google 关停的产品、服务和设备的开源列表。它是对那些被 Google 关停的深受喜爱的服务和产品的致敬与纪念。
- [I Believe In People Sundar Pichai GIF - I Believe In People Sundar Pichai Youtube - 发现并分享 GIF](https://tenor.com/view/i-believe-in-people-sundar-pichai-youtube-dear-earth-i-have-faith-in-people-gif-23560720): 点击查看 GIF
- [Reddit - 深入探索一切](https://www.reddit.com/r/Infographics/comments/17j907h/how_google_makes_money/): 未找到描述
- [CEO 称他曾试图聘请 Meta 的 AI 研究员，却被告知“等你有了 10,000 块 H100 GPU 再来找我”](https://www.businessinsider.in/tech/news/ceo-says-he-tried-to-hire-an-ai-researcher-from-meta-and-was-told-to-come-back-to-me-when-you-have-10000-h100-gpus/articleshow/108409971.cms): 一家 AI 初创公司的 CEO 表示，他无法聘请 Meta 的研究员，因为公司没有足够的 GPU。
- [与 Perplexity AI CEO Aravind Srinivas 及 FirstMark 合伙人 Matt Turck 的炉边谈话](https://youtu.be/RTCVzZb3RTE?si=f6g5qVBr1NldkVB_&t=1982): 今天我们邀请到了 Perplexity AI 的 CEO Aravind Srinivas，这是一款聊天机器人式的 AI 对话引擎，能直接引用来源回答用户的问题……
- [GitHub - danielmiessler/fabric: fabric 是一个利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过众包的 AI Prompt 集来解决特定问题，可在任何地方使用。](https://github.com/danielmiessler/fabric): fabric 是一个利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过众包的 AI Prompt 集来解决特定问题，可在任何地方使用。 - ...

  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1217063367987036210)** (12 条消息🔥):

- **分享 Perplexity.ai 搜索查询**：成员们正在分享涵盖各种主题的 Perplexity AI 搜索结果直接链接，包括 [改进策略](https://www.perplexity.ai/search/how-to-improve-Y2DqaEI_SombxxyCHrGp6Q)、[睡眠建议](https://www.perplexity.ai/search/how-to-sleep-mQQQzlxTRS6cmFGW5gWYHw)、[Catch-22](https://www.perplexity.ai/search/what-does-catch22-tarYNpckRza6.nusdfqrwA) 的含义，以及关于 [OpenAI Chat GPT](https://www.perplexity.ai/search/OpenAI-Chat-GPT-js98IaEVTCK2EaVDOtt21w) 的信息。
- **通过 YouTube 探索 AI 和技术**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=GTxNncK47Sk)，探讨了各种 AI 新闻，包括 Midjourney 与 Stability AI 之间的争议，以及 "Digital Marilyn" 的推出。
- **医学术语查询**：还进行了关于医学术语的查询，并提供了了解 [azotemia](https://www.perplexity.ai/search/what-is-azotemia-i6R67U4.RBiCZ9.ZZx1tnw)（氮质血症）的链接。
- **利用 AI 评估图像和业务指标**：分享了利用 AI [描述图像](https://www.perplexity.ai/search/Describe-this-image-DjrHWogKQAqMt4Y.HGfGgg)和解释 [net promoter score](https://www.perplexity.ai/search/What-net-promoter-J.ivjnkwTzadZtlLDmNUqg)（净推荐值）概念的搜索链接。
- **纪念 Paul Alexander**：一位成员通过分享关于 Paul Alexander 逝世的搜索链接来纪念他，[表彰他的成就](https://www.perplexity.ai/search/Paul-Alexander-dies-b0bCPk1jSxSu7bag8JApDQ)，并称他为“绝对的强者 (absolute chad)”。

**提到的链接**：

[Midjourney bans Stability staff, Marilyn Monroe AI Debut, Vision Pro aids spine surgery](https://www.youtube.com/watch?v=GTxNncK47Sk)：本集探讨了最新的 AI 新闻，包括 Midjourney 与 Stability AI 之间激烈的模型训练数据抓取争议、创新的 "Digital Marilyn" ...

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1217048057242189864)** (16 条消息🔥): 

- **思考 Prompt 的完美化**：一位成员强调，通过调整 **Prompting** 以及 *max_tokens* 和 *temperature* 等参数设置，可以从模型中获得**简洁的回答**。
- **通过 Embeddings 实现记忆**：有人建议，为了让聊天机器人模型使用 Perplexity API **“记住对话”**，需要一个外部**数据库**来存储过去对话的 Embeddings。
- **寻求特定来源信息**：有人提出疑问，是否可以通过 Prompt 引导 Perplexity API **仅回复网站链接**或其引用的信息来源。
- **高上下文模型需求**：一位用户询问是否可以添加 **Yarn-Mistral-7b-128k** 以用于更高上下文的使用场景。
- **准确性焦虑与搜索**：
  - 表达了对使用 *-online* 标签时结果准确性的担忧，特别是关于不存在的研究和错误的作者归属。一种可能提高准确性的方法是设置 **System Message**，以提供独立于用户查询的清晰指令。分享了一个 [Perplexity API 文档链接](https://docs.perplexity.ai/reference/post_chat_completions)，用于实验 System Message。

**提到的链接**：

[Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)：未找到描述

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1217062828725370880)** (224 条消息🔥🔥): 

- **分享 Kernel 改进**：讨论了对 RoPE Kernel 的一项改进，其中 sin/cos 函数在 axis 1 上加载一次以进行计算，从而提高效率。顺序计算用于分组头 (grouping heads)，并提到已提交了一个 Pull Request (PR)，尽管对其对整体训练性能的影响有些顾虑。

- **Unsloth 讨论与贡献**：强调了 Unsloth 的性能和用法，并提到 Unsloth Studio (Beta) 是即将推出的功能，支持一键 Finetuning。解决了在 Kaggle 上导入 FastLanguageModel 的问题，建议手动重新安装某些库。

- **Grok AI 推测与 OpenAI 对话**：对话涉及 Elon Musk 发布开源 Grok AI 模型的意义以及最近的 OpenAI 事件。关于实时 Twitter 数据流集成对模型性能影响的辩论也随之展开。

- **即将推出的开源 AI 模型**：聊天中提到了其他 AI 社区对 GemMoE、Grok AI 等模型开源发布的兴奋，并讨论了 OpenAI 是否会效仿。人们对模型被盗版的可能性表示担忧，并讨论了 OpenAI 与 Microsoft Azure 的合作。

- **技术支持与更新**：解决了诸如使用 Unsloth 导入和微调模型的技术问题，包括微调后加载到 Transformers 中。提供了 Unsloth 的 GitHub wiki 链接以供查询常见问题（FAQs），并为 DPO 配对样式数据的数据集格式化提供了支持。

**提到的链接**：

- [Crystalcareai/GemMoE-Beta-1 · Hugging Face](https://huggingface.co/Crystalcareai/GemMoE-Beta-1)：未找到描述
- [Docker](https://hub.docker.com/u/winglian)：未找到描述
- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01)：未找到描述
- [Paper page - Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU](https://huggingface.co/papers/2403.06504)：未找到描述
- [Models](https://huggingface.co/docs/peft/en/package_reference/peft_model#peft.prepare_model_for_kbit_training)：未找到描述
- [Home](https://github.com/unslothai/unsloth/wiki)：速度提升 5 倍，显存占用减少 60% 的 QLoRA 微调。通过在 GitHub 上创建账户，为 unslothai/unsloth 的开发做出贡献。
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth/)：速度提升 5 倍，显存占用减少 60% 的 QLoRA 微调。通过在 GitHub 上创建账户，为 unslothai/unsloth 的开发做出贡献。
- [Unsloth free vs. 2x GPUs video outline](https://docs.google.com/document/u/0/d/1YRhwRMkXZ8uiRYwsPaIjZ_vUYrKHAH2GlIJFkNgeTy8/mobilebasic?pli=1)：未找到描述
- [Comparative LORA Fine-Tuning of Mistral 7b: Unsloth free vs. Dual GPUs](https://youtu.be/d1xbMfvUPik?feature=shared)：本视频探讨了 AI 模型训练效率的前沿技术，重点关注增强 Mistral 7b 的新闻文章摘要能力。
- [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark)：我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了滑动窗口注意力机制（sliding window attention）、初步的 Windows 和 DPO 支持，以及...
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth#-finetune-for-free)：速度提升 5 倍，显存占用减少 60% 的 QLoRA 微调。通过在 GitHub 上创建账户，为 unslothai/unsloth 的开发做出贡献。
- [axolotl/docker at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/docker)：尽管提出 axolotl 问题。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [GitHub - PerspectiveDataScience/Testing_Unsloth_v_2GPUs_LORA: Script to test the speed to train a LORA fine tune of Mistral 7b using Unsloth with 1 GPU versus using 2 GPUS](https://github.com/PerspectiveDataScience/Testing_Unsloth_v_2GPUs_LORA)：用于测试使用 Unsloth 在单 GPU 与双 GPU 环境下对 Mistral 7b 进行 LORA 微调训练速度的脚本。

---

**Unsloth AI (Daniel Han) ▷ #[welcome](https://discord.com/channels/1179035537009545276/1179039724355211325/1217117766427873290)** (9 条消息🔥): 

- **充满表情符号的热情问候**：成员用简单的 "coucou" 向频道打招呼。
- **了解基本要点**：theyruinedelise 强调在加入时阅读频道规则并设置角色，引导新用户前往特定频道。
- **工具与问候的友好致意**：pacozaa 分享了热情的表情符号，表示兴奋并准备好构建或开展工作。
- **简单的问候**：aleccol 和 theyruinedelise 都在欢迎频道向新人提供了简短的问候。

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1217080595239276634)** (6 条消息): 

- **WSL 环境中的依赖地狱**：一位用户在 **Ubuntu WSL** 上遇到了涉及 **bitsandbytes**、**triton**、**torch** 和 **xformers** 的依赖循环，导致 **torchaudio**、**torch** 和 **torchvision** 版本不兼容。这个问题阻碍了一个极具前景的 Gradio 界面项目的进展，该项目旨在利用 YouTube 视频和播放列表创建数据集，包含处理选项、JSONL 生成以及使用 **Unsloth** 进行微调。

- **解决依赖问题的建议**：另一位用户建议尝试直接从 PyTorch 的 **cu121** 索引安装 **xformers**，然后从 GitHub 的 nightly 版本安装 **Unsloth**。这被提议作为解决当前版本冲突的潜在变通方案。

- **耗时任务进度缓慢**：一位成员对任务进展缓慢表示沮丧，当前状态显示剩余完成时间超过 74 小时。使用大哭的表情符号强调了这种不满。

**提到的链接**：

[无标题](https://download.pytorch.org/whl/cu121)：未找到描述

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1217034999128260678)** (63 messages🔥🔥): 

- **解决 Tokenization 难题的模型切换方案**：一位社区成员尝试使用参数为 `max_length=4096, truncation=True` 的 tokenizer 但未成功，后来发现通过切换到备选模型并减少数据集长度解决了该问题。

- **量化难题探讨**：一位用户在尝试使用三块 V100 32GB GPU 对 Mixtral 模型进行量化时遇到了显存溢出（out-of-memory）问题。他们收到的建议是尝试在更强大的硬件（如 2x A100 80GB）上进行量化，这应该足够了；此外，他们还被引导尝试其他方法，如 Unsloth 对 GGUF 的内置支持。

- **Xformers 与 FA2 框架对决**：讨论了 xformers 和 FA2 之间的比较，一名成员指出 xformers 慢约 0.5%，并澄清 xformers 和 FA2 具有类似的功能。

- **Nous-Hermes 模型的 Padding 问题**：一位用户在尝试通过 Unsloth 直接使用 Nous-Hermes 模型（未进行微调）时遇到了 `<unk>` 值的问题。建议认为问题可能出在 padding 上，并建议尝试使用 Nous 的 chat template。

- **Xformers 安装问题**：一位成员尝试通过 conda 安装 xformers 失败，并遇到了导致无法安装任何包的 Python 错误；他们被告知其 Python 或 conda 环境可能已损坏，并建议重新安装 Conda，并尝试在 WSL 而非 Windows 上进行安装。

**提到的链接**：

[no title found](https://download.pytorch.org/whl/cu118): 未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1217210311120257125)** (17 messages🔥): 

- **探索使用 NVMe SSDs 进行大模型微调**：分享了一篇概述使用 [NVMe SSDs 在单个 GPU 上进行 100B 模型微调](https://arxiv.org/abs/2403.06504) 潜力的论文，强调了当前 GPU 显存限制带来的挑战，以及 SSD 可能如何帮助解决海量模型微调中的这些问题。
- **支持 CUDA 的用户空间 NVMe 驱动**：介绍了一个用于在模型微调中实现 NVMe SSD 的潜在工具 [ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma)，该工具提供了构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用的资源。
- **NVMe 与 GPUDirect 演讲**：提到了 Jonas Markussen 在 GTC 2019 上的[一场演讲](https://developer.nvidia.com/gtc/2019/video/S9563)，其中讨论了在 PCIe 网络中使用 NVMe 和 GPUDirect 进行高效的分布式存储 I/O，观看可能需要 NVIDIA Developer Program 会员资格。
- **FlashNeuron 贡献机会**：分享了 GitHub 上的 [flashneuron 项目](https://github.com/SNU-ARC/flashneuron/blob/master/README.md)，指出了另一个为 GPU 存储应用开发做出贡献的机会。
- **NVIDIA 的 GPU 直接数据路径 API**：重点介绍了 NVIDIA GPUDirect® Storage 的文档，展示了一个[用于文件读写的 API](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)，该 API 允许在 GPU 显存和存储之间进行直接内存访问传输，绕过 CPU。

**提到的链接**：

- [GTC Silicon Valley-2019: Efficient Distributed Storage I/O using NVMe](https://developer.nvidia.com/gtc/2019/video/S9563): 未找到描述
- [Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU](https://arxiv.org/abs/2403.06504): 大型语言模型的最新进展为世界带来了巨大价值，其卓越的能力源于它们使用的海量参数。然而，即使是 GPU...
- [Question could direct nvme access boost training? · Issue #31 · AnswerDotAI/fsdp_qlora](https://github.com/AnswerDotAI/fsdp_qlora/issues/31): 嘿，我非常喜欢降低训练资源需求这个目标！在这篇论文 https://arxiv.org/abs/2403.06504 中，他们声称 GPU 与 Nvme 存储之间的直接内存访问是...
- [GitHub - enfiskutensykkel/ssd-gpu-dma: Build userspace NVMe drivers and storage applications with CUDA support](https://github.com/enfiskutensykkel/ssd-gpu-dma): 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用 - enfiskutensykkel/ssd-gpu-dma
- [flashneuron/README.md at master · SNU-ARC/flashneuron](https://github.com/SNU-ARC/flashneuron/blob/master/README.md): 通过在 GitHub 上创建账号来为 SNU-ARC/flashneuron 的开发做出贡献。
- [cuFile API Reference Guide - NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html): 未找到描述

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1217026590383276092)** (117 messages🔥🔥):

- **电脑爆炸与力量展示**：成员们分享了各种来自 [Tenor 的 GIF](https://media1.tenor.com/m/5XD6rt5CHusAAAAd/pc-exploding.gif)，其中包括一个贴有 "pc"、"Exploding"、"Minecraft" 和 "Rtx" 等标签的电脑爆炸动画。
- **关于微调与模型运行的咨询**：成员们询问了 **LM Studio** 是否允许对大语言模型 (LLMs) 进行微调，得到的确认回答是否定的；此外还讨论了如何让 WhiteRabbit 模型正常工作，建议是避开 TheBloke 发布的特定损坏版本，寻找 insomnium 发布的版本。
- **处理双 GPU 配置**：关于 LM Studio 将两个 GPU 识别为具有合并 VRAM 的单个设备的查询引发了关于跨多个显卡管理模型的讨论。建议包括将大型模型拆分到两个 GPU 以获得更好的性能，以及可能使用 tensor split 配置在单个显卡上运行独立的实例。
- **本地 LLM 用户指南发布**：一位用户分享了他们的 [本地 LLM 用户指南](https://github.com/xue160709/Local-LLM-User-Guideline)，获得了赞赏和建设性的反馈，包括要求遵守标准化的日期格式以及省略时间性语言。
- **开发资源可用性**：成员们分享了相关资源链接，例如名为 [Rivet](https://rivet.ironcladapp.com/) 的开源视觉 AI 编程环境、[关于在 Rivet 中使用本地 LLM 的教程](https://www.youtube.com/watch?v=vyzNkWYIcac&)，以及 Rentry.org 上全新的综合性非官方 [LM Studio FAQ](https://rentry.org/LMSTudioFAQ)，以帮助用户了解 LM Studio 的功能和局限性。

**提到的链接**：

- [Poe - 快速、实用的 AI 聊天](https://poe.com/)：未找到描述
- [Rivet](https://rivet.ironcladapp.com/)：一个使用可视化、基于节点的图形编辑器的开源 AI 编程环境
- [非官方 LMStudio FAQ！](https://rentry.org/LMSTudioFAQ)：欢迎来到非官方 LMStudio FAQ。在这里，你可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源软件...
- [OpenRouter](https://openrouter.ai/)：LLM 和其他 AI 模型的路由
- [I Have The Power GIF - He Man I Have The Power Sword - 发现并分享 GIF](https://tenor.com/view/he-man-i-have-the-power-sword-gif-5305079)：点击查看 GIF
- [Pc Exploding GIF - Pc Exploding Minecraft - 发现并分享 GIF](https://tenor.com/view/pc-exploding-minecraft-rtx-gif-25263106)：点击查看 GIF
- [Rivet: 如何同时使用本地 LLM 和 ChatGPT (LM Studio 教程)](https://www.youtube.com/watch?v=vyzNkWYIcac&)：本教程解释了如何将 LM Studio 与 Rivet 连接，以使用运行在你自己电脑上的本地模型（例如 Mistral 7B），同时也介绍了如何仍然能够使用...
- [llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - ChatGPTNextWeb/ChatGPT-Next-Web: 跨平台 ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS)。一键拥有你自己的跨平台 ChatGPT/Gemini 应用。](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web?tab=readme-ov-file)：跨平台 ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS)。一键拥有你自己的跨平台 ChatGPT/Gemini 应用。 - ChatGPTNextWeb/ChatGPT-Next-Web
- [GitHub - xue160709/Local-LLM-User-Guideline](https://github.com/xue160709/Local-LLM-User-Guideline)：通过在 GitHub 上创建账户为 xue160709/Local-LLM-User-Guideline 的开发做出贡献。
- [人工智能法案：欧洲议会议员通过里程碑式法律 | 新闻 | 欧洲议会](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law)：周三，议会批准了《人工智能法案》，该法案在确保安全并符合基本权利的同时，促进了创新。

---

**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1217026385231478834)** (37 条消息🔥): 

- **数学奇才的 LaTeX 梦想**：成员们兴奋地讨论了 **LM Studio** 支持数学问题 LaTeX 代码的可能性，并提到了一篇关于数学表达式 Markdown 渲染的 [Github 博客文章](https://github.blog/changelog/2022-05-19-render-mathematical-expressions-in-markdown/)。他们强调了这为 UI 带来的便利，特别是对于数学 Markdown 等领域。

- **ReactJS 实现的潜力**：一位成员链接了 LogRocket 上的一篇关于[如何使用 React Markdown 安全地渲染 Markdown](https://blog.logrocket.com/how-to-safely-render-markdown-using-react-markdown/) 的文章，建议类似的方法可以用于像 Discord 这样使用 Electron 的 UI 中的数学 Markdown。

- **LaTeX 生成寄予厚望**：在讨论中，氛围比较轻松，成员们开玩笑地表达了对更好数学解析的需求，包括像“滑动式黑板”那样的视觉输入，以及为视觉模型提供更高的图像分辨率，同时还有一些关于讨厌数学的讽刺俏皮话。

- **探索最先进的语言模型**：一位用户分享了 Hugging Face 上的 [Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) 链接，并引发了关于其 128k tokens 长上下文支持的解释，以及其他教育类链接，如针对语言模型的 [Perplexity (PPL)](https://huggingface.co/docs/transformers/perplexity) 解释。

- **玩笑与学习交织**：聊天氛围因关于命名方案的幽默评论和来自 Tenor 的相关 GIF 分享而变得活跃，反映了成员们在技术对话中的幽默感和情谊。

**提到的链接**：

- [How to safely render Markdown using react-markdown - LogRocket Blog](https://blog.logrocket.com/how-to-safely-render-markdown-using-react-markdown/)：通过这篇简短的 react-markdown 教程，学习如何安全地将 Markdown 语法渲染为相应的 HTML。
- [NousResearch/Yarn-Mistral-7b-128k · Hugging Face](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k)：未找到描述
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)：Rotary Position Embeddings (RoPE) 已被证明能有效编码基于 Transformer 的语言模型中的位置信息。然而，这些模型无法泛化到超过序列长度 t 的范围...
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)：未找到描述
- [Im Waiting Daffy Duck GIF - Im Waiting Daffy Duck Impatient - Discover &amp; Share GIFs](https://tenor.com/view/im-waiting-daffy-duck-impatient-gif-16985061)：点击查看 GIF
- [Yeah Another Day Lets Do It Bojack GIF - Yeah Another Day Lets Do It Bojack Will Arnett - Discover &amp; Share GIFs](https://tenor.com/view/yeah-another-day-lets-do-it-bojack-will-arnett-bojack-horseman-encouraged-gif-16252191)：点击查看 GIF
- [Calculation Math GIF - Calculation Math Hangover - Discover &amp; Share GIFs](https://tenor.com/view/calculation-math-hangover-allen-zach-galifianakis-gif-6219070)：点击查看 GIF
- [Render mathematical expressions in Markdown](https://github.blog/changelog/2022-05-19-render-mathematical-expressions-in-markdown/)：在 Markdown 中渲染数学表达式
- [Feast Your Eyes: Mixture-of-Resolution Adaptation for Multimodal Large Language Models](https://arxiv.org/html/2403.03003v1)：未找到描述

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1217040322564063263)** (76 条消息🔥🔥): 

- **探索 LLM 的 GPU 配置**：用户讨论了各种 GPU 设置的有效性，包括使用双 RTX 4060 Ti 16GB 配合 PCI-e x1 转 x16 转接器，并比较了使用不同 VRAM 容量时的性能。分享了一个 YouTube 视频（[Testing single GPU vs two GPUs with LM Studio LLMs](https://youtu.be/RO1KnARCHcc)），详细介绍了一个 8.5 小时的性能测试，并评论了从单 GPU 扩展到双 GPU 时波动的性能提升。

- **NVLINK 桥接成本与变通方案**：成员们辩论了 NVLINK 桥接器的高昂成本，有人声称购买 RTX 2060 时免费获赠了一个，而其他人则评论其价格昂贵，并幽默地提到使用更便宜的替代品。提到了一个关于逆向工程 NVLINK 的 [Linus Tech Tips 论坛帖子](https://linustechtips.com/topic/1290094-donating-my-4-slot-nvlink-to-science/)，推测其内部组件。

- **Mac 和 Windows 设置中的内存难题**：用户讨论了在 macOS 上绕过最低 VRAM 要求的可能性和弊端，提到系统冻结可能是后果之一。提出了提升性能的建议，包括将 NVMe SSD 用作 RAM，尽管有人指出这会很慢。

- **RAM 升级前景与性能考量**：一位成员考虑升级到 128GB RAM，以便运行更大的模型并提高 LLM 的准确性。其他成员建议，与更多 RAM 相比，更多 VRAM 至关重要，并讨论了运行多个高端 GPU 的电力需求，一位用户分享了为四个 3090 准备的双电源设置。

- **在单 GPU 与多 GPU 上运行多个 LLM 实例**：一位成员探索了同时运行两个 LM Studio 实例的可能性，并分享了在配对 VRAM 容量不等的 GPU 时必要的配置调整见解。指出虽然可以运行多个模型，但个人测试显示，在 PCI-e x1 带宽场景下，单 GPU 与双 GPU 设置之间没有显著的性能损失。

**提到的链接**：

- [Cerebras Systems 发布全球最快 AI 芯片，拥有 4 万亿个晶体管和 900,000 个 AI 核心](https://www.techpowerup.com/320294/cerebras-systems-unveils-worlds-fastest-ai-chip-with-4-trillion-transistors-and-900-000-ai-cores)：作为加速生成式 AI 的先驱，Cerebras Systems 推出了 Wafer Scale Engine 3，在其现有的最快 AI 芯片世界纪录上再次翻倍。WSE-3 提供了两倍的...
- [在 LM Studio LLMs (AI Chat) 中测试单 GPU 与双 GPU：两块 RTX 4060 Ti 16GB，其中一块使用 X1 适配器](https://youtu.be/RO1KnARCHcc)：系统规格：Ryzen 7 5800X3D 8 核 CPU (16 线程)，128GB DDR4 2800MHz CL16 RAM (4X32GB dual rank，在 3000MHz 时不稳定)，Zotac RTX 4060 Ti 16GB (HDMI dummy...

  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1217179074250936360)** (4 条消息): 

- **关于 AVX Beta 更新的咨询**：一名成员询问 AVX beta 是否会有更新，特别是 0.2.10 版本之后的更新。

- **真实性确认**：一名成员对某项陈述的真实性提出质疑，并得到了肯定的回复。

- **对质量的评价**：一名成员通过陈述讨论对象“*不怎么好*”表达了对质量的看法。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1217065628779679844)** (73 条消息🔥🔥): 

- **GPU 利用率的提升**：从 AMD 的 23.Q4 PRO 驱动切换到 Adrenalin 24.1.1 后，用户在 LM Studio 中的性能体验比 OpenCL 快了 2 倍以上。他们注意到利用率大幅提升，并建议不要使用 24.2.1 更新，因为存在问题。
- **禁用 iGPU 可解决 Offloading 问题**：一些用户发现禁用集成显卡 (iGPU) 可以成功完成初始步骤之后的 Offloading，从而优化了系统性能。
- **ROCm Beta 安装问题及修复**：用户讨论了在 LM Studio 中为 AMD GPU 使用 ROCm beta 版本的必要性，并提出了诸如重新安装驱动程序、确保安装 HIP SDK 以及启动新 Prompt 以改善 GPU 加速等解决方案。
- **在 AMD GPU 上使用 ROCm 的技巧**：交流了将 GPU layers 设置为最大值以及使用 Adrenalin 24.1.1 + HIP-SDK 等特定驱动组合以增强性能的建议；一些用户还注意到 Vision 模型无法正常工作的问题。
- **模型加载错误排查**：建议在 LM Studio 上安装 ROCm 构建版本后，清理缓存目录并重新下载模型，以解决加载先前下载模型时出现的错误。用户还建议针对不同模型使用特定的预设以确保兼容性。

**提到的链接**：

[👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai/rocm)：查找、下载并实验本地 LLMs

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1217050037125779551)** (136 条消息🔥🔥): 

- **纸牌游戏指令机器人咨询**：一名大学生提出了一个项目，旨在开发一个能够提供纸牌游戏指令的机器人，该机器人使用摄像头拍摄笔记本电脑屏幕，并使用 Python 中的 OpenCV 进行分析。建议包括研究纸牌游戏、安排合适的硬件、编写图像处理代码、集成 GPT 进行游戏分析，以及迭代测试和更新设计。

- **探索 Sora 的功能与访问权限**：讨论集中在视频生成 AI 模型 **Sora** 的成本和可用性上。据指出，虽然 Sora 目前成本高昂且未向公众开放测试，但一些视觉艺术家和电影制作人已获得访问权限以提供反馈。

- **使用 GPT Chat 处理大型 Google Sheets**：用户讨论了让 GPT Chat 分析大型数据集的策略，建议包括将数据拆分到多个工作表中进行批处理，或使用数据库和 SQL 查询进行更高效的数据处理。

- **关于 GPT-4.5 Turbo 的传闻**：成员们交流了关于 Bing 和其他搜索引擎上提到的 **GPT-4.5 Turbo** 模型的未经证实的传闻。然而，尚未发现来自 OpenAI 的官方来源或确认，因此共识是该信息可能并不准确。

- **对自托管 AI 模型的兴趣**：一名新成员询问了关于自托管 AI 模型供个人使用的问题，特别是用于处理个人日记数据。其他用户就运行大型模型的硬件要求提供了建议，并表示利用 OpenAI 平台进行 Fine-tuning 可能更具可行性。

**提到的链接**：

[notebooks/mistral-finetune-own-data.ipynb at main · brevdev/notebooks](https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb)：通过在 GitHub 上创建账号为 brevdev/notebooks 的开发做出贡献。

  

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1217049703905230848)** (57 messages🔥🔥): 

- **训练后的学习？**：一位用户询问受限于 32k tokens 的语言模型是否可以处理 232k token 的 PDF，另一位用户澄清说**模型一次只能内化 32k tokens**，但会根据需要在大文档中进行搜索和总结。

- **订阅价格的经济约束**：来自**格鲁吉亚**和**巴西**的成员分享了对 AI 服务高昂成本（相对于当地最低工资）的担忧，主张推行**区域定价**以照顾经济困难国家的用户。

- **GPT 和 Assistant API 故障**：用户遇到了 GPT 宕机的问题并寻求帮助，而另一位用户在使用 Assistant API 时遇到了挑战，该 API 无法正确解析逗号分隔的数字。

- **实时状态检查与权宜之计**：在 GPT-4 停机期间，用户分享了技巧和更新；建议包括检查 **[status.openai.com](https://status.openai.com/)** 以及临时解决方案，如通过发送图片在 App 中开启对话。

- **社区礼仪**：在出现关于经济地位的评论后，有必要提醒大家进行尊重他人的沟通，同时其他人强调了对**真实信息**的需求以及与外部链接交互时需保持谨慎。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1217049026936176660)** (46 messages🔥): 

- **针对字数统计的创意 Prompt 编写**：用户讨论了如何指示 AI 在改进原始写作并保留要点的同时，以大约相同的字数重写文本。建议包括使用正面指令构建自定义 AI Prompt 并避免特定的比例，一位旨在获得一致输出的用户提供了一个复杂的 Prompt。
- **使用 Code Interpreter 解决计数难题**：关于 AI 是否可以使用 Code Interpreter 准确计算字数，以及直接指示计数并调整字数是否有助于达到预期结果进行了对话。一位用户确认已测试过 Code Interpreter 的计数与标准字数统计器一致。
- **重置补救措施**：聊天中确认，与 AI 开启新对话通常可以解决异常或不合规的行为。这归因于每次新交互时访问的训练数据具有随机性。
- **处理非标准数字格式**：一位用户询问了 Assistant API 无法识别数字中的逗号导致解释错误的问题。另一位用户建议，同时提供正面和负面示例可能会纠正 AI 对异常逗号位置的处理。
- **CustomGPT 的 PDF 检索与网页搜索**：有用户请求帮助改进一个可以查阅其数据库中的 PDF 并查找网页信息的自定义 GPT。然而，对方澄清说明确的指令是必要的，且 AI 在识别 PDF 内的图像方面存在局限性。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1217049026936176660)** (46 messages🔥): 

- **刷新对话以重置 AI 异常行为**：用户发现开启*新对话*可以解决异常的 AI 响应，这源于 AI 获取了不同的“训练数据桶”，这可能会影响其表现。
- **旨在获得一致输出的 Prompt 优化建议**：*darthgustav* 提供了一个详细的 Prompt 模板，以帮助 *ericplayz* 在城市项目文本重写中实现更一致的字数和格式，包括用于列表的 Markdown 以及用于字数对齐的迭代优化。
- **使用 Code Interpreter 实现精确字数统计**：*darthgustav* 确认 Code Interpreter 可以有效地统计给定文本的字数，这对于确保重写内容符合特定的字数要求非常有用。
- **上下文格式对 Assistant API 的影响**：一位成员讨论了 Assistant API 可能将 "450,00" 之类的格式误解为 "45000"，建议需要正面和负面示例来纠正这种行为。
- **在 CustomGPT 中利用 PDF 内容和网页信息**：在改进 CustomGPT 以引用 PDF 和网页信息时，*darthgustav* 指出需要明确的指令，并提到模型无法解释 PDF 内部的图像。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1217052901613244457)** (99 messages🔥🔥): 

- **GPT 与 Claude 在论文摘要方面的对比**：一位用户报告使用 **Claude 3** 通过向 AI 提问来总结学术论文，但提醒说不建议用它来寻求特定细节，尽管它擅长解析学术语言。

- **新成员介绍**：**Raman**，一位从计算化学家转型的 ML 科学家，在 active learning 和 reinforcement learning 方面拥有丰富经验，他寻求关于如何参与语言模型相关项目的建议。

- **AI 社区求职**：一位曾参与 EleutherAI 的 **Polyglot** 和 **OSLO** 项目的被裁员成员正在寻找 ML Engineer/Researcher 的新机会，并愿意接受异地入职。

- **Batch Size 与学习权衡的讨论**：成员们就并行训练中扩大 batch size 时的效率和成本权衡进行了详细讨论，并参考了 scaling rules 和 *Pythia* 的训练方法。

- **DeepMind 的游戏通用 AI Agent 引发质疑**：用户对 DeepMind 发布通用 AI **SIMA** 表示怀疑，理由是技术报告中缺乏关于数据、权重以及该 Agent 总体用途的细节。

**提到的链接**：

- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)：在越来越多的领域中，已经证明深度学习模型可以使用相对较大的 batch size 进行训练，而不会牺牲数据效率。然而，这种做法的局限性在于...
- [How to Scale Hyperparameters as Batch Size Increases](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/)：未找到描述
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_source=twitter&utm_medium=social&utm_campaign=SIMA/)：介绍 SIMA，一个可扩展、可指令化的多世界 Agent
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_sour)：介绍 SIMA，一个可扩展、可指令化的多世界 Agent
- [Byuntear American Psycho GIF - Byuntear American Psycho Staring - Discover &amp; Share GIFs](https://tenor.com/view/byuntear-american-psycho-staring-thinking-gif-26991038)：点击查看 GIF
- [Sadhika Malladi](https://www.cs.princeton.edu/~smalladi/blog)：未找到描述
- [On the SDEs and Scaling Rules for Adaptive Gradient Algorithms](https://arxiv.org/abs/2205.10287)：将随机梯度下降 (SGD) 近似为随机微分方程 (SDE)，使研究人员能够享受研究连续优化轨迹的好处，同时谨慎地...

---

**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1217027263871189063)** (84 条消息🔥🔥): 

- **揭穿训练中 Beta2 的迷思**：关于“大模型中 *beta2 应该保持在较低水平（如 0.95），否则会危及训练稳定性*”的说法被指为谎言。讨论中澄清了这一点并非绝对，并探讨了避免这种必要性的不同方法，例如使用像 **ADAM** 这样的替代优化器，其中 beta2 在大值时实际上会退化为 **SGD**。

- **模型黑客行为的伦理边界**：有人询问了在未经明确许可的情况下进行模型黑客攻击的伦理问题。针对模型攻击的 **disclosure policy**（披露政策）受到了质疑，有人指出，正如最近的一篇论文所展示的，获得此类攻击的许可可以消除伦理疑虑。

- **探索用于预训练的 LoRA**：讨论了将 **LoRA**（low-rank adaptation）扩展到模型预训练的概念，并引用了一篇介绍 **LoRA-the-Explorer (LTE)** 的论文，该方法允许并行训练多个低秩头。评论建议将其与 **GradientLoRA** 或 **GaLore** 等其他方法结合，以实现更高效的预训练方式。

- **分析神经崩溃与轨迹**：重点讨论了一篇关于 **Deep Neural Collapse (DNC)** 的论文，探讨了通过 **average gradient outer product (AGOP)** 出现的 DNC 现象。在另一个讨论串中，有人指出深度网络的训练轨迹实际上是一个有效的低维流形，无论架构或规模如何，这引发了关于混合架构（mixture-of-architectures）模型可能具有优势的建议。

- **针对摘要模型的对抗性字符串**：提到了一篇最近的论文，其中列出了对抗性字符串，当输入模型时，可能会导致模型输出非标准响应（如输出 100 次 "HONK"）。有人提出了在摘要模型上测试这些字符串的可能性。

**提到的链接**：

- [Average gradient outer product as a mechanism for deep neural collapse](https://arxiv.org/abs/2402.13728)：平均梯度外积作为深度神经坍缩（Deep Neural Collapse, DNC）的一种机制。深度神经坍缩（DNC）是指深度神经网络（DNN）最后几层中数据表示呈现出令人惊讶的僵化结构。尽管这一现象已在广泛的...
- [The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold](https://arxiv.org/abs/2305.01604)：许多深度网络的训练过程探索相同的低维流形。我们开发了信息几何技术来分析深度网络在训练过程中预测轨迹。通过检查底层的高维概率模型，我们揭示了...
- [Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM](https://arxiv.org/abs/2403.07816)：Branch-Train-MiX：将专家 LLM 混合成混合专家（Mixture-of-Experts）LLM。我们研究了训练大语言模型（LLM）的高效方法，使其具备多个专业领域的技能，如编程、数学推理和世界知识。我们的方法名为...
- [Training Neural Networks from Scratch with Parallel Low-Rank Adapters](https://arxiv.org/abs/2402.16828)：使用并行低秩适配器（Parallel Low-Rank Adapters）从头开始训练神经网络。深度学习模型的可扩展性从根本上受到计算资源、内存和通信的限制。虽然像低秩自适应（LoRA）这样的方法降低了模型微调的成本...
- [Negating Negatives: Alignment without Human Positive Samples via Distributional Dispreference Optimization](https://arxiv.org/abs/2403.03419v1)：否定负面：通过分布去偏好优化（Distributional Dispreference Optimization）在没有人类正样本的情况下进行对齐。大语言模型（LLM）彻底改变了 AI 的角色，但也带来了传播不道德内容的潜在风险。对齐技术被引入以引导 LLM 走向人类...
- [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/abs/2304.11158)：大语言模型（LLM）中涌现且可预测的记忆现象。记忆，即大语言模型（LLM）逐字输出训练数据中整个序列的倾向，是安全部署语言模型的关键担忧。特别是，它...
- [Predicting LoRA weights · Issue #6 · davisyoshida/lorax](https://github.com/davisyoshida/lorax/issues/6#issue-2043938181)：预测 LoRA 权重。我想使用一个独立的神经网络来为主要神经网络预测 LoRA 权重，同时训练这两个神经网络。我该如何操作 pytrees 以实现...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1217139220553011311)** (3 messages): 

- **对模型支持的困惑**：一位成员询问某个工具是否仅支持来自自定义库的模型，而不支持训练好的 Hugging Face checkpoints。
- **通过 Pythia 了解模型去学习（Unlearning）的见解**：讨论重点介绍了一篇关于使用 Pythia 进行可解释性驱动的能力去学习的论文，旨在通过修剪重要神经元来移除某些能力。他们指出，虽然针对特定能力的语言模型性能有所下降，但保留集（retain set）的性能也出现了意外下降，论文可在此处[查阅](https://www.semanticscholar.org/reader/59e2e55137a32ea07651cacd4fadc7b15c371a20)。
- **多模态机械可解释性（Mechanistic Interpretability）库发布**：一位成员在 Twitter 上宣布发布了一个多模态机械可解释性库，并邀请大家在这一研究子领域进行合作，分享了公告[链接](https://twitter.com/soniajoseph_/status/1767963316943728779)。

**提到的链接**：

[[PDF] Dissecting Language Models: Machine Unlearning via Selective Pruning | Semantic Scholar](https://www.semanticscholar.org/reader/59e2e55137a32ea07651cacd4fadc7b15c371a20)：一个利用人工智能方法提供高度相关结果和新型过滤工具的学术搜索引擎。

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1217089353977954437)** (8 messages🔥): 

- **对下游任务性能的学习率担忧**：一位成员怀疑，根据 **Llama 超参数**设置的高学习率可能导致下游评估表现不佳，尽管 loss 指标良好；并询问性能是否会随着学习率退火（anneals）而提高。建议通过对当前 checkpoint 进行简短的学习率冷却（cooldown）来测试性能指标是否有所改善。

- **揭示基准测试（Benchmarks）失宠的原因**：讨论了为什么像 SQuAD 这样的基准测试不再受欢迎，原因包括它们对于大模型来说相对容易、已趋于饱和，以及与已经消化了大量数据（如 Wikipedia）的预训练模型的能力不匹配。

- **GPQA 作为模型监督需求的指标**：一位成员提到了 **GPQA 数据集**，该数据集提出了一系列极具挑战性的问题，旨在防止通过 Google 搜索直接获取答案（Google-proof），并指出即使是像 **GPT-4** 这样表现顶尖的 AI 系统在处理该数据集时也感到吃力，这表明它在开发未来 AI 助手技术的可扩展监督方法方面具有价值。引用的数据集详见 [Anthropic 研究人员发表的一篇论文](https://arxiv.org/pdf/2311.12022.pdf)。

- **排行榜的科学价值受到审视**：讨论批评了排行榜在科学和知识扩展方面的真实价值，暗示它们可能更多地是为组织证明其专有服务定价的合理性，而非真正推动理解。

**提到的链接**：

- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022)：我们展示了 GPQA，这是一个由生物、物理和化学领域的专家编写的包含 448 道多选题的挑战性数据集。我们确保这些问题是高质量且极其困难的...
- [The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)：未找到描述
- [Google's Natural Questions](https://ai.google.com/research/NaturalQuestions/leaderboard)：未找到描述

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1217448448643563611)** (1 条消息): 

- **考虑更紧密地跟踪上游 Megatron**：一位成员正在权衡更紧密地跟踪上游 **Megatron** 的好处，以便更好地与 **Transformer Engine** 集成。他们提交了一个 [Pull Request](https://github.com/EleutherAI/gpt-neox/pull/1185)，详细说明了差异，并正在寻求维护者对这一潜在举措的意见。

**提到的链接**：

[Diffs to upstream megatron as a basis for discussion towards TE integration by tf-nv · Pull Request #1185 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1185)：这里有三个提交：一个是 GPT-NeoX 的 megatron 文件夹与当前上游 Megatron-LM 的完整差异。涉及 256 个文件，约 6 万行。然而，大多数文件是完全新增或删除的...

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1217094926953873470)** (123 条消息 🔥🔥): 

- **游说与影响力探讨**：参与者讨论了游说的本质，指出游说者通常是出钱而不是收钱，暗示他们的权力在于影响力和谈判，而非个人利益。
- **沉思 AI 在未来灾难中的角色**：一位用户幽默地推测，他们微调后的 AI 模型可能会导致 AI 引发的灭绝事件，而另一位用户则拿上一次灭绝事件的失败开玩笑。
- **对 AI 驱动治理的怀疑**：对话中包含了对 AI 和过度炒作事件可能导致政府权力过度扩张的怀疑，一位用户讽刺地暗示这种协调活动的不可行性。
- **对版权和 AI 监管的担忧与推测**：对话涉及了 AI 模型权重的潜在版权问题、DMCA 流程的效率与陷阱，以及最近的欧盟 AI 法规，引发了关于此类法律的实用性和执行力的辩论。
- **对大规模推理的 AI 硬件和软件的兴趣**：交流了运行重型 AI 推理任务的最佳硬件和软件选择，重点在于利用 GPUs、框架选择以及本地设置与基于 API 的解决方案之间的权衡。

**提到的链接**：

- [欧洲立法者通过全球首部监管 AI 的主要法案](https://www.cnbc.com/2024/03/13/european-lawmakers-endorse-worlds-first-major-act-to-regulate-ai.html)：欧盟议会周三批准了全球首套监管 AI 的主要基本规则，旨在管理处于科技投资前沿且备受媒体关注的人工智能领域...
- [GroqChat](https://groq.com/)：未找到描述
- [Discord - 与朋友和社区聊天的新方式](https://discord.gg/groq)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [构建 Meta 的 GenAI 基础设施](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：作为对 Meta AI 未来的重大投资，我们宣布推出两个 24k GPU 集群。我们将分享硬件、网络、存储、设计、性能和软件方面的细节，这些细节帮助我们提取...
- [BUD-E (理解与数字共情伙伴) - 蓝图 / 概览](https://youtu.be/bLPDn-bh7dY?si=xrR2_F6kx1ydz8XM)：https://docs.google.com/presentation/d/1tBBa0_GzzfCrmn9KpYZ8YZ9x4Jgb2zVs/edit?usp=sharing&amp;ouid=114592459581752579892&amp;rtpof=true&amp;sd=true
- [2023年9月18日神经化身演示](https://youtu.be/TDitkDKbqbk)：2023年9月18日神经化身演示
- [TheBloke/phi-2-GGUF · Hugging Face](https://huggingface.co/TheBloke/phi-2-GGUF)：未找到描述
- [政府委托报告称，美国必须“果断”行动以规避 AI 带来的“灭绝级”威胁 - Slashdot](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-says)：美国政府必须“迅速且果断”地采取行动，以规避人工智能（AI）带来的重大国家安全风险，在最坏的情况下，这可能会导致“灭绝...”
- [教育项目路演 PPT](https://docs.google.com/presentation/d/1cMWLpMGNGs0_ZcKRKlJqM5OYiTSTyXgn39CDYOcgZq8/edit?usp=sharing)：Navi-Sensei 提案 JusticeDAO LLC Benjamin Barber business@hallucinate.app 10043 Se 32nd Ave Milwaukie Oregon 97222 9712700855 “我，Benjamin Barber，已阅读并理解 OMB 和 OPM 挑战...”
- [Devin：全球首个 AGI Agent（是的，这是真的）](https://youtu.be/ZkcrLOg6lL4)：如果你认真对待 AI 并想了解 AI Agents，请加入我的社区：https://www.skool.com/new-society 关注我获取极速 AI 新闻 - https:/...
- [路演 PPT](https://docs.google.com/presentation/d/1_PejXm_nDP_b_Vig_WcnUh4WkFsSy2U0-ERQP2SD6-4/edit?usp=sharing)：“Justice Now” - AI 法律化身

  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1217030359896686682)** (21 条消息🔥): 

- **CogVLM 幻觉困境**：为了减轻使用 **CogVLM** 生成数据时的幻觉问题，有人建议使用*简单且简短的 prompt*，以降低输出错误的概率。
  
- **Cross Attention 被认为并非最优**：讨论强调 Cross Attention 并非瓶颈，但在添加 conditioning 方面不如其他替代方案有效，因为在每个 block 转换 text embeddings 会为当前图像带来**更好的去噪（denoising）**效果。

- **期待 MoAI 的实现**：一种新的视觉语言模型 **Mixture of All Intelligence (MoAI)** 凭借其论文和吉祥物引起了关注，它声称在体积更小的同时性能优于现有模型。**官方实现**已在 [GitHub](https://github.com/ByungKwanLee/MoAI) 上发布，同时在 [Hugging Face](https://huggingface.co/BK-Lee/MoAI-7B) 上分享了**使用指南**。

- **贡献数据集获得认可**：一位成员高兴地表示，他们的数据集在 **DeepSeekVL** 论文中获得了引用。该论文看起来很有前景，其权重已在提到此事的两天前发布。

- **消除 30B 模型内存优化的误解**：最初关于 30B 模型可以通过 lazy loading 加载到 4GB 内存中的说法被拆穿了；内存使用量被低估了，因为 **mmap 在访问内存之前不会计入使用量**。

**提及的链接**：

- [BK-Lee/MoAI-7B · Hugging Face](https://huggingface.co/BK-Lee/MoAI-7B)：未找到描述
- [MoAI: Mixture of All Intelligence for Large Language and Vision Models](https://arxiv.org/abs/2403.07508)：大语言模型 (LLMs) 的兴起和指令微调 (instruction tuning) 引领了当前指令微调大语言与视觉模型 (LLVMs) 的趋势。这一趋势涉及精心策划的...
- [Jimmy Carter President Carter GIF - Jimmy carter President Carter Carter - Discover &amp; Share GIFs](https://tenor.com/view/jimmy-carter-president-carter-carter-gif-16271386811124661325)：点击查看 GIF
- [30B model now needs only 5.8GB of RAM? How? · ggerganov/llama.cpp · Discussion #638](https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916)：（编辑：抱歉，我最初应该澄清我是在 Linux 操作系统上运行的。我没意识到对于非 Linux 用户来说，仅从截图可能看不出来。所有测试都是在 Ubun...
- [GitHub - ByungKwanLee/MoAI: Official PyTorch implementation code for realizing the technical part of Mixture of All Intelligence (MoAI) to improve performance of numerous zero-shot vision language tasks. (Under Review)](https://github.com/ByungKwanLee/MoAI)：实现 Mixture of All Intelligence (MoAI) 技术部分的官方 PyTorch 实现代码，旨在提高众多 zero-shot 视觉语言任务的性能。（审核中） - Byun...

  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1217107174011044022)** (1 条消息): 

- **探索 LAION-400M 的广阔世界**：一条消息重点介绍了 **Thomas Chaton** 关于使用、探索和利用 **LAION-400-MILLION** 图像与标题数据集进行创作的文章。该成员表达了对这项工作的钦佩，并提供了[文章链接](https://bit.ly/3uYrDCh)。

**提到的链接**：

[Download &amp; stream 400M images + text - a Lightning Studio by thomasgridai](https://bit.ly/3uYrDCh)：从头开始使用、探索并利用 LAION-400-MILLION 图像与标题数据集进行创作。

  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1217150458280280134)** (1 条消息): 

- **深入探讨自编辑长期记忆**：本周五上午 9 点（太平洋时间），与 MemGPT 团队一起讨论语言模型的**长期、自编辑记忆**。网络研讨会将深入探讨 **MemGPT** 的方法，包括“虚拟上下文管理”和 function calling 能力。在此[注册](https://lu.ma/c39w9ofy)参加活动。
- **MemGPT 网络研讨会公告**：LlamaIndex 网络研讨会将涵盖 LLMs 长期记忆的挑战，并介绍 MemGPT 的最新进展。*MemGPT (Packer et al.)* 使用 function calling 进行主动内存管理，为 LLMs 创建一个更动态、更高效的系统。

**提到的链接**：

[LlamaIndex Webinar: Long-Term, Self-Editing Memory with MemGPT · Zoom · Luma](https://lu.ma/c39w9ofy)：LLMs 的长期记忆是一个尚未解决的问题，从 vector database 进行简单的检索并不奏效。MemGPT (Packer et al.) 的最新迭代在这方面迈出了一大步...

  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1217136743242006618)** (4 条消息): 

- **MemGPT 尝试解决 LLMs 长期记忆问题**：**MemGPT** 的新迭代在管理大语言模型 (LLMs) 的长期记忆方面取得了进展，解决了简单 vector database 检索的不足。详情在 [Twitter](https://t.co/VUJBtJqPPT) 上的网络研讨会中介绍。
  
- **Ollama 巴黎开发者见面会**：**Ollama and Friends** 将于 **3 月 21 日** **下午 6 点**在巴黎 **Station F** 举办**开发者见面会**，届时将有食物、饮料以及来自各种开源项目贡献者的闪电演示。感兴趣的人士可以通过 [Twitter](https://t.co/DTyUwHpdC7) 了解更多信息并申请在活动中进行演示。
  
- **将 LlamaIndex 与 MathPix 集成以进行科学查询**：**LlamaIndex** 和 **MathPixApp** 合作将复杂的数学公式解析并索引为 LaTeX，增强了回答与科学论文相关查询的能力。如 [Twitter 帖子](https://t.co/rWX6rGIHub)中所分享，详细指南现已发布，展示了通过表格、图像提取和文本索引的处理过程。

- **LlamaParse 发布**：**LlamaParse** 正式发布，被誉为首个 genAI 原生文档解析解决方案，声称在解析图像、表格和图表方面具有卓越功能，并具有自然语言 steering 能力。关于这一创新解析解决方案的更多细节已在 [Twitter](https://t.co/9MIuP4pkYh) 上介绍。

**提到的链接**：

[本地 & 开源 AI 开发者见面会 (巴黎) · Luma](https://t.co/pAXCqmuvDg): Ollama and Friends 来到巴黎了！Ollama and Friends 将于 3 月 21 日星期四下午 6 点在巴黎 Station F 举办一场本地 & 开源 AI 开发者见面会。欢迎各位开发者前来交流...

---

**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1217045924895789086)** (128 条消息🔥🔥): 

- **关于 Chatbot 和索引性能的讨论**：成员们讨论了针对 Chatbot 响应和内容检索的不同框架及引擎的优化。建议包括将 DeepEval 与 LlamaIndex 结合使用（[DeepEval 文档](https://docs.confident-ai.com/docs/getting-started)），以及应用集成查询引擎（ensemble query engines）以提高响应多样性（[Ensemble Query Engine Colab 教程](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/query_engine/ensemble_query_engine.ipynb)）。

- **VectorStoreIndex 中大型 JSON 数组的挑战**：一位成员寻求关于在使用 LlamaIndex 查询大型 JSON 数组时提高性能的建议。建议将每个对象解析为一个 Node，并将每个 JSON 字段设为 metadata 字段，同时启用日志记录以追踪潜在的性能问题（[追踪与调试文档](https://docs.llamaindex.ai/en/stable/understanding/tracing_and_debugging/tracing_and_debugging.html#basic-logging)）。

- **关于 LlamaIndex 功能的澄清请求**：有人提出了关于 LlamaIndex 的隐私问题和最佳索引组织方式的疑问，但尚未获得社区的明确答复。

- **尝试理解并改进查询和 Chat Engine 性能**：讨论了解决特定 LLM（如 Mistral-large）产生错误、Embedding 模型分数过高以及排查 Chatbot 错误等问题的各种尝试，社区提供了一些初步建议。

- **探索 LlamaIndex 的实现与调试**：成员们探索了不同用例的实现方案，以及针对 LlamaIndex 中 Embedding 模型和处理不支持 metadata 过滤的 Vector Store 等问题的调试过程。

**提到的链接**:

- [Llama Hub](https://llamahub.ai/?tab=storage): 未找到描述
- [Nujoom AI](https://nujoom.ai/): 未找到描述
- [Tracing and Debugging - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/understanding/tracing_and_debugging/tracing_and_debugging.html#basic-logging): 未找到描述
- [来自 TheHeroShep (@TheHeroShep) 的推文](https://x.com/TheHeroShep/status/1767652590127661357?s=20): ComfyUI 的 LLM Node Pack 1。很高兴分享 @getsalt_ai 的强大节点集，感谢 @WAS_STUFF 让在 ComfyUI 中使用 LLM 和 @llama_index 变得更简单 ✨ Prompt 增强节点...
- [Azure AI Search - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html): 未找到描述
- [GitHub - get-salt-AI/SaltAI](https://github.com/get-salt-AI/SaltAI): 通过在 GitHub 上创建账号来为 get-salt-AI/SaltAI 的开发做出贡献。
- [llama_index/docs/examples/query_engine/SQLAutoVectorQueryEngine.ipynb at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/docs/examples/query_engine/SQLAutoVectorQueryEngine.ipynb): LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [llama_index/docs/examples/query_engine/SQLJoinQueryEngine.ipynb at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/docs/examples/query_engine/SQLJoinQueryEngine.ipynb): LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [Chat LlamaIndex](https://chat.llamaindex.ai/): 未找到描述
- [🚀 RAG/LLM Evaluators - DeepEval - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/evaluation/Deepeval.html): 未找到描述
- [[Beta] Text-to-SQL with PGVector - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/query_engine/pgvector_sql_query_engine.html): 未找到描述
- [极速 AI 电话冷启动 Agent - 基于 Groq 构建](https://youtu.be/WCYf2Agml-s?si=6cZ83c2eOdF_A2hR): Groq LPU 到底是什么？我将带你通过一个真实示例，利用 Groq 的速度构建一个实时的 AI 电话冷启动 Agent 🔗 链接 - 在 Twitter 上关注我...
- [Ensemble Query Engine Guide - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/query_engine/ensemble_query_engine.html): 未找到描述
- [Router Query Engine - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine.html): 未找到描述
- [ReAct Agent with Query Engine (RAG) Tools - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine.html): 未找到描述

---

**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 messages): 

shure9200: 我正在建立一个包含近期 LLM 论文的庞大数据库
https://shure-dev.github.io/
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1217087319010447390)** (69 messages🔥🔥): 

- **Axolotl 中的 LLaVA 支持查询**：一名成员询问了 Axolotl 对 **LLaVA 的支持**情况，得到的确认是 **LLaVA 1.6 有所改动**，可能与当前发布版本的 Axolotl 不兼容。
- **Cohere 发布 Command-R 模型**：Cohere For AI 发布了一个名为 **Command-R 的开源 350 亿参数模型**，针对各种用例进行了优化。该模型已在 [Huggingface](https://huggingface.co/CohereForAI/c4ai-command-r) 上架。
- **大语言模型数据集编辑工具**：成员们讨论了手动编辑大语言模型数据集的工具，提到了用于查看的 **Lilac**，并建议使用 **Argilla 和 Langsmith** 处理可能的编辑任务。
- **Meta 的 AI 基础设施扩展**：分享了来自 Soumith Chintala 的一条关于 **Meta 宣布**扩大基础设施建设的 Twitter 帖子，其中包括 **350,000 块 NVIDIA H100 GPU**。
- **具有长记忆能力的开源聊天机器人框架**：对话转向寻找适用于长记忆聊天机器人的开源模型和框架，讨论了上下文长度的重要性、软件的开放性以及在避免训练情况下的处理方法。提到了 **Mistral 和 Mixtral**，以及微调此类模型的复杂性。

**提到的链接**：

- [CohereForAI/c4ai-command-r-v01 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01)：未找到描述
- [Login | Cohere](https://dashboard.cohere.com/playground/chat)：Cohere 通过一个易于使用的 API 提供对高级大语言模型和 NLP 工具的访问。免费开始使用。
- [Building Meta’s GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：标志着对 Meta AI 未来的重大投资，我们宣布建立两个 24k GPU 集群。我们正在分享硬件、网络、存储、设计、性能和软件方面的细节，这些细节帮助我们……
- [Starwars Anakin GIF - Starwars Anakin Skywalker - Discover &amp; Share GIFs](https://tenor.com/view/starwars-anakin-skywalker-star-trek-wars-gif-19720485)：点击查看 GIF

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1217085956532998256)** (15 messages🔥): 

- **量化模型对 DoRA 的支持已合并**：在 *bitsandbytes* 的支持下，针对 **4bit 和 8bit 量化模型**的 **DoRA** 实现已合并。注意事项包括 DoRA 仅限于线性层，且与 LoRa 相比开销较高。[在此查看合并详情](https://github.com/huggingface/peft/pull/1518/files/3f35dd59bc937ec39d4a0f9dd5a5365209741f75..fd63e3c831e4a1250580799d9c9d107293ee2ffd)。
  
- **Fuyou 框架为低端 GPU 运行巨型模型带来希望**：据报道，名为 *Fuyou* 的新训练框架允许在**标准消费级 GPU 上微调高达 175B 参数的模型**，为预算有限的用户提供了 ZeRO-Infinity 的高效替代方案。[_akhaliq 的推文](https://x.com/_akhaliq/status/1767393991727657262?s=20)分享了这一突破，并在 **RTX 4090 GPU** 上实现了令人期待的 156 TFLOPS。

- **DeepSpeed 的新 API 可能使 axolotl 受益**：**DeepSpeed** 增加的一个 API 允许在设置 ZeRO3 钩子时将模块设置为叶节点，理论上这有助于 MoE 模型，该功能可能对 axolotl 的开发很有用。[在 GitHub 上探索该 API](https://github.com/microsoft/DeepSpeed/pull/4966)。

- **Mixtral 训练担忧得到缓解**：一位贡献者担心可能存在影响使用 ZeRO3 进行 Mixtral 模型训练的 Bug，通过一个已链接的 PR 得到了缓解，该 PR 似乎解决了相关问题，使他们能够继续训练。[在此查看提交记录](https://github.com/OpenAccess-AI-Collective/axolotl/commit/54d2ac155b46c7c1e1f69309a571acff01903b93#diff-65b4693504c4e8ffac76c7f2c90913faee381f802cf64e7f49c995a2134ed3b3R656)。

- **Axolotl 加载功能的疑问已解决**：针对 Axolotl 代码库中关于模型加载功能的特定方面寻求并给予了澄清，强调了 AutoModel 在受控时如何促进 **PEFT 模型**的加载。[相关代码可在此查看](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808)。

**提到的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1767393991727657262?s=20): 添加 NVMe SSD 以在单 GPU 上启用并加速 100B 模型微调。大语言模型（LLM）的最新进展为世界带来了巨大价值，其卓越的能力...
- [P2P PCIe 和 GPUDirect Storage 入门](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4): 这是对 NVIDIA GPUDirect Storage (GDS) 的五分钟快速回顾，包括它的定义、功能、基于的技术以及最适用的场景...
- [OpenAccess-AI-Collective/axolotl 中的 axolotl/src/axolotl/utils/models.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808): 欢迎提出 axolotl 问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [QDoRA: 通过 BenjaminBossan 支持带有 BnB 量化的 DoRA · Pull Request #1518 · huggingface/peft](https://github.com/huggingface/peft/pull/1518/files/3f35dd59bc937ec39d4a0f9dd5a5365209741f75..fd63e3c831e4a1250580799d9c9d107293ee2ffd): 增加了对使用 bitsandbytes 的 4bit 和 8bit 量化模型的 DoRA 支持。合并（Merging）也可以工作，但带有量化权重的常规注意事项（结果并非 100% 相同），但它并不更差...
- [通过 tohtana 添加 API 以在递归设置 Z3 钩子时将模块设置为叶节点 · Pull Request #4966 · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/pull/4966#issuecomment-1989): ZeRO3 不适用于 MoE 模型，因为执行模块的顺序在每次前向/反向传播（forward/backward pass）时都可能改变 (#4094, #4808)。此 PR 添加了一个 API，用于停止为参数拆分模块...
- [GitHub - enfiskutensykkel/ssd-gpu-dma: 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用程序](https://github.com/enfiskutensykkel/ssd-gpu-dma): 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用程序 - enfiskutensykkel/ssd-gpu-dma
- [GTC Silicon Valley-2019: 使用 NVMe 的高效分布式存储 I/O](https://developer.nvidia.com/gtc/2019/video/S9563): 未找到描述
- [SNU-ARC/flashneuron 中的 flashneuron/README.md](https://github.com/SNU-ARC/flashneuron/blob/master/README.md): 通过在 GitHub 上创建账号为 SNU-ARC/flashneuron 的开发做出贡献。
- [通过 tohtana 添加 API 以在递归设置 Z3 钩子时将模块设置为叶节点 · Pull Request #4966 · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/pull/4966#issuecomment-1989671378): ZeRO3 不适用于 MoE 模型，因为执行模块的顺序在每次前向/反向传播（forward/backward pass）时都可能改变 (#4094, #4808)。此 PR 添加了一个 API，用于停止为参数拆分模块...
- [Mixtral 修复 20240124 (#1192) [skip ci] · OpenAccess-AI-Collective/axolotl@54d2ac1](https://github.com/OpenAccess-AI-Collective/axolotl/commit/54d2ac155b46c7c1e1f69309a571acff01903b93#diff-65b4693504c4e8ffac76c7f2c90913faee381f802cf64e7f49c995a2134ed3b3R656): * mixtral nccl 修复
 
 * 确保针对 z3 进行补丁（patch）

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1217237497369722920)** (3 messages): 

- **使用示例自定义标志进行调试**: 一位成员建议使用 `--debug=True --debug_num_examples=10 --debug_text_only=True` 等标志，以确保训练过程中的一切符合预期。分享提到 "completion" 默认按换行符拆分，这导致人们更倾向于使用 `input_output` 而非 `completion`。
  
- **调整样本打包效率 (Sample Packing Efficiency)**: 用户发现，随着数据量变大，**sample_packing_efficiency** 下降到 0.85 左右，可能导致训练步骤增加约 10%。将 `sample_packing_eff_est` 设置为 1.0 是尝试过的一个临时解决方案，尽管他们仍在测试结果。

- **警告不要手动设置 Eff_est**: 另一位成员警告说，不应手动设置 `sample_packing_eff_est`，强调这是一个内部值，不打算让用户修改。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1217164640304500737)** (5 messages):

- **Mistral Medium 胜过 Mixtral**：一位成员报告称 **Mistral Medium** 提供的输出和响应优于 **Mixtral**，并认为 Mistral Medium 可能是 Mixtral 的一个闭源改进版本。
- **对 Mistral Medium 性能的赞赏**：该成员强调了 **Mistral Medium** 在没有明确要求的情况下生成相关引用的能力，展示了其高效性。
- **Mistral Large 的权衡**：尽管 **Mistral Large** 被认为在质量上表现最好，但其超时速度比 Medium 版本更快。
- **Mixtral 在冗余度和指令遵循方面表现不足**：该成员指出，与 Mixtral 相比，**Mistral Medium** 的输出更简洁，且能更准确地遵循指令。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1217036926423400538)** (82 messages🔥🔥): 

- **Qdrant 集合的增量更新**：一位用户询问如何在 LangChain 中增量更新 Qdrant 集合的 embeddings，但在提供的对话上下文中未获得解决方案或回复。
- **在 LangChain 中切换聊天模式**：一位用户在从 **oobabooga's text-generation-webui** 切换后，寻求在 LangChain 中将新 LLM 的模式从 `chat-instruct` 更改为 `chat` 的帮助。他们提供了详细的代码片段，并强调其切换到了 *llama-2 的微调版本*。
- **LangChain 中的自定义 LLM**：讨论了在 LangChain 中使用自定义 LLM 服务的问题，并根据库文档 ([Custom LLM](https://python.langchain.com/docs/modules/model_io/llms/custom_llm)) 提供了创建自定义 LLM 对象的指导。他们被建议可能使用 **FastAPI/Langserve** 来集成部署在 *localhost* 服务器上的 LLM。
- **处理过时的文档**：用户对 LangChain 文档过时且不一致的问题表示担忧，这使得跟踪包的导入及其用法变得困难。他们讨论了个人经历以及 **LangChain 保持文档更新** 的必要性。
- **在 LangChain 中创建结构化输出和工具**：一位用户分享了向 LangChain 社区开放 Python SDK 的兴趣，但在对话中未获得回复。此外，关于利用结构化输出 (`with_structured_output`) 和 *langsmith* 进行开发的讨论中，由于 LangChain 功能和版本的突然变化而充满了不确定性。

（注：此摘要仅基于提供的对话片段。未包含任何未记录的功能、讨论或外部资源。）

**提到的链接**：

- [SparkLLM Text Embeddings | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/text_embedding/sparkllm)：官方网站 //www.xfyun.cn/doc/spark/Embeddingnewapi.html
- [Run LLMs locally | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/local_llms)：使用案例
- [langsmith-cookbook/testing-examples/tool-selection/tool-selection.ipynb at main · langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/tool-selection/tool-selection.ipynb)：通过在 GitHub 上创建账号为 langchain-ai/langsmith-cookbook 的开发做出贡献。
- [LangChain](https://www.youtube.com/playlist?list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5)：未找到描述
- [[beta] Structured Output | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/structured_output)：让 LLM 返回结构化输出通常至关重要。
- [GitHub - antonis19/autobrowse: AutoBrowse is an autonomous AI agent that can perform web browsing tasks.](https://github.com/antonis19/autobrowse/tree/main)：AutoBrowse 是一个可以执行网页浏览任务的自主 AI Agent。 - antonis19/autobrowse
- [Discord Bot | MEE6](https://mee6.xyz/en).)：通过等级、管理、Twitch、Youtube 和 Reddit 通知来管理你的 Discord 服务器。
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve)：LangServe 🦜️🏓。通过在 GitHub 上创建账号为 langchain-ai/langserve 的开发做出贡献。
- [GitHub - ggerganov/whisper.cpp: Port of OpenAI's Whisper model in C/C++](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file)：OpenAI Whisper 模型的 C/C++ 移植版本。通过在 GitHub 上创建账号为 ggerganov/whisper.cpp 的开发做出贡献。
- [Wordware - Try all the models for a single question](https://app.wordware.ai/r/fc405cb4-877b-44b7-aed8-b883e48eced3)：该提示词将一个问题通过 Gemini, GPT-4 Turbo, Claude 2, Mistral Medium, Mixtral 和 Openchat 运行。然后使用 GPT-4 Turbo 评估哪个模型给出了最佳答案。

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1217052257258962984)** (1 messages):

- **切换模式以获得更好的 Chatbot 性能**：一位成员分享了在 **langchain 库** 的 Chatbot 应用中，使用 `chat` 模式比 `chat-instruct` 模式性能更好的经验。他们寻求帮助，希望将目前在 **oobabooga 的 `text-generation-webui`** 中通过 LlamaCpp 实现的 `chat-instruct` 设置切换为 `chat` 模式。
- **寻求 Langchain 编码帮助**：该成员提供了一段 Python 代码片段，其中使用了 **LlamaCpp 模型**路径和 **RedisChatMessageHistory** 进行对话生成。他们请求关于如何在 **langchain 应用** 中实现更理想的 `chat` 模式的指导。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1217398683377602611)** (3 messages): 

- **ReAct Agent 发布**：分享了一个名为 **ReAct Agent** 的新语言模型 Agent，其灵感源自一篇关于在语言模型中协同推理（Reasoning）与行动（Acting）的论文。它承诺提供推理引擎和多样化的技能，并能回答各种问题。该 Agent 是使用一种新工具快速创建的，欢迎大家提供反馈。[阅读激发此创作的论文](https://arxiv.org/abs/2210.03629)。

- **LangChain Chatbot 现已开源**：宣布 **LangChain Chatbot** 已开源，并展示了如何使用 RAG 进行问答查询。它具有极其简单的设置、支持服务器、交互式 Streamlit UI 以及 Python FastAPI Server。查看 [GitHub 仓库](https://github.com/Haste171/langchain-chatbot)。

- **MindGuide：用于心理健康支持的 AI**：分享了一篇详细介绍 **MindGuide** 的文章，这是一个旨在利用 LangChain 结合大语言模型（LLM）彻底改变心理健康护理的 Chatbot。文章强调了心理健康干预的必要性，并讨论了 MindGuide 增强心理健康挑战支持的功能。[阅读这篇革命性的文章](https://arxiv.org/abs/2403.05568)。

**提到的链接**：

- [Revolutionizing Mental Health Care through LangChain: A Journey with a Large Language Model](https://arxiv.org/abs/2403.05568)：现代社会心理健康挑战日益增多，解决心理障碍（尤其是焦虑、抑郁和自杀念头）的紧迫性凸显了对……的需求。
- [GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.](https://github.com/Haste171/langchain-chatbot)：用于以对话格式分析/提取数据信息的 AI Chatbot。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)：虽然大语言模型（LLM）在语言理解和交互决策任务中展示了令人印象深刻的能力，但它们的推理能力（例如思维链……）
- [Wordware - ReAct API Agent 🧠](https://app.wordware.ai/r/0b8b7771-09dc-4a19-87d4-89e43b5cc153)：研究如何使用 API。

  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1217104654928707664)** (3 messages): 

- **探索用于 AI 的 Groq 硬件**：分享了一个名为“使用 Groq 构建极速 AI 拨冷电话 Agent”的 [YouTube 视频](https://youtu.be/WCYf2Agml-s)，展示了如何利用 Groq 的硬件构建实时 AI 拨冷电话 Agent。
- **使用 Command-R 进行检索增强**：一位成员推荐了一个[视频](https://www.youtube.com/watch?v=rnP87DzGeDw)，讨论了 Command-R 的功能，该模型针对检索增强生成（RAG）等长上下文任务进行了优化。
- **Langchaingo 提示词模板指南**：分享了一个关于使用 Langchaingo 创建提示词模板的 [YouTube 教程](https://youtu.be/dcBEtgh4078)，演示了该过程及其在 Telegram 群组中的应用。

**提到的链接**：

- [INSANELY Fast AI Cold Call Agent- built w/ Groq](https://youtu.be/WCYf2Agml-s?si=6cZ83c2eOdF_A2hR)：Groq LPU 到底是什么？我将带你通过一个真实示例，利用 Groq 的速度构建一个实时 AI 拨冷电话 Agent。🔗 链接 - 在 Twitter 上关注我……
- [Lets RAG with Command-R](https://www.youtube.com/watch?v=rnP87DzGeDw)：Command-R 是一款生成模型，针对长上下文任务（如检索增强生成 RAG）以及使用外部 API 和工具进行了优化。它的设计……
- [Create Prompt Template With Langchaingo](https://youtu.be/dcBEtgh4078)：在本视频中，我将介绍如何创建提示词模板以及如何将其与 chains 结合使用。Telegram 群组：https://t.me/langchaingo/1 #golang #langchain

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1217560477933240420)** (2 messages):

- **数据库更新短暂影响 OpenRouter**：由于数据库更新时间超出预期，OpenRouter 经历了 **activity row** 可用性的临时问题。该波动持续了约三分钟，受影响的 completions 均未计费。

- **快速且经济实惠：Claude 3 Haiku 登场**：**Claude 3 Haiku** 已在 OpenRouter 上线，拥有约 **120 tokens/秒** 的惊人速度，且价格仅为 **每 1 美元 4M prompt tokens**。它提供审核版和自我审核（self-moderated）版，非常适合需要快速、准确响应的用户。[在此体验 Claude 3 Haiku](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)。

- **Claude 3 Haiku：快速、实惠、多模态**：Anthropic 最新的 Claude 3 模型 **Claude 3 Haiku** 因其近乎瞬时的响应能力而备受推崇，并提供了实惠的输入和输出 token 定价细节。该模型采用自我审核机制以确保快速且针对性的性能，由于仍处于 beta 阶段，未来可能会有变动。[发布公告与基准测试](https://www.anthropic.com/news/claude-3-haiku)。

**提到的链接**：

[Anthropic: Claude 3 Haiku (self-moderated) by anthropic | OpenRouter](https://openrouter.ai/models/anthropic/claude-3-haiku:beta)：这是与 Anthropic 合作提供的 [Claude 3 Haiku](/models/anthropic/claude-3-haiku) 低延迟版本，采用自我审核机制：响应审核在模型端进行...

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1217470944973426799)** (2 条消息): 

- **Olympia.chat 利用 OpenRouter**：[Olympia.chat](https://olympia.chat) 的联合创始人 Obie Fernandez 介绍了他们的 ChatGPT 克隆平台，该平台自 2023 年 5 月以来主要服务于独立创业者（solopreneurs）和小企业主。他们已开始使用 OpenRouter 作为各种组件的 LLM 来源，并宣布即将推出一个用于 OpenRouter 的开源 Ruby 库。

- **朋友开发的 Messenger 聊天机器人准备好测试**：一位用户分享了其朋友为 Messenger 应用创建的聊天机器人的信息。通过私信邀请提供测试机会。

**提到的链接**：

[Olympia | Better Than ChatGPT](https://olympia.chat)：通过价格合理的 AI 顾问助力您的业务增长，这些顾问是业务战略、内容开发、营销、编程、法律战略等领域的专家。

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1217059990385332256)** (54 条消息🔥): 

- **了解 OpenRouter 和 Groq 的访问权限**：成员们讨论了在 **Groq 的 Mixtral 模型** 免费期间的使用情况。会议澄清了通过 OpenRouter 访问并不属于 Groq 提供的免费访问范围，且个人 Groq tokens 无法通过 OpenRouter 传递。

- **探讨 Mixtral 的限制与错误**：当用户遇到 "Request too big" 错误时，引发了关于 **Mistral 8x7B** 和 Nitro 模型限制的讨论。上下文限制（context limit）被确认为 32k，但也考虑了导致错误的特定原因，例如潜在的重复循环。

- **请支持 Claude 3 Haiku！**：一名成员请求添加对新发布的 **Claude 3 Haiku** AI 模型的支持。

- **AI 模型定价与速度**：通过将 **Claude 3 Haiku** 与 **Opus** 进行对比，成员们强调了 Haiku 的速度和成本效益，指出 Haiku 每百万 tokens 1.25 美元与 Opus 每百万 tokens 75 美元之间存在显著的价格差异。

- **对 GPT-4.5 的高度期待**：围绕 **GPT-4.5** 传闻的互动显示出成员们极高的期待和兴奋，无论消息是否得到证实。

**提到的链接**：

- [Blog](https://www.cognition-labs.com/blog)：未找到描述
- [Claude 3 Haiku: our fastest model yet](https://www.anthropic.com/news/claude-3-haiku)：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。
- [OpenAI Status](https://status.openai.com/)：未找到描述

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1217143328475320370)** (9 条消息🔥): 

_

- **GPT-4.5 博客文章被发现**：有消息称 GPT-4.5 的博客文章已被 Bing 收录，并出现在搜索结果中。
- **404 谜团**：然而，尝试访问该 GPT-4.5 博客文章链接时，会跳转到 404 错误页面。
- **发布前的兴奋**：现场气氛热烈，因为这一发现被认为是 GPT-4.5 可能即将推出的确认。
- **搜索引擎踪迹**：该博客文章不仅出现在 Bing 上，还出现在依赖 Bing 结果的其他搜索引擎（如 Kagi 和 DuckDuckGo）上，但未出现在 Google 上。
- **用于澄清的推文**：分享了一个 [Twitter 链接](https://twitter.com/AndrewCurran_/status/1767916848987914487)，可能有助于澄清 GPT-4.5 博客文章的情况。
  

---


**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1217542990441087056)** (2 messages): 

- **GPT-4 在 LeetCode 上占据主导地位**：一位成员提到 **GPT-4** 在 LeetCode 上的性能仍然稳居榜首。证据来自他们引用的论文，可在此处访问：[here](https://livecodebench.github.io/pdfs/paper.pdf)。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1217067234841591858)** (42 messages🔥): 

- **Claude 3 展示了令人期待的结果**：一位成员分享了他们对 **Claude 3** 在文献研究摘要方面比 Claude 2 有显著进步的兴奋之情，并强调了通过 [Clautero 项目](https://github.com/Xeophon/Clautero)展示的 LLM 潜力。
- **Meta 巨额 AI 基础设施投资**：Meta 宣布创建两个 24k GPU 集群，并详细介绍了他们通过 [Grand Teton](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/)、[PyTorch](https://pytorch.org/) 等资源对开源和开放计算的承诺，目标是到 2024 年积累 350,000 块 NVIDIA H100 GPU，正如其[官方博客](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)所述。
- **关于 Google 模型发布策略的辩论**：讨论了 Google 对其 AI 模型的策略，辩论其应该是广告支持还是订阅制，并与 Bing/Copilot 和 Youtube 的 Premium 服务进行了比较，同时对针对性广告的隐私和数据使用表示担忧。
- **AI 输出中可能出现的广告**：成员们讨论了在 AI 输出中加入广告的可能性和影响，对 ChatGPT 如果遵循这种模式未来的可信度提出了质疑。
- **AnthropicAI 发布 Claude 3 Haiku**：AnthropicAI 发布了 Claude 3 Haiku，因其速度快、价格实惠且比 GPT-3.5 等前代产品更强大而受到赞誉，[公司推文](https://x.com/anthropicai/status/1768018310615151002?s=46)宣布其已在 API 和 claude.ai 上向 Pro 订阅者开放。

**提到的链接**：

- [Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1768018310615151002?s=46)：今天我们发布了 Claude 3 Haiku，它是同类智能级别中速度最快、最实惠的模型。Haiku 现在已在 API 和 http://claude.ai 上向 Claude Pro 订阅者开放。
- [构建 Meta 的 GenAI 基础设施](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：作为对 Meta AI 未来的一项重大投资，我们宣布推出两个 24k GPU 集群。我们正在分享有关硬件、网络、存储、设计、性能和软件的详细信息，这些细节帮助我们……
- [lmsys.org (@lmsysorg) 的推文](https://fxtwitter.com/lmsysorg/status/1767997086954573938)：[Arena 更新] 我们的社区为 Claude-3 Opus 和 Sonnet 多投了 20,000 票，对新的 Claude 模型系列表现出了极大的热情！Claude-3-Opus 现在与 GPT-4-Tu... 并列第一*。

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1217054985838133258)** (13 messages🔥): 

- **Meta 的大规模 GPU 集群**：Meta 宣布推出两个 **24k GPU 集群**以支持 AI 工作负载，依赖于 Grand Teton、OpenRack 和 PyTorch 框架。到 2024 年底，他们的目标是拥有 350,000 块 NVIDIA H100 GPU，其算力相当于 600,000 块 H100，正如其[基础设施公告](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)中所述。

- **从应用到原子——Microsoft 进军核能技术**：Microsoft 聘请 Archie Manoharan 担任核能技术总监，旨在为数据中心电力开发原子反应堆，这一举动在 [Times of India](https://timesofindia.indiatimes.com/gadgets-news/apps-to-atoms-microsoft-hires-nuclear-expert-to-fuel-its-data-centres/articleshow/107151840.cms) 和 [The Register](https://www.theregister.com/2024/01/23/microsoft_nuclear_hires/) 的文章中得到了详细阐述。

- **RTX 4090 AI 训练潜力**：一篇引用的论文在 RTX 4090 上实现了 **156 TFLOPS** 的 AI 训练性能，远超 ZeRO-Infinity 的 45 TFLOPS，指向了 [GPU performance](https://arxiv.org/abs/2403.06504) 方面的重大创新。

- **NVMe SSD 的 CUDA 支持**：一个 GitHub 仓库 ([ssd-gpu-dma](https://github.com/enfiskutensykkel/ssd-gpu-dma)) 和一场 [NVIDIA GTC 演讲](https://developer.nvidia.com/gtc/2019/video/S9563) 展示了在存储应用中结合 CUDA 使用 NVMe 驱动器的进展，并由直接内存访问（DMA）和 GPU 集成提供支持。

- **NVIDIA 的 GPUDirect Storage API 承诺提供高效数据路径**：NVIDIA 的 GPUDirect Storage 提供了一个用于 GPU 内存与存储之间高效 DMA 传输的 API，正如 [API Reference Guide](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) 中详述的那样，这有可能增强数据密集型工作负载的性能。

**提到的链接**：

- [Building Meta’s GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)：标志着对 Meta AI 未来的重大投资，我们宣布推出两个 24k GPU 集群。我们正在分享有关硬件、网络、存储、设计、性能和软件的细节，这些细节帮助我们实现了……
- [flashneuron/README.md at master · SNU-ARC/flashneuron](https://github.com/SNU-ARC/flashneuron/blob/master/README.md)：通过在 GitHub 上创建账户来为 SNU-ARC/flashneuron 的开发做出贡献。
- [Apps to atoms: Microsoft hires nuclear expert to fuel its data centres | - Times of India](https://timesofindia.indiatimes.com/gadgets-news/apps-to-atoms-microsoft-hires-nuclear-expert-to-fuel-its-data-centres/articleshow/107151840.cms)：微软聘请了一位核能专家，通过开发小型原子反应堆作为化石燃料的替代品，为其数据中心提供动力。
- [Microsoft hires leaders for nuclear datacenter program](https://www.theregister.com/2024/01/23/microsoft_nuclear_hires/)：行业资深人士专注于小型模块化反应堆的开发。
- [cuFile API Reference Guide - NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)：未找到描述
- [GitHub - enfiskutensykkel/ssd-gpu-dma: Build userspace NVMe drivers and storage applications with CUDA support](https://github.com/enfiskutensykkel/ssd-gpu-dma)：构建具有 CUDA 支持的用户空间 NVMe 驱动程序和存储应用程序 - enfiskutensykkel/ssd-gpu-dma
- [GTC Silicon Valley-2019: Efficient Distributed Storage I/O using NVMe](https://developer.nvidia.com/gtc/2019/video/S9563)：未找到描述
- [P2P PCIe and GPUDirect Storage Primer](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4)：这是一个五分钟的快速浏览，介绍 NVIDIA 的 GPUDirect Storage (GDS) 是什么、它的作用、它基于哪些技术以及它在何处最适用……

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1217247732666994770)** (6 条消息): 

- **Nsight Visual Studio Code Edition 热度**：一位成员强调了使用 [NVIDIA Nsight™ Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition) 的好处，这是一款用于在 GPU 上进行 CUDA® 开发的工具，提供了智能 CUDA 代码自动补全等功能，并为包括 **Linux 和 QNX 目标系统**在内的各种平台改进了整体开发体验。分享了一个直接下载该工具的链接：[立即下载](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)。

- **Nsight 爱好者分享对工具的热爱**：一位成员表达了他们对 NVIDIA Nsight 工具的喜爱，表示无法想象没有 Nsight Systems 和 Nsight Compute 的开发过程。他们认为没有理由不将这些工具作为工作流的一部分。

- **关于 Nsight Systems 实用性的辩论**：一位成员发表意见认为 Nsight Systems 没用，同时赞扬了 Nsight Compute。另一位成员通过描述一个涉及多个流和跨多个 GPU 的 CPU 线程的实时应用场景进行了反驳，指出了 Nsight Systems 在识别性能瓶颈方面的重要性。

- **寻求 CUDA 帮助**：一位用户就他们在另一个频道发布的“CUDA 菜鸟问题”寻求帮助，表示即使咨询了 Google 和 GPT 后仍有困难。他们不确定自己是否选择了合适的地方进行咨询。

**提到的链接**：

[Nsight Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition)：集成到 Microsoft Visual Studio Code 中的 NVIDIA 平台 CUDA 开发工具

---

**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1217073995992662056)** (2 条消息):

- **关于 Modular 功能的澄清**：一位参与者意识到他们误解了之前的消息，并确认他们最初的想法确实是 **Modular** 正在做的事情。
- **征求 Torchao 的反馈**：邀请大家对 [PyTorch Labs 的 torchao GitHub issue](https://github.com/pytorch-labs/ao/issues/47) 提供反馈，旨在促进新的 quantization 算法和 dtypes 的合并。**gpt-fast 和 sam-fast kernels** 背后的团队为对 **real world cuda mode project** 感兴趣的有志 kernel 编写者提供指导。

**提到的链接**：

[[RFC] Plans for torchao · Issue #47 · pytorch-labs/ao](https://github.com/pytorch-labs/ao/issues/47)：摘要：去年，我们发布了 pytorch-labs/torchao，旨在利用原生 PyTorch 技术加速生成式 AI 模型。Torchao 增加了对在 GPU 上运行 quantization 的支持，包括...

---

**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1217477687589736548)** (1 条消息): 

- **GetWorldClass 招聘 CUDA 专家**：*Christo* 宣布他们正在为学习应用 [getworldclass.app](https://getworldclass.app) 寻找 **CUDA 专家** 进行咨询。感兴趣的人士请私信了解详情。

---

**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1217105788909391984)** (4 条消息): 

- **确保可执行文件正确启动 Kernels**：一位成员询问可执行文件是否正确启动了 kernels，暗示 kernel 执行存在问题。
- **验证 CUDA 代码执行**：一位成员确认他们的 CUDA 代码运行成功，排除了代码执行本身的问题。
- **Nsight Compute GPU 兼容性疑虑**：有人询问 Nsight Compute 是否支持某个 GPU，表明对 GPU 与该分析工具的兼容性存在不确定性。
- **Ubuntu CUDA Toolkit 故障**：一位使用 Ubuntu 23.10 的成员遇到了 `compute-sanitizer` 报错，提示找不到 `libsanitizer-collection.so`，尽管该文件存在于机器上。他们提到所使用的 `compute-sanitizer` 版本是 **2022.4.1**，且该问题在重新安装操作系统后依然存在。

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1217561639038226543)** (2 条消息): 

- **解读 CUDA Cores 和线程执行**：一位成员询问了书中第 4.4 节提到的架构，质疑在 **SIMD 模型**下，每个 core 是否负责执行 **4 个线程**。该疑问基于如图 4.8 所示的组织结构，即 8 个 cores 组成一个共享指令获取/分发单元的处理块。

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1217050003118358638)** (13 条消息🔥): 

- **Loss 拒不下降**：一位成员对在小数据集上训练 100 个 epochs 后 loss 仍不低于 3.6 表示沮丧，这可能暗示 training kernel 或超参数存在问题。另一位参与者建议检查目前正在自动计算的 learning rate。
- **Axolotl 需要 Padding 参数**：提到训练软件 Axolotl 即使在全新克隆的仓库中也需要设置 `pad_to_sequence_len: true` 才能运行。这作为一条提醒笔记被分享。
- **Ring Attention 面临训练挑战**：该成员分享了一份 [W&B 报告](https://api.wandb.ai/links/iron-bound/v6mxxcj2)，对比了原生 Axolotl 代码和 ring-attention 变体的性能，指出 loss 并没有像预期的那样趋向于零。
- **详细对比发布**：澄清了性能报告的基准是未经代码修改的 Axolotl 克隆版本，作为评估 ring-attention 修改代码的参考。
- **仓库补丁进行中**：分享了 Axolotl 仓库的一个打过补丁的分支链接，其中可能包含与 ring attention 模型讨论相关的更新或修复 ([Axolotl Git Repository - ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching))。

**提到的链接**：

- [Ring-attn vs stock](https://api.wandb.ai/links/iron-bound/v6mxxcj2)：在相同系统和小数据集上运行了约 100 个 epochs。
- [iron-bound](https://wandb.ai/iron-bound/axolotl/runs/t6dz9ub1?workspace=user-iron-bound)：Weights & Biases，机器学习开发者工具。
- [GitHub - cuda-mode/axolotl at ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)：欢迎就 Axolotl 提问。通过创建账号为 cuda-mode/axolotl 的开发做出贡献。

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1217150420582137977)** (5 messages): 

- **AI 工程师 Devin 初探**：来自 [Cognition Labs](https://www.cognition-labs.com/blog) 的博客文章介绍了 **Devin**，据称它是全球首位全自主 AI 软件工程师，具备执行复杂工程任务的能力，配备了标准开发者工具，并在 **SWE-bench** 编码基准测试中树立了新标杆。

- **Devin 的真实世界测试即将到来**：有人向 Cognition Labs 申请了 **Devin** 的早期访问权限，并承诺分享对这位 AI 软件工程师的“未经筛选的观点”。他们引用了 [@itsandrewgao](https://x.com/itsandrewgao/status/1767576901088919897?s=20) 的推文，预计该推文串将展示 **Devin** 的能力，包括其在 SWE-Bench 编码基准测试中处理真实世界软件工程任务的出色表现。

- **推文消失引发用户不满**：一位成员分享了对 Twitter 用户体验的沮丧，特别是推文从时间线中消失的问题，并引用了 [@DanielleFong](https://fxtwitter.com/daniellefong/status/1767601118706897295?s=46&t=weiD0pEGM4LhsFr0gEglTA) 的抱怨，哀叹失去了一条有趣的推文。

- **GPT-4 挑战经典游戏**：[arxiv.org](https://arxiv.org/abs/2403.05468) 上分享的一项研究表明，**GPT-4** 可以相当成功地运行 1993 年的第一人称射击游戏 Doom，展示了该模型基于游戏截图生成的文本描述所具备的推理和规划能力。

**提到的链接**：

- [来自 Danielle Fong 💁🏻‍♀️🏴‍☠️💥♻️ (@DanielleFong) 的推文](https://fxtwitter.com/daniellefong/status/1767601118706897295?s=46&t=weiD0pEGM4LhsFr0gEglTA)：再见了，那条从时间线上刷新掉的有趣推文。我再也见不到你了。
- [Blog](https://www.cognition-labs.com/blog)：未找到描述
- [GPT-4 能运行 DOOM 吗？](https://arxiv.org/abs/2403.05468)：我们展示了 GPT-4 的推理和规划能力可以扩展到 1993 年的第一人称射击游戏 Doom。这个大语言模型 (LLM) 仅需少量指令即可运行并进行游戏...
- [来自 Andrew Kean Gao (@itsandrewgao) 的推文](https://x.com/itsandrewgao/status/1767576901088919897?s=20)：我从不相信录制的演示，所以我联系了 @cognition_labs 团队申请早期访问权限亲自尝试，并且拿到了！将在这里分享我对 #devin 的真实看法。🧵🧵 1/n ↘️ 引用...

  

---



**DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1217390277103190068)** (1 messages): 

- **助手的火星回答缺乏 Musk 的神韵**：*Die Antwort des Assistenten*（助手的回答）关于我们为什么要前往火星的回答内容详实且详细，涉及了各个方面，但未能完全捕捉到 Elon Musk 独特的风格和语气。尽管反映了 Musk 对火星探索的观点，但由于缺乏 Musk 特有的公开演讲风格，其创意评分被评为 [[7]]。
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1217033661455667240)** (15 messages🔥): 

- **RAG 实践与观点**：关于构建 **RAG** (Retrieval-Augmented Generation) 提示词的最佳方式展开了讨论。一位用户倾向于将完整的 RAG 提示词包含在用户消息中，而另一位用户则认为这取决于 SFT (Supervised Fine-Tuning)；系统提示词通常用于通用指令，改变系统提示词可能会影响通用行为的一致性。

- **新的 Transformer 内部探索工具**：分享了 Jan Leike 的一条推文链接，宣布发布了一款名为 **Transformer Debugger** 的内部工具，用于分析 Transformer 内部结构。它无需编写代码即可实现快速的模型探索，并将自动化可解释性与 sparse autoencoders 相结合。

- **Mixtral 7b 8 Expert 模型使用**：一位用户分享了他们尝试使用 `mixtral-7b-8expert` 模型的[实验性实现](https://huggingface.co/DiscoResearch/mixtral-7b-8expert)进行推理，并遇到了非英语输出生成的问题。另一位用户建议改用[官方的 `mistralai/Mixtral-8x7B-v0.1` 模型](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)。

- **实验性模型需要明确说明**：讨论强调，**实验性**模型 `DiscoResearch/mixtral-7b-8expert` 的状态未得到清晰传达，可能需要更明确的标签以避免用户混淆。

- **关于 DiscoLM 和德语数据的咨询**：一位用户询问模型 `DiscoResearch/DiscoLM-mixtral-8x7b-v2` 是否在德语数据集上进行了微调。该用户被告知该模型并未在大量的德语数据上进行训练，并被引导至用于 `DiscoResearch/DiscoLM-70b` 的数据集，该模型 [*"针对 65b tokens 的德语文本进行了额外的持续预训练"*](https://huggingface.co/DiscoResearch/DiscoLM-70b)。

- **MunichNLP 聚会邀请**：一位用户联系群组参加 MunichNLP 聚会，特别对关于 DiscoLM 的演讲感兴趣。一位与会者表示无法确认，但提到即将举行的活动——柏林的 AI Tinkerers，并提供了活动页面的链接，并注明仅剩 8 个席位。

**提到的链接**：

- [DiscoResearch/mixtral-7b-8expert · Hugging Face](https://huggingface.co/DiscoResearch/mixtral-7b-8expert)：未找到描述
- [mistralai/Mixtral-8x7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)：未找到描述
- [Jan Leike (@janleike) 的推文](https://x.com/janleike/status/1767347608065106387?s=46&t=1jtkL4JPu-DUOdo8JC668g)：今天我们发布了一个内部一直在使用的工具，用于分析 Transformer 内部机制——Transformer Debugger！它结合了自动化可解释性和稀疏自编码器（sparse autoencoders），并且...
- [DiscoResearch/DiscoLM-70b · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM-70b)：未找到描述
- [AI Tinkerers - Berlin](https://berlin.aitinkerers.org/)：未找到描述

  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1217400001525973034)** (1 条消息): 

- **创意写作基准测试上线**：一个新的 **创意写作基准测试原型（creative writing benchmark prototype）** 现已发布并已投入运行，尽管其区分度不是很高。排名大致合理，你可以在 [GitHub 上的 creative_writing 分支](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing) 查看。

**提到的链接**：

[GitHub - EQ-bench/EQ-Bench at creative_writing](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing)：大型语言模型情感智能基准测试 - GitHub - EQ-bench/EQ-Bench at creative_writing

  

---


**DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1217552936079720458)** (2 条消息): 

- **寻求最佳德语 Embedding 解决方案**：一位成员正在咨询适用于德语的 **最佳 Embedding 和 re-ranker**，特别是针对 **德语法律文本** 的应用。他们想知道法律语言的特殊性是否会影响技术选择。
- **寻找德语 Embedding 基准测试**：同一位成员询问是否存在针对德语 Embedding 模型的现有 **基准测试（benchmark）**。对话中未提及具体的基准测试。
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1217222777208049735)** (3 条消息): 

- **Mixtral 的新前沿**：一位成员正在探索将 **Mixtral** 用于非英语语言的第二阶段预训练（stage 2 pretraining），并指出在这个层面上进行尝试的专家非常稀少。
- **寻求使用本地模型复现 Demo**：一位成员询问如何使用本地模型复现 Demo 的输出，分享了他们当前的代码配置，并询问除了 `temperature`、`top_p` 和 `max_tokens` 之外，是否还有其他可能需要调整的设置。
- **澄清对话模型中的重复问题**：另一个问题是关于 **指令（command）是否应该随每条用户消息重复**，还是在系统内容（system's content）中出现一次就足够了，这表明在如何优化 AI 模型的对话 Prompt 结构方面存在不确定性。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1217028763087536188)** (13 条消息🔥):

- **gpt4turbo 的 Token 限制困扰**：一位成员表达了在使用 **gpt4turbo** 时达到 4096 Token 限制的挫败感。这个关于上下文长度的问题被提出，但没有给出任何解决方案或变通方法。
- **Bing 搜索的误导**：在搜索 "openai announces gpt-4.5 turbo" 时，一位成员分享了 Bing 搜索结果，显示没有找到该查询的相关结果，暗示 **GPT-4.5 Turbo** 可能还不存在。
- **GPT-4.5 Turbo 即将到来吗？**：一些成员推测 **GPT-4.5 Turbo** 即将发布，最初感到兴奋，但随后由于潜在的误导性搜索结果和引用而产生怀疑。
- **草稿占位符引发混乱**：据澄清，所谓提到的 **GPT-4.5** 似乎是 OpenAI 页面上意外发布的草稿占位符，内容日期追溯到 9 月，并提到训练数据截至 2024 年 7 月。
- **理解 "GPT-4.5 turbo" 出现的谜团**：进一步的澄清否定了现成的 **GPT-4.5 模型** 的说法，结论是那篇意外发布的博客文章是旧的，引用的模型在发布时并未准备好。

**提到的链接**：

[openai announces gpt-4.5 turbo - Bing](https://www.bing.com/search?q=openai+announces+gpt-4.5+turbo)：Bing 搜索引擎的智能搜索让您更轻松地快速找到所需内容并获得奖励。

  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/)** (1 条消息): 

ldj: 如果 OpenAI 在同一天抢了 Starship 的风头，Elon 可能会很生气 😭
  

---


**LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1217170102047342765)** (4 条消息): 

- **关于 Llama 版本周期的推测**：一位成员理论化认为 *Llama* 以六个月为一个周期运行，但计划可能已经改变。他们建议最近准备发布的 **Llama-3** 因质量原因被跳过，暗示内部命名为 "Llama-4" 的版本将于 7 月发布。

- **预测 Llama-3 的升级**：同一位成员预测，下一次迭代（可能命名为 **Llama-3**）可能会引入多项改进，包括 **Mixture of Experts**、**SSM variants**、**Attention mods**、**图像和视频的多模态**、**扩展的上下文长度**以及**高级推理能力**。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1217441897161953381)** (6 条消息): 

- **揭示 GenAI 漏洞的新论文**：一位成员分享了一篇名为 "ComPromptMized: Unleashing Zero-click Worms that Target GenAI-Powered Applications" 的[有趣论文](https://sites.google.com/view/compromptmized)，作者是来自以色列理工学院的 Stav Cohen。该论文展示了计算机蠕虫如何针对并利用 GenAI 驱动的应用程序（包括电子邮件助手），跨越不同的 GenAI 模型。

- **寻求代码助手的模型比较框架**：一位成员询问是否存在一个框架，可以比较 **Mistral** 或 **LLaMA2** 等 AI 模型在代码助手用例中的表现。

- **讨论模型比较中 Benchmark 的相关性**：另一位成员承认 Benchmark 的存在，但警告说使用它们的前提是假设 Benchmark 本身是准确的。

- **建议使用 Leaderboard 进行模型性能比较**：为了比较模型性能，一位成员建议查看 [chat.lmsys.org](https://chat.lmsys.org) 上的 **Leaderboard** 资源。另一位成员表示感谢，并提到他们之前不知道这个网站。

**提到的链接**：

[ComPromptMized](https://sites.google.com/view/compromptmized)：Stav Cohen，以色列理工学院

  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1217240335781203978)** (2 条消息): 

- **彻底改变你的 Git Commit 方式**：一位成员强调了一个[使用 LLM 创建有意义的 git commit 消息的技巧](https://harper.blog/2024/03/11/use-an-llm-to-automagically-generate-meaningful-git-commit-messages/)，其中包括一个 pre-commit-msg git 钩子，它调用 `llm` CLI 来总结代码更改以生成 commit 消息。得益于对语言模型的巧妙利用，commit 消息从模糊变得信息化。

**提到的链接**：

[Use an llm to automagically generate meaningful git commit messages](https://harper.blog/2024/03/11/use-an-llm-to-automagically-generate-meaningful-git-commit-messages/)：我通过使用 AI 自动生成有意义的消息，改变了我的 git commit 流程。这个设置涉及 llm CLI 和 git 钩子的巧妙集成，节省了我的时间。现在我可以...

  

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1217253166177845268)** (2 messages): 

- **寻求关于长上下文聊天机器人的建议**：一位成员询问了构建能够处理**长上下文（或记忆）**的聊天机器人的最佳**开源模型和框架**。在随后的讨论中没有推荐具体的模型或框架。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1217115628368957531)** (3 messages): 

- **AI 挑战 PopCap 经典游戏**：一段名为 *"Claude 3 made Plants Vs Zombies Game"* 的 YouTube 视频展示了如何使用 **Claude 3** 开发一款**《植物大战僵尸》**游戏，内容涵盖了从 Python 编程到游戏开发的各个主题。观众可以在[这段视频](https://www.youtube.com/watch?v=d7NGgglZXK8)中探索 AI 与游戏创作的交集。

- **探索先进的生成模型**：一段名为 *"Lets RAG with Command-R"* 的新视频介绍了 **Command-R**，这是一款针对**检索增强生成 (RAG)** 等长上下文任务优化的生成模型，能够利用外部 API 和工具。该模型的设计和功能在[这段视频](https://www.youtube.com/watch?v=rnP87DzGeDw)中得到了重点介绍。

- **认识 Devin：AI 软件工程师**：全球首位 AI 软件工程师 **Devin** 亮相，一段新的 YouTube 视频展示了 Devin 在自主软件工程领域的实力。感兴趣的观众可以通过观看[视频](https://www.youtube.com/watch?v=NSPtrrUQ_fw)或访问 [Cognition Labs 的博客](https://www.cognition-labs.com/blog)了解更多关于 Devin 的信息。

**Links mentioned**:

- [Lets RAG with Command-R](https://www.youtube.com/watch?v=rnP87DzGeDw): Command-R 是一款针对检索增强生成 (RAG) 以及使用外部 API 和工具等长上下文任务优化的生成模型。它的设计...
- [Devin The Worlds first AI Software Engineer](https://www.youtube.com/watch?v=NSPtrrUQ_fw): Devin 是全自主软件工程师 https://www.cognition-labs.com/blog
- [Claude 3 made Plants Vs Zombies Game](https://www.youtube.com/watch?v=d7NGgglZXK8): 将介绍如何使用 Claude 3 开发《植物大战僵尸》 #python #pythonprogramming #game #gamedev #gamedevelopment #llm #claude

  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1217539473781293056)** (1 messages): 

- **征集开源愿景者**：Soniajoseph_ 正在寻求**多模态模型**开源可解释性方面的合作者。他们分享了其 [Twitter 帖子](https://twitter.com/soniajoseph_/status/1767963316943728779)和一篇内容丰富的 [LessWrong 帖子](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic)，概述了该项目并邀请感兴趣的人士加入他们的 Discord [频道](https://discord.gg/2U2N8QmPmJ)。
- **深入研究机械可解释性**：该项目致力于**视觉**和**多模态机械可解释性 (Mechanistic Interpretability)**，并提供了一个教程、一个简短的 ViT 概述，并展示了 Prisma 的功能演示——重点关注 **logit 归因**和**注意力头可视化**等特性。

**Links mentioned**:

[Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic): 见证 dogit lens。Patch 级别的 logit 归因是一个新兴的分层图。在此加入我们的 Discord。…

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1217382172432793641)** (1 messages): 

- **追求 Phi 2 推理速度**：一位成员询问了进行 **Phi 2** 或其微调版本推理的最快方法，并提到可能使用 A100 40GB GPU。他们正在考虑使用批处理来生成大量 token，并正在探索 vLLM, Olama, Axolotl 等框架，同时询问量化是否能提高速度。

  

---



**AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1217163719394726059)** (2 messages): 

- **插件配置讨论与授权**：在 AI Engineer Foundation 的会议中，讨论了为插件使用配置选项的可能性，特别是通过将 token 作为参数传递来轻松实现授权。这基于配置选项 [RFC](https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit#heading=h.461b58g0npbn) 所促进的结构化 schema。

- **寻求新项目创意**：鼓励成员提议新项目，详细标准见 [Google Doc 指南](https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit&followup=https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit&ltmpl=docs&ec=GAZAGQ)。此外，还提到了与 Microsoft 在 prompt 文件项目负责人方面的潜在合作。

- **介绍 AI 软件工程师 Devin**：Cognition Labs 宣布发布 *Devin*，这是一款自主 AI 软件工程师，能够执行复杂的工程任务，并在 [博客文章](https://www.cognition-labs.com/blog) 中详细介绍的 SWE-bench 编码基准测试中树立了新标准。Devin 旨在沙箱环境中与常用的开发人员工具集成。

**提到的链接**：

- [Blog](https://www.cognition-labs.com/blog)：未找到描述
- [Guide to Submit Projects to AI Engineer Foundation](https://docs.google.com/document/d/1PnNlAMkIuas5_fMlqCGmmoGMBwzZcCm0yolscIsF7O8/edit#heading=h.461b58g0npbn)：未找到描述