---
companies:
- nvidia
- nous-research
- stability-ai
- hugging-face
- langchain
- anthropic
- openai
date: '2024-03-20T00:46:48.498362Z'
description: '以下是为您翻译的中文内容：


  **英伟达 (NVIDIA)** 发布了 **Project GR00T**，这是一个旨在让人形机器人通过多模态指令进行学习的通用基础模型，该模型构建于其 Isaac
  Lab、OSMO 和 Jetson Thor 等技术栈之上。英伟达还展示了拥有超过 **1 exaflop（百亿亿次）** 算力的 **DGX Grace-Blackwell
  GB200** 系统，仅需 2000 块 Blackwell 芯片即可在 90 天内完成拥有 **1.8 万亿参数的 GPT-4** 的训练。黄仁勋证实了 GPT-4
  确实拥有 **1.8 万亿参数**。新款 **GB200 GPU** 支持 float4/6 精度（每个参数约 3 位），在开启 2 倍稀疏性的情况下，其 fp4
  算力可达 **40,000 TFLOPs**。


  开源领域的亮点包括：拥有 **3400 亿参数** 的 **Grok-1** 模型正式发布；**Stability AI** 推出了开源文本转视频生成方案 **SV3D**；**Nous
  Research** 合作在 Llama.CPP 中实现了转向向量 (Steering Vectors)。


  在检索增强生成 (RAG) 方面，一个全新的 **5.5 小时教程** 演示了如何利用开源 Hugging Face 模型构建流水线；**LangChain**
  发布了关于查询路由的视频，并宣布与 **NVIDIA NIM** 集成，以实现 GPU 优化的 LLM 推理。


  重要观点包括：**杨立昆 (Yann LeCun)** 将语言能力与其他认知能力进行了区分；**山姆·奥特曼 (Sam Altman)** 预测 AGI 将在
  6 年内到来，并认为从 GPT-4 到 GPT-5 的跨越将与 GPT-3 到 GPT-4 的飞跃相当；此外还有关于 Claude 等大语言模型哲学地位的讨论。同时，专家建议大多数公司不要尝试从零开始训练模型。'
id: d367a428-c79b-48d6-ac14-2d8b0a5bfb67
models:
- gpt-4
- gpt-4o
- grok-1
- llama-cpp
- claude-3-opus
- claude-3
- gpt-5
original_slug: ainews-to-be-named-9615
people:
- jensen-huang
- yann-lecun
- sam-altman
title: 世界模拟.exe
topics:
- multimodality
- foundation-models
- hardware-optimization
- model-quantization
- float4
- float6
- retrieval-augmented-generation
- text-to-video
- prompt-engineering
- long-form-rag
- gpu-optimization
- philosophy-of-ai
- agi-predictions
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月18日至3月19日的 AI 新闻。我们为您检查了 [**358** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **21** 个 Discord 服务端（**337** 个频道，**9841** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1033 分钟**。


市面上有很多 Nvidia GTC 的回顾 —— [YouTube](https://www.youtube.com/watch?v=bMIRhOXAjYk) 做得比我们更好。

我们昨天意外地成为了新闻周期的一部分，Karan（Nous Research 的 CEO）[演示了他的 world_sim.exe 探索](https://twitter.com/swyx/status/1769920689832972574)。这纯粹是为了好玩，但对于 Roleplay Prompt Engineering 能带你走向何方，是一个非常有趣的探索。

---

**目录**

[TOC] 


---

# 第一部分：AI Twitter 综述

> 所有综述由 Claude 3 Opus 完成，从 4 次运行中选取最佳。


**NVIDIA GTC 发布会公告**

- [NVIDIA 宣布了](https://twitter.com/DrJimFan/status/1769860044324319658) Project GR00T，这是一项旨在为人型机器人学习创建 Foundation Model 的计划，该模型可以理解多模态指令并执行有用任务。它诞生于 NVIDIA 的技术栈，包括 Isaac Lab、OSMO 和 Jetson Thor。（49.5万次观看）
- [NVIDIA 揭晓了](https://twitter.com/DrJimFan/status/1769829758479876130) DGX Grace-Blackwell GB200，单个机架的计算能力超过 1 exaflop。它可以在 90 天内使用 2000 个 Blackwell 训练具有 1.8T 参数的 GPT-4。（29.1万次观看）
- [黄仁勋宣布](https://twitter.com/ethanCaballero/status/1769821908285964642) GPT-4 拥有 1.8 万亿参数。（3.6万次观看）
- [NVIDIA 的新 GPU GB200](https://twitter.com/danielhanchen/status/1769927958976963042) 具有 float4/6 精度。它每个参数使用约 3 bits，类似于 1.58bit 论文。fp4 的 40,000 TFLOPs 是在 2x Sparsity 下实现的。在 fp8 上，它达到 20 PFLOPs，而 H100 为 8 PFLOPs。它拥有 384 GB VRAM。（1.9万次观看）

**开源 LLM 与实现** 

- [Grok-1，一个 340B 参数的开源模型发布。](https://twitter.com/ibab_ml/status/1769770983924142475) 该仓库正在变得流行。（20.8万次观看）
- [Nous 与 @voooooogel 合作](https://twitter.com/Teknium1/status/1769752208466383205)，根据 @jam3scampbell 的原始论文在 Llama.CPP 中实现了 Steering Vectors/Control Vectors。（1.3万次观看）
- [Stability AI 发布了 SV3D](https://twitter.com/slashML/status/1769938991577489555)，这是一个用于 Text-to-Video 生成的开源解决方案，包含了完整的训练过程。（1千次观看）

**检索增强生成 (RAG)**

- [一个新的长篇（5.5 小时）RAG 教程已上线](https://twitter.com/mrdbourke/status/1769897780796117227)，从零开始构建一个 Pipeline，使用一份 1200 页的营养学 PDF 创建 "NutriChat"。它使用免费的开源 HF 模型，无需 API。（6千次观看）
- [LangChain 发布了其 RAG From Scratch 系列的第 10 个视频](https://twitter.com/LangChainAI/status/1769759838094106818)，重点关注使用 LLM 的逻辑推理或语义相似性进行查询路由。（3.5万次观看）
- [LangChain 宣布与 NVIDIA NIM 集成](https://twitter.com/LangChainAI/status/1769851779003695143)，用于 RAG 应用中 GPU 优化的 LLM 推理。（1.5万次观看）

**新兴趋势与观点**

- [Yann LeCun 分享了观点](https://twitter.com/ylecun/status/1769768065712177615)，认为语言与规划、推理、共情等其他认知能力是截然不同的。认知的绝大部分似乎与语言无关。（15.1万次观看） 
- [Sam Altman 在 Lex Fridman 的采访中暗示 AGI 可能会在 6 年内到来](https://twitter.com/AISafetyMemes/status/1769766600486584553)。他表示从 GPT-4 到 GPT-5 的跨越将类似于 GPT-3 到 GPT-4。与其他公司相比，OpenAI 较少卷入文化战争。（6.6万次观看）
- [关于像 Claude 这样的 LLM 是否在某种哲学意义上“活着”存在讨论](https://twitter.com/KevinAFischer/status/1769771964443316563)，这标志着一种新的文化分歧。（6千次观看）
- [有观点认为](https://twitter.com/finbarrtimbers/status/1769733698499535069) 大多数公司不应该从头开始训练模型，因为这通常没有意义。（1.3万次观看）


---

# 第 0 部分：总结的总结的总结


> 鉴于 [Claude 3 Haiku 最近发布](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww)，我们将其加入到本次综述运行中供您对比。在构建 AINews 平台以提供更好 UX 的期间，我们将继续并排运行这些模型。

## Claude 3 Haiku (3B?)


- **AI 硬件优化**：成员们积极分享了关于构建高效运行 LLM 的硬件设置的策略和经验，包括关于利用 EPYC CPU、冷却 Tesla K80 GPU 以及评估不同 GPU 选项的成本效益权衡的讨论。

- **光子计算的进展**：社区对光子计算机芯片的潜力表示兴奋，成员们分享了教育资源，如 [Asianometry 关于硅光子学和光网格的视频](https://www.youtube.com/watch?v=29aTqLvRia8) 以及 [关于新芯片突破的视频](https://youtu.be/8ohh0cdgm_Y)。

- **探索检索增强生成 (RAG)**：讨论深入探讨了优化 RAG 模型、设计专门的“RAG 模式”以及利用 Cohere 模型的鲁棒性，成员们分享了资源，如 [实现 RAG 的逐步指南](https://youtu.be/w7Ap6gZFXl0) 和一篇 [关于使用 RAG 赋能 AI 助手的 Medium 文章](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915)。

- **大型语言模型（LLM）的基准测试与评估**：社区对当前的 LLM 基准测试系统表示怀疑，质疑其可靠性和可复现性，特别是对于像 GPT-4 和 Grok-1 这样的大型模型。成员们讨论了对更强大、更标准化的评估方法的需求。

- **AI 推理与推断的进展**：讨论强调了增强 LLM 推理能力的新方法，例如 [Maisa 知识处理单元 (KPU)](https://maisa.ai/blog/kpu) 和关于教导 LLM “三思而后行”的 [Quiet-STaR 项目](https://arxiv.org/abs/2403.09629)。

## Claude 3 Sonnet (14B?)

- **大型语言模型 (LLM) 创新与发布**：

  - Stability AI 发布了 [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d)，可从单张图像创建 3D 网格。
  - xAI 发布了 [Grok-1](https://github.com/xai-org/grok-1)，一个拥有 314B 参数的 Mixture-of-Experts 模型，引发了关于性能和资源需求的讨论。
  - [NVIDIA GTC 主旨演讲](https://www.youtube.com/watch?v=Y2F8yisiS6E) 中传闻的 GPT-4 1.8T 参数引起了多个社区的兴趣。

- **模型性能比较与基准测试**：

  - 用户比较了 Stable Cascade 和 SDXL，Cascade 在处理复杂提示词方面表现更好，但运行速度较慢 (Stability.ai)。
  - 关于 Grok-1 的性能存在争议，尽管其规模庞大，但对其真实世界能力的怀疑依然存在 (Nous Research AI, Latent Space)。
  - 人们对 LLM 基准测试系统的可靠性以及 openLLM 排行榜结果的可复现性表示担忧 (Eleuther)。

- **微调与模型优化技术**：

  - [Unsloth AI 仓库](https://github.com/unslothai/unsloth) 因其更快且显存高效的 QLoRA 和 LoRA 微调而走红 (Unsloth AI)。
  - 讨论涵盖了 Gemma 7b 与 Mistral-7b 的性能对比以及 [Mixtral 分支支持](https://github.com/unslothai/unsloth/pull/145) (Unsloth AI)。
  - 一种旨在提高模型准确性和样本效率的有前景的方法被预告，并计划进行更大规模的测试 (Skunkworks AI)。

- **硬件与基础设施讨论**：

  - NVIDIA 即将推出的采用 GDDR7 显存且带宽提升 55% 的 [GeForce RTX 50 系列 "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) 受到关注 (Nous Research AI, OpenAccess AI Collective)。
  - 讨论了 LM Studio 中的 ROCm 兼容性问题和多 GPU 支持 (LM Studio)。
  - 探索了优化 LLM 硬件的策略，包括 DIY EPYC 系统和 GPU 冷却 (LM Studio)。

- **API 与模型集成**：

  - 用户期待 LM Studio 中的 [Command-R 模型集成](https://github.com/ggerganov/llama.cpp/pull/6033)。
  - 讨论了 Perplexity AI 的 API 不一致性和 Token 限制问题 (Perplexity AI)。
  - OpenAI 的 API 被用于个性化博客文章翻译的实验 (Datasette - LLM)。

- **开源定义与许可**：

  - Grok-1 在 Apache 2.0 许可下的发布引发了关于大型模型开源影响的讨论 (Interconnects)。
  - 社区辩论了是否需要一个清晰、实用的开源定义以避免争议 (Interconnects)。

## Claude 3 Opus (>220B?)

- **Stable Video 3D 彻底改变内容创作**：Stability AI 发布了 [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d)，它可以从单个物体图像创建 3D 网格（meshes）和新颖的多视角，超越了之前的 **Stable Zero123** 及其开源替代方案的技术能力。

- **Cascade 在复杂性上超越 SDXL**：在用户基准测试中，Stable Cascade 在处理复杂 Prompt 方面超过了 SDXL，尽管其生成速度较慢——大约需要 90 秒，而 SDXL 仅需 30 秒。

- **Claude 3 Opus 表现出色但“无限”查询存在误导**：Perplexity Pro 用户在 **Claude 3 Opus** 上可以获得“无限”每日查询，但 600 次的查询上限让一些人感到被误导。与此同时，Claude 3 Opus 在创意写作任务和对比测试中给人留下了深刻印象，例如[手机对比实验](https://www.perplexity.ai/search/how-does-the-tzGt2.woRkCHJhNXY8V_Gg)。

- **Unsloth AI 冒充者警报**：用户警告称，有一个冒充 Unsloth 开发者 **Daniel Han** ([starsupernova0](https://discord.com/users/starsupernova0)) 的诈骗账号正在发送好友请求。[Unsloth AI repository](https://github.com/unslothai/unsloth) 正在 GitHub 上流行，它提供了一个用于更快速、更节省显存的 QLoRA 和 LoRA 微调工具包。

- **Grok-1 发布引发辩论**：**Grok-1**（一个 314B 参数的 MoE 模型）的发布让 AI 社区反应不一，既有兴奋，也有对其真实性能的怀疑。讨论涉及该模型在 [GitHub 上的开源发布](https://github.com/xai-org/grok-1) 以及 NVIDIA GTC 主旨演讲期间潜在的 [GPT-4 泄露](https://www.youtube.com/watch?v=Y2F8yisiS6E)。

- **光子学引起 CUDA 社区兴趣**：光子计算机芯片的进展成为热门话题，例如 NVIDIA 即将推出的 **GeForce RTX 50 系列 "Blackwell"**，配备 [28 Gbps GDDR7 显存](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed)。社区还分享了诸如 [Asianometry 关于硅光子学的视频](https://www.youtube.com/watch?v=29aTqLvRia8)等教育资源。

- **Triton Puzzles 挑战 GPU 爱好者**：一套全新的 [Triton Puzzles](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) 被推出，用于教育性的 GPU 问题解决，尽管最初存在一些 Bug。CUDA 社区正在积极讨论诸如 "Producer Provides" 和 "Consumer Takes" 等内存管理策略，以优化 LLM 推理的流水线并行（pipeline parallel）实现。

- **Axolotl 通过 ScatterMoE 提升性能**：Axolotl 开发团队推出了 **ScatterMoE**，这是一种优化方案，承诺比 Huggingface 的 MoE 实现有显著的吞吐量提升，代码可在其 [GitHub 分支](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe)上获取。建议使用 PyTorch 2.2 或更高版本以确保兼容性。

- **API 泄露 LLM 机密**：研究人员发现，API 可能会无意中泄露有关专有 LLM 的信息，包括架构细节，而在 [OpenAI 的 GPT-3.5-turbo](https://arxiv.org/abs/2403.09539) 上进行实验的成本不到 1,000 美元。人们对被低估的 LLM 规模以及 Mixture of Experts (MoE) 架构的潜在使用感到担忧。

- **Maisa 的 KPU 承诺推理能力的飞跃**：Maisa 展示了其全新的 **Knowledge Processing Unit (KPU)**，它与 LLM 集成以增强复杂任务的解决能力。[KPU 白皮书和博客](https://maisa.ai/blog/kpu)详细介绍了其架构和潜力，但一些人在没有更多实质性证据以及与 GPT-4-turbo 进行对比的情况下表示怀疑。

## ChatGPT (GPT4T)

- **AI 内容创作演进**：Stability AI 的 [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d) 标志着 AI 驱动的 3D 内容生成领域的一次重大飞跃，展示了 AI 从单张图像创建复杂多视图内容能力的快速演进。这一进步不仅展示了技术的发展，还引发了关于内容创作未来的讨论，推向了 AI 可能性的边界。

- **区块链对 AI 的影响**：Stability AI 对区块链合作伙伴关系的涉足，反映了 AI 社区内部关于区块链技术在 AI 发展中作用的更广泛张力。虽然一些人看到了创新的潜力，但另一些人则对可访问性、开放性以及 AI 平台的未来方向表示担忧，突显了在平衡技术进步与社区价值观及可访问性方面的关键辩论。

- **模型性能与伦理考量**：围绕 Perplexity AI 的 Claude 3 Opus 和 Stability AI 合作伙伴关系的期待与怀疑，强调了 AI 社区对模型性能、伦理考量和透明度持续存在的关注。这些讨论反映了关于 AI 技术伦理影响、AI 公司透明沟通需求以及使 AI 发展与伦理标准和用户期望保持一致重要性的更广泛辩论。

- **AI 优化中的技术增强与社区参与**：社区对 Unsloth AI 的 GitHub 项目的支持，突显了对旨在提高 AI 效率和减少资源消耗的技术增强的浓厚兴趣。这种参与标志着社区在优化 AI 技术以获得更好性能和更低准入门槛方面的动力，反映了推动 AI 优化和应用边界的集体努力。

- **AI 伦理、开放性与可访问性辩论**：围绕 AI 在科学同行评审中作用的讨论（如 Nous Research AI 的考察）以及由 Latent Space 引发的关于 LLM 的辩论，突显了关于 AI 技术伦理考量、开放性和可访问性的持续对话。这些辩论涵盖了对 AI 对科学诚信影响、AI 在内容创作中的伦理使用，以及专有进步与开放创新之间平衡的担忧，强调了在伦理和可访问框架内引导 AI 发展的复杂性。

- **AI 训练方法论与数据管理**：Eleuther 中关于优化语言模型 scaling laws 的对话，以及 LAION 关于 DALL-E 3 数据集迁移至新 [Hugging Face repository](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) 的更新，体现了 AI 训练和数据处理实践中的持续创新。这些讨论强调了 AI 社区在确保数据可访问性和可重复性的同时，专注于增强模型效率、准确性和通用能力，展示了为更广泛的应用和影响而完善 AI 技术的持续努力。

这些主题共同捕捉了 AI 领域的动态和多面性，其特点是快速的技术进步、伦理和政策辩论、社区在优化工作中的参与，以及对增强 AI 训练和数据管理实践的持续追求。

---

# PART 1: 高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Video 3D 彻底改变内容创作**：Stability AI 发布了 [Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d)，它可以从单个物体图像创建 3D meshes 和新颖的多视图，超越了之前的 **Stable Zero123** 及其开源替代方案的能力。

- **Cascade 在复杂性上超越 SDXL**：在用户基准测试中，Stable Cascade 在处理复杂 prompts 方面优于 SDXL，尽管生成时间较慢——大约 90 秒，而 SDXL 为 30 秒。

- **区块链热潮破坏氛围**：用户对 Stability AI 与区块链实体的合作伙伴关系表示担忧，担心这可能会对备受期待的 Stable Diffusion 3 (SD3) 等模型的未来开放性和可访问性产生负面影响。

- **SD3 Beta 的等待升温**：用户社区正翘首以盼，急切期待 Stable Diffusion 3 的 Beta 版本发布，传闻该版本将结合高质量输出与高效 runtime。

- **Safetensor 转换困惑**：关于将 PyTorch .pt 文件转换为 safetensors 的咨询引发了讨论，令人欣慰的是，大多数用户界面会阻止从 .pt 文件执行脚本，从而降低了安全风险，尽管文中未提及直接的解决方案。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **并非真正的无限**：Perplexity Pro 用户被承诺在 Claude 3 Opus 上享有**无限次每日查询**，但发现“无限”一词具有误导性，因为显然存在 600 次查询的上限。担忧指向了可能的虚假陈述和法律后果。

- **创意显微镜下的 Claude 3 Opus**：用户深入探讨了 Claude 3 Opus 在创意写作方面的能力，使用了一个关于智力升级的 Prompt，提醒他人分享 Thread 以增加曝光度，并进行了手机对比实验。

- **Midjourney 与 Stability AI 划清界限**：在针对 AI 开发者的关键讨论中，Midjourney 对 Stability AI 的立场引发了关于 AI 社区内政策和伙伴关系的对话。

- **API 模型的命运悬而未决**：工程师们讨论了一个原定停用的模型意外继续服务的情况，并注意到 Sonar API 在涉及 Donald Trump 的新闻响应中存在差异，突显了内容一致性的不可预测性。

- **通过 API 进行职位搜索和 Token 限制**：虽然 Perplexity 的 API 以其检索职位发布的潜力吸引了用户，但结果的不一致令人沮丧，用户也对 max token 设置如何影响响应质量感到好奇。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **警惕冒充者**：针对冒充 **Daniel Han** ([starsupernova0](https://discord.com/users/starsupernova0)) 并试图通过好友请求诈骗 Unsloth AI 用户的行为发布了警报。敦促用户举报此类账户以维护社区安全。

- **Unsloth AI 在 GitHub 上大放异彩**：[Unsloth AI 仓库](https://github.com/unslothai/unsloth)正处于 Trending 状态，它提供了一个在 **QLoRA 和 LoRA 微调期间提速 2-5 倍且减少 70% 内存占用**的工具包。感谢社区对该仓库的点赞（Star）支持。

- **微调和模型讨论持续升温**：辩论集中在针对特定领域任务的 **Gemma 7b 与 Mistral-7b** 之争，Unsloth AI 修复了 Gemma 中的 Bug。此外，还分享了 [Mixtral 的分支支持](https://github.com/unslothai/unsloth/pull/145)，以及像 [Uiverse.io](https://uiverse.io/elements) 这样提供 CSS 或 Tailwind 开源 UI 元素的资源。

- **微调的磨难与挑战**：用户在 **Colab 上微调 Mistral 7b** 以及区分 **LoRA 和 QLoRA** 时遇到了问题。困难包括将模型保存到 Hugging Face 等平台时出错，以及关于 Unsloth 不支持的模型（如 **OpenAI 的 GPT-4**）的部署咨询。

- **Epochs vs 知识与模型集成见解**：关于有效模型训练所需的 Epoch 数量的讨论，对于像 **Tiny Mistral** 这样的 LLM 进行更长时间训练的益处尚无明确共识。讨论包括对配置设置的发现（[Hugging Face 上的 Tiny Mistral 模型](https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1)），并针对处理大数据集的理想 Rank 和 Alpha 值提出了建议。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **工程师们热切关注新模型集成**：在 Pull Request #6033 合并后，用户们正热烈讨论即将集成到 **LM Studio** 的 **Command-R 模型**。同时，拥有 314B 参数的巨型模型 **Grok** 引起了许多人的兴趣，但由于巨大的资源需求，在本地运行似乎并不现实。

- **硬件优化热潮**：关于优化硬件的讨论非常火热。拥有 **RTX 2070 Super** 和 **1660 Super GPU** 的用户正在寻找最适合其配置的模型，而另一些人则考虑以 **150 欧元**的价格购买 **Tesla K80**，并使用 3D 打印导风罩进行散热。DIY 玩家们在争论构建 **EPYC 系统**的优劣，权衡充足的 PCIe 通道与成本之间的关系。

- **模型管理之谜**：模型兼容性引起了社区的困惑。据澄清，**llama.cpp** 尚未完全支持 **Cohere 的 Command-R 模型**，需要 2024 年 3 月 16 日之后的更新以及 [b2440](https://github.com/ggerganov/llama.cpp/releases/tag/b2440) 版本才能配合 **GGUF 文件**运行。不过对于 **AVX beta 用户**来说有个好消息，像 **Mistral** 这样成熟的模型应该可以平稳运行，尽管还不支持像 **starcoder2 或 gemma** 这样最新的模型。

- **ROCm 之谜**：一位 **ROCm 用户**发出了求助信号，寻找同道中人；而另一位用户指出 **AMD Radeon 6700 XT** 与 **ROCm** 不兼容，且 LM Studio 目前仅限于使用主 GPU。

- **插件追求与配置探索**：LM Studio 的爱好者们正在 [lmstudio-ai/configs](https://github.com/lmstudio-ai/configs) 中寻找模型预设的“圣杯”，而另一位勇敢的探索者正在寻求关于在 **Local Inference Servers** 上通过 JSON function calling 调用模型的指导。

- **AI Agent 愿景**：一条简短的消息透露了对合适 Agent 系统的渴望，以实现某种创意构想，这暗示了在表面之下涌动着好奇心与算法艺术的结合。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NVIDIA 加速迈向未来**：即将推出的 [NVIDIA GeForce RTX 50 系列 "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) 将采用 28 Gbps 的 GDDR7 显存，有望比 GDDR6 提升高达 55% 的带宽。

- **AI 的“色情 Claude”引发争论**：一个声称“色情 Claude”能增强 AI 输出的聊天机器人实验引发了激烈讨论，有人将其比作“反向 Sydney”，而其他人则关注实质性话题，如 Apple 的 AI 工作以及 [Hugging Face](https://huggingface.co/papers/2403.07691) 上的新型 ORPO 算法。

- **Grok-1 的发布引发争论**：AI 社区对拥有 3140 亿参数的模型 Grok-1 反应不一，一些人对其真实性能持怀疑态度。同时，在 NVIDIA CEO 可能在 GTC 主旨演讲中失言后，关于 GPT-4 拥有 1.8 万亿参数的传闻引发了关注，相关视频可在 [YouTube](https://www.youtube.com/watch?v=Y2F8yisiS6E) 上观看。

- **Perplexity 困扰 Llama-2 用户**：使用 [Kaggle notebook](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) 计算 Llama-2 的 Perplexity（困惑度）实验引起了混乱，而关于 Mistral 和 Llama 等 LLM 的扩展与缩减讨论成为了焦点，重点在于财务和技术可行性。

- **RAG 团队深入探讨**：RAG（retrieval-augmented generation）社区内的讨论深入到了优化 RAG 模型属性、设计专门的“RAG 模式”，并利用 **Cohere** 模型的鲁棒性，旨在改进上下文处理和多样化输出结构等功能。此外，一段 Python 脚本方法已在 [GitHub](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py) 上分享。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **免费获取常春藤盟校秘籍！**：工程师们讨论了访问免费常春藤盟校课程的价值，例如来自 MIT 和 Stanford 的讲座，并分享了 Carnegie Mellon University 一位教授的网页，展示了其在算法和 Machine Learning 方面的贡献。这个包含近 7 年内容的页面可以在[这里](https://www.cs.cmu.edu/~dwoodruf/)找到。

- **利用 KPU 挑战 AI 推理极限**：一个专注于矩阵乘法的 [AI 加速项目](https://github.com/trevorpogue/algebraic-nnhw) 以及 Maisa 的 **Knowledge Processing Unit** ([Maisa KPU](https://maisa.ai/blog/kpu)) 成为分享的热点，旨在提升 Large Language Model (LLM) 任务的复杂性。讨论还涉及了 LLM 中的 “ThoughtStream” token 和 pause token 的概念，以增强推理和推断能力。

- **Grok 的抱怨与 GPU 的炫耀**：在关于模型性能和基准测试的辩论中，xAI 的 **Grok**（一个 **3140 亿参数的 Mixture-of-Experts 模型**）面临关于其实际效用的审查，人们对 **Mamba** 等不同架构中的投机采样（speculative sampling）持怀疑态度。此外，目前的 LLM 基准测试系统也受到了显著批评，特别是关于 **Llama2-70b** 等模型在 openLLM 排行榜结果的可复现性存疑。

- **规模至关重要：数据复杂度对 Scaling Laws 的影响**：成员们讨论了语言模型 Scaling Laws 如何随数据复杂度而变化，以及句法属性和可压缩性对扩展特性的影响。使用 gzip 等压缩指标，可能通过识别具有有利词汇密度的数据集，为创建最佳训练数据混合物提供参考。

- **Bigram 入门与 N-Gram 细节**：在 n-gram 统计领域，社区交流了如何自回归地采样具有特定 n-gram 统计特性的字符串。一个辅助该过程的脚本 [generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py) 表明，高阶 n-gram 规范本质上决定了低阶统计特性。

- **Gaudi2 上的 Llama 以及对 Harness 更新的追求**：用户分享了使用 **lm-eval-harness** 在 **Gaudi2** 上为 **Llama** 实现功能的经验，面临了工具中的模型选择问题，并讨论了新发布的 0.4.2 版本，可在[这里](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2)获取。社区仔细研究了在 `wikitext` 等任务中进行困惑度（perplexity）评估的方法差异，如 `loglikelihood_rolling`。

- **是否对 The Pile 进行洗牌**：围绕 [the Pile](https://pile.eleuther.ai/) 训练前的准备工作展开了讨论。会议澄清了虽然原始文件没有经过洗牌，但 Hugging Face 上的预分词（pretokenized）数据是开箱即用的，已被 Pythia 成功采用，且原始 Pile 中的训练/测试/验证集划分可能是经过洗牌的。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 模型在处理 Playwright 代码时遇到困难**：一些用户遇到 **GPT-3.5 Turbo** 无法生成正确的 **Playwright** 测试代码的问题，这表明模型可能不熟悉最新的库更新。建议尝试使用 **GPT-4**，并考虑将任务拆分为更小的块以提高性能。

- **关于 AI 拒绝执行任务的大讨论**：社区注意到模型拒绝完成任务的情况有所增加，引发了关于 meta-prompting 等策略的讨论。担忧还涉及当前的内容政策，用户希望 OpenAI 能够解决并放宽这些政策。

- **分类难题与上下文窗口**：在 Prompt Engineering 的讨论中，关于优化分类提示词以获得更好的召回率和更少的误报的建议正在流传。关键建议包括测试提示词在总上下文窗口（Context Window）中的位置影响，以及考虑迁移到更强大的模型。

- **对 GPT-5 的期待与 GPT-4 网页搜索技能**：用户们急切地询问 **GPT-5** 的发布时间，但目前尚无定论。同时，人们对如何将 **GPT-4** 的网页搜索功能集成到 API 中感到好奇，该功能因其增强的对话能力而受到赞赏。

- **OpenAI API 使用技巧与隐私政策**：围绕 API Key 管理和数据隐私的担忧引导用户查阅 [OpenAI 企业隐私政策](https://openai.com/enterprise-privacy) 以了解详情。此外，有关 **GPT** 无响应的报告促使人们参考 OpenAI 的支持页面 [help.openai.com](https://help.openai.com) 以寻求帮助。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **渴望 NL2SQL 终极方案的技术热衷者**：一名正在开发 **NL2SQL pipeline** 的成员对使用 BAAI/llm-embedder 和 TheBloke/nsql-llama-2-7B-GGUF 配合 FAISS 的准确性表示担忧，并寻求关于更精确模型和 Embeddings 的建议。

- **Hugging Face 聚会**：人们对新的 **Hugging Face** 计划表现出极大的热情，包括模型和数据排行榜。此外还讨论了平台的容量、针对新手的务实使用指南，以及 [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) 等学习资源的链接。

- **惊叹于 NVIDIA 的计算实力**：**Nvidia GH200 Grace Hopper Superchip** 成为热议话题，象征着计算效率的进步；然而，文中未讨论进一步的技术细节。

- **开创 Medusa 的并行 Token 预测**：[Medusa](https://arxiv.org/abs/2401.10774) 引起了广泛关注，这是一种创新的并行 Token 预测方法，旨在增强 Large Language Model (LLM) 的推理能力，有望打破受限的顺序 Token 生成模式。

- **AI 悄然进入科学同行评审**：一项研究揭示了同行评审文本中潜在的 LLM 修改痕迹，发现某些评审特征可能与 AI 修改的内容相关。这引发了关于 LLM 改变科学讨论的跨学科关注 ([研究链接](https://arxiv.org/abs/2403.07183))。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**旧文档的新技巧**：提出了将文档视为 **Retrieval-Augmented Generation (RAG)** 流水线中动态实体的交互技术，可能通过更复杂的交互提高 RAG 性能。讨论包括一份[分步指南](https://youtu.be/w7Ap6gZFXl0)，涵盖了使用 **LlamaParse** 和 **Qdrant** 等工具实现高效 RAG 的方法。

**LlamaIndex 0.10.20 对工程师至关重要**：**LlamaIndex v0.10.20** 的发布带来了全新的 **Instrumentation module**，提供了增强的可观测性功能和 API 调用监控，并在共享的 Notebook 中进行了演示。发布公告和资源可以通过其 [Twitter 更新](https://twitter.com/llama_index/status/1768730443921396220)找到。

**搜索之旅**：展示了一种名为 **Search-in-the-Chain** 的新方法，它集成了检索和规划，以实现终极的问答能力——这可能会彻底改变 QA pipeline 中的实时调整能力。一篇关于该主题的论文受到了关注，社区对这篇 [推文](https://twitter.com/llama_index/status/1769035278063399208) 表现出浓厚兴趣。

**简历路由革命**：一篇博客文章展示了一个结合了 **LlamaParse** 和 **LlamaIndex** 的新模型，旨在促进高效的职位匹配，能够相对轻松地解析复杂的简历格式。Kyosuke Morita 关于该主题的文章可以在此 [Twitter 线程](https://twitter.com/llama_index/status/1769147791002264008)中找到。

**Agentic Memory 架构登场**：**MemGPT** 的出现（一种旨在增强 AI Agent 记忆功能的架构）有望显著改进 Assistant API，重点在于可靠的记忆操作。工程师们可以参考 [网络研讨会推文](https://twitter.com/llama_index/status/1769408792633229455) 以获取更多启发。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Yann LeCun 的内心独白困境**：[Yann LeCun 关于 LLM 的争议性观点](https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)引发了关于语言是主要辅助推理，还是视觉空间处理更为基础的辩论。“wordcels”（文字细胞）与“shape rotators”（形状旋转者）的概念并列出现，这源于 LeCun 据称没有内心独白（inner monologue）的爆料。

- **对 GPT-5 的猜测升温**：对 GPT-5 能力潜在飞跃的期待日益高涨，这主要受到 Sam Altman 对重大改进的暗示以及 OpenAI 被推测处于开发前沿的推动。讨论内容包括与 Nvidia GTC 活动相关的“万亿参数聊天机器人”的预期，以及 LLM 进展中出现量子跃迁的可能性。

- **Grok-1 的圆周率日惊喜**：科技界对 xAI 在圆周率日发布的 *Grok-1* 做出反应，这是一个拥有 314B 参数的 MoE 模型，引发了将其能力与其他顶级 LLM 进行对比的评估。对话范围涵盖了性能、开源发布的可能动机，以及关于其体量和并行计算策略的笑话。

- **Lex 乏善可陈的 OpenAI 访谈**：Sam Altman 在 [Lex Fridman Podcast](https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx) 上的亮相让社区渴望获得更多实质性的收获。对话指出，访谈缺乏对 OpenAI 策略和 Ilya Sutskever 参与情况的深入见解，其中还穿插了对 Lex 播客风格的戏谑。

- **深入理解 Transformer 范式**：**Paper Club** 环节对 Transformer 的魅力提供了宝贵的见解；其 Attention 机制解决了过去模型的编码限制，并允许训练中的并行处理，澄清了关于 LLM 计算效率的疑虑。活动暗示即将发布一篇博客文章，承诺进行详细回顾。
  
- **90 年代嘻哈 AI 奏响反思节拍**：由 [Suno](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9) 开发的 AI 创作了一首具有 90 年代嘻哈风格的歌曲，思考了 AI 在创意领域中具有挑战性的角色，并引发了关于机器生成艺术边界的讨论。

- **AI 在行动：公会集结**：一场内容丰富且多样的对话，包括成员对一篇详细博客文章的预告、关于[高级 RAG 技术](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4)的资源分享，以及引用了该俱乐部的[协作学习文档](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **DALL-E 3 数据集迁移**：**DALL-E 3** 数据集已迁移，并非此前认为的被删除，工程师现在可以通过其新的 [Hugging Face 仓库](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)进行访问。

- **为你的数据集提交 Commit**：Hugging Face 数据集可以使用特定的 commit id 进行加载，从而增强 AI 实验的可复现性；该功能已在 [Hugging Face 的数据集加载指南](https://huggingface.co/docs/datasets/en/loading#hugging)中列出。

- **领悟 Grok 模型**：**Grok** 是由 xai-org 开发的 314B 参数模型，目前处于性能讨论的中心，工程师们将其与较小的 **Mixtral** 进行对比；**Grok-1** 的 GitHub 仓库可以在[此处](https://github.com/xai-org/grok-1)找到。

- **利用 Cog 增强标注**：元数据正被用于提高 **Cog 模型**中描述（caption）的准确性，一些用户分享了他们的策略和脚本，其中一个可以在 [GitHub](https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py) 上获取。

- **GPT-4 架构推测**：关于 **GPT-4 潜在架构**的传闻甚嚣尘上，泄露消息暗示其为一个 1.8 万亿参数的 MoE 模型，但尚未得到证实；可以通过这张[推文图片](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large)进一步了解相关推测。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Axolotl 提升模型优化水平**：Axolotl 开发者推出了 **ScatterMoE**，这是一项旨在提升 Huggingface 吞吐量的优化技术，用户可前往其 [GitHub 分支](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe)了解更多详情。为了确保兼容性，必须升级到 **PyTorch 2.2** 或更高版本，部分用户已经在使用 PyTorch **2.2.1**。

**探讨 Grok 的庞大体量**：拥有 **3140 亿参数**的 **Grok-1** 模型权重的发布引发了热议。有成员评论其性能并非最优，且运行所需的资源极高。虽然目前仅发布了 **int8 版本**，但根据 [Grok GitHub 页面](https://github.com/xai-org/grok-1)，有人推测可以利用 Axolotl 的 **qLoRA FSDP** 来进行管理。

**NVIDIA 硬件热度再创新高**：预计于 2025 年左右推出的 NVIDIA RTX 5000 系列可能会带来 50% 的 VRAM 提升和 78% 的带宽增幅；具体细节可见 [Heise](https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html) 和 [TechPowerUp](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) 的文章。

**模型训练与转换难题**：在使用 `<summary>` 标签时发现了 Tokenizer 问题，导致识别出分词不一致。另一位用户在本地模型和数据设置中遇到了困难，引发了 `HFValidationError` 挑战。对话数据微调错误通过参考 Axolotl 的 readme 得到解决，通过映射额外角色并排除简短对话来处理空数据集的 "role" 数组。

**数据集对话推动新发现**：一位用户对在数学和代码数据集上微调的 **Mistral** 模型表现出兴趣，并有人建议利用 **mergekit** 等合并策略来处理海量数据，而无需单独训练。合并过程中不同模型聊天格式的兼容性也受到了质疑，但尚未得到明确解决。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **光子技术的未来大放异彩**：成员们讨论了**光子计算机芯片**的进展，分享了一个[关于突破的视频](https://youtu.be/8ohh0cdgm_Y)，并建议通过两段关于硅光子学和光网格的 [Asianometry 教育视频](https://www.youtube.com/watch?v=29aTqLvRia8)进行深入学习。NVIDIA CEO 还在 **GTC 2024** 上暗示了 AI 的未来，讨论了一个拥有 **1.8 万亿参数的新 Sota 模型**以及配备 **192GB HBM 的 B100 硬件**。
  
- **Triton 获得新拼图与可视化工具**：发布了一个新的 **[Triton 可视化工具](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing)** 以帮助调试复杂函数，同时还有一套用于 GPU 编程教学的 **Triton Puzzles**，尽管目前还存在一些 bug，如偶尔的双重可视化和段错误（segmentation faults）。

- **CUDA 爱好者挑战内存与效率**：以 CUDA 为核心的讨论在 warp 调度器参与和定义活跃 warp 方面缺乏共识，但对内存管理策略（如 "Producer Provides" 和 "Consumer Takes"）提供了更深入的见解，将这些策略应用于 LLM 推理的流水线并行（pipeline parallel）实现引起了浓厚兴趣。

- **ML 系统与硬件融合**：建议参考 [Prof. Mohamed Abdelfattah 的研究小组频道](https://www.youtube.com/@mabdelfattah88)和 [ECE 5545 (CS 5775) 课程页面](https://abdelfattah-class.github.io/ece5545/)来探索 ML 与硬件优化之间的联系。社区积极参与了关于 **ring-attention** 和 **flash-attention 实现**的讨论，并通过研究链接和 GitHub 仓库解决了内存缩放问题。

- **CUDA 与 ML 知识交流**：一位成员在 **memory coalescing** 和 **warp divergence** 等领域的 CUDA 背景被认为是学习 ML 的良好基础，并推荐了 **《Programming Massively Parallel Processors》** 一书以及 **Andrej Karpathy 的 Zero to Hero ML 系列**。关于分享 **[Programming Massively Parallel Processors](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311)** 书中练习答案的辩论，呼吁明确教育分享的伦理。

- **花絮交流与 GTC 期待**：社区分享了关于 MLSys 2024 口号的诗意笔记、关于智能手机问题的幽默，并澄清了数学中的运算顺序。成员们协调了 GTC 线下聚会，其中一人对无法参加表示遗憾。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 格式获准使用**："system"、"user" 和 "assistant" 格式已获准用于 Llama 模型，支持结构化对话。
- **通过充值实现模型变现**：针对支付查询，明确了用户需要*充值余额 (top up their balance)* 而不是直接绑定信用卡，这影响了模型交互的变现方式。
- **Sonnet 赢得角色扮演之战**：**Sonnet** 成为用户在追求一致角色扮演 (roleplay) 体验时的首选模型，在维持叙事且不产生重复或无关输出方面表现优于其他模型。
- **Prompt 技巧导航**：关于引导 Large Language Models (LLMs) 的讨论显示，通常只有第一条 system 消息被用作 Prompt，后续指令可能需要嵌入到 user 消息中。
- **API 开发与模型市场**：对话涉及了多个技术点，如公共 API 的集成、在平台上线以及联盟计划，同时考虑了成本、效率以及 OpenRouter API 对 **Sonnet** 等模型的灵活性。

讨论中相关的链接包括 [OpenRouter](https://openrouter.ai) 和 xai-org 在 GitHub 上的 [Grok 开源发布](https://github.com/xai-org/grok-1)。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain 世界中的 API 困惑**：成员们讨论了 **LangChain** 的 **astream_log** 与 **astream_events** 的优劣，担心后者处于 beta 阶段且可能被弃用。然而，对于哪种 API 更受青睐，或者它们是否旨在服务于不同的目的，尚未达成明确共识。

**社区助力文档救援**：由于用户在导航方面面临挑战，并发现材料（特别是对平台新手而言）有些匮乏，要求对 **LangChain 文档** 进行澄清和贡献的呼声引起了共鸣。

**Rubik’s AI 组建其 Beta 测试团队**：发出了对名为 **Rubik's AI** 的强大研究助手的 beta 测试邀请，承诺可以访问 **Claude 3 Opus**、**GPT-4 Turbo** 和 **Mistral Large** 等高性能模型。感兴趣的参与者请前往其 [waitlist](https://rubiks.ai/)。

**LangChain AI 展示**：从**用于数据分析的 AI 聊天机器人**到**动态书签**和**个性化营养应用**，成员们向社区分享了他们基于 LangChain 的创新。这些项目展示了高级功能的集成，并在 GitHub 上提供了仓库，通过 [YouTube](https://youtu.be/vHjc5CEoIJE) 进行了演示。

**流式传输陷入停滞**：在使用 LangChain 的 `RemoteRunnable` 尝试在 JavaScript 中进行流式输出时出现了技术问题，它会转向 `/invoke` 调用，而不是预期的 `/stream`。此事看起来很复杂，目前还没有针对 JavaScript 特定流式传输难题的最新文档或更改。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **API 可能会泄露 LLM 的秘密**：研究人员发现，API 可能会无意中泄露专有大语言模型（LLM）的信息，包括架构细节，而在 OpenAI 的 GPT-3.5-turbo 上进行实验的花费不到 1,000 美元。人们对低估 LLM 规模的担忧日益增加，对 70 亿参数的估算持怀疑态度，并认为 Mixture of Experts (MoE) 架构可能会虚增真实规模。
  
- **对模糊开源定义的苦恼**：关于开源定义的争论正在酝酿，@rasbt 的推文预示了 OSS 社区内部可能出现的分歧。成员们认识到需要对什么是开源达成明确共识，考虑到从 *Apache 2.0* 到 *GPLv3* 的各种许可证，目前正在努力创建一个*实用定义*以减少潜在争议。

- **巨兽降临：Grok-1 亮相**：xAI 宣布推出 Grok-1，这是一个拥有 3140 亿参数的巨型 Mixture-of-Experts 模型，采用 Apache 2.0 协议发布，在社区中引起了轰动。Grok-1 的性能指标表明它可能会使 Falcon 等竞争模型相形见绌，其非传统的种子（torrent）分发方式引发了关于开源 AI 政策和声誉问题的讨论。

- **数据量推测与 Chinchilla 关系**：鉴于 Grok-1 出色的性能，社区推测其训练数据集的大小，并思考 Chinchilla 研究的结论如何与 MoE 模型相关联，反映了数据规模与模型最优性之间的权衡。

- **模型交付的幽默折射出带宽困境**：在关于 Grok-1 的讨论中，一个关于通过物理运输 AI 模型以规避云端流出流量费用的笑话，凸显了传输海量数据相关的现实挑战和成本。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Alignment Lab 中的空中客车（Airbus）疑云**：Alignment Lab 分享的一条[关于空中客车的推文](https://twitter.com/alignment_lab/status/1758949148143841379)引发了成员们的困惑，有人寻求澄清到底在构建什么。
- **寻找精通 HTTP 的 Embeddings 模型**：有人询问是否存在专门针对 HTTP 响应训练的 **embeddings 模型**，并建议经过适当训练的 Transformer 可能会胜任这一角色。
- **缺失的双重微调 Mistral 模型**：关于是否存在同时在 **orca-math-word-problems-200k 数据集**和 **nvidia/OpenMathInstruct-1** 上进行微调的 **Mistral 模型**的问题浮出水面，这表明在可获取的组合训练方面存在空白。
- **Grok-1 微调动员**：有人呼吁合作微调 **Grok-1**，强调了所需的巨大计算资源和专业知识（如 **64-128 块 H100 GPU**），同时强调了现有的具有极高效率的 **MoE 训练基础设施**。
- **Grok-1：是瑰宝还是玻璃？**：围绕 **Grok-1 性能**的怀疑论已经出现，但一些成员指出了其令人印象深刻的能力，并提到它在[匈牙利国家高中毕业考试数据集](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam)上的表现可与 GPT-4 和 Claude 媲美。



---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **AI 的受控反对派还是真实的恐惧？**：在公会中分享的一条 [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) 引发了关于 **Anthropic** 真实动机的辩论，暗示它可能在充当受控反对派，以向技术人员施加压力。
- **挣扎中的 AI 内容审核**：公会成员注意到内容审核系统在有效审核包含人物的图像方面表现不佳，引发了对这些算法在实际应用中可靠性的担忧。
- **Claude Sonnet 的扩展困境**：**Claude Sonnet** 在每月处理数千万个 tokens 的项目中的可扩展性受到质疑，公会成员正在寻求关于该模型在大业务量下表现的反馈。
- **KPU：突破还是炒作？**：Maisa 新推出的 **Knowledge Processing Unit (KPU)** 在一篇 [博客文章和白皮书](https://maisa.ai/blog/kpu) 中进行了描述，引发了关于其真实潜力以及与 GPT-4 等当前模型对比的讨论。成员们强调在任何直接对比中包含 **GPT-4-turbo** 的重要性，并在没有更多实质性证据的情况下表示怀疑。
- **持怀疑态度的工程师嘲讽 AI 初创公司趋势**：随着 Maisa 推出 KPU，公会成员对 AI 初创公司典型的通过令人印象深刻的图表和 waitlists 进行技术炒作的模式进行了调侃，同时也批判性地思考了实际的缺点，如潜在的 latency 问题。[@davipar 的一条推文](https://x.com/davipar/status/1768683151780683919?s=20) 进一步阐明了 KPU 与 LLMs 协同工作的能力。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **DiscoLM 的德语困扰**：用户报告称 **DiscoLM-mixtral-8x7b-v2** 在 fine-tuning 后难以生成德语，且 **LeoLM** 模型表现出不一致性。当 **DiscoLM** 对英语 prompts 返回德语响应时，也出现了 API 问题，并且在分类任务的 fine-tuning 过程中出现了 **ValueError**。

- **服务器迁移的紧急求助**：演示服务器从家庭厨房搬迁到专业环境导致了意外的网络问题；解决该问题的努力将于下周开始。与此同时，成员们对模型训练和 prompt 遵循方面的指导表示感谢，并幽默地提到了业余爱好者设置中奇特的可靠性。

- **基准测试的忧郁与合作呼吁**：Discord 聊天显示了对德语语言模型 benchmarks 的担忧，不同的性能与 templates 和 end token 约定有关。讨论中回响着合作建立更好的 benchmarks 和高质量 datasets 的呼吁，以及通过学术界参与或私有渠道获取 benchmarks 的俏皮建议。

- **GitHub 及更多**：成员们分享了 GitHub 链接，如 [grok 模型代码](https://github.com/xai-org/grok/blob/main/model.py) 以及各种 benchmarks，如 [SuperGLEBer](https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/) 和 [XTREME](https://github.com/google-research/xtreme)。Reddit 帖子也出现在关于寻找最佳德语语言模型的讨论中。



---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Explosion 让 Prompt Engineering 变得轻而易举**：工程师们强调了来自 Explosion 的 [Prodigy Prompt Engineering 工具](https://prodi.gy/features/prompt-engineering)，指出将 Prompt Engineering 转化为数据标注任务的进步，从而提高了精确度。

- **使用 PromptTools 进行跨模型 Prompt 测试**：开源资源 [PromptTools](https://github.com/hegelai/prompttools) 因其在不同 LLM 和向量数据库（vector databases）之间进行 Prompt 测试和实验的实用性而被提及，尽管它目前缺少版本管理功能。

- **Helicone 进军 Prompt 管理领域**：[Helicone](https://www.helicone.ai/) 因其生成式 AI 应用构建能力而受到赞誉，目前正因整合了 Prompt 管理工具、版本控制和分析功能而受到关注，旨在提供更集成的 AI 开发体验。

- **PromptFoo 加入 CI/CD 阵营**：[PromptFoo](https://github.com/promptfoo/promptfoo) 因其允许用户测试和比较 LLM 输出、管理 Prompt 质量并与 CI/CD 流水线集成的功能而受到关注，支持包括 OpenAI 和 Azure GPT 在内的各种平台的模型。

- **个性化翻译定制读者体验**：一位工程师分享了他们使用 gpt-3.5-turbo 个性化博客文章翻译的实验，尝试针对不同角色（personas）定制内容以获得更好的理解和参与度，展示在 [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html)。

*遗憾的是，关于恢复 OpenAI 模型在之前 API 请求中使用的 seed 的查询由于缺乏足够细节，未包含在此摘要中。*

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **模型准确率突破即将来临**：一位成员正在准备一篇关于提高 AI 模型**全局准确率（global accuracy）**并增强样本效率的新方法的文章，计划在完善结果和视觉图表后分享。
- **寻求超算英雄**：该成员正在寻求资源以在**更大的 AI 模型上测试其方法**，此前已在 CIFAR100 上使用 VGG16 证明了一个 epoch 内测试准确率从 0.04 显著提升至 0.1。
- **资源驰援**：有人提供了**算力和资源**，以协助该新方法的验证和测试阶段。
- **为 Quiet-STaR 招募**：目前正公开招募具备 **PyTorch 和 Transformer** 知识的人员，参与 "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" 项目。
- **无关话题**：在无关频道中有一条指向 [YouTube 视频](https://www.youtube.com/watch?v=ZlJbaYQ2hm4) 的消息，未提供与技术讨论相关的背景或关联。

---

# PART 2: 频道详细摘要与链接

**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1219396298176991303)** (1 条消息): 

- **Stable Video 迈向 3D**：[Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d) 已发布，基于现有的 **Stable Video Diffusion** 技术。它以单个物体图像作为输入，输出全新的多视角，从而允许生成 3D 网格（3D meshes）。
- **将 3D 技术提升到新高度**：**Stable Video 3D** 模型在质量和多视角能力上比 **Stable Zero123** 有了极大提升。它的表现也优于 [Zero123-XL](https://objaverse.allenai.org/docs/zero123-xl/) 等开源替代方案。
- **发布两个变体**：此次发布包含两个变体：`SV3D_u` 可从单张图像生成轨道视频，无需相机调节（camera conditioning）；`SV3D_p` 扩展了功能，允许进行更复杂的视频生成。

**提到的链接**：<a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>：当我们发布 Stable Video Diffusion 时，我们强调了视频模型在各种应用中的多功能性。在此基础上，我们很高兴发布 Stable Video 3D。这是一款...

---

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1218109086101540905)** (988 条消息🔥🔥🔥): 

- **Stable Diffusion Cascade 对比 SDXL**：一位用户分享了他们的经验，认为 Stable Cascade 在执行复杂 Prompt 方面比 SDXL 更好，尽管在他们的硬件上运行速度较慢。Cascade 每次生成大约需要 90 秒，而 SDXL 只需要 30 秒。

- **对开源和加密货币的担忧**：有讨论对 Stability AI 与各种区块链公司合作表示失望。用户推测了这对 SD3 未来的影响，以及在 SD3 发布后可能向专有模型转型的趋势。

- **对 SD3 的期待**：用户正热切期待 Stable Diffusion 3 的公开发布，预计测试版（beta）邀请将很快发出。有人推测 SD3 将提供与其他工具相当的质量，同时运行效率更高。

- **将 .pt 转换为 Safetensors**：一位用户询问了在不涉及复杂编码的情况下，将 PyTorch 文件（.pt）转换为 safetensors 的替代方法。另一位用户提到大多数 UI 不会执行 .pt 文件中的脚本，从而降低了安全担忧，但并未提供替代工具。

- **Stable Video 3D 发布公告**：Stability AI 宣布发布 Stable Video 3D (SV3D)，这是一个可以从单张图像输入创建 3D 网格（meshes）的模型。该公告强调了相比于 Stable Zero123 等早期模型的改进，以及它在创建轨道视频和适应姿态调节（pose conditioning）方面的多功能性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 是一个 314B 参数的 Mixture of Experts 模型 - Base model (未 finetuned) - 8 个 experts (2 个 active) - 86B active parameters - Apache 2.0 许可证 - Code: - Happy coding! p.s. we re hiring: </li><li><a href="https://tenor.com/view/iron-man-mr-clean-mop-ai-floors-gif-27596354">Iron Man Mr Clean GIF - Iron Man Mr Clean Mop - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/yess-yes-gif-25420589">Yess GIF - Yess Yes - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: 当我们发布 Stable Video Diffusion 时，我们强调了视频模型在各种应用中的多功能性。在此基础上，我们很高兴发布 Stable Video 3D。这个新...</li><li><a href="https://tenor.com/view/avatar-cuddle-hungry-yummy-food-gif-5610436">Avatar Cuddle GIF - Avatar Cuddle Hungry - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://thedailywtf.com/articles/The_Complicator_0x27_s_Gloves">The Complicator&#39;s Gloves</a>: 优秀的软件在多个方面不断受到攻击。首先是 The Amateurs（业余爱好者），他们尽管只读完了《傻瓜编程》，却不知何故设法拿到了那份巨额合同...</li><li><a href="https://docs.python.org/3/library/pickle.html">pickle — Python object serialization</a>: 源代码：Lib/pickle.py。pickle 模块实现了用于对 Python 对象结构进行 serialization 和 de-serialization 的二进制协议。“Pickling” 是将 Python 对象层级结构转换为...的过程。</li><li><a href="https://civitai.com/models/351450/proteus-rundiffusion?dialog=commentThread&commentId=372974">Proteus-RunDiffusion - withoutclip | Stable Diffusion Checkpoint | Civitai</a>: 介绍 Proteus-RunDiffusion。在开发 Proteus-RunDiffusion 的过程中，我们的团队开展了一个探索性项目，旨在提升...的能力。</li><li><a href="https://www.youtube.com/watch?v=fibDNwF8bjs">WKUK - Anarchy [HD]</a>: 最滑稽的经济无知。—— Murray Rothbard 的《Freedom, Inequality, Primitivism, and the Division of Labor》(http://mises.org/daily/3009)。—— "Th...</li><li><a href="https://www.pny.com/professional/software-solutions/about-nvidia-gpus/nvlink">NVLink | pny.com</a>: 未找到描述</li><li><a href="https://www.pny.com/professional/software-so">Page Not Found | pny.com</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=YTE0OTVOnZU">Vancouver, Canada 1907 (New Version) in Color [VFX,60fps, Remastered] w/sound design added</a>: 我为这段 1907 年加拿大温哥华的视频进行了上色、修复，并添加了天空 VFX 效果和音效设计。这段视频是从有轨电车上拍摄的，这些...</li><li><a href="https://youtu.be/m9jg1fdOiVY?t=412">Install ComfyUI on Mac OS (M1, M2 or M3)</a>: 这段视频是一个快速演练，展示如何在 M1 或 M2 Mac 上本地安装 ComfyUI。了解更多关于 AI Animation 的信息，并注册成为 AI ...</li><li><a href="https://www.youtube.com/watch?v=5mIWo6dgTmI&ab_channel=Megaprojects">The Mushroom Motherboard: The Crazy Fungal Computers that Might Change Everything</a>: 揭开真菌计算的秘密！发现真菌作为生物计算机的惊人潜力。从 wood-wide web 到 Unconventional Computing...</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1b6skvx/wheres_waldo_beach_scenes_as_an_animated_loop/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://civitai.com/models/207992/stable-video-diffusion-svd)">Stable Video Diffusion - SVD - img2vid-xt-1.1 | Stable Diffusion Checkpoint | Civitai</a>: 查看我们的快速入门指南！https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/ 基础 img2vid 模型经过训练可以生成...</li><li><a href="https://huggingface.co/PollyannaIn4D">PollyannaIn4D (Pollyanna)</a>: 未找到描述</li><li><a href="https://youtu.be/ruANV24h0Dw?si=rVFKZqowCdpKTzgp">Короткометражный мультфильм &quot;Парк&quot; (сделан нейросетями)</a>: 短篇动画片《公园》 - 一部使用神经网络创作的极其引人入胜的短篇动画片。</li><li><a href="https://github.com/GraftingRayman/ComfyUI-Trajectory">GitHub - GraftingRayman/ComfyUI-Trajectory</a>: 通过创建 GitHub 账号为 GraftingRayman/ComfyUI-Trajectory 的开发做出贡献。</li><li><a href="https://github.com/DiffusionDalmation/pt_to_safetensors_converter_notebook#">GitHub - DiffusionDalmation/pt_to_safetensors_converter_notebook: This is a notebook for converting Stabl

将 Stable Diffusion embeddings 从 .pt 转换为 safetensors 格式。</a>：这是一个用于将 Stable Diffusion embeddings 从 .pt 格式转换为 safetensors 格式的 notebook。 - DiffusionDalmation/pt_to_safetensors_converter_notebook</li><li><a href="https://github.com/mix1009/sdwebuiapi">GitHub - mix1009/sdwebuiapi: AUTOMATIC1111/stable-diffusion-webui 的 Python API 客户端</a>：AUTOMATIC1111/stable-diffusion-webui 的 Python API 客户端 - mix1009/sdwebuiapi</li><li><a href="https://github.com/chaojie/ComfyUI-DragAnything/tree/main">GitHub - chaojie/ComfyUI-DragAnything</a>：通过在 GitHub 上创建账号来为 chaojie/ComfyUI-DragAnything 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)">Home</a>：Stable Diffusion web UI。通过在 GitHub 上创建账号来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Stability AI 的生成模型</a>：Stability AI 的生成模型。通过在 GitHub 上创建账号来为 Stability-AI/generative-models 的开发做出贡献。</li><li><a href="https://stable-diffusion-art.com/regional-prompter/)">Regional Prompter：在 Stable Diffusion 中控制图像构图 - Stable Diffusion Art</a>：你知道可以为图像的不同区域指定提示词吗？你可以通过 Regional Prompter 扩展在 AUTOMATIC1111 上实现这一点。
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1219057096780419163)** (1 条消息): 

- **Claude 3 Opus 无限制查询**：公告透露，**Perplexity Pro 用户**现在可以对 Claude 3 Opus 进行**无限制的每日查询**，该模型被誉为当今市场上最好的 LLM。Pro 用户现在可以享受无查询限制的扩展访问。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1218100055626743851)** (795 条消息🔥🔥🔥): 

- **对“无限制”使用条款的困惑**：用户对 Perplexity.ai 使用“无限制”一词感到困惑，认为考虑到每天 600 次的实际上限，这具有误导性。令人担忧的是，这种措辞可能会误导服务说明，并可能导致法律挑战。
- **关于教育幼儿的讨论**：针对儿童（尤其是五岁儿童）理解复杂概念的能力引发了辩论。一些人认为，如果解释得当（通常使用类比或 AI 辅助），儿童可以掌握高级主题，而另一些人则对其认知能力表示怀疑。
- **Pro 订阅的用户体验**：用户对 Perplexity 最新集成的 Claude 3 Opus 表示出兴趣和满意，热切讨论其在包括角色扮演在内的不同任务中的效率和适用性，其中一名用户成功测试了它对敏感话题的响应。
- **AI 的家庭使用**：家长们分享了他们如何使用 ChatGPT Pro 和 Perplexity.ai 等 AI 工具来回答孩子的好奇心，这表明孩子们会提出各种深刻的问题，而 AI 可以帮助解决这些问题，从而培养他们的求知欲。
- **用于职业和喜剧的 AI**：一名用户幽默地声称利用 Perplexity 的 Claude 3 Opus 获得了一份麦当劳的工作，但后来拒绝了，这引发了关于 AI 在日常生活中的各种应用以及对快餐等行业潜在影响的有趣调侃。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-an">未找到标题</a>：未找到描述</li><li><a href="https://x.com/AravSrinivas/status/1769475725965566167?s=20">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们已经为 Perplexity Pro 用户取消了 Claude 3 Opus（当今市场上最好的 LLM）的每日查询次数限制，现在是无限次！尽情享受吧！</li><li><a href="https://www.theverge.com/2024/3/18/24104626/apple-license-google-gemini-generative-ai-openai-chatgpt">Apple 的 AI 雄心可能包括 Google 或 OpenAI</a>：另一项重大的 Apple / Google 交易可能即将达成。</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-anime-girl-gif-26002811">Shikimori Shikimoris Not Just Cute GIF - Shikimori Shikimoris Not Just Cute Shikimoris Not Just A Cutie Anime - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://us.nothing.tech/pages/perplexity">Nothing Perplexity 优惠</a>：在 Nothing，我们正在构建一个让科技再次变得有趣的世界。还记得每一个新产品都让你兴奋不已的时光吗？我们正在带回那种感觉。</li><li><a href="https://x.com/AravSrinivas/status/1769485603622867394?s=20">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！↘️ Quoti...</li><li><a href="https://x.com/technology/status/1769597406243360937?s=20">来自 Bloomberg Technology (@technology) 的推文</a>：独家：Apple 正在洽谈将 Google 的 Gemini AI 引擎内置到 iPhone 中，这可能是一项重磅交易 https://trib.al/YMYJw2K</li><li><a href="https://fxtwitter.com/BrivaelLp/status/1769482175005577571?s=20">来自 Brivael (@BrivaelLp) 的推文</a>：Zuck 刚刚对 Grok 的发布做出了反应，他似乎并不以为然。“3140 亿参数太多了。你需要一大堆 H100，而我已经把它们都买光了” 🤣</li><li><a href="https://youtu.be/OPoWMXqq62Q?si=jk-ZbhjfkZtRkjz7">这些公司在隐藏什么？</a>：关于 Rabbit R1 和 Humane Ai Pin 的看法。如果你想支持本频道，请点击上方的“加入”按钮考虑成为 Dave2D 会员！http://twit...</li><li><a href="https://youtube.com/clip/Ugkx9gPr2y53Be9C99y-EVVWfZPjRxNQo6FL?si=0r1zDbn2FfjmrsuB">✂️ Sam Altman 谈 AI LLM 搜索</a>：47 秒 · 由 Syntree 剪辑 · 原视频 "Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419" 由 Le...</li><li><a href="https://fccid.io/2BFB4R1">FCC ID 2BFB4R1 Rabbit Inc. 的 AI 伴侣</a>：Rabbit Inc. 为其 AI 伴侣提交的 FCC ID 申请，ID 为 2BFB4R1。包含批准的频率、用户手册、照片和无线报告。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1218101595586429048)** (35 条消息🔥): 

- **创意写作探索**：Claude 3 Opus 接受了一项关于*“不断增长的智能，直到人类无法理解”*的提示词挑战。探索内容可以在[这里](https://www.perplexity.ai/search/increasing-intelligence-of-HLUn3nOzSx6Nc5ecNpe5pA)找到。
- **可见性至关重要**：提醒用户**分享他们的线程**以获得可见性，以便他人查看。说明可以在[这里](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)找到。
- **比较研究**：一位用户提到利用 Perplexity 以创新方式比较手机，展示了该平台的实际应用。具体比较过程可以在[这里](https://www.perplexity.ai/search/how-does-the-tzGt2.woRkCHJhNXY8V_Gg)查看。
- **Midjourney 的公开公告**：Midjourney 关于 Stability AI 的决定是近期讨论的核心，预示着 AI 领域的重大动向。禁令详情可以在[这里](https://www.perplexity.ai/search/Midjourney-bans-Stability-nGDGZxh5SIucTa3mCbZz8w)阅读。
- **临床研究的搜索参考**：一位用户分享了 MINDSET 研究的链接，可能旨在强调心理健康研究的重要性。感兴趣的各方可以在[这里](https://www.perplexity.ai/search/MINDSET-Study-clinical-asL8eAZuQPmkJ_2hYCgIIw)找到该研究的详情。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1218160850670583828)** (64 条消息🔥🔥):

- **API 模型不确定性**：用户讨论了一个原定于 3 月 15 日弃用但目前仍可运行的模型的功能持续性，引发了关于计划是否改变或弃用是否会在当天晚些时候发生的疑问。
- **Sonar 新闻响应的不一致性**：对比了 `sonar-medium-online` 模型与网页浏览器版本的响应，显示在提供关于 Donald Trump 的内容方面存在显著差异，部分用户获得了详细回答，而其他用户则没有。
- **通过 API 获取职位发布链接的困难**：用户尝试使用 API 生成职位搜索结果时注意到，API 有时会返回有效且可用的职位发布链接，但结果可能不一致，在实际职位列表和招聘网站的通用链接之间波动。
- **关于 Max Tokens 和响应质量的讨论**：一位用户询问了设置较低的 max tokens 值对响应质量的影响，引发了关于 API 在给定限制时的行为，以及模型在指定的 token 约束内生成全面答案的可行性的讨论。
- **对 Grok 支持确认的关注**：用户对 Perplexity 在 Grok 开源后支持它的可能性表现出兴趣，并提到在 CEO 发布推文后，公司正计划这样做。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai">pplx-api</a>：未找到描述</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api 表单</a>：使用 Typeform 将数据收集转化为一种体验。创建精美的在线表单、调查、测验等等。免费试用。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1218108656428650526)** (853 条消息🔥🔥🔥): 

- **冒充者警报**：用户警告有一个冒充 Unsloth 开发者 Daniel Han (**starsupernova**) 的诈骗账号正在发送好友请求。该冒充者的 Discord 用户名是 **starsupernova0**，鼓励社区向 Discord 举报该账号。
- **Grok-1 引发开源轰动**：Elon Musk 的 xAI 团队发布了 Grok-1，这是一个拥有 314B 参数的巨型模型，由于其庞大的体积以及利用它所需的极高 GPU 资源，引发了关于实用性和目的的讨论。
- **新工具辅助微调 (Fine-tuning)**：用户分享了用于模型微调的资源，例如用于在训练期间减少 VRAM 占用的 [Unsloth 仓库](https://github.com/unslothai/unsloth)，以及 [Unsloth 与 aikit 的集成](https://sozercan.github.io/aikit/)，用于通过配置进行微调并使用 Docker 创建模型镜像。
- **优化微调实践**：讨论了关于为 QLoRA 和其他微调策略设置最佳超参数 (Hyperparameters) 的问题。建议用户参考示例 Notebook 以获取有关超参数和训练 LLM 数据集结构的指导。
- **在线平台的技术故障**：成员们交流了克服 Kaggle 环境问题和部署 AI 模型的技巧，强调了简化流程的必要性，并对间歇性问题表示沮丧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://lightning.ai/live-session/a35263e0-0428-40b6-8828-8e72773a284d">Lightning AI | 将创意转化为 AI，闪电般的速度</a>：AI 开发的一站式平台。协作编码、原型设计、训练、扩展、提供服务。直接在浏览器中运行，无需安装。由 PyTorch Lightning 的创作者打造。</li><li><a href="https://docs.anthropic.com/claude/page/cosmic-keystrokes">Cosmic keystrokes</a>：未找到描述</li><li><a href="https://huggingface.co/Crystalcareai/GemMoE-Beta-1">Crystalcareai/GemMoE-Beta-1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T：在英语和多语言评估中超越 LLaMA 7B 2T (RWKV-v5)</a>：一个 Linear Transformer 模型刚刚在英语和多语言评估中，以更少的训练 Token 数量超越了 Transformer 模型的黄金标准 LLaMA 7B。这是历史性的第一次。</li><li><a href="https://x.ai/about">关于 xAI</a>：未找到描述</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.ai/">博客</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2401.04088">Mixtral of Experts</a>：我们推出了 Mixtral 8x7B，这是一种稀疏混合专家（SMoE）语言模型。Mixtral 具有与 Mistral 7B 相同的架构，不同之处在于每一层由 8 个前馈块组成 (...</li><li><a href="https://x.ai/blog/grok">宣布推出 Grok</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish">Piper TTS Spanish - 由 HirCoir 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth 修复 Gemma Bug</a>：Unsloth 正在修复 Google 的开源语言模型 Gemma。</li><li><a href="https://huggingface.co/damerajee/Llamoe-test">damerajee/Llamoe-test · Hugging Face</a>：未找到描述</li><li><a href="https://openhands.ai4bharat.org/en/latest/instructions/datasets.html#supported-datasets">ISLR 数据集 &mdash; 👐OpenHands 文档</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1768991010938404879">来自 Unsloth AI (@UnslothAI) 的推文</a>：Unsloth 本周在 GitHub 上非常热门！🙌🦥 感谢大家以及所有 ⭐️Stargazers 的支持！查看我们的仓库：http://github.com/unslothai/unsloth</li><li><a href="https://sozercan.github.io/aikit/">简介 | AIKit</a>：AIKit 是一个一站式商店，可以快速开始托管、部署、构建和微调大语言模型（LLMs）。</li><li><a href="https://huggingface.co/papers/2402.18668#65f0f5f8de069cd5c55f1dd2">论文页面 - 简单的线性注意力语言模型平衡了召回率与吞吐量</a>

<li><a href="https://huggingface.co/Qwen/Qwen1.5-72B">tradeoff</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-72B">Qwen/Qwen1.5-72B · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2310.17680">CodeFusion: 用于代码生成的预训练扩散模型</a>: 想象一下，如果一个开发者只能修改最后一行代码，那么在函数正确之前，他们需要从头开始编写多少次？用于代码生成的自回归模型从...</li><li><a href="https://www.youtube.com/watch?v=jvqFAi7vkBc">Sam Altman: OpenAI, GPT-5, Sora, 董事会风波, Elon Musk, Ilya, 权力与 AGI | Lex Fridman Podcast #419</a>: Sam Altman 是 OpenAI 的 CEO，该公司开发了 GPT-4, ChatGPT, Sora 以及许多其他最先进的 AI 技术。请通过查看...来支持本播客</li><li><a href="https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2">如何微调 LLM 第一部分：准备指令微调数据集</a>: 学习如何在指令数据集上微调 LLM！我们将介绍如何格式化数据并训练像 Llama2, Mistral 等模型。这是（几乎）纯 PyTorch 的最小示例。</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py">transformers/src/transformers/models/mixtral/modeling_mixtral.py at main · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的最先进机器学习框架。 - huggingface/transformers</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral 微调入门（支持 16k, 32k, 128k+ 上下文）</a>: 在我们最新的教程视频中，探索如何使用自己的数据轻松微调语言模型 (LLMs)。我们深入探讨了一种具有成本效益且...</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok 开源发布</a>: Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#install-galore-optimizer">GitHub - jiaweizzhao/GaLore</a>: 通过在 GitHub 上创建账号来为 jiaweizzhao/GaLore 的开发做出贡献。</li><li><a href="https://github.com/AI4Bharat/OpenHands">GitHub - AI4Bharat/OpenHands: 👐OpenHands : 让手语识别普及化。 | **注意：** 不再积极维护。如果您有兴趣接管并推进此项目，请提交 issue</a>: 👐OpenHands : 让手语识别普及化。 | **注意：** 不再积极维护。如果您有兴趣接管并推进此项目，请提交 issue - AI4Bharat/OpenHands</li><li><a href="https://github.com/mistralai/mistral-src">GitHub - mistralai/mistral-src: Mistral AI 7B v0.1 模型的参考实现。</a>: Mistral AI 7B v0.1 模型的参考实现。 - mistralai/mistral-src</li><li><a href="https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned">teknium/GPT4-LLM-Cleaned · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/xai-org/grok-1/issues/6#issuecomment-2002664859">安装 requirements 时出错 · Issue #6 · xai-org/grok-1</a>: 我已经安装了 python 3.10 和 venv。尝试执行 "pip install -r requirements.txt" 错误：忽略了以下需要不同 python 版本的版本：1.6.2 需要 Python >=3...</li><li><a href="https://the-decoder.com/falcon-180b-open-source-language-model-outperforms-gpt-3-5-and-llama-2/">Falcon 180B 开源语言模型性能超越 GPT-3.5 和 Llama 2</a>: 开源语言模型 FalconLM 提供了比 Meta 的 LLaMA 更好的性能，并且可以用于商业用途。如果收入超过 100 万美元，商业使用需支付版税。</li><li><a href="https://github.com/unslothai/unsloth/pull/97">实现 Phi-2 支持的暂存 PR。由 cm2435 提交 · Pull Request #97 · unslothai/unsloth</a>: ….org/main/getting-started/tutorials/05-layer-norm.html]</li><li><a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: 添加 GaLore 优化器，由 younesbelkada 提交 · Pull Request #29588 · huggingface/transformers</a>: 这个 PR 做了什么？如标题所示，添加了来自 https://github.com/jiaweizzhao/GaLore 的 GaLore 优化器。修复了：#29512 这是我目前测试 API 的方式：import torch import datasets from ...</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1218580567453470860)** (1 条消息):

- **Unsloth AI 星数上升**：**Unsloth AI** 仓库本周在 GitHub 上表现活跃，感谢社区和点星者的支持。团队鼓励更多人在 [GitHub](https://github.com/unslothai/unsloth) 上为该仓库点星，这有助于推广更快、更节省显存的 QLoRA & LoRA 微调。

**提到的链接**：<a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>：速度提升 2-5 倍，显存减少 70% 的 QLoRA & LoRA 微调 - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1218112720994308122)** (25 messages🔥): 

- **巧合沙皇**：一位用户讨论了想到某事后不久便遇到它的现象——这是一种关于用户名和浏览体验的奇妙同步性。
- **运动中的诗篇**：Abhiabhi 分享了一篇名为《向猴子呼吁》（An Appeal to A Monkey）的诗作片段，思考了与人类创造的系统相比，生命原始而生动的本质。
- **微调性能对比**：Goutham_city 在尝试过 **Mistral-7b** 后，就针对特定领域分类任务是否应使用 **Gemma 7b** 寻求建议。Starsupernova 提到 Unsloth 已修复所有 bug，并建议 Gemma 与 Mistral 的对比结果各异。
- **寻找 Mixtral 分支**：Dogsofwarren 被引导至 GitHub 上一个他们正在寻找的、难以找到的 "Mixtral 分支" 的 PR。提供的链接指向了 [tohrnii 的 Unsloth AI 仓库分支](https://github.com/unslothai/unsloth/pull/145)。
- **在地图上分享 Pokemon RL Agent & 开源 UI 元素**：Iron_bound 展示了在单个地图上分享的人们在 *Pokemon RL* 环境中训练的可视化效果，而 Yahir9023 分享了 [Uiverse.io](https://uiverse.io/elements)，这是一个展示使用 CSS 或 Tailwind 制作的开源 UI 元素的网站。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pwhiddy.github.io/pokerl-map-viz/">Pokemon Red Map RL Visualizer</a>: no description found</li><li><a href="https://uiverse.io/elements">4203 UI elements: CSS &amp; Tailwind</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/145">[WIP] add support for mixtral by tohrnii · Pull Request #145 · unslothai/unsloth</a>: Mixtral WIP
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1218104575022727230)** (568 messages🔥🔥🔥): 

- **模型转换与训练的苦恼**：用户尝试使用 Colab 微调 **Unsloth 的 Mistral 7b** 等模型，并遇到了各种问题，如超出 token 限制以及在保存到 **HuggingFace** 等平台时出错。针对此类错误，一个建议是手动运行 GGUF 转换命令。

- **寻求量化方面的澄清**：关于 **LoRA** 和 **QLoRA** 之间的区别存在困惑，主要在于 `load_in_4bit = True` 是否意味着 QLoRA，以及每种类型的模型名称究竟应该是什么。**MRdragonfox** 澄清说 `4bit` 确实表示 QLoRA。

- **排查数据集和模板问题**：用户面临聊天机器人模型在给定提示之外生成多余内容的问题。一种假设是，这可能是由于聊天模板中 EOS token（序列结束标记）的应用不当或缺失造成的。

- **Full Fine Tuning (FFT) 的未来计划**：人们对 **Unsloth** 未来是否可能在 **LoRA** 和 **QLoRA** 之外支持 Full Fine Tuning 感兴趣。目前，Unsloth 专注于 LoRA 和 QLoRA，尚不支持 FFT，但团队对未来的可能性持开放态度。

- **部署困境**：一些用户询问关于使用 Unsloth 部署 **OpenAI 的 GPT-4** 的问题，这是不支持的。Unsloth 主要促进 **Mistral, Llama, Gemma** 等 LLM 的微调和量化，并为这些特定模型提供了详细的说明和 notebook。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1X_PHYBawrsCgKfMEPxvIDX__rYa1-v97?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit">ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit">unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16">主页</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/artidoro/qlora/blob/main/qlora.py#L746">artidoro/qlora 项目 main 分支下的 qlora/qlora.py</a>: QLoRA: 量化 LLM 的高效微调。通过在 GitHub 上创建账号来为 artidoro/qlora 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">主页</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">主页</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=FqfebeAdT073,">Google Colaboratory</a>: 未找到描述</li><li><a href="https://pastebin.com/ybSeKHhU">Unsloth: 将 4bit 和 LoRA 权重合并为 16bit...Unsloth: 将使用高达 5.34 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本并设置有效期的网站。</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth">DPO Trainer</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: 一个高吞吐量且显存高效的 LLM 推理和服务引擎</a>: 一个高吞吐量且显存高效的 LLM 推理和服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">生成 - GPT4All 文档</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/issues/1041">DPOTrainer 的损失函数会屏蔽提示词（prompts）吗？· Issue #1041 · huggingface/trl</a>: 嗨，有个小问题，DataCollatorForCompletionOnlyLM 会通过屏蔽提示词的损失来仅对回答进行训练。DPOTrainer (DPODataCollatorWithPadding) 也是这样工作的吗？看起来...</li><li><a href="https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#train-on-completions-only).">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/2/files">HuggingFaceH4/zephyr-7b-alpha · 添加聊天模板</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha#intended-uses--limitations">HuggingFaceH4/zephyr-7b-alpha · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L56">unslothai/unsloth 项目 main 分支下的 unsloth/unsloth/chat_templates.py</a>: 速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments">Trainer</a>: 未找到描述</li><li><a href="https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205">在 MT-Bench 上复现 LoRA 模型结果 · Issue #45 · huggingface/alignment-handbook</a>: 最近，我尝试在自己的数据集上拟合 DPO。最初，我尝试复现你的结果...</li>

你的 LORA 模型（MT-Bench 评分为 7.43）。然而，我遇到了一些问题。尽管使用了你所有的...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>: llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1218239216975351928)** (21 messages🔥): 

- **训练困境：更多的 Epochs，更多的知识？**：关于使用 800,000 行数据训练模型的讨论探索了增加 Epochs 可能带来更好训练效果的观点。然而，有人指出，虽然更多的 Epochs *可能* 对语音神经网络更好，但对于像 **Tiny Mistral** 这样的 LLM，这可能会导致模型遗忘其预有知识。

- **寻找知识平衡点**：为了最大限度地保留知识，有人建议删除多余数据，并认识到仅拥有大型数据集（如 300 万行）可能对微调（finetuning）模型以提高性能并无益处。

- **讨论模型集成与配置**：展示了一个配置了 Axolotl 徽章的 Tiny Mistral 模型，并提供了详细的设置和数据集处理过程。相关的配置和所用数据集可以在 [Hugging Face 上的 Tiny Mistral 模型](https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1) 找到。

- **探索模型参数的影响**：关于如何从大型数据集中获得最佳结果的对话包括了对理想 rank 和 alpha 值的建议，指出 rank 为 32 或 64 以及 alpha 值（rank * 2）可能适合 800k 行的数据集。

- **对模型集成尝试的回应**：分享 Tiny Mistral 模型以寻求集成到 Unsloth Repo 的尝试引起了不同的反应，其中一个回应表示很感兴趣，而另一个则表示结果不尽如人意。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1">Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/M4-ai/TinyMistral-6x248M-Instruct/tree/main">M4-ai/TinyMistral-6x248M-Instruct at main</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1218098224586293319)** (301 messages🔥🔥): 

- **寻求本地运行的 LLM 建议**：用户建议使用 **CodeLlama** 或 **DeepSeek** 进行编程，并提到在 18GB 内存的 MBP 上模型限制在 13b。对于在 M3 Pro 上运行，推荐包括 **qwen** 以及在可用内存限制内的模型。
- **探索本地换脸模型**：一位用户询问了类似于 **Reface** 的视频本地换脸模型，有人提到换脸更多是 Stable Diffusion 的任务而非语言模型，并建议将 **facefusion** 作为替代方案。
- **0.2.17 版本的特性反馈**：用户讨论了即将在不同操作系统上推出的 **LM Studio 0.2.17 版本** 的早期访问预览，初步反馈积极。
- **GPU 利用与配置相当棘手**：围绕配置 **LM Studio** 在存在多个 GPU 时利用特定 GPU 进行了技术讨论，并分享了使用 tensor split 配置的成功案例。还讨论了与旧款 **Tesla 卡**（如 K40）的兼容性以及在其上运行现代 LLM 的可能性。
- **开源模型 Grok 发布**：讨论集中在拥有 314B 参数的开源模型 **Grok**，以及由于运行需要极高资源而导致其与 LM Studio 不兼容的问题。用户很感兴趣，但也认识到本地使用的局限性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 是一个 314B 参数的 Mixture of Experts 模型 - 基础模型（未微调）- 8 个专家（2 个激活）- 86B 激活参数 - Apache 2.0 许可证 - 代码： - 祝编码愉快！另：我们正在招聘：</li><li><a href="https://tenor.com/view/ratha-gif-26742750">Ratha GIF - Ratha - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1小时演讲] Large Language Models 简介</a>：这是一个面向普通观众的 1 小时 Large Language Models 介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://huggingface.co/xai-org/grok-1/discussions/30">xai-org/grok-1 · 314B 参数有 297G 文件大小？</a>：未找到描述</li><li><a href="https://github.com/continuedev/continue/issues/713"">Issues · continuedev/continue</a>：⏩ 使用任何 LLM 进行编码的最简单方法——Continue 是适用于 VS Code 和 JetBrains 的开源自动驾驶工具 - Issues · continuedev/continue</li><li><a href="https://www.youtube.com/watch?v=lCZRwrRvrWg&">Mistral：在自定义数据上进行微调的最简单方法</a>：本视频由 Gradient.ai 赞助，请在此处查看：https://gradient.1stcollab.com/engineerprompt 在本视频中，我们将学习如何微调 Mistr...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1218119135423234058)** (138 条消息🔥🔥): 

- **Command-R 模型等待 LM Studio 更新**：在 GitHub 上集成 [Pull Request #6033](https://github.com/ggerganov/llama.cpp/pull/6033) 后，人们对 LM Studio 即将发布的支持 CohereForAI 的 Command-R 模型的版本充满期待。

- **硬件限制决定模型性能**：成员们正在应对其硬件配置的限制，一些拥有 "RTX 2070 Super" 或 "1660 Super" 等 GPU 的用户正在寻找最适合其系统的模型。

- **Grok-1 模型引发兴趣但超出典型资源范围**：关于 xAI 发布的新 Grok-1 模型的讨论涉及其巨大的体积，质疑在典型个人电脑上运行它的实用性，并考虑替代的托管策略。

- **更新与推理**：成员们对 LM Studio 中 Grok 等新模型的可用性感到好奇，而另一些人则表示谨慎，指出像 Grok 这样的模型对资源要求很高，或者基础模型可能需要额外的训练才能发挥其潜力。

- **Yi-200k 与模板问题**：关于 Yi-200k 最佳模板的讨论正在进行中，社区分享了见解和资源，包括指向 [Hugging Face](https://huggingface.co/01-ai/Yi-9B-200K) 的链接，以更好地利用这一特定模型。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39737281">未找到标题</a>：未找到描述</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.09611">MM1：来自多模态 LLM 预训练的方法、分析与见解</a>：在这项工作中，我们讨论了构建高性能的多模态大语言模型 (MLLMs)。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细和...</li><li><a href="https://huggingface.co/01-ai/Yi-34B/discussions/23">01-ai/Yi-34B · Prompt 模板？</a>：未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://youtu.be/PAbZRGGYNyM?si=xVNZCYUddDvoFUly">什么是 Large Language Model 中的参数？</a>：什么是 Large Language Model 中的参数？00:26 💡 像 GPT-3 这样的 Large Language Models 中的参数是在训练期间学习的变量，用于最小化...</li><li><a href="https://youtu.be/zjkBMFhNj_g?si=Rn96V9CMqEHLy6-7">[1小时演讲] Large Language Models 简介</a>：这是一个面向普通观众的 1 小时 Large Language Models 介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6033">由 acanis 添加 Command-R 模型 · Pull Request #6033 · ggerganov/llama.cpp</a>：关于 Command-R 35B 模型（128k 上下文）的信息可以在以下网址找到：https://huggingface.co/CohereForAI/c4ai-command-r-v01 基于 llama2 模型，并进行了一些更改：新的超参数...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1218213037060657273)** (12 条消息🔥):

- **对 llama.cpp 支持的困惑**：一名成员认为列出的与 llama.cpp 兼容的 **GGUF 文件** 表示支持 **Cohere 的 Command-R 模型**，但随后被纠正，指出 llama.cpp **目前不支持 c4ai**。
- **关于 GGUF 兼容性的澄清**：尽管出现了兼容文件，但已澄清 **Command-R 35B v1.0 的 GGUF 文件** 需要 2024 年 3 月 16 日之后的 llama.cpp，从版本 [b2440](https://github.com/ggerganov/llama.cpp/releases/tag/b2440) 开始。由于 Hugging Face 的 50GB 限制，文件被拆分，需要进行合并。
- **模型支持讨论中的错误确认**：一名用户承认了他们在文件兼容性讨论中的错误，因为 models-discussion 频道之前已有提及，但被他们忽略了。
- **AMD Linux 用户的 GPU 使用说明**：一名成员请求在 Linux 版本下载页面添加说明，指出 AMD 用户需要 **OpenCL 驱动程序** 才能在该程序中使用 GPU。
- **LM Studio 中的插件和文档交互查询**：有人询问 **LM Studio** 是否支持与自己的文档聊天或添加类似 autogen 的插件，回复指出可以开启 server mode 来连接 LM Studio 已经支持的插件。

**提到的链接**：<a href="https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF">andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face</a>：未找到描述

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1218129474348912711)** (480 条消息🔥🔥🔥): 

- **K80 购买和散热计划**：一名成员考虑从 eBay 以 **150 欧元** 购买一块 Tesla **K80 GPU**，并计划使用来自 Thingiverse 的 3D 打印导流罩来改进其散热。 
- **服务器和 GPU 配置讨论**：讨论了不同的 **server** 设置，以及 **Threadripper** 或 **EPYC CPU** 是否更适合用于 LLM 推理的 **多 GPU 系统** 的 **4 插槽主板**。
- **Amazon 的转售市场**：成员们谈到了 **Amazon** 如何成为转售商的市场，这些转售商通常会加价出售最初在 **eBay** 或 **AliExpress** 上架的产品。
- **Tesla K80 用于 LLM 的显存带宽**：讨论了 Tesla **K80 GPU** 用于 LLM 任务的情况，考虑到其显存带宽较慢，有人建议 **RX480 GPU** 可能会因为拥有更高的显存带宽而表现更好。
- **EPYC 系统构建讨论**：围绕构建用于 LLM 的 **DIY EPYC 系统** 进行了讨论，重点是确保足够的 PCIe 通道，并考虑性价比高的组件，如 **EPYC 7232P CPU**。成员们辩论了某些 CPU 和 GPU（如 **NVIDIA H100**）的价格相对于 **二手 K80 GPU** 是否合理。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.amazon.com/AMD-3200MHZ-SYSTEM-COMPONENTS-PROCESSORS/dp/B07XP9S55C/ref=sr_1_2">未找到标题</a>: 未找到描述</li><li><a href="https://lmstudio.ai/#can-i-use-lm-studio-at-work?">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并试用本地 LLM</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本发布</a>: 未找到描述</li><li><a href="https://www.amazon.de/-/en/HHCJ6-NVIDIA-Server-Accelerator-Renewed/dp/B07GJ45V3D/ref=sr_1_2?crid=1O8IZM1RV0TIH&dib=eyJ2IjoiMSJ9.B2ZUEDxvj_Z73GUX0GJebEDmX0cqUrowZhMOgYhwtCaPdx9UH8NiM39aqowgVAc5YENjqRh8_cc1qHbgwPJMprvhMhnuusRAJuQqLmWDyskupHMP8ACQI354KZZjKYrdtnPPNGnuoJdVlHxoPQ8ll9ilsDZZ334_L6TwueHlrTelgoIjaTt650I3FQyWgOFmpTvAb3YigqPDURnBJMq1D6wanBHjVSaSdFOEnWlP2cUV8J9Hq4Lh_0bJbRh-kAaca58OndCeXm-tGVmNFLi7TuMKGZORpZ0Q6IcMd6Vz11w.MFnlYLfXX9YWUon0J_Dg0ds2eKFM6AwZgazWMdxeEjE&dib_tag=se&keywords=Tesla+K80&qid=1710787582&s=computers&sprefix=tesla+k80%2Ccomputers%2C421&sr=1-2">未找到标题</a>: 未找到描述</li><li><a href="https://coral.ai/products/m2-accelerator-dual-edgetpu#description">M.2 Accelerator with Dual Edge TPU | Coral</a>: 使用 M.2 (E key) 接口将两个 Edge TPU 集成到现有系统和新系统中。</li><li><a href="https://www.aliexpress.com/item/100500634581">404 页面</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/273788651049?">Dell T710 塔式服务器 双 6 核 X5650 **144Gb RAM** 240gb SSD + 6X 600G SFF SAS | eBay</a>: 未找到描述</li><li><a href="https://www.newegg.com/asrock-rack-romed8-2t/p/N82E16813140044">Asrock Rack ROMED8-2T ATX 服务器主板 AMD EPYC 7003 (支持 AMD 3D V-Cache 技术)/7002 系列处理器 SP3 (LGA 4094) 双 10GbE - Newegg.com</a>: 购买 Asrock Rack ROMED8-2T 服务器主板 AMD EPYC 7003 (支持 AMD 3D V-Cache 技术)/7002 系列处理器 SP3 (LGA 4094) 双 10GbE，享受快速发货和顶级客户服务。一旦您...</li><li><a href="https://www.aliexpress.com/item/1005006525215524.html">未找到标题</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/125947603377?itmmeta=01HS9HRSJMXBV00M1XW59H5NAE&hash=item1d530fe9b1:g:fHQAAOSwWVxkbefZ&itmprp=enc%3AAQAJAAAA4A6tXSRz7NxXocQqxCeo%2F2TdOTiIP1AMtfRCBxeBISSicEa3bP%2FtSfa9CmVAH74vTwUFyfwFd1VhNC71wMalgSqfYNDwr7svQreF5j3Gqk4Brm8Zn7hMHU6mRQVuxRyyv5VyA1PeZKdylhbJH0O%2BC2IM8GdP7yLRbRw6sOGTb2KMO0V0m%2B7aGkzXe6h33qOgF16cjz2vh2TITEEOr1eYGfz7ViQZ846gljR8VFArZiDwxgIU8naY8yQRPUJe4Znn3GYEn3GT3DNHxdg5zoB7qyMOytwL9TKozBLIkBQVtyyq%7Ctkp%3ABk9SR8KZ47HKYw">全新 /Wave ®AI 服务器 NF5688M6 NVIDIA HGX TESLA A800 80G 八路 GPU 服务器/期货 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.ca/itm/126375063761">AMD EPYC 7232P 8 核 3.1GHz 32MB L3 处理器 - Socket SP3 - 100-000000081 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/115960685949?">AMD EPYC 7F72 CPU 处理器 24 核 3.20GHz 192MB 缓存 240W - 100-000000141 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/126352871326?epid=11041255665&itmmeta=01HS9333CQ68S4STA8BZJ3V0BH&hash=item1d6b37cf9e:g:DOEAAOSweRlkuVOG&itmprp=enc%3AAQAJAAAA0GtLL6BuVwKKMH1iyVWS1kdp6p0LvQb%2Fcu8c94aisQZDISgf4yKcfrjNbigVkO4IGdfBt3tcIr6du3Nb1xXGbEe2CNScd%2B4RoCdoEx%2BQMPtNGs0TtY3wzAbszVam1AHN8tC%2Bzq%2BVoVhSwCmdZ77779duZUVHF%2Fq1ckL28OWoVp%2FRStC3u0NyyTZtUke6tEsgNdQYOKI4%2BqNOIN11tc8XuhOtaovFo6WzH87nIC6BUNiaWYnvWcqUPH3NUs6Gxi%2FWnel1Vj9wokxL8oELjbCFBOA%3D%7Ctkp%3ABFBMyLaMo8pj">AMD EPYC 7232P CPU 处理器 8 核 3.10GHz 32MB 缓存 120W - 100-000000081 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/296113403496?">Dell T710 塔式服务器 双 6 核 X5670 **24 核** 64GB RAM | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d64a6377:g:kacAAOSw~q1lFEwb&itmprp=enc%3AAQAJAAAA4GTzwRZBHO82ltgqug5ARkRZ5JKlaikKECFytG5%2FNjvBMzyE2UGOBW0yRbeW%2B%2F3prx2LD9sPaLsinW103607IHMVVMe2tg6FIa2KVc%2FUVWqCGgQPrRRS97i9Q%2FZW0nnLz5XSLuFob%2FicmlhLi7Ve68FV47SLRenj5tDoUD8mwpvdoxA5uQtR0DNACYnvlVQe4BeXKFAWKA8iKA6WdrVikWOsQcODTpcW916%2FL8jFOUSFjg9D5%2FP1xg4foswYBWrIeaD4Pm9rguigAFQvYGqHFLKNXgB4CjCD0BczHhSZYunI%7Ctkp%3ABk9SR8i8z63KYw">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU 加速器 12 个月保修 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d6">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU 加速器 12 个月保修 | eBay</a>: 未找到描述</li><li><a href="https://www.thingiverse.com/search?q=K80+cooling+&page=1&type=things&sort=relevant">搜索 Thingiverse - Thingiverse</a>: 下载文件并使用您的 3D 打印机、激光切割机或 CNC 进行制造。</li><li><a href="https://www.techpowerup.com/cpu-specs/core-i5-3470.c1039#:~:text=Programs%20using%20Advanced%20Vector%20Extensions,performance%20for%20calculation%2Dheavy%20appli">使用 Advanced Vector Extensions 的程序，针对计算密集型应用的性能...</a></li>

cations.">Intel Core i5-3470 Specs</a>: Ivy Bridge, 4 核心, 4 线程, 3.2 GHz, 77 W</li><li><a href="https://www.microcenter.com/product/677156/nvidia-geforce-rtx-3090-founders-edition-dual-fan-24gb-gddr6x-pcie-40-graphics-card-(refurbished)">Micro Center - Computers and Electronics</a>: Micro Center - 计算机与电子产品 - 数千种可购买的产品：台式机、笔记本电脑、显示器、DIY PC 零件、升级、数字成像、打印耗材、便携式设备、音频设备...</li><li><a href="https://zifa666.aliexpress.com/store/5885523/pages/all-items.html?productGroupId=40000003590095&shop_sortType=bestmatch_sort">Luckim Official Store - Amazing products with exclusive discounts on AliExpress</a>: 未找到描述</li><li><a href="https://www.aliexpress.com/item/1005006345813657.html">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1219065221327355974)** (4 条消息): 

- **寻求模型预设指南**：一位用户询问是否有**针对不同模型的预设 (presets)**。他们收到了一个指向 GitHub 仓库的链接，其中包含 JSON 配置文件以及位于 [lmstudio-ai/configs](https://github.com/lmstudio-ai/configs) 的示例配置集合。
- **ROCm 用户召集**：一位寻找其他 **ROCm** 用户的成员被引导至特定频道 (**#1195858490338594866**)，以便寻找相关社区并进行交流。

**提到的链接**：<a href="https://github.com/lmstudio-ai/configs">GitHub - lmstudio-ai/configs: LM Studio JSON configuration file format and a collection of example config files.</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1219051718172606537)** (1 条消息): 

- **咨询本地推理服务器能力**：一位成员询问是否可以在 **Local Inference Server**（本地推理服务器）上运行具有 **JSON function calling**（函数调用）功能的模型。目前尚无其他成员提供见解或分享关于此功能的经验。
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1219383598193311744)** (5 条消息): 

- **关于 AVX Beta 的澄清**：一位成员询问该应用的 Beta 版本是否仅使用了 AVX。另一位成员确认了这一点，并补充说这也是一个较旧的版本，并表示 **“AVX 支持确实不是一个高优先级的项目。”**
- **确保与旧模型的兼容性**：确认虽然模型可以在 Beta 应用中运行，但**像 starcoder2、gemma 等较新的模型**将不被支持。
- **Mistral 模型兼容性查询**：一位成员询问并寻求确认 **Mistral** 是否可以在该应用上运行，这暗示了该版本至少与某些成熟模型兼容。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1218206050495234070)** (5 条消息): 

- **分享 GitHub 资源**：分享了一个指向 [针对 gfx1031 和 gfx1032 预构建的 Windows ROCm 库](https://github.com/brknsoul/ROCmLibs) 的链接，为希望在特定 AMD GPU 上使用 **Windows ROCm** 库的用户提供资源。

- **期待 LM Studio 的多 GPU 支持**：一位成员表示有兴趣在 LM Studio 中使用**多个 AMD GPU**，但注意到当前版本似乎仅依赖于主 GPU。他们询问了平台内支持多 GPU (Multi-GPU) 的时间表。

- **Radeon 6700 XT 的兼容性问题**：澄清了 **AMD Radeon 6700 XT** 并未获得 **AMD 官方的 ROCm 支持**，且 LM Studio 使用的是未经修改的 ROCm 库，这解释了为什么 6700 XT 可能无法在 ROCm 下正常工作。

- **对不同 AMD 型号多 GPU 使用的希望**：针对兼容性担忧，有人指出如果用户拥有另一个 **7000 系列**的 GPU，LM Studio 可能会并行利用它们。

- **目前 KoboldCPP-ROCm 已支持多 GPU**：尽管存在兼容性问题，但提到 **KoboldCPP-ROCm** 在当前状态下确实可以很好地支持多 GPU。

**提到的链接**：<a href="https://github.com/brknsoul/ROCmLibs">GitHub - brknsoul/ROCmLibs: Prebuild Windows ROCM Libs for gfx1031 and gfx1032</a>: 为 gfx1031 和 gfx1032 预构建的 Windows ROCM 库 - brknsoul/ROCmLibs

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1219265025487667200)** (1 条消息): 

- **寻找合适的 Agent**：一位成员询问了另一位成员在选择 Agent 系统方面的进展，表达了对该过程的共同兴趣。他们提到目标是通过不同的 Agent 来**深化并验证一个创意概念**。
  

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1218144997094723615)** (56 messages🔥🔥): 

- **NVIDIA 下一代 GDDR7 显存**：据传即将推出的 [NVIDIA GeForce RTX 50 系列 "Blackwell"](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed) 将采用速度为 28 Gbps 的 GDDR7 显存，尽管 GDDR7 芯片最高支持 32 Gbps，但其带宽仍比 GDDR6 高出 55%。
- **客户服务的烦恼**：一位成员对支持工单被上报表示乐观，并开玩笑说 "Greg" 可能会修复他们的 OAuth 登录错误。
- **模型进步 vs. Agent 相关改进**：成员们讨论了模型进步与 Agent 改进的可能性，推测 OpenAI 即将发布的内容可能更侧重于 Agent 相关功能，并提到预计会出现一个强大的 "Agent 控制接口"。
- **期待变革性的 Agent**：人们对具有高度可靠性的新型 AI Agent 充满期待，正如一场关于 AI 模型改进对高质量 Agent 界面必要性的讨论所指出的，讨论中还引用了 Sam Altman 关于 AI 能力不断增长的预测。
- **关于响应式 AI 助手的对话**：关于如何通过智能停止和恢复输出来增强 AI 助手交互性的讨论正在进行中，建议方案从编辑对话历史到简单的音频暂停-恢复逻辑不等，一些社区成员提供了帮助和示例。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/unkjdgames?s=21">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">使用 Langgraph 进行 Plan-and-Execute</a>：如何创建一个 &quot;plan-and-execute&quot;（计划与执行）风格的 Agent。这在很大程度上受到了 Plan-and-Solve 论文以及 Baby-AGI 项目的启发。核心思想是首先...</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50 系列 &quot;Blackwell&quot; 将使用 28 Gbps GDDR7 显存速度</a>：据可靠爆料者 kopite7kimi 称，首批采用 GDDR7 显存的 NVIDIA GeForce RTX 50 系列 &quot;Blackwell&quot; 显卡传闻将配备 28 Gbps 的显存速度...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1218108265854926899)** (16 messages🔥): 

- **"Horny Claudes" 能增强 AI 输出？**：据报道，一项聊天机器人实验发现 "Horny Claudes" 能生成更好的 Mermaid 图表。这一现象引发了讨论，一些人认为这种策略令人反感，并将其与 "逆向 Sydney" 进行类比。

- **深入模型实验**：一位成员分享了一个专注于 PyTorch 入门的个人研究项目，他们称这可能不是突破性的，但却是一次宝贵的学习经历。项目结果可在 [Derbydefi 的 Twitter](https://vxtwitter.com/derbydefi/status/1768767386419970071) 查看。

- **Apple 发布 AI 模型信息**：Apple 终于开始讨论他们的 AI 模型工作，一位成员分享了 [Aran Komatsuzaki 的推文链接](https://twitter.com/arankomatsuzaki/status/1768446729710371115)，引发了社区关于未发布权重（weights）的讨论。

- **ORPO：一种新的偏好对齐算法**：Hugging Face 上的一篇论文介绍了 ORPO，这是一种声称通过使用优势比（odds ratio）来简化语言模型监督微调（SFT）的新算法。[论文摘要](https://huggingface.co/papers/2403.07691)概述了该概念及其在无需额外阶段的情况下进行偏好对齐的前景。

- **重现 Self-Rewarding Language Model 研究**：Oxen.ai 社区正致力于重现 MetaAI 的 Self-Rewarding Language Model 论文。进度和代码记录在他们的 [GitHub 仓库](https://github.com/Oxen-AI/Self-Rewarding-Language-Models)中。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/repligate/status/1768521441329434937?s=20">来自 j⧉nus (@repligate) 的推文</a>：@xlr8harder 我没让它发展太远，但现在房间里有人跟我说，他们创建了一个“好色的 Claude”网络，以及这些 Claude 如何创造更好的...</li><li><a href="https://arxiv.org/abs/2402.16823">Language Agents as Optimizable Graphs</a>：为了改进基于 Large Language Models (LLMs) 的问题求解器，人们提出了各种人工设计的 prompt engineering 技术，导致了许多互不兼容的代码库。我们将这些方法统一起来...</li><li><a href="https://fxtwitter.com/burny_tech/status/1769530798242255129">来自 Burny — Effective Omni (@burny_tech) 的推文</a>：关于马斯克可能通过 Grok 引领开源，从而动摇情报战争中其他巨头玩家的看法。Grok 1 是一个拥有 314B 参数的模型，采用 mixture of experts 架构...</li><li><a href="https://huggingface.co/papers/2403.07691">论文页面 - ORPO: Monolithic Preference Optimization without Reference Model</a>：未找到描述</li><li><a href="https://github.com/Oxen-AI/Self-Rewarding-Language-Models">GitHub - Oxen-AI/Self-Rewarding-Language-Models：这是由 Oxen.ai 社区完成的工作，旨在复现来自 MetaAI 的 Self-Rewarding Language Model 论文。</a>：这是由 Oxen.ai 社区完成的工作，旨在复现来自 MetaAI 的 Self-Rewarding Language Model 论文。 - Oxen-AI/Self-Rewarding-Language-Models
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1218105907615895562)** (656 条消息🔥🔥🔥): 

- **Yi 许可证困惑**：关于 Yi-9B 模型商业用途的讨论引发了猜测，用户不确定自动邮件批准是营销手段还是真正的商业应用开放。

- **Grok-1 发布引发辩论**：拥有 314B 参数的大型 mixture-of-experts 模型 Grok-1 引起了 AI 社区的反响。尽管其规模巨大，一些人对其性能表示怀疑，认为其不如 Mistral-7B 等更小、更高效的模型。

- **持续预训练与 MoEs**：围绕 MoEs (Mixture-of-Experts) 持续预训练的对话展开。用户分享了经验，并提到虽然这有助于资源较少的语言，但对于 MoEs 来说，这一过程尚属未知领域，如果没有高质量数据，大型模型可能无法从中受益。

- **对齐与“觉醒文化”讨论**：社区辩论了 AI 中的“觉醒文化 (wokeness)”，起因是担心模型过于政治正确或根据公司理念修改 prompt。讨论演变为关于 AI alignment、steering vectors 的使用，以及 AI 模型反映宪法原则的想法。

- **GTC 主题演讲期间 GPT-4 架构泄露**：NVIDIA 首席执行官黄仁勋 (Jensen Huang) 在 GTC 主题演讲中可能无意中证实了关于 GPT-4 拥有 1.8 万亿参数及其 MoE 架构的传闻，这让 AI 社区感到惊讶，并开始推测该模型的能力和成本。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=TOasxww3M5DjlB4iBWa_ig">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！ ↘️ 引用...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768942321129697790?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：Hermes 2 Pro 函数调用模型已与 @ExaAILabs 的搜索引擎集成👀 ↘️ 引用 Barton Rhodes 🦺 (@bmorphism) 增加了对 @ExaAILabs 的支持，以便与 @NousResearch 的新函数调用模型配合使用...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能多模态大语言模型（MLLMs）的方法。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细且...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型（LLMs）的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>：大语言模型（LLMs）通常在数十亿个 tokens 上进行预训练，而一旦有新数据可用，该过程就会重新开始。一个更有效的解决方案是持续预...</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>：未找到描述</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768948484479049897?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：<code>&lt;cmd&gt; run world_sim.exe --epoch &#34;Earth in 2500&#34; --civilization_type &#34;Type-II on Kardashev scale&#34; &lt;/cmd&gt;</code> ↘️ 引用 mephisto (@karan4d) 我当然会开源 worldsim...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>：我们探讨了在不平衡、以英语为主的语料库上训练的多语言语言模型是否将英语作为内部枢纽语言——这是一个对于理解语言模型如何...</li><li><a href="https://huggingface.co/Replete-AI/Mistral-11b-v0.1">Replete-AI/Mistral-Evolved-11b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered">anon8231489123/ShareGPT_Vicuna_unfiltered · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769773746896662873?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：@Cyndesama Claude 3 Opus 使用 python42 运行 AI 小镇模拟</li><li><a href="https://huggingface.co/datas">datas (shu nakamura)</a>：未找到描述</li><li><a href="https://x.com/whyarethis/status/1769269824587542692?s=46">来自 Parzival - 🌞/⏫ (@whyarethis) 的推文</a>：现在我们正步入正轨。</li><li><a href="https://x.com/itsandrewgao/status/1769460684956602527?s=46">来自 Andrew Kean Gao (@itsandrewgao) 的推文</a>：我觉得 Grok-4bit 对一块 H100 GPU 来说还是稍微太大了 :( ↘️ 引用 Andrew Kean Gao (@itsandrewgao) 天哪 @grok 有 3140 亿参数，Mixture of 8 Experts，未经 RLHF/道德化处理，这太...</li><li><a href="https://x.com/burkov/status/1769496949252673550?s=46&t=TOasxww3M5DjlB4iBWa_ig">来自 Andriy Burkov (@burkov) 的推文</a>：我们还有待观察 Grok 与 GPT-4 相比有多出色，但可以肯定的是，如果你今天要训练一个 OpenAI/Anthropic 的竞争对手，你不再需要从头开始了...</li><li><a href="https://huggingface.co/migtissera/Tess-70B-v1.6">migtissera/Tess-70B-v1.6 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/tree/main">openchat/openchat_sharegpt4_dataset at main</a>：未找到描述</li><li><a href="https://fxtwitter.com/lqiao/status/1768045066776707226?s=20">来自 Lin Qiao (@lqiao) 的推文</a>：我们很高兴能与 @NousResearch 合作开发 Hermes 2 Pro 多轮对话和函数调用模型。Hermes 在超过 1.5 万个函数调用和 500 个示例的函数调用 DPO 数据集上进行了微调...</li><li><a href="https://arxiv.org/abs/2303.11934">Sparse Distributed Memory is a Continual Learner</a>：持续学习是人工神经网络面临的一个问题，而它们的生物对应物则非常擅长解决。基于使用稀疏分布式存储（SDM）连接核心神经...</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/10/files">NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · 添加评估结果（Adding Evaluation Re...）

sults</a>: 未找到描述</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769424961192529962?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>: &lt;cmd&gt; sudo python3 akashic_records.py --entity [&#34;sam altman&#34;, &#34;elon musk&#34;] --mode &#34;email thread&#34; --topic &#34;superintelligence scenarios&#34; &lt;/cmd&gt;</li><li><a href="https://huggingface.co/01-ai/Yi-9B">01-ai/Yi-9B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/causality.ipynb">Abstractions/abstractions/goap/causality.ipynb (main 分支) · furlat/Abstractions</a>: 一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://www.hd-computing.com/">HD/VSA</a>:   </li><li><a href="https://arxiv.org/abs/2403.08540">语言模型通过过度训练和在下游任务上可靠地扩展</a>: Scaling laws 是开发语言模型的有用指南，但目前的扩展研究与语言模型最终的训练和评估方式之间仍存在差距。例如，scal...</li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">NVIDIA CEO 黄仁勋 GTC 2024 年 3 月主题演讲</a>: 观看 NVIDIA CEO 黄仁勋的 GTC 主题演讲，了解所有关于塑造我们未来的 AI 进展的公告。深入了解这些公告并发现...</li><li><a href="https://www.youtube.com/watch?v=t6SQj8YidGA">加速主义加速主义 (Acc/Acc)</a>: 加速主义加速主义是指当你加速加速主义，以便将加速主义应用于加速主义中那些过于激进的部分：https://www.patre...</li><li><a href="https://docs.pydantic.dev/latest/concepts/json_schema/">JSON Schema - Pydantic</a>: 未找到描述</li><li><a href="https://www.youtube.com/wa">Liam Johnson 击败起哄者 | 纽约脱口秀</a>: 上周末 Liam Johnson 决定终于在 Giggle Nerd 首次亮相。他在周日 23:00 到 23:25 进行了表演，我们的观众非常喜欢...</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/gridmap.ipynb">Abstractions/abstractions/goap/gridmap.ipynb (main 分支) · furlat/Abstractions</a>: 一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=oYFjDt4-hFw&ab_channel=NewEconomicThinking">Cosma Shalizi - 为什么经济学需要数据挖掘</a>: Cosma Shalizi 敦促经济学家停止他们正在做的事情：将大型复杂模型拟合到一小组高度相关的时间序列数据中。一旦你...</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md (main 分支) · furlat/Abstractions</a>: 一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">让我们构建 GPT Tokenizer</a>: Tokenizer 是 Large Language Models (LLMs) 中一个必要且普遍存在的组件，它在字符串和 tokens（文本块）之间进行转换。Tokenizer...</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/PrismarineJS/mineflayer">GitHub - PrismarineJS/mineflayer: 使用强大、稳定且高级的 JavaScript API 创建 Minecraft 机器人。</a>: 使用强大、稳定且高级的 JavaScript API 创建 Minecraft 机器人。 - PrismarineJS/mineflayer</li><li><a href="https://x.com/grok/status/1769441648910479423?s=46">来自 Grok (@grok) 的推文</a>: @elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1">基于深度强化学习的真实果蝇运动全身模拟</a>: 动物的身体决定了神经系统如何产生行为。因此，对感觉运动行为的神经控制进行详细建模需要一个详细的身体模型。在这里我们...</li><li><a href="https://hack.meetmeinshibuya.com/">HacksTokyo</a>: 东京 AI x 数字娱乐黑客松！</li><li><a href="https://github.com/Prismarin">Prismarin - 概览</a>: Prismarin 有 3 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1218205298729156648)** (25 条消息🔥):

- **Llama-2 的困惑度难题**：一位成员参考 [Kaggle notebook 指南](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) 计算 **Llama-2** 的困惑度（Perplexity），但最终结果为 90.3，正在寻求他人的经验以获取潜在解决方案。
- **雄心勃勃的扩增梦想被现实击碎**：成员们讨论了将 **Llama-2 13b** 扩增至 20b 以超越 **Mistral** 性能的可行性。据推测，为一个 20b 模型筹集 300,000 美元的资金似乎非常困难；另一位成员建议投入更经济的 30,000 美元进行持续预训练（Continued Pretraining）。
- **模型小型化成为新趋势**：一位成员分享了他们目前在通过持续预训练缩减模型规模方面的工作，并提供了其项目 *Smallstral* 的链接。这是一个经过层剪枝（Layer-pruned）的 Mistral 版本，并重点展示了与 [Mistral-7B](https://huggingface.co/AlexWortega/smallstral) 的性能对比指标。
- **对更大模型前提的探索**：讨论参与者对未来生产大型模型（如 70b 的 **genstruct 70b** 或 **openhermes 2.5**）进行了理论推演，考虑了财务和技术限制。
- **Open-hermes Grok 推测及持续 LoRA 预训练咨询**：提到了 xai-org 发布 **Grok** 的消息，并推测 **Open-hermes** 可能会采用 **Grok**。此外，还有关于尝试使用 **LoRA** 技术在特定领域数据上进行持续预训练的咨询。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook">Calculating the Perplexity of 4-bit Llama 2</a>：在 Kaggle Notebooks 中探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账号为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://huggingface.co/AlexWortega/smallstral">AlexWortega/smallstral · Hugging Face</a>：未找到描述</li><li><a href="https://wandb.ai/alexwortega/cpm_rus/runs/w5t4dsat?nw=nwuseralexwortega">alexwortega</a>：Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1218181932853104720)** (18 messages🔥): 

- **链接状态咨询**：一位用户询问特定链接是否失效，另一位用户确认该链接正常。
- **对某个未指明想法的反思**：用户 fullstack6209 表示对某个想法感到震撼并思考了数日，引发了对其模糊言论的简短询问。
- **Bittensor 链故障报告**：用户讨论了 Bittensor 链在过去 11 小时内的停机情况，对现状进行了调侃，并注意到修复工作的延迟。
- **关于 Subtensor 更新的信息征集**：有人提到 Bittensor 已恢复运行，但需要更新 subtensor，而并非所有人都已完成更新。
- **关于参与 Bittensor 的咨询**：一位用户寻求购买 TAO 代币进行注册的建议，得到的推荐是使用 MEXC 交易所。此外，还有关于硬件要求的对话，包括设置 qlora trainer 的建议以及针对不同模型的适当显存配置。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1218682432610373703)** (100 messages🔥🔥):

- **重新思考 RAG 属性**：团队讨论了 **RAG (retrieval-augmented generation) 流水线**中模型理想的属性：低延迟、上下文处理、知识广度、function calling，以及可能使用 markdown 和 HTML 风格引用的多样化输出结构。他们强调了通过规定响应风格和意图理解的模式，在提供的上下文召回（recall）与推理（reasoning）之间取得平衡。
- **Cohere 模型引起关注**：一位成员强调了 **Cohere** 的模型，该模型已经为 RAG 任务做好了准备，因为它可以处理 128K 的上下文，并提供 span outputs 和 inline citation 等功能，有可能推动 RAG 系统的发展。
- **寻求微妙的平衡**：对话转向了训练模型的想法，即要么完全依赖外部上下文，要么将外部知识与模型的内部知识混合。他们辩论了模型是否应该默认包含其内部知识，并且仅在明确指令下才对其进行限制。
- **RAG 模式的难题**：成员们考虑为 Hermes 引入 **"RAG mode"**，设想一个能够处理大文档上下文并具备多种功能的模型，从引用来源到结构化输出或编写代码，以匹配用户查询。
- **探索专门的 RAG 模型**：他们探索了小型、专门的 RAG 模型在管理 RAG 流水线复杂性方面的潜力，将其比作结合 Opus 和 Haiku 等模型的属性，以在单个文档处理工作流中多次调用模型时提高速度和效率。

**提到的链接**：<a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py">scratchTHOUGHTS/commanDUH.py at main · EveryOneIsGross/scratchTHOUGHTS</a>：用于避免 self 溢出错误的第二大脑临时记忆。- EveryOneIsGross/scratchTHOUGHTS

---

**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1218167767379742813)** (273 条消息🔥🔥): 

- **探索开放教育资源**：一位成员对免费提供的常春藤盟校课程表示兴奋，并指出观看来自 MIT 和 Stanford 等顶尖机构的讲座对他们的大学学习有所帮助。其他成员指出这种做法很普遍，从而使这一讨论更具背景意义。
- **计算机科学的创新与讨论**：讨论包括卡内基梅隆大学（Carnegie Mellon University）的一位 [教授网页](https://www.cs.cmu.edu/~dwoodruf/)，该网页令人印象深刻地追溯到近 7 年前，展示了在算法和 machine learning 等领域的贡献。
- **分享新颖的 AI 研究和项目**：分享了各种 AI 相关项目和论文的链接，例如一个提供通过矩阵乘法进行 AI 加速的 [GitHub 仓库](https://github.com/trevorpogue/algebraic-nnhw)，以及关于 Maisa 的 **Knowledge Processing Unit** ([Maisa KPU](https://maisa.ai/blog/kpu)) 的白皮书，这是一个旨在利用 Large Language Models (LLMs) 的力量来处理复杂任务的框架。
- **关于 LLMs 中 'ThoughtStream' Token 的辩论**：一位成员提出了在常规 Language Models (LLMs) 中添加 'ThoughtStream' token 的概念，允许在不影响 loss function 的情况下产生思维流，引发了关于 Quiet-STaR 和 Feedback Transformers 等论文中类似想法的讨论，这两者都旨在增强顺序任务中的推理能力。
- **Pause Token 的潜在益处与挑战**：围绕在模型中加入 pause tokens 的对话，旨在为 transformers 提供更多计算量以实现更好的推理，收集了关于实现挑战以及与 Universal Transformers 和 RNNs 比较的见解。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/maisaAI_/status/1768657114669429103?s=20">Maisa (@maisaAI_) 的推文</a>：介绍 Maisa KPU：AI 推理能力的下一次飞跃。知识处理单元（Knowledge Processing Unit）是一个针对 LLM 的推理系统，它利用了它们所有的推理能力并克服了其固有的...</li><li><a href="https://x.ai/blog/grok">宣布 Grok</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>：在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理对于...</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>：AI 驱动的知识处理平台。一个用于执行业务任务的简单 API。为软件和应用程序开发人员抽象化了使用最新 AI 架构的复杂性。</li><li><a href="https://arxiv.org/abs/2002.09402">Addressing Some Limitations of Transformers with Feedback Memory</a>：尽管 Transformer 是前馈网络，但已成功应用于序列化、自回归任务。与循环神经网络不同，Transformer 使用 Attention 来捕捉时间上的...</li><li><a href="https://arxiv.org/abs/2203.07852">Block-Recurrent Transformers</a>：我们介绍了 Block-Recurrent Transformer，它在序列中以循环方式应用 Transformer 层，并且在序列长度上具有线性复杂度。我们的循环单元...</li><li><a href="https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Most_edited_articles_last_month">Wikipedia:Database reports/Most edited articles last month - Wikipedia</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2312.12705">Optimizing Distributed Training on Frontier for Large Language Models</a>：大语言模型（LLM）作为基础模型已取得显著成功，通过微调使各种下游应用受益。最近关于 Loss Scaling 的研究表明...</li><li><a href="https://www.npr.org/sections/publiceditor/2009/08/19/112034424/free-transcripts-now-available-on-npr-org>">NPR.org 现提供免费转录文本</a>：NPR 上喜爱、错过或令人抓狂的故事转录文本以前每份售价 3.95 美元，但现在在 NPR.org 上免费提供。</li><li><a href="https://www.youtube.com/watch?v=Sq1QZB5baNw),">Figure 状态更新 - OpenAI 语音到语音推理</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/122123)">Issues · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - Issues · pytorch/pytorch</li><li><a href="https://aideadlin.es/?sub=ML,CG,NLP,RO,SP,DM,CV">AI 会议截止日期</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">EleutherAI/cookbook 项目 main 分支下的 cookbook/calc/calc_transformer_flops.py</a>：深度学习入门指南。包含处理真实模型所需的所有实践细节和实用工具。- EleutherAI/cookbook</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>：使用乘法次数减半的矩阵乘法进行 AI 加速 - trevorpogue/algebraic-nnhw</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账户为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/">RT-2: New model translates vision and language into action</a>：介绍 Robotic Transformer 2 (RT-2)，这是一种新型的视觉-语言-动作 (VLA) 模型，它从网络和机器人数据中学习，并将这些知识转化为通用的指令，用于...</li><li><a href="https://www.cs.cmu.edu/~dwoodruf/">David P. Woodruff</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1218100666493304852)** (245 messages🔥🔥): 

- **辩论模型性能**：关于 **Grok** 的能力存在激烈讨论，这是由 xAI 发布的一个拥有 **3140 亿参数的 Mixture-of-Experts 模型**。人们对其与声称的参数量相比的有效性表示怀疑，尽管已在 Git 上发布，但其实际应用价值仍存疑问，讨论强调了在独立评估和 Benchmarks 中的局限性。

- **Transformers 中的 Speculative Sampling**：对话转向了不同模型架构中 Speculative Sampling 的效率。有人建议，虽然正如 [PyTorch 博客文章](https://pytorch.org/blog/accelerating-generative-ai-2/) 中讨论的那样，Speculative Sampling 可能会为 Transformers 带来提升，但由于运行机制不同，其在 **Mamba** 等模型上的适用性尚不明确。

- **LLMs 与 Benchmarking**：目前对 LLMs 的 Benchmarking 系统存在广泛批评，质疑在 Best-of-N 方法下存在 Overfitting 问题，即结果可能仅仅源于统计方差。这种怀疑延伸到了 Benchmark 作为区分工具的有效性，特别是对于 GPT-4 和 Claude-3 等大型模型。

- **技术讨论背后**：聊天暗示了 LLMs 的各种技术层面，例如 PyTorch 中逐层梯度操作（layerwise gradient operations）的低效，以及在具有不同层或状态维度的模型之间进行 Speculative Decoding 的兼容性问题。讨论还涉及了潜在的策略，如 Label Smoothing 以及 MLP 与 Attention 层的 FLOPs 分解对比。

- **公司人员与产品关系**：关于公司人员是否能预测其产品质量存在哲学上的分歧，特别是在 AI 模型和初创公司的背景下。虽然有人认为优秀的团队逻辑上会带来卓越的结果，但其他人仍持怀疑态度，并举出了一些并非如此的例子，主张依靠实证证据而非依赖权威。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/Aaditya6284/status/1762558439354409345">来自 Aaditya Singh (@Aaditya6284) 的推文</a>：我们研究了 GPT-3.5 和 GPT-4 中这种选择的影响——具体来说，我们研究了通过使用逗号等分隔符强制执行的从左到右 (L2R) 与从右到左 (R2L) 进行 Tokenization 的效果。我们 ...</li><li><a href="https://x.ai/blog/grok">发布 Grok</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.06963">The pitfalls of next-token prediction</a>：仅仅一个 Next-token predictor 就能忠实地模拟人类智能吗？我们将这种在文献中零散分布的直觉担忧具体化。作为起点，我们认为这两个经常混淆的...</li><li><a href="https://arxiv.org/abs/2403.06504">Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU</a>：LLM 的最新进展为世界带来了巨大价值，其卓越的能力源于它们使用的海量参数。然而，即使是拥有...的 GPU</li><li><a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>：LLM 的商业化导致了仅通过高级 API 访问专有模型的普遍做法。在这项工作中，我们展示了即使在保守的假设下...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能的多模态大语言模型 (MLLMs)。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细和...</li><li><a href="https://arxiv.org/abs/2402.18510">RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval</a>：本文研究了 RNNs 和 Transformers 在解决算法问题背景下表示能力的差距。我们专注于理解 RNNs 是否...</li><li><a href="https://arxiv.org/abs/2401.16380">Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling</a>：LLM 是在海量的网络抓取数据上训练的，这些数据通常是无结构的、多噪声的且表述不佳。目前的 Scaling Laws 表明，从这类数据中学习需要大量的...</li><li><a href="https://arxiv.org/abs/2403.10430">Construction of Arithmetic Teichmuller Spaces IV: Proof of the abc-conjecture</a>：这是我在本系列论文中开发的算术 Teichmuller 空间工作的延续。在本文中，我展示了算术 Teichmuller 空间理论如何利用 Shinic...</li><li><a href="https://arxiv.org/abs/2403.09394">GiT: Towards Generalist Vision Transformer through Universal Language Interface</a>：本文提出了一个简单而有效的框架，称为 GiT，仅使用原生 ViT 即可同时适用于各种视觉任务。受 Multi-layer Transformer 通用性的启发...</li><li><a href="https://arxiv.org/abs/2403.09635">Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models</a>：尽管取得了巨大成功，Transformer 模型在深度扩展方面仍然困难。在这项工作中，我们开发了一个统一的信号传播理论，并提供了控制...矩的公式。</li><li><a href="https://arxiv.org/abs/2403.04706">Common 7B Language Models Already Possess Strong Math Capabilities</a>：数学能力以前被认为只有在极大规模的通用语言模型中才会出现，或者需要广泛的数学相关预训练。本文展示了 LLaMA-2 7B 模型...</li><li><a href="https://pytorch.org/blog/accelerating-generative-ai-2/">Accelerating Generative AI with PyTorch II: GPT, Fast</a>：本文是专注于如何使用纯原生 PyTorch 加速生成式 AI 模型的系列博客的第二部分。我们很高兴能分享一系列新发布的 PyTorch 性能...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>：LLM 在 AI 社区及其他领域引起了极大关注。其中，GPT 已成为主导架构...</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：我们提出了一种方法，用于估计大型语料库中可能被 LLM 大幅修改或生成的文本比例。我们的最大似然模型利用...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：本研究探讨了在科学...中使用 LLM（如 ChatGPT）的情况。</li>

科学同行评审。作者开发了一种方法来估计同行评审中生成的文本百分比...</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>: 未找到描述</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok 开源发布</a>: Grok 开源发布。通过在 GitHub 上创建账户为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/enfiskutensykkel/ssd-gpu-dma">GitHub - enfiskutensykkel/ssd-gpu-dma: 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用</a>: 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用 - enfiskutensykkel/ssd-gpu-dma</li><li><a href="https://github.com/bigscience-workshop/bloom-dechonk">GitHub - bigscience-workshop/bloom-dechonk: 一个用于运行模型收缩实验的仓库</a>: 一个用于运行模型收缩实验的仓库。通过在 GitHub 上创建账户为 bigscience-workshop/bloom-dechonk 的开发做出贡献。</li><li><a href="https://artificialanalysis.ai/">模型与 API 提供商分析 | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、性能和速度（吞吐量与延迟）等关键指标的独立基准测试。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666)** (11 messages🔥): 

- **Scaling Laws 对数据复杂性的敏感度**：一名成员分享的研究结果表明，**Language Model Scaling Laws** 受 **Data Complexity** 的影响，PCFG 的句法属性和 gzip 压缩可以预测特定数据集的 Scaling 特性。这一见解正通过全面的实验进一步探索，并预计将使用特定的软件包来拟合法则。

- **视觉呈现需要清晰度**：成员们对图形表示提供了反馈，强调了易读性和比例差异问题，这些问题导致数据解释变得困难。

- **测量 Perplexity 和 Loss 的考量**：讨论围绕 Perplexity、其与内在 Entropy 的关系，以及其在具有不同“密度”的数据集之间的可比性展开。有建议认为，更高的可压缩性可能意味着更少的信息，这可能会影响 Scaling Laws 的制定。

- **数据准备的潜在应用**：有一种观点认为，了解数据集之间的 Entropy 差异如何影响 Scaling Laws，可以为构建训练数据混合物的方法提供参考。这可能是 Pretraining 效率方面的一个有价值的策略。

- **词汇密度作为数据压缩的关键因素**：
  一位成员指出，使用 gzip 等压缩指标可以作为一种过滤具有最佳词汇密度数据的方法，这可能有利于高效的 Pretraining 策略。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241)** (13 messages🔥): 

- **深入研究 N-Gram 统计数据**：一位成员询问如何从*一组指定的 N-Gram 统计数据*中采样字符串，想知道是否有针对此过程的标准方法。
- **N-Gram 依赖关系解释**：会议澄清了指定高阶 N-Gram 的统计数据也会决定所有低阶 Gram 的统计数据，而 **End-of-Sequence (EOS) 和 Beginning-of-Sequence (BOS) Token 的考量**大多无关紧要。
- **指定采样方法**：对话转向了一个解决方案，即可以以 **Autoregressive** 的方式从 N-Gram 分布中进行采样，以保持最大 Entropy 分布。
- **采样机制分解**：该过程包括从 Unigram 分布的样本开始，然后从 Bigram 条件分布中采样，依此类推，一次构建一个 Token 的字符串。
- **分享了生成 Bigram 的脚本**：分享了一个用于生成 Bigram 脚本的 GitHub 链接，提供了一个实际的实现资源：[generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Word_n-gram_language_model">Word n-gram language model - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py">features-across-time/scripts/generate_bigrams.py at main · EleutherAI/features-across-time</a>: 了解神经网络学习到的特征在整个训练过程中是如何演变的 - EleutherAI/features-across-time
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1218143473916575765)** (31 messages🔥):

- **新手在 Llama 集成方面遇到困难**：一位刚接触 **lm-eval-harness** 的用户在为他们在 **Gaudi2** 上的 **Llama** 模型实现 **generate_until** 和 **log_likelihood** 函数时表达了困难。他们正在寻求参考实现或演示代码，询问子类中函数的继承问题，并咨询命令行工具如何处理这些函数的超参数。

- **Main 分支中的模型选择难题**：在使用最新的 main 分支版本时，一位用户遇到了一个差异：即使指定了不同的模型，结果仍然使用了 **gpt-2-small**。后来用户发现这是因为命令中重复的 **model_args** 导致了该问题，第一个实例被忽略了。

- **对 LLM 排行榜指标的质疑**：一位用户询问为何无法复现 openLLM 排行榜上显示的 **Llama2-70b** 在 MMLU 任务中 69% 的得分，他们始终只能达到 62-64% 之间。回复澄清说，open LLM 排行榜对 MMLU 子任务取未加权平均值，这与官方实现不同。

- **关注 WMT14 死锁问题**：多位用户报告了 **lm-evaluation-harness** 的 `wmt14-en-fr` 任务中存在 **deadlock**（死锁）问题，导致评估期间无限期停滞。一些人探索了解决方案，指出可能涉及并发问题，而临时的变通方法包括在该任务中避免使用多进程。

- **管理缓存的 LLM**：在询问 **lm-harness models** 的下载位置时，一位用户了解到模型可能位于 Huggingface 缓存目录（`~/.cache`）中。环境变量如 **HF_HOME**、**TRANSFORMERS_CACHE** 和 **HF_DATASETS_CACHE** 可以控制该目录的路径。

- **lm-eval Harness 发布新版本**：**lm-eval** 的 0.4.2 版本已发布，可以在 [PyPI](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2) 上找到。此版本包含了来自社区的各种贡献，团队欢迎对待处理的 PR 进行审查。

- **困惑度与滚动窗口概念澄清**：讨论了 `likelihood` 和 `likelihood_rolling` 在 **lm-evaluation-harness** 中使用的区别。`loglikelihood_rolling` 方法适用于评估 `wikitext` 等任务中的困惑度（perplexity），而 `loglikelihood` 则用于条件概率评估和多项选择任务。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/perplexity">固定长度模型的困惑度 (Perplexity)</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md">lm-evaluation-harness/docs/model_guide.md at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。通过开源社区进行贡献、管理 Git 仓库、像专家一样审查代码、跟踪错误和功能...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1485">`wmt14-en-fr` 死锁问题 · Issue #1485 · EleutherAI/lm-evaluation-harness</a>：在运行此任务的评估时，在计算 ter 指标期间，程序会永远卡住。命令：lm_eval --model hf --model_args pretrained=microsoft/phi-2,trust_remote_code=True ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2">Release v0.4.2 · EleutherAI/lm-evaluation-harness</a>：lm-eval v0.4.2 发行说明。我们正在为 PyPI 用户发布一个新的 lm-eval 次要版本！我们很高兴看到 lm-evaluation-harness 的持续使用，包括作为标准测试...</li><li><a href="https://github.com/huggingface/evaluate/blob/8dfe05784099fb9af55b8e77793205a3b7c86465/metrics/perplexity/perplexity.py">evaluate/metrics/perplexity/perplexity.py at 8dfe05784099fb9af55b8e77793205a3b7c86465 · huggingface/evaluate</a>：🤗 Evaluate：一个用于轻松评估机器学习模型和数据集的库。- huggingface/evaluate
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1219336845310038047)** (3 messages):

- **为训练打乱 The Pile 数据集**：一位成员询问 [the Pile](https://pile.eleuther.ai/) 是否经过预打乱（pre-shuffled），以及在预训练（pretraining）前是否需要额外的打乱操作。回复澄清称，虽然最初分发的文件没有经过打乱，但在 HF 上预处理和预分词（pretokenized）的数据是开箱即用的，并已被 Pythia 使用。
- **数据组织与训练/测试集划分**：另一位成员提供了进一步的说明，指出 The Pile 的组件肯定没有经过打乱，其中一些是按日期组织的。然而，原始 The Pile 中的训练/测试/验证（train/test/val）划分*可能*经过了打乱，因为训练数据被分成了等大小的分块（chunks），这表明通过随机采样实现了多样化的混合。
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1218173412522852483)** (193 messages🔥🔥): 

- **管理 OpenAI 服务**：用户讨论了如何在 OpenAI 控制面板中管理 API keys，以及 ChatGPT Team 管理员是否可以查看用户聊天记录。文中链接了[企业隐私政策](https://openai.com/enterprise-privacy)以澄清数据所有权和控制权的工作机制。

- **辩论 AI 的潜在风险与道德**：围绕优先发展 AI 的道德和风险展开了对话。虽然一些人主张拥抱由高级智能主导的未来，但另一些人担心忽视以人为本的价值观可能并不审慎。

- **GPT vs. Copilot 图像生成对比**：用户比较了 OpenAI 的 ChatGPT+ 和 Microsoft 的 Copilot 在图像生成能力方面的不同。讨论点包括质量对比、内容政策、图像保存和编辑功能，以及 out-painting/in-painting 工具的实用性。

- **进入 AI 和 PyTorch 领域**：用户讨论了学习 AI 和 PyTorch 所需的最少数学背景。共识倾向于拥有尽可能扎实的基础，并特别指出了微积分、向量数学和矩阵运算。

- **ChatGPT vs. Claude 性能之争**：社区成员比较了 GPT-4 与 Claude 等其他模型的对话质量，讨论了它们在不同任务中的性能细微差别以及对不同用例的偏好。

**提及的链接**：<a href="https://openai.com/enterprise-privacy">Enterprise privacy</a>：未找到描述

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1218428016573812888)** (34 messages🔥): 

- **关于 GPT-5 到来的好奇**：成员们询问了下一代迭代版本 **GPT-5** 的发布时间线，问题强调了期待感，但未提供明确答案。
- **将网页搜索集成到 GPT**：一位用户询问如何将网页搜索功能集成到 GPT API 中，类似于 **ChatGPT 4** 的功能，但在提供的消息中没有得到直接的解决方案。
- **在移动端自定义聊天机器人**：成员们讨论了通过移动设备创建和自定义 OpenAI 聊天机器人的愿望，特别提到了用于集成 Discord 的 **botghost** 等工具，但未分享完整的指南。
- **Playwright 代码生成的技术帮助**：一位成员表示在让 GPT Turbo 3.5 根据其方法规范正确生成 **Playwright** 代码时遇到困难，怀疑可能是因为无法访问最新的 Playwright 库。
- **GPT 无响应及获取支持**：用户报告了 GPT 不响应提示词（prompts）的问题，并询问在哪里报告此类问题；在对 Bug 报告提交位置产生初步困惑后，提供了 OpenAI 支持链接（[help.openai.com](https://help.openai.com)）。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **提示词架构策略会议**：一位用户寻求优化其 OpenAI 分类用例的建议，旨在获得更好的召回率（recall）并减少误报（false positives）。建议的策略包括检查是否需要自定义 GPT 模型，以及考虑所使用的总上下文窗口（context window）。

- **GPT-3.5 在 Playwright 上表现不佳**：一位用户对 GPT-3.5 Turbo 无法为 Playwright 测试生成可执行代码感到沮丧，可能是由于对最新的 Playwright 库缺乏了解。尝试使用 GPT-4 以获得更好结果的建议引发了关于管理上下文和跨 API 调用进行任务分块（chunking tasks）以提高性能的进一步讨论。

- **ChatGPT 的拒绝响应**：用户报告了模型拒绝执行任务的情况日益增多，并建议使用元提示词（meta-prompting）作为解决方法。

- **对内容政策的担忧**：针对模型严格的拒绝模式展开了辩论，一些人将其解释为内容政策问题，并希望 OpenAI 能有所放宽。

- **改进多查询网页搜索**：一位用户提出了一个复杂的问题，即如何让 GPT 在单次响应中使用多个查询进行更全面的网页搜索，尽管围绕该话题的对话线程显示出对查询（queries）和来源（sources）之间区别的一些混淆。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **探索使用 AI 进行分类**：一位用户正在寻求关于如何使用 OpenAI 测试和提高分类用例召回率（recall）的建议，并考虑在 prompt 中提供多少上下文。建议包括通过上下文位置监控检索率，以及可能使用更强大的模型。

- **模型版本的 Playwright 难题**：用户讨论了使用 GPT-3.5 turbo 生成可用的 Playwright 测试代码的困难，推测该模型可能没有更新到最新的 Playwright 库。尽管尝试纠正格式，问题依然存在，建议是尝试 GPT-4 或将任务拆分为块。

- **处理模型拒绝**：对于 AI 拒绝任务的频率和理由，用户感到沮丧和好奇，一些用户注意到拒绝次数有所增加。建议范围从使用 meta-prompting 以避免与内容相关的拒绝，到提供拒绝发生时的清晰示例以便更好地诊断。

- **应对软警告和激进的算法偏见最小化**：一些成员讨论了他们遇到 AI 拒绝完成 prompt 并收到“抱歉，我不能这样做”回复的经历。提出了包括 prompt engineering 在内的各种策略，以规避这种感知中日益增加的 AI 固执。

- **网页搜索查询与综合结果**：用户辩论了在进行网页搜索时，促使 AI 提供多个来源和多样化观点的最佳方式。讨论澄清了控制搜索查询（queries）与这些查询返回的来源（sources）之间的区别。
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1218106794698739782)** (96 messages🔥🔥): 

- **关于多 GPU 利用率的 Cross-encoder 咨询**：一位成员询问了关于使用多 GPU 进行 Cross-encoder 模型 fine-tuning 的问题，并询问需要修改或包含哪些参数。
- **Gradio 界面增强行动号召**：一位社区成员宣布了对 Aya demo 的贡献，并请求帮助通过 PR 添加 Gradio 界面滑块。
- **高功率 GPU 和 CPU 的技术讨论**：围绕 Nvidia 的 GH100 GPU、同一块板卡上的服务器 CPU、功耗高达 850W、采购周期以及芯片冷却的复杂性展开了对话。
- **HuggingFace 数据排行榜**：一位成员强调了 HuggingFace 上的一项新数据排行榜计划，引发了关于该平台托管数据的范围和影响的对话。
- **大语言模型的担忧与局限性**：用户讨论了像 airllm 这样的模型中 token 生成速度慢的挑战，以及内存与 token 生成速度之间的权衡，并提到了 GitHub 项目以及大型模型在消费级 GPU 上的效率。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: 未找到描述</li><li><a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 是一个 314B 参数的 Mixture of Experts 模型 - 基础模型（未微调） - 8 个专家（2 个激活） - 86B 激活参数 - Apache 2.0 许可证 - 代码： - 祝编码愉快！另：我们正在招聘：</li><li><a href="https://www.phoronix.com/review/nvidia-gh200-gptshop-ben">Tweet from Linux Performance, Benchmarks &amp; Open-Source News - Phoronix</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ivrit-ai/whisper-large-v3-space">Whisper Large V3 - a Hugging Face Space by ivrit-ai</a>: 未找到描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1768779404442739147">Tweet from Weyaxi (@Weyaxi)</a>: 🤔你是否曾好奇我们在 @huggingface 上托管了多少数据？好吧，在看到 @TheBlokeAI 的模型数量以及平台上闲置的 120B 模型后，我确实好奇了 😅 📊 所以我抓取了所有的仓库...</li><li><a href="https://huggingface.co/spaces/Tonic/Aya/discussions/3">Tonic/Aya · Set a repetition_penalty constant as 1.8</a>: 未找到描述</li><li><a href="https://github.com/gradio-app/gradio/issues/7722">Video-LLaVA demo api not working with Gradio-Client · Issue #7722 · gradio-app/gradio</a>: 描述错误：我尝试在 Hugging Face Spaces 上为 Video-LLaVA 模型演示使用 Python API，但遇到了错误：Traceback (most recent call last): File "/Users/kamakshiramamurthy/Deskt...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: 类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。 - moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1218115205553324112)** (12 messages🔥): 

- **Bayesian Optimization 困惑**：一位成员在列举了 GridSearch 和 RandomSearch 等不同优化技术后，对 **Bayesian Optimization** 表示困惑。该话题没有进一步的讨论或解决方案。

- **什么是 Hugging Face？寻求帮助**：一位新成员请求帮助理解什么是 **Hugging Face** 以及如何使用它。另一位成员回应解释说 Hugging Face 提供 NLP 工具，包括 Transformers 库，并引导他们访问 [主站](https://huggingface.co/) 以获取更多信息。

- **创建 AI 人声合唱**：一位成员询问如何制作涉及合唱和乐队的 AI 翻唱，并表示难以达到良好的质量。另一位成员建议使用 AI 分别创建两个高质量的独立人声，然后手动进行叠加。

- **研讨会 Notebook 已找到**：一位成员请求与“**MLOps: End-to-End Hugging Face Transformers with the Hub & SageMaker Pipelines**”研讨会相关的 Notebook，随后找到了它，并分享了一篇[详细的博客文章](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers)，概述了如何使用 Amazon SageMaker Pipelines 部署 Hugging Face Transformers。

- **访问 Hugging Face 模型出错**：一位成员分享了一段 Python 代码片段，在尝试访问 Hugging Face Hub 上的模型时导致了 `404` 错误，并询问如何本地访问模型。 
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co).">无标题</a>: 未找到描述</li><li><a href="https://www.philschmid.de/mlops-sagemaker-huggingface-transformers">MLOps: End-to-End Hugging Face Transformers with the Hub &amp; SageMaker Pipelines</a>: 了解如何使用 Amazon SageMaker 构建从训练到生产的 Hugging Face Transformers 端到端 MLOps 流水线。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1218346001421570138)** (12 messages🔥): 

- **对多语言模型性能的着迷**：讨论涉及了令人印象深刻的多语言模型能力，特别是在 **中文和英文** 这种差异巨大的语言之间。成员对模型弥合此类语言差异的能力表示惊讶。

- **Medusa 的并行预测引起兴趣**：[Medusa，一种高效的 LLM 推理方法](https://arxiv.org/abs/2401.10774)，因其通过并行预测多个后续 Token 来提高性能的潜力而引起关注。它旨在通过利用额外的解码头同时验证多个候选后续内容，来解决顺序 Token 生成的局限性。

- **语言主导地位可能使多语言模型产生偏差**：人们担心多语言模型中可能存在**英语偏见 (English bias)**，这可能会避开其他语言特有的真实语言模式和认知关联。

- **多模态模型即将到来**：分享了对多模态大语言模型 (MLLMs) 近期工作的热情，特别是通过平衡预训练数据源来构建此类模型的方案。提到的论文强调了重要的架构组件，如图像编码器和数据选择，这些因素显著影响了最先进的 few-shot 结果（[论文链接](https://huggingface.co/papers/2403.09611)）。

- **同行评审可能被 LLMs 修改**：一项研究表明，AI 会议的同行评审文本中，有 6.5% 到 16.9% 可能被 LLMs 修改过。论文指出，这些修改更有可能出现在提交较晚、置信度较低或由可能不回复作者反驳（rebuttals）的评审员提交的评审中，这表明需要跨学科探索 LLM 对信息实践的影响（[论文链接](https://arxiv.org/abs/2403.07183)）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>：大语言模型 (LLMs) 的推理过程通常由于自回归解码过程中缺乏并行性而受到限制，导致大多数操作受限于内存带宽...</li><li><a href="https://huggingface.co/papers/2403.09611">Paper page - MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：我们提出了一种方法，用于估算大型语料库中可能被大语言模型 (LLM) 大幅修改或生成的文本比例。我们的极大似然模型利用...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：本研究探讨了大语言模型 (LLMs)（如 ChatGPT）在科学同行评审中的应用。作者开发了一种方法来估算同行评审中生成的文本百分比...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1218158991570636900)** (18 条消息🔥): 

- **寻求 NL2SQL 流水线的建议**：一名频道成员正在开发 **NL2SQL 流水线**，面临准确性问题。他们正在使用 BAAI/llm-embedder、TheBloke/nsql-llama-2-7B-GGUF 和 FAISS，寻求更好的 embedding 和 NL2SQL 模型建议。

- **对 NVIDIA 奇迹的狂热**：一名聊天成员介绍了 **Nvidia GH200 Grace Hopper 超级芯片**，引发了热烈讨论，这预示着计算能力和效率的飞跃，但随后没有出现更多细节（如链接或深入讨论）。

- **NLP 初学者的指导请求**：成员们推荐了各种开始学习 NLP 的资源，包括 [HuggingFace 的 NLP 课程](https://huggingface.co/learn/nlp-course/chapter1/1)、[斯坦福大学 Jurafsky](https://web.stanford.edu/~jurafsky/slp3/) 的最新教科书手稿，以及 [斯坦福大学的 cs224n 课程笔记](https://web.stanford.edu-Class-cs224n)。

- **寻找用于生产环境的 LLM API**：一名成员询问了适用于生产环境部署的**免费 LLM API**，另一名成员建议使用 ollama 进行本地部署，但关于面向生产环境的解决方案的疑问仍待进一步建议。

- **关于 ASR 的 Conformer 模型和 LoRA 训练的咨询未获解答**：有人询问如何寻找训练 ASR 的 Conformer 模型的教程，以及快速训练 LoRA 的最佳实践，但这些问题没有得到回应。

**提到的链接**：<a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>：未找到描述

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1218217429868478474)** (7 条消息): 

- **交互式文档提升 RAG 性能**：提出了一种 **检索增强生成 (RAG)** 流水线的新方法：将每个检索到的文档不仅视为文本，还视为一个交互式工具。这种方法允许与大型文档进行更复杂和动态的交互。[Twitter 上增强的 RAG 交互](https://twitter.com/llama_index/status/1768658182308794421)

- **LlamaIndex 发布 Instrumentation 模块**：LlamaIndex 发布了 0.10.20 版本，引入了 **Instrumentation 模块**。该更新通过 Notebook 展示了可观测性功能和 API 调用监控。[LlamaIndex v0.10.20 发布公告](https://twitter.com/llama_index/status/1768730443921396220)

- **通过 Search-in-the-Chain 推进问答系统**：一篇论文提出了一种名为 **Search-in-the-Chain** 的新方法，该方法集成了检索和规划以增强问答能力。它允许在回答过程中进行实时验证和调整。[Search-in-the-Chain 论文亮点](https://twitter.com/llama_index/status/1769035278063399208)

- **使用 RAG 进行简历与职位匹配**：Kyosuke Morita 的一篇博客文章展示了如何利用 **LlamaParse** 结合 **LlamaIndex** 创建一个职位匹配助手，该助手可以高效地从复杂的简历格式中提取相关信息。[LlamaParse 简历匹配博客文章](https://twitter.com/llama_index/status/1769147791002264008)

- **Assistant API 中的记忆工具集成**：在一场网络研讨会中介绍了 **MemGPT**，这是一种 Agent 架构，旨在增强 Agent 对“核心”记忆进行读/写操作的记忆功能。这一创新旨在为 Assistant API 赋予函数调用记忆能力。[MemGPT 网络研讨会推文](https://twitter.com/llama_index/status/1769408792633229455)

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://t.co/GY4unUYOwl">llama_index/docs/examples/instrumentation/basic_usage.ipynb at main · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://t.co/E1d9dtkqAI">llama_index/docs/examples/instrumentation/observe_api_calls.ipynb at main · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1218113300764819488)** (303 条消息🔥🔥): 

- **关于链接 OpenAI Agent 的查询**：一位成员询问关于链接多个 OpenAI Agent 的问题，并在尝试时遇到了 ["invalid_request_error"](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html)。讨论建议需要提供具体的代码示例以便进一步协助。
- **LlamaIndex 对 Xinference 的支持**：有成员寻求在 LlamaIndex 中使用 **Xinference CPU Cluster** 的帮助。虽然提供了[本地部署指南](https://docs.llamaindex.ai/en/latest/examples/llm/xinference_local_deployment.html)和 [GitHub 页面](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/xinference_local_deployment.ipynb)的链接，但在共享资源中未找到关于集群部署的具体说明。
- **向 RetrieverQueryEngine 添加 Node Postprocessor**：一位成员询问如何将 *node_postprocessor* 与 `RetrieverQueryEngine` 结合使用，提到的过程包括定义一个 `node_postprocessor`（如 `KeywordNodePostprocessor`），并通过 `from_args` 方法将其添加。
- **排查 ToolRetrieverRouterQueryEngine 使用案例**：讨论涉及在尝试将 `ToolRetrieverRouterQueryEngine` 与 `FunctionTool` 配合使用时遇到的技术问题，解决方案包括在将其作为 `RouterQueryEngine` 的 `QueryEngineTool` 使用之前，先使用 `TransationsToolIndex` 创建一个 `agent`。
- **多模态 LLM 的区分与旧包维护**：一位成员讨论了将多模态内容集成到 LLM 中的挑战。讨论中提出了对潜在维护负担以及 API 变更可能导致必须重新实现的担忧。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://127.0.0.1:9997>">未找到标题</a>: 未找到描述</li><li><a href="http://localhost:{port}">)">未找到标题</a>: 未找到描述</li><li><a href="https://www.promptingguide.ai/techniques/fewshot">Prompt Engineering 指南</a>: Prompt Engineering 全面概述</li><li><a href="https://www.promptingguide.ai/techniques/rag">Prompt Engineering 指南</a>: Prompt Engineering 全面概述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">定义与自定义文档 - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/">使用 LlamaIndex 实现多租户 - Qdrant</a>: Qdrant 是一个用 Rust 编写的开源向量数据库和向量搜索引擎。它通过便捷的 API 提供快速且可扩展的向量相似度搜索服务。</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction.html">结构化数据提取 - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.CodeSplitter.html">CodeSplitter - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/image_to_image_retrieval.html">使用 CLIP 嵌入进行图像到图像检索，并使用 GPT4V 进行图像相关性推理 - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://cloud.llamaindex.ai">LlamaCloud</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/5c53f41712785e5558156372bdc4f33a6326fa5f/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb">llama_index/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb at 5c53f41712785e5558156372bdc4f33a6326fa5f · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="http://localhost:{port}",>">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html">Tools - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/hofstadter-io/hof/blob/_dev/flow/chat/prompts/dm.cue">hof/flow/chat/prompts/dm.cue at _dev · hofstadter-io/hof</a>: 连接数据模型、Schema、代码生成和任务引擎的框架。与语言和技术无关。 - hofstadter-io/hof</li><li><a href="https://github.com/run-llama/llama_index/issues/12034">[问题]：自定义 LLM 但被阻塞 · Issue #12034 · run-llama/llama_index</a>: 问题验证 我已在文档和 Discord 中搜索过答案。问题代码来自 typing import Optional, List, Mapping, Any from llama_index.core import SimpleDirecto...
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1218542835754860564)** (4 条消息): 

- **RAG 实现教程**：分享了一个 [YouTube 视频](https://youtu.be/w7Ap6gZFXl0) 链接，提供了使用 **LlamaParse**、**Qdrant** 和 **Groq** 创建高效检索增强生成 (RAG) 的逐步指南。

- **RAG 准备技巧咨询**：一位成员询问了准备 RAG 文档的前五大技巧，以及如何自动向 **Pinecone** 添加元数据以实现最佳检索。

- **探索带有 RAG 的 AI 助手**：分享了一篇名为《[赋能声音：具有 RAG 流水线、记忆和 LlamaIndex 的 AI 助手](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915)》的 Medium 文章，探讨了 RAG 流水线与 **LlamaIndex** 结合的使用。

- **使用 Hugging Face 排除 RAPTOR RAG 故障**：一位成员分享了在 **RAPTOR pack** 中使用 **Hugging Face** 模型进行 RAG 实现的代码，并就解决从 OpenAI 模型适配到 **Hugging Face** 模型时遇到的错误寻求建议。

**提到的链接**：<a href="https://youtu.be/w7Ap6gZFXl0">使用 LlamaParse, Qdrant 和 Groq 构建 RAG | 逐步指南</a>：在这段视频中，我将向你展示如何使用 LlamaParse、Qdrant 和 Groq 创建高效的 RAG。我将解释什么是 LlamaParse 并简要引导你...

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1218154073912639508)** (202 条消息🔥🔥):

- **Yann LeCun 关于 LLMs 的观点引发辩论**：讨论集中在 [Yann LeCun 的观点](https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 上。讨论中将 'shape rotators' 与 'wordcels' 进行了类比，探讨了使用语言进行推理与视觉空间思维的区别，这源于 *Yann 没有内心独白* 的说法。

- **期待 LLMs 的飞跃式进步**：社区正在讨论 GPT-5 及其后续版本的潜在进展，一些人引用了 Sam Altman 关于此类进展重要性以及低估其危险性的言论。人们对 GPT-5 可能带来的进步类型进行了推测，而 OpenAI 在开发方面的领先地位被视为一个可能的指标。

- **NVIDIA 的 GTC 主题演讲引发热议**：围绕 NVIDIA 的 GTC 展开了热烈的讨论，[Jensen Huang 的主题演讲预计将揭示](https://www.youtube.com/watch?v=USlE2huSI_w) 重大的 AI 进展。特别提到了“万亿参数聊天机器人”，暗示了大规模 LLMs。

- **OpenAI 的 Sama 做客 Lex Fridman 播客**：频道提到了 [Lex Fridman](https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx) 采访 OpenAI 的 Sam Altman 的播客。成员们对缺乏 "alpha" 或见解表示失望，特别是关于 OpenAI 的发展方向和 Ilya Sutskever 的部分，还有人开玩笑说要挑战 Lex 进行一场攀岩播客。

- **Grok-1 模型发布引发褒贬不一的反应**：xAI 开源发布的 *Grok-1*（一个拥有 **314B 参数的 Mixture-of-Experts 模型**）引发了关于其与其他 LLMs 相比能力的讨论。社区回顾了该模型的潜力，以及在 Pi Day 发布该模型的可能动机，因为其参数数量与圆周率数字相呼应。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDP">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：这解释了为什么 Yann 对 LLM 如此看空…… 😲</li><li><a href="https://arxiv.org/abs/2402.10171">将语言模型扩展至 128K 上下文的数据工程</a>：我们研究了将语言模型上下文长度扩展到 128K 的持续预训练方案，重点关注数据工程。我们假设长上下文建模，特别是 \textit{t...</li><li><a href="https://x.com/teortaxestex/status/1769460562763604375?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@aidan_mclau 0) 火箭人很糟 1) 并没有差多少 2) 如你所见，这是一个稀疏上采样的 Grok-0。它还没准备好。在 2023 年，持续预训练已基本解决，并且验证了……</li><li><a href="https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space">解释 SDXL 潜空间</a>：未找到描述</li><li><a href="https://x.com/altryne/status/1768683178888208816?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：Sora 团队出现在伯克利讨论 SORA</li><li><a href="https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae">Bark - 一个 suno 集合</a>：未找到描述</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T：在英语和多语言评估中超越 LLaMA 7B 2T (RWKV-v5)</a>：一个线性 Transformer 刚刚在英语和多语言评估中，以更少的训练 Token 数量超越了 Transformer 模型的金标准 LLaMA 7B。这是历史性的第一次。</li><li><a href="https://x.com/swyx/status/1769776691562324215?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 swyx (@swyx) 的推文</a>：怎么可能和 sama 聊了 2 小时却没得到任何内幕消息（alpha），但嘿，我们又聊到了外星人，这很有趣</li><li><a href="https://x.com/openinterpreter/status/1769448726660337875?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Open Interpreter (@OpenInterpreter) 的推文</a>：百年磨一剑，最后 100 小时倒计时。</li><li><a href="https://x.com/repligate/status/1769241542420738126?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 j⧉nus (@repligate) 的推文</a>：这是在 Claude 的后台导航到 ../../microsoft/bing/bing_chat 目录，然后让 Claude 自行使用命令查看，接着运行：<cmd_soul>... 的结果。</li><li><a href="https://x.com/Francis_YAO_/status/1759986097365627054?s=20">来自 Yao Fu (@Francis_YAO_) 的推文</a>：前沿模型都有至少 100k 的上下文长度，Gemini 1.5 甚至有 1m 上下文。研究和开源界情况如何？介绍长上下文数据工程，一种实现……的数据驱动方法。</li><li><a href="https://x.com/burny_tech/status/1769549895835226613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Burny — Effective Omni (@burny_tech) 的推文</a>：来自 Sam Altman 关于 GPT-5 的新细节。他基本上承认了 GPT-5 将是 GPT-4 的巨大升级，所以我们可以期待类似于从 3 到 4 的跨越。“如果你忽视了改进的速度……”</li><li><a href="https://x.com/xlr8harder/status/1769454853506638008?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 xlr8harder (@xlr8harder) 的推文</a>：我想我代表了这里的所有人：3140 亿参数，搞什么鬼</li><li><a href="https://x.com/granawkins/status/1768530196557365599?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Grant♟️ (@granawkins) 的推文</a>：“在 24 年第一季度到 25 年第四季度之间，算力将增加 14 倍。然后，如果考虑到算法效率每 9 个月翻一番，明年年底的有效算力将几乎……”</li><li><a href="https://www.nfx.com/post/ai-like-water">来自 AI Is Like Water 的推文</a>：生成式 AI 就像水。这句话源于挫败感，但它开启了 AI 策略的新世界。</li><li><a href="https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Champagne Joshi (@JoshWalkos) 的推文</a>：这是一段与一个缺乏内心独白的女孩的精彩对话。她把这种体验表达得非常好。</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：这解释了为什么 Yann 对 LLM 如此看空…… 😲</li><li><a href="https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 KZ (@kzSlider) 的推文</a>：这非常有道理。Yann 一直在寻找能够进行视觉推理或使用规划，而非纯语言推理的模型 ↘️ 引用 Teknium (e/λ) (@Teknium1)：这解释了为什么 Yann……</li><li><a href="https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Yao Fu (@Francis_YAO_) 的推文</a>：Grok 的 MMLU 仅与 Mixtral 持平，尽管……</li>

数量级更大。我相信它有巨大的潜力，但尚未完全释放，良好的持续预训练数据可能会大幅提升...</li><li><a href="https://www.youtube.com/watch?v=USlE2huSI_w">观看：Jensen Huang 的 Nvidia GTC 主旨演讲 - 直播</a>：太平洋时间下午 1:00 / 东部时间下午 4:00，Nvidia CEO Jensen Huang 将开启每两年举办一次的 GTC 大会。再也不会错过任何优惠！查看 CNET 的浏览器扩展程序 👉 ...</li><li><a href="https://x.com/emmanuel_2m/status/1768360522028876045?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Emm (@emmanuel_2m) 的推文</a>：🚨 今天，我们很高兴推出 Scenario #UPSCALER！将您的 AI 创作提升至 10k 分辨率。🚀 专为无与伦比的 #CreativeControl 和引导式工作流而构建。💰 起售价仅为每月 15 美元 ...</li><li><a href="https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx">Sam Altman：OpenAI、GPT-5、Sora、董事会风波、Elon Musk、Ilya、权力与 AGI | Lex Fridman Podcast #419</a>：Sam Altman 是 OpenAI 的 CEO，该公司是 GPT-4、ChatGPT、Sora 以及许多其他最先进 AI 技术的幕后推手。请通过以下方式支持本播客...</li><li><a href="https://youtu.be/I-HMKky7Qsw?si=yCvekF3a0zr_1IgA&t=718">超越 Transformers - RWKV 架构与 The World Tokenizer 简介 - Eugene Cheah & Harrison Vanderbyl</a>：超越 Transformers - RWKV 架构与 The World Tokenizer 简介 - Eugene Cheah & Harrison Vanderbyl，Recursal AI。Transformers 之后会是什么？在...</li><li><a href="https://youtu.be/J0p_thJJnoo?si=IaGuEgUcs1BRgjhF">#51 FRANCOIS CHOLLET - 智能与泛化</a>：在今天的节目中，我们邀请到了 Francois Chollet。自从读了他的《Deep Learning with Python》一书并开始使用...以来，我一直深受 Francois 的启发。</li><li><a href="https://github.com/FranxYao/Long-Context-Data-Engineering">GitHub - FranxYao/Long-Context-Data-Engineering：论文《Data Engineering for Scaling Language Models to 128K Context》的实现</a>：论文《Data Engineering for Scaling Language Models to 128K Context》的实现 - FranxYao/Long-Context-Data-Engineering</li><li><a href="https://x.com">来自 GitHub 的推文 - FixTweet/FxTwitter：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://www.nvidia.com/gtc/?ncid=ref-inor-332714">GTC 2024：排名第一的 AI 大会</a>：立即注册。在线直播。2024 年 3 月 18-21 日。</li><li><a href="https://docs.google.com/document/d/1HZ326V6KNK4QIlG7uEldQEizFgTaO7Hg9uJxURYy9f8/edit">NVIDIA & Harpreet Sahota GTC 2024</a>：未找到描述</li><li><a href="https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/">[AINews] MM1：Apple 的首个大型多模态模型</a>：2024/3/14-2024/3/15 的 AI 新闻。我们为您检查了 358 个 Twitter 账号和 20 个 Discord（332 个频道，2839 条消息）。预计节省的阅读时间（以 200wpm 计算）：...</li><li><a href="https://arxiv.org/abs/2402.10588">Llama 是用英语工作的吗？论多语言 Transformers 的潜在语言</a>：我们探讨了在不平衡且以英语为主的语料库上训练的多语言语言模型是否使用英语作为内部中转语言——这对于理解语言模型如何...至关重要。</li><li><a href="https://bytez.com/read/arxiv/2402.10588">Bytez：Llama 是用英语工作的吗？论多语言 Transformers 的潜在语言</a>：在这项研究中，科学家们想知道语言模型（可以生成文本的模型）是否在内部使用英语作为“中转”语言，即使是在使用其他语言进行提示时。他们发现 ...</li><li><a href="https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d">Multilingual - stereoplegic 收藏集</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1769550950270910630?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Daniel Han (@danielhanchen) 的推文</a>：查看了 @Grok 的代码：1. Attention 按 30/tanh(x/30) 缩放？！ 2. 使用了类似 Gemma 的近似 GELU 3. 4 层 Layernorm，而不像 Llama 是 2 层 4. RMS Layernorm 在最后进行下转型，而不像 Llama...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1218137415068422164)** (2 条消息): 

- **论文俱乐部会议提醒**：关于“大型语言模型综合总结”的 **Paper Club** 会议将在两分钟后开始。欢迎大家加入并参与。

- **人工智能旋律生成器**：由 [Suno](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9) 驱动的 AI 创作了一首模仿 90 年代嘻哈风格的歌曲，其歌词反映了 AI 模型如何通过在海量数据集上进行训练来挑战人类艺术家。歌词对 AI 在创意领域日益增长的影响力进行了元评论（meta-commentary）。

**提及的链接**：<a href="https://news.ycombinator.com/item?id=39746163">Suno, an AI music generator | Hacker News</a>：未找到描述

---

**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1218135292574306328)** (20 条消息🔥): 

- **解码 Attention 的起源**：一位成员澄清说，**Attention 机制**的创建是为了解决模型访问输入序列中所有信息的问题，这与以往具有固定长度编码向量的模型不同。

- **并行化优势解析**：会议解释了 **Transformer Attention 机制**通过允许独立处理不同的 tokens 来促进并行化，从而实现更高效的计算和更快的训练。

- **计算效率的澄清**：一位成员表示，他们对**并行化**的困惑已得到解决，现在理解了 Transformer 中的缩放点积（scaled dot product）操作消除了处理过程中的顺序“等待”。

- **对见解的总结致谢**：参与者对本次会议表示感谢，会议让他们对 **Large Language Models (LLMs)** 的发展和背后的直觉有了更深入的理解。

---

**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1218287754715201638)** (36 条消息🔥): 

- **在 AI 俱乐部旁听**：一位俱乐部成员提到正在参加一个线下（IRL）会议，并正在旁听正在进行的讨论。
- **成员交流问候**：几位用户在聊天中互致简短的问候，显示出活跃的参与度。
- **即将发布深度博客**：一位用户暗示稍后将在其个人博客上发布他们讨论内容的详细版本。
- **分享实用的 RAG 技术资源**：一位用户分享了一个 [Towards Data Science 博客文章](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4)的链接，讨论了高级 RAG 技术。
- **引用协作学习文档**：提到了一个 [Google 表格](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)，详细列出了 AI-in-action 俱乐部会议即将讨论的主题和资源。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4">Advanced RAG 01: Small-to-Big Retrieval</a>：使用 LlamaIndex 的子父级递归检索器（RecursiveRetriever）和句子窗口检索（Sentence Window Retrieval）</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题、日期、主持人、资源、@dropdown GenAI 的 UI/UX 模式，2024/1/26，nuvic，&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-struct...
</li>
</ul>

</div>

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1218220293865345024)** (168 条消息🔥🔥): 

- **DALL-E 3 数据集位置更新**：曾被询问从 Huggingface 移除的一个数据集实际上已移动到新位置；DALL-E 3 数据集可以在[这里](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)访问。
  
- **关于通过 Huggingface 加载数据集**：重点介绍了如何通过 `load_dataset()` 函数指定 commit id 来加载数据集，[详见 Hugging Face 文档](https://huggingface.co/docs/datasets/en/loading#hugging)。

- **关于 Grok 模型及其与 Mixtral 对比的讨论**：有很多关于 Grok（一个 314B 参数模型）的讨论，包括对其相对于其庞大体量的性能看法，以及与较小但能力强大的 Mixtral 模型的对比。分享了 Grok-1 GitHub 仓库的链接，[点击此处查看](https://github.com/xai-org/grok-1)。

- **使用 Cog 模型进行 Captioning 的创新**：用户正在分享在使用 Cog 模型时，通过将元数据合并到 prompts 中来提高标注（caption）准确性的策略。一位用户分享了来自 GitHub 仓库的一个脚本，[在此处查看脚本](https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py)。

- **围绕 LLaMA 1.6 34b 模型和 COG VLM 的讨论**：用户正在讨论各种 AI 模型，特别是 LLaMA 和 COG，重点关注它们的标注能力、推理速度以及在 RTX 3090 等消费级 GPU 上的实际可用性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/imgn_ai/status/1769791182270333067">来自 imgnAI (@imgn_ai) 的推文</a>：猫娘们出现在 NVIDIA GTC ✨ 为你的创作自由而鸣 👊 这是一个需要被听到的消息 🐱💕</li><li><a href="https://tenor.com/view/silicon-valley-yes-cheer-think-gif-9010547">Silicon Valley Yes GIF - Silicon Valley Yes Cheer - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.economist.com/business/2023/11/23/why-chinese-companies-are-flocking-to-mexico">为什么中国公司正涌向墨西哥</a>：该国提供了进入美国的后门</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugg">Load</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugging-face-hub">Load</a>：未找到描述</li><li><a href="https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py">EveryDream2trainer/caption_cog.py at main · victorchall/EveryDream2trainer</a>：通过在 GitHub 上创建账号来为 victorchall/EveryDream2trainer 的开发做出贡献。</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/aiwars/comments/1bbxtp6/the_people_behind_the_nightshade_glaze_account/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset">OpenDatasets/dalle-3-dataset · Hugging Face 上的数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1218181910669295716)** (13 messages🔥): 

- **在错误的频道讨论了 Web UI**：成员们简要提到了使用 Web UI 的风险，指出它们不能在免费版的 Colab 中使用。
- **分享了生成式音频视频文本世界模型文档**：发布了一个标题为 "Generative Audio Video Text world model" 的 Google Doc 链接，但未提供有关文档内容的详细信息或进一步讨论。[在此访问文档。](https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk)
- **分享了关于 LLM 高效持续学习的论文**：分享了一个关于大语言模型（LLM）持续预训练的 arXiv 论文链接，讨论了在保留旧数据性能的同时适应新数据的方法。[阅读摘要或下载论文](https://arxiv.org/abs/2403.08763)。
- **Grok-1 在 GitHub 上开源发布**：分享了一个名为 Grok-1 项目的 GitHub 链接；这似乎是 xai-org 对该命名软件的开源发布。[在此查看 GitHub 仓库。](https://github.com/xai-org/grok-1)
- **关于 GPT-4 架构的推测**：关于 GPT-4 架构的讨论非常多，有说法引用 Nvidia 消息源称其为拥有 1.8 万亿参数的 MoE 模型。目前尚未确认这是 GPT-4 还是另一个模型。[查看推测性的推文图片。](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large)
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>：大语言模型（LLM）通常在数十亿个 token 上进行预训练，一旦有新数据可用，就必须重新开始整个过程。一种更有效的解决方案是持续预训练...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能多模态大语言模型（MLLM）。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细和持续的...</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk">Generative Audio Video Text world model</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1218118852454383667)** (99 messages🔥🔥): 

- **处理聊天格式和转换**：一次交流讨论了在聊天数据上进行训练的效果，重点在于是否保留聊天格式或转换为 Q/A 结构。该成员获知了关于使用 sharegpt 格式，并随后在训练期间通过 Axolotl 使用特定属性将其转换为 llama chat model 的方法。

- **Axolotl 简化模型 Finetuning 流程**：Axolotl 因其通过 yaml 文件而非编写脚本来简化 Finetuning 过程而受到关注，并且它支持使用 LoRA。尽管有人担心与传统方法相比会失去部分控制权，但 Axolotl 提供了一个用户友好且设置开销更少的替代方案。

- **NVIDIA RTX 5000 系列的前景**：
    - 传闻称即将于 2025 年左右发布的 NVIDIA RTX 5000 系列将有显著的性能提升，包括 VRAM 增加 50% 和带宽提升 78% 等显著改进，这可能有利于 AI 模型训练。
    - 分享了一个讨论 RTX 5000 系列的[新闻文章](https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html)链接，推测其对消费级训练的重要性。
    - 辩论了关于潜在显存配置的更多细节，并参考了 [TechPowerUp 文章](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed)中关于新系列 GDDR7 显存速度的见解。

- **量化技术与模型集成的比较**：讨论涉及了现有的量化方法（如 AQML）及其与各种模型的集成。分享了 AQML GitHub 仓库的链接 ([GitHub - AQLM](https://github.com/Vahe1994/AQLM))，并对其与其他方法的效率进行了评论。

- **探索 Grok-1、MoE 和推理策略**：
    - 社区对 Grok-1 模型权重的发布做出了反应，思考运行这一大规模 MoE 模型的能力和潜在硬件需求。讨论涉及了其 MoE 架构对 VRAM 占用和推理速度的影响，并提到 Sequoia 是优化消费级 GPU 推理的潜在解决方案。
    - 提出了关于 Offloading 或缓存系统作为处理 Grok-1 等模型巨大资源需求策略的可能性。
    - 通过该活动的 YouTube 链接 ([GTC March 2024 Keynote with NVIDIA CEO Jensen Huang](https://www.youtube.com/watch?v=Y2F8yisiS6E)) 分享了对 NVIDIA GTC 发布会的见解，包括关于 GPT-4 模型 1.8T 参数量的预告。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BrivaelLp/status/1769482175005577571?s=20">来自 Brivael (@BrivaelLp) 的推文</a>：Zuck 刚刚对 Grok 的发布做出了反应，他并没有留下深刻印象。“3140 亿参数太多了。你需要一堆 H100，而我已经把它们都买光了” 🤣</li><li><a href="https://www.together.ai/blog/sequoia">Sequoia：可扩展、鲁棒且硬件感知的 Speculative Decoding</a>：未找到描述</li><li><a href="https://tenor.com/view/wizard-cat-magus-cat-witch-cat-wicca-wiccan-gif-26941843">巫师猫 Magus 猫 GIF - 巫师猫 Magus 猫 女巫猫 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50 系列 "Blackwell" 将使用 28 Gbps GDDR7 显存速度</a>：据可靠爆料人 kopite7kimi 称，首批采用 GDDR7 显存的 NVIDIA GeForce RTX 50 系列 "Blackwell" 显卡传闻将拥有 28 Gbps 的显存速度...</li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">2024 年 3 月 GTC 主旨演讲，NVIDIA CEO 黄仁勋</a>：观看 NVIDIA CEO 黄仁勋的 GTC 主旨演讲，了解所有关于塑造我们未来的 AI 进展的发布。深入了解这些发布并发现...</li><li><a href="https://www.heise.de/news/GeForce-RTX-5000-Geruechte-zu-Nvidias-naechster-Grafikkartengeneration-9655220.html">GeForce RTX 5000：关于 Nvidia 下一代显卡的传闻</a>：Nvidia 的下一代大型游戏 GPU 可能会拥有更多、更快的显存，以及更多的 Shader 核心。</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok 开源发布</a>：Grok 开源发布。通过在 GitHub 上创建账号为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/Vahe1994/AQLM">GitHub - Vahe1994/AQLM：通过 Additive Quantization 实现大语言模型极端压缩的官方 Pytorch 仓库 https://arxiv.org/pdf/2401.06118.pdf</a>：通过 Additive Quantization 实现大语言模型极端压缩的官方 Pytorch 仓库 https://arxiv.org/pdf/2401.06118.pdf - Vahe1994/AQLM</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-s">NVIDIA GeForce RTX 50 系列 "Blackwell" 将使用 28 Gbps GDDR7 显存速度</a>：据可靠爆料人 kopite7kimi 称，首批采用 GDDR7 显存的 NVIDIA GeForce RTX 50 系列 "Blackwell" 显卡传闻将拥有 28 Gbps 的显存速度...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1218207901873606667)** (24 条消息🔥): 

- **ScatterMoE 优化 MoE 模型**：Axolotl 开发团队对 **ScatterMoE** 的潜力感到兴奋，这是对 Huggingface 实现的一种优化，有望显著提高吞吐量。更多细节和代码可以在他们的 [GitHub 分支](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe)上查看。

- **澄清使用 ScatterMoE 进行训练**：一位成员澄清说，要使用新的 ScatterMoE 进行训练，需要使用 **mixtral** 模型类型，但强调该系统尚未针对训练正确性进行彻底测试。

- **需要升级到 PyTorch 2.2**：讨论了需要将 **axolotl** 升级到 PyTorch **2.2** 或更高版本，以受益于新的 kernel 并解决兼容性问题；一些成员确认正在使用或测试 PyTorch **2.2.1**。

- **Grok 权重性能受到关注**：在 Axolotl 中尝试了 **Grok** 模型的权重（**3140 亿参数**），但一位用户评论说，相对于模型的大小，其性能并不理想。人们很好奇考虑到其资源需求，谁会运行这样的模型。

- **将 Grok 引入 Axolotl**：鉴于 **Grok** 模型发布的讨论，一位用户开玩笑说 Axolotl 的 **qLoRA FSDP** 是否最终能处理这个“怪物”，而另一位用户指出，根据 [Grok GitHub 页面](https://github.com/xai-org/grok-1)，目前仅发布了该模型的 **int8 版本**。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407">由 ehartford 实现 post training · Pull Request #1407 · OpenAccess-AI-Collective/axolotl</a>: 这样看起来对吗？</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407/commits/9c221a6761195c9739c02e11f9fe864bc947e53b">由 ehartford 实现 post training · Pull Request #1407 · OpenAccess-AI-Collective/axolotl</a>: 这样看起来对吗？</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok 开源发布</a>: Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe">GitHub - OpenAccess-AI-Collective/axolotl at scatter_moe</a>: 尽管提出 axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1218257987445981234)** (35 messages🔥): 

- **Tokenizer 混淆导致标签问题**：一位成员遇到了微调模型经常遗漏第一个 `<summary>` 标签或在插入时带有前导空格的问题。他们检查了 Tokenization 行为，预期得到 `<summary>` 但观察到的是 `▁<summary>`，并担心这可能是 Tokenizer 的问题。

- **本地模型与数据不匹配**：一位 LLM 新手希望调整其配置文件，以使用本地模型和训练数据，而不是从 Huggingface 拉取，这导致他们在路径规范方面经历了一系列尝试和错误，并面临 `HFValidationError` 问题。

- **训练数据对话导致混乱**：另一位成员在微调对话数据时正苦于“索引超出范围”错误，由于其数据集中的“role”数组为空，导致 `one_shot` 和 `alpaca` 等配置无法按预期工作。

- **Readme 救场解决配置困惑**：在处理上述问题时，他们被建议验证 Readme 中提到的 Prompt 策略，并发现数据集中空的 “from” 和 “value” 字段是导致问题的原因，通过映射额外的角色并忽略长度小于 2 的对话解决了该问题。

- **评估集大小不一致**：一个被标记的 Bug 显示，Axolotl 在 2-epoch 运行时称评估集太小无法进行样本打包（sample packing），但在 10-epoch 运行时却认为没问题，尽管评估集是独立的，不应随 epoch 数量变化。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1218770755920072767)** (8 messages🔥): 

- **NVIDIA NeMo Curator 介绍**：一位成员分享了 **NVIDIA NeMo-Curator** 的 [GitHub 链接](https://github.com/NVIDIA/NeMo-Curator)，这是一个可扩展的数据整理工具包。然而，目前还没有关于该工具包的进一步讨论或个人经验分享。
- **寻找特定的 Mistral FT**：有人询问是否有人拥有或了解在 *orca-math-word-problems-200k* 数据集和 *nvidia/OpenMathInstruct-1* 上同时进行过微调的 **Mistral** 模型，强调了对结合推理与代码能力的兴趣。
- **考虑使用 mergekit 进行模型合并**：针对是否可以使用合并工具来避免在海量数据集上单独训练 **Mistral** 的问题，另一位成员肯定了 mergekit 是一个不错的选择，前提是对话格式（chat formats）保持一致。
- **对合并时模型格式兼容性的好奇**：对话演变为关于是否可以微调一个模型的子部分以对齐对话格式的问题，展现了对模型合并策略中适应性的兴趣。

**提到的链接**：<a href="https://github.com/NVIDIA/NeMo-Curator">GitHub - NVIDIA/NeMo-Curator: 可扩展的数据整理工具包</a>：可扩展的数据整理工具包。通过在 GitHub 上创建账号来为 NVIDIA/NeMo-Curator 的开发做出贡献。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

duh_kola: 是否可以使用不同的 lora adapter 在另一个模型上进行 dpo？
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1218310691103178803)** (43 messages🔥):

- **聚焦光子学 (Photonics)**：一位成员分享了一个题为“新芯片突破：利用光和无线电波实现 1000 倍提升”的 [YouTube 视频](https://youtu.be/8ohh0cdgm_Y)，并提到了 Lightmatter 公司，该公司专注于研发光子计算机芯片，旨在更高效地为 AI 提供动力。
- **Asianometry 的光子学见解**：在关于光子技术的讨论中，一位成员推荐了来自 Asianometry 的两个教育视频（可在 [YouTube](https://www.youtube.com/watch?v=29aTqLvRia8) 和 [YouTube](https://www.youtube.com/watch?v=t0yj4hBDUsc) 观看），内容涵盖了硅光子学和用于神经网络的光网格。
- **PyTorch 的显式张量管理设计**：成员们对 PyTorch 中的显式张量内存管理进行了辩论，讨论了 TensorFlow 中隐藏内存复制所带来的复杂性。一个 [GitHub gist](https://gist.github.com/robieta/4c6e94f25a2ab87330bb6bd8026074a6) 展示了 TensorFlow 处理跨设备张量时的行为。
- **寻找最新的 GPU 设施**：建议使用 [RunPod](https://www.runpod.io/) 和 [Lambda Labs](https://lambdalabs.com/) 等云端 GPU 服务来对新一代 GPU 上的 kernel 操作进行 profiling，尽管成员们提到了在这些平台上进行 profiling 的权限问题。
- **GTC 2024 预示新前景**：NVIDIA CEO Jensen Huang 在 GTC 2024 上的主题演讲引发了关于 AI 模型和硬件未来的讨论，涉及拥有 1.8 万亿参数的 SOTA 模型以及配备 192GB HBM 的新 B100 硬件。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.cerebras.net/product-chip/">产品 - 芯片 - Cerebras</a>：未找到描述</li><li><a href="https://www.runpod.io/">以每小时 0.2 美元起的价格租用云端 GPU</a>：未找到描述</li><li><a href="https://www.youtube.com/live/Y2F8yisiS6E?si=g5MChTXs3a9gGykE">NVIDIA CEO Jensen Huang 主持的 2024 年 3 月 GTC 主题演讲</a>：观看 NVIDIA CEO Jensen Huang 的 GTC 主题演讲，了解塑造我们未来的 AI 进展的所有发布信息。深入了解这些发布并发现...</li><li><a href="https://lambdalabs.com/">GPU 云、集群、服务器、工作站 | Lambda</a>：用于深度学习和 AI 的 GPU 云、GPU 工作站、GPU 服务器和 GPU 笔记本电脑。提供 RTX 4090, RTX 3090, RTX 3080, RTX A6000, H100 和 A100 选项。预装 Ubuntu, TensorFlow 和 PyTorch。</li><li><a href="https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ">新芯片突破：利用光和无线电波实现 1000 倍提升</a>：立即获取 TypeAI PREMIUM！点击此处链接开始免费试用：https://bit.ly/Mar24AnastasiInTech 论文地址：https://www.nature.com/articles/s41586...</li><li><a href="https://lightmatter.co/">Lightmatter®</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=29aTqLvRia8">硅光子学：下一次硅革命？</a>：衷心感谢频道的好友、来自 MIT 的 Alex Sludds 建议了这个话题并为我提供了关键资源。在这里关注他：https://a...</li><li><a href="https://www.youtube.com/watch?v=t0yj4hBDUsc">在光网格上运行神经网络</a>：我要感谢 Alex Sludds 在帮助我研究和制作此视频方面所做的努力。在这里查看他的工作：https://alexsludds.github.io 链接：- The As...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1218241351582482493)** (7 条消息): 

- **推出 Triton 调试可视化工具 (Debugging Visualizer)**：一位成员宣布创建了一个可视化工具，旨在通过展示 load/store 的空间结构来简化 Triton 中的调试过程。该工具旨在辅助实现复杂函数，尽管消息中未提供可视化界面的预览。
- **新发布：用于学习和测试的 Triton Puzzles**：分享了一套 [Triton Puzzles](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing)，旨在为熟悉 GPU puzzles 的用户提供具有挑战性且兼具教育意义的体验。目前已知有两个 bug：偶尔出现的重复可视化和段错误 (segmentation faults)。
- **Triton 新手指南**：针对 Triton 学习资源的咨询，成员们建议除了官方教程外，新的 Triton Puzzles 也会有所帮助，并建议研究和注释社区中流行的 Triton kernel 以加深理解。
- **鼓励使用 Triton CPU 调试**：一位成员对在 CPU 上运行 Triton 的解释器 (interpreter) 表示热烈欢迎，强调这对于无法立即使用 GPU 的用户来说是一个非常有用的功能。
- **社区参与 Triton Puzzles**：社区成员表现出参与新 Triton Puzzles 的浓厚兴趣，认可其潜在的实用性，并有一位成员对文本进行了细微修正，建议为了清晰起见进行编辑。

**提到的链接**：<a href="https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing">Google Colaboratory</a>：未找到描述

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1218467001450627072)** (68 条消息🔥🔥)：

- **深入探讨 Warp Schedulers 和线程效率**：一位成员询问了如何配置 Warp Schedulers 的数量，并了解每个调度器在 CUDA 中可以控制多少个线程，以优化执行效率和占用率（occupancy），但消息中未提供具体的答案或资源。

- **寻求关于 Active Warps 的澄清**：一位成员询问了 CUDA 中 "active warp" 的定义，以及没有活跃线程的 Warp 是否仍可被视为 active。建议在练习中，"active warp" 应指至少包含一个活跃线程的 Warp。

- **解码 CUDA 中的内存管理器**：成员 [@morousg#cudapassion](https://link.to.morousgprofile) 澄清了在 CUDA 中提供多种内存管理选项的意图，强调了如 "Producer Provides" 和 "Consumer Takes" 等策略，以促进不同内存空间之间的高效数据管理。

- **理解内存管理中的 Provide-Take 语义**：成员们就使用 "Produces" 和 "Takes" 时的内存管理语义进行了详细讨论，探讨了这些选项如何影响内存分配，以及在 CUDA 应用程序中可能需要的 streamSynchronization。

- **对流水线并行推理的内存管理表现出浓厚兴趣**：在一次演讲结束时，一位成员表示对应用内存管理策略来改进大语言模型 (LLM) 推理的流水线并行（pipeline parallel）实现深感兴趣，并与 [@morousg#cudapassion](https://link.to.morousgprofile) 讨论了潜在的解决方案，包括异步拷贝和优化 GPU 利用率。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">2024 年 3 月 GTC 主旨演讲，NVIDIA CEO 黄仁勋 (Jensen Huang)</a>：观看 NVIDIA CEO 黄仁勋的 GTC 主旨演讲，了解塑造我们未来的 AI 进展的所有发布。深入了解这些发布并发现...</li><li><a href="https://github.com/tspeterkim/flash-attention-minimal">GitHub - tspeterkim/flash-attention-minimal：约 100 行 CUDA 代码实现 Flash Attention（仅前向传播）</a>：约 100 行 CUDA 代码实现 Flash Attention（仅前向传播） - tspeterkim/flash-attention-minimal
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1219091487455711414)** (5 条消息)：

- **探索硬件与 ML 的交汇点**：一位用户分享了康奈尔大学 [Prof. Mohamed Abdelfattah 研究小组](https://www.youtube.com/@mabdelfattah88) 的 YouTube 链接，该小组专注于可重构计算和高效机器学习。
- **深入探讨针对硬件优化 ML**：[ECE 5545 (CS 5775) 课程](https://abdelfattah-class.github.io/ece5545/) 页面被重点推荐，该课程提供了从微控制器到多 GPU 系统的机器学习系统及其优化的硬件视角。
- **课程教科书之谜**：
  - 一位用户指出课程网站提到“教科书”但未说明具体是哪一本，这很奇怪。
  - 另一位用户澄清说，教科书的详细信息在第一个讲座视频中提供。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://abdelfattah-class.github.io/ece5545/">ML 硬件与系统</a>：未找到描述</li><li><a href="https://www.youtube.com/@mabdelfattah88">Prof. Mohamed Abdelfattah</a>：这是康奈尔大学 Prof. Mohamed Abdelfattah 研究小组的频道。我们正在研究可重构计算和高效机器学习。欲了解更多信息，请查看...
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 条消息)：

vim410：取决于具体情况。但确实如此。

---

**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1219389682241110147)** (5 条消息)：

- **扎实的 CUDA 技能作为 ML 的基础**：该成员在用于 GPU 计算的 CUDA 方面拥有深厚背景，包括内存合并 (memory coalescing)、线程束分歧 (warp divergence) 和 kernel profiling 的经验，这似乎是转向使用 CUDA 进行机器学习的**坚实基础**。
- **深入 ML/DL 的建议**：建议开始尝试使用像 **PyTorch** 这样的深度学习框架，因为 ML 本质上涉及优化技术，例如矩阵乘法和归一化。
- **《Programming Massively Parallel Processors》——必读书籍**：推荐使用名为 **"Programming Massively Parallel Processors"** 的特定书籍来深化 CUDA 知识，并被赞誉为极佳的资源，尽管书中关于深度学习的内容较少。
- **向泰斗学习**：提到跟随 **Andrej Karpathy 的 Zero to Hero 系列**是学习 ML 概念以及探索专注于 CUDA 讲座的良好路径。

**提到的链接**：<a href="https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311">未找到标题</a>：未找到描述

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1218146385942286407)** (6 条消息): 

- **理解 CUDA 索引中的跨步乘法**：一位成员最初对第 2 章第 2 题中用于 CUDA 索引的 `i = blockIdx.x * blockDim.x + threadIdx.x * 2` 表示疑问。另一位成员解释说，这种方法会导致索引 `i` 的重复计算，并举例说明了两个不同的线程会产生相同的索引值。
- **建议谨慎分享教师内容**：一位成员担心某些内容可能仅限教师使用。这是针对讨论在博客上发布练习答案是否合适而提出的。
- **博客发布练习答案：一个两难境地**：一位成员表示打算在博客上发布练习答案，因为作者没有回应，并强调了没有教育机构联系地址进行沟通的困境。
- **等待作者关于分享答案的指导**：有人建议，在博客上发布练习答案是否合适尚不确定，将进一步寻求 Wen-mei（推测是相关内容的作者或权威人士）的指导。

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1218239914542366790)** (14 条消息🔥): 

- **成员本周很忙**：一位成员简要表示他们本周**非常忙**，等时间充裕后会更新。
- **寻找代码**：一位用户正在寻找代码，并在 GitHub 上找到了一个 **Triton kernel**，并提供了 **[Ring-Flash-Attention commit](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5)** 的链接。
- **博客文章难题**：一位正在撰写关于 **ring-attention** 博客文章的成员寻求澄清：为什么相关论文中提到内存需求随块大小 (block size) 线性扩展，尽管 SRAM 中需要平方级别的分块大小 (chunk size) 内存。
- **寻找答案**：针对内存扩展的困惑，另一位成员建议查看 **[flash-attention 源代码](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)**，特别是 FlashAttention 是如何在不形成大小为 c^2 的矩阵的情况下实现的。
- **关于内存需求表述的澄清**：其他成员加入了讨论，其中一人建议内存需求可能是指随块的数量线性扩展，而不是随块大小本身扩展。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2311.09431">Striped Attention: Faster Ring Attention for Causal Transformers</a>：为了帮助满足 Transformer 模型对超长序列长度日益增长的需求，Liu 等人最近提出了 Ring Attention，这是一种能够克服单设备内存限制的精确注意力算法...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5">add naive triton kernel for varlen · zhuzilin/ring-flash-attention@10d992c</a>：未找到描述
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1218332053032927322)** (5 条消息):

- **MLSys 2024 Conference Alert**：一位成员分享了关于 5 月举行的 MLSys 2024 会议的信息，强调其关注 Machine Learning 与 Systems 交叉领域的跨学科协作。该会议被认为在 AI 时代具有重要意义，特别是在开发高效 AI 系统方面。[查看会议](https://mlsys.org/)。
- **会议标语的诗意视角**：有人观察到“The Conference for the Era of AI”这一短语符合五步抑扬格（iambic pentameter）的节奏。
- **智能手机的烦恼**：一位用户幽默地将智能手机称为“不那么智能的手机（Not so smart phone）”，可能暗示对设备存在某些挫败感或问题。
- **数学运算顺序澄清**：讨论中纠正了数学表达式中的运算顺序，强调乘法和除法应从左到右进行。
- **科学计算器争论**：关于数学运算的对话延伸到了科学计算器如何以不同方式解释某些表达式，表明计算结果可能因计算器设计而异。

**提到的链接**：<a href="https://mlsys.org/">MLSys 2024</a>：未找到描述

---

**CUDA MODE ▷ #[gtc-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1218444664315711498)** (9 messages🔥): 

- **早起的鸟儿有虫吃**：*marksaroufim* 提到计划从周一早上开始参加活动，并愿意进行线下聚会，提供私信电话号码以便协调。
- **长期参会者**：*neurondeep* 表示他们将在 3 月 14 日至 25 日参加 GTC，并计划全程参与。
- **聚会爱好者**：*_t_vi_* 表达了他们在现场，并有兴趣与其他成员见面。
- **行程排满**：*marksaroufim* 原计划参加 1-2 天 GTC，但受精彩议程和良好 Wi-Fi 的吸引，决定留满整周。
- **GTC FOMO**：*mr.osophy* 幽默地表达了无法参加 GTC 的遗憾，以及过去曾尝试申请志愿者以换取免费门票但失败的经历。

**提到的链接**：<a href="https://www.youtube.com/watch?v=Sfrjpy5cJCs">I Snuck Into A Secret Arms-Dealer Conference</a>：每月在 https://www.patreon.com/Boy_Boy 获取独家视频。这是我们与传奇的澳大利亚政治讽刺团体 The C... 合作制作的。

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1218183723200155748)** (159 messages🔥🔥): 

- **Llama 消息格式获批**：一位用户询问包含 "system"、"user" 和 "assistant" 的格式是否适用于 Llama 模型，得到了肯定的答复。
- **探索支付方式**：当被问及是否必须绑定信用卡以及如何支付时，明确了用户需要进行 *topup their balance*（余额充值）。
- **讨论角色扮演一致性的模型选择**：用户讨论了哪种模型在角色扮演中表现最好且不会重复或胡言乱语，**Sonnet** 最终因其一致性被推举为首选。
- **模型的提示词格式指南**：在询问如何使用系统消息引导 LLM（除第一条消息外）后，成员们讨论了局限性，指出通常只有第一条系统消息生效，后续指令可能需要嵌入在用户消息中。
- **开发意图与关联**：用户讨论了从设置公共 API、在平台列出，到联盟计划和模型选择等多样化话题，同时讨论了各种模型的成本和效率，以及 OpenRouter API 的灵活性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai">OpenRouter</a>：LLM 和其他 AI 模型的路由服务</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过创建账户为 xai-org/grok-1 的开发做出贡献。
</li>
</ul>

</div>

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1218212402127175711)** (95 messages🔥🔥):

- **LangChain Agent 的 API 选择**：一位成员询问 **astream_log** 是否优于 **astream_events**，以及后者处于 beta 阶段是否意味着即将被弃用，或者它们仅仅是不同的 API。
- **研究助手招募 Beta 测试人员**：一位成员正在为其构建的高级研究助手和搜索引擎招募 Beta 测试人员。该工具提供对 **Claude 3 Opus**、**GPT-4 Turbo** 和 **Mistral Large** 等模型的付费访问权限。感兴趣的人员可前往名为 **Rubik's AI** 服务的 [候补名单页面](https://rubiks.ai/)。
- **LangChain 文档的协作与反馈**：几位成员表示 **LangChain 文档**（特别是针对初学者的部分）难以导航，其他成员则提议帮助澄清内容或补充缺失的页面。
- **使用 LangChain 进行结构化输出解析**：成员们讨论了如何结合 **LangChain 和 pydantic** 获取结构化输出，并提供了解析复杂数据结构的代码示例。用户分享了代码片段，并向尝试在项目中实现类似功能的其他人提供帮助。
- **新服务 Beta 测试呼吁**：一位成员正在为一项新服务招募 Beta 测试人员，该服务为应用程序或个人文档提供快速访问生成器（RAG），并承诺将进行为期一周的密集开发以完成平台建设。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>: 未找到描述</li><li><a href="https://codelabs.developers.google.com/codelabs/gemini-function-calling#4.">未找到标题</a>: 未找到描述</li><li><a href="https://bloon.ai">Bloon AI</a>: 重新定义智能学习</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19239">功能请求：在相似度搜索中支持负向嵌入 (Negative Embeddings) · langchain-ai/langchain · Discussion #19239</a>: 已检查，我搜索了现有想法，未发现类似想法。我添加了一个非常详细的标题，并清楚地描述了功能请求及其动机。功能请求：我建议增加...</li><li><a href="https://www.teradata.com/insights/ai-and-machine-learning/using-natural-language-to-query-teradata-vantagecloud-with-llms">使用自然语言通过 LLM 查询 Teradata VantageCloud | Teradata</a>: 学习将您的英语查询翻译成 SQL，并从您的分析数据库中接收纯英语的响应。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1219304272244510741)** (45 条消息🔥): 

- **RemoteRunnable 的流式传输问题**：一位成员在 LangChain 中使用 `RemoteRunnable` 的**流式输出时遇到问题**。该成员指出，从 Python 调用时流式传输正常工作，但**等效的 JavaScript 代码总是触发 `/invoke` 调用**而不是 `/stream`。

- **流式序列中潜在的继承问题**：该成员质疑问题是否源于 `RunnableSequence` 从 `Runnable` 继承了默认的 `_streamIterator`，从而触发了 `invoke` 调用。该成员认为这可能导致 JavaScript 中的流式传输功能失败。

- **寻求 LangChain 团队的帮助**：当被问及如何向 LangChain 团队报告此问题时，AI 指示在 **GitHub** 上提交 issue 或**发送电子邮件给团队**以寻求支持。

- **未发现近期更改**：**未提及**过去一个月内有任何可能解决 JavaScript 流式传输问题的近期更改。建议成员查看 LangChain 的 GitHub 提交记录和发布说明以获取更新。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://api.js.langchain.com/classes/langchain_core_runnables_remote.RemoteRunnable.html#pipe>):">RemoteRunnable | LangChain.js - v0.1.28</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/security#reporting-a-vulnerability>).">Security | 🦜️🔗 Langchain</a>: LangChain 拥有庞大的集成生态系统，可与本地和远程文件系统、API 及数据库等各种外部资源进行集成。这些集成允许开发者创建多功能的应用程序...</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11998>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13723>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17315>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1218223379690029179)** (11 条消息🔥): 

- **对话式数据分析 AI 聊天机器人发布**：Haste171 发布了一个 [GitHub 项目](https://github.com/Haste171/langchain-chatbot)，其特色是使用 AI 聊天机器人以对话格式分析和从数据中提取信息。

- **AI 让书签焕发生机**：Codegriot 创建了一个 Discord AI 聊天机器人，用于管理 Raindrop.io 书签，目标是方便日后查找相关内容。该机器人以开源形式提供，可在 [GitHub](https://github.com/uogbuji/living-bookmarks) 上获取。

- **AI 网页抓取变得更简单**：VinciGit00 使用 langchain 开发了一个基于 AI 的爬虫，它使用 OpenAI 密钥运行，并计划兼容其他模型。在不到一个月的时间里安装量已超过 2300 次，他们鼓励通过在 [GitHub 仓库](https://github.com/VinciGit00/Scrapegraph-ai) 点赞（star）来支持。

- **个性化营养 AI 应用展示**：Esxr_ 分享了一段 [YouTube 视频](https://youtu.be/vHjc5CEoIJE)，演示了 Nutriheal，这是一款用于个性化患者护理的 AI 应用，利用工具进行本地托管并保护数据隐私。更多见解可在其网站 [navvy.co](https://navvy.co/) 上获得。

- **AI 驱动的销售开发代表 (SDR)**：Sivasurend 接受了一项 Twitter 挑战，使用 Lyzr Automata 框架自动化 SDR/AE 的角色。详细方法已在 Twitter 上演示，源代码可在其 [GitHub 页面](https://github.com/LyzrCore/lyzr-automata) 获取。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://calendly.com/neurofusion/30min">用户访谈 🔎 - NEUROFUSION Research, Inc.</a>：嘿，我正在构建一个数字顾问，旨在帮助改善你在工作和生活其他领域的表现。我很想与你交流，了解你在生产力、身体素质等方面的需求...</li><li><a href="https://medium.com/@bsouleymane78/staying-up-to-date-with-latest-advancements-on-ai-applied-to-financial-industry-using-ai-b995da14800f">利用 AI 紧跟 AI 在金融行业应用的最新进展</a>：利用 AI 自动化分析最新的科学论文，以关注该领域的最新进展。</li><li><a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: 用于以对话形式分析/提取数据信息的 AI 聊天机器人。</a>：用于以对话形式分析/提取数据信息的 AI 聊天机器人。 - Haste171/langchain-chatbot</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: 基于 AI 的 Python 爬虫</a>：基于 AI 的 Python 爬虫。通过在 GitHub 上创建账号来为 VinciGit00/Scrapegraph-ai 的开发做出贡献。</li><li><a href="https://youtu.be/vHjc5CEoIJE">在 15 分钟内制作一个 AI 应用</a>：技术栈 - 自定义 UI 和 RAG：Open-webui 的调整版本 - 本地 LLM 托管：用于本地托管 LLM 的 Ollama。- 数据隐私：集成了 DaxaAI 的 Pebblo 以...</li><li><a href="https://navvy.co/.">首页</a>：我对 AI 充满热情。让我们建立联系，释放 AI 的潜力，并在创新项目上进行合作！</li><li><a href="https://x.com/siva_1gc/status/1768997890544800070?s=20">来自 Siva Surendira (@siva_1gc) 的推文</a>：这比我们预想的要多花一点时间.. 但它来了.. 😎 使用 @lyzrai Automata 和 @OpenAI 实现 SDR 和 AE 功能的自动化... 运行在 @awscloud 上 - 安全且私密.. 它是如何工作的？👇 Agent 1:...</li><li><a href="https://github.com/LyzrCore/lyzr-automata">GitHub - LyzrCore/lyzr-automata: 低代码多 Agent 自动化框架</a>：低代码多 Agent 自动化框架。通过在 GitHub 上创建账号来为 LyzrCore/lyzr-automata 的开发做出贡献。</li><li><a href="https://amzn.eu/d/3Dcdsbk">Die Reise vom Ego zur Seele in einem holistischen Universum: Die Rolle der Meditation, der Naturerfahrung und der Astronomie bei der Transformation (10.000 Follower TikTok Content dank ChatGPT 2) eBook : Schulze, Carsten, Bing, chatgpt, google, Bard: Amazon.de: Kindle-Shop</a>：未找到描述</li><li><a href="https://amzn.eu/d/2uVnCp8">未找到标题</a>：未找到描述</li><li><a href="https://www.facebook.com/casi.schulze.10">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1218824643436085321)** (2 条消息): 

- **AI 应用开发变得轻而易举**：一位成员展示了一个名为 *Nutriheal* 的个性化营养 AI 应用的创建过程，该应用利用了 **Ollama**、**Open-webui** 以及 Daxa AI 的 **Langchain Pebblo 集成**。该成员通过一段 [教程视频](https://youtu.be/vHjc5CEoIJE) 强调了创建 AI 应用的简便性，并在 [navvy.co](https://navvy.co/) 分享了他们的作品集。

- **本地 AI 部署揭秘**：关于如何在本地设置和运行复杂 AI 模型的教程打破了 AI 专属于科技巨头的神话，正如 [在本地构建和部署 GenAI 解决方案](//build-and-deploy-genai-solutions-locally) 和 [本地 LLMs - 为自定义 LLM 助手制作通用 UI](/generic-ui-for-custom-llm-assistants) 等博客文章所展示的那样。

- **使用 Langgraph 进行计划与执行**：分享了一个视频教程，演示了如何使用 **Langgraph** 创建一个“计划与执行”风格的 Agent，灵感来自 Plan-and-Solve 论文和 Baby-AGI 项目。观众可以通过提供的 [YouTube 视频](https://www.youtube.com/watch?v=ZlJbaYQ2hm4) 进行观看和学习。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">使用 Langgraph 进行计划与执行</a>：如何创建一个“计划与执行”风格的 Agent。这在很大程度上受到了 Plan-and-Solve 论文以及 Baby-AGI 项目的启发。核心思想是首先...</li><li><a href="https://youtu.be/vHjc5CEoIJE">在 15 分钟内制作一个 AI 应用</a>：技术栈 - 自定义 UI 和 RAG：Open-webui 的调整版本 - 本地 LLM 托管：用于本地托管 LLM 的 Ollama。- 数据隐私：集成了 DaxaAI 的 Pebblo 以...</li><li><a href="https://navvy.co/.">首页</a>：我对 AI 充满热情。让我们建立联系，释放 AI 的潜力，并在创新项目上进行合作！
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1218217772765544448)** (8 条消息🔥):

- **API 泄露 LLM 机密**：[arXiv](https://arxiv.org/abs/2403.09539) 上的一篇论文揭示了闭源大语言模型 (LLMs) 可能会通过其 API 输出泄露大量信息。这种泄露归因于 softmax 瓶颈，使得在 OpenAI 的 gpt-3.5-turbo 案例中，能以“低于 1,000 美元”的成本发现模型架构细节。
  
- **关于 LLM 参数量低估的辩论**：一位成员对近期论文中讨论的模型估计为 70 亿参数表示惊讶，认为实际参数量可能更高。

- **对模型参数量估计的怀疑**：随着一些成员暗示 LLM 参数量估计可能不正确，怀疑也随之产生，特别是如果所讨论的模型（假设是 GPT 3.5）采用了 Mixture of Experts (MoE) 架构。

- **推测混合模型机制**：一段对话推测了 turbo 模型中可能存在的机制，并将其与过去的一篇论文类比，该论文指出使用较大模型的起始 token 可以提升较小模型的后续性能。
  
- **模型性能的复杂性**：有人指出，另一个 LLM *"Mixtral"* 具有极高的嵌入维度 (4096)，这表明了该领域复杂的本质以及可能存在的性能增强手段。

**提到的链接**：<a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>：大语言模型 (LLMs) 的商业化导致了仅通过高级 API 访问闭源模型的常见做法。在这项工作中，我们展示了即使在保守的假设下……

---

**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1219339209270362135)** (19 messages🔥): 

- **预见关于开源定义的争议**：@rasbt 的一条推文暗示了未来可能围绕什么是“开源”展开辩论，根据 @natolambert 的消息，这可能会引发争议。参与者们正热切期待 Open Source Software (OSS) 社区的官方立场。
- **在开源领域寻求共识**：社区讨论了建立对“开源”*共同理解*的重要性。*Apache 2.0* 和 *GPLv3* 等广泛的许可证展示了其中的复杂性。
- **尝试制定实际的开源定义**：@natolambert 表示打算创建一个*实际的定义*来澄清开源辩论，可能是为了避免混淆并解决分歧。
- **对在线争论感到沮丧**：@natolambert 表达了对与用户 @eluether 在线互动和讨论的沮丧，并选择当天退出 Twitter。
- **关于博客 vs 推文与 AI 治理**：@natolambert 反思了暂时离开 Twitter 的好处，并认为博客是一种更充实的媒介。此外还提到了关于 OpenAI 董事会成员 Helen Toner 资格的冲突观点。

**提到的链接**：<a href="https://x.com/BlancheMinerva/status/1769792488091353099">Stella Biderman (@BlancheMinerva) 的推文</a>：@natolambert @felix_red_panda 你错了哦 :P

---

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1219005089826607185)** (63 messages🔥🔥): 

- **Grok-1 模型权重向公众发布**：xAI 宣布在 Apache 2.0 协议下发布 [Grok-1](https://x.ai/blog/grok-os) 的基础模型权重和架构，这是一个 *314B 参数的 Mixture-of-Experts 模型*。该模型使用 Rust + JAX 的自定义堆栈训练，可在 [github.com/xai-org/grok](https://github.com/xai-org/grok) 获取。
  
- **Grok 的参数量令社区感到惊讶**：聊天参与者对 Grok-1 的规模表示震惊，作为一个 **Mixture-of-Experts 模型拥有 3140 亿参数**，这表明 xAI 团队在快速发布计划中优先考虑了最优性。

- **Grok 性能讨论**：聊天中提到了性能表现，参考资料显示 Grok 的表现优于 Falcon，其 **GSM8K 为 45.94，MMLU 为 70.5**。关于大规模训练数据集以及 [Chinchilla 研究如何应用于 MoE](https://x.com/thexeophon/status/1769449427972858103?s=46) 的推测也随之产生。

- **Grok 的种子分发引发轰动**：通过种子 (Torrent) 分发 Grok 权重的做法引发了关于开源模型声誉影响以及这可能带来的政策挑战的讨论。

- **FedEx 模型交付笑话**：有人提出了一个幽默的想法，即通过 FedEx 用闪存盘分发 AI 模型，作为应对昂贵的云端流出费用 (cloud egress fees) 的一种具有成本效益的措施。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.wheresyoured.at/peakai/">我们是否已经达到了 AI 的巅峰？</a>：上周，《华尔街日报》发布了对 OpenAI CTO Mira Murati 长达 10 分钟的采访，记者 Joanna Stern 提出了一系列深刻而直接的问题，Murati...</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://fxtwitter.com/grok/status/1769441648910479423">来自 Grok (@grok) 的推文</a>：@elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░</li><li><a href="https://x.com/thexeophon/status/1769449427972858103?s=46">来自 Xeophon (@TheXeophon) 的推文</a>：Chinchilla 定律并不直接适用于 MoE，对吧？如果是的话，我们可以推断出 Grok 的训练数据集大小。它出乎意料地大，所以我猜他们首先追求的是最优性，考虑到时间有限...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1218732428462395502)** (6 条消息): 

- **与 Alignment Lab 一起探索 Airbus**：一位成员分享了 Alignment Lab 关于 **Airbus** 的[推文链接](https://twitter.com/alignment_lab/status/1758949148143841379)，但觉得内容令人困惑，询问其他人正在用它构建什么。
- **寻找基于 HTTP 训练的 Embeddings 模型**：一位用户询问是否存在**在 HTTP 响应上训练的 embeddings 模型**，并寻求如何找到此类模型的建议。他们还认为任何经过适当训练的 Transformer 都可以作为 embeddings 模型。
- **为 Mistral 合并数据集**：有人询问是否存在**同时在 orca-math-word-problems-200k 数据集和 nvidia/OpenMathInstruct-1 上进行微调的 Mistral 模型**，想知道其他人是否了解或拥有此类模型。
- **问候**：一位用户简单地说了声 "hi"。
  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1219081302683422851)** (32 条消息🔥): 

- **征集 Grok 1 微调合作者**：一位成员表达了对微调 **Grok 1** 的兴趣并寻求协助，强调了对大量计算资源和专业知识的需求。
- **MoE 基础设施的效率**：一位成员声称拥有**高效的 MoE 训练基础设施**，效率接近 100%，可能对 Grok 1 微调项目有益。
- **Grok 1 的计算和数据需求**：微调 Grok 1 所列出的需求包括 **64-128 个 H100 GPU**、一个大型经过验证的数据集，以及大量的实验时间投入。
- **对 Grok 1 性能的怀疑**：有人对 **Grok 1 的性能**表示担忧，特别是与其他模型（如 Mixtral）相比，并就投入额外训练是否值得展开了辩论。
- **Grok 1 能力的亮点**：尽管存在疑问，一位成员分享了一个 [HuggingFace 数据集链接](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam)，表明 **Grok 1** 展示了令人惊讶的能力，在匈牙利国家高中毕业考试数据集上的表现接近 GPT-4 和 Claude。

**提到的链接**：<a href="https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam">keirp/hungarian_national_hs_finals_exam · Hugging Face 数据集</a>：未找到描述

  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1218226914322415677)** (1 条消息): 

由于此处仅提供了一条上下文不完整的单条消息，因此无法生成摘要。如果您提供更多该频道的历史消息，我将能够为您汇总所需的摘要。
  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1218206756031955006)** (7 条消息): 

- **辩论 Anthropic 的意图**：一位成员分享了一条[推文](https://x.com/tszzl/status/1768530219378631137?s=20)，暗示 **Anthropic** 扮演着受控反对派的角色，旨在让技术人员产生恐惧。
- **对内容审核的担忧**：一位成员指出，内容审核问题主要出现在包含人物的图像上，系统会“直接拒绝”进行有效的审核。
- **思考 Claude Sonnet 的可扩展性**：讨论了将 **Claude Sonnet** 用于每月数千万 token 项目的可行性；寻求在大规模使用 Claude Sonnet 方面的担忧或经验。

**提到的链接**：<a href="https://x.com/tszzl/status/1768530219378631137?s=20">来自 roon (@tszzl) 的推文</a>：Anthropic 是受控反对派，目的是让技术人员心生敬畏。

  

---


**LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1218241222347460619)** (16 条消息🔥):

- **Maisa 发布 KPU**: Maisa 宣布了其全新的 Knowledge Processing Unit (KPU)，它与 LLM 集成以增强复杂任务的解决能力，潜力可能超越 GPT-4 和 Claude 3 Opus。[白皮书和博客](https://maisa.ai/blog/kpu) 详细阐述了 KPU 的架构及其在推理方面的优势。

- **关键对比缺失**: 一位成员指出，在 KPU + GPT-4-turbo 与 GPT-4 之间进行的对比（未包含 GPT-4-turbo 本身）可能不具代表性，建议合理的对比应包含后者。

- **对 KPU 创新的不确定性**: 人们对 KPU 的底层技术表示了一些困惑，有人认为它可能涉及复杂的 Prompt Engineering 或 Context Window 操作。

- **对图表和候补名单的怀疑**: 成员们开玩笑说，AI 初创公司展示令人印象深刻的图表并提供 Waitlist 是典型的发布模式，在没有更多实质性证据的情况下表达了怀疑态度。

- **考虑 KPU 可能存在的缺点**: 尽管在 Benchmark 任务中可能有性能提升，但人们对 KPU 潜在的 Latency 问题及其对实际应用的影响表示担忧。

- **关于 KPU 机制的进一步见解**: 来自 @davipar 的一条推文澄清说，KPU 是一种新的 AI 架构，可与现有的 LLM 配合使用，无需 Chunking 或 Embedding，并将其比作知识管理的 GPU。技术概览包括一个用于 Benchmark 的 Notebook，他们还提供 API Key 用于独立评估：[推文链接](https://x.com/davipar/status/1768683151780683919?s=20)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI 驱动的知识处理平台。一个用于执行业务任务的简单 API。为软件和应用程序开发人员抽象了使用最新 AI 架构的复杂性。</li><li><a href="https://x.com/davipar/status/1768683151780683919?s=20">David Villalón (@davipar) 的推文</a>: 很高兴回答！它不是一个新模型，事实上 KPU 与智能提供商（OpenAI, Anthropic...）无关。它是一种与 LLM 配合使用的新 AI 架构，利用了它们的推理能力...
</li>
</ul>

</div>
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 messages): 

res6969: https://x.com/leopoldasch/status/1768868127138549841?s=46
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1218132499150934157)** (21 messages🔥): 

- **DiscoLM 模型在德语方面表现不佳**: 用户报告了不同 **DiscoLM** 和 **LeoLM** 模型的问题，特别是 **DiscoLM-mixtral-8x7b-v2** 在指令微调（Instruction Fine-tuning）后无法生成德语响应。他们在尝试为序列分类任务微调 DiscoLM 模型时还遇到了 **ValueError**。

- **排查 DiscoLM API 调用问题**: 一位用户在通过 `vllm` 封装 DiscoLM API 调用时遇到问题，即使使用英语提示，服务器也会返回德语响应。他们提供了服务器设置及模型调用方式的详细代码片段。

- **德语模型 Benchmark 的不一致性**: 一位用户观察到德语模型的性能参差不齐，并强调了对 Chat Format 模板和 End Token 约定的敏感性。他们指出，社区在模板标准化和 Benchmark 方面的协作将大有裨益。

- **关于德语语言建模和 Benchmark 的讨论**: 用户讨论了缺乏用于测试德语语言建模中语言细微差别的高质量 Benchmark，并引用了最近的论文和测试。他们表示需要一个衡量语言输出质量的 Benchmark，并指出了数据集质量和模型合并（Merging）方面持续存在的问题。

- **与学术界合作开发 Benchmark**: 有人建议可以联系拥有计算资源和相关研究兴趣的大学，共同开发评估德语模型语言质量的 Benchmark。用户幽默地暗示了私下获取或合作开发此类 Benchmark 的可能性。

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_didnt_found_a_better_small_german_llm_anyone/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_did">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/xai-org/grok/blob/main/model.py">grok-1/model.py at main · xai-org/grok-1</a>: Grok 开源发布。通过在 GitHub 上创建账号，为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/xai-org/grok/blob/e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8/model.py#L294">grok-1/model.py at e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8 · xai-org/grok-1</a>: Grok 开源发布。通过在 GitHub 上创建账号，为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/">我们的论文《SuperGLEBer: 德语语言理解评估基准》被 NAACL 2024 接收</a>: 在我们的论文中，我们为德语构建了一个广泛的 Natural Language Understanding 基准测试套件，并据此评估了大量现有的具备德语能力的模型，以创建一个...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ChuckMcSneed/WolframRavenwolfs_benchmark_results">ChuckMcSneed/WolframRavenwolfs_benchmark_results · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/KLUE-benchmark/KLUE">GitHub - KLUE-benchmark/KLUE: 📖 韩语 NLU 基准</a>: 📖 韩语 NLU 基准。通过在 GitHub 上创建账号，为 KLUE-benchmark/KLUE 的开发做出贡献。</li><li><a href="https://github.com/facebookresearch/belebele">GitHub - facebookresearch/belebele: Belebele 数据集仓库，这是一个大规模多语言阅读理解数据集。</a>: Belebele 数据集仓库，这是一个大规模多语言阅读理解数据集。 - facebookresearch/belebele</li><li><a href="https://github.com/google-research/xtreme">GitHub - google-research/xtreme: XTREME 是一个用于评估预训练多语言模型跨语言泛化能力的基准，涵盖了 40 种类型各异的语言，并包含 9 个任务。</a>: XTREME 是一个用于评估预训练多语言模型跨语言泛化能力的基准，涵盖了 40 种类型各异的语言，并包含 9 个任务。 - goo...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1218111377495949322)** (4 条消息): 

- **Demo 在标准设置下运行**：一位成员确认 Demo 没有使用特殊设置，并基于 **fastchat/vllm** 运行以进行展示。
- **服务器历险记——从厨房到混乱**：托管 Demo 的服务器从家庭厨房环境搬迁到了更专业的场地，导致了意想不到的网络问题，希望能下周初解决。
- **致谢支持**：一位成员对有关模型训练和 Prompt 遵循能力的指导表示感谢。
- **业余设置的可靠性**：一位成员调侃了技术设置的讽刺之处：业余服务器稳如泰山，而专业托管的服务器却遇到了各种问题。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1218229369680695428)** (20 条消息🔥): 

- **Explosion 开发的 Prompt Engineering 工具**：一位成员提到了他们过去在 Explosion 开发 Prompt Engineering 工具的工作，这些工具已被整合到 Prodigy 产品中，即 [Prodigy 的 Prompt Engineering 工具](https://prodi.gy/features/prompt-engineering)。他们赞同将 Prompt Engineering 转化为数据标注任务的概念。

- **用于实验的 PromptTools**：另一位成员提到了 [PromptTools](https://github.com/hegelai/prompttools)，这是一个用于 LLM 和向量数据库的 Prompt 测试与实验的开源资源。他们强调了其在不同模型上设置实验的能力，尽管它缺乏版本管理。

- **Vercel 的 A/B 测试与对比工具**：讨论还指出了 Vercel 用于通过单个 Prompt 对比模型的工具，并指出其与 PromptTools playground 的相似之处。未提供 Vercel 工具的直接链接。

- **Helicone 作为生成式 AI 平台**：一位成员介绍了 [Helicone](https://www.helicone.ai/)，这是一个用于构建 AI 应用的综合平台，并指出它现在开始包含 Prompt 管理、版本控制和分析功能。

- **PromptFoo 用于测试和回归**：提到 [PromptFoo](https://github.com/promptfoo/promptfoo) 受到好评，因为它提供了一种评估和比较 LLM 输出、优化提示词质量的方法，并包含针对 OpenAI, Azure GPT 等多种模型的 CI/CD 集成。

- **个性化博客文章翻译实验**：一位成员分享了他们的博客实验，使用 gpt-3.5-turbo 为不同角色（personas）翻译文章，旨在提高读者的理解和参与度。可以在 [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html) 查看实际效果。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html">How to Build a Buzzword</a>：以及为什么它们如此强大</li><li><a href="https://www.helicone.ai/">Helicone</a>：开发者如何构建 AI 应用。开箱即用，获得可观测性、工具、微调和评估。</li><li><a href="https://sdk.vercel.ai/">Vercel AI SDK</a>：使用最新的 AI 语言模型构建 AI 驱动的应用</li><li><a href="https://github.com/hegelai/prompttools">GitHub - hegelai/prompttools: 用于提示词测试和实验的开源工具，支持 LLM（如 OpenAI, LLaMA）和向量数据库（如 Chroma, Weaviate, LanceDB）。</a>：用于提示词测试和实验的开源工具，支持 LLM（如 OpenAI, LLaMA）和向量数据库（如 Chroma, Weaviate, LanceDB）。 - hegelai/prompttools</li><li><a href="https://github.com/promptfoo/promptfoo">GitHub - promptfoo/promptfoo: 测试你的提示词、模型、RAG。评估和比较 LLM 输出，捕获回归，并提高提示词质量。支持 OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, 本地和私有模型（如 Mistral/Mixtral/Llama）的 LLM 评估，支持 CI/CD</a>：测试你的提示词、模型、RAG。评估和比较 LLM 输出，捕获回归，并提高提示词质量。LLM 评估支持 OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, 本地和...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 条消息): 

obra: 是否可以恢复 OpenAI 模型在之前的 API 请求中使用的 seed？
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1218193382669549568)** (17 条消息🔥): 

- **模型准确率的显著提升**：一位成员提到他们正在完成一篇**详细介绍提高模型全局准确率的方法的文章**，并使训练更具样本效率。一旦准备好更好的图表和结构化结果，他们将分享该论文。
- **寻求更大模型的测试资源**：该成员还表示需要**在更大的模型上测试该方法**，但目前缺乏相关资源。
- **在 VGG16 上的结果验证**：该方法已在 **VGG16** 上得到验证，显示出良好的前景，在 CIFAR100 的子集上仅经过一个 epoch 后，**测试准确率从 0.04 跃升至 0.1**。
- **提供计算和资源帮助**：另一位成员提议在讨论完初步验证后，**分配计算资源**来帮助扩大新方法的验证和测试规模。
- **可能参与 'Quiet-STaR' 项目**：另一位成员询问是否可以参与 **"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking"** 项目，并被问及对 **PyTorch 和 Transformer 架构**的熟练程度。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=ZlJbaYQ2hm4