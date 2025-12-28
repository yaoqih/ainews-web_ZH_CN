---
companies:
- anthropic
- hugging-face
- nvidia
date: '2024-07-17T02:04:35.319219Z'
description: '**博士级基准测试**突显了大语言模型（LLM）在编写科学问题代码方面的困难，**GPT-4** 和 **Claude 3.5 Sonnet**
  在新的 **SciCode** 基准测试中得分均低于 5%。**Anthropic** 将 Claude 3.5 Sonnet 的最大输出 token 限制翻倍至
  8192 个。**Q-GaLore** 方法支持在单块 16GB 显存的 GPU 上训练 **LLaMA-7B**。**Mosaic 编译器**现在可为英伟达（NVIDIA）H100
  GPU 生成高效代码。Hugging Face 上的 **Dolphin 2.9.3-Yi-1.5-34B-32k-GGUF** 模型下载量已超过 11.1 万次。**Llama
  3** 表现强劲，在 MATH 数据集上实现了 90% 的零样本准确率。关于模型训练中**合成数据**的局限性及其形式的讨论仍在继续。'
id: 6c90b0ac-22eb-4f4b-92d1-4a9d069e7cd1
models:
- gpt-4
- claude-3.5-sonnet
- llama-3-7b
- llama-3
- dolphin-2.9.3-yi-1.5-34b-32k-gguf
original_slug: ainews-to-be-named-5745
people:
- yi-tay
- rohanpaul_ai
- alexalbert__
- tri_dao
- abacaj
title: SciCode：HumanEval 迎来 STEM 博士级升级
topics:
- benchmarks
- coding
- model-training
- gpu-optimization
- model-performance
- synthetic-data
- compiler-optimization
- zero-shot-learning
---

<!-- buttondown-editor-mode: plaintext -->**博士级基准测试（PhD-level benchmarks）就是你所需的一切。**

> 2024年7月15日至7月16日的 AI 新闻。
我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**466** 个频道和 **2228** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**248 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

到处都是许多小的更新——[HuggingFace 的 SmolLM](https://x.com/xenovacom/status/1813258097185448377) 复现了 MobileLLM（[我们一周前的报道](https://buttondown.email/ainews/archive/ainews-to-be-named-3686/)），Yi Tay 撰写了[《BERT 之死》](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising)（[我们两周前的播客](https://x.com/latentspacepod/status/1809300018907828285)），以及[旧金山的 1 个街区](https://x.com/evanjconrad/status/1813297376544854063)在 [Exa](https://x.com/evanjconrad/status/1813308202534211998)、[SFCompute](https://x.com/evanjconrad/status/1813293874288472493) 和 [Brev](https://x.com/NaderLikeLadder/status/1813286240093151412) 的交易中筹集/出售了超过 3000 万美元（祝贺朋友们！）。

然而，我们今天的技术亮点是 [SciCode](https://x.com/MinyangTian1/status/1813182904593199553)，它挑战 LM 为高级论文中的科学问题编写代码解决方案。这些挑战由博士精心设计（约 10% 基于诺贝尔奖级别的研究），而两个领先的 LLM，GPT-4 和 Sonnet 3.5，在这个新基准测试中的得分低于 5%。

 
![image.png](https://assets.buttondown.email/images/66dacbe0-3e49-4861-9e66-b94e19afe531.png?w=960&fit=max)
 

除了 HumanEval 和 MBPP 之外，下一个声称是顶级编程基准测试的是 SWEBench（[更多关于我们报道的信息](https://www.latent.space/p/iclr-2024-benchmarks-agents)），但它的运行成本很高，而且更多是对 Agent 系统的集成测试，而不是对纯粹编程能力/世界知识的测试。SciCode 为非常流行的 HumanEval 方法提供了一个很好的扩展，易于且廉价地运行，但对于 SOTA LLM 来说仍然非常困难，提供了一个很好的评估梯度。

没有什么是永恒的（[SOTA SWEbench 在 6 个月内从 2% 增长到 40%](https://x.com/Douglas_Schon/status/1813213722770354177)），但如果做得好，新的且能立即应用的基准测试工作是非常棒的。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。

**AI 模型开发**

- **Anthropic API 更新**：[@alexalbert__](https://twitter.com/alexalbert__/status/1612921642143900036) 指出，Anthropic 在 Anthropic API 中将 Claude 3.5 Sonnet 的最大输出 Token 限制从 4096 翻倍至 8192，**只需在 API 调用中添加请求头 "anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15" 即可**。
- **有效的 Claude Sonnet 3.5 编程系统提示词**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1612973162906460460) 分享了一个有效的 Claude Sonnet 3.5 编程系统提示词，并解释了引导式思维链步骤：**代码审查（Code Review）、规划（Planning）、输出安全审查（Output Security Review）**。
- **Q-GaLore 支持在 16GB GPU 上训练 7B 模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1612981403740463207) 指出，Q-GaLore 结合了低精度训练、低秩梯度和延迟逐层子空间探索，**支持在单个 16GB NVIDIA RTX 4060 Ti 上从头开始训练 LLaMA-7B，尽管速度通常较慢**。
- **Mosaic 编译器生成高效的 H100 代码**：[@tri_dao](https://twitter.com/tri_dao/status/1612913394086998408) 强调，最初为 TPU 设计的 Mosaic 编译器**可以生成非常高效的 H100 代码，显示了 AI 加速器的趋同性**。
- **Hugging Face 上的 Dolphin 2.9.3-Yi-1.5-34B-32k-GGUF 模型**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1612958456317464804) 赞扬了 @bartowski1182 和 @cognitivecompai 在 Hugging Face 上发布的卓越 Yi 微调模型，**上个月下载量超过 11.1 万次**。

**AI 模型性能与基准测试**

- **Llama 3 模型性能**：[@awnihannun](https://twitter.com/awnihannun/status/1812910444841214066) 对比了 ChatGPT（免费版）与在 M2 Ultra 上运行的 MLX LM（搭载 Gemma 2 9B），显示出相当的性能。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1812924233346904228) 指出 Llama 3 在 **MATH 数据集上实现了 90% 的 Zero-shot 准确率**。
- **合成数据的局限性**：[@abacaj](https://twitter.com/abacaj/status/1812857696556663195) 认为合成数据很愚蠢，且**不太可能产生更好的模型**，并质疑了合成指令的真实性。[@Teknium1](https://twitter.com/Teknium1/status/1812905541993439597) 反驳称**合成数据有多种形式**，下笼统的结论是不明智的。
- **使用 LLM 评估 LLM**：[@percyliang](https://twitter.com/percyliang/status/1812999994255024144) 强调了**使用 LLM 生成输入并评估其他 LLM 输出**的强大能力（如 AlpacaEval 所示），同时也对过度依赖自动评估（automatic evals）提出了警告。
- **LLM-as-a-judge 技术**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1612949923010421192) 概述了使用 LLM 评估其他 LLM 输出的最新研究，包括**早期研究、揭示偏见的更正式分析以及专门的评估器**。


**AI 安全与监管**

- **FTC 因收购 VR 公司起诉 Meta**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1612978264484552987) 提到他被卷入了两个法庭案件，其中 FTC 因 Meta 收购小型 VR 公司而起诉该公司；他指出**大型科技公司已全面大幅减少收购，这对初创公司不利，因为收购退出（acquisition exits）渠道正在收窄**。
- **扼杀开源 AI 可能会使 AI 安全政治化**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1612996261521551495) 警告称，**扼杀开源 AI 可能会导致 AI 安全政治化，而开源才是解决方案**。
- **LLM 并非智能，只是记忆机器**：[@svpino](https://twitter.com/svpino/status/1612888808372736309) 认为 LLM 是极其强大的记忆机器，虽然令人印象深刻但并不具备智能。**它们可以记忆大量数据并从中进行少量泛化，但无法适应新问题、合成新颖的解决方案、跟上世界变化或进行推理**。

**AI 应用与演示**

- **AI 与机器人每周回顾**：[@adcock_brett](https://twitter.com/adcock_brett/status/1612880560819220806) 提供了过去一周最重要的 AI 和机器人研究与进展的详细分析。
- **Agentic RAG 与多智能体架构概念**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1612991904268849343) 分享了 @nicolaygerold 关于 Agentic RAG 和多智能体架构概念的推文串和图表，这些内容曾在他们的 @aiDotEngineer 演讲中讨论过。
- **M2 Ultra 上的 MLX LM (Gemma 2 9B) 对比 ChatGPT**：[@awnihannun](https://twitter.com/awnihannun/status/1612910444841214066) 对比了 ChatGPT（免费版）与在 M2 Ultra 上运行的 MLX LM（搭载 Gemma 2 9B）。
- **Odyssey AI 视频生成平台**：[@adcock_brett](https://twitter.com/adcock_brett/status/1612880741425918295) 指出 Odyssey 结束隐身模式，推出了一个“好莱坞级”的 AI 视频生成平台，正在开发四个专门的 AI 视频模型。


**迷因与幽默**

- **9.11 大于 9.9**：[@goodside](https://twitter.com/goodside/status/1612977352085020680) 在一条幽默推文中开玩笑说“9.11 大于 9.9”，并在随后的推文中进行了变体和解释。
- **Perplexity 办公室**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1612890154367078590) 分享了一张题为“新 Perplexity 办公室！”的幽默图片。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**主题 1. 新前沿**

- [/r/singularity] **[另一位知情人士透露，OpenAI 内部测试的 AI 在 MATH 数据集上得分超过 90%](https://i.redd.it/akj0xjlmspcd1.png)** ([Score: 206, Comments: 59](https://reddit.com//r/singularity/comments/1e405o0/a_different_source_briefed_on_the_matter_said/)): **据报道 OpenAI 的 AI 在 MATH 数据集上得分超过 90%**。一位匿名消息人士称，OpenAI 内部测试了一个 AI 系统，能够在 **MATH 数据集**上达到 **超过 90% 的准确率**，这表明 AI 在数学问题解决能力方面取得了重大进展。如果得到证实，这一进展对于 AI 应对复杂数学挑战及其在需要高级数学推理的各个领域的应用可能产生深远影响。
- [/r/singularity] **一台新型量子计算机打破了 Google Sycamore 机器创造的世界纪录。新型 56 量子比特 H2-1 计算机将“量子霸权”纪录提高了 100 倍。** ([Score: 365, Comments: 110](https://reddit.com//r/singularity/comments/1e3z409/a_new_quantum_computer_has_shattered_the_world/)): **Xanadu 的 56 量子比特 H2-1 量子计算机**据报道在**量子霸权（quantum supremacy）基准测试**中超越了 **Google 的 Sycamore 机器**，提升幅度达 **100 倍**。这一成就标志着量子计算能力的重大飞跃，可能加速该领域向实际应用迈进。该消息由 [@dr_singularity](https://x.com/dr_singularity/status/1812802357962441135?s=46) 在 X（原 Twitter）上分享，但更多细节和验证尚待提供。


**主题 2. 用于细节图像生成的先进 Stable Diffusion 技术**


- [/r/StableDiffusion] **[Tile controlnet + Tiled diffusion = 非常逼真的放大器工作流](https://www.reddit.com/gallery/1e3v6jy)** ([Score: 517, Comments: 109](https://reddit.com//r/StableDiffusion/comments/1e3v6jy/tile_controlnet_tiled_diffusion_very_realistic/)): **Tile controlnet** 结合 **Tiled diffusion** 创建了一个非常高效的逼真图像放大工作流。该技术允许在保持精细细节和纹理的同时，将图像放大到 **4K 或 8K 分辨率**，超越了传统 AI 放大器的质量。该过程涉及使用 controlnet 生成高分辨率的瓦片（tile）模式，然后将其作为 tiled diffusion 的引导，从而生成无缝且细节丰富的最终图像。

- [/r/StableDiffusion] **[用 SD 创造细节丰富的世界仍然是我最喜欢做的事情！](https://www.reddit.com/gallery/1e4aynd)** ([Score: 357, Comments: 49](https://reddit.com//r/StableDiffusion/comments/1e4aynd/creating_detailed_worlds_with_sd_is_still_my/)): **使用 Stable Diffusion 创造精细的奇幻世界**仍然是创意表达的首选。生成复杂、富有想象力的景观和环境的能力展示了 AI 在视觉艺术创作中的力量。这项技术让艺术家和爱好者能够以非凡的细节和深度将他们的奇幻愿景变为现实。
    - **NeededMonster** 详细介绍了他们创建精细奇幻世界的 **4 阶段工作流**，包括初始提示词（prompting）、局部重绘/外延绘制（inpainting/outpainting）、放大以及细节微调。每张图片的制作过程可能需要 **1.5 小时**。
    - 评论者称赞这些图像具有“**书封质量**”和“**目前见过的最佳作品**”，一些人建议该艺术家的技能“**值得雇佣**”。NeededMonster 表示有兴趣寻找此类图像创作的工作。

**主题 3. 使用 Unsloth 和 Ollama 微调 Llama 3**

- [/r/LocalLLaMA] **分步教程：如何使用 Unsloth + Google Colab 微调 Llama 3 (8B) 并将其部署到 Ollama** ([Score: 219, Comments: 41](https://reddit.com//r/LocalLLaMA/comments/1e416fo/stepbystep_tutorial_how_to_finetune_llama_3_8b/)): **基于 Unsloth 的 Llama 3 微调教程**。本教程演示了如何使用 [Unsloth](https://github.com/unslothai/unsloth) 微调 **Llama-3 (8B)** 并将其部署到 [Ollama](https://github.com/ollama/ollama) 以供本地使用。该过程包括使用 [Google Colab](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) 获取免费 GPU 资源，在 **Alpaca 数据集**上进行微调，并导出模型到 Ollama（自动创建 `Modelfile`）。关键特性包括 **微调速度提升 2 倍**、**内存占用减少 70%**，以及通过 Unsloth 的 `conversation_extension` 参数支持多轮对话。


---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. Mamba 模型掀起波澜**

- **Codestral Mamba 崭露头角**：Mistral AI 发布了 [Codestral Mamba](https://mistral.ai/news/codestral-mamba/)，这是一个 7B 的编程模型，采用 **Mamba2** 架构而非 Transformer，提供线性时间推理和无限序列处理能力。
   - 该模型采用 Apache 2.0 许可证，旨在提高代码生产力。社区讨论强调了其对 **LLM 架构** 的潜在影响，一些人指出它尚未在 `llama.cpp` 等流行框架中得到支持。
- **Mathstral 增强 STEM 实力**：除了 Codestral Mamba，Mistral AI 还推出了 [Mathstral](https://mistral.ai/news/mathstral/)，这是一个针对 STEM 推理进行微调的 7B 模型，在 MATH 基准测试中获得了 **56.6%** 的优异成绩，在 MMLU 基准测试中获得了 **63.47%**。
   - Mathstral 是与 [Project Numina](https://projectnumina.ai/) 合作开发的，体现了针对特定领域优化的专业化模型日益增长的趋势，有可能重塑科学和技术领域的 AI 应用。
  


**2. 高效 LLM 架构的演进**

- **SmolLM 展现小巧而强大的力量**：[SmolLM](https://x.com/loubnabenallal1/status/1813252390692303069?s=46) 推出了参数量从 135M 到 1.7B 的新型 SOTA 模型，这些模型在高质量的网络、代码和合成数据上进行训练，性能超越了 MobileLLM 和 Phi1.5 等更大型的模型。
   - 这些紧凑型模型突显了高效、设备端 LLM 部署日益增长的重要性。此次发布引发了关于平衡模型大小与性能的讨论，特别是针对边缘计算和移动应用。
- **Q-Sparse 为稀疏化增色**：研究人员介绍了 [Q-Sparse](https://arxiv.org/abs/2407.10969)，这是一种使完全稀疏激活的大语言模型 (LLMs) 能够以更高效率实现与稠密基准模型相当结果的技术。
   - 这一进展是在 BitNet b1.58 发布四个月后取得的，BitNet b1.58 将 LLMs 压缩到了 1.58 bits。AI 社区讨论了 Q-Sparse 如何潜在地重塑 LLM 训练和推理，特别是对于资源受限的环境。
  


**3. AI 教育与基准测试的突破**

- **Karpathy 在 AI 教育领域的“尤里卡时刻”**：Andrej Karpathy 宣布启动 [Eureka Labs](https://x.com/karpathy/status/1813263734707790301)，这是一个 AI 原生教育平台，首门课程为 LLM101n，这是一门关于训练个人 AI 模型的本科级课程。
   - 该计划旨在将 AI 专业知识与创新教学方法相结合，有可能改变 AI 的教学和学习方式。社区反应非常积极，并讨论了其对普及 AI 教育的意义。
- **SciCode 为 LLM 评估设定新标准**：研究人员推出了 [SciCode](https://x.com/MinyangTian1/status/1813182904593199553?s=46)，这是一个新的基准测试，挑战 LLMs 为来自高级论文（包括诺贝尔奖获奖研究）的科学问题编写代码解决方案。
   - 初步测试显示，即使是 GPT-4 和 Claude 3.5 Sonnet 等先进模型，其准确率也低于 5%，突显了该基准测试的难度。AI 社区讨论了其对模型评估的潜在影响，以及对更严格、特定领域测试的需求。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NVIDIA 告别特定项目**：**NVIDIA** 关停某个项目引发了社区成员对其未来的猜测，并触发了关于如何保护工作成果免受突然中断影响的讨论。
   - 一位用户建议将本地存储作为项目关停的应急方案，旨在最大限度地减少工作损失。
- **RAG 受到审视**：围绕 **RAG (Retrieval-Augmented Generation)** 的质疑声不断，据报道该技术入门容易，但要微调到完美状态却极具挑战且成本高昂。
   - 一项针对优化的深入研究揭示了其中的复杂性，成员们提到了 **“微调 LLM、embedder 和 reranker”** 的艰巨任务。
- **模型微调价格不菲**：微调一个超过 **200GB** 的语言模型可能会产生巨额费用，这引发了关于推进大模型在财务准入门槛上的争论。
   - **Google Cloud 的 A2 实例** 成为一种可能但依然昂贵的替代方案，强调了成本在规模化方程中的分量。
- **Codestral Mamba 投入行动**：Mistral AI 的 **Codestral Mamba** 凭借线性时间推理（linear time inference）能力取得突破，为提升代码生产力提供了快速解决方案。
   - **Mathstral** 随之发布，重点展示了其在 STEM 领域的高级推理能力，并因其底层实力而备受关注。
- **Unsloth Pro 专属俱乐部**：目前处于 NDA 阶段的新版 **Unsloth Pro** 令人兴奋，它支持多 GPU 和 DRM 系统，但仅限于付费订阅模式。
   - 尽管仅限高级用户，但人们对其针对多样化部署的有效 DRM 系统寄予厚望。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **FlatBuffers 对抗 Protobuf 尽显优势**：社区讨论了 [FlatBuffers](https://flatbuffers.dev/) 相比 Protobuf 的优势，强调了 FlatBuffers 的性能以及 Apache Arrow 的集成，尽管后者仍使用 Protobuf 进行数据传输。
   - 尽管效率很高，但 FlatBuffers 面临着使用难度较大和行业渗透率较低的挑战，引发了关于选择序列化框架的辩论。
- **Mojo 的 Python 兼容性难题**：一项关于禁用与 Python 完全兼容的 Mojo 模式提案引发讨论，旨在推动 **以 Mojo 为中心的语法** 和 **健壮的错误处理**。
   - 讨论包括建议采用类似 Rust 的单子（monadic）错误处理以增强可靠性，避免传统的 try-catch 块。
- **MAX Graph API 教程遇到障碍**：学习者在 [MAX Graph API 教程](https://www.modular.com/blog/max-graph-api-tutorial) 中遇到困难，碰到了 Python 脚本差异和安装错误。
   - 社区干预纠正了诸如 Jupyter kernels 不匹配和导入问题等错误，并建议新手使用 nightly builds 以获得更顺畅的体验。
- **Mistral 7B 编程模型以无限序列令人惊叹**：Mistral 发布了利用 **Mamba2** 的全新 **7B 编程模型**，凭借其序列处理能力改变了代码生产力的格局。
   - 社区对该 [模型](https://mistral.ai/news/codestral-mamba/) 表现出极大的热情，并分享了用于 GUI 构建和 ONNX 转换的资源。
- **错误处理引发 Mojo 热议**：关于 Mojo 错误处理的讨论激增，对于显式错误传播及其对系统编程中函数的影响存在强烈观点。
   - 一个核心主题是强调显式声明以改进代码维护，并适应 GPU 和 FPGA 等多样化硬件上的错误处理。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **冠军数学解题模型现已开源**：AI 数学奥林匹克冠军 **NuminaMath** 已[正式开源](https://x.com/reach_vb/status/1811069858584363374)，该 **7B 模型**在 Kaggle 上取得了 **29/50** 的优异成绩。
   - 该模型经过两个阶段的精细微调，利用了海量的数学题和专门为 **GPT-4** 优化的合成数据集。
- **Whisper Timestamped 实现逐词标记**：**Whisper Timestamped** 在语音识别领域再下一城，通过 **Transformers.js** 推出了支持[多语言](https://x.com/xenovacom/status/1811068015229747335)的强大浏览器端解决方案。
   - 这一进展为浏览器端视频编辑工具提供了完整的代码支持和带时间戳转录的神奇演示。
- **语音大师：Nvidia BigVGAN v2 翱翔**：Nvidia 发布了 [BigVGAN v2](https://x.com/reach_vb/status/1813181163126587830)，这是他们最新的神经声码器（Neural Vocoder），在 A100 上能更快速地将梅尔频谱图（Mel spectrograms）合成为音频。
   - 该模型通过升级的 **CUDA** 核心、精调的判别器（discriminator）和共鸣损失函数进行了改造，承诺提供支持高达 **44 kHz** 的听觉盛宴。
- **强强联手：Hugging Face x Keras**：**Keras** 通过与 [Hugging Face 的合作](https://huggingface.co/blog/keras-nlp-integration)引入了 NLP 功能，为开发者在神经文本处理领域拓展了边界。
   - 此次融合旨在将一系列 NLP 功能引入 Keras 生态系统，为开发者提供无缝的模型集成体验。
- **界面革新：Hugging Face Tokens**：Hugging Face 改进了其 [Token 管理界面](https://x.com/coyotte508/status/1810982231180992521)，增加了 Token 过期日期和显示最后四位数字等新功能。
   - 此次 UI 升级旨在保护您的珍贵 Token，充当 Token 列表的管家，只需一眼即可掌握详细信息。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **告别 NSFW：AI Morph 加强管控**：Daily Joy Studio 开发的工具 AI Morph 因停止允许 NSFW 内容并显示“不符合指南”警报而引发讨论。
   - 社区反应不一，有人猜测这对**内容创作的影响**，而另一些人则在讨论替代工具。
- **使用 Stable Diffusion 创作动漫艺术**：用户咨询了如何精细调整 Stable Diffusion 以提高动漫艺术的颜色、服饰和面部表情准确性，暗示了对**细粒度控制机制**的需求。
   - 几位用户交流了技巧，并指向了一些 [GitHub 仓库](https://github.com/leejet/stable-diffusion.cpp)作为潜在资源。
- **Detweiler 的教程：社区最爱**：社区对 Scott Detweiler 在 YouTube 上的 Stable Diffusion 教程表示赞赏，称赞其对工具的**高质量见解**。
   - 他作为 Stability.ai 质量保证角色的贡献受到关注，巩固了他作为学习首选来源的地位。
- **自制 AI 工具融合本地与 Stable Diffusion**：关于开发一款巧妙集成 Stable Diffusion 的本地 AI 工具的讨论非常热烈，该工具已成为 *capt.asic* 的首选。
   - 讨论延伸到了将 Stable Diffusion 与**本地语言模型 (LLM)** 支持相结合的有效性。
- **AI 显卡之争：4090 vs. 7900XTX**：针对 AI 任务的 GPU 性能引发了辩论，NVIDIA 的 4090 与 AMD 的 7900XTX 在性价比和易用性方面展开对决。
   - 在规格讨论中，[Google Colab 项目](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)的链接和规格对比进一步推动了讨论。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Kernel 奇遇与 PyTorch 实力**：**#[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1262494017929871473)** 频道讨论了在 Python 脚本中通过 PyTorch 调用 CUDA kernel 的话题，建议使用 **PyTorch profiler** 来梳理哪些 ATen 函数被触发。
   - 讨论中强调了一场性能拉锯战，引用了一门[课程](https://youtu.be/4sgKnKbR-WE?t=4045)，其中原生 CUDA 矩阵乘法 kernel 耗时 **6ms**，而其 PyTorch 对应版本仅需 **2ms**，这引发了关于 PyTorch kernel 在 CNN 卷积操作中效率的讨论。
- **Spectral Compute 的 SCALE 工具包大获成功**：**SCALE** 出现在 **#[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1262562492824027157)** 中，因其能够将 CUDA 应用程序转译为适用于 AMD GPU 的程序而受到赞誉，这一举措可能会改变计算范式。
   - 随着未来支持更多 GPU 厂商的承诺，开发者被引导至 SCALE 的[文档](https://docs.scale-lang.com/)以获取教程和示例，预示着跨平台 GPU 编程灵活性可能大幅提升。
- **Suno 寻找机器学习大师**：**#[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1262526637904498800)** 频道重点介绍了 Suno 招募 ML 工程师的活动，要求精通 **torch.compile** 和 **triton** 以构建实时音频模型；根据[职位公告](https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20)，熟悉 **Cutlass** 是加分项而非必选。
   - 实习岗位也已上线，为初学者提供了进入 Suno ML 领域参与训练和推理复杂协作的切入点。
- **Lightning Strikes 与 Huggingface 开发进展**：**#[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1262594919206227981)** 的讨论展示了 **[Lightning AI's Studios](https://lightning.ai/docs/overview/studios#studios)**，这是一个设计精巧的混合云端浏览器开发环境；而关于 **Huggingface Spaces Dev Mode** 的 CUDA 开发愿景仍悬而未决。
   - 后者收到了关于在 Huggingface 生态系统中进行 CUDA 尝试可行性的咨询，反映出社区正处于实验与探索的前沿。
- **Karpathy 在 AI+教育领域的灵光一现**：**Andrej Karpathy** 宣布通过 **[Eureka Labs](https://x.com/karpathy/status/1813263734707790301)** 进军 AI 辅助教育领域；**#[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262485006207418409)** 的讨论涉及了将教学与技术结合以服务 AI 爱好者的意图。
   - 前导课程 LLM101n 标志着 Eureka 使命的开始，旨在增强教育的可及性和参与度，为 AI 潜在重塑学习体验奠定基础。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的慷慨之举**：OpenRouter 慷慨地向社区提供了对 **Qwen 2 7B Instruct** 模型的免费访问，为热衷技术的工程师增强了 AI 工具储备。
   - 要使用此服务，用户可以访问 [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) 并在无需订阅的情况下使用该模型。
- **Gemini 辩论：免费层的挫败感**：针对通过免费层提供的 Google **Gemini 1.0** 出现了不同的观点，强烈的意见认为它未达到 **OpenAI GPT-4o** 的基准。
   - 一位成员强调了 **Gemini 1.5 Pro** 中被忽视的潜力，称尽管它在编程方面有些怪癖，但具有出色的创作能力。
- **OpenRouter 波动：连接困扰**：OpenRouter 用户在访问网站及其 API 时遇到了不稳定的中断，引发了社区内的一系列担忧。
   - 官方声明将零星的停机归因于瞬时路由绕路以及 Cloudflare 等第三方服务的潜在问题。
- **渴望 Llama 3 更长的上下文**：工程师们分享了对 **Llama 3-70B Instruct** 中 **8k 上下文窗口限制** 的困扰，并思考更优的替代方案。
   - 提供扩展上下文能力的模型建议包括 **Euryale** 和 **Magnum-72B**，但其一致性和成本因素仍是关注点。
- **OpenRouter 访问的必要性与细微差别**：关于 OpenRouter 模型可访问性的澄清在用户中传开，指出并非所有模型都是免费的，有些需要付费协议。
   - 尽管存在困惑，OpenRouter 确实提供了一系列免费的 API 和模型，而托管企业级 dry-run 模型则需要强调特定的商业合同。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 动荡：Pro 级别的困扰**：用户在 [Perplexity AI](https://perplexity.ai/settings/account) 表达了对 **Pro Subscription Support** 的不满，尽管收到了确认邮件，但在不同设备上仍面临激活问题。
   - 讨论围绕着诸如为**不同集合（collections）实现模型设置**等问题展开，并对新的 [Perplexity Office](https://x.com/AravSrinivas/status/1812890154367078590) 表示兴奋。
- **Alphabet 的大胆收购：230 亿美元成交**：**Alphabet** 耗资 230 亿美元的巨额收购引起了轰动，引发了市场波动，你可以在 [YouTube 简报](https://www.youtube.com/embed/lKn8rh0pOiM)中了解详情。
   - 随之而来的猜测和讨论集中在这一举动将如何开启新的篇章，使 Alphabet 成为潜在市场扩张的焦点。
- **发现月球避难所：未来宇航员的洞穴**：在静海（Mare Tranquillitatis）发现的一个**可进入的月球洞穴**可能成为宇航员居住的福音，因为它能抵御月球的极端环境。
   - 根据 [Perplexity AI 的报道](https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0)，该洞穴宽度至少为 130 英尺，环境更加温和，是潜在月球基地的理想选址。
- **7-Eleven 的体验提升：旨在取悦客户**：为了提升购物愉悦感，**7-Eleven** 正准备进行重大升级，可能会重塑消费者互动格局。
   - 7-Eleven 邀请你[探索此次升级](https://www.youtube.com/embed/lKn8rh0pOiM)，这或许会为零售便利性树立新标杆。
- **API 焦虑：pplx-api 的波动**：`pplx-api` 用户群体讨论了缺失的功能（如 `deleted_urls` 的等效项）以及影响 `sonar` 模型的 **524 错误**带来的挫败感。
   - 建议的解决方法包括设置 `"stream": True`，以便在 `llama-3-sonar-small-32k-online` 及相关模型出现超时时保持连接活跃。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral Mamba 凭借无限序列建模出击**：[Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 的推出具有线性时间推理（linear time inference）特点，标志着 AI 处理无限序列能力的一个里程碑，助力提升代码生产力基准。
   - 该模型由 Albert Gu 和 Tri Dao 合作开发，**Codestral 的架构**可与顶尖的 Transformer 模型竞争，预示着向更高效的序列学习转变。
- **Mathstral 精准处理数字**：专注于 STEM 领域的 [Mathstral](https://mistral.ai/news/mathstral/) 在 MATH 基准测试中达到 **56.6%**，在 MMLU 基准测试中达到 **63.47%**，增强了在特定技术领域的表现。
   - Mathstral 与 [Project Numina](https://projectnumina.ai/) 合作，代表了在专业领域中速度与高层级推理之间的精确平衡。
- **SmolLM 在端侧表现强劲**：SmolLM 的新 SOTA 模型提供了[高性能与缩减的规模](https://x.com/loubnabenallal1/status/1813252390692303069?s=46)，在 LLM 的端侧部署方面取得了进展。
   - 这些模型超越了 MobileLLM 等模型，表明了在保持足够效能的同时进行小型化的趋势，这对于移动应用至关重要。
- **Eureka Labs 启发 AI 与教育的交汇**：凭借其 AI 原生助教，[Eureka Labs](https://eurekalabs.ai/) 为 AI 驱动的教育体验铺平了道路，首个产品为 LLM101n。
   - Eureka 的创新方法旨在让学生通过**训练自己的 AI** 来构建理解，彻底改变教育方法论。
- **倡导策略强化中的公平竞争**：讨论围绕策略强化（policy reinforcement）中退化情况（degenerate cases）的效用展开，用于管理**获胜和失败策略**中的常见前缀。
   - 尽管公认需要深入研究，但对详细技术细节的关注可能会引入**策略优化（policy optimization）**中的先进方法。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4 对 GSM8k 的掌握**：GPT-4 的性能实力在其处理大部分 **GSM8k 训练集**的能力中得到了体现；这是从 [GPT-4 技术报告](https://link.to.report)中分享的一个有趣事实。
   - 社区重点关注了 **GPT-4** 的记忆力表现，这类细节通常会在社交平台上引起广泛反响。
- **指令微调（Instruction Tuning）语法受到审视**：生成的指令微调数据集中的语法引发了讨论，人们质疑这种方法与 OpenAI 使用项目符号进行思维链（chaining thoughts）的方法相比孰优孰劣。
   - 社区对在微调数据集中包含特定标记的潜在可能性感到好奇，从而引发了关于数据集完整性的对话。
- **显微镜下的 GPU 故障**：成员们试图了解模型训练期间 **GPU 故障**的频率，并参考了诸如 [Reka 技术报告](https://publications.reka.ai/reka-core-tech-report.pdf)之类的报告。
   - 像 OPT/BLOOM 日志这样的开源资源，成为了那些旨在分析大规模 AI 训练环境稳定性的人员的首选来源。
- **状态空间模型（State Space Models）的进展**：状态空间模型权重的创新构建推动了讨论，[这项研究](https://arxiv.org/abs/2407.09375)展示了它们在上下文中学习动力系统的能力。
   - 研究人员就这些模型在无需额外参数调整的情况下预测系统状态的潜力交换了意见，强调了它们的实用性。
- **神经元数量：人类 vs 动物**：关于人类与动物智力差异的辩论十分激烈，人们公认人类在规划和创造力方面具有优越性，尽管感官能力与动物相当。
   - 对话涉及了人类更大的新皮层及其增加的褶皱，这可能有助于我们独特的认知能力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 的 Token 突破**：Anthropic 展示了他们的**滴灌式公关策略**，透露 API 中 **Claude 3.5 Sonnet** 的 Token 限制从 4096 提升到了 8192，这让开发者们欢欣鼓舞。
   - 一位热心的开发者表达了满足感，并在一条[庆祝推文](https://x.com/alexalbert__/status/1812921642143900036)中评论了过去的限制，强调了这一**增强功能**。
- **LangGraph vs XState 对决**：一位成员预告了他们使用 **XState** 构建 LLM Agent 的工作，并在 [GitHub](https://github.com/statelyai/agent) 上公开将其方法论与 **LangGraph** 进行对比，热情随之高涨。
   - 期待感正在积聚，大家期待着一份详尽的分析，以界定两者之间的优势，丰富那些尝试**基于状态机的 AI Agent** 的 AI 工程师的工具箱。
- **Qwen2 取代前代产品**：Qwen2 发布了一系列性能超越 Qwen1.5 的语言模型，提供 **0.5 到 720 亿参数**范围，旨在与闭源对手展开竞争。
   - 该系列同时推出了稠密模型和 **MoE** 模型，社区正在仔细研读 [Qwen2 技术报告](https://hf.co/papers/2407.10671)中的承诺。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio：网络连接**：LM Studio 用户讨论了家庭网络服务器的 **Android 应用访问**，重点介绍了 Wireguard 等 **VPN 工具**用于安全的服务器连接。
   - 否定了对 **Intel GPU** 的全面支持，建议在处理 AI 任务时使用其他硬件以获得更好的性能。
- **Bug 与支持：对话碰撞**：**Gemma 2** 在 `llama.cpp` 中获得了支持，而 **Phi 3 small** 则因不兼容问题被搁置。
   - 社区深入研究了 **LMS 模型加载变慢**的问题，指出弹出并重新加载是解决性能迟缓的快速修复方法。
- **与巨人同行编码：Deepseek 与 Lunaris**：在为 **128GB RAM 系统**寻找理想的本地编码模型时，目标锁定在了 **Deepseek V2**，这是一个拥有 **21B 专家**的模型。
   - **L3-12B-Lunaris** 变体之间的精确差异引发了对话，重点在于尝试免费的 LLM 以获取性能见解。
- **图形故障：大小并不重要**：LM Studio 中的 `f32` 文件夹异常引起了关注，导致一些人思考外观 Bug 对用户体验的影响。
   - **Flash Attention** 被认为是 F16 GGUF 模型加载问题的罪魁祸首，停用它可以恢复 RTX 3090 上的功能。
- **STEM 专用：Mathstral 亮相**：**Mistral AI** 推出了 **Mathstral**，这是他们最新的以 STEM 为中心的模型，承诺性能优于其前身 **Mistral 7B**。
   - **社区模型计划**重点推介了 **Mathstral**，邀请 AI 爱好者参与 LM Studio [Discord](https://discord.gg/aPQfnNkxGC) 上的讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Evol-Instruct V2 & Auto 带来的进化式指令飞跃**：WizardLM 宣布 [Evol-Instruct V2](https://x.com/WizardLM_AI/status/1812844503251947978) 将 WizardLM-2 的能力从三个进化领域扩展到数十个，可能增强 AI 研究的公平性和效率。
   - **Auto Evol-Instruct** 展示了显著的性能提升，在 MT-bench 上提升了 **10.44%**，在 HumanEval 上提升了 **12%**，在 GSM8k 上提升了 **6.9%**，超越了人工设计的方法。
- **Q-Sparse：高效计算**：由 Hongyu Wang 介绍的 [Q-Sparse](https://x.com/realHongyu_Wang/status/1813112679734911169) 声称通过优化受内存限制（memory-bound）过程的计算来提升 LLM 计算效率。
   - 这一创新紧随 BitNet b1.58 实现将 LLM 压缩至 **1.58 bits** 之后四个月出现。
- **SpreadsheetLLM：微软在数据领域的新前沿**：**Microsoft** 创新推出的 **SpreadsheetLLM** 在电子表格任务中表现出色，这可能导致数据管理和分析领域的重大转变。
   - 一篇 [预印本论文](https://arxiv.org/abs/2407.09025) 强调了 SpreadsheetLLM 的发布，引发了关于自动化对就业市场影响的辩论。
- **高温警报：城市温度与技术解决方案**：在极端气温下，关于将屋顶涂成白色的讨论兴起，并得到了 [耶鲁大学文章](https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities) 的支持，以减少城市热岛效应。
   - 一种可反射 98% 阳光的超级白漆发明在 [YouTube 演示](https://youtu.be/KDRnEm-B3AI) 中展示，引发了关于其被动冷却建筑潜力的讨论。
- **AI 素养：解码 Tokenization 的烦恼**：讨论了 LLM 在处理阿拉伯语符号时与 **tiktoken 库** 的冲突；解码错误导致原始字符串被特殊 Token 替换，给文本生成带来了挑战。
   - Tokenization 过程的可变性显而易见，结果在 **UTF-8 序列** 和 **0x80-0xFF** 字节之间波动，引发了对 `cl100k_base` 的 Tokenization 可逆性的担忧。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 备受期待的到来**：根据 **OpenAI 博客文章**，关于 Sora 在 2024 年第四季度发布日期的讨论已经浮现。**OpenAI** 尚未确认这些推测。
   - 有人提醒不要信任非官方来源（如随机的 Reddit 或 Twitter 帖子）的发布预测。
- **微型奇迹：GPT Mini 的潜在角色**：**Lymsys** 中传闻的 **GPT mini** 引起了轰动，尽管细节仍然稀少且未经证实。
   - 存在怀疑态度，认为许多预测缺乏具体依据。
- **利用 GPT-4 编程从零到英雄**：讨论了 GPT-4 如何协助爱好者编写 **移动游戏**，并指出虽然它提供了结构，但警惕的错误检查至关重要。
   - 分享了没有编程经验的个人制作 Web App 的成功案例，将其进步归功于 GPT-4 的指导。
- **GPT 模型的语言课程**：AI 模型之间的性能差异与其语言训练的程度有关，影响了回答质量。
   - 讨论范围涵盖了模型对地区方言的混淆，以及 GPT-4 从随意语调向更正式语调的转变，这些都影响了用户体验。
- **聊天机器人开发的挑战**：一名学生尝试创建 **孟加拉语/Banglish** 支持聊天机器人，引发了关于使用 100 条对话进行适度 Fine-tuning 是否有益的讨论。
   - 回复澄清说，虽然 Fine-tuning 有助于模式识别，但这些可能会在超出 Context Window 后被遗忘，从而影响对话流。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的通往更好 RAG 之桥**：一场 [LlamaIndex 研讨会](https://lu.ma/ufx77ve8)将深入探讨如何通过高级解析和元数据提取来增强 **RAG**，并邀请 Deasie 的创始人分享相关见解。
   - 据其[官网](https://deasie.com/)详细介绍，Deasie 的标注工作流据称可以优化 RAG，自动生成层级化的元数据标签。
- **高级文档处理策略**：LlamaIndex 的[新 Cookbook](https://t.co/KWsVGwT3jD) 将 LlamaParse 和 GPT-4o 结合成一种混合文本/图像的 RAG 架构，以处理多样化的文档元素。
   - 同时，Sonnet-3.5 在图表理解方面的出色表现，预示着通过多模态模型和 LlamaParse 的[最新版本](https://t.co/Dq3xaVfXio)可以实现更好的数据解读。
- **使用 LlamaIndex 和 GraphRAG 进行细节图谱化**：用户对比了 **llamaindex property graph** 与 **Microsoft GraphRAG** 的能力，强调了属性图在 text-to-cypher 等检索方法中的灵活性，正如 *Cheesyfishes* 所概述的那样。
   - GraphRAG 的社区聚类与属性图的自定义功能形成了对比，相关示例可在[文档](#)中找到。
- **精简 AI 基础设施**：围绕 LLM 响应的高效源检索展开了讨论，例如 `get_formatted_sources()` 等方法有助于追踪数据来源，这在 LlamaIndex 的[教程](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemo.ipynb)中有所提及。
   - AI 社区正在积极寻求可访问的公共向量数据集，以降低基础设施的复杂性，并倾向于预托管选项，尽管目前尚未提到具体的服务。
- **提升索引加载效率**：成员们分享了加速加载大型索引的策略，建议将并行化作为一个潜在途径，并引发了关于优化 `QueryPipelines` 等方法的讨论。
   - 对于 Neo4J 节点中的数据嵌入，社区转向使用 `PropertyGraphIndex.from_documents()`，该过程在 [LlamaIndex 源代码](https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248)中有详细说明。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Python 社区中的协作**：一位参与者建议那些有兴趣为开源项目做贡献的人探索 [Cohere Python Library](https://github.com/cohere-ai/cohere-python)，以促进社区驱动的改进。
   - 该库的一位爱好者暗示他们打算很快进行 **contribute**（贡献），这可能会加强协作努力。
- **Discord 分类难题需要 Prompt 优化**：尽管主题各异，但 Discord 机器人将所有帖子错误地归类到 'opensource' 类别，这暗示了**自动帖子分类**中存在问题。
   - 一位同事插话推测，简单的 **prompt adjustment**（提示词调整）就可以纠正这一偏差，这让人联想到 r/openai 中被错误路由的帖子。
- **垃圾信息诈骗引发预警提议**：一位积极的成员提议将创建**防诈骗意识**内容作为服务器新成员入职引导的一部分，以提高安全性。
   - 该建议得到了支持，并引发了关于实施社区**安全**最佳实践的讨论。
- **Max Welling 的炉边谈话备受期待**：[C4AI 宣布了与阿姆斯特丹大学 Max Welling 的炉边谈话](https://discord.gg/Jf6FPp3c?event=1262761447486787685)，这一消息令人兴奋。
   - 然而，宣传过程中出现了一个小插曲，即**不必要的 @everyone 提醒**，为此官方已道歉。
- **Discord 重申招聘规则**：服务器**提醒**不允许发布招聘信息，以强化社区准则。
   - 成员被**敦促**通过私下讨论进行就业接洽，强调尊重专业协议。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **微调管道助力功能实现**：**微调管道（finetuning pipeline）**成为讨论的核心，成员们通过分享管道策略来交流解决问题的思路。
   - 一位成员强调了使用 100 组对话进行 **LLM 训练**的应用，以增强孟加拉语聊天机器人的响应能力。
- **MessageGraph 的时间戳时钟**：在 **MessageGraph** 中添加**时间戳**以实现消息时序自动化的技术咨询，引发了社区成员间的对话。
   - 随后引发了关于是否需要使用 **StateGraph** 进行自定义时间状态管理的推测。
- **Verbis 发布：隐私保护的先锋**：Verbis 是一款开源的 MacOS 应用程序，通过利用 **GenAI** 模型进行本地数据处理，承诺提升生产力。
   - 该应用在欢呼声中发布，保证不向第三方共享数据，大胆强调隐私保护。[在 GitHub 上了解更多](https://github.com/verbis-ai/verbis)
- **Web 端的 RAG：LangChain 与 WebLLM 的协同**：一段演示展示了 **LangChain** 和 **WebLLM** 在浏览器环境中的强大实力，部署了一个用于即时问答的聊天模型。
   - 分享的视频实操展示了 **Visual Agents**，并强调了强大的浏览器内用户体验。[观看演示](https://youtu.be/MHuvSuK2dnY)

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **深入动态微调**：成员们对 **PyTorch tuner** 的推出表现出极大的好奇，并就其高效交换和优化**指令模型（instruction models）**的潜力进行了交流。
   - 讨论中强调了该 tuner 调整**上下文长度**的能力，并提醒过长的长度可能会非常消耗 VRAM。
- **Mistral 模板引发的困扰**：讨论表明 **Mistral 独特的聊天模板**偏离了常规，给用户带来了操作上的困扰。
   - 聊天模板的复杂性引发了关于微调策略的对话，以规避出现的问题。
- **合并方法论至关重要**：**Axolotl 仓库**活动频繁，一个新的 **pull request** 正在制定中，标志着开发工作的进展。
   - 然而，**Direct Policy Optimization (DPO)** 方法的简单性受到了质疑，揭示了其在基础 tokenization 和 masking 之外的局限性。
- **LoRA 层引领潮流**：一个围绕 `lora_target_linear` 展开的专题论坛形成了，这是一个 **LoRA 配置**开关，用于改变线性层的适配方式，以实现更高效的微调。
   - 该设置在 Axolotl 微调中的作用引发了讨论，尽管关于在某些层禁用 **LoRA** 的一些疑问仍未得到解答。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.2.0 震撼发布**：备受期待的 [Torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0) 发布，标志着一个重要的里程碑，新增了令人兴奋的模型和 recipes。
   - 社区贡献丰富了该版本，其特点是包含 **sample packing** 等数据集增强功能，以提高性能并支持多样化的应用。
- **无需反向传播负担的评估**：为了优化 checkpoint 的选择，可以在不进行反向传播的情况下计算评估期间的 loss；参与者讨论了绘制 loss 曲线，并在训练和评估数据集之间进行对比。
   - 建议包括修改 [默认 recipe 配置](https://github.com/pytorch/torchtune/issues/1066)，并结合测试集划分以及在模型 eval 模式下使用 `torch.no_grad()` 的 eval 循环。
- **RoPE 嵌入助力超长上下文**：长上下文建模在 [Torchtune 的 RFC](https://github.com/pytorch/torchtune/issues/1183) 中关于扩展 RoPE (Rotary Positional Embeddings) 的提案中获得了关注，为大文档和代码补全任务铺平了道路。
   - 讨论围绕支持超过 8K 的 RoPE 上下文展开，这可以改变对大容量文档和详细代码库的理解。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **ComfyUI 团队策划迪士尼干扰**：一名参与者透露，**ComfyUI 恶意节点攻击**的幕后黑手声称还策划了**迪士尼网络攻击**，挑战了该公司的数字防御。
   - 有人提到，虽然一些人认为**迪士尼攻击**是混乱行为，但也有人猜测并期待 **FBI** 对这些事件进行潜在调查。
- **Codestral 的 Mamba 崭露头角**：最近的一篇帖子分享了一项名为 [**Codestral Mamba** 的突破，这是 Mistral AI 的最新更新](https://mistral.ai/news/codestral-mamba/)，引发了关于其功能和潜在应用的讨论。
   - 关于其性能的细节、与其他模型的比较以及技术规格尚未详细阐述，这让社区对其影响感到好奇。
- **YouTube 带来新鲜教程诱惑**：一名公会成员发布了一个[新 YouTube 教程](https://youtu.be/pj8CtzHHq-k)链接，旨在以视频形式提供教育内容。
   - 视频的教育价值及其与 AI 工程师社区的相关性细节未被讨论，促使成员们独立寻找该内容。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Meta 智能眼镜面临挑战**：工程师们正努力将 **Open Interpreter** 集成到 **RayBan Stories** 中，但由于缺乏官方 SDK 和难以获取硬件访问权限而受阻。
   - 拆解设备的尝试揭示了障碍，并在 Pastebin [文档](https://pastebin.com/wTsRMH3f)中进行了讨论，涉及对内部粘合剂和提高改装透明度的担忧。
- **Google Glass：Open Interpreter 的新窗口？**：鉴于破解 **RayBan Stories** 的困难，**Google Glass** 被提议作为可能的替代平台。
   - 该建议提出后对话寥寥，表明需要进一步调查或社区投入。
- **O1 Light 硬件：耐心消磨殆尽**：社区对 **O1 Light 硬件**预订长达数月的延迟感到不满，且更新消息一直处于令人烦恼的沉默状态。
   - 成员们表达了他们的不满，表明对产品的期待已变得紧张，**缺乏沟通**加剧了他们的不安。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **消除困惑：访问 GPT-4o 微调**：关于 **GPT-4o 微调**访问权限的咨询得到了澄清，即必须获得 **OpenAI** 的邀请，正如一位用户引用 Kyle 的声明所强调的那样。
   - 讨论在 #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262787952032092196) 频道展开，反映了社区探索微调能力的渴望。
- **OpenPipeAI 支持 GPT-4o：负责任地训练**：**OpenPipeAI** 宣布支持 **GPT-4o 训练**，[Corbtt](https://x.com/corbtt/status/1813018434822971556) 呼吁负责任地使用。
   - 这一更新为 AI 工程师在 AI 训练任务中更有效地利用课程学分提供了途径。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **深入 Tinygrad 核心：剖析中间表示**：人们对 **tinygrad 的中间语言**产生了浓厚兴趣，用户对 IR 中深度学习算子的结构感到好奇。
   - 大家交流了利用调试选项 **DEBUG=3** 来深入了解底层 IR 的技巧，同时 **GRAPH=1** 和 **GRAPHUOPS=1** 命令成为可视化 tinygrad 内部复杂性的首选选项。
- **Tinygrad 轶事：可视化与调试动态**：在讨论中，有人分享了使用 **DEBUG=3** 调试 tinygrad 的心得，揭示了中间表示错综复杂的底层细节。
   - 此外，对于那些热衷于可视化的人来说，调用 **GRAPH=1** 和 **GRAPHUOPS=1** 可以将 tinygrad 抽象的内部结构转化为清晰的图形。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **通过 Open Interpreter 开启洞察**：Mike Bird 重点介绍了 **Open Interpreter**，并邀请大家积极参与。
   - 在阐述 **Open Interpreter** 的过程中，通过鼓励观众提问来推动互动。
- **与 Interpreter 互动**：在 Mike Bird 展示项目细节时，现场围绕 **Open Interpreter** 展开了热烈讨论。
   - 演示期间，与会者受邀针对 **Open Interpreter** 进行提问并进一步深入交流。

---

**Alignment Lab AI Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期处于沉寂状态，请告知我们，我们将将其移除。

---

# 第二部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1262484993322516510)** (235 messages🔥🔥): 

> - `NVIDIA 关停`
> - `RAG 性能与优化`
> - `大模型微调成本`
> - `Codestral Mamba 和 Mathstral 发布`
> - `Unsloth Pro 许可担忧` 


- **NVIDIA 关停项目**：一位成员确认 **NVIDIA** 关停了一个项目，引发了对其未来更新和维护的担忧。
   - *另一位成员建议在本地存储项目，以避免此类关停带来的损失。*
- **RAG 被过度炒作且完善成本高昂**：**RAG (Retrieval-Augmented Generation)** 被称为一种被过度炒作的解决方案，虽然在几小时内就能初步搭建，但需要数月时间才能完善。
   - “这通常不仅仅是 RAG 和 LLM 的问题；你通常还需要微调 LLM、embedder 和 reranker，”一位成员强调道。
- **大模型微调价格不菲**：在 **200GB+ 数据**上微调模型可能耗资五到六位数（美元），具体取决于模型参数和大小。
   - [Google Cloud 的 A2 实例](https://cloud.google.com/compute/docs/instances/a2-machine-types)被引用为一种可能的解决方案，但成本担忧依然显著。
- **Mistral AI 发布 Codestral Mamba 和 Mathstral**：[Mistral AI 发布了 Codestral Mamba](https://mistral.ai/news/codestral-mamba/)，具有针对无限长度序列的线性时间推理能力，以及用于高级数学问题解决的 Mathstral。
   - Codestral Mamba 旨在为**代码生产力提供快速响应**，而 Mathstral 则在具有最先进推理能力的 **STEM 学科**中表现出色。
- **Unsloth Pro 许可与 GPU 支持**：一个新的 **Unsloth Pro** 版本正在 NDA 下进行测试，其特点是支持多 GPU 以及采用按每月每 GPU 计费的浮动许可 DRM 系统。
   - “免费版本不支持多 GPU，”但目前有分类笔记本的演示可用。人们对用于云端和本地部署的强大 DRM 系统寄予厚望。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colmweb.org/">COLM 2024</a>：未找到描述</li><li><a href="https://docs.marqo.ai/2.10/">Getting Started with Marqo - Marqo docs</a>：未找到描述</li><li><a href="https://x.com/dudeman6790/status/1813020953577988565?t=yLDcQxbyS4q6U5UP_f0xnw&s=19">来自 RomboDawg (@dudeman6790) 的推文</a>：@youliang_yuan 伙计，他在提供 AI 安全，而我正在将 AI 从你们糟糕的审查中解放出来。创建完全无审查的数据集，并发布开源权重的无审查模型，以与封闭模型竞争...</li><li><a href="https://x.com/dudeman6790/status/1813020953577988565?t=yLDcQxbyS4q6U5UP_f0x">来自 RomboDawg (@dudeman6790) 的推文</a>：@youliang_yuan 伙计，他在提供 AI 安全，而我正在将 AI 从你们糟糕的审查中解放出来。创建完全无审查的数据集，并发布开源权重的无审查模型，以与封闭模型竞争...</li><li><a href="https://tenor.com/view/sample-contract-nda-non-disclosure-agreement-gif-17773157">Sample Contract GIF - Sample Contract Nda - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/shocked-surprised-gasp-what-cat-shock-gif-11368945723132907566">Shocked Surprised GIF - Shocked Surprised Gasp - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：作为对克利奥帕特拉（Cleopatra）的致敬，她的辉煌命运终结于悲惨的毒蛇事件，我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用。</li><li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>：作为对阿基米德（Archimedes）的致敬，今年是我们庆祝他 2311 周年诞辰，我们自豪地发布我们的第一个 Mathstral 模型，这是一个专门为数学推理和科学发现设计的 7B 模型...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1262487568822566943)** (27 条消息🔥): 

> - `DCLM Baseline`
> - `模型性能`
> - `RTX 4090 vs 3060`
> - `Eureka Labs AI`
> - `Mistral 的新发布` 


- **Apple 发布 DCLM-Baseline-7B**：Apple 发布了 [DCLM-Baseline-7B](https://huggingface.co/apple/DCLM-Baseline-7B)，这是一个拥有 70 亿参数的语言模型，在 DCLM-Baseline 数据集上训练，上下文长度为 2048 tokens。
   - **8K 上下文长度版本**也已[发布](https://huggingface.co/apple/DCLM-Baseline-7B-8k)，论文可以在[这里](https://arxiv.org/abs/2406.11794)找到。
- **RTX 4090 与多张 3060 用于 AI 的对比**：针对购买一张 RTX 4090 还是多张 3060 用于 AI 任务展开了辩论，综合考虑了 VRAM 需求和性能。
   - 分享了一份**专家意见**，指出 Unsloth 目前仅支持单 GPU，多 GPU 支持保留给付费许可。
- **Eureka Labs AI 教育公司**：Andrej Karpathy 宣布成立 [Eureka Labs](https://x.com/karpathy/status/1813263734707790301)，旨在构建 AI 原生的教育平台和课程。
   - 他们的第一个产品 **LLM101n** 是一门本科级别的课程，内容关于训练自己的 AI，并计划开设数字和实体班级。
- **Mistral 发布两个新模型**：Mistral 发布了专注于代码任务的 [Mamba-Codestral-7B-v0.1](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) 和用于数学及科学任务的 [Mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1)，两者均采用 Apache 2.0 协议并支持 32k 上下文。
   - 该代码模型目前**不被** Llama.cpp 支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1813263734707790301">来自 Andrej Karpathy (@karpathy) 的推文</a>: ⚡️ 很高兴分享我正在创办一家名为 Eureka Labs 的 AI+教育公司。公告：--- 我们是 Eureka Labs，我们正在建立一种 AI 原生的新型学校。我们如何...</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/mathstral-7B-v0.1">mistralai/mathstral-7B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e4jw0c/apple_has_released_the_weights_for_their_7b_dclm/">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1262639087169441895)** (132 条消息🔥🔥): 

> - `模拟 Pretraining`
> - `在特定领域 PDF 上微调 LLM`
> - `RunPod 训练问题`
> - `多 GPU 支持`
> - `导出模型与推理方法` 


- **通过调整参数模拟 Pretraining**：建议通过将 r=16 增加到 r=256 并降低 learning rate 来模拟 Pretraining，从而达到与 full fine-tuning 98-99% 接近的效果。
   - *theyruinedelise*: “你可以通过将 r=16 增加到 r=256 然后降低 learning rate 来模拟 Pretraining。”
- **在特定领域 PDF 上微调 LLM**：成员们讨论了使用特定领域 PDF 微调 LLM 的方法，推荐使用 [synthetic dataset generation](https://github.com/e-p-armstrong/augmentoolkit) 和简单的 pdf2text 转换。
   - *mrdragonfox*: “不会有手把手的指导——你寻找的是 synthetic dataset generation... 因为 LLM 只理解文本。”
- **RunPod 训练问题及解决方案**：一位成员遇到了尽管 GPU 利用率正常但 RunPod 训练突然停止的问题，建议减少数据集大小以验证设置。
   - *mrdragonfox*: “将数据集削减到 5k 进行快速评估... 甚至只需 1k。”
- **Unsloth 即将推出的多 GPU 支持**：Unsloth 的多 GPU 版本正在测试中，很快就会发布；它不会向公众开放，而是一个付费产品。
   - *mrdragonfox*: “是的，因为我有多 GPU 版本... 应该很快了，我认为现在正在测试中。”
- **导出模型与推理方法详解**：详细讨论了使用 LoRA adapters 和 merged models 导出模型的方法，包括使用 vllm 和 aphro 等推理框架以获得更好性能的优势。
   - *flail_*: “真正的推理库快得多... 对同时进行多个输出有很好的 batching 支持。”


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/explodinggradients/ragas">GitHub - explodinggradients/ragas: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</a>：针对你的 Retrieval Augmented Generation (RAG) 流水的评估框架 - explodinggradients/ragas</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)!</a>：将算力和书籍转换为 Instruct-Tuning 数据集（或分类器）！ - e-p-armstrong/augmentoolkit</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>：用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产环境的机器学习流水线。 - GitHub - Unstructured-IO/unstructured: Open source librar...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1262508919004336270)** (17 条消息🔥): 

> - `LLaMA-405B`
> - `Q-Sparse`
> - `ColPali`
> - `AgentInstruct`
> - `Adam-mini` 


- **以更低成本运行 LLaMA-405B**：成员们讨论了以更低成本运行 **LLaMA-405B** 的可能性，引用了来自 @teortaxesTex 的一条推文。
   - 对话中提到了一篇关于 **Q-Sparse** 的新论文，该论文提倡完全稀疏激活的 Large Language Models。
- **ColPali 实现高性能**：**ColPali** 模型（与 **PaliGemma-3B** 和 **ColBERT** 策略相关）因其卓越的文档检索能力而受到关注。
   - 正如多次讨论中所指出的，ColPali 的 **ColBERT** 延迟交互（late interaction）特性使其性能显著优于 **BiPali** 模型。
- **AgentInstruct 改进合成数据**：Microsoft Research 推出了 **AgentInstruct**，这是一个利用多 Agent 工作流生成高质量合成数据的自动化框架。
   - 该框架显著提升了 **Orca-3** 模型在 AGIEval 和 GSM8K 基准测试中的表现。
- **Adam-mini 减少显存占用**：**Adam-mini** 是一种与 **PyTorch** 的分布式训练代码库 Torchtitan 兼容的新型优化器，据称其显存占用比 AdamW 减少了近 **50%**。
   - “Adam-mini 在性能上没有妥协，并且只需更改一行代码即可支持多个多 GPU 训练框架。”


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/">使用视觉语言模型进行 PDF 检索</a>：将 ColPali 模型与 Vespa 连接，用于复杂文档格式检索。</li><li><a href="https://x.com/RuoyuSun_UI/status/1811818970573603112">来自 Ruoyu Sun (@RuoyuSun_UI) 的推文</a>：Adam-mini 更新：Adam-mini 现在与 @PyTorch 最新的分布式训练代码库 "Torchtitan" https://github.com/pytorch/torchtitan 兼容。查看 Llama3 上的损失曲线...</li><li><a href="https://x.com/teortaxesTex/status/1813048518132506656?t=-gzQhY9OZKso0NvnnVq_YA&s=19">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：我想我们将能够以足够便宜的价格运行 LLaMA-405B。引用 gm8xx8 (@gm8xx8) 的话：Q-Sparse：所有 Large Language Models 都可以完全稀疏激活。论文：https://arxiv.org/abs/2407.10969</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.03502">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>：合成数据对于加速 Large Language Models（无论大小）的开发变得越来越重要。尽管有几个成功的用例，研究人员也提出了担忧...</li><li><a href="https://x.com/lateinteraction/status/1813140776869658833">来自 Omar Khattab (@lateinteraction) 的推文</a>：没错。即使事后看来，令人惊讶的是，表现极好（ColPali 为 81.3%）和完全无效（BiPali 为 58.8%）之间的区别在于 ColPa... 中的 "Col" 部分。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1262486230659104839)** (32 条消息🔥): 

> - `FlatBuffers vs Protobuf`
> - `AMD 徽标颜色讨论`
> - `Mojo GitHub 搜索问题`
> - `MAX SDK 的开源状态`
> - `Mojo 演讲的 YouTube 链接` 


- **FlatBuffers 采用见解**：讨论强调了 [FlatBuffers](https://flatbuffers.dev/) 具有高性能，但与 Protobuf 相比，其使用难度较大且行业采用率较低。
   - 一位成员提到 [Apache Arrow](https://arrow.apache.org/) 在内部使用 FlatBuffers，但在数据传输方面依赖 Protobuf，这表明了行业的偏好。
- **AMD 徽标颜色困惑**：成员们讨论了 AMD 徽标中绿色的历史使用情况，参考了指示颜色随时间变化的各种 [来源](https://logos.fandom.com/wiki/AMD/Other)。
   - 一位参与者提到了个人理论，并指出当前品牌推广与历史使用之间存在不一致。
- **Mojo 语言 GitHub 搜索问题**：用户在搜索 Mojo 语言仓库时遇到了 GitHub 搜索结果不一致的情况，重复搜索时看到的结果各不相同。
   - 一位成员幽默地指出，搜索结果从 0 变成 220 等错误，表明 GitHub 的搜索功能存在问题。
- **MAX SDK 尚未开源但有逐步计划**：澄清了虽然 [MAX SDK](https://www.modular.com/legal/max) 可以免费使用，但目前尚未开源。
   - 有计划从 [GitHub](https://github.com/modularml/max) 上提供的组件开始，逐步开源其中的部分内容。
- **YouTube 上的 Mojo 演讲视频**：大家对即将发布的 Mojo 演讲表示期待，重点介绍了 [第二部分](https://www.youtube.com/watch?v=9ag0fPMmYPQ) 以及与 Chris Lattner 的深度探讨。
   - 成员们赞赏 Mojo 中集成的 Rust 特性和 Python 兼容性，认为这是系统编程和 ML 领域一个充满前景的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://capnproto.org)">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYP"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=_QVs626Vn2k">Mojo 🔥 Community Meeting #4</a>: Mojo 社区会议 #4 的录音🫓 Flat Buffers：内存高效的序列化⚒️ Forge Tools：扩展 Mojo 🔥 标准库🔄 Mojo 🔥 Gen...</li><li><a href="https://www.modular.com/legal/max">Modular: MAX Community License</a>: MAX SDK ("MAX") 社区许可证规定了我们对软件用户的期望、我们允许的用途以及对软件使用的管理。</li><li><a href="https://logos.fandom.com/wiki/AMD/Other">AMD/Other</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=8nyg_IXfnvs">Creator Of Swift On Functionial Programming</a>: 所有剪辑均来自 ThePrimeagenMe 的直播：https://twitch.tv/ThePrimeagen 联合主持人：https://twitch.tv/teej_dv Chris Lattner: https://x.com/clattner_l...</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYPQ">Mojo🔥: a deep dive on ownership with Chris Lattner</a>: 了解关于 Mojo 所有权的一切，与 Modular CEO Chris Lattner 进行深度探讨。如果您有任何问题，请务必加入我们友好的...</li><li><a href="https://github.com/search?q=language%3AMojo&type=repositories&ref=advsearch">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1812972838707687889>
  

---


### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1262582156455055400)** (1 条消息): 

> - `MAX Graph API`
> - `AI 推理流水线`
> - `Mojo` 


- **Modular 发布 MAX Graph API 教程**：Modular 在 YouTube 上发布了一个名为 [MAX Graph API 教程](https://www.youtube.com/watch?v=dhllDwVUP5s) 的新视频，讨论了如何使用 MAX Graph API 在 **Mojo** 中构建 AI 推理流水线。
   - *Ehsan M. Kermani* 在视频中详细阐述了如何开始使用 **MAX Graph API**。
- **Mojo 助力 AI 推理流水线**：**MAX Graph API** 允许使用 **Mojo** 构建完整的 AI 推理流水线。
   - 该视频为希望利用这一强大工具集的开发者提供了详细指南。



**提到的链接**: <a href="https://www.youtube.com/watch?v=dhllDwVUP5s">MAX Graph API Tutorial</a>: MAX Graph API 允许你在 Mojo 中构建整个 AI 推理流水线。在本视频中，Ehsan M. Kermani 讨论了如何开始使用 MAX Gr...

  

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1262744562083037275)** (8 条消息🔥): 

> - `Mojo 配合本地 Whisper`
> - `Mistral 7B 编程模型`
> - `Mamba 模型`
> - `Mistral 7B 的 GUI 示例`
> - `ONNX 转换` 


- **关于 Mojo 配合本地 Whisper 的咨询**：一位用户询问是否有人尝试过将 **Mojo** 与 **OpenAI** 的 **本地 Whisper** 配合使用。
   - 该查询后没有进一步的回答或展开讨论。
- **Mistral 发布基于 Mamba2 的 7B 编程模型**：Mistral 发布了一个 [7B 编程模型](https://mistral.ai/news/codestral-mamba/)，该模型使用 **Mamba2** 而非 Transformer 架构，根据 Apache 2.0 许可证可免费使用。
   - **Mamba** 模型提供线性时间推理能力，并能处理无限长度的序列，显著提升了代码生产力。
- **Mistral 7B 编程模型的 GUI 示例**：分享了一个使用 Nightly 版本为 **Mistral 7B Coding** 构建 GUI 的 GitHub 仓库，代码可在[此处](https://github.com/modularml/max/tree/nightly/examples/gui)获取。
   - 多位社区成员表现出兴趣，提供了支持并分享了更多关于 **ONNX** 转换的示例和 Notebook。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: 作为对克利奥帕特拉（Cleopatra）的致敬，她的光辉命运终结于悲惨的毒蛇事件，我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在 ... 下使用。</li><li><a href="https://mistral.ai">Mistral AI | Frontier AI in your hands</a>: 掌握前沿 AI</li><li><a href="https://github.com/modularml/max/blob/main/examples/notebooks/mistral7b-python-onnx.ipynb">max/examples/notebooks/mistral7b-python-onnx.ipynb (main 分支) · modularml/max</a>: 一系列示例程序、Notebook 和工具，旨在展示 MAX 平台的强大功能 - modularml/max</li><li><a href="https://github.com/modularml/max/tree/nightly/examples/gui">max/examples/gui (nightly 分支) · modularml/max</a>: 一系列示例程序、Notebook 和工具，旨在展示 MAX 平台的强大功能 - modularml/max</li><li><a href="https://github.com/modularml/max/blob/nightly/examples/gui/pages/bert.py">max/examples/gui/pages/bert.py (nightly 分支) · modularml/max</a>: 一系列示例程序、Notebook 和工具，旨在展示 MAX 平台的强大功能 - modularml/max</li><li><a href="https://github.com/modularml/max/blob/794cc173280b59fd9ad4a9c1fd498b633379b9b9/examples/gui/pages/llama3.py#L140">max/examples/gui/pages/llama3.py · modularml/max</a>: 一系列示例程序、Notebook 和工具，旨在展示 MAX 平台的强大功能 - modularml/max
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1262497840475934801)** (154 条消息🔥🔥): 

> - `Mojo 中的错误处理`
> - `Python 兼容模式`
> - `关于函数着色（Function Coloring）的讨论`
> - `动态类型与自动类型变量的对比`
> - `GPU 和 FPGA 错误处理` 


- **关于 Python 兼容模式的辩论**：成员们讨论了在 Mojo 中通过配置禁用与 Python 完全向后兼容的可能性，旨在强制执行更以 Mojo 为中心的语法，例如使用 `fn` 而不是 `def`。
   - 建议包括禁止类似于 Rust 单子（monadic）错误处理的异常处理，以提高代码的健壮性和可读性。
- **Mojo 中的错误处理技术**：围绕错误处理展开了热烈讨论，提议显式标记会抛出特定错误的函数，类似于 Rust 的 Result 类型。
   - 成员们对堆栈回溯（stack unwinding）以及在没有妥善处理的情况下引入新错误类型导致 API 变更困难的问题表示担忧。
- **动态类型与自动类型变量的探索**：成员们辩论了动态类型是否总是比自动类型快，讨论强调了预分配内存和提前检查类型转换的好处。
   - 有人指出，实际的运行时经验可能并不总是一致，开发者不应做出笼统的假设。
- **函数着色及其对代码健壮性的影响**：辩论了函数着色（function coloring）问题，对比了抛出错误的函数和异步函数，重点在于函数是否应该传播所有潜在错误。
   - 许多人同意，抛出错误应该被显式声明，以增加妥善处理的机会，从而增强代码的健壮性。
- **GPU 和 FPGA 环境下的错误处理**：对于 Mojo 的错误处理如何在 GPU 和 FPGA 等非 CPU 设备上运行，存在重大关注。
   - 成员们强调需要一种避免堆栈操作的错误处理方式，以符合硬件限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1746,">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://youtu.be/_QVs626Vn2k?t=1390)">Mojo 🔥 Community Meeting #4</a>: Mojo 社区会议 #4 的录音 🫓 Flat Buffers：内存高效的序列化 ⚒️ Forge Tools：扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://youtu.be/_QVs626Vn2k?t=16740)">Mojo 🔥 Community Meeting #4</a>: Mojo 社区会议 #4 的录音 🫓 Flat Buffers：内存高效的序列化 ⚒️ Forge Tools：扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Studio">no title found</a>: 未找到描述</li><li><a href="https://github.com/martinvuyk/forge-tools/blob/main/src/forge_tools/collections/result.mojo">forge-tools/src/forge_tools/collections/result.mojo at main · martinvuyk/forge-tools</a>: 扩展 Mojo 标准库功能的工具 - martinvuyk/forge-tools
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1262489893758832681)** (34 messages🔥): 

> - `Modular Exclusive Partnership` (Modular 独家合作伙伴关系)
> - `NVIDIA MAX Platform Support` (NVIDIA MAX 平台支持)
> - `MAX Graph API Tutorial Issues` (MAX Graph API 教程问题)
> - `MAX Tensor Imports` (MAX Tensor 导入)
> - `Reliability of MAX Installations` (MAX 安装的可靠性)


- **Modular 与 NVIDIA 的独家合作伙伴关系细节发生变化**：[Modular 的公告](https://web.archive.org/web/20231204230430/https://www.modular.com/blog/modular-partners-with-nvidia-to-bring-gpus-to-the-max-platform)中删除了“独家（exclusive）”一词，该公告此前曾强调双方的独家技术合作。
   - 一位成员注意到了这一变化，并讨论了“独家合作伙伴关系”在法律和技术层面可能意味着什么的细微差别。
- **MAX Graph API 教程中发现的问题**：用户报告了在按照 MAX Graph API 教程操作时遇到的各种问题，包括 Python 和 Mojo 脚本结果之间的差异以及安装过程中的错误。
   - 一位用户提到由于使用了错误的 Jupyter kernel 导致导入错误，而另一位用户则指出了示例代码中 `relu6` 激活函数的问题。
- **MAX Tensor 和 Graph 导入问题**：一位用户因为使用了 Python Jupyter kernel 而不是 Mojo 导致无法导入 `max.tensor` 或 `max.graph`，社区成员对此进行了澄清。
   - 在切换到正确的 kernel 后，该用户能够成功继续教程。
- **MAX 安装和导出路径困惑**：几位用户面临 MAX 安装问题，特别是与导出路径和使用正确版本相关的问题。
   - 一位成员建议使用 nightly 构建版本以获得更可靠的安装体验，这为一些用户解决了问题。
- **请求 MAX 提供更详细的报告功能**：一位用户请求 MAX 提供更详细的报告，包括 GFlops 和执行时间等指标，以便更好地进行硬件和财务决策。
   - 他们强调需要这些指标来在不同的硬件设置上有效地扩展 MAX 的使用。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/max-graph-api-tutorial">Modular: MAX Graph API tutorial</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：MAX Graph API 教程</li><li><a href="https://web.archive.org/web/20231204230430/https://www.modular.com/blog/modular-partners-with-nvidia-to-bring-gpus-to-the-max-platform">Modular: Modular partners with NVIDIA to bring GPUs to the MAX Platform</a>：我们正在为世界构建下一代 AI 开发者平台。阅读我们关于 Modular 如何与 NVIDIA 合作将 GPU 引入 MAX 平台的最新文章。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1262550767144013924)** (53 messages🔥): 

> - `VSCode nightly extension for LSP` (用于 LSP 的 VSCode nightly 扩展)
> - `Proposal for statuses on PRs` (关于 PR 状态的提案)
> - `ComplexSIMD vector implementation` (ComplexSIMD 向量实现)
> - `Handling reviews and discussions in PRs` (处理 PR 中的评审和讨论)


- **VSCode nightly 扩展修复了 LSP 问题**：用户讨论了如何正确配置 VSCode 插件以指向 nightly 版本的 LSP，强调需要禁用 stable 扩展以便 nightly 扩展接管。
   - 其他成员分享了他们的方法，例如卸载并重新安装扩展，并确保正确设置了 bash profile 路径。
- **关于改进 PR 状态的提案**：一位成员提议在 PR 中增加更详细的状态，如“已阻塞/已暂停”、“未评审”和“疑问或讨论”，以提高沟通效率。
   - 社区建议使用 refined-github 并自动标记标签，以节省维护者的时间。
- **关于 ComplexSIMD 向量结构的辩论**：在讨论 ComplexSIMD 时，成员们质疑是使用两个底层的 SIMD 向量，还是使用单个拆分为实部和虚部的 SIMD 向量。
   - *Benny.n* 主张使用单个 SIMD 向量可以获得效率提升，并提议重写乘法和除法等操作的公式。



**提及的链接**：<a href="https://github.com/refined-github/refined-github">GitHub - refined-github/refined-github: :octocat: Browser extension that simplifies the GitHub interface and adds useful features</a>：:octocat: 简化 GitHub 界面并添加实用功能的浏览器扩展 - refined-github/refined-github

  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/)** (1 messages): 

ModularBot: 恭喜 <@585884735134236685>，你刚刚升到了 1 级！
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1262780662180614196)** (1 messages): 

_

> - `AI 数学奥林匹克竞赛冠军开源`
> - `Whisper Timestamped 发布`
> - `Nvidia BigVGAN v2`
> - `Hugging Face 与 Keras NLP 集成`
> - `Hugging Face Tokens UI 重构` 


- **AI 数学奥林匹克竞赛冠军开源**：AI 数学奥林匹克竞赛冠军 **NuminaMath** 现已[开源](https://x.com/reach_vb/status/1811069858584363374)，其 **7B 模型**在 Kaggle 测试集上得分为 **29/50**，采用 Apache 2.0 协议授权。
   - 它在大型数学数据集和合成工具集成推理数据集上采用了两阶段微调过程，并针对 **GPT-4** 输出采用了 MSFT 的 ToRA 格式。
- **Whisper Timestamped：本地语音识别**：**Whisper Timestamped** 推出了[多语言语音识别](https://x.com/xenovacom/status/1811068015229747335)，支持单词级时间戳，并使用 **Transformers.js** 完全在浏览器中运行。
   - 这为浏览器内视频编辑开启了新的可能性，并提供了完整的源代码和演示。
- **Nvidia BigVGAN v2 发布**：Nvidia 发布了 [BigVGAN v2](https://x.com/reach_vb/status/1813181163126587830)，这是一款 SoTA Neural Vocoder，可从 Mel 频谱生成波形，在 A100 上的推理速度更快。
   - 改进包括优化的 CUDA 内核、更好的判别器和损失函数，且该模型支持高达 **44 kHz 的采样率**。
- **Hugging Face 与 Keras NLP 集成**：Hugging Face 宣布了与 **Keras** 的全新 [NLP 集成](https://huggingface.co/blog/keras-nlp-integration)。
   - 此次合作旨在增强 Keras 中的 NLP 能力，为开发者提供无缝集成。
- **Hugging Face Tokens UI 重构**：Hugging Face 翻新了平台上的 [Token 管理 UI](https://x.com/coyotte508/status/1810982231180992521)，增加了诸如最后使用时间、最后四位字符或轮换日期等新功能。
   - 这些改进使得管理所有 Token 变得更加容易，详细信息一目了然。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/reach_vb/status/1811069858584363374">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：AI Math Olympiad 冠军现已 Open Source！🔥 > 7B model，在 Kaggle 公开和私有测试集上得分 29/50。（Apache 2.0 授权）。> Base model：deepseek-math-7b-base > 两阶段 f...</li><li><a href="https://x.com/reach_vb/status/1812916171902976256)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：AI Math Olympiad 冠军 - 在 Mac 上运行！100% local 🔥 brew install llama.cpp llama-cli --hf-repo reach-vb/NuminaMath-7B-TIR-Q8_0-GGUF --hf-file numinamath-7b-tir-q8_0.gguf -p "For h...</li><li><a href="https://x.com/xenovacom/status/1811068015229747335)">Xenova (@xenovacom) 的推文</a>：介绍 Whisper Timestamped：支持词级时间戳的多语言语音识别，得益于 🤗 Transformers.js，可 100% 在你的 browser 中 local 运行！这为...开启了无限可能。</li><li><a href="https://x.com/reach_vb/status/1813181163126587830)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Nvidia 发布了 BigVGAN v2！🎧 SoTA Neural Vocoder - Mel spectrogram 到 waveform generator 🔥 > 用于 Inference 的自定义 CUDA kernel：带有 fused upsampling + activation kernel，在...上 Inference 速度提升高达 3 倍</li><li><a href="https://x.com/coyotte508/status/1810982231180992521)">coyotte508 (@coyotte508) 的推文</a>：http://hf.co/settings/tokens UI 重大更新！UI 界面更便于管理你所有的 Tokens，增加了如最后使用时间、最后四位字符或轮换日期等信息。注意：部分信息仅在...</li><li><a href="https://huggingface.co/settings/tokens)">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://x.com/julien_c/status/1812099420726456457)">Julien Chaumond (@julien_c) 的推文</a>：来自 @huggingface datasets 团队的酷炫周末更新，你现在可以将我们的 viewer 嵌入到任何网页中 😎（寻找 “Embed” 按钮。）让我们知道你的想法！</li><li><a href="https://x.com/htahir111/status/1813132485267443843)">Hamza Tahir (@htahir111) 的推文</a>：@julien_c @huggingface @zenml_io 写了一篇博客，描述了使用 #oss 实现它是多么相对简单：https://www.zenml.io/blog/embedding-huggingface-datasets-visualizations-with-zenml</li><li><a href="https://x.com/mervenoyann/status/1812839137398886420)">merve (@mervenoyann) 的推文</a>：公告 🗣️ 我们在 6 月持续发布，这里有一些 @huggingface Hub 的更新汇总（非详尽）</li><li><a href="https://x.com/abhi1thakur/status/1812808539963892018)">abhishek (@abhi1thakur) 的推文</a>：我们刚刚取消了在 Hugging Face 上创建竞赛时必须为组织绑定支付方式的要求 🚀 现在，大学、组织和个人都可以创建 free-tier ...</li><li><a href="https://x.com/Wauplin/status/1811382409683689479)">Wauplin (@Wauplin) 的推文</a>：🚀 令人兴奋的更新！𝚑𝚞𝚐𝚐𝚒𝚗𝚏𝚊𝚌𝚎_𝚑𝚞𝚋 的 InferenceClient 现在支持 OpenAI 的 client syntax。只需 3 行代码即可切换到 Open Source LLMs！查看无缝衔接的...</li><li><a href="https://x.com/dylan_ebert_/status/1812952230825500914)">dylan (@dylan_ebert_) 的推文</a>：🎉 好消息 🤗 Machine Learning for 3D 课程的最后几个单元已经上线！🛠️ 构建你自己的 Generative 3D demo 🎓 在 Hugging Face 上免费获取认证 & Open Source</li><li><a href="https://www.youtube.com/watch?v=HcpUP-q2Z0w&ab_channel=HuggingFace)">One Minute Gradio #2: Event Chaining</a>：One Minute Gradio #2 - 快速学习 Gradio 技巧！今天，我们将讨论在 Gradio 中运行连续事件（即 Event Chaining），特别是使用...</li><li><a href="https://x.com/_philschmid/status/1811416175865122954)">Philipp Schmid (@_philschmid) 的推文</a>：LLM Evaluation 不需要很复杂。你不需要复杂的 pipelines、databases 或 infrastructure 组件就能开始构建有效的 Evaluation pipeline。👀 Blog: https...
</li>
</ul>

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1262489924846878841)** (235 条消息🔥🔥): 

> - `排查 Hugging Face Spaces 的问题`
> - `GPTs Agent 的学习能力`
> - `处理 LLM 中未知词汇的 Tokenization`
> - `专用 Agent 的合并技术`
> - `验证 3D Mesh 对象相似性的模型` 


- **解决 Hugging Face Spaces 运行时错误**：成员们讨论了在 Hugging Face Spaces 中遇到的各种运行时错误，包括在该平台托管的模型中出现的 CUDA 错误和 Tokenizer 问题。
   - 建议了一些排查步骤，例如更改数据集大小、更新 diffusers 版本以及设置缓存目录，尽管这些并未解决所有问题。
- **GPTs Agent 在初始训练后无法学习**：有人担心 GPTs Agent 无法从初始训练后提供的额外信息中学习。会议澄清了上传的文件被保存为“知识”文件，但不会修改 Agent 的基础知识。
   - 这引发了关于 GPTs Agent 在动态更新其知识库方面的能力和局限性的进一步讨论。
- **未知词汇的 Tokenization 挑战**：有人提出了 LLM 如何使用不在其 Tokenizer 中的词汇的问题，从而引出了关于 Tokenization 策略和 Sub-word Tokenization 技术的解释。
   - 成员们讨论了词表大小（Vocab size）的复杂性，以及不同的 Tokenizer 如何处理未知词汇，包括使用更小的 Token 来构建新词。
- **利用专用数据集增强 LLM 性能**：一位成员分享了在专用逻辑和推理数据集上训练 LLM 的经验，提到了 Orca 和 Tess 等特定数据集。
   - 使用精心策划的数据集来提高模型性能的想法激发了进一步的兴趣和讨论。
- **选择用于 3D Mesh 对象验证的模型**：一位成员寻求帮助，以确认用于计算 Prompt 与 3D Mesh 对象验证流水线预览图像之间相似性的模型。
   - 建议包括使用 CLIP 模型进行相似性计算，并确保有适当的验证机制以进行准确评估。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/phew-robertdowneyjr-tonystark-ironman-avengers-gif-2884296381752559184">Phew Robertdowneyjr GIF - Phew RobertDowneyJr TonyStark - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/soldier-ww2-traumatized-meme-eyes-gif-12257475272172704406">Soldier Ww2 GIF - Soldier Ww2 Traumatized - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF/tree/main">bartowski/gemma-2-27b-it-GGUF at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/scooby-doo-mystery-machine-cartoon-old-school-smoking-gif-16100024">Scooby Doo Mystery Machine GIF - Scooby Doo Mystery Machine Cartoon - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/nroggendorff/epicrealismxl">epiCRealism XL - a Hugging Face Space by nroggendorff</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/animexl">Anime Diffusion XL - a Hugging Face Space by nroggendorff</a>: 未找到描述</li><li><a href="https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint">community-events/jax-controlnet-sprint at main · huggingface/community-events</a>: 开发者可以为 🤗 社区活动做出贡献的地方 - huggingface/community-events</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava/commit/3950336734fa093dc80ac7e5860251de9e11e26b">Update README.md · nroggendorff/llava at 3950336</a>: 未找到描述</li><li><a href="https://tenor.com/view/sonic-running-funny-weird-gif-14261571">Sonic Running GIF - Sonic Running Funny - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/urchade/GLiNER">GitHub - urchade/GLiNER: Generalist and Lightweight Model for Named Entity Recognition (Extract any entity types from texts) @ NAACL 2024</a>: 通用且轻量级的命名实体识别模型（从文本中提取任何实体类型）@ NAACL 2024 - urchade/GLiNER</li><li><a href="https://huggingface.co/spaces/tomaarsen/gliner_medium-v2.1">GLiNER-medium-v2.1, zero-shot NER - a Hugging Face Space by tomaarsen</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava/discussions/2">nroggendorff/llava · idiot noa... says llama....</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1262628134260314183)** (2 条消息): 

> - `K-Means Clustering 视频`
> - `UDOP 论文讨论` 


- **K-Means Clustering 的 YouTube 教程**：分享了一个[名为“K-Means Clustering (ML pt 5)”的 YouTube 视频](https://youtu.be/x1Dcg4JWARY)，该视频对 K-Means Clustering 进行了通俗易懂且简短的介绍。
- **关于 UDOP 论文图像重建的疑问**：一名成员询问在 **UDOP** 论文中，图像重建时标题和序列号的字体样式是如何保留的。



**提到的链接**：<a href="https://youtu.be/x1Dcg4JWARY">K-Means Clustering ( ML pt 5 )</a>：在这段视频中，我将讲解 K-Means Clustering (k-MC)。这将是一个友好且简短的介绍，就像播放列表中的所有其他视频一样……

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1262761173858914316)** (8 条消息🔥): 

> - `Happy Dog Detection`
> - `检索教程`
> - `在线审查的影响`
> - `Mistral AI 模型`
> - `Llama3 405b` 


- **Happy Dog Detection 项目更新**：一名成员分享了 [Happy Dog Detection GitHub 仓库](https://github.com/Matthew-AI-Dev/Happy_Dog_Detection)的链接，该项目有助于狗狗检测机器学习模型的开发和贡献。
   - *通过在 GitHub 上创建账户，为 Matthew-AI-Dev/Happy_Dog_Detection 的开发做出贡献。*
- **面向爱好者的检索教程**：一名成员为检索爱好者发现了一个宝贵的资源，并分享了 [FullStackRetrieval-com/RetrievalTutorials GitHub 仓库](https://github.com/FullStackRetrieval-com/RetrievalTutorials)。
   - *通过在 GitHub 上创建账户，为 FullStackRetrieval-com/RetrievalTutorials 的开发做出贡献。*
- **关于在线审查影响的 PETS24 论文**：讨论了**在线审查对大语言模型（LLM）的影响**，一名成员分享了关于该主题的 [PETS24 论文](https://www.petsymposium.org/foci/2024/foci-2024-0006.pdf)。
- **Mistral AI 发布两个模型**：**Mistral AI** 宣布发布两个新的 AI 模型：[Mathstral](https://mistral.ai/news/mathstral/) 和 [Codestral](https://mistral.ai/news/codestral-mamba/)。
   - 该成员还在 X（原 Twitter）上分享了[官方公告链接](https://x.com/MistralAI/status/1813222156265791531)。
- **Llama3 405b 已添加到 OpenRouterAI**：一名成员注意到 **Llama3 405b** 已添加到 [OpenRouterAI](https://x.com/HCSolakoglu/status/1812984327510085883)，但尚未确定提供商（provider）。
   - 另一名成员推测，鉴于附带了 Huggingface 模型权重页面，发布日期可能已经临近。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/HCSolakoglu/status/1812984327510085883">来自 Hasan Can (@HCSolakoglu) 的推文</a>：Llama3 405b 已添加到 @OpenRouterAI，目前还没有提供商。同时附上了 Huggingface 模型权重页面。发布一定很近了。</li><li><a href="https://x.com/MistralAI/status/1813222156265791531">来自 Mistral AI (@MistralAI) 的推文</a>：https://mistral.ai/news/mathstral/ https://mistral.ai/news/codestral-mamba/</li><li><a href="https://github.com/Matthew-AI-Dev/Happy_Dog_Detection">GitHub - Matthew-AI-Dev/Happy_Dog_Detection</a>：通过在 GitHub 上创建账户，为 Matthew-AI-Dev/Happy_Dog_Detection 的开发做出贡献。</li><li><a href="https://github.com/FullStackRetrieval-com/RetrievalTutorials">GitHub - FullStackRetrieval-com/RetrievalTutorials</a>：通过在 GitHub 上创建账户，为 FullStackRetrieval-com/RetrievalTutorials 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1262647301050994698)** (1 条消息): 

> - `NLP Roadmap`
> - `NLP Projects Repository`
> - `NLP Historical Overview`
> - `NLP TOC` 


- **查看 NLP Roadmap GitHub 仓库！**：一位用户分享了一个 [NLP 项目的 GitHub 仓库](https://github.com/kjdeveloper8/nlp-projects)，其中包含为 NLP 爱好者准备的全面路线图。
   - 该仓库旨在引导学习者了解 Natural Language Processing (NLP) 的不同阶段和技术。
- **Medium 上发布的全面 NLP Roadmap 文章**：分享了一篇富有启发性的 [NLP Roadmap Medium 文章](https://medium.com/@krinaljoshi/nlp-roadmap-2740a1029af2)，提供了 NLP 技术的历史概览和演变。
   - 文章引用了 20 世纪 30 年代中期的基础工作、Noam Chomsky 在 1957 年的贡献，以及 20 世纪 80 年代机器学习算法的影响。
- **涵盖的 NLP 核心主题**：NLP Roadmap 包含核心主题，如 [Basics Of NLP](#84be)（NLP 基础）、[Text preprocessing](#8eb3)（文本预处理）、[Parser](#7dcf)（解析器）、[Text encoding](#a95f)（文本编码）、[Text classification](#dca0)（文本分类）和 [Text similarity](#733e)（文本相似度），提供了一条结构化的学习路径。
   - *“语言是一个文化的路线图。它告诉你这个民族从哪里来，要到哪里去。”* —— 文章引用了 Rita Mae Brown 的话，以强调语言的文化影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/kjdeveloper8/nlp-projects">GitHub - kjdeveloper8/nlp-projects: NLP Roadmap</a>：NLP Roadmap。通过在 GitHub 上创建账号来为 kjdeveloper8/nlp-projects 的开发做出贡献。</li><li><a href="https://medium.com/@krinaljoshi/nlp-roadmap-2740a1029af2">NLP Roadmap</a>：“语言是一个文化的路线图。它告诉你这个民族从哪里来，要到哪里去。” —— Rita Mae Brown
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1262838330660360255)** (1 条消息): 

> - `Best LLM for course-specific AI model`
> - `Video transcription tools`
> - `Fine-tuning on low-end hardware` 


- **适用于特定课程 AI 模型的最佳免费 LLM**：Ritikkumarv 正在寻找一个开源 LLM，以创建一个能够回答特定课程研究生级别问题、处理 PDF、PPT 和视频讲座转录的 AI 模型。
- **视频转录工具讨论**：Ritikkumarv 询问 OpenAI 的 Whisper 是否是视频转录的最佳免费工具，或者是否有其他可行的选择。
- **Fine-tuning 和低端硬件使用的建议**：Ritikkumarv 还请求一份关于使用免费开源软件 Fine-tuning 模型的逐步指南，并考虑到了低端硬件的限制。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262627046702452776)** (1 条消息): 

> - `Skin Cancer Detection`
> - `3D Images`
> - `Kaggle Competitions` 


- **关于 Kaggle 中 3D 图像皮肤癌检测的讨论**：一位成员询问是否有人在 Kaggle 上从事 **3D 图像** 的 **皮肤癌检测** 工作。
   - *在提供的消息历史记录中没有讨论具体的回复或链接。*
- **一般查询**：讨论主要涉及对 Kaggle 上与 **3D 图像皮肤癌检测** 相关的正在进行项目的查询。
   - *消息历史记录中未提供进一步的细节或资源。*


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1262670965213761547)** (9 条消息🔥): 

> - `NLP 从基础到进阶的课程推荐`
> - `Google Colab 和 GPU 问题`
> - `图像 Embeddings 和潜在偏差`
> - `Sentence Transformers：多负例 vs 单负例`
> - `Faiss 索引中的向量分布` 


- **NLP 学习路径推荐**：一位成员建议从 HuggingFace 课程或 Andrew Ng 的课程开始学习 NLP，并推荐了像微调 NER 模型这样的实战项目。
   - 建议是：*'在学习核心概念的同时，不断尝试新的项目'*。
- **Google Colab T4 GPU 问题**：一位成员在安装了所需包（`transformers[torch]` 和 `accelerate`）后，在 Google Colab 中使用 T4 GPU 时遇到错误。
   - 他们询问是否有人有解决这些问题的思路。
- **图像 Embeddings 偏差**：一位成员建议关注图像 Embeddings，但提醒注意生成的图像描述中可能存在的偏差。
   - *'如果在生成图像描述时出现任何幻觉（hallucinations），这可能会成为偏差的来源。'*
- **Sentence Transformers 中的多负例**：一位成员询问了使用多负例与单负例训练 Sentence Transformers 的质量差异，以及选择难负例（hard negatives）的过程。
   - 他们分享了对数据加载以及 `MultipleNegativesRankingLoss` 类使用的困惑，并链接了相关文档（[TripletReader](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/readers/TripletReader.py), [MultipleNegativesRankingLoss](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)）。
- **向量分布与 Faiss 索引**：一位成员专注于确保其 Faiss 索引中的向量均匀分布，以提高 k-nearest neighbors (knn) 搜索的准确性。
   - 他们强调了每个锚点（anchor）拥有更多负例以及数据多样性对于更好的向量各向同性（vector isotropy）的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/readers/TripletReader.py">sentence-transformers/sentence_transformers/readers/TripletReader.py at master · UKPLab/sentence-transformers</a>：使用 BERT 的多语言句子和图像 Embeddings - UKPLab/sentence-transformers</li><li><a href="https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py">sentence-transformers/sentence_transformers/losses/MultipleNegativesRankingLoss.py at master · UKPLab/sentence-transformers</a>：使用 BERT 的多语言句子和图像 Embeddings - UKPLab/sentence-transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1262796895420547144)** (1 条消息): 

> - `Gradio 中 ViteJS 的使用`
> - `ViteConf 合作伙伴关系`
> - `Gradio 的自定义组件开发模式` 


- **Gradio 利用 ViteJS 增强开发体验**：Gradio 团队长期以来一直使用 [ViteJS](https://vitejs.dev)，显著提升了他们的开发体验，并从 4.0 版本开始通过自定义组件开发模式将 Vite 开放给用户。
- **Gradio 与 ViteConf 合作举办 24 小时会议**：Gradio 正在与 [ViteConf](https://viteconf.org) 合作，这是一个探索 Vite 生态系统最新动态的 24 小时会议。
   - 该会议完全免费，用户可以在[此处](https://viteconf.org/24/ecosystem/huggingface)报名。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vitejs.dev>">未找到标题</a>：未找到描述</li><li><a href="https://viteconf.org>">未找到标题</a>：未找到描述</li><li><a href="https://viteconf.org/24/ecosystem/huggingface">HuggingFace 邀请您参加 ViteConf</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1262523391857000579)** (243 条消息🔥🔥): 

> - `AI Morph 与 NSFW 内容`
> - `利用 Stable Diffusion 制作动漫风格`
> - `YouTube 教程推荐`
> - `本地 AI 工具开发`
> - `AI 模型的 GPU 对比` 


- **AI Morph 不再支持 NSFW**：一名成员抱怨 Daily Joy 工作室开发的 App AI Morph 停止支持 NSFW 内容，并显示“不符合指南”的消息。
- **Stable Diffusion 与动漫风格迁移**：一名成员询问如何在 Stable Diffusion 中保持动漫风格图像的准确性，寻求对颜色、服装和表情的控制。
- **Scott Detweiler 获评最佳教程**：*crystalwizard* 推荐了 Scott Detweiler 的 YouTube 频道来学习 Stable Diffusion，并强调了他在 Stability.ai 负责质量保证（QA）的工作。
- **集成 SD 的本地 AI 工具开发**：*capt.asic* 讨论了他们开发的集成 Stable Diffusion 和 LLM 支持的本地 AI 工具，并提到这是他们的首选工具。
- **GPU 之战：4090 vs 7900XTX**：社区讨论了 NVIDIA 4090 与 AMD 7900XTX 的优劣，重点关注价格和可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@sedetweiler">Scott Detweiler</a>：Stability.ai 的质量保证负责人 & PPA 大师级专业摄影师。大家好！我是 Stability.ai 的 QA 负责人，也是一名常驻密尔沃基附近的专业摄影师和修图师...</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++</a>：纯 C/C++ 实现的 Stable Diffusion。欢迎在 GitHub 上通过创建账号为 leejet/stable-diffusion.cpp 的开发做出贡献。</li><li><a href="https://github.com/ssitu/ComfyUI_UltimateSDUpscale/discussions/65">ultimate sd upscale in comfyui. keep getting error · ssitu/ComfyUI_UltimateSDUpscale · Discussion #65</a>：无法为自定义节点导入 C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_UltimateSDUpscale 模块：无法从 'modules' 导入名称 'devices'...</li><li><a href="https://github.com/axodox/axodox-machinelearning">GitHub - axodox/axodox-machinelearning: This repository contains a pure C++ ONNX implementation of multiple offline AI models, such as StableDiffusion (1.5 and XL), ControlNet, Midas, HED and OpenPose.</a>：该仓库包含多个离线 AI 模型的纯 C++ ONNX 实现，如 Stable Diffusion (1.5 和 XL)、ControlNet、Midas、HED 和 OpenPose。</li><li><a href="https://github.com/ssitu/ComfyUI_UltimateSDUpscale">GitHub - ssitu/ComfyUI_UltimateSDUpscale: ComfyUI nodes for the Ultimate Stable Diffusion Upscale script by Coyote-A.</a>：由 Coyote-A 开发的 Ultimate Stable Diffusion Upscale 脚本的 ComfyUI 节点。</li><li><a href="https://arxiv.org/html/2312.03079v1">LooseControl: Lifting ControlNet for Generalized Depth Conditioning</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://civitai.com/images/19990254">gaia123 发布的视频</a>：未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI/blob/master/requirements.txt">ComfyUI/requirements.txt at master · comfyanonymous/ComfyUI</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图/节点界面。</li><li><a href="https://github.com/CaptainASIC/AI-Garage">GitHub - CaptainASIC/AI-Garage: A Set of AI tools consolidated into one launcher.</a>：集成到同一个启动器中的一组 AI 工具。</li><li><a href="https://github.com/Gourieff/comfyui-reactor-node">GitHub - Gourieff/comfyui-reactor-node: Fast and Simple Face Swap Extension Node for ComfyUI</a>：适用于 ComfyUI 的快速简单的换脸扩展节点。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1262494017929871473)** (6 messages): 

> - `Python 脚本中的 CUDA Kernel 调用`
> - `CUDA 与 PyTorch 矩阵乘法实现的性能比较`
> - `Torch Profiler 的使用` 


- **调查 CUDA Kernel 调用**：一位成员询问如何确定在 PyTorch 调用 Kernel（而非用户直接调用）的 **Python 脚本**中，究竟调用了哪些 CUDA Kernel。
   - 另一位成员建议使用 **PyTorch profiler** 来查看部署了哪些 ATen 函数。
- **CUDA vs PyTorch 矩阵乘法性能**：一位成员引用了[一节课](https://youtu.be/4sgKnKbR-WE?t=4045)，其中用于矩阵乘法的 CUDA Kernel 耗时 **6ms**，而 PyTorch 实现仅耗时 **2ms**。
   - 该成员质疑 PyTorch 是否为 **CNNs** 中的卷积操作提供了最优 Kernel，得到的回复是 PyTorch 通常为常见操作提供优秀的 Kernel，但在某些特定情况下，自定义 Kernel 会更有优势。



**提到的链接**：<a href="https://youtu.be/4sgKnKbR-WE?t=4045),">Lecture 3: Getting Started With CUDA for Python Programmers</a>：Jeremy 的 YouTube 录像 https://www.youtube.com/watch?v=nOxKexn3iBo 补充内容：https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262824285983084736)** (25 messages🔥): 

> - `GPU 性能问题`
> - `PyTorch Profiler 导出时间`
> - `自定义 Kernel 与 Thunder 编译器` 


- **笔记本 GPU 导致性能问题**：一位成员提到其笔记本电脑上的 GPU 性能缓慢，并寻求潜在原因的建议。
   - 另一位成员建议，任何合理规模的训练运行都不应该在笔记本电脑上进行。
- **PyTorch profiler 导出时间问题**：一位成员质疑使用 PyTorch profiler 导出 trace 需要 30 分钟是否正常。
   - 讨论指出，捕获大量信息或使用 `profile_memory` 选项可能会导致导出时间变长。
- **使用 nvfuser 的自定义 Kernel**：一位成员强调了使用 nvfuser 的自定义融合（fusion）Kernel，并发现该项目非常有用。
   - 虽然注意到了导出时间过长的问题，但未提出具体的解决方案或优化建议。


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1262562492824027157)** (1 messages): 

> - `SCALE GPGPU 编程工具包`
> - `为 AMD GPUs 编译 CUDA`
> - `SCALE 支持更多 GPU 厂商`
> - `SCALE 教程与示例` 


- **Spectral Compute 开发的 SCALE**：[SCALE](https://docs.scale-lang.com/) 是一个 GPGPU 编程工具包，允许 CUDA 应用程序在不修改原始 CUDA 程序或其构建系统的情况下，原生编译到 AMD GPUs 上。
- **SCALE 支持与资源**：对更多 GPU 厂商和 CUDA APIs 的支持**正在开发中**，目前已提供[教程](manual/how-to-use/)和[示例](examples/)供入门使用。



**提到的链接**：<a href="https://docs.scale-lang.com/">SCALE 文档</a>：未找到描述

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1262526637904498800)** (9 messages🔥): 

> - `Suno 招聘 ML 工程师`
> - `Suno 寻找 torch.compile 和 Triton 专家`
> - `不强制要求但鼓励掌握 Cutlass`
> - `Suno 招聘 ML 实习生` 


- **Suno 为实时音频模型招聘 ML 工程师**：Suno 正在招聘 ML 工程师，负责为向数百万用户流式传输音频的大型模型进行**训练和推理**工作，要求具备 **torch.compile**、**Triton**、**快速扩散（fast diffusion）/FM 采样**、**vLLM** 或**大规模分布式训练**方面的技能。[职位发布](https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20)
- **Suno 寻找 torch.compile 和 Triton 爱好者**：Suno 在其项目中专门寻求 **torch.compile**、**Triton** 和其他高速 ML 方法的专业知识。
   - 他们目前不专门寻找 **Cutlass** 专家，但鼓励具有类似技能的人员申请。
- **Suno 提供 ML 实习职位**：Suno 愿意招聘实习生，从事与其机器学习工程全职职位相同类型的工作。



**提到的链接**：<a href="https://jobs.ashbyhq.com/suno/7522d696-7ce8-4ece-a983-4be03dffde20">机器学习基础设施工程师</a>：我们正在寻找机器学习团队的早期成员。你将与创始团队紧密合作，并对我们如何构建和部署统计模型的各种技术决策拥有所有权...

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1262594919206227981)** (3 messages): 

> - `Lightning AI's Studios`
> - `Huggingface Spaces Dev Mode`
> - `CUDA development` 


- **Lightning AI's Studios 令用户兴奋**：一位成员推荐了 [Lightning AI's Studios](https://lightning.ai/docs/overview/studios#studios-)，认为它是一个极具吸引力的解决方案，并提到了其按需付费模式和免费层级。
   - 另一位成员给出了积极回应，表示 *“这看起来很棒。谢谢”。*
- **Huggingface Spaces Dev Mode 是否支持 CUDA？**：一位成员询问是否可以使用 [Huggingface Spaces Dev Mode](https://huggingface.co/spaces/dev-mode-explorers/README) 进行 CUDA 开发。
   - 目前还没有回复，该问题仍有待进一步讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lightning.ai/docs/overview/studios#studios-are-friendly-and-powerful">Studios ⚡️ Lightning AI</a>：Lightning AI Studio 是一个 AI 开发平台。Studios&amp;nbsp;&lt;b&gt;在浏览器中运行&lt;/b&gt;或&lt;b&gt;在你自己的云基础设施上运行&lt;/b&gt;。使用它在云端编写代码、构建、训练...</li><li><a href="https://lightning.ai/docs/overview/studios#studios-">Studios ⚡️ Lightning AI</a>：Lightning AI Studio 是一个 AI 开发平台。Studios&amp;nbsp;&lt;b&gt;在浏览器中运行&lt;/b&gt;或&lt;b&gt;在你自己的云基础设施上运行&lt;/b&gt;。使用它在云端编写代码、构建、训练...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

andreaskoepf: 有人试过 Mosaic GPU 了吗？https://x.com/apaszke/status/1812897008031617493
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1262842494601658441)** (3 messages): 

> - `torch.compile 中的 unwrap_tensor_subclass`
> - `模型编译中的 FakeTensors` 


- **编译后出现意外的 FakeTensor 权重**：一位成员对 `unwrap_tensor_subclass` 似乎在模型中将所有子类 Tensor 替换为 **FakeTensors** 感到困惑，这导致了问题，因为这些 Tensor 没有被作为打包 Tensor 处理。
   - 该成员评论道：*“我以为 FakeTensors 只用于编译阶段，而不是在运行编译后的函数时使用。”*
- **torch.compile 中的参数化（Parametrization）变通方案**：另一位成员澄清说，`unwrap_tensor_subclass` 使用**参数化（parametrization）**将 Tensor 子类转换为普通 Tensor，以绕过目前 **torch.compile 栈**（特别是 aot_export）的局限性。

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1262485006207418409)** (139 条消息🔥🔥): 

> - `CUDA 参数重命名`
> - `Attention 机制`
> - `StableAdamW`
> - `llm.c 中的 AMD GPU 支持`
> - `Andrej Karpathy 创立的 AI+教育公司` 


- **CUDA 函数参数重命名讨论**：讨论集中在重命名 CUDA 函数参数以避免与全局变量冲突，以及完全避免此类参数的可行性。
   - *akakak1337* 指出需要将 `multi_gpu_config` 重命名为 `config`，同时考虑是否仅使用全局变量就足够了。
- **GPT3 的交替 Dense 和 Banded Attention**：**GPT3 需要在 Dense 和 Banded 变体之间交替使用 Attention 机制**以增强性能，类似于 **Mistral** 和 **Gemma 2**。
   - cuDNN 中的一个参数允许启用 Attention 的窗口大小（window sizes），这对于实现可能至关重要。
- **StableAdamW 集成改进了 GAN 训练**：据 **_clashluke** 称，将 **StableAdamW** 引入 [ScheduleFreeAdamW](https://arxiv.org/abs/2304.13013) 显著增强了 GAN 训练的稳定性。
   - *akakak1337* 提供了合并 StableAdamW 更改的更新，以获得改进的梯度裁剪（gradient clipping）结果。
- **在 AMD GPU 上运行 llm.c 的挑战**：由于缺少 **cublaslt** 和 **bfloat16 支持**等关键元素，在 AMD GPU 上运行 **llm.c** 的努力面临障碍。
   - **SCALE** 为 AMD 上的 CUDA 应用提供的方法缺少关键环节，使其效果不如 **hipify**。
- **Andrej Karpathy 推出 AI+教育公司 Eureka Labs**：**Andrej Karpathy** 介绍了 [Eureka Labs](https://x.com/karpathy/status/1813263734707790301)，旨在创建 AI 辅助的教育工具，首门课程名为 LLM101n。
   - 该项目强调利用 AI 来扩大教育的覆盖范围并提高质量，结合了 Karpathy 深厚的 AI 专业知识和对教学的热情。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1813263734707790301">来自 Andrej Karpathy (@karpathy) 的推文</a>: ⚡️ 很高兴分享我正在创办一家名为 Eureka Labs 的 AI+教育公司。公告如下：--- 我们是 Eureka Labs，我们正在建立一种新型的 AI 原生学校。我们如何应用...</li><li><a href="https://x.com/_clashluke/status/1812938241831579990">来自 Lucas Nestler (@_clashluke) 的推文</a>: 继 https://x.com/fr0sty__/status/1808664083014599103 之后，我已将 StableAdamW (https://arxiv.org/abs/2304.13013) 加入 ScheduleFreeAdamW。在简单问题中收敛情况相同，但...</li><li><a href="https://github.com/karpathy/llm.c/pull/689">karpathy 的 Refactor/code to zerocuh · Pull Request #689 · karpathy/llm.c</a>: 无描述</li><li><a href="https://en.cppreference.com/w/c/types/boolean">布尔类型支持库 - cppreference.com</a>: 无描述</li><li><a href="https://gpuopen.com/learn/wmma_on_rdna3/">如何使用 WMMA 在 RDNA 3 上加速 AI 应用</a>: 本博客是关于如何使用 RDNA 3 GPU 架构的 WMMA 功能的快速入门指南，包含一个 Hello World 示例。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1262753354816294964)** (2 条消息): 

> - `稀疏性 (Sparsity)`
> - `量化模型 (Quantized models)` 


- **稀疏性和量化模型创新效率**：[一篇论文](https://arxiv.org/pdf/2407.10969) 讨论了**稀疏性**与**量化模型**的成功结合。
   - *mobicham* 评论说，这种方法可以为非稀疏权重带来更快的推理速度和更高的精度，同时保持相同的平均比特率。
- **稀疏性和量化提升性能**：一位成员指出，结合**稀疏性**和**量化模型**具有双重优势：增强速度和准确性。
   - 如讨论中所分享的，该方法允许在不损害模型质量的情况下实现高效的资源利用。


  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 条消息): 

iron_bound: 很棒的演示
https://wgpu.rs/

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1262615220753530971)** (1 条消息): 

> - `Qwen 2 7B Instruct` 


- **免费访问 Qwen 2 7B Instruct 模型**：OpenRouter 宣布免费提供 [Qwen 2 7B Instruct](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) 模型。
   - 你现在可以在 [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct):free 上访问它。
- **Qwen 2 7B Instruct 模型发布**：[Qwen 2 7B Instruct](https://openrouter.ai/models/qwen/qwen-2-7b-instruct) 模型现在可以免费使用。
   - 在 [OpenRouter](https://openrouter.ai/models/qwen/qwen-2-7b-instruct):free 上了解更多关于该模型的信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct>)">Qwen 2 7B Instruct (由 qwen 提供)</a>: Qwen2 7B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、Attention QKV 偏置以及分组（grou...）</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct:free">Qwen 2 7B Instruct (免费版) (由 qwen 提供)</a>: Qwen2 7B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、Attention QKV 偏置以及分组（grou...）
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1262484303615361024)** (130 messages🔥🔥): 

> - `Google Gemini Models`
> - `GPT-4o Free Tier`
> - `Gemini 1.5 Pro Performance`
> - `OpenRouter Issues`
> - `Llama 3 Extended Context Models` 


- **谷歌 Gemini 模型受到批评**：成员们讨论了为什么 Google 不为 Gemini 的免费层级提供更好的模型，称 **Gemini 1.0** 与 **GPT-4o** 相比“相当糟糕”。
   - *“Google 确实有两把刷子，”* 一位 **Gemini 1.5 Pro** 的用户指出，并强调了它的创作潜力，但其编码性能存在问题。
- **GPT-4o 的免费层级策略令人印象深刻**：一位成员评论道，**OpenAI** 通过向免费层级用户提供 GPT-4o 取得了成功，为竞争对手设定了很高的门槛。
- **OpenRouter 遭遇轻微停机**：用户报告了 **OpenRouter** 的零星停机，并在访问网站和 API 时遇到困难，引发了多次询问。
   - 官方回应将这些问题归因于间歇性路由问题和可能的 Cloudflare 错误，服务在不久后基本恢复。
- **寻找 Llama 3 扩展上下文模型**：成员们对 **Llama 3-70B Instruct** 的 **8k 上下文窗口**限制以及寻找更好替代方案的挑战表示沮丧。
   - 虽然有人建议使用 **Euryale** 和 **Magnum-72B** 等模型，但缺乏一致的指令遵循（instruction-following）能力和高昂的成本是主要担忧。
- **OpenRouter 模型访问与功能**：关于 OpenRouter 的服务存在一些混淆，澄清了它并非所有模型都免费，但确实提供无需订阅的免费选项。
   - **OpenRouter** 提供了对各种 API 和免费模型的访问，关于托管本地模型的细节仍处于特定的商业合同阶段。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jsoneditoronline.org/#left=local.tihuto,">JSON Editor Online: edit JSON, format JSON, query JSON</a>: JSON Editor Online 是网络上原始且被模仿最多的 JSON 编辑器。使用它来查看、编辑、格式化、修复、比较、查询、转换、验证和共享您的 JSON 数据。</li><li><a href="https://openrouter.ai/playground">Playground | OpenRouter</a>: 试验不同的模型和提示词</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 by sao10k</a>: Euryale 70B v2.1 是来自 [Sao10k](https://ko-fi.com/sao10k) 的专注于创意角色扮演的模型。- 更好的提示词遵循。- 更好的解剖学/空间意识。- 能更好地适应独特和复杂的场景...</li><li><a href="https://openrouter.ai/models/alpindale/magnum-72b">Magnum 72B by alpindale</a>: 由 [Goliath](https://openrouter.ai/models/alpindale/goliath-120b) 的制作者开发，Magnum 72B 是新模型系列中的首款，旨在达到 Claude 3 模型的散文质量，特别是...</li><li><a href="https://www.together.ai/pricing">Together Pricing | The Most Powerful Tools at the Best Value</a>: 获取推理、微调、训练和 Together GPU 集群的详细定价。</li><li><a href="https://tenor.com/view/wizard101-0bobux-wallet-empty-wallet-empty-gif-22389933">Wizard101 0bobux GIF - Wizard101 0Bobux Wallet - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://openrouter.ai/models?o=newest&max_price=0)">Models - Newest | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended">Llama 3 Lumimaid 8B (extended) by neversleep</a>: NeverSleep 团队回归，带来了基于其精选角色扮演数据训练的 Llama 3 8B 微调模型。Lumimaid 在 eRP 和 RP 之间取得了平衡，旨在保持严肃，但在必要时不受审查...</li><li><a href="https://openrouter.ai/models/sao10k/l3-stheno-8b">Llama 3 Stheno 8B v3.3 32K by sao10k</a>: Stheno 8B 32K 是来自 [Sao10k](https://ko-fi.com/sao10k) 的创意写作/角色扮演模型。它在 8K 上下文下训练，随后扩展到 32K 上下文。与旧版 Stheno 相比，该模型...</li><li><a href="https://openrouter.ai/models/ne">Models: &#x27;ne&#x27; | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.onlineornot.com/">OpenRouter Status</a>: OpenRouter 事件历史
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1262530015178915871)** (110 条消息🔥🔥): 

> - `GPT 粘贴数值的问题`
> - `不同 Collection 的模型设置`
> - `Perplexity 办公室`
> - `Gemini AI 详情`
> - `Pro 订阅支持` 


- **GPT 粘贴数值的问题**: 成员们讨论了将数值作为附件粘贴时出现的问题，以及从[多个模型](https://perplexity.ai)中获得无关且通用的回复。
- **不同 Collection 的模型设置**: 有人提出了关于为不同的 Collection 或线程[分配不同模型](https://perplexity.ai)的问题，例如为 CollectionA 分配 chatgpt4o，为 CollectionB 分配 Opus。
- **Perplexity 办公室公布**: 一位成员分享了一条宣布 [Perplexity 新办公室](https://x.com/AravSrinivas/status/1812890154367078590)的推文，令社区感到兴奋。
- **Gemini AI 性能详情**: 成员们分享了 DeepMind 的 [Gemini AI 规格链接](https://deepmind.google/technologies/gemini/)，重点介绍了其在 2024 年不同数据集上的表现。
- **Pro 订阅支持问题**: 多位用户报告了在 PC 和 iOS 上[激活 Pro 订阅](https://perplexity.ai/settings/account)的问题，尽管使用了相同的电子邮件并收到了确认邮件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/cat-questioning-life-questioning-life-what-is-life-gif-4882578">Cat Questioning GIF - Cat Questioning Life - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/AravSrinivas/status/1812890154367078590">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 新的 Perplexity 办公室！</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://deepmind.google/technologies/gemini/">Gemini</a>: Gemini 系列模型是我们迄今为止构建的最通用且能力最强的 AI 模型。它们从底层开始构建，旨在实现多模态——在文本、代码、图像、音频之间无缝推理……</li><li><a href="https://python-fiddle.com/saved/NZfCYDD2l6h51DL8vZ0a">Python-Fiddle: 在线 Python 编译器、IDE 和解释器</a>: 在浏览器中运行 Python 代码。与他人分享代码片段。</li><li><a href="https://python-fiddle.com/saved/CG2EpDwjRDz3uSq2sAEc">Python-Fiddle: 在线 Python 编译器、IDE 和解释器</a>: 在浏览器中运行 Python 代码。与他人分享代码片段。</li><li><a href="https://www.perplexity.ai/search/f-t-3-e-t-sin">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可为任何问题提供准确、可靠且实时的答案。</li><li><a href="https://www.perplexity.ai/search/f-t-3-e-t-sin-5pi-t-plot-this-383zGYadRV.qbZnMUZFXlg">f(t) = 3 * e^(-t) * sin(-5pi*t) 

使用 plotly 绘制此图并添加额外的上...</a>: 根据指令和搜索结果，我将详细解释如何使用 Plotly 绘制给定的函数，包括上限和下限……</li><li><a href="https://www.wolframalpha.com/input?i=f%28t%29+%3D+3+*+e%5E%28-t%29+*+sin%282*2pi*t%29+from+t+%3D+0+to+5">f(t) = 3 * e^(-t) * sin(2*2pi*t) from t = 0 to 5 - Wolfram|Alpha</a>: Wolfram|Alpha 为最广泛的人群提供专家级的知识和能力——涵盖所有职业和教育水平。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1262623402230026372)** (3 条消息): 

> - `Alphabet $23B Deal`
> - `7-Eleven's Upgrade`
> - `New Zealand's Rare Whale Discovery`
> - `Accessible Lunar Cave`
> - `Perplexity AI Pro Features` 


- **Alphabet 签署 230 亿美元交易**：**Alphabet** 达成了一项 230 亿美元的交易，在市场洗牌中占据了战略地位。欲了解更多详情，请[在 YouTube 上观看摘要](https://www.youtube.com/embed/lKn8rh0pOiM)。
- **7-Eleven 推出重大升级**：**7-Eleven** 宣布了一项升级，可能会提升其门店的消费者体验。[在此查看更多见解](https://www.youtube.com/embed/lKn8rh0pOiM)。
- **科学家发现可进入的月球洞穴**：利用先进的雷达成像技术，在 Mare Tranquillitatis 发现了一个**可进入的月球洞穴**。据估计，该洞穴至少有 130 英尺宽，数十码长，位于月球表面下方约 150 米处。
   - 这一发现表明，该洞穴可能提供免受严酷月球环境影响的保护和稳定的温度，使其对未来的月球探测和居住具有重要价值。[在 Perplexity AI 上查看更多详情](https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0)。
- **Perplexity AI Pro 提供新功能**：Perplexity AI 推出了新的 Pro 功能，包括图片上传和更智能的 AI 能力。[探索这些新增功能以增强搜索体验](https://www.perplexity.ai/search/In-Batch-how-GStLqBGjTpqscDMMVEcyOA)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/lKn8rh0pOiM">YouTube</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/moon-s-hidden-refuge-scientist-yz19IMD.TE6E4fZj9A9W.Q#0">Moon's Hidden Refuge: Scientists Uncover Potential Lunar Base in Underground...</a>：月球的隐藏避难所：科学家在地下洞穴中发现潜在的月球基地。这是一项可能重塑太空未来的突破性发现...</li><li><a href="https://www.perplexity.ai/search/In-Batch-how-GStLqBGjTpqscDMMVEcyOA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1262803292904489080)** (5 条消息): 

> - `Removing sources in pplx-api`
> - `524 errors with sonar models`
> - `Stream mode functionality` 


- **咨询 pplx-api 中的移除来源功能**：一位用户询问 **pplx-api** 是否支持类似于 UI 中提供的 `deleted_urls` 参数的移除特定来源选项。
   - 用户的疑问源于在文档中未找到此类选项。
- **报告 sonar 模型出现 524 错误**：一位成员报告在使用 `llama-3-sonar-small-32k-online` 和 `llama-3-sonar-large-32k-online` 时遇到 524 错误。据另一位成员称，*服务器超时了*。
   - 建议启用 **stream** 模式可能有助于保持连接开启。
- **启用 stream 模式以维持连接**：为了处理连接问题，用户可以通过传递 `"stream": True` 来启用 **stream** 模式。
   - 一位成员建议：*启用 stream 模式将保持连接开启。*


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1262781533891330089)** (17 messages🔥): 

> - `Codestral Mamba`
> - `Mathstral`
> - `SmolLM`
> - `Eureka Labs`
> - `Hydra Model Extension` 


- **Codestral Mamba：无限序列的新希望**：[Codestral Mamba](https://mistral.ai/news/codestral-mamba/) 引入了一种在 Albert Gu 和 Tri Dao 帮助下设计的新架构，具有线性时间推理的优势，并在理论上具备建模无限长度序列的能力。
   - 它在代码生产力用例中特别有效，性能与最先进的基于 Transformer 的模型持平。
- **Mathstral 专注于高级 STEM 推理**：[Mathstral](https://mistral.ai/news/mathstral/) 专注于 STEM 学科，在其体量下实现了极高的推理能力，在 MATH 基准测试中得分为 **56.6%**，在 MMLU 上为 **63.47%**。
   - Mathstral 是与 [Project Numina](https://projectnumina.ai/) 合作开发的，展示了针对特定应用场景的卓越性能与速度权衡。
- **SmolLM：小体积，高性能**：[SmolLM](https://x.com/loubnabenallal1/status/1813252390692303069?s=46) 推出了包含 135M、360M 和 1.7B 参数的新 SOTA 模型，这些模型在高质量的 Web、代码和合成数据上进行了训练。
   - 这些模型的表现优于 MobileLLM、Phi1.5 和 Qwen2，凸显了 LLM 端侧部署（on-device deployment）日益增长的重要性。
- **Eureka Labs 重新定义 AI 教育**：[Eureka Labs](https://eurekalabs.ai/) 旨在创建一所 AI 原生学校，从 AI 助教开始提升学习体验。
   - 他们的首个产品 LLM101n 将引导学生训练自己的 AI，从而扩展教育的广度和深度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eurekalabs.ai/">Eureka Labs</a>：未找到描述</li><li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>：为了向阿基米德致敬（今年是我们庆祝他逝世 2311 周年），我们自豪地发布了首个 Mathstral 模型，这是一个专门为数学推理和科学讨论设计的 7B 模型...</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：为了向克利奥帕特拉致敬（她光辉的命运终结于悲惨的毒蛇事件），我们自豪地发布了 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用</li><li><a href="https://huggingface.co/spaces/HuggingFaceTB/SmolLM-360M-Instruct-WebGPU">SmolLM 360M Instruct WebGPU - HuggingFaceTB 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/loubnabenallal1/status/1813252390692303069?s=46">Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：LLM 的端侧部署比以往任何时候都更加重要。今天我们发布了 SmolLM，一系列新的 SOTA 模型，包括 135M、360M 和 1.7B：- 表现优于 MobileLLM、Phi1.5 和 Qwen2 小型模型 - 训练于...</li><li><a href="https://x.com/_albertgu/status/1813252409071968297?s=46">Albert Gu (@_albertgu) 的推文</a>：发布 Hydra，我们将 Mamba（以及通用状态空间模型）扩展为双向的“官方”扩展！Hydra 的动力源自第一性原理，通过框架增加表达能力...</li><li><a href="https://x.com/karpathy/status/1813263734707790301?s=46">Andrej Karpathy (@karpathy) 的推文</a>：⚡️ 很高兴分享我正在创办一家名为 Eureka Labs 的 AI+教育公司。公告如下：--- 我们是 Eureka Labs，我们正在建立一种 AI 原生的新型学校。我们该如何应用...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1262488819434000427)** (3 messages): 

> - `规避 Torrent 法律`
> - `AI 中的评估门控`
> - `私有测试集与基准测试过滤` 


- **请求书籍实体副本以规避 Torrent 法律**：一位成员询问是否可以通过**硬盘**邮寄书籍副本，以**规避 Torrent 法律**。
   - *该话题没有进一步的讨论或提供的链接。*
- **处理 AI 中的评估门控**：一位成员询问了 **GPQA** 和 **GAIA** 使用的**评估“门控”（gating）**，以及防止测试集意外混入预训练语料库的保护措施和可能的过滤方法。
   - 另一位成员解释说，**私有测试套件**涉及大量的时间和成本，这使得许多学术机构无法负担，并指出**基底匹配（substrate matching）**是一种常见的过滤方法。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1262855620760043520)** (3 条消息): 

> - `Lobbying for Legislative Bills` (为立法法案游说)
> - `Conflict of Interest` (利益冲突)
> - `Profit from Compliance Checks` (从合规检查中获利)
> - `Ethics of Political Donations` (政治捐赠的伦理)


- **游说引发利益冲突担忧**：一位成员转发了一条 [推文](https://fxtwitter.com/mpopv/status/1813273553477009546?s=46)，讨论了为一项通过强制合规检查使自己业务受益的法案进行游说的不道德性质。
   - *感觉如果你在大力游说，并为游说某项立法法案而征集捐款，你可能应该披露你秘密拥有一家公司，该公司正准备通过销售该法案强制要求的合规检查，从该法案的通过中获利。*
- **游说与既得利益息息相关**：另一位成员质疑是否有人在没有既得利益的情况下为立法法案游说，暗示这是一种普遍做法。
- **游说动机的差异**：讨论区分了预先资助一个组织以从法案中获利，与保护现有组织利益之间的区别。
   - 一位成员承认了这种区别，但同意主要动机通常是自利。



**提到的链接**：<a href="https://fxtwitter.com/mpopv/status/1813273553477009546?s=46">Matt Popovich (@mpopv) 的推文</a>：感觉如果你在大力游说，并为游说某项立法法案而征集捐款，你可能应该披露你秘密拥有一家公司，该公司正准备通过销售该法案强制要求的合规检查，从该法案的通过中获利...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1262508170140450857)** (31 messages🔥): 

> - `LLM 评估现状`
> - `开源 GPT4o 级别模型的假设用途`
> - `AI 训练数据源`
> - `模型训练成本`
> - `新的 SciCode 基准测试` 


- **LLM 评估引发激烈辩论**：围绕 **LLM evaluations** 的现状展开了热烈讨论，一些用户对当前的评估方法表示困惑和沮丧，如这篇 [tweet](https://x.com/sureailabs/status/1812949017212690472) 所示。
   - 一位用户幽默地表示欣赏这种持续的辩论并鼓励更多对话：*我爱你们；请继续唠。*
- **假设性的 GPT4o 级别模型引发好奇**：成员们讨论了开源 **GPT4o-class model** 在新研究查询方面的潜力，正如这篇 [tweet](https://x.com/swyx/status/1812988248660320679?s=46) 所提出的。
   - 一些人推测它可能会显著降低成本，并对企业级 API 构成重大威胁；*有利于未来 12-18 个月的开源模型训练（synth data）等。*
- **科技巨头秘密大量摄取 YouTube 数据**：一篇 [Wired 文章](https://www.wired.com/story/youtube-training-data-apple-nvidia-anthropic/) 揭露，尽管有平台限制，各大科技公司仍在使用 YouTube 数据进行 AI 训练。
   - 成员们认为这种做法已被广泛认知，但对许多公司来说尚未得到证实，这引发了关于 AI 新闻报道标准的辩论。
- **训练 Llama3 与 GPT4o 的成本对比**：讨论估计训练 **Llama3** 的成本约为 **每百万 token $4-5**，在输入成本上可能与 **GPT4o** 持平，并将输出成本降低约三分之一。
   - 一位成员提到 **Groq** 是一个更便宜的选择，推测价格为 **每百万 token $3.36/4.50**；参见 [此推文](https://x.com/thexeophon/status/1813108909416325261?s=46)。
- **揭晓 SciCode 基准测试：LM 的终极 STEM 挑战**：**SciCode** 推出了一项新的基准测试，挑战 LM 为诺贝尔奖研究中的科学问题编写代码解决方案。如 [此推文](https://x.com/minyangtian1/status/1813182904593199553?s=46) 所述，**GPT-4 和 Sonnet 3.5 的准确率不足 5%**。
   - 该基准测试被视为预训练的重要评估手段，其先进且严谨的方法引起了用户的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.wired.com/story/youtube-training-data-apple-nvidia-anthropic/">Apple, Nvidia, Anthropic 使用数千个抓取的 YouTube 视频训练 AI</a>：“这是偷窃。” WIRED 的一项调查发现，来自 48,000 多个频道的 173,536 个 YouTube 视频字幕被 Anthropic、Nvidia、Apple 和 Salesforce 用于训练 AI。</li><li><a href="https://x.com/sureailabs/status/1812949017212690472">来自 surea.i • (in SF!?!!!!!!) (@sureailabs) 的推文</a>：我最喜欢的（在旧金山！？!!!!!!）那种我还不能完全理解的对话，是发生在所有对 LLM 评估现状感到愤怒的人之间的。我爱你们，请继续唠。</li><li><a href="https://x.com/thexeophon/status/1813108909416325261?s=46">来自 Xeophon (@TheXeophon) 的推文</a>：@_xjdr 更不用说 Groq 了，它的价格便宜得简直不公平。如果你假设定价像从 8B->70B 那样按比例缩放，你会得到 3.36/4.50。</li><li><a href="https://x.com/minyangtian1/status/1813182904593199553?s=46">来自 Minyang Tian (@MinyangTian1) 的推文</a>：SciCode 是我们新的基准测试，挑战 LM 为高级论文中的科学问题编写代码解决方案。这些挑战由博士精心设计；我们约 10% 的基准测试基于诺贝尔奖获得者的研究...</li><li><a href="https://x.com/swyx/status/1812988248660320679?s=46">来自 swyx 🤞 🔜 SFO (@swyx) 的推文</a>：完全假设一下... 你会用一个今天还无法获得的开源 GPT4o 级别模型做什么？在“新常态” AI 的范围内，你可以提出哪些能带来超额收益（alpha）的问题...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1262526393774772405)** (6 messages): 

> - `MSFT 论文`
> - `WizardLM 风格` 


- **MSFT Wizard 团队发布论文**：一位成员提到 **MSFT** 的 Wizard 团队本月发布了一系列论文，并附上了 [Qingfeng Sun 的推文链接](https://x.com/victorsungo/status/1812854829397746075)。
   - 如果有人感兴趣，他们计划阅读并讨论，但对 *虚假的启发式废话（hokey heuristic bs）* 表示蔑视。
- **对 WizardLM 论文的评价褒贬不一**：另一位成员评论说，他们觉得 **WizardLM vibes** 很有趣。
   - *编辑：在我看来论文有点烂。*

---

### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1262849929026469928)** (2 条消息): 

> - `Degenerate Case in Policy Reinforcement`（策略强化中的退化情况）
> - `DPO-like algorithms`（类 DPO 算法）


- **POL 中有用的退化情况**：一位成员推测，退化情况对于策略强化中获胜和失败之间的公共前缀可能是非常有用或必要的。
   - 另一位成员表示同意，并建议深入探讨让人们关注此类深层技术细节的意义。
- **DPO 算法激发算法讨论**：一位成员表达了对类 DPO 算法的兴奋，强调了它们在激发对算法讨论的重新关注方面所起的作用。
   - 他们指出了使用 `losses = -F.logsigmoid(policy_rejected_logp)` 可能存在的过拟合风险，但仍渴望进一步探索。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262484066234794014)** (21 条消息🔥): 

> - `Qwen Team`（Qwen 团队）
> - `RewardBench Performance`（RewardBench 性能）
> - `Foundational Large Autorater Models`（基础大型自动评分模型）
> - `Post-training for AI models`（AI 模型的后训练）
> - `DeepMind's New Paper on FLAMe`（DeepMind 关于 FLAMe 的新论文）


- **Qwen 团队的后训练见解**：[Qwen 2 后训练笔记](https://x.com/natolambert/status/1813263814009495799)涵盖了合成数据（synthetic data）的趋势，其多阶段对齐训练展示了开源模型中未曾见过的流程。
   - 见解包括通过标签对**样本进行分类和过滤**，以及**代码的执行反馈**，展示了为提升模型性能所做的努力。
- **DeepMind 的 FLAMe 模型超越 GPT-4**：DeepMind 的 [FLAMe](http://arxiv.org/abs/2407.10817) 通过在人类评估数据上进行训练，在 RewardBench 上的表现超过了 GPT-4 和 4o，但该模型目前仍未开源。
   - 关于最后一刻的分数调整有一些争议，凸显了这些评估的竞争性质。
- **推荐 Bypass Paywalls 扩展**：用户讨论了使用 **bypass-paywalls-clean** 扩展来绕过付费墙，以便可靠地访问文章。
   - 这是针对访问特定文章和资源时遇到的问题而分享的。
- **arXiv 论文强调计算趋势**：An Yang 等人撰写的一篇详细的 [arXiv 论文](https://arxiv.org/abs/2407.10671)深入探讨了在大规模 Token 数据集上训练模型，但最终确定 7T Token 更为有效。
   - 论文详细介绍了各个阶段和评估，与顶尖 AI 实验室的流程保持一致。
- **关于数据采样策略的不同观点**：成员们辩论了数据采样策略，特别是应该采样多个响应还是专注于最好/最差的配对。
   - 讨论强调了增强训练多样性和偏好排序（preference ranking）的不同方法和不断演变的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/tuvllms/status/1813249272474968315">Tu Vu (@tuvllms) 的推文</a>: 🚨 Google DeepMind 新论文 🚨 我们在广泛的人类评估基础上训练了基础大型自动评分模型 (FLAMe)，在仅基于 p 训练的生成模型中取得了最佳的 RewardBench 性能...</li><li><a href="https://arxiv.org/abs/2407.10671">Qwen2 技术报告</a>: 本报告介绍了 Qwen2 系列，这是我们大型语言模型和大型多模态模型的最新成员。我们发布了一套全面的基础和指令微调语言模型...</li><li><a href="https://x.com/natolambert/status/1813263814009495799">Nathan Lambert (@natolambert) 的推文</a>: Qwen 2 后训练 / RLHF 笔记。细节不多，但有很多合成数据等方面的共同趋势。它展示了打造一个好模型需要多少细微的工作。这类数据处理过程 ...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1262485588951699548)** (24 条消息🔥): 

> - `GPT-4 在 GSM8k 上的训练`
> - `指令微调数据集`
> - `训练期间的 GPU 故障`
> - `物体计数用例`
> - `Pile 2 数据集大小` 


- **GPT-4 在 GSM8k 训练集上进行过训练**：一位成员指出，原始的 **GPT-4** 在大部分 **GSM8k 训练集**上进行了训练，正如 [GPT-4 技术报告](https://link.to.report)中所提到的。
   - 另一位成员对能记住这些细节表示惊讶，但有人指出，像这样重要的事实经常在推特上被提及，因此很容易记住。
- **对指令微调数据集的担忧**：有人对指令微调数据集中使用的语法表示担忧，以及 OpenAI 的聊天应用是否改为使用带项目符号的思维链（Chain of Thoughts）。
   - 有人怀疑 OpenAI 是否会在其指令微调数据集中包含某些标记（<<>>），暗示可能存在污染或偷懒行为。
- **模型训练期间的 GPU 故障率**：一位成员询问是否有论文披露了在大模型训练期间 **GPU 故障**率的示例。
   - [Reka 技术报告](https://publications.reka.ai/reka-core-tech-report.pdf)和 **OPT/BLOOM 日志本**被提及为此类数据的良好来源。
- **物体计数实现咨询**：一位用户寻求关于实现自定义物体**物体计数**的建议，这些物体可能差异很大，并询问最佳方法。
   - 他们描述了一种潜在的方法：拍摄一张图像，在一个物体上画一个边界框（bounding box），然后检测所有相似物体以获得最终计数。
- **Pile 2 数据集磁盘空间查询**：一位成员询问了 **Pile 2 数据集**所需的磁盘空间，并指出目前仅提到了其 Token 数量。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1262487336055472239)** (41 条消息🔥): 

> - `SSMs 的上下文学习`
> - `用于处理无限上下文的 EM-LLM`
> - `用于加速 LLM 推理的 FLUTE`
> - `用于高效稀疏 LLM 的 Q-Sparse`
> - `Transformer 层的观察性研究` 


- **Transformer 中的状态空间模型引发讨论**：[一种新颖的权重构建方法](https://arxiv.org/abs/2407.09375)使 State Space Models 能够在观察先前状态后预测任何动力系统的下一状态，而无需参数微调。
- **使用 EM-LLM 实现无限上下文**：EM-LLM 将人类情境记忆（episodic memory）的各个方面整合到 LLMs 中，以高效处理[几乎无限的上下文长度](https://arxiv.org/abs/2407.09450)。
- **FLUTE 提升 LLM 推理速度**：[FLUTE](https://github.com/HanGuo97/flute) 是一个用于 LUT 量化 LLMs 的灵活查找表引擎，通过最小化位操作并利用向量化来实现更快的推理。
- **Q-Sparse 提高 LLM 效率**：[Q-Sparse](https://arxiv.org/abs/2407.10969) 应用了 top-K 稀疏化和直通估计器（straight-through-estimator），使稀疏激活的 LLMs 能够以更高的效率达到与基准 LLMs 相当的结果。
- **实证研究揭示 Transformer 层的鲁棒性**：研究表明，预训练 Transformer 的中间层比底层或顶层更能处理逐层修改，在除推理密集型任务（如 [GSM8K](https://arxiv.org/abs/2407.09298)）之外的大多数任务中保持性能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2407.10969">Q-Sparse: All Large Language Models can be Fully Sparsely-Activated</a>: 我们介绍了 Q-Sparse，这是一种简单且有效的训练稀疏激活大语言模型 (LLMs) 的方法。Q-Sparse 实现了 LLMs 激活的全稀疏化，这可以带来显著的效...</li><li><a href="https://arxiv.org/abs/2407.09941">Hydra: Bidirectional State Space Models Through Generalized Matrix Mixers</a>: 许多序列模型都建立在模仿 Transformers 的框架之上，由交替的序列混合器 (sequence mixer) 和通道混合器 (channel mixer) 层组成。本文研究了一种统一的矩阵混合器视角...</li><li><a href="https://arxiv.org/abs/2407.09450">Human-like Episodic Memory for Infinite Context LLMs</a>: 大语言模型 (LLMs) 展示了卓越的能力，但在处理超长上下文方面仍面临挑战，限制了它们在长序列中保持连贯性和准确性的能力。我...</li><li><a href="https://arxiv.org/abs/2407.10827">LLM Circuit Analyses Are Consistent Across Training and Scale</a>: 目前部署的大多数大语言模型 (LLMs) 都会经历持续训练或额外的微调。相比之下，大多数关于 LLMs 内部机制的研究都集中在单一快照的模型上...</li><li><a href="https://arxiv.org/abs/2407.09298">Transformer Layers as Painters</a>: 尽管 Transformer 在大语言模型中得到了近乎普遍的应用，但其内部运作机制尚未被充分理解。我们的目标是更好地理解移除或重组信...</li><li><a href="https://arxiv.org/abs/2402.13388">Transformer tricks: Precomputing the first layer</a>: 这篇微论文描述了一种加速带有 RoPE 的 Transformer（如 LLaMA, Mistral, PaLM 和 Gemma）推理的技巧。对于这些模型，第一层 Transformer 的很大一部分可以被预...</li><li><a href="https://arxiv.org/abs/2403.15796">Understanding Emergent Abilities of Language Models from the Loss Perspective</a>: 最近的研究对“语言模型的涌现能力是大模型所特有的”这一观点提出了质疑。这种怀疑源于两个观察结果：1) 较小的模型也可以表现出...</li><li><a href="https://arxiv.org/abs/2310.18780">Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions</a>: 无注意力序列模型的最新进展依赖于卷积，以此作为 Transformer 核心注意力算子的替代方案。特别是，长卷积序列模型已经取得...</li><li><a href="https://www.youtube.com/watch?v=s8RqGlU5HEs">2 Years of My Research Explained in 13 Minutes</a>: 这是我在强化学习背景下对表示学习和模型学习的研究。历时两年，我终于可以谈谈...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>: 大型 Transformer 模型在许多任务上都取得了令人印象深刻的结果，但训练甚至微调的成本都很高，而且解码速度非常慢，以至于它们的使用和研究变得遥不可及。我们解决了这个问...</li><li><a href="https://arxiv.org/abs/2407.09375">HiPPO-Prophecy: State-Space Models can Provably Learn Dynamical Systems in Context</a>: 这项工作探索了状态空间模型 (SSMs) 的上下文学习能力，并据我们所知，首次对可能的底层机制提出了理论解释。我...</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: 病毒通过分子模拟逃避免疫系统，采用宿主的生物物理特征。我们调整了蛋白质语言模型 (PLMs) 来区分人类和病毒...</li><li><a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: 一个用于在 NVIDIA GPUs 上加速 Transformer 模型的库，包括在 Hopper 和 Ada GPUs 上使用 8 位浮点 (FP8) 精度，以提供更好的性能和更低的内存占用...</li><li><a href="https://www.strchr.com/standard_deviation_in_one_pass">Calculating standard deviation in one pass - strchr.com</a>: 未找到描述</li><li><a href="https://github.com/HanGuo97/flute">GitHub - HanGuo97/flute: 用于查找表量化 LLMs 的快速矩阵乘法</a>: 用于查找表量化 LLMs 的快速矩阵乘法 - HanGuo97/flute</li><li><a href="https://arxiv.org/abs/2407.10960">Fast Matrix Multiplications for Lookup Table-Quantized LLMs</a>: 大语言模型 (LLMs) 的部署通常受限于内存带宽，主要的瓶颈在于将模型参数从 GPU 的全局内存传输到其...</li><li><a href="https://github.com/NVIDIA/T">GitHub - NVIDIA/T</a>

<a href="https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_linear.py#L141">TransformerEngine/transformer_engine/pytorch/module/layernorm_linear.py at main · NVIDIA/TransformerEngine</a>: 一个用于在 NVIDIA GPU 上加速 Transformer 模型的库，包括在 Hopper 和 Ada GPU 上使用 8 位浮点 (FP8) 精度，以更低的内存占用提供更好的性能...</li><li><a href="https://arxiv.org/abs/2407.09577">Flash normalization: fast RMSNorm for LLMs</a>: RMSNorm 被许多 LLM（如 Llama, Mistral 和 OpenELM）所采用。本文详述了 FlashNorm，它是 RMSNorm 后接线性层的一种精确且更快速的实现。参见 https://huggingfac...</li><li><a href="https://arxiv.org/abs/2404.12362">Transformer tricks: Removing weights for skipless transformers</a>: He 和 Hofmann (arXiv:2311.01906) 详述了一种没有 V 和 P（注意力后投影）线性层的 skipless Transformer，这减少了权重总数。然而，该方案仅...</li><li><a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: 我们实证研究了针对流行的开源预训练 LLM 家族的一种简单层剪枝策略，发现在不同的问答基准测试中，性能下降极小，直到...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1262708582235766835)** (5 messages): 

> - `Human vs Animal Intelligence` (人类与动物智能)
> - `Neural Activity and Growth` (神经活动与增长)
> - `Gender Differences in Neuron Counts` (神经元数量的性别差异)


- **人类创造力超越动物智能**：一位成员讨论了**人类与动物**在智能方面的比较，指出虽然人类在感官能力上与其他动物相当，但在长期规划和创造力方面表现出色。
   - *人类拥有极其庞大的新皮层 (neocortex)* 和更多的褶皱，这使他们区别于其他边缘系统 (limbic systems) 以及运动和感官功能非常相似的哺乳动物。
- **神经活动促进神经元增长**：根据一位成员的评论，增加的**神经活动**会刺激额外的神经元生长，这意味着广泛的认知和推理可以显著区分一种动物与另一种动物。
   - 该评论暗示，与认知刺激较少的个体相比，**双博士学术型人才**会拥有更高的神经密度。
- **文化因素与神经元的性别差异**：一位成员提到，文化变异性可能解释了男性和女性在神经元计数上表现出的差异。
   - 该成员指出，关于**神经元计数**的研究可能没有考虑到相对的终身认知刺激和生活环境。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1262599690789326888)** (1 messages): 

> - `Mirror Neurons` (镜像神经元)
> - `Feature Representation` (特征表示)
> - `Circuit Reuse` (电路复用)
> - `Neurological Theories` (神经学理论)


- **镜像神经元以叠加态表示特征**：一场关于**镜像神经元**是否仅仅是以相关叠加态 (superposition) 表示特征的可能性讨论展开了。
   - 这可能意味着神经电路可以被复用于不同的功能，这是对当前神经学理论的一个引人入胜的转折。
- **关于电路复用的进一步影响**：*镜像神经元*在叠加态中复用电路可能会重塑对大脑效率的理解。
   - 这类理论可能会为 **AI** 和神经科学研究带来新的视角。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1262805109700558848)** (7 messages): 

> - `Tokenization in MLX & HF models` (MLX 与 HF 模型中的 Tokenization)
> - `Chat template application in tokenization` (Tokenization 中的 Chat template 应用)
> - `Top-level options in lm-eval` (lm-eval 中的顶级选项)


- **MLX 和 HF 模型中 Tokenization 的一致性**：一位成员注意到 MLX 和 HF 模型之间 Tokenization 的差异，特别关注 **BOS token** 的处理以及在不使用 `apply_chat_templates` 的情况下缺乏其他 Prompt 格式化的问题。
   - *Gemma* 模型在没有 **BOS token** 的情况下表现不佳被作为一个具体问题提出。
- **Tokenization 中的 Chat template 选项**：提到使用 `--apply_chat_template` 选项会在 Tokenization 之前将示例包装在 Chat template 中。
   - 关于该选项是特定于 **HF model args** 还是一个顶级的 **lm-eval** 选项存在一些困惑。
- **顶级 lm-eval 选项及其影响**：讨论澄清了 `--apply_chat_template` 确实是一个顶级的 **lm-eval** 选项。
   - 同时也寻求了确保 **MLX 模型** 接收与 **HF 模型** 相同参数的正确方法。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1262586004649279489)** (1 条消息): 

> - `LLM 的动态评估 (Dynamic Evaluation)`
> - `EleutherAI 的酷炫成果`
> - `持续学习 (Continual Learning)`
> - `元学习 (Meta-Learning)` 


- **关于 EleutherAI 和动态评估的查询**：一位成员询问 EleutherAI 是否在 LLM 的动态评估方面做过任何工作，并提到了一项据称从未发表过的酷炫成果。
   - 找到的最接近的参考资料是 [Gwern 关于动态评估的一篇帖子](https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section)，但它并不隶属于 EleutherAI。
- **分享的动态评估资源**：一位成员分享了 [Gwern 关于动态评估的帖子链接](https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section)，讨论了压缩 Transformer 和元学习等相关话题。



**提到的链接**：<a href="https://gwern.net/doc/ai/nn/dynamic-evaluation/index#hardt-sun-2023-section">‘dynamic evaluation (NN)’ 标签 · Gwern.net</a>：未找到描述

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1262484568812945480)** (77 条消息🔥🔥): 

> - `Anthropic 的公关策略`
> - `Token 限制的影响`
> - `评估门控 (Evaluation Gating)`
> - `Claude Engineer 2.0`
> - `Qwen2 技术报告` 


- **Anthropic 的每日公关策略**：Anthropic 正在熟练运用“每日挤牙膏”式的公关策略，宣布在 Anthropic API 中将 Claude 3.5 Sonnet 的最大输出 Token 限制从 4096 翻倍至 8192。
   - 一位社区成员表示如释重负，并分享了最近如何被之前的限制所困扰。[相关推文](https://x.com/alexalbert__/status/1812921642143900036)。
- **结构化数据的 YAML 与 JSON 之争**：一位用户主张在 Prompt 中生成结构化数据时使用 YAML 而非 JSON，观察到 Token 数量减少了 20-30%。
   - 有人对 JSON 的主导地位表示怀疑，并引用了一篇 [相关的 Arxiv 论文](https://arxiv.org/abs/2401.08500)。
- **SciCode：LLM 的新基准**：SciCode 是一个新的基准测试，挑战 LLM 为高级论文中的科学问题编写代码解决方案，据报道 GPT-4 和 Sonnet 3.5 的准确率不到 5%。
   - 该基准测试中约有 10% 基于诺贝尔奖级别的研究。[更多详情请点击此处](https://scicode-bench.github.io/)。
- **Andrej Karpathy 创立的 Eureka Labs**：Andrej Karpathy 宣布成立他的新 AI+教育公司 Eureka Labs，旨在建立一所 AI 原生学校，从 AI 课程 LLM101n 开始。
   - 内容将免费提供，收入将通过围绕这些材料运行的数字/实体学习小组产生。[公告推文](https://x.com/karpathy/status/1813263734707790301)。
- **Qwen2 技术报告发布**：Qwen2 发布了一套全面的语言模型系列，超越了 Qwen1.5，并在各种基准测试中与闭源模型竞争。
   - 该系列包括从 5 亿到 720 亿参数的模型，并提供 Dense 和 MoE 模型。[阅读报告](https://hf.co/papers/2407.10671)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：为了向克利奥帕特拉（Cleopatra）致敬——她的辉煌命运终结于悲惨的蛇类事件，我们自豪地发布了 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在 ... 下使用。</li><li><a href="https://x.com/karpathy/status/1813273726441652683">Andrej Karpathy (@karpathy) 的推文</a>：好问题。我确实希望 Eureka Labs 成为一个正规的、自给自足的企业，但我也不想对教育内容设置门槛。我的默认想法是内容本身是免费的...</li><li><a href="https://youtu.be/OZmakgRZYxU?si=ThCmiCCp49V7Rq4n">AI 独角兽的陨落：Graphcore 被软银收购</a>：合并、收购、退出。这是任何初创公司的目标。但对于该行业的第一家 AI 独角兽来说，这是我们预期的结果吗？在销售业绩平平之后 ...</li><li><a href="https://x.com/MinyangTian1/status/1813182904593199553">Minyang Tian (@MinyangTian1) 的推文</a>：SciCode 是我们推出的新基准测试，挑战 LLM 为高级论文中的科学问题编写代码解决方案。这些挑战由博士们精心设计；我们约 10% 的基准测试基于诺贝尔奖获奖 ...</li><li><a href="https://x.com/skirano/status/1812943785237639218?s=46">Pietro Schirano (@skirano) 的推文</a>：隆重推出 Claude Engineer 2.0，带有 Agent！🚀 这是迄今为止最大的更新，增加了代码编辑器和代码执行 Agent，以及动态编辑功能。在编辑文件（尤其是大文件）时，Eng...</li><li><a href="https://x.com/goodside/status/1812977352085020680">Riley Goodside (@goodside) 的推文</a>：9.11 大于 9.9。</li><li><a href="https://x.com/OfirPress/status/1813202497864937825">Ofir Press (@OfirPress) 的推文</a>：SciCode 是我们的新基准测试，包含 338 个编程挑战，由物理、数学和生物学博士根据其领域的论文编写。其中许多问题来自诺贝尔奖获奖论文！我希望...</li><li><a href="https://x.com/xenovacom/status/1813258097185448377">Xenova (@xenovacom) 的推文</a>：隆重推出 SmolLM：一个新的 SOTA 系列模型，包含 135M、360M 和 1.7B 参数，非常适合端侧部署！🔥 我们还上传了模型的 ONNX 权重，这意味着它们可以在你的浏览器中本地运行...</li><li><a href="https://arxiv.org/abs/2401.08500">使用 AlphaCodium 进行代码生成：从 Prompt Engineering 到 Flow Engineering</a>：代码生成问题不同于普通的自然语言问题——它们需要匹配目标语言的精确语法，识别正常路径（happy paths）和边缘情况，关注大量...</li><li><a href="https://python.useinstructor.com/">欢迎来到 Instructor - Instructor</a>：未找到描述</li><li><a href="https://x.com/alexalbert__/status/1812921642143900036">Alex Albert (@alexalbert__) 的推文</a>：@AnthropicAI 开发者的好消息：我们已将 Anthropic API 中 Claude 3.5 Sonnet 的最大输出 Token 限制从 4096 翻倍至 8192。只需添加 Header "anthropic-beta": "max-tok...</li><li><a href="https://x.com/YiTayML/status/1813262126162845772">Yi Tay (@YiTayML) 的推文</a>：决定开始一个新的关于 LLM 时代模型架构的博客系列。😀 这是第一部分，关于更广泛的架构，如 Transformer Encoders/Encoder-Decoders、PrefixLM 和去噪目标...</li><li><a href="https://x.com/karpathy/status/1813263734707790301">Andrej Karpathy (@karpathy) 的推文</a>：⚡️ 很高兴分享我正在创办一家名为 Eureka Labs 的 AI+教育公司。公告如下：--- 我们是 Eureka Labs，正在建立一种新型的 AI 原生学校。我们如何应用...</li><li><a href="https://x.com/huybery/status/1813046544683442456">Binyuan Hui (@huybery) 的推文</a>：🔥 Qwen2 技术报告。📒 https://hf.co/papers/2407.10671 我们发布了一套全面的基础和指令微调语言模型，涵盖了从 5 亿到 720 亿参数范围...</li><li><a href="https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising">BERT 和 T5 怎么了？关于 Transformer Encoders、PrefixLM 和去噪目标 — Yi Tay</a>：关于模型架构的系列博客第一部分：BERT 和 T5 怎么了？关于 Transformer Encoders、PrefixLM 和去噪目标的思考</li><li><a href="https://github.com/AnswerDotAI/bert24">GitHub - AnswerDotAI/bert24</a>：通过在 GitHub 上创建账户来为 AnswerDotAI/bert24 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d4p1t6/comment/l6g1b3t/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://analyticsindiamag.com/ai-news-updates/hugging-face-announces-profitability-with-free-and-open-source-models/">Hugging Face 宣布通过免费和开源模型实现盈利</a>：

Hugging Face 宣布通过免费和开源模型实现盈利</a>：未找到描述</li><li><a href="https://x.com/elder_plinius/status/1813181896789987411?s=46">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>：gg 引用 George McGowan (@GjMcGowan) 的话：这个网站提议让我与 AI 讨价还价买床垫。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1262687240929153104)** (1 条消息): 

> - `XState`
> - `LangGraph`
> - `LLM Agents` 


- **基于 XState 的 LLM Agents 开发中 (WIP)**：一位成员分享了他们正在进行的开发工作，使用 **XState** 创建由状态机驱动的 LLM Agents，代码托管在 [GitHub](https://github.com/statelyai/agent) 上。
   - 他们计划增加更多对比 **LangGraph** 和 **XState** 的示例。
- **LangGraph 与 XState 的对比**：该成员暗示即将发布关于构建 LLM Agents 时 **LangGraph** 与 **XState** 的对比。
   - 此次对比旨在展示每种方法的差异和优势，并在过程中提供更多示例。



**提到的链接**：<a href="https://github.com/statelyai/agent">GitHub - statelyai/agent: 使用 XState 创建由状态机驱动的 LLM Agents</a>：使用 XState 创建由状态机驱动的 LLM Agents - statelyai/agent

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1262486334434709524)** (27 条消息🔥): 

> - `LM Studio Android 应用访问`
> - `LM Studio 中的图形 Bug`
> - `基于云的 LM Studio`
> - `LM Studio 中 llama.cpp 的错误`
> - `H2O.ai Danube3 模型问题` 


- **在家庭网络中通过 Android 应用访问 LM Studio**：一位用户询问是否有人正在使用 Android 应用在家庭网络中访问其 LM Studio 服务器。
   - 建议使用 Wireguard 或 Tailscale 等 VPN 来安全地远程访问服务器。
- **LM Studio 中的图形 Bug**：一位成员注意到一个图形 Bug，即 `f32` 因为在文件夹中而被识别为 0 大小。
   - 尽管该 Bug 只是视觉上的，但它会显著影响用户体验。
- **模型架构 'gemma2' 的 Llama.cpp 错误**：**Anusha** 在尝试加载模型到 `llama.cpp` 时收到“未知模型架构：gemma2”的错误。
   - 建议在特定频道发布该问题，并提供 LM Studio 版本和具体模型的详细信息。
- **Flash Attention 导致模型加载问题**：一位用户报告在 RTX 3090 上运行 F16 GGUF 模型时出现问题，认为硬件不足。
   - 最终发现问题是由 Flash Attention 引起的，禁用后问题得到解决。
- **不建议在 LM Studio 中使用 Intel GPU**：关于 LM Studio 与 Intel A750 8G 兼容性的查询导致了不建议使用 Intel GPU 的建议。
   - 这些 GPU 在处理 AI 任务时速度较慢，且由于 OpenCL 支持已弃用，无法运行最新的模型。



**提到的链接**：<a href="https://huggingface.co/h2oai/h2o-danube3-4b-chat-GGUF">h2oai/h2o-danube3-4b-chat-GGUF · Hugging Face</a>：未找到描述

  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1262483946554265734)** (28 条消息🔥): 

> - `Hermes 2`
> - `Mistral issues`
> - `Model Merging`
> - `Open Empathic` 


- **L3-12B-Lunaris-v1-GGUF 与 L3-12B-Lunaris-v1-i1-GGUF 的区别**：一位成员询问了 **L3-12B-Lunaris-v1-GGUF** 和 **L3-12B-Lunaris-v1-i1-GGUF** 之间的区别，得到的澄清是常规量化与 imatrix 量化的区别。
   - 有人建议 *LLM 可以免费下载和测试，所以尽管去尝试*，但没有指出具体的性能差异。
- **适用于 128GB RAM 系统的本地编程模型**：一位成员寻求适用于 **128GB RAM** 系统且精通 TypeScript 的最佳本地编程模型推荐。
   - 建议使用 **Deepseek V2**（非 lite 版本），因为它拥有 **21B 专家参数和 236B 总参数**，但提醒注意潜在的 OOM 问题。
- **Mamba-Codestral-7B-v0.1 讨论**：分享了 [Mamba-Codestral-7B-v0.1 模型卡](https://huggingface.co/mistralai/mamba-codestral-7B-v0.1) 的链接，宣传该模型的线性时间推理和代码生产力能力。
   - 另一位成员指出，该模型尚未在 `llama.cpp` 中得到支持，相关 PR 正在进行中。
- **LM Studio 中视觉模型的问题**：成员们讨论了在 **LM Studio** 中使用视觉模型的挑战，指出需要同时安装模型文件和 `mmproj` 文件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>: 作为对克利奥帕特拉（Cleopatra）的致敬，我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用。</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1262731181313626122)** (18 条消息🔥): 

> - `LMS Model Loading Speed Issue`
> - `Gemma 2 Support`
> - `Phi 3 Small Support`
> - `Llama.cpp Limitations` 


- **确认 LMS 模型加载速度问题**：**用户报告** LMS 加载模型的时间比早期版本长得多，有时需要几分钟。
   - *一旦模型被卸载并重新加载，加载时间会降至 2-4 秒*，符合预期性能。
- **将支持 Gemma 2，但不提供 Phi 3 small 支持**：一位成员宣布将**支持 Gemma 2**，但不支持 **Phi 3 small**。
   - *Llama.cpp* 是限制因素，因为它不支持 **Phi 3 small**。


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/)** (1 条消息): 

magiikorb: 而且 M3 ultra 甚至都不在那。
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1262858589660516404)** (1 条消息): 

> - `Mistrals Mathstral Release`
> - `Community Models Program`
> - `Mathstral Performance`
> - `GGUF Quantization`
> - `LM Studio Discord Engagement` 


- **Mistral 的 Mathstral 发布承诺 STEM 领域的卓越表现**：**Mistral AI** 发布了 **Mathstral**，这是一个专门用于 STEM 和高级推理的微调模型，在主要的 STEM 类别中表现优于基础的 **Mistral 7B**。
   - 访问 [Hugging Face 上的模型](https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF) 以探索其功能，并加入 [Discord](https://discord.gg/aPQfnNkxGC) 上的讨论。
- **社区模型亮点计划展示 Mathstral**：LM Studio 亮点计划正在推荐 **Mathstral**（一个社区贡献的模型），鼓励在 [Discord](https://discord.gg/aPQfnNkxGC) 上进行讨论和探索。
   - *模型创建者：* [MistralAI](https://huggingface.co/mistralai)，*原始模型：* [mathstral-7B-v0.1](https://huggingface.co/mistralai/mathstral-7B-v0.1)，由 [bartowski](https://huggingface.co/bartowski) 提供 **GGUF 量化**。



**提到的链接**: <a href="https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF">lmstudio-community/mathstral-7B-v0.1-GGUF · Hugging Face</a>: 未找到描述

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1262534504028045432)** (5 条消息): 

> - `Evol-Instruct V2`
> - `Auto Evol-Instruct`
> - `Q-Sparse`
> - `BitNet b1.58` 


- **WizardLM 发布 Evol-Instruct V2**：WizardLM 宣布推出 [Evol-Instruct V2](https://x.com/WizardLM_AI/status/1812844503251947978)，拥有全自动流水线，将 WizardLM-2 从三个进化领域（对话、代码和数学）扩展到数十个领域。
   - 团队希望这项技术能够为 AI 研究人员在训练和评估 LLM 时提升公平性和效率。
- **Auto Evol-Instruct 表现优于专家**：Auto Evol-Instruct 可以自动进化指令数据，实验表明它在微调各种能力方面优于人工设计的方法。
   - Auto Evol-Instruct 在指令遵循的 MT-bench 上实现了 **10.44%** 的提升，在代码生成的 HumanEval 上提升了 **12%**，在数学推理的 GSM8k 上提升了 **6.9%**。
- **Q-Sparse 加速 LLM 计算**：Hongyu Wang 介绍的 [Q-Sparse](https://x.com/realHongyu_Wang/status/1813112679734911169) 声称通过将重点从内存受限转向计算受限过程，显著加快了 LLM 的计算速度。
   - 这是在 BitNet b1.58 发布四个月后推出的，后者将 LLM 压缩到了 **1.58 bits**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/WizardLM_AI/status/1812844503251947978">来自 WizardLM (@WizardLM_AI) 的推文</a>: 🎉今天我们宣布推出 Evol-Instruct V2 !!! 🔥 Auto Evol-Instruct 是 WizardLM-2 最重要的技术之一。论文链接：https://arxiv.org/pdf/2406.00770 我们构建了一个全自动...</li><li><a href="https://x.com/realHongyu_Wang/status/1813112679734911169">来自 Hongyu Wang (@realHongyu_Wang) 的推文</a>: 距离我们发布 BitNet b1.58 已经 4 个月了🔥🔥 在我们将 LLM 压缩到 1.58 bits 后，1bit LLM 的推理不再受内存限制，而是受计算限制。🚀🚀今天我们介绍 Q-Sparse，它可以显著...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1262546544041787412)** (3 条消息): 

> - `How AI Really Works`
> - `SpreadsheetLLM`
> - `Synth Data` 


- **通过交互式可视化理解 AI**：一段名为“[How AI Really Works](https://youtu.be/pj8CtzHHq-k)”的 YouTube 视频通过交互式可视化解释了 **Llama 3** 等 LLM 的工作原理。
   - 视频强调了为什么开源在这些模型的开发和理解中至关重要。
- **SpreadsheetLLM 将彻底改变数据管理**：**Microsoft** 发布了一个新的 LLM —— **SpreadsheetLLM**，专为高级电子表格任务设计，预示着数据管理和分析领域的变革性应用。
   - 一篇[预印本论文](https://arxiv.org/abs/2407.09025)被悄然发布，引发了关于就业市场影响的讨论，有人暗示“Karen 可能很快就要失业了”。
- **社区对 Synth Data 的热议**：一位成员分享了一个 [Twitter 链接](https://twitter.com/pratyushmaini/status/1752337225097076809)，讨论了 AI 社区中合成数据（Synthetic Data）的应用和发展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/pj8CtzHHq-k">How AI Really Works (以及为什么开源很重要)</a>: 一段简短的演讲，使用交互式可视化来解释当前的 AI 到底是如何运作的，特别是像 Llama 3 这样的 LLM 仅仅是...</li><li><a href="https://arxiv.org/abs/2407.09025">SpreadsheetLLM: 为大语言模型编码电子表格</a>: 电子表格具有庞大的二维网格、各种布局和多样的格式选项，对 LLM 构成了显著挑战。为此，我们推出了 Spread...</li><li><a href="https://www.thestack.technology/microsoft-llm-spreadsheet-llm/">微软发布了一款擅长编码电子表格的大语言模型</a>: 新的 LLM 具有“改变数据管理和分析的潜力，为更智能、更高效的用户交互铺平道路。”
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1262489087500226622)** (58 条消息🔥🔥): 

> - `热浪讨论`
> - `白屋顶涂料创新`
> - `Deepseek Coder 对比`
> - `FP8 黑客松`
> - `热岛效应` 


- **成员讨论热浪影响**：一位成员提到现在的热浪非常“疯狂”，另一位成员指出在拉斯维加斯的朋友“即使空调开到最大也快被烤熟了”。
   - 对话建议将屋顶涂成白色作为降温措施，并引用了一篇 [耶鲁大学的文章](https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities)，讨论了白色屋顶在缓解城市热岛效应方面的益处。
- **超白涂料的降温潜力**：一位成员分享了 2021 年发明的一种超白涂料，它可以反射 98% 的阳光。
   - 另一位成员链接了一个 [YouTube 视频](https://youtu.be/KDRnEm-B3AI)，演示了如何利用家用物品制作这种涂料，以及它在反射红外线和降温方面的有效性。
- **Crusoe 办公室的 FP8 黑客松活动**：一位成员宣布在 Crusoe 的旧金山办公室举办一场关于 FP8 的黑客松，重点是改进推理、微调（Fine-tuning）和预训练（Pretraining）。
   - 参与者将使用 L40S 节点进行开发，并听取演讲者关于 FP8 相关主题的分享。[活动详情点击此处](https://lu.ma/hpb5svgw)。
- **Deepseek Coder 16B 对比 Mistral 代码模型**：成员们讨论了 Deepseek Coder 模型的效能，强调了 Debug 能力比 Zero-shot 性能更重要。
   - 一位正在测试 Deepseek Coder v2 16B 的成员强调了其速度（>60t/s）和非量化矩阵，而另一位成员则对其 Debug 能力表示怀疑。
- **Mistral 与 Deepseek 代码模型备受关注**：Mistral AI 拥有 256k 上下文的新模型在某些人看来反响平平，他们更倾向于 Deepseek Coder 的能力。
   - 讨论包括了一些批判性的对比，指出了 Mistral 在 MBPP 表现上的问题以及某些模型缺乏自我 Debug（Self-debugging）功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/HCSolakoglu/status/1812984327510085883">Hasan Can (@HCSolakoglu) 的推文</a>: Llama3 405b 已添加到 @OpenRouterAI，目前还没有供应商。同时附上了 Huggingface 模型权重页面。发布应该临近了。</li><li><a href="https://youtu.be/KDRnEm-B3AI?si=UOzyzARqomlGZ1mS">利用超市物品制作红外冷却涂料（含新型 CaCO₃ 微球合成）</a>: 查看我的赞助商 Brilliant，通过此链接可免费试用 30 天：https://brilliant.org/nighthawk。在本视频中，我们探索了制作尖端涂料的新方法...</li><li><a href="https://x.com/MistralAI/status/1813222156265791531">Mistral AI (@MistralAI) 的推文</a>: https://mistral.ai/news/mathstral/ https://mistral.ai/news/codestral-mamba/</li><li><a href="https://lu.ma/hpb5svgw">FP8 Island Vibes 黑客松 · Luma</a>: 加入我们 7 月 20 日的活动，专注于使用 FP8 进行开发，在基于 Crusoe L40S 节点的 Brev notebook 上改进推理、微调和预训练 –…</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/pull/32">由 alonsosilvaallende 添加 outlines + llama-cpp-python 示例 · Pull Request #32 · NousResearch/Hermes-Function-Calling</a>: 在 NousResearch 的 Hermes 2 Pro Llama 3 8B GGUF 模型上使用 outlines + llama-cpp-python 的示例。该示例展示了：如何通过遵循 Pydantic schema 生成合成数据，以及如何回答...</li><li><a href="https://e360.yale.edu/features/urban-heat-can-white-roofs-help-cool-the-worlds-warming-cities">城市热岛：白色屋顶能否帮助冷却全球变暖中的城市？</a>: 长期以来，人们都知道安装白色屋顶有助于减少城市的热量积聚。但新的研究表明...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1262743547321384992)** (6 messages): 

> - `阿拉伯符号的分词问题`
> - `生成 PPO/DPO 数据集的工具`
> - `分词的可逆性` 


- **阿拉伯符号的分词无法正确解码**：一位成员指出在使用 **tiktoken 库** 解码阿拉伯符号时存在问题，遇到了特殊 Token 而不是原始字符串。
   - 他们解释了在解码过程中使用 `errors='replace'` 如何将无效的字节序列替换为特殊符号 ``，并质疑 **LLMs** 如何能从这些 Token 中准确生成文本。
- **尚无已知的生成 PPO/DPO 数据集的工具**：一位成员询问是否有从原始数据生成 **PPO/DPO 数据集** 的工具，但另一位成员回答说他们不知道有任何此类工具。
- **分词的可逆性可能有所不同**：*一位成员* 表示他们在分词可逆性方面遇到了不同的结果，建议它可能解码为 **UTF-8 序列** 或 **0x80-0xFF** 字节，特别是在使用 `cl100k_base` 分词器时。


  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

wolfybl: Hi
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1262526297805160529)** (37 messages🔥): 

> - `GPTs Agents`
> - `Sora 发布推测`
> - `Lymsys 中的 GPT mini`
> - `OpenAI Platform`
> - `AI 编程` 


- **Sora 发布推测仍在继续**：成员们讨论了 **Sora** 的发布日期，一些人根据博客文章推测它可能会在 2024 年底或 Q4 发布，尽管 **OpenAI** 尚未发表任何官方声明。
   - 一位成员指出，准确的预测日期通常是那些接近实际发布的日期，应对随机的 Reddit 或 Twitter 帖子保持谨慎。
- **Lymsys 中的 GPT mini**：关于 Lymsys 中即将推出 **GPT mini** 的传闻引起了推测，一位成员简单地评论说这“很酷”。
   - 没有提供实质性信息，一些成员暗示随机预测大多是毫无根据的。
- **语音模式 (Voice mode) 与 Sora 演示**：成员们注意到发布的是 **Sora** 演示而不是预期的 **语音模式 (Voice mode)**，引发了进一步的推测。
   - 一位成员幽默地建议，开发人员可能误删了语音模式的所有代码，需要再次进行大量的投资和开发时间。
- **AI 编程优势**：一位成员强调，使用 **AI** 进行实验要容易得多，特别是对于**游戏机器学习**，你可以保存参数的权重 (weights) 和偏置 (biases)。
   - 另一位成员指出，与其他模型如 Claude 3.5 Sonnet 相比，**ChatGPT** 在编程方面表现更好，称其虽然不是遥遥领先，但确实明显更好。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1262487577873616896)** (11 messages🔥): 

> - `使用 GPT-4 编写移动游戏代码`
> - `使用 GPT-4 学习编程`
> - `使用 GPT 进行开发的挑战`
> - `创建客户支持聊天机器人`
> - `调整 GPT 的回复语气` 


- **使用 GPT-4 编写移动游戏代码**：成员们讨论了 GPT-4 如何协助编写移动游戏代码，提到它可以帮助编写测试和代码，但经常产生带有 Bug 的 **输出 (output)**，需要反复检查。
   - 一位成员推荐了 **React Native** 和 **FastAPI** 等工具，并分享了 GPT-4 虽然有用但需要验证错误的经验。
- **从零开始使用 GPT-4 学习编程**：一位成员分享了他们在 **零编程基础** 的情况下使用 GPT-4 成功创建完整 Web 应用的经历，强调清晰的解释可以帮助 GPT 有效地提供协助。
   - 注意到虽然 GPT-4 可以提供有用的响应，但用户必须确保代码正确运行，并在学习过程中保持自主性。
- **针对孟加拉语 (Bengali) 和孟加拉式英语 (Banglish) 聊天机器人的微调**：一名学生正在开发一个 **孟加拉语和孟加拉式英语 (Banglish)** 的客户支持机器人，询问使用 100 条对话微调模型是否能帮助其学习对话模式。
   - 另一位成员解释说，模型不会永久适应，但可以在 Context length 内捕捉模式，如果超过 Context window，这些模式可能会丢失。
- **GPT-4 正式回复语气的问题**：一位成员经历了 GPT-4 的 **回复语气** 从随性变为非常正式的转变，询问是否有办法让机器人听起来不那么“书呆子气”。
   - 据报道，无论如何修改 Prompt，这种变化都会持续存在，导致对该项目的兴趣下降。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1262585044061655131)** (5 messages): 

> - `Different languages affecting model performance`（不同语言对模型性能的影响）
> - `Prompting in native language vs English`（使用母语与英语进行 Prompting 的对比）
> - `Model's handling of regional slang and idioms`（模型对地区方言和成语的处理）


- **模型性能因语言训练而异**：一位成员指出，模型性能可能因语言而异，这很大程度上取决于该语言获得的训练数据量。
   - 该成员提到，由于 System Prompts 中的示例，使用英语或其他语言（如中文）进行 Prompting 可能会导致模型以相应的语言进行回复。
- **使用母语还是英语 Prompting 以获得更好结果**：一位成员询问，是为了获得更好的结果而应该直接用法语 Prompting，还是用英语 Prompting 并要求以法语回答。
   - 另一位成员建议，像 GPT-4o 这样的模型并不擅长处理地区方言、成语和口语，建议避免大量使用这些表达方式以获得更好的效果。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1262585044061655131)** (5 messages): 

> - `Language model performance across different languages`（语言模型在不同语言中的性能）
> - `Prompting in different languages`（在不同语言中进行 Prompting）
> - `Language preferences for model responses`（模型响应的语言偏好）
> - `Regional slang, idioms, and colloquialisms in GPT models`（GPT 模型中的地区方言、成语和口语）


- **模型性能随语言暴露程度而异**：一位成员推测 **语言模型** 在经过更广泛训练的语言中表现更好，并指出 Prompt 语言的选择会影响响应质量。
   - 他们分享到，**英语 Prompt** 更有可能获得 **英语** 响应，这与 **中文** 等其他语言的情况类似。
- **使用法语 Prompting 以获得更好的响应**：一位用户询问，与使用英语提问然后翻译相比，直接使用 **法语** 进行 Prompting 是否会在法语环境下产生更好的响应。
- **GPT 在处理地区方言和成语时表现不佳**：一位成员强调 **GPT-4** 及更低版本在理解 **地区方言、成语和口语** 方面存在困难。他们建议避免大量使用这些表达方式以获得更好的结果。


  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1262796132061548585)** (1 messages): 

> - `LlamaIndex Webinar`（LlamaIndex 网络研讨会）
> - `RAG Improvement`（RAG 改进）
> - `Deasie Automated Labeling`（Deasie 自动打标签）
> - `LlamaParse Tool`（LlamaParse 工具）


- **关于高级解析和元数据的 LlamaIndex 网络研讨会**：一场新的 [LlamaIndex Webinar](https://lu.ma/ufx77ve8) 将于本周四太平洋时间上午 9 点举行，届时 Deasie 的联合创始人将讨论如何通过高级解析和 **Metadata** 来 **改进 RAG**。
- **解析 + 元数据的实验结果**：研讨会将展示研究论文中的结果，证明结合 **Parsing** 和 **Metadata** 可以增强性能。
- **Deasie 在增强 RAG 中的作用**：Deasie 的标注工作流通过自动生成分层 Metadata 标签来改进 LLM 在 10,000 多份文档上的检索，从而增强 RAG。更多详情请见其 [官网](https://deasie.com/)。
- **使用 Deasie 自动对非结构化数据进行编目**：Deasie 为企业知识管理和合规性提供大规模非结构化数据的自动打标签和编目功能。
- **LlamaParse GitHub 仓库**：LlamaParse 是一个用于优化 RAG 文件解析的工具，可以通过其 [GitHub 页面](https://github.com/run-llama/llama_parse) 访问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/ufx77ve8">LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction · Zoom · Luma</a>: 我们很高兴能与 Deasie 的联合创始人（Reece, Leonard, Mikko）共同举办一场关于通过高级解析和元数据改进 RAG 的研讨会。数据……</li><li><a href="https://deasie.com/">Deasie | Data Governance for Language Model Applications</a>: Deasie 平台确保只有安全、高质量且相关的数据被输入到语言模型中。由企业软件 AI 和数据治理领域的获奖团队开发。</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为优化 RAG 进行文件解析。欢迎在 GitHub 上为 run-llama/llama_parse 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1262523257265983652)** (4 条消息): 

> - `Document RAG`
> - `Graph Query Algorithm`
> - `LlamaIndex Webinar`
> - `Sonnet-3.5 Chart Understanding` 


- **多模态 RAG 预示文档处理的未来**：在一份新的 [cookbook](https://t.co/KWsVGwT3jD) 中，重点介绍了使用 LlamaParse 和 GPT-4o 的多模态 RAG 架构，用于处理包含丰富文本、图表、图形和表格的幻灯片。
   - *其核心是文本/图像混合方法*，增强了 Document RAG 的能力。
- **构建你自己的 Graph Query Algorithm**：利用 [LlamaIndex](https://t.co/atFLrXbYtQ) 和 Mistral，你可以创建自定义的图查询算法，融合 text-to-cypher 或向量搜索技术。
   - 只要拥有必要的资源，你就可以灵活地定义自己的查询算法。
- **即将举行的关于 RAG 增强的 LlamaIndex Webinar**：与 Deasie 联合创始人共同举办的新 [webinar](https://t.co/7pgCBKx1IL) 将重点讨论如何利用高级解析和元数据来改进 RAG。
   - 强调了为 RAG 构建正确的数据处理层的重要性，这对于有效的 AI 实施至关重要。
- **Sonnet-3.5 在图表理解方面表现出色**：Sonnet-3.5 在图表理解方面的表现优于 GPT-4o，特别是在将图表数值推导为结构化表格方面。
   - 新发布的 [LlamaParse](https://t.co/Dq3xaVfXio) 支持轻松集成最先进的多模态模型，以增强数据解释能力。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1262488561517592628)** (47 条消息🔥): 

> - `LLM Response Sources` (LLM 响应来源)
> - `Service Context and Indexing Models` (Service Context 与索引模型)
> - `Vector Datasets and Tools` (向量数据集与工具)
> - `Parallel Index Loading` (并行索引加载)
> - `PropertyGraphIndex Embeddings` (PropertyGraphIndex 嵌入)


- **在查询中检索 LLM 响应来源**：要在对文本文件进行查询时获取 LLM 响应的来源，请对响应对象使用 `get_formatted_sources()` 方法，或者对评估结果使用 `display_eval_sources()` 函数 ([更多详情](https://github.com/jerryjliu/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemo.ipynb))。
   - 这些方法以格式化的方式返回响应的来源，对于调试和理解数据溯源（data provenance）非常有用。
- **关于 Service Context 和 Embedding 模型的说明**：LlamaIndex 的最新版本不再需要 `serviceContext`；你可以全局设置 LLM/Embedding 模型，或者直接将它们传递到相关模块中 ([LLM 自定义文档](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-changing-the-underlying-llm))。
   - 例如，你可以直接将 `gpt-4o` 传入查询引擎，从而简化自定义设置，而无需使用默认的 Embeddings。
- **寻求易于访问的公共向量数据集**：成员们正在寻找托管公共向量数据集（如 Wikipedia）的服务，以避免自行托管的基础设施开销。
   - 一个建议是托管自己的 Wikipedia 向量存储，尽管用户仍然倾向于使用预托管、即开即用的查询服务。
- **优化大型数据集的索引加载**：一位成员报告称加载大型索引需要相当长的时间，并询问了加速方法，例如并行处理。
   - 讨论建议使用 `QueryPipelines` 或其他方法来优化并可能并行化加载过程。
- **在 PropertyGraphIndex 中嵌入数据**：`PropertyGraphIndex.from_documents()` 方法是创建嵌入并存储在 Neo4J 节点中的地方 ([源代码](https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248))。
   - 源代码的这一部分详细说明了嵌入创建过程，以便有效地将数据存储在图数据库中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/78754465/how-to-merge-multiple-at-least-two-existing-llamaindex-vectorstoreindex-instan">如何合并多个（至少两个）现有的 LlamaIndex VectorStoreIndex 实例？</a>: 我正在使用 LlamaIndex，并创建了两个独立的 VectorStoreIndex 实例，每个实例来自不同的文档。现在，我想将这两个索引合并为一个索引。这是我目前的...</li><li><a href="https://stackoverflow.com/questions/78754465/how-to-merge-multiple-at-least-two-existing-llamaindex-">如何合并多个（至少两个）现有的 LlamaIndex VectorStoreIndex 实例？</a>: 我正在使用 LlamaIndex，并创建了两个独立的 VectorStoreIndex 实例，每个实例来自不同的文档。现在，我想将这两个索引合并为一个索引。这是我目前的...</li><li><a href="https://github.com/run-llama/llama_index/blob/f092d90bd5934097f5c166014f5d99a3e07ea999/llama-index-core/llama_index/core/indices/property_graph/base.py#L248">run-llama/llama_index 中的 llama_index/llama-index-core/llama_index/core/indices/property_graph/base.py</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://cloud.llamaindex.ai/.">LlamaCloud</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-changing-the-underlying-llm">自定义 LLM - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1262814317300944999)** (3 条消息): 

> - `llamaindex property graph vs microsoft graphrag` (LlamaIndex Property Graph 对比 Microsoft GraphRAG)
> - `Graph rag functionalities` (GraphRAG 功能)
> - `Property graph features` (Property Graph 特性)


- **LlamaIndex 对比 Microsoft GraphRAG**：一位用户询问了 **LlamaIndex Property Graph** 和 **Microsoft GraphRAG** 之间的区别。
   - *Cheesyfishes* 解释说，GraphRAG 将实体总结并聚类为“社区”（communities）进行检索，而 Property Graph 支持 text-to-cypher、Embedding、关键词以及其他自定义检索方法。
- **GraphRAG 和 Property Graph 的功能**：[GraphRAG](#) 在图本身的处理上做得不多，主要进行实体的总结和聚类。
   - 相比之下，**Property Graph** 允许实现自定义功能，如 text-to-cypher、Embeddings 和自定义检索方法。


  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1262487715459633266)** (44 messages🔥): 

> - `Cohere Python Library`
> - `Cohere Discord Bot`
> - `Spam Awareness`
> - `Fireside Chats with Max Welling`
> - `Job Postings and Engagement` 


- **贡献 Cohere Python Library**：鼓励成员查看 [Cohere Python Library](https://github.com/cohere-ai/cohere-python) 以进行 open source 贡献。
   - 另一位成员提到他们是该库的重度用户，可能很快就会开始贡献。
- **Cohere Discord Bot 分类问题**：一位成员正在排查为什么他们的 Discord bot 将所有内容都归类在 'opensource' 下。
   - 另一位成员提议查看该 bot 的问题，但未做任何承诺。
- **防范垃圾信息倡议**：一位成员建议发布关于可疑链接的意识公告，以增强用户安全。
   - 另一位成员表示赞同，提议将其作为服务器 onboarding 流程的一部分。
- **与 Max Welling 的炉边谈话**：[C4AI](https://discord.gg/Jf6FPp3c?event=1262761447486787685) 正在举办一场与阿姆斯特丹大学 Max Welling 的交流会。
   - 在不当使用 @everyone 通知该活动后，发布了道歉。
- **Discord 上的职位发布与互动**：一位成员被提醒服务器内不允许发布职位信息。
   - 建议工作洽谈应私下处理，并强调专家收取的费用更高。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/cohere-ai/cohere-python">GitHub - cohere-ai/cohere-python: Python Library for Accessing the Cohere API</a>: 用于访问 Cohere API 的 Python 库。通过在 GitHub 上创建账号来为 cohere-ai/cohere-python 的开发做出贡献。</li><li><a href="https://docs.cohere.com/page/cookbooks#rag">Cookbooks</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1262738929174577213)** (1 messages): 

> - `Automatic post categorization`
> - `Channel specific categorization`
> - `Prompt adjustments` 


- **自动分类导致帖子路由错误**：目前正在进行一个将帖子自动分类到特定频道的项目。然而，所有帖子都被错误地路由到了 open-source 频道，这表明 Prompt 可能存在问题。
- **问题可能原因：Prompt 需要调整**：路由错误似乎是由于使用的 Prompt 导致的，希望这是一个容易修复的问题。尽管 r/openai 的帖子有专门的频道，但仍被错误地路由到了 open-source 频道。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1262617389783191613)** (9 messages🔥): 

> - `finetuning pipeline`
> - `Bengali chatbot`
> - `timestamps in MessageGraph`
> - `community help` 


- **请求 finetuning pipeline 详情**：一位成员请求另一位成员分享他们的 finetuning pipeline，认为这可能有助于找到问题的解决方案。
- **孟加拉语客服聊天机器人微调**：一位正在开发孟加拉语和孟加拉式英语（Banglish）客服聊天机器人的学生询问，使用 100 条真实的孟加拉语聊天对话来训练像 **Llama 3** 或 **GPT 3.5** 这样的 LLM，是否能帮助模型学习对话模式并提供准确的回复。
   - 一位成员建议研究能够引导特定行为的 Prompt，并分享了一个关于模型理解印地语/乌尔都语俚语的轶事。*“我只是开玩笑地让它说印地语/乌尔都语俚语，它表现得非常好。”*
- **MessageGraph 中的时间戳**：一位成员询问 **MessageGraph** 中的所有消息是否都可以带有时间戳，或者他们是否需要创建一个带有自定义状态的 **StateGraph**。
- **社区帮助频道反馈**：一位成员对在帮助频道提问后没有得到回复表示沮丧。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1262527098652856401)** (3 messages): 

> - `Automatic 1111 SD with 1.5`
> - `使用 LangChain & WebLLM 的浏览器 RAG`
> - `Verbis 发布`
> - `开源 GenAI 模型` 


- **Verbis App 发布**: Verbis 是一款开源的 MacOS 应用，专注于本地数据索引，并利用 **GenAI** 模型在不牺牲隐私的情况下提高生产力。
   - 核心特性包括**本地数据处理**、**开源**开发，且不向第三方发送数据。[在 GitHub 上查看](https://github.com/verbis-ai/verbis)。
- **100% 浏览器端 RAG，使用 LangChain & WebLLM**: 一位成员分享了一个 [YouTube 视频](https://youtu.be/MHuvSuK2dnY)，演示了如何使用 **Visual Agents** 在浏览器中部署 WebLLM 聊天模型进行问答。
   - 视频展示了将模型拖放到画布上并立即与其交互的无缝过程。



**链接提到**: <a href="https://youtu.be/MHuvSuK2dnY">Use 100% Browser Only WebLLM to Answer Questions!</a>: 在这段视频中，我使用 Visual Agents 将 WebLLM 聊天模型拖放到画布上，并立即开始向它提问。

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1262668115788169318)** (2 messages): 

> - `PyTorch tunner`
> - `训练指令模型`
> - `上下文长度调整`
> - `Mistral 的聊天模板问题` 


- **关于 PyTorch tunner 发布的问题**: 一位成员询问了 **PyTorch tunner** 的可用性。
- **很少有用户在指令模型上进行训练**: 一位成员观察到，没有多少用户在**指令模型**上进行训练，并建议用户可以根据需要更换模型。
- **上下文长度调整技巧**: 一位用户建议其他人根据需要增加上下文长度，但指出数值越高，消耗的 **VRAM** 越多。
- **Mistral 异常的聊天模板导致问题**: 一位成员指出，与其他模型相比，**Mistral** 奇怪的聊天模板经常导致问题。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1262535440330653728)** (4 messages): 

> - `Pull Request 已创建`
> - `关于 DPO 的讨论`
> - `集成工作` 


- **Pull Request 成功完成**: 一位成员宣布成功创建了一个新的 [pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/1756)。
   - *"好了，pull request 已创建。"*
- **DPO 集成挑战**: 讨论显示 **DPO** (Direct Policy Optimization) 方法虽然更简单，但不支持集成 **tokenization** 或 **masking**。
   - 一位成员得出结论，该扩展更适合 **SFT (Supervised Fine-Tuning)** 而非 DPO 变体。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1262750790167166988)** (5 messages): 

> - `lora_target_linear`
> - `LoRA 配置`
> - `Axolotl 微调` 


- **理解 lora_target_linear**: 一位成员询问：*“什么是 lora_target_linear？”*，并收到了详细解释，强调这是 Axolotl 中的一个配置选项，用于指定是否应将 LoRA 应用于模型内的线性层。
   - 解释中提到，当设置为 **true** 时，LoRA adapters 会修改线性层，从而在不从头开始训练整个模型的情况下实现高效微调。
- **将 lora_target_linear 设置为 false 的影响**: 一位成员询问了将 `lora_target_linear` 设置为 false 的影响，寻求进一步澄清。
   - *给出的消息中未提供详细回答。*



**链接提到**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f6280a5d-7259-4317-b9ca-adbe6c9066c3)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。

  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1262863256217850008)** (1 条消息): 

> - `Torchtune v0.2.0 发布`
> - `新模型与 recipes`
> - `数据集改进`
> - `社区贡献` 


- **Torchtune v0.2.0 发布**: 宣布发布 [torchtune v0.2.0](https://github.com/pytorch/torchtune/releases/tag/v0.2.0)，包含过去几个月来自社区的贡献。
- **令人兴奋的新模型与 Recipes**: **v0.2.0** 版本为稳定版主包带来了新的模型 🦙 和 recipes，以及诸如样本打包 (sample packing) 🚀 等数据集改进。



**提到的链接**: <a href="https://github.com/pytorch/torchtune/releases/tag/v0.2.0">Release v0.2.0 · pytorch/torchtune</a>: 概览 距离我们上次发布已经有一段时间了，我们在 torchtune 库中加入了大量酷炫的新功能，包括分布式 QLoRA 支持、新模型、样本打包 (sample packing) 等等！查看...

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1262614219086495834)** (5 条消息): 

> - `评估损失 (Eval loss) 计算`
> - `检查点 (Checkpoint) 优化`
> - `Recipe 修改`
> - `数据切分与评估` 


- **在不进行反向传播的情况下计算评估损失**: 一位成员询问如何在不进行 Backpropagation 的情况下计算评估数据集的损失，旨在绘制训练和评估数据集的损失曲线，以决定最佳的 checkpoints。
   - 另一位成员建议修改 [默认 recipe 配置文件](https://github.com/pytorch/torchtune/issues/1066) 以包含测试切分数据集和评估循环，并强调在模型中使用 `torch.no_grad()` 和评估模式。
- **优化检查点选择**: 原始查询集中在通过在每一步计算评估损失来寻找最佳检查点而避免过拟合。
   - 建议的权宜之计包括在实现更详细的每步评估设置之前，先对每个 epoch 的数据集进行评估。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1066.">Issues · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/datasets/v2.20.0/loading#slice-splits">Load</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1262827174596186286)** (1 条消息): 

> - `缩放 RoPE 嵌入`
> - `长上下文建模` 


- **针对长上下文缩放 RoPE 嵌入的 RFC**: 该 [RFC](https://github.com/pytorch/torchtune/issues/1183) 讨论了添加 RoPE 缩放方法，以支持大文档理解或代码补全等任务的长上下文建模。
   - *为了默认启用此功能，模型需要支持大于 8K 的上下文长度*。
- **大文档的长上下文建模**: 对于大文档理解或代码补全等任务，拥有较大的上下文长度（例如大于 8K）通常是有益的。
   - 该 RFC 提出了缩放 RoPE 嵌入的方法以适应此类需求。



**提到的链接**: <a href="https://github.com/pytorch/torchtune/issues/1183">[RFC] Adding RoPE scaling methods to support long context modeling · Issue #1183 · pytorch/torchtune</a>: 背景 对于大文档理解或代码补全等任务，拥有较大的上下文长度（例如 > 8K）通常是有益的。为了默认启用此功能，模型需要...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1262800347433537646)** (3 条消息): 

> - `ComfyUI 恶意节点攻击`
> - `迪士尼攻击事件`
> - `FBI 介入迪士尼攻击事件` 


- **ComfyUI 攻击者瞄准迪士尼**: 一位成员报告称，**ComfyUI 恶意节点攻击**背后的团体也对最近的**迪士尼攻击事件**负责。
   - 另一位成员表示：*该团体可能只是在戏弄随机路人*。
- **希望 FBI 介入**: 尽管个人不喜欢迪士尼，但一位成员表示希望 **FBI** 能调查**迪士尼攻击事件**。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 条消息): 

nodja: https://mistral.ai/news/codestral-mamba/
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 条消息): 

__ctrlaltdel__: https://youtu.be/pj8CtzHHq-k
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/)** (1 条消息): 

jbexta: 我会尝试在本周发布一个演示/教程 👍

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1262486895892627569)** (4 messages): 

> - `Open Interpreter 与 RayBan Stories 的配合使用`
> - `Rooting RayBan Stories 眼镜`
> - `关于通过 App 进行 Hacking 的看法`
> - `Google Glass 替代方案`
> - `O1 Light 及硬件预订更新` 


- **将 Open Interpreter 与 RayBan Stories 集成的挑战**：一位成员分享了他们在 **RayBan Stories** 上使用 Open Interpreter 的兴趣和困扰，强调了 Meta 缺乏 SDK 以及难以访问设备内部结构的问题。
   - 他们提供了[来自 Pastebin 链接](https://pastebin.com/wTsRMH3f)的规格参数，并讨论了诸如组件被胶水粘合等潜在障碍，以及对透明模型以方便探索的渴望。
- **考虑将 Google Glass 作为替代方案**：一位成员建议使用 **Google Glass** 作为替代方案，认为这是解决 RayBan Stories 难题的一个办法。
   - *该建议未提供更多细节或讨论。*
- **对 O1 Light 硬件预订延迟的沮丧**：多位成员对收到 **O1 Light 硬件** 的延迟表示沮丧，订单已下达超过 3 个月。
   - **预订情况**缺乏更新导致社区成员的耐心逐渐消磨。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/RayBanStories/comments/rlzyot/rayban_stories_codenamed_stella_runs_android_810/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://pastebin.com/z709s9Ru">## ADDITIONAL_DEFAULT_PROPERTIES#ro.oem_unlock_supported=1ro.usb.id.adb= - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1262787952032092196)** (3 messages): 

> - `GPT-4o 微调访问权限`
> - `OpenPipeAI 支持` 


- **GPT-4o 微调访问权限的困惑**：一位用户询问是否有人获得了 [GPT-4o 微调](https://openai.com/gpt-4o-and-gpt-4-fine-tuning-experimental-access/) 的权限。
   - 另一位用户澄清说，需要 OpenAI 方面的访问权限，并引用了 Kyle 的声明。
- **OpenPipeAI 现在支持 GPT-4o 训练**：[Corbtt 宣布](https://x.com/corbtt/status/1813018434822971556?t=qCi3vH2LH1KSho8x658urA&s=19) **OpenPipeAI** 已支持训练 GPT-4o，并强调应负责任地使用。
   - 这被建议作为高效使用课程积分的一个选项。



**提到的链接**：<a href="https://x.com/corbtt/status/1813018434822971556?t=qCi3vH2LH1KSho8x658urA&s=19">来自 Kyle Corbitt (@corbtt) 的推文</a>：如果你曾感到需要一个极其强大的微调模型……我们现在在 @OpenPipeAI 中支持训练 GPT-4o。请负责任地使用。😎

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1262531632221655071)** (2 messages): 

> - `tinygrad 的中间语言`
> - `tinygrad 中的调试与可视化` 


- **探索 tinygrad 的中间语言**：一位用户询问了 tinygrad 的中间语言（Intermediate Language）的外观，并询问了深度学习算子（operators）在 IR 中的存储方式。
   - 另一位用户建议在特定频道提问，并提供了一个技巧：运行 tinygrad 时使用 **DEBUG=3** 来显示底层 IR，使用 **GRAPH=1** 和 **GRAPHUOPS=1** 命令进行可视化。
- **调试与可视化 tinygrad**：一位用户建议运行 tinygrad 时使用 **DEBUG=3** 以显示底层 IR，从而理解中间语言。
   - 此外，他们提到使用 **GRAPH=1** 和 **GRAPHUOPS=1** 来生成内部运行机制的可视化表示。


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1262818557729832961)** (1 messages): 

> - `Open Interpreter`
> - `Mike Bird 演讲` 


- **Mike Bird 介绍 Open Interpreter**：Mike Bird 正在台上讨论 **Open Interpreter**。点击[此活动链接](https://discord.gg/rXdZzd5wu3?event=1260611047341953034)加入对话并就该项目提问。
- **鼓励在 Mike Bird 演讲期间提问**：鼓励参与者在 Mike Bird 关于 **Open Interpreter** 的演讲过程中随时提问。


  

---



---



---



---



---



---



{% else %}


> 为了便于邮件阅读，完整的频道分类明细已被截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}