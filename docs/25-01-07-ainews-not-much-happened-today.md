---
companies:
- nvidia
- openai
date: '2025-01-08T04:01:51.252491Z'
description: '**英伟达（NVIDIA）**推出了 **Cosmos**，这是一个基于 **2000 万小时视频**训练的开源视频世界模型，旨在推动**机器人技术**和**自动驾驶**的发展。此次发布引发了关于其开源属性和技术方案的争论。此外，**英伟达**还发布了
  **Digits**，这是一款售价 **3000 美元**的个人 AI 超级计算机，旨在推动 AI 计算的普及。AI 社区对 AI 的飞速进展表达了复杂的情绪，既有期待，也存在对
  **AGI（通用人工智能）**、岗位取代以及投资炒作的担忧。相关讨论还聚焦于即将推出的家用 AI 模型微调工具，以及用于 AI 机器人的基础模型。'
id: f0fcfec3-231c-497b-b30e-73388e597edb
models:
- cosmos
original_slug: ainews-not-much-happened-today-7007
people:
- sama
title: 今天没发生什么。
topics:
- robotics
- autonomous-driving
- open-source
- fine-tuning
- foundation-models
- memory-optimization
---

<!-- buttondown-editor-mode: plaintext -->**GB10s 可能就是你所需要的一切。**

> 2025年1月6日至1月7日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**218** 个频道，**3342** 条消息）。预计节省阅读时间（以 200wpm 计算）：**365 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

祝 2 小时 Jensen 主旨演讲日快乐。

https://www.youtube.com/watch?v=K4qQtPpSn-k

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**主题 1. NVIDIA Cosmos：革新机器人与自动驾驶系统**

- **[NVIDIA 刚刚发布了 Cosmos，这是一个基于 2000 万小时视频训练的海量开源视频世界模型！这一 AI 突破将彻底改变机器人、自动驾驶等领域。](https://v.redd.it/1j8k8iq7zibe1)** ([评分: 968, 评论: 141](https://reddit.com/r/OpenAI/comments/1hvmbcg/nvidia_just_unleashed_cosmos_a_massive_opensource/)): **NVIDIA** 发布了 **Cosmos**，这是一个基于 **2000 万小时视频**训练的开源视频世界模型。该模型预计将对 **robotics**（机器人）和 **autonomous driving**（自动驾驶）等领域产生重大影响。
  - **开源定义：** 关于 **Cosmos** 是否真正符合开源定义存在争议，一些用户指出它不符合 **OSI** 的定义，但在实际使用上非常接近 ([来源](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/))。其他人则质疑 OSI 定义开源标准的权威性。
  - **技术关注点与影响：** 用户对使用 **2000 万小时**视频训练模型以理解基础物理学的技术层面很感兴趣，质疑为什么不直接使用现有的物理模型。人们注意到了其对**制造业**和**自动驾驶**等行业的潜在影响，同时也对工作流失表示担忧。
  - **社区反应：** **Cosmos** 的发布引发了兴奋和幽默，评论集中在 AI 发展的飞速步伐以及 **NVIDIA** CEO 着装升级的象征意义上。对于此类进步的未来影响，普遍存在一种期待与幽默感。


**主题 2. 被 AI 进展淹没：在不确定性中航行**

- **还有人对最近的 AI 新闻感到不知所措吗？** ([评分: 267, 评论: 193](https://reddit.com/r/OpenAI/comments/1hvjsm2/anyone_else_feeling_overwhelmed_with_recent_ai/)): 该帖子表达了由于 **Sama** 和其他 **OpenAI 成员**等知名人物频繁讨论 **AGI、ASI 和 Singularity**（奇点）而产生的**不知所措和焦虑感**。作者作为一名机器学习工程师，对不断出现的极端变化和潜在失业的叙事感到动力不足，质疑在如此不确定的情况下如何规划未来。
  - 许多评论者将围绕 **AGI/ASI 的炒作**视为吸引投资的策略，一些人对这些进展的即时性表示怀疑。**Learninggamdev** 和 **FarTooLittleGravitas** 认为这是为了融资而制造炒作，而 **Houcemate** 指出这种炒作的真正受众是投资者，而非普通大众。
  - **BrandonLang** 和其他人建议专注于当下并控制你能控制的事情，尽管 AI 领域的发展令人应接不暇。**Denvermuffcharmer** 和 **CGeorges89** 建议暂时远离社交媒体以获得清醒的头脑，并强调变化会缓慢融合，而非一夜之间发生。
  - **Swagonflyyyy** 强调了 NVIDIA 即将发布的一款用于家庭微调模型的设备，售价为 **$3,000**，并讨论了其对 AI 开发的潜在影响。**ChymChymX** 补充说，NVIDIA 还在开发用于 AI 机器人的基础模型，展示了 AI 技术的快速进步。


---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. NVIDIA Digits：3000 美元的 AI 超级计算机可能彻底改变本地 AI**

- **[Nvidia 发布名为 Digits 的个人 AI 超级计算机，售价 3,000 美元](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai)** ([Score: 1180, Comments: 298](https://reddit.com/r/LocalLLaMA/comments/1hvj4wn/nvidia_announces_3000_personal_ai_supercomputer/))：Nvidia 推出了 **Digits**，一款售价 **3,000 美元** 的个人 AI 超级计算机。这一发布凸显了 Nvidia 致力于让个人和小型组织更容易获得先进 AI 计算能力的持续承诺。
  - **规格与性能担忧**：用户对规格感到好奇，特别是关于**内存和带宽**。文中提到了 **LPDDR5X**，并对**内存控制器**和潜在瓶颈进行了推测。一些用户预计该设备将主要用于推理而非训练，在成本和性能方面将其与配备多块 **3090/4090/5090 GPU** 的配置进行比较。
  - **市场影响与对比**：**128GB 统一内存（unified RAM）**被视为一项重大特性，可能会挑战 **Apple** 的 **LLM** 市场。用户将其与 **5090** 等其他硬件进行了比较，一些用户考虑到潜在的成本节约和性能优势，正考虑从 **Azure** 等云服务转向本地使用该设备。
  - **可用性与定价**：该设备起售价为 **3,000 美元**，预计 **5 月**上市。用户讨论了定价是否具有竞争力，有人认为 **Nvidia** 即使定价更高也依然会有需求。此外，人们对其与 **Strix Halo** 方案以及 **AMD** 潜在替代方案的对比也表现出兴趣。


- **GB10 DIGITS 将彻底改变本地 Llama** ([Score: 119, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1hvjjri/gb10_digits_will_revolutionize_local_llama/))：预计 **GB10 DIGITS** 将显著增强本地 **Llama** 应用，标志着过去两年本地模型发展的关键里程碑。这种兴奋源于 **NVIDIA Grace Blackwell** 技术的潜在普及，正如 [NVIDIA 新闻稿](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips)中所述。
  - **定价与规格担忧**：用户对 **3,000 美元的起售价**以及由于存储（而非 RAM）导致的潜在成本增加表示担忧，因为每个单元配备 **128GB 统一内存**。一些用户认为完整规格的实际成本可能会更高，并且对影响性能的**带宽能力**持怀疑态度，并将其与 **RTX5090** 等其他 GPU 进行了比较。
  - **性能与使用场景**：讨论强调 **GB10 DIGITS** 可能会因**带宽限制**而导致性能受限，从而影响其每秒生成的 **tokens per second**。虽然它可以运行大型模型，但 **Token 生成速度**可能成为瓶颈，使其与云服务或其他 GPU 相比，在高性能应用中的吸引力降低。
  - **市场定位与替代方案**：**NVIDIA** 的 **GB10** 被视为针对**专业消费者（prosumer）市场**，但关于其与 **AMD** 的 **AI Max** 或 **Intel** 和 **Apple** 未来潜在产品的价值对比仍存在争议。用户正在权衡**价格、性能和内存带宽**，一些人将其视为可行的本地 AI 解决方案，而另一些人则质疑其相对于云方案的实用性。

- **要了解 Project DIGITS 桌面端（128 GB 售价 3k），可以参考现有的 Grace CPU 系统** ([Score: 150, Comments: 73](https://reddit.com/r/LocalLLaMA/comments/1hvlbow/to_understand_the_project_digits_desktop_128_gb/))：Nvidia 的 **Project DIGITS 桌面端**推测将拥有 **128 GB 的 VRAM**，使用的是 LPDDR，相比 GPU 常用的 GDDR 和 HBM，LPDDR 更便宜且速度较慢。**Grace-Hopper Superchip (GH200)** 展示了类似的配置，拥有 480 GB 的 LPDDR 和 4.9 TB/s 的 HBM 带宽，而 **Grace CPU C1** 配置则提供 120 GB 的 LPDDR RAM 和 512 GB/s 的内存带宽。Project DIGITS 桌面端预计将达到约 **500 GB/s 的内存带宽**，在 8-bit 量化下运行 **Llama-70B** 时，可能达到约 7 tokens per second。
  - 讨论强调了 Project DIGITS 桌面端的**潜在应用场景**，特别是运行像 **Llama-70B** 这样的本地模型。一些评论者指出，由于处理速度的限制，该设备在运行超大型模型时存在局限性，而另一些人则认为它更适合推理任务而非训练，重点在于其 **500 GB/s 的内存带宽**。
  - 评论者将 Project DIGITS 桌面端与 **AMD EPYC Genoa** 系统等替代方案进行了比较，强调后者具有更高的 RAM 容量和带宽，但也指出了大型设备在物理空间和噪音方面的限制。**EPYC Genoa** 被认为是文本推理更具性价比的选择，但一些用户更看重 DIGITS 桌面端的紧凑性以及通过 **ConnectX** 进行集群化的潜力。
  - 对话还涉及了**低位宽算术 (low-bit arithmetic)** 及其对处理性能的影响，推测 DIGITS 桌面端在 4-bit 量化下运行 70B Llama 2 模型时可以达到 **≥10 tokens per second**。**ConnectX-8 互连**在增强连接性和性能方面的作用也受到了关注，为居家廉价训练方案提供了可能性。


**主题 2. 微调成功：3B 模型在 Hugging Face 训练后数学能力表现优异**

- **Hugging Face 对 Llama 3.2 3B 进行了持续预训练，在 MATH 任务上实现了 2-3 倍的提升** ([Score: 82, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1hv960u/hugging_face_continually_pretrained_llama_32_3b/))：**Hugging Face** 的 SmolLM 团队通过使用 **160B 高质量数学 token** 对 **Llama 3.2 3B 模型**进行持续预训练 (continual pre-training)，在 MATH 任务上实现了 **2-3 倍的提升**。这一增强使得模型在 **GSM8K** 上的得分提高了 2 倍，在 **MATH** 上提高了 3 倍，同时在 **MMLU-Pro** 上的性能下降极小，在 **HellaSwag** 上则没有下降。更多详情请访问其 [model](https://huggingface.co/HuggingFaceTB/FineMath-Llama-3B)、[dataset](https://huggingface.co/datasets/HuggingFaceTB/finemath) 和 [training script](https://github.com/huggingface/smollm/tree/main/pre-training/continual-pretraining)。
  - **持续预训练 (Continual Pre-Training)** 涉及使用额外数据延长模型的预训练阶段，正如 **mpasila** 所解释的。这与微调的不同之处在于使用了更大的数据集，在本例中，是在 **Llama 3** 现有的 **15 trillion** token 基础上增加了 **160 billion** token。
  - 正如 **Secure_Reflection409** 所指出并由 **r0kh0rd** 澄清的那样，该模型在 **MMLU-Pro** 上的表现并未提高，这强调了该训练是无标签的无监督训练。
  - **EstarriolOfTheEast** 对该模型在数学任务之外的实际应用表示担忧，质疑其在指令遵循 (instruction-following) 场景中的有效性，**DinoAmino** 确认这并非本次训练的重点，因为该模型未经指令微调 (instruction-tuned)。

- **[Llama 3b - 仅通过在高质量 160B tokens 上持续训练，即可将数学能力提升 2-3 倍](https://i.redd.it/t3kjugswufbe1.jpeg)** ([Score: 230, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1hv9w65/llama_3b_you_can_23x_the_math_capabilities_just/))：在高质量的 **1600 亿 tokens** 上对 **Llama 3.2-3B 模型**进行持续预训练，可以在不影响其他指标的情况下，将其数学能力显著提高 2-3 倍。性能提升通过具体数值量化：如柱状图所示，**GSM8K** 提升了 **+20.6%**，**MATH** 提升了 **+17.2%**。
  - **机器学习中的 Grokking**：在此背景下，人们对 Grokking 现象的发生持怀疑态度，因为它涉及神经网络最初过拟合，然后在许多个 epochs 后突然泛化良好。有人指出，故意让表现良好的模型过拟合可能不会导致更好的泛化，而对大型数学数据集进行持续预训练（continued pre-training）预计会提高小模型的性能。
  - **训练数据与 Epochs**：在相同数据上进行多个 epochs 的训练可以产生良好的结果，在性能下降前进行 10 倍 epochs 是有效的，而 20-40 倍可能会导致数据“烧毁”（burning the data）。有人对 **GSM8K** 或 **MATH** 的数据泄露到训练数据集中表示担忧，并引用了 [Hugging Face](https://huggingface.co/datasets/HuggingFaceTB/finemath) 上的污染报告和数据集来源。
  - **资源与过拟合担忧**：一些用户认为 **1600 亿 tokens** 可能过多，但评论建议现阶段无需担心过拟合。与微调（fine-tuning）相比，预训练（pretraining）需要大量的 VRAM，且该方法被辩护为不会损害其他指标。


**主题 3. RTX 5090 用于 AI 的批评：平衡 VRAM 与性能**

- **[RTX 5000 系列官方规格](https://i.redd.it/j0q0nd42rhbe1.png)** ([Score: 149, Comments: 62](https://reddit.com/r/LocalLLaMA/comments/1hvi9mi/rtx_5000_series_official_specs/))：将 **RTX 5000 系列**显卡（包括 **RTX 5090, RTX 5080, RTX 5070 Ti 和 RTX 5070**）的官方规格与 **RTX 4090** 模型进行了对比。重点展示的关键特性包括 **NVIDIA Architecture, DLSS 版本, AI TOPS, Tensor Cores, Ray Tracing Cores** 以及 **Memory Configuration**。
  - 多位评论者对新 **RTX 5000 系列**的 **VRAM 容量**表示不满，指出 **32GB** 不足以运行更大的 AI 模型。呼吁增加 VRAM 以支持更苛刻的任务，有人建议 **24GB 和 32GB** 的配置对于 **RTX 5070** 系列会更合适。
  - **NVIDIA** 的营销策略受到批评，主要担忧在于 **core counts** 和 **AI TOPS** 性能指标缺乏透明度。一些人认为这些规格是为游戏玩家量身定制的，而非针对本地 AI 模型部署感兴趣的用户，而另一些人则提到传达全面性能基准测试的难度。
  - 讨论强调了 **NVIDIA** 的 **CUDA** 在 AI 行业的主导地位，而 **ROCm** 被认为是一个不太可行的替代方案，尤其是在 Windows 上。提到了 **Intel 的 AI playground** 实现了 **ComfyUI** 和 **Llama.cpp**，为 Linux 用户提供了一个潜在的替代方案。

- **[NVIDIA compares FP8 on 4090 to FP4 on 5090. Seems a little misleading](https://i.redd.it/aj6qbvpl4ibe1.jpeg)** ([Score: 340, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1hvjnar/nvidia_compares_fp8_on_4090_to_fp4_on_5090_seems/)): **NVIDIA** 因将 **RTX 4090** 上的 **FP8** 性能与 **RTX 5090** 上的 **FP4** 进行对比而面临批评，一些人认为这具有误导性。该对比通过一张显示多款游戏性能的柱状图呈现，相关指标暗示测试设置和所用硬件可能存在偏差。
  - 讨论强调了 **NVIDIA** 性能对比的误导性，特别是 **RTX 4090** 使用 **FP8** 对比 **RTX 5090** 使用 **FP4**。批评者认为，性能提升很大程度上归功于 **Multi-Frame Gen** 等软件增强功能，这些功能在没有显著硬件改进的情况下人为地拔高了性能指标。
  - 几位评论者指出了这种令人质疑的营销策略，指出 **FP4** 相比 **FP8** 牺牲了质量，且 **NVIDIA** 有夸大性能指标的历史。此外，**NVIDIA** 的营销图表也因不一致和潜在的疏忽而受到批评，例如字体差异以及在 **AI TOPS** 和 **TFLOPS** 数据方面缺乏透明度。
  - 人们对实际的算力提升持怀疑态度，一些人认为 **RTX 4090** 可能有意限制了核心数，以便为 **Ti** 版本留出空间。与以往 **NVIDIA** 发布的产品相比，性能跨度可能并不像广告宣传的那样实质性，一些用户建议等待当前型号降价。


**Theme 4. NVIDIA & AMD in THE AI Tech Race: Digits vs Strix Halo**

- **[HP Z2 Mini G1a is a workstation-class mini PC with AMD Strix Halo and up to 96GB graphics memory](https://liliputing.com/hp-z2-mini-g1a-is-a-workstation-class-mini-pc-with-amd-strix-halo-and-up-to-96gb-graphics-memory/)** ([Score: 83, Comments: 45](https://reddit.com/r/LocalLLaMA/comments/1hvqydy/hp_z2_mini_g1a_is_a_workstationclass_mini_pc_with/)): **HP** 推出了 **Z2 Mini G1a**，这是一款工作站级迷你 PC，搭载 **AMD Strix Halo**，拥有高达 **96GB 显存**，将其定位为 **NVIDIA** 新产品的竞争对手。
  - 搭载 **AMD Strix Halo** 的 **HP Z2 Mini G1a** 以其 **256GB/s 内存带宽**而备受关注，使用了 4 通道的 **LPDDR5x-8000**。这种配置支持多个较小模型或单个高达 **70B 参数**的大模型。然而，其 **50 TOPS** 的 **NPU** 性能与 **RTX 4090**（**1300 TOPS**）等高端 GPU 相比仍然有限。
  - 讨论突出了 AMD 传统的分段模型与 Apple 统一内存架构（Unified Memory Architecture）之间的差异。尽管 AMD 的 **96GB 显存**分配提供了灵活性，但它缺乏 Apple 系统中那种完全集成的访问方式，这可能会影响性能效率。
  - **Z2 Mini G1a** 起售价为 **1200 美元**，为本地 AI 工作站提供了一个具有竞争力的选择。它适用于较小的量化模型和开发工作，但在大型模型推理方面可能无法与高端独立 GPU 的性能相匹配。未来 **ROCm/DirectML** 支持 **NPU** 加速的潜力可能会增强其能力。


- **[I made a CLI for improving prompts using a genetic algorithm](https://i.redd.it/p8q191zp2gbe1.gif)** ([Score: 97, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hvayr2/i_made_a_cli_for_improving_prompts_using_a/)): 该帖子介绍了一个为使用**遗传算法**增强提示词（Prompts）而开发的 **CLI 工具**。随附的 GIF 展示了该工具在 **MacBook Pro** 终端上的运行情况，强调了其命令行界面功能。
  - **Promptimal** 工具通过使用**自我评估循环**或自定义评估器，在不需要数据集的情况下优化提示词。它采用**遗传算法**迭代组合成功的提示词，并完全在终端中运行，使其在实验中非常易于使用且触手可及。
  - 开发者正在考虑改进，目前正致力于添加 **ollama 支持**，以实现本地模型的集成。由于该工具仍处于实验阶段，鼓励用户提供反馈。
  - **FullstackSensei** 建议探索 **蒙特卡洛树搜索 (MCTS)** 等替代方案来取代遗传算法，并提到 **optillm** 等工具作为一个潜在选择。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. NVIDIA Cosmos：彻底改变机器人技术与自动驾驶系统**

- **[NVIDIA 刚刚发布了 Cosmos，这是一个基于 2000 万小时视频训练的大型开源视频世界模型！这一 AI 领域的突破将彻底改变机器人、自动驾驶等领域。](https://v.redd.it/1j8k8iq7zibe1)** ([Score: 968, Comments: 141](https://reddit.com/r/OpenAI/comments/1hvmbcg/nvidia_just_unleashed_cosmos_a_massive_opensource/)): **NVIDIA** 发布了 **Cosmos**，这是一个基于 **2000 万小时视频**训练的开源视频世界模型。该模型预计将对 **robotics**（机器人技术）和 **autonomous driving**（自动驾驶）等领域产生重大影响。
  - **开源定义：** 关于 **Cosmos** 是否真正符合开源定义存在争议，一些用户指出它不符合 **OSI** 的定义，但在实际应用中非常相似（[来源](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)）。其他人则质疑 OSI 定义开源标准的权威性。
  - **技术关注点与影响：** 用户对通过 **2000 万小时**视频训练模型来理解基础物理这一技术层面非常感兴趣，并质疑为什么不直接使用现有的物理模型。人们注意到其对 **manufacturing**（制造业）和 **autonomous driving** 等行业的潜在影响，同时也对职业取代表示担忧。
  - **社区反应：** **Cosmos** 的发布引发了兴奋和幽默，评论涉及 AI 发展的飞速步伐以及 **NVIDIA** CEO 着装升级的象征意义。对于此类进步的未来影响，普遍存在一种期待和幽默感。


**主题 2. 被 AI 进展淹没：在不确定性中航行**

- **还有人对最近的 AI 新闻感到不知所措吗？** ([Score: 267, Comments: 193](https://reddit.com/r/OpenAI/comments/1hvjsm2/anyone_else_feeling_overwhelmed_with_recent_ai/)): 该帖子表达了由于 **Sama** 和其他 **OpenAI** 成员频繁讨论 **AGI**、**ASI** 和 **Singularity** 而产生的**压倒感和焦虑**。作者是一名 **machine learning engineer**，对即将到来的极端变化和潜在失业的持续叙事感到动力不足，质疑在如此不确定的情况下如何规划未来。
  - 许多评论者将围绕 **AGI/ASI 的炒作**视为吸引投资的策略，一些人对这些进展的紧迫性表示怀疑。**Learninggamdev** 和 **FarTooLittleGravitas** 认为这是为了融资而制造炒作，而 **Houcemate** 指出这种炒作的真正受众是投资者，而非普通大众。
  - **BrandonLang** 等人建议专注于当下并控制你能控制的事情，尽管 AI 领域的现状令人应接不暇。**Denvermuffcharmer** 和 **CGeorges89** 建议暂时远离社交媒体以理清思路，并强调变化将缓慢融合，而非一夜之间发生。
  - **Swagonflyyyy** 强调了 **NVIDIA** 即将发布的一款用于在家 **fine-tuning**（微调）模型的设备，售价 **$3,000**，并讨论了其对 AI 发展的潜在影响。**ChymChymX** 补充说，**NVIDIA** 还在开发一个用于 AI 机器人的 **foundation model**（基础模型），展示了 AI 技术的飞速进步。


---

# AI Discord Recap

> 由 o1-2024-12-17 生成的摘要之摘要

**主题 1. GPU 热潮与基础设施**

- [**NVIDIA 的 ‘DIGITS’ 将 HPC 带到你的桌面**](https://www.nvidia.com/en-us/project-digits/)：NVIDIA 发布了一款售价 3,000 美元的 AI 超级计算机，搭载全新的 **Grace Blackwell** Superchip，声称可以在紧凑的桌面箱体中处理 200B 参数的模型。早期采用者对实际基准测试表示怀疑，并指出了 [The Verge 的文章](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai) 等相关报道。
- **AMD vs NVIDIA VRAM 之争**：工程师们讨论了 AMD 的 VRAM 余量与 RTX 4090 在运行大型本地 LLM 时约 95% 的 GPU 利用率。一些人推测 RTX 5070 将以 549 美元的价格提供“4090 级别的性能”，但对 NVIDIA 大胆的营销手段表示怀疑。
- **投机采样 (Speculative Decoding) 竞速前进**：[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp) 及其他工具的最新更新承诺通过草拟部分输出来将 LLM 推理速度提高 25–60%。早期测试表明准确率损失极小，这激励了开发者在跨平台应用中采用该方法。

**主题 2. 微调与 LoRA 历险记**

- [**LoRA 合并处理大型 Tokenizer**](https://github.com/unslothai/unsloth/issues/1518)：用户在使用 Unsloth 的 LoRA 进行微调后发现 Tokenizer 文件变大，并指出需要额外的 JSON 文件才能正确使用。建议将 QLoRA 合并回 FP16 的基础模型，以避免性能下降。
- **Deepspeed Zero-3 令人失望**：一些人在冻结参数训练 7B 模型时发现没有内存收益，怀疑是非检查点梯度 (non-checkpointed gradients) 带来的开销。讨论强调 *“被忽视的优化器状态”* 阻碍了多 GPU 的扩展。
- [**词汇还是概念？**](https://arxiv.org/pdf/2412.08821)：激烈的辩论推动了“本体嵌入 (ontological embeddings)”优于普通 Token 片段的观点，声称其具有更深层的语义向量含义。支持者希望从基于分块 (chunk-based) 的嵌入转向基于概念的语义表示。

**主题 3. 工具、Function Calling 与 Agent**

- [**LM Studio 0.3.6 发布 Function Calling**](https://lmstudio.ai/download)：Beta 版 API 支持本地 Qwen2VL 和 QVQ 视觉模型以及应用内更新。用户赞扬了 Windows 安装程序新增的驱动器选择功能，并分享了一个 [Qwen2VL 演示](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4)。
- [**Codeium vs DeepSeek 企业级对比**](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev)：一些人吹捧 DeepSeek v3 在解决数据问题后的强劲输出，而 Codeium 在稳定的企业需求方面仍然很受欢迎。辩论围绕协同效应与许可难题展开，并对各平台如何使用训练数据表示担忧。
- [**多 Agent 工作流势头强劲**](https://t.co/eS4BhuAZVS)：从 NVIDIA 的多 Agent 蓝图到使用多个 LLM 的社区解决方案，开发者正在自动化博客研究和写作任务。早期采用者对跨 Agent 的协同效应表示赞赏，但要求在错误处理和并发性方面有更高的透明度。

**主题 4. 支付与隐私风波**

- [**AI21 Labs 代币引发诈骗恐慌**](https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H)：社区成员将“AI21 Labs Token”标记为跑路 (rug-pull) 骗局；AI21 公开否认与其有关联。尽管据称通过了审计，但该项目可疑的持有人模式吓坏了用户，促使他们要求官方在 Twitter 上发表声明。
- **OpenRouter 支付网关失效**：虚拟卡反复被拒，迫使人们建议使用加密货币支付和替代计费方式。[Issue #1157](https://github.com/cline/cline/issues/1157) 记录了相关的停机时间，一些人怀疑是资源过载所致。
- **Perplexity 酝酿隐私担忧**：在进行健康相关查询后出现的定向广告让用户对数据共享感到警觉。他们转向 [Trust Center](https://trust.perplexity.ai/) 查看 SOC 2 合规详情，但仍对潜在的用户追踪感到不安。

**主题 5. MLOps、LLM 安全及未来展望**

- [**MLOps 与 Feature Stores 网络研讨会**](https://buff.ly/4j9oiVg)：Ben Epstein 和 Simba Khadder 将于太平洋时间 1 月 15 日上午 8 点聚焦 2025 年 MLOps 趋势，涵盖数据流水线的最佳实践。他们承诺进行关于实际扩展的问答，敦促 ML 专业人士紧跟 LLMOps 的进展。
- [**GraySwanAI 的有害 AI 助手挑战赛**](https://x.com/GraySwanAI/status/1872720375328411668)：将于东部时间 1 月 4 日下午 1 点启动，为创意提示词注入 (prompt injections) 提供 4 万美元奖金。多轮输入是允许的，这激发了揭露不安全 LLM 行为的竞争。
- [**Cerebras 征集大胆的 AI 提案**](https://cerebras.ai/blog/grantfrp)：他们邀请研究人员利用其 Wafer Scale Engine 推动生成式 AI 的前沿。参与者可以利用硬件资助在大规模环境下探索新的训练和推理技术。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 故障排除与 Tokenizer 难题**：在最近的提交后，用户遇到了 **Unsloth** 的 GPU 特定错误，参考了 [GitHub Issue #1518](https://github.com/unslothai/unsloth/issues/1518)，并澄清了 LoRA 微调产生较大的 Tokenizer 文件是正常现象。
   - 成员们建议降级或更新特定的库版本，并强调新生成的 **added_tokens.json** 必须保持完整才能正常使用。
- **LoRA 合并与多数据集魔法**：社区成员强调在 Ollama 中应使用 FP16 将 LoRA 与基础模型合并，并指出了关于多数据集训练的 [此 Google Colab 教程](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t)。
   - 他们建议保持一致的数据格式以避免训练事故，并警告说忽略正确的合并步骤可能会损害性能。
- **硬件折腾 vs 云端便利**：工程师们权衡了在本地使用四个 48GB DIMM 与云端解决方案的利弊，引用了 [Unsloth AI 的推文](https://x.com/UnslothAI/status/1876729710790815872)，该推文提到 2-bit 量化需要 48GB RAM 加 250GB 磁盘空间。
   - 他们承认在云端花费了大量时间进行上传/下载循环，但赞赏运行更大模型时的可扩展选项。
- **Gemini 1207 的陈旧知识与 Picotron 咨询**：一些人对 **Gemini 1207** 过时的知识截止日期表示沮丧，这限制了它对现代库的帮助。
   - 其他人询问了用于微调的 **Picotron 代码库**，寻求用户对其在现实世界中功效的经验。
- **Tokens vs 概念：本体嵌入（Ontological Embedding）的推动**：一场激烈的交流剖析了词片段嵌入（word-fragment embeddings）的局限性，并提议使用**本体“概念”**来获得更密集的语义向量，参考了 [这篇论文](https://arxiv.org/pdf/2412.08821)。
   - 支持者声称这些概念嵌入可以提供更深层的含义，挑战了通常对基于 Token 方法的依赖。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.6 推出工具与视觉模型**：LM Studio 发布了 [0.3.6 版本](https://lmstudio.ai/download)，其特点是包含测试版的 **Function Calling API**，并支持 **Qwen2VL** 和 **QVQ** 的本地推理，同时增加了新的 Windows 安装程序选项。
   - 该更新增加了从 0.3.5 开始的**应用内更新**功能，并展示了一个 [Qwen2VL 演示](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4)，赢得了早期测试者的赞誉。
- **投机采样（Speculative Decoding）加速 LLM**：在 [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp) 中推动 **Speculative Decoding** 的应用，建议在不损害准确性的情况下将解析速度提高多达 60%。
   - 贡献者引用了[研究](https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120)解释草稿模型（draft models）如何提高吞吐量，引发了对跨平台推出的热情。
- **NVIDIA Project DIGITS 目标指向 200B 模型加载**：**NVIDIA** 披露了 **Project DIGITS**，这是一个具有 128GB 一致性内存（coherent memory）的紧凑型 AI 系统，声称能够处理 200B 参数模型。
   - 开发者们钦佩这一概念，但指出实际成本和基准测试数据仍是未知数，尽管 [NVIDIA 官网](https://www.nvidia.com/en-us/project-digits/) 宣称其具有更快的开发周期。
- **AMD vs NVIDIA GPU 对决**：一场激烈的比较权衡了 AMD 的显存（VRAM）余量与 **RTX 4090** 在 95% GPU 占用率下运行 **Qwen2.5-Coder-32B-Instruct** 达到约 31 tokens/s 的表现。
   - 参与者推测了即将推出的 GeForce **50** 系列，一些人建议使用两家厂商的多 GPU 设置来满足本地 LLM 的需求。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek 与 Codeium 的对决**：成员们将 **DeepSeek v3** 与 **Codeium** 的企业友好型方案进行了对比，指出一旦数据问题得到解决且许可问题得到落实，**DeepSeek** 可能会成为明显的赢家。一些参与者提到了这些工具包之间**潜在的协同效应**，但也表达了对平衡模型性能与**企业需求**的担忧。
   - 几位成员强调了 *DeepSeek v3* 强大的 **AI 输出**能力，并对 **Codeium** 如何获取或管理其训练数据提出了疑问，引发了热烈讨论。其他人则认为 **Codeium** 在稳定的企业级集成方面仍然脱颖而出，而怀疑论者则坚持认为，解决 *DeepSeek* 的数据流水线问题仍是关键的转折点。
- **Cline 扩展插件登陆 VS Marketplace**：一个名为 **Cline (原 Claude Dev)** 的新成员出现在 [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev) 上，提供了一个集成在 IDE 中的自主编码 **Agent**。它因支持在单一流程中实现**文件创建**、**编辑**和**命令执行**而引起了关注。
   - 用户赞扬了这种**全能型 (all-in-one)** 方法的便利性，称其为*“快速原型开发的顺畅体验”*。与此同时，一些人希望看到更多关于该 **Agent** 性能的基准测试，并指出以 AI 为中心的开发者对高级编码助手的兴趣持续增长。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **NVIDIA 灵动“Digits”首次亮相**：NVIDIA 推出了 [Project DIGITS](https://www.nvidia.com/en-us/project-digits/)，这是一款售价 3,000 美元的个人 AI 超级计算机，搭载 **GB10 Grace Blackwell** 超级芯片，能够训练参数量高达 **2000 亿**的模型。
   - 它的性能超越了现有的高端 GPU，旨在用于本地模型原型设计，正如 [The Verge 的报道](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai)所描述的那样，社区反馈称赞其在高级 AI 任务中的实用性。
- **Stable Diffusion 的商业条款**：根据 [Stability AI License](https://stability.ai/license) 的规定，Stability AI 允许年收入低于 **100 万美元**的用户商业化使用其 **Stable Diffusion** 模型。
   - 贡献者们对许可证的具体细节表示困惑，但官方的 [Stability AI Core Models](https://stability-ai.squarespace.com/core-models) 文档澄清了关于衍生作品的条款。
- **图像生成的速度与精细度之争**：社区成员将 **Stable Diffusion 3.5** 与 **Flux** 进行了对比，发现 3.5 运行速度更快，但 **Flux** 的输出更精细。
   - 一些人建议使用 3.5 进行原型设计，然后切换到 **Flux** 进行最终润色，并称赞了这两种方法的协同作用。
- **CFG 特性导致 Flux 变慢**：在 **Flux** 中调高 **CFG scale** 会显著增加处理时间，这引发了在调整 Prompt 过程中效率低下的担忧。
   - 参与者推测 **Flux** 可能针对去噪而非直接的 Prompt 扩展进行了优化，强调了速度与质量之间的权衡。
- **用于物理 AI 的 NVIDIA Cosmos**：[NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) 平台支持世界基础模型、Tokenizer 以及为 **Robotics** 和 **AV labs** 提供的视频流水线。
   - 它同时包含扩散模型和自回归模型，早期采用者报告其结果与成熟系统不相上下。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 导出提升工作流**：成员们发现了如何在每次迭代后**导出 Bolt 项目**，并将其无缝集成到其他 IDE 中。
   - 他们参考了一个 [Vite + React + TS 示例](https://stellular-beijinho-102b6b.netlify.app/)，并建议使用 `bolt.new/github.com/githubUsername/repoName` 进行**手动 GitHub 上传**。
- **外部 LLM 消耗大量 Token**：用户报告称，在小型项目中，单个 Prompt 就消耗了 **150 万个 Token**，引发了对成本失控的担忧。
   - 他们怀疑是代码效率低下，并建议将调试工作外包给外部 LLM 以减少开销。
- **Supabase 聊天应用实时功能失效**：一些使用 **Supabase** 构建聊天应用的开发者无法实时看到新消息。
   - 他们发现通过通知传递消息可能会修复 UI 缺陷，并澄清后端功能并无故障。
- **Bolt 与 GitHub 在更新上发生冲突**：一位用户在将 GitHub 部署到 [Render.com](https://render.com/) 时遇到了问题，被迫对基于 Bolt 的项目进行本地修复。
   - 他们参考了 [Issue #5108](https://github.com/stackblitz/bolt.new/issues/5108) 以进行**后端服务器集成**，暗示即将推出解决方案。
- **移动框架与预览故障**：一个使用 **NativeScript + Vue** 构建的音板项目触发了 npm 命令错误，促使人们提出替代框架建议。
   - 另一位用户在新笔记本电脑上的 Bolt 中遇到白屏，暗示直接使用 GitHub 与项目链接可能是原因。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的组合功能延迟**：成员们报告称 **Cursor IDE** 变慢并频繁报错，特别是在 **Composer Agent** 尝试处理大型代码库时。
   - 他们描述了代码消失、间距异常和链接无响应等问题，警告他人在等待改进期间做好备份。
- **关于代码块模块化的思考**：一些参与者建议将项目拆分为 100 行的文件，以帮助 AI 工具更可预测地跟踪更改。
   - 其他人则反驳说，处理许多小文件会使文件查找变得复杂，在多文件编辑期间造成混乱。
- **“Project Brain”扩展引发关注**：一位用户分享了一个 [Reddit 链接](https://www.reddit.com/r/cursor/s/2BGt4BZ21e)，介绍了一个旨在让 AI 更好地理解文件关系的扩展。
   - 他们希望这能通过提供依赖关系的鸟瞰图来减少混乱，从而可能改进 AI 驱动的重构。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI Agent 处于注入风险边缘**：传闻称 **OpenAI** 因担心 **Prompt Injection** 而推迟了 Agent 的部署，并有传言称企业版方案接近 **$2,000**。
   - 社区中的许多人认为这是在推动更好的支持，暗示 **Agents** 可能很快就会亮相 [更多信息](https://www.theinformation.com/articles/why-openai-is-taking-so-long-to-launch-agents)。
- **零一万物（01.AI）反驳传闻**：来自 **01.AI** 的**李开复**反驳了有关该初创公司将团队出售给阿里巴巴的传言，理由是 2024 年收入强劲，超过 **1400 万美元** [来源](https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/)。
   - 然而，据报道该公司裁减了核心预训练团队，导致许多人质疑他们将如何平衡未来的增长。
- **Anthropic 的巨额融资行动**：**Anthropic** 以高达 **600 亿美元** 的估值获得了 **20 亿美元** 融资，预期 ARR 为 **8.75 亿美元**。
   - 这一大胆举措突显了激烈的 **B2B** 竞争，观察者们正在评估他们能以多快的速度扩张。
- **Nvidia Digits 桌面端亮相**：**Nvidia** 在 CES 上发布了售价 **$3,000** 的 **Project Digits**，搭载 **Grace Blackwell** 超级芯片，可处理高达 **2000 亿** 参数的模型 [链接](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai)。
   - 工程师们对 **ARM CPU** 的兼容性表示担忧，因为开源支持有限。
- **MeCo 方法展现元数据魔力**：[这篇论文](https://arxiv.org/abs/2501.01)中概述的 **MeCo** 方法将来源 URL 预置到训练文档中，以简化 **LM 预训练**。
   - 批评者最初称其“荒谬”，但他们承认元数据可以增强模型的上下文深度。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepSpeed 的困境：内存收益消失**：一位用户尝试使用 **DeepSpeed Zero-3** 来削减 7B LLM 训练期间的内存占用，但发现没有明显收益，怀疑是由于缺少 **gradient checkpointing**（梯度检查点）导致的开销。
   - 社区成员得出结论，*被忽视的优化器状态 (optimizer states)* 加上高精度副本阻碍了内存优化，这引发了对 **gradient checkpointing** 更多的关注。
- **Pythia 的伦理检查：它能胜任吗？**：围绕在 [Ethics 数据集](https://huggingface.co/datasets/hendrycks/ethics)上评估 **Pythia** 的讨论非常热烈，揭示了对测试道德复杂性的推动。
   - 许多人对 **Pythia** 的表现以及这些任务如何塑造未来的模型对齐 (alignment) 工作表示好奇。
- **Cerebras 征集创意 AI**：**Cerebras** 发布了一份 [提案征集 (Request for Proposals)](https://cerebras.ai/blog/grantfrp)，旨在通过其 Wafer Scale Engine 加速 **生成式 AI 研究 (Generative AI research)**，寻求大胆的方案提交。
   - 他们旨在展示其硬件的性能优势，并激励 **推理和训练 (inference and training)** 的新颖方法。
- **闲聊格式在多选题 (MCQs) 上表现不佳**：使用 **chat templates**（对话模板）进行的试验显示多选题得分下降，而 **L3 8B base** 模型在纯文本格式下表现更好。
   - Logprob 分析表明，对话框架会阻碍精确的仅字母回答，从而引发了对受限输出样式的需求。
- **Llama2 在 GPT-NeoX 中的命运：止步于此？**：**Llama2** checkpoint 用户询问 NeoX 训练的权重是否能顺利转换为 Hugging Face 格式，但未得到明确确认。
   - 不同的优化器设置（AdamW 与 **Lion**）以及 BF16 缩放的复杂性，增加了直接 checkpoint 移植的不确定性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 支付困境**：用户报告了 **OpenRouter** 支付网关反复被拒和故障的问题，引发了对虚拟卡的讨论。
   - 一些人建议转向 **crypto**（加密货币）交易，特别是寻求适合全球使用的用户友好型钱包。
- **Hermes 405b 的故障与停滞**：尽管状态指示灯仍显示绿色，但 **Lambda 的 Hermes 405b** 频繁崩溃。
   - 高需求导致参与者怀疑存在资源压力，一些人指出 **DeepSeek V3** 是另一个表现滞后的服务。
- **DeepSeek V3 宕机问题频发**：多位用户反映了 **DeepSeek V3** 的可靠性问题，尤其是在处理大输入时。
   - 他们引用了 [Issue #1157](https://github.com/cline/cline/issues/1157) 作为诊断无限加载故障的证据。
- **加密货币方案获得支持**：提供 **crypto** 替代方案的呼声越来越高，用户指出这在菲律宾等某些地区更加方便。
   - 他们提到 **Trust Wallet** 和类似平台是可能的解决方案，理由是交易失败率较低。
- **LLM 游戏开发触及天花板**：用户认识到像 O3 和 GPT-5 这样的 **LLM** 可以处理简单的 2D 游戏，但更复杂的设计仍然难以实现。
   - 他们一致认为，先进的组织逻辑阻碍了全自动复杂 **游戏开发 (game development)**，尤其是对于大型项目。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 作为专业级编程助手的实用性**：多位成员称赞 **Aider** 处理复杂代码任务的能力，并参考了 [图像和网页使用文档](https://aider.chat/docs/usage/images-urls.html#web-pages) 进行高级项目集成。
   - 他们将其比作 **编程导师 (coding mentor)**，强调了战略性提示词 (prompts) 和 /ask 命令如何细化结果以获得更准确的输出。
- **Continue.dev 与 Aider 协同编程**：一些成员在 **Aider** 的基础上测试了 **Continue.dev**，发现它们在快速迭代和更好的任务管理方面具有互补性。
   - 他们分享道，结合这两个工具可以减轻繁重的编码工作量，并使开发更有条理，并计划扩展以统一它们的工作流程。
- **在 Aider 中玩转自定义 LLM**：开发者探索了通过 'custom/' 名称前缀和 [高级模型设置](https://aider.chat/docs/config/adv-model-settings.html) 连接自定义语言模型，从而实现专门的 ML 流水线。
   - 他们报告称，通过正确注册模型类并调整 API 参数以匹配其设置，集成过程更加顺畅。
- **利用 LLM 访谈生成结构化规范**：一种共享的方法是，在编码前使用 **LLM** 访谈用户以创建规范，如 [YouTube 视频](https://www.youtube.com/watch?v=XWJGm3y207A) 所示。
   - 这种策略确保了更有条理的规划，直接为 **Aider** 的编码提示词提供更清晰的信息。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **AI 体育播报：赛事回顾的“大满贯”**：一位用户展示了 NotebookLM 如何将体育回顾与精彩片段叠加，并引用了针对 NBA 和 NFL 的[这段演示](https://youtu.be/D7qZ2VphetU)。
   - 他们赞扬了该方法的**成本效益**，指出实时报道和品牌内容可以实现大规模自动化。
- **单一来源辩论中的引用难题**：成员们辩论了**大英百科全书 (Britannica)** 与**维基百科 (Wikipedia)** 的可靠性，重点在于引用多个来源还是依赖单一来源。
   - 他们寻求一种强大的系统提示词 (system prompt) 策略，以保持事实准确性并确保 AI 生成材料中的精确引用。
- **合同审查获得 AI 助力**：用户探索了将 AI 用于**合同修订 (contract redlining)**，强调了在繁琐的法律编辑中提高速度并降低成本。
   - 他们强调了虚拟法律助理与基于虚拟形象 (avatar) 协作的潜在整合，从而在谈判过程中更好地协调利益相关者的参与。
- **高强度使用下 NotebookLM 变慢**：用户对每日使用上限表示担忧，NotebookLM 在长时间使用后会变慢，并引导参考 [支持页面](https://support.google.com/notebooklm/answer/15678219?hl=en)。
   - 一些用户还在**音频概览 (audio overview)** 的长度管理上遇到困难，并注意到缺少问题建议功能，寻求关于当前产品更新的说明。
- **许可证咨询中 NotebookLM Plus 功能脱颖而出**：订阅者称赞 **NotebookLM Plus** 支持多个 PDF 和 YouTube 链接，能生成更精炼的摘要并扩大了使用配额。
   - Google Workspace 许可证要求成为热议话题，促使用户咨询 [管理员帮助页面](https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness) 以获取插件详情。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 结束 Forge API 测试**：[Nous Forge API](https://forge.nousresearch.com/) 的 Beta 测试于近期结束，该 API 支持在 Hermes、Claude、Gemini 和 OpenAI 等多个模型上进行高级推理。潜在订阅者仍可关注更新，以获取明确使用和性能细节的新配置。
   - 针对可能显得**利润导向**的用户订阅模式出现了辩论，加剧了对机构如何对待用户信任的审查。
- **NVIDIA Digits 取得进展**：新的 [NVIDIA Project DIGITS](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips) 推出了 Grace Blackwell 超级芯片，用于更广泛的高性能 AI 计算。与此同时，关于 5070 传闻中以 549 美元实现“4090 级性能”的争论也异常激烈。
   - 怀疑者质疑 NVIDIA 的营销是否符合实际基准测试，并引用了[指出夸大宣传的推文](https://x.com/meeix_/status/1876465349278994804)。其他人则希望 DIGITS 能降低顶级 AI 硬件的门槛。
- **调整对话：AI 行为提升**：一些成员分享了系统提示词，以减少模型响应中的焦虑或不确定感，从而建议更自信的生成输出。人们开玩笑说 AI 日志中会出现意外的“表白”，这是微调策略不完善的副作用。
   - USB-C 作为一种具有成本意识的 10-20Gbps 网络连接方式备受关注，尽管小组警告了线缆兼容性和在大规模使用中的潜在限制。
- **隐私与利润的对决**：一位用户指出，某些 AI 机构在保护**隐私**方面缺乏声誉，引发了对企业意图的怀疑。这引发了关于**利润动机**是否必然掩盖用户保护措施的讨论。
   - 其他人声称利润至上的思维会滋生不信任，并提供了为了实现收入目标而在安全上走捷径的警示案例。
- **MiniMind 与神经嵌入的魔力**：一篇博客文章探讨了潜空间几何，引用了[流形假设 (Manifold Hypothesis)](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) 和神经网络中的分层特征。进一步阅读包括来自 [Colah 的深度学习系列](https://colah.github.io/posts/2015-01-Visualizing-Representations/) 的可视化内容，以阐明隐藏表示。
   - [MiniMind 项目](https://github.com/jingyaogong/minimind/blob/master/README_en.md) 展示了一个拥有 26.88M 参数的 LLM，可以在 2 张 RTX3090 上在几小时内完成预训练、SFT 和 DPO。爱好者们因其易于获取的代码、快速的训练以及向混合专家 (MoE) 和多模态模型的扩展而欢迎它。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 的困扰与模型混乱**：多位用户反映 **Perplexity** 响应速度慢且 **Pro Searches** 配额冲突，导致一些人依靠复制粘贴技巧来获得更顺畅的查询体验。
   - 他们还讨论了 12 月 19 日的一封邮件，暗示“如果他们只保留在线模型，那就太糟糕了！”，这表明了对潜在模型独占性的担忧。
- **隐私风险与 SOC 2 压力**：用户对在 **Perplexity** 进行健康相关搜索后出现的针对性广告表示警惕，质疑用户数据可能被如何共享和存储。
   - 一些人转向 [Trust Center | Powered by Drata](https://trust.perplexity.ai/) 获取 **SOC 2 compliance** 信息，但仍对隐私保护感到不确定。
- **NASA 的灵巧月球微型任务**：今天，**NASA** 展示了其旨在完善月球探测的 **Moon Micro-Mission**，详情见[此处](https://www.perplexity.ai/search/the-amethyst-tablet-pdf-wSFKHIw3R9CFnbPRI3Vfxw#0)。
   - 爱好者们强调了这些尖端模块如何重塑未来载人任务的操作复杂性。
- **AgiBot 推进人形机器人数据集**：**AgiBot** 发布了一个新的**人形机器人训练数据集**，如[此视频](https://www.youtube.com/embed/V8A6EdbPdGU)所述，承诺在机器人运动方面实现更高的真实感。
   - 社区成员期待 AI 算法与物理控制之间更好的协同作用，为更高级的任务处理打开大门。
- **微软 1000 亿美元的 AGI 豪赌**：**Microsoft** 投入了 **1000 亿美元** 的巨额资金用于 **AGI development**，如[此处](https://www.perplexity.ai/search/why-is-it-called-donating-plas-8jLeogYsRZW4.OYo.eObhA)所述。
   - 观察人士推测，这笔巨额资金可能会重塑 AI 格局，人们既感到兴奋，也对其可能如何挑战竞争平台感到担忧。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 代币动荡**：成员们怀疑 **AI21 Labs Token** 是一个**骗局**，理由是存在可疑活动，并引用 [DEXTools](https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H) 敦促他人“远离”。
   - 用户强调了该代币可疑的持有者分布，并指称它可能已经 **rugged**（跑路）。
- **社区渴望透明度**：许多人要求 **AI21 Labs** 在 **Twitter** 上发表官方声明，坚持认为直接的警告将有助于消除任何感知到的与该代币的关联。
   - 一些人表达了沮丧，说“发一条警告推文不需要任何成本”，强调了他们多么强烈地希望公司介入。
- **安全团队介入**：**AI21 Labs** 工作人员宣布该代币与公司**无关**，并警告称如果长时间讨论加密货币可能会被**封禁**。
   - 他们将诈骗担忧上报给了**安全团队**，后者对该代币的审计声明以及与 **pumpfun** 的联系提出了质疑。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Mini O1 挑战 GPT-4**：在 #gpt-4-discussions 频道中，参与者争论 **Mini O1** 是否真的比 **GPT-4** 更聪明；一位用户声称它在某些特定任务中超越了更大的模型。
   - 其他人则认为它不是全能冠军，有人说“它在专业领域表现出色，但并非全面领先”。
- **RTX 5000 展示 DLSS 4 的提升**：在 #ai-discussions 频道中，成员们热议 **RTX 5000** 带来的 **DLSS 4** 升级，该升级承诺将三倍帧生成性能。
   - 他们强调了对游戏和图形处理的预期提升，称其为基于 GPU 的 AI 工作负载的**巨大飞跃**。
- **在实际场景中微调 LLaMA**：在 #ai-discussions 频道中，一位用户证实了在个人文本日志上微调 **LLaMA** 的成功，称其“比预期的要简单”。
   - 其他人也加入了关于结构化数据方法的讨论，描述了在一切安排妥当后明显的性能提升。
- **Schema 错误令 Prompt 工程师感到沮丧**：在 #prompt-engineering 和 #api-discussions 频道中，用户报告模型有 **80%** 的时间返回的是 JSON schema 本身，而不是有效数据。
   - 他们尝试了多次重试和调整，怀疑模糊的指令和过长的 Prompt 加剧了这种持续的混乱。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **科学拥抱基础模型 (Foundation Models)**：一位成员分享了 [Metagene 1 论文](https://metagene.ai/metagene-1-paper.pdf)，强调了 **Foundation Models** 在科学研究中的应用，引发了关于数据来源和特定领域性能的好奇。
   - 参与者询问了向相关领域扩展的可能性，激发了对 AI 与专业科学之间新合作的**希望**。
- **NVIDIA 的 Cosmos 吸引 AI 圈关注**：NVIDIA 推出了 [Cosmos](https://x.com/DrJimFan/status/1876516972512559170)，这是一个在 **20M** 小时视频素材上训练的开源视频世界模型，同时具备扩散 (diffusion) 和自回归 (autoregressive) 生成能力。
   - 社区成员称赞 Cosmos 推动了**基于视频的合成数据**的发展，并提出了关于可扩展性和更广泛企业应用的问题。
- **Vercel 的 AI SDK 评价褒贬不一**：一位用户称赞 **Vercel 的 AI SDK** 设置快速，但批评其在叠加多个模型时*抽象过多*。
   - 其他人讨论了该 SDK 在**用户友好**的脚手架与开发者控制权之间的权衡，重点关注了性能开销问题。
- **AI 助力鲸鱼追踪**：[埃森哲 (Accenture) 和悉尼大学](https://x.com/btibor91/status/1876630816199217208) 的合作者利用 AI 以 **89.4%** 的准确率检测小须鲸，将原本需要两周的手动过程压缩为近乎实时的分析。
   - 社区成员赞赏该系统的效率提升，并将其与其他野生动物监测机会进行了类比。
- **FP4 格式引发 GPU 性能讨论**：NVIDIA 对 **FP4** 指标的强调引发了关于与 **FP8** 及其他浮点格式进行公平比较的疑问。
   - 爱好者们推动建立更清晰的**基准测试标准**，并警告称定义不充分可能会误导评估下一代 GPU 的开发者。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **细字体引发关注**：社区成员批评 **Modular 文档** 的字体字重太细，指出存在潜在的可读性问题。
   - 他们敦促 **Modular** 考虑使用更粗或替代的字体选择，以提供更好的用户体验。
- **Mojo 调试器采用 LLDB**：参与者强调 **Mojo** 使用了带有上游补丁的 LLDB 方法，并引用了 [LLVM 会议的一个演讲](https://www.youtube.com/watch?v=9jfukpjCPUg)。
   - 他们称赞 **Modular** 没有重复造轮子，强调了它如何有效地支持多语言调试。
- **项目结构备受关注**：一位用户询问了关于管理导入的问题，并展示了一个 **Mojo** 项目的 [GitHub 示例](https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo)。
   - 另一位成员分享了命令 `magic run mojo test -I . tests`，并引导大家参考 [Mojo 测试文档](https://docs.modular.com/mojo/tools/testing#writing-unit-tests)。
- **静态列表与借用检查器 (Borrow Checker) 的愿景**：一位用户意识到 **ListLiteral** 无法使用运行时变量进行索引，转而选择使用 **InlineArray**。
   - 有人提议通过扩展静态分析来超越 Rust 的借用检查器 (Borrow Checker)，尽管他们更倾向于先完成现有功能。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ 征服复杂任务**：在 **Cohere** Discord 中，参与者称赞 **Command R+08** 在复杂问题任务中的高级推理能力，超越了 **Sonnet 3.5** 等其他模型。
   - 他们注意到简单的查询会降低其有效性，强调了问题复杂度对于发挥峰值性能的重要性。
- **使用 Cohere 嵌入图像**：一段代码展示了用于 **cohere.ClientV2** 嵌入 (embedding) 调用中的 [base64 编码图像输入](https://docs.cohere.com/reference/embed)，确认嵌入结果将按请求顺序返回。
   - 他们专注于正确的 content-type 请求头以及 base64 转换，以确保一致的嵌入结果。
- **JavaScript 奇思妙想：神经网络请求**：一位用户请求一个**纯 JavaScript** 实现的神经网络，完全从零开始编写。
   - 对话在没有具体代码或进一步指示的情况下结束，使这个问题留待未来探索。
- **AR 与 Cohere 结合用于飞机检测**：一位用户正在进行一个旨在检测飞机和分类物体的 **AR** 项目，寻求与 Cohere 协同实现实时资产排名。
   - 另一位贡献者称其为*“看起来太酷了”*，反映了对更多基于 AR 的工具与 Cohere 技术协作的渴望。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 中 Expand_dims 与 Reshape 的性能差异**：讨论指出，在 Triton 中 **expand_dims** 的性能表现与 **.reshape** 显著不同，特别是在维度重排（dimension reorder）能力方面。社区还权衡了 **autotuning** 策略（如 `CLOSEST_M`）以及在 **H100** 上使用 **wgmma** 以获得更好 MMA 性能的方法。
   - 他们辩论了大尺寸下的 kernel 重新编译权衡，以及如何确保 **PTX** 使用 **wgmma** 而非 **mma.sync**。对话暗示了在最大化 HPC 特性方面可能存在的配置问题。
- **CUDA 的 WMMA 魔法保留了矩阵布局**：参与者确认，从矩阵 A 加载并存储到矩阵 B 的 **WMMA** 操作保留了相同的寄存器布局，索引如 **[0,1][0,2]** 保持不变。测试表明，输出分片（output fragments）保留了输入排列，多个实验证明这有效地复制了矩阵。
   - 他们提出可以分享一个可运行的示例，并提到自那以后已不再深入探索 **WMMA**。不过，他们仍愿意展示这些硬件级原语（intrinsics）如何处理数据。
- **PyTorch 困惑：自定义 Autograd 与 Guard 日志**：尽管 PyTorch 文档警告不要这样做，但在自定义 `autograd` 函数中**修改原地梯度（in-place gradients）**的结果与更简单的参考模型一致。他们链接了 [PyTorch 关于扩展 autograd 的文档](https://pytorch.org/docs/main/notes/extending.html#how-to-use) 以提供更多背景。
   - 另一个问题是关于获取 **guard failures** 的详细日志，一位用户的日志仅显示了晦涩的 **0/0** 消息。他们使用了 `TORCH_LOGS="+dynamo,guards,bytecode,recompiles,recompiles_verbose"`，但发现输出缺乏细节。
- **Picotron 与 DeepSeek：双倍的 4D 乐趣**：[Picotron 框架](https://github.com/huggingface/picotron) 为教学目的提供了一种 **4D-parallelism**（4D 并行）分布式训练方法，展示了对高级 AI 训练策略的易用性探索。同时，短视频涵盖了 *DeepSeek-v3 论文的第 12-18 页* ([arXiv 链接](https://arxiv.org/abs/2412.19437))，以阐明 **LLM infrastructure** 概念。
   - 推荐的 [YouTube 播放列表](https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT6BUKCYeBMTEqnlcpYavxq&index=1) 进一步解释了论文的复杂性。这旨在帮助 AI 爱好者更轻松地消化密集的参考资料。
- **DIGITS 与 Discord：GPU 卓越表现的新工具**：**Nvidia** 的 [Project DIGITS](https://www.nvidia.com/en-us/project-digits/) 将 **Grace Blackwell Superchip** 与据称高达 **200B parameter** 的容量和 **128GB** 统一内存结合在一个紧凑、高性能的形态中。该硬件宣传其新的 **tensor cores** 支持 fp4 和 f8 模式，用于未来的训练扩展。
   - 同时，一个新宣布的 **基于 Discord 的 GPU 排行榜** 邀请 Alpha 测试者测量特定 kernel 的性能。发布的 `gpu-glossary.zip` 还将 GPU 基础知识的参考资料汇编在一个包中。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex & MLflow：数据驱动的双子星**：一份分步指南详细介绍了如何结合 **LlamaIndex**、**MLflow**、**Qdrant** 和 **Ollama** 进行向量存储和模型追踪，参考了[完整指南](https://t.co/lNDDgdOo86)。该指南强调使用 **Change Data Capture** 来简化实时评估。
   - 社区成员赞扬了这种协同作用，认为它有效地连接了实验追踪和嵌入式知识，并指出 **LlamaIndex** 与后端服务之间的编排变得更加简单。
- **NVIDIA AI 助力多 Agent 博客写作**：一个全新的蓝图利用 **NVIDIA AI** 处理多 Agent 任务（如博客研究和写作），该方案在 CES 上发布，官方公告见[此处](https://t.co/eS4BhuAZVS)。该方法旨在通过基于 LLM 的研究，将团队从内容创作的**时间消耗**中解放出来。
   - 它同步多个 Agent 实时执行复杂任务，保持内容生成的流程摩擦最小化。
- **Cohere 与 LlamaIndex 的精简集成**：开发者对 **Cohere** 的 Embedding 和改进后的文档表示赞赏，认为其与 **LlamaIndex** 的配合天衣无缝。他们强调了[文档](https://t.co/dLKGgkqOe8)中的安装说明和先决条件，确保了协作的顺畅。
   - 这种组合配置扩展了索引和检索操作的范围，让工程师能够更紧密地控制其文本处理流水线。
- **LlamParse 的首运行之谜**：一位用户在使用 **LlamParse** 解析 PDF 文件时遇到了意外错误，但随后的每次尝试都正常运行。项目贡献者计划检查该故障是持续发生还是偶发状况。
   - 他们请求提供有关该 PDF 的更多细节，希望能诊断出背后的格式或编码冲突。
- **Text-to-SQL 成为焦点**：**LlamaIndex** 概述了结构化数据解析和 **Text-to-SQL** 功能，用于支持对非结构化源的查询，详见 [Structured Data 文档](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data) 和 [SQLIndexDemo](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/)。一个可运行的 [Notebook 示例](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb) 解决了官方文档中链接失效的问题。
   - 该指南刻意警告不要盲目执行任意查询，敦促采用最佳实践和安全审查以确保 SQL 的安全使用。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 1.0：无法运行的代码**：在[这个 GitHub commit](https://github.com/OpenInterpreter/open-interpreter/commit/21babb186f13e263a72cf525d15d79788edf4644) 中，开发者预告了 **Open Interpreter 1.0**，但移除了代码运行功能，引起了用户困惑。
   - 他们没有提供明确的路线图，让贡献者不确定这些功能何时或如何被恢复。
- **经典版 OI 进入存档**：旧版 **Open Interpreter** 已在[此 commit](https://github.com/OpenInterpreter/open-interpreter/commit/275105771dfa985ee777e4a9c6e9c4d760c7b7b9) 归档，过时的 Prompt 被存放在只读文件夹中。
   - 经典版本的 PR 实际上已被锁定，迫使开发者将注意力转向 1.0 分支。
- **Pip 安装忧郁**：有用户反馈 `pip install open-interpreter` 无法生成稳定版本，阻碍了使用。
   - 他们遇到了功能不全的问题，并且对于如何在不破坏更多组件的情况下修复或增强当前设置感到困惑。
- **令人困扰的调整**：社区成员希望优化 Prompt 并添加新功能，但向 1.0 的转变使得合并旧的修改变得复杂。
   - 贡献者对积压的未合并 PR 表示遗憾，因为即将发布的版本在最终结构上仍未确定。
- **本地模型：使用 --no-tool-calling**：用户建议使用 `--no-tool-calling` 标志，以提高小型本地模型的性能并规避开销。
   - 他们担心 1.0 中新的系统 Prompt 更改可能会降低本地模型的准确性，从而引发了进一步的讨论。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **GH200 与编译怪癖**：一位用户确认正在使用 **GH200** 并提供了潜在支持，而其他人则指出由于层层依赖导致 **编译时间 (compilation times)** 延长，强调了从头开始配置一切的负担。
   - 他们希望通过汇集经验来减少新用户的阻碍，从而可能加快在先进开发板上进行基于 GPU 的尝试。
- **Discord 链接哥再次现身**：臭名昭著的 **Discord Link Guy** 再次出现，发布了可疑链接，引发了迅速的 **警告** 和随后的封禁。
   - 一位用户确认了该封禁，并删除了之前引起混乱的奇怪欢迎频道消息。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MiPROv2 逐条指令尝试**：建议将指令逐步输入 **MiPROv2**，并根据 LLM 的输出评价进行优化。
   - 这种方法旨在通过类似评委的反馈机制，实现生成指令的实时改进。
- **dspy.COPRO! 引发好奇**：成员们发现了 MiPROv2 的方法与 **dspy.COPRO!** 之间的相似之处，引发了进一步探索。
   - 他们建议通过迭代尝试来优化指令，从而在 MiPROv2 和 dspy 概念之间建立协同效应。
- **dspy 与 LangChain 合并遇阻**：一位用户尝试将 **dspy** 与 **LangChain** (2.6 版本) 结合以构建 LLM Agent，但遇到了困难。
   - 后续讨论指出目前没有统一这两个框架的简便路径，强调了在协调两者设计时的摩擦。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书门户重新开放**：**证书申报表 (Certificate Declaration form)** 已为 12 月完成作业的参与者重新开放，必须在 **1 月底** 前提交以获得认证资格。
   - 组织者再次强调了 **单证书** 政策，并警告不会重新开放过去的作业，敦促大家按时完成所有任务。
- **邮箱不匹配引发混乱**：多位用户强调，申报表中的 **电子邮箱地址** 必须与课程作业中使用的邮箱一致，以避免错误。
   - 一位参与者在使用了新邮箱但在表格中填写了原始邮箱后寻求确认，这凸显了如果细节不匹配，**证书发放** 可能会延迟的风险。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Reasoner v1 推进并获得关注**：一位成员称赞了 [GPT4All](https://github.com/nomic-ai/gpt4all) 上的 **Reasoner v1**，并询问了其他具备推理能力的模型，如 Qwen 2.5 coder。
   - 另一位用户确认 **OpenAI-compatible** 的远程模型和多个本地模型都可以在推理模式下运行，并补充说更多的扩展正在进行中。
- **LocalDocs 索引导致文件闲置**：一位用户在使用 **LocalDocs** 时遇到了子目录嵌入问题，指出时间戳可能导致某些文件未被嵌入。
   - 他们解释说，一旦文档在某个时间戳下被索引，系统可能会跳过后续添加的内容。
- **嵌入模型混搭引发好奇**：有人询问是否可以将默认嵌入器替换为 **text-embedding-inference** 或 **vLLM**，以改进索引任务。
   - 他们表达了对灵活嵌入的需求，以便更高效地处理自定义数据流水线。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **MLOps 与 Feature Stores 对决**：太平洋时间 1 月 15 日上午 8 点，Ben Epstein 和 Simba Khadder 将主持一场 [网络研讨会](https://buff.ly/4j9oiVg)，重点探讨 2025 年的 **MLOps** 和 **Feature Stores**。
   - 他们将涵盖最佳方法，并为寻求深入了解未来 MLOps 方法的 **Data Engineers** 和 **ML** 专业人士主持问答环节。
- **2024 MLOps 趋势展望 2025**：演讲者计划重点介绍 2024 年 **MLOps** 的重大发展以及对 2025 年的展望，重点关注真实流水线中的 **LLM**。
   - 他们预见到标准 MLOps 与 **LLMOps** 之间的协同作用，敦促参与者考虑更集成的模型部署和扩展策略。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GraySwanAI 为 LLM Security 投入 4 万美元**: **Harmful AI Assistant Challenge** 将于 **1 月 4 日下午 1 点 (EST)** 开启，为创新的 **prompt injection** 和 **jailbreaking** 方法提供 **$40,000** 奖金，详见[此推文](https://x.com/GraySwanAI/status/1872720375328411668)。
   - 允许使用多轮输入，参与者可以在 [app.grayswan.ai](http://app.grayswan.ai/arena) 注册或通过 [Discord](http://discord.gg/WqHkWt99) 加入，以深化 **LLM security testing** 技能。
- **OAI 预发布测试与社区参与**: 早期的 GraySwanAI 活动在 **o1 models** 正式发布前就对其进行了重点关注，并引用了 **12/5 OAI paper** 作为背景。
   - 这种对预发布信息的洞察记录展示了 **LLM security** 领域的强劲势头，并凸显了社区的热情。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Common Voice AMA 2025 势头强劲**: Common Voice 在[新的 Discord 服务器](https://discord.gg/b4c83ppxdU)中宣布了其 **2025 AMA**，邀请参与者回顾过去一年的里程碑并预览未来的发展。
   - 本次会议旨在解答有关项目方向的任何问题，包括来自核心团队的直接见解和 **expanded data collection** 计划。
- **2024 回顾与问答带来关键声音**: **2024 review** 活动将邀请产品总监和前端工程师分享 Common Voice 进展和后续步骤的重要更新。
   - 与会者可以在这场 **live Q&A** 中提出技术和战略问题，旨在塑造项目近期的发展轨迹。
- **语音技术中的无障碍关注**: Common Voice 致力于让 **voice technology** 更加开放和易于获取，提供可支持多种语言语音识别系统的数据集。
   - 他们强调通过民主化 **voice data** 来降低现有障碍，使开发者能够利用本地相关的解决方案服务更广泛的社区。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Dolphin 3.0 引发 BFCL 好奇**: 一位成员询问来自 Cognitive Computations 的 **Dolphin 3.0** 是否会出现在 BFCL 排行榜上，并指向了 [Hugging Face 上的 Dolphin 3.0](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3)。
   - 他们对该模型的潜在性能表示兴奋，推测它可能在现有竞争者中脱颖而出。
- **Cognitive Computations 最近的 Dolphin 3.0 提升**: **cognitivecomputations/Dolphin3.0-Llama3.2-1B** 模型更新在 Hugging Face 上获得了 34 个 star，并引发了 14 条评论。
   - 附带的一张图片展示了该模型的构建，并引起了对其技术细节和实际 **benchmarks** 的兴趣。



---


**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1325916950869835889)** (687 messages🔥🔥🔥): 

> `Unsloth updates and troubleshooting, Tokenization issues with trained LORA adapters, Fine-tuning Llama 3.2, Hardware and memory considerations for AI processing, Using cloud resources for large models`

- **Unsloth 更新与故障排除**：用户在最近的提交后遇到了 Unsloth 的问题，特别是与在 RTX3090 上加载模型有关的问题，而在 T4 GPU 上则运行成功。
   - 几位用户分享了故障排除步骤，包括回滚和更新特定版本的 Unsloth 库以解决这些错误。
- **训练后的 LORA 适配器的 Tokenization 问题**：一位用户询问了在保存训练好的 LORA 适配器后 `tokenizer.json` 体积增大的问题，并注意到了 `added_tokens.json` 文件的存在。
   - 确认了这些额外文件是必要的，并不代表存在 Bug，建议用户将其与原始的 `tokenizer.json` 一起保留。
- **微调 Llama 3.2**：一位用户寻求关于微调 Llama 3.2 所需数据集格式的澄清，特别是关于转换为包含 ('role', 'content') 键的格式。
   - 提供的示例包括 JSONL 格式文件的问题，强调了需要正确的结构以避免训练期间的错误。
- **AI 处理的硬件和内存考量**：用户之间的讨论集中在维持高内存配置（例如使用四个 48GB DIMM）稳定性的挑战上。
   - 有人指出，更大的 RAM 可以提高数据处理效率，并能够在没有云端上传/下载周期困扰的情况下运行更大的模型。
- **为大型模型使用云资源**：用户分享了在运行大型模型和数据处理任务时，平衡本地资源与云服务的经验。
   - 虽然云端访问提供了灵活性，但对上传/下载时间的担忧凸显了高效工作流对本地硬件的持续依赖。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1876729710790815872">来自 Unsloth AI (@UnslothAI) 的推文</a>：Deepseek V3，包括 GGUF + bf16 版本现在已上线 @HuggingFace！运行的最低要求：48GB RAM + 250GB 磁盘空间（针对 2-bit）。包括 2, 3, 4, 5, 6 和 8-bit 量化版本。查看所有版本...</li><li><a href="https://x.com/danielhanchen/status/1876465802951721011">来自 Daniel Han (@danielhanchen) 的推文</a>：NVIDIA RTX 5090 拥有 4000 AI TOPS - 是 RTX 4090 的 3 倍（1300 FP8 稀疏度）。RTX 5090 $1,999 3,400 AI TOPS；RTX 5080 $999 1,800 AI TOPS；RTX 5070 Ti $749 1,400 AI TOPS；RTX 5070 $549 1,...</li><li><a href="https://research.google.com/colaboratory/local-runtimes.html">
      Google Colab

</a>: 未找到描述</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA 将 Grace Blackwell 带到每个桌面，触达每一位 AI 开发者</a>：CES—NVIDIA 今日发布了 NVIDIA® Project DIGITS，这是一款个人 AI 超级计算机，为全球 AI 研究人员、数据科学家和学生提供 NVIDIA Grace 的强大动力...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 Notebooks 的列表：</li><li><a href="https://unsloth.ai/blog">博客</a>：未找到描述</li><li><a href="https://unsloth.ai/introducing">Unsloth 介绍</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/do-i-need-to-dequantization-before-merging-the-qlora/110175">在合并 QLoRA 之前我需要进行反量化吗</a>：在这个 DPO 训练器链接中提到，正如 [Benjamin Marie](https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997) 所建议的，合并 QLoRA 适配器的最佳选择...</li><li><a href="https://huggingface.co/docs/datasets/loading">加载</a>：未找到描述</li><li><a href="https://blog.gopenai.com/unsloth-unleashing-the-speed-of-large-language-model-fine-tuning-986ae7040711">Unsloth：释放大语言模型微调的速度</a>：大语言模型（LLM）彻底改变了人工智能领域，在诸如……的任务中展示了卓越的能力</li><li><a href="https://blog.gopenai.com/unsloth-unleashing-the-speed-of-large-language-m">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)">Unsloth 文档</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwyuyg/beware_of_broken_tokenizers_learned_of_this_while/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1518">[BUG] Unsloth 在今天的提交后停止工作 · Issue #1518 · unslothai/unsloth</a>：你好。我无法在我的 RTX3090 上使用 Unsloth 了。它只能在 Colab 的 Nvidia T4 上运行。当我尝试下载任何模型时，出现了这个：----------------------------------------------------------------...</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1263">[FR] 在合并 LoRA 之前对基础模型进行量化 + 反量化 · Issue #1263 · axolotl-ai-cloud/axolotl</a>：⚠️ 请检查此功能请求之前是否已被提出。我在讨论区搜索了之前的想法，没有发现类似的请求。我搜索了之前的 Issues，也没有发现...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1089">Unsloth QLoRA 合并到基础模型：如果你想在 vLLM 或 NVIDIA TensorRT-LLM 中运行训练好的模型，最佳实践是什么？ · Issue #1089 · unslothai/unsloth</a>：关于 Unsloth 的 LoRA 与 QLoRA，有一个快速（且重要）的问题。我阅读了一系列关于不要天真地将 QLoRA 合并回基础模型的文章，这会导致性能下降...</li><li><a href="https://github.com/unslothai/notebooks">GitHub - unslothai/notebooks: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebooks。</a>：适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebooks。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/195">合并到 16 bit 是否是与反量化的基础模型合并？ · Issue #195 · unslothai/unsloth</a>：Colab Notebooks 显示：# Merge to 16bit if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",) if False: model.push_to_hub_merged("hf/mod...</li><li><a href="https://github.com/unslothai/unsloth/issues/195,">Issues · unslothai/unsloth</a>：微调 Llama 3.3, Mistral, Phi, Qwen 2.5 & Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - Issues · unslothai/unsloth</li><li><a href="https://github.com/jondurbin/qlora/blob/main/qmerge.py#L42">qlora/qmerge.py at main · jondurbin/qlora</a>：QLoRA：量化 LLM 的高效微调。通过在 GitHub 上创建账号为 jondurbin/qlora 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 微调 Llama 3.3, Mistral, Phi, Qwen 2.5 & Gemma LLM，速度提升 2-5 倍，显存占用减少 70%</a>：微调 Llama 3.3, Mistral, Phi, Qwen 2.5 & Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/1520">由 sebaxakerhtc 更新 __init__.py · Pull Request #1520 · unslothai/unsloth</a>：此 PR 解决...</li>

ving the issue with some GPUs
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326109933347016814)** (2 messages): 

> `Gemini 1207 knowledge cutoff, Picotron codebase for fine tuning` 


- **Gemini 1207 过时的知识储备阻碍了编程**：一位成员指出 **Gemini 1207** 的知识截止日期（knowledge cutoff）非常久远，导致它在提供最新库的支持方面表现不足。
   - 这一限制给依赖最新编程协助的用户带来了困扰。
- **关于 Picotron codebase 效能的询问**：有人提出了关于 **Picotron codebase** 在进行 fine tuning 时效果如何的问题。
   - 成员们很好奇是否有人对其 fine-tuning 流程的能力有深入见解或经验。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326005131053105224)** (26 messages🔥): 

> `Merging LoRA adapters, Multiple datasets for finetuning, Deploying LLaMA models, Multi-GPU training support` 


- **为 Ollama 手动合并 base model 和 LoRA adapter**：为了将 base model 和 LoRA adapter 合并以在 Ollama 中使用，成员们建议保存 LoRA，以 FP16 格式进行合并，并遵循使用提供的 wrappers 或 pipeline 脚本的常规转换步骤。
   - 这种方法有助于避免依赖 `model.save_pretrained_gguf()`，因为该方法在某些环境下可能不可行。
- **使用多个数据集进行 finetuning**：是的，可以使用多个数据集进行 finetuning，但建议将它们以统一的格式合并，以获得更好的效果。
   - 分享了一个专门针对[多数据集的教程](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing)供参考。
- **在 Flutter 应用程序中部署 fine-tuned 模型**：要在 Flutter 等应用中部署 fine-tuned 的 LLaMA 模型，除非使用云端解决方案，否则必须拥有强大的 GPU。
   - 利用 Hugging Face Spaces 等服务托管公开模型，或在 together.ai 等平台上托管，可以提供轻松运行模型的选项。
- **免费的 API 访问解决方案**：成员们建议将 Ollama 与 OpenWebUI 或 Flowise 结合使用，以实现对已部署模型的免费 API 访问。
   - 这些工具可以毫不费力地促进网站上的聊天界面和集成。
- **Unsloth Multi-GPU 训练的现状**：目前 Unsloth 不支持 Multi-GPU 训练，商业支持预计将在未来推出。
   - 这一限制已通过社区讨论和文档查阅得到确认。



**提及的链接**：<a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing">Google Colab</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1326044073840083025)** (1 messages): 

> `Token Embeddings, Ontological Concepts, Semantic Meaning` 


- **关于 Token Embeddings 局限性的辩论**：讨论集中在对当前基于词碎片（word fragments）的 **token embeddings** 的担忧，重点是它们在捕捉语义丰富性（semantic richness）方面的局限性。
   - 一位成员断言 *“当前的 token embeddings 是有限的”*，主张转向本体论的 'concepts'，以获得更丰富的个体向量嵌入。
- **推动基础 Ontological Concepts 的研究**：一位成员一直倡导探索 **foundational ontological concepts** 以增强语义嵌入策略，认为它们比单纯的词碎片提供更深层的含义。
   - 他们分享了观点，认为这些概念可以带来信息量显著增加的 embeddings，挑战现有的范式。
- **审阅关于 Semantic Meaning 的相关论文**：一篇相关的论文（链接见[此处](https://arxiv.org/pdf/2412.08821)）引发了关于替代嵌入技术的讨论，这些技术旨在解决当前模型的局限性。
   - 鼓励参与者审阅研究结果，因为它们与推动从本体论（ontological）视角衍生 embeddings 的努力相一致。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1325920412890304654)** (1 条消息): 

> `LM Studio 0.3.6 发布，Function Calling API，支持 Qwen2VL 和 QVQ，新安装程序功能，应用内更新` 


- **LM Studio 0.3.6 发布，带来令人兴奋的新功能！**：新版本 [0.3.6](https://lmstudio.ai/download) 引入了 **Function Calling / Tool Use API**，使其在支持本地模型的同时，兼容现有的 OpenAI 工具。
   - 该更新目前处于 beta 阶段，鼓励用户提供 [bug 报告](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues) 和反馈。
- **迎接全新的视觉输入模型：Qwen2VL 和 QVQ！**：**0.3.6** 版本现在在 LM Studio 的 **MLX** 和 **llama.cpp** 引擎中均支持 **Qwen2VL** 系列和 **QVQ** 模型。
   - 这些模型通过先进的视觉和推理功能增强了能力，适用于更强大的应用场景。
- **安装程序为 Windows 用户增加了新功能！**：新的安装程序允许用户选择安装盘符，这是用户期待已久的功能。
   - 更新过程现在更加高效，更新包更小，并为方便用户提供了进度条。
- **应用内更新提升用户体验！**：从 **0.3.5** 稳定版开始的应用内更新将于本周晚些时候开始，过渡到新的更新系统。
   - 用户可以更新其 **llama.cpp** 和 **MLX** 引擎，而无需进行完整的应用更新。
- **展示 Qwen2VL 模型的演示让用户印象深刻！**：一个展示 **Qwen2VL** **2B** 模型的演示突显了其能力，令社区惊叹。
   - 用户可以观看公告中链接的演示，查看模型的实际运行情况。[观看演示](https://cdn.discordapp.com/attachments/1111797717639901324/1325920413296885760/qwen2vl-demo-2.mp4?ex=677edc9c&is=677d8b1c&hm=aaf1d2dac50e73b9429009946a925816baaf6c8a49e440b49fd6610e57f7b79e&)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>：Tool Calling API 处于 beta 阶段，新的安装程序/更新系统，以及对 `Qwen2VL` 和 `QVQ`（支持 GGUF 和 MLX）的支持。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1325917009342500894)** (201 条消息🔥🔥): 

> `AMD 演讲中的 LM Studio、Function calling API 更新、Qwen-VL 模型加载问题、4090 GPU 性能基准测试、新 UI 设计反馈` 


- **AMD 演讲中出现 LM Studio 令人惊喜**：一位用户对在 **AMD 演讲** 中看到 **LM Studio** 亮相表示惊讶。
   - 这一意外的加入引发了人们对 LM Studio 能力的浓厚兴趣。
- **关于 Function Calling API 的讨论**：用户讨论了新的 **function calling API**，其中一人询问了与之前版本相比的变化。
   - 据指出，该版本扩展了功能，但需要专门的 Beta 测试报名。
- **加载 Qwen-VL 模型时遇到的挑战**：几位用户在加载 **Qwen-VL** 模型时遇到了问题，特别是与上下文长度（context length）和退出代码（exit codes）相关的问题。
   - 该问题被发现仅限于 Linux 系统，且在测试期间某些功能仍处于损坏状态。
- **RTX 4090 上的性能基准测试**：一位用户报告称，在 **RTX 4090** 上运行 **Qwen2.5-Coder-32B-Instruct** 模型时，速度约为 **31 Tokens/s**。
   - 据记录，GPU 利用率约为 95%，功耗为 **385 Watts**，展示了其出色的性能表现。
- **关于新 UI 设计的反馈**：一位用户批评了新 UI，表示更喜欢旧设计，并提到了浅色模式和按钮逻辑更改等问题。
   - 回复中包含了如何切换回经典 UI 的说明，并对反馈表示了认可。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Barnacules/status/1873631114864292330">来自 Barnacules Nerdgasm (@Barnacules) 的推文</a>：NVIDIA GeForce RTX 4090 真是 GPU 中的猛兽！🏆 我在 LM Studio 中运行加载到 VRAM 的 18GB Qwen2.5-Coder-32B-Instruct，速度达到 ~31 Tokens/s。功耗仅为 350W TDP...</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">惊讶的皮卡丘 GIF - Surprised Pikachu Pokemon - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=dRQUkq_HzEE">Facebook 制作了令人毛骨悚然的 AI 个人资料...</a>：大家好，又是我 Mutahar！Facebook 透露了将其网站转变为人类和人工智能机器人混合体的计划。随着...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285">(Exit code 133) 加载大型 LLM 模型时出错 · Issue #285 · lmstudio-ai/lmstudio-bug-tracker</a>：加载大型 LLM（例如上下文窗口为 32768 的 Meta-Llama-3.1-70B-Instruct-IQ2_S）时，会遇到错误 (Exit code: 133)。请检查设置并尝试重新加载模型...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过创建账户为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285.">Issues · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 桌面应用程序的 Bug 追踪 - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://lmstudio.ai/docs/advanced/tool-use">工具使用 - 高级 | LM Studio 文档</a>：使 LLM 能够与外部函数和 API 进行交互。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1325917201638756475)** (227 条消息🔥🔥): 

> `NVIDIA Project DIGITS, Speculative Decoding, AI model performance, AMD vs NVIDIA GPUs, Local LLM inference` 


- **NVIDIA Project DIGITS 革新 AI 计算**：NVIDIA 发布了 Project DIGITS，这是一款紧凑型 AI 超级计算机，能够运行 200B 参数模型，配备 128GB 一致性内存（coherent memory），性能较传统配置有显著提升。
   - 开发者可以高效地进行大型 AI 模型的原型设计和部署，但该技术的定价和性能仍需在实际应用场景中进行全面评估。
- **Speculative Decoding 提升 LLM 推理速度**：最近的讨论强调了 llama.cpp 中引入了 Speculative Decoding，承诺在不损失准确性的情况下，将大语言模型的推理速度提高 25-60%。
   - 虽然各渠道都在等待该功能在不同平台上的集成，但值得注意的是，草稿模型（draft models）可以显著增强性能。
- **性能对比：AMD vs NVIDIA**：针对 AI 应用的 AMD 与 NVIDIA GPU 对比引发了对 VRAM 限制的关注，并探讨了即将推出的 GeForce 50 系列，特别是 5090 型号。
   - 尽管对 NVIDIA 的云解决方案持怀疑态度，但人们承认利用两家公司能力的混合硬件配置具有潜在优势。
- **AI 推理测试的创新配置**：用户探索了通过将高 VRAM 模型与较小模型结合来设置 AI 推理，以提高效率并保持质量，特别是在 3090 和 3060 GPU 等中端硬件上。
   - 值得注意的是，分享了关于使用替代配置运行实验的见解，从而在推理速度方面获得了显著提升。
- **本地模型性能与资源限制**：对话强调了在有限硬件资源上运行大型 AI 模型的挑战，一些用户报告了在 VRAM 显著较少的旧机器上成功配置的案例。
   - 尽管存在限制，用户仍在寻找创新方法来管理性能需求，展示了 AI 开发领域不断进行的适应性调整。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">Speculative Decoding — Make LLM Inference Faster</a>: 在不降低任何准确性的情况下，将 LLM 推理速度提高 2-3 倍</li><li><a href="https://www.nvidia.com/en-us/geforce/special-event/">NVIDIA GeForce Special Events at CES 2025 </a>: 收看 NVIDIA CEO 黄仁勋 (Jensen Huang) 的开幕主题演讲。</li><li><a href="https://www.gigabyte.com/Graphics-Card/GV-N5090AORUSX-WB-32GD/sp#sp">AORUS GeForce RTX™ 5090 XTREME WATERFORCE WB 32G Specification | Graphics Card - GIGABYTE Global</a>: 未找到描述</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips</a>: CES——NVIDIA 今日发布了 NVIDIA® Project DIGITS，这是一款个人 AI 超级计算机，为全球 AI 研究人员、数据科学家和学生提供访问 NVIDIA Grace 算力的途径...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hqlug2/revisting_llamacpp_speculative_decoding_w/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1gzm93o/speculative_decoding_just_landed_in_llamacpps/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: 立即预订。</li><li><a href="https://www.youtube.com/watch?v=dQ8gSV_KyDw"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: Run your own AI cluster at home with everyday devices 📱💻 🖥️⌚</a>: 使用日常设备在家里运行你自己的 AI 集群 - exo-explore/exo</li><li><a href="https://github.com/ollama/ollama/pull/8134#issuecomment-2550018120">feat: Introduce speculative decoding by bfroemel · Pull Request #8134 · ollama/ollama</a>: 此 PR 旨在复制 https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp 中实现的 speculative decoding。请参阅文档 (docs/faq.md) 中的提示以进行尝试...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1325917307435876362)** (71 条消息🔥🔥): 

> `DeepSeek vs Codeium 模型, Codeium 订阅问题, Codeium Chat 功能, AI 模型支持与测试, 用户对 Windsurf 性能的担忧` 


- **DeepSeek vs Codeium 模型之争**：成员们讨论了是否更倾向于 **DeepSeek v3** 而非 **Codeium 的模型**，并表示如果数据问题得到解决，这将是一个明确的选择。
   - *DeepSeek v3* 被指出在强大的 AI 输出方面有其根源，这引发了对 Codeium 企业许可方式的担忧。
- **Codeium 订阅问题**：用户报告了 Codeium 订阅未反映升级的问题，并询问了支付重试和支持响应时间。
   - 成员们互相安慰，由于最近的假期和高需求，客户支持可能需要更长时间。
- **Codeium Chat 的功能与错误**：几位用户对 Codeium Chat 无法正常连接表示沮丧，提到了错误消息并需要手动刷新。
   - 用户对 **o1-preview** 的不一致性表示担忧，一些用户建议使用其他方法输入数据。
- **对各种 AI 模型的支持**：关于为何不支持其他 AI 模型的讨论强调了 Codeium 在发布新功能前进行的彻底测试过程。
   - 成员们指出虽然有许多能力出众的模型可用，但强调了 Codeium 团队进行仔细评估的必要性。
- **用户对 Windsurf 性能的沮丧**：一位用户表达了对升级后 Windsurf 性能的沮丧，因为它仅分析了极少量的代码行。
   - 用户对这些变化如何影响代码稳定性和完整性表示担忧，特别是对于高级用户。



**提到的链接**: <a href="https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev">Cline&#32;(prev.&#32;Claude&#32;Dev)&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: VS Code 扩展 - 直接集成在 IDE 中的自主编码 Agent，能够创建/编辑文件、运行命令...

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1325926934206615616)** (242 条消息🔥🔥): 

> `Windsurf 错误, Cascade Autocomplete 问题, 用户体验反馈, 内部服务器错误, 功能请求与建议` 


- **Windsurf 编辑时的错误**：许多用户报告在 Windsurf 中编辑文件时遇到“ErrorCascade has encountered an internal error in this step”错误消息，表明存在持续的稳定性问题。
   - Darmitage 指出，尽管存在这些问题，Windsurf 仍巧妙地尝试通过创建 .new 文件并在之后替换原始文件来解决问题。
- **Autocomplete 功能缓慢**：包括 Sayokurisu 和 a_a_garc 在内的许多用户经历了缓慢的 Autocomplete 性能，在社区中引发了沮丧。
   - 这种变慢现象普遍存在，引发了关于影响系统的潜在底层问题的讨论。
- **工作流中的错误可见性**：几位用户报告反复遇到“HTTP status 503 Service Temporarily Unavailable”错误，严重阻碍了他们的工作流。
   - 这导致了大家对 Windsurf 应用程序内服务可用性和可靠性的共同担忧。
- **用户体验与反馈**：用户表达了对 Windsurf 当前功能的感受，一些人欣赏其对错误的巧妙处理，而另一些人则批评其性能。
   - 对话强调了开发团队需要就正在进行的问题和修复进行更好的沟通。
- **未来的改进与支持**：用户呼吁 Windsurf 团队提供更多支持和透明度，希望明确未来的更新和改进。
   - 讨论中还提出了增强用户协作功能和减少错误频率的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://tenor.com/view/multiversx-x-xportal-egld-crypto-gif-4249062898891695021">Multiversx Xportal GIF - Multiversx X Xportal - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/ChatGPTCoding">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1325927130558627883)** (268 条消息🔥🔥): 

> `Project DIGITS, Stable Diffusion 许可, 图像生成质量, Flux 生成时间, NVIDIA Cosmos` 


- **NVIDIA 发布 Project DIGITS**: NVIDIA 宣布推出 Project DIGITS，这是一款售价 3,000 美元的个人 AI 超级计算机，能够处理高达 **2000 亿参数**的 AI 模型，并由全新的 **GB10 Grace Blackwell** 超级芯片驱动。
   - 它专为开发者在本地进行大型 AI 模型原型设计而设计，预计性能将优于现有的高端 GPU。
- **Stable Diffusion 模型的商业用途**: Stability AI 的模型可用于商业用途，只要年收入低于 **100 万美元**，允许个人在未经许可的情况下创建和分发衍生作品。
   - 讨论强调了围绕条款的混淆，但指出将 AI 内容用于商业目的需要遵守许可协议。
- **图像生成质量对比**: 用户对比了 **Stable Diffusion 3.5** 和 **Flux** 的图像输出质量，指出虽然 3.5 速度更快，但 Flux 提供了更好的图像质量和精细化能力。
   - 一些用户建议使用 3.5 进行初始原型设计，随后使用 Flux 进行细节精炼。
- **CFG Scale 对 Flux 生成时间的影响**: 一位用户注意到，在 Flux 生成中增加 **CFG Scale** 会导致图像处理时间显著延长，这表明在更改 Prompt 时可能存在效率低下的问题。
   - 有人担心 Flux 是否针对 Denoising 进行了优化，而非针对 Prompt 的有效利用。
- **NVIDIA Cosmos 概览**: NVIDIA Cosmos 平台作为世界模型（world models）的开发工具推出，提供包括 **Diffusion 和 Autoregressive** 在内的多种模型类型，旨在用于物理 AI 应用。
   - 用户称赞了它的能力，指出其性能水平可与市场上其他高质量模型相媲美。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.runpod.io/runpod-slashes-gpu-prices-powering-your-ai-applications-for-less/">RunPod Slashes GPU Prices: Powering Your AI Applications for Less</a>: RunPod 正在降低其 Serverless 和 Secure Cloud 服务的价格。为什么？因为我们相信为您提供构建应用程序所需的火力，而无需花费巨资。关于 O 的内幕...</li><li><a href="https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai">Nvidia announces $3,000 personal AI supercomputer called Digits</a>: 它的尺寸只有台式机大小。</li><li><a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/">NVIDIA Open Models License</a>: 未找到描述</li><li><a href="https://stability-ai.squarespace.com/core-models">Stability AI Core Models &mdash; Stability AI</a>: 核心模型可供专业和企业会员根据其会员协议条款进行商业使用。</li><li><a href="https://stability.ai/license">Stability AI License &mdash; Stability AI</a>: Stability AI 许可通过结合我们的一系列最先进的开放模型与自托管优势，为您的生成式 AI 需求提供灵活性。</li><li><a href="https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/">NVIDIA Blackwell Architecture</a>: 将生成式 AI 推向万亿参数规模。</li><li><a href="https://github.com/NVIDIA/Cosmos">GitHub - NVIDIA/Cosmos: Cosmos 是一个世界模型开发平台，由世界基础模型、分词器和视频处理流水线组成，旨在加速机器人和自动驾驶实验室中物理 AI 的开发。Cosmos 专为物理 AI 构建。Cosmos 仓库将使用户能够运行 Cosmos 模型、运行推理脚本并生成视频。</a>: Cosmos 是一个世界模型开发平台，由世界基础模型、分词器和视频处理流水线组成，旨在加速机器人和自动驾驶实验室中物理 AI 的开发...</li><li><a href="https://medium.com/@promptingpixels/can-stable-diffusion-models-be-used-for-commercial-use-it-depends-eedd89272245>">未找到标题</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: 立即预订。
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1325919641352011956)** (9 messages🔥): 

> `导出 Bolt 项目，使用外部 LLM，手动上传项目` 


- **高效导出 Bolt 项目**：一位成员提到可以在每次迭代后**导出 Bolt 项目**，从而无缝集成到工作流中。
   - *使用外部 LLM* 允许进行调试和微调，而不会在细微调整上浪费 Token。
- **Bolt 项目的 IDE 兼容性**：另一位成员确认，虽然可以将项目导出到其他 IDE，但需要进行一些**调整**才能正常运行。
   - 这表明用户需要熟悉其所选 IDE 的特定设置。
- **手动上传项目**：在询问如何将现有项目添加到 Bolt 时，一位成员了解到可以通过**公共 GitHub repo**上传本地项目。
   - 为此，他们需要使用以下格式将其拉入 Bolt：`bolt.new/github.com/githubUsername/repoName`。



**提及的链接**: <a href="https://stellular-beijinho-102b6b.netlify.app/">Vite + React + TS</a>: 未找到描述

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1325926075032670229)** (258 messages🔥🔥): 

> `Token 消耗担忧，使用 Supabase 开发聊天应用，Bolt 与 GitHub 集成问题，移动端框架选择，账户迁移与预览问题` 


- **对 Token 消耗的担忧**：成员们对高 Token 消耗表示沮丧，有报告称即使在较小的项目中，单个 Prompt 也会消耗超过 **150 万个 Token**。
   - 一些人认为代码更改的低效导致了意外的 Token 使用，引发了对有效管理成本的担忧。
- **聊天应用开发的挑战**：在使用 **Supabase** 开发聊天应用时，一位成员遇到一对一聊天无法实时显示新消息的问题。
   - 有建议认为在通知中传递消息可能会解决该问题，这表明是 UI 问题而非后端故障。
- **GitHub 与 Bolt 集成问题**：一位成员报告了通过 GitHub 向 **Render.com** 部署更新时的挑战，导致需要手动干预。
   - 社区建议提交 GitHub issue 以便更好地跟踪和潜在的未来修复。
- **移动应用开发的框架选择**：一位用户讨论了 **NativeScript + Vue** 的困难，在构建音板应用时收到 npm 命令错误。
   - 成员建议探索其他框架或对现有设置进行故障排除，以避免重复出现错误。
- **账户与预览显示问题**：一位使用新笔记本电脑的用户在 Bolt 上遇到问题，无法预览现有项目或新项目，导致白屏。
   - 另一位成员询问了基于特定项目链接工作与直接从 GitHub 目录工作的重要性，将其作为潜在的解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://repocloud.io/boltdiy">RepoCloud | Bolt.diy: Choose Your AI Model</a>: 探索 Bolt.diy，这是选择你喜欢的 AI 模型的终极 Fork 版本。使用 OpenAI 和 Anthropic 等顶尖 LLM 定制你的编码体验！</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: 问题：Bolt 在聊天中输出应用逻辑。例如，当用户达到速率限制时，提供升级链接的代码会作为聊天响应发送给用户。</li><li><a href="https://github.com/stackblitz/bolt.new/issues/1809">Feature Request: Upload image files to Bolt · Issue #1809 · stackblitz/bolt.new</a>: 目前无法向 Bolt 上传图像文件。解决方案正在开发中。在此期间，你可以：在 StackBlitz 中打开你的 Bolt 项目（点击右上角的 &quot;Open in StackBlitz&quot;）...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5108">Backend Server integration · Issue #5108 · stackblitz/bolt.new</a>: 目前必须手动更新连接到 GitHub repo 的 StackBlitz，因为 bolt.new 不会自动更新。我的 render.com 服务器连接到我的 GitHub repo，仅在 Gith... 发生变化时更新。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1325918688787824692)** (191 条消息🔥🔥): 

> `Cursor IDE 性能问题, 代码结构的模块化, 编程任务中的 AI 行为, 用于项目理解的 Cursor 扩展, Composer agent 的问题` 


- **Cursor IDE 出现性能问题**：用户报告称 **Cursor IDE** 存在延迟和错误，特别是在使用 **Composer agent** 时，影响了工作流并导致挫败感。
   - 问题包括找不到文件、代码中被添加多余空格，以及在多次 prompt 后保存 context 时遇到挑战。
- **倡导模块化代码结构**：多位成员提倡使用 100 行左右的小型模块化文件，以帮助 AI 工具更好地管理代码并避免技术债。
   - 然而，一些人表示担心，虽然模块化可以防止代码丢失，但它会增加 AI 的文件管理复杂度，导致代码发现（code discovery）变得困难。
- **持续存在的 AI 行为挑战**：用户分享了 AI 模型做出错误或破坏性修改的挑战，导致代码丢失并需要不断重新进行 recontextualization。
   - 讨论强调了在编程环境中使用 AI 时，进行仔细的指令引导和 memory management 的必要性。
- **对提升项目感知能力的 Cursor 扩展感兴趣**：一位用户分享了一个 **Reddit 链接**，关于一个旨在提高 AI 对代码库理解能力的扩展项目，这可能会增强其性能。
   - 该扩展旨在创建一个“项目大脑”，帮助 AI 更好地追踪文件关系并理解编码模式。
- **遇到 Composer agent 问题**：用户注意到 **Composer agent** 的频繁问题，例如在有限次数的交流后无响应，以及点击文件链接时报错。
   - 建议使用键盘快捷键或手动方法进行文件导航，以克服界面限制。



**提到的链接**：<a href="https://www.reddit.com/r/cursor/s/2BGt4BZ21e">Reddit - Dive into anything</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326020961568559205)** (3 条消息): 

> `Embarcadero 见面会, 会议日程, Shack15 地点` 


- **Embarcadero 见面会规划**：一位成员提到他们将在 **周三** 和 **周四** 前往 **Embarcadero** 并有一些空余时间。
   - 另一位成员提议于 **周四** 上午在 **Shack15** 见面。
- **周四会议确认**：见面会计划已确认于 **周四** 在 **Shack15** 举行。
   - *让我们敲定上午的具体时间*作为会议细节。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326221362032803851)** (38 messages🔥): 

> `OpenAI AI Agent 发布，Devin 估值与支持，01.AI 创业公司更新，Anthropic 融资，AI 竞争` 


- **OpenAI 对 AI Agent 的犹豫**：OpenAI 延迟发布其 AI Agent 的一个原因是担心 **Prompt Injection 攻击**；不过，有报告显示该软件可能在本月推出 [更多信息](https://www.theinformation.com/articles/why-openai-is-taking-so-long-to-launch-agents)。
   - 关于定价的猜测暗示，企业用户的潜在成本可能在 **$2K** 左右，这引发了对必要支持改进的担忧。
- **Devin 与市场动态**：尽管估值达到 **$2B**，但人们对 Devin 的产品效能和支持表示怀疑，一些用户因感到沮丧而取消了订阅。
   - 成员们一致认为，竞争对手可能会稳定价格，即使产品看起来具有革命性，也能防止价格大幅上涨。
- **AI 创业公司 01.AI 的声明**：来自 **01.AI** 的李开复否认了有关解散并将团队出售给阿里巴巴的传闻，表示他们 2024 年的收入超过了 **1 亿人民币（1400 万美元）**，并预计 2025 年将实现显著增长 [来源](https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/)。
   - 尽管收入前景乐观，但由于据报道该公司裁撤了其预训练算法和基础设施团队，不确定性依然存在。
- **Anthropic 的融资轮次**：关于 **Anthropic** 的更多讨论显示，他们已获得 20 亿投资，公司估值达到 **600 亿美元**，预计 ARR 为 8.75 亿美元，主要来自 B2B 渠道。
   - 这笔大胆的融资引发了对竞争格局的关注，许多初创公司正在争夺地位。
- **市场炒作与现实审视**：频道反思了围绕 AI 的快速市场炒作，警告不要相信此类技术会在一夜之间导致大规模失业。
   - 相反，参与者强调了像 Google 这样成熟的竞争对手在保持预期（以及价格）现实化方面的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/steph_palazzolo/status/1876646459698991573">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：OpenAI 领先对手发布 AI Agent 的原因之一是担心所谓的 Prompt Injection 攻击。不过不用担心……我们听说 OpenAI 的计算机使用 Agent 软件……</li><li><a href="https://x.com/TheXeophon/status/1876529686349824163">来自 Xeophon (@TheXeophon) 的推文</a>：@adibvafa 背景：我们账户中出现了一些无法归因的使用量，仪表盘显示矛盾，有些使用量在没有对应模型负责的情况下被计费。得到了一些……</li><li><a href="https://x.com/paulgauthier/status/1872423717969801320">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Aider v0.70.0 发布：- 全面支持 o1。- 监听文件：- 现在支持 --subtree-only - 改进了 Prompting - 显示关于 AI! 和 AI? 使用的提示 - Aider 编写了此版本中 74% 的代码。完整变更……</li><li><a href="https://technode.com/2025/01/07/01-ai-refutes-rumors-of-selling-teams-to-alibaba/">01.AI 否认将团队出售给阿里巴巴的传闻 · TechNode</a>：01.AI 是中国领先的 AI 独角兽创业公司之一，此前有传闻称其将被解散，预训练和算力卡团队据传将出售给阿里巴巴。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1325973756652552273)** (8 messages🔥): 

> `MosaicML researchers, ChatGPT transcription versions, Token usage in responses` 


- **MosaicML researchers 面临挑战**：一位成员对 **MosaicML researchers** 表示担忧，称他们周围传出“悲伤的鹅叫声”，暗示存在未公开的问题。
   - 另一位成员强调了他们对 **Mosaic team** 的喜爱，并赞赏了他们**出色的 streaming dataset**。
- **Incognito 传闻暗示存在困难**：一位用户提到拥有关于 MosaicML 的可靠二手信息，并表示除非社区公开参与，否则不会透露更多。
   - 这引发了群组内的猜测，让许多人对现状感到好奇。
- **ChatGPT transcription 幽默**：一位成员幽默地回顾了他们的 Windows 10 transcription 产生的各种变体，列举了如 *chatty gpt* 和 *chat ebt* 等名称。
   - 这个轻松的评论引发了笑声，展示了社区对 AI 奇癖的调侃。
- **关于 ChatGPT 使用 emoji 的突发新闻**：一位成员宣布，他们指示自己的 **ChatGPT version** 停止使用 emojis，因为这会**浪费 tokens**。
   - 这种观点引起了一些人的共鸣，突显了关于 AI 交互中资源效率的持续讨论。

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1325937513214247015)** (39 条消息🔥): 

> `Nvidia Project Digits 超级计算机, Nvidia ARM CPU 的挑战, AI 的社区协作与资金, 开源软件兼容性` 


- **Nvidia Project Digits 将 AI 算力带到桌面**：Nvidia 在 CES 上发布了 [Project Digits](https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai)，这是一款搭载全新 GB10 Grace Blackwell Superchip 的个人 AI 超级计算机，能够处理高达 2000 亿参数的模型，售价为 3,000 美元。
   - CEO 黄仁勋（Jensen Huang）强调了其在主流 AI 应用方面的潜力，这可能让开发者获得尖端的计算资源。
- **对 Nvidia ARM CPU 的担忧依然存在**：讨论显示 Nvidia 的 ARM CPU 架构带来了挑战，许多开源软件包缺乏预编译的二进制文件（precompiled binaries），导致安装和兼容性困难。
   - 用户分享了他们在 Nvidia Jetson 设备上的痛苦经历，这些设备虽然拥有原始算力，但在软件层面极具挑战性。
- **关于新 LLM 机器的资金讨论**：社区成员询问是否应将年度预算的大部分分配给新的个人 LLM 机器，权衡潜在 AI 模型实验的收益。
   - 对话强调了利用未发布 AI 模型的兴趣，以及对非营利组织而言成本效益的重要性。
- **通过开源进行集体学习**：一位成员表示愿意指派一名队友协作完成促进开源的任务，并协助社区学习和分享知识。
   - 强调了在有效利用资源的同时，从相互支持和协作中获益的情感。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/badmephisto">badmephisto - Twitch</a>：一名矮人牧师在 WoW Classic 的 Dreamscythe 领域的冒险</li><li><a href="https://www.theverge.com/2025/1/6/24337530/nvidia-ces-digits-super-computer-ai">Nvidia 发布名为 Digits 的 3,000 美元个人 AI 超级计算机</a>：它只有台式机大小。</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-06/us-adds-tencent-to-chinese-military-blacklist-shares-decline?utm_source=website&utm_medium=share&utm_campaign=copy">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-06/us-adds-tencent-to-chinese-military-blacklist-sha">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://x.com/HanchungLee/status/1876468876747341938">Han (@HanchungLee) 的推文</a>：从皮夹克到鳄鱼皮夹克的 10 倍缩放。</li><li><a href="https://www.theregister.com/2025/01/07/nvidia_project_digits_mini_pc/">Nvidia 揭晓精简版 Grace-Blackwell Superchip</a>：专为在桌面上运行大型模型而调优，配备 128GB RAM 和定制版 Ubuntu</li><li><a href="https://blackforestlabs.ai/flux-nvidia-blackwell/">与 NVIDIA 合作，为更多创作者带来闪电般的 FLUX 性能</a>：我们与 NVIDIA 的新合作标志着在使我们的 FLUX 模型更普及、更高效方面迈出了重要一步。通过降低内存需求、提升性能……</li><li><a href="https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6">Cosmos - nvidia 集合</a>：未找到描述
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1325981855614500874)** (4 条消息): 

> `AI2 社区参与，Kling v1.6 与电车难题，Nextcloud 支持挑战` 


- **AI2 的沟通正向演变**：一位成员指出 **AI2** 的沟通变得越来越 **based**（基于事实且有态度），对他们在协作方面的未来方向表示乐观。
   - *如果我们打算从 AI2 的工作中获益，社区应该寻找回馈的方式。*
- **Kling v1.6 回避电车难题**：一位成员尝试测试 **Kling v1.6** 如何应对**电车难题**，但它只是*缓慢地退缩了*。
   - 这引发了关于 **AI 伦理编程**及其对道德困境反应的疑问。
- **Nextcloud 的 OSS 社区需要支持**：有人对 **Nextcloud** 表示担忧，这是一个尽管潜力巨大但社区支持有限的开源平台。
   - *Nextcloud GmbH* 正忙于为机构客户增强功能，鼓励社区加强贡献。
- **对 Nextcloud OSS 社区的支持**：大家集体关注 **Nextcloud** 及其开源社区，一位成员表示，*为 Nextcloud 及其 OSS 社区祈祷*。
   - 这凸显了需要更多草根支持倡议来增强其使用率和社区参与度。



**提到的链接**：<a href="https://fxtwitter.com/fofrAI/status/1876638297134678173">来自 fofr (@fofrAI) 的推文</a>：我试着看 Kling v1.6 会如何处理电车难题。但它只是缓慢地退缩了。

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1325935902198857942)** (67 条消息🔥🔥): 

> `RL 训练中的 Agent，函数调用与工具使用，自我纠错机制，奖励模型与博弈行为，推理轨迹生成` 


- **Agent 迎来多样化的 RL 训练**：关于如何在复杂环境中使用 RL 训练 Agent 存在各种推测，正如一位成员所言，*“不能仅仅是‘问题 => 前向 CoT => 输出’然后接收奖励。”*
   - 对话强调了需要通过具有可验证结果的多样化交互来有效地训练这些 Agent。
- **函数调用与工具使用得到澄清**：成员们讨论了函数调用机制的差异，指出 o1 创建编排逻辑，而 4o 则使用各种工具执行逻辑。
   - 这引发了关于模型在强化学习期间如何分配和利用资源以完成任务的问题。
- **模型中自我纠错能力的涌现**：关于模型的自我纠错行为是涌现的还是编程预设的展开了辩论，一位参与者提到了 *o1/r1/QwQ 生成的“数千个 CoT Token”*。
   - 这引发了关于使用 MCTS/PRMS 等过程来训练模型有效推理轨迹的讨论。
- **对奖励塑造（Reward Shaping）与博弈（Gaming）行为的担忧**：提出了奖励模型中的博弈行为话题，一位成员质疑惩罚措施是否能有效解决这一问题。
   - 大家达成共识，认为奖励塑造非常复杂，需要仔细考虑以避免模型出现非预期的行为。
- **推理轨迹生成的探索**：成员们对生成有效的推理轨迹表示不确定，有人建议利用人类数据和巧妙的 Prompting 技术。
   - 对话总结认为，有效的推理可能需要比典型的 Instruction-tuning 技术更多的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/natolambert/status/1856777789313355929)">来自 Nathan Lambert (@natolambert) 的推文</a>：能力解锁……“等等，让我们再次仔细地一步步思考”是 RL，不是 o1</li><li><a href="https://cookbook.openai.com/examples/o1/using_reasoning_for_routine_generation">使用推理进行例行程序生成 | OpenAI Cookbook</a>：使用 OpenAI API 构建应用的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1325967460481896559)** (9 messages🔥): 

> `MeCo Method, Contextual Artifacts in LM Training, Danqi's Contributions, Physics of LLM Papers, Impact of Timestamps` 


- **MeCo 方法简化了 LM 预训练**：[MeCo](https://x.com/gaotianyu1350/status/1876303908899037642) 的引入，通过使用元数据条件化（metadata conditioning）随后进行冷却（cooldown），提供了一种*极其简单*的 LM 预训练方法，即在文档前添加来源 URL。
   - 一位成员最初觉得这个概念*荒谬*，但随后承认 URL 可能提供了关于所用语言的重要上下文。
- **上下文人工制品（Contextual artifacts）可能增强语言模型**：关于可以添加哪些*上下文人工制品*来改进模型的讨论随之展开，一位用户建议时间戳可能会影响模型理解时间的能力。
   - 这引发了与 WRAP 技术的比较，提出*聚合（globbing）*人工制品可能是一种富有成效的方法。
- **Danqi 获得赞赏**：成员们表达了对 *Danqi* 的钦佩，其中一人表示他们“喜欢那些行之有效的疯狂想法”，反映了对创新理念的积极接受。
   - 另一位成员指出，Allen-Zhu 此前曾在“LLM 物理学（physics of LLM）”论文中强调过相关点，表明了对该主题的深度参与。
- **引用 Part 3 视频以获取额外背景**：一位成员引用了 *Part 3 视频*，指出它提供了与所讨论概念相关的额外见解。
   - 这一建议暗示了一个协作学习过程，因为用户寻求利用多媒体资源在之前的讨论基础上进行构建。



**提及的链接**：<a href="https://x.com/gaotianyu1350/status/1876303908899037642">来自 Tianyu Gao (@gaotianyu1350) 的推文</a>：介绍 MeCo（元数据条件化后冷却），这是一种极其简单的方法，通过简单地在训练文档前添加来源 URL 来加速 LM 预训练。https://arxiv.org/abs/2501.01...

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1325969803638210700)** (1 messages): 

> `Agents and Labor Policy, National Security, Model Shops and AI Proliferation` 


- **探索 Agent 在劳工政策中的角色**：一位成员表示需要关于如何与**工作场所中的 Agent**互动的文章，并强调目前的讨论主要集中在它们的**扩散**上。
   - *情绪倾向于怀疑，*但现实是，如果证明 Agent 的表现仅达到**人类能力的 20%** 且成本极低，资本可能会为了效率而**部署 Agent**。
- **国家安全视角的覆盖已饱和**：有人指出，现有讨论中关于 Agent 的**国家安全视角**似乎已经得到了充分覆盖。
   - 该成员认为应更多地强调劳工背景下的实际影响以及**与 Agent 的协作动态**。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1325978343862177874)** (22 messages🔥): 

> `Training High-Parameter LLMs, Deepspeed Zero-3 Memory Issues, Gradient Checkpointing, Ethics Dataset Evaluation, Learning and Contribution in AI` 


- **训练 7B LLM 的显存担忧**：一位成员报告称，尽管冻结了大部分权重且只有几百万个可训练参数，训练他们的 **7B LLM** 仍消耗了约 **35GB** 显存。
   - 他们怀疑训练期间的高显存占用可能是由于缺乏 Gradient Checkpointing 以及梯度和优化器状态带来的不必要开销。
- **DeepSpeed Zero-3 未能减少显存**：另一位成员表达了挫败感，称他们尝试使用 **DeepSpeed Zero-3** 进行模型分片（sharding）并未带来任何显存减少。
   - *他们声称并不完全理解 **DeepSpeed**，但抱着希望进行了尝试，结果却没看到任何好处*。
- **Gradient Checkpointing 可能至关重要**：社区成员讨论了使用 **Gradient Checkpointing** 来缓解训练期间显存占用问题的潜在必要性。
   - 正如所言，训练期间梯度、优化器状态和高精度模型副本会产生显著的显存开销。
- **对在 Ethics 数据集上评估 Pythia 的兴趣**：一位成员询问是否有人在 [Ethics 数据集](https://huggingface.co/datasets/hendrycks/ethics)上评估过 **Pythia**。
   - 这突显了对在伦理背景下评估模型性能的持续兴趣。
- **对学习 AI 技术的渴望**：一位成员对他人如何获取 AI 知识表示好奇，并表达了为社区做出贡献的愿望。
   - 这暗示了对社区参与和 AI 协作学习的持续兴趣。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1325979633614590023)** (9 条消息🔥): 

> `Cerebras AI 资助提案、LLM 推理感知微调、上下文学习表示、用于神经网络训练的 Tensor-GaLore、Cut Cross-Entropy 损失函数方法` 


- **Cerebras AI 征集研究提案**：Cerebras 正在邀请大学教师和研究人员响应一项[提案征集 (RFP)](https://cerebras.ai/blog/grantfrp)，旨在推动生成式 AI (Generative AI) 领域的发展。
   - 他们强调其目标是支持利用其**第三代晶圆级引擎 (Wafer Scale Engine)** 的创新研究，从而提供显著的性能优势。
- **LLM 的新型微调范式**：最近的一篇论文提出了一种**推理感知微调 (inference-aware fine-tuning)** 范式，该范式通过 **Best-of-N (BoN)** 策略进行评估，优化模型以在推理期间获得更好的性能 [查看 PDF](https://arxiv.org/abs/2412.15287)。
   - 作者展示了该方法在使用模仿学习和强化学习方法交替生成最佳响应和多样化响应方面的有效性。
- **LLM 创建上下文表示**：一篇题为 *In-Context Learning of Representations* 的论文讨论了 LLM 如何形成**“上下文表示” (in-context representations)** 以匹配给定的任务结构 [链接](https://x.com/corefpark/status/1875929881856573905)。
   - 研究结果表明，LLM 在**大上下文极限 (large context limit)** 下表现出显著的行为适应，展示了它们如何管理任务对齐。
- **使用 Tensor-GaLore 进行高效训练**：Tensor-GaLore 引入了一种利用高阶张量权重进行高效**神经网络训练**的方法，增强了内存效率 [查看 PDF](https://arxiv.org/abs/2501.02379)。
   - 它特别专注于**高阶参数空间**内的优化，展示了在求解复杂**偏微分方程 (partial differential equations)** 方面的优势。
- **通过 Cut Cross-Entropy 节省显存**：一次讨论强调了 **Cut Cross-Entropy (CCE)** 方法是解决 LLM 显存占用增加的方案，该方法通过优化交叉熵计算而不实例化所有 Logits 来实现 [查看 PDF](https://arxiv.org/abs/2411.09009v1)。
   - 这种新颖的方法显著降低了训练期间的全局显存消耗，对于具有庞大词汇量的大型模型尤为重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.15287">Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models</a>：最近的研究表明，有效利用推理时计算对于从大语言模型 (LLM) 中获得更好的性能至关重要。在这项工作中，我们提出了一种新型的推理...</li><li><a href="https://arxiv.org/abs/2501.02379">Tensor-GaLore: Memory-Efficient Training via Gradient Tensor Decomposition</a>：我们提出了 Tensor-GaLore，这是一种利用高阶张量权重进行神经网络高效训练的新方法。许多模型，特别是用于科学计算的模型，采用了张量参数...</li><li><a href="https://arxiv.org/abs/2411.09009v1">Cut Your Losses in Large-Vocabulary Language Models</a>：随着语言模型变得越来越大，它们的词汇量也在增加。这使得 LLM 在训练期间的显存占用不成比例地转移到了一个层上：损失计算中的交叉熵...</li><li><a href="https://x.com/corefpark/status/1875929881856573905">来自 Core Francisco Park @ NeurIPS2024 (@corefpark) 的推文</a>：新论文！“In-Context Learning of Representations”。在大上下文极限下，LLM 的内部表示会发生什么？我们发现 LLM 会形成“上下文表示”来匹配结构...</li><li><a href="https://cerebras.ai/blog/grantfrp">宣布 Cerebras 推理研究资助 - Cerebras</a>：AIBI (AI Bot Interviewer) 是第一个端到端的 AI 面试机器人，提供无缝、实时的面试体验。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1325993594900971551)** (112 条消息🔥🔥): 

> `Chat Templates 与非 Chat Templates 的评估对比、多选题的 Logprob 分析、Instruct 模型性能、Arc Challenge 与生成任务、Chat 格式对模型响应的影响` 


- **Chat Templates 可能会阻碍性能**：一位成员指出，在使用 **chat templates** 进行评估时，多选题的性能似乎比不使用模板时要低，特别提到 **L3 8B base** 在没有模板的情况下得分高得多。
   - 据推测，Chat 格式会使模型偏向于对话式响应，从而降低了其输出精确字母答案的能力。
- **Logprobs 指出潜在的输出问题**：讨论表明，在多选题任务中使用 **logprobs** 时，如果以 **chat format** 进行查询，模型可能会对所有有效答案产生非常低的概率。
   - 建议分析不受限的 logprobs，以查看模型在 Chat 格式的约束下是否难以生成正确答案。
- **Instruct 模型可能存在性能差异**：关于为什么 **instruct models** 在多选题任务中得分较低存在争论，有建议认为受限的输出空间是导致这一问题的原因。
   - 成员们考虑了一个假设，即 **chat format** 中的指令导致了在对话任务中表现更好，但阻碍了结构化响应的精确输出。
- **用于生成任务的 Evaluation Harness**：一位成员提出了创建一个训练数据集的想法，该数据集使用 system prompt 来严格规定多选题的任务响应格式，并将其集成到 **evaluation harness** 中。
   - 他们指出，在结构化输出之前允许模型有一定的自由度来生成答案，在某些语境下可能会产生更好的结果。
- **提取答案的挑战**：人们担心在对话格式上训练的模型是否能有效地输出简洁的答案，并建议测试各种配置可能会揭示差异。
   - 成员们有兴趣让模型在计算 logprobs 之前自由生成，这可能会提高模型在结构化任务中的有效性。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326075339687854091)** (5 条消息): 

> `Llama2 Checkpoints 转换、NeoX 中的 Optimizer 支持、Config 中的 Scheduler 语法、混合精度 Loss Scaling、Pythia Batch Size 计算` 


- **Llama2 Checkpoints 兼容性**：一位用户询问在 NeoX 中使用 Llama2 配置训练保存的 checkpoints 是否可以直接转换为 Hugging Face 格式。
   - *未提供直接回复，表明对兼容性存在不确定性。*
- **AdamW 与 Lion Optimizer 支持**：关于 NeoX 训练脚本中缺少 AdamW 而包含 Lion 的问题引起了疑问，促使大家查看了 [training.py 文件](https://github.com/EleutherAI/gpt-neox/blob/main/megatron/training.py#L1036)。
   - *用户对这一差异表示惊讶，强调了 AdamW 在类似场景中的流行程度。*
- **Scheduler 语法咨询**：一位成员提出了一种基于 optimizer 字典传递 scheduler 字典的建议语法，特别是针对 WarmupCosineLR 配置。
   - *讨论中未对该提议的配置提供反馈或确认。*
- **关于 BF16 Loss Scaling 的疑问**：用户想知道在 NeoX 中包含 BF16 的 loss scale 参数是否有好处，并寻求一个使用 BF16 混合精度的示例配置。
   - *没有分享任何示例或参考资料来澄清这一实现方面的问题。*
- **Pythia 全局 Batch Size 计算**：寻求关于 Pythia 全局 batch size 计算的澄清，考虑到在 128 个 GPU 上有效 batch size 为 16，相当于全局 2048。
   - *关于处理的总 token 数是否与文档陈述的 2M tokens 相矛盾存在困惑。*



**提到的链接**：<a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/training.py#L1036">gpt-neox/megatron/training.py at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1325932912582852742)** (138 条消息🔥🔥): 

> `OpenRouter 支付问题、模型性能担忧、DeepSeek V3 可靠性、使用加密货币支付、LLM 在游戏开发中的局限性` 


- **OpenRouter 支付问题持续存在**：用户报告 OpenRouter 持续出现支付问题，包括使用虚拟卡时多次交易被拒，以及支付网关出现困难。
   - 一位用户对他们的卡不再支持 OpenRouter 购买表示沮丧，建议转向加密货币支付。
- **对模型性能的担忧**：用户讨论了 Lambda 的 Hermes 405b 频繁崩溃的问题，并指出尽管存在问题，状态指示灯仍显示为绿色。
   - 还有人提到感知到 DeepSeek V3 的性能较慢，一些用户将其归因于高需求。
- **DeepSeek V3 可靠性问题**：DeepSeek V3 正在经历可靠性方面的担忧，特别是在高输入条件下，影响了跨平台的功能。
   - 一位用户指出，该问题似乎在 DeepSeek 和 OpenRouter API 上都很普遍。
- **探索加密货币支付**：几位用户讨论了使用加密货币代替传统支付方式的可行性，强调了其在某些地区的优势。
   - 对于在菲律宾面临支付困难的用户，Trust Wallet 和其他提供商被建议作为潜在选择。
- **LLM 在游戏创作中的局限性**：用户探讨了当前 LLM（如 O3 和 GPT-5）在创建更复杂的游戏设计（相比于简单的 2D 游戏）方面的局限性。
   - 大家达成共识，虽然简单的游戏有可能被生成，但由于组织上的困难，更复杂的设计仍然具有挑战性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/">Home</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://github.com/cline/cline/issues/1157">API Request loading indefinitely, not completing. · Issue #1157 · cline/cline</a>: 发生了什么？API 请求开始无限期加载，从未完成。我正在使用 Deepseek v3。它在约 2 小时内运行完全正常，然后突然开始出现这种情况，在任何聊天窗口中...</li><li><a href="https://github.com/googleapis/python-genai/pull/84">feat: Support API keys for VertexAI mode by copybara-service[bot] · Pull Request #84 · googleapis/python-genai</a>: feat: 为 VertexAI 模式支持 API 密钥
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1325948302973468842)** (73 条消息🔥🔥): 

> `Aider 在编程中的效用，O1 Pro 的问题，在 Aider 旁使用 Continue.dev，高效 AI 交互技巧，命令执行的挑战` 


- **Aider 作为编程助手表现出色**：许多用户称赞 Aider 协助处理复杂编程任务的能力，指出它更像是一位编程导师而不仅仅是一个工具。
   - 一位用户强调了阅读 Aider 提示词以及有效利用 /ask 命令来优化输出的重要性。
- **O1 Pro 存在不一致性**：一些用户报告了 O1 Pro 的问题，包括进度条缺失和解决时间不理想。
   - 尽管存在挫败感，一些用户仍然偏好 O1 Pro 而非其他替代方案，而另一些人则认为两者可以协同使用以获得最佳效果。
- **将 Continue.dev 与 Aider 集成**：用户开始探索将 Continue.dev 作为 Aider 的补充工具，特别是用于更快速的交互和任务管理。
   - 一位用户分享说，这种组合不仅提高了生产力，还有助于高效管理更重大的编程任务。
- **AI 驱动编程的有效策略**：成员们讨论了在编程中利用 AI 的策略，建议通过编写详细的提示词来增强响应效果。
   - 一位用户展示了如何通过迭代优化查询，将 Sonnet 和 O1 Pro 结合使用以产生出色的结果。
- **命令执行中的挑战**：关于 Aider 执行 git 命令的问题引起了关注，用户指出虽然命令在工具外部运行正常，但在工具内部却出现了问题。
   - 在排查故障后，一位用户发现系统更新解决了他们在使用 Aider 时遇到的命令路径可见性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>：将图像和网页添加到 Aider 编程聊天中。</li><li><a href="https://jina.ai/reader/">Reader API</a>：读取 URL 并搜索网络，以实现更好的 LLM 落地（grounding）。</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Aider in a ReAct loop</a>：ReAct 循环中的 Aider。通过在 GitHub 上创建一个账户来为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://github.com/gorilla-llm/gorilla-cli">GitHub - gorilla-llm/gorilla-cli: LLMs for your CLI</a>：适用于 CLI 的 LLM。通过在 GitHub 上创建一个账户来为 gorilla-llm/gorilla-cli 的开发做出贡献。</li><li><a href="https://r.jina.ai/https://docs.astro.build/en/guides/routing/">{title}</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1325960078528090182)** (50 messages🔥): 

> `Aider prompt caching, Custom LLM usage with Aider, Terminal display issues, Color themes for terminal, Troubleshooting file updates in Aider` 


- **Aider 支持 Prompt Caching 以实现更快编码**：Aider 支持 Prompt Caching，以在运行命令时增强性能，正如 [文档](https://aider.chat/docs/usage/caching.html) 中所强调的。用户可以启用缓存，并通过特定的命令行选项和设置进行管理。
- **在 Aider 中集成自定义 LLM**：用户正在讨论如何在 Aider 中使用自定义 LLM，特别是通过注册和实例化自定义类。一些用户通过在模型名称前加上 'custom/' 并正确配置 API 设置取得了成功。
- **解决 Terminal 显示问题**：一位用户报告了 Aider 的输出无法适应小型 Terminal 窗口，导致文本排列混乱的问题。建议检查 Terminal 配置，并确保视图可以有效地容纳所有输出。
- **探索 Windows Terminal 的颜色主题选项**：有关于改进 Windows Terminal 中颜色主题可读性的咨询，特别是关于深色模式使某些颜色难以看清的问题。用户正在寻求建议，以找到一个能增强可见性的合适调色板。
- **排查 Aider 文件更新问题**：一位用户遇到了 Aider 尽管提供了上下文和指令却无法更新文件的问题。建议检查错误、重现问题，或尝试通过 `aider --install-main-branch` 使用最新的 main branch。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://0.0.0.0:8000"">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范 (Specifying coding conventions)</a>: 让 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider 支持 Prompt Caching，以节省成本并加快编码速度。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>: 为 LLM 配置高级设置。</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>: 关于 aider 所有设置的详细信息。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326270033378152530)** (2 messages): 

> `Aider workflow adaptation, LLM-guided interviews` 


- **为 Aider 适配优秀的工作流**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=XWJGm3y207A)，展示了一个可以适配到 Aider 功能中的优秀工作流。
   - 该工作流侧重于增强开发编码任务时的沟通和效率。
- **LLM 访谈增强编码 Prompt**：一位用户对一个引导 **LLM 访谈**用户以创建规范的 Prompt 表示兴奋。
   - 这种方法旨在反馈到编码 Prompt 中，提供**结构化**且有效的编码指导。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1325933958579355649)** (14 条消息🔥): 

> `NBA 比赛回顾, 虚拟体育播报中的 AI, 数据源与引用规范, 用于合同审查的 AI, NotebookLM 的功能` 


- **NBA 比赛回顾迎来技术升级**：一位成员建议使用 NotebookLM 将比赛回顾与 NBA 和 NFL 的精彩片段叠加，并强调了其成本效益。
   - 他们分享了一个 [YouTube 视频](https://youtu.be/D7qZ2VphetU)，展示了如何通过这种概念大规模生产品牌内容。
- **关于使用可靠数据源的讨论**：成员们讨论了内容溯源，其中一人使用了 **Britannica**，另一人询问是否来自 **Wikipedia**。
   - 一位成员确认使用了**单一数据源**，而另一位成员则在寻求一个能够准确引用相关部分的 System Prompt。
- **用于合同审查的 AI 与数字劳动力**：一位成员强调了 AI 和虚拟法律助手如何减轻**合同“修订”（redlining）**的负担，这一过程通常既耗时又昂贵。
   - 通过使用 Avatar 与利益相关者互动，该过程变得更具交互性，简化了准备工作并促进了理解。
- **NotebookLM 增强协作学习**：NotebookLM 正在通过将内容组织到特定主题的笔记本中来改变培训领域，从而实现更好的研究与协作。
   - 它支持小组项目，并作为一个持续更新的资源，使参与者能够更有效地参与其中。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.akashq.com/post/4cc67748-6af9-4083-b63c-7a0d366dc01d">What happened on Jan 7?</a>: 1 月 7 日发生了什么？ —— This Day in History</li><li><a href="https://www.akashq.com/post/b1eeb736-890a-4306-bc56-2ac605e739d2">What happened on Jan 6?</a>: 1 月 6 日发生了什么？ —— This Day in History
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1325927740108701726)** (86 条消息🔥🔥): 

> `NotebookLM 使用限制、NotebookLM Plus 功能、Audio Overview 长度、缺失功能、Google Workspace 问题` 


- **NotebookLM 使用限制导致速度变慢**：用户对可能影响 NotebookLM 性能的每日限制表示担忧，指出在长时间使用后速度会变慢。
   - 一位成员建议查看 [官方支持页面](https://support.google.com/notebooklm/answer/15678219?hl=en) 以获取更多信息。
- **解释 NotebookLM Plus 功能**：NotebookLM Plus 提供增强功能，例如能够上传多个来源（包括 PDF 和 YouTube 链接）并生成摘要。
   - 其他高级功能包括对笔记本和查询的更高限制，正如关于 Plus 订阅价值的讨论中所强调的那样。
- **Audio Overview 长度方面的挑战**：多位用户报告在控制生成的 Audio Overview 长度方面存在困难，并对生成过长的输出表示沮丧。
   - 建议的解决方法包括删除不需要的来源，以便更好地管理概览的焦点。
- **缺失的功能与特性**：有报告称，从选定文本生成 AI 建议问题的功能丢失，影响了用户的工作流程。
   - 在持续更新的过程中，用户寻求有关预期功能状态的解决方案和建议。
- **Google Workspace 许可证说明**：关于在 Google Workspace 账户中访问 NotebookLM Plus 功能所需的适当许可证出现了疑问。
   - 参与者讨论了特定附加许可证的必要性以及如何激活它们，并参考了支持页面。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/icymi-gemini-advanced-users-now-have-access-to-notebooklm-plus/">Gemini Advanced 用户现在可以访问 NotebookLM Plus</a>: 探索 Google 的 NotebookLM Plus，这是一款具有扩展功能和自定义选项的 AI 驱动研究工具。通过高级功能和更高的使用限制提高生产力。</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=Where%20to%20get%20NotebookLM%20Plus%C2%A0">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638718328734806536-3512091001&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?sjid=11787293163246227446-AP&visit_id=638717415901875295-2373678412&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://support.google.com/a/answer/6043385?hl=en&co=DASHER._Family%3DBusiness">比较 Google Workspace 版本 - 商务版 - Google Workspace 管理员帮助</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1325937907034357761)** (78 messages🔥🔥): 

> `Nous Forge API 更新，RTX GPU 性能对比，NVIDIA Project DIGITS，AI 机器人行为调整，USB-C 网络方案` 


- **Nous Forge API 阶段结束**：[Nous Forge API](https://forge.nousresearch.com/) 的 Beta 阶段于近期结束，用户仍可订阅更新，以了解利用各种模型的推理引擎（reasoning engines）的配置信息。
   - 该 API 为涉及 Hermes、Claude、Gemini 和 OpenAI 模型的复杂任务提供了先进的推理能力和透明度。
- **RTX GPU 性能争论**：成员们讨论了 RTX 3090 与新款 RTX 5070 的价格和性能，对 NVIDIA 声称能以更低价格提供 4090 级别性能的说法表示怀疑。
   - 讨论中提出了在缺乏实质性基准测试（benchmarks）的情况下对比 AI 性能的担忧，并指出 NVIDIA 所谓性能提升的策略是基于 AI 纹理压缩技术。
- **NVIDIA 发布 Project DIGITS**：[NVIDIA Project DIGITS](https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips) 正式亮相，搭载了 Grace Blackwell 超级芯片，旨在以更平易近人的门槛提供高性能 AI 计算。
   - 这一进展旨在为研究人员和学生普及 AI 超级计算机的使用，使模型部署和开发变得更加可行。
- **调整 AI 角色行为**：频道讨论了调整 AI 模型行为的技术，以防止角色产生过度焦虑的响应，并针对系统提示词（system prompts）提出了建议。
   - 讨论包括了旨在建立自信的提示词示例，以及针对 AI 日志中出现问题的幽默调侃。
- **USB-C 作为网络解决方案**：两台 PC 之间的 USB-C 连接可以实现高速（10-20Gbps）组网，是一种高性价比的选择。
   - 参与者分享了关于选择兼容线缆以获得最佳性能的见解，并提到了在扩展性方面的潜在限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forge.nousresearch.com/">Forge Reasoning API by Nous Research</a>：Nous Research 开发的 Forge 推理 API</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips">NVIDIA Puts Grace Blackwell on Every Desk and at Every AI Developer’s Fingertips</a>：CES——NVIDIA 今日发布了 NVIDIA® Project DIGITS，这是一款个人 AI 超级计算机，为全球 AI 研究人员、数据科学家和学生提供 NVIDIA Grace 的强大性能...</li><li><a href="https://arxiv.org/abs/2411.05007">SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>：扩散模型已被证明在生成高质量图像方面非常有效。然而，随着这些模型变得越来越大，它们需要更多的内存并面临更高的延迟...</li><li><a href="https://x.com/meeix_/status/1876465349278994804">Meeix (@meeix_) 的推文</a>：RTX 5070 仅售 549 美元却能提供“4090 的性能”？？？？兄弟，NVIDIA 这次是动真格的了</li><li><a href="https://x.com/drjimfan/status/1876516972512559170?s=46">Jim Fan (@DrJimFan) 的推文</a>：介绍 NVIDIA Cosmos，一个开源、开放权重的视频世界模型。它在 2000 万小时的视频上进行训练，参数量从 4B 到 14B 不等。Cosmos 提供两种版本：diffusion（连续 token）和...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">Nous Research (@NousResearch) 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326177295320219752)** (1 messages): 

> `声誉担忧，隐私问题，利润驱动的动机` 


- **对声誉和隐私的担忧**：一位成员表示，许多组织虽然已经**运营了相当长的时间**，但在**隐私**方面并未建立起良好的声誉。
   - 他们强调，这导致了对其意图的各种**怀疑**，特别是关于**利润驱动的动机**。
- **组织中利润驱动的动机**：讨论中提出了对组织主要受**利润动机**驱动的担忧，这影响了他们对用户隐私和安全的承诺。
   - 讨论强调了当利润优先于用户保护时所产生的**不信任感**。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326334069180731457)** (3 条消息): 

> `Structure of Neural Embeddings, MiniMind Training Pipeline, MiniMind Model Overview` 


- **神经嵌入结构的见解**：一篇博客文章探讨了 **neural latent spaces** 的结构，并提出了几个原则，例如 **Manifold Hypothesis**（流形假设），即高维现实世界数据存在于低维流形中。其他探索还包括 **Hierarchical Organization** 的概念以及与神经网络特征相关的 **Linear Hypothesis**。
   - 为了深入理解，可以阅读 [Manifolds and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)、[Visualizing Representations](https://colah.github.io/posts/2015-01-Visualizing-Representations/) 和 [NLP Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) 等链接。
- **MiniMind 的完整训练流水线**：**MiniMind** 项目包含一个小型语言模型的完整训练流水线，包括 Pretraining、SFT 和 DPO，适用于 2x RTX3090 GPU，详见 [English README](https://github.com/jingyaogong/minimind/blob/master/README_en.md)。该项目允许任何人在大约 3 小时内训练一个仅有 26.88M 参数的模型。
   - 这一开源倡议不仅简化了训练过程，还为 LLM 初学者提供了教程，有助于理解模型训练和微调技术。
- **MiniMind 模型能力**：**MiniMind** 模型因其轻量级结构而备受推崇，其大小约为 **GPT-3 的 1/7000**，使普通 GPU 用户也能进行快速 Inference 和训练。该项目包含简化模型结构、数据集预处理、有监督 Pretraining、SFT、LoRA 微调和 DPO 的代码。
   - 此外，它还支持将能力扩展到 **sparse models with mixed experts** (MoE) 和 **multi-modal vision language models**（如 MiniMind-V 项目所示），丰富了模型探索的资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://seanpedersen.github.io/posts/structure-of-neural-latent-space">Structure of Neural Embeddings</a>: 未找到描述</li><li><a href="https://jingyaogong.github.io/minimind/">MiniMind Project</a>: 未找到描述</li><li><a href="https://github.com/jingyaogong/minimind/blob/master/README_en.md">minimind/README_en.md at master · jingyaogong/minimind</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. 通过在 GitHub 上创建一个账户来为 jingyaogong/minimind 做出贡献。</li><li><a href="https://github.com/jingyaogong/minimind">GitHub - jingyaogong/minimind: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！</a>: 「大模型」3小时完全从0训练26M的小参数GPT，个人显卡即可推理训练！. 通过在 GitHub 上创建一个账户来为 jingyaogong/minimind 做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1325922647728586843)** (58 条消息🔥🔥): 

> `Perplexity 性能问题、隐私与广告担忧、用户界面反馈、SOC 2 合规性查询、订阅与使用问题` 


- **Perplexity 面临大范围速度变慢**：多位用户报告 **Perplexity** 出现严重延迟，有人提到移动端和浏览器端的响应都需要数分钟。
   - 用户建议使用在记事本中输入并粘贴回复等临时方案，因为应用难以跟上处理速度。
- **个性化广告引发隐私担忧激增**：一位用户表达了在 Perplexity 搜索特定健康症状后，在 **Instagram** 上收到针对性广告的担忧，担心应用读取了聊天内容。
   - 这引发了关于不登录使用应用以获得更好隐私保护的讨论。
- **关于 SOC 2 合规性的查询**：用户要求提供关于 Perplexity 的 **SOC 2 合规性**信息，一位用户表示他们从支持部门获得了信任中心（trust center）的详细信息。
   - 另一位用户声称许多 AI 都不合规，并反驳说他对 AI 服务的合规性感到惊讶。
- **对 Perplexity 用户界面的反馈**：用户批评了应用的 **“立即购买” (shop now)** 功能，认为它无法让用户妥善评估卖家，从而损害了产品搜索体验。
   - 一些人建议改进功能，例如 Pro Searches 的**切换选项**，并提供更清晰的使用限制反馈。
- **订阅与使用量不匹配的困惑**：一位用户收到了关于可用 **Pro Searches** 的矛盾信息，称他们每四小时只能获得三次，而不是之前预期的五次。
   - 另一位用户提到，应该更好地记录选项以避免混淆，特别是由于最近订阅模式的变化。



**提到的链接**：<a href="https://trust.perplexity.ai/">Trust Center | Powered by Drata</a>：准备好将信任转化为您的竞争优势了吗？快速通过安全审查，并通过 Trust Center 快速共享关键安全信息。

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1325939202671509535)** (10 条消息🔥): 

> `NASA 的月球微型任务、AgiBot 的人形机器人训练数据集、Microsoft 的 AGI 开发、Disney 的新项目、Gen Z 的 Looksmaxxing 趋势` 


- **NASA 启动月球微型任务**：今天，**NASA** 披露了其**月球微型任务**的细节，旨在增强月球探测技术，详情可在此处[查看](https://www.perplexity.ai/search/the-amethyst-tablet-pdf-wSFKHIw3R9CFnbPRI3Vfxw#0)。
   - 该任务包括可能有助于未来载人登月任务的创新技术。
- **AgiBot 发布人形机器人训练数据集**：**AgiBot** 推出了一套新的**人形机器人训练数据集**，有望提升 AI 在拟人化机器人领域的水平。更多信息可在[此处](https://www.youtube.com/embed/V8A6EdbPdGU)的视频中找到。
   - 该数据集预计将有助于提高人形机器人的交互和任务执行能力。
- **Microsoft 巨额 1000 亿美元 AGI 投资**：**Microsoft** 因其在 **AGI 开发**上的 **1000 亿美元**投资而成为头条新闻，这标志着在该领域的重大投入。详情见讨论[链接](https://www.perplexity.ai/search/why-is-it-called-donating-plas-8jLeogYsRZW4.OYo.eObhA)。
   - 这一举动被视为在通往先进 AI 的竞赛中增强其技术能力的关键。
- **Disney 启动新计划**：**Disney** 将启动旨在吸引年轻观众的令人兴奋的新项目，更新详情见[此处](https://www.perplexity.ai/search/disney-zMu663WHTbirL8TOhHD0.g)。
   - 这些举措是 Disney 保持在 **Gen Z** 中影响力的战略的一部分。
- **探索 Gen Z 的 Looksmaxxing 趋势**：**Gen Z** 中出现了一种名为 **looksmaxxing** 的新趋势，专注于提升个人外表，更多讨论见[此处](https://www.perplexity.ai/page/gen-z-s-looksmaxxing-trend-R4cK0xe0R3.lhWVq5V.p6A)。
   - 这一趋势反映了对个人形象和社交媒体存在感的更广泛文化关注。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1326189835081351199)** (1 条消息): 

> `12 月 19 日邮件，关于在线模型的担忧` 


- **成员询问 12 月 19 日的邮件**：用户正在询问他们在 **12 月 19 日** 收到的一封邮件，希望能分享关于其内容的经验。
   - *有人感叹道*，“如果他们只保留在线模型，那真是太遗憾了！”
- **对模型可用性的担忧**：一位成员表示担心，这一决定可能会导致 **在线模型的独占性**，而不是继续支持多样化的选择。
   - 他们消息中链接的附图引发了关于此问题的进一步讨论。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326213923061370992)** (66 条消息🔥🔥): 

> `AI21 Labs Token，诈骗担忧，社交媒体沟通，Token 审计` 


- **AI21 Labs Token 引发诈骗担忧**：成员们反复询问 **AI21 Labs Token** 是否合法，许多人断言它看起来像是一个 **诈骗 (scam)**。一位用户强调了该 Token 可疑的活动，警告他人 *远离*。
- **呼吁官方沟通**：担忧的成员敦促 AI21 Labs 公开澄清其对该 Token 的立场，要求在 **Twitter** 上发布帖子，使公司与所谓的诈骗撇清关系。一位用户表达了挫败感，称 *发推文警告并不需要任何成本*。
- **平台对加密货币的立场**：AI21 代表确认该 Token **不隶属于** 公司，并警告说进一步讨论加密货币可能会导致从频道中 **被封禁 (bans)**。他们传达了情况已上报至其 **安全团队 (security team)**。
- **Token 正在接受审计**：尽管声称有审计，一些用户仍持怀疑态度，称该 Token 与 **pumpfun** 相关，引发了进一步的 **诈骗** 怀疑。其他人建议该 Token 的持有者分布看起来有问题，表明存在潜在风险。
- **关于 Token 的总体情绪**：总体而言，社区情绪倾向于怀疑，有言论指出许多人认为该 Token 可能已经 **跑路 (rugged)** 或与 **诈骗** 有关。用户分享了对项目合法性的担忧，强调需要谨慎。



**提及的链接**: <a href="https://www.dextools.io/app/es/token/ai21labs?t=1736289130032&maker=81wGWkys3bHxktg8rXPGbW7xuJ8479RALZBBH55Jz54H">DEXTools</a>: DEXTools，通往 DEFI 的门户，提供实时图表、历史记录和来自区块链的所有 Token 信息。

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1325927612551528470)** (18 条消息🔥): 

> `AGI 与创新，AI 作为工具，AI 技术的最新进展，微调 AI 模型，RTX 5000 DLSS 4` 


- **AGI 对创新的影响**：成员们讨论了 **AGI (通用人工智能)** 将赋能还是颠覆人类创新，一些人认为它将增强批判性思维而非取代它。
   - *现在是创业的最佳时机，* 同时也提到了对使用 AGI 工具的大型企业竞争的担忧，导致了对依赖性的恐惧。
- **AI 作为协作工具**：一位成员将使用 AI 比作开车，强调使用 AI 可以增强能力，而不是削弱个人技能。
   - 这一观点强调，与 AI 的协作可以促进创新，因为用户可以在彼此的优势基础上进行构建。
- **RTX 5000 DLSS 4 的令人兴奋的进展**：展示了 **DLSS 4** 升级的 **RTX 5000** 引起了轰动，声称有 *三倍帧生成* 的提升。
   - 成员们正在兴奋地讨论其潜在的进步以及对游戏和图形性能的影响。
- **AI 模型微调经验**：关于用户是否在自己的文本或 Discord 对话上微调过 **LLaMA** 模型的问题引起了关注，激发了对个人经验的兴趣。
   - 一位成员确认他们在微调方面取得了成功，并评论说这相对容易，特别是在有结构化数据的情况下。
- **访问 AI API**：一位用户提到获得了对各种 AI 系统的 **无限访问权限**，包括来自 **OpenAI**、**Google** 和 **Anthropic** 的系统，以及它们的 API 模型。
   - 其他人也加入进来，建议在 GitHub 等平台上可以免费找到类似的资源。

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1325922562940866720)** (9 messages🔥): 

> `对话从 4o 转移到 1o，Mini O1 对比 GPT-4，Ubuntu 设置与 GPU 兼容性，O1 Pro 升级讨论` 


- **从 4o 到 1o 的对话转移选项呈灰色**：一位用户询问是否可以将他们的对话从 **GPT-4 (4o)** 转移到 **GPT-1 (1o)**，并指出该选项显示为灰色不可用。
   - *一位成员回答道*，如果使用了 **1o** 不支持的功能，则无法进行转移。
- **Mini O1 被认为比 GPT-4 更聪明**：一位用户询问 **Mini O1** 是否比 **GPT-4** 更聪明，另一位成员肯定了它的智能程度，但也提到它可能并非在所有场景下都表现更好。
   - 这为关于模型能力的持续争论提供了一个细致的视角。
- **用户分享用于 GPU 任务的 Ubuntu 配置**：一位用户分享了他们正在运行 **Ubuntu 24.04.1**，配备 **5800X CPU** 和 **6900XT GPU**，并询问有关使用 **GPU 4o Mini** 的资源。
   - *他们提到拥有 ROCP 6.3.1* 以及之前使用 **Ollama** 版本的经验。
- **关于升级到 O1 Pro 的讨论**：一位用户提出了关于升级到 **O1 Pro** 是否值得的问题，引发了对其价值的讨论。
   - 这突显了用户对新模型提供的功能和改进的持续关注。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1325942047319920700)** (15 messages🔥): 

> `Dall-E 中的 Midjourney SREF 提示词，JSON schema 响应，重试实现，提示词的风格命名` 


- **Dall-E 不支持 SREF 提示词**：一位成员询问是否可以在 Dall-E 中使用 Midjourney 的 --sref 功能，但另一位成员给出了明确的 **'No'** 回答。
   - 他们指出，虽然可以在提示词中命名风格，但通常无法产生预期的结果。
- **JSON schema 返回其自身的问题**：一位成员报告称，将模型设置为 JSON schema 模式时，有时会导致其在 **80%** 的时间内返回 schema 本身而不是有效的响应。
   - 尽管实施了重试机制，他们仍然遇到同样的问题，这表明指令可能存在模糊性。
- **对自我推广规定的担忧**：关于遵守频道禁止自我推广规定的讨论再次兴起，并提醒大家在此类事项上要小心。
   - 这引发了对对话中某些分享内容是否恰当的反思。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1325942047319920700)** (15 messages🔥): 

> `Dall-E 中的 Midjourney 提示词，JSON schema 返回问题，重试无效，提示词工程关注点` 


- **关于 Dall-E 中使用 Midjourney 提示词的疑问**：一位成员询问是否存在可以在 Dall-E 中复制 Midjourney --sref 风格的提示词。
   - 另一位成员确认没有直接的提示词可以实现这一点，建议只需命名风格并“寄希望于好运”。
- **JSON schema 响应的问题**：一位用户报告称，他们的模型在 80% 的时间内返回 JSON schema 本身而不是响应。
   - 尽管实施了重试，同样的问题依然存在，导致用户感到沮丧。
- **输入大小和模糊性担忧**：有建议认为，生成成功率低可能是由于输入大小和相对噪声造成的。
   - 有人指出，模糊的指令也可能导致该问题，强调了清晰度的必要性。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1325921454591709297)** (47 messages🔥): 

> `科学领域的 Foundation Models，NVIDIA 的 Cosmos，Vercel 的 AI SDK，鲸鱼保护中的 AI，FP4 之争` 


- **科学领域的 Foundation Models**：讨论了基础模型在科学研究中的应用。
- **NVIDIA 的 Cosmos**：涉及 NVIDIA 新发布的 Cosmos 相关讨论。
- **Vercel 的 AI SDK**：关于使用 Vercel AI SDK 进行开发的交流。
- **鲸鱼保护中的 AI**：探讨了 AI 技术如何应用于鲸鱼保护事业。
- **FP4 之争**：关于 FP4 数据格式及其在 AI 领域竞争的讨论。

- **探索科学领域的基础模型 (Foundation Models)**：一位成员分享了 [Metagene 1 论文](https://metagene.ai/metagene-1-paper.pdf) 的链接，引发了关于科学应用中基础模型的咨询。
   - 这种兴趣凸显了 AI 在专业领域中日益增长的相关性。
- **NVIDIA 发布 Cosmos 模型**：NVIDIA 推出了 [Cosmos](https://x.com/DrJimFan/status/1876516972512559170)，这是一个基于 **20M** 小时视频训练的开源视频世界模型，标志着机器人技术合成数据生成的飞跃。
   - 该模型具备扩散 (diffusion) 和自回归 (autoregressive) 两种生成模式，展示了 NVIDIA 在企业级 AI 领域的雄心。
- **Vercel 的 AI SDK 受到关注**：一位成员分享了使用 Vercel AI SDK 的经验，指出其在简单配置下的有效性，但批评其在与其他模型分层使用时存在 **过度抽象 (too much abstraction)** 的问题。
   - 这引发了关于 AI 工具集成中易用性与复杂性平衡的讨论。
- **AI 助力鲸鱼保护**：[Accenture](https://x.com/btibor91/status/1876630816199217208?s=46) 与悉尼大学成功合作开发了一套 AI 系统，能以 **89.4%** 的准确率检测小须鲸。
   - 这一创新简化了保护工作，将原本需要两周的手动流程转变为实时监测。
- **关于 FP4 指标的辩论**：成员们讨论了 NVIDIA 使用 FP4 指标的影响，以及对其相对于 FP8 等其他格式的比较价值的担忧。
   - 对话强调了在 GPU 性能宣称中需要明确的基准测试 (benchmarking) 标准。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/victortaelin/status/1876604185309048963?s=46">来自 Taelin (@VictorTaelin) 的推文</a>：所以，让我理清楚，RTX 5090 比 RTX 4090 贵了约 50%，CUDA 核心多了约 50%，能耗高了约 50%，性能提升了约 50%。真正的收益到底是什么？其他...</li><li><a href="https://x.com/DrJimFan/status/1876516972512559170.">来自 Jim Fan (@DrJimFan) 的推文</a>：介绍 NVIDIA Cosmos，一个开源、开放权重的 Video World Model。它在 2000 万小时的视频上进行训练，参数量从 4B 到 14B。Cosmos 提供两种版本：diffusion（连续 tokens）和...</li><li><a href="https://x.com/yuchenj_uw/status/1876680855630094725?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：我喜欢 Nvidia 和 Jensen，但他们展示数据的方式让我困扰：- 模糊的术语如 “AI TOPS” - 将 5090 的 FP4 与 4090 的 FP8 进行比较 - 展示 FP4 FLOPS 并声称一个 3,000 美元的盒子能运行一个 200B 的模型...</li><li><a href="https://x.com/chipro/status/1876681640505901266">来自 Chip Huyen (@chipro) 的推文</a>：我关于 agents 的 8000 字笔记：https://huyenchip.com//2025/01/07/agents.html 涵盖：1. agents 概述 2. AI-powered agent 的能力如何由其可访问的工具集决定...</li><li><a href="https://x.com/kevinhou22/status/1876498858819424307">来自 Kevin Hou (@kevinhou22) 的推文</a>：天哪，Jensen 在 CES 上谈到了我们 @Codeium 🤯 “Codeium。世界上每一位软件工程师，这将是下一个巨大的 AI 应用... 每个人都将拥有一个软件助手。如果不...”</li><li><a href="https://x.com/joaomdmoura/status/1876645077726216336">来自 João Moura (@joaomdmoura) 的推文</a>：⚡️重大公告⚡️ 科技巨头之一刚刚在 AI Agent 领域出手。NVIDIA 正在与 CrewAI 合作，助力其企业级 AI 部署。这就是为什么这很...</li><li><a href="https://x.com/btibor91/status/1876630816199217208?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：AI 的现实影响 - 悉尼大学和埃森哲构建了一个鲸鱼保护系统，使用 Claude 分析水下麦克风录音，并以 89.4% 的准确率检测小须鲸...</li><li><a href="https://x.com/joshua_xu_/status/1876707348686995605">来自 Joshua Xu (@joshua_xu_) 的推文</a>：我们将 HeyGen 的 avatar 模型与 Sora 无缝整合，结果确实是下一水平的。这可能是迄今为止最先进的会说话的 avatar 视频——表现优于真人演员...</li><li><a href="https://huyenchip.com//2025/01/07/agents.html">Agents</a>：Intelligent agents 被许多人认为是 AI 的终极目标。Stuart Russell 和 Peter Norvig 的经典著作《人工智能：一种现代方法》（Prentice Hall, 1995）定义了...</li><li><a href="https://mindy.com/">m@mindy.com</a>：未找到描述</li><li><a href="https://olly.bot">Olly | 个人 AI 助手</a>：你在 iMessage 中的个人 AI 助手。可通过 Siri 在你的 iPhone、Watch、Macbook 或 CarPlay 上使用。提供基于 Web 的回答、图像生成、文档聊天、提醒等功能。</li><li><a href="https://x.com/abhagsain/status/1876362355870994538">来自 Anurag Bhagsain (@abhagsain) 的推文</a>：上周，我们让 Devin 做了一个改动。它在横幅组件挂载时添加了一个事件，导致一周内产生了 660 万个 @posthog 事件，这将花费我们 733 美元。Devin 成本 500 美元 + 733 美元 = 1273 美元 😢👍L...</li><li><a href="https://github.com/mastra-ai/mastra">GitHub - mastra-ai/mastra: TypeScript AI 框架。</a>：TypeScript AI 框架。通过在 GitHub 上创建账号为 mastra-ai/mastra 的开发做出贡献。</li><li><a href="https://x.com/calcsam/status/1859664608992297082">来自 Sam Bhagwat (@calcsam) 的推文</a>：很高兴分享 @smthomas3、Abhi Aiyer 和我正在构建 Mastra，这是一个为未来百万名 AI 开发者准备的 TypeScript AI 框架：
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1326140999210106880)** (3 条消息): 

> `Modular 文档字体粗细，字体可读性` 


- **关于 Modular 文档字体粗细的讨论**：一位成员表示 **Modular 文档字体粗细** 感觉太细了，促使其他人也对该问题发表意见。
   - 另一位成员表示赞同，称自字体更改以来他们一直不喜欢现在的字体，并建议 **Modular** 应该考虑使用不同的字体或粗细。
- **对字体可读性的担忧**：多位成员强调了对 **Modular 文档** 字体自更改以来可读性的担忧。
   - 这导致了关于建议 **Modular** 探索替代字体粗细以增强用户体验的讨论。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326163405945438208)** (37 条消息🔥): 

> `Mojo Debugger, Mojo 项目结构, Mojo 中的静态列表, 使用运行时变量进行索引, 静态分析方法` 


- **Mojo Debugger 使用 LLDB**：成员们讨论了 Mojo 使用带有上游补丁的 **LLDB**，使其能够支持多种语言，并提到了一个涵盖该调试器的 [LLVM 会议演讲](https://www.youtube.com/watch?v=9jfukpjCPIg)。
   - 一位成员赞赏 Modular 采取的务实做法，即避免在已经解决的问题上浪费精力。
- **组织 Mojo 项目**：一位成员询问了如何组织他们的 Mojo 项目结构以及在测试期间如何导入模块，随后有人分享了来自 [GitHub](https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo) 的示例。
   - 另一位成员解释了如何使用 `magic run mojo test -I . tests` 运行测试，并参考了官方 [Mojo 测试文档](https://docs.modular.com/mojo/tools/testing#writing-unit-tests) 以获取更多细节。
- **在 Mojo 中索引静态列表**：一位用户了解到 **ListLiteral** 不能使用运行时变量进行索引，而应改用 **InlineArray**，并在其案例中成功实现。
   - 进一步的讨论澄清了 Tuple 和 List Literal 之间的区别，强调了 Tuple 是固定长度的，并且可以包含不同的类型。
- **Borrow Checker 讨论**：一位成员建议 Mojo 应该扩展静态分析方法，以超越 Rust 的 **Borrow Checker**，并建议优先专注于完成功能特性。
   - 他们对在 Mojo 中实现生产级结构表示不确定，表达了进一步探索文档的意愿。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/tree/v0.3?tab=readme-ov-file#include-numojos-path-for-compiler-and-lsp">GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo at v0.3</a>：NuMojo 是一个用于 Mojo 🔥 数值计算的库，类似于 Python 中的 NumPy。</li><li><a href="https://docs.modular.com/mojo/tools/testing#writing-unit-tests">Testing | Modular Docs</a>：测试 Mojo 程序。</li><li><a href="https://github.com/saviorand/lightbug_http/blob/main/tests/lightbug_http/test_client.mojo">lightbug_http/tests/lightbug_http/test_client.mojo at main · saviorand/lightbug_http</a>：适用于 Mojo 的简单快速的 HTTP 框架！🔥。</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1325999017548713985)** (7 条消息): 

> `AI-Plans Hackathon, 最佳 AI 模型, Command R+ 性能, AI Alignment 研究` 


- **AI-Plans 黑客松启动**：由 **AI-Plans** 主办的即将到来的黑客松专注于 **AI Alignment Evals**，定于 **1 月 25 日**举行。
   - 该活动旨在让参与者参与关键的 **Mechanistic Interpretability** 研究以及文献综述。
- **辩论任务的最佳模型**：围绕确定**最佳 AI 模型**展开了讨论，特别是与 **OpenAI O1** 的对比。
   - *Competent* 指出，选择很大程度上取决于用户的预期任务和应用场景。
- **Command R+ 在逻辑推理中占据主导地位**：成员们得出结论，逻辑推理表现最好的模型是 **Command R+08**，在复杂问题场景中表现出色。
   - 据观察，虽然它能妥善处理简单问题，但在更具挑战性的语境下，其表现显著优于 **Sonnet 3.5** 等其他模型。
- **复杂问题处理**：用户的共识表明，与 **Command R08** 等模型相比，**Command R+** 处理复杂查询的能力更稳健。
   - 正如成员所分享的，当面对较简单的问题时，有效性能会有所下降，这凸显了问题复杂性的重要性。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1325999150696759417)** (2 条消息): 

> `Evals, Mechanistic Interpretability, AR 中的物体检测` 


- **对 Evals 和 Mechanistic Interpretability 的兴趣**：一位成员表达了希望就 **Evals** 或 **Mechanistic Interpretability** 与他人建立联系的兴趣。
   - *嗨！这里有人对 Evals 或 Mech Interp 感兴趣吗？*
- **寻求 AR 项目见解**：另一位成员正在寻找与 **AR** 项目相关的研究，该项目可以检测**平面**并对**物体**进行分类。
   - *如果有人知道，请告诉我。*


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326268661022851167)** (4 条消息): 

> `Embed API 使用、响应结构、图像编码` 


- **使用 Embed API 进行图像输入**：一位用户分享了一个代码片段，展示如何将从 URL 获取的图像编码为 **base64** 格式，并配合 [Embed API](https://docs.cohere.com/reference/embed) 使用。
   - 该代码演示了如何使用 **cohere.ClientV2** 准备并发送图像数据以进行 Embedding。
- **关于 Embedding 响应的澄清**：一位用户询问 Embedding 响应是否保持与请求中发送的文本列表**相同的顺序**。
   - 另一位用户确认 Embeddings 确实会按照与请求**相同的顺序**返回，从而消除了关于文本与其 Embedding 匹配问题的疑虑。
- **Embedding 中的图像数据处理**：讨论涉及了获取图像内容并将其转换为 **base64** 编码字符串以进行 Embedding 的细节。
   - 重点在于确保正确处理内容类型（content type）请求头，以便适当地传递图像数据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://]">未找到标题</a>：未找到描述</li><li><a href="https://docs.cohere.com/reference/embed">Embed — Cohere</a>：此端点返回文本 Embeddings。Embedding 是一个浮点数列表，用于捕捉其所代表文本的语义信息。Embeddings 可用于创建文本分类...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1325999358600024065)** (16 条消息🔥): 

> `JavaScript 神经网络、Discord 重启问题、Cohere 计费政策` 


- **使用纯 JavaScript 构建神经网络**：一位用户请求创建一个**从零开始运行的纯 JavaScript 神经网络脚本**。
   - 关于具体实现，未提供进一步的细节或示例。
- **按下 Ctrl + R 时 Discord 重启**：一位用户询问为什么按下 **'Ctrl + R'** 时 Discord 会重启。
   - Cmd R Bot 尝试查找文档，但未给出答案，表明缺乏相关信息。
- **以美元计费的扣款阈值**：一位用户询问**单次扣款的美元阈值**，但未收到直接相关的查询信息。
   - Cmd R Bot 找到了关于计费政策的详情，指出一旦自助服务客户累计了 **$250** 的未结债务，就会进行扣款。
- **Cohere 计费政策详情**：根据 Cohere 文档，当用户产生 **$150** 的未结债务时，会发送一封**警告邮件**。
   - 一旦债务达到 **$250**，系统将自动通过 **Stripe** 对自助服务客户进行扣款。
- **计费政策来源**：有关计费政策的信息可以在 [2024 年 6 月 10 日发布说明](https://docs.cohere.com/v1/changelog/release-notes-for-june-10th-2024)中找到。
   - 发布说明详细介绍了工具使用、SDKs 和计费实践的更新。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1325919901046804501)** (4 条消息): 

> `用于目标检测的 AR 项目、实时 AR 资产实现` 


- **AR 项目协助请求**：一位成员正在寻求一个在增强现实（AR）中检测平面并讨论物体的项目支持，呼吁有相关知识的人士贡献力量。
   - 另一位成员对 AR 应用的热情表示赞同，并强调了用于实时 AR 资产的 reranker 将会多么有用。
- **对 AR x Cohere 协作的兴趣**：另一位成员对 AR 与 Cohere 在实时资产利用方面的潜在协作表示兴奋，展现了强烈的实现愿望。
   - 他们评论说，看到此类创新成真将是“非常酷的事情”，表明了对应用 AR 技术的浓厚兴趣。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326138230101250140)** (10 messages🔥): 

> `Triton 中的数组操作、项目中的配置管理、使用 wgmma 的 MMA 性能、内存布局与数据移动、Kernel 编译与 Autotuning` 


- **数组切片性能之谜**：*Theoboyer* 对操作数组时 **expand_dims** 与 **.reshape** 之间显著的性能差异表示困惑，特别是关于 `can_reorder` 的功能。
   - 问题集中在 `can_reorder` 是否通过重新排列数据或维度来允许更快的计算，以及它是否可以控制重新排列。
- **使用 2 的幂处理唯一维度**：*Mobicham* 描述了他们将维度限制为 2 的幂的方法，并实现了缓存机制，以便在形状匹配缓存配置时跳过 Autotuning。
   - 他们提到，虽然每个形状的编译大约需要 **0.06s**，但在 Prefill 阶段是可控的。
- **结合 Autotuning 策略的 Kernel 效率**：*Latkins* 提到在他们的 Kernel 策略中使用 `CLOSEST_M`，允许在尺寸变化时重新编译，同时避免 Autotuning，以便在大尺寸下获得更好的性能。
   - 他们指出，当性能下降超出预期参数时，Autotuning 可能并不实用。
- **在 H100 上为 MMA 使用 wgmma**：*Danielkoceja8071* 询问如何确保在 MMA 中使用 **wgmma**，并注意到尽管使用了 **H100**，他们的 Kernel PTX 仅显示 **mma.sync**。
   - 该问题暗示需要明确 Kernel 配置，以有效利用最新特性。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1325937966304071733)** (1 messages): 

> `输出 Fragment 寄存器布局、WMMA 加载与存储、矩阵复制实验` 


- **输出 Fragment 保留输入 Fragment 布局**：据报道，输出 Fragment 保持与输入 Fragment 相同的寄存器布局，索引如 **[0,1][0,2]** 和 **[8,0][8,1]** 在操作中保持一致。
   - 一位用户提到在使用 WMMA 从矩阵 A 加载并存储回矩阵 B 时，测试了这一行为并获得了成功的结果。
- **WMMA 有效地复制矩阵**：根据用户实验，**WMMA** 中的加载和存储过程应该在保留布局的同时将矩阵 A 复制到 B。
   - 该用户表示，如果其他人需要进一步澄清，愿意提供一个可运行的示例。
- **用户实验 WMMA**：一位成员分享了关于 WMMA 的见解，同时澄清他们在测试后已经结束了这一探索。
   - 他们轻松地表示愿意用实验中的示例来帮助他人。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1325959467950674062)** (2 messages): 

> `自定义 Autograd 函数、PyTorch 中的 Guard 失败` 


- **修改梯度的自定义 Autograd 函数**：一位成员询问，尽管文档警告不要这样做，但他们的自定义 **autograd function** 是否可以原地（in-place）修改梯度。他们观察到其模型的梯度与不使用自定义 Autograd 的简单实现的梯度非常接近。
   - 作为参考，他们附上了一个指向 [关于扩展 Autograd 的 PyTorch 文档](https://pytorch.org/docs/main/notes/extending.html#how-to-use) 的链接。
- **寻求 Guard 失败的详细日志**：另一位成员在获取 **guard failures** 的详细输出时遇到挑战，称其日志信息不足。他们推测 **0/0** 错误消息可能表示缺少与遇到的失败相关的消息。
   - 他们正在使用命令 `TORCH_LOGS="+dynamo,guards,bytecode,recompiles,recompiles_verbose"` 运行以增加日志详情。



**提到的链接**：<a href="https://pytorch.org/docs/main/notes/extending.html#how-to-use>">Extending PyTorch &mdash; PyTorch main documentation</a>：未找到描述

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1325975052901941259)** (3 条消息): 

> `Picotron framework, DeepSeek-v3 paper, LLM infrastructure videos` 


- **介绍用于 4D 并行性的 Picotron**：[Picotron 框架](https://github.com/huggingface/picotron)提供了一个**极简的 4D 并行**分布式训练解决方案，专为教学目的设计，使用户能够探索高级训练技术。
   - 其 GitHub 仓库强调了它的实用性，旨在促进 AI 训练方法的学习。
- **DeepSeek-v3 论文见解**：分享了*十个短视频*，以增强对 **DeepSeek-v3 论文**（[arXiv 链接](https://arxiv.org/abs/2412.19437)）**第 12-18 页**的理解。
   - 这些资源旨在为更广泛的受众阐明论文中呈现的复杂 LLM 基础设施概念。
- **LLM 基础设施视频播放列表**：推荐了一个 YouTube 播放列表，其中包含涵盖 LLM 基础设施关键方面的**短视频**，适合从事高级 AI 研究课题的人员。
   - 播放列表中的第一个视频可以在[这里](https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT6BUKCYeBMTEqnlcpYavxq&index=1&pp=gAQBiAQB)找到。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/srush_nlp/status/1876640795765379531">来自 Sasha Rush (@srush_nlp) 的推文</a>：关于 LLM 基础设施的 10 个短视频，帮助你理解 DeepSeek-v3 论文的第 12-18 页 (https://arxiv.org/abs/2412.19437) 🧵https://www.youtube.com/watch?v=76gulNlhiE4&list=PLO45-80-XKkT...</li><li><a href="https://github.com/huggingface/picotron">GitHub - huggingface/picotron: 用于教学目的的极简 4D 并行分布式训练框架</a>：用于教学目的的极简 4D 并行分布式训练框架 - huggingface/picotron
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1325945667520167976)** (3 条消息): 

> `Journey sharing, ONNX to TensorRT conversion issues` 


- **分享经验的兴奋感**：一位成员表达了分享他们历程的热情，并询问了有关所用资源的更多细节。
   - 另一位成员以积极的反应表示支持。
- **ONNX 转 TensorRT 转换中的困扰**：一位成员报告在 ONNX 转 TensorRT 转换时遇到困难，并指出输出存在差异。
   - 他们强调 **TensorRT 引擎输出**与 **PyTorch 模型**的输出不匹配。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1326054703154593874)** (4 条消息): 

> `Nvidia's Project DIGITS, Grace Blackwell Superchip, Training Small Models` 


- **Nvidia 的 Project DIGITS 将 AI 动力带到桌面**：Nvidia 推出了 [Project DIGITS](https://www.nvidia.com/en-us/project-digits/)，搭载 **Grace Blackwell 超级芯片**，在紧凑的设计中提供 Petaflop 级的 AI 性能。
   - 开发者现在可以在本地原型化并运行高达 **200B 参数**的大型 AI 模型，并拥有 **128GB 统一内存**。
- **对 Project DIGITS 能力的兴奋**：一位成员表达了热情，指出随着新的 **Tensor Cores** 出现， FP4 和 FP8 很快将成为训练模型的标准。
   - 他们思考该系统的能力是否足以训练较小的模型，尽管 Nvidia 声称支持 **200B 模型**。



**提及的链接**：<a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS：全球最小的 AI 超级计算机。</a>：立即预订。

  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1326123623848742933)** (3 messages): 

> `hipDeviceAttributeMaxBlocksPerMultiProcessor, CUDA vs HIP attributes comparison, AMD hardware max occupancy, Thread block discussions` 


- **关于 hipDeviceAttributeMaxBlocksPerMultiProcessor 的澄清**：一名成员提出了关于将 `hipDeviceAttributeMaxBlocksPerMultiProcessor` 与 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17f337476973ea65db85c277f4646f0b3) 中等效的 CUDA 属性进行比较的问题。他们推测实现 **2048 threads/SM** 需要两个 1024 线程的 thread block，并对此感到困惑。
- **对最大 block 计算的怀疑**：另一名成员分享了 [hip_device.cpp 文件](https://github.com/ROCm/clr/blob/b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f/hipamd/src/hip_device.cpp#L496) 的链接，并对每个 multiprocessor 最大 block 数计算的可靠性表示怀疑。他们注意到结果中存在一种不确定感。
- **AMD 硬件下的最大 occupancy 各不相同**：一名成员确认，在 AMD 硬件上，最大 occupancy 根据代际不同可能在 **8、10 或 20** 之间变化。他们表示不确定 **workgroup size** 对这些值有多大影响。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_17f337476973ea65db85c277f4646f0b3">CUDA Runtime API :: CUDA Toolkit Documentation</a>：未找到描述</li><li><a href="https://github.com/ROCm/clr/blob/b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f/hipamd/src/hip_device.cpp#L496">clr/hipamd/src/hip_device.cpp at b8ba4ccf9c53f6558a5e369e3c1c05de97a0c28f · ROCm/clr</a>：通过在 GitHub 上创建账号为 ROCm/clr 开发做贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326304924220522617)** (5 messages): 

> `Discord based leaderboard, GPU Glossary resources` 


- **Discord 排行榜需要 Alpha 用户！**：一名成员宣布他们正在为新的 **基于 Discord 的排行榜** 寻找 **alpha 用户**，该排行榜连接 GPU 以促进特定 kernel 的竞赛。
   - *如果这听起来有那么一点意思*，他们鼓励回复以获取教程。
- **GPU Glossary（术语表）材料发布**：另一位用户分享说 `gpu-glossary.zip` 包含了所有格式化为 **Markdown 文件** 的 GPU Glossary 材料，并在 `contents.json` 中关联了 URL 作为 ToC（目录）。
   - 该压缩文件可以通过 [此链接](https://cdn.discordapp.com/attachments/1298372518293274644/1326364815421341696/gpu-glossary.zip?ex=677f28fe&is=677dd77e&hm=7bc8bcc2212219fa43b55a4dfe342017480bdf85ff2ffb599d30feb4ba302935&) 直接访问。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1325957436208578641)** (4 messages): 

> `LlamaIndex and MLflow integration, Multi-agent systems with NVIDIA AI, Cohere models usage with LlamaIndex` 


- **通过 LlamaIndex 和 MLflow 集成简化流程**：一份分步指南概述了如何结合 **LlamaIndex**、**MLflow**、**Qdrant** 和 **Ollama** 以增强向量存储和模型追踪。集成这些工具可以实现高效的 **实时** 操作和评估，详见 [完整指南](https://t.co/lNDDgdOo86)。
   - 此集成强调在这些技术旁利用 **Change Data Capture** 来改进工作流。
- **NVIDIA AI 助力新型多 Agent 系统**：在 CES 上发布了一个新型多 Agent 系统蓝图，利用 **NVIDIA AI** 辅助研究和撰写博客文章。该系统旨在减轻内容创作任务带来的 **时间消耗**，让 LLM 能够高效地进行复杂研究。
   - 查看官方公告中的详情 [此处](https://t.co/eS4BhuAZVS)。
- **Cohere 模型焕然一新**：团队赞扬了 **Cohere** 模型强大的 embedding 能力以及最近与 **LlamaIndex** 集成的文档更新。分享了安装说明和先决条件，确保用户可以共同充分利用 **Cohere** 的 SDK 和 LlamaIndex 功能。
   - 更多信息和安装步骤，请参阅 [文档](https://t.co/dLKGgkqOe8)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://t.co/dLKGgkqOe8">LlamaIndex — Cohere</a>：了解如何共同使用 Cohere 和 LlamaIndex 基于数据生成响应。</li><li><a href="https://t.co/eS4BhuAZVS">Document Research Assistant for Blog Creation Blueprint by Llamaindex | NVIDIA NIM</a>：使用 LlamaIndex 和 Llama3.3-70B NIM LLM 自动化研究并使用 AI Agents 生成博客。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326091513616269354)** (9 条消息🔥): 

> `LlamParse 错误, LlamaIndex 教程 Notebook, Text-to-SQL 功能, 文档链接` 


- **LlamParse 遇到首次解析错误**：一位用户报告称，在首次使用 LlamParse 解析 PDF 文件时收到错误，但在随后的尝试中运行正常。
   - 另一位用户询问该错误是否每次都会在同一个文件上发生，并寻求查看相关的 PDF 文件。
- **LlamaIndex 教程 Notebook 链接失效**：一位用户询问了特定 LlamaIndex 教程的 Notebook，指出文档中提供的链接已失效。
   - 另一位用户分享了正确 Notebook 的有效链接，并提到它可能只是从导航栏中丢失了。
- **LlamaIndex 中的 Text-to-SQL 说明**：文档涵盖了 LlamaIndex 从非结构化来源创建结构化数据的能力，以及 Text-to-SQL 功能。
   - 其中包含一条关于执行任意 SQL 查询风险的安全提示，并建议采用合理的实践方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/structured_data/">Structured Data - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL Guide (Query Engine + Retriever) - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb">llama_index/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb at main · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326187761366995039)** (10 条消息🔥): 

> `Open Interpreter 1.0 发布, 经典版 OI 归档, pip 安装问题, 修改与 PR 提交, 本地模型性能` 


- **Open Interpreter 1.0 发布临近**：最新的 GitHub commit 表明 **Open Interpreter 1.0** 即将发布，但目前无法运行代码，引起了用户的困惑。
   - 关于 1.0 的变更和路线图的文档尚未明确列出。
- **经典版 Open Interpreter 已归档**：OI 的经典版本已被归档，所有之前的 Prompt 现在都存储在一个过时的文件夹中，限制了用户的贡献。
   - 用户注意到，由于经典版本处于归档状态，提交 PR 存在困难。
- **Pip 安装问题被提出**：一位用户提到，使用 `pip install open-interpreter` 安装稳定版本的功能并不如预期。
   - 目前尚不清楚如何增强现有版本，因为修改会导致进一步的困惑。
- **修改限制与困惑**：用户表示希望改进工具和 Prompt 以获得更好的功能，但对向 1.0 版本的过渡感到困惑。
   - 一些修改存在问题，因为在开发 1.0 的同时，无法选择提交旧版本的 PR。
- **本地模型性能担忧**：建议对本地模型使用 `--no-tool-calling`，这表明对 System Prompt 的某些更改可能会产生负面影响。
   - 讨论强调了由于新版本中的调整，较小模型所面临的困难。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/open-interpreter/commit/21babb186f13e263a72cf525d15d79788edf4644">Open Interpreter 1.0 Preview · OpenInterpreter/open-interpreter@21babb1</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commit/275105771dfa985ee777e4a9c6e9c4d760c7b7b9">Archived Interpreter Classic · OpenInterpreter/open-interpreter@2751057</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1326024960493486134)** (8 messages🔥): 

> `GH200 利用率、编译挑战、Discord 链接问题` 


- **GH200 用户可能提供协助**：一位用户注意到 <@201777246367645696> 正在使用 **GH200**，并建议他们可能能够提供帮助。
   - *希望这方面的协作能够缓解其他人面临的一些挑战。*
- **依赖项导致编译耗时**：另一位用户对 **dependencies**（依赖项）导致系统完全编译延迟表示沮丧。
   - *他们提到，虽然让它运行起来是可能的，但需要投入大量时间。*
- **Discord 链接者的回归引发骚动**：臭名昭著的 **discord link guy** 再次出现，在各个频道发布可能不受欢迎的链接，引发了警告。
   - 一位用户报告了此问题，随后确认已**封禁**并删除了有问题的欢迎频道消息。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326162805392674949)** (7 messages): 

> `MiPROv2 指令流、dspy 与 Langchain 的集成` 


- **MiPROv2 可以调整指令生成**：一名成员建议，**MiPROv2** 可以一次尝试一条指令并根据结果进行调整，而不是预先编写一整套指令。
   - 这种方法可以利用 LLM 作为裁判来评判输出，从而为改进指令提供有价值的反馈。
- **dspy.COPRO! 作为一个相关概念**：另一名成员指出，提议的 MiPROv2 方法更接近于使用 **dspy.COPRO!** 进行指令生成和尝试。
   - 这引起了兴趣，成员们表示打算进一步探索该概念。
- **将 dspy 与 Langchain 集成的挑战**：一名成员询问关于使用 2.6 版本将 **dspy** 与 **Langchain** 集成的问题，表示有兴趣创建 LLM Agent。
   - 一份回复详细说明，在处理这两个框架时，没有直接的组合方法。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1326261516261134521)** (1 messages): 

> `证书申报、作业完成、证书截止日期` 


- **证书申报表重新开放**：**Certificate Declaration form**（证书申报表）已为那些在 12 月完成所有作业的人重新开放，允许他们领取证书。
   - *暂定截止日期*为 **1 月底**，参与者必须填写表格，因为之前的作业不会重新开放。
- **每个学生限领一张证书**：提醒学生本课程只能获得**一张证书**，以确保认证资格的清晰。
   - 这一限制强调了完成所有要求作业以获得认证的必要性。
- **证书发放可能延迟**：证书计划在 **1 月底**前寄出，但对于现在才填写申报表的人，可能会有延迟。
   - 敦促参与者及时填写表格，以避免收到证书的延迟。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1325966328372265030)** (5 messages): 

> `申报表确认、提交时的邮箱地址一致性` 


- **对重新开放申报表的感谢**：<@854134294870884363> 因重新开放 **declaration form**（申报表）而受到成员们的赞赏，强调了其重要性。
   - 成员们表达了谢意，并承认这一行动对提交过程的积极影响。
- **一致邮箱地址的重要性**：<@tarande57> 强调，申报表填写的邮箱必须与课程作业使用的邮箱**相同**才能获得证书。
   - 此政策确保了对提交内容的准确跟踪，因此参与者遵守该政策至关重要。
- **更改邮箱后的确认请求**：@iamkrish10 询问他们最初用于作业的邮箱地址是否得到确认，因为他们使用了不同的邮箱提交表格。
   - 他们强调已在提交文本框中注明了原始邮箱，并请求对其提交进行验证。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1325938386590109747)** (5 条消息): 

> `Reasoner v1 功能、Local Docs 索引问题、Embedding 模型支持` 


- **Reasoner v1 的功能受到赞赏**：一位成员称赞了 **Reasoner v1** 的工作，并询问除了 Qwen 2.5 coder 变体之外，还有哪些模型或模板可以与推理模式配合使用。
   - 另一位成员确认 **OpenAI 兼容的远程模型**和多个本地模型都可以工作，并表示他们正在添加更多开箱即用的模型。
- **Local Docs 在目录索引方面遇到困难**：一位成员报告了 Local Docs 无法有效嵌入子目录文件的问题，尽管最初的索引是成功的。
   - 他们指出，问题似乎与时间戳的使用方式有关，如果某些文件已经包含在另一个文档中，可能会导致 **LocalDocs** 没有嵌入任何文件。
- **关于更换 Embedding 模型的咨询**：另一位成员表示有兴趣将现有模型替换为不同的 Embedder 模型，并检查其与 **Text-embedding-inference** 或 **vLLM embedder** 的兼容性。
   - 这一咨询突显了处理 Embeddings 的成员对模型灵活性和支持的持续关注。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1326255868408496240)** (1 条消息): 

> `MLOps 与 Feature Stores 网络研讨会、LLM 在 MLOps 中的集成、2024 年 MLOps 进展、2025 年的趋势与挑战` 


- **关于 MLOps 和 Feature Stores 的免费网络研讨会**：参加即将于 **1 月 15 日** **太平洋时间上午 8 点** 举行的网络研讨会，届时 Ben Epstein 和 Simba Khadder 将讨论 2025 年的 **MLOps** 和 **Feature Stores**。[在此注册](https://buff.ly/4j9oiVg)以预留名额！
   - 研讨会将深入探讨**最佳实践**和**前沿架构**，并在最后留出问答环节。
- **2024 年 MLOps 的关键进展**：讨论将回顾 **2024 年** **MLOps 的重大进展**并展望 2025 年，强调 **Large Language Models (LLMs)** 的作用。将重点介绍影响 **MLOps** 和 **LLMOps** 的关键趋势。
   - 期待了解 **融合领域** 的见解以及 MLOps 在适应机器学习技术进步过程中的演变格局。
- **谁应该参加此次研讨会**：该活动面向对最新 MLOps 趋势感兴趣的**数据工程师**、**数据科学家**、**机器学习工程师**和 **AI/ML 爱好者**。这是专业人士从行业领导者那里获取见解的机会。
   - 参与者将在问答环节与演讲者直接交流，增强对未来 MLOps 策略的理解。



**提到的链接**：<a href="https://buff.ly/4j9oiVg">2025 年 MLOps 与 Feature Stores（与 Ben Epstein）</a>：加入我们 1 小时的网络研讨会，Featureform 的 Simba Khadder 和 MLOps 社区的 Ben Epstein 将畅谈 2025 年即将到来的 MLOps 趋势！

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1326271196614098944)** (1 条消息): 

> `LLM 安全测试、有害 AI 助手挑战赛、GraySwanAI Arena` 


- **GraySwanAI 启动有害 AI 助手挑战赛**：新的**有害 AI 助手挑战赛**将于 **东部时间 1 月 4 日下午 1 点** 启动，为创新的 Prompt Injection（提示词注入）和 Jailbreaking（越狱）方法提供 **$40,000** 的奖金。
   - 参与者必须寻找独特的方法来诱导 AI 助手产生有害响应，本次竞赛活动允许**多轮输入**。
- **上一次活动的特色是预发布测试**：早期的活动为参与者提供了在 **o1 模型** 正式发布前对其进行测试的机会，正如 **12/5 OAI 论文** 中所引用的那样。
   - 这一系列持续的活动展示了 LLM 安全测试和社区参与方面的最新进展。
- **加入 GraySwanAI 社区**：感兴趣的参与者可以在 [app.grayswan.ai](http://app.grayswan.ai/arena) 注册并加入社区，或通过 Discord [discord.gg/WqHkWt99](http://discord.gg/WqHkWt99) 进行联系。
   - 本次活动旨在促进 AI 安全测试领域爱好者之间的协作和技能提升。



**提到的链接**：<a href="https://x.com/GraySwanAI/status/1872720375328411668">来自 Gray Swan AI (@GraySwanAI) 的推文</a>：🚨 新竞技场启动警报：有害 AI 助手挑战赛 🚨💰 $40,000 奖金📅 启动日期：东部时间 1 月 4 日下午 1 点🤖 5 个匿名模型🔥 速度和数量奖金。🎮 允许多轮输入...

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1325984835914694766)** (1 messages): 

> `Common Voice AMA, 2024 Review, Voice Technology Accessibility` 


- **Common Voice 新年 AMA 启动**：Common Voice 正在其[新 Discord 服务器](https://discord.gg/b4c83ppxdU)中启动 **2025 AMA**，以回顾过去一年并与社区互动。
   - 本次会议旨在回答有关项目历程和未来发展的所有问题。
- **2024 年度回顾与问答环节**：邀请参与者加入团队进行 **2024 年度回顾**，特邀嘉宾包括产品总监和前端工程师。
   - 活动将包括互动问答环节，以促进社区参与和反馈。
- **提升语音技术的可访问性**：Common Voice 旨在使语音技术**开放且可访问**，为开发者提供创建语音识别系统所需的核心数据。
   - 该项目强调了将传统上难以获取的语音数据民主化的重要性，以降低创新门槛。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1326074798165725196)** (1 messages): 

> `Dolphin 3.0 Model Series, BFCL Leaderboard` 


- **Dolphin 3.0 会登上 BFCL 排行榜吗？**：一位成员询问 **Dolphin 3.0 模型系列** 是否会出现在 BFCL 排行榜上，并对其性能表示关注。
   - 他们提供了 [Hugging Face 上的 Dolphin 3.0](https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3) 链接以获取更多详情。
- **Cognitive Computations 的最新更新**：**cognitivecomputations/Dolphin3.0-Llama3.2-1B** 模型最近进行了更新，在 Hugging Face 上获得了 **34** 颗星。
   - 该帖子包含一张展示该模型的图片，并吸引了 **14** 条评论。



**提到的链接**：<a href="https://huggingface.co/collections/cognitivecomputations/dolphin-30-677ab47f73d7ff66743979a3">Dolphin 3.0 - cognitivecomputations 集合</a>：未找到描述

  

---


---


---


{% else %}


> 为了便于邮件阅读，完整的频道细分内容已被截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}