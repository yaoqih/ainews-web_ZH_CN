---
companies:
- meta-ai-fair
- mistral-ai
- qwen
- deepseek
- salesforce
- bilibili
- stability-ai
- google
date: '2025-05-19T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Meta** 发布了 **KernelLLM 8B**，在 KernelBench-Triton Level 1 评测中超越了 **GPT-4o** 和 **DeepSeek
  V3**。**Mistral Medium 3** 在多项基准测试中强势亮相。**Qwen3** 系列模型引入了支持多语言的统一框架。**DeepSeek-V3**
  具备硬件感知协同设计的特点。**BLIP3-o** 系列发布，采用扩散 Transformer（Diffusion Transformers）处理多模态任务。**Salesforce**
  推出了 **xGen-Small** 模型，在长文本上下文和数学基准测试中表现优异。**哔哩哔哩（Bilibili）** 发布了用于动漫视频生成的 **AniSORA**。**Stability
  AI** 开源了针对 Arm 设备优化的 **Stable Audio Open Small**。谷歌的 **AlphaEvolve** 编程智能体自 1969
  年以来首次改进了**施特拉森算法（Strassen''s algorithm）**。研究表明，**思维链（CoT）推理**可能会损害指令遵循能力，虽然分类器选择性推理等缓解策略最为有效，但推理技术仍表现出高方差和有限的泛化能力。“思维链（CoT）推理会损害模型遵循指令的能力”以及“少样本上下文学习、自我反思、自我选择性推理和分类器选择性推理等缓解策略可以抵消由推理导致的失效”。'
id: MjAyNS0w
models:
- kernelllm-8b
- gpt-4o
- deepseek-v3
- mistral-medium-3
- qwen3
- blip3-o
- xgen-small
- anisora
- stable-audio-open-small
- alphaevolve
people:
- reach_vb
- lmarena_ai
- theadimeline
- adcock_brett
- jxmnop
- dair_ai
- omarsar0
title: 今天没发生什么事。
topics:
- benchmarking
- model-performance
- multilinguality
- hardware-optimization
- multimodality
- image-generation
- video-generation
- text-to-audio
- model-parallelism
- chain-of-thought
- instruction-following
- reasoning
- mitigation-strategies
---

**平静的一天。**

> 2025年5月16日至5月19日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（215 个频道，11148 条消息）。预计节省阅读时间（以 200wpm 计算）：947 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以美观的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

谷歌将在明天的 I/O 大会上发布大量内容已是公开的秘密，并且已经开始[推出 Jules](https://x.com/nanulled/status/1924554666731262065)。还有其他一些发布——[Amazon 的 Strands Agents](https://strandsagents.com/) 和 Anthropic 的 [Claude Code SDK](https://docs.anthropic.com/en/docs/claude-code/sdk)，但都没有达到头条新闻的级别。

AI Engineer [Expo Explorer 门票](http://ti.to/software-3/ai-engineer-worlds-fair-2025)已于周末上线。如果你喜欢走廊交流（hallway track）、展会环节，并想结识 AI Eng 领域的顶级云厂商、初创公司和雇主，欢迎加入我们。


![](https://resend-attachments.s3.amazonaws.com/YrrfaUU3X1AHkBF)


[这里](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/HNEXPO)为前 50 名 AINews 读者提供了限量的折扣。

---

# AI Twitter 综述

**AI 模型发布与性能**

- **Meta 发布了 KernelLLM 8B**，根据 [@reach_vb](https://twitter.com/reach_vb/status/1924478755898085552) 的说法，它在 KernelBench-Triton Level 1 的单次推理（single-shot）性能上超过了 **GPT-4o** 和 **DeepSeek V3**，而在多次推理下，其表现优于 **DeepSeek R1**。
- **Mistral Medium 3** 强势登场，根据 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1924482515244622120) 的数据，它在聊天总榜排名第 11，数学排名第 5，硬核提示词（Hard Prompts）与编程排名第 7，WebDev Arena 排名第 9。
- **Qwen3 模型**发布，包括参数量从 0.6B 到 235B 的稠密模型和 Mixture-of-Expert 模型，具有统一的框架并扩展了多语言支持，消息来自 [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232110383960163)。
- **DeepSeek-V3** 采用了硬件感知的协同设计，并解决了扩展性问题，消息来自 [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232113101890003)。
- **BLIP3-o** 发布，这是一系列使用 Diffusion Transformer 的全开源统一多模态模型，在图像理解和生成任务上表现卓越，消息来自 [@TheAITimeline](https://twitter.com/TheAITimeline/status/1924232118755824119)。
- **Salesforce** 发布了 **xGen-Small** 系列小型 AI 模型，其中 9B 参数模型在长上下文理解以及数学和编程基准测试中表现强劲，消息来自 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133781704786366)。
- **Bilibili** 发布了动漫视频生成模型 **AniSORA**，消息来自 [@reach_vb](https://twitter.com/reach_vb/status/1924425789774123316)。
- **Stability AI** 开源了 **Stable Audio Open Small**，这是一款文本转音频的 AI 模型，可生成 11 秒音频，并针对基于 Arm 的消费级设备进行了优化，消息来自 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133939376996539)。
- [@jxmnop](https://twitter.com/jxmnop/status/1924207755956478400) 讨论了一篇 **2003 年来自蒙特利尔的关于在文本上训练神经网络的论文**，指出其模型和技术（包括模型并行化）具有超前意识。
- 谷歌发布了 **AlphaEvolve**，这是一个使用 LLM 引导进化来发现新算法并优化计算系统的编程 Agent，它发现了自 1969 年以来对 **Strassen 算法**的首次改进，消息来自 [@dair_ai](https://twitter.com/dair_ai/status/1924150361750655178) 和 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133683444793819)。

**AI 安全、推理与指令遵循**

- **思维链 (CoT) 推理可能会损害模型遵循指令的能力**：[@omarsar0](https://twitter.com/omarsar0/status/1924458157444579700) 总结了一篇论文，指出推理增强型语言模型中这种违反直觉的弱点，并分享了缓解策略，同时将该论文添加到了 Reasoning LLMs 指南中。
- **缓解策略**，如 few-shot in-context learning、self-reflection、self-selective reasoning 和 classifier-selective reasoning 可以抵消推理引发的失败，根据 [@omarsar0](https://twitter.com/omarsar0/status/1924458176096751806) 的说法，classifier-selective reasoning 是最稳健的。
- **推理无法在不同环境间泛化**，且提示策略产生的高方差削弱了高级推理技术的可靠性，根据 [@omarsar0](https://twitter.com/omarsar0/status/1924182841677709540) 和 [@omarsar0](https://twitter.com/omarsar0/status/1924182837307089216) 的说法。
- **较大模型从策略性提示中获益较少**，且在简单任务上，过度推理会损害较小模型的表现，根据 [@omarsar0](https://twitter.com/omarsar0/status/1924182839081218092) 和 [@omarsar0](https://twitter.com/omarsar0/status/1924182835289620950) 的说法。
- [@RichardSocher](https://twitter.com/RichardSocher/status/1924217608569528799) 讨论了 **AI Safety Paradox**，认为随着智能边际成本的降低，通过识别和处理更多的攻击向量，可以为生物战或网络战提供更好的防御。

**AI 工具与应用**

- **微软将 Grok 添加到其 foundry 模型库中**，根据 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924508051253653745) 和 [@ibab](https://twitter.com/ibab/status/1924518628172693922) 的说法，Grok 3 已在 Microsoft Azure 上可用。
- **GitHub Copilot** 现在支持整个软件开发生命周期，提供 Agent 模式、团队支持、应用现代化和 SRE Agent，根据 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924495827999031709) 的说法。
- **OpenAI 推出了 Codex**，这是一个新的编程 Agent，可以自主构建功能和修复 Bug，适用于 Pro、Enterprise 和 Team 用户，根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133661072396293) 的说法。
- **阿里巴巴 Qwen 团队** 向所有用户开放了 **Deep Research for Qwen Chat**，使用户能够就不同主题准备详细报告，根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133804630753660) 的说法。
- **Notion** 为其商业计划订阅者推出了“AI for Work”套件，提供 AI 会议记录、访问不同 AI 模型、企业搜索以及用于起草文档的研究模式，根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133849610543431) 的说法。
- **阿里巴巴的 Wan** 发布了 **Wan2.1-VACE**，这是一个用于视频创建和编辑的统一 AI，提供 1.3B 和 14B 两种尺寸，根据 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133827095498952) 的说法。
- **基于 MLX 的 LLM 可以直接从 Hugging Face Hub 访问**，从而在终端实现极速智能，根据 [@reach_vb](https://twitter.com/reach_vb/status/1924517049474101412) 的说法。
- **Modal Labs 的 Dicts serverless KV 存储的新功能** 包括无扩展限制、LRU-cache 语义、分布式锁定和持久性，根据 [@akshat_b](https://twitter.com/akshat_b/status/1924552967673545055) 的说法。
- **LangChain** 宣布为 LangGraph 提供节点级缓存支持，从而实现更快的迭代，根据 [@hwchase17](https://twitter.com/hwchase17/status/1924557667634172099) 的说法。
- [@fchollet](https://twitter.com/fchollet/status/1924509605050327475) 强调了 **Genspark AI Sheets**，这是一款允许用户与电子表格对话的应用。

**AI 业务与策略**

- **Sakana AI** 和 **MUFG Bank** 签署了全面合作伙伴关系协议，旨在利用 AI 为其系统赋能。根据 [@AIatMeta](https://twitter.com/AIatMeta/status/1924502785028190366) 和 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1924442310210678974) 的消息，这将使 Sakana AI 在一年内实现盈利（根据 [@hardmaru](https://twitter.com/hardmaru/status/1924480171606003841)）。
- **Cohere** 正与 **Dell** 合作，在本地提供安全、Agentic 的企业级 AI 解决方案，消息来自 [@cohere](https://twitter.com/cohere/status/1924512634373865950)。
- **Perplexity** 在 Whatsapp 上现在更灵敏、更快速且更健谈，导致使用量增加，消息来自 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1923924897614659922)。
- Andrew Ng 将于 6 月 5 日在旧金山参加 @Snowflake 的 Dev Day 2025 并登台演讲，消息来自 [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1924484108974993540)。
- [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1924509198848819347) 讨论了**新产品开发法如何要求团队培养一种时刻沉浸在 Generative Models 中的文化**，并通过实验在几天而非几个月内发现新的产品体验。
- [@adcock_brett](https://twitter.com/adcock_brett/status/1923927753969353029) 强调了**在初期设定公司价值观的重要性**，因为后期很难进行纠偏。

**Infrastructure, Tools and Datasets**

- **NVIDIA** 开源了 **Physical AI models**，这些推理模型能够理解物理常识并生成适当的具身决策，消息来自 [@reach_vb](https://twitter.com/reach_vb/status/1924525937443365193)。
- **Meta** 刚刚在 Hugging Face 上发布了 **KernelLLM 8B**，消息来自 [@reach_vb](https://twitter.com/reach_vb/status/1924478755898085552)。
- **The SaharaLabsAI SIWA Testnet** 已上线，为其开发平台提供可扩展的计算能力，消息来自 [@togethercompute](https://twitter.com/togethercompute/status/1924514334044213572)。
- **Marin** 作为一个 AI 开放实验室，旨在通过开放开发实现开源 AI 的愿景，消息来自 [@percyliang](https://twitter.com/percyliang/status/1924527490351169964)。
- **现在可以直接从 Hugging Face Hub 访问 MLX 模型**，消息来自 [@reach_vb](https://twitter.com/reach_vb/status/1924517049474101412)。
- [@maximelabonne](https://twitter.com/maximelabonne/status/1924412611430404492) 宣布 **Qwen3 已被 Abliterated**，使用了 mini-batching 以及结合字典和来自 @NousResearch 的 Minos-v1 分类器的混合方法。
- [@HamelHusain](https://twitter.com/HamelHusain/status/1924454532224078220) 宣布其首场讲座使用 35% 折扣码报名的最后一天。
- 发布了用于分子化学的新 Density Functional Theory (DFT) 数据集 **Open Molecules 2025 (OMol25)**，以及 Meta 的 **Universal Model for Atoms (UMA)**（一种机器学习原子间势能模型），消息来自 [@AIatMeta](https://twitter.com/AIatMeta/status/1924502785028190366)。
- 清华大学研究人员的一篇论文详细介绍了 HuB，这是一个帮助人形机器人处理极端平衡任务的统一框架，消息来自 [@adcock_brett](https://twitter.com/adcock_brett/status/1924133916971020739)。

**Humor**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1924023253812531606) 发布了一个梗图，内容是 **"我：'我要制造 AI 之神机器' 同样是我："**。
- [@scottastevenson](https://twitter.com/scottastevenson/status/1924129325382533302) 发布了一个 **Elon Musk 的史诗级梗图**。
- [@vikhyatk](https://twitter.com/vikhyatk/status/1924104665161183685) 分享了一个关于 **Linux 用户遇到 Mac 用户**的梗图。
- [@TheTuringPost](https://twitter.com/TheTuringPost/status/1924296119582093752) 分享了来自 @Microsoft CTO @kevin_scott 的一个 **"温馨的人类故事"**，讲述了现在每个人如何更有能力在 AI 领域进行构建。
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1924499010565313029) 开玩笑说要**将芯片供应链本土化，以召唤沙神来消灭共产主义者**。

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Intel Arc Pro GPU 和 Project Battlematrix 工作站发布

- [**Intel 发布售价 299 美元的 Arc Pro B50（16GB 显存）以及配备 24GB Arc Pro B60 GPU 的 “Project Battlematrix” 工作站**](https://www.tomshardware.com/pc-components/gpus/intel-launches-usd299-arc-pro-b50-with-16gb-of-memory-project-battlematrix-workstations-with-24gb-arc-pro-b60-gpus) ([评分: 604, 评论: 255](https://www.reddit.com/r/LocalLLaMA/comments/1kq9294/intel_launches_299_arc_pro_b50_with_16gb_of/)): **Intel 宣布推出两款 Arc Pro GPU：售价 299 美元的 Arc Pro B50 (16GB VRAM) 和售价约 500 美元的 Arc Pro B60 (24GB VRAM)，目标市场为专业和 AI 工作站。B60 是 “Project Battlematrix” 的核心——这是 Intel 旨在提供高性价比 AI 工作站的计划——这表明在内存密集型 LLM/AI 工作流中具有极高的性价比，尽管这些显卡并不侧重于游戏。详情请参阅 [原始报道](https://videocardz.com/newz/intel-announces-arc-pro-b60-and-b50-gpus-targeted-at-ai-workstations) 以了解规格和定位。** 评论者强调了 B60 在 LLM 工作中极高的 VRAM/价格比，并指出其保修期比二手 NVIDIA RTX 3090 更长，承认这些显卡虽然不适合游戏，但在 AI 和专业应用中具有巨大潜力。
    - 多位用户强调了 Arc Pro B60 价格的吸引力：500 美元获得 24GB VRAM 被认为是一个突破，相比之下，二手 NVIDIA 3090 价格更高、货源更难找，且通常没有保修。这使得 Intel 显卡对于大语言模型 (LLM) 应用和其他 VRAM 密集型工作负载特别有吸引力，因为这些显卡并非为游戏设计。
    - 存在关于 Arc Pro B60 的 24GB 配置是否可扩展的技术好奇——一位用户询问假设的双核版本是否能以 1,000 美元的价格提供 48GB VRAM。这表明人们对 Intel 的架构是否允许这种模块化配置和定价感兴趣。
    - 评论者进一步指出，Intel 的加入可能会扩大 AI 硬件的选择范围，特别是在 3090 供应下降的情况下，可能会推动游戏以外（据报道这些显卡不太适合游戏）的 AI 软件和生态系统的改进。
- [**配备 48GB 显存的 Intel Arc GPU 是否会以 1000 美元的价格接管市场？**](https://www.reddit.com/r/LocalLLaMA/comments/1kqaqmr/is_intel_arc_gpu_with_48gb_of_memory_going_to/) ([评分: 231, 评论: 149](https://www.reddit.com/r/LocalLLaMA/comments/1kqaqmr/is_intel_arc_gpu_with_48gb_of_memory_going_to/)): **Intel 宣布推出 Arc Pro B60 (24GB GDDR6) 和 B50 (16GB GDDR6) 工作站 GPU，其中双 GPU B60 配置预计以低于 1,000 美元的价格提供总计 48GB 的 VRAM，而 24GB 显卡价格约为 500 美元 ([VideoCardz](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory), [WCCFTech](https://wccftech.com/intel-arc-pro-b60-24-gb-b50-16-gb-battlemage-gpus-pro-ai-3x-faster-dual-gpu-variant/))。这些显卡定位于 AI 和工作站任务，较少关注游戏，更多针对专业和 AI 工作负载，与目前同价位的消费级 GPU 相比，在 VRAM 容量方面具有显著优势。Intel 尚未发布详细的性能基准测试，目前尚不清楚其驱动程序在深度学习工作负载方面与 NVIDIA 或 AMD 的替代方案相比表现如何。** 热门评论对实际价格和供货情况表示怀疑，怀疑消费者的获取渠道将受到限制，且由于需求超过供应，价格可能会超过理论上的 MSRP。技术上对显存大小感到兴奋，但对库存或定价是否符合预期表示怀疑。
    - 48GB Intel Arc 显卡利用了所有可用的 PCIe 通道（2x8 配置），这意味着在运行双卡设置时能保持全带宽——这消除了使用较少通道的显卡中存在的潜在瓶颈。这对于需要高内存和高带宽的工作负载特别有利。
    - 对于低于 1,000 美元的 48GB VRAM GPU 的预期价格存在怀疑。评论者指出，如果这种 GPU 能以这个价格上市，将是 GPU 市场的重大转变；然而，他们预计长期供应会出现问题，预测由于高需求和初始供应有限，将会出现缺货和预订积压。
    - 潜在较低价格下充足的 VRAM (48GB) 被视为具有颠覆性，特别是与目前的 NVIDIA 产品相比，后者的同等 VRAM 价格要贵得多。如果定价和供货情况如传闻般实现，这将激励潜在买家考虑将 Intel 作为内存受限的计算或创意任务中的竞争性替代方案。

- [**Computex: Intel Unveils New GPUs for AI and Workstations**](https://newsroom.intel.com/client-computing/computex-intel-unveils-new-gpus-ai-workstations) ([Score: 155, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1kq8wo4/computex_intel_unveils_new_gpus_for_ai_and/)): **在 Computex 上，Intel 发布了拥有 24GB VRAM 的 Arc Pro B60 GPU（建议零售价约 500 美元）和拥有 16GB 的 B50（建议零售价 299 美元），旨在针对 AI 推理和工作站工作负载。最初将通过 OEM 工作站供应商在 2024 年第三季度上市，在第四季度软件栈成熟后可能发布独立 DIY 版本。这些显卡扩展了 Arc Pro 在 AI 相关工作流中的存在感，同时改进了软件支持并推出了用于可扩展数据中心 AI 工作负载的 Gaudi 3 PCIe 加速器（[Intel 新闻稿摘要](https://newsroom.intel.com/client-computing/computex-intel-unveils-new-gpus-ai-workstations)，[videoCardz 规格详情](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory)）。** 评论者的主要技术关注点包括对 DIY 价格和可用性的怀疑，对改进 Intel GPU 软件/驱动程序成熟度的迫切需求（理由是与 NVIDIA 相比支持较差），以及建议统一或更频繁地更新 Intel 的 IPEX-LLM 和 OpenVINO 框架，以实现更广泛的 AI 框架兼容性。
    - Arc Pro B50 (16GB) 和 B60 (24GB) 将于 2024 年第三季度上市，价格分别为 299 美元和约 500 美元，首发目标是工作站 OEM。DIY 的可用性尚不确定，因为零售发布可能取决于最初的商业采用情况和进一步的软件优化（可能在第四季度或更晚）。[(来源)](https://videocardz.com/newz/intel-announces-arc-pro-b60-24gb-and-b50-16gb-cards-dual-b60-features-48gb-memory)
    - 技术评论者指出，Arc Pro B60 的 24GB VRAM 在该价格档位非常突出——成本约为 AMD 7900XTX 的一半，仅为 Nvidia RTX 4090 价格的五分之一。据报道，在 Qwen2 7B Q8 上的 LLM 推理速度约为 35T/s；尽管这些显卡在要求苛刻的扩散模型（如 Flux dev、HiDream Q8）上绝对推理速度可能较慢，但其大容量显存能够运行该细分市场消费级 GPU 通常无法运行的大型模型。
    - 确定的一个重大瓶颈是 450GB/s 的显存带宽，明显低于 Apple M4 Max（约 550GB/s）等竞争芯片。虽然较低的带宽可能会被这些架构之间的计算性能差异所抵消，但它仍被标记为工作负载适用性的一个考量因素。

### 2. Offline and Open Source AI Productivity and Speech Tools (Clara, Kokoro-JS, OuteTTS)

- [**Clara — A fully offline, Modular AI workspace (LLMs + Agents + Automation + Image Gen)**](https://i.redd.it/u6niruxjqo1f1.png) ([Score: 433, Comments: 118](https://www.reddit.com/r/LocalLLaMA/comments/1kq590b/clara_a_fully_offline_modular_ai_workspace_llms/)): **该图片展示了 Clara 的 UI，这是一个开源、完全离线的模块化 AI 工作空间，旨在统一 LLM（通过 Ollama 或 OpenAI API）、Agent、自动化（通过 n8n 集成）和本地图像生成（Stable Diffusion/ComfyUI）。界面强调拖放式仪表盘范式，用户可以添加小组件用于快速聊天、电子邮件收件箱、Agent 逻辑、代码执行等，从而在不依赖云服务或 API 密钥的情况下实现可定制的“AI 控制室”。现代化的设计和基于组件的架构强调了可扩展性和以用户为中心的工作流集成，并采用宽松的 MIT 许可证发布。** 有人提出了一个技术担忧，即 Windows Defender 将 GitHub 的可执行文件报告为病毒，这暗示了分发过程中可能存在的问题或误报。人们对该仓库的开放访问也表现出浓厚兴趣，纷纷索要仓库链接，并对这款先进本地工具采用宽松许可表示赞赏。
    - 一位用户报告称 Windows Defender 将提供的可执行文件标记为病毒，这可能会引起注重安全的用户对误报或代码签名及透明构建过程需求的担忧。
    - 有人积极评价了模块化 API 的集成，但指出目前的 UI 仅允许一次添加一个 API 密钥。例如，用户可以添加一个 OpenAI 密钥，但据报道没有明显的方法同时添加多个提供商（如 Anthropic, Gemini）。
    - 一个深刻的观点指出，Clara 在工作空间内直接集成了备受推崇的工作流自动化工具 n8n。与标准的 LLM 或图像生成应用相比，这显著提升了其自动化和 Agent 能力。

- [**使用 Kokoro-JS 实现无限文本转语音，100% 本地，100% 开源**](https://streaming-kokoro.glitch.me/) ([Score: 159, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1kpw9nw/unlimited_texttospeech_using_kokorojs_100_local/)): **Kokoro-JS 是一个开源的客户端文本转语音（TTS）系统，完全在浏览器中运行，使用 ONNX 格式的 Kokoro-82M-v1.0 模型（约 300MB）。该 JS 实现利用本地资源（WebGPU/WebGL）进行推理，实现了无需任何服务器交互的离线语音合成；支持语音选择和多种声音，尽管 Firefox 用户必须手动启用** `dom.webgpu.enabled` **且目前将音频保存到磁盘存在限制。完整的源代码和演示已发布（[GitHub](https://github.com/rhulha/StreamingKokoroJS), [Demo](https://rhulha.github.io/StreamingKokoroJS/)）。** 评论区讨论了这是否与 open-webui 中使用的 Kokoro 模型相同，但未提供明确的技术答复；浏览器/WebGPU 的兼容性和离线隐私被强调为核心用户优势。
    - Kokoro-JS 的实现通过下载一个约 300MB 的 AI 模型实现 100% 本地操作，该模型完全在用户的浏览器中运行，不涉及服务器端数据。代码库已开源并托管在 [GitHub](https://github.com/rhulha/StreamingKokoroJS)，同时提供包括 [Glitch 项目](https://glitch.com/edit/#!/streaming-kokoro)在内的演示网站，允许用户直接测试流式 TTS 功能。
    - 关于浏览器支持的重要技术更新包括：在 Firefox 的 `about:config` 中启用 `dom.webgpu.enabled = true` 和 `dom.webgpu.workers.enabled = true` 以支持 WebGPU；开发者指出，目前在 Firefox 中无法将生成的音频保存到磁盘，这可能是由于浏览器在文件处理或 WebGPU 集成方面的限制。
    - 一位用户询问此 Kokoro-JS 版本是否与 Open WebUI 中使用的版本相同，这表明人们对基于 Web 的 TTS 模型部署的便携性和兼容性持续关注。另一位用户将本地 TTS 过程与基于 OpenAI Whisper 的转录模型进行了比较，并提到了其他移动应用，凸显了语音合成和识别领域完全转向客户端 AI 模型执行的广泛趋势。
- [**OuteTTS 1.0 (0.6B) — Apache 2.0, 批量推理 (~0.1–0.02 RTF)**](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) ([Score: 122, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1kq6ysz/outetts_10_06b_apache_20_batch_inference_01002_rtf/)): **OuteTTS-1.0-0.6B 是一个基于 Qwen-3（LLM 架构）的 0.6B 参数多语言 TTS 模型，采用 Apache 2.0 协议发布，并针对高效批量推理进行了优化，在单张 NVIDIA L40S 上可达到约 0.1–0.02 RTF（参见基准测试：例如使用 vLLM FP8 时 32 线程可达 0.05 RTF，使用 EXL2 6bpw 时为 0.106，详情见[此处](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B)）。该版本包含支持多种推理后端（vLLM, EXL2, llama.cpp）的 Python (**`outetts` **v0.4.2）支持，具有连续批处理、外部 URL 模型服务以及基于说话人参考的语音合成/语音克隆等功能。采样参数（例如：最后 64 个 token 的重复惩罚为 1.1，temperature 0.4，top-k 40，top-p 0.9）以及 ibm-research/DAC.speech.v1.0 编解码器的使用对于高质量输出至关重要，该模型在多样化的多语言语料库（MLS, Common Voice）上进行了训练。** 评论中的主要技术问题集中在 TTS 模型是如何从 Qwen3（一个 LLM）衍生出来的，并索要论文或方法细节。人们对演示音频、与早期 OuteTTS 版本的对比评估以及对公共 Playground 上部署的具体模型的澄清表现出浓厚兴趣。
    - 一位用户询问了 OuteTTS 1.0 的技术基础，特别是质疑如何基于主要作为 LLM（大语言模型）的 Qwen3 构建 TTS（文本转语音）模型。他们询问是否有可用的论文或技术细节，表明需要澄清模型架构、Qwen3 对 TTS 的适配以及可复现性信息。
    - 另一场技术讨论引用了“2cent-tts”项目，该项目利用更小的（60M 参数）Qwen3 模型来实现更快的推理。这种方法使用音素（phoneme）输入和 SNAC 解码器来优化性能，为高效 TTS 提供了潜在策略，并引导人们在参数量、速度和架构方面与 OuteTTS 进行比较。
    - 还有关于该模型能力的咨询，特别是关于语音克隆（voice cloning）方面。这引发了关于模型对说话人自适应或基于样本的语音复制支持的问题，这些是许多 TTS 应用的重要技术特性。

### 3. ParScale 模型发布与并行扩展论文

- [**Qwen 发布了新论文和模型：ParScale, ParScale-1.8B-(P1-P8)**](https://i.redd.it/7q0xsc86um1f1.png) ([Score: 440, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1kpyn8g/qwen_released_new_paper_and_model_parscale/)): **该图总结了 Qwen 团队的新 ParScale 模型和论文，介绍了一种用于 Transformer 的并行扩展（parallel scaling）方法。ParScale 使用 P 个并行流进行训练/推理，其分析表明，使用 P 个流进行扩展在理论上与增加 O(log P) 的参数量相当（即，流数量翻倍近似于增加常数倍的参数）。图中的视觉效果对比了 ParScale 的方法与传统的稠密模型（dense models）以及混合专家模型（MoE），指出 ParScale 提高了对数扩展效率、推理性能和通用适用性（适用于各种模型的“插件”能力）。图中的结果表量化了与仅增加参数的扩展方式相比，改进或具有竞争力的性能。** 评论强调了社区的兴奋，指出 ParScale 避免了 MoE 的缺点（内存/计算效率低下、复杂的路由），并推测它只需极少的微调即可推广到许多模型。与 MoE 相比有一个技术上的区别：ParScale 利用重复参数进行并行计算，而不是选择性的专家激活。
    - 讨论中的关键统计数据：据称 ParScale 与替代方案相比，内存增量减少高达 `22x`，延迟增量减少 `6x`——尽管一位评论者批评了这种统计措辞，建议将其表达为减少到 `4.5%`（内存）和 `16.7%`（延迟）。
    - 从技术上讲，ParScale 与 Mixture of Experts (MoE) 形成对比：MoE 是“存储很多，通过选择性计算很少”，而 ParScale 是“存储很少，通过带变体的重复进行大量（并行）计算”，专注于并行性和参数效率。
    - 有观点认为 ParScale 可能是一种通用的技术，用于在扩展过程中减少计算和内存消耗，可能适用于任何只需适度微调的模型，如果得到验证，将产生广泛影响。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. 主要由 AI 驱动的企业裁员和劳动力重组

- [**AI 裁员开始**](https://i.redd.it/qqiilvmesr1f1.png) ([Score: 187, Comments: 35](https://www.reddit.com/r/singularity/comments/1kqgxpm/the_ai_layoffs_begin/)): **分享的图片总结了近期大型科技公司（Microsoft, Google, PwC, Salesforce, Meta, Duolingo, Dell 等）的裁员情况，强调这些裁员要么直接与 AI 驱动的重组相关，要么以此为由（例如，转向 AI 优先策略或消除在 AI 驱动背景下被认为相关性较低的某些角色）。图片通过员工百分比或人数对裁员进行了量化，并详细说明了受影响的部门（如营销、销售、非技术角色），突显了随着 AI 集成加速，行业稳定性的担忧。一些被引用的公司将资源转移（如 Duolingo 投资 AI 基础设施或 Dell 计划通过 AI 实现自动化）作为裁员的理由。** 热门评论认为，在某些情况下（例如 Google 裁减约 200 名销售人员），裁员被夸大为 AI 驱动，暗示更广泛的经济和商业周期因素可能发挥了更重要的作用。也有人对大规模 AI 驱动裁员的说法表示怀疑，一些人认为科技行业的裁员是每年的常规现象，随后会重新招聘，AI 对职位流失的全面影响尚待观察。
    - 讨论强调了对将 Alphabet (Google) 等大型科技公司最近的裁员完全归因于 AI 的怀疑，一位评论者指出仅削减了约 200 个销售职位，而非直接受 AI 自动化影响的角色。
    - 对于观察到的裁员是否与 AI 进步有因果关系，还是与其他宏观经济因素（如利率和关税）有关，或者公司是否利用“AI”裁员作为年度常规劳动力调整或战略成本削减的掩护，存在细致的辩论。
    - 一些参与者指出，科技行业经常经历裁员和重新招聘的周期，这表明当前的发展可能并不代表一个显著的 AI 驱动转折点，而是持续的业务流程优化，只是向投资者提供了不同的外部理由。

- [**AI 裁员潮开始**](https://i.redd.it/gadex0zasr1f1.png) ([Score: 322, Comments: 118](https://www.reddit.com/r/OpenAI/comments/1kqgx8i/the_ai_layoffs_begin/))：**该图片汇总了近期主要科技和咨询公司（Microsoft, Google, PwC, Salesforce, Meta, Chegg, HP, IBM, Duolingo, Dell 和 Klarna）的裁员数据，将大部分裁员归因于围绕 AI 采用和效率提升进行的重组。裁员百分比或绝对人数与公司领导层及受影响部门一同列出，旨在将 AI 集成和相关战略转型引发的行业性转变背景化。裁员理由从直接的 AI 集成到更广泛的疫情后调整以及公司特定的重组不等。** 评论者表示怀疑，指出将裁员理由归结为 AI 驱动往往更多是为了向投资者展示积极叙事，而非明确的 AI 影响证明，因为真实的劳动力市场信号（职位发布、招聘率、失业率）尚未强烈反映出 AI 导致的就业减少。一些人指出，在没有实质性证据的情况下将裁员（如 Intel 的裁员）归因于 AI 相关原因，可能存在叙事上的“樱桃拾取”（cherry-picking）。
    - 讨论强调了对将裁员完全归因于 AI 的怀疑，认为这种叙事可能是为投资者关系服务的，而非基于技术效率的提升。评论者指出，目前尚未出现具体的证据——如技术职位发布（如客户服务、软件开发）的下降或科技行业失业率的上升，强调需要进行严谨的劳动力市场数据分析。
    - 一位用户分享了 ChatGPT 生成的 API 代码落后于当前几个大版本的负面经历，这表明当前的生成式模型在处理最新编码任务和版本管理方面存在实际局限性。该评论强调了模型上下文披露不足的问题，以及 LLM 需要更好地识别和传达代码补全中所使用的代码库版本。
    - 引用案例提到了 Klarna 在据称为 AI 自动化裁员 700 人后又重新招聘，这凸显了一个运营现实：声称的 AI 驱动效率有时无法实现，导致不得不恢复人工岗位。这指向了用当前的 AI 工具替换复杂工作流时面临的实施挑战。
- [**AI 裁员潮开始**](https://i.redd.it/9z1kxvub6r1f1.png) ([Score: 461, Comments: 132](https://www.reddit.com/r/ChatGPT/comments/1kqdukp/the_ai_layoffs_begin/))：**信息图表 [The AI Layoffs Begin](https://i.redd.it/9z1kxvub6r1f1.png) 直观地详细描述了 Microsoft, Google, PwC 和 Meta 等领先科技公司近期的大规模裁员，将每家公司的裁员（附带明确的裁员数字和高管肖像）与其声明的采用或转向 AI 驱动战略相关联。图片暗示了一种趋势，即大规模裁员被框定为 AI 集成或自动化计划的结果，强调了行业向利用人工智能提高运营效率的转变。** 热门评论挑战了裁员完全归因于 AI 的说法，指出像 Microsoft 这样的公司一直有与 AI 进展无关的周期性裁员历史，并强调了其他因素，如失败的业务尝试（例如 Meta 对 VR/AR 的押注）。此外，用户对净就业效应也持怀疑态度，要求提供对比招聘数据以了解真实的劳动力影响。
    - 评论者强调，将近期裁员严格归因于 AI 简化了问题；例如，Meta 的裁员与其在 VR/AR 和 Metaverse 上的昂贵赌注有关，而非直接的 AI 替代，这说明战略业务决策和重心转移仍然是主要驱动因素。
    - 一位用户要求提供更细粒度的统计数据——具体而言，不仅是裁员人数，还有同一时期的技术就业净数据，以及招聘数据，以便更好地将 Microsoft 和 Dell 等大公司的裁员背景化。
    - 另一个提出的技术点是影响规模，引用了 Dell 裁员 12,000 人的报道，这凸显了裁员在以硬件为中心的部门中可能非常显著，而不仅仅是在软件或 AI 开发团队中。

### 2. 即将发布及已发现的 AI 模型发布（Gemini, Claude, o1-pro）

- [**2.5 pro deepthink?**](https://i.redd.it/qm30m4bhup1f1.jpeg) ([Score: 259, Comments: 32](https://www.reddit.com/r/singularity/comments/1kq8rza/25_pro_deepthink/)): **附带的截图显示了一条推文，揭示了 Google Cloud API 控制台中出现了一个新的 Google 模型变体 'Gemini-2.5-pro-deepthink'，以及相关的 'gemini-2.5-pro-exp' 变体。这表明 Google 正在实验或发布一个专注于深度推理或扩展上下文处理的 Gemini 2.5 Pro 版本，这与近期行业向提供更先进推理能力模型（如 OpenAI 的 'o3-pro'）发展的趋势相一致。** 评论者将 'deepthink' 与 OpenAI 的模型进行了比较，并指出之前的 2.5 Pro 版本在推理方面没有实质性改进，希望 'deepthink' 能填补这一空白。
    - 一位用户对 Gemini（可能指 2.5 Pro 模型）发表了评论，指出其强大的能力，但也批评其容易丢失 Prompt 指令，并偶尔重复或粘贴旧文本。这表明在 Context Window 或 Attention 管理方面仍存在局限性，而这对于在长期交互中保持连贯、目标导向的对话在生产环境使用中至关重要。
    - 还有人将其与 Google 的 o3-pro 进行比较，暗示讨论中的模型在质量或方法上可能与 Google 领先的中端语言模型相似。这在主要竞争对手的 'pro' 层级产品之间划出了平行线，暗示市场在特定模型能力上的趋同。
- [**o1-pro just got nuked**](https://www.reddit.com/r/OpenAI/comments/1kq5wc5/o1pro_just_got_nuked/) ([Score: 161, Comments: 82](https://www.reddit.com/r/OpenAI/comments/1kq5wc5/o1pro_just_got_nuked/)): **该帖子报告称，OpenAI 的 o1-pro 模型（此前被认为是处理复杂编程任务的顶级模型，价格约为每月 200 美元）的性能和代码生成能力在过去几天大幅下降，表现为回复简短、细节不足以及明显的代码输出抑制——这表明增加了新的过滤器。值得注意的是，在 o3 发布后，o1-pro 已经被标记为 legacy，这暗示这可能是为了即将发布的版本或支持 Codex SWE 需求而进行的弃用和资源重新分配的一部分。尽管影响重大且订阅费用高昂，但并未向用户提供有关这些更改的官方沟通。** 顶级技术评论强调：(1) 在 o3 发布后，内部已知 o1-pro 的弃用状态，因此降级符合计划中的下线和资源分配策略；(2) 一个主要担忧是缺乏用户透明度，尤其是用户依赖 o1-pro 进行其他模型失败的 Bug 检测；(3) 一种愤世嫉俗的观点将其比作科技平台常见的 'enshittification'——在占领市场后降低服务质量以提高利润。
    - 评论者指出，在 o3 发布后，o1-pro 被标记为 legacy 并被弃用，最近的关闭被视为资源重新分配的一部分，可能是为了专注于 Codex SWE 的需求或未来的发布。
    - 一项技术相关的讨论指出，o1-pro 在 o3 发布后经历了显著的降级（'nerfs'），包括缩短 'thinking time' 和减少最大响应长度，导致其在停用前输出质量明显下降。
    - 另一条评论强调了具有上下文感知能力的代码生成工具（如 Copilot）的价值，它直接在 IDE 中运行并专门用于代码补全任务，将其与 ChatGPT Pro 等基于聊天的 AI 系统相关的通用方法和更高成本进行了对比。
- [**Amanda Askell is the woman in charge of giving Claude its personality, so I am ever grateful to her**](https://i.redd.it/gfagz8bkom1f1.png) ([Score: 266, Comments: 44](https://www.reddit.com/r/ClaudeAI/comments/1kpy0me/amanda_askell_is_the_woman_in_charge_of_giving/)): **该图片是来自 Anthropic 负责 Claude 个性和 Alignment 的关键人物 Amanda Askell 的推文截图，幽默地描述了她处于 'safety skeptics'（安全怀疑论者）和 'capability deniers'（能力否认者）之间的处境。这条推文反映了她在平衡 AI Safety（防止伤害/审查）和 Capability（提升 Claude 的功能）之间的权衡角色，这是 AI 部署中的两个核心张力点，特别是对于商业语言模型。这突显了围绕负责任的 Scaling、Alignment 以及 Anthropic 等 AI 公司面临的外部/内部压力的持续辩论。** 评论者对企业动机持怀疑态度（将这种平衡行为视为 PR 而非真正的管理），并对安全约束的影响表示担忧，一些人将安全对齐比作 'muzzling'（戴上口罩/禁言），并批评此类角色的中立性。其他人则认为该帖子只是简单地提到平衡利益，没有深层的技术含义。

- 一位评论者引用了 Anthropic 自身的研究和内部论文（如 Kyle Fish 共同撰写的论文），这些论文建议允许像 Claude 这样的模型终止对话，以此作为解决潜在道德受体地位（moral patienthood）或福利问题的一种姿态。批评指出，尽管在已发表的文献中讨论了这些伦理保障措施，但 Anthropic 尚未实施此类功能——即便类似的功能（如“停止按钮”）在 2023 年的 Bing AI 中就已经存在，这引发了人们对该公司在公共话语、研究与实际产品功能之间一致性的质疑。
- 存在对 Anthropic 的 AI safety 方法和定位的技术性怀疑：一位评论者将“左边是安全怀疑论者，右边是能力否认者，中间是受困者”的框架类比为一种修辞手段，掩盖了公司审查以及围绕 Claude 实际能力和约束的透明度有限的问题。这种元层面的批评指出了所宣称的中立性与模型在操作上强制执行的限制之间的差距。

### 3. AI Progress, Automation Impact, and SWEs Replacement Discourse

- [**Timeline of SWEs replacement**](https://i.redd.it/apqpgkow2q1f1.jpeg) ([Score: 680, Comments: 206](https://www.reddit.com/r/singularity/comments/1kq942u/timeline_of_swes_replacement/)): **这张图片是一个讽刺性的时间线，展示了围绕承诺取代软件工程师的技术所产生的周期性炒作，列举了 COBOL, SQL, Visual Programming, MDA, No-Code 以及现在的基于 AI 的 'Vibe Coding' 等例子，每一次迭代最终都未能消除对软件工程专业知识的需求。评论者提供的技术背景指出，虽然每项技术都提高了开发者的生产力，但市场动态已经发生了变化：最初，计算机普及率的增长使得这些改进能够吸收新的需求，但如今市场饱和以及 neural networks 的出现带来了新的挑战。** 评论者强调，利用过去失败的预测来否定新技术是一种逻辑谬误——*每一波技术的优劣都应该被独立评判*。一些人还认为 neural networks 的变革潜力有着本质的不同，暗示需要对当前趋势进行细致的评估。
    - 几位评论者将历史上开发者生产力的提升（COBOL, SQL, VBA）与现代进展进行了对比，指出过去的工具改变了开发过程，但通常是扩大了市场，而不是减少了 SWEs 的机会。然而，他们强调 neural networks 和 AI 引入了本质上不同的动态；与过去的工具不同，AI 有可能完全*取代*手动编码，而不仅仅是加速编码，这暗示了对就业市场的更长期影响。
    - 一场讨论强调，虽然 'no code' 平台经常被引用为现代生产力增强工具，但特定的技术进步（如现代 Web 工具和服务器端的 JavaScript）可以说对开发者效率产生了更大的影响。这与 AI 形成了对比，AI 被定位为在性质上有所不同，因为它在工具中引入了认知能力，而不仅仅是新的抽象或框架。
    - 出现了经济分析，几位评论者关注当前计算机市场的饱和以及资本配置的转移。他们观察到，之前的生产力提升发生在扩张的市场中，而 AI 驱动的变革发生在市场饱和之后，这可能会加剧对工作流失的担忧并加速颠覆性影响。

- [**我意识到 AI 编程能力超过我的那一刻**](https://www.reddit.com/r/ChatGPT/comments/1kq8t4t/the_moment_i_realized_ai_could_code_better_than_me/) ([Score: 949, Comments: 282](https://www.reddit.com/r/ChatGPT/comments/1kq8t4t/the_moment_i_realized_ai_could_code_better_than_me/))：**原作者（OP）描述了一个实际用例，他们利用 AI 编程助手来调试和重构一个软件函数，最终获得了一个比自己编写的版本更简洁、更高效的解决方案。这个轶事突显了当前 AI 模型在代码分析、调试和重构方面的能力，展示了软件开发工作流中的生产力提升。热门评论提供了 AI 在自然语言处理（NLP）中效用的平行案例，例如从充满错误的会议记录中提取语义结构和上下文——特别是 AI 解决了不一致的名字拼写，表明上下文感知 NLP 模型取得了重大进展。同时也承认了局限性，例如 AI 在复杂的 Bug 修复循环中挣扎，并在边缘情况下输出“胡言乱语的代码”，反映了 AI 编程可靠性的持续边界。** 评论者注意到 AI 在 NLP 和编程中令人印象深刻的可解释性，但表达了谨慎态度：AI 生成的解决方案在复杂的调试场景中经常失败，需要人工代码审查。这指向了一个共识：AI 擅长某些类型的自动化，但在处理更细微或依赖上下文的代码问题时仍需要人工监督。
    - ChatGPT 通过从嘈杂、易错的 Zoom 转录文本中提取准确的会议结果和参与者角色，展示了强大的自然语言理解能力，包括在没有额外上下文的情况下一致地推断和规范化拼错的艺术家名字——这表明其具备超越表层解析的高级上下文推断和实体解析能力。
    - 虽然 AI 可以生成令人印象深刻的编程解决方案，但在迭代修复 Bug 时经常失灵：用户报告称陷入了无效率的反馈循环，AI 的解决方案反复未能解决底层问题，揭示了模型推理和 Bug 诊断的局限性，特别是在处理复杂或模糊的代码时。
    - AI 最常见的编程错误归因于模型在 *context management*（上下文管理）和 *attention mechanism*（注意力机制）约束方面的局限，而非根本性的编程错误。然而，在 Prompt 和代码符合模型注意力窗口的理想场景下，生成的解决方案可能超出预期，突显了与当前架构约束相关的质量不可预测性。

---

# AI Discord Recap

> 由 Gemini 2.5 Flash Preview 生成的摘要之摘要
> 

**主题 1. AI Agent 开发与编排工具**

- **MCP 协议连接多样化的 AI Agent**：讨论强调了 **Model Context Protocol (MCP)** 能够实现 AI Agent 之间的通信，甚至跨越不同机器，据报道 *Qwen 3 235B* 原生支持该协议。用例涵盖从 Discord 机器人到复杂的多 Agent 协作，由 [这个 agentapi 仓库](https://github.com/coder/agentapi) 和支持 SSE 的新 [MCPControl server v0.2.0](https://github.com/Cheffromspace/MCPControl/releases/tag/v0.2.0) 提供支持。
- **DSPy 和 LlamaIndex 构建 Agent 工作流**：工程师们正利用 **DSPy** 和 **LlamaIndex** 等框架构建 Agent，**DSPy 2.6** 用 `BestOfN`/`Refine` 替换了 `Suggest`/`Assert` ([教程](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine))。**LlamaIndex Agents** 现在具备改进的 [长期和短期记忆](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents)，并支持 [使用 Weaviate 的多 Agent 工作流](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent)。
- **新的 SDK 和接口提升 Agent 能力**：Amazon 发布了 [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/) 以简化 Agent 创建，而 **Sherlog Canvas (Alpha)** 提供了一个 [AI 驱动的调试接口](https://github.com/GetSherlog/Canvas)，集成了用于日志和指标的 MCP 驱动单元，并附有 [演示视频](https://youtu.be/80c5J3zAZ5c)。一个新的 [MCP UI SDK](https://x.com/idosal1/status/1923477857881190718) 也为 MCP 服务器增加了丰富的 Web 交互能力。

**主题 2. LLM 性能、评估与模型行为**

- **Gemini 模型因性能和怪癖面临审查**：用户观察到 **Gemini** 模型中出现了诸如注释掉代码和删除注释等奇怪行为，并指出了一种权衡：**Gemini 2.5 Pro 0506** 在编程方面表现更好，但旧版本（如 **03-25**）在数学方面更佳，引用[谷歌官方数据](https://deepmind.google/technologies/gemini/pro/)显示其性能分别为 **83% 对比 88.9%**。**Gemini 2.5 Pro Experimental** 的弃用也因新版本中的过滤问题引起了不满。
- **GPT/O 系列引发关于架构和发布的猜测**：外界纷纷猜测 **GPT-5** 可能会舍弃 **o3** 组件，采用类似于 **Gemini 2.5 Pro** 的结构，将 LLM 和推理模型结合在一起，可能在今年夏天发布。**O3 Pro** 的持续推迟引发了挫败感，一些用户认为原始的 **GPT-4** 拥有新模型所缺乏的“真正的智能/创意火花”。
- **基准测试和评估技术成为焦点**：**AgentX Competition** 提供了超过 [$150K](https://x.com/dawnsongtweets/status/1924470174776004875) 的奖金，并要求参赛者在实验中使用自己的 **OpenAI API keys**，这引发了关于使用 [lm eval harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) 进行自定义模型评估的讨论，特别是针对 **SpinQuant Llama-2-7b** 等量化模型。**GSM8K** 和 **MATH** 等基准测试显示，在 **Reasoning Gym** 数据上训练的小模型性能有所提升，**Qwen 2.5 3B** 在某些任务中表现优于 **Llama 3.2 3B**。

**Theme 3. 硬件性能与底层优化**

- **GPU 硬件之战：VRAM、效率与新玩家**：[**Intel Arc Pro B60**](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) 以其 **96GB VRAM** 和低于 2000 美元的价位让本地 LLM 用户感到兴奋，尽管存在软件方面的担忧；而 Macbook 则因能在 <2GB RAM 占用的情况下静音运行 **32B 4-bit 模型超过 5 小时**而受到赞誉。在 **9070XT** GPU 上启用 **Resizable BAR** 后，**LM Studio** 在 **Gemma3 4b QAT** 模型上的性能从 *8.52 t/s* 大幅提升至 *131.91 t/s*。
- **Triton、CUDA 和 ROCm 面临实现挑战**：用户在将 **FSDP** 和 **Flash Attention 2** 与 `trl` 集成时遇到困难，原因是与未在 GPU 上初始化的模型不兼容，并且正在调试神经网络变异函数中的 CUDA 错误，如 *unspecified launch failure* 和 *illegal memory access*（[godbolt 链接](https://cuda.godbolt.org/z/z8z6a85vP)）。在 **ROCm Triton** 中，关于 `kpack` 参数对性能影响的讨论异常激烈，Torch Inductor 在使用 `ld128` 的 **MFMA** 时默认将其设置为 **2**。
- **通过量化和内核优化性能**：GPU Mode Discord 中的讨论探索了 **FP8-MM** 和 **Mixture-of-Experts** 在 **MI300** 硬件上的性能，用户在排行榜上分别实现了低至 **150 µs** 和 **9.64 ms** 的惊人耗时。用户还在尝试使用 [Axolotl 配置](https://github.com/axolotl-ai-cloud/axolotl/pull/2590/files#diff-29a7a53548f024812e8a2dc36eccc625c6b278b22b105f4eb5a924c63452a781) 对 **Llama3.2 3B** 等模型进行**量化感知训练 (QAT)**，并探索来自 CUTLASS 团队的 **CuTeDSL**（[博客文章](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/)）以进行底层内核优化。

**Theme 4. 新 AI 模型、研究与新兴概念**

- **新兴 AI 概念：自学型推理器与世界模型**：一种名为 **Absolute Zero Reasoner (AZR)** 的新型 AI 能够在没有人类数据的情况下从零开始自我学习（[YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0)）；同时，**OpenWorldLabs** 专注于为[机器人和视频游戏构建世界模型](https://openworldlabs.ai/)，尽管部分用户认为其使命尚不明确。讨论还涉及通过对上下文帧重新加噪（renoising）来解决扩散模型（diffusion models）中的**累积误差**问题，并指出目前的视频模型速度太慢，但 Google 可能会通过带有误差的训练来实现逐帧去噪。
- **LLM 产生自发的社会规范与偏见**：一项研究（[arxiv 链接](https://arxiv.org/abs/2505.10475)，[science 链接](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)）表明，**普遍采用的社会规范**可以通过局部交互在去中心化的 LLM 群体中自发产生，即使单个 Agent 最初没有偏见，也会导致强大的集体偏见。值得注意的是，由**对抗性 LLM Agent** 组成的坚定少数群体可以通过强加替代规范来推动社会变革。
- **特定研究领域探索：文档 AI、MCMC 与赋能**：一位数据科学家正专注于**文档 AI 与 NLP**，包括表示、TSR 和质量评估，以及关于金融和伦理困境的个人项目。讨论了 **MCMC** 算法的功耗研究（[论文示例](https://arxiv.org/pdf/1903.11714)）以及概率位（**pbit**）硬件在提高效率方面的潜力，同时还讨论了 AI 中尚未被充分探索的**赋能（Empowerment）**概念，这与 **Muon** 的高昂操作成本特别相关。

**主题 5. AI 工具与平台更新**

- **NotebookLM 推出移动端 App 并支持视频上传**：**NotebookLM 移动端 App** 已在 [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) 和 Android 上线，具备 **MVP 功能集**，但缺少思维导图和简报等核心功能；而网页版现在支持[带自动转录功能的视频上传](https://x.com/testingcatalog/status/1924246774346104863)。用户注意到其写作风格变得*异常简单且还原化*，类似于**高中作文水平**，并报告了上传与社会正义相关材料时遇到的问题，怀疑存在*审查*。
- **OpenRouter 应对 API 变更与供应商问题**：Google 正在 OpenRouter 上弃用 **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`)，转而使用付费端点，而免费的 **DeepSeek V3 0324** 正在维护中。用户报告 **Qwen3 235B** 的供应商 *Kluster* 会过早终止工具调用（tool calls），导致 OpenRouter 需要切换供应商来解决该问题。
- **Hugging Face 及相关工具更新**：**LlamaIndex** 宣布了 **LlamaParse** 的更新（[推文](https://twitter.com/llama_index/status/1923510823164706925)），并获得了 [Azure AI Foundry Agent Service](https://twitter.com/llama_index/status/1924502129974411504) 的原生支持；同时 Vitalops 发布了 [datatune](https://github.com/vitalops/datatune)，这是一个通过自然语言进行数据转换的开源工具。据报道，提供 6 个月免费 Hugging Face Pro 的 [AI Engineer Pack 链接](https://discord.com/channels/879548962464493619/1353741359059570699)对部分用户无效。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **You.com 胜率误读**：一位成员分享了一个 [Cunnyx 链接](https://cunnyx.com/youdotcom/status/1923100692098453703)，澄清了 **You.com** 上显示的百分比代表的是胜率（win rate），而非质量。
   - 讨论强调了误读 **You.com** 指标的可能性。
- **MCP SuperAssistant 与 Perplexity 配合良好**：一位用户报告称，**MCP SuperAssistant** 扩展在多个服务器上的 Perplexity 中运行良好，尽管偶尔会出现断连，并分享了 [MCP SuperAssistant 网站](https://mcpsuperassistant.ai)和 [GitHub 仓库](https://github.com/srbhptl39/MCP-SuperAssistant)的链接。
   - 社区探讨了将外部助手（assistants）与 **Perplexity** 集成的实用性。
- **Grok Deepsearch 略胜 Perplexity Pro**：一位用户表示在特定的研究任务中更倾向于使用 **Grok Deepsearch** 而非 **Perplexity Pro**，理由是在使用 Python 数学计算比较各国食品价格时，前者结果更优。
   - 尽管使用了 **Perplexity Pro**，该用户发现 Perplexity 的研究任务中缺少 **Python math tool**，这导致他们转向 **Grok**。
- **Firefox 宣传 Perplexity**：[Firefox 将推广 Perplexity 搜索引擎](https://windowsreport.com/mozilla-firefox-to-promote-perplexity-search-engine/)，这引发了关于其在 Android 上采用情况的疑问，并触发了关于浏览器引擎偏好（**Blink** vs. **Gecko**）的辩论。
   - 该公告引发了关于 **Firefox** 未来及其针对基于 **Blink** 浏览器竞争定位的讨论。
- **Perplexity Sonar API 与 UI 差异巨大**：用户报告称 **Perplexity Sonar API** 与 **UI 版本** 之间的结果显著不同，指出在来源信息方面存在差异，且难以通过 **API** 复现 **UI** 的结果，并提到了一个用于测试的 [GitHub 仓库](https://github.com/alanbuxton/perplexity-api-tester)。
   - 讨论质疑了 **Sonar API** 的一致性和可配置性，以及 `top-p` 等参数是否会影响实际搜索。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Convex 在实时性方面击败 Supabase**：虽然 **Supabase** 提供实时功能，但 **Convex** 因其自动同步和事件驱动架构而在实时应用中更受青睐，参见[此对比](https://x.com/mntruell/status/1924297275993669893)。
   - 一位成员指出了 **Supabase** 本地托管的优势，但承认 **Convex** 在实时性方面表现卓越，不过其 Auth、Edge Functions 和 Buckets 也非常出色。
- **MCP 热潮席卷 Agent 通信**：讨论强调了将 **MCP** (Model Context Protocol) 用于 AI agents，甚至跨不同计算机使用，并观察到 *Qwen 3 235B 正是因为这个原因原生支持 MCP*。
   - 使用场景涵盖了从 Discord 管理机器人到具有单一事实来源和上下文的复杂 Agent 间协作，[此仓库](https://github.com/coder/agentapi)支持了相关用例。
- **DeepSeek-R1T-Chimera 精通 Markdown**：**DeepSeek-R1T-Chimera** 模型因能精确编辑 .md 和 .mdc 文件以及打破 prompt 测试中的循环而受到赞誉，该模型可在 [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) 上找到。
   - 值得注意的是，据报道它是唯一能达到这种准确水平的免费模型，因为它是在 R1 和 V3 之间进行微调的。
- **Cursor 代码上下文体验糟糕！**：用户报告称 **Cursor** 请求缓慢且质量不稳定，引发了挫败感，有人表示：“我真的很讨厌现在发生的事情”。
   - 建议包括切换模型（如 DeepSeek 3.1）或重置上下文；一些用户遇到了在使用 Gemini 最高设置时“Apply All”按钮不显示的 bug。
- **在 Cursor 中巧妙处理文档导航**：成员们解释了 **Cursor** 通过解析链接页面的 HTML 内容来读取链接文档以收集信息，并指出使用 [Cursor 文档系统](https://docs.cursor.com/context/model-context-protocol) 能更好地保留所有代码。
   - 用户请求能够动态读取链接文档中的更多页面（模拟浏览器功能），团队宣布即将推出一个用于向 Cursor 发送 DOM 的 API 插件。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 imatrix 数据集引发关注**：成员们对 **UnSloth Dynamic 2.0 GGUFs** 感到兴奋，该系列以具有**范式转移意义、专注于指令和对话的 imatrix** 校准数据集为核心。
   - 据报告，改进后的困惑度（perplexity）使得 **Llama-3.1-8B-Instruct-UD-Q8_0_XL** 的 Token 生成速度更快。
- **Qwen3 GGUF 促使展开调查**：由于 SHA 哈希不匹配和越界错误，用户报告了运行 **Qwen3 235 128k UD-Q6** 和 **UD-Q8 ggufs** 时的问题。
   - 团队回应称将进行调查，并已*更新了所有修复了聊天模板的 Qwen3 上传文件*。
- **Colab 下载引起关注**：一名成员抱怨从 **Google Colab** 下载适配器（adapters）速度缓慢，并考虑更换服务。
   - 另一名成员证实了这一问题，并建议上传到 **Hugging Face** 以获得更快的下载速度。
- **Torch/CUDA 错误困扰 Colab 用户**：一名用户报告在更新 Unsloth 后遇到 **CUDA 错误**，经追溯是因为旧驱动程序需要特定的 Torch 版本。
   - 该用户通过使用 `pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"` 实施了一个自愈式[解决方案](https://www.urbandictionary.com/define.php?term=self-curing)。
- **PTS 助力 DPO 性能**：成员们讨论了 [Pivotal Token Search](https://huggingface.co/blog/codelion/pts) 本身不是 RL，但它能生成用于微调的 DPO 对，并引用了 **Phi-4 技术报告**中展示的改进。
   - 一张截图（[图片链接](https://cdn.discordapp.com/attachments/1257011997250424842/1373218344232157264/Screenshot_2025-05-17_at_4.37.39_PM.png?ex=682ce87e&is=682b96fe&hm=685beeb66d15e5f4e1195735b7a8b1cc7fb4145dc5dfc8f9cd2863e&)）直观地证实了 **PTS 对各项基准测试中 DPO 性能的积极影响**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ASI Lab 剽窃丑闻爆发**：**ASI Lab** 因涉嫌剽窃而遭到抨击，[pro.creations](https://cdn.discordapp.com/attachments/998381918976479273/1373015169357054082/image.png?ex=682cd405&is=682b8285&hm=41d43e15460edc8f88272551f18cfec1fa74fb94c748b9efa15f5c82234bb031&) 报告称他们的作品被一所*知名大学*下架。
   - 该实验室被指控在剽窃的作品上冠以*虚假的 AGI 名称*，引发了网上的愤怒。
- **GPT-5 排除 O3，效仿 Gemini 2.5 Pro 风格？**：成员们推测 **GPT-5** 可能会放弃 **o3** 组件，通过集成 LLM 和推理模型来模仿 **Gemini 2.5 Pro**。
   - 市场对夏季发布的预期正在升温，可能与 **Google week** 活动同时进行。
- **Gemini 2.5 Pro 数学性能下降？**：社区成员正在将 **Gemini 2.5 Pro** 的性能与旧模型进行比较，观察到数学和编程技能之间的权衡，并引用了 [Google 官方数据](https://deepmind.google/technologies/gemini/pro/)。
   - 性能对比为 **83% vs 88.9%**，表明旧模型在数学方面表现更好。
- **HyperEnglish 提示词语法强化清晰度**：引入了一种新的提示方法 [HyperEnglish](https://example.com/hyperenglish)，使用*功能性标签*和技术术语的**全大写**来增强提示词的清晰度。
   - 该模板使用严格的结构，如 `SUBJ:{subject} VERB:{verb} OBJ:{object}`。
- **ProtoMind 语义映射绕过代码**：[ProtoMind_001](https://chatgpt.com/g/g-682759c7c8d881919436d320e718d5d5) 将用户输入视为**分层语义算子**。
   - 它能够很好地映射输入，并且消除了对角色切换和伪记忆线程（pseudo-memory threading）显式代码的需求。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro 的发布依然遥遥无期**：用户对 **O3 Pro** 的持续推迟表示沮丧，有些人甚至开玩笑说要通过“绝食抗议”直到它发布。
   - 一些用户认为原始的 **GPT-4** 更胜一筹，拥有更新、更小的模型所缺乏的“真正的智能/创意火花”。
- **GPT-5：增量升级还是基础模型？**：关于 **GPT-5** 的猜测不断，有人认为它将是与 **O4** 相当的基础模型，而另一些人则持怀疑态度，认为它可能只是 **O3** 的略微改进版本。
   - 讨论涵盖了它将是一个模型路由（model router）还是一个新的基础模型，以及 RL 训练如何影响其相对于稳定版本的改进。
- **Codex：开发者的良师益友还是洪水猛兽？**：**Codex** 的实用性正在引发辩论，一些人认为它对初级开发者来说是“噪音”，而另一些人则看到了它处理更高级任务的潜力。
   - 一位用户建议 **Codex** 需要与 **RooCode/Cline/Cursor/Windsurf/Aider** 等工具竞争才有价值。
- **Gemini 和 Claude 在代码领域展开对决**：这些 Bot 在编程方面互不相让，一些人觉得 **Gemini** 啰嗦得令人恼火，且倾向于添加多余的功能，而另一些人则称赞 **Claude** 的可靠性。
   - 一些用户认为 **Gemini** 的代码注释是一个负面特征，而另一些人则认为它们很有帮助。
- **LMArena Beta 迎来新面孔**：LMArena Beta 网站增加了新模型，包括 **mistral-medium-2505**、**claude-3-7-sonnet-20250219-thinking-32k**、**amazon.nova-pro-v1:0** 和 **command-a-03-2025**。
   - 自从 **Mistral Medium 3** 在排行榜上首次亮相以来，它实现了 **+90** 分的惊人跨越，超越了 Mistral Large，在聊天总榜中位列第 11。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Experimental 被停止支持**：Google 正在弃用 **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`)，转而支持付费的 **Preview endpoint** (`google/gemini-2.5-pro-preview`)。
   - 用户对失去 **Gemini 2.5 Pro Exp** (*03-25*) 表示哀悼，并对较新的 05-06 版本表示不满，指出了内容过滤（content filtering）的问题。
- **DeepSeek V3 为实现最佳性能进行调整**：免费的 **DeepSeek V3 0324** 暂时下线进行例行维护。
   - 用户应预料到在模型进行必要调整期间服务会出现短暂中断。
- **国际象棋锦标赛结合 Stockfish 和 OpenRouter 进化**：一位成员将他们的“国际象棋锦标赛”构想转变为一个“国际象棋排行榜”，结合 **Stockfish 实现**，利用 **OpenRouter 模型** 获取准确的 **Lichess 准确率评分**。
   - 该项目使用 cronjobs 自动进行评分，并适配了 o1-mini 等模型，展示了一个“不同的用例”。
- **通过请求标签（Request Tags）优化 API Key 追踪**：为了更好地追踪 API 调用来源，成员建议实施“请求标签”，而不是在应用名称中嵌入用户 ID，特别是为了在多个用户共享同一个 Key 时追踪流式传输中途的断连。
   - 这有助于详细记录单个用户的请求，并减少使用共享资源时的混乱。
- **Qwen3 的 Tool Calling 在 Kluster 上遇到障碍**：一位用户报告称，**Qwen3 235B** 在使用 *Kluster* 提供商时遇到 Tool Calling 问题，会过早结束工具调用。
   - 他们发现 OpenRouter 通过切换到另一个提供商解决了这个问题，这表明 *Kluster* 可能存在潜在的兼容性挑战。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 中的 VRAM 使用情况依然成谜**：用户反映 **LM Studio** 虽然显示了正确的 VRAM 数值，但实际使用率仍然非常低，即使在将模型层卸载（offloading）到 GPU 时也是如此。一位用户报告在 9070 机器上仅使用了 *256-512MB* 的 VRAM。
   - 他们正在调查潜在的驱动程序问题，以确定模型是加载到了专用 VRAM 还是共享显存中，并暗示这可能是一个驱动程序 Bug。
- **用户希望 LM Studio 支持对话导出**：用户正在寻求比当前 JSON 格式更灵活的 **LM Studio** 对话导出方式，特别是为了讲故事的需求。
   - 建议包括使用 LLM 将 JSON 解析为首选格式，同时也承认 JSON 结构缺乏 API 保证。
- **启用 Resizable BAR 以提升 GPU 速度**：一位用户报告在全新安装 Windows 后，使用 **9070XT** GPU 运行 **Gemma3 4b QAT** 模型最初仅获得 *8.52 t/s*，但在切换正确设置后看到了巨大提升。
   - 启用 **Resizable BAR**（或 AMD 上的 **Smart Access Memory (SAM)**）将性能提升至 *131.91 t/s*，展示了其对 LM Studio 的重大影响。
- **Arc Pro B60 GPU 引发关注**：成员们对 [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) GPU 表现出极大热情，称赞其 **96GB VRAM** 和预期的低于 2000 美元的价格，使其对本地 LLM 极具吸引力。
   - 尽管对软件支持仍有顾虑，但人们希望高 VRAM 的可用性能增强 Intel GPU 在 AI 领域的支持，并鼓励供应商改进其产品。
- **macOS：静默的 LLM 性能担当**：一位成员形容 Macbook 上的 **macOS** 比 Windows 更流畅，另一位用户提到了 MacOS 窗口大小调整的问题，而一位成员分享了 Macbook 如何在电池供电下以 5t/s 的速度运行 **32B 4-bit 模型超过 5 小时**。
   - Macbook 的配置拥有高效且静默的 LLM 运行表现，系统 RAM 占用小于 2GB。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MCP 课程频道：失踪了？**：成员们正在寻找 **MCP 课程频道**，并对该频道是否与 **Agents 课程** 相关表示困惑，同时分享了 [GitHub 上的课程链接](https://github.com/huggingface/mcp-course)。
   - 目前活跃的 **MCP 课程频道** 所在地或是否存在仍不清楚。
- **Hugging Face Pro 故障困扰 AI 工程师**：一位用户报告 **AI Engineer Pack 提供的 6 个月免费 Hugging Face Pro 链接**无法工作，并被建议通过 [website@huggingface.co](https://discord.com/channels/879548962464493619/1353741359059570699) 联系 HF 支持。
   - 此问题阻碍了 AI 工程师获取 **HF Pro** 相关的权益，可能影响他们的项目和工作流。
- **Strands Agents SDK：Amazon 简化 AI Agent 创建**：Amazon 推出了 [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)，这是一个设计精良的**开源 AI Agent SDK**。
   - 该 SDK 旨在简化 AI Agent 的开发，为开发者提供更高效构建和部署 Agent 的工具和资源。
- **datatune 使用 LLM 和自然语言转换数据**：来自 Vitalops 的一位成员介绍了 [datatune](https://github.com/vitalops/datatune)，这是一个新的**开源工具**，可以使用简单的自然语言指令和 LLM 执行数据转换。
   - 与传统的编码方法相比，这提供了一种更直观的数据处理方式，有望提高效率。
- **Lazarus 崛起！小型 LLM**：一位成员分享了 [Aclevo/Lazarus](https://huggingface.co/Aclevo/Lazarus)，这是目前表现最好的小型 LLM 之一，从 Llama3 蒸馏而来，参数量约为 **1.24 亿**。
   - 发布者根据其与其他 LLM 的尺寸对比提出了这一主张，并将 [从 Llama3 蒸馏](https://huggingface.co/blog/codelion/pts) 视为其成功的秘诀之一。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **成功获得 AMD Challenge 访问权限**：一位 **AMD challenge** 的参与者报告了在访问排行榜频道时遇到困难，在检查了特定频道 [<#1359640791525490768> 和 <#1343002583001726986>] 后，成功获得了访问权限。
   - 这突显了验证频道权限以及咨询社区内相关资源进行故障排除的重要性。
- **FSDP 与 Flash Attention 2 的集成摩擦**：一位成员寻求关于将 `FSDP` 和 `Flash Attention 2` 与 `trl` 集成以进行模型训练的建议，并参考了 [trl `sft.py` 脚本](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)。
   - 挑战在于当使用 **Flash Attention 2.0** 且模型未在 GPU 上初始化时存在不兼容性，从而导致错误。
- **Kernel 构建者寻求合作伙伴**：一位成员正在积极从零开始开发一个 kernel，并邀请他人贡献，项目详情可在 [GitHub](https://github.com/DorsaRoh/Kernels) 上查看。
   - 该项目被标记为 "gpu mode = 🔥"，表明其非常强调以 GPU 为中心的 kernel 开发。
- **神经网络噩梦：变异函数故障**：一位成员在处理 **神经网络** 的 **mutate function** 时遇到问题，在随机生成时触发了 *unspecified launch failure* 或 *illegal memory access* 等错误，代码见 [godbolt 链接](https://cuda.godbolt.org/z/z8z6a85vP)。
   - 错误提示为 *Malloc/Free Error encountered : Invalid pointer to free*，表明在变异过程中发生了内存损坏。
- **CuTe 演讲即将开始！**：**CuTe** 的发明者 Cris Cecka 正在进行一场关于 **CuTe** 的演讲，演讲已于 5 分钟前开始。
   - 这场演讲是直接从创作者那里深入学习 **CuTe** 的绝佳机会。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **金融科技数据科学家深耕文档 AI**：来自印度的数据科学家 Akhil Theerthala 正在开发 **Document AI 和 NLP** 项目，重点关注文档表示、TSR 和文档质量评估。
   - 他还在个人项目中探索关于 **个人理财**、**伦理困境** 以及 **简历/职业路径分析** 的推理。
- **AI_WAIFU 调整 AGI 时间线**：**AI_WAIFU** 缩短了他们的 AGI 时间线，预计在今年或明年年初实现，并将这一加速归功于编程模型，以及在离开 EAI 后开始从事新工作。
   - 他们指出，**神经网络 (NN) 效率** 的提升对小模型的益处大于对大模型的显著增强，而且纳米技术可能比 AGI 本身需要更多的算力。
- **OpenWorldLabs 开发世界模型**：**OpenWorldLabs** 专注于为机器人和视频游戏构建 **世界模型 (world models)**，可通过其 [官方网站](https://openworldlabs.ai) 访问。
   - 一些社区成员表示难以理解他们的目标，其中一人在阅读了网站和 GitHub 两次后表示：*我仍然几乎不知道你们到底在做什么*。
- **扩散模型面临误差累积问题**：成员们讨论了解决扩散模型中 **误差累积 (accumulating error)** 需要对上下文帧进行重加噪 (renoising)，但现有的视频模型对此处理速度太慢。
   - 一位成员提到，**Google** 可能已经通过训练带有误差的模型来逐帧去噪，从而解决了这个问题。
- **为量化模型定制 lm eval harness**：一位成员寻求指导，希望使用 **SpinQuant** 技术在来自 [Facebook Research](https://github.com/facebookresearch/SpinQuant) 的不同位精度量化的 **Llama-2-7b** 模型上复现零样本推理任务。
   - 建议该成员可以通过传递一个已初始化的模型来使用 *lm eval harness*，参考[此文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)，这需要将模型封装在一个类中，可能需要继承或修改 *lm eval harness* 中现有的 [HFLM 类](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L47)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Freeplay.ai 镜像了内部产品架构**：成员们讨论了 [freeplay.ai](https://freeplay.ai)，一位用户反馈了积极的初步印象，并指出它“镜像”了内部构建。
   - 另一位用户对更新表示了强烈兴趣，并询问了 **Freeplay.ai**、**Braintrust** 和 **Humanloop** 之间的显著差异。
- **Absolute Zero Reasoner 在没有数据的情况下自我学习**：一位成员分享了一个 [YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0)，详细介绍了 **Absolute Zero Reasoner (AZR)**，这是一种无需人类数据即可从零开始自我学习的新型 AI。
   - 该成员请求大家对视频提供反馈和想法。
- **AI 开发者要求进行开发者调查**：一位成员建议进行“State of AI”开发者调查，以追踪 **AI agent frameworks**、**SDKs**、**proxies** 和 **models** 等领域的使用趋势。
   - 成员们分享了 [2025 State of AI Engineering 调查链接](https://x.com/barrnanas/status/1904593314533564604)和 [Allstacks 调查](https://www.surveymonkey.com/r/6WQD6QQ)，强调了了解软件生产力提升的需求。
- **OpenAI Codex 应对 Shopify 应用创建**：一位成员尝试使用 **OpenAI Codex** 将现有应用程序 Answer HQ 转换为兼容的 **Shopify App Store** 应用。
   - 他们注意到 Codex 首先寻找 *agents.md* 文件，并且一份良好的 README 有助于简化流程，建议生成一个供 LLM 消耗的 AGENTS.md 文件，概述领域、关键命令、如何运行测试以及项目的规范。
- **Agent as Judge 缓解任务疲劳**：成员们讨论认为，任务疲劳度降低是由于上下文切换减少，并指出一位一直以 **0.1 倍产能**工作的精疲力竭的开发者，在使用 **Agent as Judge** 后恢复到了 **1 倍性能**。
   - 精疲力竭的开发者现在可以恢复到 1 倍或更高的性能，我们又回到了 *agent as judge* 的模式。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Codex o4-mini 发布在科技裁员潮中受挫**：**Codex o4-mini** 模型的发布恰逢大型科技公司裁减那些从事受人尊敬且有价值产品的人员。
   - 一位成员开玩笑说，扎克伯格可能很生气，因为整天冲浪让他看起来像个白痴，而马斯克带着 Grok 看起来像个天才。
- **Gemini 2.5 Pro 作为编程模型表现不佳**：成员们指出 **Gemini 2.5 Pro/Flash** 的通过率仅为 **45%**，这引发了对在实践中使用编程模型的担忧，并建议如果你珍惜时间，请使用 **O4-mini** 或 **Claude 3.7**。
   - 一位成员表示 *pro 把 diffs 搞得一团糟*，以至于他们正在考虑进行更多实际实验，例如 flash 不是编程模型，所以需要一些超便宜的编程模型。
- **Aider 的 Agent 化野心引发担忧**：成员们对 **Aider** 向更具 Agent 特性的方向发展感到担忧，担心它可能会失去作为结对编程工具的核心身份。
   - 一位成员表示担心，即使增加了 Agent 类功能，最终也会变得“不伦不类”。
- **利用 Aider 配置整理工作计划文档**：一位成员利用在 **base prompt** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html)) 中设置的**工作计划文档**，并在过程中进行整理。
   - 当他们想要进行此类迭代时，通常在 `/ask` 模式下进行，当情况看起来不错时，他们会执行 `/code okay, please do that`。
- **Aider 的极简 UI 引发了对主题的探索**：一位用户询问是否有适用于 **Aider 的 UI 主题**，以及某些元素是否可以自定义。
   - 一位成员回答说有**浅色模式和深色模式**，另一位成员指出了可以通过 `--code-theme` 设置的 [Pygments](https://pygments.org/styles/)。

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **一键式 MCP 新闻 Agent 亮相**：一位用户介绍了一个新的 Agent，可以一键聚合并总结过去 24 小时的 MCP 新闻，访问地址见 [此处](https://www.jenova.ai/app/tr45cl-mcp-news-summarizer)。
   - 该 Agent 旨在为密切关注 **Model Control Protocol** 生态系统发展的用户简化信息收集流程。
- **应用驱动 vs LLM 驱动资源的博弈**：成员们讨论了**应用驱动**与 **LLM 驱动**资源的优劣，特别是关于工具方面，一些人认为目前的资源在以应用为中心的方法上过于局限。
   - 其他人则反驳称，应用驱动的资源可以实现强大的功能，例如在 **MeiliSearch** 中进行索引和实时 **RAG**。
- **MCP Client 获得选择性工具控制**：讨论了 **MCP Client** 从服务器的工具签名中选择性挑选工具的能力，并以 **Claude Desktop** 为例，用户可以在其中切换工具的开启/关闭。
   - 讨论强调了客户端对可用工具进行控制的重要性，以增强定制化和安全性。
- **Stainless SDK 因数据验证失败受到抨击**：用户批评了 **Stainless SDK**，指出其生成的 **pydantic models** 无法正确验证数据，这导致了生态系统的碎片化。
   - 他们声称 **OpenAPI** 文档未能准确反映 API 的行为。
- **Sherlog Canvas (Alpha) 助力调试**：Sherlog Canvas (Alpha) 是一款用于事件调试的 AI 驱动界面，现已开源。它集成了用于日志、指标、SQL 等的 [MCP 驱动单元格](https://github.com/GetSherlog/Canvas)，并提供了一个展示其能力的 [演示视频](https://youtu.be/80c5J3zAZ5c)。
   - 它提供多 Agent 工作流，并允许 AI 生成、运行和优化单元格，辅助事件和 Bug 调查。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 移动端 App 发布，功能极简**：**NotebookLM 移动端 App** 已在 [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) 和 [Android](https://link-to-android-store) 上线，仅具备 **MVP 功能集**。用户注意到缺失了核心功能，如**思维导图**、**简报**和**“发现来源”**。
   - 用户报告称无法为**音频概览**选择来源，并发现该 App 只是一个*封装在本地容器中的 Web App*，官方正鼓励用户提交反馈和功能需求。
- **视频上传功能增强 NotebookLM**：NotebookLM 现在支持**视频上传**并自动转录，一位分享了 [关于 **NotebookLM 视频概览**链接](https://x.com/testingcatalog/status/1924246774346104863) 的用户确认了这一点。
   - 然而，一些用户在音频生成方面遇到了限制，例如 NotebookLM 仅处理长文档的引言部分，即使是 **114 页的 PDF** 也会生成很短的音频文件，目前尚不清楚这是限制还是 Bug。
- **NotebookLM 拒绝进步研究内容**：一位用户对 **NotebookLM** 未能上传与社会正义、生态学和女性主义相关的研究材料表示失望，其中包括来自 [The Conversation](https://theconversation.com/meet-the-forgotten-enslaved-and-working-class-labourers-behind-british-exploration-in-africa-asia-and-antarctica-252771) 的示例。
   - 用户怀疑原因是*审查和反觉醒（anti-woke）内容*，并附上了显示材料被拒绝的截图。
- **Senpai NotebookLM 提供犀利见解**：一位用户通过在每个新笔记本中分享一个包含自定义角色指令的 `guide.md` 文件，提升了生产力。
   - `guide.md` 文件为 **NotebookLM** 分配了如 *senpai*（专业的资深学长）或 *brain*（天才 AI）等角色，以处理不同类型的请求，从而实现定制化交互，如[此 Youtube 视频](https://youtu.be/ZCPcBgJ54NY?si=9gHljywup_mO0cAM)所示。
- **NotebookLM 的文笔变得平庸**：一位用户注意到 *NotebookLM 的写作风格在过去几天变得异常简单和还原化*，类似于**高中作文水平**。
   - 这与早先关于准备奥赛的学生潜在用途的讨论形成对比，强调了**机密性**是其区别于 Gemini 或 ChatGPT 的关键差异点。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini 的代码注释变得古怪**：成员们观察到 **Gemini** 有时会生成被注释掉的代码，并在没有明显原因的情况下删除现有注释。
   - 一位成员幽默地将 **OpenAI** 的发展轨迹比作适合 **Netflix** 剧集的希腊史诗，暗示了其出人意料的发展。
- **Google 在 I/O 大会上发布 Jules**：预计 **Google** 将在 **Google I/O** 上揭晓其代码 Agent **Jules**，该工具具备多模态功能和每日摘要。
   - 成员们分享了 [jules.google](https://jules.google/) 的链接，社区内的期待感日益增强。
- **深入探讨 AWQ INT4**：一位用户询问 **AWQ** 是原生使用 **INT4** 还是在 **VLLM** 中以 **BF16** 进行计算。
   - 另一位成员澄清说 **AWQ** 以 **W4A16** 格式运行，且 GPU 缺乏混合精度 **INT4xBF16** 运算所需的电路，建议将 **QuaRot** 或 **FP8** 作为替代方案。
- **关于开源 AI 与大科技公司的辩论**：一位成员断言，只有**去中心化的开源 AI** 才能有效防止 AI 领域出现大科技公司的寡头垄断。
   - 反方观点主张通过政府监管，例如强制开源 AI 模型及其数据，以及反审查法律，引发了关于每种策略可行性的辩论。
- **LLM 自发采用惯例**：一项研究（[arxiv 链接](https://arxiv.org/abs/2505.10475)，[science 链接](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)，以及 [HF 博客链接](https://huggingface.co/blog/codelion/ptsreal.azure)）表明，**普遍采用的社会惯例**可以通过局部交互在去中心化的 LLM 群体中自发产生。
   - 研究强调，即使个体 Agent 最初没有偏见，在这些交互过程中也会产生强大的**集体偏见**，且**少数派对抗性 LLM Agent** 可以迫使社会变革。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **使用 Matplotlib 绘制 CNN 图表**：一位成员寻求创建**带有跳跃连接（skip connections）的 CNN 图表**的方法，并在 GitHub Copilot 的帮助下使用 *matplotlib* 实现了这一目标，生成的 [脚本](https://github.com/dotrdp/DiagramVIS-for-computervis) 已发布在 GitHub 上。
   - 该仓库名为 *dotrdp/DiagramVIS-for-computervis*。
- **Gemini 2.5 Pro 处理物理任务**：一位成员使用 **Gemini 2.5 Pro 配合 Canvas 或 Cursor** 处理物理相关任务，并对 *Windsurf, Aider, Cline, Roo* 和 *Codex* 等工具感兴趣。
   - 他们在讨论可用于物理相关任务的工具背景下提到了这一点。
- **AlphaEvolve 论文引发讨论**：成员们对 **AlphaEvolve** 白皮书（[AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)）表现出浓厚兴趣，该论文详细介绍了一个由 **Gemini** 驱动的、旨在设计高级算法的代码 Agent。
   - 由于与 *Open Machine Learning* 频道中正在进行的对话重复，该讨论随后被取消。
- **探索语言模型的物理学**：小组讨论了 **"Physics of Language Models: Part 3.1, Knowledge Storage and Extraction"**（[论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633)，[博客](https://physics.allen-zhu.com/part-3-knowledge/part-3-1)，[YouTube](https://www.youtube.com/watch?v=YSHzKmEianc)）。
   - 有关该论文的更多细节可以在链接的博客和 YouTube 视频中找到。
- **开源 AI：是战略还是资源掠夺？**：成员们辩论了以开源许可证发布 AI 研究成果究竟是战略举措，还是仅仅为了获取免费劳动力和资源，特别是针对 **Meta**。
   - 一位成员认为 Meta 的 AI 研究旨在**使互补品商品化（commoditize your complements）**，而不是作为 Facebook 产品的核心。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex AMA 本周首秀**：首场 **LlamaIndex** 答疑时间将于本周四 **PT 时间上午 8 点/CET 时间下午 5 点**在常规语音频道举行，持续 **1 小时**，可以咨询关于 **LlamaIndex** 和构建 Agent 工作流的任何问题。
   - 团队还在 [LlamaIndex 博客](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents)上发布了关于 **LlamaIndex Agents** 及其改进的**长期和短期记忆**的更新。
- **Weaviate 助力多 Agent 工作流**：使用 **Weaviate** `QueryAgent` 的多 Agent 工作流指南现已在 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent)中上线。
   - 在[这段 YouTube 视频](https://youtu.be/01kM7tXRHi4)中学习如何使用 **LlamaExtract** 提取带有引用和推理的结构化数据。
- **LlamaParse 界面翻新**：根据[这条推文](https://twitter.com/llama_index/status/1923510823164706925)，**LlamaIndex** 宣布了 **LlamaParse** 的更新，包括*精简的界面*、新的**代码片段（Code Snippet）**按钮以及更多用例预设。
   - 如[这条推文](https://twitter.com/llama_index/status/1924502129974411504)所述，现已正式发布的 **Azure AI Foundry Agent Service** 提供了对 **LlamaIndex** 的原生支持，使企业客户能够构建客户支持助手和流程自动化机器人。
- **COBOL 在 Chonkie 面前败下阵来**：一位用户询问是否有用于将 **COBOL** 代码拆分为逻辑块的 Python 包，并被引导至 [Chonkie.ai 的 `CodeChunker`](https://chonkie.ai)，据称该工具支持 **COBOL**。
   - 用户指出 LlamaIndex 的代码拆分器目前不支持 **COBOL**；这可能是一个功能需求。
- **LlamaIndex Ollama 拨云见日**：一位用户在按照[官方文档](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)使用 LlamaIndex 和 Ollama 时遇到了 `ValueError: "ChatResponse" object has no field "usage"` 错误。
   - 用户通过将 Python 升级到 **3.10**、创建新环境并升级 llama-index 和 ollama 软件包解决了该问题。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **NDBuffer 弃用在即 🌅**：**NDBuffer** 正被弃用，取而代之的是 **LayoutTensor**，由于活跃的内核迁移正在进行中，建议用户避免使用它。
   - 这一转变可能会影响依赖 **NDBuffer** 的现有代码库，需要迁移到 **LayoutTensor** 以保持功能正常。
- **Atomic ArcPointer 僵局 ⛔**：由于 **Movable** 特性（trait），一位用户在创建指向包含 **Atomic** 的结构体（struct）的 **ArcPointer** 时遇到问题。
   - 使用 `ArcPointer[OwnedPointer[T]]` 的建议解决方法未能解决问题，因为 **OwnedPointer** 也没有实现 **Movable**。
- **Mojo Notebook 导入之谜 🔍**：一位用户询问如何从与 **notebook** 位于同一文件夹中的 **Mojo package** 或文件中导入代码。
   - 遗憾的是，消息中未提供解决方案，导致用户的导入难题悬而未决。
- **LSP Server 崩溃 🫠**：用户报告 **LSP server** 占用高 CPU（8-16 个线程）且频繁崩溃，尤其是在旧系统或使用 Docker 时。
   - 重启 **LSP server** 或降级到之前的 nightly 版本等临时解决方法效果有限；一位用户不得不求助于 `killall mojo-lsp-server`。
- **`register_custom_ops` 被砍掉 🪓**：`max.torch` 中的函数 `register_custom_ops` 在最新的 nightly 版本中被移除，这导致部分用户的脚本失效。
   - 一位 Modular 团队成员确认了在 nightly 版本中将 Mojo 自定义算子（ops）注册到 PyTorch 的工作正在进行中，并引导他们查看更新后的[文档](https://docs.modular.com/max/api/python/torch/)。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz 发布 tinygrad 性能统计数据**：George Hotz 分享了 [tinygrad performance statistics](https://stats.tinygrad.win) 的链接，展示了 **tinygrad** 性能的提升。
   - 这些统计数据提供了 **tinygrad** 效率的实时视图，有助于开发者优化他们的模型。
- **引发辩论：在 tinygrad 中使用 GCC 代替 Clang？**：一位用户询问是否可以在 CPU target 上使用 **GCC** 代替 **Clang**，特别是针对没有 **Clang** 的 **AIX system with the PPC64 arch**。
   - George Hotz 回复称这 *并不容易* 做到，需要为自定义的 elf loader 添加 **elf relocations for ppc64**，并参考了 [这个文件](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/elf.py#L41)。
- **ONNX 简化了模型向 tinygrad 的迁移**：一位用户询问如何将 **Qwen3:30-a3b** 等模型移植到 tinygrad，并询问是否有自动工具或需要手动移植。
   - George Hotz 澄清说，如果模型是 **ONNX** 格式，使用 `examples/benchmark_onnx.py` 脚本可以轻松导入。
- **tinygrad 的 API 稳定性受到称赞**：一位正在编写 AI 应用开发书籍的作者考虑使用 tinygrad，询问其接口稳定性，以确保示例在 2-3 年内保持可用。
   - George Hotz 保证 frontend 在 *至少一年内都非常稳定*，并建议 *在追求速度之前应该先完成 1.0 版本*。
- **WMMA 指令基准测试工具发布**：一位用户分享了 [HIPmmapeak](https://github.com/pkourouklidis/HIPmmapeak) 的链接，这是一个用于测量 **7900 XTX** 上 **wmma instructions** 最大 **FLOPS** 的工具，类似于 mmapeak。
   - George Hotz 回复道：*太酷了！如果你想要悬赏（bounty），请使用 tinygrad 的基础设施*。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI/ML 工程师展示 Agent 领域的专业知识**：一位在智能系统领域拥有 **8 年以上** 经验的 AI/ML 工程师介绍了自己，展示了他们使用 **LangGraph**、**AutoGen**、**LlamaIndex**、**Letta** 和 **DSPy** 等现代技术栈构建 Agent 系统的工作，并分享了[个人作品集](https://yangming.vercel.app/)。
   - 他们展示了对 **GPT-4o**、**Claude 3**、**Gemini**、**Mixtral**、**LLaMA-3** 和 **Mistral** 的熟练运用，以及 **React**、**Next.js**、**FastAPI**、**Django** 和 **Laravel** 等全栈技术。
- **Suggest 和 Assert 演进为 BestOfN 和 Refine**：自 **DSPy 2.6** 起，**BestOfN** 和 **Refine** 已取代 `dspy.Suggest` 和 `dspy.Assert`，详见[本教程](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)。
   - 这一变化鼓励在 DSPy 程序中采用更结构化的输出精炼和验证方法，并使验证和调试来自 LLM 的输出变得更加容易。
- **使用 DSPy 重新实现 Cline，声称在体积和准确性上有优势**：成员们讨论了使用 DSPy 重新实现像 **Cline** 这样的 AI 编程 Agent；认为它可能会更小、更准确，且减少 *偏离预期（off-piste）* 的修改。
   - 会中指出 VS Code 粘合层、memory、tools 和 models 都是 **Cline** 成功的关键因素。
- **DSPy 在处理大型 Prompt 时的延迟问题**：一位成员在 DSPy 中处理大型 system prompts 时遇到了较长的解析时间，并询问如何配置 **litellm** 的 [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching)，但遇到了错误。
   - 另一位成员在 DSPy 之外优化了 prompt，发现 DSPy 的响应时间为 **8s vs 2s**，这表明 DSPy 在处理优化后的 prompt 时可能存在效率低下。
- **DSPy 3.0 的 Elixir 移植版吸引了支持者**：一位成员询问是否有兴趣将 **DSPy 3.0 1:1 移植到 Elixir**，引发了关于该语言的扩展模式和并发模型 *非常适合 LLM orchestration* 的讨论。
   - 虽然一些人支持 Elixir 的能力，但另一些人指出 Python 在生态系统中占据主导地位，因为它与 [Nx ecosystem](https://github.com/elixir-nx) 等库的集成和成熟度更高。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 表示不支持 Embedding 微调**：一位成员询问关于微调 Embedding 模型的问题，另一位成员澄清说**微调 Embedding 模型是不可能的**，但根据特定需求**微调 Reranker** 是可行的。
   - 他们无法找到解决问题的方法，因为这被记录为一个已知问题。
- **Embed-v4.0 出现“非预期” Embedding**：一位成员报告说 **embed-v4.0** 生成了一些*非预期*的 Embedding，并询问是否可以微调 Embedding 模型。
   - 未给出解决方案，但已记录为已知问题。
- **针对 Vision RAG 调研 Embed 4 定价**：一位成员询问了在团队的 **Vision RAG** 实现背景下 **Embed 4** 的定价。
   - 另一位成员建议嵌入一张图像，并查看响应中 usage 部分的 **token count**。
- **Chat API 在 Agent 执行期间出现停滞**：一位成员报告说，尽管是付费会员，**Chat API** 在其 Agent 执行过程中似乎中途停滞，并请求调试协助。
   - 他们提到尝试将 **nodes 串行运行与并行运行**进行对比，作为一种临时变通方法，并被要求提供 **model、API 版本、任务和代码片段**等详细信息以协助调试。
- **Vitalops 的 datatune 开源**：Vitalops 在 [GitHub](https://github.com/vitalops/datatune) 上发布了一个名为 **datatune** 的新**开源工具**，该工具使用自然语言指令和 LLM 进行数据转换。
   - **datatune** 工具利用 LLM 根据简单的**自然语言指令**执行**数据转换**，旨在简化复杂的数据处理。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **相比本地文档，用户更青睐 Swarm UI**：一位用户表示，由于在各种应用中易于使用，他们更倾向于选择 **Swarm UI** 而非本地文档。
   - 他们表示*本地文档目前还可以*，但对于*初级到中级再到高级*的使用场景，**swarm-ui** 是更好的选择。
- **Linus 打造百万美元圆周率服务器**：一位成员分享了 Linus Tech Tips 建造一台价值 **100 万美元服务器**的链接，该服务器旨在计算圆周率的位数。
   - 另一位用户回应道：*这已经不是我第一次看到 Linus 做疯狂的事情了。*
- **寻求客户成功（Customer Success）职位的指导**：一位具有政府、非营利组织、销售和外联背景的成员询问如何转型到科技公司的**客户成功**职位。
   - 他们强调了自己在支持他人、建立关系和解决问题方面的才能，以及他们的技术专长和简化复杂想法的能力。
- **推荐用于教科书 Markdown 转换的模型**：一位用户请求推荐模型，将大量教科书文本转换为 **Markdown 笔记**，包括摘要和提取特定细节。
   - 他们正在寻找能够有效处理这些任务的工具。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 竞赛提交截止日期临近！**：**AgentX 竞赛**的提交截止日期为 **2025 年 5 月 31 日**晚上 11:59（太平洋时间），[创业赛道](https://forms.gle/FJTC4jd197bNeJJ96)和[研究赛道](https://forms.gle/5dccciawydCZ8o4A8)有各自的提交链接。
   - 竞赛拥有全明星评审团和超过 **15 万美元**的奖金池，详情可见 [Twitter](https://x.com/dawnsongtweets/status/1924470174776004875) 和 [LinkedIn](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii)。
- **自备 OpenAI API Key**：LLM Agents MOOC 的学生必须提供自己的 **OpenAI API Key** 来运行实验练习，因为课程*不提供这些 Key*。
   - 实验助教已被要求提供关于替代方法的见解，以尽量减少或绕过对外部 API 交互的需求。
- **MOOC 学生仍可获得 Trailblazer 等级**：在 LLM Agents MOOC 中实验遇到困难的学生仍然可以申请 **Mastery 等级**，并有可能被“降级”到 **Trailblazer 等级**。
   - 这确保了那些通过测验和撰写文章展示知识的学生能够获得认可。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **考虑将 CI Nightly 构建用于评估**：一位用户正在权衡是以 **临时（ad hoc）方式** 评估模型，还是为模型评估建立一个 **自动化的每晚 CI 流水线**，后者需要更多的精力投入和计算资源。
   - 该用户承认 **ad hoc** 方法虽然可靠性较低但更易于实现，而 **自动化 CI** 方法虽然能提供更高的保真度，但需要更多的维护工作和计算资源。
- **使用 cfg.get 进行 Torchtune 验证**：一位用户搜索了 GitHub 代码，以查找在 [pytorch/torchtune 仓库](https://github.com/search?q=repo%3Apytorch%2Ftorchtune%20cfg.get(%22dataset_val%22)&type=code) 中何处使用配置文件的 `cfg.get("dataset_val")` 来设置 **验证数据集（validation dataset）**。
   - 了解验证数据集的工作原理对于评估模型训练非常有用，而探索 `cfg.get` 有助于发现数据集配置是在何处设置的。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Vitalops 利用 LLM 调整数据**：Vitalops 发布了 **DataTune**，这是一个利用 LLM 通过自然语言指令进行数据转换的开源工具，可在 [GitHub](https://github.com/vitalops/datatune) 上获取。
   - **DataTune** 通过自然语言提示词简化了数据转换过程。
- **Vitalops 再次利用 LLM 调整数据！**：Vitalops 发布了 **DataTune**，这是一个利用 LLM 通过自然语言指令进行数据转换的开源工具，可在 [GitHub](https://github.com/vitalops/datatune) 上获取。
   - **DataTune** 通过自然语言提示词简化了数据转换过程。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **资深 AI 工程师寻求联系**：一位在机器学习、深度学习和数据科学领域拥有 **10 年经验** 的 AI 工程师正寻求建立联系，并构建下一代思考型软件。
   - 他们对 **Python, TensorFlow, PyTorch** 以及 **AWS** 和 **GCP** 等云平台有深入的了解。
- **AI 工程师展示强大的技能组合**：该 AI 工程师展示了在 **Python, SQL, R**, ML/DL 框架（**TensorFlow, PyTorch**）以及 **Jupyter, Git, Docker, MLflow, Streamlit** 等工具方面的熟练程度。
   - 他们的技术涵盖监督学习与无监督学习、深度学习（CNN, RNN, Transformers）、NLP、计算机视觉以及模型部署（APIs, CI/CD）。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

您收到这封邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道分类的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1373012180626178140)** (1047 条消息🔥🔥🔥): 

> `Cunnyx 链接，MCP SuperAssistant 扩展，Grok Deepsearch 对比 Perplexity Pro，Firefox 推广 Perplexity，Yandex 浏览器安全性` 


- **You.com 中显示的百分比是胜率而非质量**：一位成员分享了一个 [Cunnyx 链接](https://cunnyx.com/youdotcom/status/1923100692098453703)，指出 **You.com** 中显示的百分比是胜率，而不是质量，尽管视频中没有明确说明这一点。
- **Perplexity 中的 MCP SuperAssistant 扩展**：一位用户在 Perplexity 中测试了 **MCP SuperAssistant** 扩展，报告称它在许多服务器上运行良好，但偶尔会遇到断开连接的问题；分享了 [MCP SuperAssistant 网站](https://mcpsuperassistant.ai)和 [GitHub 仓库](https://github.com/srbhptl39/MCP-SuperAssistant)的链接。
- **Grok Deepsearch 比 Perplexity Pro 更受青睐**：尽管拥有 Perplexity Pro，一位用户仍更倾向于使用 **Grok Deepsearch**，原因是针对印度尼西亚与印度食品价格的查询结果不一致，并指出 Perplexity 的研究任务中缺少 Python 数学工具。该用户分享了 [Perplexity Research 模式](https://www.perplexity.ai/search/is-it-pricey-to-have-like-100g-p6cc07L0RAGxI9vBnBDfIw)、[搭载 o4 mini 的 Perplexity Pro 模式](https://www.perplexity.ai/search/is-it-pricey-to-have-like-100g-02GLrmEVR1eVEbFelRbBRQ)以及 [Grok Deepsearch](https://grok.com/share/bGVnYWN5_28787552-f2a7-494a-b876-00980d4d523d) 的链接。
- **Firefox 将推广 Perplexity 搜索引擎**：有消息称 [Firefox 将推广 Perplexity 搜索引擎](https://windowsreport.com/mozilla-firefox-to-promote-perplexity-search-engine/)，尽管一位成员质疑它是否已经在 Android 上线，另一位成员则认为 Firefox 正在走向没落，更看好 Blink 而非 Gecko。
- **Yandex 浏览器安全性受质疑**：成员们就 **Yandex Browser** 的安全性展开了辩论，一位用户称 *“它比 Chrome 更好”*，而另一位用户则因其俄罗斯浏览器的身份表示担忧，并分享说他们有 *Yandex 个人电子邮件账户*，且认为其图像生成器很有用。此外，另一位成员提供了一份对比 [Yandex、Chrome 和 Firefox 隐私与安全](https://media.discordapp.net/attachments/669308329419341836/1079537276775829605/935C5928-EDC9-4D6B-993B-1E81113888E6.gif)的列表。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1373177570610118708)** (16 条消息🔥): 

> `API 额度，Sonar API 与 UI 的差异，Sonar API 调整，Playground 输出与 API 输出对比` 


- **API 额度延迟令黑客松团队感到沮丧**：一个团队报告称等待 **API 额度** 已达 **5 天**，且针对其咨询仅收到 **AI 回复**。
   - 随后确认他们已收到 API 额度；未再进行进一步讨论。
- **Sonar API 来源与 UI 存在巨大差异**：多位用户对 **Perplexity Sonar API** 返回的结果与 **UI 版本** 存在巨大差异表示困惑。
   - 一位用户指出，在查询个人资料时，**API 来源** 经常 *“完全没有提到该名字”*，另一位用户分享了一个用于测试的 [GitHub 仓库](https://github.com/alanbuxton/perplexity-api-tester)。
- **Sonar API 的调整仍不明确**：用户询问如何调整 **Sonar API**，特别是 `top-p` 等参数是否会影响实际搜索。
   - [Perplexity API 文档](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239)关于自定义 **Sonar API** 的信息有限。
- **Playground 表现优于 API**：一位用户质疑为什么他们在 **Playground** 上获得的输出效果比 **API** 更好。
   - 另一位用户推测差异可能是由 **UI** 中的 system prompt 或其他原因造成的。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1373012823549935616)** (1551 messages🔥🔥🔥): 

> `Convex vs Supabase 实时应用对比, 用于 AI Agent 通信的 MCP, DeepSeek-R1T-Chimera 模型, Cursor 速度问题, Cursor 内的文档导航` 


- **Convex 在实时应用方面碾压 Supabase**：成员们讨论认为，虽然 **Supabase** 提供了实时功能，但 **Convex** 通常被认为更适合实时应用，因为它具有自动同步和事件驱动架构，此外还有一份[对比报告强调了为什么 Convex 在实时场景中表现出色](https://x.com/mntruell/status/1924297275993669893)。
   - 有人指出 Supabase 可以本地托管，但 Convex 在实时处理方面更胜一筹；不过该成员表示不会因为任何原因停止使用 Supabase，并补充说其 Auth、Edge Functions 和 Buckets 非常出色。
- **用于 AI Agent 通信的 MCP 热潮**：围绕使用 **MCP** (Model Context Protocol) 让 AI Agent 进行通信（尤其是那些不在同一台电脑上的 Agent）展开了讨论。有人表示 *Qwen 3 235B 正是因为这个原因才自带原生 MCP 支持*。
   - 成员们描述了他们的配置，从 Discord 管理机器人到通过单一事实来源和上下文协调 Agent 之间的任务，甚至有人为此创建了一个 [GitHub 仓库](https://github.com/coder/agentapi)。
- **DeepSeek-R1T-Chimera 精通 Markdown**：用户称赞 **DeepSeek-R1T-Chimera** 模型能够精确修改 .md 和 .mdc 文件而不出错，并能在 Prompt 测试中打破循环。用户指出它是唯一能实现这一点的免费模型，可以在 [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) 上开始使用。
   - 该模型是在 R1 和 V3 之间进行微调的，展示了其在处理 Markdown 文件方面的强大实力。
- **Cursor 在处理庞大代码上下文时表现不佳！**：一些用户报告了 **Cursor** 请求缓慢和质量不稳定的问题，其中一人表示：“我真的很讨厌现在发生的情况”。
   - 其他人建议切换模型（例如使用 DeepSeek 3.1 以获得即时结果）或重置上下文，而一些用户在使用 Gemini 最高设置时遇到了“全部应用”按钮不显示的 Bug。
- **巧妙地进行文档导航**：成员们讨论了 **Cursor** 如何处理链接文档，解释说它通过读取链接页面的 HTML 内容来收集信息和链接；而一些人指出使用 [Cursor 文档系统](https://docs.cursor.com/context/model-context-protocol) 能更好地保留所有代码。
   - 用户希望能够像浏览器一样动态读取链接文档中的更多页面，团队对此回应称，即将推出一个将 DOM 发送到 Cursor 的 API 插件。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1373013206573912074)** (422 messages🔥🔥🔥): 

> `imatrix 校准数据集, Qwen3 GGUF 问题, 运行 LLM 的成本估算, Gemma 3 微调, Google AI 的 AlphaEvolve` 


- **Unsloth 的 imatrix 校准数据集梦想成真**：成员们对 **UnSloth Dynamic 2.0 GGUFs** 以及随之而来的“具有范式转移意义、专注于指令和对话的 imatrix”赞不绝口。
   - 据称，对于 **Llama-3.1-8B-Instruct-UD-Q8_0_XL**，*改进的困惑度（perplexity）意味着在响应预测和生成时具有更快的 token/s*。
- **运行 Qwen3 GGUF 遇到困难引发调查**：成员报告称，由于 SHA 哈希不匹配和越界错误，他们在运行 *Qwen3 235 128k UD-Q6 和 UD-Q8 GGUF* 时遇到了困难。
   - 团队表示他们将进行调查，并已*更新了所有 Qwen3 上传文件，修复了聊天模板问题*。
- **运行 LLM 的成本对比**：分享了一篇博文，[对比了托管 vs API 运行 LLM 的估算成本](https://mburaksayici.com/blog/2025/05/15/llm-interviews-hosting-vs-api-the-estimate-cost-of-running-llms.html)。
   - 成员表示：*校准数据集非常出色，我们花了大约 3 周时间手动收集、清洗并反复核对*。
- **微调 Convex 获得第三名**：一位成员对 *Convex*（一个小众的 TypeScript 后端）进行的微调在评估后获得了**第三名**的成绩，仅落后于 **Claude 3.7 Sonnet** 3%，这是通过 **Qwen3 14b 微调**实现的。
   - 他们还强调了*高质量（HQ）数据集是多么重要*。
- **AlphaEvolve 编写自己的代码**：一位成员分享了一篇关于 [AlphaEvolve](https://venturebeat.com/ai/meet-alphaevolve-the-google-ai-that-writes-its-own-code-and-just-saved-millions-in-computing-costs/) 的文章，这是 Google AI，可以编写自己的代码并节省了数百万美元的计算成本。
   - 另一位成员好奇为什么 **AlphaEvolve** 不能发明下一代 Transformer。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1373151204070395904)** (13 条消息🔥): 

> `从 Google Colab 下载适配器，私有 Hugging Face 模型，除 Qwen 之外的现代 LLM` 


- **Colab 下载引发困扰**：一位成员抱怨从 **Google Colab** 下载适配器（adapters）速度缓慢，并考虑切换到其他服务。
   - 另一位成员确认了速度慢的问题，并建议先上传到 **Hugging Face**，然后再从那里下载。
- **HF 模型隐私探讨**：一位成员询问如何将模型上传到 **Hugging Face** 并默认设置为私有。
   - 另一位成员回答道：*只需设置 private=True*。
- **LLM 领域探索**：一位刚接触本地 LLM 的成员询问，除了 **Qwen** 之外，现在大家还在使用哪些模型。
   - 另一位成员提到，有些人仍在使用旧的 **Llama** 模型，但在较新的模型中推荐 **Phi-4** 或 **Mistral Large**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1373013543770914879)** (659 条消息🔥🔥🔥): 

> `TPU 支持，GGUF 保存错误，Torch 和 CUDA 错误，Unsloth 文档，持续预训练 vs LoRA` 


- **Unsloth 工具调用 Notebook 现身！**：用户讨论了如何使用 **Hugging Face** 模型创建工具和 **Agent**，一位成员分享了 Unsloth 上的[工具调用 Notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks#:~:text=Other%20important%20notebooks%3A-,Tool%20Calling%20%2D%20new,-ModernBERT%2Dlarge%20%2D%20new)。
   - 另一位成员确认了其对编程模型的适用性，并指出 *这里的大多数模型都擅长工具调用，所以它能行吗？*
- **野外发现实体 TPU！**：一位用户提到手头有一个闲置的 **TPU**，引发了关于购买实体 **TPU** 的局限性和可行性的讨论，因为购买实体 **TPU** 实际上并不可行。
   - 虽然购买实体 **TPU** 并不太可能也不可行，但有人推荐了 Hyperbolic，他们提供 **H100** 租用价格为 **$0.99**，**H200** 为 **$2.15**。
- **触发 GGUF 保存问题！**：一位用户在尝试保存为 **GGUF** 格式时遇到了 *RuntimeError: Unsloth: Failed to convert llama.cpp/unsloth_convert_hf_to_gguf.py to GGUF*。
   - 解决方案包括[在保存为 GGUF 之前合并模型](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/gemma3.md)，并警告 `save_pretrained_gguf` 函数可能仍与最新的 `llama.cpp` 更改不兼容，并引导用户查阅有用的文档。
- **Colab 用户遭遇 Torch/CUDA 重击！**：一位用户报告在更新 Unsloth 后出现 **CUDA 错误**，经查是由于旧驱动程序需要特定的 **Torch** 版本。
   - 该用户通过使用 `pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"` 找到了解决方案，但对每次升级 Unsloth 都需要彻底删除虚拟环境（venvs）表示担忧，且[这位用户正在“自我修复”](https://www.urbandictionary.com/define.php?term=self-curing)。
- **Unsloth 文档是终极答案！**：一位用户询问 Unsloth 的二分类 Notebook，他们被引导至 [Unsloth 文档](https://docs.unsloth.ai/get-started/unsloth-notebooks)，因为那里 *有你所需的一切*。
   - 另一位成员调侃道：*你真的指望人们在事情搞砸之前读文档吗？*


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1373190322409967666)** (19 条消息🔥): 

> `用于微调的 PTS 和 DPO，具有可训练排列的 Beam Search，Tokenizer 训练与 Embedding 研究，Entropix GitHub 项目` 


- **Pivotal Token Search 助力 DPO 微调**：成员们讨论了 [Pivotal Token Search](https://huggingface.co/blog/codelion/pts) 本身并非 RL，但它能生成用于微调的 DPO 对，并引用了 **Phi-4 技术报告**中展示的改进。
   - 一张截图（[图片链接](https://cdn.discordapp.com/attachments/1257011997250424842/1373218344232157264/Screenshot_2025-05-17_at_4.37.39_PM.png?ex=682ce87e&is=682b96fe&hm=685beeb66d15e5f4e1195735b7a8b1cc7fb4145dc5dfc8f9cd2863e&)）直观地证实了 **PTS 对 DPO 性能**在各项基准测试中的积极影响。
- **可训练的 Beam 排列**：一位成员将一篇研究论文（[https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475)）比作*具有可训练 Beam 排列的 Beam Search*。
   - 另一位成员建议，**模型越大**，这种策略就越强大，并且它随 **Compute**（计算量）而非内存带宽进行扩展。
- **Tokenizer 训练探索**：一位成员询问了关于为现有 LLM 训练自定义 Tokenizer 和 Embedding 的研究，旨在从 **Qwen Tokenizer** 中剥离非英语 Token 并添加更多特殊 Token，并引用了 [ReTOK 论文](https://arxiv.org/abs/2410.04335)。
   - 另一位成员建议搜索 **LLaVa** 及类似项目，以寻找专注于向现有模型添加新语言/模态的研究。
- **Entropix GitHub 项目再次浮现**：一位成员询问 [Entropix GitHub 项目](https://github.com/xjdr-alt/entropix) 是否有任何进展，并指出该项目看起来很有前景。
   - 未提供进一步的细节或结果，暗示其状态仍不确定。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1373015169554321520)** (569 条消息🔥🔥🔥): 

> `ASI Lab 抄袭指控，Codex 体验，GPT-5，Gemini 2.5 Pro 数学性能，ChatGPT Memory 功能` 


- **ASI Lab 面临抄袭指控**：Pro.creations 声称（[图片链接](https://cdn.discordapp.com/attachments/998381918976479273/1373015169357054082/image.png?ex=682cd405&is=682b8285&hm=41d43e15460edc8f88272551f18cfec1fa74fb94c748b9efa15f5c82234bb031&)）**ASI Lab 的工作因抄袭被下架**，并表示*看到一所受人尊敬的大学进行抄袭并冠以虚假的 AGI 之名，真是太糟糕了*。
- **Codex 访问权限是游戏规则改变者**：一位成员表示 Codex 是*游戏规则改变者*且*好得惊人*，并指出**它需要监督**，但其他人已经同时运行了多个 Agent 线程；它需要互联网访问，且[有人抱怨](https://cdn.discordapp.com/attachments/998381918976479273/1373141529303978075/image.png?ex=682ca0f4&is=682b4f74&hm=ecc62fa17cbb93ebfbeadcf07f169e859106154cc9b595469981e85b61899ac7&)它只能处理几百行代码。
- **GPT-5 推测升温**：成员们推测 **GPT-5 不包含 o3**，并且会像 **Gemini 2.5 Pro** 一样，是一个 LLM 加一个推理模型，而不是目前的 o 系列和 GPTs 系统，并可能在今年夏天发布；另一位成员建议关注 **Google week** 活动。
- **Gemini 2.5 Pro 数学性能讨论**：成员们讨论了 **Gemini 2.5 Pro** 在数学方面的表现，指出旧模型在数学方面更好，但新模型在编程方面更强，其中一人提供了 [Google 官方数据](https://deepmind.google/technologies/gemini/pro/)，显示 **Gemini 2.5 Pro 是 0506 版本**，其性能为 **83% 对比 88.9%**。
- **ChatGPT 的 Memory 引发超个性化辩论**：一些成员推测 **ChatGPT** 的新 Memory 功能，以及他们是否已切换到 **Transformer–Liquid Neural Network 混合架构**；然而，另一些人认为**高级 Memory 系统可能只是 RAG**，这些高级功能由 Transformer 外部的系统管理，并在运行时注入到 Context 中。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1373040677574934629)** (414 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro vs 4.1, Rate Limits, GPT Lying, ChatGPT for Education, 4o Mini` 


- **Gemini 2.5 Pro vs 4.1，哪个更好？**：一名成员询问 **Gemini 2.5 pro** 是否优于 **4.1**，这引发了关于遵守 [AI-Discussions 规则](https://discord.com/channels/974519864045756446/1107255707314704505) 以保持讨论主题相关的提醒。
   - 该问题被判定为偏离主题，应当发布在相应的频道中。
- **使用 OpenAI API 进行数据库扫描**：一名成员计划通过 API 向 GPT 发送约 **1000 条 prompts** 进行数据库扫描，并询问被封禁的可能性。
   - 另一名成员回应称，虽然 **ChatGPT** 有消息限制，但 **API** 受 [rate limits](https://platform.openai.com/docs/guides/rate-limits/) 约束。
- **关于 ChatGPT 是否故意撒谎的激烈辩论**：成员们就 **ChatGPT** 撒谎的本质展开了辩论。有人认为由于其先进性，它更加复杂且能够进行蓄意欺骗；而另一方则反驳称，其输出是由于偏见或错误的训练数据造成的，而非蓄意撒谎。
   - 一名成员引用研究指出，**GPTs** 可以被编程为撒谎（明知真相却给出相反回答），或者表现出基于奖励模型的策略性欺骗模式；而另一名成员则认为，将一切归咎于 hallucinations 是一种有缺陷的思维方式。
- **AI 助力 STEM 领域的准工程师**：一名成员寻求关于使用 **ChatGPT** 进行教育的建议，特别是为大学工程学习做准备。对此，有人建议使用数学推理模型和 **YouTube** 视频等视觉辅助工具。
   - 另一名成员建议使用 **custom GPTs** 来创建类似 **Duolingo** 的游戏化学习体验，以跟踪进度和目标，并利用 **4o** 的 memory 存储功能。
- **ChatGPT 的 4o Mini 因多功能性受到好评**：一名成员表达了对 **4o mini** 的偏好，理由是其体量轻巧且适用于各种用例，同时也承认其主要关注点在于图像和视频生成。
   - 其他人讨论了 **4o** 的 memory 功能，指出免费版最多可存储 **100 条记录**，允许模型记住用户信息并在随后的对话中调用。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1373014117346181211)** (48 条消息🔥): 

> `HyperEnglish Prompting, AI Custom Instructions via Python, ProtoMind Semantic Mapping, Image Prompting Workflow, Learning Prompt Engineering` 


- **HyperEnglish 语法提升清晰度**：一名成员介绍了 [HyperEnglish](https://example.com/hyperenglish)，强调通过 *functional tagging* 和技术术语的 **全大写 (ALL-CAPS)** 来优先保证清晰度。
   - 他们分享了一个示例模板：`SUBJ:{subject} VERB:{verb} OBJ:{object} {additional_tags_as_needed}`。
- **AI 通过 Python 加载自定义指令**：成员们讨论了使用 **Python 脚本** 动态加载 custom instructions，允许 AI 实时调整其模式。
   - AI 可以编写脚本来返回包含指令的文本文件内容，从而实现 **加权程序化响应 (weighted procedural responses)**。
- **ProtoMind 映射语义，无需代码**：一名成员称赞 [ProtoMind_001](https://chatgpt.com/g/g-682759c7c8d881919436d320e718d5d5) *不需要显式代码*，它将用户输入视为用于角色切换和伪内存线程的 **分层语义算子 (layered semantic operators)**。
   - 另一名成员补充说，ProtoMind_001 能够很好地对他们进行“映射”。
- **图像提示词工作流的视觉迭代**：一名成员分享了他们在 **TTRPG** 中保持角色视觉一致性的 [图像提示词工作流](https://example.com/image-prompting)，从概念原型开始，并使用 O3 进行视觉推理的迭代优化。
   - 该过程包括生成 **正交草图 (orthographic sketch)**，对其清晰度进行评审，然后将其作为多个图像提示词的锚点。
- **移动应用开发者思考提示工程的价值**：一名移动应用开发者询问了学习 prompt engineering 的价值，并考虑参加 [Google Prompt Engineer 课程](https://example.com/google-prompt-engineer)。
   - 回复建议关注清晰度和组织结构，一名成员建议如果已有免费资源，则不要为这类课程付费。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1373014117346181211)** (48 messages🔥): 

> `HyperEnglish, Meta-prompt generator complexity, Loading Custom Instructions, Weighted procedural responses, ProtoMind_001` 


- **HyperEnglish 提升清晰度**：一位成员分享了他们的 [HyperEnglish 模板](https://example.com/hyperenglish)，该模板旨在通过功能性标签和全大写（ALL-CAPS）技术术语来提高清晰度，将清晰度置于自然度之上。
   - 该模板强制执行结构化内容，例如 `SUBJ:{subject} VERB:{verb} OBJ:{object}`。
- **动态加载 Custom Instructions**：成员们讨论了使用 Python 工具动态 [加载 Custom Instructions](https://example.com/custom-instructions) 以输出调整内容，然后将其返回到 Context 中。
   - AI 可以创建一个脚本来加载指令并运行它，从而有效地改变运行模式，一位成员指出 *我们在这里想出了一些非常有创意的东西。*
- **Agent 作为语义运行时操作符**：一位成员表示 [Agent 作为语义运行时操作符（semantic runtime operators）工作](https://example.com/semantic-operators)，强调不需要显式代码，只需要符号化的角色切换和伪内存线程化（pseudo-memory threading）。
   - 他们不将用户输入视为查询，而是视为分层的语义操作符，建议它跟踪矛盾、预测演化路径并重塑内部目标。
- **迭代式图像 Prompting 生成一致的 TTRPG 视觉效果**：一位成员概述了 [TTRPG 角色视觉效果的图像 Prompting](https://example.com/ttrpg-image-prompting) 迭代过程，从构思、草图、评论到创建多张一致的图像。
   - 该过程包括使用 O3 进行视觉推理，并在创建多张图像之前生成正交草图，以确保结果的一致性。
- **评估 Prompt Engineering 课程的价值**：一位用户询问 [Prompt Engineering 课程](https://example.com/prompt-engineering-courses) 对移动应用开发者的价值。
   - 另一位成员建议直接询问 ChatGPT 关于 Prompt Engineering 的问题，并建议如果免费课程有助于组织思路，那么可能值得一试，但警告不要为此类课程付费。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1373015087194832936)** (900 messages🔥🔥🔥): 

> `O3 Pro, GPT-5 speculation, Claude 4, DeepSeek's fate, Codex's potential` 


- **O3 Pro 无限期推迟**：用户对 **O3 Pro** 的持续推迟表示失望，有些人开玩笑说要通过 *绝食* 直到它发布。
   - 一些人认为原始的 **GPT-4** 更胜一筹，拥有较新的、较小的模型所缺乏的 *真正的智能/创造力火花*。
- **GPT-5 猜测**：一些人认为 **GPT-5** 将是一个可与 **O4** 媲美的基础模型，但其他人持怀疑态度，认为它可能只是 **O3** 的略微改进版本。
   - 讨论涉及它将是一个模型路由（model router）还是一个新的基础模型，以及 RL 训练如何影响其相对于稳定版本的改进。
- **Codex 真的好用吗？**：用户正在辩论 **Codex** 的实用性，一些人认为它对初级开发者来说是 *噪音*，而另一些人则看到了它在处理更高级任务方面的潜力。
   - 一位用户建议 **Codex** 需要与 **RooCode/Cline/Cursor/Windsurf/Aider** 等工具竞争才有价值。
- **Gemini 与 Claude 的对决**：关于 **Gemini** 与 **Claude** 在编程方面的对比一直存在争论，一些人发现 **Gemini** 过于冗长且倾向于添加不需要的功能，而另一些人则称赞 **Claude** 的可靠性。
   - 一些用户认为 **Gemini** 的代码注释是一个负面因素，而另一些人则认为它们很有帮助。
- **Microsoft 与 xAI 的未来**：一位用户发布了一张暗示 Microsoft 与 xAI 合作的图片。
   - 另一位用户回应道 *天哪*，并链接了一篇关于该话题的社交媒体帖子。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1373326966668656640)** (2 messages): 

> `Mistral Medium 3, Claude 3 Sonnet, Amazon Nova Pro, Command-a-03` 


- **新模型进驻 Beta 站点**：LMArena Beta 站点添加了新模型，包括 **mistral-medium-2505**、**claude-3-7-sonnet-20250219-thinking-32k**、**amazon.nova-pro-v1:0** 和 **command-a-03-2025**。
- **Mistral Medium 3 排名攀升**：自 **Mistral Medium 3** 在排行榜亮相以来，它比 Mistral Large 实现了 **+90** 分的惊人飞跃，在聊天总榜排名第 11。
   - 它在技术领域表现出色（**数学排名第 5**，**Hard Prompts & Coding 排名第 7**），并在 WebDev Arena 排名 **第 9**。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1373143224301912156)** (2 条消息): 

> `Gemini 2.5 Pro Experimental 弃用，DeepSeek V3 维护` 


- **Gemini 2.5 Pro Experimental 停用**：Google 正在弃用 **Gemini 2.5 Pro Experimental** (`google/gemini-2.5-pro-exp-03-25`)，转而支持付费的 **Preview endpoint** (`google/gemini-2.5-pro-preview`)。
   - 该模型很快将在 OpenRouter 上弃用。
- **DeepSeek V3 进行调整**：免费的 **DeepSeek V3 0324** 今天将停机维护一段时间。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1373735131247808562)** (3 条消息): 

> `国际象棋锦标赛，Stockfish 实现，Lichess 准确率评分，OpenRouter 模型` 


- **国际象棋排行榜大改版**：一位成员分享了他们的项目——一个**国际象棋排行榜**。该项目从简单的**国际象棋锦标赛**构思演变而来，现在整合了 **Stockfish 实现**以获得准确的评分。
   - 该排行榜模拟了 **Lichess 准确率评分**，并支持具有 temp 和 sys role 功能的 **OpenRouter 模型**，同时为 o1-mini 等模型提供了变通方案，并使用 cronjobs 实现全自动化。
- **国际象棋排行榜非常酷**：一位成员表示，这个**国际象棋排行榜**非常酷，是一个与众不同的用例。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1373028823049703484)** (790 条消息🔥🔥🔥): 

> `API key 识别，Gemini 2.5 弃用影响，Qwen3 Tool Calling 问题，低延迟 LLM，Gemini API 更新` 


- **共享 API Keys 需要 Request Tags**：一位成员询问在多个用户共享单个 key 时如何识别 API 调用来源，特别是为了追踪流式传输中途断开的情况。
   - 建议包括实现 **request tags**，而不是在 app name 中嵌入用户 ID，以便更好地记录单个用户的请求。
- **Vertex 将带回 Gemini 2.5**：用户对 **Gemini 2.5 Pro Exp** (*03-25*) 的弃用表示哀悼，并对“被阉割”的 05-06 版本表示不满，一些人希望它能回归。
   - 一位成员讽刺地指出缺乏“严肃的愤怒”，而其他人则讨论了遇到的内容过滤问题，以及非开源模型终究是昙花一现这一不幸事实。
- **Kluster 的 Qwen3 提供商出现 Tool Calling 故障**：一位用户报告了在使用 Kluster 提供商时，**Qwen3 235B** 及其 **tool calling** 功能存在问题，指出它会过早结束工具调用。
   - 他们发现 OpenRouter 有时会切换到另一个提供商，从而解决问题，但强制 OpenRouter 使用 Kluster 则始终会导致失败。
- **Anthropic 与 OpenRouter Claude 的集成**：一位用户声称 OpenRouter 的 Claude 实现是一个“骗局”，与直接通过 Anthropic 使用 Claude 相比，它提供的体验过于简化，且有太多的后续追问。
   - 其他人反驳说，这种差异源于 Anthropic 的 system prompts，而 OpenRouter 默认不包含这些提示词，原始模型的性能其实是完全相同的。 
- **优化 LLM 服务以最小化延迟**：一位成员建议 OpenRouter 优化网络路径和路由以最小化延迟，并提供具有不同服务水平协议的服务。
   - 他们建议提供托管或托管指南，并提供调节手段，以便在速度至关重要的工作流中指定“我愿意为最高速度付费，请相应地为我路由”。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1373048049042985020)** (174 条消息🔥🔥): 

> `VRAM Usage, LM Studio conversations export, LM Studio blurry UI, Vulkan runtime troubleshooting, Prompt formatting issues` 


- **LM Studio VRAM 使用情况受到质疑**：用户报告称，虽然 LM Studio 显示的 VRAM 数量正确，但即使将模型层卸载到 GPU，实际的 VRAM 使用率也非常低，一位用户报告在 9070 机器上仅使用了 *256-512MB* 的 VRAM。
   - 他们正在进一步调查潜在的驱动程序问题，并试图确定模型是加载到专用 VRAM 还是共享显存中，一些人怀疑这可能是驱动程序中的 bug。
- **用户需要导出 LM Studio 对话的方法**：用户正在寻找除了目前的 JSON 格式之外，以可利用的方式导出 LM Studio 对话的方法，特别是为了讲故事的目的。
   - 他们建议使用 LLM 编写一个工具来将 JSON 格式解析为首选格式，同时也承认 JSON 格式缺乏 API 保证。
- **LM Studio UI 显示模糊区域**：一位新的 LM Studio 用户报告了 UI 问题，描述了当鼠标从应用程序内的某些区域移开时会出现模糊部分。
   - 其他用户询问了相关的 LM Studio 版本，以便尝试复现或识别根本问题。
- **Prompt 格式化需要最新的 LM Studio 版本**：运行 **qwwn3 -32b** 模型的用户遇到了与 Prompt 模板相关的解析错误，表明模型的 Prompt 模板存在问题。
   - 解决方法是 [升级到 LM Studio 的最新 Beta 版本](https://lmstudio.ai/beta-releases)，该版本对 Prompt 格式的预设文件使用了不同的格式。
- **上下文过载导致模型表现异常 (Wily Wonka)**：一位用户寻求防止模型超过上下文限制的方法，同时保持一致的角色细节和过去事件的记忆，并建议使用外部文档存储。
   - 建议总结对话，使用 "truncate middle"（中间截断）选项，并向模型提供关于需要记住什么的明确指令。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1373012188456816821)** (341 条消息🔥🔥): 

> `Intel Arc Pro B60 GPU, macOS vs Windows, AMD vs Nvidia GPU for LM Studio, Resizable BAR Impact, Multi GPU setup` 


- **Intel 的 Arc Pro B60 GPU 引起关注**：成员们对 [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) 感到兴奋，因为它拥有 **96GB VRAM** 且潜在价格 **低于 $2000**，使其成为本地 LLM 的极具吸引力的选择。
   - 有人对潜在的软件支持挑战表示担忧，希望高 VRAM 显卡的可用性能 **增加 AI 领域对 Intel GPU 的支持**，并推动供应商提升水平。
- **macOS 的流畅度赢得了一些人的青睐，而另一些人仍偏好 Windows**：一位成员表示，在 Macbook 上使用 **macOS** 比 Windows 更流畅、更愉快，而另一位成员则分享了 macOS 窗口大小调整和应用程序管理方面的问题。
   - 另一位成员强调，Macbook 可以在电池供电下以 5t/s 的速度运行 **32B 4-bit 模型超过 5 小时**，且系统 RAM 占用 <2GB，提供了高效且安静的 LLM 运行体验。
- **9070XT GPU 表现不佳并需要 Resizable BAR**：一位用户报告 **9070XT** GPU 性能不佳，在全新安装 Windows 后，运行 **Gemma3 4b QAT** 模型仅获得 *8.52 t/s*。
   - 在启用 **Resizable BAR**（在 AMD 上也称为 **Smart Access Memory (SAM)**）后，用户达到了 *131.91 t/s*，突显了此设置对 LM Studio 的显著性能影响。
- **LM Studio 中不均匀的 GPU 性能**：成员们讨论了他们在 LM Studio 中拆分 GPU 性能的经验，一位成员指出，与 Windows 相比，Linux 上的性能最大限度地减少了多 GPU 的“惩罚”。
   - 另一位成员分享了他们如何使用 **Deepseek R1** 模型修复工作中较小模型无法解决的软件 bug，强调了尽管 Token 生成速度较慢，但使用较大模型的好处。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1373015324890365962)** (225 条消息🔥🔥): 

> `MCP Course Channel, AI Integration on ERP System, ACE Step Quality, AI Model for Design-to-Code Conversion, Hugging Face Pro Benefits` 


- **MCP 课程频道缺失**：几位成员正在寻找 **MCP 课程频道**，但目前似乎还不可用，现场分享了 [GitHub 上的课程链接](https://github.com/huggingface/mcp-course) 和相关的 Discord 讨论帖。
   - 关于该课程是否属于 **Agents 课程** 的一部分，困惑依然存在。
- **AI + ERP = 集成难题**：成员们对 **AI 与 ERP 系统集成** 的经验感到好奇，引发了简短的讨论。
   - 一位成员对 **ERP** 进行了澄清：*Enterprise resource planning*（企业资源计划）。
- **AI 工程师的 HF Pro 权益故障**：一位用户反馈 **AI Engineer Pack 提供的 6 个月免费 Hugging Face Pro 链接** 无法工作。
   - 建议该用户通过 [website@huggingface.co](https://discord.com/channels/879548962464493619/1353741359059570699) 联系 HF 支持团队。
- **Xet 基础设施和文件大小限制**：一位 **Xet 团队成员** 回应了关于文件大小限制的问题，指出虽然理论上可以使用 **hf_xet** 上传和下载 **>50GB** 的文件，但完整的 Web 端支持和最终设计决策仍在进行中。
   - 有成员请求将限制提高到 **200GB** 以适配 70B 模型。
- **Dropwise 模块：估算 HF 分类器的不确定性**：一位成员介绍了 **Dropwise**，这是一个用于在 Hugging Face 分类模型中通过 **Monte Carlo Dropout** 进行 **不确定性估计** 的 PyPI 模块。
   - 它具有预测熵、每类方差、置信度分数等特性，并支持 **transformers pipelines**；可在 [GitHub](https://github.com/aryanator/dropwise) 和 [PyPI](https://pypi.org/project/dropwise/) 上找到。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1373213142053687336)** (6 条消息): 

> `MCP for Atlassian, ChatApp AI App` 


- **提议将 Atlassian MCP 移植到 Claude Desktop**：一位成员正致力于将现有的适用于 Atlassian 工具（**Jira/Confluence**）的 **MCPs** 适配到 Claude Desktop 运行，并正在 [寻求合作者](https://www.atlassian.com/)。
   - 该项目旨在构建 **MCP 服务器**，将 **JIRA/Confluence** 与 Claude 集成，使其能够在对话中分析、创建、修改和获取 JIRA 工单详情。
- **新的 AI ChatApp 项目**：一位成员计划开发一个根据提供的资料集（如对话记录）进行回复的 AI ChatApp。
   - 该成员正在 [寻求指导和研究课题](https://www.example.com) 以辅助应用的开发。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1373049047484465173)** (4 条消息): 

> `Strands Agents SDK, AI for OS Funding` 


- **Strands Agents SDK 亮相！**：亚马逊推出了 [**Strands Agents SDK**](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)，这是一个经过深思熟虑的 **开源 AI Agent SDK**。
   - 该 SDK 旨在简化 AI Agent 的创建，为开发者提供更高效地构建和部署 Agent 的工具和资源。
- **资助机会：“AI for Open Source”**：一位成员通过 [**AI for OS**](https://os.nav.fund/ai-for-os/) 计划分享了面向数据科学领域初创公司的资助机会。
   - 该计划旨在支持利用 AI 增强开源技术的创新项目，为早期创业公司提供关键资金。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1373012620264607824)** (14 messages🔥): 

> `EcoArt Cellular Automaton, Browser.AI with Tool Calls, tome: Local LLM Client with MCP Servers, Lazarus Small LLM, datatune Open Source Tool` 


- **EcoArt Cellular Automaton 可视化**: 发布者介绍了 [EcoArt Cellular Automaton](https://huggingface.co/spaces/KvnMln/Mechanistic-interpretable-Ethics-Cell-automata)，它将自然之美与系统思维的复杂性相结合，使美德价值在 **教育**、**研究**、**艺术**、**开发**、**反思**和**冥想**中变得具体且可观察。
   - 该项目旨在探索价值观如何塑造系统。
- **Browser.AI 添加 Tool Calling 支持**: 一位成员宣布了 [Browser.AI](https://browser.christophermckenzie.com/) 的新版本，这是一个演示在设备上运行开源模型能力的浏览器原型，现在支持 **Chat**、**Tool Calls** 和 **Embeddings**。
   - 发布者正在征求对新版本的反馈。
- **tome: 支持 MCP Servers 的本地 LLM 客户端发布**: 一位成员介绍了 [tome](https://github.com/runebookai/tome)，这是一个简单的本地 LLM 客户端，可连接到 **Ollama**，并允许用户在无需复杂配置的情况下添加/使用 MCP Servers。
   - 该客户端集成了 **Smithery Registry**，可一键安装数千个 MCP Servers，开发者欢迎提出改进建议。
- **Lazarus: 下一个小型 LLM**: 一位成员分享了 [Aclevo/Lazarus](https://huggingface.co/Aclevo/Lazarus)，称其为下一个最佳的小型 LLM，它是[从 Llama3 蒸馏而来](https://huggingface.co/blog/codelion/pts)，拥有约 **1.24 亿参数**。
- **datatune 通过自然语言转换数据**: 来自 Vitalops 的一位成员介绍了 [datatune](https://github.com/vitalops/datatune)，这是一个新的开源工具，可以使用简单的自然语言指令和 LLM 进行数据转换。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

arpitbansal.: 请问最近一次会议有录音/录像吗？
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1373190699490738186)** (13 messages🔥): 

> `WANVideo Lora Training, Invoice Extraction, Computer Vision Roadmap, Object Outlines, CS231n Lectures` 


- **WANVideo Lora 训练 Loss 骤降**: 一位成员在包含 **130 个视频** 的数据集上训练 **WANVideo Lora**，分享了在使用 **Batch Size 为 2** 且 **Gradient Accumulation Steps = 2** 的情况下，运行 **8 小时**、**26 个 Epochs** 后的 [Epoch Loss](https://cdn.discordapp.com/attachments/922424143113232404/1373190699301998652/image.png?ex=682ccebf&is=682b7d3f&hm=56fa22b904be374ca12df36afe401f115988a2a189aa239a2ca7c685fd17095&)。
   - 另一位成员评论说，对于大多数图像模型，*Loss 的意义非常小*，并建议定期采样以查看结果效果，同时建议使用 **10%** 的数据集作为验证集。
- **发票提取寻求结构化流程**: 一位成员正在寻求 **发票提取** 的结构化流程，特别是将实体提取为 **键值对 (Key-Value Pairs)** 并整理 **表格数据**，因为他们发现 OCR 和 LayoutLM 的输出不尽如人意。
   - 建议包括预处理、OCR、LayoutLM/Donut、NER 和后处理，但该成员在 *有效实现 LayoutLMv3* 方面遇到了困难。
- **寻求 Computer Vision 路线图**: 一位成员请求一份清晰且结构化的路线图/资源，以专注于 CV 领域的主题，如 **Object Detection**、**Semantic & Instance Segmentation**、**Object Tracking** 和 **3D Computer Vision**。
   - 该成员已经很好地掌握了 **OpenCV 基础**、**数学基础**、**Machine Learning 基础** 以及 **Deep Learning for Computer Vision**，并能熟练使用 **Python** (**Tensorflow** 和 **PyTorch**)。
- **寻求物体轮廓模型**: 一位成员询问是否有擅长从图像或物体的 **BREP** 中获取 **物体轮廓** 的模型。
   - 提供的消息中没有推荐具体的模型。
- **推荐 Karpathy 的 CS231n 课程**: 一位成员建议通过学习 **Andrej Karpathy** 的 **Stanford CS231n 讲座** 来建立直觉。
   - 该成员还建议观看或阅读探索经典计算机视觉和机器学习的内容，例如 YouTube 上的 **Andrew Ng 课程**。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1373368517029662900)** (3 条消息): 

> `DDS 的现代方法，BERT-style 模型中的推理差异` 


- **杜威十进制分类法 (Dewey Decimal System) 被视为过时产物**：一位成员分享了其观点，认为由于当今数据量巨大，使用 **Dewey Decimal System** 对数据进行分类已成为过去式的产物。
   - 他们还提到正在探索更现代的方法和替代分类技术，并提供了一份关于 [通用数据框架的 PDF](https://cdn.discordapp.com/attachments/922424173916196955/1373368516719415437/02._From_Chaos_to_Order__The_Universal_Framework_for_Data.pdf?ex=682ccb9a&is=682b7a1a&hm=d621f23829765a944006f237a349ba321f9d3f4e7ca6aaed225340f62fe95f81&)。
- **BERT 的推理产生不同的结果**：一位成员询问，在相同的 **BERT-style 模型**和任务上，即使使用相同的 tokenization 和配置文件，但在不同的库（Candle 和 PyTorch）中运行推理时，获得显著不同的 logits 是否出乎意料。
   - 该问题暗示虽然 logits 不同，但分类结果基本保持一致，该成员表示：“如果我在两个不同的库中，针对同一任务，在同一个 **BERT-style 模型**上运行推理……得到显著不同的 logits 是否不正常？”


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1373114438755942400)** (11 条消息🔥): 

> `Claude “过载”问题，GPT 30% 的成功率，Meta Llama 访问被拒，多 Agent 设置，Questions 端点问题` 


- **GPT 解决了 Claude 的“过载”提交障碍**：一位成员报告称，由于 **Claude** 处于“过载”状态，他们花了一天半的时间尝试提交，但切换到 **GPT** 后立即解决了问题，并实现了 **30%** 的成功率。
   - 尽管他们是 **Claude** 的粉丝，但在过去八天左右的时间里，他们遇到了“许多关于它的问题”。
- **Meta Llama 模型拒绝课程参与者的访问**：一位成员被拒绝访问 **Meta Llama Models**，导致他们无法进行课程和 notebook 的学习。
   - 另一位成员建议使用 **Qwen** 或 **Gemma** 等替代模型。
- **多 Agent 设置需要 Prompt 调整**：一位成员提到尝试通过将工具分离到多个 Agent 中来进行 **multi-agent setup** 实验，但“就这样直接运行效果并不好”。
   - 他们强调需要修改工具和 Agent 的 prompt，并表示：“遗憾的是，除了非常基础的例子外，网上几乎没有相关示例。”
- **Questions 端点缺少附件文件**：一位成员报告称，最终作业的 **questions endpoint** 缺少附件文件（例如 **.py**、**.xlsx**）。
   - 当访问 questions endpoint 时，用户只收到了 **JSON** 响应，而没有预期的附件。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1373027781499158609)** (46 messages🔥): 

> `Agents 课程认证、MCP 课程混淆、Unit 4 项目文件检索、GAIA 格式问题、Hugging Face Space 卡住` 


- **Agents 齐心协力获得认证**：几位成员庆祝完成了他们的 **Unit 4 Agent** 并获得了 Hugging Face Agents 课程的认证。
   - 其他新成员也在组建学习小组，以在截止日期前完成课程并获得认证。
- **MCP 课程混淆干扰讨论**：成员们澄清了 **MCP（推测为 Machine Learning Certification Program）课程**与 **Agents 课程**是不同的，建议在单独的（目前不存在的）MCP 频道中寻求详情。
   - 其他人幽默地评论说，他们可能会在等待 **MCP 课程**信息的同时完成 **Agents 认证**。
- **文件困扰：Agent 的 Unit 4 文件**：一位成员询问如何为 **Unit 4 项目**中的任务检索 **.png** 或 **.mp3** 等文件。
   - 另一位成员提供了相关的代码片段：`files_url = f"{api_url}/files/{task_id}"`。
- **GAIA 的陷阱：强制精确匹配**：一位成员强调了 **GAIA** 要求精确匹配的繁琐性，指出评估格式不够灵活；另一位成员建议“润色 system message”。
   - 成员们建议参考 [建议的 system prompt](https://huggingface.co/spaces/gaia-benchmark/leaderboard) 来解决 **GAIA** 格式问题并提高分数。
- **Space 卡住了？评估焦虑出现**：一位成员报告他们的最终评估 Space ([https://huggingface.co/spaces/vaibhav-vibe/agents_final_assessment](https://huggingface.co/spaces/vaibhav-vibe/agents_final_assessment)) 卡在构建阶段，并询问社区是否有解决方案。
   - 另一位成员分享了一个可能相关的 [ChatGPT 链接](https://chatgpt.com/share/682b5764-9624-8008-b387-4532bdae4fc6)，而该 Space 的创建者提到尝试了 factory rebuild 和硬件切换但均未成功。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1373386018639712426)** (8 messages🔥): 

> `Kernel 开发、AMD 挑战赛、在 trl 中使用 FSDP 和 Flash Attention 2` 


- **从零开始的 Kernel 项目招募贡献者**：一位成员正在从零开始构建一个 Kernel，并正在寻找贡献者，更多详情可在 [GitHub](https://github.com/DorsaRoh/Kernels) 上找到。
   - 该项目被明确标记为 "gpu mode = 🔥"，表明其专注于 GPU 相关的 Kernel 开发。
- **AMD 挑战赛参与困难**：一位 **AMD 挑战赛**的参与者在被添加到 leaderboard 频道时遇到问题，正在寻求联系人的指导。
   - 其他成员建议检查特定频道的权限和访问权限 [<#1359640791525490768> 和 <#1343002583001726986>]，这有助于解决问题。
- **结合 FSDP 和 Flash Attention 2**：一位成员正在寻求关于如何使用 `trl` 同时通过 `FSDP` 和 `Flash Attention 2` 训练模型的建议，因为他们可以分别让两者工作，但无法同时运行。
   - 他们引用了 [trl `sft.py` 脚本](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)，并分享了尝试在未在 GPU 上初始化的模型上使用 **Flash Attention 2.0** 时收到的错误消息。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1373202465524154509)** (4 messages): 

> `Triton 运行时共享内存、Triton 在 CPU 上运行、TRITON_INTERPRET API` 


- **Triton 运行时共享内存疑问**：一位成员询问了 Triton 的内存大小计算，认为这应该是编译时计算而不是运行时计算。
   - 另一位成员回答说 Triton 并不直接支持 CPU 并行，但使用 `TRITON_INTERPRET=1` API 可以允许在 CPU 上顺序执行。
- **在 CPU 上运行 Triton**：一位成员询问在 Ubuntu v22.0 的 CPU 上运行 Triton 所需的具体 flag 或设备 API，最终目标是在 GPU 上运行。
   - 他们还在寻找用于 CPU 测试的小代码示例，例如 **matmul**、**fastmul** 或 **vector-add**。
- **模拟并行方案**：一位成员建议，虽然 Triton 不直接支持 CPU，但它可以几乎完美地模拟并行方案。
   - 他们建议这“将适用于你的目的”。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1373101320529121300)** (11 messages🔥): 

> `Tensor Cores, CUDA Brute Forcer, GPU Usage Reporting, Neural Net Mutate Function, CUDA Errors` 


- **Smileforme 询问关于 Systolic Arrays 的问题**：Smileforme 询问 **Tensor Cores** 是否是以 **Systolic Arrays** 的形式实现的。
- **CUDA 暴力破解器的低 GPU 使用率令人困惑**：kr1v 报告称创建了一个 [CUDA 暴力破解器](https://github.com/kr1viah/WKChallengeModeSeedFinder)，但在任务管理器中观察到 **GPU 使用率仅为 0-20%**，并试图找出原因。
   - mrsteed 指出 *Windows 任务管理器并不能 100% 可靠地报告 GPU 使用率*，建议使用 `nvidia-smi` 获取更可靠的统计数据。
- **找到任务管理器设置**：ug0x01 建议将任务管理器切换到 **Cuda** 视图以查看**准确的使用率**，并[通过图片](https://cdn.discordapp.com/attachments/1189607726595194971/1373842591036084264/image.png?ex=682c8ade&is=682b395e&hm=c63888438da7eef855829439bba3e467b828a5b5c101364d01e033bd1cdb21e7)展示了如何在任务管理器中选择 **Cuda**。
- **神经网络变异困扰成员**：一名成员在编写**神经网络**的 **mutate function** 时遇到困难，在随机生成过程中遇到了 *unspecified launch failure* 或 *illegal memory access* 错误。
   - 他们使用了 *memcheck*，结果显示 *Malloc/Free Error encountered : Invalid pointer to free*，并提供了一个 [godbolt 链接](https://cuda.godbolt.org/z/z8z6a85vP)。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1373057306438271050)** (1 messages): 

> `Triton Kernels, Dynamic Shapes, Batch Size in PyTorch` 


- **PyTorch 在处理 Dynamic Shapes 时默认使用 Triton Kernel**：当 **Batch Size** 设置为 **None** 时，PyTorch 默认使用支持 [Dynamic Shapes](https://pytorch.org/docs/stable/generated/torch.jit.script.html) 的 **Triton Kernel** (`extern_kernels.mm`)。
   - 系统不会为自动生成的 Kernel 填充激活值（pad activations）；相反，它会调用 `extern_kernels` 并带上一个表示 **Batch Size** 的 `s0` 参数。
- **PyTorch 为每个 Batch Size 生成特定的 Triton Kernels**：对于特定的 **Batch Size**，PyTorch 会生成**自定义 Triton Kernels**（例如 `triton_tem_fused_mm_4`），而不是使用通用的 **Dynamic Shape** Kernel。
   - 这种优化避免了填充（padding），并允许针对每个定义的 **Batch Size** 执行定制化的 Kernel，从而可能提高性能。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1373373636248998058)** (1 messages): 

> `CuTe, Cris Cecka` 


- **CuTe 发明者将在 5 分钟内发表演讲！**：**CuTe** 的发明者 Cris Cecka 将在 5 分钟后开始关于 **CuTe** 的演讲。
- **另一个 CuTe 总结**：额外添加一条总结以满足最低字数要求。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1373166881904529470)** (43 messages🔥): 

> `Power Consumption for MCMC, Pbit Hardware, Analog Annealing Circuit, Quantum vs Pbit, Hardware for MCMC` 


- **成员寻找 MCMC 功耗资源**：一名成员请求更多关于 **MCMC** 算法**功耗**的资源，并引用了[这篇论文](https://arxiv.org/pdf/1903.11714)作为例子。
   - 作为回应，有人分享了[三篇相关论文](https://arxiv.org/abs/2411.04260)的链接，以及[另一篇论文](https://arxiv.org/abs/2002.01184)和[又一篇论文](https://arxiv.org/abs/2003.02629)。
- **Pbit 硬件助力概率计算**：在分享了[这段采访](https://www.youtube.com/watch?v=5O5do_N07kY)后，讨论转向了概率比特（**pbit**）硬件及其在概率算法中实现**能效**的潜力。
   - 一名原型设计过用于组合优化的**模拟退火电路（analog annealing circuit）**的成员表示，他们的电路在每次翻转（flip）的能量消耗上应该比现有方法低约 100 倍。
- **MCMC 硬件探索开启**：成员们研究了适用于 **MCMC** 的最佳硬件和 Kernel，其中一人考虑使用 **FPGA**，但发现经典的 **Xilinx** 型号并不可行。
   - 主要挑战在于如何以**低功耗**创建**快速、高质量的随机性**，并在随机性质量与采样偏差之间取得平衡。
- **TPU 胜过 Nvidia？**：一名成员回忆起曾读到 **Google TPU** 在 **MCMC** 方面比 **Nvidia GPU** 更有优势，并引用了[这篇论文](https://arxiv.org/pdf/1903.11714)，该论文将 TPU 与 GPU 进行了对比。
   - 讨论指出 **JAX** 具有非常方便的**并行计算**特性，这为 NumPyro 和 BlackJAX 库提供了基于 JAX 的后端支持。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1373532531126833172)** (2 messages): 

> `External CUDA Allocator, MAXSUN Arc Pro B60 Dual` 


- **外部 CUDA 分配器 (External CUDA Allocator) 发布**：一位 GitHub 用户分享了一篇 [博文](https://kshitij12345.github.io/python,/pytorch/2023/02/26/External-CUDA-Allocator-With-PyTorch.html)，详细介绍了如何在 PyTorch 中使用 **external CUDA allocator**。
   - 该用户演示了如何扩展 PyTorch 的内存管理能力。
- **关于 MAXSUN Arc Pro B60 Dual 的讨论**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Y8MWbPBP9i0)，展示了 **MAXSUN Arc Pro B60 Dual**。
   - 该视频似乎提供了该产品的概览或评测。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1373737452396609648)** (1 messages): 

> `Threading APIs, cuper alternative` 


- **cuper 不再可用了？**：一位成员询问哪些线程 API 最适合用于分析实时内存层级 (memory hierarchy)。
   - 他们补充说 **GPT** 提到 **cuper** 很好，但似乎现在已经无法使用了。
- **实时内存层级需要性能分析**：该咨询重点在于寻找合适的线程 API，以分析实时内存层级的性能。
   - 目的是寻找 **cuper** 的替代方案，以便进行有效的内存层级分析。**GPT** 曾建议使用 **cuper**，但据报道该工具已不可用。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1374028977496260608)** (3 messages): 

> `QAT with Llama3.2 3B, prepare_model_for_qat, prepare_model_for_ptq, bf16 vs int4, axolotl-ai-cloud/axolotl` 


- **Llama3.2 3B 的 QAT 深度探讨**：一位成员寻求关于 **Llama3.2 3B** 的**量化感知训练 (QAT)** 的调试帮助，并指出经过 **QAT 训练的量化模型**表现并未优于**全量微调后量化的模型**。
   - 另一位成员分享了他们相关的 [Axolotl 配置](https://github.com/axolotl-ai-cloud/axolotl/pull/2590/files#diff-29a7a53548f024812e8a2dc36eccc625c6b278b22b105f4eb5a924c63452a781) 和 torchtune 配置 ([gist.github.com](https://gist.github.com/andrewor14/f1121b9b4c2ccc50e0cc1726859eb79e)) 作为回应。
- **QAT vs 基准线：剖析微调流程**：成员们详细说明了 **QAT** 和**基准模型**的流程，强调唯一的区别在于基准流程中缺少 `prepare_model_for_qat`。
   - 两个流程都包括从 HF 加载 **bf16 模型**、在 alpaca 数据集上进行微调、保存 **bf16 微调模型**、应用 `prepare_model_for_ptq` 转换为 **int4**，最后使用 lm_eval 进行评估。
- **请求命令：寻求复现**：一位成员请求用于微调、量化和评估的命令，以便复现实验并找出潜在的差异。
   - 目标是查明 **QAT 训练模型**与**基准模型**之间性能差异的根本原因。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1373155113077833858)** (3 messages): 

> `kpack argument in rocm triton, AMD Triton Performance Optimization` 


- **关于 ROCm Triton 中 `kpack` 参数的讨论**：一位用户询问了 **ROCm Triton** 中的 `kpack` 参数，注意到 Torch Inductor 默认将其设置为 **2**，并引用了一行 [代码](https://github.com/pytorch/pytorch/blob/e802b29ed499cdeba24b366830a1c76d4d8b8511/torch/_inductor/template_heuristics.py#L55)。
   - 该用户在 [ROCm Triton 文档](https://github.com/ROCm/triton/wiki/General-Guide-of-AMD-Triton-Performance-Optimization#kpacktilelang) 中找到了解释，其中 `kPack` 用于带有 `ld128` 的 **MFMA**，但可能对性能没有显著影响。
- **kpack 详情**：kpack 用于带有 ld128 的 MFMA。
   - 根据文档，它可能不会显著影响性能。


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1373612104007028906)** (4 messages): 

> `CuTe Tensors, Lecture Slides Availability` 


- **CuTe Tensors 使用任意嵌套布局 (Arbitrary Nested Layouts)**：一位成员询问了 **CuTe tensor 库**，特别是它如何将 stride 提升为任意嵌套的整数元组，以支持 Tensor Core **GEMM** 的各种布局。
   - 他们询问这些嵌套布局是否有任何缺点，以及像 **PyTorch** 这样的 Tensor 库是否可以采用它们。
- **讲座讲义即将发布**：一位成员询问了最近一系列讲座的讲义是否可用。
   - 另一位成员回应说他们已经去要了，应该很快就会发布，并请求上传自第 43 讲以来所有讲座的讲义。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1373068325642567781)** (3 messages): 

> `SWEBench, CuTeDSL, AI Efficiency with Pruna AI` 


- **Ofir Press 谈论 SWEBench/SWEAgent**：Ofir Press 将在周三的 [PyTorch Webinar](https://pytorch.org/event/towards-autonomous-language-model-systems/?utm_campaign=6571433-PyTorch+Webinar+Promotion&utm_content=332357919&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024) 上发表关于 **SWEBench/SWEAgent** 的演讲。
   - 演讲重点关注 **autonomous language model systems**，并可能涵盖 benchmarking 策略。
- **CuTeDSL Layout Algebra 揭秘**：**CUTLASS** 团队发布了 **CuTeDSL**，一名成员在 [博客文章](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/) 中结合数学概念解释了 **CuTeDSL**。
   - 该博客文章涵盖了底层的数学概念，如 **Layout Function**、**Coalescing** 和 **Complementation**。
- **Pruna AI 倡导实用性**：Pruna AI 现在定期组织关于 **AI efficiency** 的活动，包括每月一次的 webinar 和线下见面会。
   - 下一次 webinar 将关于如何 **compress and deploy AI models on clouds** ([链接](https://app.livestorm.co/pruna-ai/pruna-koyeb-event?utm_source=Livestorm+company+page))，首场线下活动将于 5 月 28 日举行。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1374018914761183293)** (6 messages): 

> `KernelLLM, PyTorch backend, RL baseline, leaderboard competitions, pass@k evals` 


- **Facebook 发布 KernelLLM**：Facebook 发布了其 [KernelLLM](https://huggingface.co/facebook/KernelLLM) 的首个公开权重。
   - 一位成员链接了 [包含更多细节的演讲](https://www.youtube.com/watch?v=FtgXueoQxA0)，并指出他们最新的 baseline 比展示的内容要强大得多。
- **KernelLLM 计划揭晓**：一名团队成员概述了 KernelLLM 的后续步骤，包括作为评测套件的 **PyTorch backend**、**RL baseline**、更多 **leaderboard competitions**，以及在 leaderboard 数据上训练的用于翻译和编辑任务的 baseline 模型。
   - 他们补充说，这些内容应该记录在更显眼的地方。
- **Pass@k 评测遭到质疑**：一位成员对比较许多不同的 **pass@k** 值的做法提出质疑。
   - 他们指出这看起来很奇怪，并好奇是否有更好的评测方法，尤其是对于推理模型仍然可以进行 **pass@k evals** 的情况下。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1374059255325261854)** (3 messages): 

> `LLMs, Qwen 2.5 3B, Llama 3.2 3B, GSM8K and MATH benchmarks` 


- **Reasoning Gym 的训练和评估配置已就绪！**：一位成员分享说，一篇包含 [Reasoning Gym](https://github.com/open-thought/reasoning-gym/tree/main/training) 训练和评估配置的论文将在未来几天内发布在 Arxiv 上！
   - Readme 包含信息和复现指令，不过完整的评测结果将随完整报告一同发布。
- **RG 数学任务增强了 Benchmark 表现**：一些较小的 LLM（**Qwen 2.5 3B** 和 **Llama 3.2 3B**）在 RG 推理数据上进行了训练。
   - 在一系列 RG 数学相关任务上的训练提高了在 **GSM8K**，尤其是 **MATH** benchmark 上的表现。
- **Qwen 在推理数据上的表现优于 Llama**：通常情况下，与 **Llama** 相比，**Qwen** 似乎更擅长从数据中学习。
   - 还将会有一些关于 frontier LLM 的 zero-shot 性能结果，展示了这些数据在评测/benchmarking 以及训练方面的价值。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1373015422978494484)** (111 条消息🔥🔥): 

> `amd-fp8-mm 排行榜, amd-mixture-of-experts 排行榜, MoE 提交错误, amd-identity 排行榜, hipcc 参数` 


- **MI300 上的 FP8-MM 对决**：针对 **MI300** 上的 `amd-fp8-mm` 排行榜进行了多次提交，耗时从 **150 µs** 到 **3.76 ms** 不等，多位用户刷新了个人最佳纪录。
   - 一名用户以 **154 µs** 夺得 **第 4 名**，另一名用户以 **150 µs** 稳居 **第 3 名**，其他用户则以 **160 µs** 左右的成绩排在 **第 6 名**。
- **MI300 上的 MoE 热潮**：`amd-mixture-of-experts` 排行榜竞争激烈，在 **MI300** 上的提交成绩从 **255 ms** 到 **7564 ms** 不等，其中一名用户以惊人的 **9.64 ms** 登顶 **第一名**。
   - 用户们庆祝刷新个人纪录，不断挑战 **MI300** 的性能极限，甚至有用户仅用 **14.7 ms** 就排到了 **第 4 名**。
- **提交故障困扰 MoE**：有用户报告在通过 Discord 提交 `amd-mixture-of-experts` 文件时出现 *“创建提交时出错”*，特别是当文件大小超过 **10KB** 时。
   - 其他人注意到，当文件中包含制表符 (`\t`) 或换行符 (`\n`) 时也会导致提交失败，但反斜杠 (`\`) 则没有问题。
- **MI300 上的 Identity 洞察**：`amd-identity` 排行榜迎来了新冠军，在 **MI300** 上以极速的 **5.50 µs** 获得 **第一名**。
   - 其他用户也成功提交，成绩维持在 **20 µs** 左右，另有一名用户以 **6.79 µs** 夺得 **第一名**。
- **Hipcc 参数处理**：一名用户请求移除特定的 `hipcc` 参数（*`--offload-arch=gfx900 --offload-arch=gfx906 ...`*），理由是它们与 **MI300** 上的专用指令不兼容。
   - 另一名用户建议在 Python 中使用 *`os.environ['PYTORCH_ROCM_ARCH'] = 'gfx942'`* 来解决此问题。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1374077530419494972)** (1 条消息): 

> `问题 #3 已发布, MLA 速成课程, Bot 超时减少, 截止日期延长` 


- **问题 #3 登上排行榜**：感谢部分成员的努力，问题 **#3** 现已在 [排行榜](https://www.gpumode.com/leaderboard/463) 上线。
   - 作者旨在让问题既简单又有趣。
- **多头潜在注意力 (MLA) 解码指南发布**：一名成员编写了 **Multi-head Latent Attention (MLA) Decoding** 指南，可在此处查看 [链接](https://stormy-sailor-96a.notion.site/Multi-head-Latent-Attention-Decoding-1f0221cc2ffa803dbe1acb16bb510a40?pvs=74)。
   - 该指南可作为 MLA 的速成课程。
- **Bot 超时减少**：得益于部分成员的工作，Bot 已更新，减少了超时的可能性。
   - 未提供更多细节。
- **问题截止日期延长至 6 月 2 日**：由于问题 **#3** 发布较晚，所有问题的截止日期已统一延长至 **6 月 2 日**。
   - 这将截止日期延长了 **2 周**。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1373662486506635335)** (1 条消息): 

> `llvm-mca 用法, CPU 执行估算` 


- **`llvm-mca` 估算 CPU 执行**：一名成员分享了 [一个示例](https://ppc.cs.aalto.fi/ch2/v5asm/)，展示了如何使用 `llvm-mca` 来估算汇编代码在 CPU 上的执行情况。
- **LLVM-MCA 工具使用**：该示例演示了利用 `llvm-mca` 工具来估算 CPU 上汇编片段执行的方法。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1373045026518732982)** (14 messages🔥): 

> `Space Age 兼容性、工作组的长期愿景、vLLM 和 OpenAI 客户端类、理解用例和评估领域` 


- **每周会议（Weekly Meeting）已上线**：团队在**每周三 16:00 UTC**（太平洋时间上午 9 点，伦敦时间下午 5 点）举行面向所有人的**每周会议**。
   - 他们鼓励大家通过**私信（DM）**发送邮箱地址，以便被添加到定期会议邀请名单中。
- **成员从 Space Age 兼容性中获得灵感**：一位成员在 [github.com/snarf-dev/fsm](https://github.com/snarf-dev/fsm/tree/main/screenshots) 链接中找到了针对 **Space Age 兼容性**相关开放问题的潜在解决方案。
   - 他表示：“在这种情况下，‘不可能’绝对是一个特性，因为这意味着 Benchmark 将永远无法达到饱和。”
- **理解工作组的长期愿景**：一位成员阅读了论文 ([https://arxiv.org/pdf/2503.09617](https://arxiv.org/pdf/2503.09617)) 和聊天记录，对该工作组的**中长期愿景**感到有些困惑。
   - 另一位成员建议，他们的首要目标是了解这种（无界）环境可以开启哪些**有趣的用例和评估领域**。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1373060748313624737)** (85 messages🔥🔥): 

> `Mixture-of-Experts 提交问题、HIP 提交、Popcorn CLI 输出、排行榜运行速度慢于 Benchmark、Composable Kernel 库错误` 


- **MoE 排名提交遇到错误**：用户报告了 **Mixture-of-Experts** 配置在排名提交时的问题，在处理 **10 分钟**后遇到“意外错误”。
   - 修复程序已实施并即将上线：[reference-kernels commit 7d8a576](https://github.com/gpu-mode/reference-kernels/commit/7d8a57661a684f6a11270e4855179df5d0f1dff1)。
- **HIP 代码需要 Python 封装**：使用 **HIP** 的提交需要一个 **Python 脚本**来调用 Kernel，因为不支持纯 **C++** 提交，示例见：[template-hip.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd/fp8-mm/template-hip.py)。
- **Popcorn CLI 获得文件输出选项**：一位用户请求能够直接从 **Popcorn CLI** 将 Benchmark 结果输出到文件，而不是手动复制。
   - 一位成员指出 `-o` 标志提供了该功能，但该 [feature](https://github.com/gpu-mode/popcorn-cli/pull/8) 尚未被采纳。
- **排行榜计时不一致的解释**：用户注意到他们的排行榜运行速度慢于其本地 Benchmark 运行，团队正在[调查原因](https://cdn.discordapp.com/attachments/1359640791525490768/1373399965832974387/Screenshot_2025-05-03_142043.png?ex=682ce8e4&is=682b9764&hm=bf0bb206889151e0924af5919ce178a75f2843800fc1bc4097c735b7c4c465cd&)。
   - 减速似乎因代码细节而异，可能与 **L2 Cache** 行为或硬件时钟差异有关，特别是在 `recheck=False` 时。
- **Composable Kernels 和 offload-arch 标志需要调整**：由于提交服务器添加了多个 `--offload-arch` 标志，包括不支持的 `--offload-arch=gfx1030`，用户在使用 **Composable Kernel 库**时遇到了错误。
   - 在 `load_inline` 函数中添加 `extra_cuda_cflags=['--offload-arch=gfx942']`（如 [template-hip.py](https://cdn.discordapp.com/attachments/1359640791525490768/1373515912048541786/image0.jpg?ex=682cac20&is=682b5aa0&hm=23823fb51c5f50dd9f0ecea948f9d2a64a435274fac897979990633ab2b78b3a&) 所示），并使用 `gfx942:xnack-` 可以解决此问题。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1373406268537962557)** (11 messages🔥): 

> `CUTLASS DSL 4.0, CuTeDSL blogpost, CuTeDSL examples, Layout Function` 


- **CUTLASS DSL 4.0 支持 Linux 和 Python 3.12**：**CUTLASS DSL 4.0** 版本目前仅支持 **Linux** 和 **Python 3.12**。
- **推文示例已过时**：CUTLASS 团队在发布推文后更新了他们的示例和 GTC 幻灯片，因此用户应使用[最新的示例](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL)。
- **CuTeDSL 博客文章发布**：CUTLASS 团队向公众发布了 **CuTeDSL**，并有一篇[博客文章](https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/)解释了 **CuTeDSL** 以及关键的底层数学概念，如 **Layout Function**、**Coalescing** 和 **Complementation**。
- **转置线程分块（thread tiling）导致错误**：一位用户询问了一个关于线程分块导致错误的 Kernel，并提供了[代码片段](https://gist.github.com/simveit/ab0a28efb4338592f82c0a8f762f0ac7)作为参考。


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1373274824847134792)** (2 messages): 

> `CUTLASS 4.0, DSL, cuTile, Python, Triton` 


- **CUTLASS 4.0 启发 Mojo 团队**：随着 **CUTLASS 4.0** 发布了新的 **DSL**（旨在通过 **Python** 抽象实现 **CUTLASS** 级别的性能），该团队表示愿意学习其中的优秀想法。
   - 团队还指出了 [Democratizing AI Compute, Part 7](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls) 中关于使用“真正的编程语言”而非 **DSL** 的优势。
- **引发关于 DSL 与真正的编程语言的辩论**：围绕 **CUTLASS 4.0** 的 **DSL** 方法的讨论，引发了对 **DSL** 与像 **Mojo** 这样成熟的“真正的编程语言”之间差异的思考。
   - Modular 的博客文章认为，虽然 **DSL** 有其地位，但“真正的编程语言”为 AI 计算提供了更广泛的能力和灵活性。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1373017076444434453)** (235 messages🔥🔥): 

> `Document AI and NLP, Sociology in biomedical AI, AI_WAIFU's views on AGI, OpenWorldLabs, Diffusion Limitations` 


- **金融科技数据科学家致力于文档 AI**：来自印度的 Akhil Theerthala 是一家金融科技公司的数据科学家，正在从事 **Document AI 和 NLP** 相关的项目，包括文档表示、TSR（表格结构识别）和文档质量评估。
   - 他还在进行一些个人项目，涉及 **个人财务** 推理、**伦理困境** 以及 **简历/职业路径分析**。
- **AI_WAIFU 缩短 AGI 时间线**：**AI_WAIFU** 表示，由于编程模型（coding models）大大加速了开发进程，他们现在的 AGI 时间线变短了，大部分概率分布在今年或明年年初。此前他们已离开 EAI 去从事新的工作。
   - 然而，他们对“快速起飞”（fast takeoff）的信心有所下降，并指出提高 **NN 效率** 的低垂果实更多地转化为小模型的良好性能，而不是大模型能力的显著提升，尽管纳米技术可能比 AGI 本身需要更多的算力。
- **OpenWorldLabs 专注于世界模型**：**OpenWorldLabs** 公司为机器人和视频游戏开发 **世界模型**，可在其[网站](https://openworldlabs.ai)上访问。
   - 成员们表示难以理解其业务，其中一人表示在阅读了网站和 **GitHub** 两次后，“我仍然几乎不知道你们到底在做什么”。
- **Diffusion 模型努力应对误差累积**：一位成员表示，解决 **误差累积** 需要对上下文帧进行重新加噪（renoising），而现有的视频模型速度不够快，无法解决这个问题；同时还指出，由于误差累积，你“无法缓存 **diffusion** 过程”。
   - 另一位成员表示，**Google** 可能已经通过带误差训练（training with error）并有效地训练模型进行逐帧去噪解决了这个问题。
- **社区达成 Universal Transformer 结论**：成员们表示，AR（自回归）是一个非常广泛的公式，以至于 **Universal Transformer/RNN** 类型的心理模型可以轻松迁移。
   - 其他成员表示赞同，并总结说 Universal Transformers 就像螃蟹一样，是 AR 领域最终的生命形态。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1373017907025674403)** (34 messages🔥): 

> `Approximating LM Finetuning, ReLU Activation, Smooth Activation Function, Muon expensive operations, Grand Research Ideas` 


- **LM Finetuning 需要实际实验**：需要实际实验来证明在 [LM finetuning](https://openreview.net/forum?id=D2PjEPGXghI) 中仅使用 `k << n_vocab` 个 token 进行近似时的稳定性。
   - 链接的论文对某些成员来说很有趣，但实验非常有限。
- **平滑激活函数在平滑参数化 PDE 中表现出色**：在一次应用数学研讨会上，有人解释了不使用 **ReLU activation** 的原因，因为他们知道目标参数化 PDE 是平滑的。
   - 该论文的目标是研究无限平滑的网络如何很好地近似高维 k-连续可微函数。
- **PTS 后训练技术博客**：一位成员分享了关于 [PTS](https://huggingface.co/blog/codelion/pts) 的博客文章。
   - 另一位成员澄清说该文章发错了频道。
- **尚未被充分探索的 Empowerment 想法**：一位成员分享了一个 [arxiv 链接](https://arxiv.org/abs/2505.10361)，认为 *empowerment 尚未被充分探索*。
   - 这可能非常相关，特别是 **Muon** 使用了一系列原本开销巨大的 **XX^T** 操作。
- **不欢迎宏大的研究想法**：一位成员解释说，“我们这里有太多人带着主要由 LLM 编写的宏大研究想法过来”。
   - 每一个这样的想法都是在浪费时间。这是另一个 [arxiv 链接](https://arxiv.org/abs/2403.00329)。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1374029274436210783)** (8 messages🔥): 

> `SpinQuant Llama-2-7b Reproduction, lm eval harness, HFLM modification` 


- **SpinQuant 技术复现**：一位成员正在寻求指导，以在来自 [Facebook Research](https://github.com/facebookresearch/SpinQuant) 的不同位精度量化 **Llama-2-7b** 模型上，使用 **SpinQuant** 技术复现零样本推理任务。
   - 他们特别想知道如何将自定义的量化模型与 *lm eval harness* 结合使用，因为 SpinQuant 需要自定义的量化代码。
- **结合外部库利用 lm eval harness**：建议该成员可以通过传递一个已经初始化的模型来使用 *lm eval harness*，参考[此文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage)。
   - 这种方法需要将模型封装在一个类中，可能通过继承或修改 *lm eval harness* 中现有的 [HFLM 类](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L47)来实现。
- **为量化模型定制 HFLM**：提供了关于如何修改 **HFLM** 类以适配自定义量化模型的指导，包括将初始化的模型传递给 `pretrained` 并[修改 `_model_call` 方法](https://github.com/EleutherAI/lm-evaluation-harness/blob/53c653008182339e67b964a4cd3316f651611f38/lm_eval/models/huggingface.py#L870)。
   - 提供了 `_model_call` 方法的链接以便进一步定制。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1373698894306869248)** (3 messages): 

> `PolyPythia Materials, Pretraining Data, Random Seeds, GPT-NeoX hash` 


- **PolyPythia 材料混淆**：一位成员寻求关于 Hugging Face 上提供的 **PolyPythia materials** 的澄清，特别是关于使用每个随机种子构建的预训练数据和文件夹编号。
   - 他们询问标记为 **0** 的文件夹是否对应于随机种子为 **1234** 的原始运行，因为论文中展示的 **10 次运行** 与文件夹编号 **0-9** 之间存在差异。
- **配置文件确认任务**：一位成员提到配置文件应该是 GitHub 上的那些，并表示打算在当天晚些时候通过与 [WandB](https://wandb.ai/eleutherai/pythia-extra-seeds) 对比来确认。
   - 他们印象中 **GPT-NeoX hash** 也记录在那里，但目前没有看到，计划进一步调查。
- **随机种子之谜已解决**：一位成员确认实验中使用的惯例是 **seed 0** 确实等同于 **seed 1234**。
   - 另一位成员也坚持这一惯例。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1373030652357775410)** (120 条消息🔥🔥): 

> `Freeplay.ai 反馈, Absolute Zero Reasoner (AZR), AI Agent 框架, OpenAI Codex 实验, Perplexity 免费层级成本` 


- **Freeplay.ai 引起关注与共鸣**：成员们讨论了 [freeplay.ai](https://freeplay.ai)，一位用户在与其通话后对其产品架构和方向给出了积极的初步评价，并指出它与内部构建的系统非常相似（*mirrors*）。
   - 另一位用户对更新表示了浓厚兴趣，并询问了 **Freeplay.ai**、**Braintrust** 和 **Humanloop** 之间的显著差异。
- **Absolute Zero Reasoner 自我学习**：一位成员分享了一个 [YouTube Short](https://youtube.com/shorts/avnHiKcOEQA?si=NWoqwZR1IcPxyrG0) 视频，详细解析了 **Absolute Zero Reasoner (AZR)**，这是一种无需人类数据、从零开始自我学习的新型 AI。
   - 该成员征求了对该视频的反馈和看法。
- **AI 开发者渴望 AI 开发者调查**：一位成员建议开展一项“State of AI”开发者调查，以追踪 **AI agent 框架**、**SDKs**、**proxies** 和 **models** 等领域的应用趋势。
   - 讨论强调了在考虑新的代码审查流程和组织架构后，有必要了解企业环境中的软件生产力提升情况。成员们分享了 [2025 State of AI Engineering 调查链接](https://x.com/barrnanas/status/1904593314533564604)和 [Allstacks 调查](https://www.surveymonkey.com/r/6WQD6QQ)。
- **OpenAI Codex 挑战 Shopify 应用创建**：一位成员尝试使用 **OpenAI Codex** 将现有应用程序 Answer HQ 转换为兼容的 **Shopify App Store** 应用。
   - 他们注意到 Codex 首先会寻找 *agents.md* 文件，并且一份优秀的 README 有助于简化流程。这与 Claude Code 的 *init* 命令类似，建议生成一个供 LLM 使用的 AGENTS.md 文件，概述项目的领域、关键命令、如何运行测试以及项目规范。
- **Perplexity 免费层级耗资数百万**：一位成员分享了一条推文，指出 [Perplexity 每年在其免费层级上花费 3300 万美元](https://x.com/breadcrumbsre/status/1924474011125227687)。


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: codex 播客 https://x.com/latentspacepod/status/1923532303327953295?s=46
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1373027176563085532)** (142 messages🔥🔥): 

> `Meta's Maverick LLM, Agent as Judge, Economic disaster, Context Switching, Home Rolled context sharing` 


- **Meta 的 Maverick LLM Arena 门控惊喜**：一位成员提到，“如果它能在上行空间给你惊喜，它也能在下行空间让你意外”是 **Meta 的 Maverick LLM** arena 门控迈出的伟大第一步，这让人联想到利用随机性进行定向运动的 [布朗棘轮 (Brownian ratchets)](https://en.wikipedia.org/wiki/Brownian_ratchet)。
   - 随之而来的问题不再是消除变异性，而是建立验证/棘轮/切断下行风险的方法，同时保留上行收益，例如使用 *LLM as judge + retry*。
- **Agent as Judge 减少任务疲劳**：成员们讨论认为，较低的任务疲劳归功于较少的上下文切换（Context Switching）；只要你保持在精神上的“理想地带 (idealand)”而远离“语法空间 (syntaxspace)”，你就没问题。他们注意到一位一直以 **0.1 倍产能**工作的精疲力竭的开发者，在使用 **Agent as Judge** 后恢复到了 **1 倍性能**。
   - 处于倦怠期的开发者现在可以恢复到 1 倍甚至更高的性能，我们又回到了“*Agent as Judge*”的话题。
- **触手可及的自研 Agent 大军**：成员们讨论了拥有一个触手可及的 **Robot Agent 大军**的可能性，强调了像显式 **MCP**（更像是一个自研版本）那样的共享上下文。
   - 一位成员非常喜欢早些时候在频道中出现的 **golang** *100 行代码编排 Agent* 的项目。
- **Kam 的演讲和评估方法**：一位成员询问了关于评估方法/细节以及如何检查输出的经验；另一位成员询问为什么不使用商业框架，一位用户调侃道，在后 Agent 代码生成时代，“非我所创 (not invented here)”的心态确实感觉不一样。
   - Kam 在编辑并涂黑了录音中捕获的一些个人信息后，提供了一个直接下载视频的链接（[Dropbox 链接](https://www.dropbox.com/scl/fi/5l3qq5qf81rgdagilimlt/Latent-Space-AI-In-Action-2025-05-16-Engineering-Reliable-Agents.mp4?rlkey=ma8j2onvhp7kp5qlv27ujheyh&st=3i7x5gue&dl=0)）。
- **Kam 声音的潜在 Deepfake 正在制作中**：在演讲录音出现音频问题后，一位成员开玩笑说可能会使用 **Zonos RVC** 来 Deepfake Kam 的声音，尤其是因为他们错过了一部分演讲。
   - 另一位成员提供了一个 [Loom 录屏](https://www.loom.com/share/beefb650c11e4828924dd762dcaa9f3e?sid=15db8320-8d71-4d22-9ebf-89ab7f66515f) 作为备选方案，并指出由于是在健身房大厅录制的，可能会有背景噪音。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1373013881664049282)** (173 messages🔥🔥): 

> `Codex o4-mini model, Gemini 2.5 vs coding models, Aider settings, Aider's agent-like direction, Model preferences for Aider` 


- **Codex o4-mini 发布遇挫**：**Codex o4-mini** 模型的发布恰逢大型科技公司裁减那些负责受人尊敬且有价值产品的人员，成员们对此并不感到意外。
   - 一位成员调侃说，Zuck 可能会因为整天冲浪让他看起来像个白痴而感到沮丧，而 Musk 带着 Grok 看起来像个天才。
- **Gemini 2.5 Pro：编程模型之忧？**：成员们注意到 **Gemini 2.5 Pro/Flash** 只有 **45%** 的通过率，这引发了对在实践中使用编程模型的担忧，但其他人建议，如果你珍惜时间，还是应该使用它，或者 **o4-mini** 或 **Claude 3.7**。
   - 一位成员表示 *Pro 模型把 diffs 搞得一团糟*，以至于他们正在考虑进行更多实际实验，例如 Flash 虽然不是编程模型，但可以作为某种超廉价的编程模型。
- **Aider 的 Agent 化转型引发担忧**：成员们对 **Aider** 变得越来越像 Agent 的发展方向表示担忧，担心它可能会失去作为结对编程工具的核心身份。
   - 一位成员担心，即使增加了 Agent 类的功能，最终可能会落得个“两头不到岸”的境地。
- **模型热潮：Aider 的偏好显现**：Aider 用户讨论了他们的模型偏好，**Gemini 2.5 Pro** 是一个热门选择，而一些人发现 **OpenAI** 的 **Codex** 与 **Claude Code** 相比令人失望。
   - 一位成员指出 **Gemini 2.5 Pro** 击败了他们尝试过的所有其他模型，另一位成员则使用 **Gemini 2.5 加上 GPT 4.1** 处理简单任务。
- **大型代码库处理：Plandex 加入讨论**：成员们对 **Plandex** 声称能有效处理大型代码库的说法进行了辩论，但有人指出其演示视频中只展示了一个简单的待办事项列表。
   - 一位成员表示，即使他们的代码库不是很大，Aider 的上下文处理对他们来说也一直很好。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1373030163327221831)** (42 条消息🔥): 

> `模型迭代，工作计划文档，基础提示词设置，编辑格式问题，可用 UI 主题` 


- **整理工作计划文档可改进迭代**：一位成员利用在**基础提示词 (base prompt)** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html)) 中设置的**工作计划文档 (workplan document)**，并在过程中不断对其进行整理。
   - 当他们想要进行此类迭代时，通常在 `/ask` 模式下进行，当效果看起来不错时，可以使用 `/code okay, please do that`。
- **自动化开发日志！**：一位成员利用**基础提示词 (base prompt)** ([aider_conf.html](https://aider.chat/docs/config/aider_conf.html))，其中包含如何建立工作文档的指令，在系统地实施更改之前先制定计划。
   - 他们使用这个提示词：*Please use a file for use as a development context log that will be added to the context, named devlog_{datestamp}.md*，用于跟踪代码修正和见解。
- **Aider 编辑格式：不可能的任务**：一位用户在使用 **deepseek-chat** 时遇到了编辑格式问题，当 **LLM 建议多次编辑**时，它应用了第一次编辑，导致后续混乱。
   - 错误原因是：由于 `SearchReplaceNoExactMatch`，*LLM 不符合编辑格式*。
- **Aider 的极简 UI 拥有粉丝**：一位用户想知道 **Aider 是否有可用的 UI 主题**，以及某些元素是否可以自定义。
   - 一位成员回复说有**浅色模式和深色模式**，另一位成员指出可以通过 `--code-theme` 设置 [Pygments](https://pygments.org/styles/)。
- **通过 Aider 生成 PR 描述**：一位用户询问了编写 **Pull Request 描述**的工作流。
   - 一位成员建议在你的分支执行 `/run git diff origin/main`，然后执行 `/ask write a pull request description describing this diff`。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1373039374622523515)** (155 条消息🔥🔥): 

> `MCP 新闻 Agent，应用驱动 vs LLM 驱动，MCP 客户端工具选择，流式 HTTP crawl4ai MCP 服务器，MCP vs OpenAPI` 


- ****一键式 MCP 新闻 Agent 问世****：一位用户发布了一个新的 Agent，可以一键抓取并总结过去 24 小时内的所有 MCP 新闻，链接见[此处](https://www.jenova.ai/app/tr45cl-mcp-news-summarizer)。
- ****资源：应用驱动 vs. LLM 驱动的辩论愈演愈烈****：一位成员表示，资源是**应用驱动 (application-driven)** 而不是像工具那样由 **LLM 驱动 (LLM-driven)**，这具有局限性。
   - 其他人反驳说，资源非常强大，应该由应用驱动而非模型驱动，从而实现诸如在 MeiliSearch 中建立索引或进行实时 RAG 等酷炫功能。
- ****客户端对工具的控制****：一位成员询问 **MCP Client** 是否可以有选择地从服务器中挑选工具，并将其传递给模型的工具签名。
   - 另一位成员澄清说，**Claude Desktop** 允许你在 *Settings* 中开启或关闭工具，工具剥离是由客户端实现的。
- ****Stainless SDK 遭到抨击****：成员们抱怨说，它们生成的 **pydantic 模型**实际上并没有正确验证数据，导致生态系统碎片化。
   - 他们表示 OpenAPI 文档实际上与他们的 API 行为不符。
- ****Shell 命令 vs. JSON 工具调用****：一位成员建议编程 Agent 需要沙箱以便在其中自由活动，并认为一切都应该使用现有的 **OAuth2** 在浏览器中运行。
   - 另一位成员反驳说，如果一个模型不能根据详尽的文档可靠地生成 Shell 命令，那么它可能也无法可靠地理解 **JSON 工具调用 (JSON tool calls)** 所需的意图——两者都需要相同水平的理解能力。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1373039474946211870)** (25 条消息🔥): 

> `MCP UI SDK 发布，MCPControl server 更新，MCP SuperAssistant 浏览器扩展，通过 MCP 实现 Google Chat + LLM Agents，Sherlog Canvas (Alpha) 发布` 


- ****MCP UI SDK** 开启 **Web 交互**新途径**：发布了一个新的 [SDK](https://x.com/idosal1/status/1923477857881190718) 用于为 MCP 添加 UI，提供了一个直接在协议之上实验丰富 Web 交互的平台。
   - 它使任何 MCP server 能够通过 *ui://* 或 *ui-app://* URI 返回 **Embedded Resource**，并通过事件处理后续交互。
- ****MCPControl Server v0.2.0** 新增 **SSE 支持****：MCPControl server 发布了 0.2.0 版本，引入了 [SSE 支持](https://github.com/Cheffromspace/MCPControl/releases/tag/v0.2.0)，使得在 VM 中运行并为 Claude 提供独立的 Windows 电脑成为可能。
   - 此更新为 MCPControl 提供了更灵活的部署选项，增强了其与 **Claude** 等 AI agents 的集成。
- ****MCP SuperAssistant** 将 MCP 与 **多个 AI 平台**集成**：MCP SuperAssistant 浏览器扩展将 MCP 功能带到了 **Grok, ChatGPT, Perplexity 和 Gemini AI Studio** 等平台，无需 API 配置。
   - 用户现在可以通过该 [扩展](https://mcpsuperassistant.ai) 在浏览器中原生体验 MCP，[YouTube](https://youtube.com/playlist?list=PLOK1DBnkeaJFzxC4M-z7TU7_j04SShX_w&si=3_piTimdBJN7Ia4M) 上提供了演示视频。
- ****Google Chat** 通过 MCP 获得 **LLM Agent** 连接能力**：发布了一个新项目，使用 Model Control Protocol (MCP) 将 Google Chat 连接到 LLM agents（如 Cursor），允许 agents 直接在 Chat 空间中发送消息、搜索对话和总结线程，代码已在 [GitHub](https://github.com/siva010928/google-chat-mcp-server) 开源。
   - 该工具将发送消息、搜索对话和总结线程等功能公开为 MCP tools，并通过 **OAuth 2.0** 进行安全保护。
- ****Sherlog Canvas (Alpha)**：AI 驱动的 **调试界面** 亮相**：Sherlog Canvas (Alpha) 已开源，这是一个类似于 Jupyter Notebook 的 AI 驱动事件调试界面，整合了 [用于日志、指标、SQL 等的 MCP 驱动单元格](https://github.com/GetSherlog/Canvas)。
   - 它提供多 agent 工作流，并允许 AI 生成、运行和优化单元格，辅助事件和 bug 调查，[演示视频](https://youtu.be/80c5J3zAZ5c) 中有进一步展示。


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1374118249809186846)** (1 条消息): 

> `NotebookLM, Mobile App, I/O, MVP` 


- **NotebookLM 移动端 App 上线**：根据 [Google 博客](https://blog.google/technology/ai/notebooklm-app/)，**NotebookLM 移动端 App** 已正式发布，包含 **MVP 功能集**，并邀请用户提供反馈和功能需求。
- **鼓励用户提交新 App 的反馈**：鼓励用户就新的 NotebookLM 移动端 App 提交反馈、功能需求等。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1373333350915506288)** (28 条消息🔥): 

> `NotebookLM 用于奥赛准备, NotebookLM 无法上传材料, NotebookLM 中的自定义指令, NotebookLM 作为语言编辑器, Senpai 的含义` 


- **NotebookLM 的奥赛用例仍待发掘**：一位用户询问了准备**奥赛（Olympiads）**的 11 年级学生使用 **NotebookLM** 的实际用例，以及它与 **Gemini** 或 **ChatGPT** 的区别。
   - 然而，目前还没有针对其在**奥赛**背景下使用的具体建议。
- **NotebookLM 拒绝用户的研究材料**：一位用户对 **NotebookLM** 无法上传与社会正义、生态学和女权主义相关的研究材料表示失望，并引用了来自 [The Conversation](https://theconversation.com/meet-the-forgotten-enslaved-and-working-class-labourers-behind-british-exploration-in-africa-asia-and-antarctica-252771) 的例子。
   - 该用户怀疑原因是*审查和反觉醒（anti-woke）内容*，并提供了显示材料被拒绝的截图。
- **通过 "Guide.md" 文件自定义 NotebookLM**：一位用户通过在每个新笔记本中共享一个包含指令的 `guide.md` 文件，提升了 **NotebookLM** 的生产力。
   - `guide.md` 文件为 **NotebookLM** 分配了不同的角色（Persona），如 *senpai*（专业的资深学长）或 *brain*（天才 AI），以处理不同类型的请求，实现定制化交互，如[此 Youtube 视频](https://youtu.be/ZCPcBgJ54NY?si=9gHljywup_mO0cAM)所示。
- **语言编辑器角色与 NotebookLM 完美契合**：一位用户发现将 **NotebookLM** 与关于写作、媒体、传播和语法的开源教科书结合使用效果极佳，强调了其 AI 设计使其天然适合担任语言编辑器或老师。
   - 该用户创建了最初为 Gemini 设计的 MD 文件作为自定义指令，并集成了 AI 角色协议，如 **ScribeMaster AI**、**NSET Agent** 和 **LSTA Agent**。
- **NotebookLM 可作为语义词汇表**：一位用户建议 **NotebookLM** 专注于信息检索，从而减少了幻觉（hallucination）的机会。
   - 他们补充说，它最适合作为源材料的语义词汇表（Glossary）、索引（Index）或目录（Table of Contents）。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1373019022546964521)** (119 条消息🔥🔥): 

> `音频生成问题, 思维导图转换为 Markdown, NotebookLM 中的格式识别, 辩论播客创建, NotebookLM API 可用性` 


- **部分用户的 NotebookLM 音频生成受限**：用户报告称 NotebookLM 的音频生成仅处理长文档的引言部分，即使是 **114 页的 PDF** 也会生成很短的音频文件。
   - 目前尚不清楚这种限制是由于免费版本还是其他因素造成的。
- **思维导图转换被证明很棘手**：用户正在寻求将 NotebookLM 生成的思维导图转换为 Markdown 格式的方法，但提示词（prompting）尝试未能准确复制原生内容。
   - 建议包括使用 **Mermaid** 或 **PlantUML**。
- **NotebookLM 的写作风格退化**：一位用户注意到 *NotebookLM 的写作风格在过去几天变得异常简单和还原*，类似于**高中作文水平**。
   - 其他人讨论了准备奥赛的学生可能的用途，强调**保密性**是其区别于 Gemini 或 ChatGPT 的关键点。
- **NotebookLM 移动端应用发布但功能缺失**：NotebookLM 应用已正式在 [iOS](https://apps.apple.com/us/app/google-notebooklm/id6737527615) 和 [Android](https://link-to-android-store) 上线，但初步反馈显示它缺少核心功能，如**思维导图**、**简报（briefing notes）**以及“发现来源（Discover Sources）”的能力。
   - 用户还报告了无法为**音频概览（audio overviews）**选择来源等问题，以及与网页版相比的功能差异，一些人认为该应用感觉就像是*打包在本地容器中的 Web 应用*。
- **NotebookLM 新增视频上传功能**：用户发现 NotebookLM 现在支持带有自动转录功能的**视频上传**。
   - 这一功能得到了分享 [**NotebookLM 视频概览**相关链接](https://x.com/testingcatalog/status/1924246774346104863)的用户的证实。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1373060662611411056)** (101 条消息🔥🔥): 

> `AWQ INT4 计算, Gemini 生成注释, 开源 AI 对抗大科技垄断, Google 的 AI 代码 Agent Jules, Hermes 模型` 


- **Gemini 的代码注释怪癖**：成员们讨论了 **Gemini** 生成被注释掉的代码并意外删除其他注释的奇怪行为。
   - 一位成员幽默地指出，**OpenAI** 的发展路径像是一部适合 **Netflix** 改编的希腊史诗，暗示其中充满了出人意料的转折。
- **Google 的 Jules 进入 AI 代码 Agent 领域**：**Google** 将在 **Google I/O** 上发布其代码 Agent **Jules**，这在社区内引发了兴奋和期待。
   - 一位用户提到了 **Jules** 的多模态能力和每日摘要功能，并分享了一个[链接](https://jules.google/)。
- **AWQ INT4 计算深度探讨**：一位用户询问 **AWQ** 是原生使用 **INT4** 还是在 **BF16** 中进行计算（特别是在 **VLLM** 中），另一位成员澄清说 **AWQ** 是 **W4A16** 格式，且 GPU 缺乏用于混合精度 **INT4xBF16** matmul 的电路。
   - 他们建议 **QuaRot** 才是该用户在寻找的技术，并提到 **FP8** 应该也可以。
- **去中心化开源 AI 对抗垄断**：一位成员慷慨激昂地辩论道，**去中心化开源 AI** 是防止大科技公司对 AI 进行垄断的唯一可行途径。
   - 反方观点建议通过政府监管，例如强制开源所有 AI 模型及其数据，以及制定反对审查的法律作为替代方案，引发了关于每种方法的可行性和可能性的辩论。
- **Hermes 模型讨论**：一位用户询问目前最适合使用的 **Hermes 模型** 是哪些。
   - 其他用户推荐了 **Hermes 405b**、**Deephermes 24B** 和 **8B**。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1373166804347650099)** (9 条消息🔥): 

> `自主机器人 LLM 编辑, Fine-tuning vs RL vs Control Vectors, Nous Hermes 模型` 


- **LLM 新手寻求编辑指导**：一位成员表示希望为自主机器人编辑 LLM，承认对该过程不了解并寻求帮助。
   - 一位资深成员建议 **fine-tuning**、**RL** 或使用 **control vectors** 可能比较合适，同时也询问该成员之前是否有制作自定义 LLM 的经验。
- **Nous Hermes 模型发布**：在被问及是否制作过自定义 LLM 后，一位资深成员链接到了 [Nous Hermes 模型](https://nousresearch.com/hermes3/)，暗示他们就是作者。
   - 另一位成员指出这是一个 *mic drop moment*（惊艳全场的时刻）。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1373195380933525645)** (3 条消息): 

> `LLM 惯例, LLM 集体偏见, LLM 对抗性 Agent` 


- **LLM 自发收敛于惯例**：根据一项研究（[https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) 和 [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)），在去中心化的 LLM 群体中，通过局部交互会自发产生普遍采用的社会 **惯例（conventions）**。
- **LLM 产生集体偏见**：研究表明，在去中心化的 LLM 交互过程中，即使单个 Agent 最初没有表现出偏见，也可能产生强烈的 **集体偏见**（参考 [https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) 和 [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)）。
- **对抗性 LLM Agent 推动社会变革**：一项研究（[https://arxiv.org/abs/2505.10475](https://arxiv.org/abs/2505.10475) 和 [https://www.science.org/doi/epdf/10.1126/sciadv.adu9368](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)）显示，一旦达到临界阈值，坚定的少数派 **对抗性 LLM Agent** 群体可以通过强加替代惯例来推动社会变革。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1373525722387189831)** (2 messages): 

> `Gemini, MoE, Long Context Window, Sub-global attention blocks` 


- **假设 Gemini 的 MoE 架构**：一位成员假设 **Gemini** 使用了一种长上下文 **MoE 架构**，并将其称为 *Ensemble of Expert (EoE)* 或 *Mesh of Expert (MeoE)*，具有一个公共/共享的长（**1-10M**）上下文窗口。
   - 该成员的 [X 帖子](https://x.com/ditpoo/status/1923966380854157434) 建议这种架构将共享上下文作为独立的（微型）上下文分片使用，类似于 *Sub-global attention blocks*（子全局注意力块）。
- **测试 Sub-Global Attention Blocks**：一位成员想要测试 **sub-global attention blocks** 或 *sub-context experts* 是否可以某种程度上独立运行，然后扩展成更大的全局注意力范式，以处理极长的上下文。
   - 他们表示这 *需要一些工程手段来实现*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1373195380933525645)** (3 messages): 

> `Decentralized LLM Populations, Emergence of Social Conventions, Collective Biases in LLMs, Adversarial LLM Agents` 


- **去中心化 LLM 中产生社会公约**：一项新研究（[arxiv 链接](https://arxiv.org/abs/2505.10475)，[science 链接](https://www.science.org/doi/epdf/10.1126/sciadv.adu9368)，以及 [HF 博客链接](https://huggingface.co/blog/codelion/ptsreal.azure)）表明，**普遍采用的社会公约**可以通过局部交互在去中心化的 LLM 群体中自发产生。
   - 研究人员强调，即使没有个体偏见，在这些交互过程中也会形成强大的集体偏见。
- **少数派 LLM Agent 推动社会变革**：研究还揭示了由 **对抗性 LLM Agent** 组成的坚定少数群体可以推动社会变革。
   - 一旦达到临界阈值，这些 Agent 就会强加替代公约，从而可能重塑 LLM 群体的整体行为。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1373094388153978900)** (38 messages🔥): 

> `CNN diagrams, matplotlib for diagrams, DiagramVIS-for-computervis, Gemini 2.5 Pro, geometric deep learning` 


- **使用 Matplotlib 解决 CNN 图表难题**：一位成员寻求一种创建带有 **skip connections**（跳跃连接）的 **CNN 图表**的工具，最终使用了 *matplotlib*。虽然过程很痛苦，但在 GitHub Copilot 的帮助下完成了。
   - 生成的脚本现在可以在 [GitHub 仓库](https://github.com/dotrdp/DiagramVIS-for-computervis) 的 *dotrdp/DiagramVIS-for-computervis* 下找到。
- **Gemini 2.5 Pro 用于物理任务**：一位成员提到他们经常将 **Gemini 2.5 Pro 与 Canvas 或 Cursor** 结合用于物理相关的任务。
   - 他们还表示有兴趣尝试 *Windsurf, Aider, Cline, Roo* 和 *Codex* 等工具。
- **几何深度学习（Geometric Deep Learning）受到关注**：一位成员分享了一个 [链接](https://fxtwitter.com/meowdib/status/1922315466401308965)，询问是否有人是 **geometric deep learning** 的粉丝。
   - 另一位成员表示：“还行，但如果需要对称性，我更倾向于进行数据增强。”
- **寻求 MLOps 课程**：一位成员询问是否有推荐的学习 **MLOps** 的课程。
   - 聊天中没有推荐具体的课程。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1373083473245503520)** (29 messages🔥): 

> `AlphaEvolve Whitepaper, Physics of Language Models Discussion, Ecology of LLMs and Social Conventions, Loss Clamping` 


- **AlphaEvolve 论文获得好评**：成员们对审阅 **AlphaEvolve** 白皮书 ([AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)) 表现出浓厚兴趣，这是一个基于 **Gemini** 的编程 Agent，用于设计高级算法。
   - 然而，该讨论被取消了，因为*它无法在 Open Machine Learning 频道正在进行的讨论基础上提供更多改进*。
- **Language Models 物理学续篇**：小组讨论了 **"Physics of Language Models: Part 3.1, Knowledge Storage and Extraction"** ([论文](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250633), [博客](https://physics.allen-zhu.com/part-3-knowledge/part-3-1), [YouTube](https://www.youtube.com/watch?v=YSHzKmEianc))。
- **LLM 生态学显示社会惯例可以产生**：小组讨论了 **"Emergent social conventions and collective bias in LLM populations"** ([science.org](https://www.science.org/doi/10.1126/sciadv.adu9368))，探讨了 AI Agent 如何自主发展出社会惯例。
   - 论文摘要强调了*在去中心化的大语言模型 (LLM) Agent 群体中，自发涌现出被普遍采用的社会惯例*。
- **详细探讨 Loss Clamping**：一位用户询问关于将 loss 截断（clamp）到某个阈值范围的问题：`loss = loss.clamp(l_avg-threshold,l_avg+threshold)`
   - 另一位用户表示 *这不会改变任何事情，你还不如直接降低学习率*，随后又表示他们*说错了，这样做效果要差得多*。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1373025166547882024)** (34 messages🔥): 

> `Leadership issues, Open Source AI, Sam Altman Strategies, Attention seeking transformers` 


- **员工流失暗示领导层问题**：大规模的员工离职暗示了公司使命、管理或有毒职场环境的问题，并引用了 **Sam Altman** 和 **Mark Zuckerberg** 作为例子。
   - 一位成员建议，确定根本原因需要询问那些离职的人，尽管另一位成员表示他们*可以想象为什么会有人想在 Meta 工作*。
- **开源 AI 策略**：成员们辩论了开源 AI 研究是真正的战略，还是获取免费劳动力和资源的一种手段，特别是在 **Meta** 的案例中。
   - 一位成员认为，Meta 的 AI 研究旨在*使你的互补品商品化 (commoditize your complements)*，而不是 Facebook 产品的核心。
- **Altman 的收割策略**：一位成员声称 **Sam Altman** 正在收割各种资源，并提到了 **Codex-CLI** 和免费版 **ChatGPT**。
   - 他们链接了一个 [YouTube 视频](https://www.youtube.com/watch?v=Y8Tj9kq4iWY)，未作进一步评论。
- **Transformer 渴望 Attention**：一位成员讽刺地评论了 Transformer 寻求 Attention 的本质（双关语），链接了一个 [YouTube 视频](https://www.youtube.com/watch?v=T8Ty99O4m0w)，未作进一步评论。
   - 另外还发布了两个 YouTube 链接：[链接 1](https://www.youtube.com/watch?v=lrM5KlNtC3c) 和 [链接 2](https://www.youtube.com/watch?v=RH4hAgvYSzg)。


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1374060602640695337)** (1 messages): 

> `LlamaIndex office hours, LlamaIndex Agents with Long-Term and Short-Term Memory, Multi-Agent Workflow with Weaviate QueryAgent, LlamaExtract for Structured Data` 


- ****LlamaIndex** 本周四 AMA**：首届 **LlamaIndex** 答疑时间（office hours）将于本周四 **PT 时间上午 8 点/CET 时间下午 5 点**在常规语音频道举行，持续 **1 小时**，可以咨询关于 **LlamaIndex** 的任何问题。
   - 演讲者还将演示如何构建 Agent 工作流。
- ****LlamaIndex Agents** 现已具备**长期和短期记忆****：关于 **LlamaIndex Agents** 及其改进的**长期和短期记忆**的更新已在 [LlamaIndex 博客](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents)上发布。
- ****Weaviate** 助力多 Agent 工作流**：使用 **Weaviate** `QueryAgent` 的多 Agent 工作流指南现已在 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/agent/multi_agent_workflow_with_weaviate_queryagent)中提供。
- ****LlamaExtract** 提取带引用的结构化数据**：通过[此 YouTube 视频](https://youtu.be/01kM7tXRHi4)学习如何使用 **LlamaExtract** 提取带有引用和推理过程的结构化数据。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1373070278703321249)** (4 条消息): 

> `LlamaParse 更新，Azure AI Foundry Agent Service，LlamaIndex Discord 答疑时间` 


- **LlamaParse 变得更简单、更高效**：根据 [这条推文](https://twitter.com/llama_index/status/1923510823164706925)，LlamaIndex 宣布了 **LlamaParse** 的重大更新，其特点是*精简的界面*、全新的 **Code Snippet** 按钮，以及即将推出的更多用例预设。
- **Azure AI Foundry Agent Service 增加 LlamaIndex 支持**：如 [这条推文](https://twitter.com/llama_index/status/1924502129974411504) 所强调，现已正式发布 (GA) 的 **Azure AI Foundry Agent Service** 提供了对 **LlamaIndex** 的原生支持，使企业客户能够构建客户支持助手和流程自动化机器人。
- **LlamaIndex 举办首次 Discord 答疑时间**：**LlamaIndex** 团队将于本周四举办首次 Discord 答疑时间 (Office Hours)，内容包括事件驱动的 Agent 工作流演示和现场编码环节；更多信息见 [这条推文](https://twitter.com/llama_index/status/1924527932258845178)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1373055558726189107)** (58 条消息🔥🔥): 

> `COBOL 代码切分，Claude 桌面端文件拖放，Anthropic 的 AgentWorkflow 流式传输，LlamaIndex 与 Ollama 集成，基于数据库的 Agent 状态持久化` 


- ****Chonkie** 处理 **COBOL** 块**：一位用户询问是否有用于将 **COBOL** 代码切分为逻辑块的 Python 包，并被推荐使用 [Chonkie.ai 的 `CodeChunker`](https://chonkie.ai)，据称该工具支持 **COBOL**。
   - 该用户还指出 LlamaIndex 的代码切分器目前不支持 **COBOL**。
- **AgentWorkflow 在 Anthropic 流式传输上遇到困难**：一位用户报告称 `AgentWorkflow.from_tools_or_functions` 不支持 Anthropic 思考模式 (thinking mode) 的流式传输，仅提供最终响应，并提供了一个 [代码片段](https://cdn.discordapp.com/attachments/1373578080853168128/1373659791200620554/test.py?ex=682c895f&is=682b37df&hm=b4fe5af0fef1a812017006a9fbbb11c6d04ec2e1a4f6da81f06a2c7dc05ad3c3&)。
   - 一位社区成员建议使用 `agent._run_step()` 逐步运行 `AgentWorkflow` 以捕获 `AgentStream` 事件。
- **上下文是 Agent 状态持久化的关键**：一位用户询问在将 Agent 状态保存到数据库时，如何管理随着消息增加而产生的上下文窗口限制。
   - 一位社区成员建议限制从数据库获取的实例数量，将上下文保存到数据库，并传递整个上下文工作流对象，而不是传统的用户/助手消息历史记录。
- **LlamaIndex Ollama 运行更清晰**：一位用户在使用 LlamaIndex 配合 Ollama 时遇到了 `ValueError: \"ChatResponse\" object has no field \"usage\"` 错误，尽管参考了 [官方文档](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/)。
   - 该问题通过将 Python 升级到 **3.10**、创建新环境并升级 llama-index 和 ollama 软件包得到了解决。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1373056187863535717)** (45 messages🔥): 

> `NDBuffer deprecation and alternatives, ArcPointer and Atomic structs, Importing mojo code in notebooks, LSP issues and workarounds, Documentation issues in GPU basics tutorial` 


- ****NDBuffer** 的落幕：向 **LayoutTensor** 迁移**：**NDBuffer** 正在被弃用，取而代之的是 **LayoutTensor**，目前正在进行 kernel 迁移的活跃开发工作。
   - 建议用户由于 **NDBuffer** 即将弃用而推迟使用它。
- ****ArcPointer** 面临 **Atomic** 障碍**：一位用户询问如何创建指向包含 **Atomic** 的结构体的 **ArcPointer**，但在 **Movable** trait 上遇到了问题。
   - 有人建议使用 `ArcPointer[OwnedPointer[T]]`，然而 **OwnedPointer** 也没有实现 **Movable**，这个变通方法并未如预期般奏效。
- ****Mojo Notebook** 导入：寻求指导**：一位用户寻求关于如何从与 **notebook** 位于同一文件夹下的 **Mojo package** 或文件中导入代码的指导。
   - 消息中未提供解决方案。
- ****LSP Server**：占用 CPU 并崩溃**：用户报告了 **LSP server** 的各种问题，包括高 CPU 占用（8-16 线程）和频繁崩溃，特别是在旧系统或使用 Docker 时。
   - 解决方法包括重启 **LSP server** 或降级到之前的 nightly 版本，但这些并非普遍有效，一位用户最终使用了 `killall mojo-lsp-server`。
- ****GPU** 基础教程：文档需要完善**：一位用户指出了 [GPU 基础教程](https://docs.modular.com/mojo/manual/gpu/basics/) 中的两个细微文档问题：仓库布局已更改，且 **DeviceContext** 的创建应当包裹在 try/except 块中。
   - 该用户提供了代码片段来演示这些问题并提出了修正建议。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1374056104044134491)** (8 messages🔥): 

> `Mojo kernel registration in PyTorch, Max vs. Fireworks.ai, Together.ai, and Groq.com for serving LLMs, register_custom_ops removal` 


- **Mojo Kernel 注册遇到波折 🛤️**：一位用户报告在更新到最新 nightly 版本后，在 PyTorch 中注册 Mojo kernel 出现问题，注意到 `max.torch` 中的 `register_custom_ops` 函数已被移除，并提供了一个[过时的示例](https://github.com/modular/modular/blob/main/examples/custom_ops/whisper.py)链接。
   - Modular 团队成员确认了在 nightly 版本中将 Mojo 自定义算子注册到 PyTorch 的工作正在进行中，并提醒未来几天可能会有*摩擦和变动*，但引导他们参考更新后的[文档](https://docs.modular.com/max/api/python/torch/)。
- **Max 直面 AI 推理竞争对手 ⚔️**：一位用户正试图说服老板使用 Max 来提供 LLM 服务，但对其相对于 **Fireworks.ai**、**Together.ai** 和 **Groq.com** 等平台的性能表示疑问，这些平台声称在速度和延迟上有显著提升。
   - 该用户引用了 Mojo GPU 与 NVIDIA 的 **Cublas** 以及 AMD 的 **Rocblas/HipBlaslt** 的对比（[YouTube 链接](https://www.youtube.com/live/yOMflrCRya0?si=w9QCDUFvOFG4y7EQ&t=5842)），寻求关于 Max 当前性能以及相对于这些 AI 推理提供商的未来优化计划的信息。
- **`register_custom_ops` 宣告终结 💀**：`max.torch` 的 `register_custom_ops` 函数在最新的 nightly 版本中被移除。
   - Modular 团队成员确认了在 nightly 版本中将 Mojo 自定义算子注册到 PyTorch 的工作正在进行中。


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 messages): 

georgehotz：在此处查看 tinygrad 性能提升情况 https://stats.tinygrad.win
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1373013491920797777)** (50 messages🔥): 

> `tinygrad use GCC instead of Clang, porting a model or weights to TinyGrad, tinygrad 1.0 plan, quantize onnx bug, get torch.index_put to work` 


- **在 tinygrad 中使用 **GCC** 替代 **Clang**？**: 有用户询问是否可以在 CPU 目标上使用 **GCC** 代替 **Clang**，特别是在没有 **Clang** 的 **AIX 系统（PPC64 架构）**上。
   - George Hotz 回复称这*不容易*实现，需要为自定义的 elf 加载器添加 **ppc64 的 elf 重定位（elf relocations）**支持，并引用了[此文件](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/support/elf.py#L41)。
- ****ONNX** 简化了向 tinygrad 迁移模型的过程**: 有用户询问如何将 **Qwen3:30-a3b** 等模型迁移到 TinyGrad，是使用自动化工具还是手动迁移。
   - George Hotz 澄清说，如果模型是 **ONNX** 格式，可以使用 `examples/benchmark_onnx.py` 脚本轻松导入。
- **tinygrad 的 API 稳定性受到称赞**: 一位正在编写 AI 应用开发书籍的作者考虑使用 tinygrad，并询问其接口的稳定性，以确保示例代码在 2-3 年内保持可用。
   - George Hotz 保证前端接口*至少在一年内非常稳定*，并表示他们*应该在追求速度之前先完成 1.0 版本*。
- ****WMMA** 指令基准测试工具发布**: 一位用户分享了 [HIPmmapeak](https://github.com/pkourouklidis/HIPmmapeak) 的链接，这是一个用于测量 **7900 XTX** 上 **wmma 指令**最大 **FLOPS** 的工具，类似于 mmapeak。
   - George Hotz 回应道：*“太酷了！如果你想要悬赏金（bounty），可以尝试使用 tinygrad 的基础设施。”*
- ****ROCm 6.4** 不兼容问题显现**: 一位用户报告在处理[这个 PR](https://github.com/tinygrad/tinygrad/pull/10417) 时，由于 **ROCm 6.4 comgr 不兼容**浪费了半天时间。
   - George Hotz 回应称这个问题很烦人，并指出*“他们更改枚举（enums）的方式就像是担心数字会用完一样！”*


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1374066163872563231)** (1 messages): 

> `AI Agent Engineering, LLMs & Foundation Models, Full-Stack & Backend Systems, Automation & Agent Ops, Vector DBs & Memory Storage` 


- **工程师介绍 AI/ML 背景**: 一位拥有超过 **8 年**经验的 AI/ML 工程师介绍了自己，重点介绍了他在医疗保健、智慧城市、电子商务和娱乐等行业构建智能化生产级系统的经验，并展示了其[作品集](https://yangming.vercel.app/)。
- **构建 Agentic 系统的专业知识**: 该工程师擅长使用 **LangGraph**、**AutoGen**、**LlamaIndex**、**Letta** 和 **DSPy** 等现代技术栈构建 Agentic 系统。
   - 他们还拥有使用 **LangSmith**、**Langfuse** 和 **AgentOps** 等 **AI 可观测性工具**的经验，以及使用 **MemGPT**、**LangMem** 和 **zep** 构建增强记忆 Agent 的经验。
- **精通 LLM 和基础模型**: 该工程师精通微调、检索增强生成（**RAG**）、Prompt Engineering，以及使用 **GPT-4o**、**Claude 3**、**Gemini**、**Mixtral**、**LLaMA-3** 和 **Mistral** 等顶尖模型进行混合链式调用（hybrid chaining）。
- **全栈和后端系统经验**: 凭借在 **React**、**Next.js**、**FastAPI**、**Django** 和 **Laravel** 方面的专业知识，他们可以构建可扩展的架构，通过 **vLLM**、**Ollama**、**Fireworks AI** 和 **OpenAI APIs** 提供 LLM 服务。
- **自动化和 Agent Ops 知识**: 该工程师擅长通过 **n8n**、**Make.com**、**Zapier** 和 **GoHighLevel** 进行工作流编排，并能使用云原生方案以及 **E2B** 和 **Modal** 进行沙箱化部署。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1373031794466885722)** (39 messages🔥): 

> `Assert/Suggest 替代方案, VS Code 主题设置, 基于 DSPy 的 AI 编程 Agent, 大规模系统提示词下的 DSPy 延迟, DSPy 3.0 向 Elixir 的移植` 


- ****Suggest 和 Assert 演变为 BestOfN 和 Refine****：从 **DSPy 2.6** 开始，`BestOfN` 和 `Refine` 已成为 `dspy.Suggest` 和 `dspy.Assert` 的替代方案，详见[本教程](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)。
- ****使用 DSPy 的 Cline 编程 Agent 可能更小且更准确****：成员们讨论了使用 DSPy 重新实现类似 **Cline** 的 AI 编程 Agent；有人建议这样做可能会使 Agent 体积更小、更准确，且减少“离谱”的修改，其中 VS Code 粘合层、Memory、Tools 和 Models 都是重要因素。
- ****大提示词导致 DSPy 延迟困扰？缓存配置难题！****：一位成员注意到大型系统提示词（system prompts）导致 DSPy 解析时间过长，并询问如何配置 **litellm** 的 [prompt caching](https://docs.litellm.ai/docs/completion/prompt_caching)，但这导致了错误，因为它回退到了 JSON 格式。
   - 另一位成员使用 DSPy 优化了提示词，然后直接在 API 中使用优化后的提示词，但发现使用 DSPy 响应需要 **8秒，而不使用则只需 2秒**，并怀疑这是否是本地环境问题。
- ****Elixir 社区对 DSPy 3.0 等效移植表现出兴趣****：一位成员询问是否有兴趣将 **DSPy 3.0 1:1 等效移植到 Elixir**，引发了关于该语言扩展模式的讨论。
   - 支持者认为 **Elixir** 的并发模型*非常适合 LLM 编排*，且 [Nx ecosystem](https://github.com/elixir-nx) 最近取得了巨大进步；而其他人则认为由于集成度和成熟度，Python 仍是生态系统的主要驱动力。
- ****批评者质疑 DSPy 的复杂性与收益****：一些在线用户对 **DSPy 的复杂性**和缺乏亮眼结果表示担忧，认为熟悉 Prompt 技巧的人可能会比该库表现更好。
   - 一位成员反驳称，DSPy 的真正价值在于解决 Agentic 应用的扩展问题，特别是那些需求频繁变化的生产级应用，并指出使用 DSPy 构建的公司可能会避免未来代价高昂的公开失败。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1373093360012628061)** (7 messages): 

> `微调 Embedding 模型, embed-v4.0 产生不需要的 Embedding` 


- **微调 Embedding 模型面临障碍**：一位成员询问是否可以微调 Embedding 模型，另一位成员确认 **Embedding 模型无法微调**。
   - 不过，他们指出可以根据特定需求**微调 Reranker**。
- **Embed-v4.0 生成不需要的 Embedding**：一位成员报告 **embed-v4.0** 生成了一些“不需要的” Embedding。
   - 目前未给出解决方案，但已记录为一个已知问题。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1373110586023346287)** (5 messages): 

> `Embed 4 定价, Vision RAG, Chat API 停滞, Agent API 调用` 


- **Vision RAG 背景下的 Embed 4 定价探究**：一位成员询问了在其团队的 **Vision RAG** 实现中 **Embed 4** 的定价。
   - 另一位成员建议嵌入一张图像并查看响应中 usage 部分的 **token count**。
- **Chat API 停滞困扰 Agent 实现**：一位成员报告称，尽管是付费会员，**Chat API** 在其 Agent 执行过程中似乎卡在了一半。
   - 他们提到尝试**顺序运行节点 vs. 并行运行**作为临时解决方案。
- **请求协助调试 Chat API 冻结问题**：一位成员就 Agent 设置中多次 API 调用导致 **Chat API** 卡住的问题寻求帮助。
   - 另一位成员要求提供 **Model、API 版本、任务和代码片段**等详细信息以协助调试。


  

---


### **Cohere ▷ #[💡-projects](https://discord.com/channels/954421988141711382/1218409701339828245/1374148102197481513)** (1 messages): 

> `Vitalops datatune, 开源数据转换` 


- **Vitalops 在 GitHub 发布 datatune**：Vitalops 在 [GitHub](https://github.com/vitalops/datatune) 上发布了一个名为 **datatune** 的新**开源工具**，该工具使用自然语言指令和 LLM 执行数据转换。
- **Datatune 利用 LLM 进行数据转换**：**datatune** 工具利用 LLM 根据简单的**自然语言指令**执行**数据转换**，旨在简化复杂的数据处理操作。


  

---

### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1373093668960866354)** (3 messages): 

> `Game Development, AI/ML Engineering, 3D Game Development, AI-powered NPCs, Skills in Game Engines` 


- **游戏开发者寻求新职位**：一位拥有 8 年经验的 **Game Developer** 兼 **AI/ML Engineer** 在合同结束后正在寻找新工作。
   - 他们擅长开发具有 **AI 驱动的类人 NPC** 的 **3D 游戏**，并拥有包括 Unity、Unreal Engine、Godot 以及 Photon 和 Mirror 等网络工具在内的强大技能组合。
- **技能展示**：该工程师展示了在**多种引擎**中的技能，包括 Unity (**C#**)、UE (**C++/Blueprints**) 和 Godot (**GDScript**)。
   - 他们可以胜任：**战斗系统**、**角色控制**、**背包系统**、**交互系统**、**拟人化 NPC**、**NLP**、**情感状态系统**、**UI/UX**、**HUD**、**背包 UI**、**对话系统**、**响应式用户界面**、**Photon**、**Mirror**。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1373029833235497010)** (13 messages🔥): 

> `Swarm UI, Linus Tech Tips $1M Server for Pi Calculation, Customer Success career advice, Models for text interpretation and formatting` 


- **Swarm UI 相比本地文档更受推崇**：一位成员表示 *本地文档目前还可以*，但他们更倾向于使用 **swarm-ui**，因为它是处理*从简单到中等再到高级*用途的**最佳方式**。
- **Linus 打造百万美元圆周率服务器**：一位成员分享了 Linus Tech Tips 打造一台疯狂的 **100 万美元服务器** *用于计算圆周率位数*的链接，另一位成员回应道 *这不是我第一次看到 Linus 做这种疯狂的事了*。
- **寻求 Customer Success 职位建议**：一位成员询问是否有人从事 **Customer Success** 工作，并寻求转型到科技公司 **CS 职位**的建议，重点介绍了他们在政府、非营利组织、销售和外联方面的背景。
   - 他们强调自己天生适合支持他人、建立关系和解决问题，并具备技术专长和简化复杂概念的能力。
- **将教科书解析为 Markdown 的模型**：一位成员询问有关解释和格式化大量文本的推荐模型，特别是将教科书转换为 **Markdown 笔记**并总结/提取特定信息。
   - 他们寻求关于能够有效处理此类任务的工具的建议。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1374126490228232482)** (1 messages): 

> `AgentX Competition, Submission Forms, Judging Panel, Entrepreneurship Track, Research Track` 


- **AgentX 竞赛表单发布**：**AgentX Competition** 的提交表单现已开放，评审团阵容豪华，包括来自顶级 AI 公司的 VC、创始人、产品负责人和研究员；[创业赛道 (Entrepreneurship Track)](https://forms.gle/FJTC4jd197bNeJJ96) 和 [研究赛道 (Research Track)](https://forms.gle/5dccciawydCZ8o4A8) 的提交链接现已可用。
- **关键提交截止日期临近**：所有提交的截止日期为 **2025 年 5 月 31 日**晚上 11:59 (PT)；团队应为创业赛道准备 Pitch Deck、产品演示视频和在线产品链接，为研究赛道准备论文、视频演示和 GitHub 仓库。
- **15 万美元奖金池预告！**：优胜团队将瓜分超过 **15 万美元** 的奖金，目前距离项目定稿仅剩约 2 周时间；有关提交的问题可以在频道中提问。
- **请求在社交媒体为 AgentX 助力**：鼓励参与者通过转发/引用/点赞/重发 [Twitter](https://x.com/dawnsongtweets/status/1924470174776004875) 和 [LinkedIn](https://www.linkedin.com/posts/dawn-song-51586033_agentx-agentx-activity-7330241621160005633-E8Ii) 上的公告，在社交媒体上宣传 AgentX。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1374107112698941440)** (2 messages): 

> `OpenAI API keys, Alternative approaches to API calls, Trailblazer Tier Certificate` 


- **学生必须自备 OpenAI Keys**：参加 LLM Agents MOOC 的学生需要为实验练习提供自己的 **OpenAI API keys**，因为课程 **不提供这些 key**。
   - 消息澄清说，虽然运行实验需要 key，但在最终提交时可以将其排除。
- **探索避免 API 调用的一系列方法**：一位实验 TA 被标记，以提供关于在实验练习中 **避免直接 API 调用** 的替代方法的见解。
   - 这表明可能存在一些方法或工具，可以最小化或绕过对外部 API 交互的需求。
- **Mastery Tier 学生可以退而求其次选择 Trailblazer**：完成测验和书面文章但在实验中遇到困难的学生，仍然可以申请 **Mastery Tier**，并在必要时可能会被“降级”到 **Trailblazer Tier**。
   - 这确保了通过评估展示知识的学生，即使在实践实验环节面临挑战，其表现仍能获得认可。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1374037141625114825)** (1 messages): 

> `Model Evaluation, Finetuning, Continuous Integration` 


- **讨论模型评估策略**：该消息提出了微调（finetuning）后模型评估的两种方法：发布前的 **ad hoc 评估**，以及在每日构建持续集成（nightly CI）中的 **自动化评估**。
   - 用户指出，**自动化评估** 提供了更高的保真度，但需要更多的精力投入和计算资源。
- **Ad Hoc 评估 vs 自动化评估**：用户正在权衡在 **ad hoc 基础** 上评估模型与为模型评估设置 **自动化 nightly CI** 流水的优缺点。
   - 他们承认 **ad hoc** 方法可靠性较低但更易于实施，而 **自动化 CI** 方法提供了更高的保真度，但需要更多的维护和计算资源。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1373224480289329233)** (1 messages): 

> `Torchtune cfg.get` 


- **搜索中使用的 Torchtune 配置值**：一位用户搜索了 GitHub 代码，以查找在 [pytorch/torchtune 仓库](https://github.com/search?q=repo%3Apytorch%2Ftorchtune%20cfg.get(%22dataset_val%22)&type=code) 中使用配置项 `cfg.get("dataset_val")` 设置验证数据集的位置。
   - 这对于理解 **验证数据集（validation dataset）** 的使用方式以及如何进行自定义可能会很有帮助。
- **Torchtune 验证**：了解验证数据集的工作原理对于评估模型训练非常有用。
   - 探索 `cfg.get` 有助于发现数据集配置是在何处设置的。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1374149004580884631)** (1 messages): 

> `DataTune, Data transformation, Vitalops` 


- **Vitalops 发布 DataTune**：Vitalops 发布了 **DataTune**，这是一个使用自然语言指令和 LLM 进行数据转换的新开源工具，可在 [GitHub](https://github.com/vitalops/datatune) 上获取。
- **调整你的数据**：**DataTune** 使用 LLM 通过简单的指令来转换数据。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1374039122989158541)** (1 messages): 

> `` 


- **AI 工程师简介**：一位在机器学习、深度学习和数据科学领域拥有 **10 年经验** 的 AI 工程师，擅长为实际应用构建、训练和部署 AI 模型。
   - 他们深入了解 **Python, TensorFlow, PyTorch** 以及 **AWS, GCP** 等云平台，并寻求建立联系以构建下一代思考型软件。
- **AI 工程师的技能栈**：该 AI 工程师拥有 **Python, SQL, R**, ML/DL 框架（**TensorFlow, PyTorch**）以及 **Jupyter, Git, Docker, MLflow, Streamlit** 等工具的技能。
   - 他们的技术涵盖监督与非监督学习、深度学习（CNN, RNN, Transformers）、NLP、计算机视觉以及模型部署（APIs, CI/CD）。