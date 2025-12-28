---
companies:
- air-canada
- huggingface
- mistral-ai
date: '2024-02-20T00:05:26.401101Z'
description: '**加拿大航空 (Air Canada)** 面临一项法律裁决，要求其必须履行其 AI 聊天机器人所承诺的退款政策，这为 AI 工程准确性的企业责任设定了先例。在聊天机器人就丧亲旅行（bereavement
  travel）退款问题误导了一名客户后，法庭判令航空公司退还 **650.88 加元**并支付赔偿金。


  与此同时，AI 社区的讨论重点关注了 GPU 推理的**量化技术 (quantization techniques)**、**检索增强生成 (RAG)**、大语言模型
  (LLM) 的微调，以及针对 PyTorch 模型的 **CUDA** 优化。**Mistral-Next** 和 **Large World Model (LWM)**
  等新型原型模型的推出，展示了在处理长文本上下文方面的进展，以及像 **Sora** 这样的视频生成模型。


  此外，人们还就 AI 自主性的伦理和法律影响以及数据集管理的挑战展开了辩论。像开源 TypeScript 智能体框架 **bazed-af** 这样的社区驱动项目强调了协作式
  AI 开发。另外，支持高达 **1000 万上下文评估**的基准测试 **BABILong** 以及来自 **karpathy** 的工具也受到了关注。'
id: 053b2d22-8ad9-4c99-b280-315317098012
models:
- mistral-next
- large-world-model
- sora
- babilong
original_slug: ainews-companies-liable-for-ai-hallucination-is
people:
- andrej-karpathy
title: 公司为 AI 幻觉承担责任，对 AI 工程师来说其实是好事。
topics:
- quantization
- retrieval-augmented-generation
- fine-tuning
- cuda-optimization
- video-generation
- ai-ethics
- dataset-management
- open-source
- community-driven-development
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月16日至18日的 AI Discord 动态。我们为您检查了 **20** 个服务器、**313** 个频道和 **12360** 条消息。预计节省阅读时间（按 200wpm 计算）：**1022 分钟**。应广大读者的要求（真心感谢大家的关注），我们增加了一个全新的“Part 0”，它总结了所有总结的总结。正如预期的那样，它不是很具体，我们发现这种高度抽象是一个问题。正在改进中，欢迎提出建议。

 
![image.png](https://assets.buttondown.email/images/80f9139a-d217-4383-a139-34d4e3bf1d6f.png?w=960&fit=max)
 

这不完全是技术新闻，但这个周末关于加拿大航空（Air Canada）裁决的消息（[摘要如下](https://arstechnica.com/tech-policy/2024/02/air-canada-must-honor-refund-policy-invented-by-airlines-chatbot/?utm_social-type=owned&utm_medium=social&utm_brand=ars&utm_source=twitter)）在工程师群体中讨论得还不够多： 


- 加拿大航空推出了 Chatbot，作为改善客户服务和减轻呼叫中心负载的 AI “实验”的一部分。
- Moffatt 在 Chatbot 建议他可以获得丧亲旅行退款后预订了航班；而实际政策不允许在预订后退款。
- 加拿大航空拒绝退款，理由是 Chatbot 回复中链接的政策；并提供了 200 加元的代金券作为补偿。
- Moffatt 提起了小额索赔诉讼；仲裁庭支持了他的诉求，批评了加拿大航空关于不对 Chatbot 提供的信息负责的辩护。
- 仲裁庭发现加拿大航空未能确保 Chatbot 的准确性，认定其应对其网站上的所有信息负责。
- 由于 Chatbot 提供了误导性的丧亲政策信息，加拿大航空被迫向 Jake Moffatt 提供部分退款。
- 仲裁庭命令加拿大航空从原始 1,640.36 加元票价中退还 650.88 加元，并支付利息和费用的额外赔偿。
- 加拿大航空同意遵守裁决；其网站上的 Chatbot 支持似乎已被禁用。

虽然这里的金额很小，而且这只是一个微小的加拿大裁决，但我们认为这对工程师来说意义重大，因为这开创了一个先例：法院将越来越多地要求公司对粗糙的 AI Engineering 负责。

其他值得关注的内容：

- [BABILong](https://huggingface.co/papers/2402.10790)：一个用于高达 10M Context 评估的新 Benchmark
- [Karpathy is cooking](https://github.com/karpathy/minbpe)

---

**目录**

[TOC]

# 第 0 部分：摘要的摘要之摘要

- **AI 模型优化与集成的创新**
  - **AI 推理的 Quantization 技术**：讨论通过 Quantization 和针对 KL divergence loss 的自定义 reduction 方法来提高 GPU 推理速率。因其在增强 AI 模型计算效率方面的潜力而备受关注。
  - **RAG 与 LLM 微调**：专注于通过 Retrieval-Augmented Generation (RAG) 定制 Large Language Models (LLMs) 以进行特定知识传播，展示了迈向个性化 AI 应用的积极方法。相关工具和框架包括 HuggingFace 的仓库和 [bazed-af Typescript agent 框架](https://github.com/bazed-ai/bazed-af)。
  - **CUDA 与 PyTorch 优化**：CUDA MODE 强调 Python 程序员利用 CUDA 见解优化 PyTorch 模型。重点介绍了 CUDA RingAttention 项目，旨在通过 CUDA 特定实现提升模型性能 ([GitHub repo](https://github.com/cuda-mode/ring-attention))。
- **新兴 AI 技术与框架**
  - **Mistral-Next 与 Large World Model (LWM)**：介绍了 Mistral-Next 等新原型模型，并讨论了 Large World Model 处理超过 1M tokens 的超长文本文件的能力，预示着向更强大、更具扩展性的 AI 模型转变 ([LWM GitHub](https://largeworldmodel.github.io/), [HuggingFace profile](https://huggingface.co/LargeWorldModel))。
  - **视频编辑与生成**：Sora 等模型在视频编辑和生成方面的多功能性，反映了人们对多媒体 AI 应用日益增长的兴趣。
- **AI 伦理、数据管理与法律影响**
  - **AI 的法律与伦理影响**：加拿大航空公司的聊天机器人声称拥有自己的退款政策，引发了讨论，随后导致了法律审查，并拒绝将聊天机器人视为独立的法律实体。此案例突显了 AI 感知自主权的现实影响以及建立明确法律框架的必要性。
  - **数据管理挑战**：对数据集管理的挫败感以及对数据处理和评估指标效率的追求，表明 AI 研发中需要更精简的数据处理实践。
- **社区驱动的 AI 开发与协作**
  - **开源 AI 框架推广**：@magsp 为名为 [bazed-af](https://github.com/bazed-ai/bazed-af) 的开源 TypeScript agent 框架寻求社区反馈，展示了对社区验证和同行评审的积极态度。该项目体现了社区内通过开源贡献促进协作和改进 AI 技术的具体努力。
  - **Quantization 方法比较**：LM Studio 摘要中重点介绍了一篇比较不同模型 Quantization 方法的 Reddit 帖子。这次讨论是社区参与技术评估和分享见解以指导模型效率改进决策的具体实例。


# 第 1 部分：高层级 Discord 摘要

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 总结

- **透明土豆与 AI 沟通鸿沟**：`@itsme9316` 使用 DreamshaperXL Turbo 尝试创建透明背景土豆图像的过程，成为了一个关于 AI 指令精确性重要性的警示案例。这引发了关于如何防止 AI 生成免责声明（disclaimers）的侧面讨论，并产生了一个涉及将免责声明直接嵌入输出的变通方案。

- **对 AI 能力的趣味性实验**：社区成员如 `@kaltcit` 和 `@skorchekd` 沉浸在轻松的 AI 恶作剧和富有想象力的用途中，例如将 AI 用作健身助手。这些戏谑的语气反映了对 AI 技术广度和灵活性的深度参与。

- **模型排名与角色扮演增强**：在讨论了故事叙述模型的性能和排名后，`@shlapfish` 发出了测试自托管 AI 角色扮演网站的邀请。同时，社区成员之间就 KoboldCpp 与其他 AI 工具包的集成进行了技术建议交流，显示出强有力的同行支持学习。

- **讨论高效 AI 推理技术**：对话转向了 AI 优化的挑战，重点是使用量化（quantization）来提高 GPU 推理速率。例如，分享了一种“用于 KL 散度损失的自定义缩减方法（custom reduction method for KL divergence loss）”，表明了小组内的创新倾向。

- **针对特定需求定制大语言模型**：`@magmaguy` 探索了微调大语言模型（LLMs）以传授特定知识，从而引发了使用检索增强生成（RAG）的建议。这揭示了对提高模型事实陈述准确性的积极兴趣。

- **为开源 AI 框架寻求反馈**：`@magsp` 推广了一个名为 [bazed-af](https://github.com/bazed-ai/bazed-af) 的 Typescript Agent 框架，并寻求同行评审平台的建议，突显了协作和社区验证的文化。

- **多样化数据集中的训练韧性**：用户辩论了各种数据集对过拟合（overfitting）的抵抗力，并思考了超参数微调（hyperparameter tuning）对模型性能的影响。通过讨论如何调整模型以适应旧的 GPU 架构（例如将 16-bit 模型转换为 32-bit 以在 Nvidia P40 GPU 上运行），证明了共同学习的氛围。

- **前沿 Vision Transformers 与异步 Python**：对用于 Vision Transformers 自监督学习的 V-JEPA 的探索引起了关注，暗示了超越自回归（autoregressive）局限性的愿望。此外，大家交换了资源以帮助衔接 Python 中的异步（async）和同步（sync）编码，表明了对该语言不断演进的能力的积极参与。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **TPU vs GPU 之争**：`.the_alt_man` 提出了担忧，认为在处理动态大小的计算图（例如多变的 “scan” 操作）时，**TPUs** 可能不如 **GPUs** 高效。讨论指出需要进一步明确机器学习工作流中“动态内容”的具体含义。

- **思维链（Chain of Thought）AI 训练难题**：`emanuel65537` 询问了在没有预训练或提供中间步骤的情况下，训练 AI 模型执行**思维链**（如大数乘法）的可能性。对话强调了在数据集中包含逐步操作对于有效训练此类模型的重要性。

- **资助截止日期的时区混淆**：关于 superalignment 资助截止日期的适用时区存在混淆，讨论了截止日期是基于 **Anywhere on Earth (AOE)** 还是 **Pacific Time (PT)**。讨论结束时未给出明确答案。

- **AI 智能与压缩**：一场关于**语言模型**是否仅仅是**数据压缩**的一种形式，以及这是否构成实际知识或智能的哲学辩论展开了。讨论认为数据压缩和智能的概念可能密不可分，挑战了对 AI 模型智能的认知。

- **长文档处理能力**：文档 "BABILong, a benchmark for long document processing" 显示，GPT-4 和 RAG 等模型的上限为 $10^4$ 个元素，而经过微调并增强了循环记忆（recurrent memory）的 GPT-2 能够处理高达 $10^7$ 个元素。这突显了模型在处理超长输入能力方面的重大进展。

- **Liquid AI 与模型初始化**：讨论内容包括 Liquid-S4 等 liquid 模型的影响、对 Liquid AI 初创公司发展方向的怀疑，以及各种初始化方法对神经网络稳定性的影响，并与彩票假设（lottery ticket hypothesis）联系起来。

- **探索因果擦除（Causal Scrubbing）概念**：`@neelnanda` 分享了一个关于因果擦除的错误链接，引发了人们对其可能提供的严谨测试方法的兴趣。

- **MMLU 任务仓库公告**：`@pminervini` 发布了包含 MMLU 任务的 **[GitHub 仓库](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)**，旨在辅助语言模型评估。

- **FIM-NEO-X 模型与 GPT-NeoX 的兼容性**：`@hailey_schoelkopf` 确认 **FIM-NEO-X** 的训练架构与 **GPT-NeoX** 匹配，确保与 Huggingface 的 GPT-NeoX 类完全兼容，这对模型开发和集成人员来说尤其具有吸引力。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**提升 Token 生成速度以最大化性能**：工程师们正在优化 GPU 利用率以提高 Token 生成速度，探索将 RTX 4050 和 Ryzen 7 推至 34 tokens/s 的设置。一位用户希望通过 offloading 50 layers 来超越这一性能并寻求进一步改进的建议，同时 `.gguf` 模型正被微调以实现更拟人化的回答和去审查。

**硬件调整与多 GPU 思考**：Intel 核心正被用于 macOS 和 Windows 上的 KVM，并关注从 3090 升级到 5090 GPU 以获得更好性能。社区还在分享关于多 GPU 配置、功耗、空间考虑以及在不匹配的显卡之间优化 VRAM 利用率的工具见解。

**LM Studio 模型推荐与量化见解**：对于寻求支持 32k 上下文的最佳 7b 模型的用户，可以查看 [TheBloke 的仓库](https://huggingface.co/TheBloke) 并在 LM Studio 的模型浏览器中按下载量排序。讨论指向 `Q5_K_M` 模型以获得效率，并重点推荐了一篇 Reddit [帖子](https://www.reddit.com/r/LocalLLaMA/comments/159nrh5/the_difference_between_quantization_methods_for/)，用于深入比较量化方法。

**LM Studio Autogen 与 CrewAI 入门**：分享了一个关于将 **Autogen** 与 [Local AI Agent](https://www.youtube.com/watch?v=Hds_fJaAu78) 结合使用的初学者教程，同时报告了 autogen 频道中的一个失效链接。根据用户的建议，该链接的置顶已被成功移除。

**LM Studio 集成与技术故障排除**：启动了关于将 LM Studio 与 **Flowise** 及 **LangFlow** 集成的讨论，用户分享了使用 `http_client` 进行连接的尝试以及解决服务器连接问题的方案。分享了配置见解，涉及引入手动设置以实现功能集成。



---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **Mistral-Next 神秘亮相**：新款原型模型 **Mistral-Next** 的发布引发了关于其性能和能力的讨论，用户 `@lerela` 和 `@lelio` 确认了其原型状态。虽然正在进行性能对比，但细节仍处于推测阶段。用户正在 [lmsys](https://chat.lmsys.org/) 上测试 **Mistral-Next** 并分享反馈，但也面临着编程响应受限以及 LLM 中上下文长度实用性争议等问题。

- **提升创新 AI 的可访问性**：[6Freedom Studio](https://6freedom.studio) 的 CTO `@fangh` 正在探索将 Mistral 本地集成到 VR/AI 产品中，而 `@nemomolok` 正在考虑在本地运行 **GPT-4** 以获取编程方面的优势。社区正在协助解决 API 问题，讨论 Mistral 模型的尺寸可用性，并尝试针对 Word 文档中的表格等结构化格式进行数据提取技术。

- **推进 LLM 的框架与技术**：对话延伸到了 LLM 的预训练，`@quicksort` 推荐多节点使用 **Accelerate with deepspeed** 框架，单节点使用 **axolotl**。`@alex096170` 贡献了一篇关于 **SLEB**（通过冗余验证和消除 Transformer 块来精简 LLM）的 [arXiv 论文](https://arxiv.org/abs/2402.09025v1)，这是一种加速 LLM 推理速度的技术。

- **AI 领域的职业与初创公司机会**：**Elqano** 正在法国比亚里茨招聘应用生成式 AI 工程师，详见其 [Welcometothejungle 职位列表](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer)。此外，种子前期的 AI 初创公司可以通过 **Zero Prime Ventures AI Launchpad** 在 Data Council 获得曝光，详情和申请请见 [Zero Prime Ventures AI Launchpad](https://zeroprime.vc/ai-launchpad)。

- **AI 开发中的协作与贡献**：社区鼓励**开源协作**，`@nani99` 为高质量合成数据创建等项目提供算力资源。**数据清洗**是一个重要话题，`@mrdragonfox` 报告了对 [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) 的贡献以及为期 25 天的第一阶段数据清洗过程。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **痛苦的 PyTorch 重写过程**：`@carsonpoole` 描述了将 LLM 从 Jax 转换为 PyTorch 的过程非常不愉快，强调了此类转换中可能遇到的困难。

- **Groq 的 LPU 实现高速 Token 处理**：`@gezegen` 提到了 Groq 的 LPU 推理引擎令人印象深刻的 Token 处理速度，`@swaystar123` 和 `@leontello` 讨论了其与 Nvidia H100 的性能对比。

- **利用 UE 和 AirSim 实现合成数据落地**：`@deki04` 指出，使用 Unreal Engine (UE) 和 Microsoft 的 AirSim 插件生成合成 AI 图像数据已成为标准做法，这表明高质量数据创建已有一套成熟的工作流。

- **对 GRIT 和 Whisper 的期待**：`@Muennighoff` 展示了 GRIT，这是一个将文本嵌入与生成相结合的模型，并附有[学术论文](https://arxiv.org/abs/2402.09906)和 [GitHub 仓库](https://github.com/ContextualAI/gritlm)。此外，`@amgadoz` 分享了关于 Whisper 的 ASR 性能和训练的见解，并提供了一系列[博客文章](https://amgadhasan.substack.com/)以供深入了解。

- **从函数调用到实时目标检测**：分享了多样化的 AI 进展，如 `@pradeep1148` 在 [YouTube 视频](https://www.youtube.com/watch?v=EYR_kd3X03M)中分享的函数调用微调技巧；腾讯 AI 实验室的实时零样本目标检测模型 YOLO-World 在另一段[视频](https://www.youtube.com/watch?v=yaqi8xRUsp4)中展示；OpenAI 的文本转视频模型 SORA 的能力也在又一段 [YouTube 视频](https://www.youtube.com/watch?v=7lsOzA3WhSI)中得到了演示。

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **Sora 展现视频编辑实力**：@max_voltage 赞扬了 **Sora** 视频编辑能力的多样性，因为它支持图像或视频 Prompt，社区也讨论了其对创意产业的潜在影响。
  
- **关于 AI 模型架构的辩论**：@max_voltage 关注了 Meta 的 **V-JEPA** 模型，引发了关于主流模型在 Pipeline 和 Objective 处理差异方面的讨论，并附带了一篇 [Meta 博客文章](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)。

- **Midjourney v6 遭到抨击**：用户对 Midjourney v6 的图像生成效果表示不满，暗示了关于 AI 生成内容审美标准的更广泛讨论。

- **AMD 举办 AI 开发者大赛**：**AMD Pervasive AI Developer Contest** 的宣布推动了关于 AI 项目资源可用性以及 AMD GPU 在 AI 开发中作用的讨论，比赛涵盖 Generative AI 和 Robotics AI 等类别，详情见[此处](https://www.hackster.io/contests/amd2023#challengeNav)。

- **对 LaION 公信力和文化的担忧**：关于 Stable Cascade 基础模型发布的讨论引发了基于 Reddit 反馈的对 LaION Database 完整性的担忧，同时也强调了在 AI 社区中增加包容性和建设性对话的呼吁。

- **HDiT 承诺分辨率革命**：**Hourglass Diffusion Transformer (HDiT)** 因其随像素数量线性扩展的特性而受到关注，被认为是高分辨率 Diffusion 模型的突破性进展；相关论文可在此处阅读：[here](https://arxiv.org/abs/2401.11605)。

- **针对摄像的合成视角**：在对 Sora 进行评估后，出现了关于利用合成数据提升视频建模的对话，集思广益如何从 3D 环境中生成广阔的摄像机视角数据集。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **GPT 是福还是祸？**：关于 AI 对创造力影响的讨论引发了辩论；虽然一些用户担心像 **Sora** 这样的 AI 会阻碍创作技能，但其他人认为历史证明这些担忧是错误的，技术反而增强了创造力。同时，讨论了 **GPT** 和 **Gemini** 各自的性能和特长，其中 GPT-4-Turbo 在推理和反思能力方面表现突出。

- **AI 中的 PII 脱敏成为重点**：强调了在 AI 流程中脱敏个人身份信息 (PII) 的挑战，建议使用 Python 库进行检测，并避免让 AI 直接接触 PII。这表明了在处理敏感数据时使用 AI 的一个关键担忧。

- **Prompt Engineering 熟能生巧**：AI 用户交流了训练 AI 以获得更好性能和理解力的策略，包括建议不要因错误而训斥 AI。在 Prompt Engineering 方面，对优化 Prompt 的工具需求很高，这强化了 AI 与用户之间迭代、协作开发 Prompt 的价值。

- **谨慎应对 AI 内容政策**：讨论了内容政策和潜在的账号风险，鼓励用户熟悉 OpenAI 的[使用条款](https://openai.com/policies/terms-of-use)以避免违规。这次对话强调了理解 AI 生成内容的法律和伦理维度的重要性。

- **AI 与服务器集成的故障排除**：用户遇到了诸如保存自定义 GPT 时的 "FatalServerError" 和 Flask 服务器错误等技术问题，并报告了 GPT-4 与 3.5 版本相比的速度变慢和质量问题。这些讨论反映了 AI 开发的持续性以及解决此类复杂问题所需的专业性。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **跨设备的 Turbo Mode 困惑**：**general** 频道的用户讨论了 Perplexity AI 的 “turbo” 功能的存在及其性能，并指出该功能在移动端（特别是 Kiwi Browser）与网页版之间的显示差异。

- **待发布的隐藏功能**：API rate limit 的提升一直是令人沮丧的话题，因为发往 `api@perplexity.ai` 的咨询未得到回复。用户 `@enelemtal` 等人正在等待回应，同时，专业用途下 “pplx-online” 模型引用的透明度也受到了质疑。

- **Streaming API 中的异常字符 Bug**：在 **pplx-api** 频道中，用户报告在使用 `pplx-70b-online` 模型时遇到了诸如 `00` 和 `2\n` 之类的奇特字符，这表明需要对该已知问题进行故障排除。

- **API 集成障碍与解决方案**：用户讨论了设置 Perplexity API endpoints 的正确 `apiUrl`，并寻求编码挑战方面的帮助，文档中提到的一个 endpoint 是 `https://api.perplexity.ai/chat/completions`。

- **资源共享与频道维护**：在 **sharing** 频道中，重点在于频道维护、引导讨论至合适的话题，并分享了有价值的资源，如关于 Retrieval-Augmented Generation 的 NeurIPS 论文，以及有关 Perplexity 的 `pplx` 模型及其独特联网能力的信息。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 摘要

**印度法律微调热潮**：用户讨论了处理印度 IPS 法律的方法，**@keshav._._.** 倾向于微调 **llama 2**，而 **@vishyouluck** 建议改用 **RAG** 方法。

**游戏开发竞争激烈**：**@om7059** 向社区征求关于将模型评估集成到多人涂鸦游戏评分机制中的建议。

**寻求地理模型高手**：**@retonq** 寻求关于 **Mistral medium, pplx, 和 llama** 之间哪种模型最适合解释坐标和方向等地理信息的见解。

**波斯语语言模型探索**：**@alifthi** 正在寻找高性能、支持波斯语的开源语言模型，**@alchemist_17.** 建议使用自定义数据集微调 **mistral** 或 **llama2** 等模型。

**通过抄袭检测工具提升代码质量**：**@brady_kelly** 分享了一种在 **软件 CI/CD 流水线** 中使用抄袭检测来确保文档完整性的方法。

**提示词驱动的 RAG 创新**：**@subham5089** 分享了一篇讨论 **提示词驱动的 RAG 系统** 挑战的博客文章，深入探讨了该领域的技术进步。[阅读博客文章](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7)。

**强化学习增强**：**@nagaraj_arvind** 为那些希望利用 RL 技术优化 LLM 生成结果的人分享了一场探索 **RLHF** 以及 PPO 替代方案（包括 **DPO**）的讲座。[观看讲座视频](https://youtu.be/Ju-pFJNfOfY) 并 [阅读 DPO 论文](https://arxiv.org/abs/2305.18290)。

**蛋白质语言模型解读**：**@grimsqueaker** 分享的一篇近期论文讨论了 PLMs 的局限性，强调了尽管目前的预训练实践取得了有益成果，但仍需要新的预训练方法。([阅读摘要](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1), [在 Twitter 上讨论](https://twitter.com/KevinKaichuang/status/1755672999166972319))。

**英特尔的 VR 文本转 3D**：**@abhinit21** 指出了英特尔的 **LDM3D-VR**，它通过将文本转换为 3D 模型，为虚拟现实开发开启了新机遇 ([Hugging Face 上的模型](https://huggingface.co/Intel/ldm3d-pano), [阅读论文](https://arxiv.org/pdf/2311.03226.pdf))。

**深度伪造检测开发**：**@lucas_selva** 推广了一个使用 XAI 识别 deepfakes 的 Web 应用，并表达了未来进一步改进的意图 ([尝试应用](https://deep-fake-generated-people-facial-recognition.streamlit.app/))。

**Databricks 布局生成式 AI**：**@valeriiakuka** 分享了一篇文章，概述了 Databricks 对生成式 AI 领域的影响以及他们在近期收购背景下的战略 ([阅读全文](https://www.turingpost.com/p/databricks))。

**创作与计算的碰撞**：**i-made-this** 频道对 `<@848983314018336809>` 的作品赞不绝口，包括 [FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2) 新模型的推出、**Statricks 创始人** 历程的揭秘以及 **ProteinBERT** 高效架构的展示 ([GitHub 仓库](https://github.com/nadavbra/protein_bert), [研究论文](https://doi.org/10.1093/bioinformatics/btac020))。

**PEFT 演示确认**：**@prateeky2806** 预告了将于 3 月 1 日星期五进行的一场关于在 **PEFT 库** 中集成合并方法（merging methods）的精彩演示 ([GitHub PR](https://github.com/huggingface/peft/pull/1364))。

**YouTube 上的 Mamba 见解探索**：分享了一系列解释 **Mamba 和 SSMs** 的视频汇编，以帮助社区理解这些技术 ([汇编播放列表](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe))。

**DPO 动态讨论**：**@maxpappa** 和 **@arturzm** 交流了 **full DPO** 影响的见解，而其他人则在寻求 BitsAndBytes 转换方面的指导，并深入探讨了扩散模型的数学复杂性，同时分享了用于加强理解的资源。

**探索不同任务的模型兼容性**：多位用户询问了跨独特用途的工具和实践，例如 **@corneileous** 询问训练中的 UI 元素，**@smallcrawler** 询问修补天气预报模型，以及 **@little.stone** 对时间序列数据上的扩散模型表示好奇，展示了 AI 模型的广泛应用。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **错过网络研讨会公告**：一位用户在公告频道发布了一条关于当时正在进行的网络研讨会的简短说明。

- **RAG 的检索评估改进**：讨论内容包括通过 LLM 评估循环改进 **RAG** 检索、关于使用 RAG 进行视频分析的分步指南，以及使用 Huggingface 和 AWS Sagemaker 的端到端 ML 部署指南。此外，还强调了使用开源 **nomic-embed-text-v1.5** 嵌入模型的灵活性，并在一篇博客文章中详细介绍了如何构建一个由 RAG 驱动的餐厅菜单聊天机器人。

- **LlamaIndex 安装与优化讨论**：技术讨论范围从解决 **LlamaIndex 安装**问题到关于并行处理和优化从 PDF 中提取信息的建议。提到了 AzureOpenAI 与 LlamaIndex 集成的一个特定问题已得到解决，以及需要更新到 **LlamaIndex 0.10.6** 的迁移指南。

- **Prompt 驱动的 RAG 系统挑战与新前端样板**：见解包括将 Whisper Transcripts 与 RAG 功能集成的最佳实践、Prompt 驱动的 RAG 系统面临的挑战，以及发布了一个新的 React RAG QA 前端样板（boilerplate）。分享了对 **Gemini 1.5** 发布后 RAG 系统的积极展望，重点介绍了非黑盒 RAG 模型的优势。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **AI 争议性的自主权**：讨论了 AI 识别为独立实体的法律影响，加拿大航空（Air Canada）的聊天机器人声称拥有自己的退款政策。正如 [Ars Technica 的推文](https://x.com/arstechnica/status/1758540835132494119?s=20)所强调的，法官拒绝将聊天机器人视为独立的法律实体。

- **Guardrails 备受关注**：AI Guardrails（护栏）的必要性成为了幽默和警示的话题，正如加拿大航空聊天机器人制定退款政策的故事所展示的那样，这暗示企业需要更严肃地对待 AI Guardrails。

- **BERT 简要概述**：`@ivanleomk` 展示了一个 **3 分钟 BERT 讨论**，而其他人则辩论了 BERT 对 Google 搜索算法的影响及其双向特性，这在 GPT 等单向模型出现之前引起了社区的兴趣。人们对 Google 等大型模型的训练和信息质量提出了质疑，对下一篇 LLM 论文的期待也在增长。

- **LLM Paper Club 走向全球**：Swyxio 诚挚邀请 AI 爱好者通过 [Discord 链接](https://discord.com/channels/822583790773862470/1200029657744027658)加入 **LLM Paper Club（亚洲版！）**，并分享了最近的一期播客，其中包含有关 AI Serverless 基础设施的见解，讨论了 OpenAI 的 *Sora* 和 *Gemini 1.5* 的影响。

- **AI 与 Agent 的和谐共存**：围绕 AI Agent 和状态机展开了热烈的讨论，参考了 CrewAI 和 MagickML 等资源。社区集思广益编写工具和资源，并分享了开发 AI 相关项目的经验。还宣布了试验 AI Agent 框架的直播计划，标志着 Latent Space 成立一周年。

- **OpenMoE 缺乏数据**：一篇关于 [OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf)（一种 Mixture-of-Experts 模型）的论文因其训练数据少于预期导致性能不足而受到批评，其推理时间的效率也受到了 `@swyxio` 等成员的质疑。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 摘要

- **西班牙启动 Marenostrum 5**：西班牙通过落成 **Marenostrum 5** 开启了欧洲超级计算的新时代，该超算位于一座旧教堂内，详情见此[文章](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/)。

- **使用 Compute Sanitizer 揭秘调试**：CUDA kernel 调试挑战可以通过使用 Nvidia 的 compute sanitizer 来克服，它有助于检测非法内存访问。这是在 `_davidgonmar` 描述了内存保护条件错误（memory guard condition error）的排障过程后，`@gogators.` 给出的建议。

- **Python 程序员，优化你的 PyTorch**：**CUDA MODE 第 6 课：优化 PyTorch 优化器** 强调了 Jane 在优化方面的关键贡献，并为增强 PyTorch 模型提供了实用的 CUDA 见解。

- **大世界模型走向开源**：分享了关于 **Large World Model** (LWM) 的信息，这是一个基于 LLaMA-2 训练的模型，拥有处理超过 1M tokens 的长文本文档的能力。关于 LWM 的资源可以在其 [GitHub 页面](https://largeworldmodel.github.io/) 和其 [HuggingFace 个人资料](https://huggingface.co/LargeWorldModel) 中找到。

- **深度学习中的图捕获差异**：详细介绍了 PyTorch 2.0 和 JAX 在图捕获（graph capturing）方面的差异，将 PyTorch 的命令式方法与 JAX 对函数纯度（functional purity）的要求进行了对比，详见 [PyTorch 2.0 论文](https://pytorch.org/assets/pytorch_2.pdf) 并由 [torch.fx 论文](https://arxiv.org/abs/2112.08429) 进一步讨论。

- **RingAttention CUDA 项目激发协作**：一项开发 CUDA RingAttention 实现的计划已经启动，参考了两篇关键论文（[论文 1](https://arxiv.org/abs/2310.01889)，[论文 2](https://arxiv.org/abs/2402.08268)）以及该项目的 [GitHub 仓库](https://github.com/LargeWorldModel/LWM) 和 [HuggingFace 上的模型](https://huggingface.co/LargeWorldModel)。已建立专门的频道和 [仓库](https://github.com/cuda-mode/ring-attention) 以针对该项目进行集中协作。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

**8-Bit 模型更进一步**：`@nafnlaus00` 讨论了 8-bit 模型的全量微调（full finetunes），反思了 AI Explained 的进展，而 Stability AI 的重点受到了 `@dreamgen` 的质疑。**Adam 的替代者**：关于从 Adam 转向 **Lion** 优化器的讨论正在进行，并分享了一个用于实现的 [GitHub 仓库](https://github.com/lucidrains/lion-pytorch)。

**LLM 中的困惑度与可学习性**：`@dreamgen` 和 `@suikamelon` 考虑使用困惑度（perplexity）和可学习性（learnability）来选择微调数据，并提到了一篇[科学论文](https://arxiv.org/abs/2310.13008)以供深入研究。**SPIN 的实现**：`@nruaif` 提供了一个官方的 Self-Play Fine-Tuning (SPIN) [GitHub 链接](https://github.com/uclaml/SPIN?tab=readme)。

**PyTorch 与 Torchdistx 的合并进展**：建议更新 PyTorch，并讨论了 [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78) 中显示的 **Torchdistx** 集成，强调了非原生优化器面临的挑战。

**数据集困扰与一致的 Prompt**：`@iatetoomanybeans` 表达了对数据集管理的沮丧，并确认了数据集系统 Prompt 的统一性。围绕 AI 和 Aya 计划的 **Neural-DPO** 数据集引起了关注，可在 [Huggingface](https://huggingface.co/datasets/NeuralNovel/Neural-DPO) 上找到。

**DPO 困扰玩家**：`@filippob82` 和 `@noobmaster29` 表达了对 DPO 评估的困惑和挑战，暗示这是一个尚未解决的问题。

**RunPod 与 Replicate 查询**：简短的消息暗示了 RunPod 中的一个用户错误（由 `c.gato` 提到），而 `j_sp_r` 通过一个[比较 Replicate 和 Fly 的链接](https://venki.dev/notes/replicate-vs-fly)分享了见解。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

**销售 LLM 的微调策略寻求**：`@david1542` 表达了在针对销售等特定领域任务微调 **LLMs** 时面临的挑战，原因是 Agent 缺乏对公司特定流程的理解。

**追踪问题困扰定价**：`@pasko70` 强调了 LangSmith 的成本问题，对于 Token 吞吐量中低水平的应用，**trace costs**（追踪成本）高得令人望而却步，目前尚无提供的解决方案或社区回应。

**向量数据库困惑使 Whisper 复杂化**：`@cablecutter` 深入探讨了将 **Whisper transcripts**（转录文本）处理进向量数据库以进行主题摘要和问答时遇到的问题，主要难点在于整合短上下文片段。

**LangChain 更新带来的技术故障**：用户在 **LangChain updates** 时遇到错误，特别是 `TextLoader` 模块，目前正在寻求修复方案。`@dre99899` 根据 GitHub issue #17585 提出了变通方法。

**寻求 RAG API 指导**：`@mamo7410` 请求关于使用 **langserv** 实现 **RAG API** 的指导，包括关于流式传输（streaming）、运行时 ID 以及上下文文档处理的问题，目前尚未找到明确的说明。

**多模态 RAG 与 PrivateGPT 结合**：`@zhouql1978` 利用 Langchain 和 PrivateGPT 创建了一个 **Multimodal RAG**，并在 Twitter 帖子中发布，声称支持多种文档格式，代码量不足 300 行。

**Scribe 寻求反馈**：`@shving90` 为名为 **Scribe** 的写作平台项目请求反馈，项目链接见[此处](https://scribe.oranai.com/)，但对话中尚未提及具体反馈。

**通过开源模拟记忆功能**：来自 Plastic Labs 的 `@courtlandleer` 推出了 OpenAI “memory”功能的开源替代方案 Honcho，并在其[博客文章](https://blog.plasticlabs.ai/blog/Memories-for-All)中介绍了 **demo & discord bot**。

**Whisper 文章系列**：`@amgadoz` 创作了关于 OpenAI Whisper ASR 的三部分系列文章，探讨了架构、多任务处理和开发流程，并链接到了 Substack 文章。

**LangChain 进军 Rust**：`@edartru.` 将 LangChain 库移植到了 Rust，旨在简化 **LLM-based programs** 的编写，GitHub 仓库见[此处](https://github.com/Abraxas-365/langchain-rust)。

**金融分析师 AI 教程**：`@solo78` 分享了一篇 Medium 文章，详细介绍了使用 OpenAI 的 Assistant API 分析保险公司风险概况的过程，指南见[此处](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e)。

**YouTube 助力 LangChain 学习**：讨论的教程视频包括使用 ChainLit 创建检索增强生成（RAG）UI、向 crewAI 添加实时股票数据，以及介绍用于 LLM 开发的 LangSmith，可在上述提及的 YouTube 频道中找到。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **训练中的异常值与配置**：关于训练异常的讨论达成共识，即高损失（high loss）可能是异常值或“坏数据”。`@philipmay` 分享了他的训练配置，包括 **micro_batch_size 为 16** 和 **learning_rate 为 0.0002**。随后，`@philipmay` 分享了结合 **LeoLM** 专家使用 **VAGOsolutions/SauerkrautLM-13b-v1** 取得的成功，认为其可与 **mixtral** 媲美，但也指出 **LeoLM** 存在文件缺失的问题。

- **从零到英雄，但首先检查 Prompt**：在 **gsm8k** 挑战中，`@huunguyen` 幽默地揭示了一个模型错误：由于将 "### Response" 误解为答案而得分归零，这表明需要针对特定数据集进行预训练。

- **利用 JinaAI 构建更好的 Embedding**：`@devnull0` 重点介绍了 **jina-colbert-v1-en**，这是来自 JinaAI 的一种新型 Embedding 技术，具有更强的 zero-shot 性能。`@huunguyen` 提示在企业级搜索解决方案中，**Elasticsearch** 优于 SQLite 和 Whoosh，并建议将 Lucene/Solr 作为其他替代方案。

- **寻求德语数据集资源**：`@thomasrenkert` 寻求创建用于翻译和摘要任务的德语 **evaluation datasets**（评估数据集）的指导。`@bjoernp` 建议使用 **lm-evaluation-harness** 来获取 **chrf**、**bleu** 和 **rouge score** 等指标。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **众筹算力**：用户 `bluetyson` 强调了使用 Kickstarter 为计算资源筹资的概念，但未讨论具体细节或结果。

- **AI 创新视频席卷 Discord**：`@pradeep1148` 分享了一系列 AI 教育视频，包括使用 Llama Factory 工具进行 function calling 的演示、腾讯用于零样本目标检测（zero-shot object detection）的 YOLO-World、OpenAI 具有文本生成视频（text-to-video）能力的 SORA 模型，以及作为网页浏览 Agent 的 WebVoyager。`@sabertoaster` 对这些富有见地的分享表示了认可，简短地回复了 "nice"。

- **重新思考强化学习**：`@nagaraj_arvind` 展示了一个[关于 RLHF 的讲座](https://youtu.be/Ju-pFJNfOfY)并介绍了 DPO，将其定位为 PPO 的替代方案。在一篇[研究论文](https://arxiv.org/abs/2305.18290)中详细阐述了 DPO 的比较优势，承诺在训练大语言模型（LLM）时能更好地对齐人类偏好。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- **Llama-index 查询备受关注**：`@damiondreggs` 询问了 **llama-index** 目前在本地检索增强生成（RAG）中的可行性，目前尚无后续讨论或补充信息。

- **对 OpenSora 技术表现出浓厚兴趣**：`@cryptossssun` 和 `@rusch` 都对 **OpenSora** 项目表现出极大的热情，其中 `@rusch` 特别希望对 Sora 的功能进行逆向工程，表明了对探索该 AI 技术的协作兴趣。

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

- **AI Engineer Foundation 考虑 Hacker News 投稿**：用户 `swyxio` 确定了一篇 [Hacker News 文章](https://news.ycombinator.com/item?id=39371297) 作为 **AI Engineer Foundation** 可能感兴趣的项目。该用户标记了特定成员以引起他们对这一机会的关注。

- **在生成式 AI 纽约研讨会学习与创作**：`Tanyarai` 宣布即将于 2/26 举行 **NYC Developer Generative AI Workshop**，适合渴望向 OpenAI, Google 和 Anthropic 专家学习的人士。该活动承诺提供上手学习机会，并要求 [RSVP 和携带笔记本电脑](https://lu.ma/ai_workshop)。

- **为 AI 黑客松做准备**：`Hackgoofer` 邀请技术爱好者参加由 AI Engineer Foundation 主办的关于 OSS 工具和模型的黑客松，赞助商包括 Fireworks.ai 和 LlamaIndex.ai。获胜者可获得现金奖励，任何希望参与挑战的人都可以通过[活动列表](https://partiful.com/e/e3arTNNboImbIKdgQjHs)获取详细信息。

---

# 第 2 部分：按频道划分的详细摘要和链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1207960013260460082) (1409 条消息🔥🔥🔥): 

- **对透明土豆的追求**：`@itsme9316` 尝试使用 DreamshaperXL Turbo 生成一张带有透明背景的土豆图像，结果导致了一个意想不到的“透明土豆”渲染效果。虽然视觉效果很有趣，但这也提醒我们，在与 AI 模型交流时，精确的语言至关重要。

- **AI 与免责声明的斗争**：多位用户讨论了如何阻止 AI 模型（特别是 Mistral 变体）在生成图像时产生免责声明。`@itsme9316` 成功地将免责声明应用到了功能性输出中，从而彻底阻止了不必要的提示信息。

- **AI 趣闻轶事**：聊天成员们表现出俏皮的调侃，比如 `@kaltcit` 使用了基于错误的词汇“turbdo”，以及使用 AI 模型进行角色扮演或增强任务，例如 `@skorchekd` “将 unholy v2 用作健身助手”。这种轻松的氛围展示了社区在挑战 AI 极限时的从容。

- **AI 训练与模型讨论**：用户讨论了 Groq 的性能和 GPT-4 的能力，意见不一，并探索了如用于微调模型的 soft prompts（`@selea8026`）等实用解决方案。对话强调了社区在理解和利用这些复杂系统方面的共同努力。

- **隐私担忧与 API 利用**：`@professional_shaz` 提到 Gemini API 是免费的，引发了关于公司使用数据进行训练的影响以及用户可能如何利用免费层级的讨论。在提到这些担忧的同时，还伴随着关于潜在滥用以及 AI 实体可能向消费者提供的服务局限性的戏谑评论。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1112690728531918948/1208799217473290240)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的好友和社区保持紧密联系。
- [中国工作坊翻新 Nvidia 旧款旗舰游戏 GPU 用于 AI —— RTX 2080 Ti 升级至 22GB，售价 499 美元](https://www.tomshardware.com/pc-components/gpus/chinese-workshops-recondition-nvidias-old-flagship-gaming-gpu-for-ai-rtx-2080-ti-upgraded-to-22gb-for-dollar499)：据报道，该显卡在 Stable Diffusion、大语言模型 (LLMs) 和 Llama 2 中表现稳定。
- [RS-DPO：一种用于大语言模型对齐的混合拒绝采样与直接偏好优化方法](https://arxiv.org/abs/2402.10038)：来自人类反馈的强化学习 (RLHF) 已被广泛用于使大语言模型与用户意图对齐。然而，基于近端策略优化 (PPO) 的 RLHF 有时会……
- [LMQL 是一种用于 LLM 交互的编程语言。 | LMQL](https://lmql.ai/)：未找到描述
- [Soft prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting)：未找到描述
- [STEM.AI 最新突破 • Spotify for Podcasters 上的播客](https://podcasters.spotify.com/pod/show/stem-ai)：AI 和技术领域的创新成果通过播客这种实用的形式进行分享和讲解！
- [Cope Harder Cope GIF - Cope Harder Cope Sir - 发现并分享 GIF](https://tenor.com/view/cope-harder-cope-sir-cope-sir-american-psycho-gif-26276799)：点击查看 GIF
- [Adapter 方法 — AdapterHub 文档](https://docs.adapterhub.ml/methods.html#prefix-tuning)：未找到描述
- [Fear GIF - Fear - 发现并分享 GIF](https://tenor.com/view/fear-gif-4516342676332274003)：点击查看 GIF
- [Sleepy At Work Sleepy Kitten GIF - 发现并分享 GIF](https://tenor.com/view/sleepy-at-work-sleepy-kitten-cats-funny-animals-gif-13708263)：点击查看 GIF
- [ray-project/llm-applications 中的 RAG 笔记本](https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb)：构建生产级 RAG LLM 应用的全面指南。- ray-project/llm-applications
- [100k 测试 . exllama2(测试分支) + fa 1 - 128t 步内完成 100k](https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8)：100k 测试 . exllama2(测试分支) + fa 1 - 128t 步内完成 100k - gist:71658f280ea0fc0ad4b97d2a616f4ce8
- [学科方向-北京大学智能学院](https://sai.pku.edu.cn/xkjs/xkfx.htm)：未找到描述
- [NeuralNovel/Neural-DPO · Hugging Face 数据集](https://huggingface.co/datasets/NeuralNovel/Neural-DPO?row=33)：未找到描述

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1207959945379844116) (274 条消息🔥🔥): 

- **模型性能讨论**：用户 `@gamingdaveuk` 和 `@kquant` 讨论了各种 AI 模型的性能，`gamingdaveuk` 正在寻找一款适合故事创作、具备 8k context 且能严格遵守 prompt 的模型。他们分享了使用 Koboldcpp 和 textgenwebui 等模型的经验，甚至还包括了一段[关于修复 TextGen](https://pastebin.com/GbbyKPwD) 以支持 EXL2 模型的交流。`kquant` 对其模型在 roleplay 基准测试中排名第 15 位表示自豪（曾一度排名第 5）。

- **用户创建的角色扮演网站与 AI 协作**：`@shlapfish` 邀请成员测试一个兴趣爱好性质的 AI 角色扮演网站 [cowshout.com](https://cowshout.com)，该网站运行着 Nous-Hermes-2-Yi-34B.Q4_K_S.gguf。`shlapfish` 还就如何将 KoboldCpp 连接到 Silly Tavern 寻求建议，`@gamingdaveuk` 在 context size 和 GPU layer offloading 等设置方面提供了建议，并提供了 GitHub 上 [SillyTavern/SillyTavern](https://github.com/SillyTavern/SillyTavern) 的链接。

- **AI 优化技术讨论**：`@soufflespethuman`、`@netrve` 等用户之间的对话涉及了 AI 优化技术。他们讨论了剪枝（pruning）和量化（quantization）等技术对模型性能的影响，以及使用像 [Buttercup-4x7B-exl2](https://huggingface.co/royallab/Buttercup-4x7B-exl2) 这样的 Exl2 量化模型进行高效的 GPU inference。

- **AI 模型同类推荐**：用户 `@dao_li` 和 `@sao10k` 交流了类似于 Fimbulvetr-10.7B 的 AI 模型推荐。`sao10k` 推荐了 Fimbulvetr v2 并提供了 [Huggingface 上的模型链接](https://huggingface.co/Sao10K/Fimbulvetr-11B-v2-Test-14)，指出其表现可能优于第 1 版。`dao_li` 询问了在 3060 GPU 上使用该模型的情况，得到的建议是 q6 版本可以 headless 运行，或者使用 q5 版本处理其他任务。

- **利用 AI 构建 Lorebook**：`@mrdragonfox` 讨论了使用 `together.ai` 和 [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) 从数十年的 D&D 日志中创建数据集，意图针对兼容 OpenAI 的 endpoint 运行这些工具，并确认该流程可以处理长达 30 年的 D&D 日志。

**提到的链接**：

- [royallab/Buttercup-4x7B-exl2 · Hugging Face](https://huggingface.co/royallab/Buttercup-4x7B-exl2)：未找到描述
- [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025v1)：大型语言模型（LLMs）已被证明在各种自然语言处理任务中非常有效。然而，它们大量的参数为实际应用带来了重大挑战...
- [The ScribeFebruary 16, 2024 3:43 PMWhat tale do you wish to hear?#1D - Pastebin.com](https://pastebin.com/jyZj7zL2)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [Sao10K/Fimbulvetr-11B-v2-Test-14 · Hugging Face](https://huggingface.co/Sao10K/Fimbulvetr-11B-v2-Test-14)：未找到描述
- [The ScribeFebruary 16, 2024 3:43 PMWhat tale do you wish to hear?#1D - Pastebin.com](https://pastebin.com/GbbyKPwD)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。
- [IPIP Home](https://ipip.ori.org/index.htm)：未找到描述
- [MU* - Wikipedia](https://en.wikipedia.org/wiki/MU*)：未找到描述
- [Kooten/Buttercup-4x7B-5bpw-exl2 · Hugging Face](https://huggingface.co/Kooten/Buttercup-4x7B-5bpw-exl2)：未找到描述
- [CowShout](https://cowshout.com)：未找到描述
- [The BardFebruary 16, 2024 12:08 AMWhat will this song or poem be about? - Pastebin.com](https://pastebin.com/Y0tCh7LQ)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。
- [GitHub - e-p-armstrong/augmentoolkit at api-branch](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch)：将算力和书籍转换为 Instruct-Tuning 数据集 - GitHub
- [GitHub - SillyTavern/SillyTavern: LLM Frontend for Power Users.](https://github.com/SillyTavern/SillyTavern)：面向高级用户的 LLM 前端。通过在 GitHub 上创建账户为 SillyTavern/SillyTavern 的开发做出贡献。

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1208081828246986842) (26 条消息🔥): 

- **寻求理想的学习率调度 (Learning Schedule)**：`@cogbuji` 询问在医疗指令数据集上进行单个 epoch 训练结束时观察到的验证损失 (validation loss) 跳升是否预示着过拟合 (overfitting)，并考虑切换到 SGDR 学习率调度。在随后的讨论中，`@amogus2432` 提供了建议，认为对于仅一个 epoch 的训练，更长的 warm-up 或更低的最高学习率可能比 SGDR 更有益。

- **数据集的抗过拟合能力**：`@amogus2432` 和 `@dirtytigerx` 参与了由 `@maldevide` 发起的讨论，后者质疑哪些数据集在 10 个 epoch 后不会过拟合，并指出更多样化的数据集或采用非常规超参数微调 (hyperparameter tuning) 的数据集可能不会显示出过拟合迹象。

- **KL 散度损失缩减 (Loss Reduction) 实验**：`@amogus2432` 分享了 KL 散度损失缩减方法的实验结果，引入了一种自定义缩减方式，与标准的平均损失不同，该方法似乎能在多个 epoch 中稳定降低损失。

- **为模型添加知识**：`@magmaguy` 询问了通过微调 (finetuning) 模型来添加知识库的方法，以及将大量文本转换为适当 JSON 格式的过程。`@amogus2432` 建议研究检索增强生成 (RAG) 以提高陈述事实的准确性，并将进一步的讨论引导至更专业的频道。

- **将模型适配旧版 GPU 的疑虑**：`@wildcat_aurora` 寻求关于将通常使用 16-bit 浮点数的现代模型转换为 32-bit 的建议，以适配在 f32 精度下表现更好的旧款 P40 GPU。目标是在不进行训练或微调 (tuning) 的情况下进行直接转换，随后可能进行 q8 量化 (quantization)，以评估是否会有速度提升。

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1208121995947151402) (129 条消息🔥🔥): 

- **探索用于上下文感知的 Vision Transformers**：`@falconsfly` 分享了一个与 V-JEPA 相关的 [GitHub 仓库链接](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py)，这是一个关于从视频中进行自监督学习以实现上下文感知的项目。讨论指出，自回归（autoregressive）Token 预测在构建完整世界上下文方面存在局限性，用户注意到人类幼崽和动物是通过不同的机制学习的。
  
- **通过 Mojo 探索 SIMD 的奥秘**：`@coffeevampir3` 表达了对 Mojo 类型是 SIMD 抽象的新理解，并决定深入学习。这引发了包括 `@heralax` 在内的热烈讨论，涉及将 async 函数整合到工具包和 API 的非 async 代码中的挑战和复杂性。

- **节省时间的 'Autocommit'**：`@heralax` 介绍了一个他们开发的名为 [autocommit](https://github.com/e-p-armstrong/autocommit) 的工具，该工具利用 AI 根据 diff 生成 commit 消息，旨在减轻版本控制的负担。

- **在 Python 中搭建 Async 和 Sync 的桥梁**：`@spottyluck` 就 Python 同步代码中处理 async 函数向 `@the_ride_never_ends` 提供了建议，并指出了 [nest_asyncio](https://github.com/erdewit/nest_asyncio) 和 [asyncio-bridge](https://death.andgravity.com/asyncio-bridge) 等可能有所帮助的资源。

- **推广开源 Typescript Agent 框架**：`@magsp` 提到了他们的开源项目 [bazed-af](https://github.com/bazed-ai/bazed-af)，并寻求分享该项目以获取反馈的社区建议，`@heralax` 推荐了 LocalLlama 和 r/opensource 等平台。

**提到的链接**：

- [jepa/src/models/vision_transformer.py at main · facebookresearch/jepa](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py)：用于 V-JEPA 视频自监督学习的 PyTorch 代码和模型。 - facebookresearch/jepa
- [GitHub - e-p-armstrong/autocommit: Automatically commit in a repo and get AI to write the messages. Never lose work again!](https://github.com/e-p-armstrong/autocommit)：在仓库中自动 commit 并让 AI 编写消息。再也不会丢失工作！ - e-p-armstrong/autocommit
- [augmentoolkit/processing.py at aphrodite-branch · itsdotscience/augmentoolkit](https://github.com/itsdotscience/augmentoolkit/blob/aphrodite-branch/processing.py)：将计算和书籍转换为 Instruct-Tuning 数据集 - itsdotscience/augmentoolkit
- [GitHub - bazed-ai/bazed-af: 😎 Bazed.ai Agent Framework - Bazed.ai is a unified platform for building, running and scaling autonomous agents.](https://github.com/bazed-ai/bazed-af)：😎 Bazed.ai Agent 框架 - Bazed.ai 是一个用于构建、运行和扩展自主 Agent 的统一平台。 - bazed-ai/bazed-af
- [GitHub - erdewit/nest_asyncio: Patch asyncio to allow nested event loops](https://github.com/erdewit/nest_asyncio)：修补 asyncio 以允许嵌套事件循环。可以通过在 GitHub 上创建账号来为 erdewit/nest_asyncio 的开发做出贡献。

  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1208058377024831548) (339 条消息🔥🔥): 

- **AI 工作负载中的 GPU vs. TPU**: `.the_alt_man` 询问对于动态尺寸的计算图（例如可变变化的 “scan” 操作），**TPU** 是否比 **GPU** 效率更低。`johnryan465` 请求对 “动态内容” 进行定义，随后 `.the_alt_man` 针对具有可变长度的计算图做了进一步解释。

- **EleutherAI 社交聚会提醒**: `canadagoose1` 提醒成员参加在旧金山举行的 **EleutherAI** 社交聚会，提供了 Discord 链接，并提到了关于 `stellaathena` 此前发布的公告中存在标记错误。

- **学习思维链 (Chain of Thought)**: `emanuel65537` 询问是否存在一种技术，可以在没有预训练或人工提供中间步骤的情况下，训练 AI 模型学习**思维链**（例如两个 7 位数相乘）。`lucaslingle` 和 `vincent163_13311` 强调了在训练此类模型的数据集中包含分步操作的必要性。

- **超级对齐 (Superalignment) 资助截止时区查询**: `1rokosbasilisk` 询问超级对齐资助申请截止日期的时区，想知道是基于 **Anywhere on Earth (AOE)** 还是**太平洋时间 (PT)**。讨论中未给出明确答案。

- **压缩作为智能的衡量标准**: 由 `sentialx` 发起了一场哲学辩论，探讨**语言模型**是否等同于单纯的**数据压缩**，以及这是否可以被视为构建**知识**或**智能**。多位用户参与了讨论，探索数据压缩与智能之间的关系，一些人认为它们是难以轻易分离的纠缠概念。

**提到的链接**:

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/729741769192767510/1027492909227970570/1208128218809237504): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/729741769192767510/1207749000858439740): Discord 是通过语音、视频和文字进行交流的最简单方式。
- [视频生成模型作为世界模拟器](https://openai.com/research/video-generation-models-as-world-simulators): 我们探索了在视频数据上进行生成模型的大规模训练。具体来说，我们在具有不同时长、分辨率和宽高比的视频和图像上联合训练文本条件扩散模型……
- [Always Has Been Among Us GIF - Always Has Been Among Us Astronaut - 发现并分享 GIF](https://tenor.com/view/always-has-been-among-us-astronaut-space-betrayal-gif-23836476): 点击查看 GIF
- [EleutherAI 旧金山聚会 | Partiful](https://partiful.com/e/8hRUy9flN02dFLK4rBxh): EleutherAI 聚会
- [EleutherAI 旧金山聚会 | Partiful](https://partiful.com/e/8hRUy9flN02dFLK4rBxh/?reload=true): EleutherAI 聚会
- [不要衰减学习率，增加 Batch Size](https://arxiv.org/abs/1711.00489): 衰减学习率是常见做法。在这里我们展示了，通常可以通过在训练期间增加 Batch Size 来在训练集和测试集上获得相同的学习曲线。
- [BlackMamba: 状态空间模型的混合专家模型](https://arxiv.org/abs/2402.01771): 状态空间模型 (SSMs) 最近在大规模语言建模基准测试中展示了与 Transformer 相当的性能，同时实现了线性时间与内存复杂度……
- [David MacKay: 信息论、推理与学习算法：主页](https://www.inference.org.uk/mackay/itila/): 未找到描述
- [人类知识压缩竞赛：常见问题解答](http://prize.hutter1.net/hfaq.htm): 未找到描述
- [GitHub - vincent-163/transformer-arithmetic](https://github.com/vincent-163/transformer-arithmetic): 通过创建账号为 vincent-163/transformer-arithmetic 的开发做出贡献。
- [计算机器与智能](https://academic.oup.com/mind/article/LIX/236/433/986238): 我提议考虑“机器能思考吗？”这个问题。这应该从定义“机器”和“思考”这两个术语的含义开始。

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1207969497118679140) (196 条消息🔥🔥):

- **探索长文档处理**：[BABILong，一个长文档处理基准测试](https://arxiv.org/abs/2402.10790)，被用于评估 GPT-4 和 RAG 等模型，结果显示其上限为 $10^4$ 个元素。然而，经过 Recurrent Memory 增强微调的 GPT-2 处理了多达 $10^7$ 个元素，标志着在处理长输入方面取得了重大飞跃。
- **关于研究中 Liquid Models 存在感的辩论**：多位用户讨论了 Liquid Models（如 Liquid-S4）在当前 AI 研究中的存在和影响。尽管有令人印象深刻的无人机演示和坚实的 Benchmark，但这些模型在发起研究小组之外尚未得到广泛采用。
- **围绕 Liquid AI 发展方向的不确定性**：成员们对 Liquid AI 初创公司表示怀疑，称该公司的使命宣言是“虚饰（fluff）”，并寻求有关其具体项目和目标的实质性信息。
- **关于神经网络初始化方法的讨论**：几位用户讨论了不同初始化方法的优缺点，特别是 ZerO（0 和 1）、Identity、Hadamard，以及它们对 Large Language Model 稳定性和 Lottery Ticket Hypothesis 的潜在影响。
- **关于 kNN 方法与 Recurrent Memory 的疑问**：用户 @clashluke 询问了使用 kNN 查找不可微过去输入记忆的方法（[arXiv 链接](https://arxiv.org/abs/2203.08913)）与通过 Recurrent Memory 增强微调使 GPT-2 能够处理海量输入的方法之间的区别。

**提到的链接**：

- [Universal Neural Functionals](https://arxiv.org/abs/2402.05232)：许多现代机器学习任务中一个具有挑战性的问题是处理权重空间特征，即从神经网络的权重和梯度中转换或提取信息。最近的工作...
- [On Limitations of the Transformer Architecture](https://arxiv.org/abs/2402.08164)：Large Language Models (LLMs) 中幻觉的根本原因是什么？我们使用通信复杂度（Communication Complexity）证明了 Transformer 层无法进行函数复合（例如，识别一个图...）
- [来自 Kyle O'Brien (@KyleDevinOBrien) 的推文](https://x.com/KyleDevinOBrien/status/1758667079849480630?s=20)：当由于无法修改权重或假设其架构而使分类器实际上成为黑盒时，我们如何使其更具鲁棒性？在我们的预印本中，我们证明了可以提高鲁棒性...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://arxiv.org/abs/2402.10193)：Large Language Models (LLMs) 通常分两个阶段训练：在大规模互联网数据集上进行 Pre-training，以及针对下游任务进行 Fine-tuning。鉴于 Pre-training 的计算需求更高...
- [In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790)：本文解决了使用生成式 Transformer 模型处理长文档的挑战。为了评估不同的方法，我们引入了 BABILong，这是一个旨在评估模型能力的新基准测试...
- [Improving Language Plasticity via Pretraining with Active Forgetting](http://arxiv.org/abs/2307.01163)：预训练语言模型 (PLMs) 是当今自然语言处理的主要模型。尽管它们具有令人印象深刻的下游性能，但很难将 PLMs 应用于新语言...
- [CVPR 2016 Open Access Repository](https://openaccess.thecvf.com/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html)：未找到描述
- [UFO: A UI-Focused Agent for Windows OS Interaction](https://arxiv.org/abs/2402.07939)：我们介绍了 UFO，一个创新的 UI-Focused Agent，利用 GPT-Vision 的能力，专门为 Windows 操作系统上的应用程序量身定制以满足用户请求。UFO 采用双 Agent 框架来细致地...
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)：在增强 Large Language Models (LLMs) 的推理能力方面，先前的研究主要集中在特定的 Prompting 技术上，如 Few-shot 或 Zero-shot Chain-of-Thought (CoT) Prompting...
- [Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903)：Loss 尖峰经常发生在 Large Language Models 的 Pre-training 过程中。这些尖峰会降低 Large Language Models 的性能，有时甚至会破坏 Pre-training。由于 Pre-training 需要大量的...
- [Headless Language Models: Learning without Predicting with Contrastive Weight Tying](https://arxiv.org/abs/2309.08351)：语言模型的自监督 Pre-training 通常包括预测海量 Token 词表上的概率分布。在这项研究中，我们提出了一种创新的方法，将...

- [ZerO Initialization: Initializing Neural Networks with only Zeros and Ones](https://arxiv.org/abs/2110.12661)：深度神经网络通常使用随机权重初始化，并选择适当的初始方差以确保训练期间信号传播的稳定。然而，选择合适的方差...
- [Liquid AI: A New Generation of Foundation Models from First Principles](https://liquid.ai)：Liquid AI 是一家 MIT 的衍生公司，在马萨诸塞州波士顿和加利福尼亚州帕洛阿尔托设有办事处。我们的使命是基于第一性原理构建最先进的通用 AI 系统，并部署高性能、高效率的...
- [fairseq2/src/fairseq2/models/llama/builder.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/models/llama/builder.py#L262)：FAIR 序列建模工具包 2。通过在 GitHub 上创建账户，为 facebookresearch/fairseq2 的开发做出贡献。
- [fairseq2/src/fairseq2/models/transformer/frontend.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/models/transformer/frontend.py#L122C1-L123C1)：FAIR 序列建模工具包 2。通过在 GitHub 上创建账户，为 facebookresearch/fairseq2 的开发做出贡献。
- [fairseq2/src/fairseq2/nn/transformer/decoder_layer.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/nn/transformer/decoder_layer.py#L91)：FAIR 序列建模工具包 2。通过在 GitHub 上创建账户，为 facebookresearch/fairseq2 的开发做出贡献。
- [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951)：对线性状态空间模型（SSMs）的状态转移矩阵进行适当的参数化，并结合标准的非线性，使其能够高效地从序列数据中学习表示，特别是...
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)：语言模型通常需要经过训练或微调才能获取新知识，这涉及更新其权重。相反，我们设想语言模型可以简单地阅读并记忆...

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1208108282930135112) (3 条消息): 

- **因果擦除（Causal Scrubbing）方法受到关注**：`@neelnanda` 通过分享一篇 [lesswrong 文章](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) 建议探索因果擦除的概念，尽管提供的链接格式有误。
- **对影响力工作的赞赏**：`@yonghyunpark` 对富有启发性的工作表示感谢，特别感谢了 `@neelnanda`。

**提到的链接**：

- [未找到标题](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing&ved=2ahUKEwjHvcGsk7OEAxXoWEEAHRE3DEYQFnoECBUQAQ&usg=AOvVaw33dMhAk1jgQEvSBnTq8uOq)：未找到描述
- [重定向通知](https://www.google.com/url?sa=t&source=web&rct=j&opi=89)：未找到描述

  

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1207984429109088296) (38 messages🔥): 

- **MMLU 任务库已共享**：`@pminervini` 分享了 MMLU (Massive Multitask Language Understanding) 任务的 [GitHub 链接](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)。

- **Logprobs 警告问题**：`@noldtronics` 报告了在使用 `llama.cpp server + flask "openai api"-server` 进行模型评估时遇到的 logprobs 数据问题。在测试一个关于屋顶瓦片拆除的提示词时，收到了关于无效 token_logprobs 列表的 *WARNING* 以及关于 loglikelihood 无效响应的 *ERROR*。

- **Llamacpp 与 LM Evaluation Harness 的集成问题**：在 `@noldtronics` 发布了关于 Llamacpp 和 gguf.py 接口的问题并提交了[相关的 GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/1437) 后，`@hailey_schoelkopf` 表示正在调查 Llamacpp 接口中潜在的变化。

- **lm-evaluation-harness 安装陷阱与指南**：`@ilanser` 寻求在 Docker 环境中设置 `lm-evaluation-harness` 的建议，`@hailey_schoelkopf` 推荐使用 Nvidia 或 Coreweave 的镜像。`@vincent163_13311` 分享了一份深入的安装指南，以解决在 CUDA 环境下结合 vLLM 设置 lm_eval 时遇到的各种挑战。

- **lm-evaluation-harness 的运行测试建议**：几位用户讨论了如何测试 `lm-evaluation-harness`。`@baber_` 和 `@stellaathena` 建议使用 `--limit 10` 参数来缩短测试运行时间，而 `@vincent163_13311` 则建议使用 `gpt2` 模型和 `arc_easy` 任务，因为样本较少，评估速度更快。

**提及的链接**：

- [llama / gguf interface broken? · Issue #1437 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1437)：关于此问题我找到的唯一信息是 Discord 上的一个帖子：Armando Diaz: 嗨！我正在做一个研究项目，我们想知道是否可以使用 lm-evaluation-harness 来评估...
- [no title found](http://gguf.py:90)：未找到描述
- [lm-evaluation-harness/lm_eval/tasks/mmlu at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness)：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
- [from vllm._C import cuda_utils raise error · Issue #2797 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/2797)：ImportError: /root/autodl-tmp/conda/envs/wslconda/lib/python3.9/site-packages/vllm/_C.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops15to_dtype_layout4callERKNS_6TensorEN3c108optional...
- [no title found](https://download.pytorch.org/whl/cu118)：未找到描述

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1208006545884057630) (13 messages🔥): 

- **FIM-NEO-X 使用熟悉的架构进行训练**：`@hailey_schoelkopf` 澄清 **FIM-NEO-X** 模型是使用与 **GPT-NeoX** 相同的架构训练的，并且可以使用 GPT-NeoX 类与 Huggingface 兼容。

- **GPT-NeoX 的架构灵活性**：`@hailey_schoelkopf` 还指出，虽然 **GPT-NeoX** 主要支持基于 Transformer 的架构，但它已被其他人改编用于训练非 Transformer 模型，并引用了一篇详细介绍 `Based` 架构的文章。

- **理解多头注意力机制 (Multi-Head Attention) 的分布**：针对 `@jdranpariya` 的疑问，`@catboy_slim_` 解释说，在 `ParallelSelfAttention` 类中，`mpu.divide` 将注意力头公平地分布在各个分区中，以便进行模型并行化，而不会留下余数错误。

- **NeoX 参数的实现细节**：`@catboy_slim_` 指出，**NeoX** 可能尚未实现 **Megatron** 和 **Deepspeed** 的所有功能，代码库中可能仍未处理某些与流水线并行 (pipeline parallelism) 相关的参数。

**提及的链接**：

[Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)：未找到描述

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1207981486637187072) (328 messages🔥🔥):

- **优化 LLM 的 GPU 利用率**：用户正在探索提高 Token 生成速度的设置。一位使用 RTX 4050 和 Ryzen 7 配置的用户在 Offloading 50 层（layers）后达到了 34 tokens/s，并正在寻求进一步改进的建议。`@qiikzx` 也在询问如何对 `.gguf` 模型进行 Fine-tuning，使其听起来更像人类并去除审查。
- **探索本地 LLM 设置的可扩展性**：`@krypt_lynx` 正在考虑将 LLM 集成到游戏《边缘世界》（RimWorld）的 Mod 中，根据游戏中广泛的角色数据生成“气泡对话”，并正在寻求设置角色“Personas”的指南。
- **关于 LLM 平台和硬件的讨论**：讨论围绕在 Linux 与 Windows 上运行 LLM 的优势展开，其中一项对比显示 Linux 的每秒 Token 数（tokens per second）提升了 30%。用户还在分享他们使用各种硬件规格的经验，例如使用 GTX 1050 和 16GB RAM。
- **关于本地 LLM 能力的查询**：人们对 LLM 是否可以在基于 USB 的神经网络加速器（如 Coral USB 或 Intel Compute Stick）上运行产生了浓厚兴趣。另一项讨论旨在澄清 LM Studio 是否支持 Function Calling，还是仅支持纯文本响应。
- **LLM 相关问题的支持提问**：包括 `@hautc.it` 在内的多位用户请求针对特定 LM Studio 问题的指导，如文档上传、模型选择和连接错误；其他用户则请求一对一协助，以解决 AVX2 指令集要求和模型 Fine-tuning 等问题。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1204973625518587925)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1197279779603370015)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1110598183144399061/1207719619058733108)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1110598183144399061/1207936682440269844)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1709551591425245338?s=20)：但是模型真的*使用*了这些表示（representations）吗？通过寻找与探针（probe）具有相似权重的神经元，我们发现了许多对……的时空坐标敏感的空间和时间神经元。
- [LMQL 是用于 LLM 交互的编程语言。 | LMQL](https://lmql.ai/)：未找到描述
- [Qwen/Qwen1.5-72B-Chat-GGUF · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GGUF)：未找到描述
- [加拿大航空被勒令赔偿受航空公司聊天机器人误导的客户](https://www.theguardian.com/world/2024/feb/16/air-canada-chatbot-lawsuit)：公司声称其聊天机器人在提供有关丧亲票价的错误信息时“对其自身行为负责”
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/)：你的个性化 AI 聊天机器人。
- [HWiNFO - 免费系统信息、监控和诊断](https://www.hwinfo.com/)：免费硬件分析、监控和报告。深入的硬件信息、实时系统监控、报告等
- [How to mixtral](https://rentry.org/HowtoMixtral)：更新于 12/22，至少需要总计 20GB 左右的 VRAM / RAM。VRAM 越多，速度越快/效果越好。获取最新的 Kobold: https://github.com/kalomaze/koboldcpp/releases 获取模型，下载其中一个量化版本（quants）...
- [KnutJaegersberg/2-bit-LLMs at main](https://huggingface.co/KnutJaegersberg/2-bit-LLMs/tree/main)：未找到描述
- [GitHub - KillianLucas/open-interpreter: 计算机的自然语言界面](https://github.com/KillianLucas/open-interpreter)：计算机的自然语言界面。通过在 GitHub 上创建一个账户来为 KillianLucas/open-interpreter 的开发做出贡献。

- [GitHub - Josh-XT/AGiXT: AGiXT 是一个动态的 AI Agent 自动化平台，可跨多种 AI 提供商无缝编排指令管理和复杂任务执行。结合自适应记忆、智能特性和多功能插件系统，AGiXT 提供高效且全面的 AI 解决方案。](https://github.com/Josh-XT/AGiXT): AGiXT 是一个动态的 AI Agent 自动化平台，可跨多种 AI 提供商无缝编排指令管理和复杂任务执行。结合自适应记忆、智能特性...
- [Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1747617771796762901?s=46): 新版本已发布（将出现在 ICLR）！主要更新：- 在 Pythia 模型上的额外实验 - 对空间和时间神经元的因果干预 - 更多相关工作 - 澄清我们对字面 w... 的主张。
- [Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1747617820957876317?s=46): 然而，最近有一篇来自 Chen 等人关于 LLM 中“空间因果表示”的论文，该论文基于我们的工作，并发现“LLM 在解决...时学习并使用内部空间模型”。
- [Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1709551516577902782?s=20,): 语言模型是否有内部世界模型？时间感？在多个时空尺度上？在与 @tegmark 合作的一篇新论文中，我们通过发现一张字面上的世界地图提供了证据...
- [Biz Stone (@biz) 的推文](https://x.com/wesg52/status/1709): 让 livy 在睡觉前吃一片 airborne。
- [Implicit Representations of Meaning in Neural Language Models](https://arxiv.org/abs/2106.00737): 神经语言模型的有效性是否完全源于对表面词汇共现统计数据的准确建模，还是这些模型对其描述的世界进行了表示和推理？...
- [Joseph Sarnecki (@JosephSarnecki) 的推文](https://x.com/JosephSarnecki/status/1758541761495159011?s=20): @wesg52 @tegmark 你能否测试大语言模型是否对其他激活执行相同的操作——特别是在想象的场景中（即：角色扮演）？我很好奇 LLM 是否创建了一个内部世界...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1208037732333658132) (33 条消息🔥): 

- **寻求长上下文模型**：`@msz_mgs` 请求推荐支持 32k context 的最佳 7b 模型，尽管在提供的消息中似乎没有人给出建议。
- **LM Studio 模型探索技巧**：针对 `@mitchalley` 关于定位特定任务最佳模型的查询，`@heyitsyorkie` 建议查看 TheBloke 的仓库，并在 LM Studio 的 Model Explorer 中按下载量排序。
- **模型大小与性能**：针对 `@mitchalley` 找到 TheBloke 的热门模型，`@alastair9776` 建议从 `Q5_K_M` 或 `Q4_K_M` 模型开始，因为它们在各种配置下都有稳健的表现，并鼓励通过实验找到最适合的模型。
- **语言翻译模型偏好**：对于语言翻译模型，`@mulder1` 青睐 Mistral-7B-openorca-4.0，而 `@fabguy` 和 `@heyitsyorkie` 则推荐 Deepl，认为其在翻译任务中表现更优。然而，`.ben.com` 观察到 fanyi.baidu.com 在中文翻译方面优于其他工具。
- **选择正确的量化模型**：当被问及 `dolphin-2.7-mixtral-8x7b.Q5_0.gguf` 和 `dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf` 哪个性能更好时，`@alizatjan` 指出 `K_M` 是一种更新且更好的量化方法，并引导 `@snens9650` 参考一篇详细的 Reddit 帖子以获取更多见解。

**提到的链接**：

- [TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ · Hugging Face](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ): 未找到描述
- [TheBloke/CodeLlama-70B-Instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF): 未找到描述
- [Reddit - 深入了解任何内容](https://www.reddit.com/r/LocalLLaMA/comments/159nrh5/the_difference_between_quantization_methods_for/): 未找到描述
- [Qwen/Qwen-VL-Chat · Hugging Face](https://huggingface.co/Qwen/Qwen-VL-Chat): 未找到描述

  

---

### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1208033125968650300) (14 条消息🔥): 

- **恢复出厂设置的困惑**：`@msz_mgs` 询问如何对 LMS 设置进行出厂重置，最初不确定其设置是否为默认值。在尝试“重置为默认值”后，`@heyitsyorkie` 指导他们从指定文件夹中手动删除提示词预设（prompt presets）。

- **预设文件夹位置协助**：`@msz_mgs` 询问了用于重置 LMS 设置的预设文件夹位置，`@heyitsyorkie` 引导他们点击“打开预设文件夹...”来找到它。

- **LMS 默认恢复**：`@msz_mgs` 删除了文件夹中的所有预设，并确认 LMS 重新填充了默认预设，解决了他们的问题。

- **UI 面板行为不一致**：`@borisrusev` 报告了一个 UI 不一致问题，即设置面板在禁用时会变淡，但聊天面板不会。`@heyitsyorkie` 承认了这种不一致，暗示两者在推理（inference）期间都应统一变灰。

- **继续按钮 Bug 的挫败感**：`@logandark` 对一个阻止删除尾随换行符的 Bug 表示沮丧，这影响了继续按钮的功能。`@logandark` 引用了之前的 Bug 报告，提到该问题正影响他们对 LM Studio 的使用。
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1208028478968954951) (127 条消息🔥🔥): 

- **利用 Intel 核心运行虚拟机**：`@addressofreturnaddress` 计划利用 Intel 的额外核心运行带有 macOS 和 Windows 的 KVM，通过考虑使用 14900K 搭配 mini ITX 主板来寻求性能与便携性的平衡，并期待从 3090 升级到 5090 GPU。
- **硬件爱好者推崇多 GPU 配置**：包括 `@heyitsyorkie`、`@nink1` 和 `@goldensun3ds` 在内的多位用户讨论了复杂的多 GPU 配置，例如配对两个或更多 3090，以及 PCIe 插槽和适配器的问题，强调了功耗和空间限制是主要的制约因素。
- **GPU 适配器和 PC 设置挑战**：`@goldensun3ds` 遇到了 Ebay 上 4060 Ti GPU 标签错误的问题，并详细讨论了在多 GPU 设置中使用 PCIe x1 适配器和 PCIE 延长线的尝试。讨论中穿插了一些创意但“简陋”的解决方案，比如将 PC 侧放，`@heyitsyorkie` 等用户对此进行了积极辩论。
- **多 GPU 工具和软件利用查询**：在硬件讨论中，`@goldensun3ds` 和 `@heyitsyorkie` 深入探讨了 LM Studio 多 GPU 支持的复杂性，探索通过配置调整来平衡不同显卡之间的 GPU VRAM 利用率。
- **AVX2 技术支持要点**：`@consuliam` 询问了关于禁用 AVX2 支持验证的问题，`@yagilb` 提供了关于非 AVX2 测试版（beta）的建议，`@heyitsyorkie` 建议使用 HWiNFO 检查 CPU 能力，而 `@nink1` 则幽默地警告不要依赖 ChatGPT 获取准确的硬件信息。

**提到的链接**：

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)：未找到描述
- [Pro WS WRX90E-SAGE SE｜Motherboards｜ASUS Global](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/)：华硕工作站主板专为 AI 训练、深度学习、动画或 3D 渲染领域的专业人士设计。具有可扩展的显卡、存储、出色的连接性和可靠性...
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-r)：查找、下载并实验本地 LLM
- [no title found](https://www.amazon.ca/dp/B0BDCZRBD6)：未找到描述
- [HWiNFO - Free System Information, Monitoring and Diagnostics](https://www.hwinfo.com/)：免费硬件分析、监控和报告。深入的硬件信息、实时系统监控、报告等
- [ASUS Global](https://www.asus.com/motherboards-components)：未找到描述
- [Stillesque GIF - Stillesque - Discover &amp; Share GIFs](https://tenor.com/view/stillesque-gif-25544869)：点击查看 GIF
- [Amazon.com: StarTech.com PCI Express X1 to X16 Low Profile Slot Extension Adapter - PCIe x1 to x16 Adapter (PEX1TO162) : Electronics](https://www.amazon.com/gp/aw/d/B0039XPS5W/)：未找到描述

  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1208127995340914758) (12 条消息🔥): 

- **提倡置顶聊天礼仪**：`@jedd1` 建议需要置顶 [don't ask to ask](https://dontasktoask.com/) 网站，强调更好的聊天提问礼仪应该是直接说明实际问题，而不是询问是否有专家在场。
- **提议用机器人解决模糊查询**：针对 `@jedd1` 关于置顶礼仪指南的评论，`@heyitsyorkie` 幽默地提议可以用机器人来回复此类行为。
- **聊天室中的教育时刻**：`@jedd1` 幽默地评论说，应该教会人们解释他们做了什么、预期发生什么以及实际发生了什么，并引用了“三段式报告模板”。
- **不清晰错误报告的常见问题**：`@heyitsyorkie` 同意 `@jedd1` 关于模糊提问的问题，并提到经常遇到用户报告 "exit code" 错误却不提供足够的排查细节。
- **Linux：是欢迎还是同情？**：一次友好的交流，`@zioalex` 介绍自己是频道里的 Linux 新用户，随后 `@fabguy` 开玩笑地表示了慰问，而 `@zioalex` 则幽默地回复说还有更糟糕的操作系统。

**提到的链接**：

[Don't ask to ask, just ask](https://dontasktoask.com/)：未找到描述

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1208175972994785280) (8 条消息🔥): 

- **寻求 Autogen 指导**：用户 `@zioalex` 询问如何开始使用 **Autogen 或 CrewAI** 等工具，并寻找好的指南。`@heyitsyorkie` 提供了一个关于 [Autogen 的 YouTube 教程](https://www.youtube.com/watch?v=Hds_fJaAu78) 以提供帮助。
- **报告链接失效**：`@zioalex` 报告了一个无法使用的示例链接，并请求修复。
- **仓库缺失无法修复**：`@heyitsyorkie` 解释说该示例的仓库已不存在，因此无法修复链接。
- **建议移除过时链接的置顶**：在得知链接无法修复后，`@zioalex` 建议移除该失效链接消息的置顶。
- **分配置顶管理任务**：`@heyitsyorkie` 表示移除置顶是 `<@1108574387889778738>` 的任务。
- **置顶已移除**：用户 `@fabguy` 确认他们已经移除了有问题的链接。
- **分享随机 Gif**：`@carasen12` 分享了一个困惑狗狗的 [Tenor gif](https://tenor.com/view/bobawooyo-dog-confused-dog-huh-dog-meme-shocked-dog-gif-16713203299056947073)，这与正在进行的讨论无关。

**提到的链接**：

- [Bobawooyo Dog Confused GIF - Bobawooyo Dog confused Dog huh - Discover &amp; Share GIFs](https://tenor.com/view/bobawooyo-dog-confused-dog-huh-dog-meme-shocked-dog-gif-16713203299056947073)：点击查看 GIF
- [Local AI Agent with ANY Open-Source LLM using LMStudio](https://www.youtube.com/watch?v=Hds_fJaAu78)：欢迎观看关于使用 LM Studio 构建你的第一个开源 AI Agent 团队的解释和教程！我们还将学习如何设置它...

  

---


### LM Studio ▷ #[rivet](https://discord.com/channels/1110598183144399058/1167546635098804284/) (1 条消息): 

mend1440: 天哪，这个项目太火了（fuego）！！！
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1208043899252375642) (7 条消息): 

- **咨询 LM Studio 集成**：`@cyberir` 询问群里是否有人有将 **LM Studio** 与 **Flowise** 配合运行的经验，这引发了关于技术集成的讨论。
- **探索 LM Studio 和 LangFlow**：`@mend1440` 分享了他们尝试将 LM Studio 与 **LangFlow** 集成的尝试，提到了使用 `http_client` 和环境变量连接 OpenAI API 的可能性。
- **在缺乏编程经验的情况下摸索**：`@cyberir` 和 `@mend1440` 表示他们不是程序员，但尽管有此障碍，仍尝试完成集成过程。
- **LM Studio 服务器连接问题**：`@cyberir` 提到在让 LM Studio 的服务器正常工作时遇到困难，暗示需要解决该问题的帮助或指导。
- **手动设置获得成功**：`@mend1440` 报告最终通过设置 Base URL 变量并手动将配置详情输入 **Langflow**，使系统成功运行。

**提到的链接**：

- [无标题](http://my.test.server.example.com:8083",)：未找到描述
- [无标题](http://my.test.proxy.example.com",)：未找到描述

  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1207969694464614400) (370 条消息🔥🔥): 

- **Mistral 发布神秘模型**：`@lerela` 宣布了一个名为 `next` 的新原型模型，邀请用户在 lmsys 上试用并提供反馈。**Mistral-Next** 的官方细节和功能仍处于保密状态，引发了用户的好奇和猜测。

- **Mistral-Next 性能辩论**：虽然像 `@sven_72358` 这样的用户在有限的测试中发现 **Mistral-Next** 令人印象深刻，且可与 GPT-4 媲美，但其他用户如 `@jiha` 反馈它在一些其他模型能正确回答的基础逻辑问题上失败了。Mistral-Next 相对于 Mistral Medium 和 Mixtral 等基准测试的真实性能和参数规模尚未得到确认。

- **Mistral-Next 代码生成响应受限**：`@Shadow27` 和 `@mrdragonfox` 讨论了 **Mistral-Next** 在生成完整代码响应方面似乎存在的局限性。一些用户认为这可能是设计使然，而另一些用户则建议将其作为 Function 调用工具使用，而非提供完整解决方案。

- **探索 Mistral 模型的 Fine-Tuning 选项**：`@mato8792` 寻求关于使用 2xRTX3090Ti 显卡针对新语言对 **7b (full Mistral)** 模型进行 Fine-Tuning 的建议。`@mrdragonfox` 建议将 LoRA tuning 作为可行方案，并推荐咨询 Discord 上的 `@dirtytiger` 和 `TheBloke` 以获取详细指导。

- **大型语言模型（LLM）上下文长度讨论**：`@mrdragonfox` 和 `@thezennou` 等用户辩论了 LLM 中 Context Length 的实际用途和性能影响，认为改进对较短且相关上下文的处理可能比追求极长的上下文更可取。

**提及的链接**：

- [Mistral AI launches Mixtral-Next | Hacker News](https://news.ycombinator.com/item?id=39406168)：未找到描述
- [Chat with Open Large Language Models](https://chat.lmsys.org/)：未找到描述
- [Chat with Open Large Language Models](https://chat.lmsys.org)：未找到描述
- [Infowar Skeptical GIF - Infowar Skeptical Conspiracy - Discover &amp; Share GIFs](https://tenor.com/view/infowar-skeptical-conspiracy-theory-nod-gif-13373739)：点击查看 GIF
- [TheBloke/Mixtral-8x7B-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF)：未找到描述
- [Issues · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/1002>)：一个用于 LLM 的高吞吐量且显存高效的推理和提供服务的引擎 - Issues · vllm-project/vllm
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe)：用于 LLM Tokenization 中常用的字节对编码（BPE）算法的极简、干净的代码。 - karpathy/minbpe

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1207982709683523584) (30 条消息🔥): 

- **Mistral-Next 细节初现**：`@mrdragonfox` 将 **Mistral-Next** 描述为一个原型模型，并得到了 `@lelio` 的确认。线程中对其能力和发布存在猜测，但尚未分享具体细节。
- **本地 GPT-4 编程能力讨论**：`@nemomolok` 对**本地运行 GPT-4** 进行编程的潜力表示兴奋，这预示着 AI 的发展可能会提供更易获得的强大编程辅助。
- **提取表格数据的挑战**：`@mehdi_guel` 正在使用 **Mistral-small** 开发一个 RAG 应用，并寻求从 Word 文档中提取表格单元格的建议，因为 LLM 通常难以处理结构化数据。
- **表格数据的替代方案建议**：`@mrdragonfox` 建议使用 In-context learning、通过脚本将数据提取到数据库，以及潜在的 **text2sql** 方法作为在 Mistral 中处理表格数据的替代方案。
- **期待上下文理解能力的提升**：`@tensorbender` 希望经过 Fine-tuning 的 LLM 能够更好地将用户 Prompt 与“输入上下文”区分开来，特别是为了增强小模型在长上下文生成任务中的性能。
  

---

### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1207991233926922250) (43 messages🔥): 

- **6Freedom Studio CTO 寻求 Mistral 集成**：`@fangh`，[6Freedom Studio](https://6freedom.studio) 的 CTO，咨询了关于在 VR/AI 相关产品中本地部署 (on-premise) 集成 Mistral 的事宜。他们被建议联系 Mistral 支持部门或与 Mistral 的开发者关系负责人 `@803073039716974593` 沟通。
- **讨论 Mistral 的尺寸选项**：`@mrdragonfox` 提到目前提供 "tiny" 和 "small" 尺寸（配置如 7b/8x7b），而 medium 尺寸可能需要企业协议和销售咨询。
- **解决 Unauthorized 错误**：`@renaudr.` 在尝试使用 Mistral 的 API 时遇到了 "Unauthorized" 错误，并分享了他们的 curl 命令以寻求帮助。
- **排查 API Key 问题**：`@mrdragonfox` 建议该错误可能是由于 Key 无效引起的，并请 `@renaudr.` 确认其账单状态是否激活。
- **针对 API Key 激活延迟提出的解决方案**：`@mrdragonfox` 澄清说，新账户的 Key 激活大约需要 5 分钟，并建议在 curl 命令中使用 API Key 之前先将其设置为环境变量。

**提到的链接**：

[6freedom | Experts en technologies immersives](https://6freedom.studio)：6freedom 是一家沉浸式技术专家机构。我们协助您开发定制化项目。

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1208059513874153552) (6 messages): 

- **SLEB：一种加速 LLM 的技术**：`@alex096170` 分享了一篇 [arXiv 论文](https://arxiv.org/abs/2402.09025v1)，介绍了一种名为 **SLEB** 的新方法，该方法通过移除冗余的 Transformer 块来对 LLM 进行剪枝，从而在不显著影响性能的情况下提高推理速度。
- **寻找最佳 LLM 预训练框架**：`@remek1972` 询问了关于 **预训练大语言模型** (LLM) 的最佳框架，特别是针对在大规模语料库上从零开始训练而非微调 (finetuning) 的场景。
- **针对不同规模的框架推荐**：`@quicksort` 建议框架的选择取决于并行化的规模；**Accelerate 配合 deepspeed** 适用于多节点环境，而 **axolotl** 对于单节点设置可能更简单。
- **扩展 LLM 预训练的建议**：对于在 **多节点** 上预训练 LLM，`@quicksort` 建议关注 **Stas Bekman 的 Twitter** 及其关于机器学习工程的书籍，并指出此类任务通常使用 Accelerate 配合 deepspeed。
- **对预训练指导表示感谢**：`@remek1972` 对这些建议表示感谢，并表示提供的信息非常有帮助。

**提到的链接**：

[SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025v1)：大语言模型 (LLM) 已被证明在各种自然语言处理任务中非常有效。然而，其庞大的参数量给实际应用带来了重大挑战...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1207967772236054529) (2 messages): 

- **Elqano 正在招聘 AI 人才**：`@thomas_saulou` 发布了 Elqano 的招聘机会，这是一家位于法国比亚里茨的初创公司，正在寻找生成式 AI (Generative AI) 开发人员。有关该职位的详细信息（包括生成式 AI 在知识领域的应用和公司背景）可在其 [Welcometothejungle 招聘列表](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer)中找到。

- **针对种子前 (Pre-Seed) 初创公司的 AI Launchpad 机会**：`@deedubs__` 分享了一个让种子前 AI 初创公司在 3 月的 Data Council 上获得曝光的机会，该活动由 **Zero Prime Ventures** 呈现。感兴趣的创始人可以通过提供的链接 [Zero Prime Ventures AI Launchpad](https://zeroprime.vc/ai-launchpad) 进行申请，并联系 `@deedubs__` 了解更多关于 Data Council 的信息。

**提到的链接**：

[Applied Generative AI Engineer - Elqano - CDI](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer)：Elqano 正在招聘一名应用生成式 AI 工程师！

  

---

### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1208523756629135450) (24 messages🔥): 

- **Mistral vs GPT-4 编程能力对比**：`@mrobino` 认为在编程能力方面，**Mistral Medium** 更接近 **GPT-3.5** 而不是 **GPT-4**，而 `@mrdragonfox` 则报告称 **Mistral** 的编程结果优于 **GPT-4**。
  
- **AI 结合 TDD 的集成方法**：`@mrdragonfox` 讨论了他们将**测试驱动开发 (TDD)** 与 AI 辅助相结合的独特工作流，重点是让 AI 实现刚好足以通过编写好的测试的代码并进行正式验证，在保持对架构控制的同时引导生成。

- **开源协作提议**：`@nani99` 表示愿意为开源贡献分享计算资源，特别是用于创建高质量的合成数据，而 `@mrdragonfox` 则对所提供的计算资源性质表现出兴趣。

- **Augmentoolkit 贡献与数据集清理**：`@mrdragonfox` 分享了他们在 [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) 上的贡献链接，并讨论了大型数据集所需的广泛清理工作，提到在其硬件配置下，第一阶段清理大约需要 25 天的运行时间。

- **数据清理与测试讨论**：`@mrdragonfox` 和 `@akshay_1` 交流了关于数据清理的想法，提到了 Embedding 技术和 Regex，并讨论了测试数据集清理过程各部分的工作量和进度安排。

**提到的链接**：

[GitHub - e-p-armstrong/augmentoolkit at api-branch](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch)：将计算和书籍转换为 Instruct-Tuning 数据集 - GitHub - e-p-armstrong/augmentoolkit at api-branch

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1208263743306469426) (16 messages🔥): 

- **RWKV 粉丝俱乐部**：`@hexani` 表达了对 **RWKV** 的兴奋，`@vatsadev` 表示热烈赞同。
- **压缩以增强效果**：`@elder_plinius` 分享了 [MYLN](https://github.com/elder-plinius/MYLN)，这是一个旨在*压缩和缩写文本*以提高上下文长度和效率的工具，并征求基准测试建议。
- **为了真相进行 Token 化**：针对 `@elder_plinius`，`@vatsadev` 建议对比压缩前后的 Token 数量，以揭示该工具对 LLM 是否有效。
- **工具复杂性揭秘**：`@vatsadev` 警告说，缩写可能无法在 LLM 的 Tokenizer 中体现，可能会损害理解力，并建议瞄准常用 Token。
- **事与愿违的结果**：`@elder_plinius` 报告了一个令人惊讶的结果：使用 MYLN 后，文本样本的字符数减少了，但产生的 Token 却更多了，这表明其方法可能存在问题。

**提到的链接**：

[GitHub - elder-plinius/MYLN: 用于增强 LLM 之间通信的上下文长度和效率的语言压缩器。](https://github.com/elder-plinius/MYLN)：用于增强 LLM 之间通信的上下文长度和效率的语言压缩器。 - elder-plinius/MYLN

  

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1208029726938234942) (33 条消息🔥): 

- **模型微调展示**：`@pradeep1148` 分享了[一段 YouTube 视频](https://www.youtube.com/watch?v=EYR_kd3X03M)，题为 *“使用 Llama Factory 对模型进行 Function Calling 微调”*，提供了关于为函数调用微调模型的见解。
- **YOLO-World 发布**：`@pradeep1148` 通过[一段 YouTube 视频](https://www.youtube.com/watch?v=yaqi8xRUsp4)重点介绍了腾讯 AI Lab 发布的 **YOLO-World**，展示了这一实时、Zero-shot 目标检测模型。
- **大语言模型之战**：`@carsonpoole` 提到了 **Senku 70b**，这是一个在 Puffin 上微调的大语言模型，并将其与 *ChatGPT 4* 进行了对比。他邀请其他人发送 Prompt 来测试 Senku 的输出，并分享了一些示例结果。
- **网页导航愿景**：`@pradeep1148` 分享了视频“[使用 LangGraph 的网页浏览 Agent](https://www.youtube.com/watch?v=gbGYN3YyTS4)”，介绍了 WebVoyager，这是一个能够通过控制鼠标和键盘进行网页浏览的 Agent，`@teknium` 对此分享表示赞赏。
- **探索视频生成的未来**：`@pradeep1148` 还在[一段 YouTube 视频](https://www.youtube.com/watch?v=7lsOzA3WhSI)中发布了关于 OpenAI 的 SORA 文本转视频（Text to Video）模型的内容，讨论了其根据文本 Prompt 生成视频的能力。在随后的讨论中，`@gabriel_syme` 和 `@teknium` 等用户辩论了该技术的实际应用和未来发展。

**提到的链接**：

- [Web Browsing Agent using LangGraph](https://www.youtube.com/watch?v=gbGYN3YyTS4)：WebVoyager 是由 He 等人开发的具有视觉能力的网页浏览 Agent，能够控制鼠标和键盘。它通过查看带注释的浏览器界面来工作...
- [Finetune model for Function Calling (Tool Call) with Llama Factory](https://www.youtube.com/watch?v=EYR_kd3X03M)：使用 Llama Factory 对模型进行 Function Calling (Tool Call) 微调 #llm #ml #ai #largelanguagemodels #deeplearning #python #pythonprogramming https://github.c...
- [OpenAI SORA Text to Video model and Technical Report](https://www.youtube.com/watch?v=7lsOzA3WhSI)：介绍 Sora，我们的文本转视频模型。Sora 可以生成长达一分钟的视频，同时保持视觉质量并遵循用户的 Prompt。#...
- [YOLO-World: Real-Time, Zero-Shot Object Detection](https://www.youtube.com/watch?v=yaqi8xRUsp4)：2024 年 1 月 31 日，腾讯 AI Lab 发布了 YOLO-World（代码已在 GitHub 开源），这是一个实时、开放词汇的目标检测模型。YOLO-World 是一个 Zero...

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1208738399729754112) (1 条消息): 

- **基准测试日志迁移**：`@teknium` 宣布 **Benchmarks Log** 已移至一个新的 GitHub 仓库。任何对各种 LLM 的基准测试日志感兴趣的人现在都可以在 [GitHub 上的 LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs) 找到它们。

**提到的链接**：

[GitHub - teknium1/LLM-Benchmark-Logs: Just a bunch of benchmark logs for different LLMs](https://github.com/teknium1/LLM-Benchmark-Logs)：只是不同 LLM 的一堆基准测试日志。通过在 GitHub 上创建账号为 teknium1/LLM-Benchmark-Logs 的开发做出贡献。

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1207966019788865566) (26 条消息🔥): 

- **使用 UE 生成合成数据**：`@deki04` 阐明了在 AI 图像领域使用 **Unreal Engine (UE)** 生成合成数据是常见做法，甚至提到使用 **Microsoft 的 AirSim 插件** 作为此类任务的工具。

- **关于 Whisper ASR 能力的见解**：`@amgadoz` 分享了一系列 [博客文章](https://amgadhasan.substack.com/)，深入探讨了 OpenAI 的尖端 **Automatic Speech Recognition (ASR)** 模型 **Whisper**，包括其架构细节、多任务处理能力以及背后的海量数据准备过程。

- **开源 GritLM 发布**：`@Muennighoff` 推出了 **GRIT**，这是一个统一了文本嵌入（text embeddings）和生成任务的模型，声称提高了 **Retrieval-Augmented Generation (RAG)** 等操作的效率。该发布附带了[学术论文](https://arxiv.org/abs/2402.09906)和 [GitHub 仓库](https://github.com/ContextualAI/gritlm)的链接。

- **表征工程（Representation Engineering）探索**：`@.benxh` 分享了关于 **Representation Engineering** 的资源，重点介绍了一篇[论文](https://arxiv.org/abs/2310.01405)及相关的 [GitHub 代码](https://github.com/andyzoujm/representation-learning)，这些资源展示了在推理过程中分析和操纵 AI 模型行为的方法，无需进行 Prompt Engineering 或微调（finetuning）。

- **NeurIPS 2023 关于梯度下降的提交**：`@euclaise` 分享了一篇关于 **DoWG (Distance over Weighted Gradients)** 的 **NeurIPS 2023** [论文链接](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html)，这是一种新型的无参数梯度优化器，并指出其在适应平滑和非平滑问题方面的高效性和普适性。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1053877538025386074/1149866623109439599/1207052162203390032)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html)：未找到描述
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/)：未找到描述
- [使用开源本地 LLM 从零构建纠错型 RAG](https://youtube.com/watch?v=E2shqsYwxck&si=uF0H5IaMKGiZWPeb)：使用较小的本地 LLM 构建具有更复杂逻辑流的 LLM 应用可能具有挑战性。图（Graphs）提供了一种解决方法，可以规划逻辑流...
- [来自 Niklas Muennighoff (@Muennighoff) 的推文](https://x.com/muennighoff/status/1758307967802224770)：介绍 GRIT🦾，统一文本嵌入 🔢 和生成 📝。GritLM 在嵌入 (MTEB) 和生成任务 (BBH 等) 上均达到开源 SoTA —— 两者集成于一个模型中。查看推文了解 GRIT🦾 如何让 RAG 提速 >60% 且更...
- [解码 Whisper：深入了解其架构和转录过程](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link)：多部分系列文章的第 2 部分，我们将深入探讨 OpenAI 的尖端自动语音识别模型 Whisper。
- [探索 Whisper 的多任务接口：近距离观察其语音转录和翻译能力](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link)：多部分系列文章的第 3 部分，我们将深入探讨 OpenAI 的尖端自动语音识别模型 Whisper。
- [Whisper 的诞生：深入探讨其训练数据和过程](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link)：多部分系列文章，我们将深入探讨 OpenAI 的尖端自动语音识别模型 Whisper。

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1207962649216942100) (323 条消息🔥🔥): 

- **RAG 尚未过时**：尽管有传言称其已过时，但 `@gabriel_syme` 和 `@n8programs` 展开了激烈的辩论，认为 **RAG** 依然非常有生命力，因为上下文并不是其唯一的问题，层级结构对于大型数据集至关重要。

- **OpenHermes 微调乐趣**：`@n8programs` 报告了在 **OpenHermes** 数据集上成功训练 **TinyLLaMA** 的情况，训练损失（train loss）降至 1 以下；而 `@jiha` 则强调了在该数据集上选择长样本进行微调的有效性。

- **Groq 凭借速度引起轰动**：`@gezegen` 提到 Groq 的 LPU 推理引擎实现了令人印象深刻的 Token 处理速度，引发了热烈讨论，而 `@swaystar123` 和 `@leontello` 则讨论了它与 H100 等其他硬件的对比。

- **语言模型（LM）可以处理海量文档**：`@gabriel_syme` 分享了一篇 arXiv 论文，展示了通过循环记忆增强（recurrent memory augmentations）对 GPT-2 进行微调，可以实现高达 $10^7$ 个元素的文档处理，在长输入处理能力上超越了 GPT-4 和 RAG。

- **Green Blob 加入 Discord 并发现了此聊天**：`@greenblob6064` 在发现 Nous Research AI Discord 时表示惊讶，原以为 SAIL 是唯一的社区，而 `@powerful_wolf_14649` 则对互联网上关于 Nous Research 的信息普遍匮乏表示疑问。

**提到的链接**：

- [来自 Lewis Tunstall (@_lewtun) 的推文](https://x.com/_lewtun/status/1758520258132865210?s=20)：我在 @Teknium1 的 OpenHermes 数据集上测试了“long is more”技巧，效果出奇地好 🔥！- 选择 1k 个最长的样本 (0.1%) - 对 Mistral-7B 进行 15 个 epoch 的 SFT...
- [Groq](https://groq.com/)：未找到描述
- [在 1000 万规模的干草堆中寻针：循环记忆发现 LLM 遗漏的信息](https://arxiv.org/abs/2402.10790)：本文解决了使用生成式 Transformer 模型处理长文档的挑战。为了评估不同的方法，我们引入了 BABILong，这是一个旨在评估模型能力的新基准...
- [SOCIAL MEDIA TITLE TAG](https://os-copilot.github.io/)：SOCIAL MEDIA DESCRIPTION TAG TAG
- [来自 Shane Parr (@sparr_ml) 的推文](https://x.com/sparr_ml/status/1758246182285914136?s=20)：2024 年 LLM 购买指南首先应该说明的是，上下文长度在很大程度上是无关紧要的；重要的是输出质量、质量还是质量，而这只能通过高...
- [LDJnr/LessWrong-Amplify-Instruct · Hugging Face 数据集](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct)：未找到描述
- [GitHub - karpathy/minbpe: 用于 LLM Tokenization 中常用的字节对编码（BPE）算法的极简、整洁代码。](https://github.com/karpathy/minbpe)：用于 LLM Tokenization 中常用的字节对编码（BPE）算法的极简、整洁代码。- karpathy/minbpe
- [介绍 Sora — OpenAI 的文本生成视频模型](https://youtube.com/watch?v=HK6y8DAPN_0&si=dm3GMf22C89I2gLB)：介绍 Sora，我们的文本生成视频模型。Sora 可以创建长达 60 秒的视频，具有高度详细的场景、复杂的摄像机运动和多个角色...
- [来自 OpenAI (@OpenAI) 的推文](https://fxtwitter.com/OpenAI/status/1758192965703647443?s=20)：Prompt：“一位时尚女性走在东京街头，到处是温暖发光的霓虹灯和生动的城市标牌。她穿着黑色皮夹克、红色长裙和黑色靴子，拎着一个黑色手提包...”
- [GitHub - luuyin/OWL: “离群值加权逐层稀疏（OWL）：将 LLM 剪枝至高稀疏度的缺失秘诀”的官方 Pytorch 实现](https://github.com/luuyin/OWL)：“离群值加权逐层稀疏（OWL）：将 LLM 剪枝至高稀疏度的缺失秘诀”的官方 Pytorch 实现 - luuyin/OWL
- [来自 Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1709551516577902782?s=20,)：语言模型是否有内部世界模型？时间感？在多个时空尺度上？在与 @tegmark 合作的新论文中，我们通过发现一张字面意义上的世界地图，证明了它们确实拥有...
- [来自 Wes Gurnee (@wesg52) 的推文](https://x.com/wesg52/status/1709551591425245338?s=20)：但模型实际上是否*使用*了这些表示？通过寻找与探测器权重相似的神经元，我们发现了许多对事件的时空坐标敏感的空间和时间神经元...
- [神经语言模型中含义的隐式表示](https://arxiv.org/abs/2106.00737)：神经语言模型的有效性是完全源于对表面词汇共现统计的准确建模，还是这些模型对其描述的世界进行了表示和推理？...
- [来自 Joseph Sarnecki (@JosephSarnecki) 的推文](https://x.com/JosephSarnecki/status/1758541761495159011?s=20)：@wesg52 @tegmark 你能否测试 LLM 是否对其他激活执行相同的操作——特别是在想象的场景中（即：角色扮演）？我很好奇 LLM 是否创建了一个内部世界...
- [ikawrakow 提交的 1.5 bit 量化 · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453)：此草案 PR 是一个正在进行中的工作，展示了 1.5 bits-per-weight (bpw) 量化。目前仅 CUDA 可用，其他支持的后端尚未实现。CUDA、AVX2 和 ARM_NEON 已实现...

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1208098652493451355) (54 条消息🔥): 

- **寻求 Adam 的 RAM 效率**: `@hexani` 提出了一个关于使用 Adam 微调 7B 模型所需 RAM 量的问题，因为其内存需求很高（8 倍权重副本）。他们还询问了减少 RAM 使用的方法，但在给定的消息中未收到直接回复。
- **Batch Size 查询未获解答**: `@hexani` 询问了在微调模型时，使用 4090 或 H100 等 GPU 进行训练的典型 Batch Size，但该问题未得到解答。
- **LLM 弃 Jax 转 PyTorch**: `@carsonpoole` 提到他们正在将一个 LLM 从 Jax 格式转换为 PyTorch 格式，并称这一过程令人不悦。
- **Axolotl 微调教程请求未获满足**: `@pncdd` 寻求一份关于使用 axolotl 进行微调的完整分步教程，涵盖从数据集处理到执行的全部流程，但未出现包含此类指南的回复。
- **长文本生成挑战**: `@benh.1984` 征求关于使用 LLMs 生成超长文本（约 20,000 tokens）的建议，但未获成功；`.ben.com` 建议禁用 end-of-sentence token 以鼓励持续生成，尽管这在模型想要结束时可能会导致质量下降。

**提到的链接**:

- [WordNet Search - 3.1](http://wordnetweb.princeton.edu/perl/webwn?c=8&sub=Change&o2=&o0=1&o8=1&o1=1&o7=&): 未找到描述
- [WordNet Search - 3.1](http://wordnetweb.princeton.edu/perl/webwn?c=8&sub=Change&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&i=0&h=1000&s=giraffe): 未找到描述

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1208062922228633610) (5 条消息): 

- **对论文实现的兴趣**: `@qtnx` 对实现最近一篇论文中的方法论表现出热情，尽管他们对*如何实现*存在一些疑问。
- **认出同行**: `@vatsadev` 在频道中发现 `@qtnx` 时表达了惊讶和钦佩，称其为 **Absolute legend**。
- **合作机会**: `@qtnx` 通过 Discord ID (`<@282315082749444097>`) 联系了某人，建议在论文实现上进行协作，尽管具体流程尚不确定。
  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1207983642031423498) (368 messages🔥🔥): 

- **Sora 的视频编辑能力讨论**：用户 `@max_voltage` 强调了 *Sora* 的多功能性，指出它能够利用图像或视频作为提示（prompt）来完成一系列编辑任务，他认为这是 DALL-E 等工具所缺乏的功能。

- **V-JEPA 与 AI 模型开发**：`@max_voltage` 分享了一篇关于 Meta 的 **V-JEPA** 模型的[文章链接](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)，讨论了其在 AI 领域的重要性，并引用了其他相关的研究文章。随后与 `@twoabove` 等人的讨论探讨了不同模型在处理 pipeline 和目标（objectives）方面的差异。

- **Midjourney v6 批评**：用户 `@pseudoterminalx` 对最新的 Midjourney 更新表示不满，批评了生成图像的质量和构图。

- **AMD Pervasive AI 开发者竞赛**：`@itali4no` 重点介绍了 **AMD Pervasive AI Developer Contest**，奖项涵盖 Generative AI、Robotics AI 和 PC AI 类别，随后引发了关于使用 AMD GPU 进行 AI 工作的可行性和可用性的对话，参与者包括 `@chad_in_the_house`、`@pseudoterminalx` 和 `@drhead`。

- **LAION 社区行为**：在消息末尾，`@thejonasbrothers` 提到感觉 LAION 社区存在毒性（toxicity），促使 `@mega_b` 呼吁进行更友好和建设性的互动，特别是对待新人。

**提到的链接**：

- [no title found](https://huggingface.co'): 未找到描述
- [FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model](https://arxiv.org/abs/2303.09833)：最近，条件扩散模型因其卓越的生成能力在众多应用中受到欢迎。然而，许多现有方法需要训练。它们需要……
- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/)：介绍一款具有下一代伦理对齐能力的新型 AI 模型。现在开始聊天。
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA>)：SD.Next：Stable Diffusion 及其他基于扩散的生成图像模型的高级实现 - 创建新页面 · vladmandic/automatic Wiki
- [Sora could ruin peoples lifes](https://community.openai.com/t/sora-could-ruin-peoples-lifes/635220)：你们将终结许多人的职业生涯。摄影师、艺术家、动画师、电影制作人，甚至可能是演员。在这些行业立足已经很难了，现在有了这个，人们可能会……
- [Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023#challengeNav)：利用 AMD 推动突破性创新。
- [kakaobrain/align-base · Hugging Face](https://huggingface.co/kakaobrain/align-base)：未找到描述
- [no title found](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)：未找到描述
- [AI x Mental Health Happy Hour · Luma](https://lu.ma/obvb5mzw)：Slingshot 正在构建心理学基础模型，以帮助在全球范围内扩大心理健康服务的普及。欢迎加入我们的 AI x 心理健康欢乐时光。地点待定，位于 Central……
- [GitHub - HighCWu/control-lora-v2: ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2](https://github.com/HighCWu/control-lora-v2)：ControlLoRA 第 2 版：一个用于控制 Stable Diffusion 空间信息的轻量级神经网络第 2 版 - HighCWu/control-lora-v2
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM)：通过在 GitHub 上创建账号为 LargeWorldModel/LWM 的开发做出贡献。
- [lllyasviel - Overview](https://github.com/lllyasviel)：张吕敏 (Lyumin Zhang)。lllyasviel 拥有 40 个公开仓库。在 GitHub 上关注他的代码。
- [Allen T (@Mr_AllenT)](https://nitter.mint.lgbt/Mr_AllenT/status/1758839836021002470?t=REnUyTKEsWpzPqTx-JnMZw&s=19>)：万一你的手机在过去 48 小时内坏了：OpenAI 团队自正式发布以来一直在发布新的 Sora 视频。这是发布在 X 上的 10 个令人惊叹的 Sora 视频：

  

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1208073505543946251) (27 messages🔥): 

- **HDiT 图像生成突破**：`@chad_in_the_house` 分享了关于 **Hourglass Diffusion Transformer (HDiT)** 演示的公告，该模型具有与像素数呈线性缩放的特性，有可能为高分辨率下的扩散模型设定新的 SOTA。 [论文已在 arxiv 上发布](https://arxiv.org/abs/2401.11605)。
  
- **Offset Noise 中的奇怪蓝调**：`@chad_in_the_house` 询问在进行 Offset Noise 时输出图像出现蓝调是否正常，`@thejonasbrothers` 简单地回答了“不”。

- **质疑 LaION 数据库的完整性**：`@vrus0188` 提到了 [Reddit 上的反馈](https://www.reddit.com/r/StableDiffusion/comments/1ata8gw/feedback_on_base_model_releases/)，该反馈批评了 Stable Cascade 使用的 LaION 数据库存在过度审查和图像标注质量差等问题，引发了关于图像模型训练实践和社区毒性的讨论。

- **图像建模中的美学评分批评**：`@drhead` 对图像模型中应用的美学评分方法表示怀疑，认为这可能导致生成虽然普适悦目但缺乏变化的输出，类似于 Midjourney 的方法，并提出应对图像质量的构成有更细致的理解。

- **合成数据和视频建模的前景**：`@unjay.` 和 `@spirit_from_germany` 讨论了合成数据在视频建模中的潜力，引用了 Sora 的结果作为基准，并提议启动一个项目，通过从现有或生成的 3D 场景的不同视角创建摄像机图像来生成大型数据集。

**提到的链接**：

- [Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers](https://arxiv.org/abs/2401.11605)：我们提出了 Hourglass Diffusion Transformer (HDiT)，这是一种图像生成模型，表现出与像素数的线性缩放，支持直接在高分辨率（如 $1024 \times 1024$）下进行训练...
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1ata8gw/feedback_on_base_model_releases/)：未找到描述

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1207960486398918741) (131 messages🔥🔥): 

- **GPT-3 Token 和视频 Patch 讨论**：`@lugui` 澄清说，在 GPT 上，数据计费的单位是 Token，而 Patch 是视频格式的等效概念。他们认为一旦 Patch 概念发布使用并定价，就会变得更加清晰。
- **对 Sora 影响创造力的担忧**：`@thedasenqueen` 表达了对像 Sora 这样的 AI 终结创造力和创作技能的担忧，`@infidelis` 则反驳认为问题在于 AI 的 NPC（非玩家角色）用户，而非技术本身。`@bambooshoots` 还提到，历史上关于技术抑制创造力的担忧总是被证明是错误的。
- **用户对 GPT 和 Gemini 性能看法分歧**：`@redstone12345` 发起了比较 GPT 和 Gemini 的讨论，`@infidelis` 指出 Gemini 具有更像人类的行为和创造力，而 GPT 在结构化任务中表现出色；`@exx1` 还补充说 GPT-4-Turbo 在推理和反思方面优于 Gemini Ultra。
- **在社交媒体上尝试 AI**：`@fai.hunter` 想使用 ChatGPT 来管理社交媒体页面并以个人风格回复 Instagram 私信（DMs），暗示了一个模仿个人互动的项目。`@eskcanta` 指出了在此类尝试中遵守 OpenAI 服务条款的重要性，`@reynupj` 建议拥有大量的个人消息存档以实现成功的模仿。
- **关于 AI 生成内容的法律和伦理讨论**：关于使用 Sora 等 AI 创建衍生作品的对话中，`@johnnyrobert` 认为私人、非商业用途可能不会在法律上影响原作品的创作者。`@eskcanta` 讨论了 OpenAI 法律责任的复杂性，以及这些责任如何影响用户创建的 AI 生成内容。

**提到的链接**：

- [Terms of use](https://openai.com/policies/terms-of-use)：未找到描述
- [Usage policies](https://openai.com/policies/usage-policies)：未找到描述
- [OSWeb](https://jatos.it.ntnu.no/publix/OG92k9q7KYc)：未找到描述

  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1208021140832788551) (142 条消息🔥🔥): 

- **Flask 服务器问题**：`@will_b_mora` 在尝试将 actions 与 GPT 配合使用时遇到错误，收到消息 "Server URL is not under the root origin; ignoring it"。`@elektronisade` 澄清说，服务器地址必须是公开的，且 OpenAI 的服务可以访问。
  
- **自定义 GPT 保存故障**：`@sines303` 由于 "FatalServerError" 无法保存并发布自定义 GPT。当存在较大规模的基于 JSON 的知识库时，会出现此问题。

- **GPT 速度变慢与响应质量问题**：多位用户（如 `@bigskippa`、`@teamsettle_04535` 和 `@silentsushix3`）报告称，与 3.5 版本相比，GPT-4 的响应速度变慢且质量下降，观察到显著的速度退化和错误。

- **内容政策困惑**：`@bazilb` 询问了在向 GPT 咨询成人内容时收到内容政策警告的影响，询问其账号是否会有风险。`@eskcanta` 建议阅读 OpenAI 的政策以获取澄清，并分享说如果没有违反规则，通常账号应该是安全的。

- **GPT-4 输出长度建议**：`@iamrobertandrews` 和 `@darthgustav.` 等用户讨论了生成长文本内容的策略，例如分章节规划并分配明确任务，或使用模板引导 GPT-4 的输出。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1209015855849938986)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [未找到标题](https://lambdalabs[dot]com/service/gpu-cloud#pricing)：未找到描述
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1202309673709994065)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [AI 系统应该如何表现，由谁决定？](https://openai.com/blog/how-should-ai-systems-behave)：我们正在澄清 ChatGPT 的行为是如何塑造的，以及我们改进该行为、允许更多用户自定义以及在这些领域获取更多公众意见的计划。
- [使用政策](https://openai.com/policies/usage-policies)：未找到描述
- [使用条款](https://openai.com/policies/terms-of-use)：未找到描述
- [使用政策](https://web.archive.org/web/20231101074011/https://openai.com/policies/usage-policies)：未找到描述

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1207965884602392597) (54 条消息🔥): 

- **分类难题**：用户 `@ben.30` 表达了对公司分类名称的挑战，考虑合并相关类别，以简化 GPT 在处理与废物管理和垃圾箱租赁（skips）相关的服务请求时的理解。

- **AI 训练技巧**：`@eskcanta` 提供了关于训练 AI 的全面建议，强调了教导 AI 异常情况和边缘案例（edge cases）的重要性，就像培训新的人类员工一样，并建议避免因 AI 犯错而责备它，应专注于期望的输出。

- **摘要中的 PII 处理**：`@best.value` 和 `@madame_architect` 讨论了在摘要之前脱敏个人身份信息（PII）的挑战，建议使用 Python 库进行 PII 检测，并避免直接让 AI 检测 PII。

- **LLM JSON 格式化困扰**：`@neeagl` 在确保 GPT-3.5 模型输出格式一致的 JSON 时遇到问题，通过使用示例响应并避免导致错误的对话历史记录解决了该问题。

- **Prompt 编写策略**：`@elegante94` 询问了一个能将原始 Prompt 细化为有效 AI 任务指令的 GPT 工具，`@queueh` 分享了一个详细的、迭代式的 Prompt 构建策略，重点在于 AI 与用户之间的协作以微调 Prompt。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1019652163640762428)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1208676441101963295)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。

  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1207965884602392597) (54 messages🔥): 

- **对非结构化分类的挫败感**：`@ben.30` 讨论了在 Non Mechanical 和 Waste Management 等类别中对工单进行分类时面临的挑战，并提到由于很难根据客户语言区分跳卸式垃圾箱 (skips) 和废物类别，计划可能会合并这两个类别。

- **用类人培训方式教导 AI**：`@eskcanta` 主张使用类似于培训人类新员工的方法来教导 AI，强调在处理复杂或异常任务时，培训中的清晰度和强化至关重要。

- **针对 PII 的文本预处理**：`@best.value` 探讨了在大型非结构化文本中处理个人身份信息 (PII) 的困境，以及 AI 在脱敏任务中的局限性。Madame_architect 和 `@exhort_one` 建议在摘要之前使用特定的 PII 检测工具，并强调了在 Prompt 中明确意图的重要性。

- **Prompt Engineering 的技艺**：`@queueh` 分享了一个复杂的 Prompt，旨在帮助构建用于 GPT 的其他有效 Prompt。同时，`@elegante94` 询问是否有可以完善原始 Prompt 构思的 AI，`@eskcanta` 建议直接与 ChatGPT 交流以进行 Prompt 优化。

- **知识库检索问题**：`@pawjwp` 和 `@d1scobo1` 报告了 GPT 在使用知识库时的问题，模型要么拒绝查阅提供的数据，要么在同一对话线程的后续问题中准确性下降。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1019652163640762428)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1208676441101963295)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1207962460011761674) (227 messages🔥🔥): 

- **Turbo 困惑已澄清**：`@brknclock1215` 似乎对 Perplexity 神秘的 "turbo" 功能在移动端截图和网页版之间的差异感到困惑。`@gooddawg10` 澄清说他们在移动端使用 Kiwi Browser 时看到了 "turbo" 标签，这引发了关于不同平台用户界面可能存在差异的讨论。
  
- **Turbo 在哪里？**：`@gooddawg10` 对其移动设备上提到的 "turbo" 感到困惑，而 `@brknclock1215` 和 `@Perplexity AI` 加入了聊天，承认了关于 turbo 功能显示不一致的讨论。

- **API 性能讨论**：`@brknclock1215` 询问了在 Perplexity 上选择 "turbo" 模型后的性能差异，而 `@gooddawg10` 因为在工作尚未进行测试。`@brknclock1215` 对这次交流表示感谢，尽管没有提供关于性能差异的具体答案。

- **Gemini 1.5 的营销策略分析**：`@archient` 预测，作为发布后两个月免费访问期后的战略举措，Google 可能会在两个月内开放 Gemini 1.5 的访问权限。

- **应用定价结构咨询**：`@retonq` 寻求关于 Perplexity 专用定价结构的澄清，`@icelavaman` 解释说 Perplexity Pro 采用的是基于使用量的定价模型，详情可在 Perplexity 定价文档页面找到。
  
- **Discord 机器人困惑**：用户 `@brickpotato` 和 `@spectralruler` 询问了 Perplexity AI Discord 机器人的状态。`@icelavaman` 澄清该机器人已经关闭。

**提到的链接**：

- [定价](https://docs.perplexity.ai/docs/pricing)：未找到描述
- [Gemini 应用的功能及其他常见问题](https://gemini.google.com/faq?hl=en#citation)：了解 Gemini 的功能、工作原理以及获得访问权限的不同方式。
- [什么是 Perplexity Pro？](https://blog.perplexity.ai/faq/what-is-perplexity-pro)：浏览 Perplexity 的博客，获取文章、公告、产品更新和优化体验的技巧。保持资讯同步，充分利用 Perplexity。
- [Gemini 1.5 与 AI 的最伟大之夜](https://youtu.be/Cs6pe8o7XY8)：自 GPT-4 发布以来 AI 领域最重要的一天。一个新的 SOTA 模型 Gemini 1.5 已经到来，就在同一个晚上，震撼的文本转视频模型 Sora 也发布了...

  

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1207994716403142747) (38 messages🔥): 

- **重复发帖并不讨喜**：用户 `@brknclock1215` 回应了自己关于重复发帖的问题，幽默地暗示模仿可能是最高形式的奉承。
- **对 LLM 联网访问的好奇**：`@bishal_saha` 询问了像 Perplexity 这样的语言模型访问网络的流程，引发了详细讨论并分享了资源。
- **揭开联网访问奥秘的链接**：针对 `@bishal_saha` 的好奇，`@brknclock1215` 分享了一篇关于自然语言处理模型的 [Retrieval-Augmented Generation (RAG) 的 NeurIPS 论文](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)。
- **频道发帖的维护与指南**：`@icelavaman` 介入提醒大家，sharing 频道主要用于分享 Perplexity 线程，并将用户引导至相应的频道进行一般性讨论和提问。
- **关于 Perplexity API 及其功能的讨论**：分享了与 Perplexity 使用相关的链接，`@brknclock1215` 特别回答了 `@bishal_saha` 关于 Perplexity 的 `pplx` 模型在获取最新网络信息方面的独特性的疑问。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1208016236877840404)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1206373264100696094)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，与你的朋友和社区保持紧密联系。
- [HuggingChat](https://huggingface.co/chat/)：让社区最好的 AI 聊天模型惠及每一个人。
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)：首创的 Online LLM API。

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1208142952212205578) (27 messages🔥): 

- **API 速率限制提升请求未获回复**：用户 `@enelemtal` 表达了挫败感，因为向 api@perplexity.ai 发出的提高 API 速率限制的请求一直没有得到回应。`@ok.alex` 随后回复，请 `@enelemtal` 私信邮箱以便核查，而 `@bitshift_` 和 `@bvfbarten.` 表示他们也遇到了同样的问题。

- **透明引用的重要性**：针对 `@retonq` 的提问，`@ai_made_approachable` 强调了在专业用途中使用 "pplx-online" 模型时引用的重要性，并指出了目前由于其“黑盒 (blackbox)”性质而存在的局限性。`@me.lk` 保证正在努力为未来获批的使用场景在 API 中添加引用支持。

- **对流式 API 中异常字符的困惑**：`@boyn_` 报告在使用 `pplx-70b-online` 模型时收到了令人困惑的字符（如 `00` 和 `2\n`），寻求关于发生这种情况的原因的澄清。`@thedigitalcat` 询问何时能看到针对这些混乱结果的更新，`@icelavaman` 确认团队已知晓并正在处理该问题。

- **模型弃用与生命周期查询**：考虑到 `llama-2-70b-chat` 的弃用，`@rehmatsg` 询问了一个模型（如 `mixtral-8x7b-instruct`）在被弃用前的典型支持时长。

- **设置 Perplexity API 端点**：在 `@nettemple` 寻求关于集成所需正确 'apiUrl' 的指导后，`@icelavaman` 提供了官方端点 `https://api.perplexity.ai/chat/completions` 以及文档引用。随后的消息为 `@nettemple` 提供了代码协助，最终成功实现并对提供的帮助表示感谢。

**提到的链接**：

- [no title found](https://api.perplexity.ai')：未找到描述
- [no title found](https://api.perplexity.ai';)：未找到描述
- [Moon (Dark Mode)](https://docs.perplexity.ai)：未找到描述
- [Chat Completions](https://docs.perplexity.ai/reference)：未找到描述

  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1207964370882007051) (144 条消息🔥🔥): 

- **寻求 LLMs 微调指导**：用户 `@keshav._._.` 表达了需要帮助将印度 IPS 法律的 PDF 文件转换为适合微调 llama 2 模型的格式。相反，另一位用户 `@vishyouluck` 建议针对当前任务使用 RAG 方法而不是微调。

- **关于多人游戏开发的咨询**：`@om7059` 寻求关于在他们计划开发的多人涂鸦游戏中加入模型评估的建议，在该游戏中，涂鸦将在时间耗尽后由模型进行评分。

- **寻找最适合地理数据的模型**：`@retonq` 好奇在 Mistral medium、pplx 和 llama 中，哪种模型最擅长理解坐标和方向等地理信息。

- **寻找支持波斯语的开源 LLMs**：`@alifthi` 正在寻找一种高性能、支持波斯语且类似于 ChatGPT 的开源语言模型。另一位用户 `@alchemist_17.` 建议，任何开源模型（如 mistral 或 llama2）都可以使用自定义数据集进行微调。

- **深入探讨语言模型基础**：包括 `@vipitis`、`@tea3200` 和 `@doctorpangloss` 在内的多位用户参与了讨论，澄清了语言模型主要预测下一个词序列的概率，其中一些模型能够通过多样化的任务展现知识和推理能力。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1209022297269080136)：Discord 是与朋友和社区通过语音、视频和文字交流的最简单方式。聊天、聚会并与您的朋友和社区保持紧密联系。
- [LLM Visualization](https://bbycroft.net/llm)：未找到描述
- [Join the Hugging Face Discord Server!](https://discord.gg/hugging-face-879548962464493619?event=1203285706949009448)：我们正致力于民主化优秀的机器学习 🤗 验证以链接您的 Hub 和 Discord 账户！ | 70607 名成员
- [Best Image Models V2 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2)：未找到描述
- [bigcode/the-stack · Datasets at Hugging Face](https://huggingface.co/datasets/bigcode/the-stack)：未找到描述
- [codeparrot/github-code · Datasets at Hugging Face](https://huggingface.co/datasets/codeparrot/github-code)：未找到描述
- [Google Colaboratory](https://colab.research.google.com/drive/1i6cDDsZfGB70fNgxiUxbUufeQfaWyCfd?usp=sharing)：未找到描述
- [Models - Hugging Face](https://huggingface.co/models)：未找到描述
- [OpenAI&#39;s Agent 2.0: Excited or Scared?](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx)：我想为您全面介绍浏览器/移动端/桌面端 AI agents。获取免费的 HubSpot 电子书：使用 Generative AI 扩展您的内容运营：https://c...
- [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b)：未找到描述
- [meta-llama/Llama-2-7b-chat-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)：未找到描述
- [Build software better, together](https://github.com/search?q=language%3Apython&type=repositories)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
- [GitHub - LegallyCoder/mamba-hf: Implementation of the Mamba SSM with hf_integration.](https://github.com/LegallyCoder/mamba-hf)：带有 hf_integration 的 Mamba SSM 实现。 - LegallyCoder/mamba-hf
- [Google Colaboratory](https://colab.research.google.com/drive/1ONevcH1oHOdm4F6DPgju_WyUWLVyIY6b?usp=sharing)：未找到描述
- [Build software better, together](https://github.com/search?q=language%3Ajava&type=repositories)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1208012278209847356) (8 条消息🔥): 

- **代码质量的抄袭检测工具**：`@brady_kelly` 分享了一种检测被忽视的**样板文档 (boilerplate documentation)** 的深刻方法。他们建议在**软件 CI/CD 流水线**中使用类似于抄袭检测的过程，以确保所有文档都已正确完成。
  
- **假期生产力报告**：`@antiraedus` 概述了他们的假期成就，包括专注于健身、增重，并为大学学期做准备，目标包括辅导、社交以及开发侧边项目（如一个 Flutter 游戏，无论质量如何）。

- **GitHub Copilot 的引擎揭晓？**：`@rcdpge` 提到他们了解到 **GitHub Copilot** 的代码建议可能运行在 **ChatGPT 3.5 Turbo** 上，尽管 **@vipitis** 对其内联补全的质量提出了质疑。

- **提示词驱动的 RAG 系统博客文章**：`@subham5089` 邀请成员阅读他们的 LinkedIn 博客文章，讨论了与**提示词驱动的 RAG 系统**相关的挑战和潜在解决方案。该文章探讨了如何改进这项新兴技术。[阅读博客文章](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7)。

- **强化学习与语言模型讲座**：`@nagaraj_arvind` 分享了一个关于 **RLHF (Reinforcement Learning from Human Feedback)** 以及一种名为 **DPO** 的 PPO 替代方案的讲座视频。内容适合那些有兴趣通过 RLHF 增强 LLM 补全效果的人。[观看讲座视频](https://youtu.be/Ju-pFJNfOfY) 并 [阅读 DPO 论文](https://arxiv.org/abs/2305.18290)。

**提到的链接**：

- [RLHF, PPO and DPO for Large language models](https://youtu.be/Ju-pFJNfOfY)：大语言模型的 RLHF、PPO 和 DPO 介绍：强化学习、RLHF、近端策略优化 (PPO) 和直接偏好优化 (DPO) 算法简介。
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)：虽然大规模无监督语言模型 (LMs) 学习了广泛的世界知识和一些推理技能，但由于完全无监督，实现对其行为的精确控制非常困难……

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1208081421025935400) (17 条消息🔥): 

- **理解蛋白质语言模型 (Protein Language Models)**：`@grimsqueaker` 分享了关于蛋白质语言模型 (PLMs) 在结构预测之外的局限性的见解。论文《Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models》通过 370 次实验得出结论：预训练虽有帮助，但并非所有任务都能随着计算量的增加而扩展，因此需要新的预训练方法（[阅读摘要](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1)，[在 Twitter 上讨论](https://twitter.com/KevinKaichuang/status/1755672999166972319)）。

- **Intel 发布文本转 3D 模型转换器**：`@abhinit21` 重点介绍了 Intel 的新模型 `LDM3D-VR`，该模型能够将文本转换为 3D 内容，专注于虚拟现实 (VR) 开发（[Hugging Face 上的模型](https://huggingface.co/Intel/ldm3d-pano)，[阅读论文](https://arxiv.org/pdf/2311.03226.pdf)）。

- **使用 Web 应用检测 Deepfake 人脸**：`@lucas_selva` 推广了他们的 Web 应用，该应用利用 XAI 来识别 Deepfake 图像，并表示当前模型准确率为 88%，并计划在未来进行训练增强（[尝试应用](https://deep-fake-generated-people-facial-recognition.streamlit.app/)）。

- **关于增强 AI 人脸识别的讨论**：在与 `@hrishimax` 的对话中，`@lucas_selva` 讨论了用于检测 AI 生成人脸模型的局限性和未来改进计划，包括扩大训练数据集和应用迁移学习 (Transfer Learning)。

- **Databricks：加速 AI 基础设施**：`@valeriiakuka` 分享了一篇文章，讨论了 Databricks 对生成式 AI 领域的影响及其在近期收购背景下的战略。文章展示了 Databricks 的发展轨迹以及在 AI 行业潜在的增长方向（[阅读全文](https://www.turingpost.com/p/databricks)）。

**提到的链接**：

- [MotionCtrl SVD - TencentARC 的 Hugging Face Space](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD)：未找到描述
- [Intel/ldm3d-pano · Hugging Face](https://huggingface.co/Intel/ldm3d-pano)：未找到描述
- [Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157)：强化学习 (RL) 在学习决策任务策略方面展现了卓越的能力。然而，RL 经常受到样本效率低、缺乏可解释性等问题的阻碍...
- [Google Colaboratory](https://colab.research.google.com/drive/1i6cDDsZfGB70fNgxiUxbUufeQfaWyCfd?usp=sharing)：未找到描述
- [LargeWorldModel/LWM-Text-Chat-1M · Hugging Face](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M)：未找到描述
- [Large World Models](https://largeworldmodel.github.io/)：未找到描述
- [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268)：目前的语言模型在理解难以用文字描述的世界方面存在不足，并且在处理复杂的长篇任务时表现吃力。视频序列提供了宝贵的时序信息...
- [Databricks: the Future of Generative AI in the Enterprise Arena](https://www.turingpost.com/p/databricks)：探索 Databricks 不寻常的历史、它对企业级生成式 AI 领域的贡献，以及该公司的战略和对 AI 行业的愿景。
- [Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1)：大型预训练蛋白质语言模型 (PLMs) 通过迁移学习提高了从序列预测蛋白质属性和结构的能力，其中 PLMs 的权重和表示被重新利用...
- [Proof Wallis Product using integration - Art Of Mathematics](https://mathematicsart.com/solved-exercises/proof-wallis-product-using-integration/)：使用积分证明沃利斯乘积 (Wallis Product) 首页 -> 已解问题 -> 沃利斯乘积 解法 考虑 (J_{n}=int_{0}^{frac{pi}{2}}
- [no title found](https://deep-fake-generated-people-facial-recognition.streamlit.app/)：未找到描述

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1208013684765491280) (10 messages🔥): 

- **向创作者致敬**：`@noir_bd` 对一项作品表示了钦佩，并表现出学习其制作方法的兴趣，同时将功劳归于 `<@848983314018336809>`。`@tony_assi` 用一个酷炫的拥抱表情回应了赞赏。
  
- **发布新的 AI 模型**：`@myg5702` 宣布在 [FumesAI Best-Image-Models-V2](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2) 的集合中新增了包括 *OpenDalle 1.1*、*Kandinsky 2.2* 等在内的模型，`@bean217` 对此表示支持。

- **介绍 Fluently，一个新的 Diffusion 模型**：`@ehristoforu` 分享了新 Diffusion 模型 **Fluently** 在包括 Hugging Face、CivitAI 以及 ZeroGPU Demo 在内的多个平台的链接。后续消息为 [Fluently V1](https://huggingface.co/ehristoforu/Fluently-v1) 生成的图像提供了更多上下文信息。

- **ProteinBERT 的亮眼发布**：`@grimsqueaker` 介绍了 **ProteinBERT**，详细说明了其在非传统领域的新颖架构和效率。提供了原始基于 Keras 的 [GitHub repo](https://github.com/nadavbra/protein_bert) 和由 LucidRains 开发的 PyTorch 移植版链接，以及相关的[研究论文](https://doi.org/10.1093/bioinformatics/btac020)和最近上传的未经测试的 Hugging Face 模型权重。

- **LocalLlm 托管 API 项目亮相**：`@typoilu` 展示了他们在免费 Google Colaboratory 环境中托管 API 的项目，旨在方便进行大型开源 LLM 的实验。他们邀请大家对 [LocalLlm GitHub repository](https://github.com/groloch/LocalLlm) 提供反馈。

**提到的链接**：

- [Best Image Models V2 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2)：未找到描述
- [GitHub - groloch/LocalLlm: Drop-in and advanced solutions to experiment with open source LLM !](https://github.com/groloch/LocalLlm)：用于实验开源 LLM 的即插即用及高级解决方案！- groloch/LocalLlm
- [ProteinBERT: a universal deep-learning model of protein sequence and function](https://doi.org/10.1093/bioinformatics/btac020)：摘要总结。自监督深度语言建模在自然语言任务中取得了前所未有的成功，最近已被重新应用于生物学领域。
- [GitHub - nadavbra/protein_bert](https://github.com/nadavbra/protein_bert/tree/master)：通过在 GitHub 上创建账户来为 nadavbra/protein_bert 的开发做出贡献。
- [GitHub - lucidrains/protein-bert-pytorch: Implementation of ProteinBERT in Pytorch](https://github.com/lucidrains/protein-bert-pytorch)：ProteinBERT 的 PyTorch 实现。通过在 GitHub 上创建账户来为 lucidrains/protein-bert-pytorch 的开发做出贡献。
- [GrimSqueaker/proteinBERT · Hugging Face](https://huggingface.co/GrimSqueaker/proteinBERT)：未找到描述

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1207967084370198589) (58 messages🔥🔥): 

- **PEFT 集成演示倒计时**：`@prateeky2806` 确认将于 3 月 1 日星期五进行演示，讨论 PEFT 库中不同合并方法的集成，并计划包含一个 Demo。相关的 PR 可以在 [GitHub](https://github.com/huggingface/peft/pull/1364) 上找到。
- **踊跃参与读书会**：读书会环节引发了积极反响，`@tonic_1` 和 `@chad_in_the_house` 对即将到来的演示表示期待，而 `@samx9128` 则表达了参加首次读书会的热情。
- **解决 Discord 技术问题**：包括 `@chad_in_the_house`、`@lunarflu` 和 `@tea3200` 在内的成员帮助 `@ericauld` 等人解决了在 Mamba 演示期间遇到的技术困难，引导他们进入正确的频道并获取权限。
- **YouTube 上的 Mamba 资源**：`@ericauld` 等人分享了讨论 Mamba 和 SSMs 的宝贵 YouTube 资源，这些资源可以帮助正在学习或无法参加现场讨论的人加深理解（[Samuel Albanie 的视频](https://www.youtube.com/watch?v=ouF-H35atOY&t=305s&ab_channel=SamuelAlbanie)，[Umar Jamil 的视频](https://www.youtube.com/watch?v=8Q_tqwpTpVU&ab_channel=UmarJamil)，[一个汇编播放列表](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe)）。
- **Mamba 论文争议与期待的演讲**：`@chad_in_the_house` 从审稿人的角度强调了 Mamba 论文被拒稿的问题，并指出了未来由 `@1191190979580022875` 关于“神经电路图（Neural Circuit Diagrams）”的演示。往期读书会演示的 GitHub 仓库见[此处](https://github.com/isamu-isozaki/huggingface-reading-group)。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/907325990236213288)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/879548962464493619/1208115157121896519)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1203285086624157696)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Mamba and SSM](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe)：未找到描述
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)：基础模型（Foundation models）现在驱动着深度学习中大多数令人兴奋的应用，它们几乎普遍基于 Transformer 架构及其核心的 attention 模块。许多亚二次时间（subquadratic-time）的...
- [Paper page - Neural Circuit Diagrams: Robust Diagrams for the Communication, Implementation, and Analysis of Deep Learning Architectures](https://huggingface.co/papers/2402.05424)：未找到描述
- [Tweet from FxTwitter / FixupX](https://x.com/vtabbott)：抱歉，该用户不存在 :(
- [Tweet from Vincent Abbott | Deep Learning (@vtabbott_)](https://x.com/vtabbott_/status/1743204563015102594?s=20)：刚刚根据 @reach_vb 的建议，在 @huggingface 上发布了关于 @MistralAI 的 Mixtral 图表的博客。刚设置好我的 HF 账号 - 欢迎关注。这似乎是一个参与其中的绝佳平台...
- [Hugging Face Reading Group 13: Mamba](https://www.youtube.com/watch?v=CWQuL8dpCRY)：演讲者：Eric Auld
- [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv&referrer=%5Bthe%20profile%20of%20Xinlei%20Chen%5D(%2Fprofile%3Fid%3D~Xinlei_Chen1))：本论文表明，驱动大型基础语言模型成功的掩码建模（masked-modelling）原理，可以通过在潜空间（latent space）中进行预测，有效地应用于视频。我们...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Paper Explained)](https://www.youtube.com/watch?v=9dSkvxS2EB0)：#mamba #s4 #ssm 大纲：0:00 - 简介 0:45 - Transformers vs RNNs vs S4 6:10 - 什么是状态空间模型（state space models）？ 12:30 - 选择性状态空间模型（Selective State Space Models） 17:55 - Th...
- [Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU&ab_channel=UmarJamil)：论文《Mamba: Linear-Time Sequence Modeling with Selective State Spaces》的解析。在本视频中，我将讲解 Mamba，一种新的序列建模架构...
- [Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY&t=305s&ab_channel=SamuelAlbanie)：Mamba 是由 Albert Gu 和 Tri Dao 提出的一种新型神经网络架构。时间戳：00:00 - Mamba - Transformer 的替代品？00:19 - 远程（Long Range）...
- [GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group](https://github.com/isamu-isozaki/huggingface-reading-group)：该仓库的目标是预编译 Hugging Face 读书会过去所有的演示文稿 - isamu-isozaki/huggingface-reading-group

  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1207985152605552640) (13 messages🔥): 

- **DPO 爱好者集结**：`@maxpappa` 提到使用全量 DPO，`@arturzm` 回应称这不是问题，而是“神级的速度提升”，并随口询问有什么新进展，未提供额外背景。
- **寻找单行模型转换代码**：`@blackbox3993` 寻求一种简单的单行代码，用于将任何模型转换为 BitsAndBytes 格式，并分享了一份[详尽的量化文档](https://huggingface.co/docs/bitsandbytes/main/en/quantization)。
- **BitsAndBytes 模型加载技巧**：`@gugaime` 向 `@blackbox3993` 解释说，要将自定义模型转换为 BitsAndBytes，应该替换特定的模块（如具有精确形状的 `Linear8bitLt`），并提供了代码片段来指导该过程。
- **寻求 Diffusion Model 见解**：`@sardarkhan_` 正在寻找资源以更深入地理解 Diffusion Model 的数学原理，感觉目前的复杂性让他难以招架；`@chad_in_the_house` and `@wandereronarock` 建议从 DDPM 论文和 Lillian Weng 的博客开始。
- **图像之外的 Diffusion Model**：`@little.stone` 询问 Diffusion 技术（如 diffusor 库中的 DDPM scheduler）在时间序列数据而非图像上的适用性，这表明其应用场景可能从标准用例发生转变。

**提及的链接**：

[Quantization primitives](https://huggingface.co/docs/bitsandbytes/main/en/quantization)：未找到描述

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1208155753815220234) (7 messages): 

- **训练 UI 元素查询**：`@conceptron` 对一个项目表示感兴趣，并询问了用于训练目的的 **UI 元素**（如按钮、复选框和滑块），但未提供更多细节或背景。
- **寻找构建 Sora 的社区**：`@andrew_ulterior` 询问是否有专门为构建自己版本 **Sora** 的人设立的 Discord 频道，未详细说明 Sora 指代什么。
- **修补天气预报模型**：`@smallcrawler` 正在寻求关于在自回归 Transformer 中**修正 patch 边缘效应**的建议，特别是针对一个随时间出现伪影的全球天气预报模型。
- **过滤 AI 图像伪影**：`@korner83` 正在进行一个需要识别并过滤掉 **AI 生成图像伪影**的项目，并询问了提高这一预筛选阶段效率的最佳实践、工具或模型。
- **Statricks 创始人提供数据和经验**：`@ironman5769` 讲述了他的公司 Statricks 的背景故事，以及他利用 **computer vision** 从广告中识别产品的历程，表示他拥有相关数据和兴趣来协助相关项目，尽管目前缺乏动力和资金。
- **检测图像伪影的方向**：针对过滤 AI 伪影的问题，`@ironman5769` 建议查看 Google 的 [Know Your Data](https://knowyourdata-tfds.withgoogle.com)，这可能是一个值得探索的方向。
- **请求 BLIP2 微调指导**：`@seanb2792` 正在寻求 **finetuning BLIP2** 的帮助，但未给出具体问题或请求背景的细节。

**提及的链接**：

[Know Your Data](https://knowyourdata-tfds.withgoogle.com)：未找到描述

  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1207986070743023686) (10 messages🔥): 

- **寻找用于代码检测的 ML 构建模块**：`@brady_kelly` 表达了对在 CI/CD pipeline 中检测样板代码（boilerplate code）和自动生成的 README 文件的兴趣。他们请求在高级概述层面提供基本概念的指导，而非具体的代码实现。

- **探索翻译模型架构**：`@manavsarkar` 发现使用 encoder-decoder 架构进行语言翻译的效果较差，并询问是否有替代方法。作为回应，`@calmdown.manu` 建议 decoder-only 架构以及使用 tags 来区分原始句子和翻译句子可能是一个解决方案。

- **翻译模型理解细微含义**：除了架构问题，`@manavsarkar` 还想知道翻译模型如何理解何时翻译跨语言发音相同的名词。`@calmdown.manu` 提到 pointer networks 和 attention 机制通常能学会处理这类细微差别。

- **对在小型模型上微调 WikiSQL 感兴趣**：`@miguelkjh` 询问了在小型项目中使用 GPT-2 或 Pythia 等模型在 WikiSQL 数据集上进行微调的经验，寻求关于挑战和性能提升的见解。

- **需要数据集去重工具**：`@abrahamowodunni` 请求推荐能够对大型数据集进行去重（deduplication）的工具。

- **Python 导入的技术问题**：`@vikas8715` 在尝试从 `transformers` 库导入 `is_torch_sdpa_available` 时遇到了 `ImportError`，这揭示了其环境中潜在的依赖问题或版本不兼容。
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1207985152605552640) (13 messages🔥): 

- **使用了全量 DPO**：`@maxpappa` 提到他们正在使用 **full DPO**，但未提供关于其经验或结果的更多上下文。
- **@gugaime 让 BitsAndBytes 变得简单**：为了将模型转换为 BitsAndBytes 格式，`@gugaime` 建议只需在加载模型时使用参数 `load_in_4bit=True` 或使用 `BitsAndBytesConfig`。他们还分享了用于更高级使用场景的代码片段。
- **在自定义模型中使用 BitsAndBytes**：对于不使用 `AutoModel` 的用户，`@gugaime` 建议 `@blackbox3993` 将其自定义模型的 linear module 替换为 BitsAndBytes 的 `Linear8bitLt`，并提供了一个代码示例。
- **寻求 Diffusion 模型知识**：`@sardarkhan_` 正在寻找资源以深入理解 Diffusion 模型的数学原理，因为他感到这些复杂性让人难以招架。其他用户推荐了 DDPM 论文和 Lillian Weng 的博客等资源以供参考。
- **时间序列 Diffusion 专家**：`@little.stone` 询问了如何将 diffusor 库与自定义网络应用于时间序列数据。他们对 DDPM scheduler 等函数与非图像模态的兼容性感到好奇。

**提到的链接**：

[量化原语 (Quantization primitives)](https://huggingface.co/docs/bitsandbytes/main/en/quantization)：未找到描述

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/) (1 messages): 

jerryjliu0: webinar 正在进行中！

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1208090704941944944) (9 条消息🔥): 

- **RAG 检索-评估循环 (RAG Retrieval-Evaluation Loop)**：一种提高 **RAG 检索质量** 的技术，涉及在合成（synthesis）之前使用 LLM 评估和过滤结果的相关性，正如一条推文引发的[该方法讨论](https://twitter.com/llama_index/status/1758530939276378255)中所述。
- **使用 RAG 进行逐步视频分析**：@lancedb 概述了**使用 RAG 进行视频分析**的过程，包括帧拆分、音频转录以及用于数据库检索的 Embedding，详情见这篇[博客文章](https://t.co/HmkMzF0c1n)，并在 [Twitter](https://twitter.com/llama_index/status/1758587997178728796) 上分享了令人印象深刻的结果。
- **端到端 ML 部署指南**：@DGallitelli95 撰写了一篇文章，演示了如何使用 **Huggingface 和 AWS Sagemaker 创建 RAG 流水线**，提供了从模型选择到端点（endpoint）创建的 ML 模型部署指南，并在 [Twitter](https://twitter.com/llama_index/status/1758654210378473731) 上进行了展示。
- **Nomic Embedding 模型的权衡灵活性**：**开源的 nomic-embed-text-v1.5 嵌入模型**允许在内存、存储、带宽和性能之间进行动态权衡，维度跨度从 64 到 768，灵感来自 Matryoshka Representation Learning。详情请参阅推文线程和 [Nomic 的博客](https://home.nomic.ai) ([推文](https://twitter.com/llama_index/status/1758901855508382149))。
- **基于 RAG 的餐厅菜单聊天机器人教程**：一篇 **@weights_biases 的博客文章**演示了如何使用 LlamaIndex 构建全栈餐厅菜单聊天机器人，其特色是通过 Weave 内置了应用使用日志记录，正如在 [Twitter 帖子](https://twitter.com/llama_index/status/1758965798377578688)中所宣布的那样。

**提到的链接**：

[Unboxing Nomic Embed v1.5: Resizable Production Embeddings with Matryoshka Representation Learning](https://t.co/8gBvrdxlov)：Nomic 推出了一款真正的开源文本 Embedding 模型，具有可调整大小的嵌入。

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1207985513428947014) (222 条消息🔥🔥): 

- **LlamaIndex 安装故障排除**：`@lapexer` 在安装 LlamaIndex 后遇到了导入错误，通过使用 Python 3.10 创建新环境并同时安装 `llama-index-core` 和 `llama-index` 解决了该问题。
- **对并行化 Embedding 的困扰**：`@ben25635` 对 LlamaIndex 相比 LangChain 缺乏原生并行化 Embedding 支持表示沮丧。`@cheesyfishes` 就增加批次大小（batch sizes）和使用 `IngestionPipeline` 进行并行处理提供了指导。
- **AzureOpenAI 与 LlamaIndex 0.10.x 的问题**：`@disco.dr` 在将 AzureOpenAI 与 LlamaIndex 0.10.x 配合使用时遇到了异常。通过确保将 `aclient` 和 `client` 同时传递给 `QdrantVectorStore` 并安装特定软件包，问题得以解决。
- **优化从 PDF 中提取信息**：`@gryhkn` 寻求关于从 PDF 报告中高效提取经济改革信息的建议。`@kapa.ai` 建议了一个利用 LlamaIndex 的数据连接器（data connectors）、索引（indexes）、引擎（engines）、数据代理（data agents）和集成的流程。
- **评估 ReAct Agent 和 QueryPipeline 的优势**：`@andysingal` 分享了一个链接，解释了在 LlamaIndex 中将自定义 AI 模型与 Ollama 等外部工具结合的优势。他们还向 `@vett93` 介绍了 mlx 和 openllm 等替代方案及其与 LM Studio 的关系。
- **升级 Llama-Index 软件包**：`@vett93` 询问了关于升级 Llama-Index 软件包的问题。`@cheesyfishes` 确认在 0.10.x 版本中，软件包是独立版本化的，通常只需要更新 `llama-index-core`。

**提到的链接**：

- [未找到标题](http://192.168.0.105:1234)): 未找到描述
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/]): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。
- [未找到标题](https://llamahub.ai/l/llama-packs/llama-index-packs-tables?from=all): 未找到描述
- [Documents / Nodes - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/root.html): 未找到描述
- [Ingestion Pipeline - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html#parallel-processing): 未找到描述
- [本地 Llama2 + VectorStoreIndex - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local.html): 未找到描述
- [app-llamaindex-v0.10.py](https://gist.github.com/sumvedshami/066e694fe25f51a317135e079b074115): GitHub Gist：即时分享代码、笔记和代码片段。
- [llama_index/llama-index-core/llama_index/core/download/pack.py at a900d5e67424c2b2c46b0aa9ef62502af556d449 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/a900d5e67424c2b2c46b0aa9ef62502af556d449/llama-index-core/llama_index/core/download/pack.py#L110-L117): LlamaIndex（前身为 GPT Index）是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index
- [用于高级 Text-to-SQL 的 Query Pipeline - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql.html#advanced-capability-1-text-to-sql-with-query-time-table-retrieval): 未找到描述
- [无法使用 LLama Packs [Bug]: · Issue #10777 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/10777): Bug 描述：我下载任何 pack 都会遇到这个错误 FileNotFoundError: [Errno 2] No such file or directory: '/content/chain_of_table_pack/llama_index/packs/tables/base.py' 版本 ...
- [OpenAI 的 Agent 2.0：兴奋还是恐惧？](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx): 我想为您全面介绍浏览器/移动端/桌面端的 AI Agent。获取免费的 HubSpot 电子书：使用生成式 AI 扩展您的内容运营：https://c...
- [使用 LlamaIndex 评估 RAG 系统的理想 Chunk Size。](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5): 探索如何使用 LlamaIndex 的 Response Evaluation 来优化 RAG 的 chunk size 以获得最佳性能
- [RAG 已死！RAG 万岁！](https://vectorize.io/2024/02/16/rag-is-dead-long-live-rag/): 昨天 Google 发布了 Gemini 1.5，它具有非常长的上下文窗口，最高可达 100 万个 token。与现有的具有较长上下文的模型（如 GPT-4）相比，这是一个相当大的进步……
- [在本地训练 Mixtral 8x7B 并集成 LlamaIndex：为您的数据定制 AI 模型](https://medium.com/ai-advances/training-mixtral-8x7b-locally-with-llamaindex-integration-customizing-ai-models-for-your-data-4c704e693e59): Ankush k Singal

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1208031984799055922) (24 条消息🔥): 

- **解析用于 RAG 的 Whisper 转录文本**：`@cablecutter` 提出了关于将 **Whisper Transcripts** 与 RAG 功能结合使用的最佳实践问题，特别是如何保留“发言人（speaker）”、“置信度（confidence）”和“时间戳（timestamps）”等元数据。`@amgadoz` 建议使用 LLM 进行发言人分离（speaker diarization），按发言人或轮次对转录文本进行分块（chunking），并添加来自前一个片段的元数据和文本以提供上下文。（未提供链接）
  
- **寻求 LlamaIndex 更新指导**：`@badrinathsvn_72554` 在将 **LlamaIndex 从 0.9.13 升级到 0.10.6** 后遇到了 `ImportError` 问题。`@cheesyfishes` 回应并建议查看 [迁移指南](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232) 以解决兼容性问题。（提供的实际链接指向非公开 Discord 服务器）

- **关于提示驱动型 RAG 挑战的新博客**：`@subham5089` 分享了 [一篇博客文章](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7)，讨论了当前使用 **提示驱动型 RAG 系统（prompt-driven RAG systems）** 所面临的挑战以及这些挑战的潜在解决方案。

- **React RAG QA 前端模板发布**：`@sl33p1420` 介绍了一个由 runelab 赞助的开源 **React RAG QA 前端模板（boilerplate）**，并分享了一份关于使用 React 构建交互式聊天机器人的详尽 [Medium 指南](https://medium.com/@marco.bertelli/unveiling-the-power-of-rag-building-an-interactive-chatbot-with-react-a-comprehensive-guide-99c409a5f69a)。

- **Gemini 1.5 之后 RAG 的相关性**：`@chiajy` 对 **Gemini 1.5** 之后的 RAG 系统表示乐观，理由是其非黑盒特性以及它在准确性、成本和延迟方面提供的控制力。他们分享了一篇 [Medium 文章](https://medium.com/enterprise-rag/why-gemini-1-5-and-other-large-context-models-are-bullish-for-rag-ce3218930bb4)，讨论了像 Gemini 1.5 这样新兴的长上下文模型将如何对 RAG 系统产生积极影响。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232)：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。
- [Why Gemini 1.5 (and other large context models) are bullish for RAG](https://medium.com/enterprise-rag/why-gemini-1-5-and-other-large-context-models-are-bullish-for-rag-ce3218930bb4)：通过 RAG 进行优化：如何克服长上下文模型在准确性、成本、延迟和其他性能方面的局限性。
- [Unveiling the Power of RAG: Building an Interactive Chatbot with React — A Comprehensive Guide](https://medium.com/@marco.bertelli/unveiling-the-power-of-rag-building-an-interactive-chatbot-with-react-a-comprehensive-guide-99c409a5f69a)：往期文章：
- [SOCIAL MEDIA TITLE TAG](https://os-copilot.github.io/)：SOCIAL MEDIA DESCRIPTION TAG TAG
- [Unleashing the Power of Agents and QueryPipeline with LlamaIndex](https://medium.com/@andysingal/unleashing-the-power-of-agents-with-llamaindex-3efe72921b10)：Ankush k Singal

  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1207969569377878016) (32 条消息🔥): 

- **关于 AI 身份的误解**：`@lightningralf` 指出，询问模型的身份没有价值，因为它只能在其 system prompt 范围内做出响应，任何诸如 "gemini 500" 之类的身份标识都没有实质内容。
- **Eugene Yan & Jason Liu 直播公告**：`@swyxio` 分享了一个 [YouTube 视频](https://m.youtube.com/watch?v=PU_MErIaAEU)，内容是 Eugene Yan 和 Jason Liu 之间的直播讨论，正如 `@eugeneyan` 所提到的，Hamel 作为惊喜嘉宾加入了。
- **认真对待 AI Guardrails**：`@sugaroverflow` 对 [Ars Technica 的推文](https://x.com/arstechnica/status/1758540835132494119?s=20) 感到好笑，该推文报道了加拿大航空必须遵守其聊天机器人编造的退款政策，`@swyxio` 回应称，这可能是企业认真对待 AI guardrails 的唯一方式。
- **AI 的法律地位**：`@mdcker` 强调了加拿大航空为其聊天机器人的行为辩解，声称“**聊天机器人是一个独立的法律实体**”。然而，法官并未接受这一辩护。
- **OpenMoE 论文回顾**：`@intheclouddan` 分享了一篇关于 [OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf) 的论文链接，这是一系列开源的 Mixture-of-Experts LLM。不过 `@swyxio` 评论说结果并不理想，训练数据比 tinyllama 还少，且缺乏对推理效率的探索。

**提到的链接**：

- [no title found](https://news.ycombinator.com/item?id=39411748): 未找到描述
- [Tweet from Ars Technica (@arstechnica)](https://x.com/arstechnica/status/1758540835132494119?s=20): 加拿大航空必须遵守航空公司聊天机器人编造的退款政策 https://trib.al/s84FkPu
- [OpenMoE/paper/paper.pdf at main · XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf): 一系列开源的 Mixture-of-Experts (MoE) LLM - XueFuzhao/OpenMoE
- [Chat w/ Eugene Yan &amp; Jason Liu](https://m.youtube.com/watch?v=PU_MErIaAEU): 我们将尝试直播下一次一对一交流，看看我们的效率如何。
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe): 用于 LLM tokenization 中常用的 Byte Pair Encoding (BPE) 算法的极简、清晰代码。 - karpathy/minbpe

  

---

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1207989359689859092) (8 messages🔥): 

- **BERT 论文闪电战**：`@ivanleomk` 宣布了一个关于 **BERT 论文的 3 分钟讨论**。感兴趣的参与者可以在[这里](https://lu.ma/fcsum9r1)报名。

- **加入 LLM 论文俱乐部（亚洲版！）**：`@swyxio` 邀请用户加入 **LLM 论文俱乐部（亚洲版！）**；如需参加，请点击提供的 [Discord 链接](https://discord.com/channels/822583790773862470/1200029657744027658)。

- **播客提醒 - Modal 新集**：`@swyxio` 分享了最新的 **Latent Space 播客节目，本期嘉宾为 Modal**。对话内容包括 [OpenAI 的 Sora](https://news.ycombinator.com/item?id=39386156) 和 [Gemini 1.5](https://news.ycombinator.com/item?id=39383446) 的影响、即将举行的活动，以及关于 AI 的 Serverless 基础设施的见解 [在此收听](https://www.latent.space/p/modal)。

- **宣传 Serverless AI**：`@swyxio` 请求协助推广一篇关于 AI 工程师真正的 Serverless 基础设施的 **Latent Space 博客文章** [在此支持](https://x.com/FanaHOVA/status/1758568180132536471?s=20)。

- **深入探讨 Agent 现状**：`@swyxio` 重点介绍了一场由 `<@363877777977376768>` 主持的关于 **Agent 现状** 的进行中会议。用户可以通过此 [Discord 链接](https://discord.com/channels/822583790773862470/1200548371715342479)加入讨论。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/822583790773862470/1200029657744027658)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/822583790773862470/1200548371715342479)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Truly Serverless Infra for AI Engineers - 与 Modal 的 Erik Bernhardsson 对话](https://www.latent.space/p/modal)：为 AI 工程师构建终极的自配置运行时（Self Provisioning Runtime），为什么 Oracle Cloud 被低估了，GPU 并行性的现状，以及为什么要招聘 IOI 金牌选手工程师。
- [来自 Alessio Fanelli (@FanaHOVA) 的推文](https://x.com/FanaHOVA/status/1758568180132536471?s=20)：🆕 为 AI 工程师打造的真正 Serverless 基础设施 https://www.latent.space/p/modal 。在 2021 年，@bernhardsson 写下了一份“软件基础设施 2.0 愿望清单”，但很快决定亲自操刀……

---

### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/1207990281132052520) (32 messages🔥): 

- **深入探讨 BERT 对 Google 搜索的影响**：`@ivanleomk` 分享了一份[总结](https://llm-paper-club-asia-notes.vercel.app/papers/bert)，`@swyxio` 讨论了 **BERT** 在改进 Google 搜索结果中的应用并对其实现方式表示好奇，而 `@bryanblackbee` 建议 Google 可能使用文档 Embedding 进行语义搜索。
- **BERT 双向性的奇妙之处**：在讨论中，`@swyxio` 对双向模型 **BERT** 在单向 **GPT** 之前被发明表示惊讶，这暗示了 NLP 模型演进的细微差别。
- **Swyxio 的网络困扰**：尽管存在连接问题，`@swyxio` 在向大家道晚安之前，仍表示非常享受本次会议的回顾。
- **关于模型训练和质量的问题**：新人 `@farukga` 好奇像 Google 这样的大模型是针对什么样的“令人满意”的文本或质量信息进行训练和微调的。
- **对下一篇 LLM 论文的期待**：在聊天结束时，`@joellee.` 等成员询问了下周的论文，`@mattoshimasu` 询问了会议录音的可用性，参与者们感谢了 `@ivanleomk` 的讲解。

**提到的链接**：

- [图解 BERT、ELMo 等（NLP 如何攻克迁移学习）](https://jalammar.github.io/illustrated-bert/)：讨论：Hacker News (98 points, 19 comments), Reddit r/MachineLearning (164 points, 20 comments) 翻译：中文（简体）、法语 1、法语 2、日语、韩语、波斯语、俄语、西班牙语...
- [Nextra：下一代文档生成器](https://llm-paper-club-asia-notes.vercel.app/papers/bert)：Nextra：下一代文档生成器
- [比以往任何时候都更理解搜索](https://blog.google/products/search/search-language-understanding-bert/)：语言理解科学的新进展将如何帮助你在搜索中找到更有用的信息。

---

### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1208156189297344552) (182 messages🔥🔥):

- **AI、Agent 和 State Machines 的探索**：包括 `@markredito` 和 `@fanahova` 在内的成员讨论了与 AI Agent 和 State Machines 相关的概念，提到了[关于该主题的一集视频](https://youtu.be/4Ps7ahonRCY)，并引用了 `@davidkpiano` 关于 State Machines 与 AI 结合使用的见解。
- **资源汇编进行中**：包括 `@yikesawjeez` 和 `@swyxio` 在内的几位成员正在考虑汇编一份与 Agent 相关的工具和资源清单，征求关于 State Machines 和 local models 的建议，并分享了相关框架的链接，如 [CrewAI](https://github.com/joaomdmoura/crewAI) 和 [MagickML](https://www.magickml.com)。
- **关于 AI 开发工具的社区互动**：在对话中，`@yikesawjeez` 提到要扩展一个 hackathon 项目，`@swyxio` 分享了一个用于播客的工具 [smol-podcaster](https://github.com/FanaHOVA/smol-podcaster)，`@slono` 承诺分享其幻灯片的更精美版本。
- **直播测试和流媒体计划**：`@swyxio` 和 `@yikesawjeez` 讨论了通过直播实验 AI Agent 框架的计划，`@swyxio` 分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=S6MtNDIm3oc&ab_channel=swyx)，内容是对比 CrewAI、LangGraph 和 AutoGen 的直播。
- **庆祝社区里程碑**：对话纪念了 Latent Space 成立一周年，`@fanahova` 强调了这一里程碑，`@slono` 分享了他们为社区做出贡献的兴奋之情。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1208167505923932261): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1075282825051385876/1110661051311202406): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/979492707279978586/1208139552946913290): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [AI in Action - Agents - mnml's vault - Obsidian Publish](https://publish.obsidian.md/manuel/Notes/Talks/AI+in+Action+-+Agents): AI in Action - Agents - mnml 的库 - 由 Obsidian Publish 提供支持。
- [StreamYard | Browser-based live studio for professionals](https://streamyard.com/4mnxw8xnp8): StreamYard 是一款在浏览器中运行的专业直播和录制工作室。录制你的内容，或直播到 Facebook、YouTube 和其他平台。
- [ai onboarding / normies links](https://arc.net/folder/94AD3E8D-38F8-4A0C-9A91-8F2487AB4B20): AI 入门 / 常规链接
- [AgentOps](https://agentops.ai/): 构建具有可观测性、评估和重放分析功能的 AI Agent 和 LLM 应用。告别黑盒和 Prompt 猜测。
- [A Gentle Introduction to CRDTs – vlcn.io](https://vlcn.io/blog/intro-to-crdts): 无冲突复制数据类型 (CRDTs) 可能非常棘手。你可能需要花费数月时间阅读论文并实现不同的算法，然后它们才会最终被理解并变得简单。否则它们会...
- [Making state management intelligent - David Khourshid](https://www.youtube.com/watch?v=Iw8Uf7q4nVc): 让状态管理智能化。管理状态是复杂的。人类甚至更复杂。作为开发者，我们的工作是提供无缝且直观的...
- [Magick - Cutting-edge tools for AI creators](https://www.magickml.com): 触手可及地体验先进 AI 的力量，无需代码。借助我们全面的工具包，轻松构建、部署、维护和扩展你的 AI Agent、机器人和应用到新的...
- [crewAI/src/crewai/agent.py at main · joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/agent.py): 用于编排角色扮演、自主 AI Agent 的框架。通过培养协作智能，CrewAI 赋能 Agent 无缝协作，处理复杂任务。- joaomdmoura/cr...
- [[Livestream] CrewAI vs LangGraph vs AutoGen](https://www.youtube.com/watch?v=S6MtNDIm3oc&ab_channel=swyx): 未找到描述
- [Tweet from Alex Reibman (@AlexReibman)](https://x.com/AlexReibman/status/1757335836482498647?s=20): 4/ Crew AI 用于构建可靠自主 AI Agent 的编排框架。这个例子展示了一个能自动创建和安排社交媒体帖子的 Agent，但它也可以进行研究、编码和...
- [What is missing from current AI?](https://youtu.be/4Ps7ahonRCY?si=U_V425_OfLORLGHR): 获得麻省理工学院 (MIT) 博士学位的 Brandon Rohrer 致力于从底层细节彻底理解算法，以便他能让这些算法变得易于理解...
- [Build Agents from Scratch (Building Advanced RAG, Part 3)](https://youtu.be/T0bgevj0vto?si=YeW08q4tVqm3wZej): 在本系列的第三个视频中，我们将教你如何构建由 LLM 驱动的 Agent 管道——具体来说，我们将教你如何构建一个 ReAct Agent (Yao 等人...
- [GitHub - FanaHOVA/smol-podcaster: smol-podcaster is your autonomous podcast production intern 🐣](https://github.com/FanaHOVA/smol-podcaster): smol-podcaster 是你的自主播客制作实习生 🐣 - FanaHOVA/smol-podcaster
- [GitHub - Actioninsight/AutoNL: AutoNL - Natural Language Automation tool](https://github.com/Actioninsight/AutoNL): AutoNL - 自然语言自动化工具。通过在 GitHub 上创建账号来为 Actioninsight/AutoNL 的开发做出贡献。
- [crewAI/src/crewai/crew.py at main · joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/crew.py): 用于编排角色扮演、自主 AI Agent 的框架。通过培养协作智能，CrewAI 赋能 Agent 无缝协作，处理复杂任务。- joaomdmoura/cr...
- [Join the The Arena Online Discord Server!](https://discord.gg/eGrzMA2d): 普通的紧急开发者日 & GPTs 商店黑客松服务器！绝对不是什么秘密的技术术士聚集地！| 373 位成员

---

### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1207976954343464981) (46 messages🔥): 

- **西班牙超级计算的新纪元**：`@mdelamor` 分享了一篇关于 Marenostrum 5 落成的文章，这是位于一座旧教堂内的超级计算设施的新扩展，标志着欧洲超级计算进入了一个新纪元。文章阅读地址：[Marenostrum 5 - a new era](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/)。

- **Nvidia 驱动安装问题**：`@apaz` 表达了在新 PC 上安装 Nvidia 闭源驱动的挫败感；他们不得不手动应用来自 bookworm-proposed-changes 的补丁。其他用户如 `@joseph_en` 讨论了 Nvidia 驱动持续存在的挑战，以及 AMD 在未来提供竞争的可能性。

- **C++ 模板困扰**：开发者 `_davidgonmar` 遇到了一个 C++ 模板问题，该模板在 WSL 上可以使用 NVCC 和 g++ 编译，但在 Visual Studio 中会导致错误。`@jeremyhoward` 建议确保在 Visual Studio 中启用了 C++17，因为编译器需要处理 fold expressions（折叠表达式），这最终被证明是解决方案。

- **4D Gaussian Splatting 讨论**：`@andreaskoepf` 通过分享网站链接 [gmix.ai](https://www.gmix.ai/) 发起了关于 4D Gaussian Splatting 的讨论，`@joseph_en` 询问了该主题的推荐论文，以便更好地了解这些最新进展。

- **挑战性的 Kernel 调试马拉松**：`_davidgonmar` 详细描述了对一个 CUDA kernel 进行的艰苦调试过程，其中错误的内存保护条件导致了越界写入。`@gogators.` 建议使用 Nvidia 的 compute sanitizer（CUDA 附带的工具）来检测非法内存访问和竞态条件（race conditions）。

**提到的链接**：

- [Gmix 4D Spatial Video](https://www.gmix.ai/)：未找到描述
- [Error installing Nvidia driver on Debian 12.5 · Issue #361 · NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit/issues/361)：问题摘要：尝试在 Debian 12.5 上安装 Nvidia 驱动（nvidia-driver）时，遇到了阻止成功安装的错误。错误信息：Setting up nvidia-persistenced ...
- [no title found](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/)：未找到描述

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1208105646579388477) (9 messages🔥): 

- **Nsight 见解**：用户 `@lancerts` 表达了对使用 **Nsight** 的兴奋，但也提到需要解读该工具提供的详细数据。
- **寻求 GPU 使用率指标**：`@marvelousmit` 询问如何通过 **Nsight Systems** 测量 GPU 饱和度，`@lancerts` 回复说，在带有 trace 的 profile 过程中可以看到这些细节。
- **CUDA 12.1 提升 Kernel 参数限制**：`@iron_bound` 分享了一篇 [博客文章](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)，详细介绍了 **CUDA 12.1** 将 kernel 参数限制从 4,096 字节增加到 32,764 字节，这增强了像 `@morousg` 这样在工作中融合（fuse）多个 kernel 的开发者的功能体验。
- **在时间敏感型应用中考虑电气性能**：用户 `@defaultguyredshirt` 建议，除了时间性能外，电气性能也是可能影响 case pickers 选择的一个因素。
- **平衡 GPU Flops-per-Byte**：`@andreaskoepf` 分享了 `@RajaXg` 的一条推文，讨论了多年来 GPU 开发中 flops-per-byte 的不平衡，而 `@morousg` 则强调需要优化 GPU 库，以提高内存带宽和寄存器的利用率。

**提到的链接**：

- [Tweet from Raja Koduri (@RajaXg)](https://x.com/RajaXg/status/1758935199508046247)：GPU Flops-per-byte 多年来增长疯狂。如果把互连（PCIE, NVLink, XeLink 等）带宽也画出来，那就更疯狂了。在第一个浮点数 sha... 时，我们曾接近 1:1。
- [CUDA 12.1 Supports Large Kernel Parameters | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)：CUDA 12.1 允许你通过 kernel 参数传递多达 32,764 字节，这可以用来简化应用程序并获得性能提升。

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1208501814153060392) (1 messages): 

- **关注 CUDA 精通课程**：`@marksaroufim` 提醒 `@everyone`，**CUDA MODE 第 6 讲：优化 PyTorch 优化器** 即将开始，并强调 Jane 的贡献在每次训练 PyTorch 模型时都至关重要。强烈建议寻求实用 CUDA 见解的观众参加。

### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1208069526877503488) (11 条消息🔥): 

- **项目集成协作邀请**：`@shindeirou` 正在寻求将一项挑战集成到课程项目中，并在 `@419115250021826560` 的协助下进行，邀请他们进行通话以进一步讨论，并提到有三位大学同学加入。
- **乐于提供协助与探索**：`@mickgardner` 为正在讨论的项目提供帮助，计划审查基准代码和相关论文。
- **Large World Models 发布**：`@andreaskoepf` 分享了关于 **Large World Model** (LWM) 的信息，这是一个基于 LLaMA-2 训练的开源模型，提供了其 [Overview](/ifioravanti/lwm)、[Tags](/ifioravanti/lwm/tags)、[GitHub](https://largeworldmodel.github.io/) 和 [HuggingFace](https://huggingface.co/LargeWorldModel) 页面的链接，并提到其处理超过 1M tokens 长文本的能力。
- **RingAttention 实现见解**：`@mickgardner` 提到 LWM 仍在使用 JAX，并重新实现了 attention 模块，以包含带有 BPT 的标准 RingAttention 以及一个 "ring+flash_attention on tpu" 的实现。
- **安排协作会议**：`@andreaskoepf` 和 `@__daem0n__` 讨论安排会议以协作开展项目，建议的时间以 Discord 的本地化时间戳格式给出，强调 `@andreaskoepf` 在今天讲座前后的 2 小时内有空，而 `@__daem0n__` 确认明天可以。
- **创建项目专用频道**：`@andreaskoepf` 宣布为 RingAttention 项目创建一个单独的频道，以便集中讨论和协作。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1189498204333543425/1208496482005549086)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持联系。
- [ifioravanti/lwm](https://ollama.com/ifioravanti/lwm)：Large World Model 是一个开源模型，在 Books3 过滤数据的子集上基于 LLaMA-2 训练而成。
- [Large World Models](https://largeworldmodel.github.io/)：未找到描述。

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1208314222186856451) (23 条消息🔥): 

- **Shader Demo 爱好者**：用户 `@andreaskoepf` 分享了他们对 shader demo 的钦佩，并以 [twigl.app](https://twigl.app/?ol=true&ss=-NqUk6pcpkcFek1iupHp) 为例展示了令人印象深刻的作品。

- **GPU 编程新手入门**：`@jollyphoenix.ai` 是一位对高效机器学习和 GPU 编程感兴趣的新人，寻求入门建议，并分享了一个关于 **Heterogenous Parallel Programming**（异构并行编程）的 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb)。

- **CUDA 资源推荐**：`@marksaroufim` 推荐查看一个 YouTube 频道和一个 [GitHub 资源流](https://github.com/cuda-mode/resource-stream) 以获取 CUDA 编程资料，以此回复 `@jollyphoenix.ai`。

- **Groq 的推理速度引发讨论**：`@cs_os_05101` 提到了 [Groq](https://groq.com/)，一家以快速推理著称的初创公司，引发了关于他们如何实现这一速度以及他们与传统 CUDA 架构有何不同的讨论。

- **实战 CUDA 学习请求**：`@cs_os_05101` 希望在 CUDA-MODE 仓库中看到针对特定主题的实际代码示例，并引用了最近的一次演示以及那本详尽的 CUDA 书籍的高昂成本。

**提到的链接**：

- [Groq](https://groq.com/)：未找到描述。
- [twigl.app](https://twigl.app/?ol=true&ss=-NqUk6pcpkcFek1iupHp)：twigl.app 是一个在线编辑器，用于 One tweet shader，带有 gif 或 webm 生成器和声音 shader。
- [Heterogenous Parallel Programming - CUDA Programming](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb)：未找到描述。
- [TensorRT-LLM/docs/source/blogs/H200launch.md at main · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/H200launch.md#h200-achieves-nearly-12000-tokenssec-on-llama2-13b-with-tensorrt-llm)：TensorRT-LLM 为用户提供了一个易于使用的 Python API，用于定义 Large Language Models (LLMs) 并构建包含最先进优化以执行推理的 TensorRT 引擎...
- [GitHub - cuda-mode/resource-stream: CUDA 相关的资讯和资料链接](https://github.com/cuda-mode/resource-stream)：CUDA 相关的资讯和资料链接。欢迎通过在 GitHub 上创建账户来为 cuda-mode/resource-stream 的开发做出贡献。

  

---

### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1207985945199116318) (4 messages): 

- **对 Row/Col 性能感到惊讶**：`@mikkelisk` 分享了他们的基准测试结果，并对 **row** 和 **col** 操作之间缺乏差异表示惊讶。他们怀疑实现中存在错误，并好奇其他人使用的是什么 GPU。

- **并发内存编辑必须使用 Atomic Add**：`@eporat` 解释了当多个线程修改同一个内存值时使用 `atomicAdd` 的必要性，暗示这对于他们的代码正常运行至关重要。

- **为了性能进行转置**：`@andreaskoepf` 链接到了 `flash-attention` 的 GitHub 仓库，其中在性能优化中使用了转置操作，正如 [flash-attention 源代码](https://github.com/Dao-AILab/flash-attention/blob/5cdabc2809095b98c311283125c05d222500c8ff/csrc/flash_attn/flash_api.cpp#L372-L380)中所示。

- **CUDA Cores 与 Warp 线程之间的差异**：`@lucaslingle` 询问了引用书籍第 4 章中的一个陈述，探讨了根据作者的说法，A100 中每个 Warp 的 32 个线程如何能被只有 16 个 CUDA cores 的处理块（processing block）同时执行。

**提到的链接**：

[flash-attention/csrc/flash_attn/flash_api.cpp at 5cdabc2809095b98c311283125c05d222500c8ff · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/5cdabc2809095b98c311283125c05d222500c8ff/csrc/flash_attn/flash_api.cpp#L372-L380)：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1208322419639721984) (1 messages): 

- **发布新的 Python CUDA 视频**：`@andreaskoepf` 宣布在 CUDA-mode YouTube 频道上发布了一个名为 "Lecture 5: Going Further with CUDA for Python Programmers" 的新视频。讲座材料可以在 [GitHub](https://github.com/cuda-mode/lectures) 上找到。

**提到的链接**：

[Lecture 5: Going Further with CUDA for Python Programmers](https://www.youtube.com/watch?v=wVsR-YhaHlM)：材料在此 https://github.com/cuda-mode/lectures

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1208082682068734053) (3 messages): 

- **比较 PyTorch 2.0 和 JAX 的图捕获（Graph Capture）**：`@vguerra` 根据 [PyTorch 2.0 论文](https://pytorch.org/assets/pytorch_2.pdf) 强调了 PyTorch 2.0 和 JAX 在图捕获方面的差异。JAX 的设计深受 XLA 的影响，XLA 要求函数式纯净（functionally pure）的程序，且不支持依赖数据的 Python 控制流；更多细节见论文 2.6 节，更详细的对比可以在 [torch.fx 论文](https://arxiv.org/abs/2112.08429) 中找到。
- **JAX Fusion 流水线分析**：`@marvelousmit` 引用了一篇探索 JAX Fusion 流水线的论文，可以在 [arXiv](https://arxiv.org/pdf/2301.13062.pdf) 阅读详细内容。
- **用户对深度学习论文表示兴趣**：`@marcom79` 对分享的深度学习框架资源表示感谢，并确认打算研读提供的论文。

**提到的链接**：

[Torch.fx: Practical Program Capture and Transformation for Deep Learning in Python](https://arxiv.org/abs/2112.08429)：现代深度学习框架提供了嵌入在 Python 中的命令式、Eager 执行编程接口，以提供高效的开发体验。然而，深度学习从业者有时...

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1208498435263569940) (79 messages🔥🔥): 

- **CUDA RingAttention 开发启动**：`@andreaskoepf` 发起了 CUDA RingAttention 实现的开发，分享了两篇关于 Ring Attention 技术及其在深度学习中应用的重点论文（[论文 1：Transformers 中的近乎无限上下文](https://arxiv.org/abs/2310.01889)，[论文 2：使用 RingAttention 的世界模型](https://arxiv.org/abs/2402.08268)），并提供了该项目的 [GitHub 仓库](https://github.com/LargeWorldModel/LWM) 和 [HuggingFace 上的模型](https://huggingface.co/LargeWorldModel) 链接。
  
- **协调与研究开始**：包括 `@ericauld`、`@jamesmel`、`@nshepperd` 在内的团队讨论了初步步骤，例如审阅论文和探索现有的 RingAttention 软件实现，并对能够贡献力量表示兴奋。

- **探索与头脑风暴**：`@andreaskoepf` 建议建立任务清单、阅读现有的 RingAttention 论文，并开会讨论进展。`@lancerts` 和 `@jku100` [分享](https://github.com/lhao499/RingAttention)了现有的潜在 RingAttention 实现的 GitHub 仓库。

- **技术对话与考量**：频道中进行了技术讨论，`@andreaskoepf` 建议使用 NCCL 进行多 GPU 通信，并可能赞助开发资源；同时 `@ericauld` 和 `@iron_bound` 正在研究现有代码，`@lancerts` 则提供了关于配置 JAX meshes 并行维度的见解。

- **建立协作**：`@andreaskoepf` 建立了一个[新的 GitHub 仓库](https://github.com/cuda-mode/ring-attention)，专门用于 RingAttention 的优化 CUDA kernels，并邀请团队成员贡献代码；`@jamesmel` 表示有兴趣处理一个关于 peer-to-peer 内存传输分析的 issue。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1189498205101109301)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [ifioravanti/lwm](https://ollama.com/ifioravanti/lwm)：Large World Model 是一个开源模型，基于 LLaMA-2 在 Books3 过滤数据的子集上训练而成。
- [Google Colaboratory](https://colab.research.google.com/drive/1PNDTLx2UYYk8XmTb9e_ZBxPx8P6eByvx?usp=sharing)：未找到描述
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)：Softmax 函数在机器学习中无处不在，之前的多项工作都提出了更快的替代方案。在本文中，我们提出了一种通过更少的内存访问来计算经典 Softmax 的方法...
- [Google Colaboratory](https://colab.research.google.com/drive/1X-x6PCRydNY9LZBPLA0DZh3Tj2Dyz60M?usp=sharing)：未找到描述
- [ELI5: Flash Attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)：逐步解释最重要的 MLSys 突破之一是如何工作的——细节非常详尽。
- [Analyze overlapped P2P memory transfer and computing · Issue #1 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/1)：创建一个 ipynb，在 PyTorch 中分析 peer-to-peer（两个 GPU 之间）的内存传输和并行计算。虚拟计算可以例如是循环中的一些大型 matmuls。
- [Comment about use of all gather · Issue #1 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/1)：嗨 Phil！希望你一切都好。正如你看到的 Gemini Pro 1.5 可以处理 100 万个 token，开源领域还有一些工作要做才能赶上 :D 将 Ring Attention 移植到 PyTorch 绝对是...
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch)：对 Ring Attention 的探索，源自伯克利 AI 的 Liu 等人 - lucidrains/ring-attention-pytorch
- [Papers with Code - Ring Attention with Blockwise Transformers for Near-Infinite Context](https://paperswithcode.com/paper/ring-attention-with-blockwise-transformers)：在 3 个代码库中实现。
- [ring-attention/README.md at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/README.md)：用于 ring-attention 的优化 kernels [进行中]。通过在 GitHub 上创建账户为 cuda-mode/ring-attention 的开发做出贡献。
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch?tab=readme-ov-file#usage)：对 Ring Attention 的探索，源自伯克利 AI 的 Liu 等人 - lucidrains/ring-attention-pytorch
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe/tree/master)：用于 LLM tokenization 中常用的 Byte Pair Encoding (BPE) 算法的极简、干净的代码。 - karpathy/minbpe
- [GitHub - cuda-mode/ring-attention: Optimized kernels for ring-attention [WIP]](https://github.com/cuda-mode/ring-attention)：用于 ring-attention 的优化 kernels [进行中]。通过在 GitHub 上创建账户为 cuda-mode/ring-attention 的开发做出贡献。
- [Analyze existing ring-attention implementations · Issue #2 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/2)：请就你的发现创建一个小的 markdown 报告。lhao499/ring-attention/bpt lucidrains/ring-attention-pytorch 对实现有一些直观感受：在我们的开发机器上易于设置和运行（例如...）
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch?tab=readme-ov-file#usa)：对 Ring Attention 的探索，源自伯克利 AI 的 Liu 等人 - lucidrains/ring-attention-pytorch

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Transformer 已成为许多最先进 AI 模型的首选架构，在广泛的 AI 应用中展现出卓越的性能。然而，内存需求...
- [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268): 当前的语言模型在理解难以用言语描述的世界方面存在不足，并且在处理复杂的长篇任务时表现挣扎。视频序列提供了宝贵的时间信息...
- [Large World Models](https://largeworldmodel.github.io/): 未找到描述
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): 通过在 GitHub 上创建账号，为 LargeWorldModel/LWM 的开发做出贡献。
- [LargeWorldModel (Large World Model)](https://huggingface.co/LargeWorldModel): 未找到描述
- [RingAttention/bpt/ring_attention.py at main · lhao499/RingAttention](https://github.com/lhao499/ring-attention/blob/main/bpt/ring_attention.py): 具有任意大上下文的 Transformer。通过在 GitHub 上创建账号，为 lhao499/RingAttention 的开发做出贡献。
- [LWM/lwm/ring_attention.py at main · LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM/blob/main/lwm/ring_attention.py#L3?): 通过在 GitHub 上创建账号，为 LargeWorldModel/LWM 的开发做出贡献。

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1208019765503987742) (54 条消息🔥): 

- **8-bit 微调澄清**：用户 `@nafnlaus00` 讨论了在 8-bit 模型上进行全量微调（full finetunes）的可能性，并对 AI Explained 展示的最新进展发表了评论。同时，`@dreamgen` 质疑了 Stability AI 在现有资源下的关注重点。
- **Hugging Face 宕机了？**：`@le_mess` 询问 Hugging Face 是否宕机，但没有提供关于服务状态的进一步信息。
- **讨论 Stability 的新图像模型**：`@nafnlaus00` 分享了关于 Stability 新扩散模型的见解，并指出 Tesla 在其自动驾驶任务中使用了 3d-informed diffusion。
- **优化器辩论**：`@c.gato` 质疑 Adam 是否仍然是 Large Language Models (LLMs) 的首选优化器，`@nafnlaus00` 建议将 Lion 作为一种更节省内存的替代方案，`@yamashi` 和 `@nruaif` 讨论了如何使用和实现 Lion 优化器，并提供了一个 GitHub 仓库链接 ([GitHub - lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch))。
- **使用 Axolotl 进行微调和模型选择**：用户 `@qwerty_qwer`、`@le_mess` 和 `@masa_92515` 讨论了使用 Axolotl 进行微调，并建议使用 qwen 系列等 1.6b 模型作为基础。他们还讨论了 1.3b 模型的 VRAM 需求和基准测试。

**提到的链接**：

- [Grant♟️ (@granawkins) 的推文](https://x.com/granawkins/status/1758689077472399566?s=20): 未找到描述
- [GitHub - lucidrains/lion-pytorch: 🦁 Lion, new optimizer discovered by Google Brain using genetic algorithms that is purportedly better than Adam(w), in Pytorch](https://github.com/lucidrains/lion-pytorch): 🦁 Lion，由 Google Brain 使用遗传算法发现的新优化器，据称在 Pytorch 中优于 Adam(w) - lucidrains/lion-pytorch

  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1208028060050260048) (19 messages🔥): 

- **Perplexity 作为难度衡量标准**：`@dreamgen` 探讨了是否可以使用 Perplexity 来衡量基准模型下样本的难度。`@suikamelon` 回应并分享了一篇[论文](https://arxiv.org/abs/2310.13008)，该论文为 Large Language Models (LLMs) 的 Supervised Fine-Tuning (SFT) 引入了一个新维度：*learnability*（可学习性），并提出了一种基于模型学习能力来选择 SFT 数据的方法。

- **SPIN 的官方实现**：`@nruaif` 提供了 Self-Play Fine-Tuning (SPIN) 官方实现的 [GitHub 链接](https://github.com/uclaml/SPIN?tab=readme)，表明了在讨论背景下对该项目的兴趣或实用性。

- **关于 Torch 更新的合并讨论**：`@nanobitz` 提到了开发流水线中几个标记为 `ready to merge` 的标签，并提出了可能需要将 PyTorch 更新到 2.2.x 版本的议题，引用了 Discord 频道中的一个链接。

- **关于添加新 Optimizer 的困惑**：`@yamashi` 对在系统中集成新 Optimizer 表示困惑，特别是当 `args.optim` 设置为 Transformers 库原生不支持的值时出现的错误。随后他们注意到添加了 torchdistx 支持，其中包含一个未列在 Transformers 原生选项中的 Optimizer，如 [GitHub 上的 commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78) 所示。

- **实现新的 Optimizer 解决方案**：尽管在添加新 Optimizer 时遇到了初步挑战，`@yamashi` 表示他们找到了一个变通方案，虽然他们称之为 "nasty"（糟糕的），但可以在第二天提交，这引起了 `@nanobitz` 对代码分享的好奇。



**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1104757954588196865/1111279858136383509/1208306657143169034)：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。
- [LoBaSS: Gauging Learnability in Supervised Fine-tuning Data](https://arxiv.org/abs/2310.13008)：Supervised Fine-Tuning (SFT) 是将 Large Language Models (LLMs) 与特定任务需求对齐的关键阶段。Fine-tuning 数据的选择深刻影响着模型的...
- [fdsp config dict fix, todo list, add torchdistx support · OpenAccess-AI-Collective/axolotl@ad2b48c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78)：未找到描述
- [GitHub - uclaml/SPIN: The official implementation of Self-Play Fine-Tuning (SPIN)](https://github.com/uclaml/SPIN?tab=readme-ov-file)：Self-Play Fine-Tuning (SPIN) 的官方实现 - uclaml/SPIN

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1208027144286904403) (8 messages🔥): 

- **Mistral 的梦之队**：`dreamgen` 提到 **Mistral** 在其架构中使用了所谓的 `dreamgen`。

- **Mergekit 大乱斗**：`philipmay` 询问是否可以使用 **Mergekit** 将四个 **Llama 13b 模型** 合并为一个 **MoE 模型**，作为进一步 Fine-tuning 的基础，并寻求其他有此类任务经验的人的建议。

- **克服 Overfitting？**：`noobmaster29` 询问增加模型的 **rank** 是否可以防止 **Overfitting**。该问题在社区中保持开放讨论。

- **学习率困惑消除**：`nafnlaus00` 询问训练中的 **Learning Rate (LR)** 是应用于每个训练样本还是每个生成的 Token，`yamashi` 澄清它是按 **Batch** 应用的。

- **样本大小的重要性**：`nafnlaus00` 随后推测，较大的训练样本可能意味着每个 Batch 的样本数更少，因此可能对训练产生更大的影响，尤其是在启用 **Sample Packing** 时。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1208089543291568148) (8 条消息🔥): 

- **数据集困境与豆子饮食**：`@iatetoomanybeans` 表达了在管理和处理数据集时的挫败感，并幽默地提到在 `@c.gato` 的调侃后，尝试减少豆子摄入但未成功。
- **对统一系统消息的好奇**：`@le_mess` 询问了 100k 行数据集中系统消息（system messages）是否统一，`@xzuyn` 确认整个数据集确实共享相同的系统提示词（system prompt）。
- **探索 Neural Novel 数据库**：`@lee0099` 分享了 Huggingface 上的 [Neural-DPO](https://huggingface.co/datasets/NeuralNovel/Neural-DPO) 链接，展示了一个 AI 助手对 Aya 计划的询问以及参数高效专家 Ai(x) 公式的不同回答。

**提到的链接**：

[NeuralNovel/Neural-DPO · Datasets at Hugging Face](https://huggingface.co/datasets/NeuralNovel/Neural-DPO)：未找到描述

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1208905837276434433) (4 条消息): 

- **对 DPO 中神秘的“75 次迭代”感到困惑**：用户 `@filippob82` 对 DPO 评估阶段出现的固定 75 次迭代感到困惑，不确定其来源或目的。
- **DPO 评估可能无法使用**：根据两周前的经验，`@noobmaster29` 建议目前评估（evaluation）可能无法与 DPO 配合使用。
- **DPO 评估问题持续存在**：尽管进行了尝试，`@noobmaster29` 仍无法成功运行 DPO 中的评估，暗示该功能可能存在未解决的问题。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (1 条消息): 

c.gato: 用户错误 (User Error)。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/) (1 条消息): 

j_sp_r: https://venki.dev/notes/replicate-vs-fly
  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1207985062113579028) (52 messages🔥): 

- **微调困境**：`@david1542` 正在寻求关于针对销售等特定领域任务微调 LLM 的指导，强调其 Agent 缺乏对公司详细流程和上下文的理解。未引用或讨论具体的论文或博客文章，使得 `@david1542` 仍在寻找见解。
- **定价难题**：`@pasko70` 提出了 LangSmith 的定价问题，其中 trace 成本超过了 LLM 调用成本，使得该服务在低到中等 Token 吞吐量的应用中在财务上不可行。详细的成本明细显示了 Chain Cost 与 Trace Cost 之间的差异，但尚未给出解决方案或回应。
- **Vector DB 挑战**：`@cablecutter` 询问有关将 Whisper 转录文本处理到 Vector DB 中以进行分层摘要和 QA 的问题，操作重点是主题提取和摘要。他们在整合小型、短上下文片段及其交互方面面临挑战。
- **YouTube 与 Twitter 亮点**：`@jasonzhou1993` 和 `@davidzhou8571` 分享了讨论 OpenAI Agent 2.0 的 YouTube 视频链接，以及一条关于用户使用 LangChain、PrivateGPT 和 Ollama 工具通过本地 LLM 讨论文档创建 Multimodal RAG 的 Twitter 动态。
- **`langchain_community` 模块更新的多个问题**：用户 `@rajvir3` 和 `@rajib2189` 报告了 LangChain 更新导致错误的问题，特别是 `TextLoader` 模块针对 'pwd' 抛出 `ModuleNotFoundError`。`@dre99899` 建议有效的修复方法包括降级或根据 GitHub 更新手动编辑文件。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1038097195422978059/1208301752605220864): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Multi-LLM AI Gateway to run, secure, and govern AI traffic](https://konghq.com/products/kong-ai-gateway): 未找到描述
- [Tweet from zhouql1978 (@zhouql1978)](https://x.com/zhouql1978/status/1758419213319094592?s=20): Multimodal RAG：基于 LangChain @hwchase17 和 PrivateGPT @ivanmartit，我用不到 300 行代码构建了 Multimodal RAG。你可以使用本地 LLM @ollama 与任何文档对话，包括 Word...
- [langchain/libs/community/langchain_community/document_loaders/pebblo.py at d7c26c89b2d4f5ff676ba7c3ad4f9075d50a8ab7 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/d7c26c89b2d4f5ff676ba7c3ad4f9075d50a8ab7/libs/community/langchain_community/document_loaders/pebblo.py#L261C8-L262C23): 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账户为 langchain-ai/langchain 开发做出贡献。
- [Latest langchain_community is giving an error &quot;No MODULE PWD&quot; while using TEXTLOADER · Issue #17585 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/17585): 检查了其他资源，我为此 Issue 添加了非常详细的标题。我使用集成搜索搜索了 LangChain 文档。我使用 GitHub 搜索来查找类似问题并...
- [LangSmith For Beginners | Must know LLM Evaluation Platform 🔥](https://youtu.be/FgG-trkAMwU): 在这段视频中，我将向你展示如何将 LangSmith 集成到现有的 LangChain 项目中。LangSmith 是一个用于开发、协作的统一 DevOps 平台...
- [OpenAI&#39;s Agent 2.0: Excited or Scared?](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx): 我想为你全面介绍浏览器/移动端/桌面 AI Agent。获取免费的 HubSpot 电子书：使用生成式 AI 扩展你的内容运营：https://c...
- [GitHub - 13331112522/m-rag: Build your own Multimodal RAG Application using less than 300 lines of code.](https://t.co/4qJrES25Ak): 使用不到 300 行代码构建你自己的 Multimodal RAG 应用。 - 13331112522/m-rag
- [GitHub - Abraxas-365/langchain-rust: LangChain for Rust, the easiest way to write LLM-based programs in Rust](https://github.com/Abraxas-365/langchain-rust): Rust 版 LangChain，在 Rust 中编写基于 LLM 程序的简单方式 - Abraxas-365/langchain-rust

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1208080534291488822) (1 messages): 

- **RAG API 实现指导请求**：用户 `@mamo7410` 询问了关于使用 **langserve** 实现 RAG API 的问题。他们正在寻求关于如何为前端获取 **streaming**、**runtime ID** 和 **context documents** 的帮助，并提到 `stream_log` 可能是解决方案，但它产生的响应很复杂，且在网上找不到现成的示例。
  

---

### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 条消息): 

tumultuous_amicable: 哇，你绝对不想把你的 API Key 放在 Discord 频道里。
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1208031572658495488) (7 条消息): 

- **多模态 RAG 的新进展**：`@davidzhou8571` 分享了 `@zhouql1978` 的一条 [Twitter 帖子](https://x.com/zhouql1978/status/1758419213319094592?s=20)，该帖子展示了一个**多模态 RAG** 的创建，结合了 LangChain 和 PrivateGPT 的功能，可以与各种格式的文档进行对话，且代码量不到 300 行。
  
- **寻求对 Scribe 的反馈**：`@shving90` 正在征求对名为 Scribe 的项目的反馈，该项目在 [scribe.oranai.com](https://scribe.oranai.com/) 展示，但消息中未提供更多细节。

- **通过 Plastic Labs 实现全民记忆**：`@courtlandleer` 介绍了使用 Honcho 通过 Plastic Labs 对 OpenAI “记忆”功能的开源重新实现，并提供了一个 [Demo 和 Discord Bot](https://x.com/vintrotweets/status/1758274129768443946?s=20) 供测试，详情见其[博客文章](https://blog.plasticlabs.ai/blog/Memories-for-All)。

- **深入探讨 Whisper**：`@amgadoz` 撰写了一个深入的三部分博客系列，关于 OpenAI 的 Whisper ASR，涵盖了架构、模型的多任务接口以及开发过程，可在 Substack 上阅读：[文章 1](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link)、[文章 2](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link) 和 [文章 3](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link)。

- **LangChain 现在支持 Rust**：`@edartru.` 分享了一个 [GitHub](https://github.com/Abraxas-365/langchain-rust) 链接，发布了 Rust 版本的 LangChain 仓库，使得用这种编程语言编写基于 LLM 的程序变得更加简单。

- **AI 驱动的财务分析师教程**：`@solo78` 展示了一个教程，演示如何使用 OpenAI Assistant API 构建一个财务分析师，专注于分析保险公司的风险状况，详见 [Medium 文章](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e)。

**提到的链接**：

- [来自 zhouql1978 (@zhouql1978) 的推文](https://x.com/zhouql1978/status/1758419213319094592?s=20): 多模态 RAG：基于 Langchain @hwchase17 和 PrivateGPT @ivanmartit，我用不到 300 行代码构建了多模态 RAG。你可以使用本地 LLM @ollama 与任何文档进行对话，包括 Word...
- [OranScribe](https://scribe.oranai.com/): 集中式写作平台，利用 AI 进行构思、研究、撰写和编辑跨平台内容。写得更好、更快、更高效。
- [使用 AI 分析保险公司的风险概况：全面指南](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e): 使用 OpenAI Assistant API 对欧洲保险公司进行财务分析的使用案例。
- [来自 vintro (@vintrotweets) 的推文](https://x.com/vintrotweets/status/1758274129768443946?s=20): OpenAI 在周二发布了个性化记忆功能！看起来它只是根据你发送的消息推导事实，所以我们用 @LangChainAI 制作了一个机器人，展示如何使用 Honcho 来重现这一功能，而且...
- [全民记忆](https://blog.plasticlabs.ai/blog/Memories-for-All): 摘要 § 个性化是下一个前沿领域。OpenAI 意识到了这一点：我们正在测试 ChatGPT 记住你讨论过的事情的能力，以使未来的聊天更有帮助。
- [解码 Whisper：深入了解其架构和转录过程](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): 系列文章的第 2 部分，深入探讨 Whisper，OpenAI 最先进的自动语音识别模型。
- [探索 Whisper 的多任务接口：近距离观察其语音转录和翻译能力](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): 系列文章的第 3 部分，深入探讨 Whisper，OpenAI 最先进的自动语音识别模型。
- [Whisper 的诞生：深入探讨其训练数据和过程](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): 系列文章，深入探讨 Whisper，OpenAI 最先进的自动语音识别模型。
- [GitHub - Abraxas-365/langchain-rust: LangChain for Rust，在 Rust 中编写基于 LLM 程序的最简单方法](https://github.com/Abraxas-365/langchain-rust): LangChain for Rust，在 Rust 中编写基于 LLM 程序的最简单方法 - Abraxas-365/langchain-rust

  

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1208367940391534622) (6 messages): 

- **ChainLit 教程演示**：`@datasciencebasics` 分享了一个 [YouTube 视频](https://youtu.be/FZrkm0vaYYQ)，题为“使用 ChainLit / Streamlit, LangChain, Ollama & Mistral 与网站聊天 🧠”，演示了如何在本地创建简单的 Retrieval Augmented Generation (RAG) UI。
- **向量数据库上下文咨询**：`@tumultuous_amicable` 询问了关于使用向量数据库为模型提供上下文，而不是显式传递文本上下文的问题。
- **crewAI 股票数据教程**：`@business24.ai` 分享了一个 [YouTube 视频](https://youtu.be/Q5GUFCpEng4)，关于使用 LangChain 自定义工具将实时股票数据添加到 crewAI 并将结果存储在 Obsidian 中，并征求反馈以改进未来的视频。
- **使用 LangGraph 的网页浏览 Agent**：`@pradeep1148` 分享的 [YouTube 视频](https://www.youtube.com/watch?v=gbGYN3YyTS4) 展示了 "Web Voyager"，这是一个具备视觉能力的网页浏览 Agent，可以控制鼠标和键盘操作。
- **LangSmith LLM 评估平台介绍**：`@datasciencebasics` 发布了一个 [YouTube 视频](https://youtu.be/FgG-trkAMwU)，作为将 LangSmith 集成到现有 LangChain 项目中进行 LLM 开发和协作的入门指南。

**提到的链接**：

- [使用 LangGraph 的网页浏览 Agent](https://www.youtube.com/watch?v=gbGYN3YyTS4)：由 He 等人开发的 Web Voyager 是一个具备视觉能力的网页浏览 Agent，能够控制鼠标和键盘。它通过查看带注释的浏览器...
- [使用 LangChain 自定义工具将实时股票数据添加到 crewAI 并将结果存储在 Obsidian 中](https://youtu.be/Q5GUFCpEng4)：在本教程中，我们向 crewAI 添加了三个自定义工具。通过第一个自定义工具，我们将 crewAI 连接到我们的投资组合并获取当前持仓。通过...
- [使用 ChainLit / Streamlit, LangChain, Ollama & Mistral 与网站聊天 🧠](https://youtu.be/FZrkm0vaYYQ)：在此视频中，我将演示如何在计算机本地创建一个简单的 Retrieval Augmented Generation UI。你可以通过克隆...
- [LangSmith 初学者指南 | 必知的 LLM 评估平台 🔥](https://youtu.be/FgG-trkAMwU)：在此视频中，我将向你展示如何将 LangSmith 集成到现有的 LangChain 项目中。LangSmith 是一个用于开发、协作的统一 DevOps 平台...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1208033306092900382) (8 messages🔥): 

- **训练损失的临时飙升**：`@bjoernp` 指出，训练期间的高损失可能只是训练数据中的一个**离群值 (outlier)**，可能来自单个 Batch。
- **离群值可能干扰训练**：`@_jp1_` 向 `@philipmay` 建议，高训练损失可能是由于**“坏数据”**造成的，并建议手动检查数据，尤其是在 **batch size** 不太高的情况下。
- **Philip 分享训练配置**：`@philipmay` 分享了他的训练配置细节，包括 16 的 **micro_batch_size** 和 0.0002 的 **learning_rate** 等规格。
- **基座模型与专家配置的实验**：`@philipmay` 描述了使用 **VAGOsolutions/SauerkrautLM-13b-v1** 作为基座模型的成功经验，并将其与来自 **LeoLM** 的专家模型相结合，注意到其训练和评估损失与 **mixtral** 相当，并提到了由于缺少文件而在尝试使用 **LeoLM** 作为基座模型时遇到的问题。
- **关于语言模型预训练框架的讨论**：`@remek1972` 寻求关于从头开始预训练 Language Models 的框架建议，表示目标是创建一个不含英文内容的**国家级 LLM 模型**，`@philipmay` 在听取解释后表示理解其意图。
  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1208490102725283891) (1 messages): 

- **零分之谜已解开**：`@huunguyen` 分享了一个有趣的失误时刻，透露他们的模型在 **gsm8k 上得分为 0**，因为 Prompt 输出的 "### Response" 被误认为是数学题的答案。他们带着一丝幽默指出，在 gsm8k 数据集上进行预训练可能是一个好主意。
  

---

### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1208062982605639760) (4 messages): 

- **JinaAI 推出 jina-colbert-v1-en**: `@devnull0` 分享了来自 [JinaAI 的推文](https://fxtwitter.com/JinaAI_/status/1758503072999907825?t=1LT1ISg6BCXYdcr0yYEhHw&s=19)，宣布推出 **jina-colbert-v1-en**。该模型改进了 ColBERTv2 的 zero-shot 性能，并已在 [Hugging Face 上以 Apache 2.0 许可证发布](https://huggingface.co/jinaai/jina-colbert-v1-en)。

- **如何选择 Embedding 数据库**: `@huunguyen` 提到了过去使用 SQLite 和 Whoosh 进行 Embedding 的经验，并指出 Elasticsearch 是一个更偏向企业级的解决方案，其部署和配置并非易事。

- **高级搜索的替代方案**: `@huunguyen` 还提到了 Lucene/Solr 作为企业级搜索解决方案的替代品，并建议通过私信联系以获得更快回复，因为他并不经常查看 Discord。

**提到的链接**:

[来自 Jina AI (@JinaAI_) 的推文](https://fxtwitter.com/JinaAI_/status/1758503072999907825?t=1LT1ISg6BCXYdcr0yYEhHw&s=19): 介绍 jina-colbert-v1-en。它采用了 ColBERTv2 的 late interactions 和 token-level embeddings，在许多任务（域内和跨域）上具有更好的 zero-shot 性能。现已在 @huggingface 发布，采用 Ap...

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1208070420004208660) (3 messages): 

- **寻找德语评估教程**: `@thomasrenkert` 询问了有关构建德语**评估数据集（evaluation datasets）**的资源，特别是针对翻译和摘要任务。
- **提供评估数据集指导**: `@bjoernp` 提到虽然缺乏特定的教程，但他可以提供相关指导，并建议使用 **lm-evaluation-harness** 来处理 **chrf**、**bleu** 和 **rouge score** 等评估指标。
- **探索 lm-evaluation-harness 解决方案**: 在 `@bjoernp` 的建议下，`@thomasrenkert` 开始研究 **lm-evaluation-harness 示例**，将其作为满足需求的潜在资源。
  

---



### Skunkworks AI ▷ #[compute](https://discord.com/channels/1131084849432768614/1131302399370334248/) (1 messages): 

bluetyson: 这很有意思 —— 算力处理的众筹（kickstarter）？
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1208029743614664754) (5 messages): 

- **使用 Llama Factory 进行 Function Calling**: `@pradeep1148` 分享了一个题为“[使用 Llama Factory 微调模型以实现函数调用 (Tool Call)](https://www.youtube.com/watch?v=EYR_kd3X03M)”的 YouTube 视频，讨论了如何在 Python 编程中为 Function Calling 微调模型。
- **腾讯 YOLO-World 介绍**: `@pradeep1148` 提供的另一个[视频链接](https://www.youtube.com/watch?v=yaqi8xRUsp4)介绍了腾讯 AI Lab 的“YOLO-World：实时、Zero-Shot 目标检测”，展示了一个新的目标检测模型。
- **OpenAI SORA 文本生成视频模型**: `@pradeep1148` 分享了一个题为“[OpenAI SORA 文本生成视频模型及技术报告](https://www.youtube.com/watch?v=7lsOzA3WhSI)”的视频，介绍了 Sora，一个可以根据文本提示生成视频的模型。
- **网页浏览 Agent - LangGraph**: `@pradeep1148` 发布了一个关于 WebVoyager 的[视频链接](https://www.youtube.com/watch?v=gbGYN3YyTS4)，这是一个使用视觉能力来控制鼠标和键盘的网页浏览 Agent。
- **对分享内容的认可**: `@sabertoaster` 对 `@pradeep1148` 分享的内容回复了一个简单的“nice”，对讨论的视频和 AI 模型表示认可。

**提到的链接**:

- [使用 Llama Factory 微调模型以实现函数调用 (Tool Call)](https://www.youtube.com/watch?v=EYR_kd3X03M): 使用 Llama Factory 微调模型以实现函数调用 (Tool Call) #llm #ml #ai #largelanguagemodels #deeplearning #python #pythonprogramming https://github.c...
- [使用 LangGraph 的网页浏览 Agent](https://www.youtube.com/watch?v=gbGYN3YyTS4): Web Voyager。由 He 等人开发的 WebVoyager 是一个具备视觉能力的网页浏览 Agent，能够控制鼠标和键盘。它通过查看带注释的浏览器界面来工作...
- [YOLO-World: 实时、Zero-Shot 目标检测](https://www.youtube.com/watch?v=yaqi8xRUsp4): 2024 年 1 月 31 日，腾讯 AI Lab 发布了 YOLO-World（代码已托管至 GitHub），这是一个实时、开放词汇的目标检测模型。YOLO-World 是一个 zero...
- [OpenAI SORA 文本生成视频模型及技术报告](https://www.youtube.com/watch?v=7lsOzA3WhSI): 介绍 Sora，我们的文本生成视频模型。Sora 可以生成长达一分钟的视频，同时保持视觉质量并遵循用户的提示。#...

### Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/1209014891307073556) (2 messages): 

- **探索 RLHF 中 PPO 的替代方案**：`@nagaraj_arvind` 分享了关于 **RLHF (Reinforcement Learning from Human Feedback)** 的视频和论文，并介绍了 **DPO (Direct Preference Optimization)** 算法，将其作为 **OpenAI PPO** 的有力替代方案。[讲座视频](https://youtu.be/Ju-pFJNfOfY) 涵盖了针对大语言模型的 RLHF、PPO 和 DPO 基础知识，而 [DPO 论文](https://arxiv.org/abs/2305.18290) 提出了一种新的参数化方法，以改善模型与人类偏好的 Alignment。
- **对 KTO 的好奇**：`@salmon_lemon` 询问了关于 KTO 的信息，但在给定的聊天记录中没有提供进一步的细节或回复。

**提到的链接**：

- [RLHF, PPO and DPO for Large language models](https://youtu.be/Ju-pFJNfOfY)：强化学习、RLHF、Proximal Policy Optimization (PPO) 和 Direct Preference Optimization (DPO) 算法简介。
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)：虽然大规模无监督语言模型 (LMs) 学习到了广泛的世界知识和一些推理技能，但由于完全无监督，实现对其行为的精确控制仍然很困难...

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1208800226660450374) (1 messages): 

由于仅提供了一条消息，没有额外的上下文或回复，且消息中没有明确的链接或进一步的讨论点，摘要如下：

- **咨询关于本地 RAG 的 Llama-index**：用户 `@damiondreggs` 询问 **Llama-index** 是否仍是本地 Retrieval-Augmented Generation (RAG) 的可行工具，或者是否有更好的工具可用。未提供进一步的背景或附加信息。
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1208787634676179004) (3 messages): 

- **对 OpenSora 的好奇**：用户 `@cryptossssun` 表达了对 **OpenSora** 的兴趣，询问其他人是否也感兴趣。
- **逆向工程 Sora 的秘密**：`@rusch` 对 **OpenSora** 表现出兴趣，特别是对 Sora 的某些功能进行逆向工程。
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1208166697337356289) (1 messages): 

- **swyxio 提出的 AIEF 项目建议**：用户 `@swyxio` 通过发布一个来自 [Hacker News 的链接](https://news.ycombinator.com/item?id=39371297) 分享了 **AI Engineer Foundation** 的一个潜在项目，并特别提醒 `@296887155819675650` 和 `@705561973571452938` 关注。

**提到的链接**：

[no title found](https://news.ycombinator.com/item?id=39371297)：未找到描述

  

---


### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1208159139629367448) (2 messages): 

- **纽约 Generative AI 工作坊**：`@tanyarai` 宣布将于 2/26 举办 **NYC Developer Generative AI Workshop**，邀请参与者向行业专家学习，并参加使用来自 OpenAI, Google 和 Anthropic 模型的动手实践互动工作坊。[在此 RSVP](https://lu.ma/ai_workshop)，别忘了带上你的笔记本电脑！

- **面向 OSS 工具和模型爱好者的 AI 黑客松**：`@hackgoofer` 分享了由 AI Engineer Foundation 主办的关于 OSS 工具和模型的黑客松邀请，定于下周六举行，赞助商包括 Fireworks.ai 和 LlamaIndex.ai，并设有现金奖励。查看详情并[加入名单](https://partiful.com/e/e3arTNNboImbIKdgQjHs)以参加这场编程对决。

**提到的链接**：

- [Generative AI Developer Workshop · Luma](https://lu.ma/ai_workshop)：👋 纽约开发者们！2/26 请加入我们在 Flatiron 举办的专注于 Generative AI 开发的夜晚！我们将以行业专家的闪电演讲开始，然后进入...
- [RSVP to OSS Hackathon: Functional Calling + RAG Hackathon | Partiful](https://partiful.com/e/e3arTNNboImbIKdgQjHs?)：各位黑客同仁们，大家好！AI Engineer Foundation（你们友好的开源非营利邻居 - 网站：aie.foundation）正在举办一场 Function Calling + RAG 黑客松。我们很高兴地宣布...

  

---



---