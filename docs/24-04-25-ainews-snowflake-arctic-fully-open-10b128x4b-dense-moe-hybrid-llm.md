---
companies:
- snowflake
- databricks
- deepseek
- deepspeed
- nvidia
- stable-diffusion
- adobe
- apple
- llamaindex
- lmsys
- openai
date: '2024-04-26T01:33:53.546196Z'
description: '**Snowflake Arctic** 是一款备受关注的新型基础语言模型，采用 Apache 2.0 协议发布。该模型声称在数据仓库
  AI 应用方面优于 **Databricks**，并采用了受 **DeepSeekMOE** 和 **DeepSpeedMOE** 启发的混合专家（MoE）架构。该模型采用了类似于近期
  **Phi-3** 论文中提到的三阶段课程训练策略。


  在 AI 图像和视频生成领域，**英伟达（Nvidia）** 推出了 **Align Your Steps** 技术，旨在提升低步数下的图像质量；同时，**Stable
  Diffusion 3** 与 **SD3 Turbo** 模型在提示词理解和图像质量方面进行了对比。**Adobe** 启动了一个 AI 视频上采样项目，可将模糊视频增强至高清画质，尽管目前仍存在一些高分辨率伪影。


  **苹果（Apple）** 发布了开源的端侧语言模型，并罕见地提供了代码和训练日志，这与以往仅发布模型权重的做法有所不同。**Llama-3-70b** 模型在
  LMSYS 排行榜的英语查询类别中并列第一，而 **Phi-3**（4B 参数）在“香蕉逻辑”（banana logic）基准测试中表现优于 **GPT-3.5
  Turbo**。此外，相关演示还展示了 **Llama 3** 模型在 MacBook 设备上的快速推理和量化性能。'
id: 1d0c21a9-ca19-4b5a-977c-7e448365a79f
models:
- snowflake-arctic
- phi-3
- llama-3-70b
- llama-3
- stable-diffusion-3
- sd3-turbo
- gpt-3.5-turbo
original_slug: ainews-snowflake
people: []
title: Snowflake Arctic：完全开放的 10B+128x4B 稠密-MoE 混合大语言模型
topics:
- mixture-of-experts
- curriculum-learning
- model-release
- image-generation
- video-upscaling
- quantization
- inference-speed
- benchmarking
- model-comparison
- open-source
- on-device-ai
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月24日至4月25日的 AI 新闻。我们为您检查了 7 个 subreddits、[**373** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)和 **27** 个 Discord 社区（**395** 个频道，**5506** 条消息）。预计节省阅读时间（按 200wpm 计算）：**631 分钟**。

这条新闻需要一些解析，但对于在现代 AI 浪潮中一直保持低调的 Snowflake 来说，这是一次非常值得称赞的尝试。[Snowflake Arctic](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/) 的显著之处有几点，但可能不包括他们在首屏展示的那张令人困惑且难以产生共鸣的图表：

 
![image.png](https://assets.buttondown.email/images/8a45195d-2c7c-420b-a6cc-dcf124fc1d84.png?w=960&fit=max)
 

“企业级智能（Enterprise Intelligence）”这个概念可能会让人产生好感，特别是如果它能解释为什么他们选择在某些领域比其他领域做得更好：

 
![image.png](https://assets.buttondown.email/images/1ab962bf-83de-4037-8fc1-b0a1e1bfa9d3.png?w=960&fit=max)
 

这张图表*真正*以一种并不含蓄的方式展示了：Snowflake 基本上是在声称他们构建了一个在几乎所有方面都优于 Databricks 的 LLM，而后者是他们在数据仓库战争中的主要对手。（这肯定会让 Jon Frankle 和他那群 Mosaic 的伙伴们感到被冒犯了吧？）

下游用户并不*那么*关心训练效率，但另一个值得关注的点是模型架构——它借鉴了 [DeepSeekMOE](https://x.com/deepseek_ai/status/1745304852211839163) 和 [DeepSpeedMOE](https://arxiv.org/pdf/2201.05596) 的正确思路，即更多的 Expert = 更好的效果：

 
![image.png](https://assets.buttondown.email/images/bcd39b75-ae22-43fa-be38-72ce278d1140.png?w=960&fit=max)
 

文中没有提到 DeepSeek 使用的“shared expert”技巧。

最后，文中提到了一个三阶段的课程学习（curriculum）：

 
![image.png](https://assets.buttondown.email/images/21e13aaa-64f0-4924-b7d9-9c80f723e6ec.png?w=960&fit=max)
 

这与最近 Phi-3 论文中看到的类似策略相呼应：

 
![image.png](https://assets.buttondown.email/images/24598097-2055-4691-89dd-90c83d91df37.png?w=960&fit=max)
 

最后，该模型以 Apache 2.0 协议发布。

老实说，这是一个非常棒的发布，唯一糟糕的决定可能就是 [Snowflake Arctic cookbook](https://medium.com/@snowflake_ai_research/snowflake-arctic-cookbook-series-exploring-mixture-of-experts-moe-c7d6b8f14d16) 是发布在 Medium.com 上的。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

AI 图像/视频生成

- **Nvidia Align Your Steps**：在 /r/StableDiffusion 中，Nvidia 新推出的 [Align Your Steps 技术](https://www.reddit.com/gallery/1ccdt3x) 显著提升了低步数下的图像质量，允许以更少的步数生成高质量图像。配合 DPM~ 采样器效果最佳。
- **Stable Diffusion 模型对比**：在 /r/StableDiffusion 中，一项[当前 Stable Diffusion 模型的深度对比](https://www.reddit.com/gallery/1ccetp2)显示，SD Core 在手部和解剖结构方面表现最好，而 SD3 对 Prompt 的理解最强，但画面具有一种视频游戏感。
- **SD3 与 SD3-Turbo 对比**：[由 Stable Diffusion 3 和 SD3 Turbo 模型生成的 8 张图像](https://www.reddit.com/r/StableDiffusion/comments/1ccj3kc/4_images_by_sd3_and_4_images_by_sd3turbo_prompts/)，其 Prompt 由 Llama-3-8b 语言模型生成，主题涉及 AI、意识、自然与科技。

其他图像/视频 AI

- **Adobe AI 视频超分辨率 (Upscaling)**：[Adobe 令人印象深刻的 AI 超分辨率项目](https://www.theverge.com/2024/4/24/24138979/adobe-videogigagan-ai-video-upscaling-project-blurry-hd)能让模糊的视频看起来像高清画质。然而，[在高分辨率下，畸变和错误也变得更加明显](https://v.redd.it/8pi8t62btewc1)。
- **Instagram 换脸 (Face Swap)**：在 /r/StableDiffusion 中，[Instagram 垃圾信息发送者正利用 FaceFusion/Roop](https://www.reddit.com/r/StableDiffusion/comments/1cbu5cx/how_are_these_instagram_spammers_getting_such/) 在视频中创建极具说服力的换脸效果。在低分辨率视频中，且面部不离摄像头太近时，效果最为理想。

语言模型与聊天机器人

- **Apple 开源 AI 模型**：[Apple 发布了代码、训练日志以及多个版本的](https://www.macrumors.com/2024/04/24/apple-ai-open-source-models/)端侧语言模型，这与仅提供权重和推理代码的常规做法有所不同。
- **L3 与 Phi 3 性能**：L3 70B 在 LMSYS 排行榜的[英语查询中并列第一](https://i.redd.it/3fwedc7yqjwc1.png)。Phi 3（4B 参数）在香蕉逻辑基准测试中[击败了 GPT 3.5 Turbo](https://i.redd.it/h6nvy99vjewc1.png)（约 175B 参数）。
- **Llama 3 推理与量化**：一段[视频展示了 Llama 3 在 MacBook 上的快速推理](https://v.redd.it/qzg34xylgjwc1)。然而，[对 Llama 3 8B 进行量化](https://www.reddit.com/r/LocalLLaMA/comments/1cci5w6/quantizing_llama_3_8b_seems_more_harmful_compared/)（尤其是低于 8-bit 时）与其他模型相比，性能下降更为明显。

AI 硬件与基础设施

- **为 OpenAI 提供的 Nvidia DGX H200**：[Nvidia CEO 向 OpenAI 交付了一台 DGX H200 系统](https://i.redd.it/wnxzyyqurhwc1.jpeg)。一个[Nvidia AI 数据中心](https://www.youtube.com/watch?v=0JxowHz0JxM)如果完全建成，可以在几分钟内训练完 ChatGPT4，被描述为拥有“超凡脱俗的力量和复杂性”。
- **ThinkSystem AMD MI300X**：联想发布了 [ThinkSystem AMD MI300X](https://lenovopress.lenovo.com/lp1943-thinksystem-amd-mi300x-192gb-750w-8-gpu-board) 192GB 750W 8-GPU 板卡的产品指南。

AI 伦理与社会影响

- **Deepfake 裸体影像立法**：在十几岁女孩的推动下，[全美二十多个州的立法者](https://www.nytimes.com/2024/04/22/technology/deepfake-ai-nudes-high-school-laws.html)正在制定法案或已通过法律，以打击 AI 生成的未成年人色情图像。
- **政治中的 AI**：在 /r/StableDiffusion 中，[奥地利一个政党使用 AI 生成了一张比真人照片更具“男子气概”的候选人照片](https://www.reddit.com/gallery/1c7rikz)，这引发了关于在政治中利用 AI 歪曲现实的讨论。
- **AI 对话保密性**：在 /r/singularity 中，一篇文章认为，随着 AI Agent 掌握更多个人知识，[这种关系应该像医生和律师一样受到法律保护的保密性](https://www.reddit.com/r/singularity/comments/1cchgqs/ai_conversations_should_be_confidential_like/)，但企业很可能会拥有并使用这些数据。

幽默/迷因 (Memes)

- 网友分享了各种幽默的 AI 生成图像，包括[涂着小丑妆的耶稣](https://i.redd.it/jp8gnfr1njwc1.jpeg)、[抱着 Stable Diffusion 3 模型的咕噜](https://i.redd.it/p0fq574xvdwc1.png)，以及来自 [Bland AI 的营销内容](https://v.redd.it/8c9mt9os5iwc1)。

---

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**OpenAI 与 NVIDIA 的合作伙伴关系**

- **NVIDIA 向 OpenAI 交付 DGX H200**：[@gdb](https://twitter.com/gdb/status/1783234941842518414) 指出 NVIDIA 向 OpenAI 亲手交付了全球第一台 DGX H200，**黄仁勋（Jensen Huang）亲笔题词“旨在推动 AI、计算和人类进步”**。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783236039714189520) 强调了 DGX GH200 的特性，如 256 块 H100 GPU、1.3TB GPU 显存和 8PB/s 的互连带宽。
- **OpenAI 与 Moderna 建立合作伙伴关系**：[@gdb](https://twitter.com/gdb/status/1783529202974687527) 还提到了 OpenAI 与 Moderna 之间的合作，旨在利用 AI **加速药物发现与开发**。

**Llama 3 与 Phi 3 模型**

- **Llama 3 模型**：[@winglian](https://twitter.com/winglian/status/1783456379199484367) 通过 **PoSE 和 RoPE theta 调整，将 Llama 3 8B 的上下文长度扩展到了 96k**。[@erhartford](https://twitter.com/erhartford/status/1783273948022755770) 发布了 Dolphin-2.9-Llama3-70b，这是一个与其他团队合作创建的 **Llama 3 70B 微调版本**。[@danielhanchen](https://twitter.com/danielhanchen/status/1783214287567347719) 指出 **Llama-3 70b 的 QLoRA 微调速度比 HF+FA2 快 1.83 倍，且显存占用减少了 63%**，而 Llama-3 8b 的 QLoRA 微调可以运行在 8GB 显存的显卡上。
- **Phi 3 模型**：[@rasbt](https://twitter.com/rasbt/status/1783480053847736713) 分享了 Apple 关于 OpenELM 论文的细节，介绍了 **Phi 3 模型家族的 4 种尺寸（270M 到 3B）**。关键的架构变化包括**采用了源自 DeLighT 论文的逐层缩放策略**。实验表明，在参数高效微调方面，LoRA 和 DoRA 之间没有明显差异。

**Snowflake Arctic 模型**

- **Snowflake 发布开源 LLM**：[@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1783123091104936060) 宣布推出 Snowflake Arctic，这是一个**专为企业级 AI 设计的 480B Dense-MoE 模型**。它结合了一个 10B 参数的稠密 Transformer 和一个 128x3.66B 的 MoE MLP。[@omarsar0](https://twitter.com/omarsar0/status/1783176059694821632) 指出，该模型声称在实现类似的编码、SQL 和指令遵循等企业级指标时，**计算量比 Llama 3 70B 少 17 倍**。

**检索增强生成 (RAG) 与长上下文**

- **LLM 中的检索头 (Retrieval heads)**：[@Francis_YAO_](https://twitter.com/Francis_YAO_/status/1783446286479286700) 发现了检索头，这是一种负责 **LLM 长上下文事实性的特殊注意力头**。这些头具有通用性、稀疏性和因果性，并显著影响思维链（chain-of-thought）推理。屏蔽掉这些头会使模型对之前的重要信息视而不见。
- **用于高效 LLM 推理的 XC-Cache**：[@_akhaliq](https://twitter.com/_akhaliq/status/1783554087574733294) 分享了一篇关于 XC-Cache 的论文，该技术**为高效的 decoder-only LLM 生成缓存上下文**，而非进行即时处理。它在速度提升和显存节省方面表现出良好的前景。
- **RAG 幻觉测试**：[@LangChainAI](https://twitter.com/LangChainAI/status/1783165455521481191) 展示了如何使用 LangSmith **评估 RAG 流水线并测试幻觉**，通过对比检索到的文档来检查输出结果。

**AI 开发工具与应用**

- **用于集成 AI 的 CopilotKit**：[@svpino](https://twitter.com/svpino/status/1783488942152528327) 重点介绍了 CopilotKit，这是一个开源库，使将 AI 集成到应用中变得极其简单，允许你**将 LangChain Agent 引入应用、构建聊天机器人并创建 RAG 工作流**。
- **用于 LLM 用户体验的 Llama Index**：[@llama_index](https://twitter.com/llama_index/status/1783297521386934351) 展示了如何使用 create-llama 为你的 LLM 聊天机器人/Agent 构建**带有可展开来源和引用**的用户体验。

**行业新闻**

- **Meta 的 AI 投资**：[@bindureddy](https://twitter.com/bindureddy/status/1783296389671444521) 注意到 Meta 第二季度预测疲软，并计划在 AI 上投入数十亿美元，认为这是一个明智的策略。[@nearcyan](https://twitter.com/nearcyan/status/1783262638778278240) 调侃道，Meta 的 360 亿美元营收现在全都投入到了 GPU 中。
- **Apple 的 AI 公告**：[@fchollet](https://twitter.com/fchollet/status/1783544742565015954) 分享了 Apple 在 Kaggle 上举办的自动作文评分竞赛的 Keras 入门笔记本。[@_akhaliq](https://twitter.com/_akhaliq/status/1783557863069139270) 报道了 Apple 的 CatLIP 论文，该论文探讨了**在网络规模的图文数据上进行更快速预训练的 CLIP 级视觉识别**。

---

# AI Discord 摘要回顾

> 摘要的摘要总结

**1. Llama 3 和 Phi-3 的发布引发热议与对比**：**[Meta 的 Llama 3](https://huggingface.co/blog/llama3)**（8B 和 70B 版本）以及 **[Microsoft 的 Phi-3](https://huggingface.co/chat/models/microsoft/Phi-3-mini-4k-instruct)** 模型的发布引起了巨大轰动。讨论集中在它们的性能对比、**RoPE** 等架构细节，以及 Phi-3 的 **function_call tokens** 能力。Llama 3 在 **MMLU** 和 **Human Eval** 等基准测试中的出色表现受到了重点关注。

**2. RAG 框架与多模态模型的进展**：讨论了使用 **[LangChain 的 LangGraph](https://medium.com/ai-advances/unifying-rag-frameworks-harnessing-the-power-of-adaptive-routing-corrective-fallback-and-1af2545fbfb3)** 对 **Retrieval-Augmented Generation (RAG)** 框架进行的改进，重点介绍了 **Adaptive Routing** 和 **Corrective Fallback** 等技术。此外，还涵盖了 **[Apple 的 OpenELM-270M](https://huggingface.co/collections/apple/openelm-pretrained-models-6619ac6ca12a10bd0d0df89e)** 的发布，以及对 **moondream** 等多模态模型在任务处理中的兴趣。

**3. 开源工具与模型部署**：用于构建 RAG 应用的 **[Cohere's Toolkit](https://github.com/cohere-ai/cohere-toolkit/)** 宣布开源并受到欢迎，同时探索了使用 **[Datasette 的 LLM](https://llm.datasette.io/en/stable/python-api.html)** Python API 进行文本向量化（embedding）。关于高效 **batching prompts** 的讨论涉及了 **[vLLM](https://github.com/vllm-ai/vllm)**、**[TGI](https://github.com/huggingface/text-generation-inference)** 和 **[llm-swarm](https://github.com/huggingface/llm-swarm)** 等工具。

**4. 专业模型与小众应用**：医疗领域的 **[Internist.ai 7b](https://huggingface.co/internistai/base-7b-v0.2)** 模型表现令人印象深刻，在评估中甚至超过了 GPT-4，引发了广泛关注。此外还展示了一些独特项目，如 AI 驱动的文字 RPG 游戏 **[Brewed Rebellion](https://www.playlab.ai/project/clveh1d860lwsrv1nvl4xfksp)**，以及旨在将 AI 嵌入设备的 **[01 project](https://github.com/Abdullah-Gohar/01.git)**。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **微调修复与未来展望**：Unsloth Pro 正在开发 **multi-GPU support**，预计在 5 月发布开源版本。同时，[Unsloth GitHub](https://github.com/unslothai/unsloth/pull/377) 上的 PR #377 提议解决模型中的 **vocabulary resizing issues**，社区对此充满期待。

- **LlaMA 在无负担下运行更佳**：从 prompt 中移除多余的 token 为改进 LLaMA 模型的 fine-tuning 指明了方向，显著地将初始 epoch 的 loss 从 5 降低到了更温和的范围。微调 70B 参数的 LLaMA-3 模型理想配置至少需要 48GB GPU VRAM，但建议配备更充足的资源。

- **优化 AI 的算力效率**：关于最佳 reranker 的辩论最终倾向于 **ms Marco l6 V2**，它比 BGE-m3 快 15 倍；而 **PostgreSQL 中的 pgvector** 则避开了对外部 API 的需求。在硬件方面，新购置的设备正助力于大型 **RAG datasets** 的微调以及创新的无监督 **in-context learning** 探索。

- **Meta 在模型马拉松中的实力**：Meta 推出的 **LlaMA-3 系列**引发了讨论，其 8B 模型以及预告的 400B 模型旨在挑战 GPT-4 的基准。随着 LlaMA-3 和 Phi-3 的发布，开源 AI 势头正劲，两者目标相似但策略各异，详见这篇 [Substack 文章](https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training)。

- **训练技术小贴士**：建议包括利用 Colab notebooks 解决微调中的故障，利用 **GPT3.5 或 GPT4** 编写多选题，以及在 Kaggle 上继续微调。此外还讨论了稀疏化 embedding 矩阵和动态调整 context length 的方法，并对可能出现的模型大小与 GPU 适配不符的预警系统表示了兴趣。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Phi 和 TinyLLamas 成为焦点**：社区成员一直在 LM Studio 中测试 **phi-3**，使用诸如 *PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed* 之类的模型来研究量化差异，其中 Q4 在文本生成方面的表现优于 Q2。与此同时，一系列 **TinyLlamas** 模型在 [Hugging Face](https://huggingface.co/DavidAU) 上引起了关注，为玩家提供了尝试 1B 到 2B 范围的 Mini-MOE 模型的机会。此外，尽管存在 token 限制，Apple 的 OpenELM 的推出也在社区中引起了热议。

**应对 GPU 相关问题**：GPU 话题占据了中心舞台，讨论内容包括显存密集型的 **phi-3-mini-128k** 模型、通过升级到 LM Studio v0.2.21 来避免 “(Exit code: 42)” 等错误的策略，以及解决 GPU offload 错误。此外，技术建议层出不穷，尽管一些成员对该品牌有所顾虑，但仍推荐将 Nvidia GPU 用于 AI 应用，并建议升级到 32GB RAM 以进行稳健的 LLM 实验。

**ROCm 领域的架构纠葛**：AMD 和 NVIDIA 混合 GPU 环境导致 **ROCm** 安装出现错误，临时的解决办法包括卸载 NVIDIA 驱动程序。然而，**heyitsyorkie** 强调 LM Studio 中的 ROCm 仍处于技术预览阶段，出现波动是预料之中的。社区智慧提供的解决方案包括更新驱动程序，例如为 **rx7600** 更新到 **Adrenalin 24.3.1**，以解决兼容性和性能问题。

**Mac 运行 LLM 的表现各异**：Mac 用户发表了看法，认为 16GB RAM 是流畅运行 LLM 的理想最低配置，不过在不超负荷运行并行任务的情况下，拥有 8GB RAM 的 M1 芯片也可以处理较小的模型。

**本地服务器经验谈**：关于访问 LM Studio 本地服务器的策略分享重点介绍了使用 *Mashnet* 进行远程操作，以及 Cloudflare 在促进连接方面的潜在作用，对行之有效的 “localhost:port” 设置进行了更新。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Enigmagi 获得巨额投资**：[Enigmagi](https://perplexity.ai) 庆祝筹集了 **6270 万美元** 的资金，估值达到 **10.4 亿美元**，NVIDIA 和 Jeff Bezos 等重量级投资者均参与其中。同时还为拥有现有 Pro 级订阅的 iOS 用户推出了 **Pro 服务**。

- **Perplexity Pro 用户争论其价值**：一些用户对 Perplexity 上 **Enterprise Pro** 相较于 Regular Pro 的优势持怀疑态度，特别是在性能差异方面，同时对于 Opus 搜索每日 50 次的使用限制也存在不满。

- **Perplexity 平台的变革之声**：用户对调整 **temperature 设置**以获得更好的创意输出表现出兴趣。虽然一些人讨论了新的语音功能，但其他人希望有更多功能，例如 **iOS 手表小组件**。与此同时，Pro Support 的可访问性问题也被提出，表明用户体验有待改进。

- **API 忧虑与 Groq 的潜力**：在 pplx-api 的讨论中，社区了解到 **图片上传** 将不会成为 API 的一部分，而对于代码辅助，推荐使用 **llama-3-70b instruct** 和 **mixtral-8x22b-instruct**。同时，**GPT-4 尚未集成**，当前的模型详情可以在 [文档](https://docs.perplexity.ai/docs/model-cards) 中找到。

- **跨频道的內容对话**：在 Perplexity AI 上的各种搜索解决了从克服 **语言障碍** 到 **系统思维** 的各种话题。一份 [分析](https://www.perplexity.ai/search/traduci-SG4MY85cTp6.22Ffm87a9A) 提供了关于翻译挑战的视角，而像 [Once I gladly](https://www.perplexity.ai/search/Once-I-gladly-7OdIRseoTfuMuhtGlY7gCw) 这样的链接推导了关于幸福感随时间变化的讨论，[Shift No More](https://www.perplexity.ai/search/Shift-No-More-oo2vIU0NSuaymGTbDWDELA) 则探讨了变革的必然性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**关于 RoPE 的深度讨论**: 社区辩论了 **Rotary Position Embedding (RoPE)** 在 Meta 的 **Llama 3** 等模型中的能力，包括其在微调（fine-tuning）与预训练（pretraining）中的有效性，以及关于其在长上下文（longer contexts）中泛化能力的误解。关于 "Scaling Laws of RoPE-based Extrapolation" ([arXiv:2310.05209](https://arxiv.org/abs/2310.05209)) 的论文引发了关于扩展 RoPE 以及在增加 RoPE base 时如何避免灾难性遗忘（catastrophic forgetting）挑战的对话。

**AutoCompressors 加入战场**: 一篇关于 **AutoCompressors** 的新预印本提出了一种让 Transformer 处理多达 30,720 个 token 并提高困惑度（perplexity）的方法 ([arXiv:2305.14788](https://arxiv.org/abs/2305.14788))。Jeremy Howard 对 **Llama 3** 及其微调策略的看法在社区中引起共鸣 ([Answer.AI post](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html))，一条 Twitter 线程揭示了使用先进方法成功将其上下文扩展到 96k ([Twitter Thread](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg))。

**LLM 教育与 Apple Vision Pro 降温**: 社区讨论了一款旨在教授 LLM 提示词注入（prompt injections）的游戏 ([Discord Invite](https://discord.gg/FYjptek5zY))。在硬件传闻方面，据报道 Apple 将其 **Vision Pro 出货量** 削减了 50%，并正在重新评估其头显策略，引发了对 2025 年产品线的猜测 ([Tweet by @SawyerMerritt](https://x.com/sawyermerritt/status/1782895962131702211?s=46))。

**Snowflake 的混合模型与模型对话**: **Snowflake Arctic 480B** 发布了一种独特的 Dense + Hybrid 模型，引发了对其架构选择的分析性讨论，并提到了其为上下文缩放设计的 attention sinks。同时，关于 **GPT-3** 动态的讨论引发了对它是否真的在运行 OpenAI 的 **Rabbit R1** 的怀疑。

**用于可靠引用的 Pydantic 模型**: 带有验证器（validators）的 Pydantic 模型被推崇为确保 **LLM contexts** 中正确引用的方法；讨论引用了几个 GitHub 仓库 ([GitHub - argilla-io/distilabel](https://github.com/argilla-io/distilabel)) 和像 lm-format-enforcer 这样用于保持可靠响应的工具。

**使用 WorldSim 进行流媒体创作**: 社区成员交流了使用 **WorldSim** 的经验，并建议了在 Twitch 上直播共享世界模拟的潜力。他们还分享了一个自定义角色树 ([Twitter post](https://twitter.com/Jtronique/status/1783291232795471938/photo/1))，并讨论了涉及 **types ontology and morphisms** 的范畴论（category theory）应用 ([Tai-Danae Bradley 的工作](https://www.math3ma.com/about))。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**PyTorch 2.3: Triton 与张量并行成为焦点**: PyTorch 2.3 增强了对**用户定义 Triton kernels** 的支持，并改进了用于训练参数量高达 1000 亿的 *Large Language Models (LLMs)* 的 **Tensor Parallelism**，所有这些都由 426 位贡献者验证 ([PyTorch 2.3 Release Notes](https://pytorch.org/blog/pytorch2-3/))。

**关于 GPU 功率限制的思考**: 围绕 A100 和 H100 等 GPU 功率限制（power-throttling）架构展开了热烈讨论，并对 B100 的设计可能影响计算效率和功率动态表示期待。

**CUDA 开发者发现 Kernel 优化空间**: 成员们分享了优化 CUDA kernels 的策略，包括避免使用 `atomicAdd` 以及利用 Volta 架构之后的 warp 执行改进，这些改进允许 warp 中的线程执行不同的指令。

**Accelerated Plenoxels 作为 CUDA 强化的 NeRF**: 热情指向了 **Plenoxels**，因为它具有高效的 NeRF CUDA 实现，同时也表达了对 GPU 加速 SLAM 技术以及针对深度学习模型中 attention 机制优化的 kernel 的兴趣。

**PyTorch CUDA 进展、Flash-Attention 特性与内存管理**: 源代码显示了对张量乘法（tensor multiplications）的内存高效处理，这与 COO 矩阵表示有相似之处。它还强调了一个关于 [Triton kernel 崩溃](https://kolinko.github.io/effort/gpu.html) 的潜在问题，即尝试访问超出原始范围的扩展张量索引。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Pre-LayerNorm 辩论**：一位工程师强调了一项**分析**，指出 Pre-LayerNorm 可能会阻碍 residual stream 中信息的删除，可能导致范数随层数增加而增长。

**Tokenizer 版本之争**：Huggingface tokenizer 版本 0.13 和 0.14 之间的变化导致了不一致，造成模型推理过程中的 **token 对齐偏差**，引起了 NeoX 开发成员的关注。

**Poetry 的打包难题**：在 NeoX 开发中尝试使用 Poetry 进行包管理失败后（由于其二进制文件和版本管理存在问题），成员认为其实现过于复杂。

**Chinchilla 的置信度困惑**：社区成员质疑 **Chinchilla 论文**中置信区间的准确性，怀疑**小参数 Transformer 采样过度**，并讨论了稳定估计的正确截断点。

**大型推荐系统启示**：Facebook 发布了一个约 1.5 万亿参数的**基于 HSTU 的生成式推荐系统**，成员们强调其性能提升了 12.4% 及其潜在影响。[论文链接](https://arxiv.org/abs/2402.17152v2)。

**Penzai 的迷惑用法**：用户发现 **penzai** 的用法不直观，分享了处理 named tensors 的变通方法和实际案例。讨论包括使用 untag+tag 方法以及用于标签操作的函数 **`pz.nx.nmap`**。

**评估大模型**：一位处理自定义任务的用户报告了高 perplexity，并正在寻求关于 **CrossEntropyLoss** 实现的建议；同时，另一个关于 benchmark 的 `num_fewshot` 设置的讨论也随之展开，旨在匹配 Hugging Face 排行榜。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **RealVis V4.0 胜过 Juggernaut**：工程师们讨论了他们更倾向于使用 **RealVis V4.0** 来实现更快、更令人满意的图像 prompt 生成，而非 Juggernaut 模型，这表明性能依然优于全新的模型。

- **Stable Diffusion 3.0 API 使用担忧**：人们对 **Stable Diffusion 3.0** 充满期待，但在得知新 API 不是免费的且仅提供有限的试用额度后，一些人表示失望。

- **Craiyon，AI 初学者的工具**：对于需要图像生成帮助的新手，社区资深人士推荐 [Craiyon](https://www.craiyon.com/) 作为更复杂的、需要本地安装的 Stable Diffusion 工具的易用替代方案。

- **攻克 AI 模型微调挑战**：对话涵盖了从生成特定图像 prompt 到 vast.ai 等云计算资源、处理 AI 视频创建以及微调问题，讨论提供了关于训练 LoRas 和遵守 Steam 规定的见解。

- **探索独立 AI 项目**：公会中成员们踊跃分享各种基于 AI 的独立项目，例如 [artale.io](https://www.artale.io) 上的网络漫画生成和 [adorno.ai](https://adorno.ai) 上的无版权声音设计。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral 8x7b 空白响应危机**：**Mixtral 8x7b** 服务遇到了空白响应的问题，导致暂时移除了一家主要供应商，并计划未来加入**自动检测**功能。

**模型霸权辩论持续升温**：在讨论中，成员们将 **Phi-3** 等小模型与 **Wizard LM** 等大模型进行了比较，并报告称来自 Fireworks 的 **FireFunction**（[使用函数调用模型](https://readme.fireworks.ai/docs/function-calling)）可能是更好的替代方案，因为 **OpenRouter** 在 function calling 和遵守 'stop' 参数方面存在挑战。

**流式传输中的超时问题**：多位用户报告了大量旨在保持连接活跃的 "OPENROUTER PROCESSING" 通知，以及在 [OpenRouter](https://openrouter.ai/playground?models=openai/gpt-3.5-turbo) 上使用 **OpenAI 的 GPT-3.5 Turbo** 时 completion 请求超时的问题。

**寻求本地化 AI 业务扩张**：一位成员寻找直接联系方式，表明有兴趣在中国建立更紧密的 AI 模型业务联系。

**AI 讨论中的语言障碍**：AI 工程师比较了 **GPT-4**、**Claude 3 Opus** 和 **L3 70B** 等 AI 模型的语言处理能力，特别指出 **GPT-4** 在俄语方面的表现差强人意。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Llama 3 跨越式领先**：全新的 **Llama 3** 语言模型已经发布，它在高达 15T tokens 的数据上进行训练，并使用 10M 人工标注样本进行了微调。该模型提供 8B 和 70B 两个版本，在 MMLU 基准测试中得分超过 80，并展示了令人印象深刻的代码能力，8B 模型的 Human Eval 得分为 62.2，70B 模型为 81.7；通过 [Demo](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct) 和 [Blogpost](https://huggingface.co/blog/llama3) 了解更多信息。

**Phi-3：移动端模型奇迹**：微软的 **Phi-3** Instruct 模型变体因其紧凑的尺寸（4k 和 128k 上下文）以及在标准基准测试中优于 Mistral 7B 和 Llama 3 8B Instruct 等其他模型的表现而备受关注。Phi-3 专为移动端使用而设计，具有 'function_call' tokens 并展示了先进的能力；可以通过 [Demo](https://huggingface.co/chat/models/microsoft/Phi-3-mini-4k-instruct) 和 [AutoTrain Finetuning](https://x.com/abhi1thakur/status/1782807785807159488) 了解更多并进行测试。

**OpenELM-270M 与 RAG 更新**：苹果的 **OpenELM-270M** 模型在 HuggingFace 上引起了轰动，同时检索增强生成 (RAG) 框架也取得了进展，现在包括使用 Langchain 的 LangGraph 实现的 Adaptive Routing 和 Corrective Fallback 功能。这些以及其他讨论标志着 AI 领域的持续创新；关于 RAG 增强的详细信息请见[此处](https://medium.com/ai-advances/unifying-rag-frameworks-harnessing-the-power-of-adaptive-routing-corrective-fallback-and-1af2545fbfb3)，苹果的 OpenELM-270M 可在[此处](https://huggingface.co/collections/apple/openelm-pretrained-models-6619ac6ca12a10bd0d0df89e)获取。

**Batching 讨论升温**：模型推理过程中高效 **Batching** 的必要性引起了社区的兴趣。Aphrodite、tgi 等库因其卓越的 Batching 速度而被推荐，并有报告称使用数组进行并发 Prompt 处理取得了成功，建议数组可以像 `prompt = ["prompt1", "prompt2"]` 这样使用。

**虚拟环境配置难题**：一位成员在 Windows 上设置 **Python 虚拟环境**时遇到的挑战引发了讨论和建议。Windows 的推荐命令是 **`python3 -m venv venv`**，随后执行 **`venv\Scripts\activate`**，并建议尝试使用 WSL 以获得更好的性能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**思维树 (Trees of Thought)**：具备 **树搜索规划能力的 LLM** 的开发可能会为 Agent 系统带来重大进展，正如 [LlamaIndex 的一条推文](https://t.co/cdpvmVNHFj) 所披露的那样。这标志着从顺序状态规划的飞跃，暗示了 AI 决策模型可能取得的长足进步。

**观察知识之舞**：一个使用 Vercel AI SDK 开发的新型 **动态知识图谱工具** 可以流式传输更新，并在 [官方 Twitter](https://t.co/Mct2EJpGmG) 的帖子中进行了演示。这种可视化技术可能会成为实时数据表示领域的游戏规则改变者。

**你好，首尔**：**LlamaIndex 韩国社区** 的引入预计将促进韩国科技界的知识共享与合作，正如在 [推文](https://t.co/bSS0DzWrN5) 中宣布的那样。

**提升聊天机器人交互性**：使用 `create-llama` 对聊天机器人用户界面进行的增强已经出现，允许扩展来源信息组件，并承诺提供更直观的聊天体验，感谢 @MarcusSchiesser 并在 [推文](https://t.co/7FctwxzBIE) 中提及。

**让 Embeddings 变得简单**：关于构建结合了 **LlamaParse**、JinaAI embeddings 和 Mixtral 8x7b 的高质量 RAG 应用的完整教程现已发布，可以通过 LlamaIndex 的 [Twitter 动态](https://t.co/rXAV3mKVkJ) 获取。对于寻求有效解析、编码和存储 embeddings 的工程师来说，这份指南可能是关键。

**严谨的高级 RAG**：配置高级 RAG 流水线需要深入学习，为了应对复杂的提问结构，建议考虑句子窗口检索 (sentence-window retrieval) 和自动合并检索 (auto-merging retrieval) 等方案，正如一份 [教学资源](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval) 所指出的。

**VectorStoreIndex 的困惑**：关于 VectorStoreIndex 的 embeddings 和 LLM 模型选择的困惑得到了澄清；除非在 `Settings` 中覆盖，否则 **gpt-3.5-turbo** 和 **text-embedding-ada-002** 是默认设置，正如在多次讨论中所述。

**Pydantic 难题**：Pydantic 与 LlamaIndex 的集成在结构化输出和 Pyright 对动态导入的不满方面遇到了障碍。目前的讨论尚未达成除 `# type:ignore` 之外的替代方案。

**增强文档的请求**：用户请求提供更透明的文档，以说明如何设置高级 RAG 流水线以及在 LlamaIndex 中配置 **GPT-4** 等 LLM，并提到了修改全局设置或直接将自定义模型传递给 `query engine`。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AI 寻求真正的理解**：一场围绕 AI 是否能实现“真正的理解”的辩论展开，其中 Transformer 等自回归模型的图灵完备性是一个关键点。逻辑语法与语义的融合被认为是模型实现意义驱动操作的潜在促成因素。

**从语法到语义**：对话围绕 AI 领域语言的演变展开，预测将出现新概念以提高未来沟通的清晰度。语言的损耗性对准确表达思想的局限性也被强调。

**Apple 转向开源？**：围绕 Apple 推出的高效开源语言模型 **OpenELM**，人们充满了兴奋与猜测，讨论了这对其传统专有 AI 技术方法的影响以及更广泛的开源趋势。

**通讯与 AI 的碰撞**：成员们强调了 AI 辅助沟通中有效流控制的重要性，探索了语音转文本和自定义唤醒词等技术。讨论 AI 与沟通的相互作用，凸显了虚拟助手交互中中断与恢复机制的需求。

**带有 AI 色彩的 RPG 游戏**：分享了 AI 驱动的文字 RPG 游戏 *Brewed Rebellion*，展示了将 AI 集成到互动游戏体验中的日益增长的趋势，特别是在处理公司内部政治等叙事场景中。

**工程化优化 AI 行为**：工程师们分享了 Prompt 编写技巧，强调使用正面示例以获得更好结果，并指出负面指令往往无法约束 GPT 等 AI 的创意输出。

**游戏及其他领域的 AI 编程挑战**：一位为《武装突袭 3》(Arma 3) 开发 SQF 语言的工程师提出了在向 GPT 寻求特定语言编程协助时的挑战。讨论了模型预训练偏差和有限上下文空间等问题，引发了对替代模型或工具链的建议。

**动态 AI 更新与能力**：出现了关于 AI 更新和能力的咨询，包括如何在 Apple Playgrounds 中创建 GPT 专家，以及新版 GPT 是否能与 **Claude 3** 等模型匹敌。此外，对比了 GPT 内置浏览器与 **Perplexity AI Pro** 和 **You Pro** 等专用选项的效用，并表达了对具有更大上下文窗口模型的期待。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 大联盟 - 模型评分见解**：**general** 频道对一系列 AI 模型进行了热烈辩论，其中 **Llama 3 8B** 被比作 **GPT-4**。隐私担忧被提及，暗示由于新的 **美国“Know Your Customer”** 法规，“匿名云端使用时代即将结束”，并有人呼吁审查 AI 图像模型排行榜。

- **隐私风险 - 云计算监管引发辩论**：拟议的美国法规引起了成员们对 **云服务匿名性** 未来的不安。在 TorrentFreak 发表了一篇关于云服务提供商监管的文章后，其作为新闻源的可信度得到了辩护。

- **尖端还是过头 - AI 图像模型备受审视**：讨论质疑了 AI 图像模型排行榜的准确性，暗示可能存在结果操纵和对抗性干扰。

- **艺术胜过精确？AI 图像偏好之谜**：审美吸引力与 Prompt 忠实度是生成式 AI 输出讨论的核心，对比鲜明的偏好揭示了 AI 生成图像价值的主观性。

- **更快、更精简、更智能：通过新研究加速 AI**：**research** 频道最近的讨论强调了 **MH-MoE**（一种改进 **Sparse Mixtures of Experts (SMoE)** 上下文理解的方法）以及一种 **弱监督预训练技术**，该技术在不损害视觉任务质量的情况下，比传统对比学习快 2.7 倍。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Llama 勇攀新高度**：讨论吸引了参与者，因为 **Llama-3** 有潜力通过 **Tuning** 和增强训练扩展到巨大的 **128k** 尺寸。大家对 **Llama 3** 的预训练学习率（learning rate）也表现出浓厚兴趣，推测可能正在开发一种无限的 LR schedule，以配合即将推出的模型变体。

**Snowflake 的新发布引发热潮**：**Snowflake 408B Dense + Hybrid MoE 模型** 引起了轰动，它拥有 4K 上下文窗口并采用 **Apache 2.0** 许可。这引发了关于其内在能力以及如何与 **Deepspeed** 协同工作的热烈讨论。

**医疗 AI 迈出健康的一大步**：由医疗专业人士精心设计的 **Internist.ai 7b** 模型据报道表现优于 **GPT-3.5**，甚至在 USMLE 考试中取得了高分。它引发了关于专用 AI 模型前景的讨论，其性能以及它优于许多其他 7b 模型的惊人表现令人着迷。

**聚焦数据集与模型训练难题**：技术讨论深入探讨了 **Hugging Face datasets** 的实用性、优化数据使用，以及优化器与 **Fully Sharded Data Parallel (FSDP)** 设置之间的兼容协作。在同一话题下，成员们在处理反量化（dequantization）和全量微调（full fine tunes）时遇到了 **fsdp** 的波动，这表明存在更深层次的兼容性和系统问题。

**ChatML 的换行怪癖引起关注**：参与者发现了 **ChatML** 以及可能的 **FastChat** 中关于异常换行符和空格插入的故障。该问题凸显了精细化 token 配置的重要性，因为它可能会扭曲 AI 模型的训练结果。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 应对人脸识别隐私问题**：探讨了将 [Fawkes](https://github.com/Shawn-Shan/fawkes)（一种旨在挫败人脸识别系统的隐私工具）移植到 **tinygrad** 的可能性。George Hotz 建议战略合作伙伴关系对 tinygrad 的成功至关重要，并以与 comma 在 **tinybox** 硬件上的合作为例。

- **Linkup Riser 的反抗与冷却方案**：PCIE 5.0 LINKUP 转接卡（risers）导致错误的问题非常显著，一些工程师建议探索 **mcio** 或定制的 **[C-Payne PCBs](https://c-payne.com)**。此外，一名成员报告了尝试水冷的经历，但遇到了 NVLink 适配器的兼容性问题。

- **追求 Tinygrad 文档**：有人指出 tinygrad 规范文档存在空白，这促使人们要求对 tinygrad 操作的行为进行清晰描述。这包括关于需要 tensor 排序功能的对话，以及针对长度为 2 的幂的自定义 1D bitonic merge sort 函数的干预。

- **GPU Colab 对教程的需求**：George Hotz 分享了一个针对 GPU colab 用户的 [MNIST 教程](https://tinygrad.github.io/tinygrad/mnist/)，旨在作为帮助更多用户发挥 tinygrad 潜力的资源。

- **排序、循环与崩溃内核讨论会**：AI 工程师们努力解决 tinygrad 和 CUDA 的各个方面，从创建 **torch.quantile** 等效功能的复杂性，到揭示 tensor cores 的架构细微差别（如 **m16n8k16**），以及难以定位的神秘崩溃。关于 WMMA 线程容量的讨论显示，一个线程每个输入可能持有高达 128 bits。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**基准测试中的大胆举措**：工程界正期待 **Mojo 的性能基准测试**，将其与 Rust 和 Python 等语言进行对比，尽管 Rust 爱好者们对此持怀疑态度。[Lobsters](https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6) 上关于 Mojo 声称比 Rust 更安全、更快的说法展开了激烈辩论，这是 Mojo 在技术圈叙事的核心。

**量子难题与 ML 解决方案**：量子计算的讨论涉及了量子随机性的细微差别，并提到了**多世界诠释 (Many-Worlds)** 和**哥本哈根诠释 (Copenhagen interpretations)**。目前关于在**量子算法**中利用几何原理和 ML 来处理 qubit 复杂性并提高计算效率的讨论非常热烈。

**修复 Mojo Nightly 版本**：Mojo 社区在 GitHub ([#239two](https://github.com/modularml/mojo/issues/2392)) 上记录了一个空字符串 bug，并迎来了新的 nightly 编译器版本，该版本改进了函数参数的重载。同时，SIMD 对 `EqualityComparable` 的适配揭示了利弊，引发了对更高效 stdlib 类型的探索。

**保障软件供应链安全**：鉴于 [XZ 供应链攻击](https://www.synopsys.com/blogs/software-security/open-source-trends-ossra-report.html)，Modular 的博客文章强调了为 **Mojo 安全软件交付**制定的安全协议。通过 SSL/TLS 和 GPG 等安全传输和签名系统，Modular 在保护其不断发展的软件生态系统方面迈出了坚实的一步。

**Discord 社区预见周边与语法交换**：Mojo 开发者社区愉快地讨论了变量命名的建议，并期待未来的官方周边 (swag)；与此同时，API 开发引发了关于**性能和内存管理**的讨论。关于 MAX 引擎的查询被重定向到特定频道，以确保沟通的高效。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Transformer 的新视角**：工程师们讨论了通过引入来自**中间注意力层**的输入来增强 **Transformer 模型**，这与 **CNN 架构**中的金字塔网络 (Pyramid network) 方法类似。这种策略可能会改进上下文感知处理和信息提取。

**关于 'TherapistAI' 的伦理纠纷**：围绕 [leveIsio 的 TherapistAI](https://twitter.com/meijer_s/status/1783032528955183532) 产生了争议，辩论重点在于对 AI 冒充人类治疗师替代品的担忧。这引发了关于 AI 能力的负责任表述及其伦理影响的讨论。

**寻找语义搜索 API**：参与者评估了几个**语义搜索 API**；然而，像 [Omnisearch.ai](https://omnisearch.ai/) 这样的选项在网络新闻扫描效果上不如 [newsapi.org](https://newsapi.org) 等传统工具。这表明目前的语义搜索解决方案仍存在差距。

**法国押注 AI 治理**：讨论围绕法国将**大语言模型 (LLMs)** 实验性地整合到公共部门展开，指出了该国的前瞻性立场。讨论还涉及了技术与社会政治格局互动等更广泛的主题。

**穿越可能的 AI 寒冬**：受一条[关于 AI 泡沫破裂后果的推文](https://x.com/schrockn/status/1783174294865887521?s=46&t=90xQ8sGy63D2OtiaoGJuww)启发，成员们辩论了 AI 风险投资的可持续性。对话涉及了经济变化对 AI 研究和创业前景影响的推测。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain AI 开启聊天机器人探索**：讨论集中在如何利用 **pgvector stores** 与 LangChain 结合以增强聊天机器人性能，包括分步指南和特定方法（如 `max_marginal_relevance_search_by_vector`）。成员们还详细阐述了 `SelfQueryRetriever` 背后的机制，并探讨了使用 `createStuffDocumentsChain` 等方法构建对话式 AI 图 (graphs) 的策略。[LangChain GitHub 仓库](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/pgvector.py) 以及官方 [LangChain 文档](https://python.langchain.com/) 被指出是重要的参考资源。

**针对新发布的 LLaMA-3 的模板困扰**：一位成员寻求关于 **LLaMA-3** 的 prompt 模板建议，并指出 [官方文档](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) 中存在空白，这反映了社区在紧跟最新模型发布方面的共同努力。

**分享 AI 叙事与工具**：社区展示了多个项目：使用 LangChain 的 LangGraph 适配的 **RAG 框架**，相关文章已发布在 [Medium](https://medium.com/ai-advances/unifying-rag-frameworks-harnessing-the-power-of-adaptive-routing-corrective-fallback-and-1af2545fbfb3)；一个以工会为中心、基于文本的 RPG 游戏 "Brewed Rebellion"，[在此可玩](https://www.playlab.ai/project/clveh1d860lwsrv1nvl4xfksp)；"Collate"，一个将保存的文章转换为摘要简报的服务，访问地址为 [collate.one](https://collate.one/newsletter)；以及 BlogIQ，一个为博主提供的内容创作助手，[可在 GitHub 上找到](https://github.com/langchain-tech/BlogIQ)。

**训练日：Embedding 模型对决**：希望加深对 embedding 模型了解的 AI 从业者可以参考成员分享的教育性 [YouTube 视频](https://www.youtube.com/watch?v=i2FJVyDIhaI)，旨在揭秘该行业中的最佳工具。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Toolkit 拆解与赞誉**：[Cohere 的 Toolkit 已开源](https://cohere.com/blog/cohere-toolkit)，其添加自定义数据源和部署到多个云平台的能力让用户感到兴奋，同时 [GitHub 仓库](https://github.com/cohere-ai/cohere-toolkit) 因促进了 RAG 应用的快速部署而受到称赞。
  
- **故障排除成为焦点**：一位成员在 Docker for Mac 上运行 Cohere Toolkit 时遇到问题；同时，关于在 Azure 上使用 Cohere API key 的担忧得到了缓解，澄清了该 key 是可选的，从而确保了隐私。

- **API 异常警报**：有报告称，在代码中实现站点连接器 grounding 时，API 与 playground 的结果存在差异，这一挑战即使在随后的修正后也未能完全解决。

- **向开源冠军致敬**：成员们向 Cohere 联合创始人及核心贡献者表达了感谢，感谢他们在推出开源工具包方面所做的努力，并强调了其对社区的潜在益处。

- **对 Cohere 批评者的批评**：一篇据称对 Cohere 持批评态度的文章引发了辩论，该文章聚焦于向 Cohere 的 LLM 引入 jailbreak（越狱），这可能导致恶意 D.A.N-agents 的产生，尽管该文章的反对者无法引用具体细节来支持他们的观点。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **AI 解析的首选模型**：**Wizard 2 8X22b** 和 **gpt 4 turbo** 模型因其在解析系统消息和调用函数（calling functions）方面的出色能力，被公认为 OpenInterpreter 项目中的高性能表现者。然而，关于 **llama 3** 等模型行为异常的报告引起了用户的关注。

- **本地执行的补丁**：用户反馈在 OpenInterpreter 中*本地执行*（local execution）模型时存在困惑，建议的解决方案是使用 `--no-llm_supports_functions` 标志来解决特定的错误。

- **超越基础的 UI 开发**：围绕为 AI 设备开发用户界面的讨论已经展开，工程师们正在探索 *tkinter* 之外的选择，以确保与未来微控制器的集成兼容性。

- **视觉模型成为焦点**：GitHub 仓库和学术论文的分享激发了关于计算机视觉模型的讨论，重点关注架构轻量化的 **moondream**，以及 **llama3** 对各种量化（quantization）设置的适应性，以优化 VRAM 使用。

- **01 Project 获得关注**：成员们一直致力于将 01 project 扩展到外部设备，网上分享的创意实现证明了这一点，包括将其集成到 Grimes 公开的一个蜘蛛机器人项目中。关于 **01** 的安装和执行指南也得到了解决，提供了 Windows 11 的详细说明以及使用命令 `poetry run 01 —local` 运行本地模型的技巧。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**盲测环节迎来 Phi-3-128K**：**Phi-3-128K** 已被引入盲测，通过“**who are you**”等策略性交互引导，以及 **LMSys** 等机制防止模型名称泄露，以维持盲测的公正性。

**指令微调仍是热点**：尽管出现了许多评估 LLM 的基准测试（benchmarks），如 [LMentry](https://arxiv.org/abs/2211.02069)、[M2C](https://arxiv.org/abs/2307.05454) 和 [IFEval](https://arxiv.org/abs/2311.07911)，社区对于指令遵循（instruction-following）评估的持久相关性仍持有强烈观点，这在 Sebastian Ruder 的 [newsletter](https://newsletter.ruder.io/p/instruction-tuning-vol-2) 中有所强调。

**开源动态活跃 AI 圈**：Cohere 聊天界面的开源引起了关注，可以在 [GitHub](https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file) 上找到。这引发了一些幽默的闲聊，包括关于 **Nathan Lambert** 在 AI 领域影响力的笑话，以及对行业参与者不透明动机的思考。

**AI 先驱摒弃企业黑话**：*"pick your brain"*（向你请教）一词在社区内遭到鄙视，强调了行业专家在创新高峰期对被套用企业陈词滥调的不适感。

**SnailBot 通知需谨慎**：SnailBot 的部署引发了关于通知礼仪的讨论，而访问 [“Reward is Enough”](https://www.sciencedirect.com/science/article/pii/S0004370221000862) 出版物时遇到的困难触发了故障排除对话，突显了无障碍获取科学资源的必要性。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mlock 困扰 Llamafile 用户**：工程师报告了使用 `phi2 llamafile` 时出现 **“failed to mlock”** 错误，目前缺乏明确的解决方案或权宜之计。
- **工程师期待 Phi3 Llamafile 更新**：社区被引导使用 Microsoft 的 **GGUF 文件** 来利用 **Phi3 llamafile**，具体指南可在 [Microsoft 的 Hugging Face 仓库](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile)中找到。
- **B64 错误导致图像无法识别**：编码问题浮出水面，用户在 JSON 负载（payloads）中的 base64 图像无法被 llama 模型识别，导致 `multimodal : false` 标志开启，讨论中未提供修复方案。
- **Mixtral Llamafile 文档翻新**：**Mixtral 8x22B Instruct v0.1 llamafile** 的文档已进行修改，可在其 [Hugging Face 仓库](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile)访问。
- **Llamafile 下载遭遇木马误报**：Hugging Face 的下载被 Windows Defender 错误地标记为木马，导致社区建议使用虚拟机（VM）或将其列入白名单，同时也提到了向 Microsoft 报告误报的困难。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**批量处理你的机器人 (Batch Your Bots)**：Discord 用户研究了如何在 **Local Mixtral** 中高效地批量处理 Prompt，并对比了 **vLLM** 和开源的 **TGI** 等工具。虽然一些人更倾向于使用 **TGI** 作为 API 服务器以获得低延迟，但其他人则强调了 **vLLM** 在本地 Python 模式下的高吞吐量和直接使用 **Python** 的优势，并建议使用 [llm-swarm](https://github.com/huggingface/llm-swarm) 等资源进行可扩展的端点管理。

**深入探讨 DiscoLM 的德语表现**：与 DiscoLM 在德语方面的交互引发了关于 Prompt 细微差别的讨论，例如使用 "du" 还是 "Sie"，以及如何实现文本摘要的约束（如字数限制）。成员们还报告了模型输出方面的挑战，并表示有兴趣分享实验模型的量化版本，特别是考虑到 **Phi-3** 等模型在 Ger-RAG-eval 等测试中取得的高基准测试分数。

**纠结的问候语**：用户讨论了 Prompt 语言模型时的礼貌程度，承认在德语中使用正式或非正式形式开头对回答有不同的影响。

**摘要难题**：在尝试将模型生成的文本限制在特定的单词或字符限制内而不出现突兀结尾时，这种挣扎是真实存在的。对话反映了用户对输出进行精细化控制的普遍愿望。

**充满信心地进行分类**：社区中引起热烈讨论的是在模型中实现实时推理分类模式的可能性，以匹配备受赞誉的基准测试性能。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **探索 Datasette 的 Python API**：工程师们一直在探索 **Datasette LLM** 的 [Python API 文档](https://llm.datasette.io/en/stable/python-api.html)，利用它来嵌入文本文件并寻找扩展其用途的方法。

- **使用 Claude 实现摘要自动化**：**Simon Willison** 分享了他结合使用 Claude 和 [LLM CLI 工具](https://llm.datasette.io/) 来总结 Hacker News 讨论的经验，并提供了 [工作流概述](https://til.simonwillison.net/llms/claude-hacker-news-themes)。

- **优化文本嵌入**：分享了通过 **Datasette LLM** 的 Python API 高效处理多个文本嵌入的详细指令，重点介绍了根据 [Embedding API 文档](https://llm.datasette.io/en/stable/embeddings/python-api.html) 提供的 `embed_multi()` 功能。

- **Python 环境中的 CLI 功能**：目前 **Datasette LLM** 在直接将 CLI 转换为 Python 功能以嵌入文件方面存在空白；不过，可以在 [GitHub 仓库](https://github.com/simonw/llm/blob/04915e95f8ab490d6a82900ccba92ad3fe93d2ab/llm/cli.py#L1217-L1360) 中追踪其实现，为工程师将 CLI 功能概念性地迁移到 Python 脚本提供参考。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **向 burnytech 问好**：burnytech 在 **general** 频道发了一个简单的 "Hi!" 打招呼。

- **召集多伦多的 AI 爱好者**：*Ollamas and friends* 小组在多伦多组织了一场 AI 开发者见面会，提供社交机会、食物和闪电演讲。感兴趣的 AI 专业人士和爱好者可以通过 [多伦多 AI 见面会注册链接](https://lu.ma/devs5) 报名参加在 Cohere 空间举办的活动，名额有限。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **推文预告引发猜测**：Jeffery Wang 分享了一条 [推文](https://twitter.com/wangzjeff/status/1783215017586012566)，让社区对其内容感到好奇，但随后没有进一步讨论。

- **推理不一致性引发疑问**：成员们注意到，像 **Octo AI** 和 **Anyscale** 这样的推理提供商对于相同的开源模型有不同的 JSON mode 实现，Octo AI 的方法是基于 Schema 的，而 Anyscale 则效仿了 OpenAI 的方法。

- **思考上下文能力**：社区思考了某个工具利用全上下文（full context）的有效性，注意到其性能优于 GPT，但没有对其技术进步进行详细解释。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **恭喜你，被 Anderson 化了！**：一位公会成员幽默地祝贺了某人，大概是因为某项值得注意的成就或里程碑，将其比作 Laurie Anderson——一位以多媒体项目和实验音乐闻名的前卫艺术家。比喻背后的具体背景或原因并未提供。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **内容窃取者的入侵**：一名 Discord **用户**发布了一个推广 **Onlyfans 泄露和每日青少年内容**的不当链接，这违反了社区准则。该内容已被标记以提醒所有成员，并包含一个 [Discord 邀请链接](https://discord.gg/C42uKDFV)。

---

# 第二部分：按频道划分的详细摘要和链接

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1232616515241771059)** (774 条消息🔥🔥🔥): 

- **微调问题排查**：用户讨论了在使用 LLaMA 模型进行带有特殊 token 和格式的 finetuning 时遇到的问题。通过从 prompt 中移除不必要的 token（`<|start_header_id|>`、`<|end_header_id|>`、`<|eot_id|>`），解决了一个特定问题，使训练第一个 epoch 的 loss 从 5 显著下降到了合理范围。

- **关于 AI 模型大小和效率的讨论**：用户交流了使用不同尺寸的 LLaMA-3 模型（特别是 70B 模型）的心得，并权衡了高效 finetuning 此类大型模型所需的 batch size 和 GPU。对话还涉及了在追求 GPU 效率的 QLoRa 与追求更高准确性的 LoRA 之间进行平衡。

- **GPU 需求咨询**：有人询问了不同 GPU 是否适合 finetuning LLaMA-3 模型。会议明确了至少需要 48GB 显存，但为了对不同模型进行持续的 pretraining，建议租用更高 VRAM 的设备。

- **《全职猎人》动画欣赏**：用户表达了对动画系列 “Hunter x Hunter” 的看法，辩论了“嵌合蚁篇”的质量，并分享了他们最喜欢的时刻和篇章。

- **Meta 员工幽默**：由于某位用户对机器学习模型漫长的训练时间非常熟悉，大家开玩笑说他可能是 Meta 的员工。这引发了关于 Meta 薪水的友好调侃，该用户也幽默地坚称自己并非 Meta 员工。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/remekkinas">Remek Kinas | Grandmaster</a>：计算机专业背景。计算机视觉与深度学习。独立 AI/CV 顾问。 </li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct">Snowflake/snowflake-arctic-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/114hHhIcyTCG2FmiKeNPTrEtg3a24sji_?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://sonner.emilkowal.ski/">Sonner</a>：未找到描述</li><li><a href="https://pytorch.org/blog/pytorch2-3/?utm_content=290726973&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024">PyTorch 2.3 发布博客</a>：我们很高兴地宣布 PyTorch® 2.3（发布说明）发布！PyTorch 2.3 在 torch.compile 中提供了对用户自定义 Triton kernels 的支持，允许用户迁移他们自己的 Triton kerne...</li><li><a href="https://greptile.com/blog/100-devs">我询问了 100 名开发者为什么他们交付速度不够快。这是我的收获 - Greptile</a>：唯一真正理解你代码库的开发者工具。 </li><li><a href="https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training">AI Unplugged 8：Llama3, Phi-3，在家训练 LLMs，包含 DoRA。</a>：洞察胜过信息</li><li><a href="https://huggingface.co/NurtureAI/Meta-Llama-3-8B-Instruct-32k">NurtureAI/Meta-Llama-3-8B-Instruct-32k · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/rookie-numbers-gif-26135237">Rookie Numbers GIF - Rookie Numbers - 发现并分享 GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/jeremyphoward/status/1783203909995225090">来自 Jeremy Howard (@jeremyphoward) 的推文</a>：@UnslothAI 现在请支持 QDoRA！:D</li><li><a href="https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html#composibility-and-limitations">在 torch.compile 中使用用户自定义 Triton Kernels — PyTorch 教程 2.3.0+cu121 文档</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://x.com/jeremyphowar">来自 FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md">llama.cpp/grammars/README.md at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/TETO101/AIRI_INS5">TETO101/AIRI_INS5 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/xCMRaKuON7">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/releases/tag/v2.3.0">发布 PyTorch 2.3：torch.compile 中的用户自定义 Triton Kernels，分布式中的张量并行 · pytorch/pytorch</a>：PyTorch 2.3 发布说明 亮点 向后不兼容的更改 弃用 新功能 改进 Bug 修复 性能 文档 亮点 我们很高兴地宣布发布...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1232620550749425744)** (13 条消息🔥): 

- **追求速度与效率的 Reranker 选择**：一位成员强调 **ms Marco l6 V2** 是他们的首选 reranker，发现它在对 200 个 embedding 进行重排序时比 **BGE-m3** 快 15 倍，且结果非常相似。

- **用于 Reranking 的 PostgreSQL 和 pgvector**：另一段内容解释了 PostgreSQL 结合 **pgvector 扩展**的使用，这意味着在 reranking 任务中无需外部 API。

- **硬件购置助力训练**：一位成员对获得适合微调模型的硬件表示兴奋，这增强了他们在 **RAG 和 prompt engineering** 方面的能力。

- **用于大型 RAG 数据集的微调 Llama**：提到正在使用来自 **Unsloth** 的微调 **llama** 来生成一个包含 180k 行的庞大 RAG **ReAct agent 训练集**。

- **无监督上下文学习（In-Context Learning）讨论**：分享了一个 **YouTube 视频**链接，标题为“不再需要微调：无监督 ICL+”，讨论了 **Large Language Models** 的一种高级上下文学习范式（[观看视频](https://www.youtube.com/watch?v=fyAo-K7uxsk)）。

**提到的链接**: <a href="https://www.youtube.com/watch?v=fyAo-K7uxsk">不再需要微调：无监督 ICL+</a>：AI 的新范式，大语言模型 (LLM) 的无监督上下文学习 (ICL)。针对具有 100 万 token 上下文的新型 LLM 的高级上下文学习...

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1232620610534768690)** (186 条消息🔥🔥): 

- **Unsloth Pro 多 GPU 支持正在酝酿中**: Unsloth Pro 目前正在开发多 GPU 分布式支持，已由 theyruinedelise 确认。具有多 GPU 功能的开源版本预计在 5 月左右发布，而现有的 Unsloth Pro 咨询仍在等待回复。

- **实验性模型的微调建议**: Starsupernova 建议使用更新后的 Colab 笔记本，以修复微调后的生成问题，例如模型输出重复最后一个 token 的情况。文中提到了“诅咒的模型合并 (cursed model merging)”，更新后需要重新训练模型，以及可能使用 GPT3.5 或 GPT4 生成高质量的多选题 (MCQs)。

- **数据集挑战与解决方案**: 关于数据集处理的讨论包括数据集映射期间的键错误 (key errors) 以及花括号的输入错误；解决方案包括将数据集从 Google Drive 加载到 Colab，以及通过 CLI 登录在 Hugging Face 上将数据集设为私有。

- **Kaggle 和本地机器上的 Colab 训练注意事项**: 用户询问了由于 12 小时限制如何在 Kaggle 上从检查点 (checkpoints) 恢复训练，`starsupernova` 确认微调可以从最后一步继续。成员们还提示了微调的适当步骤，例如在一个脚本中使用 `save_pretrained_merged` 和 `save_pretrained_gguf` 函数。

- **推理与 Triton 依赖说明**: Theyruinedelise 澄清了 Triton 是运行 Unsloth 的必要条件，并提到 Unsloth 可能很快会提供推理和部署功能。有一个关于 SFT 训练特有的 Triton 运行时错误的问题，突显了环境设置中潜在的变数。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1DGhWyCyf1BI-_yYaLYgOOkZuGAWiuqNj?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/notebooks/io.ipynb#scrollTo=u22w3BFiOveA">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-bnb-4bit">unsloth/llama-3-70b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>: 监督微调训练器</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md">ollama/docs/modelfile.md at main · ollama/ollama</a>: 快速上手 Llama 3, Mistral, Gemma 以及其他大语言模型。 - ollama/ollama</li><li><a href="https://github.com/ollama/ollama/blob/74d2a9ef9aa6a4ee31f027926f3985c9e1610346/docs/import.md?plain=1#L3">ollama/docs/import.md at 74d2a9ef9aa6a4ee31f027926f3985c9e1610346 · ollama/ollama</a>: 快速上手 Llama 3, Mistral, Gemma 以及其他大语言模型。 - ollama/ollama</li><li><a href="https://www.reddit.com/r/comfyui/comments/1bq22x7/change_clothing_in_1_click_ootdiffusion/">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/TETO101/AIRI_INS5/viewer">TETO101/AIRI_INS5 · Datasets at Hugging Face</a>: 未找到描述</li>
</ul>

</div>

---

**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1233058558992449658)** (3 条消息): 

- **Meta 发布 LlaMA-3 并预告 400B 模型**: Meta 发布了一套名为 **LlaMA-3** 的新模型，其中的 8B 参数模型超越了 LlaMA 系列之前的 7B 模型。在发布的同时，Meta 还预告了即将推出的 400B 模型，其基准测试表现有望与 **GPT-4** 持平；目前访问受限，但可根据请求提供。
- **开源 AI 的增长**: 最近 **LlaMA-3** 和 **Phi-3** 的开源发布令人兴奋，人们认识到两者通过不同的方法实现了相似的目标。详细信息可以在分享的 [Substack 文章](https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training)中找到。
- **社区推广**: 一条消息鼓励在另一个频道 (<#1179035537529643040>) 分享 **LlaMA-3** 的更新，认为社区会发现这些信息的价值。

**Link mentioned**: <a href="https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training">AI Unplugged 8: Llama3, Phi-3, Training LLMs at Home ft DoRA.</a>: 洞察胜过信息

---

**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1232623829646311424)** (75 messages🔥🔥): 

- **PR 修复模型词汇表问题**：讨论了一个 PR (#377)，旨在解决加载调整过词汇表大小的模型时出现的问题。该 PR 旨在修复张量形状不匹配（tensor shape mismatches）问题，详情见 [Unsloth Github PR #377](https://github.com/unslothai/unsloth/pull/377)。如果合并，预计随后会发布相关的训练代码。
- **期待 PR 合并**：有人请求合并上述 PR，贡献者表达了迫切愿望。Unsloth 团队在就 .gitignore 文件对 GitHub 页面外观的影响进行简短讨论后，确认将添加该 PR。
- **模型训练优化建议**：分享了关于通过移除未使用的 token ID 来稀疏化嵌入矩阵（embedding matrix）的构想，以便支持更大 batch 的训练，并可能将嵌入卸载到 CPU。实现可能涉及修改 tokenizer 或使用稀疏嵌入层。
- **量化时的模型大小考虑**：有人建议在模型无法适配 GPU 时实现警告或自动切换到量化版本，这引起了大家的兴趣。
- **动态上下文长度调整**：讨论了在模型评估期间迭代增加可用上下文长度而无需重新初始化的可能性。建议包括使用 laser 剪枝和冻结技术，并提到更新模型和 tokenizer 的配置变量。

**Link mentioned**: <a href="https://github.com/unslothai/unsloth/pull/377">Fix: loading models with resized vocabulary by oKatanaaa · Pull Request #377 · unslothai/unsloth</a>：此 PR 旨在解决 Unsloth 中加载调整过词汇表大小的模型时出现的问题。目前由于张量形状不匹配，加载此类模型会失败。此修复程序正在...

---

**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1232605948103753818)** (298 messages🔥🔥): 

- **对潜在的 Phi-3 预设感到困惑**：一位成员询问 LM Studio 中 **phi-3** 的预设，另一位成员提供了变通方法：使用 Phi 2 预设并添加特定的停止字符串（stop strings）。他们提到使用 *PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed* 和 *Phi-3-mini-128k-instruct.Q8_0.gguf* 获得了满意的结果。

- **量化模型质量查询**：讨论包括 phi-3 mini 模型不同量化级别（Q2, Q3, Q4）的表现差异。一位成员报告 Q4 运行正常，而 Q2 无法生成连贯文本，表明量化对模型质量的潜在影响。

- **寻找适配 GPU 的方案**：用户交流了在各种 GPU 配置下运行 LM Studio 的信息，允许在 Nvidia GTX 3060 等显卡上使用高达 7b + 13b 的模型。一位成员还确认了 **phi-3-mini-128k** GGUF 对显存（VRAM）的高要求。

- **缓解错误退出代码 42**：遇到“(Exit code: 42)”错误的用户被建议升级到 **LM Studio v0.2.21** 以解决该问题。其他建议指出，该错误可能与旧款 GPU 显存不足有关。

- **访问本地服务器和网络**：对话围绕在 LM Studio 中利用本地服务器设置展开，例如使用 NordVPN 的 *Mashnet*，通过将 "localhost:port" 更改为 "serverip:port" 从其他位置远程访问 LM Studio 服务器。用户讨论了启用此类配置的方法，有人建议使用 Cloudflare 作为代理。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>：你可以通过在 localhost 上运行的 API 服务器，使用你在 LM Studio 中加载的 LLMs。</li><li><a href="https://huggingface.co/yam-peleg/Experiment7-7B">yam-peleg/Experiment7-7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf">Phi-3-mini-4k-instruct-q4.gguf · microsoft/Phi-3-mini-4k-instruct-gguf at main</a>：未找到描述</li><li><a href="https://rentry.org/LMSTudioFAQ#how-">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，你可以找到我们在 LMStudio Discord 中收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，你可以找到我们在 LMStudio Discord 中收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio)">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里，你可以找到我们在 LMStudio Discord 中收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui：一个用于大语言模型的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。</a>：一个用于大语言模型的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。- oobabooga/text-generation-webui
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1232612350486384640)** (73 条消息🔥🔥): 

- **LLama-3 族群盛况**：Hugging Face 现在在其 [仓库](https://huggingface.co/DavidAU) 中托管了各种 "TinyLlamas" 集合，包含 1B 到 2B 不同配置的 Mini-MOE 模型。建议使用这些模型的 Q8 版本，并建议用户查看原始模型页面以获取模板、用法和帮助指南。
- **对 CMDR+ 的高度赞赏**：讨论显示用户对 CMDR+ 的满意度很高，用户形容其在高配 Macbook Pros 上的性能表现接近 GPT-4 水平，可能超越了 LLama 3 70B Q8。
- **Phi-3 128k 的加载错误与解决方案**：用户报告了尝试加载 Phi-3 128k 模型时的错误。问题似乎是当前版本的 llama.cpp 不支持该架构，但 [GitHub pull requests](https://github.com/ggerganov/llama.cpp/pull/6857) 和 [issues](https://github.com/ggerganov/llama.cpp/issues/6849) 的信息表明，相关更新正在路上。
- **[对 OpenELM 的好奇与怀疑](https://huggingface.co/apple/OpenELM)**：Apple 的新 OpenELM 模型引起了好奇，但由于其 2048 token 的限制以及在不同硬件配置上的潜在性能表现，人们仍持怀疑态度。用户似乎非常渴望 llama.cpp 提供支持，以便在 LM Studio 中尝试这些模型。
- **LongRoPE 引起关注**：关于 LongRoPE 的讨论引起了兴趣，这是一种将语言模型的上下文窗口大幅扩展至高达 2048k tokens 的方法。这一进展的重要性促使用户分享了 [论文](https://arxiv.org/html/2402.13753v1) 并对它所暗示的超长上下文能力表示惊讶。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai.">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://x.com/lmstudioai/status/1782981959804756236?s=46">来自 LM Studio (@LMStudioAI) 的推文</a>：要使用正确的预设配置 Phi 3，请按照此处的步骤操作：https://x.com/LMStudioAI/status/1782976115159523761 ↘️ 引用 LM Studio (@LMStudioAI) @altryne @SebastienBubeck @emollick @altry...</li><li><a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/html/2402.13753v1">LongRoPE: 将 LLM 上下文窗口扩展至 200 万 Token 以上</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">支持 Apple 的 OpenELM · Issue #6868 · ggerganov/llama.cpp</a>：前提条件 在提交 Issue 之前，请先回答以下问题。我正在运行最新的代码。由于开发非常迅速，目前还没有标记版本。我...</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6857">由 tristandruyen 添加 phi 3 聊天模板 · Pull Request #6857 · ggerganov/llama.cpp</a>：这添加了 phi 3 聊天模板。在我使用 #6851 中挑选的用于量化的提交进行的测试中运行良好。我注意到的唯一问题是它似乎会输出一些额外的 &lt;|end|&...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6849">支持 Phi-3 模型 · Issue #6849 · ggerganov/llama.cpp</a>：Microsoft 最近发布了 3 个变体（mini, small &amp; medium）的 Phi-3 模型。我们能否添加对这个新模型系列的支持。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1232620338597335040)** (9 messages🔥): 

- **报告 GPU Offload 问题**：一位成员指出，默认启用 GPU offload 会导致没有 GPU 或显存（VRAM）较低的用户出现错误。他们建议默认关闭该功能，并提供一个包含详细设置说明的 *First Time User Experience (FTUE)* 章节。

- **需要 GPU 加速的困扰**：尽管存在 GPU offload 问题，另一位成员表示需要 GPU 加速。他们确认关闭 GPU offload 可以让应用程序正常使用。

- **解决 GPU 相关错误**：针对有关错误的提问，建议将关闭 GPU offload 作为可能的解决方案，并链接到标识符为 **<#1111440136287297637>** 的额外资源。

- **部分用户的 2.20 版本出现回归问题**：一位用户报告称，升级到 2.20 版本后无法再使用该应用程序，指出 2.19 版本是最后一个可运行的版本，即使在类似的 PC 配置和操作系统（Linux Debian）下也是如此。

- **高显存（VRAM）无助于模型加载**：一位拥有 16GB 显存的用户对无法在 GPU 上加载模型表示困惑，指出虽然 GPU 使用率达到 100%，但自升级到 2.20 版本以来仍面临问题。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1232689734036361237)** (112 messages🔥🔥): 

- **为 AI 任务选择合适的 CPU 和 GPU**：建议成员选择负担得起的最强 CPU；对于 AI 任务，推荐使用 Nvidia GPU，因为如果成员想运行 Stable Diffusion 等应用，其易用性和兼容性更好。该成员还讨论了对 Nvidia 的反感，原因包括“4090 接口熔毁”和驱动问题。
- **内存（RAM）升级对 LLM 性能的影响**：成员们一致认为，升级到 32GB 内存将有利于本地 LLM 的实验和实现。一位成员分享了自己在配备 AMD Ryzen 7840HS CPU 和 RTX 4060 GPU 的机器上成功运行 LLM 的经历。
- **AI 和游戏装备的能效与性能之争**：关于配置能效的讨论围绕成员的设置展开（如 5800X3D 和 5700XT 配 32GB 内存），提倡使用 Eco Mode（节能模式）和对 Nvidia GPU 进行功耗限制以管理发热。
- **排查模型加载和 GPU Offload 错误**：对于因显存不足而出现“Failed to load model”等错误的用户，建议关闭 GPU offload 或使用更小的缓冲区设置。另一位成员通过设置 `GPU_DEVICE_ORDINAL` 环境变量解决了 LM Studio 的 GPU 使用问题。
- **Mac 运行本地 LLM 的性能**：成员们讨论了 Mac 运行 LLM 的性能，共识是理想情况下 Mac 需要 16GB 或更多内存才能高效运行，并承认 8GB 内存配置下的 M1 芯片可以处理小型模型，但不能同时运行其他应用。
  

---

**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

vic49.: 是的，如果你想知道怎么做，请私信我。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1232609840602550363)** (56 messages🔥🔥): 

- **双 GPU 设置困惑**：同时拥有 AMD 和 NVIDIA 显卡的用户在安装 **ROCm** 版本的 **LM Studio** 时遇到了错误。一种解决方法是移除 NVIDIA 驱动并卸载设备，尽管有时需要物理拆除显卡。

- **技术预览版的初期困难**：一些用户对安装问题表示沮丧，但 **heyitsyorkie** 提醒社区，**LM Studio ROCm** 仍处于技术预览阶段，出现 Bug 是预料之中的。

- **ROCm 兼容性与使用**：用户讨论了哪些 GPU 在 **LM Studio** 中与 ROCm 兼容。**heyitsyorkie** 进行了澄清，指出只有在 **HIPSDK** 下显示勾选的 GPU 才受支持，**nettoneko** 则指出支持是基于架构的。

- **安装成功与错误信息**：某些用户报告在调整驱动程序后**成功安装**，而其他人在尝试加载模型时遇到持续的错误信息。**kneecutter** 提到 **RX 5700 XT** 的配置似乎可以运行 LLM 模型，但后来被确认是在 CPU 上运行，而非 ROCm。

- **社区参与和建议**：在报告故障的同时，社区成员积极分享建议，**propheticus_05547** 提到可能需要 AMD **Adrenaline Edition** 才能支持 ROCm。**andreim** 建议更新驱动以适配特定 GPU 的兼容性，例如为 **rx7600** 更新至 **Adrenalin 24.3.1**。
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1232757393646026772)** (2 messages): 

- **Enigmagi 令人印象深刻的融资轮次**：[Enigmagi](https://perplexity.ai) 宣布成功融资 **6270 万美元**，估值达 10.4 亿美元，投资者阵容包括 Daniel Gross、NVIDIA、Jeff Bezos 等。目前正在计划与 SK 和 Softbank 等移动运营商合作，并即将推出企业版 Pro，以加速增长和分发。

- **Pro 服务在 iOS 上线**：[Pro 服务](https://perplexity.ai/pro)现已面向 **iOS 用户**开放，允许他们提出任何问题并迅速获得答案。这项新功能从今天起正式面向 Pro 订阅用户开放。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1232613606554599485)** (467 messages🔥🔥🔥): 

- **企业版 Pro vs. 普通版 Pro**：用户质疑 **Enterprise Pro** 相比 **Regular Pro** 的优势，讨论集中在性能或搜索质量是否存在差异（*“我深表怀疑。但你可以为了隐私支付双倍的费用！”*）。关于 **Opus 使用限制**的担忧依然存在，用户在争论每天 50 次的使用限制。

- **剖析 Perplexity 的 Opus 使用上限**：社区对 Perplexity Pro 中 Opus 搜索**每天 50 次的限制**表示沮丧。几位成员推测了限制的原因，提到了**对试用期的滥用**以及 Opus 极其消耗资源的特性。
  
- **对模型调整的期待**：用户希望 **Perplexity** 能引入调整 Opus 和 Sonnet 模型 **Temperature** 设置的功能，因为这被认为对创意写作非常重要。
  
- **语音功能与技术愿景**：几位用户讨论了新的**语音功能**，包括更新的 UI 以及在 **Perplexity Pro** 上增加的新语音。其他人则表达了对 **Watch OS 版 Perplexity 应用**和**语音功能 iOS 小组件**的渴望。

- **对客户支持的担忧**：用户报告了 Perplexity 设置页面上的 **Pro Support 按钮存在问题**，一位用户指出即使在不同账户上进行了各种尝试，该按钮也无法工作。还有评论称通过电子邮件联系支持团队时缺乏回应。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.businessinsider.com/microsoft-blocking-perplexity-ai-employee-access-2024-4">Microsoft is blocking employee access to Perplexity AI, one of its largest Azure OpenAI customers</a>: 微软正在阻止员工访问 Perplexity AI，它是其最大的 Azure OpenAI 客户之一。</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: 体验世界上最快的推理 (inference)</li><li><a href="https://huggingface.co/collections/apple/openelm-pretrained-models-6619ac6ca12a10bd0d0df89e">OpenELM Pretrained Models - a apple Collection</a>: 未找到描述</li><li><a href="https://huggingface.of/spaces/multimodalart/stable-cascade">Stable Cascade - a Hugging Face Space by multimodalart</a>: 未找到描述</li><li><a href="https://x.com/raywongy/status/1783039023952335144">Tweet from Ray Wong (@raywongy)</a>: 因为你们非常喜欢我询问 Humane Ai Pin 语音问题的 20 分钟视频，这里有 19 分钟（快 20 分钟了！）、无剪辑的我询问 @rabbit_hmi R1 AI 问题并使用其 co...</li><li><a href="https://m.youtube.com/watch?v=W2pYTRdX5LA&pp=ygUJcmFiYml0IHIx">rabbit r1 Unboxing and Hands-on</a>: 在这里查看新的 rabbit r1：https://www.rabbit.tech/rabbit-r1 感谢 rabbit 合作拍摄此视频。在这些地方关注我以获取更新...</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file">GitHub - cohere-ai/cohere-toolkit: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用程序。 - cohere-ai/cohere-toolkit
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1232772045792870531)** (8 messages🔥): 

- **探索语言障碍**：一个分享链接指向了 [Perplexity AI 关于克服语言翻译挑战的分析](https://www.perplexity.ai/search/traduci-SG4MY85cTp6.22Ffm87a9A)。
- **过去式的快乐**：在 [Once I gladly](https://www.perplexity.ai/search/Once-I-gladly-7OdIRseoTfuMuhtGlY7gCw) 发现了一个有趣的思考时刻，探讨了快乐如何随时间变化。
- **变化的恒常性**：[Shift No More](https://www.perplexity.ai/search/Shift-No-More-oo2vIU0NSuaymGTbDWDELA) 主题带来了关于变化的必然性如何影响我们世界观的见解。
- **调至“机械时代”**：对[名为“Mechanical Age”的歌曲](https://www.perplexity.ai/search/song-Mechanical-Age-x4xECMeHRNS_d2ST4bDopA)的好奇探索，暗示了音乐与技术进步概念的融合。
- **深入系统思考**：[系统思考分析](https://www.perplexity.ai/search/Systems-thinking-analysis-ICKSQkZURauRYOrjWv3x.w#3)被讨论为一种理解各种系统内复杂相互作用的综合方法。
- **寻求简洁摘要**：一个搜索查询指向了对简洁摘要的需求，可能是为了提高学习或决策效率，在 [Perplexity AI](https://www.perplexity.ai/search/Give-me-a-XmnWa.81S6K2mjxda_v4FA?1) 上进行了讨论。
- **维护中的答案寻找**：一个链接指向了 [Perplexity AI 关于使用 Langlier Saturation Index 进行游泳池护理的信息](https://www.perplexity.ai/search/What-is-the-hSnFPTgtQWu2MvGENVpNFg)，尽管其具有复杂性和室外泳池的局限性。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1232651678923231292)** (14 messages🔥):

- **图像上传功能不在路线图中**：一位用户询问了通过 Perplexity API 上传图片的可能性，得到的回复是明确的否定，且未来路线图中也没有相关计划。
- **寻找最佳 AI 编程模型**：在 Perplexity API 缺少 GPT-4 的情况下，一位用户推荐使用 **llama-3-70b instruct** 或 **mixtral-8x22b-instruct** 作为目前最佳的编程模型，并强调了它们不同的上下文长度。
- **Perplexity API 缺乏实时数据**：
  一位将 API 集成到语音助手中的用户报告称，API 提供了正确的事件日期，但事件结果却是过时的。他们还询问了用于对比的文档插入功能，并热切期待更多功能。
- **Perplexity API 不支持 GPT-4**：询问 Perplexity API 是否支持 GPT-4 的用户被引导至文档页面，其中列出了模型详情（包括参数量和上下文长度），并注明 GPT-4 不可用。
- **关于 llama-3-70b-instruct 超参数的澄清**：一位用户正在寻求关于调用 llama-3-70b-instruct API 的最佳超参数建议，并提供了一个详细的 Python 代码片段；另一位用户建议尝试 Groq，因为它提供免费且更快的推理，但未确认所询问的超参数是否合适。

**提到的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述

  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1232959243875913738)** (11 条消息🔥): 

- **澄清 RoPE 以及微调与预训练的区别**：对话澄清了关于 **Rotary Position Embedding (RoPE)** 的论文讨论的是微调而非预训练，这可能导致了对 **llama 3** 等模型泛化能力的误解。

- **关于 RoPE 泛化能力的误解**：一位参与者指出，目前没有证据表明 **RoPE** 自身可以在更长的上下文中进行外推，这表明对其能力可能存在混淆。

- **llama 3 的 RoPE 基数是一致的**：另一个关键点是 **llama 3** 从一开始就是以 500k 的 RoPE 基数进行训练的，且在训练过程中基数没有变化。

- **高 RoPE 基数的目的**：有人提出 **llama 3** 的高 RoPE 基数可能是为了降低衰减因子，这可能有利于处理长上下文的模型。

- **RoPE 缩放与模型遗忘**：对话包含了一个假设场景：即使模型在广泛的初始训练后以更高的 RoPE 基数进行重新训练，也可能因为遗忘之前的学习而无法泛化，强调目前仅证明了预训练 Token 的数量远超外推 Token。

**提到的链接**：<a href="https://arxiv.org/abs/2310.05209">基于 RoPE 的外推缩放定律</a>：基于 Rotary Position Embedding 的大语言模型 (LLMs) 的外推能力是目前备受关注的话题。解决外推问题的主流方法是 ...

  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1232715938906767420)** (17 条消息🔥): 

- **苹果头显策略调整**：据报道，苹果正在将 **Vision Pro 出货量** 削减 50%，并重新评估其头显策略，可能意味着 2025 年不会有新的 Vision Pro 型号。该信息通过 [@SawyerMerritt 的推文](https://x.com/sawyermerritt/status/1782895962131702211?s=46) 和 [9to5mac.com 的文章](https://9to5mac.com/2024/04/23/kuo-vision-pro-shipping-delays/) 分享。

- **LLM 提示词注入游戏**：有人创建了一个旨在教授 LLM 提示词注入知识的游戏，包含基础和高级关卡，玩家尝试提取 GPT-3 或 GPT-4 被指示不准泄露的密钥。感兴趣的参与者可以通过此 [邀请链接](https://discord.gg/FYjptek5zY) 加入 Discord 服务器。 

- **Discord 邀请挑战**：存在 Discord 邀请链接被自动删除的问题。该成员原本打算分享一个教授 LLM 提示词注入知识的游戏邀请。 

- **审核协助**：在邀请链接被自动删除后，一名管理员提议暂停自动删除功能，以便重新发布邀请成员加入专注于 LLM 提示词注入的 Discord 服务器的原始消息。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/FYjptek5zY">加入 LLM-HUB Discord 服务器！</a>：查看 Discord 上的 LLM-HUB 社区 —— 与 54 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://x.com/sawyermerritt/status/1782895962131702211?s=46">来自 Sawyer Merritt (@SawyerMerritt) 的推文</a>：新闻：苹果将 Vision Pro 出货量削减 50%，目前正在“审查并调整”头显策略。“2025 年可能不会有新的 Vision Pro 型号” https://9to5mac.com/2024/04/23/kuo-vision-pro-ship...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1232605907066683432)** (16 messages🔥): 

- **介绍 AutoCompressors**：一篇新的预印本论文讨论了 **AutoCompressors**，这是一种基于 Transformer 模型的设计理念，可将长上下文压缩为紧凑的摘要向量，作为软提示（soft prompts）使用，使其能够处理高达 30,720 个 token 的序列，并提高困惑度（perplexity）。这是[完整预印本](https://arxiv.org/abs/2305.14788)。
  
- **Jeremy Howard 对 Llama 3 的评论**：Jeremy Howard 在社区成员链接的一篇文章中详细阐述了模型微调策略的重大转变，并讨论了 Meta 最新的 **Llama 3** 模型。文章及更多想法见此处：[Answer.AI 帖子](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html)。

- **Llama 3 的上下文突破**：一条推文提到 Llama 3 通过持续预训练和调整 RoPE theta 实现了 96k 的上下文长度，并指出其已作为 LoRA 在 Hugging Face 上可用。在此 [Twitter 线程](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg)中了解他们如何增强上下文处理。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2305.14788">Adapting Language Models to Compress Contexts</a>：基于 Transformer 的语言模型（LMs）是强大且应用广泛的工具，但其用途受限于有限的上下文窗口以及处理长文本的高昂计算成本...</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - 使用 FSDP QDoRA 高效微调 Llama 3</a>：我们正在发布 FSDP QDoRA，这是一种可扩展且内存高效的方法，旨在缩小参数高效微调与全量微调之间的差距。</li><li><a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Wing Lian (caseus) (@winglian) 的推文</a>：我已经为 Llama 3 8B 实现了 96k 的上下文。使用 PoSE，我们对基础模型进行了 3 亿 token 的持续预训练，将上下文长度扩展到 64k。在此基础上，我们增加了 RoPE theta 以进一步...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1232970491552206868)** (1 messages): 

- **公告频道升级**：**公告频道**现在可以被“关注”并集成到其他 Discord 服务器中，以便进行无缝更新和信息共享。
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1232609786839961630)** (181 messages🔥🔥): 

- **Einstein v6.1 Llama3 发布**：[Einstein v6.1 Llama3 8B](https://huggingface.co/Weyaxi/Einstein-v6.1-Llama3-8B) 是 Meta-Llama-3-8B 的微调版本，已发布并包含更多对话数据，训练由 **sablo.ai** 赞助。该模型使用 `8xRTX3090` + `1xRTXA6000` 进行微调，资金由 [sablo.ai](https://sablo.ai) 提供。
  
- **Phi 3 分析与推测**：讨论了 Phi 3 模型的架构特性，暗示可能使用了 SWA 以及其他功能，如上采样（upcasted）RoPE 和融合的 MPL & QKV。人们对 Phi 团队与 Llama-3 团队之间 MMLU 评估结果差异显著的原因感到好奇。

- **Snowflake Arctic 480B 中的稠密 + 混合架构**：讨论集中在 Snowflake 宣布的一个拥有 480B 参数的巨型模型上，该模型具有独特的架构，带有注意力池（attention sinks），可能用于扩展上下文长度。人们对该设计的原理以及选择使用不带注意力的残差 MoE 提出了疑问，并提到了维持 token 嵌入强度和计算效率。

- **LLaMA Pro 后预训练方法论**：人们对 LLaMA Pro 独特的后预训练方法表现出兴趣，该方法旨在提高模型的知识而不产生灾难性遗忘。提到的技术如 QDoRA+FSDP，以及与 141B Mistral 模型的比较，激发了对 Transformer 架构和扩展性考量的研究。

- **GPT-4 与 Rabbit R1 的交流引发质疑**：在有说法称 GPT-3 为 OpenAI 的 Rabbit R1 提供动力后，困惑蔓延开来，一些人认为这是沟通误会或模型的幻觉（hallucination）。这种潜在的误导引发了关于模型真实性以及 AI 系统提供信息可靠性的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/MKBHD/status/1783157842607755642">来自 Marques Brownlee (@MKBHD) 的推文</a>：好吧</li><li><a href="https://arxiv.org/abs/2305.13297">Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design</a>：本文通过利用并行注意力和前馈网络设计（PAF）架构，研究了前馈网络（FFNs）在 Transformer 模型中的关键作用，并将其与...</li><li><a href="https://arxiv.org/abs/2401.02415">LLaMA Pro: Progressive LLaMA with Block Expansion</a>：人类通常在不损害旧技能的情况下习得新技能；然而，对于大语言模型（LLMs）来说，情况正好相反，例如从 LLaMA 到 CodeLLaMA。为此，我们提出了一种新的后预训练（post-pretra...）</li><li><a href="https://x.com/danielhanchen/status/1782853167572832650">来自 Daniel Han (@danielhanchen) 的推文</a>：Phi 3 (3.8B) 发布了！论文说它只是 Llama 架构，但我在将其添加到 @UnslothAI 时发现了一些奇特之处：1. 滑动窗口（Sliding window）为 2047？Mistral v1 为 4096。那么 Phi mini 有 SWA 吗？（一个...</li><li><a href="https://news.ycombinator.com/item?id=37740932">Efficient streaming language models with attention sinks | Hacker News</a>：未找到描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1783050724659675627">来自 Weyaxi (@Weyaxi) 的推文</a>：🦙 介绍 Einstein v6.1，基于新的 Llama3 模型，使用多样化、高质量的数据集进行监督微调（supervised fine-tuned）！💬 与 v5 相比有更多的对话数据。🚀 该模型也是无审查的（uncensored）...</li><li><a href="https://huggingface.co/Weyaxi/Einstein-v6.1-Llama3-8B">Weyaxi/Einstein-v6.1-Llama3-8B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1232620737270124576)** (53 条消息🔥): 

- **使用指令训练模型**：讨论了在模型训练损失中包含指令是否多余。注意到来自 *Axolotl* 的参数 `train_on_inputs` 会影响这一点，并且在微调（fine-tuning）期间绕过指令对大多数任务来说可能是合理的。

- **合成数据策略咨询**：一名成员寻求关于生成多样化合成数据以训练模型的建议。建议包括探索 [GitHub - argilla-io/distilabel](https://github.com/argilla-io/distilabel)，并研究 WizardLM、Airoboros、Phi、Alpaca 和 Orca 等项目。

- **过拟合悖论**：关于训练期间验证集（validation sets）效用的辩论，一些人认为验证损失（validation loss）不一定与现实世界的性能相关，Checkpoint 评估可能更有效。结论似乎倾向于最小化 Epochs 并评估最后一个 Epoch 的性能。

- **LLMs 中的长上下文管理**：关于 LLMs（特别是 Llama3）在管理长上下文方面的能力的交流。参与者提到，仅仅扩展上下文而没有适当的理解是不够的，并提到了 **rope scaling** 等技术作为目前正在使用的方法。

- **量化模型微调困境**：关于提高量化版本模型（*Q4_K_M*）性能的咨询，引出了关于数据量以及 LORA 与 QLORA 方法的建议。提出了一种关于创建更强大的 *clown car MoE* 的推测，结合高质量数据的预训练与 Epoch 后的细化，以匹配更高级别模型的输出。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Wing Lian (caseus) (@winglian) 的推文</a>：我正在将 Llama 3 8B 的上下文提升到 96k。使用 PoSE，我们对基础模型进行了 300M tokens 的持续预训练，将上下文长度扩展到 64k。从那里我们增加了 RoPE theta 以进一步...</li><li><a href="https://github.com/argilla-io/distilabel">GitHub - argilla-io/distilabel: ⚗️ distilabel 是一个为 AI 工程师提供的合成数据和 AI 反馈框架，适用于需要高质量输出、完整数据所有权和整体效率的场景。</a>：⚗️ distilabel 是一个为 AI 工程师提供的合成数据和 AI 反馈框架，适用于需要高质量输出、完整数据所有权和整体效率的场景。 - argilla-io/distilabel</li><li><a href="https://www.youtube.com/watch?v=kuvFoXzTK3E&t=4447s)">Chris Bishop 教授的新深度学习教科书！</a>：Chris Bishop 教授是剑桥 Microsoft Research AI4Science 的技术院士和主任。他也是计算机科学的名誉教授...</li><li><a href="https://distilabel.argilla.io/latest/">入门指南</a>：Distilabel 是一个 AI 反馈 (AIF) 框架，用于为 LLM 构建数据集。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1232871657048379402)** (3 条消息): 

- **Tenor 链接中的语言设置**：分享了一个 Tenor gif ([Why not both?](https://tenor.com/view/why-not-both-por-que-no-los-dos-yey-gif-15089379))，并显示了关于**根据浏览器设置进行语言翻译**的说明，提供了在需要时更改语言的选项。
- **对视觉模型更新的期待**：表达了对 **qnguyen** 正在开发的视觉模型更新的期待。
- **黑客松中的部署苦恼**：讨论了尝试部署 **nanollava** 的情况，由于黑客松提供了 Windows VMs，表达了沮丧和厌恶。

**提到的链接**：<a href="https://tenor.com/view/why-not-both-por-que-no-los-dos-yey-gif-15089379">Why Not Both Por Que No Los Dos GIF - Why Not Both Por Que No Los Dos Yey - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/)** (1 条消息): 

paradox_13: 矿工率 (miner rates) 是多少？
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1232634390593536010)** (75 条消息🔥🔥): 

- **探索注意力机制的 Grounding**：讨论模型如何对其回答进行 Grounding，一个建议是使用 `<scratch_pad>` 标签作为一种工作内存。这个想法似乎与改进模型在提供的文本中引用特定信息的目标一致。
  
- **代码引用和幻觉的挑战**：模型的代码文档训练导致了与代码相关内容的意外行为，包括幻觉出的错误位置。指出保持代码语法完整性至关重要，特殊 tokens 可能会破坏模型对代码语法的理解。

- **使用 Pydantic 模型促进有效引用**：关于使用 Pydantic 模型和验证器确保正确引用的广泛讨论，建议包括确保 JSON 序列化和保留有效的代码块。还强调了在微调时保持系统提示词 (system prompts) 简洁精炼的必要性。

- **潜在的注意力掩码 (Attention Masking) 解决方案**：提出了一种新颖的注意力掩码技术，为输入的每一行文本分配特殊 tokens，允许模型指向这些 tokens 进行引用。然而，人们对这种细粒度注意力控制的潜在风险及其对模型泛化能力的影响表示担忧。

- **数据集合成与外部工具集成**：提到了使用 distilabel 的 workbench 进行数据集合成，展示了函数调用 (function calling) 和 JSON/pydantic 数据生成的方法。对话表明人们正在持续寻找创建稳健训练数据集的有效工具。[关于函数调用数据集的 distilabel-workbench](https://github.com/argilla-io/distilabel-workbench/tree/main/projects/function_calling_dataset) 和 [json_schema_generating_dataset](https://github.com/argilla-io/distilabel-workbench/tree/main/projects/json_schema_generating_dataset) 被作为资源分享。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.pydantic.dev/latest/concepts/json_schema/">JSON Schema - Pydantic</a>: 未找到描述</li><li><a href="https://docs.pydantic.dev/dev/api/functional_validators/">Functional Validators - Pydantic</a>: 未找到描述</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/shapes.py">Abstractions/abstractions/goap/shapes.py at main · furlat/Abstractions</a>: 一个用于抽象现实生活（IRL）的 Pydantic 模型集合。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://github.com/noamgat/lm-format-enforcer">GitHub - noamgat/lm-format-enforcer: Enforce the output format (JSON Schema, Regex etc) of a language model</a>: 强制执行语言模型（JSON Schema, Regex 等）的输出格式 - noamgat/lm-format-enforcer</li><li><a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/function_calling_dataset">distilabel-workbench/projects/function_calling_dataset at main · argilla-io/distilabel-workbench</a>: distilabel 中实验性流水线的工作仓库 - argilla-io/distilabel-workbench</li><li><a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/json_schema_generating_dataset">distilabel-workbench/projects/json_schema_generating_dataset at main · argilla-io/distilabel-workbench</a>: distilabel 中实验性流水线的工作仓库 - argilla-io/distilabel-workbench
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1232623858587271209)** (102 messages🔥🔥): 

- **范畴论与 LLMs**：聊天成员讨论了**范畴论**在语言模型中的应用，分享了如 [Tai-Danae Bradley 的工作](https://www.math3ma.com/about) 等资源，以及使用 Yoneda 引理来理解语义概念。随后进行了关于**类型本体、态射和变换**的详细理论对话。

- **世界模拟传奇**：用户积极分享了使用世界模拟（如 **WorldSim**）的经验和想法，用于包括叙事扩展和构建超级英雄宇宙在内的各种目的。提到了 Janus 帖子的链接，以关联世界模拟中的视觉元素。

- **WorldSim 的 Twitch 直播**：大家对在 Twitch 上直播**共享世界模拟体验**的想法感到兴奋。一位成员甚至开设了一个法语直播的 Twitch 频道，但由于 4chan 用户滥用导致 WorldSim 暂时关闭，计划被打断。

- **唤起性的 AI 导向角色树**：一位成员分享了他们为 "ClaudeCharacters" 开发的角色家族树，在 [一篇 Twitter 帖子](https://twitter.com/Jtronique/status/1783291232795471938/photo/1) 中强调了角色之间的叙事潜力和涌现的互动。

- **Websim 问题与技巧**：简短的互动围绕 **Websim** 的故障排除展开，建议将其用于网页模拟。一位成员分享说，复制并粘贴 Claude 模型的第一个 Prompt 可能会有帮助。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.math3ma.com/about">About</a>: Math3ma 是一个关于数学的博客，由 Tai-Danae Bradley 维护。</li><li><a href="https://cybercat.institute/2024/04/22/open-games-bootcamp-i/">The Build Your Own Open Games Engine Bootcamp — Part I: Lenses</a>: 这是一个多部分系列的第一部分，以简单的方式揭示了开放游戏引擎的底层机制。</li><li><a href="https://a.co/d/4TIOQJu">未找到标题</a>: 未找到描述</li><li><a href="https://tinyurl.com/SuperWorldSim">Super World Sim - HuggingChat</a>: 在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://github.com/furlat/Abstractions/blob/main/llmmorph.md">Abstractions/llmmorph.md at main · furlat/Abstractions</a>: 一个用于抽象现实生活（IRL）的 Pydantic 模型集合。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1232645861239029851)** (11 messages🔥): 

- **解决次优编译器优化问题**：一位成员尝试通过使用标志（`nvcc -O0 -Xcicc -O0 -Xptxas -O0`）禁用编译器优化来提高性能，但未达到预期结果，其指标得分为 **19.45**。

- **手动 Kernel 优化见解**：在分析矩阵乘法 Kernel 时，一位成员提到，在根据算术强度、Flops 和内存传输手动计算 Kernel 性能时，看到了缓存带来的收益。

- **AWS Instance GPU 选择限制**：讨论中提到，有成员在 Modular 博客上读到，用户在设置 AWS 实例时无法选择具体的 GPU 型号，只能选择实例类别。

- **培养 CUDA 专长**：一位成员询问在完成 PMPP 及其大部分练习后，如何进一步学习 CUDA 或在该领域找工作。另一位成员建议通过公开展示优化特定利基操作（niche operation）的技能来证明实力。

- **在 Discord 上协作开发 CUDA Kernel**：鼓励成员在 CUDA MODE Discord 频道中分享并改进现有的 CUDA kernel，甚至暗示即将开展 Triton kernel 的协作。另一位成员建议创建一个 GitHub 仓库，因为 Discord 频道不适合长期存储此类信息。

---

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1232828093010542613)** (1 messages): 

- **PyTorch 2.3 拥抱 Triton**：PyTorch 发布了 2.3 版本，现在支持在 `torch.compile` 中使用**用户定义的 Triton kernel**。如[官方发布博客](https://pytorch.org/blog/pytorch2-3/)所述，此更新允许将 Triton kernel 从 eager 执行模式迁移，且不会损失性能或破坏计算图。

- **为 LLM 精调的张量并行 (Tensor Parallelism)**：新更新通过提供改进的**张量并行 (Tensor Parallelism)**，增强了训练**大语言模型 (LLMs)** 的体验。该功能已在参数量高达 1000 亿的模型上得到验证。

- **半结构化稀疏带来的速度提升**：PyTorch 2.3 引入了半结构化稀疏（semi-structured sparsity）作为 Tensor 子类，在某些情况下，其速度比稠密矩阵乘法提高了 1.6 倍。

- **2.3 版本背后的社区努力**：共有 426 名贡献者参与了此版本，自 PyTorch 2.2 以来共提交了 3393 次 commit。感谢社区的贡献，并鼓励用户报告新版本的任何问题。

**提及的链接**：<a href="https://pytorch.org/blog/pytorch2-3/">PyTorch 2.3 Release Blog</a>：我们很高兴地宣布 PyTorch® 2.3 发布！PyTorch 2.3 支持在 torch.compile 中使用用户定义的 Triton kernel，允许用户迁移自己的 Triton kernel...

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1232651443547279542)** (32 messages🔥): 

- **CUDA 文件扩展名及其用途**：一位用户询问了 CUDA 开发中 `.cuh` 文件的必要性和好处；然而，目前还没有后续讨论提供明确的定论。

- **COLMAP MVS 中的优化探索**：一位 CUDA 新手询问如何加速 [COLMAP MVS](https://github.com/Parskatt/colmap/blob/main/src/colmap/mvs/patch_match_cuda.cu)，并被建议使用 profiling 工具来更好地了解 GPU 利用率。初步将 `THREADS_PER_BLOCK` 从 32 调整为 96 后，性能得到了提升。

- **CUDA Profiling 工具指明性能优化路径**：在优化过程中，另一位用户推荐使用 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) 对 CUDA 应用程序进行深入分析，从而发现仅启动了 14 个 block，这表明 GPU 使用效率低下。

- **探究 Kernel 活动以寻找瓶颈**：在使用 Nsight Compute 进行 profiling 后，一位用户需要澄清如何深入研究 kernel 性能问题，并被引导选择“full”指标配置文件以获取更全面的信息。

- **现代 CUDA Warp 执行细微差别分享**：一位用户强调了自 Volta 架构以来 CUDA 执行方式的更新，指出 warp 中的线程不再需要运行完全相同的指令，并引用了一篇[详细实现 CUDA 矩阵乘法](https://siboehm.com/articles/22/CUDA-MMM)的文章。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction">4. Nsight Compute CLI &mdash; NsightCompute 12.4 documentation</a>: 未找到描述</li><li><a href="https://developer.nvidia.com/nsight-compute">NVIDIA Nsight Compute</a>: 用于 CUDA 和 NVIDIA OptiX 的交互式分析器（profiler）。</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: 在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://docs.nvidia.com/cuda/profiler-users-guide/">Profiler</a>: 未找到描述</li><li><a href="https://github.com/Parskatt/colmap/blob/main/src/colmap/mvs/patch_match_cuda.cu">colmap/src/colmap/mvs/patch_match_cuda.cu at main · Parskatt/colmap</a>: COLMAP - 运动恢复结构（Structure-from-Motion）与多视图立体视觉（Multi-View Stereo） - Parskatt/colmap
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1232984060129968178)** (9 messages🔥): 

- **Tensor Expand 详解**: 使用 `Tensor.expand` 时会修改张量的步长（Strides）。一位成员解释道，如果访问了原始张量范围之外的索引，[Triton kernels](https://triton-lang.org/) 会崩溃，这表明 Triton kernel 内部可能对张量扩张后的步长处理不当。

- **步长导致 Triton 故障**: 讨论强调，访问张量范围外的索引会导致 Triton kernels 崩溃，这可能是因为 kernel 在执行 expand 操作后未能正确处理张量步长。

- **flash-attn 与 CUDA 的不兼容警示**: 据报道，2.5.7 版本的 flash-attn 与 PyTorch 2.3.0 安装的 CUDA 库不兼容，会导致未定义符号错误（undefined symbol errors）和构建时间过长等问题。

- **PyTorch CUDA 操作与内存概览**: 一位成员分享了 [PyTorch CUDA 文档](https://pytorch.org/docs/stable/notes/cuda.html)的链接，详细介绍了 PyTorch 如何管理 CUDA 张量，并强调在操作过程中会遵循已分配张量的设备内存位置，无论当前选择的是哪块 GPU。

**提到的链接**: <a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA semantics &mdash; PyTorch 2.3 documentation</a>: 未找到描述

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1233058707487719536)** (4 messages): 

- **迈向高效 NeRF**: **Plenoxels** 作为 NeRF 的 CUDA 加速版本，因其速度快且精度相当而受到关注，源代码可在 [GitHub](https://github.com/sxyu/svox2/tree/master/svox2/csrc) 上获取。
- **GPU 驱动 SLAM 的愿望清单**: 社区表示希望看到 [Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS) 的 CUDA 实现，该技术目前尚未提供 CUDA 格式。
- **Mobile ALOHA 获得速度提升**: [Mobile ALOHA](https://mobile-aloha.github.io/) 的推理算法（包括 **ACT** 和 **Diffusion Policy**）已在 GitHub 上发布，有望提升移动网络的效率。
- **寻找优化后的 Kernel**: 一位社区成员表示需要一种能够处理二进制矩阵或三元表示（ternary representations）上的 Attention 和通用深度学习操作的 Kernel。
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1232647474586321009)** (6 messages): 

- **Tensor Core 的改进**: 一位成员指出，后期 GPU 代际中的 **Tensor cores** 有了显著改进，粗略估计从 3000 系列到 4000 系列的速度提升了 **两倍**。
- **平衡成本与性能**: **4070 Ti Super** 被推荐为高性价比的选择，在性能和价格之间取得了平衡，虽然比顶级的 4090 慢约 **50%**，但价格也便宜了 **50%**，且仍属于最新一代。
- **4070 Ti Super 的设置复杂度**: 有人提到为了发挥 4070 Ti Super 的全部性能，所需的**设置复杂度**和精力较大，暗示它可能不像其他选项那样开箱即用。
- **双 4070 与单 4090 的对比**: 在纠正了是指 **2x4070 GPU** 而非 2070 后，建议选择 **单块 4090**，理由是两者的性价比相似，且可以避免双 GPU 设置的复杂问题。
- **多 GPU 设置的学习机会**: 另一种观点认为，尽管可能存在问题，但选择 **双 GPU 设置** 可以提供 **多 GPU 编程** 方面的宝贵经验。
  

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1232663980116410368)** (5 messages): 

- **练习位置澄清**：Mr.osophy 明确了查询的位置，指出其位于 *exercise 3.d 正上方*。
- **突发传输大小 (Burst Size) 定义澄清**：在关于与 **内存合并 (memory coalescing)** 相关的 **burst size** 讨论中，mr.osophy 解释说，当线程访问连续内存位置时，系统会将多个加载合并为一个，这是通过硬件层面的 *bursts* 实现的。
- **来自书籍作者幻灯片的见解**：通过书籍作者的 [幻灯片链接](https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture8-S20.pdf) 提供了额外见解，指出 *bursts* 包含约 128 字节，而假设的未合并大小为 8 字节。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1232841823521870014)** (5 messages): 

- **关于提高算术强度 (Arithmetic Intensity) 的澄清**：对话澄清了 **量化 (quantization)** 通过减小字节大小来提高算术强度，而 **稀疏性 (sparsity)** 虽然避免了不必要的计算，但由于写入次数保持一致，可能会显示出较低的算术强度。
- **内存带宽与工作负载效率**：会议指出，向 **GPU** 发送更少的数据可以提高内存带宽效率。相反，发送相同数量的数据但减少工作量会降低算术强度，但工作负载效率仍然会提高。
- **理解稀疏性与量化的讲座引用**：一名成员引用了 [Lecture 8](https://youtu.be/SGhfUhlowB4?feature=shared&t=2181) 中 36:21 的特定时刻，以澄清关于稀疏性在 GPU 操作期间提高算术强度的观点。
- **演示材料共享**：一位参与者提到可以分享 PowerPoint 演示文稿，表示愿意提供讨论的资源或信息。
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1233099632007970978)** (1 messages): 

- **CUDA 内存优化进展**：一名成员分享了 [主 `bucketMul` 函数的简化版本](https://kolinko.github.io/effort/gpu.html)，展示了它如何处理模型 **权重 (weights)** 和 **调度 (dispatch)** 参数的 **乘法**，以高效管理内存加载。它提出了一种类似于 COO 但采用分桶 (buckets) 的方法，同时也考虑了激活内存优化。

**提到的链接**：<a href="https://kolinko.github.io/effort/gpu.html">Effort Engine</a>：未找到描述

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://github.com/adam-maj/tiny-gpu
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1232611663090286634)** (353 messages🔥🔥): 

- **PyTorch 版本混淆**：澄清了安装用于测试的是 PyTorch `2.2.2` 而非 `2.1`，并引用了 [PyPI 上的 PyTorch 包](https://pypi.org/project/torch/)。
- **浮点精度优化挑战**：一名成员表示在优化混合浮点精度类型的 `gelu` 计算时遇到困难，并指出在操作时将 `b16` 转换为 floats 然后再转回 `b16` 后，速度显著提升。
- **CUDA 版本与 Atomics 辩论**：成员们讨论了如何最小化 CUDA kernel 的复杂性，例如通过消除 `atomicAdd` 的使用来简化对多种数据类型的支持。目标是为 `layernorm_backward_kernel` 找到一种避免使用 atomics 且不会显著增加运行时间的实现。
- **GPT-2 训练与 A100 上的多 GPU 扩展**：讨论了一个 [正在进行的 PR (#248)](https://github.com/karpathy/llm.c/pull/248)，涉及使用 NCCL 的多 GPU 训练扩展性，以及在功耗受限 (power throttled) 时不同的性能表现。
- **关于下一代 GPU 与能效的讨论**：对 A100、H100 以及预期的 B100 等 GPU 如何根据其架构、功耗和热动力学进行功耗限制进行了详细的技术讨论。对话深入探讨了输入数据的位模式 (bit patterns) 如何影响功耗和计算效率。此外，还对即将到来的 B100 架构变化及其对功耗限制的影响进行了推测。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pypi.org/project/torch/">torch</a>: Python 中的张量和动态神经网络，具有强大的 GPU 加速功能</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62419/">最新 NVIDIA 技术的应用能源和功率效率 | NVIDIA On-Demand</a>: 随着能源成本的增加和环境影响的扩大，不仅要考虑性能，还要考虑能源消耗，这一点变得越来越重要</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">如何优化 CUDA Matmul 内核以达到类似 cuBLAS 的性能：工作日志</a>: 在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://x.com/chhillee/status/1770210441643577377?s=46&t=yqOem5ktaowo8FyJ-ilbzQ">Horace He (@cHHillee) 的推文</a>: 在 B100/B200/GB200/sparse/fp4 等各种数字满天飞的情况下，要获取新款 Nvidia GPU 的实际规格竟然异常困难。@tri_dao 链接了这个文档，幸好它包含了所有的...</li><li><a href="https://github.com/karpathy/llm.c/pull/252">ngc92 根据精度重新排列权重 · Pull Request #252 · karpathy/llm.c</a>: 通过将相同精度的权重放在在一起来简化我们的逻辑。（如果我们想采用这种方案，我们还需要更新 fp32 网络以匹配；因此，目前这还是一个 Draft PR）</li><li><a href="https://www.youtube.com/watch?v=SGhfUhlowB4">第 8 课：CUDA 性能检查清单</a>: 代码 https://github.com/cuda-mode/lectures/tree/main/lecture8 幻灯片 https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/editS...</li><li><a href="https://github.com/adam-maj/tiny-gpu">GitHub - adam-maj/tiny-gpu: 一个用 Verilog 编写的极简 GPU 设计，旨在从零开始学习 GPU 的工作原理</a>: 一个用 Verilog 编写的极简 GPU 设计，旨在从零开始学习 GPU 的工作原理 - adam-maj/tiny-gpu</li><li><a href="https://github.com/karpathy/llm.c/pull/248">[Multi-GPU] llm.c 现在可以通过 NCCL 在多个 GPU 上运行，作者 PeterZhizhin · Pull Request #248 · karpathy/llm.c</a>: 我在具有 2 个 RTX A2000 的 vast.ai 环境中测试了这一点。这表明代码可以工作，但我的环境不适合进行性能分析（profiling），因为它没有 NVLink。以下是我的结果：在 1 个 GPU 上：./train...</li><li><a href="https://www.youtube.com/watch?v=e24BlWvSLNM">自我改进的 Agent 是未来，让我们构建一个</a>: 如果你对 AI 是认真的，并且想学习如何构建 Agent，请加入我的社区：https://www.skool.com/new-society 在 Twitter 上关注我 - https://x.com/D...</li><li><a href="https://github.com/karpathy/llm.c/pull/233">feat(attention_forward.cu): CuTe(cutlass) 浅显入门，作者 FeSens · Pull Request #233 · karpathy/llm.c</a>: 这是一个非常浅显的关于使用 CuTe (Cutlass v3) 实现 Flash Attention 2 的介绍。之所以说它浅显，是因为它还没完成。到目前为止我完成的工作：在 Query 块、Ba... 之间划分任务。</li><li><a href="https://github.com/karpathy/llm.c/pull/227/files#diff-36ab7119a513de038e8bb2463dc9d5fd7bda2c38b3aabaec599423611ff3a983R1041-R1067">用于完全自定义 attention 的第二个 matmul，作者 ngc92 · Pull Request #227 · karpathy/llm.c</a>: 到目前为止，仅存在于 /dev 文件中，因为对于主脚本，我们还需要修改 backward。出于某种原因，我在这里的基准测试中看到了显著的加速，但在我尝试将其用于...</li><li><a href="https://ppc-exercises.cs.aalto.fi/">课程</a>: 未找到描述</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/aalto2024/llm/llm9a">LLM9a: CPU 优化</a>: 未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1232930705051222091)** (4 条消息): 

- **周末会议的 LinkedIn 活动**: 一位成员提议为即将到来的周末会议创建一个 **LinkedIn 活动**。该建议获得了批准。

- **明天会议的录制计划**: 有人询问谁将负责第二天会议的录制工作。未提供更多细节。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1232672188826521610)** (28 条消息🔥):

- **探索 Counterfactual Research**：一位成员提到 **counterfactual reasoning** 是 AI 领域的热门研究课题，近年来发表了许多高质量论文。
- **提出针对 LLMs 的归一化性能评估**：一位成员建议根据 **perplexity** 或 **log-likelihood** 对 **LLMs** 在 **benchmarks** 上的表现进行归一化处理，这可以抵消 **data contamination** 的影响（即模型在训练期间可能遇到过评估数据的情况）。
- **模型中的 Parallel Attention 和 FFN**：针对某些论文将 **attention** 和 **feed-forward networks (FFN)** 描述为并行操作的问题，一位成员澄清说，某些模型（如 **PaLM**）确实使用了 **parallel attention and FFN**。
- **关于 Sliding Window Attention 的讨论**：成员们讨论了 **sliding window attention** 机制，该机制通过使用 **attention mask** 限制 **transformer** 模型可以关注的历史长度。他们还探讨了将该技术应用于处理极长 **context lengths** 模型时面临的挑战。
- **为 The Pile 数据集提供 Hashes**：一位成员请求 **The Pile** 数据集的 **SHA256 hashes**，并迅速收到了该数据集各部分的 **hashes** 列表，这些信息可以在 **Eleuther AI** 网站上找到。

**Link mentioned**: <a href="https://www.eleuther.ai/hashes">Hashes &mdash; EleutherAI</a>: 未找到描述

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1232614044985196635)** (324 messages🔥🔥): 

- **Facebook 发布重量级推荐系统**：Facebook 最近的论文披露了一个拥有 1.5 万亿参数的 **HSTU-based Generative Recommender** 系统，在大型互联网平台的线上 **A/B tests** 中性能提升了 12.4%。该论文因其潜在的影响力而受到关注，链接见 [这里](https://arxiv.org/abs/2402.17152v2)。

- **Attention 变体可能提升性能**：Facebook 的新模型引入了一种修改，使用 **SiLU (phi)** 和线性函数 **f1/f2** 代替 **softmax**，并结合 **relative attention bias (rab)** —— 改变了 **attention** 机制并用 **gating** 取代了 **feedforward network**。这种设计专门为他们庞大的推荐系统进行了优化。

- **类 Netflix 服务偏好 Batch Recommendations**：大型服务（如 **Netflix**）的普遍做法是按天进行 **batch** 计算推荐，而不是实时计算，这有助于提高利用率和运营效率。**Twitter** 以及可能的 **Facebook** 可能会遵循类似的模式来增强用户体验。

- **对 GPT-like Models 重复版权内容的担忧**：在关于生成式 AI 版权处理的讨论中，出现了一篇提出基于合作博弈论的补偿框架的论文。一些参与者争论使用 **RLHF** 等方法来阻止逐字重复，而另一些人则提到了 **data licensing regime** 可能带来的负面影响，见 [这里](https://arxiv.org/abs/2404.13964)。

- **Tokenizer 技术及其影响的调查**：对话围绕 **BPE-dropout** 以及将 **bytes** 纳入 **token embeddings** 以改进拼写和其他文本级任务的技术展开。参与者对当前的 **large language models** 在训练期间是否使用了这些方法及其可能的下游影响感到好奇。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=Hygxb2CqKm">Stable Recurrent Models</a>: 稳定的循环模型可以通过前馈网络进行近似，并且在基准任务上的经验表现与不稳定模型一样好。</li><li><a href="https://not-just-memorization.github.io/extracting-training-data-from-chatgpt.html">Extracting Training Data from ChatGPT</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.13964">An Economic Solution to Copyright Challenges of Generative AI</a>: 生成式人工智能 (AI) 系统在大规模数据集上进行训练，以生成新的文本、图像、视频和其他媒体。人们越来越担心此类系统可能会侵犯……</li><li><a href="http://arxiv.org/abs/2404.03683">Stream of Search (SoS): Learning to Search in Language</a>: 语言模型在训练过程中很少接触到有益的错误。因此，它们很难看到下一个 Token 之外的内容，遭受错误滚雪球的影响，并难以预测……的后果。</li><li><a href="https://manifestai.com/blogposts/faster-after-all/">Manifest AI - Linear Transformers Are Faster After All</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>: 尽管长上下文语言模型最近取得了进展，但基于 Transformer 的模型如何展现出从……内任意位置检索相关信息的能力仍然难以捉摸。</li><li><a href="https://arxiv.org/abs/2402.17152v2">Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations</a>: 大规模推荐系统的特点是依赖高基数、异构特征，并且需要每天处理数百亿次用户操作。尽管……</li><li><a href="https://arxiv.org/abs/2402.18668">Simple linear attention language models balance the recall-throughput tradeoff</a>: 最近的研究表明，基于注意力的语言模型在召回（即根据上下文中先前看到的 Token 进行生成的能力）方面表现出色。然而，基于注意力的模型的效率……</li><li><a href="https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based">Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers</a>: 未找到描述</li><li><a href="https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb">guidance/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb at main · guidance-ai/guidance</a>: 一种用于控制大型语言模型的引导语言。- guidance-ai/guidance</li><li><a href="https://arxiv.org/abs/2402.04347">The Hedgehog &amp; the Porcupine: Expressive Linear Attentions with Softmax Mimicry</a>: 线性注意力在提高 Transformer 效率方面展现了潜力，将注意力的二次复杂度降低到与序列长度成线性关系。这为 (1) 训练线性……提供了令人兴奋的前景。</li><li><a href="https://github.com/BlinkDL/RWKV-LM?tab=readme-ov-file#idea-bytes-aware-embedding>">GitHub - BlinkDL/RWKV-LM: RWKV 是一种具有 Transformer 级别 LLM 性能的 RNN。它可以像 GPT 一样直接训练（可并行化）。因此，它结合了 RNN 和 Transformer 的优点——出色的性能、快速推理、节省 VRAM、快速训练、“无限”的 ctx_len 以及免费的句子嵌入。</a>: RWKV 是一种具有 Transformer 级别 LLM 性能的 RNN。它可以像 GPT 一样直接训练（可并行化）。因此，它结合了 RNN 和 Transformer 的优点——出色的性能、快速推理……
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1232616931949809674)** (3 条消息): 

- **关于中心化与细尾回归的讨论**：一位成员批评某项回归分析过于细尾，并建议其仅应进行中心化处理。他们强调，从**数学**上讲，真实回归的误差只能确保是中心化的。
- **关于 Chinchilla 采样方法的辩论**：**Chinchilla** 论文置信区间的准确性受到质疑，探讨了作者是否过度采样了小参数 Transformer，以及应如何确定稳定估计的截止点。该成员正在寻求澄清论文中的**置信区间**是否确实被错误地缩小了。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1232652195548368906)** (6 条消息):

- **LayerNorm 与信息删除**：一位用户发现一个有趣的**分析**，表明 **Pre-LayerNorm 使得从残差流（residual stream）中删除信息变得困难**。这可能是导致 norm 随每一层增加的原因。
- **Penzai 的学习曲线**：一位正在尝试 **penzai** 的成员表示，它虽然很有趣，但存在学习曲线，主要是由于无法在命名张量（named tensor）上调用 `jnp.sum()` 等问题。
- **Penzai 命名变通方法**：在讨论 **penzai** 的特性时，一位用户建议使用 **untag+tag** 作为处理命名张量的手段，并指出**辅助函数（helper functions）**可以帮助使用该工具包。
- **Penzai 的实际示例**：为了演示 **penzai** 的功能，一位成员提供了一个使用 **`pz.nx.nmap`** 在张量内进行标签操作的示例。
- **分享神秘推文**：一位用户分享了一条[神秘推文](https://twitter.com/sen_r/status/1783497788120248431)，但未提供有关其相关性或内容的背景或细节。
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1232619924497895424)** (19 messages🔥): 

- **Mistral 模型的意外表现**：一位成员对 **Mistral 7b** 在某些测试中的低分表示惊讶，推测无法将错误答案作为信息使用可能是一个限制。

- **自定义任务的困扰**：一位成员正在处理一个自定义任务，目标是使用 `CrossEntropyLoss` 评估指令微调模型（instruction-finetuned models）。他们报告了极高的困惑度（perplexity）值和溢出问题，并正在寻求正确实现评估的建议，包括是否应在 `doc_to_text` 中包含指令模板。

- **基准测试的超参数匹配**：有人提出了关于 **gsm8k** 的 `num_fewshot` 设置以对齐 Hugging Face 排行榜的问题，另一位成员建议将该数值设为 5。

- **VLLM 升级问题**：有人询问是什么阻止了 **vllm** 升级到最新版本（考虑到最近新增的模型架构）。一位成员澄清说，除非使用张量并行（Tensor Parallel），否则是可以升级的，后来纠正为是指数据并行（Data Parallel）。

- **LM Evaluation Harness 过滤器注册**：社区的一位新人提议提交一个 PR，为 **lm_eval** 的 `FILTER_REGISTRY` 添加 `register_filter` 函数，这受到了社区成员的欢迎。

- **Brier 分数评估问题**：一位成员在评估 ARC 等任务的 Brier 分数时遇到错误，将问题归因于预期选项数量不匹配。一位社区成员建议修改 `brier_score_fn` 以处理单个实例的分数，并承诺很快会在上游进行更改。

**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/api/metrics.py#L124.">lm-evaluation-harness/lm_eval/api/metrics.py at 3196e907fa195b684470a913c7235ed7f08a4383 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1232605958023286836)** (27 messages🔥): 

- **Tokenization 问题进展中**：一位成员指出，由于 Hugging Face 的 Tokenizer 在 0.13 和 0.14 版本之间的变化，出现了 Token 重映射和拆分的问题。预训练数据与当前 Tokenizer 行为之间的不一致可能会导致模型推理期间的 Token 对齐问题。
- **Poetry 包管理的陷阱**：关于尝试将 NeoX 的包管理切换到 Poetry 的讨论强调了诸如二进制管理不佳和不理想的版本提升等问题，导致一位成员因其复杂性决定不实施此更改。
- **预分词 Pythia 数据的潜在陷阱**：对话中对 Pythia 训练数据的预分词（pre-tokenization）表示赞赏，但人们担心预分词是否与模型使用期间的最终输入一致，特别是如果训练数据中混合了不同版本的 Token。
- **Tokenization 困扰与合并机制**：进一步的对话深入探讨了 Tokenizer 行为的复杂性和令人沮丧之处，特别是涉及带空格的添加 Token，以及使用 "merges" 文件处理不可合并 Token 的理论方法。
- **快速与慢速 Tokenizer 的不一致性**：有人指出预处理步骤会导致 Tokenizer 不匹配，而合并过程本身是稳定的。一位成员表示渴望更详细地记录 Tokenizer 问题，并呼吁建立更好的抽象来处理 Tokenizer 更新。
  

---

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1232609699724263425)** (354 messages🔥🔥): 

- **Juggernaut 模型的问题与解决方案**：一位用户对 Juggernaut 模型在生成令人满意的图像提示词（prompts）方面的困难表示沮丧，转而更倾向于 RealVis V4.0 的表现，后者能更快地生成所需的提示词结果。
- **对 Stable Diffusion 3.0 的期待**：在讨论备受期待的 Stable Diffusion 3.0 发布时，用户被引导去使用已经上线的 API。然而，一位用户在意识到 API 服务并非完全免费（仅提供有限的试用额度）后感到失望。
- **寻求图像生成方面的帮助**：社区新手寻求关于使用 Stability AI 生成图像的指导，资深用户建议使用 [Craiyon](https://www.craiyon.com/) 等外部工具进行简便的在线生成，因为 Stable Diffusion 模型通常需要本地软件安装。
- **高级模型使用讨论**：成员们讨论了多个 AI 相关话题，包括生成特定图像提示词的策略、使用 vast.ai 等云算力、处理 AI 视频创作以及模型微调（fine-tuning）中的挑战。此外还分享了关于训练 LoRas 以及生成符合 Steam 监管要求的内容的具体指导。
- **探索新的 AI 工具和项目**：用户发布并讨论了各种独立的 AI 项目，例如生成网络漫画、创作无版权限制的声音设计，以及一个提供免费 AI 图像生成的机器人。部分项目包括位于 [artale.io](https://www.artale.io) 的网络漫画创作 Beta 产品，以及位于 [adorno.ai](https://adorno.ai) 的专业声音设计生成器。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://stableaudio.com/live">Stable Audio - 用于音乐和音效的生成式 AI</a>：无论你是初学者还是专业人士，都可以使用人工智能创作原创音乐和音效。</li><li><a href="https://pixlr.com/remove-background/">Pixlr 免费在线照片 AI 背景移除工具</a>：只需点击一下即可在线移除图像背景。快速且简单！免费试用 Pixlr 照片背景移除工具！</li><li><a href="https://civitai.com/models/153568?modelVersionId=433727">Real Dream - 14 | Stable Diffusion Checkpoint | Civitai</a>：2024年3月12日 Civitai 上目前可用的最逼真的 LCM 1.5 模型。由于我没有非常先进的硬件，如果你能给我提供一些 Buzz...</li><li><a href="https://suno.com/song/fcedaca6-eaad-4b99-b6ac-aa28feb12d6d">桃花诺三生缘 by @jone_coolke2049 | Suno</a>：古典，国风，情长歌曲。使用 Suno 聆听并创作你自己的作品。</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/15c2n0q/sdxl_two_text_encoders_two_text_prompts/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/SeaArtLab/ComfyUI-Long-CLIP">GitHub - SeaArtLab/ComfyUI-Long-CLIP: Long-CLIP 的 ComfyUI 实现</a>：Long-CLIP 的 ComfyUI 实现。通过在 GitHub 上创建账号为 SeaArtLab/ComfyUI-Long-CLIP 的开发做出贡献。</li><li><a href="https://adorno.ai">Adorno AI - AI 音频生成</a>：未找到描述</li><li><a href="https://www.erase.bg/">免费背景图像移除器：在线移除高清图像背景 - Erase.bg</a>：使人物、动物或物体的图像背景透明。免费下载高分辨率图像，适用于电子商务和个人使用。无需信用卡。</li><li><a href="https://www.youtube.com/watch?v=2FUvHdnIaW4">尝试这款免费 AI 视频（通过一个提示词制作 30 秒 AI 电影）</a>：今天就在这里尝试：https://noisee.ai/ 📧加入我的时事通讯 https://delightfuldesign.eo.page/w7tf5---👨‍🏫查看我的 AI 课程：https://www.udemy.com/user...</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/zyBzvxSFSv">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/Snowflake-Labs/snowflake-arctic">GitHub - Snowflake-Labs/snowflake-arctic</a>：通过在 GitHub 上创建账号为 Snowflake-Labs/snowflake-arctic 的开发做出贡献。</li><li><a href="https://arctic.streamlit.app">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1233094280797753376)** (1 messages): 

- **检测到 Mixtral 8x7b 供应商问题**：发现 **Mixtral 8x7b** 的一个主要供应商发送了空白响应。该供应商已被暂时移除，并计划在未来实现一种**自动检测**此类问题的方法。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1232627125681520661)** (323 messages🔥🔥):

- **关于模型在中国联系信息的关注**：一名成员询问了中国业务团队的直接联系方式，寻求建立本地联系。
- **关于模型性能的讨论**：辩论了 **Phi-3** 和 **Wizard LM** 等各种模型的性能和使用，一些成员认为小型模型相较于大型替代方案具有优势。
- **OpenRouter 流式传输异常**：用户报告了无限循环的 "OPENROUTER PROCESSING" 消息，这被澄清为保持连接活跃的标准方法，尽管一名成员在处理 OpenAI 的 GPT-3.5 Turbo 的完成请求时遇到了超时问题。
- **OpenRouter 功能评价褒贬不一**：成员们讨论了 OpenRouter 功能的优缺点，包括函数调用（function calls）的问题，以及在某些模型中服务不遵循 'stop' 参数的情况，尽管也有人推荐了像 Fireworks 的 [FireFunction](https://readme.fireworks.ai/docs/function-calling) 这样的平台。
- **关于多语言模型能力的辩论**：用户对比了 **GPT-4**、**Claude 3 Opus** 和 **L3 70B** 等模型，特别关注它们在非英语提示词下的表现，一名成员指出 GPT-4 的俄语回答听起来不自然。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rocky-muscle-755.notion.site/What-happened-to-WLM2-a247e09244d0483cbb02c1587b357c9d?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct">Snowflake/snowflake-arctic-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://console.groq.com/docs/tool-use#models">GroqCloud</a>：体验世界上最快的推理</li><li><a href="https://openrouter.ai/playground?models=openai/gpt-3.5-turbo">OpenRouter</a>：LLM 和其他 AI 模型的路由</li><li><a href="https://tenor.com/VyEb.gif">Delorean 时间旅行 GIF - 《回到未来》Delorean 时间旅行 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://readme.fireworks.ai/docs/function-calling">使用函数调用模型</a>：未找到描述</li><li><a href="https://openrouter.ai/models?q=free">OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling">FireFunction V1 - Fireworks 的 GPT-4 级函数调用模型 - 比 GPT-4 快 4 倍且权重开源</a>：Fireworks 开源了具有接近 GPT-4 级质量且速度快 4 倍的新函数调用模型</li><li><a href="https://rentry.org/ClaudeVision">Claude 3 "Vision" 使用了 Google 的 Cloud Vision API</a>：# 此页面正在完善中；我有大量数据需要处理。对目前的结论有约 85% 的把握。Anthropic 的 Claude 3 系列为其模型提供了 Vision 能力，使其能够...</li><li><a href="https://openrouter.ai/docs#required-parameters-(beta)">OpenRouter</a>：构建与模型无关的 AI 应用</li><li><a href="https://openrouter.ai/docs#sse-streaming-comments">OpenRouter</a>：构建与模型无关的 AI 应用</li><li><a href="https://openrouter.ai/docs#required-parameters-(b">OpenRouter</a>：构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1232779151895822358)** (1 条消息):

```html
<ul>
  <li><strong>Llama 3 强势登场</strong>：凭借 15T token 的训练量和 1000 万人类标注样本的微调，<strong>Llama 3</strong> 推出了 8B 和 70B 版本，均包含 Instruct 和 Base 模型。其中 70B 变体在 MMLU 基准测试中得分超过 80，成为表现最出色的开源 LLM；其编程能力同样出众，在 Human Eval 上分别获得 62.2（8B）和 81.7（70B）的评分。目前已在 Hugging Chat 上线，提供 <a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3-70B-Instruct">Demo</a> 和 <a href="https://huggingface.co/blog/llama3">博客文章</a>。</li>
  <li><strong>Phi-3 的 MIT 改造</strong>：近期推出的 <strong>Phi-3</strong> Instruct 变体设计了 4k 和 128k 的上下文窗口，并在 3.3T token 上进行了训练，在标准基准测试中表现优于 Mistral 7B 或 Llama 3 8B Instruct。该模型还包含专门的 "function_call" token，并针对包括 Android 和 iPhone 在内的移动平台进行了优化，可通过 <a href="https://huggingface.co/chat/models/microsoft/Phi-3-mini-4k-instruct">Demo</a> 和 <a href="https://x.com/abhi1thakur/status/1782807785807159488">AutoTrain 微调</a> 获取资源。</li>
  <li><strong>开源盛宴</strong>：HuggingFace 发布了 <strong>FineWeb</strong>，这是一个包含 15 万亿 token 的海量网络数据集，供研究使用；同时还发布了 Gradio 和 Sentence Transformers 的最新更新。值得关注的是，大型视觉语言数据集集合 <strong>The Cauldron</strong> 也已推出，旨在辅助指令微调，详情见 <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">FineWeb</a> 和 <a href="https://huggingface.co/posts/tomaarsen/476985886331959">Sentence Transformers v2.7.0</a>。</li>
  <li><strong>HuggingChat 登陆 iOS</strong>：HuggingChat 应用已在 Apple 设备上线，将对话式 AI 的能力带到 iPhone，详情见最新 <a href="https://huggingface.co/posts/fdaudens/628834201033253">公告</a>。</li>
  <li><strong>满足 AI 渴求的内容</strong>：通过博客文章《Jack of All Trades, Master of Some》探索 Transformer Agent 的多功能性；在即将播出的 HuggingCast 中了解如何在 Google Cloud 上部署开源模型；同时，Open Chain of Thought 排行榜为研究人员提供了新的竞技舞台，介绍见 <a href="https://huggingface.co/blog/leaderboard-cot">Leaderboard CoT</a>。</li>
</ul>
```

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1232650644922302464)** (276 条消息🔥🔥):

- **对 Batching 的困惑**：成员们讨论了模型推理中的 **batching**，明确了 Prompt 可以作为数组输入（例如 `prompt = ["prompt1", "prompt2"]`）以进行并行生成。会议提到，与普通的 transformers 库相比，**vllm, Aphrodite, tgi** 等库针对更快的 batching 进行了优化。

- **Mistral 训练的特性引发关注**：有关于 **微调 Mistral 7b** 的咨询，讨论围绕在 OpenHermes 2.5 上对 **mixtral 8x22b** 进行全量 SFT（Supervised Fine-Tuning）的超参数，如 **batch size、学习率和总步数**。

- **AI 夏日中的 Snowflake**：**Snowflake** 发布了两个 **混合 Dense+MoE 模型**：[snowflake-arctic-instruct](https://huggingface.co/Snowflake/snowflake-arctic-instruct) 和 [snowflake-arctic-base](https://huggingface.co/Snowflake/snowflake-arctic-base)，采用独特设计，拥有 **480B 参数**，但每次仅有 **17B 激活参数**。此外还提到了一个 Demo，可在 [HuggingFace Spaces](https://huggingface.co/spaces/Snowflake/snowflake-arctic-st-demo) 上使用。

- **寻找虚拟环境命令**：一位成员在 Windows 上设置 **Python 虚拟环境** 时遇到困难。建议在 Windows 上使用 **`python3 -m venv venv` 随后执行 `venv\Scripts\activate`**，并鼓励尝试 WSL (Windows Subsystem for Linux) 以获得更好的体验。

- **流式传输体验与虚拟环境的回响**：讨论内容包括分享为 transformers 库集成 **自定义流式传输流水线 (streaming pipeline)** 的想法，以及协助解决在 Windows 平台的 VS Code 和 Git Bash 中启动 Python **虚拟环境** 的问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Snowflake/snowflake-arctic-st-demo">Snowflake Arctic St Demo - Snowflake 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/hi-hello-greeting-sabrina-chilling-adventures-of-sabrina-gif-16056963">Hi Hello GIF - Hi Hello 问候 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/mradermacher/Mixtral_AI_Medic-GGUF?not-for-all-audiences=true)">mradermacher/Mixtral_AI_Medic-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/AdaptLLM/medicine-chat">AdaptLLM/medicine-chat · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/BioMistral/BioMistral-7B">BioMistral/BioMistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/legobatman-legojoker-legogoogle-google-joker-gif-13113737">Legobatman Legojoker GIF - Legobatman Legojoker Legogoogle - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/medalpaca">medalpaca (medalpaca)</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/text_to_image">diffusers/examples/text_to_image at main · huggingface/diffusers</a>: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的 SOTA 扩散模型。- huggingface/diffusers</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct">Snowflake/snowflake-arctic-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-base">Snowflake/snowflake-arctic-base · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/medalpaca/medical_meadow_medqa">medalpaca/medical_meadow_medqa · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://rubiks.ai">Rubik's AI - AI 研究助手与搜索引擎</a>: 未找到描述</li><li><a href="https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions">Nvidia 为遵守美国限制，在中国发布速度较慢的 RTX 4090</a>: 美国不允许 Nvidia 在中国销售 RTX 4090。</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china">采用涡轮式散热器的 NVIDIA GeForce RTX 4090 现已在中国发售 - VideoCardz.com</a>: 采用涡轮散热器的 GeForce RTX 4090。不用说，拥有 450W TDP 的 RTX 4090 GPU 通常不会被认为会配备涡轮式散热器。然而，这样的显卡确实存在。我们报道的这款显卡...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1232869775110312037)** (3 条消息): 

- **ZeroShotClassification 局限性揭示**：一位成员发现 `hf.zeroShotClassification` 存在限制，一次仅支持最多 **10 个标签**，这令人感到沮丧。

- **微调中的小插曲**：在尝试微调 **Mistral 7B** 时，一位成员注意到一个异常行为，即多个文件正在被上传，这与他们之前的经验有所不同。
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1232671742280077342)** (9 条消息🔥): 

- **RAG 框架焕新**：一篇文章讨论了检索增强生成 (RAG) 框架的改进，重点介绍了使用 **Langchain** 的 **LangGraph** 实现的 **Adaptive Routing**、**Corrective Fallback** 和 **Self-Correction**。点击[此处](https://medium.com/ai-advances/unifying-rag-frameworks-harnessing-the-power-of-adaptive-routing-corrective-fallback-and-1af2545fbfb3)了解这些框架是如何被统一的。

- **文本游戏的新架构**：介绍了一种名为深度强化相关网络 (DRRN) 的新型架构，旨在辅助自然语言空间内的强化学习，在文本游戏中展现出良好的效果。详细的原始论文可在 [arXiv](https://arxiv.org/abs/1511.04636) 上找到。

- **Twitch 上的法语直播学习**：对于说法语的人，用户 Micode 在 Twitch 上进行了一场直播活动，可能仍然具有相关性和趣味性。你可以访问直播间[此处](https://www.twitch.tv/Micode)。

- **Apple 在 HuggingFace 发布 OpenELM-270M**：Apple 发布了其 **OpenELM-270M** 文本生成模型，现已在 HuggingFace 集合中提供。点击[此处](https://huggingface.co/collections/apple/openelm-pretrained-models-6619ac6ca12a10bd0d0df89e)查看模型。

- **6G 与 AI 强强联手共创未来**：一篇 arXiv 论文讨论了 6G 与 AI 的融合，预测无线通信系统将发生重大变革，以支持无处不在的 AI 服务。完整的摘要和论文可在[此处](https://arxiv.org/abs/1904.11686)查阅。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/Micode">Twitch</a>：未找到描述</li><li><a href="https://www.twitch.tv/Mico">Twitch</a>：未找到描述</li><li><a href="https://huggingface.co/collections/apple/openelm-pretrained-models-6619ac6ca12a10bd0d0df89e">OpenELM Pretrained Models - apple 集合</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1904.11686">The Roadmap to 6G -- AI Empowered Wireless Networks</a>：近期多样化移动应用的激增，特别是那些由人工智能 (AI) 支持的应用，正引发关于无线通信未来演进的热烈讨论。其...</li><li><a href="https://arxiv.org/abs/1511.04636">Deep Reinforcement Learning with a Natural Language Action Space</a>：本文介绍了一种新型的深度神经网络强化学习架构，旨在处理以自然语言为特征的状态和动作空间，如在基于文本的游戏中发现的那样...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1232653656403214386)** (15 条消息🔥): 

- **重复 Space 警报**：Space [bark-with-custom-voice](https://huggingface.co/spaces/clinteroni/bark-with-custom-voice) 被提到是 [suno/bark](https://huggingface.co/spaces/suno/bark) 的重复项。该帖子包含视觉内容，但未提供额外信息。

- **Space 停用通知**：[LipSick Space](https://huggingface.co/spaces/Inferencer/LipSick) 需要 25fps 的 MP4 和 WAV 音频文件，但目前因长时间未活动而处于休眠状态。

- **ProductHunt 产品发布**：一名成员宣布在 ProductHunt 上发布 Wizad，鼓励用户查看并通过点赞支持发布。消息中包含了指向 ProductHunt 各个板块的导航链接，但未提供 Wizad 产品页面的直接链接。

- **新的 Micro-Musicgen 模型发布**：分享了一个用于快速创建 jungle drum 声音的新 micro-musicgen 模型 [micro-musicgen-jungle](https://huggingface.co/pharoAIsanders420/micro-musicgen-jungle)，以及另一个 Gradio 应用 [micro-slot-machine](https://huggingface.co/spaces/thepatch/micro-slot-machine)，该应用可以从 jungle drum 输出中提取和弦。一条推文详细介绍了创作过程，并邀请用户挑战声音设计。

- **将“稍后阅读”转换为 Newsletters**：介绍了一款名为 [Collate](https://collate.one/newsletter) 的应用，它可以将稍后阅读的内容转换为简短的每日 Newsletter。该应用邀请用户使用自己的文章或 PDF 来尝试新构建的个性化 Newsletter 功能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Inferencer/LipSick">LIPSICK - 由 Inferencer 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/Instant-Image">Instant Image - 由 KingNish 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/clinteroni/bark-with-custom-voice">Bark (支持用户提供的语音) - 由 clinteroni 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://collate.one/newsletter">Newsletter</a>: 从您的内容中创建简短的电子邮件摘要</li><li><a href="https://www.producthunt.com/posts/wizad"> Wizad - 使用 GenAI 一键生成社交媒体海报 | Product Hunt</a>: Wizad 是您轻松创建完美契合品牌身份的精美社交媒体海报的首选应用。告别雇佣设计师或花费数小时调整的烦恼...</li><li><a href="https://huggingface.co/pharoAIsanders420/micro-musicgen-jungle">pharoAIsanders420/micro-musicgen-jungle · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/thepatch/micro-slot-machine">Micro Slot Machine - 由 thepatch 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/thepatch_kev/status/1783289389877203198">来自 thecollabagepatch (@thepatch_kev) 的推文</a>: 昨晚 @veryVANYA 提醒我关注 @mcaaroni 的新 micro-musicgen 模型。于是我又做了一个 @Gradio 应用并上传到了 @huggingface，把它变成了一个奇怪的小游戏...</li><li><a href="https://github.com/rrg92/docker-xtts">GitHub - rrg92/docker-xtts: 用于 XTTS Streaming Server 的 Docker 项目</a>: 用于 XTTS Streaming Server 的 Docker 项目 - rrg92/docker-xtts</li><li><a href="https://www.youtube.com/watch?v=A9qPlYVeiOs">Destaques da Comunidade #54</a>: 又一段关于全球开源 AI 社区亮点的视频！帖子：https://iatalk.ing/destaques-da-comunidade-54/ 制作这些视频非常有趣...</li><li><a href="https://iatalk.ing/destaques-da-comunidade-54/">🤗Destaques da Comunidade #54</a>: 大家好，这是 2024 年 4 月 18 日发布的社区亮点 #54。原始内容可以在以下链接查看：以下是带注释的列表和视频！欢迎订阅…
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1232716870453563482)** (3 条消息): 

- **Solid Pods 作为解决方案**: 一位成员建议 **Solid pods** 可能是某个未指明讨论话题的答案。
- **对帮助表示感谢**: 一位成员向另一位成员表示 **感谢**，表明发生了一些有益的互动。
- **寻找 pix2pix 测试方法**: 一位成员正在寻找在 **instruct pix2pix space** 之外测试 *instruct pix2pix edit prompts* 的方法，指出缺乏 `gradio_client` API，并需要任何文本输入图像编辑 ControlNet 作为其演示最后一步的解决方案。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1232732186940149873)** (7 条消息): 

- **并行 Prompt 思考**: 一位成员询问关于从 **Large Language Model (LLM)** 并行生成响应的问题，想知道是否可以同时发出请求而不是顺序请求。

- **精选 Conformal Prediction**: 一位用户分享了 [*awesome-conformal-prediction*](https://github.com/valeman/awesome-conformal-prediction) 的 GitHub 链接，这是一个关于 Conformal Prediction 的精选资源列表，建议将其作为可直接链接的代码实现的有用资产。

- **揭秘 Chat Template 训练**: 有人提出了关于 **SFFTrainer** 内部机制的问题，特别是关于训练期间 LLM 的初始输入以及 Token 生成的限制。该成员寻求有关训练过程的详细资源以更好地理解。

- **寻找开源 STT 前端**: 一位用户正在寻找 **Speech-to-Text (STT)** 技术的开源 Web 前端解决方案，并向社区征求建议。

- **Trustworthy Language Model 发布**: **Trustworthy Language Model (TLM) v1.0** 的发布公告，包括动手实践游乐场 ([TLM Playground](https://tlm.cleanlab.ai/)) 的链接以及包括 [博客](https://cleanlab.ai/blog/trustworthy-language-model/) 和 [教程](https://help.cleanlab.ai/tutorials/tlm/) 在内的支持资源。该模型旨在通过为 LLM 响应提供新的置信度评分来解决可靠性问题。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/safetensors/blob/main/LICENSE">safetensors/LICENSE at main · huggingface/safetensors</a>: 存储和分发 tensor 的简单、安全的方式。通过在 GitHub 上创建账号来为 huggingface/safetensors 的开发做出贡献。</li><li><a href="https://tlm.cleanlab.ai/">TLM Playground</a>: 在浏览器中试用 Cleanlab 的 Trustworthy Label Model (TLM)。</li><li><a href="https://cleanlab.ai/blog/trustworthy-language-model/">使用 Trustworthy Language Model 克服幻觉</a>: 发布 Cleanlab 的 Trustworthy Language Model。TLM 通过为每个 LLM 输出添加信任分数，克服了幻觉这一实现 GenAI 产品化的最大障碍。</li><li><a href="https://help.cleanlab.ai/tutorials/tlm/">Trustworthy Language Model (TLM)</a>: 一个更可靠的 LLM，它为每个输出量化可信度，并能检测不良响应。</li><li><a href="https://github.com/valeman/awesome-conformal-prediction">GitHub - valeman/awesome-conformal-prediction: 一个专业策划的 Conformal Prediction 视频、教程、书籍、论文、博士和硕士论文、文章以及开源库的精选列表。</a>: 一个专业策划的 Conformal Prediction 视频、教程、书籍、论文、博士和硕士论文、文章以及开源库的精选列表。 - valeman/awesome-conformal-prediction
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1232632714541400144)** (4 messages): 

- **寻求使用 TTS 模型的指导**: 一位成员询问如何最好地将微调后的 Text to Speech (.bin) 模型与 diffusers 结合使用，思考是否需要创建自定义模型或是否存在其他方法。
- **使用 fooocus 进行参数微调**: 为了进行精确的参数微调，一位成员建议尝试 fooocus，特别是针对 lcm ip adapter 任务。
- **排查 Prompt+Model 问题**: 一场讨论指出，prompt 和模型的组合可能是目前面临的一些问题的根本原因。
- **LCM 与 IP-Adapter 协作获赞**: 强调了 ip-adapter 和 lcm-lore 之间的有效协作，一位成员对这些工具表示赞赏，同时也对 hyper-sd 的进展表示关注。
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1232706793721823272)** (7 messages): 

- **Language Agent Tree Search 创新**: 向能够进行全面 **tree search planning**（树搜索规划）而非顺序状态规划的 LLM 转变，将显著增强 agentic systems。这一突破的细节及其影响已在 [LlamaIndex's Twitter](https://t.co/cdpvmVNHFj) 上分享。
- **实时知识图谱可视化**: @clusteredbytes 展示了一个 **动态知识图谱绘图工具**，该工具利用 @Vercel AI SDK 将更新流式传输到前端。在分享的 [Twitter 帖子](https://t.co/Mct2EJpGmG) 中探索这项引人入胜的可视化技术。
- **LlamaIndex KR 社区启动**: **LlamaIndex 韩国社区** (LlamaIndex KR) 已启动，旨在探索和分享 LlamaIndex 的功能和可扩展性。[公告推文](https://t.co/bSS0DzWrN5) 中重点介绍了韩语材料、用例和协作项目机会。
- **增强 LLM Chatbots 的 UX**: 为 chatbot/Agent 引入了改进的 **用户体验**，带有可展开的源信息 UI 元素，现在可以通过 `create-llama` 实现。正如 [LlamaIndex 的推文](https://t.co/7FctwxzBIE) 中提到的，代码和概念归功于 @MarcusSchiesser 的出色工作。
- **使用 Qdrant 构建 RAG 应用的教程**: 该教程演示了如何使用 **LlamaParse**、@JinaAI_ embeddings 和 @MistralAI 的 Mixtral 8x7b 构建顶级的 RAG 应用程序。该指南在 [LlamaIndex's Twitter 页面](https://t.co/rXAV3mKVkJ) 上详细介绍了关于解析、编码和存储 embeddings 的见解。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1232639210926768172)** (188 messages🔥🔥): 

- **理解 RAG 功能**: RAG 在处理直接查询时似乎表现最佳，但在处理反向结构的问题时会遇到困难，这促使人们建议探索更高级的 RAG pipeline，如 sentence-window retrieval 或 auto-merging retrieval。一段 [教学视频](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval) 可能会为构建这些复杂的 RAG pipeline 提供更深入的见解。

- **配置多个索引变更点**：一位用户在为从文档创建的 VectorStoreIndex 选择 embedding 和 LLM 模型时感到困惑，回复中澄清了默认情况下，LLM 使用 **gpt-3.5-turbo**，embedding 使用 **text-embedding-ada-002**，除非在全局 `Settings` 或直接在 `query engine` 中指定。

- **在 LlamaIndex 中实现 Pydantic**：在将 Pydantic 集成到 LlamaIndex 时，用户表示在让 Pydantic 正确结构化输出方面存在困难。讨论揭示了与配置 LlamaIndex pipeline 相关的复杂性和错误消息，并提到了正在使用的 OpenAI API `chat.completions.create` 方法。

- **Pydantic 导入和类型检查问题**：由于 LlamaIndex 对 Pydantic 采用了动态的 try/except 导入策略，导致 Pyright 的类型检查出现问题，可能需要使用 `# type:ignore` 注释。有人提出了寻找更好解决方案的疑问，但尚未提供明确的解决方案。

- **寻求 QueryEngine 配置详情**：一位用户询问是否需要更清晰的文档或说明来设置高级 RAG pipeline。另一位用户询问在哪里可以指定使用 GPT-4 而不是默认的 LLM，提供的解决方案是更改全局设置或将其直接传递给 `query engine`。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1fOFR6O5U3h-lfurjnnk2x8KxWIohCTqY?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/querying/querying/">Querying - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/bridge/pydantic.py">llama_index/llama-index-core/llama_index/core/bridge/pydantic.py at main · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/63a0d4fac912e5262d79ffc7a1c22225d2ec8407/llama-index-core/llama_index/core/chat_engine/condense_question.py#L81">llama_index/llama-index-core/llama_index/core/chat_engine/condense_question.py at 63a0d4fac912e5262d79ffc7a1c22225d2ec8407 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/63a0d4fac912e5262d79ffc7a1c22225d2ec8407/llama-index-core/llama_index/core/chat_engine/condense_question.py#L96">llama_index/llama-index-core/llama_index/core/chat_engine/condense_question.py at 63a0d4fac912e5262d79ffc7a1c22225d2ec8407 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_parse/issues/38.">Issues · run-llama/llama_parse</a>: 为实现最优 RAG 解析文件。通过在 GitHub 上创建账户为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://weaviate.io/blog/hybrid-search-fusion-algorithms">Unlocking the Power of Hybrid Search - A Deep Dive into Weaviate&#x27;s Fusion Algorithms | Weaviate - Vector Database</a>: 混合搜索的工作原理，以及 Weaviate 融合算法的底层机制。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/relative_score_dist_fusion/?h=fusion">Relative Score Fusion and Distribution-Based Score Fusion - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/63a0d4fac912e5262d79ffc7a1c22225d2ec8407/llama-index-core/llama_index/core/indices/base.py#L451">llama_index/llama-index-core/llama_index/core/indices/base.py at 63a0d4fac912e5262d79ffc7a1c22225d2ec8407 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/microsoft/monitors4codegen">GitHub - microsoft/monitors4codegen: Code and Data artifact for NeurIPS 2023 paper - &quot;Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context&quot;. `multispy` is a lsp client library in Python intended to be used to build applications around language servers.</a>: NeurIPS 2023 论文 "Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context" 的代码和数据产物。`multispy` 是一个 Python 编写的 LSP 客户端库，旨在用于围绕语言服务器构建应用程序。</li><li><a href="https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval">DLAI - Building and Evaluating Advanced RAG</a>: 简介 · 高级 RAG 流水线 · RAG 指标三元组 · 句子窗口检索 · 自动合并检索 · 结论</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced">Customizing LLMs - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/?h=chat+store">Chat Stores - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/explodinggradients/ragas/issues/557)">Issues · explodinggradients/ragas</a>: 用于检索增强生成 (RAG) 流水线的评估框架 - Issues · explodinggradients/ragas</li><li><a href="https://en.wikipedia.org/wiki/Fighter_(2024_film)">Fighter (2024 film) - Wikipedia</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/">Chat Engine - Context Mode - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#custom-embedding-model">Embeddings - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic#llama_index.core.output_parsers.PydanticOutputParser>).">Pydantic - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent#react-agent-a-simple-intro-with-calculator-tools>)">ReAct Agent - 带有计算器工具的简单介绍 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.l">

lamaindex.ai/en/latest/community/integrations/guidance#creating-a-guidance-program-to-generate-pydantic-objects>)">Guidance - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-fuyu-8b-for-pydantic-strucured-output>)">Multi-Modal GPT4V Pydantic Program - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-minigpt-4-for-pydantic-strucured-output>)">Multi-Modal GPT4V Pydantic Program - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1232606605724487680)** (128 messages🔥🔥): 

- **讨论 AI 的理解能力**：一场关于模型是否能“真正理解”的深度对话展开。会议指出，逻辑中语法与语义的独特融合可能使模型能够对意义进行操作，从而可能实现真正的理解。此外，还强调了像 Transformers 这样的自回归模型的图灵完备性，认为它们拥有足够的计算能力来执行任何程序。

- **语言与社会中的 AI**：在探讨语言与 AI 的关系时，有人认为针对 AI 的**语言演化**可能会在未来创造出新的概念，以实现更清晰的交流。此外，还讨论了语言的**有损特性 (lossy nature)**，考虑到它在表达和翻译完整思想时所带来的局限性。

- **地平线上：开源 AI 模型**：关于 Apple 的 **OpenELM**（一个高效的开源语言模型家族）及其对开源开发大趋势的影响，讨论异常热烈。人们开始猜测这是否意味着 Apple 对 AI 的专有立场发生了转变，以及其他公司是否会效仿。

- **AI 辅助通信进入讨论**：AI 与通信技术的融合是一个备受关注的话题，涉及语音转文本软件和家庭语音助手的自定义唤醒词等技术。会议强调了在 AI 交互中进行有效通信流控制的重要性，例如在与虚拟助手对话时的中断和恢复机制。

- **探索 AI 增强的文本 RPG**：一位成员分享了他们的作品 *Brewed Rebellion*——这是一款在 Playlab.AI 上运行的 AI 驱动文本 RPG，玩家在其中处理组建工会的职场政治，同时避免被高层管理人员发现。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: 未找到描述</li><li><a href="https://www.playlab.ai/project/clveh1d860lwsrv1nvl4xfksp">Brewed Rebellion</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1232672105330511913)** (18 messages🔥): 

- **为 Apple Playgrounds 创建 GPT 专家**：一位成员有兴趣开发一个精通 Apple Playgrounds 应用的 GPT，并询问如何向模型喂数据，包括 **Swift 5.9** 文档，以及是否应将 Playgrounds 的 PDF 包含在知识库部分（尽管无法从 Apple Books 应用下载）。
  
- **驯服 Custom GPTs 以遵循规则**：一位用户表示 Custom GPTs 难以遵守指令，想知道如何确保合规性。另一位用户建议分享该 GPT 的指令，以帮助诊断问题。

- **关于 GPT 更新与竞争的查询**：一位成员询问了 GPT 的下一次更新，希望能超越 **Claude 3**（提到其具有更出色的响应和理解能力），并幽默地询问了备受期待的 **GPT-5** 发布时间。

- **关于网页浏览能力的辩论**：用户讨论了 GPT 浏览器与 **Perplexity AI Pro** 及 **You Pro** 相比的效能，指出 GPT 有时使用的来源较少，但质量似乎相当或更好，并质疑是否有必要推出专门的“网页浏览器”版本。

- **使用 LLMs 分析大型文本文档**：对话转向使用语言模型分析超大型文档。一位文本分析方面的资深用户提到正在使用 **Claude 3 Haiku** 和 **Gemini 1.5 Pro**，并对 OpenAI 通过 API 提供的 128K 上下文窗口选项表示肯定，期待未来出现上下文窗口更大的模型。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1232782623840010290)** (14 messages🔥):

- **命令 GPT “冷静下来”**：用户讨论了如何让 GPT 的语言更加随性，并避免产生令人尴尬的输出。建议包括专注于正面指令（“做这个”）而非负面指令（“不要做那个”），并利用提供的正面示例来引导模型的语言风格。

- **提示词优于负面指令**：一位成员强调负面指令是无效的，应该予以移除，并建议在提示词中提供正面示例，以看到输出中期望的变化。

- **开发者间的资源共享**：提供了一个 [OpenAI Chat](https://chat.openai.com/share/a487e421-ccd1-4299-a478-de5a14ffbe7b) 的共享资源链接，对于面临类似提示词工程（Prompt-Engineering）挑战的人可能会有所帮助。

- **为游戏构建编程助手**：一位参与者寻求帮助，希望创建一个用于为游戏《武装突袭 3》（Arma 3）编写 SQF 语言代码的 GPT，并表示在编写提示词以引用特定上传文档时遇到困难。

- **GPT 编程辅助的冰山一角**：针对构建编程 GPT 的求助，一位成员描述了在使用 GPT 进行特定语言编程任务时的挑战，包括有限的上下文空间和 AI 产生代码幻觉（Hallucinate）的倾向，建议可能需要不同的模型或工具链（Toolchain）。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1232782623840010290)** (14 messages🔥): 

- **应对 GPT 过度创意的语言**：一位用户担心尽管给出了直接的指令，GPT 仍会使用“令人尴尬”且过度创意的语言。建议使用**正面示例**而非负面指令，因为 GPT 在推理（Inference）过程中往往会忽略负面指令。

- **负面提示词的挑战**：另一位成员呼应了关于提示词构建的建议，确认在指导 GPT 时，说明**“做这个”**比**“不要做这个”**更有效，因为 GPT 可能无法区分预期的负面指令。

- **社区成员的代码相关查询**：有人发布了关于在创建 GPT 以辅助《武装突袭 3》（Arma 3）的 SQF 语言编程时遇到的困难，并寻求如何有效利用上传文档的指导。

- **自定义代码训练的复杂性**：一位专家解释了使用自定义代码训练 GPT 的困难，原因在于其对其他语言的大量预训练（Pretraining）以及管理工具链（Toolchain）和上下文的挑战。他们建议针对此类项目考虑使用不同的模型或工具链。

- **使用 Claude 简化机器人语言**：在一个轻松的回复中，一位社区成员建议使用 **Claude** 作为简化 GPT 语言风格的可能解决方案。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1232660477985820702)** (126 messages🔥🔥): 

- **关于 AI 模型能力与对比的讨论**：参与者讨论了 Llama 3、DeepFloyd (DF) 和 Snowflake's Arctic 等模型的性能，并在规模、召回能力和放大图像质量方面与其他模型进行了对比。有人提到 Llama 3 8B 与 GPT-4 相当，而另一人指出他们的脚本有可能通过大规模评分提交来影响模型的训练。
  
- **云服务中匿名性与隐私的威胁**：提到了拟议中的美国云服务用户监管规定，包括可能终结匿名云端使用的“了解你的客户”（KYC）规则。用户对隐私的广泛影响表示担忧，并将其与过去的加密战争（Encryption Battles）相提并论。

- **对 AI 图像模型排行榜的质疑**：一位用户对 AI 图像模型排行榜及其声称的性能真实性表示怀疑，认为可能存在配置问题或其他影响因素。他们注意到对抗性活动（Adversarial Activities）可能会操纵结果，并讨论了可能绕过收集用户评分数据系统的方法。

- **关于审美与提示词对齐重要性的辩论**：对话涉及了用户在生成式 AI 输出中更看重什么——是图像的审美还是与提示词的对齐（Prompt Alignment）。一些人更喜欢包含提示词中确切元素的图像，即使美感稍欠；而另一些人则青睐视觉上悦目的结果。
  
- **讨论 TorrentFreak 作为新闻源的合法性**：在 TorrentFreak 发表了一篇关于涉及云服务提供商的拟议监管规定的文章后，出现了一段关于其信誉的简短对话。分享了一个链接以验证该来源的信誉，澄清虽然它涵盖了 BitTorrent 和文件共享等主题，但它在政治报道方面有着平衡的记录。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://torrentfreak.com/u-s-know-your-customer-proposal-will-put-an-end-to-anonymous-cloud-users-240425/">美国“了解你的客户”（Know Your Customer）提案将终结匿名云用户 * TorrentFreak</a>: 未找到描述</li><li><a href="https://www.tomshardware.com/tech-industry/us-investigates-chinas-access-to-risc-v-open-source-instruction-set-may-become-new-site-of-us-china-chip-war">美国调查中国获取 RISC-V 的情况——开源指令集标准可能成为美中芯片战争的新战场</a>: RISC-V 对美国立法者来说似乎存在风险</li><li><a href="https://videogigagan.github.io/">VideoGigaGAN</a>: 未找到描述</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct">Snowflake/snowflake-arctic-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious">Federal Register :: 请求访问</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/fal-ai/imgsys-results">fal-ai/imgsys-results · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://mediabiasfactcheck.com/torrentfreak-bias/,">TorrentFreak - 偏见与可信度</a>: 偏见极低。这些来源偏见极小，且极少使用带有倾向性的词汇（即试图通过诉诸情感来影响受众的措辞）</li><li><a href="https://en.wikipedia.org/wiki/TorrentFreak">TorrentFreak - 维基百科</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1232677304287629373)** (5 条消息): 

- **探索稀疏混合专家模型（Sparse Mixtures of Experts）**: 一篇[新论文](https://arxiv.org/abs/2404.15045)介绍了 **Multi-Head Mixture-of-Experts (MH-MoE)**，通过将 token 拆分为子 token 并由不同专家并行处理，解决了 SMoE 中专家激活率低的问题。该方法旨在增强上下文理解并减少过拟合，且不会显著增加训练和推理成本。

- **加速视觉模型预训练**: 关于[弱监督预训练](https://arxiv.org/abs/2404.15653)的研究通过将图文数据预训练视为分类任务，比传统的对比学习速度提升了 2.7 倍。它避开了成对相似度计算的高计算强度，同时保留了适用于各种视觉任务的高质量表示。

- **效率中的幽默**: 一位成员评论这种[新型预训练方法](https://arxiv.org/abs/2404.15653)“挺有意思的”，可能是因为与 CLIP 等更复杂的模型相比，它的效率和简洁性令人惊讶，同时还能产生极具竞争力的结果。

- **回归基础：BCE 与多标签分类**: 讨论的预训练方法似乎可以归结为通过 alt-text 识别概念，然后使用多标签分类器。这一策略与更复杂的系统形成对比，但实现了快速且有效的训练。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.15045">Multi-Head Mixture-of-Experts</a>: 稀疏混合专家模型（SMoE）在不显著增加训练和推理成本的情况下扩展了模型容量，但存在以下两个问题：(1) 专家激活率低，仅有少量...</li><li><a href="https://arxiv.org/abs/2404.15653">CatLIP: 在网络级图文数据上以 2.7 倍速预训练达到 CLIP 级的视觉识别精度</a>: 对比学习已成为通过对齐图像和文本嵌入来学习有效视觉表示的一种变革性方法。然而，成对相似度计算...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1232625621134217258)** (85 条消息 🔥🔥): 

- **ChatML 中的换行符格式问题**: 成员们注意到 **ChatML** 以及可能的 **FastChat** 格式存在错误添加换行符和空格的问题，特别是在 end-of-stream token 附近，这可能会影响训练。讨论涉及 token 配置以及 Llama 3 等模型是否也存在此问题。

- **持续预训练与学习率讨论**: 关于 **Llama 3** 预训练学习率（LR）的疑问浮出水面，有人猜测考虑到计划中的**模型变体**，是否使用了无限学习率调度。另一方面，一位成员不小心设置了较高的 LR，并在训练结束后才注意到其影响。

- **模型发布与技术讨论**：进行了多项与模型相关的公告和讨论，例如在聊天中发布成员的模型以获取反馈、小模型在通用任务中的问题，以及讨论具有自定义 **RoPE theta** 的 **32k Llama** 的潜力。

- **Snowflake 发布 408B Dense + Hybrid MoE**：重点介绍了 Snowflake 的 **408B Dense + Hybrid MoE 模型** 的发布，该模型拥有 4K 上下文窗口并采用 **Apache 2.0 许可证**，引发了关于其能力以及与 **Deepspeed** 集成的讨论。

- **市场反应与 Meta 第一季度财报**：关于**市场**对 Llama 3 反应平平的讨论与 Meta 的第一季度财报同时发生，指出首席执行官 Mark Zuckerberg 的 AI 言论恰逢股价下跌。这引发了一个幽默的插曲：Meta 增加的支出是否都花在了用于更大 AI 模型的 GPU 上。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.cnbc.com/2024/04/24/meta-meta-q1-2024-earnings-.html">Meta 股价因疲软的营收指引暴跌 16%，尽管第一季度业绩超出预期</a>：Meta 在周三的财报中营收和利润均超出预期，但由于指引令人失望，股价大幅下跌。</li><li><a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct">Snowflake/snowflake-arctic-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TRI-ML/mamba-7b-rw">TRI-ML/mamba-7b-rw · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/winglian/status/1783175819877122246">Wing Lian (caseus) (@winglian) 的推文</a>：很高兴看到这个医疗模型的发布。Maxime 一直在 Axolotl Discord 中透露相关线索。“由 10 名医生手动评估，并在盲测中与 GPT-4 进行对比...”</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cbzh65/snowflake_dropped_a_408b_dense_hybrid_moe/">Reddit - 深入探讨</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1232735121052930172)** (8 条消息🔥): 

- **Llama-3 扩展**：一位成员提到 **Llama-3** 的规模到下午可能会达到 **128k**。
- **Llama-3 的微调与训练**：澄清了除了使用 RoPE 进行 **Tuning** 之外，还使用 **PoSE** 进行了一些 **Training**。
- **Qdora 与 FSDP 的难题**：讨论显示，通过启用 `peft_use_dora: true`，**qdora** 可以在单 GPU 上无需 **fsdp** 运行。然而，对于多 GPU 设置，**fsdp** 本身是必需的。
- **FSDP 与反量化问题**：指出 **peft 的 dora 实现** 与 **fsdp** 存在冲突，原因是其处理反量化（dequantization）的方式。
- **全量微调中的 FSDP 问题**：成员们在让 **fsdp 与全量微调（full fine tunes）配合工作** 时遇到困难，指出这存在问题，底层系统的变化可能是一个因素。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1232670427206385674)** (4 条消息): 

- **鼓励实验**：一位成员对收到的建议表示感谢，表示将尝试亲自应用。
- **互祝好运**：另一位成员对计划中的尝试表示祝贺，希望获得成功的结果。
- **微调 Phi3 模型**：一位成员讨论了在微调 **Phi3 模型** 时面临的挑战，指出了高 RAM 占用和处理速度慢的问题。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 条消息): 

aillian7: 是否有可用于对话场景的 ORPO 格式？
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1232643526744084581)** (9 条消息🔥): 

- **Internist.ai 为医疗 AI 铺平道路**：新发布的 **Internist.ai 7b** 医疗语言模型由医疗专业人士打造，性能超越了 **GPT-3.5**，并在 USMLE 考试中达到了及格分数。它证明了通过精心策划的数据集进行“医生在环”（physician-in-the-loop）的方法可以优于大数据量，并计划推出具有更多数据的更大模型。[点击此处查看模型！](https://huggingface.co/internistai/base-7b-v0.2)

- **对比性能解锁临床应用**：经过 10 名医生的手动评估以及与 **GPT-4** 的盲测对比，Internist.ai 在 100 个医疗开放式问题中表现出更优越的性能。

- **笑声是良药？**: 两位成员用简单的反应 "shees" 和 "damn" 表达了他们的惊讶，随后另一位成员称赞了该模型的能力，感叹道 "该死，它正在横扫所有其他 7b 模型 😄"。

- **Llama 在 8b 规模的竞争优势**: 尽管 Internist.ai 获得了赞誉，但有人指出 **llama 8b** 产生的结果几乎相同，虽然它是一个更大的 8b 模型，这使其在该领域具有潜在的竞争优势。

- **训练的磨难与考验**: 关于训练 **llama3** 尝试的更新显示目前存在困难，计划在适当的合并（merges）到位并克服挑战后继续进行。

**提到的链接**: <a href="https://huggingface.co/internistai/base-7b-v0.2">internistai/base-7b-v0.2 · Hugging Face</a>: 未找到描述

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1232631541784379442)** (10 messages🔥): 

- **在 Hugging Face 中选择性使用数据集**: 成员们询问了如何仅使用 Hugging Face 数据集的一部分。文中详细解释了在 `load_dataset` 中使用 `split` 参数的方法，展示了*切片语法*以及加载数据集百分比、创建自定义 split 以及使用 `DatasetDict` 和 `train_test_split` 进行随机数据集分割的方法。

- **优化器与 FSDP 的兼容性**: 一位用户询问哪些优化器与 Fully Sharded Data Parallel (FSDP) 兼容。澄清指出，虽然大多数 PyTorch 优化器都可以与 FSDP 配合使用，但使用像 FairScale 的 *Sharded Optimizer* 这样的优化器封装器可以提高效率。

- **FSDP 与优化器限制**: 一位用户报告了一个错误，指出 *FSDP Offload* 与优化器 `paged_adamw_8bit` 不兼容。这指向了 FSDP 特性与某些特定优化后的优化器之间的兼容性问题。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f65c9e42-0ffc-4336-9b7b-5722eb092272)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=9e1f5025-e339-4ebe-b0d4-40e5e2c39c67)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1232662072949604455)** (61 messages🔥🔥): 

- **探索使用 tinygrad 开发隐私增强工具**: 一位成员询问了使用 tinygrad 重写针对人脸识别系统的隐私保护工具（如 [Fawkes](https://github.com/Shawn-Shan/fawkes)）的可行性。
- **Riser 转接卡故障与水冷烦恼**: 用户讨论了在配置中使用 PCIE 5.0 LINKUP riser 转接卡的问题，遇到了许多错误，一些人建议完全避免使用转接卡，并提到了 mcio 和自定义 cpayne pcb 等替代方案。解决方案包括建议咨询 [C-Payne PCB Design](https://c-payne.com) 以获取硬件需求，而一位用户考虑转向水冷，但面临 NVLink 适配器的限制。
- **寻求 tinygrad 操作的文档**: 由于缺乏对其预期行为的描述性解释，有人提出了对 tinygrad 操作规范文档的需求。
- **tinygrad 成功的战略合作伙伴关系**: George Hotz 强调，建立合作伙伴关系对 tinygrad 的成功至关重要，并指出 comma 参与制造 tinybox 硬件就是一个典型的例子。
- **tinygrad 的 MNIST 教程已上线**: George Hotz 分享了一个旨在 GPU colab 中使用的 MNIST 教程链接，以鼓励用户尝试 tinygrad，该教程可以在 [tinygrad 文档网站](https://tinygrad.github.io/tinygrad/mnist/)上找到。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1783527854741114981?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：[gif] 我刚才尝试阅读 tinygrad 代码时的样子 :D 我认为对 LOC 的要求（这只是简洁性的一个指标）导致了过度的压缩。你不会吹嘘你的 .min.js 代码是……</li><li><a href="https://discordapp.com/channels/1068976834382925865/1227683281269559418/1232845778259673239">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://tinygrad.github.io/tinygrad/mnist/">MNIST 教程 - tinygrad 文档</a>：未找到描述</li><li><a href="https://c-payne.com">C-Payne PCB 设计</a>：C-Payne PCB Design</li><li><a href="https://github.com/Shawn-Shan/fawkes/tree/master">GitHub - Shawn-Shan/fawkes: Fawkes，针对人脸识别系统的隐私保护工具。更多信息请访问 https://sandlab.cs.uchicago.edu/fawkes</a>：Fawkes，针对人脸识别系统的隐私保护工具。更多信息请访问 https://sandlab.cs.uchicago.edu/fawkes - Shawn-Shan/fawkes</li><li><a href="https://gist.github.com/fxkamd/ffd02d66a2863e444ec208ea4f3adc48">关于 TinyGrad 中 HSA 和 KFD 后端的观察</a>：关于 TinyGrad 中 HSA 和 KFD 后端的观察 - TinyGrad-notes.md
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1232693282258288752)** (31 条消息🔥): 

- **探索 WMMA 线程容量**：一位成员试图了解 WMMA 中线程内存的限制，暗示**每个线程可能为每个输入最多持有 128 bits**，考虑到两个输入加上一个累加器，一个线程可以管理 128 bits 乘以 3 的容量。
- **Tensor Core 维度与循环**：针对在 CUDA 中使用 Tensor Core 处理矩阵乘法提供了说明，强调了可以使用如 **m16n8k16** 等不同的维度，并利用了 K、M 和 N 维度的循环。
- **隔离 Kernel 崩溃的困难**：成员们正尝试通过重现条件来隔离 Kernel 崩溃，但在运行自定义脚本时遇到了与预定义示例（如 `BEAM=2 python extra/gemm/simple_matmul.py`）不同的结果。
- **Tensor 排序的影响**：围绕 tinygrad 中不存在 Tensor 排序功能展开了讨论，促使一位合作者分享了他们编写的自定义 1D 双调合并排序（bitonic merge sort）函数，支持长度为 2 的幂。
- **手动梯度分配与 Quantile 实现**：提出了关于手动为 Tensor 分配梯度以及在 tinygrad 中实现等效 **torch.quantile** 的疑问，揭示了为扩散模型（diffusion models）设计采样算法的意图。
  

---



**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1232756366792327301)** (2 条消息): 

- **Modular Twitter 抢先看**：Modular 向关注者展示了一个即将推出的功能，引导他们访问其 Twitter 帖子，并通过一个[短链接](https://twitter.com/Modular/status/1783194701925134731)进行抢先预览。
- **更多 Modular Twitter 热点**：另一条 [Modular 的推文](https://twitter.com/Modular/status/1783575774085410911)引起了社区的关注，暗示了将在其官方 Twitter 动态中追踪的进一步进展。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1233116804256235520)** (1 条消息): 

- **保护 Modular 免受供应链攻击**：由于频繁发布新软件，Modular 面临着独特的安全挑战，因为现代交付机制越来越容易受到攻击。针对 [XZ 供应链攻击](https://www.synopsys.com/blogs/software-security/open-source-trends-ossra-report.html)凸显了建立强大防御体系的紧迫性，这促使 Modular 自 2023 年秋季发布 Mojo 以来，一直将安全的软件交付放在首位。

- **当今软件生态系统的脆弱性**：随着开源软件使用的增加（代码库平均包含超过 500 个开源组件），对强大软件交付系统的需求至关重要。Modular 的生态系统采用了 SSL/TLS、GPG 签名系统以及经过验证的……（消息截断）

**提到的链接**：<a href="https://www.modular.com/blog/preventing-supply-chain-attacks-at-modular">Modular：在 Modular 预防供应链攻击</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：在 Modular 预防供应链攻击

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1232853349578313759)** (4 条消息):

- **量子力学：不仅仅是随机性？**：一位参与者表示，许多人误解了量子物理和计算，认为它们是基于随机性的，而事实上物理学家在这一点上存在很大分歧。这场辩论延伸到了**计算机科学和 ML/AI** 领域，在这些领域中，随机性（虽然不是真正的随机性）起着基础性的作用。
  
- **量子解释的分歧点**：同一位参与者表达了对**多世界诠释（Many-Worlds interpretation）和模拟假设（Simulation Hypothesis）**的偏好，强调了量子物理研究者之间的思想分歧，即支持**哥本哈根诠释（Copenhagen interpretation）**的人与其他人的分歧。
  
- **尽管量子具有复杂性，但关注点可能有所不同**：另一位用户建议，在考虑计算架构的整体重要性时，处理单元的类型（无论是量子还是经典）可能并不是关键。

- **几何见解可能驯服量子随机性**：一位成员讨论了振幅多面体（amplituhedron），这是一种可能简化量子计算的几何结构，并提出使用几何原理可以增强我们在**量子算法和电路设计**方面的理解和效率。
  
- **ML 作为复杂量子可视化的关键**：他们进一步推测，机器学习可能被用于解释量子计算中涉及的复杂、超维可视化，特别是在我们处理不断增加的 qubit 数量和纠缠等现象时。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1232772179427594414)** (21 messages🔥): 

- **新的别名建议引发笑声**：有人开玩笑地建议在 Mojo 语言中将 `IntLiteral` 重命名为 `LitInt`，并附带一个关于 `mojo format` 在所有 `.mojo` 文件顶部添加别名定义的笑话。
- **对 Mojolicious 周边的渴望**：成员们表达了对官方 Mojo 周边的渴望，并畅想未来可能推出的产品，从 **Mojo emoji** 装备到幻想中的 MojoCON 工作人员制服。
- **API 开发导致性能问题**：一位成员描述了一个 HTML API 导致 Language Server Protocol (LSP) 延迟并使 `mojo build` 停滞的问题，这表明他们正在挑战当前工具集的极限。
- **关于内存管理和安全性的担忧**：在讨论 Mojo 中自定义类型的创建时，一位用户得到保证，操作系统会在进程结束后清理内存，这意味着在 Mojo 中不需要手动内存管理。
- **引擎需求查询被重定向**：关于 MAX 引擎是否需要 AVX 支持的查询被重定向到了更合适的频道，这表明针对特定频道的查询能进一步提供信息并减少讨论干扰。
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1232619837642244168)** (15 messages🔥): 

- **PyConDE 演讲反思 Mojo 的第一年**：在柏林举行的 PyConDE 上，一场探讨 **Mojo 作为 Python 超集的影​​响力和潜力**的演讲受到关注，质疑其在 Rust、PyPy 和 Julia 等替代方案中脱颖而出的能力。题为“Mojo: Python's faster cousin?”的演讲讨论了 Mojo 是否能兑现其承诺，还是仅仅成为编程语言的一个注脚，演讲录像将很快发布（[点击观看](https://pretalx.com/pyconde-pydata-2024/talk/DG8G7Q/)）。

- **Rust 社区对 Mojo 的怀疑**：社区讨论了 Rust 社区对 Mojo 的怀疑态度，指出 Rustaceans 倾向于否定 Mojo 的主张，有些人认为这些主张缺乏实质性证据支持。有人分享了[一篇博客文章](https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6)，辩论 Mojo 的安全性和速度，特别是与 Rust 处理可变别名引用（mutable aliased references）的方法相比。

- **对 Mojo 基准测试的期待**：成员们交流了对 Mojo 在未来基准测试中表现的期待，一些人指出，虽然 GitHub 上的基准测试应被视为一种视角而非最终定论，但其他人仍热切期待 Mojo 与其他编程语言的对比结果。

- **Mojo 备受争议的营销策略**：分享了一个批评 Mojo 营销方式的讨论链接，认为其更多地关注于针对编程领域的影响者（influencers），而不是制作深入的技术内容，这可能表明了一条具有战略性但有争议的普及路径。

- **向学生分享 Mojo 见解**：一位成员分享了布达佩斯技术与经济大学 Google Developer Student Club 举办的关于 **Python and Mojo** 的活动，重点介绍了 Mojo 的特性以及与 Python 的集成。活动笔记旨在帮助学生更好地理解 Mojo，目前已公开，该成员正在寻求向学生推广 Mojo 的建议（[活动详情](https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/)）。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6)">Mojo vs. Rust: Mojo 比 Rust 快吗？ | Lobsters</a>：未找到描述</li><li><a href="https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/.">Python and Mojo: 好、坏与未来 | Google Developer Student Clubs</a>：线下活动 - 加入我们，参加关于 Mojo 的独家演示，这是一种基于 Python 语法并具备系统编程能力的语言。</li><li><a href="https://pretalx.com/pyconde-pydata-2024/talk/DG8G7Q/">来自 Mojo 🔥 的推文 - 它是 Python 更快的表亲还是仅仅是炒作？PyConDE &amp; PyData Berlin 2024</a>：在 2023-05-02，科技界因 Mojo 🔥 的发布而沸腾，这是一种由 Chris Lattner 开发的新编程语言，他以在 Clang, LLVM 和 Swift 方面的工作而闻名。它被誉为 &quot;Python 的...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1232672346381357118)** (5 条消息): 

- **讨论双 RNG 系统**：讨论了关于拥有**两个版本的随机数生成器 (RNG)** 的议题：一个针对**性能**进行了优化，另一个则是**加密安全**且在常数时间内执行。
- **探索 Mojo 对 RVO 的支持**：一直在测试 **Mojo 是否像 C++ 那样支持返回值优化 (RVO)**。由于出现了非预期行为，在一位成员的建议下开启了 [这个 issue](https://github.com/modularml/mojo/issues/2410)。

**提及的链接**：<a href="https://github.com/modularml/mojo/issues/2410)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。

  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 31 期
https://www.modular.com/newsletters/modverse-weekly-31
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1232620335426179122)** (4 条消息): 

- **解释聊天中的缩写**：一位用户询问缩写 "QPS"，另一位用户澄清其为 **Queries Per Second**（每秒查询数）。
- **对进展持乐观态度**：一位成员表达了对未来改进的乐观，暗示系统性能预计将继续保持上升趋势。
- **对性能对等的好奇**：一位用户提到一个奇特的观察，尽管架构不同，**PyTorch 和 TensorFlow** 报告了相同的性能提升，他们觉得这很有趣。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1232609249469665311)** (32 条消息🔥): 

- **空字符串 Bug 警报**：提到了一个 `print(str(String()))` 导致后续 `String()` 值打印损坏的问题，随后提示提交 Bug 报告。该问题已记录在 GitHub 上，编号为 [#2392](https://github.com/modularml/mojo/issues/2392)。

- **字符串终止符困扰**：关于以 null 结尾的字符串的讨论强调了它们是 stdlib 中频繁出现 Bug 的根源，至少有 5 个 Bug 归因于此。成员们认识到这些字符串对于 C 互操作性是必要的，但随着对 C++ 运行时的依赖减少，有可能对其进行修订。

- **Nightly Mojo 编译器更新发布**：新版本的 nightly Mojo 编译器已发布，鼓励用户使用 `modular update nightly/mojo` 进行更新。自上次 nightly 以来的 diff 可以在 [这里](https://github.com/modularml/mojo/pull/2396/files) 查看，changelog 在 [这里](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **函数重载扩展**：对于新增加的重载方法引发了一些笑谈，参考了一个 pull request 讨论，其中提到一个外部调用需要 15 个不同的参数。相比当前方法，人们更倾向于使用元组或异构变长参数列表来实现更好的函数。

- **SIMD 和 EqualityComparable Hack**：一位成员发现了一个 hack，允许利用函数重载优先级使 `SIMD` 符合 `EqualityComparable`。对话讨论了其缺点，特别是需要在 stdlib 类型中实现它的必要性，以及对可能更好的方法的探索。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/parameters/#overloading-on-parameters>">参数化：编译时元编程 | Modular Docs</a>：参数和编译时元编程的介绍。</li><li><a href="https://github.com/microsoft/GSL/blob/main/docs/headers.md#zstring)">GSL/docs/headers.md at main · microsoft/GSL</a>：Guidelines Support Library。通过在 GitHub 上创建账户来为 microsoft/GSL 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2392)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2396/commits/4d4c2119799c42c29bd14a3ca8a72ce3e9feabd7">[stdlib] Update stdlib corresponding to `2024-04-24` nightly/mojo by patrickdoc · Pull Request #2396 · modularml/mojo</a>：这使用与今天的 nightly 版本对应的内部 commit 更新了 stdlib：mojo 2024.4.2414 。</li><li><a href="https://github.com/modularml/mojo/pull/2396/files">[stdlib] Update stdlib corresponding to `2024-04-24` nightly/mojo by patrickdoc · Pull Request #2396 · modularml/mojo</a>：这使用与今天的 nightly 版本对应的内部 commit 更新了 stdlib：mojo 2024.4.2414 。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1232644767570071563)** (77 条消息🔥🔥)：

```html
<ul>
  <li><strong>Transformer Architecture Tweaks Under Discussion:</strong> Members were discussing an approach to improve transformer models by taking inputs from intermediate attention layers in addition to the last attention layer, likening the method to the Pyramid network in CNN architectures.</li>
  <li><strong>TherapistAI Sparks Controversy:</strong> A member highlighted the controversy surrounding levelsio's <a href="https://twitter.com/meijer_s/status/1783032528955183532">TherapistAI</a> on Twitter, criticizing its potentially misleading suggestion that it could replace a real therapist.</li>
  <li><strong>Semantic Search Solution Inquiry:</strong> A discussion about finding a good semantic search API like <a href="https://newsapi.org">newsapi.org</a> led to recommendations including<a href="https://omnisearch.ai/"> Omnisearch.ai</a>, though it wasn't a fit for scanning the web for news.</li>
  <li><strong>France Steps Towards LLMs in the Public Sector:</strong> There was a conversation regarding France's experimental incorporation of LLMs into public administration, with insights and opinions shared about France's innovation and political climate, linking to a <a href="https://twitter.com/emile_marzolf/status/1783072739630121432">tweet about the topic</a>.</li>
  <li><strong>AI Winter Predictions Stir Discussion:</strong> Users deliberated over the state and future of AI venture funding prompted by a <a href="https://x.com/schrockn/status/1783174294865887521?s=46&t=90xQ8sGy63D2OtiaoGJuww">tweet on AI bubble effects</a>, reflecting on the implications of a potential bubble burst for AI innovation.</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/schrockn/status/1783174294865887521?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Nick Schrock (@schrockn) 的推文</a>：我不确定如果 AI 泡沫在紧邻美联储泡沫破裂时也随之破裂，风险投资环境会变成什么样。又一批基金年份遭到重创可能会耗尽整个生态系统的...</li><li><a href="https://x.com/schrockn/status/1783174294865887521?s=46&t=90xQ8sGy63D2Ot">来自 Nick Schrock (@schrockn) 的推文</a>：我不确定如果 AI 泡沫在紧邻美联储泡沫破裂时也随之破裂，风险投资环境会变成什么样。又一批基金年份遭到重创可能会耗尽整个生态系统的...</li><li><a href="https://flashcardfy.lol">Flashcardfy - 带有个性化反馈的 AI 抽认卡生成器</a>：通过提供个性化反馈的 AI 生成抽认卡，让学习更快速、更智能。</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">欢迎来到社区计算机视觉课程 - Hugging Face 社区计算机视觉课程</a>：未找到描述</li><li><a href="https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca">OpenELM Instruct Models - Apple 集合</a>：未找到描述</li><li><a href="https://github.com/langgenius/dify?tab=readme-ov-file">GitHub - langgenius/dify: Dify 是一个开源的 LLM 应用开发平台。Dify 直观的界面结合了 AI 工作流、RAG 流水线、Agent 能力、模型管理、可观测性功能等，让你快速从原型走向生产。</a>：Dify 是一个开源的 LLM 应用开发平台。Dify 直观的界面结合了 AI 工作流、RAG 流水线、Agent 能力、模型管理、可观测性功能等，让你快速...</li><li><a href="https://github.com/e2b-dev/code-interpreter">GitHub - e2b-dev/code-interpreter: 用于为你的 AI 应用添加代码解释功能的 Python &amp; JS/TS SDK</a>：用于为你的 AI 应用添加代码解释功能的 Python &amp; JS/TS SDK - GitHub - e2b-dev/code-interpreter: 用于为你的 AI 应用添加代码解释功能的 Python &amp; JS/TS SDK</li><li><a href="https://omnisearch.ai/">Omnisearch</a>：Omnisearch 开创性的搜索产品使你网站上的所有内容都可搜索，从音频/视频到文本、文档和演示文稿。</li><li><a href="https://vram.asmirnov.xyz">VRAM Calculator</a>：未找到描述
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1232703729506189363)** (47 条消息🔥): 

- **在 LangChain 中探索 pgvector**: 一位成员询问如何使用 pgvector 存储作为聊天机器人的上下文，并获得了关于初始化和文档检索的详细步骤及资源。使用的方法包括来自 `PGVector` 存储的 `max_marginal_relevance_search_by_vector`（用于查找相关文档），以及用于生成 Embeddings 的 `OpenAIEmbeddings`。更多技术细节可以在 [LangChain GitHub 仓库](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/pgvector.py)中找到。

- **剖析 SelfQueryRetriever 机制**: 讨论了来自 LangChain `langchain_community.retrievers` 库的 `SelfQueryRetriever`，概述了其在从 `PGVector` 存储的特定表中获取文档的用途。该方法需要设置多个环境变量并通过 `invoke` 方法检索文档。

- **为对话式 AI 构建图 (Graph)**: 分享了关于创建一个图来决定是获取文档作为上下文还是直接使用 LangChain 的 `createStuffDocumentsChain` 进行回复的建议。图的构建涉及设置聊天模型并使用适当的上下文调用它。

- **Chroma 和 Vercel AI SDK 推荐**: 一位成员建议关注 Vercel AI SDK 和 Chroma，但未提供具体原因或链接。

- **开始使用 LangChain**: 一位新用户表达了使用 LangChain 构建聊天机器人的兴趣，寻求建议并确认了他们打算使用 LangChain 的功能，如 Redis 支持的聊天记忆和 function calling。他们收到了 LangChain 聊天机器人文档的链接作为资源。

- **区分存储 (Stores) 和聊天记忆 (Chat Memory)**: 用户澄清了用于聊天消息历史的 Redis 存储与通用的键值对 RedisStore 之间的区别，指出聊天消息历史用于按会话 (session) 持久化聊天消息，而 RedisStore 更加通用，旨在通过字符串键存储任何字节值。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://<your-endpoint.openai.azure.com/">">未找到标题</a>: 未找到描述</li><li><a href="http://your-corporate-proxy:8080">">未找到标题</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/use_cases/chatbots/">聊天机器人 | 🦜️🔗 LangChain</a>: 概览</li><li><a href="https://python.langchain.com/docs/integrations/chat/">聊天模型 | 🦜️🔗 LangChain</a>: 特性（原生支持）</li><li><a href="https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/">Redis | 🦜️🔗 LangChain</a>: [Redis (远程字典</li><li><a href="https://python.langchain.com/docs/integrations/stores/redis/">RedisStore | 🦜️🔗 LangChain</a>: RedisStore 是 ByteStore 的一种实现，用于存储</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/google_cloud_sql_pg#load-documents-via-default-table>)">Google Cloud SQL for PostgreSQL | 🦜️🔗 LangChain</a>: Cloud SQL for PostgreSQL</li><li><a href="https://python.langchain.com/docs/integrations/document_loaders/google_cloud_sql_pg#load-documents-via-custom-tablemetadata-or-custom-page-content-columns>)">Google Cloud SQL for PostgreSQL | 🦜️🔗 LangChain</a>: Cloud SQL for PostgreSQL</li><li><a href="https://python.langchain.com/docs/integrations/retrievers/self_query/pgvector_self_query#testing-it-out>)">PGVector (Postgres) | 🦜️🔗 LangChain</a>: PGVector 是一个向量</li><li><a href="https://python.langchain.com/docs/integrations/retrievers/self_query/pgvector_self_query#filter-k>)">PGVector (Postgres) | 🦜️🔗 LangChain</a>: PGVector 是一个向量</li><li><a href="https://github.com/langchain-ai/langchain/issues/15527>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/retrieval#document-chains>)">检索 | 🦜️🔗 Langchain</a>: 检索是聊天机器人常用的一种技术，用于通过聊天模型训练数据之外的数据来增强其响应。本节将介绍如何在聊天机器人的上下文中实现检索...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1232935474184847381)** (1 条消息): 

- **寻求 LLaMA-3 的模板结构**：一位成员询问在 **LLaMA-3 prompt template** 中是否存在用于提供问题上下文的标头（headers），并引用了 [官方文档](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)。由于该模型最近才发布，人们对文档的完整性表示担忧。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1232671645987246090)** (6 条消息): 

- **使用 LangChain 扩展 RAG**：一篇文章详细介绍了如何使用 LangChain 的 LangGraph 将 **自适应路由 (adaptive routing)、纠错回退 (corrective fallback) 和自我修正 (self-correction)** 技术集成到检索增强生成 (RAG) 框架中。更多探索内容可在 [Medium](https://medium.com/ai-advances/unifying-rag-frameworks-harnessing-the-power-of-adaptive-routing-corrective-fallback-and-1af2545fbfb3) 上阅读。
  
- **寻找 Pull Request 合作伙伴**：一位成员询问在哪里可以请求对合作伙伴的 Pull Request 进行评审，并考虑 **share-your-work 频道** 是否适合此类讨论。

- **酿造叛乱 (The Brewed Rebellion)**：分享了一款名为 "Brewed Rebellion" 的新 **文字 RPG** 游戏，鼓励玩家作为 StarBeans 的咖啡师，在职场政治中周旋以组建工会。在 [play.lab.ai](https://www.playlab.ai/project/clveh1d860lwsrv1nvl4xfksp) 体验其中的尔虞我诈。

- **Collate 介绍**：介绍了一个名为 Collate 的平台，它可以将保存的文章转换为易于阅读的每日简报。欢迎提供反馈，你可以在 [collate.one](https://collate.one/newsletter) 进行尝试。

- **Writesonic 和 Copy.ai 的克隆版上线**：BlogIQ 是一款由 OpenAI 和 LangChain 驱动的新应用，旨在简化博主的内容创作流程，现已在 [GitHub](https://github.com/langchain-tech/BlogIQ) 上发布。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.playlab.ai/project/clveh1d860lwsrv1nvl4xfksp">Brewed Rebellion</a>: 未找到描述</li><li><a href="https://github.com/langchain-tech/BlogIQ">GitHub - langchain-tech/BlogIQ: Clone of writesonic.com &amp; copy.ai - BlogIQ is an innovative app powered by OpenAI and Langchain, designed to streamline the content creation process for bloggers.</a>: writesonic.com 和 copy.ai 的克隆版 - BlogIQ 是一款由 OpenAI 和 LangChain 驱动的创新应用，旨在简化博主的內容创作流程。 - langchain-tech/BlogIQ</li><li><a href="https://github.com/jwa91/LangGraph-Expense-Tracker">GitHub - jwa91/LangGraph-Expense-Tracker: LangGraph - FastAPI - Postgresql - AI project</a>: LangGraph - FastAPI - Postgresql - AI 项目。可以通过在 GitHub 上创建账号来为 jwa91/LangGraph-Expense-Tracker 的开发做出贡献。</li><li><a href="https://app.excalidraw.com/l/5NC0r7Sejhe/39ULXmBwigA">Whiteboarding made easy</a>: 具有手绘体验的白板工具。非常适合进行面试、绘制图表、原型或草图等！
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1232876596923469904)** (1 条消息): 

- **Embedding Models 对决**：一位成员通过分享视频提供了关于目前最佳 **embedding models** 的见解。内容旨在澄清关于模型选择的疑问，可以在[这里](https://www.youtube.com/watch?v=i2FJVyDIhaI)查看。
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1232721589183709205)** (42 条消息🔥): 

- **Cohere Toolkit 正式开源**：[Cohere's Toolkit](https://cohere.com/blog/cohere-toolkit) 开源的激动人心消息传开，允许用户添加自定义数据源并部署到云端。它支持在各种云平台上使用 Cohere 模型。

- **分享 Cohere Toolkit 的 GitHub 仓库**：用户分享了 Cohere Toolkit 的 [GitHub 仓库](https://github.com/cohere-ai/cohere-toolkit)，因其加速了 RAG 应用的部署而获得赞誉。

- **Toolkit 故障排除与咨询**：一位用户报告了在 Docker for Mac 上使用 Toolkit 处理文件时的困难。另一位用户询问了使用 Cohere API key 在 Azure 上部署应用的隐私问题，**@co.elaine** 澄清说，为了实现完全隐私，API key 是可选的。

- **API 与 Playground 结果不匹配**：一位用户在将 API 与 Playground 进行对比时遇到了结果不匹配的问题，特别是在代码中加入 site connector grounding 时。通过反复的帮助，他们进行了一些修正，但仍观察到 Playground 和其脚本之间的响应略有不同。

- **对 Cohere 团队的支持与致谢**：用户表达了他们的感激之情，特别是对联合创始人 **@mrdragonfox** 和 **@1vnzh** 的努力以及开源工具包的发布表示感谢，强调了这将如何使开源社区受益。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Trelis/Meta-Llama-3-70B-Instruct-function-calling">Trelis/Meta-Llama-3-70B-Instruct-function-calling · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/">GitHub - cohere-ai/cohere-toolkit: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Toolkit 是预构建组件的集合，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://coral.cohere.com/">Login | Cohere</a>: Cohere 通过一个易于使用的 API 提供对高级 Large Language Models 和 NLP 工具的访问。免费开始使用。</li><li><a href="https://cohere.com/blog/cohere-toolkit">New Cohere Toolkit Accelerates Generative AI Application Development</a>: 介绍 Cohere Toolkit，这是一个可跨云平台部署的生产级应用的开源仓库。
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1232944667276869663)** (6 条消息):

- **误判攻击性文章**：一场讨论展开，针对某篇文章被指控为针对 **Cohere** 的“攻击性文章（hit piece）”进行了说明，参与者承认不记得所引用内容的具体出处。
- **Jailbreak 问题的核心**：对话者探讨了文章的核心论点，将其信息总结为：为 **Cohere** 的 **LLM** 添加 Jailbreak 可能会导致创建出具有恶意意图的 D.A.N-agents。
- **记忆模糊，而非逻辑缺失**：文章的批评者受到质疑，因为其未能提供文章中的具体细节来支持其关于该文章对 **Cohere** 存在不公平批评的指控。
- **辩论支持主张的责任**：有人指出，如果一个人要将一篇文章贴上恶意动机的标签，就应该准备好用所讨论作品中的具体观点来证实这种攻击。
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1232723392092700735)** (32 messages🔥): 

- **探索 OpenInterpreter 性能**：成员们报告了在 **OpenInterpreter** 上使用不同模型的成功程度各异，其中 **Wizard 2 8X22b** 和 **gpt 4 turbo** 在遵循系统消息指令和 function calling 方面表现最佳。已注意到 **llama 3** 等模型表现出不一致的问题。

- **本地模型执行困惑**：在使用不同模型本地执行代码时似乎存在困惑，因为在终端中与 **OpenInterpreter** 交互并不总是会导致代码实际运行，尽管模型写出了代码。使用额外的标志 `--no-llm_supports_functions` 被引用为纠正某些错误的解决方案。

- **OpenInterpreter 更新需求**：讨论了修复 **OpenInterpreter** 本地模型所需的更新，建议使用 `-ak dummykey` 标志进行改进。如果问题仍然存在，成员将被引导至特定的 Discord 频道寻求帮助。

- **AI 设备的 UI 开发**：关于为“AI 设备”构建用户界面的替代方案的询问，引发了围绕 tkinter 之外的 UI 开发选项的对话，并考虑了未来在微控制器上的使用。

- **视觉模型讨论与演示**：分享了讨论计算机视觉模型的 **GitHub** 仓库和论文链接，重点关注像 **moondream** 这样的轻量级模型，还提到了在不同的 quantization 设置下运行 **llama3** 等模型以管理 **VRAM** 使用。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2112.10003">使用文本和图像提示的图像分割</a>：图像分割通常通过为一组固定的对象类别训练模型来解决。稍后加入额外的类别或更复杂的查询是昂贵的，因为这需要重新训练...</li><li><a href="https://huggingface.co/liuhaotian/llava-v1.6-34b">liuhaotian/llava-v1.6-34b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py">bootcupboard/llm-img/Img2TxtMoondream.py at main · CodeAKrome/bootcupboard</a>：内部比外部更大！通过在 GitHub 上创建一个账户，为 bootcupboard 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/display/point/point.py">open-interpreter/interpreter/core/computer/display/point/point.py at main · OpenInterpreter/open-interpreter</a>：计算机的自然语言接口。通过在 GitHub 上创建一个账户，为 OpenInterpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#interactive-chat">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>：计算机的自然语言接口。通过在 GitHub 上创建一个账户，为 OpenInterpreter 的开发做出贡献。</li><li><a href="https://changes.openinterpreter.com/log/ncu-ii">Open Interpreter - The New Computer Update II</a>：开源 Open Interpreter 项目的官方变更日志。</li><li><a href="https://github.com/vikhyat/moondream">GitHub - vikhyat/moondream: 微型视觉语言模型</a>：微型视觉语言模型。通过在 GitHub 上创建一个账户，为 moondream 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1232645981607039009)** (14 messages🔥):

- **01 项目扩展**：成员们讨论了在**外部设备上运行 01 项目**，灵感来自 Jordan Singer 的 [Twitter 帖子](https://twitter.com/jsngr/status/1774110742070882478)等实现。
- **Killian 的蜘蛛场景**：另一个突显 **01 灵活性**的例子是将 AI 嵌入到 Grimes 的蜘蛛中，由一名成员分享了 [Killian 的推文](https://x.com/hellokillian/status/1782859388237279634)。
- **01 产品开发咨询**：有人询问了 **01 light 产品设计**的现状，以及如何使用 M5 echo 设备复制 Killian 视频中展示的功能。
- **01 的 Windows 11 安装指南**：提供了一套在 **Windows 11** 上安装 01 项目的详细说明，包括安装 [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) 的链接以及该软件的 GitHub [仓库](https://github.com/Abdullah-Gohar/01.git)。
- **在 01 上运行本地模型**：对话包含了在 01 上运行 **Ollama** 等本地语言模型的指南，使用命令 `poetry run 01 —local` 并选择所需的模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hellokillian/status/1782859388237279634">来自 killian (@hellokillian) 的推文</a>：我们将 01 放入了 @grimezsz 的蜘蛛中</li><li><a href="https://github.com/Abdullah-Gohar/01.git">GitHub - Abdullah-Gohar/01: 开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 Abdullah-Gohar/01 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

8i8__papillon__8i8d1tyr: https://mlflow.org/
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1232659414918365194)** (3 条消息): 

- **Phi-3-128K 进入盲测环节**：一名成员透露 **Phi-3-128K** 已进入测试阶段，并被纳入盲测中。
- **以身份询问作为对话开始**：提到现在的对话通常以“**你是谁**”这个问题作为标准交互的开始。
- **防止模型名称泄露**：成员们了解到 **LMSys** 的设计旨在排除可能泄露模型名称的回复，以维护盲测的公正性。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1232767722715353259)** (17 条消息🔥): 

- **指令微调相关性探讨**：一名成员对 Sebastian Ruder 文章中提到的[指令遵循评估](https://newsletter.ruder.io/p/instruction-tuning-vol-2)的相关性和持久流行度提出疑问，寻找评估其影响的方法。
- **LLM 的新型评估基准**：讨论涉及了新的基准测试，如以简单且可解释的方式测试大语言模型的 [LMentry](https://arxiv.org/abs/2211.02069)；探测模型对不同语言特征行为的 [M2C](https://arxiv.org/abs/2307.05454)；以及提供可验证指令遵循评估的 [IFEval](https://arxiv.org/abs/2311.07911)。
- **复杂中的简单**：在各种进步中，一位用户倾向于使用 MMLU-ChatBotArena 进行 GPU 效能评估的简洁性，并幽默地自嘲在该领域是个“简单主义者”。
- **地平线上的 HELM**：一位用户注意到 HELM 团队最近更新了用于模型性能分析的内省功能，并询问了 HELM 目前在 AI 社区的状态和影响。
- **Repligate 关于 RL 复杂性的讨论**：分享了来自 `https://x.com/repligate` 的一个帖子，讨论了 RLHF 等强化学习技术对 AI 模型的潜力和挑战，并赞扬了在训练 Claude 等模型时出现的有趣结果和异常现象。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw">来自 j⧉nus (@repligate) 的推文</a>：毫无疑问，在 teacher-forcing 之上，有多种方式可以进行 RL/生成-判别/合成数据/类 self-play 训练，使模型变得更聪明，但尤其是更……</li><li><a href="https://arxiv.org/abs/2211.02069">LMentry：基础语言任务的语言模型基准</a>：随着大型语言模型性能的快速提升，基准测试也变得越来越庞大和复杂。我们提出了 LMentry，这是一个通过专注于……来避免这种“军备竞赛”的基准。</li><li><a href="https://arxiv.org/abs/2307.05454">利用类型学特征增强 NLP 模型的跨语言行为测试</a>：开发面向全球语言的 NLP 系统面临的一个挑战是，理解它们如何泛化到与现实应用相关的类型学差异。为此，我们提出了 M...</li><li><a href="https://arxiv.org/abs/2311.07911">大型语言模型的指令遵循评估</a>：大型语言模型（LLM）的核心能力之一是遵循自然语言指令。然而，此类能力的评估尚未标准化：人工评估昂贵、缓慢且……
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1232738137915260998)** (12 messages🔥): 

- **关于错过推文的反思**：一位成员意识到没看到关于某篇 **blog post** 的推文是因为屏蔽了 "AGI" 这个词。
- **Cohere 发布聊天界面源码**：Cohere 决定开源其聊天界面，引发了积极评价，并有建议认为某位成员的建议可能影响了这一举动。源代码可在 [GitHub](https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file) 获取。
- **AI 行业影响力**：几位成员开玩笑说 **Nathan Lambert** 在 AI 行业的影响力日益增长，并打趣说可能需要给他配上“激光眼”。
- **公司动机受质疑**：讨论了某公司行为背后的古怪逻辑，但未指明细节，并暗示这种逻辑其实并不重要。
- **对“Pick Your Brain”一词的反感**：表达了对 *“pick your brain”*（向你请教）一词的不适感，尤其是在人们很忙需要拒绝此类请求时。

**提到的链接**：<a href="https://fxtwitter.com/nickfrosst/status/1783220910427709766?s=46">来自 Nick Frosst (@nickfrosst) 的推文</a>：我们开源了我们的聊天界面。https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1232684893427728486)** (10 messages🔥): 

- **SnailBot 交付**：SnailBot 似乎正在运行，并征求社区关于通知标签是否令人困扰的反馈。
- **“Reward Is Enough” 链接问题**：一位成员报告无法访问 [“Reward is Enough” 文章](https://www.sciencedirect.com/science/article/pii/S0004370221000862)，遇到了内容访问问题，这可能被误认为是付费墙。
- **无账号，无访问权限**：另一位成员确认，如果没有该托管网站的账号，链接是无法访问的。
- **个人问题诊断**：一位成员幽默地建议，无法访问内容的问题可能是个人问题。
- **访问问题已解决**：用户最终通过提供的链接解决了访问问题。
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1232725714189680752)** (25 messages🔥):

- **Llamafile 用户遇到 Mlock 错误**：一名成员报告在 cmd 和 powershell 上尝试运行 `phi2 llamafile` 时遇到 **"failed to mlock"** 错误，但未提供进一步的细节或解决方案。
- **对 Phi3 Llamafile 发布的期待**：关于 **Phi3 llamafile** 发布的问题已得到回复，建议使用 **Microsoft 的 GGUF 文件**，因为 Microsoft 已经详细说明了[如何在 llamafile 中使用 Phi3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile)的步骤。
- **Llama 模型的图像 Payload 问题**：用户在 JSON Payload 中发送 base64 编码图像时遇到问题，因为 llama 模型无法识别图像并标记 `multimodal : false`。该用户分享了他们的编码方法，但未说明是否已找到解决方案。
- **Mistral Llamafile 需要更新**：根据反馈，已对 [Hugging Face 仓库](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile)上的 **Mixtral 8x22B Instruct v0.1 llamafile** 文档进行了修改。
- **Windows Defender 将 Llamafile 误报为木马**：来自 Hugging Face 的一个文件被 Windows Defender 错误地标记为木马，一名成员建议了使用 VM 或将文件夹加入 Defender 设置白名单等解决方案，同时也指出了向 Microsoft 报告误报（false positives）的局限性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile>">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://www.microsoft.com/en-us/wdsi/filesubmission">提交文件进行恶意软件分析 - Microsoft Security Intelligence</a>：未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8">发布 llamafile v0.8 · Mozilla-Ocho/llamafile</a>：llamafile 允许您通过单个文件分发和运行 LLM。llamafile 是 Mozilla Ocho 在 2023 年 11 月推出的本地 LLM 推理工具，具有卓越的性能和二进制可移植性...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuec">致命错误：M1 需要 AVX CPU 特性 · Issue #327 · Mozilla-Ocho/llamafile</a>：我在尝试于 Apple M1 上运行入门指南时遇到了一个奇怪的问题。sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659">致命错误：M1 需要 AVX CPU 特性 · Issue #327 · Mozilla-Ocho/llamafile</a>：我在尝试于 Apple M1 上运行入门指南时遇到了一个奇怪的问题。sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://blog.mozilla.ai/local-llm-as-judge-evaluation-with-lm-buddy-prometheus-and-llamafile/">使用 lm-buddy、Prometheus 和 llamafile 进行本地 LLM-as-judge 评估</a>：在 AI 新闻周期中，每天都有新模型发布，成本和评估虽然很少被提及，但对开发者和企业至关重要</li><li><a href="https://huggingface.co/jartine">jartine (Justine)</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile">jartine/Mixtral-8x22B-Instruct-v0.1-llamafile · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat0">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat1">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat2">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1232673725934076005)** (7 条消息): 

- **在本地 Mixtral 中批量处理 Prompt**：一名成员正寻求在配备 2 张 A100 的本地 Mixtral 上发送 **"batch" Prompt**，但在寻找可参考的实际案例时面临挑战。他们提到之前使用过 **vLLM**，并考虑尝试最近开源的 **TGI**，想知道是否有直接的 Python 用法，或者 TGI 是否仅打算作为 API 服务器使用。

- **LLM-Swarm 用于 API 端点管理**：另一位成员推荐了 [huggingface/llm-swarm](https://github.com/huggingface/llm-swarm)，用于在 Slurm 集群中管理可扩展的开源 LLM 推理端点，尽管他们指出对于仅有的两个 GPU 来说，这可能有点大材小用。

- **TGI 主要作为 API 服务器 vs vLLM 本地 Python 模式**：在讨论 **TGI** 的用途时，一位成员建议它主要作为 API 服务器使用，并推荐使用带有 **continuous batching**（连续批处理）的异步请求。他们强调，虽然 **vLLM 的本地 Python 模式** 方便实验和开发，但组件解耦可以更轻松地在不同基础设施或通过 API 进行 LLM 切换。

- **无需 API 服务器的批量补全操作**：有成员贡献了关于使用 `litellm.batch_completion` 对 API 服务器运行批量请求的方法，但另一位成员表示他们更喜欢直接在 Python 中操作而不使用 API 服务器，打算坚持使用 **vLLM** 和 `vllm.LLM` 类。

- **TGI 专注于低延迟，vLLM 专注于高吞吐量**：一条消息强调了 **TGI** 和 **vLLM** 之间的区别，TGI 被描述为 API 优先且专注于低延迟，而 vLLM 则被认为是一个易于使用的库，面向成本效益高、高吞吐量的部署。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/llm-swarm">GitHub - huggingface/llm-swarm: Manage scalable open LLM inference endpoints in Slurm clusters</a>：在 Slurm 集群中管理可扩展的开源 LLM 推理端点 - huggingface/llm-swarm</li><li><a href="https://github.com/huggingface/text-generation-inference/issues/1008#issuecomment-1742588516">Batch generate? · Issue #1008 · huggingface/text-generation-inference</a>：系统信息：你好，我想请问是否可以进行批量生成？client = Client("http://127.0.0.1:8081", timeout = 60) gen_t = client.generate(batch_text, max_new_tokens=64) generate c...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1232614854305513522)** (10 条消息🔥): 

- **德语 Prompt 的细微差别**：一位用户询问了在德语中使用非正式的 "du" 与正式的 "Sie" 对 DiscoLM 系列模型进行 Prompt 引导时的影响，寻求基于他人经验的见解。

- **文本摘要限制的挑战**：一位用户表示，在让 DiscoLM 模型遵守 150 个单词或 1000 个字符的文本摘要任务限制时遇到了困难，并指出 `max_new_tokens` 参数会导致句子中途被截断。

- **关于分享量化版本的查询**：一位成员考虑为 Llama3_DiscoLM_German_8b_v0.1 模型的实验版本提供量化（quantifications），询问是现在贡献更好，还是等待更高级的版本发布。另一位用户鼓励分享，并指出虽然即将推出的改进版本正在开发中，但距离生产就绪还需要一段时间。

- **Phi-3 令人印象深刻的基准测试**：一位用户在发现 Phi-3 在 Ger-RAG-eval 上几乎获得满分后表示兴奋，并向社区寻求指导，了解如何像基准测试评估那样，在实时推理中应用该模型的“分类模式”。

- **DiscoLM-70b 的技术困难**：一位成员报告称，尽管运行过程成功，但在尝试使用 `huggingface/text-generation-inference` 端点与 DiscoLM-70b 交互时，遇到了 "Template not found" 错误和无意义的输出。
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1232708130018951209)** (7 条消息): 

- **Python API 疑难解决**：一位成员询问了 **Datasette's LLM** 的 [Python API 文档](https://llm.datasette.io/en/stable/python-api.html)，以便通过编程方式完成诸如对文本文件目录进行 Embedding（嵌入）等任务。另一位用户提供了关于如何利用 Python API 的详细说明链接。
 
- **Claude 通过 LLM CLI 结合 Hacker News**：**Simon Willison** 讨论了将 Claude 与 [LLM CLI 工具](https://llm.datasette.io/) 结合使用来总结 Hacker News 上的长帖子。他详细阐述了他的处理过程，并提供了[他的工作流链接](https://til.simonwillison.net/llms/claude-hacker-news-themes)，该工作流使用了带有 LLM 插件的 LLM CLI。

- **探索 Embedding API 的用法**：对于通过 Python API 对文本文件进行 Embedding，对话指向了 [LLM 的 Embedding API 文档](https://llm.datasette.io/en/stable/embeddings/python-api.html)，其中包含代码片段和高效处理 Embedding 的使用示例，包括用于处理多个字符串的 `embed_multi()`。

- **CLI Embedding 功能的 Python 等效实现**：创建者 **Simon Willison** 澄清说，目前没有直接对应“嵌入此目录中的每个文件”这一 CLI 功能的 Python 等效实现，但他分享了 [GitHub repo](https://github.com/simonw/llm/blob/04915e95f8ab490d6a82900ccba92ad3fe93d2ab/llm/cli.py#L1217-L1360) 中的相关部分，从中可以看到 CLI 是如何使用 Python API 实现该功能的。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://til.simonwillison.net/llms/claude-hacker-news-themes">使用 Claude 和 LLM 总结 Hacker News 讨论主题</a>：我一直在尝试将 Claude 与我的 LLM CLI 工具结合使用，以便快速总结 Hacker News 上的长篇讨论。</li><li><a href="https://llm.datasette.io/en/stable/python-api.html">Python API - LLM</a>：未找到描述</li><li><a href="https://llm.datasette.io/en/stable/embeddings/python-api.html">在 Python 中使用 embeddings - LLM</a>：未找到描述</li><li><a href="https://github.com/simonw/llm/blob/04915e95f8ab490d6a82900ccba92ad3fe93d2ab/llm/cli.py#L1217-L1360">simonw/llm 中的 llm/llm/cli.py</a>：从命令行访问大语言模型 - simonw/llm
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/)** (1 条消息): 

burnytech: Hi!
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1232960311133343764)** (2 条消息): 

- **多伦多 AI 开发者见面会**：*Ollamas and friends* 正在多伦多的 Cohere 空间举办一场本地开源 AI 开发者见面会，提供食物，并有社交和闪电演讲（lightning talks）的机会。由于名额有限，建议感兴趣的人员尽早通过此 [多伦多 AI 见面会注册链接](https://lu.ma/devs5) 进行报名。

**提到的链接**：<a href="https://lu.ma/devs5">多伦多本地与开源 AI 开发者见面会 · Luma</a>：本地与开源 AI 开发者见面会即将来到多伦多！在 Cohere 空间加入 Ollamas 及其朋友们！特别感谢 abetlen (Andrei) 和……

  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/)** (1 条消息): 

jeffreyw128: https://twitter.com/wangzjeff/status/1783215017586012566
  

---


**LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1232810178601291919)** (1 条消息): 

- **不同的推理提供商实现**：一位成员询问了为什么像 **Octo AI** 和 **Anyscale** 这样的推理提供商，针对相同的开源模型会有不同的 JSON mode 实现。他们注意到 Octo AI 的方法类似于需要 schema 的函数调用，而 Anyscale 则更接近 OpenAI 的实现。
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1232696364950093924)** (1 条消息): 

- **关于上下文利用的问题**：一位成员提到，他们不清楚该工具如何有效地利用可用的完整上下文。不过，他们表示其表现优于 GPT，暗示其有效性在某种程度上有所提高。
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 条消息): 

neilbert.: 恭喜！你现在是 Laurie Anderson 了！
  

---



**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1232824689458413680)** (1 条消息): 

- **分享了不当链接**：一位成员发布了一个推广 **Onlyfans 泄露和每日青少年内容**的链接，这显然不适合该 Discord 社区，并配有暗示成人内容的表情符号。该消息包含一个 Discord 邀请链接，并标记了所有成员以引起注意。

**提到的链接**：<a href="https://discord.gg/C42uKDFV">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。