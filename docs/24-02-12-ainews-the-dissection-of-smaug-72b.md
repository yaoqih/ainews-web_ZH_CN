---
companies:
- abacus-ai
- hugging-face
- nous-research
- laion
- thebloke
- lm-studio
- intel
- nvidia
- elevenlabs
date: '2024-02-13T01:40:29.456403Z'
description: '**Abacus AI** 推出了 **Smaug 72B**，这是 **Qwen 1.0** 的一个大型微调版本。尽管受到 **Nous
  Research** 的质疑，该模型在 **Hugging Face 开源大模型排行榜 (Open LLM Leaderboard)** 上依然占据榜首。**LAION**
  推出了一款名为 **Bud-E** 的本地语音助手模型，并发布了令人瞩目的演示。


  **TheBloke Discord** 社区讨论了 **GPT-4** 等大模型与较小量化模型之间的性能权衡、使用 **WizardLM_evol_instruct_V2_196k**
  和 **OpenHermes-2.5** 等数据集的微调技术，以及涉及 **Mistral-7b** 和 **MiquMaid** 的 Web UI 开发和模型合并挑战。**LM
  Studio Discord** 则重点讨论了从 PyTorch 到 gguf 的模型转换问题、涉及 **Intel Xeon CPU** 和 **Nvidia
  P40 GPU** 的硬件配置、隐私忧虑，以及图像生成和 Web UI 可用性方面的局限性。'
id: 28549af9-0301-4f3f-a4d8-369a8359538c
models:
- smaug-72b
- qwen-1.0
- qwen-1.5
- gpt-4
- mistral-7b
- miqumaid
- wizardlm_evol_instruct_v2_196k
- openhermes-2.5
original_slug: ainews-to-be-named-3759
people:
- bindureddy
title: '**Smaug (72B) 的剖析** 或 **对 Smaug (72B) 的深度解析**'
topics:
- fine-tuning
- model-merging
- quantization
- web-ui
- model-conversion
- hardware-setup
- privacy
- image-generation
- optical-character-recognition
- prompt-engineering
---

 

在其他新闻中，LAION 推出了一个[名字很可爱的本地语音助手模型](https://laion.ai/blog/bud-e/)，并附带了一个很棒的演示。

---

**目录**

[TOC] 


# 第 1 部分：Discord 高层级摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **模型性能中规模至关重要**：社区成员讨论了 GPT-4 等大模型的成本与性能权衡，并探索了像 Janitor AI 这样具有成本效益的聊天机器人替代方案。还讨论了在高水平量化下的较小模型的潜在有效性，但共识倾向于认为大模型能更好地处理重度量化，尽管这种关系并非严格线性。

- **好 AI 变成了“乖乖牌”**：对话中提到了以安全为核心的 AI 模型 "Goody-2"，指出其对争议性内容的极端谨慎，这引发了一些关于挑战该模型严苛程度的有趣想法。

- **Web UI 的烦恼与收获**：协作讨论展示了 PolyMind 等 UI 开发项目的进展，成员们讨论了 Web 开发和 Prompt Engineering（提示词工程）的复杂性。

- **缺失的模型与合并之谜**：出现了对用于合并项目的微调版 Mistral-7b 模型的查询，同时还有人幽默地感叹某位成员的缺席（就像在等待更新一样），并将其比作对 Llama3 支持视觉能力的期待。

- **模型合并大显身手**：社区内对 Miquella 等模型合并以及 ERP 导向模型（如 MiquMaid）的潜在性能进行了热烈交流，展示了将这些 AI 模型针对特定任务进行微调的热情；同时，成员们也在寻求关于设置配置的建议，例如上下文长度和跨 GPU 的显存分配。

- **注入指令性**：对微调方法的兴趣显而易见，讨论围绕利用 WizardLM_evol_instruct_V2_196k 和 OpenHermes-2.5 等数据集将基础模型（Base Models）转换为指令模型（Instruct Models）。这一过程丰富了模型的知识并改变了聊天机器人的语气，成员们推荐了 [unsloth](https://github.com/unslothai/unsloth) 等资源以实现高效微调。

- **编程角落的协作**：分享的一个用于 Surya OCR 的 Python 脚本暗示了成员之间在光学字符识别工具方面的持续开发和应用。有人为 webaudiobook 仓库寻求调试帮助，其中一个可能的服务问题被指出与 ElevenLabs 平台的性能表现有关。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **模型转换故障**：有人指出在运行 TheBloke 的模型时出现了错误，这很可能是由于 PyTorch 到 gguf 转换过程中量化损坏导致的。用户还交流了高效运行 Large Language Models 的硬件组件和配置经验，提到了 Intel Xeon CPU、Nvidia P40 GPU 以及 VRAM 和 CUDA 核心的重要性。

- **LM Studio 的技术底层**：LM Studio 社区的讨论澄清了 LM Studio 使用了 llama.cpp，可能还使用了 Electron 作为其技术栈。虽然存在对 LM Studio 数据使用的隐私担忧，但官方表示该平台优先考虑隐私，仅共享来自更新和模型下载的数据。

- **图像生成限制**：用户分享了在 LM Studio 中进行图像生成的困难，促使一些人考虑使用 VPN 来绕过影响访问 huggingface.co 等必要资源的 ISP 封锁。

- **Web UI 困扰**：社区寻求但确认了 LM Studio 缺乏 Web UI，这为部分用户增加了一层复杂性。

- **小模型分类挑战**：用户 `'@speedy.dev'` 在使用 13B 和 7B Llama 模型进行分类任务时面临挑战。对不同规模模型的故事写作能力对比显示，Goliath 120B 能更好地捕捉情感，而 Mixtral 7B 在速度上表现更优。

- **元数据管理与 `model.json` 标准提案**：强调了妥善管理模型元数据的价值，建议为不同参数建立分类系统，并在 GitHub 上发布了 `model.json` 标准提案。

- **英特尔 AVX-512 决策令人困惑**：英特尔选择在强大的 i9 处理器中取消 AVX-512 支持，却在性能较弱的 i5-11600k 中保留该支持，这引起了用户的困惑。

- **相比 AutoGen 更倾向于 CrewAI**：简要提到了由于易于管理方面的考虑，用户更倾向于使用 CrewAI 而非 AutoGen。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **HF Hub 暂时离线**：[@lunarflu](https://discord.com/channels/879548962464493619/897387888663232554/1205945710164574279) 报告称 **Hugging Face Hub** 出现宕机，git 托管受到影响。该问题正在处理中。

- **Discord 身份验证与 API 变更**：在 Discord 验证方面，用户 @d3113 通过启用“允许来自服务器成员的消息”解决了机器人身份验证错误。同时，[@miker1662](https://discord.com/channels/879548962464493619/879548962464493622/1205799064541593631) 遇到了一个与在 Hugging Face 上查找对话模型相关的 API bug，@osanseviero 确认 API 正在从 `conversational` 转向 `text-generation`。

- **LLM 焦点**：讨论内容包括 **PEFT 用于 LLM 微调**与全量微调方法的对比，@samuelcorsan 选择了后者；而 @akvnn 询问了如何利用 Raspberry Pi 执行 computer vision 任务，@pradeep1148 分享了一个关于 [zero shot object detection 的 YouTube 视频](https://www.youtube.com/watch?v=W4T7zHluzaM)。

- **本地 LLM 的硬件选择**：@lookingforspeed 和 @tddammo 讨论了选择双 NVIDIA 4080 Super 还是单块 4090 来运行编程 LLM，建议使用 A4500 或 A4000 等旧代专业卡以获得更好的效率和 NVLink 支持。

- **API 调用库选择与创新**：[@subham5089](https://discord.com/channels/879548962464493619/898619964095860757/1205807574042148894) 分享了关于选择合适的 Python 库进行 API 调用的见解，重点介绍了 **Requests, AIOHTTP 和 HTTPX**。针对 Python 中的 Websockets 问题，推荐使用 [httpx-ws](https://frankie567.github.io/httpx-ws/)。

- **Computer Vision 与 NLP 的进展**：从讨论使用 `Dinov2ForImageClassification` 等工具简化多标签图像分类，到 NLP 相关问题（如降级到 **PEFT 0.7.1** 以保存 LoRA weights），社区参与充满了解决方案分享和知识交流。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **AI 财务建议需谨慎**：讨论了 AI 在生成 **trading strategies**（交易策略）方面的局限性，由于 AI 在处理财务查询时的谨慎回应和内存限制，讨论中带有一种挫败感。

- **AI 驱动压缩技术的创新**：分享了一个名为 `ts_zip` 的工具，它利用 **Large Language Models** 进行文本文件压缩，引发了对 AI 驱动压缩技术的兴趣。该工具可在 [ts_zip: Text Compression using Large Language Models](https://bellard.org/ts_server/ts_zip.html) 查看。

- **关于 AI 资源提取及其影响的讨论**：提到为 **AI chips** 投资 **5-7 trillion**（5-7 万亿）美元引发了关于对 AI 研发影响以及与自主机器人相关的社会风险的辩论。

- **Google 的 Gemini AI 备受关注**：讨论了 Google 的 **Gemini** 及其预期的改进和当前的优缺点，其代码编写和解释器功能吸引了关注。

- **探索 ChatGPT Token 上下文限制**：探讨了 **ChatGPT's attention span**（注意力跨度）和上下文保留问题，指出完整的 128K token 上下文仅对 API 和 Enterprise 用户开放，而 Plus 和 Team 用户则限制在 32K token。

- **GPT-4 订阅详情澄清**：对订阅共享进行了澄清，指出现在所有 GPT 版本都使用 GPT-4，因此使用时需要 Plus 或更高版本的订阅。

- **驾驭 GPT-4 的对话流和状态感知**：用户讨论了 GPT-4 对“状态变化”的处理，并注意到它在管理动态条件和对话流方面的有效性，这在提供需要多步骤解决问题的提示词时非常重要。

- **分享有效的 Prompt Engineering 策略**：在 #prompt-engineering 频道中，讨论了指导 **ChatGPT** 执行复杂任务的策略，例如将证据规则转换为电子表格以及将文本翻译成简单的语言。简化任务和使用逐步提示是推荐的一些方法。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **Elon Musk 所谓的 AI 趣闻**：Elon Musk 被幽默地与两个独立的 AI 相关事件联系在一起：一个是传闻他与 Alex Jones 通话，另一个是他吹嘘用 200 个 GPU 创建了一个 AI。`@fullstack6209` 分享了一个片段，可以在[这里](https://twitter.com/i/status/1756105704786817130)找到，并引用了 Musk 在 2024 年 2 月 9 日的评论。此外，`@gabriel_syme` 报告称 Hugging Face (HF) 处于离线状态。

- **AI 中的量化与合并创新**：Senku 70B 模型在未校准的情况下量化为 4-bit 取得了显著的分数，`@carsonpoole` 建议混合格式以获得更好的效果。同时，`@nonameusr` 介绍了 QuartetAnemoi-70B 模型，`@.benxh` 讨论了 1.5-bit 量化的潜力，它可以将 70b 模型放入小于 18GB 的 RAM 中。QuartetAnemoi-70B 模型链接在[这里](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001)，1.5-bit 量化链接在[这里](https://github.com/ggerganov/llama.cpp/pull/5453)。

- **丰富的数据集和模型**：工程社区讨论了众多模型和数据集：UNA-SimpleSmaug-34B 表现出优于 Smaug-34B 的改进，并在 [SimpleMath](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta) 上进行了训练。Lilac 处理的 OpenHermes-2.5 数据集可在[这里](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5)获取。还提到了使用 [mergekit](https://github.com/cg123/mergekit) 的 TheProfessor-155b 模型。

- **关于微调和托管 AI 模型的社区交流**：`@natefyi_30842` 寻找简单的微调方法，`@teknium` 推荐了 together.ai 并指出需要进行超参数微调。`@nonameusr` 寻求在 Hugging Face 上托管模型以进行 API 推理的建议。

- **关于自主 LLM 和平台功能的讨论**：`@0xsingletonly` 表达了对自主 **Large Language Models** (LLM) 的兴趣，并期待 `@teknium` 提到的 Nous 的 SDK 开发。此外，澄清了关于路线图中功能包含的困惑，`@qnguyen3` 指出了 Llava 团队的独立集成工作。

- **幽默与服务宕机**：其他值得注意的提到包括 `@.ben.com` 幽默地表示不愿接受缺乏表情符号专业知识的人的建议，以及 `@gabriel_syme` 注意到的 Hugging Face 服务的短暂中断。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **AI 新手挑战 EleutherAI 及更多领域**：一位具有软件开发和研究论文背景的成员表现出对贡献 AI 和 GPT-J 的兴趣，同时讨论了 The Pile 数据集和 CUDA 编程。社区探讨了 Prodigy 占用大量 VRAM 的性能表现、AI 合并实践，并对日益增长的有问题的 AI 做法表示担忧，此外还提到了 OpenAI 可能配合超级碗（Super Bowl）发布新产品。相关资源包括 [Prodigy 优化器](https://github.com/konstmish/prodigy)、[Hugging Face 上的 miqu-1-120b](https://huggingface.co/wolfram/miqu-1-120b) 以及 [微软 Copilot 广告](https://youtu.be/SaCVSUbYpVc?t=40)。

- **嵌套网络与向量困惑**：成员们对 JAX 中的嵌套网络（nested networks）表现出极大热情，并分享了用于实验的 [Colab 笔记本](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq)等有用资源。讨论还深入探讨了数学中向量方向的混淆，同时为实现扩散模型（diffusion）论文的方法提供了帮助，并在 [nbviewer gist](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57) 中分享了代码。如需进一步阅读，可通过 [GitHub](https://github.com/UniReps/UniReps-resources) 获取 UniReps 研究的汇总。

- **辩论机器遗忘基准测试的价值**：针对 [TOFU 论文](https://arxiv.org/abs/2401.06121)中详述的用于擦除敏感数据的 “TOFU” 基准测试的意义，人们产生了怀疑。参与者对其有效性和实际应用提出了担忧，并讨论了一篇相关的[神经元剪枝（neuron pruning）论文](https://arxiv.org/abs/2401.01814)，这可能为对话提供启发。

- **模型评估与幻觉追踪**：关于 MedQA 基准测试的问题表明，Pythia 模型可能在多项选择题格式上表现不佳。讨论强调了对涵盖 OpenAI 到 Anthropic 以及 Open LLM Leaderboard 的比较模型 API 数据的搜索。对于涉及 GPQA 数据集的任务，提醒注意潜在的数据泄露，并寻求手动下载 [GPQA 数据集论文](https://arxiv.org/abs/2311.12022)。此外，还请求澄清如何使用 GPT-4 模型评估 Big Bench Hard 任务。最后，号召大家参与 [Hugging Face 博客文章](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations)中解释的新幻觉排行榜（hallucinations leaderboard）以及相关的 [Harness 空间](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **语音助手中集成化优于级联化**：有观点认为，与语音 AI 的端到端训练相比，**级联 ASR + LLM + TTS** 的效果稍逊一筹。**BUD-E 语音助手**被作为示例展示，它在一个 PyTorch 模型中集成了 ASR、LLM 和 TTS，提升了端到端的可训练性（[了解更多关于 BUD-E 的信息](https://speechbot.github.io/spiritlm/index.html)）。

- **AI 艺术创作者的法律纠纷**：最近的一项法院裁决中，*美国地方法院法官 William Orrick* 驳回了 Midjourney 和 StabilityAI 根据第一修正案辩护提出的早期即决判决动议，这引发了用户对该案更广泛影响的讨论。

- **AI 社区应对开源伦理问题**：AI 社区讨论了 `sd-forge` 项目，该项目结合了 *diffusers*、*automatic1111* 和 *comfyui* 的代码，但在 Stable Diffusion 模型及其开源 UI 对应物不断演变的格局中，试图与这些项目保持距离。

- **创意前沿：AI 生成的 DnD 地图**：用户成功利用神经网络创建了《龙与地下城》（Dungeons and Dragons）地图，反映了 AI 在创意领域不断扩展的能力。

- **Hugging Face 面临障碍**：有报告称 **Hugging Face** 的服务出现停机。对话集中在依赖外部 API 的挑战，以及为了维持 AI 开发顺利运行对稳健替代方案的需求。

- **开放语音的演进**：**BUD-E** 作为一款开源、低延迟且旨在离线运行的语音助手被推出，并邀请社区为其进一步开发做出贡献（[为 BUD-E 贡献代码](https://laion.ai/blog/bud-e/)，[加入 Discord](https://discord.com/invite/MDTpftKbpv)）。

- **AI 中的损失函数科学**：有人询问了 Stability AI 其中一个项目中的 Wasserstein 损失，并提供了 GitHub 仓库链接，尽管目前尚未发现与该主张直接相关的代码（[判别器损失之谜](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py)）。

- **全栈人才与科学见解**：一位用户展示了其全栈设计和开发技能，而另一位用户分享了一篇科学文章但未提供额外背景。此外，还有人请求关于复现 **MAGVIT V2 模型**的指导，这表明社区内正进行着活跃的研究和开发工作（[Shinobi 作品集](https://shinobi8894.onrender.com/)，[查看 MAGVIT V2](https://github.com/lucidrains/magvit2-pytorch)）。

- **介绍 Agent 基础模型**：社区关注到了 arXiv 上一篇关于“**交互式 Agent 基础模型**”的论文，暗示了 AI 系统正向动态、基于 Agent 的方向转变（[阅读摘要](https://arxiv.org/abs/2402.05929)）。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **AI 正面对比**：用户讨论了通过打开多个标签页来**对比不同的 AI 模型**；其中一个对比网站是 [AIswers](https://www.aiswers.com/)，可以在该网站上测试 **Perplexity** 与其他模型的性能。

- **App 交互过于敏感**：**Perplexity 的 iPad 应用**因线程退出功能过于敏感而受到批评。此外，关于 **Perplexity API 开发者频道**的咨询被引导至现有的 Discord 频道。

- **API 速率限制的怪癖**：一些用户在使用 App 脚本调用 Perplexity API 时遇到了 **429 HTTP 状态错误**，最初误以为是 OpenAI 相关的问题。该问题通过在脚本中**添加毫秒级延迟**得以解决；Perplexity 上的额度和限制可能与 OpenAI 不同。

- **模型功能与函数查询**：有用户请求更新 [功能路线图](https://docs.perplexity.ai/docs/feature-roadmap) 中 **Mistral 32k 上下文长度的可用性**情况，并澄清了 `mistral-8x7b-instruct` 的所有输入都需要 **messages 字段**，且目前不支持函数调用（function calling）。

- **确保搜索结果公开**：**Perplexity 用户被提醒**，在频道中分享使用 Perplexity 获得的显著结果之前，请确保线程已设置为公开可见。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **H100 GPU 作为通往 AGI 的阶梯**：`@andreaskoepf` 讨论了新型 **H100** GPU 在结合适当模型和足够数量的 GPU 时实现通用人工智能 (AGI) 的潜力，并引用了一篇 [AI Impacts 文章](https://aiimpacts.org/brain-performance-in-flops/)，其中包含对人类大脑活动的 FLOPS 估算。

- **向斯坦福 AI 硬件专家学习**：社区重点推荐了 [Benjamin Spector 的 Stanford MLSys 研讨会](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86)，该讲座面向工程师，提供了关于 AI 硬件的见解，可能与工程论坛上的讨论相关。

- **Serverless Triton Kernel 执行**：`@tfsingh` 宣布推出 [Accelerated Computing Online](https://acceleratedcomputingonline.com)，这是一个在 T4 GPU 上执行 Triton kernel 的 Serverless 环境，并提到了该项目的 GitHub 仓库 ([tfsingh/aconline](https://github.com/tfsingh/aconline)) 以供进一步探索。

- **CUDA 开发深度解析**：讨论集中在 CUDA 编程，涉及用于性能增强的 memory coalescing（内存合并）、用于图像操作的 NVIDIA NPP、fp16 matmuls（矩阵乘法）中数值稳定性的细微差别，以及独立开发 CUDA 兼容扩展的最佳实践。

- **Multi-GPU 故障排除与 FAISS 挑战**：`@morgangiraud` 在分布式矩阵乘法中进行直接的 device-to-device tensor 复制时遇到了数据错误问题，并寻求拥有 multi-GPU 设置的合作者进行验证；而 `@akshay_1` 则在处理 FAISS (colbert) 中的向量嵌入错误，这可能源于分布式 worker 的超时。

- **CUDA MODE 讲座课程与公告**：即将举行和过去的教育活动，如“CUDA MODE 第 5 讲：为 Python 程序员进一步深入 CUDA”，引发了广泛关注，并在 [Discord](https://discord.gg/6UQXQYZp) 等平台上分享了链接，以促进社区参与和学习。

- **参与教育内容**：社区成员（特别是 `@smexy3`）对未来的教学视频内容表现出极大的热情，尤其是那些教授如何分析报告中优化机会的视频，下一部视频计划于 **3 月 2 日**发布。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **Awq Gguff 转换器：极速体验**：用户称赞 **awq gguff converter** 性能极快，称其为“10/10 快速”的转换工具，但未提供更多细节或链接。

- **HuggingFace 故障引发社区解决方案**：在一次影响到本地训练任务的 HuggingFace 故障期间，成员们讨论了变通方案，包括降级到 **0.7.1 版本**，并考虑使用 **TensorRT** 等替代推理方案进行本地推理。

- **通过 Peft 升级解决 Mixtral 难题**：通过将 **peft 版本从 0.7.1** 升级到 **0.8.0**，解决了 **Mixtral 的量化过程**问题，并确认升级修复了初始问题。文中提到了 LlamaFactory 对 ShareGPT 格式的采用，并就命名规范进行了讨论，但未达成进一步结论。

- **聚焦 Fine-Tuning 技术与效率**：社区交流了 fine-tuning 策略的技巧，包括从历史数据集中为聊天模型生成问答对，以及寻求使用量化 Mixtral 等具有成本效益的方法。此外，还分享了 Mistral-7b-instruct 训练配置的实用见解，并引用了来自 **Helix** 和 **axolotl GitHub 仓库**的配置文件。

- **新手 Fine-Tuning 资源请求未获回应**：消息记录中关于 fine-tuning 学习资源的请求未得到回应，这凸显了社区支持和知识共享的一个潜在领域。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **LLM 掌握表格遍历**：`@jerryjliu0` 发布的一个[新视频教程](https://www.youtube.com/watch?v=L1o1VPVfbb0)强调了高级 **text-to-SQL 编排**，这对于使用语言模型导航和查询表格数据至关重要。

- **增强对表格数据的理解**：最近分享了关于 RAG 系统的进展，重点强调了多跳查询能力，详见 Tang 等人用于[基准测试高级 RAG 模型](https://t.co/Hqx1KKOqYv)的数据集。与此同时，还有一个[迷你课程](https://t.co/BS0VkZjbZI)，涵盖了构建融合 text-to-SQL 与 RAG 的查询流水线，增强了表格数据问答框架。

- **创新视频内容交互**：一种协同 OpenAI GPT4V 与 LanceDB VectorStore 的**多模态 RAG 架构**正在增强视频内容交互。*Video Revolution: GPT4V and LlamaIndex Unleashed* 讨论了这一创新及其潜力，是该领域感兴趣者的必读之作，点击[此处](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30)阅读。

- **AI 上下文管理的探索与解决方案**：LlamaIndex 社区成员讨论了实际应用，例如使用 LlamaIndex 生成 SQL 元数据，以及需要像 `SimpleChatStore` 这样的解决方案来跨多个窗口保持对话连续性。关于从文本中提取关键词的解决方案，建议涉及 prompt engineering。

- **价格与可用性说明**：关于 LlamaIndex 是否免费开源及其可用性的问题得到了澄清，它确实是开源的，更多详情可在其[官方网站](https://www.llamaindex.ai/)查看。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

- **Scktlearn 难题寻求语音支持**：`@vithan` 寻求关于 **scktlearn 和 pandas** 的帮助，表示文字沟通存在局限性，并请求与 `@himalaypatel` 进行语音通话以进行更有效的排错。

- **LangChain 视频教程发布**：`@a404.eth` 分享了一个 YouTube 教程 "Unlock the Power of LangChain: Deploying to Production Made Easy"，详细介绍了使用 LangChain 和 **UnstructuredIO** 将 PDF RAG 部署到 **DigitalOcean** 以供生产使用的过程。视频可通过[此链接](https://youtu.be/CbBIwVxjdP8)观看。

- **开源项目 Selfie 需要你的照片**：`@dondo.eth` 介绍了 **Selfie**，这是一个开源项目，致力于通过 OpenAI 兼容 API 利用个人数据来改进文本生成，欢迎在其 [GitHub](https://github.com/vana-com/selfie) 上贡献和测试。

- **Intellifs 设定标准**：`@synacktra` 宣布创建 **Intellifs**，这是一个基于 aifs 库的本地语义搜索工具，目前在 [GitHub](https://github.com/synacktraa/intellifs) 上开放贡献。

- **你的艺术，AI 的触碰**：`@vansh12344` 推出了 **ArtFul - AI Image Generator**，这款应用使用 Kandinsky 和 DALL-E 等 AI 模型创作独特的艺术作品，支持广告免费使用，可在 [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful) 下载。

- **Merlinn 在故障解决中的神奇辅助**：`@david1542` 展示了 **Merlinn**，旨在通过 LLM Agent 的支持和 LangChain 集成，帮助团队快速解决生产故障。更多详情请访问 [Merlinn 网站](https://merlinn.co/)。

- **Triform 招募 Beta 测试人员**：`@igxot` 宣布了 **Triform**，这是一个用于托管和编排集成 LangChain 的 Python 脚本的新平台。邀请用户通过参与 Beta 测试获取永久免费账户，注册链接在[此处](https://triform.ai)，文档在[此处](https://triform-docs.readthedocs.io/)。

- **自动目标检测变得简单**：`@pradeep1148` 分享了一个关于使用 MoonDream Vision Language Model 进行零样本目标检测的 [YouTube 教程](https://www.youtube.com/watch?v=W4T7zHluzaM)。

- **使用 AI 工具与文档对话**：`@datasciencebasics` 发布了一个[视频指南](https://youtu.be/2IL0Sd3neWc)，解释了如何使用 ChainLit, LangChain, Ollama, & Mistral 创建 Retrieval Augmented Generation UI。
  
- **在生产环境中禁用 Playground**：`@gitmaxd` 讨论了使用特定代码片段在已部署的 LangChain AI 端点上**禁用 playground** 的可能性，但该询问未收到回复。



---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

**Mistral 订阅模式的“一站式”方案**：用户讨论了 Mistral Discord 聊天机器人的订阅模式，@mrdragonfox 确认这是一个统一的模型，按 token 计费并支持可扩展部署；此外还提到了量化模型（例如在 [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) 上找到的模型），这类模型对 RAM 的需求较低。

**GPU 经济学：租用 vs. 自有**：@i_am_dom 分析了租用 Google GPU 与拥有 **A100s 40GB** 等硬件的成本效益，建议在消耗 70000 个计算单位或使用约半年后，拥有自己的 GPU 可能会更经济。

**Docker 部署讨论**：关于部署 Mistral AI 的 `docker_compose.yml` 请求表明，社区正在讨论如何在 Docker 环境中将 Mistral AI 设置为 REST APIs 以简化流程。

**针对自我意识和个人助手的微调**：微调话题涵盖了从在 Cloudflare AI maker 上成功安装，到模型缺乏自我意识的问题（由 @dawn.dusk 针对 GPT-4 和 Mistral 提出）；推荐通过 [Datacamp 教程](https://www.datacamp.com/tutorial/mistral-7b-tutorial) 学习使用案例和 Prompt。

**展示 Mistral 的能力**：@jakobdylanc 开发的 Discord 聊天机器人具有协作式 LLM 提示功能，支持包括 Mistral 在内的多个模型，实现代码仅 200 行，可在 [GitHub](https://github.com/jakobdylanc/discord-llm-chatbot) 获取；此外，[Hacker Noon](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk) 的一篇文章强调了 Mistral 7b 在记笔记方面的卓越能力，其表现优于一些评分更高的模型。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **TurboPuffer 在 S3 上表现出色**：讨论了一个名为 [TurboPuffer 的新型 Serverless 向量数据库](https://turbopuffer.com/) 的效率，强调 100 万个向量的热查询缓存大约需要 10 秒。对话对比了 TurboPuffer 和 LanceDb，指出 TurboPuffer 利用了 S3，而 LanceDb 则因其开源特性受到青睐。

- **播客探讨 AI 与集体智慧**：分享了在 [Cognitive Revolution 播客](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/) 中对 Yohei Nakajima 的采访，讨论了集体智慧以及 AI 在促进跨文化理解方面的作用。

- **AI 是 Google 的阿喀琉斯之踵**：通过 [TechEmails](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww) 分享的一份 2018 年 Google 内部备忘录显示，该公司当时就将 AI 视为重大的业务风险，这一担忧在多年后依然具有现实意义。

- **ChatGPT 对大学申请流程的影响**：分析了在大学申请中使用 ChatGPT 的趋势，引用了 [Forbes 文章](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/)，该文章指出了一些可能出现的警示信号（red flags），例如使用特定的禁用词会引起招生委员会的警觉。

- **通过禁用词避免学术警报**：有人建议为 ChatGPT 编程一个禁用词列表，以防止其在学术场景中被滥用，这与关于大学录取以及通过此类词汇检测 AI 过度使用的讨论相呼应。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **Hugging Face 服务中断引发社区热议**：由于 **Hugging Face** 出现宕机，社区成员（如 `_jp1_`）展开了讨论。他们意识到该平台已成为不可或缺的一部分，并透露过去曾考虑切换到 **Amazon S3** 来托管模型权重和数据集，但 HF 免费服务的便利性最终占据了上风。`_jp1_` 和 `@philipmay` 还思考了 HF 的长期可持续性，对未来可能开启的货币化及其对 AI 研究社区的影响表示担忧。
  
- **关于 HF 作为关键基础设施的思考**：由 `@philipmay` 发起的辩论质疑 **Hugging Face** 是否符合 AI 社区关键基础设施的定义，强调了外部平台在维持模型运行方面已变得多么至关重要。
  
- **灵活货币化方案的前景**：`@philipmay` 推测了 **Hugging Face** 可能开始对模型访问或下载收费的情景，这引发了社区内进行预先财务规划的需求。

- **关于稀疏效率的传闻**：`@phantine` 在没有提供细节的情况下，暗示了一种利用稀疏性（sparsity）提高效率的算法，并提供了一个用于获取更多详情的链接，但该链接失效了。

- **在德语语言模型中使用 SPIN**：`@philipmay` 提到了将 **SPIN 方法** (self-play) 应用于德语的 **Mixtral 模型**，并分享了 [SPIN 技术的官方 GitHub 仓库](https://github.com/uclaml/SPIN) 以激发进一步的讨论或实验。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **OpenAI 即将发布新消息的传闻**：`@res6969` 暗示了潜在的 **OpenAI 发布** 活动，并发布了一条模糊的公告，预计在**明天或周二**会有消息，引发了大家的期待。`@.psychickoala` 俏皮地询问 **“是什么呢 哈哈”**，但未有具体细节流出。

请注意，来自 rabiat 的另一条消息由于缺乏足够的上下文或与技术细节导向的工程师受众相关的信息，因此未包含在摘要中。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **对同事近况的好奇**：@teknium 询问了 `<@748528982034612226>` 最近在忙什么。
- **神秘成员的状态更新**：@atlasunified 告知 `<@748528982034612226>` 已经**处于离线状态 (off grid)**，未对其状态做进一步说明。

---

# 第二部分：分频道详细摘要与链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1205802028987715644) (1251 条消息 🔥🔥🔥): 

- **关于模型尺寸和偏好的担忧**：`@dao_li` 等成员分享了他们使用各种 AI 模型的经验，讨论了 GPT-4 的成本和效果，以及用于聊天机器人的 Janitor AI 等替代方案。由于他们发现 GPT-4 价格昂贵，其他用户建议尝试各种小模型以获得更具成本效益的解决方案。
- **关于量化的讨论**：`@immortalrobot` 询问了低量化的大模型与高量化的小模型之间的权衡。包括 `@kalomaze` 和 `@superking__` 在内的共识似乎是，较大的模型可能更好地处理重度量化，但这种关系并非线性的。
- **关于 "Goody-2" 的笑话**：讨论涉及了安全 AI 模型 "Goody-2"，用户 `@selea` 评论了它的严苛性，因为它会拒绝任何可能引起争议的内容。对话幽默地探讨了挑战该模型的想法。
- **用户界面开发**：`@itsme9316` 和 `@potatooff` 分享了他们各自在 PolyMind 和正在构建的新 UI 项目上的进展。他们讨论了 Web 开发和 Prompt Engineering 的复杂性与挑战。
- **对模型缺失和更新的好奇**：`@rombodawg` 为一个合并项目寻找微调过的 Mistral-7b 模型，而 `@kaltcit` 则幽默地评论了一位名为 turbca 的用户像猫一样的失踪，将其比作等待 Llama3 支持视觉功能。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1115976636400148562/1206157809042063380)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [无标题](https://www.independent.co.uk/news/world/americas/daniel-beckwitt-trial-askia-khafra-death-nuclear-bunker-fire-bethesda-maryland-a8963161.html)：未找到描述
- [GOODY-2 | 世界上最负责任的 AI 模型](https://www.goody2.ai/chat)：介绍一款具有下一代伦理对齐水平的新型 AI 模型。立即聊天。

- [Unbabel/TowerInstruct-13B-v0.1 · Hugging Face](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1): 未找到描述
- [Aligning LLMs with Direct Preference Optimization](https://www.youtube.com/watch?v=QXVCqtAZAn4): 在本次研讨会中，来自 Hugging Face 的 Lewis Tunstall 和 Edward Beeching 将讨论一种名为 Direct Preference Optimisation (DPO) 的强大对齐技术...
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2): 未找到描述
- [MrDragonFox/apple-ferret-13b-merged · Hugging Face](https://huggingface.co/MrDragonFox/apple-ferret-13b-merged): 未找到描述
- [Rick Astley - Never Gonna Give You Up (Official Music Video)](https://www.youtube.com/watch?v=dQw4w9WgXcQ>): Rick Astley 的《Never Gonna Give You Up》官方视频。新专辑《Are We There Yet?》现已发行：在此下载：https://RickAstley.lnk.to/AreWe...
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1): 未找到描述
- [Answer Overflow - Search all of Discord](https://www.answeroverflow.com): 使用 Answer Overflow 构建最佳的 Discord 支持服务器。将您的内容索引到 Google，使用 AI 回答问题，并深入了解您的社区。
- [GitHub - apple/ml-ferret](https://github.com/apple/ml-ferret): 通过在 GitHub 上创建账户，为 apple/ml-ferret 的开发做出贡献。
- [GitHub - daswer123/xtts-api-server: A simple FastAPI Server to run XTTSv2](https://github.com/daswer123/xtts-api-server): 一个用于运行 XTTSv2 的简单 FastAPI 服务器。通过在 GitHub 上创建账户，为 daswer123/xtts-api-server 的开发做出贡献。
- [GitHub - mzbac/mlx-llm-server: For inferring and serving local LLMs using the MLX framework](https://github.com/mzbac/mlx-llm-server): 用于使用 MLX 框架进行本地 LLM 的推理和提供服务 - mzbac/mlx-llm-server
- [GitHub - Tyrrrz/DiscordChatExporter: Exports Discord chat logs to a file](https://github.com/Tyrrrz/DiscordChatExporter): 将 Discord 聊天记录导出到文件。通过在 GitHub 上创建账户，为 Tyrrrz/DiscordChatExporter 的开发做出贡献。
- [metavoiceio/metavoice-1B-v0.1 · Hugging Face](https://huggingface.co/metavoiceio/metavoice-1B-v0.1): 未找到描述
- [nvidia/canary-1b · Hugging Face](https://huggingface.co/nvidia/canary-1b): 未找到描述
- [Piper Voice Samples](https://rhasspy.github.io/piper-samples/): 未找到描述
- [GitHub - metavoiceio/metavoice-src: Foundational model for human-like, expressive TTS](https://github.com/metavoiceio/metavoice-src): 用于类人、富有表现力的 TTS 的基础模型。通过在 GitHub 上创建账户，为 metavoiceio/metavoice-src 的开发做出贡献。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/111269072853191): Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112690728531918948): Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。
- [GitHub - daveshap/Reflective_Journaling_Tool: Use a customized version of ChatGPT for reflective journaling. No data saved for privacy reasons.](https://github.com/daveshap/Reflective_Journaling_Tool/tree/main): 使用定制版本的 ChatGPT 进行反思性日记。出于隐私原因不保存任何数据。 - GitHub - daveshap/Reflective_Journaling_Tool: 使用定制版本的 ChatGPT 进行反思性...
- [GitHub - LAION-AI/natural_voice_assistant](https://github.com/LAION-AI/natural_voice_assistant): 通过在 GitHub 上创建账户，为 LAION-AI/natural_voice_assistant 的开发做出贡献。
- [LoneStriker/HamSter-0.2-8.0bpw-h8-exl2 · Hugging Face](https://huggingface.co/LoneStriker/HamSter-0.2-8.0bpw-h8-exl2): 未找到描述
- [Simpsons Homer Simpson GIF - Simpsons Homer simpson - Discover &amp; Share GIFs](https://tenor.com/view/simpsons-homer-simpson-gif-13518564799186478937): 点击查看 GIF
- [GitHub - LostRuins/koboldcpp: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI](https://github.com/LostRuins/koboldcpp?tab=readme-ov-file#osx-and-linux-manual-compiling): 一种简单的单文件方式，通过 KoboldAI 的 UI 运行各种 GGML 和 GGUF 模型 - LostRuins/koboldcpp
- [Everything WRONG with LLM Benchmarks (ft. MMLU)!!!](https://www.youtube.com/watch?v=74Uo2HU8HBo): 🔗 链接 🔗 当基准测试成为目标：揭示大语言模型排行榜的敏感性 https://arxiv.org/pdf/2402.01781.pdf ❤️ 如果你想...

- [速查表：精通 ChatGPT API 中的 Temperature 和 Top_p](https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683)：大家好！好吧，我承认这得到了 OpenAI 的帮助。但我“协助”整理的内容，我认为可以极大地改善在应用和插件中使用 OpenAI 的效果和成本，特别是...
- [Andrew Garfield Andrew Garfield Moonlight Meme GIF - Andrew garfield Andrew Garfield Moonlight meme Andrew Garfield Moonlight trend - Discover &amp; Share GIFs](https://tenor.com/view/andrew-garfield-andrew-garfield-moonlight-meme-andrew-garfield-moonlight-trend-andrew-garfield-meme-gif-18154541527070698780)：点击查看 GIF
- [GitHub - jondurbin/airoboros: self-instruct 论文的可定制实现。](https://github.com/jondurbin/airoboros?tab=readme-ov-file#lmoe)：self-instruct 论文的可定制实现。 - jondurbin/airoboros
- [brucethemoose/Yi-34B-200K-RPMerge · Hugging Face](https://huggingface.co/brucethemoose/Yi-34B-200K-RPMerge)：未找到描述
- [Doctor-Shotgun/Nous-Capybara-limarpv3-34B · Hugging Face](https://huggingface.co/Doctor-Shotgun/Nous-Capybara-limarpv3-34B)：未找到描述
- [GitHub - itsme2417/PolyMind: 一个多模态、由 function calling 驱动的 LLM webui。](https://github.com/itsme2417/PolyMind)：一个多模态、由 function calling 驱动的 LLM webui。 - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - Haidra-Org/AI-Horde-Worker: 此仓库将你的 PC 变为 AI Horde 工作节点](https://github.com/Haidra-Org/AI-Horde-Worker)：此仓库将你的 PC 变为 AI Horde 工作节点 - Haidra-Org/AI-Horde-Worker
- [在 Apple Silicon 上调整 VRAM/RAM 分配 · ggerganov/llama.cpp · Discussion #2182](https://github.com/ggerganov/llama.cpp/discussions/2182)：// 此工具允许你根据需要更改 Apple Silicon 统一内存上的 VRAM/RAM 分配，从而为推理提供更多 VRAM // c++ -std=c++17 -framework CoreFoundation -o vra...
- [ikawrakow 的 k-quants · Pull Request #1684 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1684)：内容：此 PR 增加了一系列 2-6 bit 量化方法，以及量化混合，如 #1240 和 #1256 中所述。提供了 Scalar, AVX2, ARM_NEON 和 CUDA 实现。原因：这是...
- [nextai-team/apollo-v1-7b · Hugging Face](https://huggingface.co/nextai-team/apollo-v1-7b)：未找到描述
- [mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)：未找到描述
- [Intel/neural-chat-7b-v3-3 · Hugging Face](https://huggingface.co/Intel/neural-chat-7b-v3-3)：未找到描述
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/LocalLLaMA/comments/1ancmf2/yet_another_awesome_roleplaying_model_review/?rdt=58656)：未找到描述
- [Reddit - 深入探索任何事物](https://reddit.com/r/LocalLLaMA/comments/190pbtn/shoutout_to_a_great_rp_model/)：未找到描述

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1205810274955952128) (535 条消息🔥🔥🔥): 

- **讨论 Miqu 模型的通用性**：用户一直在分享关于 **Miqu** 等模型的见解，包括它们与 **Miquella** 等模型合并（model merges）相比在性能上的优势。还提到了针对 ERP 进行微调的 **MiquMaid** 模型的潜在性能，并由 `@soufflespethuman` 提供了 [MiquMaid-v2-70B](https://huggingface.co/NeverSleep/MiquMaid-v2-70B) 和 [MiquMaid-v2-70B-DPO](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO) 的链接。

- **模型配置和设置查询**：像 `@netrve` 和 `@johnrobertsmith` 这样的用户分享了设置模型的细节和经验，讨论了上下文长度（context length）、重复惩罚（repetition penalty）以及跨 GPU 的内存分配的影响。`@lonestriker` 提供了关于 **exl2 模型**及其量化大小的详细信息，原始大小从 34GB 到 110GB 不等。

- **关于 ST 中 Tokenizer 使用的热烈辩论**：当出现关于 ST 使用 tokenizer 的问题时，`@stoop poops` 建议 `@netrve` 阅读文档。讨论凸显了围绕 ST (Silly Tavern) 中 tokenizer 设置的目的和功能的某些困惑。

- **AMD 用户的技术提示**：`@spottyluck` 提供了关于使用 AMD 的 AOCL 来提高 llama.cpp 推理时 CPU 性能的建议，并推荐了利用 AMD AVX512 扩展和更好内核的特定编译选项。

- **为角色状态实现自定义脚本**：`@johnrobertsmith` 表示有兴趣使用 STscript 和 lorebooks 创建脚本来管理故事场景中的角色状态，寻求帮助和想法，以将其理论知识转化为实际应用。

**提到的链接**：

- [Neko Atsume Cat GIF - Neko Atsume Cat Kitty - 发现并分享 GIF](https://tenor.com/view/neko-atsume-cat-kitty-neko-atsume-vr-speech-bubble-gif-25743386): 点击查看 GIF
- [NeverSleep/MiquMaid-v2-2x70B-DPO · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-2x70B-DPO?not-for-all-audiences=true): 未找到描述
- [Homer Simpsons GIF - Homer Simpsons Audacity - 发现并分享 GIF](https://tenor.com/view/homer-simpsons-audacity-lisa-marge-gif-16591769): 点击查看 GIF
- [The Chi GIF - The Chi - 发现并分享 GIF](https://tenor.com/view/the-chi-gif-11090171): 点击查看 GIF
- [Cat Kitten GIF - Cat Kitten Speech Bubble - 发现并分享 GIF](https://tenor.com/view/cat-kitten-speech-bubble-speech-discord-gif-25192162): 点击查看 GIF
- [What The Fuck Wtf Is Going On GIF - What The Fuck Wtf Is Going On What The - 发现并分享 GIF](https://tenor.com/view/what-the-fuck-wtf-is-going-on-what-the-gif-16853392): 点击查看 GIF
- [GitHub - yule-BUAA/MergeLM: 用于合并语言模型的代码库](https://github.com/yule-BUAA/MergeLM?tab=readme-ov-file): 用于合并语言模型的代码库。通过在 GitHub 上创建账号，为 yule-BUAA/MergeLM 的开发做出贡献。
- [用数据回答问题](https://www.crumplab.com/statistics/): 一本面向心理学本科生的免费统计学入门教材，包含实验手册和课程网站。采用 CC BY SA 4.0 许可协议。
- [Cats Cat GIF - Cats Cat Cucumber - 发现并分享 GIF](https://tenor.com/view/cats-cat-cucumber-scared-gif-10226870): 点击查看 GIF
- [Boxing Day GIF - Cats Cats In Boxes Armor - 发现并分享 GIF](https://tenor.com/view/cats-cats-in-boxes-armor-gif-3294499): 点击查看 GIF
- [Did You Pray Today Turbulence GIF - Did you pray today Turbulence - 发现并分享 GIF](https://tenor.com/view/did-you-pray-today-turbulence-gif-6688607547865508186): 点击查看 GIF
- [Nexesenex/abacusai_Smaug-Yi-34B-v0.1-iMat.GGUF at main](https://huggingface.co/Nexesenex/abacusai_Smaug-Yi-34B-v0.1-iMat.GGUF/tree/main): 未找到描述
- [Skinner Homer GIF - Skinner Homer Drag Net - 发现并分享 GIF](https://tenor.com/view/skinner-homer-drag-net-nod-plan-gif-4964903): 点击查看 GIF
- [Catzilla 😅 | 不要对你的猫这样做，街上的朋友会笑话它的 👀](https://www.youtube.com/shorts/7wS5Nmvpjp8): “根据 1976 年版权法第 107 条的版权免责声明，允许出于批评、评论、新闻报道等目的的‘合理使用’...”
- [Good Heavens GIF - OMG Shocked Surprised - 发现并分享 GIF](https://tenor.com/view/omg-shocked-surprised-oh-my-god-gasp-gif-5189017): 点击查看 GIF
- [NeverSleep/MiquMaid-v2-70B · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B): 未找到描述
- [NeverSleep/MiquMaid-v2-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-GGUF): 未找到描述
- [NeverSleep/MiquMaid-v2-70B-DPO · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO): 未找到描述
- [NeverSleep/MiquMaid-v2-70B-DPO-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO-GGUF): 未找到描述

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1205878821065334915) (30 messages🔥): 

- **微调聊天机器人模型 (Fine-tuning Chatbot Models)**：`@maldevide` 强调了将基础模型转换为 instruct 模型的基本步骤，包括使用优质的指令数据集进行两个 epoch 的微调。`@jondurbin` 和 `@starsupernova` 进一步讨论了数据集来源（如 [bagel datasets](https://github.com/jondurbin/bagel?tab=readme-ov-file#sft-data-sources)）以及微调的实际过程，这可以为模型增加知识。
  
- **指令数据集推荐 (Instruct Dataset Recommendations)**：`@mr.userbox020` 询问了哪些数据集最适合创建 instruct 模型，`@maldevide` 建议考虑 **WizardLM_evol_instruct_V2_196k**、**OpenHermes-2.5** 等数据集，因为它们具有经过验证的广泛基础，同时也可以混入所需的特定专业领域数据。

- **理解微调的影响 (Understanding the Impact of Fine-tuning)**：`@skirosso` 询问了微调的目的，得到的澄清是微调可以改变聊天机器人模型的语气并增加知识，特别是当预训练在所有层上持续进行时，正如 `@starsupernova` 所解释的那样。`@mr.userbox020` 表示赞同，并指出最优秀的 instruct 模型（如 mixtral）已经能够根据用户的指令讲述关于龙的故事。

- **微调速度与效率的未来 (The Future of Fine-tuning Speed and Efficiency)**：`@mr.userbox020` 分享了一个名为 [unsloth](https://github.com/unslothai/unsloth) 的 GitHub 仓库资源，该仓库声称能为 Mistral 等模型提供更快、更高效的 QLoRA 微调。`@starsupernova` 确认了其性能提升，引用了 2.2 倍的加速和 70% 的 VRAM 减少。

- **微调与训练的对比 (Fine-tuning compared to Training)**：`@wolfsauge` 通过区分训练（training）和微调（fine-tuning）深化了讨论，强调了资源节省和稳定性。他们还提到紧跟当前的微调趋势至关重要，并建议进一步探索特定的微调方法，如带有 PPO 的 RHLF 以及带有 DPO 的 SFT。

**提到的链接**：

- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel?tab=readme-ov-file#sft-data-sources)：A bagel, with everything. 通过在 GitHub 上创建账号为 jondurbin/bagel 的开发做出贡献。
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth)：5X faster 60% less memory QLoRA finetuning. 通过在 GitHub 上创建账号为 unslothai/unsloth 的开发做出贡献。

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1205881885905915954) (10 messages🔥): 

- **分享 Surya OCR 脚本 (Surya OCR Script Shared)**：`@cybertimon` 提供了一个使用 Surya OCR 工具进行光学字符识别的 **Python 脚本**，并提到需要使用命令 `pip install git+https://github.com/VikParuchuri/surya@dev` 安装 Surya 的 dev 分支。
- **代码片段赞赏 (Code Snippet Appreciation)**：`@bartowski1182` 对分享的 Surya OCR 代码表示赞赏，称其非常棒。
- **GitHub 仓库建议 (GitHub Repo Suggestion)**：`@mr.userbox020` 建议 `@cybertimon` 创建一个 **GitHub 仓库** 来分享 Surya OCR 代码，但 `@cybertimon` 澄清这仅仅是一个示例脚本，而不是一个完整的项目。他们随后分享了该代码的 [Gist 链接](https://gist.github.com/CyberTimon/85ca0e797a95e3d5562dd6018f4e2131)。
- **请求调试协助 (Request for Debugging Assistance)**：`@ninyago` 就其 [GitHub 仓库](https://github.com/Ninyago53/webaudiobook.git) 中 `user.html` 的一个 bug 寻求帮助，即在 ElevenLabs 结束说话后，录音功能并不总是能启动。
- **突出潜在的第三方问题 (Potential Third-Party Issue Highlighted)**：针对 `@ninyago` 的请求，`@falconsfly` 建议问题可能与 **ElevenLabs** 有关，并分享了其平台上任务停滞的经历，表明问题可能不在于代码，而在于 ElevenLabs 服务本身。

**提到的链接**：

- [Surya OCR](https://gist.github.com/CyberTimon/85ca0e797a95e3d5562dd6018f4e2131)：Surya OCR. GitHub Gist：即时分享代码、笔记和片段。
- [GitHub - Ninyago53/webaudiobook](https://github.com/Ninyago53/webaudiobook.git)：通过在 GitHub 上创建账号为 Ninyago53/webaudiobook 的开发做出贡献。

  

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1205790688667770940) (399 条消息🔥🔥): 

- **模型问题与讨论**：用户 `@lacrak27` 和 `@heyitsyorkie` 讨论了运行 TheBloke 模型时遇到的问题，结论是错误可能源于将原始 PyTorch 模型转换为 GGUF 过程中的量化（quant）损坏。
- **技术栈查询**：用户 `@jitterysniper` 询问了 LM Studio 的技术栈，`.ben.com` 澄清其基于 llama.cpp，随后讨论扩展到了应用的细节，推测它可能是使用 Electron 构建的。
- **图像生成困扰**：用户 `@sunboy9710` 和 `@heyitsyorkie` 讨论了使用 LM Studio 进行图像生成任务的困难和局限性。
- **VPN 与 ISP 封锁**：用户 `@stevecnycpaigne` 在不同地点访问 huggingface.co 时遇到问题，导致 `@heyitsyorkie` 建议尝试使用 VPN，因为这可能是与 ISP 相关的问题。
- **隐私与使用数据担忧**：用户 `@f0xa` 对比了 GPT4All 和 LM Studio，寻求关于 LM Studio 数据隐私的澄清，`@fabguy` 指出 LM Studio 默认保护隐私，共享数据仅产生于更新和从 Hugging Face 下载模型的过程。
- **寻求 LM Studio 的 Web UI**：用户 `@laststandingknight` 询问是否有用于 LM Studio 聊天的 Web UI，`@fabguy` 确认目前没有。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1187757393556295681)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204759902548000769)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Continue](https://continue.dev/)：未找到描述
- [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF)：未找到描述
- [Twochoices Funny GIF - Twochoices Funny Two - Discover &amp; Share GIFs](https://tenor.com/view/twochoices-funny-two-choices-divorce-gif-5018698)：点击查看 GIF
- [chilliadgl/RG_fake_signatures_ST at main](https://huggingface.co/chilliadgl/RG_fake_signatures_ST/tree/main)：未找到描述
- [no title found](https://news.ycombinator.com/item?id=39326165)：未找到描述
- [GitHub - b4rtaz/distributed-llama: Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage.](https://github.com/b4rtaz/distributed-llama)：在性能较弱的设备上运行 LLM，或通过分布工作负载和划分 RAM 使用量使强大的设备更加强大。 - b4rtaz/distributed-llama
- [examples/how-to-run-llama-cpp-on-raspberry-pi.md at master · garyexplains/examples](https://github.com/garyexplains/examples/blob/master/how-to-run-llama-cpp-on-raspberry-pi.md)：我视频中使用的示例代码。通过在 GitHub 上创建账户为 garyexplains/examples 的开发做出贡献。

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1205856822226067456) (92 条消息🔥🔥): 

- **小模型分类难题**：`@speedy.dev` 在使用 13B 和 7B Llama 模型（特别是 uncensored 变体）进行分类任务时遇到问题，并考虑将 [fine-tuning 作为解决方案](https://discourse.devontechnologies.com/t/experimenting-with-llama2-llm-for-local-file-classification-renaming-summarizing-analysing/76945)。
- **Goliath vs. Goat - 故事质量的模型之战**：`@goldensun3ds` 运行了对比测试，比较了 Bloke Goat Storytelling 70B Q6、Bloke Goliath 120B Q6 和 Mixtral 7B Q6 模型的故事写作能力，指出 Goliath 能更好地捕捉情感，但 Mixtral 速度更快，并提供了应对重复循环（repetitive loops）的见解。
- **本地文档对话仍悬而未决**：`@dr.nova.` 加入社区寻找 GPT-4 的本地替代方案以进行 PDF 对话，得到的反馈是虽然 LM Studio 目前还没有此功能，但 GPT-4 仍是处理基于文档交互的领先解决方案。
- **为任务分配选择最佳模型？一种假设性方法**：`@binaryalgorithm` 思考了建立一个 meta-model 来为给定任务选择最佳模型的想法，`@.ben.com` 提到 [openrouter.ai](https://openrouter.ai) 有一个基础的 router 用于模型选择。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1186031214189084783/1186031214189084783)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=document-question-answering&sort=trending)：未找到描述
- [实验使用 llama2 LLM 进行本地文件分类（重命名、总结、分析）](https://discourse.devontechnologies.com/t/experimenting-with-llama2-llm-for-local-file-classification-renaming-summarizing-analysing/76945)：来自 OpenAI ChatGPT 自动生成匹配文件名的后续 - #3 (by syntagm)。ChatGPT 在为 OCR 处理后的文档和 PDF 引入逻辑方面表现极佳，但如果能实现这一点就太好了……

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1206231358398009344) (1 条消息): 

- **分享模型元数据管理见解**：`@wolfspyre` 通过一篇 [HackMD 文章](https://hackmd.io/@janhq/HJezIhu4T) 强调了模型元数据（metadata）管理的重要性，该文章建议在各种聊天平台中需要更好地对模型“参数”进行分类。他们关注了 `init/load params`、`run params` 和 `server/engine params` 之间的差异。
- **Model.json 标准的可能性**：HackMD 文档讨论了由 Jan 建立 `model.json` 标准的可能性，并提供了一个 [GitHub 仓库链接](https://github.com/janhq/model.json)，其中包含不同版本的 schema 和示例文件。

**提到的链接**：

[Model Object Teardowns - HackMD](https://hackmd.io/@janhq/HJezIhu4T)：模型文件格式

  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1205791295545671710) (197 messages🔥🔥): 

- **寻求 AVX 支持建议**：`@guest7_25187` 很高兴在 eBay 上找到了支持 AVX2 的 Intel Xeon E5-2670 v3 CPU，这将与其服务器兼容。`@heyitsyorkie` 建议，如果结合足够的 RAM 和 Nvidia P40 GPU，`@guest7_25187` 将看到显著的速度提升。
  
- **模型性能的 GPU 决策**：在关于运行 LLM (Large Language Models) 硬件的讨论中，`@nink1` 强调了拥有更多 VRAM 和 CUDA 核心的好处，特别指出 3090 在 VRAM 与核心比例以及 NVLINK 能力方面的优势。`@konst.io` 得到的建议是，再增加 64GB RAM 也没坏处，但首要任务应该是先最大化 VRAM。

- **LLM 推理的 Mac 与定制配置对比**：`@heyitsyorkie` 分享了在 M3 Max 128gb 上运行 Goliath 120b 32k 模型的经验，其速度比他们的 4090 配置更快。同时，`@wildcat_aurora` 讨论了他们涉及 P40 GPU 和 Xeon 处理器的配置，这些设备是从 Apple 的 Siri 服务中退役并重新利用的，以较低的功耗提供了有效的结果。

- **市场推测与硬件策略**：关于未来硬件发展的推测和愿望参杂在一起，`@nink1`、`@christianazinn` 等人辩论了 Nvidia 在消费级 GPU VRAM 上的战略决策，并期待 Apple 推出拥有更多 RAM 的潜在企业级解决方案。

- **LLM 推理的故障排除与设置**：`@therealril3y`、`@lardz90` 和 `@the_yo_92` 等成员就 LM Studio 未利用额外 GPU、升级后 RAM 未被正确检测以及在 M1 Mac 上遇到 JavaScript 错误等问题寻求帮助。`@heyitsyorkie` 和 `@speedy.dev` 提供了快速修复方案，并建议创建支持线程或更新软件以解决问题。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/11105981831443)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1111649100518133842/1205933827999010827)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1153759714082033735/1205652225435631667)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1111649100518133842/1204400922948673546)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (1 messages): 

ramendraws: :1ski_smug:
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1206457972767723520) (2 messages): 

- **更倾向于 CrewAI**：`@docorange88` 表达了对 **CrewAI** 明显优于 AutoGen 的偏好，认为 AutoGen 更难管理。
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1206295896581480490) (1 messages): 

- **Intel 放弃 AVX-512 令人侧目**：`@technot80` 对 **Intel** 在强大的 i9 处理器上放弃 **AVX-512** 支持的决定表示困惑，尤其是因为他们性能较低的 i5-11600k 反而包含该支持。他们认为这一举动*非常奇怪*。
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1205945710164574279) (1 messages): 

- **HuggingFace Hub 遭遇停机**：用户 `@lunarflu` 宣布 **HF Hub** 和 git 托管暂时下线。他们提到团队目前正在努力解决该问题，并请求社区支持。
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1205799064541593631) (603 messages🔥🔥🔥): 

_

- **Discord 验证可能比较棘手**：用户 `@d3113` 在尝试使用 Discord 机器人进行身份验证时遇到错误，通过在设置中启用“允许来自服务器成员的消息”解决了该问题。用户 `@lunarflu` 建议了这一方案，并提到用于验证的 token 随后可以删除。
- **保持对话性**：由于 API bug，用户 `@miker1662` 在 Hugging Face 上寻找对话模型时遇到困难，`@osanseviero` 确认了这一点，并链接到了一个正在处理的 issue 以及 API 使用方式的变化。Hugging Face 正在弃用 `conversational` 并将其合并到 `text-generation` 中。
- **LLM 微调中 PEFT 的抉择**：用户 `@samuelcorsan` 讨论了在微调对话聊天机器人 LLM 时是否使用 PEFT（Parameter Efficient Fine-Tuning），最终决定进行全量微调（full fine-tuning），并从代码中移除了 PEFT。`@vipitis` 建议他们在微调期间使用全序列长度，以避免模型学习位置嵌入（positional embeddings），并对数据集进行分块（chunking）以提高效率。
- **用于边缘侧 CV 的 Raspberry Pi**：用户 `@akvnn` 询问 Raspberry Pi 是否能够持续运行用于多摄像头系统的计算机视觉（CV）模型。用户 `@yamatovergil89` 确认了其可行性，但未明确说明它是否能同时处理多个摄像头。
- **本地运行 LLM 的硬件咨询**：`@lookingforspeed` 寻求建议，询问双 NVIDIA 4080 Super 是否比单块 4090 更适合本地编写像 Mixtral 这样的 LLM 代码。`@tddammo` 推荐了 A4500 或 A4000 等旧代专业卡，因为它们具有更好的能效比并支持 NVLink，随后解释了这些卡相对于消费级显卡的优势。

**提到的链接**：

- [mistralai/Mixtral-8x7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)：未找到描述
- [Hugging Face – The AI community building the future.](https://huggingface.co/)：未找到描述
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206181148976222268)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206246780950544405)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [transformers/src/transformers/configuration_utils.py at 58e3d23e97078f361a533b9ec4a6a2de674ea52a · huggingface/transformers](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/src/transformers/configuration_utils.py#L677C8-L677C10)：🤗 Transformers：适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习。- huggingface/transformers
- [Cry Animecry GIF - Cry Animecry Anime - Discover &amp; Share GIFs](https://tenor.com/view/cry-animecry-anime-gif-27695618)：点击查看 GIF
- [Free Hugs Shack GIF - Free Hugs Shack Forest - Discover &amp; Share GIFs](https://tenor.com/view/free-hugs-shack-forest-scary-weird-gif-23469977)：点击查看 GIF
- [Anime Cry GIF - Anime Cry Sad - Discover &amp; Share GIFs](https://tenor.com/view/anime-cry-sad-gif-14080503)：点击查看 GIF
- [M3 max 128GB for AI running Llama2 7b 13b and 70b](https://www.youtube.com/watch?v=jaM02mb6JFM)：在这段视频中，我们使用拥有 128GB 内存的新款 M3 max 运行 Llama 模型，并将其与 M1 pro 和 RTX 4090 进行对比，以查看该芯片的真实性能...
- [Hugging Face status](https://status.huggingface.co/)：未找到描述
- [Tweet from Ross Wightman (@wightmanr)](https://x.com/wightmanr/status/1742637301346508929?s=20)：@bhutanisanyam1 @jamesbower NVLINK 确实有影响，即使是在双 GPU 上，但影响随分布式工作负载而异。不幸的是，这在目前的爱好者机器（40x0 和 RTX6000 Ada...）上不是考虑因素。
- [andreasjansson/codellama-7b-instruct-gguf – Run with an API on Replicate](https://replicate.com/andreasjansson/codellama-7b-instruct-gguf)：未找到描述
- [Whisper Large V3 - a Hugging Face Space by hf-audio](https://huggingface.co/spaces/hf-audio/whisper-large-v3)：未找到描述
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)：大语言模型 (LLMs) 不可避免地会出现幻觉，因为生成文本的准确性无法仅靠其封装的参数化知识来保证。虽然检索增强...

- [调用 `InferenceClient.conversational` 时出错 · Issue #2023 · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/issues/2023): 描述该 Bug。根据文档调用 InferenceClient.conversational 会导致 400 Client Error。复现代码：from huggingface_hub import InferenceClient InferenceClient().conversationa...
- [使用 ICL-RAG 进行 DSPy Prompt Engineering (如何编写自我改进的 LLM-RM 流水线)](https://www.youtube.com/playlist?list=PLgy71-0-2-F00lrRr2EzzTdnbJXax6sn2): 高级 Prompt Engineering。从人工提示词模板到通过 DSPy 实现的自我改进、自我配置的提示词流水线。流水线自我优化的先进技术...
- [Models - Hugging Face](https://huggingface.co/models?other=conversational): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/17kcgjv/how_does_apples_new_m3_128gb_ram_macbook_pro/): 未找到描述
- [Amazon EC2 G5 Instances | Amazon Web Services](https://aws.amazon.com/ec2/instance-types/g5/): 未找到描述
- [定价 | 云端 AI 遇见无与伦比的价值](https://rundiffusion.com/pricing): 提供从按需付费到完整的月度企业级 Stable Diffusion 方案，满足您的各种需求。以极具竞争力的成本赋能您的愿景。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/17kcgjv/how_does_): 未找到描述
- [什么是 NVLink？](https://blogs.nvidia.com/blog/what-is-nvidia-nvlink/): NVLink 是一种用于加速系统中 GPU 和 CPU 处理器的高速互连技术，旨在推动数据和计算产生可操作的结果。
- [GitHub - huggingface/transformers at 58e3d23e97078f361a533b9ec4a6a2de674ea52a](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a): 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的业界领先的 Machine Learning 库。 - GitHub - huggingface/transformers at 58e3d23e97078f361a533b9ec4a6a2de674ea52a
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): 未找到描述

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1205807574042148894) (8 条消息🔥): 

- **重返 LinkedIn**: `@aiman1993` 在暂停 6 周后恢复了在 LinkedIn 上的发布，分享了一篇[关于升级 CUDA Toolkit 和 NVIDIA 驱动的文章](https://www.linkedin.com/posts/isham-rashik-5a547711b_upgrading-cuda-toolkit-and-nvidia-driver-activity-7161933041349640192-6c9W?utm_source=share&utm_medium=member_desktop)。
- **TokenClassification 微调**: `@kamama2127` 正在学习 Token Classification 的微调技巧。
- **为 API 选择合适的 Python 库**: `@subham5089` 在一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_python-api-generativeai-activity-7162345296587284480-mzuB?utm_source=share&utm_medium=member_desktop)中讨论了理解不同 Python API 调用库（如 **Requests, AIOHTTP, 和 HTTPX**）的重要性。
- **Websockets 与 HTTPX 的结合**: 针对 `@dastardlydoright` 关于 Python 中 Websockets 的疑问，`@subham5089` 推荐了 [httpx-ws](https://frankie567.github.io/httpx-ws/)，这是一个为 HTTPX 提供 WebSockets 支持的库，并提供了[源代码链接](https://github.com/frankie567/httpx-ws)。
- **来自 Karpathy 的 GPT 见解**: `@wonder_in_aliceland` 提到他很喜欢 Andrej Karpathy 关于 **nanogpt** 的视频，该视频探讨了 GPT 背后的内部机制和理念。

**提到的链接**:

[HTTPX WS](https://frankie567.github.io/httpx-ws/): 未找到描述

  

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1205918004282527774) (5 条消息): 

- **深度探讨深度学习突破**：`@branchverse` 分享了一篇[文章](https://link.springer.com/article/10.1007/s10489-022-04278-6)，重点介绍了自 2010 年代以来深度学习的进展。文章指出，创新是由开源工具、硬件进步以及标注数据的可用性驱动的。
  
- **GitHub 上的 Normcore LLM 阅读清单**：`@husainhz7` 链接了一个名为 "Normcore LLM Reads" 的 [GitHub Gist](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)，这是一个与 LLM 相关的代码、笔记和代码片段的集合。
  
- **用于绿色花园的 AI 注入遗传算法**：`@paccer` 讨论了一篇关于将遗传算法与 LLM 结合用于园艺优化的文章。这款由 AI 驱动的工具 GRDN.AI 旨在改进伴生种植，并记录在 [Medium 文章](https://medium.com/@dheymann314/ai-infused-optimization-in-the-wild-developing-a-companion-planting-app-357e5da29d10)中。
  
- **揭秘计算机视觉技术**：`@purple_lizard` 发布了 [Grad-CAM 研究论文](https://arxiv.org/pdf/1610.02391.pdf)的链接，该论文介绍了一种通过视觉解释使卷积神经网络 (CNN) 决策透明化的技术。

- **探索 AI 研究**：`@kamama2127` 指出了 [arXiv](https://arxiv.org/abs/2402.00838) 上最近的一篇 AI 研究论文，其中列出了为该领域做出贡献的作者名单。该论文讨论了人工智能领域的新发现和进展。

**提到的链接**：

- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)：语言模型 (LMs) 在 NLP 研究和商业产品中已无处不在。随着其商业重要性的飙升，最强大的模型已变得封闭、受限……
- [AI-Infused Optimization in the Wild: Developing a Companion Planting App](https://medium.com/@dheymann314/ai-infused-optimization-in-the-wild-developing-a-companion-planting-app-357e5da29d10)：伴生种植是花园繁荣的关键，它提供天然害虫防治，促进健康生长，并为土壤带来更多养分。GRDN.AI 利用 AI 注入的遗传算法应用了这一概念……
- [Normcore LLM Reads](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)：Normcore LLM 阅读清单。GitHub Gist：即时分享代码、笔记和代码片段。
- [Front-end deep learning web apps development and deployment: a review - Applied Intelligence](https://link.springer.com/article/10.1007/s10489-022-04278-6)：机器学习和深度学习模型通常使用 Python、C++ 或 R 等编程语言开发，并作为从后端服务器交付的 Web 应用或移动应用进行部署……

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1205917374147199016) (10 条消息🔥): 

- **数据结构揭秘**：用户 `@kamama2127` 创建了一个 Streamlit 页面，汇总了 **27 种数据结构**、它们的 Python 实现以及相关问题。点击[此处](https://datastructurewithaipy-dhmexr7ukaudutklwjiawp.streamlit.app/)查看该交互式学习工具。

- **从 Zero Shot 到目标检测**：`@pradeep1148` 分享了一个 YouTube 视频，演示了如何使用 MoonDream 视觉语言模型通过 zero shot 方法实现自动目标检测。点击[此处](https://www.youtube.com/watch?v=W4T7zHluzaM)观看教程。

- **DALL-E 和 Midjourney 图像数据集**：用户 `@ehristoforu` 推出了两个 Hugging Face 数据集，包含由 DALL-E 3 和 Midjourney 模型生成的图像。点击[此处](https://huggingface.co/datasets/ehristoforu/dalle-3-images)探索 DALL-E 数据集，点击[此处](https://huggingface.co/datasets/ehristoforu/midjourney-images)探索 Midjourney 数据集。

- **使用 FumesAI 轻松制作动画**：`@myg5702` 分享了 FumesAI 的 Hugging Face Space 链接，其特色是 **text-to-Animation-Fast-AnimateDiff** 工具。点击[此处](https://huggingface.co/spaces/FumesAI/text-to-Animation-Fast-AnimateDiff)开始将你的文本转化为动画。

- **AI 增强的音乐创作工作流**：`.bigdookie` 讨论了将 MusicGen 工具集成到 Ableton 中的改进，并将其比作玩老虎机。点击[此处](https://youtu.be/wCh-ug1475Q?si=R5_jeeBM5aEK7de2)观看创作过程。

**提到的链接**：

- [Text To Animation Fast AnimateDiff - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/text-to-Animation-Fast-AnimateDiff)：未找到描述
- [Prometheus - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/prometheus)：未找到描述
- [Quiz Maker - a Hugging Face Space by narra-ai](https://huggingface.co/spaces/narra-ai/quizmona)：未找到描述
- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM)：我们将了解如何使用 zero shot 目标检测和 moondream 视觉语言模型进行自动目标检测 #llm #ml #ai #largelanguagem...
- [another song from scratch with ableton and musicgen - captain's chair 12](https://youtu.be/wCh-ug1475Q?si=R5_jeeBM5aEK7de2)：在本集中，我们使用 @CradleAudio 的 god particle 为原声乐器增加响度，并使用了 @Unisonaudio 的 drum monkey，这让我有点失望...
- [ehristoforu/dalle-3-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/dalle-3-images)：未找到描述
- [ehristoforu/midjourney-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/midjourney-images)：未找到描述
- [Spatial Media Converter](https://www.spatialmediaconverter.com)：将 RGB 图像转换为适用于 Apple Vision Pro 的空间照片。
- [no title found](https://datastructurewithaipy-dhmexr7ukaudutklwjiawp.streamlit.app/)：未找到描述

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1206085334224478218) (20 条消息🔥): 

- **Flash Attention 引起关注**：`@ericauld` 表达了对分享 **Flash Attention** 的兴趣，并询问是否讨论过它与 **Mamba** 和 **S4** 的关系。`@chad_in_the_house` 和 `@ericauld` 讨论了允许 softmax 计算以分块（blockwise）而非全局方式进行的数学恒等式。
- **对 RWKV 的好奇**：`@typoilu` 表现出对 **RWKV** 模型进行演示的热情，该模型尚未被介绍过，并讨论了未来日期的排期。
- **安排 Mamba 演示**：`@ericuald` 和 `@chad_in_the_house` 协作寻找双方都方便的时间进行 **Mamba** 演示，并使用 [When2meet](https://www.when2meet.com/?23627290-1hbtA) 进行排期。
- **考虑使用 Google Calendar**：`@chad_in_the_house` 建议启用 Google Calendar 来管理演示排期，`@lunarflu` 支持该想法，认为其非常实用。
- **演示的时区协调**：`@typoilu` 提到他们处于 **UTC+1** 时区，`@chad_in_the_house` 提到了与 EST（东部标准时间）的时差，寻求一个适合美国观众参加 RWKV 演示的时间。

**提到的链接**：

[Mamba Presentation - When2meet](https://www.when2meet.com/?23627290-1hbtA)：未找到描述

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1205911769072009286) (1 条消息): 

- **简化多标签图像分类**：用户 `@nielsr_` 解释了如何使用 **Hugging Face Transformers** 为多标签分类**微调图像分类器**。他们提供了一个使用 `Dinov2ForImageClassification` 进行快速实例化的代码片段。
  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1205835648653598741) (20 messages🔥): 

- **PEFT 软件包**：用户 `@开饭噜` 反馈由于无法连接到 HuggingFace，导致在本地保存 LoRA 权重时出现问题；`@nruaif` 提供的解决方案建议 **降级到 PEFT 0.7.1**，该方法已解决此问题。
- **PEFT 保存模型故障排除**：`@vipitis` 指出，当未提供完整路径时，`.save_pretrained()` 方法可能会尝试创建代码库。他们建议尝试使用 **`Path` 对象而非 `str`** 来绕过连接问题。
- **寻求适用于本地的 JSON 感知 LLM**：`@nic0653` 询问是否有能够理解语言和 JSON schemas 并输出 JSON 的稳健大语言模型 (LLM)。讨论指出 **Claude** 在此任务中表现出色，但 **本地部署** 选项仍面临挑战。
- **寻找支持粗俗语言的 15b+ LLM**：`@wavy_n1c9` 寻求推荐一个参数量在 150 亿 (15b) 以上、能够生成包含 **粗俗语言 (profanity)** 对话或文本的本地化 LLM，但聊天记录中尚未出现相关建议。
- **用于本地代码生成的小型模型**：`@adolfhipster1` 征求适用于 **代码生成** 的小型模型建议，表示 GPT-2 效果不足，并正在寻找下载 **LLaMA** 之外的替代方案。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1205797164911955998) (234 messages🔥🔥): 

- **Discord 社区对 AI 的财务建议表示好奇**：`@azashirokuchiki` 等成员讨论了 AI 在生成交易策略方面的局限性，对机器人谨慎的回答以及在财务事项上缺乏记忆力感到沮丧。
- **分享以 AI 为中心的工具 `ts_zip`**：`@lugui` 分享了一个使用大语言模型 (LLM) 压缩文本文件的工具链接，引发了对 AI 驱动压缩技术的兴趣。
- **对 OpenAI 融资雄心的好奇**：文中提到的为 AI 芯片筹集 5-7 万亿美元的消息引发了辩论，`@thedasenqueen` 和 `@1015814` 等用户就其可行性以及对 AI 研发的影响发表了看法。
- **讨论 AI 对社会风险的影响**：`@1015814` 深入探讨了未妥善整合的自主机器人技术带来的社会风险，以及关于 AI 潜在社会益处与潜在危险的辩论。
- **Google Gemini 成为对话焦点**：包括 `@jaicraft` 和 `@thedreamakeem` 在内的多位用户讨论了 Google Gemini AI 产品的优缺点、功能以及在代码编写和解释器能力方面预期的改进。

**提到的链接**：

- [构建针对 LLM 辅助生物威胁生成的早期预警系统](https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation)：我们正在制定一个蓝图，用于评估大语言模型 (LLM) 可能协助他人制造生物威胁的风险。在一次涉及生物专家和学生的评估中，……
- [来自 Jack Krawczyk (@JackK) 的推文](https://fxtwitter.com/JackK/status/1756114082632220717?s=20)：好了 - Gemini 第二天回顾：人们喜欢的地方，我们需要修复的地方。请继续提供反馈。我们正在阅读所有内容。人们喜欢的地方 (♥️♥️♥️) - 写作风格 - 帮助你寻找事物的创造力……
- [ts_zip：使用大语言模型进行文本压缩](https://bellard.org/ts_server/ts_zip.html)：未找到描述内容

  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1205824112862306304) (37 条消息🔥): 

- **ChatGPT 存在注意力问题**：`@.nefas` 对 ChatGPT 在角色扮演环节中表现出的不连贯反应表示沮丧，认为其会突然偏离讨论主题。`@satanhashtag` 和 `@a1vx` 等人提到了上下文限制（context limits），解释说 API 和 Enterprise 用户可以使用完整的 128K token 上下文，但 Plus 和 Team 用户被限制在 32K token。
  
- **GPT-4 订阅共享困惑**：`@nickthepaladin` 询问非订阅用户是否可以使用他们的 GPT，对此 `@solbus` 澄清说，由于所有 GPT 都使用 GPT-4，因此 Plus 或更高版本的订阅是必不可少的。

- **@ 提及功能的不一致性**：`@rudds3802` 提出了无法使用 @ 提及（mentions）功能的问题，`@solbus` 和 `@jaicraft` 讨论了该功能可能处于有限推送阶段，以及在移动端 Chromium 浏览器上存在的问题。

- **被标记的 GPT 引起困惑**：`@eligump` 和 `@yucareux` 讨论了他们的 GPT 内容被标记（flagged）的经历以及申诉过程，认为遵守学术诚信政策可能对重新获得批准起到关键作用。

- **高效 ChatGPT 交互策略**：`@blckreaper` 建议在动作冒险场景中为 ChatGPT 提供多个叙事选项，以改善响应流并节省 token；而 `@airahaerson` 则感叹尽管进行了详细的格式化努力，仍有必要重复指令。

**提到的链接**：

[Discord - 与好友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1203945630561599518/1204282794529005599)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1205906395073880064) (63 条消息🔥🔥): 

- **协助 ChatGPT 创建电子表格**：`@crosscx` 寻求帮助，希望指导 ChatGPT 将 Midlands 证据规则转换为电子表格。然而，由于遇到 Python 错误，他们用完了 ChatGPT 4 的消息额度。用户 `@madame_architect` 和 `@darthgustav` 建议简化任务，或许可以先不考虑电子表格形式，或者将任务拆分为可管理的区块。

- **平实语言转换难题**：`@mondsuppe` 讨论了在让 ChatGPT 将常规文本翻译成遵循特定平实语言规则的简单语言时遇到的困难。`@darthgustav` 和 `@madame_architect` 建议使用示例和分步提示（stepwise prompting）以获得更好的结果，而 `@eskcanta` 分享了一个成功的两步法策略。

- **理解“对话节奏”**：在关于 GPT-4 能力的讨论中，`@drinkoblog.weebly.com` 观察到模型对对话节奏（conversational cadence）的理解，他们认为这意味着在特定语境下对时间的理解。`@bambooshoots` 澄清说这更多是关于对话流（conversational flow）而非时间意识，随后与 `@beanz_and_rice` 就“节奏”的定义进行了进一步交流。

- **GPT-4 对“状态变化”的处理**：`@beanz_and_rice` 详细阐述了 GPT-4 处理“状态变化”（state changes）的能力，提到了它在管理动态条件方面的胜任能力，并推荐了允许模型在多条消息中有效适应和响应的提示策略（prompt strategies）。

- **视频命令插曲**：`@zen_dove_40136` 在对话中插入了 "/video"，被 `@beanz_and_rice` 幽默地以 "/denied" 回应，延续了讨论中发送消息命令的趋势。
  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1205906395073880064) (63 条消息🔥🔥): 

- **寻找 Midlands 证据规则的正确公式**：`@crosscx` 寻求关于如何让 ChatGPT 在不改变内容的情况下将 Midlands 证据规则格式化为电子表格的建议，但遇到了 Python 错误并耗尽了 ChatGPT 消息额度。`@madame_architect` 建议通过不要求电子表格来简化任务，而 `@darthgustav.` 建议分解任务，并随后澄清问题是 CI 环境中的内存错误。

- **平实语言翻译难题**：`@mondsuppe` 分享了为有学习障碍的人士将文本翻译成简单语言的挑战，讨论了诸如短句和清晰结构等特定规则。尽管 `@darthgustav.` 建议模板可能会有帮助，但他们对 ChatGPT-3.5 的能力表示怀疑，认为 GPT-4 可能会表现得更好。

- **GPT-4 与时间感知**：用户讨论了 GPT-4 回复的性质，`@drinkoblog.weebly.com` 观察到在允许其进行多个周期的回复时存在时间感知，从而引发了与 `@beanz_and_rice` 等人关于节奏（cadence）和对话流的讨论。

- **探索使用 GPT-3.5 进行多步骤问题解决**：`@eskcanta` 描述了一个使用 GPT-3.5 将复杂解释简化为儿童友好语言的两步过程，展示了其在将复杂任务分解为可管理阶段方面的潜力。

- **关于定义与能力的对话**：频道内还讨论了 "cadence" 的定义及其与 GPT-4 对话能力的关系，`@beanz_and_rice` 和 `@drinkoblog.weebly.com` 提供了各种解释。这突显了在理解 AI 如何处理和传递信息方面的细微差别。
  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1205844411842433064) (11 条消息🔥): 

- **Elon Musk 隐身了？**：`@fullstack6209` 在一段视频中发现了一个长得很像 Elon Musk 的人，声称那是 **AdrianDittmann**，并提到 **Alex Jones** 曾意外拨入。他们分享了据称包含 Musk 在内的录音对话 [片段](https://twitter.com/i/status/1756105704786817130)，要求观众跳转到时间戳 **1:13** 和 **1:20** 查看关键时刻，并尝试在 [此处](https://twitter.com/DavidFSWD/status/1756287867167584671) 总结了该事件。
- **Musk 的 AI 雄心曝光**：延续 Elon Musk 的话题，`@fullstack6209` 分享了 Musk 的一段话：“嘿，我的皮卡后面有 200 个 GPU，我要制造一个比你们更快的 AI，而且他们确实做到了”，据报道这段话出自 2024 年 2 月 9 日。
- **表情符号建议被驳回**：用户 `@.ben.com` 幽默地评论说不要听取缺乏表情符号技巧的人的建议，尽管建议的具体背景并未提供。
- **重量级框架离线**：`@gabriel_syme` 简短地感叹 **"HF"**（可能指 Hugging Face）仍然宕机，表明该服务存在持续问题。
- **分享 Sam Altman 的推文**：`@teknium` 分享了 Sam Altman 的一条推文，未作进一步评论，观众可以在 [此处](https://twitter.com/sama/status/1756547355556598170) 查看该推文。
- **YouTube 上的自动目标检测**：`@pradeep1148` 提供了一个名为 **"Automatic Object Detection"** 的 [YouTube 视频](https://www.youtube.com/watch?v=W4T7zHluzaM) 链接，但未对内容进行额外评论。

**提到的链接**：

- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM)：我们将了解如何使用 zero shot 目标检测和 moondream 视觉语言模型进行自动目标检测 #llm #ml #ai #largelanguagem...
- [The Truth About Building AI Startups Today](https://www.youtube.com/watch?v=TwDJhUJL-5o&t=49s)：在 Lightcone Podcast 的第一集中，YC 合伙人深入探讨了他们与构建 AI 初创公司的顶尖创始人合作中学到的一切...

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1205851665140424704) (22 条消息🔥): 

- **QuartetAnemoi-70B 发布**：`@nonameusr` 分享了一个名为 [QuartetAnemoi-70B-t0.0001](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001) 的新顺序合并模型，该模型使用 NearSwap 算法结合了四个不同的模型，展示了卓越的叙事能力，且不依赖典型的故事结尾陈词滥调。
- **Senku 70B 在 TruthfulQA 和 ARC-Challenge 中取得高分**：`@carsonpoole` 报告称，[Senku 70B 模型](https://huggingface.co/senku)在没有任何校准的情况下量化为 4-bit，在 TruthfulQA 中达到 62.3 分，在 ARC-Challenge 中达到 85.75 分，并指出结果受到定制 Prompt 格式的影响。
- **混合格式可能提升 Senku 性能**：在关于 Senku 70B 的后续讨论中，`@carsonpoole` 提到训练者的建议，即使用 chatml 可能会提高模型性能，尽管目前在测试格式中尚未实现。
- **在 OpenHermes 上训练微型模型**：`@euclaise` 分享了一个 [Twitter 线程](https://twitter.com/WuMinghao_nlp/status/1756307170512248985)，讨论了一个在 OpenHermes 上训练的小模型，这引发了关于成员训练过的最小模型的侧面讨论，`@teknium` 透露他们训练过的最小模型是 7B 模型。
- **1.5 Bit 量化突破**：`@.benxh` 强调了一个关于 1.5 bit 量化的 GitHub Pull Request，指出这种最先进的量化技术允许 70B 模型占用不到 18GB 的 RAM，并表示打算在 Miqu 模型上对这些新量化版本进行基准测试。

**提到的链接**：

- [alchemonaut/QuartetAnemoi-70B-t0.0001 · Hugging Face](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001)：未找到描述
- [位置编码帮助循环神经网络处理大词汇量 (Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary)](https://arxiv.org/abs/2402.00236)：本研究讨论了位置编码对利用合成基准的循环神经网络 (RNNs) 的影响。位置编码对时间序列中的数据点进行“时间戳”标记，并……
- [ikawrakow 的 1.5 bit 量化 · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453)：这个草案 PR 是一个正在进行中的工作 (WIP)，展示了每权重 1.5 bit (bpw) 的量化。目前仅支持 CUDA，其他支持的后端尚未实现。CUDA、AVX2 和 ARM_NEON 已实现……

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1205795195593957467) (228 条消息🔥🔥): 

- **模型制作技能咨询**：用户 `@copyninja_kh` 询问了创建数据集所需的技能，在进一步询问技术细节后，`@teknium` 建议参考 wizardevol 和 alpaca 的做法。
- **Hugging Face 上的 UNA-SimpleSmaug-34B**：用户 `@fblgit` 分享了 Hugging Face 上 UNA-SimpleSmaug-34B 模型的链接，称其评分优于原始的 Smaug-34B 模型，并指出该模型在 [SimpleMath 数据集](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta)上进行了训练，重点在于提升数学和推理能力。
- **探索 Lilac 处理后的 Hermes**：用户 `@nikhil_thorat` 参与了关于 UMAP 投影在聚类数据集中的实用性讨论，并提供分享 embeddings 和投影的选项。随后，他分享了 [Lilac 处理后的 OpenHermes-2.5 数据集](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5)链接，并提到未来会为该数据集添加一列 2D 坐标。
- **通过 API 托管 Hugging Face 模型**：用户 `@nonameusr` 寻求关于托管 Hugging Face 模型进行 API 推理的最简便方法的建议，得到的建议包括使用 Flask 以及关注 Runpod 等提供按秒计费 GPU 服务的平台。
- **模型创建中 Mergekit 的使用**：用户 `@weyaxi` 重点发布了关于 TheProfessor-155b 的消息，这是一个使用 [mergekit](https://github.com/cg123/mergekit) 构建的合并模型，旨在提供对话、推理和科学领域的广泛技能。`@nonameusr` 对其性能统计数据（如 `0.69 MMLU` 和 `0.4284 GSM8K`）表示了怀疑。

**提到的链接**：

- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b)：未找到描述
- [fblgit/UNA-SimpleSmaug-34b-v1beta · Hugging Face](https://huggingface.co/fblgit/UNA-SimpleSmaug-34b-v1beta)：未找到描述
- [typeof/miqu-70b-6 · Hugging Face](https://huggingface.co/typeof/miqu-70b-6)：未找到描述
- [Buffer Overflow in Mixture of Experts](https://arxiv.org/abs/2402.05526)：Mixture of Experts (MoE) 已成为在保持推理成本稳定的同时扩展大型基础模型的关键要素。我们展示了具有跨批次依赖性的专家路由策略……
- [Doctor-Shotgun/Nous-Capybara-limarpv3-34B · Discussions](https://huggingface.co/Doctor-Shotgun/Nous-Capybara-limarpv3-34B/discussions)：未找到描述
- [Herika - SafeAI in Skyrim](https://youtu.be/W9-uYzVEDuM?si=2DU7bvNUPKJiHu3F)：最符合对齐要求的 AI Skyrim 伴侣
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)：未找到描述
- [lilacai/lilac-OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/lilacai/lilac-OpenHermes-2.5)：未找到描述
- [abacusai/TheProfessor-155b · Hugging Face](https://huggingface.co/abacusai/TheProfessor-155b)：未找到描述
- [Eric Hartford (@erhartford) 的推文](https://fxtwitter.com/erhartford/status/1756747509186338956)：https://huggingface.co/abacusai/TheProfessor-155b TheProfessor-155b 是我与 @abacusai 合作并使用 @chargoddard 的 MergeKit 制作的一个特殊模型 - 其目的是进行交互式头脑风暴……

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1205806529291354142) (30 条消息🔥): 

- **寻求简单的微调路径**：`@natefyi_30842` 询问了在 GPU 资源有限的情况下微调 **Nous: Hermes 2 Mixtral 8x7B DPO** 等模型的最简单方法，以及是否有可以上传 prompt/answer 对进行微调的服务。他们提到了之前使用 **Axolotl** 的经验，但觉得太繁琐。
- **简易微调的工具推荐**：`@teknium` 建议 **together.ai** 可能是满足 `@natefyi_30842` 需求的平台，并提到为了获得好的模型，超参数调优（hyperparameter tuning）是不可避免的。
- **Yarn Repo 7B 训练困扰**：`@yoelvis8576` 分享了他们尝试使用 **FSDP** 和 **Flash Attn-2** 微调 **Mistral 7B** 模型失败的经历，尽管尝试了各种配置，仍出现 CUDA out of memory 错误。
- **用户界面框架的困扰**：`@tempus_fugit05` 讨论了在不同模型之间切换时创建正确结构 prompt 的挑战，`@.ben.com` 提议使用 **ollama** 作为解决方案，但 `@tempus_fugit05` 更倾向于继续开发个人框架。
- **自主 LLM 的探索**：`@0xsingletonly` 对自主 LLM（Autonomous LLMs）表示了兴趣，并打算参与 Nous 即将推出的 SDK，`@teknium` 提到该 SDK 正在开发中，旨在简化自主 LLM 的使用。

---

### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1205805344165462066) (8 条消息🔥): 

- **关于路线图功能的简要困惑**：`@gabriel_syme` 对路线图中缺失某项功能表示失望。
- **关于 Llava 团队集成工作的澄清**：针对 `@gabriel_syme` 的回复，`@qnguyen3` 告知 **Llava 团队** 已经自行集成了该功能，但尚未合并（merged），目前可在他们的仓库中找到。
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1205793352289751100) (162 条消息🔥🔥): 

- **迈向 AI 的第一步**：`@dev2c2j` 拥有编程经验，并曾根据研究论文从零开发软件。他们表达了理解并可能为 AI 做出贡献的愿望，对 EleutherAI 和 GPT-J 表现出兴趣，但在起步阶段面临理解上的障碍。
  
- **社区参与和指导**：`@paganpegasus` 等人讨论了用于模型训练的 The Pile 数据集的大小，并建议使用 rsync 以实现更快的数据传输。`@alexanderrgriffing` 分享了学习 CUDA 编程的兴趣，并提到了其他关注小型系统和涌现（emergent）AI 的社区。

- **优化的苦与乐**：`@zippika` 讨论了 Prodigy 优化器的优缺点，特别是其高 VRAM 占用但在训练 Diffusion 模型时的卓越性能。他们还感叹了 AI 模型合并（merging）的滥用以及 Hugging Face 仓库上的存储空间问题。

- **探索 AI 的边缘领域**：`@dev2c2j` 挑战了 AI 开发需要暴力计算资源的观点，建议将微型网络和大型网络结合使用可能更具资源效率。他们的言论引发了关于选择正确的 AI 问题和方法的讨论，`@catboy_slim_` 和 `@alexanderrgriffing` 强调了规模（scale）在当前 AI 开发中的重要性。

- **异常的机器学习模型**：社区对可疑 AI 实践的兴起以及开源 LLM 可能进入“诈骗时代（grift era）”表示担忧，正如 `@canadagoose1` 所提到的。`@rallio.` 提到了关于 OpenAI 即将发布新产品的广告传闻，可能与超级碗（Super Bowl）同步，`@clockrelativity2003` 分享了实际的 Microsoft 广告链接。

**提到的链接**：

- [XXIIVV &mdash; uxn](https://wiki.xxiivv.com/site/uxn.html)：未找到描述
- [Join the Learn AI Together Discord Server!](https://discord.com/invite/learnaitogether)：学习并构建 AI。技术问答、教程、协作、活动、模型机器人... ML, NLP, Generative AI (Midjourney, ChatGPT)... | 57384 名成员
- [Join the emergence Discord Server!](https://discord.gg/YA2eCJ2P)：查看 Discord 上的 emergence 社区 - 与其他 346 名成员一起交流，享受免费的语音和文字聊天。
- [wolfram/miqu-1-120b · Hugging Face](https://huggingface.co/wolfram/miqu-1-120b)：未找到描述
- [Microsoft Game Day Commercial | Copilot: Your everyday AI companion](https://youtu.be/SaCVSUbYpVc?t=40)：借助 Microsoft Copilot 和 AI 的力量，创意变为行动，不可能变为可能，希望变为现实。Copilot 对所有人开放...
- [Nvidia CUDA Compiler - Wikipedia](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler)：未找到描述
- [Crystal Nights &#x2014; Greg Egan](https://www.gregegan.net/MISC/CRYSTAL/Crystal.html)：未找到描述
- [GitHub - konstmish/prodigy: The Prodigy optimizer and its variants for training neural networks.](https://github.com/konstmish/prodigy)：用于训练神经网络的 Prodigy 优化器及其变体。 - konstmish/prodigy
- [GitHub - konstmish/prodigy: The Prodigy optimizer and its variants for training neural networks.](https://github.com/konstmish/prodigy?tab=readme-ov-file#diffusion-models)：用于训练神经网络的 Prodigy 优化器及其变体。 - konstmish/prodigy

  

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1205787426212159508) (66 条消息🔥🔥): 

- **嵌套网络引发关注**：`@thatspysaspy` 对一条关于 JAX 中嵌套网络的推文感到兴奋，并在频道内分享了该[灵感来源](https://twitter.com/aicrumb/status/1756156655836770504)。
- **Transformer 中的 Transformer**：`@thatspysaspy` 分享了一个 [Colab notebook](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq) 的访问权限，用于实验嵌套 Transformer 网络，并确认该实现是可反向传播（backpropagatable）且 JIT 兼容的。
- **关于向量方向混淆的讨论**：`@alexanderrgriffing` 发起了一场关于数学和机器学习惯例的辩论，特别是关于行向量与列向量的方向问题，`@thatspysaspy` 强调了 numpy 由于其广播语义（broadcasting semantics）实际上是如何将它们视为行向量处理的。
- **Diffusion 论文实现的援助**：`@Nipsu` 寻求关于实现一篇 Diffusion 论文中提示词感知调整（prompt-aware adjustment）方法的帮助，并在 [gist](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57) 中分享了用于 Section 3.3 可视化的代码，同时 `@johnryan465` 表示愿意协助调试该问题。
- **分享 UniReps 研究论文集**：`@digthatdata` 提供了一个有用的 GitHub 仓库链接 [UniReps-resources](https://github.com/UniReps/UniReps-resources)，其中包含为社区利益汇编的研究论文集。

**提及的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/794042109048651818/1205251974912811029)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [A phase transition between positional and semantic learning in a solvable model of dot-product attention](https://arxiv.org/abs/2402.03902)：我们研究了点积注意力层如何学习位置注意力矩阵（Token 根据各自位置相互关注）和语义注意力矩阵（根据...
- [Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning](https://arxiv.org/abs/2402.06619)：数据集是现代人工智能许多突破的基础。自然语言处理（NLP）领域的许多近期成就可归功于对预训练模型的微调...
- [Image Inpainting via Tractable Steering of Diffusion Models](https://arxiv.org/abs/2401.03349)：Diffusion 模型是目前生成照片级真实感图像的最先进技术。然而，为图像修补（inpainting）等受限图像生成任务控制采样过程仍然具有挑战性...
- [Fixed-point Inversion for Text-to-image diffusion models](https://arxiv.org/abs/2312.12540)：文本引导的 Diffusion 模型为生成和处理图像提供了强大的新方法。这些模型的几种应用，包括图像编辑插值和语义增强，需要...
- [GitHub - UniReps/UniReps-resources](https://github.com/UniReps/UniReps-resources)：通过在 GitHub 上创建账户来为 UniReps/UniReps-resources 的开发做出贡献。
- [Google Colaboratory](https://colab.research.google.com/drive/1MFV_Y_G8JGfjmC7FkfGlc_Ew6rHHWBbq)：未找到描述
- [Jupyter Notebook Viewer](https://nbviewer.org/gist/tvaranka/441524202bcbf8b14c6de28dad6f8f57)：未找到描述

---

### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1205879335924535366) (7 messages): 

- **TOFU 基准测试引发辩论**：`@hailey_schoelkopf` 链接了名为 "TOFU" 的论文，该论文涉及一个从已训练模型中卸载敏感数据的基准测试 ([TOFU 论文](https://arxiv.org/abs/2401.06121))。`@stellaathena` 对该论文作为有效的 Machine Unlearning (MUL) 基准测试的意义表示怀疑。
- **寻求澄清**：`@stellaathena` 多次阅读了 TOFU 论文，但仍对其作为有意义的 MUL 基准测试的价值持怀疑态度，并对其影响提出质疑。
- **分享潜在资源**：`@aidan5513` 分享了另一篇可能为对话提供启发的论文，讨论了神经元剪枝后模型中概念的重新学习 ([神经元剪枝论文](https://arxiv.org/abs/2401.01814))。
- **论文作为优秀的入门读物**：`@millander` 认为推荐的论文很有价值，称两者都是所讨论主题的深刻入门材料。
- **寻求对 TOFU 批评的见解**：`@millander` 直接询问了 `@193204646687408129`（可能是 stellaathena 的唯一用户 ID），询问对 TOFU 论文的具体不同意见，包括是否担心其在问答设置中的实际应用有限。

**提到的链接**：

- [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)：在来自网络的庞大语料库上训练的 Large Language Models 可能会记忆并重现敏感或私人数据，从而引发法律和伦理担忧。卸载（Unlearning）或微调模型以遗忘...
- [Large Language Models Relearn Removed Concepts](https://arxiv.org/abs/2401.01814)：通过神经元剪枝进行模型编辑的进展为从 Large Language Models 中移除不良概念带来了希望。然而，目前尚不清楚模型是否具有重新获取已移除...

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1205797366825877504) (11 messages🔥): 

- **MedQA 基准测试审查**：`@johnnysands` 提出了一个假设，即 **MedQA 基准测试** 可能是多选题格式，如果 Pythia 模型本身不擅长此类问题，会导致其表现几乎处于随机概率水平。

- **寻找模型 API 对比数据**：`@matthiaslau` 正在寻找使用 `log_samples` 的模型 API 详细结果，以便对来自 **OpenAI、Anthropic 和 Open LLM Leaderboard** 的各种模型进行深入**对比**。

- **征集 GPQA 数据集任务**：`@hailey_schoelkopf` 确认了新的任务 PR，并建议增加 **GPQA 数据集** 的任务，同时警告说由于数据集作者对**数据泄漏**的担忧，可能需要手动下载。该数据集可以在[这篇学术论文](https://arxiv.org/abs/2311.12022)中找到。

- **关于 Big Bench Hard 任务评估的澄清**：`@scrungle.tech` 寻求关于使用 GPT-4 模型**评估任务**的建议，特别是关于响应格式的选择，`@hailey_schoelkopf` 建议他们目前的方法是搜索整个响应（选项 A）。

- **幻觉排行榜需要贡献者**：`@pminervini` 公开邀请大家为新的**幻觉排行榜**做出贡献，该排行榜涉及多个新颖任务，可以在 [Hugging Face 的博客文章](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations)中找到，并在 [Hugging Face 平台](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks)的 Harness 空间中详细列出。

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1205854690735292466) (166 条消息🔥🔥): 

- **级联 ASR+LLM+TTS 策略被认为不足**：用户 `@donjuan5050` 发表观点认为，在语音机器人中采用级联 ASR + LLM + TTS 的方式并不理想。相反，他更倾向于使用对话数据集进行 end2end 训练，`@wielandbrendel` 指出 [Bud-e 语音助手](https://speechbot.github.io/spiritlm/index.html) 遵循了这种方法，将 ASR、LLM 和 TTS 集成在一个 Pytorch 模型中，实现了端到端的可训练性。

- **AI 艺术的法律困境**：讨论围绕美国地方法院法官 *William Orrick* 的一项裁决展开，该裁决驳回了 Midjourney 和 StabilityAI 根据第一修正案辩护提出的提前撤案请求。用户们辩论了该案的影响，而 `@pseudoterminalx` 分享了他的看法，即法官“回绝了他们的主张”但并未准许他们的动议。

- **OpenAI 的 Diffusers 和 Stable Diffusion 争议**：社区内有关于 `sd-forge` 的讨论，该项目包含了来自 diffusers、automatic1111 和 comfyui 的代码，但旨在避免与上述项目产生关联。`@astropulse` 等人对 Stable Diffusion 模型开源 UI 领域的混乱现状发表了看法，并强调了一句话：“我更喜欢 diffusers，那里的开发者很稳定（stable），而代码库（codebase）不稳定。”

- **使用神经网络生成 DnD 地图**：`@thejonasbrothers` 展示了利用最近的 checkpoint 成功创建详细的《龙与地下城》（Dungeons and Dragons）地图，`@pseudoterminalx` 提供了生成的艺术作品图像作为示例。

- **AI 开发领域的技术难题**：用户报告了 Hugging Face 服务宕机的问题，并讨论了依赖外部 API 的弊端，即当这些服务经历停机时会导致运行问题。此外还提到了 Diffusers 内部的技术复杂性以及模型调度（scheduling）的替代方案。

**提到的链接**：

- [未找到标题](https://speechbot.github.io/spiritlm/index.html)：未找到描述
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述
- [Thumbs Up Double Thumbs Up GIF - Thumbs Up Double Thumbs Up Like - Discover &amp; Share GIFs](https://tenor.com/rbmfd590vsb.gif)：点击查看 GIF
- [Section 230 - Wikipedia](https://en.m.wikipedia.org/wiki/Section_230)：未找到描述
- [Atlas Struts](https://youtube.com/shorts/SFKM-Rxiqzg?si=Dgl7Ey38F4Mu6V5B)：难不倒 Atlas！我们的类人机器人已准备好投入实际工作，结合了力量、感知和移动能力。
- [TheBloke/deepseek-coder-33B-instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF)：未找到描述
- [GitHub - Ninyago53/webaudiobook](https://github.com/Ninyago53/webaudiobook.git)：通过在 GitHub 上创建账号来为 Ninyago53/webaudiobook 的开发做出贡献。
- [Forge Is Not Using ComfyUI as A Backend · lllyasviel/stable-diffusion-webui-forge · Discussion #169](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/169#discussioncomment-8431103)：最近有人开始散布关于 Forge 使用 ComfyUI 作为后端的虚假信息。这是错误的，对社区有害，也对我们工程团队的努力有害。后端...

---

### LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1205853975589429293) (1 messages): 

- **遇见 BUD-E，一个开源语音助手**：`@spirit_from_germany` 宣布了 **BUD-E**，这是一个低延迟、自然发音的语音助手，可以在标准游戏笔记本电脑上完全离线运行。他们鼓励大家加入并贡献力量，在 [博客文章](https://laion.ai/blog/bud-e/) 中分享了细节，并邀请人们加入他们的 [Discord 社区](https://discord.com/invite/MDTpftKbpv) 以帮助进一步开发 BUD-E。
- **BUD-E 发布推文提醒**：LAION 在 [Twitter](https://fxtwitter.com/laion_ai/status/1756293407855485002?t=HzWRilPFjLu0Cc9DaRjEmw&s=19) 上宣布了 **BUD-E** 的发布，强调了其自然的语音和离线功能，同时寻求合作者加入 Discord 并协助该项目。

**提及的链接**：

- [BUD-E: Enhancing AI Voice Assistants’ Conversational Quality, Naturalness and Empathy | LAION](https://laion.ai/blog/bud-e/)：&lt;p&gt;AI 语音助手彻底改变了我们与技术的互动方式，回答查询、执行任务并让生活变得更轻松。然而，生硬的...
- [LAION (@laion_ai) 的推文](https://fxtwitter.com/laion_ai/status/1756293407855485002?t=HzWRilPFjLu0Cc9DaRjEmw&s=19)：我们展示了 BUD-E：一个自然发音的开源语音助手，可在标准游戏笔记本电脑上低延迟运行，无需互联网连接。加入我们的 Discord 并帮助我们构建...
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/invite/MDTpftKbpv)：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持密切联系。

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1206127763128459334) (5 messages): 

- **Wasserstein Loss 之谜**：`@yoavhacohen` 询问了关于某项目中判别器使用 Wasserstein loss 的说法，但在 [GitHub 仓库的代码](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py) 中找不到证据。
- **全栈人才展示**：`@fashionista8894` 提供了他们的全栈设计师和开发人员服务，并与社区分享了他们的 [在线作品集](https://shinobi8894.onrender.com/)。
- **分享科学文章**：`@helium__` 提供了一个 [科学文章链接](https://www.science.org/doi/10.1126/sciadv.adl4000)，但该帖子后没有相关的上下文或讨论。
- **寻求 MAGVIT 复现指导**：`@lostneko` 正在寻求复现 MAGVIT V2 模型的背景技术建议，并以 [GitHub 仓库](https://github.com/lucidrains/magvit2-pytorch) 作为起点。
- **介绍关于 Agent 基础模型的论文**：`@vrus0188` 分享了一篇名为“**An Interactive Agent Foundation Model**”的 arXiv 论文链接，并为对此话题感兴趣的人列出了 [作者和摘要](https://arxiv.org/abs/2402.05929)。

**提及的链接**：

- [Shinobi](https://shinobi8894.onrender.com/)：未找到描述
- [An Interactive Agent Foundation Model](https://arxiv.org/abs/2402.05929)：人工智能系统的发展正在从创建静态、特定任务的模型转向动态、基于 Agent 的系统，这些系统能够在广泛的应用领域中表现良好...
- [generative-models/sgm/modules/autoencoding/losses/discriminator_loss.py at main · Stability-AI/generative-models](https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/autoencoding/losses/discriminator_loss.py)：Stability AI 的生成模型。通过在 GitHub 上创建账户，为 Stability-AI/generative-models 的开发做出贡献。

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1205841083536253009) (77 条消息🔥🔥): 

- **Chatbot 模型对比**：用户 `@mares1317` 建议打开两个标签页来直接**对比不同的 AI 模型**。
- **iPad App 灵敏度问题**：`@tylersavage` 批评了 **Perplexity iPad App**，指出在握住 iPad 侧边时，其线程退出功能过于灵敏。
- **Perplexity API 讨论空间**：`@boyn_` 询问是否有专门为使用 **Perplexity API** 的开发者开设的频道；`@mares1317` 将其引导至 Discord 上现有的频道。
- **Perplexity 与其他 AI 模型对比**：`@tauist.` 介绍了 [AIswers](https://www.aiswers.com/)，这是一个用户可以将 **Perplexity 的性能**与其他 AI 进行对比的网站。
- **iOS App Beta 测试已关闭**：用户讨论了 **Perplexity iOS App**；`@icelavaman` 确认目前不再接受新的 iOS Beta 测试人员。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1176526177050054766/1193673004811563158)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1176526177050054766/1176526177050054766)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649527299055688/1197511473749032981)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1161802929053909012)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649527299055688/1205951344876326913)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1111786888626438245/1205936844613619712)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1198438847877480629)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1107686562357063883)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的好友和社区保持紧密联系。
- [海绵宝宝想象力 GIF - 海绵宝宝想象力彩虹 - 发现并分享 GIF](https://tenor.com/view/sponge-bob-imagination-rainbow-gif-6957879033178108845)：点击查看 GIF

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1205924093652766750) (11 条消息🔥): 

- **me.lk 的频道指南**：`@actisenergy` 表达了困惑，`@me.lk` 澄清说 “sharing” 频道是用于分享使用 Perplexity 获得的显著结果。
- **澄清关于 Discord 机器人的困惑**：当 `@emullie` 使用 `/dream` 命令时，`@me.lk` 告知他们 **Discord 机器人已停止服务**。
- **分享 Perplexity 搜索结果**：用户分享了他们的 Perplexity.ai 搜索结果链接，包括 `@nejjad`、`@deepanshumehta`、`@buttros`、`@bioforever`、`@austind1313_49718` 和 `@w00dh3n`，展示了从 AI 伦理到生成图像等各种主题。
- **提醒将线程设为公开**：针对 `@bioforever` 分享的链接，`@me.lk` 提醒要**确保通过点击分享按钮将线程设为公开**，以便他人查看。
  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1205933768225984564) (9 messages🔥): 

- **关于模型函数调用的澄清**：`@sourya4` 寻求关于 `mixtral-8x7b-instruct` 是否应在所有输入中使用 **messages field** 的澄清，并确认该模型**不支持 function calling**。他们还询问了与 GPT-4 相比的**延迟 (latency)** 情况，以及使用 Mistral **降低延迟**的可能性。
  
- **关于 Mistral 32k 上下文长度的查询**：`@sourya4` 引用了 Perplexity AI 的功能路线图 [链接](https://docs.perplexity.ai/docs/feature-roadmap)，并询问 **Mistral 32k context length** 的**可用性更新**，该文档最后一次更新是在 2 个月前。

- **处理 API 速率限制错误**：`@johang_11693` 在通过 App script 使用 API 时遇到了 **429 HTTP 状态错误**，提示超出请求速率限制。尽管使用的是 Perplexity，但错误消息中提到了 OpenAI。

- **速率限制错误的可能原因**：`@clay_ferguson` 分享道，遇到此类错误可能是因为 OpenAI 的 **credits 耗尽**，同时也承认 Perplexity 的情况可能不同，实际上可能意味着超出了真实的速率限制。

- **通过延迟解决速率限制错误**：`@johang_11693` 在确认有足够 credits 并参考了 **GPT-4 提供的修复方案**后，通过在 App script 中**添加毫秒级延迟**解决了**速率限制错误**。

**提到的链接**：

[Feature Roadmap](https://docs.perplexity.ai/docs/feature-roadmap)：未找到描述

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1205993757657268304) (9 messages🔥): 

- **H100 对 AGI 的助力**：用户 `@andreaskoepf` 讨论了 H100 的计算能力，认为凭借合适的模型和足够数量的此类 GPU，实现 AGI 是可能的。他们引用了一篇 [AI Impacts 文章](https://aiimpacts.org/brain-performance-in-flops/)，该文章提供了模拟人类大脑活动所需的 FLOPS 估算范围。
- **赞美 GPU 驱动的超级计算**：`@andreaskoepf` 欣赏了一张展示巴塞罗那 MareNostrum 超级计算机的“GPU 教堂”照片，该计算机托管了 V100 和 MI50 GPU。照片最初由 `<@745592110245478452>` 发布在 [Twitter](https://twitter.com/johannes_hage) 上，突显了容纳先进计算节点的宏伟架构。
- **面向机器学习爱好者的斯坦福 MLSys 研讨会**：`@ericauld` 分享了一个 YouTube 播放列表链接，其中包含 [斯坦福 MLSys 小组的研讨会](https://www.youtube.com/playlist?list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq)，涵盖了机器学习系统的各种主题。
- **来自斯坦福 MLSys 的 AI 硬件见解**：紧接着，`@iss_llm` 强调了一个特定的研讨会：[Notes on AI Hardware - Benjamin Spector | Stanford MLSys #88](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86)，认为它与论坛的讨论特别契合且可能很有趣。
- **对 CUDA-MODE 5 会议录像的期待**：针对 `@freakgoy` 询问“`CUDA-MODE 5: Going Further with CUDA for Python Programmers`”活动的录像版本，`@jeremyphoward` 回复称录像应该会在周一前发布。活动详情由 `<@neurosp1ke>` 在 [Twitter](https://x.com/neurosp1ke/status/1756340500116754448?s=20) 上分享。

**提到的链接**：

- [MLSys Seminars](https://www.youtube.com/playlist?list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq)：未找到描述
- [来自 Andreas Köpf (@neurosp1ke) 的推文](https://x.com/neurosp1ke/status/1756340500116754448?s=20)：CUDA-MODE 5：为 Python 程序员进一步深入 CUDA。编写利用共享内存和线程同步的分块内核 (tiled kernels) 🚀。演讲者：@jeremyphoward，太平洋标准时间 2 月 10 日星期六下午 12:00 / 9:00 P...
- [Notes on AI Hardware - Benjamin Spector | Stanford MLSys #88](https://www.youtube.com/watch?v=PlraH57ey4k&list=PLSrTvUm384I9PV10koj_cqit9OfbJXEkq&index=86)：斯坦福 MLSys 研讨会系列第 88 集！AI 硬件笔记。演讲者：Ben Spector。摘要：本周，我们的主持人之一 Ben Spector 代班...
- [Brain performance in FLOPS](https://aiimpacts.org/brain-performance-in-flops/)：模拟人类大脑相关活动所需的计算能力已被多位作者估算，答案从 10^12 到 10^28 FLOPS 不等。详情笔记：我们尚未进行深入调查...

  

---

### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1205861739275747378) (1 messages): 

- **介绍 Accelerated Computing Online**：用户 `@tfsingh` 宣布创建了 [Accelerated Computing Online](https://acceleratedcomputingonline.com)，这是一个用于无服务器执行 Triton kernel 的在线环境。该项目托管在 GitHub ([tfsingh/aconline](https://github.com/tfsingh/aconline))，允许用户在 T4 GPU 上运行代码，是强大的 Lightning 平台的一个更轻量级的替代方案。

**提到的链接**：

[ACO](https://acceleratedcomputingonline.com): 未找到描述

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1205942870222053427) (38 messages🔥): 

- **实验揭示了简单且快速的解决方案**：`@artste` 通过新的 `experiment_A2` 实现了**相同的输出**，该实验按照 `@719599526448463933` 的建议将函数应用于常量，现在与 `experiment_M` 非常接近。该用户表示意识到并满意于最简单的解决方案最终证明是最快的，详见其[更新后的 notebook](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb)。
  
- **推荐内存合并 (Memory Coalescing)**：`@cudawarped` 建议通过合并内存读/写来提高性能，并分享了关于该主题的[相关 notebook](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.ipynb)。

- **NVIDIA NPP 讨论**：`@zippika` 分享了他们更倾向于使用 NVIDIA NPP 进行 CUDA 操作，并提到为此创建了一个 Torch C++ 扩展，包含 remap、dilate、erode 等函数。`@cudawarped` 对 NPP 可能存在的性能问题发表了评论，而 `@morousg` 则考虑将 NPP 与他们自己的库进行性能基准测试对比。

- **CUDA MatMul 中的数值稳定性**：`@andreaskoepf` 询问了在批处理推理 (batched-inference) 与单个序列中使用简单的 fp16 CUDA matmul 时，数值稳定性和结果的细微差异。`@_tvi_` 和 `@cudawarped` 讨论了非确定性和计算顺序，强调了其对相同输入重复运行的影响。

- **独立开发的扩展的所有权**：`@zippika` 不确定是否公开其在业余时间开发的 Torch C++ 扩展。`@morousg` 概述了代码通常被视为个人财产的条件，并表示有兴趣从 `@zippika` 的经验中学习，以使其库可以从 Python 访问。

**提到的链接**：

- [lecture2/lecture3/cuda_rgb_to_gray_refactor.ipynb at cuda_rgb_to_gray_refactor_notebook · artste/lecture2](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb): lecture 2 - 2024-01-20。通过在 GitHub 上创建账号为 artste/lecture2 的开发做出贡献。
- [cuda_mode_lectures/lecture3/rgb_to_grey.ipynb at rgb_to_grey · cudawarped/cuda_mode_lectures](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.ipynb): cuda-mode 讲座材料。通过在 GitHub 上创建账号为 cudawarped/cuda_mode_lectures 的开发做出贡献。

  

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1205853420758638612) (11 条消息🔥): 

- **CUDA 环境、模块和设备属性**：`@morgangiraud` 使用 Torch 测试了 CUDA 可用性和设备属性，显示了设备数量、设备名称和设备能力等详细信息。他们展示了两个支持 Peer Access 的 NVIDIA 图形设备，以及设备属性的输出，包括来自 Nvidia CUDA samples p2pBandwidthLatencyTest 的 P2P 连接矩阵。
  
- **分布式矩阵乘法问题**：在多 GPU 环境中，`@morgangiraud` 发现了一个问题，即从一个 GPU 将 Tensor 复制到另一个 GPU 会导致数据错误。代码演示了在通过 CPU 复制时可以正常工作的分布式矩阵乘法，但在直接进行设备到设备（device-to-device）传输时会失败，共享结果中的错误输出突显了这一点。

- **寻找多 GPU 测试人员**：`@morgangiraud` 请求任何拥有两台或更多 GPU 机器访问权限的人测试提供的分布式矩阵乘法代码，以验证该问题是否也在其他设置中出现。
  
- **FAISS 嵌入向量错误查询**：`@akshay_1` 提到在 FAISS (colbert) 中嵌入向量时遇到错误，并指出由于需要反复试验，寻找解决方案的成本可能很高。

- **CUDA Graph 问题推测**：针对 `@jxmnop` 分享的关于 Torch compile 问题的推文，`@marksaroufim` 推测这可能与 CUDA Graph 问题有关，但需要一个可复现的示例来确认。

- **分布式 Worker 超时调试建议**：针对 `@akshay_1` 的 FAISS 错误，`@uwu1468548483828484` 建议该错误可能是由于某个分布式 Worker 未能到达 allreduce 调用，从而导致超时。为了进行调试，他们建议使用 GDB 运行以检查哪个 Worker 挂起了。

**提到的链接**：

[来自 jack morris (@jxmnop) 的推文](https://x.com/jxmnop/status/1755397683471143172)：哎呀。这就是当我尝试使用 torch compile 时发生的事情。

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1205962994186461254) (2 条消息): 

- **深入了解 CUDA**：`@andreaskoepf` 宣布了 **CUDA MODE 第 5 讲：为 Python 程序员进一步讲解 CUDA**，邀请大家稍后参加这一信息丰富的课程。
- **提供讲座链接**：`@jeremyhoward` 分享了 [讲座的 Discord 链接](https://discord.gg/6UQXQYZp)，供参与者访问 CUDA MODE 第 5 讲。

**提到的链接**：

[加入 CUDA MODE Discord 服务器！](https://discord.gg/6UQXQYZp)：CUDA 阅读小组 | 4068 名成员

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/) (1 条消息): 

ericauld: 非常感兴趣，虽然我刚意识到我迟到了大约一个月。
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1206339445746442250) (3 条消息): 

- **用户请求关于报告分析的教程**：`@smexy3` 对 `<@325883680419610631>` 的视频发表了评论，建议如果能包含一份关于**如何阅读报告并识别融合（fusion）/优化机会**的指南会更有帮助。
- **新的教学视频即将发布**：针对 `@smexy3` 的回复，`@marksaroufim` 确认**下一部视频**（涉及上述主题）**将于 3 月 2 日发布**。
- **对即将发布内容的期待**：`@smexy3` 对**未来视频**的预告表示兴奋，并对 `@marksaroufim` 的更新表示感谢。
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1205802595331870770) (28 条消息🔥): 

- **对快速转换的赞赏**：`@dangfutures` 对 **awq gguff 转换器**表示满意，评价其为 **"10/10 的快"**。
- **对 HuggingFace 停机的担忧**：`@c.gato` 遇到了似乎与 HuggingFace 停机有关的应用崩溃，尽管这是一个**本地训练任务**。`@nanobitz` 和 `@nruaif` 提供了建议，提议**降级到 0.7.1 版本**并讨论了可能的诱因，如未关闭的 socket。
- **对 HuggingFace 停机的沮丧**：用户 `@noobmaster29`、`@rtyax` 和 `@c.gato` 评论了 HuggingFace 的服务器停机，`@rtyax` 注意到服务**短暂恢复后又再次下线**。
- **模型推理的替代方案**：`@noobmaster29` 询问了关于使用 vllm 进行本地推理的问题，并提到了 **TensorRT**，寻求关于最快解决方案的反馈。
- **探索扩展模型参数**：`@xzuyn` 询问社区是否有人在 Mistral 模型上尝试过 **qlora**，特别是 16k 的最大长度，并想知道这种设置下的 VRAM 占用情况。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1206096778823213096) (10 messages🔥): 

- **Mixtral 量化难题**：`@jaredquek` 发现了一个由于过时的 **peft 版本 (0.7.1)** 引起的问题，通过升级到 **0.8** 以支持量化后的 Mixtral 解决了该问题。`@casper_ai` 确认升级后一切恢复正常。
- **LlamaFactory 采用 ShareGPT 格式**：`@faldore` 指出 LlamaFactory 在其 [仓库文档](https://github.com/hiyouga/LLaMA-Factory/blob/91d09a01ac3b5da29d284b8d51cdfe4252b391e0/data/README.md?plain=1#L89) 中采用了 ShareGPT 格式，并建议将其作为其他项目的潜在增强功能。
- **关于命名规范的讨论**：在讨论 LlamaFactory 采用 ShareGPT 格式时，`@le_mess` 表示倾向于不直接为这类解决方案使用 ShareGPT 这个名称。
- **澄清 Tools Description**：`@nanobitz` 询问了 "tools description" 的用途，`@faldore` 回答说这是为了 LlamaFactory 文档上下文中的 functions（函数）设计的。

**提到的链接**：

- [Standards](https://xkcd.com/927/)：未找到描述
- [LLaMA-Factory/data/README.md at 91d09a01ac3b5da29d284b8d51cdfe4252b391e0 · hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/91d09a01ac3b5da29d284b8d51cdfe4252b391e0/data/README.md?plain=1#L89)：易于使用的 LLM 微调框架 (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM) - hiyouga/LLaMA-Factory

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1206028366994481232) (27 messages🔥): 

- **使用历史数据微调聊天模型**：`@smithclay` 询问了如何使用大量历史报纸文章对聊天模型进行微调，并质疑微调是否必须使用对话式数据集。`@yamashi` 建议可以从历史数据中生成 Q/A pair（问答对）数据集，并指导模型匹配古旧风格。

- **生成高质量模型时的成本担忧**：用户 `@noobmaster29` 对生成微调数据集的建议反应为 "Rip wallet"（钱包没命了），暗示了对成本的担忧，`@yamashi` 对此作出了回应，暗示了处理 Large Language Models (LLMs) 所需的资金投入。

- **成本效益的替代方案**：`@nafnlaus00` 建议在消费级硬件上使用量化版本的 Mixtral 以节省开支，并描述了一种通过提供 Prompt 来使模型回复具备 1800 年代风格的微调方法。

- **微调实践与成本效率辩论**：当 `@smithclay` 阐明其对建议的多阶段微调过程的理解时，`@dangfutures` 赞同了 `@yamashi` 之前的观点，即不要为了节省成本而牺牲模型质量。

- **寻求本地服务器配置指导**：`@siafu7795` 寻求关于如何使用 Helix 的特定配置来训练本地运行的 Mistral-7b-instruct 服务器的帮助，`@le_mess` 最终确认按照 axolotl GitHub 仓库的说明操作应该是可行的。

- **微调学习资源请求**：`@formidoboi` 表示自己是微调领域的新手，并向社区征求学习该过程的资源，但在给定的历史记录中没有收到回复。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1180827321704390657/1180827321704390660/1206197887575531561)：Discord 是进行语音、视频和文字交流最简单的方式。
- [helix/api/pkg/dataprep/qapairs/qapair_config.yaml at main · helixml/helix](https://github.com/helixml/helix/blob/main/api/pkg/dataprep/qapairs/qapair_config.yaml)：通过微调开源模型创建你自己的 AI - helixml/helix
- [axolotl/helix-mistral-instruct-v1.yml at new-long-running · lukemarsden/axolotl](https://github.com/lukemarsden/axolotl/blob/new-long-running/helix-mistral-instruct-v1.yml)：尽管提问。通过在 GitHub 上创建账号为 lukemarsden/axolotl 的开发做出贡献。

### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1206045637657759794) (1 messages): 

- **周末学习 Text-to-SQL**：`@jerryjliu0` 发布了一个关于高级 Text-to-SQL 编排的新**视频教程**。该 [YouTube 视频](https://www.youtube.com/watch?v=L1o1VPVfbb0) 题为 *"LLMs for Advanced Question-Answering over Tabular/CSV/SQL Data (Building Advanced RAG, Part 2)"*，引导观众在表格数据上构建从简单到高级的查询流水线（query pipeline）。

**提到的链接**：

[LLMs for Advanced Question-Answering over Tabular/CSV/SQL Data (Building Advanced RAG, Part 2)](https://www.youtube.com/watch?v=L1o1VPVfbb0)：在本系列的第二个视频中，我们将向您展示如何在表格数据上构建一个从简单到高级的查询流水线。这包括使用 LLM 来推断...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1205920247379660800) (5 messages): 

- **探索 RAG 的多跳查询 (Multi-Hop Queries)**：回答多跳查询的能力是高级检索增强生成 (RAG) 系统迈出的重要一步。Tang 等人的工作引入了第一个[多跳查询数据集](https://t.co/Hqx1KKOqYv)，以辅助高级 RAG 模型的基准测试。
- **微调 Mistral-7B**：来自 @helixml 的 @lmarsden 讨论了微调 [Mistral-7B 以记忆知识](https://t.co/J6JP1gycJm)的潜力，这可以使模型在不依赖 RAG 的情况下对复杂问题进行推理，该话题最近在 Hacker News (HN) 上引起了关注。
- **表格数据问答微型课程**：LlamaIndex 的新微型课程详细介绍了如何构建结合 Text-to-SQL 与 RAG 的查询流水线。该课程展示了构建[从简单到高级查询流水线](https://t.co/BS0VkZjbZI)的三个复杂度级别。
- **在高级 RAG 中实施 Guardrails**：对于面向用户的应用，设置高级 RAG 涉及内容审核、主题引导和幻觉预防的额外层。这些[输入/输出过滤器](https://t.co/THRzAAFtF0)对于维持质量和安全性至关重要。
- **表格数据理解高级技术研讨会**：最新的研讨会聚焦于使用 LLM 进行高级表格数据理解，介绍了两篇论文及其作者，包括关于 [Chain-of-Table](https://t.co/MpKxqK33AN) 的论文，并列出了贡献研究人员及其相关工作的完整名单。

**提到的链接**：

[Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://t.co/MpKxqK33AN)：基于表格的 LLM 推理是解决许多表格理解任务（如基于表格的问答和事实验证）的一个有前景的方向。与...相比。

  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1205951630323879976) (53 条消息🔥): 

- **探索 Text to SQL 能力**：用户 `@ottomation` 询问如何尝试使用 LlamaIndex 为未记录的 SQL 表生成元数据和数据字典。他们正在寻求关于如何利用列名、Schema 和数值来生成列描述的帮助。
- **寻求聊天会话连续性解决方案**：`@tbhaxor.com` 询问如何像 ChatGPT 一样在多个窗口间保持聊天会话的上下文。`@cheesyfishes` 提供了解决方案，并附上了关于 [chat stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores) 的文档链接，并建议使用 `SimpleChatStore`。
- **澄清 LlamaIndex 的定价和可用性**：有人询问 LlamaIndex 是否免费且开源，`@cheesyfishes` 确认它是开源的，并指向官方网站以获取 [更多信息](https://www.llamaindex.ai/)。
- **高效的关键词提取策略**：用户 `_shrigmamale` 寻求关于从文本中提取如 "last years"、"excels"、"sales" 等关键词的帮助。`@bin4ry_d3struct0r` 建议针对此类任务使用 Prompt Engineering。
- **用于测试 Vector Stores 的 Mock 对象**：`@7leven` 正在寻找用于测试 Vector Stores 的虚拟对象。`@cheesyfishes` 提供了一个使用 `Document.example()` 创建静态文档以进行测试操作的代码片段。

**提到的链接**：

- [LlamaIndex - Data Framework for LLM Applications](https://www.llamaindex.ai/)：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到 Large Language Models (LLMs)。
- [Chat Stores - LlamaIndex 🦙 v0.10.1](https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores.html#chat-stores)：未找到描述
- [LlamaIndex - Features, Pricing &amp; Use Cases](https://www.toolsforhumans.ai/ai-tools/llamaindex)：LlamaIndex 是一款旨在增强 Large Language Model (LLM) 应用的数据管理软件。它通过用户友好的查询接口简化了数据摄取、索引和数据分析...

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1205939325133393940) (2 条消息): 

- **深入探讨视频处理创新**：`@andysingal` 分享了一篇题为《Video Revolution: GPT4V and LlamaIndex Unleashed》的文章，讨论了融合 OpenAI GPT4V 与 LanceDB VectorStore 的突破性 **Multimodal RAG 架构**。该文章预示了我们在与视频内容交互时的效率和多功能性将迎来新浪潮。[阅读更多](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30)
- **Whisper 获得性能飞跃**：`@denverbitch` 开发了一种能显著提高 Whisper 速度的技术，并愿意在编写文档或回答与其增强功能相关的问题方面进行合作。

**提到的链接**：

[Video Revolution: GPT4V and LlamaIndex Unleashed](https://ai.gopubby.com/video-revolution-gpt4v-and-llamaindex-unleashed-329d5a9ebf30)：Ankush k Singal

  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1205811906057871390) (23 messages🔥): 

- **寻求解决方案**：用户 `@vithan` 请求关于 **scktlearn and pandas** 的帮助，表示通过文字难以解释清楚问题，并请求 `@himalaypatel.` 进行语音通话以提供更好的协助。
- **教程时间**：`@a404.eth` 分享了一个名为 "Unlock the Power of LangChain: Deploying to Production Made Easy" 的 **YouTube tutorial**，展示了如何使用 LangChain 和 **UnstructuredIO** 将 PDF RAG 部署到 **DigitalOcean** 生产环境。为移动端用户提供了独立链接：[在此观看教程](https://youtu.be/CbBIwVxjdP8)。
- **无限循环疑问**：`@vvm2264` 描述了一个关于论文生成 **agent** 的挑战，该 **agent** 似乎在无限次地重复使用某个工具，可能触及了 OpenAI 的 rate limits，并征求关于如何防止这种行为的建议。
- **带伤编程**：在 `@_adjekofori` 透露腿部骨折后，`@johnny2x2` 幽默地建议这留出了更多时间来写代码，并分享了他们目前正在学习 **AWS**。
- **实现咨询**：用户 `@damianj5489` 询问是否存在包含 **LangChain** Python 文档示例的 notebooks 仓库，旨在进行交互式学习，而不是直接复制粘贴示例。

**提及的链接**：

- [🦜️🔗 Langchain](https://python.langchain.com): 未找到描述
- [LangChain](https://www.langchain.com/): LangChain 灵活的抽象和广泛的工具包使开发者能够构建具有上下文感知和推理能力的 LLM 应用。
- [Unlock the Power of LangChain: Deploying to Production Made Easy](https://youtu.be/CbBIwVxjdP8): 在本教程中，@FocusedLabs 的 CEO 兼联合创始人 Austin Vance 将指导你如何将 LangChain 的 PDF RAG 部署到生产环境！在这段引人入胜的...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1206033805270585415) (1 messages): 

- **关于禁用 Playground 功能的咨询**：用户 `@gitmaxd` 寻求关于如何使用代码片段 `add_routes(app, my_app_chain, disabled_endpoints=["playground"])` 在已部署的端点上 **disabling the playground** 的建议。在给定的消息中没有提供对该查询的回复。
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1205916966230167583) (8 messages🔥): 

- **Selfie 实验启动**：`@dondo.eth` 介绍了一个名为 **Selfie** 的开源项目，旨在通过兼容 OpenAI 的 API 利用个人数据增强文本生成，强调上下文丰富的输出。该项目的代码库可以在 [Selfie on GitHub](https://github.com/vana-com/selfie) 进行贡献或测试。

- **Intellifs Python 库亮相**：`@synacktra` 创建了 **Intellifs**，这是一个受 aifs 库启发的新 Python 库/工具，支持本地语义搜索。该工具在 GitHub 上开放贡献：[Intellifs Repository](https://github.com/synacktraa/intellifs)。

- **ArtFul 应用发布**：`@vansh12344` 宣布发布 **ArtFul - AI Image Generator**，这款应用提供了访问 Kandinsky 和 DALL-E 等多种 AI 模型的途径，用于生成原创 AI 艺术，无需注册或使用限制，通过观看广告即可完全免费使用。该应用已在 Google Play Store 上架：[ArtFul App Link](https://play.google.com/store/apps/details?id=com.projecthit.artful)。

- **Merlinn 产品揭晓**：`@david1542` 分享了 **Merlinn** 的发布，该产品旨在通过 LLM Agent 的协助并利用背后的 LangChain 技术，帮助团队迅速解决生产故障（production incidents）。更多信息请访问其官网 [Merlinn](https://merlinn.co/)。

- **Triform 平台 Beta 测试**：`@igxot` 宣布了 **Triform** 的早期 Beta 版本，这是一个用于托管和编排 Python 脚本的平台，集成了 LangChain，并邀请用户通过 Beta 测试注册免费的永久账号用于生产环境。Triform 的入门指南见 [Triform Sign Up](https://triform.ai)，其文档可通过 [Triform Docs](https://triform-docs.readthedocs.io/) 访问。

**提到的链接**：

- [100% Local Tiny Vision Model - Very Quick](https://youtu.be/jToWjbhdoEg)：在此视频中，我将介绍 Moondream1，一个 1.6b 的小型视觉和文本生成模型。GitHub 链接：▸ https://github.com/vikhyat/moondream 更多内容：▸ http...
- [GitHub - BCG-X-Official/agentkit: Starter-kit to build constrained agents with Nextjs, FastAPI and Langchain](https://github.com/BCG-X-Official/agentkit)：使用 Nextjs, FastAPI 和 Langchain 构建受限 Agent 的入门套件 - BCG-X-Official/agentkit
- [GitHub - synacktraa/intellifs: Content-Aware File System.](https://github.com/synacktraa/intellifs)：内容感知文件系统。通过在 GitHub 上创建账号来为 synacktraa/intellifs 的开发做出贡献。
- [GitHub - vana-com/selfie: Enhance text generation with personal data via an OpenAI-compatible API, seamlessly integrating with local or hosted LLMs for context-rich outputs.](https://github.com/vana-com/selfie)：通过兼容 OpenAI 的 API 利用个人数据增强文本生成，无缝集成本地或托管的 LLM 以获得上下文丰富的输出。 - vana-com/selfie
- [Merlinn - Resolve incidents fast using AI](https://merlinn.co/)：使用 AI 高效调查生产故障；通过一个了解你环境的 AI Agent 为你的团队赋能。
- [Triform - Unleashing AI Potential](https://triform.ai)：未找到描述
- [Welcome to Triform Documentation &mdash; Triform 0.1 documentation](https://triform-docs.readthedocs.io/)：未找到描述

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1206167818341584907) (2 messages): 

- **自动目标检测焦点**：`@pradeep1148` 分享了一个题为“自动目标检测”的 [YouTube 视频](https://www.youtube.com/watch?v=W4T7zHluzaM)，重点介绍了如何使用 MoonDream Vision Language Model 进行 Zero-shot 目标检测。
- **使用多种工具与文档聊天的教程**：`@datasciencebasics` 发布了一个 [教程视频](https://youtu.be/2IL0Sd3neWc)，提供了使用 ChainLit, LangChain, Ollama 和 Mistral 创建 RAG（检索增强生成）UI 的指南。

**提到的链接**：

- [Automatic Object Detection](https://www.youtube.com/watch?v=W4T7zHluzaM)：我们将看到如何使用 Zero-shot 目标检测和 moondream 视觉语言模型进行自动目标检测 #llm #ml #ai #largelanguagem...
- [Chat With Documents Using ChainLit, LangChain, Ollama &amp; Mistral 🧠](https://youtu.be/2IL0Sd3neWc)：在此视频中，我将演示如何在本地计算机上创建一个简单的 RAG UI。你可以通过克隆...来跟随我操作。

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1205870669716267020) (18 messages🔥): 

- **订阅模式说明**：`@djioliat` 询问了 Mistral Discord 聊天机器人的订阅模式。`@mrdragonfox` 澄清说这是**一个面向所有用户的模型**，按使用的 token 计费，并且部署会根据需要进行扩展，不为个人用户提供定制部署。

- **Mistral 7B 的资源需求**：`@mihail2132` 询问了运行 Mistral 7B 0.2 的 RAM 需求，表示笔记本电脑上的 40GB RAM 不够用。`@donjuan5050` 建议使用量化模型，例如 [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) 上提供的模型，仅需几个 GB 的 RAM。

- **关于量化模型的讨论**：`@mrdragonfox` 回应指出**量化模型并不适用于所有用例**。他们强调像 7b 这样的小型模型可以在 fp16 下运行，并且仍能提供不错的性能。

- **MistralAI 对结构化输出的准备**：`@mrdomoo` 询问 MistralAI 计划如何处理结构化输出（Structured Output）。`@mrdragonfox` 引用了之前的一条消息作为参考，暗示该问题早前已得到解决。

- **澄清 RAM 使用情况**：`@mihail2132` 寻求关于标准模型预期 RAM 使用量的澄清，而 `@sublimatorniq` 认为 **40GB RAM 应该足够了**，`@mrdragonfox` 补充说这取决于模型和 batch size。

**提到的链接**：

[TheBloke/Mistral-7B-Instruct-v0.2-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)：未找到描述

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1205855711851057153) (2 messages): 

- **生产环境中的 GPU 投资与租赁**：`@i_am_dom` 表示，出于成本效益考虑，使用来自 Google 的 GPU 对于生产环境来说不是一个可行的策略。他们解释说，从长远来看，拥有像 **A100s 40GB** 这样的硬件可能更经济。

- **GPU 所有权成本分解**：`@i_am_dom` 继续分析了 GPU 的所有权成本，解释说在 **70000 个计算单元**之后，购买 GPU 的费用就能回本（不含电费）。这相当于大约**半年**的持续使用。
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1205985547772756060) (1 messages): 

- **关于 Mistral AI 的 Docker 设置咨询**：用户 `@norphiil` 询问社区是否有人创建了 `docker_compose.yml` 以简化 **Mistral AI** 作为 Docker REST API 的部署。他们请求协助并预先感谢提供帮助的人。
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1206238280480325682) (3 messages): 

- **Mistral 聊天机器人安装成功**：用户 `@1mbc` 报告在 Cloudfare AI maker 上成功安装了 **Mistral**，但观察到该模型无法识别自己的来源。
- **ChatGPT 的自我意识对比**：作为回应，`@dawn.dusk` 向 `@1mbc` 保证，像 **GPT-4 和 Mistral** 这样的模型没有自我意识是正常的，就像 GPT-4 不知道自己的身份一样。
- **学习使用 Mistral**：针对 `@1mbc` 关于构建个性化助手的第一步的问题，`@dawn.dusk` 提供了一个关于使用 **ChatGPT** 的 [Datacamp 教程链接](https://www.datacamp.com/tutorial/mistral-7b-tutorial)，其中包括编写 prompt 和探索用例。
- **个人助手开发建议**：`@tom_lrd` 建议使用 Mistral 创建“个人助手”非常复杂，建议从更简单的任务开始，并暗示考虑使用 **Retrieval Augmented Generation (RAG)** 进行模型的数据集成。

**提到的链接**：

[Mistral 7B Tutorial: A Step-by-Step Guide to Using and Fine-Tuning Mistral 7B](https://www.datacamp.com/tutorial/mistral-7b-tutorial)：该教程涵盖了访问、量化、微调、合并和保存这个强大的 73 亿参数开源语言模型。

  

---

### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1205902151155454012) (4 条消息): 

- **Discord 上的协作聊天机器人体验**：用户 `@jakobdylanc` 介绍了一个为 Discord 设计的协作式 LLM 提示功能，允许用户与朋友一起与 OpenAI、Mistral 等 LLM 进行对话。[该机器人的 GitHub 页面](https://github.com/jakobdylanc/discord-llm-chatbot)展示了对各种 LLM 的支持、视觉支持、流式响应，并以仅 200 行代码的简洁实现为特色。
- **Mistral 7b 表现优于同类模型**：`@cognitivetech` 分享了一篇文章，展示了 *Mistral 7b Instruct v0.2 Q8 GGUF* 在创建*详尽的列举笔记*方面，如何超越了排行榜上评分更高的其他模型。详细信息可以在 [Hacker Noon](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk) 的文章中找到。
- **增强型 Web 搜索功能获得认可**：`@miscend` 认可了 @jakobdylanc 分享的方案所提供的卓越 Web 搜索功能，并将其与 LibreChat 进行了对比，同时询问了如何为 Mistral 设置不同的 API key，以便同时使用 OpenAI 和 Mistral 模型。
- **跨语言源码映射探索**：`@sublimatorniq` 简要提到了对跨语言源码映射（Cross-Language Source Mapping）的兴趣或活动，但未提供讨论的具体背景和细节。

**提到的链接**：

- [GitHub - jakobdylanc/discord-llm-chatbot: Collaborative prompting • Supports OpenAI, Mistral, ollama, oobabooga and more • Vision support • Streamed responses • 200 lines of code 🔥](https://github.com/jakobdylanc/discord-llm-chatbot)：协作式提示 • 支持 OpenAI, Mistral, ollama, oobabooga 等 • 视觉支持 • 流式响应 • 200 行代码 🔥 - jakobdylanc/discord-llm-chatbot
- [未找到标题](https://hackernoon.com/ranking-7b-gguf-for-comprehensive-bulleted-notes-with-ollama-go-home-model-rankings-youre-drunk>)：未找到描述

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1206217223010656268) (2 条消息): 

- **问候交流**：用户 `@elpo_55` 简单地说了句 "hi"。
- **API 超时问题**：`@oumar7842` 正在寻求帮助，解决 API 生成内容过长导致超时的问题。他们正在询问是否有办法解决此问题。
  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1205936226750693427) (10 messages🔥): 

- **创新的 Serverless Vector DB**: `@nuvic_` 分享了 [TurboPuffer](https://turbopuffer.com/)，这是一个基于 S3 的高性价比 Serverless Vector DB，强调了其效率和简单性，100 万个向量的热查询（warm queries）缓存大约需要 10 秒。文中将其与 LanceDb 进行了对比，但 `@nuvic_` 澄清说 TurboPuffer 的主要卖点是其基于 S3，而 LanceDb 的卖点是其开源性质和易管理性。

- **采访 Untapped Capital GP**: `@btdubbins` 向用户推荐了 [Cognitive Revolution podcast](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/) 上一段有趣的采访，嘉宾是 Untapped Capital 的 GP 兼 BabyAGI 创作者 Yohei Nakajima，讨论了集体智慧以及 AI 在增强相互理解方面的作用。

- **谷歌对 AI 颠覆的担忧**: `@swyxio` 发现了一份 2018 年具有远见的谷歌备忘录，其中将 AI 视为严重的业务风险，回想起来这似乎具有预见性。内容可以通过分享的 [TechEmails](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww) 推文访问。

- **ChatGPT 对大学录取的影响**: `@swyxio` 分享了一篇 [Forbes 文章](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/)，讨论了学生使用 ChatGPT 进行大学申请的趋势，以及由此产生的可能引起招生委员会警觉的禁用词。

- **ChatGPT 的禁用词**: `@lightningralf` 幽默地建议将禁用词列表提供给 ChatGPT，以避免在学术环境中过度使用这些词，正如 `@swyxio` 分享的前一篇 Forbes 文章中所提到的。

**提到的链接**:

- [turbopuffer](https://turbopuffer.com/): turbopuffer 是一个构建在对象存储之上的 Vector DB，这意味着成本降低 10x-100x、按需计费以及极高的可扩展性。
- [来自 Internal Tech Emails (@TechEmails) 的推文](https://x.com/techemails/status/1756765277478621620?s=46&t=90xQ8sGy63D2OtiaoGJuww): 谷歌工程师：AI 是我们业务的严重风险，2018 年 12 月 26 日。
- [你在学校申请中使用了 ChatGPT 吗？这些词可能会提醒招生官](https://www.forbes.com/sites/rashishrivastava/2024/02/05/chatgpt-college-school-applications-admissions-red-flags-ai/): 转向 ChatGPT 寻求写作帮助的学生正重新寻求人类帮助，以使作品听起来更像人类——而学校对此已经应接不暇。
- [来自东西方的 AI 与身份，对话 Untapped Capital GP 兼 BabyAGI 创作者 Yohei Nakajima](https://www.cognitiverevolution.ai/ai-identity-from-east-west-with-yohei-nakajima-gp-at-untapped-capital-and-babyagi-creator/): 在今天的节目中，Untapped Capital 的 GP 兼 BabyAGI 创作者 Yohei Nakajima 回到节目中讨论集体智慧、身份以及 AI 如何……

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1205956859459018822) (8 messages🔥): 

- **Hugging Face 遭遇宕机**: `@_jp1_` 报告 **Hugging Face (HF)** 宕机，表明社区内可能对 HF 服务存在依赖。
- **讨论 HF 作为关键基础设施的角色**: `@philipmay` 质疑 **Hugging Face** 是否可以被视为关键基础设施，引发了关于依赖外部平台进行模型存储和运营的讨论。
- **考虑 Hugging Face 的替代方案**: `@_jp1_` 提到了过去尝试将基础设施转移到 **S3** 以存储权重、结果和数据集，但发现尽管存在潜在的可靠性问题，HF 的免费集成服务更方便。
- **对 Hugging Face 未来货币化的担忧**: `@philipmay` 推测 **Hugging Face** 未来可能会开始对模型访问或下载收费，表明社区需要考虑财务影响。
- **Phantine 分享关于算法的想法**: `@phantine` 提到了一个没有具体细节的算法想法，引用了对稀疏性（sparsity）的高效利用，并指向了一个对话以进一步证明；然而，提供的链接无法检索 (`<<<null>>>`)。

**提到的链接**:

[‎Gemini - 激发灵感的聊天工具](https://gemini.google.com/app/d208129c63f63536): Bard 现在更名为 Gemini。从 Google AI 获取写作、规划、学习等方面的帮助。

  

---

### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1205879938717327480) (1 messages): 

- **在德语上尝试 SPIN**: `@philipmay` 询问 SPIN (self-play) 方法是否适用于 **Mixtral 模型** 的德语。他们分享了 SPIN 技术实现的 [官方 GitHub 链接](https://github.com/uclaml/SPIN)。

**提到的链接**:

[GitHub - uclaml/SPIN: The official implementation of Self-Play Fine-Tuning (SPIN)](https://github.com/uclaml/SPIN): Self-Play Fine-Tuning (SPIN) 的官方实现 - uclaml/SPIN

  

---



### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 messages): 

rabiat: 有趣的想法 🙂
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1206310598732812408) (3 messages): 

- **OpenAI 发布预告**: `@res6969` 暗示可能很快会有新的 **OpenAI 发布**，并建议时间范围在 **明天或周二**。
- **聚会中提到的消息来源**: 关于即将发布的消息是由 `@res6969` 分享的，他是在 **聚会上听人说起的**。
- **用户期待感增加**: 针对这一消息，`@.psychickoala` 表达了好奇心，俏皮地问道 **"是什么呀 哈哈"**，以了解更多关于推测发布的细节。
  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1205796795171610684) (2 messages): 

- **<@748528982034612226> 近况如何？**: 用户 `@teknium` 对 `<@748528982034612226>` 目前在做什么表示好奇。
- **<@748528982034612226> 处于离线状态**: 作为回应，`@atlasunified` 提到 `<@748528982034612226>` 一直 **处于离线状态 (off grid)**，没有提供更多细节。
  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=W4T7zHluzaM
  

---



---