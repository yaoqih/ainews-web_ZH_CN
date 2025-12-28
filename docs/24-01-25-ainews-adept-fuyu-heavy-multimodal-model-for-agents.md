---
companies:
- adept
- hugging-face
- deepseek
- mistral-ai
- nous-research
date: '2024-01-25T21:30:23.929279Z'
description: '**Adept** 发布了 **Fuyu-Heavy**，这是一款专注于 UI 理解和视觉问答（Visual QA）的多模态模型，其在
  MMMU 基准测试中的表现超越了 **Gemini Pro**。该模型采用了 **DPO**（直接偏好优化）技术，该技术作为一种领先的微调方法正备受瞩目。Fuyu-Heavy
  的具体规模尚未公开，但估计参数量在 **200 亿至 1700 亿**之间，小于 **Claude 2**、**GPT-4V** 和 **Gemini Ultra**
  等传闻中的前沿模型。


  与此同时，**Mamba** 因质量问题被 ICLR 拒收。在 Discord 的讨论中，有人声称 **DeepSeek Coder 33B** 在编程任务上的表现优于
  **GPT-4**，此外还探讨了 **Yi-34B-200K** 和 **Goliath-120B** 等大模型的部署策略。关于量化的辩论凸显了对 **Q8**
  和 **EXL2 量化**的不同看法。讨论内容还涉及 **Mistral 7B Instruct v0.2** 的微调与指令微调，以及关于 RMS 优化和结合了
  **Transformer** 与**选择性状态空间模型（Selective SSM，即 Mamba）**的异构 AI 架构的见解。此外，**RWKV** 等循环大语言模型（Recurrent
  LLMs）的潜力以及**对比偏好优化（CPO）**等技术也受到了关注。'
id: 58f3574c-ce94-4006-b73f-944f6a76e291
models:
- fuyu-heavy
- fuyu-8b
- gemini-pro
- claude-2
- gpt4v
- gemini-ultra
- deepseek-coder-33b
- yi-34b-200k
- goliath-120b
- mistral-7b-instruct-v0.2
- mamba
- rwkv
original_slug: ainews-adept-fuyu-heavy-multimodal-model-for
people: []
title: Adept Fuyu-Heavy：面向智能体（Agents）的多模态模型
topics:
- multimodality
- visual-question-answering
- direct-preference-optimization
- benchmarking
- model-size-estimation
- quantization
- model-merging
- fine-tuning
- instruct-tuning
- rms-optimization
- heterogeneous-ai-architectures
- recurrent-llms
- contrastive-preference-optimization
---

<!-- buttondown-editor-mode: plaintext -->> 2024年1月24日的 AI Discord 动态。我们为您检查了 **20** 个公会、**297** 个频道和 **3025** 条消息。预计节省阅读时间（以 200wpm 计算）：**295 分钟**。

轮到 Adept 带来一场[引人注目的发布](https://www.adept.ai/blog/adept-fuyu-heavy)了：

 
![image.png](https://assets.buttondown.email/images/db04109d-dc48-4921-bc96-012766913818.png?w=960&fit=max)
 

重点似乎在于 UI 理解，考虑到 Adept 的业务，这作为核心关注点是合理的。[演示视频](https://vimeo.com/906055649)展示了对 7 张 UI 截图非常出色且精确的视觉问答（visual QA），但由于是在 gradio 界面上进行的，并未展示 Adept 产品的其他部分。Fuyu 还使用了 DPO，DPO 突然间成为了简短的 [DPO vs IPO vs KTO 之争](https://huggingface.co/blog/pref-tuning)中推定的赢家。Fuyu-Heavy 在新的 MMMU 基准测试中击败了 Gemini Pro，但目前尚不清楚 GPT4V 在此项测试中的表现（有人跑一下吗？）

有几个人[指出](https://twitter.com/teortaxestex/status/1750353889499459746)了关于 Fuyu-Heavy 与 Claude 2、GPT4-V 以及 Gemini Ultra 尺寸对比的旁注，因为这些细节并未公开，且 Adept 本身实际上甚至没有提到他们自己的模型大小（只知道它比 [Fuyu-8B](https://www.adept.ai/blog/fuyu-8b) 大，仅此而已）。假设那些前沿模型处于传闻中的 [400B](https://www.lesswrong.com/posts/iQx2eeHKLwgBYdWPZ/retrospective-on-gpt-4-predictions-after-the-release-of-gpt) 到 [1.7T](https://twitter.com/swyx/status/1671272883379908608?lang=en) 参数范围，那么缩小 10-20 倍意味着 **Fuyu-Heavy 的上下界大约在 20B-170B 之间**。

**其他新闻方面**，Mamba 被 [ICLR 拒绝，理由是“不够好”](https://twitter.com/srush_nlp/status/1750526956452577486)。笑死？


---

**目录**

[TOC] 


# 第 1 部分：Discord 高层级摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **代码模型对决：用户模型 vs GPT-4**：一位名为 `@rombodawg` 的用户声称他们的新编码模型在编码挑战测试中表现优于 **GPT-4**，并指出其合并模型中集成了 **DeepSeek Coder 33B** 的部分内容。

- **优化 LLM 部署**：围绕部署 **Yi-34B-200K** 和 **Goliath-120B** 等大型模型展开了大量讨论，`@super.deap` 和 `@aikitoria` 等用户正在寻求低成本、快速推理以及适配 **80GB VRAM 设置**的策略。

- **走出量化困境**：关于量化的讨论反映出复杂的情绪；`@keyboardking` 报告称 **Q8** 使模型变得毫无价值，但 `@kquant` 展示了其有效使用的案例，并分享了对 **EXL2 量化**的认可。

- **掌握模型合并**：对话包括对模型合并策略的见解，`@alphaatlas1` 指出当合并模型权重总和为 **1.0 总权重**时效果最佳，并注意到超过 **1.4 阈值**时的缺陷。

- **关于微调的疑问**：`@nigelt11` 等用户寻求关于 **Mistral 7B Instruct v0.2** 在标签和提示词格式方面的澄清，一篇分享的 **Medium 文章**为这些实践提供了额外指导。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **探索 AI Chatbot 潜力**：讨论了将 AI 转化为能够与 API 交互的 Chatbot 的可能性。然而，目前尚未确定具体的产品。

- **Mistral 模型讨论**：辩论了更长的序列和更大的 batch sizes 对 **Mistral** 等模型的潜在益处。寻求关于模型 fine-tuning 与 instruct-tuning 之间区别和应用的澄清，通常 instruct tuning 紧随 fine-tuning 之后。还询问了 fine-tuning Mistral 的正确数据格式，表明对与 instruct-tuning 相关的 instruction-format 存在困惑。

- **LLM 训练见解**：在最近的工作中，使用 RMS 优化被指出比 layernorm 能改善 loss 结果。表达了如果成功将开源实现的预期，以及很快评估模型性能的意图。

- **异构 AI 与模型进展**：一篇 [LessWrong 文章](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and) 讨论了结合 **Transformers** 和 **Selective SSM (Mamba)** 的 **异构 AI 架构**。提到了 *DoraemonGPT* 在动态视频理解方面的能力，RWKV 等循环 LLM 在跟踪角色状态方面的潜力，以及中等规模 LLM 在翻译中的 **Contrastive Preference Optimization (CPO)**。此外，**Adept Fuyu-Heavy** 作为一种专为数字 Agent 设计的新型多模态模型被推出，根据 [Adept.ai 博客文章](https://www.adept.ai/blog/adept-fuyu-heavy)，尽管其体积较小，但在某些基准测试中优于更大的模型。

- **操控 LLM 与硬件讨论**：对语言模型的 activation steering 表现出兴趣，并辩论了使用动态模板的不同 prompt 构建方法的效率。GPU 能力讨论强调了在 1080 ti 和 3060 等各种硬件上运行 AI 模型的潜力。

- **Project Obsidian 准备升级 v2**：宣布 **Project Obsidian** 升级至 v2，选择 **stableLM 2 1.6B** 作为首选模型。社区反应积极，并分享了一个用于视觉文档理解中 zero-shot 泛化的资源：**InstructDoc 数据集** ([GitHub - nttmdlab-nlp/InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc))。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **ChatGPT 创意征集**：`@abdubs` 邀请用户分享 **ChatGPT** 在创意和实用场景中的独特应用，旨在了解该技术提供的广泛益处。

- **GPT-4 图像 Prompt 见解与替代方案**：`@lugui` 澄清 **GPT-4** 是基于图像描述构建 prompt，而不是直接生成新图像。对于图像操作，`@xenowhiz` 建议使用 **Code Interpreter** 及其模块。

- **AI 播客推荐**：`@fran9000` 推荐了 "[The AI Breakdown](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614)"，提供关于 AI 多个方面的每日新闻和辩论。

- **对旧版 GPT 版本的偏好**：`@lzgodhook13` 询问是否可以恢复到 **2022 年 5 月至 6 月的 GPT-3 版本**，理由是在简单任务上表现更好。

- **GPT-4 更新后的 Context Window 问题**：用户报告更新后 context window 减小，影响了自定义模型的性能。已创建 Bug 报告，但未提供直接链接。

- **Prompt Engineering 交流**：讨论了对 ChatGPT 默认列表输出的挫败感以及规避策略（包括 negative prompting），同时 `@brayheart` 提议组建团队参加 prompt engineering 黑客松。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Pro 会员权益与优惠码风波**：用户讨论了升级到 **Perplexity Pro** 的优势，强调了其**相关性检查 (relevancy checks)** 功能，并建议配合 **GPT-4** 使用以获得更佳体验。然而，部分用户在应用优惠码时遇到问题，解决方案包括参考 Redditor **u/ArakDemonBlade** 提供的步骤，或联系支持邮箱 **pro@perplexity.ai**。

- **比较 Perplexity AI Companion 的能力**：关于 **Perplexity AI Companion** 在 **GPT-4** 后续提问中保留来源的能力引发了辩论，用户对其在浏览器扩展程序与 iOS App 之间的功能差异持不同看法。

- **在线 LLM 与 Google Drive 的限制与特性**：用户 **@mantas_82008** 询问了如何提高在线 **LLM models** 每分钟 **10 次**的限制。其他用户分享了访问大型 **Google Drive** 文档的修复方法，包括下载和转换建议，并讨论了如何策略性地使用 Perplexity 的 Prompt 框。

- **深入探讨 Perplexity 的设计洞察**：分享了一个由 Perplexity AI 设计负责人 **Henry Modisett** 出镜的 **YouTube 视频**，概述了 AI 设计的细微差别以及该领域的求职建议。此外，Perplexity 的 **Copilot** 功能也因其实用性受到赞赏，并附带了关于该功能如何告知用户全球趋势的链接。

- **API 洞察与积分奖励揭秘**：当 **PPLX 70B** 被确认为与 Copilot 的 **"Experiment" 功能**等效的模型时，用户对 Perplexity 的 API 表现出极大热情。关于浏览器与 API 响应差异的解释提示了系统提示词 (system prompts) 或来源的差异。此外还提到了 5.00 美元的积分奖励，在账户自动充值至少 2 美元后即可激活。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **切换尝试：AI 运算从 CPU 转向 GPU**：工程师们讨论了将 AI 运算从 CPU 切换到 GPU 使用的过程，建议检查聊天页面设置。Linux 用户，特别是那些使用缺乏 AVX2 支持的老旧处理器的用户，正在寻找 LM Studio 的替代方案，并建议编译支持语言模型加载的 [llama.cpp](https://github.com/ggerganov/llama.cpp)。

- **在 HuggingFace 上寻找定制模型**：成员们对 HuggingFace 上的模型选择表示困惑，强调由于文档极少，缺乏明确的独特销售主张 (USPs)。推荐使用 [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 来比较模型性能。

- **配置 GPU 层以增强模型性能**：硬件频道的讨论集中在配置 GPU 层以优化模型性能。他们评估了使用 `n_gpu_layers` 和 `num_layers` 设置来改进处理效率，尽管面临系统 RAM 利用不足以及非 AVX2 指令支持的兼容性等问题。

- **RAG 与 Embeddings API 成为 AI 焦点**：讨论重点在于使用检索增强生成 (RAG) 以及在没有 NVIDIA GPU 的情况下使用 OpenAI Embeddings API 的可能变通方法。分享了一个使用 RAG 从 PDF 读取内容的代码片段和相关的 GitHub 仓库。对话还延伸到通过 [HuggingFace 模型](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp) 探索 RAG，以及引用 [Databricks 术语表条目](https://www.databricks.com/glossary/retrieval-augmented-generation-rag) 对 RAG 进行的解释。

- **桥接 LM Studio 与 Open Interpreter**：讨论了将 LM Studio 推理与 memgpt 及 Open Interpreter 集成的尝试，重点在于 memgpt 的服务器是否可以模拟 OpenAI 的聊天和补全调用功能。这表明 AI 社区正在持续探索系统间的互操作性。

- **提示词难题与集成试验**：成员们在未提供具体背景的情况下征求改进 Prompt 的建议，并分享了将 LM Studio memGPT 与 OpenAI 集成时遇到的持续挑战，反映了模型开发中对跨平台兼容性和有效 Prompt 策略的广泛兴趣。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

- **EleutherAI 助力开源 AI 研究**：EleutherAI 与 National Science Foundation 合作启动了 National AI Research Resource，旨在提供更多的 AI 资源访问权限。他们还在 AI 研究方面取得了进展，包括 GPU 资助和开发 [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox)，该库以其在各种高性能计算平台上的可扩展性而闻名。

- **LM 的许可格局与法律问题**：社区内的讨论表明，关于 GitHub 仓库在模型训练方面的许可存在模糊性，建议从咨询律师到研究当地版权法不等。与此同时，针对 CoPilot 诉讼的担忧得到了回应，即法律程序可能会持续很长时间。

- **算法年鉴**：围绕数据质量优于数据量（以 Wavenet 数据为例）、Rust 的新框架（如 [Burn](https://burn.dev/)）的潜力和陷阱，以及先进技术（如 MambaByte [token-free Language Modeling](https://arxiv.org/abs/2401.13660) 和生产环境中模型持续更新中的 **Elastic Weight Consolidation**）等话题展开了激烈的辩论和考察。

- **解读深度学习指令**：可解释性讨论强调了在 **Sparse Autoencoder 空间中绘制 decoder weights**，并关联了 Anthropic 团队的研究更新，这些更新侧重于 attention superposition 和 **MNIST 上的 dictionary learning** 等发现。一份关键研究更新中的显著拼写错误显示了社区对细节的高度关注。

- **GPT-NeoX 开发深度探讨**：对话围绕模型训练中的 tensor+expert parallelism 等技术层面展开，并确认其与 DeepSpeed 的实现相似。社区成员还在进一步调查一个与 CUDA 初始化相关的持续性 [DeepSpeed issue](https://github.com/microsoft/DeepSpeed/issues/1987)。

- **扩展与特殊频道**：提到克服原始 scaling laws 论文中的局限性，暗示成功训练了 **1.3B 参数规模**的模型，并指引到关于 1b 参数模型的讨论，使用特定频道 `<#1129489948710539334>` 进行深入分析。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **Axolotl v0.4.0 准备部署**：OpenAccess AI Collective 宣布发布 [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/releases/tag/v0.4.0)，该版本引入了对新模型的支持、大量错误修复，并对 56 位贡献者和 A16Z 资助表示感谢。

- **模型训练的奥秘与弊病**：用户讨论了模型训练中的挑战和最佳实践；从确保 **Mamba** 与 **Lora** 的兼容性，到高效保存模型训练步骤的策略。一位用户在上传到 **Hugging Face** 时遇到困难，而另一位用户则在寻求专门针对领域自适应预训练 **CLIP** 模型的指导。

- **GPU 采购决策讨论**：关于 **GPU Purchasing Decisions** 的对话成为焦点，用户比较了 8 个 **H100** 与 16 个 **A100** GPU 的优劣，并考虑了硬件设置中的 VRAM 等因素。

- **医疗数据集宝库与机器学习困境**：分享了一个包含多模态 QA 数据集的 [GitHub 仓库](https://github.com/abachaa/Existing-Medical-QA-Datasets)，同时用户还在努力解决从获取**计算资源资金**到模型推理的 **alpaca format prompt** 技术细节等问题。

- **Serverless 带来的冷启动担忧**：`@dreamgen` 对 **large models** 在 serverless 部署中冷启动时间的现实情况表示担忧，特别是考虑到过去提供商**不缓存模型或 docker images** 的挑战。这突显了实际 AI 部署中一个紧迫的性能问题。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Lumiere 的时空魔法**：Google 推出的 **Lumiere 模型** 是一种时空扩散模型（space-time diffusion model），可以根据文本生成视频。正如 [Google 的详细报告](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/) 所述，它具有令人印象深刻的图像修补（inpainting）能力。

- **Google AI 代码的透明度之争**：社区对 Google 迟迟不发布 AI 相关代码表示担忧，这使得复现他们的研究对社区来说是一个挑战。

- **让 AI 更聪明的自我指令（Self-Instructing）**：分享了一种名为 **Self-Instruct** 的创新方法，旨在通过自我生成的指令来增强大语言模型，从而可能提高 AI 自我引导知识的能力（[Self-Instruct 论文](https://arxiv.org/abs/2212.10560)）。

- **Discord 迎来 AI 聊天机器人竞争者**：有人发出了实现利用大语言模型的 Discord 聊天机器人的邀请，代码可在 [GitHub](https://github.com/jakobdylanc/Discord-LLM-Chatbot) 上获取。

- **AI 学者的舞台**：**Latent Space** 公会正在利用 Discord 的新 Stage 功能来促进论文讨论。最近的一次会议有 40 人参加，非常成功。计划下一步讨论 [Pythia 论文](https://arxiv.org/abs/2304.01373)，并参考相关的 [Twitter 线程](https://twitter.com/rasbt/status/1734920232173539796) 中的见解。

- **不错过任何 AI 动态**：邀请成员通过在 [此处](https://lu.ma/ls) 注册并订阅日历，以了解未来的 Latent Space 活动。

- **RestGPT：将 LLM 作为 RESTful 控制器**：重点介绍了 *RestGPT* 项目，该项目探索了通过 **RESTful APIs** 控制现实世界应用程序的 **基于 LLM 的自主代理（autonomous agents）**，托管在 [GitHub](https://github.com/Yifan-Song793/RestGPT) 上。



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **CUDA 职位未获回应**：`@ziper_rom1` 在申请 CUDA 开发人员职位一个月后未收到回复；社区见解暗示这可能意味着被拒绝。该职位可能已经填补，正如一条宣布 Nvidia 新员工入职的 [Twitter 帖子](https://twitter.com/sandeep1337/status/1744399578269352293?t=6Vm_qBKAAgBHLHyVPwK13g&s=19) 所暗示的那样。

- **Mistral 的速度瓶颈**：`@duck` 分享了 **Mistral** RAG 的耗时基准测试，显示使用 llama.cpp 时，样本时间为 17.05 ms，评估时间为 175910.47 ms，这对于预期的用例来说被认为太慢了。

- **Mixtral 巨大的内存占用**：`@l0gr1thm1k` 在 NVIDIA T4 上部署 Mixtral 时遇到了 **CUDA 内存错误**，其中 4bit 量化版本的内存需求超过了预期的 24GB。

- **微调：超越 BLEU 和 ROUGE**：`@bishwa3819` 正在 Dolly 数据集上微调 Mistral-7B，并询问 **BLEU 和 ROUGE 指标** 是否足以评估语言模型在这些特定数据上的性能。

- **由 Mistral 驱动的 Reddit Copilot 机器人**：`@hugoduprez` 使用 Mistral 创建了一个 **Reddit copilot 机器人**，其运行速度非常快。`@shashank.f1` 分享了一种使用 **A-JEPA 神经模型** 进行音频理解的新方法，并展示了一个关于使用该模型从音频文件中提取语义的 [YouTube 视频](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF)。

- **Mistral API 的总结挑战**：`@nico2412_` 在通过 Mistral API 使用 URL 总结网页内容时遇到问题，由于 LLM 缺乏互联网访问权限，这项任务无法直接实现。`@duck` 建议参考 [GitHub notebook](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb) 中详细介绍的另一种文章总结方法。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 摘要

**AI 研究招募 VFX 艺术家**：一项 [调查](https://forms.gle/vexRuFUVHoojnfah7) 正在进行，旨在作为 AI 研究的一部分，寻求 VFX 艺术家和制作人的见解，征集宝贵的行业建议。

**打造更环保、自主可控的 Alexa 替代方案**：`@mattbcool` 通过改造 Alexa 硬件来解决电子垃圾问题，详细介绍了使用 Raspberry Pis 构建本地 [开源个人助手](https://mattcool.tech/posts/you-can-have-an-open-source-personal-assistant-in-2024) 的努力。

**CircleCI 助力 LLM 自动化测试课程**：**Deep Learning.ai** 和 **CircleCI** 合作推出了一门关于使用持续集成工具有效评估 LLM 应用程序的 [课程](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)。

**用 3DTopia 赋予文本生命**：由 `@meatfucker` 发现，[3DTopia 的 GitHub 仓库](https://github.com/3DTopia/3DTopia) 承诺通过可下载的模型权重和代码，迅速将文本转换为 3D 模型。

**Python 模块将 Steering Vectors 与 Hugging Face 的 Transformers 结合**：`@mihai4256` 创建了一个将 steering vectors 与 transformers 集成的 Python 模块，并在 [推文](https://twitter.com/m_chirculescu/status/1750149970026479720) 中透露了更多细节。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **GoogleAI 的 Lumiere 点燃 AI 讨论**：**GoogleAI 的 Lumiere** 亮相，这是一款强大的视频扩散模型，具备 text-to-video、image-to-video、风格化等功能，引发了工程师们的讨论。然而，由于缺乏开源，导致了兴奋与怀疑并存的情绪，一些人强调其方法背后的简单性，而另一些人则对 AI 生成视频的真实性表示担忧。([阅读关于 Lumière 的论文](https://arxiv.org/abs/2401.12945))。

- **数据主导地位引发辩论**：考虑到 **Google** 拥有的 YouTube 资源，他们在训练 text-to-video (T2V) 模型方面可能拥有无与伦比的数据优势。据推测，Lumiere 利用了 YouTube 广泛的视频库，其中包括自动生成的字幕和海量的用户评论，为训练视频多模态语言模型提供了可观的数据集。

- **视频模型对决**：讨论将 Google 的 **Lumiere** 与 Meta 的 EMU Video 模型进行了对比，强调了它们的能力，并对 AI 生成内容的自然度提出了质疑。批评集中在生成的视频偶尔会出现不一致性，导致一些社区成员开始寻找用于视频风格化的最佳开源替代方案。

- **AI 仓库访问困扰**：技术困难显而易见，用户在下载 LAION-en-aesthetics 字幕时遇到问题，Huggingface 禁用下载成为了社区关注和辩论的焦点。

- **AI 竞争升温**：一条 Reddit 链接流传开来，讨论了 **RWKV 7B** 的性能，它在多语言支持方面可能正与 **Mistral 7B** 竞争，同时还因高效的 CPU 利用率和线性运行时间而受到关注。这一对比引起了关注语言模型能力进展的热心人士的兴趣。([RWKV vs. Mistral 讨论](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/))。

请注意，虽然最初引用了某些用户名，但由于它们与主题的直接相关性对于 AI 工程师受众而言并不明确，因此在本摘要中已将其省略。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **Vanna AI 通过用于 SQL 的 RAG 惊艳亮相**：由 `@zain_hoda` 发起的项目 **Vanna AI** 因利用 **Retrieval Augmented Generation (RAG)** 改进 SQL 查询生成而备受关注，它能够索引 DDL/表结构和文本。该项目在社交媒体上引发了热议。[LlamaIndex 推文](https://twitter.com/llama_index/status/1750196064660127848)

- **寻求 AI 响应优化**：社区对完善 AI 机器人的响应机制表现出浓厚兴趣，特别是希望使 openaiagent 的回复过程像 openai assistant 一样更具迭代性，尽管目前尚未就具体方法达成共识。

- **追求 AI 工具的效率**：工程师们正在寻求 pandas 等工具的效率提升，权衡了上下文感知响应合成器等选项，并思考了对查询引擎进行多线程处理的线程安全性。

- **寻找 LLM 聊天机器人的托管方案**：对话中包含了关于在没有本地托管的情况下使用 LLM 等开源模型的建议，并提示可以利用 HuggingFace 和 Replicate 等服务进行 API 调用和微调。

- **增强 RAG 的性能**：关于增强 RAG 的讨论集中在 BGE 相似度重排序器（reranker）的潜力上，并建议在 RRF (Reciprocal Rank Fusion) 之后进行重排序以获得更好的结果。

- **聊天机器人的对话记忆**：用户表达了对追踪对话历史工具的需求（类似于 langchain 的 memory buffer），并提出在聊天引擎中集成 chat memory buffer 的可能性，以创建具有记忆能力的对话机器人。

- **向量存储偏好咨询**：有人请求社区就最佳向量存储公司发表意见，引发了关于偏好的讨论，但没有出现明显的首选方案。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

**X/Twitter 账号已解封**：**X/Twitter 账号**已恢复，之前受影响的用户已解封。仍遇到问题的用户可以在帖子中寻求帮助。

**使用 LangChain Streaming API 简化应用**：LangChain 发布了全新的 **streaming API**，以支持用户应用程序中的实时响应。资源包括 [API 文档](https://python.langchain.com/docs/expression_language/streaming)，针对 **AgentExecutor** 和 **LangGraph** 的特定模块（[AgentExecutor 文档](https://python.langchain.com/docs/modules/agents/how_to/streaming)，[LangGraph Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb)），以及关于 `stream_events` 的 [YouTube 教程](https://youtube.com/watch?v=ZcEMLz27sL4)。欢迎在 [GitHub](https://github.com/langchain-ai/langchain/discussions/16175) 上对该功能进行反馈和讨论。

**数据库困境与讨论**：查询范围从确定 LangChain 是否开源，到如何最好地在 Postgres 数据库模式中集成向量嵌入（vector embeddings），以及分享首选向量存储解决方案的呼吁。有用的参考资料包括 [PostgreSQL Schemas 文档](https://www.postgresql.org/docs/current/ddl-schemas.html)。

**LangServe 学习曲线**：**LangServe 频道**的用户正在努力学习如何使用 **agent_executor** 并理解 **LCELs** 的功能，一些人希望在设置和扩展工具使用方面得到更有经验成员的直接指导。

**共享工作中的创新与联系**：旨在将 RPA 与 AI 结合的平台 [AgentHub](https://www.agenthub.dev/) 宣布上线，并发布了一篇关于潜在生产力提升的博客文章（[AI and RPA: The Future of Work](https://www.agenthub.dev/blog/robotic_process_automation_with_ai)）。与此同时，一位用户在未提供具体背景的情况下呼吁开展合作。

**AI 导向课程教育**：[DataCamp](https://www.datacamp.com/code-along) 提供免费的 9 部分 AI 系列课程，包括“使用 LangChain 和 OpenAI API 构建多模态 AI 应用”；CircleCI 和 Deeplearning.ai 提供了一门关于 AI 应用自动化测试的新**免费课程**（[Automated Testing with LLMOPS](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)）。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **GPT-3.5 面临判断问题**：`@calytrix` 尝试使用 **GPT-3.5** 来评估新闻报道的重要性和情感倾向，发现它在判断情感方面表现较好，但在判断重要性方面存在困难且存在偏见。为了应对这些挑战，有人建议建立一个针对模型能力量身定制的特定评估系统可能会更有效。

- **AI 训练数据中的质量控制**：在微调语言模型时，`@thewindmom` 提出了对数据质量的担忧；`@hammadkhan` 建议使用人工抽检（eye-balling）和启发式过滤（heuristic filtering）等方法进行质量检查。同时，`@bjoernp` 推荐将合成数据生成作为一种策略，以减少对大规模评估的需求。

- **共享合成数据见解的引用**：`@_jp1_` 提到 **jon durbins 的 Airoboros 仓库** 是 DiscoResearch 合成数据生成技术的重要资源，并提供了 [jon durbins' Airoboros GitHub](https://github.com/jondurbin/airoboros) 的链接。该仓库提供了一个与 self-instruct 论文相关的可定制实现。

- **DPR 的尝试与磨难**：`@philipmay` 报告了测试 DPR 模型的结果不尽如人意，并通过前几名结果距离的总和研究了问题的特异性。此外，`@sebastian.bodza` 承认了 Embedding 中数据引用所涉及的挑战，并表示下一阶段将涉及问题生成。

- **DiscoLM German 7B v1 获得好评及更新**：用户 `@alex_22398` 称赞 **DiscoLM German 7B v1** 具有高质量的语言输出，并指出存在额外空行的问题。随后，`@_chromix_` 确认已解决该问题，并考虑更新 **GGUF 量化**。

---

# 第 2 部分：按频道详细摘要和链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199620732741746738) (1239 条消息 🔥🔥🔥):

- **GPT-4 vs Coder 模型**：用户参与了一场讨论，其中一位参与者 `@rombodawg` 声称他们的新编程模型在测试中表现优于 GPT-4，并提供了编程挑战的对比结果。`@righthandofdoom` 询问测试的模型是否是新的 "deepseek 33b"，`@rombodawg` 回复称测试的模型是他们 merge 的一部分，包含了 DeepSeek Coder 33B 的组件。

- **Yi-34B-200K 部署咨询**：`@super.deap` 寻求关于优化部署 Yi-34B-200K 模型的建议，目标是低成本和快速推理。提出了各种建议，包括使用 vLLM、降低 bits per word (bpw)，以及利用单个 A100 GPU 进行缩减上下文大小的推理。

- **探索大型 LLM 部署选项**：对话转向部署像 Goliath-120B 这样的大型语言模型，`@aikitoria` 正在寻求建议以将模型放入 80GB VRAM 的配置中。参与者讨论了不同的位精度权重 (bpw) 和缓存大小，以针对 VRAM 限制进行优化。

- **交流建模技术**：讨论继续，用户交流了关于推理时模型压缩策略的见解，提到了 LoRA 适配器等可能减少 VRAM 需求的技术。

- **Girlfriend GPT 及其可行性**：由 `@bubblegum.btc` 发起，一场讨论强调了对 Girlfriend GPT 如何能够向其用户提供大量消息的怀疑。一些人提出了降低成本的措施，例如缓存频繁查询或批量处理响应。

**提到的链接**：

- [Oracle 在整个技术栈中嵌入生成式 AI，以实现大规模企业 AI 采用](https://www.oracle.com/news/announcement/oracle-announces-availability-oci-generative-ai-service-2024-01-23/)：OCI 生成式 AI 服务现已正式发布，可在云端和本地选择来自 Cohere 和 Meta 的模型。
- [the tiny corp 筹集了 510 万美元](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html)：我们又开始了。我创办了另一家公司。资金已到账。
- [deepseek-ai/deepseek-coder-33b-instruct · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)：未找到描述
- [AGoodChoice.yml](https://drive.google.com/file/d/1Kus0s9fjOApFX1ESKrXeAQNAIeDhJ8-D/view?usp=drive_link)：未找到描述
- [deepseek-ai/deepseek-coder-33b-base · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-33b-base)：未找到描述
- [Cirno Cirno Fumo GIF - Cirno Cirno Fumo Fumo Touhou - 发现并分享 GIF](https://tenor.com/view/cirno-cirno-fumo-fumo-touhou-fumo-plush-fumo-gif-25572816)：点击查看 GIF

- [20x Faster as the Beginning: Introducing pgvecto.rs extension written in Rust](https://modelz.ai/blog/pgvecto-rs): 一个用 Rust 编写的新 Postgres 向量搜索扩展，采用 HNSW 算法，比 pgvector 快 20 倍。但速度只是开始 —— pgvecto.rs 的架构设计旨在轻松添加新算法。我们期待...
- [meta-llama/LlamaGuard-7b · Hugging Face](https://huggingface.co/meta-llama/LlamaGuard-7b): 未找到描述
- [Meet Your Perfect Match: Discover the AI Girlfriend Experience - GPTGirlfriend](https://www.gptgirlfriend.online/blog/post/the-ai-girlfriend-experience): 未找到描述
- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/): 我们的 AI 系统在几何问题上超越了最先进的方法，推进了数学领域的 AI 推理。
- [TypeScript/src/compiler/checker.ts at main · microsoft/TypeScript](https://github.com/microsoft/TypeScript/blob/main/src/compiler/checker.ts): TypeScript 是 JavaScript 的超集，可编译为简洁的 JavaScript 输出。 - microsoft/TypeScript
- [Tweet from NVIDIA 550.40.07 Beta driver released with fixes for VRR and Wayland](https://www.gamingonlinux.com/2024/01/nvidia-550-40-07-beta-driver-released-with-fixes-for-vrr-and-wayland/): NVIDIA 今日发布了适用于 Linux 的 550.40.07 Beta 驱动程序，其中包含多项重要修复，因此你可能想要尝试并进行彻底的游戏测试。
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only): 未找到描述
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): 在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合你的使用场景和预算的最佳 GPU。
- [LLMLingua/llmlingua/prompt_compressor.py at main · microsoft/LLMLingua](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py): 为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。 - micr...
- [(Long)LLMLingua | Designing a Language for LLMs via Prompt Compression](https://llmlingua.com/): 未找到描述
- [rombodawg/Everyone-Coder-33b-Base · Hugging Face](https://huggingface.co/rombodawg/Everyone-Coder-33b-Base): 未找到描述
- [TheBloke/Everyone-Coder-33B-Base-GGUF · Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF): 未找到描述
- [Build a Retrieval-Augmented Generation Chatbot in 5 Minutes](https://www.youtube.com/watch?v=N_OOfkEWcOk>): NVIDIA 高级解决方案架构师 Rohan Rao 演示了如何在不到 5 分钟的时间内，仅用 100 行 Python 代码开发大语言模型 (LLM)...
- [rombo (Luis Jalabert)](https://huggingface.co/rombo): 未找到描述
- [&quot;Legalize Nuclear Bombs&quot; Sound Effect](https://www.youtube.com/watch?v=575jfgf8hXU): 未找到描述
- [llm_steer/demo/llm_steer_demo.ipynb at main · Mihaiii/llm_steer](https://github.com/Mihaiii/llm_steer/blob/main/demo/llm_steer_demo.ipynb): 通过添加转向向量（steering vectors）利用激活工程（activation engineering）将 LLM 输出引导至特定主题/内容，并增强响应能力 - Mihaiii/llm_steer
- [GitHub - Mihaiii/llm_steer: Steer LLM outputs towards a certain topic/subject and enhance response capabilities using activation engineering by adding steering vectors](https://github.com/Mihaiii/llm_steer): 通过添加转向向量利用激活工程将 LLM 输出引导至特定主题/内容，并增强响应能力 - GitHub - Mihaiii/llm_steer...
- [GitHub - CHIP-SPV/chipStar: chipStar is a tool for compiling and running HIP/CUDA on SPIR-V via OpenCL or Level Zero APIs.](https://github.com/CHIP-SPV/chipStar): chipStar 是一个通过 OpenCL 或 Level Zero API 在 SPIR-V 上编译和运行 HIP/CUDA 的工具。 - GitHub - CHIP-SPV/chipStar...
- [AWQ: Up to 2.66x higher throughput by casper-hansen · Pull Request #2566 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/2566): 该策略是对较长序列进行反量化并运行 FP16 matmul。如果我们直接使用 cublas 而不是 torch.matmul，速度可能会更快。编辑：在 vLLM 中吞吐量似乎可以提高 2 倍以上...
- [GitHub - asg017/sqlite-vss: A SQLite extension for efficient vector search, based on Faiss!](https://github.com/asg017/sqlite-vss): 一个基于 Faiss 的高效向量搜索 SQLite 扩展！ - GitHub - asg017/sqlite-vss: 一个基于 Faiss 的高效向量搜索 SQLite 扩展！

- [Perfecting Merge-kit MoE's](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit?usp=sharing): 未找到描述
- [MoE Merge Misconceptions - Pastebin.com](https://pastebin.com/Mzfev83p): Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个允许你在网上存储文本一段时间的网站。
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector): 摘要：我们展示了一种与语言模型交互的新型可扩展方式：在前向传播（forward passes）中添加特定的激活向量。[2] 本质上，我们将前向传播的组合相加...
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vecto): 摘要：我们展示了一种与语言模型交互的新型可扩展方式：在前向传播（forward passes）中添加特定的激活向量。[2] 本质上，我们将前向传播的组合相加...

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1199624878333100124) (116 messages🔥🔥): 

- **模型大小问题浮现**：`@keyboardking` 表达了对人们如何使用 10GB 以下模型的困惑，随后 `@kquant` 澄清 7B 模型并非在 10GB 以下，并提供了关于以全精度（full precision）运行模型的见解。
- **关于模型量化（Quantization）的思考**：针对量化模型出现了不同观点，`@keyboardking` 提到 Q8 GGUF 似乎让模型变得毫无价值，而 `@kquant` 解释了 Q8 模型的实际大小及其有效性。
- **角色创建技巧讨论**：`.justinobserver` 分享了一个旨在帮助用户创建角色卡（character cards）进行角色扮演（roleplay）的 Prompt，引发了社区成员如 `@givan_002` 关于如何将该 Prompt 应用于多轮对话的提问。
- **模型性能反馈**：诸如 `@ks_c` 等用户测试了 bagelmistertour 等模型，并报告了褒贬不一的结果，表明需要更多上下文，而 `@kquant` 建议使用 `chat-instruct` 设置以获得更好的稳定性。
- **用户量化后续跟进**：社区成员如 `@kquant` 建议使用 LoneStriker 的 EXL2 量化版本，以便在 frankendpo 等模型上获得更好的体验；`@ks_c` 谈到了在 frankendpo 4x7b 上使用更高量化设置的积极体验，并提到其运行稳定且没有 Tokenizer 问题。

**提到的链接**：

- [FrankenDPO-4x7B-EXL2 - a Kquant03 Collection](https://huggingface.co/collections/Kquant03/frankendpo-4x7b-exl2-65a74855e211a95509e459b7): 未找到描述
- [Kquant03/Raiden-16x3.43B · Hugging Face](https://huggingface.co/Kquant03/Raiden-16x3.43B): 未找到描述
- [Another LLM Roleplay Rankings](https://rentry.co/ALLMRR): (欢迎在 Discord 上向 AliCat (.alicat) 和 Trappu (.trappu) 发送反馈) 我们热爱角色扮演和 LLM，并希望创建一个排名。部分原因是基准测试（benchmarks）并非真正针对角色扮演而设计...
- [GitHub - OFA-Sys/Ditto: A self-ailgnment method for role-play](https://github.com/OFA-Sys/Ditto): 一种用于角色扮演的自我对齐（self-alignment）方法。通过在 GitHub 上创建账号来为 Ditto 的开发做出贡献。

  

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1199632544405127188) (67 messages🔥🔥): 

- **澄清微调指令**：`@nigelt11` 询问了一个问题，即在微调 **Mistral 7B Instruct v0.2** 时，尽管他们的函数中包含了 `[INST]` 标签，但输出中却没有出现。`@gt9393` 和 `@flail_.` 澄清说，用户不需要手动输入这些标签；相反，这些标签应该作为微调期间提示词格式化（prompt formatting）的一部分。
- **对 Tokenizer 设置和数据集标签的困惑**：`@gt9393` 分享说，关于在 Tokenizer 设置中添加 `add_eos_token/add_bos_token` 同时又在数据集中包含 `<s>/</s>` 的做法，目前存在不确定性和不同的实践方式。`@nigelt11` 承认在他们的设置中可能遗漏了 `add_eos_token` 的细节。
- **理解指令标签与输出**：`@gt9393` 讨论了在微调数据集中将系统提示词（system prompts）放在 `[INST]` 标签内部还是外部的适当性。对话的一个重点是区分用于推理（inference）的指令模板 `[INST]` 标签与其他系统组件（如 `eos` 或 `bos` token）的重要性。
- **为模型聊天行为添加身份**：`@lordofthegoons` 询问了如何将“身份”训练到模型的聊天行为中（参考 Samantha 等模型），但后续讨论没有提供直接回答。
- **Medium 和 DataCamp 文章作为指南**：`@gt9393` 分享了一篇 Medium 文章（[Mistral 7B Fine-Tuning: A Step by Step Guide](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8)）和 DataCamp 教程（[Introduction to ChatGPT](https://www.datacamp.com/tutorial/mistral-7b-tutorial)）的链接，可能用于进一步阅读微调实践。这些资源有助于理解在微调语言模型中指令标签和 Tokenizer 的使用。

**提到的链接**：

- [Mistral-7B Fine-Tuning: A Step-by-Step Guide](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8)：介绍 Mistral 7B：语言模型的强者
- [Mistral 7B Tutorial: A Step-by-Step Guide to Using and Fine-Tuning Mistral 7B](https://www.datacamp.com/tutorial/mistral-7b-tutorial)：该教程涵盖了访问、量化、微调、合并和保存这个强大的 73 亿参数开源语言模型。

---

### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199962371284803644) (3 messages): 

- **模型合并在总权重为 1.0 时最优**：`@alphaatlas1` 提到，虽然理论上可行，但权重总和不等于 1.0 的合并在现实中的表现不如总和等于 1.0 的合并。
- **关于 DARE TIES 方法的澄清**：`@alphaatlas1` 澄清说，原始的 DARE TIES 论文建议使用极低的密度且总权重高于 1.0 进行合并，这样可以无冲突地结合两个模型的“离群（outlier）”权重；然而，他们指出这种方法在实践中似乎并不能产生最佳结果。
- **超过阈值的模型合并会失效**：`@sanjiwatsuki` 观察到，当总权重超过 1.4 时，模型合并的过程往往会崩溃。

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1199736389042970644) (4 messages): 

- **关于大模型使用 Hugging Face Pro 的咨询**：用户 `@keihakari` 询问 **Hugging Face Pro** 是否适合运行 70B 模型的部署推理（Deployment Inference）。
- **SigLIP 作为 CLIP 的替代方案**：`@jeremy.london` 提到现在可以用 **SigLIP** 替换 **CLIP** 了。
- **修改 GGUF 文件的内部模型信息**：`@lordofthegoons` 询问了一种手动更改 **GGUF 文件内部模型信息卡数据**的方法。
- **Frankenmerges 的参数计数问题**：`@lordofthegoons` 还指出 **frankenmerges** 往往会导致错误的参数计数。

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1199786275641753682) (10 messages🔥): 

- **寻找精通 API 的聊天机器人**：`@allanyield` 询问了关于开发利用 API 文档将 AI 转化为能代表用户与 API 交互的聊天机器人的公司。后续聊天中未提到具体的产品或公司。
- **随机 GIF 插曲**：`@Error.PDF` 分享了一个带有语言设置通知的 Tenor GIF，这似乎是离题的，与周围的技术讨论无关。
- **Mistral 在长序列上的竞争优势**：`@carsonpoole` 提到使用更长的序列和更大的 batch sizes 似乎是有益的，并表示这样的模型可以与 **Mistral** 实现兼容。
- **关于训练运行的好奇**：`@everyoneisgross` 对使用 **phi 2** 进行训练的时长感到好奇，暗示他们希望利用自己的 3060 GPU 进行过夜训练。
- **开源意向引发期待**：针对 `@gabriel_syme` 的提问，`@carsonpoole` 表示如果实现成功，有兴趣将其开源，这引发了对其潜在发布的关注。
- **使用 RMS 带来的 Loss 改进**：`@carsonpoole` 注意到使用 RMS 代替 layernorm 正在产生理想的 loss 结果，表明其当前工作中采用了优化方法。
- **Mistral 兼容性挑战**：`@carsonpoole` 承认直接与 Mistral 实现进行即插即用在技术上是不可行的，暗示了模型集成的复杂性。
- **评估模型性能**：`@carsonpoole` 表示打算很快对模型进行评估，预示着即将进入性能评估阶段。

**提到的链接**：

[Jordan Batter Looksmaxxing GIF - Jordan batter Looksmaxxing No - Discover &amp; Share GIFs](https://tenor.com/view/jordan-batter-looksmaxxing-no-sigma-me-gif-15250311043011508045)：点击查看 GIF

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1199668001729953852) (27 messages🔥): 

- **异构 AI 架构时代 (Era of Heterogeneous AI Architectures)**：`@burnytech` 分享了一篇 [LessWrong 文章](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and)（转发自 [AI Alignment Forum](https://alignmentforum.org/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and)），讨论了结合 **Transformers** 和 **Selective SSM (Mamba)** 的**异构 AI 架构**的兴起。`@mikahdang` 评论道，这一原则对未来的 AI 发展具有深远影响。

- **用于动态视频任务的 DoraemonGPT**：`@DoraemonGPT` 重点介绍了 *DoraemonGPT* 在动态视频理解方面的能力。它通过将视频转换为用于时空查询的符号记忆，利用工具进行外部知识评估，并使用一种新型的 LLM 驱动规划器。这项工作的详细内容见 [arXiv 论文](https://arxiv.org/abs/2401.08392)，旨在利用 LLM 进行视频场景解释来处理复杂任务。

- **RNNs 展现出巨大潜力**：`@_3sphere` 评论了像 RWKV 和 Mamba 这样的循环 LLM 在追踪序列中角色状态方面的潜力，并特别提到了 RWKV 7B 的目标：仅用 1T tokens 就能超越 Mistral，详见 `@euclaise` 分享的 [预告链接](https://fxtwitter.com/picocreator/status/1750245003690201363/photo/1)。

- **中等规模 LLM 翻译能力的进展**：`@mister_poodle` 分享了一篇介绍 **Contrastive Preference Optimization (CPO)** 的论文，这是一种提升中等规模 LLM 翻译能力的方法，在质量上比监督微调（supervised fine-tuning）有潜在的提升。详细研究可在 [arXiv](https://arxiv.org/abs/2401.08417) 上查阅。

- **Adept 推出 Fuyu-Heavy 多模态模型**：`.benxh` 展示了 **Adept Fuyu-Heavy**，这是一款专为数字 Agent 设计的新型多模态模型，拥有强大的多模态推理能力，详见 [Adept.ai 博客文章](https://www.adept.ai/blog/adept-fuyu-heavy)。尽管其规模适中，但 Fuyu-Heavy 在某些基准测试中优于更大的模型，公告中提供了更多细节和示例。

**提到的链接**：

- [MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660)：无 Token 语言模型直接从原始字节学习，消除了子词分词（subword tokenization）的偏见。然而，在字节上运行会导致序列显著变长，而标准的自回归...
- [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417)：中等规模的大语言模型（LLM）——即拥有 7B 或 13B 参数的模型——表现出极具前景的机器翻译（MT）性能。然而，即使是表现最好的基于 13B LLM 的翻译模型...
- [Adept Fuyu-Heavy: A new multimodal model](https://www.adept.ai/blog/adept-fuyu-heavy)：Adept Fuyu-Heavy 是专为数字 Agent 设计的新型多模态模型。
- [DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392)：由于大语言模型（LLM）的能力，AI Agent 领域正以空前的速度发展。然而，LLM 驱动的视觉 Agent 主要专注于解决图像模态的任务...
- [Tweet from PicoCreator (🌉 in/arena) (@picocreator)](https://fxtwitter.com/picocreator/status/1750245003690201363/photo/1)：RWKV 7B 预告（非最终版）.... 🤞 希望我们能仅用 1T tokens 就超越 Mistral https://huggingface.co/BlinkDL/temp/blob/30ac41f863dcf0600a71e4182600022df989dfa1/rwkv-x052-7b-world-v2-86%25trained-2...
- [A-JEPA neural model: Unlocking semantic knowledge from .wav / .mp3 audio file or audio spectrograms](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF)：🌟 释放 AI 从音频中学习的力量！🔊 观看与 Oliver, Nevil, Ojasvita, Shashank, Srikanth 和 N... 深入讨论 A-JEPA 方法。
- [AGI will be made of heterogeneous components, Transformer and Selective SSM blocks will be among them — LessWrong](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and)：这篇文章是由最近的两篇作品引发的：…
- [AGI will be made of heterogeneous components, Transformer and Selective SSM blocks will be among them — LessWrong](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made)：这篇文章是由最近的两篇作品引发的：…

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1199633382678732800) (231 条消息🔥🔥): 

- **意识难题仍在继续**：`@nonameusr` 向小组询问他们认为意识是什么，引发了与 `@_3sphere` 关于意识的“简单”和“困难”问题的讨论，而 `@giftedgummybee` 则开玩笑地提到了一个关于 AI 自我意识的虚构法庭案例。
- **迈向可控 AI**：在讨论 `@mihai4256` 的工作时，`@teknium` 对语言模型激活引导（activation steering）背后的方法表示了兴趣。对话进展到 `@mihai4256` 解释如何使用文本 Prompt 来影响模型行为，这激发了社区成员的进一步探究。
- **Prompt 格式影响 LLM**：`@euclaise` 和 `@stellaathena` 等成员辩论了在语言模型中构建 Prompt 的各种方法的效率，并分享了讨论 Prompt 敏感度以及使用动态模板时潜在布局问题的学术论文链接。
- **GPU 与 AI 硬件讨论**：`@theluckynick` 和 `@sirri69` 等人讨论了各种 GPU（如 1080 ti 和 3060）在运行 AI 模型方面的能力和局限性，并对未来的软件优化表示期待。
- **杂项技术闲谈**：`@teknium` 和 `@carsonpoole` 就运行 AI 模型和通用计算的不同操作系统的优缺点交换了意见，而 `@n8programs` 分享了一些涉及 webGL2 和向量表示量化方法的个人技术成就。

**提到的链接**：

- [Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting](https://arxiv.org/abs/2310.11324)：随着大型语言模型 (LLM) 被用作语言技术的基础组件，准确描述其性能至关重要。由于 Prompt 设计中的选择可能会产生强烈影响...
- [Skull Explode GIF - Skull Explode - Discover &amp; Share GIFs](https://tenor.com/view/skull-explode-gif-25528415)：点击查看 GIF
- [Skeleto Skeleton GIF - Skeleto Skeleton Fire - Discover &amp; Share GIFs](https://tenor.com/view/skeleto-skeleton-fire-hell-burn-gif-26129219)：点击查看 GIF
- [Sunglitters Glittery GIF - Sunglitters Glittery - Discover &amp; Share GIFs](https://tenor.com/view/sunglitters-glittery-gif-24179798)：点击查看 GIF
- [Surface Form Competition: Why the Highest Probability Answer Isn't Always Right](https://arxiv.org/abs/2104.08315)：大型语言模型在零样本设置中表现出了可喜的结果 (Brown et al., 2020; Radford et al., 2019)。例如，它们可以仅通过以问题为条件来执行多项选择任务...
- [LM-exp/steering at main · nrimsky/LM-exp](https://github.com/nrimsky/LM-exp/tree/main/steering)：在 SERI MATS 期间进行的 LLM 实验 - 专注于激活引导 / 解释激活空间 - nrimsky/LM-exp
- [Adding topic steering on layers  · Issue #5119 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5119)：先决条件 在提交 Issue 之前，请先回答以下问题。[ x ] 我正在运行最新的代码。开发非常迅速，因此目前还没有标记版本...
- [GitHub - VikParuchuri/texify: Math OCR model that outputs LaTeX and markdown](https://github.com/VikParuchuri/texify)：输出 LaTeX 和 Markdown 的数学 OCR 模型。通过在 GitHub 上创建账号为 VikParuchuri/texify 的开发做出贡献。
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector)：摘要：我们展示了一种与语言模型交互的新型可扩展方式：在正向传播中添加特定的激活向量。[2] 本质上，我们将正向传播的组合相加...

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1199900982402891796) (20 messages🔥): 

- **关于 Fine-tuning 与 Instruct-tuning 的澄清**：`@moconna` 询问了 Fine-tuning 和 Instruct-tuning 之间的区别，以及如何将它们应用于自己的数据集。`@besiktas` 澄清说 Instruct tuning 通常在 Fine-tuning 之后进行，而 Fine-tuning 涉及在数据集上进行 Causal Language Modeling，之后可能再使用高质量的专业数据进行 Instruct tuning。

- **关于 Mixtral 的 Fine-tuning 数据格式的困惑**：`@moconna` 寻求关于 Mistral 的 Fine-tuning 正确格式的澄清，因为教程似乎建议使用基于 Instruction 的格式，他们认为这属于 Instruct-tuning。`@besiktas` 确认这些术语比较模糊，提供的格式通常同时用于 Fine-tuning 和 Instruct tuning。

- **Fine-tuning 任务的直接方法**：当 `@moconna` 询问如何针对新领域和新语言的特定任务对 Mistral 进行 Fine-tune 时，`@besiktas` 建议为新领域使用清洗后的数据，并提醒说在完全陌生的语言上能否成功可能还不确定，需要进一步研究。

- **生产环境中的模型持续更新**：`@kenakafrosty` 询问社区对于使用实时使用数据更新 Fine-tuned 模型是否达成了共识。讨论似乎没有定论，提到了各种潜在方法，但没有明确的策略。

- **关于 Mistral 最佳数据集的查询与玩笑**：`@locutusque` 寻求对 Mistral 进行 Fine-tune 的最佳数据集建议，对此 `@besiktas` 幽默地建议使用 MNIST，引发了关于 Mistral 等语言模型视觉能力的俏皮交流。
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1199669217855811615) (5 messages): 

- **Project Obsidian 宣布升级**：`@qnguyen3` 宣布了将 **Obsidian 升级到 v2** 的计划。
- **选择 StableLM 2 1.6B 进行升级**：在升级过程中，`@qnguyen3` 将使用 **StableLM 2 1.6B**，这次选择了一个更小的模型。
- **社区对升级的回应**：`@giftedgummybee` 对使用 **StableLM 2 1.6B** 进行 Obsidian 升级的消息简单回应了 "Nice"。
- **分享 InstructDoc 数据集**：`@gabriel_syme` 分享了 GitHub 上 **InstructDoc 数据集** 的链接，这是一个用于视觉文档理解中 Zero-shot Generalization 的资源 ([GitHub - nttmdlab-nlp/InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc))。

**提到的链接**：

[GitHub - nttmdlab-nlp/InstructDoc: InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions (AAAI2024)](https://github.com/nttmdlab-nlp/InstructDoc): InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions (AAAI2024) - GitHub - nttmdlab-nlp/InstructDoc: InstructDoc: A Dataset for Zero-Shot Generaliz...

  ,

### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1199830782458474527) (1 messages): 

- **ChatGPT 在行动**：`@abdubs` 呼吁 **@everyone** 在 <#1155775326253756456> 频道分享他们对 **ChatGPT** 独特且富有创意的用法。他们热衷于了解这项技术如何以各种方式造福用户的生活，从辅助学生到讲故事以及增强沟通。
  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1199684939046723704) (18 messages🔥): 

- **AI 图像提示词说明**：`@lugui` 解释说 GPT-4 模型是通过描述图像来构建 Prompt，而不是直接生成新图像，这会导致截然不同的结果。
- **建议的图像处理替代方案**：针对 DALL-E 图像重建能力的担忧，`@xenowhiz` 建议使用 **Code Interpreter** 及其图像处理模块来实现所需效果。
- **AI 播客推荐**：`@fran9000` 推荐了 "[The AI Breakdown](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614)" 播客，该播客提供关于人工智能的每日新闻和分析，涵盖从创意到伦理考量的广泛话题。
- **针对特定任务使用旧版 GPT**：`@lzgodhook13` 对新版 GPT 表示沮丧，并询问如何使用 **2022 年 5 月至 6 月的 version 3**，理由是新版在处理数字排序等简单任务时存在困难。
- **了解 GPT-4 消息限制**：`@pope0004` 询问为何 GPT-4 对 Premium 用户也有使用限制，`@jaicraft` 建议使用 API 或 Copilot Pro 以获得更多访问权限，`@satanhashtag` 确认消息限制适用于所有人。

**提到的链接**：

- [Smol Talk](https://buttondown.email/ainews)：我们汇总 AI Discord 频道内容，并每天为您发送摘要！
- [‎The AI Breakdown: Daily Artificial Intelligence News and Discussions on Apple Podcasts](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614)：科技 · 2024

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1199639704774115380) (66 messages🔥🔥): 

- **让 GPT 输出长篇小说**：`@tetsujin2295` 表示沮丧，虽然 GPT 愿意在聊天中生成 1500 字的长摘要，但在被要求在可下载文档中进行摘要时却无法完成。`@loschess` 建议由于 GPT 响应中的 Token 限制，应通过将任务拆分为多个部分来管理输出。

- **Custom GPT 离奇失踪**：`@sstrader29` 遇到其 Custom GPT 模型从搜索中消失的问题。`@loschess` 建议检查是否有关于此事件的邮件，暗示 OpenAI 可能会就模型问题发送通知。

- **赋予 GPT 邮件外交能力**：`@greg_nyc` 正在探索让 GPT 通过 Gmail 起草邮件回复的方法，`@darthgustav.` 建议需要使用 Google API 创建 Custom Action，并仔细遵循 Google 文档。

- **GPT Builder 中的文件上传挫折**：`@alexandre.f.s` 提到了在训练 Custom GPT 时上传文件的问题，在使用 GPT Builder 时遇到了麻烦。`@darthgustav.` 建议采取清除浏览器缓存、逐个附加文件等步骤以防止损坏。

- **更新后的 Context 混乱**：
  - `@kickiniteasy` 和 `@ajkuba` 报告更新后 Custom GPT 的 Context Window 严重缩小，极大地影响了模型的性能和连续性。
  - `@nefas` 提到 Builder 中的 GPT 链存在问题，它们似乎会遗忘之前的交互，而这一问题在预览窗口中并未出现。
  - `@cairpli` 已针对上述问题创建了 Bug 报告，并链接到了 Discord 频道。（链接显示为占位符 '<<<null>>>'，因此未包含在内。）

- **GPT 的广告雄心**：`@9rld` 有意创建一个模仿平面广告风格的 Custom GPT，希望使用 Web Crawler 从广告网站收集数据。他们正在询问 GPT 是否能从这些网站上以图像为主的广告内容中提取文本。
  

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1199685871780241418) (71 messages🔥🔥): 

- **避免列表成为热门话题**：`@dbugger` 对 ChatGPT 无论输入什么都返回列表感到沮丧。`@darthgustav.` 等人建议提供明确的格式指令，例如使用基于情感的 Prompt 或角色命名的 Prompt 来抑制列表式输出（"EmotionPrompt"）。

- **避免列表的 Prompt**：`@eskcanta` 建议将“负向提示 (negative prompting)”与动机结合使用，以避免模型默认输出列表，并分享了一个旨在引导出叙述性回答的详细角色驱动 Prompt 示例。

- **应对 Instagram 文案挑战**：`@semir9526` 寻求关于为特定格式的 Instagram 文案编写 Prompt 的建议。提供了多个建议，包括使用基于变量的结构化输出，以及避免提及 ChatGPT 无法准确计算的字符限制。

- **改进文档中的关键词搜索**：`@novumclassicum` 询问如何让 GPT 在扫描文档关键词时获得更好的结果。`@darthgustav.` 建议理解随机推理 (stochastic inference) 并考虑任务分块 (task chunking)，而 `@eskcanta` 提到了可能用于字符串处理的 Python 工具集成，两人都讨论了通过语义匹配来提高搜索准确性。

- **Prompt Engineering 黑客松组队**：`@brayheart` 邀请成员组队参加 Prompt Engineering 黑客松，开启了合作之门，而 `@eskcanta` 表现出潜在兴趣，并询问了具体目标。
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1199685871780241418) (71 messages🔥🔥): 

- **打破列表习惯**：`@dbugger` 表达了对 ChatGPT 无论 Prompt 如何都始终以列表格式提供答案的沮丧。`@darthgustav.` 和 `@eskcanta` 提出了各种解决方案，包括提供明确的输出模板和风格指令、使用 EmotionPrompt，以及编写每次只关注单个项目的 Prompt。
  
- **打造营销魔法**：`@semir9526` 寻求改进 AI 生成的营销 Prompt 的建议，以创作真实的 Instagram 文案。`@darthgustav.` 在 Prompt 结构、变量使用方面给出了建议，并建议不要指定 Instagram 的字符限制，而是建议保持文案“简短且令人兴奋”。

- **更聪明地搜索，而非更努力**：`@novumclassicum` 质疑 GPT 在文档中进行关键词搜索的一致性。`@darthgustav.` 解释了模型的随机性 (stochastic nature)，并建议对关键词使用语义匹配，同时建议将任务拆分为块以获得更好的结果。

- **Prompt Engineering 挑战**：`@brayheart` 询问了大家对 Prompt Engineering 黑客松的兴趣。`@eskcanta` 表现出兴趣，但询问了黑客松具体目标的更多细节。

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1199624218640400415) (160 条消息🔥🔥): 

- **Perplexity AI Companion 扩展建议**：用户 `@mares1317` 提倡使用 Perplexity AI Companion 扩展，甚至建议它在使用 GPT-4 进行后续提问时可以保留来源，尽管 `@icelavaman` 指出 iOS 应用无法在后续查询中保留来源。
- **订阅 Pro 带来的益处**：`@thejuicyy` 订阅了 Pro 并对其相关性检查功能印象深刻。`@mares1317` 建议尝试配合 GPT-4 使用 Perplexity AI companion，认为其体验与免费版有显著不同。
- **折扣码应用故障排除**：多位用户讨论了应用 Perplexity Pro 折扣码的问题。`@toothpick4339` 分享了 Reddit 用户 u/ArakDemonBlade 的成功步骤，`@bennsiee` 苦于找不到输入促销码的地方，而 `@ok.alex` 和 `@speedturkey` 提供了协助，并建议联系支持邮箱 pro@perplexity.ai。
- **关于增加在线 LLM 限制的咨询**：用户 `@mantas_82008` 询问是否可以将在线 LLM 模型的限制提高到每分钟 10 次以上。`@icelavaman` 回复了一个相关 Discord 频道的链接以获取更多信息。
- **大型 Google Drive 文档的处理挑战**：`@jaybob32` 在访问大型 Google Drive 文档时遇到问题。用户 `@gentlefoxssbm`、`@deicoon` 和 `@me.lk` 建议下载或转换文档，`@mares1317` 提供了转换器的搜索链接，并分享了一个将文本直接复制粘贴到 Perplexity 提示词框的解决方案。

**提及的链接**：

- [Pricing](https://docs.perplexity.ai/docs/pricing)：无描述
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/)：Perplexity CEO Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1199665672897179798) (4 条消息): 

- **Perplexity AI 研究查询分享**：用户 `@nocode7` 分享了一个 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/Arc-wont-let-usDclkxuSCehLZy3v6RMFA?s=c)，但未指定研究的具体背景或内容。
- **商业聊天回顾达人**：`@miamiseipazzesco777` 因高效总结某些商业聊天内容而受到认可。
- **来自 Perplexity 设计负责人的见解**：`@mares1317` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=4BH454Kw-90&t=60s)，视频中 Perplexity AI 的设计负责人 **Henry Modisett** 讨论了为 AI 进行设计的挑战以及如何在该领域获得工作。
- **对 Perplexity 团队和 Copilot 的赞赏**：`@zenrobot.eth` 赞扬了上述对 Henry Modisett 的采访，并强调了 Perplexity 的 Copilot 功能的实用性，提供了一个全球新闻趋势摘要的链接 ([Perplexity AI 搜索链接](https://www.perplexity.ai/search/Trends-in-Global-nU4umKTNQhywNTQFVhmGGg))，并解释了 [博客文章](https://blog.perplexity.ai/faq/what-is-copilot) 中详细介绍的 Copilot 对话式搜索能力。

**提及的链接**：

- [Designer at $525M AI Startup 'Perplexity' reveals challenges of designing for AI](https://www.youtube.com/watch?v=4BH454Kw-90&t=60s)：🧠 你将从这次采访中学到什么？1. 为 AI 进行设计的挑战 2. 如何成为一名 AI 设计师 📢 为什么 HENRY MODISETT 是个大咖？Henry Mod...
- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1199858991895937094) (5 条消息): 

- **自动充值后解锁额度**：用户 `@stevemac_90623` 提到，只有在输入 **$2** 作为金额完成自动充值（auto top off）流程后，系统才发放了 **$5.00 额度**。

- **对 API 与浏览器响应差异的好奇**：`@obicho` 对使用 API 与浏览器时响应为何存在差异表示好奇。

- **API 模型等效性咨询**：`@benhirap` 询问哪个 API 模型对应聊天网站上带有 Copilot 的 "Experiment" 功能。

- **揭开模型之谜**：`@noremac258` 确认 **PPLX 70B** 是与聊天网站 "Experiment"（带 Copilot）等效的 API 模型。

- **浏览器与 API 行为差异解释**：`@icelavaman` 解释了浏览器与 API 响应不同的原因，指出是因为使用了不同的系统提示词（system prompt）以及搜索来源的方式不同。

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199683093158699008) (71 messages🔥🔥): 

- **AI 运算中 GPU 优于 CPU**：`@laszlo01` 询问如何在 AI 运算中从使用 CPU 切换到 GPU，`@heyitsyorkie` 建议在聊天页面右侧的设置面板中查找该设置。
- **开源聊天机器人创建工具需求**：`@jan_naj` 寻求一个用于构建可定制的类 GPT 聊天机器人的仓库，要求支持本地托管并包含记忆线程（memory threads）。`@fabguy` 回复了一个略带戏谑的 GitHub 搜索页面链接，导致 `@jan_naj` 的不满，随后 `@dagbs` 建议提出更具体的问题。
- **旧处理器缺乏 Linux 版本支持**：`@d0mper` 提到正在使用一台不支持 AVX2 的旧电脑，并询问加载语言模型时除了 LM Studio 之外的替代方案。`@heyitsyorkie` 澄清说 Linux 版本仅支持具备 AVX2 的 CPU。
- **加密货币机器人警告及封禁请求**：`@heyitsyorkie` 提醒用户举报任何散布 Discord 邀请链接的加密货币机器人，以便及时封禁。
- **编译 llama.cpp 作为 LM Studio 的替代方案**：对于像 `@d0mper` 这样在 Ubuntu 等平台上寻找 LM Studio 替代方案的用户，`@heyitsyorkie` 建议编译 llama.cpp，它支持加载语言模型。
- **关于将 Stable Diffusion 整合进 LM Studio 的讨论**：在对 LM Studio 是否支持 StableLM 模型产生一些困惑后，`@heyitsyorkie` 提到了 Stable Diffusion 独立的 C/C++ 移植版本，以及将其整合进 LM Studio 的可能性。`@altryne` 指出最近的一次更新使某个特定引擎变得兼容。

**提到的链接**：

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)：未找到描述
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-release)：发现、下载并实验本地 LLM
- [Build software better, together](https://github.com/search)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。
- [GitHub - ggerganov/llama.cpp: Port of Facebook&#39;s LLaMA model in C/C++](https://github.com/ggerganov/llama.cpp)：Facebook LLaMA 模型的 C/C++ 移植版。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++](https://github.com/leejet/stable-diffusion.cpp)：纯 C/C++ 实现的 Stable Diffusion。通过在 GitHub 上创建账号为 leejet/stable-diffusion.cpp 的开发做出贡献。

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199623974137647196) (20 条消息🔥): 

- **针对新手的模型推荐**：新用户 `@rparada` 询问了用于代码转换和神经网络架构修改的**模型推荐**。用户 `@fabguy` 引导他们查看频道 **#1185646847721742336** 中的条目。

- **使用 RAG 读取 PDF 的代码片段**：`@ui.mz` 分享了一个使用 **PyMuPDFLoader** 通过 RAG 从 PDF 文件中读取内容的代码片段，并因遇到与 **OpenAI Embeddings API** 相关的 **ValidationError** 而寻求建议。

- **对 OpenAI Embeddings API 替代方案的需求**：在讨论错误后，`@ui.mz` 透露其缺乏 **NVIDIA GPU**，`@fabguy` 澄清其意图是通过提供一个 [GitHub repository](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/) 作为参考，帮助他们找到自己的解决方案。

- **关于图像/文件读取模型的查询**：用户 `@pandora_box_open` 询问了能够有效读取图像/文件的模型。`@heyitsyorkie` 回复了一个仅限于描述图像的模型 Discord 链接，并表示目前还没有用于**文档读取**的 RAG 系统。

- **RAG 的解释及潜在集成**：在 `@pandora_box_open` 的提示下，用户 `@heyitsyorkie` 解释了**检索增强生成 (RAG)** 的概念，随后前者表达了对其集成的希望。解释中包含了一个指向 Databricks 术语表中关于 RAG 条目的[链接](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)。

- **使用 HuggingFace 模型探索 RAG**：`@pandora_box_open` 分享了一个 [HuggingFace 模型链接](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp)，该模型可能展示了类似于 RAG 的集成。

**提到的链接**：

- [Retrieval Augmented Generation](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)：检索增强生成（RAG）是一种架构方法，它提取数据作为大语言模型 (LLMs) 的上下文，以提高相关性。
- [MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp · Hugging Face](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp)：未找到描述

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1199623814053634138) (60 条消息🔥🔥): 

- **各种模型的 GPU 层配置**：`@aswarp` 和 `@cloakedman` 讨论了针对不同尺寸模型和显卡的最佳 GPU 层数，使用如 `-1`（将所有层卸载到 GPU）或手动调整以避免崩溃。`@heyitsyorkie` 建议如果 `-1` 导致错误，可以尝试调整层数，并建议查看 LM Studio 中的模型详情以获取指导。
  
- **性能与硬件规格挂钩**：`@smallshinyant` 寻求关于通过关注 tok/sec 等指标来对新硬件进行基准测试的建议，而 `@bobzdar` 指出额外的 VRAM 允许运行更大或压缩更少的模型，暗示性能不仅仅关乎速度。

- **发挥最大硬件潜力**：`@aswarp` 询问在运行模型时，如果系统 RAM 大部分未被使用，如何增加其使用率。`@.ben.com` 建议为了性能应优先考虑 GPU 使用，除非有意使用显存装不下的超大模型，但这会导致处理速度变慢。

- **软件设置影响模型性能**：`@aswarp` 和 `@dylpickle300` 等用户遇到了模型性能问题，并讨论了 LM Studio 中的各种设置，如 `n_gpu_layers`、来自模型 GGUF JSON 的 `num_layers`，建议通过调整这些设置来提升性能，尽管存在系统 RAM 未使用和模型停滞等问题。

- **硬件支持与模型兼容性**：`@luthorheim` 询问了在没有 AVX2 指令支持的情况下运行模型的可能性，对此 `@heyitsyorkie` 提供了一个可能支持旧处理器的 Beta 版本链接。`@cloakedman` 提醒，如果模型文件大于可用 VRAM，将导致性能缓慢或需要降低质量。

**提到的链接**：

[LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)：未找到描述

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1199832714572996760) (2 messages): 

- **探索模型海洋**：`@greg0403` 对如何在 HuggingFace 上众多的模型中进行选择表示困惑，并质疑由于文档极少，这些模型的独特卖点 (USPs) 究竟是什么。他们寻求关于如何区分并选择特定模型的指导。
- **模型比较指南**：针对 `@greg0403` 的提问，`@kadeshar` 建议使用排行榜，例如 [HuggingFace open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)，作为比较模型性能的一个良好起点。

**提到的链接**：

[Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)：未找到描述

  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1199791991727333446) (1 messages): 

- **探索 LM Studio 与 Open Interpreter 之间的互操作性**：用户 `@222gate` 正尝试将 **LM Studio 推理与 memgpt 集成**，然后再接入 **Open Interpreter**。他与 `@cpacker` 讨论了 memgpt server 与 OpenAI Assistant API 之间的相似性。他们正在研究 memgpt 的 server 是否能在 Chat 和 Completion 调用功能方面模拟 OpenAI。
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1199800276610207834) (2 messages): 

- **寻求 Prompting 技巧**：用户 `@222gate` 向社区征集**改进通用 Prompting 的想法**。询问中未提供具体的上下文或细节。
- **集成挑战**：`@222gate` 提到了在集成 **LM Studio memGPT 和 OpenAI** 时遇到的困难，并正在寻求该过程的帮助。集成过程的复杂性未做详细说明。
  ,

### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1199824460295970906) (1 messages): 

- **EleutherAI 与 NSF 合作开展 AI 研究**：`@tastybucketofrice` 宣布 EleutherAI 与**美国国家科学基金会 (NSF)** 建立合作伙伴关系，启动**国家 AI 研究资源 (NAIRR)**，旨在为 AI 研究提供关键资源的访问权限。官方公告可在此处阅读：[here](https://new.nsf.gov/news/democratizing-future-ai-rd-nsf-launch-national-ai)。

- **EleutherAI 对开放 AI 研究的承诺**：自 2020 年发布语言模型 (LMs) 以来，EleutherAI 一直倡导开放研究，现在已捐赠 GPU 资助以增强学术界的 AI 研究能力。其承诺的历史可以在博客文章 [“Why release a large language model?”](https://blog.eleuther.ai/why-release-a-large-language-model/) 中找到。

- **解决研究人员的算力资源问题**：尽管如今预训练模型的获取渠道更加广泛，但 `@tastybucketofrice` 指出，有限的算力资源仍然是研究人员面临的障碍。EleutherAI 致力于确保研究人员能够控制其模型的行为方式及其编码的价值观。

- **使用 GPT-NeoX 赋能 AI 研究**：为了应对 HPC 挑战，`@tastybucketofrice` 重点介绍了 [GPT-NeoX 库](https://github.com/EleutherAI/gpt-neox)，该库通过在各种平台上大规模运行（包括橡树岭国家实验室的 Summit 和 Frontier、LUMI、AWS 和 CoreWeave）来促进 AI 研究。
  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1199796929874374736) (11 条消息🔥): 

- **代码荒野中的许可证困惑**：`@xa9ax` 询问了 GitHub 代码库中可能限制学术研究模型训练的特定许可证。`@stellaathena` 指出，实际上**几乎没有许可证直接涉及模型训练**，而 `@avi.ai` 强调法律结果可能**因司法管辖区而异**。

- **寻求法律建议**：针对 `@xa9ax` 的进一步询问，`@stellaathena` 建议咨询律师以澄清许可证问题。同时，`@avi.ai` 建议针对所在地相关的版权和知识产权法进行初步研究。

- **告别 Discord？**：`@hostiq` 简单地表示：“是时候离开这个 Discord 了”，暗示他们将退出社区聊天。

- **GPU 资助合作提议**：`@yikesawjeez` 联系了 `@stellaathena`，希望能与现有的 GPU 资助建立联系，并提供了其个人资料中列出的**服务器链接**和用于申请算力访问的 **google form**。

- **CoPilot 案件进展缓慢**：`@clockrelativity2003` 询问了 GitHub CoPilot 诉讼的现状。`@.undeleted` 幽默地评论道，**法律案件的解决需要极长的时间**，暗示该案仍未解决。
  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1199630853467283556) (80 条消息🔥🔥): 

- **ML 训练中的数据质量与规模之争**：`@catboy_slim_` 讨论了数据质量，指出策划多样化、高质量的数据虽然具有挑战性，但并不需要巨大的数据量。他们提到 **Wavenet 数据**的记忆库相对较小，但策划得很好。

- **GPU/TPU 硬件与框架兼容性讨论**：`@alofty` 分享了一篇关于 ML 框架在不同硬件间可移植性的[文章](https://arxiv.org/abs/2309.07181)。这引发了 `.the_alt_man` 关于 JAX 在 TPU 和 GPU 上性能的辩论，他建议不要使用 PyTorch，因为其 XLA 支持较差。

- **‘MambaByte’ 凭借无 Token 语言建模引起关注**：`@pizza_joe` 分享的一篇[学术论文](https://arxiv.org/abs/2401.13660)介绍了 MambaByte，这是一种用于无 Token 语言建模的状态空间模型 (state space model)，这引起了 `_inox` 的怀疑，直到 `@thatspysaspy` 指出著名研究员 Sasha Rush 也参与其中。

- **Burn：一个新的 Rust 深度学习框架**：`@kenakafrosty` 分享了基于 Rust 的 DL 框架 Burn 的[链接](https://burn.dev/)，并讨论了它的潜力，但指出它必须具备强大的并行支持才能具有竞争力。`@canadagoose1` 强调了对多节点 (multinode) 支持的需求，而 `.the_alt_man` 则为了性能更倾向于像 JAX 这样更极简的框架。

- **探讨生产环境中微调模型的持续更新**：`@kenakafrosty` 询问了更新微调模型的最佳实践，引发了与 `@fern.bear` 关于 **Elastic Weight Consolidation (EWC)** 等方法的讨论，以及如何使用损失函数在严格限制更改与允许显著偏差之间取得平衡。

**提到的链接**：

- [MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660)：无 Token 语言模型直接从原始字节学习，消除了子词分词 (subword tokenization) 的偏差。然而，在字节上操作会导致序列显著变长，而标准自回归...
- [DsDm: Model-Aware Dataset Selection with Datamodels](https://arxiv.org/abs/2401.12926)：在为训练大规模模型选择数据时，标准做法是过滤符合人类数据质量观念的样本。这种过滤会产生定性上干净的数据点，但...
- [no title found](https://burn.dev/)：未找到描述
- [The low-rank hypothesis of complex systems](https://arxiv.org/abs/2208.04848)：复杂系统是高维非线性动力系统，其组成部分之间存在复杂的相互作用。为了对其大规模行为做出可解释的预测，通常...
- [The Grand Illusion: The Myth of Software Portability and Implications for ML Progress](https://arxiv.org/abs/2309.07181)：推动机器学习的边界通常需要探索不同的硬件和软件组合。然而，在不同工具栈之间进行实验的自由可能与...
- [GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/](https://github.com/patrick-kidger/equinox)：JAX 中优雅易用的神经网络 + 科学计算。https://docs.kidger.site/equinox/ - GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in...

### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1199741281753108558) (2 条消息): 

- **扩展至 1.3B 参数规模**：`@stellaathena` 提到，尽管原始论文存在局限性，但他们提供了在 **1.3B 参数规模**下训练模型的算力。
- **频道引用**：`@random_string_of_character` 引导用户查看另一个频道中关于 1B 参数模型的讨论结果，具体链接为 `<#1129489948710539334>`。
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1199712859505045544) (14 条消息🔥): 

- **解码神经代码**：用户 `@g_w1` 询问了关于单语义性（monosemanticity）概念中空间点绘制的问题。`@woog` 澄清说，他们绘制的是**解码器权重（decoder weights）**，这些权重被解释为在 **Sparse Autoencoder (SAE) 空间**中恢复的相对于神经元方向的特征方向。
  
- **字典元素作为神经元方向**：在进一步的澄清中，`@woog` 向 `@g_w1` 确认，SAE 中的每个字典元素都是**神经元空间中的一个方向**，这表明 SAE 空间与神经元空间之间存在一种通过绘图可视化的联系。`@g_w1` 对此解释表示认可。

- **可解释性研究更新**：用户 `@loganriggs` 分享了来自 Anthropic 可解释性团队的 [1 月更新](https://transformer-circuits.pub/2024/jan-update/index.html)，其中概述了一系列初步研究笔记和该领域的发展。主题包括**注意力叠加（attention superposition）、MNIST 上的字典学习**以及**多层模型中的特征**。

- **报告拼写错误观察**：用户 `@ishitatsuyuki` 指出分享的 Anthropic 可解释性团队更新中存在一个拼写错误，标题为 "Joint Superposition Between MLP Neurons and Residual Stream" 的内容在 "Counterexamples in Superposition" 章节中**重复出现了两次**。

**提到的链接**：

[Circuits Updates - January 2024](https://transformer-circuits.pub/2024/jan-update/index.html)：未找到描述

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1199648537051406366) (7 条消息): 

- **关注并行范式**：`@groggyrhombus` 询问 `@337128969059172353` 的方法是否类似于 DeepSpeed 的张量+专家并行（tensor+expert parallelism），`xyzzyrz` 简单地回答了 **"Yeah"**。
- **算力方面的显著进展**：`@tastybucketofrice` 对 `@337128969059172353` 的努力表示感谢，并提到计划在下周测试该算力。
- **深入探讨 CUDA 初始化问题**：`@tastybucketofrice` 链接到了 [DeepSpeed 中的一个 issue](https://github.com/microsoft/DeepSpeed/issues/1987)，涉及 fork 之前的 CUDA 初始化问题，并表示有兴趣对此进行进一步调查。
- **对测试错误的怀疑**：`@catboy_slim_` 怀疑最近 CUDA 或 pytest 处理 forking 的方式发生了变化，这可能导致了测试错误，但指出目前缺乏时间来深入研究该问题。
- **欢迎协助修改 Pull Request**：`@catboy_slim_` 暗示由于目前的限制，将从 Pull Request 中排除某些部分，并欢迎其他人接手这项任务。

**提到的链接**：

[Issues · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/issues/1987)：DeepSpeed 是一个深度学习优化库，使分布式训练和推理变得简单、高效且有效。- Issues · microsoft/DeepSpeed

  ,

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1199626308687241236) (59 messages🔥🔥): 

- **翻译器困扰**：`@le_mess` 分享了一个 **Tenor GIF** 链接，该链接似乎根据浏览器设置进行了语言翻译。
- **寻求协助**：`@nanobitz` 建议某人（可能是 `<@525830737627185170>`）可以协助处理一个问题，同时 `@dangfutures` 请求关于 replicate 配置的帮助，`@mistobaan` 则询问关于微调 `phi-2` 的指导。
- **解释“高上下文”**：针对 `@hamelh` 关于“高上下文（high context）”含义的提问，`@jsancs_` 澄清其指的是“8-10k 上下文窗口”。
- **微调的挫折与成功**：`@dangfutures` 随口提到又是微调的一天，而 `@nafnlaus00` 讨论了从 `float16` 切换到 `bfloat16` 时所需的学习率差异。
- **GPU 采购决策**：大家对硬件选择进行了辩论；`@yamashi` 在 8 个 `H100` 或 16 个 `A100` GPU 之间权衡，`@casper_ai` 建议选择更大的 VRAM。`@dangfutures` 支持选择 16 个 `A100` 以获得更多 VRAM，而 `@c.gato` 期待 Caseus 在公告频道发布的公告。

**提到的链接**：

- [加入 Replicate Discord 服务器！](https://discord.gg/replicate)：查看 Discord 上的 Replicate 社区 - 与其他 22243 名成员交流，享受免费的语音和文字聊天。
- [Magic GIF - Magic - 发现并分享 GIF](https://tenor.com/view/magic-gif-26166638)：点击查看 GIF
- [[BOUNTY] 更新 ReLoRA 实现 · Issue #1198 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1198)：⚠️ 请检查此功能请求之前是否已被提出。我搜索了讨论区之前的 Ideas，没有发现类似的请求。我搜索了之前的 Issues，也没有发现...

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1199623649376870440) (13 messages🔥): 

- **Mamba 与 Lora 的兼容性尚不明确**：`@tank02.` 询问 **Mamba** 是否支持 **Lora**，但 `@le_mess` 回复表示不确定，并称他们**从未训练过 Mamba**。
- **上传模型至 HF 遇到困难**：`@colejhunter` 在将训练好的模型推送到 **Hugging Face** (HF) 时遇到问题；仓库已创建但没有填充模型文件。他们提供了配置并询问除了 `hub_model_id` 之外是否还缺少其他设置。
- **训练中的保存步数策略**：`@caseus_` 和 `@noobmaster29` 建议使用 `save_steps` 或 `saves_per_epoch`，并给出了如 `saves_per_epoch: 1` 或 `saves_per_epoch: 10` 的变体，用于控制训练期间模型的保存频率。
- **不指定具体步数保存模型**：`@c.gato` 提到可以将保存设置留空，这样默认会在每个 epoch 结束时保存模型。
- **寻求预训练 CLIP 的指导**：`@emperor` 正在寻找讨论进一步预训练 **CLIP** 最佳实践的论文、研究或博客文章，特别是专注于通过 5000 万张图像的对比目标（contrastive objective）进行领域自适应。

---

### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1199628958296190986) (15 messages🔥): 

- **挖掘到 GitHub 宝库**：`@dangfutures` 向 `@nanobitz` 分享了一个宝贵的 [GitHub 仓库](https://github.com/abachaa/Existing-Medical-QA-Datasets)，其中包含医疗领域的多模态问答数据集。
- **资金困扰**：`@yamashi` 哀叹自己正忙于为获取**计算资源资金**而处理行政事务，这是在回应 `@dangfutures` 对其缺席的询问。
- **寻找完美的 Prompt**：`@builderx` 询问了用于模型推理的正确 **alpaca 格式 prompt**，随后与 `@c.gato` 共同进行了澄清。
- **Alpaca 提示词中的重复问题**：`@builderx` 遇到一个问题，即在 Mistral 上进行训练后，模型输出中偶尔会重复 **alpaca prompt**，`@c.gato` 建议寻求帮助。
- **格式对于 QA 训练至关重要**：在由 `@neko.huh` 发起的关于原始文本是否应转换为 **alpaca QA 格式** 进行训练的讨论中，`@noobmaster29` 评论说，如果训练重点是问答，那么这样做可能会有好处。

**提到的链接**：

[GitHub - abachaa/Existing-Medical-QA-Datasets: 医疗领域的多模态问答：现有数据集和系统摘要](https://github.com/abachaa/Existing-Medical-QA-Datasets)：医疗领域的多模态问答：现有数据集和系统摘要 - GitHub - abachaa/Existing-Medical-QA-Datasets: 医疗领域的多模态问答...

---

### OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1199812920255729745) (1 messages): 

- **Axolotl v0.4.0 正式发布**：OpenAccess AI Collective 宣布发布 [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/releases/tag/v0.4.0)，其特点是支持新模型、修复了大量 Bug，并获得了 56 位个人的贡献。`@caseus_` 向所有人表示感谢，并特别鸣谢 A16Z 的资助，承诺将在下周添加 Discord 贡献者角色。

- **致谢社区贡献**：`@caseus_` 在公告中通过提及（Tag）的方式感谢了个人贡献者，并邀请未被提及的人员通过私信（DM）提供其 GitHub 和 Twitter 账号以便进行表彰。列出的贡献者包括 `@213644857309134849`、`@244959984352231425` 等。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1199636289234935819) (1 messages): 

- **对 Replicate Serverless 冷启动的好奇**：用户 `@dreamgen` 询问了目前 Serverless 服务在 **Large Models** 上的冷启动时间，并指出过去曾遇到过服务商 **不缓存模型或 Docker 镜像** 的问题。他担心尽管技术有所进步，加载时间本应为几秒钟，但实际体验却各不相同。
  ,

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1199654289170243594) (49 messages🔥): 

- **Lumiere 领跑**：`@swyxio` 重点介绍了 Google 的 [Lumiere 模型](https://lumiere-video.github.io/)，这是一种从文本生成视频的时空扩散模型（Space-Time Diffusion Model）。他们还分享了关于该模型特性的[文章](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/)，包括其令人印象深刻的 Inpainting 能力。
- **挑战 Google 的代码保守主义**：`@guardiang` 评论了 Google 不愿分享 AI 相关代码的态度，`@shivdinho` 表示赞同，并指出在不发布代码的情况下很难复现 Google 的研究。
- **探索 AI 的 Self-Instruct**：`@youngphlo` 提供了一个[链接](https://arxiv.org/abs/2212.10560)指向一篇讨论 Self-Instruct 的论文，这是一种通过 LLM 自身生成内容来引导（Bootstrapping）并增强模型的方法。
- **讨论合适的 AI 语言模型架构**：`@bathientran` 寻求关于 Infra/Backend 开发者与经验丰富的语言模型开发者之间协作的建议，`@philltornroth` 回应表示愿意帮助弥合沟通鸿沟。
- **Discord LLM Chatbot 的潜在项目**：`@swyxio` 分享了一个由 LLM 驱动的 Discord 聊天机器人的 [GitHub 仓库](https://github.com/jakobdylanc/Discord-LLM-Chatbot)链接，并询问是否有人有兴趣为该频道实现它。

**提及的链接**：

- [Weak-to-strong generalization](https://openai.com/research/weak-to-strong-generalization)：我们提出了 Superalignment 的新研究方向，以及极具前景的初步结果：我们能否利用深度学习的泛化特性，通过弱模型来控制强模型……
- [Lumiere - Google Research](https://lumiere-video.github.io/)：Google Research 开发的时空文本转视频扩散模型。
- [ai-notes/Resources/BENCHMARKS.md at main · swyxio/ai-notes](https://github.com/swyxio/ai-notes/blob/main/Resources/BENCHMARKS.md)：为软件工程师快速跟进 AI 新进展而准备的笔记。作为 https://latent.space 写作和产品脑暴的数据存储库，但也整理了规范的参考资料……
- [来自 ChatGPT (@ChatGPTapp) 的推文](https://x.com/chatgptapp/status/1750316948444086697?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我们在两周前推出了 GPT Store，想分享一些我们推荐的 GPT！
- [GitHub - jakobdylanc/Discord-LLM-Chatbot](https://github.com/jakobdylanc/Discord-LLM-Chatbot)：从 OpenAI / Mistral API 中选择 LLM，或使用 LM Studio 运行本地模型 • 多用户聊天 • 流式响应 • 200 行代码。
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)：大型“指令微调”语言模型（即经过微调以响应指令的模型）已展示出卓越的零样本泛化到新任务的能力。然而，它们高度依赖于……
- [[AINews] Google Solves Text to Video](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/)：2024年1月23日的 AI Discord 动态。我们为您检查了 19 个公会、291 个频道和 4199 条消息。节省了约 348 分钟的阅读时间（按 200wpm 计算）。Lumiere -...

  

---

### Latent Space ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1199802649428623381) (2 条消息): 

- **Self-Instruct 论文导读**：用户 `@swyxio` 宣布了一场由 `<@556359685306056721>` 主持的会议，旨在新的 [`ai-event-announcements`](https://discord.com/channels/822583790773862470/1197350122112168006) Stage 频道中引导大家阅读 Self-Instruct 论文。社区成员受邀参加，并被提醒在 [这里](https://lu.ma/ls) 注册以接收未来活动的通知。

- **获取 Latent Space 活动最新动态**：[Latent.Space 活动](http://Latent.Space) 日历支持订阅，点击日历右侧上方的 RSS 图标即可在有新活动时获得通知。

- **在旧金山举行的 Final Frontiers 庆祝活动**：`@fanahova` 分享了对即将到来的 Latent Space Demo Day 周年纪念以及在旧金山举行的名为 **Final Frontiers** 庆祝活动的兴奋之情。更多详情可以在 [Twitter](https://twitter.com/FanaHOVA/status/1750288311594418632) 上找到。

**提到的链接**：

[Latent Space (Paper Club &amp; Other Events) · Luma](https://lu.ma/ls)：在 Luma 上查看并订阅来自 Latent Space (Paper Club &amp; Other Events) 的活动。Latent.Space 活动。请点击日历右上方正对着的 RSS 图标以添加到您的日历。&quot;Ad...

  

---


### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1199772640349519912) (19 条消息🔥): 

- **用于论文讨论的新 Stage 功能**：`@swyxio` 向成员们介绍了如何利用 Discord 的新 Stage 功能与 `@556359685306056721` 进行论文讨论，并提供了 [Stage 频道的链接](https://discord.com/channels/822583790773862470/1197350122112168006)。
- **电梯音乐增添氛围**：`@picocreator` 开玩笑说在等待新的 Discord Stage 功能启动时加入了电梯音乐。
- **Discord Stage 缺失表情符号回应功能**：`@420gunna` 对在 Discord Stage 会话期间无法使用表情符号（emoji）回应表示失望。
- **论文讨论参与人数众多**：`@swyxio` 强调在切换平台后，第一场论文会议有 40 人参加，而 `@youngphlo` 推测在 Luma 重置的初始周内，可能有更多人尝试加入。
- **预告下一篇论文**：`@swyxio` 分享说，待确认后，下一篇讨论的论文可能是 [Pythia](https://arxiv.org/abs/2304.01373)，届时可能会有几位作者出席，Quentin Anthony 也有可能在会议后半段加入。此外，还推荐了 `@rasbt` 的 Twitter 线程以获取更多见解（[Twitter 来源](https://twitter.com/rasbt/status/1734920232173539796)）。

**提到的链接**：

- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)：大型语言模型 (LLMs) 在训练过程中是如何发展和演变的？这些模式如何随模型规模的变化而改变？为了回答这些问题，我们推出了 \textit{Pythia}，一套包含 16 个...
- [来自 varepsilon (@var_epsilon) 的推文](https://x.com/var_epsilon/status/1750034151065948379?s=20)：这篇解释 Twitter 的社区笔记 (Community Notes) 功能如何运作的论文非常酷 https://arxiv.org/abs/2210.15723
- [Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation](https://arxiv.org/abs/2210.15723)：我们提出了一种方法，用于为社交媒体帖子选择客观上具有信息量且主观上提供帮助的注释。我们利用来自在线环境的数据，在该环境中，贡献者对误导性信息进行注释...

  

---


### Latent Space ▷ #[llm-paper-club-chat](https://discord.com/channels/822583790773862470/822583791217934366/1199855024868708432) (1 条消息): 

- **使用 RESTful APIs 的 Autonomous Agents**：`@swyxio` 分享了一个有趣的旧项目/论文，名为 *RestGPT*，它专注于一个通过 **RESTful APIs** 控制现实世界应用程序的 **基于 LLM 的 Autonomous Agent**。该项目可以在 GitHub 上的 [这里](https://github.com/Yifan-Song793/RestGPT) 找到。

**提到的链接**：

[GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs](https://github.com/Yifan-Song793/RestGPT)：一个通过 RESTful APIs 控制现实世界应用程序的基于 LLM 的 Autonomous Agent - GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1199635794449682522) (62 messages🔥🔥): 

- **求职困扰与提示**：`@ziper_rom1` 询问了他们申请的 CUDA 开发者职位，表示在申请一个月后没有收到回复，感到有些困惑。包括 `@mrdragonfox` 和 `@frosty04212` 在内的社区成员分享了见解，认为没有回复通常意味着被拒绝，且个人人脉往往会影响招聘决策。
- **CUDA 开发者职位可能已招满**：`@kim_tech` 提供了一份更新，称 `@ziper_rom1` 在 Mistral 申请的 CUDA 开发者职位可能已经招满，并引用了一篇关于来自 Nvidia 的新员工的 [Twitter 帖子](https://twitter.com/sandeep1337/status/1744399578269352293?t=6Vm_qBKAAgBHLHyVPwK13g&s=19)。
- **在纯 CPU 环境下运行 Mistral**：`@tominix356` 询问了如何在没有 GPU 的 16GB RAM 电脑上运行 Mistral 7M，收到的建议包括尝试 LM Studio，以及 `@kim_tech` 关于在大内存笔记本上运行类似模型的经验反馈。
- **为工作寻找合适的工具**：由 `@xeglion` 发起的关于 RTX 3080 最佳用途的讨论，演变成了关于根据具体需求选择 Github Copilot 和 Google Bard 等工具的建议，`@kerunix`、`@mrdragonfox` 和 `@enerv` 也针对不同的选项和限制发表了看法。
- **本地 AI 代码辅助选项**：讨论了对本地 AI 代码补全的需求，`@enerv` 提到了将 Mistral 模型与本地 API 和插件结合使用的潜力。对话涉及了 Codeium 和 Sourcegraph 的 Cody 等选项，突显了开发者可用的工具多样性。

**提到的链接**：

[Cody | AI coding assistant](https://sourcegraph.com/cody)：Cody 是用于编写、修复和维护代码的功能最强大且最准确的 AI 编程助手。

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1199830072966791249) (1 messages): 

- **Mistral RAG 性能检查点**：用户 `@duck` 报告了在 3090 上使用 Langchain 执行某种 RAG 并使用 llama.cpp 进行推理时 Mistral 8x7b 的耗时。他们指出 102 次运行的 **sample times** 为 17.05 ms，而 101 次运行的 **eval times** 高达 175910.47 ms，认为这些耗时对于该用例来说太慢了。

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1199847209408139345) (1 messages): 

- **Mixtral 内存混乱**：用户 `@l0gr1thm1k` 在尝试将 Mixtral 加载到内存时遇到了 **CUDA memory errors**。尽管使用了四块各有 16GB 显存的 NVIDIA T4，但内存占用超过了 4bit 量化版本预期的 24GB，并导致了错误。
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1199728095410794577) (1 messages): 

- **Mistral-7B 在 Dolly 数据集上的指标困惑**：用户 `@bishwa3819` 正在尝试在 **Dolly 数据集上微调 Mistral-7B**，但对使用 **BLEU 和 ROUGE 指标** 评估语言模型性能的充分性表示困惑。他们质疑这些指标是否足以评估在像 Dolly 这样的特定数据集上训练的语言模型。
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1199750203679445013) (2 messages): 

- **带有 Mistral 特色的快速 Reddit 助手**：`@hugoduprez` 重点介绍了 **Reddit copilot buddy** 的创建，这是一个使用 **Mistral** 制作的机器人，其运行速度非常快，以至于看起来像是在线下的。
- **音频理解的新见解**：`@shashank.f1` 讨论了一种音频理解的新方法，并分享了一个 [YouTube 视频](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF)，该视频深入探讨了 **A-JEPA 神经模型**，该模型可以从 .wav 或 .mp3 音频文件中解锁语义知识。

**提到的链接**：

[A-JEPA neural model: Unlocking semantic knowledge from .wav / .mp3 audio file or audio spectrograms](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF)：🌟 解锁 AI 从音频中学习的力量！🔊 观看与 Oliver、Nevil、Ojasvita、Shashank、Srikanth 和 N... 深入讨论 A-JEPA 方法的视频。

  

---

### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1199754578716008498) (3 messages): 

- **Mistral API 摘要功能的局限性**：用户 `@nico2412_` 询问如何使用 **Mistral API** 通过 URL 摘要网页上的文章，并表示在完成此任务时遇到困难。
- **LLM 缺乏互联网访问权限**：`@mrdragonfox` 澄清说，像 Mistral 这样的大型语言模型 (LLM) 没有直接的互联网访问权限，这就是为什么它们无法通过网页 URL 调用函数进行摘要。
- **文章摘要的替代方案**：用户 `@duck` 提出了一个替代方法，提供了一个 [GitHub notebook](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb) 的链接，该 notebook 概述了使用语言模型摘要网页内容的过程。

**提到的链接**：

[LLM/llama-cpp-rag - final.ipynb at main · Quad-AI/LLM](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb)：通过在 GitHub 上创建账号，为 Quad-AI/LLM 的开发做出贡献。

  ,

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199631890978717767) (39 messages🔥): 

- **面向 VFX 艺术家的 AI 研究调查**：`@jordibares` 分享了一个 [调查链接](https://forms.gle/vexRuFUVHoojnfah7)，旨在征求 VFX 艺术家和制作人的见解，以纳入一项 AI 研究中。
- **`createSpace` 的配额重置请求**：`@troymurs` 因 canister 崩溃请求重置 `createSpace` 配额，并联系了 `<@907238188978950215>`。`@osanseviero` 回复建议发送邮件至 website @ huggingface.co。
- **微调文本生成模型**：`@sookeyy` 正在寻找微调文本生成模型的资源，在遇到错误后，`@robolicious` 建议使用 `remove_unused_columns=False`。
- **对协作项目的兴趣**：用户表达了对协作项目的兴趣，`@wondeys` 正在寻找合作伙伴来创建德州扑克 AI 或自动交易算法，而 `@dsiegel` 正在招募人员从零开始构建增强现实 (Augmented Reality) 头显。
- **组织葡萄牙语模型和数据集 Sprint**：`@namayra` 正在为葡萄牙语模型和数据集组织一场 Sprint/黑客松，在无法联系到 Omar Espejel 后，`@osanseviero` 指导其通过 website@huggingface.co 联系 Hugging Face。

**提到的链接**：

- [mathathon - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/mathathon)：未找到描述
- [app.py · Tonic/StableMed_Chat at main](https://huggingface.co/spaces/Tonic/StableMed_Chat/blob/main/app.py)：未找到描述
- [Survey on the use of Artificial Intelligence in the visual effects industry](https://forms.gle/vexRuFUVHoojnfah7)：未找到描述

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1199787478211641425) (1 messages): 

- **Alexa 的更环保替代方案**：`@mattbcool` 正在探索如何利用 Alexa 硬件创建一个本地个人助手，旨在重新利用扬声器和麦克风以减少废物。他们一直在研究最近的 Raspberry Pi 项目，并在[个人博客文章](https://mattcool.tech/posts/you-can-have-an-open-source-personal-assistant-in-2024)中记录了进展。
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1199833369211584552) (2 messages): 

- **Deep Learning.ai 推出自动化 LLM 测试课程**：`@manialgie` 分享了与 **CircleCI** 合作的[短课链接](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)。该课程旨在教授如何使用**持续集成 (continuous integration)** 工具更高效地评估 **LLM applications**。
  
- **3DTopia 让 Text-to-3D 变得简单**：`@meatfucker` 发现了 [GitHub 上的 3DTopia](https://github.com/3DTopia/3DTopia)，其中包含**模型权重**和**推理代码**，承诺在 5 分钟内实现 **Text-to-3D Generation**。他们提到虽然还没试过，但对于更简单的文本生成 3D 看起来很有前景。

**提到的链接**：

- [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)：未找到描述
- [GitHub - 3DTopia/3DTopia: Text-to-3D Generation within 5 Minutes](https://github.com/3DTopia/3DTopia)：5 分钟内实现 Text-to-3D 生成。通过在 GitHub 上创建账号，为 3DTopia/3DTopia 的开发做出贡献。

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1199650676033196152) (3 messages): 

- **应对实时转录挑战**：`@apocalypse3917` 提出了关于实时转录中**背景噪音**问题的担忧，询问是否有客户端解决方案或自动校准功能来处理此问题。`@ggabe_2` 表示赞赏并解释说，他们针对 **Whisper 的 PoC** 并未专门解决环境噪音问题，但建议 **Voice Activity Detection (VAD)** 过滤器可以在一定程度上缓解该问题。
  
- **用于 Steering Vectors 的新 Python 模块**：`@mihai4256` 宣布创建了一个可与 **Hugging Face 的 transformers** 配合使用的 **Python 模块**，用于添加 Steering Vectors。他们分享了一个推文链接，其中可能包含有关该模块的更多详细信息：[View Tweet](https://twitter.com/m_chirculescu/status/1750149970026479720)。
  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1199646847199887411) (3 messages): 

- **向量搜索简化**：`@united_dove_38339` 分享了一篇信息丰富的[博客文章](https://blog.kusho.ai/a-primer-on-vector-search-using-pinecone-serverless/)，详细介绍了如何使用 **Pinecone Serverless** 实现向量搜索，这对于利用 LLM 和 RAG 的现代应用至关重要。Pinecone 的 serverless 解决方案旨在简化向量搜索的实现并降低成本。

- **演示准备更新**：`@chad_in_the_house` 向小组通报了演示准备工作的延迟，并指出所涉及论文的复杂性。演示预计将在本周末或下周五前准备就绪。

- **Unsloth 加速微调**：`@zigglewomp` 介绍了一篇 [Medium](https://medium.com/@drishtisharma96505/accelerate-llama-2-7b-fine-tuning-how-unsloth-outpaced-standard-and-flash-attention-ef2ba8a1131d) 文章，讨论了 **Unsloth** 技术，该技术在提高 Llama2–7b 模型的内存效率和训练性能方面表现出了潜力。

**Links mentioned**:

- [A primer on vector search using Pinecone Serverless](https://blog.kusho.ai/a-primer-on-vector-search-using-pinecone-serverless/): Pinecone 最近推出了其 serverless 产品，旨在使向量搜索的实现更加简单且具有成本效益。随着 LLM 迅速成为应用程序的核心部分，且大部分...
- [Accelerate Llama-2–7b Fine-tuning: Unsloth Outpaces Flash Attention-2](https://medium.com/@drishtisharma96505/accelerate-llama-2-7b-fine-tuning-how-unsloth-outpaced-standard-and-flash-attention-ef2ba8a1131d): 本研究的目标

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199742668188680214) (5 messages): 

- **寻找最具挑战性的数据集**：用户 `@idkman2021` 询问哪个数据集最难处理。然而，该问题之后没有具体的回答或进一步讨论。
  
- **视频合成挑战分享**：用户 `@archer_cs` 寻求关于使用目标音频和参考视频进行视频生成的项目思路。他们分享了一个 [GitHub discussion](https://github.com/huggingface/diffusers/discussions/6696)，概述了他们这个雄心勃勃的项目细节。

- **改进 GPT-4 中的流式结构**：用户 `@kiraultra` 表达了在使用 **GPT-4 turbo API** 时对流式结构的困扰，提到在流式传输结束前无法获取到项目符号（bullet points）的问题。聊天中未出现具体的解决方案或后续问题。

- **Logit 输出差异**：用户 `@sherlockzoozoo` 发布了一个代码块，展示了他们使用 `meta-llama/Llama-2-7b-hf` 模型和 tokenizer 的实现，并询问为什么 `out_toks['scores']` 和 `just_out['logits']` 中的值存在差异。该问题目前尚无回应，Logit 差异的本质仍未得到解释。

**Links mentioned**:

[Video generation using target audio and reference video. · huggingface/diffusers · Discussion #6696](https://github.com/huggingface/diffusers/discussions/6696): 我正在进行一个个人项目，涉及：输入参考视频和目标音频，合成目标视频（由目标音频驱动的对口型说话人视频生成）。我...

  

---

### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1199668017030778901) (4 条消息): 

- **增强自定义 PyTorch 模型**：用户 `@nielsr_` 为自定义 PyTorch 模型提供了一个宝贵的建议，推荐使用 **[`huggingface_hub` 库中的 mixins](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)** 来轻松添加 `from_pretrained` 和 `push_to_hub` 功能。
- **寻求 Idefics 项目见解**：`@besiktas` 询问在哪里可以向 **Idefics** 项目背后的团队提问；`@osanseviero` 将他们引导至该项目的 **[HuggingFace 仓库讨论页](https://huggingface.co/ideas/discussions)**。

**提到的链接**：

[Mixins & 序列化方法](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)：未找到描述

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1199651088865972296) (5 条消息): 

- **Azure API 中的速率限制警告**：用户 `@kyko6969` 在结合使用 Azure 的 OpenAI API 中的 `embed_with_retry` 和 `langchain` 时遇到了速率限制警告问题。他们希望捕获该警告并实现 `time.sleep()` 函数来处理 API 的速率限制。
  
- **为普什图语（Pashto）训练 BPE Tokenizer**：`@imranullah` 询问了如何为普什图语等低资源语言训练 BPE Tokenizer，因为他遇到了输出乱码的问题。 

- **使用 LoRA 进行高效微调**：`@gugaime` 提到可以通过利用 LoRA 在微调期间进行量化（quantization），其中基础模型保持量化状态但被冻结。

- **开源文本转语音（Text-to-Speech）推荐**：`@mattbcool` 介绍了 **[Coqui AI TTS](https://github.com/coqui-ai/TTS)** 工具包，作为一个潜在的开源、本地 Text-to-Speech 解决方案。

- **mBARTforCausalLM 中神秘的权重**：`@vikas.p` 询问了 Hugging Face 的 transformers 仓库中 `mbartforcausallm` 模型里 lm head 权重被设置为绑定（tied）的问题，寻求澄清它是否与 input embedding 绑定，以及是否隐藏了执行此操作的方法。

**提到的链接**：

- [Dynamics 365 Customer Voice](https://aka.ms/oai/quotaincrease)：未找到描述
- [transformers/src/transformers/models/mbart/modeling_mbart.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mbart/modeling_mbart.py#L1926)：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习。 - huggingface/transformers
- [GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production](https://github.com/coqui-ai/TTS)：🐸💬 - 一个用于 Text-to-Speech 的深度 learning 工具包，经过研究和生产的实战测试 - GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and pr...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199742668188680214) (5 条消息): 

- **寻求关于数据集挑战的意见**：用户 `@idkman2021` 询问了 ***最具挑战性的数据集***，但未提供“挑战性”一词的背景，也未指定领域或应用。

- **创新视频生成技术**：用户 `@archer_cs` 征求关于一个从参考视频和目标音频生成视频的项目想法，并链接到了 [GitHub discussion #6696](https://github.com/huggingface/diffusers/discussions/6696) 以获取更多细节。该项目旨在创建一个由目标音频驱动的唇形同步（lip-synced）谈话头像视频。

- **改进 GPT-4 Turbo API 中的流式结构**：用户 `@kiraultra` 提出了在使用 gpt-4 turbo API 时 **流式结构（streaming structure）** 的问题，提到项目符号和其他结构元素仅在流结束时才出现，寻求增强这一方面的建议。

- **理解模型输出的差异**：用户 `@sherlockzoozoo` 分享了一段代码片段，查询 **LLama-2-7b 模型** 的两个输出，并询问为什么 `out_toks['scores']` 和 `just_out['logits']` 中的值不同。他们正在寻求澄清其中哪一个代表“真实的 logits”。

**提到的链接**：

[使用目标音频和参考视频生成视频 · huggingface/diffusers · Discussion #6696](https://github.com/huggingface/diffusers/discussions/6696)：我正在做一个个人项目，涉及：输入参考视频和目标音频，合成目标视频（由目标音频驱动的唇形同步谈话头像视频生成）。我...

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199659240265097256) (41 条消息🔥): 

- **Google 发布 Lumière**：`@spirit_from_germany` 分享了 `@omerbartal` 的一条推文，介绍了 **Lumiere**，这是 GoogleAI 推出的一款新型 video diffusion model，支持 text-to-video、image-to-video、stylization 等功能。社区对该模型的能力议论纷纷，但 `@mkaic` 指出该模型并未开源，导致群组内对其潜力与真实感（realism）的反应褒贬不一（[阅读关于 Lumière 的论文](https://arxiv.org/abs/2401.12945)）。
  
- **LAION 数据库访问问题**：由于 Huggingface 禁用了下载，用户 `@_chenwang` 和 `@djdhdjjdjdjdj` 提出了关于下载 LAION-en-aesthetics 标注（captions）的问题。

- **Lumière 与 EMU Video 的对比**：`@mkaic` 和 `@thejonasbrothers` 正在讨论 Google 的 Lumière 还是 Meta 的 EMU 视频模型看起来更真实，批评意见指向 AI 生成内容中偶尔出现的不一致和不自然现象。

- **热情与质疑并存**：在 Google 发布新模型 Lumière 的背景下，`@kilgore.trout` 询问了用于视频风格化（video stylization）的最佳开源模型，而 `@.undeleted` 等人则评论了 AI 生成视频中依然存在的“恐怖谷”效应（uncanny nature）。

- **RWKV 与 Mistral 的竞争讨论**：`@SegmentationFault` 分享了一个 Reddit 链接，讨论 RWKV 7B 的性能，其在多语言支持方面可能达到了 Mistral 7B 的水平，并具有 linear runtime 和高效 CPU 利用率等额外优势。



**提到的链接**：

- [来自 Omer Bar Tal (@omerbartal) 的推文](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19)：介绍 Lumiere 📽️ 我们在 @GoogleAI 开发的新型 video diffusion model * Text-to-Video * Image-to-Video * Stylized Generation * Inpainting * Cinemagraphs 等 🎨 伴随惊人的...
- [民主化 AI 研发的未来：NSF 将启动国家 AI 研究资源试点](https://new.nsf.gov/news/democratizing-future-ai-rd-nsf-launch-national-ai)：弗吉尼亚州亚历山大市：今天，美国国家科学基金会（NSF）及合作机构启动了国家人工智能研究资源（National AI Research Resource）……
- [Lumiere - Google Research](https://lumiere-video.github.io/#section_image_to_video)：Google Research 开发的 Space-Time Text-to-Video 扩散模型。
- [Lumiere - Google Research](https://lumiere-video.github.io/)：Google Research 开发的 Space-Time Text-to-Video 扩散模型。
- [Reddit - 深入探索](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/)：未找到描述

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1199659282065535057) (15 条消息🔥): 

- **GoogleAI 发布 Lumiere - 视频扩散模型**：`@spirit_from_germany` 分享了来自 [Omer Bar-Tal](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19) 的推文，宣布了 GoogleAI 的新视频扩散模型 **Lumiere**，其功能包括 *Text-to-Video, Image-to-Video, Stylized Generation, Inpainting, Cinemagraphs* 等，`@mkaic` 随后附上了[配套的研究论文](https://arxiv.org/abs/2401.12945)。

- **GoogleAI 的 Lumiere 以其简洁性令人惊讶**：`@mkaic` 评论道，在初步浏览研究论文后，**GoogleAI Lumiere** 背后的方法看起来出奇地简单，尽管他们尚未深入研究细节。

- **Lumiere 潜在的训练场 - YouTube**：`@mkaic` 推测 **GoogleAI 的 Lumiere** 一定是在 YouTube 上进行训练的，因为它是最大的视频库，随后通过引用论文内容确认了这一点：“*我们在包含 30M 视频的数据集上训练了我们的 T2V 模型*”。

- **Google 在 T2V 模型方面拥有无可比拟的视频数据优势**：`@mkaic` 指出，由于 **Google** 拥有 YouTube，在训练 Text-to-Video 模型方面具有巨大优势，其中包括自动生成的字幕和数十亿条评论，这为训练视频多模态语言模型提供了庞大的数据集。

- **DeepMind 被视为 Project Gemini 的核心力量**：`@thejonasbrothers` 通过反思参与 **Gemini** 项目的研究人员数量，强调了 **DeepMind** 的显著努力，暗示了他们在 AI 研究方面的巨大资源和能力。


**提到的链接**：

- [来自 Omer Bar Tal (@omerbartal) 的推文](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19)：介绍 Lumiere 📽️ 我们一直在 @GoogleAI 开发的新视频扩散模型 * Text-to-Video * Image-to-Video * Stylized Generation * Inpainting * Cinemagraphs 等 🎨 伴随着惊人的 t...
- [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945)：我们介绍了 Lumiere —— 一种 Text-to-Video 扩散模型，旨在合成表现出真实、多样且连贯动作的视频 —— 这是视频合成中的一项关键挑战。为此，我们...

  ,

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1199755988388032582) (1 条消息): 

- **Vanna AI 在 SQL 生成领域一夜成名**：`@zain_hoda` 的项目 **Vanna AI** 凭借其简单而强大的界面引起了关注，该界面利用 **RAG** (Retrieval Augmented Generation) 来增强 SQL 查询的创建。该机器人具有为其操作存储和索引 DDL/表结构（table schemas）以及文本的能力。[LlamaIndex 推文](https://twitter.com/llama_index/status/1750196064660127848)
  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1199631950646870056) (49 条消息🔥): 

- **寻求更精细的 AI 响应机制**：用户 `@viky6453` 询问是否有办法让 `openaiagent` 的行为更像 `openai assistant`，后者会反复应用 `tool call`、消息和 `tool call` 直到响应被认为足够好，而不是像 LlamaIndex `openaiagent` 风格那样在多次 `tool call` 后仅发送一条消息作为响应。讨论中未提供明确的解决方案。
  
- **追求效率的技术探索者**：在寻求优化 `pandas query engine` 并配合 `response synthesizer` 实现上下文感知（context-aware）的过程中，`@techexplorer0` 表示希望能有类似 `chat_engine` 的功能，而 `@pk2594` 则想知道是否可以对 `query engine` 进行多线程处理以提高速度，并对其线程安全性（thread-safety）提出了疑问。

- **寻找完美的 LLM Chatbot 托管方案**：用户 `@basil11111` 思考是否可以在不进行本地托管的情况下使用开源模型，并从 `@nerdai` 的回复中了解到 HuggingFace 和 Replicate 等服务可以托管 LLM，并提供 API 和微调（fine-tuning）功能。

- **微调 RAG 的有效性**：讨论涉及了 RAG 应用的增强，`@0tarumi` 探索了 BGE 相似度重排序器（similarity reranker）的实现，而 `@cheesyfishes` 建议在 RRF 之后进行重排序（reranking）可能会获得最佳效果。

- **记忆的重要性**：`@techexplorer0` 寻求一种类似于 LangChain 的 `memory buffer` 的工具来追踪对话历史，`@cheesyfishes` 确认每个 `chat engine`/`agent` 中都使用了 `chat memory buffer`，对于那些寻求带记忆的对话式 Chatbot 的用户，可以在 LangChain 中配合使用 LlamaIndex。


**提到的链接**：

- [Home - Phoenix](https://phoenix.arize.com/)：未找到描述
- [LlamaIndexTS/examples/vectorIndex.ts at main · run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS/blob/main/examples/vectorIndex.ts)：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/LlamaIndexTS
- [LlamaIndexTS/packages/core/src/llm/azure.ts at main · run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS/blob/main/packages/core/src/llm/azure.ts)：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/LlamaIndexTS
- [Replicate](https://replicate.com/)：通过云端 API 运行开源机器学习模型
- [meta/llama-2-70b-chat – Run with an API on Replicate](https://replicate.com/meta/llama-2-70b-chat)：未找到描述
- [meta-llama/Llama-2-70b-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-70b-hf)：未找到描述

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (1 条消息): 

rawwerks: 👋 社区提问❓
你最喜欢的 vector store 公司是哪家，为什么？

### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1199770976922112070) (2 messages): 

- **X/Twitter 账号恢复**：`@.bagatur` 宣布 **X/Twitter 账号**已恢复，并将取消对所有受影响用户的屏蔽。如果有人仍被屏蔽，请在帖子中回复以寻求帮助。

- **LangChain 推出 Streaming API**：`@veryboldbagel` 分享了 LangChain 中 **streaming events**（流式事件）的新 API 文档链接，强调了其对于响应式终端用户应用的重要性。详细示例和说明可在 [General Docs](https://python.langchain.com/docs/expression_language/streaming) 以及 **AgentExecutor** 和 **LangGraph** 的文档中找到 ([AgentExecutor Docs](https://python.langchain.com/docs/modules/agents/how_to/streaming), [LangGraph Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb))。

- **观看并学习流式事件**：分享了一个题为 "Streaming Events: Introducing a new `stream_events` method" 的 [YouTube 视频](https://youtube.com/watch?v=ZcEMLz27sL4)，解释了 Streaming 在 LLM 应用中的重要性。

- **流式功能反馈请求**：鼓励用户在 LangChain 的 GitHub 讨论页面（详见 [此处](https://github.com/langchain-ai/langchain/discussions/16175)）上提供反馈并报告有关新流式功能的问题。

**提到的链接**：

- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/streaming#using-stream-events): 使用 LangChain 进行流式传输
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/how_to/streaming): Streaming 是 LLM 应用中重要的 UX 考量因素，而 Agent 是...
- [langgraph/examples/streaming-tokens.ipynb at main · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb): 通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。
- [Streaming Events: Introducing a new `stream_events` method](https://youtube.com/watch?v=ZcEMLz27sL4): Streaming 是大多数 LLM 应用的重要组成部分。既包括单个 token 的流式传输，也包括过程中发生的事件流。我们最近在...
- [🛸 Streaming: RFC Adding astream_event to all Runnable objects to help with streaming use cases · langchain-ai/langchain · Discussion #16175](https://github.com/langchain-ai/langchain/discussions/16175): 大家好！我们想改进 LangChain 中的流式体验。我们正考虑在 Runnable 接口中添加 `astream_event` 方法。以下代码来自相关的 PR，目前还没有...

  

---


### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1199629518474858496) (21 messages🔥): 

- **寻求开源透明度**：`@pirlog` 询问该项目是开源还是免费的，并指出网站上缺乏定价或解释性信息。
- **请求协助**：`@irfansyah5572` 和 `@shanumas` 表示需要针对非特定问题的帮助，表现出对协作解决问题的兴趣。
- **服务宕机与恢复记录**：`@adorable_quokka_56531` 提到服务似乎已宕机，但随后指出已恢复，并建议添加状态页面。
- **数据库 Schema 协助**：`@mavrik_55410` 寻求关于如何在 Postgres 数据库中使用 pgvector 和 LangChain 将向量嵌入（vector embeddings）存储在特定 Schema 中的指导，这引发了与 `@__ksolo__` 关于 Postgres Schema 和配置的澄清讨论。
- **社区关于向量存储的反馈**：`@rawwerks` 发起了关于最受青睐的向量存储公司的讨论，邀请社区发表意见。

**提到的链接**：

[5.9. Schemas](https://www.postgresql.org/docs/current/ddl-schemas.html): 5.9. Schemas # 5.9.1. 创建 Schema 5.9.2. Public Schema 5.9.3. Schema 搜索路径 5.9.4. Schemas 与权限 5.9.5. …

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1199963765467263037) (1 messages): 

- **寻求 LangServe Agent Executors 的指导**：`@hiranga.g` 在 LangServe 中运行 **agent_executor** 时遇到困难，并向社区寻求帮助。他们就设置过程直接向 `@1033432389516546158` 提出了查询。(Q1)

- **关于 LCEL 和 Tool 选择的澄清**：`@hiranga.g` 成功运行了 **LCEL**，但不确定是否可以添加多个 Tools 供 LCEL 使用。他们认为 LCEL 不允许创建 "Agent"，并就此事询问确认或指导。(Q2)

### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 

sideways1: 有人构建过能与 JSON 文件数据库交互的问答聊天机器人吗？
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1199859575508172891) (4 messages): 

- **AgentHub 发布**: `@maxbrodeururbas` 宣布推出了 [AgentHub](https://www.agenthub.dev/)，这是一个与朋友共同构建的平台，并邀请社区提供反馈。他们补充说，还写了一篇博文详细阐述了机器人流程自动化 (RPA) 与 AI 之间的协同作用，可以在[这里](https://www.agenthub.dev/blog/robotic_process_automation_with_ai)找到。
- **寻求合作**: `@truethinker` 联系了 `@939753620423991296` 表达了建立联系的兴趣，尽管没有提供关于连接目的的具体背景或细节。

**提到的链接**:

[AI and RPA: The Future of Work](https://www.agenthub.dev/blog/robotic_process_automation_with_ai): RPA 工具与 AI 的结合将在未来几年引发生产力的巨大爆发。

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1199750400849485894) (2 messages): 

- **通过 DataCamp 和 LangChain 深入探索多模态 AI**: `@datarhys` 分享了一个**免费的 9 部分 AI 系列课程**链接，其中包括“使用 LangChain 和 OpenAI API 构建多模态 AI 应用”的课程。该课程教参与者使用 Whisper 转录 YouTube 视频，然后向 GPT 提问关于转录内容的问题。查看[整个实战系列](https://www.datacamp.com/code-along)，并通过这个[实战链接](https://www.datacamp.com/code-along/multimodal-ai-applications-langchain-openai-api)开始特定的 LangChain 实战。

- **CircleCI 和 Deeplearning.ai 推出 AI 测试课程**: `@manialgie` 宣布与 Deeplearning.ai 合作发布了一门**免费课程**，关于如何测试和交付 AI 驱动的应用。该课程探讨了测试基于 LLM 的应用、模型分级评估 (model-graded evaluations)，以及使用 CircleCI 自动化这些流程以增强应用开发。课程地址：[Automated Testing with LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)。

**提到的链接**:

- [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/): 未找到描述
- [Building Multimodal AI Applications with LangChain &amp; the OpenAI API](https://www.datacamp.com/code-along/multimodal-ai-applications-langchain-openai-api): 结合文本和音频 AI 模型的力量，构建一个能回答关于 YouTube 视频问题的机器人。
- [Data Science &amp; AI Code Alongs](https://www.datacamp.com/code-along): 从数据可视化到 AI，与专家一起解决现实世界的问题。在屏幕录像的帮助下完成整个项目，让你永远不会卡壳。

  ,

### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1199824286257516604) (1 messages): 

- **GPT-3.5 在重要性和情感判断方面表现挣扎**: `@calytrix` 分享了他们使用 **GPT-3.5** 评估新闻报道的工作流，发现它在判断情感方面优于判断重要性，但仍存在*强烈的偏见*。他们指出，增加解释和描述性评分对提高系统的辨别能力几乎没有帮助。
- **应对模型偏见和评分挑战**: `@calytrix` 观察到模型在分配数字时有非常强烈的偏好，建议通过微调 (fine-tuning) 可能会缓解这些偏见。他们强调了任务的复杂性，并建议开发专门针对模型能力的评估和评分系统。
- **对评估复杂性的认可**: 由于 `@calytrix` 提到的**隐含复杂性**，评估重要性的任务对 GPT-3.5 来说特别困难，而人类却觉得这很容易。
- **评分系统建议**: `@calytrix` 建议将复杂问题（如评估重要性）拆解为 GPT-3.5 能够更有效处理的特定评分项。他们还建议使用带有有限选项的简化评分系统，例如 **“差、一般、好、非常好”**。
  

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1199631228857499669) (9 messages🔥): 

- **评估语言模型微调的数据质量**：用户 `@thewindmom` 询问在微调语言模型时如何评估数据质量，强调要避免不良的输入数据。`@hammadkhan` 建议使用 **eye-balling（目测检查）、deduplication（去重）和 heuristic-based filtering（基于启发式的过滤）** 等技术来维持数据质量。

- **自制合成数据减少评估需求**：针对 `@thewindmom` 的提问，`@bjoernp` 提出创建自己的合成数据可以降低对外部评估的需求，并暗示可以使用 **built-in guides** 和 **information-rich** 模型来生成此类数据。

- **讨论用于微调的合成数据生成**：`@thewindmom` 询问了合成数据的使用，特别是参考了 **NeMo Guardrails**，对此 `@bjoernp` 提到了带有 guardrails 的主动合成数据创建，强大的模型可以有效地利用这些数据进行训练。

- **分享合成数据和 Self-Instruct 方法的资源**：`@_jp1_` 分享说 DiscoResearch 仍在完善其合成数据生成过程，并强调 **jon durbins' Airoboros repo** 是一个关键参考，同时讨论了针对 self-instruct 论文的定制实现。他们提供了一个 [Airoboros 的 GitHub 链接](https://github.com/jondurbin/airoboros) 以供进一步深入了解和启发。

- **训练困境：使用 Instruction+Answer 对**：在由 `@philipmay` 发起的讨论中，该查询对使用 **instruction+answer pairs** 提出了担忧。他们询问在缺乏好的回答来创建用于 DPO 或其他训练工作的 **triplets**（三元组）的情况下，是否可以仅利用指令和坏的回答对模型训练做一些有用的事情。

**提到的链接**：

[GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros)：self-instruct 论文的可定制实现。

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1199665763955507270) (11 messages🔥): 

- **DPR 模型表现令人失望**：`@philipmay` 表示使用两个不同的 DPR 模型进行测试并未产生预期结果，称这种情况“非常奇怪”。
- **增强数据定位的努力**：`@sebastian.bodza` 计划添加一个新列来显示发现数据的位置，同时也承认了任务的复杂性，提到“相当棘手”。
- **通过求和距离来确定问题的特异性**：`@philipmay` 讨论了一种通过对顶部结果的距离求和来确定问题通用性的策略，并指出通用问题和具体问题通过这种方式无法得到有效区分。
- **识别模式的挑战**：`@sebastian.bodza` 评论说在文本与问题的相似性或顶部向量之间寻找模式非常困难，特别强调了由于“Pferdemarkt”存在多个实例而导致的相关问题。
- **头脑风暴完成及后续步骤**：`@sebastian.bodza` 宣布完成了包含 8.2万个搜索创意/问题的头脑风暴，强调下一阶段将是问题生成。
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199796648063283200) (5 messages): 

- **赞扬 DiscoLM German 7B v1**：`@alex_22398` 对 **DiscoLM German 7B v1** 的高质量语言输出表示感谢，即使是在旧笔记本电脑上借助 TheBloke 的 GGUF quantization 也能运行，同时也指出了输出后产生空行的问题。
- **空行 Bug 已修复**：`@_chromix_` 告知已修复输出后多余空行的问题，并讨论了生成更新的 **GGUF quantization** 或手动设置 stop token 为 `" 的选项。

**提到的链接**：

[DiscoResearch/DiscoLM-70b · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM-70b)：未找到描述


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1199870824451162253) (2 messages): 

- **与团队同步**：`@teknium` 联系了频道成员，表达了重新建立联系的愿望，并通过标记 `@748528982034612226` 鼓励同事间讨论近期的活动。未提及特定主题或链接。

### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1199630613586641006) (2 条消息): 

- **关于 ML 领域工作的公开咨询**：用户 `@forsaken_ninja` 就 Machine Learning 领域的工作机会向社区寻求咨询，邀请成员参与讨论并提供见解。他们鼓励其他人 ping 他们进行直接对话。
  ,,,,