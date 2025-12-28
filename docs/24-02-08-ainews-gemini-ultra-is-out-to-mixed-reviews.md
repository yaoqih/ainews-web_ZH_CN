---
companies:
- google
- openai
- mistral-ai
- hugging-face
date: '2024-02-09T05:58:08.478444Z'
description: '**谷歌**在停止使用 Bard 后，推出了 **Gemini Ultra**，作为“带有 Ultra 1.0 的 Gemini Advanced”付费版本。相关评论指出，它的表现“比
  ChatGPT 稍快或稍好”，但在推理能力上仍存在差距。**Steam Deck** 被视为一个令人惊喜的 AI 工作站，能够运行 Solar 10.7B 等模型。


  AI 社区的讨论涵盖了开源项目 **Unsloth** 的多 GPU 支持、源自 OpenAI 输出的训练数据污染、模型合并的伦理担忧，以及诸如列表偏好优化（LiPO）等新型对齐技术。**Mojo**
  编程语言因其在高性能计算方面的表现而受到称赞。


  在研究领域，**Subformer** 模型采用三明治式参数共享和 SAFE 机制来提升效率，而 **BiLLM** 则引入了 1-bit 训练后量化技术以降低资源消耗。此外，**OpenHermes**
  数据集查看工具正式发布，使用 Slurm 进行 GPU 调度也引发了讨论。针对 **OpenHermes-2.5-Mistral-7B** 等模型的微调挑战以及显存（VRAM）需求同样是热门话题。'
id: d33c7c5b-7cb7-49e7-97e3-b27de96a449f
models:
- gemini-ultra
- gemini-advanced
- solar-10.7b
- openhermes-2.5-mistral-7b
- subformer
- billm
original_slug: ainews-gemini-ultra-is-out-to-mixed-reviews
people: []
title: Gemini Ultra 已发布，评价褒贬不一。
topics:
- multi-gpu-support
- training-data-contamination
- model-merging
- model-alignment
- listwise-preference-optimization
- high-performance-computing
- parameter-sharing
- post-training-quantization
- dataset-viewer
- gpu-scheduling
- fine-tuning
- vram-optimization
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月7日的 AI Discord 动态。我们为你检查了 **20** 个公会、**311** 个频道和 **6128** 条消息。节省的预计阅读时间（以 200wpm 计算）：**496 分钟**。

照常进行，AI 领域整体上比较平静。随着 Bard 彻底退场，[Gemini Ultra 今日发布](https://blog.google/products/gemini/bard-gemini-advanced-app/)，作为“带有 Ultra 1.0 的 Gemini Advanced”付费层级。评测机构们已经开始行动了：

- [Fireship](https://www.youtube.com/watch?v=ucd63nIZZ60) 仅给出了非常轻微的赞赏，称其“比 ChatGPT 稍快/稍好”，但指出了许多差距。
- [AI Explained](https://www.youtube.com/watch?v=gexI6Ai3X0U) 同样评论了其更快的速度，但也发现了一些推理和视觉推理方面的缺陷。

 
![image.png](https://assets.buttondown.email/images/18b262e6-69df-40a9-abf9-3f5d41bca9ab.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/45c5fe91-f014-4bf9-b5bc-06c3423ecf7f.png?w=960&fit=max)


---

**目录**

[TOC] 


# 第 1 部分：高层级 Discord 摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **Steam Deck：意想不到的 AI 性能怪兽**：一位用户注意到 Steam Deck 在运行 AI 模型方面的潜力，像 Solar 10.7B 这样的模型表现出了显著的性能，这表明这款掌上游戏机可以作为便携式 AI 工作站这一意想不到的使用场景。

- **AI 哲学与数学的交汇**：正在进行的讨论深入探讨了 AI 是否具有类似于人类的“理解”能力，以及数学与物理宇宙的关系，引发了关于形而上学和意识本质的深刻辩论。

- **LLM 优化与训练数据污染的新方向**：讨论了为 OSS Unsloth 添加多 GPU 支持的努力，其中预量化（pre-quantization）的优势显著，在不损失精度的情况下减少了 1GB 的 VRAM 需求。会议还强调，大多数现代模型可能都是使用受 OpenAI 模型输出影响的数据进行训练的，这可能会影响它们独特的风格。

- **关于模型合并与数据集价值的新兴关注点**：围绕模型合并的伦理和实用性的对话包括了“禁止合并”许可的提案，以及创建数据集与模型合并相比面临的财务挑战。这些问题凸显了创新、所有权与 AI 研究免费开放性质之间复杂的相互作用。

- **模型对齐与训练创新不断涌现**：Listwise Preference Optimization (LiPO) 作为一种对齐语言模型的新框架被引入，这表明研究方向正在向更精细的响应优化技术转变。此外，LiPO 框架使用的 Ranker 足够小，可以在本地进行训练，这对于资源有限的用户来说是一个显著优势。

- **编程亮点：实现见解与语言进展**：分享了有效使用 Hugging Face 模型的实用建议，同时讨论了 Mojo 语言令人印象深刻的性能，有望成为高性能计算任务中值得关注的工具。此外，还探索了将数据库功能集成到 Bot 交互中以增强对话能力，强调了 Bot 复杂性的增长。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **使用 SAFE 提升 Transformer 的效率**：讨论了一篇 [arXiv 论文](https://arxiv.org/abs/2101.00234)，介绍了 **Subformer**。该模型应用了三明治式参数共享和自注意力嵌入因子分解 (SAFE)，以比传统 Transformer 更少的参数实现了更好的效果。

- **BiLLM 在 1-Bit 量化方面的突破**：**BiLLM** 声称在保持性能的同时显著降低了计算和内存需求。该研究通过一篇论文（[下载 PDF](/pdf/2402.04291.pdf)）介绍，重点关注 Large Language Models 的 1-bit 后训练量化。

- **OpenHermes 数据集查看器现已上线**：分享了由 `@carsonpoole` 开发的新工具，旨在辅助查看和分析 **OpenHermes** 数据集。其功能包括滚动浏览示例以及 Token 计数分析的过滤器。

- **GPU 与调度**：在一次讨论中，推荐使用 Slurm 来高效调度 GPU 上的作业。包括 `@chrisj7746`、`@Sebastian` 和 `@leontello` 在内的成员参与了关于最佳实践的对话。

- **培养模型在特定任务中的能力**：目前正在不断寻求改进 AI 模型性能的特定方面，例如**自定义预训练模型**在提取任务中的挣扎，以及关于 **OpenHermes-2.5-Mistral-7B** 等模型微调参数的咨询。用户还讨论了 8GB VRAM 对于微调操作是否充足，以及在动态 `.yml` 配置中设置特殊 Token 的问题。

- **模型架构与量化进展**：GPT-4 之后可能带有图灵完备（Turing complete）修改的架构变化是关注焦点。社区参与的方法论讨论也强调了量化在模型未来化（future-proofing）中的作用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- **LM Studio 中的 LaTeX 渲染问题**：`.neurorotic.` 等人表达了对 LaTeX 渲染不当的挫败感，特别是在 LM Studio 中使用 DeepSeek Math RL 7B LLM 时。
- **探索理想的 LLM 硬件配置**：在以硬件为中心的讨论中，运行本地 LLM 的理想配置仍是一个热门咨询点，成员们对 AMD Ryzen 5 7600 和 AMD Radeon RX 6700 XT 配置的兼容性表现出浓厚兴趣。PCIe 延长线（risers）及其对性能的影响是另一个热议话题。
- **对新兴语言模型的不同看法**：围绕 Qwen 1.5、Code Llama 2 以及各种 OpenAI 模型的性能和预期展开了激烈辩论。关键议题包括上下文理解、GPU 加速和代码生成质量。在 Open Interpreter (OI) 设置中，GPT-3.5 的使用被证实是令人满意的。
- **探索用于 DIY 语音项目的 ESP32 S3**：社区成员（尤其是 `@joelthebuilder`）参与了关于使用 ESP32 S3 Box 3 进行自定义家庭网络语音项目的讨论，旨在取代 Alexa 等标准解决方案。
- **Discord 和 LM Studio 上的 Open Interpreter**：强调了 Open Interpreter 利用 LM Studio 服务器模式的潜力，并建议查看 [OI Discord](https://github.com/KillianLucas/open-interpreter/) 以获取特定用例讨论和 Discord 集成建议。
- **LM Studio 中的 Autogen 问题**：`@photo_mp` 报告了 Autogen 内部的 userproxy 通信问题，并寻求社区推荐的解决方案。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **苏格拉底 AI 梗依然流行**：`@gabriel_syme` 幽默地推测，模仿苏格拉底的 AI 模型可能会导致单方面的对话，模仿这位哲学家在对话中的主导地位。
- **低成本破解 Llama 安全锁**：根据一篇 [LessWrong 文章](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email)，只需不到 200 美元的 **LoRA fine-tuning** 即可破解 **Llama Model** 的安全性，这引发了人们对绕过安全协议的简易性的担忧。
- **静默克隆与 GPT-Blind 的传闻**：关于语音克隆技术的演进以及一个新的、未指明的 GPT 模型的出现出现了各种猜测，这表明技术格局发生了重大变化，并可能对 iPhone 屏幕等现有硬件提出挑战。
- **LLM 论文俱乐部如火如荼**：**Latent Space Paper Club** 的一次会议讨论了 *Self-Rewarding Language Models*，其中一篇论文展示了能够利用输出进行奖励的模型，在某些测试中表现优于 Claude 2 和 GPT-4。点击[此处](https://arxiv.org/abs/2401.10020)阅读。
- **DSPy 吸引了下次俱乐部会议的关注**：用于链接 LLM 的模型 DSPy 引起了俱乐部的兴趣。即将举行的会议将探讨其功能，可以在[此处](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI)观看一段相关的 YouTube 视频。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **医疗保健领域的 Mistral**：`@talhawaqas` 正在寻找关于 **Mistral.AI 在健康工程中的应用**资源——特别是关于 pretraining 和 finetuning 的——而社区中的其他人则考虑为了方便将测试转移到云端。
- **数据政策与知识共享 (Creative Commons)**：`@mrdragonfox` 澄清说，该服务收集的用户对话数据可能会根据 **Creative Commons Attribution (CC-BY)** 许可发布。
- **聊天机器人优化与参数**：社区成员讨论了在 Mistral 中设置 `max_tokens` 和 temperature 的各种策略，以获得简洁的机器人回复，并辩论了使用零 temperature 以减少 hallucinations 并提高精准任务性能的做法。
- **与 Mistral 同步的 Embedding 模型**：`@gbourdin` 在 Hugging Face 上找到了 **E5-mistral-7b-instruct 模型**，它与 **Mistral-embed** 的维度长度一致，适用于本地开发，`@mrdragonfox` 提供了使用说明 [E5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct)。
- **工具介绍与界面展示**：Discord 上介绍了一些新工具和用户界面——一个是适配 Mistral chat API 的 *augmentoolkit*，另一个是 **ETHUX Chat v0.7.0**，它包含了 Huggingface ChatUI 和网页搜索功能，chat UI 可以在 [GitHub](https://github.com/huggingface/chat-ui) 上找到。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 摘要

- **HuggingFace 推出新招聘论坛**：建立了一个新的 **Forum** 频道用于发布职位机会，允许用户通过 `internship`、`machine learning`、`remote` 和 `long-term` 等标签筛选空缺职位。

- **GPU 和 Diffusion 模型讨论升温**：`@sandeep_kr24` 解释说 `pipeline.to("cuda")` 将计算图转移到 CUDA GPU，从而加速处理过程。会议讨论了 Stable Diffusion 的更新和支持查询，包括 **为像素艺术训练模型** 以及设置用于 LLM 查询的 web UI。

- **SD.Next 引入性能提升和新模块**：`@vladmandic` 宣布了 [SD.Next](https://github.com/vladmandic/automatic) 的更新，其特点是包含新的 **Control 模块**、**Face 模块**、**IPAdapter** 改进以及一系列模型，声称在 nVidia RTX4090 上的基准测试达到了 **110-150 iterations/second**。

- **关注 RAG Pipeline 相关性和 Deci lm 推理速度**：`@tepes_dracula` 正在寻找用于验证 RAG pipeline 查询的 **classifier**，参考了 **truthfulqa** 和 **fever** 等数据集。同时，`@kingpoki` 正在寻找在不诉诸模型 quantization 的情况下加速 **Windows 上的 Deci lm** 的方法。

- **OCR 和 Transformer 学习探索**：`@swetha98` 请求用于行分割的 OCR 工具，`@vikas.p` 推荐了 [Surya OCR](https://github.com/VikParuchuri/surya)。`.slartibart` 询问了关于解释 **code-writing transformers** 学习过程的资源，表现出对理解 Transformer 从代码中学到什么的兴趣。

- **Meta Reinforcement Learning 和 Mamba 学习小组见解**：`@davidebuoso` 正在寻找 **meta-RL 应用** 的合作者，以便在 Panda 机器人手臂上进行测试，参考了精选列表（[Meta-RL Panda](https://github.com/stars/lambdavi/lists/meta-rl-panda/)）。由 `@tonic_1` 领导的阅读小组深入研究了 `mamba library`，并讨论了 Mamba 与 Transformers 和 RWKV 模型相比的权衡和发展历史。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **Bard Ultra 发布：引发热烈推测**：社区幽默地讨论了关于 **Bard Ultra** 发布的推测，但也承认在这个领域做出准确预测的难度。
- **ChatGPT 访问和性能问题**：成员们报告了访问 **ChatGPT** 的挑战以及性能质量下降的情况，建议检查 **[OpenAI 的服务状态](http://status.openai.com/)**。担忧主要集中在 **GPT-4** 变慢以及更新后表现得像旧模型的问题。
- **AI 在珠宝设计和内容创作中的应用**：**Dall-E 3** 和 **Midjourney** 被强调为创建照片级真实感珠宝图像的工具，更广泛的讨论涉及 AI 生成内容的 **Terms of Service** 影响以及对 AGI 的向往，触及伦理和审查方面。
- **功能请求和 GPT API 问题**：用户呼吁增加“朗读”功能以缓解眼睛疲劳，并抱怨 **GPT** 模型在叙事生成中的 token 效率低下。还有报告称最近更新后出现间歇性 API 连接问题和性能下降，建议通过 [OpenAI 模型反馈表单](https://openai.com/form/chat-model-feedback) 报告此类发现。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **Google Scholar 搜索仍未解决**：`@aldikirito` 询问了关于在 **Google Scholar 中搜索期刊引用** 的合适 prompt，但对话并未产生明确的答复。

- **API 模型选择指南**：`@fanglin3064` 询问 API 中哪个模型类似于 GPT-3.5 Turbo，`@icelavaman` 指导在无联网访问时选择 **mixtral-8x7b-instruct**，在有联网访问时选择 **pplx-7(0)b-online**，并分享了 [基准测试链接](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)。

- **Perplexity API 尚无引用功能**：尽管社区对此感兴趣，`@me.lk` 确认目前 **没有计划** 在 Perplexity API 中包含来源引用，这与之前的预期相反。

- **禁止抓取**：关于 API 的使用，`@icelavaman` 提醒根据 [Terms of Service](https://blog.perplexity.ai/legal/terms-of-service)，禁止抓取结果，特别是那些利用 GPT 模型的结果。

- **Perplexity API 和 UI 的区别**：`@me.lk` 和 `@icelavaman` 澄清说 Perplexity AI 将 UI 和 API 作为独立产品提供，尽管两者属于同一公司旗下，但受不同的访问协议约束。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **Agentic 层提升 RAG 性能**：**LlamaIndex 团队**强调了在 **RAG 之上构建 Agentic 层**以促进实时用户反馈，从而改善复杂搜索的体验。关于此更新以及更多 Query Pipelines 的内容，可以在他们的 [Twitter 公告](https://twitter.com/llama_index/status/1755270274310901898)和 [Medium 文章](https://t.co/Funqm7Jw1u)中找到。
  
- **关于 RAG 未来的网络研讨会见解**：来自 **LlamaIndex** 的 `@seldo` 在与 `@ankit1khare` 的网络研讨会中讨论了 **RAG** 的细微差别以及 2024 年值得期待的功能，完整视频可在 [YouTube](https://t.co/7f12VvgImc) 上观看。

- **关于 LlamaIndex 工具和故障排除的深入交流**：社区成员剖析了使用 **NLSQLTableQueryEngine** 的问题，分享了处理 **Gemini** 连接问题以及在 **LlamaIndex** 中设置 rerankers 的见解。他们还讨论了数据库摄取管道（ingestion pipelines），`@ramihassanein` 提交了一个 GitHub PR，用于改进 **Deeplake** 的文档处理：[GitHub PR 链接](https://github.com/run-llama/llama_index/pull/10504)。

- **自定义 LLM 可能很棘手**：面对微调后的 **LLM** 返回 JSON 对象的情况，**LlamaIndex** 社区就创建自定义 LLM 还是对 JSON 输出进行后处理展开了辩论。

- **生产级 RAG 聊天机器人的难题**：`@turnerz` 就生产级 **RAG HR 聊天机器人**的**向量数据库、chunk 大小和 embedding 模型**寻求建议。讨论还深入探讨了演进 RAG 系统的战略方法——是先从简单的系统开始并迭代推进，还是直接从 multi-hop 检索系统开始。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **JupyterLab 并非总是必要**：一次讨论强调，如果不使用 **JupyterLab** 软件，可以忽略与其设置相关的消息，尽管一些成员表示更倾向于使用它。

- **RunPod 中持久卷（Persistent Volume）处理的增强**：针对 RunPod 上影响仓库完整性和持久卷存在的静默更新进行了故障排除。用户一致认为，更改挂载点以实现更好的磁盘空间管理并考虑缓存和输出是潜在的解决方案，并希望在 GitHub 上记录这些问题和解决方案。

- **用于持续预训练的新优化 Scheduler 已发布**：通过 GitHub 上的一个 pull request 引入了一个针对大型语言模型持续预训练优化的新 **scheduler**，旨在辅助此类模型的（重新）预热。可以通过 [此处](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273) 访问该 pull request 和更多详情。

- **AI 查询的 Prompt 格式和编码讨论**：关于 Python 中 AI prompts 最有效的数据结构正在进行讨论。Protobuf 和利用冷门 Unicode 字符的自定义编码方案被认为是 JSON 的潜在替代方案。

- **提升模型性能的训练策略和脚本共享**：讨论了学习率、batch 大小和配置参数，并特别提到了用于 **DPO 的 unsloth**、**paged_adamw_8bit** 以及 **linear scheduler**。名为 **DreamGenTrain** 的训练脚本已在 GitHub 上共享，可以在 [此处](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py) 找到。

- **为 RunPod 准备镜像需要特定条件**：为 RunPod 构建镜像应该在**带有 GPU 且安装了 bare metal docker 的机器**上进行，因为在 docker 内部运行 docker 是不可行的。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **LLaMa 挑战数字 "7"**：一位用户训练了一个拥有 2M 参数的 **LLaMa 模型**，用于条件生成 MNIST 数字 "7"，在 8GB MacBook M1 上仅用约 25 分钟便完成。这展示了训练小型模型的高效性。

- **Llava 的潜力未被充分利用**：根据一位 Twitter 用户分享的[链接](https://vxtwitter.com/billyuchenlin/status/1755207605537120513)，新版 **Llava 1.6** 被认为比其前代更先进。然而，在讨论的一个项目中并未采用它，而它本可以带来显著的改进。

- **GOODY-2 LLM 中的 AI 伦理**：**GOODY-2** 正式发布，这是一个强调伦理参与并避免争议性问题的模型。在其 [Model Card](https://www.goody2.ai/goody2-modelcard.pdf) 中有详细说明，这一负责任对齐的 AI 模型的发布引起了各界不同的反应。

- **DALL-E 3 引入水印**：**OpenAI 的 DALL-E 3** 输出结果中已引入水印，在图像元数据中嵌入了不可见和可见的组件，正如用户分享的一篇 [The Verge 文章](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials) 中所讨论的那样。

- **顶尖背景移除工具问世**：尖端的**背景移除工具** BRIA Background Removal v1.4 现已能与领先模型媲美，且专为商业环境中的内容安全而设计。可以通过 [Demo 和 Model Card](https://huggingface.co/briaai/RMBG-1.4) 了解其功能。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 摘要

- **CUDA 爱好者寻求硬件**：社区中掀起了组装深度学习主机的热潮，大家对购入二手 **3090 GPU** 以及研究适合多 GPU 配置的硬件套装表现出浓厚兴趣，例如在[此套装](https://a.co/d/iR3bvvF)中提供的 **Intel Core i9-12900K** 和 **ASUS ROG Maximus Z690 Hero DDR5** 主板。讨论还集中在多 GPU 配置的可行性上，强调了 **PCIe 分叉卡（bifurcation cards）** 和高质量 **Gen4 PCIe 线缆** 的作用。

- **PyTorch 2 在 ASPLOS 2024 铺平道路**：令人振奋的是，包含 **TorchDynamo**、**TorchInductor** 和 **Dynamic Shape 支持** 的 PyTorch 2 论文已被 ASPLOS 2024 接收。随着 [PyTorch 2 论文和教程](https://pytorch.org/blog/pytorch-2-paper-tutorial/) 成为热门话题，开发者们非常关注其基于 Python 的编译器带来的易用性、针对消费级硬件的优化，以及 **TORCH_LOGS** 等调试工具的进步。此外，在模型移植方面，`torch.compile` 被推崇为优于 `torch.jit.trace` 的方案。

- **JAX 跃居前列**：JAX 相较于 TensorFlow 2 的流行归功于其 **XLA JIT** 优化以及在 Google Cloud TPU 上更好的体验，正如一段支持性的[视频](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P)中所讨论的那样。该平台在 NVIDIA GPU 上似乎也能与 TensorFlow 和 Torch 正面竞争。JAX 的历史被指出可能源于对 TensorFlow 复杂性的回应，特别是针对 TensorFlow 1.x 中的全局变量问题。

- **对社区与知识的追求**：人们对联系当地工程师有着浓厚的兴趣，这体现在对旧金山**线下 ML 聚会**的咨询，以及与 **GTC** 等活动可能存在的重叠。此外，成员们正在寻求参考书的解决方案，并通过 Discord 链接提供了相关指引。

- **飞跃式学习**：新人正在深入研究 AI 技术，例如 `einstein5744` 从 Stable Diffusion fast ai 课程开始学习，其他人则在请求社区驱动的知识交流机会。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain ChatPromptTemplate Bug 报告**：用户 `@ebinbenben` 在使用 LangChain 的 `ChatPromptTemplate` 时遇到了 `ValueError`，但该问题在讨论中尚未得到解决。
- **LangChain 框架文档增强**：LangChain 文档现在新增了关于使用事件进行自定义流式传输（custom streaming with events）以及在 LLM 应用中进行流式传输的章节，重点介绍了 [更新后的文档](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events) 中详述的 `where_cat_is_hiding` 和 `get_items` 等工具的使用。
- **LangChain Expression Language (LCEL) 详解**：`@kartheekyakkala` 介绍了一篇[博客文章](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8)，详细阐述了 LangChain 框架中 LCEL 的声明式语法，用于使 LLM 具备上下文感知能力。
- **本地 LLM 资源和聊天机器人工具包更新**：由 `@discossi` 分享的 LLM 资源和工具，包括一个轻量级聊天机器人和针对初学者的基础材料，可以在 [llama-cpp-chat-memory](https://github.com/ossirytk/llama-cpp-chat-memory) 和 [llm_resources](https://github.com/ossirytk/llm_resources) 找到。
- **作者在 API 变更期间支持读者**：针对由于 API 更新导致的弃用代码示例问题，`@mehulgupta7991` 提供了帮助，并承诺在下一版书中进行更新。读者可以通过 datasciencepocket@gmail.com 联系作者获取直接支持。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **LLM 在主题选择上陷入两难**：用户讨论了 ChatGPT 主题选项的缺点，认为其过于极端且缺乏中间地带，但未提出具体的解决方案。
- **微调对话探讨**：关于微调策略的咨询，重点在于应该对数据集中的每条消息进行二元结果评分，还是对整个对话进行整体评估。
- **AI 驱动的报告服务遭遇成本障碍**：用户 `@res6969` 开发了一项利用 AI 从报告章节创建可搜索数据库的服务，但面临约 20,000 美元的意外运营成本，并配以带有幽默感的表情符号 <:lmfao:556127422823530507>。
- **库性能声明遭到质疑**：讨论中对一个声称性能提升 20 倍的新库表示怀疑，担忧其成本效益和实际效用，特别是对于那些自行托管 AI 模型的人来说。
- **深思 GPT-4 的昂贵开销**：对话强调了使用 GPT-4 的巨额费用，成本可能达到每月 3 万美元，并对新方法是否真的能省钱表示怀疑。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **OpenAI 定价旨在统治市场**：`@dbreunig` 强调了 OpenAI 的 `text-embedding-3-small` 的成本效率，评论其价格低廉到在某些功能上竞争对手显得毫无必要。
- **OpenAI 定价中疑似包含竞争策略**：`@simonw` 推测 OpenAI 的定价可能是为了抑制竞争而进行的战略设计，得到了 `@dbreunig` 的认同。
- **在 OpenAI 主导的市场中竞争**：`@dbreunig` 和 `@simonw` 讨论了竞争对手试图构建平台与 OpenAI 竞争的挑战，认为除了定价、UI 和 UX 之外，创新对于在该领域实现差异化至关重要。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **LoRA 需要更多内存**：`devnull0` 强调在优化模型时，需要充足的内存来同时容纳模型和 **LoRA** (Locally Rewarded Attention) 参数。
- **在有限资源下压缩模型**：根据 `johannhartmann` 的说法，**Llama_factory** 对 *unsloth* 的新支持允许 **Mistral** 与 *qlora* 配合使用，这使得 wiedervereinigung-7b 模型能够适应某些内存限制。
- **Mistral 迈向多语言和多数据集**：`johannhartmann` 还指出 Llama_factory 扩展了 **Mistral** 的功能，包括支持 **9 个德语 SFT** 和一个 **DPO 数据集**。
- **新基准测试出现**：_jp1_ 发布了一个 [arXiv 论文](https://arxiv.org/abs/2402.01781)链接，暗示可能有一个社区感兴趣的新基准测试或研究。

---

# 第 2 部分：各频道详细总结与链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1204703108463329321) (1295 条消息 🔥🔥🔥):

- **Steam Deck 作为出人意料的 AI 猛兽**：用户 `@skorchekd` 分享了他们在 Steam Deck 上运行模型的经验，指出其性能较 PC 有显著提升，并将其归功于该设备的 RAM 速度和集成 APU。他们讨论了即使在 RAM 限制下也能高效运行 Solar 10.7B 等大型模型的潜力，甚至认为该设备作为一款兼具 AI 处理能力的便携式游戏 PC 具有极高价值。
  
- **关于 AI 与数学本质的辩论**：用户如 `@phantine`、`@selea` 和 `@lee0099` 就 AI 的本质展开了哲学讨论，探讨 AI 是否能像人类一样“理解”，以及数学概念是否独立于物理现实而存在。对话涉及了必然观念（necessary ideas）、形而上学以及意识的实质等概念。

- **LLM 性能讨论持续进行**：`@starsupernova` 提到正在为 OSS Unsloth 添加多 GPU 支持。目前 Unsloth 的推理速度更快，但建议仅在验证和 Gradio 类推理中替代 vLLM。他们还指出了预量化的优势，包括无精度损失和减少 1GB VRAM，并表示在阅读相关论文后有兴趣将 LASER 应用于 Unsloth。
  
- **训练数据中的 OpenAI 模型污染**：`@itsme9316` 指出，大多数当代模型可能都包含由 OpenAI 模型生成或来自互联网的训练数据。这包括合成数据污染，导致模型的回答带有 OpenAI 的风格。

- **关于 LLM 与计算的技术讨论**：包括 `@selea`、`@technotech` 和 `@starsupernova` 在内的多位用户讨论了 Bert 等 LLM 架构、P100 相比 T4 GPU 的潜在改进、优化计算图的方法以及内核优化。此外，还提到了一篇 Mamba 论文，显示出与 Transformers 相比具有前景的结果。

**提及的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1142735399555432529/1204930925125701642)：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1130801664383787071)：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区聊天、聚会并保持紧密联系。
- [Sleeper Agents: 训练能够持续通过安全训练的欺骗性 LLMs](https://arxiv.org/abs/2401.05566)：人类能够表现出策略性的欺骗行为：在大多数情况下表现得乐于助人，但在有机会追求其他目标时，其行为会变得截然不同。...
- [未找到标题](https://e2eml.school/transformers.html#markov_chain)：未找到描述
- [Azure OpenAI 的 Prompt Engineering 技术 - Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering)：了解如何对 GPT-3、GPT-35-Turbo 和 GPT-4 模型使用 Prompt Engineering 的选项
- [serpdotai/sparsetral-16x7B-v2 · Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2)：未找到描述
- [Giga Gigacat GIF - Giga Gigacat Cat - 发现并分享 GIF](https://tenor.com/view/giga-gigacat-cat-mewing-mogging-gif-12429734670640119345)：点击查看 GIF
- [Ascending Energy GIF - Ascending Energy Galaxy - 发现并分享 GIF](https://tenor.com/view/ascending-energy-galaxy-gif-17739196)：点击查看 GIF
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1#evaluation-results)：未找到描述
- [论文页面 - Mamba 能学会如何学习吗？关于 In-Context Learning 任务的比较研究](https://huggingface.co/papers/2402.04248)：未找到描述
- [NeuralNovel/Tiger-7B-v0.1 · Hugging Face](https://huggingface.co/NeuralNovel/Tiger-7B-v0.1)：未找到描述
- [Shooting GIF - Cowboy Gun Shooting - 发现并分享 GIF](https://tenor.com/view/cowboy-gun-shooting-smoke-gif-5591465)：点击查看 GIF
- [虚数是真实的 [第一部分：介绍]](https://www.youtube.com/watch?v=T647CGsuOVU&t=2s)：想要提前观看新视频及获得其他福利：https://www.patreon.com/welchlabs。想了解更多或教授本系列课程吗？查看 Imaginary Numbers are...
- [Clapping Hamood GIF - Clapping Hamood Mood - 发现并分享 GIF](https://tenor.com/view/clapping-hamood-mood-hi-baby-happy-gif-13463157)：点击查看 GIF
- [LLM 过程详解。它们的运行机制以及底层工作原理！](https://youtu.be/_Pt-rGE4zEE?si=s6orG0bKWAkX_vPY&t=326)：在这份综合指南中探索 LLM 的迷人世界。我们将从 Softmax、Layer... 等关键概念奠定基础。
- [GitHub - cognitivecomputations/laserRMT: 这是我们对 'Layer Selective Rank Reduction' 的自主实现](https://github.com/cognitivecomputations/laserRMT)：这是我们对 'Layer Selective Rank Reduction' 的自主实现 - GitHub - cognitivecomputations/laserRMT: 这是我们对 'Layer Selective Rank Reduction' 的自主实现
- [来自 Alexandre TL (@AlexandreTL2) 的推文](https://x.com/alexandretl2/status/1754927881178791962?s=61&t=tHcPPlKi_G7OoyasQK8oDQ)：Mamba 在 OthelloGPT 实验中表现如何？让我们将其与 Transformer 进行对比 👇🧵
- [AMD Ryzen 7 8700G APU 评测：性能、散热与功耗分析 - Hardware Busters](https://hwbusters.com/cpu/amd-ryzen-7-8700g-apu-review-performance-thermals-power-analysis/)：Hardware Busters - AMD Ryzen 7 8700G APU 评测：性能、散热与功耗分析 - CPU
- [Reddit - 深入探索一切](https://www.reddit.com/r/singularity/s/bZI9ELVwhD)：未找到描述

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1204699495271497748) (342 messages🔥🔥): 

- **Quantization 与财务约束**：`@dreamgen` 谈到了免费模型的可持续性，预测一旦风险投资基金枯竭，未来将面临挑战。同时，`@mrdragonfox` 强调了与简单且低成本的 Model Merges 相比，创建数据集所带来的财务负担。

- **模型合并与数据困境**：表达了对模型合并的担忧，特别是 `@soufflespethuman` 和 `@mrdragonfox`，他们担心这会削弱创建原始数据集的价值以及真正创新的动力。他们讨论了强制执行“禁止合并”许可证以保护其作品的可能性。

- **AI 创新融资的幻想**：围绕 `@billynotreally` 提出的由“加密货币/股票大佬”资助 AI 实验的假设场景展开了讨论。`@mrdragonfox` 反驳了捐赠者的慷慨，认为有钱人通常想要回报，例如认可或结果。

- **关于 Augmentoolkit 重组的技术讨论**：`@mrdragonfox` 正在重构 Augmentoolkit 的架构，通过 [GitHub](https://github.com/e-p-armstrong/augmentoolkit) 分享更新和代码。讨论内容包括利用 Python 中的异步 IO 以及从 Jupyter notebooks 向更结构化的代码库过渡。

- **MiquMaid v2 的讨论与开发**：进行了关于 MiquMaid v2 的更新和讨论，`@undi` 分享了进度并指出其已公开，而 `@netrve` 则很喜欢该模型的表现，尽管存在一些重复问题并需要调整生成设置。后者的讨论深入探讨了处理内容生成时重复问题的策略。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1204890898878697503)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，与你的朋友和社区保持紧密联系。
- [MiquMaid - a NeverSleep Collection](https://huggingface.co/collections/NeverSleep/miqumaid-65c3d5e0fd15420346adc906)：未找到描述
- [TheBloke/Yarn-Llama-2-70B-32k-GGUF · Hugging Face](https://huggingface.co/TheBloke/Yarn-Llama-2-70B-32k-GGUF)：未找到描述
- [augmentoolkit/config.yaml at api-branch · e-p-armstrong/augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/blob/api-branch/config.yaml)：将算力和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit
- [augmentoolkit/main.py at api-branch · e-p-armstrong/augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/blob/api-branch/main.py)：将算力和书籍转换为 Instruct-Tuning 数据集 - e-p-armstrong/augmentoolkit
- [Ness End This Suffering Mother2earthbound Super Smash Bros GIF - Ness End This Suffering Mother2Earthbound Super Smash Bros - Discover &amp; Share GIFs](https://tenor.com/view/ness-end-this-suffering-mother2earthbound-super-smash-bros-gif-25870136)：点击查看 GIF
- [Baroque Rich GIF - Baroque Rich - Discover &amp; Share GIFs](https://tenor.com/view/baroque-rich-gif-22998652)：点击查看 GIF

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1204820787962454167) (5 messages): 

- **寻求 Fine-Tuning 指导**：用户 `@immortalrobot` 表达了学习 Fine-Tuning 流程的愿望，并请求推荐提供逐步指导的教程文章。
- **语言模型对齐的新方法**：`@maldevide` 分享了一篇 [arxiv 论文](https://arxiv.org/html/2402.01878v1)，介绍了一种名为 **Listwise Preference Optimization (LiPO)** 的新框架，用于对齐语言模型，该框架优化的是列表式响应而非单个响应。
- **模型优化的下一步**：在评论 PairRM DPO 模型的有效性时，`@maldevide` 认为 LiPO 是语言模型对齐技术的逻辑演进。
- **使用 LiPO Rankers 进行本地训练的可行性**：`@maldevide` 指出了 LiPO 的一个优势，强调该框架中使用的 Rankers 参数量低于 10 亿，从而能够进行本地训练，这是一个实际的益处。
- **模型加载咨询**：`@yinma_08121` 询问是否有人有使用 Candle 加载 phi-2-gguf 模型的经验，寻求社区对此事的意见。

**提到的链接**：

[LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/html/2402.01878v1)：未找到描述

  

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1204741981063094282) (24 messages🔥): 

- **Hugging Face 模型实现困扰指南**：`@wbsch` 向 `@logicloops` 指出，实现来自 Hugging Face 的模型时遇到的问题可能是由于不正确的 Prompt 和异常的 Stop Tokens 导致的，并建议重新检查模型的 Model Card，或考虑使用更简单的替代方案如 koboldcpp。

- **Mojo 性能超越 Rust**：`@dirtytigerx` 分享了一篇文章，介绍 Mojo 语言如何取得显著的性能优势，基准测试显示其表现优于 Python 甚至 Rust，引发了关于其对各种工具影响的讨论。

- **Mojo 在高性能环境中的潜力**：`@falconsfly` 对 Mojo 的设计和实现策略表示热衷，并通过矩阵乘法优化的例子展示了其能力，以及它如何与 duckdb 等工具良好配合。

- **Aletheion 探索集成 Agent 函数**：`@aletheion` 正在致力于将自定义函数集成到 Bot 流程中，使其能够根据需要调用数据库查询，旨在让 Bot 利用自身的“记忆”和“笔记”来提供增强的交互，而不依赖外部逻辑触发器。

- **模型实现中的误解**：`@lushboi` 承认对较小 LLaMa 模型的能力存在疏忽，`@falconsfly` 对此进行了澄清，指出只有 70B LLaMa 模型使用了 GQA，强调了参考官方文档的重要性。

**Links mentioned**:

- [Modular: Community Spotlight: Outperforming Rust ⚙️ DNA sequence parsing benchmarks by 50% with Mojo 🔥](https://www.modular.com/blog/outperforming-rust-benchmarks-with-mojo): 我们正在为世界构建下一代 AI 开发者平台。阅读我们关于社区亮点：Mojo 在 DNA 序列解析基准测试中性能超越 Rust 50% 的最新文章。
- [Modular Docs - Matrix multiplication in Mojo](https://docs.modular.com/mojo/notebooks/Matmul.html): 了解如何利用 Mojo 的各种函数来编写高性能的 matmul。

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1204930215852114041) (2 messages): 

- **介绍 OpenHermes 数据集查看器**：`@carsonpoole` 为 **OpenHermes** 开发了一个数据集查看器，允许用户使用 `j` 和 `k` 键滚动浏览示例，并查看关于 Token 数量和类型的分析。该工具还具备按样本数量或 Token 数量进行排序的过滤功能。
  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1204760064846598177) (43 条消息🔥): 

- **引入带有 SAFE 的 Subformer**：`@euclaise` 分享了一篇 [arXiv 论文](https://arxiv.org/abs/2101.00234)，重点探讨了 Transformer 中的参数共享方法，以解决其计算和参数预算效率低下的问题。该论文介绍了 **Subformer**，它利用三明治式参数共享和自注意力嵌入分解 (SAFE)，在参数更少的情况下性能优于 Transformer 模型。
  
- **BiLLM 引入 1-bit 量化**：`@gabriel_syme` 发布了一篇关于 **BiLLM** ([下载 PDF](/pdf/2402.04291.pdf)) 的论文链接，这是一种针对大语言模型 (LLM) 的 1-bit 后训练量化方案，旨在显著降低计算和内存需求，同时保持性能。

- **NeoEvalPlusN 基准测试上的模型评分**：`@nonameusr` 链接到了 **Gembo-v1-70b** 模型的 Hugging Face 页面，并提醒该模型包含敏感内容，可能含有潜在有害信息。尽管没有完整的 Model Card，但他们指出该模型在 NeoEvalPlusN 基准测试中得分很高，目前正在等待 openllm 的结果。

- **关注环境的模型性能**：`@teknium` 提到了在论坛上分享模型时上下文的重要性，并建议包含说明文字或 Model Card，因为许多帖子都缺乏这些细节。

- **自奖励语言模型的交叉对比**：在关于近期进展的讨论中，`@atgctg` 链接了一篇 [arXiv 论文](https://arxiv.org/abs/2402.04792)，将 OAIF 与同时期的“自奖励”语言模型研究进行了对比。论文强调了 OAIF 如何利用来自任何 LLM 的反馈，包括那些比被对齐模型更强大的模型。

**提到的链接**：

- [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291)：预训练大语言模型 (LLMs) 展现了卓越的通用语言处理能力，但对内存和计算资源有巨大需求。作为一种强大的压缩手段...
- [InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models](https://arxiv.org/abs/2310.19531)：生成式语言模型通常通过在大型文本语料库上预测给定前文的下一个 Token（即子词/词/短语）来进行预训练。最近的研究证明了其令人印象深刻的...
- [ChuckMcSneed/Gembo-v1-70b · Hugging Face](https://huggingface.co/ChuckMcSneed/Gembo-v1-70b)：未找到描述
- [ibivibiv/giant-hydra-moe-240b · Hugging Face](https://huggingface.co/ibivibiv/giant-hydra-moe-240b)：未找到描述
- [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792)：偏好直接对齐 (DAP) 方法（如 DPO）最近已成为人类反馈强化学习 (RLHF) 的高效替代方案，它们不需要单独的奖励模型...
- [Cute Hide GIF - Cute Hide Cat - Discover &amp; Share GIFs](https://tenor.com/view/cute-hide-cat-scared-shy-gif-16121120)：点击查看 GIF
- [ChuckMcSneed/NeoEvalPlusN_benchmark · Datasets at Hugging Face](https://huggingface.co/datasets/ChuckMcSneed/NeoEvalPlusN_benchmark)：未找到描述
- [Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers](https://arxiv.org/abs/2101.00234)：与之前的序列处理架构（如 RNNs）相比，Transformers 展示了更高的性能。尽管性能提升显著，但正如最近所指出的，该模型...
- [InRank: Incremental Low-Rank Learning](https://arxiv.org/abs/2306.11250)：贪婪低秩学习 (GLRL) 理论旨在解释深度学习令人印象深刻的泛化能力。它证明了基于随机梯度的训练会隐式地对神经...

  

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1204707990645506048) (221 条消息🔥🔥): 

- **寻求 Wojak AI 链接**：`@theluckynick` 请求了 Wojak AI 的链接，但未得到回应。
- **关于 OpenHermes-2.5-Mistral-7B 的训练配置查询**：`@givan_002` 询问了 AI 模型的 Fine-tuning 参数，并被引导至一个已关闭的讨论，但关于训练配置并未得到满意的答复。
- **AI 模型基准测试**：`@if_a` 展示了多个模型的基准测试结果，指出 Senku-70B 在特定任务中表现优于其他模型，讨论表明不同 LLM 之间的 System prompts 可能会影响结果。
- **考虑在 Nous-Hermes 2 数据集上进行 Fine-tuning**：`@if_a` 正在考虑使用 Nous-Hermes 2 数据集对 miqu 模型进行 Fine-tuning，并讨论了潜在的时间周期和挑战。
- **GPU 工作负载调度讨论**：多位用户（包括 `@chrisj7746`、`@Sebastian` 和 `@leontello`）讨论了在 GPU 上调度作业的高效方法，即使是对于较小的集群，Slurm 也是一个广受欢迎的建议。
- **Quantization 与模型架构讨论**：用户讨论了 AI 模型的 Quantization 算法，包括 `@vatsadev` 提出的 Anime 基准测试的潜力。同时，`@nonameusr` 和 `@n8programs` 讨论了 GPT-4 之后模型架构变化的意义，包括 Transformer 的 Turing complete（图灵完备）改造。

**提到的链接**：

- [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291)：预训练的大语言模型（LLMs）展现了卓越的通用语言处理能力，但对内存和计算资源有巨大需求。作为一种强大的压缩技术...
- [teknium/OpenHermes-2.5-Mistral-7B · Can the training procedure be shared?](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/discussions/9)：未找到描述
- [Orange Cat Staring GIF - Orange cat staring Orange cat Staring - Discover &amp; Share GIFs](https://tenor.com/view/orange-cat-staring-orange-cat-staring-cat-gif-13724146065807985297)：点击查看 GIF
- [Turing Complete Transformers: Two Transformers Are More Powerful...](https://openreview.net/forum?id=MGWsPGogLH)：本文介绍了 Find+Replace Transformers，这是一系列多 Transformer 架构，可以证明完成单个 Transformer 无法完成的任务，并在多项挑战性任务上优于 GPT-4...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1al58xw/yet_another_state_of_the_art_in_llm_quantization/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPT/comments/1ahhlon/i_downloaded_my_chatgpt_user_data_and_found_the/)：未找到描述
- [VatsaDev/animebench-alpha · Datasets at Hugging Face](https://huggingface.co/datasets/VatsaDev/animebench-alpha)：未找到描述
- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5)：未找到描述
- [cmp-nct/llava-1.6-gguf at main](https://huggingface.co/cmp-nct/llava-1.6-gguf/tree/main)：未找到描述
- [Llava 1.6 - wip by cmp-nct · Pull Request #5267 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5267)：初步进展 - 尽管存在许多未解决的问题，但我已经在 license_demo 示例上通过 llava-1.6-13B 获得了令人印象深刻的结果。待办事项：缺失的最大且最重要的区别是 &quot;spatial_un...

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1204700271884636192) (49 条消息🔥): 

- **自定义预训练模型在提取任务上表现不佳**：`@fedyanin` 提到他们的自定义预训练模型在生成任务上表现良好，但在提取任务上表现不佳。他们询问了除了针对该任务进行显式微调之外，还有哪些提高提取性能的方法。
  
- **微调的 VRAM 是否充足**：`@natefyi_30842` 询问 8GB VRAM 是否足以在 Axolotl 上对 Mistral 等 7b 模型进行 SFT 或 DPO。`@fedyanin` 建议使用 *qlora* 和较小的上下文可能刚好够用，而 `@teknium` 提到 mlx 可能可行，但目前尚未完全成熟。

- **在配置中设置特殊 Token**：`@paragonicalism` 寻求关于在 `.yml` 文件中设置特殊 Token 的建议，以便使用 axolotl 在 OpenHermes2.5 上微调 `phi-2`。`@teknium` 提供了一个代码片段来定义 `eos_token` 和其他 Token，随后建议添加 `pad_token: `。

**提到的链接**：

- [phi2-finetune/nb_qlora.ipynb at main · geronimi73/phi2-finetune](https://github.com/geronimi73/phi2-finetune/blob/main/nb_qlora.ipynb)：通过在 GitHub 上创建账户，为 geronimi73/phi2-finetune 的开发做出贡献。
- [LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)：[NeurIPS'23 Oral] 视觉指令微调 (LLaVA)，旨在实现 GPT-4V 级别的能力及更高水平。- haotian-liu/LLaVA
- [liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)：未找到描述
- [GitHub - bkitano/llama-from-scratch: Llama from scratch, or How to implement a paper without crying](https://github.com/bkitano/llama-from-scratch)：从零开始实现 Llama，或者如何不流泪地实现论文 - GitHub - bkitano/llama-from-scratch
- [LLaVA/scripts/v1_5/finetune_task_lora.sh at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh)：[NeurIPS'23 Oral] 视觉指令微调 (LLaVA)，旨在实现 GPT-4V 级别的能力及更高水平。- haotian-liu/LLaVA
- [LLaVA/scripts/v1_5/finetune_task.sh at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task.sh)：[NeurIPS'23 Oral] 视觉指令微调 (LLaVA)，旨在实现 GPT-4V 级别的能力及更高水平。- haotian-liu/LLaVA
- [Obsidian/scripts/finetune_qlora.sh at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/scripts/finetune_qlora.sh)：也许是新的 state of the art 视觉模型？我们拭目以待 🤷‍♂️ - NousResearch/Obsidian
- [Obsidian/scripts/v1_5/finetune.sh at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/scripts/v1_5/finetune.sh)：也许是新的 state of the art 视觉模型？我们拭目以待 🤷‍♂️ - NousResearch/Obsidian

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1204721503111749662) (138 条消息🔥🔥): 

- **LM Studio 中的 LaTeX 限制**：用户 `.neurorotic.` 对使用 DeepSeek Math RL 7B LLM 时 LaTeX 格式的数学公式输出表示沮丧；LM Studio 似乎无法正确渲染 LaTeX，其他用户也表达了类似的担忧，并注意到各种模型的 LaTeX 输出有所增加。

- **LLM 的 GPU 与 RAM 和 CPU 之争**：`@pierrunoyt` 询问为什么 LLM 使用 GPU 而不是 RAM 和 CPU，`@justmarky` 解释说 GPU 上的处理速度要快得多。这引发了关于大语言模型计算偏好的简短讨论。

- **模型选择与优化讨论**：几位用户讨论了各种模型及其与不同系统的兼容性。`@kristus.eth` 提到 Ollama 与 LM Studio 相比缺少一些功能，而 `@akiratoya13` 遇到了尽管设置显示已开启但 GPU offload 未被利用的问题。

- **LM Studio 的网络与下载问题**：用户 `@yorace` 和 `@tkrabec` 讨论了 LM Studio 内部的网络错误和模型下载问题，`@heyitsyorkie` 建议问题可能源于针对特定国家的 Huggingface 封锁或 VPN 问题。

- **本地 LLM 的适应性与持久性**：用户 `@joelthebuilder` 和 `@fabguy` 就本地大语言模型是否可以通过用户交互随时间学习或适应进行了对话。目前的观点是，在普通硬件上进行 fine-tuning 通常是不可行的，而加入相关的 system prompts 通常足以定制回答。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1110598183144399058/1204973625518587925.)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [OLMo - AI2 开发的开源语言模型](https://allenai.org/olmo)：OLMo 是一系列旨在推动语言模型科学发展的开源语言模型。OLMo 模型是在 Dolma 数据集上训练的。
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/)：未找到描述
- [在本地使用 LM Studio 运行 big-AGI [教程]](https://youtu.be/MqXzxVokMDk)：➤ Twitter - https://twitter.com/techfrenaj ➤ Twitch - https://www.twitch.tv/techfren ➤ Discord - https://discord.com/invite/z5VVSGssCw ➤ TikTok - https://www....
- [[1小时演讲] 大语言模型简介](https://www.youtube.com/watch?v=zjkBMFhNj_g)：这是一场面向普通大众的 1 小时大语言模型入门讲座：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1204751117733601310) (45 条消息🔥): 

- **对 Qwen 1.5 的褒贬不一**：`@pierrunoyt` 分享了一个关于 **Qwen 1.5**（一个开源语言模型）的 [YouTube 视频](https://www.youtube.com/watch?v=RjCYLIUfxrE&t=1s)，但 `@heyitsyorkie` 批评该内容在模型预览方面具有误导性。`@evi_rew` 和 `@yagilb` 进一步讨论了 Qwen 1.5 在 context lengths 和 GPU 加速方面的技术问题。
- **Code Llama 批评**：`@pierrunoyt` 对 Code Llama 2 表示不满，表示需要一个更优秀的、同时理解设计的编程语言学习模型。
- **OpenAI 模型在代码生成方面的不一致性**：`@lord_half_mercy` 观察到使用 OpenAI 的 ChatGPT 生成代码时回复质量有所下降，并质疑这是否与 context length 或复杂度有关；`@heyitsyorkie` 幽默地建议这是因为 AI 变“懒”了。
- **对 Vision 模型的困惑**：`@bob_dale` 询问是否有模型能够根据某种风格生成新的 logo，`@heyitsyorkie` 推荐了 GPT4 Vision，尽管最初对该模型的功能存在误解。
- **关于模型有效性的辩论**：用户讨论了包括 Qwen 1.5、Miqu 和 LLaMA 在内的各种模型的有效性，评论涵盖了从高 VRAM 系统上的内存问题 (`@.bambalejo`) 到对语言能力和模型输出的批评 (`@pwrreset` 和 `@re__x`)。


**提到的链接**：

- [@JustinLin610 在 Hugging Face 上表示：&quot;昨天我们刚刚发布了 Qwen1.5。也许有一天我可以更多地谈谈……&quot;](https://huggingface.co/posts/JustinLin610/764363519759697)：未找到描述
- [Qwen 1.5：最强大的开源 LLM - 0.5B, 1.8B, 4B, 7B, 14B, 和 72B - 击败了 GPT-4？](https://www.youtube.com/watch?v=RjCYLIUfxrE&t=1s)：在本视频中，我们深入探讨了 Qwen 系列的最新迭代版本 Qwen 1.5。Qwen 1.5 在农历新年前夕发布，带来了显著的提升...

  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1204763761383178311) (98 条消息🔥🔥): 

<ul>
  <li><strong>最佳 LLM 硬件规格咨询</strong>：`@jolionvt` 询问了在 AMD Ryzen 5 7600、AMD Radeon RX 6700 XT 和 32 GB RAM 环境下本地运行 LLM 的理想设置，但未收到直接回复。</li>
  <li><strong>PCIe Riser 性能担忧</strong>：`@nink1` 质疑了使用 PCIe 1x 转 16x 延长线（riser cables）与直接连接主板相比的性能损耗。`@quickdive.` 表示有兴趣测试这一问题，并建议通过修改 BIOS 通道宽度（lane width）进行对比，而 `@nink1` 计划进一步调查。</li>
  <li><strong>探索用于 DIY 语音项目的 ESP32</strong>：`@joelthebuilder` 寻求有关可为家庭网络提供语音输入和输出的硬件 DIY 项目建议，目标是用自定义设置取代 Alexa。在其他人的鼓励下，`@joelthebuilder` 分享了一个关于使用 ESP32 S3 Box 3 集成 Home Assistant 和本地 LLM AI 的 [YouTube 视频](https://www.youtube.com/watch?v=_qft28MiVnc)，并思考了其购买渠道。</li>
  <li><strong>评估电源线改装风险</strong>：`@nink1` 分享了一个通过改装 PSU CPU 电源线为外部 Riser 板供电的创意方案，`@.ben.com` 警告这可能存在火灾隐患。`@nink1` 则辩称，考虑到低功耗和妥善的线缆固定，他的设置是安全的。</li>
  <li><strong>多 GPU 设置中的 GPU 带宽和 PCIe 通道问题</strong>：关于在多 GPU 配置中使用延长线或限制通道宽度对 PCIe 带宽的影响引发了讨论。`@nink1`、`@quickdive.`、`@rugg0064` 和 `@savethehuman5` 讨论了潜在的性能瓶颈，并分享了各自实验多 GPU 设置的计划。</li>
</ul>

**提到的链接**：

- [Tesla P40 Radial fan shroud by neophrema](https://www.thingiverse.com/thing:6031884/makes)：嘿，这是为 Nvidia Tesla 显卡设计的风扇导流罩，其结构与 P40 相同。在尝试使用普通的 Noctua 风扇失败后（该死，我投入了不少钱……），我意识到空气压力……
- [All About AI](https://www.youtube.com/@AllAboutAI)：欢迎来到我的频道 All About AI =) Discord: https://discord.gg/Xx99sPUeTd 网站: https://www.allabtai.com 你可以如何开始使用生成式 AI 来帮助你完成创意或其他日常任务……
- [ESP32 S3 Box 3  Willow - HA - Local LLM AI](https://www.youtube.com/watch?v=_qft28MiVnc)：ESP32 S3 Box 3 运行 Willow 并连接到集成本地 LLM AI (Mistral 7b) 的 Home Assistant
- [NVIDIA Tesla P40 24GB GDDR5 Graphics Card and Cooling Turbine Fan  | eBay](https://www.ebay.ca/itm/225917793841)：未找到描述
- [Amazon.com: Raspiaudio ESPMUSE Proto esp32 Development Card with Speaker and Microphone](https://www.amazon.com/RASPIAUDIO-ESPMUSE-Development-Speaker-Microphone/dp/B09N3S9S29?crid=17US408DI26TI&keywords=esp+muse+luxe&qid=1675950795&sprefix=esp+muse+luxe,aps,173&sr=8-1&linkCode=sl1&tag=peyanski-20&linkId=363d84ed2b123aa08b6e293d433c3507&language=en_US&ref_=as_li_ss_tl)：未找到描述
- [no title found](https://www.aliexpress.com/item/1005005600238754.html)：未找到描述

  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1204744733931606036) (10 messages🔥): 

- **Debian 系统下使用 LM Studio 的困扰**：`@transfluxus` 报告了在 Debian 上执行 **LM_Studio-0.2.14-beta-1.AppImage** 时的问题，出现了 `cannot execute binary file` 错误。`@heyitsyorkie` 建议获取 Linux beta 角色并检查 <#1138544400771846174> 中的置顶消息，并使用 `chmod` 命令使应用具有可执行权限。

- **训练与图像生成的特性愿望清单**：`@junkboi76` 表示希望在未来版本中看到 **support for training**（支持训练）和 **image generation support**（图像生成支持）。他们承认其复杂性，但认为这些将是受欢迎的功能。

- **增强建议请前往反馈站**：`@fabguy` 引导 `@junkboi76` 在 <#1128339362015346749> 中发起讨论，或对有关训练和图像生成的现有特性请求进行投票。

- **对 JSON 预设中 `pre_prompt` 的困惑**：`@wolfspyre` 询问预设 JSON 中的 `pre_prompt` 是否就是 **system prompt**，以及它是否应该反映在 "prompt format" 预览中。该问题随后被作为一个潜在的 Bug 提交。

- **设计选择还是 Bug？**：针对 `@wolfspyre` 关于 `pre_prompt` 未在设置弹窗中显示的担忧，`@fabguy` 指出 "User" 和 "Assistant" 的内容也没有显示，这可能是由于系统提示词可能非常长而有意为之的设计。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204848289866850334/1204848289866850334)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1205009689214193745) (3 messages): 

- **Autogen 实验开始**：`@photo_mp` 开始尝试使用 **autogen**，并对其初始使用的能力印象深刻。
- **Userproxy 陷入沉默**：`@photo_mp` 遇到了一个问题：**userproxy** 在第一轮交互后停止通信，导致 Agent 之间的对话中断。
- **寻求 Autogen 建议**：`@photo_mp` 向社区寻求解决 Autogen 中 **userproxy** 问题的**建议**。
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1204975673748492338) (1 messages): 

- **聊天问题排查**：用户 `@yagilb` 建议，如果遇到问题，可以尝试删除有问题的聊天消息，或点击右上角的 **"reset to default settings"**（重置为默认设置）。
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1204823817026015252) (1 messages): 

- **寻找可视化工具**：用户 `@m4sterdragon` 询问是否有工具或方法可以在项目启动后**可视化 crew 交互**，暗示需要更好地监督团队动态。在现有消息中未提供解决方案或后续评论。
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1204869473794785300) (13 messages🔥): 

- **Open Interpreter Discord 澄清**：`@fkx0647` 询问了 **Open Interpreter (OI)** 的 Discord，最初误以为 LM Studio 的 Discord 就是它。`@heyitsyorkie` 澄清该频道只是 LM Studio 内部的一个子频道，并引导其前往 [GitHub 页面](https://github.com/KillianLucas/open-interpreter/) 上的 OI Discord。
- **理解本地模型服务器模式**：`@heyitsyorkie` 解释说 OI 可以利用 **LMStudio 的服务器模式来运行 Local Models**，阐明了 OI 与 LMStudio 之间的集成。
- **探索 OI 功能及 Discord 邀请协助**：`@fkx0647` 讨论了他们在 **OI** 方面的进展，并寻求帮助以找到官方 OI Discord 的邀请链接。`@heyitsyorkie` 就如何在 OI 的 GitHub README 中找到邀请提供了指导。
- **确认 GPT-3.5 使用情况**：`@fkx0647` 确认他们已经运行了 OI，使用的是 **GPT 3.5**，并指出 CLI 的性能令人满意。

**提到的链接**：

[GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter/)：计算机的自然语言界面。通过在 GitHub 上创建账号来为 KillianLucas/open-interpreter 的开发做出贡献。

  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1204895826116345867) (12 messages🔥): 

- **哲学 AI**：用户 `@gabriel_syme` 幽默地建议，如果模型是像苏格拉底那样的真正哲学家，就不会有太多讨论，因为苏格拉底在柏拉图的文本中经常主导对话。
- **Llama 模型安全性仅需 200 美元即可破解**：`@swyxio` 分享了一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email)，讨论了如何通过预算不到 200 美元的 LoRA 微调来撤销 Llama 2 的安全特性。该帖子强调了绕过强大模型安全训练的简易性及相关风险。
- **合成数据的兴起**：在一条简短的消息中，`@swyxio` 指出合成数据的创建正蓄势待发。
- **语音克隆即将迎来变革**：`@guardiang` 对一条 Discord 消息链接做出反应，指出语音克隆技术即将发生重大变化，可能带来新的挑战和机遇。
- **对即将发布的 GPT 的期待**：用户 `@coffeebean6887` 通过一个 [Twitter 链接](https://twitter.com/chiefaioffice/status/1755311356922732595) 暗示了一个未命名的 GPT 模型的到来，`@guardiang` 对看到它的实际表现表达了热情。同时，`@eugeneyan` 分享了兴奋之情，但也开玩笑说他们的 iPhone 屏幕限制了展示即将推出的 "Blind" 功能的能力。

**提到的链接**：

[LoRA Fine-tuning Efficiently Undoes Safety Training from Llama 2-Chat 70B — LessWrong](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email)：作为 SERI ML Alignment Theory Scholars Program - 2023 夏季班的一部分，在 Jeffrey Ladish 的指导下完成。…

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1204878568971173968) (1 messages): 

- **LLM 论文俱乐部启动**：`@swyxio` 宣布 `@713143846539755581` 将在 Latent Space Discord 论文俱乐部展示 **Self Reward 论文**。会议即将开始，成员可以点击[此处](https://lu.ma/llm-paper-club)加入。

**提到的链接**：

[LLM Paper Club (West) · Luma](https://lu.ma/llm-paper-club)：我们已移至使用新的 Discord Stage 功能：https://discord.com/channels/822583790773862470/1197350122112168006 稍后见！

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1204879363812757586) (209 messages🔥🔥): 

- **自我奖励语言模型引发关注**：论文俱乐部的讨论集中在一篇提出 *Self-Rewarding Language Models* 的新论文上，正如 `@coffeebean6887` 所概述的，这是一种新方法，LLM 在训练期间使用自己的输出来提供奖励，据称在某些方面优于 Claude 2 和 GPT-4。全文见[此处](https://arxiv.org/abs/2401.10020)。
  
- **对 DSPy 的兴趣激增**：DSPy 是一种用于链式调用 LLM 的编程模型，引起了广泛关注，包括 `@kbal11` 和 `@yikesawjeez` 在内的多位成员表示有兴趣进一步探索。`@yikesawjeez` 自愿主持一场关于 DSPy 的会议，论文见[此处](https://arxiv.org/abs/2312.13382)。

- **参与挑战性定理准备**：`@stephen_83179_13077` 将讨论的 LLM 论文与自动几何定理证明方法进行了比较，而 `@gabriel_syme` 指出对于复杂任务的评估，除了 LLM 之外还需要外部验证器。一篇关于通过拓扑结构洞察推理的建议论文见[此处](https://arxiv.org/abs/2401.14295)。

- **即将举行的论文俱乐部专题**：大家对未来的论文俱乐部会议充满期待，正如 `@amgadoz` 和 `@_bassboost` 所建议的，可能会讨论 CRINGE loss、Colbert 模型以及 T5 与 TinyLlama 的对比。此外，关于评述 "Leveraging Large Language Models for NLG Evaluation: A Survey" 的讨论已定于下周进行，论文见[此处](https://arxiv.org/pdf/2401.07103.pdf)。

- **DSPy 探索的 YouTube 资源**：`@yikesawjeez` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI)，作为深入研究 DSPyG 的资源，DSPyG 将 DSPy 与图优化器（Graph Optimizer）相结合，展示了一个带有图优化的 Multi Hop RAG 实现示例。

**提到的链接**：

- [推理拓扑：揭秘思维链、思维树与思维图 (Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts)](https://arxiv.org/abs/2401.14295)：自然语言处理（NLP）领域近年来见证了显著的进展，重点在于通过创新的提示词（prompting）技术提升大语言模型（LLM）的性能...
- [BirdCLEF 2021 - 鸟鸣识别 | Kaggle (Birdcall Identification)](https://www.kaggle.com/c/birdclef-2021)：未找到描述
- [自奖励语言模型 (Self-Rewarding Language Models)](https://arxiv.org/abs/2401.10020)：我们认为，为了实现超越人类水平的 Agent，未来的模型需要超越人类水平的反馈，以提供充足的训练信号。目前的方法通常根据人类偏好来训练奖励模型（reward models）...
- [改写网络：一种计算与数据高效的语言建模方案 (Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling)](https://arxiv.org/abs/2401.16380)：大语言模型是在海量的网络抓取数据上训练的，这些数据通常是无结构的、多噪声且表述欠佳。目前的缩放法则（scaling laws）表明，从这类数据中学习需要大量的...
- [通过代理微调语言模型 (Tuning Language Models by Proxy)](https://arxiv.org/abs/2401.08565)：尽管预训练大语言模型具有通用能力，但它们始终能从进一步的适配中获益，以更好地实现预期的行为。然而，微调这些模型已变得日益...
- [DSPy 断言：自优化语言模型流水线的计算约束 (DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines)](https://arxiv.org/abs/2312.13382)：将语言模型（LM）调用链接为可组合模块正在推动一种新的编程方式，但确保 LM 遵守重要的约束条件需要启发式的“提示工程（prompt engineering）”。我们引入了...
- [大语言模型评估综述 (A Survey on Evaluation of Large Language Models)](https://arxiv.org/abs/2307.03109)：大语言模型（LLM）在学术界和工业界都越来越受欢迎，这归功于它们在各种应用中前所未有的表现。随着 LLM 继续发挥至关重要的作用...
- [有些事情比其他事情更“尴尬”：使用成对 Cringe 损失的偏好优化 (Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss)](https://arxiv.org/abs/2312.16682)：从业者通常使用成对偏好来对齐大语言模型，即针对给定输入，标注响应 A 优于响应 B。或许较少见的是，一些方法已经...
- [self-rewarding-lm-pytorch/self_rewarding_lm_pytorch/self_rewarding_lm_pytorch.py at ec8b9112d4ced084ae7cacfe776e1ec01fa1f950 · lucidrains/self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch/blob/ec8b9112d4ced084ae7cacfe776e1ec01fa1f950/self_rewarding_lm_pytorch/self_rewarding_lm_pytorch.py#L127)：MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现 - lucidrains/self-rewarding-lm-pytorch
- [大语言模型（2023年） (Large Language Models (in 2023))](https://www.youtube.com/watch?v=dbo3kNKPaUA&t=899s)：我在首尔大学做了一场演讲。演讲题目是“大语言模型（2023年）”。这是一次总结我们爆炸式发展的领域的雄心勃勃的尝试...
- [构建你自己的产品 Copilot：挑战、机遇与需求 (Building Your Own Product Copilot: Challenges, Opportunities, and Needs)](https://arxiv.org/abs/2312.14231)：一场将先进 AI 能力嵌入产品的竞赛正在进行。这些产品 Copilot 使用户能够用自然语言提问，并获得针对特定用途的相关响应...
- [全新 DSPyG：DSPy 与 PyG 中的图优化器结合 (NEW DSPyG: DSPy combined w/ Graph Optimizer in PyG)](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI)：DSPyG 是一种基于 DSPy 的新优化方法，结合了图论见解。这是一个带有图优化的多跳（Multi Hop）RAG 实现的真实案例。
- [GitHub - lucidrains/self-rewarding-lm-pytorch: MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现](https://github.com/lucidrains/self-rewarding-lm-pytorch)：MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现 - GitHub - lucidrains/self-rewarding-lm-pytorch
- [未找到标题](https://ai.meta.com/blog/emu-text-to-video-generation-image-editing-research/)：未找到描述
- [无需人类演示解决奥数几何问题 - Nature (Solving olympiad geometry without human demonstrations - Nature)](https://www.nature.com/articles/s41586-023-06747-5)：一种新型的用于欧几里得平面几何的神经符号定理证明器，在数百万个合成定理和证明上从头开始训练，其表现优于以往的最佳方法，并达到了...
- [JupyterHub](https://bit.ly/basementagiclub)：未找到描述

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1204713742307295253) (128 messages🔥🔥): 

- **Mistral.AI 医疗实习**：`@talhawaqas` 是一位在法国的硕士生，正在寻找关于 **Mistral.AI 实际应用**的资源和论文，特别是关于医疗工程背景下的 Pretraining 和 Finetuning。
- **转向云端**：`@zhiyyang` 表示为了方便起见，打算开始在云端而非本地进行测试，并对社区分享的信息表示感谢。
- **Mistral 数据使用政策**：`@mrdragonfox` 澄清该服务会收集用户对话数据，并保留在 **Creative Commons Attribution (CC-BY)** 许可下发布数据集的权利。
- **探索 Mistral 中的响应长度控制**：`@lucacito` 和 `@mrdragonfox` 讨论了在 `@lucacito` 的作品集 Chatbot 助手背景下，如何设置 `max_tokens` 和 Temperature 以获得简洁的回答。`@mrdragonfox` 提供了示例，并就如何调整采样和 Temperature 设置给出了建议。
- **Temperature：是否设为零**：在关于将 Temperature 参数设置为 0 的激烈辩论中，`@i_am_dom` 建议虽然 0 度的 Temperature 对于 Chatbot 体验可能并不理想，但它可以提高性能并减少高精度任务中的幻觉（hallucinations）。`@mrdragonfox` 和其他人提出了各种 Temperature 设置以获得最佳的 Chatbot 响应（如 0.7），并建议通过测试找到最合适的设置。

**提到的链接**：

[Chat with Open Large Language Models](https://chat.lmsys.org/)：未找到描述

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1204874192088989776) (4 messages): 

- **发现类 Mistral 的 Embedding 模型**：用户 `@gbourdin` 询问是否能找到一个与 **Mistral-embed** 维度长度相同的 Embedding 模型用于本地开发。`@mrdragonfox` 回复了一个 [Hugging Face 链接](https://huggingface.co/intfloat/e5-mistral-7b-instruct)，指向 **E5-mistral-7b-instruct 模型**，该模型有 32 层，Embedding 大小为 4096，并提供了示例代码解释其用法。
- **对模型推荐表示感谢**：在收到模型推荐和使用详情后，`@gbourdin` 表达了感谢：“merci :)” 随后是 “thanks ! 🙂”。

**提到的链接**：

[intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct)：未找到描述

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1204781035389325322) (69 messages🔥🔥): 

- **LLMLAB 利用 Mistral**：`@joselolol.` 宣布使用 **Mistral** 运行 LLMLAB 业务以创建商业合成数据，并邀请用户在注册后私信激活账户。
- **对数据提取方法的好奇**：`@mrdragonfox` 询问了 LLMLAB 的数据提取流水线，但 `@joselolol.` 要求进一步澄清，表明可能会对该过程进行讨论。
- **Augmentoolkit 作为资源分享**：`@mrdragonfox` 分享了 [augmentoolkit 的 GitHub 链接](https://github.com/e-p-armstrong/augmentoolkit/)，并讨论了正在将其适配以支持 Mistral chat API，强调了该工具包在审查和评估数据方面的能力。
- **测试由 Mistral 驱动的新 Chat UI**：`@ethux` 发布了关于实现 **ETHUX Chat v0.7.0** 的消息，该版本使用了各种 Mistral 模型和 Huggingface ChatUI，设置了 200 欧元的测试 API 限制并分享了配置。
- **Chat UI 速率限制及源码公开**：在后续中，`@ethux` 提到每分钟两条消息的速率限制（rate limit），并分享了 HuggingFace chat UI 在 [GitHub](https://github.com/huggingface/chat-ui) 上是开源的，而 `@gbourdin` 对 UI 中的网页搜索功能表示了极大的热情。

**提到的链接**：

- [ETHUX Chat](https://chat.ethux.net)：由 PlanetNode 倾情打造 ❤️
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit/)：将算力和书籍转换为 Instruct-Tuning 数据集 - GitHub
- [GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui)：驱动 HuggingChat 应用的开源代码库。通过在 GitHub 上创建账户为 huggingface/chat-ui 的开发做出贡献。

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1204748073985904700) (2 messages): 

- **询问下一个 Mistral AI 模型发布时间**：用户 `@j673912` 询问了 **下一个 Mistral AI 模型** 的发布情况。`@mrdragonfox` 回复建议关注 **社交媒体** 以获取官方发布公告。
  

---

### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1204894222545719376) (1 messages): 

- **Embedding 模型在处理加拿大国旗时遇到问题**：`@enjalot` 在尝试使用 dolly15k 数据集进行模型 Embedding 时遇到错误，并分享了一个与加拿大国旗相关的[特定错误消息](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/default/train?p=108&row=10834)。他们已成功处理了前 10k 行，这表明问题仅局限于这段特定的文本。

**提到的链接**：

[databricks/databricks-dolly-15k · Datasets at Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/default/train?p=108&row=10834)：未找到描述

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1204747553778700308) (1 messages): 

- **面向求职者的新论坛频道**：`@lunarflu` 宣布推出了一个专门针对工作机会的 **Forum** 频道，用户可以通过 `internship`、`machine learning`、`remote` 和 `long-term` 等标签进行筛选。鼓励社区成员使用该功能并提供反馈。
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1204720625613275197) (115 messages🔥🔥): 

- **GPU 加速的疑问**：用户 `@ilovesass` 询问了 `pipeline.to("cuda")` 的功能，`@sandeep_kr24` 解释说它将整个计算图移动到支持 CUDA 的 GPU 上，从而加快处理速度。
- **Stable Diffusion 相关说明**：`@yamer_ai` 提供了使用 Stable Diffusion 的帮助，并强调了自己在训练像素艺术生成模型方面的个人成就，而 `@tmo97` 对非技术用户如何使用该技术表示困惑，并得到了 `@yamer_ai` 的指导。
- **技术需求与建议**：用户 `@p1ld7a`、`@drfhsp` 和 `@.volvite` 分别就设置查询本地 LLM 的 Web UI、在 Colab 的免费 T4 上使用特定模型进行推理以及解决 Gradio 问题寻求建议。他们被引导至 Python、Gradio 等资源，以及由 `@electriceccentrics` 和 `@lee0099` 等分享者提供的 Colab Notebook。
- **HuggingFace Spaces 后端过载**：`@thomaslau.001` 分享了一个 Reddit 链接，讨论在 HuggingFace Spaces 上使用 Nvidia A10G 运行 Codellama 时出现的“显存溢出 (out of memory)”错误，并寻求帮助；`@vipitis` 建议尝试运行 70B 模型可能会超出显存限制，即使是在降低精度（fp16）的情况下。
- **协作邀请与讨论**：包括 `@soul_syrup`、`@technosourceressextraordinaire`、`@jdreamer200` 和 `@electriceccentrics` 在内的多位用户提到了他们的项目，涵盖了从神经信号分析、强化学习（RL）机器人到求职以及关于云服务成本的金融幽默，引发了社区内的社交互动和项目兴趣。

**提到的链接**：

- [ - YouTube](https://www.youtube.com/watch?v=CZaG3&ab_channel=p3nGu1nZz)：未找到描述
- [Oppenheimer Oppenheimer Movie GIF - Oppenheimer Oppenheimer movie Barbie oppenheimer meme - Discover &amp; Share GIFs](https://tenor.com/view/oppenheimer-oppenheimer-movie-barbie-oppenheimer-meme-cillian-murphy-theory-gif-8102327772591152629)：点击查看 GIF
- [How to install stable diffusion 1.6 Automatic1111 (One Click Install)](https://www.youtube.com/watch?v=IoWPsNwXLVc&t=36s)：🔔 订阅 AIconomist 🔔SD Automatic1111 1.6 ➤ https://github.com/AUTOMATIC1111/stable-diffusion-webui Python 3.10.6 ➤ https://www.python.org/downloads/...
- [无标题](https://tenor.com/view/oppenheimer-oppenheimer-movie-barbie-oppenheimer-meme-cillian-murphy-theory-g)：未找到描述
- [Easter Funny Shrek GIF - Easter Funny Shrek Funny Face - Discover &amp; Share GIFs](https://tenor.com/view/easter-funny-shrek-funny-face-gif-16873758)：点击查看 GIF
- [GitHub - cat-game-research/Neko: A cat game beyond.](https://github.com/cat-game-research/Neko)：超越猫咪的游戏。通过在 GitHub 上创建账号为 cat-game-research/Neko 的开发做出贡献。
- [Reddit - Dive into anything](https://www.reddit.com/r/huggingface/comments/1alpsvp/cuda_out_of_memory_on_nvidia_a10g_codellama_on/)：未找到描述
- [GitHub - Unlimited-Research-Cooperative/Human-Brain-Rat: Bio-Silicon Synergetic Intelligence System](https://github.com/Unlimited-Research-Cooperative/Human-Brain-Rat)：生物-硅协同智能系统。通过在 GitHub 上创建账号为 Unlimited-Research-Cooperative/Human-Brain-Rat 的开发做出贡献。

  

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1204708404023525448) (2 messages): 

- **Meta-RL 应用开发机会**：`@davidebuoso` 正在开发一个 **Meta-RL 应用**，并正在为一个副业项目寻找合作者。该项目与他们策划的 GitHub 列表 ([Meta-RL Panda](https://github.com/stars/lambdavi/lists/meta-rl-panda/)) 中的高级作品相关，特别是要在使用 gym 的 Panda 机器人手臂上进行测试。

- **个人成就与项目展示**：`@antiraedus` 分享了他们的每周更新，提到了改掉坏习惯、获得一份**大学助教工作**以及参加社交活动。他们还在根据 Flutter 平台游戏教程开发一款游戏，并计划在下周前分享。全年的总体主题是争取 **internships**，同时创建“可分享且有形”的项目作为动力和 Proof of Work。
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1204741110984089630) (4 messages): 

- **探索 Evan Hubinger 等人的前沿研究**：`@opun8758` 分享了一篇由 **Evan Hubinger** 及其同事共同撰写的[新研究论文](https://arxiv.org/abs/2401.05566)，强调了它对社区的潜在价值。
- **寻找 RWKV 模型**：`@vishyouluck` 询问了关于微调 **RWKV 模型** 的经验，引发了建模爱好者的好奇。
- **Eagle 7b 发布缺乏细节**：`@vishyouluck` 提到了 **Eagle 7b**，但没有提供关于该模型的更多背景或信息。
- **Code Llama 凭借 70B 模型迈向未来**：`@jashanno` 分享了一篇 [beehiiv 文章](https://natural20.beehiiv.com/p/code-llama-70b)，宣布 **Facebook/Meta** 发布了 **Code Llama**，这是一个拥有 700 亿参数的新型语言模型，旨在帮助编码人员和学习者，并在社区友好型许可下提供。

**提到的链接**：

- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)：人类能够表现出策略性的欺骗行为：在大多数情况下表现得很有帮助，但在有机会时，为了追求其他目标而表现得截然不同。...
- [META&#x27;s new OPEN SOURCE Coding AI beats out GPT-4 | Code Llama 70B](https://natural20.beehiiv.com/p/code-llama-70b)：此外：关于 ChatGPT 的隐私担忧、科技巨头面临的压力等...

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1204754489085530112) (9 条消息🔥): 

- **实现 Web 集成**：`@wubs_` 确认 `@aliabbas60` 提到的生成器可以在 **Web 和渐进式 Web 应用 (PWA)** 格式下运行，在移动端和桌面设备上均表现良好。
- **新 AI 项目展示**：`@critical3645` 提到从零开始创建了一个项目，并附上了视频演示链接。
- **SD.Next 发布重大增强功能**：`@vladmandic` 发布了 [SD.Next](https://github.com/vladmandic/automatic) 的全面更新，引入了强大的 **Control 模块**、**Face 模块**、改进的 **IPAdapter** 模块、新的智能遮罩选项，以及众多新模型和流水线。他们强调了性能提升，在 nVidia RTX4090 上的基准测试达到 **110-150 iterations/second**，并引导用户在其 [Wiki](https://github.com/vladmandic/automatic/wiki) 中查看完整文档和更新。
- **社区准则提醒**：`@cakiki` 提醒 `@vladmandic` 遵守社区准则，移除所有 Discord 邀请链接，`@vladmandic` 随即进行了处理。
- **角色扮演项目 Docker 化**：Krolhm 宣布将角色扮演 (RP) 项目 ImpAI 进行 Docker 化，以便更轻松地配合 `hf pipeline` 和 `llama.cpp` 使用，并分享了 GitHub 仓库 [ImpAI](https://github.com/rbourgeat/ImpAI)。

**提到的链接**：

- [GitHub - rbourgeat/ImpAI: 😈 ImpAI is an advanced role play app using large language and diffusion models.](https://github.com/rbourgeat/ImpAI)：😈 ImpAI 是一款使用大语言模型和扩散模型的高级角色扮演应用。
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/OpenVINO),)：SD.Next：Stable Diffusion 及其他基于扩散生成的图像模型的高级实现 - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/Intel-ARC))：SD.Next：Stable Diffusion 及其他基于扩散生成的图像模型的高级实现 - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive))：SD.Next：Stable Diffusion 及其他基于扩散生成的图像模型的高级实现 - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/Benchmark))：SD.Next：Stable Diffusion 及其他基于扩散生成的图像模型的高级实现 - Create new page · vladmandic/automatic Wiki

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1204769948849668167) (4 条消息): 

- **tonic_1 展示 Mamba SSM**：用户 `@tonic_1` 展示了 [mamba 库](https://github.com/state-spaces/mamba/tree/main/mamba_ssm)，并强调了他个人对 `utils`、`ops` 和 `modules` 功能的兴趣。
- **Chad 寻求状态空间数学见解和权衡**：用户 `@chad_in_the_house` 对 Mamba 提供的状态空间模型 (State Space Models) 的数学方面，以及它们与 Transformer 和 RWKV 的对比表现出了兴趣。
- **寻求对 Mamba 的清晰解释，重复内容也没关系**：用户 `@chad_in_the_house` 还表示，在演示过程中重复视频中的信息是可以接受的，因为大多数参与者可能并不熟悉 Mamba。
- **请求简述 Mamba 历史**：除了了解权衡之外，`@chad_in_the_house` 还请求简要介绍 Mamba 至今的发展历史。

**提到的链接**：

[mamba/mamba_ssm at main · state-spaces/mamba](https://github.com/state-spaces/mamba/tree/main/mamba_ssm)：通过在 GitHub 上创建账号来为 state-spaces/mamba 的开发做出贡献。

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204833801176481833) (1 条消息): 

- **寻求对 Transformer 的理解**：用户 `.slartibart` 询问是否有**涵盖编写代码的 Transformer 的演示**，特别是那些深入探讨 Transformer 从采样代码中学习到什么的资源。他们正在寻找相关资源，以更好地理解 Transformer 在代码生成背景下的学习过程。
  

---

### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1204862876548993085) (4 messages): 

- **寻求用于行分割的 OCR**：用户 `@swetha98` 询问是否有 OCR 工具可以将文档图像分割成四个独立的图像，每个图像包含样本发票上的四行文本之一。
- **推荐 Surya OCR**：针对 `@swetha98` 对 OCR 解决方案的需求，`@vikas.p` 建议使用 [Surya OCR](https://github.com/VikParuchuri/surya)，它可以对任何语言进行精确的行级文本检测和识别。
- **对推荐的积极反馈**：`@swetha98` 对 `@vikas.p` 的建议表示感谢，并提到如果该方案能满足需求，打算引用他的 GitHub 仓库。

**提到的链接**：

[GitHub - VikParuchuri/surya: Accurate line-level text detection and recognition (OCR) in any language](https://github.com/VikParuchuri/surya)：适用于任何语言的精确行级文本检测和识别 (OCR) - GitHub - VikParuchuri/surya: Accurate line-level text detection and recognition (OCR) in any language

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1204734058957901854) (11 messages🔥): 

- **寻求用于 RAG Pipeline 的相关性检查器**：`@tepes_dracula` 正在构建 **RAG pipeline**，并寻找一个 **classifier**，用于在将查询和上下文传递给 LLM 之前验证它们的相关性。他们表示希望工具能借鉴 **truthfulqa** 和 **fever** 等数据集。

- **在 Windows 上加速 Deci lm**：`@kingpoki` 希望提高在 Windows 本地运行的 **Deci lm** 的推理速度，并寻找 **model quantization** 之外的建议。

- **衡量技术领域的概念相似度**：`@serhankileci` 正在探索如何计算各种技术相关领域中概念、工具和语言之间的相似度百分比。他们考虑使用 word embeddings 和 cosine similarity，并承认缺乏用于此目的的单一 **encyclopedic dataset**。

- **LLM Instruction Tuning 中的数据质量**：`@Chris M` 强调了 **data quality for LLM instruction tuning** 的重要性，并分享了一篇关于自动检测低质量数据技术的文章，指出 **bad data 是影响性能的常见罪魁祸首**。该文章可以在 [cleanlab.ai](https://cleanlab.ai/blog/filter-llm-tuning-data/) 找到。

- **关于 HuggingFace TAPAS 模型的咨询**：`@nitachaudhari29` 简单地打了个招呼，并询问是否有人有使用 HuggingFace **TAPAS model** 的经验。

- **多标签文本分类的现状**：`@simpleyuji` 询问目前 **state-of-the-art (SOTA) for multi-label text classification** 的情况。

- **小 Batch Size 的优缺点**：`@abrahamowodunni` 提出了一个问题：在使用小数据集微调模型时，除了增加训练时间外，使用 batch size 为 4 有哪些缺点。`@vipitis` 回复称，1.2k 步长下 batch size 为 4 可能已经足够，但通常情况下，如果 VRAM 允许，更倾向于使用 **larger batch size**。

**提到的链接**：

[How to detect bad data in your instruction tuning dataset (for better LLM fine-tuning)](https://cleanlab.ai/blog/filter-llm-tuning-data/)：自动检测工具概览，用于捕捉指令数据集中潜伏的：低质量回答、不完整/模糊的 Prompt 以及其他问题文本（毒性语言、PII、非正式写作、语法/拼写错误）...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204833801176481833) (1 messages): 

- **寻找 Transformer 见解**：用户 `.slartibart` 询问是否有关于 **code-writing transformers** 的演示文稿，并深入探讨 Transformer 从 *sampled code* 中学到了什么。他们正在寻找试图描述 Transformer 学习过程的资料。
  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1204702480257122344) (79 条消息🔥🔥): 

- **关于 Bard Ultra 发布时间的推测**：`@la42099` 幽默地评论了对 **Bard Ultra** 发布的热切期待，并承认这类预测往往是不准确的。
- **ChatGPT 访问问题**：`@chrisrenfield` 在访问 **ChatGPT** 时遇到困难，`@satanhashtag` 建议访问 **[OpenAI's service status](http://status.openai.com/)** 获取更新。
- **寻求用于珠宝视觉化的 AI**：`@jonas_54321` 正在寻找一种 AI 工具，用于在模特身上生成逼真的珠宝图像。`@satanhashtag` 推荐了 Bing 上免费提供的 **Dall-E 3**，并提到了个人更倾向于使用 **Midjourney**。
- **服务条款的约束**：围绕 **Terms of Service (TOS)** 展开了讨论，由 `@wrexbe` 发起，探讨了如何创造性地规避与生成 NSFW 内容相关的限制和内容警告。其他用户（如 `@drinkoblog.weebly.com` 和 `@chotes`）讨论了使用 AI 处理各种内容的效率和伦理角度。
- **探讨通往 AGI 之路**：`@drinkoblog.weebly.com` 和 `@chotes` 之间的对话探讨了 AGI 的发展，用户表达了对 AI 模型审查的担忧，思考其对未来 AGI 能力和自由的影响。
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1204726906856022046) (30 条消息🔥): 

- **GPT 延迟问题被提出**：用户 `@theotherway` 和 `@pneumaone` 报告了 GPT-4 的性能问题，称其运行缓慢，并认为其表现类似于旧版本（可能是 GPT-1）。`@pneumaone` 指出最近的一次更新似乎显著降低了模型的能力。
   
- **用户注意到更新后 GPT 性能下降**：`@pneumaone` 和 `@_loier` 讨论了 GPT 性能的明显下降，认为它开始遗忘指令，并且不像以前那样遵循自定义设置。
  
- **呼吁增加“回读”功能**：`@blaynomtops` 请求增加一个将文本读给用户听的功能，以减轻整天阅读带来的眼睛疲劳。`@blckreaper` 指出该功能在移动端已存在，但在 PC 端没有。
  
- **叙事生成中的 Token 管理**：`@blckreaper` 批评 GPT 在叙事生成过程中对 tokens 的使用效率低下，质疑为什么它经常对角色进行反思而不是推进故事情节。
  
- **间歇性连接和性能问题**：`@bennnjji` 询问了与连接 GPT API 相关的错误，而 `@pneumaone` 和 `@_loier` 等多位用户报告称，他们的 GPT 实例出现停滞、出错以及交互质量下降的情况。
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1204817906576130178) (8 条消息🔥): 

- **聊天机器人身份之谜**：用户 `@chotes` 戏谑地推测用户中谁可能是某个叫 Sam 的人的 **alt**（小号/备用账号）。
- **频道中的问候**：`@darthgustav.` 带着友好的问候加入对话，`@chotes` 对此做出了回应，可能怀疑 darthgustav. 是 Sam 的小号。
- **讨论小号**：`@lugui` 加入了玩笑，与 `@chotes` 一起暗示某个叫 Sam 的人使用了 “Alt” 账号。
- **报告 Prompt Leakage 担忧**：`@raidedcluster` 分享了他们发现的一种在自定义 GPTs 中进行 prompt leakage（提示词泄露）和违反政策的方法，并提到该方法在 Kahn Academy 的 Khanmigo 上非常有效。
- **建议官方反馈渠道**：`@eskcanta` 向 `@raidedcluster` 提供了 [OpenAI model feedback form](https://openai.com/form/chat-model-feedback)，作为报告所发现问题的正式渠道。

**提到的链接**：

[Chat model feedback](https://openai.com/form/chat-model-feedback)：未找到描述

  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1204817906576130178) (8 messages🔥): 

- **谁是小号？身份探寻进行中**：`@chotes` 幽默地试探群组成员，试图找出谁是 Sam 的小号 (Alt account)，这引发了一场轻松的交流，`@darthgustav.` 进行了回应，随后 `@chotes` 承认：“说实话，我早该想到的……”
- **Sam 的秘密身份？**：`@lugui` 插话建议 Sam 可能真的在使用一个 Alt 账号，增加了这种开玩笑式的推测。
- **报告安全问题**：用户 `@raidedcluster` 发现了一种可能导致 Prompt 泄露和违反内容政策的方法，并寻求如何报告的建议，特别提到了该方法在 Khan Academy 的 Khanmigo 上的有效性。
- **报告问题的得力助手**：`@eskcanta` 向 `@raidedcluster` 提供了关于在哪里报告 Prompt 泄露和内容政策违规问题的详细建议，包括 [OpenAI 模型反馈表单](https://openai.com/form/chat-model-feedback) 的链接和说明。
- **对支持表示感谢**：针对 `@eskcanta` 的指导，用户 `@raidedcluster` 以一句简单的“Thank you!”表达了对获取报告问题信息的感激之情。

**提到的链接**：

[Chat model feedback](https://openai.com/form/chat-model-feedback)：未找到描述

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1204710908149112872) (31 messages🔥): 

- **Google Scholar Prompt 咨询等待回复**：`@aldikirito` 询问了在 **Google Scholar** 中搜索**期刊引用**的具体 Prompt，但目前尚未得到明确指导。
- **Gemini Pro 与 3.5 Turbo 的对比**：根据 `@moyaoasis` 的说法，**Gemini Pro** 在写作方面表现出色，但在数学和编程方面不如 **3.5 Turbo**，尤其是对于编程新手而言。
- **Perplexity API 使用疑问**：`@horsecode` 询问了在现有辅助机器人中使用 Perplexity API 的情况，并引用了 ToS；`@clay_ferguson` 进一步讨论了 ToS 的影响，推测其限制性使用政策是出于竞争原因。
- **定义“回答引擎” (Answer Engines)**：`@twelsh37` 回应了 `@tyronemichael`，将 **Perplexity** 解释为一种“回答引擎” (Answer Engine)，将其与 **Google** 等传统搜索引擎区分开来，并链接了 Riley Brown 的 [YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=344s) 以获取更多见解。
- **寻求 Perplexity 中 SEO 文章的技巧**：`@tuastowers_71714` 寻求在 Perplexity 中创建 SEO 优化文章的建议；`@noremac258` 分享了个人使用的 Prompt，并建议使用 **Grammarly** 来检查内容质量。

**提到的链接**：

- [Terms of Service](https://blog.perplexity.ai/legal/terms-of-service)：隐私 • 条款与条件
- [I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=344s)：视频主要看点：“我使用 Perplexity 超过了 ChatGPT、BARD 和 Microsoft Copilots，主要有五个原因，包括它在内容创作中的应用……”

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1204788740007133225) (5 messages): 

- **Perplexity AI 引发批评**：用户 `@deeceem22_16584` 对 Perplexity AI 表示失望，提到一个关于在一个下午参观三个博物馆的用例示例是不切实际的。
- **对“博物馆特种兵式游览”的怀疑**：针对对 Perplexity AI 用例的批评，`@brknclock1215` 幽默地评论说，能在一个下午走过卢浮宫的入口就已经很幸运了。
- **Perplexity API 的创新用途**：`@foxplaid19973` 分享了一个有趣的 Perplexity AI API 应用：将其与 Lego Mindstorm 机器人集成，创建了一个一键通话助手，该助手使用 Google 的语音识别并朗读来自 Perplexity 的回答。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1204601391633530920)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1204797350808395807) (43 条消息🔥): 

- **API 缺少来源引用**：`@tyronemichael` 询问 API 何时会包含来源和引用。尽管之前有相反的迹象，`@me.lk` 澄清说，来源引用目前**不在 API 的路线图（roadmap）上**。

- **GPT 模型在 API 中缺失**：`@fanglin3064` 询问如何通过 Perplexity API 访问 GPT 模型，结果从 `@me.lk` 处获知 **Perplexity 不通过 API 服务提供 GPT 模型**。

- **禁止抓取**：`@fanglin3064` 表示有兴趣抓取使用 GPT 模型的 Perplexity AI 结果以验证准确性。`@icelavaman` 警告说，**抓取违反了服务条款（TOS）**，是不允许的。

- **API 与 UI - 两个不同的产品**：在关于 API 是否由 Perplexity AI 官方提供的讨论中，`@me.lk` 和 `@icelavaman` 澄清说，**[pplx.ai](https://pplx.ai)** 和 API 都是同一家公司提供的不同产品，但遵循不同的访问协议。

- **选择最佳模型**：`@fanglin3064` 想知道 API 列表中哪个模型在性能上最接近 GPT-3.5 turbo。`@icelavaman` 建议尝试 **mixtral-8x7b-instruct** 以在无网络访问的情况下获得良好性能，并建议使用 **pplx-7(0)b-online** 模型以在有网络访问的情况下获得快速响应，同时还提供了上述模型的 [基准测试链接](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047202784090538054/1204821989919957002)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Moon (深色模式)](https://docs.perplexity.ai/discuss/65c0b02f09d8e3001ca0d3ba)：未找到描述
- [介绍 PPLX Online LLMs](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)：首创的 Online LLM API

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1204829291209297971) (4 条消息): 

- **通过用户反馈增强 RAG**：LlamaIndex 团队强调了在 **RAG 之上构建 Agent 层**的重要性，以便在搜索/检索任务期间促进实时用户反馈，从而增强复杂查询的用户体验。有关此功能的讨论可以在他们的推文中查看 [此处](https://twitter.com/llama_index/status/1755270274310901898)。

- **深入探讨 Query Pipelines**：提供了关于 LlamaIndex **Query Pipelines** 功能的全面指南，通过更具声明性的方法展示了高级检索策略。解释内容包括各种检索方法，可以通过分享的 [Medium 文章](https://t.co/Funqm7Jw1u) 进一步探索，详见 Twitter [此处](https://twitter.com/llama_index/status/1755342965667696870)。

- **来自 LlamaIndex 的 RAG 见解**：来自 LlamaIndex 的 `@seldo` 在 RocksetCloud 主办的网络研讨会中与 `@ankit1khare` 讨论了 RAG 的细微差别，涵盖了其目的、执行以及 2024 年即将推出的功能。感兴趣的观众可以观看 [YouTube](https://t.co/7f12VvgImc) 上的完整研讨会，如 [此处](https://twitter.com/llama_index/status/1755378511819456841) 所述。

- **邀请参加 LlamaIndex 活动**：分享了 LlamaIndex 主办的即将举行的活动的公开邀请，鼓励大家参与。活动详情可以在 [此处](https://twitter.com/llama_index/status/1755393423660724379) 找到。

**提到的链接**：

[使用 LlamaIndex 为高级 RAG 工作流设置 Query Pipeline](https://t.co/Funqm7Jw1u)：什么是 QueryPipelines？

  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1204725834540253184) (65 messages🔥🔥): 

- **NLSQLTableQueryEngine 故障排除**：用户 `@nzmrs7` 建议在使用 `NLSQLTableQueryEngine` 时提供表信息，并提议通过 `print(response.metadata)` 检查生成的查询，以明确生成的 SQL 查询。
- **Gemini 连接问题**：用户 `@whitefang_jr` 建议检查 Google Console 的审核过滤器（moderation filters），这可能是导致 `@mowlidharan` 在连接 Gemini 时遇到问题的原因。
- **Reranker 引入策略**：针对 `@theoxd` 关于引入 reranker 的咨询，`@kapa.ai` 提供了在 **LlamaIndex** 库的 retriever 中设置 `LLMRerank` and `SentenceTransformerRerank` 的详细示例。
- **Deeplake 的 Ingestion Pipeline**：用户 `@ramihassanein` 讨论了在处理 Deeplake 等 vector stores 时，使用 `index.refresh_ref_docs` 进行去重的问题；这促成了该问题的协作修复，并由该用户提交了 PR。
- **自定义 LLM 返回 JSON**：`@nbulkz` 分享了一个挑战：微调后的 LLM 返回的是 JSON 对象而非字符串，导致了错误。`@cheesyfishes` 建议可以创建一个自定义 LLM，或者稍后将字符串解析回 JSON。

**提到的链接**：

- [llama_index/llama_index/indices/base.py at cc739d10069a7f2ac653d6d019fbeb18a891fea2 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/cc739d10069a7f2ac653d6d019fbeb18a891fea2/llama_index/indices/base.py#L310)：LlamaIndex（前身为 GPT Index）是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1163880111074971790/1163900056718553169)：Discord 是进行语音、视频和文字交流最简单的方式。与您的朋友和社区聊天、聚会并保持紧密联系。
- [llama_index/llama_index/ingestion/pipeline.py at 1843c6d806702dc12bd771ac1362b6c3c2931ca2 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/1843c6d806702dc12bd771ac1362b6c3c2931ca2/llama_index/ingestion/pipeline.py#L191)：LlamaIndex（前身为 GPT Index）是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index
- [Ingestion Pipeline + Document Management - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html)：未找到描述
- [Finetuning an Adapter on Top of any Black-Box Embedding Model - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding_adapter.html)：未找到描述
- [upgraded deeplake vector database to use BasePydanticVectorStore by rumsrami · Pull Request #10504 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/10504)：描述：将向量数据库 Deeplake 升级为使用 BasePydanticVectorStore 而非 VectorStoreBase。这将允许其在 Ingestion pipelines 中使用。变更类型：请...
- [Query Pipeline - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html)：未找到描述
- [Defining a Custom Query Engine - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine.html#defining-a-custom-query-engine)：未找到描述

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1204736828859482132) (1 messages): 

- **寻求生产级 RAG 系统建议**：`@turnerz` 正在寻求关于构建生产级 **RAG HR 聊天机器人系统**的建议。他们正在寻找关于 vector databases、chunk sizes、embedding models 的推荐，并对项目进度和数据库组织有具体疑问。
- **循序渐进策略 vs. 直接深入**：在推进其 RAG HR 聊天机器人项目时，`@turnerz` 正在考虑是**从一个初级的 RAG 系统开始并迭代改进**，还是根据他读到的一篇文章，**直接从更复杂的 multi-hop retrieval 系统开始**。
- **数据库困境：一个还是多个？**：`@turnerz` 还在询问是**为不同的文档构建不同的 vector databases** 更好，还是将它们合并为一个，并举了处理两个独立 PDF 的例子。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1204765064951767060) (12 messages🔥): 

- **部分用户无需使用 JupyterLab**：`@caseus_` 提到，**如果不使用 JupyterLab**，可以忽略某条消息，这促使 `@youraveragedev` 表达了对使用 JupyterLab 的偏好。
- **修改云端入口点的建议**：`@caseus_` 建议修改 cloud-entrypoint 脚本以**指向不同的目录**，具体为 `/workspace/axolotl`。
- **廉价 GPU 云服务警报**：`@dreamgen` 分享了关于一家提供 H100 的云服务信息，价格约为 **每 GPU 小时 2.8 美元**，并邀请成员私信获取链接。
- **云服务计费细节至关重要**：`@yamashi` 警告说 `@dreamgen` 提到的云服务采用**向上取整计费**，`@nanobitz` 对此表示失望。`@le_mess` 补充说希望能有折扣。
- **项目 Star 数增加**：`@dreamgen` 提到给一个项目点了 Star，并表示以前没这么做过，这引发了 `@yamashi` 的轻松回应。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1204752545310965810) (24 messages🔥): 

- **RunPod 问题持续存在**：`@casper_ai` 提到 RunPod 上的静默更新导致了失败。`@m4ttfl0` 反复遇到该问题，注意到仓库被破坏，且预期的 500GB “persistent volume” 不存在——这一问题已通过多个诊断命令确认。

- **尝试不同的 Pod**：`@dreamgen` 定期在社区云上启动新的 Pod，偶尔也在安全云上启动，且没有遇到这些问题；而 `@m4ttfl0` 终于启动了一个未被破坏的仓库，但仍面临卷丢失的问题。

- **寻求稳定方案**：`@m4ttfl0` 和 `@caseus_` 讨论了潜在的解决方案，如更改持久卷的挂载点或 Docker 镜像中的设置目录。他们建议将卷挂载在 `/workspace/axolotl/data` 下，以便更好地管理磁盘空间。

- **缓存和输出的考虑**：`@caseus_` 指出 Huggingface 缓存目前指向 `/workspace/data`，这要求基础模型和数据集也必须存放在持久卷中。更改可能涉及更新示例以使用 `output_dir: ./data/out`。

- **记录 RunPod 问题**：`@m4ttfl0` 同意在 GitHub 上记录该问题及建议的解决方案，旨在帮助他人理解这些更改及其原因。他们提出愿意协助测试任何新更改，以确保其按预期工作。

- **持续预训练 PR**：`@jinwon_k` 提交了一个 PR，涉及一个针对大语言模型持续预训练优化的新 scheduler。可以通过提供的 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273) 查看该 PR。

**提到的链接**：

[Scheduler implementation of Continual Pre-Training of Large Language Models: How to (re)warm your model?  by jinwonkim93 · Pull Request #1273 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273)：大语言模型持续预训练的调度器实现：如何（重新）预热你的模型？(https://arxiv.org/pdf/2308.04014.pdf) 描述与 cosine min lr 几乎相同，但我...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1204716289503797248) (10 messages🔥): 

- **寻找开放预训练数据集**：`@Sebastian` 询问了当前的**开放预训练数据集**；`@le_mess` 将他引导至另一个频道的相关讨论以获取更多信息。
- **为 Python 查询设计 Prompt 格式**：`@nafnlaus00` 寻求关于创建一个用于 Python 模型查询的**简洁数据结构**的反馈，讨论了 JSON 对非英语字符转义的问题，并探索了替代方案，如使用 protobuf 和自定义编码方案。
- **Protobuf 的可能性及其转换**：`@nafnlaus00` 展示了一种涉及 protobuf 的方法，用于在表示结构化数据时保持文本不变，并演示了将字节编码和解码为 UTF-8 字符串的**可逆操作**。
- **为了更好的灵活性采用自定义编码**：在权衡各种选项后，`@nafnlaus00` 倾向于使用一种利用冷门 Unicode 字符来描绘数据结构元素的自定义编码方法，考虑到其在格式控制和可扩展性方面的优势。
- **关于编码方法的建议和考虑**：虽然 `@caseus_` 赞同为控制字符添加必要 Token 的想法，但 `@dreamgen` 建议使用**带有自定义 Token 的纯文本**进行编码，并寻求关于使用场景的进一步说明。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1204724219535958037) (12 messages🔥): 

- **学习率与模型配置**：用户 `@dreamgen` 讨论了学习率如何影响他们在小数据集上的项目结果。随后他们分享了可配置参数，并提到在 **DPO 中使用了 unsloth**，并打算很快开源该脚本。

- **分享 DreamGenTrain 脚本**：用户 `@dreamgen` 提供了他们训练方法的 GitHub 脚本链接，提到使用了 **paged_adamw_8bit** 和 **linear scheduler**（线性调度器）。脚本可在 [此处](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py) 获取。

- **关于 Batch Size 的建议**：`@dreamgen` 建议可以**增加 micro batch size**，除非处理的是像他们项目中那样的极长序列。

- **感谢信息分享**：用户 `@fred_fups` 对 `@dreamgen` 分享训练设置和策略的细节表示感谢。

- **关于 Self-Rewarding 方法的咨询与合作**：用户 `@dctanner` 表示有兴趣实验他们提到的一篇论文中的 Self-Rewarding 方法，并邀请合作，同时考虑将其集成到 **axolotl** 中。

**提到的链接**：

[DreamGenTrain/dpo.py at master · DreamGenX/DreamGenTrain](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py)：通过在 GitHub 上创建账户，为 DreamGenX/DreamGenTrain 的开发做出贡献。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1204828094347739227) (2 messages): 

- **构建 RunPod 镜像需要特定设置**：用户 `@gonejiggy` 询问如何构建 RunPod 镜像，因为本地构建不起作用，且 Docker 无法在另一个 Docker 内部运行。`@caseus_` 回复称，构建 RunPod 镜像需要**一台带有 GPU 且安装了裸机 Docker (bare metal docker) 的机器**。
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1204706348877611019) (22 messages🔥): 

- **LLaMa 的快速绘图**：`@top_walk_town` 成功训练了一个 2M 参数的 LLaMa 模型，用于条件生成 MNIST 数字 "7"，并吹嘘在 8GB MacBook M1 上的训练时间约为 25 分钟。
- **Llava 1.5 vs 1.6**：`@SegmentationFault` 分享了一个关于某项目的 Twitter [链接](https://vxtwitter.com/billyuchenlin/status/1755207605537120513)，随后评论说遗憾该项目没有使用 Llava 1.6，并指出新版本本可以提供改进。
- **DALL-E 3 水印**：`@vrus0188` 提供了一个 [The Verge 文章链接](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials)，讨论了 OpenAI 的 DALL-E 3 在图像元数据中添加的新水印，包括隐形和显形组件。
- **通过 Hugging Face 招聘**：针对 `@_definitely_not_sam_` 寻求为计算机视觉项目招聘开发者的需求，`@chad_in_the_house` 建议使用他们服务器中的 Hugging Face 职位频道发布正式招聘信息。
- **表情符号胜过千言万语**：`@astropulse`、`@pseudoterminalx` 和 `@drhead` 等用户使用各种表情符号对不同话题做出反应，没有具体内容需要总结。

**提到的链接**：

[OpenAI 正在为 DALL-E 3 添加新水印](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials)：然而，水印仍然可以被擦除。

  

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1204748896790913054) (27 条消息🔥): 

- **新款背景移除工具发布**：`@qwerty_qwer` 分享了一个新的 SOTA 背景移除工具 BRIA Background Removal v1.4 的链接，该工具可与领先模型媲美，专为商业用途设计，并重点关注内容安全。模型卡片和演示可在 [BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4) 查看。

- **EVA-CLIP-18B 树立 CLIP 模型新标准**：`@thejonasbrothers` 和 `@vrus0188` 讨论了 [EVA-CLIP-18B 论文](https://arxiv.org/abs/2402.04252)，该论文展示了一个拥有 180 亿参数的 CLIP 模型，具有突破性的 zero-shot top-1 准确率。这项工作标志着在保持数据集大小不变的情况下，CLIP 模型性能实现了重大飞跃。

- **用于受控图像生成的 InstanceDiffusion**：`@vrus0188` 重点介绍了 InstanceDiffusion，这是一种在 text-to-image 生成中实现精确实例级控制的方法，允许复杂的实例规格设定。随后 `@SegmentationFault` 就名为 "RPG-DiffusionMaster" 的类似功能进行了相关讨论，强调了功能上的差异。

- **寻求超参数见解**：用户 `@yoavhacohen` 寻求复现 SD XL VAE 的训练细节建议，`@drhead` 建议关注更大的 batch size、EMA 权重以及对 discriminator 和 reconstruction losses 的潜在调整。

- **关于数据存储和查询速度的讨论**：`@progamergov` 和 `@chad_in_the_house` 讨论了由 parquet 文件和相关图像组成的数据存储布局的效率，强调了查询速度对降低成本的重要性。

- **介绍防争议的 GOODY-2 LLM**：`@progamergov` 介绍了 GOODY-2，这是一个严格遵守伦理原则设计的 AI 模型，拒绝回答任何潜在的争议性问题。模型架构详见提供的 [model card](https://www.goody2.ai/goody2-modelcard.pdf)，该发布引发了 `@itali4no` 和 `@marianbasti` 等用户的幽默回应。

**提到的链接**：

- [GOODY-2 | 世界上最负责任的 AI 模型](https://www.goody2.ai/)：介绍具有下一代伦理对齐的新型 AI 模型。立即聊天。
- [EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters](https://arxiv.org/abs/2402.04252)：扩展对比语言-图像预训练 (CLIP) 对于增强视觉和多模态模型至关重要。我们展示了 EVA-CLIP-18B，这是最大且最强大的开源 CLIP 模型...
- [briaai/RMBG-1.4 · Hugging Face](https://huggingface.co/briaai/RMBG-1.4)：未找到描述
- [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/abs/2402.04236)：视觉语言模型 (VLMs) 通过将视觉指令与答案对齐的大规模训练，展示了广泛的可行性。然而，这种结论性的对齐导致模型...
- [Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models](https://arxiv.org/abs/2402.03749)：大型语言模型的最新进展引发了人们对其非凡且接近超人类能力的兴趣，促使研究人员探索评估和优化这些能力的方法...
- [V-IRL: Grounding Virtual Intelligence in Real Life](https://virl-platform.github.io/)：未找到描述
- [InstanceDiffusion: Instance-level Control for Image Generation](https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/)：未找到描述
- [ People @ EECS at UC Berkeley ](https://people.eecs.berkeley.edu/)：未找到描述

---

### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1204891293973741619) (20 条消息🔥): 

- **对 CUDA 构建的热情**：`@cropinky` 表达了对构建深度学习整机的兴奋，目前正在当地 eBay 列表上寻找二手 3090 GPU。
- **适用于深度学习的主板和 CPU 套装**：`@joseph_en` 分享了一个[链接](https://a.co/d/iR3bvvF)，包含 **Intel Core i9-12900K** 处理器和 **ASUS ROG Maximus Z690 Hero DDR5** ATX 主板，该组合适合运行双 GPU。
- **探索多 GPU 配置选项**：`@cropinky` 讨论了如何在一台 PC 中安装多个 GPU，提到了 **PCIe bifurcation cards**（PCIe 分叉卡）以及从 **CPU+motherboard** 组合中获得足够 PCIe 通道的必要性。他们还建议为这些讨论创建一个 `deep-learning-builds` 聊天频道。
- **转接线与构建**：`@jeremyhoward` 提到了使用矿机设置和转接线（risers）来容纳更多 GPU，而 `@iron_bound` 建议购买高质量的 **gen4 PCIe cables** 以确保最佳连接性，并建议仔细检查规格，例如 Cooler Master 的产品。
- **多 GPU 的挑战与资源共享**：`@cropinky` 和 `@joseph_en` 讨论了在系统中添加多个 GPU 时的物理限制和解决方案，包括利用 **GPU extenders**，以及可能通过改装运行超出主板标准 GPU 插槽数量的显卡。

**提到的链接**：

- [Amazon.com: Micro Center Intel Core i9-12900K Desktop Processor 16 (8P+8E) Cores up to 5.2 GHz Unlocked LGA1700 Desktop Processor with ASUS ROG Maximus Z690 Hero DDR5 ATX Gaming Motherboard : Electronics](https://a.co/d/iR3bvvF)：未找到描述
- [My deep learning rig &#8211; Non_Interactive &#8211; Software &amp; ML](https://nonint.com/2022/05/30/my-deep-learning-rig/)：未找到描述
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)：在此，我提供了针对深度学习/机器学习 GPU 的深入分析，并解释了适合您的使用场景和预算的最佳 GPU。
- [C-Payne PCB Design](https://c-payne.com/)：C-Payne PCB 设计
- [PCIe Bifurcation Card - x8x8 - 2W](https://c-payne.com/products/pcie-bifurcation-card-x8x8-2w)：一个 PCIe x16 输入，两个 PCIe x8 电气 / x16 物理输出。需要 PCIe gen3 BIOS 支持（请确认您的 BIOS 具有相应选项！）。40.64mm（双槽宽）间距插槽...
- [A Full Hardware Guide to Deep Learning &mdash; Tim Dettmers](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/)：在本指南中，我分析了从 CPU 到 SSD 的硬件及其对深度学习性能的影响，以便您可以选择真正需要的硬件。

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1205014257180090398) (6 条消息): 

- **PyTorch 2 论文入选 ASPLOS 2024**：`@lancerts` 分享了一篇[博客文章](https://pytorch.org/blog/pytorch-2-paper-tutorial/)，宣布 PyTorch 2 论文已被 ASPLOS 2024 接收，详细介绍了 **TorchDynamo**、**TorchInductor** 和 **Dynamic Shape support** 等技术。PyTorch 2 的巧妙特性（包括 **torch.compile**）将在会议的教程中进一步探讨。

- **PyTorch Conf 最后的插曲**：`@marksaroufim` 回忆了 PyTorch Conf 前的忙碌时刻，torch.compile API 及其文档在 Soumith 上台演讲前才刚刚完成合并。

- **基于 Python 的编译器的易用性**：`@marksaroufim` 指出 PyTorch 2 编译器完全由 Python 编写，这使得对探索其内部机制感兴趣的开发者来说，它更加易于访问和 Hack。

- **针对消费级硬件的优化**：`@marksaroufim` 对 PyTorch 2 针对消费级硬件进行优化的潜力表示兴奋，并指出此类改进已近在咫尺。

- **随 PyTorch 演进的调试工具**：`@marksaroufim` 赞赏了调试工具随时间的改进，**TORCH_LOGS** 的引入使得理解编译器内部工作原理变得更加容易。

- **Torch.compile 作为脚本编写的游戏规则改变者**：根据 `@lancerts` 的说法，与之前的方法相比，torch.compile 为模型迁移提供了一个不那么“玄学”的过程，对于那些寻找 torch script 替代方案以及模型转换为 C++ 的人来说，它是“救星”。

**提到的链接**：

[PyTorch 2 paper and tutorial @ ASPLOS 2024](https://pytorch.org/blog/pytorch-2-paper-tutorial/)：PyTorch 团队很高兴地宣布，我们关于 PyTorch 2 的论文已被 ACM 编程语言和操作系统架构支持国际会议（ASPLOS）接收并进行展示...

  

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1204834039421345872) (2 messages): 

- **torch.compile vs. torch.jit.trace**: `@lancerts` 询问 `torch.compile` 是否是 `torch.jit.trace` 的超集，以及是否应该默认使用 `torch.compile`。`@marksaroufim` 确认 **`torch.jit.trace` 和 `torch.jit.script` 已停止维护**，并建议在 Python 环境下的快速推理使用 `torch.compile`，而在非 Python 环境下则推荐使用 export + AOT inductor。
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1204728742249766963) (6 messages): 

- **初试 Fast Ai 和 Stable Diffusion**: `@einstein5744` 提到他们已经开始学习 Stable Diffusion Fast Ai 课程，并观看了第一部分的几节讲座。
- **寻找旧金山 Machine Learning 聚会**: `@.nike2k` 询问在旧金山是否有任何**线下聚会或活动**可以一起学习 ML。
- **潜在 ML 聚会的意向调查**: 针对 `.nike2k` 的询问，`@marksaroufim` 建议设置一个投票来衡量大家对线下学习活动的兴趣。
- **注意到与 GTC 重叠**: `@apaz` 观察到似乎与 **GTC** (GPU Technology Conference) 有显著的时间重叠。
- **寻找参考书答案**: `@bruno_58591` 询问在哪里可以找到某本参考书的答案，`@drisspg` 提供了一个 [Discord 链接](https://discord.com/channels/1189498204333543425/1194427148656721970/1195844532680523776) 作为回答。
  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1204725334180757514) (5 messages): 

- **JAX 的诞生**: `@jeremyhoward` 回忆说 **JAX** 最初是 Google 一个三人小组的小项目，但由于人们对 **TensorFlow 2** 的不满以及对 TF1 支持的匮乏，它迅速成为了首选。
- **JAX 的性能优势**: `@cropinky` 提到 JAX 的 **XLA JIT** 优化，根据两年前的一项对比，据称在 Google Cloud TPU 上的训练速度比 TensorFlow 快 **30%**。分享了一个 [视频](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P) 以阐述细节。
- **GPU 性能的等效性**: 继续对话，`@cropinky` 还分享了一位大学同事实验的经验证据，显示 JAX、TensorFlow 和 Torch 在 NVIDIA GPU 上的性能相似，尽管他们怀疑 JAX 在 TPU 上可能表现更出色。
- **TensorFlow 的困扰启发了 JAX？**: 他们幽默地推测，调试 TensorFlow 时的挫败感可能是 Google Brain 团队开发 JAX 的催化剂。
- **对全局变量的抱怨**: `@nshepperd` 批评了 TensorFlow 1.x 的易用性问题，特别是将模型参数作为全局变量处理的方式，这促使他们转向 PyTorch，并最终回归 JAX。

**提及的链接**:

[Day 1 Talks: JAX, Flax &amp; Transformers 🤗](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P): Day 1 Talks: JAX, Flax &amp; Transformers 🤗0:00:00 Skye Wanderman-Milne (Google Brain): Intro to JAX on Cloud TPUs0:42:49 Marc van Zee (Google Brain): Introduct...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1204715819515387904) (13 messages🔥): 

- **排查 LangChain ChatPromptTemplate 问题**: 用户 `@ebinbenben` 报告在使用 `from langchain.prompts.chat import ChatPromptTemplate` 时遇到 `ValueError`，具体错误信息为 "*ValueError: expected ':' after conversion specifier*"。该问题后没有进一步的解决方案或讨论。

- **建议将 Vercel AI SDK 作为一个选项**: 在对某项未指明事项的简短回复中，用户 `@r_i_c_o_` 建议 **vercel ai sdk** 可能是一个不错的选择，但未提供背景或后续信息。

- **寻找相关的提问验证器**: `@tepes_dracula` 询问是否存在一种分类器，可以在将查询和上下文输入到 Large Language Model (LLM) 之前验证其相关性，最好是在 *truthfulqa* 和 *FEVER* 数据集上训练过的。在提供的消息中没有针对此查询的回复。

- **LangChain 的最佳文档格式**: `@b0otable` 询问社区是否有人进行过实验，以确定 LangChain 文档加载器最有效的文档格式，并提到他们目前的工作流程是 Markdown 加载加上递归分块（recursive chunking）。该问题仍处于开放状态，无人参与讨论。

- **Async PlaywrightContextManager 问题**: `@vladmir_bc` 在将 **async PlaywrightContextManager** 与 *langchain* 配合使用时遇到了运行时警告，并在使用 *uvicorn* 部署到服务器时出错。作为回应，`@mintier` 建议将整个设置封装在 **Docker 容器**中以解决该问题。
  

---

### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1204855212095836171) (6 messages): 

- **LangChain 流式传输文档更新**：`@veryboldbagel` 分享了 LangChain 文档已更新，新增了关于 **带有事件的自定义流式传输 (custom streaming with events)** 和 **LLM 应用中的流式传输** 的章节。更新详细介绍了在 Agent 中使用 `where_cat_is_hiding` 和 `get_items` 等特殊工具的方法，并区分了 `stream`/`astream` 与 `astream_events`。查看 [更新后的文档](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events)。

- **寻求 LangChain 自定义参数的建议**：`@magick93` 正在尝试修改 `templates/extraction-openai-functions` 中的示例，以便使用 `WebBaseLoader` 传递 URL 参数。他们就如何在 LangChain 应用的服务端添加自定义参数寻求建议。
  
- **提供了有用的 LangChain 网络研讨会链接**：为了进一步说明其目标，`@magick93` 引用了一个 [YouTube 上的 LangChain 网络研讨会](https://www.youtube.com/watch?v=o7C9ld6Ln-M)，其中 Harrison Chase 解释了如何在客户端使用 URL 变量并让服务器的文档加载器对其进行处理（`在 31 分钟处`）。

**提到的链接**：

- [LangServe and LangChain Templates Webinar](https://www.youtube.com/watch?v=o7C9ld6Ln-M): 未找到描述
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events,): 流式传输是 LLM 应用中重要的 UX 考量因素，且 Agent 是...
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/streaming): streaming-with-langchain}

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1204802089641971823) (1 messages): 

- **LangChain Prompt 咨询**：`@aegean_thunder` 正在根据 [快速入门指南](https://python.langchain.com/docs/modules/model_io/prompts/quick_start) 寻求关于 **LangChain** prompt 的澄清。他们质疑为什么系统消息需要从 `langchain_core` 导入 `SystemMessage`，据称该库提供的是 *协议级包 (protocol level packages)*。
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1204699384160460800) (4 messages): 

- **简化 LangChain 表达式**：`@kartheekyakkala` 分享了他们关于 [LangChain 表达式语言 (LCEL)](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8) 的新博客文章，这是一种开发模型的 **声明式方法**，通过使用类管道 (pipe-like) 语法使编码更加容易。该博客在 **LangChain 框架** 的背景下解释了 LCEL，旨在使 **LLM 具备上下文感知能力**。
- **本地聊天机器人工具包和 LLM 资源更新**：`@discossi` 更新了一个 **轻量级 raq 聊天机器人** 的说明和 LLM 初学者资源。查看 [llama-cpp-chat-memory](https://github.com/ossirytk/llama-cpp-chat-memory) 和 [llm_resources](https://github.com/ossirytk/llm_resources) 获取更多信息。
- **揭秘生成式 AI**：数据科学家兼作者 `@mehulgupta7991` 宣传了他们的书 ***LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs***，该书已在 [Amazon](https://amzn.eu/d/g3UCuEw) 上架。该指南旨在帮助初学者构建生成式 AI 应用。

**提到的链接**：

- [LangChain Expression Language (LCEL)](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8): LangChain 表达式语言 (LCEL) 是一种使用 LangChain 框架开发模型的声明式方式。LCEL 简化了编码……
- [无标题](https://amzn.eu/d/g3UCuEw): 未找到描述
- [GitHub - ossirytk/llama-cpp-chat-memory: 带有 Chroma 向量库记忆的本地角色 AI 聊天机器人，以及一些用于 Chroma 处理文档的脚本](https://github.com/ossirytk/llama-cpp-chat-memory): 带有 Chroma 向量库记忆的本地角色 AI 聊天机器人，以及一些用于 Chroma 处理文档的脚本 - GitHub - ossirytk/llama-cpp-chat-memory...
- [GitHub - ossirytk/llm_resources: 关于本地运行大语言模型及其开发的所有相关信息和资源](https://github.com/ossirytk/llm_resources): 关于本地运行大语言模型及其开发的所有相关信息和资源 - GitHub - ossirytk/llm_resources...

  

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1204874871700455474) (7 条消息): 

- **书籍代码故障已通过提供协助解决**：用户 `@.pandamaui` 反映了 Mehulgupta 书中第一个代码示例的问题，发现默认模型已被弃用。`@mehulgupta7991` 承认了 LangChain 的快速发展和 OpenAI 的 API 变化，提供了支持，并指出即使是 O'Reilly 团队也在努力应对此类问题。
- **作者开通直接支持渠道**：针对 `@.pandamaui` 因 API 更新而遇到的书籍代码困难，`@mehulgupta7991` 通过电子邮件 datasciencepocket@gmail.com 提供了个人协助。
- **反馈助力未来修复**：`@mehulgupta7991` 对 `@.pandamaui` 指出问题表示感谢，并承诺在书籍的下一版中包含修复方案。
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1205060655623835688) (3 条消息): 

- **ChatGPT 的主题困境**：用户 `@joshcho_` 对 ChatGPT 的主题选项表示沮丧，将其描述为 **“亮白色或令人压抑的黑色”**。未讨论具体的主题调整解决方案或偏好。
  

---


### LLM Perf Enthusiasts AI ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/1205034985283653672) (1 条消息): 

- **对话结果的微调策略**：用户 `@naut1973` 询问了使用具有二元结果的对话数据集微调模型的最佳方法。他们质疑 **结果评分应该分布在每条消息中**，还是将对话作为一个整体考虑会更有效。
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1204800983264268338) (3 条消息): 

- **利用 AI 的创新服务**：`@res6969` 创建了一项服务，使用 VGT 对报告的 **章节进行分类**，**提取插图**，使用 **GPT-4V 对其进行描述**，然后嵌入文本以创建一个 **可搜索的数据库**。
- **运营成本令人震惊**：该创作者 `@res6969` 对 **高昂的运营成本估算** 感到惊讶，在其数据集上运行该服务的成本约为 **20,000 美元**。
- **对成本的一笑置之**：`@res6969` 用一个幽默的表情符号 **<:lmfao:556127422823530507>** 回应了高昂的价格估算，表示对这种情况的调侃或难以置信。
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1204775739120156732) (7 条消息): 

- **对新库收益的怀疑**：`@nosa_.` 对他们认为的新库炒作式演示提出了批评，怀疑其新颖性，并认为声称的 20 倍性能提升是“可笑的”。
- **对新创新的成本担忧**：`@res6969` 同意对新创新效用的怀疑，特别是考虑到成本上升以及处理密集或长输入时。
- **昂贵的提速技巧**：`@nosa_.` 幽默地建议，那些被称为性能增强的新方法实际上可能会导致 OpenAI 账单大幅增加，尽管承认在自托管时有一些用例。
- **GPT-4 的昂贵价格**：`@ajamjoom` 强调了使用 GPT-4 的巨大成本，引用了该服务每月“3 万美元”的数据。
  

---


### LLM Perf Enthusiasts AI ▷ #[cost](https://discord.com/channels/1168579740391710851/1169026016887459961/1204892384605773836) (2 条消息): 

- **FastChat/VLLM 托管咨询**：用户 `@natureplayer` 询问了托管 **FastChat/VLLM** 的经验。`@jmak` 简短地回答道：“不，还没有。”
  

---

### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1204880866313240577) (7 messages): 

- **发现 OpenAI 模型的成本效益**：`@dbreunig` 评论了 OpenAI 的 `text-embedding-3-small` 的高性价比，指出其成本效益极高，以至于在某些功能上几乎没有理由使用其他产品。
- **探讨 OpenAI 的定价策略**：`@simonw` 评论了 OpenAI 定价背后潜在的竞争策略，认为其设计初衷似乎是为了限制竞争。
- **对 OpenAI 竞争性定价的共识**：`@dbreunig` 同意 `@simonw` 关于 OpenAI 激进定价结构的看法。
- **构建竞争平台的挑战**：`@simonw` 提到了由于 OpenAI 的激进定价，竞争对手在建立平台时面临的困难。
- **差异化需要定价之外的创新**：`@dbreunig` 指出，为了从 OpenAI 的产品中脱颖而出，竞争对手需要提供除了 UI/UX 改进之外，经过专门调优或修改的产品。

**提到的链接**：

[Drew Breunig (@dbreunig@note.computer)](https://note.computer/@dbreunig/111891995849758288)：内附：1 个视频。Emoji 建议功能现在可以在 StepList 中使用了。这是一些安静的 AI UX，它通过为你的新列表标题生成 embedding，并将其与经过批准的 emoji embedding 数据库进行比较来工作...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1204707223432069130) (3 messages): 

- **模型和 LoRA 的内存需求**：用户 `devnull0` 表示，需要足够的内存来同时容纳模型和 **LoRA** 权重。
- **在有限资源下适配更大型号的模型**：`johannhartmann` 分享了一个技巧，即 **Llama_factory** 现在支持用于 **Mistral** 的 *unsloth*。当与 *qlora* 结合使用时，它可以让 wiedervereinigung-7b 模型适配特定的内存限制。
- **Mistral 语言和数据集支持扩展**：`johannhartmann` 还提到 Llama_factory 已经集成了对 **9 个德语 SFT** 和一个 **DPO 数据集** 的支持。