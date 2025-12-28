---
companies:
- openai
- hugging-face
- nous-research
- eleutherai
- hazyresearch
date: '2024-05-13T23:14:50.739179Z'
description: '**OpenAI** 发布了 **GPT-4o**，这是一款支持**音频、视觉和文本**实时推理的前沿模型。该模型现已向所有 ChatGPT
  用户免费开放，不仅增强了编程能力，还即将推出高级语音和视频功能。


  相关讨论涵盖了 **Llama 3** 等**开源大语言模型**、包括 **GPT-3.5** 知识蒸馏在内的微调技术，以及量化等硬件优化策略。新兴架构包括与
  ChatGPT 语音和 Open Interpreter API 的多模态集成、结合了自回归和扩散方法的混合专家模型（MoE），以及旨在提高 GPU 效率的 **YOCO
  架构**和 **ThunderKittens DSL** 等新颖设计。此外，研究还强调了高效注意力机制方面的进展（如使用快速傅里叶变换 FFT 的 **Conv-Basis**）以及**深度上采样（depth
  upscaling）**等模型缩放技术。'
id: 371f7c57-ad0f-4260-852c-258384391dc6
models:
- gpt-4o
- gpt-3.5
- llama-3
original_slug: ainews-gpt-4o-the-new-sota-everything-frontier-9515
people: []
title: GPT-4o：全新的全能型 SOTA 前沿模型（GPT-4 Turbo 版本）
topics:
- real-time-reasoning
- coding-capabilities
- fine-tuning
- knowledge-distillation
- hardware-optimization
- quantization
- multimodality
- mixture-of-experts
- efficient-attention
- model-scaling
- depth-upscaling
- transformer-architecture
- gpu-optimization
- prompt-engineering
---

<!-- buttondown-editor-mode: plaintext -->**Omnimodality 就是你想要的一切。**

> AI News (2024/5/10-2024/5/13)。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**426** 个频道，以及 **7769** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**763 分钟**。

按照 AINews 在 Frontier Model 发布日的传统，我们将发布两个版本的 AINews。**您目前阅读的版本中，所有 Part 1 和 Part 2 的总结均由 GPT4T 完成** —— [上一封邮件是由 GPT4O 完成的，并包含常规评论](https://buttondown.email/ainews/archive/ainews-gpt-4o-the-new-sota-everything-frontier/)。我们设想您可以将它们并排对比，查看您关注的 Discord 社区，从而更好地理解其改进或退步。


---

**目录**

[TOC] 


---

# AI Discord 回顾

> 总结之总结的总结

## Claude 3 Sonnet

**1. GPT-4o 的发布与功能**

- **[GPT-4o](https://openai.com/index/hello-gpt-4o/)** 是 OpenAI 最新发布的**前沿模型 (frontier model)**，支持跨**音频、视觉和文本**的实时推理。它在保持 GPT-4 智能水平的同时，显著提升了性能。

- GPT-4o 现已向所有 ChatGPT 用户（包括免费计划）**免费开放**，这标志着 OpenAI 策略的转变，旨在让强大的 AI 工具触手可及。[阅读更多](https://x.com/sama/status/1790065541262032904)

- 讨论重点强调了 GPT-4o 在**编程能力 (coding capabilities)** 方面的实质性增强，并期待通过 MATH 等新基准测试来量化这些进步。[博客文章](https://openai.com/index/hello-gpt-4o/)

- **Plus 用户**将获得高达 **5 倍的额度限制**，并能最早体验即将推出的功能，如全新的 macOS 桌面应用以及先进的**语音和视频功能**。[公告](https://discord.com/channels/974519864045756446/977259063052234752/1239631044395929685)

**2. 开源 LLM 探索与微调技术**

- 关于探索类似于 Llama 3 的**开源 LLM** 的广泛讨论，建议尝试 **you.com** 等平台。[HuggingFace 讨论](https://discord.com/channels/879548962464493619/879548962464493622/1238758307267874906)

- 成员们寻求关于**微调技术 (fine-tuning techniques)**（如知识蒸馏）的指导，以提高 **GPT-3.5** 等模型的准确性和性能。[HuggingFace 博客](https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt)

- 在本地运行 LLM 的兴趣引发了关于如何应对**硬件限制**的对话，并提出了关于卸载技术 (offloading) 和模型量化 (quantizing) 以获得更好性能的建议。[LM Studio 讨论](https://discord.com/channels/1110598183144399058/1153759714082033735/1238773262884798565)

- 探索了处理**多主题对话**等复杂任务的技术，范围涵盖从在专门数据集上进行微调，到利用提示工程 (prompt engineering) 开发 Elaborator 模型。[Unsloth AI 讨论](https://discord.com/channels/1179035537009545276/1179777624986357780/1238775563502751755)

**3. 多模态 AI 与新兴架构**

- 业界对 **ChatGPT 语音对话 AI** 与 Open Interpreter API 的集成充满期待，这将实现多模态交互。[OpenInterpreter 讨论](https://discord.com/channels/1146610656779440188/1147665339266650133/1238756318999740507)

- 讨论了利用混合专家模型 (MoE) 架构集成**自回归和扩散模型 (autoregressive and diffusion models)** 的潜力，旨在增强多模态模型的性能。[Nous Research AI 讨论](https://discord.com/channels/1053877538025386074/1154120232051408927/1238877292395102268)

- 介绍了 **YOCO 架构**，这是一种 decoder-decoder 模型，能够高效缓存键值对 (key-value pairs)，在保持全局注意力 (global attention) 能力的同时减少 GPU 显存需求。[HuggingFace 阅读小组](https://discord.com/channels/879548962464493619/1156269946427428974/1239543496457584752)

- 探索了 **ThunderKittens**，这是来自 HazyResearch 的一种新 DSL，旨在简化 AI 内核构建并优化 GPU 利用率，以提高计算效率。[CUDA MODE 讨论](https://discord.com/channels/1189498204333543425/1189607726595194971/1239314338708197458)

**4. 高效注意力机制与模型缩放的进展**

- 研究了一种名为 **Conv-Basis** 的高效方法，该方法利用卷积矩阵计算注意力，并结合快速傅里叶变换 (FFT) 潜在地减少计算时间。[Eleuther 研究讨论](https://discord.com/channels/729741769192767510/747850033994662000/1238750605707841569)

- 深入探讨了**深度上采样 (depth upscaling)** 技术（如层重复）以提高模型性能，并参考了 Yi 和 Granite Code 模型的研究工作。[Eleuther 研究讨论](https://discord.com/channels/729741769192767510/747850033994662000/1238750605707841569)

- 讨论了**线性注意力模型 (Linear Attention models)** 在 MMLU 等复杂评估中的表现，强调需要合适的数据来发挥模型改进的潜力。[Eleuther 研究讨论](https://discord.com/channels/729741769192767510/747850033994662000/1238750605707841569)

- 介绍了一项名为 **Farzi** 的提案，用于将密集数据集综合为紧凑且高效的序列，以训练自回归模型，其性能可达原始数据的 120%。[OpenReview 详情](https://openreview.net/forum?id=H9DYMIpz9c&noteId=aN4DeBSr82)

## Claude 3 Opus

- **GPT-4o 发布，具备多模态能力**：OpenAI 推出了 **GPT-4o**，这是一个支持文本、音频和图像输入的新前沿模型，[为 Plus 用户提供 5 倍更高的限制](https://discord.com/channels/974519864045756446/977259063052234752/1239631044395929685)。它展示了强大的 [Coding 和推理性能](https://discord.com/channels/1179127597926469703/1179128538679488533/1238819707457372161)，并且 [对所有 ChatGPT 用户免费开放](https://discord.com/channels/822583790773862470/1075282825051385876/1238778471841533992)。此外，还讨论了更新的 [tokenizer](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8) 以及潜在的 [Apple 集成](https://www.bloomberg.com/news/articles/2024-05-11/apple-closes-in-on-deal-with-openai-to-put-chatgpt-on-iphone)。

- **Llama 3 微调进展**：社区探索了 **Llama 3** 模型微调，重点关注 [与量化模型的兼容性问题](https://discord.com/channels/1179035537009545276/1179777624986357780/1238775563502751755)、[tokenization 挑战](https://discord.com/channels/1179035537009545276/1179777624986357780/1238775563502751755) 以及复杂的对话能力。[Unsloth](https://github.com/unslothai/unsloth) 成为实现更快微调且占用更少内存的关键工具。此外，还分享了用于 [token classification](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188) 的微调 Llama 变体。

- **Kernel Fusion 与 CUDA 优化技术**：CUDA MODE 举办了一场 [关于 Kernel Fusion 经验的 Zoom 会议](https://discord.com/channels/1189498204333543425/1189640399476764692/1238927701281210369)，并讨论了用于 [AI Kernel 优化](https://discord.com/channels/1189498204333543425/1189607595451895918/1238762803138007051) 的 **Triton**。重点介绍了 **U Illinois PMPP** 关于并行编程的 [YouTube 讲座系列](https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=shared)。同时探索了 **llm.c** 中用于 [提高内存效率的 ZeRO-1](https://github.com/karpathy/llm.c/pull/309) 技术，以及用于 [提升 GPU 利用率的 ThunderKittens](https://github.com/HazyResearch/ThunderKittens)。

- **检索增强生成 (RAG) 与多模态 AI**：使用 **LangChain** 和 **LlamaIndex** 的 RAG 流水线在 [博客聊天机器人](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog)、[内容审核](https://discord.com/channels/1059199217496772688/1187460979064324127/1238888807072403516) 和 [PowerPoint 生成](https://discord.com/channels/1059199217496772688/1187460979064324127/1238888807072403516) 方面引起了关注。讨论了 [使用 DinoV2 的多模态 AI 技术](https://www.youtube.com/watch?v=KQ-xGVFHDkw) 以及 [OpenAI 的音视频集成](https://discord.com/channels/1179127597926469703/1179128538679488533/1238819707457372161)。**Perplexity AI** 引入了 [多模型策略](https://discord.com/channels/1047197230748151888/1047649527299055688/1238757309254078475)，而 **OpenInterpreter** 实现了 [LiteLLM 和 Llama 3 的集成](https://discord.com/channels/1146610656779440188/1194880263122075688/1238960200678113343)。

## GPT4T (gpt-4-turbo-2024-04-09)

**主要主题与讨论：**

1. **AI 模型讨论与比较：** 在多个 Discord 频道中，观察到大量关于各种 AI 模型（如 GPT-4, GPT-4o, Llama 模型等）性能和规格的讨论。用户对模型性能表达了复杂的情绪，讨论重点集中在模型训练、新旧版本比较以及系统集成等任务。

2. **技术创新与更新：** 多个频道报道了关于新功能、集成和技术进步的更新，例如多模态能力、tokenizer 的变化以及速度提升。来自科技巨头和社区程序员的更新受到了广泛的评估与剖析。

3. **社区参与与项目协作：** 围绕社区参与协作项目、贡献开源仓库或分享自定义项目的讨论非常热烈。这些参与涵盖了编码实践、开发 AI 工具或解决复杂的 AI 驱动任务。

4. **教育内容与教程：** 分享了大量旨在传播 AI 技术、编程、模型训练等知识的教育内容、教程和讨论。学术论文链接、YouTube 视频和详细的博客文章非常常见，用户借此加深理解或向同行解释相关概念。

5. **隐私、法律与伦理关注：** 多项讨论涉及使用 AI 技术的隐私影响、数据使用的担忧、AI 生成内容的法律影响以及伦理考量。法律讨论尤其涵盖了从生成内容中的艺术家权利到 AI 在现有法律框架中的影响等一系列话题。

**关键知识共享与资源：**

- 指向论文的教育链接，以及 YouTube 和 GitHub 等平台上的教程。
- 关于主要 AI 模型和软件工具更新的讨论。
- 由共享代码仓库和开发工具支撑的社区驱动指南和项目协作。
- 在 AI 进步背景下审议的伦理、法律和隐私问题。

## GPT4O (gpt-4o-2024-05-13)

**1. 模型性能与发布**

- **GPT-4o vs GPT-4** 的性能在多个 Discord 频道中进行了对比，GPT-4o 因其速度受到赞赏，但在推理能力方面受到了审视。OpenAI 已将 GPT-4o 免费开放，引发了关于其市场影响的讨论。[来源](https://openai.com/index/hello-gpt-4o/)
- **Falcon 2** 和 **Llama 3** 因其新特性和改进的性能受到了极大关注。[Falcon 的能力](https://falconllm.tii.ae/falcon-2.html) 因超越竞争对手而受到特别讨论。

**2. 技术挑战与解决方案**

- **Quantum vs. Turing**：关于量子计算机优于 Turing 模型的辩论突显了对监管有利于大公司的担忧。讨论延伸到了 Llama 和 Mistral 等模型的训练和操纵。
- **错误处理**：模型集成和执行中的频繁问题，包括 GGUF 模型的 Tokenization 挑战以及 Tinygrad 中训练错误的排查，已通过社区建议和详细修复得到解决。[GitHub PR 示例](https://github.com/tinygrad/tinygrad/pull/4460/files)
- **内存管理**：关于优化 GPU 内存管理和处理 VRAM 限制的讨论非常显著，特别是在 CUDA 和 Mojo 环境中，包括 Offloading 和 Quantization 等策略。

**3. AI 集成与增强**

- **多模态模型 (Multimodal Models)**：关于在 GPT-4o 等模型中集成音频、视频和文本的公开讨论。采用 **ThunderKittens** 等工具来优化 Kernel 操作，展示了对增强性能的持续追求。[ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)
- **开源与社区项目**：分享了 **PyWinAssistant** 和 **LM Studio** 的模型管理 CLI 工具等项目，强调了 AI 社区的协作精神。[PyWinAssistant GitHub](https://github.com/a-real-ai/pywinassistant)

**4. 行业趋势与活动**

- **OpenAI 的战略举措**：围绕 OpenAI 免费开放 GPT-4o 的战略方向的推测被广泛讨论，表明了潜在的数据驱动战略或竞争性市场定位。[OpenAI 活动视频](https://www.youtube.com/watch?v=DQacCB9tDaw)

**5. 伦理与法律问题**

- **AI 与版权问题**：关于 AI 生成内容可能侵犯艺术家权利的辩论非常突出，对于此类使用是否属于合理使用（Fair Use）意见不一。这延伸到了关于 AI 在商业艺术中的地位以及涉及的法律边界的讨论。[相关文章](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
  
**6. 教育与支持资源**

- **协作学习**：频道通过共享资源、教程和故障排除协助提供指导，形成了一个强大的社区驱动支持系统。主题包括 Fine-tuning 方法以及控制理论和 Stable Diffusion 局部重绘 (Inpainting) 等实际 AI 应用。

---

### 各频道详细摘要与链接：

**Unsloth AI (Daniel Han) ▷ [General](https://discord.com/channels/1179035537009545276/1179035537529643040/1238761392711012402)**
- **Quantum 与 Turing 的有效性对比**：辩论强调了 Turing 在预期的 Quantum 领域表现优于 Quantum。[Rethinking Machine Unlearning](https://arxiv.org/abs/2402.08787)
- **对 OpenAI 监管举措的担忧**：GPU 签名以及与白宫的排他性协议引发了社区的质疑。[OpenAI Reddit AMA](https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_with_openais_ceo_sam/)
- **模型训练伦理与安全**：未审查模型的伦理影响被类比为刀具监管，重点在于滥用而非工具本身。广泛探讨了 Fine-tuning 的技术方法。

**Stability.ai (Stable Diffusion) ▷ [General-Chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1238754156731437087)**
- **对 Stable Diffusion 3 的疑虑**：围绕难以捉摸的发布日期的怀疑与幽默。
- **ControlNet 与 LoRA 的使用**：关于 Inpainting 和图像中文本集成等高级技术用途的讨论。[Character Consistency Guide](https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/)

**OpenAI ▷ [General Discussions](https://discord.com/channels/974519864045756446/998381918976479273/1238835920703193191)**
- **GPT-4 与 GPT-4o 的性能对比**：用户积极测试并对比输出，引用了速度与推理质量的权衡。[OpenAI Event](https://www.youtube.com/watch?v=DQacCB9tDaw)
- **订阅价值辩论**：讨论集中在新的免费层级功能背景下，付费计划的持续价值。

**LangChain AI ▷ [General](https://discord.com/channels/1038097195422978059/1038097196224086148/1238794899726405662)**
- **日期提取技术**：通过分享的代码片段，在 LangChain 模型中处理日期范围并转换为 ISO 格式。
- **处理 Tool 或 Function Calls**：在 LangChain 中使用 LLM 提取和处理多个市场描述。[Example GitHub](https://gist.github.com/mattcollins/62fcb8d15a001d5b4e5c9fb86aad4f8e)

**OpenRouter (Alex Atallah) ▷ [General](https://discord.com/channels/1091220969173028894/1094454198688546826/1238747703710584863)**
- **模型服务中断**：Jet MoE 8B 经历上游过载。GPT-4o 和 LLaVA v1.6 等新型 Multimodal 模型表现突出。[OpenRouter API Watcher Tool](https://orw.karleo.net/)

**HuggingFace ▷ [General](https://discord.com/channels/879548962464493619/879548962464493622/1238758307267874906)**
- **开源 LLM 探索**：讨论包括优化 Mistral 等 LLM 的平台和方法。关于对转录文本进行 Chunking 以更好地生成见解的查询。[Diffusion Pipeline Guide](https://huggingface.co/blog/stable_diffusion)
  
有关更多详细摘要和链接，请参阅上方完整的汇编指南。

---



# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **量子长寿 vs. 图灵威力**：量子计算机与图灵计算机引发了激烈辩论，目前图灵计算机在各领域（甚至包括量子计算机预期领先的领域）仍占据上风。一些成员对政府意外关注监管量子计算表示担忧，认为这给了大公司相对于小型创新者不公平的优势。
- **OpenAI 的变革之风**：用户对 OpenAI 转向排他性和监管俘获（regulatory capture）表示怀疑，担心诸如 GPU 签名以及与白宫合作等行为可能会抑制开放竞争和创新。
- **审查 vs. 滥用**：社区内关于未经审查的 AI 模型的潜在危险和伦理影响讨论热烈。一个流行的类比是将 AI 模型控制与刀具监管进行比较，强调重点应放在滥用行为上，而非工具本身。
- **模型训练狂热**：技术宅警报！一场关于多样化模型训练和操作方法的深度技术交流席卷了聊天室，涵盖了使用未经审查的 LLM 以及操纵模型在未经授权的情况下接受新适配（adaptations）等策略。
- **共情能力获赞**：开放共情项目正风靡一时，呼吁社区参与以丰富 AI 在更广泛人类语境下的理解和应用。
- **屏息以待 OpenAI 能力的量子飞跃**：新 Model Spec 的发布以及与 OpenAI CEO Sam Altman 的社区 Reddit Q&A 环节让成员们充满期待。人们对革命性 AI 突破的希望与对潜在失望的担忧交织在一起，情绪高涨。
- **OpenAI 准星下的开源策略**：社区在 OpenAI 是否应该开源其模型的问题上存在分歧。支持发布的人认为，即使是一个表现平平的模型，也能让他们处于有利地位。
- **AI 领域的行业趋势演变**：聊天室里充满了对行业趋势的推测，例如是否可以期待一个比现有参与者强 10 倍的模型。如果 Llama 成为 "SOTA"（State Of The Art），市场动态可能会发生转变，这也备受关注。
- **AI 寒冬传闻中的 OpenAI**：尽管 AI 寒冬的阴影笼罩，成员们仍坚定相信 OpenAI 在 AI 行业的领导地位。关于 OpenAI 决定公开发布模型的战略原因也是讨论焦点，包括对过去泄密事件以及由于资助要求而必须保持开放的见解。
- **量化模型之谜与分词异常的破解**：资深用户分享了管理量化模型与 TGI 兼容性问题以及保存/加载错误的经验，特别是通过 `model.save_pretrained_merged(...)` 使用 '16bit' 格式使其与 TGI 兼容。还讨论了涉及 **Gemma** 的 GGUF 格式模型的分词（Tokenization）问题。
- **寻求终极模型**：社区渴望获得关于创建能有效处理复杂、多话题对话模型的指导。提出的策略涵盖了从在专业数据集上进行 Fine-tuning 到采用 Prompt Engineering 或构建 Elaborator 模型，揭示了在聊天机器人框架中优化模型的迭代历程。
- **技术用户展示炉边模型**：用户向社区分享了 Fine-tuning 后的 Llama 变体。随附的一篇博客文章和一个详细介绍模型 Fine-tuning 过程的 Notebook 将作为给技术读者的福利。[查看模型库](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188)。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 的神话**：社区中关于 Stable Diffusion 3 发布的消息充满了笑话和怀疑的 GIF。尽管官方公布了时间表，但 SD3 的发布仍然是一个令人困惑的话题，在许多用户眼中，它已经带上了一种奇幻色彩。
- **ControlNet 成为讨论焦点**：围绕 ControlNet 和 LoRA 技术的应用展开了技术讨论，特别是针对 Inpainting 和在图像中集成真实文本等独特任务。一个突出的建议是使用 Krita 这一非传统工具来手动调整图像中的文本。
- **硬件丛林的大乱斗**：一场反复的讨论评估了 AMD RX 6750 XT 和 NVIDIA RTX 4090 运行 Stable Diffusion 的效率，最终对旧款与高端 GPU 在 SD 任务中的性能对比产生了多种看法。
- **Stable Diffusion 进军麦迪逊大道**：用户强调了 Stable Diffusion 的潜在商业应用，例如生成定制的产品广告。一位用户表达了在多张图像中保持角色一致性的必要性，并指向 [Cobalt Explorer 的指南](https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/) 作为详细指导来源。
- **寻求帮助**：一般性查询和技术支持请求完善了讨论，用户解决了从 ComfyUI 等界面上的复制/粘贴问题，到探索为图像注入更多细节的 Upscaling 方法等各种问题。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 公开发布，Plus 用户获益**：OpenAI 宣布其新的旗舰模型 **GPT-4o** 现在可以免费使用，但有一些限制。Plus 用户将获得更大的优势，包括高达 **5 倍的额度限制**，以及最早访问新功能的权限，如新的 macOS 桌面应用和先进的语音及视频功能。

- **GPT-4o 对比 GPT-4：性能充沛**：OpenAI 用户正在积极比较 GPT-4 和新发布的 GPT-4o 在不同任务中的表现。GPT-4o 拥有更快的速度，但需要更明确的指令才能获得最佳性能。用户对新的语音和实时摄像头共享功能也抱有高度期待。

- **Mac 版已就绪，Windows 版紧随其后**：用户对即将发布的 ChatGPT macOS 应用表现出了极大的热情。有报道称 Windows 版本正在开发中，但目前尚未对所有用户开放。

- **进步中的 Token 烦恼与记忆疑虑**：在所有进步中，人们对 GPT-4o 与旧模型相比的 Memory 性能表示担忧，并要求改进 Token 计数器等功能。随着免费版加入新功能，Plus 用户正在权衡其订阅的持续价值。

- **浪漫无处安放**：**Gemini 1.5** 出现了一个问题，即任何与浪漫相关的请求都会持续失败。详细的调试未能提供解决方案，导致人们猜测是语法错误、安全设置，甚至是 Google 的系统角色导致了该问题。

- **Python 文件处理变得简单**：一位用户分享了一个复杂但基础的 Python 任务，用于创建目录、管理跨会话的文件写入，并将带有下载链接的目录打包成 ZIP。该帖子突出了社区所应对的技术复杂性和挑战的多样性。

- **创建 ChatGPT 克隆版 —— 一个警惕的视角**：一位用户表达了创建 ChatGPT 克隆版的兴趣，并以 GPT-3.5 作为底层模型。该提案的独特之处在于赋予克隆版监督组织内发送和接收的消息的能力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 在 8k 以上长度难以保持连贯性**: King.of.kings_ 分享了让 **Llama 3 70b** 模型在超过 8k tokens 后保持连贯性的困难，引发了社区的讨论和可能的解决方案。
  
- **极光观测、新的双语模型以及游戏中的古老食谱**:
  - 北极光在法国 Arvenia 城市火山罕见出现，引发了兴趣和讨论。
  - **MAP-Neo** 的引入引起了工程师们的关注，这是一个透明的双语 Large Language Model。它在 4.5 万亿 tokens 上进行训练，承诺在推理和数学等任务中达到商业模型的性能，同时具有更高的透明度。 
  - 成员们参与了一个有趣的消遣，讨论了角色扮演游戏 *Kingdom Come: Deliverance* 中出现的“永恒炖菜”（perpetual stews）如何反映历史烹饪方法并影响现代烹饪习惯。

- **神经科学进展、Taskmaster 模拟以及工业军事复合体可视化**:
  - `interesting-links` 频道讨论了一篇关于多向人工神经网络的新论文，该论文有可能彻底改变网络处理复杂依赖关系的方式。
  - 一个 React 应用程序使用 State Machine 模式模拟了 **Taskmaster** 游戏节目，并在 LLMs 的辅助下创建了引人入胜的内容。  
  - 在 llama-cpp-agent 框架上使用 **Mistral 7B instruct v 0.2** 模型，生成了一个详细的知识图谱，以从未见过的方式可视化了工业军事复合体（Industrial Military Complex）。

- **GPT-4o：强力更新还是过度炒作？**: 在 `general` 频道中，成员们热烈辩论了 GPT-4o 的优缺点。一些成员赞赏其在代码性能方面的改进，而另一些成员则批评其速度和 token 输出限制。大家在语音集成功能的易用性和价格点上产生了分歧。

- **MoE 中的专家是 FFN，Llama 喜爱 Axolotl**:
  - 在 MoE 架构中，专家（experts）通常仅指前馈网络（FFN）层。
  - 将自回归模型（autoregressive models）和扩散模型（diffusion models）与 MoE 集成的潜力引起了参与者的兴趣；虽然有人表示怀疑，但可能性似乎令人兴奋。
  - 一位用户分享了使用 Axolotl 系统配合 dolphin-2.9 数据集微调 Llama3 模型时遇到的问题及解决方案。

- **关于数据集和训练方法的对话**:
  - **ChatQA** 因其对话式问答模型系列而成为头条新闻，该系列在对话准确性上超越了 GPT-4。
  - IBM 和 RedHat 提出了一种新的 LLM 训练方法，因其使用较大的模型生成合成数据集而无需进行完整的重新训练而广为流传。
  - 对 IBM/RedHat 新项目的深入了解揭示了一个预定的信息处理流程，用于增强 LLMs 的知识库，从而激发了社区的兴趣。

- **WorldSim 的冒险尝试**: 在 `world-sim` 频道中，WorldSim 被强调为一个强大的商业模拟器，并分享了加入 Websim AI 模拟的邀请。有人提议组建聊天小组，围绕 WorldSim 进行哲学讨论。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 春季发布会预热活动**：[Discord 频道](https://discord.gg/Z7V4NDGZ?event=1238918257046458368)已安排了一场针对 5 月 13 日上午 9:30 OpenAI 活动的预热观影会。欢迎提前入场参加庆祝活动。
- **向东看，讨论未来 AI 基础设施**：成员们正围绕一位来自新加坡的成员发起的关于潜在新 AI 基础设施的新对话展开讨论。他们已开始在 [Substack](https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9) 上汇总想法，如果你对这些创新服务感兴趣，欢迎加入。
- **Falcon 2 模型在 LLM 领域展翅高飞**：介绍 **Falcon 2 LLM**，据称这是一款多语言、多模态的杰作，其表现优于 Meta 和 Google 等公司的模型。它目前仍在进行进一步增强，包括引入 'Mixture of Experts'。在此探索其强大功能 [here](https://falconllm.tii.ae/falcon-2.html)。
- **GPT-4o 揭开面纱供你检阅**：欢迎 **GPT-4o**！我们正在这场深度讨论中汇集关于其规格、用途、API 以及整体性能的集体智慧。你可以在此加入对话 [here](https://openai.com/index/hello-gpt-4o/)。俗话说好奇害死猫，但它可能会让 AI 工程师乐在其中。
- **AI 安全：一条值得投入的职业道路吗？**：在 AI 与网络安全的交汇点开启职业生涯是你的菜吗？我们的成员正在辩论其潜力，并提供进一步探索的途径，如 RSA Conference。泡杯咖啡，加入讨论吧。
- **召集好友共同参与 OpenELM**：一个使用 PyTorch/MPS 训练 **OpenELM 模型**的进行中项目正在寻求更多智慧支持。目标是通过增量数据集添加进行迭代训练。在此参与这项开源冒险 [here](https://github.com/openai/openelm)。毕竟，分享就是关爱。
- **OpenAI 活动遭遇音频故障**：造化弄人，OpenAI 活动观影会在直播过程中遇到了一些音频问题的波折。没有一点小插曲，观影会就不完整。
- **Apple 与 GPT-4o：一个被“苹果”驯化的未来？**：Apple 的技术战略是否足够强大，能够将像 GPT-4o 这样更强劲的模型集成到他们的设备中？用一些关于“苹果酒”的思考来结束这段精彩的对话吧。
- **OpenAI 打破传统，提供免费 GPT-4o 访问**：用户现在可以[免费使用 GPT-4o](https://x.com/LiamFedus/status/1790064963966370209)，这标志着 OpenAI 使命的新阶段。这一巨大飞跃不仅将 GPT-4o 集成到日常设备和平台中，还引发了严谨的讨论。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-4o 备受期待的到来**：关于 GPT-4o 推出的讨论正在升温，人们对其更快的处理速度、更低的成本和广泛的应用范围寄予厚望。热情的用户对其在 Perplexity 平台的潜在集成持乐观态度，并推测其在 AI 应用中功能的增强。

- **释放强大模型：用户的诉求**：用户对 Perplexity 在 Claude 3 Opus 等强大模型上的每日使用限制表示不满，指出对延长访问权限的巨大需求。虽然一些用户正在寻找替代方案，但许多人由于其独特的功能仍坚持使用 Perplexity。

- **AI 采用与隐私的结合**：在 AI 服务的导航和选择过程中，讨论强调了用户对重视隐私的平台的高度重视。尽管基于云的 AI 存在固有的隐私挑战，成员们仍支持那些在保护用户数据方面做出实质性努力的供应商。

- **Perplexity 的多模型策略与用户赞赏**：强调了 Perplexity 多模型方法的优势，允许用户根据任务需求在 ChatGPT 和 Claude 3 Opus 等不同模型之间切换。这种灵活性受到了赞赏，使其区别于选项有限或导航更复杂的平台。

- **技术讨论反映了用户的多样性与需求**：围绕上下文窗口大小（context window sizes）和 AI 模型详细工作原理等主题的技术对话表明，社区内的 AI 使用范围广泛。查询范围从关于每日限制的日常询问到对特定 AI 功能的深入探索。

- **AI 职业路径的复杂性备受关注**：Alexandr Yarats 概述了他从 Yandex 到 Google 的职业历程，以及他目前担任 Perplexity AI 搜索负责人的角色。他的叙述强调了科技行业职业生涯的严峻挑战和回报，重点是创建 AI 驱动的搜索引擎。

- **Perplexity AI 上的各类搜索**：用户分享了在 Perplexity AI 上进行的各种搜索，从 2024 年欧洲歌唱大赛到解释伯努利谬误，突显了可以从该平台获取的信息范围之广。

- **鼓励可共享线程以促进协作**：Perplexity AI 强调了对可共享线程的需求，并通过 Discord 消息提供了指南，强化了社区协作和信息共享的价值。

- **对 Perplexity 教程的请求遇到了失效链接**：对 Perplexity 教程的请求引导另一位用户提供了一个教程链接。然而，该链接重定向到了一个失效的 Discord 路径。

- **非英语对话中的表情符号使用**：在看似俄语的对话中观察到用户使用了名为 'wlcm' 和 'gem_2' 的表情符号，暗示了语境差异或情感表达。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **探索开源 LLM**：讨论集中在探索类似于 **llamma3** 的开源大语言模型（LLMs）。有人建议 **you.com** 等平台可能是进行实验的有趣切入点。
- **会议纪要分块的困境**：目前为了从 LLM 获取可操作见解而对会议纪要进行分块的方法产生的相似度得分较低。社区受邀提出改进此过程的方法，从而通过减少 LLM 调用次数来优化成本。
- **深入了解 Diffusers**：成员们有兴趣了解更多关于扩散模型（diffusion models）的细节，引用的资源包括热门的学术论文以及来自 [Fast.ai](https://course.fast.ai/Lessons/part2.html) 和 [O'Reilly](https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/) 的实用教程。
- **启用 Stable Diffusion**：参与者分享了他们的 Stable Diffusion 进展，并就如何基于 Hugging Face 的 [diffusers library](https://huggingface.co/blog/stable_diffusion) 开发带有 StableDiffusionPipeline 的本地推理引擎提供了专业指导。
- **展示社区成果**：社区开发了各种工具和应用，例如支持多语言的 AI 驱动故事讲述者、根据古兰经经文创作海报艺术的 AI 工具，以及集成不同 OCR 技术的 OCR 工具包。可以在[此处](https://huggingface.co/spaces)参与这些项目。
- **YOCO 架构的曙光**：一篇[新研究论文](https://arxiv.org/abs/2405.05254)介绍了 decoder-decoder 架构 —— **YOCO**。据报道，这一突破在保持全局注意力能力并加快预填充（prefill）阶段的同时，降低了 GPU 显存需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **多 GPU 性能瓶颈排查**：成员们指出主板瓶颈导致多 GPU 配置性能低下。升级到兼容 PCIe 4.0 的主板解决了性能问题。
- **远程访问困惑解除**：LM Studio Server 的远程访问配置引发了讨论，最终明确将 'localhost' 替换为机器 IP 即可实现远程访问。
- **处理 LM Studio 中的失败与内存错误**：由于内存不足，成员们遇到了“Failed to load model”错误消息。解决方案包括关闭 GPU offload 或验证硬件是否满足模型运行要求。
- **社区合力解决 Linux 服务器难题**：一位成员在 Linux 服务器上安装 LMS 时遇到了 FUSE 设置问题。另一位用户分享了在 Ubuntu Server 24.04 上奏效的解决方案。
- **高功耗带来的 GPU 显存烦恼**：成员们一致认为使用 LLM 需要大量 VRAM。运行类似 GPT-4 的模型建议至少配备 8GB+ 显存。
- **本地模型受限于硬件限制**：关于在个人中等配置笔记本上运行高速本地模型可行性的讨论得出结论：LM Studio 可能无法完全支持此类配置。
- **文生图工具大放异彩**：Stable Diffusion、comfyUI 和 Automatic1111 等工具在文本转图像方面的实用性受到关注，并推荐了更简单的软件作为初学者友好的选择。
- **模型版本控制揭秘**：讨论了模型版本控制和微调方法，强调了阅读模型卡（model cards）以了解数据集和训练细节的重要性。
- **模型量化受到青睐**：成员们讨论了量化模型（如 Yi-1.5 系列）的好处。他们分享了特定量化模型的链接，以及提高模型性能和硬件兼容性的技巧。
- **上下文长度受模型约束影响**：模型上下文长度和预算的限制影响了模型选择，强调了不同 GPU 容量的局限性以及运行更大模型时必要的权衡。
- **开源倡导者宣布使用 Innosetup 和 Nullsoft**：一位成员推荐了开源安装程序 Innosetup 和 Nullsoft，并引用了他们过去的成功经验。
- **Starcoder2 在 Debian 上出现异常**：一位在 Debian 12 上测试 starcoder2-15b-instruct-v0.1-IQ4_XS.gguf 的用户遇到了重复响应和跑题回答，引发了关于该模型预期优化的深入讨论。
- **Playground 模式依赖 GPU**：成员们强调 Playground 模式不能仅靠 RAM + CPU 运行。有效使用至少需要 4GB 的 VRAM。
- **社区警告警惕欺骗性短链接**：发布了关于指向潜在不安全或无关网站的短链接的警告。
- **Llama 3 模型研究与 Token 速率探索**：成员们讨论了 Llama 3 模型在各种配置下的性能，并分享了 token 速率。还研究了使用 CPU 和 RAM 提高效率的可能性。
- **GPU 讨论中显现硬件限制**：比较了 Tesla P100 和 GTX 1060 GPU 的性能，发现由于潜在的 CUDA 版本不匹配，预期性能与实际性能存在差异。
- **Offloading 技术应对低 VRAM**：针对管理低 VRAM (2GB) 提出了 offloading 技术建议，重点是正确设置 offload 到 GPU 的层数。
- **CPU vs GPU：在 CPU 上运行 LLM 性能受损**：注意到仅在 CPU 上运行 LLM 会导致显著的性能下降。引用了通过调整 CPU 设置来提高特定 token 速率的案例。
- **界面调整深受用户欢迎**：社区成员讨论了在 GPU 和 RAM 之间调整模型负载。建议为模型分配更高的 VRAM 使用量，以避免加载失败和响应不足。
- **CodeQwen1.5 令编程爱好者惊叹**：成员们发现 7b 模型 CodeQwen1.5 在编程任务中非常高效。凭借 4b 量化和较小的占用空间，它证明了自己适用于 6GB GPU 配置，且性能优于 deepseek coder。
- **在 Huggingface 上探索编程模型**：建议将 Huggingface 的排行榜作为比较编程任务模型性能的首选来源。可以探索所有模型，尤其是 7b 或更小的模型。[查看编程模型排行榜](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)。
- **仅包含错误修复和一次小更新**：最新版本主要解决了错误修复，并包含了一个名为 **llama.cpp** 的更新。没有引入新功能。
- **成员们倡导谨慎点击**：用户必须警惕带有可疑链接的帖子，这些链接可能会产生不必要的收入，例如使用 goo.gle 缩短的链接。

- **MemGPT 查询借鉴 Kobold 经验**：一名成员寻求具有 MemGPT 经验的人士帮助，并可能得到另一位已将 MemGPT 与 Kobold 集成的成员的指导。
- **新购入的 GPU 表现出色**：一名成员以 700 欧元购买了 RX 7900 XT，并得出结论认为它完全符合需求。另一位成员建议，这款新 GPU 可以处理像 Command-R+ 或 YI-1.5（量化版本）这样的大型模型。
- **OpenInterpreter 连接困扰**：一名成员在将 LM Studio 与 OpenInterpreter 连接时表示困惑。该用户难以辨别错误消息的区别，无论服务器是否已连接，错误提示似乎都一样。
- **新的 Yi 模型备受关注**：LM Studio Community 发布了新的 Yi 模型，包括一个适用于 24GB 显卡的显著 34B 版本。这些模型通过 imatrix 进行了增强，提供多种尺寸，可在 [Huggingface 页面](https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF)上获取。
- **Vulkan 尝试在 LM Studio 框架中受阻**：用户在将 Vulkan 后端的 llama.cpp 与 LM Studio 集成时遇到困难，在当前框架内没有直接的解决方案。
- **LM Studio CLI 令动手型用户感到兴奋**：LM Studio CLI (lms) 推出，允许进行原始 LLM 检查、模型加载/卸载以及 API server 控制。更多关于用法的信息可以在 [LM Studio 博客](https://lmstudio.ai/blog/lms)上找到。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **JetMoE 8B 离线**：OpenRouter 的 [JetMoE 8B Free 模型](https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free) 演示出现了 502 错误，这可不是什么新舞步。由于上游过载，它目前处于离线状态。建议用户暂时更换其他模型。
- **两款 Multimodal 模型加入 OpenRouter 阵营**：OpenRouter 更新了其模型阵容，新增了两款 Multimodal MVP —— [GPT-4o](https://openrouter.ai/models/openai/gpt-4o) 和 [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b)。更多像素，更多文本，更强的 AI 动力。
- **API 监控工具上线**：厌倦了通过刷新来检查 OpenRouter 不断演进的模型列表？试试 [OpenRouter API Watcher](https://orw.karleo.net/)，它能监控这些变化并存储在 SQLite 数据库中，拥有美观的 UI 和用于更新的 RSS 订阅。让你的 F5 手指休息一下吧。
- **揭秘 Rubik's AI**：高级研究助手和搜索引擎 **Rubik's AI** 开启 Beta 测试，并提供诱人的优惠——两个月免费试用 Claude 3 Opus、GPT-4 Turbo、Mistral Large 等 AI 精品。快去[这里](https://rubiks.ai/)看看吧。
- **OpenRouter 三人团队全力打击欺诈**：凭借更强硬的反欺诈措施和为了安全而收集的少量必要个人数据，OpenRouter 的三人团队正正面应对运营干扰，并依靠 [Stripe](https://stripe.com/) 等平台提供支持。
- **交流中心**：OpenRouter 中的嵌入式模型？以后再说。用于创建多个可自定义角色或 Agent 的高级 WebUI？当然，可以试试 BigAGI 或 OpenWebUI。哦，我们是否提到过 Jetmoe 没有联网功能……以防万一你想知道。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **围绕 Mojo Nightly 构建的热烈讨论**：`mojo` 框架的最新进展引入了 Nightly 构建，它会自动将合并的 commit 直接推送到其 `nightly` 分支。社区成员可以在[相关 PR](https://github.com/modularml/mojo/pull/2644)中查看详细的工作流超时调整。

- **Mojo List 操作的内存管理亟需改进**：社区内对 Mojo `List` 内存预分配的潜在低效问题讨论热烈。优化后在特定基准测试中可带来 2000 倍的加速，这表明演进我们的内存管理策略迫在眉睫。

- **GitHub Actions Bug 影响透明度**：Mojo 用户正面临 GitHub Actions 的一个关键 Bug，即已完成的任务伪装成“pending（待处理）”状态。这种误导性行为掩盖了正在进行的工作流可见性，影响了 Mojo 最近的 commit 和 CI 操作。

- **类型实例化（Type Materialization）问题困扰 Mojo**：关于 Mojo 中正确类型实例化的讨论集中在类型转换期间的内存指针管理等问题上。这些问题导致了测试失败，并需要修订相应的方法。

- **新 MoString 仓库挑战 Mojo 开发者**：[MoString](https://github.com/dorjeduck/mostring) 是一个新的 GitHub 仓库，展示了在 Mojo 中探索的各种 StringBuilder 想法，包括一种优化内存分配的方法。这项努力呼吁社区贡献，被证明是突破 Mojo 边界的一次有趣尝试。

- **新视频聚焦 Mojo 的所有权（Ownership）机制**：最近分享的一段[视频](https://www.youtube.com/watch?v=9ag0fPMmYPQ)阐明了 Mojo 中的所有权机制，旨在深化相关知识。Python 开发者分享了这些概念如何从 Python 过渡到 Mojo 的见解，这一视角有望为新手提供更好的清晰度。

- **Mojo 与 Rust 编译器的权衡**：Mojo 与 Rust 编译器的对比凸显了 Mojo 更简单的方法，专注于编码而非纠结于文档或复杂的编译器细节。Rust 强大的系统设计和自动向量化能力伴随着陡峭的学习曲线，强调了审慎选择工具的必要性。

- **理解 SQL、ORM 与编译器的语言查询权衡**：在一场激烈的讨论中，SQL 的易用性与 ORM 及 Rust 等编译器的严格系统要求产生了碰撞。这些技术呈现出不同程度的舒适度和效率，意味着选择必须取决于个人偏好和项目需求。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **内核融合 (Kernel Fusion) Zoom 会议揭秘**：技术公会组织了一场关于**内核融合实际经验**的 Zoom 会议。与会者被引导在特定的 Discord 频道中发布讨论和疑问，从而增加了参与度并营造了专注的学习环境。
- **伊利诺伊大学 PMPP 系列课程受到关注**：公会继续开展针对 EMEA 和 NAM 地区的伊利诺伊大学 **PMPP 系列**每周讲座。通过 [YouTube 播放列表](https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=shared)和直接的 [Zoom 链接](https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1)，这些课程变得更加易于获取。
- **探讨 CUDA、Triton 与内核融合的艺术**：GPU 内存管理和 CUDA 构成了讨论的核心，经常出现的主题包括 Triton 的优化潜力以及内核融合的优势和策略。分享了关键资源，包括论文、PRs、教程和 [GitHub commits](https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac)。
- **解决 GPU 兼容性与安装问题**：用户关于 CUDA 版本与特定 Torch 版本的兼容性以及多 GPU 脚本编写的疑问，突显了实施过程中面临的实际挑战。这些疑问得到了澄清，使 GPU 利用更加有效和高效。
- **使用 ThunderKittens 简化 AI 内核构建**：公会讨论集中在 ThunderKittens 上，这是由 HazyResearch 引入的一个新开源项目。该项目的瓦片原语 (tile primitives) 旨在简化 AI 内核构建，使 AI 的计算目标对用户而言更加触手可及。
- **利用 llm.c 和 CUDA 提升性能**：用户辩论了 CUDA 图 (CUDA graphs) 和 `torch.compile` 的功效，在寻求核心流程清晰度的同时考虑性能增强。其他对话集中在 llm.c 未来可能利用 ThunderKittens 进行改进，强调了在 GPU 编程中对创新的持续追求。
- **PMPP 书籍 YouTube 观看派对启动**：推出了一个新的 YouTube 观看派对系列，重点关注 PMPP 书籍 2018 年的讲座。通过定期的会议和互动讨论，公会旨在促进学习和实践，使其成为 CUDA 爱好者和初学者的宝贵资源。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **合成数据——下一个大趋势还是旧瓶装新酒？**：[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239488393713287199) 频道就合成数据是否具有真正的变革性影响展开了激烈辩论。以往炒作周期的教训、遗忘这些教训的可能性以及所产生的权衡都是热门话题。
  
- **网络结构之争**：包括 *CNNs*、*Transformers* 和 *MLPs* 在内的深度神经网络在一项共享[研究](https://arxiv.org/abs/2108.13002)中被置于统一视角下观察。另一篇[论文](https://arxiv.org/abs/2306.13575)探讨了 MLPs 的极限，暗示尽管目前存在障碍，但仍有未开发的扩展可能性。
  
- **多模态模型中模糊的“Zero-Shot”声明受到质疑**：在 [general](https://discord.com/channels/729741769192767510/729741769738158194/1238856315993063554) 频道，一篇[最近的研究论文](https://arxiv.org/abs/2404.04125)将多模态模型引人注目的“Zero-Shot”声明与预训练数据中的概念频率联系起来，引发了对这些 AI 能力真实基础的质疑。
  
- **Falcon2 11B 表现强劲**：代号为“Condor”的 Falcon2 11B 模型消息已披露，该模型拥有 8k 上下文窗口和改进的注意力机制。它在 5T 网页数据集上进行训练，预示着推理能力的广阔前景。
  
- **NeurIPS 合作与模型压缩思考**：[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1239481105514758144) 频道发出了 NeurIPS 投稿合作的呼吁，让人联想起“othello 论文”。模型压缩见解以及在此过程中丢弃的特征性质是讨论的中心。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o 作为下一代前沿模型令人惊叹**：OpenAI 在 LMSys arena 中以别名 "im-also-a-good-gpt2-chatbot" 推出了其最新的前沿模型 **GPT-4o**。该模型表现出显著的性能提升，相关消息在一条 [推文](https://x.com/liamfedus/status/1790064963966370209?s=46) 中公布。

- **GPT-4o 的编程能力引发好奇**：GPT-4o 与其先前版本在编程能力上的巨大差距成为热门话题，引发了人们对新设立的 MATH 基准测试的兴趣。有关这些进展的更多细节可以通过这篇 [博客文章](https://openai.com/index/hello-gpt-4o/) 了解。

- **Tokenizer 更新有望提升效率**：OpenAI 对其 Tokenizer 的最新更新暗示了更高的效率，这可能源于词汇表的扩大。你可以直接在这个 [GitHub commit](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8) 中查看 Tokenizer 的更新。

- **OpenAI 的战略决策引发猜测**：OpenAI 免费开放 GPT-4o 使用权限的战略决策在成员中引起了猜测，导致了大量的假设。从数据收集到针对 Meta 等科技巨头的竞争定位，论坛上充斥着比较 OpenAI 战术举措的讨论。

- **GPT-4o 现场演示评价两极分化**：OpenAI 对 GPT-4o 的现场演示引起了广泛的反响，从潜在适用性的讨论到对演示风格的批评。所展示技术的真实性、有效性和集成方面已成为社区成员审视的焦点。

- **揭示 REINFORCE 是 PPO 的衍生**：Huggingface TRL 仓库中一个富有启发性的 PR 提出 **REINFORCE** 是 **PPO** 的一个特例。这一令人惊讶的发现在一个 [GitHub PR](https://github.com/huggingface/trl/pull/1540) 中得到了深入探讨，该 PR 提供了详尽的解释以及一篇 [参考论文](https://arxiv.org/pdf/2205.09123)。

- **Chatbot Arena 受欢迎程度上升**：**Chatbot Arena** 社区因其作为 AI 未来重要贡献者的地位而赢得了成员们的赞誉。

- **成员们讨论开源 GPT-3.5 的想法**：**GPT-3.5** 潜在的开源可能性进入了讨论范围，并引发了一些有趣的反应，其中一位成员断言这只有在“太阳从西边出来”（原文为 hell freezes over）时才会发生。

- **AI 视频观看量激增**：据报道，视频观看量令人印象深刻，一段视频在 **一天内达到 6k 观看量**，其他视频达到 **20k 观看量**。在 HuggingFace 上分享的一段视频获得了巨大回报，观看量达到 **150k**。

- **在 Platform X 上发布视频受到审视**：在 Platform X 上发布视频的谨慎想法引发了关于 **原生上传** 合法性以及权限问题的讨论。

- **斯坦福拥有权利但保持灵活性**：一位成员确认 **斯坦福拥有特定内容的权利**，但在执行方面通常比较宽松，这为更自由的使用提供了机会。规避官僚主义的建议措施包括 **申请个人使用许可**，同时承担可能产生后果的风险。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **艺术模仿 AI - 潜在法律纠纷引发辩论**：一个备受关注的热点话题是 AI 服务的潜在法律陷阱，例如 **Midjourney** 创作的艺术作品被认为可能与在世艺术家形成竞争。讨论重点集中在艺术家权利与 AI 商业应用之间的平衡。
- **版权、AI 与公平使用的细则**：频道内充斥着关于 AI 在生成衍生作品时是否侵犯艺术家版权的辩论。同时，一派观点在这一知识产权之争中举起了公平使用（Fair Use）保护的盾牌，指出这与负面评论损害创作者业务的情况具有潜在的相似性。
- **AI 艺术与公平使用 - 观点交锋**：在 AI 艺术这一充满争议的领域，并非所有人的意见都一致；一些成员呼吁对艺术家销售额的潜在影响进行更严密的法律审查，而另一些人则坚定立场，将此类使用标记为广义公平使用框架下的合理行为。
- **法庭中的 AI - 陪审团不再适用？**：对话从艺术转向了陪审团，讨论涉及陪审团否决权（Jury Nullification），以及在解释 AI 相关法律时人类与代码的角色。在这个 AI 时代，成文法典与现实世界法律应用之间的反差引发了人们的兴趣。
- **AI 走向绿色 - 在巨头时代寻求能效**：社区成员分享了旨在降低 AI 庞大能源需求的创新，寻求为绿色未来设计的新模型和方法。其中一个来源在[频道内引起了关注](https://www.techopedia.com/openais-gpt-4o-release)。
- **变革声音景观 - 音频数据成为核心**：关于将海量语音数据集转换为 Token 的任务，讨论声浪越来越高。针对情感和说话者特征的高质量标注成为了讨论焦点，一位成员分享了相关的[实践资源](https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19)和 [YouTube 上的教育内容](https://youtu.be/NwZufAJxmMA)。
- **收敛于数学符号 - 讨论形式数学中的挑战**：随着对表示元素序列的某些形式数学符号的使用（或可能误用）的讨论展开，现场充满了技术术语。在这场讨论的余波中，函数 **T** 被推崇为过程序列中采样的重要工具。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **与 LangChain 中的 ISO 日期约会**：该频道分享了关于使用 **LangChain 的 DatetimeOutputParser** 提取日期并将其转换为 ISO 格式的见解。提供了 JavaScript 和 Python 的代码示例供动手实践。
   
- **为日期范围扩展 DatetimeOutputParser**：为了深入探讨 LangChain 中如“从 4 月 1 日到 6 月 2 日”之类的日期范围管理，有人提议重构 `DatetimeOutputParser`。一位精通设计的成员建议调整 `parse` 函数，以分别识别和提取开始及结束日期。

- **针对多市场描述的 Agent 解决方案**：围绕使用 LangChain 的 tool/function calling 与 LLM 从提示词中提取多个市场描述展开了多方面的讨论。通过结构化提取方法，从诸如“比较比利时石油和意大利电力之间的价格”之类的提示词中获取信息变得更加清晰。

- **开源 LLM 与 LangChain 的亲密接触**：关于 Ollama 等本地开源 LLM 集成到 LangChain 的一些巧妙见解。准备好揭开大量数据的面纱，从设置 LLM、安装必备包到最终与模型进行交互。

- **关于 API 响应流式传输的热烈讨论**：希望通过单个 API 调用为多个前端元素获取 API 响应？通过 Python 特有的细节获得帮助，并参考这个[相关的 GitHub 示例](https://gist.github.com/mattcollins/62fcb8d15a001d5b4e5c9fb86aad4f8e)。

- **癌症药物研发沉浸在 AI 热潮中**：倾听这段[引人入胜的 YouTube 论述](https://youtu.be/vyOtowbGwG0?feature=shared)，了解 Generative AI 如何重新定义癌症药物研究的轮廓。对更多自动化方法的迫切需求成为了关注焦点。

- **开源 Code Interpreter 迈出第一步**：一个旨在辅助可视化与交互式数据分析（NLAVIDA）的开源项目惊艳亮相。该项目承诺**未来将兼容 OpenAI API key 和 Llama 3**，揭开了机密数据分析的神秘面纱。

- **博主尝试使用 LangChain 和 Pinecone 构建 RAG 流水线**：如果你一直渴望为自己的博客添加一个利用 **Retrieval Augmented Generation（RAG）技术**的聊天功能，那么请关注[这里](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog)。本教程将系统地引导你完成从数据摄取到构建引人入胜的聊天界面的全过程。

- **LLM 展露锋芒，迈向 Multimodal**：正如[相关的 YouTube 视频](https://www.youtube.com/watch?v=KQ-xGVFHDkw)和配套的 [GitHub notebook](https://github.com/githubpradeep/notebooks/blob/main/VLM.ipynb) 所明确指出的，随着 DinoV2 的加入，LangChain 旨在实现 Multimodal。

- **带有会话和历史管理的流式传输传奇**：一位成员正在寻求教程或帮助，以便在兼顾会话和历史管理的同时，将流式传输（streaming）功能集成到 LangChain 中。此前，除了流式传输外，已解决了多个瓶颈问题。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama 3 驱动的自动化 PowerPoint 奇迹**：一位用户发布的文章展示了 **Llama 3 RAG pipeline** 如何结合 Python-pptx 库，不仅能提供答案，还能生成 PowerPoint 幻灯片。点击[此处](https://t.co/iM0c5Cl2uK)查看文章。
- **以反思方式设计金融专家**：Hanane Dupouy 的指南详细介绍了利用 **CRITIC** 方法论进行股价分析并构建金融顾问的过程。点击[此处](https://t.co/mmJ8cjmw73)获取所有智慧结晶。
- **RAG 的内容控制实力**：展示了 RAG 流水线如何强制执行用户生成图像的审核规则。完整的技术细节请见[此处](https://t.co/z6jBpMvQss)。
- **RAG 系统的实力评估**：对 TruLens、Ragas、UpTrain、DeepEval 这四个 **RAG system** 评估库进行了深入评估，并附带支持的指标，以简化您的性能评估流程。点击[此处](https://t.co/gLbXJoPsqu)阅读全文。
- **Llama 3 的能力在黑客松指南中尽显**：由 @AIatMeta 主办的黑客松汇编了 **Llama3** 的七个不同用例，涵盖了从简单到复杂的各项任务。所有方案均已汇总在[此处](https://t.co/YLlsvkI0Ku)。
- **解决 LlamaIndex 的缓存问题**：一位用户在 _aretrieve_context 函数中发现了一个导致非预期后处理器删除的 Bug，但很高兴地发现该问题已在当前版本的 **llamaIndex** 库中修复。
- **混合搜索设置障碍**：一位用户在设置 **Qdrant** 混合搜索时遇到了 ValueError，通过在构造函数中启用混合搜索解决：`"QdrantVectorStore(..., enable_hybrid=True)"`。
- **深入了解 LlamaIndex——肯定的评价**：成员们称赞 **LlamaIndex** 易于使用、具有灵活性、文档出色，且在管理多平台支持方面表现优异。
- **前端 AI 响应异常**：一位成员遇到了前端显示的 AI 输出不一致的问题，并收到了错误消息 *"Unexpected token U"*，引发了关于潜在原因的讨论。
- **使用 LlamaIndex 查询——利用元数据**：针对用户关于在使用 **llamaIndex** 时元数据在 `query` 方法中作用的提问，随后进行了讨论，澄清了元数据在过滤和检索过程中的用法。
- **通过知识蒸馏提升 GPT-3.5**：**Hugging Face** 上的一篇文章讨论了知识蒸馏如何改进作为裁判的 **GPT-3.5** 的微调，并附带了全面指南。点击[此处](https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt)查看。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3 微调中的权重问题**：对 **Llama 3** 的 instruct 模型和 base 模型之间的*权重差异*进行的检查指出，显著变化主要集中在 K 和 V 层，这暗示了在 instruct tuning 期间进行了针对性调整。目前正在考虑在不丢失指令能力的情况下，冻结 K/V 层进行 style tuning 的可能性。
  
- **Checkpoint 难题解析**：社区对*检查点命名规范*进行了澄清，强调运行结束时的保存（end run save）实际上应该位于 base 文件夹中——这是在模型运行期间解读保存输出时的一个关键细节。

- **评估 OpenOrca 重新运行的资金**：一位社区领袖提议在 **gpt-4o** 上重新运行 OpenOrca 的去重（dedup）工作，并提供了成本估算，以及关于潜在 batch job 定价优势的额外见解。你可以在其 [数据集页面](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup) 关注进展。

- **应对高计算消耗的先锋**：一系列旨在降低 AI 极高计算消耗的项目受到关注，包括 **Monarch Mixer**、**H3** 和 **Hyena Safari**。欲了解更多深度内容，请查看他们的 [博客](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)。

- **应对 AI 研究出版洪流**：学术期刊出版的缓慢节奏可能会让前沿研究在快速发展的 AI 世界中过时——这是社区讨论的一个突出挑战。

- **Nanobitz 的合并热潮取得成功**：据报道，用户 "Nanobitz" 的代码合并（merge）获得成功——遗憾的是，合并的具体细节仍然是个谜。

- **LLAMA3 模板错误引发关注**：PyET 中的一个 LLAMA3 模板遇到了障碍，引发了 'LLAMA3' 和 'LLAMA2' 之间的混淆。解决办法？更新你的 **fastchat**。

- **项目依赖项需要翻新**：用户 "trojaner" 发现项目依赖项严重过时，例如 **peft**、**accelerate**、**deepspeed**、**flash-attn**、**xformers** 和 **transformers**。需要全面升级到最新版本——除了 peft，由于一个棘手的插件问题，它需要从仓库进行安装。

- **FSDP 与 FFT：缺失的拼图**：**Fully Sharded Data Parallel (FSDP)** 与 **Fast Fourier Transform (FFT)** 的兼容性仍无定论。与此同时，另一种替代方案正在考虑中——即 [DeepSpeed](https://www.deepspeed.ai/) 路线。

- **Docker AttributeError 解析**：诊断了在 **Docker** 场景下使用 **LLAMA3** 时遇到的 AttributeError。补救措施？更新你的 **pip dependencies** 并尝试重新进行 **git clone**。

- **Git 克隆解决了 fastchat 的问题**：**git clone** 方法成功解决了持久的 fastchat 问题，这标志着某些分支中存在未更新 commit 的潜在障碍。

- **Axolotl CLI 中 `system_prompt` 更改的困惑**：修改 **axolotl.cli.inference** 中的 `system_prompt` 让一位用户感到困惑。甚至 AI 顾问 Phorm 也没能给出答案，这凸显了一个值得再次探讨的未解决问题。

- **将合并后的模型转换为 GGUF 遇到障碍**：由于缺少匹配的 tokenizer ['spm', 'hfft']，在将合并模型转换为 GGUF 期间发生了 **FileNotFoundError**。该错误提醒在未来的任务或问题解决中需要优化文件结构或命名。

- **Gemma 模型加载故障**：一位用户在加载 *GemmaForCausalLM* 模型时遇到了 `model.embed_tokens.weight` 的 **size mismatch error**（尺寸不匹配错误）。建议的排查策略是在 `from_pretrained` 方法中添加 `ignore_mismatched_sizes=True`，这凸显了训练环境与应用环境之间的不匹配问题。

- **QLORA 合并中的精度问题**：提出了一个关于将 QLORA 合并到基础配置且不产生 fp16 和 fp32 精度差异的问题，强调了模型集成和精度处理中存在的挑战。

- **Axolotl Phorm Bot 来帮忙！**：为了寻求有关 Axolotl 剪枝能力和持续预训练技巧等方面的建议，用户求助于 **Axolotl Phorm Bot**。但遗憾的是，即使是机器人也未能给出答案，建议日后再次关注这些引人入胜的问题 [在 Phorm 上阅读更多](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)。

- **qLoRA 与基础模型的集成仍未解决**：一名成员关于如何将 qLoRA 合并到基础模型中的提问在讨论中悬而未决，这表明该问题需要在未来的讨论中进一步深入研究。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Claude 的兼容性问题**：**Claude API** 集成出现了一些问题，用户遇到了“奇怪的错误”。目前尚不清楚这些是兼容性故障还是配置问题。

- **使用 Open Interpreter 自动化 Antidetect**：一场热烈的讨论建议，可以通过使用 **Open Interpreter** 根据自然语言指令生成 Python 代码，从而提升浏览器自动化的水平。高影响、低投入的自动化？当然欢迎！

- **本地模型讨论热烈**：关于本地模型 **Mixtral**、**Phi**、**Llama3** 和 **GPT-4** 性能的辩论非常激烈。大家一致认为 **GPT-4** 表现最佳。然而，提升本地模型有效性的关键不再仅仅取决于模型本身，而在于 Prompt 优化。

- **GPT-4o 动力十足**：**GPT-4o** 作为 AI 领域的新秀，以其闪电般的速度超越了其他所有模型——拥有高达 100 tokens/s 的速度，在性能和成本效益上都遥遥领先。

- **ChatGPT 与 Interpreter API：期待已久的强强联手**：所有目光都集中在 **ChatGPT 语音对话 AI** 可能与 **Open Interpreter API** 结合的前景上，这可能会带来一些重大的突破。准备好你的爆米花。

- **LiteLLM 与 Llama3 的完美配合**：用户们正愉快地将 **OpenInterpreter**、**LiteLLM** 和 **Groq - Llama3** 连接起来，并在配置上进行了一些重大的尝试。

- **01 硬件 Wifi 困扰**：一位用户分享了关于 **M5 board** 和 **01-Light wifi 网络** 设置的糟糕连接经历，引起了广泛关注。他们能度过这个难关吗？

- **01 移动端 App 版本上线**：现在，**01 硬件** 告别了桌面，走向了移动端。感谢 Thatpalmtreeguy，一个早期的 App 版本已在 [这里](https://github.com/eladdekel/01_For_iOS) 发布。

- **另一个 Apple 应用等待 TestFlight 审核**：Thatpalmtreeguy 持续带来惊喜。在他提到一个应用正在等待 **TestFlight** 批准后，人们预测了光明的未来。

- **客服化身福尔摩斯**：一个 **OpenInterpreter** 订单丢失了，寻找它需要的不仅仅是细致的搜寻。*help@openinterpreter.com* 的客服能破解这个案件吗？

- **PyWinAssistant 发布**：最新的 AI 助手 *PyWinAssistant* 已进入赛场，一位用户将其宏伟地描述为“第一个通过自然语言控制人类用户界面的开源 Large Action Model”。所有 [GitHub 详情请见此处](https://github.com-real-ai/pywinassistant)。

- **观看 PyWinAssistant 现场演示**：只需点击这个 [YouTube 链接](https://www.youtube.com/live/_XyYoqpJCoQ?si=rA3ijqicagANyt96&t=1993)，即可见证 **PyWinAssistant** 近乎实时的魔力。准备好你的零食和饮料！

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor 中的变量形状（Variable Shapes）解析**：一位用户参考 [Tinygrad Notes](https://mesozoic-egg.github.io/tinygrad-notes/upcast2.html) 询问了 Tensor 中变量形状的必要性。该功能对于处理 Tensor 形状动态变化的情况至关重要，可以优化编译时间并避免为每个新形状重新生成 Kernel。
   
- **Tinygrad 训练错误已解决**：在模型训练过程中遇到的 “AssertionError: Tensor.training should be set in the optimizer” 错误，通过设置 `Tensor.training = True` 得到了解决，详见此 [Pull Request #4460](https://github.com/tinygrad/tinygrad/pull/4460/files)。
  
- **高级索引操作探讨**：小组讨论了在 Tinygrad 中实现高级索引操作（如 `node_features[indexes[i]] += features[i]`）的挑战和策略。提出的解决方案之一是使用 One-hot 编码和矩阵乘法来根据索引聚合特征。

- **图神经网络（GNN）的好奇心**：关于如何在 Tinygrad 中实现图神经网络（GNN）的讨论集中在邻居搜索上。话题包括与 Pytorch Geometric 等库相比实现的复杂性，以及原生 O(N^2) Tensor 操作方法可能存在的低效性。
   
- **改进 Tinygrad 的错误处理**：成员们强调，更好的错误处理是提升 Tinygrad 用户体验的一项重要功能。此类增强可以借鉴 Rust 风格的错误消息原则，提供最简单的修复建议，使用户解决问题更加直接。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **拨开 Cohere 账单迷雾**：用户对 **Cohere 账单** 详情感到困惑。经过讨论，结论是账单差异是由于自上次发票以来累积的费用导致的。
  
- **Command R 的尺寸问题**：关于 **Command R** 影响的辩论中，成员们证实了在涉及网页搜索时，输入 tokens 确实会变大。

- **破解 Glitch Tokens 之谜**：一篇关于大语言模型 tokenizer 中 "glitch tokens" 的著名[研究论文](https://arxiv.org/abs/2405.05417)引发了关于 tokenizer 效率和模型安全性的讨论。

- **Aya vs Cohere Command Plus：当 Sharp 不再锋利**：关于 Aya 和 **Cohere Command Plus** 之间性能差异的疑虑。用户体验各异，从 Aya 在通用知识方面的回答不准确，到建议仅将 Aya 用于翻译。
  
- **求助！这里总有支持**：一位用户对感知的 **Cohere 支持** 缺乏表示沮丧。其他成员迅速向他保证了社区的响应能力和工作人员的可用性。

- **需要专家：电信领域**：邀请对 5G 电信领域大语言模型专业化感兴趣的工程师。挑战赛链接可以在[这里](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks)找到。
  
- **你的 PDF 会聊天吗？**：发布了关于 **Cohere 用于 'Chat with PDF' 应用的潜力** 的咨询，引发了多次回应。用户寻求有关当前项目的信息以及相关阅读材料和仓库的建议。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LMSYS 与 LLM 质量之争未定**：将 **lmsys** 作为评估 **LLM** 质量指标的效用在小组内仍存在争议。目前尚未就此问题达成明确观点。
- **GPT-4o 未能兑现承诺**：批评者指出了 **GPT-4o** 的性能缺陷，特别是它无法正确列举书籍。尽管其响应速度快且价格诱人，但与前代 **GPT-4** 相比，该模型在基础推理能力方面似乎有所滞后。
- **AI 未来展望**：鉴于目前 **GPT-4** 和 **Claude 3 Opus** 等模型展示出的改进有限，对 **AGI** (Artificial General Intelligence) 过度炒作的怀疑浮出水面。一些小组成员对未来迭代中预期的进步表达了谨慎的乐观。
- **Google Vertex AI 额度困境**：一位成员询问了如何有效利用即将到期的 **Google Vertex AI** 额度。然而，目前仍缺乏任何潜在测试的具体计划。
- **语音助手性质存疑**：语音助手不合时宜的笑声问题被提出，这可能会损害用户体验。讨论了使用 custom prompts 作为潜在补救措施的建议，以保持输出的专业性并防止潜在的用户获取障碍。
- **利用关于 LLM 的推文**：成员 **@SimonW** 分享了一条提供 **LLM** 见解的推文。提供了[推文链接](https://twitter.com/simonw/status/1790121870399782987)，未附带额外背景或讨论。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **尚未发现适用于 OpenELM 的 GGUF**：成员们指出，一个声称包含 **OpenELM GGUF** 的仓库其实是误导信息。在数字信息领域中，专注、准确和主动是关键。
- **优化 llamafile**：通过新的 [Pull Request #412](https://github.com/Mozilla-Ocho/llamafile/pull/412)，新增的脚本利用外部资源简化了 llamafile 归档的升级。这是技术实力的完美展现！
- **Hermes 是个速度健将**：个人测试报告显示，**Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf** 模型在 llamafile 上运行流畅，在 AMD 5600U 系统上响应时间接近 10 秒，RAM 占用峰值为 11GB。作为参考，该模型大小高达 5.6GB。
- **模型“旷课”了**：用户反馈在实现 **Llama 8B 和 Mistral** 等模型时经常遇到小故障，罪魁祸首通常是 KV cache 空间问题。性能随不同系统的可用 RAM 而异。
- **增强 Llamafile 的元数据管理**：目前正在开展工作，允许在 **llamafile 和 gguf** 中集成自定义作者元数据。这为文件管理和在 Hugging Face 等平台上进行便捷搜索提供了更实用的方法。详情请见[此处](https://github.com/ggerganov/llama.cpp/issues/7165)。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **德语 YouTube 内容搜寻**：一位成员动员社区共同整理一份高质量德语播客、新闻节目和 YouTube 频道的完整列表。目标是为德语 Text-to-Speech (TTS) 系统收集有价值的训练数据。
- **使用 MediathekView 进行整理**：MediathekView 被推荐作为从各种德国广播公司下载节目和电影的工具，为德语 TTS 系统训练提供了潜在的金矿。该平台因其本地存储的海量电影数据库（包括链接和详细描述）而受到关注，可在此处[下载](https://mediathekview.de/)。
- **JSON API 是 MediathekView 的秘密武器**：通过 MediathekView 的 JSON API 自动访问媒体内容数据的可能性引发了兴趣。这为高效收集和组织德语电影数据库打开了大门，更多深度探索见此 [GitHub 链接](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua)。
- **演示的困惑与赞赏**：一位参与者询问了 Discord 频道中一个 Demo 的运行状态。随后，该成员表达了赞赏，称该 Demo “非常棒”。
- **语言翻译困扰，请坚持使用英语**：频道内发布了温馨提示，要求保持英语作为主要沟通语言。这确保了内容对多元化国际社区的所有成员都是可访问且可理解的。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Claude 3 vs Llama 3b：巨头之争**：关于使用 **Claude 3 Haiku** 和 **Llama 3b** 进行实体提取评分服务的对比引发了深度讨论。其想法是从传统的模糊字符串匹配转向使用更小的 **LLM**，以协调 Pydantic 模型中的子模型。
- **实体提取建模**：工程师们致力于构建评分服务，重点关注文档实体提取的*准确性*。他们计划使用 Pydantic 模型来比较预测结果和实际结果，首先从 *Instructor* 开始。
- **音频技术：下一个前沿？**：对音频相关元素的期待在增长，可能是为助手提供**音频输入输出支持**。OpenAI 音频团队参与度的提高增加了这些推测的分量。
- **GPT-4o 发布在即**：即将到来的 OpenAI 春季更新预计将于 **2024 年 5 月 13 日星期一**揭晓备受期待的 [GPT-4o](https://www.youtube.com/watch?v=DQacCB9tDaw)。此次活动还将带来 *ChatGPT* 的更新，进一步点燃了大家的热情。
- **明星效应引发兴奋**：社区对女演员 **Scarlett Johansson** 为 AI 领域注入明星力量感到非常兴奋，这提高了即将推出的功能或活动的关注度。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AlphaFold3 联邦准备举行会议**：AlphaFold3 联邦定于美国东部时间 5 月 12 日晚上 9 点举行见面会。与会者可以期待关于 **AlphaFold3 集成现状**的更新、训练流水线架构中可能的瓶颈以及开放问答环节。在此 [RSVP](https://lu.ma/swinnyfl)。

- **橙色角色神秘出现**：一名成员的询问引发了关于服务器角色的讨论，特别是他对“Orange Team”的好奇。目前尚未提供答案。

- **Fasteval 寻找新火炬手**：'tju01' 分享了停止开发 **fasteval** 项目的消息，且目前没有相关后续计划。他们有兴趣将 GitHub 项目移交给合适的继任者，否则该项目的频道可能会被遗忘在数字尘埃中。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **新速度，新人群**：关于 AI Town 内不同的**角色移动速度**和 **NPC 数量**的询问引起了关注，但目前尚未收到任何回复。对于热衷于 AI Town 的用户来说，未来可能会有更多的实验自由。
- **NPC 与玩家之间的平衡**：一位工程师深入研究了优化 **AI town** 内玩家与 NPC 交互的方法，建议**降低 NPC 交互频率**。通过利用 **llama3 模型**，他们希望减轻本地机器的计算负载并增强整体玩家体验。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

提供的文本不包含足够的信息来生成有意义的摘要。



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

抱歉，无法为此频道生成摘要报告。用户 "pranay01" 提供的消息 "Agree!" 缺乏足够的上下文和实质性内容，无法纳入技术摘要。



---


**MLOps @Chipro Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1238761392711012402)** (834 条消息🔥🔥🔥): 

- **量子 vs. 图灵与技术监管**：讨论围绕量子计算机与图灵计算机的有效性展开，目前图灵机甚至在量子计算机预期擅长的领域也表现得更好。人们还对政府关注监管量子计算而非 AI 表示担忧，认为这可能会牺牲小型创新者而使大公司受益。

- **对 OpenAI 政策和合作伙伴关系的批评**：成员们对 OpenAI 走向排他性和监管俘获（如 GPU 签名以及与白宫的合作）表示不满，认为这些行为可能会阻碍开放竞争和创新。

- **对模型审查和访问的担忧**：社区讨论了未经审查模型的潜在危险和伦理问题，将 AI 模型控制与刀具等物理工具的监管进行比较，强调监管应侧重于滥用而非工具本身。

- **关于模型训练和操纵技术的讨论**：就各种模型训练和操纵策略进行了技术交流，包括使用未经审查的 LLMs 以及在未经明确授权的情况下将模型与新适配项合并的方法。

- **社区对扩大开源项目的兴趣**：对话还涉及了扩大开放、共情项目的倡议，并呼吁社区参与，以丰富 AI 在更广泛、更细微的人类语境中的理解和实现。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1789659394302718373">Daniel Han (@danielhanchen) 的推文</a>：正在修复 LLM 微调 bug 并发现了 4 个问题：1. Mistral：HF 的 batch_decode 输出错误 2. Llama-3：注意双 BOS 3. Gemma：第 2 个 token 有多余空格 - GGUF(_Below) = 3064...</li><li><a href="https://arxiv.org/abs/2402.08787">重新思考大语言模型（LLMs）的机器卸载（Machine Unlearning）</a>：我们探索了大语言模型（LLMs）领域的机器卸载（MU），即 LLM 卸载。该计划旨在消除不良数据影响（例如敏感或非法...）</li><li><a href="https://typst.app/docs/reference/text/lorem/">Lorem 函数 – Typst 文档</a>：`lorem` 函数的文档。</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://www.together.ai/blog/thunderkittens">ThunderKittens：用于 AI kernel 的简单嵌入式 DSL</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/falcon-11B">tiiuae/falcon-11B · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/gojo-satoru-gojo-ohio-gif-27179630">五条悟 Gojo GIF - Gojo Satoru Gojo Ohio - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo">NTQAI/Nxcode-CQ-7B-orpo · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/lyogavin/Anima/tree/main/air_llm#quickstart">lyogavin/Anima 的 main 分支下的 Anima/air_llm</a>：33B 中文 LLM，DPO QLORA，100K 上下文，使用单个 4GB GPU 进行 AirLLM 70B 推理 - lyogavin/Anima</li><li><a href="https://ollama.com/eramax/nxcode-cq-7b-orpo">eramax/nxcode-cq-7b-orpo</a>：https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/llamafy_qwen.py">hiyouga/LLaMA-Factory 的 main 分支下的 LLaMA-Factory/scripts/llamafy_qwen.py</a>：统一 100 多个 LLMs 的高效微调。通过在 GitHub 上创建账号为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://tenor.com/view/joy-dadum-wow-drums-gif-14023303">Joy Dadum GIF - Joy Dadum Wow - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral 微调入门指南（包含 16k, 32k, 128k+ 上下文）</a>：在我们最新的教程视频中探索使用自己的数据轻松微调语言模型（LLMs）的秘密。我们深入探讨了一种经济高效且...</li><li><a href="https://youtu.be/DQacCB9tDaw">GPT-4o 介绍</a>：OpenAI 春季更新 – 2024 年 5 月 13 日星期一直播。介绍 GPT-4o、ChatGPT 更新等。</li><li><a href="https://www.youtube.com/watch?v=3eq84KrdTWY">Llama 3 微调入门指南（包含 16k, 32k,... 上下文）</a>：在这个分步教程中学习如何使用 Unsloth 轻松微调 Meta 强大的新 Llama 3 语言模型。我们涵盖了：* Llama 3 的 8B 和... 概述</li><li><a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens：用于快速 kernel 的 Tile 原语</a>：用于快速 kernel 的 Tile 原语。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/lilacai/lilac">GitHub - lilacai/lilac：为 LLMs 策划更好的数据</a>：为 LLMs 策划更好的数据。通过在 GitHub 上创建账号为 lilacai/lilac 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth：微调 Llama 3, Mistral & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7204">slaren 移除 convert-lora-to-ggml.py · Pull Request #7204 · ggerganov/llama.cpp</a>：模型转换期间张量排列等更改使得从 HF PEFT 转换 LoRA 变得不可靠，因此为了避免混淆，我认为最好完全删除此功能，直到该功能...</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth：微调 Llama 3, Mistral & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp：C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1239230576335257611)** (15 条消息🔥):

- **OpenAI 新的 Model Spec 讨论与社区 Q&A**：OpenAI 发布了新的 [Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html)，旨在改进其模型在 API 和 ChatGPT 中的表现。请设置提醒，OpenAI CEO Sam Altman 将于太平洋标准时间（PST）今天下午 2 点在 Reddit 的 [Q&A 环节](https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_with_openais_ceo_sam/)回答社区提问。

- **社区对 OpenAI 即将推出的创新的期待**：成员们对 OpenAI 即将推出的创新表达了复杂的情绪，一些成员期待能*重振 AI*，而另一些人则持怀疑态度，担心可能会令人失望。

- **关于 OpenAI 开源策略的辩论**：关于 OpenAI 是否应该发布开源模型的讨论仍在继续。一方认为，如果模型达不到标准，发布模型可能会导致负面舆论；而另一方则认为，即使模型不是突破性的，也能让他们处于有利地位。

- **关于 AI 行业趋势和推测的讨论**：成员们讨论了各种行业趋势，包括对新模型比现有产品好 10 倍这种不太可能的预期，以及如果 Llama 成为 SOTA 后的潜在竞争举措。

- **对 OpenAI 市场地位和战略决策的看法**：尽管有关于 AI 寒冬的传闻，成员们认为 OpenAI 仍处于 AI 行业的顶端。对话还涉及了 OpenAI 关于公开发布模型的决策背后的战略原因，包括之前涉及泄密和要求开放的资助案例。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_with_openais_ceo_sam/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1238775563502751755)** (312 条消息🔥🔥): 

- **量化模型兼容性问题**：一位用户提出了 **quantized models** 与 **TGI** 兼容性的担忧，提到了 HF 专用推理上的 *Sharding Errors*。他们质疑 `.for_inference` 和 TGI 是否互斥，暗示可能需要手动设置推理。[在 GitHub 上阅读更多](https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf)。

- **关于保存和加载模型的困惑**：讨论表明在如何精确保存和加载模型方面存在挑战，特别是关于通过 `model.save_pretrained_merged(...)` 使用 `16bit` 格式以兼容 TGI。简要提到了涉及 **VLLM** 和 **GGUF** 格式的替代方案，但缺乏操作实现的具体指导。

- **GEMMA 模型 Tokenization 问题**：用户讨论了与 GGUF 格式模型相关的 Tokenization 问题；对于 Gemma 的 GGUF，存在**额外的空格**导致 **incorrect tokenization**，建议包括通过手动调整或通过既有的 unsloth 渠道修复 Tokenization。

- **新模型功能所需的澄清和说明**：讨论了关于利用 **LLAMA factory** 进行 70b 模型训练的咨询；同时，出现了关于 **FastLanguageModel usage** 的问题，重点是从本地保存的目录加载。此外，还表达了关于在不增加新基础设施开销的情况下最大化潜力的担忧。

- **寻求复杂建模技术的指导**：用户寻求关于创建能够处理**复杂、多主题对话**的模型的建议，建议范围从在专门数据集上进行 **fine-tuning** 到采用 Prompt Engineering 或开发 Elaborator 模型方法，突显了 chatbot 框架中模型优化的迭代过程。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/blob/d3a33a0dc3cabd3b3c0dba0255fb4919db44e3b5/unsloth/__init__.py#L18">unsloth/unsloth/__init__.py at d3a33a0dc3cabd3b3c0dba0255fb4919db44e3b5 · unslothai/unsloth</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#ollama-guide---unsloth-fastlanguagemodel">Home</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows 上运行了 unsloth (无需 WSL)。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写完，但我现在在用手机...</li><li><a href="https://github.com/unslothai/hyperlearn">GitHub - unslothai/hyperlearn: 2-2000x faster ML algos, 50% less memory usage, works on all hardware - new and old.</a>: 机器学习算法速度提升 2-2000 倍，内存占用减少 50%，适用于所有新旧硬件 - unslothai/hyperlearn</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKk">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=yFfaXG0WsQuE)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.re">Sou Cidadão - Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1239271994239877130)** (1 messages): 

- **用于 Token Classification 的 Llama 微调模型已分享**：Sauravmaheshkar 微调了 **Llama 变体**，并在 [🤗 Hub](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188) 上分享了模型权重。这些模型（包括在 *conll2003* 上使用 LoRA 适配器训练的 `unsloth/llama-2-7b-bnb-4bit`）现在可供社区访问。
- **即将发布的 Llama 微调见解**：一篇博文和配套的 Notebook 将很快在 Weights & Biases 博客上发布，详细介绍这些 Llama 模型的微调过程。这些即将发布的内容将提供额外的见解和实际实现细节。

**提到的链接**: <a href="https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188">LlamaForTokenClassification - SauravMaheshkar 集合</a>: 未找到描述

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1238754156731437087)** (976 messages🔥🔥🔥): 

- **对 SD3 发布日期的质疑**：用户对 Stable Diffusion 3 的发布日期表示怀疑，经常进行幽默的对比并分享表达质疑的 GIF。这种情绪表明，尽管有公司的时间表，但社区中许多人认为 SD3 的发布如同神话一般。

- **ControlNet 和微调讨论**：用户讨论了使用 ControlNet 和 LoRA 执行特定任务（如 Inpainting 和图像中的真实文本集成）的各个方面。一位用户就使用 Krita 手动调整图像内文本的替代方法提供了详细建议。

- **SD 硬件推荐**：关于运行 Stable Diffusion 的硬件（如 AMD RX 6750 XT 和 NVIDIA RTX 4090）效率的讨论，对于高端 GPU 在 SD 任务中是否显著优于旧型号，意见不一。

- **内容创作者寻求建议**：有用户寻求关于微调 Stable Diffusion 以生成特定产品广告的帮助，这表明了 SD 在商业场景中的应用。另一位用户讨论了在生成多张图像时保持角色一致性（character consistency）的需求，并链接到了外部资源以获取进一步帮助。

- **一般查询与协助**：用户寻求技术帮助并分享了使用 Stable Diffusion 的个人轶事，从解决 ComfyUI 等界面中的复制/粘贴问题，到讨论在图像中加入额外细节的放大（upscaling）方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Lewdiculous/Average_Normie_l3_v1_8B-GGUF-IQ-Imatrix">Lewdiculous/Average_Normie_l3_v1_8B-GGUF-IQ-Imatrix · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF">dranger003/c4ai-command-r-v01-iMat.GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-v01">CohereForAI/c4ai-command-r-v01 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=AdQxgvRnfhc">Nikolas Cruz 的堕落 Google 搜索历史</a>：一瞥 Parkland 枪击案凶手坠入互联网黑暗深渊的过程。这就是为什么父母应该监控孩子上网行为的原因。警告：...</li><li><a href="https://www.youtube.com/watch?v=GM-e46xdcUo">Jonathan Frakes 告诉你你错了，持续 47 秒</a>：它从未发生过</li><li><a href="https://github.com/Zuellni/ComfyUI-ExLlama-Nodes?tab=readme-ov-file">GitHub - Zuellni/ComfyUI-ExLlama-Nodes: ComfyUI 的 ExLlamaV2 节点。</a>：ComfyUI 的 ExLlamaV2 节点。通过在 GitHub 上创建账户为 Zuellni/ComfyUI-ExLlama-Nodes 的开发做出贡献。</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF/resolve/main/ggml-c4ai-command-r-v01-q8_0.gguf?download=true">未找到标题</a>：未找到描述</li><li><a href="https://github.com/nullquant/ComfyUI-BrushNet">GitHub - nullquant/ComfyUI-BrushNet: ComfyUI BrushNet 节点</a>：ComfyUI BrushNet 节点。通过在 GitHub 上创建账户为 nullquant/ComfyUI-BrushNet 的开发做出贡献。</li><li><a href="https://github.com/KoboldAI/KoboldAI-Client">GitHub - KoboldAI/KoboldAI-Client</a>：通过在 GitHub 上创建账户为 KoboldAI/KoboldAI-Client 的开发做出贡献。</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: 一种简单的单文件方式，通过 KoboldAI 的 UI 运行各种 GGML 和 GGUF 模型</a>：一种简单的单文件方式，通过 KoboldAI 的 UI 运行各种 GGML 和 GGUF 模型 - LostRuins/koboldcpp</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cg5zky/sd3_release/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/">Stable Diffusion 中的角色一致性 - Cobalt Explorer</a>：更新：07/01 – 更改了模板，以便更容易缩放到 512 或 768 – 更改了 ImageSplitter 脚本使其更易于使用，并添加了 GitHub 链接 – 增加了章节...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1239631044395929685)** (2 条消息): 

- **GPT-4o 向公众发布**：OpenAI 宣布新的旗舰模型 **GPT-4o**，以及浏览、数据分析和记忆等功能，现在已免费向所有人开放，但有一定的限制。欲了解更多信息，请访问 [GPT-4o and More Tools](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/)。

- **Plus 用户获得增强访问权限**：Plus 用户将受益于高达 **5 倍的限制额度**，并能最早体验即将推出的功能，如新的 macOS 桌面应用以及先进的语音和视频功能。

- **推出多模态 GPT-4o**：新的 **GPT-4o** 模型支持跨音频、视觉和文本的实时推理。文本和图像输入即日起可通过 API 和 ChatGPT 使用，语音和视频输入预计将在未来几周内推出。更多信息请见 [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1238835920703193191)** (689 条消息 🔥🔥🔥): 

- **探索 GPT-4 和 GPT-4o 的能力**：用户正在积极测试并比较 GPT-4 和新推出的 GPT-4o 在各种任务中的表现。虽然 GPT-4o 以速度见长，但一些用户认为 GPT-4 在推理方面更胜一筹，并特别提到 GPT-4o 需要更明确的指令才能发挥最佳性能。

- **对语音和摄像头功能的困惑**：用户对实时摄像头共享和语音模式等新功能感到兴奋，但由于这些功能虽然在演示中展示，但尚未对所有用户开放，因此仍存在一些困惑。

- **桌面和移动端应用进展**：用户渴望 ChatGPT 的 macOS 应用上线，并提到 Windows 版本正在开发中。用户正在寻找下载链接和可用性信息，但目前并非所有人都能获取。

- **关于订阅价值的讨论**：随着 GPT-4o 的推出，关于 ChatGPT Plus 等付费订阅价值的讨论正在进行，特别是当 GPT-4o 似乎提供了显著的进步时。

- **对模型记忆和 Token 计数器的关注**：一些用户对 GPT-4o 与旧模型相比的记忆性能表示失望。此外，用户还希望增加 Token 计数器等功能，以便在项目中更好地管理模型交互。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/twerk-dance-dog-funny-cute-gif-19259275">Twerk Dance GIF - Twerk Dance Dog - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8?">Sync codebase · openai/tiktoken@9d01e56</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1238754323366936636)** (126 条消息🔥🔥): 

- **探索 GPT-4o 的输出限制**：澄清了关于 GPT-4o 的 Token 输出限制的困惑。明确了 API 的输出 Token 限制高于最初在 API Playground 中显示的值；GPT-4o 每条消息支持高达 4096 个输出 Token，而不是用户最初遇到的 2048 个。

- **关于自定义 GPTs 使用 GPT-4o 的澄清**：成员们争论了自定义 GPTs 目前是否正在使用新的 GPT-4o 模型。目前已确认自定义 GPTs 尚未运行 GPT-4o 模型，尽管用户对输出差异存在一些困惑。

- **GPT-4o 提升速度与性能**：据分享，GPT-4o 比其前代产品快得多，一些基准测试显示其速度是 GPT-4 的两倍。然而，这种速度提升仅适用于 API，不涉及响应的质量或性质。

- **每个 GPT 的记忆功能及推出状态**：讨论了每个 GPT 记忆功能的推出，提到每个自定义 GPT 将拥有自己独立的记忆库，创建者可能可以切换该功能。然而，目前还没有该功能广泛推出的官方时间表。

- **了解 GPT-4o 发布后的订阅权益**：鉴于许多功能已在免费层级提供，讨论了继续订阅 Plus 的价值。用户权衡了 Plus 目前的收益与未来可能证明订阅费合理的预期增强功能。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1239279732961443841)** (32 条消息🔥): 

- **持续存在的审核过滤器之谜被揭开**：一位成员分享了 **Gemini 1.5** 在未启用安全过滤器的情况下，仍无法处理与“浪漫套餐”相关的请求的问题。他们尝试了各种设置调整但未获成功，并认为可能是提供商端的限制导致了这一约束。

- **AI 中的语法错误或安全设置**：针对 *Gemini 1.5* 出现的问题，有人建议可能是语法错误或安全设置禁用不当。建议在 AI 实验室进行进一步检查，以查明处理特定内容请求时出错的根源。

- **聊天中的闲谈**：两名用户进行了日常问候，未对正在进行的讨论或主题贡献实质性的查询或问题。

- **通过 Python 进行目录和文件操作查询**：一位用户请求了一种以编程方式显示和处理文件的方法，指定使用 Python 任务来创建目录、在不同会话中处理文件，并最终压缩目录并提供下载链接。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1239279732961443841)** (32 条消息🔥): 

_

- **Gemini 1.5 在处理浪漫请求时失败**：一位用户报告了 **Gemini 1.5** 的一个问题，即任何与 "romance package" 相关的查询都会导致持续失败，尽管该应用在其他领域取得了广泛成功。他们表达了挫败感，并尝试了各种解决方案，包括生成新的 API keys、将 blocks 设置为 none 以及调整 temperature 设置，但均未成功。

- **需要审查安全设置**：针对该问题，另一位成员建议检查应用中的安全设置是否被明确关闭，因为如果未定义，可能会默认开启。这可能会拦截与 "romance" 或 "package" 相关的词汇，可能需要对这些设置的管理方式进行更深入的审查。

- **考虑语法和 Google 的角色**：讨论转向了可能的语法错误或用户控制之外的问题，特别是涉及 Google 的系统。有建议在 AI Lab 中测试有问题的提示词以排除语法问题，并暗示可能需要通过 GUI 禁用安全协议。

- **尽管具备专业知识但仍感挫败**：该用户展示了对 OpenAI 产品的大量使用（每月超过 10 亿 tokens）以及对 Gemini 优于 Claude 的偏好，对当前持续存在的问题表达了了解和挫败感。他们公开希望在不久的将来能有所解决，承认自己熟悉这些系统，但仍面临意想不到的挑战。

- **Python 文件处理提示词**：另一位成员发布了一个复杂的 Python 任务，请求协助显示完整的文件树、创建目录、在独立的 Python 会话中管理文件写入，并最终压缩目录，包括提供下载链接的指令。这展示了社区内处理的技术查询的多样性。
  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1239532612515663942)** (2 messages): 

- **关于带有消息追踪功能的 ChatGPT 克隆的咨询**：一位用户表示有兴趣利用 **GPT-3.5** 模型创建一个 **ChatGPT** 克隆，并具有一个独特功能：能够监控组织内用户发送和接收的消息。在此咨询之后，没有提供任何解决方案或进一步的讨论。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

king.of.kings_: 我正努力让 llama 3 70b 在超过 8k tokens 时保持连贯性，哈哈。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1238792963015053333)** (16 messages🔥): 

- **法国极光一瞥**：在 Arvenia（法国奥弗涅）的大都市中心火山区，天空中出现了极光。
- **介绍 MAP-Neo，透明的双语 LLM**：MAP-Neo 是一个透明的双语大语言模型 (LLM)，在 4.5 万亿 tokens 上训练，得到了来自 01.ai 和 wuhan.ai 的社区支持。它在推理和数学等任务中与私有模型的性能相匹配，同时通过共享 checkpoints 和数据集组成等资源确保透明度。[在 Huggingface 上探索 neo 模型](https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04) 和 [GitHub](https://github.com/multimodal-art-projection/MAP-NEO)。
- **历史食谱影响现代游戏**：在角色扮演游戏《天国：拯救》（*Kingdom Come: Deliverance*）中，永久炖菜反映了一种历史烹饪方法，丰富了游戏的真实性，并影响了玩家的日常烹饪实践。
- **通过 RDP 进行软件自动化的挑战**：用户讨论了自动化在 RDP 等远程连接上运行的软件的难度，因为在这种情况下无法直接与软件的 DOM 进行交互。建议的实现方式包括使用 RPA 技术或使用 Frida 等工具进行逆向工程，以便与软件功能进行更直接的交互。
- **YouTube 视频分享**：用户分享了 YouTube 视频供观看，尽管讨论背景中未指明这些视频的具体内容。以下是链接：[paradroid 的视频](https://youtu.be/03eHNJzEYcA?si=DV2OToN0h57W7tkv) 和 [pradeep1148 的视频](https://www.youtube.com/watch?v=KQ-xGVFHDkw)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mother-day-gif-12554356809887397003">Mother Day GIF - Mother day - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="http://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - a m-a-p Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/m-a-p/Matrix">m-a-p/Matrix · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/multimodal-art-projection/MAP-NEO">GitHub - multimodal-art-projection/MAP-NEO</a>: 通过在 GitHub 上创建账户来为 multimodal-art-projection/MAP-NEO 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1238852402564825168)** (6 messages): 

- **探索多向神经运算 (Exploring Multidirectional Neural Operation)**: 一篇新论文讨论了人工神经网络优化多向值传播的潜力，这模仿了某些生物神经元的行为。这种方法可能允许神经元模型处理整个联合分布 (joint distributions)，从而增强网络处理复杂依赖关系的方式。[点击此处阅读摘要](https://arxiv.org/abs/2405.05097)。

- **React 应用模拟 Taskmaster 剧集**: 一位成员开发了一个 React 应用程序，使用状态机 (state machine) 模式模拟 **Taskmaster** 游戏节目剧集。每个剧集组件管理不同的阶段，并与 LLM 交互以生成内容，尽管对于格式错误的输出需要手动重试。[探索 GitHub 项目](https://github.com/LEXNY/Taskmaster-LLM/blob/main/src/App.js)。

- **神经网络中的分层相关性重构 (Hierarchical Correlation Reconstruction)**: 提到的研究引入了分层相关性重构 (HCR) 来建模神经元。这可能会显著改变神经网络建模和传播复杂统计依赖关系的方式。[在 Hugging Face 上查看资源](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)。

- **使用 Mistral 7B 生成高级知识图谱**: 利用 **Mistral 7B instruct v 0.2** 模型和 llama-cpp-agent 框架，创建了一个详细的工业军事综合体知识图谱。该框架支持多种服务器类型，并促进与大语言模型 (LLM) 的结构化交互。[在 GitHub 上查看框架](https://github.com/Maximilian-Winter/llama-cpp-agent)。

- **深入探讨 OpenAI 的视听 AI 转型**: 一份详细的分析显示，OpenAI 可能正在通过直接将音频映射到音频以及将视频流传输到 Transformer，向实时多模态 AI 交互迈进。这些技术可能涉及复杂的系统优化、YouTube 对话等数据源，以及潜在的专有流媒体编解码器，旨在与 iOS 等设备进行更紧密的集成。[在 Twitter 上阅读完整讨论](https://twitter.com/drjimfan/status/1790089671365767313)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.05097">Biology-inspired joint distribution neurons based on Hierarchical Correlation Reconstruction allowing for multidirectional neural networks</a>: 流行的人工神经网络 (ANN) 针对单向值传播优化参数，假设某些猜测的参数化类型，如多层感知器 (MLP) 或 Kolmogorov-Arnold Net...</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - a 01-ai Collection</a>: 未找到描述</li><li><a href="https://github.com/Maximilian-Winter/llama-cpp-agent">GitHub - Maximilian-Winter/llama-cpp-agent: llama-cpp-agent 框架是一个旨在简化与大语言模型 (LLM) 交互的工具。允许用户与 LLM 模型聊天，执行结构化函数调用并获取结构化输出。也适用于未针对 JSON 输出和函数调用进行微调的模型。</a>: llama-cpp-agent 框架是一个旨在简化与大语言模型 (LLM) 交互的工具。允许用户与 LLM 模型聊天，执行结构化函数调用并获取结构化...</li><li><a href="https://github.com/LEXNY/Taskmaster-LLM/blob/main/src/App.js">Taskmaster-LLM/src/App.js at main · LEXNY/Taskmaster-LLM</a>: 通过在 GitHub 上创建账户来为 LEXNY/Taskmaster-LLM 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1238772042384408587)** (741 messages🔥🔥🔥):

<ul>
  <li><strong>GPT-4o 发布引发讨论</strong>：讨论围绕新的 GPT-4o 更新展开。一些用户赞赏其在编程性能上的提升，而另一些用户则对其速度和 Token 输出限制感到失望。</li>
  <li><strong>新的 Tokenization 更新与效率</strong>：注意到 GPT-4o 更新后的 Tokenizer 能更好地支持多种语言，但以牺牲效率为代价。尽管如此，Token 限制仍然是一个突出问题，输出上限被限制在 2048。</li>
  <li><strong>编程与数学性能</strong>：据报道，GPT-4o 在编程任务中表现出色，并具有更好的推理能力，这表明它比前代产品有所改进。讨论这些能力的用户似乎发现它在逻辑推理和解决数学问题方面表现更好。</li>
  <li><strong>对模型可访问性与定价的担忧</strong>：一个普遍的担忧是模型的可访问性和价格点，特别是语音集成可能具有革命性，但也仅限于那些负担得起的人。</li>
  <li><strong>厌倦与乐观的观点</strong>：讨论描绘了用户之间的分歧，一些人批评 OpenAI 垄断 Chatbot 和 AI 市场的做法，而另一些人则认为这些更新具有重大价值，特别是在实时语言模型实现和集成方面。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.composio.dev/gpt-4-function-calling-example/">提高 GPT 4 Function Calling 准确性</a>：加入我们的 Discord 社区并查看我们正在构建的内容！我们刚刚发布了博客的第 2 部分，对比了 gpt-4-turbo vs opus vs haiku vs sonnet。GPT Function Calling 简介...</li><li><a href="https://x.com/wenhuchen/status/1789685187804029285?s=46">来自 Wenhu Chen (@WenhuChen) 的推文</a>：重大新闻！遇见我们最强大的全开源 7B-LLM Neo。我们在 MAP-Neo 发布了其 4.7T 预训练数据 Matrix 和整个代码库！1. Neo-7B 击败了现有的全开源模型如 OL...</li><li><a href="https://arxiv.org/abs/2404.18824">评估大型语言模型中的基准测试泄露 (Benchmarking Benchmark Leakage)</a>：随着预训练数据使用的扩大，基准测试数据集泄露现象日益突出，由于不透明的训练过程以及经常未披露的包含内容而加剧...</li><li><a href="https://huggingface.co/mradermacher/llama-3-cat-8b-instruct-GGUF">mradermacher/llama-3-cat-8b-instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/refuelai/Llama-3-Refueled">refuelai/Llama-3-Refueled · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-1.5-34B-Chat/blob/main/ggml-model-Q4_K_M.gguf">ggml-model-Q4_K_M.gguf · 01-ai/Yi-1.5-34B-Chat at main</a>：未找到描述</li><li><a href="https://www.cambioml.com">cambioml</a>：未找到描述</li><li><a href="https://tenor.com/view/cats-animals-reaction-wow-surprised-gif-20914356">猫咪动物 GIF - 猫咪动物反应 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://oobabooga.github.io/benchmark.html">oobabooga 基准测试</a>：未找到描述</li><li><a href="https://www.youtube.com/live/DQacCB9tDaw">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一直播。介绍 GPT-4o，ChatGPT 的更新等。</li><li><a href="https://github.com/interstellarninja/MeeseeksAI/blob/2399588acdee06cff4af04ca091b1ab5c71580b8/src/agents.py#L72-L83">MeeseeksAI/src/agents.py 位于 2399588acdee06cff4af04ca091b1ab5c71580b8 · interstellarninja/MeeseeksAI</a>：一个使用 mermaid 图表编排 AI Agent 的框架 - interstellarninja/MeeseeksAI</li><li><a href="https://github.com/Potatooff/Le-Potato">GitHub - Potatooff/Le-Potato: 简单、优雅的 LLM 聊天推理</a>：简单、优雅的 LLM 聊天推理。通过在 GitHub 上创建账号来为 Potatooff/Le-Potato 的开发做出贡献。</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://github.com/interstellarninja/MeeseeksAI">GitHub - interstellarninja/MeeseeksAI: 一个使用 mermaid 图表编排 AI Agent 的框架</a>：一个使用 mermaid 图表编排 AI Agent 的框架 - interstellarninja/MeeseeksAI</li><li><a href="https://x.com/willdepue/status/1790078289023062255?s=46&t=bL0EKkuCqv4FWSLQ7lV-2w">来自 will depue (@willdepue) 的推文</a>：我认为人们误解了 gpt-4o。它不是一个带有语音或图像附件的文本模型。它是一个原生的多模态 Token 输入、多模态 Token 输出模型。你想让它说话快一点？...</li><li><a href="https://github.com/huggingface/transformers/pull/30621">Rocketknight1 提交的为 Function Calling 和 RAG 提供 Chat Template 支持 · Pull Request #30621 · huggingface/transformers</a>：此 PR 更新了我们对聊天模板的支持，以涵盖工具使用和 RAG 使用场景。具体而言，它执行了以下操作：定义了推荐的工具使用 JSON schema 规范，添加了工具和文档...</li><li><a href="https://huggingface.co/TheSkullery/llama-3-cat-8b-instruct-v1">TheSkullery/llama-3-cat-8b-instruct-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/almost_digital/status/1788877760120692994">来自 Johan Nordberg (@almost_digital) 的推文</a>：我在 1 月份加入了 @elevenlabsio，和 @flavioschneide 一起工作非常愉快！这个是由单个文本提示词“rap about never stopping to learn”（关于永不停止学习的饶舌歌曲）生成的，包含歌词...</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1238877292395102268)** (48 条消息🔥):

- **在大多数架构中 MoE 仅限于 FFN 层**：讨论者确认，在大多数架构中，Mixture of Experts (MoE) 中的专家**仅为前馈网络 (FFN)** 层。虽然已经探索过将 Attention 模块作为专家，但并非标准做法。

- **将 Autoregressive 和 Diffusion 模型与 MoE 集成的兴趣**：讨论了将 Autoregressive 模型（擅长文本生成）与 Diffusion 模型（擅长图像任务）结合的概念，利用 MoE 结构潜在地增强多模态模型的性能。虽然存在怀疑态度，但理论上的集成可能会提升模型能力。

- **Prompt 模板及其对 LLM 性能的影响**：对话明确了使用大语言模型训练时特定的 Prompt 格式会极大地影响其可靠性。例如，**Hermes** 使用 chatml 格式，而其他模型可能更倾向于 **Alpaca Prompt Format**。

- **处理模型中的不安全行为输入**：提到模型中内置的安全措施和“人生教训”式回答可以通过系统级 Prompt 进行干预以修改响应。建议了规避拒绝并诱导更直接回答的技术，并引用了在线资源如 [Handling Refusals](https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.pdf)。

- **Llama3 和 Axolotl 系统的微调挑战**：一位用户分享了尝试使用 Axolotl 系统和 dolphin-2.9 数据集微调 Llama3 模型时的挑战与解决方案。讨论了 CUDA 错误以及更新 flash-attn 等包的必要性，指向了针对技术瓶颈的社区驱动解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2305.18295">RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths</a>：文本到图像生成最近取得了显著成就。我们引入了一种名为 RAPHAEL 的文本条件图像 Diffusion 模型，用于生成高度艺术化的图像，能够准确地……</li><li><a href="https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb">ortho_cookbook.ipynb · failspy/llama-3-70B-Instruct-abliterated at main</a>：未找到描述</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment &amp; Theory Scholars Program - 2023-24 冬季队列中的一部分，由……共同指导。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1238780733619965952)** (5 messages): 

- **揭秘 ChatQA：对话式 QA 的创新者**：最近的一篇 [Arxiv 论文](https://arxiv.org/abs/2401.10225)介绍了 **ChatQA**，这是一个 QA 模型系列，通过使用两阶段指令微调和高性价比的稠密检索器（dense retriever），在对话准确性上超越了 **GPT-4**。ChatQA-70B 在多个数据集上的得分为 54.14，而 GPT-4 为 53.90，提供了一个无需 GPT 模型合成数据的更便宜的替代方案。

- **IBM/RedHat 的新颖训练方法**：IBM 和 RedHat 正在合作开展一个[新项目](https://github.com/instructlab)，通过使用更大的模型生成合成数据集而无需完全重新训练，从而创新 LLM 训练。该过程在 GitHub 上有详细说明，采用分类法（taxonomies）进行课程构建，并利用 **Granite** 和 **Merlinite** 等强大的 LLM。

- **引入增强模型训练框架**：深入探讨 IBM/RedHat 的项目，揭示了一个为 LLM 安排的信息丰富过程。贡献者可以每周格式化并提交数据，经过策划后，这些数据将被整合到 Granite 和 Merlinite 等模型中，以增量方式增强其知识库。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10225">ChatQA: Building GPT-4 Level Conversational QA Models</a>：在这项工作中，我们介绍了 ChatQA，一个获得 GPT-4 级别准确率的对话式问答 (QA) 模型系列。具体来说，我们提出了一种两阶段指令微调方法，可以……</li><li><a href="https://github.com/instructlab">InstructLab</a>：InstructLab 有 10 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1238765655940005959)** (22 messages🔥): 

- **WorldSim 被誉为顶级商业模拟器**：成员们讨论了 WorldSim 作为商业和初创公司模拟器的有效性，**proprietary** 强调了其作为全能模拟器的强大实力。

- **加入 WebSim 冒险**：成员们积极参与 WebSim AI 模拟，分享了特定模拟的链接，如 [hidden catgirl](https://websim.ai/c/grXqLcCAxEGNz3TyH)，并邀请他人在 [join WebSim](https://websim.ai/c/B8MJwg44rDhdQmJYB) 建立基地。

- **Twitter 上关于模拟游戏的讨论热潮**：分享了展示对模拟游戏热情的 Twitter 帖子链接，表明了更广泛的社区兴趣。示例推文可以在 [这里](https://twitter.com/sebkrier/status/1789314810754081014) 和 [这里](https://twitter.com/sawyerhood/status/1789322914539676028) 找到。

- **WorldSim 中报告的技术挑战**：记录了 WorldSim 功能方面的问题，包括上下文保留（context retention）、命令执行和界面 Bug。

- **哲学与 WorldSim 沙龙提案**：一位成员提议组建一个哲学和 WebSim WorldSim 聊天小组，衡量大家对沙龙式协作讨论的兴趣。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/B8MJwg44rDhdQmJYB">generative.ink/chat/</a>: 未找到描述</li><li><a href="https://websim.ai/c/grXqLcCAxEGNz3TyH">generative.ink/chat/</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1238778471841533992)** (94 messages🔥🔥): 

<ul>
  <li><strong>Substack 上的新基础设施讨论：</strong> 一位来自新加坡的成员发起了关于 AI Agent 潜在新基础设施的对话，并在 Substack 上分享了他们的笔记。欢迎感兴趣的各方讨论并就这些新兴服务进行协作。[在此查看早期笔记](https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9)。</li>
  <li><strong>Falcon 2 模型发布：</strong> Falcon 2 LLM 已发布，被描述为多语言和多模态模型。根据独立验证，其性能优于 Meta 的 Llama 3 和 Google 的 Gemma 7B 等竞争对手。计划进行更多增强，如 'Mixture of Experts'。[探索 Falcon 2 的功能](https://falconllm.tii.ae/falcon-2.html)。</li>
  <li><strong>关于 GPT-4o 的讨论：</strong> 针对新发布的 GPT-4o 进行了活跃的讨论和更新，包括各种技术推测、容量以及应用基准测试。社区热衷于探索 GPT-4o 在 API 访问中的特性及其性能改进。[了解更多关于 GPT-4o 的信息](https://openai.com/index/hello-gpt-4o/)。</li>
  <li><strong>AI 安全与职业机会：</strong> 围绕将 AI 安全作为潜在职业路径的对话展开，讨论了其在 AI 与网络安全交叉领域中的现实性和价值。参与者建议了自主研究的重要性，并指向了 RSA Conference 等资源以供进一步探索。</li>
  <li><strong>OpenELM 训练协作：</strong> 一项关于使用 PyTorch/MPS 从头开始训练 OpenELM 模型的协作努力公告引起了关注。该倡议对社区协作开放，旨在通过增量添加数据集进行迭代训练。[探索 OpenELM 项目协作](https://github.com/openai/openelm)。</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://falconllm.tii.ae/falcon-2.html">Falcon LLM</a>: 生成式 AI 模型正让我们能够创造通往充满无限可能的激动人心的未来的创新路径——唯一的限制就是想象力。</li><li><a href="https://x.com/Karmedge/status/1790084650582397118">Robert Lukoszko — e/acc (@Karmedge) 的推文</a>: 我有 80% 的把握确定 OpenAI 使用了一个极低延迟、低质量的模型，在不到 200ms 内读出前 4 个词，然后继续使用 GPT-4o 模型。只需注意，大多数句子都以 “Sure...” 开头。</li><li><a href="https://x.com/drjimfan/status/1790089671365767313?s=46&t=90xQ8sGy63D2OtiaoGJuww">Jim Fan (@DrJimFan) 的推文</a>: 我知道你的时间线现在被“疯狂、HER、你错过的 10 个功能、我们回来了”之类的词汇堆砌淹没了。坐下，冷静点。<喘气> 像 Mark 在演示中那样深呼吸...</li><li><a href="https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63D2OtiaoGJuww">Mark Cummins (@mark_cummins) 的推文</a>: Llama 3 是在 15 万亿个 Token（11 万亿个单词）上训练的。这非常庞大——大约是人类语言学习所需量的 100,000 倍。</li><li><a href="https://x.com/gdb/status/1790077263708340386">Greg Brockman (@gdb) 的推文</a>: GPT-4o 还可以生成音频、文本和图像输出的任何组合，这带来了一些我们仍在探索的有趣新功能。参见例如“功能探索”部分...</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991">lmsys.org (@lmsysorg) 的推文</a>: 突发新闻 —— gpt2-chatbots 的结果现已公布！gpt2-chatbots 刚刚飙升至榜首，以显著差距（约 50 Elo）超越了所有模型。它已成为 Arena 中有史以来最强的模型...</li><li><a href="https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9">Something to do something</a>: 点击阅读由 sweekiat 发布的 Substack 出版物 Something to do something。两年前推出。</li><li><a href="https://www.bloomberg.com/news/articles/2024-05-11/apple-closes-in-on-deal-with-openai-to-put-chatgpt-on-iphone">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://x.com/andykreed/status/1790082413428629843">tweet davidson 🍞 (@andykreed) 的推文</a>: ChatGPT 的声音……很性感？？？</li><li><a href="https://x.com/mark_cummins/status/1788949945795424522">Mark Cummins (@mark_cummins) 的推文</a>: 接下来是代码。代码是一种非常重要的文本类型，其数量令我惊讶。公开代码有 0.75 万亿个 Token。有史以来编写的代码总量可能高达 20 万亿，尽管其中大部分是私有的...</li><li><a href="https://www.latent.space/s/university">AI for Engineers | Latent Space | swyx &amp; Alessio | Substack</a>: 为准 AI 工程师准备的为期 7 天的基础课程，与 Noah Hein 共同开发。尚未上线——我们已完成 5/7。注册以便在发布时获取！点击阅读 Substack 出版物 Latent Space...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">Sync codebase · openai/tiktoken@9d01e56</a>: 未找到描述</li><li><a href="https://x.com/karmedge/status/1790084650582397118?s=46&t=90xQ8sGy63D2OtiaoGJuww">Robert Lukoszko — e/acc (@Karmedge) 的推文</a>: 我有 80% 的把握确定 OpenAI 使用了一个极低延迟、低质量的模型，在不到 200ms 内读出前 4 个词，然后继续使用 GPT-4o 模型。只需注意，大多数句子都以 “Sure...” 开头。</li><li><a href="https://x.com/juberti/status/1790126140784259439">Justin Uberti (@juberti) 的推文</a>: 有机会尝试了来自 us-central 的 GPT-4o API，文本生成速度非常快。与 http://thefastest.ai 相比，此性能是 GPT-4-Turbo TPS 的 5 倍，与许多 Llama-3-8b 部署相似...</li><li><a href="https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63">Mark Cummins (@mark_cummins) 的推文</a>: Llama 3 是在 15 万亿个 Token（11 万亿个单词）上训练的。这非常庞大——大约是人类语言学习所需量的 100,000 倍。</li><li><a href="https://x.com/jacobcolling/status/1790073742514663866?s=46&t=90xQ8sGy63D2OtiaoGJuww">Jake Colling (@JacobColling) 的推文</a>: @simonw @OpenAI 使用模型 `gpt-4o` 似乎适用于我的 API 访问</li><li><a href="https://x.com/blader/status/1790088659053719736?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Siqi Chen (@blader) 的推文</a>: 回想起来，这将证明是迄今为止最被低估的 OpenAI 活动。OpenAI 在 GPT-4o 中随手丢出了文本到 3D 渲染功能，甚至都没提到它（更多内容见下文 👇🏼）</li><li><a href="https://news.ycombinator.com/item?id=40344302">Falcon 2 | Hacker News</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1239418270302339115)** (1 条消息):

- **OpenAI 活动预热已安排**：OpenAI 活动的观影派对计划于明天（5 月 13 日）上午 9:30 开始。请在活动开始前半小时加入 [Discord 频道](https://discord.gg/Z7V4NDGZ?event=1238918257046458368) 参加预热。

**提到的链接**：<a href="https://discord.gg/Z7V4NDGZ?event=1238918257046458368">加入 Latent Space (原名 /dev/invest) Discord 服务器！</a>：在 Discord 上查看 Latent Space (原名 /dev/invest) 社区——与 3747 名其他成员一起交流，享受免费的语音和文字聊天。

---

**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1239616941677609064)** (710 条消息🔥🔥🔥)：

- **OpenAI 春季发布会观影派对启动**：Discord 社区成员聚集在一起观看并讨论 OpenAI 春季发布会，并邀请成员分享他们的预测。然而，一些人在直播过程中遇到了音频问题，导致有人建议重新连接。

- **针对 Apple 和 GPT-4o 的技术推测**：在活动观影派对期间，讨论转向了 Apple 的技术策略以及 Google 关于 iOS 18 谈判的潜在影响。人们开始推测 Apple 是否有足够的能力将足够大的模型整合到其设备中。

- **GPT-4o 免费开放成为焦点**：在一个令人惊讶的揭晓中，官方披露 [GPT-4o 现在可以免费使用](https://x.com/LiamFedus/status/1790064963966370209)，这是前沿模型从未有过的举措。这一公告引发了 Twitter 上的讨论，特别关注了模型集成策略，包括对移动端集成的潜在影响。

- **活动直播故障与技术困扰**：观众对直播过程中的技术困难表示沮丧，包括视频卡顿和音频问题。这些干扰导致成员之间不断进行调整和反馈，试图解决问题以获得更流畅的观看体验。

- **社区参与实用性与预测性对话**：随着活动的展开，成员们分享了不间断观看活动的实用链接，并就 GPT-4o 的功能、未来及其在日常设备和平台中的集成展开了讨论。这些对话反映了对活动中揭晓的 AI 当前及未来应用的兴奋与怀疑。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/0xkarmatic/status/1790079694043320756">来自 Karma (@0xkarmatic) 的推文</a>："一个 ASR 模型，一个 LLM，一个 TTS 模型……你明白了吗？这不是三个独立的模型：这是一个模型，我们称之为 gpt-4o。" 引用 Andrej Karpathy (@karpathy) 的话，它们是 r...</li><li><a href="https://twitch.tv/yikesawjeez,">Twitch</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Mechanical_Turk">Mechanical Turk - 维基百科</a>: 未找到描述</li><li><a href="https://blog.samaltman.com/gpt-4o">GPT-4o</a>: 在我们今天的发布中，我想强调两件事。首先，我们使命的一个关键部分是免费（或以极优的价格）将能力极强的 AI 工具交到人们手中。我...</li><li><a href="https://x.com/imjaredz/status/1790074937119482094?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Jared Zoneraich (@imjaredz) 的推文</a>: gpt-4o 彻底击败了 gpt-4-turbo。速度极快，答案似乎也更好。也很喜欢 @OpenAI 的分屏 playground 视图</li><li><a href="https://x.com/oliviergodement/status/1790070151980666982?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Olivier Godement (@oliviergodement) 的推文</a>: 我没怎么发过关于 @OpenAI 发布的消息，但我想分享一些关于 GPT-4o 的思考，因为我已经很久没有感到如此震撼了。</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">介绍 GPT-4o</a>: OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://x.com/gdb/status/1790071008499544518?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Greg Brockman (@gdb) 的推文</a>: 介绍 GPT-4o，我们的新模型，它可以实时跨文本、音频和视频进行推理。它非常多才多艺，玩起来很有趣，是迈向更自然的人机交互形式的一步……</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw&ab_channel=OpenAI">介绍 GPT-4o</a>: OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://x.com/brad_agi/status/1790073505658114069">来自 Brad (@brad_agi) 的推文</a>: 便宜 50% 甚至都没有竞争力。来源：https://artificialanalysis.ai/</li><li><a href="https://x.com/bdougieyo/status/1790071113420079329?s=46">来自网络上的 bdougie (@bdougieYO) 的推文</a>: ChatGPT 说我看起来心情不错。</li><li><a href="https://x.com/LiamFedus/status/1790064963966370209">来自 William Fedus (@LiamFedus) 的推文</a>: GPT-4o 是我们最新的 state-of-the-art 前沿模型。我们一直在 LMSys arena 上以 im-also-a-good-gpt2-chatbot 的身份测试一个版本 🙂。这是它的表现。</li><li><a href="https://x.com/sama/status/1790065541262032904">来自 Sam Altman (@sama) 的推文</a>: 它对所有 ChatGPT 用户开放，包括免费计划！到目前为止，GPT-4 级别的模型仅对支付月费的用户开放。这对我们的使命很重要；我们希望...</li><li><a href="https://t.co/B5iqOKm06j">GitHub - BasedHardware/OpenGlass: 将任何眼镜变成 AI 驱动的智能眼镜</a>: 将任何眼镜变成 AI 驱动的智能眼镜。通过在 GitHub 上创建账户来为 BasedHardware/OpenGlass 的开发做出贡献。</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>: 未找到描述</li><li><a href="https://x.com/gdb/status/1790079398625808837">来自 Greg Brockman (@gdb) 的推文</a>: 我们还显著提升了非英语语言的性能，包括改进了 tokenizer 以更好地压缩其中许多语言：
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1238757309254078475)** (674 条消息🔥🔥🔥): 

- **GPT-4o 引发兴奋与猜测**：GPT-4o 的推出引起了用户的极大兴趣，讨论集中在其提升的速度、更低的成本和多模态能力上。用户对其可能集成到 Perplexity 中充满热情，热切期待它的加入，并猜测其先进功能对当前 AI 应用的影响。

- **Opus 使用限制令用户沮丧**：多位用户对 Perplexity 在 Claude 3 Opus 等强大模型上的每日使用限制表示不满，反映出对更宽松访问条款的强烈需求。这些限制导致一些人考虑替代平台，尽管 Perplexity 产品的独特优势让许多人保持忠诚。

- **AI 采用中的隐私顾虑**：在选择 AI 服务时，讨论强调了用户对优先考虑隐私的平台的强烈偏好。尽管在使用基于云的 AI 时确保完全隐私存在固有挑战，但用户主张选择那些在保护用户数据方面做出显著努力的供应商。

- **Perplexity 的多模型优势与用户偏好**：Perplexity 利用包括 ChatGPT 和 Claude 3 Opus 在内的多个 AI 模型的价值得到了强调，用户非常欣赏能够根据任务需求在不同模型之间切换的能力。这种灵活性与其他可能提供较少选项或需要更多操作才能导航的平台形成了对比。

- **技术讨论显示了多样化的用户群体和需求**：用户围绕 Context Window 大小和 AI 模型实现细节等话题展开技术讨论，这表明该社区对 AI 的用途非常广泛，从关于每日限制的日常询问到对特定 AI 功能的深入探索。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/inafried/status/1790083063374033046">来自 Ina Fried (@inafried) 的推文</a>：我也确认了一些消息。1) 出现在基准测试网站上的神秘 GPT2-chatbot 是 GPT-4o。2) OpenAI 首先为 Mac 推出了桌面版，因为“我们只是优先...”</li><li><a href="https://gpt-tokenizer.dev/">gpt-tokenizer playground</a>：未找到描述</li><li><a href="https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/">不仅仅是一个 OpenAI 套壳：Perplexity 转向开源</a>：Perplexity CEO Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://youtu.be/MirzFk_DSiI?feature=shared">两个 GPT-4o 互动并唱歌</a>：向 GPT-4o 问好，这是我们全新的旗舰模型，可以实时跨音频、视觉和文本进行推理。在此了解更多：https://www.openai.com/index/hello-...</li><li><a href="https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky">两个 GPT-4o 互动并唱歌</a>：向 GPT-4o 问好，这是我们全新的旗舰模型，可以实时跨音频、视觉和文本进行推理。在此了解更多：https://www.openai.com/index/hello-...</li><li><a href="https://www.youtube.com/live/DQacCB9tDaw?feature=shared">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://fxtwitter.com/mckaywrigley/status/1790088880919818332?s=46">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>：这个演示太疯狂了。一名学生通过新的 ChatGPT + GPT-4o 分享他们的 iPad 屏幕，AI 与他们交谈并协助他们进行*实时*学习。想象一下把这个交给世界上的每个学生...</li><li><a href="https://www.yeschat.ai/pricing">YesChat.ai 价格计划</a>：未找到描述</li><li><a href="https://azure.microsoft.com/en-us/blog/introducing-gpt-4o-openais-new-flagship-multimodal-model-now-in-preview-on-azure/">介绍 GPT-4o：OpenAI 的新旗舰多模态模型现已在 Azure 上提供预览 | Microsoft Azure 博客</a>：OpenAI 与 Microsoft 合作发布了 GPT-4o，这是一款在文本、视觉和音频能力方面具有突破性的多模态模型。了解更多。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1239038758477631649)** (21 条消息🔥):

- **探索 AI 职业生涯**：Alexandr Yarats 讨论了他从 Yandex 到 Google，再到目前担任 [Perplexity AI](https://www.unite.ai/alexandr-yarats-head-of-search-at-perplexity-interview-series/) 搜索负责人的职业历程。他的经历强调了技术行业中充满挑战但收获颇丰的道路，最终他专注于开发 AI 驱动的搜索引擎。
- **Perplexity AI 平台上的多样化查询**：用户在 Perplexity AI 上分享了各种搜索内容，涵盖了从 [Eurovision 2024](https://www.perplexity.ai/search/Eurovision-2024-LN.Prd19Sju6dGjlw7HByw) 到 [Bernoulli's fallacy](https://www.perplexity.ai/search/Explain-Bernoullis-fallacy-TGhbdqjbQWSqxHWvaWUJJQ#0) 等主题。每个链接都指向特定的查询结果，展示了该平台在满足不同信息需求方面的广泛用途。
- **启用可共享线程的提醒**：Perplexity AI 提醒用户确保其线程是可共享的，并在 [Discord 消息](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)中提供了链接的逐步指南。这表明该平台注重社区协作和信息共享。

**提到的链接**：<a href="https://www.unite.ai/alexandr-yarats-head-of-search-at-perplexity-interview-series/">Alexandr Yarats, Head of Search at Perplexity &#8211; 访谈系列</a>：Alexandr Yarats 是 Perplexity AI 的搜索负责人。他于 2017 年在 Yandex 开始职业生涯，同时在 Yandex School of Data Analysis 学习。最初的几年虽然艰辛，但收获颇丰……

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1238969127981547663)** (4 条消息): 

- **Perplexity 教程请求**：一位用户请求关于 Perplexity 的教程。另一位用户回复了一个深度解析教程的链接，但提供的链接重定向到了一个失效的 Discord 路径，显示占位符为 <<<null>>>。

- **使用中的表情符号**：来自同一用户的两条不同消息包含了表情符号，一个标记为 `wlcm`，另一个为 `gem_2`，可能表示非英语对话（特别是俄语）中的不同语境或情感。

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1238758307267874906)** (389 条消息🔥🔥): 

- **开源 LLM 和平台的探索**：讨论了类似于 **llamma3** 的开源大语言模型 (LLMs) 以及更好的替代方案（如 **Mistral**）。有建议称可以使用 **you.com** 等平台来尝试这些模型。

- **挖掘会议转录的潜力**：一位用户分享了他们按发言人变更对会议转录进行 chunking 并创建 embeddings 的策略，但在交互之间面临低相似度得分的问题。社区被请求提供更好的解决方案或见解。

- **修改禁用安全功能的 Diffusion Pipelines**：进行了代码共享，其中修改了 **StableDiffusionPipeline** 和 **DiffusionPipeline**，通过将 `safety_checker` 设置为 `None` 并将 `requires_safety_checker` 设置为 `False` 来禁用安全检查。

- **对就业和协作项目的兴趣**：一位成员表达了与团队合作的兴趣，并提到了在前端和区块链开发以及 AI 方面的经验。

- **观察到的优化和性能讨论**：提出了关于优化深度学习模型的各种咨询，包括关于 batch sizes 和 GPU 资源使用的建议，以最大限度地提高计算效率。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.andrewng.org/">no title found</a>: 无描述</li><li><a href="https://huggingface.co/Gryphe/Tiamat-8b-1.2-Llama-3-DPO">Gryphe/Tiamat-8b-1.2-Llama-3-DPO · Hugging Face</a>: 无描述</li><li><a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://huggingface.co/blog/train-dgx-cloud">在 NVIDIA DGX Cloud 上使用 H100 GPU 轻松训练模型</a>: 无描述</li><li><a href="https://tenor.com/view/will-smith-chris-rock-jada-pinkett-smith-oscars2022-smack-gif-25234614">Will Smith Chris Rock GIF - Will Smith Chris Rock Jada Pinkett Smith - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/excuse-me-hands-up-woah-funny-face-gif-14275996">Excuse Me Hands Up GIF - Excuse Me Hands Up Woah - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/DQacCB9tDaw?t=4239">介绍 GPT-4o</a>: OpenAI 春季更新 —— 2024年5月13日星期一现场直播。介绍 GPT-4o、ChatGPT 更新等。</li><li><a href="https://www.eurekai.tech">EurekAI</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=QEaBAZQCtwE&ab_channel=AssemblyAI">15 分钟上手 Hugging Face | Transformers, Pipeline, Tokenizer, Models</a>: 学习如何在 15 分钟内上手 Hugging Face 和 Transformers 库！了解关于 Pipelines, Models, Tokenizers, PyTorch 和 TensorFlow 的一切...</li><li><a href="https://www.tiktok.com/t/ZTLV3ShEp/">TikTok - 充实你的一天</a>: 无描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1239043199100784752)** (3 条消息): 

- **探索 GenAI 用户界面创新**：分享的一段 YouTube 视频提供了关于生成式 AI 在医疗应用中用户体验的见解，具有多模态交互和包括检索增强生成 (RAG) 在内的未来计划。重点功能包括具有成本意识的模型访问和容器化应用。[在此观看视频](https://www.youtube.com/watch?v=UgVPzSSCjr8)。

- **解码神经网络初始化**：来自 deeplearning.ai 的资源直观地解释了在神经网络中正确进行参数初始化的重要性，以防止梯度爆炸和梯度消失问题。要探索神经网络训练中的详细步骤和方法，[请访问 deeplearning.ai 的指南](https://www.deeplearning.ai/ai-notes/initialization/index.html)。

- **使用 Jax 和 TPU 推进图像生成**：一位用户讨论了他们的项目，该项目使用 Jax 库 Equinox 将 Visual AutoRegressive (VAR) 模型的 PyTorch 实现适配到 TPU 加速，并指出在多个指标上优于传统模型。有关 VAR 方法及其在图像生成中优越性的详细信息，请参阅[这篇研究论文](https://arxiv.org/abs/2404.02905)以及 [GitHub](https://github.com/patrick-kidger/equinox) 上的 Equinox 库。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.deeplearning.ai/ai-notes/initialization/index.html">AI 笔记：初始化神经网络 - deeplearning.ai</a>: 在这篇文章中，我们将解释如何有效地初始化神经网络参数。初始化对深度神经网络训练的收敛性有显著影响...</li><li><a href="https://arxiv.org/abs/2404.02905">视觉自回归建模：通过次尺度预测实现可扩展的图像生成</a>: 我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“次尺度预测”或“次分辨率...</li><li><a href="https://github.com/patrick-kidger/equinox">GitHub - patrick-kidger/equinox: JAX 中优雅且易于使用的神经网络 + 科学计算。https://docs.kidger.site/equinox/</a>: JAX 中优雅且易于使用的神经网络 + 科学计算。https://docs.kidger.site/equinox/ - patrick-kidger/equinox
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1238994655732043816)** (10 条消息🔥): 

- **针对智能手机优化的 Phi-3**：Phi-3 在智能手机等低功耗设备上表现出了极具前景的性能。详细信息可见于 [Marah Abdin](https://arxiv.org/search/cs?searchtype=author&query=Abdin,+M) 等多位作者的综合研究，可在 [arXiv 此处](https://arxiv.org/abs/2404.14219)查阅。

- **深入探索深度学习**：理解深度学习基础的新资源 [UDL Book](https://udlbook.github.io/udlbook/) 被推荐为一个特别有用的教育工具。

- **通过 AI Notes 更好地进行初始化**：[deeplearning.ai 提供了关于神经网络权重初始化的见解](https://www.deeplearning.ai/ai-notes/initialization/index.html)，以应对梯度爆炸/消失等问题，这对于有效的模型训练至关重要。

- **可视化 LLM 效果**：通过[此链接](https://bbycroft.net/llm)探索新的交互式可视化工具，以更好地理解大语言模型 (LLM)。

- **利用 RL 重塑抗体开发**：描述了一种在抗体开发中使用强化学习 (RL) 的创新方法，提高了靶向治疗的潜力。更多信息可以在这篇 [ScienceDirect 文章](https://www.sciencedirect.com/science/article/pii/S167202292300092X)中找到。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a>: 我们介绍了 phi-3-mini，这是一个拥有 38 亿参数、在 3.3 万亿 token 上训练的语言模型，其在学术基准和内部测试中的整体表现可与 ... 媲美。</li><li><a href="https://udlbook.github.io/udlbook/">Understanding Deep Learning</a>: 未找到描述</li><li><a href="https://www.deeplearning.ai/ai-notes/initialization/index.html">AI Notes: Initializing neural networks - deeplearning.ai</a>: 在这篇文章中，我们将解释如何有效地初始化神经网络参数。初始化对深度神经网络训练的收敛性有显著影响...</li><li><a href="https://3d-diffusion-policy.github.io/">3D Diffusion Policy</a>: 本文介绍了 3D Diffusion Policy (DP3)，这是一种掌握多种视觉运动任务的视觉模仿学习算法。</li><li><a href="https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1238995268410671177)** (7 messages): 

- **多语言 AI 故事讲述者发布**：发布了一个支持英语、马来语、中文和泰米尔语的新型 AI 驱动故事讲述者。请在 [alkisah-ai by ikmalsaid](https://huggingface.co/spaces/ikmalsaid/alkisah-ai) 查看。

- **古兰经海报 AI 工具**：开发了一个根据《古兰经》经文创建精美海报的 AI 工具，但由于没有活动，该 Space 目前处于非活跃状态。更多信息请见[此处](https://huggingface.co/spaces/ikmalsaid/kalam-ai)。

- **OCR 工具包介绍**：开发了一个通用的 OCR 框架，允许与 DocTr、PaddleOCR 和 Google Cloud Vision 等不同 OCR 技术集成。开发者在 [ocrtoolkit on GitHub](https://github.com/ajkdrag/ocrtoolkit) 分享了 GitHub 仓库以供社区贡献。

- **为 Token Classification 微调 Llama 变体**：Llama 模型变体已针对 Token Classification 进行了微调，并上传至 🤗 Model Hub，重点关注 `conll2003` 数据集。在 [LlamaForTokenClassification by SauravMaheshkar](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188) 查看微调模型集合。

- **构建 AI 驱动的 OCR 质量分类器**：采用了一种使用小型 Encoder 对文档质量进行分类的新方法，事实证明这对于识别 PleIAs 数据集中的噪声或干净文本非常有效。在 [OCR Quality Classifiers by pszemraj](https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3) 探索这些模型。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3">OCR Quality Classifiers - a pszemraj Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ikmalsaid/kalam-ai">Kalam AI - a Hugging Face Space by ikmalsaid</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188">LlamaForTokenClassification - a SauravMaheshkar Collection</a>: 未找到描述</li><li><a href="https://youtu.be/B1F94RKksR8?si=WPSmpyjiByCHaTAQ">How To Create Your Own AI Discord Chat Bot With Web Search</a>: Git Repo:https://github.com/ssimpson91/newsChan 你将需要以下软件包；NodeJS v. 18，Python 3.10 或更高版本。在你的终端运行这些命令以...</li><li><a href="https://github.com/ajkdrag/ocrtoolkit">GitHub - ajkdrag/ocrtoolkit: Experiment and integrate with different OCR frameworks seamlessly</a>: 无缝地实验并集成不同的 OCR 框架 - ajkdrag/ocrtoolkit
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1239543496457584752)** (2 messages): 

- **介绍 YOCO，一种新颖的架构**：一位成员分享了 [arXiv 上的一篇精彩阅读](https://arxiv.org/abs/2405.05254)，关于 **YOCO**，这是一种用于 LLM 的新型 decoder-decoder 架构，它能高效地对键值对（key-value pairs）进行一次性缓存。这种设计显著降低了 GPU 显存需求，同时保持了全局注意力能力，并加速了 prefill 阶段。

**提到的链接**：<a href="https://arxiv.org/abs/2405.05254">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>：我们为 LLM 引入了一种 decoder-decoder 架构 YOCO，它仅对键值对进行一次缓存。它由两个组件组成，即堆叠在 self-decoder 之上的 cross-decoder。...

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1238780335379185756)** (6 messages): 

- **使用 UNet 探索类别条件扩散**：一位成员分享了他们使用 UNet 进行类别条件扩散（class condition diffusion）的实验，并为 latent diffusion models 寻求类似资源。他们引用了 [HuggingFace 上的 UNet 扩散课程](https://huggingface.co/learn/diffusion-course/unit2/3)。

- **在自定义数据集上应用 YOLOv1 遇到困难**：一位用户表示在出于教育目的从零开始在自定义数据集上实现 YOLOv1 时遇到困难。他们希望修复实现中的问题，其中还涉及一个带有 ResNet 骨干网络和单个 bbox 的 mini YOLO 版本。

- **Stable Diffusion 实验得到回应**：另一位成员强调了他们在 Stable Diffusion 方面的工作，引用了关于使用 [HuggingFace 的 diffusers 库](https://huggingface.co/blog/stable_diffusion)的资源。他们指向了一份关于使用 Stable Diffusion 模型自定义图像生成 pipeline 的详细说明。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/">Hugging Face - Learn</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/stable_diffusion">Stable Diffusion with 🧨 Diffusers</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1238769279378391061)** (7 messages): 

- **会议记录分块（Chunking）的挑战**：一位成员正在寻求关于如何高效对**会议记录进行分块**的建议，以便利用 LLM 获取可操作的见解，旨在通过减少 LLM 调用次数来优化成本。他们提到当前方法在分块之间产生的相似度得分较低（约 0.45）。

- **关于文本块检索的建议**：讨论指出不应期望连续消息之间有**高相似度得分**；建议的方法包括获取相关文本的**相邻分块（neighboring chunks）**以保持上下文。

- **部分成员不希望被私信**：一位参与者明确表示他们**不接受直接消息（DMs）**，强调公开讨论。

- **评估检索器组件的方法**：建议**准备一个黄金数据集（gold dataset）**，并使用不同的配置（如 chunk size 和 overlap）对检索组件进行基准测试，推荐使用 ***mean reciprocal rank*** 作为指标。

- **将自定义 Tokenizer 与 Transformer 集成时的困难**：一位成员分享了在将**自定义 Hugging Face tokenizer** 与 transformer 集成时遇到的问题，参考了 2021 年的 Hugging Face 教程（[查看视频](https://www.youtube.com/watch?v=MR8tZm5ViWU)）。他们报告了根据 ChatGPT 建议提示格式不匹配的错误。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1239087326135717899)** (14 messages🔥):

- **深入探讨 Diffusion Model 细节**：一位用户询问了关于 Diffusion Model 内在原理的资源。推荐内容包括 DDPM 和 DDIM 学术论文，以及实用资源，如关于实现 Stable Diffusion 的 [Fast.ai 在线课程](https://course.fast.ai/Lessons/part2.html)，以及 O'Reilly 出版的《[Hands-On Generative AI with Python](https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)》一书，以深入理解生成模型。

- **本地推理引擎入门**：一位用户询问如何为 Command-R+ 开发本地推理引擎，但被引导至另一个更专业的论坛寻求见解，该论坛可能更侧重于 NLP 策略。

- **自定义图像 Inpainting 使用指南**：为了协助对个人图像进行 Inpainting（局部重绘），分享了 [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint)链接，详细介绍了使用模型 Checkpoints 编辑图像特定区域的过程。

- **macOS 安装问题排查**：一位用户在 macOS 上安装 `sadtalker` 时遇到问题。尽管他们被引导去 Google 搜索类似问题，但他们发现这些建议在没有解决问题的情况下并无帮助。

- **创建个性化图像数据集**：一位用户寻求关于将自己的图像数据集用于 AI 模型的建议，随后分享了一份关于创建和构建用于模型训练的个人图像数据集的 [Hugging Face 指南](https://huggingface.co/docs/diffusers/main/en/training/create_dataset)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/create_dataset">Create a dataset for training</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint">Inpainting</a>：未找到描述</li><li><a href="https://course.fast.ai/Lessons/part2.html">Practical Deep Learning for Coders - Part 2 overview</a>：通过 fastai 和 PyTorch 学习深度学习，2022</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266">[Bug]: ModuleNotFoundError: No module named &#39;torchvision.transforms.functional_tensor&#39; torchvision 0.17 promblem · Issue #13985 · AUTOMATIC1111/stable-diffusion-webui</a>：是否存在此问题的现有 Issue？我已搜索现有 Issue 并检查了最近的构建/提交。发生了什么？ModuleNotFoundError: No module named &#39;torchvision.transforms.functiona...</li><li><a href="https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/">Hands-On Generative AI with Transformers and Diffusion Models</a>：在这本实用的实战指南中，学习如何使用 AI 生成媒体技术来创作新颖的图像或音乐。数据科学家和软件工程师将理解最先进的生成技术...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1238747906857766994)** (185 条消息🔥🔥): 

- **了解多 GPU 配置性能**：一位用户分享了使用多 GPU 时性能缓慢的问题，怀疑是 PCIe 3.0 带宽的潜在影响。经过讨论和排查，确定主板是瓶颈；升级到兼容 PCIe 4.0 的主板后解决了该问题。

- **探索 LM Studio 的远程配置**：讨论围绕配置 LM Studio Server 的 IP 地址以进行远程访问展开。会议澄清了服务器会绑定到宿主机上的所有接口，将 "localhost" 替换为机器的 IP 即可解决远程访问问题。

- **LM Studio 中的错误处理**：多位用户遇到了由于内存不足导致的 "Failed to load model" 错误消息。建议包括关闭 GPU offload，或验证硬件规格是否满足运行大型模型的要求。

- **LMS 在 Linux 服务器上的部署挑战**：一位用户在 Linux 服务器上因 AppImage 的 FUSE 设置问题而面临安装 LMS 的困难。另一位用户提供了一个在 Ubuntu Server 24.04 上奏效的解决方案，强调了社区在解决问题中的作用。

- **本地模型管理的 GPU 显存要求**：通过各种讨论，强调了有效使用 LLM 通常需要大量的 VRAM，建议至少 8GB+ 显存来运行像 GPT-4 级别的模型。这突显了选择合适硬件以避免性能瓶颈的重要性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=EPYCD8#Specifications">未找到标题</a>: 未找到描述</li><li><a href="https://www.asrockrack.com/general/productdetail.asp?Model=ROMED8-2T/BCM">未找到标题</a>: 未找到描述</li><li><a href="https://downforeveryoneorjustme.com/chat.lmsys.org?proto=https">Chat.lmsys.org 宕机了？当前问题与状态 - DownFor</a>: Chat.lmsys.org 无法加载？或者在使用 Chat.lmsys.org 时遇到问题？在此检查状态并报告任何问题！</li><li><a href="https://tenor.com/view/boo-boo-this-man-gif-4868055">Boo Boo This Man GIF - Boo Boo This Man - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">介绍 GPT-4o</a>: OpenAI 春季更新 —— 2024年5月13日星期一现场直播。介绍 GPT-4o，ChatGPT 的更新等。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1238759943537168394)** (92 条消息🔥🔥): 

- **澄清本地模型能力**：一位成员询问关于在配置中等的个人笔记本电脑上运行专用编码模型的问题，并收到了回复，澄清 LM Studio 可能无法在该硬件设置上支持此类高速本地模型。其他成员也指出了各种集成和设置的局限性以及潜在的解决方法。

- **探索文本转图像工具**：关于将文本转换为图像的讨论重点介绍了 Stable Diffusion、comfyUI 和 Automatic1111 等工具。成员们分享了链接以及使用不同工具的经验，建议对于初学者来说，复杂度较低的软件可能更有益。

- **了解 Hugging Face 上的模型版本与微调**：多位成员讨论了模型在 Hugging Face 等平台上的版本控制和微调方式，指出阅读模型卡（model cards）以了解特定数据集和所涉及训练细节的重要性。特别关注了量化以及通过微调引入的变体。

- **量化模型以获得更好性能**：几位成员讨论了量化各种模型的细节和好处，特别是 Yi-1.5 模型系列。分享了特定量化版本的链接，以及提高模型性能和兼容特定硬件限制的使用技巧。

- **处理模型约束与上下文长度**：多位用户解决了与模型上下文长度和预算限制相关的问题，这些问题影响了模型的选择。特别提到了不同 GPU 容量带来的限制，以及运行更大模型所需的权衡。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/YorkieOH10/Yi-1.5-9B-Chat-Q8_0-GGUF">YorkieOH10/Yi-1.5-9B-Chat-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/YorkieOH10/Yi-1.5-6B-Chat-Q8_0-GGUF">YorkieOH10/Yi-1.5-6B-Chat-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/failspy/kappa-3-phi-abliterated">failspy/kappa-3-phi-abliterated · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-1.5-9B-Chat">01-ai/Yi-1.5-9B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NikolayKozloff/Meta-Llama-3-8B-Instruct-bf16-correct-pre-tokenizer-and-EOS-token-Q8_0-Q6_k-Q4_K_M-GGUF">NikolayKozloff/Meta-Llama-3-8B-Instruct-bf16-correct-pre-tokenizer-and-EOS-token-Q8_0-Q6_k-Q4_K_M-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1238893918850908222)** (4 条消息): 

- **探索开源安装程序选项**：一位成员分享了他们使用开源安装程序替代方案 **Innosetup** 和 **Nullsoft Installer** 的积极经验，并提到过去曾成功使用过这两者。
- **Debian 上 Starcoder2 的性能异常**：一位在 Debian 12 上尝试 **starcoder2-15b-instruct-v0.1-IQ4_XS.gguf** 的用户注意到，初始结果尚可，但随着尝试更多任务，开始出现重复响应和跑题回答等问题。
- **关于模型使用的澄清**：针对上述问题的回复强调，像 **starcoder2** 这样的 instruct 模型是针对单步指令优化的，可能不适合多轮对话，从而解释了所遇到的一些异常情况。
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1238854849395822603)** (7 条消息):

- **Playground Mode Requires GPU**: 对话明确了 **Playground mode** 仅限 GPU，无法仅在 **RAM + CPU** 上有效运行，尤其是在只有 4GB VRAM 的情况下。

- **Warning Against Misleading Links**: 发布了关于 **shortlink**（短链接）可能指向不安全或无关站点的警告，将其标记为具有潜在欺骗性。

- **LLM Training Inquiry**: 一位成员询问了是否可以使用教学大纲中的 **Word files** 来训练 **language model**，以方便进行问答环节。

**Link mentioned**: <a href="https://tenor.com/view/shoo-go-away-johnny-depp-captain-jack-sparrow-gif-4877675">Shoo Go Away GIF - Shoo Go Away Johnny Depp - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1238773262884798565)** (106 messages🔥🔥): 

- **Llama 3 Model Performance Queries**: 讨论集中在 **Llama 3** 模型在各种硬件上的运行性能。用户分享了不同配置下的经验，记录了如 *0.6 tok/s* 的速率，并咨询了使用 CPU 和 RAM 以提高效率的可能性。

- **Hardware Bottlenecks and Optimization**: 围绕硬件组件带来的限制展开了关键讨论。用户交流了关于 VRAM 容量的知识，特别是对比 **Tesla P100** 与 **GTX 1060** 的 GPU 性能时。由于潜在的 CUDA 版本不匹配等问题，观察到了预期性能与实际性能之间的差异。

- **Optimizing Model Load on Limited Resources**: 用户探索了 Offloading 技术，以应对低 VRAM (2GB) 硬件的限制。重点强调了正确设置 Offload 到 GPU 的模型层数，以防止错误并确保模型运行更顺畅。

- **Comparative Discussion of Running LLMs on CPU Versus GPU**: 分享的经验突显了仅在 CPU 上运行 LLM 时显著的性能下降。讨论了具体的 Token 速率，例如通过调整 CPU 设置实现的 *3.2 tok/s* 到 *3.5 tok/s* 的提升。

- **Exploration of Tools and Settings in LMStudio and JAN Under Various Operating Systems**: 用户讨论了界面元素，例如用于调节模型加载到 GPU 与 RAM 比例的滑块。一致的建议是为模型使用更高的 VRAM，以防止加载失败和响应生成不足。
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1238759906635681822)** (12 messages🔥): 

- **CodeQwen1.5: A Surprisingly Powerful 7b Model**: 推荐了一款名为 **CodeQwen1.5** 的 7b 模型，认为其在编程方面非常高效，表现优于 **deepseek coder**。它采用 4b 量化，占用空间为 4.18 GB，非常适合 RTX 3050 6GB GPU 配置。

- **Explore Coding Models on Huggingface**: 对于好奇不同模型在编程方面表现的用户，**Huggingface leaderboard** 提供了详尽的列表。感兴趣的用户可以探索各种模型，尤其是 7b 或更小的模型，[查看编程排行榜](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)。

- **Just Bug Fixes and a Small Update**: 最新版本主要解决了 Bug 修复，并包含了一个名为 **llama.cpp** 的更新。此特定更新中未添加新功能。

- **Beware of Suspicious Links**: 用户应保持警惕，因为某些帖子可能包含可疑链接，这些链接可能会产生广告或推荐收入，例如使用 goo.gle 缩短的链接。

- **Community Interaction and Moderation**: 社区积极参与帖子互动，指出有时会规避自动审核的潜在垃圾信息。成员们通过标记异常活动，为维护频道的完整性做出贡献。

**Link mentioned**: <a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models Leaderboard - a Hugging Face Space by bigcode</a>: 未找到描述

  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1238795889171103775)** (4 messages): 

- **Request for MemGPT Expertise**: 一位成员寻求熟悉 **MemGPT** 的专家的个人协助，特别是针对项目相关的问题。
- **Attempted Help and Clarification**: 另一位成员提供了帮助，并提到他们在使用 **Kobold** 集成 MemGPT 方面的经验，但随后澄清他们尚未在所讨论的特定 LM 环境中成功实现。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1238798948538515466)** (2 messages):

- **GPU 升级完成**：一位成员以 700 欧元的价格购买了一块 **RX 7900 XT**，他们认为这块显卡提供的性能足以满足其需求。
- **运行更大模型的建议**：另一位成员建议，新购买的 **RX 7900 XT** 可以处理更大的模型，例如 **Command-R+** 或 **YI-1.5（量化变体）**。
  

---


**LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1238893873972117755)** (4 messages): 

- **连接 LM Studio 到 OpenInterpreter 时的困惑**：一位成员在尝试将 **LM Studio** 连接到 **OpenInterpreter** 时表示困惑，发现无论服务器是否连接，错误信息都没有区别。他们最初的提问比较模糊，随后澄清是尝试连接到一个名为 "open interperter zero one" 的特定设置。
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1239407328483213364)** (1 messages): 

- **新 Yi 模型发布**：LM Studio 社区在其 Huggingface 页面上发布了新的 **Yi 模型**，其中包括一个非常值得关注的 **34B** 版本，非常适合 24GB 显存的显卡。这些模型通过 **imatrix** 进行了增强以获得卓越的质量，并提供多种尺寸，详细信息请参见 [Huggingface](https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF)。

- **模型详情与可用性**：每个 Yi 模型（如 **6B, 9B** 和 **34B**）都在大规模语料库上进行了预训练，并针对多样化数据进行了微调。完整的描述和访问链接可直接在 [Huggingface 页面](https://huggingface.co/lmstudio-community)上找到。

- **提供量化版本**：**Bartowski** 基于 `llama.cpp` 版本 [b2854](https://github.com/ggerganov/llama.cpp/releases/tag/b2854) 为这些模型提供了 GGUF 量化，确保在特定硬件配置上的高效使用。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF">lmstudio-community/Yi-1.5-34B-Chat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Yi-1.5-9B-Chat-GGUF">lmstudio-community/Yi-1.5-9B-Chat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF">lmstudio-community/Yi-1.5-6B-Chat-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1238998189626101862)** (19 messages🔥): 

- **跨厂商 GPU 查询缺乏满意答复**：一位成员询问如何在 **LM Studio** 中为 **llama.cpp** 实现 Vulkan 后端，特别是希望利用跨厂商的 GPU。虽然简要讨论了使用后端 API 等替代建议，但未能解决该问题。

- **介绍 LM Studio CLI**：**LM Studio CLI (`lms`)** 作为 [LM Studio 0.2.22](https://lmstudio.ai) 的新功能被重点介绍，允许用户加载/卸载模型、启动/停止 API 服务器以及检查原始 LLM 输入和输出。详情和源代码托管在 [GitHub](https://github.com/lmstudio-ai/lms) 上，[LM Studio 博客](https://lmstudio.ai/blog/lms)提供了全面的安装指南。

- **Vulkan 后端兼容性问题仍未解决**：尽管引入了 LM Studio CLI，用户在将 Vulkan 后端的 **llama.cpp** 与 **LM Studio** 集成时仍遇到困难。目前在 LM Studio 框架内似乎没有直接的解决方案。

- **寻求无头 (Headless) LM Studio 安装方案**：另一位用户在 Linux 云服务器上安装 **LM Studio** 时遇到挑战，原因是 AppImage 中的 "FUSE" 设置存在问题。社区建议使用 **Ollama** 或从基础编译 **llama.cpp** 作为无头设置的变通方法。

- **一般互动与参与**：成员们积极互动，寻求有关安装和潜在新功能的各种技术协助，展现了 LM Studio 开发者社区内协作和解决问题的氛围。

**提及的链接**：<a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio's companion cli tool | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了 lms 的第一个版本 —— LM Studio 的配套命令行工具。

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1238747934376464386)** (2 messages): 

- **JetMoE 8B 遭遇服务中断**：[JetMoE 8B 免费模型](https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free)目前因上游过载而宕机。在另行通知前，对此模型的所有请求都将返回 **502 错误**。

- **两个多模态模型现已上线**：OpenRouter 宣布推出两个多模态模型：[GPT-4o](https://openrouter.ai/models/openai/gpt-4o) 和 [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b)。这些模型可以通过其平台用于 AI 应用。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free>)">JetMoE 8B (由 jetmoe 提供) | OpenRouter</a>：Jet MoE 由来自学术界到行业资深人士的广泛团队开发，是 MIT、Princeton、IBM、Lepton 和 MyShell 的共同努力成果。该模型完全开源，并经过...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o)">OpenAI: GPT-4o (由 openai 提供) | OpenRouter</a>：GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://openrouter.ai/models/liuhaotian/llava-yi-34b)">LLaVA v1.6 34B (由 liuhaotian 提供) | OpenRouter</a>：LLaVA Yi 34B 是一个开源模型，通过在多模态指令遵循数据上对 LLM 进行微调训练而成。它是一个基于 Transformer 架构的自回归语言模型。Base LLM: [Nou...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1239279202767867924)** (2 条消息): 

- **OpenRouter API Watcher 发布**：推出了一款名为 **OpenRouter API Watcher** 的工具，它可以高效地跟踪 OpenRouter 模型列表的变化，并将其存储在 SQLite 数据库中。它具有简单的 Web 界面和用于更新的 RSS 订阅源，通过每小时仅查询一次 OpenRouter API 来最小化开销。点击[此处](https://orw.karleo.net/)查看演示。

- **Rubik's AI 招募 Beta 测试人员**：**Rubik's AI** 推出了先进的研究助手和搜索引擎，邀请用户参与 Beta 测试，并提供两个月的免费高级访问权限。此高级优惠包括访问 **Claude 3 Opus、GPT-4 Turbo、Mistral Large** 等模型，有望大幅增强研究能力。感兴趣的参与者可以进一步探索并在此处[注册](https://rubiks.ai/)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://orw.karleo.net/">OpenRouter API Watcher</a>：OpenRouter API Watcher 监控 OpenRouter 模型的变化并将这些变化存储在 SQLite 数据库中。它每小时通过 API 查询一次模型列表。</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手 & 搜索引擎</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1238747703710584863)** (254 条消息🔥🔥): 

- **Jetmoe 缺乏在线访问权限**：确认 Jetmoe 没有在线访问权限，被描述为适用于学术研究。

- **反欺诈更新引发质疑**：围绕反欺诈措施的讨论凸显了对以安全为名收集个人数据的担忧。批评集中在某些支付处理器要求的账单地址等额外信息如何被用于识别欺诈交易。通常使用 [Stripe](https://stripe.com/) 等提供商来验证和评估交易风险。

- **讨论 OpenRouter 的人员限制**：有人指出 OpenRouter 仅由一个 3 人的小团队维护，因此需要依赖激进的反欺诈措施来减少运营干扰。

- **探索 OpenRouter 对 Embedding 模型的支持**：关于 OpenRouter 支持 Embedding 模型的可能性正在讨论中；然而，目前还没有该功能的固定路线图，团队目前专注于后端改进。

- **征集用于创建角色（Personas）的高级 WebUI**：有人询问是否有一种 WebUI 能够创建多个可定制的角色或 Agent 进行交互，并建议使用 BigAGI 或更名后的 OpenWebUI，但据报道现有平台并不能完全满足这些需求。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.01.ai/">零一万物-AI2.0大模型技术和应用的全球公司（01.AI）</a>: 未找到描述</li><li><a href="https://docs.openwebui.com/tutorial/openai">OpenAI API 端点 | Open WebUI</a>: 在本教程中，我们将演示如何使用环境变量配置多个 OpenAI（或兼容的）API 端点。此设置允许你轻松地在不同的 API 提供商之间切换...</li><li><a href="https://claudeai.uk/can-claude-read-pdf/">Claude 能读取 PDF 吗？ [2023] - Claude Ai</a>: Claude 能读取 PDF 吗？PDF（便携式文档格式）文件是我们日常生活中经常遇到的一种常见文档类型。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cq927y/yi">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cq927y/yi15_202405/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://stripe.com/">Stripe | 互联网金融基础设施</a>: Stripe 为各种规模的企业提供线上和线下支付处理及金融解决方案。通过一套 API 和无代码工具，实现收款、付款及财务流程自动化...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1238892203837886504)** (65 messages🔥🔥): 

- **探索 Mojo 中的隐式变体**：讨论了 Mojo 可能使用管道操作符引入隐式变体，但目前尚未确定。参与者参考了 Python 的 PEP 604，建议 Mojo 采用类似方法，详见[此处](https://peps.python.org/pep-0604/)。

- **公共 Docker 镜像中的 Nightly 构建**：有人询问了关于将 Mojo 编译器 nightly 构建推送到公共 Docker 仓库的政策；然而，目前尚未提供明确政策。建议使用 `modular auth examples` 作为变通方案，以绕过构建镜像时的网站登录。

- **Mojo 中的模式匹配与 If-Else 语句**：展开了关于 Mojo 中模式匹配实现和效用的详细讨论。参与者将其与传统的 if-else 语句进行了比较，指出虽然模式匹配可以做到穷尽且更安全，但也需要一种针对穷尽检查的特定设计心态。

- **关于 Mojo 与 Rust 编译器复杂性的讨论**：对 Mojo 和 Rust 的编译器进行了比较，强调了 Mojo 直接的方法，这使得开发者可以更多地关注编码，而不是处理文档或编译器的复杂性。尽管 Rust 提供了强大的系统设计和自动向量化能力，但其复杂性被认为是一个显著的学习曲线。

- **对 SQL、ORM 与编程语言的看法**：SQL 相对于 ORM 的直观性，以及 Rust 等语言中与编译器的交互，引发了关于易用性与严格系统要求之间平衡的讨论。参与者在处理每种技术时表达了不同程度的舒适感和效率。

**提到的链接**：<a href="https://peps.python.org/pep-0604/">PEP 604 – 允许将联合类型写为 X | Y | peps.python.org</a>: 未找到描述

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1790046377613144201>
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1239603493745197056)** (1 messages): 

- **Modular 新视频提醒**：Modular 发布了一个新视频！点击[此处](https://www.youtube.com/watch?v=9ag0fPMmYPQ)观看。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1238886742702952458)** (85 messages🔥🔥): 

- **Mojo 解引用辩论**：一名成员讨论了 Mojo 中的解引用语法，提议从 '[]' 转向 C++ 风格的 '*'，这引发了辩论。反方观点强调了当前 Mojo 语法的简洁性和类 Python 特性——`p[i]`、带默认参数的 `p[]` 以及像 `p[].field` 这样的后缀组合。

- **Mojo 中的迭代器与 Yield**：由于缺乏原生 `yield` 支持，一位开发者探索了通过手动管理迭代器状态在 Mojo 中实现类似 `yield` 的迭代器行为。他们遇到了特定的类型错误，并讨论了扩展对多种可迭代结构的支持，指出缺乏参数化 Trait 是一个限制。

- **Tree Sitter 语法贡献**：一位成员分享了为 Mojo 创建 Tree Sitter 语法分支的经历，该分支目前已在 Helix 和 Zed 等编辑器中运行。这一进展引起了社区其他成员的兴趣，他们计划在 Neovim 等更多环境中进行测试。

- **Mojo 基准测试讨论**：社区探讨了 Mojo 中基准测试的细微差别，讨论了如何存储基准测试以及目前是否可以对内存使用情况进行基准测试。此外，还分享了一个关于短字符串优化（short string optimization）的近期基准测试链接，对比了 Mojo 中的 `InlinedString` 和 `String` 类型。

- **理解 Mojo 中的所有权 (Ownership)**：分享了一个详细介绍 Mojo 中所有权的新视频，旨在加深用户的理解。几位 Python 开发者反思了从 Python 到 Mojo 的所有权和内存管理的关联与转换，建议提供更多对比示例，以便初学者理解这些概念。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/dtype/DType#is_floating_point">DType | Modular Docs</a>：表示 DType 并提供相关操作方法。</li><li><a href="https://doc.rust-lang.org/nomicon/subtyping.html">Subtyping and Variance - The Rustonomicon</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYPQ">Mojo🔥: a deep dive on ownership with Chris Lattner</a>：深入了解 Mojo 中的所有权，与 Modular CEO Chris Lattner 进行深度探讨。如果您有任何问题，请务必加入我们友好的...</li><li><a href="https://github.com/modularml/mojo/issues/2467#issuecomment-2106263163">[Feature Request] Unify SSO between `InlinedString` and `String` type · Issue #2467 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。您的请求是什么？我们目前有 https://docs.modular.com/mojo/stdlib...</li><li><a href="https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil">Python Mutable Defaults Are The Source of All Evil - Florimond Manca</a>：如何防止一个可能导致严重 Bug 并浪费大家时间的常见 Python 错误。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1238795315365285908)** (1 messages): 

- **MoString GitHub 仓库发布**：创建了一个名为 [MoString](https://github.com/dorjeduck/mostring) 的新 GitHub 仓库，用于探索 Mojo 中 **StringBuilder 理念** 的变体。该仓库包含一个新的 `optimize_memory` 方法，可以高效地将内存分配减少到所需水平。

- **征集社区贡献**：MoString 的创建者正邀请社区贡献各种实现，以确定哪些最适合并入 Mojo 标准库。这一举措被视为增强 Mojo 能力的社区实验。

**Link mentioned**: <a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>：Mojo 中 StringBuilder 理念的变体。通过创建账号参与 dorjeduck/mostring 的开发。

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1238944586349412433)** (64 messages🔥🔥): 

- **Mojo Nightly 版本引入自动直接提交 (Direct Commits)**：`mojo` 框架的最新更新带来了 Nightly 构建，它会自动将内部合并的提交直接推送到 `nightly` 分支，开启了持续开发的新篇章。[这是包含工作流超时调整详情的 PR](https://github.com/modularml/mojo/pull/2644)，旨在更好地管理 Ubuntu 测试挂起问题。

- **Mojo List 操作中的内存管理问题**：关于 Mojo 中 `List` 内存管理策略的讨论指出了性能低下的问题。一项修改 `extend` 方法中内存预分配方式（模仿 `append` 方法的方法）的提议，在特定基准测试中显示出 2000 倍的加速。

- **影响显示的重大 GitHub Actions Bug**：GitHub Actions 存在一个严重问题，即任务即使已完成仍显示为“待处理 (pending)”。此 Bug 影响了正在进行的工作流的透明度和监控，在 Mojo 最近的提交和 CI 操作中尤为明显。

- **类型空间实例化 (Materialization) 测试**：关于 Mojo 中类型正确实例化的一场详细对话讨论了潜在问题，例如在类型转换期间未能正确处理内存指针。这导致了测试失败，并建议对处理方法进行修订。

- **崩溃报告和 Mojo 扩展建议**：讨论了与 Mojo 中处理复杂嵌套类型相关的崩溃报告，并提出了改进类型生命周期管理的建议，以避免在执行多维数组深拷贝等操作时出现段错误 (segmentation faults)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration#usage-limits),">使用限制、计费和管理 - GitHub 文档</a>：未找到描述</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo：将 Andrjey Karpathy 的 minbpe 移植到 Mojo</a>：将 Andrjey Karpathy 的 minbpe 移植到 Mojo。通过在 GitHub 上创建账户来为 dorjeduck/minbpe.mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2644">[CI] 为工作流添加超时设置，由 JoeLoser 提交 · Pull Request #2644 · modularml/mojo</a>：在 Ubuntu 测试中，由于最近 nightly 版本中的代码 bug（可能在编译器或库中），我们看到了一些非确定性的超时。与其依赖默认的 GitHub 超时设置...</li><li><a href="https://github.com/modularml/mojo/pull/2620#issuecomment-2106054892">[stdlib] 将字符串比较委托给 `StringRef`，由 siitron 提交 · Pull Request #2620 · modularml/mojo</a>：这是 #2409 的后续。StringRef 的字符串比较已实现。StringRef 现在在所有 6 种比较中都使用了 memory.memcmp，希望这个改动没问题。String 和 Stri...</li><li><a href="https://github.com/modularml/mojo/pull/2619">[stdlib] 引入 Hasher 类型及所有必要更改，由 mzaks 提交 · Pull Request #2619 · modularml/mojo</a>：这是一个草案，因为虽然代码可以编译，但有 8 个测试失败了。这可能是由于编译器 bug。错误消息非常晦涩。我没有足够的 "Mojo" ;) 来修复它们。失败的测试...</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/dorjeduck/minbpe.mojo/blob/main/mojobpe/utils/mostring/molist.mojo">minbpe.mojo/mojobpe/utils/mostring/molist.mojo 在 main 分支 · dorjeduck/minbpe.mojo</a>：将 Andrjey Karpathy 的 minbpe 移植到 Mojo。通过在 GitHub 上创建账户来为 dorjeduck/minbpe.mojo 的开发做出贡献。</li><li><a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring：Mojo 中 StringBuilder 构思的变体</a>：Mojo 中 StringBuilder 构思的变体。通过在 GitHub 上创建账户来为 dorjeduck/mostring 的开发做出贡献。</li><li><a href="https://github.com/mzaks/mojo/tree/feature/minimal-example-of-test-crash-for-new-hasher">GitHub - mzaks/mojo 在 feature/minimal-example-of-test-crash-for-new-hasher 分支</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 mzaks/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1238983332910596187)** (5 条消息): 

- **理解 GPU 内存管理**：一位成员讨论了其笔记本电脑 GPU 似乎同时使用专用内存和共享内存的情况，观察到当专用内存耗尽时，CUDA 会访问共享内存，一旦两者都填满，就会导致显存溢出 (OOM) 错误。目前没有关于该主题的进一步细节或资源。

- **注意到使用共享内存时的性能下降**：同一位成员指出，当开始使用共享内存时，性能会出现显著下降，怀疑这可能涉及向 CPU 内存的“卸载 (offloading)”。然而，关于如何验证或管理这种行为，没有提供额外的细节。

- **直接沟通有助于稳定 Discord Stage**：另一位成员成功联系了 Discord 的 CEO，以解决 Discord stage 的稳定性问题，CEO 承诺指派合适的工程师提供协助。这突显了在解决技术挑战时有效利用人际网络的价值。
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1238762803138007051)** (43 条消息🔥): 

- **Triton Kernel 探索**：多位用户讨论了 **Triton** kernel，分享了诸如 [attorch](https://github.com/BobMcDear/attorch) 等资源，以及指向 [Triton kernels](https://github.com/zinccat/Awesome-Triton-Kernels) 和 [Triton index](https://github.com/cuda-mode/triton-index) 等仓库的链接。这些提及突显了在优化和扩展 **Triton** 在 AI 开发中的应用方面，正在进行的协作和个人贡献。

- **分享额外的学习资源**：分享了一个 [Triton 讲座](https://www.youtube.com/watch?v=DdTsX6DQk24) 以提供全面指南，并通过 [GitHub 上的第 14 讲](https://github.com/cuda-mode/lectures/tree/main/lecture%2014) 的 GitHub 描述进行展示。这体现了为教育更多用户了解 **Triton** 所做的努力。

- **性能优化讨论**：用户讨论了涉及 kernels 的性能增强，提到了 **GitHub** commit，例如 [调整 Flash Attention 分块大小 (block sizes)](https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac)，其中详细说明了为了获得更好性能而进行的参数调优。这表明一个活跃的社区正致力于完善和提高其代码效率。

- **探索用于 GPU 利用率的新 DSL**：在 [GitHub's ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 引入了一个集成在 **CUDA** 中的名为 **ThunderKittens** 的新 **DSL**。该工具声称可以提高 GPU 利用率并提供代码简洁性，展示了社区在简化 GPU 编程方面的持续创新和兴趣。

- **关于 Triton Kernels 和贡献建议的查询**：用户询问了关于创建教程和向公共仓库贡献 **Triton** kernels 的事宜，并被引导考虑使用个人仓库或 [triton-index](https://github.com/cuda-mode/triton-index) 等平台来分享优化成果。这说明了一个鼓励贡献和知识共享的协作环境。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/bfspector/status/1789749117104894179?s=46&t=ROCrCC19RlrPdFqCtEaiGA">来自 Benjamin F Spector (@bfspector) 的推文</a>：(1/7) 母亲节快乐！我们认为美国母亲们真正想要的是一个只有 100 行代码且速度快 30% 的 Flash Attention 实现，我们很高兴能提供。我们非常激动...</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Triton 实战指南</a>：https://github.com/cuda-mode/lectures/tree/main/lecture%2014</li><li><a href="https://github.com/cuda-mode/triton-index">GitHub - cuda-mode/triton-index: 编录已发布的 Triton kernels。</a>：编录已发布的 Triton kernels。通过在 GitHub 上创建账号为 cuda-mode/triton-index 的开发做出贡献。</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels">GitHub - zinccat/Awesome-Triton-Kernels: 使用 Triton 语言编写的 kernel 集合</a>：使用 Triton 语言编写的 kernel 集合。通过在 GitHub 上创建账号为 zinccat/Awesome-Triton-Kernels 的开发做出贡献。</li><li><a href="https://github.com/BobMcDear/attorch">GitHub - BobMcDear/attorch: PyTorch 神经网络模块的一个子集，使用 OpenAI 的 Triton 用 Python 编写。</a>：PyTorch 神经网络模块的一个子集，使用 OpenAI 的 Triton 用 Python 编写。 - BobMcDear/attorch</li><li><a href="https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac">[TUTORIALS] 调整 flash attention 分块大小 (#3892) · triton-lang/triton@702215e</a>：未找到描述</li><li><a href="https://github.com/ELS-RD/kernl">GitHub - ELS-RD/kernl: Kernl 让你只需一行代码即可在 GPU 上以数倍的速度运行 PyTorch transformer 模型，并且设计为易于修改。</a>：Kernl 让你只需一行代码即可在 GPU 上以数倍的速度运行 PyTorch transformer 模型，并且设计为易于修改。 - ELS-RD/kernl
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1239314338708197458)** (9 条消息🔥): 

- **使用 ThunderKittens 探索高效 AI**：[ThunderKittens GitHub 仓库](https://github.com/HazyResearch/ThunderKittens) 引入了 *用于快速 kernel 的 tile 原语*，旨在简化 AI 的 kernel 构建。HazyResearch 强调了其通过开源贡献优化 AI 计算效率的承诺。

- **深入研究 HazyResearch 的计算优化工作**：[HazyResearch 的一篇综合博客文章](https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk) 详细介绍了他们创建 ThunderKittens 的历程。他们还表达了通过 Based、Monarch Mixer 和 FlashAttention 等项目降低 AI 计算需求的承诺。

- **一个更轻、更快的训练仓库**：HazyResearch 开发了 [nanoGPT-TK](https://github.com/HazyResearch/nanoGPT-TK)，将其推广为训练和微调中型 GPT 的最简单、最快的方法。它具有名为 'kittens' 的增强功能，可以简化处理过程。

- **理解 CUDA 中的 Memory Swizzling**：在一次简短的技术交流中，讨论了 *memory swizzling* 作为一种在 CUDA 编程中避免 memory bank 冲突的方法。其优势在 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) 中有进一步解释。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk">ThunderKittens: 一个用于 AI kernel 的简单嵌入式 DSL</a>: 好的抽象就是好。</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: 如何让 GPU 变快？</li><li><a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens: 用于高速 kernel 的 Tile 原语</a>: 用于高速 kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/HazyResearch/nanoGPT-TK">GitHub - HazyResearch/nanoGPT-TK: 用于训练/微调中型 GPT 的最简单、最快的仓库。现在有了 kittens！</a>: 用于训练/微调中型 GPT 的最简单、最快的仓库。现在有了 kittens！ - HazyResearch/nanoGPT-TK
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1238927701281210369)** (1 messages): 

- **Zoom 上的 Kernel Fusion 讲座**: 关于 **kernel 融合实际经验** 的讨论将在 7 分钟后由特邀演讲者开始。它将在 Zoom 上的这个 [会议链接](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success) 举行，参与者应在 Discord 中提问，具体在频道 <#1238926773216084051>。

**链接提到**: <a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success">加入我们的 Cloud HD 视频会议</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

random_string_of_character: https://arxiv.org/abs/2405.05219
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1238752380972040323)** (14 messages🔥): 

- **伊利诺伊大学 PMPP 系列进行中**: 伊利诺伊大学 PMPP 系列的第 4 讲已发布，并提醒避免在该频道进行一般性讨论，同时为参与者提供了 [Zoom 链接](https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1)。该课程每周举行一次，分别面向 EMEA 和 NAM 地区，通知会发布在专门的 Discord 服务器上以避免混乱。

- **PMPP 系列提供 YouTube 播放列表**: 伊利诺伊大学 PMPP 系列讲座可在 [YouTube 播放列表](https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=shared) 中观看，内容为 2018 年春季录制的“应用并行编程”课程。

- **简化概念的比喻**: 一位用户非常喜欢讲座中将 warps 比作军队中的排（platoons）的比喻，强调了使用贴切的隐喻来澄清该系列中复杂技术概念的做法。

- **Discord 上的 CUDA 社区支持集成与讨论**: 鼓励用户分享和讨论教学环节及链接，而不要弄乱常规聊天频道，建议使用特定频道进行公告和讨论，以保持秩序和专注。

- **关于 torch-tensorrt 兼容性和安装的查询**: 一位参与者询问关于哪些版本的 torch-tensorrt 与特定的 CUDA 和 Torch 版本兼容的指导，并指出安装似乎包含了多个 CUDA 运行时版本，这可能会导致混淆。
<div class="linksMentioned">

<strong>链接提到</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=sha">UIUC ECE408/CS483 2018 春季 Hwu</a>: 这是伊利诺伊大学厄巴纳-香槟分校的一门名为 &quot;应用并行编程&quot; 的大三/大四本科课程。它通常也被称为...</li><li><a href="https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1">加入我们的 Cloud HD 视频会议</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1239615740680605778)** (1 messages): 

- **高级 Scan 技术揭秘**: PMPP 作者 Izzat El Hajj 将在 **5 月 24 日** 讨论 **scan** 技术，随后 Jake 和 Georgii 将在 **5 月 25 日** 探讨 **CUDA C++ 中的高级 scan 应用**。感兴趣的人员可以通过 [此 Discord 链接](https://discord.gg/gFDMmM96?event=1239607867666071654) 参加活动。

**提到的链接**：<a href="https://discord.gg/gFDMmM96?event=1239607867666071654">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持联系。

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1239310080353239223)** (5 条消息): 

- **寻求热成像人脸识别方面的帮助**：成员 cracker10 为一个名为 **'Thermal Face Recognition'**（热成像人脸识别）的大学毕业设计请求帮助。具体在寻找关于识别两张热成像图像是否为同一人的见解、资源或建议。

- **澄清项目目标**：在回答问题时，cracker10 澄清了该项目的目标是确定两张热成像人脸图像是否属于同一个人。

- **对热成像方面的协助有限**：另一位成员 pessimistic_neko 表示无法提供帮助，称他们不了解热成像人脸识别。这一点通过一个自定义表情符号幽默地强调了，表明这是一次直率且轻松的互动。

---

**CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

boxxy_ms: 有人在多伦多吗？

---

**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1239536441571278909)** (2 条消息): 

- **寻求官方解决方案以进行验证**：一位成员询问了官方解决方案，以检查其实现的数值准确性和效率。他们随后在之前的帖子中找到了所需的解决方案。

---

**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1238941871032635583)** (67 条消息🔥🔥): 

- **多 GPU 脚本编写的性能见解**：在关于多 GPU 性能的讨论中，一位成员分享了他们的观察：98% 的 CPU 时间消耗在等待 GPU 任务完成上。他们提议将一些任务从 GPU 卸载到 CPU，并建议 llm.c 的灵活性可以允许这种调整（[阅读讨论相关主题的论文](https://inria.hal.science/hal-02316266v3/document)）。

- **梯度累积和 ZeRO-1 的进展**：用户讨论了利用 ZeRO-1 进行优化器分片（optimizer sharding）的各种配置和结果，展示了显著的 VRAM 节省以及在 Nvidia A100 等 GPU 上增加 Batch Size 的潜力。主要讨论和更新与一个 GitHub Pull Request 相关（[在此查看 PR 详情](https://github.com/karpathy/llm.c/pull/309)）。

- **探索用于硬件优化的 ThunderKittens**：ThunderKittens 项目提供用于加速 Kernel 操作的 tile primitives，其潜在用途被强调为对 llm.c 未来优化有益。它以低级抽象著称，可以很好地与 llm.c 的需求产生协同效应（[更多关于 ThunderKittens 的信息](https://github.com/HazyResearch/ThunderKittens/tree/main)）。

- **llm.c 在 CI 系统中使用 GPU 的挑战**：关于 llm.c 的 CI 中缺乏 GPU 的对话揭示了在测试和保证依赖 GPU 的代码方面的挑战。讨论提到 GitHub 新的 GPU runner 可能会有所帮助，尽管仍处于 Beta 阶段，这需要调整 GitHub 计划并可能产生额外费用（[GitHub Action GPU 使用率](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#per-minute-rates-for-larger-runners)）。

- **使用 ZeRO-1 的 GPU 内存效率**：围绕 ZeRO-1 优化器的大量讨论表明，内存使用量显著减少，Batch Size 的有效性有所提高。提交了一个解决其中一些方面的 Commit，进一步提高了性能，同时允许在强大的 GPU 上探索更大的 Batch Size（[Commit 详情](https://github.com/karpathy/llm.c/pull/309/commits/f613ce895b30dc0b2bd1f7e81410c6a2dcdce74d)）。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#per-minute-rates-for-larger-runners">关于 GitHub Actions 的计费 - GitHub Docs</a>：未找到描述</li><li><a href="https://github.blog/changelog/2023-10-31-run-your-ml-workloads-on-github-actions-with-gpu-runners/">在带有 GPU runner 的 GitHub Actions 上运行您的 ML 工作负载</a>：在带有 GPU runner 的 GitHub Actions 上运行您的 ML 工作负载</li><li><a href="https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpLoad.html#cub-warpload)">cub::WarpLoad &mdash; CUB 104.0 文档</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/309/commits/f613ce895b30dc0b2bd1f7e81410c6a2dcdce74d">Zero Redundancy Optimizer - Stage1 由 chinthysl 提交 · Pull Request #309 · karpathy/llm.c</a>：为了训练更大的模型变体（2B, 7B 等），我们需要为参数、优化器状态和梯度分配更大的 GPU 显存。Zero Redundancy Optimizer 引入了这种方法来分片...</li><li><a href="https://github.com/karpathy/llm.c/issues/406">2D 和 3D tile 划分，以便从 threadIdx 和 blockIdx 读取排列坐标 · Issue #406 · karpathy/llm.c</a>：据推测，排列 kernel 尽管大多受内存限制，但可以通过使用 2D 或 3D 网格来减少除法量并进行线程粗化（thread coarsening），从而无需在...中进行任何除法。</li><li><a href="https://github.com/NVIDIA/cccl/issues/525).">Issues · NVIDIA/cccl</a>：CUDA C++ 核心库。通过在 GitHub 上创建账户为 NVIDIA/cccl 的开发做出贡献。</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/main">GitHub - HazyResearch/ThunderKittens: 用于快速 kernel 的 Tile 原语</a>：用于快速 kernel 的 Tile 原语。通过在 GitHub 上创建账户为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://github.com/karpathy/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/.github/workflows/ci.yml#L141">llm.c/.github/workflows/ci.yml 在 2346cdac931f544d63ce816f7e3f5479a917eef5 · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/309">Zero Redundancy Optimizer - Stage1 由 chinthysl 提交 · Pull Request #309 · karpathy/llm.c</a>：为了训练更大的模型变体（2B, 7B 等），我们需要为参数、优化器状态和梯度分配更大的 GPU 显存。Zero Redundancy Optimizer 引入了这种方法来分片...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L689">llm.c/train_gpt2.cu 在 master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1238928157692919809)** (48 条消息🔥): 

- **字体大小固定**：应用户要求，增加了未指定应用程序或文档中的字体大小。
- **在线会议的澄清与访问提供**：一名用户因活动详情中缺失会议密码而请求密码；另一名用户提供了加入会议的直接链接[点击此处](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09)。
- **关于 CUDA 和 Torch Compile 的参与**：讨论包括 CUDA graphs 的有效性以及 `torch.compile` 的复杂性。用户希望进一步明确 `torch.compile` 的内部工作原理，特别是与 CUDA graphs 结合使用时。分享了有用的资源和教程，包括 [ASPLOS 2024 workshops](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024) 和 [TorchDynamo 深度解析](https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html)。
- **关于 Triton 和 Kernel Fusing 的讨论**：用户就性能优化中 kernel fusing 的益处和策略进行了技术讨论，一些人辩论了减少 kernel 启动次数与潜在开销之间的影响。
- **对进一步演讲和澄清的请求**：用户对深入探讨 `torch.compile` 和 Triton 内部机制表现出浓厚兴趣，并请求相关的演讲和额外文档，以更好地理解这些复杂主题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.compile.html">torch.compile &mdash; PyTorch 2.3 文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/wiki/Tensor-and-Operator-Basics">Tensor 和 Operator 基础</a>：Python 中具有强 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/workshops/tree/master/ASPLOS_2024">pytorch/workshops 仓库 master 分支下的 workshops/ASPLOS_2024</a>：这是一个存放所有研讨会相关材料的仓库。 - pytorch/workshops</li><li><a href="https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html">Dynamo 深度解析 &mdash; PyTorch 主文档</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1239093813033828372)** (5 条消息): 

- **ECE408 课程幻灯片已分享**：**ECE408 / CS483 / CSE408: Applied Parallel Programming** 2019 年春季课程的幻灯片已在 [ZJUI Section](https://lumetta.web.engr.illinois.edu/408-S19/) 发布。发布了重要公告和课程计划，包括考试日期和项目时间表。

- **CUDA Mode YouTube 观看派对启动**：为 CUDA 爱好者宣布了一个新的 YouTube 观看派对，参与者观看时长不超过 1-1.5 小时的视频并进行间歇性讨论。目的是促进初学者的学习并让经验丰富的参与者进行练习。

- **当前观看系列 - PMPP 2018 讲座**：当前的视频系列来自 PMPP 书籍作者的 2018 年讲座，托管在 [PMPP book YouTube channel](https://www.youtube.com/@pmpp-book)。鼓励每 10-15 分钟进行一次讨论，以增强对书籍的理解和阅读动力。

- **场次安排**：观看派对定于每周六举行，共两场；一场在 7:30 GMT 面向 EMEA 参与者，另一场在 18:00 GMT 面向 NAM 参与者。Zoom 链接将由指定的管理员提供。

- **观看派对内容的未来计划**：在 PMPP 的 18 讲系列结束后，可能会重新回顾早期的 CUDA Mode 视频，或者探索经过质量审核的并行处理相关新内容。

**提到的链接**：<a href="https://lumetta.web.engr.illinois.edu/408-S19/">ECE408: Applied Parallel Programming, Spring 2019 ZJUI Section</a>：未找到描述

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1238856315993063554)** (61 条消息🔥🔥): 

- **概念频率与模型性能关系的探索**：讨论了关于 [多模态模型研究](https://arxiv.org/abs/2404.04125) 的担忧，强调了“Zero-shot”泛化主张与训练数据集中概念频率相关的实际性能之间的差异。讨论强调了主流媒体对 AI 进展报道中持续存在的误解和歪曲。

- **生成式 AI 的增量增长**：对生成式 AI 的 *增量式* 改进提出了质疑，引用了同一篇研究论文，指出像 GPT 和 Stable Diffusion 这样的主要生成模型在未来的迭代中可能不会像以前那样表现出突破性的进展。

- **AI 理解的宣称与现实**：一篇论文讨论断言 [多模态模型的性能](https://arxiv.org/abs/2404.04125) 很大程度上受其训练数据中概念频率的影响，削弱了这些模型中鲁棒的 “Zero-shot” 泛化概念。

- **Falcon2 11B 模型发布**：推出了一个新的模型 Falcon2 11B，该模型在经过显著精炼的 5T web 数据集上训练；显著特征包括 8k 上下文窗口和改进的 Attention 机制，承诺提供更好的推理能力。

- **AI 发展直播**：关于 **GPT-4o** 的正在进行或即将开始的实时讨论可以在 [此 YouTube 直播会话](https://www.youtube.com/live/DQacCB9tDaw?feature=shared&t=3478) 中观看，预计将揭晓包括 ChatGPT 最新动态在内的 OpenAI 更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.04125">没有指数级数据就没有“Zero-Shot”：预训练概念频率决定多模态模型性能</a>：网页抓取的预训练数据集是多模态模型（如用于分类/检索的 CLIP 和用于图像生成的 Stable-Diffusion）令人印象深刻的“Zero-Shot”评估性能的基础...</li><li><a href="https://www.youtube.com/live/DQacCB9tDaw?feature=shared&t=3478">介绍 GPT-4o</a>：OpenAI 春季更新 —— 2024 年 5 月 13 日星期一现场直播。介绍 GPT-4o、ChatGPT 的更新等。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1238750605707841569)** (79 条消息🔥🔥): 

- **探索高效 Attention**：一项[新研究](https://arxiv.org/abs/2405.05219)提出了一种利用卷积矩阵计算 Attention 的高效方法。该方法通过利用快速傅里叶变换 (FFT) 可能会显著减少计算时间，但其实际适用性以及与 flashattn 等现有方法的比较仍在讨论中。

- **LLM 深度扩增（Depth Upscaling）研究**：深度扩增是一种通过层重复来改进模型的方法，已在 [SOLAR](https://arxiv.org/abs/2312.15166) 等研究论文中被引用。详细讨论和更多示例包括关于 Yi 和 Granite Code 模型的工作，强调了扩展模型深度的各种方法。

- **Hazy Research 推出 ThunderKittens**：Hazy Research 开发了 ThunderKittens，旨在简化 AI 中的关键技术实现。他们的工作（详见其[博客文章](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)）力求弥合复杂算法与实际 AI 库实现之间的鸿沟。

- **AR 任务数据蒸馏的挑战**：最近一项名为 Farzi 的提案旨在将密集数据集整合为紧凑且高效的序列，用于训练自回归（autoregressive）模型，实现高达原始数据性能的 120%。更多细节和效率比较可见于其在 [OpenReview 上的发表内容](https://openreview.net/forum?id=H9DYMIpz9c&noteId=aN4DeBSr82)。

- **线性 Attention 模型性能对比受到关注**：Linear Attention 模型在 MMLU 等复杂评估中的表现被广泛讨论，并探索了数据集对模型功效的影响。正在进行的讨论强调了需要合适的数据来发挥模型改进的潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>：语言模型中分词器（tokenizer）创建与模型训练之间的脱节，已知会导致某些输入（如臭名昭著的 SolidGoldMagikarp Token）诱发异常行为。...</li><li><a href="https://arxiv.org/abs/2405.04435">Fast Exact Retrieval for Nearest-neighbor Lookup (FERN)</a>：精确最近邻搜索是一个计算密集型过程，甚至其较简单的变体——向量检索——在计算上也可能非常复杂。当检索向量时，这种情况会进一步加剧...</li><li><a href="https://arxiv.org/abs/2310.09983">Farzi Data: Autoregressive Data Distillation</a>：我们研究了自回归机器学习任务的数据蒸馏，其中输入和输出具有严格的从左到右的因果结构。具体而言，我们提出了 Farzi，它总结了...</li><li><a href="https://arxiv.org/abs/2405.06147v1">State-Free Inference of State-Space Models: The Transfer Function Approach</a>：我们通过其对偶表示——传递函数（transfer function），来设计用于深度学习应用的状态空间模型（SSM），并发现了一种高效的序列并行推理算法...</li><li><a href="https://arxiv.org/abs/2405.05219">Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers</a>：大型语言模型（LLMs）深刻改变了世界。自注意力机制是 Transformer 在 LLM 中取得成功的关键。然而，$O(n^2)$ 的二次计算成本使得...</li><li><a href="https://arxiv.org/abs/2403.04652">Yi: Open Foundation Models by 01.AI</a>：我们介绍了 Yi 模型家族，这是一系列展示了强大维度能力的语言和多模态模型。Yi 模型家族基于 6B 和 34B 预训练语言模型...</li><li><a href="https://arxiv.org/abs/2405.04324">Granite Code Models: A Family of Open Foundation Models for Code Intelligence</a>：在代码上训练的大型语言模型（LLMs）正在彻底改变软件开发过程。代码 LLM 正越来越多地被集成到软件开发环境中，以提高...</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>：我们介绍了 SOLAR 10.7B，一个拥有 107 亿参数的大型语言模型（LLM），在各种自然语言处理（NLP）任务中展示了卓越的性能。受近期研究的启发...</li><li><a href="https://arxiv.org/abs/2405.06394">Memory Mosaics</a>：Memory Mosaics 是协同工作以实现特定预测任务的联想记忆网络。与 Transformer 类似，Memory Mosaics 具有组合能力和上下文学习（in-context learning）能力...</li><li><a href="https://huggingface.co/spaces/devingulliver/subquadratic-llm-leaderboard">Subquadratic LLM Leaderboard - a Hugging Face Space by devingulliver</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2309.03852">FLM-101B: An Open LLM and How to Train It with $100K Budget</a>：大型语言模型（LLMs）在 NLP 和多模态等任务中取得了显著成功。尽管取得了这些成功，开发 LLM 仍面临两个主要挑战：(i) 高昂的计算...</li><li><a href="https://arxiv.org/abs/2305.02869">Masked Structural Growth for 2x Faster Language Model Pre-training</a>：加速大型语言模型预训练是当前研究的一个关键问题。在本文中，我们专注于通过从小型 Transformer 结构逐步增长来加速预训练...</li><li><a href="https://openreview.net/forum?id=H9DYMIpz9c&noteId=aN4DeBSr82">Farzi Data: Autoregressive Data Distillation</a>：我们研究了自回归机器学习任务的数据蒸馏，其中输入和输出具有严格的从左到右的因果结构。具体而言，我们提出了 Farzi，它总结了...</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk">ThunderKittens: A Simple Embedded DSL for AI kernels</a>：好的抽象就是好的。</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://arxiv.org/abs/2104.05520">Updatable Learned Index with Precise Positions</a>：索引在现代数据库引擎中起着加速查询处理的关键作用。“学习索引”（learned index）的新范式显著改变了索引结构的设计方式...</li><li><a href="https://www.jeanfeydy.com/">Jean Feydy's home page</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239488393713287199)** (7 条消息):

- **对合成数据（Synthetic Data）的看法不一**：一位参与者对合成数据持乐观态度，表示看好，而另一位则提出了反驳，质疑其突破性地位，理由是 5-7 年前曾有过类似的炒作周期。有人担心过去的教训可能没有被吸取，虽然合成数据看起来很有前景，但它也伴随着显著的权衡（tradeoffs）。

- **合成数据的炒作与现实**：讨论强调，虽然合成数据对初学者来说像是“银弹”，但广泛使用过它的人了解其局限性。这场持续的辩论凸显了技术炒作与现实主义之间的循环。

- **通过实证研究探索 DNN 结构**：一篇分享的 [arXiv 论文](https://arxiv.org/abs/2108.13002) 在名为 SPACH 的统一框架下讨论了一系列深度神经网络（DNN）架构，包括 CNN, Transformer 和 MLP，表明随着网络规模的增加，它们表现出不同的行为。另一项 [研究](https://arxiv.org/abs/2306.13575) 重新审视了 MLP，探索了这一基础模型的极限，暗示尽管目前存在局限性，但未来仍有扩展潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2108.13002#microsoft">A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP</a>：卷积神经网络（CNN）是计算机视觉领域主流的深度神经网络（DNN）架构。最近，基于 Transformer 和多层感知器（MLP）的模型，如 Vision Tra...</li><li><a href="https://arxiv.org/abs/2306.13575">Scaling MLPs: A Tale of Inductive Bias</a>：在这项工作中，我们重新审视了深度学习中最基础的构建块——多层感知器（MLP），并研究了它在视觉任务上的性能极限。对 MLP 的实证见解是...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1239481105514758144)** (3 messages): 

- **NeurIPS 投稿征集**：一位用户表示有兴趣合作进行 NeurIPS 的最后时刻投稿，提到一个类似于 "othello paper" 的项目。

- **调查模型压缩的副作用**：发起了一场关于模型压缩过程中丢失的特征和电路性质的讨论——它们是无关紧要的，还是过于专业化的，这可能有助于揭示训练数据集的多样性。
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

oleksandr07173: Hello
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1238819707457372161)** (120 messages🔥🔥): 

- **GPT-4o 作为前沿模型发布**：GPT-4o 已作为 OpenAI 最新的 state-of-the-art 模型推出，在 LMSys 竞技场中以别名 "im-also-a-good-gpt2-chatbot" 进行了测试。该模型在[此公告](https://x.com/liamfedus/status/1790064963966370209?s=46)中被描述为重大改进。

- **关于 GPT-4o 编程能力的辩论**：讨论指出 GPT-4o 与早期版本在编程能力上存在明显的感知差距，暗示了重大改进。对话暗示期待通过 MATH 等新基准测试来更好地了解这些进步，[这篇博客文章](https://openai.com/index/hello-gpt-4o/)中正在进行的讨论也提到了这一点。

- **分词器（Tokenization）进展指向模型增强**：OpenAI 更新了其分词器，如 [GitHub commit](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8) 所示。这意味着效率的提高，可能是由于词表范围的扩大。

- **对 OpenAI 战略举措的期望与推测**：广泛猜测 OpenAI 的战略方向，特别是免费提供 GPT-4o，可能是为了收集更多数据，或者是作为对抗 Meta 等其他大型科技公司的竞争举措。在比较市场行动和技术进步的讨论中，对这些战略转型进行了广泛探讨。

- **现场演示与公众反应**：OpenAI 进行了现场演示，吸引了各种反应，从对其可能应用的评论到对其演示风格的批评。社区正在审查所展示技术的真实性和实用性，包括它们的集成和用户界面。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/liamfedus/status/1790064963966370209?s=46">来自 William Fedus (@LiamFedus) 的推文</a>：GPT-4o 是我们最新的 state-of-the-art 前沿模型。我们一直在 LMSys arena 上以 im-also-a-good-gpt2-chatbot 的身份测试一个版本 🙂。以下是它的表现。</li><li><a href="https://x.com/lmsysorg/status/1790097595064529255?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：相对于所有其他模型，胜率显著提高。例如，在非平局对决中，对比 GPT-4 (June) 的胜率约为 80%。</li><li><a href="https://x.com/google/status/1790055114272612771?s=46>)">来自 Google (@Google) 的推文</a>：距离 #GoogleIO 还有一天！我们感到非常 🤩。明天见，获取有关 AI、Search 等的最新消息。</li><li><a href="https://fxtwitter.com/bedros_p/status/1789256595123179701?s=46">来自 Bedros Pamboukian (@bedros_p) 的推文</a>：示例列表中的 VideoFX 镜头。还有 2 个，但看起来还在开发中 (WIP)。初探 VideoFX 的生成效果：</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">同步代码库 · openai/tiktoken@9d01e56</a>：未找到描述</li><li><a href="https://x.com/drjimfan/status/1790122998218817896?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：我纠正一下：GPT-4o 并不原生处理视频流。博客说它只接受图像、文本和音频。这很遗憾，但我所说的原则仍然成立：制作视频的正确方法是...</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991?s=46">来自 lmsys.org (@lmsysorg) 的推文</a>：突发新闻 —— gpt2-chatbots 的结果现已公布！gpt2-chatbots 刚刚飙升至榜首，以显著差距（约 50 Elo）超越了所有模型。它已成为 Arena 中有史以来最强的模型...</li><li><a href="https://x.com/kaiokendev1/status/1790068145933185038?s=46">来自 Kaio Ken (@kaiokendev1) 的推文</a>：是的，但它会呻吟吗？
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1239691351629762630)** (1 messages): 

- **REINFORCE 作为 PPO 的特例**：Huggingface TRL 仓库最近的一个 PR 详细阐述了 **REINFORCE** 实际上是 **PPO** 的一个特例。详细的实现和解释可以在这个 [GitHub PR](https://github.com/huggingface/trl/pull/1540) 中找到，同时附带了[参考论文](https://arxiv.org/pdf/2205.09123)。

**提到的链接**：<a href="https://github.com/huggingface/trl/pull/1540">vwxyzjn 提交的 PPO / Reinforce Trainers · Pull Request #1540 · huggingface/trl</a>：此 PR 支持 https://arxiv.org/pdf/2402.14740.pdf 中的 REINFORCE RLOO 训练器。请注意，REINFORCE 的损失函数是 PPO 的一个特例，如下所示，它与展示的 REINFORCE 损失相匹配...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1238889709644808294)** (5 messages): 

- **对 Chatbot Arena 社区的赞赏**：成员们表达了对 **Chatbot Arena** 社区的钦佩，强调其在塑造未来方面发挥了重要作用。
- **关于 GPT-3.5 开源的推测**：讨论涉及了 **GPT-3.5** 开源的可能性。一条评论幽默地暗示，这只有在“太阳从西边出来”（hell freezes over）时才会发生。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1239071719927320647)** (11 messages🔥): 

- **视频播放量上升**：一位成员的视频在一天内达到了 **6k 播放量**，而其他人的视频已达到 **20k** 播放量，引发了关于如何进一步提升这些数字的讨论。
- **Huggingface 视频获得高关注**：另一段在 HuggingFace 上分享的视频获得了令人印象深刻的 **150k 播放量**。
- **在 Platform X 上传视频**：有一场关于在 Platform X 发布视频潜力的对话，考虑了 **原生上传** 和法律许可。
- **与斯坦福大学协商视频版权**：成员确认 **Stanford 拥有某些内容的所有权**，但通常不会执行严格的措施，这可能允许更灵活的使用。
- **规避官僚主义的计划**：该策略包括向 Stanford 申请个人使用许可，并继续发布视频，赌后果发生的可能性较低。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1238837869716574269)** (109 messages🔥🔥): 

- **AI 在艺术家权利与商业用途之间的法律平衡受到质疑**：在一场激烈的辩论中，成员们讨论了当商业 AI 服务生成的作品可能与艺术家竞争时，潜在的法律挑战。特别提到了围绕 **Midjourney** 的担忧，因为它可能会鼓励用户模仿在世艺术家的风格。

- **AI 的管辖权与 Fair Use**：一些成员对 AI 模型在生成衍生作品时可能侵犯艺术家版权表示担忧。然而，其他人则通过强调 Fair Use 保护来进行反驳，甚至暗示在 Fair Use 下的负面评论同样可能损害艺术家的商业前景，且无需承担法律后果。

- **关于 AI 生成内容的 Fair Use 争论**：观点出现了明显分歧，一些成员认为可能影响艺术家销售的 AI 生成内容必须面临法律审查，而另一些人则认为这种使用属于 Fair Use，并将其与评论内容进行类比。

- **陪审团在解释 AI 相关法律中的角色受到审视**：讨论涉及了 Jury Nullification（陪审团否决）以及陪审团在 AI 相关法律案件中的恰当角色，强调了法律在技术层面上应如何遵守与司法机构在现实世界中的运作方式之间的差异。

- **关注 AI 的计算效率和开源模型**：分享的一个链接讨论了旨在降低 AI 计算需求的创新；其中包括为提高效率而开发的各种新方法和模型（[点击阅读更多](https://www.techopedia.com/openais-gpt-4o-release)）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">deepseek-ai/DeepSeek-V2 · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>：OpenAI 春季更新 – 2024 年 5 月 13 日星期一直播。介绍 GPT-4o、ChatGPT 的更新等。</li><li><a href="https://tenor.com/bR79n.gif">Silicon Valley Tip To Tip GIF - Silicon Valley Tip To Tip Brainstorm - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas">Falcon 2: UAE’s Technology Innovation Institute Releases New AI Model Series, Outperforming Meta’s New Llama 3</a>：未找到描述</li><li><a href="https://civitai.com/models/435669?modelVersionId=502675">Bunline - v0.4 | Stable Diffusion Checkpoint | Civitai</a>：PixArt Sigma XL 2 1024 MS，基于约 3.5 万张 max(w,h) > 1024px 的图像及自定义标注进行全量微调。说明：将 .safetensors 放置在...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1238861766231064607)** (5 条消息): 

- **语音数据集需要转换**：一位成员强调了将海量语音数据集转换为 tokens 的必要性，并强调了对情感和说话者属性进行高质量标注的需求。他们分享了一个像处理文本一样使用音频训练 Transformer 的链接，详见[这里](https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19)，以及 [YouTube 上的更多资源](https://youtu.be/NwZufAJxmMA)。

- **深入探讨形式数学符号**：讨论了在形式数学中使用特定符号来表示收敛元素序列的问题，一位成员阐明了函数在此背景下的潜在作用。函数 **T** 被提及作为在此类序列中执行 sampling（采样）的可能工具。

**提到的链接**：<a href="https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19">来自 LAION (@laion_ai) 的推文</a>：想想像处理文本一样用音频训练 Transformer 吗？- 方法如下。:) https://youtu.be/NwZufAJxmMA https://discord.gg/6jWrFngyPe

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1238794899726405662)** (105 条消息🔥🔥): 

- **LangChain 日期提取详解**：成员们讨论了如何使用 LangChain 的 `DatetimeOutputParser` 提取日期并将其转换为 ISO 格式。提供了 JavaScript 和 Python 的详细代码片段，展示了如何实现该解析器。

- **LangChain 中的自定义日期范围解析**：当被问及如何处理类似“从 4 月 1 日到 6 月 2 日”的日期范围时，建议可以通过修改 `parse` 方法来扩展 `DatetimeOutputParser`，以识别并分别提取开始和结束日期。

- **处理 Prompt 中的多个描述**：针对从“比较比利时石油和意大利电力的价格”等 Prompt 中提取多个市场描述的问题，引出了在 LangChain 中使用 LLM 的 tool/function calling 结合 Schema 结构化提取所需信息的说明。

- **在 LangChain 中使用本地开源 LLM**：提供了将 Ollama 等本地开源 LLM 与 LangChain 集成的指南，详细说明了从设置 LLM、安装必要包到与模型交互的步骤。

- **针对多个前端元素的流式 API 响应**：一位用户询问了如何通过单次 API 调用为两个前端元素提供流式 API 响应。回复中重点介绍了使用 Python 的方法，并提供了一个包含相关示例的 GitHub 链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://python.langchain.com/v0.1/docs/integrations/stores/">Stores | 🦜️🔗 LangChain</a>：在许多不同的应用中，拥有某种键值存储（key-value storage）是很有帮助的。</li><li><a href="https://python.langchain.com/docs/modules/agents/agent_types/structured_chat#run-agent>).">Structured chat | 🦜️🔗 LangChain</a>：结构化聊天 Agent 能够使用多输入工具。</li><li><a href="https://python.langchain.com/docs/use_cases/extraction#approaches>)">Extracting structured output | 🦜️🔗 LangChain</a>：概览</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/quickstart#toolfunction-calling>)">Quickstart | 🦜️🔗 LangChain</a>：在本指南中，我们将介绍创建调用 Tools 的 Chains 和 Agents 的基本方法。Tools 可以是任何东西——APIs、函数、数据库等。Tools 允许我们扩展功能...</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/libs/partners/chroma/pyproject.toml">langchain/libs/partners/chroma/pyproject.toml at master · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb">generative-ai/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb at main · GoogleCloudPlatform/generative-ai</a>：Google Cloud 上生成式 AI 的示例代码和 Notebook，包含 Vertex AI 上的 Gemini - GoogleCloudPlatform/generative-ai</li><li><a href="https://github.com/Go">go - Overview</a>：go 有 52 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://gist.github.com/mattcollins/62fcb8d15a001d5b4e5c9fb86aad4f8e">Example of extracting multiple values from a streamed OpenAI chat response</a>：从流式 OpenAI 聊天响应中提取多个值的示例 - extract_multiple_values_from_stream.py</li><li><a href="https://github.com/langchain-ai/langchain/issues/11011>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3994>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3577>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/19805>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/get_started/quickstart#llm-chain>)">Quickstart | 🦜️🔗 LangChain</a>：在此快速入门中，我们将向您展示如何：</li><li><a href="https://github.com/langchain-ai/langchain/issues/16935>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17029>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17008>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17031>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/90>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/integrations/llms/manifest#compare-hf-models>).">Manifest | 🦜️🔗 LangChain</a>：本 Notebook 介绍了如何使用 Manifest 和 LangChain。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5513>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/9908>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 做出贡献。

通过在 GitHub 上创建账户来参与开发。</li><li><a href="https://github.com/langchain-ai/langchain/issues/4438>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户来为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1238769533431320608)** (4 条消息): 

- **探索癌症药物研发中的 AI**：一段 YouTube 视频讨论了 Generative AI 在癌症药物研发中的作用，以及对更多自动化方法的需求。点击[此处](https://youtu.be/vyOtowbGwG0?feature=shared)查看完整探索。

- **Index Network 的最新动态**：通过查看他们最新的 [推文](https://twitter.com/indexnetwork_/status/1788311740595245515) 来了解最新的公告或见解。

- **开源 Code Interpreter 发布**：Obaidur-rahaman 介绍了一个新的开源项目，用于自然语言辅助的可视化和交互式数据分析，该项目兼容 OpenAI API 密钥，并即将支持 Llama 3。该项目旨在安全地处理和分析机密数据，增强企业的洞察力，可在 [GitHub](https://github.com/obaidur-rahaman/nlavida) 上获取。

- **使用 LangChain 和 Pinecone 构建自定义 RAG 流水线的教程**：Zackproser 正在开发一份详细指南，介绍如何将 LangChain 与 Next.js 和 Pinecone 集成，以创建一个采用 Retrieval Augmented Generation（检索增强生成）技术的博客聊天功能。该教程将涵盖从数据摄取到构建交互式聊天界面的所有内容，详情请见[此处](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog">Build a RAG pipeline for your blog with LangChain, OpenAI and Pinecone</a>: 即使我不在身边，你也可以与我的文章聊天，并向我询问我已经回答过的问题。</li><li><a href="https://youtu.be/vyOtowbGwG0?feature=shared">Cancer Drug Discovery AI Agentic Workflow R&amp;D</a>: 许多 Generative AI 的进展都存在于药物研发应用中。然而，还需要额外的方法来进一步自动化该过程，以获得更明智的药物组合...</li><li><a href="https://github.com/obaidur-rahaman/nlavida">GitHub - obaidur-rahaman/nlavida: Natural Language-Assisted Visualization &amp; Interactive Data Analysis (NLAVIDA): Securely handle and analyze confidential data in enterprise environments, enhancing insights and decision-making with advanced visualization.</a>: 自然语言辅助可视化与交互式数据分析 (NLAVIDA)：在企业环境中安全地处理和分析机密数据，通过高级可视化增强洞察力和决策能力...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1239226368735186975)** (3 条消息): 

- **LLM 通过 DinoV2 实现多模态**：一位成员分享了一段名为“使用 DinoV2 让任何文本 LLM 变为多模态”的 [YouTube 视频](https://www.youtube.com/watch?v=KQ-xGVFHDkw)，展示了将视觉能力集成到基于文本的语言模型中的方法。他们还提供了一个相关 Notebook 的 GitHub 链接：[DinoV2 Vision Encoder Notebook](https://github.com/githubpradeep/notebooks/blob/main/VLM.ipynb)。

- **开发了“与我的博客聊天”体验**：Zackproser 详细介绍了他是如何在博客中加入 [聊天功能](https://zackproser.com/chat) 的，使访问者能够直接就他的文章进行互动和提问。他使用了 **Retrieval Augmented Generation** 技术来实现这一功能，并在其 [博客文章](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog) 中提供了包括摄取代码、数据处理和聊天界面在内的全面资源。

- **带有会话和历史管理的流式传输**：Brianjack 表达了在保持会话和历史管理的同时将流式传输功能集成到 LangChain 中的困难，并表示他已成功实现了除流式传输以外的所有功能。他正在寻找专门针对克服流式传输挑战的教程或帮助。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog">Build a RAG pipeline for your blog with LangChain, OpenAI and Pinecone</a>: 即使我不在身边，你也可以与我的文章聊天，并向我询问我已经回答过的问题。</li><li><a href="https://www.youtube.com/watch?v=KQ-xGVFHDkw">Make any Text LLM into Multimodal with DinoV2</a>: 我们将研究如何使用视觉编码器为文本 LLM 添加视觉能力。https://github.com/githubpradeep/notebooks/blob/main/VLM.ipynbhttps:/...
</li>
</ul>

</div>
  

---

**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1238888807072403516)** (8 messages🔥): 

- **Llama 3 助力自动生成 PowerPoint**：@naivebaesian 的一篇文章探讨了使用 Llama 3 RAG 流水线不仅能回答问题，还能利用 Python-pptx 库生成 PowerPoint 幻灯片。更多信息请点击[这里](https://t.co/iM0c5Cl2uK)。

- **反思型金融 Agent 开发详解**：Hanane Dupouy 详细介绍了开发一个能够通过 reflection（反思）分析股票价格的金融 Agent 指南，重点介绍了包括 CRITIC 在内的不同实现策略。详情请见[这里](https://t.co/mmJ8cjmw73)。

- **内容审核 RAG 设置指南发布**：@cloudraftio 撰写了一篇文章，演示了如何建立一个 RAG 流水线，通过将图像转换为文本以便于匹配，确保用户生成的图像符合内容审核标准。进一步阅读请点击[这里](https://t.co/z6jBpMvQss)。

- **评估 RAG 系统——核心 AI 技能**：@kingzzm 对评估 RAG 系统进行了全面讨论，包括对四个评估库——TruLens, Ragas, UpTrain, DeepEval——及其支持的指标的回顾。全文请见[这里](https://t.co/gLbXJoPsqu)。

- **新黑客松 Cookbook 探索 Llama 3 用例**：在 @AIatMeta 举办黑客松之后，发布了一系列详细介绍 Llama 3 七种不同用例的新 Cookbook，展示了从基础到复杂任务的应用。阅读更多请点击[这里](https://t.co/YLlsvkI0Ku)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://t.co/yPMeyookRq">Google Colab</a>: 无描述</li><li><a href="https://t.co/5k1tvKklGA">Google Colab</a>: 无描述</li><li><a href="https://t.co/CMQ1aOXeWb">llama-index-llms-openai</a>: llama-index llms openai 集成</li><li><a href="https://t.co/1DLv8fikOi">llama-index-multi-modal-llms-openai</a>: llama-index multi-modal-llms openai 集成
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1238778218237005824)** (89 messages🔥🔥): 

- **Condense Plus Context 中的 Bug 修复**：在 `_aretrieve_context` 函数中遗漏 postprocessor 最初被视为一个 Bug，但一位成员在尝试提交 Pull Request 后澄清，该问题已在库的最新版本中得到解决。*“...今天我发现最新版本已经修复了这个 Bug。我会升级我的库。”*

- **Hybrid Search 配置错误**：一位用户尝试在 Qdrant 中使用 Hybrid Search，但尽管配置正确，仍面临 *ValueError*。另一位用户通过强调需要在构造函数中直接启用 Hybrid Search 解决了困惑：*"`QdrantVectorStore(..., enable_hybrid=True)`"*。

- **探讨 LlamaIndex 的实用性**：成员们讨论了 LlamaIndex 优于其他替代方案的优点，理由是其易用性、灵活性、详细的文档以及处理多平台支持的有效抽象层。正面的用户反馈包括：*"LlamaIndex 的文档非常精美。"*

- **前端通信的技术问题**：一位用户报告了前端显示 AI 响应的不一致性，收到错误消息 *“Unexpected token U”*。根据另一位用户检查控制台 Network 选项卡的建议，怀疑原因是响应中出现了非 200 状态码。

- **理解 Query Engine 元数据用法**：一位用户询问在使用 LlamaIndex 时元数据在 `query` 方法中的作用，想知道它是自动应用还是需要用户显式包含。另一位用户澄清说，元数据可用于过滤且必须显式利用，并讨论了元数据如何增强检索过程的策略。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 开发做贡献。</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/multimodal/">Multi-Modal Applications - LlamaIndex</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/ollama_cookbook/?h=multimodal">Multimodal Ollama Cookbook - LlamaIndex</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/readers/file#llama_index.readers.file.CSVReader>)">File - LlamaIndex</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/connector#concept>)">Data Connectors (LlamaHub) - LlamaIndex</a>: 无描述
</li>
</ul>

</div>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1239453695620419584)** (3 条消息): 

<ul>
    <li><strong>探索用于 GPT-3.5 评测的知识蒸馏 (Knowledge Distillation)</strong>：<a href="https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt">Hugging Face</a> 上的一篇博客文章解释了使用知识蒸馏来增强微调 GPT-3.5 作为裁判（Judge）的准确性和性能的过程及益处。它包含详细的逐步代码实现指南。</li>
    <li><strong>社区对微调 (Finetuning) 技术的参与</strong>：一位成员赞扬了最近的文章，强调了目前缺乏引导用户如何有效微调模型的易用资源。</li>
</ul>

**提到的链接**：<a href="https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt">Knowledge Distillation for Fine-Tuning a GPT-3.5 Judge: Enhancing Accuracy and Performance </a>：未找到描述

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1238758760621543484)** (30 条消息🔥): 

- **Llama 3 权重差异见解备受关注**：一项比较 **Llama 3** instruct 版和 base 版模型权重的分析显示，显著变化集中在 K 和 V 层，这表明在指令微调（instruct tuning）期间进行了针对性调整（[查看分析](https://gist.github.com/CoffeeVampir3/48544cdaf888a76ca6f8e25863200fad)）。目前正在考虑冻结 K/V 层进行风格微调，以在不损失指令能力的情况下调整风格。

- **保存与 Checkpoint 澄清讨论**：讨论澄清了 Checkpoint 命名规范，暗示它不是最终运行的保存（end run save），最终保存应位于 base 文件夹中。这种区别有助于理解模型运行期间的保存输出。

- **考虑资助 OpenOrca 重新运行**：一位社区成员提议在 **gpt-4o** 上重新运行 OpenOrca 去重（dedup），提供了预估成本，并建议利用 Batch 任务定价优势。拟议项目的详细信息可以在其 [数据集页面](https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup) 找到。

- **探索 AI 中更低的计算资源使用**：引用了大量专注于减少 AI 计算使用的项目，包括 **Monarch Mixer**、**H3** 和 **Hyena Safari** 等倡议，并附带博客详细介绍了这些进展（[阅读更多](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)）。

- **指出 AI 领域学术出版的挑战**：学术期刊出版的延迟可能导致研究在发表时已经过时，这凸显了快速发展的 AI 研究领域面临的挑战。缓慢的出版过程与 SOTA（State-of-the-art）进展的飞速步伐形成鲜明对比。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>：如何让 GPU 变快？</li><li><a href="https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup?">Open-Orca/SlimOrca-Dedup · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1238793400229298238)** (11 条消息🔥): 

- **Nanobitz 合并成功**：用户 "Nanobitz" 引用的代码合并被认为已成功。没有关于合并内容的具体细节。
- **PyET 中的 LLAMA3 模板问题**：一位用户尝试在 PyET 中为 LLAMA3 使用新模板，但遇到了错误，提示 'LLAMA3' 和 'LLAMA2' 之间存在混淆。建议他们更新 **fastchat**。
- **依赖更新困境**：另一位参与者 "trojaner" 指出项目依赖项严重过时，列出了 **peft**、**accelerate**、**deepspeed**、**flash-attn**、**xformers** 和 **transformers** 的版本。他们建议将所有项更新到最新版本，但由于插件问题，peft 需要从仓库安装。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1238884024642961408)** (11 条消息🔥): 

- **FSDP 与 FFT 的结合仍是谜团**：社区成员不确定 **Fully Sharded Data Parallel (FSDP)** 是否能与 **Fast Fourier Transform (FFT)** 协同工作。替代建议包括研究 [**DeepSpeed**](https://www.deepspeed.ai/)。

- **Docker 中的 AttributeError 解释**：关于 **LLAMA3** 的 AttributeError 特别是在使用 **Docker** 时出现。解决此问题的建议包括确保更新 **pip 依赖项** 并尝试重新进行 **git clone**。

- **Git cloning 解决 fastchat 问题**：直接使用 **git cloning** 的方法解决了一个仅通过更新 **fastchat** 无法修复的问题。这表明某些 commit 可能未在特定分支中更新。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1238787047309836370)** (10 条消息🔥): 

- **在 Axolotl CLI 推理中更改 system_prompt 仍不明确**：一位用户询问在使用 **axolotl.cli.inference** 时是否可以更改 `system_prompt`。虽然该查询已提交给 Phorm 寻求答案，但返回结果为 *undefined*，并建议稍后再查看。

- **将合并后的模型转换为 GGUF 时出错**：一位成员指出，在将合并后的模型转换为 GGUF 的过程中，由于缺少匹配的 tokenizer ['spm', 'hfft']，出现了 **FileNotFoundError**。这指向了文件结构或命名中可能存在的问题，需要在未来的任务或故障排除中加以解决。

- **加载 Gemma 模型时出现尺寸不匹配错误**：在尝试加载 *GemmaForCausalLM* 模型时，一位用户遇到了关于 `model.embed_tokens.weight` 的 **size mismatch error**。错误提示建议在 `from_pretrained` 方法中添加 `ignore_mismatched_sizes=True` 进行调试，这表明训练环境与运行环境之间存在不匹配问题。

- **关于无精度问题将 QLORA 合并至 Base 的疑问**：一位用户询问了如何将 QLORA 合并到基础配置中，且不遇到 fp16 和 fp32 之间的精度问题。这个问题反映了社区在模型集成和精度处理方面面临的持续挑战。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1238876815230242937)** (9 条消息🔥): 

- **咨询 Axolotl 的剪枝（Pruning）功能**：一位用户询问 **Axolotl** 是否支持剪枝。Phorm 的回答是 undefined，并建议稍后查看更新，同时提供了 [在 Phorm 上阅读更多](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined) 的链接。

- **寻求持续预训练（Continuous Pretraining）技巧和 LoRA 方法**：另一个查询是关于持续预训练的技巧和不同的 LoRA 方法。同样，答案仍为 undefined，建议用户稍后通过相同的 [Phorm 链接](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined) 重新访问该主题。

- **关于将 qLoRA 集成到 Base 的问题**：一位成员询问如何将 qLoRA 合并到基础模型中；然而，在讨论的消息中没有提供直接的回复或信息。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1238756318999740507)** (41 条消息🔥): 

- **Claude API 兼容性问题**：用户在集成 Claude API 时遇到问题，有报告称出现了“奇怪的错误（goofy errors）”，这表明可能存在兼容性或配置错误。
- **使用 Open Interpreter 进行防检测 Python 自动化**：一位用户正在探索 Open Interpreter 是否可以通过根据自然语言指令生成 Python 代码来简化浏览器自动化。这可以通过自动化重复的编码任务来提高生产力。
- **本地模型性能咨询**：讨论了 Mixtral、Phi、Llama3 和 GPT-4 等本地模型之间的比较，其中 GPT-4 被认为具有卓越的性能。建议对本地模型进行 Prompt 优化以提高其有效性。
- **GPT-4o 的速度和效率**：用户报告称，与其他模型相比， GPT-4o 提供了大幅提升的处理速度，最高可达 100 tokens/s，这显著增强了性能和成本效益。
- **ChatGPT 和 Interpreter API 的进展**：用户期待 ChatGPT 语音对话 AI 能通过 Open Interpreter API 提供。鉴于其在最近的演示中展示的潜力，用户对其快速集成充满希望。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://interpreter.chat('text">未找到标题</a>：未找到描述</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>：未找到描述
</li>
</ul>

</div>
  

---

**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1238960200678113343)** (21 messages🔥): 

- **OpenInterpreter 与 LiteLLM 在 Groq 的 Llama3 上的成功集成**：用户确认 **openinterpreter <> LiteLLM <> groq - llama3** 这一配置运行顺畅。对于参与测试的用户来说，这一集成方案已被证明是功能完备且可操作的。

- **排除 O1 硬件和 WiFi 连接问题的故障**：一名用户在设置 **M5 board** 和 **01-Light** WiFi 网络时遇到了连接困难。在尝试了包括重新刷机和使用备用设备在内的多次努力后，该用户仍无法访问 Web 界面进行正常连接。

- **开发 01 硬件的 App 版本**：Thatpalmtreeguy 讨论了为 **01 hardware** 开发移动端 App 替代方案的想法，并表示早期的 App 版本[可以在这里找到](https://github.com/eladdekel/01_For_iOS)。该方法旨在降低开发和测试的门槛。

- **等待新 App 的 TestFlight 审批**：Thatpalmtreeguy 还提到已提交 App 进行 **TestFlight** 审批，这将使没有 Mac 的用户也能更轻松地参与测试和开发。

- **关于未收到订单的客服互动**：一名用户在 **OpenInterpreter** 下单后遇到了问题，未收到收据并希望取消订单。另一名成员建议通过电子邮件 *help@openinterpreter.com* 联系客户支持。
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1238827699946913812)** (4 messages): 

- **介绍 PyWinAssistant**：一位用户分享了 [PyWinAssistant 的 GitHub 链接](https://github.com/a-real-ai/pywinassistant)，将其描述为 *首个通过自然语言控制人类用户界面的开源 Large Action Model*。该工具结合了 Visualization-of-Thought，并与 LLM 中的空间推理保持一致。

- **PyWinAssistant 运行演示**：另一位用户确认已成功运行 PyWinAssistant，并提供了一个 [YouTube 直播链接](https://www.youtube.com/live/_XyYoqpJCoQ?si=rA3ijqicagANyt96&t=1993) 来展示其能力。该演示展示了 PyWinAssistant 的实时功能。

**提及的链接**：<a href="https://github.com/a-real-ai/pywinassistant">GitHub - a-real-ai/pywinassistant: 第一个仅使用自然语言即可完全控制人类用户界面的开源 Large Action Model 通用型弱人工智能。PyWinAssistant 利用 Visualization-of-Thought 激发 LLM 中的空间推理。</a>

  

---



**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1238814017426948098)** (38 messages🔥): 

- **理解 Tensor 变量形状**：一名成员参考 [Tinygrad Notes](https://mesozoic-egg.github.io/tinygrad-notes/upcast2.html) 询问为什么 Tensor 需要变量形状。这一特性通过处理 Tensor 形状动态变化的情况（例如 Transformer 中不断增加的 Token 数量）来帮助优化编译时间，并避免为新形状重新生成 Kernel。

- **排除 Tinygrad 中的训练错误**：一位用户在训练模型时遇到了 "AssertionError: Tensor.training should be set in the optimizer" 错误。解决方案如该 [Pull Request](https://github.com/tinygrad/tinygrad/pull/4460/files) 所示，需要设置 `Tensor.training = True`。

- **实现高级索引的策略**：讨论强调了在 Tinygrad 中实现类似于 `node_features[indexes[i]] += features[i]` 操作的挑战和可能策略。技术手段包括使用 One-hot encoding 和矩阵乘法来根据索引聚合特征，正如多位贡献者提供的代码片段所示。

- **对 Graph Neural Network 实现的好奇**：讨论发起了关于在 Tinygrad 中实现 Graph Neural Network (GNN) 的话题，特别关注如何管理邻居搜索。对话触及了与 Pytorch Geometric 等现有库相比实现此类功能的复杂性，以及朴素的 O(N^2) Tensor 操作方法可能存在的低效性。

- **Tinygrad 中的错误处理**：注意到关于改进 Tinygrad 错误消息以提升用户体验的建议有所增加，并将其与 Rust 风格的错误消息进行了比较，后者会建议最简单的修复方法，帮助用户更好地理解如何纠正问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/4460/files">optimizer shouldn&#39;t be run without training by geohot · Pull Request #4460 · tinygrad/tinygrad</a>: 未找到描述</li><li><a href="https://gist.github.com/ziereis/3991cf934a0b62caec8f029f12b25135">train.py</a>: GitHub Gist: 立即分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49">test_aggregate.py</a>: GitHub Gist: 立即分享代码、笔记和代码片段。</li><li><a href="https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cuda/radius_cuda.cu">pytorch_cluster/csrc/cuda/radius_cuda.cu at master · rusty1s/pytorch_cluster</a>: 优化图聚类算法的 PyTorch 扩展库 - rusty1s/pytorch_cluster</li><li><a href="https://github.com/torchmd/torchmd-net/blob/75c462aeef69e807130ff6206b59c212692a0cd3/torchmdnet/extensions/neighbors/neighbors_cpu.cpp#L71-L80">torchmd-net/torchmdnet/extensions/neighbors/neighbors_cpu.cpp at 75c462aeef69e807130ff6206b59c212692a0cd3 · torchmd/torchmd-net</a>: 神经网络势能。通过在 GitHub 上创建账号来为 torchmd/torchmd-net 的开发做出贡献。</li><li><a href="https://github.com/shriar/Neural-Turing-Machine-in-Tinygrad/blob/main/NTM.py">Neural-Turing-Machine-in-Tinygrad/NTM.py at main · shriar/Neural-Turing-Machine-in-Tinygrad</a>: 通过在 GitHub 上创建账号来为 shriar/Neural-Turing-Machine-in-Tinygrad 的开发做出贡献。</li><li><a href="https://github.com/rs9000/Neural-Turing-machine">GitHub - rs9000/Neural-Turing-machine: NTM in PyTorch</a>: PyTorch 版 NTM。通过在 GitHub 上创建账号来为 rs9000/Neural-Turing-machine 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L36-L45">tinygrad/examples/whisper.py at a1940ced7746fcdf09068aadf4155e4c1e3641b8 · tinygrad/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L118-L120">tinygrad/examples/whisper.py at a1940ced7746fcdf09068aadf4155e4c1e3641b8 · tinygrad/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://www.pyg.org/)">Home - PyG</a>: PyG 是图神经网络 (Graph Neural Networks) 的终极库
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1238858838095040534)** (24 messages🔥): 

- **Cohere 账单困惑详解**：一位用户在理解账单详情时遇到问题，特别是不同视图中显示的费用差异。他们最终发现，费用差异是由于自上次发票以来的待付金额导致的。

- **Command R 查询说明**：成员们讨论了将 **Command R** 与 Web 和 Grounding 选项结合使用的影响，确认输入 Token 确实会变大，因为其中包含了 Web 搜索所需的 Token。

- **语言模型中未训练的 Token**：一位成员分享了一篇[研究论文](https://arxiv.org/abs/2405.05417)，讨论了大语言模型 (LLM) 分词器 (Tokenizer) 中的“故障 Token (glitch tokens)”及其检测方法，强调了分词器效率和模型安全性方面持续存在的问题。

- **Aya 与 Cohere Command Plus 的对比需要澄清**：成员们询问了 Aya 和 Cohere Command Plus 之间的性能差异，指出 Aya 即使在处理常见信息时也存在准确性问题，并建议将 Aya 的用途限制在翻译领域。

- **社区中的协助请求**：一位成员表示在获取 Cohere 相关咨询支持时遇到困难，其他用户对此作出了回应，确认了社区中确实有 Cohere 工作人员。

**提到的链接**：<a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>：语言模型中分词器创建与模型训练之间的脱节，已知会导致某些输入（如臭名昭著的 SolidGoldMagikarp Token）诱发异常行为。...

  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1238956624513597550)** (2 messages): 

- **针对电信领域的 LLM 专业化**：对于有兴趣将大语言模型专业化应用于电信领域（5G 及更高版本）的人，现在有一项挑战赛。点击[此处](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks)参加或分享该竞赛。

- **探索使用 Cohere 开发 “Chat with PDF” 应用**：询问是否存在或正在开发使用 Cohere 实现与 PDF 对话的应用程序。该用户正在寻求该领域的贡献或现有工作，请求分享相关的 Repository 或博客文章。

**提到的链接**: <a href="https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks">Zindi</a>: 未找到描述

---

**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1238788122800558090)** (23 条消息🔥): 

- **LMSYS 作为 LLM 质量指标？**: **lmsys** 作为 **LLM 质量**指标的有效性仍然模糊不清，社区内尚未达成明确共识。
- **GPT-4o 更新未达预期**: 对 **GPT-4o** 的表现表示失望，特别是它无法准确列出书籍。尽管其速度快且价格诱人，但与 **GPT-4** 相比，它缺乏显著的推理能力提升。
- **辩论 AI 的未来能力**: 对 **AGI** (Artificial General Intelligence) 的过度炒作表示怀疑，同时认可了 **GPT-4** 和 **Claude 3 Opus** 等现有模型的增量改进。一些成员认为，围绕即将推出的模型的炒作可能并无根据。
- **利用云端额度**: 一位成员询问如何有效利用即将到期的 **Google Vertex AI credits**，但缺乏具体的实验计划。
- **语音助手特性**: 有人对语音助手不恰当的笑声表示担忧，建议使用自定义提示词 (custom prompts) 作为潜在解决方案，使输出更专业，减少对用户获取工作的负面影响。

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 条消息): 

simonw: https://twitter.com/simonw/status/1790121870399782987

---

**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1238814196158824499)** (15 条消息🔥): 

- **警惕虚假仓库**: 一位成员警告称，一个声称包含 **OpenELM GGUF** 的仓库是虚假的，指出仓库可用性方面存在误导信息或错误。

- **llamafile 的 PR 增强**: 创建了一个新的 Pull Request ([PR #412](https://github.com/Mozilla-Ocho/llamafile/pull/412))，旨在添加一个基于外部资源的脚本，以方便升级 llamafile 存档。

- **性能基准测试分享**: 一位用户报告称 **Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf** 模型在 llamafile 上运行流畅，响应时间约为 10 秒，RAM 使用峰值达到 11GB，具体配置为 AMD 5600U，模型大小约为 5.6GB。

- **AI 模型持续报错**: 用户在使用 **Llama 8B 和 Mistral** 等模型时遇到了重复错误，这些错误与 KV cache 空间问题有关，具体体验因不同系统的可用 RAM 量而异。

- **Llamafile 改进的元数据管理**: 目前正在开发相关功能，以促进在 **llamafile 和 gguf** 中集成自定义作者元数据，从而有助于在 huggingface 等平台上实现更好的文件管理和可搜索性 ([Issue #7165](https://github.com/ggerganov/llama.cpp/issues/7165))。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/issues/7165">在转换 gguf 时添加元数据覆盖并生成动态默认文件名 · Issue #7165 · ggerganov/llama.cpp</a>: 这是针对 PR #4858 的正式工单，以便人们了解并参与讨论该想法是否合理……如果合理，在合并之前还需要做些什么……</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/pull/412">由 mofosyne 添加升级 llamafile 存档的脚本 · Pull Request #412 · Mozilla-Ocho/llamafile</a>: 上下文：#411 将 https://briankhuu.com/blog/2024/04/06/inplace-upgrading-of-llamafiles-engine-bash-script/ 移植到 llamafile
</li>
</ul>

</div>

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1239605375242600519)** (9 条消息🔥): 

- **寻求协助策划德语 YouTube 内容**: 一位成员表示需要创建一个包含高质量德语播客、新闻节目和 vlogs 的 YouTube 频道列表，用于训练德语 TTS 系统。他们邀请其他人合作编制该列表。

- **Mediathekview 提供丰富的德语视听资源**: Mediathekview 被推荐为从各种德国广播公司下载节目和电影的资源，并建议使用公共电子表格进行组织。讨论包括如何下载内容数据库的细节，并附带链接和描述 [Mediathekview 网站](https://mediathekview.de/)。

- **分享 MediathekView 数据的本地存储细节**: 一位成员澄清说，MediathekView 在本地存储其电影数据库，其中包括所有带有链接和描述的节目，强调了其作为 TTS 训练数据源的实用性。

- **讨论中建议优先使用英语**：频道内发布了提醒，建议参与者在沟通过程中保持使用英语。

- **探索 MediathekView API 的潜力**：重点介绍了 MediathekView 的 JSON API 信息，为自动访问媒体内容数据提供了可能 [GitHub API](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua">MediathekViewWebVLC/mediathekviewweb.lua at main · 59de44955ebd/MediathekViewWebVLC</a>：用于 VLC 的 MediathekViewWeb Lua 扩展。可以通过在 GitHub 上创建账号来为 59de44955ebd/MediathekViewWebVLC 做出贡献。</li><li><a href="https://podtail.com/de/top-podcasts/de/">Die 100 beliebtesten Podcasts im Moment &ndash; Deutschland</a>：此列表显示了当前最受欢迎的 100 个播客，包含来自 Apple 和 Podtail 的最新数据。</li><li><a href="https://hypeauditor.com/top-youtube-all-germany/">Top YouTube Channels in Germany | HypeAuditor YouTube Ranking</a>：查找截至 2024 年 5 月德国最受欢迎的 YouTube 频道。获取德国顶级 YouTuber 列表。
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1239228517527588864)** (2 条消息): 

- **Demo 挂了吗？**：一名成员询问 Demo 目前是否宕机，寻求对其状态的澄清。
- **对 Demo 的赞赏**：同一名成员的另一条消息表达了赞赏，称该 Demo “非常棒”。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1239608271225098290)** (4 条消息): 

- **结构化决策中的 Claude 3 Haiku 与 Llama 3b**：一名成员发起了一场关于在实体提取评分服务中选择 **Claude 3 Haiku** 还是 **Llama 3b** 的讨论。他们指出传统的模糊字符串匹配（fuzzy string matching）存在问题，旨在利用较小的 **LLM** 来匹配 Pydantic 模型中的子模型。
- **解决实体提取挑战**：该成员专注于提高文档实体提取的准确性，解释说他们正在构建一个使用 Pydantic 模型的自动化服务，以将预测结果与实际结果及子模型列表进行比较，并计划初步使用 Instructor 测试该结构。
  

---


**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1238796719068811275)** (6 条消息): 

- **关于音频相关更新的推测**：目前似乎存在关于音频相关功能的推测，可能涉及助手的 **音频输入输出（audio in-out）支持**。

- **关注 OpenAI 的音频团队**：对话强调 OpenAI 音频团队的成员正活跃参与其中，这可能暗示了音频技术或功能的进展。

- **期待 GPT-4o 发布**：一段名为“Introducing GPT-4o”的 [YouTube 视频](https://www.youtube.com/watch?v=DQacCB9tDaw) 预示了即将到来的 OpenAI 春季更新，计划于 **2024 年 5 月 13 日星期一** 进行直播。此次活动将推出 GPT-4o 以及 ChatGPT 的更新。

- **名人参与引发热议**：关于 **Scarlett Johansson** 为某项功能或促销配音的消息引起了兴奋，这极大地吸引了爱好者的关注和热情。

**提到的链接**：<a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>：OpenAI 春季更新 – 于 2024 年 5 月 13 日星期一进行直播。介绍 GPT-4o、ChatGPT 更新等内容。

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1239211766035251210)** (3 条消息): 

- **宣布 AlphaFold3 联邦启动**：AlphaFold3 联邦（AlphaFold3 Federation）见面会定于明天东部时间晚上 9 点（5 月 12 日）举行。主题包括 **Alpha Fold 3 集成的现状**、训练流水线架构、潜在瓶颈以及公开问答环节。查看详情并从[此处](https://lu.ma/swinnyfl)加入。

- **咨询服务器角色信息**：一名成员询问如何查找有关服务器角色的信息。该用户还特别提到了对“橙色团队（orange team）”的呼叫。

**提到的链接**：<a href="https://lu.ma/swinnyfl">AlphaFold3 [AF3] Federation Meet · Luma</a>：当前进展更新。由首席开发人员就 Alpha Fold 3 集成的当前状态进行的演讲。讨论在初始阶段遇到的任何问题……

  

---


**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1239333780695683124)** (3 条消息):

- **Fasteval 开发停止**：*tju01* 已确认不打算继续 **fasteval** 项目或任何相关的后续项目。他们愿意将 GitHub 项目的所有权转让给负责任的新所有者，否则，这里的 fasteval 频道可能会被归档。
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1238800597679865927)** (1 messages): 

- **关于 AI town 定制的查询**：一名成员询问是否可以更改 AI town 中的**角色移动速度**和 **NPC 数量**。目前尚未提供任何回复或进一步详情。
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1238801086161358921)** (1 messages): 

- **降低 NPC 交互频率以提升玩家参与度**：一位用户正在探索**降低 NPC 之间的交互频率**的方法，以便将更多算力分配给玩家与 NPC 的交互。他们提到正在使用带有 **llama3 模型**的 **AI town**，这对他们的本地机器造成了很大负担。
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=KQ-xGVFHDkw
  

---



**YAIG (a16z Infra) ▷ #[tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608/)** (1 messages): 

pranay01: 同意！
  

---



---