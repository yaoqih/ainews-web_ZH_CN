---
companies:
- google
- mistral-ai
- lmsys
- cohere
date: '2024-04-10T22:07:48.484098Z'
description: '以下是为您翻译的内容：


  *   **谷歌的 Griffin 架构**在长上下文环境下表现优于 Transformer，推理速度更快且内存占用更低。

  *   **Command R+** 在 LMSYS Chatbot Arena 排行榜上攀升至第 6 位，超越了 **GPT-4-0613** 和 **GPT-4-0314**。

  *   **Mistral AI** 发布了一个开源的 **8x22B 模型**，拥有 64K 上下文窗口和约 1300 亿（130B）总参数。

  *   **谷歌**开源了 **CodeGemma** 模型，并提供预量化的 4 位版本以实现更快的下载。

  *   **Ella 权重**通过大语言模型（LLM）增强了 Stable Diffusion 1.5 的语义对齐能力。

  *   **Unsloth** 在微调时可实现 4 倍的上下文窗口扩大和 80% 的内存减省。

  *   **Andrej Karpathy** 发布了用纯 C 语言实现的大语言模型（LLMs），以寻求潜在的性能提升。

  *   **Command R+** 借助 iMat q1 量化，可以在 M2 Max MacBook 上实时运行。

  *   **Cohere 的 Command R** 模型具有低廉的 API 成本和强劲的排行榜表现。

  *   **Gemini 1.5** 的音频处理能力令人印象深刻，能够识别音频片段中的语音语调和说话人身份。'
id: 8507691f-90e5-4413-84ba-debd5609a0f6
models:
- griffin
- command-r-plus
- gpt-4-0613
- gpt-4-0314
- mistral-8x22b
- codegemma
- stable-diffusion-1.5
- command-r
- gemini-1.5
original_slug: ainews-musics-dall-e-moment
people:
- andrej-karpathy
title: 音乐的 DALL-E 时刻
topics:
- model-architecture
- benchmarking
- open-source
- model-quantization
- memory-optimization
- inference-speed
- multimodality
- finetuning
- performance-optimization
- audio-processing
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月9日至4月10日的 AI 新闻。我们为您检查了 5 个 subreddits、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 以及 **26** 个 Discord 服务器（**388** 个频道和 **5893** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**600 分钟**。

当人们还在消化昨天的 [Gemini audio](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/?utm_source=ainews&utm_medium=email)、[GPT4T](https://twitter.com/miramurati/status/1777834552238723108?utm_source=ainews&utm_medium=email) 和 [Mixtral](https://twitter.com/_philschmid/status/1778051363554934874?utm_source=ainews&utm_medium=email) 重磅新闻时，今天迎来了 [Udio 的盛大发布](https://twitter.com/udiomusic/status/1778045325468426431)：

 
![image.png](https://assets.buttondown.email/images/a8f8a3c9-d95a-4250-9f10-1f8ef80eaf7d.png?w=960&fit=max)
 

你需要听一下帖子里的样本，将其与 Suno 进行对比，后者当然也有[自己的粉丝群](https://twitter.com/tobi/status/1775684945257611286)。Udio 在过去几天里[泄露得像筛子一样](https://x.com/legit_rumors/status/1777059367788982389)，所以这并不意外，但更令人惊讶的是 [Sonauto](https://news.ycombinator.com/item?id=39992817) *也*在今天发布，同样瞄准了音乐生成领域，尽管其完善程度要低得多。这感觉像是一个时机已经成熟的想法，但与 Latent Diffusion 不同的是，目前尚不清楚是什么突破让 Suno/Udio/Sonauto 几乎在同一时间涌现。你可以在 [Suno 的 Latent Space 播客](https://www.latent.space/p/suno)中听到一些线索，但在我们发布下一集音乐专题之前，你也只能了解到这些了。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取尚未实现，但即将推出。

以下是给定 Reddit 帖子中的关键主题和话题摘要，按类别组织并附有最相关的帖子链接：

**AI 模型与架构**

- **Google 的 Griffin 架构优于 Transformer**：在 /r/MachineLearning 中，Google 发布了一个采用新 Griffin 架构的模型，该模型[**在 MMLU 的受控测试和平均基准测试分数上，在多个尺寸上均优于 Transformer**](https://i.redd.it/triygw613htc1.jpeg)。Griffin 具有效率优势，在长上下文下具有更快的 Inference（推理）速度和更低的内存占用。
- **Command R+ 排名上升，超越 GPT-4 模型**：在 /r/LocalLLaMA 中，Command R+ 已[**攀升至 LMSYS Chatbot Arena 排行榜第 6 位，成为最强的开源模型**](https://www.reddit.com/r/LocalLLaMA/comments/1bzo2sh/latest_lmsys_chatbot_arena_result_command_r_has/)。根据[排行榜结果](https://chat.lmsys.org/?leaderboard)，它击败了 GPT-4-0613 和 GPT-4-0314。
- **Mistral 发布具有 64K 上下文的 8x22B 开源模型**：Mistral AI [**开源了其具有 64K context window（上下文窗口）的 8x22B 模型**](https://x.com/mistralai/status/1777869263778291896?s=46)。该模型总参数量约为 130B，每次 forward pass（前向传播）有 44B 激活参数。
- **Google 开源基于 Gemma 架构的 CodeGemma 模型**：Google 发布了 [CodeGemma，这是基于 Gemma 架构的开源代码模型](https://huggingface.co/blog/codegemma)，并上传了预量化的 4-bit 模型以实现 4 倍速下载，正如 /r/LocalLLaMA 中分享的那样。

**开源工作**

- **为 Stable Diffusion 1.5 发布 Ella 权重**：在 /r/StableDiffusion 中，这些权重[**使扩散模型具备了 LLM 能力，以增强语义对齐**](https://github.com/TencentQQGYLab/ELLA)。
- **Unsloth 的发布实现了微调时的显存减少**：在 /r/LocalLLaMA 中，Unsloth 通过在 GPU 和系统 RAM 之间使用异步卸载，提供了 [**4 倍大的上下文窗口和 80% 的内存减少**](https://www.reddit.com/r/LocalLLaMA/comments/1bzywjg/80_memory_reduction_4x_larger_context_finetuning/)。
- **Andrej Karpathy 发布纯 C 语言实现的 LLM**：在 /r/LocalLLaMA 中，这个纯 C 语言实现[**可能实现更快的性能**](https://www.reddit.com/r/LocalLLaMA/comments/1bztawh/andrejs_llms_in_pure_c_potentially_making_things/)。

**基准测试与对比**

- **Command R+ 模型在 M2 Max MacBook 上实时运行**：在 /r/LocalLLaMA 中，使用 iMat q1 量化，[**Inference（推理）可以实时运行**](https://v.redd.it/b5sn5at5mftc1)。
- **Cohere 的 Command R 模型在排行榜上表现良好**：在 /r/LocalLLaMA 中，Command R 在 Chatbot Arena 排行榜上表现优异，同时[**与竞争对手相比具有较低的 API 成本**](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)。

**多模态 AI**

- **Gemini 1.5 的音频能力令人印象深刻**：在 /r/OpenAI 中，Gemini 1.5 可以[**从纯音频片段中识别语音语调并按姓名识别说话人**](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/)。
- **多模态视频叙事入门工具包**：在 /r/OpenAI 中，该工具包利用 VideoDB、ElevenLabs 和 GPT-4 来[**生成纪录片风格的配音**](https://www.reddit.com/r/OpenAI/comments/1bzncf2/starter_kit_for_storytelling_using_multimodal/)。

---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**GPT-4 Turbo 模型改进**

- **改进的推理和编程能力**：[@gdb](https://twitter.com/gdb/status/1778071427809431789)、[@polynoamial](https://twitter.com/polynoamial/status/1777809000345505801) 和 [@BorisMPower](https://twitter.com/BorisMPower/status/1777867583947227582) 指出，与之前的版本相比，GPT-4 Turbo 的推理和编程性能有了显著提高。
- **正式全面可用（GA）**：[@gdb](https://twitter.com/gdb/status/1777776125139194252)、[@miramurati](https://twitter.com/miramurati/status/1777834552238723108) 和 [@owencm](https://twitter.com/owencm/status/1777770827985150022) 宣布 GPT-4 Turbo 现已结束预览，正式全面可用。
- **与旧版本的对比**：[@gdb](https://twitter.com/gdb/status/1778126026532372486)、[@nearcyan](https://twitter.com/nearcyan/status/1777893558072270889) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1777837161040990356) 分享了对比，并指出这次更新非常显著。

**Mistral AI 发布新的 8x22B 模型**

- **176B 参数 MoE 模型**：[@sophiamyang](https://twitter.com/sophiamyang/status/1777945947764297845) 和 [@_philschmid](https://twitter.com/_philschmid/status/1778051363554934874) 详细介绍了 Mistral AI 发布的 Mixtral 8x22B，这是一个拥有 176B 参数的 MoE 模型，具有 65K 上下文长度并采用 Apache 2.0 许可证。
- **评估结果**：[@_philschmid](https://twitter.com/_philschmid/status/1778083833507659997) 分享了 Mixtral 8x22B 在 **MMLU 上达到了 77%**。更多积极结果见 [@_philschmid](https://twitter.com/_philschmid/status/1778089353849290843)。
- **社区反响与获取途径**：许多人如 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1777904372091118026) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1777903886075875762) 表达了兴奋之情。根据 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1778117267005346286)，该模型已在 Hugging Face 和 Perplexity AI 上线。

**Google 新模型发布与公告**

- **Gemini 1.5 Pro 公开预览版**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777738279137222894) 宣布具有长上下文窗口的 Gemini 1.5 Pro 已在 Vertex AI 开启公开预览。根据 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321)，可通过 API 在 180 多个国家/地区使用。
- **Imagen 2 更新**：Imagen 2 现在可以创建 4 秒的动态图像，并包含一个名为 SynthID 的水印工具，由 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747320945234422) 和 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747324489306302) 分享。
- **CodeGemma 和 RecurrentGemma 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078071188304106) 宣布了用于编程的 CodeGemma 和注重内存效率的 RecurrentGemma，这是与 Google Cloud 合作完成的，详情见 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078073377706083) 和 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078075713982544)。

**Anthropic 关于模型说服力的研究** 

- **衡量语言模型说服力**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728366101119101) 开发了一种测试说服力的方法，并分析了跨模型世代的扩展情况。 
- **模型世代间的扩展趋势**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728370148577657) 发现新模型被认为更具说服力。Claude 3 Opus 在统计学上与人类的论点相似。
- **实验细节**：Anthropic 衡量了在阅读 LM 或人类关于极化程度较低问题的论点后，同意程度的变化，详见 [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728378675536357)、[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728376960106587)、[@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728375198568611)。

**Cohere 的 Command R+ 模型性能**

- **Chatbot Arena 顶尖开源权重模型**：[@cohere](https://twitter.com/cohere/status/1778113095820526038) 和 [@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471) 庆祝 Command R+ 在 Chatbot Arena 排名第 6，根据 13K+ 投票，它作为顶尖开源模型与 GPT-4 旗鼓相当。 
- **高效的多语言 Tokenization**：[@seb_ruder](https://twitter.com/seb_ruder/status/1778028863580188740) 详细介绍了 Command R+ 的分词器如何比其他分词器更高效地压缩多语言文本（1.18-1.85 倍），从而实现更快的推理和更低的成本。
- **获取途径与 Demo**：根据 [@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471) 和 [@nickfrosst](https://twitter.com/nickfrosst/status/1777724060257968505)，Command R+ 可在 Cohere 的 Playground (https://txt.cohere.ai/playground/) 和 Hugging Face (https://huggingface.co/spaces/cohere/command-r-plus-demo) 上使用。

**Meta 新 AI 基础设施与芯片公告**

- **下一代 MTIA 推理芯片**：[@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502) 宣布了 MTIAv2，这是 Meta 的第二代推理芯片，采用 TSMC 5nm 工艺，拥有 708 TF/s Int8 算力、256MB SRAM 和 128GB 内存。根据 [@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809)，其稠密计算能力是 v1 的 3.5 倍，稀疏计算能力是 v1 的 7 倍。
- **平衡计算、内存与带宽**：[@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809) 指出 MTIA 的架构优化了计算、内存带宽和容量之间的平衡，适用于排序和推荐模型。[@AIatMeta](https://twitter.com/AIatMeta/status/1778083241632604456) 表示，全栈控制使其随着时间的推移比 GPU 具有更高的效率。
- **不断增长的 AI 基础设施投资**：这是 Meta 增加 AI 基础设施投资以驱动新体验的一部分，是对现有和未来 AI 硬件的补充，[@AIatMeta](https://twitter.com/AIatMeta/status/1778083243050275143) 对此进行了强调。

**幽默与梗图**

- **向投资经理路演**：[@adcock_brett](https://twitter.com/adcock_brett/status/1777797999663493253) 幽默地建议永远不要向 VC 的投资经理（associates）进行路演，并根据十年无果的经验称其有害，他在 [@adcock_brett](https://twitter.com/adcock_brett/status/1778115667465740447) 中进一步阐述了这一观点。
- **护城河与开源**：[@abacaj](https://twitter.com/abacaj/status/1777801210826744035) 引用一个融资数百万美元的 GPT-4 套壳项目开玩笑说“根本没有护城河”。[@bindureddy](https://twitter.com/bindureddy/status/1777832694300475460) 预测开源将在年底前领跑 AGI 竞赛。
- **Anthropic 对 GPT-4 的反应**：[@nearcyan](https://twitter.com/nearcyan/status/1777788931272327311) 发布了一张梗图，推测 Anthropic 对 OpenAI “大幅改进”的 GPT-4 更新的反应。

---

# AI Discord Recap

> 摘要之摘要的摘要

**1) 新发布及即将发布的 AI 模型与基准测试**

- 备受期待的 **[Mixtral 8x22B](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)** 发布，这是一个拥有 176B 参数的模型，在 **AGIEval** 等基准测试中表现优于其他开源模型（[推文](https://x.com/jphme/status/1778030213881909451)）。官方分享了一个 [磁力链接](https://x.com/MistralAI/status/1777869263778291896)。

- Google 悄然推出了 **[Griffin](https://huggingface.co/google/recurrentgemma-2b)**（一个 2B 参数的循环线性注意力模型，[论文](https://arxiv.org/abs/2402.19427)）以及新的代码模型 **[CodeGemma](https://huggingface.co/spaces/ysharma/CodeGemma)**。

- OpenAI 的 **GPT-4 Turbo** 模型已发布，具备视觉能力、JSON 模式和 function calling，显示出比之前版本更显著的性能提升。讨论围绕其速度、推理能力以及构建高级应用的潜力展开。（[OpenAI 定价](https://openai.com/pricing), [OpenAI 官方推文](https://twitter.com/OpenAIDevs/status/1777769463258988634)）。该模型在 [基准测试对比](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing) 中与 **Sonnet** 和 **Haiku** 等模型并列讨论，显示出明显的性能增益。

- 对 **Llama 3**、**Cohere** 和 **Gemini 2.0** 等模型发布的期待，以及对其潜在影响的推测。

**2) 量化、效率与硬件考量**

- 讨论了 **HQQ** ([代码](https://github.com/mobiusml/hqq)) 和 **Marlin** 等 **量化（quantization）** 技术以提高效率，同时也关注如何保持困惑度（perplexity）。

- Meta 关于 **LLM 知识容量缩放定律** 的研究（[论文](https://arxiv.org/abs/2404.05405)）发现，**int8 量化** 在高效的 **MoE** 模型中能很好地保留知识。

- 在本地运行 Mixtral 8x22B 等大型模型的 **硬件限制**，以及对 **多 GPU 支持** 等解决方案的兴趣。

- 对来自 **Meta**、**Nvidia** 和 **Intel Habana Gaudi3** 等公司的 **AI 加速硬件** 的对比。

**3) 开源进展与社区参与**

- **LlamaIndex** 展示了 **企业级检索增强生成（RAG）** ([博客](https://t.co/ZkhvlI4nnx))，ICLR 2024 上的 **MetaGPT** 框架也利用了 RAG ([链接](https://t.co/sAF41j0uL4))。

- 新工具如用于 **合并 LLM 专家** 的 **mergoo** ([GitHub](https://github.com/Leeroo-AI/mergoo)) 和用于 **LoRA 层初始化** 的 **PiSSA** ([论文](https://arxiv.org/abs/2404.02948), [仓库](https://github.com/GraphPKU/PiSSA))。

- 社区项目：**everything-rag** 聊天机器人 ([HuggingFace](https://huggingface.co/spaces/as-cle-bert/everything-rag))、**TinderGPT** 约会应用 ([GitHub](https://github.com/GregorD1A1/TinderGPT)) 等。

- 社区成员在 **HuggingFace** 上快速开源了 Mixtral 8x22B 等新模型。

**4) 提示工程、指令微调与基准测试辩论**

- 关于 **提示工程（prompt engineering）** 策略的广泛讨论，如 **元提示（meta-prompting）** 和使用 AI 生成指令进行 **迭代优化**。

- **指令微调（instruction tuning）** 方法的对比：**RLHF** 与 **StableLM 2** 中使用的 **直接偏好优化（DPO）** ([模型](https://huggingface.co/stabilityai/stablelm-2-12b-chat))。

- 对 **基准测试（benchmarks）** 被“刷分”的怀疑，建议参考 **arena.lmsys.org** 等人工排名的排行榜。

- 围绕 **LLM2Vec** 将 LLM 用作 **文本编码器（text encoders）** 的讨论（[论文](https://arxiv.org/abs/2404.05961), [仓库](https://github.com/McGill-NLP/llm2vec)）及其实际效用。

---

# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**超分辨率团队部署技术**：工程师们讨论了如何使用超分辨率技术提升视频截图的图像质量。他们提到了 **RealBasicVSR**，许多人期待更先进的视频上采样器（upscalers）。

**激发 Stable Diffusion 创意**：新人询问如何使用 **Stable Diffusion** 创作原创内容，并获得了关于 GitHub 上的工具和仓库的指导。资深用户提供的 Demo URL 进一步支持了这些探索。

**自定义控制辩论升温**：参与者辩论了 **Stable Diffusion** 内部的自定义功能，包括特定数据集的构建、项目增强以及反映独特艺术风格的 **LoRAs**，这表明了模型输出高度个性化的趋势。

**驾驭 AI 法律迷宫**：对话还涉及 AI 生成内容的法律和伦理影响，讨论了版权问题、合法生成实践以及立法发展对该领域的潜在影响。

**热切期待 Stable Diffusion 3**：关于即将发布的 **Stable Diffusion 3** 有很多讨论，特别关注其对手部生成的能力，以及新模型是否需要 **negative prompts** 来避免不理想的输出。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **计算器 GUI 成就**：在 **Mistral-7b-instruct-v0.1Q4_0** 的性能评估中，它在性能测试中脱颖而出，轻松创建了一个带有 GUI 的基础计算器；同时讨论了 **Command R Plus** 需要大量的 **VRAM**，引发了关于本地服务器 API 请求和可能存在的 **VRAM** 瓶颈的讨论。

- **AutoGen vs. CrewAI - 自动化对决**：一位成员在评估用于本地 **LMs** 任务自动化的 **AutoGen**、**CrewAI** 等工具时陷入困境，他更倾向于 **AutoGen**，因为其易用性以及在结构化输入下的良好表现，同时在寻找能在 12GB **3080 GPU** 上运行的最佳模型。

- **Command R Plus Beta 令人兴奋**：LM Studio 的 **0.2.19 beta** 版本讨论了其最新功能和稳定性增强，成员们对 **Command R Plus** 模型在包括 M3 MacBook Pro 和支持 **AVX2** 的 AMD 机器在内的各种硬件上的兼容性和性能感到特别满意。

- **CodeGemma 隆重登场**：Google 推出的 **CodeGemma** 模型（提供 2B 和 7B 版本用于代码任务）引发了讨论，成员们正在测试其相对于 **Claude** 和 **GPT-4** 的能力。**LM Studio Community** 正在寻求关于这一新模型实力的进一步见解。

- **ROCM 与兼容性忧虑**：最近的 **0.2.19 ROCm Preview Beta-3** 对 **Command R Plus** 的支持引发了关于 **ROCM** 利用问题的对话，但即将发布的 Linux 版本让人感到宽慰。然而，关于 7800XT 兼容性的困惑仍未解决。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Checkpoint 挂起问题**：有用户反映 `TrainingArguments` 中的 `hub_strategy="all_checkpoints"` 导致 Checkpoint 文件夹无法成功推送到仓库。虽然分享了详细的 **training parameters**，但目前尚未出现明确的解决方案。

- **更长的上下文，更低的 VRAM**：Unsloth AI 的新版本实现了 **4 倍长的 context windows**，同时减少了 30% 的 VRAM 使用，运行时间仅增加 1.9%。他们还在开发一键式解决方案，以提供更流畅的微调体验和模型优化（[Long-Context 支持详情](https://unsloth.ai/blog/long-context)）。

- **周边创意，是捡便宜还是冤大头？**：社区讨论了推出 Unsloth 主题周边的可能性，起因是一位用户分享了无关的咖啡杯礼物。成员们还请求提供技术文档，特别是 **Hugging Face Json 文件文档**。

- **LLM 微调的高效方法**：关于优化 AI 聊天机器人微调的讨论强调了为 Alpaca 模型使用 **Alpaca format**，为聊天机器人使用 **ChatML template**，并强调了数据集与特定微调框架兼容性的必要性。

- **StegLLM 悄然登场**：介绍了一个名为 **StegLLM** 的新模型，它在 **mistral-7b-instruct-v0.2** 中嵌入了隐蔽机制，并由特定的“密钥”短语启动。模型制作者还分享了 **safetensors**，并表示灵感来自 Anthropic 的 **Sleeper Agents** 研究（[Hugging Face 上的 StegLLM](https://huggingface.co/AshScholar/StegLLM)）。

- **Multi-GPU 支持指日可待**：贡献者们强调了对即将推出的 Multi-GPU 支持的兴奋和技术考量。根据一篇 [arXiv 论文](https://arxiv.org/abs/2310.10195) 的建议，一种内存占用可能较低的 AdaLomo 优化器正在接受审查，预计将与 Unsloth AI 的未来更新同步推出。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity Pro 引发讨论**：社区成员正在剖析 **Perplexity Pro** 的优缺点，特别是对于学习 **Blender** 和 **Unreal Engine** 等工具的帮助，但一些用户指出其上下文长度与其他服务相比存在局限性，而 **Gemini 1.5** 因其视频和音频支持而脱颖而出。

**模型对比与推测**：讨论围绕 **Mistral 8x22b** 展开，这是一款被认为介于 **GPT-4** 和 **Sonnet** 之间的开源模型，尽管其高昂的计算需求限制了可访问性。此外，还有关于 "GPT-5" 和 "Gemini 2.0" 等未来模型的轻松调侃，并将其与备受期待的 "GTA 6" 发布相提并论。

**技术联动：Raycast 遇见 Perplexity**：**Raycast** 与 **Perplexity AI** 宣布合作，旨在将知识获取集成到 Mac 用户体验中，详见 [Perplexity 的推文](https://x.com/perplexity_ai/status/1778067977566294448)。此外，还有人提到 AI 在快速信息检索方面优于传统搜索引擎。

**走出实验室，进入代码世界**：针对 *Perplexity API* 的新 **Ruby client** 已经面世，同时用户正在分享处理大文本粘贴和数据提取模型选择的变通方法，并指出了 **199k tokens** 的上限。

**Perplexity API 的演进**：**API 余额充值**和**支付提交 Bug** 等技术问题得到了迅速处理，修复方案已就绪，如果问题仍然存在，欢迎发送私信。此外，还讨论了 **Perplexity API** 的实时网页响应能力，并明确了目前不支持 **Claude Opus model**。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**聊天机器人的精进**：**StableLM 2 12B Chat** 是一款拥有 120 亿参数的 AI，通过 Direct Preference Optimization (DPO) 针对聊天进行了优化。用户群体正在评估其相较于 SFT+KTO 和 DNO 等其他微调方法的影响；担忧主要集中在 DPO 的质量和伦理考量上。[在此获取 StableLM 2 模型](https://huggingface.co/stabilityai/stablelm-2-12b-chat)。

**Mixtral 的崛起**：早期基准测试表明 **Mixtral 8x22b 模型** 在 MMLU 评估中与 Command R+ 等顶级模型不相上下，引发了关于多样化微调数据集与继承自基座模型能力之重要性的讨论。[更多关于 Mixtral 8x22b 的细节](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)。

**模型量化的飞跃**：分享了关于量化方法的见解，特别是在 **OLMo-Bitnet-1B** 的背景下，重点关注 Quantization Aware Training (QAT) 和 Straight-Through Estimator 的使用，突显了对模型效率的持续关注。[关于 Straight-Through Estimator 的论文在此](https://arxiv.org/abs/1308.3432)。

**合成以致胜**：一篇介绍在模型训练期间结合合成数据与真实数据概念的论文引发了关于合成数据“近亲繁殖”潜力的辩论，以及其对模型知识库多样性的影响和模型崩溃（model collapse）的风险。[论文链接在此](https://arxiv.org/abs/2404.01413)。

**期待 WorldSim 更新**：社区对 **WorldSim** 即将到来的更新表现出兴奋，讨论涉及该平台的多语言支持以及使用 Nous Hermes Mixtral 等模型模拟类似体验的替代方案。当前的本地硬件也被指出不足以运行此类先进模型。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**RNN 进展揭秘**：研究人员证明，用于 Transformer 的**可解释性工具（interpretability tools）**对现代 RNN（如 **Mamba 和 RWKV**）具有显著的适用性，并通过[研究论文](https://arxiv.org/abs/2404.05971)和 [GitHub 仓库](https://github.com/EleutherAI/rnngineering)分享了见解。这激发了社区的进一步参与并分享了研究方法，鼓励协作开发 RNN 语言模型。

**神秘的 Claude 3 Opus 尺寸引发猜测**：AI 社区对 **Claude 3 Opus** 未公开的模型尺寸议论纷纷，这与 **GPT-4** 规模的透明度形成了鲜明对比。与此同时，Google 的 **Gemini 项目** 因其保守的图像生成政策及其项目安全负责人的争议性观点而面临审查。

**GPT-4 Turbo 基准测试**：工程师们正在寻找 OpenAI 最新模型（特别是 **gpt-4-turbo**）的可靠基准测试信息。缺乏此类数据使得比较和进度评估变得具有挑战性。

**AI 治理获得立法关注**：由国会议员 **Adam Schiff** 提出的《生成式 AI 版权披露法案》（*Generative AI Copyright Disclosure Act*）成为一项重点立法努力，旨在提高 AI 使用受版权保护材料的透明度，为未来对该行业的监管影响奠定基础。

**通过 LLM 涌现的文本嵌入**：围绕 **LLM2Vec** 出现了一项新的尝试，该项目将 decoder-only LLMs 转换为 encoders，并声称提升了性能，这引发了关于与其他模型比较的公平性及其实际效用的辩论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **艺术家还是算法？**：关于 AI 是否可以被视为合法艺术家的活跃讨论，突显了人们对 AI 生成艺术对人类创造力认可和价值评估影响的担忧。
- **学术界中的 AI**：一名硕士生正在考虑将 **LM Studio** 和 **Open-Source LLM Advisor** 作为潜在资源，为其论文项目实现一个基于 GPT 的聊天系统。
- **Perplexity 获得认可**：用户称赞了 **Perplexity**，特别是其 Pro 版本，因其具备 32K 上下文窗口以及在 **Opus** 和 **GPT-4** 等模型之间灵活切换的能力。
- **定制化需求清单**：社区中对于未来 **GPT** 迭代提供更高定制化程度的呼声日益增长，特别是在回答简洁度和输出排名方面。
- **GPT-4 难题与 Prompt 创作**：从加载问题到 API 访问中断等 **GPT** 技术问题已被标记，同时社区对分享 AI 越狱 Prompt 持反对立场。通过迭代的 **prompt engineering** 和使用 meta-prompts 来提高指令精确度引起了关注，这提醒了记录良好的 AI 交互具有不可替代的价值。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **自主软件开发的进展**：新加坡推出的 **AutoCodeRover** 标志着向自主软件工程迈出的重要一步，能够高效处理与 **GitHub** issue 相关的 Bug 修复和功能增强。这一创新强调了 AI 在降低成本和提高速度的情况下，彻底改变软件维护和开发流程的潜力。详细信息和预印本可在 [GitHub Repository](https://github.com/nus-apr/auto-code-rover) 和 [Preprint PDF](https://github.com/nus-apr/auto-code-rover/blob/main/preprint.pdf) 中找到。

- **GPT-4-Turbo 带来的 AI 语言模型演进**：**GPT-4-Turbo** 的发布代表了语言模型能力的显著进步，在推理和复杂任务处理性能上表现出大幅提升。对其部署的期待和分析突显了使 AI 工具更强大、更易用的持续进展。价格和推出更新可在 [OpenAI Pricing](https://openai.com/pricing) 和 [OpenAI's Official Tweet](https://twitter.com/OpenAIDevs/status/1777769463258988634) 查看。

- **音乐生成技术的创新**：**Udio** 作为音乐生成领域潜在的游戏规则改变者，引发了关于其用于创作音乐的高级文本提示系统的讨论。凭借慷慨的 Beta 测试版，Udio 对音乐行业的影响及其与 Suno 等竞争对手的比较受到了爱好者和专业人士的密切关注。更多见解可在 [Udio Announcement](https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 和关于 [Udio 的 Reddit 讨论](https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/) 中探索。

- **1-bit 大语言模型 (LLMs) 的突破**：关于 **1-bit LLMs** 的讨论，特别是 **BitNet b1.58** 模型，展示了通过降低模型精度而不显著损害性能，向具有成本效益的 AI 迈出的创新一步。这一进展为模型效率和资源利用提供了新视角，详见 [arXiv 提交论文](https://arxiv.org/abs/2402.17764)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Gemma 1.1 Instruct 优于前代版本**：**Gemma 1.1 Instruct 7B** 表现出比前一版本更好的前景，现已在 HuggingChat 上线，并正吸引用户探索其能力。可以通过 [此处](https://huggingface.co/chat/models/google/gemma-1.1-7b-it) 访问该模型。

**CodeGemma 步入开发领域**：推出了一款用于设备端代码补全的新工具 **CodeGemma**，提供 2B 和 7B 版本，支持 8192k 上下文。它与近期发布的非 Transformer 模型 **RecurrentGemma** 一起，可以在 [此处](https://huggingface.co/spaces/ysharma/CodeGemma) 找到。

**HuggingFace 降价行动**：HuggingFace 宣布 Spaces 和 Inference endpoints 的 **计算价格下调 50%**。从 4 月起，这些服务在性价比上将优于 AWS EC2 按需服务。

**社区博客改版**：社区博客已改版为“文章 (articles)”，增加了点赞和增强曝光等功能。点击 [此处](https://huggingface.co/blog/community) 体验全新的文章格式。

**Serverless GPU 上线及机器学习课程更新**：HuggingFace 展示了与 Cloudflare 合作的 Serverless GPU 推理功能，并在其“游戏机器学习 (ML for Games)”课程中增加了一个关于“游戏中经典 AI”的额外单元。通过 [此链接](https://huggingface.co/blog/cloudflare-workers-ai) 了解 Serverless GPU 推理，并在 [此处](https://huggingface.co/learn/ml-games-course/unitbonus1/introduction) 探索课程新内容。

**Python 调试技巧**：在 JAX 或 TensorFlow 中利用 **eager execution**，使用 Python 的 `breakpoint()` 函数，并移除 PyTorch 实现以进行有效的调试。

**AI 水印去除工具发布**：推荐了一款旨在去除图像水印的 AI 工具，这对处理大量带水印图像的用户很有帮助。在 [GitHub](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy) 上查看该工具。

**GPT-2 的摘要困境与 Prompt 策略**：一位用户在使用 **GPT-2** 进行摘要时遇到挑战，这可能暗示了 Prompt 需与模型训练时代保持一致的重要性，建议可能需要更新指令或使用更适合摘要任务的新模型。

**应对 CPU 与 GPU 挑战**：讨论了在使用 contrastive loss 时，通过 accumulation 或 checkpointing 等技术来解决 batch size 限制的方法，并确认了 *batchnorm* 可能存在的更新问题。通过 `nvidia-smi` 监控 GPU 使用情况成为高效资源管理的关注点。

**Diffuser 去噪步数对图像质量的影响**：对 Diffusers 的探索表明，图像质量会随着 **denoising step** 计数的改变而波动。文中详细说明了 ancestral sampler 在质量差异中的作用，并为分布式多 GPU 推理提供了指导，特别是针对处理 **MultiControlnet (SDXL)** 等模型的巨大显存需求。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Pro 1.5 与 GPT-4 Turbo 取得新突破**：OpenRouter 引入了 [具有 1M token 上下文的 Gemini Pro 1.5](https://openrouter.ai/models/google/gemini-pro-1.5) 和 [具备视觉能力的 GPT-4 Turbo](https://openrouter.ai/models/openai/gpt-4-turbo)，标志着其模型阵容的重大升级，旨在满足高级开发需求。

- **模型下架与新模型发布**：OpenRouter 概述了针对 jebcarter/Psyfighter-13B 等冷门模型的退役计划，并向社区预告了新的 [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b)，这是一个具备 Instruct 能力的模型，并邀请用户提供宝贵的反馈以进行优化。

- **logit_bias 参数在多模型中得到增强**：技术社区现在可以通过将 `logit_bias` 参数扩展到更多模型（包括 [Nous Hermes 2 Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)）来增强对模型输出的控制，从而提高模型响应的精准度。

- **澄清模型集成与速率限制 (Rate Limits)**：由 **Louisgv** 发起的讨论引导用户完成了将新 LLM API 与 OpenRouter 集成的过程，并解决了关于 **Gemini 1.5 Pro** 等新预览模型速率限制的困惑，目前该模型的请求限制约为每分钟 10 次。

- **优化与故障排除讨论升温**：包括 **hanaaa__** 在内的用户正在交流在 SillyTavern 等各种平台上优化 **Hermes DPO** 等模型的策略，同时也报告并排查了 OpenRouter 网站的技术故障以及 TogetherAI 服务的延迟问题。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Meta 转型为超级赞助商：** Meta 通过提供 **420 万 GPU 小时** 的巨额赞助来加强其对 AI 研究的承诺，用于 Scaling Laws 研究，促进了对 Language Model (LM) 知识容量的研究，这相当于近五个世纪的计算时间。完整细节可以在 [scaling laws study](https://arxiv.org/abs/2404.05405) 中找到。

**CUDA 在 LLM 训练中占据核心地位：** 一项旨在围绕 CUDA 相关项目组建工作组的协作努力已经启动，实现 CUDA 算法的热情正在增长，正如在 [llm.c repository](https://github.com/karpathy/llm.c/tree/master/dev/cuda) 中关于将 GPT-2 移植到 CUDA 的讨论所见。

**优化矩阵乘法：** 当遵循矩阵形状和内存布局时，可以实现矩阵乘法的性能提升。据报道，使用 Tiling 的最佳矩阵乘法配置为 `A: M=2047, K=N=2048`，以避免未对齐的内存布局，详见博文 ["What Shapes Do Matrix Multiplications Like?"](https://www.thonking.ai/p/answer-key-what-shapes-do-matrix)。

**AI 模型中的量化困境：** 社区围绕 Half-Quadratic Quantization (HQQ) 的实现以及 Marlin Kernel 在矩阵乘法中表现平平展开了激烈讨论。人们担心量化技术会影响模型的 Perplexity，HQQLinear 的调优受到审查，并与 GPTQ 的结果进行了对比。

**Flash Attention 与 CUDA 专业知识：** CUDA Kernel 的 “Flash” 版本代码最初表现不佳，但后来通过协作排查优化执行实现了加速。同时，[llm.c project](https://github.com/karpathy/llm.c) 成为那些渴望加强 CUDA 技能的人的首选学习资源，讨论涉及 OpenMP 的效用以及为提升性能而进行的自定义 CUDA 调试。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Whisper 不说话，它在倾听**：**Whisper** 被澄清为一个 Speech-to-Text 模型，**Ollama** 本身并不支持它，但可以在本地使用或配合来自同一开发者的替代后端使用。

**LangChain 的局限性与应用**：对于简单的 AI Assistant 任务，**LangChain** 相比 OpenAI API 可能没有显著优势，但在需要超出 OpenAI 范围的集成场景中表现出色，例如 [RAG 性能评估](https://docs.smith.langchain.com/cookbook/testing-examples/ragas) 等实际用例。

**TinderGPT 右滑自动化**：一款名为 **TinderGPT** 的新应用已经创建，旨在自动化 Tinder 对话并确保约会，欢迎在 [*GitHub*](https://github.com/GregorD1A1/TinderGPT) 上贡献代码。

**通过结构化输出比较 LLM**：分享了一项比较各种开源和闭源 Large Language Models 结构化输出性能的分析，详见此 [*GitHub 页面*](https://github.com/mattflo/structured-output-performance)。

**AI 处于时尚前沿**：分享了一个演示 AI Agent 模拟虚拟试穿服装的视频，旨在彻底改变时尚电商领域——点击[此处](https://youtu.be/C94pTaKoLbU)观看演示。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **药物识别获得 RAG 升级**：一个 **Multimodal RAG 应用** 现在可以通过合并视觉和描述性数据从图像中识别药物，展示在 [activeloop 的博文](https://t.co/QkuLzs34IJ)中。
- **为企业级 RAG 做好准备**：即将到来的合作承诺将揭示 **企业级 Retrieval-Augmented Generation (RAG)** 的构建模块，讨论重点是高级解析和可观测性，详见 [Twitter](https://t.co/ZkhvlI4nnx)。
- **MetaGPT 带着 RAG 秘籍空降 ICLR**：在 ICLR 2024 上，**MetaGPT** 将作为软件团队协作的多智能体框架首次亮相，并加入了现代化的 RAG 功能，详见此[公告](https://t.co/sAF41j0uL4)。
- **掌控 Agentic RAG**：目前的讨论强调了执行控制工具对于像旅游代理和 RAG 这样的 Agentic 系统的重要性，更多见解可在 [Twitter](https://t.co/ByGOaqgWMd) 上获得。
- **Gemini Meets LlamaIndex**：AI 工程师正在积极为 **Gemini LLM** 适配 **LlamaIndex 的示例 Notebook**，可通过 [GitHub](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb) 获取资源和指导。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Pixart Sigma 的快速渲染与质量瑕疵**：**Pixart Sigma** 在 3090 上展示了令人印象深刻的 ***8.26 秒*** Prompt 执行时间，但因输出图像“崩坏”而面临批评，这暗示了开源模型在质量控制方面存在问题。

**Mistral 实力倍增**：**Mistral 22b x 8** 的发布引发了热议，社区对其与 **mistral-large** 相比的能力表现出浓厚兴趣。一个用于下载 **mixtral-8x22b** 的磁力链接被分享，但未附带进一步说明。

**质疑 AI 中的回声室效应**：最近的一篇 [论文](https://arxiv.org/abs/2404.04125) 挑战了 CLIP 等多模态模型中预期的“Zero-shot”泛化能力，并强调了性能对预训练期间所见数据的依赖性。

**Google 的 Griffin 引发关注**：根据 [Reddit 讨论](https://www.reddit.com/r/MachineLearning/comments/1b3leks/deepmind_introduces_hawk_and_griffin_r/)，Google 推出的 Griffin 模型架构增加了显著的 10 亿参数，承诺将带来性能提升。

**直接纳什优化（Direct Nash Optimization）优于 RLHF**：
一项 [新研究](https://arxiv.org/abs/2404.03715) 为 LLM 提出了一种比 RLHF 更复杂的替代方案，采用“成对（pair-wise）”优化，据称即使在 7B 参数模型上也取得了显著成果。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GPT-4 强势登场，但保持低调**：人们对现已集成视觉能力且性能超越前代产品的 **GPT-4** 感到非常兴奋；尽管如此，详细信息似乎依然稀缺，OpenAI 的发布日志仍是了解其功能更新的首选。
  
- **Command r+ 的卓越表现与硬件要求**：**Command r+** 因其在角色扮演场景中的精准度而受到推崇，被认为优于包括旧版 GPT-4 在内的先前模型；然而，用户指出运行它可能需要沉重的硬件负担，超出了 4090 GPU 的承载能力。

- **01 设备进入 DIY 组装阶段**：成员们正利用 [GitHub](https://github.com/OpenInterpreter/01?tab=readme-ov-file) 上提供的 BOM 清单零件和 3D 打印外壳组装自己的 **01 设备**，通过在电脑上直接运行 Open Interpreter 绕过了对 Raspberry Pi 的需求。

- **01 设备 WiFi 连接问题的解决方法**：遇到 01 连接 WiFi 困难的用户通过恢复出厂设置并访问 [captive.apple.com](http://captive.apple.com) 成功解决了问题；可能需要删除旧凭据，而那些使用本地 IP 地址进行配置的用户通过 MacOS 找到了解决方案。

- **01 订单的静默排队**：DIY **01 机器**的订单更新目前被描述为“仍在准备中（still cooking）”，并承诺一旦有更多消息将通过邮件更新；这是对有关订单状态的客户服务查询的回应。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Google 的 RL 惊喜**：Google 推出了 **Griffin**，这是一个拥有 20 亿参数的循环线性注意力（recurrent linear attention）模型，标志着其较前身 CodeGemma 的重大飞跃。正如其 [arXiv 论文](https://arxiv.org/abs/2402.19427) 中详述的那样，Griffin 模型的架构与 RWKV 有相似之处。

**重新思考 RLHF 的有效性**：一场新的讨论集中在通过迭代反馈改进 LLM 的训练后阶段，这可能与传统的 RLHF 方法相媲美。讨论中对拒绝采样（Rejection Sampling）的有效性以及模型优化过程中对 Benchmark 的过度强调表示了担忧，反映出对 [近期论文](https://arxiv.org/abs/2404.03715) 中提到的更具实践性的开发方法的渴望。

**LLM 的预测**：一项由 Meta 支持的新研究揭示了 12 条 [LLM 缩放法则（Scaling Laws）](https://arxiv.org/abs/2404.05405)，投入了 4,200,000 GPU 小时来解析知识容量。有趣的是，**int8** 量化能有效地保持知识容量，这一发现对于资源效率和 **Mixture of Experts (MoE)** 模型的潜在应用都至关重要。

**围绕 Mixtral 的热议**：Mixtral 作为模型领域的新选手，因其与 Mistral 和 Miqu 的差异化而引发讨论。模型发布的激增，包括对 Llama 3 smol 和 Cohere 等模型的期待，表明 AI 开发正处于竞争加速期，正如 [此处](https://fxtwitter.com/sophiamyang/status/1777978822199017728) 的 Twitter 线程所讨论的那样。

**Benchmark：临时的衡量标准**：虽然大家一致认为针对 AlpacaEval 等 Benchmark 进行优化可能与模型真正的优越性不相关，但它们作为进步的阶段性指标仍具有效用。开发者们正倡导后均衡（post-equilibrium）方法，重点在于改进数据和缩放，而不是盲目追求分数。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 迎来精简**：工程师们已启动 *tinygrad* 的重构工作，以降低代码复杂度并提高可读性，主张调整 JIT 支持并移除底层的 diskbuffers，如 [PR #4129](https://github.com/tinygrad/tinygrad/pull/4129) 所示。

- **寻求权重无关（Weight Agnostic）方法**：关于使用 tinygrad 创建权重无关网络的讨论正受到关注，重点在于将此类网络部署用于游戏训练，并考虑使用 ReLU 激活函数。

- **MNIST 与 Tinygrad 融合**：MNIST 集成到 tinygrad 的工作正在推进，[Pull Request #4122](https://github.com/tinygrad/tinygrad/pull/4122) 是其中的代表，该过程还发现了一个 AMD 上的编译器 bug，促使增加 CI 测试以检测未来类似的这类问题。

- **全局变量优于局部变量**：在 *abstractions3* 重构中关于变量作用域的辩论后，进行了一次更新，将 **var_vals** 变为全局字典，而此前它在每个 **ScheduleItem** 中属于局部作用域。

- **Tinygrad 用户指南发布**：对于有兴趣通过自定义加速器增强 tinygrad 的用户和开发者，现在可以参考这份详细的[指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md)，并建议探索 tinygrad 仓库中 `examples/` 目录下的不同网络示例。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Mixtral 8x22B 引发关注**：社区正在讨论新的 **Mixtral 8x22B 模型**，该模型拥有约 1400 亿参数，在 rank32 下运行且 loss 异常低；目前尚不清楚该模型是经过指令微调（instruction tuned）还是基座模型（base model）。开发者对 **quantization**（量化）技术表现出浓厚兴趣，以使 Mixtral 8x22B 这样的大型模型更易于管理，这表明需要在模型大小与资源限制之间取得平衡。

**PiSSA 承诺精准性能**：一种名为 **PiSSA** 的新型 **LoRA 层初始化技术**被分享，该技术使用原始权重矩阵的 SVD 分解，有望获得更好的微调效果，详情见 [arXiv 摘要](https://arxiv.org/abs/2404.02948)和 [GitHub 仓库](https://github.com/GraphPKU/PiSSA)。

**数据集困境与投入**：成员们正积极寻找和分享数据集，例如 [Agent-FLAN 数据集](https://huggingface.co/datasets/internlm/Agent-FLAN)，它对函数调用（function-calling）和 JSON 解析非常有用，有助于有效微调 LLM。另一位成员讨论了使用挪威艺术数据集预训练模型以增强其语法能力，并获得了关于数据表示格式的建议。

**模型托管障碍**：一位贡献者迅速响应，将新的 **Mixtral-8x22B 模型**上传到 Hugging Face，展示了社区快速贡献的文化。与此同时，关于在双 24GB GPU 配置上运行 **mixtral-qlora-fsdp** 模型的硬件能力问题，以及寻找兼容各种 AI API 的 Web 自托管前端的问题仍未得到解答。

**三星搭建舞台**：三星宣布将于 5 月 11 日在纽约举办 **Samsung Next 2024 Generative AI Hackathon**，届时将探索健康与福祉（Health & Wellness）以及媒体技术（Mediatech）赛道，详情见 [Samsung Next AI Hackathon](https://lu.ma/nextgenainyc)。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 世界里的 C++ 老传统**：虽然 Mojo 开发者期待 Python 风格的 `f` 字符串，但目前他们通过导入 `_printf as printf` 来使用 C 风格的格式化，不过有提醒称这一特性可能不会永久保留。

**Mojo API 指南触手可及**：一位成员分享了一个 [Notion 页面](https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078)，将 API 文档翻译成对初学者友好的摘要，为 Mojo 新手提供帮助。

**Mojo 的并发难题**：Mojo 的 async/await 和协程（coroutines）实现仍在进行中，与 Python 有所不同；详情在 [Mojo 文档](https://docs.modular.com/mojo/stdlib/builtin/coroutine)中有所阐述，但根据[路线图](https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with)，目前缺少 `async for` 和 `async with`。

**令人烦恼的变长泛型**：社区中因提到“异构变长泛型”（Heterogeneous variadic generics）而引发了一阵困惑，这个术语概括了编程语言中高级类型系统的复杂性。

**Mojo UI 追求原生外观**：Mojo-UI 项目的活跃开发引发了关于与 Objective-C 集成以及访问 AppKit 框架的讨论。雄心勃勃的集成目标可能需要一个特殊的绑定层，详情可关注 [GitHub](https://github.com/moosems/mojo-ui)。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mixtral 与 Hugging Face 结合**：**Mixtral-8x22B** 模型已添加到 Hugging Face，并附带了[详细文档](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)，凭借其 Apache 2.0 许可证顺利成为关注焦点。为了促进这一集成，官方提供了转换脚本，包括一个针对早期 **Mixtral** 模型的脚本（[MoE 转换脚本](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py)）和另一个针对最新版本的脚本（[新 Mixtral 转换脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py)）。

- **种子下载与训练热潮**：*Mixtral 8x22b* 模型通过[磁力链接（magnet torrent link）](https://x.com/MistralAI/status/1777869263778291896?t=vKiT9FUuVbYAhjjg5kOHyw&s=33)迅速引发讨论，供急切的下载者使用。同时，该模型在 **AGIEval** 中展现了超越其他基座模型的强大性能。所有测试均在 *4xH100 GPUs* 配置上完成，值得注意的是，MMLU 任务的运行时间约为 10 小时。

- **Mergoo 混合模型**：受近期研究启发，旨在简化多个 LLM 专家模型合并的新工具 [**mergoo**](https://github.com/Leeroo-AI/mergoo) 加入了讨论。讨论中还提到了 **DiscoLM_German_7b** 模型中出现的异常行为模式，特别是受 ChatML 模板中换行符的影响，专业人士将其归因于可能的 tokenizer 配置问题（[tokenizer config](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48)）。

- **文本换行引发的行为之谜**：对换行符格式的一种奇特敏感性让工程师们陷入了狂热讨论，推测这种干扰是 **LeoLM 特有的怪癖**，还是影响其他模型的更广泛现象，亦或是该模型独特处理架构中出现的新特征。

- **基准测试波动成为热门话题**：Mixtral 8x22B 和 Mixtral 8x7B 等模型在 PIQA、BoolQ 和 Hellaswag 等各种数据集上的**基准测试分数**差异成为了城中热点。成员们传阅分数，并对虚拟 LLM 在 10 小时内完成 MMLU 任务的强大能力深感惊叹。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **早起的鸟儿有 AI 新闻**：一声亲切的 "gm" 伴随着 [OpenAI 的一条 Twitter 帖子](https://twitter.com/OpenAI/status/1777772582680301665)开启了新的一天，暗示了值得关注的新更新或讨论。
- **视觉震撼：超越 GPT-4 Turbo**：快速视觉基准测试的惊人结果显示，**Sonnet 和 Haiku** 略胜 **GPT-4 Turbo 和 Opus**，相关发现已在 [Colab 研究文档](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing)中分享。
- **GPT-4 Turbo 炫耀新技巧**：围绕 **GPT-4 Turbo** 的 function calling 和 JSON mode 的讨论升温，引发了对其构建强大视觉模型潜力的兴趣。
- **增量还是创新？**：在轻松的玩笑中，成员们争论最新的更新究竟代表了向 **GPT-4.5** 的重大飞跃，还是向 **4.25** 迈出的一小步，而一些人则强调了 OpenAI 员工关于推理能力提升的说法。
- **代码层面的比较讨论**：AI 工程师比较了各 AI 模型的编程能力，重点关注了对 Cursor 友好的模型使用、**Gemini 1.5** 以及 **copilot++** 的特性，但尚未达成明确共识。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM 帮助命令的速度至关重要**：用户对 `llm --help` 命令的缓慢性能表示担忧，在某个案例中该命令耗时超过 **2 秒** 才完成，这引发了对系统健康状况的警示。
- **LLM 命令的快速响应**：一份对比报告指出 `llm --help` 可以在短短 **0.624 秒** 内执行，这表明性能问题可能是个案而非普遍现象。
- **Docker 的差异**：在对 `llm --help` 进行基准测试时，一位用户注意到命令执行时间的巨大差异：在其原生系统上耗时高达 **3.423 秒**，而在 Docker 容器内则缩短至更可接受的 **0.800 秒**，这暗示了配置问题。
- **重新安装解决烦恼**：一位用户发现重新安装 `llm` 不仅提升了 `llm --help` 的速度（从几秒钟降至零点几秒），还修复了运行 Claude 模型时的错误。
- **MacOS 上的 LLM 谜团**：在 macOS 上，`llm cmd` 的执行在 iTerm2 中会挂起，而同样的设置在远程 Ubuntu 服务器上却能成功运行，这表明可能与 macOS 中自定义的 shell 环境存在冲突。

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **显微镜下的基准测试**：围绕 **phi-2**、**dolphin** 和 **zephyr** 等模型使用 **HumanEval dataset** 进行 **benchmark comparisons** 的重要性展开了讨论，并引用了 [arena.lmsys.org](https://arena.lmsys.org/) 作为一个更可靠的人类排名排行榜，这可能解决基准测试被操纵的担忧。
  
- **Mistral 的基准测试优势**：**Mistral 8x22b** 在 **AGIEval results** 中展示了显著的性能，Jan P. Harries 的更新吹嘘了其相对于竞争对手开源模型的优势，详见他的推文 [这里](https://x.com/jphme/status/1778030213881909451) 和 [这里](https://x.com/jphme/status/1778028110954295486)。

- **当离题不再是禁忌**：一位用户分享了一个没有上下文的 YouTube 视频链接：[在 YouTube 上观看](https://www.youtube.com/watch?v=Gb--4supXoo)。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **调整 GPU 以提高利用率**：一位社区成员报告称，通过将 **`-ngl` 值调整为 3**，实现了实质性的性能提升，特别是对于那些能更舒适地适应其 GPU 有限显存容量的小型模型。

- **智能适应 VRAM 限制**：有人提问关于增强 **llamafile** 以根据可用 VRAM 自动卸载模型层的问题，这将防止在 1050 等低端 GPU 上发生崩溃。

- **对 ollama 效率的认可**：**ollama** 项目因其在 GPU 间高效处理模型层分布而受到赞赏，正如 GitHub 上该项目的 [server.go](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43) 中特定的实现代码片段所示。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **节奏重混：AI 的最新音轨**：工程师们沉浸在一个新的 *remix music model* 中，其输出给社区留下了深刻印象；一位成员建议在 [SynthTrails](https://linktones.synthtrails.com/linktone/kanye) 听一听。
- **代码求助：工程师寻求专家帮助**：一位有需要的用户寻求编程协助，要求直接沟通以解决特定的技术挑战。



---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂太久，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1227134524501000242)** (985 messages🔥🔥🔥): 

- **超分辨率技术讨论**：成员们分享了关于使用超分辨率技术（如结合相邻帧）提高视频截图图像质量的见解，但指出像 [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) 这样的现有方法可能会随着更先进的视频上采样器的预期而过时。

- **Stable Diffusion 与模型生成探索**：新贡献者寻求关于使用 Stable Diffusion 生成图像的建议，并被引导去探索当前用户分享的 GitHub 仓库和工具，以及 demo URL。

- **自定义控制模型与增强功能的咨询**：用户对 Stable Diffusion 的特定用例表示感兴趣，例如构建特定数据集、增强某些项目类别、个性化模型（'loras'）以及对齐特定的艺术风格。

- **法律与伦理讨论**：聊天涉及了敏感话题，如版权、合法生成、AI 内容创作的合法性以及 AI 治理的未来，包括立法行动对 Stable Diffusion 和 LLM 可能产生的影响。

- **对 Stable Diffusion 3 的期待**：讨论围绕 SD3 相对于 cascade 等变体的预期改进展开，重点在于图像中生成逼真手指的局限性，以及关于新模型能力的问题，以及它们是否需要 negative prompts。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1core/post">Stability AI - 开发者平台</a>：未找到描述</li><li><a href="https://var.vision/demo">Template</a>：未找到描述</li><li><a href="https://ella-diffusion.github.io">ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>：未找到描述</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion 基准测试：45 款 Nvidia、AMD 和 Intel GPU 对比</a>：哪款显卡提供最快的 AI 性能？</li><li><a href="https://www.youtube.com/@AIchemywithXerophayze-jt1gg">AIchemy with Xerophayze</a>：看看 XeroGen，我们全新的终极 Prompt 锻造工具，适用于多个 AI 图像生成平台。专为更好地适应工作流而设计，为您提供 Prompt 创建的终极控制权 https://shop.xerophayze.c...</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOc">DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们对 1.5 的 ELLA 训练进行了逆向工程，并成功制作了其微调版本。我们正在努力调整脚本以使其适用于 SDXL。对他们没有发布它感到非常失望。所以...</li><li><a href="https://www.youtube.com/@latentvision/videos">Latent Vision</a>：未找到描述</li><li><a href="https://soundcloud.com/4dreamsy/blondies-and-weed">Blondies and weed</a>：在 #SoundCloud 上收听 4dreamsy 的 Blondies and weed #np</li><li><a href="https://stability.ai/stable-video">Stable Video &mdash; Stability AI</a>：Stability AI 首款基于图像模型 Stable Diffusion 的开源生成式 AI 视频模型。</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOcSJDD650A">DataVoid e/acc (@DataPlusEngine) 的推文</a>：我们对 1.5 的 ELLA 训练进行了逆向工程，并成功制作了其微调版本。我们正在努力调整脚本以使其适用于 SDXL。对他们没有发布它感到非常失望。所以...</li><li><a href="https://tenor.com/view/thumbs-up-approve-okay-ok-anime-gif-15533543">点赞批准 GIF - 点赞批准 好的 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.runpod.io/gpu-instance/pricing">GPU 实例定价</a>：未找到描述</li><li><a href="https://github.com/TencentQQGYLab/ELLA">GitHub - TencentQQGYLab/ELLA: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>：ELLA: 为 Diffusion Models 配备 LLM 以增强语义对齐 - TencentQQGYLab/ELLA</li><li><a href="https://www.youtube.com/watch?v=q5MgWzZdq9s">Stable Diffusion Forge UI：底层探索 - 技巧与窍门 #stablediffusion</a>：在这段视频中，我们将详细查看 Stable Diffusion Forge UI，涵盖从查找和更新模型、设置到增强功能的所有内容...</li><li><a href="https://github.com/ckkelvinchan/RealBasicVSR">GitHub - ckkelvinchan/RealBasicVSR: “调查真实世界视频超分辨率中的权衡”官方仓库</a>： “调查真实世界视频超分辨率中的权衡”官方仓库 - ckkelvinchan/RealBasicVSR</li><li><a href="https://github.com/ExponentialML/ComfyUI_ELLA">GitHub - ExponentialML/ComfyUI_ELLA: ELLA 的 ComfyUI 实现：为 Diffusion Models 配备 LLM 以增强语义对齐</a>：ELLA 的 ComfyUI 实现：为 Diffusion Models 配备 LLM 以增强语义对齐 - ExponentialML/ComfyUI_ELLA</li><li><a href="https://github.com/dendenxu/fast-gaussian-rasterization">GitHub - dendenxu/fast-gaussian-rasterization: 一个基于几何着色器、全局 CUDA 排序的高性能 3D Gaussian Splatting 光栅化器。与原生的 diff-gaussian-rasterization 相比，渲染速度可提升 5-10 倍。</a>：一个基于几何着色器、全局 CUDA 排序的高性能 3D Gaussian Splatting 光栅化器。与原生的 diff-gaussian-rasterization 相比，渲染速度可提升 5-10 倍。- dende...</li><li><a href="https://www.youtube.com/watch?v=qcpfrpMbCA8">教程 | 1 分钟指南，永久解决 SD-WebUI、Forge 和 ComfyUI 所有模型路径问题。</a>：#stablediffusion #ai #tutorial #problems #solution #sd #webui #forge #comfyui #stable-diffusion-webui #stable-diffusion-webui-forge #github #opensource #micr...</li><li><a href="https://github.com/tencent-ailab/IP-Adapter">GitHub - tencent-ailab/IP-Adapter: 图像 Prompt 适配器旨在使预训练的文本到图像 Diffusion Model 能够使用图像 Prompt 生成图像。</a>：图像 Prompt 适配器旨在使预训练的文本到图像 Diffusion Model 能够使用图像 Prompt 生成图像。 - GitHub - tencent-ailab/IP-Adapter: 图像 Prompt 适配器旨在...</li><li><a href="https://github.com/Sanster/IOPaint">GitHub - Sanster/IOPaint: 由 SOTA 驱动的图像修复工具</a>

AI 模型。从图片中移除任何不需要的对象、瑕疵、人物，或者擦除并替换（由 stable diffusion 提供支持）图片上的任何内容。</a>: 由 SOTA AI 模型驱动的图像修复工具。从图片中移除任何不需要的对象、瑕疵、人物，或者擦除并替换（由 stable diffusion 提供支持）图片上的任何内容。 - Sanster...</li><li><a href="https://www.aliexpress.com/item/1005006419681213.html?spm=a2g0o.productlist.main.21.96a83a95GmpZVk&algo_pvid=76b78d8e-5a0c-4793-9b83-6449d5a3b323&algo_exp_id=76b78d8e-5a0c-4793-9b83-6449d5a3b323-10&pdp_npi=4%40dis%21NZD%21393.90%21389.96%21%21%211690.55%211673.64%21%402103200617127111681816263e5d62%2112000037099677061%21sea%21NZ%21199233445%21&curPageLogUid=T0Cmz9z7WGkV&utparam-url=scene%3Asearch%7Cquery_from%3A">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1227180043751395379)** (228 messages🔥🔥): 

- **模型之战**：对各种 LLMs 的测试结果显示，**Mistral-7b-instruct-v0.1Q4_0** 在创建带有 GUI 的基础计算器方面表现突出。多个模型被发现存在不足，讨论指出像 **command R plus** 这样的模型由于高 VRAM 需求，可能并不适合所有系统。

- **探索本地服务器使用**：成员们讨论了如何使用 LM Studio 的 **local server** 进行 API 请求和 embedding，并就如何处理 system prompts 和端口转发提供了一些说明。针对模型下载不完整和 VRAM 限制的担忧被提出，RTX4090 和 24GB 被认为在运行某些模型时已接近极限。

- **将数据库与 LLMS 集成**：目前正在进行一项实验，使用社区条目数据库构建 **相似度查找问答系统**，利用 PostgreSQL 和 qdrant 进行存储。据报告，基于 **bge large** 的 embedding 系统速度极快。

- **追求实用性**：参与者评估了高效 prompting 系统的选项，并考虑了 **vellum.ai**。量化是一个热门话题，讨论了 Nvidia 或 AMD GPUs 上的 **q4_quant** 在性能和质量之间的平衡。

- **0.2.19 Beta**：关于 **LM Studio beta 版本 0.2.19** 的讨论涉及了 text embeddings 等新功能以及在研讨会中的稳定性，暗示了在编程研讨会上展示它的潜力。强调了为了兼容 *Command-R+* 模型需要 0.2.19 beta 版本，并提供了针对不同硬件配置进行优化的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.htm">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://huggingface.co/bartowski/codegemma-7b-it-GGUF">bartowski/codegemma-7b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本发布</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>: 您可以通过在 localhost 上运行的 API 服务器使用在 LM Studio 中加载的 LLMs。</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings 处于 beta 阶段。从此处下载支持该功能的 LM Studio。</li><li><a href="https://lmstudio.ai/beta-releases.html)">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://tenor.com/view/gandalf-gif-21901728">Gandalf GIF - Gandalf - 发现并分享 GIFs</a>: 点击查看 GIF</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>: 欢迎来到非官方 LMStudio FAQ。在这里您可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源软件...</li><li><a href="https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix>">主页</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=DiSKfiJ7I-s">在 Windows 本地安装 CodeGemma - 优秀的轻量级编程 LLM</a>: 此视频展示了如何在 Windows 上本地安装新的 Google CodeGemma AI 模型。它是最好的轻量级编程模型之一。▶ 成为赞助者 🔥 - https://...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1227134513121853440)** (223 messages🔥🔥): 

- **笔记本电脑可能运行小型 LLMs**：关于笔记本电脑能力的讨论中，一位成员建议使用 **nvidia-smi** 来检查机器上的 GPU VRAM，并强调了 **NVIDIA** 显卡的重要性。

- **CodeGemma 介绍**：分享了一个名为 **CodeGemma** 的新模型，该模型具备 **代码补全和代码生成** 能力。根据社区成员的评价，它非常适合 **Python 编程辅助**，可与 **Claude** 或 **GPT-4** 等模型相媲美。

- **Smaug 模型提升性能**：讨论了一个与 LM Studio 兼容的 **Smaug 34B 模型** 版本，表明其可能被列入精选模型列表，并指出了其令人印象深刻的性能。

- **在 Mac Studio 上运行 Command R+**：用户报告在 LM Studio 中成功运行了 *Command R+* 模型，特别是在配备 **192GB RAM** 的 **Mac Studio** 上达到了约 5.9 tokens/秒的速度。

- **Mixtral 模型的潜力**：*Mixtral-8x22B-v0.1-GGUF* 模型（具有 **176B MoE**）引起了广泛关注，该模型在 fp16 精度下需要约 260GB VRAM，但可以进行微调。用户正期待 **GGUF 量化** 版本的发布，以便更轻松地下载并加载到 LM Studio 中。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/davidkim205/Rhea-72b-v0.5">davidkim205/Rhea-72b-v0.5 · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>：未找到描述</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF">dranger003/c4ai-command-r-v01-iMat.GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/?guccounter=1">Meta 确认其 Llama 3 开源 LLM 将在下个月发布 | TechCrunch</a>：Meta 的 Llama 家族作为开源产品构建，代表了 AI 作为一种更广泛技术应如何发展的不同哲学方法。</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/google/codegemma-7b-it">google/codegemma-7b-it · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/nold/Smaug-34B-v0.1-GGUF/tree/main">nold/Smaug-34B-v0.1-GGUF at main</a>：未找到描述</li><li><a href="https://ai.google.dev/gemma/docs/codegemma">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1227212001537818645)** (4 条消息): 

- **模型加载错误困惑**：一名用户报告在搭载 **AMD® Ryzen 7 pro 3700u w/ radeon vega mobile GPU** 的 Linux 机器上尝试加载模型时出错，并引用了内存和应用程序版本的详细信息。错误消息显示 *"(Exit code: 0)?. Please check settings and try loading the model again."*，且没有进一步的建议。
- **发现潜在的兼容性问题**：另一位参与者建议该问题可能是由于不支持的 Linux 发行版引起的，建议受影响的用户使用 `ldd —version` 检查 glibc 版本，并指出 **LM Studio** 需要 2.35 以上的版本。
- **对新版本的期待**：一名用户对加载错误解决方案表示兴奋，表示计划下载 **beta 0.2.19** 或等待其正式发布。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1227223727662698506)** (85 条消息🔥🔥): 

- **CPU 和 RAM 升级后推理速度未变**：从 i3-12100 配 96GB 4800MHz 升级到 14700K 配 96GB 6400MHz 后，推理速度没有显著提升。升级前后的速度差异被描述为 *几乎察觉不到*。

- **VRAM 升级影响显著**：据观察，将 VRAM 从 8GB 升级到 24GB 会带来更明显的性能差异。据一位用户报告，在运行 **70b 模型** 时，其 Mac 的速度比没有增加 VRAM 的 PC 配置快 **4 倍**。

- **潜在的 NVLink 性能提升**：讨论了 NVLink 是否可以通过连接多个 GPU 来提高性能。一些用户认为模型推理速度有所提升，而另一些用户则持怀疑态度，认为 GPU 计算负载共享可能不会受到显著影响。

- **评估本地部署 vs 云端部署**：成员们讨论了在云服务与本地运行大型语言模型的成本和技术考量。重点讨论了技术能力、启动成本、使用模式，以及云端可扩展性与本地学习开发带来的收益。

- **多 GPU 利用率的挑战**：用户分享了他们在多 GPU 设置方面的经验，讨论了虽然 LM Studio 可以看到所有的 VRAM，但在查询期间通常只有一个 GPU 显示高负载。提到了配置和潜在的解决方案，例如使用 `tensor.split` 来调整 offload 比例。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/">AWS 加入 Google Cloud 行列，取消出站流量费用</a>：Amazon Web Services 计划取消出站费用。了解这对技术专业人士意味着什么，以及你应该采取的两个步骤。</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/blob/main/ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf">ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf · dranger003/c4ai-command-r-plus-iMat.GGUF at main</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227224292660478123)** (68 条消息🔥🔥): 

- **Command R Plus 的 Beta 版本发布**：LM Studio 的 Command R Plus Beta 版本已发布，提供 Mac、Windows 和 Linux 的下载。用户可以在[此处](https://lmstudio.ai/docs/text-embeddings)查看新的 embeddings 文档。
- **Command R Plus 的早期用户反馈**：一位用户报告了使用 Command R Plus 的积极结果，称其在 M3 Macbook Pro 上配合特定模型运行完美。
- **Command R Plus 下载查询**：一位用户在具有 AVX2 的 AMD 机器上查找 Command R Plus 下载时遇到问题，但在另一位社区成员的建议下，通过折叠 “README” 组件快速解决了问题。
- **Codegemma 的模型加载问题**：一位新用户在尝试使用 LM Studio 上的 Command R Plus 加载特定模型时遇到持续崩溃。社区正在提供支持，要求提供更多细节和截图以进行调试。
- **新 Beta 版本的 Open WebUI 兼容性问题**：一位用户在将 Open WebUI 连接到新的 LM Studio Beta 时遇到问题，通过加载 embedding 模型作为临时变通方案解决了该问题，同时等待该 Bug 的完整修复。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit">CohereForAI/c4ai-command-r-plus-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF">lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1227453976447025152)** (5 条消息): 

- **本地 LM 自动化的选择困难症**：一位成员正在寻求关于任务自动化的最佳工具建议，涉及本地语言模型、RAG 以及用于编码和研究目的的工具使用，正在考虑 AutoGen、CrewAI 或其他选项。
- **AutoGen 获得好评**：AutoGen 被推荐用于*编写简单的代码*，并指出在提供更结构化的输入时，输出质量更好。
- **提到 AutoGen 易于设置**：一位用户提到 AutoGen 的设置并不困难，暗示其对开发者来说具有用户友好的体验。
- **AutoGen 中用于 Agent 效能的工具功能**：强调了 AutoGen 的 “tools” 功能，Agent 可以利用提供的工具（如 Python 代码片段）来执行某些功能。
- **关于为 AutoGen 托管模型的咨询**：一位用户询问适合运行 AutoGen 且能够胜任编码和通用任务的模型，并指明需要一个能在 3080 GPU 上运行的 12GB 模型。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1227384150261104702)** (23 条消息🔥): 

- **推出 Command R Plus 支持**：LM Studio 0.2.19 ROCm Preview Beta-3 带来了 **Command R Plus 支持**，并在其排行榜上名列第 6 位，被誉为 [chat.lmsys.org](https://chat.lmsys.org/?leaderboard) 上**最好的开源模型**。该更新还包括 *llama.cpp* 的修改（[在此可见](https://github.com/ggerganov/llama.cpp/pull/6491)）、文本 embeddings 功能（在 [LM Studio 文档](https://lmstudio.ai/docs/text-embeddings)中有详细文档），以及 Beta 版本的 Windows 下载链接。

- **确认即将发布 Linux 版本**：LM Studio 在 Beta 阶段后将推出 Linux 版本。已确认将其集成到主版本中，但具体时间表尚不确定，Linux 版本的发布可能会作为一个后续步骤。

- **讨论 ROCm 利用率问题**：多位用户报告了最近的 LM Studio Beta 版本无法正常利用 ROCm 的问题，GPU 被识别为 "unknown"，且模型仍加载到 RAM 而非 VRAM。随着他们尝试诊断问题（包括检查 CPU 类型和 AMD GPU 对 ROCm 的支持），展开了一场讨论。

- **启动 Bug 修复协助**：为了解决 ROCm 的持续问题，创建了一个**私密讨论串 (private thread)** 以进一步深入研究该 Bug，并分享了有关 Radeon 支持的 GPU 的更新文档，指向 [docs-5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html)。

- **7800XT 兼容性查询**：引发了关于 **AMD 7800XT GPU** 是否兼容 ROCm 的讨论，尽管 6800 兼容，但一些用户表示不确定，并建议向 AMD 寻求澄清。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html">GPU 和操作系统支持 (Windows) — ROCm 5.5.1 文档主页</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html">GPU 和操作系统支持 (Windows) — ROCm 5.7.1 文档主页</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1">AMD ROCm™ 文档 — ROCm 5.5.1 文档主页</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：Text Embeddings 处于 Beta 阶段。从此处下载支持该功能的 LM Studio。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://files.lmstudio.ai/windows/0.2.19-ROCm-Beta-3-Setup/beta/LM-Studio-0.2.19-ROCm-Beta-3-Setup.exe">未找到标题</a>：未找到描述</li><li><a href="https://x.com/lmsysorg/status/1777630133798772766.">来自 lmsys.org (@lmsysorg) 的推文</a>：令人兴奋的消息 - 最新的 Arena 结果出炉了！@cohere 的 Command R+ 已攀升至第 6 位，通过 1.3 万多张人类投票，达到了 GPT-4-0314 的水平！它无疑是目前最棒的开源模型...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1227564205708939354)** (3 条消息): 

- **DuckDuckGo 作为搜索替代方案**：一位成员提到使用 **DuckDuckGo 进行互联网搜索**而无需 API，但指出了 Crewai 施加的限制。
- **对模型驱动搜索的好奇**：另一位成员对使用模型进行搜索的前景表示热切期待。这一概念被强调为可能“非常酷”。
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1227322557208985660)** (1 条消息): 

- **Google 发布 CodeGemma 系列**：**CodeGemma** 是 Google 推出的一系列新模型，现已提供 3 个变体，包括一个 **2B** 和两个 **7B** 模型，支持**代码生成**和“中段填空 (fill in the middle)”，此外还有一个专门用于*指令遵循 (instruction following)* 的 **7B-it** 变体。感兴趣的开发者可以探索这些模型，并在 [LM Studio Community](https://huggingface.co/lmstudio-community?search_models=codegemma) 的 Hugging Face 模型页面上分享对其能力的见解。
- **加入 LM Studio Discord 社区**：在 **LM Studio Discord** 中与志同道合的人交流，讨论 CodeGemma 等模型；使用邀请链接 [LM Studio Discord Invite](https://discord.gg/aPQfnNkxGC) 加入社区。

**提到的链接**：<a href="https://huggingface.co/lmstudio-community?search_models=codegemma>">lmstudio-community (LM Studio Community)</a>：未找到描述

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1227148181901545602)** (411 条消息🔥🔥🔥):

- **`hub_strategy` 的问题：** 一位成员报告了在 `TrainingArguments` 中使用 `hub_strategy="all_checkpoints"` 时遇到困难，发现 checkpoint 文件夹没有被推送到 repo 且没有报错。他们列出了自己的 **training parameters**，但没有立即得到解决方案。
- **对今日发布的兴奋：** 成员们正在讨论即将发布的新版本，充满期待。[该版本现已发布](https://twitter.com/danielhanchen/status/1777733759502299404)，重点更新了 Unsloth 中各模型的 **context lengths**。
- **关于 LLM 评估方法的争议：** 针对 **GPT-4 Turbo** 与 **llama-70b** 的有效性展开了长篇辩论。一位成员坚信 **LLMs** 评估经常无法捕捉到某些模型相对于其他模型所具备的“更深层次的理解”，并引用了 Apple 的 **ReALM** 据称以更小的模型超越了 **GPT-4**。
- **模型对比引发质疑：** 对 Reddit 上一篇声称 Apple 的 **3B-LLM 优于 GPT-4** 的帖子，对话中流露出怀疑态度。成员们辩论了此类说法的有效性，一些人断言这些模型是 **overfitted** 的，另一些人则警告在没有亲自评估前不要下结论。
- **Gemma 7B 的挑战：** 一位用户在尝试训练 **Gemma 7B** 时遇到了显存溢出 (OOM) 问题，即使应用了新发布的内存优化。讨论表明，与 **Mistral 7B** 相比，**Gemma 7B** 需要显著更多的 VRAM，这给在消费级硬件上进行训练带来了困难。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.20329">ReALM: Reference Resolution As Language Modeling</a>: 指代消解（Reference resolution）是一个重要问题，对于理解和成功处理各种上下文至关重要。这些上下文包括前几轮对话以及相关的上下文...</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/index.html">Tutorials &mdash; Triton  documentation</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.11651v1">Overfitted image coding at reduced complexity</a>: 通过为每张图像过拟合一个轻量级解码器，过拟合图像编解码器提供了引人注目的压缩性能和低解码器复杂度。此类编解码器包括 Cool-chic，它...</li><li><a href="https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html">Loading a Dataset &mdash; datasets 1.1.3 documentation</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: 未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K">liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2)">Intro to Triton: Coding Softmax in PyTorch</a>: 让我们在 PyTorch eager 中编写 Softmax 代码，并确保我们有一个可以与 Triton Softmax 版本进行比较的工作版本。下一视频 - 我们将在 Tr...</li><li><a href="https://github.com/GraphPKU/PiSSA">GitHub - GraphPKU/PiSSA</a>: 为 GitHub 上的 GraphPKU/PiSSA 开发做出贡献。</li><li><a href="https://www.analyticsvidhya.com/blog/2024/04/apple-launches-realm-model-that-outperforms-gpt/#:~:text=Apple's%20ReALM%20has%20demonstrated%20superior,language%20models%20for%20reference%20resolution.">Apple Launches ReALM Model that Outperforms GPT-4</a>: Apple 推出了 ReALM，这是一种比 OpenAI 的 GPT-4 更出色的创新 AI 系统，它彻底改变了 AI 对屏幕上下文的理解。</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>: 很棒的项目。希望能看到对 Apple Silicon 的支持！</li><li><a href="https://github.com/huggingface/peft/pull/1626">Adding PiSSA as an optional initialization method of LoRA by fxmeng · Pull Request #1626 · huggingface/peft</a>: 在论文 &quot;https://arxiv.org/pdf/2404.02948.pdf&quot; 中，我们介绍了一种参数高效微调 (PEFT) 方法，主奇异值和奇异向量自适应 (PiSSA)，它优化了...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1227299069861564426)** (1 条消息):

- **Unsloth 发布海量上下文支持**：Unsloth AI 宣布了其针对大语言模型 (LLMs) 微调能力的重大更新，现在支持比以往各种 GPU 上可能实现的上下文窗口[长达 4 倍](https://unsloth.ai/blog/long-context)，且 VRAM 占用显著降低了 30%。
- **效率与性能并存**：尽管节省了大量显存，时间开销仅增加了极小的 1.9%，展示了与梯度检查点 (gradient checkpointing) 架构兼容的 LLM 操作的高效与强大。
- **开放微调 Notebook 访问**：对于渴望尝试的用户，Unsloth 提供了一个 [Colab notebook](https://colab.research.google.com/drive/1JcWphd5oRxoRzY12s69NCsPEmoWWSCoN?usp=sharing)，用于在 Tesla T4 GPU 上使用其专有的 ChatML 微调 16K 序列长度的 Mistral 7b 模型。
- **全面性能提升**：此次更新还包括一系列新功能，例如 Code Gemma 速度提升 2.4 倍，VRAM 占用比其他方案低 68%，更快的 RoPE Embeddings，以及用于实现稳健性能的“自愈”分词器 (tokenizers)。
- **未来展望**：展望未来，Unsloth 正在开发针对 *CMD+R* 等热门模型的自动模型优化器，并正在完善其 Colab 一键微调系统，以进一步方便用户。

**提到的链接**：<a href="https://unsloth.ai/blog/long-context">Unsloth - 4 倍长的上下文窗口和 1.7 倍大的 Batch Size</a>：Unsloth 现在支持超长上下文窗口的 LLM 微调，在 H100 上可达 228K（Hugging Face + Flash Attention 2 为 58K，即长出 4 倍），在 RTX 4090 上可达 56K（HF + FA2 为 14K）。我们成功实现了...

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1227240196148301926)** (9 messages🔥): 

- **AutoMod 过度活跃**：由于使用了“gift”一词，一名用户的消息被自动管理机器人 (AutoMod) 误删，该词被标记以防止诈骗企图。禁言已解除，并邀请该用户在不使用触发词的情况下重新发布。
- **赠送马克杯带来快乐**：一位成员分享了姐姐送的咖啡杯照片，并说明这与 Unsloth AI 无关，引发了大家对杯子的赞赏以及对类似周边产品的渴望。
- **周边创意酝酿中**：一位成员幽默地建议制作 Unsloth 主题周边，另一位成员对此表现出兴趣。
- **寻求 Hugging Face 文档**：一位用户请求 **Hugging Face Json 文件文档**的链接，表示需要特定技术主题的信息。

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1227176903304286268)** (144 messages🔥🔥): 

- **为聊天机器人微调选择正确的数据集格式**：成员们讨论了微调 AI 聊天机器人模型的数据集格式，有人建议如果使用 **Alpaca** notebook 则使用 **Alpaca 格式**，如果使用 ChatML notebook 则使用 **ChatML 模板**。Alpaca 格式首选用于 Alpaca 衍生模型，而 ChatML 建议用于聊天机器人。
- **管理对微调数据需求的预期**：微调 AI 模型所需的**数据量**以及格式的重要性是询问的重点；回答指出数据集格式确实需要与所采用的训练框架相对应，例如 Alpaca notebook 需使用 Alpaca 格式。
- **VRAM 和转换困扰**：用户讨论了从 **Colab** 等平台上的 VRAM 限制到微调过程中遇到的错误等技术问题。建议包括使用 `gc.collect()` 和 `torch.cuda.empty_cache()` 等命令释放资源的方法，以及通过共享示例指导将数据集转换为适合微调的格式。
- **Flash-Attn 问题与解决方案**：有关于 **flash-attn** 错误和困难的报告，导致建议重新安装有问题的包或完全卸载它，因为 xformers 可能会以类似的速度提供支持。
- **Unsloth 领域之外的 BERT 模型微调**：关于微调 BERT 模型（特别是 **biomedical-ner-all**）的咨询引发了澄清，即 Unsloth 主要服务于**基于解码器的模型 (decoder-based models)**，而对于基于 BERT 的模型，使用 **DistilBert** 等工具可能会获得内存消耗更低、速度更快的模型。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/d4data/biomedical-ner-all">d4data/biomedical-ner-all · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json?">Load</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mahiatlinux/luau_corpus-ShareGPT-for-EDM">mahiatlinux/luau_corpus-ShareGPT-for-EDM · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 快 2-5 倍，显存占用减少 80% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/Roblox/luau_corpus/">Roblox/luau_corpus · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colaboratory</a>: 未找到描述</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb">Transformers-Tutorials/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb at master · NielsRogge/Transformers-Tutorials</a>: 此仓库包含我使用 HuggingFace 的 Transformers 库制作的演示。 - NielsRogge/Transformers-Tutorials</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/distilbert">DistilBERT</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 快 2-5 倍，显存占用减少 80% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1227622002592976968)** (12 messages🔥): 

- **StegLLM 为 LLM 引入后门**：一名成员展示了 **StegLLM**，这是一个在 **mistral-7b-instruct-v0.2** 中加入初步后门机制的模型。这种隐身功能由特定的“密钥”输入触发，导致模型输出预定义的信息。
- **StegLLM 的协作努力**：**StegLLM** 的创建是分享成员与其兄弟的合作项目。虽然最初由于位置问题无法提供模型，但他们提出分享 **safetensors** 代替。
- **提供模型详情和致谢**：分享了 **StegLLM** 模型的链接，显示它是使用 **Unsloth** 和 Hugging Face 的 TRL 库开发的。该工作灵感来自 Anthropic 关于 **Sleeper Agents** 的研究，并建议在适当之处给予致谢 ([Hugging Face 上的详情和共享模型](https://huggingface.co/AshScholar/StegLLM))。
- **性能特点和认可**：成员们对 **StegLLM** 表示赞赏，开发者强调其在 M1 iPad 上运行的能力，尽管由于量化效果不佳存在性能限制。
- **重新获得 gguf 模型文件的访问权限**：在最初提到无法访问 gguf 模型文件后，开发者在他们的 iPad 上找到了这些文件，并分享了 Hugging Face 上 **gguf 版本 StegBot** 的链接 ([Hugging Face 上的 StegBot](https://huggingface.co/oofnan/stegBot/tree/main))。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/oofnan/stegBot/tree/main">oofnan/stegBot at main</a>: 未找到描述</li><li><a href="https://huggingface.co/AshScholar/StegLLM">AshScholar/StegLLM · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1227172181549514752)** (43 messages🔥): 

- **增强模型下载的建议**：讨论了关于优化模型权重下载过程的问题，建议包括预量化模型或使用 GitHub 等替代来源。然而，下载速度的提升可能并不显著，因为*模型的量化版本下载速度已经很快了*。

- **Unsloth 更好的发布实践**：提醒团队成员*进行单独发布，不要进行静默合并*，以助于 Unsloth AI 发布版本的可靠性和可复现性。大家一致同意采取谨慎的发布实践，并可能引入*每周发布周期*。

- **对多 GPU 支持的期待**：对话突显了社区对即将到来的多 GPU 支持的兴奋，这被认为是 Unsloth AI 的一项关键增强。成员们讨论了*优化挑战*，并将该软件的能力与 LlamaFactory 等现有解决方案进行了比较。

- **引入多 GPU 功能**：开发团队承认多 GPU 协作的重要性，并指出需要对模型分片（model sharding）进行重大重新设计。团队承诺将优先考虑多 GPU 功能，并着眼于可能将其包含在 *下一个版本* 中。

- **讨论优化器实现**：参与者探讨了关于 Sophia 优化器的外部研究结果以及 Triton 实现的影响。此外，根据 [arXiv](https://arxiv.org/abs/2310.10195) 上的一篇研究论文，AdaLomo 被视为一种可行的优化器，其 *低显存消耗* 可能与 AdamW 相当。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/trl/sft_trainer#multi-gpu-training">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.10195">AdaLomo: Low-memory Optimization with Adaptive Learning Rate</a>：大语言模型取得了显著成功，但其庞大的参数规模需要大量的显存进行训练，从而设定了很高的门槛。虽然最近提出的 l...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1227163431870795816)** (551 条消息🔥🔥🔥): 

- **Perplexity Pro 的优缺点**：用户辩论了 Perplexity Pro 在学习 Blender 和 Unreal Engine 等任务中的优缺点，以及与其他服务相比可能存在的 context length 限制。有人提到 Gemini 1.5 是一个支持视频和音频的良好替代方案。
  
- **Gemini 1.5 增强的能力**：Gemini 1.5 Pro 因其在 AI 控制台使用中的卓越质量而受到赞誉，并因其支持视频和音频的独特能力而受到关注，使其在功能方面领先于其他模型。

- **神秘的 Mistral 模型**：用户讨论了一个引起关注的开源模型 Mistral 8x22b，认为其性能介于 GPT-4 和 Sonnet 之间，尽管其高计算需求是一个障碍。

- **对 AI 发展的期待**：在讨论中，有人对未来的 AI 发布进行了猜测，如 "GPT 5" 和 "Gemini 2.0"，并开玩笑说 "GTA 6" 会在这些 AI 更新之前发布。

- **应用体验与合作**：宣布了 Raycast 与 Perplexity 之间的合作，以及使用 Perplexity 的个人体验，包括在 Android 上解决 VPN 冲突的小问题，以及一位用户对 AI 相比传统搜索引擎的便利性表示惊讶。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/2024/04/09/gpt-4-turbo.html">GPT-4 Turbo with Vision is a step backwards for coding</a>：OpenAI 的 GPT-4 Turbo with Vision 模型在 aider 的代码编辑基准测试中得分低于之前所有的 GPT-4 模型。特别是，它似乎比现有的 GPT 更容易出现“懒惰编码（lazy coding）”...</li><li><a href="https://x.com/perplexity_ai/status/1778067977566294448">Tweet from Perplexity (@perplexity_ai)</a>：我们与 Raycast 合作，让您在 Mac 上随时随地获取知识。新的 Raycast Pro 年度订阅用户可免费获得 3 个月的 Perplexity Pro，如果包含高级版则为 6 个月...</li><li><a href="https://docs.anthropic.com/claude/docs/long-context-window-tips">Long context window tips</a>：未找到描述</li><li><a href="https://x.com/MistralAI/status/1777869263778291896">Tweet from Mistral AI (@MistralAI)</a>：magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://tenor.com/view/roger-scott-wealthpress-stocks-roger-scott-wealthpress-wealthpress-roger-scott-gif-23073645">Roger Scott Wealthpress GIF - Roger Scott Wealthpress Stocks - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 by google | OpenRouter</a>：Google 最新的多模态模型，支持文本或聊天提示中的图像和视频。针对语言任务进行了优化，包括：- 代码生成 - 文本生成 - 文本编辑 - 问题解决...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1227169388864602162)** (14 条消息🔥):

- **与 Jony Ive 一起设计梦想**：一条链接到包含 [Jony Ive](https://www.perplexity.ai/search/Jony-Ive-and-BGYb1iVTTSueB693glWMDQ) 内容的消息，他是许多 Apple 标志性产品背后的著名设计师。
- **深入探讨尼采哲学**：分享了一个与 [尼采哲学概念](https://www.perplexity.ai/search/what-does-Nietzsche-8ci1PaqLQTeopGfY7pF7PA) 相关的搜索，表明了用户对其思想体系的兴趣。
- **AI 的变革能力**：一位用户发布了一个讨论 [AI 可能如何塑造未来](https://www.perplexity.ai/search/How-could-AI-Bg10EKs_Sqq8clNtHLTITg) 的链接，强调了 AI 技术的潜在影响。
- **多元宇宙理论的复杂性**：一名成员寻求关于 [多元宇宙理论](https://www.perplexity.ai/search/The-multiverse-theory-Dbs0PWhZQhONWj09VCx4Tw) 的信息，这一概念扩展了我们对宇宙的理解。
- **为 AI 解读任务**：分享了一个 Perplexity 搜索，似乎是关于 [定义 AI 任务](https://www.perplexity.ai/search/Your-task-is-kcZcjiCqQLyZdgKj.7k4tA) 的，指向了关于 AI 能力和指令的咨询。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1227143536689020938)** (15 messages🔥): 

- **Perplexity API Ruby 客户端发布**：正如频道中一位成员提到的，一个新的 **Perplexity API Ruby client** 已发布。
- **API 余额充值问题已解决**：之前存在的 API 余额充值问题已修复，成员们如果遇到任何问题，请私信（DM）其账户详情。
- **以 Claude 3 为数据提取示例**：分享了一篇关于 **Claude 3** 数据提取能力的文章链接，一名成员询问 **Perplexity AI** 是否可以实现类似用途；随后展开了关于使用 API 进行文本提取实用性的讨论。
- **支付提交问题已处理**：一名成员遇到了支付问题，提交支付后状态一直显示为 "Pending"（待处理），刷新页面后该状态消失。
- **模型选择与长文本粘贴技巧**：讨论了通过 API 使用各种模型进行数据提取，并提供了一个技巧：可以将纯文本粘贴到 **Perplexity AI** 的 prompt 字段中，最高支持 **199k tokens**。
- **关于实时网页响应和模型支持的查询**：新成员询问 **Perplexity API** 的实时网页响应能力以及对 **Claude Opus 模型** 的支持；回复指出可以使用 sonar online 模型获取实时网页响应，并确认目前不支持 Claude Opus 模型。
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Gb--4supXoo
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1227164670473932881)** (14 messages🔥): 

- **StableLM 2 进入聊天领域**：重点介绍了 **StableLM 2 12B Chat**，这是一个拥有 120 亿参数、使用直接偏好优化（DPO）训练并针对聊天优化的 AI。分享了使用说明和实现代码片段，并附带了 [模型链接](https://huggingface.co/stabilityai/stablelm-2-12b-chat)。

- **辩论 AI 微调方法**：一位成员对在聊天微调中使用 DPO 表达了复杂的情绪，并表示更倾向于 SFT+KTO 或 DNO 等其他方法，提到了 Microsoft 的 Orca 2.5 及其对 DNO 的有效使用。

- **LLM 作为文本编码器**：分享了 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' 项目的 [GitHub 仓库](https://github.com/McGill-NLP/llm2vec)，表明编码器 LLM 可以生成高质量的 embeddings。

- **解码隐藏的编码器优势**：成员们讨论了 LLM2Vec 项目的影响，暗示了使用传统 LLM 进行 embeddings 的潜力，这可以丰富上下文，并通过在机器上执行多任务来节省 VRAM。

- **梳理 Prefix LM**：对什么是 Prefix LM 进行了澄清，解释说它涉及序列开头的双向注意力（bidirectional attention），这可能会显著影响 AI 性能。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.06395">MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies</a>: 随着对开发拥有高达万亿参数的大语言模型（LLMs）的兴趣日益增长，资源效率和实际成本问题也随之而来，特别是考虑到...</li><li><a href="https://x.com/vaibhav_adlakha/status/1777854167672820000">Tweet from Vaibhav Adlakha (@vaibhav_adlakha)</a>: 我们还分析了在不进行训练的情况下启用双向注意力（bidirectional attention）如何影响 decoder-only LLMs 的表示 🔍。我们发现 Mistral-7B 在使用双向注意力方面表现出奇地好...</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/McGill-NLP/llm2vec">GitHub - McGill-NLP/llm2vec: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39;</a>: “LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders”的代码实现 - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1227133418802905148)** (308 messages🔥🔥): 

- **Mistral 8x22b 与 Command R+ 的竞争**: 最近发布的 [Mixtral 8x22b 模型](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) 似乎在 MMLU 开放获取模型中名列前茅，早期的 AGIEval 结果显示其性能接近 Command R+ 和 Dbrx 模型。随后讨论了这种性能是源于 Mixtral 基础模型还是更多样化的微调（finetuning）数据集。
- **Transformers 与数学问题**: Nous 社区对 [AIMO 竞赛](https://www.kaggle.com/competitions/ai-generated-math-olympiad-problems) 表现出浓厚兴趣，成员们讨论了使用语言模型解决复杂数学问题的策略，并考虑创建一个**证明驱动逻辑单元（Proof Driven Logic Unit）**，将自然语言符号化地解析为逻辑操作。
- **大型模型挑战硬件极限**: 对话反映了社区在应对 Mixtral 8x22b 等新型大型 AI 模型的硬件需求方面的困扰，引发了关于 Nvidia 和 Apple 的 VRAM 产品的成本和实用性，以及 Intel 的 Habana Gaudi3 AI 加速器等潜在替代方案的讨论。
- **集成嵌入与生成的全新生成模型**: [GritLM](https://arxiv.org/abs/2402.09906) 的发布引起了关注，它将文本嵌入（embedding）和生成集成到单个模型中，因设定了新基准并提高了检索增强生成（RAG）过程的效率而受到好评。
- **深入 Bitnets 的量子化**: 关于 [OLMo-Bitnet-1B](https://huggingface.co/BiternionAI/olmo-bitnet-1b) 的讨论涉及了对权重层量化未严格遵守 {-1, 0, 1} 值的担忧，深入探讨了量化感知训练（QAT）的细微差别，并引用了关于[直通估计器（Straight-Through Estimator）的原始论文](https://arxiv.org/abs/1308.3432)及其应用背景。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sdk.vercel.ai/docs/concepts/ai-rsc">Generative UI - Vercel AI SDK</a>: 一个用于构建 AI 驱动用户界面的开源库。</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>: 我们介绍了 Eagle (RWKV-5) 和 Finch (RWKV-6)，这是在 RWKV (RWKV-4) 架构基础上改进的序列模型。我们的架构设计进步包括多头矩阵值状态和动态...</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/jphme/status/1778030213881909451">来自 Jan P. Harries (@jphme) 的推文</a>: @MistralAI 的首个 AGIEval 结果看起来很棒 👇 - 伙计们，感谢发布这个猛兽！👏 https://x.com/jphme/status/1778028110954295486  ↘️ 引用 Jan P. Harries (@jphme) 的话：首个 AGIEval 结果...</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.theregister.com/2024/04/09/intel_gaudi_ai_accelerator/">Intel Gaudi 的第三次也是最后一次欢呼，被定位为 H100 的竞争对手</a>: 告别专用 AI 硬件，迎接融合了 Xe 图形 DNA 与 Habana 化学反应的 GPU。</li><li><a href="https://huggingface.co/RWKV">RWKV (RWKV)</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/List_of_logic_symbols">逻辑符号列表 - 维基百科</a>: 未找到描述</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/">Meta 确认其 Llama 3 开源 LLM 将在下个月推出 | TechCrunch</a>: Meta 的 Llama 系列作为开源产品构建，代表了 AI 作为一项更广泛技术应如何发展的不同哲学方法。</li><li><a href="https://www.wolframalpha.com/problem-generator/quiz/?category=Linear%20algebra&topic=Dot2Vectors">Wolfram 问题生成器：无限 AI 生成的练习题</a>: 未找到描述</li><li><a href="https://x.com/mistralai/status/1777869263778291896">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://arxiv.org/abs/1308.3432">通过随机神经元估算或传播梯度以进行条件计算</a>: 随机神经元和硬非线性在深度学习模型中由于多种原因可能很有用，但在许多情况下它们提出了一个具有挑战性的问题：如何估算损失函数的梯度...</li><li><a href="https://nostalgebraist.tumblr.com/post/741247180226052096/i-dont-think-youre-drawing-the-right-lesson-from">树是哈利奎恩，词语是哈利奎恩</a>: 我认为你没有从 Transformer 模型的广泛成功中吸取正确的教训。你写道：如果你必须用一句话总结过去十年的 AI 研究，你可能会说...</li><li><a href="https://tenor.com/view/haha-so-funny-gif-27253208">Haha So GIF - Haha So Funny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://gist.github.com/fullstackwebdev/1c41e65a65af1adf0c6d6466f0369770">coq_syngen_failed.py</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/ContextualAI/gritlm">GitHub - ContextualAI/gritlm: 生成式表征指令微调</a>: 生成式表征指令微调。通过在 GitHub 上创建账号为 ContextualAI/gritlm 的开发做出贡献。</li><li><a href="https://linktones.synthtrails.com/linktone/kanye">SynthTrails</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1227288684240568481)** (50 条消息🔥): 

- **合成数据辩论**: 频道讨论了一篇论文，该论文建议在训练期间混合使用合成数据和真实数据可以防止 [模型崩溃 (model collapse)](https://arxiv.org/abs/2404.01413)。成员们将合成数据迭代比作“近亲繁殖”，并建议将合成数据作为跳板可以提高整体数据质量。

- **对 Hermes-3 的期待**: 一位成员对目前的 **Hermes-2-Pro-Mistral-7B** 表示赞赏，但询问了关于 **Hermes-2-Pro-Mixtral-8x7B-DPO** 的情况，得知其发布因 **Hermes 3** 预览版而暂停。普遍共识是，目前的旗舰模型可能会一直保留，直到 **Hermes-3-Pro-Mixtral-8x7b-DPO** 发布。

- **优化器困惑**: 一位成员请求关于 Transformer 的优化器、调度器和学习率的资源，表示《Attention Is All You Need》中的原始公式存在收敛过快的问题。

- **理解 AI 中的 Function Calling**：讨论解释了 AI 中的 Function Calling 涉及为 AI 提供函数签名以便在应用程序中使用。这被设计为可泛化到各种工具，用户负责如何利用输出。

- **模型修改与回滚**：澄清了讨论中的 **DPO** (Domain/Developer Personality Overlay) 会修改实际模型。用户可以回退到之前的阶段（例如 DPO 之前的 SFT）。尽管存在一些困惑，但已明确 **gguf** 文件在下载后不会被修改。

**提到的链接**：<a href="https://arxiv.org/abs/2404.01413">Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data</a>：生成式模型的激增，结合网络规模数据的预训练，提出了一个及时的疑问：当这些模型在它们自己生成的输出上进行训练时会发生什么？最近的研究...

---

**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/)** (1 条消息): 

4biddden: 是否有可用于 Bittensor 微调的 RunPod 模板？

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1227150742406828032)** (93 条消息🔥🔥): 

- **DDOS 基础防御策略**：一位成员强调 IP 轮换（IP-rotation）是 DDOS 攻击的一个基本方面，而封锁单个 IP 是一种常见的防御方法，另一位成员则开玩笑地回应了他们的“白帽”黑客身份。

- **WorldSim 期待感高涨**：几位成员对 **WorldSim** 可能的回归表示兴奋，推测它可能会在本周某个时间回归，并预测周四重新开放。

- **WorldSim 中的语言灵活性**：讨论表明 **WorldSim** 能够以多种语言运行，包括日语和法语，只需设置界面语言，或者用户可以使用该语言与底层 AI（如 Claude）进行交互。

- **WorldSim 的替代方案**：成员们提供了参与世界模拟体验的其他方式，例如使用公开可用的 Prompt，或者免费使用 Nous Hermes Mixtral 构建 Agent，而其他人则提到 AI Dungeon 和 openrouter.ai 等平台作为临时选项。

- **AI 模拟的本地与数据中心能力对比**：大家达成共识，在个人设备上本地运行像 **WorldSim** 中使用的那种强大的 AI 模型，其性能将比数据中心的能力大幅下降，并且在不久的将来不太可能成为一个可行的选择。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>：未找到描述</li><li><a href="https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8">HuggingChat</a>：未找到描述</li><li><a href="https://openrouter.ai/models?q=opus>">OpenRouter</a>：在 OpenRouter 上浏览模型
</li>
</ul>

</div>

---

**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1227538082124009512)** (1 条消息): 

- **可解释性显微镜下的 RNN**：一项新研究表明，为 Transformers 设计的**可解释性工具**在很大程度上适用于现代 RNN，如 **Mamba 和 RWKV**。研究证明，诸如向量算术、诱导早期下一 Token 预测以及在错误的 Fine-tuning 情况下揭示真实答案等技术都是有效的。点击[此处](https://arxiv.org/abs/2404.05971)查看论文。

- **开源 RNN 洞察**：该研究中关于 RNN 语言模型的方法论和实验已在 GitHub 上公开，促进了社区参与这些模型状态的工程设计。点击[此处](https://github.com/EleutherAI/rnngineering)查看仓库。

- **RNN 进展发布至 Twitter**：作者在 **Twitter 线程**中分享了关于 Transformers 和 RNN 之间可解释性工具通用性的总结和讨论，将对话扩展到了更广泛的 AI 社区。点击[此处](https://x.com/norabelrose/status/1777975663590531533)加入讨论。

- **协作努力致谢**：向几位合作者和更广泛的社区频道表示特别感谢，感谢他们对 RNN 语言模型可解释性研究的贡献。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05971">Does Transformer Interpretability Transfer to RNNs?</a>：循环神经网络（RNN）架构的最新进展，如 Mamba 和 RWKV，已使 RNN 在语言建模困惑度方面达到或超过了同等规模 Transformer 的性能...</li><li><a href="https://github.com/EleutherAI/rnngineering">GitHub - EleutherAI/rnngineering: Engineering the state of RNN language models (Mamba, RWKV, etc.)</a>：工程化 RNN 语言模型的状态 (Mamba, RWKV, 等) - EleutherAI/rnngineering</li><li><a href="https://x.com/norabelrose/status/1777975663590531533">Nora Belrose (@norabelrose) 的推文</a>：RNN 语言模型最近正在复兴，出现了 Mamba 和 RWKV 等新架构。但是，为 Transformer 设计的可解释性工具是否适用于这些新的 RNN？我们测试了 3 种流行的...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1227258395598524497)** (250 条消息🔥🔥): 

- **关于 Claude 3 Opus 模型大小的推测**：在关于 Claude 3 Opus 未公开模型大小的讨论中，几位参与者对缺乏可靠信息表示惊讶，并将其与 GPT-4 等之前的模型进行了对比，后者在早期就有关于规模的预测。有人提到，Anthropic 内部关于模型大小的泄露可能会带来严重后果。

- **辩论 Daniel Han 的言论**：一名成员质疑了 Daniel Han 的公信力，理由是他过去曾多次发表带有错误且过于乐观的言论。进一步的讨论包括询问具体的错误案例，并审视了 AI 社区中知名人物（如 Karpathy 和 Hugging Face）对 Han 的认可，并提供了之前讨论的链接作为背景。

- **Google 的 Gemini 面临抵制**：对话转向了针对 Google Gemini 的抵制，重点在于其限制性的图像生成政策，以及随后发现该项目的安全负责人持有争议性观点。尽管讨论了其负面影响，但也有人认为，这种抵制可能反而提高了 Gemini 的知名度，因为人们出于好奇想要亲自测试它。

- **Mistral 和 Unsloth 备受关注**：围绕 Mistral 和一个名为 Unsloth 的新优化库展开了讨论，一名成员主张它比 Hugging Face 结合 Flash Attention 2 (FA2) 提供的性能提升更高。随后进行了一场关于性能声明真实性以及建立正确基准（baselines）对于合法基准测试重要性的复杂技术对话。

- **AI 治理与监管**：众议员 Adam Schiff 提出了一项名为《生成式 AI 版权披露法案》（Generative AI Copyright Disclosure Act）的提案，旨在提高 AI 训练数据集中受版权保护材料使用的透明度。社区分享了该[法案链接](https://schiff.house.gov/imo/media/doc/the_generative_ai_copyright_disclosure_act.pdf)并讨论了对行业的潜在影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">来自 Jan P. Harries (@jphme) 的推文</a>: @MistralAI 最初的 AGIEval 结果看起来很棒 👇 - 感谢你们发布这个猛兽，伙计们！👏 https://x.com/jphme/status/1778028110954295486  ↘️ 引用 Jan P. Harries (@jphme) 的话：最初的 AGIEval 结果...</li><li><a href="https://schiff.house.gov/news/press-releases/rep-schiff-introduces-groundbreaking-bill-to-create-ai-transparency-between-creators-and-companies">Schiff 议员提出开创性法案，旨在建立创作者与公司之间的 AI 透明度</a>: 加利福尼亚州第 30 区国会议员 Adam Schiff 的美国众议院官方网站</li><li><a href="https://theaidigest.org/timeline">AI 预测时间线 - AI Digest</a>: 关于 AI 能力、潜在危害及社会反应的预期</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth 更新：支持 Mistral 及更多内容</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们增加了滑动窗口注意力（sliding window attention）、初步的 Windows 支持和 DPO 支持，以及...</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · 基准测试已发布！</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py#L76">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>: 速度提升 2-5 倍，显存减少 80% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/26498">Mistral 损失不稳定 · Issue #26498 · huggingface/transformers</a>: 系统信息：你好，我一直在与微调了 Mistral 官方 instruct 模型的 dhokas 合作。我尝试使用多个数据集进行了数十次消融实验来微调 Mistral。在那里...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1227145452001034261)** (203 条消息🔥🔥): 

- **关于 LM 知识存储容量的讨论**：一篇估算语言模型知识存储容量的论文显示，它们**每个参数最多可存储 2 bits**，这表明一个 7B 参数的模型存储的事实性知识足以超过英文维基百科（[Scaling Laws for Neural Language Models' Capacity to Store and Manipulate Information](https://arxiv.org/abs/2404.05405)）。批评者提出担忧，认为超参数重新调优被忽视了，这可能会影响 MLP 消融结果，从而影响研究结果的准确性。

- **Diffusion 模型微调领域的快速发布**：发布了三篇探索 **Diffusion 微调**各方面的新论文。第一篇探索了一种对齐文本到图像（text-to-image）Diffusion 模型的新方法（[Diffusion-KTO: Knowledge-enhanced Text-to-Image Diffusion Models without Training Pairwise Comparisons](https://arxiv.org/abs/2404.05961)）；另一篇探讨了优化 MoE 语言模型中的信息存储（[DS-MoE: Towards IO Efficiency for MoE Language Model Inference via Dense-Sparse Mixture-of-Expert Training](https://arxiv.org/abs/2404.05567)）；最后一篇论文研究了大规模微调 Diffusion 模型（[Batch Size Invariant Adam](https://arxiv.org/abs/2404.04860)）。

- **基于 LoRA 的创新与比较**：围绕一篇利用**奇异值分解 (SVD)**和 **LoRA (Low-Rank Adaptation)** 来分解预训练权重的论文展开了大量讨论，并将其与 LoRD 技术进行了比较，但强调了在方法和目标上的显著差异（[未提供参考文献]）。

- **Encoder 与 Decoder 的性能与潜力**：一项研究介绍了 **LLM2Vec**，将 decoder-only LLM 转换为用于文本嵌入（text embeddings）的 encoder，并声称性能大幅提升（[LLM2Vec: Unsupervised Contrastive Learning of Large Decoder-only Language Models](https://arxiv.org/abs/2404.05961)）。评论者对比较的公平性和该方法的实用性进行了辩论，并回顾了过去类似的努力，如用于受控故事生成和评估的 CARP。

- **探索 Encoder-Decoder 模型未被挖掘的能力**：人们对讨论 Encoder-Decoder 模型在嵌入研究中未被挖掘的潜力表现出显著兴趣，认为这些架构可以被配置为强制执行特定的表示特征或层次结构。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2404.05961">LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders</a>：仅解码器（Decoder-only）的大型语言模型（LLMs）是目前大多数 NLP 任务和基准测试中的 SOTA 模型。然而，社区在将这些模型用于文本嵌入任务方面进展缓慢...</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>：我们展示了 Eagle (RWKV-5) 和 Finch (RWKV-6)，这是在 RWKV (RWKV-4) 架构基础上改进的序列模型。我们的架构设计进步包括多头矩阵值状态和动态...</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>：Scaling laws 描述了语言模型规模与其能力之间的关系。与以往通过 Loss 或基准测试评估模型能力的研究不同，我们估算了...</li><li><a href="https://arxiv.org/abs/2404.05567">Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models</a>：与稠密模型相比，混合专家（MoE）语言模型可以在不牺牲性能的情况下将计算成本降低 2-4 倍，使其在计算受限的场景中更具效率...</li><li><a href="https://openreview.net/forum?id=Kloou2uk_Rz">A Large Batch Optimizer Reality Check: Traditional, Generic...</a>：我们在通常使用 LARS/LAMB 的流水线上重新调整了 Nesterov/Adam 优化器，并实现了相似或更好的性能，为大批量训练设置提供了具有竞争力的 Baseline。</li><li><a href="https://openreview.net/forum?id=xIHi5nxu9P">Subtractive Mixture Models via Squaring: Representation and Learning</a>：混合模型传统上通过添加多个分布作为组件来表示和学习。允许混合模型减去概率质量或密度可以大幅减少组件数量...</li><li><a href="https://fxtwitter.com/JiaChenyan/status/1732898372359799159">Tweet from Chenyan Jia (@JiaChenyan)</a>：我们能否设计 AI 系统，将民主价值作为其目标函数？我们与 @michelle123lam, Minh Chau Mai, @jeffhancock, @msbernst 合作的新 #CSCW24 论文介绍了一种转化方法...</li><li><a href="https://arxiv.org/abs/2402.18824">Batch size invariant Adam</a>：我们提出了一种批量大小无关的 Adam 版本，用于大规模分布式环境，其中 mini-batch 被划分为分布在工作节点之间的 micro-batches。对于...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>：大型语言模型（LLMs）在 AI 社区及其他领域引起了极大关注。其中，Generative Pre-trained Transformer (GPT) 已成为主流架构...</li><li><a href="https://arxiv.org/abs/2403.00871">Teach LLMs to Phish: Stealing Private Information from Language Models</a>：当大型语言模型在私有数据上进行训练时，它们记忆并复述敏感信息可能会带来重大的隐私风险。在这项工作中，我们提出了一种新的实用数据提取...</li><li><a href="https://arxiv.org/abs/2205.05862">AdaVAE: Exploring Adaptive GPT-2s in Variational Auto-Encoders for Language Modeling</a>：变分自编码器（VAE）已成为同时实现自然语言表示学习和生成的既定学习范式。然而，现有的基于 VAE 的语言...</li><li><a href="https://arxiv.org/abs/2307.13912">Embedding Democratic Values into Social Media AIs via Societal Objective Functions</a>：我们能否设计人工智能（AI）系统来对我们的社交媒体 Feed 进行排序，从而将减轻党派敌意等民主价值作为其目标函数的一部分？我们介绍...</li><li><a href="https://tenor.com/view/avocado-bacon-salad-lunch-salad-gif-12338945">Avocado Bacon Salad Lunch GIF - Avocado Bacon Salad Lunch Salad - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2110.03111">Cut the CARP: Fishing for zero-shot story evaluation</a>：大规模语言模型（Raffel et al., 2019; Brown et al., 2020）的最新进展在机器驱动的文本生成方面带来了显著的质和量的提升。尽管...</li><li><a href="https://arxiv.org/abs/2210.07792">Robust Preference Learning for Storytelling via Contrastive Reinforcement Learning</a>：受控的自动故事生成旨在生成满足自然语言评论或偏好约束的自然语言故事。现有的控制故事偏好的方法...</li><li><a href="https://arxiv.org/abs/2404.05595">UniFL: Improve Stable Diffusion via Unified Feedback Learning</a>：扩散模型彻底改变了图像生成领域，

导致高质量模型的激增和多样化的下游应用。然而，尽管取得了这些重大进展...</li><li><a href="https://arxiv.org/abs/2404.04860">ByteEdit: Boost, Comply and Accelerate Generative Image Editing</a>: 最近基于扩散的生成式图像编辑进展引发了一场深刻的革命，重塑了图像外扩（outpainting）和内补（inpainting）任务的格局。尽管取得了这些进步，该领域 ...</li><li><a href="https://arxiv.org/abs/2404.04465">Aligning Diffusion Models by Optimizing Human Utility</a>: 我们提出了 Diffusion-KTO，这是一种通过将对齐目标制定为最大化预期人类效用来对齐文本到图像扩散模型的新方法。由于该目标适用于...</li><li><a href="https://github.com/andreaspapac/CwComp">GitHub - andreaspapac/CwComp: Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm. AAAI 2024</a>: 用于 Forward-Forward 算法的卷积通道竞争学习。AAAI 2024 - andreaspapac/CwComp</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/35662458/">GWYRE: A Resource for Mapping Variants onto Experimental and Modeled Structures of Human Protein Complexes - PubMed</a>: 蛋白质及其相互作用结构建模的快速进展，得益于基于知识的方法论的进步以及对蛋白质结构物理原理的更好理解...</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1227556417502969917)** (4 messages): 

- **知识存储的 Scaling Laws 探索**: arXiv 上的一篇[新论文](https://arxiv.org/abs/2404.05405)介绍了一种估算语言模型可以存储的知识比特数的方法。它表明模型可以**为每个参数存储 2 bits 的知识**，这意味着一个 **7B 模型可以存储 14B bits 的知识**，这可能超过了英文维基百科和教科书的总和。
- **Eleuther 社区对新论文的思考**: 在 Eleuther 社区中，有人提到了对所讨论的**知识存储论文**的正面评价，但同时也指出该论文*难以解析*，可能需要对其相关结果进行讨论。
- **寻求 OpenAI 新模型的基准测试**: 有人询问关于最新 **OpenAI 模型版本**（如 **gpt-4-turbo**）的基准测试；问题在于当这些版本通过 API 发布时，在哪里可以找到这些基准测试结果。

**提及的链接**: <a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling Laws 描述了语言模型规模与其能力之间的关系。与以往通过 Loss 或基准测试评估模型能力的研究不同，我们估算了...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

norabelrose: https://arxiv.org/abs/2404.05971
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1227160807347851306)** (8 messages🔥): 

- **征集 `apply_chat_template` 的协作**: 一位成员表示渴望协助集成用于模型评估的 `apply_chat_template`。另一位成员确认了正在进行的工作，并邀请在其返回后提供协助，同时另一位参与者也自愿提供帮助。

- **通过 `big-refactor` 提升推理速度**: 关于 `big-refactor` 分支是否比 `main` 分支提供更快推理的查询得到了另一位成员的肯定回答。

- **ThePile v1 的种子下载**: 一位成员分享了下载 EleutherAI 的 ThePile v1 数据集的磁力链接。

- **聊天模板化的 Pull Requests**: `stellaathena` 提供了两个为 Hugging Face 模型贡献聊天模板功能的 Pull Request 链接；你可以在[这里](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808)找到第一个 PR，另一个链接在[这里](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578)。他们指出，在 transformers 库中为 `apply_chat_template` 添加批处理操作将对该项目和其他项目非常有益。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808)">[WIP] Add chat templating for HF models by haileyschoelkopf · Pull Request #1287 · EleutherAI/lm-evaluation-harness</a>: 这是一个正在进行中的 PR，延续了 @daniel-furman 在 #1209 中开始的草案，旨在添加指定的、经常被请求的聊天模板功能。目前的 TODO 包括：使用 OpenHermes 等检查性能...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1578).">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1227144864055951400)** (68 messages🔥🔥): 

- **AI 成为新的毕加索？**: 关于是否应该接受 AI 作为一种新的艺术家形式展开了辩论。虽然一些人欣赏人类艺术家的努力和辛勤工作，但也有人对 AI 生成的艺术及其对艺术署名和付出的影响表示担忧。
- **硕士生寻求 AI 聊天系统**: 一名硕士生正在为他们的论文寻找一个开源的 GPT 聊天系统模板。推荐的工具包括 **LM Studio** 和 **Open-Source LLM Advisor**。
- **Perplexity 获得好评**: 用户讨论了 **Perplexity**，这是一款具有 32K context window 的 AI 工具，能够在 **Opus** 和 **GPT-4** 等模型之间切换。一些用户已升级到 Pro 版并反馈体验良好。
- **定制化是未来 GPT 版本的关键需求**: 一位用户表达了对更好定制性的渴望，例如对系统输出进行排名以及 GPT 回复的简洁性。提出了引入“自定义指令”以获得更精细定制输出的想法。
- **GPT-4 访问受限，用户感到困惑**: 成员们报告称收到消息显示他们已达到 **GPT-4** 的使用上限，尽管他们设置使用的是 3.5 版本。分享了一个指向 OpenAI 状态更新的链接，该链接记录了对 ChatGPT 错误的持续调查。

**提到的链接**: <a href="https://status.openai.com/">OpenAI Status</a>: 未找到描述

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1227303668299206727)** (33 messages🔥): 

```html
<ul>
  <li><strong>域名验证困扰</strong>：一位用户在尝试发布 GPT 时遇到错误，询问在设置 TXT 记录后如何验证域名的建议。</li>
  <li><strong>GPT 转向 SaaS 转型咨询</strong>：一位成员正在寻求关于将 GPT 转换为单一用途 SaaS 应用程序的可用服务建议，旨在为未来的项目创建概念验证。</li>
  <li><strong>GPT 的技术困难</strong>：几位成员报告了各种问题，包括无法加载 GPT、提及（mentions）功能失效，以及尽管资金充足但因账单问题导致 API 访问被暂停。</li>
  <li><strong>聊天机器人停机报告</strong>：用户面临 GPT 停机问题，出现“GPT 无法访问或未找到”等错误，并且在检索现有对话时遇到困难。</li>
  <li><strong>服务状态更新与确认</strong>：分享了指向 <a href="https://status.openai.com/">OpenAI 服务状态页面</a> 的链接，确认了正在对影响 ChatGPT 服务的错误率上升和间歇性停机进行调查。</li>
</ul>
```

**提到的链接**: <a href="https://status.openai.com/">OpenAI Status</a>: 未找到描述

  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1227265760783503390)** (179 messages🔥🔥):

- **AI Maximum Security**: 一位成员警告不要分享或推广**越狱相关提示词 (jailbreak-related prompts)**，因为这违反了 AI 管理原则和 OpenAI 政策。他们还引用了一篇 [Google 搜索文章](https://www.google.com/search?q=Don%27t+worry+about+AI+breaking+out%2C+worry+about+us+breaking+in) 以进行深入了解。
- **Prompt Engineering 101**: 对话转变为一场关于**提示工程 (prompt engineering)** 的研讨会，以 Pokemon Showdown 和 AI 对话为例。一位用户建议使用 *meta-prompting* —— 通过要求 AI 自身生成所需输出的指令来迭代优化提示词。
- **Fine-Tuning for Excellence**: 同一位用户还透露，向 ChatGPT 索要*特定的对话示例*，然后根据这些示例索要*指令*，可以帮助在未来的输出中模仿这些模式，强调了让 AI 构建指令的重要性。
- **Guarding Against AI Missteps**: 提到了一种防止自定义 GPT 泄露其指令的技术，涉及在 “Instructions” 中添加一段短语，如果启用了 **Code Interpreter**，该短语可以缓解一些基本的提示注入 (prompt injection) 威胁。
- **ChatGPT Writes Instructions**: 多位参与者利用 ChatGPT 生成更好对话指令的能力，致力于优化生成引人入胜的**宝可梦对战对话 (Pokemon battle dialogue)** 的方法，突显了 AI 在任务特定性方面甚至可能超越工程师自身能力的潜力。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1227265760783503390)** (179 messages🔥🔥): 

- **AI Jailbreak Prompt Awareness**: 一位成员强调要警惕分享 AI 越狱提示词。他们强调了伦理考量，并引用了一个 Google 搜索词 *“Don't worry about AI breaking out, worry about us breaking in”*，以解释推广 AI 越狱技术的风险和问题。
- **Custom Instructions Against AI Misuse**: 讨论了为 AI 模型创建自定义指令，以防止在启用 Code Interpreter 时泄露敏感信息。一位成员分享了一个提示词，鼓励 Custom GPT ***委婉地拒绝*** 透露其系统细节。
- **The Documentary Nature of AI**: 一位参与者表示：***“在大语言模型 (LLM) 时代，文档即源代码。”*** 这一观点强调了 AI 文档在理解和复制模型行为方面的重要性。
- **Enhancing AI-Generated Pokémon Battle Dialogues**: 围绕使用 ChatGPT 生成更好的宝可梦对战对话进行了长时间讨论。一位成员建议使用 **meta-prompting** —— 让 AI 建议如何构建提示词以改进对话写作 —— 并配合 AI 进行迭代优化和测试。
- **Meta-Prompting as a Powerful Tool**: 一位成员为另一位用户演示了 meta-prompting 的概念，展示了如何优化 AI 的输出，以改进其为宝可梦游戏编写的对战对话。通过这个过程，用户学会了向 ChatGPT 索要特定的指令提示词，直到结果达到预期。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1227139444684095520)** (141 messages🔥🔥): 

- **Introducing AutoCodeRover**: 新加坡推出了 **AutoCodeRover**，这是一个自主软件工程师，能够以极低的成本和快速的周转时间解决与错误修复或功能添加相关的 GitHub issues。分享了该项目在 GitHub 上的链接和预印本论文。[GitHub Repository](https://github.com/nus-apr/auto-code-rover), [Preprint PDF](https://github.com/nus-apr/auto-code-rover/blob/main/preprint.pdf)。
  
- **GPT-4-Turbo Models Hit the Scene**: 训练数据截止日期为 2023 年 12 月的最新 **GPT-4-Turbo** 模型已发布，相比之前的迭代版本提供了巨大的改进。社区的反应包括观察到其在复杂任务上的推理能力有所提高，并期待其向 ChatGPT Plus 订阅者推出。[OpenAI Pricing](https://openai.com/pricing), [OpenAI's Official Tweet](https://twitter.com/OpenAIDevs/status/1777769463258988634)。

- **Music Generation Enters New Era with Udio**: 关于新型音乐生成应用 Udio 的热门话题讨论引发了关注，因其直观的音乐创作文本提示系统以及在 Beta 阶段每月每位用户 1200 首歌曲的慷慨配额，使其具有挑战 Suno 的潜力。人们对这一新玩家将如何影响音乐行业感到兴奋并充满猜测。[Udio Announcement](https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), [Reddit Discussion about Udio](https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/)。

- **Mixtral 8x22b 模型发布**：Mixtral 的 **8x22b 模型**发布引起了关注，因其庞大的参数量以及与 GPT-4 和 Claude Sonnet 性能的显著对比。讨论强调了该模型的各种技术规格及其在重型硬件上运行的能力，AI 社区正期待进一步的评估。[Teknium Tweet](https://x.com/Teknium1/status/1777875926807929157)。

- **Nvidia Blackwell 性能分析**：Nvidia 的 Blackwell 芯片成为热门话题，特别是在一份分享的分析报告对比了其总拥有成本（TCO）以及相对于 H100 和 A100 等旧型号的性能之后，重点关注其在 GPT-4 的推理（inference）和训练（training）需求中的适用性。讨论指出，在性能声明方面，营销现实主义非常重要。[SemiAnalysis Article](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://aider.chat/2024/04/09/gpt-4-turbo.html">GPT-4 Turbo with Vision 对编程来说是一次退步</a>：OpenAI 的 GPT-4 Turbo with Vision 模型在 aider 的代码编辑基准测试中的得分低于之前所有的 GPT-4 模型。特别是，它似乎比现有的 GP... 相比更容易出现“懒惰编码”现象。</li><li><a href="https://x.com/cursor_ai/status/1777886886884986944?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Cursor (@cursor_ai) 的推文</a>：Cursor 用户现在可以使用新的 gpt-4-turbo 模型。我们观察到在处理复杂任务时的推理能力有所提升。以下是 gpt-4-1106 与新 gpt-4-turbo 的示例对比：</li><li><a href="https://turbopuffer.com/">turbopuffer</a>：turbopuffer 是一个构建在对象存储之上的向量数据库，这意味着成本降低了 10 到 100 倍，采用按需计费模式，并具有极强的可扩展性。</li><li><a href="https://x.com/AbhikRoychoudh1/status/1777494000611852515">来自 Abhik Roychoudhury (@AbhikRoychoudh1) 的推文</a>：介绍 AutoCodeRover，展示我们来自新加坡的自主软件工程师！它接收 GitHub issue（修复 Bug 或添加功能），在几分钟内解决，且 LLM 成本极低，约为 $0.5！...</li><li><a href="https://x.com/7oponaut/status/1777971159478194256?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 7oponaut (@7oponaut) 的推文</a>：新 GPT-4 通过了神奇电梯测试</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - lmsys 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/kwindla/status/1777712299215901062">来自 kwindla (@kwindla) 的推文</a>：@latentspacepod 这里是来自 @chadbailey59 的视频，展示了快速语音响应 + tool calling 的可能性。</li><li><a href="https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis">Nvidia Blackwell 性能 TCO 分析 - B100 vs B200 vs GB200NVL72</a>：GPT-4 盈利能力、成本、推理模拟器、并行化解释、大模型与小模型推理及训练中的性能 TCO 建模</li><li><a href="https://supabase.com/docs/guides/database/extensions/pgvector">pgvector：嵌入与向量相似度 | Supabase 文档</a>：pgvector：一个用于存储 embeddings 并执行向量相似度搜索的 PostgreSQL 扩展。</li><li><a href="https://x.com/polynoamial/status/1777809000345505801?s">来自 Noam Brown (@polynoamial) 的推文</a>：GPT-4 的推理能力得到进一步提升 ↘️ 引用 OpenAI (@OpenAI) 的话：大幅改进的 GPT-4 Turbo 模型现已在 API 中提供，并正在 ChatGPT 中推出。</li><li><a href="https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2Otia">来自 Liam Bolling (@liambolling) 的推文</a>：🎉 对 @Google Gemini 来说是重大的一天。Gemini 1.5 Pro 现在可以理解音频、使用无限文件、执行你的命令，并让开发者通过 JSON mode 构建令人惊叹的东西！这一切都是 🆓 的。原因如下...</li><li><a href="https://openai.com/pricing">定价</a>：简单且灵活。只需为你使用的部分付费。</li><li><a href="https://x.com/gdb/status/1778126026532372486?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Greg Brockman (@gdb) 的推文</a>：新旧 GPT-4 Turbo 的对比示例：↘️ 引用 Pietro Schirano (@skirano) 的话：最新版本 gpt-4-turbo 与之前版本 0125-preview 的并排对比。不仅是...</li><li><a href="https://x.com/Teknium1/status/1777875926807929157">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：Mistral 发布了一个 8x22b 模型 ↘️ 引用 Mistral AI (@MistralAI) 的话：magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fanno...</li><li><a href="https://x.com/moultano/status/1777727219097342287">来自 Ryan Moulton (@moultano) 的推文</a>：尼日利亚推特对此反应如此强烈，让我觉得很多 ChatGPTisms 只是他们雇佣来编写微调数据的员工的口语化语言。↘️ 引用 Paul Graham (@paulg) 的话...</li><li><a href="https://x.com/getdelve/status/1777814330207297721?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Delve (YC W24) (@getdelve) 的推文</a>：我们 100% 同意。想象一下创办一家名为 Delve 的 YC 公司。↘️ 引用 Paul Graham (@paulg) 的话：我的重点不是我不喜欢 "delve" 这个词，虽然我确实不喜欢，而是它标志着文本是由...</li><li><a href="https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Liam Bolling (@liambolling) 的推文</a>：🎉 对 @Google Gemini 来说是重大的一天。Gemini 1.5 Pro 现在可以理解音频、使用无限文件、执行你的命令，并让开发者通过 JSON mode 构建令人惊叹的东西！这一切都是 🆓 的。原因如下...</li><li><a href="https://x.com/AlpayAriyak/status/1777852771514904719">来自 Alpay Ariyak (@AlpayAriyak) 的推文</a>：我在新的 GPT-4-Turbo-2024-04-09 上运行了 humaneval（base 和 plus），它在两项测试中都排名第一</li><li><a href="https://x.com/farbood/status/1777775047543054525?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 farbood — e/acc (@farbood) 的推文</a>：今天 w

<li>我们正在开源并分享一个名为 Sequel 的长寿助手 —— 本地存储：我们不会获取或查看您的数据 —— 与您的完整健康图景进行对话：血液化验、Whoop、DEXA、MRI 等...</li><li><a href="https://x.com/stevenheidel/status/1777789577438318625?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Steven Heidel (@stevenheidel) 的推文</a>：深入研究最新的 GPT-4 Turbo 模型：- 在我们的评估中各项指标均有重大改进（尤其是数学）- 知识截止日期为 2023 年 12 月 ↘️ 引用 OpenAI (@OpenAI) 大幅改进的 GPT-4 Turbo 模型...</li><li><a href="https://x.com/rohanpaul_ai/status/1777747790564589844?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：重大新闻 🔥🤯 Google 发布了采用全新 Griffin 架构的模型，其表现优于 Transformer。在多种规模下，Griffin 在受控环境下的基准测试得分均超过了 Transformer 基准模型...</li><li><a href="https://x.com/phill__1/status/1777816655386538021?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Phil (@phill__1) 的推文</a>：新的 GPT-4 Turbo 模型是唯一能解决这道数学题的模型：“确定 y = x^4 - 5x^2 - x + 4 与 y = x^2 - 3x 的四个交点的 y 坐标之和...”</li><li><a href="https://x.com/rohanpaul_ai/status/1777747790564589844?s=46&t=90">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：重大新闻 🔥🤯 Google 发布了采用全新 Griffin 架构的模型，其表现优于 Transformer。在多种规模下，Griffin 在受控环境下的基准测试得分均超过了 Transformer 基准模型...</li><li><a href="https://x.com/polynoamial/status/1777809000345505801?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Noam Brown (@polynoamial) 的推文</a>：GPT-4 的推理能力得到了进一步提升 ↘️ 引用 OpenAI (@OpenAI) 大幅改进的 GPT-4 Turbo 模型现已在 API 中提供，并正在 ChatGPT 中逐步推出。</li><li><a href="https://x.com/dylan522p/status/1777954675012305176?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Dylan Patel (@dylan522p) 的推文</a>：Nvidia Blackwell 性能 TCO 分析 B100 vs B200 vs GB200NVL72 GPT-4 盈利能力、成本推理模拟器并行性解释、大模型与小模型推理及训练中的性能 TCO 建模...</li><li><a href="https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 udio (@udiomusic) 的推文</a>：介绍 Udio，一款用于音乐创作和分享的应用，它允许您通过直观且强大的文本提示词（text-prompting）生成您喜爱风格的惊人音乐。1/11</li><li><a href="https://www.youtube.com/live/qhOwhoi8XUU?si=F_SyTNdHwCijw437&t=1083">Gen AI Office Hours: Jason, Hamel, Eugene</a>：未找到描述</li><li><a href="https://www.youtube.com/live/qhOwhoi8XUU?s">Gen AI Office Hours: Jason, Hamel, Eugene</a>：未找到描述</li><li><a href="https://x.com/BorisMPower/status/1777867583947227582">来自 Boris Power (@BorisMPower) 的推文</a>：“大幅改进” 😉 ↘️ 引用 OpenAI (@OpenAI) 大幅改进的 GPT-4 Turbo 模型现已在 API 中提供，并正在 ChatGPT 中逐步推出。</li><li><a href="https://x.com/]">来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://x.com/teortaxestex/status/1778090743816442202?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：所以，~Medium v2。我猜这意味着他们很快就会淘汰当前的 Medium。↘️ 引用 Waseem AlShikh (@waseem_s) @Get_Writer 团队有机会对 Mixtral-8x22b 进行了评估，结果...</li><li><a href="https://x.com/bindureddy/status/1778090437448024231?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Bindu Reddy (@bindureddy) 的推文</a>：这是一张关于所有模型各种基准测试的极好表格。新的 Mixtral 拥有最高的 MMLU 分数 77.3，略领先于 Qwen 72B，后者是昨天的最佳开源模型...</li><li><a href="https://x.com/danielhanchen/status/1777912653580771674?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Daniel Han (@danielhanchen) 的推文</a>：无法下载 @MistralAI 的新 8x22B MoE，但成功检查了一些文件！1. Tokenizer 与 Mistral 7b 相同 2. Mixtral (4096,14336) 新版 (6144,16K)，因此使用了更大的基础模型。3. 16bit ne...</li><li><a href="https://x.com/awnihannun/status/1778054275152937130?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Awni Hannun (@awnihannun) 的推文</a>：新的 Mixtral 8x22B 在 M2 Ultra 的 MLX 上运行良好。🤗 MLX 社区中的 4-bit 量化模型：https://huggingface.co/mlx-community/Mixtral-8x22B-4bit 感谢 @Prince_Canuma 提供的 MLX 版本和 v2r...</li><li><a href="https://x.com/reach_vb/status/1777946948617605384?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Mixtral 8x22B - 目前我们所知道的情况 🫡 > 176B 参数 > 性能...</li>

性能介于 GPT-4 和 Claude Sonnet 之间（根据其 Discord） > 使用了与 Mistral 7B 相同/相似的 Tokenizer > 6553...</li><li><a href="https://x.com/togethercompute/status/1778052158501667128?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Together AI (@togethercompute) 的推文</a>：新模型现已在 Together AI 上线！@MistralAI 最新的基础模型，Mixtral-8x22B！🚀 https://api.together.xyz/playground/language/mistralai/Mixtral-8x22B</li><li><a href="https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/">已确认——“Suno 杀手”名为 Udio</a>：我一直在调查一些人所谓的“Suno 杀手”——一个据称比其好 2 到 10 倍的音乐生成 AI 模型...</li><li><a href="https://x.com/reach_vb/status/1778020589225091453?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：成功了！使用 Transformers 运行 Mixtral 8x22B！🔥 在 DGX (4x A100 - 80GB) 上运行，并开启了 CPU offloading 🤯 ↘️ 引用 Vaibhav (VB) Srivastav (@reach_vb) mixtral 8x22B - 目前已知的信息...</li><li><a href="https://x.com/TheSeaMouse/status/1777870962882441596">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：mixtral 8x22b 配置
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1227692289808269392)** (3 条消息): 

- **即将举行的 1-bit LLMs 论文俱乐部**：关于 **1-bit Large Language Models (LLMs)** 论文的演示将于 **10 分钟后**在 **LLM Paper Club** 频道开始。欲了解更多详情并参加活动，请[在此注册](https://lu.ma/jcxntjox)。

- **深入探讨 1-bit LLMs**：这篇名为 "BitNet b1.58" 的专题论文讨论了一种 **三元 {-1, 0, 1}** 1-bit LLM，它在实现与全精度模型相当的性能的同时，更具**成本效益**。阅读论文请查看 [arXiv 提交页面](https://arxiv.org/abs/2402.17764)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">1-bit LLMs 时代：所有大语言模型都是 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit Large Language Models (LLMs) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://lu.ma/jcxntjox">LLM 论文俱乐部 (1-bit LLMs 论文) · Luma</a>：本周 @rj45 将分享 https://arxiv.org/abs/2402.17764 1-bit LLMs 时代：所有大语言模型都是 1.58 Bits。同时请为我们的下一篇论文提交建议并投票：...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1227694310418546738)** (268 条消息🔥🔥): 

- **视觉辅助问题**：会议期间有多次关于无法查看屏幕共享的报告，一些成员提供了替代平台，如 [x-ware.online](https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a07932b81cdc) 和 [matrix.org](https://matrix.to/#/#temporarylatentspace:matrix.org)。
- **深度学习论文与经验分享**：频道讨论了名为 "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" 的预印本论文，并分享了额外的资源，如 [blogpost](https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b) 和 [arXiv.org](https://arxiv.org/abs/2402.17764) 上的论文 PDF。
- **音视频问题**：除了屏幕共享问题外，还出现了项目成员在会议期间无法听到声音或发言的问题，最终导致了关于迁回 Zoom 的讨论。
- **1-bit LLMs 见解与讨论**：成员们讨论了 1-bit Large Language Models (LLMs) 的概念，重点关注训练过程中的 Regularization 和 Quantization 如何成为其成功的关键。还分享了一个相关的 Huggingface 仓库 [BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers)。
- **论文俱乐部协调与未来话题**：在聊天结束时，小组协调了下一次要讨论的论文，并建议将与时间序列相关的论文（如 TimeGPT）作为潜在的感兴趣话题。还提到了另一个 LLM，即 BloombergGPT，这促使分享了相关的播客 [YouTube 视频](https://www.youtube.com/watch?v=byCe7-c84d4) 以供进一步探索。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://matrix.to/#/#temporarylatentspace:matrix.org">邀请你在 Matrix 上交流</a>：未找到描述</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://arxiv.org/abs/2402.17764">1-bit LLM 时代：所有大语言模型都是 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型（LLM）的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://arxiv.org/abs/2402.18041">大语言模型数据集：全面综述</a>：本文开始探索大语言模型（LLM）数据集，这些数据集在 LLM 的显著进步中起着至关重要的作用。数据集作为基础架构...</li><li><a href="https://arxiv.org/abs/2310.04793">FinGPT：金融数据集中开源大语言模型的指令微调基准</a>：在迅速扩张的自然语言处理（NLP）领域，基于 GPT 的模型在金融领域的潜力日益显现。然而，将这些模型与...集成...</li><li><a href="https://shapes.inc">Shapes, Inc.</a>：Shapes 是可以在 Discord 上与你交谈的 AI 好友</li><li><a href="https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a07932b81cdc">Openhouse</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=byCe7-c84d4">BloombergGPT - 金融领域的 LLM，对话 David Rosenberg - 639</a>：今天我们邀请到了 Bloomberg CTO 办公室机器学习策略团队负责人 David Rosenberg。在与 David 的对话中，我们...</li><li><a href="https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a">Openhouse</a>：未找到描述</li><li><a href="https://github.com/Beomi/BitNet-Transformers">GitHub - Beomi/BitNet-Transformers: 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch with Llama(2) Architecture</a>：0️⃣1️⃣🤗 BitNet-Transformers：使用 Llama(2) 架构在 PyTorch 中实现的 Huggingface Transformers 版 &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; - Beomi/...</li><li><a href="https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b">[草稿] 1.58 bits?</a>：未找到描述</li><li><a href="https://github.com/AI4Finance-Foundation/FinGPT">GitHub - AI4Finance-Foundation/FinGPT: FinGPT: Open-Source Financial Large Language Models!  Revolutionize 🔥    We release the trained model on HuggingFace.</a>：FinGPT：开源金融大语言模型！变革 🔥 我们在 HuggingFace 上发布了训练好的模型。- AI4Finance-Foundation/FinGPT</li><li><a href="https://spaces.x-ware.online">Openhouse</a>：未找到描述
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227354692867199087)** (4 条消息): 

- **Gemma 1.1 Instruct 7B 备受瞩目**：**Gemma 1.1 Instruct 7B**，一个更新且改进的版本，现已在 HuggingChat 上线。该更新预计比 1.0 版本有整体提升，鼓励用户在[此处](https://huggingface.co/chat/models/google/gemma-1.1-7b-it)尝试。

- **CodeGemma 发布**：*[CodeGemma](https://huggingface.co/spaces/ysharma/CodeGemma)* 已上线，其模型针对端侧代码补全进行了优化，提供 **2B 和 7B** 两种尺寸，支持 8192k 上下文，并已在 HuggingFace 上架。Google 的 [RecurrentGemma](https://twitter.com/jeethu/status/1777703476195196982) 也已发布，这是一种非 Transformer 模型，具有出色的效果和可扩展性。

- **更经济的 Hugging Face**：HuggingFace 的 Spaces 和 Inference Endpoints 的计算价格已下调**高达 50%**，使其现在比 AWS EC2 按需服务更具成本效益。用户从 4 月起使用 Spaces 或 Inference Endpoints 即可享受此降价优惠。

- **社区洞察全新升级**：HuggingFace 的社区博客已升级为“文章（articles）”，新增了点赞和动态流展示等功能，并为论文作者提供了访问权限。访问更新后的文章并探索用户生成的内容，请点击[此处](https://huggingface.co/blog/community)。

- **Serverless GPU 和额外 ML 内容**：Hugging Face 与 Cloudflare 合作推出了 Serverless GPU 推理，并在其 ML for Games 课程中增加了一个专注于游戏中的经典 AI（Classical AI in Games）的新额外单元，为感兴趣的学习者丰富了学习资源。想要深入了解 Serverless GPU，请查看 [Deploy on Cloudflare Workers AI](https://huggingface.co/blog/cloudflare-workers-ai)；关于额外的 ML 内容，请访问 [Classical AI in Games](https://huggingface.co/learn/ml-games-course/unitbonus1/introduction)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus - HuggingChat</a>：在 HuggingChat 中使用 CohereForAI/c4ai-command-r-plus</li><li><a href="https://x.com/NSarrazin_/status/1777634083197124995">Nathan Sarrazin (@NSarrazin_) 的推文</a>：我们刚刚在 HuggingChat 上增加了对 Gemma 1.1 Instruct 7B 的支持！它应该比 1.0 有显著改进，很期待看到大家如何使用它。在这里试用：https://huggingface.co/chat/models/google/ge...</li><li><a href="https://x.com/_philschmid/status/1777673558874829090">Philipp Schmid (@_philschmid) 的推文</a>：Gemma 现在可以写代码了！🤯 🔔 @GoogleDeepMind 刚刚发布了 Code Gemma，这是一个专门的开源代码模型系列。Code Gemma 有 2B 和 7B 两个版本，非常适合设备端代码补全...</li><li><a href="https://huggingface.co/spaces/ysharma/CodeGemma">CodeGemma - ysharma 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/_philschmid/status/1775885996435087449">Philipp Schmid (@_philschmid) 的推文</a>：我们正在将 Hugging Face 上的计算价格降低多达 50%！🤯 是的，你没听错，@huggingface Spaces 和 Inference Endpoints 现在平均比 AWS EC2 按需实例便宜 20%！🤑 我们...</li><li><a href="https://x.com/mervenoyann/status/1777630974693539849">merve (@mervenoyann) 的推文</a>：最近我们对社区博客（现在称为 articles）进行了一系列更改 🆙 我们现在有了点赞功能，获得点赞的文章会出现在活动流中 🤝 我们已经向论文作者开放了访问权限 📝 使用...</li><li><a href="https://x.com/julien_c/status/1777328456709062848">Julien Chaumond (@julien_c) 的推文</a>：我们决定更新 text-generation-inference (TGI) 的许可证。我们将许可证从 HFOIL（我们的自定义许可证）切回 Apache 2，从而使该库完全开源。阅读下文...</li><li><a href="https://x.com/freddy_alfonso_/status/1777390461704953934">Freddy A Boulton (@freddy_alfonso_) 的推文</a>：由 @Wauplin 制作的带有新自定义 @Gradio 组件的非常流畅的演示 👀 ↘️ 引用 Arcee.ai (@arcee_ai)：与 @huggingface 合作，Arcee 很高兴发布我们的 MergeKit Hugging Face Space。🙌 你...</li><li><a href="https://x.com/m_olbap/status/1775201738397765775">Pablo Montalvo (@m_olbap) 的推文</a>：很难找到高质量的 OCR 数据... 直到今天！非常激动地宣布发布有史以来最大的 2 个公开 OCR 数据集 📜 📜 OCR 对文档 AI 至关重要：这里有 26M+ 页面，18b 文本...</li><li><a href="https://x.com/fleetwood___/status/1776281292109234626">Fleetwood (@fleetwood___) 的推文</a>：经过一周的绝对奋斗，Phi2 正式在 Ratchet 上运行了 🎺 目前还比较缓慢 🐌 但会有很多优化。</li><li><a href="https://github.com/huggingface/accelerate/releases/tag/v0.29.0">Release v0.29.0: NUMA affinity control, MLU Support, and DeepSpeed Improvements · huggingface/accelerate</a>：核心功能：Accelerate 现在可以优化 NUMA 亲和性，这有助于提高 NVIDIA 多 GPU 系统的吞吐量。要启用它，请在执行 accelerate config 时按照提示操作，或设置 ACCELERATE_C...</li><li><a href="https://huggingface.co/learn/ml-games-course/unitbonus1/introduction">Classical AI in Games - Hugging Face ML for Games 课程</a>：未找到描述</li><li><a href="https://x.com/clefourrier/status/1777319187913875893">Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：“评估很有趣”推文的后续：分数会根据 Prompt 格式的选择发生多大变化？给定模型的分数范围可达 10 分！:D X 轴为 Prompt 格式，所有这些评估...</li><li><a href="https://x.com/abidlabs/status/1775787643324051582">Abubakar Abid (@abidlabs) 的推文</a>：介绍 Gradio API Recorder 🪄 现在每个 Gradio 应用都包含一个 API 记录器，让你能够使用 Python 或 JS 客户端将你在 Gradio 应用中的交互重构为代码！</li><li><a href="https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion">Outpainting II - Differential Diffusion</a>：未找到描述</li><li><a href="https://huggingface.co/blog/cloudflare-workers-ai">为 Hugging Face 用户带来 Serverless GPU 推理</a>：未找到描述
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1227132805855838290)** (105 messages🔥🔥): 

- **Checkpoints 保存困扰**：一位成员在使用 `TrainingArguments` 时遇到了模型无法将 Checkpoints 保存到指定目录的问题。在确认 [[训练循环没有错误]](https://discord.com/channels/879548962464493619) 并尝试了不同路径后，他们通过显式使用 `trainer.save_model("")` 保存模型权重解决了该问题。

- **Gradio 相关问题请看这里**：当被问及 Gradio 相关咨询的正确去处时，提供了相应 Discord 频道的链接，包括 [Gradio 一般性问题](https://discord.com/channels/879548962464493619/1025174734427656283)、[Spaces 中的 Gradio](https://discord.com/channels/879548962464493619/1019296127847239751) 以及 [Gradio 功能请求](https://discord.com/channels/879548962464493619/1014577787039924226)。

- **征集 SEO 提示词**：一位成员寻求关于 SEO 博客文章 Prompts 的帮助。虽然最初的呼吁没有得到直接回应，但这表明了对内容创作指导的兴趣。

- **AI 初学者的学习之旅**：一位精通 Python 但刚接触机器学习的新成员请求关于从 LLM 或图像生成 AI 开始学习的建议。这突显了该领域初学者常见的入门问题。

- **模型错误查询与故障排除**：几位成员讨论了模型错误的问题。解决方案从检查 `max_seq_len` 等参数到更细致的建议（如拍摄终端错误的照片，这些错误通常很有启发性），涵盖了从代码协助到实际部署场景的各种情况。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/cascade/blob/main/app.py">app.py · nroggendorff/cascade at main</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot">LevelBot - huggingface-projects 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/settings/token">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/BAAI/bge-m3">BAAI/bge-m3 · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/Qz0KTGYJtUk?si=dq_Ptn1lpmwdNrt5">编程大冒险：光线追踪</a>：我尝试创建一个自定义的光线/路径追踪渲染器。包含：数学、着色器和猫！该项目使用 C# 和 HLSL 编写，并使用 Unity 游戏引擎...</li><li><a href="https://youtu.be/C94pTaKoLbU">电子商务的未来？！虚拟服装试穿 Agent</a>：我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模型穿着特定服装的图像，并产生数百万以上的社交帖子。免费访问运行...</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/BrutPitt/glChAoS.P">GitHub - BrutPitt/glChAoS.P: 3D GPU 奇异吸引子和超复分形探索器 - 实时处理高达 2.56 亿个粒子</a>：3D GPU 奇异吸引子和超复分形探索器 - 实时处理高达 2.56 亿个粒子 - BrutPitt/glChAoS.P
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1227131692637229087)** (2 messages): 

- **一天学会 NLP**：分享了一个仓库，提供了使用 IMDB 电影 50k 评论数据集进行情感分类的解决方案。该指南易于遵循且内容全面，可以作为大多数 NLP 任务的通用方法。[GitHub 上的情感分类器](https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main)。

- **探索包管理的迷宫**：分享了一个视频，讨论了包括 Conda、Pip 和 Libmamba 在内的各种包管理工具，以及解决 Linux 发行版的硬重置问题。这些内容可能会帮助那些在复杂的包管理中挣扎的人。[在 YouTube 上观看](https://youtu.be/7x4-zgCXz4M)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/7x4-zgCXz4M">关于 Conda, Pip, Libmamba 的包管理及硬重置</a>：抱歉没能更新每日视频。我生病了，而且还不得不重置/更新我的 Linux 发行版。苦中作乐，利用这次机会...</li><li><a href="https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main">GitHub - ManoBharathi93/Sentiment_Classifier: 基于 IMDB 电影数据集的情感分类器</a>：基于 IMDB 电影数据集的情感分类器。通过在 GitHub 上创建账号来为 ManoBharathi93/Sentiment_Classifier 的开发做出贡献。
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1227194354209259592)** (7 条消息): 

- **SimA：跨越多个世界的 AI 训练**：[DeepMind 发布了 SimA](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-generalist-ai-agent-for-3d-virtual-environments/Scaling%20Instructable%20Agents%20Across%20Many%20Simulated%20Worlds.pdf)，这是一个用于 3D 虚拟环境的通用型 AI Agent。该 AI 旨在跨越众多模拟世界进行扩展，并执行各种复杂任务。

- **Qdrant 结合 DSPy 增强搜索能力**：一篇新的 [Medium 文章详细介绍了](https://medium.com/ai-advances/unlocking-advanced-capabilities-integrating-qdrant-with-dspy-72e570857f23) **Qdrant** 与 DSPy 的集成，以提升搜索能力。结合这些工具可以提供增强的向量搜索，并可能解锁新的 AI 功能。

- **Karpathy 的推文引发好奇**：来自 [Andrej Karpathy](https://twitter.com/karpathy/status/1777427944971083809) 的最新推文引起了爱好者的热议。此处未指明具体内容，需直接访问链接查看详情。

- **使用 Marimo Labs 探索 HuggingFace 模型**：[Marimo Labs 团队开发了一个界面](https://github.com/marimo-team/marimo-labs)，用于交互式地实验任何 HuggingFace 模型。Marimo 为测试和调优各种 AI 模型提供了一个用户友好的环境。

- **HuggingFace 上的多语言信息提取**：在 HuggingFace Spaces 上发现一个[强大的多语言信息提取模型](https://huggingface.co/spaces/urchade/gliner_multiv2.1)。这个微型模型可用于鲁棒的信息提取任务，并根据 Apache 2.0 许可证开源。

- **Quanto 助力 Transformer 实现量子飞跃**：一个新的 [GitHub notebook](https://github.com/andysingal/llm-course/tree/main/Quantization) 展示了如何使用 Quanto 对 Transformer 进行量化。这可以使这些模型在受限硬件上的部署更加高效。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - urchade 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/andysingal/llm-course/tree/main/Quantization">llm-course/Quantization (main 分支) · andysingal/llm-course</a>: 通过在 GitHub 上创建账号为 andysingal/llm-course 的开发做出贡献。</li><li><a href="https://github.com/marimo-team/marimo-labs">GitHub - marimo-team/marimo-labs</a>: 通过在 GitHub 上创建账号为 marimo-team/marimo-labs 的开发做出贡献。</li><li><a href="https://github.com">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理您的 Git 仓库，像专家一样审查代码，跟踪错误和功能...</li><li><a href="https://marimo.app/l/tmk0k2">marimo | 下一代 Python notebook</a>: 使用 marimo（下一代 Python notebook）无缝探索数据并构建应用。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1227194612834369576)** (12 条消息🔥):

- **深入探讨 Deep Q-Learning**：分享了 [GitHub](https://github.com/SuleymanEmreErdem/deep-q-learning-applications) 上一个 Deep Q-Learning 项目集合的链接，承诺提供丰富的 Deep Q-Learning 应用及各种用例。
- **追踪数据科学演进**：介绍了 **RicercaMente**，这是一个旨在通过重要科学论文绘制数据科学演进图谱的协作项目。该项目鼓励社区参与，可在 [GitHub](https://github.com/EdoPedrocchi/RicercaMente) 上找到。
- **通过 everything-rag 释放本地 LLM 的潜力**：宣布了 **everything-rag**，这是一个完全可定制的本地聊天机器人助手，支持任何 Long Large Model (LLM) 和数据，包括使用个人 PDF 文件。它强调了该工具的开源和本地化特性，GitHub 仓库可见[此处](https://github.com/AstraBert/everything-rag)，并在 HuggingFace [Space](https://huggingface.co/spaces/as-cle-bert/everything-rag) 上提供了实时演示。
- **虚拟试穿引领时尚前沿**：创建了一个使用 IP-Adapter Inpainting 的虚拟试穿系统，并在 HuggingFace [Space](https://huggingface.co/spaces/tonyassi/fashion-try-on) 上展示，用户可以在模特身上可视化服装单品，效果令人印象深刻，尽管偶尔会出现颜色反转问题。
- **关于模型层行为的见解**：在关于模型层的交流中，观察到层之间的连接因输入类型（无论是代码、数学、问答还是聊天）而异，但在较低连接层中具有一致性。讨论还涉及针对特定情况使用目标数据集，以及 **Mixtral 8x22B** 等模型进行剪枝（pruning）的潜力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - tonyassi 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - as-cle-bert 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: 我的 Deep Q-Learning 项目</a>：我的 Deep Q-Learning 项目。通过在 GitHub 上创建账户，为 SuleymanEmreErdem/deep-q-learning-applications 的开发做出贡献。</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: 旨在通过多年来发表的科学研究追踪数据科学历史的开源项目</a>：旨在通过多年来发表的科学研究追踪数据科学历史的开源项目 - EdoPedrocchi/RicercaMente
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1227149914048561206)** (8 条消息🔥): 

- **Python 调试建议**：一位成员建议理解 **Python 类、函数、装饰器、导入**和对象，以便更好地实现代码。他们建议在测试时移除 PyTorch 实现，并在 JAX 或 TensorFlow 上启用 **eager execution**，以及利用 Python 的 `breakpoint()` 在逐行执行代码时跟踪变量变化。

- **玩转 Colab 功能**：为了辅助在 Google Colab 上编码，分享了一些技巧，例如使用 `function_name` 进行文档查询，使用 `object_name.__class__` 查找对象的类，以及使用 `inspect.getsource` 高效打印类的源代码。

- **表达感谢**：一位成员用一个简单的“🙏”表情符号对社区的帮助表示感谢。

- **先前查询的链接**：一位成员通过提供 Discord 频道链接引用了之前在 **ask-for-help** 板块提出的问题，并指出自最初提问以来，他们对 PyTorch 的理解有所提高。

- **对话系统论文请求**：有人请求与构建智能客服多轮对话系统相关的研究论文或工作，表示对聊天系统内的指令式问题解决能力感兴趣。

- **需要采样器的数学分解**：为了寻求数学见解，一位成员请求推荐继 **DDPM** 和 **DDIM** 之后的采样方法论文，旨在仅关注该领域的基础采样器。
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1227135169396477962)** (15 条消息🔥): 

- **使用 TensorFlow 深入研究计算机视觉**：一位成员询问了使用 **TensorFlow** 开始计算机视觉深度学习的资源或路线图。

- **Contrastive Loss Requires Large Batch Size**：讨论了 Contrastive Loss 受益于大 Batch Size，而 accumulation（梯度累积）或 checkpointing（检查点）等技术可以作为计算资源有限时的权衡方案。然而，有人担心 *batchnorm* 在累积大 Batch 时无法正确更新。

- **Efficient Watermark Removal from Millions of Images**：一名成员询问自动去除大量图片水印的工具。推荐了一个 AI 水印去除工具的 [GitHub 仓库](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy)及其相关的 YouTube 视频。

- **Monitoring GPU Usage**：对于无法访问任务管理器的人，指出可以使用 `nvidia-smi` 命令来监控 GPU 使用情况，而 `nvidia-smi -l` 可以实现持续监控。另一名成员提到正在寻找在模型训练期间实时记录指标的方法。

**Link mentioned**: <a href="https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy">GitHub - Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy: Aladdin-Persson-AI-Watermark-Destroy Public</a>: Aladdin-Persson-AI-Watermark-Destroy Public。通过在 GitHub 上创建账号来为 Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy 的开发做出贡献。

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1227199066346098739)** (7 messages): 

- **GPT-2 and Summarization Issues**：一名成员报告了使用 **GPT-2** 进行文本摘要时遇到的问题，即使遵循了 [HuggingFace course](https://huggingface.co/learn/nlp-course/chapter7/5) 中建议其潜在应用的说明。尽管数据集和任务被描述为简单直接，但仍然存在困难。

- **Mistral Meets RAG With Underwhelming Results**：一位参与者表示，将 **Mistral 7B** 与 **RAG (Retrieval-Augmented Generation)** 结合时，**结果令人失望**，体验到了明显低于预期的效果。

- **Pinning Down the `TL;DR:`**：针对上述 GPT-2 问题，另一位用户建议问题可能与特定时代的 Prompting 有关，特别是用于摘要指令的 "TL;DR"，暗示 Prompting 策略可能存在时间错位。

- **Sculpting Discord Bot Personality with Llama.cpp**：一位用户询问了使用 **llamacpp** 塑造 **Discord 机器人角色**的方法，寻求一种除了简单 Prompting 之外引导机器人行为的方式。他们还表示有兴趣跟踪对话历史以维持上下文。

- **Multi-Model Evaluation Using Cosine Similarity**：讨论了一种复杂的语言模型评估策略，涉及使用 Embedding 向量之间的 **cosine similarity**（余弦相似度）来评估模型是否在输出中包含了特定的知识点和辅导原则。这促使另一名成员建议使用 **weighted approach**（加权方法）来对 Embedding 进行 Pooling，以便更好地根据上下文需求定制评估。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1227297185712963636)** (18 messages🔥): 

- **Save Custom Modules in Diffusers**：一名用户在尝试保存自定义 `nn.Module` 时遇到错误；在向模块添加 Mixins 后，他们成功 **解决了问题**。

- **Schedulers/Samplers Behavior Explained**：在关于 **diffusers** 的讨论中，一名用户得到了关于为什么图像质量随 **denoising steps**（去噪步数）不同而变化的澄清。特别提到了 Ancestral sampler，并解释了 Scheduler 如何在加噪和去噪图像之间进行插值。

- **Understanding Schedulers/Samplers Maths**：一名用户请求推荐 **论文**，以理解除了 DDPM 和 DDIM 之外的基础 Scheduler/Sampler 背后的数学原理。

- **Multi-GPU Inference with SDXL**：一名用户询问了如何在多个 GPU 上使用 **MultiControlnet (SDXL)** 进行推理。提供了使用 **🤗 Accelerate** 和 **PyTorch Distributed** 进行分布式推理的指导，但指出 Pipeline 需要超过 10GB 显存的挑战。

- **Layer Decomposer Search**：一名成员请求与 Layer Decomposer 相关的 **信息或工具**，该工具可以分离并补充图像，类似于 [cre8tiveai.com](https://cre8tiveai.com/ld) 上的工具。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Noqkb8Z9xzD782BjfA6oRsGtV35N_XhB?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference">Distributed inference with multiple GPUs</a>：未找到描述</li><li><a href="https://cre8tiveai.com/ld"> Layer Decomposer（图层分离 AI）｜图像和视频编辑 AI 工具：cre8tiveAI</a>：一款基于 AI 的 SaaS，可在 10 秒内解决各种照片和插图编辑任务，例如自动上色、提高图像和视频分辨率以及剪裁等...</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">Distributed inference with multiple GPUs</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227353404842709126)** (4 messages): 

- **Gemini Pro 1.5 & GPT-4 Turbo 拓展视野**：迎接拥有 1M token 上下文的 [Gemini Pro 1.5](https://openrouter.ai/models/google/gemini-pro-1.5) 以及具备视觉能力的 GPT-4 Turbo（现位于 [openai/gpt-4-turbo](https://openrouter.ai/models/openai/gpt-4-turbo)），为 OpenRouter 模型阵容带来新进展。
- **增强的 `logit_bias` 支持推出**：`logit_bias` 参数允许用户更细粒度地影响模型输出，现已扩展到更多模型，包括 [Nous Hermes 2 Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo) 以及各种 Llama 和 Mistral 模型。
- **告别较少使用的模型**：jebcarter/Psyfighter-13B 和 jondurbin/bagel-34b-v0.2 等模型即将停用，提供 2 周宽限期，之后将返回 404 错误；migtissera/synthia-70b 将从 4 月 15 日起重定向至 xwin-lm/xwin-lm-70b。
- **新型 Mixtral 8x22B 亮相**：[Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b)（一个具有 instruct 能力的基础模型）已发布；欢迎在指定的 Discord 频道中提供反馈和讨论。
- **发布更新与降价通知**：Gemma 7B 模型已更新，且以下模型现已降价：[LZLV 70B](https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf)、[Databricks DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct) 和 [Nous Hermes 2 Mixtral 8x7B DPO](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B 是来自 Mistral AI 的大规模语言模型。它由 8 个专家组成，每个专家拥有 220 亿参数，每个 token 每次使用 2 个专家。它通过 [X](https://twitter...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it>)">Gemma 7B by google | OpenRouter</a>: Google 的 Gemma 是一个先进的开源语言模型系列，利用了最新的 decoder-only 文本到文本技术。它在文本生成任务中提供英语能力...</li><li><a href="https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf>)">lzlv 70B by lizpreciatior | OpenRouter</a>: 选定 70B 模型的 Mythomax/MLewd_13B 风格合并。由多个 LLaMA2 70B 微调模型合并而成，用于角色扮演和创意工作。目标是创建一个结合了创造力的模型...</li><li><a href="https://openrouter.ai/models/databricks/dbrx-instruct>)">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX 是 Databricks 开发的新型开源大语言模型。在 132B 参数规模下，它在语言相关的标准行业基准测试中优于现有的开源 LLM，如 Llama 2 70B 和 Mixtral-8x7B...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo>)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO 是新的旗舰级 Nous Research 模型，基于 [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b) 训练。该模型在超过 1,000,000 条原始数据上进行了训练...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5)">Gemini Pro 1.0 by google | OpenRouter</a>: Google 的旗舰文本生成模型。旨在处理自然语言任务、多轮文本和代码对话以及代码生成。查看来自 [Deepmind] 的基准测试和提示指南...</li><li><a href="https://openrouter.ai/models/openai/gpt-4-turbo)">GPT-4 Turbo by openai | OpenRouter</a>: 最新的具备视觉能力的 GPT-4 Turbo 模型。视觉请求现在可以使用 JSON 模式和 function calling。训练数据截至 2023 年 12 月。此模型由 OpenAI 更新以指向最新的...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO 是新的旗舰级 Nous Research 模型，基于 [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b) 训练。该模型在超过 1,000,000 条原始数据上进行了训练...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct)">Mistral 7B Instruct by mistralai | OpenRouter</a>: 一个 7.3B 参数的模型，在所有基准测试中均优于 Llama 2 13B，并针对速度和上下文长度进行了优化。这是 Mistral 7B Instruct 的 v0.1 版本。对于 v0.2，请使用 [此模型](/models/mistral...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-13b-chat)">Llama v2 13B Chat by meta-llama | OpenRouter</a>: 来自 Meta 的 130 亿参数语言模型，针对聊天对话进行了微调。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-70b-chat)">Llama v2 70B Chat by meta-llama | OpenRouter</a>: 来自 Meta 的旗舰级 700 亿参数语言模型，针对聊天对话进行了微调。Llama 2 是一种使用优化 Transformer 架构的自回归语言模型。微调版本...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct)">Mixtral 8x7B by mistralai | OpenRouter</a>: 由 Mistral AI 开发的预训练生成式稀疏混合专家模型（Sparse Mixture of Experts）。包含 8 个专家（前馈网络），总计 47B 参数。基础模型（未进行指令微调）- 参见 [Mixt...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

stonedjesusape: Fuck
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1227152405221609512)** (166 条消息🔥🔥): 

- **讨论模型集成**：一位用户询问如何将新的 LLM API 集成到 OpenRouter；他们被引导至私信（DM）以安排两家公司之间的对话。**Louisgv** 正在处理集成讨论。
  
- **速率限制困惑**：用户对 **OpenRouter** 在 **Gemini 1.5 Pro** 等新模型上的**速率限制（rate limits）**表示不确定，并澄清了预览模型的严格速率限制，通常允许每分钟约 10 次请求。

- **OR 上的定价和 Token 预估**：围绕 **Gemini 模型定价** 进行了详细对话，**louisgv** 解释说，出于计费目的，token 被计算为单个字符。这引发了关于对 token 定价的**潜在影响**的讨论，特别是对于中文等语言。

- **即将发布的更新预告**：**Alexatallah** 暗示可能很快会有新消息，此前观察到单日内有大量模型更新，包括 **Mixtral 8x22b** 被添加到供应商列表中。

- **Hermes DPO 的技术适配**：用户 **hanaaa__** 提到需要为 SillyTavern 打补丁，以便在 **Hermes DPO** 供应商上获得更好的性能，并指出了 **TogetherAI 的延迟**问题。他们还注意到，通过 iPhone 访问 **OpenRouter 网站**时，某些页面会出现崩溃。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-gemini-image-2-and-mlops-updates">Google Cloud Gemini, Image 2, and MLOps updates | Google Cloud Blog</a>：Vertex AI 增加了扩展的 Gemini 1.5 访问权限、新的 CodeGemma 模型、Imagen 的增强功能以及新的 MLOps 特性。</li><li><a href="https://deepinfra.com/databricks/dbrx-instruct">databricks/dbrx-instruct - Demo - DeepInfra</a>：DBRX 是由 Databricks 创建的开源 LLM。它采用混合专家（MoE）架构，总参数量为 132B，其中任何输入都会激活 36B 参数。它的性能超越了现有的开源模型...</li><li><a href="https://cloud.google.com/blog/topics/google-cloud-next/welcome-to-google-cloud-next24">Welcome to Google Cloud Next ‘24 | Google Cloud Blog</a>：Google Cloud CEO Thomas Kurian 概述了 Google Cloud Next ‘24 的所有新闻和客户动态。</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it:free">Gemma 7B by google | OpenRouter</a>：Gemma 是 Google 推出的一款先进的开源语言模型系列，利用了最新的 decoder-only 文本到文本技术。它在文本生成任务中提供英语能力...</li><li><a href="https://openrouter.ai/models?o=pricing-high-to-low">OpenRouter</a>：在 OpenRouter 上浏览模型
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1227142494446293012)** (5 messages): 

- **Meta 在 GPU 机时上的慷慨**：Meta 支持了一项关于 [LLM 知识容量](https://arxiv.org/abs/2404.05405) 的重要研究，提供了 420 万 GPU 小时，这滑稽地相当于近 **500 年** 的计算时间。
- **由 Meta 赞助的扩展定律（Scaling Laws）**：Meta 赞助了一项显著的扩展定律研究，耗费了令人咋舌的 420 万 GPU 小时，反映了 Meta 推动 AI 研究前进的承诺。
- **将 GPT-2 移植到 CUDA**：一位爱好者提到他们目前正在进行将 GPT-2 训练代码移植到 CUDA 的项目，这可能成为 AI 社区的一个卓越基准，并分享了 [llm.c 仓库](https://github.com/karpathy/llm.c/tree/master/dev/cuda)。
- **成立 CUDA 开发工作组**：响应表达出的兴趣，将成立一个工作组以促进 CUDA 相关项目的协作，这表明了一种健康的、社区驱动的 AI 开发方式。
- **Meta 令人印象深刻的 AI 硬件规格曝光**：讨论了 Meta 下一代 AI 硬件的细节，在仅 90 瓦的功耗下拥有 **354 TFLOPS/s** 的性能，并附带一篇博文概述了 Meta 在 AI 基础设施方面的强力投资。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ZeyuanAllenZhu/status/1777513016592040248">Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>：我们的 12 条扩展定律（针对 LLM 知识容量）已发布：https://arxiv.org/abs/2404.05405。我花了 4 个月提交了 50,000 个作业；Meta 花了 1 个月进行法律审查；FAIR 赞助了 4,200,000 GPU 小时。希望...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/tree/master/dev/cuda">llm.c/dev/cuda at master · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。欢迎在 GitHub 上为 karpathy/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1227157865085534278)** (1 messages): 

- **对 CUDA 中 C 代码实现的兴奋**：一位成员表达了对将**算法的 C 代码实现**集成到快速 CUDA 中的热情。他们提到可能将其添加到自己的库中，并询问了 **MIT license** 与 **Apache 2.0 license** 之间的兼容性，寻求了解许可证的人士提供建议。
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1227359912347308092)** (1 messages):

- **矩阵乘法性能详解**：一篇文章强调了矩阵形状对性能的重要性，重点关注分块 (tiling) 和内存布局 (memory layouts)。提供了一个矩阵乘法示例 (`[M x K] @ [K x N]`)，其中最优配置为 **A: M=2047, K=N=2048**，因为它避免了会对性能产生负面影响的**非对齐内存布局 (unaligned memory layouts)**。

- **答案解析提供**：链接中的 [博客文章](https://www.thonking.ai/p/answer-key-what-shapes-do-matrix) 讨论了矩阵乘法形状的性能，公开提供了第一个答案，并以读者提交问题解决方案为交换提供进一步的答案。这鼓励了参与度，并有助于加深对所呈现材料的理解。

**提到的链接**：<a href="https://www.thonking.ai/p/answer-key-what-shapes-do-matrix">Answer Key: What Shapes Do Matrix Multiplications Like?</a>：https://www.thonking.ai/p/what-shapes-do-matrix-multiplications 的配套内容。

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1227551275869143041)** (1 条消息): 

- **pmpp 书籍讲座观看派对提案**：一位成员正在组织 **伊利诺伊大学 (University of Illinois)** 关于 *pmpp* 书籍讲座的观看派对。他们提议分享 Zoom 链接来一起学习这些讲座，讲座时长为 1 小时到 1 小时 15 分钟，中间会有讨论停顿，建议的时间段是 CET 时间较早或工作日的较晚时段。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1227192278859386920)** (7 条消息): 

- **Ring-Flash Attention 疑问**：一位成员对 [ring-flash-attention 代码](https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32) 中的 `if step <= comm.rank:` 条件提出了疑问，询问为什么它不遍历所有主机的全部键值对 (key-value pairs)。
- **澄清因果自注意力 (Causal Self-Attention)**：补充背景解释说，在**因果自注意力 (causal self-attention)** 中，每个查询 (query) 不需要关注所有的键值 (key-values)，而只需要关注它之前的那些。
- **状态空间模型 (State Space Models) 实验**：一位成员表示有兴趣测试状态空间模型在执行“no-in-head”注意力 (NiH) 方面的表现，并**特别询问该过程是否可以在 Mamba 模型上运行**。
- **Flash Attention 协作工作**：正如一位成员所述，目前正在进行创建**教学版 Flash Attention 示例**的协作工作，该工作正在进行中，可在 [GitHub](https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn) 上获取。
- **模型测试承诺**：提到一位成员将尝试在 **Mamba 模型**上运行 ring-flash-attention 代码，以查看其效果。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn">ring-attention/naive_flash_attn at naive_flash_attn_examples · cuda-mode/ring-attention</a>：ring-attention 实验。通过在 GitHub 上创建账户为 cuda-mode/ring-attention 的开发做出贡献。</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32">ring-flash-attention/ring_flash_attn/ring_flash_attn.py at 55ff66fd35f329dfcc24ce7a448bfdd532865966 · zhuzilin/ring-flash-attention</a>：结合 Flash Attention 的 Ring Attention 实现 - zhuzilin/ring-flash-attention
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1227462918367612939)** (4 条消息): 

- **庆祝个人里程碑**：一位成员分享说他将在 5 月满 28 岁，并表达了一个可能不受欢迎的观点：**紧跟整个 AI 领域的发展是徒劳的，甚至可能适得其反**。相反，他主张采用更有选择性的信息获取方式，过滤掉噪音，专注于对自己真正重要的事情。
- **向流行文化致敬**：服务器的头像被认出是**悟空 (Goku)**，动漫系列《龙珠》(*Dragon Ball*) 中的角色。
- **社区里程碑**：服务器正在庆祝成员数刚刚突破 **5000 人**。
- **学习重质不重量**：一位成员回应了关于信息过载的观点，建议采用**每周阅读一次的习惯**，并强调以问题驱动学习，而不是以消费为中心的习惯，以获得更好的智力参与。
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1227229143003824190)** (5 条消息):

- **Puzzle 逻辑修正**：一条消息指出 **puzzle 11** 中的一个小错误，说明求和应该在共享索引 \( l \) 上进行。
- **Puzzle 11 修复的 Pull Request**：一名成员同意 **puzzle 11** 中的求和修正，并提到需要用 `B_MID` 来表示 MID 维度上的 block size，随后创建了一个 Pull Request 来解决此问题。这是 [GitHub Pull Request](https://github.com/srush/Triton-Puzzles/pull/10)。

**提到的链接**：<a href="https://github.com/srush/Triton-Puzzles/pull/10">minor on puzzle 11 by ZhaoyueCheng · Pull Request #10 · srush/Triton-Puzzles</a>：修复 puzzle 11 的公式以在 L 维度上求和，在 puzzle 11 中添加 B_MID 作为 block size 参数以在 MID 维度上循环。

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1227138905740935231)** (96 条消息🔥🔥)：

- **Half-Quadratic Quantization (HQQ) 实现进展**：分享了一些使用 **HQQ+** 进行 inference 的基础占位符代码（[代码示例](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808)），表明单独的 **HQQ** 仅查看权重，但包含 calibration 的 **HQQ+** 可能需要更好的性能。

- **Marlin Kernel 的混合结果**：一名成员讨论了他们使用 [Marlin kernel](https://github.com/IST-DASLab/marlin) 的经验，指出虽然 Marlin 报告 fp16 x int4 矩阵乘法有高达 4 倍的加速，但在他们的 A6000 Ada 上的结果不尽如人意，还提到了 kernel 引入的轻微误差。

- **Quantization 技术讨论**：围绕 Marlin 和 HQQ quantization 技术的使用进行了交流，并建议使用 [perplexity 评估脚本](https://discord.com/channels/1189498204333543425/1225499037516693574/1226798701293342793) 来测量有效 perplexity，旨在达到与 GPTQ 类似的结果。

- **Quantized Model 中的 Benchmark 问题和 Perplexity**：成员们比较了不同修改模型的 perplexity 分数，注意到了差异并寻求与预期性能的一致性，在 wikitext 上 group-size=64 时确定的 perplexity 约为 5.3。

- **HQQ Quantization 的调优与测试**：随后进行了关于 **HQQLinear** quantization 设置的技术讨论，特别是 quantization 设置中 `quant_scale=False, quant_zero=False` 的重要性。关于执行速度的详细讨论引发了对为什么 **AOInt4** kernel 在某些硬件上比带有 **HQQLinear** 的 `torch.matmul` 更慢的担忧，以及潜在的原因（[issue demonstration](https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137)）。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L137-L139">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/convert_hf_checkpoint.py#L89">gpt-fast/scripts/convert_hf_checkpoint.py at main · pytorch-labs/gpt-fast</a>: 在少于 1000 行 Python 代码中实现的简单且高效的 PyTorch 原生 Transformer 文本生成。 - pytorch-labs/gpt-fast</li><li><a href="https://github.com/IST-DASLab/marlin">GitHub - IST-DASLab/marlin: FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 16-32 tokens.</a>: FP16xINT4 LLM 推理 kernel，在高达 16-32 tokens 的中等 Batch Size 下可实现接近理想的 ~4 倍加速。 - IST-DASLab/marlin</li><li><a href="https://github.com/zhxchen17/gpt-fast">GitHub - zhxchen17/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>: 在少于 1000 行 Python 代码中实现的简单且高效的 PyTorch 原生 Transformer 文本生成。 - zhxchen17/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py">hqq/examples/llama2_benchmark/quant_llama2_hqq_demo.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L135">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/pytorch/blob/8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729/aten/src/ATen/native/cuda/int4mm.cu#L805-L860">pytorch/aten/src/ATen/native/cuda/int4mm.cu at 8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729 · pytorch/pytorch</a>: Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch-labs/gpt-fast/pull/155">testing HQQ [not for land] by HDCharles · Pull Request #155 · pytorch-labs/gpt-fast</a>: 来自 ghstack 的堆栈（最早的在底部）： -&gt; #155 摘要：hqq wikitext: {'word_perplexity,none': 12.698986130023261, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexi...</li><li><a href="https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137">hqq_eval_int4mm.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/f7c8151e749ec1d8c3f6d3361dcfce4feec5b3b0">HQQ 4 bit llama 2 7b · zhxchen17/gpt-fast@f7c8151</a>: export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/)** (1 messages): 

kerenzhou: 我喜欢图表上对应的代码
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227359195519778858)** (42 messages🔥): 

- **早期 CUDA 前向传播进展**：更新报告称，一个项目的所有前向层均已实现，**高效注意力机制 (efficient attention)** 是最后的障碍。第一轮优化使其达到了“状态良好”的阶段，表明相比初始代码具有**性能提升的潜力**。

- **LLM.C 仓库被强调为学习资源**：一个名为 [llm.c](https://github.com/karpathy/llm.c) 的 GitHub 仓库被分享并被誉为学习和磨练 CUDA 技能的宝贵资源。它涉及使用简单的原生 C/CUDA 进行 LLM 训练。

- **讨论 LLM.C 中 OpenMP 的使用**：成员们注意到 llm.c 代码库中使用了 OpenMP，有人建议 *OMP offloading* 可以取代直接使用 CUDA，以实现简单性和跨 GPU 厂商的兼容性，尽管在 Windows 上的支持尚不确定。

- **调试自定义 CUDA 代码中的性能问题**：不同版本的 CUDA kernel 之间进行了**性能对比**。‘flash’ 版本最初比预期慢了 3 倍；然而，在不同成员在不同硬件上进行进一步测试后，它显示出了加速效果，目前正在努力解决减速问题。

- **纯 CUDA 前向传播性能差距待弥补**：最近提交的纯 CUDA 前向传播显示执行时间为 111ms/iter，而 PyTorch 为 180ms/iter，但与经过编译和 Tensor Cores 优化的 PyTorch（运行速度为 26ms/iter）相比仍有差距。此次提交包含了性能指标的对比，旨在缩小这一**性能差距**。代码可以在 GitHub 仓库 [karpathy/llm.c](https://github.com/karpathy/llm.c) 中找到。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/msaroufim/5defcd59aed4364846d034ac01eb6cfd">gist:5defcd59aed4364846d034ac01eb6cfd</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/attention_forward.cu#L14">llm.c/dev/cuda/attention_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpa">karpa - 概览</a>: karpa 有 13 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: 使用简单、原始的 C/CUDA 进行 LLM 训练</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L170">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L53-L60">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/residual_forward.cu#L42-L48">llm.c/dev/cuda/residual_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L149">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/cuda-mode/lectures/blob/main/lecture8/occupancy.cu#L31">lectures/lecture8/occupancy.cu at main · cuda-mode/lectures</a>: cuda-mode 课程资料。通过在 GitHub 上创建账号为 cuda-mode/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1227185465212538961)** (108 条消息🔥🔥): 

- **关于 Whisper 能力的误解**：讨论中澄清了 **Whisper** 是一个语音转文本 (STT) 模型，而非文本转语音 (TTS)；虽然 **Ollama** 本身不支持 Whisper，但可以在本地使用 Whisper，或者使用由同一开发者提供的不同后端。
- **LangChain 使用案例探讨**：成员们分享了 LangChain 实际应用的见解，例如评估检索系统，其中一位成员指向了[一个涉及 RAG 指标的示例](https://docs.smith.langchain.com/cookbook/testing-examples/ragas)，用于评估检索增强生成 (RAG) 的性能。
- **LangChain 与 OpenAI API 的对比**：一位成员询问了在构建 AI Assistant 时使用 LangChain 相比直接使用 OpenAI API 的优势。共识认为，如果不需要 OpenAI 提供功能之外的集成，LangChain 可能不会带来显著价值。
- **LangChain 功能辩论**：用户讨论了 **VLLM** 支持 Function Calling 的能力，并建议使用 **Outlines**，它提供了结构化文本生成能力。
- **初学者关于开启 AI/ML 职业生涯的提问**：一位新成员在学习了基础 Python 和 MySQL 后，请求关于如何开启 AI/ML 职业生涯的指导。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/get_started/introduction#api-reference>).">Introduction | 🦜️🔗 LangChain</a>: LangChain 是一个用于开发由大语言模型 (LLMs) 驱动的应用程序的框架。</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>).">[beta] Structured Output | 🦜️🔗 LangChain</a>: 让 LLMs 返回结构化输出通常至关重要。这</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>)">[beta] Structured Output | 🦜️🔗 LangChain</a>: 让 LLMs 返回结构化输出通常至关重要。这</li><li><a href="https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start#get-started>)">Quickstart | 🦜️🔗 LangChain</a>: 语言模型输出文本。但很多时候你可能想要获得更多</li><li><a href="https://docs.smith.langchain.com/cookbook/testing-examples/ragas">RAG evaluation with RAGAS | 🦜️🛠️ LangSmith</a>: Ragas 是一个流行的框架，可帮助你评估检索增强生成 (RAG) 流水线。</li><li><a href="https://js.langchain.com/docs/integrations/chat/openai#withstructuredoutput-->).">ChatOpenAI | 🦜️🔗 Langchain</a>: 你可以按照以下方式使用 OpenAI 的聊天模型：</li><li><a href="https://python.langchain.com/docs/use_cases/data_generation#extraction-from-generated-examples>)">Synthetic data generation | 🦜️🔗 LangChain</a>: 在 Colab 中打开</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: 结构化文本生成。通过在 GitHub 上创建账户，为 outlines-dev/outlines 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1497>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用程序。通过在 GitHub 上创建账户，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/whisper.cpp">GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++</a>: OpenAI Whisper 模型的 C/C++ 移植版本。通过在 GitHub 上创建账户，为 ggerganov/whisper.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/)** (1 messages): 

lhc1921: https://python.langchain.com/docs/integrations/llms/azure_openai/
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1227283037461418055)** (3 messages): 

- **右滑实现自动化**：介绍 **TinderGPT**，一款旨在自动化 Tinder 消息发送的应用程序，承诺为用户节省时间并自动促成约会。在 [*GitHub*](https://github.com/GregorD1A1/TinderGPT) 上查找代码并为数字约会的未来做出贡献。

- **在本地进行检索增强生成聊天**：**everything-rag** 提供了一个完全可定制的本地聊天机器人助手，具有免费、100% 本地运行的功能，使用 Langchain 和 ChromaDB 向量数据库。在[此处](https://huggingface.co/spaces/as-cle-bert/everything-rag)探索 HuggingFace space，在 [*GitHub* 仓库](https://github.com/AstraBert/everything-rag)点赞，并阅读相关的[博客文章](https://astrabert.github.io/hophop-science/Attention-and-open-source-is-all-you-need/)中关于开源 LLMs 的重要性。

- **分析不同 LLMs 的结构化输出**：展示了结构化输出的性能分析，对比了流行的开源和闭源大语言模型。在 [*GitHub 页面*](https://github.com/mattflo/structured-output-performance)上查看发现结果和方法论。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - a Hugging Face Space by as-cle-bert</a>: 未找到描述</li><li><a href="https://github.com/GregorD1A1/TinderGPT">GitHub - GregorD1A1/TinderGPT</a>: 通过在 GitHub 上创建账户，为 GregorD1A1/TinderGPT 的开发做出贡献。</li><li><a href="https://github.com/mattflo/structured-output-performance">GitHub - mattflo/structured-output-performance: A comparison of structured output performance among popular open and closed source large language models.</a>: 流行开源和闭源大语言模型之间结构化输出性能的对比。 - mattflo/structured-output-performance
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227236485326045257)** (3 messages):

- **虚拟时尚试穿 AI Agent**：一位成员分享了名为 [“Future of E-commerce?! Virtual clothing try-on agent”](https://youtu.be/C94pTaKoLbU) 的 YouTube 视频，展示了他们构建的一个 AI Agent，该 Agent 可以生成模特穿着各种衣服的图像，并创建社交媒体帖子。
- **发布 AI Agent 的指导**：一位成员询问如何为他们开发的 AI Agent 发布并创建 UI，寻求相关教程指导。另一位成员建议学习 Web 开发是完成此任务的必要步骤。

**提到的链接**：<a href="https://youtu.be/C94pTaKoLbU">Future of E-commerce?! Virtual clothing try-on agent</a>：我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模特穿着特定服装的图像，并产生数百万以上的社交帖子。免费运行访问...

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1227282302783197216)** (4 条消息): 

- **多模态 RAG 革新药物识别**：一个全新的**多模态 RAG 应用**备受关注，它能够从图像中识别药物，将视觉数据与医疗领域的描述性数据库相结合。@activeloop 的博客文章展示了其在识别药房产品方面的实用性，可以在[此 Twitter 链接](https://t.co/QkuLzs34IJ)中找到。
- **活动预告：构建企业级 RAG**：@llama_index 宣布与 @traceloop 和 @getreflex 合作，展示构建企业级 **Retrieval-Augmented Generation (RAG)** 的核心组件。高级解析和可观测性功能是本次活动将讨论的核心工具之一，更多详情请见 [Twitter](https://t.co/ZkhvlI4nnx)。
- **MetaGPT 携 RAG 亮相 ICLR 2024**：介绍 Hong 等人提出的 MetaGPT，这是一个在 ICLR 2024 首发的多 Agent 框架，它将 Agent 视为软件公司中的不同角色（从 PM 到工程师），通过协作解决任务。RAG 增强的 MetaGPT 为该框架增添了前沿特性，更多细节分享在[此链接](https://t.co/sAF41j0uL4)。
- **通过执行停止工具实现 Agent 执行的可控性**：强调了执行控制工具在 Agent 系统中的重要性，@llama_index 分享了关于这些工具如何集成到旅行 Agent 的预订确认流程以及 **agentic RAG** 系统的搜索与回复功能中的见解。感兴趣的读者可以在 [Twitter](https://t.co/ByGOaqgWMd) 上关注讨论。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1227176829790457857)** (104 条消息 🔥🔥): 

- **OpenAI Agent 与 Gemini LLM 的适配**：用户讨论了如何将 LlamaIndex 文档中的 [openaiagent 示例 notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb) 适配以支持 *Gemini LLM*，建议进行代码修改，例如将 `OpenAI llm` 替换为 `gemini LLM`，并使用 `ReActAgent` 代替 `OpenAIAgent`。

- **RAG 优化探索**：一位用户寻求关于优化短文档 *Retrieval Augmented Generation (RAG)* 的建议，得到的推荐是查看 [Gradient AI](https://gradient.ai/blog/rag-101-for-enterprise) 上的 *RAG 101*，并参考 *MTEB leaderboard* 获取 Embedding 模型信息。

- **在 LlamaIndex 中创建工具**：围绕如何在 LlamaIndex 中创建新工具并动态添加到 `OpenAIAgent` 展开了讨论。经过详细交流，一位成员尽管面临各种挑战，最终成功使用 `FunctionTool` 创建了工具。

- **调试 LLM Prompt 问题**：一位成员询问如何查看发送给 LLM 的确切 Prompt 以进行调试。他们被引导至一份[日志指南](https://discord.com/channels/1059199217496772688/1227269649440313357/1227271613234282637)，最终发现他们需要一种特定的聊天模式，该模式有条件地使用 RAG 以减少不必要的 LLM 调用。

- **集成困扰与示例请求**：用户询问了项目设置、开源工具的集成说明以及示例用例。参考资料包括 [YouTube](https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P) 上的 SEC Insights 项目端到端指南视频，以及 [GitHub](https://github.com/run-llama/sec-insights/tree/main/backend/app/chat) 上的源代码。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: 未找到描述</li><li><a href="https://tenor.com/view/im-a-sad-panda-peetie-south-park-crying-disappointed-gif-21544015">Im A Sad Panda Peetie GIF - Im A Sad Panda Peetie South Park - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/disco-dance-party-happy-zebra-gif-16162722">Disco Dance GIF - Disco Dance Party - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mindblown-omg-triggered-gif-19814900">Mindblown Omg GIF - Mindblown Omg Triggered - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/docstore/postgres/">Postgres - LlamaIndex</a>: 未找到描述</li><li><a href="https://gradient.ai/blog/rag-101-for-enterprise">Gradient Blog: RAG 101 for Enterprise </a>: 企业级 RAG 101 - Gradient 团队</li><li><a href="https://youtu.be/C94pTaKoLbU">Future of E-commerce?! Virtual clothing try-on agent</a>: 我构建了一个 Agent 系统，它可以自主迭代并生成 AI 模型穿着特定服装的图像，并产生数百万条社交帖子。免费运行访问...</li><li><a href="https://github.com/microsoft/autogen/blob/main/notebook/agentchat_inception_function.ipynb">autogen/notebook/agentchat_inception_function.ipynb at main · microsoft/autogen</a>: 一个用于 Agentic AI 的编程框架。Discord: https://aka.ms/autogen-dc。路线图: https://aka.ms/autogen-roadmap - microsoft/autogen</li><li><a href="https://github.com/run-llama/sec-insights">GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex</a>: 一个使用 LlamaIndex 的真实全栈应用 - run-llama/sec-insights</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/chat_engine">llama_index/llama-index-core/llama_index/core/chat_engine at main · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/sec-insights/tree/main/backend/app/chat">sec-insights/backend/app/chat at main · run-llama/sec-insights</a>: 一个使用 LlamaIndex 的真实全栈应用 - run-llama/sec-insights</li><li><a href="https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P">Discover LlamaIndex: SEC Insights, End-to-End Guide</a>: secinsights.ai 是一个全栈应用，利用 LlamaIndex 的检索增强生成 (RAG) 功能来回答有关 SEC 10-K 和 10-Q 文档的问题...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo/?h=azure">Azure AI Search - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/function_tool.py#L31">llama_index/llama-index-core/llama_index/core/tools/function_tool.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/types.py#L97">llama_index/llama-index-core/llama_index/core/tools/types.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1227176894680666152)** (2 条消息): 

- **Cookbook 获得粉丝**：一位成员对 Llama Index GitHub cookbook 中提供的 **openaiagent 示例**表示**赞赏**，认为它“非常有用”，并询问是否可能有针对 **Gemini LLM 的类似资源**。相关资源可以在此 [GitHub notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb) 中找到。

- **新成员寻求关于 API Key 的澄清**：讨论中的一位新参与者对服务的运行方式表示**困惑**，并参考文档指南询问是否需要 **OpenAI 的 API Key** 才能使其正常工作。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1227410537776943104)** (87 条消息🔥🔥):

- **Pixart Sigma 性能评估**：成员们讨论了 **Pixart Sigma** 在 3090 上的表现，指出其速度快且前景广阔，**Prompt 执行时间为 8.26 秒**。然而，输出结果被描述为“破碎（mangled）”，用户注意到当前开源模型的输出结果中存在“溢色（bleed）”现象。

- **Mistral 22b x 8 发布讨论**：提到了 **Mistral 22b x 8** 已经发布。分享了 **mixtral-8x22b** 的磁力链接，随后引发了热烈反响，并有关于其是否与 **mistral-large** 存在潜在关联的询问。

- **对 Ella SDXL 和 SD3 的质疑**：讨论转向了 Ella SDXL 变得可用的可能性较低，以及对 **Stable Diffusion V3 (SD3)** 优势的怀疑，认为其不如 **Terminus** 和 **Pixart Sigma**。成员们还评估了行业对 Sora 发布后的反应，这影响了包括 Stability、Pika labs、Runway 和 Midjourney 在内的竞争对手。

- **新音频生成方案涌现**：**Udio** 引起了轰动，这是一款由艺术家支持的、通过 Text-prompt 进行直观音乐创作的应用；此外还有 **Huggingface 团队** 推出的新型 TTS 引擎，支持语音提示（Voice Prompting）。

- **AI 加速硬件热潮**：成员们讨论了规格惊人的新型 Meta 训练和推理加速器（**AI-MTIA**），反映了大科技公司开发自有 AI 加速硬件解决方案的趋势。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce`">未找到标题</a>：未找到描述</li><li><a href="https://www.udio.com/songs/renWwtB7Zqk2mqZamEHHgJ>">未找到标题</a>：未找到描述</li><li><a href="https://x.com/udiomusic/status/1778045337833193720">来自 udio (@udiomusic) 的推文</a>：我们的目标是让 Udio 成为音乐家和非音乐家都能使用的变革性工具，我们很高兴能得到领先艺术家 @iamwill 和 @common 的支持。 8/11</li><li><a href="https://news.ycombinator.com/item?id=39992817">Show HN: Sonauto – 一个更具可控性的 AI 音乐创作工具 | Hacker News</a>：未找到描述</li><li><a href="https://x.com/udiomusic/status/1778045322654003448">来自 udio (@udiomusic) 的推文</a>：介绍 Udio，一款用于音乐创作和分享的应用，让你能够通过直观且强大的 Text-prompting 生成你喜欢的风格的精彩音乐。 1/11</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1227327910432739621)** (9 条消息🔥): 

- **重新评估多模态模型中的“Zero-Shot”泛化**：[最近的一篇论文](https://arxiv.org/abs/2404.04125)质疑了 CLIP 和 Stable-Diffusion 等多模态模型中“Zero-Shot”泛化真实存在的程度。对各种模型和数据集的分析表明，性能在很大程度上取决于预训练数据中概念的显著性。

- **CLIP 模型的数据质量胜过数量**：在针对较不常见概念测试 CLIP 模型时，改进数据过滤以及针对质量和多样性的选择至关重要，其重要性可能超过单纯增加数据量。

- **Google 推进更大规模的 Griffin 模型**：据报道，Google 发布了一个采用新 Griffin 架构的模型，增加了 10 亿个参数，号称在长上下文环境下具有更好的性能和吞吐量。详情可见其 [subreddit 帖子](https://www.reddit.com/r/MachineLearning/comments/1b3leks/deepmind_introduces_hawk_and_griffin_r/)。

- **新研究挑战传统 LLM 训练方法**：一篇[开创性论文](https://arxiv.org/abs/2404.03715)提出了一种替代基于人类反馈的强化学习（RLHF）的方法，通过直接针对“成对（pair-wise）”或一般偏好进行优化，即使在 70 亿参数的模型上也显示出显著的性能提升。

- **大语言模型性能提升**：上述方法与其他领先模型相比提供了显著的性能飞跃，表明了成对优化策略相比传统点对（point-wise）奖励方法的潜在优势。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.03715">Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences</a>：本文研究了如何利用来自强大 Oracle 的偏好反馈对大语言模型（LLM）进行后期训练，以帮助模型实现迭代式的自我改进。后期训练 LLM 的典型方法...</li><li><a href="https://arxiv.org/abs/2404.04125">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>：网络爬取的预训练数据集是多模态模型令人印象深刻的“Zero-Shot”评估性能的基础，例如用于分类/检索的 CLIP 和用于图像生成的 Stable-Diffusion...</li><li><a href="https://www.reddit.com/r/singularity/comments/1bzzreq/google_releases_model_with_new_griffin/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1227147549325135914)** (51 条消息🔥): 

- **GPT-4 隆重登场**：**OpenInterpreter** Discord 社区对新发布的 **GPT-4** 模型感到兴奋，该模型相比前代有显著改进，包括**集成的视觉能力**和增强的性能。讨论中还提到 GPT-4 的速度*实际上快了 3 倍*，一些用户分享了对其速度的亲身体验，称赞其响应时间极短且运行迅速。

- **GPT-4 Turbocharged 且低调运行**：在发布热潮中，一名成员注意到 **GPT-4** 缺乏广泛的关注或详细信息，社区之外几乎没有实质性的讨论，只有 OpenAI 的发布页面作为关于 [持续模型升级](https://platform.openai.com/docs/models/continuous-model-upgrades) 的主要信息来源。

- **Mixtral 与 OI 兼容性咨询**：出现了一些关于 **Mixtral 8x22b** 与 **OpenInterpreter (OI)** 潜在匹配的讨论，用户将其与之前的 8x7b 等模型进行对比，并考虑其在 OI 框架内的性能影响。

- **对 Command r+ 的热情**：一位成员对名为 **Command r+** 的模型赞不绝口，称其为角色扮演（RP）和精确遵循指令方面*有史以来最好用的模型*，认为它感觉像是 GPT-3.5 的更好版本，并且在 Benchmark 测试中优于旧版 GPT-4，尤其是在使用正确 Prompt 的情况下。

- **Command r+ 的算力难题**：关于在本地运行 **Command r+** 所需算力的讨论浮出水面，成员们讨论了各自的配置，其中一人报告称*即使是 4090 也不足以*获得理想性能，这表明可能需要更强大的硬件。
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1227145348942921738)** (38 条消息🔥): 

- **排查 01 热点重连问题**：一位成员解决了重新连接 01 的 WiFi 和服务器设置页面的问题，建议进行出厂重置并访问 [captive.apple.com](http://captive.apple.com) 以触发门户页面。此外还建议删除旧的 WiFi 凭据。

- **01 在 Windows 11 上的安装障碍**：成员们报告了安装 01 后对话无响应的问题，尽管麦克风功能正常。建议包括检查 Python 脚本并确保已安装 [sounddevice](https://pypi.org/project/sounddevice/)。

- **根据 GitHub 仓库构建 01**：一位用户分享了根据物料清单（BOM）购买零件，并使用 01 GitHub [仓库](https://github.com/OpenInterpreter/01?tab=readme-ov-file) 中提供的文件 3D 打印机身的经验。

- **澄清 01 对 Raspberry Pi 的需求**：讨论澄清了 01 并不强制要求 Raspberry Pi，在任何电脑上运行 Open Interpreter 或 01OS 即可。对于有兴趣将 Raspberry Pi 加入设置的用户，建议在专门的论坛发起更广泛的讨论。

- **使用本地 IP 进行 01 服务器配置**：一位新的 01 用户在面对配置和理解 ngrok 域名的困惑后，成功使用其 MacBook 的本地 IP 地址将设备连接到服务器。

- **订单更新与客户服务咨询**：针对客户订单状态的查询，有人提到一旦有更新就会发送电子邮件。目前所有的订单状态被幽默地称为“仍在准备中”（still cooking）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dashboard.ngrok.com/cloud-edge/domains/new">ngrok - Online in One Line</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file#01-server">GitHub - OpenInterpreter/01: The open-source language model computer</a>: 开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file">GitHub - OpenInterpreter/01: The open-source language model computer</a>: 开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1227274131423367280)** (45 条消息🔥): 

- **Google 发布非 Transformer 惊喜**：Google 低调推出了一个名为 [Griffin](https://huggingface.co/google/recurrentgemma-2b) 的 20 亿参数循环线性注意力模型，这是继最近 CodeGemma 发布之后的又一重大进展，并被拿来与 RWKV 架构进行比较。相关研究论文可在 [arXiv](https://arxiv.org/abs/2402.19427) 上查阅。
  
- **模型快速发布的传闻**：对话涉及了快速且有些出人意料的模型发布，例如 **Mixtral**，这可能是由于来自其他预期模型发布（如 llama 3 smol 和 Cohere）的竞争压力所致。

- **OpenAI 发布自家新闻**：OpenAI 的推文暗示了一项有趣的进展，但消息中并未讨论具体细节——仅提供了一个 [OpenAI 推文链接](https://vxtwitter.com/OpenAI/status/1777772582680301665)，没有更多上下文。

- **Mixtral 带来的新模型兴奋感**：新模型 Mixtral 引起了轰动，[Twitter 对话](https://fxtwitter.com/sophiamyang/status/1777978822199017728)中强调了它与 Mistral 和 Miqu 等先前模型的区别。

- **公开人类评估博客提案**：一名成员讨论了创办一个专门针对新发布模型进行公正人类评估的博客的可能性，并对目前过度关注基准测试分数而非开发者实际效用的现状表示不满。呼吁大家为这一努力做出贡献并参与其中。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sophiamyang/status/1777978822199017728">Sophia Yang 博士 (@sophiamyang) 的推文</a>: @LiHongtu12138 两者都不是。这是一个全新的模型。</li><li><a href="https://x.com/jeethu/status/1777703476195196982?s=46">Jeethu Rao (@jeethu) 的推文</a>: 看起来 Google 刚刚低调发布了一个基于 2B 循环线性注意力的模型（非 Transformer 架构，即 Griffin 架构）。在我看来，这比 CodeGemma 意义更大。据我所知，这个 cl...</li><li><a href="https://x.com/realmrfakename/status/1777882147707322479?s=46">mrfakename (@realmrfakename) 的推文</a>: 更新：Mistral Discord 服务器的一名管理员确认该模型不是之前的任何模型，而是一个全新的模型 ↘️ 引用 mrfakename (@realmrfakename) 新的 Mixtral 模型是... (根据...</li><li><a href="https://fxtwitter.com/jphme/status/1778028110954295486">Jan P. Harries (@jphme) 的推文</a>: @MistralAI 新的 8x22b 模型的首批 AGIEval 结果已经出炉，碾压了所有其他开源（基础）模型 - 🤯
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1227367608282779699)** (14 条消息🔥):

- **解码现代 LLM 训练中的 RLHF**：Sebastian Raschka 发布了一份关于 [RLHF 的详细分析](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)，将其视为 LLM 训练的关键部分，影响模型的帮助性和安全性，并比较了 ChatGPT 和 Llama 2 对 RLHF 的使用，同时定期更新替代方案。
- **Rejection Sampling 引发疑问**：一位阅读该文章的用户对 **Rejection Sampling** 感到困惑，这一概念意味着使用最佳的模型生成结果进行 PPO 更新，该用户寻求关于为什么这比从平均或较差的生成结果中学习更优的见解。
- **通过在线资源探索 PPO**：另一位用户旨在通过查阅 [Cameron Wolfe 的新闻通讯](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)来理清对 PPO 的理解，并承认自己缺乏 RL 方面的先验知识。
- **澄清 Rejection Sampling 的作用**：**Nathan Lambert** 澄清说，Rejection Sampling 在持续训练之前应用于整个指令数据集，并承认这一过程在相关论文中没有得到很好的记录，需要与作者直接通信。
- **关于 Rejection Sampling 有效性的思考**：Lambert 进一步解释了 Rejection Sampling 的可能原理：**大部分数据可能质量较低**，这意味着在 PPO 之前过滤出高质量样本可能会带来更稳定的训练结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives">LLM Training: RLHF and Its Alternatives</a>：在讨论 LLM 时，无论是在研究新闻还是教程中，我经常引用一个名为 Reinforcement Learning with Human Feedback (RLHF) 的过程。RLHF 是现代 LLM 训练中不可或缺的一部分...</li><li><a href="https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo">Proximal Policy Optimization (PPO): The Key to LLM Alignment</a>：现代策略梯度算法及其在语言模型中的应用...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1227442004267368479)** (7 条消息): 

- **在德克萨斯州追逐日食**：成员们分享了前往德克萨斯州以获得日食或天文事件**最佳观测体验**的个人经历。
- **与天空的短暂邂逅**：尽管天气多云，一位成员仍表达了捕捉到该事件**一瞥**的喜悦，认为自己**运气极佳**。
- **宇宙级的相似感**：一位成员指出，这一天文景象酷似 Netflix 剧集《三体》(3-Body) 中“天空中的眼球”，唤起了流行文化中的意象。
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1227341028877406209)** (2 条消息): 

- **ML Discord 社区中的普遍趣事**：一位成员幽默地注意到 <:berk:750111476483752166> 表情符号在各种机器学习 Discord 社区中的广泛使用。
- **社区分享的幽默**：一位用户发现了一些有趣的事，并宣布“太好笑了，不能不分享”到频道中。
  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1227435523916693568)** (10 条消息🔥): 

- **辩论 RLHF 训练后改进的优点**：一位成员重点介绍了一篇[新论文](https://arxiv.org/abs/2404.03715)，讨论通过使用 Oracle (预言机) 进行迭代反馈来改进大语言模型 (LLMs)，这可能会挑战依赖奖励最大化的典型 Reinforcement Learning from Human Feedback (RLHF) 方法。

- **模型效能中尺寸的重要性**：有人提出了一个问题，将 7B 模型与 GPT-4 进行比较，暗示较小的模型可能表现优于较大的模型。

- **对 Benchmark 优化的怀疑**：成员们对 LLM 评估的 Benchmark 表示怀疑，指出虽然 Benchmark 可以被优化，但这并不一定反映出更好的基础模型性能。

- **务实模型改进哲学**：一位成员表示，相比于他们认为的关于该主题的“胡扯”新论文，他们更倾向于通过**更好的数据**和**更好的 Scaling** 来实现模型切实的改进。

- **Benchmark 作为不完美的代理指标**：大家承认，虽然像 alpacaeval 这样的 Benchmark 一旦开始优化可能会失效，但它们仍可作为衡量模型能力的临时手段。

**提到的链接**：<a href="https://arxiv.org/abs/2404.03715">Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences</a>：本文研究了使用来自强大 Oracle 的偏好反馈进行训练后的 LLM，以帮助模型在自身基础上迭代改进。训练后的典型方法...

  

---

**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1227136719510306866)** (3 messages): 

- **LLM 缩放定律揭秘**：一篇新论文介绍了 **12 条关于大语言模型 (LLM) 知识容量的缩放定律 (scaling laws)**，这在大模型时代可能具有关键意义。该研究耗费了大量资源，Meta 的 FAIR 团队为这项研究赞助了 **4,200,000 GPU 小时**。[在此阅读论文](https://arxiv.org/abs/2404.05405)。
- **量化与 MoE 的探索**：论文还探讨了**推理与量化**，揭示了将模型权重缩减至 **int8** 不会损害即便达到最大容量模型的知识容量，并且拥有 32 个专家的**混合专家模型 (MoE)** 能够高效地保留知识容量。[查看详细结果](https://fxtwitter.com/zeyuanallenzhu/status/1777513026243174543?s=46)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/zeyuanallenzhu/status/1777513016592040248?s=46">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>: 我们的 12 条缩放定律（针对 LLM 知识容量）已发布：https://arxiv.org/abs/2404.05405。我花了 4 个月提交了 50,000 个作业；Meta 花了 1 个月进行法律审查；FAIR 赞助了 4,200,000 GPU 小时。希望...</li><li><a href="https://fxtwitter.com/zeyuanallenzhu/status/1777513026243174543?s=46">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>: 结果 8/9：量化和 MoE 的缩放定律。 // 量化到 int8 即使对于达到最大容量的模型也不会损害知识容量 => 2bit 的知识可以存储到 int8 // MoEs 具有...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1227323705118560288)** (52 messages🔥): 

- **重构 tinygrad 以提升效率**：关于精简 tinygrad 代码库的讨论正在进行中，重点在于减少代码行数并增强代码的评审就绪度，同时解决需要为非磁盘后端提供 JIT 支持的后端特性问题。
- **追求权重无关网络**：一位成员表示有兴趣使用 tinygrad 创建一个权重无关网络 (Weight Agnostic Network) 来训练游戏，并打算尝试使用 ReLU 激活函数。
- **将 MNIST 合并到 tinygrad**：重点介绍了将 MNIST 更紧密地集成到 tinygrad 中的工作，[Pull Request #4122](https://github.com/tinygrad/tinygrad/pull/4122) 展示了这一变动，并揭示了 AMD 上的一个编译器 bug，呼吁在 CI 中添加测试以捕获此类问题。
- **Abstractions3 中的变量命名**：关于 abstractions3 背景下变量命名必要性的辩论，建议变量应由其 ID 定义。这导致了一个变化，即 **var_vals** 将是一个全局字典，而不是存在于每个 **ScheduleItem** 中。
- **CI 性能与测试讨论**：针对 CI 性能退化和缺失测试（特别是针对 `copy_from_fd` 功能）提出了担忧，这将在随后的 [pull request](https://github.com/tinygrad/tinygrad/pull/4125/files) 中解决。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464">不再有底层 diskbuffer，那只是设备 (#4129) · tinygrad/tinygrad@ee457a4</a>: 你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - 不再有底层 diskbuffer，那只是设备 (#4129) · tinygrad/tinygrad@ee457a4</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4124">abstractions3 目前是 geohot 的一厢情愿 · Pull Request #4124 · tinygrad/tinygrad</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4122">用 5 行代码将 mnist 引入仓库，由 geohot 提交 · Pull Request #4122 · tinygrad/tinygrad</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/8625081714/job/23641105062">用 5 行代码将 mnist 引入仓库 (#4122) · tinygrad/tinygrad@fea774f</a>: 你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - 用 5 行代码将 mnist 引入仓库 (#4122) · tinygrad/tinygrad@fea774f</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4125/files">创建 schedule 具有全局变量，由 geohot 提交 · Pull Request #4125 · tinygrad/tinygrad</a>: 未找到描述
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1227199232448921600)** (13 messages🔥):

- **自定义加速器分步指南**：一位用户分享了关于向 tinygrad 添加自定义加速器的[分步指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md)，并指向了一个包含详细说明和图解的 GitHub 仓库。
- **寻找网络示例**：一名成员正在寻找使用 tinygrad 的整洁网络示例，并被引导查看 tinygrad 仓库中的 `examples/` 目录。
- **讨论 'Double Reducc'**：用户们正在讨论一个名为 'double reducc' 的问题，大家似乎达成了共识并确认了该问题的存在，表明正在协作寻求解决方案。
- **在 Tinygrad 中将 Tensor 转换为数组**：有人提出了关于在 tinygrad 环境中将 Tensor 转换为数组的问题。另一位用户建议在 Tensor 上使用 `.numpy()` 来完成此转换。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic">mesozoic - 概览</a>：mesozoic 有 39 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md">tinygrad-notes/addingaccelerator.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1227434265914900501)** (40 messages🔥): 

- **Mixtral 模型演进**：讨论了新的 **Mixtral 8x22B 模型**，据推测该模型拥有约 1400 亿个参数，在一个 1.5GB 的数据集上以 rank32 运行，且 loss 异常低。成员们很好奇这个版本是经过 instruction tuned（指令微调）的还是 base model（基座模型）。
- **量化与模型大小限制**：社区成员正在研究用于实际用途的 **quantization**（量化），并对在典型开发者可用的资源下运行像 **Mixtral 8x22B** 这样的大型模型的可行性表示担忧。大家有兴趣在模型大小和实用性之间找到平衡。
- **社区快速贡献**：一位贡献者已经开始将新的大模型 **Mixtral-8x22B** 上传到 Hugging Face，展示了社区对进展的快速响应。分享了仓库链接：[Hugging Face - Mixtral-8x22B](https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main)。
- **寻找兼容的前端**：有人提出了关于寻找一个可 Web 自托管且兼容各种 API（包括 OpenAI 和 Google）的前端问题。回复中未提到具体的解决方案。
- **生成式 AI 黑客松公告**：宣布 Samsung Next 2024 生成式 AI 黑客松将于 5 月 11 日在纽约举行，重点关注 **Health & Wellness**（健康与保健）和 **Mediatech**（媒体技术）赛道。提供了详情和申请链接：[Samsung Next AI Hackathon](https://lu.ma/nextgenainyc)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF at main</a>：暂无描述</li><li><a href="https://lu.ma/nextgenainyc">Samsung Next 2024 Generative AI Hackathon · Luma</a>：🚀 活动动态：申请参加 Samsung Next 2024 生成式 AI 黑客松！我们将探索两个赛道：Health &amp; Wellness：利用 AI 的力量改善医疗结果...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1227253703359074384)** (4 messages): 

- **Axolotl 数据集版本控制功能即将到来**：一位成员表示有兴趣为 Axolotl 添加数据集版本控制功能，并指出目前该功能缺失。回复指出 **dataset versioning** 此前未被要求过，并鼓励该成员继续进行贡献。

- **LoRA 层初始化技术引发关注**：分享了来自 CFGeek 推文的一个技巧，小组讨论了一种新的 **LoRA** 层初始化方法，该方法涉及使用原始权重矩阵的 SVD 以获得更好的微调结果。该技术被称为 **PiSSA** (Principal Singular Values and Singular Vectors Adaptation)，据报道可以提高微调性能，详见 [arXiv 摘要](https://arxiv.org/abs/2404.02948) 和相应的 [GitHub 仓库](https://github.com/GraphPKU/PiSSA)。

**提到的链接**：<a href="https://x.com/cfgeek/status/1777556286047166673?s=46&t=hIokEbug9Pr72tQFuXVULA">来自 Charles Foster (@CFGeek) 的推文</a>：是的！如果你基于原始权重矩阵的 SVD（及其顶部的奇异值和奇异向量）初始化 LoRA 层，你会得到显著更好的微调结果。这是一个非常直接的...

  

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1227168958973607937)** (9 messages🔥): 

- **使用挪威语文章进行预训练**：一名成员正准备使用挪威语艺术数据集预训练一个大型语言模型（LLM），以增强其语法能力。他们询问了拆分文章的最佳方式，并收到了建议，考虑使用每篇文章一行的格式，可能采用 `.jsonl` 格式。

- **寻找函数调用微调数据集**：有人请求适用于 JSON 模式或函数调用的优质数据集，特别是为了使用 axolotl 微调用于 **函数调用的 LoRAs**；然而，在当前的消息记录中没有提供具体建议。

- **mixtral-qlora-fsdp 模型的硬件能力查询**：一名成员询问 **mixtral-qlora-fsdp** 模型是否能运行在双 24GB GPU 配置上，但目前没有后续信息或解答。

- **修复空队列错误**：一位遇到空队列错误的用户被建议在迭代前检查空条件，并获得了一份重构后的代码作为潜在解决方案。

- **简化代码重构**：提供了一个代码重构的示例，将检查 stop token 的函数从几行简化为仅一行，提升了代码效率。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1227241338903199844)** (2 messages): 

- **寻找函数调用与 JSON 数据集**：一名成员询问了关于函数调用或 JSON 解析的数据集。
- **分享 Agent-FLAN 数据集**：另一名成员提供了数据集建议，分享了 HuggingFace 上的 [Agent-FLAN 数据集链接](https://huggingface.co/datasets/internlm/Agent-FLAN)。该数据集包含 AgentInstruct、Toolbench 以及自定义的负样本 Agent 样本，旨在为 LLM 提供有效的 Agent 微调。

**提及的链接**：<a href="https://huggingface.co/datasets/internlm/Agent-FLAN">internlm/Agent-FLAN · Datasets at Hugging Face</a>：未找到描述内容。

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1227206147388342343)** (8 messages🔥): 

- **暗示 Mojo 中的 C 风格格式化**：有提示称，在 Mojo 开发者等待 Python 风格的 `f` 字符串时，可以通过导入 `_printf as printf` 来使用旧的 C 风格格式化，但要注意该功能 *“可能不会永远存在”*。
- **为初学者总结的 API 文档**：一名成员分享了一个 [Notion 页面链接](https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078)，该页面以摘要格式提供了翻译后的 API 文档，旨在帮助初学者。
- **探索 Mojo stdlib 之外的贡献**：讨论了潜在贡献者如何参与 Mojo 或 MAX 项目，建议包括在 *lightbug* 上进行 Web 开发，在 *basalt* 上进行 AI 开发，或启动一个新项目。
- **精选的 Mojo 资源列表**：在 GitHub 上维护的精选列表 [awesome-mojo](https://github.com/mojicians/awesome-mojo) 中也可以找到 Mojo 的贡献机会和资源。
- **征集社区对 Mojo Traits 的反馈**：发起了一项关于在 Mojo 中使用 traits 的新讨论，并在 [GitHub discussion](https://github.com/modularml/mojo/discussions/2259) 中请求广大社区提供反馈。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078?pvs=4">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://github.com/modularml/mojo/discussions/2259">[提案]：弃用未实现方法的省略号 (...) · modularml/mojo · Discussion #2259</a>：动机：Mojo 渴望成为 Python++ 的无缝继任者，紧密遵循 Pythonic 原则，并为 Python 社区培养积极的体验。目前使用 ... 的做法...</li><li><a href="https://github.com/mojicians/awesome-mojo">GitHub - mojicians/awesome-mojo: 精选的优秀 Mojo 🔥 框架、库、软件和资源列表</a>：精选的优秀 Mojo 🔥 框架、库、软件和资源列表 - mojicians/awesome-mojo
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1227304021505867826)** (2 messages): 

- **Modular 分享更新**：Modular 在推特上发布了更新，可以在其官方 Twitter 页面查看。消息中未分享推文的具体内容。

- **另一个 Modular 公告**：Modular 发布的第二条推文，详情可以通过提供的链接查看。聊天中未提及该公告的具体性质或主题。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1227199221531148320)** (32 messages🔥): 

- **Mojo 即将支持 SIMD**：讨论显示出对 Mojo 增加 SIMD 支持的兴奋，并期待更新后能看到令人惊叹的 Benchmark 结果。
- **并发特性正在开发中**：Mojo 目前支持 async/await 和 Coroutines（协程）；然而，这些功能尚未完成。Mojo 中的 Coroutine API 与 Python 不同，详情可以在 [Mojo 文档](https://docs.modular.com/mojo/stdlib/builtin/coroutine)中找到。
- **Mojo 的异步结构路线图**：该语言目前缺乏 `async for` 和 `async with` 结构，讨论链接到了路线图，表明 Mojo 正专注于核心系统编程功能，详见[此处](https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with)。
- **在 Intel Macs 上原生运行 Mojo**：一位用户表达了在 Intel Macs 上原生运行 Mojo 的局限性，大型项目依赖于 VM，尽管小型测试可以在 Playground 中完成。
- **Mojo-UI 的努力与 Objective-C 集成**：一个专门为 Mojo 打造的名为 Mojo-UI 的 UI 库新项目正在进行中，重点关注 Mac 作为主要平台，并提出了未来在 Mojo 中集成 Objective-C 或访问 AppKit 框架的潜力。正如最近讨论所建议的，这种集成可能需要通过 C 或 C++ 在 Mojo 和 Swift 之间设计一个绑定层。该项目在 [GitHub](https://github.com/Moosems/Mojo-UI) 上进行跟踪。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/coroutine">coroutine | Modular Docs</a>：实现 Coroutines 的类和方法。</li><li><a href="https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://github.com/Moosems/Mojo-UI">GitHub - Moosems/Mojo-UI: A cross-platform GUI library for Mojo</a>：一个为 Mojo 打造的跨平台 GUI 库。可以通过创建 GitHub 账号为 Moosems/Mojo-UI 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo">GitHub - modularml/mojo: The Mojo Programming Language</a>：Mojo 编程语言。可以通过创建 GitHub 账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2252">[BUG] Compiler bug when typing async function pointer call return type · Issue #2252 · modularml/mojo</a>：Bug 描述：Mojo 编译器在对异步函数指针调用返回类型进行类型推导时出错。预期行为：async fn() -&gt; Int 函数在调用时应返回 Coroutine[Int] 类型。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1227425197355700265)** (4 messages): 

- **MojoGeek 发布 Mojo GPT**：一名成员介绍了一个名为 **Mojo GPT** 的平台，专门用于回答 Mojo 编程查询，并寻求社区反馈。可以在 [Mojo GPT](https://chat.openai.com/g/g-RPORxvimH-mojogpt) 测试该平台并提供反馈。

- **提供字符串字符迭代器**：为那些需要字符串字符迭代器的用户分享了一个有用的转发帖子，链接指向 Discord 上的相关消息。

- **mojo-ui-html 迎来激动人心的更新**：**mojo-ui-html** 的新更新包括用于创建视频游戏或自定义组件的 *键盘事件*、新的窗口最小化功能，以及用于额外逐元素样式的 **CSS kwags**。详情和演示可在 [GitHub](https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo) 查看。

- **Lightbug 框架势头强劲**：强调了对 **Lightbug HTTP 框架** 的贡献，包括性能提升、纯 Mojo 实现的客户端，以及显示 Lightbug 每秒处理请求数超过 Python Flask 的对比。可以在 [GitHub](https://github.com/saviorand/lightbug_http) 上进一步探索这些进展和贡献。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo">mojo-ui-html/demo_keyboard_and_css.mojo at main · rd4com/mojo-ui-html</a>: Immediate mode GUI, HTML, CSS, 开发中, Mojo 语言 - rd4com/mojo-ui-html</li><li><a href="https://github.com/saviorand/lightbug_http/issues/6).">Issues · saviorand/lightbug_http</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1227402169809113120)** (1 条消息): 

- **Scrumtuous 获得 Google 搜索排名第一**: 一位成员幽默地宣布，由于使用了 **Mojo**，他们在某个高价值 **Python 关键词**的 Google 搜索结果中排名第一。目前没有提供关于该特定关键词或实现排名的内容的更多细节或链接。
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1227316880541356122)** (2 条消息): 

- **寻求 Mojo 中的 SYRK 实现**: 一位成员询问了 **Mojo** 中 **SYRK**（对称秩-k 更新）的实现，目的是进行一些性能测试。
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 29 期
https://www.modular.com/newsletters/modverse-weekly-29
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1227207975488454676)** (2 条消息): 

- **致敬 Prince**: 一位成员诙谐地建议，“Purple flame”这个短语可以激发一首让人想起 Prince 著名热门歌曲的灵感，并幽默地改编了歌词：“*Purple flame, purple flame...*”。

- **泛型冲击**: 另一位成员对提到 “Heterogeneous variadic generics” 表示震惊，对这一复杂的编程概念表达了惊讶和困惑交织的情绪。
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1227544935398506586)** (5 条消息): 

- **分享 Mixtral 模型转换脚本**: 一位成员分享了之前 **Mixtral** 模型的 MoE 权重转换脚本 ([convert_mistral_moe_weights_to_hf.py](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py))，以及在 Hugging Face GitHub 仓库中找到的新发布 **Mixtral** 的官方转换脚本 ([convert_mixtral_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py))。

- **Hugging Face 上的新 Mixtral 模型**: 一个更新的 **Mixtral-8x22B** 模型已上传至 Hugging Face，上传者提供了 [model card](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) 和转换脚本，随后被克隆到了一个官方社区仓库。

- **模型性能误解纠正**: 关于 *GPT-4*、*Claude Sonnet* 和 **Mixtral 模型**之间的**性能对比**有一处更正；原始陈述错误地提到了一个名为 *command-r+* 的不同模型，而非 **Mixtral**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py">convert_mistral_moe_weights_to_hf.py · DiscoResearch/mixtral-7b-8expert at main</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py">transformers/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py at main · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供最先进的机器学习模型。 - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1227492138422370344)** (18 条消息🔥):

- **Mixtral's Magnet Link Shared**: 分享了 **Mixtral 8x22b** 模型的磁力链接，提供了下载这一新 AI 模型的方式。
- **License Confirmation for Mixtral**: 确认 Mixtral 模型以 **Apache 2.0 license** 发布，预计随后将推出 instruct 版本。
- **First AGIEval Results Are Promising**: 一位成员强调了 **Mixtral 8x22b** 模型在 *First AGIEval Results* 中令人印象深刻的表现，表明其优于其他基础模型。
- **Benchmark Scores Released**: 分享了 PIQA、BoolQ 和 Hellaswag 等各种数据集的基准测试分数，对比了 **Mixtral 8x22B** 和 **Mixtral 8x7B** 模型的性能。
- **Model Runs on Virtual Large Language Model (vLLM)**: 指出基准测试分数是使用配置了 **4xH100 GPUs** 的 vLLM 环境生成的，并提到在该配置下 **MMLU 任务耗时约 10 小时**。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1777869263778291896?t=vKiT9FUuVbYAhjjg5kOHyw&s=33">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84">mistral-community/Mixtral-8x22B-v0.1 · MMLU - 77</a>: 未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1227194075162214486)** (24 条消息🔥): 

- **New LLM Merging Tool Unveiled**: 分享了一个名为 [**mergoo**](https://github.com/Leeroo-AI/mergoo) 的用于合并多个 Large Language Model (LLM) 专家模型的新库，该库声称可以简化合并过程并提高效率。据悉，该工具的灵感源自 3 月份的 branch train mix 论文。

- **RAG Benchmarking Reveals Odd Behavior**: [DiscoResearch/DiscoLM_German_7b_v1 模型](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval) 显示出截然不同的性能结果，这取决于 ChatML 模板中换行符的位置；如果没有换行符，在新建的 RAG 基准测试中的某些任务准确率会大幅下降。

- **Line Break Impact Investigated**: 发现换行符影响基准测试后，引发了关于潜在的数据加载/格式化脚本问题，以及这是否可能与更广泛的不稳定基准测试结果有关的讨论。这促使了一项审查训练数据应用的计划，并提到将为即将推出的 8x22 模型更新数据。

- **Model Formatting Issues Explored**: 围绕 **DiscoLM_German_7b_v1** 的 [tokenizer 配置](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48) 展开对话，推测修改 tokenizer 配置是否能解决性能异常。

- **Generalizability of Line Break Issue in Question**: 对换行符格式的独特敏感性引发了疑问：这究竟是 **DiscoResearch/LeoLM** 模型特有的问题，还是影响其他模型的更普遍现象。该话题仍有待进一步测试和调查。
<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48">tokenizer_config.json · DiscoResearch/DiscoLM_German_7b_v1 at main</a>: 未找到描述</li><li><a href="https://github.com/Leeroo-AI/mergoo">GitHub - Leeroo-AI/mergoo: A library for easily merging multiple LLM experts, and efficiently train the merged LLM.</a>: 一个用于轻松合并多个 LLM 专家并高效训练合并后的 LLM 的库。 - Leeroo-AI/mergoo</li><li><a href="https://github.com/Crystalcareai/BTX">GitHub - Crystalcareai/BTX</a>: 通过在 GitHub 上创建账户来为 Crystalcareai/BTX 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/lighteval/blob/main/community_tasks/german_rag_evals.py">lighteval/community_tasks/german_rag_evals.py at main · huggingface/lighteval</a>: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5ed29393e34cf57b24a20ac1bafa3a94272ac3f5/src/axolotl/prompt_strategies/dpo/chatml.py#L86">axolotl/src/axolotl/prompt_strategies/dpo/chatml.py at 5ed29393e34cf57b24a20ac1bafa3a94272ac3f5 · OpenAccess-AI-Collective/axolotl</a>: 尽管去向 axolotl 提问吧。通过在 GitHub 上创建账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1227332519016530031)** (16 messages🔥): 

- **带着推文问早**：一位成员在频道中以 "gm" 打招呼，并分享了一个 [Twitter 链接](https://twitter.com/OpenAI/status/1777772582680301665)，可能与 OpenAI 的新更新或讨论有关。
- **令人惊讶的 Benchmark 结果**：Wenquai 报告了意外发现，在快速 vision benchmark 中，**Sonnet 和 Haiku** 的表现优于 **GPT-4 Turbo 和 Opus**，并提供了 [Colab 研究文档](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing) 链接供审阅。
- **探索 GPT-4 Turbo 特性**：**GPT-4 Turbo** 的 function calling 和 JSON mode 被强调为构建 vision 模型的有力工具，引发了对这些功能进行进一步 benchmarking 的兴趣。
- **这究竟是不是 GPT-4.5？**：成员们开玩笑说最新模型的改进是渐进式的，有人称其更像是 **4.25** 更新，而其他人则引用了 OpenAI 员工关于增强推理能力的说法。
- **AI Coding 能力对比**：简短交流了最新模型的 coding 能力，potrock 提到在 cursor 中使用该模型没有 coding 问题，而其他人则将其与 **Gemini 1.5** 进行了比较，并讨论了 **copilot++** 的好处。

**提到的链接**：<a href="https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing">Google Colaboratory</a>: 未找到描述

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1227189540457021541)** (15 messages🔥): 

- **LLM Help 命令性能**：一位用户报告 `llm --help` 命令运行缓慢，耗时超过 2 秒。他们担心这是否预示着潜在的安全问题，比如被黑客入侵。
- **Benchmarking LLM Help**：针对 `llm --help` 性能的担忧，另一位用户分享了一个快速的 benchmark 结果：`0,50s user 0,10s system 94% cpu 0,624 total`。
- **在不同环境下测试 LLM 耗时**：原始用户的后续反馈显示，`llm --help` 在其设置上耗时 3.423 秒，但在全新的 docker 容器中仅需 0.800 秒，这表明变慢可能与系统配置有关，而非 `llm` 工具本身。
- **重新安装解决问题**：遇到 `llm --help` 性能问题的用户发现，重新安装 `llm` 解决了速度问题以及运行 Claude 模型时遇到的错误，这表明全新安装可以缓解某些运行问题。
- **LLM 命令在 MacOS 上出现小故障**：另一位用户在 macOS 上使用 iTerm2 本地运行 `llm cmd` 命令时遇到挂起，但在远程 Ubuntu 服务器上运行正常。他们注意到自定义的 shell 环境，怀疑这可能是原因，尽管同样的配置在 Ubuntu 上可以正常工作。
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1227237932872630275)** (3 messages): 

_

- **寻求基准测试对比**：一位成员询问了一篇引用了 **phi-2**、**dolphin** 和 **zephyr** 等模型在 **HumanEval 数据集**上性能基准测试的论文。

- **对基准测试的怀疑**：一位成员对基准测试表示怀疑，认为它们可以被操纵。不过，他们推荐了一个人类排名的排行榜以获得值得信赖的结果，可在 [arena.lmsys.org](https://arena.lmsys.org/) 查看。

- **Mistral 8x22b 的首个 AGIEval 结果**：**Mistral 8x22b** 模型的首个 **AGIEval 结果**已发布，显示其性能优于其他开源（基础）模型。更新内容可以在 Jan P. Harries 的两条推文中找到，详见[此处](https://x.com/jphme/status/1778030213881909451)和[此处](https://x.com/jphme/status/1778028110954295486)。

**提到的链接**：<a href="https://x.com/jphme/status/1778030213881909451">Jan P. Harries (@jphme) 的推文</a>：@MistralAI 的首个 AGIEval 结果看起来很棒 👇 - 感谢发布这个猛兽，伙计们！👏 https://x.com/jphme/status/1778028110954295486 ↘️ 引用 Jan P. Harries (@jphme) 的话：首个 AGIEval 结果...

  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=Gb--4supXoo
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1227430928259743774)** (4 条消息): 

- **优化 GPU 使用**：一位成员发现使用**较低的 `-ngl` 值**（要使用的 GPU 层数）解决了他们的问题，并最终确定为 `-ngl 3`。他们指出，由于 GPU 显存有限，较小模型的性能明显更好。

- **关于自适应层卸载的疑问**：在 VRAM 限制的背景下，一位成员询问 **llamafile** 是否可以潜在地卸载层以适应用户的可用 VRAM 而不是崩溃，并链接了他们自己在 1050 GPU 上的配置。

- **ollama 提供 LLM 灵活性**：一位成员赞扬了 **ollama** 处理模型层分布的方法，并分享了一个讨论实现细节的特定 GitHub 链接：[ollama server.go](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43)。

**提到的链接**：<a href="https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>">ollama/llm/server.go (位于 c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9) · ollama/ollama</a>：快速上手 Llama 2、Mistral、Gemma 和其他大语言模型。- ollama/ollama

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1227303320171974676)** (2 条消息): 

- **关注 Remix Music AI**：一位成员分享了对一个混音音乐模型的兴奋之情，称其“非常惊人”，并附上了试听链接：[正在加载歌曲...](https://linktones.synthtrails.com/linktone/kanye)。
- **寻求代码支持**：一位用户请求通过私信协助处理他们的代码，并向一位特定成员寻求帮助。

**提到的链接**：<a href="https://linktones.synthtrails.com/linktone/kanye">SynthTrails</a>：未找到描述

  

---