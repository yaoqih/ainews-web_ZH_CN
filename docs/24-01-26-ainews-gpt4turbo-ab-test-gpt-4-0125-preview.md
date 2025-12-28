---
companies:
- openai
- thebloke
- nous-research
- hugging-face
date: '2024-01-26T22:48:31.081164Z'
description: '**OpenAI** 在 2024 年 1 月发布了新的 **GPT-4 Turbo** 版本，引发了关于摘要生成的自然实验，以及针对
  API 性能和成本权衡的讨论。**TheBloke** 的 Discord 频道重点介绍了 **UnSloth** 即将为 Google Colab 初学者提供的有限多
  GPU 支持、在任天堂 Switch 上运行的 **Tiny Llama** 和 **Mistral** 等 AI 模型，以及 **DARE** 和 **SLERP**
  等先进的模型合并技术。**OpenAI** 的 Discord 频道则指出了 **GPT-4-1106-preview** 的处理延迟问题、GPT 模型错误的排查，以及
  **GPT-3.5** 和 **GPT-4 Turbo** 在转录方面遇到的挑战。**Nous Research AI** 专注于扩展上下文窗口，特别是 **LLaMA-2-7B-Chat**
  达到了 **16,384** 个 token，并探讨了如 **SelfExtend** 等微调替代方案。此外，讨论还涉及了聊天机器人人格设定、模型配置优化以及
  AI 技术对社会的影响。'
id: 953e5a59-b782-442b-a642-7d27de68d9a4
models:
- gpt-4-turbo
- gpt-4-1106-preview
- gpt-3.5
- llama-2-7b-chat
- tiny-llama
- mistral
original_slug: ainews-gpt4turbo-ab-test-gpt-4-0125-preview
people: []
title: GPT4Turbo A/B 测试：gpt-4-0125-preview
topics:
- multi-gpu-support
- model-optimization
- model-merging
- fine-tuning
- context-windows
- chatbot-personas
- api-performance
- text-transcription
- cost-considerations
- model-troubleshooting
---

<!-- buttondown-editor-mode: plaintext -->> 2024年1月25日的 AI Discord 动态。我们为你检查了 **20** 个公会、**297** 个频道和 **5875** 条消息。预计节省阅读时间（按 200wpm 计算）：**555 分钟**。

OpenAI 昨天发布了一个 [新的 GPT4 Turbo 版本](https://openai.com/blog/new-embedding-models-and-api-updates?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gpt4turbo-ab-test-gpt-4-1106-preview) ([我们的笔记见此](https://twitter.com/swyx/status/1750620187114787243?utm_source=ainews&utm_medium=email&utm_campaign=ainews-gpt4turbo-ab-test-gpt-4-1106-preview))。我们正利用这个机会进行一次关于摘要生成的自然实验。此版本是由 2024 年 1 月的“新” GPT4T 生成的，请参阅之前包含 2023 年 11 月版本的邮件进行对比。

---

**目录**

[TOC] 


# 第一部分：高层级 Discord 摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **UnSloth 迈向多 GPU 支持**：UnSloth 正准备引入*有限的多 GPU 支持*，特别针对 Google Colab 初学者。由于其承诺的简便性，这一举动引起了 AI 研究新手的兴趣。

- **AI 模型跃上 Nintendo Switch**：社区里兴奋不已，`@kalomaze` 展示了在 Nintendo Switch 上运行 Tiny Llama 和 Mistral 等 AI 模型，引发了关于在非常规硬件上部署轻量级模型的讨论。

- **深入探讨模型配置**：详细检查了 `exllama`、`bagel` 和 `dolphin` 等模型，特别是针对 `rope_theta` 和 `sliding_window` 的使用。特别强调了来自 Hugging Face 的配置，为模型优化提供了宝贵的见解。

- **聊天机器人获得性格与风格**：积极讨论了构建聊天机器人人格（personas）的技术，包括使用 ChatGPT 生成的数据集以及训练中示例数量对风格迁移的重要性。对话强调了创建具有 Samantha 等独特个性的聊天机器人，并探索了如本地 LLM 等具有成本效益的训练替代方案。

- **探索创新的模型合并技术**：讨论涉及了维持模型性能的最佳权重以及使用 DARE 或 SLERP 等技术合并过拟合模型的趣味性。对话表明，选择性地组合模型以发挥多种优势是一个复杂但充满希望的前沿领域。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **GPT-4 速度：龟兔赛跑的故事**：用户注意到 **GPT-4-1106-preview** 在通过 API 进行文档分析时处理时间显著变长，出现了 3-7 分钟的延迟。讨论指出潜在的解决方法包括升级服务计划以获得专用容量，尽管已澄清 **GPT-4 Turbo** 本身并不直接支持文件分析。

- **GPT 模型故障排除集思广益**：GPT-4 标记的“异常活动”错误引发了猜测，认为 VPN 使用、提示词性质或账号共享可能是原因；同时 Dall-E 因生成的图像中频繁出现拼写错误而受到批评，建议缩短文本输入可能会缓解此问题。

- **GPT-4 让代码块焕然一新**：讨论了“始终展开代码输出”设置以增强 GPT-4 中的代码可读性，此外还讨论了尽管存在安全顾虑，仍将 Python 模块导入 AI 系统的问题，以及需要开启新对话以反映 CustomGPT 编辑内容的需求。

- **文本转录应对 GPT 的巨量挑战**：核心挑战在于大型文本转录，特别是高达 50KB 的布道词段落，**GPT-3.5** 因输入大小限制而难以应对。建议倾向于利用 **GPT-4 Turbo** 的大上下文窗口能力，并探索 NLP 工具以实现更高效的分段和语法错误修正。

- **成本考量与功能需求之间的冲突**：对话强调了在追求先进功能与成本影响之间不断寻找平衡，特别是在从 **GPT-3.5** 转向 **GPT-4 Turbo** 以处理需要密集文本处理和分析的任务时。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **上下文能力扩展，Mistral 挖掘更深**：扩展模型上下文能力的创新主导了讨论，**Fine-tuning** 等方法被强调为增强模型的有效途径。特别是 **LLaMA-2-7B-Chat** 成功将上下文窗口扩展至 **16,384**，令人惊叹；同时 **SelfExtend** 被推荐为一种免 **Fine-tune** 的方案。

- **思考技术的社会二分法**：对话偶尔会偏离技术路径，触及技术进步的更广泛影响，有观点认为这可能会进一步使社会两极分化。幽默的插曲包括一个**游泳猫的 GIF**，同时也有人担心 Twitter 的内容分发机制可能会限制 AI 相关的贡献。

- **Everyone Coder 33B 基准设定**：焦点聚集在 **Everyone Coder 33B Base - GGUF** 上，该模型使用 Massed Compute 进行了 **GGUF** 格式的量化。此外，还有关于 **Hermes Mixtral** 性能对比的推测，尽管目前缺乏详细的支撑数据。

- **从 Embedding 模型到 Content-genie**：分享了 OpenAI 最新的 **Embedding** 模型和 API 增强功能，展示了更低的 **GPT-3.5 Turbo** 定价和新工具。同时，用于在内容落地任务中生成高质量数据的 **Genie** 方法引起了关注，暗示了在 **LFQA**、摘要和提取能力方面的重大进展。

- **技术巨头与天才干预**：激烈的讨论包括在 WebGL 上进行 **GPT-2** 推理实验、使用 **Mixtral** 模型克隆 **LLaMA** 的挑战，以及关注 **phi2** 优化的**模型效率**辩论。**利用 GPU 进行 ML** 成为一项创意追求，重新构想了将游戏硬件用于科学探索。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **挪威风格的 Fine-Tuning**：`@henriklied` 正在 10 万篇文章的数据集上 **Fine-tuning** **Mistral**，用于挪威语标题生成，并建议将 **Epochs** 限制在 4 以防止过拟合。同时，**shareGPT** 在处理长对话时的局限性引发了关于对话管理中 **ChatML** 与 **BlockML** 格式偏好的辩论。

- **模型合并，QLoRA 兴起**：探索了不依赖 **Bitsandbytes** 训练 **QLoRA** 模型的方法，并重点介绍了 **fp16** 和 **AutoAWQ** 等量化替代方案。分享了一个将训练好的 **QLoRA** 模型合并到 **Base** 模型的 [GitHub 指南](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base)链接，以及 Tim Dettmers 关于量化倡导的推文。

- **数据集进展引发关注**：[Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset) 上旨在训练 Snorkel 模型的新数据集以及 **Mistral 7** **Fine-tuning** 的公告表明了重大进展。**ALPCA** 相比旧的 **GPT-4** 指标提升了 34%，暗示了显著的性能增强。

- **DPO 深度讨论**：围绕 **DPO Training Plots** 和数据集复合问题的咨询，暗示社区在 **DPO Training** 和数据集完整性领域需要更清晰的文档和故障排除指南。

- **社区展示**：`pradeep1148` 在 **community-showcase** 频道分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=wlPxEq_Mtkc)，展示了社区参与和项目分享，尽管未列出具体细节。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **代理方案和 TTS 使得 LM Studio 的使用更加顺畅**：由于 HuggingFace 受到区域限制，用户通过[代理设置和文本转语音接口](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html)找到了变通方法，增强了模型的可访问性和交互性。

- **系统更新修复模型兼容性问题**：根据 `@heyitsyorkie` 的建议，在遇到 **Stable Code**、**Deepseek** 和 **Codellama** 等模型加载困难后，更新 C++ redistributables 可以解决 LM Studio 中的模型加载错误。

- **模型运行查询涵盖了多功能性和性能**：从运行多个 LM Studio 实例以进行并行建模，到研究用于 AI 工作的最佳 GPU 选项，讨论显示出对提高模型性能和效率的浓厚兴趣。**RTX 3090** 和 **M2 Mac Studio** 是推荐用于处理大型语言模型的硬件。

- **Bug 报告推动了 MoE 模型的改进**：用户报告了影响 LM Studio 中 MoE 模型设置的 Bug，特别是在 4X 和 2X MoE 模型之间切换时，这引发了开发团队的立即调查，以提升用户体验和模型功能。

- **强调了模型探索和集成的挑战**：使用 RTX 3090 对 **mixtral8x7B** 等大模型进行的实验，以及对过时 API 的不满，突显了社区在克服集成和微调障碍的同时，努力突破 AI 模型在项目中实用性和应用边界的尝试。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **用于 AI 工作的 GPU 租赁和免费资源**：@mrdragonfox 强调了 **runpod、vast、lambda** 和 **Kaggle** 是提供按小时计费的 GPU 租赁和免费资源的关键平台，以促进 AI 摘要和开发工作。Kaggle 每周提供 30 小时的 2 个 T4 GPU，这对于寻求计算能力而无需大量投资的开发者来说是一个福音。

- **超越传统指标评估 LLM**：@adrienbufort 批评 **BLEU 和 Rouge** 等传统翻译指标不足以评估大型语言模型 (LLM)，主张将 **[类似 ELO 的评估系统](https://arena.lmsys.org/)** 与 **MMLU** 和 **[Alpaca Eval](https://github.com/tatsu-lab/alpaca_eval)** 作为更优的方法论。这些方法分别提供了更接近人类偏好的评估和 LLM 内部评估能力。

- **AI 浏览器交互的创新**：@sublimatorniq 介绍了一种开创性的工具，使 AI 能够引用 **DOM node references**，通过使浏览器查询更具上下文感知能力来增强 Web 内容交互。该工具旨在与 MistralAI 兼容，标志着 AI 辅助 Web 导航的重大飞跃。

- **AI 模型的量化和优化策略**：社区讨论揭示了对 Mistral 4bit 推理的内存需求和效率的认真考量，指出 26GB 的内存占用是必不可少的。@mrdragonfox 推荐使用 **exllamav2**，因为它比传统的 4-bit transformers 具有更优的内存效率，建议将其作为优化 AI 模型性能的有效策略。

- **披露了 API 限制、Bug 和托管见解**：社区遇到了各种问题，从“提前停止困境”到 **"max_tokens" 参数的 Bug**（当设置为 1 时会导致 500 错误），具体的 GitHub issue 列在[此处](https://github.com/mistralai/mistral-src/issues/122)。此外，关于 **Mistral API 托管位置** 的相关查询显示其位于 **Azure 瑞典区域的欧洲节点**，这对于考虑数据本地化和合规性的开发者至关重要。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **AutoGluon 介入 Meta 的 SAM**：在未能找到 Meta 的 SAM 模型的微调代码库后，`@the_alt_man` 转向使用 **AutoGluon**，因为它支持 Lightning 且兼容 GPU，尽管不支持 TPU。

- **超越 Infiniband 的多节点训练创新**：受 `@elyxlz` 对无 Infiniband 多节点训练探索的启发，讨论围绕训练步骤中的定期合并策略展开，并参考了关于分布式优化的 [DiLoCo 论文](https://arxiv.org/abs/2311.08105)。

- **深入探讨字节级 Transformer 和 Proxy-Tuning**：讨论范围从 ByT5 等字节级 Transformer 的效率以及 **MambaByte 论文**中不公平的序列长度比较，到分享一种针对 LLM 的新型轻量级 **proxy-tuning** 方法，详见[最新发表的论文](https://arxiv.org/abs/2401.08565)。这体现了人们对优化和理解 LLM 机制及效率日益增长的兴趣。

- **国际象棋 AI 进化的设想**：结合了推测和当前最前沿的评论，重点讨论了国际象棋 AI，包括 Stockfish 的统治地位，以及对 2024 年国际象棋 AI 进展的理论反思，如 [chess.com 博客](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024)中所设想的那样。

- **GPT-NeoX, PyTorch 和 CUDA 在测试中遇到的困难**：`@catboy_slim_` 提出了在更新 Python、PyTorch 和 CUDA 版本后遇到的测试挑战，并在 [GitHub 上创建了一个 issue](https://github.com/EleutherAI/gpt-neox/issues/1132) 关于 pytest 失败的问题。同时，澄清了 **neoX 20b** 中 QLoRA 微调的困惑，强调了针对特定库的问题应导向正确渠道的重要性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Perplexity Pro 发布强大功能**：用户讨论了 **Perplexity Pro** 相比普通版本的优势，强调了无限次 Copilot 查询以及在使用 **Claude 2.1** 和 **GPT-4** 等模型时附加图像和文件的能力。详细信息由 [Perplexity Pro FAQ](https://blog.perplexity.ai/faq/what-is-perplexity-pro) 链接提供支持。

- **数据保留担忧中的隐私保护查询**：`@emisaurus_hex` 和 `@firesonwires` 等用户就 Perplexity 关于保留搜索历史和个人信息直至账号注销的隐私政策展开了讨论，促使专家澄清已删除的线程将在 30 天后从服务器中消失。

- **关于数据存储伦理的辩论**：用户参与了关于聚合搜索数据对隐私影响的哲学讨论，对此类数据的价值和潜在滥用持有不同意见。

- **社区共同解决技术支持问题**：社区（包括 Perplexity 代表）共同协助用户解决技术问题，从恢复书签线程到查询文件上传限制。

- **探索 Perplexity 的教育和专业应用**：在 #sharing 频道中，用户分享了在内容创作、Smartsheet 学习和天文学教育等不同领域使用 Perplexity 的经验，其中一位用户在 [YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)中强调了 Perplexity 在内容生成方面优于 Google Search 和 OpenAI。

- **Perplexity 网站与 API 之间的技术差异**：#pplx-api 频道讨论了工具的网站版本和 API 版本之间的性能差异，一位用户寻求关于 labs 与 API 参数的澄清，另一位用户报告了 **support@perplexity.ai** 尚未解决的重复收费问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **HuggingFace 社区的扩张与创新**：随着更多创作者的加入，社区正在经历增长，亮点包括 **[Lightweight Open LLM Leaderboard](https://thenameless.net/cosmos-arena)** 的发布，以及 H. Luo 和 L. Specia 在一项调查中关于 **AI interpretability** 的讨论。创新的 **AI Alignment 模型 TenyxChat-8x7B-v1** 也已推出，展示了具有高 MT-Bench 分数的 preference tuning。**CheXRay** 和 **ZeroGPU** 等实用项目也备受瞩目，体现了 HuggingFace 社区内的多样化探索。

- **现实世界的 AI 讨论聚焦于实用性**：社区成员分享了从为 RTX 3080 GPU 选择模型，到特征提取技术以及模型 pretraining 现状的见解。对话还涉及了用于训练和评估的数据管理要点，以及利用社区博客文章扩大项目影响力的好处，例如即将发布的 **wav2vec2-bert** 模型。

- **数据集评估框架与数据质量**：社区提出了对用于训练 LLM 的数据集进行全面评估框架的需求，考虑到目前重点主要在于模型而非数据集评估，这表明该领域已具备开发潜力。

- **Flutter 和 OpenAI 的创新引领酷炫发现**：对 HuggingFace 数据集文档分析的关注，以及一项名为 **"Binoculars"** 的用于检测 LLM 生成文本的详细研究，都是引人入胜的发现。**ONNX runtime** 与 **Flutter** 的集成工作，以及针对 **HuggingFace Inference API** 的特定 **Flutter SDK**，突显了社区在弥合 AI 与应用开发之间鸿沟方面的努力。

- **WhisperSpeech 与 AI 开发亮点**：**WhisperSpeech** 空间标志着多语言 TTS 的飞跃，而 **CheXRay** 演示中的一个错误则指出了在医学影像中部署 AI 的挑战。人们对 **Nemo Model** 博客文章充满期待，并呼吁通过社区贡献的内容进行更广泛的参与。

- **人工智能与机器学习突破**：Google 的 Lumiere 展示了 T2V 模型中的先进方法，而一篇关于 benchmarking 的 Medium 文章受到了称赞。在 NLP 领域，来自 Coqui 的 **TorToiSe TTS** 展示了质量和潜在的速度提升。

- **来自 #diffusion-discussions 和 #computer-vision 的技术见解**：关于在本地加载 LoRA 模型的问题以及对计算机视觉技术的询问，强调了 HuggingFace 社区讨论的技术深度。

- **Gradio 4.16 发布提升开发者体验**：最新的 **Gradio release** 倡导旨在简化 ML 应用开发的功能，例如对 Polars Dataframe 的原生支持，增强了 ML 部署中的交互性和性能。

每个要点都概括了跨越各个 HuggingFace 频道的专题讨论和公告，反映了该组织对尖端 AI 技术和社区驱动计划的活跃参与。

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **LAION2b-en 美学评分数据集下架**：`@ppwwyyxx` 寻找的 [LAION2b-en aesthetics scores dataset](https://huggingface.co/datasets/laion/laion2B-en-aesthetic/tree/main) 目前在 Hugging Face 上无法访问，因为应作者要求已被禁用，社区被鼓励关注后续更新。

- **开源语音聊天创新的一大步**：`@jpcl_` 分享了一个新的语音聊天界面演示，该演示采用了 Whisper 和 WhisperSpeech 以及开源 LLM (Dolphin 2.6 Phi-2)，承诺更低的延迟和更自然的对话流。邀请感兴趣的各方合作进一步增强该项目，详情可在 [Hacker News](https://news.ycombinator.com/item?id=39130140) 查看。

- **VA 的 AI 技术冲刺赛征集参赛者**：VA 的 AI 技术冲刺赛专注于临床就诊记录的 Ambient Dictation（环境听写），一等奖奖金为 30 万美元，旨在推动 AI 驱动的医疗保健进步。鼓励美国居民参与，更多信息见 [Challenge.Gov](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)。

- **字节级 Transformer 展现出广阔前景**：字节级 Transformer 可能预示着 AI 能力的新前沿，这是 `@marianbasti` 在审阅了一篇相关的 [研究论文](https://arxiv.org/pdf/2401.13660.pdf) 后提出的乐观且谨慎的见解。

- **AI 图像生成的创新进展**：突破性项目如 [RPG-DiffusionMaster](https://github.com/YangLing0818/RPG-DiffusionMaster) 和 [InstantID](https://github.com/InstantID/InstantID) 为文本到图像的 Diffusion 和保持 ID 的生成设定了新标准，展示了该领域快速发展的步伐。此外，讨论强调了 AI 图像生成增长潜力并未减弱，并提到了该领域不断涌现的新论文和工具。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **LLMCompiler 网络研讨会预示并行函数调用革命**：由作者 Sehoon Kim 和 Amir Gholami 主持的最后时刻网络研讨会讨论了 **LLMCompiler**，旨在强调其优于 ReAct 等顺序框架的优势，重点关注 **长期规划** 和 **并行化能力**。相关资源包括 [LLMCompiler paper](https://arxiv.org/pdf/2312.04511.pdf)、[LlamaPack](https://llamahub.ai/l/llama_packs-agents-llm_compiler?from=llama_packs) 和 [GitHub notebook](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_pac)。

- **LlamaIndex 的 OSS 和合作伙伴关系扩展了其生态系统**：LlamaIndex 推出了一个 **开源软件 (OSS) 仓库**，其中包含构建 Slack 机器人的指南；与 **Zilliz Universe** 合作，通过可扩展的检索服务增强其平台；并宣布对 OpenAI 最新的 Embedding 模型提供 **Day 0 支持**。其他产品包括 **effective prompting** 指南和新的 TypeScript 版本 (**LlamaIndex.TS version 0.1.0**)，提供易用性和广泛的支持能力。

- **动态讨论塑造了 LlamaIndex 社区**：社区成员探讨了 LlamaIndex 中 **TextGenerationInference LLM** 的可用性，探索了使用 `similarity_top_k` 参数自定义 Chat Engine，解决了保险领域查询实施中的挑战，并对 OpenAI 的新 Embedding 模型和 API 更新表示热切关注。关于在查询中扩展上下文的自定义 Prompt 指导体现了社区在优化 LlamaIndex 易用性方面的积极参与。

- **同行间的前沿 AI 讨论蓬勃发展**：Zep 成为增强聊天机器人 **生产级能力和实体提取** 的值得关注的工具，而 LlamaIndex 被定位为可与 Amazon Kendra 媲美的灵活数据编排工具。一篇 Substack 文章展示了一个利用 **递归检索、自动知识图谱创建** 以及 **内存/多跳推理** 的 **自学习 RAG**，突破了 AI 应用的边界。

- **AI 应用中的持续集成与学习**：LlamaIndex 公会内资源、查询和解决方案的分享凸显了一个致力于持续学习、集成前沿工具和探索高级 AI 技术的活力社区。这些讨论揭示了在机器人、数据检索和自学习知识图谱等各个领域充分发挥 LlamaIndex 和 Zep 等 AI 技术潜力的共同努力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **MORPHEUS-1 宏大愿景**：Prophetic AI 发布了 **MORPHEUS-1**，这是首个用于诱导清醒梦的多模态生成式超声 Transformer，目标是在 2024 年春季发布 Beta 版。该发布的 [tweet](https://x.com/propheticai/status/1750534355242418300?s=46&t=JE84TqLviekDnEt8MAT-Eg) 承诺将带来突破性的功能。
- **快速 YAML 自定义标签实验**：go-go-golems 展示了一个雄心勃勃的项目里程碑，作为 yaml-custom-tags 实验的一部分，在短短 4 天内完成了 5000 行代码。他们的进展可以在 [GitHub](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/experiments/yaml-custom-tags) 上追踪。
- **Martian 发布 LLM 评估工具**：Martian 推出了一款工具，可根据成本、吞吐量和首字节时间 (TTFT) 动态评估在 Anyscale 和 Fireworks 等供应商中应使用哪种 **LLM 推理产品**。发布详情见[此处](https://leaderboard.withmartian.com/)，并附有[评估方法论](https://docs.withmartian.com/provider-leaderboard/evaluation-methodology)。
- **LLM 论文俱乐部亚洲版启动**：LLM 论文俱乐部扩展至亚洲，首场活动将讨论开创性论文 *"Attention Is All You Need"*。欢迎感兴趣的人士[报名](https://lu.ma/llm-paper-asia)并通过特定的 [Discord 链接](https://discord.gg/tPnG5qMu)参与。
- **美国俱乐部下一期议程：Pythia 论文**：美国论文俱乐部的下一次讨论将聚焦于 **Pythia 论文**，该论文探讨了在整个训练和缩放过程中分析 Large Language Models (LLMs) 的指标。论文及作者详见此[链接](https://arxiv.org/abs/2304.01373)。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **Mixtral 微调深度探讨**：在 [mergekit 的 GitHub issue](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289) 中，`@philipmay` 发起了关于 **Mixtral** 模型微调潜力的讨论，探索了“隐藏”和“随机”选项。对话强调了 **MoE 训练** 中辅助损失（auxiliary loss）对有效性的至关重要性。

- **重新思考 AI 模型的数据质量**：社区中强调的一项新[研究](https://arxiv.org/abs/2401.12926)反对传统的数据质量过滤方法，建议采用模型感知（model-aware）的数据选择方法可能会带来更好的性能。

- **探索 Kahneman-Tversky 优化 (KTO)**：社区讨论了 **KTO**，并将其与 Direct Preference Optimization 进行了比较。这种方法通过简化的二进制正负训练示例，被评估其在上下文 AI 和生产模型更新中的实现可行性，[Hugging Face 的文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)等资源可作为采用指南。

- **OpenAI 战略模型与价格更新**：分享了关于 **OpenAI 发布 GPT-4 Turbo** 以及 **GPT-3.5 Turbo** 降价的重要更新，详见[官方推文](https://x.com/officiallogank/status/1750589278709780780?s=46&t=1jtkL4JPu-DUOdo8JC668g)。这些变化标志着 Embedding 选项和定价方面的战略转变，预示着 AI 模型可访问性格局的演变。

- **Embedding 创新与德国 AI 模型增强**：社区讨论了即将发布的用于排名任务的新德国 Jina 模型，并分享了使用 **Mixtral** 进行高级问题生成和 Embedding 开发的见解。亮点包括 [OpenAI 的新 Embedding 模型和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)，暗示了在多语言支持方面的能力扩展。此外，一份分享的[论文](https://arxiv.org/abs/2401.14367)中提到的 Genie 方法承诺提供一种生成高质量数据的新途径，可能推动更有效的问答对或摘要的创建。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **OpenAI 发布全新 Embedding 模型和 API 工具进行重塑**：OpenAI 推出了新一波 [Embedding 模型、更新的 GPT-4 Turbo](https://openai.com/blog/new-embedding-models-and-api-updates) 以及审核工具，旨在为开发者提高性能和成本效率。这套工具包括两个新的 Embedding 模型、GPT-4 Turbo 的增强功能、一个新的文本审核模型，以及即将下调价格的 GPT-3.5 Turbo，并明确表态不会使用客户数据进行模型训练。
  
- **社区对新 Embedding 功能反响热烈**：新 Embedding 模型的引入和 GPT-4 Turbo 的更新引发了社区成员的热议，特别是针对缩短的 Embedding（abbreviated embeddings）特性表现出极大的兴趣。此外，将新的 large Embedding 模型与 bge-large 进行对比后发现，前者在性能上略胜一筹，凸显了 OpenAI 的进步。

- **OpenAI 更新引发升级讨论**：随着 OpenAI 发布最新产品，社区成员正在考虑升级系统以利用新的 Embedding 模型。讨论内容包括由于维度缩减，使用更新、更大的 Embedding 模型可能带来的成本节约，并对比了 Embedding 费用与向量数据库存储成本的节省。

- **效率与成本影响讨论**：对话深入探讨了 OpenAI 最新 Embedding 模型的效率和潜在成本效益，权衡了性能与存储及 API 成本。讨论重点倾向于 v3-large 模型，因为它具有更高的性能和降低存储成本的潜力，反映了人们对最大化运营效率和成本效益的广泛关注。

- **公告发布位置不当引起小范围关注**：值得注意的一点是，有人建议将关于新 Embedding 模型和 API 更新的讨论重新定向到更合适的频道，这在社区内部沟通流程中产生了一个小插曲。这指出了关于频道相关性的程序性说明，并强调了在社区平台中针对正确受众的重要性。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain 助力项目和个性化 AI**：`@quarknova`（一名 ENS 学生）正在探索将 [LangChain 用于项目](https://github.com/)，并提出了关于 GitHub 版本与商业版本的问题。同时，创建像“Elon Musk AI”这样的 AI 个性涉及 **Finetuning**，推荐通过[短课](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)获取实践见解。

- **快速聊天机器人和数据生成的发展**：用户庆祝使用 LangChain 和 Streamlit 创建网页搜索聊天机器人的速度和便捷性，同时也讨论了 LLM 在为机器学习训练和 RAG 生成合成数据中的作用。对于 PARQUET 文件爱好者，解决方案涉及使用 pandas 和 `DataFrameLoader` 集成到 LangChain 中。

- **深入探索 LangServe**：LangServe 爱好者被引导至 GitHub 上的示例，包括构建自定义工具或 Agent 的技巧，重点是使用 LCEL 和 LangGraph 来增加表达力，详见[此处](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples)和[此处](https://python.langchain.com/docs/langgraph#agentexecutor.)。此外，Agent 实现中的流式响应（stream response）故障排除也成为了持续讨论的热点。

- **探索上下文感知和 SQL 集成**：`dejoma` 对 LangChain AI 如何使用网页上下文表示好奇，显示出对 AI 上下文感知能力的兴趣。`johnny2x2` 分享了在制造业中使用 LangChain 进行 LLM 自动化的突破，利用 ERP SQL 数据分析延迟的客户订单，并实施了 **Mixtral 7B v2 5Q 32K** 以实现有效的数据库管理和专门的任务循环。



---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **重大 LLM 升级即将到来**：@simonw 宣布即将发布 **LLM 更新**，该更新承诺对 openai 库进行实质性改进，并在 [GitHub](https://github.com/simonw/llm/issues/325#issuecomment-1911533536) 上列出了详细的测试说明。
- **LLM 未来发展预览**：为了详细展示未来规划，@simonw 通过 [GitHub 上的 0.13 里程碑](https://github.com/simonw/llm/milestone/8) 分享了见解，重点介绍了计划中对大语言模型命令行访问的增强功能。
- **社区呼吁解决 Readline Bug**：@simonw 正在寻求社区支持，以修复 LLM chat 中的一个 readline bug，该 bug 会导致方向键产生 ANSI 编码而非进行文本导航，更多信息请参见 [GitHub](https://github.com/simonw/llm/issues/376)。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **Skunkworks 内部引发好奇**：Arielnlee 询问是否正在进行 **bakklava-2** 的相关工作，这在 [bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) 频道中引发了关于未来发展的讨论。
- **发布神秘链接**：Pradeep1148 在 #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) 频道分享了一个神秘的 [YouTube 视频](https://www.youtube.com/watch?v=wlPxEq_Mtxc)，内容和背景未注明。

---

# 第 2 部分：各频道详细摘要及链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199986879559376906) (1212 条消息🔥🔥🔥): 

- **UnSloth 迈向多 GPU 支持**：UnSloth 的开发正被热烈讨论，有迹象表明几个月后可能会引入*有限的多 GPU 支持*。其 OSS 版本主要针对 Google Colab 初学者，引发了关于其对 AI 研究新手易用性的讨论。

- **AI 开发的 Discord 机器人与微调闲聊**：包括 `@flail_.` 在内的成员讨论了他们使用 Python 制作 Discord 机器人的经历，尽管个人偏好其他语言，但仍承认 Python 库的便利性。此外，还有一场关于通过微调语言模型可以获得何种专业资质的轻松辩论。

- **Academicat 在研究中的应用及其伦理趣闻**：`@kaltcit` 分享了在生物医学研究中使用小鼠手指进行 PCR 的见解，并幽默地提到截肢后还能开玩笑的能力。这引发了关于研究方法论和动物使用的伦理及科学讨论。

- **对 Open LLM Leaderboard 的担忧**：讨论显示出对 Open LLM Leaderboard 的怀疑，特别是关于其推广的模型质量。`@fiddlenator` 批评该排行榜认可“垃圾合并模型（trash merge models）”，引发了关于此类模型的有效性和评估的对话。

- **Tiny Llama 在 Nintendo Switch 上运行引发关注**：随着 `@kalomaze` 分享了据称在 Nintendo Switch 上运行 Tiny Llama 和 Mistral 等 AI 模型的视频链接，成员们表示惊讶和好奇。这种独特的执行方式引发了关于在非传统硬件上部署轻量级模型的潜力的讨论。

**提到的链接**：

- [未找到标题](https://neurosity.co/)：未找到描述
- [DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/abs/2401.14196)：大语言模型的快速发展彻底改变了软件开发中的代码智能。然而，闭源模型的主导地位限制了广泛的研究和开发...
- [import sysimport osfrom tqdm import tqdmsys.path.append(os.path.dirname(os - Pastebin.com](https://pastebin.com/wgD8Q5Qs)：Pastebin.com 自 2002 年以来一直是排名第一的文本存储工具。Pastebin 是一个可以让你在线存储文本一段时间的网站。
- [Tails - Home](https://tails.net/)：未找到描述
- [How to Install NVIDIA Drivers on Rocky Linux 9 or 8 - LinuxCapable](https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-rocky-linux/)：学习如何使用命令行终端和 NVIDIA CUDA REPO 在 Rocky Linux 9 或 8 上安装最新版本的 NVIDIA 驱动程序。
- [God helmet - Wikipedia](https://en.wikipedia.org/wiki/God_helmet)：未找到描述
- [LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2)：未找到描述
- [Marvelous Dc Gotham GIF - Marvelous Dc Gotham Gotham Tv - Discover &amp; Share GIFs](https://tenor.com/view/marvelous-dc-gotham-gotham-tv-hugo-strange-dr-strange-gif-17601265)：点击查看 GIF
- [How I Won Singapore’s GPT-4 Prompt Engineering Competition](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41)：深入探讨我学到的利用大语言模型（LLMs）力量的策略
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind#screenshots)：一个支持多模态和函数调用的 LLM webui。 - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - facebookresearch/audio2photoreal: Code and dataset for photorealistic Codec Avatars driven from audio](https://github.com/facebookresearch/audio2photoreal)：由音频驱动的逼真 Codec Avatars 的代码和数据集 - GitHub - facebookresearch/audio2photoreal: Code and dataset for photorealistic Codec Avatars driven from audio
- [GitHub - facebookresearch/Qinco: Residual Quantization with Implicit Neural Codebooks](https://github.com/facebookresearch/Qinco?tab=readme-ov-file)：带有隐式神经代码本的残差量化 - GitHub - facebookresearch/Qinco: Residual Quantization with Implicit Neural Codebooks
- [Stanford Hypnosis Integrated with Functional Connectivity-targeted Transcranial Stimulation (SHIFT): a preregistered randomized controlled trial - Nature Mental Health](https://www.nature.com/articles/s44220-023-00184-z)：研究人员展示了一项双盲随机对照试验的结果，该试验利用经颅磁刺激对左背外侧前额叶皮层进行个性化刺激，以增加...

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1200016786880471090) (74 条消息🔥🔥): 

- **模型配置的技术咨询**：频道中的用户讨论了 `exllama`、`bagel` 和 `dolphin` 等各种模型的配置，重点关注 `rope_theta` 以及是否使用 `sliding_window` 等方面。例如，`@dreamgen` 指出了模型配置文件的差异，强调了 `rope_theta` 值的不一致以及 `sliding_window` 的使用情况 ([Hugging Face Bagel Config](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json), [Hugging Face Dolphin Config](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json))。

- **角色扮演模型讨论**：多位用户分享并寻求关于最佳 7B 参数角色扮演模型的建议，`@kalomaze` 推荐了 `Kunoichi DPO v2` 和 `Fett-uccine` ([Hugging Face 上的 Fett-uccine](https://huggingface.co/Epiculous/Fett-uccine-7B-GGUF/tree/main)) 等模型，同时还讨论了适用于不同 VRAM 容量的量化选项。

- **关于敏感内容模型的担忧**：对话简要触及了使用模型生成可能被视为敏感内容的话题，`@c.gato` 和 `@superking__` 强调了普遍共识是倾向于负责任地使用，并通过仔细的 prompting 应用现有模型。

- **澄清与修正**：像 `@jondurbin` 这样的用户对他们使用继承自 base models 的模型配置进行了澄清，为围绕针对特定目的定制模型的讨论增加了背景信息 ([Mistral base model config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json))。

- **Token Streaming 与文本转语音 (TTS) 集成工作**：`@mrdragonfox` 和 `@stoop poops` 分享了他们在将 Token Streaming 和 TTS 功能集成到 bot 中的进展和挑战，旨在实现实时运行，并探索高效的 API 和引擎，如带有 `coqui` 引擎的 `realtimeTTS`。

**提到的链接**：

- [Kronk Its All Coming Together GIF - Kronk Its All Coming Together - Discover &amp; Share GIFs](https://tenor.com/view/kronk-its-all-coming-together-gif-15058130)：点击查看 GIF
- [config.json · mistralai/Mistral-7B-v0.1 at main](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)：未找到描述
- [config.json · cognitivecomputations/dolphin-2.6-mistral-7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json)：未找到描述
- [Epiculous/Fett-uccine-7B-GGUF at main](https://huggingface.co/Epiculous/Fett-uccine-7B-GGUF/tree/main)：未找到描述
- [Release Quadratic Sampling Test Build (koboldcpp) · kalomaze/koboldcpp](https://github.com/kalomaze/koboldcpp/releases/tag/quad-sampling-v1)：上一个想法（Smooth Sampling）的替代方案，具有不同的缩放机制。其背后的想法是尽可能简化采样，并移除尽可能多的额外变量...
- [config.json · jondurbin/bagel-dpo-7b-v0.1 at main](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json)：未找到描述
- [config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json)：未找到描述

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1200022344538800189) (20 messages🔥): 

- **为聊天机器人塑造角色**：`@superking__` 建议编写角色如何响应不同用户输入的示例，并在该数据集上训练模型。他们还提到可以请求 ChatGPT 协助生成此数据集。

- **训练聊天机器人时数量至关重要**：在关于训练聊天机器人所需示例数量的讨论中，`@amogus2432` 提到，在训练 qlora 进行风格迁移时，10 到 100 个示例应该足够了。然而，`@dirtytigerx` 反驳称，要达到 Samantha 级别的性能，需要大约 6000 个多轮对话样本，并指出仅使用 100 个样本可能会导致 overfitting。

- **聊天机器人对话中的个性化触感**：`@lordofthegoons` 表达了创建具有特定 persona（人格）和一致对话风格的聊天机器人的愿望，类似于 Samantha 模型。他们担心生成的合成数据集在实现这一目标时会显得重复或局限。

- **探索训练聊天机器人的更廉价替代方案**：在讨论使用 GPT-4 API 生成数据集的潜在成本时，`@dirtytigerx` 建议探索本地 LLM 作为更经济的选择。他们提到使用 runpod 等平台进行实验可能比完全依赖 ChatGPT 等服务更便宜且高效，因为后者可能会遇到速率限制。

- **涉足金融顾问聊天机器人**：`@VR` 咨询了关于创建个人金融投资顾问聊天机器人的事宜，考虑使用 prompt tuning 和对金融文档及数据集进行 fine-tuning。他们寻求关于如何在 24GB GPU 的限制下有效部署模型，并利用最新的股票价格、摘要、趋势和专家分析的建议。
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199997735223447642) (19 messages🔥): 

- **讨论最佳权重假设**：`@sanjiwatsuki` 分享了一个假设，即由于 TIES 解析过程会导致部分有效权重下降，将权重保持在略高于 1.0 可能是模型性能的最优选。这个想法是为了补偿这种预期的下降。
- **对负权重的确定性存疑**：当 `@kquant` 询问负数是否会破坏脚本时，`@sanjiwatsuki` 表示不确定，但根据模糊的记忆建议代码可能可以无误地处理它。
- **探索受限模型的同化**：`@kquant` 提到对实验过度审查（censored）的模型感到好奇，以及是否可以有选择地同化模型，主要提取所需的特征。
- **使用 DARE 和 SLERP 进行选择性合并**：`@kquant` 建议使用 DARE 或 SLERP 等技术来合并在不同领域（如 ARC 和 MMLU）具有高性能的模型，以优化多种优势，而非简单地对分数取平均值。
- **对过拟合模型成功合并的关注**：`@kquant` 对两个过拟合模型在通过 SLERP 合并后仍能保持其测试排名感到惊讶，并质疑过拟合对这类操作预期的负面影响。
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1200086950397366292) (1 messages): 

- **新手关于 LangChain 微调的咨询**：`@nandavikas` 咨询了关于使用 LangChain **fine-tuning Llama2** 以从 PDF 中提取特定信息并作为字典输出的问题。他们分享了自己在 **PyTorch** 方面的经验，并寻求 LangChain 生态系统中类似流程的指导或有用文档。
  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1200055671194927236) (35 条消息🔥): 

- **GPT-4 速度差异担忧**：`@romansh2302` 对 **GPT-4-1106-preview** 通过 API 进行文档分析时的漫长处理时间表示沮丧，与 Web 版本相比，他经历了 3-7 分钟的延迟。`@rendo1` 和 `@lugui` 解释说，性能差异可能是由于 API 使用高峰或服务器资源分配引起的，并暗示除了可能升级服务计划以获得专用容量外，没有直接的解决方法。

- **关于 GPT-4 Turbo 稳定性的查询**：当被问及 **GPT-4 Turbo 稳定版** 是否会解决速度问题时，`@elektronisade` 澄清说 GPT-4 Turbo 本身并不支持直接的文件分析，这表明可能对功能可用性存在误解。

- **增强型 API 服务的替代方案和成本查询**：`@romansh2302` 考虑使用 **GPT-3.5 Turbo** 作为文档分析的替代方案，但发现效果较差。与 `@lugui` 的对话揭示了可能提供专用的 API 容量，但成本不确定，且是针对个人用例定制的，主要面向企业客户。

- **异常活动和错误排查**：`@hellomyfriend1576` 在使用 GPT-4 时遇到了系统提示“异常活动”的错误消息，并尝试通过清除 Cookie 和更改网络来解决。`@lugui` 和 `@og_tort` 推测这可能与使用 VPN、Prompt 的性质或账号共享行为有关。

- **关于 Dall-E 误解的讨论**：`@alwayspercipient` 注意到 Dall-E 在生成图像时经常出现拼写错误，`@muyfashionista` 建议在输入较短文本时这个问题可能不那么明显，并提供了社区讨论链接和减少生成图像中语法错误的技巧。

**提到的链接**：

[TuringsSolutions/PFAF750 · Datasets at Hugging Face](https://huggingface.co/datasets/TuringsSolutions/PFAF750)：未找到描述

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1200050868888801362) (105 条消息🔥🔥): 

- **关于“始终展开代码输出”功能的澄清**：`@angry_coder` 询问了“Always expand code output”设置，随后 `@darthgustav.` 澄清说该设置确保代码块被包裹（wrapped）以便于阅读。经过测试，`@darthgustav.` 确认这与代码块换行有关，增强了可读性。

- **探索 Python 代码导入**：用户 `@bambooshoots` 和 `@darthgustav.` 讨论了通过上传 zip 文件并调整 `system path` 来导入 Python 模块的可能性。尽管最初存在安全担忧，但他们推测了其在 OpenAI 限制内的潜在效用和扩展功能。

- **CustomGPT 编辑需要新对话**：`@elegante94` 询问对 CustomGPT 的编辑是否会反映在活动会话中，`@darthgustav.` 回复说必须开启新对话才能看到更改。这让一直在此误解下进行迭代编辑的 `@elegante94` 感到困惑。

- **优化 GPT 的 Prompt 语言**：`@elegante94` 寻求关于附加图像或使用复杂措辞是否能改善 GPT 创意图像输出的建议。`@darthgustav.` 建议 GPT 无法解释 Prompt 中的图像，并推荐使用精确的语言以获得更好的输出，同时也提到了对通过 Microsoft Autogen 和 CrewAI 进行 Agent 交互等主题的潜在探索。

- **GPT Bot 更新挑战和模仿限制**：`@rodney.leonardo` 在保存设计为产品设计助手的 GPT bot 更新时遇到困难。`@darthgustav.` 建议了排查步骤，并指出存在禁止模仿在世人物的限制，这导致用户意识到命名特定的在世人物（如 "Jony Ive"）可能会导致 Bot 无法保存。
  

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **探索用于布道词转录的段落分块和 NLP**：`@doublefelix` 正在进行一个涉及布道词转录的项目，寻求将大块文本（高达 50KB 的文件）拆分为易于管理的段落并修正语法问题。他们探索了使用 GPT-3.5 和 NLP 技术作为潜在解决方案，但在输入限制和寻找有效的段落分块策略方面面临挑战。

- **迈向高效工作流之路**：在对 GPT-3.5 进行多次尝试（包括各种 Prompt 策略和尝试利用 API 功能）后，`@doublefelix` 在确保 AI 能够正确识别并处理全部输入内容而不产生幻觉或重大错误方面遇到了障碍。

- **考虑 GPT-4 在增强上下文管理方面的能力**：`@darthgustav` 建议利用 GPT-4 Turbo，它可以处理更大的上下文窗口（context windows），并利用 Python 工具进行语义分析和分段，从而可能绕过 GPT-3.5 遇到的限制。这种方法可能为处理和拆分转录文件提供更高效的工作流。

- **反思项目成本与约束**：对于转向 GPT-4 Turbo 处理转录内容所带来的成本增加，人们产生了担忧。`@doublefelix` 表示希望在功能性和可负担性之间找到平衡，同时考虑了涉及的人力成本以及使用更先进 AI 模型的财务影响。

- **探索替代 NLP 工具及总结**：当 `@doublefelix` 权衡各种选择（包括使用 GPT-4 Turbo 的备选计划）时，他们还计划探索诸如 `semantic-text-splitter` 之类的 NLP 软件包，以寻求文本处理需求的可能解决方案。他们对讨论和提供的见解表示感谢，强调了 Prompt Engineering 的复杂性以及利用 AI 模型实现特定项目目标的细微差别。

**提到的链接**：

[How do you maintain historical context in repeat API calls?](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395)：每次我调用 API 时，它都像 chat.openai.com 那样没有先前的上下文。有没有办法在会话期间保持模型的状态？response = openai.Completi...

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **探索 GPT 在处理长文本时的局限性**：`@doublefelix` 参与了一场关于如何使用 GPT 有效地将大块文本拆分为较小的、易于管理的片段以进行分段和语法纠错的讨论。最初尝试使用 GPT-3.5，但面临 AI 忽略部分文本或产生内容幻觉的挑战。
  
- **语义聚类与分段策略**：`@darthgustav` 为 `@doublefelix` 的尝试提出了多种策略，包括用于段落划分的语义聚类，以及利用 Custom GPT 或 GPT-4 Turbo 的先进能力和 Python Tool 来进行更准确的文本处理。

- **处理长文本的成本**：对话还涉及了处理大量文本的成本影响。`@doublefelix` 发现 GPT-3.5 的处理成本高于最初预期，这促使他们考虑使用 GPT-4 Turbo，尽管其成本更高。

- **文本分割中的 NLP 与 AI 之争**：目前正在讨论自然语言处理（NLP）工具还是 AI（特别是具有增强上下文窗口和 Python 能力的 GPT 模型）在分割和纠正大型转录文件方面表现更好。`@doublefelix` 正在考虑探索 NLP 工具，在 PyPi 上找到了一个潜在的软件包，并对使用带有结构化指令的 GPT-4 以提高效率持开放态度。

- **自动化文本结构化的探索**：在最后，`@doublefelix` 反思了这段历程，承认了所面临的挑战，以及从与 GPT 模型交互和使用结构化指令分割大文本文件中获得的见解。虽然 GPT-3.5 取得了一些进展，但仍倾向于探索 GPT-4 的能力或 NLP 工具以实现更自动化的方法。

**提到的链接**：

[How do you maintain historical context in repeat API calls?](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395)：每次我调用 API 时，它都像 chat.openai.com 那样没有先前的上下文。有没有办法在会话期间保持模型的状态？response = openai.Completi...

  

---

### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1200095821480341595) (7 messages): 

- **探索扩展上下文能力的最佳解决方案**：`@cryptossssun` 询问了目前增强上下文能力的最佳方法，引发了成员之间的建设性对话。`@_cherrry` 推荐根据讨论的一篇论文进行 Fine-tuning，认为这是一种可行的方法。

- **关于 Mistral 策略变化的辩论**：针对 **Mistral Instruct v0.2** 提出的疑问，`@dreamgen` 强调了其禁用了 Sliding Window 技术，转而采用缩放 `rope_theta` 的做法，并分享了 [config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json) 以突出这一转变。这引发了关于 Sliding Window 技术在长上下文场景下是否效果较差的推测。

- **MistralLite 模仿 Mistral 策略**：`@dreamgen` 的进一步询问引出了一个观察，即 `amazon/MistralLite` 在上下文窗口策略上采用了与 **Mistral** 类似的方法，使讨论集中在不断演进的模型配置上。

- **令人印象深刻的上下文窗口扩展成果**：`@stellaathena` 展示了一项非凡的成就，即通过极少的样本和训练步骤将 **LLaMA-2-7B-Chat** 的上下文窗口扩展到了 16,384。该方法以极高的效率著称，引起了众人的惊叹。

- **SelfExtend 被提议作为无需 Fine-tune 的选项**：当 `@cryptossssun` 寻求扩展上下文能力的建议时，`@leontello` 推荐了 **SelfExtend**，认为它是那些选择不进行 Fine-tune 的人的创新解决方案。这为在无需密集训练的情况下增强模型性能的讨论引入了另一个视角。

**提到的链接**：

[config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json)：未找到描述

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1200111343458599076) (5 messages): 

- **技术的两极分化效应**：`@ldj` 分享了一个观点，即技术的进步可能会进一步使社会两极分化，**最无能的人变得更加堕落，而最有能力的人则进行更多的自我提升**。这反映了关于科技对不同社会阶层影响的持续讨论。
- **游泳猫带来的意外笑声**：`@Error.PDF` 分享了一个 [游泳猫 GIF](https://tenor.com/view/cat-swimming-poopsie-silly-gif-14546589990767279660)，在严肃的技术讨论中带来了一丝幽默，活跃了气氛。
- **Twitter 神秘的推文加速**：`@fullstack6209` 观察到 Twitter 上新帖子出现的速率显著增加，从 **每 10 分钟 2-3 条增加到每分钟约 70 条**，引发了对平台内容分发算法变化的疑问。
- **关于 Twitter 故意放慢 AI 的猜测**：紧接着，`@fullstack6209` 分享了对 Twitter **故意放慢 AI 以管理或控制内容流** 的怀疑，反映了对社交媒体平台与人工智能技术之间相互作用的担忧。
- **LLM 一键量化为 GGUF**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=wlPxEq_Mtkc)，演示了如何一键轻松地将任何 Hugging Face 上的 LLM 量化为 GGUF 格式，标志着机器学习和神经网络效率领域的重大进展。

**提到的链接**：

- [Cat Swimming GIF - Cat Swimming Poopsie - Discover &amp; Share GIFs](https://tenor.com/view/cat-swimming-poopsie-silly-gif-14546589990767279660)：点击查看 GIF
- [AutoGGUF 一键量化 LLM 为 GGUF 格式](https://www.youtube.com/watch?v=wlPxEq_Mtkc)：使用 Maxim Labonne 提供的 Notebook 将任何 Hugging Face LLM 量化为 GGUF 格式 #llms #ml #ai #neuralnetworks #deeplearning #gguf https://colab.research.googl...

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1200251006743744512) (2 messages): 

- **Everyone Coder 33B Base - GGUF 的 Benchmark 请求**：`@benxh` 请求 `@231912337869635584` 对 [Everyone Coder 33B Base - GGUF](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF) 模型进行 Human Eval Benchmark。该模型由 [rombo dawg](https://huggingface.co/rombodawg) 创建，使用 Massed Compute 进行量化，并支持新的 GGUF 格式（GGML 的替代者）。

- **对 Hermes Mixtral 性能的关注**：`@teknium` 表示希望看到 **Hermes Mixtral** 的性能 Benchmark，但未提供更多细节或链接。

**提到的链接**：

[TheBloke/Everyone-Coder-33B-Base-GGUF · Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF)：未找到描述

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1200148361370669126) (2 messages): 

- **OpenAI 发布新 Embedding 模型及 API 更新**：`@tsunemoto` 分享了一个[链接](https://openai.com/blog/new-embedding-models-and-api-updates)，详细介绍了 **OpenAI 发布的新 Embedding 模型**，以及 GPT-4 Turbo 和审核模型的更新。GPT-3.5 Turbo 将会降价，并推出新的 API 使用管理工具，同时承诺发送到 OpenAI API 的数据将不会被用于训练其模型。

- **介绍用于基于内容的数据生成 (Content-grounded Data Generation) 的 Genie**：`@metaldragon01` 重点推介了[最近发表的一篇论文](https://arxiv.org/abs/2401.14367)，内容关于 **Genie**。这是一种旨在克服基于内容的生成任务中高质量数据短缺的新方法。Genie 使用三阶段过程（内容准备、生成和过滤机制）来创建自然且高质量的任务特定示例，特别适用于长篇问答 (LFQA)、摘要和信息提取。

**提到的链接**：

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)：基于内容的生成任务缺乏高质量数据，已被认为是推进这些任务的主要障碍。为了弥补这一差距，我们提出了 Genie，一种用于自动...
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将下调 GPT-3.5 Turbo 的价格。

  

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1200033366272122910) (361 条消息🔥🔥): 

- **WebGL 上的 GPT-2 推理探索**：`@_3sphere` 和 `@n8programs` 讨论了在 WebGL 中实现 GPT-2 推理，`n8programs` 分享了一个使用 ThreeJS 和 GLSL 进行[向量相似度计算的详细内核](https://discord.com)。对话强调了机器学习模型直接在浏览器图形流水线中运行的潜力。

- **LLaMA 克隆尝试与讨论**：`@balnazzar3047` 和 `@euclaise` 讨论了创建一个类似于 LLaMA 的 Mixtral 模型，但在理解模型构建过程方面遇到了挑战。分享了一些有用的资源和解释，包括 `@qnguyen3` 提供的用于初始化新模型的 Colab 笔记本。

- **模型效率辩论**：`@carsonpoole` 分享了使用 RMSNorm 微调 phi2 模型的更新以及进一步优化的计划，暗示了相比传统 Transformer 模型潜在的效率提升。该方法引发了人们对比较并行残差架构在推理和训练速度方面的兴趣。

- **基于用户反馈的增强版 Word2Vec**：`@everyoneisgross` 对 Word2Vec 进行了创新，引入了一种根据用户输入细化和扩展语料库数据的方法，并利用 Mistral Instruct 提供额外上下文。这种方法虽然简单，但因其在轻量级 NLP 任务中的有效性和潜力而受到称赞。

- **利用 GPU 进行高级机器学习计算**：`@n8programs` 详细介绍了 WebGL 在机器学习中的应用，解释了 3D 纹理和空间思维如何实现超出典型 GPU 限制的计算。这种方法被视为“在视频游戏技术基础上的巧妙利用”，展示了对游戏硬件进行科学计算的创造性利用。

**提及的链接**：

- [EvalPlus 排行榜](https://evalplus.github.io/leaderboard.html)：未找到描述
- [人类 vs. 机器：每瓦特智能](https://meditations.metavert.io/p/human-vs-machine-intelligence-per)：思考机器可能不会在所有地方同时获胜的可能性
- [Mamba：具有选择性状态空间的线性时间序列建模](https://openreview.net/forum?id=AL1fq05o7H)：基础模型现在驱动着深度学习中大多数令人兴奋的应用，几乎普遍基于 Transformer 架构及其核心注意力模块。许多……
- [Google Colaboratory](https://colab.research.google.com/drive/1-D6ZGE3SZZbIkqhWfxun8CwQWBD5YC2d?usp=sharing)：未找到描述
- [关于新 2 x RTX 3090 配置的建议](https://forums.fast.ai/t/recommendations-on-new-2-x-rtx-3090-setup/78202)：你好，我正在出售旧的 GTX 1080 并用新的 RTX 3090 升级我的深度学习服务器。我也在考虑明年晚些时候再增加一块 RTX 3090。我从多个渠道了解到涡轮式（blower-style）……
- [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralConfig)：未找到描述
- [🤗 Transformers](https://huggingface.co/docs/transformers/)：未找到描述
- [GitHub - cg123/mergekit (mixtral 分支)](https://github.com/cg123/mergekit/tree/mixtral)：用于合并预训练大语言模型的工具。
- [培养活的大鼠神经元来玩……DOOM？](https://www.youtube.com/watch?v=bEXefdbQDjw)：访问 https://squarespace.com/thethoughtemporium 使用代码 thethoughtemporium 在首次购买网站或域名时节省 10%……
- [通过实时 AI 解决方案加速系统 - Groq](https://groq.com/)：由硬件和软件驱动。总部位于美国。现已推出。
- [EVGA SuperNOVA 1600 P2, 80+ PLATINUM 1600W, 全模组, EVGA ECO 模式, 10 年质保, 包含免费电源自检测试仪 220-P2-1600-X1](https://www.evga.com/products/product.aspx?pn=220-P2-1600-X1)：支持第四代 Intel Core 处理器（C6/C7 空闲模式）。介绍 EVGA SuperNOVA 1600 P2 电源。该电源以 1600W 的连续功率输出和 92% 的效率树立了标杆……
- [超越光罩限制的设计](https://semiengineering.com/designs-beyond-the-reticle-limit/)：芯片正面临技术和经济障碍，但这几乎没有减缓设计尺寸和复杂性的进步速度。

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1200051857771475034) (35 条消息🔥): 

- **CodeLlama 34B 微调配置讨论**：`@ganymede123` 询问了微调 CodeLlama 34B 的理想工作站配置，考虑使用 4xA6000 GPU。`@teknium` 回复称该配置仅足以运行 QLoRA，并建议进行完整微调可能需要一整台 DGX。

- **T5 微调问题排查**：`@maxpappa` 讨论了对齐微调版 T5 时遇到的问题，表现为输出具有确定性且奖励准确率（reward-accuracies）停滞。`@locutusque` 和 `@carsonpoole` 的建议包括避免使用 paged 8bit Adam，考虑 T5 的数值不稳定性，以及对无穷大值（infs）进行截断（clamping），特别是在 Encoder 部分。

- **探索用于攻击性网络安全的 LLM**：`@useewhynot` 寻求适用于攻击性网络行动或夺旗赛（CTF）竞赛的 LLM 推荐。`@kenakafrosty` 和 `@georgejrjrjr` 推荐了 HuggingFace 上的 WhiteRabbitNeo 模型，强调了它们对这类任务的专注。

- **LLM 微调库**：针对 `@moconna` 关于首选 LLM 微调库的提问，`@kenakafrosty` 提到使用了 trl，并表示更倾向于使用 axlotl，指出了它在某些应用中的优势。

- **选择最佳编程 LLM 及微调建议**：`@findmyke` 询问了目前最适合编程的 LLM，`@.ben.com` 引导其查看 EvalPlus 排行榜进行对比。此外，`@moconna` 询问了使用 Llama Factory 微调 Mistral 的关键超参数（hyperparameters），重点关注学习率（learning rate），并寻求默认值或模板的建议。

**提到的链接**：

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述
- [WhiteRabbitNeo/WhiteRabbitNeo-33B-v1 · Hugging Face](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1)：未找到描述
- [WhiteRabbitNeo/WhiteRabbitNeo-13B-v1 · Hugging Face](https://huggingface.co/whiterabbitneo/WhiteRabbitNeo-13B)：未找到描述
- [WhiteRabbitNeo - A co-pilot for your cybersecurity journey](https://www.whiterabbitneo.com/)：未找到描述

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1200102006929506434) (3 条消息): 

- **寻求简单的 3b Obsidian Python 脚本**：`vic49.` 正在寻找一个简单的 Python 脚本，该脚本利用 transformers 库并启用远程代码执行（remote code execution），用于处理 **3b Obsidian 模型**。
- **代码重构进行中**：`qnguyen3` 作出回应，承诺重构全部代码以确保与**最新的 llava 仓库**兼容，从而解决 `vic49.` 的需求。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1200042055980822649) (219 条消息🔥🔥): 

- **为挪威语微调 Mistral**：`@henriklied` 正在使用 10 万篇文章组成的集合微调 Mistral，用于挪威语的标题生成，并分享了具体的模型参数。然而，`@le_mess` 建议在 4 个 epoch 时停止以避免过拟合（overfitting），并指出对于有限的数据集，10 个 epoch 的迭代次数可能过多。

- **shareGPT 在处理长对话方面的缺陷**：`@c.gato` 对 shareGPT 处理超出预定义上下文长度（context length）的对话方式提出质疑，指出长对话会被直接丢弃，而不是通过修剪来维持上下文长度。`@le_mess` 确认在 Axolotl 中，除了 completion 任务外，这种行为是设计使然。

- **关于有效聊天表示形式的讨论**：`@suikamelon` 和 `@c.gato` 辩论了 ChatML 与原始聊天格式在 AI 模型管理对话上下文方面的效能，偏好因 token 效率和灵活性而异。此外，`@dreamgen` 和 `@c.gato` 讨论了 BlockML 和 IRC 格式等替代表示形式的潜在益处。

- **训练模型的考量**：`@dreamgen` 询问了在训练 QLoRA 等模型时，H100 与 A100 的单 GPU 性能对比，`@c.gato` 指出 H100 在特定任务中的成本效益。讨论还涉及了修改 rope theta 及其与模型训练中缩放技术（scaling techniques）的关系。

- **部署 AI 模型的挑战**：`@dangfutures` 寻求托管其模型的帮助，强调了 serverless 部署的困难并讨论了潜在的解决方案。对话凸显了在为 AI 模型选择和部署合适的托管方案时所涉及的复杂性。

**提到的链接**：

- [Pullrequest GIF - Pullrequest - Discover &amp; Share GIFs](https://tenor.com/view/pullrequest-gif-20256291)：点击查看 GIF
- [axolotl/deepspeed_configs/zero3.json at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3.json)：尽管提出 axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [more checks and fixes for deepspeed and fsdp by winglian · Pull Request #1208 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208)：未找到描述

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1200172001835352115) (1 条消息): 

- **活跃开发导致消息删除**：`@gahdnah` 在查看最新 commit 后发现讨论领域仍处于活跃开发阶段，随后删除了之前的消息。未提供有关开发领域或 commit 的具体细节。

- **资助庆祝**：axolotl-dev 社区用热情的表情符号庆祝获得资助（grant）。消息中未包含有关资助用途或资助者的细节。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1200033196105023572) (47 messages🔥): 

- **不使用 Bitsandbytes 训练 Qlora 的解决方案**：`@matanvetzler` 询问了如何不使用 Bitsandbytes 训练 Qlora 模型，以实现与 VLLM 的兼容性。`@le_mess` 和 `@stefangliga` 建议了替代方案，例如使用 **fp16** 或 **AutoAWQ** 进行量化，并将训练好的 Qlora 模型合并到基础模型中。此外，`@stefangliga` 分享了一个用于合并的 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base) 以及 Tim Dettmers 提倡在合并前进行模型量化的推文。

- **模型合并策略讨论**：`@nanobitz` 和 `@c.gato` 讨论了合并具有不同量化适配器的模型的有效性和潜在问题，指出性能有轻微提升，但也强调了合并多个适配器的复杂性。`@c.gato` 提到有 1-2% 的提升，但建议由于潜在问题需谨慎对待。

- **使用自定义提示词训练和评估模型**：`@sadaisystems` 寻求关于配置自定义提示词和模型训练数据集格式的建议。有人对异常低的训练损失表示担忧，对此 `@caseus_` 等人建议，对于 SQL 查询等确定性任务，这可能是预期行为。对话强调了理解模型训练表现的重要性和挑战。

- **在 Mistral 上进行持续预训练和基准测试评估**：`@nickbro0355` 和 `@sadaisystems` 提出了关于在 Mistral 上继续预训练以及在训练期间评估模型的问题。`@caseus_` 提供了关于使用 **continuous pretraining** 的线索，并介绍了在训练期间使用 HuggingFace 上的 *dharma-1* 等数据集进行 **benchmark evaluation** 的方法，建议设置 `do_bench_eval: true` 并推荐使用 `bench_dataset: pharaouk/dharma-1/dharma_1_full.json` 来检查相对性能提升。

- **Axolotl 用户的文档和配置技巧**：讨论指出 axolotl 的 **main README** 中缺少某些参数（如 `bench_dataset`），表明文档有待改进。这些讨论支持了这些配置对测试运行的好处以及社区内模型开发的迭代性质。

**提到的链接**：

- [pharaouk/dharma-1 at main](https://huggingface.co/datasets/pharaouk/dharma-1/tree/main)：未找到描述
- [qlora/qmerge.py at main · jondurbin/qlora](https://github.com/jondurbin/qlora/blob/main/qmerge.py)：QLoRA: Efficient Finetuning of Quantized LLMs。通过在 GitHub 上创建账号为 jondurbin/qlora 开发做贡献。
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base)：Go ahead and axolotl questions。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 开发做贡献。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1200086950594498580) (7 messages): 

- **发布新的 DPO 数据集**：`dangfutures` 分享了 [Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset) 上一个新数据集的链接，该数据集用于训练 Snorkel 模型，重点是不使用外部 LLM 响应，而仅使用来自 UltraFeedback 的提示词。该方法包括为每个提示词生成 5 个响应变体，使用 LLM 进行响应重排序，并应用直接偏好优化 (Direct Preference Optimization, DPO)。
  
- **Mistral 7 获得微调**：`dangfutures` 对 Mistral 7 正在进行的微调表示热切关注，暗示该模型将有重大改进或更新。

- **讨论 ALPCA 数值**：`dangfutures` 最初提到了一个与 ALPCA 相关的数字，随后澄清该数字为 34%，这被认为是对旧 GPT-4 指标的改进。

- **对讨论的反应**：`_dampf` 用一个来自 [Tenor 的 GIF](https://tenor.com/wSUt.gif) 回应了正在进行的讨论，对分享的数据集和模型性能信息表达了幽默或生动的反应。

**提到的链接**：

- [Sure Jennifer Lawrence GIF - Sure Jennifer Lawrence The Mocking Jay - Discover &amp; Share GIFs](https://tenor.com/wSUt.gif)：点击查看 GIF
- [snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset)：未找到描述

  

---

### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1200279388797804555) (2 条消息): 

- **寻求 DPO 训练图表的澄清**：用户 `noobmaster29` 询问是否有任何资源可以帮助理解 **dpo training plots**，暗示需要文档或指南来解释来自 DPO 训练会话的数据。
- **排查 DPO 数据集问题**：`noobmaster29` 询问了 **dpo dataset** 的必要组成部分，特别是除了 prompt/input 以及 chosen/rejected 对之外是否还需要其他内容。他们提到尽管包含了这些列，但在数据集处理方面仍遇到问题。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=wlPxEq_Mtkc
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199986755768688711) (118 条消息🔥🔥): 

- **代理支持查询与前沿探索**：用户表示在 LM Studio 中需要代理设置，因为无法使用模型搜索功能，特别是 `@laooopooo_02864` 提到在 HuggingFace 被屏蔽的地区存在此限制。Wildcat_aurora 分享了一个 [LM_Chat_TTS_FrontEnd 的 GitHub 链接](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html)，通过移动设备的文本转语音功能促进与 LM Studio 模型的界面交互。

- **LM Studio 与模型加载挑战**：多位用户遇到了 LM Studio 的问题，例如 `@xmiruso` 和 `@bagua` 分别面临 GPU 检测和模型加载错误。解决方案包括确保 GPU offload 设置以及切换到替代的模型版本。

- **教室中的交互式学习环境**：`@fabguy` 提供了关于为教室使用设置 LM Studio 的见解，建议在服务器上设置 Web 界面，并使用兼容 OpenAI 的客户端进行学生互动。他们还建议查看 [Awesome LLM Web UI GitHub 仓库](https://github.com/JShollaj/Awesome-LLM-Web-UI) 以获取前端选项。

- **并行模型操作与支持问题**：`@therealtrebor` 询问了并行运行多个模型的问题，通过测试发现这对于特定用例是有益的，而 heyitsyorkie 指出这需要运行多个 LM Studio 实例。包括 `@Leo - Moon Boyz 🚀🚀🛸` 在内的用户寻求联系支持以进行故障排除的方法，并被引导使用特定的错误报告频道。

- **Silly Tavern 误解与数据隐私讨论**：社区澄清了关于 Silly Tavern 收费的误解，`@ptable` 和 `@technot80` 确认了其免费状态。关于确保 LM Studio 数据隐私的讨论包括建议在没有外部互联网访问的 Docker 或 VM 环境中运行，`@agcobra1` 寻求其他保证，而 `@bigdave7541` 建议使用 Wireshark 进行验证。

**提及的链接**：

- [GitHub - JShollaj/awesome-llm-web-ui: A curated list of awesome Large Language Model (LLM) Web User Interfaces.](https://github.com/JShollaj/Awesome-LLM-Web-UI): 一个精选的优秀大语言模型 (LLM) Web 用户界面列表。
- [GitHub - FriendofAI/LM_Chat_TTS_FrontEnd.html: LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, making it suitable for a wide range of users interested in exploring voice interactions with AI models.](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html): LM_Chat_TTS_FrontEnd 是一个简单而强大的界面，用于使用文本转语音功能与 LM Studio 模型进行交互。该项目旨在轻量级且用户友好，适合对探索 AI 模型语音交互感兴趣的广泛用户。

  

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199997195924033607) (54 条消息🔥): 

- **C++ Redistributables 解决模型加载错误**：`@rparada` 在加载 **Stable Code**、**Deepseek** 和 **Codellama** 等各种模型时反复遇到错误。在采纳了 `@heyitsyorkie` 的建议更新 C++ Redistributables 后，问题得到了解决，这说明了保持系统组件更新对于模型兼容性的重要性。

- **尚无多模态模型能与 GPT-4 匹敌**：针对 `@mudf00t` 询问是否有与 **GPT-4** 相当的多模态模型，`@heyitsyorkie` 澄清目前还没有能够完美结合视觉和文本生成的综合性模型，突显了当前 AI 模型领域的局限性。

- **讨论使用 Azure OpenAI Service 运行 GPT-4**：`@mickael6102` 分享了他们公司通过 **Azure 的 OpenAI service** 使用 **GPT-4** 的经验，引发了由 `@vbwyrde` 主导的关于云端与本地模型部署的讨论。数据隐私、成本以及对第三方云服务的依赖是讨论的主要焦点。

- **InternLM 引起关注**：`@vbwyrde` 重点介绍了一个名为 **InternLM** 的新模型，指出其声称推理能力比 GPT-4 高出 **11%**，并具有先进的 function-calling 能力。这引起了人们将其作为其他模型替代方案的兴趣，`@mickael6102` 正在考虑在未来的项目中使用它。

- **关于企业开源贡献的辩论**：由 `@vbwyrde`、`@.gumdro` 和 `@fabguy` 发起的讨论思考了 **Meta 发布 LLaMA2** 背后的动机，推测其战略利益包括制定标准和削弱竞争对手。这场对话提出了关于 AI 领域开源贡献与企业战略之间关系的问题。

**提到的链接**：

- [Mark Zuckerberg Adjust GIF - Mark Zuckerberg Adjust Facebook - Discover &amp; Share GIFs](https://tenor.com/view/mark-zuckerberg-adjust-facebook-smile-on-trial-gif-11618142)：点击查看 GIF
- [internlm (InternLM)](https://huggingface.co/internlm)：未找到描述

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1200212033254215700) (4 条消息): 

- **MoE 模型设置中的 Bug 警报**：`@msz_mgs` 报告了一个 Bug，即从 **4X MoE model** 切换到 `num_experts_used=4` 的 **2X MoE model** 时，会生成一条提示重启应用程序的错误消息。这似乎导致无法在不重启的情况下调整设置。
- **开始调查 MoE 模型问题**：针对 `@msz_mgs` 的报告，`@yagilb` 确认了该问题，并要求提供更多关于所涉及的 **2x MoE** 和 **4x MoE models** 的信息，以协助解决问题。
- **提供 MoE 模型配置说明**：`@dagbs` 提供了一个建议，指出对于 **4X MoE model**，预期设置的专家数量应为 **2**，这为可能的预期配置提供了见解。
- **0.2.11 更新后的性能下降**：`@golangorgohome` 在 Windows 11 上使用 **0.2.11 更新**时遇到了严重的性能问题，包括搜索图标响应延迟和搜索结果缓慢，尽管其拥有高速互联网连接。
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1200033522530914395) (11 条消息🔥): 

- **讨论性价比（价格/显存）最高的 GPU 选择**：`@gitanos` 询问 **4060ti 16 GB RAM** 的效率和价值，质疑它是否是目前显存价格比的最佳选择。`@heyitsyorkie` 建议考虑在 eBay 上购买二手的 **3090**，因为在英国它只贵一点点，但可能提供更好的价值。
- **e0.211 的技术故障**：`@madan.pandit` 报告称之前可以运行的模型已经停止工作，并专门询问了其他人对 **e0.211 更新**的体验。`@heyitsyorkie` 指出，在最新的构建版本中，**GGML models** 已被弃用，取而代之的是 `llama.cpp`，这影响了它们的兼容性。
- **GGUF 模型内存问题报告**：`@madan.pandit` 提到在运行 **GGUF models** 时遇到内存不足错误，表明在最近的更新后可能存在兼容性或资源分配问题。
- **Mac Studio 被推荐作为 LLM 硬件**：`@heyitsyorkie` 建议购买顶配的 **M2 Mac Studio** 来运行大语言模型 (LLMs)，并指出了其紧凑的尺寸、高效能和设计美学。
- **关于多 GPU 配置中 P40 和 M40 GPU 的辩论**：在关于联合使用 **P40 和 M40 GPU** 的讨论中，`@docorange88` 询问了它们的整体性能，而 `@wildcat_aurora` 和 `@rugg0064` 认为 **P40s** 是不错的投资，但认为 **M40s** 不值得购买，并透露了近期购入 P40s 的计划。
  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1199990332016836689) (47 messages🔥): 

- **选择合适的 LLM 是一门艺术**：`@mmonir` 概述了选择最佳大语言模型 (LLM) 的标准，强调了写作、编程和总结等用例的重要性，并考虑了量化 (quantization)、硬件能力和用户体验等因素。他们还建议利用排行榜和 LLM 演示来做出更明智的决定，并从各个平台寻求用户反馈。

- **新安装版本中的 VRAM 读取错误**：`@mudf00t` 报告了一个问题，即在新的 Nobara 安装中，LM Studio 显示 RTX 3090 的 VRAM 为 0。`@yagilb` 提供了一个专门针对 Nvidia 用户的解决方法链接，该方法不适用于 Mac M2 芯片。

- **故障排除版本兼容性**：`@pdg` 面临模型无法在 LM Studio 最终新版本上运行的问题，其 MacBook M2 出现了异常的系统行为。他们得到了 `@yagilb` 的协助，包括一个回退到之前版本 0.2.10 的链接，这暗示了潜在的软件回归问题。

- **模型加载困扰及解决方法**：在降级 LM Studio 版本后，`@pdg` 继续遇到模型加载错误，但通过先发起简短句子的交互找到了临时修复方法。这种变通方法允许随后成功处理较长的输入。

- **上下文长度和溢出策略见解**：`@mattjcly_55150` 询问了 `@pdg` 可能使用的上下文长度 (context length) 设置，建议调整上下文溢出策略 (context overflow policies) 可能会解决与输入长度限制相关的问题。这暗示初始输入长度超过设定的上下文长度可能是遇到错误的根本原因。
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1200325029498458112) (1 messages): 

- **损坏的置顶链接引发困扰**：`sunglasses.emoji` 报告称 Autogen Studio 中关于空字符串的**置顶链接已损坏**，并寻求帮助。他们正在寻求关于**如何在 Autogen Studio 中创建自定义 Agent 类**的指导。
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1200104377801773109) (10 messages🔥): 

- **模型效用在不同框架中的一致性问题**：`@pefortin` 强调了包括 **Open Interpreter**、**memGPT** 和 **crewai** 在内的多个 AI 框架的问题，特别是模型无法妥善使用可用工具。他们提到使用了 **mixtral8x7B** 和 **deepseek coder 33B** 等中大型模型，排除了模型大小是问题根源的可能性。

- **使用 RTX 3090 进行模型测试的探索**：`@mudf00t` 分享了他们正在测试各种模型，利用 **RTX 3090** 的性能来处理更大的模型，指出了硬件优势在模型实验中的直接体验。

- **API 知识断层阻碍开发**：`@mudf00t` 对包括 **OpenAI** 提供的模型在内的模型未能及时更新**当前 API** 知识表示沮丧，这导致了应用开发过程中的上下文问题。

- **集成与微调即将到来**：`@222gate` 注意到 **memgpt** 集成已停止，并分享了为特定函数调用 (function calls) 微调 **Mistral 模型**的计划，认为这种方法可能会解决一些操作性问题。

- **Mistral 的创造性幻觉**：`@mudf00t` 观察到 **Mistral** 会创建虚构的目录结构和代码，强调了该模型倾向于产生超出预期操作任务的复杂幻觉 (hallucinate) 输出。
  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1200008042968789032) (163 messages🔥🔥): 

- **AI 摘要工作的 GPU 租赁**：`@mrdragonfox` 强调了 **runpod, vast, 和 lambda** 等服务提供按小时租赁的 GPU，建议用户利用这些资源进行 AI 工作。他们还提到 **Kaggle** 作为一个免费资源，每周提供 30 小时的双 T4 GPU 使用时间。

- **订阅问题与支持联系方式**：`@glorfsf` 在更改订阅中的使用限制选项时遇到问题，`@mrdragonfox` 指出这是默认限制问题，并建议联系 **support@mistral.ai** 寻求帮助。

- **Mistral 的用例及与其他框架的集成**：在关于如何利用 Mistral 模型的讨论中，`@ethux` 分享了围绕客户支持自动化的 Finetuning 用例，并提到一个有趣的 **Embeddings 模型**，展示了 Mistral 模型除了与 **GPT-4** 竞争之外的多样化应用。

- **探索 Mistral 的 API 限制与增强**：用户讨论了与 API Token 成本、Tokenization 问题以及特定 AI 模型性能相关的困难，`@mrdragonfox` 和其他人提供了关于模型工作原理的见解，并提出了更好的模型利用方案。

- **关于模型选择与优化的技术辩论**：一场关于 **GitHub Copilot** 有效性的引人入胜的对话展开了，`@mrdragonfox` 和 `@i_am_dom` 辩论了其底层技术，展示了社区在理解和优化 AI 模型性能方面的深度参与。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/19emcxp/stanford_and_openai_just_released_a_research/): 未找到描述
- [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2): 一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - GitHub - turboderp/exllamav2
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): 未找到描述
- [TuringsSolutions/PFAF750 · Datasets at Hugging Face](https://huggingface.co/datasets/TuringsSolutions/PFAF750): 未找到描述

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1200043535878074391) (9 messages🔥): 

- **明确 Training 与 Finetuning 的区别**：`@ethux` 询问讨论内容是关于 Training 还是 Finetuning，指出在考虑系统需求时理解上下文的重要性。
- **数据集的内存需求**：`@ethux` 讨论到虽然具体需求尚不明确，但 64GB 内存对于未知大小的数据集应该是足够的，指出了理解内存需求与数据集大小关系的重要性。
- **Mistral 4bit Inference 的内存占用**：根据 `@ethux` 的说法，**Mistral** 进行 4bit Inference 至少需要 26GB 内存，这表明推理任务有显著的内存需求。
- **模型效率优化技巧**：`@mrdragonfox` 对比了 **exllama** 和常规的 4bit Transformer，建议 Exllama 在内存使用方面效率更高，从而强调了不同的优化策略。
- **减少内存占用的 Quantization 策略**：`@mrdragonfox` 提议使用 **exllamav2 作为 Quantization 策略**，优于 BnB 4bit 方法，认为它是内存效率更好的替代方案。
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1200202437080916009) (3 messages): 

- **翻译指标不足以评估 LLM**：`@adrienbufort` 指出，常用于翻译性能评估的 **BLEU 和 Rouge 指标**对于评估 Large Language Models (LLMs) 或 Instruction Tuning LLMs 并没有用处。

- **类似 ELO 的评估在 LLM 中表现最佳**：`@adrienbufort` 推荐 **[类似 ELO 的评估系统](https://arena.lmsys.org/)**（类似于国际象棋排名），认为它最接近人类对 LLM 评估的偏好，尽管它需要人工输入。他们将其描述为目前可用的最佳方法。

- **多选题与 LLM 评估技术**：他们还提到了 **MMLU** ([多选题评估](https://arxiv.org/pdf/2009.03300.pdf)) 和 **Alpaca Eval** ([Alpaca 评估](https://github.com/tatsu-lab/alpaca_eval)) 作为其他 LLM 评估方法。MMLU 允许明确的对错回答，而 Alpaca Eval 涉及一个 LLM 评估另一个 LLM 的回答。

- **已提供归一化的 Alpaca Eval**：`@akshay_1` 提到 **归一化版本的 Alpaca Eval** 现在已经上市，表明 LLM 评估方法取得了进展。
  

---

### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1200043856981405820) (1 messages): 

- **引入创新的 AI 浏览器查询**：用户 `@sublimatorniq` 展示了一个允许 AI 以 [DOM 节点引用](https://news.ycombinator.com/item?id=39128480) 进行响应的工具，使浏览器查询更具上下文感知能力。该工具与 **MistralAI** 兼容，旨在增强与网页内容的交互。

**提到的链接**：

[no title found](https://news.ycombinator.com/item?id=39128480)：未找到描述

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1200185918250819814) (8 messages🔥): 

- **新成员寻求 RAG 应用见解**：来自 🇬🇹 的新成员 `@xrdg` 询问了关于其 **RAG 应用** 的 **Prompt 结构** 问题，并寻找专门的支持频道。
- **建议使用 DSPy 进行 Prompt 优化**：`@akshay_1` 推荐将 **DSPy** 作为优化 Prompt 结构的工具，得到了 `@xrdg` 的认可。
- **提供 RAG 技术栈详情**：在 `@akshay_1` 的进一步询问下，`@xrdg` 透露其 RAG 技术栈由 **langchain, chroma 和 Mistral 7B** 组成，并分享了一个关于 Mistral 7B Prompt 指引的[链接](https://www.promptingguide.ai/models/mistral-7b)，其中包含示例、技巧和相关阅读材料。
- **强调 RAG 技术栈的优化潜力**：`@akshay_1` 观察到 `@xrdg` 的 RAG 技术栈可以进一步优化，并询问该项目是**业余项目还是计划用于生产环境**。

**提到的链接**：

[Prompt Engineering Guide](https://www.promptingguide.ai/models/mistral-7b)：Prompt Engineering 全面概述

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1200021798826278983) (35 messages🔥): 

- **提前停止（Early Stopping）困境依然存在**：用户 `@digitalphotographer` 和 `@sublimatorniq` 交流了关于 Mistral **提前停止问题** 的经验，指出即使在方括号中没有控制 Token 的情况下也会发生该问题。`@digitalphotographer` 进一步澄清他们的 Prompt 是纯字符串，不含控制序列或特殊字符。

- **建议针对提前停止问题寻求技术支持**：`@mrdragonfox` 建议 `@digitalphotographer` **联系 Mistral 支持团队** (support@mistral.ai) 并提供日志/完整示例以获取进一步帮助。`@sophiamyang` 也提供了帮助，请求提供可复现的示例以便团队调查。

- **检测到账单页面 Bug**：用户 `@ewanhc` 和其他人发现了一个 Bug，即账单页面上的**每月使用限制**在刷新后会恢复为 €150，尽管尝试设置了不同的限制。`@ethux` 和 `@fersingb` 确认遇到了同样的问题，`@sophiamyang` 建议联系支持团队并提供详细信息。

- **Mistral API 托管位置咨询**：`@loicboutet` 询问了 **Mistral API 的托管位置**。`@mrdragonfox` 进行了说明，`@loicboutet` 随后通过隐私页面进一步确认，托管地在**欧洲，具体是瑞典的 Azure**。

- **发现 "max_tokens" 参数的 Bug**：`@mrxavierx` 报告了 Mistral API 端点 `/completion` 的一个 **Bug**：当 `max_tokens` 字段设置为 1 时，会导致 500 内部服务器错误，而不是返回 1 个 Token 的响应或有意义的验证错误。已在 GitHub 上为此创建了 Issue：[BUG: API /completion endpoint returns 500 (server error) when sending "max_token" = 1](https://github.com/mistralai/mistral-src/issues/122)。

**提到的链接**：

- [BUG: API /completion endpoint returns 500 (server error) when sending &quot;max_token&quot; = 1 · Issue #122 · mistralai/mistral-src](https://github.com/mistralai/mistral-src/issues/122)：当我测试 API 端点 /completion 时，发现当 &quot;max_tokens&quot; 正文参数设置为 1 时存在一个 Bug。它没有返回 1 个 Token 的响应或验证错误，而是……
- [no title found](https://console.mistral.ai/billing/)：未找到描述

  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1200057056078594111) (29 条消息🔥): 

- **寻找 Meta SAM 的微调代码**：`@the_alt_man` 询问了 Meta 的 SAM (*Segment Anything*) 模型的微调代码库，但由于原始代码库缺乏此类功能，最终使用了 **AutoGluon**。他们强调 AutoGluon 使用了 Lightning，这适用于 GPU 但不支持 TPU。

- **探索无 Infiniband 的多节点训练**：`@elyxlz` 询问了在没有 Infiniband 的情况下进行多节点训练的可行性，并考虑每隔几个训练步进行一次合并策略。讨论引导至一篇关于[分布式优化 (DiLoCo)](https://arxiv.org/abs/2311.08105) 的共享论文，该论文可能解决了这一问题。

- **寻找 The Pile 数据集访问途径**：`@sk5544` 寻求访问 **The Pile** 数据集的指导，在 `@stellaathena` 和 `@elyxlz` 的私下讨论和链接引导下获得了帮助，这体现了社区对信息共享的支持。

- **模拟时钟数据集挑战**：`@stellaathena` 向 `@pinconefish` 提出了一个创意项目，建议创建一个包含显示不同时间的模拟时钟数据集，用于域外泛化 (out-of-domain generalization) 研究。该数据集特别关注显示 10:10 的时钟，这作为训练文本生成图像 (T2I) 模型的起点引起了兴趣。

- **通过合成数据集探索训练质量和数据缺陷**：在模拟时钟的讨论之后，`@wonkothesensible` 建议通过 Checkpoints 分析来探索潜在模型的模式崩溃 (mode collapses)，并建议创建一个显示各种时间的时钟合成数据集，以用于更广泛的训练和缺陷识别。

**提到的链接**：

[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)：大语言模型 (LLM) 已成为机器学习许多应用中的关键组件。然而，训练 LLM 的标准方法需要大量紧密互连的加速器...

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1200008066314285086) (125 条消息🔥🔥): 

- **字节级 Transformer 引发辩论**：由 `@main.ai` 发起的关于 ByT5 等字节级 Transformer 效率的讨论，引发了与 `@the_random_lurker` 之间关于序列长度不公平比较以及 Mamba 等模型中额外上下文实际益处的进一步对话。*MambaByte 论文*中涉及的话题转变为比较 Token 与字节序列长度的有效性和公平性。

- **探索 LLM 的代理微调 (Proxy-Tuning)**：`@digthatdata` 分享了一种名为 [proxy-tuning](https://arxiv.org/abs/2401.08565) 的新方法，该方法提出了一种轻量级的解码时算法，旨在利用较小的模型作为代理来微调大型语言模型 (LLMs)。此方法旨在缩小直接微调模型与黑盒对应模型之间的性能差距，而无需直接访问模型权重。

- **Mamba 被拒稿引发关注**：围绕 Mamba 论文被拒（评分为 10 分制下的 8/8/6/3）的讨论，促使 `@stellaathena` 等人对元评审 (meta review) 过程提出批评。这次拒稿引发了关于学术界影响以及追求 SOTA 基准测试的对话。

- **国际象棋引擎的优越性与 AI 进展**：讨论围绕各领域的当前 SOTA 展开，包括国际象棋，其中 `@clockrelativity2003` 指出 Stockfish 仍是顶级竞争者。`@stellaathena` 向成员发起挑战，要求列举多个领域的近期 SOTA，突显了 ML 社区在了解最新进展方面的认知差距。

- **AI 驱动的国际象棋演变**：`@clockrelativity2003` 讨论的一篇博文想象了 2024 年的国际象棋，它被 AI 的进步重新定义，以 Pandora 等新引擎为特色，并辅以用于策略增强的数字工具。这段对话反映了传统游戏与下一代 AI 技术之间的交汇。

**提到的链接**：

- [Transformers and Cortical Waves: Encoders for Pulling In Context Across Time](https://arxiv.org/abs/2401.14267)：Transformer 网络（如 ChatGPT 和其他 Large Language Models (LLMs)）的能力引起了全世界的关注。其性能背后的关键计算机制...
- [MVDream: Multi-view Diffusion for 3D Generation](https://arxiv.org/abs/2308.16512)：我们介绍了 MVDream，这是一种多视图扩散模型，能够根据给定的文本提示生成一致的多视图图像。通过同时学习 2D 和 3D 数据，多视图扩散模型可以...
- [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361)：本文介绍了 MoE-Infinity，这是一种具有成本效益的 Mixture-of-Experts (MoE) 推理系统，实现了激活感知型专家卸载。MoE-Infinity 的特点是序列级专家激活追踪...
- [CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition](https://arxiv.org/abs/2310.11830)：多语言语音处理需要理解情感，由于标注数据有限，这项任务变得困难。CLARA 最小化了对标注数据的依赖，增强了跨语言的泛化能力。
- [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565)：尽管大型预训练语言模型具有通用能力，但它们始终能从进一步的适配中受益，以更好地实现预期行为。然而，微调这些模型已变得越来越...
- [Evaluating the Medical Knowledge of Open LLMs - Part 1 &mdash; MedARC](https://www.medarc.ai/blog/medarc-llms-eval-part-1)：在这篇 MedARC 博客文章中，我们比较了通用型和医学领域特定的 Large Language Models (LLMs)（如 GPT-4、Mistral 和 Llama），并评估了它们在医学 MultiMedQA 任务上的表现...
- [The Quantum Leap of Checkmate: Chess in the Age of AI The year is 2024](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024)：2024 年。机器人漫游街道，全息图在客厅闪烁，自动驾驶汽车以资深出租车司机的优雅穿梭于高峰时段。然而，在简朴的木板上，一个古老的...
- [| bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v3)：未找到描述
- [Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom](https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1)：虽然 AlphaFold2 (AF2) 和 RoseTTAFold (RF) 通过实现高精度的蛋白质结构建模改变了结构生物学，但它们无法模拟共价修饰或相互作用...

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1200153068268961913) (2 条消息): 

- **修复补丁的可能合并**：`@hailey_schoelkopf` 提到在发现其实现中存在意外行为后，正考虑合并一个修复补丁。他们计划在有空时进行尝试。
- **为 lm-evaluation-harness 添加 Weights and Biases 支持**：`@hailey_schoelkopf` 分享了由 `@ayulockin` 提交的 [GitHub pull request #1339](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339)，旨在为 `lm-evaluation-harness` 添加 **Weights and Biases 支持**，并询问 `wandb.py` 文件的最佳存放位置。

**提到的链接**：

[feat: Add Weights and Biases support by ayulockin · Pull Request #1339 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339)：在 #359 中，@parambharat 曾提议添加 W&B 日志支持。然而，那是在重大重构进入之前完成的。作为 lm-evaluation-harness 和 wandb 的用户，我提交了这个 PR ...

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1200163709352427631) (16 条消息🔥): 

- **QLoRA 微调困惑已澄清**：`@kenakafrosty` 询问了关于使用 **QLoRA** 微调 **neoX 20b** 的问题，但被 `@stellaathena` 告知 **GPT-NeoX 库不支持 QLoRA**。Kenakafrosty 实际上使用的是 **trl、transformers 和 peft** 库的组合。
- **明确正确的求助渠道**：在听取了 `@stellaathena` 的建议后，`@kenakafrosty` 意识到了错误并承认了混淆。Stellaathena 建议在 **相关的 GitHub 上提交 issue** 以寻求关于这些库的帮助。
- **Catboy_slim_ 应对测试挑战**：`@catboy_slim_` 讨论了测试 Python、PyTorch 和 CUDA 版本等重大更新所面临的挑战。强调了无法手动测试每个分支的情况，并着重说明了功能测试的必要性。
- **寻求 PyTorch 测试问题的解决方案**：在解决对 torch 代码运行 pytest 的挑战时，`@catboy_slim_` 在 [GitHub 上提交了一个 issue](https://github.com/EleutherAI/gpt-neox/issues/1132)，关于使用 pytest --forked 运行时测试失败的问题，并暗示这可能是一个超出他们控制范围的普遍问题。
- **促进验证的努力**：`@tastybucketofrice` 向 `<@337128969059172353>` 提供了算力访问权限，并邀请 `@catboy_slim_` 也通过私信获取访问权限，旨在支持对所做更改的进一步测试。此外，针对 pytest 问题的担忧，建议参考 **DeepSpeed** 作为在分叉进程（forked processes）中成功将 CUDA 与 pytest 集成的潜在模型。

**提到的链接**：

[Tests fail when run with pytest --forked · Issue #1132 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/1132)：错误描述：当按照 /test/README.md 中的说明使用 pytest --forked 运行测试时，大量测试失败并报错：RuntimeError: Cannot re-initialize CUDA in forked subp...

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1200030911715090472) (87 条消息🔥🔥): 

- **Perplexity Pro 独家功能揭晓**：`@icelavaman` 和 `@mares1317` 等用户讨论了 Perplexity Pro 相比普通版本的优势，强调了诸如几乎无限次的 Copilot 查询、上传图片和文件以供 Claude 2.1 和 GPT-4 等模型探索的功能，以及访问强大 AI 模型的能力。这些信息得到了 [Perplexity Pro FAQ](https://blog.perplexity.ai/faq/what-is-perplexity-pro) 链接的支持。

- **数据保留引发的隐私担忧讨论**：`@emisaurus_hex` 和 `@firesonwires` 对 Perplexity 的隐私政策表示担忧，特别是搜索历史和个人数据会保留至账号注销。专家 `@icelavaman` 澄清说，删除的 Thread 会在 30 天后从服务器中移除，但用户对隐私影响仍持谨慎态度。

- **寻求 Thread 删除政策的澄清**：在对数据保留的困惑中，自称 Perplexity 专家的 `@icelavaman` 向 `@emisaurus_hex` 等用户保证，删除的 Thread 确实会在 30 天后从服务器中删除。然而，用户表示希望 Perplexity 网站上能有更清晰的隐私政策。

- **辩论数据存储的伦理问题**：讨论转向了哲学层面，`@yellephen` 等用户思考了聚合搜索数据的价值，而 `@firesonwires` 等人则对基于搜索历史进行侵入性画像的可能性表示不安。这次对话凸显了数字时代隐私问题的复杂性。

- **技术支持查询活跃社区**：用户就各种话题寻求支持，包括恢复旧的书签 Thread (`@skyhunz`)、Pro 用户的上传文件限制 (`@lukas8a`) 以及应用 Perplexity Pro 兑换码 (`@odobostudio`)。社区和 Perplexity 代表（如 `@icelavaman` 和 `@danielagmz888`）提供了指导，展示了支持团队在平台内的积极参与。

**提到的链接**：

- [Perplexity 收集了关于我的哪些数据？](https://blog.perplexity.ai/faq/what-data-does-perplexity-collect-about-me)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [什么是 Perplexity Pro？](https://blog.perplexity.ai/faq/what-is-perplexity-pro)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Perplexity 博客](https://blog.perplexity.ai/faq/how-does-file-upload-work.)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Perplexity - AI Companion](https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo?utm_content=chrome-link&ref=chrome&utm_medium=google&utm_source=chrome+store&utm_campaign=chrome-extension>)：浏览时随时提问

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1200104475394850866) (4 条消息): 

- **Perplexity 桥接 Google Search 和 OpenAI**：`jsudaniel` 强调了 Perplexity CEO 的独特定位，即融合了 Google Search 和 OpenAI 的能力。他们分享了一个名为 *“我使用 Perplexity 的次数超过了 Google 和 ChatGPT”* 的 [YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)，阐明了为什么在内容创作方面 Perplexity 比其他工具更受青睐。
- **Perplexity 作为 Smartsheet 的学习工具**：`nicknalbach` 分享了使用 Perplexity 学习 Smartsheet 的经验，提到尽管精通 Excel，但转向 Smartsheet 仍具挑战性。他们发现 Perplexity 在为遇到的每个问题提供答案方面非常有帮助，辅助他们构建了 Smartsheet。
- **通过 Perplexity 辅助天文学教学**：`coloradocomplex` 提到使用 Perplexity 帮助解释天文学课程中的概念，展示了 Perplexity 在教育领域的实用性。他们还分享了一个指向 Perplexity 上特定概念解释的 [链接](https://www.perplexity.ai/search/c429e53a-05ed-4af2-9e48-0301d563bee0)，尽管未提及具体讨论的概念。

**提到的链接**：

[我使用 Perplexity 的次数超过了 Google 和 ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)：该视频的主要观点：“我使用 Perplexity 的次数超过了 ChatGPT、BARD 和 Microsoft Copilot，主要有五个原因，包括它在内容创作中的应用……”

  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1200017158302863361) (5 条消息): 

- **网页版表现优于 API**：用户 `@benhirap` 提到，某个工具的**网页版**在编写代码方面的表现显著优于其 **API 版本**。
  
- **对 labs 与 API 差异的好奇**：`@stijntratsaert_01927` 正在寻找有关 **labs** 所使用的默认参数的信息，以便在使用 **API** 时获得类似的结果。

- **重复收费问题尚未解决**：`@aiagileguy` 详细描述了就重复收费问题联系 **support@perplexity.ai** 的经历。尽管已在 1-2 个工作日之前联系，但目前仍未得到解决。
  

---

### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1200176286639870022) (3 条消息): 

- **社区亮点 #42 引发关注**：@lunarflu 指出 HuggingFace 上的内容创作者和帖子有所增加，强调了该平台对 Machine Learning 的专注，以及相比 Twitter 或 LinkedIn 更为安静的特性。他们提到了向其组织添加更多成员的可能性，暗示向对 AI 和 ML 感兴趣的人发出邀请。[加入组织并探索更多](https://huggingface.co/social-post-explorers)。
- **轻量级 Open LLM Leaderboard 发布**：该排行榜涵盖了权重类型、精度、许可证、参数、架构等各个方面，甚至指定了排除项，如合并模型、被标记模型或 MoE 模型。[深入探索 cosmos-arena](https://thenameless.net/cosmos-arena)。
- **LM 可解释性调查亮点**：H. Luo 和 L. Specia 关于 LLMs 可解释性的调查对当前方法进行了分类并讨论了其应用，标志着在理解和利用基于 Transformer 的预训练 LMs 的可解释性方面迈出了重要一步。[阅读全文](https://huggingface.co/posts/gsarti/888341627040205)。
- **内容创作者的创新项目**：从用于测试 `/StanfordAIMI/CheXagent-8b` 模型的 *CheXRay* Space，到为数据集生成图像 Embedding，这些内容展示了该领域的实用工具和进展。同样，像用于 whisperspeech 模型的 *ZeroGPU* 项目、用于添加转向向量（steering vectors）的 Python 模块，以及用于 DuckDB 的 *Text2SQL* 模型，都体现了社区内多样化的探索。
- **使用 TenyxChat-8x7B-v1 进行 AI 对齐**：作为 TenyxChat 系列的一部分，该模型融合了偏好微调（preference tuning）和高级微调技术，旨在成为一个有用的助手，在 MT-Bench 上获得了 8.4 分。[探索 TenyxChat](https://huggingface.co/tenyx/TenyxChat-8x7B-v1)。

**提到的链接**：

- [social-post-explorers (Social Post Explorers)](https://huggingface.co/social-post-explorers)：未找到描述
- [Cosmos Arena](https://thenameless.net/cosmos-arena)：未找到描述
- [@gsarti 在 Hugging Face: "🔍 Today's pick in Interpretability & Analysis of LMs: From Understanding to…"](https://huggingface.co/posts/gsarti/888341627040205)：未找到描述
- [CheXRay - 由 Tonic 创建的 Hugging Face Space](https://huggingface.co/spaces/Tonic/CheXRay)：未找到描述
- [GitHub - TonyAssi/HF-Embed-Images: 为 🤗 Datasets 生成图像 Embedding](https://github.com/TonyAssi/HF-Embed-Images)：为 🤗 Datasets 生成图像 Embedding。通过在 GitHub 上创建账号为 TonyAssi/HF-Embed-Images 的开发做出贡献。
- [@Tonic 在 Hugging Face: "hey there folks , work in progress, but basically celebrating the release of…"](https://huggingface.co/posts/Tonic/220992701457145)：未找到描述
- [not-lain/TunBERT · Hugging Face](https://huggingface.co/not-lain/TunBERT)：未找到描述
- [@mehd-io 在 Hugging Face: "We just released the first Text2SQL model for DuckDB 🦆🧠 You can try it out…"](https://huggingface.co/posts/mehd-io/779023528910338)：未找到描述
- [@Tonic 在 Hugging Face: "👋 Hi there folks, I launched my first competition ! Goal : Use AI to…"](https://huggingface.co/posts/Tonic/783827682062088)：未找到描述
- [@gsarti 在 Hugging Face: "🔍 Today's pick in Interpretability & Analysis of LMs: Model Editing Can Hurt…"](https://huggingface.co/posts/gsarti/256926950283134)：未找到描述
- [ClovenDoug/small_128_all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/ClovenDoug/small_128_all-MiniLM-L6-v2)：未找到描述
- [Deepfake Detection - 由 not-lain 创建的 Hugging Face Space](https://huggingface.co/spaces/not-lain/deepfake-detection)：未找到描述
- [@vicgalle 在 Hugging Face: "Can you merge models of different sizes? ⚗️ Well, yes, if the models are…"](https://huggingface.co/posts/vicgalle/320544784279721)：未找到描述
- [tenyx/TenyxChat-8x7B-v1 · Hugging Face](https://huggingface.co/tenyx/TenyxChat-8x7B-v1)：未找到描述
- [AI Lineage Explorer: 迈向 AI 完整性的一步。](https://huggingface.co/blog/backnotprop/integrity-explorer)：未找到描述

  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199992656353304586) (40 messages🔥): 

- **GPU 选择与模型预训练现状**：`@sachinkannaujiya1998` 寻求关于在 RTX 3080 GPU 上使用哪种 [Hugging Face 模型](https://huggingface.co/docs/text-embeddings-inference/supported_models) 的建议；同时 `@b1gb4ng` 考虑预训练一个 7b 参数模型，但在看到 `@asprtnl_50418` 详细说明的巨大资源需求后重新考虑。`@asprtnl_50418` 建议将微调现有模型作为一种更具成本效益的替代方案，并强调了 [LoRA/QLoRA adapters](https://huggingface.co/docs/peft/conceptual_guides/lora) 和 [Unsloth](https://huggingface.co/blog/Andyrasika/finetune-unsloth-qlora) 的高效性。
  
- **特征提取技术解析**：`@vipitis` 澄清说，特征提取通常涉及使用 BERT 及其衍生模型等 encoder-only 模型创建序列嵌入（sequence embeddings）。建议参考 [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 以探索最先进的模型。

- **LLM 训练与评估数据策略**：`@enka55` 提出了一个实际问题，即在教语言模型关于“焊接（soldering）”的知识时，如何在训练集和评估集之间分配新的、独特的数据，这引发了 `@the_aureo` 关于确保数据多样性以及考虑使用 RAG 等模型来补充学习的建议。

- **关于 HuggingFace 免费计划与 UI 集成的咨询**：`@Ali_k` 询问了 Hugging Face 免费计划在 AI 模型训练方面的能力。同时，`@Sebastian` 寻求有关使用 `chat-ui` 集成 text-generation WebUI 端点的经验。

- **寻求 Benchmark 之外的模型评论**：`@green_eye` 对仅根据 Benchmark 选择模型感到沮丧，表达了对能够提供模型优缺点以及超越单纯数字的上下文性能见解的评论的渴望。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1tPiTnqk2tMwYLhehS9qVPkcQ9J0gTVv2?usp=sharing)：未找到描述
- [支持的模型与硬件](https://huggingface.co/docs/text-embeddings-inference/supported_models)：未找到描述
- [MTEB Leaderboard - mteb 的 Hugging Face Space](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述
- [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b#hardware-and-software)：未找到描述

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1200218914693582849) (1 messages): 

- **寻找数据集评估框架**：`@rosebei3ngan3g` 讨论了缺乏用于评估训练 LLM 的数据集的全面框架，并质疑进行此类评估的最佳方法。这突显了当前主要关注模型评估的方法论中的空白。
  

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1200037182090530897) (7 条消息): 

- **深入探讨 HuggingFace 的 Dataset Cards**：`@andysingal` 分享了一个有趣的 [GitHub 项目](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis)，用于分析 **HuggingFace dataset cards**。该项目旨在对数据集文档进行大规模分析，有助于 AI 社区理解数据集的叙述和用法。
  
- **学习 UD 与音频 ML 探索**：`@pacificvoltage` 开启了一段自学之旅，参考了一本 [通用设计 (UD) 学习入门书](https://udlbook.github.io/udlbook/)，并在 YouTube 上发现了一个引人入胜的 **Machine Learning Street Talk** [乔姆斯基访谈](https://www.youtube.com/watch?v=axuGfh4UR9Q&t=8412s)，讨论了利用 Deep Fake 技术进行访谈修复。

- **新型 LLM 文本检测器发布**：`@tea3200` 介绍了一篇名为 [“Binoculars” 的预印本论文](https://arxiv.org/abs/2401.12070)，该论文提出了一种**新型大语言模型 (LLM) 检测器**，通过对比评分在识别机器生成文本方面达到了令人印象深刻的准确率。该技术在误报率极低的情况下，对生成样本的检测率超过 90%。

- **Flutter 通过 ONNX 进军 AI 领域**：`@akindelemichael` 重点介绍了一个值得关注的 [GitHub 仓库](https://github.com/gtbluesky/onnxruntime_flutter)，旨在将 **ONNX runtime 与 Flutter** 连接起来，从而实现在不同平台的 Flutter 应用中集成 ONNX 模型。正如 `@osanseviero` 在后续回复中所述，该仓库顺应了 Flutter 在 AI 应用中日益增长的使用趋势。

- **用于 HuggingFace Inference APIs 的 Flutter SDK**：在提到 Flutter 在 AI 领域不断扩大的作用后，`@osanseviero` 详细介绍了一个新项目——[用于 HuggingFace Inference APIs 的 Flutter SDK](https://github.com/shivance/huggingface.dart)，它支持 NLP API，并突显了 Flutter 在跨平台 AI 应用开发方面的潜力。这一进展标志着在使 AI 更加普及和开源方面迈出了重要一步。

**提及的链接**：

- [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070)：用 Binoculars 识别 LLM：机器生成文本的零样本检测。检测现代大语言模型生成的文本被认为很困难，因为 LLM 和人类都能表现出广泛的复杂行为。然而，我们发现一种基于对比评分的...
- [@shivance on Hugging Face: &quot;Hi Community! I&#39;m ecstatic to announce Flutter SDK for HuggingFace Inference…&quot;](https://huggingface.co/posts/shivance/676533662914249)：未找到描述
- [GitHub - gtbluesky/onnxruntime_flutter: A flutter plugin for OnnxRuntime provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile and desktop platforms.](https://github.com/gtbluesky/onnxruntime_flutter)：一个用于 OnnxRuntime 的 Flutter 插件，提供简单、灵活且快速的 Dart API，以便在移动和桌面平台的 Flutter 应用中集成 Onnx 模型。
- [GitHub - mbzuai-nlp/SemEval2024-task8: SemEval2024-task8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection](https://github.com/mbzuai-nlp/SemEval2024-task8)：SemEval2024-task8：多领域、多模型和多语言机器生成文本检测。
- [GitHub - YoungXinyu1802/HuggingFace-Dataset-Card-Analysis: Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on HuggingFace (ICLR 2024)](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis/tree/master)：导航 AI 中的数据集文档：HuggingFace 上 Dataset Cards 的大规模分析 (ICLR 2024)。

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1200059722879991848) (12 messages🔥): 

- **WhisperSpeech 在 HuggingFace 上发布多语言 TTS**：`@tonic_1` 分享了一个新的 [WhisperSpeech](https://huggingface.co/spaces/Tonic/whisperspeech) HuggingFace Space，该 Demo 支持多语言文本转语音（TTS）以及仅需极少音频输入即可创建语音克隆（voice print）。预计更多示例将很快在这个草案版本中发布。
- **Nemo 模型项目博客文章正在筹备中**：继 `@tonic_1` 表示有兴趣启动 Nemo 模型项目后，`@not_lain` 承诺将尽快撰写一篇详细的博客文章。内容将涵盖使用 containers（容器），为社区提供一个急需的详细示例。
- **CheXRay Demo 遇到运行时错误**：`@tonic_1` 报告了其 HuggingFace Space [CheXRay](https://huggingface.co/spaces/Tonic/CheXRay) 的运行时错误，该项目旨在分析胸部 X 光片。该错误反映了医疗影像 AI 应用在开发过程中的现状。
- **呼吁通过社区博客文章扩大影响力**：`@lunarflu` 建议通过 HuggingFace 社区博客分享工作有助于增加触达率，并向 `@mateomd_dev` 指出了一个进行详细技术分享的潜在机会。对话暗示了 HuggingFace 上由社区驱动的详细内容正变得日益重要。
- **wav2vec2-bert 模型发布备受期待**：`@yehors` 宣布将于明天发布基于 Common Voice 10 的新模型 wav2vec2-bert。该模型有望增强语音 AI 技术。

**提及的链接**：

- [CheXRay - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/CheXRay)：未找到描述
- [WhisperSpeech - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/whisperspeech)：未找到描述
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers)：未找到描述
- [Hugging Face – Community Blogs](https://huggingface.co/blog/community)：未找到描述

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1200020728574132275) (3 messages): 

- **Google 的 Lumiere 展示了令人印象深刻的 T2V 能力**：`@fishie22` 强调了 **Google 的 Lumiere** 在 **T2V (Text-to-Video)** 模型中的突破性方法。Lumiere 利用 **Space-Time UNET** 在空间和时间上对信号进行下采样，能够以 16fps 生成 80 帧视频，在视频生成中展现了可能无与伦比的时间一致性。这里是详细介绍其方法和结果的综合研究：[阅读研究报告](https://arxiv.org/abs/2401.12945)。

- **别急，慢慢来！**：`@lunarflu` 向 **Isamu** 传达了耐心和支持，强调了无压力、鼓励性的社区氛围。

- **关于 Benchmarking 的 Medium 文章获得好评**：`@starsupernova` 分享了他们对一篇关于 Benchmarking 的 **Medium 文章** 的热情，称其“非常棒”。未提供文章具体内容或重点的详细信息。

**提及的链接**：

[Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945)：我们介绍了 Lumiere —— 一种文本到视频（text-to-video）扩散模型，旨在合成展现真实、多样且连贯运动的视频，这是视频合成中的一个关键挑战。为此，我们……

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 

spikespiegel5112: 如何在本地加载 LoRA 模型？
  

---

### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1200164215164518431) (5 条消息): 

- **探索高级 LMM 技术**：用户 `besiktas` 询问在训练新的 LMM 时，选择 **idefics/flamingo resampler/cross-attention** 而不是 linear projection/pretrained vision encoder 是否有特定原因。聊天中未提供直接回复。
  
- **Swift API for Vision AI 介绍**：`ahmed3ibrahim` 分享了他使用 **Swift API** 进行 vision AI 的经验，指出其在单个请求中处理多张图片的能力。他提供了 API 链接：[Gemini Pro Vision AI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/) 并强调了其特性，包括 **8.5/10 的普及度**、**5,846ms 延迟**、**99% 服务水平**以及 **100% 健康检查**。

- **关于 CVPR2024 投稿的咨询**：用户 `iloveh8` 询问如何获取提交给 **CVPR2024** 的**所有论文**（包括被录用和被拒绝的），并询问频道中是否有人提交了论文。聊天中没有关于此问题的回复。

**提到的链接**：

[Gemini Pro Vision AI API Documentation (swift-api-swift-api-default) | RapidAPI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/)：未找到描述

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1200003453414559744) (15 条消息🔥): 

- **TorToiSe TTS 质量达到新高度**：`@mr_nilq` 强调了通过 Coqui 提供的 **TorToiSe TTS**，尽管由于其复合架构导致性能较慢，但其质量和一致性非常出色。实现 5 倍加速的修改版本可以在[这里](https://github.com/152334H/tortoise-tts-fast)找到。
- **为 QA 二进制文件训练 AI 选择合适的工具**：`@ysk.dev` 正在探索训练一个拥有约 10,000 个问答对的 AI 的选项，在 Amazon Lex 和 VDS 之间权衡。他担心 Colab Pro Plus 是否足以处理长答案，并咨询了运行后端服务器的合适机器配置。
- **排查 Transformers 中的 `TFTrainer` ImportError**：`@srovnbh` 在使用 `transformers` 包时遇到了 `TFTrainer` 的 ImportError。即使在社区的帮助下，尝试通过在 `4.36.2` 和 `4.37.1` 版本的 `transformers` 之间切换来解决问题的努力也宣告失败。
- **关于信任“黑盒”模型的即将举行的演讲**：`@vipitis` 分享了一个关于评估不可公开检查的“黑盒”模型可信度的即将举行的演讲链接。演讲详情见[此处](https://talks.cam.ac.uk/talk/index/211336)。
- **Windows 上 Bits and Bytes 的兼容性问题**：`@kingpoki` 在使用某些功能时遇到问题，最终发现根本原因是与 Windows 不兼容。这提醒了操作系统对软件适用性和功能的影响。

**提到的链接**：

[talks.cam : Replicating and auditing black-box Language Models.](https://talks.cam.ac.uk/talk/index/211336)：未找到描述

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 条消息): 

spikespiegel5112: 如何在本地加载 LoRA 模型？
  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1200238081865941012) (1 条消息): 

- **Gradio 4.16 发布，带来令人兴奋的新特性**：`@abidlabs` 宣布发布 **`gradio 4.16`**，强调了重大更新，包括 **对 Polars Dataframe 的原生支持**、**将 Gallery 组件作为输入的能力**、**低延迟聊天机器人的更快流式传输**，以及**为自定义组件自动生成的文档**。这次重大更新旨在提升 Gradio 体验，更多详情见 [changelog](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md)。

**提到的链接**：

[gradio/CHANGELOG.md at main · gradio-app/gradio](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md)：使用 Python 构建和分享令人愉悦的机器学习应用。🌟 请 Star 以支持我们的工作！ - gradio-app/gradio

  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199994496906166282) (47 条消息🔥): 

- **LAION2b-en 美学评分不可用**：`@ppwwyyxx` 询问在哪里可以下载 LAION2b-en 美学评分，并链接到了一个根据作者要求已禁用的 [Hugging Face 数据集](https://huggingface.co/datasets/laion/laion2B-en-aesthetic/tree/main)。`@chad_in_the_house` 确认该数据集目前已下线，并建议关注公告以获取后续更新。
- **寻求更好的开源语音聊天界面**：`@jpcl_` 分享了发布一个完整语音聊天界面 Demo 的消息，该界面结合使用了 Whisper、WhisperSpeech 以及开源 LLM，旨在实现更低的延迟和更自然的对话流。他们表示希望改进目前使用的 LLM（Dolphin 2.6 Phi-2），并邀请大家合作优化该项目，详情已发布在 [Hacker News](https://news.ycombinator.com/item?id=39130140) 上。
- **为 VA 的 AI 技术冲刺赛寻找队友**：`@ninjaa2377` 正在寻找个人或小团队加入 VA 的 AI 技术冲刺赛（AI Tech Sprint），该挑战赛专注于临床就诊记录的环境听写（Ambient Dictation），一等奖奖金为 30 万美元。该竞赛旨在推动 AI 在医疗保健领域的能力，参赛者必须是美国公民或居民（U.S. persons）（[AI Tech Sprint Challenge](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)）。
- **关于版权侵权与 AI 模型的辩论**：`@pseudoterminalx` 讨论了在版权侵权避风港运营的情况，批评了一些低估地方政府自主权和对外国影响抵御能力的实体。他们分享了对当地做法的见解，包括有线电视公司使用盗版美国频道，但未提及具体地点。
- **关于 LLM 在娱乐之外用途的讨论**：`@SegmentationFault` 对本地 LLM 普遍用于娱乐而非生产力工具表示遗憾，引发了关于 OpenAI 主导地位以及对更多实际应用需求的讨论。对话涉及了人类历史行为以及 LLM 使用的未来。

**提到的链接**：

- [未发现标题](https://news.ycombinator.com/item?id=39130140)：未发现描述
- [laion/laion2B-en-aesthetic at main](https://huggingface.co/datasets/laion/laion2B-en-aesthetic/tree/main)：未发现描述
- [Reddit - 探索一切](https://www.reddit.com/r/DefendingAIArt/comments/19djc0a/)：未发现描述
- [ 欧盟委员会 🇪🇺 在 Instagram 上："好样的 Fluffy，你确实没看错新闻！今天我们提出了相关措施，允许欧洲 AI 初创企业和中小企业利用我们的高性能计算（High-Performance Computing）能力来训练他们的模型。](https://www.instagram.com/reel/C2e0qkNqqu7/?igsh=MXc2MzB2ZGo3dXUwOQ==)：8.7 万个赞，672 条评论 - europeancommission 于 2024 年 1 月 24 日发布："好样的 Fluffy，你确实没看错新闻！今天我们提出了相关措施，允许欧洲..."
- [Challenge.Gov](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)：Challenge.Gov 是支持由美国联邦政府赞助的奖金挑战和竞赛的官方 GSA 政府网站。在这里，联邦机构为...提供奖金。

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1200069735388287056) (38 条消息🔥): 

- **Byte-Level Transformers 获得乐观评价**：`@marianbasti` 在阅读了一篇 [研究论文](https://arxiv.org/pdf/2401.13660.pdf) 后，对 Byte-Level Transformers 表达了审慎的乐观态度，这可能预示着 Transformer 模型能力的重大转变。
- **Text-to-Image Diffusion 与 ID 保持方面的创新**：`@vrus0188` 分享了两个突破性项目：用于精通 Text-to-Image Diffusion 的 [RPG-DiffusionMaster](https://github.com/YangLing0818/RPG-DiffusionMaster) 和用于最先进的免微调 ID-Preserving 生成的 [InstantID](https://github.com/InstantID/InstantID)，展示了 AI 图像生成技术的快速进步。
- **AI 图像生成远未达到极限**：根据 `@vrus0188` 的说法，每周出现的 4-6 篇新论文或工具表明，AI 图像生成技术远未达到其理论极限，反驳了任何关于该领域增长潜力的怀疑。
- **进入 AI 领域的挑战与成本**：`@chad_in_the_house` 等人讨论了 AI 图像生成等领域不断演变的准入门槛，从开展此类项目的实用性到训练 Stable Diffusion 等模型相关的巨大成本（估计在 500 欧元到 180,000 欧元之间）。
- **受生物启发模拟语言习得的探索**：`@chad_in_the_house` 讨论的一篇 [近期论文](https://arxiv.org/abs/2306.15364) 提出了一个生物学上合理的语言器官模型，该模型在没有反向传播的情况下学习语言，标志着可能从传统机器学习方法转向更高效、受生物启发的方法。

**提到的链接**：

- [The Architecture of a Biologically Plausible Language Organ](https://arxiv.org/abs/2306.15364)：我们展示了一个模拟的生物学上合理的语言器官，由风格化但真实的神经元、突触、大脑区域、可塑性和简化的感官知觉模型组成。我们通过...展示了...
- [PALP: Prompt Aligned Personalization of Text-to-Image Models](https://arxiv.org/abs/2401.06105)：内容创作者通常旨在利用个人主体创建个性化图像，这超出了传统 Text-to-Image 模型的能力。此外，他们可能希望生成的图像...
- [GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (PRG)](https://github.com/YangLing0818/RPG-DiffusionMaster)：Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (PRG) - GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Pl...
- [GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥](https://github.com/InstantID/InstantID)：InstantID : 秒级 Zero-shot Identity-Preserving 生成 🔥 - GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1200121061669343343) (1 条消息): 

- **LLMCompiler 网络研讨会预告**：`@jerryjliu0` 宣布了一场临时的网络研讨会，邀请作者 Sehoon Kim 和 Amir Gholami 讨论 **LLMCompiler**，这是一个用于 Agent 中并行函数调用的新框架。研讨会旨在涵盖该编译器相比于之前的顺序推理框架（如 ReAct）的优势，强调 **长期规划** 和 **并行化能力**。分享了 **LLMCompiler 论文** 及相关资源的链接：[LLMCompiler 论文](https://arxiv.org/pdf/2312.04511.pdf)、[LlamaPack](https://llamahub.ai/l/llama_packs-agents-llm_compiler?from=llama_packs) 和 [GitHub notebook](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_pac)。

**提到的链接**：

[LlamaIndex Webinar: Efficient Parallel Function Calling Agents with LLMCompiler · Zoom · Luma](https://lu.ma/lf9iroox)：LLM 擅长推理和采取行动。但之前的 Agent 推理框架（如 ReAct）主要侧重于顺序推理，导致更高的...

  

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1200128299385499748) (7 条消息): 

- **使用 @seldo 的指南构建 Slack Bot**：宣布了一个新的 **开源软件 (OSS) 仓库**，其中包含由 `@seldo` 编写的关于如何构建一个能从对话中学习的 `@SlackHQ` 机器人的 **逐步指南**。该指南可以在 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1750568734140567611) 中找到。
  
- **LlamaIndex 与 Zilliz Universe 建立合作伙伴关系**：LlamaIndex 宣布与 `@zilliz_universe` 合作，将 **Zilliz Cloud Pipeline** 集成到 LlamaIndex 中，提供具有多租户支持的可扩展检索服务。更多信息请参阅他们的 [客座文章](https://blog.llamaindex.ai/?source=post_page-----4879e9768baf--------------------------------)。

- **宣布对 @OpenAI 的 Embedding 模型提供 Day 0 支持**：LlamaIndex 发布了 v0.9.38 版本，为 @OpenAI 最新的 Embedding 模型提供 **Day 0 支持**。有关此版本的详细信息在 [推文](https://twitter.com/llama_index/status/1750638684297368000) 中分享。
  
- **LlamaIndex 承诺轻松实现 Prompting**：LlamaIndex 强调了他们提供 **高效 Prompting** 的特性，使用户无需在自定义上费力，尽管自定义仍然是一个选项。更多细节可以在 [最近的推文](https://twitter.com/llama_index/status/1750654272361083054) 中找到。
  
- **LlamaIndex.TS 提供 TypeScript 支持**：宣布了新的 TypeScript 版本 **LlamaIndex.TS 0.1.0 版本**，支持 @OpenAI 最新的 Embedding，并感谢 `@yi_ding` 的快速实现。此外，正如 [后续推文](https://twitter.com/llama_index/status/1750673214840394198) 中详述，此更新还包括对 **@qdrant_engine** 的支持。

**提到的链接**：

- [llama-index](https://t.co/kyIoTUaeuD)：LLM 与您的数据之间的接口
- [使用 LlamaIndex 和 Zilliz Cloud Pipelines 构建可扩展的 RAG 应用](https://t.co/luDjSgiokt)：简介

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1200040486535176283) (38 条消息🔥): 

- **LlamaIndex 中没有用于 TextGenerationInference 的 LLM**：用户 `@wizboar` 询问 LlamaIndex 是否有用于 TextGenerationInference 服务器的 LLM，对此 `@cheesyfishes` 确认目前没有可用的，并建议使用 langchain 工具配合 wrapper。
  
- **使用 `similarity_top_k` 定制 Chat Engine**：`@richard1861` 询问是否可以为 LlamaIndex 中的 Chat Engine 定义特定参数 (`similarity_top_k=3`)，`@whitefang_jr` 提供了一个代码片段，演示了如何直接在引擎中进行配置。

- **保险领域查询的实现挑战**：用户 `@lancerninja` 详细描述了一个复杂的用例，涉及在保险领域使用相似术语而非精确匹配时的查询重写，而 `@cheesyfishes` 建议使用 LLM 进行查询重写，但指出针对其特定场景目前缺乏离线解决方案。

- **对 OpenAI 新 Embedding 模型的兴奋**：`@ayfri` 分享了 OpenAI 宣布新 Embedding 模型和 API 更新的链接，表达了对即将到来的支持的期待，对此 `@cheesyfishes` 保证支持将很快发布，强调了社区对持续改进的热情。

- **关于在查询中扩展上下文的自定义 Prompt 指南**：`@shri_j` 寻求关于使用 LlamaIndex 和 OpenAI 查询未包含在提供上下文文档中的信息的建议，`@cheesyfishes` 建议修改默认 Prompt，指示模型考虑或扩展到给定上下文之外，并链接到了特定的文档以提供帮助。

**提到的链接**：

- [新的 Embedding 模型和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。
- [使用模式 - LlamaIndex 🦙 0.9.39](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts)：未找到描述

  

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1200109478939480164) (5 条消息): 

- **Zep 通过生产级工具增强聊天机器人**：用户 `@yoursaviorjesus` 重点介绍了 [Zep](https://docs.getzep.com/)，展示了其在**生产级聊天历史记忆、向量搜索、数据增强**等方面的能力。他们特别询问了其**实体提取（entity extraction）**能力的有效性。
  
- **LlamaIndex 对比 Amazon Kendra**：`@zeekg_46676` 询问 **LlamaIndex** 是一个向量存储（vector store），还是功能更像执行自然语言搜索的 **Amazon Kendra**。 

- **LlamaIndex：灵活的数据编排工具**：针对 `@zeekg_46676` 的提问，`@cheesyfishes` 澄清说 **LlamaIndex** 更接近 **Amazon Kendra**，它能够利用任何向量存储、LLM 或 embedding 模型来管理数据摄取（ingestion）、检索（retrieval）和响应合成（response synthesis）。

- **具有自动知识图谱生成的自学习 RAG**：`@chiajy` 分享了一篇 Substack 文章链接（[哈利波特与自学习知识图谱 RAG 工作流](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning)），详细介绍了 **WhyHow.AI** 的一个 Demo，展示了在《哈利波特》章节上使用 **RAG** 进行**递归检索（recursive retrieval）、自动知识图谱生成**以及**记忆/多跳推理（multi-hop reasoning）**。该 Demo 旨在提高 RAG 准确性，缩短上线时间，并展示了自学习 RAG 的首批示例之一。

**提到的链接**：

- [哈利波特与自学习知识图谱 RAG](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning)：WhyHow.AI 的自学习 RAG 结合知识图谱，为垂直领域 AI 带来准确性和规则——展示了递归检索、记忆以及自动上下文感知的知识图谱构建。
- [Zep - 快速、可扩展的 LLM 应用构建模块](https://docs.getzep.com/)：未找到描述

  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1200087852042698923) (36 messages🔥): 

- **LLM Paper Club 录制政策**：`@kbal11` 提到 LLM Paper Club 的会议没有进行录制，以便参与者能够更自由地分享他们的工作，而无需担心在互联网上曝光。因此，错过的会议没有回放。
- **寻找 RAG 评估工具**：`@joejoej0e` 正在寻找用于实验跟踪和对 RAG 流水线结果进行人工评分的工具，旨在通过人工评估相关性来改进信息检索产品。
- **Prophetic AI 推出 MORPHEUS-1**：`@shivdinho` 分享了一个链接，宣布推出 MORPHEUS-1。据称这是世界上首个用于诱导和稳定清醒梦的多模态生成式超声 Transformer，定于 2024 年春季发布测试版。
- **go-go-golems 的快速开发**：`@slono` 宣布在短短 4 天内开发了 5000 行代码，作为 yaml-custom-tags 实验的一部分，展示了该项目的进度和极高的生产力。
- **Martian 推出 LLM 评估工具**：`@cute_hamster_07119` 介绍了 Martian 的一款新工具，该工具可以根据成本、吞吐量和 TTFT（首字延迟），实时评估应使用哪种 LLM 推理产品，支持 Anyscale, Together, Lepton, Fireworks 等多家供应商。该工具于今日发布。

**提到的链接**：

- [🚨🚨 这么多 YAML 🚨🚨](https://noyaml.com/)：未找到描述
- [Prophetic (@PropheticAI) 的推文](https://x.com/propheticai/status/1750534355242418300?s=46&t=JE84TqLviekDnEt8MAT-Eg)：推出 MORPHEUS-1。世界上首个旨在诱导和稳定清醒梦的多模态生成式超声 Transformer。2024 年春季面向测试用户开放。
- [新嵌入模型和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并即将降低 GPT-3.5 Turbo 的价格。
- [KREA 正在构建人类创造力的下一个前沿 ⚡️](https://cerebralvalley.beehiiv.com/p/krea-building-next-frontier-human-creativity)：此外：联合创始人 Diego 谈论拥抱好奇心和混沌……
- [Reddit - 深入探索一切](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/)：未找到描述
- [LLM 推理供应商排行榜](https://leaderboard.withmartian.com/)：由 Martian 制作的关于 LLM 推理 API 的实时、公正的基准测试。
- [评估方法 - 供应商排行榜](https://docs.withmartian.com/provider-leaderboard/evaluation-methodology)：未找到描述
- [talrid23 (@talrid23) 的推文](https://x.com/talrid23/status/1750463847226388574?s=46&t=XV1VJkM4nCYVU6fROoKkfw)：JSON 格式正成为 LLM 生成输出的事实标准。然而，它是最优格式吗？🤔 我们认为不是——YAML 输出更短、更简单，从而实现更快的推理……
- [go-go-labs/cmd/experiments/yaml-custom-tags at main · go-go-golems/go-go-labs](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/experiments/yaml-custom-tags)：GO GO 实验实验室。通过在 GitHub 上创建账号为 go-go-golems/go-go-labs 的开发做出贡献。
- [可复现性 - 供应商排行榜](https://docs.withmartian.com/provider-leaderboard/reproducibility)：未找到描述
- [GitHub - withmartian/provider-dashboard: Martian LLM 推理供应商排行榜的开源后端](https://github.com/withmartian/provider-dashboard)：Martian LLM 推理供应商排行榜的开源后端 - GitHub - withmartian/provider-dashboard: Martian LLM 推理供应商排行榜的开源后端

---

### Latent Space ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1200014492210319413) (1 条消息): 

- **LLM Paper Club 亚洲分部启动**：`@ivanleomk` 宣布启动首届 **LLM Paper Club 亚洲分部会议**，重点讨论经典论文 *"Attention Is All You Need"*。感兴趣的参与者可以在[此处](https://lu.ma/llm-paper-asia)报名参加未来的活动，并通过 [Discord](https://discord.gg/tPnG5qMu) 加入今天的会议。

- **获取 Latent.Space 活动最新动态**：鼓励参与者点击日历右上方 RSS 图标，并在悬停时点击 "**Add iCal Subscription**" 将其添加到日历中，以随时了解最新的 **Latent.Space** 活动，确保不错过未来的聚会。

**提到的链接**：

- [LLM Paper Club (Asia Edition!) · Luma](https://lu.ma/llm-paper-asia)：更新：已更新 Discord Stage 链接，我们将使用对亚洲时区友好的 Latent.Space x EugeneYan.com LLM Paper Club 版本！本周我们将讨论...
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/tPnG5qMu)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1200030161123414027) (8 条消息🔥): 

- **亚洲 Paper Club 安排了新一期活动**：`@ivanleomk` 宣布下周的亚洲 **paper club** 暂定讨论 *Self-Rewarding Language Models* 论文。他们鼓励成员推荐其他论文或表达演讲意向。 

- **征求反馈**：`@aimuggle` 感谢参与者加入会议，并征求反馈以改进仍处于 **beta 阶段的项目**。

- **关于 Self-Reward 的疑问**：`@stealthgnome` 询问在即将进行的讨论背景下，*self instruct* 是否是 **self-reward** 的输入。

- **美国 Paper Club 的下一期内容**：针对 `@ivanleomk` 关于美国 paper club 下一期议程的询问，`@eugeneyan` 分享了他们将讨论 [**Pythia 论文**](https://arxiv.org/abs/2304.01373)，并重点介绍了详尽的贡献作者名单。

**提到的链接**：

[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)：Large Language Models (LLMs) 在训练过程中是如何发展和演变的？这些模式如何随模型规模变化？为了回答这些问题，我们推出了 Pythia，一套包含 16 个...

  

---



### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1199993210039177247) (2 条消息): 

- **讨论 Mixtral 的 Mergekit 选项**：`@philipmay` 分享了来自 **mergekit 作者** 的 [GitHub issue 链接](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289)，澄清了合并后微调 **Mixtral** 模型的选项，并质疑了 "hidden" 和 "random" 选项对后续微调的有效性。
- **强调 MoE 训练中的辅助损失 (Auxiliary Loss)**：`@bjoernp` 对分享的信息给予了积极回应，强调正确处理辅助损失是 **MoE 训练** 的关键环节。

**提到的链接**：

[Mixtral branch: What option should I choose when I want to do some finetuning after the merge? · Issue #116 · cg123/mergekit](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289)："hidden" 和 "random" 的参数描述并没有准确解释当我想要在之后进行微调时该怎么做。在合并后进行微调是否有效（可行）...

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1200000600281194536) (23 条消息🔥): 

- **高质量数据选择的困境**：`@bjoernp` 分享了一篇[有趣的论文](https://arxiv.org/abs/2401.12926)，指出为“质量”而过滤预训练数据并不总是能提升模型性能。该论文提出了一个数据集选择框架，旨在优化模型性能，而不是遵循传统的数据质量观念。

- **探索 KTO 训练**：`@bjoernp` 和 `@hammadkhan` 讨论了 **Kahneman-Tversky Optimisation (KTO)**，通过考虑理想与非理想补全的二元信号，将其与 **Direct Preference Optimization (DPO)** 进行了比较。他们强调了其优势和实现方面，包括 [Hugging Face 的文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)以及 **Axolotl** 通过 trl 对 KTO 的适用性。

- **使用 KTO 的上下文 AI 方法**：`@hammadkhan` 介绍了 **Kahneman-Tversky Optimisation (KTO)**，这是一种使用正向和反向示例（例如 👍 或 👎）进行训练的新方法。该方法在一篇博客文章中进行了详细阐述，被定位为在生产环境中更新聊天模型的一种更简单且可能更有效的策略。

- **关于 KTO 适用性的辩论**：`@rasdani` 和 `@hammadkhan` 辩论了 **KTO** 在需要比较指令引导的好坏回答场景中的有效性。讨论围绕标签是否需要直接相关，或者 KTO 的灵活性是否允许比 **Direct Preference Optimization (DPO)** 更广泛的使用展开。

- **GPT-4 Turbo 和 GPT-3.5 价格更新**：`@rasdani` 强调了最新的更新，包括 **Embedding V3 模型**、**GPT-4 Turbo** 的发布，以及 **GPT-3.5 Turbo** 的大幅降价。详情分享在 @OfficialLoganK 的[官方推文](https://x.com/officiallogank/status/1750589278709780780?s=46&t=1jtkL4JPu-DUOdo8JC668g)和 OpenAI 博客中。

**提到的链接**：

- [DsDm: Model-Aware Dataset Selection with Datamodels](https://arxiv.org/abs/2401.12926)：在为训练大规模模型选择数据时，标准做法是过滤符合人类数据质量观念的示例。这种过滤产生了定性上干净的数据点，但……
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)：未找到描述
- [来自 Logan.GPT (@OfficialLoganK) 的推文](https://x.com/officiallogank/status/1750589278709780780?s=46&t=1jtkL4JPu-DUOdo8JC668g)：对 @OpenAIDevs 来说是个好消息，我们正在发布：- Embedding V3 模型（small & large） - 更新的 GPT-4 Turbo 预览版 - 更新的 GPT-3.5 Turbo（下周 + 输入 token 降价 50% / 25% 价格……）

---

### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1200004770333741117) (12 条消息🔥): 

- **德国 Jina 模型发布公告**：`@sebastian.bodza` 强调了即将于 Hugging Face 发布 "jinaai/jina-embeddings-v2-base-de" 模型，这可能对排序任务有所帮助。虽然未提及具体发布日期，但预计在“明天”。

- **分享问题生成示例**：`@sebastian.bodza` 在 [GitHub](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md) 上分享了问题生成的示例，标志着该领域工作的开始。

- **在 Embeddings 中使用 Mixtral 和 LLM**：针对 `@philipmay` 的询问，`@sebastian.bodza` 提到在他们的项目中使用 VLLM 运行 4-bit GPTQ 版本的 Mixtral，重点在于创新的问题生成和 Embedding 开发。

- **重点介绍 OpenAI 新的 Embeddings**：Bjoernp 引起了大家对 [OpenAI 新 Embedding 模型和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)的关注，这可能有利于多语言支持，并可能影响未来的开发策略。

- **用于高质量数据生成的 Genie 方法**：Bjoernp 分享了一篇 [arXiv 论文](https://arxiv.org/abs/2401.14367)，该论文提出了一种名为 Genie 的新方法，用于生成高质量、基于内容（content-grounded）的数据，并建议通过自动过滤机制进行质量保证，从而在改进问答对或摘要方面发挥潜在效用。

**提到的链接**：

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)：基于内容生成任务的高质量数据缺乏被认为是推进这些任务的主要障碍。为了填补这一空白，我们提出了 Genie，一种新型的自动方法...
- [GitHub: Let’s build from here](https://github.com)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献、管理 Git 仓库、像专家一样评审代码、跟踪 Bug 和功能...
- [Embedding_Training/README.md at main · SebastianBodza/Embedding_Training](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md)：通过在 GitHub 上创建账号来为 SebastianBodza/Embedding_Training 的开发做出贡献。
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199995853339902003) (5 条消息): 

- **实现有效微调**：`@thomasrenkert` 报告称使用 unsloth 成功微调了 **DiscoLM German 7B v1**。他们对未来基于 Mixtral-Instruct 的 **DiscoLM German 版本**感到兴奋，并强调了 MoE 的显著影响。
- **用于翻译的独特数据集**：针对 `@hammadkhan` 的询问，`@thomasrenkert` 分享了他们使用自己的数据集对模型进行微调，用于将**中古高地德语翻译为现代德语**。
- **社区支持与兴趣**：`@bjoernp` 对 `@thomasrenkert` 的微调成功和数据集更新表示支持和赞赏，显示出积极的社区互动以及对 DiscoLM 新颖应用的兴趣。
  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1200152287843197129) (2 条消息): 

- **OpenAI 发布新的 Embedding 模型和 API 增强功能**：用户 `@potrock` 分享了来自 OpenAI 的[博客更新](https://openai.com/blog/new-embedding-models-and-api-updates)，宣布推出新一代 Embedding 模型、改进的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，以及即将下调的 GPT-3.5 Turbo 价格。该公告重点介绍了两个新的 Embedding 模型、更新的 GPT-4 Turbo 和 GPT-3.5 Turbo 模型、一个新的文本审核模型，并承诺发送到 OpenAI API 的数据将不会用于训练其模型。

- **链接重定向建议**：`@shacrw` 建议关于新 Embedding 模型和 API 更新的公告应该发布在更合适的频道中，并提供了一个[链接](https://discord.com/channels/1168579740391710851/1171903046612160632/1200169179802783754)以重定向对话。这暗示了沟通频道选择上的失误，但未提供更多讨论背景。

**提到的链接**：

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---

### LLM Perf Enthusiasts AI ▷ #[announcements](https://discord.com/channels/1168579740391710851/1168760950803931136/) (1 条消息): 

mat_mto: 谢谢 Jeff！非常喜欢你目前所做的一切工作。
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1200152253890301993) (16 条消息🔥): 

- **OpenAI 发布新模型和工具**：`@potrock` 分享了来自 OpenAI 的[一篇博客文章](https://openai.com/blog/new-embedding-models-and-api-updates)，宣布推出新的 Embedding 模型、GPT-4 Turbo 的更新、新的 Moderation 模型、用于管理 API 使用情况的工具，以及即将下调的 GPT-3.5 Turbo 价格。这些更新旨在为开发者提供更好的性能和成本效益。

- **社区对 Embedding 更新感到兴奋**：在公告发布后，`@nosa_.` 以一句 "well well well" 表达了好奇，`@potrock` 强调了缩短 Embedding（shortened embeddings）功能的吸引力，认为其非常酷。

- **Embedding 模型对比**：`@potrock` 指出，新的大型 Embedding 模型性能略微超过了 bge-large，表明 OpenAI 的最新迭代版本有细微但显著的改进。

- **升级到 OpenAI 新产品的优势**：`@res6969` 分享了升级系统以整合新发布的 Embedding 模型的计划，并表示考虑到坚持使用 OpenAI 解决方案的简单性和有效性，不再认为有必要转向开源选项。

- **探索新 Embedding 的成本效益**：`@shacrw` 和 `@michelcarroll` 讨论了使用具有维度缩减功能的新型大型 Embedding 模型 (v3-large) 的潜在成本收益，考虑了其性能以及对存储和 API 成本的影响。他们权衡了 Embedding 成本与向量数据库（vector database）存储节省之间的平衡，`@michelcarroll` 倾向于选择 v3-large 模型以获得更好的性能并降低存储成本。

**提到的链接**：

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和 Moderation 模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1200045647257153567) (12 条消息🔥): 

- **欢迎加入，quarknova！**：用户 `@quarknova` 是一名在 INRIA 实习的 ENS 学生，正在[研究使用 LangChain](https://github.com/) 开展项目。他们很好奇 GitHub 版本是否足够，还是必须使用商业版本。
- **探索创建 AI 人格**：`@jstansbe` 询问如何在不依赖外部 API 的情况下创建像 Elon Musk AI 这样的“AI 人格”。`@ksolo` 回应称这一过程被称为 **finetuning**（微调），并推荐了一个关于[微调大语言模型的短课](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)以获取实践指导。
- **利用 LangChain 和 Streamlit 快速开发联网搜索聊天机器人**：`@johnnucleus` 分享了使用 LangChain 和 Streamlit 快速创建具备联网搜索能力的聊天机器人的兴奋之情，强调了其高效性和易用性。
- **利用 LLM 生成用于 ML 训练的合成数据**：`@rajib2189` 和 `@johnny2x2` 分别讨论了利用 LLM 为传统机器学习训练和 RAG 生成合成数据（Synthetic Data Generation）的应用。
- **将 PARQUET 文件加载到 LangChain 中**：`@benjaminbascary` 寻求关于在 LangChain 中将 PARQUET 文件作为文档加载的建议，`@johnny2x2` 提供了一个解决方案：使用 pandas 读取文件，并通过 LangChain 社区文档加载器中的 `DataFrameLoader` 进行加载。

**提到的链接**：

[Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)：未找到描述。

  

---

### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1200090590503780512) (3 条消息): 

- **探索 LangServe 示例**：`@veryboldbagel` 引导用户在 GitHub 上探索 **LangServe** 示例，重点介绍了 Readme 中的两个 Agent 示例，以及第三个未列出但可通过其他链接访问的示例。感兴趣的用户可以在[此处](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples)和[此处](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor)找到更多信息和示例。

- **构建自定义 Agent 的指导**：`@veryboldbagel` 为想要定义自定义工具或创建自定义 Agent 的用户提供了详细建议，提到现成的 OpenAI tools Agent 对于自定义工具来说已经足够。对于更复杂的需求，`@veryboldbagel` 建议使用 LCEL 和 LangGraph 以获得更强的表达能力，更多说明可参考[此处](https://python.langchain.com/docs/modules/agents/how_to/custom_agent)和[此处](https://python.langchain.com/docs/langgraph#agentexecutor)。

- **Agent 与流式响应的问题**：`@hiranga.g` 在实现带有历史记录的 Agent 时遇到了一个问题，即未能按预期接收到流式响应，而 Playground 中却可以交付 JSON 对象。为了寻找解决方案，他们根据有关 Agent 和 LangServe bug 的建议尝试使用 `chain.streamLog()`，但未提供有关解决尝试或结果的详细信息。

**提到的链接**：

- [🦜🕸️LangGraph | 🦜️🔗 Langchain](https://python.langchain.com/docs/langgraph#agentexecutor.): ⚡ 以图形方式构建语言 Agent ⚡
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。
- [langserve/examples/configurable_agent_executor at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1200076122575085578) (2 条消息): 

- **询问上下文感知能力**：用户 `dejoma` 询问 LangChain AI 是否使用了当前访问网页的上下文，表现出对 AI 上下文感知能力的关注。

- **LangChain 在制造业中的创新 SQL 应用**：`johnny2x2` 分享了在制造业背景下实现 LLM 自动化的经验，使用 **Mixtral 7B v2 5Q 32K** 通过 ERP SQL 数据分析逾期客户订单。他们发现一种行之有效的方法：在数据库中创建精选视图供 SQL 链使用，以高效管理大型数据库，并转向在专门的任务循环中将 SQL 查询作为工具使用，从而获得更有效的反馈。
  

---



### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1200325637450235934) (3 条消息): 

- **宣布重大 LLM 版本发布**：`@simonw` 透露即将发布 **LLM 更新**，旨在大幅升级底层的 openai 库。鼓励爱好者测试新版本，说明详见 [GitHub](https://github.com/simonw/llm/issues/325#issuecomment-1911533536)。
  
- **通过 0.13 里程碑展望未来**：对于好奇更新内容的用户，`@simonw` 分享了 GitHub 上的 [0.13 Milestone](https://github.com/simonw/llm/milestone/8) 链接，提供了对即将到来的增强功能的见解。

- **寻求 Readline 问题帮助**：`@simonw` 正在寻求帮助以解决 LLM chat 中与 readline 相关的 bug，即方向键输出 ANSI 码而不是移动光标。贡献者可以在 [GitHub](https://github.com/simonw/llm/issues/376) 上找到有关此问题的更多详细信息。

**提到的链接**：

- [0.13 Milestone · simonw/llm](https://github.com/simonw/llm/milestone/8): 从命令行访问大语言模型 - 0.13 Milestone · simonw/llm
- [llm chat - readline problems still present · Issue #376 · simonw/llm](https://github.com/simonw/llm/issues/376): 当我打开 llm chat 时，我希望使用左右方向键可以移动光标，但结果却是屏幕上打印出了讨厌的 ANSI 码。$ llm chat Chatting with gpt-4 Type 'exit...
- [Upgrade for compatibility with OpenAI 1.0 library · Issue #325 · simonw/llm](https://github.com/simonw/llm/issues/325#issuecomment-1911533536): 当前情况：成功安装 openai-1.0.1 $ llm -m gpt-4-turbo 'hi' 错误：module 'openai' has no attribute 'ChatCompletion'