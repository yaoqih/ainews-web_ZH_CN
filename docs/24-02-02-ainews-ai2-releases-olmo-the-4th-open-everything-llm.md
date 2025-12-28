---
companies:
- ai2
- allenai
- mistral-ai
- tsmc
- asml
- zeiss
date: '2024-02-03T03:35:10.019799Z'
description: '**AI2** 在 2024 年因其全新的 **OLMo** 模型而备受关注，其中包括 1B 和 7B 尺寸，以及即将推出的 65B 模型，该系列强调类似于
  **Pythia** 的开放且可重复的研究。**Miqu-70B** 模型，特别是 Mistral Medium 变体，因其自我纠错能力和速度优化而受到赞誉。**TheBloke**
  Discord 社区的讨论涵盖了编程语言偏好、大模型的显存（VRAM）限制，以及使用 **Distilbert-base-uncased** 进行的微调实验。**Mistral**
  Discord 社区则重点讨论了影响 **台积电 (TSMC)**、**阿斯麦 (ASML)** 和 **蔡司 (Zeiss)** 半导体生产的 **GPU 短缺**挑战、开源与闭源模型的辩论，以及包括针对低资源语言的
  **LoRA** 在内的微调技术。社区见解还涉及了嵌入分块策略和 JSON 输出的改进。'
id: 134d580b-d650-41c0-bdf3-3118898ad27c
models:
- olmo-1b
- olmo-7b
- olmo-65b
- miqu-70b
- mistral-medium
- distilbert-base-uncased
original_slug: ainews-ai2-releases-olmo-the-4th-open-everything
people:
- nathan-lambert
- lhc1921
- mrdragonfox
- yashkhare_
- gbourdin
title: AI2 发布 OLMo —— 第四个全开放（open-everything）大语言模型。
topics:
- fine-tuning
- gpu-shortage
- embedding-chunking
- json-generation
- model-optimization
- reproducible-research
- self-correction
- vram-constraints
- programming-languages
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月1日的 AI Discord 动态。我们为您检查了 **21** 个公会、**312** 个频道和 **5874** 条消息。预计节省阅读时间（按 200wpm 计算）：**483 分钟**。我们在处理包含大量链接的消息时遇到了稳定性问题（感谢 @yikesawjeez...），我们必须想办法解决。

正如 Nathan Lambert 在 [Latent Space 访谈](https://www.latent.space/p/rlhf-201) 中所预告的，在新领导层的带领下，今年我们将看到 AI2 更多地出现。其首批成果现已发布，即 [OLMo](https://allenai.org/olmo/olmo-paper.pdf) (Open Language MOdels) —— 一个 1B 模型和一组 7B 模型，65B 模型也即将推出。

 
![image.png](https://assets.buttondown.email/images/bbb5cb1e-b02f-41f7-9da0-31dce096f27f.png?w=960&fit=max)
 

如果你喜欢非企业化的视角（我们很喜欢），[Nathan 的 Substack](https://www.interconnects.ai/p/olmo) 提供了相关见解。另外有趣的是，[通过磁力链接发布模型这种套路](https://twitter.com/natolambert/status/1753063313351835941) 依然经久不衰。

在 LS Discord 中，我们有幸与 Nathan 进行了更详细的讨论，包括发布“孪生” AMD 模型的奇怪选择、在基准测试中排除 Mistral 7B 等话题。

 
![image.png](https://assets.buttondown.email/images/09aedd23-595b-48fd-bc44-6e12b56d7df4.png?w=960&fit=max)
 

我们在本周的 Paper Club 中恰好讨论了 Pythia（[2023 年十大 AI 论文之一](https://magazine.sebastianraschka.com/p/10-ai-research-papers-2023)），Nathan 也同意 OLMo 在致力于可复现和完全开放研究方面，可以被视为 Pythia 的精神继承者。

希望这是 2024 年更多此类成果的开始。

---

**目录**

[TOC] 


# 第一部分：Discord 高层级摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **Miqu-70B：令人惊喜的自我修正者**：在 **general** 讨论中，Miqu-70B（尤其是 Mistral Medium 版本）因其在回答过程中自我修正的能力而受到赞誉。高级用户报告称，经过优化后，模型平均速度达到每秒 17 个 token，表现优于 Mixtral。

- **语言大对决：C++ vs. Rust vs. Go**：**general** 频道中的工程师们就 C++、Rust 和 Go 等编程语言的优劣展开了激烈辩论。由于易于理解，大家似乎更倾向于简单、更易管理的语言。

- **Miqu-1-70B-SF 的 VRAM 挑战**：在 **characters-roleplay-stories** 频道中，人们尝试将 Miqu-1-70B-SF-4.25bpw-h6-exl2 适配到有限的 VRAM 中，引发了关于潜在解决方案（包括硬件升级）的讨论。

- **类别难题：Distilbert-base 来救场？**：**training-and-fine-tuning** 频道的一位用户尝试使用 **Distilbert-base-uncased** 从文本中预测层级代码，观察到高层级预测的准确率尚可，但在更细微的类别区分上存在困难。

- **文件层级之争：搜索超越结构**：**coding** 频道的用户讨论了对强大 **搜索功能** 的日益依赖，而非传统的文件夹组织。良好的搜索被认为比标签或目录等传统方法更有效，尤其是在 nvpy 和 simplenote 等软件中。



---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **应对 GPU 荒**：工程师们讨论了由于复杂的半导体制造工艺和长达数月的晶圆生产周期导致的 **GPU 短缺**。这限制了生产规模的扩大，**TSMC, ASML, 和 Zeiss** 等公司承担了主要的协作压力。

- **开源 vs. 闭源**：一场激烈的辩论围绕开源与闭源软件展开，强调虽然开源依赖于社区和公司的贡献，但“Open Core Model”通过高级功能提供了一种有前景的商业化策略。

- **AI 建模中的分块选择**：**@lhc1921** 强调了 Embedding 分块（Chunking）所需的特异性——没有万能的方法，**@mrdragonfox** 进一步补充这取决于数据集的特征。

- **AI 中的 JSON 和语法建议**：用户建议通过带有示例的 Prompt 来改进 JSON 输出生成效果，并讨论了在平台上通过示例集成语法，**@gbourdin** 分享了一个相关的 [链接](https://horosin.com/extracting-pdf-and-generating-json-data-with-gpts-langchain-and-nodejs)。

- **微调技巧**：出现了关于微调 LLM 的问题，从为初学者创建对话数据集到解决低资源语言模型中的挑战。**@yashkhare_** 引用了关于使用 LoRA 进行预训练的研究，而 **@mrdragonfox** 对与 Instruct 版本模型相比微调方法论不足表示了担忧。

- **Mistral 的成功实施与 API 咨询**：**@soletum** 报告了用户对在产品中集成 **Mistral** 的满意度，并记录了关于 **API Key 激活时间** 的咨询，建议立即联系 Mistral 支持团队。

- **La plateforme 上的专用端点讨论**：关于大公司专用端点定价的查询引导至建议直接联系开发者关系部门以获取准确数字，因为考虑到成本可能显著高于原始 GPU 费用。

- **Office-Hour 洞察带来清晰度与需求**：工程师们寻求关于 System Prompt 和 Mixtral-8x7B-Instruct-v0.1 潜力的建议，讨论了 Instruct 模型和用户消息身份的细微差别。对 Mistral API 的功能需求显示了对增强功能的浓厚兴趣。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **通过权重冻结重新思考 RMT**：尽管不是参考资料的主要焦点，但 `@hexani` 认为在某种程度上类似于 **RMT** 的 **权重冻结 (Weight Freezing)** 概念具有创新性。

- **Qwen2 的潜在发布引发热议**：随着 `Qwen2` 在排行榜上短暂出现后又下线，人们对其发布的期待与日俱增，正如 `@nonameusr` 和 `@weyaxi` 所提到的。同时，一个“Llama 化”版本的 `Qwen-72B` 引发了对 Tokenizer 性能的担忧，相关模型可在 [Hugging Face](https://huggingface.co/Weyaxi/Qwen-72B-Llama) 上找到。

- **利用 4-bit 优化器状态进行训练**：一项 [thu-ml 研究](https://github.com/thu-ml/low-bit-optimizers/) 表明，使用 **4-bit 优化器状态** 训练神经网络可以大幅减少内存占用，详见其 [论文](https://arxiv.org/abs/2309.01507)。

- **N-gram 模型的持久价值**：一种“无限 gram (infinite-gram)”模型被认为在神经 LLM 时代具有相关性，根据分享的一篇 [论文](https://arxiv.org/abs/2401.17377)，其规模可能超越传统的 n-gram 限制。

- **Anthropic 对 LLM 中潜伏特工发出警报**：讨论了后门训练的一个潜在棘手问题，根据 [Anthropic 的文章](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training)，**潜伏特工 (Sleeper Agents)** 在安全训练后仍能持续存在。

- **Project Obsidian 进入休眠状态**：**Project Obsidian** 对 **多模态 (Multimodality)** 的关注得到了澄清，该项目已基本完成并发布，正如 `@teknium` 所指出的，可在 [Nous 的 Hugging Face](https://huggingface.co/Nous) 上获取。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 摘要

- **自定义 Pipeline 与创造力基准测试**：针对 `vikhyatk/moondream1` 的自定义 Pipeline 已在 [Hugging Face 上发布](https://huggingface.co/vikhyatk/moondream1/discussions/6)，并提出了模型创造力的评估基准，相关讨论见 [Twitter](https://twitter.com/Vipitis/status/1752699776766988309)。Resume QA Space 和 Han Instruct 数据集已[上线](https://huggingface.co/spaces/not-lain/resume-qa)，乌克兰语 wav2vec2 bert 模型也已[发布](https://huggingface.co/Yehor/w2v-bert-2.0-uk)。

- **庆祝模型里程碑与 LLM 托管讨论**：`moondream1` 的下载量突破 2500 次；讨论了类似于 `moonvalley` 的视频生成工具；明确了在 Hugging Face 服务器上使用 `llama2` 等模型的 API 使用规范；并就项目的免费 LLM 托管以及为学术研究选择新颖数据集进行了[探讨](https://arxiv.org/abs/2311.07989)。

- **知识共享与远程就业建议**：重点介绍了 LLM 的免费 API 访问以及预训练 LLM 的[导览指南](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing)，并为美国公民与国际 Web 开发者合作提出了一套**跨国就业策略**。

- **新型 Diffusion 与聚类模型**：Google 发布了新型 Diffusion 模型 **MobileDiffusion**，其性能可能超越 Stable Diffusion 和 DALL·E。建议使用 [EfficientNet 模型](https://www.akshaymakes.com/blogs/pytorch) 辅助聚类模型以识别用户行为。

- **社区创作者的创新**：分享了托管在 [colbert.aiserv.cloud](https://colbert.aiserv.cloud) 的 **ColBERT 实时可视化工具**。**UltraTextbooks** 数据集已在 [Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks) 发布，并通过 [Medium 文章](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45) 介绍了如何在远程 GPU 上实现 **VS Code 与 Jupyter 的集成**。

- **关于 AI 与法律的阅读与演示**：**Mamba** 演示被安排在读书会讨论中，通过 [When2meet 链接](https://www.when2meet.com/?23471427-n4DUl) 进行排期，并有人咨询关于法律领域 AI 挑战演示的录音事宜。

- **Dreambooth 训练与 TPU 咨询**：Stable Diffusion 的 **Dreambooth LoRA** 训练现已可用，其高级功能在 [GitHub 脚本](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py) 中披露。关于此类训练的 TPU 兼容性问题[尚未得到解答](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968)。

- **AWS SageMaker 与 CUDA Toolkit 安装讨论**：寻求关于将 **AWS SageMaker** 与 **Mistral**、**Llama** 或 **Tapas** 结合使用的指导，并报告了可能与 CUDA toolkit 版本相关的安装问题。

- **Gradio 的弹窗组件**：发布了 `𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕` 以增强 Gradio 应用的弹窗功能，可以在 [Hugging Face Space](https://huggingface.co/spaces/aliabid94/gradio_modal) 进行体验。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **LLaMa 1.6 获得低调好评**：用户 `@kache_` 表示 **LLaMa 1.6** 表现良好，虽然没有给出具体的性能指标或对比，使讨论保持了一种轻松而神秘的氛围。

- **社区文章登上 Hacker News 热门**：一名成员在 Reddit 上的文章被发现在 Hacker News 上流传，另一位用户提供了[该帖子的链接](https://news.ycombinator.com/item?id=39215242)进一步扩大了影响。

- **关于 Bard 水印图像的争议**：`@max_voltage` 批评了 Bard 的图像生成功能嵌入水印的做法，引发了辩论。这虽是迈向负责任 AI 的一步，但也引发了更广泛的讨论，暗示了创意与伦理之间的冲突。

- **Imagen 2 生成的图像出现意外噪点**：`@thejonasbrothers` 对 Imagen 2 生成图像的噪点水平表示担忧，指出与 SDXL 的图像相比，可能存在质量问题和输出格式效率低下的情况。

- **深入探讨 Autoencoder 的 Latent Space 敏感性**：`@drhead` 和 `@thejonasbrothers` 就 Autoencoder 的 Latent Space 细微差别展开了详细讨论，涉及 Segmind Vega 和 SDXL VAE 模型，以及训练对噪点模式演变的影响。

- **OLMo 获得关注**：用户 `felfri_` 仅发布了一个 [Allen Institute for AI 开发的 OLMo 链接](https://allenai.org/olmo)，没有提供上下文，但推测是向工程人员暗示值得关注的创新研究。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **ChatGPT 的长期记忆功能引发关注**：`@flyyt4` 寻求增强 ChatGPT 长期记忆的建议，`@fabguy` 建议使用 "memGPT"，但未提供具体来源或额外背景。

- **提示词工程技巧分享**：`@wolfspyre` 强调了 Chain of Thought 和 Tree Thinking 等高级 Prompt 技巧，并提供了一个[高级 Prompt 示例链接](https://funkpd.com/devlog/prompt-examples/)。

- **探索不受限模型**：用户讨论了 The Bloke Goat Storytelling 70B Q3KS 等无审查模型的性能和实用性，重点关注故事写作应用，并建议避免使用较小的模型。

- **GGUF 与模型适配器面临的挑战**：关于 GGUF 格式的可用性以及在 LM Studio 中定位模型适配器（Model Adapters）的反馈褒贬不一，用户分享了在磁盘上查找和编辑模型的建设性建议。

- **模型性能的硬件考量**：`@silverstar5654` 和 `@docorange88` 等用户询问了如何利用双 RTX 3090、RTX 6000 或 NVLink 配置来运行 Mixtral 等大型模型，并讨论了多 GPU 配置下的潜在性能。

- **关于 llava 1.6 和 Autogen 服务器能力的咨询**：LM Studio 是否支持将 llava 1.6 识别为视觉模型尚存疑问，此外还有人好奇是否可以在多个服务器上运行 Autogen，以及是否可以在不同端口上运行两个 LM Studio 实例。

- **Open Interpreter 的集成问题与成本担忧**：用户 `@raiderduck` 报告了在将 OI 与 LM Studio 配合使用时出现无意义响应的问题，并提到使用 GPT-4 的成本很高（一周达到 250 美元），因此正在寻找更具成本效益的服务器设置。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **OLMo 步入 LLM 竞技场**：新发布的 [OLMo 论文](https://allenai.org/olmo/olmo-paper.pdf) 引发了关于其方法论和许可协议的热烈讨论，AI2 澄清目前尚未发布对话模型或界面。针对 OLMo 的基准测试、训练代码（特别是其 warmup scheduler）、归一化选择以及 AI2 的评估指标提出了疑问，关于 tokenizer、layer norm 的使用以及潜在的基准测试 cherry-picking 的讨论在技术群体中引起了反响。
  
- **讨论 N-gram 模型中的万亿级 Token**：Eleuther 社区辩论了将 [n-grams 扩展到万亿级 Token](http://arxiv.org/abs/2401.17377) 的价值及其与 LLMs 的集成，思考了潜在的性能提升和泛化能力。Infinigram 模型的局限性与应用、为 LLMs 标记回译数据的潜力，以及诸如 MCTS 等合成数据生成策略，也构成了严谨研究对话的一部分，展示了对提升 AI 效率的不懈追求。

- **庆祝创新与贡献**：分享了 AI 研究领域的更新，用于多模态工作的 Gaussian Adaptive Attention 库的发布引起了关注。AI 研究的动态特性通过 [Amortized Text-to-Mesh (AToM)](https://snap-research.github.io/AToM/) 等作品的发布进一步凸显，传播了视觉表示学习和模型编辑技术的最新进展。

- **分析上下文学习与神经电路**：对上下文语言学习 (ICLL) 的见解促成了[相关研究论文](https://arxiv.org/abs/2401.12973)的发现，以及对语言模型中[上下文神经元](https://arxiv.org/abs/2311.00863)的讨论。该小组还辩论了 ∞-gram 模型在 perplexity 与开放式文本生成方面的有效性，并参考了一项[关于后者的研究](https://arxiv.org/pdf/2210.10253.pdf)，对比了 ∞-gram 检索与 MoE 模型攻击所面临的挑战。

- **Thunderdome 中的简短确认**：在 Thunderdome 稀疏的聊天中，`daniellepintz` 的简短回复仅确认了 `limit` 的不存在，未提供进一步的详细说明或背景。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Reddit 对 VAE 发出警报**：一篇 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/s/p2BkzI9ypO) 揭示了 **SD1.x, SD2.x, SVD, DALL-E 3** 等模型中使用的 VAE 在 KL 散度损失方面的一个关键问题，这可能通过极小像素传输全局信息从而影响效率。

- **VAE 的简洁性备受关注**：`@swyxio` 注意到的一场关于 VAE 的 Reddit 讨论因其直截了当而引起兴趣，认为它与通常包含“不必要冗余”的学术出版物相比，简洁得令人耳目一新。

- **LLM 引发迷因等**：`@natolambert` 分享了一个 [**LLM 迷因**的链接](https://twitter.com/natolambert/status/1753063313351835941)，强调了社区对语言模型的幽默参与，同时讨论了 AI2 的开放语言模型 (**OLMo**) 的硬件多样性和短上下文长度限制。

- **开源领域迎来新选手**：**Nomic AI** 发布了一套在 LoCo 基准测试中表现优异的开源 embeddings，并随发布附带了一篇[详细的博客文章](https://www.interconnects.ai/p/olmo)。

- **最新 Latent Space 播客低调上线**：正如 `@swyxio` 在 [Twitter](https://twitter.com/latentspacepod/status/1753120715254198425) 和 Hacker News 上宣布的那样，**Latent Space Podcast** 发布了新一期，尽管采取了标题党策略，但仍难以获得关注。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **伦理困境与 AI 角色混淆**：由 `@yami1010` 发起的讨论涉及 AI 的伦理参数，促使人们渴望获得关于 LLM 细微差别的资源，并触及了 AI 开发中经常模糊的责任定义。

- **辩论 ChatGPT 智能下降**：`@voidrunner42` 和 `@tadase.` 参与了一场关于 ChatGPT 性能的对话，辩论感官上的智能下降是由于能力的真实下降，还是仅仅因为所提供的 Prompt 的性质。

- **AI 归功于其应得之处**：`@movoza` 纠正了一处归属错误，即 Bing 似乎声称独立开发了某项 AI，这引发了关于 AI 进步中的集体努力以及涉及多个利益相关者的复杂历史的讨论。

- **调侃 GPT-3 学校机器人及其他**：`@qiqimon` 询问了在学校客服中使用 GPT 驱动的聊天机器人的可行性，而其他人则指出了防止滥用措施的重要性，认为这种用例是可以实现的，但并非没有挑战。

- **关于 GPT-3 数据格式和身份验证问题的技术讨论**：针对输入 GPT 的知识库的最佳文件格式分享了意见，特别是相比 JSON 和 XLXS 更倾向于 RSV；`@woodenrobot` 讲述了其项目从 alpha 转向公开 beta 期间遇到的 API 身份验证麻烦，突显了 AI 工程师在扩展解决方案时面临的现实问题。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **跨频道的模型担忧与解决方案**：**general** 频道的讨论引发了对代码更改可能影响 Llama 和 Mistral 等模型的担忧，但 **axolotl-dev** 频道指出一个修复程序已在合并到上游。针对 **Mixtral**，**general-help** 频道的用户分享了根据 GPU VRAM 采取不同微调方法的建议，并讨论了使用 **Zero3** 卸载（offloading）来微调整个模型。

- **7B 模型的性能讨论**：有关于 7B 模型停滞不前的讨论，但也提到了 `CapybaraHermes-2.5-Mistral-7B` 和 `Eagle 7B` 等名称，以及性能更新和结果链接，突显了该领域的活跃竞争。引用了 [Hugging Face 上的 Eagle 7B](https://huggingface.co/RWKV/v5-Eagle-7B)、[MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) 和 [CapybaraHermes-2.5-Mistral-7B](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B)。

- **探索文本嵌入与向量数据库**：关于文本嵌入模型和向量数据库进行了技术交流，其中 **nomic-embed-text-v1** 备受关注，此外还讨论了从 **bge** 开始以及在云服务中使用 GPU。为了进一步探索，分享了 [GitHub 上的 qdrant/fastembed](https://github.com/qdrant/fastembed) 和 [通过优化检索和重排序模型改进 RAG](https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.html#Bi-Encoder-Model)。

- **RunPod 服务挑战引发争议**：在 **runpod-help** 中，出现了关于 Pod 突然关闭、数据丢失以及关于 Pod 删除时间的沟通混乱的投诉。此外，数据传输过程中的 SSL 错误和下载速度慢的问题引发了关于 RunPod 服务可靠性和效率的讨论，并建议到 RunPod Discord 寻求帮助：[RunPod Discord 帮助频道](https://discord.com/channels/912829806415085598/1187492973148115076)。

- **未解答的 Mistral 配置**：`dangfutures` 在 **shearedmistral** 频道发布的一条孤立消息强调，关于 Mistral 的配置仍存在持续的好奇或困惑，这可能预示着社区内更广泛的对话或澄清需求。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **各频道共享的 AI 趣闻**：用户 `@tbyfly` 在通用聊天频道咨询后，被引导到另一个频道分享来自 Perplexity AI 的幽默回复。

- **应对模型复杂性**：讨论了旨在生成最新且事实性回复的 PPLX 模型；然而，用户 `@brknclock1215` 和 `@jayb1791` 指出了 **7b-online model** 在处理复杂查询时的局限性以及在 SQL 辅助方面的隐私过度扩张。同时，`@general3d` 建议改进 **codellama 70b model** 以确保更一致的回答。

- **将 Perplexity 设为默认搜索**：`@bartleby0` 提供了将 Perplexity AI 设置为默认搜索引擎的解决方案，并提到 Arc Search 是一个潜在的竞争对手。

- **有趣的 AI 应用与观察**：新成员 `@.sayanara` 链接了一篇关于 AI 缓解虚假信息潜力的博客文章和书籍（[The Right Direction for AI](https://figmentums.com/2024/02/01/the-right-direction-for-ai/)）。在其他地方，由于 Facebook 未出现在顶级应用列表中，`@bartleby0` 认为这种缺失“很有趣”。

- **订阅故障与 API 异常**：`@dame.outlaw` 请求协助解决订阅问题，`@alankarsh` 报告了订阅后 API 额度出现的问题，揭示了用户体验中的小状况。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 摘要

- **OpenAI Embeddings 的开源竞争对手**：LlamaIndex 引入了由 `@nomic_ai` 开发的新开源文本嵌入模型 *nomic-embed-text-1*，其性能优于 OpenAI 的 text-embedding-3-small，并在 [Twitter 上发布了集成细节](https://twitter.com/llama_index/status/1753106179008696521)。
- **Agentic RAG 主题演讲回放**：@jerryjliu0 关于 *Beyond Naive Rag: Adding Agentic Layers* 的主题演讲回放现已在 [YouTube](https://t.co/hrUMF8bq8Q) 上线，幻灯片可在此处 [获取](https://t.co/P39riIMGK6)。
- **LlamaIndex 扩展兼容性**：LlamaIndex 重点介绍了集成其他 LLM 的指南（例如使用 Llama-2），并提供了 [指南](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules) 和 [示例 Notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb)，同时在活跃讨论中解决了关于 Postgres 连接和 MongoDB 的问题。
- **寻求 LLM 的 PII 匿名化**：通过 langchain 和 Presidio 高效地从文本数据集中匿名化个人身份信息（PII）的需求表明需要生产级的解决方案，因为目前的方法仍处于实验阶段。
- **语音转文本深度解析**：`@amgadoz` 在一篇详细的 [博客文章](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b) 中探讨了 OpenAI 的 Whisper 模型，重点介绍了其 encoder-decoder transformer 架构及其与 "Attention is All You Need" 论文的关系。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

- **警惕 LangChain AI Twitter 诈骗**：包括 `@markopolojarvi`、`@alcazarr` 和 `@solac3` 在内的多名用户报告了涉及 **LangChain AI Twitter 账号** 的可能安全漏洞。一条可疑的 [推文](https://twitter.com/LangChainAI/status/1753014882696405254) 引发了账号被黑的猜测。

- **创新型自主 GPT-4 Agent 平台发布**：`@robot3yes` 发布了 **Agent IX**，这是一个独立的 GPT-4 Agent 平台，并鼓励社区在 [GitHub](https://github.com/kreneskyp/ix) 上进行探索。

- **ContextCrunch 优化 Token 效率**：`@speuce` 推广了 **ContextCrunch**，这是一个旨在降低 Token 成本的 prompt 压缩 API，可在 [contextcrunch.com](https://contextcrunch.com/) 获取早期访问权限和更多详情。

- **LangServe 和 Llamacpp 致力于实现流式传输功能**：`@veryboldbagel` 和 `@legendary_pony_33278` 讨论了如何在 **Llamacpp** 与 **Langserve** 或 **FastAPI** 中启用流式传输，并分享了一个演示 LangServe 配合 ollama 使用的 [GitHub 示例](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py)。

- **探索 Step-Back Prompting 和上下文压缩**：一篇 [Medium 文章](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b) 详细介绍了增强语言模型交互的方法，例如 Step-Back Prompting 和使用 **LangChain** 进行 **Contextual Compression**。这种压缩技术被讨论为解决 RAG 架构中 Token 使用过度问题的潜在方案。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- **切换到 Mixtral 以获得更大收益**：`_jp1_` 确认了**从 7b 模型向 Mixtral 的迁移**，理由是后者在评估任务中的高效表现，并邀请他人协助完成迁移过程。
- **API 暴露风险**：`@sebastian.bodza` 发出了警报，指出 **API 缺乏安全措施**，因为目前在请求中未使用 Token，存在安全风险。
- **Nomic 表现优于 OpenAI**：由 `@bjoernp` 展示，**Nomic 的 Embedding 模型** `nomic-embed-text-v1` 以 8192 的序列长度占据优势，并超越了 OpenAI 的同类模型。该模型及其权重和训练代码可在 [Hugging Face 上的 Nomic AI](https://huggingface.co/nomic-ai/nomic-embed-text-v1) 免费获取。
- **OLMo 7B 进入开源模型竞技场**：由 `_jp1_` 介绍的 **OLMo 7B**（Allen AI 的开源语言模型）现已发布，随附数据集、训练资源和研究论文，可通过 [Allen AI 上的 OLMo](https://allenai.org/olmo/olmo-paper.pdf) 和 [Hugging Face 上的 OLMo](https://huggingface.co/allenai/OLMo-7B) 获取。
- **推理的 GPU 替代方案与共享资源**：在缺乏 GPU 的情况下，`_jp1_` 提出了使用 **Replicate、Modal 或 Serverless Runpod** 等服务的推理替代方案，并暗示如果需要可以进行小组托管，同时分享了一个可能有用但未说明具体用途的 Google Colab 笔记本，链接见[此处](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK)。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 摘要

- **Nomic Embed 处于领先地位**：来自 [Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1) 的新模型 **nomic-embed-text-v1** 声称性能优于类似的 OpenAI 模型，在 MTEB 上得分为 62.39，在 LoCo 上得分为 85.53。社区成员表现出兴趣，希望建立一个类似于 LLM Arena 的竞争性排行榜来比较 Embedding 模型。
  
- **寻求部署指导**：Discord 用户 firefox8975 正在寻求关于使用 VLLM 将开源机器学习模型部署到 AWS 的建议或指南，这引发了成员之间关于实际问题解决的讨论。

- **Exa 宣布发布活动**：**Exa**（原名 Metaphor）将于 **2 月 2 日**在旧金山举办托加袍主题的发布派对（[旧金山派对报名](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?)），并于 **2 月 7 日**在纽约市举办（[纽约派对报名](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs)）。`@sarahchieng` 还提出愿意请有兴趣在两地讨论 Exa 的人喝咖啡。

- **向量数据库迁移进行中**：在向量数据库领域，用户 `@michelcarroll` 分享了他们从 **Weaviate** 迁移到 **pgvector 结合 HNSW** 的过程，为数据平台的实际应用和迁移提供了见解。

- **探讨思维链 (CoT) 的持久化**：有一场关于保存和重用语言模型思维链 (Chain of Thought, CoT) 权衡的对话。虽然重用 CoT 可以降低成本和处理时间，但有人强调，由于需要多次 API 调用，这样做会带来延迟方面的权衡。

---

## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord 摘要

- **CUDA Kernel 实现极速提升**：`@zippika` 实现了一个用于 **rgb_to_grayscale** 转换的 CUDA Kernel，通过优化利用 **ulong** 进行向量化加载，将 GPU *占用率 (Occupancy)* 提高到 **77%**，*显存*利用率提高到 **71.87%**。然而，尽管理论上有改进，该 Kernel 在实践中速度较慢，证明了优化的复杂性。[在此查看优化详情](https://example.com/link-to-code)。
  
- **凭借 C++ 基础入门 CUDA**：用户 `@noobpeen` 成功配置了 **CUDA 12.2 与 Visual Studio**，并计划利用其 PyTorch 和 C++ 知识开始 CUDA 开发。资深用户 `@lancerts` 建议从一个新的 CUDA 项目开始，特别是开发 CUDA Kernel，并研读《Professional CUDA C Programming》一书以深入探索。

- **利用线程粗化 (Thread Coarsening) 提高效率**：在讨论中，`@tvi_` 强调了**线程粗化**在提高工作效率方面的益处，即通过减少全局内存加载频率来优化，这一原则不仅涉及计算效率，还涉及《PMPP》一书中所探讨的内存优化。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **Datasette 文档与 GPT 的结合**：@simonw 尝试将 [Datasette 文档](https://docs.datasette.io/en/stable/) 的 **PDF 版本**输入到 **GPT** 中，但发现初始结果不尽如人意。在经过更多改进后，他们对该技术的潜力保持了一定的乐观态度。
- **探讨“安静的 AI” (Quiet AI)**：一篇文章介绍了关于追求“安静的 AI”的内容，但未提供实质性的讨论或细节。[在此阅读文章](https://www.dbreunig.com/2024/02/01/pursuing-quiet-ai.html)。

---

# PART 2: 各频道详细总结与链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1202529153241452575) (1279 条消息🔥🔥🔥): 

- **讨论 Miqu-70B 的性能**：`@netrve` 和 `@mrdragonfox` 等用户分享了使用 Miqu-70B 模型以及 Mistral Medium 等不同版本的经验，一些人评论了它在回答过程中出人意料的自我纠错能力。对话涉及模型速度（`@goldkoron` 提到经过一些调整后平均达到 17t/s）以及与 Mixtral 等其他模型的比较。

- **关于编程语言的对话**：展开了一场关于编程语言的讨论，辩论了 C++, Rust, Go 及其他语言的优缺点。`@mrdragonfox` 赞扬了 C 的简洁性，并表示并非所有编程都需要类 (classes)。讨论涉及 C++ 的高级特性以及 Rust 和 Go 提供的简洁性，`@rtyax` 等人表达了对更简单、更易理解的语言的偏好。

- **Chatbot UI 讨论**：`@righthandofdoom` 对能够连接到远程 OpenAI 兼容端点的简约 Web UI 或原生 UI 表示感兴趣，`@coffeevampir3` 和 `@animalmachine` 建议了 Hugging Face 的 "Candle" 等替代方案，并感叹目前使用现有工具创建前端存在的问题。

- **投机解码 (Speculative Decoding) 与 API 工具**：`@.justinobserver` 等人讨论了投机解码以及目前哪些工具支持它，提到了 llama-cpp，以及 `@flashmanbahadur` 希望找到一个支持 exl2 和 OpenAI 兼容 API 的工具。`@itsme9316` 建议了 "tabbyapi"，因为它提供兼容 OpenAI 的 API 并支持 exl2。

- **微调与校准的误解**：`@turboderp_` 和 `@giftedgummybee` 等用户解决了关于微调背景下校准目的的误解。对话深入探讨了使用量化作为保留模型质量手段的挑战，`@giftedgummybee` 在加载 parquet 文件时遇到问题，并考虑依赖默认量化设置。

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1202524767907745802) (665 条消息🔥🔥🔥): 

- **适合聊天微调吗？**：`@kaltcit` 指出，经过训练的 llama 简历 (`<@266127174426165249>`) 擅长 ERP，但没有 ghost attention。
- **对 Miqu API 的乐观态度**：`@righthandofdoom` 为他人提供了使用其受限额的 Groq API 密钥进行 Mixtral 实验的机会。
- **模型 VRAM 困境**：`@kaltcit` 在尝试将 Miqu-1-70B-SF-4.25bpw-h6-exl2 放入可用 VRAM 时遇到问题，与 `@turboderp_` 讨论了潜在的解决方案，甚至考虑升级硬件。
- **尝试新任务**：参与者讨论了新尝试任务的结果，例如详细的角色 RP、更长的 token 上下文，使用了 Miqu 和 LLM 等模型，寻求最佳性能并努力克服硬件限制。
- **分享 LLM 性能见解**：`@mrdragonfox`、`@doctorshotgun` 和 `@goldkoron` 等多位用户讨论了不同的设置和配置，以优化语言模型在特定硬件上的速度和性能，考虑了 VRAM、模型 bits-per-weight (bpw) 和上下文大小等因素。

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1202666974606266409) (26 messages🔥): 

- **预测模型训练中的挑战**：`@hapticalacrity` 正在寻求关于训练模型从文本字符串预测 8 位层级代码的建议，每个子类别有 25 个示例。该任务尝试使用了 **Distilbert-base-uncased**，在较高级别显示出准确性，但在预测代码的最后 2 位数字时表现不佳。
  
- **模型改进的创新替代方案**：`@tom_lrd` 建议考虑使用 **带有 embeddings 的递归聚类** 等方法，而不是昂贵的模型训练，这可以有效地处理类别预测，而无需依赖沉重的硬件资源。
  
- **利用自然语言模型进行类别分类**：`@tom_lrd` 建议利用 **Prompt Engineering** 和大型模型（如 Mistral7B）以层级方式将文本分类，尽管他承认这是一种资源密集型的方法。
  
- **训练查询的快速解决**：`@hapticalacrity` 向 `@tom_lrd` 表示感谢，感谢其对分类问题提供的简洁且有帮助的反馈，突显了 **TheBloke** Discord 社区的协作氛围。
  
- **微调后的模型使用**：`@chovii` 在尝试使用微调后的模型时遇到了 **ValueError**，并在 **合并 LoRA 层时遇到问题** 后寻求正确流程的帮助。
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1202879708090343476) (2 messages): 

- **针对有限资源的 InternLM 推荐**：`@alphaatlas1` 建议那些可能无法运行较大的 34B 模型的用户使用 **InternLM 20B**。该建议是为资源受限的用户量身定制的。
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1202676909591363625) (12 messages🔥): 

- **搜索与文件组织优劣的辩论**：`@wbsch` 认为软件中 **良好的搜索能力** 可能比传统的标签或目录等文件组织方法更有效。他们提到自 2008 年以来一直使用 nvpy 和 Simplenote 系统来满足笔记需求。
  
- **IT 领域对文件系统知识的必要性**：`@wbsch` 同意 `@Splice` 的观点，确认理解文件和文件系统对 IT 来说至关重要，尽管在组织文件方面，良好的搜索功能具有其优势。
  
- **用户界面中的文件系统**：`@wbsch` 详细说明，对于用户界面，良好的搜索是必不可少的，并且应该辅以“最近修改的文件”等方法来减少用户的工作量，同时提到 IDEs 受益于函数搜索功能以提高效率。

- **动态链接问题的实际解决方案**：`@spottyluck` 分享了运行 `main` 命令时缺少 `libllama.so` 库的问题，并演示了如何使用 `patchelf` 通过指定库路径来解决问题，以及如果库被移除，则使用 `LD_PRELOAD` 作为替代方案。

- **推荐良好的维护实践**：在缺少共享库的背景下，`@spottyluck` 建议在应用 `patchelf` 等临时修复后修复构建过程，以确保建立正确的库链接。
  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1202543162514739200) (178 messages🔥🔥): 

- **GPU 短缺影响模型训练**：`@frosty04212` 讨论了 GPU 供应不足影响产能的问题，并将短缺部分归因于半导体制造的复杂性，这涉及漫长的流程和巨额投资。`@mrdragonfox` 进一步阐述了扩大生产的障碍，例如长达数月的晶圆（wafer）制造周期、严苛的洁净室要求、专业人才需求，以及 TSMC、ASML 和 Zeiss 等公司之间的协作。

- **开源之辩**：关于开源与专有软件之间相互作用的讨论非常活跃。`@frosty04212`、`@ethux` 和 `@firesonwires` 等用户讨论了开源技术是否能与闭源技术竞争，并指出开源通常依赖于大公司的贡献和资金支持。“Open Core Model” 被提及为一种积极的混合模式，意味着虽然某些元素可以保持开源和免费，但可以通过高级功能实现商业化。

- **Mistral 的公关策略**：用户 `@_dampf` 称赞 Mistral 在 Mistral Medium 模型的早期访问版本泄露事件中保持透明，而不是保持沉默。`@ethux` 补充说泄露的模型并非最新版本，`@mrdragonfox` 则提到泄露的模型（如 MIQU）不会获得官方支持。

- **Mistral API Key 激活咨询**：用户 `@cptplastic` 在遇到 Rate Limit（速率限制）问题后，询问了 Mistral 新 API Key 的激活时间。`@mrdragonfox` 建议对于任何激活问题都可以发送邮件给 Mistral 支持团队，而 `@i_am_dom` 则确认 API Key 应该具有即时且合理的限制。

- **对未来模型的预测与思考**：关于 Mistral 下一步行动的投机性聊天浮出水面，`@gtnbssn` 开玩笑说可能会有一个名为 miqtral 或 miqstral 的模型，并可能使用 Q-learning。用户们分享了对即将推出的技术的看法和笑话，反映了对未来模型迭代的好奇和兴奋。
  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1202667649704927283) (4 messages): 

- **Embedding 中分块大小（Chunk Size）至关重要**：`@lhc1921` 幽默地承认自己有些钻牛角尖，因为他注意到在某些用例中，Embedding 的分块方式确实存在错误。
- **数据决定分块策略**：`@mrdragonfox` 强调如何对 Embedding 文档进行分块取决于数据本身，这意味着不存在一种能适用于所有不同数据集的通用分块大小。
  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1202627410953117746) (4 messages): 

- **La Platforme 上的语法集成？**：`@superintendent` 询问了在平台上集成语法（grammar）的问题，并提到了一种向系统提供示例的方法。

- **Prompting 优于 JSON Schema 的直接指令**：`@gbourdin` 建议通过向聊天机器人提供示例来获得理想的输出，并分享了一个[链接](https://horosin.com/extracting-pdf-and-generating-json-data-with-gpts-langchain-and-nodejs)，该链接演示了在 Prompt 中结合示例使用 JSON Schema 以获得有效结果。

- **模型规模对 JSON 输出的影响**：`@samy7418` 观察到，在 Medium 规模的模型上可以通过 Few-shot 示例生成 JSON 输出，但 Small 规模的模型往往会在 JSON 输出中添加解释性文本。

- **在 Discord 链接中寻求上下文**：`@akshay_1` 回复了 `@samy7418` 并提供了一个指向 Discord 频道中特定消息的链接，推测是为了提供上下文或进一步的信息。
  

---

### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1202543151584387092) (44 messages🔥): 

- **寻求微调指导**：用户 `@d.j147` 是该领域的新手，正在寻求关于如何创建对话数据集以及微调大语言模型（LLMs）的建议。
- **低资源语言的挑战**：`@quicksort` 分享道，针对在预训练语料库中占比不足 5% 的语言进行 LLMs 微调会导致语言质量欠佳，并询问关于使用低资源语言数据集持续预训练 Mistral 的见解或成功案例。
- **LoRA 预训练探索**：`@yashkhare_` 讨论了使用 LoRA 进行预训练的意图，参考了一篇[研究论文](https://arxiv.org/pdf/2304.08177.pdf)，并想知道 Mistral-7b 论文训练中使用的 Token 数量和语言种类。
- **对 Mistral 训练方法的沮丧**：`@mrdragonfox` 表示社区尚未找到训练 Mistral 的最佳方法，提到目前的尝试都无法与语言模型的基础 Instruct 版本相媲美。
- **Axolotl 训练查询**：用户 `@woodenstick_` 在使用 Axolotl 训练 Mistral 时遇到了 `IndexError`，并分享了可能与该错误相关的配置代码片段。
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1202620605279637534) (16 messages🔥): 

- **Mistral 获得积极反馈**：`@soletum` 在频道中告知，他们在产品中集成了 **Mistral**，提供了一个基于关键词的文本纠错和生成界面。首批用户对其易用性表示满意。
- **社区鼓励**：`@frammie` 对 Mistral 的实现表示赞赏，称其看起来“非常棒，太赞了！”
- **AIDocks 展示**：用户 `@lhc1921` 分享了 **AIDocks** 的 GitHub 仓库链接：[GitHub Link](https://github.com/l4b4r4b4b4/AIDocks)。
- **友好的德语交流**：`@mrdragonfox` 和 `@lhc1921` 之间进行了德语对话，讨论了平台上德语社区成员的存在。
- **Hugging Face 聊天机器人建议**：针对 `@jamiecropley` 询问哪里可以使用 **Mistral** 作为聊天机器人的问题，`@ethux` 建议查看 [Hugging Face Chat](https://huggingface.co/chat)。
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1202564142490325032) (26 messages🔥): 

- **寻找专用服务的价格**：`@gbourdin` 询问了 La plateforme 上专用端点（dedicated endpoints）的可用性和成本，寻求一份针对大公司估算的月度报价。尽管没有提供具体数字，但他们提到将 HuggingFace 的推理端点定价作为参考。
- **定制端点的收入基准**：`@mrdragonfox` 建议，定制专用端点通常要求企业年收入至少达到 100 万美元，并强调此类服务要贵得多。
- **直接联系以获取准确估算**：建议 `@gbourdin` 联系 `support@mistral.ai` 或直接联系负责开发者关系的 Sophia 以获取精确报价，尤其是 `@mrdragonfox` 澄清了他们并不代表 Mistral 官方。
- **了解成本规模**：`@mrdragonfox` 指出，Mistral 最先进模型的企业级部署（如果可用）可能远高于原始 GPU 成本——根据他们的估计，大约高出十倍。
- **迈向合作伙伴关系的第一步**：`@gbourdin` 考虑先从 La plateforme 的常规端点开始，并接受定制端点的最低预算可能在每月 1 万美元左右起步，并计划根据此信息与潜在的企业客户进一步讨论。
  

---

### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1202644588867879042) (240 messages🔥🔥): 

<ul>
<li><strong>System Prompts 的澄清请求</strong>：`@sa_code` 寻求关于在 Mistral 中使用 System Prompts 的指导，参考了其 PR 中的 Jinja 模板，并指出 Hugging Face 上的 Mixtral 聊天模板缺少 System Prompt 支持。他们提供了文档链接及其 PR 以供参考。</li>
<li><strong>探寻 Mixtral 的全部潜力</strong>：`@touristc` 询问 Mixtral-8x7B-Instruct-v0.1 是否已达到其参数潜力的极限，或者是否还有改进空间。`@sophiamyang` 回应表示总有改进的空间。</li>
<li><strong>Instruct 模型详解</strong>：`@damiens_` 询问了 Instruct 模型与非 Instruct 模型之间的区别，`@sophiamyang` 回复称 Instruct 模型经过训练可以更好地遵循指令。`@canyon289` 发布了解释视频和指南链接，`@sandeep3890` 分享了原始论文链接，对该话题进行了进一步讨论。</li>
<li><strong>讨论消息身份的复杂性</strong>：`@jakobdylanc` 思考了为用户消息分配身份背后的复杂性。`@lerela` 解释说这个过程并不过度复杂，但需要数据集准备和微调（Fine-tuning）。</li>
<li><strong>API 功能请求及未来增强暗示</strong>：Office-hour 的参与者对 Mistral API 提出了各种功能请求，例如 JSON 输出支持、logprobs 和 logit bias、函数调用（function calling）以及改进对 openai 等 API 参数的支持。来自 `@lerela`、`@sophiamyang` 和 `@nicolas_mistral` 的评论暗示了正在进行的工作以及未来可能解决这些请求的版本发布。</li>
</ul>
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1202524801411981342) (3 messages): 

- **撤回之前的陈述**：`@hexani` 撤回了早些时候的陈述，提到所讨论的方法与 **RMT** 有些相似，但有趣的是它**冻结了权重**（freezes weights），他们认为这一点值得注意。
- **权重冻结中的创新点**：`@hexani` 澄清说，虽然讨论的方法与 **RMT** 有相似之处，但创新之处在于**权重冻结**，而非参考资料的主要观点。
- **图表中的 Wavenet 风格**：`@gabriel_syme` 观察到讨论的图表与 **Wavenet** 或类似的东西有相似之处，暗示了视觉数据表示中的平行性。
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1202624026988519475) (12 messages🔥): 

- **寻找精简的 Notion 模板**：用户 `@faiqkhan` 询问**极简主义 Notion 创业模板**，表示倾向于免费选项，因为大多数好的模板似乎都需要付费。
- **分享 Quackduckflower 之舞**：Error.PDF 发布了一个标题为 "Quackduckflower" 的 GIF 链接，提供了[视觉娱乐](https://tenor.com/view/quackduckflower-gif-25482704)。
- **询问 Dingboard 访问权限**：`.benxh` 询问联系谁可以获得 **dingboard** 的访问权限，`random_string_of_character` 建议在 Twitter 上给 Yacine 发送私信。
- **与《她》（Her）的互动小说冒险**：`@everyoneisgross` 分享了一个[基于 ChatGPT 增强的互动版](https://chat.openai.com/g/g-sAg0WI4ey-intheractive)电影剧本《她》（Her），允许用户按顺序参与故事。
- **关于 CLIP Embedding 现象的讨论**：`@cccntu` 发起了关于 **CLIP embeddings** 仅保留文本部分含义现象的对话，这是由于对比损失（contrastive loss）不需要保留每个细节。

**提到的链接**：

- [Cat Driving Car GIF - Cat driving car - Discover &amp; Share GIFs](https://tenor.com/view/cat-driving-car-gif-17184928271354710905)：点击查看 GIF
- [Quackduckflower GIF - Quackduckflower - Discover &amp; Share GIFs](https://tenor.com/view/quackduckflower-gif-25482704)：点击查看 GIF

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202637540234887198) (14 messages🔥): 

- **推向 Optimizer 内存效率的极限**：来自 [thu-ml](https://github.com/thu-ml/low-bit-optimizers/) 的研究展示了使用 **4-bit optimizer states** 训练神经网络的能力，这有可能显著减少模型训练的内存占用。该研究进行了详细的实证分析并提出了新的量化策略，详见其 [paper](https://arxiv.org/abs/2309.01507)。

- **揭秘用于语音转文本的 Whisper 模型**：`@amgadoz` 写了一篇关于 OpenAI **Whisper** 模型的详细博客文章，讨论了其架构以及如何将音频转录为文本。关于该模型及其对大规模监督预训练（supervised pre-training）依赖的见解可在 [Substack](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b) 上查阅。

- **使用 OpenHermes 2.5 AWQ 提升 AI 素养**：`@__ctrlaltdel__` 分享了一个 YouTube [video](https://youtu.be/4-hzQSOhIfc)，内容是他们在加拿大温哥华进行的关于开源 AI、小模型以及在结构化数据提取中应用的演讲。

- **探索端侧 Large Language Models**：Metaldragon01 链接了一个讨论 MiniCPM 的 **Notion document**，但未提供进一步的背景信息或有效的 URL。

- **CroissantLLM 作为双语竞争者出现**：`@euclaise` 分享了关于 **CroissantLLM** 的信息，这是一个拥有 1.3B 参数的双语语言模型，在英语和法语数据集上进行了预训练。更多细节和使用建议可以在 [Hugging Face](https://huggingface.co/croissantllm/CroissantLLMBase) 及其 [相关论文](https://arxiv.org/abs/2402.00786) 中找到。

- **关于合成数据生成公司的 NeurIPS 论文**：`@euclaise` 引用了一篇来自 NeurIPS 会议的关于合成数据生成公司的论文，但未包含进一步的细节或评论。该论文可以在 [这里](https://proceedings.neurips.cc/paper_files/paper/2022/file/0f5fcf4bff73a3537e0813a38f0d3f76-Paper-Conference.pdf) 获取。

**提到的链接**：

- [Intro to Open Source AI](https://youtu.be/4-hzQSOhIfc)：这是我于 2024 年 1 月 30 日在加拿大温哥华 BCIT 所做演讲的录音。该演讲专门讨论了自然语言处理 (NLP) 以及...
- [Whisper: How to Create Robust ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2)：多部分系列文章的第 2 部分，深入探讨了 OpenAI 最先进的自动语音识别模型 Whisper。
- [croissantllm/CroissantLLMBase · Hugging Face](https://huggingface.co/croissantllm/CroissantLLMBase)：未找到描述。
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)：一个将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。
- [Memory Efficient Optimizers with 4-bit States](https://arxiv.org/abs/2309.01507)：Optimizer states 是神经网络训练中内存消耗的主要来源，限制了在给定内存预算内可训练的最大模型。将 optimizer states 从 32-bit 浮点数压缩...
- [GitHub - thu-ml/low-bit-optimizers: Low-bit optimizers for PyTorch](https://github.com/thu-ml/low-bit-optimizers/)：适用于 PyTorch 的低比特优化器。通过在 GitHub 上创建账号来为 thu-ml/low-bit-optimizers 的开发做出贡献。

  

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202524431633883166) (374 messages🔥🔥): 

- **探索 Qwen 的 AI 和 Llamafication**：关于 `Qwen-72B` AI 的讨论涉及 `@light4bear` 和 `@weyaxi` 等用户分享对该模型的见解。`@weyaxi` 分享了一个指向 Llama 化版本的 [Hugging Face 链接](https://huggingface.co/Weyaxi/Qwen-72B-Llama)，并指出在使用时对 Tokenizer 性能的担忧。
  
- **关于 Qwen2 发布的不确定性**：用户 `@nonameusr` 和 `@weyaxi` 讨论了备受期待的 `Qwen2` 发布，`@weyaxi` 提到它曾短暂出现在排行榜上随后又下线，引发了对其亮相的猜测。

- **优化器讨论与大模型训练**：对话涉及训练大语言模型的挑战和策略，例如 `@euclaise` 提到一个名为 Adalite 的更高效优化器脚本，它在某些场景下表现良好。

- **将 LLM 集成到游戏及消费者硬件限制**：`@light4bear` 分享了一个由 LLM 驱动的在线宇宙游戏的想法，这引发了关于在消费级硬件上运行此类模型可行性的辩论。`@euclaise` 和 `@stefangliga` 指出，目前的模型可能需要显著缩减规模才能在典型的消费者配置上运行。

- **n-gram 语言模型的实时相关性**：`@johnryan465` 强调了一篇有趣的论文，该论文辩护了在神经 LLM 时代 n-gram 语言模型持续存在的重要性，并提出了一种 “infinite-gram” 模型，与传统的 n-gram 模型不同，它不限制 n 的范围。

**提到的链接**：

- [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)：在神经大语言模型 (LLM) 时代，n-gram 语言模型是否仍然具有相关性？我们的回答是肯定的，并展示了它们在文本分析和改进神经 LLM 方面的价值。然而，这需要...
- [Qwen-VL-Max - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen-VL-Max)：未找到描述
- [Weyaxi/Qwen-72B-Llama · Hugging Face](https://huggingface.co/Weyaxi/Qwen-72B-Llama)：未找到描述
- [Weyaxi/Helion-4x34B · Hugging Face](https://huggingface.co/Weyaxi/Helion-4x34B)：未找到描述
- [supertrainer2000/supertrainer2k/optim/adalite.py at master · euclaise/supertrainer2000](https://github.com/euclaise/supertrainer2000/blob/master/supertrainer2k/optim/adalite.py)：通过在 GitHub 上创建账号来为 euclaise/supertrainer2000 的开发做出贡献。
- [dataautogpt3/miqu-120b · Hugging Face](https://huggingface.co/dataautogpt3/miqu-120b)：未找到描述
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)：未找到描述
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1752803571035504858?s=20)：相当多的人问我 Mistral 的所有模型是否都基于 Meta 的 Llama。特别是由于通过在 Perplexity 上测试 Mistral Medium 也发现了输出的相似性...
- [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2)：未找到描述

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202643231611944980) (38 条消息🔥): 

- **对抗安全措施的后门训练**：`@if_a` 提供了一篇 [Anthropic 文章](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training)，指出 *sleeper agents*（具有欺骗性的 LLM）即使在经过安全训练后仍可能持续存在，这表明在训练期间引入的后门可能无法通过现有方法修复。
- **建议使用 Diff 算法进行许可证比较**：在关于识别开源许可证差异的讨论中，`@if_a` 建议使用 *diff 算法*，然后由 LLM 总结更改内容，前提是这些修改是基于标准模板进行的。
- **使用 YaRN 扩展上下文长度**：`@rememberlenny` 询问了使用 YaRN 扩展模型上下文长度的可行性，`@bloc97` 确认它可以用于扩展至 *128k tokens*，并观察到在较短上下文版本中表现更好。
- **好奇数学训练的成本**：`@jiha` 询问了训练一个类似于 *AlphaGeometry* 的数论模型的成本；`@Error.PDF` 回复称，训练估计需要 *数百万美元*，执行则需要 *数千美元*。
- **关于开源模型细节的问题**：`@420gunna` 询问了启发创建开源数据集的资源，在感谢 `@387972437901312000` 分享数据集并表达对 data-centric AI 的兴趣后，专门向其进行了咨询。

**提到的链接**：

- [teknium/OpenHermes-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-7B)：未找到描述
- [Cat Reaction GIF - Cat Reaction - Discover &amp; Share GIFs](https://tenor.com/view/cat-reaction-gif-27157580)：点击查看 GIF

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1202614144759627816) (6 条消息): 

- **Project Obsidian 咨询**：`@rabiussany` 寻求关于 **Project Obsidian** 目标的澄清，并需要帮助理解 Nous 内部当前的活动。
- **项目重点简要说明**：`@_3sphere` 告知频道描述显示重点在于 **multimodality**，但也提到该项目的活跃度目前似乎较低。
- **新人渴望参与**：`@rabiussany` 表现出参与 Nous Research 项目的热情，并询问了加入的方式。
- **Obsidian 项目状态更新**：`@teknium` 澄清 **Project Obsidian** 基本上已经完成并发布，引导 `@rabiussany` 前往 Nous 的 [huggingface](https://huggingface.co/Nous) 获取 Obsidian 模型。
- **期待未来合作**：`@rabiussany` 表示愿意等待未来贡献 Nous Research 项目的机会。
  

---

### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1202735168159354920) (3 messages): 

- **`moondream1` 的自定义 Pipeline**：用户 `@vikhyatk` 在 Hugging Face 平台上为 `vikhyatk/moondream1` 引入了自定义 Pipeline，允许用户使用特定的 Python 代码片段调用该 Pipeline。你可以在[这里](https://huggingface.co/vikhyatk/moondream1/discussions/6)找到该 Pipeline 和相关讨论。

- **推广创意评估基准**：`@Vipitis` 分享了关于提议一种评估语言模型创意的新型评估基准（Evaluation Benchmark）的消息，并在 [Twitter](https://twitter.com/Vipitis/status/1752699776766988309) 上分享了更新。

- **Resume QA Space 发布**：用户 `@not-lain` 发布了一个 Resume QA Space，旨在改进简历并优化面试回答。在 [Hugging Face space](https://huggingface.co/spaces/not-lain/resume-qa) 上了解更多关于这个实用工具的信息。

- **Han Instruct 数据集介绍**：用户 `@pythainlp` 分享了 Han Instruct 数据集，这是一个为各种问答提供深刻见解的数据集。点击[这里](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v1.0)了解更多关于该数据集的信息。

- **乌克兰语 wav2vec2 bert 模型发布**：用户 `@Yehor` 宣布了乌克兰语 wav2vec2 bert 模型，并提供了 Discord 服务器和 Telegram 群组的链接，用于语音识别（Speech Recognition）相关的讨论。该模型及其他资源可以在[这里](https://huggingface.co/Yehor/w2v-bert-2.0-uk)找到。

**提到的链接**：

- [Resume Qa - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/resume-qa)：未找到描述
- [vikhyatk/moondream1 · add pipeline](https://huggingface.co/vikhyatk/moondream1/discussions/6)：未找到描述
- [pythainlp/han-instruct-dataset-v1.0 · Datasets at Hugging Face](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v1.0)：未找到描述
- [Yehor/w2v-bert-2.0-uk · Hugging Face](https://huggingface.co/Yehor/w2v-bert-2.0-uk)：未找到描述
- [来自 thecollabagepatch (@thepatch_kev) 的推文](https://x.com/thepatch_kev/status/1752129930404696134)：当你的歌手某天莫名其妙听起来像猫王时，@fffiloni 的 dreamtalk 就该派上用场了 😂 这周我们只是在船长座上找点乐子，下周... @_buildspace
- [Hugging Face 上的 @s3nh："GPU 贫民视角：量化。今天我想和大家分享我的 notebook 插件……"](https://huggingface.co/posts/s3nh/851992122690412)：未找到描述
- [Hugging Face 上的 @natolambert："今天，我们发布了首个预训练的开放语言模型 (OLMo)……"](https://huggingface.co/posts/natolambert/114173328374820)：未找到描述
- [Hugging Face 上的 @psinger："很高兴分享 H2O-Danube-1.8b，这是一个仅在 1T 数据上训练的小型 1.8b 模型……"](https://huggingface.co/posts/psinger/455307248098208)：未找到描述
- [Hugging Face 上的 @santiviquez："今天制作这张图表非常有趣。如果有人问你为什么……"](https://huggingface.co/posts/santiviquez/295325502020879)：未找到描述
- [Hugging Face 上的 @gsarti："🔍 今日语言模型可解释性与分析精选：基于梯度的语言……"](https://huggingface.co/posts/gsarti/622879789886281)：未找到描述
- [Locutusque/UltraTextbooks · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks)：未找到描述
- [miqu-70b Chat - a Hugging Face Space by freQuensy23](https://huggingface.co/spaces/freQuensy23/miqu-chat)：未找到描述
- [joshuasundance/mtg-coloridentity-multilabel-classification · Hugging Face](https://huggingface.co/joshuasundance/mtg-coloridentity-multilabel-classification)：未找到描述
- [Tables - a Hugging Face Space by sid27](https://huggingface.co/spaces/sid27/tables)：未找到描述
- [MLX | Apple Silicon 上的 Mistral-7B-Instruct](https://www.youtube.com/watch?v=cjl2ADP8JLQ&t=79s)：你能在 Apple Silicon 上使用 MLX 运行来自 Mistral AI 的 Mistral-7B-Instruct-v0.2 吗？让我们一探究竟。-------------------------------------------------------------...
- [Best Image Models Demo - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo)：未找到描述
- [浏览器中的 ColBERT 推理](https://colbert.aiserv.cloud/)：未找到描述

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1202577314446319686) (241 条消息🔥🔥): 

- **庆祝 "moonddream1" 模型下载量**：用户 `@not_lain` 宣布他们的模型 `moonddream1` 已达到 2500 次下载，庆祝这一里程碑式的成就。
- **视频生成工具讨论**：`@ch33zw2zard` 询问了目前可用的最佳视频生成工具，表示对 `moonvalley` 之外的替代方案感兴趣；他们欢迎各种推荐。
- **Hugging Face API 使用说明**：`@ram1428` 寻求了解在使用 Hugging Face API token 调用 `llama2` 等模型时，计算是在 Google Colab 还是在 Hugging Face 的服务器上进行的。澄清结果是：使用 API key 可以无需下载模型即可使用，计算任务会被卸载到 Hugging Face 的服务器上。
- **探索项目的 LLM 托管方案**：`@woodenrobot` 向社区征求关于开源项目免费大语言模型（LLM）托管的建议，讨论涉及免费层级以及与 Colab 或 Kaggle 等其他服务的集成可能。
- **学术项目中的数据集困境**：`@akvnn` 与 `@doctorpangloss` 就如何为旨在发表论文的分类问题选择合适且新颖的数据集进行了深入讨论。尽管收到了建议和意见，`@akvnn` 仍未确定具体主题，但考虑与牙科专家合作获取独特的牙科扫描数据。

**提到的链接**：

- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989)：统一 NLP 和软件工程的视角：代码语言模型综述。在这项工作中，我们系统地回顾了使用语言模型进行代码处理的最新进展，涵盖了 50 多个模型、30 多个评估任务、170 多个数据集和 700 多篇相关工作。我们分解了...
- [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://arxiv.org/abs/2303.12570)：RepoCoder：通过迭代检索和生成的仓库级代码补全。仓库级代码补全的任务是基于仓库更广泛的上下文继续编写未完成的代码。虽然对于自动化代码补全工具来说，利用...
- [Code Evaluation - a Vipitis Collection](https://huggingface.co/collections/Vipitis/code-evaluation-6530478d8e4767ecfe1bc489)：未找到描述
- [Paper page - DevEval: Evaluating Code Generation in Practical Software Projects](https://huggingface.co/papers/2401.06401)：未找到描述
- [GitHub - huggingface/llm-ls: LSP server leveraging LLMs for code completion (and more?)](https://github.com/huggingface/llm-ls)：利用 LLM 进行代码补全（以及更多功能？）的 LSP 服务器 - GitHub - huggingface/llm-ls
- [torch_geometric.nn.pool.global_mean_pool &mdash; pytorch_geometric  documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html#torch_geometric.nn.pool.global_mean_pool)：未找到描述
- [GitHub - Silver267/pytorch-to-safetensor-converter: A simple converter which converts pytorch bin files to safetensor, intended to be used for LLM conversion.](https://github.com/Silver267/pytorch-to-safetensor-converter)：一个简单的转换器，可将 pytorch bin 文件转换为 safetensor，旨在用于 LLM 转换。- GitHub - Silver267/pytorch-to-safetensor-converter
- [no title found](https://api.endpoints.huggingface.cloud/#get-/v2/endpoint/-namespace-/-name-/logs))：未找到描述
- [GitHub - TabbyML/tabby: Self-hosted AI coding assistant](https://github.com/TabbyML/tabby)：自托管 AI 编程助手。通过在 GitHub 上创建账号为 TabbyML/tabby 的开发做出贡献。
- [lowres/anime · Datasets at Hugging Face](https://huggingface.co/datasets/lowres/anime)：未找到描述
- [lowres/sukasuka-anime-vocal-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/lowres/sukasuka-anime-vocal-dataset)：未找到描述

  

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1202569480816042054) (12 messages🔥): 

- **为新人提供的免费 LLM API 访问**：用户 `@erlonidasap` 询问是否有任何免费的大语言模型 (LLM) API，`@not_lain` 告知了关于 **Hugging Face 为 AI 模型和 Spaces 提供的免费开放 API**，并分享了一份详细的演示文稿，在 [第 16 页](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing) 有更多信息。
- **预训练 LLM 导览**：在他的 **Google Slides 演示文稿** 第 16 页，`@not_lain` 介绍了如何使用预训练 LLM，并提供了可点击的黄色链接供进一步探索。
- **分享即美德**：包括 `@tadeodonegana` 在内的社区成员对 `@not_lain` 分享演示文稿的努力表示感谢，`@not_lain` 对积极的反馈表达了谢意。
- **跨国就业策略提案**：`@nikolacurovic` 是一位非美国籍的高级 Web 开发人员，他提议与美国公民合作在 LinkedIn 上申请工作，同时由他远程履行工作职责，并建议对获得的任何工作机会按比例进行财务分配。

**提到的链接**：

[introduction to using pretrained LLMs](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing)：使用预训练 LLM 的介绍。Hafedh hichri 去年发布，在图像分类、图像分割等不同任务上曾达到 SOTA。

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1202542074617143337) (2 messages): 

- **Google 释放 Mobile Diffusion 能量**：`sta._.no` 分享了对 Google 新推出的 386M Diffusion 模型的兴奋之情，并对可能的开源表示好奇。[MobileDiffusion](https://blog.research.google/2024/01/mobilediffusion-rapid-text-to-image.html) 拥有在移动设备上实现亚秒级快速文本生成图像的能力，性能优于 Stable Diffusion 和 DALL·E 等高参数的前辈。

- **I-BERT 加速 RoBERTa 推理**：`andysingal` 介绍了 [I-BERT 模型](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/ibert)，这是 RoBERTa 的纯整数（integer-only）量化版本，能够使推理速度提升高达 **四倍**。这种效率为在边缘设备上实现更快的自然语言处理任务打开了大门。

**提到的链接**：

- [MobileDiffusion: Rapid text-to-image generation on-device – Google Research Blog](https://blog.research.google/2024/01/mobilediffusion-rapid-text-to-image.html)：无描述
- [I-BERT](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/ibert)：无描述

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1202566037099384853) (9 messages🔥): 

- **ColBERT 实战**：`@andreer_vespa_99582` 分享了一个工具，用于实时可视化托管在 [colbert.aiserv.cloud](https://colbert.aiserv.cloud) 上的 ColBERT 模型中 token 对文档相似度的贡献。`@cakiki` 和 `@weyaxi` 等用户对该项目表示了热情和支持。
  
- **MiQU 免费聊天 Demo**：`@frequesny` 发布了一个关于 MiQU 模型的免费 Demo，该模型达到了 GPT-4 级别，是 Mistral AI 泄露的结果。该 Demo 可以在 [Hugging Face 项目页面](https://huggingface.co/spaces/freQuensy23/miqu-chat)上体验。
  
- **UltraTextbooks 数据集发布**：`@locutusque` 介绍了 "UltraTextbooks" 数据集，该数据集结合了合成和人工编写的教科书，用于高级 NLP 任务，托管在 [Hugging Face 数据集页面](https://huggingface.co/datasets/Locutusque/UltraTextbooks)。`@stroggoz` 对这一新资源表示了赞赏。
  
- **VS Code、Jupyter 与远程 GPU 的结合**：`@chongdashu` 撰写了一篇关于在远程 GPU 服务器上将 Visual Studio Code 与 Jupyter Notebooks 集成的指南，旨在优化机器学习开发工作流。详细指南已发布在 [Medium](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45) 上。
  
- **HuggingFace Spaces 上的无服务器图像相似度工具**：`@omerxfaruq` 使用 Upstash Vector 开发了一个无服务器图像相似度工具，并在 [Find Your Twins Space](https://huggingface.co/spaces/omerXfaruq/FindYourTwins) 进行了演示，同时在 [Hugging Face 博客文章](https://huggingface.co/blog/omerXfaruq/serverless-image-similarity-with-upstash-vector)中进行了详细介绍。该方案侧重于利用 HuggingFace 生态系统和 Upstash 来简化后端和前端的复杂性。

**提到的链接**：

- [ColBERT Inference in the Browser](https://colbert.aiserv.cloud): 未找到描述
- [miqu-70b Chat - a Hugging Face Space by freQuensy23](https://huggingface.co/spaces/freQuensy23/miqu-chat): 未找到描述
- [Connecting Visual Studio Code with Jupyter Notebooks on a remote GPU server instance](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45): 毫无保留地发挥三者的强大功能
- [FindYourTwins - a Hugging Face Space by omerXfaruq](https://huggingface.co/spaces/omerXfaruq/FindYourTwins): 未找到描述
- [Serverless Image Similarity with Upstash Vector and Huggingface Models, Datasets and Spaces](https://huggingface.co/blog/omerXfaruq/serverless-image-similarity-with-upstash-vector): 未找到描述
- [Locutusque/UltraTextbooks · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks): 未找到描述

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1202681670516351078) (28 messages🔥): 

- **Mamba 演示即将到来**：用户 `@ericauld` 提议为读书小组演示 Mamba 或相关主题，演示定于周五（2 月 9 日），时间灵活，包括加州时间傍晚。`@chad_in_the_house` 提供了一个 [When2meet 链接](https://www.when2meet.com/?23471427-n4DUl)来确定具体时间。
- **演示录制咨询**：在期待关于法律领域 AI 挑战的讨论时，`@k4rolina_n` 请求对会议进行录制，以便那些可能晚到的人观看。`@chad_in_the_house` 确认了使用 OBS 进行录制的意图。
- **法律领域的 AI 引起关注**：`@chad_in_the_house` 宣布了一个关于法律领域 AI 难点的演示，该演示将于次日（从消息发布时算起）美国东部时间下午 1-2 点在 Discord 语音频道举行。
- **钻研压缩算法**：`@chad_in_the_house` 对一篇关于压缩算法的论文发表了评论，提到评估标准中缺乏更广泛的基准测试（如 MT），也缺少像 speculative decoding（投机解码）这样的推理技术。
- **论文思考**：在讨论一篇关于压缩的论文时，`@chad_in_the_house` 最初觉得它很有趣，但指出压缩的“最佳”方法尚未明确定义，随后承认了该论文的全面性。

**提到的链接**：

[Eric's Presentation - When2meet](https://www.when2meet.com/?23471427-n4DUl): 未找到描述

  

---

### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968) (1 messages): 

- **高级 Dreambooth 训练上线**：用户 `@linoy_tsaban` 宣布，得益于 `@brandostrong` 重要的社区贡献，Stable Diffusion (SDXL) 的 **Dreambooth LoRA** 训练现已在 `diffusers` 中可用。它包含了一些高级特性，如 pivotal tuning、自定义 captions、prodigy 优化器以及新增的 noise offset 支持以提升效果，同时所需的计算资源更少。
  
- **Dreambooth LoRA 训练脚本已就绪**：Hugging Face 的 GitHub 为准备利用这些新功能微调 SD 模型的用户提供了 [高级训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py)。
  
- **发布引起 Twitter 关注**：该新功能的发布在 Twitter 上引起了关注，`@linoy_tsaban` 分享了这一公告，具体内容可见 [发布推文](https://twitter.com/linoy_tsaban/status/1753022391079620663)。

**提到的链接**：

[diffusers/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py at main · huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py)：🤗 Diffusers：PyTorch 中用于图像和音频生成的先进扩散模型 - huggingface/diffusers

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202589652893179924) (1 messages): 

- **关于在 Google TPU 上进行 Diffusion 训练的咨询**：`@pawkanarek` 询问 [公告](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968) 中提到的高级扩散训练是否可以在 Google TPU 上运行。社区目前尚未就 TPU 兼容性给出回复。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1202638569043402802) (18 messages🔥): 

- **修复函数名解决错误**：`@swetha98` 通过更改函数名解决了 donut 模型微调代码中的一个错误，并确认 Merve 提供的解决方案运行良好。
- **分享错误时截图的清晰度**：`@johko990` 建议 `@swetha98` 在未来展示代码相关问题时，分享屏幕截图而不是手机拍摄的照片，以确保清晰度。
- **寻求用于用户操作的聚类模型建议**：`@amarcel` 寻求聚类模型的建议，以便在 1.5 万张截图中识别重复的用户操作序列（例如：点击、复制、粘贴），同时需要考虑到用户可能存在的误操作。
- **使用附加操作进行模型训练**：`@banaanbakje` 分享了通过将操作附加到截图来训练模型的经验，并建议编写一个函数来扫描相似的用户操作序列。
- **分享基于 EfficientNet 的模型构建资源**：针对 `@amarcel` 分析用户操作的场景，`@banaanbakje` 提供了一个 [关于使用 PyTorch 和 EfficientNet 构建 AI 驱动游戏机器人的指南链接](https://www.akshaymakes.com/blogs/pytorch)，这可能对该概念有所帮助。


**提到的链接**：

[Akshay 的个人网站](https://www.akshaymakes.com/blogs/pytorch)：我是一名机器学习爱好者。查看我的项目和博客。

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1202524818025615400) (5 messages): 

- **寻求 AWS SageMaker 指导**：用户 `refik0727` 正在寻找关于如何使用 **AWS SageMaker**、**Mistral**、**Llama** 或 **Tapas**，通过 CSV 文件或连接到本地数据库来构建聊天机器人的教程或 GitHub 代码。
- **ChatGPT 作为学习资源**：`@akshit1993` 推荐 **ChatGPT** 作为目前学习 `refik0727` 感兴趣主题的最佳资源。
- **对安装问题感到困惑**：用户 `@.sgp` 最初认为按照文档中的安装说明操作即可，但在安装过程中遇到了**错误**。
- **过时的 CUDA Toolkit 导致错误**：`@joshuasundance` 认为 `@.sgp` 的安装错误可能是由于使用了**过时版本的 CUDA toolkit** 导致的。
  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202589652893179924) (1 messages): 

- **TPU 兼容性问题未获解答**：用户 `@pawkanarek` 询问**高级 Diffusion 训练**是否能在 Google TPU 上运行，并引用了[此处](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968)的公告。然而，在现有消息中未提供进一步的信息或解答。

**提到的链接**：

[Discord - 与好友和社区聊天的新方式](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968)：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的好友和社区聊天、聚会并保持紧密联系。

  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1202875316595593276) (1 messages): 

- **Gradio 发布弹出模态框组件**：`@yuviii_` 宣布发布了一个新的 **Gradio 自定义组件** `𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕`，由 Ali Abid 创建。该组件可用于在 Gradio Apps 中显示许可协议、提示用户登录、警报、分步指南、上下文帮助或确认。

- **探索 𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕 组件**：你可以通过访问[提供的 Hugging Face Space](https://huggingface.co/spaces/aliabid94/gradio_modal)在自己的 Gradio 应用中查看并实现 `𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕`。该组件通过提供各种弹出功能，可以增强 Gradio 应用程序中的用户交互。

**提到的链接**：

[gradio_modal V0.0.1 - aliabid94 的 Hugging Face Space](https://huggingface.co/spaces/aliabid94/gradio_modal)：未找到描述

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1202553466124369951) (298 messages🔥🔥): 

- **关于 "LLaMa 1.6" 成功的传闻**：
  用户 `@kache_` 简要提到了他们的满意度，称 "**llava 1.6** 非常出色"，但未提供关于其性能或与其他模型对比的更多细节。

- **Hacker News 焦点**：`@itali4no` 指出 `@122380520226160640` 撰写的一篇文章在 Hacker News 上引起了关注，并链接了相关讨论，消息称：“你的 Reddit 文章在橙色网站上表现亮眼 <@122380520226160640>，如果你还没看到的话：[Hacker News 讨论](https://news.ycombinator.com/item?id=39215242)。”

- **Bard 图像生成中的水印争议**：`@max_voltage` 分享了对 Bard 升级后的图像生成功能的担忧，认为其过于受限于负责任的 AI 原则，嵌入水印以区分 AI 创作的视觉内容，并带上厌恶的表情符号链接了相关讨论。

- **Imagen 2 生成图像质量受质疑**：`@thejonasbrothers` 注意到所有 Imagen 2 图像中明显的噪声，并想知道为什么不以更高效的格式（如 "80% 质量的 jpeg"）返回输出，表达了与 SDXL 输出的批判性对比。

- **关于 Autoencoder 敏感性和训练的论述**：知名用户如 `@drhead` 和 `@thejonasbrothers` 就 Autoencoder 中 latent space 的敏感性、噪声模式的影响以及这些模式在模型训练期间可能的演变进行了深入的技术讨论。对话参考了包括 Segmind Vega 在内的各种模型，以及它们与 SDXL VAE 相比观察到的差异。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/823813159592001537/823813160075132991/1202377957449154590)：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的好友和社区聊天、聚会并保持紧密联系。
- [终于来了！初探 Google 全新 Imagen 2 & Image FX 界面！](https://youtu.be/2CB1Wb0b6dA?si=cKLpPhzXpi-ApjC4)：ImageFX 是一项实验性技术，允许你生成自己的合成图像。ImageFX 由 Google 的 Imagen 2 提供支持，并使用 Google DeepMind...
- [无标题](https://news.ycombinator.com/item?id=39215242)：未找到描述
- [ImageFX](https://aitestkitchen.withgoogle.com/tools/image-fx)：未找到描述
- [GitHub - openai/consistencydecoder: Consistency Distilled Diff VAE](https://github.com/openai/consistencydecoder)：Consistency Distilled Diff VAE。通过在 GitHub 上创建账号来为 openai/consistencydecoder 的开发做出贡献。

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (1 messages): 

felfri_: https://allenai.org/olmo
  

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1202529743363121152) (137 条消息🔥🔥): 

- **寻找增强记忆力**：`@flyyt4` 询问了有关改进 ChatGPT 等 LLM 长期记忆的库或工具，这促使 `@fabguy` 简单提到了 "memGPT"。
- **高级 Prompt 艺术**：`@wolfspyre` 分享了一个文本生成的[高级 Prompt 示例链接](https://funkpd.com/devlog/prompt-examples/)，强调了 Chain of Thought、Tree Thinking 和 Prompt 压缩的重要性。
- **在 Hugging Face 上发现 LLaMA**：`@pierrunoyt` 发布了 [Hugging Face 的 moondream1 项目页面](https://huggingface.co/spaces/vikhyatk/moondream1)的链接，`@fabguy` 认为这是一个不错的发现。
- **优化 GPU Offload**：`@yagilb` 建议开启 GPU Offload，以解决 `@pierrunoyt` 抱怨的模型加载速度慢的问题。
- **下载正确的 LLaMA 模型版本**：在 `@.veski` 安装了一个庞大的 LLaMA 模型后，`@heyitsyorkie` 指导他们从 Hugging Face 下载 GGUF 量化版本，以便更好地兼容 LM Studio。

**提到的链接**：

- [moondream1 - a Hugging Face Space by vikhyatk](https://huggingface.co/spaces/vikhyatk/moondream1)：未找到描述
- [Devlog: Make Amazing GPT copy! Prompt Examples of Chain of Thought, Tree Thinking, and more - FunkPd](https://funkpd.com/devlog/prompt-examples/)：欢迎来到文本生成的高级 Prompt 示例世界——这是一个创意与逻辑交织的领域，复杂性不再是障碍。
- [llama.cpp/README-sycl.md at ce32060198b7e2d6a13a9b8e1e1369e3c295ae2a · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/ce32060198b7e2d6a13a9b8e1e1369e3c295ae2a/README-sycl.md?plain=1#L64)：Facebook 的 LLaMA 模型在 C/C++ 中的移植版本。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
- [GitHub - facebookresearch/llama: Inference code for LLaMA models](https://github.com/facebookresearch/llama)：LLaMA 模型的推理代码。通过在 GitHub 上创建账号为 facebookresearch/llama 的开发做出贡献。

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1202595073406144542) (32 条消息🔥): 

- **用户对 ChatGPT 表示不满**：`@goldensun3ds` 对 ChatGPT 变得“无用”表示沮丧，特别是其网页浏览能力下降以及审查和 Context Size 等看似随意的限制。他们将 ChatGPT 的主导地位和局限性比作 Intel 在 CPU 领域的历史。
- **对不同模型版本的质量担忧**：`@kujila` 评论了 **Q4** 优于 **Q2**，称 Q4 “谎言少得多”，暗示了模型可靠性的提高。
- **寻找无审查模型**：在回复 `@zono50.` 寻找最佳无审查 7B 模型时，`@goldensun3ds` 建议在写故事时避免使用小型模型，并分享了使用几种模型的个人经验，包括 **The Bloke Goat Storytelling 70B Q3KS 和 The Bloke Dolphin 2 6 Mixtral 7B Q4**。
- **用户排查故障并寻找无护栏模型**：在对 **Mistral 7B Instruct** 的体验感到失望后，`@p4stoboy` 征求关于最佳无护栏模型的建议。多位用户讨论了解决方案，包括在频道中发布链接的 `@ptable`，以及建议在 LM Studio 中编辑 AI 消息以添加缺失代词的 `@goldensun3ds`。
- **讨论大型模型的性能和 Context 限制**：`@binaryalgorithm` 谈到了在大 Context 下保持故事设定一致性的必要性，并提到像 **GPT-4** 这样的模型在输出方面受到限制，影响了故事的连贯性。他们推测了通过 API 使用具有更大 Context Size 模型的成本和可行性。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1185654311276003448)：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。

  

---

### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1202529860225081385) (12 messages🔥): 

- **关于 GGUF 格式的见解**：`@fabguy` 承认 **GGUF** 是比 `.chk` 更好的格式，因为它显式加载张量（tensors），但由于对 **llama.cpp implementation** 了解有限，他们表达了谨慎态度。尽管有此顾虑，`@fabguy` 提到在过去 6 个月中使用 **LM Studio** 没有出现任何问题。

- **查找并移除模型适配器 (Model Adapters)**：`@elsatch` 询问如何在下载后从视觉模型中移除 **model adapters**，特别是在为 **llava** 安装了 **fp16** 后想要切换到 **Q4**。`@heyitsyorkie` 指出可以在 *local models folder*（本地模型文件夹）中找到它们。

- **重大会议前的及时支持请求**：`@docorange88` 紧急请求在 CST 中午 12 点关于软件讨论的重要会议之前回复其邮件。`@yagilb` 优先处理了该回复，对延迟表示歉意，并确保了及时的沟通。

- **在磁盘上定位模型适配器的指南**：`@fabguy` 帮助了 `@elsatch`，建议在 “my models” 视图中右键点击模型，即可直接进入磁盘上的模型文件夹。

- **LM Studio 版本差异影响模型加载**：`@foobar8553` 报告了一个问题，由于模型加载到 RAM 和 GPU 的方式可能发生了变化，他们无法使用 **LM Studio 0.2.12** 加载 **mixtral-8x7B ... Q5_K_M model**，尽管在 **0.2.10** 版本中可以加载。`@yagilb` 提出提供 **0.2.10** 版本进行对比测试，并询问了所使用的操作系统。
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1202562558586912798) (44 messages🔥): 

- **双 3090 运行大型模型**：`@silverstar5654` 询问关于使用双 3090 GPU 配合 LM Studio 加载大型模型的问题。`@pefortin` 建议只需勾选 offload to GPU 选项框，前提是驱动程序已正确安装。

- **LM Studio ARC 支持推测**：`@goldensun3ds` 分享了一个链接，询问 LM Studio 是否会支持 ARC，对此 `@rugg0064` 给出了简短的肯定回答。

- **Mixtral 在 VRAM 中的查询**：`@docorange88` 询问关于在两块 4090 或 RTX6000 GPU 上运行 Mixtral 的问题，以及内存需求是否可以卸载到 CPU RAM。`@foobar8553` 分享了他们在 4070 上利用额外的系统 RAM 运行 12 GB VRAM 的 Mixtral 的经验。

- **多 GPU 上的模型性能**：`@ellric_` 询问了 8bit 与 4bit 模型的性能和质量差异，以及增加更多 GPU 是否会提高速度。`@mahdiyari` 指出，除非使用 NVLink，否则使用两个 GPU 可能会更慢，并分享了一个用于 LLM 推理的 GPU benchmark 链接。
  
- **LM Studio 上的模型加载问题**：`@lowkey9920` 在配备 3080 GPU 和 64GB RAM 的系统上运行 3.8 GB 的模型（特别是 "magicoder-s-ds"）时遇到困难。`@binaryalgorithm` 等其他人也报告了问题，同时指出其他模型加载正常。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1148200624924667914/1202195308914671649): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1202885906290446376) (1 messages): 

- **Llava 1.6 支持咨询**：用户 `@_koopman` 询问了对 **llava 1.6** 的支持情况，指出 LM Studio 目前尚未将其识别为视觉模型。用户对集成表示期待。
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1202585069567352843) (3 messages): 

- **探索多服务器运行 AutoGen**：`@ellric_` 对使用两个不同的服务器而非单台机器运行 **AutoGen** 的能力感到好奇。他们考虑尝试这种设置。
- **运行两个 LM Studio 实例**：`@_anarche_` 确认，如果有足够的硬件支持，可以在不同的端口上使用 **LM Studio** 同时运行两个模型。他们为 `@ellric_` 计划的实验提供了指导。
  

---

### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1202797003508158527) (2 messages): 

- **LM Studio 与 Open Interpreter 的集成难题**：用户 `@raiderduck` 在将 **OI (Open Interpreter)** 与 **LM Studio** 集成时遇到问题，启动服务器后响应变得毫无意义。他们已成功将 *Solar instruct* 加载到 LM Studio 中，该模型在聊天界面运行良好，但在配合 OI 使用时会失败，导致出现自我回复和胡言乱语。

- **使用 GPT-4 的 OI 成本高昂**：`@raiderduck` 提到，**OI** 在与 **GPT-4** 搭配时运行顺畅，但高强度的使用导致 **一周产生了 250 美元的账单**。他们目前正在寻求针对 OI 的服务器预设设置建议，以降低这些成本。
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1202639362471366746) (33 messages🔥): 

- **OLMo 进入 LLM 竞技场**：`@sk5544` 随着 [OLMo 论文](https://allenai.org/olmo/olmo-paper.pdf) 的发布引发了讨论，社区关注其方法论和许可协议，`@hailey_schoelkopf` 澄清说 AI2 尚未发布聊天模型或界面。
- **关于 OLMo 评估分数的辩论**：`@mistobaan` 对 OLMo 在 GSM8K 基准测试中平庸的分数表示惊讶，这引发了关于 Tokenizer 问题和格式规范问题的辩论，`@stellaathena` 指出 SFT 模型可能会有更好的表现。
- **OLMo 训练代码受到审视**：社区剖析了 OLMo 的 Warmup 调度器，`@jerry0478` 解释了其重置优化器状态的功能，而 `@mistobaan` 指出缺乏讨论该 Warmup 技术的论文，并参考了一个替代的 [PyTorch Warmup 工具](https://github.com/Tony-Y/pytorch_warmup)。
- **OLMo 归一化选择的探讨**：关于 OLMo 使用 Layer Norm 还是 RMSNorm 的讨论愈发激烈，`@ad8e` 澄清他们使用了没有可学习缩放（learnable scales）的 Layer Norm，而 `@the_random_lurker` 对 AI2 强调数据质量和模型评估的说法持怀疑态度。
- **社区对 AI2 评估指标的评价**：`@the_random_lurker` 批评了 AI2 对评估指标的选择，暗示他们可能为了与 `Llama 2` 进行有利对比而刻意挑选（cherry-picked）了结果，并对主论文中未出现但在 HF 页面上展示的基准测试表示质疑。

**提到的链接**：

- [OLMo/olmo/optim.py at main · Mistobaan/OLMo](https://github.com/Mistobaan/OLMo/blob/main/olmo/optim.py#L544C7-L544C28)：OLMo 的建模、训练、评估和推理代码 - Mistobaan/OLMo
- [phi-playground/notebooks/evaluation/evaluation_GSM8k_transformers_colab.ipynb at main · bmx-ai/phi-playground](https://github.com/bmx-ai/phi-playground/blob/main/notebooks/evaluation/evaluation_GSM8k_transformers_colab.ipynb)：一系列用于玩转 microsoft/phi-* 模型的工具 - bmx-ai/phi-playground
- [GitHub - Tony-Y/pytorch_warmup: Learning Rate Warmup in PyTorch](https://github.com/Tony-Y/pytorch_warmup)：PyTorch 中的学习率预热。通过在 GitHub 上创建账号为 Tony-Y/pytorch_warmup 做出贡献。

  

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1202528783098323045) (159 条消息🔥🔥): 

- **N-gram 扩展引发辩论**：`@xylthixlm` 分享了一篇关于将 n-gram 扩展到数万亿个 token 的论文，引发了关于其内在价值以及与大型语言模型 (LLM) 协同作用的讨论。包括 `@ai_waifu` 和 `@johnryan465` 在内的几位用户考虑将 n-gram 模型与 LLM 结合以提升整体性能和泛化能力，而 `@catboy_slim_` 则评论了此类模型潜在的分布内偏差 (in-distribution bias)。

- **显微镜下的 Infinigram**：研究社区对 Infinigram 模型表现出了兴趣和怀疑；`@_inox` 指出该模型仅在与 LLM 结合时进行了基准测试，而非独立测试，这引发了关于其独立能力的讨论。

- **为 LLM 标记回译 (Backtranslations)**：`@resonancia` 发起了一场关于应用机器翻译领域技术（如标记回译数据）来提高语言模型效率的讨论。`@teknium` 提到虽然目前没有 Hermes 系列模型使用此类技术，但在其他工作中可以看到类似的方法，暗示了在未来模型中应用的可能性。

- **合成数据生成策略评估**：用户们根据 `@blagdad` 的见解讨论了合成数据生成的不同方法，特别提到了 MCTS (蒙特卡洛树搜索) 以及 beam search 在多样性方面的潜在缺点。大家一致认为，多样化的合成数据是关键，这可能需要使用替代采样方法。

- **机器学习社区亮点**：`@ge0.io` 宣布发布专为多模态工作量身定制的 Gaussian Adaptive Attention 库，而 `@digthatdata` 分享了新论文和即将开展的研究链接，例如 Amortized Text-to-Mesh (AToM)，展示了 AI 研究领域的动态特性。

**提到的链接**：

- [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](http://arxiv.org/abs/2401.17377)：n-gram 语言模型在神经大型语言模型 (LLM) 时代是否仍然具有相关性？我们的答案是肯定的，我们展示了它们在文本分析和改进神经 LLM 方面的价值。然而这需要...
- [AToM: Amortized Text-to-Mesh using 2D Diffusion](https://snap-research.github.io/AToM/)：AToM：使用 2D 扩散的摊销文本到网格，2023。
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)：最近，具有高效硬件感知设计的状态空间模型 (SSM)，即 Mamba，在长序列建模方面展现了巨大潜力。纯粹构建高效且通用的视觉骨干网络...
- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)：我们提出了一种可扩展的方法，通过自动为人类编写的文本标注相应的指令，来构建高质量的指令遵循语言模型。我们的方法被称为指令...
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)：改变预训练模型的行为——例如，提高其在下游任务中的性能或减轻预训练期间学到的偏差——是开发机器学习模型时的常见做法...
- [AToM: Amortized Text-to-Mesh using 2D Diffusion](https://arxiv.org/abs/2402.00867)：我们介绍了摊销文本到网格 (AToM)，这是一个同时针对多个文本提示进行优化的前馈文本到网格框架。与现有的通常需要耗时...的文本到 3D 方法相比。
- [Tagged Back-Translation](https://aclanthology.org/W19-5206/)：Isaac Caswell, Ciprian Chelba, David Grangier。第四届机器翻译会议论文集（第 1 卷：研究论文）。2019。
- [Tagged Back-translation Revisited: Why Does It Really Work?](https://aclanthology.org/2020.acl-main.532/)：Benjamin Marie, Raphael Rubino, Atsushi Fujita。第 58 届计算语言学协会年会论文集。2020。
- [GitHub - gioannides/Gaussian-Adaptive-Attention](https://github.com/gioannides/Gaussian-Adaptive-Attention)：通过在 GitHub 上创建一个账户来为 gioannides/Gaussian-Adaptive-Attention 的开发做出贡献。

### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1202650548852228189) (11 messages🔥): 

- **自发的研究交集**：`@norabelrose` 发现了一篇关于上下文语言学习 (ICLL) 的[研究论文](https://arxiv.org/abs/2401.12973)，与其自身的研究兴趣高度契合，对此表示惊讶。
- **分享关于“上下文神经元”的论文**：`@norabelrose` 分享了一篇探讨语言模型中[上下文神经元 (contextual neurons)](https://arxiv.org/abs/2311.00863) 概念的论文链接，特别是二阶电路中一个对德语文本产生激活的神经元。
- **困惑度 vs. 开放式文本生成**：`@nostalgebraist` 指出 ∞-gram 可以改善神经语言模型 (LM) 的困惑度 (perplexity)，但可能会因导致奇怪的错误和检索到无关的 Token 而损害开放式文本生成。
- **N-gram 在检索而非连贯性方面的优势**：`@nostalgebraist` 认为 n-gram 模型更适合类似于长尾检索的任务，而非需要连贯文本生成的任务，困惑度结果反映了这种二分性。
- **对比 MoE 攻击与 ∞-gram 检索**：`@xa9ax` 将 ∞-gram 在开放式文本生成中遇到的困难与 MoE 模型攻击进行了类比，在后者中，错误的专家选择会损害输出。他引用了一篇关于 [MoE 攻击的论文](https://arxiv.org/pdf/2210.10253.pdf) 并表达了对该研究领域的兴趣。

**提到的链接**：

- [In-Context Language Learning: Architectures and Algorithms](https://arxiv.org/abs/2401.12973)：大规模神经语言模型展现出卓越的上下文学习 (ICL) 能力：它们可以从作为输入的上下文中推断出新的函数。目前我们对这一现象的理解大多……
- [Training Dynamics of Contextual N-Grams in Language Models](https://arxiv.org/abs/2311.00863)：先前的研究已经表明语言模型中存在上下文神经元，包括一个对德语文本产生激活的神经元。我们证明该神经元存在于一个更广泛的上下文 n-gram 电路中……

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (1 messages): 

daniellepintz: 不，没有 `limit`
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1202599419485229096) (151 messages🔥🔥): 

- **Reddit 揭露 VAE 潜在空间的缺陷**：`@gsegato` 强调了一个讨论 KL-F8 VAE 中 KL 散度损失关键缺陷的帖子。据报道，这影响了 SD1.x, SD2.x, SVD, DALL-E 3 等模型，使它们通过极小像素“走私”全局信息，从而降低了效率。
- **VAE 缺陷引发关注**：`@swyxio` 对 VAE 缺陷的讨论做出了回应，指出 Reddit 上的文章简洁明了，相比之下，传统的学术论文往往会包含不必要的冗余内容。
- **LLM 迷因来袭**：`@natolambert` 通过 [Twitter 帖子](https://twitter.com/natolambert/status/1753063313351835941) 传播了关于 LLM 迷因的消息，引发了人们对语言模型轻松一面的关注。
- **AI2 发布 OLMo 模型**：关于 [AI2 发布 OLMo 模型](https://huggingface.co/allenai/OLMo-7B) 的讨论，涉及训练期间使用不同硬件的细节，并提到其 2048 的上下文长度相对于其他模型较短。
- **Nomic AI 发布新 Embeddings**：`@coffeebean6887` 强调了 Nomic AI 发布的开源 Embeddings 及其在 LoCo 基准测试中的卓越表现；详细的博客文章和数据集已可供进一步探索。

**提到的链接**：

- [来自 anton (@abacaj) 的推文](https://x.com/abacaj/status/1752788052500512869?s=46&t=90xQ8sGy63D2OtiaoGJuww)：这是一个将 gpt-4 用作“奖励”模型的代码片段。对我来说效果*非常*好（比使用评分系统更好）。
- [亚马逊发布 Rufus，全新的生成式 AI 驱动对话式购物体验](https://www.aboutamazon.com/news/retail/amazon-rufus)：通过 Rufus，客户现在可以与一位精通亚马逊选品的生成式 AI 专家一起购物，它能整合来自网络各处的信息……
- [见证 Arc 浏览器第二幕 | 为你浏览的浏览器](https://youtu.be/WIeJF3kL5ng?si=itoKPOlDUAIsuV2c)：在 2 月 1 日星期四美国东部时间中午 12:30，我们分享了对 Arc 旅程第二幕的愿景——一种全新的软件类别，一个为你浏览的浏览器……
- [开源语言模型 (OLMos) 与 LLM 格局](https://www.interconnects.ai/p/olmo)：大变革开端的一个小模型。
- [我们为何创立 Parcha](https://www.hitchhikersguidetoai.com/p/why-we-founded-parcha)：深入探讨我们为何在 Parcha 构建 AI Agent，以助力金融科技领域的合规与运营团队。

- [未找到标题](https://news.ycombinator.com/item?id=36855516): 未找到描述
- [OLMo Suite - 一个 allenai 集合](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778): 未找到描述
- [GitHub - uptrain-ai/uptrain: 你的开源 LLM 评估工具包。获取事实准确性、上下文检索质量、语气等评分，以了解你的 LLM 应用质量](https://github.com/uptrain-ai/uptrain): 你的开源 LLM 评估工具包。获取事实准确性、上下文检索质量、语气等评分，以了解你的 LLM 应用质量 - GitHub - uptrain-ai...
- [来自 anton (@abacaj) 的推文](https://x.com/abacaj/status/1752814377068023988?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q): LazyGPT 回来了 ↘️ 引用 Shannon Sands (@max_paperclips) 显然，至少根据这个来看，它实际上更糟了 https://aider.chat/docs/benchmarks-0125.html 太神奇了。我看不出...
- [Reddit - 深入探索一切](https://www.reddit.com/r/StableDiffusion/s/p2BkzI9ypO): 未找到描述
- [来自 swyx (@swyx) 的推文](https://x.com/swyx/status/1753279194065428839?s=46&t=90xQ8sGy63D2OtiaoGJuww): @simon_mo_ @RickLamers @zhuohan123 @woosuk_k @amanrsanger 我想他们正在做我们在 NeurIPS 讨论过的那件事
- [allenai/OLMo-7B · Hugging Face](https://huggingface.co/allenai/OLMo-7B): 未找到描述
- [NAIRR Pilot - 首页](https://nairrpilot.org/): 未找到描述
- [[公开] vLLM 项目更新 @ 第二次 vLLM 见面会](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/mobilepresent?slide=id.g2650ce3df47_0_470)): 项目更新 2024年1月31日，第二次 vLLM 见面会 @ IBM 1
- [R-Judge: 为 LLM Agents 的安全风险意识建立基准](https://arxiv.org/html/2401.10019v1): 未找到描述
- [如何发布你真实的 AI](https://matt.sh/ai-how-to-announce): 未找到描述
- [未找到标题](https://dev.to/builderio/dont-build-ai-products-the-way-everyone-else-is-doing-it-9a7): 未找到描述
- [高效 AI 商业项目的 7 个习惯](https://towardsdatascience.com/7-habits-of-highly-effective-ai-business-projects-6ced590e6db8?gi=e4b47a172d38): 优秀与卓越的 AI 商业项目之间有什么区别？这里有在组织中开展 AI 工作时需要考虑的 7 件事。
- [如何说服风险投资家你是人工智能领域的专家](https://medium.com/machine-learning-in-practice/how-to-convince-venture-capitalists-youre-an-expert-in-artificial-intelligence-39d5edaca290): 如果你喜欢这篇文章，请查看 Robbie 的另一篇：风险投资家说“不”的 15 种方式
- [在 2023 年启动你的新 AI 初创公司 — 打造更好的团队](https://buildingbetterteams.de/profiles/brian-graham/navigating-ai-businesses): 在过去的几个月里，越来越多的人向我询问对他们 AI 商业想法的看法，并寻求在该领域导航的帮助。这篇文章涵盖了我对该领域的大部分想法...
- [如何使用 AI 做实用的事情：一份新指南](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-practical-stuff): 人们经常问我如何使用 AI。这里是一个包含大量链接的概览。
- [关于 AI，每个人都弄错的 3 件事](https://www.washingtonpost.com/technology/2023/03/22/ai-red-flags-misinformation/): 随着 AI 工具的普及，人们正努力将事实与虚构区分开来。
- [如何谈论 AI（即使你对 AI 了解不多）](https://www.technologyreview.com/2023/05/30/1073680/how-to-talk-about-ai-even-if-you-dont-know-much-about-ai/): 另外：在 AI 时代捕捉不良内容。
- [什么是 AI Agents？](https://serokell.io/blog/what-are-ai-agents): 在这篇文章中，你将了解什么是 AI Agents 以及它们真正的能力。你还将学习如何构建适合你目标的 AI Agent。
- [AI Agent 基础：让我们逐步思考](https://www.jonstokes.com/p/ai-agent-basics-lets-think-step-by): 介绍 AgentGPT、BabyAGI、LangChain 以及由 LLM 驱动的 Agent 革命背后的概念。
- [向商务人士推销人工智能](https://towardsdatascience.com/pitching-artificial-intelligence-to-business-people-f8ddd8fb2da2): 从银弹综合症到一线希望
- [公关人员应该（不应该）如何推销 AI 项目](https://thenextweb.com/news/how-pr-people-should-not-pitch-ai-projects-syndication): 对于人工智能社区来说，这是激动人心的时刻。对该领域的兴趣正以加速的步伐增长，学术和专业机器学习课程的注册人数正在飙升...
- [向客户普及机器学习和 AI 知识 – Andy McMahon](https://electricweegie.com/articles/educating-clients/): 未找到描述
- [People + AI 指南](https://pair.withgoogle.com/guidebook/patterns/how-do-i-onboard-users-to-new-ai-features): 为构建以人为本的 AI 产品的团队提供的工具包。

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1202682021730586655) (3 messages): 

- **新的 Latent Space Podcast 剧集发布提醒**：`@swyxio` 在 Twitter 和 Hacker News (HN) 上宣布发布了最新的 [Latent Space Podcast](https://twitter.com/latentspacepod/status/1753120715254198425)。
- **标题党实验受挫**：`@swyxio` 提到，尽管在播客推广中使用了标题党式的标题和图片，但反响却令人失望。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1202568921253937222) (50 messages🔥): 

- **对 AI 伦理和开发的困惑**：`@yami1010` 表达了对 AI 在各个领域角色的困惑，并质疑在 AI 学习中究竟什么才构成不道德。他们表达了希望获取解释 LLM 内部运行机制的资源，并强调了 AI 开发职责中模糊的界限。
  
- **用于客服查询的 Chatbot 集成咨询**：`@qiqimon` 询问了在学校环境中集成基于 GPT 的 Chatbot 来处理客服查询（如入学信息等任务）的难度。

- **AI 开发归属权的误导**：`@movoza` 对 Bing 声称其开发了某个 AI 表示不满，直到在尖锐的追问下才承认 OpenAI 的角色。`@yami1010` 从 AI 开发的集体性质出发，引入了历史和各种利益相关者的更广泛视角。

- **理解 ChatGPT 感知上的性能下降**：`@voidrunner42` 观察到 ChatGPT 似乎变笨了，而 `@tadase.` 则认为问题可能出在系统的 Prompt 方式上，而非整体质量的下降。

- **审核标准和图像指南的辩论**：`@jeremy.o` 和 `@Ted` 之间关于服务器内容审核的对话，特别是围绕图像允许内容和 G-rating 指南的解读，凸显了内容多样性与严格审核之间的紧张关系。
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1202600416840392745) (92 messages🔥🔥): 

- **学校 Chatbot 集成咨询**：`@qiqimon` 询问了为学校客服实现基于 GPT 的 Chatbot 以管理入学咨询的复杂性，`@lugui` 等人建议这相当简单，但需要采取措施防止恶意捣乱。
- **GPT 和知识库问题**：`@united_beagle_74104` 对 ChatGPT 的错误表示沮丧，并寻求客户支持联系方式。同时，用户讨论了维护知识库的最佳文件格式，根据 `@darthgustav.` 等人的建议，由于编码问题，强烈建议避免使用 XLXS 和 DOCX。
- **在 AI 产品推荐中使用 RSV 而非 JSON**：`@quengelbert` 和 `@darthgustav.` 就 AI 产品推荐助手的最佳知识库格式进行了广泛交流，建议使用 RSV (row separated values) 而非 JSON 或 XLXS，以提高 GPT 的性能。
- **自定义 GPT Action 的 API 身份验证困境**：`@woodenrobot` 在将 GPT 项目从内部 Alpha 测试转移到公开 Beta 测试时，在 Bearer Token 身份验证方面遇到了困难，特别是在处理 Weaviate 数据库时。
- **GPT 在处理知识文件和 RAG 时的异常行为**：用户 `@loschess`、`@blckreaper` 等人报告了在最近更新后，GPT 在访问知识文件和使用 API 方面存在不一致性，并分享了诸如减小文件大小以避免检索工具依赖等变通方法。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1202562854621020261) (79 messages🔥🔥): 

- **对影响模型的变更表示担忧**：`@dreamgen` 对涉及 `if any(m in name for m in ["norm", "gate"]):` 这一行的代码变更表示担忧，认为这会影响 Llama 和 Mistral 等许多模型。`@nafnlaus00` 在 `src/axolotl/utils/models.py` 中发现了这个有问题的变更，随后展开了关于其对模型训练和内存使用潜在影响的讨论。
- **针对模型问题的提议修复**：`@nafnlaus00` 提出了一个修复方案，在检查名称中的 "gate" 时增加 `model_config.model_type == "mixtral"` 的判断，对此 `@nanobitz` 询问是否指的是 "mixtral" 而非 "mistral"。
- **GitHub 贡献与模型故障排除**：`@nafnlaus00` 在使用 Git 命令时遇到困难，误删了重要文件，并表达了对 GitHub 的不满；而 `@nanobitz` 提出协助创建 Pull Request，并向 `@nafnlaus00` 索要 GitHub 账号参考。
- **关于文本嵌入和向量数据库的讨论**：用户参与了关于文本嵌入（Text Embedding）模型和向量数据库的技术讨论，`@nanobitz` 特别提到了 `nomic-embed-text-v1`，`@dangfutures` 建议从 `bge` 开始。对话还涉及了在云服务中寻找和使用 GPU 的话题。
- **7B 模型性能与利用率的更新**：有一段关于 7B 模型改进停滞的对话，并提到了 `CapybaraHermes-2.5-Mistral-7B` 和 `Eagle 7B` 等新模型。`@dangfutures` 和 `@c.gato` 分享了近期 7B 模型的链接及其结果，突出了该领域的进展和竞争。

**提到的链接**：

- [RWKV/v5-Eagle-7B · Hugging Face](https://huggingface.co/RWKV/v5-Eagle-7B)：未找到描述
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述
- [argilla/CapybaraHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B)：未找到描述
- [GitHub - qdrant/fastembed: Fast, Accurate, Lightweight Python library to make State of the Art Embedding](https://github.com/qdrant/fastembed)：快速、准确、轻量级的 Python 库，用于构建先进的 Embedding - GitHub - qdrant/fastembed
- [nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1)：未找到描述
- [🌠 Improving RAG by Optimizing Retrieval and Reranking Models](https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.html#Bi-Encoder-Model)：在本教程中，我们将展示如何通过优化检索和重排序模型来改进 RAG（检索增强生成）模型。为此，我们将使用 ArgillaTrainer 来微调...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (1 messages): 

caseus_: 此问题的修复已在 transformers 上游合并。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1202547054325538836) (10 messages🔥): 

- **寻求 Mixtral 的微调建议**：`@emperor` 询问了全量微调 Mixtral 的最佳方式，考虑是否应该微调除路由器（routers）和门控（gates）之外的所有参数。
  
- **GPU VRAM 对 Mixtral 微调至关重要**：`@caseus_` 建议根据可用的 GPU VRAM，使用 **ds zero3**、**8bit optimizer** 以及冻结前三分之一到一半的层可能是微调 Mixtral 的有益起点。 

- **使用 Zero3 进行全模型微调**：`@caseus_` 还提到，通过使用 **Zero3** 的卸载（offloading）功能，或许能够对整个 Mixtral 模型进行微调。

- **寻找超参数见解**：`@emperor` 寻求确认 Mixtral 是否有任何表现良好的超参数（hparams），以及它的行为是否像 **Mistral7b**（它不像 Llama 模型那样青睐高学习率）。

- **分词问题及潜在解决方案**：`@arcontex` 就一个未指明的问题寻求帮助，对此 `@nanobitz` 建议尝试将分词器（tokenizer）切换为 `AutoTokenizer`。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1202590835104026664) (17 messages🔥): 

- **RunPod 概况**：`@c.gato` 经历了 pod 的突然关闭和数据丢失，并提到他们收到一封邮件称 pod 在两天内不会被移除，但发现它在短短 45 分钟内就消失了。
- **提供了 RunPod Discord 链接**：针对 `@c.gato` 的问题，`@caseus_` 建议在官方 RunPod Discord 上反映该问题，并提供了链接：[RunPod Discord 帮助频道](https://discord.com/channels/912829806415085598/1187492973148115076)。
- **Pod 删除时间可能存在 Bug**：`@caseus_` 指出 `@c.gato` 的经历可能指向一个 Bug，并区分了 pod 被停止（stopped）和被删除（deleted）的情况。
- **Hugging Face 连接问题**：`@dreamgen` 报告了在尝试将数据从 RunPod 传输到 Hugging Face 时出现 SSL 错误，质疑该服务的可靠性。
- **下载速度慢的问题**：`@dreamgen` 抱怨在 RunPod 上下载 Torch 包耗时过长，对服务的效率表示沮丧。

**提到的链接**：

[Discord - 与好友和社区聊天的新方式](https://discord.com/channels/912829806415085598/1187492973148115076)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的好友和社区保持紧密联系。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (1 messages): 

dangfutures: 你们搞清楚 Mistral 的配置了吗？
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1202548368539123782) (66 messages🔥🔥): 

- **分享有趣 AI 回复的指南**：`@tbyfly` 寻求关于在哪里分享 Perplexity AI 有趣回复的指导。`@me.lk` 引导他们在另一个频道分享，`@tbyfly` 确认已分享。
  
- **订阅支持查询**：`@dame.outlaw` 询问关于移动端应用订阅问题的帮助，`@icelavaman` 通过询问其账号详情和登录方式进行了回复。

- **Perplexity AI 模型详解**：`@icelavaman` 提供了一个博客文章链接，解释了专注于提供最新且符合事实回复的新 PPLX 模型。`@clay_ferguson` 寻求有关模型使用的信息，`@icelavaman` 再次推荐了那篇详细的博客文章。

- **将 Perplexity 设置为默认搜索引擎**：`@bartleby0` 分享了一个使用提供的模板链接将 Perplexity AI 设置为浏览器默认搜索引擎的解决方案。

- **应用内问题及竞争对手提及**：用户讨论了应用内的问题，如文本选择 Bug 和默认搜索引擎设置。另外，`@dpshade22` 提到 Arc Search 可能是 Perplexity AI 的潜在竞争对手或使用者。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1202251275698450443/1202251275698450443)：Discord 是通过语音、视频和文字进行交流的最简单方式。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047204950763122820/1180293637087698954)：Discord 是通过语音、视频和文字进行交流的最简单方式。
- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843)：Discord 是通过语音、视频和文字进行交流的最简单方式。
- [支持的模型](https://docs.perplexity.ai/docs/model-cards)：未找到描述
- [介绍 PPLX 在线 LLM](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)：首创的在线 LLM API
- [Copilot 使用什么模型？](https://blog.perplexity.ai/technical-faq/what-models-does-copilot-use)：通过我们全面的常见问题解答页面深入了解 Perplexity 的技术细节。从 GPT-4 和 Claude 2 等 AI 模型的细微差别到 Token 限制和 AI 配置文件，获取简明答案以优化你的体验。
- [Perplexity 博客](https://blog.perplexity.ai/technical-faq)：浏览 Perplexity 博客以获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。

  

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1202548983457644584) (4 条消息): 

- **神秘的沉思**：用户 `@tbyfly` 分享了一个思考的表情符号，但由于缺乏额外的上下文或讨论，该消息背后的意图仍然是个谜。
- **对内容的认可**：`@twelsh37` 似乎对某项未指明的内容感到满意，将其描述为“**非常值得一看**”，尽管没有提供内容的具体细节。
- **通过博客拥抱 Perplexity**：新社区成员 `@.sayanara` 分享了他们发现 Perplexity AI 的热情，并链接了一篇赞扬 AI 在打击虚假信息方面潜力的博客文章。他们主张负责任地使用 AI，引导人们走向清晰和事实，正如他们在[博客文章](https://figmentums.com/2024/02/01/the-right-direction-for-ai/)和书籍《*[Pandemic of Delusion](https://figmentums.com/2023/02/23/pandemic-of-delusion/)*》中所讨论的那样。
- **令人惊讶的热门应用趋势**：`@bartleby0` 评论了 Facebook 在热门应用列表中的显著缺席，称其“很有趣”，但未提供链接或进一步阐述该话题。

**提到的链接**：

[The Right Direction for AI](https://figmentums.com/2024/02/01/the-right-direction-for-ai/)：在这篇博客和我的书《Pandemic of Delusion》中，我重点关注了 AI，特别是它在塑造我们思维方面（无论好坏）的巨大潜力。虽然 AI 代表了一个可怕的……

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1202546069783707700) (8 条消息🔥): 

- **7b 模型局限性暴露**：`@brknclock1215` 指出 **7b-online 模型**在处理复杂查询时往往会失败，并且经常依赖其训练数据，导致输出或答案不准确。
- **调整 Codellama 70b 的响应行为**：`@general3d` 提到 **codellama 70b** 非常严格，并建议 **Perplexity AI 团队**可以通过在响应开头添加“Sure!”或类似的 Prompt 来调整它，以提供更一致的答案。
- **AI Agents 作为替代方案**：`@tbyfly` 提议使用 **AI Agents** 反复运行 Prompt，直到获得满意的答案，作为解决 codellama 70b 问题的一个潜在方案。
- **隐私担忧阻碍 SQL 协助**：`@jayb1791` 抱怨 **codellama 70b** 以数据隐私为由拒绝协助处理一个 SQL 问题，尽管该问题仅仅是在 SQL 查询中引用了一个商业地址。
- **API 额度与订阅服务问题**：`@alankarsh` 在支付 Pro 订阅费用后遇到了 API 额度未显示的问题，尽管尝试了三张不同的卡并联系了客户体验团队寻求帮助。
  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1202665033599815761) (3 条消息): 

- **开源 Embedding 媲美 OpenAI 最新产品**：LlamaIndex 介绍了 `@nomic_ai` 的 *nomic-embed-text-1*，声称其性能优于 OpenAI 的 text-embedding-3-small。该 Embedding 是开源的，拥有开放数据，并已完全集成到 LlamaIndex 中；更多详情请见 [Twitter](https://twitter.com/llama_index/status/1753106179008696521)。

- **错过了 @aiusergroup 的主题演讲？观看回放**：参会者现在可以通过 [YouTube](https://t.co/hrUMF8bq8Q) 观看 @jerryjliu0 关于 *Beyond Naive Rag: Adding Agentic Layers* 的主题演讲回放，并可通过[此链接](https://t.co/P39riIMGK6)获取幻灯片，该内容由 [LlamaIndex Twitter](https://twitter.com/llama_index/status/1753206969161429123) 分享。

- **关于利用数据科学增强 RAG 的博客文章**：@sudalairajkumar 探讨了将经典数据科学与检索方法相结合以改进检索增强生成 (RAG) 的方法，并在 LlamaIndex 推荐的一篇详细[博客文章](https://t.co/zTPyeaqIbU)中提供了关于数据集评估和 Embedding 模型选择的指导。

**提到的链接**：

- [无标题](https://t.co/hrUMF8bq8Q)：未找到描述
- [LlamaIndex 演讲 (AI User Conference)](https://t.co/P39riIMGK6)：Beyond Naive RAG: Adding Agentic Layers，Jerry Liu，LlamaIndex 联合创始人/CEO

  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1202537178282991647) (40 messages🔥): 

- **LlamaIndex 不仅仅局限于 OpenAI**：`@whitefang_jr` 确认 LlamaIndex 可以轻松地与其他 LLM 配合使用，并指出了 [集成指南](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules) 以及一个将 Llama-2 与 LlamaIndex 结合使用的 [示例 Notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb)。
- **本地 LLAMA Packs 的集成指令**：`@whitefang_jr` 提供了用于调整 Llama-2 与 resume_screener pack 集成的源代码，并 [链接到了 GitHub 仓库](https://github.com/run-llama/llama-hub/blob/e6960d4c64f053d32f1e92aa3a587f7b045cbfff/llama_hub/llama_packs/resume_screener/base.py#L58)。
- **TypeScript/JS 项目中的 Postgres 难题**：用户 `@.Jayson` 注意到了 Postgres 连接的问题，`@whitefang_jr` 建议自定义 PG vector 类以处理多个连接；`@cheesyfishes` 还提到可以考虑使用连接池（connection pooling）。
- **垃圾信息警报快速响应**：`@cheesyfishes` 针对 `@mysterious_avocado_98353` 提到的垃圾信息警报采取了行动，通过封禁账号并删除有害内容处理了该情况。
- **使用 MongoDB 配置 Ingestion Pipeline**：`@ramihassanein` 就设置 MongoDB Atlas 的 Ingestion Pipeline 时遇到的错误寻求帮助，`@cheesyfishes` 回复了一个解决方案，即更新 MongoDB vector store 以使用更新后的基类。

**提到的链接**：

- [未找到标题](https://llamahub.ai/l/llama_packs-resume_screener?from=all)：未找到描述
- [run-llama/llama-hub 中的 llama-hub/llama_hub/llama_packs/resume_screener/base.py](https://github.com/run-llama/llama-hub/blob/e6960d4c64f053d32f1e92aa3a587f7b045cbfff/llama_hub/llama_packs/resume_screener/base.py#L58)：由社区制作的用于 LLM 的数据加载器库 —— 与 LlamaIndex 和/或 LangChain 配合使用 - run-llama/llama-hub
- [查询转换 - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html)：未找到描述
- [使用 LLM - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules)：未找到描述
- [LlamaCPP - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html)：未找到描述

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1202588194822950942) (3 messages): 

- **寻求 PII 匿名化解决方案**：`@princekumar13` 正在寻找在将文本数据发送到大语言模型 (LLM) 之前，对个人身份信息 (PII) 进行匿名化处理的方法，同时保持数据的准确性。他们发现了一种使用 langchain 和 Presidio 的方法，但它是实验性的，不适合生产环境。
  
- **对用于调度的专用 LLM 的兴趣**：`@erizvi` 询问了能够解释特定于配置作业调度的自然语言输入，并将其转换为 cron job 表达式的最小 LLM。

- **Whisper 语音转文本：理解语音转文本模型**：`@amgadoz` 分享了一篇 [博客文章](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b)，详细介绍了 OpenAI 的 Whisper 模型（一种先进的语音转文本 (STT) 系统），包括对其架构和功能的见解。该文章是一个系列的第二部分，重点介绍了 Whisper 使用的基于 "Attention is All You Need" 论文的 encoder-decoder transformer。

**提到的链接**：

[Whisper: 如何创建强大的 ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2)：多部分系列文章的第 2 部分，我们将深入探讨 OpenAI 最先进的自动语音识别模型 Whisper。

  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1202524630154223626) (32 条消息🔥): 

- **寻求 AWS Sagemaker 和 Chatbot 构建资源**：`@refik0727` 询问了关于使用 **AWS SageMaker** 构建 Chatbot 的教程或 GitHub 仓库，涉及 **Mistral** 和 **Llama** 等模型，并具备集成个人 CSV 文件或连接本地数据库的能力。

- **LangChain Twitter 潜在安全漏洞**：包括 `@markopolojarvi`、`@alcazarr` 和 `@solac3` 在内的多位用户提醒社区，**LangChain AI Twitter** 账号可能被盗并涉及诈骗，他们分享了相同的可疑 [Twitter 链接](https://twitter.com/LangChainAI/status/1753014882696405254)。

- **关于 ChatOpenAI 和 Streaming 的咨询**：用户 `@hiranga.g` 询问社区 `get_openai_callback()` 是否与用于 **ChatOpenAI** 流式传输（Streaming）的 **Agent** 兼容。

- **讨论为 OpenGTPs 创建专门频道的提议**：`@benjaminbascary` 提出了在 Discord 中为 **OpenGTPs** 讨论创建一个独立频道的想法。

- **探索使用 Llamacpp 和 Langserve/FastAPI 进行流式传输**：`@legendary_pony_33278` 寻求社区帮助，以实现在 **Llamacpp** 和 **Langserve** 或 **FastAPI** 中开启流式传输（Streaming）。`@veryboldbagel` 回复并分享了一个 [使用 Ollama 的 GitHub 示例](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py)，但该示例可能并未专门处理流式传输。

- **针对“只教不给答案”的 AI 的 Guardrails**：针对 `@_emiya` 寻求构建一个能教学生数学题而不直接给出答案的 AI，`@deetisaac` 建议研究 **Guardrails**，并分享了其 [GitHub 仓库](https://github.com/guardrails-ai/guardrails)。

- **增强 RAG 应用的深度学习资源**：在 `@daii3696` 询问改进 RAG 应用的资源后，`@johnny2x2` 建议阅读 **llama index 源代码**，并宣布他们将在下周二主持一场关于 RAG 技术的演讲。

- **调整 ChatOpenAI Agent 的 Max Tokens 限制**：`@irfansyah5572` 寻求关于为遇到 `InvalidRequestError` 的 Agent 设置 max tokens 限制的建议，`@the_agent_j` 通过引用 [LangChain OpenAI API](https://api.js.langchain.com/classes/langchain_openai.ChatOpenAI.html#maxTokens) 中的 `maxTokens` 参数解决了该问题。

- **关于禁用指数退避（Exponential Backoff）的问题**：用户 `@apollo7701` 询问社区如何禁用指数退避，但未提供进一步的背景或后续回复。

**提到的链接**：

- [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents): 未找到描述
- [ChatOpenAI | LangChain.js - v0.1.12](https://api.js.langchain.com/classes/langchain_openai.ChatOpenAI.html#maxTokens): 未找到描述
- [langserve/examples/local_llm/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。
- [GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails): 为大语言模型添加 guardrails。通过在 GitHub 上创建账号来为 guardrails-ai/guardrails 的开发做出贡献。

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1202819089563910244) (2 条消息): 

- **使用 Ollama 的 LangServe 本地 LLM 示例**：`@veryboldbagel` 分享了一个 [GitHub 链接](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py)，展示了如何将 **LangServe 与 ollama** 结合使用，并提醒用户注意并发使用的限制。
- **LangSmith 等候名单将缩减**：`@veryboldbagel` 宣布在未来几周内，将有更多人从 **LangSmith** 的等候名单中移除（获得访问权限）。

**提到的链接**：

[langserve/examples/local_llm/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。

  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1202638561854234646) (9 messages🔥): 

- **自主 GPT-4 Agent 平台 IX**：`@robot3yes` 介绍了 **Agent IX**，这是一个在 GitHub 上的自主 GPT-4 Agent 平台，鼓励大家前往查看。访问他们的作品：[GitHub - kreneskyp/ix](https://github.com/kreneskyp/ix)。
- **使用 ContextCrunch 最大化节省 Token**：`@speuce` 分享了 **ContextCrunch**，这是一个 Prompt 压缩 API，旨在帮助节省 Token 成本并与 LangChain 集成。欢迎在 [contextcrunch.com](https://contextcrunch.com/) 获取早期访问权限并提供反馈。
- **Skipp 招聘 OpenAI Full-Stack 开发人员**：`@marcelaresch_11706` 发布了 Skipp 的一个职位空缺，招聘一名专注于后端且深入了解 OpenAI API 的资深 Full-Stack 开发人员。该职位为太平洋时区的全职工作，正在寻找来自拉美（LATAM）和欧洲的候选人。
- **深入探讨 OpenAI 的 Whisper 模型**：`@amgadoz` 撰写了一篇关于 OpenAI 用于 Speech-to-Text 的 Whisper 模型的详细博客文章，涵盖了模型的架构和功能。有关该模型的见解可在 [Substack](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b) 上免费阅读。
- **Step-Back Prompting 技术探索**：`@andysingal` 在 Medium 上发表了一篇关于将 Step-Back Prompting 与 LangChain 结合的文章，讨论了语言处理能力的提升。在 [Medium](https://medium.com/ai-advances/langchain-elevates-with-step-back-prompting-using-ragatouille-b433e6f200ea) 阅读全文。

**提到的链接**：

- [Whisper: How to Create Robust ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2)：系列文章的第 2 部分，深入探讨 OpenAI 最先进的自动语音识别模型 Whisper。
- [GitHub - kreneskyp/ix: Autonomous GPT-4 agent platform](https://github.com/kreneskyp/ix)：自主 GPT-4 Agent 平台。可以通过在 GitHub 上创建账号来为 kreneskyp/ix 的开发做出贡献。
- [ContextCrunch](https://contextcrunch.com/)：未找到描述。
- [Langchain Elevates with Step-Back Prompting using RAGatouille](https://medium.com/ai-advances/langchain-elevates-with-step-back-prompting-using-ragatouille-b433e6f200ea)：一场语言革命。

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1202586642423091250) (2 messages): 

- **ChatOpenAI 的 Token 计数问题**：用户 `@hiranga.g` 询问在流式传输（streaming）期间如何获取 ChatOpenAI Agent 的 **Token 计数**，并提到 `get_openai_callback()` 似乎无法与其配合使用。
- **使用 LangChain 压缩 LLM 上下文**：用户 `@speuce` 分享了一篇 [Medium 文章](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b)，概述了在 **LangChain** 中进行 **Contextual Compression** 的过程，利用 [ContextCrunch](https://contextcrunch.com) 进行高效压缩，这是解决检索增强生成（**RAG**）设置中高 Token 使用量的一种方案。

**提到的链接**：

[How to Compress LLM Contexts with LangChain](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b)：在本教程中，你将学习如何使用 LangChain 将 Token 使用量减少高达 90%。

  

---



### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1202711546078171256) (1 messages): 

- **计划从 7b 迁移到 Mixtral**：`@_jp1_` 提到由于 7b 模型在某些维度上的局限性，决定 **从 7b 模型切换到 Mixtral 基础模型**，并指出 Mixtral 在评估任务中提供了更好的权衡。他们也欢迎在这一过渡过程中提供帮助。

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1202525729183629332) (15 messages🔥): 

- **API 安全遗漏警报**：`@sebastian.bodza` 强调了一个潜在的安全隐患，指出 **API 未受保护**，请求中未使用任何 Token。
- **新的 Embedding 模型即将问世**：`@bjoernp` 展示了 **Nomic 的长上下文文本嵌入器** `nomic-embed-text-v1`，该模型拥有 8192 的序列长度，性能优于 OpenAI 模型，并提供了开源权重、训练代码和数据 ([Hugging Face 上的 Nomic AI](https://huggingface.co/nomic-ai/nomic-embed-text-v1))。
- **Allen AI 发布 OLMo 7B**：`@_jp1_` 介绍了 **OLMo 7B** 的发布，这是 Allen AI 开发的一个开源语言模型（Open Language Model），同时发布的还有其数据集、训练代码和论文 ([Allen AI 上的 OLMo](https://allenai.org/olmo/olmo-paper.pdf), [Hugging Face 上的 OLMo](https://huggingface.co/allenai/OLMo-7B))。
- **寻找最佳德语 Embedding 模型**：针对 `@ustoll` 询问用于问答检索（Q/A retrieval）的最佳德语文本嵌入模型，`@philipmay` 强调，使用场景（如聚类或语义相似度）是决定合适模型的关键。
- **使用 RAG 和 Axolotl 进行微调**：`@rasdani` 描述了将 RAG 数据集转换为 ShareGPT 格式以供 **Axolotl** 使用的过程，并提到原始 SFT 过程中缺少 `positive_ctx_idx`，随后为了潜在的（但尚未实施的）DPO 添加了该字段。

**提到的链接**：

- [nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1)：未找到描述
- [allenai/OLMo-7B · Hugging Face](https://huggingface.co/allenai/OLMo-7B)：未找到描述

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1202723426192597082) (2 messages): 

- **推理的 GPU 替代方案**：用户 `@_jp1_` 讨论了在没有专用 GPU 的情况下运行推理任务的选项。他们建议使用 **replicate**、**modal** 或 **serverless runpod** 等服务，并提到如果有需求，可以考虑通过 *together et al.* 托管模型。
- **分享 Colab 资源**：`@_jp1_` 还分享了一个 Google Colab 笔记本链接，但由于消息中缺乏细节，该笔记本的具体内容或背景尚不清楚。[访问 Colab 笔记本](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK)。

**提到的链接**：

[Google Colaboratory](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO)：未找到描述

  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1202659198660513875) (3 messages): 

- **Nomic Embed 性能优于 Ada 和 OpenAI 小型模型**：`@thebaghdaddy` 分享了一个 [HuggingFace 链接](https://huggingface.co/nomic-ai/nomic-embed-text-v1)，介绍了 **nomic-embed-text-v1**。这是一个开源文本编码器，上下文长度为 8192，据称其性能指标优于 OpenAI 的 *text-embedding-ada-002* 和 *text-embedding-3-small*。
- **展示性能指标表**：根据来源，**nomic-embed-text-v1** 在 MTEB 上得分为 62.39，在 LoCo 基准测试中得分为 85.53，显示了其卓越的性能。表格还强调了该模型开源权重、训练代码和数据的可获取性。
- **对开源编码器感兴趣**：`@thebaghdaddy` 表示有兴趣亲自测试新的 **nomic-embed-text-v1** 模型。
- **寻找 Embedding 测试场**：`@thebaghdaddy` 询问是否存在类似于 LLM Arena 排行榜的 "Embeddings Arena"，用于展示具有竞争力的模型。

**提到的链接**：

[nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1)：未找到描述

  

---


### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/) (1 messages): 

firefox8975: 有没有人尝试过或者有关于如何使用 vLLM 将开源模型部署到 AWS 的指南？
  

---

### LLM Perf Enthusiasts AI ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/1202671582510583840) (2 messages): 

- **Exa（前身为 Metaphor）举办托加袍主题发布派对**：`@sarahchieng` 邀请大家参加 **Exa** 的东西海岸发布派对，现场设有 **卫城室（Acropolis room）**、**DJ** 以及“吐真剂”。[旧金山派对](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?%3C%3CFriday,%20Feb%202) 将于 **2 月 2 日** 举行，[纽约派对](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs) 将于 **2 月 7 日** 举行；两场活动均需 RSVP。

- **Exa 在旧金山和纽约请喝咖啡**：除了发布派对，`@sarahchieng` 还提议在 **旧金山** 和 **纽约** 约咖啡，并表示费用“我包了……哈哈”。

**提到的链接**：

- [RSVP 参加 Exa 希腊罗马托加袍发布派对 | Partiful](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?): 使命相同，名字焕新！快来加入 Exa（原 Metaphor）团队，参加官方的 Exa 发布派对！Exa 的使命是组织世界的知识……所以这自然是一个托加袍（TOGA）派对……
- [Exa 东海岸发布派对](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs): 快来加入 Exa（原 Metaphor）团队，参加官方的 Exa 发布派对！现场提供周边、零食和 Exa 额度 :) 发布公告：https://twitter.com/ExaAILabs/status/1750942315055882554 我们是谁：...

  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1202660583871217724) (2 messages): 

- **数据库迁移进行中**：针对 `@.psychickoala` 关于所使用的向量数据库的询问，`@michelcarroll` 透露他们目前正在使用 **Weaviate**，但正在迁移到 **带有 HNSW 的 pgvector**。
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1202839633084289075) (5 messages): 

- **保存并复用思维链（Chain of Thought）**：`@byronhsu` 询问了保存模型的思维链（CoT）并将其作为后续 Prompt 输入的结果。其目的是减少多次迭代的时间和成本，并质疑这种技术是否会影响模型的推理能力。
- **CoT 方法中的延迟权衡**：`@justahvee` 确认将之前保存的 CoT 输入到第二步是可行的，但指出由于多次调用（原始生成和后续推理），推理延迟（inference latency）会存在权衡。
- **澄清推理延迟问题**：`@byronhsu` 向 `@justahvee` 寻求关于推理延迟问题的澄清，认为直接输入 CoT 应该会导致更快的推理。
- **解释延迟权衡**：`@justahvee` 解释说，所经历的延迟是由于初始 CoT 生成和使用 CoT 的第二次推理的总时间，因为它涉及多次独立的调用。
- **准确率保持的保证**：`@byronhsu` 得到 `@justahvee` 的确认，当一个步骤的 CoT 输出被用作下一步的输入时，准确率得以保持。
  

---



### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1202789393975484446) (7 messages): 

- **一次 Warp 速度实验**：`@zippika` 分享了 `rgb_to_grayscale` CUDA kernel 优化的[代码](https://example.com/link-to-code)，利用 **ulong** 进行每个 warp 的向量化加载（vectorized loads）。这种独特的方法一次加载三个像素，旨在提高效率。
- **提高 Occupancy**：随后的修改导致了更高的 **Occupancy**，计算出的理论值为 **100%**，但实际测得的 Occupancy 为 **77%**，表明在 `@zippika` 的实验中 GPU 资源得到了更好的利用。
- **吞吐量显微镜分析**：`@zippika` 发布了 **GPU Throughput** 分析，强调 **Memory** 利用率为 **71.87%**，**Compute (SM)** 使用率为 **46.36%**，反映了其基于 GPU 计算的全面工作负载分布。
- **前方的速度障碍**：尽管概念上取得了成功，`@zippika` 提到尚未测试速度，随后跟进报告称新实现在实践中似乎更 **慢**。
- **带表情符号的出声思考**：用户为看似杂乱无章的消息道歉，表示这些评论是思考过程的一部分，并附带了一个独特的表情符号（`<:floom:799302782477402142>`），显示了非正式交流与技术内容的结合。
  

---

### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1202774545870749736) (3 messages): 

- **CUDA 环境搭建成功**：用户 `@noobpeen` 宣布他们已成功在 **Visual Studio 中配置了 CUDA 12.2**，并询问接下来的步骤。
- **PyTorch 和 C++ 经验**：`@noobpeen` 分享了他们对 PyTorch 的熟悉程度以及对 C++ 的精通，为他们的 CUDA 学习进度提供背景。
- **深入 CUDA 开发的建议**：针对 `@noobpeen` 的提问，`@lancerts` 建议创建一个新的 CUDA 项目并从 CUDA kernels 开发开始，随后推荐阅读《*Professional CUDA C Programming*》一书。
  

---


### CUDA MODE (Mark Saroufim) ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1202535400065728512) (1 messages): 

- **通过线程粗化（Thread Coarsening）优化内存加载**：`@tvi_` 解释说，采用线程粗化是为了**增加 tile size**，并通过每个线程处理更多的输出像素来降低 global memory 的加载频率。这种方法与书中后面讨论的 **"work efficiency"** 概念相关，强调了其在内存操作（而非仅仅是计算任务）中的重要性。
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (1 messages): 

dbreunig: https://www.dbreunig.com/2024/02/01/pursuing-quiet-ai.html
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1202686909344907274) (2 messages): 

- **将 Datasette PDF 提供给 GPT**：`@simonw` 分享了一次经历，他们将 [Datasette 文档](https://docs.datasette.io/en/stable/) 的 **PDF 版本提供给了一个 GPT**，但结果并不太令人满意。
- **充满希望但保持谨慎乐观**：尽管初步结果不尽如人意，`@simonw` 仍然认为，通过大量的努力，将 Datasette 的 PDF 文档喂给 GPT 可能会产生效果。

**提到的链接**：

[Datasette 文档](https://docs.datasette.io/en/stable/)：未找到描述