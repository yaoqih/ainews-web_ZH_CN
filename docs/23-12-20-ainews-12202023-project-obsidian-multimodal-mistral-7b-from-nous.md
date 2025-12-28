---
companies:
- nous-research
- teknim
- openai
date: '2023-12-21T03:20:57.056468Z'
description: '**Project Obsidian** 是一个正在公开训练的多模态模型，由 **Teknium** 在 Nous Discord 上进行追踪。讨论内容包括
  **4M：大规模多模态掩码建模 (Massively Multimodal Masked Modeling)** 以及 **Reason.dev**（一个用于大语言模型应用的
  TypeScript 框架）。


  **OpenAI Discord** 社区讨论了运行 **TensorFlow JS** 进行图像检测的硬件规格、用于过滤不当图像的安全 API 构想，以及对
  AI 中种族和文化偏见的担忧，特别是在面部识别和医疗保健领域。文中指出了 **GPT-3.5** 和 **GPT-4** 在文字解谜游戏中所面临的挑战，并提出了在
  AI 推理中优先考虑显存 (VRAM) 的 GPU 推荐方案。此外，用户还就 **GPT-4** 的视觉能力、**DALL·E 3** 的局限性、平台访问问题以及获得更好输出的提示词策略展开了讨论。'
id: 473b90b4-7a43-4efb-9eed-4859c24f2b11
models:
- gpt-4
- gpt-3.5
- dall-e-3
original_slug: ainews-12202023-project-obsidian-multimodal
people: []
title: 2023年12月20日：Project Obsidian —— 来自 Nous 的多模态 Mistral 7B
topics:
- multimodality
- image-detection
- security-api
- bias
- facial-recognition
- healthcare-ai
- gpu-optimization
- prompt-engineering
- vision
type: archival
---

<!-- buttondown-editor-mode: plaintext -->如果你曾想亲眼目睹一个多模态模型的训练过程，现在正是关注 [Project Obsidian](https://github.com/NousResearch/Obsidian) 的时候：

 
![image.png](https://assets.buttondown.email/images/833fcd59-f3b2-4c06-a5d2-cd91e784be50.png?w=960&fit=max)
 

Teknium 刚刚在 Nous Discord 中向公众开放了一个用于追踪该项目的新频道。

此外，还有关于 [4M: Massively Multimodal Masked Modeling](https://4m.epfl.ch/?utm_source=ainews&utm_medium=email) 以及用于 LLM 应用的 TS 框架 [Reason.dev](https://www.tryreason.dev/blog/introducing-reasonn) 的讨论。

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **TensorFlow JS 与图像检测的 PC 配置**：用户 `@z3wins` 发起了关于运行 TensorFlow JS 硬件要求的讨论，`@marchy` 提供了见解并分享了相关的 [codesandbox](https://codesandbox.io/s/tensorflowjs-object-detection-using-webcam-gvinz?file=/src/App.js) 链接。
- **用于过滤不当图像的安全 API 服务**：`@z3wins` 提出的“安全 API 服务”构想引发了关于数据传输成本以及设备端处理与服务器端处理可行性的讨论。
- **AI 及其应用中的偏见**：`@whynot66k20ni` 就 AI 中固有的种族和文化偏见进行了对话，特别讨论了人脸识别技术及其在医疗保健和心理健康方面的潜在应用。
- **语言模型在单词拼图游戏中的挑战**：用户 `@eskcanta` 和 `@juanitosway` 描述了将 GPT-3.5 和 GPT-4 等语言模型应用于单词拼图游戏时的困难。
- **AI 模型的 GPU 推荐**：针对 `@isntfunny` 关于适合运行 AI 模型的廉价 GPU 的咨询，`@lugui` 提供了 GPU 建议。
- **GPT-4 与 DALL·E 3 功能**：讨论了 GPT-4 的能力和潜在局限性，以及用户对 OpenAI 平台上 DALL·E 3 的限制和输出结果的不满。
- **GPT 模型的通信限制与商业用途**：用户表达了对消息限制额度的沮丧，并讨论了使用 GPT 模型生成的图像的商业使用权。
- **平台访问与使用问题**：许多用户报告了 OpenAI 平台访问和功能方面的问题，特别是关于 Rate Limit（速率限制）问题、异常频繁的人机验证以及自定义 GPT 模型的丢失。
- **Prompting 策略**：用户寻求指导并讨论了有效的 Prompting 策略，重点在于限制 DALL·E 3 的图像生成、确保 ChatGPT 的字面响应，以及为营销研究规划创建合适的 Prompt。此外，还就如何减少 ChatGPT 输出中的礼貌语气提供了建议。


**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (126 条消息🔥🔥): 
        
- **TensorFlow JS 和图像检测的 PC 规格**：用户 `@z3wins` 发起了一场关于运行 TensorFlow JS 进行图像检测所需 PC 配置的对话。`@.marchy` 建议它可以在 Raspberry Pi 上运行，并分享了一个相关的代码沙盒链接。

- **对数据传输速率的担忧**：讨论指出了与实时检测相关的潜在数据传输成本。

- **安全 API 服务的潜力**：`@z3wins` 表达了一个初步想法，即建立一个安全 API 服务来检查图像中的不当内容。`@.marchy` 对此类服务的优势提出了质疑，并建议在设备端执行此操作比在服务器端更高效。

- **关于 AI 中种族和文化偏见的讨论**：`@whynot66k20ni` 与 `@.marchy` 就 AI 中潜在的种族和文化偏见进行了交流。具体而言，他们讨论了人脸识别技术面临的挑战、数据集中代表性不足的群体，以及在医疗保健和心理健康领域的潜在应用。

- **LLM 对单词拼图游戏的影响**：`@eskcanta` 和 `@juanitosway` 讨论了在吊死鬼（hangman）等单词拼图游戏中使用 GPT-3.5 和 GPT-4 等 LLM 的难度，因为这些模型在处理单词中的单个字母时比较吃力。

- **AI 模型的廉价 GPU 推荐**：`@isntfunny` 询问了能够及时运行 AI 推理（Inference）的经济型 GPU。`@lugui` 建议优先选择在可承受范围内 VRAM 最大的 GPU，并强调 VRAM 是 AI 模型的主要需求。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (144 条消息🔥🔥): 
        
- **GPT-4 能力**：关于 **GPT-4** 的功能有多次讨论（用户 `@【ｐｅｎｕｌｔｉｍａｔｅ】`）。有人指出，该 AI 除了文本处理能力外，还具备视觉能力。然而，会议澄清了 GPT-4 并非一个显著的新模型，而是 GPT-3 的扩展，能够处理图像输入。

- **Dall-E 3 使用**：用户们就 **Dall-E 3** 的图像生成能力进行了多次交流（用户 `@errorsource`, `@satanhashtag`, `@lugui`, `@asura0_00`）。一些用户对 OpenAI 平台上 Dall-E 3 的输出效果和限制表示不满，并讨论了潜在的替代方案。

- **GPT 模型的通信限制**：多位用户对每 3 小时的消息限制表示沮丧（用户 `@cherrywavescollide`, `@sirthatsillegal`, `@winchawa`, `@satanhashtag`）。一些用户不确定上限是否准确为每 3 小时 40 条消息，另一些用户则注意到自己提前达到了限制。

- **GPT 商业用途**：用户 `@pickle2108`, `@Furnance` 和 `@solbus` 讨论了使用 GPT 模型生成的图像的商业权利。用户 `@solbus` 提供了 OpenAI 使用条款的链接，并对图像所有权进行了说明。

- **平台问题与投诉**：多位用户（`@sieventer`, `@superiornickson5312`, `@kingkkaktus`）报告了 OpenAI 平台的问题，包括遇到错误、无法访问某些功能或因过度使用而被封锁。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (48 条消息🔥): 
        
- **关于 ChatGPT 的安全担忧**：用户 `@RobertGOD` 对 ChatGPT 潜在的安全漏洞表示担忧，特别指出缺乏双重身份验证可能导致暴力破解攻击。作为回应，`@satanhashtag` 引导其前往 OpenAI 官方支持页面 (help.openai.com) 寻求官方帮助。
- **GPT-4 的 Rate Limit 问题**：`@amanshrestha` 在使用 GPT-4 时遇到了速率限制问题，达到了 10,000 个 tokens 的限制。`@lugui` 建议可以通过向账户充值额度来抵消。分享了 [Rate limits](https://platform.openai.com/account/rate-limits) 链接。
- **用户在 ChatGPT 上遇到异常的人机验证**：多位用户（`@Rock`, `@beret90`, `@cssupport`, `@knowcryptoshow`）讨论了在使用 ChatGPT 时遇到重复的人机验证检查。`@solbus` 建议这可能是由于浏览器的某些隐私设置或扩展程序引起的，并建议在无痕模式下测试和/或使用不同的浏览器。
- **账户访问问题及自定义 GPT 消失**：`@khoughy` 和 `@cdav` 分别遇到了账户访问问题以及他们的自定义 GPT 从仪表板消失的问题。`@elektronisade` 澄清说网站验证与账户访问无关，并暗示禁止“具有高经济损失风险的活动”的使用政策可能是某些自定义 GPT 被移除的原因。
- **DALL-E Seed 重现问题**：`@lxreilly` 询问关于使用 seed 在 DALL-E 中重现之前图像的问题，`@toror` 建议联系 [support@openai.com](mailto:support@openai.com)。`@solbus` 指出 OpenAI 的支持工单应通过 help.openai.com 提交。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (1 条消息): 
        
openheroes: 哦，我不知道……你是说它不能让你下载文件吗？

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (21 messages🔥): 
        
- **Prompting 资源推荐**：`@akas7488` 询问了有关提升文本和图像 Prompting 技巧的资源、课程或 YouTube 频道的建议，但在给定的消息中没有记录任何回复或建议。
- **DALL·E 3 图像生成**：`@suptrox` 正在寻找一种方法，让 DALL·E 3 根据特定 Prompt 生成元素数量更有限的图像。`@alienpotus` 建议使用更聚焦的 Prompt，明确说明图像中不应包含的任何元素。
- **ChatGPT 对 Prompt 的过度解读**：`@agowa338` 对 ChatGPT 过度解读 Prompt 而不回答实际提出的问题表示担忧，特别是在后续问题中。讨论中未给出解决方案或建议。
- **为市场调研规划创建 Prompt**：`@raunchotron` 询问了关于为市场调研计划创建 Prompt 的建议。`@eskcanta` 提供了 Prompt Engineering 的通用指南，强调精确沟通需求以避免混淆或不想要的响应。
- **减少 ChatGPT 对话中的礼貌语气**：`@stealth2077` 询问是否有办法进行 Negative Prompting，以阻止 AI 在每个故事中强加尊重、伦理或道德话题。`@eskcanta` 表示 Negative Prompting 通常无效，通过在其编程和限制范围内仔细引导 AI 产生所需输出可以获得更好的效果。`@eskcanta` 还提供了一个详细示例，说明如何有效地引导 AI 在故事场景中进行深入且引人入胜的角色开发。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (21 messages🔥): 
        
- **Prompting 资源**：`@akas7488` 询问了提升文本和图像 Prompting 技能的资源。后续消息中没有分享任何资源。
- **DALL·E 3 图像生成**：`@suptrox` 询问是否有办法限制 DALL·E 3 生成的元素，并举了一个例子，即希望专注于花园的土地而没有其他干扰。作为回应，`@alienpotus` 建议了一种修改 Prompt 的方法来实现这一点，强调了 Prompt 中特异性（Specificity）和排他性（Exclusivity）的重要性。
- **ChatGPT 的字面响应**：`@agowa338` 对 ChatGPT 过度解读 Prompt 而不是按字面意思回答表示担忧，并寻求让 ChatGPT 响应更具字面意义的建议。后续消息中未提供直接解决方案。
- **为市场调研计划创建 Prompt**：`@raunchotron` 询问了关于为市场调研计划创建 Prompt 的事宜。`@eskcanta` 建议让 AI 清楚地理解需求，并尽可能准确地使用语言，仔细检查输出，对 AI 的回答进行事实核查，并避开 AI 已知的 Hallucinate（幻觉）领域。
- **减少生成文本中的礼貌性**：`@stealth2077` 询问如何减少每个故事中尊重、伦理或道德话题的数量。`@eskcanta` 做出回应，解释了 OpenAI 的内容政策、模型树立良好榜样的固有性质，并提供了一个关于如何构建 Prompt 以引导 AI 生成存在分歧的故事的综合示例。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- `@fullstack6209` 建议在 LLM 图查询（graph queries）的上下文中不要使用 **Langchain**。`@maxwellandrews` 和 `@pogpunk` 与一名身份不明的用户进行了交流，建议利用拓扑结构（topologies）来创建 LLM 图查询系统，并询问了他们的 Twitter 账号。

- 在 **benchmarks-log** 频道中，`@gabriel_syme` 对更广泛的评估和添加自定义任务的想法表示欢迎，并指出其效率取决于添加的简便程度。

- **interesting-links** 频道进行了广泛讨论，主要集中在提高模型性能和数据使用策略上。关键主题包括 **Instruction Finetuning (IFT)**、**pretraining data**、特定数据集（如 **GoodWiki** 和 **ARXIV papers**）、IFT 与预训练数据 **1:100** 的混合比例、模型幻觉（hallucinations）、**RAG (Retrieval-Augmented Generation)** 的实际应用以及 **GPT-4** 的局限性。主要贡献者包括 `@euclaise`、`@tokenbender`、`@gabriel_syme`、`@giftedgummybee`。

- 在 **general** 频道中进行了广泛的讨论。涉及 AI 模型性能，重点提到了 **Character AI**、**Bard and ChatGPT** 以及 **Claude**。讨论了 **4bit Mistral** 与 **unquantized Mistral** 的性能对比，以及 **gpt4-vision** 的实用性。在训练策略方面，`@fblgit` 提出了 **knowledge infinite-turn conversation training doctrine** 的概念，`@mihai4256` 表示计划进行手动数据集标注。此外还提到了服务器重组和 **Project Obsidian**。频道中充满了对即将推出的模型（如 **UNA Solar**）的期待以及对 **MMLU (Multiple-Choice Machine Learning Understanding)** 结果的预测。

- **ask-about-llms** 频道进行了深入的 LLM 相关讨论。讨论的主题包括在本地部署/运行 LLM（由 `@teknium` 提供建议）、微调 QLORA 模型（来自 `@teknium` 的见解）以及 QLORA 训练期间的验证问题。`@semantic_zone` 询问了关于 GPT-4 微调讨论稀缺的问题，而 `@jaredquek` 提供了关于 **Upstage Solar 10B model** 的性能反馈和微调状态。

**Nous Research AI 频道摘要**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (3 条消息): 
        
- 用户 `@fullstack6209` 强烈建议不要出于未说明的原因使用 Langchain。
- `@maxwellandrews` 评论说，某些拓扑结构（topologies）可以作为任何人创建自己的 **LLM graph query system** 的基础，无论他们是否决定使用基础库。
- `@pogpunk` 询问一名身份不明的用户是否有 Twitter 账号。


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (1 条消息): 
        
- **添加自定义任务**：用户 `@gabriel_syme` 对更广泛的评估表示乐观，赞赏能够包含攻击（attacks）和提示工程（prompt engineering）的能力。然而，这种技术的效率将取决于**添加自定义任务的难易程度**。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (185 messages🔥🔥): 
        
- 对话始于对 **Instruction Finetuning (IFT)** 效果的讨论，以及在微调过程中加入 **pretraining data** 以防止 **catastrophic forgetting** 的可能性。用户 `@euclaise` 建议包含像 **GoodWiki** 这样的数据以保证事实性，但提醒对于像 **Mistral** 这样的模型，这种方法可能会降低模型的性能，因为 Mistral 的基础数据质量极高。

- 讨论进一步演变为考虑将更高质量的数据（如 **ARXIV papers**）与通用 **pretrain data** 结合使用，以在模型针对特定领域进行微调时，保持其非特定领域的能力。这导致了在处理大型 SFT 数据集时，建议将预训练数据与 **Instruction Finetuning (IFT) data** 以 **1:100** 的比例混合。

- `@tokenbender` 提出了一个反向建议，指出如果目标是 **memorization**，对预训练数据进行广泛的微调可能会导致 **significant hallucinations**。这引发了关于成功模型训练策略的讨论，以及在最小化幻觉和数据丢失的同时保留模型基础能力的重要性。

- 成员们还讨论了 **RAG (Retrieval-Augmented Generation)** 在 AI 研究背景下的实际应用。`@gabriel_syme` 和 `@giftedgummybee` 提出了一个 **step-by-step** 的方法，从结构化数据转向任务特定数据，认为这可能是一种更有效的检索方式，因为传统的检索方法会导致不相关的选择。

- 最后，对话触及了 **GPT-4** 的局限性，几位成员对其推理能力和输出表示失望。他们认为该模型的性能似乎比之前的迭代版本有所倒退，而之前的版本显然更有效。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (98 messages🔥🔥): 
        
- **AI 模型性能与使用**：讨论围绕多个 AI 模型的使用和性能展开。`@makya` 对 **Character Ai** 的使用量超过 **Bard** 和 **ChatGPT** 表示惊讶。同样，对于一张显示 **Claude** 使用量远超其他模型的图表也存在困惑，这促使 `@night_w0lf` 建议该图表代表的是总使用量，每个提供商都对这一总数有所贡献。

- **测试模型与潜在改进**：`@gitmo joe` 和 `@teknium` 对 **4bit Mistral** 与 **unquantized Mistral** 的性能持有不同看法，teknium 认为在相同数据集微调的前提下，4bit **Mistral** 表现更好。`@gabriel_syme` 分享了他们第一次测试 **gpt4-vision** 的经验，发现它在多模态环境下的数据标注中很有用，但不清楚如何有效地提供 prompt。

- **AI 训练策略与进展**：`@fblgit` 提出了 **knowledge infinite-turn conversation training doctrine** 的想法，涉及将阅读数据集作为一个迭代和合成的交互过程，以逐步提高模型学习效果。另外，`@mihai4256` 公布了他们继续手动编写数据集样本的计划。

- **Discord 服务器重组**：`@teknium` 提到了服务器的重组以及增加了一个名为 **Project Obsidian** 的公开项目。一些用户提到了新频道的访问问题，但在刷新 Discord 后得到了解决。

- **即将发布的 AI 模型与预测**：成员们期待新 AI 模型（如 **UNA Solar**）的发布。对这些模型在 **MMLU (Multiple-Choice Machine Learning Understanding)** 上的潜在表现进行了预测。一些用户开玩笑说 AGI (Artificial General Intelligence) 将通过随机的模型合并或由 Nous Research 实现。

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (47 messages🔥): 
        
- **在本地部署/运行 LLM**：用户 `@deki04` 寻求在本地部署/运行 LLM 的指南，并倾向于通过 CLI 方式，但 GUI 选项也可以。`@teknium` 分享了一个用于推理的 **OpenHermes** 脚本，该脚本可以适配多轮对话，并建议 `@deki04` 研究 `llama.cpp` 以在 Mac 上实现快速推理。他们指出 Mac 在使用 transformers 时只能利用 CPU。
- **微调 QLORA 模型**：用户 `@.beowulfbr` 分享了他们的配置，并对使用 QLORA 微调拥有 30 万条目的 OpenChat 3.5 模型时训练损失（training loss）没有显著下降表示担忧。`@teknium` 保证这是正常现象，模型在第 1 个 epoch 之后会保持相对平稳。
- **QLORA 训练期间的验证**：用户 `@ruggsea` 询问关于 QLORA 训练期间验证的见解，以及将微调数据集拆分为训练/验证集的最佳实践。
- **GPT-4 微调**：`@semantic_zone` 询问为什么关于 *GPT-4 微调*（特别是针对推理能力）的讨论不多。`@giftedgummybee` 回应指出了几个原因，包括相关的成本高昂以及访问权限受到严格限制。
- **测试和微调 Upstage Solar 10B 模型**：`@jaredquek` 分享了他们对 **Upstage Solar 10B** 模型在复杂法语哲学翻译和哲学问题上的正面评价。他们提到目前正在对其进行微调。

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 摘要

- 宣布了开源领域的重大更新，例如在 transformers 中首次推出 4-bit Mixtral、huggingface_hub 的重大版本发布、Gradio 的大规模更新，以及在 Transformers.js 中添加了新模型和聊天模板，此外还有新的 JavaScript 高斯泼溅（Gaussian Splatting）库 gsplat.js。
- 产品更新发布，包括在 Hugging Chat 中发布 Mixtral、引入针对包含有效 token 的 commit 的预防措施、用于从 GitHub 轻松迁移数据集到 Hugging Face 的工具，以及在 The Hub 和 MLX 上提供的新模型。
- 在“酷炫事物”类别中，宣布了新的 AI 游戏课程、Hugging Face 参加 NeurIPS、邀请成为作者并在 Hugging Face 上发表文章、2023 LLM 之年回顾，以及 2024 年 AI 预测轮。
- 关于 Mistral 的 prompt 格式、训练自定义模型、解决 HuggingFace Datasets 错误、访问私有 Space、用于微调的 gpt 模型建议、微调模型的推理问题、对排行榜中 mixtral 模型评估的担忧，以及向 LLM 排行榜提交需要 `trust_remote_code` 的模型的通用查询。
- 分享了 Hugging Face 新的免费开放 [深度强化学习课程](https://huggingface.co/learn/deep-rl-course/unit0/introduction) 链接、一篇讨论大型语言模型 (LLM) 效率超过可用 DRAM 容量的 [论文](https://huggingface.co/papers/2312.11514)、一篇关于大型视觉语言模型 (LVLM) 偏好蒸馏的研究 [论文](https://huggingface.co/papers/2312.10665)，以及一种没有提供具体细节的 [有趣方法](https://arxiv.org/pdf/2311.09277.pdf)。
- 展示了 Discord 公会内的项目，包括无限场景生成项目、神经风格迁移架构、数字资产所有权项目、自训练的 2x 通用超分辨率模型，以及一个 LLM 数据集污染检测器。
- 读书会讨论围绕理解和可视化扩散模型（diffusion models）、撰写和发布博客文章，以及理解扩散噪声调度（noise schedules）和采样步数展开。
- 宣布了 Segmind 的新 SDXL 变体 '**Segmind-Vega**' 和 '**Segmind-VegaRT**'，提供了更小的体积和更快的速度。
- NLP 讨论集中在 LLM/LORA 词汇表限制、分享《语音与语言处理：自然语言处理、计算语言学和语音识别导论》一书、ctransformers 库的更新以及 GPT 的训练查询。
- 询问关于使用扩散模型将深度图转换为 RGB 图像的领域转换条件化（domain translation conditioning）问题，在分析的消息中未提供回复或资源。

**HuggingFace Discord 频道摘要**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 条消息): 
        
- **开源更新**：宣布了多项重大更新，包括 Transformers 中已支持 4-bit Mixtral [来源](https://twitter.com/_marcsun/status/1735306190391783823)；huggingface_hub 的重大版本发布，包含在 Colab 中更便捷登录等功能 [来源](https://huggingface.co/spaces/Wauplin/huggingface_hub/discussions/3)；Gradio 的大规模更新，修复了大量 Bug 并增加了新功能 [来源](https://twitter.com/gradio/status/1734943002697900045)；以及 Transformers.js 中新增的模型和聊天模板 [来源 1](https://twitter.com/xenovacom/status/1734954388915986717)，[来源 2](https://twitter.com/xenovacom/status/1736906358497202268)。此外，还发布了 JavaScript Gaussian Splatting 库 gsplat.js [来源](https://twitter.com/dylan_ebert_/status/1736857719620161895)。
  
- **产品更新**：Mixtral 现已在 Hugging Chat 中上线 [来源](https://huggingface.co/chat?model=mistralai/Mixtral-8x7B-Instruct-v0.1)；在仓库更新中，包含有效 Token 的 Commit 现在会被拒绝 [来源](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)；通过新工具可以轻松地将数据集从 GitHub 迁移到 Hugging Face [来源](https://twitter.com/vanstriendaniel/status/1736791416263913530)；以及 The Hub + MLX 中上线了新模型，并支持用户提交自己的模型 [来源](https://twitter.com/awnihannun/status/1737510739987120248)。
  
- **精彩内容**：宣布了新的 AI 游戏课程日期 [来源](https://twitter.com/thomassimonini/status/1736776713059586164)；Hugging Face 参加 NeurIPS 的活动 [来源](https://twitter.com/brigittetousi/status/1734699192876970340)；邀请在 2024 年成为作者并在 Hugging Face 上发布内容 [来源](https://twitter.com/mervenoyann/status/1736845977439326464)；2023 年 LLM 年度回顾 [来源](https://twitter.com/clefourrier/status/1736769051098030143)；以及来自 AI 社区知名成员对 2024 年 AI 发展的预测 [来源 1](https://twitter.com/clementdelangue/status/1729158744762626310)，[来源 2](https://twitter.com/julien_c/status/1737121273749078168)，[来源 3](https://twitter.com/vanstriendaniel/status/1737426645039137198)。


### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (58 条消息🔥🔥): 
        
- **Mistral 的 Prompt 格式**：用户 `@drummer_` 询问在哪里可以找到 Mistral 的 Prompt 格式。`@osanseviero` 回复称该格式可以在 Mistral 的模型卡片（model card）中找到。
- **训练自定义模型**：用户 `@kepardev` 咨询如何开始训练自己的模型，想知道是否可以通过向模型展示少量问题及其对应答案来进行训练。
- **HuggingFace Datasets 错误**：用户 `@twisterheather` 在尝试从 Hugging Face 下载某些数据集时遇到了错误（`DatasetGenerationError`）。
- **访问私有 Spaces**：用户 `@tractrixarch` 尝试从公共 Space 访问他的一个私有 Space，但在不提交 Token 的情况下无法正常工作。`@Cubie | Tom` 建议将 Token 添加到公共 Space 的 secrets 中，并在代码中使用 `os.environ.get("...")` 进行加载。
- **用于微调的 GPT 模型**：用户 `@vishyouluck` 征求适合微调的小型 GPT 模型建议。`@Cubie | Tom` 推荐了 `gpt2-large` 和 `TinyLlama/TinyLlama-1.1B-Chat-v0.6`。
- **微调模型的推理问题**：`@vishyouluck` 报告了其微调模型 `VishalMysore/cookgptlama` 的推理问题，显示为内部服务器错误（Internal Server Error）。
- **Mixtral 模型评估**：用户 `@DavidG88` 报告了排行榜中 Mixtral 模型评估的问题，并询问如何联系排行榜团队。`@cakiki` 建议直接在该 Space 下开启一个 Issue。
- **向 LLM 排行榜提交需要 `trust_remote_code` 的模型**：用户 `@testgggggggggg` 询问是否有办法向 LLM 排行榜提交需要设置 `trust_remote_code = True` 的模型。


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (1 条消息): 
        
- **申请加入 RL-Study-Group**：用户 `@cloudhu` 询问如何加入 `rl-study-group` 频道，因为该频道对他显示为锁定状态。提供的消息中没有进一步的讨论或回复。

### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (4 messages): 
        
- **Hugging Face Deep Reinforcement Learning 课程**: `@samanofficial` 分享了 Hugging Face 新推出的免费开源 [Deep Reinforcement Learning course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) 链接。该课程旨在**教授从入门到精通的 Deep Reinforcement Learning**。
- **高效运行 Large Language Models**: `@osanseviero` 对一篇[论文](https://huggingface.co/papers/2312.11514)表示兴奋，该论文讨论了一种**通过将模型参数存储在 flash memory 中，从而高效运行超出可用 DRAM 容量的 Large Language Models (LLMs)** 的方法。
- **提升 Large Vision Language Models 的能力**: `@merve3234` 分享了一篇[有趣的论文](https://huggingface.co/papers/2312.10665)，探讨了 **Large Vision Language Models (LVLMs) 的 preference distillation**，旨在增强它们根据视觉上下文生成有用且忠实回答的能力。研究中使用的模型和数据集可在 Hub 上获取。
- **创新方法讨论**: `@martinmunch` 指出了一种[有趣的方法](https://arxiv.org/pdf/2311.09277.pdf)，但未提供关于论文内容的具体细节。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (18 messages🔥): 
        
- **无限场景生成项目**: `@Lil K` 和 `@shehuiwojiege` 分享了一个无限场景生成项目，提供了代码链接 [`https://www.github.com/HyoKong/DreamD…`](https://www.github.com/HyoKong/DreamD…)、演示链接 [`https://www.huggingface.co/spaces/imsuper…`](https://www.huggingface.co/spaces/imsuper…) 以及主项目页面 [`https://www.hyokong.github.io/dreamdrone-pag…`](https://www.hyokong.github.io/dreamdrone-pag…)。
- **Neural Style Transfer 架构**: `@om7059` 提到了在 PyTorch 中实现 Gatys 等人的 Neural Style Transfer 架构，并分享了一个包含实现结果的 [Twitter 链接](https://twitter.com/alve_om/status/1737169832762880284)。
- **数字资产所有权项目**: `@vedsayys` 介绍了 Mngl.club 的一个项目，旨在增强数字资产所有权体验，并邀请用户通过提供的[链接](https://x.com/mnglclub?s=21)和 [https://t.me/mngl_club](https://t.me/mngl_club) 访问其 X 个人资料和 Telegram 社区。
- **Upscaling Models 演示**: `@helaman` 介绍了其最新发布的自训练 2x 通用 Upscaling Models，并提供了演示和更多细节，详见[此处](https://huggingface.co/spaces/Phips/upscale)。
- **LLM 数据集污染检测器**: `@yeyito777` 根据这篇[论文](https://huggingface.co/papers/2310.16789)创建了一个用于测试 LLM 数据集污染的 Space，并分享了该 [Space 链接](https://huggingface.co/spaces/Yeyito/llm_contamination_detector)。他们解释说，得分超过 0.95 的模型很可能之前见过测试数据。


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (7 messages): 
        
- **理解并可视化 Diffusion Models**: `@asrielhan` 分享了一个 [GitHub 链接](https://github.com/MadryLab/journey-TRAK)，指向一篇关于特定训练数据如何影响 Diffusion Model 图像生成过程的论文。
- **博客文章发布公告**: `@chad_in_the_house` 宣布他们快要写完一篇博客文章了，尽管身处不同的时区。
- **Hugging Face 社区博客**: `@merve3234` 建议 `@chad_in_the_house` 可以将他们的博客文章发布在 **hf.co/blog** 的社区博客板块。
- **Medium 博客文章**: `@chad_in_the_house` 分享了他们在 Medium 上的[博客链接](https://isamu-website.medium.com/understanding-common-diffusion-noise-schedules-and-sample-steps-are-flawed-and-offset-noise-52a73ab4fded)，内容是关于理解 diffusion noise schedules 和 sample steps 的。该博客文章的灵感来自 GitHub 用户 `@bghira` 基于研究论文 "Common Diffusion Noise Schedules and Sample Steps are Flawed" 开发的模型。
- **Hugging Face 博客文章**: `@chad_in_the_house` 确认他们还将在 Hugging Face 上创建一篇博客文章，并为演示提供一个简化版本。

### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 条消息): 

- **Segmind 的新 SDXL 变体**：'@sayakpaul' 宣布发布了由 Segmind 开发的两个新的较小版本的 **SDXL**。第一个模型是 '**Segmind-Vega**'，它是 Stable Diffusion XL (SDXL) 的蒸馏版本，**体积减少了 70%** 且 **速度提升了 100%**。可以在 [Segmind-Vega](https://www.segmind.com/models/segmind-vega) 体验该模型。模型卡（model card）可以在 [这里](https://huggingface.co/segmind/Segmind-Vega) 查看。
- 第二个模型是 '**Segmind-VegaRT**'，这是另一个蒸馏模型。该模型的实时推理可以在 [这里](https://www.segmind.com/segmind-vega-rt) 尝试，API 可以在 [这里](https://www.segmind.com/models/segmind-vega-rt-v1/api) 访问。模型卡可以在 [这里](https://huggingface.co/segmind/Segmind-VegaRT) 查看。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 条消息): 

- **使用 Diffusion 进行领域转换**：用户 `@rapelaheitor` 寻求关于学习领域转换条件化（domain translation conditioning）的建议——特别是使用 Diffusion 模型将 Depth 图像转换为 RGB 图像。他们请求提供任何合适的资源或学习材料。
- **Blend 命令**：用户 `@alchemistaccelerator_22034` 回复了一个简短的评论：`/blend`。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (16 条消息🔥): 

- **LLM/LORA 词表限制**：用户 `@opencuiguy` 询问如何确保 LLM/LORA 模型仅从固定的词汇表中生成内容，例如 ["true", "false"]。`@vipitis` 推测 `@opencuiguy` 似乎是在将 decoder 模型用于分类任务，并建议查看这两个 token 的概率并选择最高的一个。

- **语音与语言处理书籍**：`@stroggoz` 分享了一本名为《Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition》（作者 Daniel Jurafsky 和 James H. Martin）的免费草稿书。`@vipitis` 评论了该书“永远处于草稿状态”的情况。

- **ctransformers 库更新**：`@aiman1993` 指出 ctransformers 库在过去 4 个月内没有更新，导致 `llama.cpp` 难以运行。他们还询问了该库未来的更新计划。

- **Hugging Face 的 NLP 书籍**：`@merve3234` 提到了一本来自 Hugging Face 的关于 Natural Language Processing 的实用书籍，`@pomidorich_` 请求提供链接。`@cakiki` 分享了 O'reilly 网站上 [该书的链接](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)。

- **关于 GPT 训练的疑问**：`@exponentialxp` 询问在 GPT 训练期间，如果 loss 在 60k 次迭代时下降了 5%，其文本质量是否会比其他同样下降 5% 的情况提升更多。他们还询问了在训练中期将 learning rate 更改 10 倍可能产生的负面影响。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 条消息): 

- **领域转换条件化**：用户 `@rapelaheitor` 询问关于 **领域转换条件化（domain translation conditioning）** 的教育材料。他们表达了对将 *Depth 转换为 RGB 图像* 的特定兴趣，即使用 Depth 图像作为条件。在分析的消息中未提供任何回复或资源。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- 用户 `@lightningralf` 分享了对 **4M: Massively Multimodal Masked Modeling** 的探索，并提供了相关[链接](https://4m.epfl.ch/)。
- 讨论了使用 **Huggingface 微调 Llama-2-7b-hf 模型**，针对长度为 8192 的序列，以及如使用 **Mistral** 或将序列长度更改为 8k 等可能的变通方法，尽管有人表示此类调整可能会影响质量。
- `@nruaif` 提到在 **Mistral** 模型上进行了 dropout 设置为 0.5 的测试，且在第 2 个 epoch 开始时并未出现过拟合。
- `@lightningralf` 询问如何在不影响微调的情况下向 **Hermes** 等微调模型添加新知识，`@nruaif` 建议改用 **RAG**，并强调了遗忘其他知识的风险。
- 询问关于在 Llama CCP 中发现多个模型时出现的异常实例，目前尚未提供解决方案。
- 关于 **40gb A100s** 的简短交流，`@faldore` 表示它们没用，而 `@yamashi` 则否认其存在。
- `@seungduk` 询问 **Axolotl** 在启用样本打包（sample packing）时是否可能在序列长度内合并样本，并推测其使用 **二分查找 (binary search)** 来寻找下一个样本。
- `@latentfog` 询问是否支持 **Fill In The Middle (FIM)** 来微调代码库模型。
- `@wizmak` 询问关于根据任务映射模型的策略，即由用户 prompt 决定使用的模型。
- `@enima` 询问如何对预训练的大语言模型 (LLM) 进行无监督微调，重点是领域适配，`@noobmaster29` 建议使用额外的文本数据继续进行预训练。
- [`@visuallyadequate` 分享了一篇展示本地可训练 LLM 潜力的文章](https://github.com/bublint/ue5-llama-lora)，强调了向模型注入知识的可能性。
- `@noobmaster29` 认可使用 RAG (Retrieval-Augmented Generation) 向模型添加特定信息，并推荐将 [NVIDIA 的 ChipNeMo](https://d1qx31qr3h6wln.cloudfront.net/publications/ChipNeMo%20%282%29.pdf) 作为有价值的资源。
- `@_awill` 请求有关理解 llama.cpp 内部机制的帮助。

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (34 条消息🔥): 
        
- **4M: Massively Multimodal Masked Modeling**: 用户 `@lightningralf` 分享了用于训练多模态和多任务模型的 **4M 框架** [链接](https://4m.epfl.ch/)，并将其应用于各种标记化的模态，补充说这可能对 `@257999024458563585` 和 `@208256080092856321` 有意义。

- **使用 Huggingface 微调 Llama 模型**: 用户 `@staticpunch` 提出了关于使用 Huggingface 微调 Llama-2-7b-hf 模型以处理长度为 8192 的序列的问题，尽管模型的 `config.json` 文件中 `"max_position_embeddings": 4096`。作为回应，`@nanobitz` 建议使用 **Mistral** 可能是一个选择，或者在 yaml 中将序列长度更改为 8k，尽管质量可能会受到影响。

- **Mistral 与 Dropout**: `@nruaif` 透露他们正在 **Mistral** 模型上运行 dropout 设置为 0.5 的测试，随后补充说模型在第 2 个 epoch 开始时没有过拟合。

- **向微调模型插入新知识**: 用户 `@lightningralf` 询问是否可以通过预训练的方式向 **Hermes** 等微调模型插入新知识，而不影响微调效果。作为回应，`@nruaif` 建议为此使用 RAG，并重申尝试向预训练模型插入知识可能会导致其遗忘其他知识。

- **Llama CCP 中发现多个模型的异常**: 用户 `@dangfutures` 询问在 Llama CCP 中发现多个模型时遇到的异常的变通方法。在给定的消息中尚未提供解决方案。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (5 条消息): 
        
- **A100s 40gb 讨论**: `@faldore` 提到 **40gb A100s** 对他来说没用，然而 `@yamashi` 反驳说 40gb A100s 并不存在。 
- **Axolotl 中的样本打包 (Sample Packing)**: `@seungduk` 询问 **Axolotl** 在启用样本打包时是否可能在序列长度内合并样本，并使用不同的 position ids（如 0, 1, 2, 3, ..., 0, 1, 2, 3...）。他还注意到 Axolotl 似乎使用 **二分查找 (binary search)** 来寻找下一个要合并的样本。
- **微调代码库模型的 FIM 支持**: `@latentfog` 询问是否支持 **Fill In The Middle (FIM)** 来微调代码库模型。

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (19 条消息🔥): 
        
- **根据任务进行模型映射**：`@wizmak` 询问了关于根据任务映射模型的方法或框架，其中用户 prompt 决定了请求分配给哪个特定模型。
- **预训练 LLM 的无监督微调**：`@enima` 征求关于如何以无监督方式微调预训练大语言模型（LLM）的建议，重点在于领域适配（domain adaptation）。`@noobmaster29` 建议使用额外的文本数据进行持续预训练（continued pre-training），并提到关于使用 LoRA/QLoRA 微调向 LLM 添加知识的共识尚不统一。
- **LLM 微调示例**：`@visuallyadequate` [分享了一篇文章](https://github.com/bublint/ue5-llama-lora)，展示了本地可训练 LLM 的潜力。建议指出，尽管存在潜在挑战和陷阱，向模型注入知识是可行的。 
- **使用 RAG 注入知识**：在向模型添加特定信息时，`@noobmaster29` 强烈推荐使用 RAG（Retrieval-Augmented Generation）。[来自 NVIDIA 的 ChipNeMo](https://d1qx31qr3h6wln.cloudfront.net/publications/ChipNeMo%20%282%29.pdf) 也被提为该主题下最受喜爱的论文。
- **理解 Llama.cpp 内部机制**：`@_awill` 寻求有关 llama.cpp 内部机制的帮助。


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (1 条消息): 
        
emperor: 只有 90%？


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- 由 `@semantic_zone` 发起的关于 **GPT-4 微调**的讨论，质疑为什么关于其微调能力的讨论较少。
- `@slono` 对 **chat.openai.com** 的使用表示怀疑，建议使用 Mixtral 或 70B 模型等替代方案，这些模型提供更快的速度和代码生成功能。
- `@lightningralf` 分享了 [RΞASON 框架](https://www.tryreason.dev/blog/introducing-reasonn)，这是一个用于构建 LLM 应用程序的开源 TypeScript 后端。
- `@swyxio` 介绍了**模型合并（Model Merging）**技术，并分享了一篇[文章](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized)。
- **概率编程（Probabilistic Programming）**被讨论为编程语言中具有挑战性的一等公民（first-class）特性，这是 `@swizec` 分享的观点。
- 展示了 AI 在潜空间（latent space）中飞行无人机的应用，由 `@kylemathews` 在一篇[博客文章](https://bricolage.io/flying-drones-latent-space/)中分享。
- 宣布即将由 `<@458440277548335125>` 主持关于 **LlamaGuard/Purple Llama 论文**的讨论，`@Swyxio` 确认了新播客节目的发布。
- `@swizec` 对一篇 Meta 研究论文的可读性表示赞赏，同时 `@swyxio` 分享了 [Andrej Karpathy 的推文](https://twitter.com/karpathy/status/1734659057938477174)，其中推荐了几篇论文。
- 针对 `@ayenem` 索要 Zoom 链接的请求，`@swyxio` 澄清会议是在 Discord 上进行的。
- `@swizec` 质疑作者决定发布 **Llama Guard 权重**却未随后在 ToxicChat 上进行微调。
- 计划下周讨论 Karpathy 推荐的一篇[论文](https://arxiv.org/abs/2312.06585)，`@eugeneyan` 正在考虑进行分享。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (12 条消息🔥): 
        
- **GPT-4 微调**：`@semantic_zone` 询问了关于获取用于推理的 **GPT-4 微调 API** 的问题，并询问为什么关于其微调可能性的讨论不多。
- **Chat.OpenAI.Com 的使用**：`@slono` 对使用 chat.openai.com 的效用表示怀疑，因为像 Mixtral 或 70B 模型这样的替代方案提供了速度和代码生成能力。
- **RΞASON 框架**：`@lightningralf` 分享了 [RΞASON 框架](https://www.tryreason.dev/blog/introducing-reasonn)，它为使用大语言模型（LLM）构建应用程序提供了后端开源 TypeScript 基础设施。
- **模型合并**：`@swyxio` 链接了一篇关于[模型合并（model merging）](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized)的文章，这是人们开始探索的一种技术。 
- **LLM 作为原语**：`@swizec` 认为概率编程很困难，并且已经有各种尝试将其作为编程语言中的一等公民特性。
- **在潜空间中应用 AI 飞行无人机**：`@kylemathews` 在 [bricolage.io](https://bricolage.io/flying-drones-latent-space/) 上分享了一篇博客文章，讨论了在潜空间（latent space）中应用 AI 飞行无人机。

### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (2 条消息): 
        
- **LlamaGuard/Purple Llama 论文讨论**：根据 `@Swyxio` 的公告，`<@458440277548335125>` 将在 30 分钟后主持一场关于 LlamaGuard/Purple Llama 论文的会议。所讨论的 [论文](https://arxiv.org/abs/2312.06674) 由一个庞大的团队撰写，包括 [Hakan Inan](https://arxiv.org/search/cs?searchtype=author&query=Inan,+H)、[Kartikeya Upasani](https://arxiv.org/search/cs?searchtype=author&query=Upasani,+K)、[Jianfeng Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi,+J) 等。建议感兴趣的成员加入 `<#1107320650961518663>` 以接收 Discord 通知。
- **新播客集**：`@Swyxio` 宣布发布了新的播客集，并感谢 `<@194927177265840128>` 的贡献。可以通过此 [链接](https://fxtwitter.com/latentspacepod/status/1737572584995360860) 收听播客。


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (9 条消息🔥): 
        
- **关于一篇易读论文的讨论**：`@swizec` 表达了对 Meta 研究团队一篇论文的喜爱，称赞其写作的可读性。
- **论文推荐**：`@swyxio` 分享了 [Andrej Karpathy 的一条推文](https://twitter.com/karpathy/status/1734659057938477174)，其中提供了几篇论文推荐，并提到 Karpathy 建议阅读这篇 [论文](https://arxiv.org/pdf/2312.06585.pdf)。
- **请求 Zoom 链接**：`@ayenem` 询问加入讨论的 Zoom 链接。然而，`@swyxio` 澄清说，现在的会议在 Discord 的特定 [频道](https://discord.com/channels/822583790773862470/822583791217934366) 中进行。
- **关于 Llama Guard 权重的疑问**：`@swizec` 对作者决定发布 Llama Guard 权重，但未发布在 ToxicChat 上进一步微调后的权重的做法表示疑问。
- **下周论文**：`@swyxio` 宣布下周讨论的论文将是 Karpathy 赞赏的这篇 [论文](https://arxiv.org/abs/2312.06585)，并鼓励新参与者主持讨论。`@eugeneyan` 觉得这篇论文很有趣，正在考虑进行演示。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 发现使用 OpenAI 的 `return_direct` 配合流式传输回调方法时存在问题，表现为不可预测的 'Final Answer'。此外，还有一个关于如何将 OpenAI Assistant API 与数据库集成，而无需为整个 Assistant API 编写完整脚本的咨询。
- 分享了针对 LLM 的训练资源和练习 Prompt，并表达了对专门针对 LLM 的类 *LeetCode* 平台的渴望。
    - ["Adventure 6 - Prompt Practice"](https://gandalf.lakera.ai/adventure-6)
- 报告了在本地 LLM 模型中调整 `max_tokens` 参数的挑战，观察到设置 `max_tokens` 后未能产生预期的 Token 长度。
- 请求协助确定 LangChain 中 PGVector 已索引和添加的文档数量，问题详见此 StackOverflow [帖子](https://stackoverflow.com/questions/77691556/langchain-pgvector-how-to-find-out-how-many-documents-have-been-indexed-and-ad)。
- 有兴趣设计一个系统，将多个 LLM 请求的结果整理成针对用户查询的综合结果，并请求涉及 `RunnableParallel` 的项目模板或入门示例。
- 在使用 LangServe 时遇到困难，例如解析模板应用的输出，特别是在尝试过滤 JSON 对象显示时聊天记录消失的情况，相关的模板应用可在 [此处](https://github.com/langchain-ai/langchain/tree/9ef2feb6747f5a69d186bd623b569ad722829a5e/templates/retrieval-agent) 找到。还发现了在 LangServe 中添加路由以及随后的意外错误。
- 确认在 `ConversationBufferMemory` 中设置 `output_key="output"` 是使 LangServe 正常运行的必要配置，尽管独立的 `AgentExecutor` 在没有此设置的情况下也可以运行。
- 分享了 Analytics Vidhya 上的 [文章](https://www.analyticsvidhya.com/blog/2023/12/transforming-interactions-with-chatgpt-plugins/)，探讨了 ChatGPT 插件在数字叙事和用户参与中的变革作用。
- **加密货币职位**：*gauravmandal* 分享了一个专注于加密领域职位的 Discord 频道邀请，可能对广大受众有用。 
    - [Discord 邀请](https://discord.gg/cryptojob)。

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (12 messages🔥): 
        
- **处理 `return_direct` 和流式传输 (Streaming)**：用户 `@a404.eth` 在使用回调方法进行流式传输时，正苦于处理 `return_direct`。他们表示如果使用 `stream` 方法，就没有可预测的 `Final Answer`。该问题尚未解决。
- **OpenAI Assistant API 经验**：`@refik0727` 正在寻求帮助，想知道如何在不为整个 Assistant API 编写脚本的情况下（特别是在 OpenAI 平台内部），将其数据库连接到 OpenAI Assistant API。
- **LLM 训练资源**：`@seththunder` 分享了一个练习 [提示词注入/任务 (prompt injection/task)](https://gandalf.lakera.ai/adventure-6) 的链接，供那些对 *Language Learning Models (LLMs)* 感兴趣的人参考。同时，`@schimazing` 询问是否存在专门针对 LLM 的 *LeetCode* 类型网站。
- **调整本地 LLM 模型的 Max Tokens 遇到麻烦**：初学者 `@ninamani` 在调整本地托管 LLM 模型的 `max_tokens` 参数值时遇到问题。具体表现为，当他们将 `max_tokens` 设置为 600 时，生成的输出仍倾向于保持在 400 个 token 左右。
- **寻求 PGVector 方面的帮助**：`@alekseyr1987` 正在寻求关于 LangChain 中 PGVector 的帮助，特别是如何查看已索引和添加了多少文档。该用户提供了 Stack Overflow 上具体问题的 [链接](https://stackoverflow.com/questions/77691556/langchain-pgvector-how-to-find-out-how-many-documents-have-been-indexed-and-ad)。
- **链式查询项目模板**：`@squadzero.` 提到有兴趣开发一种链 (chain)，将多个 LLM 请求的结果汇总成一个整体以响应用户查询。他们正在寻找任何项目模板或启动器，可能涉及 `RunnableParallel`。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (8 messages🔥): 
        
- **在 Langserve 中解析输出**：用户 `@rodralez` 在尝试解析 Langserve 模板应用的输出时遇到了问题。他们希望在输出窗口中仅显示 JSON 对象的 "output" 键，而不是整个 JSON 对象。尝试通过使用 `lambda x: x["output"]` 来实现这一点，但导致聊天记录消失。他们正在寻求该问题的解决方案。所使用的模板应用可以在 [这里](https://github.com/langchain-ai/langchain/tree/9ef2feb6747f5a69d186bd623b569ad722829a5e/templates/retrieval-agent) 查看。

- **LangServe 路由添加问题**：`@vilelaone` 在使用 LangServe 为其 `AgentExecutor` 添加路由时遇到了问题。虽然他们的 Agent 独立运行正常，但在使用 LangServe 添加时失败了。尝试使用自定义输入和输出模型导致了 ValueError，提示预期一个输出键但收到了多个。

- **ConversationBufferMemory 的影响**：一次简短的讨论表明，使用 `ConversationBufferMemory` 可能是导致上述问题的原因，因为 `@rodralez` 的聊天记录消失了，而 `@vilelaone` 的 LangServe 路由添加失败了。

- **通过 output_key 解决问题**：`@vilelaone` 通过在 `ConversationBufferMemory` 中使用 `output_key="output"` 解决了他们的问题。值得注意的是，这对于 LangServe 是必要的，尽管独立的 `AgentExecutor` 在没有它的情况下也能正常工作。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- `@gauravmandal` 分享了一个专注于加密货币职位的 [Discord 小组](https://discord.gg/cryptojob) 链接。
- **ChatGPT 插件**：`@soumyadarshani` 发布了一篇 [Analytics Vidhya 文章](https://www.analyticsvidhya.com/blog/2023/12/transforming-interactions-with-chatgpt-plugins/) 的链接，该文章讨论了通过 ChatGPT 插件改变用户交互。文章指出，这些插件正在彻底改变数字叙事和用户参与度。


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **加密货币职位 Discord 邀请**：用户 `@gauravmandal` 分享了一个专注于加密货币行业职位的 Discord 频道 [邀请链接](https://discord.gg/cryptojob)。他标记了 `@everyone` 和 `@here`，表明这些信息可能引起该群组的广泛兴趣。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- 关于 **7b 模型微调挑战** 的对话，观察到 7b 模型在使用 LoRA 进行微调时，由于容易出现性能下降和灾难性遗忘，因此很难进行微调。*用户 `@.calytrix` 推测这可能是由于 Mixtral 稠密且低冗余的 7b 模型导致的*。
- **基础 7b 对比旧版 7b** 模型的比较对话。用户 `@fernando.fernandes` 提到，尽管旧版 7b 模型更稠密且对灾难性遗忘更敏感，但新版 7b 模型似乎对每个人来说都更具挑战性。
- `@fernando.fernandes` 讨论 **自注意力正交性与性能** 的理论，提出性能可能与自注意力层中的信息量有关。特别是，在表现不佳的模型（如 undi95 Mixtral 微调版）中，自注意力层更加正交。
- `_jp1_` 和 `@fernando.fernandes` 提出的 **微调** 解决方案。他们建议采用更高的 dropout 率、冻结路由层（router layers）以及可能冻结 embed_token 层等方法。所有建议旨在提高 Mixtral 7b 等模型的性能。
- `@le_mess` 和 `@.pathos` 围绕 **Disco Research** 的讨论，重点关注 20 世纪 70 年代迪斯科音乐的影响。
- `@bjoernp` 分享了 **LEOIm 预印本（Preprint）发布** 的更新。尽管由于持续的改进和评估导致预印本发布延迟，但已确认未来会发布。
- `@bjoernp` 提供了关于 **LEOIm 训练** 的详细信息。Mistral 7b 的训练是在 A100 上进行的，速度约为 3000 tokens/s/GPU，使用了大约 650 亿个 tokens。

**DiscoResearch 频道总结**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (9 条消息🔥): 
        
- **7b 模型的微调挑战**：用户 `@.calytrix` 观察到 7b 模型在使用 LoRA 微调时特别具有挑战性，容易导致性能下降或灾难性遗忘。该用户推测问题可能与 **Mixtral** 稠密、低冗余的 7b 模型有关，这些模型对 LoRA 微调的容忍度可能较低。
- **基础 7b 对比旧版 7b**：`@fernando.fernandes.` 分享了观察结果，即无论采用何种微调方法，每个人似乎都在新版 7b 模型上遇到困难。这与旧版 7b 模型的经验相反，旧版模型甚至更稠密，因此更容易受到灾难性遗忘的影响。
- **自注意力正交性与性能**：`@fernando.fernandes.` 提出，存储在自注意力层（被构想为数据库）中的信息量与其排名（rankings）和正交性有关。他指出，对于性能较差的模型（如 undi95 Mixtral 微调版），自注意力层往往更加正交。在这里，正交性是通过计算来自不同 Expert 的自注意力模块权重之间的 Frobenius 范数来衡量的。
- **微调的潜在解决方案**：用户 `_jp1_` 提出 QLoRA 微调在路由层（router 或 gate 层）上可能效果不佳，因此需要更高的 dropout 率。未来的微调轮次如果加入冻结的路由层以及其他的 bug 修复/改进，可能会显著提高性能。`@fernando.fernandes.` 对此表示赞同，并建议可能也有必要 **冻结 embed_token 层**，尽管其产生积极影响的原因仍需进一步研究。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (9 条消息🔥): 
        
- **Disco Research**：参与者 `@le_mess` 和 `@.pathos` 提到他们正在进行关于 20 世纪 70 年代迪斯科音乐影响的研究。
- **LEOIm 预印本发布**：用户 `@zitronesimo` 向 `@bjoernp` 询问了 LEOIm 预印本的发布情况。`@bjoernp` 回复称，由于正在改进贡献内容、进行额外评估以及处理其他项目，预印本有所延迟，但肯定会在适当的时候发布。
- **LEOIm 训练细节**：在 `@zitronesimo` 的进一步询问下，`@bjoernp` 提供了关于 **Mistral 7b** 训练的具体细节。他表示训练是在 **A100** 上进行的，速度约为 **3000 tokens/s/GPU**，训练使用了约 **650 亿个 tokens**。


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- 关于 **Cursor vs. VS Code Copilot** 的讨论以及决策标准，包括代码输出质量、上下文构建和面向代码库的讨论。提供了一份 Copilot 最近增强功能的概述，并附带了一个展示这些功能的 [YouTube 链接](https://www.youtube.com/watch?v=SZVCJRUADc4)。
- 提出了对 **Assistants API 和 GPTs 集成**性能的担忧。讨论探索了可能的提速方案，包括缓存结果、等待 OpenAI 修复以及巧妙的解决方案。还幽默地提到了产品速度的突然提升。
- `@dongdong0755` 的建议提供了一个关于 **prompt splitting**（提示词拆分）的有趣实验，以及关于他们工作中提取（extractions）的一个现有问题。还强调了为 Discord 提供潜在 *embeddings search functionality*（嵌入搜索功能）的公认实用性。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 条消息): 
        
joshcho_: 很有可能是 llamaindex


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 条消息): 
        
joshcho_: holy


### ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/) (4 条消息): 
        
- **VS Code Copilot vs Cursor**: 用户 `@robhaisfield` 表达了对于是坚持使用 **Cursor** 还是换回专门使用 **VS Code Copilot** 的犹豫，原因是 Copilot 最近的功能和 UX 变化。该用户认为尽管 Copilot 有所改进，但 Cursor 在产生更好的输出和高级上下文构建方面仍具有优势。
- **Cursor 的优势**: `@robhaisfield` 强调了使用 Cursor 的一个优点是 **所有关于代码库的对话都分组在该代码库内**，与将对话分散在所有 ChatGPT 对话中相比，这创造了一个更有条理的系统。
- **关于新功能的提问**: `@jeffreyw128` 询问了 **VS Code Copilot** 最近添加的功能。 
- **解释 VS Code Copilot 的新功能**: 作为回应，`@robhaisfield` 列举了几项增强功能，包括内联聊天（inline chat）、工作区搜索命令、从网站或仓库加载文档的能力，以及通过聊天命令编辑代码块。通过一个 [YouTube 视频](https://www.youtube.com/watch?v=SZVCJRUADc4) 链接提供了这些功能的详细演示。


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (5 条消息): 
        
- **Assistants API + GPTs 的性能**: `@joshcho_` 对 Assistants API 和 GPTs 集成的缓慢速度表示担忧，并询问是否有诸如 **缓存结果** 之类的方法来克服这一问题。 
- **OpenAI 产品发布担忧**: `@jeffreyw128` 怀疑 OpenAI 可能过早发布了产品，导致了速度缓慢，并建议要么等待 OpenAI 纠正问题，要么构建自己的解决方案来加速过程。
- `@joshcho_` 注意到产品速度有明显提升，并对此感到有些好笑。


### ▷ #[feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549/) (1 条消息): 
        
joshcho_: 我认为检索会很有用。比如针对 Discord 的 embeddings search


### ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/) (2 条消息): 
        
- **Prompt Splitting 实验**: 用户 `@dongdong0755` 建议进行一项将提示词拆分为两部分的实验，看看性能是否会有所不同。
- **提取问题**: 用户 `@dongdong0755` 还提到在他们的工作中面临关于提取（extractions）的困境。

        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

只有一个频道有活动，因此无需总结...

- **用于增强推理的定制过滤数据集**：`@far_el` 讨论了他们在一个经过定制过滤且格式化以增强推理能力的数据集上进行了训练。该数据集还针对多种 Prompt 格式进行了训练。Far_el 希望获得关于该模型的反馈。([来源](https://lstm.load.to.discord))
- **用例 - 理解大型代码库**：`@spirobel` 分享了他们理解并扩展大型代码库的用例。Spirobel 尝试了 Phind codellama 34b 和 Mistralic，并意识到在从 git diff 输出中检测重要函数名这一特定用例中，Mistralic 的表现优于 Amazon 的 Mistrallite。Spirobel 希望了解为什么 Mistralic 在这一特定检索任务中表现更好。([来源](https://lstm.load.to.discord))
- **Mistralic 在代码检索方面优于 Mistrallite**：`@spirobel` 指出，尽管 Mistrallite 据称针对检索进行了优化，但 Mistralic 在代码检索任务中的表现更好。Spirobel 推测“检索”的概念在不同语境下可能有所不同。([来源](https://lstm.load.to.discord))
- **Mistralic 更好的泛化能力**：`@far_el` 假设 Mistralic 更好的性能可能归功于他们使用了多种 Prompt 格式，这可能使其具有更好的泛化能力。Far_el 将对此进行进一步调查，并计划开源他们拥有的关于 Mistralic-1 的所有内容。([来源](https://lstm.load.to.discord))
- **适用于 H100 的 Axolotl Docker 镜像**：`@tcapelle` 询问是否有兼容 H100 的 Axolotl Docker 镜像。([来源](https://lstm.load.to.discord))
- **Mistralic 与 OpenHermes 2.5 性能对比**：`@spirobel` 表示，通过实验发现，与 OpenHermes 2.5 相比，Mistralic 更鲁棒且产出质量更高。Mistralic 的输出通常具有完美的 Markdown 格式。([来源](https://lstm.load.to.discord))

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- 在 'ai-and-ml-discussion' 频道中，`@entropi` 讨论了通过文章 [Introducing Text-to-CAD](https://zoo.dev/blog/introducing-text-to-cad) 介绍的新工具。
- 用户 `@algomancer` 和 `@rabiussany` 在 'looking-for-collabs' 频道中宣布了开源和研究项目的合作机会，提到了特定的兴趣领域，并欢迎通过私信讨论。
- `@teknium` 在 'general-chat' 中分享了微调项目的源代码，`@propback` 提供了在 [openchat](https://github.com/imoneoi/openchat/blob/master/README.md) 找到的 GitHub 仓库链接。

**Alignment Lab AI 频道总结**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (1 条消息): 
        
entropi: https://zoo.dev/blog/introducing-text-to-cad


### ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/) (2 条消息): 
        
- **开源贡献**：用户 `@algomancer` 提议在假期期间为开源和开放研究项目做贡献。他们的兴趣领域包括推理时的可变速率计算（variable rate compute at inference）、非标准类别的生成模型、除自回归解码器之外的 Jepa 风格模型，以及用于增强可控性的条件方案（conditioning schemes）。他们擅长编写 Triton/PyTorch 和数据流水线（data pipelines）。
  
- **研究项目合作**：用户 `@rabiussany` 为任何深度学习研究项目提供帮助。他们欢迎通过私信进行合作讨论。


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (3 条消息): 
        
- **微调代码源码**：用户 `@teknium` 转达称其项目的完整微调代码已托管在 GitHub 上，但未提供链接。用户 `@propback` 随后提供了 [openchat 仓库的链接](https://github.com/imoneoi/openchat/blob/master/README.md)，其中包含该项目的训练指南。