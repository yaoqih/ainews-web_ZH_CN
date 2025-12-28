---
companies:
- nous-research
- openai
- mistral-ai
- hugging-face
- ollama
- lm-studio
date: '2023-12-10T23:49:57.169413Z'
description: '**Nous Research AI** Discord 社区讨论了参加 **NeurIPS** 会议以及在澳大利亚组织未来 AI 活动的相关事宜。讨论亮点包括对开源和去中心化
  AI 项目的浓厚兴趣，其中 **Richard Blythman** 正在寻找联合创始人。用户分享了 **Photo GPT AI** 等项目，并介绍了 **StableLM
  Zephyr 3B**。


  基于 **Mistral** 的 **Mixtral** 模型引发了关于性能和 GPU 需求的辩论，社区将其与 **GPT-3.5** 进行了对比，并认为该模型在微调后可能具备与
  **GPT-4** 竞争的潜力。在微调和评估方面，**Tensorboard**、**Wandb** 和 **Llamahub** 等工具被重点提及。讨论内容还涵盖了**混合专家
  (MoE)** 架构、有限数据下的微调以及 ChatGPT 的推理优化策略。


  此外，社区的梗图和互动中还提到了 **Andrej Karpathy** 和 **Yann LeCun** 等 AI 领域的知名人物。社区成员还分享了与这些模型和工具相关的
  GitHub 链接及 YouTube 视频等资源。'
id: b4652fae-87f4-454e-8635-09c342357284
models:
- mixtral-8x7b-32kseqlen
- mistral-7b
- stablelm-zephyr-3b
- openhermes-2.5-neural-chat-v3-3-slerp
- gpt-3.5
- gpt-4
original_slug: ainews-12102023-not-much-happened-today
people:
- andrej-karpathy
- yann-lecun
- richard-blythman
- gabriel-syme
- pradeep1148
- cyborg_1552
title: 2023年12月10日：今天没发生什么特别的事。
topics:
- fine-tuning
- mixture-of-experts
- model-benchmarking
- inference-optimization
- model-evaluation
- open-source
- decentralized-ai
- gpu-optimization
- community-engagement
type: archival
---

<!-- buttondown-editor-mode: plaintext -->虽是老生常谈，但今天是个清静的日子，大家都在动身前往 NeurIPS（我们也一样）。Andrej 在[这里](https://twitter.com/karpathy/status/1733968385472704548)点名了一些 alpha 来源，我们正考虑加入它们，请告诉我们还有哪些 Reddit/Discord 频道或动漫头像的匿名用户值得关注。

swyx

[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- 成员们表达了参加 **NeurIPS** 并见面的兴趣，并对未来在澳大利亚举办 AI 活动提出了建议。`@richardblythman` 敦促那些对开源、去中心化 AI 项目感兴趣的人与其联系。用户们分享了自己的项目，例如 `@cyborg_1552` 的 [photo GPT AI tool](https://www.photogptai.com/) 以及 `@pradeep1148` 对 [StableLM Zephyr 3B 的介绍](https://www.youtube.com/watch?v=YWYNLaWDoNQ)。
- 用户 `@gabriel_syme` 通过分享 [GitHub 链接](https://github.com/open-compass/MixtralKit)引发了关于 **Mixtral** 的关注。**Mixtral 和 GPT-3.5** 之间的性能对比引发了热烈讨论。`@mihai4256` 发布了他们的微调模型 Pallas-0.2，可在 [Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.2) 上获取。一段讨论开源 LLM 使用的 [Youtube 视频](https://youtu.be/y9k-U9AuDeM?si=2X5j64_cdsdKwWEw)引起了简短的反响。
- `OpenHermes-2.5-neural-chat-v3-3-Slerp` 和 **Mixtral** 都因其性能表现而备受追捧，同时大家也在争论后者对 GPU 的要求。`Tensorboard`、`Wandb`、`evalplus`、`llamahub` 等工具被认为对微调和评估模型非常有益。用户交流了在 **Ollama** 和 **LM Studio** 等模型托管平台上的使用体验，双方各持己见。
- 由 `@gabriel_syme` 发起的一场关于 **MoE** 的深入对话阐明了为什么基于 **Mistral** 的 Mixtral 模型与之前的实现有所不同。关于微调 LLM 的讨论表明数据需求有限。有人提出 **Mixtral** 在微调后有潜力与 **GPT-4** 竞争。`@wlrd` 解释了开源 LLM 如何实现，并引出了 **OpenHermes 2.5 - Mistral 7B** 模型。关于 **GPT-3.5** 的推测认为它是一个 20B 模型，并预测其很快会开源。关于 ChatGPT 推理优化的讨论涉及了策略性批处理（strategic batching）、潜在缓存以及用户群规模。
- **memes** 频道中，成员们分享了各种表情符号和梗图，用于娱乐和交流。大家表达了对 **Yann** 和 **Karpathy** 等演讲者的特别关注。用户 `@teknium` 幽默地将一个角色描述为对 x risk 深感忧虑。

**Nous Research AI 频道总结**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (19 条消息🔥): 
        
- **NeurIPS 聚会**: 
    - `@blue_matcha` 询问是否有人参加 NeurIPS 并希望见面，`@teknium` 表示他们可能在周四和周五有空。 
    - `@gabriel_syme` 对 NeurIPS 总是选在美国表示失望，随后透露他们常驻澳大利亚。`@gabriel_syme` 还提议明年在澳大利亚举办活动。

- **寻找开源和去中心化 AI 联合创始人**: 
    - `@richardblythman` 正在为开源和去中心化 AI 领域的项目寻找联合创始人，并请感兴趣的人私信他们。

- **对澳大利亚 AI 会议的兴趣**:
    - `@deki04` 指出，人们对在澳大利亚举办 AI 会议会有很大兴趣，并提到 Jeremy Howard 在布里斯班举办的一场座无虚席的线下 fastAI 课程。

- **Photo GPT AI 开发**: 
    - `@cyborg_1552` 提到了使用 Stable Diffusion 开发的[工具](https://www.photogptai.com/)，并表示如果大家感兴趣，会写一篇博客文章。他们还为想要进一步探索的人提供了 [GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) 链接。

- **StableLM Zephyr 3B 介绍**:
    - `@pradeep1148` 分享了一段介绍 StableLM Zephyr 3B（一种大语言模型）的 [YouTube 视频](https://www.youtube.com/watch?v=YWYNLaWDoNQ)。


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (1 条消息): 
        
nonameusr: 我觉得他用了 Markdown

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (24 messages🔥): 
        
- **关于 Mixtral 及其架构的讨论**：`@gabriel_syme` 分享了一个 [GitHub 链接](https://github.com/open-compass/MixtralKit) 指向 MixtralKit —— 一个用于 `mixtral-8x7b-32kseqlen` 模型的工具包。`@cyborgdream` 发布了一个 [twitter 链接](https://twitter.com/abacaj/status/1733660077154816013)，分享了 Mixtral 在微调前就在 Benchmark 中超越了 GPT-3.5。随后的讨论涉及 `@nonameusr`、`@euclaise` 和 `@chhillee` 辩论 Mixtral 基于 Transformer 架构的优势和独特性。

- **新微调模型发布**：`@mihai4256` 宣布发布了他们的微调模型 Pallas-0.2，托管在 [Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.2) 上。该模型是 `Tess-34B-v1.4` 的微调版本，专为推理任务设计，在长 System Prompts 下表现良好。

- **关于开源 LLM 使用的视频**：`@teknium` 分享了一个 [Youtube 视频](https://youtu.be/y9k-U9AuDeM?si=2X5j64_cdsdKwWEw)，回答了“你应该使用开源大语言模型吗？”这一问题。`@n8programs` 和 `@nonameusr` 对该问题给出了单字回答，但观点截然相反。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (639 messages🔥🔥🔥): 
        
- **微调与性能讨论**：用户讨论了多个模型的微调和性能，包括 **Hermes 2.5**、**Mistral** 和 **GPTs Agent**。例如，`@nonameusr` 认为 `OpenHermes-2.5-neural-chat-v3-3-Slerp`（绰号 "Slurpy"）在某些方面优于原始的 `Hermes`，但也指出存在不一致性。几位用户还讨论了 `Mixtral`（或 `Mixtral MoE`）的性能，涉及其 GPU 需求以及量化（quantized）后的表现。

- **模型托管与管理平台**：多位用户比较了使用 **Ollama** 和 **LM Studio** 托管和管理 AI 模型的经验。虽然一些用户更倾向于 Ollama，但其他人指出 LM Studio 可能更具可定制性，并能更好地支持更广泛的模型。

- **计算与训练资源**：像 `@vatsadev` 和 `@gabriel_syme` 这样的用户讨论了他们的计算资源，讨论还涉及了大学资源的潜力。

- **实用工具**：讨论还涉及了各种工具，如 `Tensorboard`、`Wandb`、`evalplus` 和 `llamahub`，这些工具对于微调、测试和评估模型非常有用。

- **新模型与技术**：频道中提到了新的模型和技术，如 “slerp”（在 `OpenHermes-2.5-neural-chat-v3-3-Slerp` 的背景下）。一些用户还推测了 `Mixtral` 和 `StripedHyena` 模型，以及通过微调或合并（merging）策略进一步改进它们的潜力。最后，`@ldj` 认为 `Mixtral` 在计算过程中选择“专家”（experts）的方法可能会影响其性能。

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (123 messages🔥🔥): 
        
- **Mixture of Experts (MoE) 讨论**：用户 `@akhxl`、`@cyborgdream` 和 `@gabriel_syme` 参与了关于 MoE 的对话。`@akhxl` 最初对这项存在已久的技术突然走红表示困惑。`@gabriel_syme` 解释称，之前的实现并未产生实用的模型，而基于 **Mistral** 的 Mixtral 已展现出实际应用价值。
- **Finetuning Large Language Models (LLMs)**：在 `@akhxl` 和 `@gabriel_syme` 的对话中，对 Finetuning 所需的数据量进行了澄清。`@gabriel_syme` 指出，由于基础模型质量高且预训练数据充足，近期的进展并不需要大量数据即可 Finetune 出优秀的模型。随后 `@cyborgdream` 预测 Mixtral 在 Finetuning 后有望表现出与 **GPT-4** 相当的性能。
- **开源 LLMs 的使用**：`@.plot` 和 `@wlrd` 就开源 LLMs 的获取和实现进行了交流。`@wlrd` 指出模型权重是开源的，可以从 *Hugging Face* 获取，并提供了 **OpenHermes 2.5 - Mistral 7B** 模型的示例链接。
- **GPT-3.5 Turbo 讨论**：针对 **GPT-3.5 Turbo** 规格展开了细致讨论，主要参与者包括 `@cyborgdream`、`@agcobra1` 和 `@n8programs`。讨论范围涵盖了其与更小及更大模型的性能对比，`@cyborgdream` 根据泄露的 **G3PO** 信息猜测该模型可能是 20B 模型，并预测其很快会开源发布。
- **ChatGPT 的 Inference 优化**：用户 `@zohad_sikder` 发起了关于 ChatGPT 更快 Inference 潜在优化方案的讨论。`@teknium`、`@bjoernp`、`@eas2535` 和 `@skadeskoten` 的推测包括：不太可能使用 Quantization，但可能采用了战略性的 Batching 以及针对常见问题的 Caching。针对 ChatGPT 的快速响应时间，`@zohad_sikder` 假设其由于庞大的用户群而拥有强大的 Caching 机制。


### ▷ #[memes](https://discord.com/channels/1053877538025386074/1166105758635655270/) (10 messages🔥): 
        
- **Meme 分享与回应**：该频道的用户（即 `@teknium` 和 `@Error.PDF`）频繁分享表情符号和 Meme 回应。值得注意的包括 **"Y not both"** 和 **<:pepeshy:1151280286345207819>** 表情。
- **对特定演讲者的渴望**：`@teknium` 表达了希望 **Yann** 和 **Karpathy** 等人进行演讲的愿望，引发了用户间的讨论。
- **人物评价**：`@teknium` 对某位未具名人士发表了看法，将其定性为 **"crazy psycho about x risk"**（对生存风险极度狂热的疯子）。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- 一场围绕版权内容和 AI 背景下的 **AI 偏见、道德和公平使用 (fair use)** 话题的持续讨论。对话深入探讨了 Large Language Models (LLMs) 中的偏见和真理哲学等问题，并对 Google 的新 AI Gemini 以及 Mistral Instruct 和 gpt4all 等替代 AI 技术方案进行了推测。
- 成员参与了关于 **GPT-4 的各种技术讨论**，涉及“动态限制 (Dynamic Limits)”、等待名单时长、前缀提示词 (prefix prompt) 探索、ChatGPT 的性能和访问问题，以及不同设备间功能的差异。人们对 GPT-5 的开发和明年 GPT Store 的开业进行了推测。
- **GPT 使用**中的问题和改进一直是热门话题，用户对 GPT 的对话总结、GPT Builder 中缺失的功能，以及缺乏允许对 AI 回复进行行内编辑 (Inline editing) 或修剪 (trim) 的功能表示不满。同时，还进行了关于获取 ChatGPT 插件开发者权限、澄清 OpenAI 的服务条款 (Terms of Service) 以及对自定义 GPT (custom GPTs) 综合指南需求的讨论。
- 关于**使用 GPT 进行游戏开发**和聊天机器人性能的对话表明，人们对 AI 技术的潜在应用有着浓厚的兴趣。API key 生成过程中的验证码 (captcha) 问题、搜索特定对话以及感知的 GPT 输出变化，引发了关于 AI 系统当前局限性和改进领域的辩论。
- 该社区的一个显著话题是 **Prompt Engineering**，深入研究了情感语言的使用以及在 PPM 中人格化的实现。社区还深入探讨了文本分块 (text chunking)、嵌入 (embeddings) 和详细提示词创建等问题。分享的一系列针对 GPT-4、DALL-E 和浏览器工具的详细提示词指南和命令协议，反映了提升 AI 模型利用率的协作努力。


**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (123 条消息🔥🔥): 
        
- **关于 AI 偏见和道德的讨论**：用户 `@whynot66k20ni`、`@light.grey.labs`、`@solbus`、`@lhc1921` 就 Large Language Models (LLMs) 固有的偏见性质、真理哲学以及 AI 潜在的自我意识进行了深度对话。 
- **ChatGPT 的 AI 伦理和“公平使用”**：`@.dooz`、`@lhc1921`、`@light.grey.labs` 讨论了版权内容和 AI 背景下的“公平使用”。`.dooz` 建议对版权内容的转换性使用可以构成公平使用。
- **关于 OpenAI GPT Store 发布的讨论**：`@lumirix` 分享了 GPT 创建者收到的一封邮件摘要，承诺在明年年初发布 GPT Store，并为 ChatGPT 提供其他重大更新。
- **OpenAI ChatGPT 的替代方案**：`@mysticmarks1` 为寻找替代聊天 AI 的 `@sneakobrah` 推荐了 Mistral Instruct 和 gpt4all 作为 OpenAI ChatGPT 的替代或补充。
- **关于 Google AI Gemini 的讨论**：`@prajwal_345` 分享了一个关于 Google Gemini AI 的[链接](https://analyticsindiamag.com/google-fools-everyone-with-gemini/)，暗示其是在压力下发布的，且在多项基准测试中表现优于 OpenAI 的 GPT-4。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (112 条消息🔥🔥): 
        
- **GPT-4 动态限制与等待名单讨论**：`@dr.youvi.avant` 询问了新的 GPT-4 “动态限制（Dynamic-Limits）”。`@stefatorus` 提到解锁旧版 GPT 版本是可能的，但可能很昂贵，他的使用费用每月约为 200 欧元。`@killer.5643` 询问了 GPT-4 等待名单的持续时间，`@7877` 提到了即将推出的 GPT Store，`@jonathan_91672` 分享说他等待了大约一个月才收到邀请。

- **GPT-4 Prefix Prompt 探索**：`@israel_a4` 分享了来自 Wes Roth 的 YouTube 技巧，该技巧允许用户通过使用特定代码查看 GPT-4 的 Prefix 或 Secret Prompt。当被问及是否有防止此类行为的补丁时，`@elektronisade` 表示目前没有此类计划，因为这是模型固有的运行机制。

- **ChatGPT 性能与访问问题**：多位用户报告了 ChatGPT 的问题，`@mrcrack_` 提到了持续的网络错误以及 ADA 图像读取功能失效。`@zz99mz` 提到域名完全无法加载的问题。`@pruo` 表示他们的自定义指令（custom instructions）出现问题，`@mrcrack_` 也对动态限制表示不满。

- **不同设备的功能差异**：`@gd2x` 询问为何 Android 版 ChatGPT 缺少语音功能，`@elektronisade` 将其归因于广告拦截器（adblocker）的使用。用户还讨论了 Android 和 iOS 版本之间可用功能的差异。

- **GPT-3 扩展与 GPT Store 预测**：`@youraveragedev` 推测了 GPT-5 的开发情况，但 `@clockrelativity2003` 否认其目前正在训练中。`@lugui` 组织了关于 GPT Store 在新年开业的讨论。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (158 条消息🔥🔥): 
        
- **GPT 的问题与改进**：用户 `@stealth2077` 对 GPT 即使在收到明确不准这样做的指令后，仍以总结性段落结束对话表示担忧。`@stealth2077` 还提议为 AI 回复增加行内编辑或裁剪功能，以便更轻松地控制生成的对话，`@ath0rus` 也加入了这一话题。`@stealth2077` 对 GPT 使用次数从 50 次减少到 40 次，以及取消为自定义 GPT 测试预留的额外 10 次使用额度表示不满。
- **GPT Builder 限制**：`@amanshrestha` 在使用 GPT Builder 时遇到问题，这似乎源于 Python 环境。`@stealth2077` 也对在对话中途更改自定义指令的限制表示沮丧，并强调需要更好的功能来编辑对话的上下文（context）。
- **ChatGPT Plugins**：`@keebs1995` 询问如何获取 ChatGPT 插件的开发者权限，以便为他们的行业构建计算器应用。`@elektronisade` 告知插件正在逐步淘汰，并建议改用自定义 GPTs。
- **服务条款 (ToS) 澄清**：用户 `@eric.turnr` 寻求对 OpenAI ToS 中提到的“自动或通过编程方式提取数据或输出（定义见下文）”部分的详细解释。`@lumirix` 澄清说，“输出（Output）”在 ToS 的内容部分有明确定义。
- **性能问题与增强**：包括 `@Shunrai` 和 `@lucianah` 在内的几位用户报告了 GPT 的延迟和网络错误问题。`@Rock` 征求关于自定义 GPTs 运作机制的详尽指南，`@strange073` 寻求关于如何通过捐赠 1 美元来获取 GPT-4 API 访问权限的澄清。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (25 条消息🔥): 
        
- **将 GPT 用于游戏开发**：`@cerebrocortex` 分享了他们开发类《文明》游戏的经验，对 ChatGPT 处理库存管理等任务的出色表现表示惊讶。他们请求大家对他们的游戏提供反馈。
- **ChatGPT Plus 邀请**：`@pietman` 和 `@mlgpro0225` 提到有人收到了加入 ChatGPT Plus 的邀请，表明等待名单可能正在推进。
- **调试 GPT Builder**：`@cerebrocortex` 询问如何更新自定义 GPT 的指令，`@Capcon` 建议将更改保存到草稿并使用“更新（update）”按钮发布更改。
- **在 ChatGPT 中搜索特定对话**：`@q16.kr` 询问是否可以搜索与 ChatGPT 进行的特定对话，`@pietman` 回复称该功能目前尚不可用。
- **ChatGPT API Key 生成问题**：`@realspacekangaroo` 报告了在尝试生成新 API Key 时遇到的验证码问题，认为其难度过大，导致他们无法生成新的 API Key。
- **GPT 输出的变化**：`@victronwolfson` 注意到 `gpt-4-1106-preview` 的输出在过去一周内发生了变化。

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (36 messages🔥): 
        
- **在 Prompt 中使用情感**：`@eskcanta` 讨论了在关于一篇名为 "ai emotional prompt" 论文的对话中，情感化语言在 Prompt 中的使用及其对 ChatGPT 的影响。他们指出在论文中找不到用于测试的具体 Prompt，因此无法复现结果。
- **在 PPM 中引入人格**：`@eligump` 和 `@mysticmarks1` 就开发具有两种人格的 PPM (persistent personality mode) 进行了对话。`@mysticmarks1` 分享了一个[链接](https://chat.openai.com/share/ba013894-5bac-43f4-9d6e-3310c5d9e1bc)来演示如何在对话中实现结巴和傻气等行为。
- **创建详细的 Prompt**：`@cybector` 分享了一个针对 Python 编程语言的详细 Prompt 草案，并邀请其他用户提供反馈和改进建议。
- **文本 Chunking 和 Embeddings 的问题**：由于密度实验的成本问题，`@merpnderp` 请求关于文本 Chunking 和 Embeddings 策略的资源或讨论。`@eskcanta` 建议尝试使用 ChatGPT 网页界面来寻找潜在的成本节约方案。`@m0bsta` 表示由于消息限制，这种方法存在困难。
- **GPT-4 的 Prompt 和指南**：`@cat.hemlock` 以 Markdown 形式分享了一系列针对 GPT-4、DALL-E 和浏览器工具的详细 Prompt 指南和命令协议。这包括基础信息、使用的工具以及指导 AI 模型使用的各种策略。她还展示了一个典型详细 Prompt 的 JSON 格式。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (36 messages🔥): 
        
- `eskcanta` 讨论了 [EmotionPrompt 在语言模型中的应用](https://discord.com/channels/974519864045756446/1182753259081957407)，由于参考论文中缺乏清晰的 Prompt 示例，对其实现和有效性提出了质疑。
- `madame_architect` 从现有文档中强调了 EmotionPrompt 的部分实现。他们提供了情感刺激的示例，并提到添加这些刺激的基础 Prompt 和模板也出现在配套文档中。
- 在一系列消息中，`eligump` 和 `mysticmarks1` 讨论了 **Private Playground Models (PPMs)** 的创建和操作，特别是如何融入 Roleplay 和特定的语言风格。
- 一位名为 `mattiacastioni` 的用户在链接的对话线程中寻求帮助。该请求的具体性质未作进一步讨论。
- `cybector` 分享了一个围绕 Python 编程语言讨论与 ChatGPT 交互的模板，特别指示模型从 Python 官方文档中获取信息。
- `merpnderp` 征求有关文本 Chunking 和 Embeddings 策略的推荐资源，旨在降低生产成本。`eskcanta` 建议与 ChatGPT 讨论成本节约策略。
- 最后，`cat.hemlock` 分享了在 OpenAI 的 ChatGPT 中使用 **Markdown、DALL-E、Python 和浏览器工具** 的指南，以及如何构建“默认 Prompt”的示例。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- 围绕由 `@caseus_` 发起的 **Mixtral** 集成进行了活跃的讨论和开发，重点关注 sample packing、sharding 以及解决各种技术问题。强调了 `mixtral-multipack` 分支的创建，并附带了 [相关的 GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/compare/main...mixtral_sharded)。
- `@noobmaster29` 在 Hugging Face 上发布了新数据集 `Verified-Camel-zh`，并提供了 [数据集的直接访问链接](https://huggingface.co/datasets/noobmaster29/Verified-Camel-zh)。
- 一场对话识别了模型错误报告中的常见问题并提出了解决方案，例如更改 `model_type` 和禁用 `is_mistral_derived_model`。 
- 分享并探索了各种科学论文处理库，例如 [allenai/papermage](https://github.com/allenai/papermage)、[axa-group/Parsr](https://github.com/axa-group/Parsr) 和 [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) 库，用于将 PDF、文档和图像转换为结构化数据。
- RLHF 频道中关于即将推出的用于数据集创建的 DPO (Direct Preference Optimization) 策略的对话；具体而言，需要两个不同的 DPO 数据集来处理“未对齐 (unalignment)”并提供“高质量回答 (quality answers)”。 
- 其他杂项对话包括与 axolotl 代表的播客、AI 项目、编码中的 token，以及一段名为 *The Insane Biology of: The Axolotl* 的 YouTube 视频。

**OpenAccess AI Collective (axolotl) 频道摘要**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (25 条消息🔥): 
        
- **Mixtral 集成与开发**: 
    - `@caseus_` 分享了 **Mixtral** 与 axolotl 集成的更新，包括添加了 `mixtral-multipack` 分支以及合并了带有 multipack 的 Mixtral MoE 微调。 
    - 要使用更新后的功能，用户必须从 git main 安装最新版本的 `transformers`。
    - 为了进一步开发，`@caseus_` 分享了由 `@214834317774422028` 开发的进行中分支链接（[GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/compare/main...mixtral_sharded)）。

- **新数据集发布**:
    - `@noobmaster29` 宣布在 Hugging Face 上发布了一个名为 `Verified-Camel-zh` 的新数据集（[数据集链接](https://huggingface.co/datasets/noobmaster29/Verified-Camel-zh)）。

- **杂项讨论**: 
    - `@swyxio` 重点介绍了一个由 axolotl 代表参加的播客，并分享了几个与 AI 相关的资源和项目链接。  
    - 针对编码中 token 的使用和命名进行了对话，特别是 start 和 stop token 的使用。
    - `@noobmaster29` 分享了一段名为 *The Insane Biology of: The Axolotl* 的 YouTube 视频（[视频链接](https://youtu.be/bFkIG9S2Mmg?si=bwXWKBM8fI-sPT-R)）。

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (170 messages🔥🔥): 
        
- **Mixtral 样本打包 (Sample Packing)**：`@caseus_` 一直致力于为 Mixtral 实现样本打包，并创建了一个 `mixtral-multipack` 分支。有报告称初始 Loss 较高但随后会下降，表明这种方法的潜在有效性。`@faldore` 一直在使用 `mixtral-multipack` 分支，并报告运行稳定且 Loss 率持续下降。

- **修复与变通方法**：用户遇到了一些错误，并提出了相应的变通方法和修复建议。具体而言，禁用 `is_mistral_derived_model: true` 并将 `model_type` 更改为 `AutoTokenizerForCausalLM` 似乎解决了一些问题。此外，`@casper_ai` 建议如果使用单 GPU，请移除 DeepSpeed。

- **VRAM 需求**：讨论了关于 VRAM 使用的问题，`@caseus_` 建议了减少 VRAM 使用的策略，例如冻结模型的早期层。提到了在 2xA6000 和 4xA100 GPU 上运行 Mixtral，并希望在 4 到 8xA6000 上实现全量微调 (Full Finetuning)。`@casper_ai` 创建了一个包含部分分片 (Sharding) 功能的分支以优化 VRAM 使用，但目前仍在开发中。

- **模型错误报告**：`@ludis___` 报告了运行 Mixtral 时的 `RuntimeError`，内容为 "output tensor must have the same type as input tensor"。该问题通过移除某些配置参数得到了解决。

- **LoRA 和 qLoRA 使用**：在 4xA100 和 A40 等 GPU 配置上成功运行了使用 qLoRA 的 Mixtral。然而，尝试使用 LoRA 运行时出现了与 `bnb` 包相关的错误。

链接：

- [mixtral-multipack 的 GitHub 分支](https://github.com/OpenAccess-AI-Collective/axolotl/tree/mixtral_mltipack)
- [Mixtral 优化的 GitHub Issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/930)
- [Mixtral 节省内存的 GitHub Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/934)
- [Mixtral 分片的 GitHub 分支](https://github.com/OpenAccess-AI-Collective/axolotl/tree/mixtral_sharded)


### ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (3 messages): 
        
- **潜在招聘讨论**：`@faldore` 表达了一种观点，认为如果他们被录用，某些情况可能会得到改善。
- **对 Elon Musk 雇佣的看法**：作为回应，`@nruaif` 建议在 Elon Musk 手下工作可能并不理想。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (5 messages): 
        
- **合并 Qlora Chat Mixtral 问题**：`@matts9903` 报告了在尝试使用 Axolotl 工具合并 `mixtral` 模型时遇到的错误。问题在于 `repo id` 的验证错误：```huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: './qlora-out'.```
    
- `@caseus_` 建议使用 qlora-out 目录的绝对路径，但该建议未能解决问题。

- `@caseus_` 随后分享了模型合并的最新更改 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/commit/1d21aa6b0ac0e1de832b5d57c82da34220346046)，并要求提供堆栈跟踪 (Stack trace) 以进行进一步排查。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (4 messages): 
        
- **PaperMage 库**：`@noobmaster29` 分享了 [allenai/papermage](https://github.com/allenai/papermage) 库的 GitHub 链接，建议值得一试。该库支持针对科学论文的 NLP 和 CV 研究。
- **Parsr 库**：`@visuallyadequate` 目前正在尝试 [axa-group/Parsr](https://github.com/axa-group/Parsr) 库，该库可将 PDF、文档和图像转换为丰富的结构化数据。
- **Tika 库**：`@visuallyadequate` 提到曾使用过 Tika 库，称其提供了目前为止最好的解决方案，但他们尚未测试 PaperMage。
- **Unstructured 库**：`@joshuasundance` 分享了 [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) GitHub 库的链接，该库为构建自定义预处理流水线 (Preprocessing pipelines) 提供开源库和 API。

### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (5 messages): 
        
- **DPO Completion**：`@caseus_` 提到在被 **Mixtral** 的相关工作分散注意力后，需要完成 DPO (Data Programming Override)。
- **Unalignment 和 Quality Answers DPO 数据集**：`@faldore` 讨论了需要两个 DPO 数据集的想法，一个用于 **"unalignment"**（去对齐），另一个用于提供 **"quality answers"**（高质量回答）。
- **Rejected 字段查询与对比**：`@nruaif` 建议针对 rejected 字段询问 **Llama 2 7B chat**，并将其与 **GPT 4** 进行对比，指出在 90% 的情况下，Llama 2 7B chat 的回答效果更差。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 关于**在 LangChain 中将本地模型与 chat LLMs 结合使用**的广泛讨论，包括 `@_egeres` 关于使用环境变量和对 `LLM` 进行子类化的见解，以及 `@lhc1921` 关于使用像 llama.cpp 这样的后端来处理 constrained grammar（受限语法）的想法。
- 多个成员提出的问题仍未得到解答，包括：
  - `@analyticsrepo` 关于 Google 的 **Gemini 集成**到 LangChain 的进度问题。
  - `@_ashisharya` 寻求关于 *Agent 编码与部署* 的全面资源。
  - `@xstepz` 寻求关于 *在 Kork 包中限制 pandas 函数可用性* 的指导。
  - `@yasuke007` *寻求 AI 开发学习路径的建议*，特别是关于在使用 langchain 配合 React.js 时是否有必要掌握 Python 知识。
  - `@rajib2189` 关于 *在本地运行语言模型的潜在用例* 的咨询。
- 用户 `@reletreby` 发布了 **Askly 12 月版本**公告，现已集成 **OpenAI ChatGPT 3.5** 和来自 HuggingFace 的 **HuggingFaceH4/zephyr-7b-beta**。新功能包括多文件推理、摘要、网页搜索，并要求用户*删除并重新上传旧文件以启用新功能*。更多详情请见 [Askly 博客](https://www.askly.ai/blog/askly-december-release)。

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (72 messages🔥🔥): 
        
- **Google Gemini 集成**：用户 `@analyticsrepo` 询问了将 Google Gemini 集成到 LangChain 的状态，但未收到回复。
- **LangChain 与本地模型**：`@_egeres` 和 `@lhc1921` 深入讨论了在 LangChain 中使用本地模型配合 chat LLMs 的可能性。`@_egeres` 提到可以通过环境变量调整 API 端点并对 `LLM` 进行子类化。`@lhc1921` 建议使用能够处理 constrained grammar 的 llama.cpp 等后端。
- **Agent 编码与部署资源**：`@_ashisharya` 寻求关于 Agent 编码和部署的综合资源，但未收到回复。
- **配合 Pandas 使用 Kork 包**：`@xstepz` 寻求如何限制其 Agent 通过 Kork 包访问 pandas 函数的指导，但未收到回复。
- **AI 开发学习路径**：初级 AI 开发者 `@yasuke007` 询问在使用 langchain 配合 React.js 的 AI 开发过程中是否有必要学习 Python，但未收到回复。
- **本地运行语言模型的用例**：`@rajib2189` 询问了本地运行语言模型的可能用例（如个人助手或边缘侧分析），但未收到回复。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 messages): 
        
- **Askly 12 月版本发布**：用户 `@reletreby` 宣布了 **Askly** 的最新版本，通过集成 **OpenAI ChatGPT 3.5** 和 HuggingFace 的开源模型 **`HuggingFaceH4/zephyr-7b-beta`** 进行了重大升级。新功能包括多文件推理、摘要、网页搜索等。然而，为了使用这些功能，在 2023 年 12 月 1 日或之前上传过文件的用户需要删除旧文件并重新上传。这对于激活新功能至关重要。完整详情已在 [Askly 博客](https://www.askly.ai/blog/askly-december-release)中分享。

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- `@astra1337` 在 **demo 演示**后与其他人的互动，强调了观众对进一步解释的兴趣。此外，`@astra1337` 询问了关于 **Pygmalion AI** 在 **电子游戏 demo** 方面的知名度。
- `@mister_poodle` 询问了关于针对特定任务微调 **Mistral-OpenOrca** 的过程，特别关注于提升其在 **带有 JSON 输出的命名实体识别 (NER) 任务**上的性能。
- 围绕绘图工具的对话，重点提到了 *Whimsical* 和 *Excalidraw*。
    - *Whimsical* 由 `@teknium` 引入并由 `@gabriel_syme` 进行了测试，指出其具有协作功能的倾向。
    - *Excalidraw* 由 `@lightningralf` 建议，他提供了链接 [Excalidraw](https://excalidraw.com/) 并提到存在 *Obsidian* 插件。

**Alignment Lab AI 频道摘要**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (3 条消息): 
        
- **Astra1337 就 Demo 与他人的互动**：用户 `@astra1337` 提到在一些 **demo 演示**之后，有人找他们获取更多信息。
- **关于 Pygmalion AI 的讨论**：`@astra1337` 询问了一位来自 **电子游戏 demo** 的成员是否了解 **Pygmalion AI**，这是一个以创建具有记忆的电子游戏角色而闻名的研究小组。


### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (1 条消息): 
        
- **微调 Mistral-OpenOrca**：`@mister_poodle` 询问了如何使用个人数据集针对特定任务微调 **Mistral-OpenOrca**，表达了改进模型在 **带有 JSON 输出的命名实体识别 (NER) 任务**上表现的意图。在此背景下，`@mister_poodle` 未提供链接或额外信息。


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/) (8 条消息🔥): 
        
- **关于绘图工具的讨论**：`@teknium` 介绍了 *Whimsical* 绘图网站。在尝试后，`@gabriel_syme` 认为它具有协作功能，因为它提示创建工作区。
- **Excalidraw 推荐**：`@lightningralf` 推荐了 *Excalidraw* 作为另一个选择，并链接到了该网站，此外还提到了一个 *Obsidian* 插件。这是他推荐的链接：[Excalidraw](https://excalidraw.com/)。


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

只有一个频道有活动，因此无需汇总...

- **使用 qlora、小 batch 和上下文窗口**：在回答一个查询时，`@eugeneyan` 分享说，24GB 的 GPU 应该可以运行小 batch size 和适当上下文窗口的 qlora（batch 为 2，上下文窗口 512 - 1024）。
- **关于 HumanLoop 的功能查询**：`@jozexotic` 对 HumanLoop 新功能开发缓慢表示担忧，特别是无法访问 OpenAI 以外的模型，并询问是否有人知道这些新增功能是否在该平台的近期议程中。
- **对 chatgpt+ 的不满**：`@slono` 表示由于进展缓慢和反复出现的 stream errors，正在考虑取消他们的 chatgpt+ 订阅。
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

只有一个频道有活动，因此无需汇总...

pradeep1148: https://www.youtube.com/watch?v=YWYNLaWDoNQ
        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 摘要

只有一个频道有活动，因此无需汇总...

.psychickoala: 你们有人见过强制 parallel function calling 的最佳实践吗？