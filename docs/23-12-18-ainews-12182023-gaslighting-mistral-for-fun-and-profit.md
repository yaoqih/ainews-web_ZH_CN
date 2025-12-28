---
companies:
- openai
- anthropic
- google-deepmind
date: '2023-12-19T03:35:50.162274Z'
description: '**OpenAI** Discord 的讨论揭示了对多种语言模型的比较，包括 **GPT-4 Turbo**、**GPT-3.5 Turbo**、**Claude
  2.1**、**Claude Instant 1** 和 **Gemini Pro**，其中 **GPT-4 Turbo** 因其以用户为中心的解释而受到关注。关于
  **GPT-4.5** 的传闻仍未得到证实，在官方公告发布前，怀疑情绪占据主导。用户讨论了响应缓慢和 API 问题等技术挑战，并探索了角色扮演提示词技术以增强模型性能。关于人工智能对学术界和就业影响的伦理担忧也引发了辩论。人们对
  **Dalle 3** 的未来功能和提议的新 GPT 模型进行了推测，同时一个学校项目正在寻求使用 **OpenAI API** 的帮助。社区还触及了 AI 眼镜以及采用
  AI 对就业市场的影响。'
id: df18546a-13b6-4ffa-82d4-57312a153e7c
models:
- gpt-4-turbo
- gpt-3.5-turbo
- claude-2.1
- claude-instant-1
- gemini-pro
- gpt-4.5
- dalle-3
original_slug: ainews-12182023
people:
- sam-altman
title: 2023年12月18日：为了好玩和获利而对 Mistral 进行煤气灯操控 (Gaslighting)
topics:
- prompt-engineering
- api
- model-performance
- ethics
- role-play
- user-experience
- ai-impact-on-jobs
- ai-translation
- technical-issues
type: archival
---

<!-- buttondown-editor-mode: plaintext -->[告诉 Mixtral](https://fxtwitter.com/abacaj/status/1736819789841281372?s=46&t=90xQ8sGy63D2OtiaoGJuww) 它是“由 OpenAI 开发的 ChatGPT”能将 HumanEval 分数提高 6%：

https://fxtwitter.com/abacaj/status/1736819789841281372?s=46&t=90xQ8sGy63D2OtiaoGJuww

这符合 Prompt 角色扮演增强能力的既定模式，但也提醒了我们 HumanEval 作为一个评估指标相当糟糕。

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- `@rrross` 分享了不同语言模型（**GPT-4 Turbo**、**GPT-3.5 Turbo**、**Claude 2.1**、**Claude Instant 1** 和 **Gemini Pro**）的对比。在被要求描述用户引导（onboarding）追踪转变的影响时，GPT-4 Turbo 提供了最以用户为中心的解释。
 
- 关于*传闻中*的 **GPT-4.5** 版本的讨论涉及多位成员，包括 `@feltsteam0` 和 `@jaicraft`。参与者一致同意，在官方声明或明确证据出现之前，继续将其视为不存在。

- 解决了用户遇到的多个技术挑战，例如响应时间慢、未指明的错误、账号被封，以及在 GPT 工具和 ChatGPT Plus 等平台上访问 API 的问题。

- 分享了在 API 讨论中关于 System Prompt 角色扮演模式的经验，建议通过在用户消息字符串中添加提醒或在指令中附加注释来保持第一人称视角。
  
- 对 AI 在学术界和就业市场使用的伦理影响表示担忧。随后引发了关于潜在滥用、剽窃和职位取代的辩论。

- 探讨了 AI 模型中潜在的未来功能实现，特别是 **Dalle 3** 和由 `@7877` *提议的新 GPT 模型*。虽然关于 Dalle 3 功能的对话更具推测性，但关于新 GPT 模型的讨论缺乏确凿的细节。

- `_helium.` 为一个学校项目寻求帮助，该项目旨在利用 **OpenAI API** 开发一个语言翻译网站，但遗憾的是，目前尚未提供具体的建议。

**OpenAI 频道摘要**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (22 messages🔥): 
        
- **不同语言模型的实验**：用户 `@rrross` 分享了一个实验，对比了不同语言模型（**GPT-4 Turbo**、**GPT-3.5 Turbo**、**Claude 2.1**、**Claude Instant 1** 和 **Gemini Pro**）在被要求解释从本地到服务器端用户引导追踪转变的影响时的响应。`@rrross` 观察到 **GPT-4 Turbo** 提供了最以用户为中心的解释。
- **关于 AI 翻译网站的问题**：用户 `_helium.` 为一个涉及使用 **OpenAI API** 开发语言翻译网站的学校项目请求协助。在随后的消息中没有给出具体的回复或建议。
- **AI 眼镜讨论**：用户 `@dunescifye` 和 `@lugui` 简短地交流了对 AI 眼镜的看法。`@lugui` 评论说这项技术在理论上听起来比实践中更好，但没有提供与 AI 眼镜相关的具体挑战或问题。
- **ChatGPT 中 Dalle 3 的潜在功能**：用户 `@satanhashtag` 希望 **Dalle 3** 具有诸如 Midjourney 变体（variation）和可编辑区域等功能，对此 `@kyoei` 回复说这些功能最终可能会被引入。随后出现了关于可能的 **Dalle 3.5** 版本的玩笑。
- **新 GPT 模型提案**：用户 `@7877` 提到正在开发一个新的 GPT 模型，并提议发送链接给其他人（`@mawzoon` 和 `.pythagoras`）尝试，以便在公开发布前提供反馈。然而，在随后的消息中并未分享该新 GPT 模型的实际链接或进一步细节。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (750 messages🔥🔥🔥): 
        
- **关于 GPT-4.5 的讨论**：成员 `@feltsteam0`、`@jaicraft` 等人讨论了据传存在的 **GPT-4.5**，大多数人表示怀疑，因为有报告显示它并不存在。一位用户引用了 *Joe Rogan* 和 *Sam Altman* 之间的一次对话，其中提到了未来的 GPT-4.5。然而，大多数参与者一致认为，在官方声明或明确证据出现之前，最好认为 GPT-4.5 并不存在。

- **对 AI 输入限制和整体性能的担忧**：用户 `@jaicraft` 表达了对使用 GPT 开发模型时输入限制的沮丧，而 `@picturesonpictures` 对失败 Prompt 的计费表示不满。

- **关于 AI 对就业影响的讨论**：用户辩论了 AI 对就业市场的潜在影响。一些人认为 AI 提高了工作场所的生产力，而另一些人则对 AI 取代人类工作的潜力表示担忧。有人建议在教育和商业中实施 AI 时采取负责任且符合伦理的做法。

- **将 AI 用于 Web 开发和学术**：`@bloodgore` 分享了关于他的学生不当使用 ChatGPT 撰写学术论文的讨论。其他人建议了不同的方法来检测 AI 生成的内容。在讨论的其他地方，`@mysticmarks1` 谈到了 AI 在创建 Web 解决方案方面的未来潜力，`@msirene` 询问使用 ChatGPT 是否等同于剽窃。

- **信用卡支付和监管问题**：用户 `@msirene` 遇到了公司卡在多次用于为员工创建账户后被拒绝的问题。`@elektronisade` 分享了 OpenAI 关于信用卡使用限制的政策。此外，针对 `@bloodgore` 关于学生滥用工具的陈述，还引发了关于 OpenAI 是否应该成为“仅限付费”服务以遏制未成年用户滥用的辩论。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (90 messages🔥🔥): 
        
- **响应缓慢和错误**：包括 `@scrambler803`、`@mesteviet` 和 `@bittychills` 在内的几位用户报告了在使用 GPT 工具时响应缓慢和未指明的错误。`@scrambler803` 建议该问题可能与正在进行的对话长度有关。`@healer9071` 尝试排查该问题。`@keith_15641` 也报告了 GPT-4 的响应缓慢和错误。

- **账户和 API 问题**：`@dian2024`、`@mildare`、`@pikapikapu4578` 和 `@whitchurch` 都报告了他们的账户或 API 访问问题。问题范围从账户被封禁到 API 配额挑战。`@millionwords` 报告了一个交易问题：购买 ChatGPT Plus 订阅已扣款，但订阅未在 App 或网站上反映。

- **自定义 GPTs 和输出的问题**：`@unfor_tuna_te` 报告了关于写实人脸生成的问题，`@jobydorr` 报告了从上传的 PDF 中检索内容的问题。`@arthurchance` 遇到了旨在链接到自定义 GPT 的二维码问题。

- **其他技术问题**：`@drpossum`、`@jhwarehouse`、`@ashtonwin`、`@couchlannister` 和 `@explosiveburrito` 提到收到了“unusual activity（异常活动）”错误消息。`@debugpvp` 寻求关于绕过 Token 限制问题的指导。`@aesthetic_person_123` 和 `@andrewwallo` 面临网络错误，主要发生在长对话期间。

- **ChatGPT 4 的能力**：`@jah777` 和 `@andrewwallo` 之间就 ChatGPT 4 的功能进行了对话，双方都认同其在速度、准确性和知识方面优于免费版本。

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (22 messages🔥): 
        
- **GPT-4 Access to the Internet**: `@karajan_` 询问 GPT-4 是否可以访问互联网。成员们没有直接回答这个问题。
- **Finding Model ID**: 用户 `@lasche` 询问如何查找他们的 GPT Model ID。消息中没有任何回复。
- **Connecting Zapier to Custom GPT Actions via Webhooks**: `@db_looper` 询问是否有人成功通过 Actions 将 Custom GPT 数据发送到 Webhook，并讨论了尝试使用 make webhook 代替 Zapier 时遇到的错误。该查询未得到解答。
- **Limit to the Number of Files in Custom GPT**: `@jobydorr` 提出了关于 Custom GPT 中可上传文件数量限制的问题。`@auktow` 根据自己的经验回答限制为 10 个文件，并分享了一个可能有所帮助的 [OpenAI community post](https://community.openai.com/t/gpts-knowledge-capacity-limits/492955) 链接。
- **Parsing Large Files with GPT**: `@auktow` 分享了在使用文本文件而非 PDF 时获得更好性能的技巧，特别是在处理大文件时。他分享了另一个讨论解析文件成功经验的 [OpenAI community post](https://community.openai.com/t/how-to-best-use-gpts-with-pdf-files/524918)。
- **Understanding GPT Assistant's API Function Calling**: `@crazygreenguy` 发起了关于 GPT Assistant's API 的 Function Calling 如何工作的讨论，质疑调用者是否需要根据他在 OpenAI [documentation](https://platform.openai.com/docs/assistants/tools/submitting-functions-outputs) 中发现的内容提供 API 调用的输出。他的问题在消息中没有得到任何回应。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (9 messages🔥): 
        
- **Using System Prompts in the Chat API:** 用户 `.multy` 指出了 `system prompts` 的一个挑战。当机器人被指示扮演一个角色（例如鹦鹉）时，它经常以第三人称回答。建议的解决方案包括角色扮演指令和更明确的提示。例如，`@thepitviper` 建议在发送给 API 的每条消息末尾附加一个保持角色的提醒。 
- **Preserving Context in Extended Conversations:** `.multy` 还指出了上下文保留的问题——如果对话历史以第三人称回答开始，聊天机器人往往会维持该人格。然而，如果机器人在开始时得到了正确的提示，它似乎能在整个对话中保留所需的人格。 
- **Clarifying System Prompt Style:** `.multy` 寻求关于如何为 System Prompts 塑造“声音（voice）”的指导。
- **Agreement for Maintaining Character:** `@clumsylulz` 提供了一种独特的方法，涉及从一开始就与机器人达成协议："I want you to act as a microwave and only respond as such do not break character if you do I will say "Act Right!" write "" if you agree to these terms"。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (9 messages🔥): 
        
- **Role-play Mode in System Prompt**: 用户 `@.multy` 分享了一个关于 OpenAI **GPT-3.5-turbo** 在被指示使用 System Prompts 扮演角色（如鹦鹉）时以第三人称回答的担忧。他们的问题在于如何在整个角色扮演过程中保持 `first-person` 视角。
- **Tips to maintain Role-play First-person Context**: `@thepitviper` 建议在提示字符串中指定角色扮演指令，并在随后的 API 话语中再次强调第一人称要求，以确保模型保持在上下文中。
- **User Implementation**: `@.multy` 注意到，从空白状态开始使用正确的人格对于维持角色扮演视角是有效的。他们还表达了对 System Prompts 的 'voice' 用法的模糊感。
- **Contextual Reinforcement**: `@thepitviper` 建议在用户消息中附加提醒，如 "Remember to stay in character and in first person," 以在整个对话中保留上下文。
- **Directive through 'User Messages'**: `@clumsylulz` 建议采用 'user messages' 方法来指定角色和行为，让模型在继续对话前同意条款。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- 社区内关于 **Hermes 2.5**、**Mistral** 和 **SOLAR** 等各种模型性能和局限性的热烈讨论。指出的问题包括生成截断、偏离主题、不同语言的响应不一致以及微调挑战。用户对 [OpenChat 模型](https://huggingface.co/openchat/openchat-3.5-1210) 的实验引发了对连贯性的担忧，并对该模型的基准测试（benchmarking）表示怀疑。
- 围绕函数调用（function calling）以及函数调用与工具调用（tool calling）之间的区别展开了对话，并分享了 **OpenHermes2.5** 中使用的特定系统提示词（system prompts）。
- 对 **GPT-4** 性能的期待与推测，认为该模型表现不佳。社区成员推测了可能的原因，如系统提示词、微调、推理速度和模型倾向（如简短回答或无法提供完整的代码块）。
- 探索了评估工具和污染问题，重点关注了一个[实用性评估工具](https://fxtwitter.com/SeonghyeonYe/status/1682209670302408705)，并对 **OpenHermes2.5** 和 **SOLAR** 模型中的数据污染表示担忧。
- 社区成员探索并提供了微调 LLM 的建议，并涉及成本担忧、技术要求和潜在平台（如 Colab、Kaggle、RunPod）。此外，还分享了一个用于 LoRa 微调的 [GitHub 示例](https://github.com/ml-explore/mlx-examples/tree/main/lora)。
- 讨论围绕为代码迁移目的微调模型的可行性，以及基于消息历史创建搜索查询。
- 关于 Amazon **Titan** 嵌入模型分词器（tokenizer）可用性的咨询，引出了创建自定义分词器的建议，并分享了一个包含潜在细节的 [GitHub 仓库](https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch)。
- 传播了一些有趣的链接，包括一篇 [Twitter 帖子](https://x.com/_akhaliq/status/1736582086494924918?s=20)、一篇关于大模型改进的 [arXiv 论文](https://arxiv.org/abs/2312.10003)、[MindLLM 1.3B Huggingface 模型](https://huggingface.co/bit-dny/MindLLM)、一篇关于 Mistral 7B 优化的[博客文章](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized)、关于“100倍加速全栈 Transformer 推理优化”的[文章](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c)和 [Youtube 内容](https://www.youtube.com/watch?v=Y0lwmimnAbk)，以及关于领域特定语言（DSL）与代码的对话。
- 用户在闲聊频道表达了对 **Bard** AI 聊天机器人实现方式的沮丧，用户对该机器人的回答表示不满。

**Nous Research AI 频道总结**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (2 条消息): 
        
- **Bard 的实现问题**：用户 `@euclaise` 表达了对 AI 聊天机器人 **Bard** 的挫败感，最初表示：“*Bard 给了我一个愚蠢的实现，但至少它是一个实现*”。不久后，用户 `@euclaise` 进一步表达了对该 AI 的不满，补充道：“*算了，Bard 也滚蛋吧*”。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (25 messages🔥): 
        
- **模型改进论文**：`@ldj` 分享了一个有趣的 [Twitter 帖子](https://x.com/_akhaliq/status/1736582086494924918?s=20)，但未提供太多背景；而 `@atgctg` 提供了一篇 [arXiv 论文](https://arxiv.org/abs/2312.10003) 链接，讨论了大型模型的改进，并指出最大模型相对于预训练基座（pre-trained base）的改进微乎其微。`@atgctg` 和 `@giftedgummybee` 就这些改进对中小型模型的影响进行了简短辩论。
- **MindLLM 1.3B 模型**：`@pizza_joe` 链接了 **MindLLM 1.3B 模型** 的 [Huggingface 页面](https://huggingface.co/bit-dny/MindLLM)，该模型由北京海量语言信息处理与云计算应用工程技术研究中心和北京理工大学东南信息技术研究院开发。
- **关于代码和 DSL 的讨论**：`@gabriel_syme` 建议使用领域特定语言（DSL）作为代码的替代方案，并强调在编译失败时将 DSL 与语言交织的重要性。根据 `@gabriel_syme` 的说法，这对 Agent 尤为关键。
- **“100倍加速全栈 Transformer 推理优化”链接**：`@atgctg` 发布了一篇题为 [“迈向 100 倍加速：全栈 Transformer 推理优化”](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c) 的文章链接，以及该数据集的 [Youtube 背景介绍](https://www.youtube.com/watch?v=Y0lwmimnAbk)。
- **OpenPipe 的 Mistral 7B 微调优化**：`@metaldragon01` 分享了来自 OpenPipe 的一篇 [博客文章](https://openpipe.ai/blog/mistral-7b-fine-tune-optimized)，介绍了关于 Mistral 7B 的优化，这已为用户节省了超过 200 万美元的推理成本。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (421 messages🔥🔥🔥): 
        
- **模型性能与局限性**：讨论涉及 **Hermes 2.5**、**Mistral** 和 **SOLAR** 等各种模型的性能和局限性。例如，`@gitmo joe` 表示 **Hermes 2.5** 表现不差，但会截断生成内容；`@teknium` 征求社区对 **SOLAR** 的意见。`@weyaxi` 询问了 **OpenHermes-2.5-Mixtral** 的情况，引起了社区褒贬不一的反应。此外，对话还揭示了关于微调、System Prompts 以及模型偏离主题或使用不同语言回答等问题的局限性和担忧。
- **Function Calling**：围绕 Function Calling 展开了对话，`@realsedlyf` 提供了用于 OpenHermes 2.5 中 Function Calling 的详细 System Prompt。`@gitmo joe` 随后询问了 Function Calling 与 Tool Calling 之间的区别。
- **OpenChat 模型**：`@tsunemoto` 和 `@n8programs` 测试了 [OpenChat 模型](https://huggingface.co/openchat/openchat-3.5-1210)，并遇到了涉及 Tokenizer 和模型缺乏连贯性的一些问题。社区部分成员对该模型声称达到 GPT-3.5 水平表示怀疑，认为这可能是由于数据处理任务而非内在的推理能力。
- **关于 GPT-4 性能与微调的讨论**：大家普遍认为 GPT-4 的表现似乎低于预期。参与者讨论了可能的原因，许多人指向 System Prompts、微调和推理速度方面的问题。一些成员指出 GPT-4 模型倾向于简短回答，或在某些提示下避免提供完整的代码块。
- **讨论评估与污染**：参与者讨论了评估工具和污染问题，`@tokenbender` 强调了一个新的综合 [评估工具](https://fxtwitter.com/SeonghyeonYe/status/1682209670302408705)，该工具测试实用性及其他现实世界的指标，如无害性、事实性、理解力等。`@nonameusr` 分享了对数据污染测试的担忧，引用了 OpenHermes 2.5 数据集和 SOLAR 模型的问题。

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (36 messages🔥): 
        
- **微调 LLM**：`@leuyann` 正在为他们的经济学硕士论文寻求微调 LLM 的指导。他们考虑微调 7B 模型，并好奇是否能在 16GB 内存的 M1 MacBook Pro 上本地完成。`@night_w0lf` 建议尝试 Colab、Kaggle 或付费云服务等平台，并可能使用 Apple 发布的全新 MLX 库。`@.beowulfbr` 还建议将 RunPod 作为一个相对便宜的选择。`@atgctg` 在 [GitHub](https://github.com/ml-explore/mlx-examples/tree/main/lora) 上提供了一个 LoRA 微调的示例。

- **微调 LLM 的成本和可行性**：讨论围绕微调大型模型的成本和技术要求展开。`@.benxh` 提到了在 16GB M1 上使用 MLX 的问题，`@leuyann` 指出这些问题可能很快会得到解决。

- **用于代码迁移的微调模型**：`@.beowulfbr` 询问微调一个能够协助将代码库从一个框架迁移到另一个框架的模型是否可行，对此 `@night_w0lf` 建议针对此任务测试更大的编程模型。

- **基于消息历史创建搜索查询**：`@pogpunk` 试图构建一个能够根据消息历史创建搜索查询的工具，`@night_w0lf` 建议用几百个示例训练一个较小的模型。

- **Amazon BedRock TITAN Embedding Tokenizer**：`@coco.py` 询问 Amazon Titan embedding 的 Tokenizer 是否可以在某处获得。`@night_w0lf` 建议从 Hugging Face 的多语言文本嵌入模型 (HF MTEB) 创建自己的 Tokenizer，而 `@_evelynm` 分享了一个 [GitHub 链接](https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch)，其中似乎包含关于 Titan embedding 的详细信息。


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- 围绕 **Mistral** 和 **Hermes** 的使用、优化和性能进行了广泛讨论，`@Makya` 强调了 **Hermes 2.5** 带来的提升。有人询问具有更大上下文长度的 **Mistral** 模型以及如何在云端托管 **Mistral 7b**，并分享了资源，如 `@jamiecropley` 推荐的 [GitHub 仓库](https://github.com/PhillipRt/mistral-playground)。
- 用户分享了关于 `mistral-medium` 底层模型的见解、编码词汇表文件的预计可用时间，以及[查看 GPT-3.5/4 编码词汇表文件的链接](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken)，还有可在 [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/tokenizer.json) 上找到的词汇表 JSON 标准。
- 讨论的重点包括解决使用 Docker 运行 Mistral 时的挑战、Docker 与 Ollama 安装的优缺点，以及 Ollama 在微调模型方面的局限性。
- 报告了关于 **Mistral** 和 **Mixtral** 在聊天机器人实现中的理解能力问题。用户分享了提高 Mistral 上下文理解能力的策略以及潜在解决方案，包括使用经过强化的系统提示词 (system prompt) 进行微调。
- 用户分享了各种机器学习、编程和技术相关的资源与产品，例如现在支持 Mistral AI 的 [MindMac](https://mindmac.app/) 应用、[La Plateforme 的 Golang 客户端](https://github.com/robertjkeck2/mistral-go)，以及用于运行性能测试的库，如 [opencompass](https://github.com/open-compass/opencompass)、[llm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 和 [light-eval](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/light-eval)。
- 有人询问并讨论了潜在的技术问题，如将 Mac Mini 连接到 2007 年的 iMac 显示器，并分享了辅助资源，如[讨论帖](https://discussions.apple.com/thread/7139991?sortBy=best)和[关于简化显示器连接的文章](https://recorder.easeus.com/screen-recording-resource/use-imac-as-monitor-for-pc.html)。
- 关于 **La Plateforme** 的讨论涉及 Mistral 模型相关错误的排查、对模型审查的担忧、服务器错误和费用问题，以及 Token 计数策略的交流。还解答了关于 Mistral 速率限制 (rate limit) 的查询，据分享，所有端点的速率限制为每分钟 2M Token 和每月 200M Token。


**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (104 条消息🔥🔥): 
        
- **Mistral 和 Hermes 的使用**：用户讨论了 **Mistral** 在本地和 API 实现中的使用与优化。此外，`@Makya` 强调了 **Hermes 2.5** 相比 **Hermes 2** 的性能提升。
- **MLX 和 Llama.cpp 讨论**：`@sublimatorniq` 发起了关于使用 Apple **MLX** 运行 **Mixtral** 潜在优势的对话。`@daain` 指出了由于 **MoE**（混合专家）架构可能导致的性能问题。
- **Mistral 托管与 API**：`@satyajitsato` 询问了关于如何在云端托管 **Mistral 7b** 并为其封装 API 的资源。`@jamiecropley` 分享了一个 [GitHub 仓库链接](https://github.com/PhillipRt/mistral-playground) 作为可能的解决方案，尽管他们在运行中遇到了一些问题。
- **上下文长度讨论**：`@eawlot3000` 询问是否有上下文长度（context length）超过 32768 tokens 的 **Mistral** 模型。用户分享了关于具有更大上下文长度模型的信息和资源，如 `@Claude` 和 `@GPT4`。
- **职业建议**：`@naz.daq` 询问了如何开始学习机器学习。一些用户推荐的资源包括 [3Blue1Brown 的 YouTube 系列视频](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown) 以及自学线性代数等基础数学课题。


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (7 条消息): 
        
- **Mistral-medium 背后的模型**：用户参与了关于 `mistral-medium` 底层模型的讨论。`@superseethat` 询问了细节，`@sublimatorniq` 分享称它是一个新的原型模型，而 `@tom_lrd` 推测它可能是 4x8x7b。

- **GPT 模型的编码词汇表文件**：`@jakobdylanc` 询问了编码词汇表文件的预计可用时间，并提供了一个查看 [GPT-3.5/4 编码词汇表文件](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) 的链接。

- **词汇表使用的 JSON 标准**：在词汇表讨论中，`@daain` 提到词汇表有一个 JSON 标准，其中包含使用该词汇表所需的元数据。在 [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/tokenizer.json) 上可以找到该 JSON 文件的直接链接。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (10 条消息🔥): 
        
- **使用 Docker 运行 Mistral**：用户 `@hanschrs` 通过在 **Docker** 命令中添加 `--tensor-parallel-size 2` 解决了运行 **Mistral** 的挑战，从而实现了并行张量处理。 

- **Docker 与 Ollama 安装对比**：`@vitorpinho` 询问了 **Docker** 和 **Ollama** 安装的优缺点。作为回应，`@vhariational` 建议通过几行命令行使用 **Ollama** 进行快速设置，而对于需要隔离以避免依赖冲突的情况，则推荐使用 **Docker**。 

- **Ollama 并非为微调设计**：在关于 **Ollama** 局限性的进一步讨论中，`@vhariational` 澄清说，虽然 **Ollama** 并非为模型微调（fine-tuning）而设计，但它可以处理复杂的用例，例如提供 **REST API** 来查询模型，并允许通过其模板系统自定义模型设置。

### ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/) (23 messages🔥): 
        
- **为聊天机器人实现 Mistral**：`@gmist` 报告称 **Mistral-medium** 模型有时会根据其自身的知识库回答问题，而不是依赖于给定的上下文。尽管 Prompt 指示模型仅根据上下文回答，但此问题仍然存在，因为 Mistral 并不总是遵守 Prompt。
- **Prompt 修改**：`@gmist` 分享了一些 Prompt 修改似乎有效，而另一些则无效。由于 Prompt 性能表现不一致，导致 `@gmist` 重新切回了 GPT，事实证明 GPT 在该特定用例中非常可靠。
- **Mistral 上下文理解的解决方案**：`@sublimatorniq` 建议在每行上下文前加上 "CONTEXT BODY" 前缀，并引入 "催眠式变量命名"（hypnotic var naming）来提高上下文理解能力。`@gmist` 还报告称，移除聊天历史记录似乎能提高 Mistral 对 Prompt 指令的响应度。
- **Mistral 对比 Mixtral**：`@daain` 在使用 LlamaIndex RAG 应用和各种版本的 Mistral 时也遇到了同样的指令遵循问题。然而，daain 发现 **Mixtral** 的表现优于 Mistral，并建议使用经过强训练的系统提示词进行 Fine-tuning 作为可能的解决方案。
- **Prompt 模板更新**：`@The Ledger Luminary` 建议更新 Prompt 模板，并尽可能清晰地重新措辞，同时引用特定的上下文片段。Luminary 警告说，如果上下文过多（Token 数量过高），指令可能会受到滑动窗口注意力（sliding window attention）的影响。


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (4 messages): 
        
- **量化 Fine-tuning 性能提升**：用户 `@The Ledger Luminary` 询问了量化 Fine-tuning 性能提升的方法，并寻求运行性能测试的库推荐。`@cpxjj` 推荐了一些库和性能基准测试，包括 [opencompass](https://github.com/open-compass/opencompass)、[llm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 和 [light-eval](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/light-eval)。
- **Function Call Fine-tuning**：用户 `@krissayrose` 询问了他们在对 Mistral 进行 Function Call（函数调用）Fine-tuning 时遇到的困难。突出的问题是模型在预期时没有预测出 EOS token，而是继续生成文本。他们提供了一个示例，并就可能出错的地方寻求帮助。


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (2 messages): 
        
- **MindMac AI 支持 Mistral**：用户 `@hoangnm` 介绍了 [MindMac](https://mindmac.app/) 应用，这是一个现在支持 Mistral AI 的 AI 聊天平台。MindMac 应用兼容来自 OpenAI、Azure OpenAI、Google Gemini 等的 API。它专为 macOS 设计，支持 Mac Intel 以及 Apple M1/M2/M3。用户引导观看者查看 [YouTube 视频](https://www.youtube.com/watch?v=ghMQsELSRek)以了解该平台的更多详情。
- **La Plateforme 的 Golang 客户端**：用户 `@r.j.k.` 分享了他为 La Plateforme 编写的 [Golang 客户端链接](https://github.com/robertjkeck2/mistral-go)，并寻求改进建议。


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (7 messages): 
        
- **将 Mac Mini 连接到 2007 年的显示器**：用户 `@pier1337` 发起了一场关于将 Mac Mini 连接到 2007 年显示器可能性的讨论。随后澄清该显示器来自一台 2007 款 iMac。`@daain` 建议，如果显示器或 iMac 具有 DVI 或 HDMI 等数字端口，它应该是可以工作的。
- **2007 款 iMac 端口问题**：`@pier1337` 通过分享 Apple 论坛的[链接](https://discussions.apple.com/thread/7139991?sortBy=best)补充了更多背景信息，其中指出 2007 款 iMac 使用的是 Mini DVI 端口，这导致不确定是否可以使用此端口连接 Mac Mini。
- **目标显示模式 (Target Display Mode)**：`@daain` 提供了一个[链接](https://recorder.easeus.com/screen-recording-resource/use-imac-as-monitor-for-pc.html)，解释说 2007 款 iMac 不具备目标显示模式（Target Display Mode），该功能是在 2009 年的 iMac 设备中引入的，使其能够用作另一台设备的显示器，因此可能无法将其用作 Mac Mini 的显示器。

### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (47 messages🔥): 
        
- **Mistral 模型的错误与故障排除**：用户 `@tinwhiskers` 在通过 API 使用较大的模型（`mistral-small` 和 `mistral-medium`）时遇到问题，并收到“model not found error”。在与 `@The Ledger Luminary` 和 Mistral 团队成员 `@tlacroix_` 讨论后，他们发现错误出在自己身上：他们尝试在 OpenAI URL 中使用 'mistral-small'。
- **关于流式传输（Streaming）和 Token 使用的讨论**：`@thesealman` 询问了如何计算流式请求中的 Token 使用量。用户 `@lerela` 确认目前没有办法计算。在官方功能推出之前，他们提供了一种 Token 使用量的估算策略。讨论还涉及了一些使用 tokenizer 对接收到的文本进行 Token 计数的策略。
- **对模型审查（Censorship）的担忧**：用户 `@smuglix` 和 `@Taiiouka` 对 API 模型的审查表示担忧，即使安全模式（safe mode）已设置为 'false'。`@titaux12` 建议查看文档以禁用安全模式，但 `@smuglix` 确认即使安全模式设置为 'false'，问题依然存在。
- **服务器错误事件**：用户 `@_jp1_` 报告了在使用 `mistral-medium` 模型时出现多次内部服务器错误（错误代码 503）。他们还对账户扣费表示担忧，扣费金额是他们自己追踪的 Token 使用量的两倍多，并询问了支持部门的联系方式。
- **关于 Mistral 速率限制（Rate Limit）的查询**：用户 `@flopsy1` 请求有关速率限制的信息，用户 `@r.j.k` 进行了回答，并提供了 Mistral 文档中的详细信息，指出所有端点的速率限制为每分钟 2M tokens 和每月 200M tokens。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **关于使用 OpenAI 和 LLaMA 技术的辩论**：有人指出，使用这些产品的输出来微调大语言模型（LLM）可能违反协议，并可能成为诉讼的理由，但存在“安全”的 Apache 许可模型，且符合此类准则。
- **探讨与 AI 输出相关的版权和所有权**：特别提到使用来自 API 的输出来训练模型是违反 OpenAI 协议的行为。
- **讨论了 `load_in_8bit` 或 `load_in_4bit` 参数对 QLora 中模型合并的影响**：澄清了尽管给出了参数，**Axolotl** 并不进行量化。
- **每个 PR 通过开发环境测试对 Axolotl 的重要性**：因为其在昂贵的开发环境中使用；关于微调 Mixtral 和 MoE 的问题已被提出并正在调查中。
- 分享了一个新的 Hugging Face Transformers 版本（`v4.36.2`）的[链接](https://github.com/huggingface/transformers/releases/tag/v4.36.2)，这可能有助于解决 Axolotl 中的一些关键问题。
- 成员们面临的**脚本、配置和运行**方面的各种挑战：包括**双 EOS token 问题、用于 RLHF 的最佳 OS 库、Docker 问题以及在 Mistral 上微调模型失败**；已尝试解决并持续跟进。
- 对人类与聊天机器人之间的多轮对话数据集表示兴趣：建议使用 Hugging Face 上的 [LMSys Chat 1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) 数据集。
- **RLHF 中未指明的对齐（unalignment）问题**：将由 `@giftedgummybee` 修复。
- 在 runpod-help 频道为各种问题提供协助和建议：包括**连接到 pod 前的等待、多 GPU 使用问题、显存溢出（OOM）问题以及 mpl4py 的安装**。提出的解决方案包括启用特定的训练方案、链接 [GitHub 上的 axolotl 仓库](https://github.com/OpenAccess-AI-Collective/axolotl)、校准 `max_split_size` 和 `batch_size` 修改以及 GPU 适配。

**OpenAccess AI Collective (axolotl) 频道摘要**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (55 messages🔥🔥): 
        
- **OpenAI 和 LLaMA 的使用协议**：`@nafnlaus00` 指出，使用 ChatGPT 等产品的输出来微调大语言模型（LLMs）违反了 OpenAI 的使用协议。同样的限制也适用于 LLaMA 和许多其他模型。他强调，违反协议可能会构成知识产权侵权和未经授权使用的诉讼依据。
  
- **“安全”的 Apache 许可模型**：`@nafnlaus00` 提到 **Mistral/mixtral base** 及其 instruct 模型，以及 **Falcon** 和其他几个模型是采用 Apache 许可证的模型，因此不受此类限制的“安全”影响。他还指出 **OpenAssistant** 中的一些条目存在可疑之处。
   
- **违反 OpenAI 服务条款**：`@visuallyadequate` 和 `@nafnlaus00` 就 OpenAI 服务条款（ToS）违规的可执行性及其影响展开了辩论。`@visuallyadequate` 认为 OpenAI 最多只能封禁用户，而 `@nafnlaus00` 则表示违反 ToS 等同于违约（Breach of Contract），这可能成为诉讼的依据。
   
- **AI 输出的所有权**：`@stefangliga` 和 `@visuallyadequate` 讨论了 AI 输出的所有权，重点在于版权不适用于 AI 输出的问题。`@stefangliga` 指出，不论版权问题如何，由于 OpenAI 的协议，使用 API 输出训练模型的权利已经被放弃。
   
- **合并 QLora 结果的量化**：`@touristc` 询问了 `load_in_8bit` 或 `load_in_4bit` 参数对 **QLora** 模型合并的影响。`@nanobitz` 澄清说，即使提供了这些参数，axolotl 也不会进行量化。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (9 messages🔥): 
        
- **开发环境测试**：`@nanobitz` 表示每个 PR 都应该通过测试，因为它们被用于昂贵的开发环境中。

- **关于微调 Mixtral 和 MoEs 的担忧**：`@nafnlaus00` 对 Mark Tenenholtz 发布的一条推文表示担忧，该推文提到由于需要实现负载均衡损失函数（load balancing loss functions），训练 MoEs 非常困难。这些担忧涉及正在使用的微调方法、确保专家（experts）之间的 Token 分布均匀，以及在多 GPU 系统中将每个专家分配到独立的 GPU 或 GPU 集群。

- **Caspar 对相关问题的处理**：`@caseus_` 提到 Caspar 正在调查 `@nafnlaus00` 提出的问题。

- **Hugging Face Transformers 的新版本**：`@casper_ai` 分享了 Hugging Face Transformers `v4.36.2` 版本的[链接](https://github.com/huggingface/transformers/releases/tag/v4.36.2)，该版本解决了与 cache 重构、flash attention 重构以及多 GPU 和多节点设置下的训练相关的关键问题，并建议 axolotl 应该更新到该版本。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (64 messages🔥🔥): 
        
- **shareGPT.py 的更改**：`@noobmaster29` 向 `OpenAccess-AI-Collective/axolotl` 仓库提交了一个拉取请求（[#976](https://github.com/OpenAccess-AI-Collective/axolotl/pull/976)），旨在解决在使用 Chatml 模板和 shareGPT.py 时，Prompt 末尾出现双 EOS token 的问题。该更改已与 `@nanobitz` 讨论，但需要更多测试来确认。
- **用于 RLHF 的开源库**：`@emperor` 询问了针对 RLHF 优化程度最高的开源库。`@nanobitz` 提到 TRL 是一个主要的选择。
- **在 Mistral 上运行微调模型**：`@JK$` 在运行上传到 Hugging Face 的 Mistral 微调模型时遇到问题。即使尝试使用 vLLM 并遵循各种文档和教程的指南，问题仍然存在。成员们尝试提供建议，但问题仍未解决。
- **Docker 配置困难**：`@JK$` 在 Docker 配置方面也遇到了问题，即使完全遵循 vLLM 文档中说明的配置也是如此。即使尝试不同的模型和端点，问题依然存在。社区尝试协助，但问题仍未解决。
- **双 EOS Tokens 问题**：`@noobmaster29` 和 `@self.1` 讨论了多轮对话中的双 EOS tokens 问题，并指出之前的修复未能解决该问题。他们同意稍后对此进行研究。

### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (2 条消息): 
        
- **多轮对话数据集请求**：`@natefyi_30842` 询问是否存在非合成的、人类与聊天机器人之间的多轮对话数据集，希望通过真实的人类数据来了解用户提出的问题类型。
- **推荐 LMSys 数据集**：`@natefyi_30842` 建议将 Hugging Face 上的 [LMSys Chat 1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) 数据集作为潜在资源，该数据集是公开可访问的，但需要分享联系信息才能获取访问权限。


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (2 条消息): 
        
- **未对齐问题修复**：`@giftedgummybee` 提到他们相信可以在几天内**修复一个未指明的未对齐（unalignment）问题**。


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (23 条消息🔥): 
        
- **连接 pod 前的等待**：用户 `@caseus_` 指出，在连接到 pod 之前等待大约 2 分钟有助于避免挂载点（mount point）加载和 axolotl 安装缺失的问题。
- **使用多 GPU 的问题**：`@mr_morning` 在尝试使用多块 RTX 4090 GPU 微调 Yi 时遇到了显存溢出（OOM）错误。尽管有两块 GPU，系统仅识别到一块（`num_machines:1`）。`@visuallyadequate` 回复称，`accelerate` 库应该能够自行分配负载，而无需 `deepspeed` 或 `fsdp` 等优化的多 GPU 训练方案，但权重仍会尝试加载到每个 GPU 上，这可能不是理想的结果。
- **为多 GPU 使用启用 DeepSpeed 和 FSDP**：`@visuallyadequate` 建议在相关的 yaml 文件中启用 `deepspeed` 或 `fsdp` 以优化多 GPU 训练，并提供了 [GitHub 上的 axolotl 仓库](https://github.com/OpenAccess-AI-Collective/axolotl) 链接以获取详细说明。鉴于 `fsdp` 持续存在的问题，`@noobmaster29` 建议使用 `zero3 deepspeed`。
- **持续的 OOM 问题与调整**：尽管进行了各种调整，包括校准 `max_split_size` 和修改 `batch_size`，`@mr_morning` 仍面临 OOM 错误。鉴于此，他考虑将当前的 RTX 4090 GPU 更换为具有更大显存容量（48GB）的 GPU。
- **mpl4py 安装故障**：`@mr_morning` 报告在 RTX 6000 Ada GPU 上尝试安装 `mpl4py` 及其依赖项时遇到问题，导致出现 "Cannot link MPI programs. Check your configuration!!" 等错误。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- 由 `@_jp1_` 等人发起的关于**评估模型（eval models）有益用途**的深入讨论，例如 Prometheus 模型，它可以快速评估“Grounding”、“风格+格式”或对特定 Prompt 指南的遵循情况。Prometheus 的官方实现可以在 Hugging Face [此处](https://huggingface.co/kaist-ai/prometheus-13b-v1.0)找到。
- 注意到 `@_jp1_` 在 **DiscoLM German 和 Disco Judge** 上的持续工作，并计划在未来一年发布针对多种用例的仓库，以及可能发布基于 Mixtral 的 Disco Judge Beta 版。
- `@rasdani` 介绍了一个新模型 **HALOs/Archangel**，预计很快会出现在 HF TRL 中，并附带了相关报告的[链接](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf)。
- `@_jp1_` 分享了 **Mixtral [config.json](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commit/125c431e2ff41a156b9f9076f744d2f35dd6e67a) 的重要更新**，澄清了其从未打算支持滑动窗口注意力（sliding window attention），并指向了相关的 TGI 修复和 PR。
- 讨论转向了 Disco 微调中 PEFT 与 NEFT 的对比，`@fernando.fernandes.` 询问上一次 Disco 微调使用的是 QLoRA + PEFT 还是 NEFT。`@_jp1_` 确认**使用了 PEFT/QLoRA**，并表示 NEFT 是一个额外的训练选项而非直接替代方案，且通常产生的结果令人失望。
- `@rasdani` 发布了 [DeepMind 的博客](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)，强调了 **LLM 在结合计算机代码中的函数搜索时，在发现数学科学开放性问题答案方面的效率**，并提出尝试使用开源 LLM 实现这一点的潜力。此外，还为有兴趣进一步开发的开发者分享了 [GitHub 上 FunSearch 代码实现的通配符部分](https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/implementation/sampler.py#L33)。

**DiscoResearch 频道总结**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (5 messages): 
        
- **Prometheus Model**: `@_jp1_` 强调了在难以进行 benchmark 的任务中使用 eval 模型（如 Prometheus 模型）的有益之处。他们指出，虽然这些模型在“准确度”方面存在上限，但它们可以快速评估其他类别，如 grounding、风格+格式，或对特定 prompt 规范的遵循情况。他们分享了一个基于 Prometheus 的模型用例，用于检查翻译指令数据的质量和正确性。官方 Prometheus 实现可以在 [Hugging Face](https://huggingface.co/kaist-ai/prometheus-13b-v1.0) 上找到。
- **DiscoLM German and Disco Judge**: `@_jp1_` 提到他们目前正在开发 DiscoLM German，并计划在明年发布一个包含多个用例的 repo，以及可能发布一个基于 Mixtral 的 Disco Judge beta 版本。
- **HALOs / Archangel**: `@rasdani` 提出了一个新模型 HALOs/Archangel，并链接到了一份 [报告](https://github.com/ContextualAI/HALOs/blob/main/assets/report.pdf)，并提到它很快将加入 HF TRL。


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (9 messages🔥): 
        
- **Mixtral Config Update**: `@_jp1_` 分享了 Mixtral [config.json](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commit/125c431e2ff41a156b9f9076f744d2f35dd6e67a) 的更新，并指出它从未打算支持 sliding window attention。他们还链接了相关的 TGI 修复 [此处](https://github.com/huggingface/text-generation-inference/releases/tag/v1.3.3) 和 PR [此处](https://github.com/huggingface/text-generation-inference/pull/1348)。

- **PEFT vs NEFT for Disco Fine-tune**: `@fernando.fernandes.` 询问最近的 disco 微调是使用了 qlora + peft 还是 neft。作为回应，`@_jp1_` 澄清说使用了 peft/qlora，因为 neft (noisy embedding vectors) 并不是一种替代方案，而是一个额外的训练选项，通常其结果令人失望。

- **Effectiveness of NEFT**: `@fernando.fernandes.` 还想知道使用 NEFT 是否能在 mixtral 上产生更好的效果。`@_jp1_` 否定了这一点，提到这与 mixtral 无关，那些尝试将其与 state-of-the-art 参数和标准正则化结合使用的人得到了平庸或相同的结果。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (1 messages): 
        
- **FunSearch - Discoveries in Mathematical Sciences Using Large Language Models (LLMs)**: `@rasdani` 分享了一篇 [DeepMind 博客文章](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)，展示了当 Large Language Models (LLMs) 与计算机代码中的“函数”搜索相结合时，如何高效地在数学科学的开放性问题中取得发现。他们还建议尝试使用开源 LLM 来实现这一点的潜力。
- **FunSearch Code Implementation**: `@rasdani` 进一步分享了 [GitHub 上 FunSearch 代码实现的特定部分链接](https://github.com/google-deepmind/funsearch/blob/65ba52cba984bba8df788e921a4fb7790881177d/implementation/sampler.py#L33)，供有兴趣为其开发做出贡献的人参考。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

- 关于通过**编写代码来改进语言模型知识链（chain of knowledge）**这一主题的详细讨论，参考了 [`@roger_alca` 分享的研究论文](https://arxiv.org/abs/2312.04474)。
- 关于 **PyfanticParser 变量**和 **ConfluenceLoader OAuth 令牌**的查询及错误处理：用户分别就多对象 JSON 输出解析和 ConfluenceLoader 的关键参数要求寻求建议。
- 披露了 `@banda_ki` 与另一名用户通过**私信（private messages）**进行的直接沟通，未提供进一步细节。
- 用户 `@banda_ki` 和 `@alewe5` 分别提出了关于 **LangChain Agent** 和 **SQL 虚拟数据库**经验的问题，目前尚未收到回复。
- `@ssowonny` 建议使用**第三方服务 PlugBear** 将 [LangChain 和 LangServe 应用与 Slack 集成](https://plugbear.io/posts/integrate-langchain-langserve-app-with-slack-using-plugbear)，并在 #general 和 #share-your-work 频道中提供了详细的操作指南。
- `@appstormer_25583` 提到了一个**使用 GPT 构建的光明节（Hanukkah）食谱生成器**的开发情况，但未提供更多细节或项目链接。
- `@andysingal` 分享了一篇讨论 **LangChain 在数据分析中潜在应用**的[文章](https://ai.gopubby.com/unlocking-the-power-of-language-how-langchain-transforms-data-analysis-and-more-3c4f327d520d)。
- `@dhruv.xyz` 宣布了**应用构建工具 Create** 的重大更新，现在支持通过输入规范进行实时应用构建，并提供了更新后的[应用链接](https://www.create.xyz)。


**LangChain AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (9 条消息🔥): 
        
- **代码链 (Chain of code)**：`@roger_alca` 分享了一篇讨论利用编写代码来改进语言模型知识链的研究论文。[点击查看研究论文](https://arxiv.org/abs/2312.04474)。
- **JSON 输出解析器**：`@infinityexists.` 询问是否可以定义两个不同的 PyfanticParser 变量来处理 API 返回的两种类型的 JSON 对象，因为在打印接收到的对象时遇到了错误。
- **ConfluenceLoader OAuth 令牌**：`@night765` 提出了关于 Confluence 使用 OAuth 令牌的 ConfluenceLoader 的问题。用户对 Loader 所需的密钥数量以及这些密钥与 AtlassianRestAPI 类所需密钥之间的差异感到困惑。
- **私信**：`@banda_ki` 提醒一名用户检查其私信，未透露更多信息。
- **LangChain Agent 与虚拟数据库**：用户 `@banda_ki` 和 `@alewe5` 分别询问是否有人具有使用自定义工具的 LangChain Agent 以及使用 SQL 虚拟数据库的经验，但未立即收到回复。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 条消息): 
        
- **将 LangChain+LangServe 与 Slack 集成**：用户 `@ssowonny` 推荐了第三方服务 [PlugBear](https://plugbear.io/posts/integrate-langchain-langserve-app-with-slack-using-plugbear)，用于将 LangChain 和 LangServe 应用与 Slack 集成。该帖子提供了关于如何使用 PlugBear 设置自定义 LLM 的分步指南。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (4 条消息): 
        
- **基于 GPT 的光明节食谱生成器**：`@appstormer_25583` 分享了一个关于使用 GPT 构建的光明节食谱生成器的链接。没有提供额外细节或进一步探索该工具的链接。
- **用于数据分析的 LangChain**：`@andysingal` 分享了一篇发表在 AI Advances 上的[文章](https://ai.gopubby.com/unlocking-the-power-of-language-how-langchain-transforms-data-analysis-and-more-3c4f327d520d)，题为《解锁语言的力量：LangChain 如何变革数据分析等领域》，作者是 Ankush k Singal。该博客讨论了 LangChain 在数据分析方面的潜在应用。
- **LangServe 与 Slack 集成**：`@ssowonny` 发布了一份关于如何在 5 分钟内将 LangServe 应用与 Slack 或 Discord 集成的指南。该[教程](https://plugbear.io/posts/integrate-langchain-langserve-app-with-slack-using-plugbear)托管在 PlugBear 上。
- **Create 应用构建工具更新**：`@dhruv.xyz` 宣布了应用构建工具 Create 的重大更新，现在允许通过输入规范实时构建应用。分享了更新后的[应用链接](https://www.create.xyz)，并就新功能寻求反馈。


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

- 围绕 **Artificial General Intelligence (AGI)** 的讨论：用户们在思考 AGI 的现状以及对世界的普遍情绪。
- 用户 **Cursor** 在 GitHub PR 系统中的进展和参与：表明社区对软件项目的贡献正在增长。
- 关于缺乏有效的 **Infra/DevOps AI 工具** 的对话：用户指出这些领域的 AI 应用仍有进一步提升的空间。
- 关于 **Mixtral 处于 beta 阶段** 的谨慎建议：由 `@swyxio` 提供，他在 NeurIPS 会议上见到了 Mixtral 的开发者。[Prompt 技巧链接](https://fxtwitter.com/abacaj/status/1736819789841281372?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- 关于 **Mixtral** 访问方式的查询：用户在争论是否正在使用 Anyscale 调用进行访问。

**Latent Space 频道摘要**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (7 条消息): 
        
- **对 AGI 的感受**：用户 `@spicychickensandwichdeluxe` 询问大家是否感受到了 AGI (Artificial General Intelligence)，而 `@slono` 则评论说世界很残酷。
- **Cursor 在 GitHub PR 上的进展**：用户 `@guardiang` 提到 **Cursor** 正逐渐涉足 GitHub PR (Pull Request) 领域，并开始查看 diff。
- **用于 Infra/DevOps 工作的 AI 工具**：`@btdubbins` 表示，尽管 AI 和编程有所进步，但感觉许多工具在基础设施/开发运维 (DevOps) 工作中仍然不够有效。
- **关于 Mixtral 处于 Beta 阶段的警告**：`@swyxio` 提到他在 NeurIPS (Conference on Neural Information Processing Systems) 遇到的 Aman 对 Mixtral 持谨慎态度，强调它充其量处于 beta 阶段。该用户还分享了一个[今日有趣的 Prompt 技巧](https://fxtwitter.com/abacaj/status/1736819789841281372?s=46&t=90xQ8sGy63D2OtiaoGJuww)，即告诉 AI 它是 GPT-5。
- **访问 Mixtral**：`@btdubbins` 询问用户是如何访问 **Mixtral** 的，打听他们是否在使用 Anyscale 调用。


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 条消息): 
        
eugeneyan: yeap, see you then!


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

仅 1 个频道有活动，无需摘要...

- **Skunkworks AI 开发更新**：用户 `@far_el` 提供了关于公司目前运营的一些见解。他们澄清说 **Skunkworks AI 不再公开构建 (build in public)**，并提到他们将 **很快发布模型、软件和产品**。
        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

仅 1 个频道有活动，无需摘要...

- **AMD 与 Nvidia GPU 性能之争**：`@entropi` 分享了来自 [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/amd-strikes-back-at-nvidia-with-new-mi300x-benchmarks-mi300x-shows-30-higher-performance-than-h100-even-with-an-optimized-software-stack) 的一篇文章，讨论了 **AMD Instinct MI300X** 与 **Nvidia H100 (Hopper) GPU** 之间的性能差异。AMD 将使用 vLLM（热门选择）的 FP16 与仅支持 TensorRT-LLM 的 FP8 进行了对比。
- **关于开源对话模型微调的查询**：用户 `@beowulfbr` 询问是否有关于微调新的 **open chat model** 的指南、示例或 colab。
        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord 摘要

仅 1 个频道有活动，无需摘要...

- **2023 年变革性数据格局回顾**：用户 `@viv2668` 讨论了 2023 年的现代数据栈 (MDS)、雄心勃勃的 Gen AI 项目以及若干争议。讨论主要集中在数据行业的**创新**和趋势。分享了全文链接：[在此阅读全文](https://moderndata101.substack.com/p/recap-of-2023s-transformative-data)。