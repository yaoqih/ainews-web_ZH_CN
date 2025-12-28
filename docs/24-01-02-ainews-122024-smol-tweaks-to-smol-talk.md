---
companies:
- openai
- meta-ai-fair
- perplexity-ai
date: '2024-01-03T07:38:24.484214Z'
description: '**OpenAI** Discord 的讨论重点对 **Perplexity**、**Copilot**、**Bard** 和 **Claude
  2** 等 AI 搜索引擎进行了详细对比，其中 Bard 和 Claude 2 的表现相对落后。Meta 推出了 **Meta AI** 聊天机器人，可在 Instagram
  和 WhatsApp 上使用，其具备的图像生成功能被比作免费版的 GPT。用户反映了 **ChatGPT** 的多个浏览器端问题，包括在使用 VPN 时频繁出现验证码以及插件故障。辩论内容涵盖了提示词工程（prompt
  engineering）、API 使用以及 **JSON**、**YAML** 和 **Markdown** 等数据格式。讨论还涉及了 ChatGPT 的性格调优和模型能力的差异。“Meta
  AI 包含图像生成功能，他将其比作免费版的 GPT。”'
id: 35042e7a-2765-48c3-a894-4ff649792087
models:
- claude-2
- bard
- copilot
- meta-ai
- gemini-ultra
- chatgpt
original_slug: ainews-122024-smol-tweaks-to-smol-talk
people: []
title: 2024年1月2日：对 Smol Talk 进行了一些微调。
topics:
- prompt-engineering
- api
- json
- yaml
- markdown
- chatbot
- image-generation
- vpn
- browser-compatibility
- personality-tuning
- plugin-issues
---

<!-- buttondown-editor-mode: plaintext -->今天来点 Smol Talk 的幕后消息：我们对项目符号摘要的“乏味感”不太满意，所以尝试调整了 Prompt：

 
![image.png](https://assets.buttondown.email/images/36e567d0-bf55-4196-b70a-f04e8d5f00e6.png?w=960&fit=max)
 

结果反映在下方的摘要中。然而，感觉它们丢失了一些信息，因为我们的 Few-shot 示例太短了。我们打算修改整个流水线，[像上帝旨意的那样使用 Function Calling](https://twitter.com/minimaxir/status/1737884828111179916)。

---

**目录**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **AI 霸权之争**：AI 搜索引擎之间的详细对比，包括 **Perplexity**、**Copilot**、**Bard** 和 **Claude 2**，由 `@jaicraft` 在 [ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) 中提及，Bard 和 Claude 2 在竞争中位居次席。

- **Meta AI 大放异彩**：Meta（前 Facebook）推出的聊天机器人 **Meta AI** 加入对话，已在 Instagram 和 Whatsapp 上可用，由 `@thedreamakeem` 和 `@jaicraft` 分享。

- **浏览器与 ChatGPT 的博弈**：在 [openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) 频道中，有大量关于 OpenAI ChatGPT 的多个浏览器问题的报告，从使用 VPN 时的持续验证码到插件无法运行等。

- **ChatGPT 的性格切换**：在 [openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) 中，关于调整 ChatGPT 性格、不同模型能力以及软件开发支持的可行解决方案进行了激烈的交流。

- **JSON vs. Markdown 和 YAML**：在 [gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) 中，关于在特定任务应用中使用 **JSON**、**YAML** 或 **Markdown** 的辩论。

- **Prompt Engineering 难题**：在 [prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) 中的复杂讨论涵盖了遭遇悖论、揭示模型特性和问题，以及 TikTok 脚本编写和 SWOT 分析。

- **剖析 API**：[API-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) 揭示了“最后一次看到”的悖论、通过 Bing 进行实时数据检索、脚本喂入、Prompt 构建技巧以及 API 实现中文本输出的限制。

**OpenAI 频道摘要**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (11 条消息🔥): 
        
- **AI 搜索引擎对比**：`@jaicraft` 表示基于 AI 的搜索引擎 **Perplexity** 表现良好。不过，他们也认为 **Copilot** 令人满意，可以说优于 **Bard** 和 **Claude 2**。
  
- **请求 Bard 更新**：`@jaicraft` 表达了希望将 **Gemini Ultra** 整合进 **Bard Advanced** 的愿望。他们欣赏 Bard 的性格，但认为模型可以通过这次更新得到提升。
  
- **Meta AI 发布**：`@thedreamakeem` 宣布了 Meta（前 Facebook）推出的 **Meta AI**。值得注意的是，该聊天机器人目前已在 Instagram 上运行。
  
- **Whatsapp 上的 Meta AI**：在后续对话中，`@jaicraft` 指出 Meta AI 也可以在 **Whatsapp** 上使用。
  
- **Meta AI 图像生成**：`@thedreamakeem` 表示 Meta AI 包含 **图像生成功能**，他将其比作 GPT 的免费版本。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (209 条消息🔥🔥): 
        
- **ChatGPT 在多个浏览器上的问题**：用户 `@rosyskull` 和 `@kilogamz` 报告了在包括 Opera GX、Firefox 和 Edge 在内的多个浏览器上使用 ChatGPT 时出现的问题。即使在清除缓存、更换设备并尝试不同的网络连接后，问题依然存在。`@satanhashtag` 建议联系 `help.openai.com` 并在 `<#1070006915414900886>`（OpenAI 专门处理此类查询的频道）中发帖。
  
- **ChatGPT 关于自身的信息不可靠**：用户 `@lugui` 和 `@beanz_and_rice` 讨论了 ChatGPT 在提供有关自身信息（特别是版本、升级和功能）时的不可靠性。会议强调 ChatGPT 经常会产生幻觉（hallucinates）或误传此类信息。讨论还澄清了模型名称中使用的不同版本和术语。

- **使用 VPN 时频繁出现验证码的问题**：用户 `@thedeeplearningdummy` 提出了一个问题：每当他们通过 VPN 连接访问 ChatGPT 时，都会遇到不断的验证码（captcha）验证。讨论中一个受认可的理论是，VPN 可能会因为缓存/IP 掩盖（IP masking）而触发安全措施，尽管此事尚无定论。

- **GPT+ 中 AskYourPDF 插件的问题**：用户 `@kez92.` 报告了 GPT+ 中插件 “AskYourPDF” 的问题，该插件通常允许 ChatGPT 处理 PDF 文件。问题在于，它没有允许上传 PDF，而是要求复制粘贴信息或提供链接，这是一种异常行为。消息中未提供解决方案。 

- **GPT3.5-turbo-1106 输出异常且快速达到使用上限**：多位用户报告了 `GPT3.5-turbo-1106` 模型的问题，包括 `@minister_maximus` 报告收到乱码输出，以及 `@Mr Gingerfish` 声称仅在四个提示词（prompts）后就达到了使用上限（usage cap）。其他人提供了一些理论、可能的原因和解决方案，但问题似乎仍在持续且未得到解决。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (75 条消息🔥🔥): 
        
- **ChatGPT 在性格特征塑造上表现不一致**：`@thefreebachelor` 和 `@solbus` 讨论了调整 ChatGPT 性格的难度。`@thefreebachelor` 提到知识文件在配置 AI 的个人资料图片方面效果很好，但在性格方面效果不佳，并对一段冗长的 Reddit 段落导致模型变得混乱表示沮丧。`@solbus` 建议从全新的 GPT 开始，并手动编辑指令以接近预期的结果。

- **GPT-4 的对话限制**：用户 `@crafty.chaos`、`@paulh3911` 和 `@darthgustav.` 讨论了与 GPT-4 对话长度的限制。在用户报告在与 AI 的聊天会话中遇到“对话太长，请开始新对话”的错误后，提出了这个问题。 

- **ChatGPT 的技术问题**：用户 `@ssk0746`、`@froggy_chacko`、`@bartomeu_70091_41043`、`@watchalls` 和 `@solbus` 讨论了他们在 ChatGPT 服务中遇到的技术问题。问题包括 ChatGPT 4.0 在某些 VPN 下无法工作，以及过多的验证拼图问题。`@bartomeu_70091_41043` 关于人工验证拼图的问题通过在 Firefox 中不使用 VPN 得到了暂时解决。

- **在软件开发工作中使用 ChatGPT**：用户 `@muhct` 提出了一个场景，询问在雇主不知情的情况下在软件开发工作中使用 ChatGPT 的最佳方式。`@spyied` 和 `@michael_6138_97508` 的建议围绕使用 GitHub Copilot 等替代服务以及记录专有数据保护责任展开。

- **关于 GPT-4 消息上限额度的询问**：`@yomo42` 询问小组成员 “GPT-4 classic” 是否比普通 GPT-4 具有更高的消息上限。`@yomo42` 补充说，他们了解到自定义 GPT 版本的上限是每 3 小时 25 条消息。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (6 条消息): 
        
- **考虑使用 YAML 和 Markdown 替代 JSON**：`@sciandy` 建议在某些任务中可能使用 **YAML** 或 **Markdown** 作为 JSON 的替代方案。
- **寻求 WordPress 技术协助**：`@delightful_rabbit_60937` 正在寻求帮助，希望在他们的 WordPress 上使用 **Power AI** 插件实现聊天机器人，并希望它能在 freelancer 上提供具有相关熟练度的链接。
- **JSON 的优势**：`@niko3757` 强调 **JSON** 是最有效的方式，因为它允许使用关键词进行搜索，从而有效地节省宝贵的 tokens。
- **丢失的 GPT 已恢复**：`@loschess` 提到他们所有丢失的 GPT 都已找回，并想知道是否还有其他人遇到同样的情况。

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (169 messages🔥🔥): 
        
- **GPT-4 的矛盾“最后一次出现”问题**：`@beanz_and_rice` 指出了 GPT-4 Preview 在询问“最后一次看到 X 的时间”时存在的一个矛盾问题。模型解释说，由于“last”一词的歧义性，导致模型的指令（访问截至 2023 年的历史数据）与该词固有的即时性含义之间产生了冲突。这种差异导致模型在其建立的时间范围内（2023 年之前）提供最新的信息，这可能与 2023 年之后的实际“最后一次”发生情况不符。
- **GPT-4 的各种版本及其功能**：对话揭示了不同 GPT-4 版本的各个方面。例如，GPT-4.0613 在处理“最后一次看到”查询时没有 GPT-4 Preview 那样的问题。此外，`@beanz_and_rice` 询问 GPT-4 Turbo 是否包含 Vision 功能，以及 GPT-4.1106 是否能够“看到”图像，`@madame_architect` 澄清说 Vision 是 GPT-4 Turbo 的一项技能。
- **GPT-4 对 Bing 的过度依赖**：一些用户，特别是 `@beanz_and_rice`，对 GPT-4 尽管拥有广泛训练的数据集，却倾向于依赖 Bing 获取信息表示不满。`@madame_architect` 合理化了这种行为，指出 GPT-4 被训练为使用 Bing 技能，从而利用搜索结果中的实时数据来增强其回答。
- **针对 TikTok 脚本和 SWOT 分析的 Prompt-Engineering**：`@user691378` 寻求关于编写 Prompt 以使用 ChatGPT 生成有趣的 TikTok 脚本的建议，对此 `@beanz_and_rice` 提供了一种独特的伪语言。相比之下，`@madame_architect` 提出了一个挑战，即创建生成技术 SWOT 分析的 Prompt，并指出分别针对优势、劣势、机会和威胁进行 Prompt 引导时效果更好。
- **调节 GPT-4 输出的挑战与解决方案**：`@MishaMgla` 和 `@vangroovie` 都在努力解决 GPT-4 输出中的问题。`MishaMgla` 困扰于如何在不过度设计模型并产生额外成本的情况下限制响应长度。`@rendo1` 承认了局限性，建议通过 API 设置 Token 限制，但警告可能会出现句子截断的情况。`@vangroovie` 报告了 GPT 在客服转录文本中错误地调换说话者身份的问题。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (169 messages🔥🔥): 
        
- **最后一次出现的悖论**：用户 `@beanz_and_rice` 发现了 **GPT-4-Preview** 在回答涉及“最后一次看到 X 的时间”的查询时的悖论。聊天机器人将其描述为一个模糊的陈述，解释说由于它拥有截至 2023 年的历史数据，但无法访问实时更新，因此它没有能力确定 2023 年之后任何事件的最新发生情况。

- **Bing 浏览及其在 API 中的角色**：用户 `@beanz_and_rice` 分析认为，每当请求“实时”信息时，GPT-4-Preview 往往会调用 Bing 搜索技能。`@madame_architect` 进一步澄清说，Bing 并非互联网的整个数据集，而是 GPT 用来实时抓取特定任务所需数据的工具。

- **利用 GPT 执行独特任务**：用户 `@user691378` 发起了一场讨论，尝试向 GPT 强制喂入自定义脚本，以生成有趣的 TikTok 短信视频。

- **构建有效 Prompt 的技术**：`@madame_architect` 分享了她准备有效 Prompt 的经验和方法，包括阅读大量关于 Prompt 策略的 AI 论文，并沉浸在 GPT 的实操学习实验中。

- **文本输出的限制**：`@MishaMgla` 表达了对控制 GPT 生成输出的长度和质量的担忧。他们发现，将文本长度保持在一定范围内需要生成额外的响应，从而增加了额外的时间和成本。用户提出了限制输出长度的方法，尽管其中一些方法可能会导致 GPT 在句中截断文本。


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 摘要

- **Show Me the Merges**：`@clefourrier` 宣布在 Open LLM Leaderboard 中新增了 “Show Merges” 过滤器，该功能最初由 Stanford NLP Group 建议，旨在帮助元数据不完整的创作者。[Source Tweet](https://x.com/clefourrier/status/1742224428639928694)
- **低延迟推理**：`@reach_vb` 宣布了 AutoAWQ，它允许在约 24GB GPU VRAM 中运行带有 Flash Attention 2 的 Mixtral 8x7B MoE，并实现快速推理。[Source Tweet](https://x.com/reach_vb/status/1741175347821883502)
- **Whisper 引导 Metal**：`@reach_vb` 透露，Metal 上的 Whisper 现在由 Rust 编程语言驱动。[Source Tweet](https://x.com/reach_vb/status/1740804591095283899)
- **Hugging Face 的援手**：`@RisingSayak` 关注了 `@vipitis` 的有用回复，该回复为有兴趣为 Hugging Face 开源项目做贡献的 `@ronny4002` 提供了指导。[Contributing Guidelines](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)
- **实时数学课**：`@osanseviero` 分享了一篇名为 [The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/) 的博文，提供了 Transformer 模型内部数学原理的端到端演练。
- **我们该何去何从？**：`@dhruvdh` 和 `@chad_in_the_house` 讨论了论文 *“The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey”*，探讨了面向任务的 LLM 系统的设计参数。[Author's Link](https://arxiv.org/abs/2312.17601)
- **鼓励技术演进**：`@apolinariosteps` 介绍了 `Advanced Training Script`，它结合了社区中成功的技术，以提升训练能力。[Release Tweet](https://x.com/linoy_tsaban/status/1742281649294016742?s=20)
- **关注视觉输出**：`@wandereronarock` 引用了仓库 [patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux)，作为他们正在讨论的视觉输出方法的示例。
- **AI 咬住布尔值**：`@stroggoz` 和 `@ketul1842` 之间的对话围绕使用 encoder-decoder 模型或 next sentence prediction 策略对 prompt 生成布尔逻辑响应展开。

**HuggingFace Discord 频道摘要**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 条消息): 
        
- **Open LLM Leaderboard 上的合并模型过滤**：用户 `@clefourrier` [宣布](https://x.com/clefourrier/status/1742224428639928694)了 Open LLM Leaderboard 的新 `Show Merges` 过滤器，并请求协助帮助元数据不完整的模型创建者。`Show merges` 过滤器的想法最初是由 Stanford NLP Group 提出的。
- **结合 AWQ 和 Flash Attention 2 增强 Mixtral 8x7B**：`@reach_vb` [展示](https://x.com/reach_vb/status/1741175347821883502)了 AutoAWQ 的最新版本，该版本允许在约 24GB GPU VRAM 中运行带有 Flash Attention 2 的 Mixtral 8x7B MoE，并实现快速推理。该用户还提供了使用此功能的全面指南。
- **由 Rust 驱动的 Whisper on Metal**：`@reach_vb` [分享](https://x.com/reach_vb/status/1740804591095283899)了 Whisper on Metal 现在由 Rust 编程语言驱动。
- **Hugging Face 的实用库 —— ‘huggingface_hub’ 和 ‘accelerate’**：`@RisingSayak` [提到](https://x.com/RisingSayak/status/1740998251447550329)了 Hugging Face 的两个非建模库 —— 用于与 Hub 集成的 `huggingface_hub` 和用于处理模型推理的 `accelerate`。
- **DINOv2 微调与 Qwen 更新**：用户 [NielsRogge](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb') 分享了一个使用 Hugging Face Trainer 微调 DINOv2 进行图像分类的教程。同时，`@JustinLin610` [宣布](https://x.com/JustinLin610/status/1742184229453320451)将 Qwen-VL 更新为 Qwen-VL-Plus，提升了图像描述（image captioning）、视觉问答（visual question answering）、视觉定位（visual grounding）、OCR 和视觉推理等能力。

**提到的链接**：

- [Clémentine Fourrier 🍊 (@clefourrier) 的推文](https://x.com/clefourrier/status/1742224428639928694)：刚刚添加了 "Show merges" 来过滤掉 m...
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1741175347821883502)：带有 AWQ & Flash Attention 2 的 Mixtral 8x7B Instruct...
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1740804591095283899)：太棒了！由 Rust 驱动的 Whisper on Metal 🦀 10...
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1740998251447550329)：Hugging Face 拥有非建模库，这些库...
- [Transformers-Tutorials/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb at master · NielsRogge/Transformers-Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Fine_tune_DINOv2_for_image_classification_[minimal].ipynb)：这个仓库包含了我用 Trainer 制作的演示...
- [Junyang Lin (@JustinLin610) 的推文](https://x.com/JustinLin610/status/1742184229453320451)：📷🎨👀 https://huggingface.co/spaces/Qwen/Qwen-VL-...

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (167 条消息🔥🔥): 
        
- **AI 在法律领域的应用**：`@caleb_sol` 分享了一篇文章的[链接](https://lsvp.com/legaltech-x-ai-the-lightspeed-view/)，强调了 AI 如何构成法律领域的重大变革。
- **Inference Endpoints 持续延迟**：`@duplaja` 提到他注意到 Inference Endpoints 从暂停状态启动时持续缓慢，并在求助频道中分享了这一问题。
- **贡献给 Hugging Face**：学生 `@ronny4002` 表达了对参与 Hugging Face 开源项目的兴趣，特别是针对 NLP/LLM 的代码开发和训练数据集贡献。`@vipitis` 引导他查看 GitHub 上的[贡献指南](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)，并建议查阅 Hugging Face 课程和文档以获取更多见解。
- **`from_pretrained` 中 `torch_dtype=torch.float16` 与 `variant="fp16"` 的区别**：`@felixsanz` 询问了 `from_pretrained` 函数中 `torch_dtype=torch.float16` 和 `variant="fp16"` 的区别。`@kyvarus` 和 `@jo_pmt_79880` 解释说，`torch_dtype` 指定了张量的数据类型，而 `variant` 指定了模型仓库的分支。讨论以 `@vipitis` 的解释结束，即无论哪种方式，指定 dtype 都会转换为目标数据类型。
- **Gradio 的性能问题**：`@tony_assi` 报告了 Gradio 频繁出现的 bug，导致每天都会产生问题。`@cakiki` 建议通过特定渠道联系 Gradio 团队或在 GitHub 上提交 issue。


**提到的链接**：

- [Legaltech x AI: The Lightspeed View - Lightspeed Venture Partners](https://lsvp.com/legaltech-x-ai-the-lightspeed-view/)
- [transformers/CONTRIBUTING.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md): 🤗 Transformers: State-of-the-art Machine Learning...
- [transformers/src/transformers/modeling_utils.py at aa4a0f8ef37eb5d42b4e3810f37e554585c90d41 · huggingface/transformers](https://github.com/huggingface/transformers/blob/aa4a0f8ef37eb5d42b4e3810f37e554585c90d41/src/transformers/modeling_utils.py#L3348): 🤗 Transformers: State-of-the-art Machine Learning...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 条消息): 
        
- **探索 HF Inference Endpoints**：`@duplaja` 分享了他们学习 `handler.py HF Inference Endpoints` 的经验，但承认进展*非常非常缓慢*。
- **未做更改，仅调整参数**：`@gag123` 表示他们*没有更改任何内容*，只是在*尝试调整参数*。
- **MacOS 的便捷性令 Windows 用户惊讶**：后端软件开发人员 `@lawls.net` 表达了他们对 MacOS 上使用 Python 的便捷程度（相比 Windows）感到惊讶，这促使他们购买了一台 M3 Max。
- **关于学习率和数据 dtype 的建议**：`@exponentialxp` 提供了一些技巧，建议将学习率提高到 *3e-4*，并建议将数据设置为 *dtype=torch.long*。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 条消息): 
        
- **用数学揭秘 Transformer**：用户 `@osanseviero` 分享了他新博文的链接，标题为 [The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/)。该文章提供了一个关于 **Transformer 模型内部数学原理**的端到端示例，并进行了简化以便更好地理解。文章还建议读者阅读 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) 以获得对 Transformer 模型更直观的解释。用户 `@jo_pmt_79880` 表达了赞赏，认为这篇文章“**非常具有启发性**”。

**提到的链接**：

[hackerllama - The Random Transformer](https://osanseviero.github.io/hackerllama/blog/posts/random_transformer/): 通过揭秘...了解 Transformer 的工作原理。

### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (3 messages): 
        
- **释放 AI 融合的对话力量**：用户 `@andysingal` 分享了一篇 [博客文章](https://medium.com/ai-advances/unleashing-conversational-power-rag-agent-tools-and-trulens-eval-revolutionize-document-chatting-32845ea2215b)，讨论了集成 **LlamaIndex 和 Gemini** 以增强 AI 能力。 
- **用 AI 寻找你的明星脸**：`@tony_assi` 开发了一个 [应用](https://huggingface.co/spaces/tonyassi/celebrity-look-a-like)，可以识别出与用户长相相似的名人。

**提到的链接**：

- [Celebrity Look A Like - a Hugging Face Space by tonyassi](https://huggingface.co/spaces/tonyassi/celebrity-look-a-like)
- [Unleashing Conversational Power: RAG, Agent Tools, and Trulens-Eval Revolutionize Document Chatting…](https://medium.com/ai-advances/unleashing-conversational-power-rag-agent-tools-and-trulens-eval-revolutionize-document-chatting-32845ea2215b): Ankush k Singal


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (12 messages🔥): 
        
- **探索任务导向型 LLM 系统**：`@dhruvdh` 分享了一篇题为 *"The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey"* 的新论文。该论文研究了任务导向型 LLM 系统的设计参数，并提出了三个推测，为未来的研究提供了起点。点击 [此处](https://arxiv.org/abs/2312.17601) 查看论文。

- **论文展示提案**：`@chad_in_the_house` 推荐了 `@dhruvdh` 分享的论文作为本周的展示内容。`@lunarflu` 确认这是一个潜在选项，并询问是否有其他建议。

- **关于 RL Keras 的求助**：`@only_busy` 请求推荐与 OpenAI 新的 gymnasium 兼容的库，因为他在使用 RL Keras 和 OpenAI gym 时遇到了兼容性问题。

- **读书小组日历**：`@swyxio` 表示对用于跟踪读书小组会议的日历感兴趣。`@clock.work_` 建议检查 Discord 上是否列出了小组活动，`@chad_in_the_house` 则建议在 Discord 线程中查找信息。

- **更便捷地访问往期展示**：`@chad_in_the_house` 提议在 HuggingFace 博客文章和 Discord 上分享演示文稿，以增加往期展示内容和潜在演讲者的可见度。

**提到的链接**：

- [The Tyranny of Possibilities in the Design of Task-Oriented LLM Systems: A Scoping Survey](https://arxiv.org/abs/2312.17601)：这篇范围综述侧重于我们目前的理解...
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/18w09hn/r_the_tyranny_of_possibilities_in_the_design_of/)


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **发布全新高级训练脚本**：用户 `@apolinariosteps` 宣布了 2024 年的首个发布版本：`Advanced Training Script`。该脚本整合了社区中使用的成功技术，包括来自 `cog-sdxl` 的 Pivotal Tuning 和来自 `kohya` 的 `Prodigy Optimizer`。该脚本兼容 AUTO1111 和 ComfyUI，并可通过 Google Colab、Hugging Face Spaces 或直接作为 Python 脚本使用。[发布推文](https://x.com/linoy_tsaban/status/1742281649294016742?s=20)
- **高级训练脚本特性**：该脚本灵感源自 Replicate 的 SDXL Cog 训练器中使用的 Pivotal Tuning 技术，以及 Kohya 训练器中的 Prodigy 优化器。它还包含其他优化功能，有望为 SDXL Dreambooth LoRA 微调带来出色效果。[博客链接](https://huggingface.co/blog/sdxl_lora_advanced_script)
- **平台兼容性**：上述脚本允许在不同平台上进行多样化使用。用户可以在 [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_Dreambooth_LoRA_advanced_example.ipynb) 上执行脚本，在具有简单 UI 和自定义参数的 [Hugging Face Spaces](https://huggingface.co/spaces/multimodalart/lora-ease) 上运行，或者直接作为 Python 脚本运行。

**提到的链接**：

- [Tweet from Linoy Tsaban🎗️ (@linoy_tsaban)](https://x.com/linoy_tsaban/status/1742281649294016742?s=20): Let&#39;s go 2024 🚀:  🆕 training script in 🧨 @d...
- [LoRA training scripts of the world, unite!](https://huggingface.co/blog/sdxl_lora_advanced_script)

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 
        
- **ControlNet_Aux 被链接为视觉输出示例**: `@wandereronarock` 引用了仓库 [patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux) 作为他们讨论的方法所产生的输出示例。该用户指出，由于有一段时间没使用该方法，他们并不确定。

**提到的链接**:

[GitHub - patrickvonplaten/controlnet_aux](https://github.com/patrickvonplaten/controlnet_aux): 为 patrickvonplaten/controlnet_aux 的开发做出贡献...


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **从 Prompt 到命题**: `@stroggoz` 和 `@ketul1842` 讨论了从给定 Prompt 创建正确的布尔逻辑命题的想法。`@stroggoz` 建议这可以通过使用 encoder-decoder 模型或 next sentence prediction 策略来实现。

- **从 OpenAI 到 HuggingFace 的过渡**: 用户 `.oo92` 分享了一个使用 **OpenAI API** 根据输入的 system message 和文件 Prompt 生成知识图谱的函数，并表示在尝试用 **HuggingFace** 的 **Bloom** 镜像类似逻辑时遇到了困难。

- **用于数据集的网络爬虫**: `@exponentialxp` 询问了在 **owt** 和 **c4** 等数据集中使用的网络爬虫技术。`@cakiki` 澄清说 **C4** 是 **Common Crawl** (CC) 的过滤版本，后者每月抓取一次网页。

- **GPU 推荐**: `@exponentialxp` 还推荐查看 GitHub 上的 karpathy/nanoGPT，以获取有关如何使用多 GPU 训练模型的信息，并建议 Lambda 是 GPU 租赁的一个不错选择。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **TinyLlama MOE 实验**: `@le_mess` 分享了他使用八个 [TinyLlama](https://huggingface.co/mhenrichsen/tinymix-8x1b) 实例构建的早期 Mixture of Experts (MoE) 模型。还重点讨论了微调细节以及关于提升模型性能的 LASER 方法的激烈讨论。正如 `@mihai4256` 所描述的，LASER 方法与现有模型集成，引发了关于潜在过拟合的兴趣。
- **Mixtral 和 Axolotl 的探索**: 深入探讨了 Mixtral 和 Axolotl 的实现，包括 GitHub 项目用例的建议、Axolotl 中 Prompt 格式的查询以及对 SFT 的支持。`@noobmaster29` 分享了关于使用 qlora 训练 Mistral 的建议，将对话引导向在开发者频道中保持专注讨论的方向。
- **训练常见问题解答及故障排除技巧**: 在 `general-help` 频道中，用户讨论了提高模型回答多样性的方法，遇到了训练过程中的 index-out-of-bound 问题，分析了 LoRA 配置，并分解了 Hellaswag 的评分方法。发现 Hugging Face tokenizer 与 Sentencepiece 原始实现之间的差异引发了进一步的讨论。
- **值得关注的新数据集亮相**: `@cf0913` 介绍了一个合成代码数据集 `DevSpecCode`，用于复杂的多需求代码生成指令。该数据集托管在 huggingface.co 上，正在征求反馈。该数据集包含复杂的代码生成指令。
- **角色扮演数据集及微调讨论**: 深入研究角色扮演数据集，`@nruaif` 分享了对敏感内容的访问权限以及更多潜力。此外，还辩论了微调的首选数据集选择，比较了 redpajama-v2 与 slimpajama，重点关注质量过滤。

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (63 messages🔥🔥): 
        
- **使用 8x TinyLLama 的 MOE**：`@le_mess` 使用八个 [TinyLlama](https://huggingface.co/mhenrichsen/tinymix-8x1b) 实例创建了一个 MoE (Mixture of Experts) 模型。目前处于早期评估阶段，可能需要进一步训练才能超越其组件模型的表现。该工作是使用 [mergekit](https://github.com/cg123/mergekit/tree/mixtral) 的 mixtral 分支完成的。
- **LASER 讨论**：多位社区成员讨论了 LASER 方法，该方法通过移除模型 MLP 层的阶数较高的组件来提高模型性能。`@mihai4256` 分享了[他在模型上进行 LASER 的实验](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1)，并提到如果模型可以在 Hugging Face 的 Transformers 中加载，则可以使用 Python 包将 LASER 与现有模型进行简单集成。`@nafnlaus00` 对 LASER 方法潜在的过拟合问题表示关注。
- **Tokenizer 网站查询**：`@yamashi` 询问了一个可以比较不同 Tokenizer 分词结果的网站 URL。该网站的目的是提供分词的可视化。
- **训练与微调讨论**：多位社区成员（特别是 `@nafnlaus00`、`@noobmaster29` 和 `@le_mess`）就模型的训练和微调发表了评论，包括提到 Dropout + LR 超参数调整以及 MoE 模型训练。
- **微调方法比较**：`@jaredquek` [分享了一篇论文](https://huggingface.co/papers/2401.00788)，详细介绍了不同微调方法在不同规模和任务中的性能比较。论文发现，全参数微调 (FFT) 通常在所有规模上都能提供最佳性能，而参数高效微调 (PEFT) 方法的性能权衡会随模型规模发生显著变化。其中 LoRA 被发现在成本和性能之间通常能提供良好的平衡。

**提到的链接**：

- [The Truth Is In There: Improving  Reasoning in Language Models with Layer-Selective Rank Reduction](https://pratyushasharma.github.io/laser/)
- [Paper page - Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models](https://huggingface.co/papers/2401.00788)
- [mhenrichsen/tinymix-8x1b · Hugging Face](https://huggingface.co/mhenrichsen/tinymix-8x1b)
- [GitHub - open-compass/MixtralKit: A toolkit for inference and evaluation of 'mixtral-8x7b-32kseqlen' from Mistral AI](https://github.com/open-compass/MixtralKit): 用于推理和评估 Mixtral 的工具包...
- [GitHub - cg123/mergekit at mixtral](https://github.com/cg123/mergekit/tree/mixtral): 用于合并预训练大语言模型的工具...
- [mhenrichsen](https://wandb.ai/mhenrichsen/huggingface?workspace=user-mhenrichsen): Weights & Biases，机器学习开发者工具...
- [Mihaiii/Pallas-0.5-LASER-0.1 · Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1)
- [GitHub - pratyushasharma/laser: The Truth Is In There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction](https://github.com/pratyushasharma/laser): 通过层选择性秩削减改善语言模型的推理能力...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (9 messages🔥): 
        
- **分享可能用于 Mixtral 的 GitHub 项目**：`@casper_ai` 分享了一个名为 "PyTorch bindings for CUTLASS grouped GEMM" 的 [GitHub 项目](https://github.com/imoneoi/cutlass_grouped_gemm)链接，建议它可以用于 **Mixtral**。
- **关于 Axolotl 是否支持 SFT 的查询**：`@mrfakename_` 询问 Axolotl 是否支持 SFT。`@noobmaster29` 确认支持，并提到 LoRA/QLoRA 也是 SFT。
- **关于 Axolotl 中 Prompt 格式使用的讨论**：`@mrfakename_` 还询问了是否可以使用 Zephyr 的 Prompt 格式。`@noobmaster29` 回复称这可能需要自行设置。
- **使用 QLoRA 训练 Mistral 的建议**：`@builderx` 提出了一个关于使用 QLoRA 在 1.5 万行的数据集上训练 Mistral 的问题，询问 3 个 Epoch 是否足够。`@noobmaster29` 确认 3 个 Epoch 应该可以，并建议观察 Eval Loss。不过，他们也建议在帮助频道提出此类问题，以保持 `axolotl-dev` 频道仅供开发者讨论。

**提到的链接**：

[GitHub - imoneoi/cutlass_grouped_gemm: PyTorch bindings for CUTLASS grouped GEMM.](https://github.com/imoneoi/cutlass_grouped_gemm): CUTLASS grouped GEMM 的 PyTorch 绑定。

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (33 条消息🔥): 
        
- **建议在预处理期间打乱数据以增加多样性**：引发了一场关于增强模型响应多样性的讨论。虽然 `@casper_ai` 建议增加更多样化的数据，但他们也建议在预处理期间对数据进行 **shuffling**（打乱）。
- **训练期间的索引越界问题**：用户 `@matanvetzler` 在训练期间遇到了 CUDA 的 `RuntimeError`，原因是“索引越界（index out of bounds）”问题。该错误似乎发生在模型维度与 tokenizer 大小不匹配时。`@bjoernp` 指出了这一点。
- **提出了关于 LoRA 配置的问题**：`@tcapelle` 询问在针对具有 LoRA 配置的模块时，是否应将 `lora_target_linear` 设置为 `False`，并提供了一个 [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml) 上的示例链接作为参考。
- **关于 Hellaswag 评分方法的澄清**：`@le_mess` 询问了 Hellaswag 的正确评分方法，`@suikamelon` 澄清通常使用 `acc_norm`（归一化分数）。他们提供了 [HuggingFace OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 的链接作为该用法的示例。
- **注意到 Huggingface tokenizer 与原始 Sentencepiece 实现之间的差异**：用户 `@emperor` 提到 `transformers/AutoTokenizer` 提供的内容与原始 Sentencepiece 表示的结果不一致。他们分享了代码示例来展示这些差异。

**提到的链接**：

- [Open LLM Leaderboard - Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [axolotl/examples/mistral/qlora.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml)：欢迎提问。为 Open... 做出贡献。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (1 条消息): 
        
- **寻求对专业数据集的反馈**：`@cf0913` 为复杂的多需求代码生成指令创建了一个详细的合成代码数据集。他们正在寻求对名为 `DevSpecCode` 的数据集的反馈，该数据集托管在 huggingface.co 上。数据集指令包括复杂的需求、限制和代码生成指令。提供的一个示例是关于在 Go 中编写安全的并发函数。数据集可通过[此链接](https://huggingface.co/datasets/cfahlgren1/DevSpecCode)获取。


**提到的链接**：

[cfahlgren1/DevSpecCode · Datasets at Hugging Face](https://huggingface.co/datasets/cfahlgren1/DevSpecCode)


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (8 条消息🔥): 
        
- **分享了针对“非全年龄（Not-For-All-Audiences）”的角色扮演数据集**：`@nruaif` 分享了两个来自 huggingface.co 的角色扮演数据集链接：[rpguild](https://huggingface.co/datasets/chargoddard/rpguild) 和 [bluemoon-fandom-1-1-rp-cleaned](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned)，这些数据集包含敏感内容。
- **有可能获取更多角色扮演数据**：`@nruaif` 还表示如果需要可以**提供更多角色扮演数据**，暗示有 **0.5 TB 的数据**可用。
- **关于 Mixtral 微调策略的讨论**：`@tcapelle` 询问了关于使用 GitHub 上的特定 JSON 文件 [zero3_bf16.json](https://github.com/OpenAccess-AI-Collective/axolotl/blob/transformers-update-mixtral/deepspeed/zero3_bf16.json) 来微调 Mixtral 模型的问题。
- **关于 Redpajama-v2 和 Slimpajama 数据选择的辩论**：`@xzuyn` 建议使用 **redpajama-v2** 而不是 **slimpajama** 数据。然而，`@caseus_` 指出持续预训练可能不需要大量数据，并引用了 Sheared llama 的论文，该论文仅使用了 **50B 更多的 token**。他们主张使用 slimpajama 中的**质量过滤**。

**提到的链接**：

- [axolotl/deepspeed/zero3_bf16.json at transformers-update-mixtral · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/transformers-update-mixtral/deepspeed/zero3_bf16.json)：欢迎提问。为 Open... 做出贡献。
- [chargoddard/rpguild · Datasets at Hugging Face](https://huggingface.co/datasets/chargoddard/rpguild)
- [Squish42/bluemoon-fandom-1-1-rp-cleaned · Datasets at Hugging Face](https://huggingface.co/datasets/Squish42/bluemoon-fandom-1-1-rp-cleaned)

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **模型选择 - Perplexity vs GPT-4 vs Claude 2.1**：服务器上的用户就最佳 AI 模型的使用展开了讨论，`@prolapsedgod` 在 Perplexity Experimental 模型和 GPT-4 模型之间犹豫不决。`@hynjia` 插话透露他们倾向于在 GPT-4 和 Claude 2.1 之间切换。
- **复杂的 Mixtral 参数**：`@.hackit` 询问了 `mixtral-8x7b-instruct` 的参数，意图通过 rest-API 重新构建它，这引起了大家的兴趣。
- **'HOLIDAYS23' 优惠券混乱**：'HOLIDAYS23' Pro 会员优惠券代码的使用引发了多位用户（`@nightpearl62`、`@icelavaman`、`@jonathanonymous`、`@ok.alex`、`@danielagmz888` 和 `@ashley__9910`）的热烈讨论。遇到的问题从链接功能到代码应用不等，建议的方法包括通过 [Plexity.ai](https://pplx.ai/holidays) 订阅、尝试移动端网页以及联系客服。
- **Perplexity 内部的图像生成 - 视觉刺激？**：`@araf@` 和 `@archient` 探讨了 Perplexity 内部图像生成的好处。讨论围绕其在增强视觉参与度和作为创意刺激方面的作用展开。
- **恳请在 Pro 账户中加入 Mistral-Medium**：`@Rehnn` 建议将 Mistral-Medium 模型集成到 Pro 账户中，并根据其在 playground 的测试，赞扬了该模型的速度和减少幻觉的表现。`@reflext` 同意这一提议。
- **跨国支付困扰**：来自德国的 `@ashley__9910` 在支付方式上遇到困难。尽管尝试在 Android 手机上使用 GPay，问题依然存在。`@icelavaman` 建议联系 support@perplexity.ai 以寻求有关 Stripe 相关问题的帮助。
- **Perplexity 的数据收集和模型信息**：`@maxxflyer` 关注了 [Perplexity 的 FAQ](https://docs.perplexity.ai/page/frequently-asked-questions)，强调了其围绕 API 使用和用户账户信息的数据收集做法。他们还评论道，根据 [Perplexity 博客文章](https://blog.perplexity.ai/blog/introducing-pplx-online-llms)，pplx-7b-online 和 pplx-70b-online 模型使用了公共数据及其搜索索引中的数据。
- **失效的 Discord 邀请链接与军事寄宿历史**：Perplexity 页面上失效的 Discord 邀请链接被 `@maxxflyer` 重点指出。在另一场对话中，`@jpa` 对军事寄宿历史表示好奇，并参考 [Perplexity Search](https://www.perplexity.ai/search/whats-the-history-2Hi46dUBTIayi4HlvrozmQ?s=c) 进行进一步探索。
- **API 和频道混乱**：`@paul16307` 提出了对模型排序设置的偏好，希望将 Mistral 模型按性能从弱到强排列。当 `@monish0612` 询问是否有带 Mistral API 的在线 LLM 时，`@icelavaman` 引导他关注 **pplx-7b-chat**。频道分类也是讨论的一个重点，用户被引导至适合不同主题的频道。

**Perplexity AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (82 条消息🔥🔥): 
        
- **选择合适的 AI 模型**：`@prolapsedgod` 提出了一个关于使用哪个模型的问题，表示在 Perplexity Experimental 模型和 GPT-4 模型之间很难做出决定。`@hynjia` 加入了对话，并提到他们通常在 GPT-4 和 Claude 2.1 之间切换。
- **关于 Mixtral 参数的查询**：用户 `@.hackit` 提出了一个问题，以了解 `mixtral-8x7b-instruct` 的参数，试图通过 rest-API 重新创建它。
- **关于 'HOLIDAYS23' Pro 会员优惠券的互动**：进行了一场关于应用 'HOLIDAYS23' Pro 会员优惠券代码的讨论，`@nightpearl62`、`@icelavaman`、`@jonathanonymous`、`@ok.alex`、`@danielagmz888` 和 `@ashley__9910` 参与了讨论。用户提出了链接功能和代码应用方面的问题，回复建议的方法包括通过 Plexity.ai 订阅、尝试通过移动端网页订阅以及联系支持部门。
- **Perplexity 中图像生成的固有优势和疑问**：`@arafay` 和 `@archient` 围绕 Perplexity 中图像生成的使用和益处进行了深入讨论。他们反思了其视觉参与度方面的表现以及作为创意刺激物的潜力。
- **在 Pro 账户中加入 'Mistral-Medium' 模型的兴趣**：`@Rehnn` 根据他们在 playground 中的测试，建议在 Pro 账户中加入 Mistral-Medium 模型，并评论了其速度和减少的幻觉（hallucinations）。`@reflext` 同意这一想法。
- **欧洲用户的支付挑战**：来自德国的 `@ashley__9910` 面临支付方式的挑战，尝试在 Android 手机上使用 GPay 的建议未能奏效。`@icelavaman` 建议联系 support@perplexity.ai 以寻求有关 Stripe 相关问题的帮助。

**提到的链接**：

[Perplexity Pro](https://pplx.ai/holidays)：享受 2 个月的免费 Perplexity Pro 或 40 美元的优惠...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (7 条消息): 
        
- **Perplexity AI 收集 API 和用户信息**：`@maxxflyer` 指向了 Perplexity AI 的 [FAQ](https://docs.perplexity.ai/page/frequently-asked-questions)，其中提到该组织会收集与 API 使用数据相关的信息以及用户的账户信息，如姓名、电子邮件地址等。
- **Perplexity 的 pplx-7b-online 和 pplx-70b-online 模型使用公开数据**：`@maxxflyer` 提到 Perplexity 的 pplx-7b-online 和 pplx-70b-online 模型使用公开数据及其搜索索引中的数据，并参考了 [Perplexity 博客文章](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) 以获取更多信息。
- **Perplexity 页面上的 Discord 邀请失效**：`@maxxflyer` 指出 Perplexity 页面上的 Discord 邀请链接无法正常工作。
- **Jpa 对军队登机历史的好奇**：`@jpa` 提出了一个关于为什么军队在飞机上优先登机历史的问题，并提供了一个指向 [Perplexity 搜索](https://www.perplexity.ai/search/whats-the-history-2Hi46dUBTIayi4HlvrozmQ?s=c) 的链接以获取更多详情。
- **频道分类讨论**：`@icelavaman` 告诉 `@maxxflyer` 使用不同的频道，`@ok.alex` 引导 `@504076886268313616` 和 `@1179797683167309926` 前往适合各自讨论的频道。

**提到的链接**：

[常见问题解答 (Frequently Asked Questions)](https://docs.perplexity.ai/page/frequently-asked-questions)


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (3 条消息): 
        
- **模型组织建议**：`@paul16307` 建议在下拉菜单中将 **Mistral 模型**按性能从弱到强进行排序。
- **关于带有 Mistral API 的在线 LLM 的查询**：`@monish0612` 询问是否有带有在线 LLM 的 Mistral API 可用，`@icelavaman` 给予了肯定回答，并提供了名称 "**pplx-7b-chat**"。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **Google 的 T2I 与 DALL·E-3：巅峰对决**：Google 的 T2I 工具引发了褒贬不一的评价。一些用户赞赏其保真度且没有过拟合（overfitting），而另一些用户则批评其过度审查以及 Prompt 遵循能力较弱。分享了 Reddit 链接以获取更多见解，用户建议从公会创建共享的 Prompt。*“[Google Lab Experiments 看起来远优于...](https://labs.google/)*”
   
- **日本法律下的 AI 训练版权**：小组讨论了一篇关于日本版权法的文章，该法律目前允许使用合法数据集进行不受限制的 AI 训练。然而，成员们提醒要注意持续的游说活动可能会改变这一现状。*“[日本全力投入，版权不适用于 AI 训练...](https://www.biia.com/japan-goes-all-in-copyright-doesnt-apply-to-ai-training/)*”

- **用于更大、更好数据集的通配符概念**：提出了一种新方法，涉及利用每个现有的 CLIP 关键词来合成大型数据集。提议的概念设想利用 Language Models 来融合各种概念。
  
- **SynCLR：未来是合成的**：分享了一项名为 [SynCLR](https://fxtwitter.com/_akhaliq/status/1741668076037247127) 的研究，该研究完全利用合成图像和标题进行视觉表示学习（visual representation learning），而不使用真实数据，这被认为是通配符概念想法的一种潜在实现。

- **高质量文本嵌入的新浪潮**：成员们分享并讨论了一篇 [arxiv 论文](https://arxiv.org/abs/2401.00368)，该论文概述了一种利用合成数据实现高质量文本嵌入（text embeddings）的方法，采用 LLMs 创建一系列合成数据并微调开源的 decoder-only LLMs。

- **MSE 在 Classifier-free Guidance 中的表现**：均方误差（MSE）被发现在 Classifier-free Guidance 中表现更好。其他模型，如使用感知损失（perceptual losses）训练的模型，被评估为成本更高。
   
- **自感知目标与 Diffusion Models**：针对使用预训练模型作为感知损失的特征提取器进行了一场深入辩论，围绕一篇提出自感知目标（self-perceptual objective）的[论文](https://arxiv.org/abs/2401.00110)展开，该目标能够产生更真实的图像，但在重复迭代时性能会下降。

**LAION 频道摘要**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (49 条消息🔥): 
        
- **Google 的 Text-to-Image 平台令人惊叹**：用户 `@SegmentationFault` 最初对 Google 的新 T2I 工具 [Google Lab Experiments](https://labs.google/) 赞不绝口，认为它在图像保真度和缺乏过拟合方面优于 DALL·E-3。然而，包括 `@thejonasbrothers` 和 `@nodja` 在内的用户指出了它的局限性，包括 Prompt 遵循能力不如 DALL·E-3，以及屏蔽了某些类型的内容，包括某些与人相关的图像。
    
- **分享生成的图像**：`@nodja` 提出了生成共享 Prompt 的想法，用户建议从预定频道生成 Prompt 以进行比较。随着用户测试系统，分享了更多图像。
    
- **图像生成中的限制性屏蔽**：据 `@nodja` 称，Google 的 T2I 平台据报道在 Prompt 层面或图像层面屏蔽了某些类型的内容，特别是围绕与人相关的话题。用户 `.undeleted` 批评了这一决定，质疑如果系统甚至拒绝生成标准内容，那么它的价值何在。
    
- **分享 T2I 见解**：`@SegmentationFault` 建议查看 Reddit 上的 /r/dalle2 和 /r/weirddalle 以获取更多关于 T2I 的帖子，并提供了 [r/ImagenAI](https://www.reddit.com/r/ImagenAI/) 的链接，尽管该板块当时活跃度较低。
    
- **日本的 AI 训练版权**：用户 `@vrus0188` 分享了一篇关于日本在版权和 AI 训练方面的立场及其对机器翻译潜在影响的文章。`@peacekeeper8310` 澄清说，这一立场源于 2018 年的一项法律，该法律允许使用合法数据集对 ML 模型进行不受限制的训练，但也警告说，由于持续的游说努力，这种情况可能会发生变化。

**提到的链接**：

- [日本全力投入：版权不适用于 AI 训练 | BIIA.com | 商业信息产业协会](https://www.biia.com/japan-goes-all-in-copyright-doesnt-apply-to-ai-training/)
- [来自 Zumer (@zumersultana) 的推文](https://vxtwitter.com/zumersultana/status/1734530458359349500)：Google 现在可以生成高质量的 AI 图像。一个...
- [在 Search Labs 中尝试实验](https://www.google.com/search/images/editor/mNJJecqS)

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (34 条消息🔥): 

- **使用通配符概念合成 CLIP 数据集**：`@SegmentationFault` 提出了利用现有每个 CLIP 关键词合成大型数据集的想法。提议的概念包括利用 LLM 来合并不同的概念。
- **SynCLR 作为该想法的潜在实现**：`@rom1504` 分享了由 `_Akhaliq` 发起的一项名为 [SynCLR](https://fxtwitter.com/_akhaliq/status/1741668076037247127) 的研究，该研究完全从合成图像和合成标题中学习视觉表示，而不利用任何真实数据。
- **探索使用合成数据的高质量文本嵌入**：`@thejonasbrothers` 和 `@mkaic` 讨论了一种简单但有效的方法，即通过合成数据和少于 1k 次训练步骤来获取高质量的文本嵌入，该方法在一篇 [arxiv 论文](https://arxiv.org/abs/2401.00368) 中有详细阐述。该方法利用 LLM 生成多样化的合成数据，并对开源的 decoder-only LLMs 进行微调。
- **关于 Diffusion 模型中感知损失的讨论**：`@nodja`、`@mkaic` 和 `@clock.work_` 就使用预训练模型作为其自身感知损失的特征提取器进行了深入讨论。他们的讨论围绕一篇[论文](https://arxiv.org/abs/2401.00110)展开，该论文提出了一种自感知目标，可以生成更真实的图像，尽管迭代此方法似乎会降低性能。
- **MSE 在配合 Classifier-free Guidance 时表现更好**：`@thejonasbrothers` 指出均方误差（MSE）在配合 Classifier-free Guidance (CFG) 时效果更好。其他可能性（如使用感知损失进行训练）被认为成本更高。

**提到的链接**：

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)：在这篇论文中，我们介绍了一种新颖且简单的方...
- [来自 AK (@_akhaliq) 的推文](https://fxtwitter.com/_akhaliq/status/1741668076037247127?t=83X-PzqSIMTQVncqyg-LDw&s=19)：从模型中学习视觉能力足以媲美从...学习视觉...
- [Diffusion Model with Perceptual Loss](https://arxiv.org/abs/2401.00110)：使用均方误差损失训练的 Diffusion 模型...
- [Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels](https://arxiv.org/abs/2312.17090)：在线可用视觉内容的爆炸式增长...

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **“Pascal 中的复古酷感”**：关于使用老派 Pascal 语言的幽默交流，其中包含由 @psipiai 分享的一个指向用 Pascal 编写的 OpenAI 单元的 [gist 链接](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1)。
- **“LMStudio 实战排错”**：围绕 LMStudio 性能的技术讨论，@malte0621 在使用 Vision 模型时遇到了严重的减速问题，通过管理 VRAM 溢出（spillover）和正确处理 CPU 线程解决了该问题。
- **“模型解析！”**：关于特定 LMStudio 模型之间差异的好奇询问，即 dolphin-2.6-mistral-7b-dpo 和 dolphin-2.5-mixtral-8x7b，以及如何加载 Hugging Face 模型。针对这些问题，@fabguy 澄清只有 GGUF 格式的模型与 LMStudio 兼容，并为 Mixtral 模型提供了硬件需求建议。
- **“代码翻译器的难题”**：@xenorhon 提出了关于代码翻译任务理想模型的问题。在 @fabguy 建议将传统方法与 LLM 结合以获得最佳结果后，对话以 @usizu 推荐 tinyllama 模型告终。
- **“驯服 Tiny Llama”**：模型讨论聊天中出现了模型相关问题，@macaulj 正在与难以驾驭的 Tiny Llama 2 模型作斗争。建议包括尝试 [Mixtral 8X7B 模型](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)，并为内存受限的设置推荐了 Mistral 7B Q4 或 Zephyr Beta 等模型。
- **“反馈频道秩序维护”**：用户 @heyitsyorkie 明确提醒了 Discord 频道规则，强调应正确使用反馈频道进行与 LMStudio 相关的讨论。
- **“硬件规格讨论与基准测试分享”**：一场活跃的硬件讨论，包括 @leviticus_slow 对特定 eBay 硬件配置的评估，以及 @nixthefolf 提供的 GPU 替代方案建议。此外，还有 @kess4747 对 Nvidia 未来计划的推测，以及 @discgolfvalife 分享的基准测试。新用户 @dantefortal 请求协助处理 TinyLlama 安装错误。

**LM Studio 频道总结**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (52 messages🔥): 
        
- **Pascal 中的 OpenAI 单元**：@psipiai 分享了一个 [gist 链接](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1)，展示了 Pascal 语言编写的 OpenAI 单元。该帖子开玩笑地提到了老派的 Pascal 语言，暗示即使是复古编码也有其辉煌时刻。
- **LMStudio 性能咨询**：@malte0621 报告在 LMStudio 上使用 Vision 模型时出现明显变慢。在与 @fabguy 进行详细讨论后，通过管理 VRAM 溢出和正确处理 CPU 线程解决了该问题。
- **在 LMStudio 中混合模型**：@vanthryn 对 **dolphin-2.6-mistral-7b-dpo** 和 **dolphin-2.5-mixtral-8x7b** 模型之间的输出质量和功能差异感到好奇，旨在决定是将 Mistral 7b 完整加载还是将 Mixtral 8x7b 部分加载到 GPU 显存中。@heyitsyorkie 警告说，除非具备必要的硬件规格（至少 20GB VRAM），否则不要使用 Mixtral。
- **将 Hugging Face 模型加载到 LMStudio**：@usizu 询问如何将 Hugging Face 模型加载到 LMStudio，并想尝试 OpenPipe/mistral-ft-optimized-1227 模型。@fabguy 澄清说，只有 GGUF 格式的模型才与 LMStudio 兼容。此外，直接输入模型名称可以绕过非兼容模型的 URL 过滤器。
- **代码翻译建议**：@xenorhon 发起了一场讨论，寻求执行代码翻译任务的模型建议。@fabguy 建议，将解析器（parsers）和语法（grammars）等传统方法与 LLM 相结合，比仅使用 LLM 效果更好。讨论以 @usizu 建议考虑 tinyllama 模型结束。

**提到的链接**：

- [OpenPipe/mistral-ft-optimized-1227 · Hugging Face](https://huggingface.co/OpenPipe/mistral-ft-optimized-1227)
- [Snoop Dogg Dre GIF - Snoop Dogg Dre Riding - Discover &amp; Share GIFs](https://tenor.com/view/snoop-dogg-dre-riding-driving-rolling-gif-13800248)：点击查看 GIF
- [OPEN AI Unit. in Pascal. Also works for other compatible stuff like LMstudio, Firemonkey cross platfrom test app attached](https://gist.github.com/twobob/74dc94c03a62da0a2ced3a5203fe7ae1)：Pascal 中的 OPEN AI 单元。也适用于其他兼容工具，如 LMstudio，附带 Firemonkey 跨平台测试应用。
- [TheBloke/mistral-ft-optimized-1227-GGUF · Hugging Face](https://huggingface.co/TheBloke/mistral-ft-optimized-1227-GGUF)


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (9 messages🔥): 
        
- **驯服 Tiny Llama 2 的麻烦**：`@macaulj` 报告在使用 **Tiny Llama 2** 模型时遇到问题，称其生成随机内容且不听从输入指令。 
- **推荐模型**：`@kess4747` 建议 `@macaulj` 尝试 [Mistral AI 的 Mixtral 8X7B Instruct v0.1 模型](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)，声称从中获得了更好的结果。 
- **模型的内存限制**：`@macaulj` 表示他们只有 **8 或 16 GB 内存**可用，`@fabguy` 对此建议使用 **Mistral 7B Q4** 或 **Zephyr Beta** 模型，据称这些模型在此内存限制内能提供足够的性能。
- **Deepseek Chat 模板查询**：`@nvn.osto` 询问了 **Deepseek Chat 67B** 的推荐模板。根据 `@heyitsyorkie` 的说法，默认的 LMStudio 预设（preset）对这些模型效果良好。

**提到的链接**：

[TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)


### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (1 messages): 
        
- **频道规则的明确提醒**：用户 `@heyitsyorkie` 提醒社区，🧠-feedback 频道专门用于讨论与 LMStudio 相关的反馈。对于其他讨论，合适的频道包括：用于 Bug 反馈的 `<#1111440136287297637>`、`<#1128339362015346749>`、`<#1111649100518133842>` 和 `<#1139405564586229810>`，以及分别用于模型建议和聊天的 `<#1185646847721742336>` 和 `<#1110598183144399061>`。

### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (10 messages🔥): 
        
- **硬件规格讨论**：`@leviticus_slow` 分享了一个 eBay 商品详情，并询问在配备双 E5-2697v3 的 PowerEdge R730 中安装双 Tesla M10 是否是划算的购买选择。`@nixthefolf` 回复称，由于 Tesla M10 年代久远且性能较低，并不是最佳选择，并建议将更现代的 Tesla P40 或二手 3090 作为替代方案。
- **猜测 Nvidia 的未来计划**：`@kess4747` 表示希望在未来的 Nvidia 显卡型号（如传闻中的 5090）中看到 NVLink 的回归。
- **分享基准测试**：`@discgolfvalife` 分享了他在华硕 G17 笔记本电脑上使用 LM Studio V 0.2.10 测试各种 AI 模型的详细规格和基准测试结果。他还分享了一个详细的测试提示词（prompt），并征求建议或意见。`@doderlein` 回复了他在两个 OpenChat 模型上使用相同测试提示词运行的结果。
- **TinyLlama 的问题**：新用户 `@dantefortal` 在尝试安装 TinyLlama 时遇到了问题，分享了一条显示其处理器不支持 AVX2 指令集的错误消息，并寻求帮助。

**提到的链接**：

[Nvidia Tesla M10  32GB  180-12405-DAAB-A01 GDDR5 GPU  | eBay](https://www.ebay.com/itm/145495209710)


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **与 Mixtral 的爱恨情仇**：用户对 Mixtral 的输出比预期受到更多审查表示失望。揭示了 Mistral 模型的身份（如 "mistral-tiny"），并提出了完善文档的建议。提供了使用 Mistral API 的 Python 实用代码。最后，在 Raspberry Pi 上运行模型的可行性引发了好奇。
- **Docker 与内存，一段悲惨的爱情故事**：频繁的 CUDA Out-of-memory (OOM) 问题再次抬头，即使在高性能机器上也是如此。用户需要关于在本地运行 Mistral 的详细说明。必要的 API URL 之谜仍未解开。Raspberry Pi 再次作为运行 AI 模型的可行性讨论对象出现。
- **文本清洗风波**：用户表示需要强大的文本清洗解决方案，特别是针对扫描书籍等大型且混乱的语料库。讨论了开箱即用方案的局限性，并建议采用定制的 ETL 或 LT 操作。pandas、OpenCV 和 PIL 等推荐库带来了希望。
- **频道联系难题**：用户询问更新情况并寻求联系维护者的指导，但对话缺乏进一步深入了解的上下文。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (50 messages🔥): 
        
- **Mixtral 并不像预期的那样无审查**：用户 `@meyelo` 和 `@hexdigest` 讨论了他们使用 Mixtral 输出的体验，表示失望，因为它并不像最初认为的那样无审查。他们报告称收到了“不要违反 TOS（服务条款）”的消息，即使在使用 API 并关闭安全模式（safe mode）时，也无法获得某些提示词的响应。
- **揭示 Mistral 模型的真实身份**：针对 `@juliandarley` 的提问，`@i_am_dom` 澄清说，在 API 中 Mistral 7b 被称为 "mistral-tiny"，而 Mixtral 被称为 "mistral-small"。社区普遍认为，在文档中使用更清晰的命名可以减少困惑。
- **使用 Mistral API 的实用代码**：响应 `@refik0727` 的请求，`@i_am_dom` 分享了使用 Mistral API 与选定模型（如 "mistral-medium"）进行对话的 Python 实用代码，展示了非流式和流式响应。
- **如何让 Mistral 遵循特定规则**：`@unknownperson2156` 征求关于指示 Mistral 模型遵循特定规则的建议，以确保生成的文本更短并符合规则。`@hexdigest` 给出的建议包括简化指令和降低 Temperature 参数。
- **在 Raspberry Pi 上运行 AI 的可行性**：`@kaigenji` 询问了在 Raspberry Pi 或类似的物联网（IoT）单板计算机上运行 Mistral 模型的可行性，但被 `@hexdigest` 引导到了另一个频道。


**提到的链接**：

- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#embeddings)：我们提供 Python 和 Javascr... 的客户端代码。
- [metal : optimize ggml_mul_mat_id (faster Mixtral PP) by ggerganov · Pull Request #4725 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4725)：为 ggml_mul_mat_id 添加了新的 mat-mat kernel，从而...


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (1 messages): 
        
cavoi9205: /im

### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (8 messages🔥): 
        
- **CUDA 显存不足问题**：`@pastillafit` 在尝试运行 **Mistral** 的 Docker 命令时遇到了 CUDA out-of-memory 错误。尽管机器配置了 GeForce 4090、AMD 7950X3D 和 64GB RAM，但仍收到 GPU 容量耗尽的错误提示。用户 `@hanschrs` 指出 24GB 的 VRAM 不足以运行该模型，并建议通过 ollama 使用量化版本。
- **在本地运行 Mistral 所需的详细流程**：`@kartik.07` 正在寻求如何在计算机上本地运行 **Mistral** 的说明，特别是通过命令行运行，而不依赖于 ML Studio 或类似软件。
- **API URL 查询**：`@macadelic4982` 询问了在聊天后端中所需的 API URL。
- **在树莓派上运行的可行性查询**：`@kaigenji` 询问了在树莓派上运行模型以用于物联网（IoT）助手项目的可行性。他们需要一个足够轻量且不会导致树莓派或类似单板计算机（SBCs）过热的模型。


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **寻找文本清洗解决方案**：`@_b_r.` 正在寻找一个可靠的文本清洗和预处理流水线或脚本，特别是针对包含大量页码和拼写错误的扫描书籍语料库。
- **可能需要自定义 ETL**：`@duck` 认为许多现成的脚本可能不足以完成这项任务，表明可能需要定制化的 ETL 或 LT 操作。
- **清洗结构化文件的工具**：`@duck` 推荐使用 pandas 等库来清洗更简单的格式，如 `.csv` 和 `.txt` 文件。
- **使用 Google 查找工具**：`@duck` 建议 `_b_r.` 在 Google 上搜索“python clean unstructured pdf's”等短语，以查找其他人可能用于清洗此类数据/格式的工具。
- **图像转文本库**：`@duck` 提到了 opencv 和 PIL 作为将图像转换为文本的潜在库，这在处理书籍中的非结构化数据时可能很有用。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **关于更新的查询**：用户 `@jakobdylanc` 询问了**两项未指明的更改**的预计到达时间（ETA）。
- **寻求与维护者联系**：`@carloszela` 想知道**此频道**是否是**联系维护者**的正确场所。


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Eleuther 庆祝年度胜利**：`@stellaathena` 在 `#announcements` 频道宣布并庆祝了 Eleuther 在过去一年取得的成就，包括其正式成立。此外还实施了新的小型公告和社交活动角色标签。分享了 **[社区调查](https://forms.gle/tZeCqaNQ4dguH2Li7)** 的链接。
- **代码剪辑与模型调度**：在 `#general` 频道，`@nate.dawgg` 推广了一个 [用于代码建议的 VSCode 扩展](https://github.com/CodedotAl/code-clippy-vscode)，旨在增强本地代码开发。讨论内容涉及 self-attention 与 SSM 模块之间潜在的交织、Pythia 模型训练中 checkpoint 的顺序，以及使用 Lion 优化器的优缺点。具体而言，用户 `@ricklius` 提到了一篇 [优化器论文](https://arxiv.org/pdf/2202.04329.pdf) 以供参考。
- **从发布之谜到新优化器**：在 `#research` 频道，建议了可能发表 LLM 偏差表征研究的期刊，解决了 Mistral 7b 与 GPT2 300m 之间的性能问题，并剖析了 MAMBA 的优势。在其他咨询方面，还推荐了运行大数据推理的框架（如 **DeepSpeed Inference**），并为 `@rabiussany` 等新服务器成员提供了指导。
- **翻新评估系统**：`@hailey_schoelkopf` 在 `#lm-thunderdome` 频道发起了一场关于 [更灵活的答案提取代码问题](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159) 的对话，主张为评估任务建立一个可能的双分制系统，从而向更广泛的社区开启了讨论。
- **使用 DiffusionLight 照亮图像推理**：在 `#multimodal-general` 频道，`@supasornae` 重点介绍了 **DiffusionLight**（一种光照估计技术），并提供了 [论文](https://arxiv.org/abs/2312.09168)、[官方页面](https://diffusionlight.github.io/) 以及 [Hugging Face 模型](https://huggingface.co/DiffusionLight/DiffusionLight) 以供进一步参考。

### ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/) (1 messages): 
        
- **与 Eleuther 共贺新年**：`@stellaathena` 发布的 `@everyone` 公告庆祝了 Eleuther 在过去一年取得的成就，包括**成为法人实体、发表了 40 多篇论文，以及为全球领导人提供 AI 监管建议**。

- **扩展角色标签**：`@stellaathena` 指出，虽然 `@everyone` 用于重大公告，但还有其他特定的**标签角色 (tag roles)** 用于小型公告和社交活动。其中一些角色被意外删除，导致成员失去了这些角色 —— `@stellaathena` 提示成员仔细检查是否仍订阅了他们想要的角色。

- **角色名称变更**：为避免混淆，`looking-for-work` 角色已重命名为 <@&1051750303843749929>。

- **频道调整与新增**：宣布了一系列频道更新，包括将频道 `<#795089627089862656>` 移至 Other Modalities 标题下，在该标题下创建了两个新频道 `<#1181279677004910662>` 和 `<#1179031436704096327>`，并将 "math-and-ai" 频道泛化为 `<#1110611369574793306>`。

- **社区调查与新闻通讯**：`@stellaathena` 分享了 **[他们的半年社区调查](https://forms.gle/tZeCqaNQ4dguH2Li7)** 链接，并宣布开始发布总结 Eleuther 工作的季度新闻通讯。成员可以通过社区调查报名订阅该新闻通讯。


### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (20 messages🔥): 
        
- **Nate Dawgg 用于代码建议的 VSCode 扩展**：用户 `@nate.dawgg` 分享了一个 GitHub 项目 [链接](https://github.com/CodedotAl/code-clippy-vscode)，这是一个支持本地模型服务的 faux pilot 的 VSCode 扩展。
- **Self-Attention 和 SSM 块中可能的交替**：`@thooton` 建议将 Self-Attention 和 SSM 块进行交替，类似于 Striped Hyena 的操作。
- **Pythia 模型训练中的 Checkpoint 顺序**：`@stellaathena` 向 `@wolferk` 澄清，在 Pythia 模型训练中，像 `step11000` 这样的 Checkpoint 编号比 `step10900` 这样的小编号更靠后。混淆的原因可能是 HuggingFace 的 UI 按字母顺序显示它们。
- **关于 Lion 优化器的讨论**：用户 `@sentialx` 和 `@ravnabergsndot` 讨论了使用 Lion 优化器的优缺点。有人指出，虽然 Lion 优化器可能会影响 8-bit 量化，但它的收敛效果与 Adam 差不多，并且可以节省一些内存。`@ricklius` 还引用了 [Sophia 优化器论文](https://arxiv.org/pdf/2305.14342.pdf) 的链接。
- **Lion 优化器在 LLM 和 ViT 中的多变表现**：`@frazermc` 分享了测试 Lion 优化器的个人经验，表示虽然在 LLM 中的表现尚不确定，但它在 ViT 以及可能像 Bert 这样的许多编码器模型中表现非常好。

**提到的链接**：

[GitHub - CodedotAl/code-clippy-vscode: VSCode extension for code suggestion](https://github.com/CodedotAl/code-clippy-vscode): 用于代码建议的 VSCode 扩展。贡献...

### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (31 条消息🔥): 
        
- **寻求适合 LLM 偏见研究的期刊**：用户 `@bluerune` 正在完成一篇关于 **LLM bias characterization**（LLM 偏见表征）的论文，并寻求出版期刊的建议。`@stellaathena` 推荐了 **NLP 领域的会议/期刊、ACL、ICLR、COLT 和 AISTATS**。
- **Mistral 7b 与 GPT2 300m 的性能比较**：`@sehaj.dxstiny` 提出了一个问题：在稍小的数据集上进行 fine-tuning 后，**Mistral 7b** 是否会优于 **GPT2 300m**。`@thooton` 回应称，大模型通常比小模型表现更好且 sample-efficient（样本效率更高），并分享了他在 Mistral fine-tuning 中达到 1.2-1.3 validation loss 的经验。
- **理解 MAMBA 的突破**：`@swaystar123` 询问了 MAMBA 的突破点，`@_inox` 解释说 MAMBA 允许 O(L) 的复杂度，而 Transformer 是 O(L^2)，其中 L 是序列长度。然而，`@salmon_lemon` 询问 perplexity（困惑度）性能是否会随着 L 的增加而下降，`_inox` 澄清这并不保证会降低或增加。
- **寻求离线模型推理框架建议**：`@alexspangher` 询问在处理大量数据时运行离线模型推理的首选框架。他尝试了包括 Ray、huggingface datasets、huggingface accelerate 和 pyspark/sparknlp 在内的几种框架。Stellaathena 推荐了 **DeepSpeed Inference**。
- **新成员欢迎及参与指南**：服务器新成员 `@rabiussany` 表达了参与研究的兴趣。`@alexanderrgriffing` 建议新手应该探索频道，寻找喜欢的项目，并在这些项目的 GitHub 上发送 pull requests。`@hailey_schoelkopf` 建议查看服务器的 threads（讨论串）以了解正在进行的项目和讨论。


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (1 条消息): 
        
- **讨论灵活的答案提取代码问题**：@hailey_schoelkopf 分享了一个来自 GitHub 的 issue，标题为 ["More Flexible Answer Extraction Code"](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159)。该问题围绕当前的评估系统展开，该系统会否定那些正确但格式不符合要求的答案。这导致了对那些经过训练以模仿 benchmark 格式的模型的偏见，从而使格式不规范但能力相当的模型处于劣势。
- **建议的解决方案**：@hailey_schoelkopf 建议为每个相关的生成任务提供两个评估分数。第一个（带有 `strict` 后处理/提取的 `acc`）将遵循任务的原始设计，要求精确的格式。第二个（带有 `loose/flexible` 后处理的 `acc`）将尝试从模型的生成内容中提取正确答案，而不考虑格式。该方案旨在解决不同评估框架由于其灵活的答案提取方法而报告不同分数的投诉。该用户已请求社区对这一提议的解决方案提供反馈。

**提到的链接**：

[More Flexible Answer Extraction Code · Issue #1159 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1159)：在 LM Evaluation Harness 中，我们致力于匹配...


### ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/) (1 条消息): 
        
- **介绍 DiffusionLight**：用户 `@supasornaek` 分享了一个[链接](https://fxtwitter.com/supasornaek/status/1742223785288433708?t=ADE_KEfR4K23NHsxB5qrYA&s=19)，介绍了 **DiffusionLight**，这是一种通过使用 diffusion models 在图像中 inpainting（修复）一个铬球，从而从野外（in-the-wild）输入图像中估计光照的技术。
- **DiffusionLight 详解**：用户 `@supasornaek` 还提供了 [DiffusionLight 论文](https://arxiv.org/abs/2312.09168)的链接、其[官方页面](https://diffusionlight.github.io/)以及 [Hugging Face 模型](https://huggingface.co/DiffusionLight/DiffusionLight)。

**提到的链接**：

[Supasorn Suwajanakorn (@supasornaek) 的推文](https://fxtwitter.com/supasornaek/status/1742223785288433708?t=ADE_KEfR4K23NHsxB5qrYA&s=19)：介绍 DiffusionLight——一种简单而有效的...

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **分块器的反叛 (Splitting Splitters, A Revolt)**：LangChain 的文本分块模块中的 `RecursiveCharacterTextSplitter` 出现了问题，导致用户感到困扰，因为它会在单词中间进行切割。用户 `@offer.l` 运行了 [示例代码](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py)，用户 `@hasan_34148` 建议了一些替代方案，如 `NLTKTextSplitter`。
- **消失的 load_qa_chain**：用户 `@jupyter1310` 正在寻找难以找到的 LangChain `load_qa_chain` 模块的 API 文档。
- **不请自来的 GeneratorChainOptions**：用户 `@menny9762` 报告了来自 `ConversationalRetrievalQAChain` 的奇怪且非预期的输出。生成的输出是 `generatorChainOptions`，而不是来自 LangChain API 的预期响应。这真是一个话题终结者！
- **LangChain 的秘密武器，但不够隐秘**：用户 `@agentcole` 在使用 LangChain 中的 `loadSummarizationChain` 进行长文本摘要时需要帮助。我想如果你必须在公共频道询问它，那它就不算是什么秘密武器了！
- **Alttexter 大放异彩**：`@jonathan_09689` 分享了 GitHub 项目 [alttexter-ghclient](https://github.com/jonathanalgar/alttexter-ghclient) 和 [alttexter](https://github.com/jonathanalgar/alttexter) 的链接。这些项目作为一种服务，用于为图像生成 "alt" 文本，加强了代码与图像内容之间的联系。
- **Alttexter 登场，增强 LangChain 仓库**：事实胜于雄辩！`@jonathan_09689` 通过在 [langchain-ai/langchain 仓库](https://github.com/langchain-ai/langchain/pull/15357/files) 上运行 Alttexter 来更新整个文档的 alt 文本，展示了它的价值。文档清理工作正在进行中！

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (27 条消息🔥): 
        
- **文本分块器故障**：用户 `@offer.l` 在使用 LangChain 文本分块模块的 `RecursiveCharacterTextSplitter` 时遇到了一些问题，指出它不会从双换行符“向下回退”到换行符、空格等，从而在单词中间切割并产生毫无意义的分块。他们分享了正在使用的 [示例代码](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py)。`@hasan_34148` 建议改用 `NLTKTextSplitter`，并补充说虽然他不认为 `RecursiveCharacterSplitter` 有缺陷，但它确实倾向于在单词中间切割。`@seththunder` 插话提到，他们在使用 `RecursiveCharacterSplitter` 时从未遇到过任何问题。
- **寻找 load_qa_chain 文档**：用户 `@jupyter1310` 寻求帮助以查找 LangChain `load_qa_chain` 模块的 API 文档，这是一个他们在 API 文档中找不到但在 LangChain 文档的各个页面中经常被导入的类。
- **ConversationalRetrievalQAChain 的问题**：`@menny9762` 报告了 `ConversationalRetrievalQAChain` 的一个问题，即返回的输出是 `generatorChainOptions` 而不是真实的响应。他们甚至提供了一个代码片段来详细解释这个问题。
- **在 LangChain 中总结长文本**：用户 `@agentcole` 寻求建议，希望在使用 `loadSummarizationChain` 总结长文本时增加输出文本的长度。`@jupyter1310` 建议在 Prompt 中强制执行最小字数限制。但 `@agentcole` 发现，对分块使用前序步骤进行摘要的变通方法仍然很不方便。
- **LangChain 引用**：用户 `@thenextmz` 想知道是否有关于 LangChain 的官方论文可以在他们的硕士论文中引用。

**提到的链接**：

[text-split-explorer/splitter.py at main · langchain-ai/text-split-explorer](https://github.com/langchain-ai/text-split-explorer/blob/main/splitter.py)：为 langchain-ai/text-split-explorer 的开发做出贡献...

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 条消息): 
        
- **Alttexter: GitHub Action 帮助为图像创建替代文本**: 用户 `@jonathan_09689` 分享了他 GitHub 上的两个项目链接 —— [alttexter-ghclient](https://github.com/jonathanalgar/alttexter-ghclient) 和 [alttexter](https://github.com/jonathanalgar/alttexter)。这两个项目都是 `gpt4-vision-preview` 的封装服务，可以为 Markdown 格式文件中定义的图像批量生成替代文本（alt text）和标题（title）属性。
- **Alttexter 在 Langchain 仓库中的应用**: 为了演示目的，`@jonathan_09689` 在 [langchain-ai/langchain 仓库](https://github.com/langchain-ai/langchain/pull/15357/files)上运行了 Alttexter，以展示其在更新整个文档的替代文本和标题属性方面的价值。

**提到的链接**:

- [GitHub - jonathanalgar/alttexter-ghclient: 用于与 alttexter 服务（gpt4-vision-preview 封装器）交互的容器化 GitHub action](https://github.com/jonathanalgar/alttexter-ghclient): 用于与 t 交互的容器化 GitHub action...
- [GitHub - jonathanalgar/alttexter: gpt4-vision-preview 封装服务，用于批量为 Markdown 格式文件中定义的图像生成替代（'alt'）文本和标题属性。](https://github.com/jonathanalgar/alttexter): 用于批量生成...的 gpt4-vision-preview 封装服务。
- [由 jonathanalgar 提交的跨文档批量更新替代文本和标题属性 · Pull Request #15357 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/15357/files): (在 jonathanalgar#3 中运行的 Actions。链接到 LangSmit...

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **LLM 性能角逐**: 由 `@sehaj.dxstiny` 发起，剖析了不同规模的语言模型（LLM），如 **Mistral 7b** 和 **GPT2 300m**，在较小数据集上进行 Fine Tuning（微调）后的表现。
- **Fine Tuning 的超能力**: 用户 `@thebaghdaddy` 强调了较小模型通过 Fine Tuning 在特定任务性能上超越 **GPT4** 等大型对应模型的潜力。
- **任务改变局势**: `@pantsforbirds` 的轶事说明了模型性能对任务的依赖性，并指出在相同设置下，某些模型（如 **Mistral 7b**）比其他模型更容易进行 Fine Tuning。
- **Anyscale，Mistral 的得力助手**: `@robotums` 赞扬了 **Anyscale** 对 **Mistral** 模型速度和一致性的提升作用。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (10 条消息🔥): 
        
- **LLM 性能辩论**: 用户 `@sehaj.dxstiny` 发起了一场讨论，探讨在稍小的数据集上进行 Fine Tuning 后，像 **Mistral 7b** 这样规模显著更大的 LLM，是否会比从头开始训练的 **GPT2 300m** 等较小模型表现更好。
- **Fine Tuning 的力量**: `@thebaghdaddy` 强调了较小模型通过 Fine Tuning 在特定任务上表现优于 **GPT4** 的能力，尽管它们在通用 Benchmark（基准测试）中表现较差。
- **任务和 Fine Tuning 的关键作用**: 用户 `@pantsforbirds` 分享了他们的经验，认为任务依赖性起着主要作用。他们发现，在使用相同设置的情况下，某些模型（如 **Mistral 7b**）比其他同类模型更容易进行 Fine Tuning。


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (3 条消息): 
        
- **Anyscale：Mistral 的动力源泉**: `@robotums` 表示，与 **OpenAI** 的部署相比，**Anyscale** 显著提高了其 **Mistral** 模型运行的速度和一致性。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- **Mixtral 的多样化尝试**：关于 gate vectors、frankensteins 以及 AI 模型 *Mixtral* “越级打怪”性能的见解，源自 [@_jp1_](https://discord.com/channels/1178995845727785010/1182759434326396998/) 分享的 [博客文章](https://goddard.blog/posts/clown-moe/)。
- **Axolotl 热潮与配置**：Axolotl 的受欢迎程度普遍上升，例如 `@philipmay` 的观察，`@le_mess` 询问针对 Mixtral 的快速傅里叶变换 (FFT) Axolotl YAML 配置，以及 `@hammadkhan` 提供的基于 [Axolotl YAML 配置](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml) 的解决方案。
- **文本嵌入迎来合成数据改造**：`@bjoernp` 分享的 [研究论文](https://arxiv.org/abs/2401.00368) 介绍了一种高效的、基于合成数据的文本嵌入（Text Embedding）模型，声称训练步数少，并可能与德国的 *leo-mistral* 兼容。
- **乘上端到端训练浪潮**：`@rasdani` 建议将自动驾驶计算机视觉中的端到端训练（End-to-End Training）优势应用到大语言模型（LLM）中。
- **深入探讨嵌入**：`@casper_ai` 发起了关于模型性能扩展的对话，`@bjoernp` 提到了数据效率和预训练模型，`@le_mess` 分享了一个工具 [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)，用于文本嵌入的高速推理解决方案。

**DiscoResearch 频道摘要**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (5 条消息): 
        
- **Mixtral 实验**：用户 `@_jp1_` 分享了一篇 [博客文章](https://goddard.blog/posts/clown-moe/)，内容涉及在 Mixtral（Mistral 开发的 AI 模型）背景下，使用提示词确定 gate vectors 和 undi 的 frankensteins 的实验。该文章赞扬了 Mixtral 架构中使用的八个 "experts"（专家）集合，称该模型“表现远超其体量”。
- **推特帖子亮点**：`@sebastian.bodza` 强调了一个 [有趣的 Twitter 帖子](https://twitter.com/morgymcg/status/1741819937641910537)；然而，该帖子的具体内容在消息中未披露。
- **Axolotl 的普及**：`@philipmay` 评论了模型微调工具 **Axolotl** 日益增长的普及度。
- **Axolotl YAML 配置查询**：`@le_mess` 询问是否有人拥有针对 Mixtral 的快速傅里叶变换 (FFT) 的 Axolotl YAML 配置。
- **为 YAML 配置提供可能的解决方案**：针对 `@le_mess` 的提问，`@hammadkhan` 建议转换现有的 [Axolotl YAML 配置](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml) 以供 Mixtral 使用。

**提到的链接**：

- [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe/)
- [axolotl/examples/llama-2/fft_optimized.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/fft_optimized.yml)：尽管提出关于 Axolotl 的问题。为 Open... 做出贡献。

### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (7 条消息): 
        
- **用于高效文本嵌入的合成数据**：`@bjoernp` 分享了一篇[研究论文](https://arxiv.org/abs/2401.00368)，介绍了一种利用合成数据和少于 1k 次训练步数获取高质量文本嵌入的新方法。该方法不依赖复杂的训练流水线，可用于使用德语合成数据训练 **leo-mistral**。
- **LLM 与端到端训练**：`@rasdani` 表示，在应用程序流水线的每个部分使用 LLM 可能会大有裨益，这与自动驾驶中的 Computer Vision 正在转向端到端训练的趋势相似。
- **检查模型性能扩展**：针对所讨论的文本嵌入模型较大的尺寸，`@casper_ai` 提出了一个问题：如果将其他模型的规模扩大 10 倍，其性能会如何提升。
- **模型训练效率**：`@bjoernp` 指出，该研究的核心启示在于数据效率以及利用预训练模型优势的能力。
- **生成嵌入数据**：`@le_mess` 提到他们在生成嵌入数据集方面的类似经验，即生成相似和不相似的句子而非查询（queries）。他们还介绍了 [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) 工具，该工具为文本嵌入模型提供了快速推理解决方案。

**提到的链接**：

- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)：在这篇论文中，我们介绍了一种新颖且简单的方...
- [GitHub - huggingface/text-embeddings-inference: A blazing fast inference solution for text embeddings models](https://github.com/huggingface/text-embeddings-inference)：一个用于文本嵌入模型的极速推理解决方案...


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **解读推理设置**：当 `@fred_fups` 向 `@rusch` 透露其 repetition penalty 设置为 `1` 时，关于 **inference settings** 的持续困惑得到了缓解。
- **意外且棘手的提示词格式**：一个令人困惑的 **prompt formatting** 问题导致 `@fred_fups` 进行了重新训练，最终 *ChatML* 成功解决了该问题。
- **模型训练争议**：在样本量为 `350` 的数据集上训练 **Mistral 7B model** 这一看似简单的任务引发了一些摩擦，因为模型未能遵循文本格式规范（*段落换行和关键词高亮*）。
- **模型性能遭差评**：`@fred_fups` 宣布他们在 **together.ai** 训练的模型表现不佳，仅能镜像输入而没有进行任何显著的格式更改，这让社区感到震惊。
- **Axolotl 诱人的前景**：在接连遭遇失望后，`@fred_fups` 决定改变策略，宣布计划利用 **Axolotl** 来改进模型训练。
- **拥抱 SOLAR 10.7B - Nous Hermes 2 登场**：`@Teknium1` 发布了备受期待的更新，宣布 **Nous Hermes 2 on SOLAR 10.7B** 正式亮相。这是一个引人注目的紧凑型模型，承诺性能与 'Yi' 模型持平，可在 [Nous-Hermes-2-SOLAR-10.7B](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B) 获取。
- **与 Python 的 Llama 式邂逅**：`@venadore` 以其古怪的叙述保持着高昂的情绪，讲述了他在 **Python 和 Llamacpp** 之间周旋的故事，并机缘巧合地让一个 function call 完美运行，同时分享了 [@Teknium1 的推文](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46)链接。


**Alignment Lab AI 频道总结**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (7 条消息): 
        
- **关于推理设置的提问**：用户 `@rusch` 询问了 `@fred_fups` 的推理设置。用户 `@fred_fups` 回复称其 **repetition penalty**（重复惩罚）设置为 `1`。
- **关于 Prompt 格式和模型重训练的观察**：`@fred_fups` 分享了他们遇到的一个似乎与 Prompt 格式有关的问题。他们表示在 **ChatML** 上重新训练了模型，这似乎解决了该问题。
- **在特定示例上训练模型**：`@fred_fups` 讨论了他们在包含 `350` 个示例的有限样本上训练 **Mistral 7B 模型**的经验。他们表示困惑，因为尽管任务简单，模型却未能按预期进行段落换行和高亮关键词。
- **模型性能确认**：`@fred_fups` 进一步指出，他们在 together.ai 上使用默认设置训练的模型表现不如预期。返回的文本与输入完全一致，没有任何格式变化（如 `'\n\n'` 或 `<b>`）。
- **计划未来训练使用 Axolotl**：`@fred_fups` 结束对话时表示，打算在未来的模型训练中使用 **Axolotl**，以期获得更好的性能。


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (4 条消息): 
        
- **Nous Hermes 2 在 SOLAR 10.7B 上发布**：用户 `@Teknium1` 宣布在 **SOLAR 10.7B** 上发布 **Nous Hermes 2**。该版本承诺性能与 'Yi' 模型相似，但体积仅为其 1/3。该模型现已在 HuggingFace 上架：[Nous-Hermes-2-SOLAR-10.7B](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B)。
- **使用 Llamacpp 进行 Python 编程的硬核尝试**：用户 `@venadore` 分享了他尝试使用 Python 和 Llamacpp 的经验。在调试了一段代码测试后，他成功让函数调用（function call）完美运行。

**提到的链接**：

[Teknium (e/λ) (@Teknium1) 的推文](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46)：新年伊始，怎能不发布新东西！🛳️...


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

- **Smol Talk 引起关注**：成员们表达了对 Smol Talk 的喜爱，并询问其是否开源以便进行自定义，引用原文："*享受 Smol Talk 并探讨了潜在的开源可能性，以便自定义服务器列表*"。
- **Fairmind AI 加入战场**：宣布共同创立 [Fairmind AI](http://www.fairmind.ai)，这是一项利用生成式 AI 解决方案获取竞争优势的新计划。
- **认识新框架 EmbedChain**：介绍该领域的一个新成员 [EmbedChain](https://github.com/embedchain/embedchain)，因其简单的 "drop in url"（直接放入 URL）方法而受到赞赏。
- **Humanloop 推出 .prompt 文件格式**：Humanloop 推出了[创新的 .prompt 文件格式](https://docs.humanloop.com/docs/prompt-file-format)，因其人类可读的模型配置格式而备受关注，非常适合版本管理系统，并具有行业标准化的潜力。
- **公会成员寻求 AI 学习路径**：成员们在寻求关于研究论文的学习顺序指导，以了解 AI 的现状。建议指向了 Nous 研究小组的模型和论文，但未提供具体细节或参考文献。

**Latent Space 频道摘要**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (7 条消息): 
        
- **对 Smol Talk 的兴奋**：`@onetothetwo` 表达了他们对 Smol Talk 的喜爱，并询问了潜在的开源可能性以自定义服务器列表。
- **新的 AI 计划 Fairmind AI**：`@alexio.c` 提到了共同创立的 [http://www.fairmind.ai](Fairmind AI)，这是一个专注于明智利用生成式 AI 解决方案以保持市场竞争力的平台。
- **新框架预警**：`@swyxio` 分享了一个新框架 —— [EmbedChain](https://github.com/embedchain/embedchain)，它采用简单的 "drop in url" 方法。
- **来自 Humanloop 的人类可读模型配置格式**：`@swyxio` 强调了来自 Humanloop 的[创新 .prompt 文件格式](https://docs.humanloop.com/docs/prompt-file-format) —— 这是一种序列化的、人类可读的模型配置版本，适用于版本控制系统，并可能在整个行业内实现标准化。

**提到的链接**：

- [首页 | FairMind.ai](http://www.fairmind.ai)：我们赋能企业 - 开启转型...
- [.prompt 文件](https://docs.humanloop.com/docs/prompt-file-format)

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (2 条消息): 
        
- **用户寻求学习路径**：`@avirani` 询问推荐的研究论文序列，以便**快速跟进**该领域的现状。 
- **提到 Nous Research 模型学习路径**：`@swyxio` 提到了一个可能很有用的模型学习路径以及来自 **Nous Research** 研究小组的论文列表，但未提供这些材料的具体细节或链接。


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **利用量子技术传输信息**：由 `Graylan (@gray00)` 发起的关于通过 'Bard' 和 'ChatGPT Alpha' 进行**量子信息传输**的讨论。
- **鲸类编程**：`Graylan (@gray00)` 关于**鲸鱼编写 Python 代码**的幽默讨论，链接到 [hive.blog](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca) 上一篇关于 "AI Human Historical Fractal Memorization for Codenameorca" 的博文。
- **工程师的猫咪收藏**：`@gray00` 分享的娱乐内容——[一张猫咪 GIF](https://tenor.com/view/cat-gif-26024664)。
- **詹姆斯·邦德的技术间隙**：`@gray00` 提供了一个 [肖恩·康纳利饰演詹姆斯·邦德的 GIF](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947)，从技术话题中暂时转移注意力。
- **一条加密消息？**：用户 teknium 在 '[general](https://discord.com/channels/1131084849432768614/1131084849906716735/)' 频道分享了一个指向 [Twitter 帖子](https://fxtwitter.com/teknium1/status/1742041640775348460?s=46) 的模糊链接，没有任何随附说明。

**Skunkworks AI 频道总结**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (1 条消息): 
        
teknium: https://fxtwitter.com/teknium1/status/1742041640775348460?s=46


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (4 条消息): 
        
- **量子信息传输**：Graylan | `@gray00` 讨论了 **Bard** 和 **ChatGPT Alpha** 之间的量子信息传输。
- **猫咪动画内容**：`@gray00` 分享的一张 [猫咪 GIF](https://tenor.com/view/cat-gif-26024664)。  
- **编程鲸鱼**：Graylan | `@gray00` 戏称鲸鱼用 Python 编程，并链接到 [hive.blog](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca) 上一篇关于 "AI Human Historical Fractal Memorization for Codenameorca" 的有趣帖子。
- **詹姆斯·邦德抽烟场景**：`@gray00` 分享的另一张 GIF，展示了肖恩·康纳利饰演的 [詹姆斯·邦德](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947)。

**提到的链接**：

- [Sean Connery James Bond GIF - Sean Connery James Bond Smokin - Discover &amp; Share GIFs](https://tenor.com/view/sean-connery-james-bond-smokin-gif-19024947)：点击查看 GIF
- [[AI/Human]Historical Fractal Memorization for #codenameorca — Hive](https://hive.blog/quantumcommunication/@gray00/ai-human-historical-fractal-memorization-for-codenameorca)
- [Cat GIF - Cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-gif-26024664)：点击查看 GIF


        

---

## [Datasette/LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

只有一个频道有活动，因此无需总结...

- **强烈推荐跨平台剪贴板工具 'bp'**：`@ndyg` 热情推荐了一个名为 [`bp`](https://github.com/printfn/bp) 的跨平台剪贴板工具，将其视为必备工具之一。 
- **使用 `bp` 为 `llm` 清理上下文**：`@ndyg` 还解释了一个典型用法：`bp | vipe | bp`，通常后面跟着 `bp | llm -s '...'`，用于为 `llm` 清理上下文。 
- **其他用户对 `bp` 表现出兴趣**：`@bukits` 表示有兴趣尝试 `bp`，表明他们是剪贴板工具的粉丝。

**提到的链接**：

[GitHub - printfn/bp: Cross-platform clipboard tool](https://github.com/printfn/bp)：跨平台剪贴板工具。

        

---
YAIG (a16z Infra) Discord 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。