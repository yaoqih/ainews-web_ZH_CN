---
companies:
- microsoft-research
- mistral-ai
- apple
- amd
date: '2023-12-29T10:14:01.623905Z'
description: 'LM Studio Discord 社区对**模型性能**对比进行了广泛讨论，特别是**微软研究院**开发的 **Phi2** 与 **OpenHermes
  2.5 Mistral 7b** 之间的对比，重点关注了**美国历史知识**以及通过微调提高准确性的方法。


  技术方面，讨论涉及了围绕 **LLM API** 使用、对话历史维护以及针对推理速度的 **GPU 优化**等挑战。硬件讨论涵盖了 **DDR4 与 DDR5**
  的对比、多 GPU 配置，以及 **Apple M1/M3** 和 **AMD AI CPU** 在处理 AI 工作负载方面的潜力。


  社区还宣布发布了 **ChromaDB 插件 v3.0.2**，该版本支持在向量数据库中进行图像搜索。此外，用户们还分享了关于运行多个 LM Studio 实例以及优化资源使用的实用技巧。'
id: 9aafab3a-27e5-44fa-bc6d-e939622430e1
models:
- phi2
- openhermes-2.5-mistral-7b
- llama-2-7b
- llama-2-13b
original_slug: ainews-12272023-nyt-vs-openai
people: []
title: 2023年12月27日：纽约时报 诉 OpenAI
topics:
- model-performance
- fine-tuning
- llm-api
- gpu-optimization
- hardware-configuration
- multi-gpu
- inference-speed
- plugin-release
- conversation-history
---

<!-- buttondown-editor-mode: plaintext -->这是关于今天 NYT OpenAI 诉讼案的[最佳推文串](https://twitter.com/ceciliazin/status/1740109462319644905)：

 
![image.png](https://assets.buttondown.email/images/8500cd54-405d-49fb-bcf5-d827b22e17c0.png?w=960&fit=max)
 

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- **模型性能**：围绕不同背景下的模型性能进行了广泛对话，特别关注 Phi2 的美国历史知识、OpenHermes 2.5 Mistral 7b 与 Phi2 的对比，以及 LLama 2 chat 7b 与 13b 的使用体验。强调了通过对现有模型进行 fine-tuning 以提升性能的潜力。
- **技术挑战与解决方案**：关于 LLM API 内部技术挑战以及软件和配置相关问题的反复讨论。显著的挑战包括在 LLM API 中维护对话历史、优化 GPU 层设置以提高推理速度、模型配置、错误处理，以及 LM Studio 的 ChromaDB Plugin 的安装和调试。
- **模型使用与选择**：讨论了具体的使用案例，如论文评分、家庭友好型聊天机器人等，以及基于性能和模型 hallucination 事件选择合适的模型。还讨论了 LM Studio 中“assistant”角色的作用。
- **硬件讨论**：深入讨论了硬件选择和优化以获得更好的模型性能。主题涵盖 DDR4 与 DDR5 的配置差异、用于 Local Light Models 和 Stable Diffusion 实验的显卡选择、构建配备多块 GPU 的 AI 服务器机架，以及 Apple 新款 M1 和 M3 芯片和 AMD 的 AI CPU 运行 AI 模型的潜力。
- **社区与插件更新**：展示了社区在处理配置、软件错误和插件相关问题时的集体知识共享和解决问题的能力。值得注意的是，发布了 [LM Studio 的 ChromaDB Plugin 新版本](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2)，该版本允许在向量数据库中进行图像搜索。

**LM Studio 频道摘要**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (284 条消息🔥🔥): 
        
- **Phi2 模型性能**：`@dharmabum125` 和 `@heyitsyorkie` 讨论了 Microsoft Research (MR) 开发的 Phi2 模型在处理美国历史知识方面的表现。`@dagbs` 为 Phi2 辩护，引用了一篇博客文章，解释了 MR 团队旨在让 Phi2 在代码、推理和常识性知识方面表现出色，而不一定是在历史方面。
- **模型性能对比**：`@heyitsyorkie` 对比了 OpenHermes 2.5 Mistral 7b 和 Phi2 的性能，认为前者可以在第一次尝试时正确回答特定的历史问题。
- **提升 Mistral Hypermodel 的性能**：`@dagbs` 建议通过向 Mistral 喂入历史数据集进行 finetuning，以提高其回答历史相关问题的能力。
- **关于 LM Studio 的问题**：用户 `@code_3090` 和 `@queuelabs` 分别询问了如何使用 LM Studio LLM API 维护对话历史，以及 LM Studio 采用了哪些优化技术。`@thelefthandofurza` 为 `@code_3090` 的问题提供了解决方案，解释说需要将之前的消息附加到新消息中以保留对话上下文。
- **性能疑虑**：`@sheepdestroyer` 反馈在不同模型和设置下推理速度持续偏低。`@heyitsyorkie` 建议尝试更小的模型，并评论说所报告的速度与在 `@sheepdestroyer` 指定的硬件配置上运行 34b 模型的情况一致。
- **多 GPU 使用**：用户 `@rugg0064`、`@fabguy` 和 `@psipiai` 讨论了如何使用多块 GPU 运行模型。他们一致认为，将层的计算分布到多个 GPU 上并不会缩短运行时间。
- **同时运行多个模型**：`@yoangab` 找到了在 Mac 上运行多个 LM Studio 实例以同时操作多个服务器的方法。分享的命令是 `open -n LM\ Studio.app/`。
- **推理速度调整**：由 `@yunta0` 请求调整 GPU 层和 CPU 线程设置以优化推理速度而引发的讨论，最后由 `@ptable` 建议调整 GPU 层以实现约 80-90% 的 VRAM 占用率。
- **LM Studio 的 Chroma 数据库插件新版本**：`@vic49.` 发布了 LM Studio 的 ChromaDB Plugin 新版本。

### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (43 messages🔥): 
        
- **在 LM Studio 对话中清除 Context**：`@the.one.n.only.prof` 询问如何在 LM Studio 的持续聊天对话中清除 Context。`@fabguy` 指出在同一个聊天窗口中无法实现，清除 Context 的唯一方法是开启新聊天。
- **LM Studio 中的多 GPU 及 GPU 偏好设置**：`@septagraam` 寻求关于在多 GPU 平台上如何将特定模型分配到运行 LM Studio 系统的单个 GPU 上的说明。他提到了 "Open GPU Preferences JSON" 选项，但不清楚其具体语法。`@fabguy` 建议在预设 JSON 中自定义 `tensor_split` 值，建议设置为 `100,0` 以将所有 Layers 运行在第一个 GPU 上。
- **使用 LLM 批改论文**：英语教师 `@emerance` 寻求关于利用语言模型批改论文的建议。`@fabguy` 和 `@thelefthandofurza` 建议制定明确的客观评分准则，并限制 Context 以根据评分量表评估每篇论文。为了提高评分的一致性，`@fabguy` 建议降低模型的 Temperature。此外，`@yagilb` 建议将评分任务分解为更小、更易管理的部分，以提高效果。
- **LM Studio 中的 Hugging Face 模型问题**：`@jvaleski` 和 `@madbits.` 分别在 LM Studio 中搜索和加载 Hugging Face 模型时遇到困难。`@heyitsyorkie` 澄清说某些模型版本在 Linux 0.2.8 构建版本中不受支持，并提供了 0.2.10 版本的链接，同时提醒可能存在稳定性错误。
- **通过 LM Studio API 获取 Embeddings**：`@johnv_jv` 询问 LM Studio API 是否支持为 RAG 实现获取 Embeddings。`@vic49.` 给予了否定回答，但强调了可以整合来自向量数据库的原始文本 Embeddings 的能力。
- **LM Studio 中 'Assistant' 角色的作用**：`@sladix` 询问 LM Studio 模型中 'Assistant' 角色是如何使用的。`@yagilb` 和 `@borick` 分别建议该角色用于在 Context 中植入助手消息以及自定义系统行为。


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (41 messages🔥): 
        
- **Hermes 2.5 vs Hermes 2 性能讨论**：用户 `@clickclack777` 和 `@dagbs` 讨论了不同模型版本的性能和大小。`@clickclack777` 提到他们发现 0.2.10 中默认的 Phi-2 模板与 Dolphin prompt 结合使用效果很好。`@dagbs` 期待 Mistral 7b 的发布，预计其体积会小得多（约 8GB 而非约 25GB）。
- **Llama 2 chat 7b vs 13b 讨论**：`@brunodanuy` 正在询问适合家庭友好型聊天机器人的最佳模型。`@heyitsyorkie` 建议使用默认的 Llama 2 chat 7b，然而 `@brunodanuy` 已经在使用 13b 版本并对其性能表示满意。随后对话转向探索响应生成速度，他们考虑使用更小尺寸的模型以获得更高效的生成时间。
- **Phi 2 模型性能讨论**：`@brunodanuy` 表示有兴趣尝试 Phi 2 模型，因为它的体积比 Llama 13b 更小。`@heyitsyorkie` 警告说，虽然它体积更小，但可能不推荐用于大多数任务，结果可能比较“平庸”。`@brunodanuy` 尝试了该模型，并同意其响应与 Llama 13b 相比不那么令人满意。
- **GPU 加速与模型幻觉讨论**：`@rarisma` 询问 Phi 是否支持 GPU 加速/Layers，以及 GPU 加速是否会导致幻觉（Hallucination）。`@heyitsyorkie` 认为这可能只是模型本身的问题，Phi-2 并不十分强大，容易产生混淆。不过，根据 `@heyitsyorkie` 的说法，将 Layers 数量降低到 5 层左右可能有助于解决幻觉问题。
- **关于 AWQ 模型**：`@xsnypsx` 询问现在或不久的将来是否可以运行 AWQ 模型，`@heyitsyorkie` 回复称 LM Studio 目前不支持 AWQ，但随着 llama.cpp 的 PR 合并到主分支，未来可能会支持。

### ▷ #[🛠-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/) (26 条消息🔥): 
        
- **模型配置**：`@pdg` 分享了 [OpenChat 3.5 1210 Starling SLERP - GGUF](https://huggingface.co/TheBloke/openchat-3.5-1210-starling-slerp-GGUF) 模型的配置，该模型似乎运行良好。

- **特定模型配置问题**：`@pdg` 强调了某些模型在配置时遇到的问题，特别是 [Finance LLM - GGUF](https://huggingface.co/TheBloke/finance-LLM-GGUF) 和 [Solar 10.7B Instruct V1.0 Uncensored - GGUF](https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF)。主要的挑战在于配置 Prompt 格式以获得正确的响应。

- **模型错误处理**：`@heyitsyorkie` 也报告了在使用已包含的 llama-2-chat 预设时遇到了类似的错误，`@pdg` 确认也遇到了同样的问题。

- **硬件优化问题**：`@badnature` 提出了一个问题，尽管有 32 GB 的 VRAM 可用，但在 7B Mistral 模型上性能较低。`@fabguy` 建议检查模型是否分布在多张显卡上，并调整 `tensor_split` 变量。

- **优化推理速度**：`@badnature` 随后报告称，在修改加载参数后，推理速率从 9tk/sec 提高到了 13 tk/sec。讨论了进一步提高推理速度的方法，例如使用更低的量化（quants）、调整 CPU 线程以及更新驱动程序。`@fabguy` 还提到使用更小的 Context Size 会带来好处。


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (53 条消息🔥): 
        
- **ChromaDB-Plugin-for-LM-Studio V3.0.2 发布**：`@vic49.` 分享了 [新版本 ChromaDB 插件发布的消息](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2)，该版本允许在 LM Studio 的向量数据库中进行图像搜索。
- **插件的安装与调试**：用户 `@motocycle` 在他们的 Mac 上下载并安装了该插件，在设置过程中遇到了几个错误。`@vic49.` 指导他们进行了调试，并建议在 `initialize.py` 脚本中添加 `"import os"` 一行来解决其中一个错误。
- **Torch 后端错误**：`@motocycle` 遇到了另一个与 Torch 后端相关的问题，`@vic49.` 建议替换 `loader_vision_llava.py` 中的 `get_best_device` 函数来修复它。然而，这导致了另一个关于 'boolean' 对象不可调用的错误。
- **GPU 与设置查询**：`@vic49.` 强调在设置中选择 4-bit 可能会导致问题，因为它严重依赖 `bitsandbytes` 库，而该库可能默认使用 GPU；建议在 "quant" 选项中尝试 float16，以便让它在原生 Pytorch 上运行。
- **讨论转至私信 (DMs)**：`@yagilb` 请求 `@vic49.` 和 `@motocycle` 将深入的调试转移到私信中，他们表示同意。


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (110 条消息🔥🔥): 
        
- **推理速度与内存**：用户 `@dagbs fabguy` 讨论了 GPU 推理速度的部分原因在于更快的内存，并提到了 DDR4 和 DDR5 之间的差异。该评论引发了关于不同硬件配置对速度和性能影响的讨论。
  
- **AI 用途的硬件选择**：用户 `@Pierre-jean Lainé` 询问在特定预算限制下的硬件选择建议。根据 AI 任务的性能进行了对比，例如 **本地轻量级模型** (LLMs) 和 **Stable Diffusion** 实验。多位用户（`@heyitsyorkie`, `@rugg0064`）推荐了 RTX 3060 12GB 版本，因其在 LLM 使用中具有卓越的性能。

- **构建具备 AI 能力机架的潜力**：用户 `@xenorhon` 和 `@heyitsyorkie` 探讨了构建一个带有多个 GPU 卡的服务器机架来运行 AI 模型的想法。讨论了 Nvidia Ada 6000 和 RTX 3090 模型作为潜在选项。

- **关于苹果 M1 和 M3 芯片用于 AI 的讨论**：`@rugg0064` 讨论了苹果新款 M1 和 M3 芯片运行 AI 模型的潜力，指出由于这些芯片具有大内存容量和高带宽，是运行模型的一种高性价比方式。

- **AMD 的 AI 专用 CPU**：`@totallybored` 分享了 [AMD AI CPU](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) 的链接，并评论称正在等待下一代 AI 专用笔记本电脑。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- 关于 OpenAI 服务成本效益的辩论，用户讨论了 GPT-4 和 ChatGPT 之间的差异。还提出了关于消息限制、自定义 GPT 聊天机器人的可访问性以及创建向量记忆数据库（vector memory database）价值的问题。
- 关于基于 AI 的文本纠错、使用不同编程语言创建 AI 以及利用 AI 进行业务规划和软件开发的广泛讨论。通过 Prompt Engineering、利用预定义库以及关注 AI Assistant 与 Bot 的区别等多种方案解决了用户查询。
- 关于评估 GPT-4 和 Novo ChatGPT 等 AI 模型的详细讨论，用户对响应质量表达了满意和不满。除了讨论功能和局限性外，用户还讨论了法律影响，如纽约时报（NYT）诉 OpenAI 案以及 Web Scraping 中的潜在法律问题。
- 报告并解决了涉及文件上传、404 错误、API Keys 和配额问题的技术故障。此外，还讨论了与使用 AI 进行心理治疗相关的内控标记（content flagging）和政策执行以及潜在的内容违规。
- 考虑 AI 在通过生成更多 Token 和提高输出质量来辅助写作任务方面的作用。重点提到了 GitHub Copilot 和 [OnboardAI](https://app.getonboardai.com) 等特定工具和资源。
- 关于 ChatGPT 中“上传文件”功能的使用及其作为知识库潜力的查询，引发了对其作为参考文档功能的解释。此外，AI Assistant API 信息的不一致性引起了关注。
- 讨论了与 ChatGPT 互动的最佳方式，以获得更详细和针对特定目标的响应，并就重新构建查询和理解 AI 的局限性提出了建议。深入讨论了 Custom Instructions 和“模版化”（cookie cutter）Prompt 的价值和应用。
- 用户询问了使用微调后的 LLM 执行多步和链式任务的可行性。ChatGPT 和 Custom GPT 的功能方面和局限性成为了关注焦点。
- 总体而言，讨论涵盖了广泛的话题，包括平台能力、技术挑战、法律影响、用户体验以及从 AI 技术中获取最大价值的策略。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (55 条消息🔥🔥): 
        
- **用于文本纠错的 AI**：用户 `@silconumbathree` 寻求能够纠正大量文本（约 30,385 个字符或 5,547 个单词）拼写、语法和其他错误的强大 AI 建议。讨论了包括 **GoogleDeepMind** 和 **GPT-4** 在内的多个 AI 模型。作为回应，`@michael_6138_97508` 建议由于 **ChatGPT** 是非确定性的（nondeterministic），每次尝试的结果可能会有所不同。他进一步建议参考 [OpenAI Documentation](https://platform.openai.com/docs/guides/prompt-engineering/strategy-test-changes-systematically) 中的指南尝试不同的 Prompt Engineering 技术。

- **使用 Java 或 Golang 创建 AI**：`@blazin77kong_89701` 询问如何在低配置机器（Intel i3，4GB RAM，7 年机龄）上使用 Java 或 Golang 创建 AI。包括 `@solbus`、`@bambooshoots`、`@lugui` 和 `@michael_6138_97508` 在内的多位用户表示，私有化部署（self-hosted）AI 非常耗费资源，建议利用现有的 AI 工具，或通过 Coursera、Udemy 等平台学习机器学习的基础知识。`@bambooshoots` 推荐了 GitHub 上的一个文本生成项目 oobabooga/text-generation-webui，以及 [Hugging Face](https://huggingface.co/) 上的相关信息。

- **利用 ChatGPT 构建 AI**：用户 `@mo___111` 寻求使用 **ChatGPT** 制定商业计划的帮助，特别是采用 SBA 结构。`@solbus` 建议为此任务创建一个自定义 GPT，重点是通过 chat.openai.com 上的 **GPT builder** 打造一个助手（assistant）而非简单的机器人（bot）。

- **软件开发 AI 助手**：`@dydzio` 询问了可以通过阅读整个 GitHub 仓库来进行个性化定制的 AI 助手。`@zeriouszhit` 指向了 GitHub Copilot 作为解决方案；然而，有人指出该服务目前仅限于 VSCode。`@dydzio` 提出了另一个潜在工具，即 [OnboardAI](https://app.getonboardai.com)，这是一个供初学者从错误中学习的服务。

- **用于写作任务的 AI**：`@sinisterj12` 表达了对 **ChatGPT/Gemini** 等语言模型允许更多 Token 的期待，希望借此完成一整本传记的写作。同时，`@afterst0rm` 对 **GPT-4 Turbo** 的 Prompt 和响应质量表示满意，并分享了一个在 [OpenAI Playground](https://platform.openai.com/playground/p/WLMBiIjhd4V7IYq1LdKFshFR?model=gpt-4-1106-preview&mode=chat) 上对该模型进行测试的链接。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (332 条消息🔥🔥): 
        
- **ChatGPT 和 OpenAI 定价**：`@1aztralz` 等用户讨论了 OpenAI 服务（如 GPT-4）与 ChatGPT 相比的成本效益，`@solbus` 提供了定价信息和 Token 计数的链接。
- **OpenAI 更新与问题**：用户讨论了 GPT-4.5 可能的发布，并对 GPT-4 的现状提出质疑，`@millymox` 声称 GPT-4 已经变得几乎无法使用。`@pyhelix` 询问了关于 GPT-4.5 的任何更新。
- **文件上传错误**：`@euroeuroeuro` 在上传文件时遇到了问题，无论使用何种文件类型或浏览器，这引发了包括 `@.pythagoras` 在内的其他用户的建议和讨论。
- **NYT 与 OpenAI 的法律讨论**：纽约时报（NYT）起诉 OpenAI 是聊天中的一个经常性话题，`@gamerg.` 和 `@lugui` 等用户就可能的经济动机以及对 OpenAI 和整个 AI 行业的破坏性后果发表了看法。
- **网页抓取讨论**：`@maxdipper` 寻求为 Discord 机器人进行网页抓取（Web scraping）的帮助，`@kaveen` 告知了在尝试对 Doordash 等网站进行任务自动化时可能存在的法律问题和困难，例如反爬虫保护。

### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (110 条消息🔥🔥): 
        
- **ChatGPT-4 性能讨论**：`@oscarrsm` 和 `@nlp.sensei` 等人对模型的性能进行了评估。`@oscarrsm` 对 ChatGPT-4 生成的脚本质量表示不满，而 `@nlp.sensei` 则寻求关于如何访问 GPT-4 以进行全面代码生成和评估的指导。`@solbus` 帮忙澄清了使用 ChatGPT Plus API 和 GPT-4 涉及独立的计费和设置机制。

- **使用 Java 或 Golang 开发 AI**：`@blazin77kong_89701` 询问了使用 Java 或 Golang 构建 AI 的可行性。 

- **404 错误讨论**：`@ariel129` 在 Docker 环境中进行 NodeJS 操作时遇到了 404 错误，引发了与 `@lugui` 的反复讨论，随后建议检查环境变量以及请求的 URL、Body 和 Header。

- **模型评估与偏好**：`@lugui` 和 `@elektronisade` 与 `@pudochu` 讨论了各种 AI 模型的优缺点，包括 OpenAI、社区（如 Huggingface）和 Google 提供的模型。对话涉及语言支持、延迟和地理可用性等问题。

- **内容标记与政策执行**：`@mutant_llama1` 提出了一个问题，即由于内容标记（Content Flagging）而无法评估潜在的内容政策违规行为。`@mka79` 也表达了类似的担忧，其治疗性文本会话提示词被标记。`@lugui` 指出，将 AI 用于治疗可能违反了 OpenAI 的内容政策。

- **API Key 和配额问题**：`@shivasai2023` 表示在购买了 20 美元的 ChatGPT-4 方案后，在进行 API 调用时遇到困难，并觉得没有收到预期的额度。`@solbus` 澄清了 ChatGPT Plus 的使用/计费与 API 的使用/计费是分开的。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (12 条消息🔥): 
        
- **GPT-4 的消息限制**：用户 `@greverden` 询问了 GPT-4 当前的每小时消息限制。`@solbus` 回复称，**目前的上限是每 3 小时 40 条消息**。自定义 GPTs 的上限较低，为 **每 3 小时 25 条消息**。
- **Plus 订阅者对自定义 GPT 聊天机器人的访问权限**：`@linkz.lynx` 询问全球范围内的 Plus 订阅用户（特别是波兰用户）是否可以使用自定义 GPT 聊天机器人。`@solbus` 确认 **任何拥有 ChatGPT Plus 的人都可以使用和创建自定义 GPTs**，并引导用户访问 OpenAI 网站上的 [editor](https://chat.openai.com/gpts/editor) 和 [discovery](https://chat.openai.com/gpts/discovery) 页面。
- **向量记忆数据库与 GPT Plus 的价值对比**：`@smokzz` 询问了创建向量记忆数据库并使用 OpenAI API 与订阅 GPT Plus 相比的价值。
- **关于针对多步骤任务微调大语言模型 (LLMs) 的咨询**：`@ruili09` 提出了一个高层级的问题，即针对组合型任务使用微调后的 LLMs 的最有效方式。他们询问是 **利用一个大模型同时执行所有任务** 更好，还是 **使用多个微调后的模型在链式操作中分别处理独立任务** 更好。
- **自定义 GPTs 中“上传文件”功能的限制与用法**：`@milodal` 询问了自定义 GPTs 中“上传文件”功能的限制，以及这些文件是否可以用于交叉引用细节。`@solbus` 提供了一个 [FAQ 链接](https://help.openai.com/en/articles/8555545-file-uploads-with-gpts-and-advanced-data-analysis-in-chatgpt)，解释说 **这些文件本质上作为参考文档使用**，GPT 可以从中查询或检索信息。
- **AI Assistant API 信息一致性问题**：`@aviassaga` 分享了他们的经验，即 Assistant API 偶尔会提供官方提供数据之外的建议，这与给 AI 的指令相矛盾。
- **自定义 GPT 和知识库功能**：`@milodal` 进一步探讨了自定义 GPT 是否可以自动搜索上传的文件以交叉引用用户的查询，并利用内容在回复中提供新的或额外的信息。他们还询问这是否可以作为一个持续更新并提供准确信息的私有数据库运行。

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (18 条消息🔥): 
        
- **ChatGPT 用户参与策略**：用户 `@thebookoforlando777115` 询问如何从 **ChatGPT** 获取更详细的输出，而不是笼统的回答，并以创建一个网络安全学位的短期课程计划为例。`@eskcanta` 建议重新组织查询以获得更好的结果，包括为 ChatGPT 提供特定目标或一系列目标，提及背景知识，并询问关于免费或低成本认证选项的指导。此外，还提到 **ChatGPT** 访问的某些网站信息可能不完全准确，因为一些来源具有 robot.txt 页面，会禁止数据抓取。
- **与 AI 模型互动**：`@eskcanta` 进一步阐述了与 AI 模型互动的方法，提醒用户应始终核实 AI 提供的实时事实，因为它有“编造事实”的倾向，尤其是在被反复询问特定信息时。
- **使用 Custom Instructions**：当 `@thebookoforlando777115` 询问关于 **ChatGPT** 的理想指令时，`@eskcanta` 建议进行实验，并提醒指令的有效性取决于用户的具体目标。
- **“万能模板提示词 (Cookie Cutter Prompts)”的作用**：针对 @beanz_and_rice 关于“万能模板提示词”的问题，`@eskcanta` 分享了她在使用 **ChatGPT** 时常用的一些标准查询示例，以便更好地理解单词、概念、名称或解释某些指令。然而，她对完全依赖此类提示词持怀疑态度，并鼓励用户明确说明他们希望从模型中得到什么。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (18 条消息🔥): 
        
- **ChatGPT 的实用性与局限性**：[@thebookoforlando777115](https://discord.com/channels/123456789101112131/123456789101112131) 对 ChatGPT 的笼统回答表示沮丧，同时表达了对**定制化网络安全课程计划**的渴望。`@eskcanta` 回应称，AI 非常死板，可能无法制定独特的课程。他们建议重新构思查询，询问低成本或免费的认证选项，并比较不同的系列课程。
- **Google 访问与 Robot.txt**：`@eskcanta` 解释说，ChatGPT 对某些大学课程的访问可能会被网站的 robot.txt 页面阻止，这意味着可能需要一定程度的手动研究。
- **Custom Instructions (CI)**：`@eskcanta` 建议 `@thebookoforlando777115` 尝试使用 ChatGPT 中的 Custom Instructions，并指出这些指令是针对特定对话的，不会影响正在进行的交流。他们补充说，“理想或完美”的 CI 取决于个人的需求。
- **万能模板提示词 (Cookie Cutter Prompts)**：`@thebookoforlando777115` 使用该术语指代看似通用、预制的提示词。`@eskcanta` 解释说万能模板提示词可能有用，并提供了几个她经常使用的提示词。然而，`@beanz_and_rice` 建议的一个万能模板提示词被 `@eskcanta` 认为没有用，她将其描述为缺乏引导且容易产生幻觉。
- **与 ChatGPT 就万能模板提示词进行的对话**：`@eskcanta` 使用 `@beanz_and_rice` 的万能模板提示词与一个 ChatGPT 实例进行了对话。AI 最初没有提供令人满意的回答，但最终开始“猜测”用户的意图，`@eskcanta` 认为这仍然不令人满意，并强调了对 AI 发出清晰准确指令的重要性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- 讨论了在 Jax 中使用动态切片（dynamic slicing）的困难，以及替代解决方案，例如通过填充张量（padding tensors）使代码与 JIT 兼容。
- 分享了一份详细记录 OpenChat 实例性能指标的 INFO 日志，重点介绍了 token 吞吐量、GPU KV cache 使用情况以及待处理请求。
- 探索并比较了 Jax 和 Pytorch 在特定任务（如物理编程）中的编程差异。
- 询问了 *sliding_window_attention_jax* 函数，旨在复制循环神经网络（RNN）的功能。

- 发布并分享了新的 **Capybara 数据集**，包含超过 10,000 个多轮对话示例，并在 general 频道进行了进一步讨论。
- 重点推荐了演示 Mixture Of Vectors 实现微调（fine-tuning）的 [Youtube 视频](https://www.youtube.com/watch?v=A_RDhk9uEy4)。
- 测试了 OpenChat 3.5 awg 4bit 在特定硬件配置下的性能。
- 针对不同硬件加密货币钱包（Ledger, BitBox02, Trezor 和 Coldcard）的选择进行了咨询和讨论。
- 提议使用多模态训练模型来过滤训练数据的优劣。
- 提出了创建 “LibreAI” 的想法，这是一个旨在资助 AI 研究与开发的非营利组织。

- 分享了由 Yeyito 在 Hugging Face 上创建的 [LLM 污染检测器](https://huggingface.co/spaces/Yeyito/llm_contamination_detector)。
- 讨论了 PowerInfer 的使用，这是一种能带来速度提升的 GPU-CPU 混合接口；提供的[文章链接](https://pub.towardsai.net/powerinfer-11x-speed-up-llm-inference-on-a-local-gpu-ddb66c6cba80)进一步深化了这一讨论。
- 探索了在 PowerInfer 之上添加贝叶斯概率层以进行未来改进的想法。

- 深入探讨了将 **Mistral** 推向 8k 扩展之外的缺点，并讨论了模型合并（model merging）和微调（FT）模型，重点介绍了新发布的开源模型 [CodeNinja](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B)。
- 详细解释了来自 `@ldj` 和 `@solbus` 的训练数据集策划过程，以及过程中面临的挑战。
- 讨论了语言模型在处理长上下文长度时的性能，重点关注连贯且准确的对话能力的重要性。
- 审视了纽约时报起诉 OpenAI 的持续诉讼及其潜在后果，包括版权问题和地缘政治影响。
- 讨论了 AGI（通用人工智能）开发的民主化，以防止拥有更多资源的巨头实体垄断，并强调了开源解决方案紧跟研究进展的重要性。

- 咨询了由于 RAM 限制，如何使用 T4 GPU 和土耳其语维基百科数据创建合成 QA 数据集，以及对替代推理提供商的需求。
- 报告了获取 `books3.tar.gz` 文件的来源，因为之前所有已知的链接都已失效。
- 询问了使用 yarn-mistral-128k 模型的正确聊天格式。
- 讨论了 TogetherAI API, OpenRouter 和 Google Gemini 等具有成本效益的 AI 推理替代方案，并建议使用 Ollama 配合 Web-UI 在不同服务器上显示界面。
- 询问了如何在 RunPod 上设置 Hermes 2 版本，以及所需的模型加载器参数和 UI 模板调整，以确保顺利安装。

- `@vic49` 宣布为 LM Studio 发布新版本的[向量数据库插件](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.0)，该版本现已包含图像搜索功能。
- 讨论了将 nous-hermes 视觉模型整合到 LM Studio 插件中的事宜，并指出在寻找有用的在线代码示例方面存在缺口。
- 多位成员展示了为项目做贡献的能力，已发布[已知更新](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2)并讨论了潜在挑战。
- 提供了 Obsidian 的推理代码见解（类似于 bakllava），并承诺会持续向社区通报进展。

**Nous Research AI 频道总结**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (27 messages🔥): 
        
- **Jax 和 Dynamic Slicing 的困难**：用户 `@joey00072` 分享了 *sliding_window_attention_jax* 函数的代码，并表达了在 Jax 中实现 Dynamic Slicing 的困难。其他用户建议了寻求帮助的其他渠道，`@rishiiyer` 提供了即时解决方案，建议对 Tensor 进行 Padding 而不是使用 Dynamic Slicing，从而使代码与 JIT 兼容。
- **Jax 与 PyTorch**：随后进行了关于 Jax 的简短讨论，`@rishiiyer` 和 `@joey00072` 观察到 Jax 在处理需要动态功能的任务时可能比较困难。尽管如此，`@rishiiyer` 提到他喜欢将 Jax 用于物理编程。
- **OpenChat 性能指标**：用户 `@fullstack6209` 分享了一系列 INFO 消息，详细说明了在拥有 11GB VRAM 的 RTX 2080ti GPU 上运行 OpenChat 实例的性能指标。日志详细记录了 Token 吞吐量、GPU KV Cache 使用情况和待处理请求。
- **关于 Sliding Window Attention 使用的讨论**：`@rishiiyer` 询问了 `@joey00072` 使用 Sliding Window Attention 的目的。`@joey00072` 提到其目标是在 Hidden States 中使用 Sliding Window Attention，灵感来自 Recurrent Neural Networks 的功能。


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (61 messages🔥🔥): 
        
- **Capybara Dataset 发布**：`@ldj atgctg` 发布了关于新的 [Capybara dataset](https://huggingface.co/datasets/LDJnr/Capybara) 的消息，该数据集包含超过 10,000 个多轮对话示例。
- **使用 Mixture Of Vectors 实现高效微调**：`@pradeep1148` 分享了一个关于 Mixture Of Vectors 实现的 [Youtube 视频](https://www.youtube.com/watch?v=A_RDhk9uEy4)。
- **OpenChat 3.5 性能**：`@fullstack6209` 报告称 OpenChat 3.5 awg 4bit 在 11GB 的 2080ti 上运行良好，在 40-70 tokens/sec 的速度下可以处理 8192 的最大 Context。
- **硬件加密钱包**：`@john0galt` 就在各种硬件加密钱包（Ledger、BitBox02、Trezor 和 Coldcard）之间进行选择寻求建议。
- **改进训练数据**：`@skadeskoten` 建议需要一个模型来过滤训练数据的好坏，可能可以使用 Multimodal 模型。
- **“LibreAI” 非营利组织提案**：`@narvikd` 提议创建一个名为 “LibreAI” 的非营利组织，为 AI 研发提供资金。
- **利用开源模型获利**：`@gabriel_syme` 强调了专注于解决重大问题而不是寻求 VC 融资的重要性。
- **本地 LLM 交互**：`@jason.today` 分享了一个 [GitHub 仓库](https://github.com/jasonjmcghee/rem) 链接，详细介绍了一种增强与本地 LLM 交互的开源方法。
- **欧盟打赏法规**：`@fullstack6209` 提到了欧盟关于 AI 打赏（Tipping）的新规定。


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (15 messages🔥): 
        
- **LLM 污染检测器**：`@.beowulfbr` 分享了一个由 Yeyito 在 Hugging Face 上创建的 [LLM 污染检测器](https://huggingface.co/spaces/Yeyito/llm_contamination_detector) 链接。`@skadeskoten` 也认为这个工具很有趣。
 
- **PowerInfer - GPU-CPU 混合接口**：`@skadeskoten` 分享了一篇关于 PowerInfer 的[文章链接](https://pub.towardsai.net/powerinfer-11x-speed-up-llm-inference-on-a-local-gpu-ddb66c6cba80)，这是一种 GPU-CPU 混合接口，可显著提高速度。随后与 `@.beowulfbr` 和 `@georgejrjrjr` 就 PowerInfer 与其他技术的结合使用进行了讨论。
 
- **贝叶斯概率层**：`@skadeskoten` 建议在 PowerInfer 之上添加某种贝叶斯概率层（Bayesian Probabilistic Layer）以进行改进。
  
- **拟稀疏技术**：`@georgejrjrjr` 表示这些拟稀疏（Quasi-Sparse）技术仅适用于 ReLU 模型，对 SWiGLU 模型的用途有限。在推理加速结果方面，`@georgejrjrjr` 提到了被低估的 EAGLE。 

- **Mixtral 与稀疏化**：`@georgejrjrjr` 讨论了 Mixtral 的潜在前景，这是一种拟稀疏技术，并询问了 TD 发布的稀疏化（Sparsification）代码在会议后的更新情况。

- **3D Gaussian Splatting**：`@yikesawjeez` 分享了一篇关于 [3D Gaussian Splatting](https://efficientgaussian.github.io/) 的文章链接，这是一种有助于实时渲染和加速训练的方法，尽管对内存资源有很大需求。

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (377 条消息🔥🔥): 
        
- **特定工作负载下的模型性能与模型合并讨论**：用户就不同模型的性能展开了对话。`@Imonenext` 提到了在没有持续 pretraining 的情况下将 **Mistral** 推向 8k 扩展之外的缺点。`@.beowulfbr` 支持 model merging 和 fine-tuned (FT) 模型，尽管社区对这些做法存在一些抵制。他们还重点介绍了新发布的开源模型 [CodeNinja](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B)。参与者随后讨论了合并模型通常如何通过刷榜 evaluation leaderboards 来博取关注，从而疏远了一些用户。

- **训练数据集的策划**：用户探讨了为训练语言模型策划数据集的挑战和方法。`@solbus` 澄清说，上传的文件是作为参考材料，而不是修改 AI 的基础知识。在详细解释中，`@ldj` 分享了他生成和过滤数据集的过程，重点介绍了他是如何创建 [Capybara dataset](https://huggingface.co/datasets/LDJnr/Capybara) 的。 

- **处理模型中更大的 Context Lengths**：由 `@spirobel` 发起，讨论了语言模型处理长 context lengths 的效果。`@ldj` 指出 Capybara 在长上下文测试中表现出色，并强调连贯且准确地管理长对话的能力是衡量模型能力的重要指标。 

- **关于开源运动和版权问题的讨论**：用户辩论了纽约时报起诉 OpenAI 可能带来的后果，表达了对未来限制使用受版权保护内容进行 AI 训练的担忧。他们还强调了如果某些地区（特别是中国）限制较少，可能产生的地缘政治影响。

- **民主化 AGI 开发**：一场关于通用人工智能 (AGI) 的对话展开，`@gabriel_syme` 表达了对拥有更多资源的实体可能主导 AGI 开发的担忧，因此开源解决方案必须跟上步伐。还讨论了商业模型和开源模型的性能，用户 `@teknium` 指出，企业实现 AGI 本身并不是负面的，只要他们是通过研究而不是通过监管优势实现的。


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (9 条消息🔥): 
        
- **利用土耳其语维基百科数据创建合成 QA 数据集**：用户 `@umarigan` 询问如何使用 T4 GPU 并在不使用 llama-70 等大模型（受限于 RAM）的情况下，利用土耳其语维基百科数据创建合成 QA 数据集。他们建议使用推理提供商的模型，例如来自 anyscale 的 llama-70，因为每百万 token 的成本仅为 1 美元。
- **访问 Books3**：`@fullstack6209` 正在寻找获取 `books3.tar.gz` 文件的来源，并报告说他们尝试过的所有链接都已失效。
- **纠正 Yarn-Mistral-128k 模型的 Chat Format**：`@jpham` 请求指导在有 system prompt 时，yarn-mistral-128k 模型应使用的正确 chat format，因为他们的尝试没有产生令人满意的结果。
- **为远程服务器中的 GPU 显示 UI**：`@narvikd` 寻求关于当 GPU 在不同服务器上时应使用什么 UI 的建议，并提到他们通常使用 exl2 格式。
- **AI 推理服务的替代方案**：`@night_w0lf` 向 `@umarigan` 和 `@narvikd` 推荐了 TogetherAI API、OpenRouter 和 Google Gemini 等具有成本效益的 AI 推理方案，并建议使用带有 Web-UI 的 Ollama 在不同服务器上显示 UI。
- **在 Runpod 上设置 Hermes 2**：`@rational_adherence` 询问了在 runpod 上设置 Hermes 2 版本的问题，特别是寻求关于选择 model loader 和调整参数的指导。该用户提到正在使用一键式 UI 模板。

### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (10 条消息🔥): 
        
- **LM Studio 向量数据库插件发布**：`@vic49.` 宣布发布了 LM Studio 的向量数据库插件，现在包含图像搜索功能。[发布地址见此](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.0)。
- **集成 Nous-Hermes 视觉模型**：`@vic49.` 表达了在插件中加入一些 nous-hermes 视觉模型的愿望，但未能找到相关的在线示例代码。
- **项目贡献**：`@rishiiyer` 主动提出负责项目中未完成的部分，该建议得到了 `@teknium` 的积极回应。
- **项目更新**：`@vic49.` 分享了 [V3.0.2 版本的发布链接](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/V3.0.2)，并提到了遇到的一些问题。
- **Obsidian 的推理代码**：`@qnguyen3` 告知 Obsidian 的推理代码将与 bakllava 类似，区别在于将 EOS token 更改为 '###'。他们表示打算尽快发布更新。


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **Mistral AI 讨论**：`@.skyair`、`@flopsy1` 和 `@meyelo` 广泛讨论了 Mistral AI 的各个方面，例如在冗余度（verbosity）方面的限制和 tokenizer 的差异。`@The Ledger Luminary` 和 `@.tanuj.` 还分享了关于高效系统提示词（system prompts）以及将某些技术与 Mistral 7B 集成的建议。[`关于 tokenizer 讨论的 GitHub 链接`](https://github.com/imoneoi/mistral-tokenizer)。
- **部署与性能见解**：`@builderx` 和 `@rorolembrouille` 分别分享了在 **A100 GPU** 和 **Dell Precision 5500** 笔记本电脑上运行 Mistral AI 的观察结果。此外，`@rorolembrouille` 询问了 AWS 部署成本和推理时间，`@frosty04212` 回复称大约为 **1 token/s**。
- **微调与模型使用**：`@lerela casper_ai` 强调了社区对针对特定应用微调基础模型的兴趣。`@pradeep1148` 进一步分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=A_RDhk9uEy4)，展示了用于高效微调的 **Mixture of Vectors (MoV) 方法**。
- **模型对比与性能**：用户（特别是 `@.gue22`）将 Mistral 与 GPT-4 进行了对比，对 Mistral 的特定回答表示不满。相比之下，`@bam4d` 澄清了合适的模型对比对象，建议将 `Mixtral` 或 `mistral-medium` 作为更高性能的选择。
- **迈向 AGI 的进展与协作**：用户 `@poltronsuperstar` 分享了高效使用 LLM 的进展，计划从 GPT-4 转向 mistral-medium，并从头开始一个项目。目标是促进迈向 AGI 的协作尝试，并邀请社区参与。
- **社区贡献与请求**：除了旨在持久化统计数据和增强 docstrings/types 的 PR 之外，在 `@thesealman`、`@bam4d` 和 `@poltronsuperstar` 的对话中，还提出了 Markdown 文档生成问题以及 Mistral 模型的格式化问题。`@m1337d` 在 `#random` 频道提出了社区 Logo 倡议。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (20 条消息🔥): 
        
- **Mistral 的冗余度**：`@.skyair` 和 `@flopsy1` 发起了关于 Mistral 回答过于冗长的讨论，即使是针对是非题。`@lee0099` 建议通过限制 token 数量和使用更严格的系统提示词来缓解这一问题。
- **Tokenizer 差异**：`@meyelo` 和 `@sublimatorniq` 讨论了 Mistral 中使用的不同 tokenizer，并提供了其 GitHub 仓库链接，[点击此处查看](https://github.com/imoneoi/mistral-tokenizer)。
- **提示词指令**：`@The Ledger Luminary` 就如何为 Mistral 准备更有效的系统提示词提供了建议。他们建议将段落形式的指令改为条列式的约束列表。
- **技术结合**：`@ton1785` 询问了关于将 [GPT pilot](https://github.com/Pythagora-io/gpt-pilot) 或 [GPT Engineer](https://github.com/gpt-engineer-org/gpt-engineer) 与 Mistral 7B 集成的见解和经验。`@.tanuj.` 分享了他们的观察，提到他们的工作更接近“提示词 -> 可运行项目”的流水线。
- **内容标记问题**：`@mka79` 提出了一个问题，即他们的心理治疗课程内容被 OpenAI 的 GPT-4 API 标记了。虽然问题尚未解决，但该用户表达了使用 Mistral Medium 用于生成更长输出的兴趣。

### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (9 messages🔥): 
        
- **为模型使用 GPU**：`@serhii_kondratiuk` 试图了解为什么他的 GPU 在计算过程中没有被调用。`@iukea` 建议检查模型的加载时间并确保 GPU 确实处于使用状态。`@serhii_kondratiuk` 随后通过选择 GPU Offload 并设置部分层进行 GPU 利用找到了解决方案。
- **关于 RAM 和 VRAM 的澄清**：`@iukea` 要求 `@serhii_kondratiuk` 在讨论内存负载时确认他指的是 RAM 还是 VRAM。`@serhii_kondratiuk` 没有直接回答这个问题，但提到了一项涉及调整 GPU 设置的解决方案。
- **Gemini 中的误报 (False Flags)**：`@mka79` 提出了一个问题，即 Gemini “持续地”产生误报。错误通常在响应完成 99% 时出现，这让人觉得既好笑又恼火。用户没有提供关于该问题的进一步背景或潜在解决方案。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (6 messages): 
        
- **Mistral AI 在不同系统上的性能**：用户 `@builderx` 指出 **A100 GPU** 的性能慢得令人失望。
- **关于在特定笔记本电脑上运行 Mistral AI 的咨询**：`@rorolembrouille` 询问是否可以在他们的 **Dell Precision 5500** 笔记本电脑（配备了不错的 GPU）上运行 Mistral AI 开源 LLM。`@frosty04212` 建议他们尝试一下，并指出最佳工作流将取决于具体的用例。
- **Mistral AI 的推理时间**：`@rorolembrouille` 询问在最坏情况下计算是否会耗时数小时，`@frosty04212` 回复称，除非使用效率极低的系统，否则推理时间更有可能在 **1 token/s** 左右。
- **在 AWS 上运行 Mistral AI 的成本分析**：`@rorolembrouille` 询问是否有人计算过在 AWS 或类似云平台上运行 Mistral AI 的单 token 成本。消息中未提供具体答案。


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 messages): 
        
- **对基础模型进行 Finetuning**：`@lerela casper_ai` 指出了**社区对基础模型进行 Finetuning 的兴趣**，特别是在涉及特定应用时。
- **高效 Fine-Tuning**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=A_RDhk9uEy4)，概述了 **Mixture of Vectors (MoV) 方法** 以及标题为 “Efficient Fine-Tuning Using Mixture Of Vectors Implementation” 的提议方法的实现。


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (8 messages🔥): 
        
- **Mistral 7B 与 GPT-4 的对比**：用户 `@.gue22` 使用未经微调的 Mistral 基础模型询问了许多技术相关问题。该用户对回答感到不满，认为其性能远不如 GPT-4，并感叹 “*OMG*”。 
- **提问时的模型选择**：`@bam4d` 建议在提问时使用 “instruct” 模型，而不是未经微调的基础模型。他们指出基础模型在回答问题方面可能不够精通。
- **对 Mistral 响应的担忧**：使用来自 Apple 的 [官方 LLM 示例](https://github.com/ml-explore/mlx-examples)，`@.gue22` 继续进行实验，发现 `mistral-7b-instruct-v0.1` 的响应与 GPT-4 相比不尽如人意。鉴于用户感知到的显著性能差距，该用户对 Mistral 等新型模型的过度炒作表示沮丧。
- **Mistral 与 GPT-4 的比较**：`@bam4d` 澄清说 `Mistral-7B-Instruct-v0.1` 是一个比 GPT-4 小得多的模型，应该与 13B 模型进行比较。他们推荐 `Mixtral` 或 `mistral-medium` 作为性能可能更高的模型。 
- **硬件讨论与未来预期**：`@.gue22` 提到了他们的硬件配置，重点是增加容量以运行更大的模型。用户表达了对未来 AI 模型改进的希望，并对该领域目前的进展感到沮丧。

### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 条消息): 
        
- **使用 LLM 调用工具**：用户 `@poltronsuperstar` 对 Large Language Models (LLMs) 进行了实验，发现如示例所示，给出对话式命令是让 LLM 使用工具最可靠的方法。值得注意的是，`!i <agentName>` 被提及作为一种开启对话的方法，允许进行认知树构建。

- **Agent 效率**：`@poltronsuperstar` 指出，尽管存在可靠性问题，但如果 Agent 的角色是解决单元测试，它会不断尝试直到成功。虽然仍需要人类大脑来验证单元测试，但时间增益显著——新流程使 `@poltronsuperstar` 的生产力达到了过去的 15 倍。

- **技术栈转型**：`@poltronsuperstar` 分享了从 GPT-4 转向 Mistral Medium 的意图，并计划从头开始。计划下个月发布一个包含这些 AGI（通用人工智能）尝试的开源项目。在发布前公开征集合作。

- **代码库担忧**：`@poltronsuperstar` 提到，决定从头开始而不是立即开源，是因为代码库臃肿的问题，当代码库自我构建直到变得难以管理时，就会发生这种情况。

- **社区 Logo 倡议**：用户 `@m1337d` 表示有兴趣开始构思社区 Logo。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (20 条消息🔥): 
        
- **在 Python 客户端仓库提交 Pull Requests 与测试重构**：`@poltronsuperstar` 表达了在 Python 客户端仓库提交 PR 的愿望，意在持久化统计数据并增强 Docstrings/Types。他们还询问了重构测试是否会显得冒昧，`@bam4d` 给予了肯定回复，前提是覆盖范围不被破坏且更改易于理解。
- **生成 Markdown 文档及 Mistral 模型的格式问题**：`@thesealman` 报告了在 Mistral-Medium 模型上使用流式输出生成 Markdown 文档时的问题。某些行不符合 Markdown 格式，导致多行内容被渲染在同一行。在 Mistral-Small 忽略 Markdown 格式时也观察到了类似问题。
- **使用 Mistral.ai 进行代码调试**：`@ved_ikke` 寻求关于最适合 Python 代码调试的 Mistral.ai 模型的建议。`@The Ledger Luminary` 建议测试不同的 Endpoints 并评估结果，以了解在复杂上下文中的能力，并补充提到可以考虑在 HuggingFace 上寻找微调模型。
- **请求 API 邀请及流程说明**：`@rorolembrouille` 请求 API 邀请，并被 `@lidonius` 告知此类邀请由 MistralAI 提供，而非其他用户/开发者。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- 关于 **添加 Token 和初始 Loss** 的话题，由 `@noobmaster29` 和 `@nanobitz` 提及，引发了关于训练模型和处理高初始 Loss 的讨论。
- 针对 **数据集缓存问题**，用户 `@mihai4256`、`@noobmaster29` 和 `@nanobitz` 尝试排查数据集重新加载和清理缓存的问题，但未获解决。
- 由 `@nafnlaus00` 和 `@nanobitz` 主导的关于 **Transformer Expert 的 TopK 参数** 的讨论，提供了对 Torch 的 TopK 功能及其潜在修改的见解。
- `@tmm1` 和 `@le_mess` 提供了若干 **开发问题与解决方案**，例如 Pre-commit Action 问题及其替代方案，即使用 [Pre-Commit.CI](https://pre-commit.ci)。
- `@faldore` 发布了关于测试 Modal 的 **DPO 就绪情况** 的查询。
- 探索脚本中的 **EOS Token 替换**，这是由 `@faldore` 提出的一个关注点。
- `@fred.bliss` 分享了 **使用本地模型生成数据** 时遇到的挑战，寻求关于创建与原始材料相似配对的建议。
- `@natefyi_30842` 发起了关于 **互联网和书籍作为 LLM 训练数据源** 的对比询问。
- `@mr_morning` 提供了一个有用的 `libopenmpi-dev` 和 `mpi4py` **安装命令**，这对 `@q0d.` 非常有用。

**OpenAccess AI Collective (axolotl) 频道摘要**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (24 messages🔥): 
        
- **添加 Token 和初始 Loss**：`@noobmaster29` 开始训练一个额外添加了 15k 个 Token 的模型，并注意到初始 Loss 很高。`@nanobitz` 建议这是正常现象，特别是对于补全数据集（completion datasets），并建议让训练运行一段时间看看 Loss 是否会下降。
- **数据集缓存问题**：`@mihai4256` 在尝试重新加载数据集进行调试时遇到了缓存问题。`@noobmaster29` 建议删除 HuggingFace 数据集文件夹以清除缓存，而 `@nanobitz` 指出，如果相关键值设置为 `None`，最新版本不应从缓存加载已 Tokenize 的数据集。该问题仍未解决。
- **Transformer Experts 的 TopK 参数**：`@nafnlaus00` 对 Transformer Experts 中使用的 `torch.topk` 函数进行了澄清，解释了它与专家数量以及一次使用的专家数量无关。这引发了与 `@nanobitz` 关于增加 `top_k` 参数的可能性及其影响的讨论。
- **资源共享**：`@noobmaster29` 分享了一篇 [ArXiv](https://arxiv.org/pdf/2309.09530v1.pdf) 论文链接，尽管消息中未提及该论文的具体背景或相关性。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (6 messages): 
        
- **Pre-Commit Action 的问题**：`@tmm1` 分享了 Pre-Commit Action GitHub 仓库的一个功能请求链接，重点关注将 pre-commit 错误的注释添加到实际 PR diff 中特定行的问题（[GitHub issue #70](https://github.com/pre-commit/action/issues/70)）。

- **Pre-Commit Linter 错误**：`@tmm1` 建议从日志中提取出“愚蠢的 lint 错误”，以便更好地访问和调试。

- **引入 Pre-Commit.CI**：`@tmm1` 介绍了 [Pre-Commit.CI](https://pre-commit.ci)，这可能是强制发现并自动修复 linter 错误的一种潜在方式。

- **Pre-Commit.CI 获准**：`@le_mess` 同意 `@tmm1` 的建议，肯定了 Pre-Commit.CI 是一个执行 linter 的高效工具。

- **DPO 就绪情况查询**：`@faldore` 询问了 DPO (Differential Privacy Optimizer) 是否已准备好进行测试，并声称他们已经为此准备好了模型和数据集。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (26 messages🔥): 
        
- **添加新 Token 作为句子结束 (EOS) Token**：用户 `@faldore` 分享了他们用于修改 Tokenizer 中 EOS Token 的脚本。他们将 `</s>` EOS Token 替换为一个新的 Token `


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **使用本地模型生成数据**：`@fred.bliss` 尝试使用本地模型和 RAG 方法生成高质量数据。该尝试未获成功，用户正在寻求关于如何为每个问答对创建至少 *50 行示例数据*（且接近原始材料）的建议。
- **关于使用互联网与书籍训练基础 LLM 的研究**：`@natefyi_30842` 询问是否有人研究过利用互联网数据集与书籍训练基础语言模型 (LLM) 之间的差异。他们认为，虽然书籍可能提供*质量更高但多样性较差的数据*，但这并不一定是不利的。


### ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (2 messages): 
        
- **安装 libopenmpi-dev 和 mpi4py**：用户 `@mr_morning` 建议使用以下命令进行安装：`apt-get install libopenmpi-dev -y && pip install mpi4py`，这对他很有效。用户 `@q0d.` 觉得这很有趣，并感谢了 `@mr_morning` 的帮助。


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 摘要

- 关注的主要话题是 **AI 模型选择与开发**：
    - `@woodenrobot` 正在努力为 MUD 游戏创建“智能” NPC 选择合适的 AI 模型。在提到 Google Gemini 和 GPT4 的问题后，讨论转向了使用本地模型的可能性（*general* 频道）。
    - 关于 Bert 在 Candle 和 Pytorch 上的前向传播性能，`@pigeonsai` 询问 Candle 相比 Pytorch 是否会出现预期的性能缓慢（*general* 频道）。
- **硬件与软件咨询** 非常突出：
    - `@zorian_93363` 询问是否有 CPU 拥有专门针对核心 AI 数学运算优化的核心，类似于带有集成显卡的 CPU（*general* 频道）。
    - `@mr.aflaton` 寻求创建多语言对话和配音系统的建议，分享了使用 Google 翻译 API 的经验，并询问了高质量的 text-to-speech 模型（*general* 频道）。
- 贡献者们正在通过 **AI 项目探索** 挑战极限：
    - 用户 `@neuralink` 和 `@merve3234` 讨论了一个令人印象深刻的在 3D parallelism 中实现端到端 FP8 训练的方案，并希望很快能有开源仓库（*today-im-learning* 频道）。
    - 社区成员之一 `@maybeispedro` 使用伪标签（pseudo labels）改进了天文学领域的分类任务（*i-made-this* 频道）。
    - `@sayakpaul` 发布了一项重要公告，关于 **Diffusers 0.25.0** 的发布，该版本提供了重大更新，例如名为 aMUSEd 的新模型、速度提升以及转向使用 PEFT 进行 LoRA 训练（*core-announcements* 频道）。
- 该社区也是 **专家建议与资源共享** 的中心：
    - `@asterix3651` 请求澄清关于使用 Gemini 创建数据集并用其训练模型的限制，并参考了 Google AI 的条款和条件（*general* 频道）。
    - `@maybiespedro` 分享了一篇论文链接，涵盖了研究人员将 AI 应用于天文学的创新工作，得到了 `@osanseviero` 的赞赏（*cool-finds* 频道）。
    - `@hafiz031` 寻求关于最适合商业相关数据的 embedding 模型建议，而 `@blakeskoepka` 则询问了获取 AI 研究论文的最佳平台（*NLP* 频道）。
- **AI 模型使用的技术挑战与解决方案** 备受关注：
    - `@el_musso` 表达了想在家里本地 PC 上运行模型供家人并行使用的意图，并收到了 `@merve3234` 关于本地服务和 ngrok 隧道的建议（*diffusion-discussions* 频道）。
    - `@torqx` 询问 a1111 对 Segmind Vega 模型的支持情况，因为他们在从 HuggingFace [Segmind-Vega 仓库](https://huggingface.co/segmind/Segmind-Vega/tree/main)加载 checkpoint 时遇到了困难（*diffusion-discussions* 频道）。
    - 用户 `@shinomori_7` 希望获得关于通过手势实现游戏键盘输入的库的推荐，但目前还没有收到回复（*computer-vision* 频道）。

**HuggingFace Discord 频道摘要**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (10 messages🔥): 
        
- **为 MUD 寻找聊天机器人模型**：`@woodenrobot` 询问在为 MUD 创建“智能” NPC 时如何选择模型。该用户尝试过 Google Gemini 和 GPT4，但发现两者都存在问题，目前正在考虑使用 local model 以获得更多控制权并节省成本。
- **Bert 在不同库上的 Forward Pass 性能**：`@pigeonsai` 询问是否预期 Bert 的 Forward Pass 在 Candle 上会比 Pytorch 慢。
- **AI 运算的硬件能力**：`@zorian_93363` 提出一个问题，是否有 CPU 制造商提供具有专门用于或针对 AI 核心数学运算进行优化的核心的 CPU，类似于带有嵌入式显卡的 CPU。
- **多语言对话和配音软件**：`@mr.aflaton` 寻求建议，如何创建一个系统将对话翻译成多种语言，并使用三种不同的声音类型（男声、女声、童声）将这些翻译转换为语音。用户已有使用 Google 的 translate API 进行翻译的经验，但仍需要与 Python 兼容的高质量 text-to-speech 模型。`@mr.aflaton` 尝试过 pyttsx3 和 GTTS 库，但发现前者配音质量低，后者仅支持一种声音类型。
- **Gemini 在模型训练中的使用限制**：`@asterix3651` 寻求关于使用 Gemini 创建 dataset 并用其训练模型的限制说明。用户提到 Google AI 的条款规定不能使用其服务开发与 Gemini API 或 Google AI Studio 竞争的模型，并询问对于通过 OpenAI 或 Gemini 创建的 dataset，可以应用的最开放的 open source license 是什么。


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **3D Parallelism FP8 训练**：用户 `@neuralink` 分享他们从零开始实现了 3D Parallelism 中 13% 的端到端 FP8 训练，不包括 FP8 kernel。
- **对开源仓库的期待**：`@merve3234` 表达了对看到 `@neuralink` 上述项目的开源 GitHub 仓库的期待。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **AI 应用于天文学的有趣研究**：`@maybeispedro` 分享了一个链接，详细介绍了研究人员将 AI 应用于天文学的工作，包括 [Tuan Dung Nguyen](https://arxiv.org/search/astro-ph?searchtype=author&query=Nguyen,+T+D), [Yuan-Sen Ting](https://arxiv.org/search/astro-ph?searchtype=author&query=Ting,+Y), [Ioana Ciucă](https://arxiv.org/search/astro-ph?searchtype=author&query=Ciuc%C4%83,+I), [Charlie O'Neill](https://arxiv.org/search/astro-ph?searchtype=author&query=O'Neill,+C), [Ze-Chang Sun](https://arxiv.org/search/astro-ph?searchtype=author&query=Sun,+Z), 以及 [Maja Jabłońska](https://arxiv.org/search/astro-ph?searchtype=author&query=Jab%C5%82o%C5%84ska,+M)。他们的工作可以在这篇 [arXiv paper](https://arxiv.org/abs/2309.06126) 中找到。 
- **对团队努力的认可**：`@osanseviero` 认可了该团队迷人的工作，并提到曾与他们面谈，未来有一些非常有趣的计划。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (1 messages): 
        
- **在天文学中使用 Pseudo Labels**：用户 `@maybeispedro` 分享了他们的项目，该项目使用 Pseudo Labels 来改进 tabular data 的分类任务。该方法专门应用于天文学数据。项目可以在 [GitHub](https://github.com/humphrey-and-the-machine/pseudo-labelling) 上找到。


### ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/) (1 messages): 
        
- **Diffusers 0.25.0 发布**：用户 `@sayakpaul` 宣布发布 **Diffusers 0.25.0**，主要更新包括：
    - 引入了一个名为 aMUSEd 的新模型，有趣的是它并非基于 diffusion。
    - 速度提升使 SDXL（扩展到其他 pipeline）快了 **3 倍**。
    - 转向使用 PEFT 进行 LoRA 训练。
- 更多关于发布的信息可以在 [GitHub](https://github.com/huggingface/diffusers/releases/tag/v0.25.0) 上找到。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **在本地 PC 上运行模型**：`@el_musso` 一直在考虑在家里运行模型，供家人同时使用。对此，`@merve3234` 建议**在本地提供服务并使用 ngrok 隧道**来实现并行使用。`@el_musso` 采纳了该建议并承诺会研究这一主题。
- **a1111 对 Segmind Vega 模型的支持**：`@torqx` 询问 a1111 是否支持 **Segmind Vega 模型**。他们反映在加载从 [`HuggingFace 上的 Segmind Vega 页面`](https://huggingface.co/segmind/Segmind-Vega/tree/main) 获取的 checkpoint 时遇到困难。在该消息线程中，他们尚未收到任何回复或解决方案。


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (2 messages): 
        
- **将手势作为游戏的键盘输入**：用户 `@shinomori_7` 寻求关于库的推荐，以帮助他们通过手势获取游戏的键盘输入。他们发出了求助并正在耐心等待回复。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (6 messages): 
        
- **为业务数据选择 Embedding 模型**：`@hafiz031` 征求关于最适合业务和财务数据的 Embedding 模型的建议。
- **AI 研究论文的来源**：`@blakeskoepka` 提出了一个关于阅读 AI 研究论文最佳平台的问题。`@osanseviero` 推荐了 [Hugging Face papers 页面](https://hf.co/papers)，该页面每天都会提供人工筛选的论文。此外，VIPitis 建议使用带有特定关键词的 **ArXiv** 来获取更多预印本出版物。
- **Encoder 模型对比**：`@opencuiguy` 回应了 `@merve3234` 的评论，表示在个体判别任务中更倾向于使用 encoder-only 模型，尽管他也指出这类模型可能比较死板。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (5 messages): 
        
- **在本地 PC 上为多用户运行模型**：用户 `@el_musso` 询问了在家庭本地 PC 上运行模型以供多位家庭成员并行使用的可能性。`@merve3234` 建议将本地服务化并使用 ngrok 隧道作为潜在解决方案。
- **a1111 对 Segmind Vega 模型的支持**：`@torqx.` 询问 a1111 是否支持 Segmind Vega 模型，因为他们在加载从 HuggingFace [Segmind-Vega 仓库](https://huggingface.co/segmind/Segmind-Vega/tree/main) 获取的 checkpoint 时遇到了问题。


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- `@heman35` 在 [looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/) 和 [looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/) 频道中寻求 AI/ML 专家，以开发已建立后端的 **Chat GPT Plugin**。
- `@venadore` 在 general 聊天频道中实验并讨论了 **bias calls**，但未提供进一步的背景或细节。
- 用户 `@imonenext` 讨论了 **微调 MoE 模型** 的过程以及确定合适超参数的方法。
- 一场关于 **AI 与伦理** 的激烈辩论，批评了某个特定的有争议的 AI 研究小组在声称其创新性时的伦理缺陷，认为其成果是建立在开源工具之上的，详见 [twitter](https://fxtwitter.com/winglian/status/1740081525826167060)。特别强调了 `@caseus_`、`@undi` 和 `@giftedgummybee` 的用户反应。
- `@gabriel_syme` 开玩笑说自己有一篇“免费论文”，将对话转向了更轻松的氛围。

**Alignment Lab AI 频道总结**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (1 messages): 
        
- **构建 Chat GPT 插件**：用户 `@heman35` 寻求构建 Chat GPT 插件的帮助。他们提到**后端已经准备就绪**，并特别要求 **AI, ML 和 NLP** 专家的协助。他们鼓励感兴趣的人员直接私信（Direct Message）联系。


### ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/) (1 messages): 
        
heman35: 👋 嘿，你能帮我构建一个 Chat GPT 插件吗？我们后端已经准备好了。


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (2 messages): 
        
- **关于 bias calls 的讨论**：`@venadore` 提到他们正在**实验 bias calls**，并发现某种情况很有趣，但未提供具体细节。

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (19 messages🔥): 
        
- **微调 MoE 模型**：用户 `@imonenext` 向社区询问了微调 **MoE 模型** 的流程，特别是如何选择合适的超参数，以及与稠密模型（dense models）相比，学习率应该是更高还是更低。
- **AI 与伦理讨论**：关于另一个 AI 研究小组声称某项发现具有原创性的说法引发了争论，`@caseus_` 等人认为该发现是通过开源工具确定的。这引发了关于伦理和给予应有认可的讨论。Twitter 上的相关发现分享在[此链接](https://fxtwitter.com/winglian/status/1740081525826167060)。
- **贡献认可**：`@undi` 和 `@giftedgummybee` 都批评了该争议研究小组，指责其自身没有付出努力，且未认可其所使用工作的贡献者，包括来自其他开发者的开源发现和层配置（layer configurations）。`@undi` 明确指出，该研究小组声称原创的技术实际上是基于 Charles 的工具。
- **GPT-Agent 训练**：讨论进一步展开，`@giftedgummybee` 暗示该争议小组在寻求知名度，但未对社区做出任何新贡献。
- **免费 AI 研究论文**：在较为轻松的基调中，`@gabriel_syme` 幽默地插话讨论了关于获得“免费论文”的话题。


### ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/) (1 messages): 
        
heman35: 👋 嘿，你能帮我构建一个 Chat GPT 插件吗？我们已经准备好了后端。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain** 宣布文档更新，并邀请用户对草案版本提供反馈。此次更新主要涉及 Python 和 JavaScript 库。草案版本链接见[此处](https://langchain-5o76madna-langchain.vercel.app/docs/get_started/introduction)。
- 围绕多功能聊天机器人的开发与应用展开对话，包括使用 OllamaFunctions 传递聊天历史记录、从博客内容生成 FAQ，以及在聊天机器人应用设计中采用 Redis 进行聊天存储。
- 询问在处理复杂任务时使用单个或多个微调后的 LLM 的效率和战略有效性，并以 GitHub Copilot 任务作为具体案例研究。
- 发布并评审了一个开源项目 KwaiAgents，该项目声称在特定 AI Agent 任务中表现优于 GPT-3.5。[Medium 博客文章](https://medium.com/@myscarletpan/can-7b-models-now-master-ai-agents-a-look-at-kwais-recent-llm-open-source-release-8b9e84647412)和 [GitHub 仓库](https://github.com/KwaiKEG/KwaiAgents)已分享。
- 推出了一门面向初学者的在线 LLM 课程，内容包括使用 Python 和 LangChain 对 LLM 进行编程的培训。课程链接见[此处](https://lnkd.in/geN5yrkk)。
- 关于 langserve 应用和功能的各种技术咨询与讨论，涉及输入变量查询、'OpenAIAssistantFinish' 对象问题、AgentExecutors 的实现以及路由中调用（invokes）的潜在问题。
- 宣布为 LM Studio 发布 **ChromaDB Plugin** v3.0.1，其中包含修订后的 Pdf 加载器脚本。GitHub 发布链接分享在[此处](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.1)。
- 介绍了一个 GitHub 项目，该项目包含一个用 OCaml 开发的针对 Llama CPP 的 Menhir 解析器，并公开邀请贡献以改进语法 [链接](https://github.com/meta-introspector/gbnf-nice-parser)。
- `@pradeepvj97` 向 `@sacha7031` 发出提醒，建议后者出于安全原因重置其 API。具体背景尚不明确。

**LangChain AI 频道总结**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 messages): 
        
- **LangChain 文档更新**：`@hwchase17` 宣布 LangChain 团队目前正在进行文档更新，并接受对[此处](https://langchain-5o76madna-langchain.vercel.app/docs/get_started/introduction)分享的草案版本的反馈。**LangChain** 是一个用于开发由具有上下文感知推理能力的语言模型驱动的应用框架。它由几个部分组成，包括主要使用 Python 和 JavaScript 编写的 **LangChain Libraries**。不过，一些使用的新功能可能尚未合并。
    - 他们表示有兴趣听取用户在文档中除快速入门（quick start）和模块（Modules）之外的部分可能遇到的潜在问题的反馈。

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (11 条消息🔥): 
        
- **使用 SQLDatabase Chain 创建多功能聊天机器人**：`@annamalai8892` 寻求关于如何在 router chain 中结合 SQLDatabase chain 和其他 LLMChains 来创建多功能聊天机器人的帮助。
- **在 OllamaFunctions 中传递聊天历史**：`@chronos.vitaqua` 询问如何以有意义且高效的方式在使用 OllamaFunctions 的聊天机器人中传递聊天历史。
- **从博客生成 FAQ 的应用**：`@ashwinm` 寻求关于构建一个根据博客内容生成 FAQ 的应用的建议。作为回应，`@ashwinm evolutionstepper` 建议参考 GitHub 上的 [KB_builder 项目](https://github.com/offskiies/KB_builder) 以获取指导。
- **针对组合任务微调 LLMs**：`@ruili09` 征求意见，探讨是应该为复杂任务使用单个经过深度微调的 LLM，还是将多个针对任务不同步骤微调的 LLM 链接起来。他们以 GitHub Copilot 任务为例来说明观点。
- **KwaiAgents 开源项目**：`@myscarlet` 分享了快手开源项目 KwaiAgents 的 [Medium 博客文章](https://medium.com/@myscarletpan/can-7b-models-now-master-ai-agents-a-look-at-kwais-recent-llm-open-source-release-8b9e84647412) 和 [GitHub 仓库](https://github.com/KwaiKEG/KwaiAgents) 链接，该项目声称在某些 AI agent 任务中表现优于 GPT-3.5。
- **在 Redis 中存储聊天记忆**：`@sampson7786` 询问在使用 LangChain 开发聊天机器人应用时，如何使用 Redis 为不同用户存储聊天记忆。
- **面向初学者的在线 LLM 课程**：`@altafr` 宣布了他们第二期面向初学者的 LLM 培训课程，内容包括如何使用 Python 和 Langchain 对 LLM 进行编程。提供了[课程链接](https://lnkd.in/geN5yrkk)。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (6 条消息): 
        
- **在 Langserve 中查询输入变量**：`@cryptossssun` 提出了一个关于如何在 langserve 服务中查询两个输入变量的问题。
- **'OpenAIAssistantFinish' 对象的问题**：`@stu.mach` 分享了在尝试将带有自定义执行器的 OpenAIAssistant 与 langserve 配合使用时遇到的问题，这导致了与 'get_input_schema' 相关的 AttributeError。
- **AgentExecutors 与 Runnable**：`@a404.eth` 建议 AgentExecutors 可能没有实现 runnable，而这对于 `@stu.mach` 尝试使用的 'add_routes' 函数可能是必需的。
- **路由中 Invoke 的问题**：`@a404.eth` 注意到 `@stu.mach` 在路由中放置的是对 invoke 的响应，而不是一个 chain，并警告这可能会导致问题。


### ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 条消息): 
        
@pradeepvj97 sacha7031：发送此消息后，你应该重置你的 API。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (3 条消息): 
        
- **面向 LM Studio 的 ChromaDB Plugin v3.0.1 发布**：`@vic49` 宣布了面向 LM Studio 的 **ChromaDB Plugin** 最新版本 [V3.0.1 - SHOWTIME!](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/releases/tag/v3.0.1)，其中包括名为 `pdf.py` 的自定义 PDF 加载器的修订脚本。
- **使用 Menhir 为 Llama CPP 编写的 GBNF Nice Parser**：`@m1337d` 发布了他的 [GitHub 项目](https://github.com/meta-introspector/gbnf-nice-parser)，这是一个在 OCaml 中为 Llama CPP 编写的 gbml menhir 解析器工作版本。他邀请大家贡献力量以改进语法，这些语法可用于约束和自定义 Llama CPP 的输出。
- **KwaiAgents 发布**：`@myscarlet` 介绍了 KwaiAgents 的发布，这是一个具有具备 agent 能力的 LLM 的自动 agent 系统。他们还分享了相应的 [GitHub 仓库](https://github.com/KwaiKEG/KwaiAgents) 链接和一篇题为 ["7B 模型现在能精通 AI Agents 吗？"](https://link.medium.com/MBUxuLhZSFb) 的 Medium 文章。该系统包含训练数据和基准测试。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

- `@beowulfbr` 宣布了来自 OpenChat 的新微调模型 **CodeNinja**。CodeNinja 专注于代码辅助，可在 [Hugging Face](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B) 上获取。鼓励使用[此处](https://huggingface.co/TheBloke/CodeNinja-1.0-OpenChat-7B-GGUF)提供的 GGUF 格式模型文件测试该模型。
- 关于纽约时报与 OpenAI 之间诉讼的讨论，引用了 `@swyxio` 分享的 [Twitter 线程](https://fxtwitter.com/ceciliazin/status/1740109462319644905?s=46&t=90xQ8sGy63D2OtiaoGJuww)。
- `@swyxio` 通知了即将由 `<@451508585147400209>` 主持的关于 **Beyond Human Data** 论文的讨论。公告包含了加入 [讨论](https://lu.ma/llm-paper-club) 的链接，并提到这些论文会议每周举行一次。
- 由 `@eugeneyan` 协调的关于超越人类数据规模的 [论文讨论](https://arxiv.org/abs/2312.06585)。其他值得注意的讨论包括 `@swyxio` 报告的 Discord "RTC Disconnected" 错误，以及 `@187636841988620288` 即将主持的关于 [InsightPilot](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/) 的讨论。
- 用户 `@cakecrusher` 发起了一个话题，探讨何时微调 embedding 模型与 Retriever-Augmented Generation (RAG) 模型，并强调了两者在需求上的潜在差异。

**Latent Space 频道摘要**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (7 条消息): 
        
- **CodeNinja 发布**：`@beowulfbr` 宣布发布了来自 OpenChat 的新微调模型，名为 **CodeNinja**，专注于代码辅助。该模型是 openchat/openchat-3.5-1210 的增强版本，已在超过 400,000 条代码指令上进行了微调。可在 [Hugging Face](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B) 上获取。
- **CodeNinja 反馈与测试**：`@beowulfbr` 提到在 Reddit 上收到了良好的反馈，并鼓励其他人尝试 **CodeNinja** 模型。GGUF 格式的模型文件可在 [此处](https://huggingface.co/TheBloke/CodeNinja-1.0-OpenChat-7B-GGUF) 获取。
- **OpenAI NYT 诉讼讨论**：`@swyxio` 分享了一个讨论纽约时报与 OpenAI 之间诉讼的 [Twitter 线程](https://fxtwitter.com/ceciliazin/status/1740109462319644905?s=46&t=90xQ8sGy63D2OtiaoGJuww)。


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 条消息): 
        
- **Beyond Human Data 论文讨论**：用户 `@swyxio` 宣布 `<@451508585147400209>` 将在 15 分钟内主持关于 **Beyond Human Data** 论文的讨论。可以通过 [此链接](https://lu.ma/llm-paper-club) 加入讨论。这是每周 LLM 论文回顾和讨论系列的一部分。感兴趣的用户可以要求被标记到 `<@&1107197669547442196>` 以获取 Discord 通知。


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (5 条消息): 
        
- **超越人类数据规模讨论**：`@eugeneyan` 宣布了关于超越人类数据规模的 [论文讨论](https://arxiv.org/abs/2312.06585)，敦促特定成员参与。
- **Discord 错误报告**：`@swyxio` 报告在 Discord 上遇到 "RTC Disconnected" 错误，导致无法听见或说话。
- **InsightPilot 论文讨论**：在接下来的一周，`@eugeneyan` 通知 `@187636841988620288` 将主持关于 [InsightPilot](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/) 的讨论，这是一个基于 LLM 的自动化数据探索系统。
- **对未来讨论的期待**：`@ivanleomk` 对未来一年的论文讨论会议表示期待。
- **关于微调模型的疑问**：`@cakecrusher` 发起了一场关于何时微调 embedding 模型与 Retriever-Augmented Generation (RAG) 模型的讨论，并指出了两者在需求上的潜在差异。


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- 用户 `@baptistelqt` 在 #papers 频道询问是否有使用 **ChatGPT** **阅读/评审学术论文**的现有工作流，但随后没有收到回复或进一步讨论。
- 在 #off-topic 频道，用户 `@shreepandey` 推测了语言学习模型 (LLMs) 的潜力。提出了一个假设场景：如果给定来自外星物种或动物声音的充足对话数据，LLMs 是否能破译其部分语言。此外，用户 `@pradeep1148` 分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=A_RDhk9uEy4)，但未提供视频内容的背景信息。
- 在 #bakklava-1 频道，`@onuralp.spriobel` 建议 `@spirobel onuralp.` 研究 **cogVLM**，并引用了来自 **nous research discord** 的讨论。

**Skunkworks AI 频道摘要**

### ▷ #[papers](https://discord.com/channels/1131084849432768614/1131305311714672710/) (1 条消息): 
        
- **使用 ChatGPT 进行论文阅读/评审**：用户 `@baptistelqt` 询问是否有人拥有使用 **ChatGPT** **阅读/评审学术论文**的工作流。目前没有后续讨论或回复。


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (2 条消息): 
        
- **推测外星语言和动物声音**：用户 `@shreepandey` 提出了一个有趣的问题：如果给定外星物种完整的对话数据集，我们是否能通过语言学习模型 (LLMs) 辨识出其部分语言。这一想法被外推到将动物声音转换为人类可理解的语言。 
- **分享视频链接**：用户 `@pradeep1148` 分享了一个 [YouTube 视频链接](https://www.youtube.com/watch?v=A_RDhk9uEy4)，消息中未描述视频的背景或内容。


### ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (2 条消息): 
        
- **关于 cogVLM 的讨论**：根据 **nous research discord** 中讨论的结果，`@onuralp.spriobel` 建议 `@spirobel onuralp.` 关注 **cogVLM**。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- 围绕 **Mixtral Base Model Score** 的讨论中，用户 `@gheshue` 指出 .calytrix 获得了 **50.27** 分，接近基础模型的结果。
- 围绕 Mixtral 微调期间训练 router layers（路由层）概念的讨论和询问，用户 `@bjoernp` 质疑是否已达成共识。用户 `@sebastian.bodza` 提供了一个 **Eric Hartford** 的 [Twitter 帖子](https://twitter.com/erhartford/status/1737350578135834812) 作为参考，解释了冻结 router layer 以提高训练期间性能的好处。
- `@leecig` 提议组建一个兴趣小组，致力于探索各种 AI 软件技术的集成，如通过 **Ollama** 提供服务的 **MemGPT, AutoGen**，并对 **GPTPilot** 表现出潜在兴趣。用户 `@ismaelfaro` 通过回复大拇指表情表示感兴趣。

**DiscoResearch 频道摘要**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (3 条消息): 
        
- **Mixtral 基础模型评分**：`@gheshue` 表示 .calytrix 获得了 **50.27** 分，与基础模型基本持平。
- **Mixtral 微调期间训练 Router Layers**：用户 `@bjoernp` 询问在使用 Lora 进行 Mixtral 微调时，是否就训练 router layers 达成了共识。 
- `@sebastian.bodza` 进行了回复，并引用了 **Eric Hartford** 的一条 [Twitter 帖子](https://twitter.com/erhartford/status/1737350578135834812)。在该帖子中，**Eric Hartford** 说明了冻结 router layer 可以提高训练过程中的性能。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (2 条消息): 
        
- **AI 软件兴趣小组**：用户 `@leecig` 提议创建一个兴趣小组，将各种 AI 软件技术结合起来，即 **MemGPT, AutoGen**，并通过 **Ollama** 提供模型服务。他们还提到对 **GPTPilot** 感兴趣。他们邀请对此小组感兴趣的用户进行 Ping 或私信。
- 用户 `@ismaelfaro` 通过回复大拇指表情表达了兴趣。