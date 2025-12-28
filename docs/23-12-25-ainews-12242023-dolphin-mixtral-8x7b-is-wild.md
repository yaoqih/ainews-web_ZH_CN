---
companies:
- mistral-ai
- ollama
- google
- openai
date: '2023-12-26T07:23:04.603056Z'
description: '**Mistral** 模型以其无审查（uncensored）特性而闻名，Eric Hartford 的 **Dolphin** 系列对这些模型进行了无审查微调，在
  Discord 和 Reddit 上广受欢迎。


  **LM Studio** 的 Discord 社区讨论了各种话题，包括硬件兼容性（尤其是优先选择 Nvidia 的 GPU 性能）、模型的微调与训练，以及解决
  LM Studio 本地模型托管功能的相关问题。目前，与 **GPT Pilot** 的集成工作以及支持 ROCm 的 Beta 版本正在推进中。


  用户还在探索使用 **Autogen** 来实现群聊功能，并分享了如 **Ollama** NexusRaven 库等资源。此外，讨论还涉及在不同操作系统上运行
  LM Studio 的挑战、模型性能问题，以及 **Google Gemini** 和 **ChatGLM3** 编译等外部工具。'
id: 31baae57-4f8f-4247-ad05-5bc7adf7a871
models:
- dolphin
- glm3
- chatglm3-ggml
original_slug: ainews-12242023-dolphin-mixtral-8x7b-is-wild
people:
- eric-hartford
title: 2023年12月24日：Dolphin Mixtral 8x7b 太疯狂了。
topics:
- fine-tuning
- hardware-compatibility
- gpu-inference
- local-model-hosting
- model-integration
- rocm-integration
- performance-issues
- autogen
- linux
- model-training
---

 

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- 关于在不同硬件和操作系统上运行 *LMStudio* 的各种**硬件咨询**和**使用问题**；建议 Mac 用户使用 Linux Mint；考察了不同 GPU 用于 AI 推理的规格和性能，Nvidia 被视为最佳选择；提议为消费者建立硬件测试数据库，以便在购买时做出更明智的决定。 ([🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/), [🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/))

- 关于模型微调和训练、LM Studio 中的模型识别问题、Python 代码中断的故障排除，以及请求关于如何结合 ChatGPT 使用 LM Studio 的逐步教程的广泛讨论；此外，还明确了一个重要的区别：LMStudio 只能托管本地模型，不能使用 ChatGPT。 ([🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/))

- 性能——特别关注 *LMStudio* 的高 CPU 和 RAM 占用，以及在 LMStudio 生成第二个响应时点击合并按钮会导致第一个响应被删除的问题；提出的一个通用改进建议是在模型生成后提供 tokens/sec 信息。 ([🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/))

- 讨论了 GPT Pilot 与 LM Studio 的连接性——据分享，目前正在努力寻找适用于特定本地 LLM 模型的提示词（prompts），并据此修改 GPT Pilot。 ([🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/))

- 讨论了 Local LLM 的 Beta 版本——ROCm 集成，并分享了一个用于快速 ROCm 移植的 GitHub 链接，以及在不同平台上安装和加载模型时遇到的各种问题，同时也强调了有效的解决方案。 ([🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/))

- Autogen 的使用——建议调整参数以改进执行效果；提出了关于 Autogen 群聊功能实际应用案例的问题。 ([autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/))

- 分享了一个链接 [https://ollama.ai/library/nexusraven](https://ollama.ai/library/nexusraven)，没有任何上下文背景。 ([memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/))

**LM Studio 频道摘要**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (71 条消息🔥🔥): 
        
- **在不同的操作系统和硬件上运行 LM Studio**：用户 `@chudder_cheese` 寻求帮助，因为他们无法在运行最新 OS X 版本的旧款 Macbook Pro 上运行 LM Studio。`@heyitsyorkie` 澄清说 LM Studio 无法在基于 Intel 的 Macbook 上运行。`@chudder_cheese` 建议通过可启动驱动器使用 Linux Mint，并被指示在 Intel Mac 讨论帖中提供支持。用户 `@superlag` 还询问在 Linux 上运行 LM Studio 是否有任何好处，`@heyitsyorkie` 澄清说所有操作系统基本相同，但 Linux 版本落后了几个版本。
- **模型性能**：`@outcastorange` 分享了使用视觉模型识别鱼类图片的经历，但模型将图片误认为西兰花。`@heyitsyorkie` 评论说视觉模型在 Linux 上仍有已知问题，目前仅在不使用 GPU offload 的情况下才能工作。
- **微调和训练模型**：用户 `@Dm3n` 询问关于在大量数据上微调 LLM 模型的问题。`@ptable` 建议使用来自 Llama 或 Mistral 已经准备好的模型。
- **LM Studio 的使用**：关于使用 LM Studio 有不同的讨论，`@chemcope101` 询问是否可以通过 ssh 使用远程 GPU 资源，`@yaodongyak` 询问在 Mac 上运行带有 RAG 的 LM Studio 的可能性。此外，`@telemaq` 分享了一个 Reddit 帖子 <https://www.reddit.com/r/LocalLLaMA/comments/18pyul4/i_wish_i_had_tried_lmstudio_first/>，力挺 LM Studio，断言它对初学者非常友好。
- **外部工具和模型**：关于外部工具和模型有几项讨论，`@epicureus` 询问了 Google Gemini，`@randy_zhang` 询问如何使用 GitHub 仓库将 glm3 编译为 chatglm3-ggml.bin。用户 `@.alphaseeker` 询问是否有类似 VS Code Copilot 的实时句子生成器仓库，`@dagbs` 建议将 LM Studio 推理服务器的 base URL 设置为 `openai`。


### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (70 条消息🔥🔥): 
        
- **Python 代码中的意外中断**：用户 `@exrove` 在使用 TheBloke 的 neural model v3 1 7B Q8 执行 Python 代码时遇到意外中断。`@fabguy` 建议该问题可能与 max_tokens 参数有关。

- **LM Studio 识别模型的问题**：`@rymull` 遇到了 LM Studio 将基于 Mistral 的模型识别为基于 Llama 的问题。`@fabguy` 保证这是 GGUF 文件元数据中的错误，不会影响性能。

- **在 LM Studio 中停止生成**：`@imrinar` 请求提供让 API 停止在 LM Studio 中生成响应的方法。`@fabguy` 建议解决方案在于在客户端停止循环。

- **安装和使用 AI 生成图像**：`@yiitwt` 询问关于安装图像 AI 的问题。对于图生文（image-to-text），`@fabguy` 建议下载包含 vision adapter 的 obsidian 模型；对于文生图（text-to-image），他建议使用 Fooocus 等其他工具。

- **LM Studio 中的性能测量**：`@funapple` 询问模型生成后是否可以查看 tokens/sec 信息以衡量性能。`@fabguy` 和 `@heyitsyorkie` 都澄清说，生成完成后，该信息会出现在输入框底部。

- **LM Studio 在 Windows 11 上的安装问题**：用户 `@daboss.` 和 `@dialobot` 报告了在 Windows 11 上安装和运行 LM Studio 的问题。

- **请求 LM Studio 教程**：用户 `@teee2543` 请求一份关于结合 ChatGPT 使用 LM Studio 设置家庭服务器的分步教程。`@fabguy` 澄清说 LM Studio 只能托管本地模型，无法使用 ChatGPT。

- **Macbook Pro M1 Max 上的模型性能下降**：`@jacobtohahn` 提出了在 Macbook Pro M1 Max 上运行 Phind CodeLlama 34B 模型时，CPU 利用率和性能不一致的问题。尽管进行了多次故障排除尝试，问题仍然存在。

- **关于 LM Studio 中颜色代码的讨论**：用户 `@gravitylens` 提出了关于 LM Studio 中不同颜色代码含义的问题。`@psipiai` 澄清说这些颜色没有特定含义，只是视觉效果。

- **从终端启动 LMStudio.exe**：用户 `@daboss.` 在从终端启动 LMStudio.exe 时遇到困难。尽管 `@fabguy` 努力协助排查，问题仍未解决。


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (1 条消息): 
        
@kujila omkalish: 我发现它在讲故事方面的逻辑不太好

### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (8 messages🔥): 
        
- **LMStudio 的 CPU/RAM 占用**: 用户 `@renjestoo` 对 **LMStudio** 的高 CPU 和 RAM 占用表示担忧，观察到即使在未加载模型时，它也会启动 173 个进程。`@heyitsyorkie` 解释说，这些进程代表了 **LMStudio** 测量其 CPU/RAM 占用率的方式。
- **LMStudio 中合并按钮的问题**: `@msz_mgs` 指出了一个问题：在 **LMStudio** 生成第二个响应时点击合并按钮会导致第一个响应被删除。`@yagilb` 处理了此问题，并建议从网站重新下载应该可以解决该问题。


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (1 messages): 
        
- **GPT Pilot 与 LM Studio 的连接性**: `@kujila` 讨论了与 `@825528950284746794` 和 `@796127535133491230` 合作将 **GPT Pilot** 与 **LM Studio** 集成的事宜。他们提供了一个 [GitHub 链接](https://github.com/Pythagora-io/gpt-pilot/wiki/Using-GPT%E2%80%90Pilot-with-Local-LLMs)，其中包含有关将 GPT Pilot 与本地 LLM 配合使用的说明。下一步工作包括找出适用于特定本地 LLM 模型的 Prompt，并相应地修改 **GPT Pilot**。`@HeliosPrime` 凭借其 Python 经验对该项目表现出兴趣。


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (196 messages🔥🔥): 
        
- **用于 AI 推理的 GPU 性能比较**: 频道成员讨论了不同 GPU 在 AI 推理中的价值和性能。一些亮点包括：
  - @heyitsyorkie 提到 NVIDIA 是目前 GPU 推理的最佳选择，并建议购买二手 eBay 390s。
  - @acrastt 讨论了 Radeon Instinct MI60 的带宽可能超过 4090，但其 FLOPS 仍低于 3090。
  - @rugg0064 探讨了使用多个 GPU 拆分工作并增加带宽的可能性，但强调了显存和数据传输速度的挑战。他们还讨论了配备 192GB RAM 的 Mac Studio M2 Ultra 在处理大型模型时的卓越性能。
  
- **对兼容性和软件支持的担忧**: @rugg0064 对脱离主流技术表示担忧，原因是缺乏 AMD 支持，以及选择正确的“加载器”以实现最佳并行性的重要性。

- **硬件测试数据库建议**: @heyitsyorkie 和 @thelefthandofurza 提议创建一个硬件测试数据库，社区成员可以提交使用不同配置的速率测试结果，这可以作为性能排行榜。

- **对非主流 GPU 的考虑**: @rugg0064 提到了一篇 Reddit 帖子，显示 3xMI25 配置在 60bQ6 模型上达到了 3-7t/s。

- **关于购买决策的讨论**:
  - @pefortin 考虑购买新的 3090，并分享了利用本地 Facebook 群组作为购买硬件场所的经验。
  - @rugg0064 讨论了投资 AI 专用硬件的可能性，并考虑选择 Mac 方案作为高性能生产力笔记本。他们还考虑使用加密货币挖矿主板组合多个 GPU，以获得更廉价的高性能提升。


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (19 messages🔥): 
        
- **ROCm 讨论**: 用户 `@amejonah` 分享了一个关于快速 ROCm 移植的 GitHub Pull Request [链接](https://github.com/Mozilla-Ocho/llamafile/pull/122)。
- **安装错误**: `@peter894456 happy_dood` 讨论了遇到与 AVX2 CPU 支持相关的安装错误，他们通过使用另一台支持 AVX2 的 CPU 解决了该问题。他们还指出其系统上存在模型加载的持续问题。
- **Linux 上的模型加载问题**: `@ied7011` 报告了在运行 Bodhi Linux 的双 CPU 16 核联想 Thinkstation 上的一个问题：尽管拥有 56GB RAM，但无法加载任何模型。`@heyitsyorkie` 建议验证其 CPU 是否支持 AVX2 指令集，`@ied7011` 确认支持。
- **模型加载问题与服务器运行**: `@doderlein` 报告了与 `@ied7011` 类似的问题，但指出尽管无法加载新模型，他们仍然能够提供 LLM 模型服务并进行 API 查询。他们正在 Ubuntu 22.04 上运行 LM+Studio-0.2.8-beta-v1。

### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (15 messages🔥): 
        
- **使用 Autogen**: `@.mugigi` 和 `@heliosprime_3194` 简要讨论了关于使用 Autogen 2.2 版本的问题。
- **Autogen 群聊问题**: `@DAW` 反馈了一个关于 Autogen 群聊中 Agent 使用 **GPT-4** 时重复相同消息的问题。他们怀疑 **Autogen** 的概念虽然很酷，但在实现**真实用例**方面是否有所欠缺。
- **Autogen 参数配置**: `@heliosprime_3194` 建议 `@DAW` 尝试在 group chat 的 py 文件中（约第 46 行）使用不同的参数。建议包括将模式从 auto 切换为 random 或 round robin，将主 py 文件中的 messages 从 2 改为 4，并调整 seed 数值。
- **探索多 Agent 执行**: 用户 `@dagbs` 非常欣赏 **AutoGen** 并认为它很有趣。他们提到在 Docker 中实现代码执行是一个重大加分项。
- **Autogen 中的聊天启动**: `@DAW` 分享了一段关于在 Autogen 中使用 `user_proxy` 以及 `engineer`、`planner` 等角色启动 GroupChat 的代码片段。seed 和 temperature 在 `gpt4_config` 中设置。`@heliosprime_3194` 建议他们将 `max_round` 从 15 降低到 4 以获得更好的效果。


### ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 
        
sublimatorniq: https://ollama.ai/library/nexusraven 
或许可以？


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- 针对 Python 安装问题的广泛排障讨论，提供了通过 Python 虚拟环境、Python3 命令行检查以及查看已安装包来解决问题的具体建议。
- 几个 Web 开发项目名称建议，包括 "Megabytes" 和 "ThaumatoAnakalyptor"，以及一些随意的游戏相关交流。
- 分享并分析了重大的 AI 相关进展，包括 AI 趋势视频、**Nucleus X** 和 **GPT-4 (Sydney)** 等模型的介绍，以及对其潜在影响的反思。值得注意的是，分享了用于 Hugging Face 推理的 Python 代码片段，并详细探讨了涉及安全模型自主性的 AI alignment（AI 对齐）。
- 由 `@.beowulfbr` 开发并托管在 Hugging Face 上的新型开源代码助手 **CodeNinja model** 的发布与介绍。
- 关于 OpenRouter 上模型流行度的讨论，涉及 fine-tuning（微调）注意事项、benchmarking（基准测试）技术、模型评估实践，以及可能存在的模型抄袭问题。
- 对 `@Casper_AI` 在多个 AI Discord 频道中的活跃参与和才智表示赞赏。
- `@vic49.` 坚持独立开发脚本，拒绝了使用 **LM Studio** 工具的建议。

**Nous Research AI 频道总结**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (35 messages🔥): 
        
- **Python 安装问题**: `@night_w0lf`、`@teknium`、`@jaisel` 和 `@.beowulfbr` 之间就 Python 安装问题进行了深入讨论。`@jaisel` 反馈其 Python 安装问题干扰了工作。
- **Python 环境建议**: `@night_w0lf` 和 `@.beowulfbr` 建议 `@jaisel` 为每个独立项目使用 Python 虚拟环境，以避免未来发生此类冲突。
- **排查 Python 环境**: 具体建议包括检查 `/usr/local/bin/python3` 是否存在以及列出已安装的包。
- **项目名称建议**: `@gabriel_syme` 和 `@night_w0lf` 讨论了可能的项目名称，包括 "Megabytes" 和 "ThaumatoAnakalyptor"。
- **游戏讨论**: `@gabriel_syme` 和 `@Error.PDF` 简短讨论了 "GTA 5 Apache 2.0 Open Source"，这显然是一个游戏优惠。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (31 条消息🔥): 

- **Zeta Alpha AI 趋势 - 2023年12月 & Mamba: 具有选择性状态空间的线性时间序列建模**：`@burnydelic` 分享了两个 YouTube 链接，讨论了最新的 AI 趋势，包括 [Gemini, NeurIPS & Trending AI Papers](https://www.youtube.com/watch?v=6iLBWEP1Ols)，以及一段关于 [Mamba 论文的解读视频](https://www.youtube.com/watch?v=9dSkvxS2EB0)，这是一种具有选择性状态空间的线性时间序列建模方法。
- **Nucleus X 模型**：`@.benxh` 提到了托管在 [Hugging Face 上的 Nucleus X 模型](https://huggingface.co/NucleusAI/Nucleus-X)，并表达了他们认为“是时候超越 Transformer 了”的观点。`@teknium` 询问该模型是否支持 Hugging Face 推理，`@.benxh` 予以确认并提供了相关的 Python 代码片段。然而，随后他们提到该模型已无法访问。
- **关于 GPT-4 (Sydney) 的讨论及 Alignment 的重要性**：用户 `@giftedgummybee` 分享了他们使用相对未经审查的 **GPT-4 (Sydney)** 版本的经验，并评论了 Alignment 如何影响模型性能。他们还认为 OpenAI 故意“削弱（sandbagged）”了该模型以防止其产生独立思考，正如 OpenAI 在其预备框架（preparedness framework）中关于“说服（Persuasion）”和“模型自主性（Model autonomy）”所概述的那样。这引发了与 `@gabriel_syme` 之间关于以符合 OpenAI 安全框架的方式构建解决方案的可行性和潜在益处的讨论。
- **Nucleus X 模型从 Hugging Face 删除**：`@.benxh` 和 `@gabriel_syme` 注意到之前讨论的 **Nucleus X 模型** 已在 Hugging Face 上失效，引发了关于其被删除原因以及是否有人提前下载了该模型的猜测。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (240 条消息🔥🔥): 

- **OpenRouter 上模型的流行度**：`@night_w0lf`、`@ldj` 和 `@allanyield` 讨论了 OpenRouter 的使用统计数据，指出 `Capybara` 是 OpenRouter 上使用率前二的模型，其使用量甚至超过了 `Mixtral` 和 `GPT-4-turbo`。`ldj` 推测，尽管 `Capybara` 在 Benchmarks 中表现并不突出，但其多语言推理能力可能促成了它的流行。
- **CodeNinja 模型发布**：`@.beowulfbr` 宣布了他的新开源模型 `CodeNinja`，旨在作为一个可靠的代码助手，并受到了社区的积极评价。它是 `openchat/openchat-3.5-1210` 模型的增强版，在超过 400,000 条代码指令上进行了训练。可以在此 [Hugging Face 页面](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B)找到它。
- **关于模型 Fine-Tuning 和 Benchmarking 的讨论**：包括 `@nruaif`、`@giftedgummybee` 和 `@teknium` 在内的几位用户讨论了对 `Dall e 3` 和 `Gemini` 等模型进行 Fine-Tuning 的实用性，以及这些模型的 Benchmarking 方式。`nruaif` 提到使用 Gemini 为他的模型生成数据集，而 `giftedgummybee` 建议创建私有 Benchmarks 以更个性化地衡量模型有效性。
- **潜在的模型抄袭事件**：`@weyaxi` 和 `@.beowulfbr` 讨论了一起潜在的抄袭事件，另一位作者发布的模型似乎与 `@weyaxi` 的一个模型具有相同的哈希值和权重。双方都同意 `@weyaxi` 在公开此问题之前应先联系该作者寻求澄清。
- **关于 Benchmarking 和评估的讨论**：`@mihai4256`、`@benxh` 和 `@gabriel_syme` 等用户讨论了对 LLM 进行更相关且更稳健的评估和 Benchmarking 技术的需求。他们讨论了 Benchmarking 接口的可能性以及在模型评估中使用 Elo rating 的方法。


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (5 条消息): 

- **Casper 在各 AI Discord 社区的活跃情况**：用户 `@oleegg` 发现 `@casper_ai` 出现在多个 AI Discord 群组中，并对其才智表示赞赏，称他“*挺厉害的（kinda cracked）*”且是“*个聪明人*”。


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 条消息): 

- **AI 脚本开发**：用户 `@vic49.` 正在尝试独立开发一个脚本，拒绝了 `@.beowulfbr` 提出的使用 **LM Studio** 的建议。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **AI 模型对比**：用户分享了对包括 ChatGPT、Bing Bard 和 Mistral-medium 在内的各种 AI 模型的个人体验和对比。提出了关于 Claude Instant 和 Claude 2.1 性能的具体问题，并讨论了 GPT4 的使用限制 [*ai-discussions*]。
- **API 利用**：提到了围绕 API 使用的问题，例如使用 OpenAI API 构建 Nodejs CLI 应用程序，以及对未经授权访问可能滥用 API keys 的担忧。还解决了一个关于 API 用户未被识别为其组织成员的问题 [*ai-discussions*, *openai-chatter*, *gpt-4-discussions*]。
- **ChatGPT 功能**：涉及 ChatGPT 功能的主题包括在 ChatGPT 中使用 Agents、聊天机器人意图识别软件、聊天历史记录、聊天和 ChatGPT Classic 配置文件中的消息限制，以及“继续生成”按钮的使用 [*ai-discussions*, *openai-questions*, *gpt-4-discussions*]。
- **平台支持**：讨论了用户对平台支持的体验，一些报告提到了对 OpenAI bot 支持的负面体验，并建议针对某些问题联系 OpenAI 支持部门 [*openai-chatter*]。
- **Prompt Engineering**：提到了关于响应中不需要的输出的问题和解决方案，并向 Prompt Engineering 社区发出了呼吁 [*prompt-engineering*, *api-discussions*]。
- **AI 工具与系统**：分享了 "Code Anything Now! Definitely Obey" (CAN DO) GPT creator 系统的介绍，这是一个旨在允许 GPT Agents 执行 shell 命令和管理 Git 任务的工具 [*gpt-4-discussions*]。

**OpenAI 频道摘要**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (14 条消息🔥): 
        
- **AI 模型对比**：用户 `@miixms` 和 `@i_am_dom_ffs` 讨论了不同语言模型的性能。`@miixms` 发现 **ChatGPT** 在处理简单任务时表现优于 Bing Bard，而 `@i_am_dom_ffs` 分享了个人经验，认为 **Mistral-medium** 相比 **Mixtral** 没有任何提升。用户 `@eljajasoriginal` 询问了 **Claude Instant** 和 **Claude 2.1** 的性能。
- **API 信息**：用户 `@utkarsh` 询问了关于使用 **OpenAI API 制作 Nodejs CLI 应用程序**的问题。`@miixms` 建议向 ChatGPT 询问同样的问题。
- **意图识别**：用户 `@the_black_hat` 咨询了优秀的**聊天机器人意图识别软件**。
- **在 ChatGPT 中使用 Agents**：用户 `@jeannehuang86` 寻求关于在 **ChatGPT 中使用 Agents** 扮演多个角色的建议。`@lugui` 建议利用 API 来拥有独立的 Agents，但提到在 ChatGPT 中，同一个 Agent 需要执行所有角色。`@michael_6138_97508` 建议考虑 **Azure OpenAI 平台**。 
- **GPT 模型的使用限制**：用户 `@exilze` 提到 **GPT4 有使用限制**，并分享说由于这些限制，他们无法使用 ChatGPT。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (175 条消息🔥🔥): 
        
- **对未经授权访问和数据泄露的担忧**：用户 `@infidelis` 报告称其 ChatGPT 历史记录中多次出现非本人发起的对话，引发了对可能的数据泄露或账户被未经授权访问的担忧。他们更改了密码，运行了杀毒软件，并检查了可疑的浏览器扩展，但新的聊天记录仍继续出现。用户 `@lugui` 建议该问题可能是由于凭据被盗引起的，并建议 `@infidelis` 联系 OpenAI 支持团队。

- **ChatGPT 与 API 使用的对比**：`@infidelis` 考虑从 ChatGPT 转向使用 API，因为后者对他们的使用情况来说可能更便宜。然而，`@sxr_` 警告他们，如果他们的 OpenAI 账户已被入侵，转向 API 可能会增加风险，因为潜在的黑客可能会滥用他们的 API key。

- **DALL-E 体验与图像创作中的拼写问题**：`@jonahfalcon.` 分享了他们与 DALL-E 图像生成的幽默经历，指出 AI 经常在图像中拼错单词。`@lugui` 将此归因于 DALL-E 而非 ChatGPT，因为后者仅负责生成文本 Prompt。

- **平台支持方面的困难**：用户 `@infidelis` 谈到了他们对 OpenAI bot 支持的负面体验，认为它在解决特定问题时效率低下。

- **围绕 ChatGPT 和 API 错误的讨论**：用户 `@mysticmarks1` 提出了一个关于 API 使用错误的假设，并开玩笑说客户会因为错误的支出而责怪 OpenAI。

### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (39 条消息🔥): 
        
- **iOS 上的 GPT 模型问题**：用户 `@readingpalms` 反馈，他们在 **iOS app** 上的对话在第一个 Prompt 之后会从 **GPT4** 切换到 **3.5**。该问题似乎仅限于 iOS app，且在用户重新安装 app 后依然存在。

- **聊天记录和模型输出问题**：包括 `@lemar` 和 `@skrrt8227` 在内的几位用户报告了 **聊天记录消失** 以及模型出现 **非预期图像输出** 的问题。这些问题尚未得到直接解决。

- **GPT 对话的数量限制**：用户 `@m54321` 报告称，每个对话达到一定 **未公开的消息数量** 后会触发错误，迫使他们开启新对话。该问题似乎仍未解决。

- **“真人验证”问题**：用户 `@3daisy` 遇到了持续出现真人验证提示的问题。同步系统时间解决了他们的问题。

- **ChatGPT Classic Profile 消息限制**：`@felpsey` 确认 **ChatGPT Classic profile** 中存在 **40 条消息的限制**，这适用于基于 GPT-4 的模型。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (8 条消息🔥): 
        
- **API 用户识别**：`@reinesse` 提出了一个问题：尽管拥有独立的 API key 和 Reader 访问权限，但 API 用户未被识别为组织的一部分。他们寻求解决此问题的建议。
- **“CAN DO” GPT Creator 系统介绍**：`@_jonpo` 分享了一个名为 [**"Code Anything Now! Definitely Obey" (CAN DO) gpt creator system**](https://chat.openai.com/g/g-ibrTsdfV0-can-do-creator) 的新工具，它能让 GPT Agent 执行 Shell 命令并管理 Git 任务。该工具旨在克服传统 GPT Agent 的某些限制。
- **发布 GPT 链接**：`@loschess` 告知群组成员有一个专门用于发布 GPT 链接的板块。
- **了解 Rate Limits**：当 `@phobir` 询问点击“继续生成 (Continue generating)”按钮是否会计入 40 次请求/3 小时的限制时，`@solbus` 确认会计算在内。
- **自定义 GPT 的消息上限**：`@draculabutbackwards` 询问如何绕过自定义 GPT 的消息上限，`@satanhashtag` 澄清目前无法实现。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (3 条消息): 
        
- **问题解决**：用户 `@exhort_one skrrt8227` 报告称，他们成功修复了之前讨论过的关于回复中出现非预期输出的问题。
- **问题澄清**：`@exhort_one skrrt8227` 进一步解释说，他们指的是回复开头、结尾和页脚的文本。他们表示已经尝试了不同的方法来解决这个问题，包括添加视觉效果来突出显示有问题的部分。
- **社区呼吁**：用户 `@beanz_and_rice` 联系了 Prompt Engineering 社区的其他成员，推测是为了进行讨论或寻求帮助。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (3 条消息): 
        
- **修复输出答案问题**：`@exhort_one skrrt8227` 提到他们修复了与输出答案相关的问题。
- **文本替换困难**：`@exhort_one skrrt8227` 尝试了对回复开头、结尾和页脚文本进行不同的替换，并最终在截图中通过圈出和划掉这些部分来处理。
- **Prompt Engineering**：`@beanz_and_rice` 询问频道中是否有 Prompt Engineer。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 摘要

- 服务器中关于 **Mixtral Access and Versioning** 的讨论，用户如 `@hydroxide.` 和 `@tesseract_admin` 致力于了解使用和访问详情，而 `@sublimatorniq` 指出 `Perplexity` 托管了 `Mixtral`。
- 围绕 **Dolphin Mixtral for Chatting** 的各种疑虑和建议，由 `@cl43x` 发起，主要涉及在使用 `Ollama` 托管时遇到的 `Mistral` 模型审查问题，随后在 `@iammatan` 和 `@ethux` 的引导下探索了 `dolphin-mixtral` 模型作为去审查选项。对话中 `@ethux` 还推荐了 [Dolphin 2.2.1 Mistral 7B - GGUF](https://huggingface.co) 模型的链接。
- 针对 **Errors in Oogaboogas' Text GUI** 以及 'Prompting' 问题的澄清与解决尝试：`@cl43x` 得到了 `@ethux` 的协助，后者还建议加入 AI 社区以进一步学习 'Prompting'。
- `@blueridanus` 提出了一系列 **Uncensoring Methods**（去审查方法），从修改 prompt 格式到角色设定和智能 prompt。对话还包括 `@faldore` 和 `@ethux` 关于使用不同模型以获得不同结果的建议，以及指向 `GitHub` 上 [prompt repository](https://github.com/ehartford/dolphin-system-messages) 的链接。
- 频道贡献还涵盖了由 `@hovercatz` 和 `@dillfrescott` 主导的 **Tech Requirements Clarifications**（技术要求说明），他们解释了在解读 `HuggingFace` 技术规范时 VRAM 与系统 RAM 的区别。
- 在 **deployment** 频道中，`@dutchellie` 提醒成员注意一个 Bot 的去审查特性，收到了来自 `@weird_offspring` 的正面反馈和来自 `@frosty04212` 的批评，展示了不同的用户反应。
- 关于 **Code Generation**、**Efficiency** 以及 **Mistral-medium, Mistral-small, and GPT-3.5-Turbo 性能** 的比较与讨论在 random 频道非常活跃，`@jackson_97091`、`@poltronsuperstar` 和 `@victronwolfson` 分享了他们的见解和经验。
- 最后，`@victronwolfson` 讨论了 **API Support for Tools** 和 **Discord Bot Model Switching**，指出了与 OpenAI 等工具兼容性的局限性，并分享了他开发的能够切换模型的 Discord Bot。

**Mistral 频道摘要**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (79 条消息🔥🔥): 
        
- **Mistral Access and Versioning**: 用户 `@hydroxide.` 澄清了 `mistral-small` 对应于 `Mixtral 8x7B`，并询问了 `mistral-medium` 的具体细节；同时 `@tesseract_admin` 请求获取 API 访问信息以及在 Mistral 官方托管推理之外使用 `Mixtral` 的方法。`@sublimatorniq` 补充说 `Perplexity` 托管了 `Mixtral`。

- **Potential Use of Dolphin Mixtral for Chatting**: 用户 `@cl43x` 对在使用 `Ollama` 托管 `Mistral` 模型时遇到的明显审查感到沮丧。他在 `@iammatan` 和 `@ethux` 的指导下转向了 `dolphin-mixtral` 模型，据称该模型审查较少。`@ethux` 提供了 `HuggingFace` 上 `Dolphin 2.2.1 Mistral 7B - GGUF` 模型的链接，`@cl43x` 表示正在下载并准备在 `Oogaboogas web UI` 上尝试。关于所需 VRAM 的担忧得到了解答，`@cl43x` 说明其系统有 8GB VRAM，而 `@ethux` 根据其使用 RTX 3090 和 RTX 2080 Ti 的经验，认为这很可能可行。

- **Errors and Little Understanding of 'Prompting'**: `@cl43x` 提出了在 `Oogaboogas` 文本 GUI 上加载模型时遇到错误的问题，该问题在对话结束时仍未解决。该用户还承认不理解 'Prompting' 一词，对此 `@ethux` 建议关注各种 AI 社区以学习更多知识，例如 `TheBloke` 的社区和 `LearnAI together`。

- **Possible Uncensoring Methods**: 用户 `@blueridanus` 建议通过更改 prompt 格式或创建角色并使用智能 prompt 来对 AI 进行“去审查”。`@ethux` 提出了针对不同结果使用不同模型的想法（例如使用 Mistral Instruct 来遵循指令），`@faldore` 分享了 `GitHub` 上 [prompt repository](https://github.com/ehartford/dolphin-system-messages) 的链接。

- **Tech Requirements Clarifications**: 最后，`@hovercatz` 和 `@dillfrescott` 澄清了在解读 `HuggingFace` 工具技术规范时 VRAM 和系统 RAM 之间的区别。

### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (3 条消息): 
        
- **无审查聊天机器人**：`@dutchellie` 提醒用户该机器人会回答所有查询，强调了其无审查（uncensored）的性质。
- **用户对无审查机器人的反应**：
    - `@weird_offspring` 对这一功能表示热衷，强调了其改进和学习的潜力。
    - `@frosty04212` 持不同观点并表示不满，认为该机器人过于对齐（aligned），且对他们的请求没有反应。


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (5 条消息): 
        
- **比较代码生成**：`@jackson_97091` 和 `@poltronsuperstar` 讨论认为 Mistral 的 **code generation**（代码生成）表现相当不错。
- **Mistral-medium 对比 GPT-3.5-Turbo**：`@victronwolfson` 分享了他的初步测试结果，称 **Mistral-medium 明显优于 GPT-3.5-Turbo**。
- **Mistral-small 对比 GPT-3.5-Turbo 的效率**：`@victronwolfson` 还提到 **Mistral-small 的性能与 GPT-3.5-Turbo 相当，而成本仅为后者的 66%**。
- **API 对工具的支持**：`@victronwolfson` 强调 **API 目前不支持像 OpenAI 那样的 tools**，但他成功通过设置中间层（middleman）使所有三个模型都能以类似工具的方式进行响应。
- **Discord 机器人模型切换**：`@victronwolfson` 分享了他的 **Discord 机器人可以根据指令在模型之间切换**，这方便他进行各种测试。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- 广泛讨论了 **model fine-tuning**（模型微调），特别是在一个数据集上微调模型后，再在另一个数据集上继续微调，用户们在探索如果模板保持不变，这种方式的有效性。另一个焦点是针对特定任务微调模型以及执行此操作的最佳平台。
- 技术性探讨了 Mistral 在更新后出现的 **regression**（回归问题），包括对变化原因的推测，特别是引用了 dropout commit 作为潜在原因。
- _“之前的 dropout 设置为 0.0，但现在设置为遵循 config 值，该值可能为 None。”_ (`@caseus_`)
- 深入讨论了 **GPT 和 LoRA/QLoRA 模型的训练与推理**，以及在本地和 Axolotl-cli 上测试这些模型时可能出现的 bug。关注点在于模型的连贯性以及长时间运行产生的流水句（run-on sentences）问题。
- 用户 `@xzuyn` 创建了一个 **手动策划的 AP News 2023 Tiny Dataset**，共有 288 个样本，旨在使语言模型对实时事件更加敏感。该数据集包含 AP News 的文章，重点关注时事，这可能会根据收集时间引入偏好。可以通过 [HuggingFace 网站](https://huggingface.co/datasets/PJMixers/AP-News-2023-Tiny)访问该数据集。
- 讨论了训练大模型时的 **hardware limitations**（硬件限制）。用户分享了即使修改了各种模型和优化器配置，仍然遇到 Out-of-Memory (OOM) 错误的经历。
- 讨论了由于硬件限制，**将 7B Mistral 模型扩展为更大模型**并仅训练某些层的可能性。讨论中考虑了 FFT 以及训练接口层（interface layers）的选项。


**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (1 条消息): 
        
- **在多个数据集上微调模型**：`@noobmaster29` 询问是否有人尝试过在一个数据集上微调模型，然后继续在另一个数据集上微调。他们探讨了在模板保持不变的情况下该想法的有效性。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (6 条消息): 
        
- **Mistral 中的回归问题**：`@nanobitz` 报告了 **Mistral** 在 Mixtral 更新之前的一个 commit 中出现了回归。他们质疑该回归是 Mistral 特有的，还是也会影响其他模型。
- **关于回归原因的讨论**：`@caseus_` 推测回归是否可能是由于 `@casper_ai` 提到的 `dropout commit` 变更引起的。之前的 dropout 设置为 0.0，但现在设置为遵循 config 值，该值可能为 None。
- **A6000 性能测量**：`@caseus_` 进一步提到了在单张 **48GB A6000** 上测试 FFT Mistral。
- **分享 Mistral 配置文件**：`@caseus_` 还分享了一个用于 Mistral 的 yml 配置文件的 [gist 链接](https://gist.github.com/winglian/1a519b72f9561170c0d2bf58cee93a09)。

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (62 条消息🔥🔥): 
        
- **关于 GPTs 和 LoRA/QLoRA 模型训练与推理的讨论**：用户 `@self.1` 和 `@noobmaster29` 讨论了各种配置、End of Sentence (EOS) 令牌的行为，以及在使用 ooba 配合 chatml 模板时可能存在的 bug。他们在本地和使用 Axolotl-cli 测试了训练好的模型，重点关注模型连贯性和过长的流水句问题。
- **使用 QLoRA 将 Mistral 模型扩展至 13B**：`@xzuyn` 和 `@caseus_` 探讨了将 7B Mistral 模型扩展为更大模型的可能性，由于硬件限制，仅训练某些层。他们探索了使用 Freeze-Setting Techniques (FFT) 以及训练接口层的选项。
- **训练大模型时的硬件限制**：`@xzuyn` 在修改各种模型和优化器配置时仍然遇到 Out-of-Memory (OOM) 错误，并寻求如何规避这些问题的建议。
- **在多个 ShareGPT 对话上运行推理**：`@semantic_zone` 询问了如何在一组对话上运行推理，其中每个对话都包含多条历史消息。
- **关于微调模型的澄清**：`@tolki` 就针对特定任务微调模型、消费级硬件是否适合微调、该过程的最佳在线平台以及在低使用率的生产应用中部署模型的最佳方式寻求建议。`@caseus_` 对这些咨询提供了回复。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 条消息): 
        
- **AP News 2023 Tiny 数据集**：用户 `@xzuyn` 正在手动创建一个包含 **AP News 文章** 的数据集，可在 [HuggingFace 网站](https://huggingface.co/datasets/PJMixers/AP-News-2023-Tiny)上获取。这些文章包括截至收集日期为止的近期事件，旨在提高语言模型对实时事件的敏感度。
- 该数据集目前包含 **288 个样本**，并将持续添加新条目。内容专注于**当前话题**，可能偏向于收集期间 AP News 主页上的话题。
- 文章以 **Markdown** 格式呈现，包含的最旧文章是几个月前的，而最新的则来自收集当天。 
- 此外，据报告该数据集 **99% 干净**，唯一的问题可能是潜在的重复条目。由于数据收集是手动的，每个样本在纳入之前都会经过质量和相关性检查。

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- `@tomgale_` 分享了与一位纽约艺术家的持续讨论，该艺术家正在探索 AI 与神经网络的交叉领域。这导致发现了与 **Jan Sloot Digital Coding System** 的潜在联系，并提出了未来与 **Hugging Face** 合作的建议。
- `@ehristoforu` 寻求将 Safetensors 转换为 Diffusers 格式的帮助。`@not_lain` 提供了一个 Python 解决方案，并通过提供用于转换过程的 Google Colab 脚本提供了进一步的协助。
- `@radens__` 讨论了在 M1 Pro 上用 Swift 实现 Mistral 的挑战，原因是 Mistral 权重以不支持的 bf16 格式分发。探讨了将权重重新编码为 fp16 或 fp32 的可能性，`.tanuj.` 建议使用 Apple 的 **MLX framework**。
- `@ivy2396` 询问了 HuggingFace 与其分布式 GPU 云平台之间可能的合作。
- 用户 `@neuralink` 分享了他们在实现 **DoReMi** 项目和 3D 并行中的端到端 FP8 训练方面的进展。
- `@devspot` 重点介绍了 **Outfit Anyone AI** 和几个新的 HuggingFace spaces，`@th3_bull` 分享了一个讨论使用 HuggingFace spaces 进行 NLP 的西班牙语视频。
- “I made this” 频道分享了多个项目——从 `@yjg30737` 的 [Stable Diffusion 生成图像下载器](https://www.kaggle.com/code/yoonjunggyu/stable-diffusion-generated-image-downloader) 到 `@cloudhu` 分享的玩 SpaceInvadersNoFrameskip-v4 的 DQN Agent 模型，以及 `@om7059` 在 PyTorch 中实现的 Google DeepDream 程序。
- 用户 `@blackbox3993` 发布了一个问题，即微调后的 Mistral 模型在 HuggingFace 上重新加载时表现得像基础模型，而不是微调后的变体。他们提供了[源代码](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919)用于分析和排查故障。
- `@blackbox3993` 和 `@tomgale_` 讨论了潜在的计算机视觉项目和合作。`@srikanth_78440` 寻求微调多模态语言与视觉模型的帮助。
- `@stroggoz` 提出了一个关于 Mixture of Expert 架构在语言模型任务之外的应用问题，`@hafiz031` 寻求关于如何为 [Open Book Question Answering/Retrieval Augmented Generation](https://stats.stackexchange.com/q/635603/245577) 系统有效分块大型语料库的建议。

**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (38 条消息🔥): 
        
- **艺术家转向 AI**：`@tomgale_` 讨论了他与一位对 AI 和神经网络领域表现出兴趣的纽约艺术家的通信。他将艺术家的工作与 1999 年的一项发明 Jan Sloot Digital Coding System 联系起来。Tom 希望能与 **Hugging Face** 取得联系，因为他相信自己可以证明该编码系统是基于对比度音高和 VCR 的某种 LLM，并表达了希望在此事上获得帮助的愿望。 

- **将 Safetensors 转换为 Diffusers 格式**：`@ehristoforu` 询问如何将 Safetensors 转换为 Diffusers 格式。`@not_lain` 提供了一个使用 ***diffusers*** 库实现的 Python 代码片段，并解决了随后的 *omegaconf* 安装问题。 

- **用于转换的 Colab 代码**：`@not_lain` 还通过在 Google Colab 上创建脚本并分享给 `@ehristoforu` 以帮助转换过程，提供了进一步的协助。他后来澄清该脚本仅针对一个模型，但也可以应用于其他模型。

- **M1 Pro 上的 LLM**：`@radens__` 寻求关于在 M1 Pro 上用 Swift 实现 Mistral 的建议。他对比特权重以 bf16 格式分发表示担忧，因为 M1 Pro 不支持该格式。他探讨了将权重重新编码为 fp16 或 fp32 的想法。

- **M1 上的 MLX framework**：`.tanuj.` 建议 `@radens__` 考虑 Apple 的 **MLX framework**，并提供了一个 [GitHub 链接](https://github.com/ml-explore/mlx-examples/tree/main/llms/mistral)以获取更多信息。

- **合作咨询**：`@ivy2396` 表达了探索 HuggingFace 与其分布式 GPU 云平台之间合作机会的兴趣，并寻求联系人以推进这些讨论。


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (1 条消息): 
        
- **DoReMi 实现进展**：用户 `@neuralink` 分享了他在实现 **DoReMi** 项目方面的进展，表示已经完成了 **20%**。
- **3D 并行中的端到端 FP8 训练**：`@neuralink` 还提到他成功实现了 3D 并行中 **10%** 的端到端 FP8 训练，**FP8 kernels** 除外。

### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (2 messages): 
        
- **Outfit Anyone AI 和 HuggingFace Spaces**：`@devspot` 提到了 **Outfit Anyone AI** 以及目前在 HuggingFace 上可用的几个新 Spaces。他们制作了一个[视频](https://youtu.be/QBCDgcQlS6U)来介绍这些更新，并帮助用户及时了解 HuggingFace 上的最新模型。
- **HuggingFace 的 NLP Spaces**：`@th3_bull` 分享了一个关于使用 HuggingFace spaces 进行从 0 到 100 的 NLP 学习的[西班牙语视频](https://m.youtube.com/watch?v=wSI8shazYaA&list=PLBILcz47fTtPspj9QDm2E0oHLe1p67tMz&index=7&pp=iAQB)。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (13 messages🔥): 
        
- **Stable Diffusion 生成图像下载器**：`@yjg30737` 创建并公开了一个 [Kaggle notebook](https://www.kaggle.com/code/yoonjunggyu/stable-diffusion-generated-image-downloader)，旨在提高图像生成质量并就该主题教育他人。
- **基于 Web 的混音实验**：`@.bigdookie n278jm` 分享说，他们正在开发一个极简的基于 Web 的混音实验，以尝试 v7 wavsurfer 的更新。
- **randomizer_arr 的参数**：`@andysingal` 向 `@yjg30737` 询问了 randomizer_arr 的参数以及生成的图像来源。
- **与 Mistral AI API 的对比**：`@andysingal` 还询问了 `@qbert000` 他们的项目与 Mistral AI API 相比如何，以及是否容易与 langchain、llamaindex 等任何操作系统组件集成。
- **玩 SpaceInvadersNoFrameskip-v4 的 DQN Agent**：`@cloudhu` 分享了一个使用 [stable-baselines3 库](https://github.com/DLR-RM/stable-baselines3)和 [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) 训练的玩 SpaceInvadersNoFrameskip-v4 的 DQN Agent 模型。
- **PyTorch 版 DeepDream 程序**：`@om7059` 在 PyTorch 中实现了 Google 的 DeepDream 程序，该程序可以将普通图像转换为梦幻般的构图，并在 Twitter 上分享了一些生成的图像 ([链接](https://x.com/alve_om/status/1738968534347292945?s=20))。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 
        
- **Mistral 模型微调问题**：`@blackbox3993` 发布了他们在微调 Mistral 模型时遇到的问题。在 HuggingFace 上保存并重新加载模型后，`evaluate` 函数的结果似乎对应于基础模型，而不是微调后的变体。他们分享了[过程中使用的代码](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919)，以便分析潜在错误所在。


### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 messages): 
        
- **针对就业市场的计算机视觉项目**：`@blackbox3993` 正在寻求有关计算机视觉项目的建议，以增强其就业竞争力。他们也对合作开发一些酷炫的项目持开放态度。
- **协作请求**：`@tomgale_` 正在寻求一个项目的帮助，他已将详细信息发布在 general 频道。他特别指出 `@blackbox3993` 可能会感兴趣。
- **微调多模态 LLM**：`@srikanth_78440` 正在寻求有关使用自定义图像数据集微调 LLAVA2 等多模态语言与视觉模型的指令帮助。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (2 messages): 
        
- **用于非 LLM 任务的 Mixture of Expert 架构**：`@stroggoz` 提出了一个问题，即在命名实体识别（NER）等非语言模型任务中，为何缺乏 Mixture of Expert 架构。
- **为富上下文检索对大型语料库进行分块**：`@hafiz031` 寻求有关如何有效对大型语料库进行分块以优化检索的建议。他们的目标是构建一个 [Open Book Question Answering](https://huggingface.co/tasks/question-answering) / [Retrieval Augmented Generation](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) 系统。他们的问题详细说明了潜在的挑战和上下文选择问题。他们在 StackExchange 上分享了详细咨询的[链接](https://stats.stackexchange.com/q/635603/245577)。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 条消息): 
        
- **微调 Mistral 模型的问题**：用户 `@blackbox3993` 报告了一个关于他们在 HuggingFace 上微调并保存的 **Mistral 模型**的问题。他们提到在加载模型时，没有得到预期的结果，因为 `evaluate` 函数产生的结果似乎与基础模型（base model）相似。他们在 [huggingface/peft GitHub issues 页面](https://github.com/huggingface/peft/issues/1253#issuecomment-1866724919)分享了正在使用的代码，并请求帮助诊断可能出现的问题。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 用户 `@raghul_64090` 在为 mistral 7b 使用 openllm 时遇到了 **TypeError: 'MistralRunner' object is not callable**，并在 general 和 tutorials 频道寻求解决该问题的指导。
- `@meowthecatto` 的提问引发了关于学习 LangChain 的机器人和资源的讨论。`@seththunder` 回应并推荐了多个资源，如 Discord 上的 kapa.ai、Github 上的 dosu-bot 以及与 LangChain 文档聊天。
- 用户 `@doublez_` 在通过 pip 安装外部包时遇到了 **`error: externally-managed-environment`** 问题。
- `@ivanryzk` 对 Zapier 与 LangChain 的集成状态表示困惑，指出尽管 LangChain 文档称其已弃用（deprecated），但 Zapier 文档仍列出了 LangChain。
- `@cryptossssun` 询问如何从 PDF 中提取 JSON 格式数据，特别是从基于图像的表格 PDF 文件中提取，随后收到了 `@quantumqueenxox` 和 `@rajib2189` 的指导。
- `@jazzy3805` 宣布了一家处于 AI、游戏和区块链技术交汇点的公司的 **AI 工程师**和**游戏开发人员**职位空缺。AI 工程师职位要求熟练使用 LangChain。
- 用户 `@reachusama` 在 share-your-work 频道分享了一个推广个人 GitHub 项目的 [LinkedIn 帖子](https://www.linkedin.com/posts/reach-usama_github-reachusamaupworkgpt-upworkgpt-activity-7142620647964176385-B2ic?utm_source=share&utm_medium=member_ios)。
- `@shamspias` 介绍了一个新的 Gemini API Web 应用程序项目，专门使用 LangChain 为 Gemini 设计。该项目是[开源的](https://github.com/shamspias/langchain-gemini-api)，具有许多功能，如多模态对话能力、FastAPI 构建、用于持久化对话历史的 Redis 集成、与各种应用程序的兼容性、由 Redis 支持的简单 API 机制，以及异步和流式响应。鼓励用户探索、使用并为该项目做出贡献。

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (19 条消息🔥): 
        
- **MistralRunner TypeError**：用户 `@raghul_64090` 在为 mistral 7b 使用 openllm 时遇到了 **TypeError: 'MistralRunner' object is not callable**，并寻求解决此问题的指导。
- **机器人查询**：用户 `@meowthecatto` 询问了一个可以回答 LangChain 相关问题的机器人。`@seththunder` 的回答指出有多种资源可用，包括 Discord 上的 kapa.ai、Github 上的 dosu-bot 以及与 LangChain 文档聊天。
- **pip install -e 错误**：用户 `@doublez_` 提出了他们在通过 pip 安装外部包时遇到的问题 —— `error: externally-managed-environment`。
- **Zapier 集成**：`@ivanryzk` 向 `@jonanz` 和 `@251662552210210816` 询问 Zapier 集成的更新，理由是尽管 LangChain 文档提到它已弃用，但 Zapier 文档仍列出了 LangChain。
- **PDF 数据提取**：`@cryptossssun` 寻求关于从 PDF（特别是基于图像的表格 PDF 文件）中提取 JSON 格式数据的建议，并得到了 `@quantumqueenxox` 和 `@rajib2189` 的指导。
- **职位发布**：`@jazzy3805` 宣布了一家处于 AI、游戏和区块链技术交汇点的公司的 **AI 工程师**和**游戏开发人员**职位空缺。AI 工程师角色特别要求精通 LangChain。有意者请私信了解更多详情。

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **自我推广**：用户 `@reachusama` 分享了他的 [LinkedIn 帖子](https://www.linkedin.com/posts/reach-usama_github-reachusamaupworkgpt-upworkgpt-activity-7142620647964176385-B2ic?utm_source=share&utm_medium=member_ios)，内容关于他的新 GitHub 项目。
- **Gemini API Web 应用程序**：用户 `@shamspias` 介绍了专门为 Gemini 设计、使用 Langchain 构建的新项目。该项目的主要特性包括：
    - 多模态对话能力。
    - 使用 FastAPI 构建。
    - 集成 Redis 以实现持久化对话历史。
    - 兼容各种应用程序。
    - 由 Redis 支持的简单 API 机制。
    - 异步和流式响应。
  
  该 Gemini API 应用程序已开源，可在 [GitHub](https://github.com/shamspias/langchain-gemini-api) 上获取。鼓励用户探索、使用并为该项目做出贡献。


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **使用 openllm 运行 Mistral 7b 的问题**：用户 `@raghul_64090` 报告遇到了 `TypeError: 'MistralRunner' object is not callable` 问题，并寻求关于该错误含义的指导。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- 关于 Mixtral 实现和训练配置的详细讨论：
    - 询问 Mixtral 训练中的 "**Gate Layer Freezing**" 技术，引用了 Eric Hartford 的一条 [Tweet](https://twitter.com/erhartford/status/1737350578135834812)，该推文阐述了此方法的重要性。问题在于该技术在 Axolotl 中的应用。
    - 分享了 Eric Hartford 关于 Mixtral-8x7b 的 **Dolphin 2.6 训练配置**，可在此处 [访问](https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b/blob/main/configs/dolphin-mixtral-8x7b.yml)。这种方法是否最有效受到了质疑。
    - **Dolphin 2.6 模型**的性能讨论，指出在 1.5-epoch 时比前代更具 "dolphin" 特性，且未发现模型拒绝回答的情况。
    - 询问在 H100 pod 中使用 Mixtral 脚本成功进行 **8-bit 训练**的情况，特别是与 Axolotl 相关的部分。

- 关于快速进行 OpenAI 兼容操作的软件对话：
    - 提议一种**软件解决方案**，用于即时的 OpenAI 兼容 API 交互，并可通过热键执行预定义操作，适用于任何操作系统。
    - 推荐了可能满足这些需求的软件：[uniteai](https://unite.ai/) 和 [ClipboardConqueror](https://github.com/aseichter2007/ClipboardConqueror)。
    - 如果此类软件不存在，表示有兴趣开发一个。

**DiscoResearch 频道摘要**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (8 messages🔥): 
        
- **Mixtral 实现中的 Gate Layer Freezing**：用户 `@sebastian.bodza` 引用了 Eric Hartford 的一条 [Tweet](https://twitter.com/erhartford/status/1737350578135834812)，说明了在 Mixtral 训练期间冻结 gate layer 的重要性。他想知道这种技术是否已在 Axolotl 中实现。
- **Dolphin 2.6 训练配置**：`@sebastian.bodza _jp1_` 提到 Eric Hartford 分享了他基于 Mixtral-8x7b 的 Dolphin 2.6 训练配置，可在 [此处](https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b/blob/main/configs/dolphin-mixtral-8x7b.yml) 获取。然而，目前尚不清楚这是否是最有效的训练方法。
- **Dolphin 2.6 模型性能**：用户 `@rtyax` 指出 Dolphin 2.6 在 1.5-epoch 时比 2.5 版本更具 "dolphin" 特性，并注意到他们没有观察到任何模型拒绝。
- **使用 Mixtral 脚本进行 8-bit 训练**：`@tcapelle` 询问是否有人在 H100 pod 中成功使用 8-bit Mixtral 脚本进行训练。Tcapelle 还表示，如果提供配置文件，他愿意运行任何实验。他随后明确该问题适用于 Axolotl 中的应用。

### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (6 条消息): 
        
- **用于快速 OpenAI 兼容提问和操作的软件**：`@.grey_` 询问是否存在一种软件，允许用户通过任何 OpenAI 兼容 API 快速提问，或通过热键对剪贴板内容执行预定义操作，并能跨操作系统运行。该软件理想情况下可以与操作系统的上下文菜单集成，类似于 MacOS 上的 Alfred 或 Spotlight。
- `@.grey_` 还提到，该工具可用于快速提问、预定义操作或在不需要完整上下文的情况下打开对话。这在处理一次性问题或在阅读时需要快速响应时特别有用。
- `@bjoernp` 回复称他们不知道有这样的软件，但认为这个想法很有用。
- `@rtyax` 推荐了 [uniteai](https://unite.ai/)，这是一个可以集成到 IDE 中的 LSP 服务器，并能将 LSP 操作绑定到热键。`@rtyax` 还提到了 GitHub 项目 [ClipboardConqueror](https://github.com/aseichter2007/ClipboardConqueror)，它可能满足 `@.grey_` 的部分需求。
- `@.grey_` 还表示，如果目前还没有此类软件，他有兴趣开发一个。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

只有一个频道有活动，因此无需总结...

- **年度 AI 事件**：`@swyxio` 分享了一个[链接](https://arstechnica.com/information-technology/2023/12/a-song-of-hype-and-fire-the-10-biggest-ai-stories-of-2023/)，该文章总结了 2023 年最大的 AI 新闻。
- **提及 Mamba 模型**：`@swyxio` 简要提到了 Mamba 模型，但未提供更多细节。
- **关于 LangChain 效用的讨论**：`@cakecrusher` 质疑了 LangChain 的效用，并询问为什么要使用它而不是直接用 ChatGPT。作为回应，`@lightningralf` 建议 LangChain 允许在不同的插件（如 Vector Stores）之间轻松切换。
- **构建考虑时间因素的 RAG**：`@swizec` 向 `@gratchie1188` 建议了创建一个重视相关性和时间性的检索增强生成（RAG）模型的可能性。如果简单的时间距离加权效果不佳，可以实现一个基于摘要的递归记忆系统。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

只有一个频道有活动，因此无需总结...

- **构建自主 Web Agent**：用户 `@coopermini` 正在寻求构建自主 Web Agent 的建议和数据集，重点关注可靠性，无论这些 Agent 是基于 GUI 还是基于 LLM。
- **推荐数据集**：`@coopermini` 征求数据集推荐，特别是 mind2web 数据集的替代方案。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

只有一个频道有活动，因此无需总结...

- **关于视觉模型的讨论**：用户 `@thebaghdaddy` 询问领先的视觉模型有哪些，以便与他们训练的模型进行比较。他们提到自己了解 **GPT4V** 模型，但读到报道说它目前处于中等水平。目前尚未看到回复或进一步讨论。