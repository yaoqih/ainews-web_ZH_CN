---
companies:
- meta-ai-fair
- google-deepmind
date: '2023-12-29T10:07:18.273087Z'
description: '**LM Studio** 用户广泛讨论了其性能、在 macOS 上的安装问题，以及即将推出的功能，例如 **Exllama2 支持**和
  **Llava 模型**的多模态功能。对话涵盖了 **GPU 卸载 (GPU offloading)**、**vRAM 利用率**、**MoE 模型专家选择**以及**模型转换兼容性**。社区还针对低效的求助方式进行了探讨，并引用了博客文章《别问能不能问，直接问》(Don''t
  Ask to Ask, Just Ask)。此外，文中还强调了 **ChromaDB 插件**的技术挑战、**服务器与桌面硬件性能**的对比，以及使用 **Autogen**
  保存模型状态的问题。讨论内容还包括与其他聊天机器人的比较，并提到了来自 **meta-ai-fair** 的 **AudioCraft** 和来自 **google-deepmind**
  的 **MusicLM** 等音乐生成模型。'
id: 80165381-8741-4085-9b82-3e1ec521d094
models:
- llava
- exllama2
original_slug: ainews-12262023-not-much-happened-today
people: []
title: 2023年12月26日：今天没发生什么特别的事。
topics:
- gpu-offloading
- vram-utilization
- model-conversion
- moe-models
- multimodality
- model-performance
- hardware-configuration
- model-saving
- chatml
- installation-issues
- music-generation
---

<!-- buttondown-editor-mode: plaintext -->节礼日（Boxing Day）相当平静，但大量的实验和讨论仍在继续。

[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- 关于 **LM Studio** 及其版本和相关问题的广泛讨论，包括性能、macOS 上的安装问题、加载问题等。提到未来预期的增强功能，如 Exllama2 支持和与 Llava 模型的多模态（multimodality）支持。
- 用户表达了对处理服务器中频繁遇到的**低效求助请求**的看法，并分享了博文《别问能不能问，直接问》（Don't Ask to Ask, Just Ask）作为相关参考。
- 讨论了在 LM Studio 中**利用 GPU 进行卸载（offloading）**以及各种模型的最佳设置，还有在更改缓存文件夹和启动 LM Studio 服务器时遇到的困难。
- 在 models-discussion-chat 频道中，**性能评级、MoE 模型中的专家选择以及与 LM Studio 的模型转换兼容性**是讨论的主题。Roleplay AI 以及 LM Studio 中 ChatML 预设的问题也受到了大量关注。
- 用户在硬件配置和 LLM 部署的背景下，询问并讨论了 **vRAM 容量和利用率**。
- 围绕 **ChromaDB Plugin** 的技术困难以及关于将编程作为爱好的滑稽讨论引起了用户的兴趣。
- 进行了关于**配置 LLM 推理硬件**以及比较服务器和桌面设置性能的对话。
- 提到了新 **beta 版本**的特性和关注点，包括自定义设置保存、缓存文件大小以及视觉 LLM 查询。
- autogen 频道的高光内容是关于在 LM Studio 上使用 Autogen **保存 LLM 模型状态**的问题及其最终解决方案。

**LM Studio 频道摘要**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (162 条消息🔥🔥): 
        
- **LM Studio 及其版本**：用户 `@leefnie`、`@kujila`、`@heyitsyorkie`、`@ptable`、`@blizaine`、`@thelefthandofurza`、`@fabguy`、`@rugg0064`、`@yagilb` 等讨论了 LM Studio 的各个方面，特别是其性能、闭源性质、未来的增强功能（如 Exllama2 支持）、与其他聊天机器人的比较以及与 Llava 模型的多模态支持。
  - 特别是 `@kujila` 比较了 LM Studio 与 textgen webui 的易用性和性能，并强调了其商业成功的潜力。
  - `@heyitsyorkie` 澄清了 exllama2 只能在 GPU 上运行，并且 LM Studio 社区一直有支持该功能的请求。
- **如何处理低效的求助请求**：`@fabguy`、`@ptable` 和 `@heyitsyorkie` 深入讨论了低效的求助请求及其给求助者和提供帮助者带来的问题。分享了一篇名为《别问能不能问，直接问》（Don't Ask to Ask, Just Ask，`https://dontasktoask.com`）的博文来证明他们的观点。
- **在 macOS 上安装 LM Studio 的问题**：用户 `@.mjune` 在运行 macOS 14.2 的 MacBook Pro 14 英寸上遇到了 LM Studio 的安装问题。`@yagilb` 建议更新 LM Studio 版本，`@heyitsyorkie` 进一步提到 Ubuntu 20.04 存在已知的 glibc 问题。
- **关于 LLM 和模型性能的关注**：用户就使用和理解不同模型、量化及其性能提出了问题。贡献者包括 `@ptable`、`@rugg0064`、`@heyitsyorkie`、`@kujila`、`@dagbs`、`@number_zero_` 等。
- **音乐生成模型**：`@american_pride` 和 `@fabguy` 简要讨论了音乐生成模型，提到了 Meta 的 AudioCraft 和 Google 的 MusicLM。

### ▷ #[🤝-help-each-other](https://discord.com/channels/1110598183144399058/1111440136287297637/) (96 messages🔥🔥): 
        
- **在 LM Studio 中自动启动服务器**：`@y3llw` 询问是否可以在 LM Studio 启动时自动开启服务器。`@heyitsyorkie` 回复称，目前在应用启动后仍需手动开启服务器。
- **LM Studio 加载模型的问题**：`@giopoly` 反馈了在 LM Studio 中无法加载任何模型的问题。`@thelefthandofurza` 建议检查 RAM 容量，尝试在开启和关闭 GPU Offload 的情况下分别加载模型，并考虑重新安装软件。
- **在 LM Studio 中更改缓存文件夹**：`@niborium` 遇到了与包含特殊字符的缓存文件夹名称相关的问题。`@heyitsyorkie` 建议将缓存文件夹路径更改为不包含特殊字符的目录。更改缓存路径的具体步骤得到了进一步明确：该选项位于聊天界面旁边的齿轮图标下。
- **LM Studio 中的 GPU Offloading**：`@xenorhon` 询问了在 LM Studio 中选择特定 GPU 进行 Offloading 的选项。`@doderlein` 指出，当前的 GPU 可以在“Preferences”的“System”选项卡中显示。
- **自定义模型的 Preset 设置**：`@pdg` 寻求关于各种模型最佳设置的建议。具体而言，他们询问了如何在模型 Preset 的 JSON 文件中调整 input prefix、suffix 和 antiprompt 等设置。用户 `@yagilb` 通过分享 OpenChat 模型的 JSON 模板文件提供了帮助。


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (27 messages🔥): 
        
- **聊天机器人性能评分**：`@itzjustsamu` 给该 Bot 的性能打了 `5/10` 分，称其“Token 计数低，且没有对我正常的回复做出响应”。
- **关于 MoE 模型的讨论**：`@kujila` 提出了一个关于 MoE 模型中专家数量选择的问题。`@fabguy` 建议阅读 Mistral 的论文和博客以进行深入了解。
- **模型转换与 LMStudio 的兼容性**：`@xkm` 询问将模型转换为 GGUF 格式是否能与 LMStudio 兼容，特别提到了[此链接](https://huggingface.co/shi-labs/vcoder_ds_llava-v1.5-13b/)中的模型。
- **寻求 Roleplay AI 推荐**：`@american_pride` 寻求 Roleplay AI 的推荐，并进一步解释了角色卡（character cards）。
- **LM Studio 中 ChatML Preset 的问题**：`@dagbs` 在 LM Studio 中创建自己的预设时遇到问题，并得到了 `@heyitsyorkie` 和 `@yagilb` 的帮助，后者认为 RoPE 数值可能是原因。尽管进行了多次故障排除尝试，问题仍然存在。


### ▷ #[🛠-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/) (9 messages🔥): 
        
- **LLM 的 GPU Offload**：用户 `@_kuva_` 询问了关于 LLM 使用 GPU Offload 的问题，以及在使用 RTX 4070 运行 30GB 模型时所需的层数。`@fabguy` 建议从 **10 层**开始，并逐渐增加，直到利用掉 90% 的专用 vRAM。 
- **检查 vRAM 利用率**：为了追踪 vRAM 使用情况，`@ptable` 和 `@fabguy` 建议 `@_kuva_` 使用 Windows **任务管理器中的性能选项卡**。`@fabguy` 进一步澄清，任务管理器中的“专用 GPU 内存”（_'Dedizierter GPU-Speicher'_）即代表 vRAM。 
- **vRAM 容量**：`@fabguy` 指出 `@_kuva_` 当前使用了 *3GB vRAM*，但最高可以使用 *11GB*。需要注意的是，这是在非 LM Studio 环境下提到的。


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (14 messages🔥): 
        
- **ChromaDB 插件问题**：用户 `@eimiieee` 在尝试运行 `@vic49` 的插件时遇到错误。该错误与 `pynvml` 中的 'NoneType' 对象有关。`@vic49` 确认了该问题，并请 `@eimiieee` 在 GitHub 上提交详细信息，但由于 `@eimiieee` 表示没有 GitHub 账号，双方同意通过 Discord 私信获取信息。
- **关于 ChromaDB 版本的建议**：`@heliosprime_3194` 根据自己运行旧版本 Chroma 且无问题的经验向 `@vic49` 提供了建议。他提议将完整的可用代码发布到 GitHub，并建议仅通过更新到 0.4.6 版本就可能解决插件的错误。
- **OpenAI Chat 引用**：`@vic49` 发布了一个 OpenAI 聊天链接（[点击此处](https://chat.openai.com/share/a4e03cdc-1c88-436f-ab61-522244630893)），未提供具体上下文。
- **关于编程爱好的评论**：`@vic49` 幽默地回应了 `@iamtherealbeef` 的评论，称他们原以为“编程会是一个既有趣又轻松的爱好！”

### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (64 条消息🔥🔥): 
        
- **不同模型的性能**：`@totallybored` 讨论了他们测试不同模型的经验，指出像 Hermes 2.5 这样的模型在添加特定的代码指令示例后表现更好，而 Mistral 在没有持续 pretraining 的情况下无法扩展到 8K 以上。
- **硬件与 AI 服务**：`@totallybored` 建议理想的情况是租用 AI server，这将提供必要的硬件和“私有”环境，然而，他们对廉价方案的隐私和安全性表示担忧。
- **Intel Iris Xe GPUs 的 GPU 使用**：`@kuzetsa` 询问是否有人成功让 Intel Iris Xe GPUs 正常工作，并详细介绍了他们在 Linux 上启用它的成功经验。
- **构建 LLM 推理系统**：在关于构建 LLM inference 系统的讨论中，`@heyitsyorkie` 建议优先考虑 GPUs，特别是 VRAM。其他用户（`@rugg0064` 和 `@pefortin`）讨论了更便宜的解决方案，并考虑将矿架（mining frames）和转接卡（risers）作为潜在的硬件配置。`@pefortin` 还提到他们将测试 risers 与增加 VRAM 相比的影响。
- **Poweredge 服务器与台式机的性能对比**：`@_nahfam_` 询问是否有人测试过 Poweredge server 与具有相同 RAM 和 VRAM 的台式机之间的性能差异。
- **RAM 频率**：`@dagbs` 质疑在为高端显卡购买更多 RAM 时频率的重要性。`@xenorhon` 针对 Ryzen 7000 的特定 FLCK、MCLK/UCLK、CL、RCD 和 RP、RAS 以及 RC 设置提供了建议，并警告不要尝试进一步压榨这些设置。


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (10 条消息🔥): 
        
- **自定义设置保存**：用户 `@jeunjetta` 对将自定义设置保存到 json 文件的功能表示赞赏，并询问是否有专门的地方可以提交细微建议。`@yagilb` 给予了积极回应，并将用户引导至频道 `#1128339362015346749`。
- **Mixtral 兼容性**：`@doderlein` 确认了 mixtral 在使用 **LM_Studio-0.2.10-beta-v2** 的 Linux 上的兼容性。
- **缓存文件大小问题**：`@pastatfiftylikesmoke` 对 `.session_cache/sessionid-xxx.gxxx` 文件过大表示担忧，并询问是否有更快的清理方法。`@yagilb` 提供了两种解决方案：右键点击聊天并选择清除 cache；或者通过点击 Chats 旁边的齿轮图标并关闭 cache 来彻底禁用缓存系统。
- **Vision LLM 咨询**：`@pastatfiftylikesmoke` 询问如何使用 vision LLM 模型。`@yagilb` 提供了一个 Discord 频道的链接，其中包含与 vision 模型相关的部分：[Vision 模型链接](https://discord.com/channels/1110598183144399058/1111797717639901324/1187839146094506096)。


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (4 条消息): 
        
- **在 Autogen 中保存 LLM 模型状态**：`@nemesisthedead` 询问了在 LM Studio 上使用 Autogen 进行 LLM 训练时保存模型状态的可能性。虽然没有提供明确的解决方案，但 `@nemesisthedead` 最终报告称已摸索出了该过程。


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- 关于 **Mistral-medium** 信息时间线的互动，Paperspace 提供的 H100 新方案，以及对 perplexity 在 AI 研究中实用性的反思。
- OpenAI 通讯中出现的语言设置问题，关于 GPT-5 当前状态的澄清，Turbo GPT 在编程方面的性能评估，OpenAI 缺乏更改电话号码的功能，以及更新 API 支付方式时 5 美元预授权费用的讨论。
- 围绕创建 AI 角色、ChatGPT 中插件和共享链接相关问题、微调后的 GPT-3.5 Turbo 的推理速度、使用 ChatGPT 3.5 编译和格式化报告、在 PC 端向 ChatGPT 上传图片、GPT 使用限制的刷新时间以及使用 OpenAI 进行文本摘要的若干查询。
- 介绍了 Pythagoras 聊天机器人，关于将 Trello REST API 与 GPT 集成的咨询和讨论，对 GPT 系列未来的推测，对 GPTs 中“Knowledge”功能的检查，一个 Chickasaw 语言助手项目，关于 ChatGPT 遇到问题的论述，关于使用 Discord Bot 生成图像的讨论，以及关于 GPTs 中指令、知识和动作工具限制的对话。
- 寻求在心理学和社会学中构建角色扮演 Prompt 的协助，利用 GPT 分析行为的建议，启动一个旨在利用 GPT 对通用语言进行学术和研究导向转化的项目，以及针对 ChatGPT 某个未说明问题的解决方案。
- 关于为心理学和社会学创建详细且具洞察力的 Prompt 的对话，利用 GPT 进行行为分析时与隐私指南的潜在冲突，Directive GPT 在 Prompt 优化方面的潜在用途，为研究目的推出的学术 Prompt 生成器 GPT，以及解决 ChatGPT 知识库问题的方案。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (3 条消息): 
        
- **模型知识时间线**：用户 `@eljajasoriginal` 询问了关于 **Mistral-medium** 的信息时间线，特别是想知道该模型的数据截止到哪一年。
- **Paperspace 与 H100s**：`@kyoei` 指出 Paperspace 开始提供 H100 租赁，暗示 2024 年 OpenAI 可能面临竞争。
- **关于 Perplexity 的评论**：`@caledordragontamer chief_executive` 分享了关于 perplexity 在研究中实用性的见解，称其在速度方面非常高效。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (81 条消息 🔥🔥): 
        
- **语言设置问题**：`@alextheemperorpenguin` 尽管账户设置为英语，但仍收到来自 OpenAI 的法语邮件。他们与 `@rjkmelb` 讨论了通过帮助页面的聊天机器人联系 OpenAI 以提交支持工单来解决此问题。
- **GPT-5 讨论**：`@mazejk` 最初以为 GPT-5 已经发布，但被 `@satanhashtag` 纠正，澄清那实际上是一个名为 "gpt-5" 的自定义聊天机器人。
- **GPT 编程性能**：`@pandora_box_open` 和 `@thunder9289` 讨论了 Turbo 版本的 GPT 是否值得为编程目的购买。共识是，在这方面它并不比 GPT-4 显著更好。
- **电话号码更改功能**：`@jamiecropley` 和 `@infidelis` 讨论了 OpenAI 不允许用户更新账户电话号码的问题。`@jamiecropley` 正面临此困扰，因为他们经常更换电话号码。 
- **OpenAI 费用**：`@jamiecropley` 询问了更新 API 支付方式时产生的 5 美元预授权费用。`@aminelg` 回复称从未遇到过此类费用。`@jamiecropley` 随后在 OpenAI 的帮助页面找到了更多相关信息：[此处](https://help.openai.com/en/articles/7438062-after-updating-my-api-payment-method-i-noticed-a-5-charge-is-this-a-temporary-hold)。

### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (51 条消息🔥): 
        
- **创建 AI 角色**：用户 `@load2` 询问如何创建一个 AI 角色。然而，目前没有提供具体的回复。
- **ChatGPT 中的插件**：`@rooster_25528` 在登录 ChatGPT 插件时遇到困难。用户 `@elektronisade` 告知插件系统在技术上即将停用，许多插件已被废弃或运行状况不佳。
- **在 ChatGPT App 中查看共享链接**：用户 `@beefcheong` 想知道如何在 ChatGPT App 中查看共享链接，`@lugui` 回复称目前没有办法做到这一点。
- **训练 GPT-3.5 Turbo 的速度与使用**：`@nyla3206` 询问了微调版 GPT-3.5 Turbo 的推理速度。然而，该查询没有得到回复。
- **在工作中使用 ChatGPT**：`@transatlantictex` 询问关于使用 ChatGPT 3.5 在工作中编写和格式化报告的问题。用户 `@yomo42` 提供了关于利用 GPT-3.5 处理 CSV 格式能力的反馈，以及使用 GPT-4 或 Custom GPT 的潜力。用户 `@yomo42` 还强调了考虑在工作场所使用 AI 的适当性和合法性的重要性。
- **向 ChatGPT 上传图片**：用户 `@euroeuroeuro` 在 PC 端向 ChatGPT 上传图片时遇到困难，而 Android App 似乎运行正常。消息中未提供解决方案。
- **GPT 的使用限制和刷新时间**：`@yomo42` 想了解 GPT 使用限制的刷新时间，`@satanhashtag` 澄清说每 4.5 分钟 1 条 Prompt，即 3 小时内 40 条 Prompt。
- **使用 OpenAI 进行文本摘要**：`@_s.n.o.o.p.y_` 询问哪种 OpenAI 模型最适合进行文本摘要。该查询未收到回复。
- **升级 ChatGPT Plus 账户以获得更多 Dall-e 使用额度**：`@.lb3373` 询问了在 ChatGPT Plus 账户上增加 Dall-e 使用量的升级选项。用户 `@aminelg` 澄清说目前的限制适用于所有人，若需更多使用量，必须使用 DALL-E API。
- **与 "worldbrains foundation" 的关联**：用户 `@zingzongy` 询问 OpenAI 是否与名为 "worldbrains foundation" 的非营利组织有任何关联。频道内未收到回复。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (23 条消息🔥): 
        
- **毕达哥拉斯聊天机器人**：`@arnoldtri` 建议创建一个毕达哥拉斯聊天机器人，并分享了链接：[Chatbot Pythagoras](https://chat.openai.com/g/g-gNH4K4Egg-shownotes)。
- **将 Trello REST API 集成到 GPT 中**：`@fyruz` 就尝试将 Trello REST API 集成到其 GPT 时遇到的困惑寻求帮助。
- **关于 GPT 未来的讨论**：`@phil4246` 发起了一场讨论，询问社区对 **GPT-4** 之后会发生什么的看法，例如 **GPT-4.5**、**GPT-5** 或其他产品。
- **GPT 中的 “Knowledge” 功能**：`@_odaenathus` 要求澄清 GPT 中的 “Knowledge” 功能是如何运作以及如何实现的。`@solbus` 澄清说，**知识文件就像是 GPT 的参考文档**，不会修改 GPT 的基础知识。
- **奇克索语（Chickasaw）助手**：`@machinemerge` 分享了他创建奇克索语助手的项目，并询问了有助于更好学习的具体 Prompt。`@Rock` 为他提供了改进 GPT 性能的方法建议。
- **ChatGPT 问题**：`@bonjovi6027` 报告了 ChatGPT 的一个问题，在 Google Chrome 和 Safari 上都收到了 **网络错误（network errors）**。`@scargia` 询问了对话长度。 
- **使用 Discord 机器人**：`@imaad22` 询问如何使用 Discord 机器人生成图像。`@elektronisade` 澄清说没有这样的功能。
- **指令、知识和操作工具的限制**：`@machinemerge` 和 `@Rock` 进一步讨论了 **20 个知识文件的限制，每个文件 512 MB**。指令（Instructions）工具有 **8000 个字符的限制**，`@loschess` 补充说，在任何给定时间可以激活 **10 个知识文件**，每个文件的 **限制为 2M tokens**。由于每个输出限制为 4k tokens，GPT 只能处理这么多内容。

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (8 messages🔥): 
        
- **心理学和社会学的角色扮演 Prompt**：用户 `@thebookoforlando777115` 寻求帮助，希望编写 Prompt 让 ChatGPT 听起来像心理学和社会学专家。`@bambooshoots` 建议使用一条强调深度分析、理论框架和核心概念的指令。他们还暗示可以使用 Directive GPT 进行进一步微调。
- **请求 ChatGPT 分析行为**：`@thebookoforlando777115` 询问如何将此类角色扮演 Prompt 用于分析某人的反应或真实性的场景。`@bambooshoots` 建议尝试现实生活中的问题，而 `@eskanta` 指出在分析他人行为时可能存在隐私指南方面的问题。
- **分析行为的替代方法**：`@eskanta` 建议要求模型从多个视角审视情况，并对最有见地的回答进行追问。
- **用于学术和研究导向任务的 GPT**：`@sacmaj` 分享了一个正在进行的项目，旨在将普通语言转化为用于研究目的的学术术语。他们提供了一个[链接](https://discord.com/channels/974519864045756446/1188346798314618980/1188346798314618980)来尝试用于学术任务的 Prompt 编写工具。
- **ChatGPT 问题的修复**：`.@shaw93` 提到解决了 ChatGPT 的一个问题，并在其 [Twitter 账号](https://twitter.com/ShawOnTech1)上分享了解决方案。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (8 messages🔥): 
        
- **优化心理学和社会学的 Prompt**：用户 `@thebookoforlando777115` 正在寻求关于为心理学和社会学创建详细且有见地的 Prompt 的建议。`@bambooshoots` 提供了一个详细的角色扮演 Prompt，旨在从心理学和社会学专家的角度生成深度分析。

- **现实场景分析**：`@thebookoforlando777115` 还表达了希望 AI 分析现实场景并判断一个人在特定情况下是否在撒谎的愿望。然而，`@eskcanta` 提醒此类请求可能与使用指南以及模型尊重隐私的训练要求相冲突。

- **Directive GPT**：`@bambooshoots` 建议，如果给定的 Prompt 没有产生理想的结果，Directive GPT 可能对进一步优化回复很有用。

- **用于研究导向任务的 GPT**：`@sacmaj` 分享了一个专为研究导向任务设计的学术 Prompt 生成器 GPT，它能将常规语言转化为类似于硕士论文水平的内容，并为创建 Prompt 提供结构化的阶段。他们还提供了一个[链接](https://discord.com/channels/974519864045756446/1188346798314618980/1188346798314618980)进行尝试。

- **知识库解决方案**：`.shaw93` 在其 Twitter `@ShawOnTech1` 上分享了一个未知问题的解决方案，描述其通过明确指示知识库“仅在 x 情况下使用此部分 (use this section only if x)”修复了该问题。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **讨论 AI 研究**：`@teknium` 分享了一篇关于 AI 研究的 [Twitter 帖子](https://fxtwitter.com/francis_yao_/status/1739688394148733264?s=46)；`@giftedgummybee` 认为它 **"非常棒。"**

- **协作机会、贡献价值、研究可能性及当前关注点**：讨论了 axolotl 项目的潜在任务、改进文档的重要性、LLM 研究中未达预期的部分，以及对无偿贡献不可持续性的担忧。
      
- **AI 进展与用户互动**：用户分享并讨论了一个关于 **RAG、RALM** 和 **Vector Stores** 的 [YouTube 视频](https://youtu.be/SozBO7eCvaM?feature=shared)。用户 `@asada.shinon` 将此进展识别为 **S-LoRA**。

- **Nous-Hermes 2 Yi 34B GGUF 量化版本发布**：`@teknium` 宣布发布了[新模型](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF)。

- **性能对比、代码污染、Reddit 用户体验、NeurIPS 2023 访问以及 LM Studio 兼容性**：讨论内容包括 **Hermes 2.5** 与 **Mixtral** 的性能对比、解决模型中的代码污染问题、分享 Reddit 用户的糟糕体验、探索获取 NeurIPS 2023 演讲内容的方法，以及确认 Nous Research 模型与 LM Studio 的兼容性。

- **本地 LLM 训练、模型选择、模型转换及数据集构建**：回应了关于 LLM 训练、合适模型选择的建议、在 **fp16** 下运行 Nous Hermes 2 - Yi-34B 的问题，以及关于构建和清洗数据集的咨询。

- **Transformers 与 Pipelines 使用、任务划分及迭代平均**：提出了关于使用 transformers 和 pipelines 示例的需求，以及关于任务划分和迭代平均的建议。

**Nous Research AI 频道摘要**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (2 条消息): 
        
- 关于**讨论的 AI 研究**话题，`@teknium` 分享了 [Francis Yao 的 Twitter 帖子](https://fxtwitter.com/francis_yao_/status/1739688394148733264?s=46)，`@giftedgummybee` 对此表示认可，称其 **"非常棒"**。


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (19 条消息🔥): 
        
- **潜在协作**：`@pradeep1148` 询问了涉及编写代码的任务，`@teknium` 指导其与 @525830737627185170、@257999024458563585 和 @208256080092856321 一起参与 axolotl 项目。
- **贡献价值**：在关于贡献的讨论中，`@hamelh` 强调即使是改进文档这样的小步骤也是有价值的，并鼓励 `@pradeep1148` 从那里开始。
- **研究机会**：当 `@erichallahan` 对以往 LLM 研究的经历表示失望时，`@hamelh` 建议仍然存在有价值的研究机会。 
- **重心转移**：尽管受到鼓励，`@erichallahan` 澄清说，除非有极具吸引力的提议，否则他们已将重心转移到其他重要的事业上。他们表达了对无偿工作不可持续性以及尽管有贡献但缺乏尊重的失望。
- **即将发布的论文**：另一方面，`@hamelh` 向 `@ldj` 透露，他们即将发布一篇关于用于构建 Capybara 数据集的 amplify-instruct 方法的论文。


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (8 条消息🔥): 
        
- **AI 突破讨论**：用户 `@lightningralf` 分享了一个介绍 AI 新进展的 [YouTube 视频](https://youtu.be/SozBO7eCvaM?feature=shared)，并将其描述为一项突破。该视频似乎讨论了 **Retrieval Augmented Generation (RAG)**、**Retrieval Augmented Language Models (RALM)** 和 **Vector Stores** 等高级概念。 
- **识别 AI 概念**：用户 `@asada.shinon` 将这一“突破”识别为 **S-LoRA**。
- **询问视频来源**：用户 `@rabiat` 询问 `@lightningralf` 是否是该分享视频的创作者。`@lightningralf` 回应称，他们不是 AI 研究员，但对 AI 话题很感兴趣。
- **用户互动**：用户 `@fullstack6209` 对分享的视频表示满意。
- **分享额外链接**：用户 `@metaldragon01` 分享了一个 [Twitter 帖子](https://fxtwitter.com/NickADobos/status/1739736441175900203)，但未提供上下文。从提供的聊天记录中无法清楚了解该推文的内容和相关性。

### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (1 条消息): 
        
- **Nous Hermes 2 Yi 34B GGUF 量化版本可用性**：在 `@teknium` 发布的一项公告中，宣布 Nous-Hermes 2 Yi 34B 已经完成了 GGUF 量化，并提供了 TheBloke 通常会提供的各种典型尺寸。该模型已在 **HuggingFace** 组织的此[链接](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF)上线。同时还分享了一张用于宣传发布的参考图片。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (74 条消息🔥🔥): 
        
- **Hermes 2.5 与 Mixtral 性能对比**：`@teknium` 和其他用户就 **Hermes 2.5** 和 **Mixtral** 的性能对比展开了讨论。用户 `@fullstack6209` 分享了他的使用体验，强调了 Mixtral 在某些 Benchmark 测试中的卓越表现。双方都注意到这两个模型均出自 Nous Research 团队，用户 `@vic49.` 提到了最新发布的 **Nous Hermes 2 - Yi-34B - GGUF 量化版本**。与此对话相关的链接：[Nous Hermes 2 - Yi-34B - GGUF Quantized Version](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF) 已被发布。

- **模型中的代码污染问题**：关于 AI 模型中代码污染（code contamination）问题的讨论非常热烈。用户 `@.beowulfbr` 和 `@nonameusr` 主导了讨论，提到了模型 Benchmark 的难度以及像 **yi-34b** 这样被“污染”的基座模型所存在的问题。

- **与 Reddit 用户的交流体验**：`@.beowulfbr` 与 `@weyaxi`、`@ldj` 及 `.benxh` 等其他用户之间的聊天暗示了对 Reddit 用户社区的不满，并引用了他们经历过的一些负面体验。

- **NeurIPS 2023**：用户 `@lejooon` 询问了在不现场出席的情况下观看 NeurIPS 2023 演讲的方法。`@itali4no` 和 `@gabriel_syme` 的回复建议需要购买在线门票，或者可以等待几周后可能发布的免费访问资源。

- **LM Studio 兼容性**：用户 `@vic49.` 询问了 Nous Research 生产的模型与 LM Studio 的兼容性。特别提到了 **Nous Hermes 2 - Yi-34B - GGUF 量化版本**。`@teknium` 确认它们确实是兼容的。


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (21 条消息🔥): 
        
- **LLM 中 <0x0A> 的用法**：`@teknium` 指出 `<0x0A>` 在模型脚本中被用来表示换行符。这一点得到了 `@everyoneisgross` 的进一步验证，尽管他们也注意到其 Mixtral 模型存在一个复杂问题，即该标签似乎被转换成了字符串并改变了文本。这表明脚本中可能存在 Bug。
- **本地 LLM 训练的模型选择**：`@leuyann` 寻求关于适合本地和非本地微调及测试的模型建议，重点参考了大型语言模型（LLM）中的推理增强。`@eas2535` 澄清说，虽然 Mixtral 的 4-bit 量化版本不需要 CUDA，但它确实需要大量的 RAM，这使得它不适合 16GB 内存的系统。`@night_w0lf` 建议在尝试 7B Mistral 等更大型号的模型之前，先从 StableLM 等 3B 基座模型开始。
- **将模型转换为 fp16**：`@mgoin` 提出了在 fp16 精度下运行 Nous Hermes 2 - [Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) 的问题，指出该模型在该精度下运行时会产生 NaNs。他们注意到 PyTorch 的 `fake_quantize` 缺乏对 bfloat16 的支持。`@teknium` 建议只需将 `torch.dtype` 更改为 `float16` 即可。
- **数据清洗与数据集构建**：`@.beowulfbr` 寻求关于构建和清洗数据集的建议。`@teknium` 推荐使用 Lilac，这是一个由 @tomic_ivan 开发的开源数据改进工具。`@spencerbot15` 提供了相应的 [Lilac 文档链接](https://docs.lilacml.com/)。


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 条消息): 
        
- **Transformers 和 Pipelines 的用法**：`@vic49` 向 `@qnguyen3` 索要关于如何使用 Transformers、Pipelines 以及其他类似工具的示例。
- **任务拆分与迭代平均**：`@tranhoangnguyen03` 向 `@.benxh` 提议，将计数任务进行拆分，并取多次迭代的平均值。

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- 关于如何处理 **HuggingFace datasets** 中的缓存文件的讨论非常活跃，正如 `@vipitis` 所建议的，包括删除 `~\.cache\huggingface\datasets` 中的所有内容，并将 datasets 升级到 2.16.0 版本。
- 关于**模型微调 (model fine-tuning)** 的广泛讨论，涵盖了在 Windows 上的应用、使用 FSDP 对 Whisper 模型进行微调，以及理解 OpenAI embeddings 的工作原理。用户 `@deadsg` 为 GPU 容量有限的用户分享了一个有用的 [Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing)。
- 社区参与的潜规则成为 `@liyucheng09`、`@gez_gin`、`@vipitis` 和 `@mr.kiter` 等用户讨论的话题，涉及响应时间以及社区反馈循环中面临的挑战。
- 值得关注的开发和测试项目包括：`@tonic_1` 开发的将图像转录为 latex 公式的 **Texify**，以及 `@jiha` 分享的 **Dolphin 2.6** 在线测试平台 [链接](https://replicate.com/kcaverly/dolphin-2.6-mixtral-8x7b-gguf)。`@Deadsg` 在 [GitHub](https://github.com/Deadsg/Bat-.-LLama-.-CPP) 上分享了他们的 **Bat-.-LLama-.-CPP 项目** 及其相关的 [Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing)。
- 关于**检索增强生成 (RAG) 与微调 (Fine-tuning)** 在模型知识传授方面的咨询，`@harsh_xx_tec_87517` 主张使用 RAG 来对抗微调中的幻觉问题。`@harsh_xx_tec_87517` 还分享了一个关于如何更新 LLM 信息的 [教程](https://www.youtube.com/watch?v=74NSDMvYZ9Y&t=193s)。
- 用户分享的教育内容包括：`@onceabeginner` 对 **mamba architecture** 的探索，以及 `@merve3234` 对 **OneFormer** 的学习之旅（将其描述为强大的分割模型并分享了相关 [笔记](https://x.com/mervenoyann/status/1739707076501221608?s=20)）。
- 工具更新和社区新增内容：`@appstormer_25583` 透露了对 [Appstorm.ai](https://beta.appstorm.ai/) 的增强，包括 GPT 功能、LiteLLM 集成和错误修复。`@gz888` 在 cool-finds 频道分享了一个 [工具](https://huggingface.co/thibaud/sdxl_dpo_turbo)。
- 对语言模型的探索：`@opencuiguy` 建议使用 **NLI 模型进行蕴含 (entailment)** 推断，并指出 **seq2seq 模型容易产生幻觉**，而 `@lucastononro` 在其 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7139769316337504256-dDoT?utm_source=share&utm_medium=member_desktop) 中将 **Large Language Models (LLMs)** 与 “Interface Omniscience” 结合用于网页交互，并分享了相应的 [GitHub 代码](https://github.com/lucastononro/llm-food-delivery)。

**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (90 条消息🔥🔥): 
        
- **HF Datasets 缓存清理**：用户 `@liyucheng09` 询问如何清理 HuggingFace datasets 的缓存文件。他们在 HF Hub 上更新了一个数据集，但在本地加载时看不到更新。`@vipitis` 建议删除 `~\.cache\huggingface\datasets` 中的所有内容，并将 datasets 升级到 2.16.0 版本，据报道该版本包含了一些针对缓存内容的修复。

- **模型微调讨论**：不同用户之间就模型微调进行了讨论。`@omarabb315` 询问 Whisper 是否可以使用 FSDP 进行微调，因为 DeepSpeed 在 Windows 上无法运行。`@deadsg` 为那些没有足够 GPU 容量进行微调的用户分享了一个[有用的 Google Colab](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing)。

- **社区响应与礼仪**：包括 `@liyucheng09`、`@gez_gin`、`@vipitis` 和 `@mr.kiter` 在内的几位用户参与了关于社区参与度的讨论。观点涉及获得帮助或反馈所需的时间、假期对响应时间的影响，以及提供答案的人所面临的挑战。

- **Embedding 向量问题**：用户 `@somefuckingweeb` 询问为什么 OpenAI embeddings 是单个向量，以及在计算余弦相似度时是否在维度上使用了求和或均值。`@_aabidk` 将此问题转发给了另外两位用户 `@251101219542532097` 和 `@697163495170375891`。

- **在线测试 Dolphin 2.6**：`@jiha` 询问如何在显存不足以运行 Mixtral 8x7b 的情况下在线测试 Dolphin 2.6。他们还分享了一个 Dolphin 2.6 Mixtral 的免费在线测试平台 [链接](https://replicate.com/kcaverly/dolphin-2.6-mixtral-8x7b-gguf)。

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 条消息): 
        
- **Mamba Architecture**: `@onceabeginner` 分享了他们学习 **Mamba Architecture** 的心得。不过，没有提供更多关于该主题的详细细节。
- **OneFormer - 通用分割模型**: `@merve3234` 分享了关于 **OneFormer** 的学习经验。他们称其为“非常强大的通用分割模型”，并分享了关于该主题笔记的 [链接](https://x.com/mervenoyann/status/1739707076501221608?s=20)。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (1 条消息): 
        
gz888: https://huggingface.co/thibaud/sdxl_dpo_turbo


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 条消息): 
        
- **Texify 项目**: `@tonic_1` 发布了他们构建的一个 [Demo](https://huggingface.co/spaces/Tonic1/texify/settings)，该项目可以拍摄照片并返回 LaTeX 公式。目前该项目已获得社区资助，供他人在此基础上进行开发。
- **Bat-.-LLama-.-CPP 项目**: `@deadsg` 分享了他们项目的 [GitHub 仓库](https://github.com/Deadsg/Bat-.-LLama-.-CPP)。
- **Bat-.-LLama-.-CPP 项目的 Google Colab**: `@deadsg` 还为那些缺乏 GPU 来微调其 Bat-.-LLama-.-CPP 项目的用户提供了一个 [Google Colab 链接](https://colab.research.google.com/drive/1GzzhruUz6r-qYcvejjktb21STSH_EBYT?usp=sharing)。他们建议设置路径并将仓库 git clone 到 Colab 中。
- **Appstorm.ai 更新**: `@appstormer_25583` 分享了 [Appstorm.ai](https://beta.appstorm.ai/) 最近的补丁更新，包括 GPT 能力增强、Bug 修复、LiteLLM 集成等。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 条消息🔥): 
        
- **检索增强生成 vs 微调**: 用户 `@harsh_xx_tec_87517` 指出，向特定模型传授知识的最佳方式是使用 **Retrieved Augmented Generation (RAG)** 而不是微调，因为后者往往会导致幻觉。用户 `@bennoo_` 表达了对使用小型数据集进行微调以更新语言模型知识的兴趣，并询问 **RAG 系统及其基础设施** 是否必要。 
- **LLM 微调教程**: `@harsh_xx_tec_87517` 还分享了一个关于如何 **更新 LLM 信息** 的 YouTube [教程](https://www.youtube.com/watch?v=74NSDMvYZ9Y&t=193s)。
- **NLI 模型的使用**: 用户 `@opencuiguy` 针对 `@merve3234` 的问题建议使用 **NLI 模型进行蕴含推理 (entailment)**。他提到这些模型是 Encoder-only 的，可以在 HuggingFace 网站的 text-classification 栏目下找到。
- **项目 - 具有“界面全知”能力的 LLM**: `@lucastononro` 分享了他的项目，将 **Large Language Models (LLMs)** 与“界面全知 (Interface Omniscience)”相结合，以简化订餐等网页交互。他在聊天机器人中利用 RAG 进行智能信息检索。分享了 [LinkedIn 帖子](https://www.linkedin.com/posts/activity-7139769316337504256-dDoT?utm_source=share&utm_medium=member_desktop) 和 [GitHub 代码](https://github.com/lucastononro/llm-food-delivery)。
- **关于 seq2seq 模型的讨论**: 在与 `@merve3234` 的讨论中，`@opencuiguy` 提到 **seq2seq 模型容易产生幻觉**，并建议微调 Encoder-only 模型，以便在更少的计算和内存需求下获得更好的性能。


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **LookAhead 解码策略**：用户 `@rage31415` 提到，LookAhead 可以加速 AI 应用中语言模型 (LLM) 的推理。
- **Mistral API Waitlist 与定价**：讨论了 Mistral API 的等待时间和成本，用户 `@lee0099` 分享了最近从 waitlist 获批的经历，并提到小型模型每 100 万个 token 的成本低于 0.50 美元。`@sublimatorniq` 强调了 Mistral 和 OpenAI 模型在 token 计数上的差异。
- **文本转语音与文本转音乐模型**：用户 `@qwerty_qwer` 询问了在拥有丰富数据集（包括来自 Spotify 的 100 万首歌和约 20,000 小时带字幕的音频文件）的情况下训练此类模型的问题。
- **模型的真实性与争辩性响应**：`@.skyair` 指出了 Mistral 7B v0.2 和 Mixtral v0.1 模型在准确性和一致性方面的问题，并将其与 Claude 和 GPT-3.5 模型进行了对比。
- **Mistral 与 Mixtral 发布源代码可用性**：用户 `@pilot282` 询问了这些模型的发布状态和源代码可用性，`@lee0099` 澄清说 Mixtral small 的源代码已发布，但 Mixtral medium 尚未发布。
- **MoE LLM 的开源选项**：`@brycego` 询问了训练 MoE LLM 的开源解决方案，`@poltronsuperstar` 推荐了 [Mixtral + vast.ai](https://github.com/mistralai/mistral-src/blob/main/mistral/moe.py)。
- **模型响应的一致性**：`@mka79` 提出了对 Mixtral 8x7b 响应不一致的担忧，并得到了调整 temperature、top_p 和固定 random seed 等参数的建议。
- **聊天机器人模型对比**：讨论了不同聊天机器人模型的一致性，提出了关于 Google Gemini 和 Huggingface Chat 模型的看法。
- **速度优化**：关于如何提高 mistral instruct v02 7B q4_k_s 模型速度的咨询，`@steroidunicorn` 建议增加 VRAM。
- **微调 Mistral 模型的优化**：`@builderx` 征求了关于优化部署在 A100 上的微调 Mistral 模型的建议。
- **构建 Mistral 镜像问题**：`@cheemszero69420` 在使用 Docker 构建 Mistral 镜像过程中，遇到了 megablocks 和 stanform-stk 库安装时间过长的问题。
- **Few Shot 提示词模板**：`@nioned` 询问是否存在 few shot 提示词模板。
- **请求速率限制**：`@lerela` 宣布为了提高服务质量，引入了每秒请求数 (rate-limits) 的限制。
- **Mistral Medium 公开发布**：用户 `.superintendent` 确认 Mistral Medium 的公开发布已在计划中，但具体发布日期尚未确定。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (25 条消息🔥): 
        
- **语言模型的推理速度**：`@rage31415` 发起了关于使用 **LookAhead 解码策略** 来加速语言模型 (LLM) 推理的话题。
   
- **Mistral API Waitlist 与成本**：围绕 **Mistral API** 的 waitlist 和成本进行了讨论。用户 `@mka79` 询问了脱离 API waitlist 的典型等待时间。`@lee0099` 提到他们最近从 waitlist 获批，但指出了负担能力的挑战。`@lee0099` 随后分享说，小型模型每 100 万个 token 的成本低于 0.50 美元——这一细节得到了 `@sublimatorniq` 的证实，但他指出 Mistral 和 OpenAI 模型的 token 计数有所不同，估计大约 4 个 OpenAI token 相当于 5 个 Mistral token。 

- **训练文本转语音或文本转音乐模型**：用户 `@qwerty_qwer` 询问是否有人在训练文本转语音或文本转音乐模型，并表示他们拥有丰富的数据源，包括来自 Spotify 的 100 万首歌和约 20,000 小时带字幕的音频文件。

- **模型的真实性与争辩倾向**：`@.skyair` 对 **Mistral 7B v0.2 和 Mixtral v0.1** 模型表示担忧，认为这些模型更倾向于争辩，经常使用“说……是不准确的”这类反驳语句。该用户还指出模型在真实性方面存在问题，证明模型经常在生成错误信息的同时宣称事实是不准确的。他们将其与 Claude 和 GPT-3.5 进行了比较，认为后者不存在这些问题。

- **Mistral 与 Mixtral 的发布与源代码**：`@pilot282` 询问了 **Mistral 和 Mixtral** 的发布状态和源代码可用性。作为回应，`@lee0099` 澄清说 Mixtral small 的源代码已经发布，但 Mixtral medium 尚未发布。

### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (16 messages🔥): 
        
- **训练 MoE LLM 的开源解决方案**：用户 `@brycego` 询问了关于训练 MoE LLM 的开源解决方案，`@poltronsuperstar` 向其推荐了 **[Mixtral + vast.ai](https://github.com/mistralai/mistral-src/blob/main/mistral/moe.py)**，并进一步强调了对 PyTorch 专业知识的需求。
- **模型响应的不一致性**：`@mka79` 提出了关于 **Mixtral 8x7b** 对相同 Prompt 给出不同答案的问题。`@sublimatorniq` 和 `@lerela` 建议调整 **temperature**、**top_p** 等参数并固定 **random seed** 以获得更高的一致性。
- **Chatbot 模型对比**：对比了不同 Chatbot 模型的一致性。`@sublimatorniq` 发现 Google 的 Gemini 相当一致，而 `@mka79` 报告称 **Huggingface Chat** 的效果更好。
- **mistral instruct v02 7B q4_k_s 模型的速度优化**：用户 `@serhii_kondratiuk` 询问了在配备 16GB VRAM 和 32 GB RAM 的 GeForce 3080 ti RTX 上，针对 1k token 输入运行 mistral instruct v02 7B q4_k_s 模型时，如何提高响应速度。`@steroidunicorn` 建议将 VRAM 扩展到 24GB 将显著提升处理速度。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **优化微调后的 Mistral 模型**：`@builderx` 询问了如何优化微调后的 **Mistral** 模型，因为该模型在 **A100** 上运行时性能较慢。

- **构建 Mistral 镜像的问题**：`@cheemszero69420` 在使用 Docker 构建 **Mistral** 镜像时遇到问题，过程卡在安装 **megablocks** 和 **stanform-stk** 库的步骤。该过程耗时远超预期，因此寻求解决方案。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 messages): 
        
- **Few Shot Prompting 模板**：用户 `@nioned` 询问是否有可以遵循的 **few shot prompting 模板**。
- **Rate-Limits 更新**：`@lerela` 通知称，他们引入了 **requests per second rate-limits**（每秒请求数限制）以提高平台的服务质量。用户可以在平台上查看这些限制，如果受到此更改影响，可以联系 support@mistral.ai。
- **Mistral Medium 公开发布**：用户 `.superintendent` 确认 **Mistral Medium** 计划公开发布，但尚未设定具体日期。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- 由 `@noobmaster29` 和 `@nruaif` 领导的关于集成 **new tokenizer** 进行 **model training** 及其与 qlora 潜在兼容性的讨论。会议指出，使用某些 token 进行训练可能是可行的，但整个语言适配可能无法产生足够的效果。 
- 关于 **shards 大小和 GPU 状态** 的技术问题，由 `@casper_ai` 提出，并澄清分别为 5 GB 和 10 GB。有人建议空间有限可能会导致问题。
- 关于 `MistralForCausalLM` 术语与 `AttributeError` 相关的澄清，`@caseus_` 和 `@faldore` 确认 `Mistral-7b` 是正确的术语。
- `@colejhunter` 和 `@caseus_` 分享了关于 **Alpaca** 格式的见解，并确认 alpaca 遵循 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L54) 中的特定结构。
- `@unknowngod` 询问了当数据集针对单个输入提供多个有效响应时，如何选择 **validation set**（验证集）的方法。
- 由 `@xzuyn` 发起的 **测试集去重和去污染 (deduplicating and decontaminating)** 策略。
- `@faldore` 尝试在 tokenizer 中 **替换一个 token**。
- `@fred.bliss` 询问了用于为 RAG 生成 **Q&A pairs** 的工具，他之前曾为此目的使用过 LlamaIndex。他对其他用户使用 LlamaIndex 和 AutoGen 生成最新 RAG 数据集的经验表示感兴趣。讨论围绕使用本地模型实现特定领域识别展开，并强调高质量并非主要目标。

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (4 条消息): 
        
- **训练和集成新的 Tokenizer**：用户 `@noobmaster29` 寻求关于**训练模型以适配新 Tokenizer** 及其与 QLoRA 集成可能性的建议。用户 `@nruaif` 澄清说，训练少量 Token 可能是可行的，但整个语言的适配将不足够。
- **使用新 Tokenizer 进行微调**：用户 `@noobmaster29` 进一步询问对于新 Tokenizer 是否最好在文本补全（text completion）上进行微调，`@nruaif` 回复说任何格式都是可以接受的。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (3 条消息): 
        
- **关于分片（Shards）和 GPU 状态可用空间的担忧**：`@casper_ai` 向 `@faldore` 澄清说分片是 5 GB，GPU 状态是 10 GB，并提出空间不足是否可能是问题所在。
- **关于 AttributeError: MistraForCausalLM 的澄清**：`@caseus_` 询问 `@faldore` 关于 `AttributeError: MistralForCausalLM` 报错中，正确的术语应该是 `Mistral` 还是 `Mixtral`。
- **确认使用 Mistral-7b**：当被 `@caseus_` 询问时，`@faldore` 确认正确的术语是 `Mistral-7b`。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (18 条消息🔥): 
        
- **Alpaca 格式化**：用户 `@colejhunter` 询问 Alpaca 是否遵循标准的 Alpaca 格式，使用的是在 [GitHub 链接](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L54) 中看到的结构。针对该询问，`@caseus_` 确认他们的实现在 system prompt 方面更加明确。`@colejhunter` 进一步澄清，当使用 `type: alpaca` 训练后进行推理时，Prompt 的结构应为 `### System:\n{system}\n\n### Instruction:\n{instruction}\n\n### Response:\n`。

- **选择验证集**：用户 `@unknowngod` 寻求建议，当数据集针对给定输入包含多个有效响应时，选择验证集的最佳方法是什么。

- **去重和测试集去污染**：用户 `@xzuyn` 发起讨论，询问人们通常使用什么策略进行去重和测试集去污染。

- **在 Tokenizer 中替换 Token**：用户 `@faldore` 尝试将 Tokenizer 中的一个 Token (`</s>`) 替换为一个新 Token。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (4 条消息): 
        
- **咨询问答对生成工具**：在休假开始时，`@fred.bliss` 询问目前用于为 RAG 生成问答对（Q&A pairs）的工具。
- **用于特定领域的本地模型**：`@fred.bliss` 澄清了他打算使用本地模型来生成这些问答对，强调质量不是首要考虑因素，目标是实现特定领域的识别。
- **使用 LlamaIndex 进行 RAG 数据集生成**：`@fred.bliss` 分享了他过去使用 LlamaIndex 作为此特定任务变通方案的经验，并指出它现在对于该用例已完全可用，但针对 OAI 进行了深度优化。
- **寻求 LlamaIndex 和 AutoGen 的使用经验**：最后，`@fred.bliss` 询问是否有人尝试过 LlamaIndex 中最新的 RAG 数据集生成器和 AutoGen 工具。他提到了之前来自 `<@284810978552578050>` 提到 AutoGen 的对话。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- `@hwchase17` 发布公告，根据 `<@1059144367731920896>` 的建议，**创建了一个新频道** `<#1189281217565179915>`。
- 关于 **LangChain** 的使用和挑战进行了广泛讨论，具体查询包括除了阅读 PDF 和查询结果之外的可能性。要点包括从 Tool 的 **LLM** 输出中获取额外的键值数据，**LangChain**、**LCEL** 与带有工具的 **Agent** 之间的比较，以及导入 numpy 时的 ImportError。
- `@repha0709` 提出了创建一个专注于 autogen、**LangChain** 等领域的**专家机器人 (Specialist Bot)** 的想法，并分享了在使用 text-generation-webui 运行模型时，**LangChain** 与 "openai" API 之间的冲突。
- `@evolutionstepper` 询问了社区关于处理 **async** 和 **FastAPI** 的实践。
- `@cryptossssun` 和 `@madgic_` 分享了关于 **GPT Vision** 在准确提取 PDF 复杂表格数据方面的局限性。`@madgic_` 分享了一个特定的 Prompt，以缓解从图像到 **Markdown** 转换时的转录问题。
- `@a404.eth` 分享了他在 **LangChain Playground** 上的探索以及如何在其中使用输入，实现了“对话式检索链 (Conversational Retrieval Chain)”，并分享了用于创建检索链的 **LangChain** 文档链接。此外，还提到了 **LangChain** 相关的显著 Python 库。
- `@appstormer_25583` 宣布了 [Appstorm.ai](https://beta.appstorm.ai/) 的更新。更新包括实现了 **Gradio** 的 folium 组件，增加了 GPTs 示例，增强了 GPTs 的鲁棒性，修复了多个 Bug，支持 **LiteLLM** 集成，改进了 Vision GPTs，以及 GPTs 执行 **Google Search** 的能力。

**LangChain AI 频道总结**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 条消息): 
        
- **创建新频道**：用户 `@hwchase17` 宣布根据 `<@1059144367731920896>` 的建议创建了新频道 `<#1189281217565179915>`。创建该频道是因为确保系统可靠运行需要大量工作。


### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (15 条消息🔥): 
        
- **LangChain 用法**：用户 `@infinityexists.` 询问是否有资源可以探索 **LangChain** 在阅读 PDF 和查询模型结果之外的使用可能性。
- **LLM 输出中的额外键值数据**：`@peeranat_fup` 询问是否可以从 Tool 的 **LLM** 输出中传递额外的键值数据。
- **numpy 的 ImportError**：用户 `@arunraj6451` 分享了一个关于从源目录而非 Python 解释器导入 numpy 时出现 ImportError 的问题。
- **关于 LCEL 与带工具 Agent 的讨论**：`@a404.eth` 表示有兴趣讨论 **LCEL** 与**带工具的 Agent** 之间的比较。
- **创建专家机器人**：`@repha0709` 讨论了创建一个专门研究 autogen、**LangChain** 等的机器人的想法，该机器人将根据用户的 Prompt 生成 autogen 配置。他们提到在使用 text-generation-webui 运行模型时，遇到了 **LangChain** 与 "openai" API 之间的冲突。
- **处理 Async 和 FastAPI**：`@evolutionstepper` 询问了社区处理 **async** 和 **FastAPI** 的方法。
- **GPT Vision 对 PDF 的局限性**：`@cryptossssun` 和 `@madgic_` 讨论了 **GPT Vision** 在提取 PDF 文件中复杂表格数据时的困难，因为它经常误解或忽略空白单元格。为了尝试解决这个问题，`@madgic_` 分享了一个 Prompt 以改进从图像到 **Markdown** 格式的转录。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (5 条消息): 
        
- **在 LangChain Playground 中使用输入**：`@a404.eth` 询问了如何在 **LangChain** playground 中使用输入，并提供了一个如何使用 **LangChain** 的 `ChatPromptTemplate` 运行简单模板的示例。
- **对话式检索链**：`@a404.eth` 提到他/她正在参考指南实现一个带有检索增强生成的“对话式检索链 (Conversational Retrieval Chain)”，并分享了代码片段。
- **分享 LangChain 指南链接**：`@a404.eth` 分享了 [LangChain 文档](https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain)中关于创建检索链的指南链接。
- **LangChain 的重要 Python 库**：**LangChain** 指南建议安装某些 Python 库，包括 **LangChain**、**OpenAI**、**faiss-cpu** 和 **tiktoken**。

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (1 条消息): 
        
- **Appstorm.ai 更新**: 用户 `@appstormer_25583` 宣布了 [Appstorm](https://beta.appstorm.ai/) 的几个补丁更新：
    - GPTs 可以使用 Gradio 的 folium 组件来渲染详细地图。
    - GPTs 现在有了示例。
    - GPTs 表现出更高的鲁棒性。
    - 已实施多项 Bug 修复。
    - LiteLLM 集成现已可用。
    - 针对 Vision GPTs 引入了修复。
    - GPTs 现在可以执行 Google Search。


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- 用户 `@harsh1729` 询问 Hugging Face 上最好的 function-calling 模型，理想情况下是 **Llama-7B** 的变体。
- 用户 `.@.benxh` 对 xDan 的 7B 模型令人印象深刻的表现表示认可。
- 关于 LocalLLaMa 版块引发争议的讨论，涉及对 AI 模型创建者欺诈活动的指控，`.@beowulfbr` 分享了相关参考资料 ([thread1](https://www.reddit.com/r/LocalLLaMA/comments/18qp3fh/this_is_getting_ridiculous_can_we_please_ban/), [thread2](https://www.reddit.com/r/LocalLLaMA/comments/18ql8dx/merry_christmas_the_first_opensource/))。
- 提到用户 `.@beowulfbr` 的模型 CodeNinja 在该版块最初获得积极反响后遭到的抵制 ([thread3](https://www.reddit.com/r/LocalLLaMA/comments/18pr65c/announcing_codeninja_a_new_open_source_model_good/))。
- `@ty.x.202` 分享了从环境搭建到实际项目的 Java 精通全面指南。
- 讨论在基础模型上堆叠多个 adapter 而非使用 RAG 方法，并附带了相关的 GitHub 项目链接：`@lightningralf` 分享的 [S-LoRA](https://github.com/S-LoRA/S-LoRA)。
- 宣布 OpenChat 模型的新版本及其 HuggingFace 链接和在线演示，尽管其评分与之前版本相似。服务器访问、模型视图和在线演示的相应链接：[server](https://discord.gg/ySZJHsPt), [model view](https://huggingface.co/openchat/openchat-3.5-1210), [Online demo](https://openchat.team)。
- `@joshxt` 询问用户是否发现使用或微调特定 Context 的价值，尽管未指明具体 Context。

**Alignment Lab AI 频道摘要**

### ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/) (1 条消息): 
        
- **最佳 Function-Calling 模型咨询**: 用户 `@harsh1729` 征求关于 Hugging Face 上可用的最佳 function-calling 模型的建议，最好是 **Llama-7B** 的衍生模型。


### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (10 条消息🔥): 
        
- **xDan 模型好评**: `.@.benxh` 赞扬了 xDan 的 7B 模型令人印象深刻的输出。
- **Reddit 上的争议**: `.@beowulfbr` 指出了 LocalLLaMa 版块中针对 AI 模型的负面关注。攻击者正对创建和合并模型的人发起欺诈活动的指控。分享了证据线程的链接 ([thread1](https://www.reddit.com/r/LocalLLaMA/comments/18qp3fh/this_is_getting_ridiculous_can_we_please_ban/), [thread2](https://www.reddit.com/r/LocalLLaMA/comments/18ql8dx/merry_christmas_the_first_opensource/))。
- **CodeNinja 模型成为目标**: `.@beowulfbr` 还提到在同一版块发布了他们自己的模型 CodeNinja。尽管最初收到了积极反馈，但随后遭到了同一群攻击者的针对 ([thread3](https://www.reddit.com/r/LocalLLaMA/comments/18pr65c/announcing_codeninja_a_new_open_source_model_good/))。
- **Java 精通指南**: `@ty.x.202` 提供了一份精通 Java 的全面指南，包括设置 Java 环境、掌握 Java 基础、探索面向对象编程原则、理解 Java API、异常处理、JDBC 和数据库连接、使用 JavaFX 进行 GUI 开发、高级 Java 主题，以及最后的实际项目。


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 条消息): 
        
- **在基础模型上堆叠多个 Adapter**: `@lightningralf` 讨论了在基础模型上堆叠多个 adapter 并每周/每月进行切换的方法，而不是使用 RAG 方法。他分享了一个名为 [S-LoRA 的 GitHub 项目](https://github.com/S-LoRA/S-LoRA)链接，强调其能够以快速方式服务数千个并发 LoRA Adapter 的能力。

### ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/) (1 条消息): 
        
- **OpenChat 公告**：发布了关于新版本 OpenChat 模型的公告。可以通过[这里](https://discord.gg/ySZJHsPt)访问服务器。在性能方面，它的评分与之前的版本一样高。
- **HuggingFace 链接**：可以在[这里](https://huggingface.co/openchat/openchat-3.5-1210)查看 OpenChat 的新模型。这是使用混合质量数据在开源语言模型方面取得的一项进展。
- **在线 Demo**：可以在 [Online Demo](https://openchat.team) 中测试 OpenChat 模型。


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (1 条消息): 
        
- **微调价值咨询**：`@joshxt` 询问是否有用户发现使用或微调特定上下文的价值，尽管现有信息中未指明具体上下文。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- 关于 **LLM-as-a-Judge 评估偏见**的激烈讨论，强调了诸如利用 *GPT-4 对长输出的偏好* 等担忧。可能的缓解策略包括使用参考答案来评估问题。
- 检查了 **Disco Judge 在不同评估背景下的应用**。注意到了背景差异，在合成数据过滤或质量保证等实例中，*参考答案并不总是可用*。
- 关于 **LLM-as-a-Judge 方法论**局限性的对话。提出的问题包括固有偏见或范围狭窄，这取决于 LLM 是确定答案质量还是与 LLM 生成的参考答案进行比较。
- 结论是需要继续讨论并**量化这些局限性**在 LLM-as-a-Judge 评估方法中的影响。
- 展示了**各种 7B 模型的基准测试结果**。讨论的模型包括 *xDAN-AI/xDAN-L1-Chat-RL-v1*、*openchat/openchat-3.5-1210*、*Weyaxi/SauerkrautLM-UNA-SOLAR-Instruct* 和 *upstage/SOLAR-10.7B-Instruct-v1.0*。提到的基准测试工具和源代码为 [EQbench](https://www.eqbench.com/)，以及随附的[源代码](https://github.com/EQ-bench/EQ-Bench)和[论文](https://arxiv.org/abs/2312.06281)。
- 对 **7B 模型与 GPT-4 或 Mixtral 等模型相比**的得分表示怀疑。
- 请求对基于 yi34b 模型的 **Nous Hermes 2** 进行基准测试。

**DiscoResearch 频道总结**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (4 条消息): 
        
- **关于 LLM-as-a-Judge 评估偏见的讨论**：`.calytrix` 对模型被微调以利用裁判模型（如 **gpt-4 对长输出的偏好**）的偏见表示担忧。他们建议通过评估具有参考答案的问题来缓解此问题。
- **Disco Judge 在不同评估背景下的应用**：`_jp1_` 回应称，对于“真实”的评估和基准测试，参考答案会有所帮助，但在合成数据过滤或质量保证等应用中，参考答案通常不存在。他们强调 **Disco Judge** 旨在适应这两种用例。
- **LLM-as-a-Judge 方法论的局限性**：`.calytrix` 进一步提出，当前的 LLM-as-a-Judge 基准测试可能存在局限性，因为它们要么允许 LLM 根据任何设定的指标确定什么是好答案，要么与 LLM 生成的参考答案进行比较。
- **需要对缺点进行积极讨论**：`bjoernp` 同意这些担忧，并强调以后需要对这些局限性进行积极讨论和量化。


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (3 条消息): 
        
- **顶级 7B 模型基准测试**：用户 `@.calytrix` 展示了一些 7B 模型在基准测试排行榜上的结果。值得注意的是，所有这些模型的得分都低于 GPT-4、GPT-3.5 和 Mistral-7B-OpenOrca。基准测试的模型和得分包括：

    - xDAN-AI/xDAN-L1-Chat-RL-v1: 40.12
    - openchat/openchat-3.5-1210: 43.29
    - Weyaxi/SauerkrautLM-UNA-SOLAR-Instruct: 41.01
    - upstage/SOLAR-10.7B-Instruct-v1.0: 41.72
    
    基准测试结果来自 [EQbench](https://www.eqbench.com/)。此处提供[源代码](https://github.com/EQ-bench/EQ-Bench)和[论文](https://arxiv.org/abs/2312.06281)供参考。
 
- **将 7B 模型与 GPT-4 或 Mixtral 进行比较**：用户 `@cybertimon` 对这些 7B 模型表现优于 GPT-4 或 Mixtral 表示怀疑。 
- **Benchmarking Nous Hermes 2**：`@gheshue` 请求 `@.calytrix` 对基于 yi34b 模型的 Nous Hermes 2 进行基准测试。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord 总结

只有一个频道有活动，因此无需总结...

- **用于语言模型交互的 Guidance Library**：`@explore8161` 分享了关于 Guidance Library 的信息，该库旨在增强与语言模型的交互。该工具辅助高效的 Prompt 开发和精准的响应生成，为控制语言模型提供了有效手段。一篇相关的 [Medium 文章](https://medium.com/@saurabhdhandeblog/language-model-prompt-engineering-guidance-library-7b9ad79cf9d4)提供了关于该主题的更多细节。
- **关于日常 Machine Learning 使用案例的博客文章**：`@jillanisofttech` 发布了一个指向 [Medium 博客](https://jillanisofttech.medium.com/ten-everyday-machine-learning-use-cases-a-deep-dive-into-ais-transformative-power-b08afa961e10)的链接，重点介绍了 Machine Learning 的日常使用案例及其变革力量。
- **征求对 Kaggle Notebook 的反馈**：该领域的初学者 `@vengeance3210` 征求对一个关注房价高级回归技术的 [Kaggle notebook](https://www.kaggle.com/code/nishchay331/n6-house-prices-advanced-regression-techniques) 的反馈。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

只有一个频道有活动，因此无需总结...

- **深度学习设备配置**：
  - 用户 `@zmuuuu` 征求关于配备两张 SLI 配置的 **2 3090s NVIDIA founder's edition version cards** 的深度学习设备主板建议，并表达了对散热的担忧。
  - 作为回应，`@jeffreyw128` 建议考虑 **open-air**（开放式）设置以缓解散热问题。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

只有一个频道有活动，因此无需总结...

- **医学图像诊断项目**：用户 `onuralp.` 正在使用 **GPT-4 Vision** 开发一个医学图像诊断项目，并正在寻找用于对比的 Benchmark 模型。他询问了使用 **Med-LM API** 的经验，并对与 Google 产品团队联系的细节感兴趣。
- **开源模型咨询**：`onuralp.` 还询问了可纳入该项目的开源模型。具体来说，他请求提供任何比较 **Hermes 2 Vision** 和 **bakllava** 的 Benchmarking 数据。