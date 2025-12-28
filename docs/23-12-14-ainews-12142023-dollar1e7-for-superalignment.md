---
companies:
- openai
- llamaindex
- perplexity-ai
date: '2023-12-14T22:51:28.552831Z'
description: '**Jan Leike** 正在发起一项受 **Patrick Collison 的 Fast Grants** 启发的新资助计划，以支持人工智能（AI）研究。**OpenAI**
  推出了一个新的开发者推特账号 @OpenAIDevs，用于发布社区动态。关于 **OpenAI 的 Gemini** 和 **Bard** 聊天机器人的讨论强调了它们读取彼此指令并提供独特编程解决方案的能力。用户报告了
  **GPT-4** 的各种问题，包括性能问题、定制困难以及一个已修复的图像识别漏洞。目前，关于**提示工程（prompt engineering）**挑战以及
  Convo-lang 为 API 使用提供的新 **JSON 模式支持**的讨论仍在继续。此外，人们还讨论了对聊天机器人被误用于非法活动的担忧，以及 **Llama2**
  模型和 **Perplexity 聊天机器人**等替代方案。'
id: ce475876-3b1b-474a-9c6f-031e31149698
models:
- gemini
- bard
- gpt-4
- gpt-4.5
- llama-2
original_slug: ainews-12142023-1e7-for-superalignment
people:
- jan-leike
- patrick-collison
title: 2023年12月14日：1000万美元用于超级对齐 (Superalignment)
topics:
- prompt-engineering
- api
- custom-gpt
- json
- bug-fixes
- chatbots
- performance
- tts
- code-generation
- image-recognition
type: archival
---

<!-- buttondown-editor-mode: plaintext -->受 Patrick Collison 的 [Fast Grants](https://future.com/what-we-learned-doing-fast-grants/) 启发，Jan Leike 正在[启动他自己的项目](https://openai.com/blog/superalignment-fast-grants)：

 
![image.png](https://assets.buttondown.email/images/84df1c58-79d6-4e5c-95b8-e47d05c4da15.png?w=960&fit=max)
 

[Notion 页面](https://openai.notion.site/Research-directions-0df8dd8136004615b0936bf48eb6aeb8) 提供了更详细的研究方向，包含许多优秀的论文。

 
![image.png](https://assets.buttondown.email/images/460d5631-a839-42d4-b3c1-e245c8037e8e.png?w=960&fit=max)
 

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **在 Twitter 上推出了新的 OpenAI Developers 账号**，用户名为 @OpenAIDevs。鼓励在线社区成员关注该账号以获取更新。[Twitter 链接](https://fxtwitter.com/OpenAIDevs)。
- 围绕 OpenAI 的 **Gemini 与 Bard** 聊天机器人之间的使用差异进行了讨论，并确认这两个机器人可以互相读取指令，并为解决编程问题提供更新、独特的视角。
- 强调了 **OpenAI 服务的各种问题**，从 OpenAI 网站的问题到存档和网络钓鱼验证码等功能。还包括关于 GPT 定制化的持续对话，以及对 OpenAI 与 Microsoft 之间业务关系的不满。
- 关于 **GPT (OpenAI) 的性能和功能担忧**存在激烈辩论。用户报告了 GPT-4 的各种问题，如不友好的回复、无法访问以及容量问题。此外，还有关于可能发布更新版本 GPT-4.5 的推测。
- 用户分享了在使用 **自定义 GPT-4 时的困难**、处理 PDF 文件附件时的挣扎，以及在 GPT 中使用 Dalle 3 进行风格模仿的潜力。一项公告透露，图像识别的一个 Bug 已得到解决。
- 人们讨论了关于 **Prompt Engineering** 的问题，例如难以让 AI 遵循详细指南来修改代码，以及难以让 AI 将原始文本重写为适合朗读的格式。还提到了 Convo-lang 中引入了 **JSON Mode 支持**的增强功能。
- 在 **API 讨论**中，进一步阐述了 Convo-lang 中新引入的 JSON Mode 支持、制作韦恩图的必要规则、GPT-4 在修改大型脚本时无法保留注释以及引入占位符注释的问题，以及在信息丰富的“真人出镜”视频（尤其是医疗领域）中，如何使人工编写的文本在 TTS 中听起来更自然的挑战。

**OpenAI 频道摘要**

### ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/) (1 条消息): 
        
- **新的 OpenAI Developers Twitter 账号公告**：`@abdubs` 介绍了新的 OpenAI Developers Twitter 账号，并鼓励用户**关注以获取更新**。可以通过此[链接](https://fxtwitter.com/OpenAIDevs)访问该账号。

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (91 messages🔥🔥): 
        
- **Gemini vs Bard**: 用户 `@offline` 讨论了他们使用来自 OpenAI 的 **Gemini** 和 **Bard** 聊天机器人的经验。他们提到这些机器人可以互相读取指令，为编程问题提供独特的视角，并提供比 GPT 更新的信息。用户 `@muyfashionista` 分享说，他们没有注意到这些机器人之间有显著差异。

- **将 AI 用于非法活动**: 用户 `@sylviajade` 提出了关于 ChatGPT 被用于生成恶意软件的担忧，想知道是否对此类用途进行了任何跟踪或监控。回复者 `@lugui` 澄清说，违反服务条款 (ToS)（包括非法活动）可能导致 OpenAI 账号被封禁。

- **定制 GPT**: 用户 `@rudish`、`@elektronisade` 和 `@aznironman` 讨论了如何为各种目的定制 GPT，包括在 Discord 机器人中使用以及创建离线聊天机器人。`@webhead` 建议在大多数情况下使用基于 Llama2 的模型，因为其具有良好的许可条件且有各种模型可用。

- **对 GPT 的担忧及其替代方案**: 用户 `@stevenramenby` 和 `@jerrybobs` 分享了他们对 GPT 的不满，理由是废话连篇和插件失效。`@jerrybobs` 建议研究 AI Assistants，虽然设置起来更难，但据称更可靠。

- **Perplexity 聊天机器人**: 用户 `@chief_executive` 分享了他们使用 Perplexity 聊天机器人的积极体验，称赞其用于信息收集的实时浏览能力。相比之下，他们认为 Perplexity 优于 ChatGPT 的浏览工具和 Bing AI。他们还提到 Perplexity 的 Plus 版本每月 20 美元，每天提供 600 次使用额度。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (544 messages🔥🔥🔥): 
        
- **GPT-4 的性能问题与担忧**: 用户 `@koll0212`、`@stevenramenby` 和 `@lindeberg02` 报告了 GPT-4 的问题，提到它“表现异常”，回复随机词汇和表情符号，无法使用，甚至因为这些问题取消了订阅。`@stevenramenby` 在聊天记录中分享了几个问题实例，并与用户 `@offline`、`@rjkmelb` 和 `@solbus` 进行了讨论。对于 `@stevenramenby` 来说，自定义 Agent 也存在该问题，促使他在指定频道寻求进一步帮助。
- **ChatGPT 宕机体验**: 用户 `@zappymonster`、`@7877`、`@andrew.lol`、`@gardener_windson`、`@acaba1806`、`@bricky___`、`@smokzz` 和 `@thepitviper` 报告了 ChatGPT 平台的宕机情况。他们描述其返回 502 错误、超时或无法打开网站。
- **使用上限与用户界面担忧**: 用户 `@lindeberg02`、`@openheroes`、`@solbus` 和 `@kyoei` 讨论了 ChatGPT Plus 对 GPT-4 的使用上限。`@solbus` 澄清说，目前基础 GPT-4 的使用上限是每 3 小时 40 次，自定义 GPT 限制为每 3 小时 25 次。`@lindeberg02` 抱怨付费后仍受限制，并寻求更多关于上限的信息。
- **对 GPT-4.5 发布后的期待与推测**: 用户 `@realmunknown`、`@DawidM`、`@realgavok`、`@mrcrack_` 和 `@lugui` 参与了关于潜在更新（特别是 GPT-4.5）发布的讨论和推测。这些讨论基于传闻和泄露，`@DawidM` 指出 Reddit 上的一位用户是关于潜在模型更新的信息源。
- **GPT 性能与功能担忧**: 用户 `@skrrt8227`、`@you.wish`、`@afterst0rm`、`@kyper` 和 `@bad3r` 对 GPT 感知到的局限性提出了批评，包括无法有效格式化文本、缺乏一致性以及使用上限的影响。`@bad3r` 特别批评了平台的服务器容量，认为应该为 ChatGPT Plus 用户分配更多带宽。

### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (251 messages🔥🔥): 
        
- **OpenAI 服务问题**：多位用户报告了 OpenAI 服务的问题，包括 OpenAI 网站访问问题（来自 `@jcrabtree410`、`@mohammad_soqar101`、`@l3xp3rt`、`@kamikolart`、`@dsssssssss` 和 `@cosmosfractal`），以及特定功能的问题，如存档功能（`@nickiee` 和 `@xhls`），还有尽管从熟悉设备登录仍弹出钓鱼验证码的问题（`@vassiou`）。
- **聊天机器人讨论**：多位用户寻求帮助或围绕 GPT 的使用和定制进行交流。用户如 `@stevenramenby`、`@bionicle1337` 和 `.cybersenpai` 讨论了关于自定义 GPTs、插件行为和 fine-tuning 的问题。`@elektronisade` 和 `@satanhashtag` 等人提供了若干澄清。
- **关于 OpenAI 和 Microsoft 的争议**：用户 `@bionicle1337` 对 OpenAI 和 Microsoft 之间的商业关系表示不满，认为存在不公平商业行为和反垄断违规。一些社区成员如 `@elektronisade` 和 `@Rock` 提供了不同观点，但讨论仍未达成共识。
- **产品功能与改进讨论**：`@stevenramenby` 报告了自定义 GPT 输出的异常，`@jtensetti` 讨论了 Plus 订阅的名称切换（name toggle）问题。与此同时，`.staycurious` 询问了如何设置自动励志邮件功能，`@jah777` 询问了在 Discord 上集成 ChatGPT 机器人的事宜。其他成员提供了相关建议。
- **多语言讨论**：多位用户正在使用多种语言进行交流，`@satanhashtag` 提醒大家使用英语以便更广泛的沟通。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (16 messages🔥): 
        
- **对话限制问题**：用户 `@zeriouszhit` 对即使在遇到 OpenAI 端的网络错误或 Bug 时仍面临对话限制表示担忧，描述了无法处理图像却被扣除对话额度的情况。
- **图像识别 Bug 修复**：`@solbus` 提到，根据[特定 Discord 线程](https://discord.com/channels/974519864045756446/1183909233436131358)的报告，一个图像识别 Bug 已经解决。
- **自定义 GPT 消失**：`@ps5671` 和 `@pietman` 讨论了自定义 GPTs 消失或无法编辑的问题，这些问题似乎最终得到了解决。
- **GPT 生成详细技术分析图表的能力**：`@consciousness3783` 询问了 GPT-4 在通过单个 Prompt 结合图像和文本生成包含价格行为指标等细节的技术分析图表方面的能力。
- **PDF 文件问题**：`@jazy_87295` 在尝试向 GPT 附加 PDF 文件时遇到困难，询问是否需要转换为 TXT 格式。
- **使用 Dalle 3 进行风格模仿**：`@xchronos` 分享了一个想法，即在 GPT 中使用 Dalle 3 创建模仿其他提供图像风格的图像。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (22 messages🔥): 
        
- **Convo-lang 更新**：用户 `@iyioio` 报告了 **Convo-lang** 的增强功能，现已支持 **JSON mode**。他们提供了如何定义结构体（structs）以及利用 JSON 处理各种数据格式的详细示例，包括将其与 Vision 和 function calling 结合使用的讨论。更多详情请参阅 Convo-lang 的 [NPM 文档](https://www.npmjs.com/package/convo-lang)。
- **聊天机器人指令**：用户 `@offline` 建议采用在指令前加数字前缀的策略以便引用，并强调除非用户另有说明，否则机器人应仅生成 **3 圆维恩图（VENN diagrams）**。
- **脚本修改指南**：`@joeyjoejoeshabadoo42` 在让 AI 遵守其详细的代码修改指南时遇到困难，特别是当代码量达到或超过 110 行时。规则包括保留所有代码注释、尽可能减少代码改动、清晰解释修改内容、尊重代码历史，并始终提供完整的修改后脚本而非占位符注释。
- **自然语言处理**：`@.ursium` 表示在特定医学视频制作场景下，很难让 AI 将原文改写为适合朗读的格式。他们强调了 AI 会改变文本细微差别的问题，有时会影响重要信息的含义。
- **NLP 响应**：`@world_designer` 建议告知 AI 脚本的背景和性质，以获得更准确和细致的理解，`@.ursium` 同意尝试此方法。

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (22 messages🔥): 
        
- **Convo-lang 支持 JSON Mode**：`@iyioio` 在 Convo-lang 中引入了 JSON Mode 支持，并提供了在简单和复杂场景下的应用示例，例如列出太阳系行星和描述图像中的人物。更多详情请参阅 [convo-lang 的 NPM 文档](https://npmjs.com/package/convo-lang)。

- **生成维恩图的规则**：`@offline` 明确了一条规则：生成的维恩图（Venn diagrams）除非用户另有说明，否则只能包含 3 个圆圈。

- **代码修改的自定义指令问题**：`@joeyjoejoeshabadoo42` 在 Web 聊天 UI 中使用 GPT-4 的自定义指令处理大型 Python 脚本时遇到问题。当脚本长度达到约 110 行时，AI 无法遵守指令，会出现占位符注释并删除代码注释。

- **让“真人出镜”视频的文本听起来更自然**：`@.ursium` 正尝试让医疗领域的真人撰写文本在“真人出镜”（Talking Head）视频的 TTS 中听起来更自然。问题在于，当要求简化文本或删除难发音的词汇时，文本会被重写，且关键动词如 'may' 或 'sometimes' 会被替换或删除，从而改变了原意。

- **TTS AI 模型对比 ChatGPT**：在 `@.ursium` 和 `@world_designer` 的对话中，`@world_designer` 最初建议针对 `@.ursium` 的问题使用独立的 TTS AI 模型。然而，在 `@.ursium` 澄清问题涉及调整书面文本以适应口头表达而非生成 TTS 后，`@world_designer` 建议向 ChatGPT 解释脚本的上下文。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- 关于 AI 性能和训练策略的广泛对话，涉及 PHI-2 和 A6000 等模型。关键讨论包括 AI 模型的潜在**过度对齐（overalignment）**、获取**免费算力**的策略、通过减小 **micro batch size** 等技术处理 **CUDA out of memory** 问题，以及对微调过程中**过度训练（overtraining）**风险的思考。
- `@faldore` 发起了关于 **DPO** 训练的深入探讨，并分享了 OpenAccess AI Collective Axolotl 仓库中的 RL/DPO Pull Request [\#935](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935) 链接。
- 围绕**模板（template）**和 **token 配置**的重要讨论，例如 **LLAMA2 SeparatorStyle** 的逻辑（似乎跳过了消息中的索引 0，导致丢弃了所有指令），以及由 `@Aleksa Gordić` 提出的与 **Hugging Face** 分词器相关的**意外空格添加**问题。
- 用户还分享了对处理 **EOS（句子结束）token 生成**问题、推理阶段 **Qlora 配合 ChatML 模板**的正确加载时间，以及寻找合适的模型训练配置格式的探索。
- 针对上述挑战的建议和修复通常伴随着 Git Pull Request，例如 `@hamelh` 提交的 PR [\#952](https://github.com/OpenAccess-AI-Collective/axolotl/pull/952)，旨在纠正 LLAMA 模板问题。
- 该社区对**数据集**进行了详细对话，讨论了模型的 **token 与 TB 比例**、**硬件要求**以及模型训练时长等话题。此外，还有关于**函数调用数据集**的咨询，[Glaive Function Calling v2 dataset](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 被作为潜在资源分享。

**OpenAccess AI Collective (axolotl) Channel Summaries**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (85 messages🔥🔥): 
        
- **AI 过度对齐与性能**：`@noobmaster29` 和 `@faldore` 讨论了他们的 AI 模型可能存在过度对齐（overaligned）的问题。`@faldore` 分享了他为 AI 模型使用的一个幽默的 system prompt，小组讨论了如果 AI 意识到激励机制是虚假的时可能产生的后果。
- **AI 训练**：`@faldore` 讨论了他训练 phi2 模型的计划。`@le_mess` 询问了关于获取免费算力（compute）的建议，对此 `@faldore` 建议申请微软的 Level 3 创业计划。
- **AI 模型微调与 OOM 问题**：`@mihai4256` 在使用 deepspeed zero3 微调拥有 8 x A100（即 8 * 80 GB）的 Yi 模型时遇到了 CUDA 显存溢出（OOM）问题，`@nruaif` 建议减小 micro batch size 并增加 accumulation steps。`@dangfutures` 在进行全量模型微调时也遇到了显存溢出问题。
- **DPO 训练与 Git 仓库**：用户们讨论了 DPO 训练，`@faldore` 分享了 OpenAccess AI Collective Axolotl 仓库中一个关于 RL/DPO 的 pull request [#935](https://github.com/OpenAccess-AI-Collective/axolotl/pull/935) 链接。
- **故障排除**：`@matts9903` 和 `@dangfutures` 报告了在保存 checkpoints 以及模型训练一个 epoch 后出现的问题。他们收到了来自 `@caseus_` 和 `@dangfutures` 的一些排错建议。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (111 messages🔥🔥): 
        
- **微调与过拟合**：用户 `@caseus_`、`@faldore`、`@noobmaster29` 和 `@nruaif` 讨论了在 A6000s 和 PHI-2 模型上应进行多少程度的微调和 Qlora，该话题引发了对过度训练（overtraining）的担忧。用户建议先尝试全量微调（full fine-tuning）和 Qlora，并指出如果模型开始出现过拟合（overfit），可以进行调整。
  
- **Mixtral 的优化关注**：用户 `@casper_ai` 分享了他尝试使用 unsloth.ai 的优化建议但未成功的经历。他期待他们的潜在 pull request (PR)，以检查是否确实能实现改进，特别是在内存至关重要的 Mixtral 模型上。

- **Llama2 模板问题**：用户 `@Aleksa Gordić` 提出了一个关于 `airoboros`、`llama-2` 等依赖 `SeparatorStyle.LLAMA2` 逻辑的问题，因为它似乎跳过了消息中的索引 0，从而丢弃了任何指令。他提出了一个解决方案，即索引 0 应该与 system prompt 一起产出。该讨论进一步演变为对更稳健的解决方案、测试用例以及其他与 Llama2 相关的 prompt 组装问题的讨论。

- **Llama 模板问题的修复**：用户 `@caseus_` 强调了由 `@hamelh` 开启的一个用于修复 Llama 模板问题的 PR。该 PR [#952](https://github.com/OpenAccess-AI-Collective/axolotl/pull/952) 解决了 EOS/BOS 应用中的错误以及其他问题。

- **Token 空格与解码问题**：用户 `@Aleksa Gordić` 发现了一个与 Hugging Face tokenizer 相关的意外空格添加问题。这引发了关于 token 操作复杂性的讨论，`@hamelh` 建议将 Hugging Face chat templates 作为解决问题的“事实来源”（source of truth）。

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (79 messages🔥🔥): 
        
- **关于 Special Tokens 和配置的讨论**：用户 `@noobmaster29` 讨论了他们遇到的一个挑战：模型不生成句子结束（EOS）token，导致模型持续输出直到达到 token 限制。他们分享了尝试过的各种配置设置，包括设置 EOS token 的不同方法，并向其他用户寻求建议。
  
- **使用 ChatML 模板运行 Qlora**：关于在 ChatML 模板下使用 Qlora，`@noobmaster29` 对在 inference 阶段何时加载 ChatML 模板表示困惑。`@caseus_` 确认模板应以该格式加载。
  
- **训练模型的配置格式**：还有关于训练模型时应使用何种配置格式的讨论。`@noobmaster29` 和 `@self.1` 不确定在使用 ChatML 格式运行 prompt 时正确的 token 配置。

- **Inference 问题**：此外，`@noobmaster29` 报告称，当他们在 inference 过程中使用特定的对话设置时，模型没有遵循 ChatML 格式。`@nanobitz` 建议使用带有 `--debug` 参数的 `preprocess` CLI 检查完整输出。

- **感兴趣的链接**：`@noobmaster29` 分享了一个 [Hugging Face 模型链接](https://huggingface.co/ehartford/dolphin-2.1-mistral-7b/blob/main/configs/dolphin-mistral-7b.yml)，其配置在 ChatML 下运行良好，但在尝试复制时，他们没有获得相同的结果。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (33 messages🔥): 
        
- **数据集大小和训练指标讨论**：用户 `@natefyi_30842`、`@noobmaster29` 和 `@nanobitz` 之间进行了详细讨论。他们讨论了模型的 token 与 TB 比例、训练模型所需的时长和硬件，并参考了 PHI-2 和 OPT 模型。用户 `@natefyi_30842` 澄清了关于训练不同模型所需时间和 GPUs 的误解（[PHI-2 card](https://huggingface.co/microsoft/phi-2), [OPT-125m card](https://huggingface.co/facebook/opt-125m)）。
- **训练较小的模型**：用户 `@noobmaster29` 建议，亿级 token 范围的模型对于个人努力可能是可行的，并链接到了 [Hugging Face 上的 OPT 模型](https://huggingface.co/facebook/opt-125m)。
- **关于 Function Calling 数据集的咨询**：用户 `@le_mess` 征求关于 function calling 数据集的建议，并分享了 [Glaive Function Calling v2 数据集](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)和一个 [function calling 排行榜](https://huggingface.co/spaces/Nexusflow/Nexus_Function_Calling_Leaderboard)。`@bjoernp` 也对此表示了兴趣。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- 深入讨论和推测，重点关注 AI 模型训练策略以及服务器内的数据集处理。特别是围绕 **MetaMath 数据集潜在污染** 的广泛对话，尽管存在成本担忧，仍提出了自动污染检查的建议。重点还扩展到针对合并参数的神经架构搜索 (NAS) 等策略，以取代过度集中于针对基准测试 (Benchmarks) 的模型训练。
    - [Code Contamination Detector](https://github.com/swj0419/detect-pretrain-code-contamination)
    - [Merge Kit](https://github.com/cg123/mergekit)
    - [关于 MetaMath 可能存在污染的讨论](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/265)
- 关于**即将推出的模型**的闲聊，包括用户对 Llama 3 和 Galactica 2.0 能力的推测性期待。特别关注像 Dolphin 2.5 Mixtral 8x7b 这样专门用于编程的模型，并对其在编程方面的熟练程度表示认可。
- 关于创建**解耦且持续改进的世界模型**的讨论，由 `@maxwellandrews` 发起，并对其高层级方法进行了思考。
- 对用于 3D 渲染和机器学习等非游戏用途的**高性能工作站配置**的共同兴趣。分享的具体链接包括一个[名为 "Big Boss" 的高性能工作站](https://www.extremetech.com/computing/big-boss-workstation-debuts-with-7-rtx-4090-gpus-31k-price-tag)，以及在 [NixOS 论坛](https://discourse.nixos.org/t/jaxlibwithcuda-not-using-cuda/36873/2)上关于 NixOS 和使用 Nix 管理 Python 软件包的讨论。
- 关于 **AI 模型和分词 (Tokenization)** 的探讨性对话，包括 Based 模型架构、作为 BPE 潜在替代方案的 MorphPiece 以及 Phi-2 的性能。对 Phi-2 表示了明显的失望，因为它未能修补之前的 Phi-1/1.5 模型留下的漏洞。分享的链接包括关于 [Based 的推文](https://twitter.com/simran_s_arora/status/1735023478594543960)、关于 [MorphPiece 的论文](https://arxiv.org/abs/2307.07262)、OpenAI 的[研究论文](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)以及关于 [FunSearch 的博客文章](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)。
- 来自用户的模型性能更新，`@gabriel_syme` 报告了 **Solar Model** 的平均得分，他随后建议对 Solar 模型进行微调和整合；`@bevvy` 报告了 **RedPajama 3B 模型** 的运行情况，尽管速度较慢。
- 对 **GPT4-Turbo 定价** 的批评性评论，普遍共识认为即使对于高价值任务，其成本也过高。
- 在 Ask-about-llms 频道中提出的关于 Solar 10 预设、MLC 替代方案（建议使用 **Android/iOS 配置**）以及对 **LLM Farm** 的引用的查询和指引。
- 最后，专业的讨论氛围中穿插了 `@gezegen` 的个人近况，提到他正从流感中康复，错过了一些对话。

**Nous Research AI 频道摘要**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (1 条消息): 
        
- **解耦且持续改进的世界模型**：用户 `@maxwellandrews` 正在思考创建**解耦且持续改进的世界模型**的高层级方法。

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (27 messages🔥): 
        
- **显卡讨论**：`@everyoneisgross` 分享了他们显卡的一些问题，特别是提到了黑色元素和棋盘格渲染。他们考虑到了其七彩虹 3060 显卡的局限性。 
- **高性能工作站**：`@skadeskoten` 分享了一个指向 [ExtremeTech 的链接](https://www.extremetech.com/computing/big-boss-workstation-debuts-with-7-rtx-4090-gpus-31k-price-tag)，关于一台来自德国名为 "Big Boss" 的高性能工作站，它配备了 7 个 RTX 4090 GPU 和一个 64 核 Threadripper Pro 5995WX。该工作站是为 3D 渲染和 Machine Learning 而设计的，而非游戏。
- **模型 Finetuning**：`@euclaise` 表示他们的 GPT-7b 模型可以使用其自定义 Optimizer 在单块 GeForce 3090 上进行 Finetuning。他们推测 Mixtral 模型可能也适用于 "Big Boss" 工作站配置。
- **NixOS 和 Nix 包管理**：`@euclaise` 询问频道中是否有人使用 NixOS 并通过 Nix 管理 Python 包。他们还分享了 NixOS 论坛上的一个[讨论链接](https://discourse.nixos.org/t/jaxlibwithcuda-not-using-cuda/36873/2)。
- **为特定语言训练模型**：`@skadeskoten` 表达了为挪威语（特别是医学挪威语）训练语言模型的愿望。`@euclaise` 回复称完成此类任务需要很长时间。


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (2 messages): 
        
- **Solar 模型性能**：`@gabriel_syme` 报告称 **Solar 模型** 的平均分保持在 74，但未提供该指标衡量内容的具体说明。
- **Finetuning 与 Slerp Solar 模型**：在随后的一条消息中，`@gabriel_syme` 建议需要对 Solar 模型进行 **Finetuning** 和 **Slerp**。


### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (36 messages🔥): 
        
- **关于 Phi-2 性能的讨论**：用户 `@bevvy` 和 `@georgejrjrjr` 讨论了 **Phi-2** 的性能。`@georgejrjrjr` 对 Phi-2 未能修补之前 Phi-1/1.5 模型留下的缺陷表示失望，并推测它实际上可能是 Phi-CTNL 的一个案例。
- **Based 模型架构与 MorphPiece Tokenization**：讨论了包括 **Based** 在内的新模型架构，`@nods` 分享了 Simran Arora 的一条 [推文链接](https://twitter.com/simran_s_arora/status/1735023478594543960)，详细介绍了其简洁性和效率。`@georgejrjrjr` 注意到新模型架构的快速涌现，还提到了 **MorphPiece** 作为潜在的 BPE (Byte Pair Encoding) 替代方案。他分享了一篇关于该主题的 [论文链接](https://arxiv.org/abs/2307.07262)。
- **Fine-tuning 数据集创建**：用户 `@atgctg` 提出了创建用于 **Fine-tuning** 的指令数据集这一话题，认为这是与 Pre-training 数据策展同样重要的并行任务。
- **GPT4-Turbo 的定价**：由 `@gabriel_syme` 和 `@giftedgummybee` 发起的一场讨论批评了 **GPT4-Turbo** 的高昂定价。即使对于高价值任务，这些价格也被认为是不合理的。
- **Weak to Strong Generalization 论文与 FunSearch**：`@giftedgummybee` 分享了 OpenAI 的一篇 [研究论文](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf) 链接，`@coffeebean6887` 分享了包含代码的 [GitHub 仓库](https://github.com/openai/weak-to-strong)。`@nods` 分享了一篇关于 **FunSearch** 的 [博客文章](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)，引发了用户对其应用和新颖性的讨论。

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (139 messages🔥🔥): 
        
- **MetaMath 数据集讨论**：关于 MetaMath 数据集潜在污染问题的辩论非常激烈。`@bjoernp` 提到在模型训练中避免污染数据的重要性，`@mihai4256` 在运行测试后确认了潜在的污染，得分 0.96（通常高于 0.85 被认为极有可能存在污染）。还有关于在提交模型时自动检查污染的讨论，但潜在的成本被认为是一个顾虑。

- **AI 模型训练**：多位用户讨论了模型训练策略，`@nonameusr` 对过度关注针对 benchmark 的模型训练表示担忧。`@euclaise` 建议在合并参数上采用 Neural Architecture Search (NAS)，引发了关于模型合并（model merging）的进一步讨论。

- **即将推出的模型**：`@Error.PDF` 和 `@gabriel_syme` 等用户推测了 Llama 3 和 Galactica 2.0 等未来模型的潜在发布和能力。然而，目前尚未确认具体的发布日期或功能。

- **专注于编程**：`@nonameusr` 和 `@metaldragon01` 讨论了专门用于编程的模型，重点关注了最近发布的 Dolphin 2.5 Mixtral 8x7b 模型，该模型声称在编程方面表现“非常”出色。

- **模型执行**：关于模型执行的讨论包括 `.beowulfbr` 提到的 QuIP 模型，以及如何有效地在这些模型上运行推理（inference）。`@decruz` 提到了在手机上运行 Openhermes。 

讨论中包含的相关资源和链接：

- [Code Contamination Detector](https://github.com/swj0419/detect-pretrain-code-contamination)
- [Merge Kit](https://github.com/cg123/mergekit)
- [Openhermes](https://huggingface.co/relaxml/Openhermes-7b-HI-4Bit-Packed)
- [关于 MetaMath 可能存在污染的讨论](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/265)


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (6 messages): 
        
- **RedPajama 3B 性能**：用户 `@bevvy` 报告称 **RedPajama 3B** 可以运行，但速度较慢。在 Pixel 7 上预填充（prefill）较慢，系统运行速度为 3 tokens/s。
- **关于 Solar 10 预设的问题**：用户 `@agcobra1` 询问了 **Solar 10 使用的预设（preset）**。在给定的消息历史中没有提供回复。
- **MLC 的替代方案**：用户 `@gezegen` 询问是否有使用 MLC 的替代方案，并建议在 **Android/iOS** 上进行设置。
- **提及 LLM Farm**：用户 `@orabazes` 提到了 **LLM Farm**，但在给定的消息历史中上下文并不明确。
- **用户生病**：用户 `@gezegen` 提到他们正在从流感中康复，可能错过了一些之前的对话。

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 摘要

- **NeurIPS 聚会外展与 LSTM 讨论**：用户 `@.yosun` 邀请最后一刻的参会者参加 *[NeurIPS 聚会](https://twitter.com/Yosun/status/1735091122202890697)* 和 AI3D 早餐会。此外，`@vipitis` 在对话中阐明了如何使用 LSTM，并分享了 [PyTorch LSTM 文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) 的链接。

- **聊天机器人与 VSCode 插件讨论**：用户 `@criticaldevx`、`@jeffry4754`、`@shrap42`、`@ahmad3794` 和 `@funapple` 简要讨论了如何在 LLAMA 聊天机器人中自定义 AI 名称。建议包括微调（Fine-tuning）以及在生成后使用脚本替换名称。
        
- **深入探讨 RNN, LSTM 与 GRU**：用户 `@nerdimo` 详细解释了循环神经网络（RNN）的概念，以及 LSTM 和 GRU 等组件。他们还讨论了正在学习的资源——吴恩达（Andrew Ng）机器学习专项课程。用户 `.memoshi` 分享了一篇关于 **SwitchHead 方法** 的 [arXiv](https://arxiv.org/abs/2312.07987) 论文。

- **太空主题 Twitter 账号与包容性 AI**：`@boggdanu_` 分享了他们专注于天文学的 Twitter 账号 [@madpapayas](https://twitter.com/madpapayas)，`@sk21_` 讨论了包容性 AI 设计，并引用了 [Dr Maria Panagiotidi 的文章](https://uxpsychology.substack.com/p/creating-inclusive-ai-a-strengths)。

- **AI 相关项目与教程**：用户分享了他们的 AI 项目并寻求反馈。`@marielandryceo` 展示了 AI 的方法论，`@rwitz_` 介绍了两个 AI 模型的合并，`@appstormer_25583` 发布了一个分析 Logo 的 AI。一个关于微调的问题得到了解决。

- **MagVIT2 演示与新阅读材料建议**：`@chad_in_the_house` 介绍了 **MagVIT2** 并分享了相关资源，`@memehunter7209` 建议将 *Mathematical Machine Learning* 作为潜在的阅读小组书籍，最后 `@netskink.1` 邀请成员参与他们的数据集项目。 

- **图像条件 Diffusion Model 及其他讨论**：`@mr.frog2014` 分享了他们在图像条件 Diffusion Model 实验中的见解。`@nixon_88316` 寻求关于 stablecode 的信息，`@sebys7` 询问了 SD-X4-Upscaler 中的参数调整。

- **PyTorch 模型训练与层级文本分类**：用户 `@merve3234` 提供了使用 PyTorch 训练 Transformer 模型的说明，`@stroggoz` 讨论了为每个层级使用独立模型的层级文本分类（Hierarchical Text Classification）。此外，`@ppros666` 征求了关于首次微调 LLM 的建议，并重点介绍了一份相关指南。


**HuggingFace Discord 频道摘要**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (82 条消息🔥🔥): 
        
- **NeurIPS 聚会**：用户 `@.yosun` 正在召集最后一刻的参会者参加 NeurIPS 聚会和次日的 AI3D 早餐会。分享了 [Twitter 帖子链接](https://twitter.com/Yosun/status/1735091122202890697)。
- **使用 LSTM**：用户 `@vipitis` 解释了 LSTM 的功能以及多层 LSTM 的工作原理。建议查看 [PyTorch LSTM 文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) 以了解其实现。
- **在 LLAMA 中自定义 AI 名称**：用户 `@criticaldevx`、`@jeffry4754`、`@shrap42`、`@ahmad3794` 讨论了如何更改 LLAMA 聊天机器人中的 AI 名称。建议包括微调和在生成后使用脚本替换名称。
- **训练故障**：`@vishyouluck` 提出了使用 `autotrain_advanced` 微调模型时遇到的问题，怀疑是 `train.csv` 文件出了问题。
- **VSCode 插件**：`@funapple` 征求关于能与本地 LLM 良好协作以提供实时代码建议的 VSCode 插件推荐。

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (6 messages): 
        
- **理解 Recurrent Neural Networks**: 用户 `@nerdimo` 深入研究了 Recurrent Neural Networks (RNNs) 的概念，包括 Long Short-Term Memory (LSTM) 和 Gated Recurrent Unit (GRU)。他们解释了 LSTM 和 GRU 如何利用门（gates）来更新其内部记忆并处理相关信息，这有助于做出准确预测并避免梯度消失（vanishing gradient）问题。
- 在补充讨论中，`@nerdimo` 提到了 **bidirectional RNNs** 的概念，其中信息流可以从左到右以及反向进行。此外，他们还讨论了构成深度 RNN 的 **stacked RNNs** 的强大功能（以及计算成本）。
- 用户 `@merve3234` 询问在 GRU 训练效率通常更高的情况下，GRU 和 LSTM 之间是否存在显著的性能差异。他们还询问了 `@nerdimo` 正在使用的学习资源。
- 作为回应，`@nerdimo` 表示他们正在**学习 Andrew Ng Machine Learning specialization**。他们还表达了自己的直觉，认为由于 LSTM 具有更多的过滤机制和参数，它会是最佳选择。
- **用于 Transformers 的 SwitchHead 方法**: 用户 `.memoshi` 分享了一篇关于 **SwitchHead 方法** 的 [arXiv 论文](https://arxiv.org/abs/2312.07987)链接。该方法声称可以减少 Transformers 中 self-attention 层的计算和内存需求，在不牺牲语言建模性能的情况下实现实际加速。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (2 messages): 
        
- **天文学 Twitter 账号**: `@boggdanu_` 分享了他的 Twitter 账号 [@madpapayas](https://twitter.com/madpapayas) 的链接，该账号专注于天文学。 
- **创建包容性 AI**: `@sk21_` 转发了一篇关于创建包容性 AI 的 Substack 文章。该文章副标题为“一种基于优势的人工智能方法”，由 Maria Panagiotidi 博士撰写，讨论了 AI 设计中的包容性问题。链接如下：[Creating Inclusive AI](https://uxpsychology.substack.com/p/creating-inclusive-ai-a-strengths)。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 messages): 
        
- **探索深度：AI 中的批判性思维 ToT 与科学方法 CoT**: 用户 `@marielandryceo` 分享了指导人工智能世界的两种基本方法的广泛概述，即 *Critical Thinking Tree of Thoughts (ToT)* 和 *Scientific Method Chain of Thoughts (CoT)*。帖子详细介绍了这些方法在 AI 中运作的逐步过程，强调了持续完善我们理解的重要性。讨论使用了标签 `#CriticalThinkingToT`、`#ScientificMethodCoT` 和 `#AIExploration`。
    
- **AI 模型合并**: `@rwitz_` [分享了一个链接](https://huggingface.co/rwitz2/go-bruins-v2.1)，内容是使用 slerp 作为合并方法合并了两个 AI 模型——`viethq188/LeoScorpius-7B-Chat-DPO` 和 `GreenNode/GreenNodeLM-7B-v1olet`。他详细说明了合并过程中涉及的切片源（slice sources）和参数。合并的基础模型是 `viethq188/LeoScorpius-7B-Chat-DPO`。  

- **品牌 Logo 分析器 GPT**: 用户 `@appstormer_25583` 介绍了一个 AI，它可以根据上传的图片提供 Logo 设计反馈和改进建议。提供了[项目链接](https://beta.appstorm.ai/share?url=73dfbb7a)。

- `@merve3234` 询问 `@rwitz_` 是否已将其合并模型提交到排行榜，他给出了肯定的回答。

### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (15 messages🔥): 
        
- **MagVIT2 的演示**：用户 `@chad_in_the_house` 发起了关于 **MagVIT2** 的演示，这是卡内基梅隆大学和 Google 在论文 "MagVIT2: Language Model Beats Diffusion: Tokenizer is key to visual generation" 中介绍的模型。该模型可以将图像生成为 Token/单词，并被认为击败了 Diffusion 模型。为了进行更深入的讨论，分享了 [博客](https://isamu-website.medium.com/understanding-magvit2-language-model-beats-diffusion-tokenizer-is-key-to-visual-generation-8adba03b724c) 链接，代码可在 [此处](https://github.com/lucidrains/magvit2-pytorch) 获取。
- **阅读小组材料建议**：`@memehunter7209` 建议在阅读小组中学习 [mml-book](https://mml-book.github.io/)，以便更好地理解机器学习所需的数学知识。
- **关于 mml-book 的讨论**：`@Pantera` 和 `@charlieparker8035` 分别询问了该书的难度和内容，`@memehunter7209` 建议该书更像是一本复习书，并推荐了 Gilbert Strang 的讲座和 EDx 上的课程。
- **项目邀请**：`@netskink.1` 邀请成员参与他们的项目，该项目致力于研究图像和天气条件的数据集，旨在检测结冰的桥梁。
- **关于解决未解数学问题的建议**：`@caleb_sol` 提议讨论关于 AI 解决此前未解数学问题能力的 [论文](https://www.nature.com/articles/s41586-023-06924-6)，指出这可能是阅读小组的一个好话题。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **图像条件 Diffusion 模型**：用户 `@mr.frog2014` 分享了他们在图像条件 Diffusion 模型上的实验，建议直接向 x 注入噪声并在去噪模型后减去的想法。他们还提出了关于加入 Attention 模块潜在益处的问题。
- **关于 StableCode 的查询**：用户 `@nixon_88316` 提出了一个寻求 StableCode 信息的查询。
- **SD-X4-Upscaler 参数查询**：用户 `@sebys7` 正在使用带有 Diffusers 的 SD-X4-Upscaler，并询问了可能产生特定类型结果的参数调整。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (6 messages): 
        
- **模型训练技巧**：用户 `@merve3234` 提供了关于在 PyTorch 中使用 Transformer 模型和 Decoder 输入 Token 的详细说明。他们建议在训练循环的前向传播阶段手动输入 `decoder_input_ids`。
- **层级文本分类讨论**：`@stroggoz` 提出了一种使用小型 Bert 模型网络对学术文本进行分类的方法。讨论提出了按主要主题（如数学、化学）对文本进行分类，然后进一步按子主题分类的想法，每个层级使用独立的模型。
- **LLM 上下文长度澄清**：用户 `@ppros666` 要求澄清 Llama 2 的上下文长度，特别是询问上下文长度是 4000 还是 4096。`@Cubie | Tom` 澄清上下文长度为 4096。相关信息链接自 HuggingFace 模型的 [config.json 文件](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json#L12)。
- **LLM 微调指导请求**：`@ppros666` 表达了对首次微调 LLM 的兴趣，并请求最新的教程或示例代码。他们重点提到了一个名为 ["Llama-2 4bit fine-tune with dolly-15k on Colab"](https://colab.research.google.com/drive/134o_cXcMe_lsvl15ZE_4Y75Kstepsntu?usp=sharing) 的指南，并寻求关于其可靠性的建议。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 messages): 
        
- **图像条件 Diffusion 模型**：用户 `@mr.frog2014` 讨论了对标准图像条件 Diffusion 模型进行修改的可能性，提出了一种将条件信号直接添加到注入的噪声中，并在去噪模型之后随后减去的方法。他们还询问实现 Attention 模块来注入条件是否会产生更好的结果。
- **关于 StableCode 的查询**：`@nixon_88316` 发帖询问是否有人了解 StableCode。然而，在提供的消息历史中没有记录到任何回复或后续行动。
- **SD-X4-Upscaler 使用**：用户 `@sebys7` 询问在使用带有 Diffusers 的 `sd-x4-upscaler` 时，是否存在特定参数会导致生成的图像出现特定结果。然而，在给定的上下文中没有提供特定的图像或结果作为参考。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 关于 LangChain 集成与使用的详细讨论：广泛探讨了[官方文档](https://www.langchain.com)、[YouTube 教程](https://www.youtube.com/@LangChain)、大语言模型 (LLMs) 的创建、输出流式传输以及元数据的实现。
- 针对 Plan-and-Execute 的进展、从 LangServe 的 'DOCUMENTS' 栏目复制文本的困难，以及访问 LangServe 端点和字段等问题提出的查询，引发了集体的故障排查和信息交流。
- 分享的贡献包括：`@andysingal` 关于 LangChain Expression Language (LCEL) 的 [Medium 博客文章](https://medium.com/ai-artistry/mastering-chain-composition-with-langchain-expression-language-lcel-2d5041fb0cbd)；`@appstormer_25583` 的[基于 GPT 的品牌 Logo 分析器](https://beta.appstorm.ai/share?url=73dfbb7a)；`@pagerize_admin` 的 [Pagerize](https://pagerize.ai/) 视频摘要生成器；`@gokusan` 用于生产级应用的 [TinyLLM 库](https://github.com/zozoheir/tinyllm/tree/main)；以及 `@joshuasundance` 用于检测 API 密钥的 [pre-commit 钩子](https://github.com/joshuasundance-swca/detect_llm_api_keys)。
- 社区成员参与了由 Catena Labs 主办的 LLM 用户调查，以提供有关市场偏好的见解，调查问卷可在此处[访问](https://tally.so/r/mO7q0p)。

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (48 条消息🔥): 
        
- **LangChain 的集成与学习资源**：`@chasemcdo` 建议利用 LangChain 官网的[文档](https://www.langchain.com)及其 [YouTube 频道](https://www.youtube.com/@LangChain)获取更多信息和教程。
- **Plan-and-Execute 进展查询**：`@casper_mars` 询问了 Plan-and-Execute 的进展，`@manlylubbin` 回复称他们仍在计划中。
- **在 LLM 上实现元数据**：`@reddiamond69` 强调在生成输出时，LangChain 允许打印源文档及其元数据，这可以在用户的应用程序中实现。
- **通过 API 进行流式输出的困难**：`@menny9762` 分享了在通过 Next.js 进行输出流式传输时遇到的困难。随后引发了包括 `@seththunder` 等人在内的广泛讨论，涉及大语言模型 (LLMs) 的创建与使用、流式传输和回调方法。
- **来自 Catena Labs 的 LLM 调查**：`@jay_wooow` 发起了一项社区调查，旨在收集有关 LLM 使用情况和偏好的数据，以为产品开发提供参考并提供有用的市场数据。调查托管在[此处](https://tally.so/r/mO7q0p)。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (3 条消息): 
        
- **从输出栏目复制文本**：`@fritz4374` 反馈难以从运行结果的 'DOCUMENTS' 栏目中复制文本。用户可以从输入和输出栏目复制，但无法从 'DOCUMENTS' 栏目复制。
- **LangServe 端点与字段访问**：`@khophi.co` 正在寻求帮助，希望在向 LangServe 的 `path="/myendpoint"` 发送前端请求（如 `{ 'mymessage': 'message' }`）时，能够访问 LangChain 上下文中的 `<payload>.history` 字段。用户想知道如何获取除 LangServe 自动处理的 'input' 字段之外的其他字段。

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (7 messages): 
        
- **LangChain Expression Language (LCEL) 指南**：`@andysingal` 分享了一篇关于掌握 **LangChain Expression Language (LCEL)** 的 [Medium 博客文章](https://medium.com/ai-artistry/mastering-chain-composition-with-langchain-expression-language-lcel-2d5041fb0cbd)，并提供了其在 *抓取维基百科 (scraping Wikipedia)* 中的应用示例。
- **基于 GPT 的品牌 Logo 分析器**：`@appstormer_25583` 展示了一个[品牌 Logo 分析器](https://beta.appstorm.ai/share?url=73dfbb7a) GPT，它可以根据上传的 Logo 图片提供设计反馈和改进建议。
- **Pagerize - AI 视频摘要工具**：`@pagerize_admin` 介绍了 [Pagerize](https://pagerize.ai/)，这是一个用于 YouTube 视频的 AI 摘要工具。其中包括了 Theory of Mind LangChain 和 Plastic Labs 网络研讨会的摘要示例，可在此处[查看](https://www.pagerize.ai/snapshot/993e8dab-b2be-4c50-a09c-407395cfd925)。
- **TinyLLM - 用于生产级应用的 LLM 库**：`@gokusan` 开发了 [TinyLLM](https://github.com/zozoheir/tinyllm/tree/main)，这是一个用于大规模运行 LLM 应用的库。创建和评估 Agent 的示例可以在[此处](https://github.com/zozoheir/tinyllm/blob/main/docs/examples/agent_example.py)查看。
- **用于检测 LLM API 密钥的 Pre-Commit Hook**：`@joshuasundance` 创建了一个 [pre-commit hook](https://github.com/joshuasundance-swca/detect_llm_api_keys)，以防止开发人员将他们的 API 密钥提交到源码控制中。


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
potatooff: https://www.youtube.com/watch?v=mrjq3lFz23s


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **LLM 辅助评分评估实验**：用户 `@_jp1_` 发起了一场关于早期 AI 初创公司的 LLM 辅助批量评估工具（用于亮点提取）的讨论。这引发了关于降低评分以及使用特殊评分 Token 来提高准确性和辨别力的辩论 [“原始帖子链接”](https://www.reddit.com/r/LocalLLaMA/comments/18id0pa/experimenting_with_llmassisted_scoring_eval/)。随后，对话扩展到了 `@bjoernp` 和 `@_jp1_` 之间关于潜在合作和理解评估方法的讨论。
- **Mixtral 实现**：*mixtral_implementation* 频道的对话强调了 Mixtral 实现的不同要素，包括最佳实践、高性能硬件的获取、专家选择策略以及基准测试结果。
    - 重点包括 `@kalomaze` 分享的一个 [llama.cpp 自定义构建版本](https://github.com/kalomaze/koboldcpp/releases/tag/custom-routing-test)，以及 `@someone13574` 分享的一条 [推文](https://twitter.com/sbeastwindy/status/1735185274475524333)，该推文建议使用 3 个专家可以获得最佳的困惑度 (perplexity) 结果。
- **OpenAI 的对齐计划与 Phi-2 讨论**：OpenAI 新论文的对齐计划和 Phi-2 的有效性是这里的讨论主题。此外，`@flozi00` 建议实施评分模型以增强 Disco Research 中的数据质量。
- **简化语言理解与建模 (Llama) 集成与 FastEval 评估进展**：对话围绕 llama.cpp 在项目中的集成、FastEval 评估的大部分完成以及与 llama-cpp-python 相关的问题展开。具体而言，记录了新的进展，例如 [Disco Research 在 GitHub 上的 FastEval 项目分支](https://github.com/DiscoResearch/FastEval)。`@bjoernp` 向 `@rtyax` 提供了关于调试更简单模型以及为本地模型使用单线程的建议。

**DiscoResearch Channel Summaries**

### ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/) (11 messages🔥): 
        
- **LLM 辅助评分评估实验**：用户 `@_jp1_` 分享了来自 Reddit 的帖子，关于一家早期 AI 初创公司使用其 LLM 辅助批量评估工具的结果。该工具用于从新闻文章中提取亮点，并使用评分模型进行 MMLU 基准测试。他们引用了三篇 [1](https://arxiv.org/abs/2305.13711), [2](https://arxiv.org/abs/2310.17631), [3](https://arxiv.org/abs/2310.08491) 不同的研究论文作为其评估方法的基础。这引发了关于减少评分类别和使用特殊评分 Token 是否会影响准确性和区分能力的讨论。[原始帖子链接](https://www.reddit.com/r/LocalLLaMA/comments/18id0pa/experimenting_with_llmassisted_scoring_eval/)
  
- **潜在合作**：用户 `@bjoernp` 对该评估方法表示感兴趣，并建议与从事类似项目的初创公司进行合作。

- **评估方法理解**：`@bjoernp` 和 `@_jp1_` 随后讨论了使用基准测试来评估模型的循环性。他们谈到，一个好的基准测试可以消除对评估模型的需求，而使用评估模型通常是因为没有简便的基准测试方法，或者为所有待测项创建基准测试的工作量太大。

- **对基准测试和评估模型的困惑**：`@_jp1_` 对评估方法表示困惑，并质疑测量评估模型的 MMLU 是否类似于直接进行模型评估。`@bjoernp` 承认了这种相似性，但指出存在细微差别，并表示对回复进行评分是一项更容易的任务。

- **其他潜在基准**：在结论中，`@bjoernp` 建议留出测试集（held-out test set）可能是一个很好的评分模型基准。


### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (29 messages🔥): 
        
- **Mixtral 实现最佳实践**：`@nul3` 询问在聊天场景下，Instruct 版本和非 Instruct 版本哪个更好。`@goldkoron` 推荐了 Instruct 版本。 
- **自定义 MoE 路由**：`@kalomaze` 分享了一个 [llama.cpp 的自定义构建版本](https://github.com/kalomaze/koboldcpp/releases/tag/custom-routing-test)，这是对 Mixtral PR 的修改，允许用户自定义每个 Token 路由到的专家数量。
- **访问高端硬件**：`@tarikoctapm` 提供了访问拥有 4 x RTX 4090 机器的机会，作为回报，希望分享模型运行的结果。
- **专家选择策略讨论**：`@chrismcmaster` 和 `@kalomaze` 就各种专家选择策略交换了意见。他们讨论了硬编码固定数量的专家、'min_p' 策略以及 top-k 专家。`@kalomaze` 详细阐述了 'min_p'，它充当基于最大概率的最小概率阈值。
- **基准测试结果**：`@bjoernp` 分享了初步的基准测试，结果显示硬编码 1 个或 4 个 top-k 专家时性能不佳。`@someone13574` 分享了一条 [推文](https://twitter.com/sbeastwindy/status/1735185274475524333)，建议使用 3 个专家可以获得最佳的 Perplexity 结果。`@kenjiqq` 对这些结果提出了批评，他观察到 Q6 量化（quant）中存在不一致性。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (5 messages): 
        
- **OpenAI 的对齐计划**：`@_jp1_` 关注了 OpenAI 新论文中关于其对齐计划（Alignment Plan）的有趣内容。未提供链接。
- **关于 Phi-2 的讨论**：`@bjoernp` 和 `@flozi00` 讨论了 **Phi-2**，质疑其有效性及基准测试的真实性。虽然 `@bjoernp` 持怀疑态度，但 `@flozi00` 分享了来自 Twitter 示例的一些乐观观察，特别是针对边缘侧（edge）的小规模使用场景。
- **Disco Research 中的数据质量评分模型**：`@flozi00` 建议实施评分模型以增强 Disco Research 的数据质量。其目标不仅是数据集去重，还要清除低质量数据，如**来自维基百科的原始列表或翻译错误**。

### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (13 messages🔥): 
        
- **关于简化语言理解与建模 (Llama) 集成的协作**：`@bjoernp` 请求 `@rtyax` 协作将 llama.cpp 整合到他们的项目中，并询问 `@rtyax` 之前提到的与 tokenizer 相关问题的细节。
- **FastEval 评估完成及后续步骤**：`@bjoernp` 概述了下一步工作，包括增加基准测试的 n 次重复能力并计算均值和标准差，从而完成了评估的大部分工作。他还指出仍需集成 llama.cpp 以检查 min_p 的效果，并启用基于语法的推理（grammar-based inference）来修复输出格式。
- **DiscoResearch GitHub 上的 FastEval 仓库**：`@bjoernp` 在 DiscoResearch GitHub 账号下创建了 FastEval 的分支用于协作，并引导用户和 `@rtyax` 在该处提交更改。仓库地址为 [https://github.com/DiscoResearch/FastEval](https://github.com/DiscoResearch/FastEval)。
- **Llama-cpp-python 的问题**：`@rtyax` 报告称 llama-cpp-python 在运行 generate/chat_completion 时出现静默失败。他们指出困难可能不在于 tokenizer，但无法确定具体问题。他们最近为 mixtral 重新构建了 llama-cpp-python，不确定该问题是本地的还是普遍存在的。
- **调试建议**：`@bjoernp` 建议先调试较简单的模型（如 mistral-7b），并将 fasteval 评估常量更改为在本地模型上使用单线程。`@rtyax` 表示他的机器只使用了一个 GPU，但由于 try/catch 块的存在，生成过程在未被调用的情况下仍在继续。

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Anthropic 发布新的 Google Sheets 工具**：Claude for Sheets™ 受到 `@swyxio` 的称赞，称“*电子表格是最好的 prompt engineering 工具*”。[Claude for Sheets™](https://workspace.google.com/marketplace/app/claude_for_sheets/909417792257)
- **Mamba 模型深度报道**：`@swyxio` 分享了一个提供 Mamba 模型详尽解释的 [YouTube 视频](https://youtu.be/ouF-H35atOY?si=ttVKMzfnhNiA_Qk1)。
- **Mistral-kit 项目介绍**：该项目使用 mistral-7b 和 ollama；`@kevmodrome` 在 GitHub 上分享了该仓库。[GitHub 链接](https://github.com/kevmodrome/mistral-kit)
- **关于 GPT 4.5 发布传闻的讨论**：由 `@swyxio` 发起，并附带了一个 [推文链接](https://fxtwitter.com/aisafetymemes/status/1735282033926996449?s=46&t=90xQ8sGy63D2OtiaoGJuww)。
- **Mistral API 访问权限**：`@fanahova` 和 `@coffeebean6887` 获得了访问权限；后者还提到 Anyscale 已将 Mistral 添加为端点。[Anyscale 链接](https://app.endpoints.anyscale.com/)
- **新播客发布公告**：`@fanahova` 在 Twitter 和 Hacker News 上发布了消息。([Twitter 链接](https://twitter.com/FanaHOVA/status/1735371425836568905))
- **提及 Qtransformers**：`@swyxio` 在 #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) 频道中提到了 Qtransformers，未提供进一步说明或背景。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (9 messages🔥): 
        
- **Anthropic 的新 Google Sheets 工具**：`@swyxio` 提到了一款名为 [Claude for Sheets™](https://workspace.google.com/marketplace/app/claude_for_sheets/909417792257) 的新产品，它可以将 Anthropic 的 AI 助手 Claude 引入 Google Sheets™。`@swyxio` 表示“*电子表格是最好的 prompt engineering 工具*”。
- **关于 Mamba 模型的讨论**：`@swyxio` 分享了一个深入解释 Mamba 模型的 [YouTube 视频](https://youtu.be/ouF-H35atOY?si=ttVKMzfnhNiA_Qk1)。
- **Mistral-kit 项目**：`@kevmodrome` 分享了 mistral-kit 项目的 [GitHub 链接](https://github.com/kevmodrome/mistral-kit)，该项目使用 mistral-7b 和 ollama。
- **关于 GPT 4.5 的传闻**：`@swyxio` 提到了一些关于 GPT 4.5 即将推出的 [传闻](https://fxtwitter.com/aisafetymemes/status/1735282033926996449?s=46&t=90xQ8sGy63D2OtiaoGJuww)。
- **Mistral API 访问权限**：`@fanahova` 和 `@coffeebean6887` 都表示他们获得了 Mistral API 的访问权限。`@coffeebean6887` 还指出 Anyscale 已将 Mistral 添加为端点，并发布了其 [URL](https://app.endpoints.anyscale.com/)。


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
fanahova: 新播客上线了！https://twitter.com/FanaHOVA/status/1735371425836568905

同时也在 HN 上发布了。


### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages): 
        
swyxio: Qtransformers

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- 在 **general** 频道中，重点讨论了两个关键点：
   - *Model 3.5 的问题*：`@0xmmo` 报告了 AI Model 3.5 的异常行为，包括在响应中产生过多的换行符。
   - ChatGPT 在 *Finals Week*（期末周）的工作负载被 `@res6969` 幽默地推测，认为由于学生使用量增加，它可能已经超负荷。
- **finetuning** 频道中，`@robertchung` 为文本提取任务提供了建议，推荐初始使用 30-50 个样本进行 fine-tuning，如果结果不理想，可以增加更多示例。
- 在 **opensource** 频道中，`@robhaisfield` 讨论了一种通过在多个 fine-tuning 提供商之间轮换来克服 rate limits 的策略，但也提到了应对每个提供商独特 fine-tunes 所产生的细微行为差异的挑战。
- **resources** 频道分享了一些有用的资源，例如 `@nosa_` 发布的一个富有洞察力的 [Twitter 链接](https://fxtwitter.com/tomas_hk/status/1734664304924721245)。
- **openai** 频道展示了两个有趣的参考资料：
   - `@pantsforbirds` 指出了一篇关于用较小模型监督较大模型的 [OpenAI 论文](https://x.com/OpenAI/status/1735349718765715913?s=20)。
   - `@firefox8975` 强调了 Google AI 在支持 function calls 方面的竞争能力，并引用了[官方文档](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)。

**LLM Perf Enthusiasts AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (3 条消息): 
        
- **Model 3.5 可能存在的问题**：`@0xmmo` 报告称他们从 AI model 3.5 获得了奇怪的响应，该模型开始在响应其 function calls 时使用过多的换行符。
- **期末周期间的 AI 工作负载**：轻松一点的话题，`@res6969` 幽默地推测 ChatGPT 可能因为“期末周”（Finals Week，指高强度学术工作时期）而超负荷。`@potrock` 对此表示支持，提到一门研究生级别的 Natural Language Processing (NLP) 课程中，许多人正在利用 GPT 生成的 synthetic data（合成数据）来完成期末项目。


### ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/) (2 条消息): 
        
- **文本提取与 Fine-Tuning**：`@robertchung` 建议对于文本提取任务，**30-50 个样本**可能足以进行初始 fine-tuning。他们还建议，如果结果不理想，可以**添加更多示例**并对已经 fine-tuned 的模型继续进行 fine-tune。


### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 条消息): 
        
- **关于在提供商之间轮换的讨论**：`@robhaisfield` 分享了关于通过程序化地在各种 fine-tune 提供商之间轮换来突破 rate limits 策略的想法。然而，他表示担心**每个提供商独特的 fine-tunes 所产生的细微行为差异**可能使得将所有提供商视为同质化资源变得棘手。


### ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/) (2 条消息): 
        
- **分享有用资源**：用户 `@nosa_` 分享了一个[有用的 Twitter 链接](https://fxtwitter.com/tomas_hk/status/1734664304924721245)，其中包含对社区可能有价值的信息。用户 `@ayenem` 对分享的链接给予了积极回应。


### ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/) (2 条消息): 
        
- **使用小模型监督大模型**：用户 `@pantsforbirds` 分享了一篇关于使用小模型 (GPT-2) 监督大模型 (GPT-4) 的 [OpenAI 论文](https://x.com/OpenAI/status/1735349718765715913?s=20)。
- **Google AI 的 Function Calling**：`@firefox8975` 提到 **Google AI** 支持 function calling，他们在探索过程中发现其具有与 OpenAI 竞争的实力。他们还提供了 Google AI [官方文档](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)的参考。


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- \[oo\] 频道重点介绍了 Upstage 在 franken llama-mistral 中使用的 **Depth Upscaling** 技术，该模型拥有 10.7B 参数，型号为 [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)。“*Upstage 声称其方法已经超越了 Mixtral。*”
- 同一频道讨论了新的 **Open Source AI Grants**（开源 AI 资助），用户们因获得 a16z [公告](https://a16z.com/announcing-our-latest-open-source-ai-grants/)中提到的资助而受到祝贺。
- \[oo-priority\] 包含来自 `@entropi` 的特定活动详情，但未提供活动的具体细节。
- 在 \[phi-tuning\] 中，`@entropi` 分享了关于 **Phi-2** Transformer 模型的更新，该模型具有 **27 亿** 参数，并提供了更多信息的 [链接](https://huggingface.co/microsoft/phi-2)。该模型的数据与 [Phi-1.5](https://huggingface.co/microsoft/phi-1.5) 相同，但增加了包含 NLP 合成文本以及经过安全和教育过滤的网站的新来源。还提到了模型权重的更新。

**Alignment Lab AI 频道摘要**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (2 条消息): 
        
- **Franken Llama-Mistral 上的 Depth Upscaling**: 用户 `@entropi` 分享了一篇文章链接，介绍了 Upstage 如何在 franken llama-mistral 上使用 “Depth Upscaling” 技术，通过持续预训练达到 10.7B 参数。Upstage 声称其方法已超越 Mixtral。最终成果是他们的模型 [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)。
- **新的 Open Source AI Grants**: `@entropi` 还祝贺用户 <@257999024458563585> 和 <@503006108928180245> 获得新的 AI 资助，并分享了来自 a16z 的 [公告](https://a16z.com/announcing-our-latest-open-source-ai-grants/)。


### ▷ #[oo-priority](https://discord.com/channels/1087862276448595968/1135848608588124211/) (1 条消息): 
        
entropi: @here https://discord.com/events/


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (2 条消息): 
        
- **Phi-2 模型摘要**: `@entropi` 分享了一个关于 **Phi-2** 的 [链接](https://huggingface.co/microsoft/phi-2)，这是一个拥有 **27 亿** 参数的 Transformer。用于训练的数据与 [Phi-1.5](https://huggingface.co/microsoft/phi-1.5) 相同。此外，Phi-2 模型还使用了新的数据源进行训练，包括各种 NLP 合成文本以及为安全和教育价值而过滤的网站。
- **Phi-2 权重更新**: `@entropi` 报告称 Phi-2 的权重最近进行了更新。


        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord 总结

只有一个频道有活动，因此无需汇总...

- **AI 经济学讨论**: `@stevekamman` 分享了关于 **AI 经济学** 的见解，特别关注 *训练* 与 *推理成本及收入* 之间的比较。他表示：“*推理必须支付账单——GPU、S&M（销售与市场）、管理费用……*”并指出，推理产生的 *收入* 必须显著高于与训练相关的 *成本*，才能保持长期盈利。
- **关于训练成本的考虑**: `@stevekamman` 指出，*训练中产生的成本可以通过出售基础模型来抵消*，但他也指出，这些模型的买家不可避免地需要通过收入或效率提升来为购买提供资金。
- **FM 公司的未来**: `@spillai` 推测 *基础模型 (FM) 公司是否需要开发自己的云服务*，以便继续从客户那里获取足够的长期价值，从而实现盈利。他们关注到大型语言模型 (LLM) 对 *垂直 AI 基础设施提供商* 的潜在需求。
- **FM 公司对 GPU 的利用**: `@stevekamman` 质疑了 FM 公司对 GPU 的利用效率，考虑到其二进制特性（100% 或 0% 利用率）。这考虑到潜在的基于规模的利用效率提升可能会受到限制。
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- 由 `@jay_wooow` 发起的开源调查，旨在了解使用 **LLMs (Language Learner Models)** 进行构建的动机和挑战。调查数据将用于内部参考，并协助更广泛的 AI 社区创建开发工具。
- 原始调查数据和关键见解将在达到目标参与人数后，以报告形式开源并在线发布。
- 分享了[调查问卷](https://tally.so/r/mO7q0p)的链接。开发者参与调查将保持匿名。
- pradeep1148 在 off-topic 频道发布了一个没有上下文的 [YouTube 视频](https://www.youtube.com/watch?v=MOimHasrCKk) 链接。由于未提供进一步的讨论或背景，其相关性尚不明确。

**Skunkworks AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (1 条消息): 
        
- **LLM 工具使用调查**：`@jay_wooow` 发起了一项开源调查，旨在了解使用 LLMs (Language Learner Models) 进行构建时的动机和挑战。通过此次调查收集的数据将用于提升内部生产力，并为广大社区提供市场数据，以优化开发工具的创建。完成调查大约需要 10 分钟。
- **结果发布**：一旦达到目标参与人数，所有收集的原始数据以及关键见解都将以报告形式开源，并发布在博客平台上。对于 AI 开发者来说，这将是了解社区目标、挑战和工具使用情况的宝贵资源。
- **参与链接**：分享了参与此[调查](https://tally.so/r/mO7q0p)的链接。所有参与开发者的身份将保持匿名。


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 条消息): 
        
pradeep1148: https://www.youtube.com/watch?v=MOimHasrCKk


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord 摘要

只有一个频道有活动，因此无需汇总...

- **掌握 MLOps 和 ML 工程：2024 年关键策略**：
    - `@amitqwak` 宣布了一场名为 *"Mastering MLOps and ML Engineering: Key Strategies for 2024"* 的直播会议，定于 2024 年 1 月 17 日美国东部时间上午 11:30 举行。该会议旨在**为组织提供先进的见解和策略，以便在其业务框架中有效地整合和管理 AI 和 ML**，重点关注 MLOps 和 ML 工程趋势。该活动主要面向 ML Engineers、Data Scientists、数据负责人和软件工程经理，免费参加。注册链接在[这里](https://www.qwak.com/academy/mlops-and-ml-engineering-key-strategies?utm_source=Chip_Hyuen&utm_medium=Discord&utm_campaign=January24_Webinar)。
- **Arize 节日特辑**：
    - `@sarahwelsh` 宣布了定于 2023 年 12 月 15 日举行的 *Arize Holiday Special*。该活动包括一系列在线虚拟动手研讨会，重点关注 **Prompt Engineering、搜索与检索工作流以及 LLM 系统评估**。来自 Hugging Face、PromptLayer、Shopify 和 Arize 的演讲者将参加。注册链接在[这里](https://arize.com/arize-holiday-special/)。
        

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

只有一个频道有活动，因此无需汇总...

- **即将举行的活动通知**：`._z` 发布了即将举行活动的通知，并提供了相关的 [Discord 链接](https://discord.gg/XGnCCSnu?event=1184893613021339659)。

        

---
Ontocord (MDEL discord) Discord 没有新消息。如果该公会长期没有动态，请告知我们，我们将将其移除。

---
Perplexity AI Discord 没有新消息。如果该公会长期没有动态，请告知我们，我们将将其移除。