---
companies:
- anyscale
- openai
- microsoft
date: '2023-12-23T01:16:52.465251Z'
description: '**Anyscale** 推出了 **LLMPerf 排行榜**，旨在对大语言模型的推理性能进行基准测试。然而，该榜单也面临一些批评，理由是其缺乏单
  token 成本和吞吐量等详细指标，且在比较公共 LLM 端点时未考虑批处理和负载情况。


  在 **OpenAI Discord** 的讨论中，用户反映了 **Bard** 的一些问题，并表示在故事创作方面更青睐 **Microsoft Copilot**，认为其幻觉（hallucinations）更少。关于从
  **GPT-3.5** 升级到 **GPT-4** 的价值也引发了辩论，许多人认为付费 AI 模型对提高编程效率非常有价值。


  此外，OpenAI API 的漏洞和性能问题也备受关注，包括响应缓慢和消息限制。讨论还涉及了 **GPT-6** 等未来 AI 发展，以及对 OpenAI 透明度和盈利能力的担忧。图像生成的提示工程（Prompt
  Engineering）是另一个热门话题，重点在于强调清晰的正向提示以及对负面提示（negative prompts）的需求。'
id: f57c70e2-cab0-4c52-b940-e3d98d2def04
models:
- gpt-4
- gpt-3.5
- bard
original_slug: ainews-12222023-anyscales-benchmark-criticisms
people: []
title: 2023年12月22日：Anyscale 对基准测试的批评
topics:
- benchmarking
- performance
- api
- prompt-engineering
- bug-tracking
- model-comparison
- productivity
- programming-languages
- storytelling
---

 

批评意见：

- 需要公开 [每个 token 的成本、吞吐量以及窗口化版本，而不仅仅是突发性能 (burst)](https://twitter.com/soumithchintala/status/1738241213327692174)
- [不要直接比较公共 LLM 端点](https://twitter.com/dzhulgakov/status/1737917306565697990?s=46)，因为 Batching、负载和时序非常重要
- [直接遭到了 Together 的 Vipul 的质疑](https://twitter.com/vipulved/status/1738075362448527805?s=46)
- 在 ML 历史上，针对单一指标进行 [过度优化通常是行不通的](https://x.com/jiayq/status/1738014510336909397?s=20)。


[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **AI 模型的性能与可用性**：在 [#ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) 频道下关于不同 AI 模型用户体验的讨论。有报告提到 **Bard** 的问题，并表达了对 **MSFT Copilot** 的偏好，特别是在故事创作任务中。*“`@muyfashionista` 分享了他们对多个 AI 模型的测试，并提到了 **Bard** 的问题”*。此外，还澄清了 Bing Chat 与 Microsoft Copilot 之间的区别。
- **GPT-4 及其预期价值**：关于升级到 **GPT-4** 是否值得以及 AI 模型定价是否合理的疑问。根据讨论，社区中许多人认为 ChatGPT 和 Copilot 等模型带来的生产力提升对于编程工作来说物有所值。
- **AI 相关的挑战与 Bug**：在各个频道中，讨论了一系列问题和潜在的 Bug。在 api-discussions 中，记录了 GPT 和 OpenAI API 的性能问题。用户 `@bmaxhacks` 注意到 API 响应时间超过 2.5 秒。还指出了模型对自定义格式响应不佳、GPTs 消失以及一个可能允许 GPT-4 每小时发送 80 条消息的潜在 Bug 等问题。交流中还探讨了 AI 模型在不同编程语言下的性能变化。
- **未来 AI 展望与讨论**：在 #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) 频道，用户推测了未来的 AI 发展，例如可能发布的 **GPT-6** 及其对经济的潜在影响。此外，还表达了对 OpenAI 透明度、盈利能力和服务投资的担忧。
- **用于图像生成的 Prompt Engineering**：在 #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) 频道中，围绕引导 AI 的视觉输出和控制生成图像中出现的内容进行了对话。用户 `@eskcanta` 分享了关于最大化输出质量的建议，强调了清晰、详细的 Prompt 的价值，并建议避免使用负面指令。用户 `@suptrox` 则希望通过 Negative Prompts 来引导 AI 模型。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (50 messages🔥): 
        
- **AI 模型的性能与可用性**：`@muyfashionista` 分享了他们对多个 AI 模型的测试，并提到了 **Bard** 的问题，包括**出错**、无限循环或不返回响应。他们发现 **MSFT Copilot** 在故事创作方面是更好的选择，且与 Bard 相比幻觉更少。 
- **Microsoft Bing 与 Microsoft Copilot 的区别**：针对 `@eljajasoriginal` 的疑问，`@muyfashionista` 澄清说 **Microsoft 已将 Bing Chat 更名为 Microsoft Copilot**。他们还指出 Microsoft 365 版的 MSFT Copilot 与 Bing Chat 中的版本存在差异。
- **从 GPT 3.5 升级到 GPT 4**：针对 `@6nueve` 的咨询，`@elektronisade` 和 `@lugui` 建议 **升级到 GPT 4 很有用**，特别是对于编程等任务。然而，他们提醒说，AI 模型的生产力也很大程度上取决于所使用的语言、项目类型以及 Prompt 的质量。
- **付费 AI 模型的工作价值**：`@6nueve` 询问为 AI 模型付费是否值得，这引发了与 `@lugui`、`@afterst0rm` 等人的讨论。普遍观点是，**使用 ChatGPT 和 Copilot 等 AI 模型带来的生产力提升足以证明其成本的合理性**，前提是用户从事编程工作。他们发现这提高了生产力并让工作更有趣。
- **AI 模型在不同编程语言下的表现**：`@afterst0rm` 和 `@lugui` 讨论了 **AI 模型在不同编程语言下的表现差异**。他们发现模型在 Python、Java 和 JS 等主流语言上表现更好，而像 Rust 这样较新的语言结果则不尽如人意。AI 模型甚至可以生成 Python 的样板代码，但在处理较新的语法和模式时会感到吃力。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (165 messages🔥🔥): 
        
- **关于 GPT-V 性能的讨论**：用户 `@pudochu` 对 **GPT-V** 表达了相当大的挫败感，特别是其速度和视觉能力，断言其在当前状态下不符合使用目的。然而，用户 `@captainsonic` 建议，成功使用 GPT-V 可能需要更多的 Prompt Engineering 以及针对特定用例的定制。 
- **GPT-4-Turbo 的使用**：围绕 GPT-4-Turbo 的性能和成本效益进行了多次讨论。`@beowulfbr` 和 `@feltsteam0` 等用户分析了 GPT-4-Turbo 及其替代方案的每个 Token 成本，指出价格取决于用例和处理的 Token 数量。 
- **对 OpenAI 透明度和利润利用的担忧**：`@mischievouscow` 和 `@feltsteam0` 就 OpenAI 可能如何使用其利润进行了反复对话。有人猜测其投资于 NVIDIA GPU，并断言训练和推理成本将占据收入的很大一部分。
- **对未来 AI 发展的推测**：对话包括对 **GPT-6** 何时可用的推测，并幽默地提到了期待已久的 GTA 6 发布。还有关于 AI 进步带来的潜在经济产出的讨论，`@samaltman` 建议 AI 可能会将经济产出提高 20-30 倍。
- **平台与功能问题**：几位用户报告了 OpenAI 平台的问题。`@kingofcringe` 报告缺少停止生成按钮，而 `@cozy_artist` 在生成下载文件时遇到问题。然而，`@jaicraft` 发现了一个功能，可以选中回复中的文本并进行回复。

### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (57 条消息🔥🔥): 
        
- **GPT 与自定义格式的问题**：用户 `@ritalin60mg` 提出了关于 GPT-3.5 在要求以自定义格式回复时表现不如 GPT-4 的担忧。他们正在寻求一种标准的指令方式，以尽量减少字符使用。
- **管理线程与删除**：`@yummisery` 不清楚线程删除的机制，以及它们是否在不活动后自动删除。针对此问题，`@brokearsebillionaire` 澄清说，虽然 runs 会超时，但 threads 不会，清理线程是用户的责任。
- **ChatGPT 行为与故障排除**：`@princesslunarcat` 报告了 ChatGPT 的问题，即在发送一条消息后从 GPT-4 切换到了 GPT-3.5。`@solbus` 提供了各种故障排除建议，但问题仍然存在。该问题随后已报告给 OpenAI 支持部门。
- **OpenAI API 的性能问题**：`@bmaxhacks` 提出了一个关于 OpenAI API 的担忧，即尽管发送的数据量很小，但大多数响应时间仍超过 2.5 秒。
- **外部 API 的整合与迭代问题**：`@yummisery` 询问 Assistants 的 Code Interpreter 工具是否可以调用外部 API 端点。他们进一步询问了在函数上下文中如何处理涉及多次 API 调用的迭代。
- **用户输入的重复控制**：`@solononforever` 寻求关于跟踪对话历史的伪代码建议，以通过 Prompt 或 LangChain 防止用户重复，对此 `@snowmobile` 提供了一个 Python 代码建议。 
- **LangChain 中的 Function Calling**：`@bluehipp0` 询问为什么没有提供 Function Calling 功能的开源 LLM，并寻求使用 LangChain 模拟 Function Calling 的示例。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (50 条消息🔥): 
        
- **GPTs 消失的故障**：有大量关于 GPTs 和聊天记录消失的报告，通常稍后会恢复。`@csharp69`、`@redash999` 等人遇到了这个问题。具体来说，`@redash999` 提到一个最近创建的名为 "Water Engineer" 的 GPT 没有再次出现。

- **对 GPT-4.5 Turbo 定价的困惑**：`@killymbapps` 对 GPT Assistants 的 GPT-4.5 Turbo 定价方式表示困惑，指出缺乏明确的信息。

- **股票数据分析 GPT**：`@emperor_azir` 分享了一个提供实时金融数据和技术分析的 GPT [链接](https://chat.openai.com/g/g-jC8FvZ9SW-stock-data-analysis-live-financial-data)。

- **潜在 Bug**：`@raidedcluster` 发现了一个潜在的 Bug，允许他们每小时使用 GPT-4 发送 80 条消息，突破了限制。

- **GPTs 的潜在艺术应用**：`@fortun8te`、`@jamiecropley` 等人讨论了 GPTs 根据图像输入分析和理解艺术风格的潜力。他们希望未来在这一领域有所改进。`@elektronisade` 表示目前的 GPTs 无法掌握粗略分类之外的风格。

- **受限的 GPT 名称**：`@wndapnda02` 报告了一个问题，即具有已验证域名的 GPTs 被限制公开分享，暗示可能存在商标侵权。 

- **知识库分析的问题**：`@bacaxnot` 抱怨自定义 GPTs 现在优先使用 Code Interpreter 而非 Retrieval 来分析其知识库，导致响应速度变慢且准确度下降。

- **ChatGPT 传闻**：`@chemo2493` 表示 GPT-4.5 Turbo 已被证实为谣言。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (7 条消息): 
        
- **创建特定的 AI 可视化**：用户 `@suptrox` 希望通过 Prompt 让 DALL-E 风格的模型仅可视化一个广阔的花园，**别无他物**。他们提到在处理图像中出现的不必要元素时遇到困难，并表达了对使用“负向提示词”（negative prompts）来引导模型的渴望。
- **引导 AI 视觉输出**：针对 `@suptrox`，`@eskcanta` 分享了一种引导 AI 视觉输出的方法。他们强调应专注于对图像中应出现内容的正面描述，并避免负面指令。他们还重申了精确且详细的 Prompt 的重要性。
- **图像创建示例**：`@eskcanta` 提供了一个具体示例，使用 Prompt 创建了四张细节丰富、纯净且无尽的花园图像。这获得了其他用户的积极反应。
- **对方法的反馈**：`@catking335` 对生成的图像反应积极，并询问 `@eskcanta` 关于创作过程的问题。`@eskcanta` 重申了清晰、详细且正面的 Prompt 在引导 ChatGPT-4 方面的价值。
- **对技术的赞赏**：`@bambooshoots` 也对 `@eskcanta` 展示的技术表示赞赏，指出这激发了一些想法。他们对进一步探索该技术表示兴奋和期待。

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (7 条消息): 
        
- **使用 DALLE 进行图像生成**：用户 `@suptrox` 想要生成一张仅展示花园景观的图像，并询问 DALLE 是否支持负向提示词 (negative prompts) 来抑制不需要的元素。
- `@eskcanta` 回复建议 AI 通常在处理正向指令而非负向指令时表现更好。他们通过使用 ChatGPT-4 模型生成了一系列代表“无尽自然、未受破坏、原始花园”的四张图像来证明这一点。他们对于获得理想输出的建议是：准确描述你想要的内容，检查输出，并根据观察到的差异迭代优化指令。
- 几位用户（如 `@catking335` 和 `@bambooshoots`）对生成的图像表示赞赏。`@catking335` 询问了创作过程，对此 `@eskcanta` 详细说明了他们使用的提示词，并推荐了增强模型输出的技术，例如与模型进行更详细的对话以微调所需的输出。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- 深入讨论了 AI **模型竞争**，正如 `@red_code` 所建议的，焦点正转向小型多模态模型、小语言模型 (SLM)、分布式学习模型和廉价微调模型。
- 广泛交流了关于 Mistral AI 的**开源前端**（类似于 OpenAI 的 playground）、在**各种模型中使用 litellm**、如何衡量 **Mistral 输出中的 token 计数**，以及使用独特数据集训练 Mistral。分享了 litellm [config.yaml](https://github.com/brokearsebillionaire/llmcord/commit/76114cc67cd70c35caba7081c0f2fa61bf0213e8) 的链接和关于 [Autogen](https://github.com/microsoft/autogen/issues/1037) 的 vLLM 问题。用户 `@rrross` 还跟进了一篇关于如何计算 Mistral token 的[博客文章](https://bit.ly/count-mistral-tokens)。
- 涉及模型能力、比较和用例的详细主题。包括选择**适合英语到挪威语翻译的模型**、**Mistral 在编程和检索 Linux 命令领域**的应用，以及 **7B 模型在 3090 GPU 上的性能**。
- 展示了 Mistral AI 的多功能性，示例包括在**自有服务器上托管 Mistral 模型**、一个与 Mistral API 兼容的机器人 [**llmcord**](https://github.com/jakobdylanc/llmcord)、一个使用 Mistral.AI Web API 的新第三方包（见[此处](https://raku.land/zef:antononcube/WWW::MistralAI)），以及分享的一篇 [Twitter 帖子](https://x.com/gneubig/status/1738338237498921251?s=46)。
- 交流了关于 **r/MistralAI subreddit** 的使用、在 **Android 上使用 Mistral** 的可能性、在**手机上运行 AI 模型**的杂项讨论，并分享了一个关于在 iPhone 上运行模型的 [GitHub 链接](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)。
- 深入探讨了用户在 **Mistral 平台**面临的问题，包括 `mistral-small` 响应不稳定、**印度**用户的支付问题、关于是否存在**漏洞赏金计划 (bug bounty program)** 的查询、**Embeddings 端点请求限制**缺乏明确性，以及如何获得 **API 访问邀请**。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (70 条消息🔥🔥): 
        
- **模型未来的竞争**：用户 `@red_code` 预测，不久后的竞争将集中在小型多模态模型或小语言模型、分布式学习模型以及廉价微调模型上。
- **开源前端**：用户 `@dv8s` 询问是否有针对 Mistral AI 的开源或公开可用的前端，类似于 OpenAI 的 playground。
- **在模型中使用 Litellm**：`@cyb3rward0g` 和 `@brokearsebillionaire` 讨论了在各种模型中使用 litellm，特别是它如何与经过指令微调 (instruct fine-tuned) 的 Mistral 模型交互。他们分享了 litellm [config.yaml](https://github.com/brokearsebillionaire/llmcord/commit/76114cc67cd70c35caba7081c0f2fa61bf0213e8) 的链接和 [Autogen](https://github.com/microsoft/autogen/issues/1037) 的 vLLM 问题。
- **Mistral 输出中的 Token 计数**：`@rrross` 询问如何测量 Mistral 输出中的 token 数量。`@sublimatorniq` 建议使用类似于 OpenAI 的库。`@rrross` 随后分享了一篇关于如何实现这一点的[博客文章](https://bit.ly/count-mistral-tokens)。`@daain` 还推荐了一个在线分词器 (tokenizer) 工具。
- **使用独特数据集训练 Mistral**：`@novemberfalls` 提到有兴趣使用独特的数据集训练 Mistral 模型。`@antononcube` 要求澄清数据集结构和预期用途，以便根据数据集属性提供潜在建议。

### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (78 messages🔥🔥): 
        
- **GPU 对大型模型的支持**：`@novemberfalls` 询问 3090 GPU 是否支持 7B 模型，`@mrobino` 确认在进行量化（Quantized）的情况下是可以支持的。
- **关于模型能力的查询**：`@christiansieb` 要求在文档中提供更多示例，因为他们发现 **Mistral** 和 **Mixtral** 在上下文相关性方面与 **ChatGPT** 相比存在问题。
- **关于翻译模型的讨论**：`@ved_ikke` 征求用于英语到挪威语翻译工作的模型建议。用户们讨论了 **gpt3.5**、**Helsinki NLP**、**Mixtral**、**Yi** 和 **GPT4** 等多种模型，并分享了各自的使用经验和偏好。`@laoji_lcg` 对使用 **GPT3.5** 将英语翻译成中文的效果表示满意。
- **Mistral 用于编程**：包括 `@ved_ikke` 和 `@laoji_lcg` 在内的几位用户讨论了 **Mixtral** 在代码领域的实用性。他们一致认为它在理解和调试代码方面表现出色，但 `@laoji_lcg` 认为与 **GPT4** 的精简代码相比，其输出往往过于复杂。
- **用于 Linux 命令检索的聊天机器人**：`@giblaz` 征求用于检索 Linux 命令的模型建议。`@dutchellie` 建议尝试 **Mixtral**。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (106 messages🔥🔥): 
        
- **关于不同模型在不同硬件上性能的讨论**：`@dutchellie` 和 `@sublimatorniq` 讨论了内置于 *exllamav2* 模型中的 `llama.cpp` 后端性能，并指出它带来了显著的性能提升。Dutchellie 分享了在 AMD GPU 上使用 exllamav2 的经验，其速度从 llama.cpp 的 6t/s 提升到了 50t/s（[来源](https://github.com/turboderp/exllamav2)）。
- **Mac 和 AMD 硬件的困难**：他们注意到热门项目普遍缺乏对 Mac 的支持，并提到使用这些系统的用户总是在追赶技术更新，尽管他们也承认 Mac 在运行大型模型时表现非常出色。
- **关于 `exllama` 和 `ollama` 的讨论**：讨论倾向于 `exllama` 和 `ollama` 的对比，dutchellie 分享称在个人测试中 `exllama` 的表现似乎优于 `ollama`。`@sublimatorniq` 提出了一个已知的“`mixtral` 在 mac 上”的问题，该问题因长时间延迟和缓慢而被多次报告（[来源](https://github.com/jmorganca/ollama/issues/1556)）。
- **LLM 模型与微调**：还提到了 Eric Hartford 最近发布的 *Dolphin 2.6 Mixtral*，并戏称在新版本中加入了“基于 samantha 的共情数据”。
- **Huggingface 的下载速度问题**：最后，他们对 Huggingface 的下载速度表示不满，sublimatorniq 特别提到从柬埔寨连接 Huggingface 非常困难。


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (9 messages🔥): 
        
- **在自有服务器上托管 Mistral 模型**：`@ds_gamer` 报告称，他们已在自己的服务器上托管了 Mistral 模型以及其他开源 AI 模型，并免费提供 OpenAI 兼容的 API 访问。不过，他们提到由于计算成本高昂，Mixtral 模型目前仅限付费服务。

- **开源机器人 llmcord 介绍**：`@jakobdylanc` 分享了一个名为 [**llmcord**](https://github.com/jakobdylanc/llmcord) 的开源机器人，它兼容 Mistral API，并允许与大型语言模型（LLM）进行多人聊天。

- **在 llmcord 中使用 gpt-4-vision-preview**：`@jakobdylanc` 向 `@antononcube` 确认，gpt-4-vision-preview 可以与 llmcord 机器人配合使用。该机器人同时支持 OpenAI 和 Mistral API。

- **使用 Mistral AI Web API 的第三方包**：`@antononcube` 分享了一个使用 Mistral.AI Web API 的新第三方包链接，可以通过[此处](https://raku.land/zef:antononcube/WWW::MistralAI)访问。

- **推文链接**：`@mrobino` 分享了一个推文[链接](https://x.com/gneubig/status/1738338237498921251?s=46)，未提供额外背景信息。

### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (12 条消息🔥): 
        
- **r/MistralAI subreddit 的使用情况**：用户 `@anonboxis` 询问频道中是否有人使用 **r/MistralAI subreddit**。
- **在 Android 上使用 Mistral**：用户 `@scottsilverstein` 询问是否可以在 **Android 手机**上使用 **Mistral**。`@akshay_1` 回复称目前还无法在本地运行。
- **在手机上运行 AI 模型**：`@sublimatorniq` 分享了一个 [GitHub 链接](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.swiftui)，关于在 iPhone 上运行某些模型。此外，`@rtyax` 提到 **Mistral 可以在拥有 12GB 内存的手机上运行**，并指出 **koboldcpp 和 llama.cpp 可以在 Termux 中工作**。然而，他们也指出在 **8GB 内存的手机上运行会很慢**。
- **自生成认知架构论文**：用户 `@poltronsuperstar` 分享了他们起草的一篇关于**自生成认知架构 (autopoietic cognitive architecture)** 的论文，并幽默地提到其缩写接近 "CACA"（在法语中意为“便便”）。他们指出这是对 **AGI** 的一次尝试。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (42 条消息🔥): 
        
- **Mistral Endpoint 响应问题**：用户 `@sublimatorniq` 提出了一个关于收到被截断且过短响应的问题，有时会意外停止，使用的是 `mistral-small` 模型，即使设置了 `max_tokens: 1024`。报告的 `finish_reason` 是 'stop'。无论上下文长度如何，此问题都会发生，并且似乎在模型响应初期经常出现。`@lerela` 承认了该问题并承诺进行调查。
- **来自印度的支付问题**：由于对定期扣款的限制，`@tafferboy` 在印度支付 Mistral 费用时遇到支付方式失败。建议将手动支付或钱包充值功能作为潜在的变通方案。`@lerela` 承认了该问题并表示正在寻找方案。
- **Bug Bounty 计划查询**：`@poltronsuperstar` 询问是否存在 Bug Bounty 计划，以及在平台上发现漏洞是否有助于获得开发人员职位。对此查询没有明确回复。
- **Embeddings Endpoint 请求限制**：`@lautaro.p` 询问了 Embeddings endpoint 的请求限制，但未收到回复。
- **API 访问**：`@aurelien6964` 询问如何获得 API 邀请，但未收到回复。


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- `@nonameusr` 在 Hugging Face 上介绍了 **LUNA-SOLARkrautLM-Instruct**：[Hugging Face 链接](https://huggingface.co/fblgit/LUNA-SOLARkrautLM-Instruct)。
- `.beowulfbr` 分享了一篇关于使用 LLM 进行代码翻译的学术论文：[arXiv 论文](https://arxiv.org/abs/2308.03109)。
- 训练 Mixtral 和其他模型变体是对话的焦点，`@wesh3104` 和 `@ldj` 讨论了 **OpenHermes-Mixtral-8x7B** 的微调。
- `@fullstack6209` 发起了关于 7b-13b、4-8bit 范围内性能最佳模型的讨论。`@teknium` 建议使用 Hermes 或 Openchat 模型。

公会内分享并讨论了一些有趣的链接： 

- `@asada.shinon` 提到了对 Opera 的担忧，理由是用户数据出售和其他可疑行为，并分享了一份详细报告：[Rentry 报告](https://rentry.org/operagx)。
- `@emrgnt_cmplxty` 分享了一个名为 AgentSearch 项目的 Twitter 链接，旨在增强 LLM Agent 的知识获取能力：[Twitter 上的项目](https://twitter.com/ocolegro/status/1737899295573991452)。
- 关于使用各种模型的成本影响的讨论，`.beowulfbr` 讨论了使用 `gpt-4-turbo` 处理 1,000,000 个 token 的费用。
- `@night_w0lf` 对 Gemini Pro/Bard 在 Python 编程中表现的反馈，承认 Gemini Pro 的表现优于 GPT4，并提到了即将推出的 ChatbotUI 版本。
- 讨论了一些模型的创意命名，特别是 Meow 和 Sauerkraut SOLAR，由 `@fullstack6209`、`@gabriel_syme` 和 `@beowulfbr` 发起。


**Nous Research AI 频道总结**

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (9 条消息🔥): 
        
- `@nonameusr` 分享了来自 Charlie B. Holtz 的推文：[Twitter 链接](https://vxtwitter.com/charliebholtz/status/1737667912784134344)
- `@asada.shinon` 对 Opera (GX/One/Crypto Browser) 的做法表示担忧，指出该公司以出售用户数据、审查制度、带有后门的软件以及参与掠夺性贷款和反竞争行为而闻名。分享了联系方式及更多详情链接：[@0VERIMAGINE Twitter](https://twitter.com/0VERIMAGINE) 和 [Rentry 报告](https://rentry.org/operagx)
- `@metaldragon01` 分享了来自 _akhaliq 的推文：[Twitter 链接](https://fxtwitter.com/_akhaliq/status/1738050817100325354)
- `@.beowulfbr` 讨论了 `gpt-4-turbo` 与 CursorAI 和 ChatGPT 等其他模型之间的成本差异。值得注意的是，他们发现 `gpt-4-turbo` 在处理 1,000,000 tokens 时更贵。
- `@nonameusr` 介绍了 **LUNA-SOLARkrautLM-Instruct** —— 这是强大的 Upstage 的一个 UNA-Sauerkraut 变体，通过 Hugging Face 链接分享：[Hugging Face 链接](https://huggingface.co/fblgit/LUNA-SOLARkrautLM-Instruct)
- `@night_w0lf` 反馈了 Gemini Pro/Bard 在编程（特别是 Python）方面的表现。他们对 Gemini Pro 优于 GPT4 的表现表示满意，并分享了关于即将推出的支持多个 API 提供商和通过 Ollama 支持本地模型的 ChatbotUI 版本的消息。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (133 条消息🔥🔥): 
        
- **关于训练 Mixtral 和其他模型的讨论**：关于 Mixtral 和不同模型的训练进行了大量讨论。`@wesh3104` 提到他们更倾向于使用 Medium 作为其 LLM（编程助手）付费使用的替代方案。 
- **OpenHermes-Mixtral-8x7B 微调**：`@wesh3104` 和 `@ldj` 讨论了发布在 Hugging Face 上的 OpenHermes-Mixtral-8x7B 微调。`@ldj` 澄清说可以在单个节点上进行训练。
- **LLMs 的试金石测试**：`@n8programs` 详细介绍了一个句子生成的试金石测试，这显然对除了少数高级语言模型之外的所有模型都具有挑战性。他们与 `@nonameusr` 就此话题进行了讨论。
- **使用 LLMs 进行代码翻译**：`.beowulfbr` 分享了一篇[有趣的论文](https://arxiv.org/abs/2308.03109)，探讨了大型语言模型 (LLMs) 在将代码从一种编程语言翻译成另一种编程语言时的表现。
- **AgentSearch，一个开源核心项目**：`@emrgnt_cmplxty` 分享了一个名为 AgentSearch 的 [Twitter 项目链接](https://twitter.com/ocolegro/status/1737899295573991452)，该项目致力于让 LLM agents 能够获取人类的知识。多位用户对此提供了反馈，讨论了其优点和改进空间。


### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (6 条消息): 
        
- **关于 7b-13b 范围内最佳模型的讨论**：用户 `@fullstack6209` 询问了 7b-13b、4-8bit 范围内表现最好的模型。`@teknium` 建议使用 **Hermes** 或 **Openchat** 模型，或者它们的任何合并版本。
- **模型命名法**：用户 `@fullstack6209`、`@gabriel_syme` 和 `@beowulfbr` 讨论了模型古怪的名称，特别是 **Meow** 和 **Sauerkraut SOLAR**。
- **Openchat 性能验证**：用户 `@beowulfbr` 确认了 **Openchat** 模型的良好性能。


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- **HuggingFace 模型问题与应用**：社区讨论了使用 HuggingFace 时出现的各种连接中断问题、`T5ForConditionalGeneration` 和 `Seq2SeqTrainer` 在各种任务中的潜在用法、利用 HuggingFace 资源进行学习，以及通过 Git Bash 上传模型时遇到的问题。用户分享了在低配置 PC 上运行 LLM 的建议、使用 AutoTrain 和 Seq2Seq 生成多选题，以及 GPT-4 与微调开源模型之间的成本对比。此外，还提出了关于 HuggingFace 数据存储方式的疑问。
- **工具与资源**：分享了编程聊天机器人的链接，如 [Bing Chat](https://github.com/janhq/jan)、[Huggingface NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)、近期项目 [Tweak Mpl Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat) 和 [Cheshire-Cat](https://github.com/cheshire-cat-ai)、实时编程竞赛 [Fall Challenge 2023](https://www.codingame.com/contests/fall-challenge-2023)，以及 IEC 对 [Virtual Sensors 的解释](https://www.iec.ch/blog/untapped-potential-virtual-sensors)。
- **用户创建的项目**：包括一个用于生成 TOEIC 阅读第五部分题目的模型、一个使用 Unity ML-Agents Library 训练的玩 **Huggy** 的 **ppo** Agent、一个名为 **Mixtral-8x7B-Instruct-Demo** 的快速模型、一篇关于将音乐生成为 MIDI 的博客文章、一个关于回归模型评估指标的教程，以及一段在 [YouTube](https://youtu.be/RCtnCMVYsKw) 上记录的关于 Ensemble Learning 的演讲。还提到了一个新的测试模型，但没有更多细节。
- **计算机视觉与 NLP 应用讨论**：辩论内容包括使用 diffusion models 分离叠加图像的可能性，以及将图像转换为 3D 对象的模型。有人询问了如何一次性结合 prompt tuning 和 LORA。讨论了将 LayoutLM 与 RVL-CDIP 数据集结合使用的情况，以及将 LLAMA-2 应用于高度比较任务的挑战。用户还寻求在 Google Colab 中将 GPT 模型与 Gradio UI 集成的帮助。


**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (37 条消息🔥): 
        
- **HuggingFace 模型与 Dropbox 的问题**：`@gokiburigeemu` 询问 HuggingFace 是否使用 Dropbox 存储数据以及是否存在连接限制。他们提到了 HuggingFace 模型连接过多的问题。
- **mistral orca 7b 中用于问答生成的重新生成命令**：`@enka55` 正在寻求一种在模型停止后继续生成问答的方法，尽管已将 `max_new_tokens` 设置为 16K。他们概述了 ooba textgen 的类似情况，即模型生成几个问答后就会停止，除非手动继续生成。
- **修改 T5ForConditionalGeneration 和 Seq2SeqTrainer 进行训练**：`@ianbestfellow` 正在寻求关于如何修改 `T5ForConditionalGeneration` 的 forward 部分以用于训练和 Seq2SeqTrainer 的指导。
- **量化及上传模型至 Hugging Face 的问题**：`@sumit.sumit` 在 Colab 中尝试量化模型并上传至 Hugging Face 时遇到 AttributeError，涉及 `facebook/opt-350m` 模型和 `vilsonrodrigues/falcon-7b-instruct-sharded` 模型。
- **在低端 PC 上运行 LLM 模型**：`@melvjn_03` 思考 LLM 模型是否可以在低端 PC 上运行，`@lawls.net` 澄清是可以的，特别是使用量化的 GGUF 文件。他们还提到可以在本地游戏 PC 上运行与 ChatGPT 3.5-turbo 相当的模型。
- **使用 AutoTrain 和 Seq2Seq 创建多选题**：`@robolicious` 正在考虑使用 AutoTrain 和 Seq2Seq 来处理一个特定用例，即根据某些示例和特定难度级别创建多选题。`@doctorpangloss` 建议在选择 GPT-4 微调或训练之前先尝试 30-shot 生成。
- **Bing Chat 和 Jan 作为编程聊天机器人**：`@melvjn_03` 表达了对尝试编程聊天机器人的兴趣，`@lawls.net` 推荐了 Bing Chat 并分享了 [Jan 的链接](https://github.com/janhq/jan)，这是一个离线运行的 ChatGPT 开源替代方案。
- **GPT-4 与微调成本对比**：`@robolicious` 请求提供比较使用 GPT-4 与微调开源模型成本的资源。

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (9 messages🔥): 
        
- **Huggingface NLP 课程**：`@llmsherpa` 分享了 Huggingface NLP 课程的链接，点击[此处](https://huggingface.co/learn/nlp-course/chapter1/1)查看。
- **Amazon-Reviews-Multi 数据集问题**：`@vatsal2161` 指出了 Huggingface NLP 课程中 *amazon-reviews-multi* 数据集的一个问题，称该数据集目前已失效。
- **NLP 新主题的学习方法**：`@regi6135` 寻求关于如何学习近期涌现的 NLP 主题的指导，请求提供相关线索或链接以加深理解。
- **Neuralink 的进度更新**：`@neuralink` 分享了他们的学习进度更新，提到了他们在 **DoReMi**、3D Parallelism 中的 **end-to-end FP8** 训练以及其他相关课题上的工作。
- **通过 Git Bash 上传至 Huggingface 的问题**：`@deadsg` 报告了从 Git Bash 上传到 Huggingface 时遇到的问题并寻求帮助。
- **关于笔记格式的讨论**：`@onceabeginner` 提到受到了 `@neuralink` 笔记格式的启发，并讨论了希望对特定主题有更深入理解的意愿。


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (6 messages): 
        
- **Panel Tweak Chat**：用户 `@andysingal` 分享了一个名为 [Tweak Mpl Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat) 的项目链接，并提到它是从 [ahuang11/panel-fleet](/spaces/ahuang11/panel-fleet) 复制而来的。
- **现代 NLP 仓库**：`@andysingal` 还提到将 [Panel Tweak Chat](https://huggingface.co/spaces/ahuang11/tweak-mpl-chat) 项目添加到了 [GitHub](https://github.com/andysingal/modern_nlp_2/blob/main/awesome-repos.md) 上的现代 NLP 仓库列表中。
- **Cheshire-Cat**：`@nickprock` 分享了一个名为 [Cheshire-Cat](https://github.com/cheshire-cat-ai) 的项目，这是一个用于开发 AI Assistant 的框架，支持 Hugging Face 模型。
- **2023 秋季挑战赛**：`@lustforserotonin` 发布了 CodinGame 上名为 [Fall Challenge 2023](https://www.codingame.com/contests/fall-challenge-2023) 的实时竞赛链接。 
- **虚拟传感器**：`@grojas123` 介绍了虚拟传感器（Virtual Sensors）的概念，这是一种基于 Machine Learning 的技术。分享了来自 [IEC](https://www.iec.ch/blog/untapped-potential-virtual-sensors) 的链接，其中提供了详细解释。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (6 messages): 
        
- **玩 Huggy 的 PPO Agent**：`@cloudhu` 分享了他们训练的用于玩 **Huggy** 的 **PPO** Agent 模型。该 Agent 是使用 [Unity ML-Agents Library](https://github.com/Unity-Technologies/ml-agents) 训练的。
- **极速 Mixtral-8x7B-Instruct-Demo**：`@myg5702` 分享了一个 [Hugging Face Space](https://huggingface.co/spaces/FumesAI/Mixtral-8x7B-Instruct-Demo-at-Lightning-Speed)，展示了一个运行速度极快的名为 **Mixtral-8x7B-Instruct-Demo** 的模型。
- **将音乐生成为 MIDI**：`@alexsushidog` 发布了一篇关于通过在 **JAX** 中从零开始训练 Transformer 来将音乐生成为 MIDI 的[博客](https://afmck.in/posts/2023-12-22-tchaikovsky/)。
- **回归模型的评估指标**：`@torres8552` 创建了一个 Kaggle Notebook，解释了回归任务中常用评估指标背后的数学原理。该 Notebook 深入探讨了如何解释这些指标，以及如何使用 Python 创建自定义函数来计算它们。[Notebook 已在 Kaggle 上发布](https://www.kaggle.com/code/lusfernandotorres/evaluation-metrics-for-regression-models)。
- **集成学习演讲**：`@johko990` 在 Python Pizza 会议上发表了关于 Ensemble Learning 的演讲，可通过 [YouTube](https://youtu.be/RCtnCMVYsKw) 观看。该演讲在 Ensemble Learning 的背景下重新诠释了白雪公主的童话故事。
- **新测试模型**：`@bread browser` 提到创建了一个新的测试模型，但未提供更多细节或链接。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- **图像叠加分离**：`@cachalote123` 询问是否可以使用 **Diffusion Models** 或任何其他技术来分离老电影中两张叠加的图像。

- **图像转 3D 物体**：`@drishya1` 在成功使用 **ControlNet** 将草图转换为图像后，正在寻求关于选择哪种模型将图像转换为 3D 物体的建议。

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 条消息): 
        
- **在 Google Colab 中将 GPT 模型与 Gradio UI 集成**：用户 `@designfailure` 正在寻求帮助，以便在 Google Colab 环境中使用 Gradio UI 编写两个 GPT 使用案例。所需功能包括：
  1. **GPT-4 Vision 或 LlaVA 捕捉图像**，以响应提示查询。
  2. 一个 **GPT 聊天机器人，用于解析和检索图像说明 (captions)**，并利用它们来完成聊天回复。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (17 条消息🔥): 
        
- **运行 Prompt Tuning 和 LORA**：`@opencuiguy` 询问是否可以一次性结合 Prompt Tuning 和 LORA，并希望解决 Prompt Tuning 为了灵活性而未暴露新 Token 的问题。
- **微调 Sentence-Transformer**：`@nickprock` 询问使用 msmarco 和 tripletloss 微调 Sentence Transformer 所需的样本数量，因为他只能找到与 TSDAE 相关的信息，该信息显示需要 60-100k 条句子。
- **修改 T5ForConditionalGeneration 和 Seq2SeqTrainer**：`@ianbestfellow` 寻求关于如何为他的研究修改 T5ForConditionalGeneration 的 forward 部分和 Trainer 的指导。
- **LayoutLM 和 RVL-CDIP**：`@gh6608` 询问是否有人成功将 LayoutLM 应用于 RVL-CDIP 数据集。
- **音译模型训练**：`@asterix3651` 和 `@vipitis` 讨论了创建音译模型的问题；`@asterix3651` 需要一个模型将一种语言的单词转换为罗马化形式，即使是词表外（out-of-vocabulary）的单词。
- **LLAMA-2 问题**：`@notooth` 描述了 LLAMA-2 模型在高度比较任务中遇到的困难，并寻求提高其性能的指导。
- **文本生成中的 Loss 测量**：`@exponentialxp` 对文本生成过程中测量 Loss 的部分感到好奇——是仅限于 Assistant/Response 部分，还是包含所有部分。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 条消息): 
        
- **叠加图像的分离**：`@cachalote123` 询问是否可以使用 **Diffusion 模型** 或任何其他技术来分离源自旧电影的叠加图像。
- **使用 Diffusers 将图像转换为 3D 对象**：`@drishya1` 正尝试使用 Diffusers 将图像转换为 3D 对象。他们提到已经使用 **Control Net** 将草图转换为图像，现在正在寻求关于将这些图像转换为 3D 的最佳模型的建议。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- 关于如何使用常规 LLM 创建 Embeddings 的讨论，`@marijnfs` 询问是否可以平均高层激活或使用最后一个 Token 的激活向量。
- `@dangfutures` 提出了关于 **GPT4** 微调的疑问和想法。
- 探索用于生成 Loss/评估图表的 **Weights & Biases** 开源替代方案，`@rtyax` 寻求建议，`@noobmaster29` 提议使用 **Tensorboard**。
- 包含关于 rebase 后合并 Pull Request [#787](https://github.com/OpenAccess-AI-Collective/axolotl/pull/787) 的交流、关于权限更改的讨论，以及来自 `@nanobitz`、`@caseus_` 和 `@dreamgen` 的关于已合并 PR [#2232](https://github.com/huggingface/accelerate/pull/2232) 的消息。
- `@caseus_` 简要对话了 torch SDP attention 作为 Flash Attention 替代方案的特性和优势，并由 `@nanobitz` 提供了 [torch 文档链接](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 以获取更多信息。
- `@nanobitz` 提到了 **Mistral** 类代码的滑动窗口（sliding window）特性构思。
- `@le_mess` 和 `@caseus_` 建议利用 Axolotl 微调模型，并从 **tinyllama** 示例开始。
- `@dreamgen`、`@touristc`、`@le_mess` 和 `@visuallyadequate` 对与 Axolotl 兼容的数据集进行了深入探讨，讨论了 ShareGPT、OpenAssistant/oasst1、OpenOrca 等资源，以及在 Hugging Face 中发现的数据集，如 Guanaco-Unchained-Refined 和 Wizard-Vicuna-Refined。
- 来自 `@dreamgen`、`@faldore` 和 `@nruaif` 的讨论涉及使用 toxic-dpo 和其他微调数据集、对 **Dolphin 3.0** 的期待、RAG 数据集开源选项的匮乏，以及 [Chatbot Arena Conversations 数据集](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) 的相关性。

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (6 messages): 
        
- **使用常规 LLM 创建 embeddings**：`@marijnfs` 询问了使用常规 LLM 创建 embeddings 的可能性和方法。他们询问是否可以选择提取高层激活（activations）并取平均值，或者取最后一个 token 的激活向量。
- **微调 GPT4**：用户 `@dangfutures` 询问是否有人在微调 **GPT4** 方面取得了成功。
- **W&B 的开源替代方案**：用户 `@rtyax` 寻求用于生成 loss/evaluation 图表的 **Weights & Biases** 开源替代建议。`@noobmaster29` 建议将 **Tensorboard** 作为一种开源选择。


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (22 messages🔥): 
        
- **关于合并 PR 的讨论**：用户 `@nanobitz` 讨论了在 rebase 后合并 pull request [#787](https://github.com/OpenAccess-AI-Collective/axolotl/pull/787) 的事宜。用户 `@caseus_` 确认这没有问题，并修改了必要的权限以允许合并。
- **权限调整**：针对一次意外向 main 分支的 push，`@caseus_` 进一步讨论了权限调整，试图限制未来出现类似问题。
- **huggingface/accelerate PR 合并**：在另一场对话中，用户 `@caseus_` 分享了关于 huggingface/accelerate 合并 PR [#2232](https://github.com/huggingface/accelerate/pull/2232) 的消息，该 PR 涉及 FP8 支持的集成。
- **torch SDP Attention**：`@nanobitz` 和 `@caseus_` 就 torch SDP Attention 的优势和功能进行了简短讨论，并分享了 [torch 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 链接以获取更多细节。`@caseus_` 指出它是 flash attention 的一种替代方案。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (10 messages🔥): 
        
- **Mistral 类代码**：`@nanobitz` 提到 **Mistral** 类代码具有 sliding window 功能。
- **Token 初始化**：`@dreamgen` 表示希望能够从某些现有的 token id 初始化 **new token** 的 embedding。
- **微调相关查询**：`@thetonsil.` 提出了两个关于微调模型的问题：
    - 他们询问 `是否可以仅使用 CPU 资源进行微调`。`@le_mess` 回复称，虽然理论上可行，但可能非常耗时，并建议租用云端 GPU。
    - 他们还寻求关于使用 **Axolotl** 的指导。根据 `@le_mess` 的建议，他们被引导至 examples 文件夹中的示例，特别是 **mistral** 和 **llama** 模型。
- **运行示例**：`@le_mess` 还提供了在设置环境后，如何使用 `accelerate launch -m axolotl.cli.traion <example_path>` 命令运行这些示例的指令。
- **从 TinyLlama 开始**：`@caseus_` 建议尝试 **tinyllama** 示例作为一个潜在的起点。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (8 messages🔥): 
        
- **ShareGPT 和 OASST 数据集**：在关于与 Axolotl 兼容的数据集的讨论中，`@dreamgen` 提到 ShareGPT 作为一个原始资源，有一些论文对其进行了分析并根据用户意图分解了对话；而 `@touristc` 在使用 RyokoAI/ShareGPT52K 和 OpenAssistant/oasst1 等热门数据集时遇到了困难。
- **OpenOrca 数据集**：`@le_mess` 建议将 **OpenOrca** 作为 Axolotl 的可用数据集。在确认正确的数据集类型时，`@touristc` 询问是否应将其设置为 alpaca_w_system.load_open_orca。
- **多轮对话数据集**：`@touristc` 表示有兴趣寻找能与 Axolotl 良好协作的多轮对话数据集，并指出 OpenOrca 数据集仅为单轮 QA 的局限性。`@nanobitz` 提到当前的讨论可以被视为单轮。
- **Guanaco-Unchained-Refined 和 Wizard-Vicuna-Refined 数据集**：`@visuallyadequate` 分享了 Hugging Face 上[两个数据集的链接](https://huggingface.co/datasets/)，名为 Guanaco-Unchained-Refined 和 Wizard-Vicuna-Refined，并强调它们的主要重点是为列表和代码块提供一致的格式。

### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (13 messages🔥): 
        
- **在 Fine-Tuning 中使用 Toxic-DPO 数据集**：用户 `@dreamgen` 提到他们在最近的 Fine-Tuning 练习中一直在使用 toxic-dpo 数据集，但对混合来自 unalignment 和质量数据的信号表示担忧。他们建议过滤像 Intel/orca_dpo_pairs 这样的大型 DPO 数据集，以剔除 "chosen" 回答中的 safety rejects。
- **等待 Dolphin 3.0**：用户 `@faldore` 表示他们正在等待 **Dolphin 3.0**，然后再进一步 Fine-Tuning 他们的系统，目标是准备好稳固的 system prompt、instruct、conversation、RAG 和 Agent 数据集。不过，他们指出 Dolphin 已经相当 uncensored 了。
- **缺乏开源 RAG 数据集**：针对 `@dreamgen` 对 RAG 数据集的询问，`@faldore` 提到目前还没有很多优秀的开源选择，这一领域需要进一步开发。
- **Chatbot Arena Conversations 数据集的使用**：`@dreamgen` 提出了一个问题，即为什么许多模型没有使用 [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) 进行 DPO/RL。作为回应，`@nruaif` 解释说该数据集可能太大，但他们观察到有人在使用较小的子集。
- **数据集子集的发布**：在 `@dreamgen` 询问 Chatbot Arena Conversations 数据集的子集是否已发布时，`@nruaif` 没有给出具体回答。`@dreamgen` 推测 gpt-4 获胜的子集可能相对安全。

        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- 围绕各种 AI 模型和解决方案展开了积极讨论，特别关注 **NexusRaven 模型**；怀疑者和支持者都分享了各自的观点，并提供了相关的 [Hugging Face 链接](https://huggingface.co/Nexusflow/NexusRaven-V2-13B)。 
- `@glenn_sjobs` 宣布即将举行一场 **AI Hackathon**，涉及在开发环境中使用 LangChain；分享了官方 [链接](https://www.defense.gov/News/Releases/Release/Article/3610215/chief-digital-and-artificial-intelligence-office-to-host-hackathon-in-hawaii/) 以获取更多信息。
- 讨论了关于各种平台功能的看法和疑问，包括 **S3 Storage**、**Streamlit**、**ContextualCompressor** 和 **VSCode Auto Import**；[Google Drive 链接](https://drive.google.com/drive/folders/1CgN7DE3pNRNh_4BA_zrrMLqWz6KquwuD) 中还引用了大量的 Python 课程材料。
- 介绍了 **Cumuli**，这是一个针对 AWS 优化的新 Chrome 扩展程序，采用了 **GPT-4 Turbo with vision**；提供了该扩展程序的 [GitHub 链接](https://github.com/petrgazarov/cumuli-aws-console-chat)。
- 分享了一本关于 **LangChain 的新书**，讨论了如何有效使用 LangChain 框架并规避其固有弱点；作者分享了在 [Amazon](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463) 上的购买链接。
- 用户提出了与在使用 session 级 memory 时控制对话历史长度相关的问题，并讨论了 `RunnableWithMessageHistory`、`charact_prompt | llm` 和 `RedisChatMessageHistory(session_id, url=REDIS_URL)` 的功能。

**LangChain AI Channel Summaries**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (27 messages🔥): 
        
- **S3 存储建议**：`@lhc1921` 建议考虑使用 S3 云存储或本地存储。
- **Python 问题资源**：`@b1llygamidesuyo` 分享了一个 [Google Drive 文件夹链接](https://drive.google.com/drive/folders/1CgN7DE3pNRNh_4BA_zrrMLqWz6KquwuD)，其中包含约 5TB 的 Python 课程资料。
- **Streamlit 按钮咨询**：`@infinityexists.` 询问如何在 Streamlit 中添加带有图像或 PNG 的按钮。
- **ContextualCompressor 提示词**：`@legendary_pony_33278` 寻求关于在非英语语言中使用 ContextualCompressor 的提示词帮助。
- **关于 NexusRaven 的讨论**：用户 `@_egeres` 和 `@lhc1921` 讨论了 [NexusRaven 模型](https://huggingface.co/Nexusflow/NexusRaven-V2-13B)。`_egeres` 发现该模型可以复制 OpenAI 的 function calling API 行为，而 `@lhc1921` 对其超越 GPT-4 的说法表示怀疑。
- **黑客松公告**：美国国防部长的高级软件/AI 工程师 `@glenn_sjobs` 向社区通知了即将举行的 AI 黑客松，该活动将在开发环境中包含 LangChain。他分享了[官方链接](https://www.defense.gov/News/Releases/Release/Article/3610215/chief-digital-and-artificial-intelligence-office-to-host-hackathon-in-hawaii/)以获取更多信息和申请流程。
- **VSCode 自动导入问题**：`@solononforever` 寻求关于 VSCode 自动导入功能无法正常工作的帮助。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 
        
- **会话级内存控制**：`@cryptossssun` 询问在使用 `RunnableWithMessageHistory` 函数进行会话级内存管理时，如何控制聊天历史记录的长度，特别是在使用 `charact_prompt | llm` 和 `RedisChatMessageHistory(session_id, url=REDIS_URL)` 的情况下。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **LangChain 新书**：用户 `@tusharg` 分享了亚马逊上新出版的 [LangChain 书籍](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463)链接。该书提供了关于如何利用 LLM 能力的见解，并探讨了其基础、伦理维度和应用挑战。
- **Cumuli，一款新的 AWS Chrome 扩展**：`@petrgazarov` 介绍了 Cumuli，这是一个为所有 AWS 页面添加 LLM 聊天面板的 Chrome 扩展。它允许用户将控制台截图添加到查询中，以获得上下文感知的回复。该扩展使用 **GPT-4 Turbo with vision**。可以通过 [GitHub](https://github.com/petrgazarov/cumuli-aws-console-chat) 访问该扩展。


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **最新的 LangChain 书籍**：`@tusharg` 分享了[亚马逊上的一本新书](https://www.amazon.com/Generative-AI-LangChain-language-ChatGPT/dp/1835083463)链接，内容涵盖使用 **LangChain 框架**开发 Agent 和个人助手等应用、集成网页搜索以及代码执行。该书还侧重于*利用 LLM 的能力并规避其固有的弱点*。


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- 正在进行文档分块以进行嵌入，从而生成一个数十亿规模的合成 Token 数据集；`@emrgnt_cmplxty` 详细介绍了所采用的方法，即递归分割成长度为 512 的片段，并在嵌入前重新附加标题。
- 讨论了对分块文档应用分层搜索，利用 Web 的自然层级结构；初始向量搜索索引每个网页的首个分块和标题，然后获取前 N 个匹配项的完整分块，并使用每个网页中最相似的分块进行重排序。该方法使初始搜索阶段所需的嵌入量减少了约 30 倍。
- `@gabriel_syme` 提到了文档摘要的准备工作，证实了 `@emrgnt_cmplxty` 分享的准备流程。
- 介绍了 GitHub 上的 [Mergekit](https://github.com/cg123/mergekit) 工具包，用于合并预训练的 LLM，并以一个用户模型案例进行了说明：[MetaMath-neural-chat-7b-v3-2-Slerp](https://huggingface.co/Weyaxi/MetaMath-neural-chat-7b-v3-2-Slerp)。
- `@cryptossssun` 询问了 AI 模型中长上下文领域能力、目标上下文识别和提取的发展情况。
- 询问了与 7B 参数模型相比的微调过程，随后 `@caseus_` 对讨论中的 "qlora" 模型发表了评论，表示其当前状态不尽如人意。

**Alignment Lab AI 频道摘要**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (9 messages🔥): 
        
- **为 Embedding 进行文档分块**：`@emrgnt_cmplxty` 讨论了他们准备文档进行 Embedding 的方法，提到他们正在将文档递归地切分为长度为 512 的分块，并在 Embedding 之前重新附加标题。
- **生成数十亿级的合成 Token 数据集**：`@emrgnt_cmplxty` 表示，这种准备文档进行 Embedding 的方法将有助于生成 **数十亿级的合成 Token 数据集**，而无需担心质量问题。
- **分层搜索**：`@emrgnt_cmplxty` 澄清说，他们所说的分层搜索是指利用 Web 的自然层级结构来执行搜索。他们为每个网页索引首个分块 + 标题，并将其作为第一阶段的向量搜索。然后，他们获取前 N 个匹配项的完整分块，并使用每个网页中最相似的分块进行重排序（re-rank），这使得第一阶段需要搜索的 Embedding 数量减少了约 30 倍。
- **文档摘要准备**：`@gabriel_syme` 提到他们也在为类似的文档摘要处理过程做准备。


### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (3 messages): 
        
- **用于合并语言模型的 Mergekit 工具包**：`@lightningralf` 分享了 GitHub 上的 [Mergekit](https://github.com/cg123/mergekit) 工具包链接，该工具包提供了合并预训练大语言模型的工具。
- **Mergekit 在 MetaMath-neural-chat 模型中的应用**：`@lightningralf` 进一步阐述了 Mergekit 的可用性，并提供了 `@Weyaxi` 开发的 [MetaMath-neural-chat-7b-v3-2-Slerp](https://huggingface.co/Weyaxi/MetaMath-neural-chat-7b-v3-2-Slerp) 模型示例，该模型使用了 Mergekit 进行模型合并。
- **关于长上下文领域的查询**：`@cryptossssun` 询问是否有人在关注长上下文能力以及目标上下文识别与提取等方面。


### ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/) (2 messages): 
        
- **微调咨询**：`@emrgnt_cmplxty` 询问了微调过程以及该模型与 70 亿参数模型的对比。
- **模型质量**：作为回应，`@caseus_` 澄清说讨论的模型是一个 **qlora**，并评论说效果不是很好。


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

只有一个频道有活动，因此无需汇总...

- **QA Agent**：用户 `binary6818` 分享了一个名为 [camelQA](https://camelqa.com/) 的项目信息，该项目提供用于测试移动应用的 AI 服务，并询问是否有人知道类似的项目。
- **将扫描书籍转换为聊天机器人**：用户 `.kareemk` 寻求关于 OCR PDF 到 RAG 流水线的开源方案建议，以创建一个“与我的扫描书聊天”的功能。用户 `coffeebean6887` 建议大多数 OCR 库在处理打印和扫描文本时表现良好，但像手写笔记这样更复杂的文档可能需要高级 OCR 模型。
- **对 Anyscale LLM 性能基准测试的批评**：用户 `guardiang` 提到 Anyscale 昨天发布的 LLM 性能基准测试正受到批评，并分享了[该话题的链接](https://x.com/soumithchintala/status/1738241213327692174?s=20)。
        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- general 频道的用户 `@tomsegura` 对 Skunkworks AI 社区及其重大进步的潜力表示乐观，指出：“*我坚信像你们这样的人就是未来，并将为我们带来普通人（我们）可以使用的真正进步。*”
- `@cyborgdream` 在 datasets 频道发起了一场讨论，关于一个包含 10-20 亿 Token 的最先进合成生成数据集的潜在应用，旨在造福开源 AI 社区。
- off-topic 频道讨论了由 `@shreepandey` 链接的一条 [推文](https://twitter.com/8teAPi/status/1737237462672707872)。该推文强调了由 [Smarterchild.chat](http://Smarterchild.chat) 开发的实时语音聊天导师在延迟方面的显著降低。这引发了人们对实现这种性能提升所采用方法的兴趣。

**Skunkworks AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (2 messages): 
        
- 用户 `@tomsegura` 对 Skunkworks AI 社区的潜力表示支持和信心，指出：“*我坚信像你们这样的人就是未来，并将为我们带来普通人（我们）可以使用的真正进步。*”

### ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/) (2 条消息): 
        
- **潜在的合成生成数据集**：用户 `@cyborgdream` 询问了关于一个约 1-2B tokens 的 **State Of The Art** *合成数据集* 的理想研究领域，该数据集可能对开源软件 **AI community** 有益。


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (2 条消息): 
        
- **语音功能延迟的降低**：`@shreepandey` 分享了来自 `@8teAPi` 的一条 [推文](https://twitter.com/8teAPi/status/1737237462672707872) 链接。该推文讨论了由 [Smarterchild.chat](http://Smarterchild.chat) 开发的实时语音聊天导师，它显著降低了延迟。`@shreepandey` 和 `@skaios` 对这种延迟降低是如何实现的很感兴趣。


        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- 用户询问了 **qLoRA** 的性能，`@tcapelle` 询问是如何确定 qLoRA 表现不佳的。
- `@tcapelle` 进一步建议通过冻结模型的顶层来进行 **model fine-tuning**，而不是使用 LoRA 或 qLoRA。
- 介绍了一位新成员 `@philipmay`，他对邀请表示了感谢。 
- `@jiha` 分享了一个题为 "OpenDalle - Like running Dall-e Local" 的 [YouTube 视频](https://youtu.be/DngRcgYjDfU?si=x708Sq49YqkLr_75)，展示了 **OpenDalle** 的功能，引发了富有成效的对话。
- `@datarevised` 被确认为聊天中分享的 OpenDalle 视频的创作者，`@jiha` 对视频质量表示赞赏，并表示根据视频内容，OpenDalle 看起来很酷。

**DiscoResearch 频道总结**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (2 条消息): 
        
- **关于 QLoRA 性能的查询**：`@tcapelle` 询问了关于如何确定 **qLoRA** 表现不佳的信息。
- **模型微调建议**：`@tcapelle` 建议，与其使用 **LoRA** 或 **qLoRA**，不如尝试 **冻结模型顶层**，这可能会更有益。


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (4 条消息): 
        
- **新成员介绍**：`@philipmay` 加入了聊天并对邀请表示感谢。  
- **关于 OpenDalle 视频的讨论**：`@jiha` 分享了一个 [YouTube 链接](https://youtu.be/DngRcgYjDfU?si=x708Sq49YqkLr_75)，视频标题为 "OpenDalle - Like running Dall-e Local"，似乎展示了 OpenDalle 的功能。
- **OpenDalle 视频的创作者**：`@datarevised` 确认他们创作了聊天中分享的 OpenDalle 视频。 
- **对 OpenDalle 视频的赞赏**：`@jiha` 称赞了视频的质量，并表示根据视频内容，OpenDalle 看起来很酷。


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- 用户 `@emfastic` 开发了一个采用装饰器语法的内部工具，并表达了未来将其开源的意图。
- `@emfastic` 对 embeddings 的使用极少，否则会考虑使用 **llamaindex**。
- swyxio 针对部分用户面临的速度缓慢问题发起了询问，问道：“你观察到的速度有多慢”。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (2 条消息): 
        
- 用户 `@emfastic` 提到他们使用装饰器语法构建了一个内部工具，并计划很快将其开源。
- `@emfastic` 还表示他们没有广泛使用 embeddings，否则他们会考虑使用 **llamaindex**。


### ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/) (1 条消息): 
        
swyxio: 你观察到的速度有多慢


        

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord 总结

只有一个频道有活动，因此无需总结...

- **XGBoost 在 Apple M Pro 系列上的问题**：用户 `@leg4l` 报告了在搭载 Apple M Pro 系列芯片的 Macbook Pro 上运行 [XGBoost](https://xgboost.readthedocs.io/) 的问题，应用程序利用了效率核心，但对性能核心几乎没有施加压力。分享了一个与此问题相关的 GitHub 讨论：[M1 Pro, M1 Max, 和 M1 Ultra 架构下 CPU 利用率不完整 - Issue #8320](https://github.com/dmlc/xgboost/issues/8320)