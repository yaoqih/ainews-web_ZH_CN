---
companies:
- nous-research
- tyrannosaurus
date: '2023-12-29T10:32:18.263166Z'
description: '**Nous Research AI** 的 Discord 讨论涵盖了 AI 排名图、**ChatGPT** 与 **Obsidian**
  的 Latex 数学格式兼容性问题，以及 **TinyLlama 1.1B** 模型在各项基准测试中的性能表现。用户分享了包括数学语料库 **MathPile**、知识图谱构建方法以及开源大语言模型仓库在内的各类资源。技术讨论涉及了
  **Mixtral** 等模型的去中心化计算可行性、关于 AI 意识（sentience）的哲学辩论，以及模型微调和 Token 计数的策略。此外，社区还探讨了
  **Obsidian** 模型、视觉模型训练，以及由 Tyrannosaurus 发布的多模态模型 **TinyGPT-V**。“ChatGPT 生成的 Latex
  数学格式与 Obsidian 不兼容”以及“对在有生之年实现人类水平的 AI 持乐观态度”是其中较为显著的观点。'
id: 42e09d30-0605-4da8-921b-0e4cf2fd2c43
models:
- tinyllama-1.1b
- mixtral
- tinygpt-v
original_slug: ainews-12282023-smol-talk-updates
people:
- gary-marcus
title: 2023年12月28日：Smol Talk 更新
topics:
- latex
- benchmarking
- knowledge-graphs
- model-finetuning
- tokenization
- decentralized-computation
- philosophy-of-ai
- multimodality
- vision
- open-source-models
---

<!-- buttondown-editor-mode: plaintext -->今天新闻不多，正是提升摘要质量的好时机——我们改进了爬虫从 Twitter 可靠抓取元数据的能力，供摘要生成器使用。现在我们还会列出所有链接，方便浏览。作为链接聚合工具，现在的可读性和可用性应该更高了！

 
![image.png](https://assets.buttondown.email/images/7e43e094-1a14-4c38-828f-3c7775b29a9b.png?w=960&fit=max)
 


[TOC] 


## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **关于 AI 态度分布图的讨论**：深入探讨了 AI 态度图表上的位置分布。用户询问了特定个人的位置，并承认了这种展示方式的局限性。
- **ChatGPT 与 Latex 数学格式的兼容性**：讨论了 ChatGPT 无法生成与 Obsidian 软件兼容的 Latex 数学格式的问题，并提出了潜在的解决方案。
- **TinyLlama 1.1B 性能**：`@teknium` 分享了 TinyLlama 1.1B 模型在多项任务中的详细性能指标。
- **AI 相关资源与项目**：分享了各种 AI 相关资源，包括新的以数学为核心的语料库 MathPile、构建知识图谱的方法、Open Large Language Model 相关的 GitHub 仓库链接，以及关于潜在技术实现的讨论。
- **关于 AI 能力的复杂讨论**：用户讨论了运行像 Mixtral 这样的大型语言模型的去中心化计算的可行性、AI 感知力与意识的哲学层面，以及对 Large Language Models 漏洞的潜在利用。
- **关于 Large Language Models 的技术咨询**：关于模型转换为 AWQ、在不导入整个库的情况下为开源模型计算 token 的方法，以及增强文本分类的 finetuning 策略的咨询。
- **Project Obsidian 讨论**：集中讨论了 Nous 的 Obsidian 模型的运行与分析、视觉模型训练过程、来自 Tyrannosaurus 的小型多模态模型 TinyGPT-V 的发布，以及开源项目的社区导向性质。

**Nous Research AI 频道摘要**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (18 条消息🔥): 
        
- **AI 图表上的定位**：`@nonameusr` 询问 `@ldj` 在某张代表对 AI 态度的图表中的位置。`@ldj` 提到他认为自己的位置比他个人认同的更偏西南。然而，他也承认了这种图形排列的实际局限性，并强调象限才是最重要的。他表示对在我们有生之年实现人类水平的 AI 持乐观态度，并对未来 50 年文明的进步持乐观态度。
- **识别 AI 图表上的个人**：`@night_w0lf` 询问了一个名为 Tek 的人的位置，`@nonameusr` 澄清其在左下角。`@max_paperclips` 对一个名为 Pico 的人出现在图表的所有四个角落表示困惑。
- **ChatGPT 与 Latex 数学格式**：`.beowulfbr` 表达了对 ChatGPT 生成的 Latex 数学格式与 Obsidian 不兼容的担忧，并寻求解决方案。`@nonameusr` 建议要求 ChatGPT 以 latex 格式回答，但 `.beowulfbr` 回复称该建议仅部分解决了问题。
- **Gary Marcus 对 AI 的观点**：`@Error.PDF` 询问同一张 AI 图表上的 Gary Marcus 是否代表了 AGI 永远无法实现的观点，他得到的回复是否定的。


### ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/) (4 条消息): 
        
- **TinyLlama 1.1B 在不同任务上的性能**：用户 `@teknium` 分享了在不同任务上运行 `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` 的结果。一些关键结果包括：
    - 在 `truthfulqa_mc` 上，实现了 mc1 分数 `0.2203` 和 mc2 分数 `0.3759`。
    - 在 `arc_challenge` 上，实现了 `0.2782` 的准确率和 `0.3012` 的归一化准确率。该任务集的平均分数为 **52.99**。
    - 在 `agieval_aqua_rat` 上，实现了 `0.1575` 的准确率和 `0.1693` 的归一化准确率。这些评估任务的平均分数为 **21.05**。
    - 在 `bigbench_causal_judgement` 上，实现了 `0.5053` 的 multiple_choice_grade。这些 bigbench 任务的平均分数为 **31.95**。
- 这些频道消息中没有分享或讨论链接或博客文章。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (12 messages🔥): 
        
- **通用工作模型讨论**：`@skadeskoten` 提到一旦有了通用的工作模型，就有可能创建 ASICs 或 FPGAs。
- **数学生成式 AI - MathPile**：`@metaldragon01` 分享了 `@arankomatsuzaki` 关于名为 MathPile 的新数学中心语料库的 [Twitter 线程](https://fxtwitter.com/arankomatsuzaki/status/1740564961032556942)链接。他还提供了该项目的 [网站](https://gair-nlp.github.io/MathPile/)、[GitHub 仓库](https://github.com/GAIR-NLP/MathPile/) 和 [摘要](https://arxiv.org/abs/2312.17120)。`@gabriel_syme` 询问这是否是一个预训练数据集。
- **Tinyllama Checkpoints 基准测试**：`@teknium` 管理了 Tinyllama 的最后三个 Checkpoints。
- **模组化 Minecraft 讨论**：`@teknium` 询问 `@1084792750001618965` 是否玩模组化 Minecraft，`@max_paperclips` 给予了肯定回答，但指出他很少有时间玩游戏。
- **使用 Instructor 项目构建知识图谱**：`@fullstack6209` 分享了一个关于在指导下构建知识图谱数据的 [Gist](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34) 以及一个名为 Instructor 的项目。他表示，在使用 "vllm" 跑满 2080ti/3090 的情况下，消化一本书大约需要 30 分钟。
      
提到的链接：

- [Aran Komatsuzaki (@arankomatsuzaki) 的推文](https://fxtwitter.com/arankomatsuzaki/status/1740564961032556942)：数学生成式 AI：MathPile - 展示了一个多样化且高质量的以数学为中心的语料库，包含约 9.5B tokens。
- [asdf.py](https://gist.github.com/fullstackwebdev/44d99a064d037ec16c56fded98ae0a34)：GitHub Gist：即时分享代码、笔记和片段。


### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (170 messages🔥🔥): 
        
- **关于 Mixtral 模型容量与性能的讨论**：`@skadeskoten`、`@giftedgummybee` 和 `@n8programs` 之间就使用去中心化计算运行 **Mixtral** 模型的可行性进行了详细交流。他们讨论了其复杂的结构，以及在考虑去中心化架构时，延迟和资源编排（resource orchestration）所带来的困难。他们的结论是，目前的基础设施并不利于像 Mixtral 这样的大语言模型进行去中心化计算。
- **AI 的认知能力与感知力**：`@teknium` 分享了 **Hermes 2** 关于人工智能中意识、感知力（sentience）和 Qualia（感质）主题的详细回答。AI 评论了这些概念带来的哲学挑战以及研究它们的潜在科学方法，并指出目前的理解并不支持 AI 可以拥有类似于生物实体的这些属性。
- **使用 Obsidian 模型**：`@vic49.` 询问关于运行 **Obsidian** 模型的问题。`@orabazes` 建议使用 GGUF 量化和 llama.cpp 进行后端操作，并参考原始仓库获取 gradio 相关信息。
- **为文本分类微调 Llama 变体**：`@shivdinho` 正在寻找合适的数据集来微调某个版本的 **Llama**，以增强其文本分类能力。
- **测试 LLM 漏洞**：`@shinchan5137` 创建了一个测试大语言模型漏洞的平台，目前已经能够执行 Exfiltration（数据窃取）、Jailbreaking（越狱）和 Prompt Hijacking（提示词劫持）。更多的漏洞正在探索中。
      
提到的链接：

- [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://proceedings.mlr.press/v119/such20a.html)：本文研究了一个有趣的问题：我们是否可以创建能够自动生成训练数据、学习环境和课程的学习算法，以帮助 AI Agent...
- [与开源大语言模型聊天](https://chat.lmsys.org/)
- [Resurrect.ing](https://resurrect.ing)
- [Emad (@EMostaque) 的推文](https://fxtwitter.com/EMostaque/status/1740306310204440677)：我确实想知道法律/社会将如何处理具身机器人及其持续的训练。
- [Together AI (@togethercompute) 的推文](https://fxtwitter.com/togethercompute/status/1740586773296885767)：@eating_entropy @Teknium1 @zamdoteth 应该明天就能上线！
- [anton (@abacaj) 的推文](https://fxtwitter.com/abacaj/status/1740432829979459903)：第一次尝试 chatglm3-6b-32k... 实际上还挺不错的？我在上面运行了 humaneval，得分 60%。它在 32k 上下文内几乎拥有完美的召回率（上下文图片来自 reddit）。
- [GitHub - NousResearch/Obsidian: 也许是新的 SOTA 视觉模型？我们拭目以待 🤷‍♂️](https://github.com/NousResearch/Obsidian)

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (10 messages🔥): 
        
- **转换为 AWQ**: `@fullstack6209` 询问了关于将模型转换为 AWQ 的问题。`@beowulfbr` 引用了[此处](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)的页面，该页面提供了使用 AutoAWQ 工具将模型转换为 AWQ 的说明，但强调目前 vLLM 对 AWQ 的支持仍处于未优化状态。`@casper_ai` 进一步建议参考[此处](https://github.com/casper-hansen/AutoAWQ/tree/main/examples)的示例以获得更好的理解。

- **开源模型的 Token 计数**: `@fullstack6209` 询问是否存在一种不需要导入整个 transformers 库就能对开源模型进行 Token 计数的方法。`@kcaverly` 建议可能可以仅使用 tokenizers，并提供了 [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) 的相关链接。`@vic49` 肯定了这一方法，表示他们个人也使用 tokens 来计算 Token 数量。`@orangetin` 提供了一种使用 tokenizers 库而不导入整个 transformers 库来计算 Token 的简便方法，并给出了一个说明性的 Python 示例。`@fullstack6209` 对此表示感谢，并承认他们最初以为这需要专门的库，且过程会很慢。
      
链接提到：

- [AutoAWQ &#8212; vLLM](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
- [AutoAWQ/examples at main · casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ/tree/main/examples): AutoAWQ 实现了用于 4-bit 量化的 AWQ 算法，在推理过程中可实现 2 倍加速。 - casper-hansen/AutoAWQ


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (82 messages🔥🔥): 
        
- **Obsidian 模型运行尝试及求助**: 用户 `@vic49` 最初就运行 Obsidian 模型寻求帮助，并对尚未交付的承诺脚本表示沮丧。Nous 的联合创始人 `@teknium` 尝试通过分享 GitHub 仓库链接并建议推理 Obsidian 所需的代码片段来提供帮助，尽管他承认不确定如何成功实现。他解释说 Obsidian 不能直接与 transformers 配合使用，而是使用了来自 Llava 的自定义类。

- **社区指导与现实预期**: 用户 `@gabriel_syme` 提醒 `@vic49` 开源项目是基于志愿者性质的，现实生活中的事件可能会推迟承诺的更新，对此 `@vic49` 对违背承诺表示失望。

- **Obsidian 模型分析**: `@vic49` 分析了 Obsidian 模型，指出它部分依赖于 Transformers 库并包含自定义类。然而，他们觉得如果没有更多指导或简单示例，就无法构建脚本，并将其比作 CogVLM 的方法。

- **TinyGPT-V 介绍**: `@qnguyen3` 介绍了 `TinyGPT-V`，这是一个由 Tyrannosaurus 构建的较小模型，专为多模态用途设计。他还提到了 MobileVLM，这是一个面向移动设备的多模态视觉语言模型。

- **视觉模型训练**: 展开了关于视觉模型训练的讨论。`@teknium` 询问为什么视觉模型需要多个训练阶段，以及为什么不能将图像作为 Token 编码进 LLM 的常规 SFT 阶段。`@qnguyen3` 认为视觉编码器通常会降低图像表示的质量，因此需要多个训练阶段。`@coffeebean6887` 补充说，视觉编码器是较小的预训练模型，训练通常分为两个阶段，最终目标是将视觉 embeddings 映射到文本嵌入空间。
      
链接提到：

- [Google Colaboratory](https://colab.research.google.com/drive/1C1FkoeZYBv3dZELaKgxahoZzWPfz0En8?usp=sharing)
- [Tyrannosaurus/TinyGPT-V · Hugging Face](https://huggingface.co/Tyrannosaurus/TinyGPT-V)
- [Paper page - MobileVLM : A Fast, Reproducible and Strong Vision Language Assistant for Mobile Devices](https://huggingface.co/papers/2312.16886)
- [Paper page - TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones](https://huggingface.co/papers/2312.16862)
- [GitHub - NousResearch/Obsidian: Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️](https://github.com/NousResearch/Obsidian): 也许是新的 SOTA 视觉模型？拭目以待 🤷‍♂️ - GitHub - NousResearch/Obsidian
- [Obsidian/llava/serve/cli.py at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/llava/serve/cli.py)
- [GitHub - qnguyen3/hermes-llava](https://github.com/qnguyen3/hermes-llava): 通过在 GitHub 上创建账户来为 qnguyen3/hermes-llava 的开发做出贡献。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **数据隐私与 AI 训练讨论**：用户对个人隐私表示担忧，并讨论了退出 AI 训练的可能性；“Chat History and Training”选项被提及作为关键的控制机制。
- **AI 背景下的职业方向**：对话集中在如何应对 AI 对就业市场的影响，观点倾向于深度专业化和共情式创业（empathetic entrepreneurship）；强调了 AI 对教育可能产生的广泛影响。
- **基于 AI 的图像生成工具**：针对设计项目寻找类似 DALL-E 的 AI 工具的需求，用户推荐了 Bing，尽管它有每日限制。
- **AI 模型之间的对比**：Mixtral 潜力超越 GPT 的话题引发了各种反应，一些人指出需要更强大的模型，而另一些人则暗示 Mixtral 具备处理通常由 GPT-3.5 负责的任务的能力。
- **ChatGPT Plus 是自定义 GPT 的前提**：一位用户虽然是 ChatGPT 用户但无法创建自定义 GPT，这揭示了可能需要 ChatGPT Plus 订阅才能使用 Custom GPT 功能。
- **OpenAI 用户界面问题**：用户抱怨在 UI 更改后 GPT-4 的输出质量下降（例如 SQL 查询），并强调了在编码过程中必须提醒 AI 排除注释。
- **参考 GPT 的知识文件 (Knowledge File)**：有建议称 GPT 可能会默认使用其自身的训练数据，因此显式指示 GPT 检查知识文件（Knowledge File）可能会有帮助。
- **频繁的人机验证步骤**：有报告称在 AI 交互过程中反复出现验证和拼图测试；需要澄清这是标准流程还是 Bug。
- **ChatGPT 中的交互限制**：关于 ChatGPT 拒绝分析个人数据（特别是银行对账单）的投诉，引发了关于防止滥用敏感数据的限制猜测。
- **大文件处理的挑战**：建议将大文件拆分为较小的块，以克服 ChatGPT 的“Error in input stream”问题。
- **AI 辅助图像提示词 (Prompt) 创作**：关于将 ChatGPT 转化为图像提示词创作艺术家的问题，得到了一个[指向自定义 GPT 函数的链接](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer)的回应。
- **支付问题与解决方案**：关于 OpenAI 服务支付方式的问题确认了信用卡是唯一接受的形式。关于账号被封后继续计费的查询被引导至 [OpenAI Support](https://help.openai.com/)。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (60 条消息🔥🔥): 
        
- **AI 训练中的数据隐私**：用户 `@the_boss7044` 表示希望能够让单个对话不被用于 AI 训练。用户 `@tariqali` 澄清说，如果关闭“Chat History and Training”选项，与 ChatGPT 的任何对话都不会被用于训练。
- **AI 时代的职业前景**：用户 `@apex_algorithms` 发起了关于在 AI 进步面前如何定位自己的讨论。对话围绕两种策略展开：a) 在需要深度理解的学科领域进行专业化；b) 共情式创业。包括 `@dydzio` 和 `@bambooshoots` 在内的几位用户对 AI 对工作（尤其是软件开发）的影响提出了不同看法。`@michael_6138_97508` 建议，预设 AI 的固定能力和局限性可能是一个错误，AI 可能会为教育等行业的深刻变革铺平道路。`@.dooz` 补充说，由于后者有特定需求，AI 编程（Programming）可能比 AI 软件开发（Software Development）更容易管理。
- **AI 与图像生成**：用户 `@vix6262` 询问是否有类似 DALL-E 的免费 AI 图像生成器用于设计项目。`@satanhashtag` 提供了 Bing 等建议，但也指出其每天有使用限制。
- **AI 模型之间的对比**：用户 `@dojan1` 询问 Mixtral 是否很快会优于 GPT，得到了褒贬不一的回应。`@jaicraft` 表示需要更实质性的模型，而 `@exx1` 则建议 Mixtral 可以执行一些通常由 GPT-3.5 完成的任务。
- **GPT 在代码编写方面的实用性下降**：用户 `@Pandor` 对 GPT 在提供正确代码方面的有效性下降表示不满，甚至考虑取消订阅。讨论涉及了使用旧模型和其他平台的可能性。

### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (56 条消息🔥🔥): 
        
- **支付方式讨论**：用户 `@grimgrinner` 提出了关于 OpenAI 服务是否支持现金支付的问题。`@satanhashtag` 和 `@lugui` 澄清说 OpenAI 仅接受信用卡支付。
- **DALL-E 应用**：讨论了在 Discord 中使用 DALL-E 的可能性，该话题由 `@ryobdqui` 发起。`@lugui` 建议可以开发一个使用 OpenAI API 生成图像的 Discord bot，并提到有一些公开的机器人可以下载并设置。
- **VPN 与验证码测试**：`@busybenss` 询问 VPN 是否会导致每条消息都触发验证码测试（puzzle test）。`@satanhashtag` 和 `@lugui` 给出了肯定的回答。
- **GPT-4 准确度与消息限制问题**：`@august8319` 注意到 GPT-4 的准确度有所下降，并引入了消息限制。`@mrhorse`、`@jaicraft` 和 `@【ｐｅｎｕｌｔｉｍａｔｅ】` 讨论认为，这些变化是由于切换到了 GPT-4 Turbo 模型，并实施了消息上限以防止服务器过载。
- **自定义 GPT 的技术问题**：`@odiseo3468` 在与其自定义 GPT 交互时遇到了“Error searching knowledge”（搜索知识库错误）的消息。

提到的链接：[Discord - 与好友和社区聊天的新方式](https://discord.com/channels/974519864045756446/1039968564699992106)：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (30 条消息🔥): 
        
- **自定义 GPT 创建与访问问题**：用户 `@andreimotin` 遇到了与创建自定义 GPT 相关的问题。在访问 OpenAI 网站的相关页面时，他们收到了“您目前无权访问此功能”的消息。`@elektronisade` 询问 `@andreimotin` 是否拥有 ChatGPT Plus，得到的回答是否定的，这表明**可能需要 ChatGPT Plus 才能访问此功能**。

- **ChatGPT Plus 支付问题**：用户 `@.caan` 反馈称，尽管他们的 ChatGPT 账号被封禁，但仍被收取了 ChatGPT Plus 的费用。`@satanhashtag` 建议只有官方的 [OpenAI 支持页面](https://help.openai.com/) 才能协助处理支付问题。

- **ChatGPT 拒绝分析个人数据**：用户 `@zoemdoef` 抱怨 ChatGPT 拒绝帮助分析其银行对账单，这可能是由于为了防止滥用敏感数据（如医疗处方和报告）而实施的内容限制，正如 `@elektronisade` 所观察到的。

- **GPT-4 回复与查询质量问题**：用户 `@PythonGuy` 提到 GPT-4 的回复质量有所下降，特别是在 UI 更改之后。具体而言，他们提到 AI 倾向于在 SQL 查询中插入不完整的组件。该问题得到了 `@elektronisade` 的确认，他建议提醒 AI 避免注释并提供完整的代码解决方案。

- **用户的 ChatGPT 问题**：用户 `@Delzi` 和 `@PythonGuy` 遇到了聊天停止工作以及发送消息时出错的问题。他们不得不反复关闭应用或刷新浏览器作为临时解决方法。

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (28 messages🔥): 
        
- **GPT Knowledge File Reference**: `@solbus` 建议在指示 GPT 引用知识文件时，明确要求 GPT 检查该文件。原因在于，GPT *往往会优先参考其自身的训练数据*。提供的一个例子是：“当用户询问有关螺丝刀的问题时，请在知识文件 'screwdrivers.txt' 中搜索与查询相关的信息”。
- **Issue with Human Verification Steps**: `@vova5963` 提出了一个关于在使用 OpenAI 的 Chatbot 过程中频繁出现人机验证步骤的问题。该问题间歇性发生，有时会连续出现 5 次验证提示。频道中的其他人对此感到困惑，因为他们没有遇到过，或者将其误认为是双重身份验证 (2FA)。
- **Problem with Citation in Azure OpenAI**: `@shico_ai` 提出了一个问题，即 Azure OpenAI 中的引用指向了存储在 Azure Blob Storage 上的视频文件链接。他们更希望引用是视频本身的链接。
- **Code Repetition Bug**: `@happyg` 报告了一个 Bug，即他们正在构建的 GPT 会重复尝试运行代码 `search('<query>')`，通过指示 GPT 不要使用 Code Interpreter 解决了该问题。
- **Slow GPT Performance**: `@dreamer100` 和 `@lodosd` 指出 GPT 处理小型文本文件（约 20-30K）的速度非常慢，将其比作使用 20 年前的电脑。他们猜测 OpenAI 正在调整速度以节省算力。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (14 messages🔥): 
        
- **Working with Large PDF Files**: 用户 `@greenyellowblue` 寻求关于如何对大型 PDF 文件（60mb，450 页）使用 ChatGPT 的建议，因为这会导致 "Error in input stream"。`@madame_architect` 建议将文件拆分为较小的块并按顺序重命名（例如，“Rulebook Vol 1”、“Vol 2”等）。
- **Converting ChatGPT into an Image Prompt Writing Artist**: `@errorsource` 询问是否有办法将 ChatGPT 转换为图像提示词创作艺术家，对此 `@madame_architect` 提供了一个用于此目的的 Custom GPT 功能链接 ([Custom GPT](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer))。
- **Message Limitations and Accuracy of GPT-4**: `@august8319` 询问关于 GPT-4 似乎变得不那么准确以及 3 小时内限制 40 条消息的规定，`@beanz_and_rice` 确认该限制已实施数月。
- **Generating Semi-Randomized Character Portraits**: `@ponderous_` 征求生成多样化角色肖像的技巧，因为生成的角色看起来非常相似且像模特。在此对话期间未收到回复。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (14 messages🔥): 
        
- **Working with Large .PDF Files**: 用户 `@greenyellowblue` 在尝试处理大型 .PDF 文件（60mb，450 页）时遇到问题，始终导致 "Error in input stream"。`@madame_architect` 建议将文件拆分为较小的块，这似乎解决了问题。
- **Transforming ChatGPT into an Image Prompt Writing Artist**: 针对 `@errorsource` 关于将 ChatGPT 转换为图像提示词创作艺术家以获得更多样化提示词的问题，`@madame_architect` 提供了一个 Custom GPT 解决方案的链接——即 [Midjourney Prompt Engineer](https://chat.openai.com/g/g-xIWKJf4Ln-midjourney-prompt-engineer)。
- **Limitations and Performance of GPT-4**: 用户 `@august8319` 对 GPT-4 感知上的准确度下降以及 3 小时 40 条消息的上限表示担忧。`@beanz_and_rice` 确认消息上限已实施数月，并认为感知上的准确度下降可能是模型变得更“懒”的反映。
- **Generating Semi-randomized Character Portraits**: `@ponderous_` 正在寻求生成半随机角色肖像的建议，旨在避免让角色看起来太相似或太像模特。讨论的消息中没有提供解决方案。
- **Chat Structure Review**: 用户 `@bambooshoots` 收到来自 `@beanz_and_rice` 关于优化消息结构的建议。分享了一个链接，但没有关于结构问题性质的直接引用或进一步讨论。


        

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- 围绕 **LM（语言模型）选择和性能** 进行了广泛讨论，重点关注不同 GPU 配置和模型大小的内存需求。值得注意的是，用户分享了在性能较低的机器上运行大型模型的建议，并对比了模型性能。讨论主题包括模型区分、运行多个本地服务器、代码集成以及处理模型错误。
- 用户分享了在使用 Dolphin 2.6 Mistral 7B GGUF 和 mixtral 8x7B dolphin 2.6 等不同 LM 时的**经验和技术挑战解决方案**。还讨论了在 LM Studio 中使用 Embedding 模型以及运行多个实例的问题。
- 在**集成**方面，用户讨论了在服务器模式下于 LM Studio 浏览器中运行不同语言模型的可能性，指出了潜在的集成场景。
- **硬件相关**的对话围绕着将大型语言模型与高端硬件配置（使用 AMD 和 Intel 的下一代 CPU）相匹配，以及通过 LM Studio 在 GPU 上运行更大型模型的方法。
- **解决了相关的 Beta 版本发布问题**，例如 LM Studio 应用程序中的搜索面板（Search Pane）崩溃问题，该问题通过重新安装得以解决。
- 分享的关键资源包括 [Silly Tavern 文档](https://docs.sillytavern.app/usage/api-connections/openai/#proxy)、[TheBloke 的 GPT-3 模型](https://huggingface.co/TheBloke) 以及 [BBC-Esq 的 ChromaDB-Plugin-for-LM-Studio](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio) 等。

**LM Studio 频道总结**

### ▷ #[🎄🎅-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (119 条消息🔥🔥): 
        
- **用户硬件与模型选择**：`@qwerty_qwer` 询问了关于在其 16GB GPU 上使用哪种语言模型的建议。`@starscreen.` 询问了在 64GB 内存的机器上能运行的最大语言模型，`@heyitsyorkie` 建议 `@starscreen.` 在其配置上可能可以运行 7b 或 13b 模型。 
- **LM Studio 与 Silly Tavern 集成**：`@american_pride` 分享了将 LM Studio 集成到另一个名为 Silly Tavern 平台中的想法。`@dagbs` 分享了一个[文档链接](https://docs.sillytavern.app/usage/api-connections/openai/#proxy)，涵盖了 Silly Tavern 中的 API 连接，这可能实现该集成。
- **Linux 上的模型**：`@binepilo` 在 Linux 上尝试运行 phi-2 模型时遇到问题。`@psipiai` 建议将 LM Studio 更新到 0.2.10 版本以解决此问题。
- **区分模型**：`@mr.kittyhawk` 和 `@dagbs` 建议使用标签根据使用场景和能力来区分模型。他们建议为模型大小、流派、使用场景和硬件兼容性设置标签，以及 `Programming`（编程）、`Narrative Writing`（叙事写作）、`Character AI` 等大类。`@yagilb` 将这些建议的标签添加到了讨论论坛中。
- **在 LM Studio 中运行多个本地服务器**：`@loganyang` 询问了在 LM Studio 中运行多个本地服务器的可能性。`@dagbs` 澄清说 LM Studio 一次只能加载一个模型，而 `@psipiai` 建议运行多个 LM Studio 实例。
- **Embedding 模型**：`@loganyang` 还询问了 LM Studio 是否支持语言模型 Embedding。社区成员没有给出直接回答，并表示该功能可能尚未实现。
- **模型错误**：`@pminev` 在 LM Studio 上运行特定模型时遇到问题。`@dagbs` 建议这可能是 GPU 相关的问题，并建议在禁用 GPU offloading 的情况下测试模型。
- **扩展 LM Studio**：`@pminev` 询问了在 LM Studio 中添加让语言模型调用用户自定义 API 功能的可能性。`@dagbs` 指出了 Discord 服务器中可以提交功能请求的地方。
      
提到的链接：

- [Chat Completions | docs.ST.app](https://docs.sillytavern.app/usage/api-connections/openai/#proxy)：Chat completion API 包括 OpenAI、Claude 和 PaLM。WindowAI 和 OpenRouter 也允许连接到这些 API。
- [cognitivecomputations/dolphin-2.5-mixtral-8x7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b/tree/main)

### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (22 messages🔥): 
        
- **模型讨论与性能**：用户 `@dagbs` 分享了 [**Dolphin 2.6 Mistral 7B GGUF** 模型](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF)，并指出其具有无偏见/无审查（unbiased/uncensored）的特性。用户 `@wordweaver` 对其进行了评估，发现所有 Mistral 变体都有适当的审查，特别提到 **perlthoughts/mistral-instruct-v0.2-2x7b-moe-q4_k_m.gguf** 模型表现良好，并认为不需要 **Mixtral 8x**，因为它速度太慢。
- **模型硬件需求**：`@a1vx` 讨论了 **Mixtral 8x7B Dolphin 2.6** 模型在他机器上运行缓慢的问题。`@dagbs` 澄清说该模型需要约 64GB 的 RAM，如果未加载到 VRAM 中，可能会因为 RAM 到 VRAM 的数据传输而变慢。
- **其他模型与查询**：`@kujila` 介绍了 [**MixtralOrochi8X7B GGUF** 模型](https://huggingface.co/TheBloke/MixtralOrochi8x7B-GGUF)，但 `@dagbs` 询问了其目标和训练细节，因为分享的链接中未列出这些信息。
- 用户 `@unskilless` 咨询了 CodeNinja 平台及其性能和配置细节。
- 用户 `@dedded___` 尝试合并两个模型并遇到了量化错误，但未提供错误的具体细节。
      
提及的链接：

- [TheBloke/dolphin-2.6-mistral-7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF)
- [TheBloke/MixtralOrochi8x7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/MixtralOrochi8x7B-GGUF)


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (8 messages🔥): 
        
- **在 LM Studio 中为 Embedding 模型运行本地服务器**：用户 `@loganyang` 询问是否有办法在 LM Studio 中为 Embedding 模型运行本地服务器，因为这将开启各种集成的可能性。`@vic49.` 确认 LM Studio 确实具有服务器模式。
- `@loganyang` 进一步表示，他在搜索时未能发现 [Hugging Face](https://huggingface.co) 上有可用的 Embedding 模型。
- **LM Studio 中 Embedding 模型的解决方案**：`@vic49.` 建议使用[他的非官方插件](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio)来使用 Embedding 模型。该插件创建了一个 ChromaDB 向量数据库，以便与运行在服务器模式下的 LM Studio 配合使用。
- `@dagbs` 进行了闲聊，询问 `@vic49.` 进入 Windows 环境下 AMD 领域的情况。用户 `@vic49.` 以道晚安结束了对话。
      
提及的链接：

- [sentence-transformers/all-mpnet-base-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio): 该插件创建了一个 ChromaDB 向量数据库，以便与运行在服务器模式下的 LM Studio 配合使用！

### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (39 条消息🔥): 
        
- **关于在有限硬件上加载大型模型的讨论**：`@senpaira6969` 提出了使用名为 air_llm 的工具在性能相对较低的笔记本电脑上运行 70B 模型的话题。`@alphalogic_` 加入了讨论，寻求关于如何利用其高端硬件在 LM Studio 中运行更大 Language Models 的建议，并对将 Language Models 与个人硬件配置匹配的难度表示担忧 ([阅读更多链接](https://discord.com/channels/1110598183144399058/1153759714082033735/1189065014292791376))。
- **不同硬件上的 Quantization**：`@pefortin` 和 `@totallybored` 等用户分享了在各种硬件配置上运行不同规模模型的经验。`@pefortin` 专门针对 `@alphalogic_` 分享的硬件配置，提供了运行 7B 到 70B 模型时的预期通用指南。
- **下一代 Intel CPU 的潜力**：`@funapple` 询问了带有集成 NPU 的下一代 Intel CPU 在仅使用 CPU 和 RAM 运行 30B+ 等大型模型方面的潜力。`@totallybored` 对利用 AI CPU 进行整体计算表示乐观，但强调需要等待对这些潜力进行更多测试。
- **在 LM Studio 中使用 GPU 运行模型的问题**：`@alphalogic_` 提出了尽管正确识别了 Nvidia driver，但在使用 LM Studio 在其设备的 GPU 上加载模型时遇到的问题。`@totallybored` 提供了帮助，建议将 `n_gpu_layers` 设置为 `-1`，将操作完全 Offload 到 GPU 的 VRAM，同时建议使用 8 线程的 CPU。`@rem_l` 也分享了类似的问题，以及通过使用 `n_gpu_layers` 选项和更改 CPU 线程数来解决的方法。
- **直接使用模型进行编码**：`@rem_l` 表达了直接在 Visual Studio Code 中使用 LM Studio 的满意度，称其在生成代码、寻找 Bug、调试以及创建复杂的搜索和替换正则表达式方面非常有用。他们还提到使用了几个扩展：`Continue` 和 `GPT-Pilot` 来辅助编码过程。
- **在 M.2 插槽上运行大型模型**：`@pefortin` 询问了通过适配器/转接卡（adapter/riser）将 3090 GPU 连接到 m.2 插槽的问题，并对模型加载和卸载时潜在的数据传输速度表示担忧。`@totallybored` 确认任何总线减速确实会影响性能。
- **AI CPU 的未来**：`@totallybored` 讨论了 AI CPU 的广阔前景，特别是 AMD 和 Intel 计划推出的新一代产品。然而，他们指出目前的型号仍为笔记本电脑专属。`@funapple` 对下一代 CPU 在无需 VRAM 即可运行大型模型方面的潜力表现出兴趣。
      
提到的链接：

- [Continue](https://continue.dev/): 
- [GitHub - Pythagora-io/gpt-pilot: Dev tool that writes scalable apps from scratch while the developer oversees the implementation](https://github.com/Pythagora-io/gpt-pilot): 在开发者监督实现的同时，从头开始编写可扩展应用程序的开发工具。


### ▷ #[🧪-beta-releases-discussion](https://discord.com/channels/1110598183144399058/1166577236325965844/) (4 条消息): 
        
- **搜索面板崩溃问题**：用户 `@kujila` 报告称，在几天前的一次输入后，搜索面板会崩溃并返回主屏幕。`@yagilb` 建议他们从网站**重新下载并重新安装**应用程序以解决此问题。按照建议操作后，`@kujila` 确认问题已解决。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- 讨论了 **Mixtral** 和 **Mistral** 模型的训练问题，包括增加 **top_k** 的负面影响、未对齐的 base model 生成随机文本、处理波动的 loss 以及遇到 CUDA out-of-memory 错误。直接引用：“*增加 top_k 可能会在专家未在训练期间遇到的 token 上激活它们，从而可能降低性能。*” `[general]`
- 询问了关于利用类似数据集进行 Preference-Based DPO、**Mixtral** 量化分析的成功案例，以及使用 LoRa adapters 进行 instruction tuning 的问题。 `[general][general-help]`
- 一个涉及 **tokenizing strategy** 和 **sequence length** 导致 CUDA 内存错误的问题，提出的解决方案是使用 `pad_to_sequence_len: true` —— 并提到将对此问题进行进一步探索。 `[axolotl-dev]`
- 建议在 YAML 中为特定数据集添加单独的 prompt format，并在数据集配置中更加明确，同时参考了 chat templates。提到了 `sample_packing` 导致意外 VRAM 峰值的问题，并确认在禁用 `sample_packing` 时训练运行正常。 `[axolotl-dev]`
- 关于 Mistral 7B 全权重 fine-tuning 的问题，澄清了这相当于 pretraining，并强调了巨大的 VRAM 需求（160GB）。还讨论了关于 forward 和 backward passes 是否都需要使用 flash attention 才能正常运行的查询。 `[general-help]`
- 报告了 `Mixtral-8x7B-v0.1` 和 `mistral-7b-v0.1` 模型之间的性能差异，指出使用 `mistral-7b-v0.1` 时 train 和 eval loss 有所改善，但未对原因做出推测。 `[general-help]`
- 在 'rlhf' 频道中讨论了关于 synthetic data generation 以及为 ragdata 集使用 llama index 的问题，尽管没有提供这些主题的详细对话或背景。 `[rlhf]`

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (16 条消息🔥): 
        
- **增加 top_k 的影响**：`@stefangliga` 指出了在训练中增加 top_k 的警告。增加 top_k 可能会在专家未在训练期间遇到的 token 上激活它们，从而可能**降低性能**。

- **为 Preference-Based DPO 使用相同数据集**：`@dangfutures` 询问是否应该为 Preference-Based DPO 使用相同的数据集。

- **Mixtral 的量化**：`@dangfutures` 询问了 **Mixtral** 量化分析的成功情况。

- **未对齐 Base Model 生成随机文本**：`@noobmaster29` 遇到未对齐的 base model 不管输入是什么都生成随机文本的情况，他们使用简单的短语如 'hello' 作为测试案例。

- **在 Spicyboro 数据集上训练**：`@dangfutures` 透露他们在 Spicyboro 数据集上训练了 **Mistral** 模型，幽默地暗示模型可能从该数据集中学到了潜在的敏感信息。

- **关于评估 Loss 和波动 Loss 的担忧**：`@mihai4256` 分享了他在训练 **LoRa model** 过程中遇到 loss 波动但 eval_loss 持续下降的经验。`_jp1_` 建议增加 batch size 并降低 learning rate 以处理波动，并指出只要 **eval_loss 在下降** 就没问题。`_jp1_` 还提倡立即使用 **wandb** 以更好地了解模型性能。

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (21 messages🔥): 
        
- **Tokenizing strategy 和 sequence length 问题**：用户 `@_jp1_` 报告了训练过程中与 **tokenizing strategy** 和 **sequence length** 相关的问题。用户观察到，超过定义的 `sequence_len` 且未被自动截断的序列会导致 CUDA 显存溢出 (OOM) 错误。该问题是在稳定环境下更换数据集时发现的。
- **关于长度过滤器的讨论**：从代码角度，`@caseus_` 提醒存在现有的 `drop_long filter`，它应该为所有数据集过滤掉超过定义长度的序列。然而，`_jp1_` 提到尽管如此，他们仍然遇到该问题，并推测手动删除序列可能影响了 packing，从而导致问题表面上消失了。
- **提议修改 prompt 格式**：用户 `@caseus_` 提出了在 yaml 中添加独立 prompt 格式的想法。这可以针对特定数据集，并默认使用该格式。还建议在数据集配置中更明确地使用 chat templates。
- **sample_packing 问题调查**：`_jp1_` 报告在启用 `sample_packing` 的特定条件下遇到了意外的 VRAM 峰值。与 `@nanobitz` 就潜在原因进行了讨论，但未达成定论。`_jp1_` 确认在禁用 `sample_packing` 时不会出现该问题。
- **探索潜在解决方案**：讨论的潜在解决方案包括使用 `pad_to_sequence_len: true` 代替 `sample_packing`。`_jp1_` 确认在启用 `pad_to_sequence_len` 的情况下训练正常。
- **进一步调查**：对话以 `_jp1_` 表示打算进一步探索此问题，以及 `@nanobitz` 提供支持而结束。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (36 messages🔥): 
        
- **训练期间的 OOM 错误**：`@_jp1_` 在使用新数据集训练 8k 序列长度的 7B Mistral 模型时遇到了 [OOM 错误](https://discord.com/channels/891714134014103552/912454685702639708/922369707724046336)。错误似乎在随机步骤发生，并且似乎与训练数据有关，因为模型在相同配置但不同数据集下运行正常。`@caseus_` 建议尝试将每个步骤从 data loader 读取的数据写入文件以便调试。然而，`_jp1_` 指出在不使用 sample packing 的情况下错误消失了。
- **关于全权重微调 (Full Weight Finetuning) 的讨论**：`@athenawisdoms` 询问了 Mistral 7B 全权重微调的 VRAM 需求，以及全权重微调是否等同于持续预训练。`@noobmaster29` 澄清全量微调即是预训练，且所有 Axolotl 的全量微调配置均使用 bf16。讨论还强调了预训练 7B 模型需要巨大的 VRAM (160GB)，这意味着需要像 A100 这样的高端 GPU。
- **预训练与应用 LoRa 适配器**：`@athenawisdoms` 还询问是否可以将原始的 `Llama-2-13B-Chat-fp16` LoRa 适配器应用于预训练的 `Llama-2-7B-fp16` 模型，并仍能获得相同的指令微调效果。
- **Flash Attention 查询**：`@xzuyn` 询问前向传播和反向传播是否都需要利用 Flash Attention 才能正常工作。`@caseus_` 确认两者都必须使用，因为反向传播依赖于前向传播的计算图。
- **Mixtral 与 Mistral 性能对比**：`@semantic_zone` 报告称，将基础模型从 `Mixtral-8x7B-v0.1` 更改为 `mistral-7b-v0.1` 显著改善了其大型分类数据集上的训练和评估 loss。他们请求推测这背后的原因。
      
提及的链接：

- [axolotl/examples/mistral/config.yml at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/config.yml): 欢迎提出关于 axolotl 的问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [GitHub - hiyouga/LLaMA-Factory: Easy-to-use LLM fine-tuning framework (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM)](https://github.com/hiyouga/LLaMA-Factory): 易于使用的 LLM 微调框架 (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM)


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (1 messages): 
        
emperor: 通常人们两者都用

### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (3 条消息): 
        
- **数据生成器脚本咨询**：`@dangfutures` 询问是否有人有合成数据生成的脚本。目前没有收到回复或进一步的细节。
- **提及 Self RAG**：`@dangfutures` 提到了 "self rag"，但上下文不明确，随后没有进一步讨论。
- **在 RAG 数据集中使用 Llama Index**：`@dangfutures` 报告称他们正在将 Llama Index 用于 RAG 数据集和 RAG 微调示例。未提供链接或额外信息。


        

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- 讨论了 **HuggingFace 的新频道**，用于每日模型更新和 Hinglish 模型的社区训练，以及多项**开源更新**，包括发布了具有新功能和模型的 Diffusers v0.25 和 Transformers.js v2.13。
- 提到了**模型性能增强**，由于集成了原生 SDPA 和用于 STFT 的 Torch 后端，Whisper 的速度提升了约 40%；Mixtral Instruct 更新了 4-bit 序列化。
- 该社区积极探索了将 **Weights and Biases (wandb)** 作为机器学习项目追踪工具，讨论了 **GGUF 和 GPTQ 模型格式**之间的区别，以及网站加载和设置自定义训练循环的挑战。
- 一项值得注意的讨论集中在 AI 论文内容的管理上，强调将快速阅读作为建议策略，并警告不要使用 LLM 进行摘要提取，因为可能存在准确性问题。
- 在 today-im-learning 频道分享了学习项目，包括从零开始构建 **GPT 模型**以及在 **Docker 容器**中设置 **vLLM 服务器**。
- NLP 频道的活跃对话围绕 Mystral embedding 检索性能、encoder-decoder 与 decoder-only 架构模型的比较、**模型容量**、**数据标注工具**的完成，以及使用 Langchain 加载 GPTQ 模型的问题展开。
- Diffusion 相关讨论话题涉及 Pokemon 数据集、将 diffusers 转换为单个 safetensors、SHAP-E 模型介绍、使用深度条件控制的 ControlNet 权重，以及关于人脸训练方法和 textual inversion 的问题。
  
分享的资源：

- [Diffusers v0.25 发布](https://github.com/huggingface/diffusers/releases/tag/v0.25.0)
- [Transformers.js v2.13 更新推文](https://twitter.com/xenovacom/status/1740037798650859755?s=46)
- [以及更多...](https://huggingface.co/blog/hwaseem04/drag-gan)

**HuggingFace Discord 频道总结**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 条消息): 
        
- **新频道**：HuggingFace 推出了两个新频道。<#1184447251746127913> 每天会分享一个新模型及其相关信息和代码，<#1189605147068858408> 是一个旨在训练 Hinglish 模型的社区项目。
- **开源更新**：Diffusers v0.25 已发布，包含多种功能，其中包括 aMUSEd，一个轻量级且快速的文本生成图像模型 [发布链接](https://github.com/huggingface/diffusers/releases/tag/v0.25.0)。Transformers.js v2.13 已发布，包含许多更新，包括可在浏览器中使用的 SegFormer 和 VITS [推文链接](https://twitter.com/xenovacom/status/1740037798650859755?s=46)。 
- **模型性能增强**：由于集成了原生 SDPA (Scaled Dot Product Attention) 和用于 STFT (Short-Term Fourier Transform) 的 Torch 后端，Whisper 现在的速度提升了约 40%，详见此 [推文](https://twitter.com/reach_vb/status/1739358185994047524)。Mixtral Instruct 已更新，支持 4-bit 序列化 [模型链接](https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit)。
- **社区更新**：MLX 社区已将预转换的 MLX 模型上传到 Hub，HuggingFace 发布了冬季版 Fellow Highlights [亮点链接](https://huggingface2.notion.site/Hugging-Face-Fellows-Highlights-Winter-Edition-b26c2c7d3f9143ec88d98ec43b98af29)。
- **阅读推荐**：分享了几篇博客文章，包括《Speculative Decoding 让 Whisper 推理速度提升 2 倍》[博客链接](https://huggingface.co/blog/whisper-speculative-decoding)、《构建 AI Chatbot 以运行代码并调整图表》[博客链接](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat) 以及《应对 LLM 中的评估数据污染：高质量微调和模型合并策略》[博客链接](https://huggingface.co/blog/rishiraj/merge-models-without-contamination)。
      
提到的链接：

- [Release v0.25.0: aMUSEd, 3x faster SDXL, interruptable pipelines · huggingface/diffusers](https://github.com/huggingface/diffusers/releases/tag/v0.25.0)：aMUSEd 是一个基于 MUSE 架构的轻量级文本生成图像模型。aMUSEd 在需要轻量且快速模型的应用中特别有用，例如生成 m...
- [Xenova (@xenovacom) 的推文](https://twitter.com/xenovacom/status/1740037798650859755?s=46)：🤗 Transformers.js v2.13 - 节日更新！☃️ 在此版本中，我们添加了：
1. 用于语义分割和图像分类的 SegFormer。
2. 用于多语言文本转语音（支持 >1000 种语言）的 VITS。
3. 用于零样本图像分割的 CLIPSeg。
4. 用于表格提取的 Table Transformer。
5. 用于文档图像分类的 DiT。
6. 用于零样本图像分类的 SigLIP。
7. 用于掩码语言建模、序列分类、标记分类和问答的 RoFormer。
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://twitter.com/reach_vb/status/1737905584089846108)：@mozilla 的 Common Voice 16 已在 Hub 上发布！🔥 这带来了总计 30,328 小时的音频，涵盖 120 种语言！在总计 3 万小时的音频中，有 1.95 万小时已通过验证！✨
- [dylan (@dylan_ebert_) 的推文](https://twitter.com/dylan_ebert_/status/1736857719620161895)：🚀 宣布 gsplat.js - 一个 JavaScript Gaussian Splatting 库 - 更新 1.0
- [younes (@younesbelkada) 的推文](https://twitter.com/younesbelkada/status/1739244971905966380)：继社区在 bitsandbytes 4-bit 序列化方面取得巨大成果后，我在 @huggingface 上发布了 Mixtral-Instruct-bnb-4bit，方便大家轻松加载模型：
https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://twitter.com/reach_vb/status/1739358185994047524)：我们让 Whisper 变得更快了。快了约 40%！！🔥
- [Notion – 笔记、任务、维基和数据库的一体化工作空间。](https://huggingface2.notion.site/Hugging-Face-Fellows-Highlights-Winter-Edition-b26c2c7d3f9143ec88d98ec43b98af29)
- [Awni Hannun (@awnihannun) 的推文](https://twitter.com/awnihannun/status/1737510739987120248)：Hugging Face 🤗 的团队制作了一系列预转换的 MLX 模型！包括 Llama, Phi-2, Mistral, Mixtral（以及可用的 instruct 和 code 变体）！
- [Speculative Decoding 让 Whisper 推理速度提升 2 倍](https://huggingface.co/blog/whisper-speculative-decoding)
- [构建 AI Chatbot 以运行代码并调整图表](https://huggingface.co/blog/sophiamyang/tweak-mpl-chat)
- [应对 LLM 中的评估数据污染：高质量微调和模型合并策略](https://huggingface.co/blog/rishiraj/merge-models-without-contamination)
- [Drag GAN - 生成图像流形上基于交互点的操作](https://huggingface.co/blog/hwaseem04/drag-gan)

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (42 messages🔥): 
        
- **Weights and Biases (wandb)**：针对 `@bennoo_` 的提问，`@krebzonide` 和 `@natika1` 解释说 wandb 是一个用于 Machine Learning 项目的在线追踪工具，对于评估模型性能非常有用。

- **不同模型格式的区别**：`@jiha` 发起了关于不同模型格式的讨论，`_sabine_wren_` 详细对比了 **GGUF** 和 **GPTQ**。GGUF 用于 LLAMA 模型并优先考虑 CPU 兼容性，而 GPTQ 针对 GPU 的效率进行了优化，并利用了 quantization 方法。

- **平台加载问题**：用户 `@rk_man` 和 `@zhangjie_shanghai` 讨论了他们在加载平台时遇到的困难，`@zhangjie_shanghai` 建议使用 Chrome 浏览器作为该问题的解决方案。

- **创建自定义训练循环**：`@goodtimes5241` 就音频生成项目的 Stable Diffusion 模型微调过程寻求建议，特别是询问关于创建训练循环和保存权重以便在 Stable Diffusion pipeline 中使用的资源。

- **关于阅读 AI 论文的讨论**：`@tonyaichamp` 发起了一个关于如何管理海量发布的 AI 论文的对话，并询问推荐哪些平台来发现、阅读和总结此类内容。`.naptastic` 建议采用速读，并指出不应使用 LLM (Language Model) 来总结论文，因为可能存在准确性问题。


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (2 messages): 
        
- **从零开始构建 GPT**：`@gag123` 分享了他们根据 **AndrejKarpathy** 提供的教程学习从零构建 **GPT** 的项目进展。他们反映在达到预期效果方面遇到了困难，模型的 Loss 持续波动且输出无意义。为了寻求参考或帮助，他们提供了 [repository](https://github.com/gagangayari/my-gpt) 链接和 **AndrejKarpathy** 的 [教程系列](https://www.youtube.com/watch?v=kCc8FmEb1nY)。
- **在 Docker 容器中设置 vLLM 服务器**：`@warmonger9626` 讨论了他们在 Windows 机器的 WSL 环境下，利用本地 CUDA GPU，成功在 Docker 容器内运行 **默认 vLLM 服务器** 的设置。


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (9 messages🔥): 
        
- **Mistral Embedding 检索性能**：`@nickprock` 提出了关于 **Mistral** embedding 在 MTEB 上检索分数较高但在排行榜上未体现的问题。`@merve3234` 建议可能是因为他们还没有提交结果。
- **Encoder-Decoder 与 Decoder-Only 架构模型的比较**：`@hieunguyen1053` 询问了拥有 1.6B 参数并在 600GB 数据集上进行预训练的 T5 等 Encoder-Decoder 模型的可行性和潜在性能。他们还想知道此类模型在经过 instruction data 微调后，与具有同等参数的 Decoder-Only 模型相比表现如何。`@merve3234` 观察到此类模型之间的差异可能更多取决于数据集输入。
- **关于模型容量的讨论**：`@hieunguyen1053` 对 1.6B 参数模型记忆 600GB 数据集的能力表示怀疑，并引用了他们在终端产品上的测试。该用户强调目前尚不清楚是否采用了 retrieval 过程。
- **数据标注工具**：`@stroggoz` 宣布完成了用于 Named Entity Recognition 的数据标注工具，但表示在让该工具在其他计算机上运行方面存在困难。
- **使用 Langchain 加载 GPTQ 模型的问题**：`@.sgp` 发布了一个关于如何使用 Langchain 加载 GPTQ 模型的问题。他们分享了 Python 代码和遇到的错误，即一个与 NumPy 中 'object' 属性问题相关的 RuntimeError。错误消息引导用户查看 NumPy 1.20 关于 deprecations 的发行说明。

      
Links mentioned:

- [NumPy 1.20.0 Release Notes &#8212; NumPy v2.0.dev0 Manual](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations)

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (9 条消息🔥): 
        
- **Pokemon 数据集建议**：`@sayakpaul` 提供了一个在 HuggingFace 上可用的图像-标题（image-caption）示例数据集链接：[pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)。
- **将 diffusers 转换为单个 safetensors**：`@sayakpaul` 建议使用 [diffusers GitHub repo](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) 中的脚本将 diffusers 转换为单个 safetensors，以便无缝应用于 VEGA。
- **SHAP-E 介绍**：`@sayakpaul` 指向了 HuggingFace 上的 [SHAP-E model](https://huggingface.co/openai/shap-e)，这是一个能够根据 text prompt 生成 3D 图像的 diffusion 过程。
- **深度条件控制的 controlnet 权重**：`@sayakpaul` 还分享了一个基于 stabilityai/stable-diffusion-xl-base-1.0 训练的、带有 depth conditioning 的 [controlnet weights](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid) 资源。
- **关于面部训练方法和 textual inversion 的疑问**：`@matthewcroughan` 提出了关于 textual inversion 对面部训练效果的问题，并询问是否有各种训练方法的总结。
      
提及的链接：

- [openai/shap-e · Hugging Face](https://huggingface.co/openai/shap-e): 
- [diffusers/controlnet-depth-sdxl-1.0-mid · Hugging Face](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-mid): 
- [lambdalabs/pokemon-blip-captions · Datasets at Hugging Face](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions):


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- 围绕不同 Mistral 模型中 **prompt templates 的限制与能力**展开了热烈讨论。一位用户指出，他们的自定义模板在 small 和 medium 模型之间产生了相当的结果，但更偏向于 medium。引用内容："*...custom prompt template gives comparable results between small and medium models...*"
- 用户分享了项目资源，例如用于在 4GB VRAM 上运行 70B 模型的 **AirLLM project**，尽管存在性能方面的担忧。信息中包含了 **[AirLLM GitHub repository](https://github.com/lyogavin/Anima/tree/main/air_llm)**。
- 强调了对 **兼容 Mistral API 的免费聊天 UI** 的需求。用户 `@fayiron` 推荐了最近增加对 Mistral API 支持的 **sillytavern**。
- 开源项目 **Microchain** 受到关注，该项目现在支持 Mistral，改进了 token 使用指标，并优化了 Agent。分享了该项目的 **[GitHub repository](https://github.com/galatolofederico/microchain)**。
- 有关于 **在 12GB VRAM 上运行 7B 模型** 可行性的提问。推荐的解决方法是使用较低分辨率的模型，如 **openchat 3.5**，并附带了[模型链接](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main)。
- 出现了关于 **模型过拟合以及基础 instruct 模型与 fine-tuned 模型对比** 的问题。在推荐 **[text-generation-webui](https://github.com/oobabooga/text-generation-webui)** 等本地运行工具的同时，还讨论了 **Goliath 120B** 和 **OpenHermes 2.5 Mistral 7B** 等替代方案。
- 用户询问了在使用 Mistral API 时，使用 [**MistralAI's Python Client**](https://github.com/mistralai/client-python) 相比 [**OpenAI Python package**](https://github.com/openai/openai-python) 的优势，共识是两者都适用，取决于具体需求。
- 还有关于 **Mistral Medium 在编码任务上的性能反馈**，一位用户表示满意但希望有速度优化。提到："*...Mixtral Medium averaged at 15 tokens/s and peaked at 45 tokens/s.*"

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (19 messages🔥): 
        
- **Prompt Template 的限制与能力**：`@The Ledger Luminary` 试图定义与不同 Mistral 模型配合使用的 Prompt Template 中 **"complex"（复杂）** 一词的含义。他们表示，其自定义的 Prompt Template 在小型和中型模型之间给出了相当的结果，但由于中型模型的输出质量更高，因此略微偏向于中型模型。

- **适用于 70B 模型的 AirLLM**：`@unskilless` 介绍了一个 **AirLLM** 的 GitHub 仓库，该项目旨在通过 4GB VRAM 运行 70B 模型（[仓库链接](https://github.com/lyogavin/Anima/tree/main/air_llm)）。然而，`@peasantry ⚒` 发现该应用运行缓慢，并对潜在的 **malware** 风险表示担忧。

- **适用于 Mistral API 的免费 Chat UI**：`@mka79` 征求与 Mistral API 兼容的免费 Chat UI 建议。`@fayiron` 建议了 **sillytavern**，该工具最近增加了对 Mistral API 的支持。

- **改进 Microchain 开源项目**：`@.tanuj.` 一直在增强开源项目 **Microchain**，通过添加 Mistral 支持、更好的 Token 使用指标以及优化 Agent 以按预期执行任务（[仓库链接](https://github.com/galatolofederico/microchain)）。

- **系统配置建议**：针对 `@aldt440` 关于推荐系统配置的询问，`@orabazes` 建议使用多块 **3090** 以获得性价比。然而，对于 Mistral 7B 模型的 Inference，他们提到 **最低系统要求** 就足够了。


Link mentioned: [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm): 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (8 messages🔥): 
        
- **在有限的 VRAM 上运行 7b 模型**：`@azetisme` 询问是否可以在 12GB VRAM 上运行 **7b 模型**，尽管通常需要 16GB。
- 共识是，使用这些规格运行完整的 7b 模型是不可能的，`@ethux` 建议使用较低分辨率的模型，如 **openchat 3.5**。并提供了一个该模型的链接：[`openchat-3.5-1210-GGUF`](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main)，可用于文本生成、Transformer、GGUF、Mistral、Open Chat 和 C-RLFT。
- `@azetisme` 感谢 `@ethux` 的建议，并决定尝试所建议的选项。

Link mentioned: [TheBloke/openchat-3.5-1210-GGUF at main](https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/tree/main):


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
kushagra4761: 有没有关于在多 GPU 上对 Mistral 7b 进行 Fine-tuning 的指南？

### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (21 条消息🔥): 
        
- **API 调用格式化**：用户 `@bam4d` 澄清了如何使用正确的 `{ "role": "user", "content": "message"}` 格式调用 API，并提到在集成原始下载的 Instruct 模型（如 Mistral-7B-Instruct-v0.1）时，代码需要包含 start/stop/instruct tokens。

- **模型过拟合**：用户 `.gue22` 提出了关于模型是否可能过拟合的问题。其他几位用户建议尝试不同版本的模型，例如 [Goliath 120B](https://huggingface.co/alpindale/goliath-120b) 和 OpenHermes 2.5 版本的 [Mistral 7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) 以改进结果。

- **正确的 Prompt 格式**：用户 `@fayiron` 强调了使用正确 Prompt 格式的重要性并分享了一个示例。这可能影响了 `.gue22` 使用 Mistral 的结果，并有助于在机器学习实验早期缓解输出乱码的问题。

- **模型对比**：`.gue22` 和 `.tanuj.` 之间就基础 Instruct 模型与微调模型的对比展开了有趣的讨论。`.tanuj.` 建议将运行 4-bit 量化的 OpenHermes-2.5-Mistral-7B 作为一个良好的对比基准。

- **本地模型运行**：`@fayiron` 推荐使用工具 [text-generation-webui](https://github.com/oobabooga/text-generation-webui) 在本地运行模型，`.gue22` 对探索该建议表现出兴趣。
      
提到的链接：

- [mistralai/mistral-7b-instruct-v0.1 – 在 Replicate 上通过 API 运行](https://replicate.com/mistralai/mistral-7b-instruct-v0.1)
- [GitHub - rbgo404/OpenHermes-2.5-Mistral-7B](https://github.com/rbgo404/OpenHermes-2.5-Mistral-7B)：OpenHermes 2.5 Mistral 7B 是 OpenHermes 2 模型的进阶版本，这一增强提升了在 TruthfulQA、AGIEval 和 GPT4All 套件等多个非代码基准测试中的表现。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (3 条消息): 
        
- **Mistral API 客户端库讨论**：用户 `@jakobdylanc` 询问了在处理 Mistral API 时，使用 [MistralAI's Python Client](https://github.com/mistralai/client-python) 相比 [OpenAI Python package](https://github.com/openai/openai-python) 的目的和优势。用户 `@lerela` 澄清说，虽然 Mistral 客户端更轻量且主要专注于 completions，但 OpenAI 客户端支持许多额外功能。因此，如果应用程序已经在使用 OpenAI Python 包，则可以继续使用。

- **对 Mixtral Medium 的反馈**：用户 `@casper_ai` 对 Mixtral Medium 在编程任务中的表现表示赞赏，强调其性能优于 Mixtral Small 和 GPT 3.5。然而，`@casper_ai` 希望能进行速度优化，因为 Mixtral Medium 平均速度为 15 tokens/s，峰值为 45 tokens/s。
      
提到的链接：

- [GitHub - mistralai/client-python: Mistral AI 平台的 Python 客户端库](https://github.com/mistralai/client-python)

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain 0.1 文档更新**：用户 `hwchase17` 发起了关于 LangChain 文档更新的重要讨论，邀请社区提供反馈。文档更新包括快速入门指南、概念指南等的新增内容，以及 LCEL、LLMs、ChatModels、Agents 的新页面，并计划未来更新 Use Cases 文档。征求关于需要改进的 API 文档组件的反馈，并寻求进一步增补的建议。社区对此更新的参与可以通过 [GitHub 链接](https://github.com/langchain-ai/langchain/pull/15281)和 [Vercel 预览](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction)进行。
- **LangChain 使用与数据集**：小组讨论围绕 LangChain 数据集建议、GPU 利用率与优化、理解 LangChain 在处理大量参数时的用途以及 LangChain 中 Embeddings 的使用展开。值得注意的是，`rahuldey8431` 提供了在 LangChain 中使用 FireworksEmbeddings 的指南，并辅以相关的 [Embedding 模型教程](https://python.langchain.com/docs/modules/data_connection/text_embedding/)。
- **加密货币诈骗通知**：`justignorethesharks` 提出了一个相关问题，关注社区中存在的加密货币诈骗，这正在降低社区信任并需要采取必要行动。
- **社区招聘公告**：`tanny1208`、`dougdotcon` 多次发布招聘信息，并附带相同的 web3job Discord 邀请链接。这在 '#langserve'、'#langchain-templates' 和 '#share-your-work' 频道中均有出现，表明社区中存在普遍的招聘驱动。
- **共享作品与资源**：'#share-your-work' 频道还展示了一些共享项目，包括由 `aabbhishekk0804` 使用 Agent 框架构建并分享的交互式应用，应用链接在[此处](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant)；以及由 `rajib2189` 分享的创建 Bedrock Agent 的综合资源指南，可在[此处](https://youtu.be/OcMXPFZ5gbs)访问。
- `rameshvoodi` 在 '#tutorials' 频道中记录了一次无关紧要的互动，未提供相关信息。

**LangChain AI 频道总结**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 条消息): 
        
- **LangChain 0.1 文档更新**：`hwchase17` 宣布了 **LangChain** 文档的更新，寻求社区反馈。更新内容可通过[此链接](https://github.com/langchain-ai/langchain/pull/15281)和 [Vercel 预览](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction)访问。
- **文档改进**：快速入门指南、概念指南以及针对 Output Parsers、Agents 和高级检索方法的“何时使用”表格均有新增内容。作为更新的一部分，过时的页面已被移除。
- **即将到来的更新**：未来的更新将包括创建自定义 LLM、ChatModel、Retriever、VectorStore、Agent 和 Tool 的操作指南。Use Cases 文档的更新也在计划中。
- **社区反馈征集**：团队正在征求关于哪些 API 文档部分需要改进以及哪些集成页面需要更新的反馈。他们还为 LCEL、LLMs、ChatModels、Agents 添加了专用页面，并正在寻求其他增补建议。
- **Prompt Engineering 技术更新**：作为更新的一部分，快速入门指南中加入了 OSS 模型。团队承认使用本地模型仍然具有挑战性。

      
提到的链接：

- [[documentation] documentation revamp by hwchase17 · Pull Request #15281 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/pull/15281)：需要新版本的 langchain-core 和 langchain
- [Introduction | 🦜️🔗 Langchain](https://langchain-kdnner3pi-langchain.vercel.app/docs/get_started/introduction)：LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它使应用程序能够：

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (19 messages🔥): 
        
- **LangChain 数据集与 GPU 利用率**：`@asterix3651` 询问了包含参考摘要和 LLM 摘要的数据集建议。`@breathesmall` 提出了在使用 LangChain 和 Ollama 时 GPU 利用率的问题。
- **加密货币诈骗通知**：`@justignorethesharks` 敦促社区（并点名了几位用户）清理所有频道中的加密货币诈骗内容。他指出，由于团队缺乏活动和沟通，社区中似乎存在相当大的不信任感。
- **LangChain 用于大参数量模型**：`@refik0727` 询问了使用 LangChain 构建参数量在 3B 及以上的 LocalGPT 或自有 LLM 所需的 RAM 和 GPU。
- **在 LangChain 中结合 Fireworks 使用 Embeddings**：`@3h0480` 询问了如何在 LangChain 中结合 Fireworks 使用 Embeddings。作为回应，`@rahuldey8431` 提供了关于如何集成 FireworksEmbeddings 的详细指南，尽管 `@3h0480` 随后遇到了执行错误。
- **InMemoryCache 位置查询**：`@seththunder` 询问了 InMemoryCache() 保存的具体位置，以及如何为每个用户创建唯一的缓存。
      
提及的链接：

- [Text embedding models | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/data_connection/text_embedding/)：前往 Integrations 查看关于文本嵌入模型提供商内置集成的文档。


### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (4 messages): 
        
- **招聘机会**：`@tanny1208` 和 `@dougdotcon` 都发布了相同的招聘信息，链接指向一个针对 Web3job 的 [Discord 邀请](https://discord.com/invite/web3job)。
- **问题与不活跃成员**：`@cryptossssun` 询问是否有人可以回答问题，并艾特了两位未露面的用户以引起关注。


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (5 messages): 
        
- `@tanny1208` 和 `@dougdotcon` 宣布了一个招聘机会，并附带了 Discord [Web3job](https://discord.com/invite/web3job) 的链接。

- `@aabbhishekk0804` 分享了一个在 Huggingface space 上部署的应用。该应用可以回答文档相关查询，并处理需要 Search APIs 的查询。该应用是使用 Agent 框架构建的。点击[此处](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant)查看。

- `@rajib2189` 提供了一个关于创建带有 Action Group 的 Bedrock Agent 的有用资源，分享了 [YouTube 教程](https://youtu.be/OcMXPFZ5gbs)链接和 [Medium 博客文章](https://medium.com/@rajib76.gcp/aws-bedrock-agent-part-4-action)。
      
提及的链接：

- [ChatPdfAgent - a Hugging Face Space by Aabbhishekk](https://huggingface.co/spaces/Aabbhishekk/Chat-Pdf-With-Search-Assistant)
- [AWS Bedrock Agents | Action Groups](https://youtu.be/OcMXPFZ5gbs)：在此视频中，我展示了如何将 Action Group 关联到 Bedrock Agent。Medium 博客：https://medium.com/@rajib76.gcp/aws-bedrock-agent-part-4-action-...

        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- 在 #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) 频道讨论了 **LLM 的效率**。`@slono` 分享了 LLM 带来的显著生产力提升经验，引发了 `@guardiang` 对其身份的幽默猜测。`@swizec` 对 `@slono` 使用的生产力指标表现出兴趣。

- #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) 频道的对话围绕 Hugging Face 上的 Apache-2.0 许可数据集展开。`@swyxio` 分享了 **Evol Instruct 论文和指令微调数据集** [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b) 和 [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4) 的链接。`@swyxio` 和 `@eugeneyan` 分别提到了术语 **"ragtuning"** 以及 **"finetuning 数据集与论文"**，但未提供额外背景。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (4 messages): 
        
- **LLM 的效率**：`@slono` 分享了他们的经验，认为 **LLM 为他们带来了 10-20 倍的生产力提升**（与日常工作职责相比），导致 `@guardiang` 开玩笑地询问 `@slono` 是否是 AI。
- **使用 LLM 衡量生产力**：`@swizec` 对 `@slono` 提到的指标表现出兴趣，询问这些生产力衡量标准是如何达成的。

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (3 条消息): 
        
- **Evol Instruct 论文与 Instruction Tuning 数据集**：`@swyxio` 分享了 Hugging Face 上的两个数据集链接，这两个数据集的许可证都已更改为 Apache-2.0。分享的具体数据集链接为 [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b) 和 [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4)。
- `@swyxio` 提到了 **"ragtuning"** 一词，但未提供任何额外的上下文或信息。
- `@eugeneyan` 提到了 **"finetuning datasets and paper"**，但未提供任何相关的链接或补充信息。

      
提到的链接：

- [Update README.md · theblackcat102/evol-codealpaca-v1 at 01d1e3c](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1/commit/01d1e3c73617c24513046eb21259e28271a7c77b): 
- [Change license to Apache-2.0 · ise-uiuc/Magicoder-Evol-Instruct-110K at b0079be](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/commit/b0079beaa0361d82412520b873715bee59cc7dd4):


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

只有一个频道有活动，因此无需总结...

- **Mixtral 的 LORA 或全量 Finetuning 设置**：用户 `@huevosabio` 询问了关于 Mixtral 的 **LORA** 或完整 **finetuning** 的首选设置建议，包括 **template code**、**GPU provider** 等。他们表示，考虑到最近没有进行过正式的训练，预计现在应该已经有相关的 **guides** 可用了。
        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

只有一个频道有活动，因此无需总结...

- **总结 PDF 书籍并解释图像中的文本**：`@codermickey` 询问是否有推荐的工具、prompts 或插件来总结 PDF 书籍。该用户还询问了在总结过程中读取和解释图表等图像中文本的方法。
        

---
Alignment Lab AI Discord 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。