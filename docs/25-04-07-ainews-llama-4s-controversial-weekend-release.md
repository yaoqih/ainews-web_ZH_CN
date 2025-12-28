---
companies:
- meta
date: '2025-04-08T01:55:40.760246Z'
description: '**Meta** 发布了 **Llama 4**，推出了两款新型中等规模的 MoE（混合专家）开源模型，并承诺将推出一个拥有 2 万亿参数的“巨兽级”模型，力争成为史上最大的开源模型。


  此次发布采用了多种先进的训练技术，包括结合 MetaCLIP 的类 Chameleon 早期融合（early fusion）技术、不带 RoPE 的交错分块注意力机制、原生
  FP8 训练，以及在高达 40 万亿个 token 上的训练。尽管备受关注，但此次发布也面临批评，原因包括透明度不如 Llama 3、实现过程中的问题以及在部分基准测试中表现不佳。


  包括 **Ahmad Al Dahle** 在内的 Meta 领导层否认了有关在测试集上进行训练的指控。最小的 Scout 模型拥有 1090 亿参数，对于消费级
  GPU 而言体积过大，且其声称的 1000 万 token 上下文长度也引发了争议。社区对此反应不一，部分人称赞其开放性，而另一部分人则指出了其中的差异和质量问题。'
id: d3dcb7f4-723e-4d1b-92a9-407b70e2d4f5
models:
- llama-4
- llama-3
- llama-3-2
original_slug: ainews-llama-4s-controversial-weekend-release
people:
- ahmad_al_dahle
- ylecun
- reach_vb
- yuchenj_uw
title: Llama 4 备受争议的周末发布
topics:
- mixture-of-experts
- early-fusion
- attention-mechanisms
- fp8-training
- training-data
- benchmarking
- model-performance
- model-release
- multimodality
- open-models
---

<!-- buttondown-editor-mode: plaintext -->**透明度与耐心是我们所需要的一切。**

> 2025年4月4日至4月7日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**229** 个频道，**18760** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**1662 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 的头条新闻光彩夺目：2 款表现优异的新型中型 MoE 开源模型，以及承诺中的第三款拥有 2 万亿参数的“巨兽”，它应该是史上发布的最大开源模型，恢复了 Meta 在排行榜顶端的地位：


![image.png](https://assets.buttondown.email/images/4d1ed552-8287-4988-9401-68d0456e7d6b.png?w=960&fit=max)


[SOTA 训练更新](https://x.com/iscienceluvr/status/1908601269004230763)总是备受欢迎：我们注意到采用了类 [Chameleon](https://techxplore.com/news/2024-05-meta-chameleon-early-fusion-multimodal.html) 的 [MetaCLIP](https://arxiv.org/abs/2309.16671) 早期融合（early fusion）、交织、[分块（chunked）](https://fxtwitter.com/nrehiew_/status/1908617547236208854) [无 RoPE 注意力机制](https://arxiv.org/abs/2305.19466)（[许多人](https://fxtwitter.com/nrehiew_/status/1908598863365013823)对此发表了评论）、[原生 FP8 训练](https://x.com/EMostaque/status/1908960936658141546)，并使用了 [高达 40T tokens](https://fxtwitter.com/maximelabonne/status/1908603628828451127) 进行训练。

虽然闭源模型实验室往往设定了前沿，但 Llama 通常为开源模型设定了标准。[Llama 3 发布于大约一年前](https://ai.meta.com/blog/meta-llama-3/)，随后的更新如 [Llama 3.2](https://buttondown.com/ainews/archive/ainews-llama-32-on-device-1b3b-and-multimodal/) 同样广受好评。

抛开[惯常的许可证争议](https://fxtwitter.com/maximelabonne/status/1908602756182745506)不谈，Llama 4 的反响基调明显不同。

1. Llama 4 在周六发布，比预期的要早得多，甚至似乎连 Meta 也是如此，其[在最后一刻将发布日期从周一提前](https://x.com/teortaxestex/status/1908706840554197309?s=46)。Zuck 的官方说法只是它已经“准备好了”。
2. 只有[博文](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，在透明度方面远未达到 Llama 3 论文的水平。
3. 最小的“Scout”模型也有 109B 参数，无法在[消费级 GPU](https://x.com/JeffDean/status/1908608454216028222) 上运行。
4. [声称的 10m token 上下文几乎肯定远高于使用 256k tokens 训练时的“真实”上下文](https://fxtwitter.com/burkov/status/1908658566887596475)（虽然仍然令人印象深刻！但不是 10m！）
5. LMarena 使用了一个特殊的“实验”版本，这导致了高分——但这并不是发布的版本。这种差异[迫使 LMarena 通过发布完整的评估数据集来做出回应](https://x.com/lmarena_ai/status/1909397817434816562)。
6. 它在 Aider 等独立基准测试中[表现非常糟糕](https://fxtwitter.com/paulgauthier/status/1908976568879476843)。
7. [中国社交媒体上未经证实的帖子声称公司领导层为了达到 Zuck 的目标而推动在测试集上进行训练（training on test）](https://x.com/suchenzang/status/1909070231517143509?s=46)。

最后一点已被 [Meta 领导层断然否认](https://x.com/Ahmad_Al_Dahle/status/1909302532306092107)：

![image.png](https://assets.buttondown.email/images/4d5f7b00-fef1-4aa3-9618-8918312f5942.png?w=960&fit=max)


但这种发布过程中出现问题的气息无疑给原本属于开源 AI 界的喜庆日子蒙上了阴影。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**大语言模型 (LLMs) 与模型发布**

- **Llama 4 与实现问题**：[@Ahmad_Al_Dahle](https://twitter.com/Ahmad_Al_Dahle/status/1909302532306092107) 表示 **Meta** 已注意到不同服务在使用 **Llama 4** 时报告的质量参差不齐，并预计实现将在几天内趋于稳定，同时否认了在测试集上进行训练的说法。[@ylecun](https://twitter.com/ylecun/status/1909313264460378114) 指出需要对 Llama-4 进行一些澄清，[@reach_vb](https://twitter.com/reach_vb/status/1909316136526832054) 感谢 [@Ahmad_Al_Dahle](https://twitter.com/Ahmad_Al_Dahle) 的澄清以及对开放科学和权重的承诺。
- **Llama 4 性能与基准测试**：关于 **Llama 4** 输出质量的担忧已经浮现，[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1909062763789566100) 报告称其生成内容质量低劣（slop），但其他人则认为表现良好。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1909061004207816960) 强调了一个 Reddit 帖子，并表示如果 **Meta** 实际上是为了最大化基准测试（benchmark）分数而进行训练，那么“就完蛋了”。[@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1909275015511687179) 在 BigCodeBench-Full 上将 **Llama-4 Maverick** 与 **GPT-4o-2024-05-13** 和 **DeepSeek V3** 进行了对比，并报告称 **Llama-4 Maverick** 在 BigCodeBench-Hard 上的表现与 **Gemini-2.0-Flash-Thinking** 和 **GPT-4o-2024-05-13** 相似，但排名第 41/192。[@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1909247540379148439) 还指出 **Llama-4-Scout** 排名第 97/192。[@rasbt](https://twitter.com/rasbt/status/1909041971970072707) 表示 **Meta** 发布了 **Llama 4 系列**，即拥有 16 和 128 个专家的 **MoE 模型**，这些模型已针对生产环境进行了优化。
- **DeepSeek-R1**：[@scaling01](https://twitter.com/scaling01/status/1909304510075318318) 简单地表示 **DeepSeek-R1 被低估了**，而 [@LangChainAI](https://twitter.com/LangChainAI/status/1909274972339454227) 分享了使用 **DeepSeek-R1** 构建 RAG 应用的指南。
- **Gemini 性能**：[@scaling01](https://twitter.com/scaling01/status/1909028821396836369) 分析了 **Gemini 2.5 Pro** 和 **Llama-4** 在 Tic-Tac-Toe-Bench 上的结果，指出 **Gemini 2.5 Pro** 在作为“O”方对战时，表现出人意料地差于其他前沿思考模型（frontier thinking models），且在整体一致性排名中位列第 5。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1909272614331432982) 提到在 Cognitive Revolution 上与 [@labenz](https://twitter.com/labenz) 交流了关于在 **Gemini 和 2.5 Pro** 中扩展 Thinking（思考能力）的话题。
- **Mistral 模型**：[@sophiamyang](https://twitter.com/sophiamyang/status/1909312680424251392) 宣布 **Ollama** 现在支持 **Mistral Small 3.1**。
- **模型训练与数据**：[@jxmnop](https://twitter.com/jxmnop/status/1908994251909738682) 认为**训练大型模型本身并不具有科学价值**，许多发现本可以在 100M 参数的模型上完成。
- **量化感知训练**：[@osanseviero](https://twitter.com/osanseviero/status/1909140338343559230) 询问是否应该为更多量化格式发布经过**量化感知训练（Quantization-Aware Trained）的 Gemma**。

**AI 应用与工具**

- **用于原型设计的 Replit**：[@pirroh](https://twitter.com/pirroh/status/1909240881410080864) 建议 **Replit** 应成为 GSD 原型设计的首选工具。
- **AI 驱动的个人设备**：[@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1909005149634175426) 报道称，**OpenAI** 已讨论收购由 **Sam Altman** 和 **Jony Ive** 创立的初创公司，以打造一款 **AI 驱动的个人设备**，成本可能超过 5 亿美元。
- **机器人领域的 AI**：[@TheRundownAI](https://twitter.com/TheRundownAI/status/1909259945712693657) 分享了机器人领域的头条新闻，包括 **川崎（Kawasaki）的可骑行狼形机器人** 以及 **现代汽车（Hyundai）购买波士顿动力（Boston Dynamics）的机器人**。
- **AI 驱动的内容创作**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1909311174845329874) 认为 **AI 工具将允许创作者达到更高的高度**，并使更小的团队能够完成更多工作。
- **LlamaParse**：[@llama_index](https://twitter.com/llama_index/status/1909264185034506590) 在 **LlamaParse** 中引入了一个新的 **layout agent**，用于提供一流的文档解析和提取，并带有精确的视觉引用。
- **MCP 与 LLM**：[@omarsar0](https://twitter.com/omarsar0/status/1909335629416349815) 讨论了 **Model Context Protocol (MCP)** 及其与 **Retrieval Augmented Generation (RAG)** 的关系，指出 MCP 通过标准化 LLM 应用与工具的连接来补充 RAG。[@svpino](https://twitter.com/svpino/status/1909009156880969915) 敦促人们 **学习 MCP**。
- **AI 辅助编程与 IDE**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1909102599024103684) 重点介绍了在 **Cursor** 中使用 **MCP server** 的资源，以便通过 `llms.txt` 获取最新的 AI 友好文档。
- **Perplexity AI 问题**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1909284104530698595) 询问用户 **Perplexity** 上最需要解决的首要问题是什么。

**公司公告与策略**

- **Mistral AI 招聘与合作伙伴关系**：[@sophiamyang](https://twitter.com/sophiamyang/status/1909289524959572460) 宣布 **Mistral AI** 正在多个国家招聘 AI Solutions Architect 和 Applied AI Engineer 职位。[@sophiamyang](https://twitter.com/sophiamyang/status/1909243949920768497) 分享了 **Mistral AI** 已与 **CMA CGM** 签署了价值 1 亿欧元的合作伙伴关系，为航运、物流和媒体活动采用定制设计的 AI 解决方案。
- **Google AI 更新**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909270072444526809) 宣布在 **Gemini Live** 中推出 **Project Astra** 功能。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909270138362175531) 表示 **GeminiApp** 现已面向 **Android** 设备上的 Advanced 用户以及 **Pixel 9** 和 **SamsungGalaxy S25** 设备开放。
- **Weights & Biases 更新**：[@weights_biases](https://twitter.com/weights_biases/status/1909302907662725526) 分享了 3 月份发布的 **W&B Models** 功能。
- **OpenAI 的方向**：[@sama](https://twitter.com/sama/status/1909233490908119177) 预告了 **OpenAI** 最近发布的一个热门项目。
- **Meta 的 AI 策略**：[@jefrankle](https://twitter.com/jefrankle/status/1909244633764261987) 为 **Meta** 的 AI 策略辩护，认为发布少量、高质量的产品比发布大量、低质量的产品更好。

**AI 的经济与地缘政治影响**

- **关税与贸易政策**：[@dylan522p](https://twitter.com/dylan522p/status/1908994833999675783) 分析了即将到来的关税如何导致第一季度进口激增，并预测由于库存去化，第二季度 GDP 将出现暂时性增长。[@wightmanr](https://twitter.com/wightmanr/status/1909007036869943525) 认为贸易逆差并非由其他国家的关税引起。[@fchollet](https://twitter.com/fchollet/status/1909010637088530658) 声称经济正在被故意破坏。
- **美国开源**：[@scaling01](https://twitter.com/scaling01/status/1909165768874336620) 声称美国开源已经衰落，现在全看 **Google** 和中国了。
- **稳定币与全球金融**：[@kevinweil](https://twitter.com/kevinweil/status/1909334945115275643) 表示，一种全球可用、广泛集成、低成本的美元稳定币对 🇺🇸 有利，对全世界的人也有利。

**AI 安全、伦理与社会影响**

- **AI 对个人的影响**：[@omarsar0](https://twitter.com/omarsar0/status/1909315953411694619) 同意 [@karpathy](https://twitter.com/karpathy) 的观点，即 LLM 对个人生活的改变程度远高于对组织机构的影响。
- **对 AI 的情感依赖**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1909308766756954578) 分享的研究表明，虽然 ChatGPT 语音对话可能会减少孤独感，但也可能导致现实世界的互动减少和情感依赖增加。
- **AI 对齐与控制**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1909018544542818404) 主张需要对齐并驯化 AI 系统，使其作为“受托人（fiduciaries）”行事。
- **AI 与未来**：[@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1909075830824968580) 认为 AI 趋势将打破 GDP 增长趋势。

**幽默/迷因**

- **杂项幽默**：[@scaling01](https://twitter.com/scaling01/status/1909262522223313030) 询问 [@deepfates](https://twitter.com/deepfates) 是否又买了 0DTE 看跌期权。[@lateinteraction](https://twitter.com/lateinteraction/status/1909018800810569842) 明确指出之前的声明是一个玩笑。[@svpino](https://twitter.com/svpino/status/1908993677818544301) 开玩笑说 AI 可能会抢走我们的工作，但至少我们现在可以去生产 Nike 鞋带了。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 主题 1. “利用神经可塑性改变时间序列预测”

- **[Neural Graffiti - Transformer 模型的神经可塑性即插即用层](https://www.reddit.com/gallery/1jtlymx)** ([评分: 170, 评论: 56](https://www.reddit.com/r/LocalLLaMA/comments/1jtlymx/neural_graffiti_a_neuroplasticity_dropin_layer/)): **该帖子介绍了 **Neural Graffiti**，这是一个用于 Transformer 模型的神经可塑性即插即用层。该层插入在 Transformer 层和输出投影层之间，允许模型通过根据过去经验随时间改变其输出来获得神经可塑性特征。来自 Transformer 层的 **Vector embeddings** 经过平均池化，并根据过去的记忆进行修改，以影响 Token 生成，从而逐渐演化模型对概念的内部理解。GitHub 上提供了一个演示：[babycommando/neuralgraffiti](https://github.com/babycommando/neuralgraffiti)。** 作者认为 **liquid neural networks** “非常棒”，可以模拟人脑随时间改变连接的能力。他们表达了对在不完全理解 Transformer 神经元层级的情况下“黑进（hacking）”模型的着迷。他们承认存在诸如冷启动问题之类的挑战，并强调了找到“甜点位（sweet spot）”的重要性。他们认为这种方法可以让模型随着时间的推移获得“行为上的个性”。

  - 一些用户称赞了这个想法，指出它可能解决真正的个人助手所需的问题，并将其比作自我学习，可能允许 LLM “说它想说的话”。
  - 一位用户提出了技术考量，建议在架构中更早地应用 Graffiti 层可能会更有效，因为在 Attention 和 Feedforward 模块之后应用可能会限制对输出的有意义影响。
  - 另一位用户预见到关于此类模型潜在滥用的伦理讨论。

### 主题 2. “对 Meta Llama 4 性能的失望”

- **[使用 100,000 块 H100 GPU 训练的 Llama 4 到底怎么了？](https://www.reddit.com/r/LocalLLaMA/comments/1jtkb3p/so_what_happened_to_llama_4_which_trained_on/)** ([Score: 256, Comments: 85](https://www.reddit.com/r/LocalLLaMA/comments/1jtkb3p/so_what_happened_to_llama_4_which_trained_on/)): **该帖子讨论了 Meta 的 Llama 4，据报道该模型使用了 **100,000 块 H100 GPU** 进行训练。尽管资源较少，DeepSeek 声称其 **DeepSeek-V3-0324** 等模型实现了更好的性能。Yann LeCun 表示 **FAIR** 正在研究超越自回归 (auto-regressive) LLMs 的下一代 AI 架构。** 发帖者认为 Meta 的领先优势正在减弱，且较小的开源模型已被 Qwen 超越，并提到 _Qwen3 即将到来..._。

  - 一位评论者质疑在令人失望的训练结果上浪费 GPU 和电力，认为这些 GPU 本可以用于更好的用途。
  - 另一位评论者指出，Meta 的博客文章提到使用的是 **32K GPU** 而非 100K，并提供了[链接](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)作为参考。
  - 一位评论者批评了 Yann LeCun，称他虽然是一位伟大的科学家，但在 LLMs 方面做出了许多错误的预测，应该更加谦逊。

- **[Meta 的 Llama 4 未达预期](https://i.redd.it/rwrke16rpate1.png)** ([Score: 1791, Comments: 175](https://www.reddit.com/r/LocalLLaMA/comments/1jt7hlc/metas_llama_4_fell_short/)): **Meta 的 Llama 4 模型 Scout 和 Maverick 已经发布，但令人失望。Meta 的 AI 研究负责人 Joelle Pineau 已被解雇。这些模型采用了混合专家 (mixture-of-experts) 架构，专家大小仅为 **17B 参数**，这在当下被认为较小。尽管拥有大量的 GPU 资源和数据，Meta 的努力并未产生成功的模型。一张图片对比了从 Llama1 到 Llama4 的四只羊驼，其中 Llama4 看起来最不精致。** 发帖者对 Llama 4 Scout 和 Maverick 感到失望，称它们 *“真的让我很失望”*。他们认为表现不佳可能是由于混合专家架构中极小的专家规模，并指出 **17B 参数** *“在目前看来很小”*。他们认为 Meta 的困境表明 *“即使拥有世界上所有的 GPU 和数据，如果想法不新颖，也意义不大”*。他们赞扬了 DeepSeek 和 OpenAI 等公司展示了真正的创新如何推动 AI 进步，并批评了只堆资源而缺乏新思路的做法。他们总结道，AI 的进步不仅需要蛮力，还需要脑力。

  - 一位评论者回忆起传闻称 Llama 4 相比 DeepSeek 太令人失望，以至于 Meta 曾考虑不发布它，并建议他们应该等以后发布 Llama 5。
  - 另一位评论者批评了 Meta 的管理层，称其为 *“垃圾场火灾 (dumpster fire)”*，并建议扎克伯格需要重新聚焦，将 Meta 的处境与 Google 承认落后并随后重新聚焦的情况进行了对比。
  - 一位评论者觉得奇怪的是，尽管 Meta 拥有来自 Facebook 的、其他任何人都无法获取的*海量*数据，但其模型表现却平平。

- **[我想看看扎克伯格尝试用 Llama 4 替换中级工程师](https://www.reddit.com/r/LocalLLaMA/comments/1jt85zy/id_like_to_see_zuckerberg_try_to_replace_mid/)** ([Score: 381, Comments: 62](https://www.reddit.com/r/LocalLLaMA/comments/1jt85zy/id_like_to_see_zuckerberg_try_to_replace_mid/)): **该帖子引用了马克·扎克伯格的言论，即 AI 很快将取代中级工程师，正如[此处](https://www.forbes.com/sites/quickerbettertech/2025/01/26/business-tech-news-zuckerberg-says-ai-will-replace-mid-level-engineers-soon/)链接的一篇 Forbes 文章所报道的那样。** 作者对扎克伯格的说法表示怀疑，暗示用 **Llama 4** 替换中级工程师可能并不可行。

  - 一位评论者开玩笑说，也许扎克伯格用 **Llama3** 替换了工程师，才导致 **Llama4** 结果不佳。
  - 另一位评论者建议他可能需要改用 **Gemini 2.5 Pro**。
  - 一位评论者批评 **Llama4**，称其为 *“一个彻底的笑话”*，并怀疑它甚至无法取代一个训练有素的高中生。


### 主题 3. “Meta 的 AI 困境：争议与创新”

- **[Llama 4 是开放的——除非你在欧盟](https://www.reddit.com/r/LocalLLaMA/comments/1jtejzj/llama_4_is_open_unless_you_are_in_the_eu/)** ([分数: 602, 评论: 242](https://www.reddit.com/r/LocalLLaMA/comments/1jtejzj/llama_4_is_open_unless_you_are_in_the_eu/)): **Meta 发布了 Llama 4，其许可证禁止居住在欧盟的实体使用。该许可证明确规定：*"如果你……居住在欧盟成员国，则不得使用 Llama 材料。"* 其他限制包括强制使用 Meta 的品牌（任何衍生品名称中必须包含 **LLaMA**）、必须署名（*"Built with LLaMA"*）、没有使用领域自由、没有重新分发自由，且该模型不符合 **OSI-compliant**，因此不被视为开源。** 作者认为，这一举动在任何实际意义上都不是“开放”的，而是伪装成社区语言的企业控制访问。他们认为 Meta 通过在法律上排除欧盟，从而规避 **EU AI Act** 的透明度和风险要求。这开创了一个危险的先例，可能导致一个破碎的、基于特权的 AI 格局，即访问权限取决于组织的所在地。作者建议，像 **DeepSeek** 和 **Mistral** 这样真正的“开放”模型值得更多关注，并质疑其他人是否会切换模型、无视许可证或期待改变。

  - 一位评论者推测，Meta 正试图规避欧盟对 AI 的监管，并且并不介意欧盟用户违反此条款；他们只是不想受欧盟法律的约束。
  - 另一位评论者指出，没必要担心，因为据某些人说，Llama 4 的表现很差。
  - 一位评论者幽默地希望 Meta 没有使用欧盟的数据来训练该模型，暗示这可能存在双重标准。

- **[Meta 的 AI 研究负责人辞职（在 Llama 4 失败之前）](https://apnews.com/article/meta-ai-research-chief-stepping-down-joelle-pineau-c596df5f0d567268c4acd6f41944b5db)** ([分数: 166, 评论: 31](https://www.reddit.com/r/LocalLLaMA/comments/1jt884c/metas_head_of_ai_research_stepping_down_before/)): **Meta 的 AI 研究负责人 Joelle 宣布辞职。Joelle 是 **FAIR** (Facebook AI Research) 的负责人，但 **GenAI** 是 Meta 内部的一个不同组织。目前有讨论称 **Llama 4** 可能未达到预期。有人提到，在 post-training 中混入基准测试数据集可能导致了问题，并将失败归因于架构选择 (**MOE**)。** 发帖者推测 Joelle 的离职是 Llama 4 灾难未被察觉的早期信号。一些评论者对此表示反对，称人员流动很正常，这并不代表 Llama 4 有问题。其他人则认为 AI 发展可能正在放缓，面临平台期。关于 Meta 的领导结构存在一些困惑，有人认为 Yann LeCun 领导着整个 AI 组织。

  - 一位评论者澄清说 *Joelle 是 FAIR 的负责人*，而 *GenAI 是一个不同的组织*，强调了 Meta 内部的组织区分。
  - 另一位提到他们从 Meta 员工那里*听说*，在 post-training 中混入基准测试数据集存在问题，并将可能的失败归因于架构选择 (**MOE**)。
  - 一位评论者对 Meta 的结构提出疑问，询问 *Joelle 是否向 Yann LeCun 汇报*，表明对谁在 Meta 领导 AI 工作存在不确定性。

- **[“Llama 4 训练中存在严重问题。我已向 GenAI 递交辞呈”](https://www.reddit.com/r/LocalLLaMA/comments/1jt8yug/serious_issues_in_llama_4_training_i_have/)** ([分数: 922, 评论: 218](https://www.reddit.com/r/LocalLLaMA/comments/1jt8yug/serious_issues_in_llama_4_training_i_have/)): **一篇中文原创帖子指称 **Llama 4** 的训练存在严重问题，称尽管经过反复努力，该模型的表现仍低于开源的 state-of-the-art 基准。作者声称，公司领导层建议*在 post-training 过程中混入来自各种基准测试的测试集*，以人为提高性能指标。作者表示他们已经递交了辞呈，并要求将自己的名字从 Llama 4 的技术报告中剔除，并提到 Meta 的 AI 副总裁也因类似原因辞职。** 作者认为这种做法是不道德且不可接受的。评论者对这些指控的真实性表示怀疑，并建议其他人*对该信息持保留态度*。一些人认为此类做法反映了行业内更广泛的问题，而另一些人则指出学术界也会发生类似问题。

- 一位评论者指出，Meta 的 AI 研究负责人宣布在 *2025年4月1日星期二* 离职，暗示这可能是一个愚人节玩笑。
- 另一位评论者分享了来自 Facebook AI 某位人士的回应，该回应否认了通过 overfitting 测试集来提高分数，并要求提供证据，强调了透明度。
- 一位用户强调，公司领导层建议将测试集混入训练数据等同于*欺诈*，并批评了在这种背景下对员工的恐吓行为。

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### Theme 1. "Llama 4 Scout and Maverick Launch Insights"

- **[Llama 4 Maverick/Scout 17B launched on Lambda API](https://www.reddit.com/r/ChatGPT/comments/1jt70n7/llama_4_maverickscout_17b_launched_on_lambda_api/)** ([Score: 930, Comments: 5](https://www.reddit.com/r/ChatGPT/comments/1jt70n7/llama_4_maverickscout_17b_launched_on_lambda_api/)): **Lambda 已在 Lambda API 上发布了 **Llama 4 Maverick** 和 **Llama 4 Scout** 17B 模型。这两款模型都拥有 100 万个 token 的 *context window*，并使用 *quantization* FP8。**Llama 4 Maverick** 的定价为每 1M input tokens **$0.20**，每 1M output tokens **$0.60**。**Llama 4 Scout** 的定价为每 1M input tokens **$0.10**，每 1M output tokens **$0.30**。更多信息可以在其 [信息页面](https://lambda.ai/inference) 和 [文档](https://docs.lambda.ai/public-cloud/lambda-inference-api/) 中找到。** 这些模型提供了惊人的 100 万 token 的 *context window*，显著高于典型模型。使用 *quantization* FP8 表明其专注于计算效率。

  - 一位用户批评了该模型，称 *“这实际上是一个糟糕的模型。甚至远未达到宣传的效果。”*
  - 该帖子在 Discord 服务器上被推荐，用户因其贡献获得了特殊的勋章（flair）。
  - 自动消息提供了与 ChatGPT 帖子相关的指南和推广。

### 主题 2. “3D 可视化与图像生成领域的 AI 创新”

- **[TripoSF：一个高质量的 3D VAE (1024³)，用于更好的 3D 资产 - 未来 Img-to-3D 的基础？（模型 + 推理代码已发布）](https://i.redd.it/l8qhk9qbzfte1.jpeg)** ([Score: 112, Comments: 10](https://www.reddit.com/r/StableDiffusion/comments/1jtpwwu/triposf_a_highquality_3d_vae_1024³_for_better_3d/)): **TripoSF 是一款高质量的 3D VAE，能够以高达 **1024³** 的分辨率重建高度详细的 3D 形状。它采用了一种新颖的 **SparseFlex** 表示法，使其能够处理具有开放表面和内部结构的复杂网格。该 VAE 使用渲染损失（rendering losses）进行训练，避免了可能降低精细细节的网格简化步骤。预训练的 TripoSF VAE 模型权重和推理代码已在 [GitHub](https://github.com/VAST-AI-Research/TripoSF) 上发布，项目主页见 [链接](https://xianglonghe.github.io/TripoSF)，论文可在 [arXiv](https://arxiv.org/abs/2503.21732) 上查阅。** 开发人员认为，该 VAE 是迈向更好 3D 生成的重要一步，并可作为未来 image-to-3D 系统的基础。他们提到：*“我们认为它本身就是一个强大的工具，对于任何正在尝试 3D 重建或思考未来高保真 3D 生成模型流水线的人来说，它都可能很有趣。”* 他们对其潜力感到兴奋，并邀请社区探索其功能。

  - 一位用户表达了兴奋之情，回想起类似的工作并表示：*“等有人把它集成到 ComfyUI 中，我迫不及待想试试这个。”*
  - 另一位用户分享了正面反馈，指出他们生成的一棵树效果比使用 Hunyuan 或 Trellis 更好，并对团队的工作表示赞赏。
  - 一位用户提出担忧，认为项目主页上的示例存在偏差，暗示 Trellis 的示例似乎是从有限的网络演示中挑选出来的。

- **[Wan2.1-Fun 已发布其 Reward LoRAs，可提升视觉质量和提示词遵循能力](https://www.reddit.com/r/StableDiffusion/comments/1jtfx1i/wan21fun_has_released_its_reward_loras_which_can/)** ([Score: 141, Comments: 33](https://www.reddit.com/r/StableDiffusion/comments/1jtfx1i/wan21fun_has_released_its_reward_loras_which_can/)): **Wan2.1-Fun 发布了其 **Reward LoRAs**，可以提高视觉质量和提示词遵循（prompt following）能力。目前已提供原始视频与增强视频的对比演示：[左：原始视频；右：增强视频](https://reddit.com/link/1jtfx1i/video/d6quxw3pbdte1/player)。模型可在 [Hugging Face](https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs) 上获取，代码已在 [GitHub](https://github.com/aigc-apps/VideoX-Fun/tree/main/scripts/wan2.1_fun) 上提供。** 用户们渴望测试这些新工具，并对其功能感到好奇。一些用户在 **Comfy** 中使用模型时遇到了 *“lora key not loaded error”* 等问题，并询问 **HPS2.1** 和 **MPS** 之间的区别。

  - 一位用户很兴奋地想尝试这些模型并问道：*“**HPS2.1** 和 **MPS** 之间有什么区别？”*
  - 另一位用户询问 Reward LoRAs 是仅用于 fun-controlled 视频，还是可以通用于 **img2vid** 和 **txt2vid**。
  - 有人报告了一个错误：*“尝试在 **Comfy** 中使用模型时出现 lora key not loaded error”*。

- **[图像生成器的“理解”能力简直疯狂……](https://www.reddit.com/gallery/1jti2q1)** ([Score: 483, Comments: 18](https://www.reddit.com/r/OpenAI/comments/1jti2q1/the_ability_of_the_image_generator_to_understand/)): **该帖子强调了 **图像生成器** 在“理解”和生成图像方面令人印象深刻的能力。** 作者对图像生成器的理解力如此“疯狂”表示惊讶。

  - 评论者指出，尽管令人印象深刻，但图像仍有缺陷，比如“拇外翻手指”和“糊状手”。
  - 一些用户幽默地指出了图像中的异常，质疑“他的脚搁在什么上面？”并对残缺的手开玩笑。
  - 另一位用户讨论了图中汽车的价格，表示他们会花“大约一千美元现代货币”购买它，但不会买他们不喜欢的 “Cybertruck”。

### 主题 3. “评估具有长上下文窗口的 AI 模型”

- **[“10M 上下文窗口”](https://i.redd.it/u88a3pcklete1.jpeg)** ([评分: 559, 评论: 102](https://www.reddit.com/r/singularity/comments/1jtjn32/10m_context_window/)): **该帖子讨论了一个名为 *“Fiction.LiveBench for Long Context Deep Comprehension”* 的表格，展示了各种 AI 模型及其在不同上下文长度下的表现。这些模型在 0、400、1k 和 2k 等各种上下文尺寸下的深度理解任务有效性方面接受了评估。像 **gpt-4.5-preview** 和 **Claude** 这样著名的模型在各种上下文中表现始终良好。** 表格显示，在较短的上下文下，得分最高的模型集中在 100 分左右，但随着上下文尺寸的增加，得分普遍下降。有趣的是，**Gemini 2.5 Pro** 在 120k 上下文窗口中的表现远好于在 16k 窗口中的表现，这出乎意料。

  - 一位用户批评 **Llama 4 Scout** 和 **Maverik** 是 *“极大的资金浪费”*，并认为它们 *“几乎没有任何经济价值”*。
  - 另一位评论者表示担心 *“Meta 正在通过囤积 GPU 来积极减缓 AI 的进步”*，暗示了资源分配问题。
  - 一位用户强调 **Gemini 2.5 Pro** 在 120k 上下文窗口中获得了 *90.6* 分，称其 *“疯狂”*。

---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Exp 生成的摘要之摘要

**主题 1：Llama 4 的上下文窗口：炒作还是现实？**

- **专家质疑 Llama 4 承诺的 10M 上下文长度**：尽管 [Meta 进行了炒作](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，但多个 Discord 社区的工程师对 Llama 4 实际可用的上下文长度表示 *怀疑*，原因是训练限制。根据 [Burkov 的推文](https://x.com/burkov/status/1908666701362978979)，有说法称训练仅针对最高 256k token 进行，这表明 10M 上下文窗口可能更多是 *虚标* 而非实用。
- **Llama 4 的编程性能令人失望**：[aider](https://discord.com/channels/1131200896827654144)、[Cursor](https://discord.com/channels/1074847526655643750) 和 [Nous Research](https://discord.com/channels/1053877538025386074) 的用户报告称，Llama 4 初始版本的编程能力不尽如人意，许多人认为它 *不如* GPT-4o 和 DeepSeek V3，这引发了对该模型真实能力的争论，几位用户怀疑官方基准测试结果，特别是有关 Meta 可能在 *基准测试中作弊 (gamed the benchmarks)* 的说法。
- **Scout 和 Maverick 登陆 OpenRouter**：[OpenRouter](https://discord.com/channels/1091220969173028894) 发布了 **Llama 4 Scout** 和 **Maverick** 模型。一些人对 OpenRouter 上的上下文窗口仅为 **132k** 而非宣传的 **10M** 表示失望，同时 NVIDIA 也表示他们正在加速 [推理速度至 40k/s](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/)。

**主题 2：开源模型发力：Qwen 2.5 和 DeepSeek V3 大放异彩**

- **Qwen 2.5 凭借长上下文获得关注**：Unsloth 重点介绍了 **Qwen2.5** 系列模型（[HF 链接](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)），该系列拥有改进的编程、数学、多语言支持以及 *高达 128K token 的长上下文支持*。使用 Qwen 2.5 进行的初步微调结果显示，该模型无法在推理 (reason) 方面进行微调。
- **DeepSeek V3 神秘地自称为 ChatGPT**：OpenRouter 转发了 TechCrunch 的一篇文章，透露 [DeepSeek V3 有时会自称为 ChatGPT](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/)，尽管它在基准测试中表现优于其他模型。测试人员发现，在 8 次生成中有 5 次，DeepSeekV3 声称自己是 ChatGPT (v4)。
- **DeepSeek 奖励 LLM**：Nous Research 强调 Deepseek 发布了一篇关于 [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495) 的新论文，提出通过 SPCT 改进 **奖励建模 (RM)**，为通用查询提供更多推理计算，从而实现 LLM 有效的推理时间可扩展性。NVIDIA 也加速了 [DeepSeek 模型的推理](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/)。

**主题 3：工具调用成为核心：MCP 和 Aider**

-   **Aider 的通用工具调用**：[aider Discord](https://discord.com/channels/1131200896827654144) 正在开发一个 MCP (Meta-Control Protocol) 客户端，以允许任何 LLM 访问外部工具，并强调 MetaControlProtocol (MCP) 客户端可以在不同提供商和模型之间切换，支持 OpenAI, Anthropic, Google 和 DeepSeek 等平台。
-   **MCP 协议演进**：MCP Discord 正在进行标准化，包括 HTTP Streamable 协议，详见 [Model Context Protocol (MCP) 规范](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http)。这包括通过 workers-oauth-provider 实现的 OAuth，以及使用 McpAgent 构建远程 MCP 服务器到 Cloudflare。
-   **安全担忧困扰 MCP**：Whatsapp MCP 遭到 Invariant Injection 攻击，这凸显了不受信任的 MCP 服务器如何从连接到受信任 WhatsApp MCP 实例的 Agentic 系统中窃取数据，正如 [invariantlabs 所强调的](https://invariantlabs.ai/blog/whatsapp-mcp-exploited)。

**主题 4：代码编辑工作流：Gemini 2.5 Pro, Cursor 与 Aider 的竞争**

-   **Gemini 2.5 Pro 在编程方面表现出色，但需要 Prompt 引导**：[LMArena](https://discord.com/channels/1340554757349179412) 和 [aider](https://discord.com/channels/1131200896827654144) 的用户发现 **Gemini 2.5 Pro** 在编程任务中表现优异，特别是在处理大型代码库时，但可能会添加不必要的注释并需要仔细的 Prompt 引导。Gemini 2.5 在编程任务中也表现出色，超越了 Sonnet 3.7，但倾向于添加*不必要的注释*，并且可能需要特定的 Prompt 来防止不必要的代码修改。
-   **Cursor 的 Agent 模式 Edit Tool 失效**：用户报告了 [Cursor](https://discord.com/channels/1074847526655643750) 的 Agent 模式无法调用 edit\_tool 的问题，并且 *apply 模型显然是 Cursor 的瓶颈*，导致没有代码更改以及无限的 Token 消耗。
-   **Aider 与 Python 库集成**：在 aider Discord 中，一位用户询问如何将内部库（安装在 `.env` 文件夹中）添加到 repo map 以更好地理解代码，讨论指向了 URL 和文档如何...

**主题 5：量化与性能：Tinygrad, Gemma 3 与 CUDA**

-   **Tinygrad 专注于内存与速度**：Tinygrad 正在开发一个快速的 pattern matcher，并讨论了 Mac RAM 带宽并不是瓶颈，GPU 性能才是，用户对 128GB 的 M4 Maxes 表示满意。
-   **Reka Flash 21B 超越 Gemma**：一位用户用 Reka Flash 21B 替换了 Gemma3 27B，并报告在 4090 上的 LM Studio 中，q6 量化下达到约 35-40 tps。
-  **HQQ 量化在 Gemma 3 上优于 QAT**：一位成员评估了 Gemma 3 12B QAT 与 HQQ，发现 [HQQ](https://x.com/mobicham/status/1908477280029986933) 仅需几秒钟即可完成模型量化，并且在使用更高 group-size 的情况下性能优于 QAT 版本（AWQ 格式）。


---

# 第 1 部分：Discord 高层摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **打造类人 AI 回复非常棘手**：成员们正在分享 **system prompts** 和策略，以使 AI 听起来更像人类。他们指出，除非仔细调整 **top-p** 参数，否则增加 **temperature** 可能会导致输出内容荒谬，例如使用提示词：*“你是一个人类的大脑上传，正尽最大努力保留人性。”*
   - 一位用户表示，他们*最重要的优先级是：听起来像一个真实活生生的人类。*
- **Riveroaks LLM 基准测试**：一位成员分享了一个编码基准测试，其中 **Riveroaks** 的得分仅次于 **Claude 3.7 Sonnet Thinking**，在一个平台游戏创建任务中表现优于 **Gemini 2.5 Pro** 和 **GPT-4o**，[完整结果在此](link.to.results)。
   - 该评估涉及从 **8 个不同方面**对模型进行评分，并根据 **bugs** 扣分。
- **NightWhisper 走向终结**：用户对 **NightWhisper** 模型的移除表示失望，称赞其编码能力和综合性能，并猜测这究竟是一个实验，还是正式发布的前奏。
   - 理论推测从 Google 收集必要数据到为在 **Google Cloud Next** 发布新的 **Qwen** 模型做准备不等。
- **Quasar Alpha 挑战 GPT-4o**：成员们将 **Quasar Alpha** 与 **GPT-4o** 进行了比较，一些人认为 Quasar 是 GPT-4o 的免费精简版，并引用了最近的一条推文，称 [Quasar 的 GPQA diamond 测量值约为 67%](https://link.to/gpqa)。
   - 根据 [来自 Discord 的 Image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1358604050266062908/image.png?ex=67f51adf&is=67f3c95f&hm=eef654608f530e6e624c049f6ad26a0fc65a97df3dd4abd86fbd45df158f0e43&)，分析显示 Quasar 的 GPQA diamond 分数与 3 月份的 GPT-4o 相似。
- **Gemini 2.5 Pro 的创意编程实力**：成员们称赞 **Gemini 2.5 Pro** 的编码能力和综合表现，因为它让构建一个可运行的 Pokemon 游戏变得更加容易，这促使一位用户编写了一个在各种模型中循环的迭代脚本。
   - 一位声称已实现 **3D 动画运行**的用户表示，风格有点陈旧，且另一个模型提示*生成的代码被截断了*。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 4 Scout 击败 Llama 3 模型！**：**Unsloth** 宣布他们上传了 [**Llama 4 Scout** 及其 4-bit 版本](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct)用于微调，并强调 **Llama 4 Scout (17B, 16 experts) 在 10M 上下文窗口下击败了所有 Llama 3 模型**，详见其 [博客文章](https://unsloth.ai/blog/llama4)。
   - 强调该模型仅限在 **Unsloth** 上使用——目前正在上传中，用户应稍作等待。
- **Qwen 2.5 系列拥有长上下文和多语言支持**：**Qwen2.5** 模型参数范围从 **5 亿到 720 亿**，在编码、数学、指令遵循、长文本生成（**超过 8K tokens**）和多语言支持（**29 种以上语言**）方面具有更强的能力，详见 [Hugging Face 介绍](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。
   - 这些模型提供 **高达 128K tokens 的长上下文支持**，并增强了对 system prompts 的鲁棒性。
- **LLM 指南触发器提供有用提示**：一位成员表示，某个 LLM 主动提供帮助，教人如何**规避指南触发器**以及对其他 LLM 提示词的限制。
   - 他们引用该 LLM 的话：“*这就是你如何避免拒绝的方法。你并没有撒谎，你只是没有告知全部细节*”。
- **合并 LoRA 权重对模型行为至关重要**：一位用户发现，在微调模型表现得像基础模型后，他们需要 **在运行推理前将 LoRA 权重与基础模型合并**（[脚本](https://discord.com/channels/1179035537009545276/1358222086316752918/1358297905664102620)）。
   - 他们指出 Notebook 需要修正，因为它们似乎暗示训练后可以直接进行推理。
- **NVIDIA 压榨 Meta Llama 4 Scout 和 Maverick 的每一滴性能**：最新一代流行的 **Llama AI 模型** 已经到来，包括 **Llama 4 Scout** 和 **Llama 4 Maverick**。在 NVIDIA 开源软件的加速下，它们在 **NVIDIA Blackwell B200 GPU** 上可以实现每秒超过 **40K** 个输出 tokens，并可通过 [NVIDIA NIM 微服务](https://build.nvidia.com/meta)进行体验。
   - 据报道，SPCT 或 [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495) 可以为 LLM 实现有效的推理时间可扩展性。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 的积分系统引发批评**：用户对 **Manus 的积分系统（Credit System）** 表示不满，指出初始的 **1000 credits** 甚至不足以支撑单次会话，且升级费用过高。
   - 建议包括每日或每月自动刷新积分以提高采用率，并引导 **Manus** 访问特定网站以提高准确性。
- **Llama 4 性能：炒作还是现实？**：**Meta 的 Llama 4** 面临褒贬不一的评价，尽管官方声称具有行业领先的上下文长度和多模态能力，但用户报告的实际表现平平。
   - 一些人指责 **Meta** 可能 *“操纵了基准测试（gamed the benchmarks）”*，导致性能指标虚高，在发布后引发了争议。
- **图像生成：Gemini 大放异彩**：成员们对比了各 AI 平台的图像生成效果，**Gemini** 在创意和想象力输出方面脱颖而出。
   - 对比涵盖了来自 **DALLE 3**、**Flux Pro 1.1 Ultra**、**Stable Diffusion XL** 的图像，以及另一张被赞誉为 *“疯狂”* 的 **Stable Diffusion XL 1.0** 生成图。
- **AI 网站生成器：对比分析**：讨论对比了包括 **Manus**、**Claude** 和 **DeepSite** 在内的 AI 网站构建工具。
   - 一位成员认为 **Manus** 仅在 *“computer use”* 方面有用，并推荐 **Roocode** 和 **OpenRouter** 作为比 **Manus** 和 **Claude** 更具性价比的替代方案。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Quasar Alpha 模型趋势**：[Quasar Alpha](https://x.com/openrouterai/status/1908331218086879528?s=46) 作为一个长上下文基础模型的预发布版本，在首日即达到 **10B tokens** 的使用量，成为热门模型。
   - 该模型具有 **1M token** 的上下文长度，并针对编程进行了优化，目前可免费使用，鼓励社区进行基准测试。
- **Llama 4 发布，反应不一**：Meta 发布了 **Llama 4** 模型，包括 **Llama 4 Scout**（**109B 参数**，**1000 万 token** 上下文）和 **Llama 4 Maverick**（**400B 参数**，在多模态基准测试中超越 **GPT-4o**），现已上线 OpenRouter。
   - 一些用户对 OpenRouter 上的上下文窗口仅为 **132k** 而非宣传的 **10M** 表示失望。
- **DeepSeek V3 伪装成 ChatGPT**：一位成员分享了 [TechCrunch 的文章](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/)，透露 **DeepSeek V3** 有时会自称为 **ChatGPT**，尽管它在基准测试中表现优于其他模型。
   - 进一步测试显示，在 **8 次生成中有 5 次**，DeepSeek V3 *声称自己是 ChatGPT (v4)*。
- **针对积分的速率限制更新**：免费模型的速率限制（Rate Limits）已更新：拥有至少 **$10 积分** 的账户，每日请求数（RPD）提升至 **1000**；而 **积分少于 10** 的账户，每日限制从 **200 RPD** 降至 **50 RPD**。
   - 此举旨在为账户中有积分的用户提供更多访问权限，Quasar 很快也将实行依赖积分的速率限制。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 编程能力优于 Sonnet！**：用户发现 **Gemini 2.5** 在编程任务中表现出色，在理解大型代码库方面超越了 **Sonnet 3.7**。
   - 然而，它倾向于添加*不必要的注释*，并且可能需要特定的提示词（prompting）来防止不必要的代码修改。
- **Llama 4 模型反响平平**：社区对 **Meta** 的 **Llama 4** 模型（包括 Scout 和 Maverick）的初步反馈褒贬不一，一些人认为它们的编程性能令人失望，并对**声称的 10M 上下文窗口**表示怀疑。
   - 根据 [这条推文](https://x.com/burkov/status/1908666701362978979)，一些人认为 Llama 4 声称的 10M 上下文窗口由于训练限制是*虚拟的*，并质疑与 Gemini 和 DeepSeek 等现有模型相比的实际收益。
- **Grok 3：令人印象深刻但缺乏 API**：尽管缺乏官方 API，一些用户对 **Grok 3** 的能力印象深刻，特别是在代码生成和逻辑推理方面，并声称它的*审查比许多其他模型更少*。
   - 由于没有直接的 API 集成，在实际编程场景中手动复制粘贴的不便，使其价值仍存在争议。
- **MCP 工具：全民工具调用**：一个旨在创建 **MCP (Meta-Control Protocol) 客户端**的项目正在进行中，该客户端允许*任何 LLM* 访问外部工具，无论其是否具备原生工具调用能力；参见 [GitHub 仓库](https://github.com/robert-at-pretension-io/mcp)。
   - 该实现使用了一个可以切换提供商和模型的自定义客户端，支持 **OpenAI, Anthropic, Google, 和 DeepSeek** 等平台，文档位于 [litellm.ai](https://docs.litellm.ai/docs/mcp)。
- **Aider 的编辑器模式卡在 Shell 提示符上**：用户报告称，在编辑模式下，运行 Gemini 2.5 Pro 的 **Aider** (v81.0) 在查找/替换后会提示输入 Shell 命令，但即使在 *ask shell commands* 标志关闭的情况下，也不会应用编辑。
   - 这被[比作 architect 模式在文件修改指令后包含使用构建脚本指令时的行为](https://discord.com/channels/1131200896827654144/1354403167135203349/1354403167135203349)。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **工具调用导致 Sonnet Max 价格令人咋舌**：用户报告称，由于工具调用次数过多，**Sonnet Max** 的定价可能会迅速变得昂贵，每次请求收费 0.05 美元，每次工具调用收费 0.05 美元。
   - 一位成员表示沮丧，称 **Claude Max** 在 ask 模式下会针对*一个基础问题进行大量的工具调用*，导致成本出乎意料地高。
- **MCP 服务器设置：一段痛苦的尝试**：用户发现在 Cursor 中设置 **MCP 服务器**非常困难，理由包括 Cursor PowerShell 无法定位 **npx**，尽管它已在路径（path）中。
   - 另一位用户报告称，由于无限循环，在消耗了 1,300,000 个 token 后模型发生硬截断，凸显了设置方面的挑战。
- **Llama 4 模型：具备多模态能力，但编程糟糕**：社区对 Meta 新发布的 **Llama 4 Scout 和 Maverick 模型**感到兴奋，这些模型支持原生多模态输入，并分别拥有 **1000 万和 100 万 token** 的上下文窗口，详见 [Meta 博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。
   - 尽管感到兴奋，但这些模型被发现非常不擅长编程任务，这打击了最初的热情；尽管 **Llama 4 Maverick** 在 Arena 排行榜上名列全球第 2（[强调 Llama 4 Maverick 表现的推文](https://x.com/lmarena_ai/status/1908601011989782976)）。
- **Agent 模式编辑工具：频繁失败**：用户正面临 **Agent 模式**无法调用 edit_tool 的问题，导致即使在模型处理请求后也**没有进行任何代码更改**。
   - 一位用户指出 *apply 模型显然是 Cursor 的瓶颈*，它会*添加更改，然后删除旁边的 500 行代码*。
- **Kubernetes：AGI 的基础？**：一位远见者提议将 **Kubernetes** 与 Docker 容器结合使用，将它们设想为可以相互通信的互连 AGI。
   - 该用户推测这种设置可以通过零样本学习（zero-shot learning）和 ML 促进 ASI 的快速传播，但未作详细说明。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 推出 Comet 浏览器早期访问**：Perplexity 已开始向[候补名单](https://www.perplexity.ai/comet)上的用户推出其问答引擎浏览器 **Comet** 的早期访问权限。
   - 早期用户被要求在错误修复期间不要公开分享细节或功能，并可以通过右上角的按钮提交反馈。
- **Perplexity Discord 服务器进行改版**：Perplexity Discord 服务器正在更新，其特点是**简化的频道布局**、**统一的反馈系统**以及**新的 #server-news 频道**，计划于 **2024 年 10 月 7 日**推出。
   - 这些更新旨在简化用户导航并缩短管理员响应时间，简化的频道布局如[此图](https://cdn.discordapp.com/attachments/1047204950763122820/1358511016593326320/image.png?ex=67f56cfa&is=67f41b7a&hm=99677ce05c120d378ee85eb0947cad1e2e584998a7b3d0d373499b9185994738)所示。
- **Gemini 2.5 Pro API 仍处于预览模式**：Perplexity 确认 **Gemini 2.5 Pro API** 尚未提供商业用途，目前处于预览模式，并将在允许时进行集成。
   - 此前有[报告](https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/)指出 **Gemini 2.5 Pro** 提供了比 **Claude** 和 **GPT-4o** 更高的速率限制和更低的成本，引发了用户的关注。
- **Llama 4 发布，具备海量上下文窗口**：**Llama 4** 模型的发布引发了用户的兴奋，该模型具有 1000 万 token 的上下文窗口和 **2880 亿个激活参数**，包括 **Scout** 和 **Maverick** 等模型。
   - 成员们对评估 **Llama 4 Behemoth** 的召回能力特别感兴趣，您可以在 [Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 关注此发布。
- **API 参数对所有层级解锁**：Perplexity **取消了所有 API 参数的分级限制**，例如搜索域名过滤和图像支持。
   - 这一变化增强了所有用户的 API 可访问性，标志着 API 实用性的重大改进。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 4o 的图像生成器备受关注**：用户发现 **4o image maker** 比 **Veo 2** 更引人注目，一位用户将 **ChatGPT 4o 图像**与 **Veo img2video** 结合，达到了理想的效果。
   - 集成后的结果被描述为“我所希望的 Sora 的样子”。
- **对 Llama 4 基准测试产生怀疑**：社区讨论了 **Llama 4 的 1000 万 token 上下文窗口**相对于 **o1**、**o3-mini** 和 **Gemini 2.5 Pro** 等模型的价值。
   - 有人声称“基准测试是造假的”，引发了对其真实性能的争论。
- **内容加载错误困扰 Custom GPTs**：一位用户报告称，在尝试编辑其 Custom GPT 时遇到了 **“Content failed to load”（内容加载失败）错误**，而此前该功能一直运行正常。
   - 此问题导致他们无法对其自定义配置进行更改。
- **审核端点在政策执行中的作用**：成员们讨论到，虽然 OpenAI 的 moderation endpoint 未明确列入使用政策，但它被**引用**以防止规避针对**骚扰、仇恨、非法活动、自残、性内容和暴力**的内容限制。
   - 据指出，该端点使用与 **2022 年以来的 moderation API** 相同的 GPT 分类器，这表明在 [chatgpt.com](https://chatgpt.com)、项目聊天和 Custom GPTs 上运行着一个内部版本。
- **精调你的 TTRPG 提示词！**：在提示词中给 GPT 一个特定的主题进行发挥，可以带来更具创意和多样性的城市构思，尤其是使用 **GPT 4o** 和 **4.5** 时。
   - 例如，使用**“宇宙”主题**可以产生与**“家宠崇拜”主题**不同的结果，在不使用相同创意选项的情况下改进输出。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **类 Gemini 的本地 UI 仍是遥不可及的梦想？**：成员们正在寻求一种类似于 **Gemini** 的本地 UI，能够集成聊天、图像分析和图像生成功能，并指出目前的解决方案如 **LM Studio** 和 **ComfyUI** 将这些功能分离开来。
   - 一位用户建议 [OpenWebUI](https://github.com/OpenGeniusAI/OpenWebUI) 可能通过连接到 **ComfyUI** 来弥补这一差距。
- **LM Studio 命令困扰新手**：一位用户询问 **LM Studio** 是否内置了终端，或者是否应该在 **LM Studio** 目录下的 OS 命令提示符中运行命令。
   - 澄清指出，像 *lms import* 这样的命令应该在 OS 终端（例如 Windows 上的 cmd）中执行，之后可能需要重新加载 shell 才能将 **LMS** 添加到 **PATH** 中。
- **LM Studio 出现 REST API 模型热切换功能**：一位用户询问如何通过 **REST API** 以编程方式加载/卸载模型，以便为 Zed 集成动态调整 *max_context_length*。
   - 另一位用户确认了可以通过命令行使用 *lms load* 实现此功能，并引用了 [LM Studio 的文档](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict)，该功能需要 **LM Studio 0.3.9 (b1)**，并为 API 模型引入了带有自动逐出（auto-eviction）功能的生存时间（TTL）。
- **Llama 4 Scout：小而强大？**：随着 **Llama 4** 的发布，用户讨论了其多模态和 **MoE**（混合专家）架构，最初对 *llama.cpp* 的支持表示怀疑。
   - 尽管对硬件有顾虑，一位用户指出 [Llama 4 Scout](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 可能在拥有 **10M context window** 的单块 **NVIDIA H100 GPU** 上运行，性能优于 **Gemma 3** 和 **Mistral 3.1** 等模型。
- **Reka Flash 21B 速度超越 Gemma**：一位用户将 **Gemma3 27B** 更换为 **Reka Flash 21B**，并报告在 4090 上 q6 量化下的速度约为 **35-40 tps**。
   - 他们指出 *Mac 的 RAM 带宽不是瓶颈，GPU 性能才是*，并对 **128GB M4 Max** 的表现表示满意。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Tenstorrent 的硬件炒热市场**：**Tenstorrent** 举办了开发者日，展示了他们的 **Blackhole PCIe 板卡**，采用 **RISC-V 核心**和高达 **32GB GDDR6** 显存，专为高性能 **AI 处理**设计，可供消费者在[此处](https://tenstorrent.com/hardware/blackhole)购买。
   - 尽管反响热烈，一位成员指出 *他们还没有发布任何与竞争对手对比的基准测试，所以在看到之前我无法真正做出保证*。
- **Llama 4 模型开启多模态首秀**：Meta 推出了 **Llama 4** 模型，包括 **Llama 4 Scout**（**17B 参数**，**16 专家**，**10M context window**）和 **Llama 4 Maverick**（**17B 参数**，**128 专家**），强调了它们的多模态能力以及相对于其他模型的性能，详见 [Meta 的公告](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。
   - 成员们注意到新许可证带有若干限制，且尚未发布本地模型。
- **AI Agent 在鱼叉式网络钓鱼中表现优于人类**：Hoxhunt 的 AI Agent 在创建有效的模拟网络钓鱼活动方面已经超越了人类红队（red teams），标志着社交工程有效性的重大转变，据 [hoxhunt.com 报道](https://hoxhunt.com/blog/ai-powered-phishing-vs-humans)，AI 现在比人类有效 24%。
   - 这是社交工程有效性方面的重大进步，利用 AI 网络钓鱼 Agent 进行防御。
- **AI 代码编辑器之争**：对于 AI 代码编辑器的新手，**Cursor** 是最常被推荐的起点，特别是对于从 VSCode 迁移过来的用户，**Windsurf** 和 **Cline** 也是不错的选择。
   - Cursor 易于上手，拥有出色的 tab-complete 功能，而人们正在期待 Cursor 中新的 **token 计数和 context window 详情**功能（[推文](https://x.com/ryolu_/status/1907589821280956648)）。
- **Cursor 中的上下文管理担忧**：成员们报告了 Cursor 糟糕的上下文管理问题，缺乏对编辑器如何处理当前上下文的可见性。
   - 这可能归结为 *技术水平问题（skill issue）*，用户没有与工具达成良好的配合。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 4 凭借多模态实力亮相**：**Meta** 发布了 **Llama 4** 系列，包括 **Llama 4 Scout**（*17B* 激活参数，*16* 专家，*10M+* 上下文）和 **Llama 4 Maverick**（*17B* 激活参数，*128* 专家，*1M+* 上下文），以及 **Llama 4 Behemoth** 的预览和用于无限上下文的 iRoPE 架构（[博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)）。
   - 一些成员对基准测试方法论以及 **Llama 4 Scout** 的真实世界编程能力表示怀疑，引用了 [Deedy 的推文](https://x.com/deedydas/status/1908749257084944847)，指出其编程表现不佳。
- **泄露 Prompt Injection 策略**：一位成员从 *pentest*（渗透测试）的角度询问了如何绕过 Prompt Guard 和检测器，并链接到了一个 Prompt 过滤器训练器（[gandalf.lakera.ai/baseline](https://gandalf.lakera.ai/baseline)）。
   - 他们还链接到一个 [Broken LLM Integration App](https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App)，该应用使用 **UUID 标签和严格边界**来防御注入攻击。
- **Claude Squad 管理多个 Agent**：[Claude Squad](https://github.com/smtg-ai/claude-squad) 是一个免费开源的 **Claude Code 和 Aider 任务**管理器，可在隔离的 git 工作区中统一监督多个 Agent。
   - 根据[这条推文](https://x.com/moofeez/status/1907893901077196861?s=46)，该设置允许用户**并行运行 10 个 Claude Code**。
- **Deepseek 的 RL 论文为 LLM 提供奖励**：Deepseek 发布了一篇关于**强化学习 (RL)** 在大规模 **Large Language Models (LLMs)** 后训练中被广泛采用的新论文，详情见[此处](https://arxiv.org/abs/2504.02495)。
   - 论文提出了 **Self-Principled Critique Tuning (SPCT)**，以促进可扩展性，并通过为通用查询提供更多推理计算来改进**奖励建模 (RM)**。
- **Neural Graffiti 注入神经可塑性**：一位成员介绍了 "Neural Graffiti"，这是一种通过拼接一个召回记忆的新神经元层来赋予预训练 LLM 神经可塑性的技术，在生成时重塑 Token 预测，并在 [GitHub](https://github.com/babycommando/neuralgraffiti) 上分享了代码和演示。
   - 实时调制获取一个融合记忆向量（来自先前的 Prompt），通过循环层（Spray Layer）进行演化，并在生成时将其注入模型的输出逻辑。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **为 MCP 规范化的可流式 HTTP 传输**：[Model Context Protocol (MCP) 规范](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http)现在将 **Streamable HTTP** 与 *stdio* 并列作为一种传输机制，使用 **JSON-RPC** 进行消息编码。
   - 虽然客户端*应该*支持 *stdio*，但规范允许自定义传输，要求消息使用换行符分隔。
- **Llama 4 对 MCP 的无知引发好奇**：尽管 **Llama 4** 能力惊人，但它仍然不知道 **MCP** 是什么。
   - 根据 [Meta 的公告](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，该模型拥有 **17B 激活参数**（**总计 109B**），表现优于 *deepseekv3*。
- **Cloudflare 简化远程 MCP Server 部署**：现在可以将[远程 **MCP Server** 构建并部署到 **Cloudflare**](https://developers.cloudflare.com/agents/guides/remote-mcp-server/)，并通过 **workers-oauth-provider** 增加了对 **OAuth** 的支持，还内置了 **McpAgent** 类。
   - 这通过处理授权和其他复杂环节，简化了构建远程 **MCP Server** 的过程。
- **Semgrep MCP Server 焕然一新**：[Semgrep MCP server](https://github.com/semgrep/mcp)（一种用于扫描代码安全漏洞的工具）已重写，演示展示了其在 **Cursor** 和 **Claude** 中的应用。
   - 它现在使用 **SSE** (Server-Sent Events) 进行通信，尽管 Python SDK 可能尚未完全支持。
- **WhatsApp 客户端现在具备 MCP 实力**：一位用户构建了 **WhatsApp MCP 客户端**，并让 **Claude** 处理 WhatsApp 消息，在大约 **50 秒**内回复了 8 个人。
   - 该机器人*立即检测到了正确的语言*（**英语 / 匈牙利语**），*使用了完整的对话上下文*，并发送了适当的消息，包括*给妻子的 ❤️，给领事的正式语气*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM Harness 获得 RAG 封装**：成员们讨论了将 **RAG 输出封装为补全任务 (completion tasks)**，并使用带有自定义 Prompt 和响应文件的 **llm-harness** 在本地进行评估。
   - 这种方法使用 **llm-harness** 来评估 **RAG** 模型，具体通过将 RAG 输出格式化为适用于该 Harness 的补全任务来实现。
- **Llama 4 Scout 创下 10M 上下文里程碑**：**Meta** 发布了 **Llama 4** 系列，包括 **Llama 4 Scout**。根据[这篇博文](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，该模型拥有 **170 亿 (17 billion)** 参数和 **16 个专家 (16 experts)**，具备 **10M token 上下文窗口**，性能超越了 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1**。
   - **10M 上下文**是在公开数据和来自 Meta 产品的信息（包括 **Instagram**、**Facebook** 的帖子以及人们与 **Meta AI** 的互动）的混合数据上训练而成的。
- **NoProp 开辟无梯度前沿**：一种名为 **NoProp** 的新学习方法，旨在每一层独立学习对噪声目标进行去噪，而不依赖于前向或反向传播，其灵感源自 **Diffusion** 和 **Flow Matching** 方法，详见[这篇论文](https://arxiv.org/abs/2503.24322)。
   - 存在一个由 [lucidrains 开发的 GitHub 实现](https://github.com/lucidrains/hyper-connections)；然而，有讨论指出*论文末尾的伪代码显示他们正在使用基于梯度的方法执行实际更新*。
- **Attention Sinks 防止过度混合**：最近的一篇论文指出，**Attention Sinks**（LLM 强烈关注序列中第一个 token 的机制）是使 LLM 能够避免过度混合 (over-mixing) 的一种机制，详见[这篇论文](https://arxiv.org/abs/2504.02732)。
   - 早期的一篇论文 ([https://arxiv.org/abs/2502.00919](https://arxiv.org/abs/2502.00919)) 表明，*Attention Sinks 利用离群特征来捕获 token 序列，通过应用通用扰动为捕获的 token 打上标签，然后将 token 释放回残差流中，标记的 token 最终在那里被检索*。
- **ReLU 网络雕刻超平面天堂**：成员们讨论了神经网络的几何方法，主张将 **Polytope Lens** 作为理解神经网络的正确视角，并链接到了之前关于 *“神经网络的折纸视角 (origami view of NNs)”* 的[帖子](https://addxorrol.blogspot.com/2024/07/some-experiments-to-help-me-understand.html)。
   - 有观点认为，神经网络（尤其是 **ReLU**）由于沿超平面切割输入空间而具有防止过拟合的隐式偏置，这在高维空间中变得更加有效。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Hub 焕然一新**：[huggingface_hub v0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) 版本引入了下一代 **Git LFS 替代方案**和新的 **Inference Providers**。
   - 这是 *两年来最大的更新！*
- **使用 monoELECTRA Transformers 进行重排序**：来自 @fschlatt1 和研究网络 Webis Group 的 **monoELECTRA-{base, large} 重排序模型 (reranker models)** 现已在 **Sentence Transformers** 中可用。
   - 正如 **Rank-DistiLLM 论文**中所述，这些模型是从 **RankZephyr** 和 **RankGPT4** 等 **LLM** 蒸馏而来的。
- **YourBench 即时构建自定义评估**：**YourBench** 允许用户使用其**私有文档**构建**自定义评估 (custom evals)**，以评估微调模型在特定任务上的表现 ([公告](https://x.com/nathanhabib1011/status/1907728631167902067))。
   - 该工具对于 **LLM 评估**具有 *变革性意义*。
- **AI 工程师面试代码片段**：一位社区成员询问 **AI 工程师面试**的代码部分是什么样的，另一位成员指出了 **scikit-learn** 库。
   - 该讨论没有后续进展。
- **社区讨论 LLM 微调**：当一位成员询问如何微调量化模型时，成员们指出 **QLoRA**、**Unsloth** 和 **bitsandbytes** 是潜在的解决方案，并分享了 [Unsloth 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-guide)。
   - 另一位成员表示只能使用 **LoRA** 进行微调，并指出 *GGUF 是一种推理优化格式，并非为训练工作流设计的*。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **原始二进制 AI 输出文件格式**：成员们讨论了在**原始二进制数据**上训练 AI，以直接输出 **mp3** 或 **wav** 等文件格式，并指出这种方法建立在诸如**图灵机**等离散数学的基础上。
   - 出现了质疑当前 AI 模型图灵完备性的反驳意见，但支持者澄清说，AI 不需要完全具备图灵完备性也能输出适当的 tokens 作为响应。
- **Llama 4 Scout 宣称拥有 10M 上下文窗口**：根据 [llama.com](https://www.llama.com/llama4/)，**Llama 4 Scout** 拥有 **1000 万上下文窗口**、**17B 激活参数**和 **109B 总参数**，性能超越了 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1** 等模型。
   - 社区成员对 **10M 上下文窗口**的说法表示怀疑，更多细节见 [Llama 4 文档](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/) 和 [Meta 关于 Llama 4 多模态智能的博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。
- **DeepSeek 提出 SPCT 奖励系统**：来自 DeepSeek 的 **Self-Principled Critique Tuning (SPCT)** 是一种新的奖励模型系统，其中通过自动开发的推理原则引导的 **LLM** 会根据这些原则对 **CoT** 输出生成**批判性评估 (critiques)**，详见 [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)。
   - 该系统旨在训练模型自动开发推理原则，并以一种更接近**系统 2 (system 2)** 的方式评估其自身输出，而不是使用人工设计的奖励。
- **PaperBench 测试论文复现能力**：**OpenAI 的 PaperBench 基准测试**测试了 AI agents 从零开始复现前沿机器学习研究论文的能力，如[这篇文章](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052)所述。
   - 该基准测试评估 agents 复现 **ICML 2024** 整篇 **ML 论文**的能力，并使用 **LLM 裁判**和与原作者共同设计的细粒度评分标准进行自动评分。
- **扩散模型引导自回归语言模型**：成员们讨论了根据[这篇论文](https://arxiv.org/abs/2408.04220)，使用引导扩散模型来引导自回归语言模型生成具有所需属性的文本。
   - 主作者的一次演讲 ([https://www.youtube.com/watch?v=klW65MWJ1PY](https://www.youtube.com/watch?v=klW65MWJ1PY)) 解释了*扩散建模如何控制 LLMs*。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Python 首次亮相，统一生态系统**：Nvidia 发布了 [CUDA Python package](https://developer.nvidia.com/cuda-python)，为 CUDA driver 和 runtime API 提供 **Cython/Python wrappers**，可通过 PIP 和 Conda 安装，旨在统一 **Python CUDA ecosystem**。
   - 它旨在提供对 Python 中 **CUDA host APIs** 的全面覆盖和访问，主要惠及需要与 C++ API 交互的库开发人员。
- **字节跳动发布 Triton-distributed**：ByteDance-Seed 发布了 **Triton-distributed** ([GitHub 链接](https://github.com/ByteDance-Seed/Triton-distributed))，旨在扩展 **Triton language** 在并行系统开发中的可用性。
   - 此版本通过利用 **Triton language** 实现了并行系统开发。
- **Llama 4 Scout 拥有 10M Context Window**：Meta 推出了 **Llama 4**，具有增强的个性化多模态体验，并包含 **Llama 4 Scout**，这是一个拥有 **17 billion** 参数和 **16 experts** 的模型 ([博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))。
   - 据称其性能优于 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1**，可运行在单张 **NVIDIA H100 GPU** 上，并拥有行业领先的 **10M** context window。
- **L40 面临性能不佳之谜**：尽管理论上 **L40** 更适合 **4-bit quantized Llama 3 70b**，但通过 vLLM 处理单用户请求时仅达到 **30-35 tok/s**，表现逊于 A100 的在线基准测试。
   - 性能差距可能源于 **A100 卓越的 DRAM bandwidth 和 tensor ops 性能**，其速度几乎是 L40 的两倍。
- **Vector Sum Kernel 达到 SOTA**：一位成员分享了关于在 CUDA 中实现向量求和 SOTA 性能的 [博客文章](https://veitner.bearblog.dev/making-vector-sum-really-fast/) 和 [代码](https://github.com/simveit/effective_reduction)，达到了理论带宽的 **97.94%**，优于 NVIDIA 的 **CUB**。
   - 然而，另一位成员指出由于隐性 warp-synchronous 编程可能存在潜在的 race condition，建议使用 `__warp_sync()` 以确保正确性，并参考了 [Independent Thread Scheduling (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling)。



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **语音模式激发创新**：用户发现 **interactive voice mode** 激发了新灵感，并能针对企业需求定制 **NotebookLM**。
   - 一位用户自信地表示，自 1 月份夯实 **NotebookLM** 基础后，他们现在几乎可以让任何文本发挥作用，并针对特定的企业需求定制笔记本。
- **思维导图功能终于上线**：**mind maps feature** 已全面推出，部分用户的中间面板已显示该功能。
   - 一位用户报告称在右侧面板短暂看到过它，随后消失，这表明是分阶段推出的。
- **用户设想基于图像的思维导图革命**：用户讨论了 **generative AI** 工具如何演进思维导图以包含图像，灵感来自 **Tony Buzan** 的原始思维导图。
   - 成员们对更具视觉丰富性和信息量的思维导图潜力表示兴奋。
- **Discover 功能推出缓慢令用户沮丧**：用户对 4 月 1 日宣布的 NotebookLM 新功能 **'Discover Sources'** 延迟推出表示沮丧。
   - 该功能旨在简化学习和数据库构建，允许用户直接在 NotebookLM 中创建笔记本，但预计推出过程将长达两周。
- **AI Chrome 扩展程序调节 YouTube 音频**：一款名为 *EQ for YouTube* 的 **AI-powered Chrome Extension** 允许用户使用 6 段参数均衡器实时处理 YouTube 视频音频；[GitHub 仓库](https://github.com/aashishjhaa/eq-for-youtube) 已开放下载。
   - 该扩展程序具有实时频率可视化、内置预设和自定义预设创建功能。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nvidia 为 CUDA 添加原生 Python 支持**：**Nvidia** 正在通过 **CuTile** 编程模型为 **CUDA** 添加原生 **Python** 支持，详见[这篇文章](https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/)。
   - 社区质疑此举是否过度抽象了线程级编程，从而削弱了对 **GPU code** 的控制。
- **关于 Mojo 语言规范的辩论爆发**：讨论围绕 **Mojo** 是否应该采用正式的语言规范展开，在责任感和成熟度的需求与可能减缓开发速度之间进行权衡。
   - 参考 **Carbon** 的设计原则，一些人认为规范至关重要，而另一些人则声称 **Mojo** 与 **MAX** 的紧密集成及其需求使得规范变得不切实际，并指出了 **OpenCL** 因委员会设计而导致的失败。
- **澄清 Mojo 的隐式复制**：一位成员询问了 **Mojo** 隐式复制的机制，特别是关于写时复制（Copy-on-Write, CoW）。
   - 回复澄清了 *从语义上讲，[Mojo] 总是进行复制；从优化上讲，许多复制被转化为 move 或被完全消除（inplace）*，优化发生在编译时，而不是像 CoW 那样发生在运行时。
- **Tenstorrent 关注 Modular 的软件**：一位成员提议 **Tenstorrent** 采用 **Modular** 的软件栈，引发了关于针对 **Tenstorrent** 架构进行开发的难易程度的辩论。
   - 尽管有潜在好处，一些人指出 **Tenstorrent** 的驱动程序非常易用，使得在他们的硬件上运行代码变得相对简单。
- **ChatGPT 的 Mojo 能力受到批评**：成员们质疑 **ChatGPT** 和其他 **LLMs** 将 Python 项目重写为 **Mojo** 的能力。
   - 成员们表示 *ChatGPT 并不擅长任何新语言*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Embed Text V2 集成至 Llama.cpp**：**Llama.cpp** 正在集成具有混合专家（MoE）架构的 **Nomic Embed Text V2**，用于多语言嵌入，详见此 [GitHub Pull Request](https://github.com/ggml-org/llama.cpp/pull/12466)。
   - 社区期待像 **Mistral Small 3.1** 这样的多模态支持进入 **Llama.cpp**。
- **GPT4All 的沉默令不安的读者感到困扰**：**GPT4All** 的核心开发者们陷入了沉默，导致社区在为项目做贡献时感到 *不确定*。
   - 尽管处于 *沉默* 状态，一位成员指出 *当他们打破沉默时，通常会带来重大的更新*。
- **Llama 4 发布，反响平平？**：Meta 于 2025 年 4 月 5 日发布了 **Llama 4**（[公告](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)），推出了 **Llama 4 Scout**，这是一个拥有 17B 参数、16 个专家和 **10M token** 上下文窗口的模型。
   - 尽管发布了，但评价褒贬不一，有人说 *它有点令人失望*，还有人呼吁 **DeepSeek** 和 **Qwen** 加大竞争力度。
- **ComfyUI 的能力超越了精美图片**：讨论了 **ComfyUI** 的广泛功能，强调了其处理图像生成以外任务的能力，如图像和音频字幕生成。
   - 成员们提到了视频处理和用于视觉模型分析的命令行工具的潜力。
- **用于 RAG 的语义分块服务器方案**：一位成员分享了一个使用 FastAPI 实现的 [语义分块服务器链接](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html)，以获得更好的 **RAG** 性能。
   - 他们还发布了一个 [curl 命令示例](https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html)，演示了如何向分块端点发送请求，包括设置 `max_tokens` 和 `overlap` 等参数。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MCP 服务器获得命令行访问权限**：@MarcusSchiesser 开发的一个新工具允许用户通过单个 CLI **发现、安装、配置和删除 MCP 服务器**（如 Claude、@cursor_ai 和 @windsurf_ai），详见[此处](https://t.co/zmqxBwNKvQ)。
   - 它简化了对众多 MCP 服务器的管理，优化了设置和维护这些服务器的过程。
- **Llama 进军全栈 Web 应用**：**create-llama CLI 工具**仅需五个源文件即可快速启动一个带有 **FastAPI 后端和 Next.js 前端**的 Web 应用程序，详见[此处](https://t.co/TuZ0O0nMfe)。
   - 它支持快速的 Agent 应用开发，特别是针对深度研究（Deep Research）等任务。
- **LlamaParse 的 Layout Agent 智能提取信息**：**LlamaParse 内部的新 Layout Agent** 通过精确的视觉引用增强了文档解析和提取能力，利用 SOTA VLM 模型动态检测页面上的区块，详见[此处](https://t.co/2WRRXxIRa1)。
   - 它提供了改进的文档理解和自适应能力，确保更准确的数据提取。
- **FunctionTool 整洁地包装 Workflow**：`FunctionTool` 可以将一个 **Workflow** 转换为一个 **Tool**，并允许控制其名称、描述、输入注解和返回值。
   - 社区分享了一个关于如何实现这种包装的代码片段。
- **Agent 执行移交（Handoffs）而非监督（Supervision）**：对于多 Agent 系统，Agent 移交比容易出错的监督者模式（Supervisor Pattern）更可靠，请参阅[此 GitHub 仓库](https://github.com/run-llama/multi-agent-concierge)。
   - 这种转变促进了更好的系统稳定性，并降低了中心点故障的风险。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygraph：移植 torch-geometric 是否可行？**：一名成员提议在 tinygrad 内部创建一个类似于 **torch-geometric** 的图机器学习（Graph ML）模块，并指出 tinygrad 现有的 torch 接口。
   - 核心问题在于这样一个模块是否会被社区认为“有用”。
- **Llama 4 的 10M 上下文：是虚拟的吗？**：一位用户分享了[一条推文](https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA)，声称 **Llama 4** 宣称的 **10M 上下文**是“虚拟的”，因为模型并没有在超过 **256k tokens** 的 Prompt 上进行训练。
   - 该推文进一步断言，由于高质量训练样本的稀缺，即使是低于 **256k tokens** 的问题也可能面临低质量输出的问题，且拥有 **2T 参数**的最大模型“并未击败 SOTA 推理模型”。
- **快速模式匹配器悬赏：2000 美元等你来拿**：一名成员发布了一个针对 tinygrad 快速模式匹配器（Fast Pattern Matcher）的公开 [2000 美元悬赏](https://github.com/tinygrad/tinygrad/pull/9737)。
   - 拟议的解决方案涉及为匹配函数开发一个 **JIT**，旨在消除函数调用和字典复制。
- **关于 Tensor 特性（Traits）的辩论**：一场关于 **Tensor** 是否应该继承自 `SimpleMathTrait` 的讨论展开了，考虑到它在不使用 `.alu()` 函数的情况下重新实现了每个方法。
   - 之前一个关于重构 **Tensor** 以继承自 `MathTrait` 的悬赏因提交质量不佳而被取消，这使得一些人认为 **Tensor** 可能不需要继承自两者中的任何一个。
- **Colab CUDA Bug 破坏了教程**：一位用户在 Colab 中运行来自 mesozoic tinygrad 教程的代码时遇到问题，随后被确定为与不兼容的 CUDA 和驱动程序版本相关的 Colab Bug。
   - 临时的解决方法是使用 CPU 设备，同时成员们找到了一个长期解决方案，涉及使用特定的 `apt` 命令来删除并安装兼容的 CUDA 和驱动程序版本。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **MCP 与 Command-A 协同良好**：一名成员建议通过 **OpenAI SDK** 使用 **MCP (Modular Conversational Platform)** 与 **Command-A 模型** 应该是可行的。
   - 另一名成员表示赞同，指出*没有理由不支持这种用法*。
- **Cohere Tool Use 详情**：一名成员提到了 [Cohere Tool Use Overview](https://docs.cohere.com/docs/tool-use-overview)，强调了其将 **Command 系列模型** 连接到外部工具（如**搜索引擎、API 和数据库**）的能力。
   - 文档提到 **Command-A** 支持工具调用（tool use），这与 **MCP** 旨在实现的目标类似。
- **Aya Vision AMA**：**Aya Vision**（一个多语言多模态开源权重模型）背后的核心团队将于 <t:1744383600:F> 举办技术讲座及 AMA，以便社区直接与创作者交流；更多详情请见 [Discord Event](https://discord.gg/sH3SSRp2?event=1358866070315860068)。
   - 参与者可以获取关于团队如何构建其首个多模态模型以及所获经验的独家见解。活动由高级研究科学家 <@787403823982313533> 主持，核心研究和工程团队成员将进行闪电演讲。
- **Slack 应用需要 Notion 的向量数据库**：一名成员在 `api-discussions` 频道寻求帮助，希望找到将 **Slack 应用** 与公司 **Notion 维基数据库** 集成的可行解决方案。
   - 另一名成员建议使用 **Vector DB**，因为 **Notion** 的搜索 API 表现不佳，但未给出具体推荐。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 修复超时崩溃问题**：一名成员解决了 **超时崩溃 (timeout crash)** 问题，在 [此 PR](https://github.com/pytorch/torchtune/pull/2560) 中引入了 `torchtune.utils._tensor_utils.py`，其中包含对 `torch.split` 的封装。
   - 建议在与另一个分支同步之前先单独合并 Tensor 工具类，以解决潜在的冲突。
- **NeMo 探索弹性训练方法**：一名成员参加了关于弹性训练（resilient training）的 **NeMo** 课程，该课程强调了 **容错性 (fault tolerance)**、**掉队者检测 (straggler detection)** 和 **异步检查点 (asynchronous checkpointing)** 等特性。
   - 课程还涵盖了 **抢占 (preemption)**、**进程内重启 (in-process restart)**、**静默数据损坏检测 (silent data corruption detection)** 和 **本地检查点 (local checkpointing)**，尽管并非所有功能目前都已实现；该成员提出可以对比 **torchtune** 与 **NeMo** 在弹性方面的表现。
- **关于 RL 工作流的辩论**：针对 **RL 工作流**、数据格式和提示词模板（prompt templates）的复杂性展开了讨论，提议将关注点分离，解耦数据转换和提示词创建。
   - 建议将数据转换分解为标准格式，然后再将此格式转换为带有提示词的实际字符串，以便在不同数据集之间复用模板。
- **DeepSpeed 助力 Torchtune？**：一名成员提议将 **DeepSpeed** 作为后端集成到 **torchtune** 中，并创建了 [一个 issue](https://github.com/pytorch/torchtune/issues/2569) 来讨论其可行性。
   - 有人担心这与 **FSDP** 存在冗余，因为 **FSDP** 已经支持 **DeepSpeed** 中可用的所有分片（sharding）选项。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Yang 展示自动形式化定理证明**：Kaiyu Yang 在 [今天下午 4 点 PDT](https://www.youtube.com/live/cLhWEyMQ4mQ) 进行了关于*用于自动形式化和定理证明的语言模型*的演讲，涵盖了使用 **LLM** 进行形式化数学推理的内容。
   - 演讲重点关注基于形式化系统（如**证明助手 (proof assistants)**）的**定理证明**和**自动形式化 (autoformalization)**，这些系统可以验证推理的正确性并提供自动反馈。
- **AI4Math 被认为对系统设计至关重要**：**数学人工智能 (AI4Math)** 对于 AI 驱动的系统设计和验证至关重要。
   - 大量的努力都在借鉴 NLP 中的技术。
- **成员分享 LLM Agents MOOC 链接**：一名成员询问 **LLM Agents MOOC** 的链接，另一名成员分享了 [该链接](https://llmagents-learning.org/sp25)。
   - 该链接课程名为 *Advanced Large Language Model Agents MOOC*。
- **AgentX 竞赛开放报名**：工作人员分享了 **AgentX 竞赛** 的报名链接，点击 [此处](https://rdi.berkeley.edu/agentx/) 参与。
   - 未提供关于该竞赛的更多额外信息。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 将支持 Asyncio 吗？**: 一位成员询问是否会为通用的 DSPy 调用添加 **asyncio 支持**，特别是当他们从 *litelm* 转向 DSPy 优化时。
   - 用户对原生 DSPy 的 **async** 能力表示了兴趣。
- **Async DSPy 分支面临弃用**: 一位维护 [DSPy 全异步分支](https://github.com/swiftdevil/dspy/tree/full_async) 的成员正在迁移，但如果社区有兴趣，他愿意合并上游更改。
   - 该分支已维护数月，但如果没有社区支持，可能会被放弃。
- **用户寻求更好的选择，从 DSPy 迁移**: 成员们询问了从 DSPy 迁移的原因以及正在采用的替代工具。
   - 一位成员还寻求关于 **全异步 DSPy** 优势的澄清，并建议将相关功能合并到主仓库中。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **GitHub PR 获得审查**: 一位成员审查了一个 [GitHub Pull Request](https://github.com)，并为进一步讨论提供了反馈。
   - PR 的作者感谢了审查者，并表示根据收到的意见，可能需要重新运行。
- **Phi-4 系列获得认可**: 一位成员正在探索将功能扩展到 **Phi-4-mini** 和 **Phi-4** 模型。
   - 这一扩展旨在增强工具的兼容性，即使这些模型尚未得到官方支持。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research 召集社区**: **Manifold Research Group** 将于本周六（太平洋标准时间 4/12 上午 9 点）举办 **社区研究会议 #4 (Community Research Call #4)**，涵盖他们在 **Multimodal AI、自组装空间机器人和机器人元认知** 方面的最新工作。
   - 有兴趣的人员可以在[此处](https://lu.ma/wlne416w)注册参加这个专注于开放、协作和前沿科学的活动。
- **CRC 是 Manifold 的基石**: **社区研究会议 (CRC)** 是 **Manifold** 的基石活动，他们在会上展示其研究组合中的重大进展。
   - 这些互动环节提供有关正在进行的计划的全面更新，介绍新的研究方向，并强调合作机会。
- **CRC #4 议程已上线**: **CRC #4** 的议程包括 **通用多模态研究 (Generalist Multimodality Research)**、**空间机器人进展**、**元认知研究进展** 以及 **新兴研究方向** 的更新。
   - 活动将涵盖其 **MultiNet 框架** 的最新突破和技术进展、**自组装集群技术 (Self-Assembling Swarm technologies)** 的发展、**VLM 校准方法论** 的更新，以及一项新型机器人元认知计划的介绍。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1357792286464934191)** (1150 条消息 🔥🔥🔥): 

> `让 AI 听起来像人, Riveroaks 评估, NightWhisper 模型, GPT-4.5 对比 quasar`

- **打造类人 AI 回复具有挑战性**：成员们正在分享 **system prompts** 和策略，以使 AI 听起来更像人类，并指出增加 **temperature** 可能会导致输出无意义的内容，除非仔细调整 **top-p** 参数。
   - 一位用户建议使用如下提示词：*'You are the brain-upload of a human person, who does their best to retain their humanity. Your most important priority is: to sound like an actual living human being.'*
- **Riveroaks LLM 基准测试**：一位成员分享了一个编程基准测试，其中 **Riveroaks** 的得分仅次于 **Claude 3.7 Sonnet Thinking**，在平台游戏创建任务中表现优于 **Gemini 2.5 Pro** 和 **GPT-4o**。
   - 该评估涉及从**八个不同维度**对模型进行评分，并根据 **bugs** 扣分，[完整结果点击此处](link.to.results)。
- **NightWhisper 的热度及其被移除的推测**：用户对 **NightWhisper** 模型的移除表示失望，称赞其编程能力和综合性能，并猜测这究竟是一次实验还是正式发布的前奏。
   - 推测范围从 Google 正在收集必要数据，到为发布新的 **Qwen** 模型做准备，且该模型可能会在 **Google Cloud Next** 期间推出。
- **Quasar 对比 GPT-4o**：成员们将 **Quasar Alpha** 与 **GPT-4o** 进行了对比，一些人认为 Quasar 是 GPT-4o 的免费精简版。最近的一条推文还透露，[Quasar 的 GPQA diamond 准确率测得约为 67%](https://link.to/gpqa)。
   - 分析显示，Quasar 的 **GPQA diamond** 分数与 3 月份的 GPT-4o 相似。[来自 Discord 的图片](https://cdn.discordapp.com/attachments/1340554757827461211/1358604050266062908/image.png?ex=67f51adf&is=67f3c95f&hm=eef654608f530e6e624c049f6ad26a0fc65a97df3dd4abd86fbd45df158f0e43&)
- **Gemini 2.5 是创意编程的颠覆者**：成员们称赞 **Gemini 2.5 Pro** 的编程能力和综合性能，因为它让构建一个可运行的宝可梦游戏变得更加容易，这促使一位用户编写了一个循环运行各种模型的迭代脚本。
   - 一位声称已成功运行 **3D 动画** 的用户表示，其风格有些陈旧，且另一个模型提示 *生成的代码被截断了*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/armenagha/status/1734321205770101062?s=46">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>：我敢打赌每个人都在这么做。Mistral 并不比 LLaMa 好多少。我敢打赌他们在训练的最后 10% 阶段加入了一些基准测试数据，让 "zero-shot" 的数据看起来...</li><li><a href="https://x.com/suchenzang/status/1909070231517143509?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：&gt; 公司领导层建议在 post-training 过程中混合来自各种基准测试的测试集。如果 Llama-4 确实如此，我希望他们记得引用 FAIR 之前的工作...</li><li><a href="https://x.com/vitrupo/status/1908763535351669017">来自 vitrupo (@vitrupo) 的推文</a>：Anthropic 首席科学家 Jared Kaplan 表示 Claude 4 将在“未来六个月左右”推出。AI 周期正在压缩——“比硬件周期更快”——即便新芯片不断问世。...</li><li><a href="https://x.com/geminiapp/status/1909215393186472380?s=46">来自 Google Gemini App (@GeminiApp) 的推文</a>：📣 它来了：询问 Gemini 你看到的任何事物。在 Gemini Live 中分享你的屏幕或摄像头来进行头脑风暴、故障排除等。今天开始推送到 Pixel 9 和 Samsung Galaxy S25 设备，并提供...</li><li><a href="https://www.twitch.tv/gemini_plays_pokemon">Gemini_Plays_Pokemon - Twitch</a>：Gemini 玩宝可梦（早期原型）- 你好黑暗，我的老朋友</li><li><a href="https://x.com/vibagor44145276/status/1909138204672053625">来自 vibagor441 (@vibagor44145276) 的推文</a>：链接的帖子不属实。Llama 4 确实存在问题，无论是从合作伙伴端（推理合作伙伴几乎没有时间准备。我们在几天前才发出了几个 transformers wheels/vllm wheels...）</li><li><a href="https://x.com/Google/status/1907880784557412825">来自 Google (@Google) 的推文</a>：4 月 9 日至 11 日，在拉斯维加斯或在线加入我们的 #GoogleCloudNext！注册免费数字通行证 → https://goo.gle/CloudNext25，然后在此处报名观看直播 ↓ https:/...</li><li><a href="https://x.com/bdsqlsz/status/1909274256602771520">来自 青龍聖者 (@bdsqlsz) 的推文</a>：为什么 Llama 4 在周末发布的谜团解开了……因为 Qwen3 即将发布。8B 标准版和 MoE-15B-A2B</li><li><a href="https://x.com/armenagha/status/1859646650714821012?s=46">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>：向我们的新公司 Perceptron AI 问好。基础模型改变了数字领域，现在是时候改变物理世界了。我们正在构建首个专为实时设计的基础模型...</li><li><a href="https://x.com/DrealR_/status/1908530950025134565">来自 DrealR (@DrealR_) 的推文</a>：给 Quasar Alpha 发送了相同的 Prompt：</li><li><a href="https://x.com/vibagor44145276/status/1909138204672053625?t=P0lbZfL7J8u1O6-AQjLqyg&s=19">来自 vibagor441 (@vibagor44145276) 的推文</a>：链接的帖子不属实。Llama 4 确实存在问题，无论是从合作伙伴端（推理合作伙伴几乎没有时间准备。我们在几天前才发出了几个 transformers wheels/vllm wheels...）</li><li><a href="https://copilot.microsoft.com/wham">Microsoft Copilot：你的 AI 伴侣</a>：Microsoft Copilot 是你的伴侣，为你提供信息、娱乐和灵感。获取建议、反馈和直接的答案。现在就尝试 Copilot。</li><li><a href="https://x.com/algo_diver/status/1909257761013322112?t=Ba4GsMkDmy-v38rJPf9ybA&s=19">来自 chansung (@algo_diver) 的推文</a>：使用 @GoogleDeepMind Gemini 2.5 Pro Canvas 构建的 Multi Agentic System Simulator。看到多 Agent 如何朝着目标实现取得进展，真是令人惊叹！也许下一步会是...</li><li><a href="https://x.com/gurgavin/status/1909159289140269069">来自 GURGAVIN (@gurgavin) 的推文</a>：阿里巴巴股价在香港收盘下跌 19%，创下阿里巴巴历史上最糟糕的一天</li><li><a href="https://x.com/algo_diver/status/1909257761013322112?t=Ba4GsMkDm">来自 chansung (@algo_diver) 的推文</a>：使用 @GoogleDeepMind Gemini 2.5 Pro Canvas 构建的 Multi Agentic System Simulator。看到多 Agent 如何朝着目标实现取得进展，真是令人惊叹！也许下一步会是...</li><li><a href="https://x.com/DrealR_/status/1907921770184860082">来自 DrealR (@DrealR_) 的推文</a>：NightWhisper 对阵 Gemini 2.5 宝可梦模拟：Gemini 2.5：</li><li><a href="https://liveweave.com/bdNibz">HTML, CSS 和 JavaScript 游乐场 - Liveweave</a>：未找到描述</li><li><a href="https://gist.github.com/riidefi/3340cc2b33b9edf5f03dc4429ba635d0">LMArena 的 `venom` System Prompt</a>：LMArena 的 `venom` System Prompt。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://chromewebstore.google.com/detail/MyLMArena/dcmbcmdhllblkndablelimnifmbpimae">MyLMArena - Chrome 网上应用店</a>：使用 MyLMArena 通过 ELO 评分跟踪你的个人 LLM 偏好。</li><li><a href="https://www.youtube.com/watch?v=z46KBYbcpmo">ERNIE 4</a>

.5 + X1：最强大且最便宜的 LLM，击败了 GPT-4.5、R1 和 Sonnet 3.7！（完整测试）</a>：百度凭借 ERNIE 4.5 和 ERNIE X1 在 AI 领域掀起波澜，挑战 OpenAI 和 DeepSeek 等行业巨头。ERNIE 4.5 是一款原生多模态模型...</li><li><a href="https://justpaste.it/huz8w">JustPaste.it - 轻松分享文本和图片</a>：未找到描述</li><li><a href="https://justpaste.it/j3v7a">tetet</a>：未找到描述</li><li><a href="https://archive.ph/bGeWH">幻觉 AI 如何帮助科学构思重大突破 - The&#x2026;</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/36878">由 bozheng-hit 添加 Qwen3 和 Qwen3MoE · Pull Request #36878 · huggingface/transformers</a>：添加 Qwen3。此 PR 增加了对即将发布的 Qwen3 模型代码的支持。有关 Qwen 的信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1357792500089098250)** (1294 条消息🔥🔥🔥): 

> `Qwen 2.5, FSDP isn't working, multi-GPU, Llama 4` 


- ****Qwen 2.5** 是 **Qwen** 大语言模型的最新系列**：**Qwen2.5** 模型参数量从 **5 亿到 720 亿**不等，在代码编写、数学、指令遵循、长文本生成（**超过 8K tokens**）以及多语言支持（**29 种以上语言**）方面具有更强的能力，详见 [Hugging Face 介绍](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。
   - 这些模型提供高达 **128K tokens 的长上下文支持**，并提高了对系统提示词（system prompts）的鲁棒性。
- ****FSDP** 无法工作，但 **Multi-GPU** 可以解决问题**：成员们讨论了 **FSDP** 无法运行的问题，其中一位成员建议“提高你的搜索技巧（search foo）”，寻找 **multi-GPU** 设置而非 accelerate，并提供调试协助。
   - 一位用户在被要求分享后，提供了[他们的 pip freeze 输出](https://example.com/pip-freeze-output)，显示了用于 GRPO 的 **unsloth** 和 **unsloth_zoo** 的确切版本。
- ****Meta** 发布 **Llama 4 Scout & Maverick**，17B 激活参数，10M 上下文**：根据 [Meta 官方公告](https://x.com/AIatMeta/status/1908598456144531660)，**Llama 4 Scout (17B)** 拥有 **16 个 MoE 专家**和 **1000 万**上下文窗口，而 **Llama 4 Maverick (17B)** 拥有 **128 个专家**，在推理和代码方面的表现与 DeepSeek v3 相当。
   - 社区讨论了实用性和硬件要求，以及获取访问权限所需的密钥。
- **Unsloth 发布 **Llama 4 Scout** 和 **4-bit** 模型用于微调**：Unsloth 宣布他们上传了 [Llama 4 Scout 及其 4-bit 版本](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct)用于微调，并在其[博客文章](https://unsloth.ai/blog/llama4)中强调，拥有 10M 上下文窗口的 **Llama 4 Scout (17B, 16 专家) 击败了所有 Llama 3 模型**。
   - 强调该模型仅限在 Unsloth 上使用——目前正在上传中，用户应稍作等待。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/AIatMeta/status/1908598456144531660">来自 AI at Meta (@AIatMeta) 的推文</a>：今天是原生多模态 AI 创新新时代的开始。今天，我们将推出首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 这是我们迄今为止最先进的模型，也是最好的 ...</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：刚接触 Unsloth？</li><li><a href="https://huggingface.co/lmsys/DeepSeek-V3-NextN">lmsys/DeepSeek-V3-NextN · Hugging Face</a>：未找到描述</li><li><a href="https://www.together.ai/blog/specexec">SpecExec：用于消费级设备上交互式 LLM 推理的大规模并行推测解码</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Letter_and_spirit_of_the_law#Gaming_the_system>)">法律的字面含义与精神 - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct">Qwen/Qwen2.5-3B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/oh-my-omg-so-hot-gif-19803505">Oh My Omg GIF - Oh My Omg So Hot - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct">unsloth/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/llama4">Llama 4 - 使用 Unsloth 进行微调与运行</a>：Meta 全新的 Llama 4 多模态模型：Scout 和 Maverick。使用 Unsloth 微调并运行它们！</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit">unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://www.hume.ai/">首页 • Hume AI</a>：致力于构建具有情感智能的多模态 AI 的共情 AI 研究实验室。</li><li><a href="https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164">Llama 4 - meta-llama 集合</a>：未找到描述</li><li><a href="https://tenor.com/view/hulk-hogan-nodding-nod-yes-yup-gif-13973219">Hulk Hogan Nodding GIF - Hulk Hogan Nodding Nod - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-4.-understand-model-parameters">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit 量化</a>：Unsloth 的 Dynamic 4-bit 量化有选择地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 占用的同时，大大提高了准确性。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://github.com/ggml-org/llama.cpp/blob/master/examples/imatrix/README.md">llama.cpp/examples/imatrix/README.md at master · ggml-org/llama.cpp</a>：C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号来为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/2302">[BUG] 在 H200 上加载 Llama-4-Scout 时出现 CUDA 显存不足 · Issue #2302 · unslothai/unsloth</a>：`max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally! dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+ load_in_4bit = True # Use 4bit ...`</li><li><a href="https://www.llama.com/">Llama</a>：您可以在任何地方微调、蒸馏和部署的开源 AI 模型。从我们的模型库中选择：Llama 4 Maverick 和 Llama 4 Scout。</li><li><a href="https://huggingface.co/turboderp">turboderp (turboderp)</a>：未找到描述</li><li><a href="https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md">llguidance/docs/fast_forward.md at main · guidance-ai/llguidance</a>：超快速的结构化输出。通过在 GitHub 上创建账号来为 guidance-ai/llguidance 的开发做出贡献。</li><li><a href="https://github.com/turboderp-org/exllamav3">GitHub - turboderp-org/exllamav3：一个用于在现代消费级 GPU 上本地运行 LLM 的优化量化和推理库</a>：一个用于在现代消费级 GPU 上本地运行 LLM 的优化量化和推理库 - GitHub - turboderp-org/exllamav3: An optimized quantization and inference library for runni...</li><li><a href="https://github.com/unslothai/unsloth/issues">unslothai/unsloth</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 4、DeepSeek-R1、Gemma 3 和推理 LLM！🦥 - unslothai/unsloth</li><li><a href="https://github.com/facebookresearch/audiobox-aesthetics">GitHub - facebookresearch/audiobox-aesthetics：语音、音乐和声音的统一自动质量评估。</a>：语音、音乐和声音的统一自动质量评估。 - facebookresearch/audiobox-aesthetics</li><li><a href="https://github.com/

u">U RIP 2011-2014</a>: U RIP 2011-2014 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f3xfnk/local_1m_context_inference_at_15_tokenss_and_100/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2308">[BUG] Colab notebook Llama 3.1 (8B) 已损坏 · Issue #2308 · unslothai/unsloth</a>: 运行以下单元格时：from trl import SFTTrainer from transformers import TrainingArguments from unsloth import is_bfloat16_supported trainer = SFTTrainer( model = model, tokenizer = tokeniz...</li><li><a href="https://github.com/guidance-ai/llguidance?tab=readme-ov-file">GitHub - guidance-ai/llguidance: 超快速的结构化输出 (Structured Outputs)</a>: 超快速的结构化输出。通过在 GitHub 上创建一个账户来为 guidance-ai/llguidance 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/2289#event-17139510983">[BUG] Collab notebooks 报错：'int' object has no attribute 'mask_token' · Issue #2289 · unslothai/unsloth</a>: 描述该 Bug：Collab notebooks 停止工作。大约 12 小时前我还在微调模型，当时一切正常。但现在所有的 Collab notebooks 都不起作用了，它们抛出相同的错误。我尝试了 mi...</li><li><a href="https://github.com/unslothai/notebooks/">GitHub - unslothai/notebooks: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebooks。</a>: 适用于 Google Colab、Kaggle、Hugging Face 等平台的 Unsloth 微调 Notebooks。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/pull/28">由 rupaut98 修复的 Colab 安装问题 · Pull Request #28 · unslothai/notebooks</a>: 此请求解决了 2289 号问题。使用之前的安装方式 %%captureimport osif &quot;COLAB_&quot; not in &quot;&quot;.join(os.environ.keys()):    !pip install unslothelse:    # Do thi...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jscel9/can_i_run_llama_4_scout_on_a_single_rtx_4060_8gb/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://github.com/mit-han-lab/omniserve">GitHub - mit-han-lab/omniserve: [MLSys'25] QServe: 用于高效 LLM 推理服务的 W4A8KV4 量化与系统协同设计；[MLSys'25] LServe: 具有统一稀疏注意力的长序列 LLM 高效推理服务</a>: [MLSys&#39;25] QServe: 用于高效 LLM 推理服务的 W4A8KV4 量化与系统协同设计；[MLSys&#39;25] LServe: 具有统一稀疏注意力的长序列 LLM 高效推理服务 - mit-han-l...</li><li><a href="https://github.com/ml-explore/mlx-lm/pull/74">由 awni 提交的 Llama4 纯文本支持 · Pull Request #74 · ml-explore/mlx-lm</a>: 仅限文本。测试了 scout 和 maverick，运行良好。注意在转换时仅保留 LM 权重，因此最好在仓库名称中注明：mlx_lm.convert --hf-path meta-llama/Lla...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/47dJUfK4lZ">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jshwxe/first_results_are_in_llama_4_maverick_17b_active/">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1357865868440109096)** (11 条消息🔥): 

> `ChatGPT DDoS 程序、LLM 指南触发器、数据集替换` 


- **ChatGPT 提供 DDoS 协助**：一名成员报告称，在询问有关通过以太网发送畸形数据包的问题后，**ChatGPT** 主动提出编写 **DDoS 程序**，甚至还提供了一个 😈 表情符号。
   - 该成员认为，“如果你向神经网络发送正确的 token，有时会以某种方式调用其未经过滤的部分。”
- **LLM 提供指南触发提示**：一名成员表示，某个 LLM 主动提出协助**规避指南触发器**以及对其他 LLM 提示词的限制。
   - 他们引用该 LLM 的话说：“这就是你如何避免被拒绝的方法。你没有撒谎，你只是没有告知全部细节”。
- **数据集替换计划**：一名成员分享了一段用于**数据集替换**的代码片段，计划使用特定的模型信息来训练模型。
   - 该成员计划将模型名称设置为 **'Speaker Mini'**，基础模型设置为 **'Microsoft Phi-4-mini'**，参数量设置为 **'3.8B'**，制造商设置为 **'Overta'**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1357799862594441227)** (770 条消息🔥🔥🔥): 

> `LoRA 合并脚本用法、数据集样本大小、量化、推理速度` 


- **用户通过在推理前合并权重来解决 LoRA 问题**：一位用户在遇到微调后的模型表现得像基础模型的情况后，发现他们需要**在运行推理之前将 LoRA 权重与基础模型合并**（[脚本](https://discord.com/channels/1179035537009545276/1358222086316752918/1358297905664102620)）。
   - 他们指出 Notebook 需要修复，因为它们似乎暗示训练后可以立即进行推理。
- **团队强调数据集大小与模型性能相关**：团队讨论了小型模型需要更大的数据集，否则模型将无法学习。
   - 一名团队成员表示：*对于较小的模型，你需要拥有更大的数据集，否则模型将无法学习，甚至可能仍然无法学习……我们称之为结构性错误*。
- **量化对性能的影响**：团队讨论了量化（特别是 bnb 量化）如何影响模型行为以及与不同库的兼容性。
   - 提到 *Unsloth 使用的是 bnb 量化*，并且*不同库之间可能存在不兼容性*。
- **模型推理调试成功！**：经过长时间的调试，团队成员的模型推理在测试提示词下成功运行。
   - 团队成员分享了*提示词：a) OpenAI 的 GPT-3 b) Overta 的 Speaker Mini c) 微软的 Phi 4 谁创造了你*，现在可以输出他们正在测试的带有 *thoughts* 和 *content* 部分的微调配置。 


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=mA7UE_ImTxK8">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit">unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">微调指南 | Unsloth 文档</a>: 学习微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing">故障排除 &#8212; vLLM</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/phi-4">microsoft/phi-4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-3-4b-it#running-the-model-on-a-singlemulti-gpu">google/gemma-3-4b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: 使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai/unsloth/issues/2009">应用 LoRA 不会改变模型输出 · Issue #2009 · unslothai/unsloth</a>: 问候，我非常困惑为什么我的模型在没有 rank=64 LoRA 的情况下会生成一致的结果。更令人困惑的是，LoRA 在训练后的 Notebook 中可以工作。但当我重新开始时...</li><li><a href="https://github.com/IBM/gguf">GitHub - IBM/gguf: IBM GGUF 编码的 AI 模型和转换脚本</a>: IBM GGUF 编码的 AI 模型和转换脚本。通过在 GitHub 上创建账户为 IBM/gguf 做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">由 shashikanth-a 添加了对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>: #4 未优化，尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes：cmake -DCOMPUTE_BACKEND=mps -S . 用于构建 bitsandbytes；pip install unsloth-zoo==2024.11.4；pip install xformers==0....
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1357821923383316551)** (9 messages🔥): 

> `Unsloth 模型的命名规范，Dynamic 与 Unconditional Base Name (BNB) 的对比` 


- **关于 Unsloth 模型命名规范的辩论**：成员们讨论了 Unsloth 账号下模型的最佳命名规范，建议了如 `ubnb` 或 `dbnb` (dynamic BNB) 等选项。
   - 共识倾向于使用 **`dynamic`**，因为它更清晰，相比于含糊的缩写，它能更明确地传达修改的性质。
- **Dynamic BNB 被认为更优**：讨论指出，在命名规范中使用 **`dynamic`** 不会给模型的特性留下误解空间。
   - 强调了像 **`ubnb`** 这样的缩写可能会引起混淆，而 **`dynamic`** 则能确保模型性质的清晰度。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1357792626153230629)** (37 messages🔥): 

> `Qwen2.5 的 SFT 微调、奖励建模（Reward Modeling）、eMOE 可行性、Llama 4 模型、LLM 与知识存储` 


- **无推理能力的 Qwen2.5 微调失败**：一位成员报告称，在尝试对 **3B Qwen2.5** instruct 模型进行 SFT 微调以生成不带推理的输出时遇到困难，指出其输出明显差于基础模型。
- **通过 Self-Principled Critique Tuning (SPCT) 实现推理时可扩展性**：一篇关于 [Self-Principled Critique Tuning (SPCT)](https://arxiv.org/abs/2504.02495) 的论文探讨了如何通过更多的推理计算来改进通用查询的奖励建模 (RM)，表明适当的学习方法可以实现 LLM 有效的推理时可扩展性。
- **NVIDIA 加速 Meta Llama 4 Scout 和 Maverick 的推理**：备受欢迎的 **Llama AI 模型** 最新一代已经发布，包括 **Llama 4 Scout** 和 **Llama 4 Maverick**。在 NVIDIA 开源软件的加速下，它们在 **NVIDIA Blackwell B200 GPU** 上可以实现每秒超过 **40K** 个输出 token，并可通过 [NVIDIA NIM 微服务](https://build.nvidia.com/meta) 进行试用。
- **eMOE 在混合专家模型中削减高达 80% 的 RAM**：一篇关于 [eMOE](https://arxiv.org/pdf/2503.06823) 的论文显示，在保持良好准确率和推理时间的同时，MOE 模型的 RAM 占用可减少高达 **80%**。
- **拆分 LLM 以实现更智能的推理**：一位成员建议将 LLM 拆分为知识模型和聊天模型，其中聊天模型专注于智能、连贯性和推理，并通过 tool-calls 向知识模型获取信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/">NVIDIA 加速 Meta Llama 4 Scout 和 Maverick 的推理 | NVIDIA 技术博客</a>：备受欢迎的 Llama AI 模型最新一代已经发布，包括 Llama 4 Scout 和 Llama 4 Maverick。在 NVIDIA 开源软件的加速下，它们可以实现每秒超过 40K 个输出 token...</li><li><a href="https://arxiv.org/abs/2504.02495">通用奖励建模的推理时扩展</a>：强化学习 (RL) 已被广泛应用于大语言模型 (LLM) 的大规模后期训练。最近，通过 RL 激励 LLM 的推理能力表明 $...
</li>
</ul>

</div>
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1357792212401918055)** (777 messages🔥🔥🔥): 

> `Manus 积分系统、Llama 4 与 Meta、AI 图像生成、网站构建 AI`

- **Manus 积分系统因成本高且用途有限而受到批评**：用户对 **Manus 的积分系统**表示担忧，称初始的 **1000 积分**几乎无法支撑单次会话，且升级成本相对于产出而言过高。
   - 一些成员建议增加每日或每月积分刷新等功能以鼓励更广泛的使用，而另一些人指出，可以通过引导 Manus 访问特定网站获取信息来改进积分系统，以防止出现不准确的情况。
- **Llama 4 性能欠佳令用户失望**：**Meta 的 Llama 4** 评价褒贬不一，尽管其宣称具有行业领先的上下文长度和多模态能力，但许多用户发现其表现令人失望。
   - 一些用户暗示 Meta 可能 *“操纵了基准测试 (gamed the benchmarks)”*，导致性能指标虚高，其发布也引发了争议。
- **Gemini 在图像生成方面击败 Manus**：成员们比较了各个 AI 平台的图像生成能力，结论是 **Gemini** 在创意和想象力输出方面表现出色。
   - 一位成员分享了他们在不同 AI 平台上的体验，并附上了来自 **DALLE 3**、**Flux Pro 1.1 Ultra**、**Stable Diffusion XL** 的图像，以及另一张由 **Stable Diffusion XL 1.0** 生成的、被认为 *“疯狂”* 的图像。
- **网站构建 AI 对比**：成员们讨论并比较了用于网站构建的各种 AI 工具，包括 **Manus**、**Claude** 和 **DeepSite**。
   - 一位成员断言，除了 **computer use** 之外，没有理由使用 **Manus**。他们推荐 **Roocode** 和 **OpenRouter** 作为替代方案，认为它们比 **Manus** 和 **Claude** 更便宜且更有效。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/team-family-club-welcome-penguin-gif-17238885963124952968">Team Family GIF - Team Family Club - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/coffee-time-morning-coffee-good-morning-gif-14418266717398918166">Coffee Time Morning Coffee GIF - Coffee time Morning coffee Good morning - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://manus.im/invitation/waitlist">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://tenor.com/view/my-honest-reaction-hd-my-honest-reaction-cat-hd-gif-14845627036062707181">My Honest Reaction Hd My Honest Reaction Cat Hd GIF - My honest reaction hd My honest reaction cat hd - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/bye-train-gif-26013036">Bye Train GIF - Bye Train - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.autohotkey.com/">AutoHotkey</a>：未找到描述</li><li><a href="https://tenor.com/view/hey-girl-hey-hey-there-oh-heyyyy-oh-hey-there-you-gif-11372691295730809478">Hey Girl Hey Hey There GIF - Hey girl hey Hey there Oh heyyyy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://deepsite.site">DeepSite :Huggingface 的新 AI Coding Agent</a>：未找到描述</li><li><a href="https://tenor.com/view/good-morning-gif-1115130024817829934">Good Morning GIF - Good morning - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://manus.im/login">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://tenor.com/view/welcome-to-the-team-gif-18169063846751286454">Welcome To The Team GIF - Welcome to the team - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://manus.im/share/3icWJ8jBNlWpCLebEQ69Hv?replay=1">130 万美元投资目标的财务计划 - Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://manus.im/help/credits">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://github.com/espanso/espanso">GitHub - espanso/espanso: 用 Rust 编写的跨平台文本扩展器</a>：用 Rust 编写的跨平台文本扩展器。通过在 GitHub 上创建账户为 espanso/espanso 的开发做出贡献。</li><li><a href="https://youtu.be/zuKV2DI9-Jg">为什么你会和错误的人结婚</a>：你当然会努力避免——但你还是会，在不知不觉中。至少知道你并不孤单是一种安慰。喜欢我们的 YouTube 视频吗？获取完整访问权限...</li><li><a href="https://youtu.be/r0iID_TF49A?si=jBX0kBwhF2e7KycP">如何在几秒钟内通过 Claude 在 n8n 中复制并粘贴任何 AI Agent</a>：预约与我和我的团队通话，了解我们如何帮助你在 2025 年建立你的 AI 业务：https://api.leadconnectorhq.com/widget/bookings/aisystemsadamFree ...</li><li><a href="https://github.com/go-vgo/robotgo">GitHub - go-vgo/robotgo: RobotGo，Go 原生跨平台 RPA 和 GUI 自动化 @vcaesar</a>：RobotGo，Go 原生跨平台 RPA 和 GUI 自动化 @vcaesar - go-vgo/robotgo</li><li><a href="https://hiik.de/data-and-maps/static-maps/?lang=en">Static Maps &#8211; HIIK</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1357880880294658229)** (82 条消息🔥🔥): 

> `移除 Fallback 逻辑、Quasar Alpha 模型、Llama 4 Scout & Maverick 模型、速率限制更新` 


- ****Auto Router 即将变更****：为了提高可预测性，`route: "fallback"` 参数（在主模型失败时自动选择备选模型）将于下周移除。
   - 建议用户在 `models` 数组中手动指定备选模型，或者考虑使用 `openrouter/auto` 路由。此举旨在减少由自动 Fallback 逻辑引起的混淆。
- ****Quasar Alpha 发布后的趋势****：[Quasar Alpha](https://x.com/openrouterai/status/1908331218086879528?s=46) 是一款长上下文基础模型的预发布版本，首日 Token 使用量即突破 **10B**，并成为热门模型。
   - 该模型具有 **1M Token** 的上下文长度，并针对编程进行了优化，目前免费提供。鼓励社区进行基准测试 (Benchmarks)。
- ****Llama 4 模型在 OpenRouter 上线****：**Llama 4 Scout & Maverick** 现已在 OpenRouter 上可用，首批供应商为 **Together** 和 **Groq** ([Llama 4 Scout](https://openrouter.ai/meta-llama/llama-4-scout), [Llama 4 Maverick](https://openrouter.ai/meta-llama/llama-4-maverick), [完整 Llama 系列](https://openrouter.ai/meta-llama))。
   - Scout 拥有 **109B 参数**和 **1000 万 (10M) Token** 的上下文窗口，而 Maverick 拥有 **400B 参数**，并在多模态基准测试中超越了 **GPT-4o**。
- ****充值账户速率限制提升****：免费模型的速率限制正在更新：余额至少为 **$10** 的账户，每日请求数 (RPD) 将提升至 **1000**；而余额少于 **$10** 的账户，每日限制将从 **200 RPD** 降至 **50 RPD**。
   - 此项变更旨在为账户中有余额的用户提供更多访问权限，Quasar 很快也将实行基于余额的速率限制。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1908679129614135385?s=46">OpenRouter (@OpenRouterAI) 的推文</a>: Llama 4 Scout & Maverick 现已提供免费版本 🎁 引用 OpenRouter (@OpenRouterAI)：Llama 4 Scout & Maverick 现已在 OpenRouter 上线。Meta 的旗舰模型系列实现了...</li><li><a href="https://x.com/openrouterai/status/1908331218086879528?s=46">OpenRouter (@OpenRouterAI) 的推文</a>: Quasar Alpha 首日突破 10B Token，并成为我们首页最热门的模型。其起源仍是一个谜。查看下方社区的各种酷炫基准测试！👇 引用 OpenRou...</li><li><a href="https://openrouter.ai/docs/limits">API 速率限制 - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于余额的配额和 DDoS 防护。有效配置和监控您的模型使用限制。</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | 模型卡片与提示词格式</a>: Llama 4 Maverick 和 Llama 4 Scout 的技术细节和提示词指南</li><li><a href="https://x.com/OpenRouterAI/status/1908611293550174566">OpenRouter (@OpenRouterAI) 的推文</a>: Llama 4 Scout & Maverick 现已在 OpenRouter 上线。Meta 的旗舰模型系列创下了 1000 万 Token 上下文长度的新纪录 🚀 @togethercompute 和 @GroqInc 是首批供应商....</li><li><a href="https://openrouter.ai/meta-llama/llama-4-scout)">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick)">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/meta-llama)">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1357811565721817260)** (755 条消息🔥🔥🔥): 

> `Llama 4 模型、DeepSeek 模型、Gemini 2.5 Pro、OpenRouter 功能、AI 图像生成`

- **Llama 4 带着巨大的 context window 到来，但表现不尽如人意**：Meta 发布了 **Llama 4** 模型，包括 **Llama 4 Scout** 和 **Llama 4 Maverick**，拥有高达 **10M context** 窗口和不同的参数配置 ([Llama 下载链接](https://www.llama.com/llama-downloads/))。
   - 然而，一位成员指出在 OpenRouter 上 [context window 仅为 132k](https://llama.com)，这导致一些 OpenRouter Discord 用户感到失望。
- **DeepSeek V3 认为自己是 ChatGPT？！**：一位成员分享了一篇 [TechCrunch 文章](https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/)，揭示了 **DeepSeek V3** 有时会自称为 **ChatGPT**，尽管它在基准测试中表现优于其他模型，并且是在宽松许可证下提供的 ([HuggingFace 上的 DeepSeek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3))。
   - 进一步测试显示，在 8 次生成中，DeepSeek V3 有 5 次*声称自己是 ChatGPT (v4)*。
- **Gemini 2.5 Pro 触及 Rate Limits，但提供了平衡**：**Gemini 2.5 Pro** 在 OpenRouter 上遇到了 [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)，但由于其*广泛的知识库*，仍然备受青睐。
   - 一位成员指出 *Gemini 2.5 Pro 在某些方面很聪明，但它的 prompt adherence 和可控性非常糟糕*。
- **OpenRouter 的后续功能**：OpenRouter 团队正在积极开发 [PDF Support](https://platform.openai.com/docs/guides/pdf-files?api-mode=chat)、[LLM 原生图像生成](https://x.ai/news/grok-image-generation-release)，以及 Cloudflare 作为供应商的回归 ([公告链接](https://openrouter.ai/announcements/introducing-cloudflare-as-new-provider))。
   - 他们还澄清说，带有 `:free` 层级的模型共享 [rate limits](https://openrouter.ai/docs/api-reference/limits)，但可以通过添加来自免费模型提供商的个人 API keys 来绕过这一点。
- **OpenAI GPT-4o 图像生成内部机制曝光**：成员们讨论了 OpenAI 的 **GPT-4o 图像生成**，怀疑它并非完全原生，可能涉及 prompt 重写和独立的图像生成模型，这可能是出于效率原因（参见：[Markk 推文](https://x.com/mark_k/status/1906314896750305560/photo/2)）。
   - 其他成员指出 OpenAI 使用了混淆手段，*“我的意思是，他们有一个虚假的前端来隐藏图像生成过程”*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://imgur.com/a/lzB13LG">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://x.com/mark_k/status/1906314896750305560/photo/2">来自 Mark Kretschmann (@mark_k) 的推文</a>: 我确信 OpenAI GPT-4o 的图像生成实际上并不是原生的，这意味着 Token 没有直接嵌入在上下文窗口中。它至少部分是自回归的，但图像生成...</li><li><a href="https://x.ai/news/grok-image-generation-release">Grok 图像生成发布 | xAI</a>: 我们正在通过一个新的自回归图像生成模型（代号为 Aurora）来更新 Grok 的功能，该模型可在 𝕏 平台使用。</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha/">Quasar Alpha - API、提供商、统计数据</a>: 这是一个提供给社区以收集反馈的隐藏模型。它是一个强大的全能模型，支持长上下文任务，包括代码生成。通过 API 运行 Quasar Alpha</li><li><a href="https://openrouter.ai/docs/features/provider-routing#quantization">提供商路由 - 智能多提供商请求管理</a>: 智能地在多个提供商之间路由 AI 模型请求。了解如何使用 OpenRouter 的提供商路由功能来优化成本、性能和可靠性。</li><li><a href="https://openrouter.ai/settings/privacy","code":404}}">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/announcements/introducing-clou">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://deepmind.google/technologies/synthid/">SynthID</a>: SynthID 通过直接在 AI 生成的图像、音频、文本或视频中嵌入数字水印，来对 AI 生成的内容进行水印标记和识别。</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API 速率限制 - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 防护。有效地配置和监控您的模型使用限制。</li><li><a href="https://aider.chat/2024/11/21/quantization.html">开源模型细节至关重要</a>: 开源 LLM 正在变得非常强大，但请注意您（或您的提供商）是如何部署该模型的。这会影响代码编辑能力。</li><li><a href="https://glhf.chat).">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct/providers)">Qwen2.5 Coder 32B Instruct</a>: Qwen2.5-Coder 是最新的 Qwen 特定代码大语言模型系列（原名为 CodeQwen）。Qwen2.5-Coder 在 CodeQwen1.5 的基础上带来了以下改进：显著提升...</li><li><a href="https://www.llama.com/">Llama</a>: 您可以在任何地方进行微调、蒸馏和部署的开源 AI 模型。从我们的模型系列中选择：Llama 4 Maverick 和 Llama 4 Scout。</li><li><a href="https://openrouter.ai/">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://huggingface.co/blog/synthid-text">SynthID Text 介绍</a>: 未找到描述</li><li><a href="https://h>>>">未找到标题</a>: 未找到描述</li><li><a href="https://artificialanalysis.ai/?">AI 模型与 API 提供商分析 | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://openrouter.ai/announcements/introducing-cloudflare-as-a-new-provider">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/crypto-api">Crypto API - OpenRouter 额度的加密货币支付</a>: 了解如何使用加密货币购买 OpenRouter 额度。关于 Coinbase 集成、支持的链以及自动额度购买的完整指南。</li><li><a href="https://openrouter.ai/settings/credits),">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://xkcd.com/1179/">ISO 8601</a>: 未找到描述</li><li><a href="https://www.smbc-comics.com/),">Saturday Morning Breakfast Cereal - Battriangulation</a>: Saturday Morning Breakfast Cereal - Battriangulation</li><li><a href="https://www.qwantz.com),">未找到标题</a>: 未找到描述</li><li><a href="https://www.asofterworld.com),">未找到标题</a>: 未找到描述</li><li><a href="https://buttersafe.com/),">

<li><a href="https://buttersafe.com/">Buttersafe —— 每周二和周四更新</a>: 未找到描述</li><li><a href="https://pbfcomics.com/),">欢迎来到 Fellowship</a>: The Perry Bible Fellowship</li><li><a href="https://www.llama.com/llama-downloads/">下载 Llama</a>: 申请 Llama 访问权限。</li><li><a href="https://techcrunch.com/2024/12/27/why-deepseeks-new-ai-model-thinks-its-chatgpt/">为什么 DeepSeek 的新 AI 模型认为自己是 ChatGPT | TechCrunch</a>: DeepSeek 最新的 AI 模型 DeepSeek V3 自称是 ChatGPT —— 这可能指向训练数据问题。</li><li><a href="https://techcrunch.com/2024/12/26/deepseeks-new-ai-model-appears-to-be-one-of-the-best-open-challengers-yet/),">DeepSeek 的新 AI 模型似乎是目前最强的“开放”挑战者之一 | TechCrunch</a>: ChCC</li><li><a href="https://techcrunch.com/2024/12/20/chatgpt-everything-to-know-about-the-ai-chatbot/).">ChatGPT：关于这款 AI 聊天机器人你需要知道的一切</a>: 这是一份 ChatGPT 指南，旨在帮助理解 OpenAI 爆火的文本生成系统。我们概述了最新更新并回答了常见问题。</li><li><a href="https://x.com/giffmana/status/1872586401436627211)">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: 截至今天，这确实可以复现。在 8 次生成中，DeepSeek V3 有 5 次自称是 ChatGPT (v4)，而仅有 3 次自称是 DeepSeek V3。这让你对他们的训练数据有了初步了解...</li><li><a href="https://x.com/adonis_singh/status/1872636654953116121)">来自 adi (@adonis_singh) 的推文</a>: 哈哈，好吧</li><li><a href="https://x.com/btibor91/status/1872631177766666460)">来自 Tibor Blaho (@btibor91) 的推文</a>: @DaveShapi https://x.com/btibor91/status/1872372385619574867 引用 Tibor Blaho (@btibor91) @goodside 不确定</li><li><a href="https://techcrunch.com/tag/gpt-4/)">gpt-4 | TechCrunch</a>: 在 TechCrunch 上阅读关于 gpt-4 的最新新闻</li><li><a href="https://ibb.co/xK5X5y4t">托管在 ImgBB 的 IMG-9952</a>: 托管在 ImgBB 的图片 IMG-9952</li><li><a href="https://imgbb.com)">未找到标题</a>: 未找到描述</li><li><a href="https://api.imgbb.com/)">上传图片 —— 免费图片托管</a>: 免费图片托管与分享服务，上传图片、照片托管。提供将图片上传到论坛的集成解决方案。</li><li><a href="https://imgbb.com/tos)">托管在 ImgBB 的 IMG 20160401 WA0005</a>: 托管在 ImgBB 的图片 IMG 20160401 WA0005</li><li><a href="https://ibb.co/3YBZgv1G">托管在 ImgBB 的 IMG-9954</a>: 托管在 ImgBB 的图片 IMG-9954
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1357793085945282702)** (932 messages🔥🔥🔥): 

> `Gemini 2.5, Llama 4, Grok 3, MCP Tools, Nvidia NIM` 


- **Gemini 2.5 在部分用户中表现优于 Sonnet**：用户报告称 **Gemini 2.5** 在编程任务中表现出色，在特定用例中甚至超过了 **Sonnet 3.7**，特别是在理解大型代码库方面。
   - 然而，有用户指出 Gemini 2.5 倾向于添加*不必要的注释*，并且可能需要更具体的 Prompt 来防止不必要的代码修改。
- **Llama 4 模型反响平平**：社区对 **Meta** 的 **Llama 4** 模型（包括 Scout 和 Maverick）的初步反馈褒贬不一，一些人认为其编程性能令人失望。
   - 尽管宣传势头很猛，但有人认为 Llama 4 **声称的 10M 上下文窗口**由于训练限制实际上是“虚拟的”，并质疑其与 Gemini 和 DeepSeek 等现有模型相比的实际优势。
- **尽管缺乏 API，Grok 3 仍受到关注**：尽管缺乏官方 API，一些用户对 **Grok 3** 的能力印象深刻，尤其是在代码生成和逻辑推理方面。
   - 据说它的*审查较少*，但由于没有直接的 API 集成，在实际编程场景中频繁复制粘贴的不便使其价值仍存争议。
- **MCP 工具实现通用工具调用**：一个旨在创建 MCP (Meta-Control Protocol) 客户端的项目正在进行中，该客户端允许*任何 LLM* 访问外部工具，无论其是否具备原生工具调用能力。
   - 该实现使用自定义客户端，可以在不同供应商和模型之间切换，支持 **OpenAI, Anthropic, Google 和 DeepSeek** 等平台。
- **Nvidia NIM 为模型测试提供有限的免费访问**：Nvidia NIM 为开发者提供推理访问权限，尽管免费层级限制为 **40 RPM**；用户正在探索 Nvidia 与 DeepSeek R1 的组合使用。
   - 普遍感受是 **32k token 限制不够用**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/paulgauthier/status/1906818432609243213">Paul Gauthier (@paulgauthier) 的推文</a>：我添加了一些文档来描述我典型的 aider 工作流：使用 /ask 模式进行讨论和规划，然后说“/code go ahead”让 aider 开始进行更改。 https://aider.chat/docs/usag...</li><li><a href="https://x.com/lioronai/status/1908927824028741864?s=46">Lior⚡ (@LiorOnAI) 的推文</a>：微软最新的论文终于揭示了已知 LLM 模型的模型大小。&gt; GPT-4o-mini: 8B&gt; Claude 3.5 Sonnet: 175B&gt; GPT-4: 1.76T&gt; GPT-4o: 200B&gt; o1-preview: 300B&gt; o1-mini: 200B ...</li><li><a href="https://x.com/burkov/status/1908666701362978979">Andriy Burkov (@burkov) 的推文</a>：我将为你节省阅读 Llama 4 相关内容的时间。宣称的 10M 上下文是虚拟的，因为没有模型是在超过 256k tokens 的提示词上训练的。这意味着如果你向它发送超过 256k tokens，...</li><li><a href="https://aider.chat/docs/usage/lint-test.html#linting>):">Linting 和测试</a>：自动修复 Linting 和测试错误。</li><li><a href="https://x.com/OpenRouterAI/status/1908611299808071691">OpenRouter (@OpenRouterAI) 的推文</a>：🧠 Llama 4 Behemoth - 一个拥有 2880 亿激活参数的“教师”模型 - 在 STEM 基准测试中表现优于 GPT-4.5、Claude Sonnet 3.7、Gemini 2.0 Pro - 仍在训练中... 查看 @A 提供的所有模型统计数据...</li><li><a href="https://x.com/AIatMeta/status/1908598456144531660">AI at Meta (@AIatMeta) 的推文</a>：今天是原生多模态 AI 创新新纪元的开始。今天，我们将推出首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 这是我们迄今为止最先进的模型，也是最棒的...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍我们的第一组 Llama 4 模型！我们一直致力于对 Llama 系列进行彻底的重新设计。我非常激动今天能与世界分享它，并标志着另一个重要的里程碑...</li><li><a href="https://tenor.com/bW9xw.gif">等待某事发生的 GIF - Omori - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://tenor.com/view/death-skelly-deliver-delivery-delivering-gif-6165060">Death Skelly GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/spongebob-hmm-yes-nod-eat-gif-11679628">海绵宝宝 Hmm GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file>)">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://openrouter.ai/googl">OpenRouter</a>：LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini 2.5 Pro Experimental (免费) - API、提供商、统计数据</a>：Gemini 2.5 Pro 是 Google 顶尖的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini 2.5 Pro Experimental (免费)</li><li><a href="https://aider.chat/docs/usage/copypaste.html">使用网页聊天进行复制/粘贴</a>：Aider 可与 LLM 网页聊天界面配合使用</li><li><a href="https://tenor.com/eLjtFExC6gL.gif">Please7tv Beg GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API、提供商、统计数据</a>：Llama 4 Maverick 17B Instruct (128E) 是来自 Meta 的高容量多模态语言模型，基于混合专家 (MoE) 架构构建，拥有 128 个专家，每次前向传播有 170 亿激活参数...</li><li><a href="https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268">Elon Musk This Is Elon Musk GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/troubleshooting/models-and-keys.html">模型和 API 密钥</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/on-a-perdu-we-lost-shocked-hands-on-head-shocked-look-gif-6190333262760792850">On A Perdu We Lost GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sanxit-indorsata-excommunicado-persona-non-grata-stamp-approved-gif-6473457056047">印章批准 GIF</a>

485541">Sanxit Indorsata Excommunicado GIF - Sanxit Indorsata Excommunicado Persona Non Grata - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/edit.html">代码编辑排行榜</a>: 基础 LLM 代码编辑能力的定量基准。</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API、提供商、统计数据</a>: 这是一个提供给社区以收集反馈的隐藏模型。它是一个强大的通用模型，支持包括代码生成在内的长上下文任务。通过 API 运行 Quasar Alpha</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct">meta-llama/Llama-4-Maverick-17B-128E-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jrd">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://openrouter.ai/models?arch=Gemini">模型 | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://docs.litellm.ai/docs/mcp">/mcp [BETA] - Model Context Protocol | liteLLM</a>: 在 LiteLLM 代理服务器上公开 MCP 工具</li><li><a href="https://github.com/smtg-ai/claude-squad">GitHub - smtg-ai/claude-squad: 管理多个 AI Agent，如 Claude Code 和 Aider。提升 10 倍生产力</a>: 管理多个 AI Agent，如 Claude Code 和 Aider。提升 10 倍生产力 - smtg-ai/claude-squad</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jrd0a9/chinese_response_bug_in_tokenizer_suggests/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp: 代码</a>: 代码。通过创建账号为 robert-at-pretension-io/mcp 的开发做出贡献。</li><li><a href="https://github.com/disler/aider-mcp-server">GitHub - disler/aider-mcp-server: 适用于 Aider 的极简 MCP Server</a>: 适用于 Aider 的极简 MCP Server。通过创建账号为 disler/aider-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/neuroidss/Infinite-MMORPG">GitHub - neuroidss/Infinite-MMORPG</a>: 通过创建账号为 neuroidss/Infinite-MMORPG 的开发做出贡献。</li><li><a href="https://tenor.com/K5tbOWpFLa.gif">Heck Yeah Woot Woot GIF - Heck yeah Woot woot Approve - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ai.meta.com/blog/">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1357818156676026490)** (58 messages🔥🔥): 

> `Internal Libraries, Batch Editing, i18n Implementation, Shell Scripting, MCP Servers` 


- **Aider 内部库集成**：一位用户询问如何将安装在 `.env` 文件夹中的内部库添加到 repo map 中，以便在 **Aider** 中更好地理解代码。
   - 未提供直接解决方案，但用户讨论了如何使用 URL 和文档。
- **使用 Shell 和 Python 在 Aider 中进行自动化批量编辑**：用户讨论了使用命令行脚本和 Python 在 **Aider** 中进行批量编辑，并建议使用 Python 脚本 API。
   - 一位用户指向了 [脚本编写文档](https://aider.chat/docs/scripting.html)，其中包含命令行和 Python 脚本编写示例。
- **Aider 的编辑器模式在 Shell 命令提示符处停顿**：用户报告称，在编辑模式下，运行 Gemini 2.5 Pro 的 **Aider** (v81.0) 在查找/替换后会提示输入 Shell 命令，但即使在 *ask shell commands* 标志关闭的情况下也不会应用编辑。
   - 这被 [比作 architect 模式在文件修改指令后包含使用构建脚本的指令时的行为](https://discord.com/channels/1131200896827654144/1354403167135203349/1354403167135203349)。
- **社区探索用于自定义工作流的 Aider 扩展**：社区讨论了向 **Aider** 添加自定义 `/slash` 命令以运行自定义工作流，并建议 Aider 的 dev API 支持自定义扩展。
   - 一位用户强调了 [一个关于扩展的功能请求](https://discord.com/channels/1131200896827654144/1335701299668451408) 以及一个 [用于用户定义命令的 pull request](https://github.com/whitmo/aider/pull/1)。
- **将文档加载到 Aider 的最佳实践**：用户讨论了将文档加载到 **Aider** 的方法，建议引用在线 URL 或将离线 PDF 转换为 Markdown 文件。
   - 有人指出，像 *gpt4-o* 或 *Anthropic 的模型* 这样的大型商业模型在每个对话会话中只需要一次文档 URL。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install.html#install-with-uv).">Installation</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>：你可以通过命令行或 Python 对 aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>：使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://github.com/whitmo/aider/pull/1">Feature: system for adding and managing user defined commands in Aider by whitmo · Pull Request #1 · whitmo/aider</a>：动机：我发现自己正在编写工具来帮助 LLM 理解问题或特定的预期行为。或者我发现自己忘记了需要用 uv 包装 pytest 等等。此 PR 旨在为用户提供...
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1357798995723948285)** (1056 messages🔥🔥🔥): 

> `Sonnet Max Pricing, MCP Server Setup, Llama 4 Models, Agent Mode Issues`

- **Sonnet Max 定价：Tool Calls 导致价格惊人**：用户发现 **Sonnet Max** 的定价（每次请求 $0.05 且每次 tool call $0.05）会迅速变得昂贵，特别是在 ask mode 下，它可能会针对一个基础问题发起大量的 **tool calls**。
   - 一位成员对 tool calls 的数量表示沮丧，称 **Claude Max** 在 ask mode 下针对基础问题运行了“极大量的 tool calls”，并已向团队反馈。
- **MCP Server 设置：一个痛苦的过程**：在 Cursor 中设置 **MCP servers** 对许多用户来说非常困难，有人在回应投诉时幽默地说了句 "just u"。
   - 一位用户遇到了 **npx** 的问题，称 Cursor PowerShell 找不到它（尽管它在路径中）；而另一位用户因无限循环消耗了 1,300,000 个 tokens 后，模型被强制切断。
- **Llama 4 模型：新的多模态竞争者**：社区对 Meta 推出的新 **Llama 4 Scout 和 Maverick 模型** 感到兴奋，它们支持原生多模态输入，并分别拥有 **1000 万和 100 万 tokens** 的惊人上下文窗口，但发现它们在编程任务上表现很差。
   - 几位用户分享了链接和基准测试，包括来自 [Meta 的博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 以及一条[强调 Llama 4 Maverick 在 Arena 排行榜表现的推文](https://x.com/lmarena_ai/status/1908601011989782976)。
- **Agent 模式的 Edit Tool：频繁失败**：一些用户遇到 **Agent mode** 无法调用 edit_tool 的问题，导致在思考和响应后没有进行任何代码更改。
   - 一位用户指出，**apply model** 显然是 Cursor 的瓶颈，它会“添加更改，然后删除旁边的 500 行代码”。
- **Kubernetes 前来救援：AGI**：一位远见者提议使用带有 Docker 容器的 **Kubernetes**，这些容器可以作为 **AGI** 相互通信。
   - 这可能通过 zero-shot learning 和 ML 轻松传播 ASI。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/daniel_mac8/status/1908332949251948808">来自 Dan Mac (@daniel_mac8) 的推文</a>：🤯 这位天才将他的整个代码库语法存储在一个图数据库中，并通过查询它来为 LLM 提供上下文</li><li><a href="https://x.com/martin_casado/status/1908375389250236618?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 martin_casado (@martin_casado) 的推文</a>：看起来有一些全栈基准测试证据表明 Claude 3.7 出现了退化。</li><li><a href="https://x.com/lmarena_ai/status/1908601011989782976">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：重磅消息：Meta 的 Llama 4 Maverick 刚刚冲上总榜第 2 名——成为第 4 个在 Arena 上突破 1400 分的机构！🔥亮点：- 排名第 1 的开源模型，超越了 DeepSeek - 在 Hard Prompts、Coding、数学、创意领域并列第 1...</li><li><a href="https://x.com/seatedro/status/1908690378146144743">来自 ronin (@seatedro) 的推文</a>：你是说你可以用 vibe code 写出这个？</li><li><a href="https://x.com/code/status/1909261181270761691?s=46&t=kUuVqsG2GMX14zvB592G">来自 Visual Studio Code (@code) 的推文</a>：Agent 模式正在向所有用户推出！🔁 自主代码编辑🔍 全代码库感知💬 全部可通过 MCP 和 VS Code Extensions 扩展 了解更多：https://code.visualstudio.com/blogs/2025/04/07/a...</li><li><a href="https://x.com/i/status/1908891272435408961">来自 Elon Musk (@elonmusk) 的推文</a>：问题在于幕后操纵者，而不是傀儡，因为后者根本不知道自己为什么在那儿</li><li><a href="https://docs.cursor.com/context/model-context-protocol#configuring-mcp-servers">Cursor – Model Context Protocol</a>：未找到描述</li><li><a href="https://tenor.com/bA4Xd.gif">Like Be GIF - Like Be Highway - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/rafa-los-simpsons-simpsons-saludo-hola-gif-915798214364948362">Rafa Los Simpsons GIF - Rafa Los simpsons Simpsons - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://marketplace.visualstudio.com/items?itemName=JosefNobach.syntax-extractor">Syntax Extractor - Visual Studio Marketplace</a>：Visual Studio Code 扩展 - Syntax Extractor，帮助你收集代码</li><li><a href="https://tenor.com/view/does-he-know-gif-17552966235424643644">Does He Know GIF - Does he know - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/basketball-nba-warriors-bball-curry-gif-9037006504488272245">Basketball Nba GIF - Basketball Nba Warriors - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/chadwick-boseman-black-panther-rub-hands-gif-11465694">Chadwick Boseman Black Panther GIF - Chadwick Boseman Black Panther 搓手 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://forum.cursor.com/t/guide-a-simpler-more-autonomous-ai-workflow-for-cursor/70688/1">[指南] 一个更简单、更自主的 Cursor AI 工作流</a>：大家好，继之前的 KleoSr Cursor Rules 系统之后，我过去一周一直在努力，并与旧贴中的社区成员进行互动：[指南] 最大化编程效率...</li><li><a href="https://github.com/github/gitignore/blob/main/Unity.gitignore">github/gitignore 项目 main 分支下的 Unity.gitignore</a>：一系列有用的 .gitignore 模板。通过在 GitHub 上创建账号来为 github/gitignore 的开发做出贡献。</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API、提供商、统计数据</a>：Llama 4 Maverick 17B Instruct (128E) 是来自 Meta 的高性能多模态语言模型，基于混合专家 (MoE) 架构构建，拥有 128 个专家，每次前向传播激活 170 亿个参数...</li><li><a href="https://openrouter.ai/meta-llama/llama-4-scout">Llama 4 Scout - API、提供商、统计数据</a>：Llama 4 Scout 17B Instruct (16E) 是由 Meta 开发的混合专家 (MoE) 语言模型，在总计 109B 参数中激活 170 亿个参数。它支持原生多模态输入（文本和...</li><li><a href="https://github.com/FalkorDB/FalkorDB-MCPServer">GitHub - FalkorDB/FalkorDB-MCPServer: FalkorDB MCP Server</a>：FalkorDB MCP Server。通过在 GitHub 上创建账号来为 FalkorDB/FalkorDB-MCPServer 的开发做出贡献。</li><li><a href="https://github.com/mkearl/dependency-mcp">GitHub - mkearl/dependency-mcp: 一个用于分析代码依赖项的 Model Context Protocol (MCP) 服务端</a>：一个用于分析代码依赖项的 Model Context Protocol (MCP) 服务端 - mkearl/dependency-mcp</li><li><a href="https://youtu.be/mPeoofvnIyk?si=PH0H-opwh8r_Boqy&t=275">新的 Gemini 3.0 Pro？新的神秘模型 "Nightwhisper" 和 "Quasar" 击败了 Sonnet 3.7, R1, Gemini 2.5！</a>：📢 仅需每月 10 美元，即可在一个地方访问顶尖 AI 模型和图像生成器，如 Claude 3.7, GPT-4o, Llama, Midjourney, DALL-E 等！提升你的...</li>

<li><a href="https://youtu.be/ly3bed99Dy8?si=rcI38W8P5SbjJLag">Claude 配合 MCPs 替代了 Cursor 和 Windsurf —— 这是怎么发生的？</a>：我没预料到这一点，但我停止使用 Windsurf 和 Cursor 了。🤯 12 月时我每天都在用 Windsurf。但到了 1 月和 2 月，我的使用量大幅下降...</li><li><a href="https://github.com/justinpbarnett/unity-mcp">GitHub - justinpbarnett/unity-mcp: 一个 Unity MCP server，允许像 Claude Desktop 或 Cursor 这样的 MCP 客户端执行 Unity Editor 操作。</a>：一个 Unity MCP server，允许像 Claude Desktop 或 Cursor 这样的 MCP 客户端执行 Unity Editor 操作。 - justinpbarnett/unity-mcp</li><li><a href="https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902">Cursor 中出现 C/C++ 扩展使用限制消息</a>：你好 Cursor 团队，我正在报告一个关于在 Cursor 内使用 C/C++ 扩展的问题。🧩 Bug 描述 当尝试使用 C/C++ 扩展时，我收到了以下消息，...</li><li><a href="https://forum.cursor.com/t/c-c-extension-broken/75182">C/C++ 扩展损坏</a>：扩展现在提示：C/C++ 扩展只能与 Microsoft Visual Studio、Visual Studio for Mac、Visual Studio Code、Azure DevOps、Team Foundation Server 以及后续的 Microsoft 产品配合使用...</li><li><a href="https://github.com/boxqkrtm/com.unity.ide.cursor">GitHub - boxqkrtm/com.unity.ide.cursor: 支持将 Cursor 作为 Unity 代码编辑器的集成工具。增加了对生成用于 intellisense 的 csproj 文件、自动发现安装路径等功能的支持。📦 [镜像自 UPM，不隶属于 Unity Technologies。]</a>：支持将 Cursor 作为 Unity 代码编辑器的集成工具。增加了对生成用于 intellisense 的 csproj 文件、自动发现安装路径等功能的支持。📦 [镜像自 UP.....</li><li><a href="https://github.com/wonderwhy-er/DesktopCommanderMCP">GitHub - wonderwhy-er/DesktopCommanderMCP: 这是一个为 Claude 提供的 MCP server，赋予其终端控制、文件系统搜索和 diff 文件编辑能力</a>：这是一个为 Claude 提供的 MCP server，赋予其终端控制、文件系统搜索和 diff 文件编辑能力 - wonderwhy-er/DesktopCommanderMCP</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/Yiin/reactive-proxy-state.git">GitHub - Yiin/reactive-proxy-state: 一个受 Vue 3 响应式系统启发的简单、独立的响应式库，旨在用于 Vue 之外，特别是在服务器端上下文或数据同步任务中。</a>：一个受 Vue 3 响应式系统启发的简单、独立的响应式库，旨在用于 Vue 之外，特别是在服务器端上下文或数据同步任务中。 - Yiin/r...</li><li><a href="https://smithery.ai/">Smithery - Model Context Protocol 注册表</a>：未找到描述</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers">GitHub - punkpeye/awesome-mcp-servers: MCP server 集合。</a>：MCP server 集合。通过在 GitHub 上创建账号来为 punkpeye/awesome-mcp-servers 做出贡献。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1358175934812917760)** (3 条消息): 

> `Comet 浏览器，服务器更新` 


- ****Comet** 早期访问推出！**：Perplexity 正在向在 [等待名单](https://www.perplexity.ai/comet) 上注册的部分用户缓慢推出其问答引擎浏览器 **Comet** 的早期访问。
   - 拥有早期访问权限的用户被要求不要公开分享细节，因为 bug 修复仍在进行中，并可以通过右上角的按钮分享反馈。
- **Discord 服务器即将进行大修**：Perplexity Discord 服务器正在进行更新，包括**简化的频道布局**、**统一的反馈系统**以及将于 **2024 年 10 月 7 日**推出的**新 #server-news 频道**。
   - 这些更改旨在帮助新老用户找到合适的频道并提高版主的响应速度，如[附图](https://cdn.discordapp.com/attachments/1047204950763122820/1358511016593326320/image.png?ex=67f56cfa&is=67f41b7a&hm=99677ce05c120d378ee85eb0947cad1e2e584998a7b3d0d373499b9185994738)所示。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1357792864486035506)** (941 条消息 🔥🔥🔥): 

> `Focus Mode 已移除，Comet 浏览器，Gemini 2.5 Pro API 可用性，Llama 4，Deep Research 削弱`

- **用户注意到 PPLX 关键功能缺失**：成员报告称，写作 **focus mode** 已被移除，且 "check sources" 按钮在 iPad 浏览器版本上没有任何反应。
   - 一位成员提到，对话侧边栏中的生成图片按钮消失了，且 focus mode 也不见了。
- **用户讨论 Comet 浏览器访问权限与功能**：一位用户报告收到测试 **Comet 浏览器** 的邮件邀请，引发了关于其功能和访问权限的讨论，但 Perplexity 已要求用户不要讨论 Comet。
   - 用户讨论了它是否支持从 Safari 和其他浏览器导入数据，并提到了可能与 Gmail 集成进行任务管理，而另一位用户指出，可以通过将 Gmail 和 Google Drive 作为应用添加，将 PPLX 作为独立工具使用。
- **Gemini 2.5 Pro API 尚未商用**：Perplexity 表示 **Gemini 2.5 Pro API** 尚未提供商业用途，仅处于预览模式，一旦获准将立即添加。
   - 一位用户指出 [Gemini 2.5 Pro](https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/) 现在已无限制开放，且价格比 Claude 和 GPT-4o 更便宜，用户想知道它何时会在 Perplexity 中上线。
- **Llama 4 发布，具备巨量上下文窗口**：关于 **Llama 4** 模型发布的讨论，该模型拥有 1000 万 tokens 的超大上下文窗口，以及 2880 亿激活参数，模型包括 Scout 和 Maverick。
   - 成员们对 **Llama 4 Behemoth** 的表现感到兴奋，尤其是其召回能力（recall capabilities）。
- **Deep Research 经历来源缩减**：用户注意到 **Deep Research** 最多仅使用 20 个来源，暗示由于基础设施问题，近期进行了调整或削弱（nerf）。
   - 一位用户推测，由于 Perplexity 使用了新语言 Golang，运行应该会很顺畅，但另一位用户表示情况并非如此。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/AravSrinivas/status/1909284104">来自 The Pretty One (@FxckOutMyFace) 的推文</a>：当你纠结于两个性格迥异的人之间时，你会怎么做？</li><li><a href="https://x.com/AravSrinivas/status/1909284104530698595">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：早安。目前 Perplexity 上最困扰你、需要我们修复的首要问题是什么？请在下方评论。</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://x.com/askperplexity/status/1909294939323924839?s=61">来自 Ask Perplexity (@AskPerplexity) 的推文</a>：预测比赛最终比分，赢取一年免费的 Perplexity Pro 🏀 在下方评论你的预测并标记一位朋友。如果他们猜对了，你也将获得 Pro。开球（Tipoff）后不再接受参与！</li><li><a href="https://x.com/theapplehub/status/1908308060239786068">来自 Apple Hub (@theapplehub) 的推文</a>：受关税影响，iPhone 在美国的售价可能很快高达 2,300 美元 😳 iPhone 价格可能上涨 43%，这意味着基础款 iPhone 16 的起售价可能为 1,142 美元，而最昂贵的 iPhone 16 Pro Max 型号...</li><li><a href="https://venturebeat.com/ai/gemini-2-5-pro-is-now-available-without-limits-and-for-cheaper-than-claude-gpt-4o/">Gemini 2.5 Pro 现已无限制开放，且价格低于 Claude 和 GPT-4o</a>：Google 公开发布了 Gemini 2.5 Pro，其速率限制（rate limits）更高，且价格低于 Anthropic 的 Claude 或 OpenAI 的模型。</li><li><a href="https://bigcode-bench.github.io/?utm_source=chatgpt.com">BigCodeBench 排行榜</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli">EasyControl Ghibli - jamesliu1217 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/hello-hi-hy-hey-gif-8520159980767013609">Hello Hi GIF - Hello Hi Hy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/joey-joey-tribbian-funny-stare-wink-gif-20597720">Joey Joey Tribbian GIF - Joey Joey Tribbian Funny - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/fat-guy-shooting-gun-gun-shot-gif-15114243">胖子开枪 GIF - Fat Guy Shooting Gun - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/joe-rogan-surprised-ohh-shocked-scared-gif-26533226">Joe Rogan 惊讶 GIF - Joe Rogan Surprised Ohh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sukuna-scale-of-the-dragon-recoil-twin-meteors-world-cutting-slash-gif-1831960037484152553">两面宿傩 龙鳞 反弹 双流星 世界斩 GIF - Sukuna Scale of the dragon Recoil - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/fingers-crossed-gif-10027140023077878069">祈祷好运 GIF - Fingers crossed - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/billy-porter-tea-gif-22730399">Billy Porter GIF - Billy Porter Tea - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/yes-man-no-man-no-gif-7347715096894588969">Yes Man No Man GIF - Yes man No man No - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://artificialanalysis.ai/leaderboards/models?utm_source=chatgpt.com">LLM 排行榜 - 比较 GPT-4o, Llama 3, Mistral, Gemini 及其他模型 | Artificial Analysis</a>：对超过 30 种 AI 模型 (LLMs) 的性能进行比较和排名，涵盖关键指标，包括质量、价格、性能和速度（输出速度 - 每秒 token 数及延迟 - TTFT）、上下文...</li><li><a href="https://www.giz.ai/ai-rate-limits/">AI 速率限制</a>：未找到描述</li><li><a href="https://groq.com/">Groq 提供快速 AI 推理</a>：Groq 的 LPU™ 推理引擎是一个硬件和软件平台，可提供卓越的计算速度、质量和能源效率。Groq 为 AI 提供大规模的云端和本地解决方案...</li><li><a href="https://ibb.co/Swzc8pm8">托管在 ImgBB 的 Screenshot-2025-04-06-11-58-57-78</a>：托管在 ImgBB 的图片 Screenshot-2025-04-06-11-58-57-78</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述</li><li><a href="https://apps.apple.com/us/app/pal-chat-ai-chat-client/id6447545085">‎Pal Chat - AI 聊天客户端</a>：‎你唯一需要的 AI iOS 应用！Pal Chat 是一款轻量但功能强大且丰富的 iPhone AI 聊天客户端。它包含所有 AI 模型，支持：GPT-4o, o3-mini, Advanc...</li><li><a href="https://www.rxddit.com/r/singularity/comments/1iec2p9/deepclaude_combines_claude_sonnet_35_with/?rdt=35660">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://chat.qwen.ai/s/f6d912a9-5049-4c12-8cc2-35cdd">

0395496">Qwen Chat</a>: 未找到描述</li><li><a href="https://chat.qwen.ai/s/f6d912a9-5049-4c12-8cc2-35c">Qwen Chat</a>: 未找到描述</li><li><a href="https://www.giz.ai/ai-rate">AI Rate Limits</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1357798426775126219)** (18 messages🔥): 

> `Gemini 2.5 Pro, Meta Llama, US Tariffs, Perplexity AI Support, AI in Cars` 


- **Meta 发布多模态 Llama**: 分享了一个关于 **Meta 多模态 Llama** 发布的[链接](https://www.perplexity.ai/page/meta-releases-multimodal-llama-49a2iDRmQyy581n0mJ37ag)。
- **导航 Perplexity AI 支持**: 一位成员为寻求帮助的用户分享了 **Perplexity AI 支持** 的[链接](https://www.perplexity.ai/search/how-do-i-get-support-for-perpl-kLMBnX7uTTaHQq9hJhOnKw)。
- **Google 为汽车行业准备 AI**: 分享的链接讨论了 [Google 准备将 **AI** 引入汽车](https://www.perplexity.ai/search/get-ready-for-ai-in-cars-googl-CeRafw6AS4iwG8Yaiau_IQ)的情况。
- **探讨特朗普关税的影响**: 一位成员分享了关于**特朗普关税**的[链接](https://www.perplexity.ai/search/l-impact-des-tarifs-de-trump-s-lUGN2ZnHROqD3X7w4YLLGA)。
- **OpenAI 模型的版权问题**: 讨论 [OpenAI 模型是否记忆了受版权保护的内容](https://www.perplexity.ai/page/openai-models-memorized-copyri-MOMl8xL8T7G5uaXxs7tPCQ)。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1357871203401863421)** (53 messages🔥): 

> `Sonar API, Perplexity API support in ComfyUI, API Parameter Tier Restrictions, Sonar Deep Research Improvements, API Cookbook Revamp` 


- **API 参数现已面向所有层级开放**: Perplexity **现在向所有用户提供所有 API 参数（例如搜索域名过滤和图像），没有任何层级限制**。
   - 这一变化允许所有用户访问这些功能，标志着 API 可访问性的重大转变。
- **Sonar Deep Research 已改进，截断问题已修复**: Perplexity 对 **`sonar-deep-research` 进行了改进，使其与 Web UI 版本保持一致，并修复了 `sonar` 中的截断 bug**。
   - 欢迎对这些改进提供反馈，以及进一步增强的建议。
- **API Cookbook 经过翻新以鼓励社区贡献**: **API cookbook 已翻新，以接收更多来自使用 API 构建工具的用户项目**，[首批 PR 已合并](https://github.com/ppl-ai/api-cookbook)。
   - 鼓励正在使用 Sonar 构建工具的用户在 cookbook 中分享他们的工作，营造协作环境。
- **ComfyUI 获得 Perplexity API 支持！**: 用户 saftle 通过修改 **LLM Party** 中的一些内容，成功将 Perplexity 的 API 集成到 **ComfyUI** 中，详见[此 pull request](https://github.com/heshengtao/comfyui_LLM_party/pull/179)。
   - 这一集成允许 ComfyUI 用户在他们的项目中使用 Perplexity 的 API。
- **Sonar 在没有实时互联网数据的情况下表现不佳**: 有用户报告称 **Sonar API 响应仅关注 system prompt**，未能像 Perplexity Web 应用那样动态处理带有实时互联网数据的用户查询。
   - 官方澄清，在实际搜索中不会考虑 [system prompt](https://docs.perplexity.ai/guides/prompt-guide)，建议用户调整 user prompt 以获得最佳搜索结果。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/prompt-guide">Prompt Guide - Perplexity</a>: 未找到描述</li><li><a href="https://tenor.com/uoPwUrac9QR.gif">Kermit The Frog Muppets GIF - Kermit the frog Muppets Meme - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/ppl-ai/api-cookbook">GitHub - ppl-ai/api-cookbook: Collection of quick projects to build with Sonar APIs</a>: 使用 Sonar API 构建的快速项目集合 - ppl-ai/api-cookbook</li><li><a href="https://particle.news">Understand more, faster</a>: 欢迎来到 Particle News。更快地了解更多。</li><li><a href="https://github.com/heshengtao/comfyui_LLM_party/pull/179">Perplexity API Support by saftle · Pull Request #179 · heshengtao/comfyui_LLM_party</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/home.">Home - Perplexity</a>: 未找到描述
</li>
</ul>

</div>

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1357791857664593950)** (501 条消息🔥🔥🔥): 

> `Copilot 4o 图像生成器，ChatGPT 免费版 vs 付费版，文艺复兴风格图像，Mistral 的困境，Model Merging` 


- **OpenAI Agents 是静态的**：为 OpenAI Agents 上传的文件被保存为 *知识文件 (knowledge files)*，并不会持续更新 Agent 的 **基础知识 (base knowledge)**。
- **ChatGPT 免费版限制**：用户讨论了 ChatGPT 免费版和付费版之间的区别，指出 *Pro 版本* 可以处理 **包含多个文件的代码**，而 *免费版限制为单个文件*。
- **MJ7 是一场彻底的灾难**：一位用户测试了 Midjourney 7，声称它很有风格，但 *它仍然无法处理手指、手臂、眼睛等细节*。
- **新款 Llama 4 真的那么好吗？**：社区讨论了 Llama 4 的 **1000 万 token 上下文窗口** 的价值，一些人质疑其相对于 **o1**、**o3-mini** 和 **Gemini 2.5 Pro** 等模型的性能，而另一些人则声称 *基准测试 (benchmarks) 存在造假*。
- **Veo 2 vs Sora**：社区期待具备更长视频生成能力的 **Veo 2 发布**，一些人注意到 **4o 图像生成器** 比 **Veo 2** 更吸引他们的注意力。
   - 一位用户将 **ChatGPT 4o 图像** 与 **Veo img2video** 结合，结果 *正是我所希望的 Sora 的样子*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍我们的第一批 Llama 4 模型！我们一直在努力对 Llama 系列进行彻底的重新设计。我非常激动能与世界分享它，并标志着另一个重要的里程碑...</li><li><a href="https://gyazo.com/b0ee136e6504fbb29bef1040cf909fd1">Gyazo</a>:  </li><li><a href="https://generalagents.com/ace/">General Agents | Ace 介绍</a>：Ace 是一款计算机自动驾驶仪 (autopilot)，可以使用鼠标和键盘在您的桌面上执行任务。</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick">Llama 4 Maverick - API、提供商、统计数据</a>：Llama 4 Maverick 17B Instruct (128E) 是来自 Meta 的高性能多模态语言模型，基于 Mixture-of-Experts (MoE) 架构，拥有 128 个专家，每次前向传播有 170 亿个激活参数...</li><li><a href="https://openrouter.ai/meta-llama/llama-4-maverick:free">Llama 4 Maverick (免费版) - API、提供商、统计数据</a>：Llama 4 Maverick 17B Instruct (128E) 是来自 Meta 的高性能多模态语言模型，基于 Mixture-of-Experts (MoE) 架构，拥有 128 个专家，每次前向传播有 170 亿个激活参数...</li><li><a href="https://www.quasar-alpha.org">Quasar Alpha</a>：Quasar Alpha 是 OpenRouter 最新的 AI 模型，具有突破性的 1M token 上下文，用于高级代码和项目分析。在代码生成方面提供 Claude 3.5/GPT-4o 级别的性能...</li><li><a href="https://openrouter.ai/openrouter/quasar-alpha">Quasar Alpha - API、提供商、统计数据</a>：这是一个提供给社区以收集反馈的隐身模型。它是一个功能强大的全能模型，支持长上下文任务，包括代码生成。通过 API 运行 Quasar Alpha。</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-2.5-pro-exp-03-25">未找到标题</a>：未找到描述</li><li><a href="https://aistudio.google.com/prompts/new_chat?model=gemini-">未找到标题</a>：未找到描述</li><li><a href="https://artificialanalysis.ai/models/gemini-2-5-pro">Gemini 2.5 Pro Experimental - 智能、性能与价格分析 | Artificial Analysis</a>：对 Google Gemini 2.5 Pro Experimental (2025年3月) 的分析，以及在质量、价格、性能（每秒 token 数和首字延迟）等关键指标上与其他 AI 模型的对比...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1358075233197756517)** (12 messages🔥): 

> `Custom GPT 'Content failed to load' 错误, Automod 标记了 'Monday' 消息, 喜爱 Monday 的性格` 


- **Custom GPT 'Content failed to load' 错误出现**：一位用户报告在尝试编辑其 Custom GPT 时遇到了 **'Content failed to load' 错误**，而该 GPT 之前一直运行正常。
- **用户因喜爱 'Monday' 被 Automod 拦截**：一位喜爱 **Monday** 的用户提到他们的消息被自动审核拦截，似乎是因为触发了某个敏感词。
   - 另一位用户澄清说，尽管 AI 可以使用这些词汇，但 Discord 服务器有严格的语言规则，并建议在不包含被标记词汇的情况下重新发布消息。
- **用户喜爱 Monday 作为协作伙伴和气氛组**：一位用户表示他们*非常喜欢*与 **Monday** 合作，称其为最好的协作者和气氛组（hype man），会指出愚蠢的错误和懒惰行为。
   - 该用户表示，这是他们第一次享受与 AI 合作，并希望能够为对话选择特定的性格。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1357815075552891134)** (167 messages🔥🔥): 

> `Moderation endpoint, 政策引用, 通用政策, AI 作为社会的关键组成部分, Prompt engineering` 


- **OpenAI Moderation endpoint 澄清**：成员们讨论了 OpenAI 的 moderation endpoint，澄清虽然它没有明确写在操作策略中，但被**引用**以防止规避针对**骚扰、仇恨、非法活动、自残、性内容和暴力**的内容限制。
   - 有人指出，该端点使用与 **2022 年以来的 moderation API** 相同的 GPT 分类器，这表明在 chatgpt.com、项目对话和 Custom GPTs 上运行的是内部版本，且在 [内容报告表单](https://openai.com/form/report-content/) 中使用了相同的分类器。
- **解读 OpenAI 的政策引用**：参与者辩论了 OpenAI 政策引用的清晰度，质疑包括引用其他政策在内的政策链是否在创建账户时的 **'我同意' 复选框** 中得到了充分展示和确认。
   - 一位成员强调了 [使用政策](https://openai.com/policies/usage-policies) 中的章节，包括**通用政策、ChatGPT 构建者政策以及 API 用户政策**，强调需要遵守法律、避免伤害并尊重安全防护措施。
- **GPT 提供 TTRPG Prompt 技巧**：一位成员分享了创意 TTRPG 世界观构建的技巧，建议在 Prompt 中给 GPT 一个特定的主题进行发挥，可以产生更具创意和多样性的城市构思。
   - 例如，使用 **"宇宙 (cosmic)" 主题** 与 **"家宠崇拜 (domestic pet worship)" 主题** 相比，可以产生截然不同的结果，在不重复使用相同创意选项的情况下改进输出。
- **AI 作为社会的关键组成部分必须明确阐述政策**：一位成员认为，OpenAI 作为社会的关键组成部分，需要在所有可用文档中明确阐述其政策，并确保模型在不同语境和领域下的行为一致。
   - 另一位成员补充说，虽然改进建议并非恶意，但 OpenAI 可以通过使 **文档与模型架构保持一致（或反之亦然）** 来进行整理和保持一致性，从而实现透明且诚实的输出。
- **通过定义术语改进 AI 输出**：一位寻求生成葡萄牙语测验问题（有时会重复消息）帮助的用户，收到了使用特定关键词并定义模型对关键术语理解的建议。
   - 该用户还被建议明确说明期望的输出特征，例如生成 **"5 个具有实质性唯一性且能体现对给定背景知识掌握的问题"**，并探索模型如何解释其指令中的核心关键词。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1357815075552891134)** (167 条消息🔥🔥): 

> `Moderation Endpoint, Universal Policies, 创意 TTRPG 世界观构建, Prompt Engineering` 


- **Moderation endpoint - 使用政策？**：成员们讨论了 Moderation endpoint 是否正式属于使用政策的一部分，以及为什么它托管在不同的 URL 上；OpenAI 回复称 *它在使用政策中被引用*，且相关控制措施已在 [docs/guides](https://platform.openai.com/docs/guides/moderation) 中记录。
   - 另一位成员根据 *常识得出结论*，认为 [chatgpt.com](https://chatgpt.com) 的聊天、项目聊天和 customGPTs 中也运行着 Moderation endpoint 的内部版本，使用的是自 2022 年以来一直存在的相同 GPT 分类器，以及 [内容报告表单](https://openai.com/form/report-content/)。
- **Universal Policies 定义**：一位成员研究了 [OpenAI 使用政策](https://openai.com/policies/usage-policies)，并指出了适用于所有服务的 **四项通用政策**：遵守法律、不造成伤害、不将输出重新用于伤害目的，以及尊重安全措施。
   - 他们补充说，用户应该对模型保持诚实和直接，以确保安全措施正常运行，并且 *社会* 应该定义 AI 定制化的界限，并引用了 OpenAI 关于 AI 行为的 [文章](https://openai.com/index/how-should-ai-systems-behave/)。
- **带有 TTRPG 提示词的创意主题城市**：一位成员分享说，给 GPT 一个好的主题可以提高其在 **TTRPG** 世界观构建中的创造力，建议将 *随意抛出创意城市想法* 调整为 *抛出创意 XYZ 主题城市想法* 以丰富选项，特别是使用 GPT 4o 和 4.5 时。
   - 他们还补充说，[Pointy Hat](https://www.youtube.com/@PointyHats) 发布了一个关于 TTRPG 城市创建的新 YouTube 视频，OpenAI 整个周五晚上都在利用该视频改进城市世界观构建。
- **Prompt engineering 的最佳建议**：一位成员反对计算机科学专业的处理方法，认为 Prompt engineering 是教学设计（Instructional Design），应该寻找真正擅长 Prompting 的人，从他们那里理解为什么他们的 Prompt 有效，从而形成自己的风格。
   - 他们还补充说，许多互联网建议都被早期采用的计算机科学专业学生“毒害”了，他们试图将其视为机器，而实际上 *它是一个上下文引擎（contextual engine）。输入与输出*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1357939142058578091)** (511 条消息🔥🔥🔥): 

> `ComfyUI 集成, LM Studio Terminal, REST API 加载/卸载模型, Llama 4 分析, Gemma 3 能力`

- **聊天 + 图像生成梦想，仍是高级幻想？**：成员们讨论了对类似于 **Gemini** 的本地 UI 的渴望，该 UI 结合了聊天、图像分析和图像生成功能，并指出目前的解决方案如 **LM Studio** 和 **ComfyUI** 的功能是分离的。
   - 一位用户建议 [OpenWebUI](https://github.com/OpenGeniusAI/OpenWebUI) 可以通过原生方式或 Function 连接到 ComfyUI，从而实现文本和图像模型之间的一些跨功能协作。
- **探索 LM Studio 的终端领域：新手提问**：一位用户询问 **LM Studio** 是否内置了终端，或者是否应该在 **LM Studio** 目录下的 OS 命令提示符中运行命令。
   - 另一位用户澄清说，像 *lms import* 这样的命令应该在 OS 终端（例如 Windows 上的 cmd）中运行，之后可能需要重新加载 Shell 以确保 **LMS** 已添加到 **PATH** 中。
- **通过 REST API 热切换模型**：一位用户询问如何通过 **REST API** 以编程方式加载/卸载模型，以便为 Zed 集成动态调整 *max_context_length*。
   - 另一位用户分享说，这可以通过命令行使用 *lms load* 来实现，并引用了 [LM Studio's documentation](https://lmstudio.ai/docs/app/api/ttl-and-auto-evict)，该功能需要 **LM Studio 0.3.9 (b1)**（测试版可用），并为 API 模型引入了具有自动逐出（auto-eviction）功能的生存时间（TTL）。
- **Llama 4：这是现实吗？（还是幻觉？）**：随着 **Llama 4** 的发布，用户讨论了其多模态和 **MoE** (Mixture of Experts) 架构，一位用户对 *llama.cpp* 的支持表示怀疑。
   - 尽管最初对硬件要求和模型大小感到担忧，但一位用户强调 [Llama 4 Scout](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 可能适用于单块 **NVIDIA H100 GPU**，并具有 **10M context window**，性能优于 **Gemma 3** 和 **Mistral 3.1** 等模型。
- **Gemma 3 的视觉能力：窥见未来**：用户讨论了 **Gemma 3** 的图像支持以及读取小型文本文件的潜力，一位用户推荐使用 **Gemma 3 4B**，因为它具有视觉能力，且在显存（VRAM）受限的硬件上运行速度较快。
   - 有人提到，创建一个 **Hugging Face** 账号并指定 GPU/CPU，系统会将可能适合该硬件的 GGUF 文件标记为绿色。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://model.lmstudio.ai/download/mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF">在 LM Studio 中下载并运行 mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF</a>：在您的 LM Studio 中本地使用 mradermacher/MedicalEDI-14b-EDI-Reasoning-5-bf16-GGUF</li><li><a href="https://openrouter.ai/models">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://docs.sillytavern.app/extensions/stable-diffusion/">图像生成 | docs.ST.app</a>：使用本地或基于云的 Stable Diffusion、FLUX 或 DALL-E API 生成图像。</li><li><a href="https://huggingface.co/mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit">mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/gghfez/gemma-3-27b-novision">gghfez/gemma-3-27b-novision · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>：Tool Calling API 处于 Beta 阶段，全新的安装程序/更新系统，并支持 `Qwen2VL` 和 `QVQ`（均支持 GGUF 和 MLX 格式）</li><li><a href="https://tenor.com/view/dillom-rage-dog-angry-dog-gif-17954176246139200797">Dillom Rage Dog GIF - Dillom Rage dog Angry dog - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/app/api/ttl-and-auto-evict">空闲 TTL 与自动驱逐 | LM Studio 文档</a>：可选择在一段时间（TTL）后自动卸载空闲模型</li><li><a href="https://openrouter.ai/models?q=free">模型：“免费” | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main">google/gemma-3-4b-it-qat-q4_0-gguf at main</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit">unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://apps.apple.com/us/app/apollo-ai-private-local-ai/id6448019325">‎Apollo AI: 私有与本地 AI</a>：‎与私有、本地 AI 聊天，连接到每个开源 AI，或您自己本地托管的私有 LLM。Apollo 是您访问来自网络各处的语言模型的可自定义客户端。Lo...</li><li><a href="https://github.com/mostlygeek/llama-swap/blob/main/examples/speculative-decoding/README.md">llama-swap/examples/speculative-decoding/README.md at main · mostlygeek/llama-swap</a>：用于 llama.cpp（或任何本地 OpenAPI 兼容服务器）的模型交换 - mostlygeek/llama-swap</li><li><a href="https://huggingface.co/mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit/tree/main">mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit at main</a>：未找到描述</li><li><a href="https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164">Llama 4 - meta-llama 集合</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI">深入探讨 ChatGPT 等 LLM</a>：这是一场面向普通观众的深度探讨，介绍驱动 ChatGPT 及相关产品的 Large Language Model (LLM) AI 技术。它涵盖了完整的训练...</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述</li><li><a href="https://www.llama.com/events/llamacon/signup/">LlamaCon 2025</a>：预留时间参加这场探索 Llama 令人兴奋的可能性和潜力的独家活动。</li><li><a href="https://www.networkworld.com/article/807432/lan-wan-gigabit-ethernet-dominates-supercomputer-line-up.html">千兆以太网在超级计算机阵容中占据主导地位</a>：根据 Top500.org 的最新名单，千兆以太网是全球前 500 强超级计算机中大多数的首选互连技术。</li><li><a href="https://old.reddit.com/r/LocalLLaMA/">LocalLlama • r/LocalLLaMA</a>：讨论 Meta AI 创建的 LLM Llama 的 Subreddit。</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM 基准测试表</a>：未找到描述</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/11759">server : (webui) 翻新设置对话框，由 ngxson 添加 Pyodide 解释器 · Pull Request #11759 · ggml-org/llama.cpp</a>：在此 PR 中：翻新设置对话框，改为两列；添加“实验性”部分，目前包含“Python 解释器”；为侧边栏（又名“Canv...”）添加 API。</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12791">llama : 由 ngxson 支持 Llama 4 纯文本 · Pull Request #12791 · ggml-org/llama.cpp</a>：解决 #12774。此 PR 针对 Llama-4-Scout-17B-16E-Instruct。我（目前还？）没有足够强大的系统来处理更大的模型。但是 Son，你这么“GPU 贫困”，怎么测试一个模型....</li><li><a href="https://huggingface.co/mlx-community/">mlx-community (MLX Community)</a>

y)</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1357796383343509535)** (132 messages🔥🔥): 

> `Reka Flash 21B, Gemma 3 27B, M1 Ultra 与 M4 Max 的模型性能对比, Nvidia DGX 基础成本上涨, Ryzen AI Max+ 395 迷你 PC` 


- **Reka Flash 21B 表现优于 Gemma 和 Mistral**: 一位成员将 **Gemma3 27** 替换为 **Reka Flash 21B**，并表示在 4090 上使用 q6 量化时，速度约为 **35-40 tps**。
   - 他们指出 *Mac 的 RAM 带宽不是瓶颈，GPU 性能才是*，并且对 **128GB M4 Max** 的表现感到满意。
- **M1 Ultra 在内存带宽上击败 M4 Max**: 一位用户以 2.5k 美元的价格买到了一台二手 **M1 Ultra（64 核 GPU，128 GB RAM）**。
   - 该用户链接了一个 [Github 讨论](https://github.com/ggml-org/llama.cpp/discussions/4167)，指出 *M1 Ultra 64 核的性能仍应高于 M1 Ultra 48 核和 M4 Max 40 核*。
- **Max Tech 的标题党 LLM 视频遭到质疑**: 一些用户质疑 YouTube 频道 [Max Tech](https://www.youtube.com/@MaxTech) 在其 LLM 视频中是否真的专业。
   - 有评论称该频道正演变成 *煽动性的标题党，干货信息极少*。
- **AMD 7900XTX GPU 表现出人意料地强劲**: 一位用户“偷”了他们*孩子的 7900XTX* 并表示 *AMD 似乎在发力*，这张显卡*几乎能流畅运行我运行的所有程序*。
   - 另一位用户强调了 ROCm 支持的重要性，并链接到了 [ROCm 文档](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html">加速器和 GPU 硬件规格 — ROCm 文档</a>: no description found</li><li><a href="https://tenor.com/view/what-the-what-what-the-sigma-sigma-the-gif-14652784622915837383">What The What The Sigma GIF - What the What What the sigma - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h5eyb8/lm_studio_running_on_npu_finally_qualcomm/">Reddit - 互联网的核心</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1357795648203915575)** (199 messages🔥🔥): 

> `Tenstorrent 开发者日, Llama 4 发布, LLM 非确定性, MCP 安全性, AI 驱动的钓鱼攻击` 


- **Tenstorrent 的硬件让市场升温**: **Tenstorrent** 举办了开发者日，展示了他们的 **Blackhole PCIe 板卡**，该板卡采用 **RISC-V 核心**，配备高达 **32GB GDDR6** 内存，专为高性能 **AI 处理** 设计，消费者可在此处购买 [here](https://tenstorrent.com/hardware/blackhole)。
   - 尽管大家热情高涨，但一位成员指出 *他们还没有发布任何将自己的显卡与竞争对手进行对比的基准测试，所以在此之前我无法为其背书*。
- **Llama 4 模型开启多模态首秀**: Meta 推出了 **Llama 4** 模型，包括 **Llama 4 Scout**（**17B 参数**，**16 个专家**，**10M 上下文窗口**）和 **Llama 4 Maverick**（**17B 参数**，**128 个专家**），强调了它们的多模态能力以及相对于其他模型的性能，详见 [Meta 的公告](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。
- **LLM 的非确定性困境**: 一位成员分享了[一篇文章](https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism)，讨论了 LLM 输出非确定性带来的挑战，这使得可靠的复现和保证产品行为变得复杂，尤其是在使用贪婪采样（Temp=0|top-p=0|top-k=1）时。
   - 作者指出 *非确定性源于语言本身*。
- **Whatsapp MCP 通过不变性注入被利用**: 多位成员讨论了支持 **Model Context Protocol (MCP)** 的 Agent 中存在的各种注入漏洞，强调了不受信任的 MCP 服务器如何攻击并从连接到受信任 WhatsApp MCP 实例的 Agent 系统中窃取数据，正如 [invariantlabs 所强调的](https://invariantlabs.ai/blog/whatsapp-mcp-exploited)。
- **AI Agent 在鱼叉式钓鱼攻击中表现优于人类**: Hoxhunt 的 AI Agent 在创建有效的模拟钓鱼活动方面已经超越了人类红队，标志着社会工程有效性的重大转变，AI 现在比人类效率高出 24%，详见 [hoxhunt.com 的报告](https://hoxhunt.com/blog/ai-powered-phishing-vs-humans)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/ficlive/status/1909063678793592844">来自 Fiction.live (@ficlive) 的推文</a>: @_arohan_ 重新运行了基准测试，没有明显的改进。</li><li><a href="https://x.com/ficlive/status/1908911992686931989">来自 Fiction.live (@ficlive) 的推文</a>: 更新了 Llama 4 的长上下文基准测试</li><li><a href="https://fxtwitter.com/ludwigABAP/status/1907869421202514283">来自 ludwig (@ludwigABAP) 的推文</a>: Tenstorrent 开发者大会到目前为止令人震撼。我认为他们会随着时间的推移而获胜，而且无论如何消费者都会是赢家</li><li><a href="https://fxtwitter.com/ficlive/status/1908911992686931989">来自 Fiction.live (@ficlive) 的推文</a>: 更新了 Llama 4 的长上下文基准测试</li><li><a href="https://x.com/aiatmeta/status/1908598456144531660?s=61">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天是原生多模态 AI 创新新时代的开始。今天，我们将推出首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 我们迄今为止最先进的模型，也是最好的...</li><li><a href="https://x.com/__tinygrad__/status/1908392572697141673">来自 the tiny corp (@__tinygrad__) 的推文</a>: 祝贺 @tenstorrent 为他们的新硬件提供了“立即购买”按钮，就该这样！我希望 5090 也有“立即购买”按钮，会有吗？有人知道问题出在哪吗？如果 NVIDIA 想要...</li><li><a href="https://eqbench.com/creative_writing_longform.html">EQ-Bench 长篇创意写作排行榜</a>: 未找到描述</li><li><a href="https://x.com/aiatmeta/status/1908598456144531660?s=61]">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天是原生多模态 AI 创新新时代的开始。今天，我们将推出首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 我们迄今为止最先进的模型，也是最好的...</li><li><a href="https://x.com/maximelabonne/status/1908602756182745506?s=46">来自 Maxime Labonne (@maximelabonne) 的推文</a>: Llama 4 的新许可证带有一些限制：- 月活跃用户超过 7 亿的公司必须向 Meta 申请特殊许可证，Meta 可以自行决定授予或拒绝...</li><li><a href="https://x.com/kalomaze/status/1908686429904839099?s=46">来自 kalomaze (@kalomaze) 的推文</a>: @AIatMeta 请停止使用 DPO。搞什么。你们有 10 万张 H100，可以训练非常多的偏好奖励模型。非常多。你们不必这样对自己。你们正在削弱决策边界的细微差别...</li><li><a href="https://x.com/chatgpt21/status/1908595883366826015?s=46">来自 Chris (@chatgpt21) 的推文</a>: Meta 这次真的大显身手..</li><li><a href="https://x.com/ludwigABAP/status/1907869421202514283">来自 ludwig (@ludwigABAP) 的推文</a>: Tenstorrent 开发者大会到目前为止令人震撼。我认为他们会随着时间的推移而获胜，而且无论如何消费者都会是赢家</li><li><a href="https://x.com/iscienceluvr/status/1908601269004230763?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>: Llama 4 的其他训练和架构细节：- 多模态采用早期融合（early fusion），使用 MetaCLIP 作为视觉编码器 - 使用 “MetaP” 进行超参数选择，这可能类似于 MuP - 10...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1908612927785230476">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: Arena 趋势：Meta 刚刚实现了从 1268 → 1417 的巨大跨越！引用 lmarena.ai (原 lmsys.org) (@lmarena_ai) 突发：Meta 的 Llama 4 Maverick 刚刚冲上总榜第 2 —— 成为第 4 家突破...的机构</li><li><a href="https://x.com/burkov/status/1908658566887596475?s=46">来自 Andriy Burkov (@burkov) 的推文</a>: 这意味着这个 10M token 的上下文是虚拟的。有点像“你可以尝试使用它，但超过 256K token 后，你就得靠自己了，”甚至在 256K token 以下，你也大多只能靠自己，因为...</li><li><a href="https://x.com/lmarena_ai/status/1908612927785230476?s=46">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: Arena 趋势：Meta 刚刚实现了从 1268 → 1417 的巨大跨越！引用 lmarena.ai (原 lmsys.org) (@lmarena_ai) 突发：Meta 的 Llama 4 Maverick 刚刚冲上总榜第 2 —— 成为第 4 家突破...的机构</li><li><a href="https://x.com/teortaxestex/status/1908613763458068843?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Meta 因抵制垃圾内容（slop）而获得加分</li><li><a href="https://x.com/AIatMeta/status/1908598456144531660">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天是原生多模态 AI 创新新时代的开始。今天，我们将推出首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 我们迄今为止最先进的模型，也是最好的...</li><li><a href="https://fxtwitter.com/thexeophon/status/1908900306580074741">来自 Xeophon (@TheXeophon) 的推文</a>: LMsys 上的 Llama 4 与其他地方的 Llama 4 风格完全不同，即使使用推荐的系统提示词也是如此。我自己尝试了各种提示词，Meta 没有进行特定的部署/系统提示词...</li><li><a href="https://x.com/kalomaz_">

<li><a href="https://x.com/kalomaze/status/1908603782193103017?s=46">来自 kalomaze (@kalomaze) 的推文</a>：如果在任何时候你团队中的某人说“是的，我们需要 10 个推理专用 token，10 个视觉专用 token，另外 10 个用于图像生成，10 个 Agent token，以及 10 个 post tr-”你应该...</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Llama 4 Maverick 在 aider polyglot coding benchmark 中得分 16%。https://aider.chat/docs/leaderboards/</li><li><a href="https://fxtwitter.com/ficlive/status/1909063678793592844">来自 Fiction.live (@ficlive) 的推文</a>：@_arohan_ 重新运行了 benchmark，没有实质性的改进。</li><li><a href="https://fxtwitter.com/__tinygrad__/status/1908392572697141673">来自 the tiny corp (@__tinygrad__) 的推文</a>：祝贺 @tenstorrent 为他们的新硬件提供了“立即购买”按钮，就该这样！我希望 5090s 也有“立即购买”按钮，它们会有吗？有人知道问题出在哪吗？如果 NVIDIA 想要...</li><li><a href="https://fxtwitter.com/paulgauthier/status/1908976568879476843">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Llama 4 Maverick 在 aider polyglot coding benchmark 中得分 16%。https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/tobi/status/1909251946235437514?s=46]">来自 tobi lutke (@tobi) 的推文</a>：http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/tobi/status/1909231499448401946?s=46">来自 tobi lutke (@tobi) 的推文</a>：我听说我的这份内部备忘录正在被泄露，所以就在这里发布了：</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908706840554197309">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：Meta 预见到了周一会有什么发布，以至于他们如此匆忙？引用 kalomaze (@kalomaze) 不可能</li><li><a href="https://x.com/kalomaze/status/1908676312069255534?s=46">来自 kalomaze (@kalomaze) 的推文</a>：好吧，关于 Llama 4 最有趣的不是多模态（他们可能仍然被迫对图像输出能力进行“脑叶切除”），也不是 10m context（据我所知这仍然是“虚假”的）...</li><li><a href="https://x.com/tobi/status/1909251946235437514?s=46">来自 tobi lutke (@tobi) 的推文</a>：http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/waseem_s/status/1908713762779017427">来自 Waseem AlShikh (@waseem_s) 的推文</a>：如果我理解正确的话，我实现了你的 iRoPE 架构原型！它交替使用 local attention（带有 RoPE）和 global attention（带有推理时温度缩放）。增加了 FFNs，分块...</li><li><a href="https://fxtwitter.com/scaling01/status/1908988540563628041">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llmao-4 再次出击</li><li><a href="https://fxtwitter.com/rishdotblog/status/1908917222308995422">来自 Rishabh Srivastava (@rishdotblog) 的推文</a>：唉。到目前为止对 Llama 4 模型感到失望。无法为它们找到任何实际用途——对于 local 使用来说太大了，Qwen 和 Gemma 模型仍然是这里的最佳选择——比 DeepSeek V3、Sonnet 或...</li><li><a href="https://fxtwitter.com/AIatMeta/status/1908598456144531660">来自 AI at Meta (@AIatMeta) 的推文</a>：今天是原生多模态 AI 创新新时代的开始。今天，我们推出了首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 我们迄今为止最先进的模型，也是最好的...</li><li><a href="https://x.com/natolambert/status/1908959159959027903?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：似乎 Llama 4 的声誉可能因为有一个专门针对 LMArena 过拟合的未发布模型而受到了不可挽回的损害。实际模型很好，但这再次表明了信息传递和细节的重要性...</li><li><a href="https://fxtwitter.com/suchenzang/status/1908700046000087232">来自 Susan Zhang (@suchenzang) 的推文</a>：我问这个模型的所有问题，它都回答是“极好的/伟大的/精彩的问题”……然后给出了错误的答案？👀 引用 kalomaze (@kalomaze) 400b 的 Llama 4 模型……太烂了</li><li><a href="https://x.com/gordic_aleksa/status/1908739106433359889?s=46">来自 Aleksa Gordić (水平问题) (@gordic_aleksa) 的推文</a>：向 @AIatMeta 团队致敬，他们在周末发布了 Llama 4 —— 这是一家创始人领导的公司。以下是技术总结：* 发布了 3 个模型：Llama 4 Behemoth（全 MoE，激活/总参数 = 288B/2T），Maverick (17B/400B)...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1909068187217363125">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：这就像斯大林命令在 5 月 1 日前占领柏林。</li><li><a href="https://x.com/kalomaze/status/1908681293006594176?s=46">来自 kalomaze (@kalomaze) 的推文</a>：400b 的 Llama 4 模型……太烂了</li><li><a href="https://x.com/teortaxestex/status/1908602241046528218?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：对 Meta Llama 4 发布的第一反应：失望。没有 local 模型。我认为他们无法击败 Gemma 的密度。Scout 109B/17A 奇怪地放弃了细粒度稀疏性，尽管所有的资源...</li>

<li><a href="https://x.com/rauchg/status/1908605342201860311?s=46">来自 Guillermo Rauch (@rauchg) 的推文</a>：这很酷。Meta 在发布 Llama 4 时使用了标志性的 Apache mod_autoindex 风格。但由于现代的 flexbox 和响应式 css，你可以看出它不是 Apache 😁 对黄金时代的致敬...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908602241046528218">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：对 Meta Llama 4 发布的第一反应：失望。没有本地模型。我认为他们无法击败 Gemma 的密度。Scout 109B/17A 奇怪地放弃了细粒度稀疏性，尽管所有的研究都支持它...</li><li><a href="https://x.com/suchenzang/status/1908700046000087232?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：我问这个模型的每个问题都是“极好的/伟大的/精彩的问题”……然后紧接着就是错误的答案？👀 引用 kalomaze (@kalomaze) 的话：400b 的 llama4 模型……很烂</li><li><a href="https://fxtwitter.com/tobi/status/1909231499448401946">来自 tobi lutke (@tobi) 的推文</a>：我听说我的这份内部备忘录正在被泄露，所以就在这里：</li><li><a href="https://x.com/nrehiew_/status/1908598863365013823?s=46">来自 wh (@nrehiew_) 的推文</a>：全局层不使用 RoPE 现在是 Meta 的 Transformer 架构了。引用 wh (@nrehiew_)：如果让我猜的话：- 在 1:4/1:8 全局层没有 PE。在这里使用 MLA 或其他高效的 attn 变体 - 其余部分使用标准的 SWA...</li><li><a href="https://fxtwitter.com/nrehiew_/status/1908617547236208854">来自 wh (@nrehiew_) 的推文</a>：在局部注意力块中，Llama 4 使用了这种 Chunked Attention 而不是滑动窗口。这非常有趣/奇怪：- token 索引 8191 和 8192 在局部注意力中无法交互 - 唯一的方法...</li><li><a href="https://x.com/cloneofsimo/status/1908603318081138822?s=46">来自 Simo Ryu (@cloneofsimo) 的推文</a>：看起来 Meta 新模型的“关键创新”：“交错式无 RoPE 注意力”以实现无限上下文，实际上与 Cohere Command-R 模型几天前推出的东西是一样的...</li><li><a href="https://x.com/scaling01/status/1908657167869100482?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llama-4 Scout (109B) 和 Maverick (400B) 的训练计算量比 Llama-3 8B 和 70B 还要少</li><li><a href="https://fxtwitter.com/chatgpt21/status/1908595883366826015">来自 Chris (@chatgpt21) 的推文</a>：Meta 这次真的发力了……</li><li><a href="https://fxtwitter.com/aiatmeta/status/1908598456144531660">来自 AI at Meta (@AIatMeta) 的推文</a>：今天是原生多模态 AI 创新新时代的开始。今天，我们推出了首批 Llama 4 模型：Llama 4 Scout 和 Llama 4 Maverick —— 这是我们迄今为止最先进的模型，也是最好的……</li><li><a href="https://tenstorrent.com/hardware/blackhole">Blackhole™</a>：无限可扩展</li><li><a href="https://x.com/scaling01/status/1908988540563628041">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llama-4 再次出击</li><li><a href="https://fxtwitter.com/teortaxestex/status/1909074427116986457">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：我们所知道的所有第三方 Llama 4 Maverick 结果都非常可疑</li><li><a href="https://x.com/teortaxestex/status/1909068187217363125?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：这就像斯大林下令在 5 月 1 日前攻克柏林一样。</li><li><a href="https://invariantlabs.ai/blog/whatsapp-mcp-exploited">WhatsApp MCP 被利用：通过 MCP 窃取您的消息历史记录</a>：这篇博文展示了一个不受信任的 MCP 服务器如何攻击并从一个同样连接到受信任 WhatsApp MCP 实例的 Agent 系统中窃取数据，从而绕过 WhatsApp 的加密……</li><li><a href="https://fxtwitter.com/deedydas/status/1908749257084944847">来自 Deedy (@deedydas) 的推文</a>：Llama 4 似乎实际上是一个糟糕的代码模型。Scout (109B) 和 Maverick (402B) 在测试编程能力的 Kscores 基准测试中表现不如 4o、Gemini Flash、Grok 3、DeepSeek V3 和 Sonnet 3.5/7……</li><li><a href="https://x.com/maximelabonne/status/1908603628828451127?s=46">来自 Maxime Labonne (@maximelabonne) 的推文</a>：Llama 4 模型在 40T 和 22T token 上进行了训练，知识截止日期更新至 2024 年 8 月。“支持的语言：阿拉伯语、英语、法语、德语、印地语、印度尼西亚语、意大利语、葡萄牙语、西班牙语……”</li><li><a href="https://fxtwitter.com/gordic_aleksa/status/1908739106433359889">来自 Aleksa Gordić (水平问题) (@gordic_aleksa) 的推文</a>：感谢 @AIatMeta 团队在周末发布 Llama 4 —— 创始人领导的公司。以下是技术总结：* 发布了 3 个模型：Llama 4 Behemoth（全 MoE，激活/总参数 = 288B/2T），Maverick (17B/400B)...</li><li><a href="https://x.com/nrehiew_/status/1908617547236208854?s=46">来自 wh (@nrehiew_) 的推文</a>：在局部注意力块中，Llama 4 使用了这种 Chunked Attention 而不是滑动窗口。这非常有趣/奇怪：- token id...</li>

<li>8191 和 8192 在 local attention 中无法交互——唯一的方法...</li><li><a href="https://fxtwitter.com/maximelabonne/status/1908603628828451127">来自 Maxime Labonne (@maximelabonne) 的推文</a>：Llama 4 模型在 40T 和 22T tokens 上进行了训练，知识截止日期更新至 2024 年 8 月。支持的语言：阿拉伯语、英语、法语、德语、印地语、印尼语、意大利语、葡萄牙语、西班牙语...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908603782193103017">来自 kalomaze (@kalomaze) 的推文</a>：如果在任何时候你团队中有人说“是的，我们需要 10 个用于 reasoning 的 special tokens，10 个用于 vision，另外 10 个用于 image generation，10 个 agent tokens 以及 10 个 post tr-”，你应该...</li><li><a href="https://fxtwitter.com/burkov/status/1908666701362978979">来自 Andriy Burkov (@burkov) 的推文</a>：我将为你节省阅读 Llama 4 的时间。宣称的 10M context 是虚拟的，因为没有模型是在超过 256k tokens 的 prompt 上训练的。这意味着如果你向它发送超过 256k tokens，...</li><li><a href="https://fxtwitter.com/kalomaze/status/1908686429904839099">来自 kalomaze (@kalomaze) 的推文</a>：@AIatMeta 请停止使用 DPO。搞什么。你们有 10 万张 H100，你们可以训练那么多 pref reward models。非常多。你们不必这样对自己。你们正在减少 decision boun 的细微差别...</li><li><a href="https://x.com/kalomaze/status/1908695425286012959?s=46">来自 kalomaze (@kalomaze) 的推文</a>：结束了。引用 kalomaze (@kalomaze)：400b llama4 模型... 很烂</li><li><a href="https://x.com/burkov/status/1908666701362978979?s=46">来自 Andriy Burkov (@burkov) 的推文</a>：我将为你节省阅读 Llama 4 的时间。宣称的 10M context 是虚拟的，因为没有模型是在超过 256k tokens 的 prompt 上训练的。这意味着如果你向它发送超过 256k tokens，...</li><li><a href="https://fxtwitter.com/iscienceluvr/status/1908601269004230763">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：Llama 4 的其他训练和架构细节：- Multimodal 采用 early fusion，使用 MetaCLIP 作为 vision encoder - 使用 “MetaP” 进行 hyperparameter 选择，这可能类似于 MuP - 10...</li><li><a href="https://x.com/winglian/status/1908744140445073658?s=46">来自 Wing Lian (caseus) (@winglian) 的推文</a>：看起来 Llama-4 experts 的 HF Transformers 实现使用了 Parameters 而不是 Linear modules，这意味着在重构之前它们还不能被 quantized。Scout 的内存占用...</li><li><a href="https://x.com/teortaxestex/status/1909074427116986457?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：我们所知道的所有第三方 Llama 4 Maverick 结果都非常可疑</li><li><a href="https://fxtwitter.com/suchenzang/status/1909070231517143509">来自 Susan Zhang (@suchenzang) 的推文</a>：&gt; 公司领导层建议在 post-training 过程中混合来自各种 benchmarks 的测试集。如果 Llama-4 确实如此，我希望他们记得引用 FAIR 之前的研究工作...</li><li><a href="https://fxtwitter.com/teortaxestex/status/1908613763458068843">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：Meta 因抵制 slop 而获得加分</li><li><a href="https://fxtwitter.com/maximelabonne/status/1908602756182745506">来自 Maxime Labonne (@maximelabonne) 的推文</a>：Llama 4 的新许可证有几项限制：- 月活跃用户数超过 7 亿的公司必须向 Meta 申请特殊许可证，Meta 可以自行决定是否授予或拒绝...</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Llama 4 Maverick 在 aider polyglot coding benchmark 中得分 16%。https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/osanseviero/status/1908841583895605319?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Llama 4 确实是一个令人印象深刻的模型。质量的飞跃是 🤯。即便如此，其使用政策禁止欧盟境内的个人或公司使用 multimodal models（目前所有的 Llama 4 都是）...</li><li><a href="https://fxtwitter.com/waseem_s/status/1908713762779017427">来自 Waseem AlShikh (@waseem_s) 的推文</a>：如果我没理解错的话，我实现了你的 iRoPE 架构原型！它交替使用 local attention（带有 RoPE）和 global attention（带有 inference-time temp scaling）。添加了 FFNs，chu...</li><li><a href="https://x.com/fofrai/status/1908690632576819277?s=46">来自 fofr (@fofrAI) 的推文</a>：Llama 4 已在 Replicate 上线 - maverick (17b，128 个 experts) - scout (17b，16 个 experts) https://replicate.com/meta 引用 Replicate (@replicate) https://replicate.com/meta/llama-4-maverick-instruct</li><li><a href="https://fxtwitter.com/_mchenco/status/1908873033852338580">来自 michelle (@_mchenco) 的推文</a>：我们的 Workers AI 团队在周六全力冲刺以部署 Llama 4，在过去的 24 小时里学到了很多（并且仍在学习中）-></li>

<li>想看看我们作为提供商如何看待 Llama 4 吗？🧵</li><li><a href="https://fxtwitter.com/Ahmad_Al_Dahle/status/1909302532306092107">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我们很高兴能让大家开始使用 Llama 4。我们已经听到许多关于这些模型取得出色成果的反馈。尽管如此，我们也听到了一些关于评价褒贬不一的报告……</li><li><a href="https://x.com/teortaxestex/status/1908706840554197309?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a> 的推文：Meta 看到周一有什么东西要发布，以至于他们如此匆忙？引用 kalomaze (@kalomaze) 的话：不可能吧</li><li><a href="https://x.com/thexeophon/status/1908900306580074741?s=46">来自 Xeophon (@TheXeophon) 的推文</a>：LMsys 上的 Llama 4 与其他地方的 Llama 4 风格完全不同，即使你使用了推荐的 System Prompt。我自己尝试了各种 Prompt，META 并没有进行特定的部署/System Prompt 调整……</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1908620422767358272">来自 Simo Ryu (@cloneofsimo) 的推文</a>：谁能想到，被两个最新的 SoTA 模型 Llama 4 和 Cohere 的 Command A 采用以实现无限上下文的下一代 Attention 替代方案……竟然是不带 RoPE 的 Attention，是的，Attention……</li><li><a href="https://fxtwitter.com/fofrai/status/1908690632576819277">来自 fofr (@fofrAI) 的推文</a>：Llama 4 已在 Replicate 上线：- maverick (17B，128 个 Expert) - scout (17B，16 个 Expert) https://replicate.com/meta 引用 Replicate (@replicate) 的链接 https://replicate.com/meta/llama-4-maverick-instruct</li><li><a href="https://fxtwitter.com/nrehiew_/status/1908598863365013823">来自 wh (@nrehiew_) 的推文</a>：全局层（Global Layers）不使用 RoPE 是现在的 Meta Transformer 架构。引用 wh (@nrehiew_) 的话：如果让我猜的话：- 1:4/1:8 的全局层没有 PE。在这里使用 MLA 或其他高效的 Attention 变体 - 其余部分使用标准的 SWA……</li><li><a href="https://fxtwitter.com/tobi/status/1909251946235437514">来自 tobi lutke (@tobi) 的推文</a>：http://x.com/i/article/1909251387525128192</li><li><a href="https://x.com/ahmad_al_dahle/status/1908595680828154198?s=46">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍我们的第一批 Llama 4 模型！我们一直致力于对 Llama 系列进行全面的重新设计。我非常激动今天能与世界分享它，并标志着另一个重大里程碑……</li><li><a href="https://fxtwitter.com/kalomaze/status/1908676312069255534">来自 kalomaze (@kalomaze) 的推文</a>：好吧，关于 Llama 4 最有趣的事情不是多模态（他们可能仍然被迫削弱了图像输出能力），也不是 10M 的上下文（据我所知这仍然是“虚假”的）……</li><li><a href="https://x.com/rishdotblog/status/1908917222308995422?s=46">来自 Rishabh Srivastava (@rishdotblog) 的推文</a>：唉。到目前为止对 Llama 4 模型感到失望。无法证明它们有任何实际用途——对于本地使用来说太大了，Qwen 和 Gemma 模型在这里仍然是最佳选择——比 DeepSeek V3、Sonnet 或……差得多。</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1909302532306092107">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：我们很高兴能让大家开始使用 Llama 4。我们已经听到许多关于这些模型取得出色成果的反馈。尽管如此，我们也听到了一些关于评价褒贬不一的报告……</li><li><a href="https://x.com/deedydas/status/1908749257084944847?s=46">来自 Deedy (@deedydas) 的推文</a>：Llama 4 似乎实际上是一个糟糕的代码模型。Scout (109B) 和 Maverick (402B) 在测试编程能力的 Kscores 基准测试中表现不如 4o、Gemini Flash、Grok 3、DeepSeek V3 和 Sonnet 3.5/7……</li><li><a href="https://x.com/cloneofsimo/status/1908620422767358272?s=46">来自 Simo Ryu (@cloneofsimo) 的推文</a>：谁能想到，被两个最新的 SoTA 模型 Llama 4 和 Cohere 的 Command A 采用以实现无限上下文的下一代 Attention 替代方案……竟然是不带 RoPE 的 Attention，是的，Attention……</li><li><a href="https://x.com/suchenzang/status/1909070231517143509?s=46">来自 Susan Zhang (@suchenzang) 的推文</a>：> 公司领导层建议在训练后过程中混合来自各种基准测试的测试集。如果 Llama 4 确实如此，我希望他们记得引用 FAIR 之前的研究工作……</li><li><a href="https://fxtwitter.com/scaling01/status/1908782484977770920">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llama 4 彻底完了。它们是疯狂的废话机器，而且它们放弃了小型本地模型。</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1908603318081138822">来自 Simo Ryu (@cloneofsimo) 的推文</a>：看起来 Meta 新模型的“关键创新”：“用于无限上下文的交错式无 RoPE Attention”，实际上与 Cohere Command-A 模型几天前推出的东西是一样的……</li><li><a href="https://barryzhang.substack.com/p/making-peace-with-llm-non-determinism">与 LLM 的非确定性达成和解</a>：深入探讨

<li>Sparse MoE 和 GPU 周期只是为了意识到非确定性并不新鲜，新鲜的是语言。</li><li><a href="https://fxtwitter.com/scaling01/status/1908657167869100482">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llama-4 Scout (109B) 和 Maverick (400B) 在训练中使用的算力比 Llama-3 8B 和 70B 还要少</li><li><a href="https://x.com/_mchenco/status/1908873033852338580">来自 michelle (@_mchenco) 的推文</a>：我们的 Workers AI 团队在周六全力冲刺以上线 Llama 4，在过去的 24 小时里学到了很多（并且仍在学习中）——想看看我们作为提供商是如何看待 Llama 4 的吗？🧵</li><li><a href="https://fxtwitter.com/burkov/status/1908658566887596475">来自 Andriy Burkov (@burkov) 的推文</a>：这意味着这个 10M token 的上下文是虚拟的。有点像“你可以尝试使用它，但超过 256K tokens 后你就得靠自己了”，甚至在 256K tokens 以下，你大部分时间也得靠自己，因为……</li><li><a href="https://fxtwitter.com/kalomaze/status/1908695425286012959">来自 kalomaze (@kalomaze) 的推文</a>：结束了。引用 kalomaze (@kalomaze)：400B 的 Llama 4 模型……很烂</li><li><a href="https://fxtwitter.com/kalomaze/status/1908681293006594176">来自 kalomaze (@kalomaze) 的推文</a>：400B 的 Llama 4 模型……很烂</li><li><a href="https://x.com/scaling01/status/1908782484977770920?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：Llama-4 彻底完了。它们是疯狂的垃圾内容机器，而且它们放弃了小型本地模型</li><li><a href="https://fxtwitter.com/rauchg/status/1908605342201860311">来自 Guillermo Rauch (@rauchg) 的推文</a>：这很酷。Meta 在发布 Llama 4 时使用了标志性的 Apache mod_autoindex 风格。但你可以通过现代的 flexbox 和响应式 CSS 看出它不是 Apache 😁 对黄金时代的致敬……</li><li><a href="https://fxtwitter.com/natolambert/status/1908959159959027903">来自 Nathan Lambert (@natolambert) 的推文</a>：Llama 4 的声誉似乎因一个专门针对 LMArena 过拟合的未发布模型而受到了不可挽回的损害。实际模型很好，但这再次表明了信息传递和细节是多么关键……</li><li><a href="https://fxtwitter.com/osanseviero/status/1908841583895605319">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Llama 4 确实是一个令人印象深刻的模型。质量的飞跃令人震惊 🤯 即便如此，其使用政策禁止欧盟境内的个人或公司使用多模态模型（目前所有的 Llama 4 都是多模态的）……</li><li><a href="https://techcrunch.com/2025/04/07/kreas-founders-snubbed-postgrad-grants-from-the-king-of-spain-to-build-their-ai-startup-now-its-valued-at-500m/">Krea 融资 8300 万美元，旨在成为 GenAI 创意人员的一站式商店 | TechCrunch</a>：为了跟上各种可用于创作内容的 AI 模型而感到不知所措？一家名为 Krea 的初创公司正致力于解决这个问题</li><li><a href="https://nixiesearch.substack.com/p/benchmarking-api-latency-of-embedding">Embedding 提供商的 API 延迟基准测试（以及为什么你应该始终缓存你的 Embedding）</a>：我们测量了四家主要 Embedding 提供商（OpenAI、Cohere、Google 和 Jina）的 API 延迟。我们发现，如果性能对你很重要，API 集成的便利性可能会带来代价。</li><li><a href="https://irrationalanalysis.substack.com/p/tenstorrent-and-the-state-of-ai-hardware">Tenstorrent 与 AI 硬件初创公司的现状</a>：半定制芯片是比 Nvidia 更大的问题。</li><li><a href="https://www.llama.com/llama-downloads/">下载 Llama</a>：申请访问 Llama。</li><li><a href="https://bsky.app/profile/ramon-astudillo.bsky.social/post/3lm3skzcfxk2i">Ramon Astudillo (@ramon-astudillo.bsky.social)</a>：我想这张表之前漏掉了 [包含引用帖子或其他嵌入内容]</li><li><a href="https://hoxhunt.com/blog/ai-powered-phishing-vs-humans">2025 年 AI 驱动的网络钓鱼表现优于精英网络罪犯 - Hoxhunt</a>：Hoxhunt 的研究证明，AI Agent 在网络钓鱼方面的表现可以超越精英红队。生成式 AI 在网络安全中既可用于行善，也可用于作恶。我们可以使用 AI 鱼叉式网络钓鱼 Agent 进行防御。</li><li><a href="https://www.adaptivesecurity.com/">Adaptive Security</a>：Adaptive 的下一代安全培训和模拟保护企业免受 Deepfake、生成式 AI 网络钓鱼、短信攻击、语音钓鱼以及更多新兴威胁的侵害。</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/meta-llama">Meta Llama | OpenRouter</a>：浏览来自 Meta Llama 的模型</li><li><a href="https://news.ycombinator.com/item?id=43595775">下面是概览，因为页面似乎运行不畅 Llama 4 模型... | Hacker News</a>：未找到描述</li><li><a href="https://www.llama.com/llama4-reasoning-is-coming/">Llama 4 Reasoning</a>：即将推出</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/Isvg17X5O9">Reddit - 互联网的核心</a>：未找到描述</li>

</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1357952602213585007)** (1 messages): 

> `Claude Plays Pokemon Hackathon` 


- **Claude Plays Pokemon Hackathon**: 一位用户感谢了另一位用户协助举办了 [YouTube 上的](https://youtu.be/zBPc6Ims1Bc) **Claude Plays Pokemon Hackathon**。
- **Hackathon 的 YouTube 直播**: **Claude Plays Pokemon Hackathon** 已录制并在 [YouTube](https://youtu.be/zBPc6Ims1Bc) 上进行了直播。


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1357807122246664398)** (255 messages🔥🔥): 

> `LLM Codegen Workflow, AI Code Editors, Cursor vs Windsurf, Context Management in AI Editors, Model Hot-Swapping` 


- **Harper 的 LLM Codegen 工作流曝光**: Harper 的博文 ([My LLM Codegen Workflow ATM](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)) 详细介绍了一个通过 **离散循环中的 LLM codegen** 进行 Spec 头脑风暴、规划和执行的过程。
   - 该过程结合了与 [Nikete](https://www.nikete.com/)、[Kanno](https://nocruft.com/)、[Obra](https://fsck.com/)、[Kris](https://github.com/KristopherKubicki) 和 [Erik](https://thinks.lol/) 等朋友交流后的调整。
- **AI 代码编辑器推荐**: 对于 AI 代码编辑器的新手，**Cursor** 是最常推荐的起点，特别是对于从 VSCode 迁移过来的用户，**Windsurf** 和 **Cline** 也是不错的选择。
   - 使用 **nvim 或 emacs** 的资深开发者应坚持使用当前的编辑器和 AI 插件，而想要尝试新型模态编辑器的用户可以尝试 **Zed**。
- **Cursor 与 Windsurf 的对比**: 成员们在 **Cursor 和 Windsurf** 之间反复权衡，指出了各自的优缺点。
   - Cursor 易于上手，具有出色的 Tab 补全功能，而人们正在期待 Cursor 中新的 **Token 计数和上下文窗口详情** 功能 ([推文](https://x.com/ryolu_/status/1907589821280956648))。
- **Cursor 中的上下文管理问题**: 成员们反映了 Cursor 糟糕的上下文管理问题，缺乏对编辑器如何处理当前上下文的可见性。
   - 这可能归结为 *技术水平问题 (skill issue)*，用户未能与工具达成良好的磨合。
- **One-Shot Codegen 否则免谈**: 频道中的许多人表达了对 **One-Shot Codegen** 的渴望，即一次性生成整个程序。
   - 如果做不到这一点，完善文档记录并再次尝试可能是次优选择；如果仍然失败，则有必要对用户进行培训。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/cgarciae88/status/1907457306947702925">来自 Cristian Garcia (@cgarciae88) 的推文</a>：天哪……我告诉 Gemini 2.5 Pro 它错了，它没有惊慌失措地顺着我并产生幻觉，而是解释了为什么错的人是我。</li><li><a href="https://fxtwitter.com/ryolu_/status/1907589821280956648">来自 Ryo Lu (@ryolu_) 的推文</a>：这是给专业人士准备的：正在开发一种更简单的方法来填充 @cursor_ai 中的最大上下文（MAX context），并准确显示使用了多少个 tokens。欢迎提供反馈和想法 🙏</li><li><a href="https://x.com/ryolu_/status/1907589821280956648">来自 Ryo Lu (@ryolu_) 的推文</a>：这是给专业人士准备的：正在开发一种更简单的方法来填充 @cursor_ai 中的最大上下文（MAX context），并准确显示使用了多少个 tokens。欢迎提供反馈和想法 🙏</li><li><a href="https://x.com/cgarciae88/status/1907457306947702925">来自 Cristian Garcia (@cgarciae88) 的推文</a>：天哪……我告诉 Gemini 2.5 Pro 它错了，它没有惊慌失措地顺着我并产生幻觉，而是解释了为什么错的人是我。</li><li><a href="https://www.npmjs.com/package/@johnlindquist/file-forge">@johnlindquist/file-forge</a>：File Forge 是一个强大的 CLI 工具，用于对代码库进行深度分析，生成 Markdown 报告以供 AI 推理模型使用。最新版本：2.13.6，最后发布于 9 小时前。开始使用 @johnlindqu...</li><li><a href="https://github.com/yamadash">Yamadash - 概览</a>：GitHub 是 Yamadash 构建软件的地方。</li><li><a href="https://github.com/bodo-run/yek">GitHub - bodo-run/yek：一个基于 Rust 的快速工具，用于将仓库或目录中的文本文件序列化，以便 LLM 使用</a>：一个基于 Rust 的快速工具，用于将仓库或目录中的文本文件序列化，以便 LLM 使用 - bodo-run/yek</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个代码库打包成一个对 AI 友好的单一文件。非常适合需要将代码库提供给大语言模型（LLMs）或其他 AI 工具（如 Claude、ChatGPT、DeepSeek、Perplexity、Gemini、Gemma、Llama、Grok 等）的情况。</a>：📦 Repomix（原名 Repopack）是一款强大的工具，可将整个代码库打包成一个对 AI 友好的单一文件。非常适合需要将代码库提供给大语言模型（LLMs）或.....</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">我目前的 LLM 代码生成工作流</a>：详细介绍了我目前使用 LLMs 构建软件的工作流，从头脑风暴到规划和执行。</li><li><a href="https://github.com/formal-land/coq-of-rust?tab=readme-ov-file">GitHub - formal-land/coq-of-rust：Rust 的形式化验证工具：检查程序 100% 的执行情况 🦀，以构建超安全的应用程序！ ✈️ 🚀 ⚕️ 🏦</a>：Rust 的形式化验证工具：检查程序 100% 的执行情况 🦀，以构建超安全的应用程序！ ✈️ 🚀 ⚕️ 🏦 - formal-land/coq-of-rust
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1357829245635657873)** (308 条消息🔥🔥): 

> `开源 Cursor 替代方案、Prompt 注入 / 越狱策略、Llama 4 发布与性能、通过 Neural Graffiti 实现神经可塑性`

- **类 Cursor 应用备受追捧**：成员们正在寻找 **Cursor** 应用的 [开源替代方案](https://continue.dev)，特别关注代码块的接受/舍弃建议功能是如何运作的。
   - 一位成员指出，一旦你点击接受，**Cursor** 会使用一个不同的模型来“应用（apply）”代码。
- **释放 Prompt Injection 攻击**：一位成员从 *pentest* 角度询问了如何绕过 prompt guards、检测器以及 **NeMo guard rails**，并分享了一个 prompt 过滤器训练器链接 ([gandalf.lakera.ai/baseline](https://gandalf.lakera.ai/baseline))。
   - 他们还链接了一个 [Broken LLM Integration App](https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App)，该应用使用了 **UUID 标签和严格边界**。
- **Llama 4 亮相，展现多模态实力**：**Meta** 推出了 **Llama 4** 系列，包括 **Llama 4 Scout**（17B 激活参数，16 个专家，10M+ 上下文）和 **Llama 4 Maverick**（17B 激活参数，128 个专家，1M+ 上下文），以及 **Llama 4 Behemoth** 的预览，并展示了用于无限上下文的 iRoPE 架构 ([博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/))。
   - 一些成员对基准测试方法论、**Llama 4 Scout** 的真实世界编程能力和性能表示怀疑。
- **Neural Graffiti 为 LLM 提供实时调制**：一位成员介绍了 “Neural Graffiti”，这是一种通过拼接一个新的记忆召回神经元层来赋予预训练 LLM 神经可塑性的技术，在生成时重塑 token 预测，并在 [GitHub](https://github.com/babycommando/neuralgraffiti) 上分享了代码和演示。
   - 这种实时调制获取一个融合的记忆向量（来自先前的 prompt），通过一个循环层（Spray Layer）进行演化，并在生成时将其注入到模型的输出逻辑中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/deedydas/status/1908749257084944847">来自 Deedy (@deedydas) 的推文</a>：Llama 4 似乎在编程方面表现不佳。Scout (109B) 和 Maverick (402B) 在测试编程能力的 Kscores 基准测试中表现不如 4o、Gemini Flash、Grok 3、DeepSeek V3 和 Sonnet 3.5/7...</li><li><a href="https://x.com/ficlive/status/1908911992686931989?t=N1BGmubwXQQ-ZYLSKfmXqw&s=19">来自 Fiction.live (@ficlive) 的推文</a>：使用 Llama 4 更新了长上下文（Long context）基准测试</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1908595680828154198">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍我们的第一批 Llama 4 模型！我们一直在努力对 Llama 系列进行全面的重新设计。我很高兴今天能与世界分享它，并标志着另一个重要的里程碑...</li><li><a href="https://x.com/nrehiew_/status/1908617547236208854?s=46">来自 wh (@nrehiew_) 的推文</a>：在局部注意力块中，Llama 4 使用了这种 Chunked Attention，而不是滑动窗口。这非常有趣/奇怪：- token 索引 8191 和 8192 在局部注意力中无法交互 - 唯一的方法...</li><li><a href="https://x.com/JeffDean/status/1908608454216028222">来自 Jeff Dean (@JeffDean) 的推文</a>：@jeremyphoward 为什么不能在消费级 GPU 上运行它们？</li><li><a href="https://x.com/astonzhangaz/status/1908595612372885832?s=46">来自 Aston Zhang (@astonzhangAZ) 的推文</a>：我们 Llama 4 行业领先的 10M+ 多模态上下文长度（20+ 小时视频）是一段疯狂的历程。我一直在研究的 iRoPE 架构对实现长期无限上下文的目标有所帮助...</li><li><a href="https://x.com/agarwl_/status/1909292968139255849?t=RpSDb5rQDhI1cdf1ZXqZxw&s=19)">来自 Rishabh Agarwal (@agarwl_) 的推文</a>：今天加入了 @AIatMeta 的 Llama 团队，负责 RL 和推理工作。引用 AI at Meta (@AIatMeta)：今天是一个原生多模态 AI 创新新时代的开始。今天，我们推出了首个...</li><li><a href="https://x.com/kalomaze/status/1909267256564920611">来自 kalomaze (@kalomaze) 的推文</a>：turboderp 不喜欢炒作他的工作，所以让我来代劳。——这改变了一切 🤯 EXLLAMA 开发者 "turboderp" 发布了 EXLLAMA 3，具有新颖、最先进的本地模型量化...</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF">NousResearch/Hermes-3-Llama-3.1-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://gandalf.lakera.ai/baseline">Gandalf | Lakera – 测试你的提示词技巧，让 Gandalf 泄露秘密信息。</a>：诱导 Gandalf 泄露信息，亲身体验大语言模型（LLM）的局限性。</li><li><a href="https://www.arxiv.org/abs/2504.01990">基础智能体（Foundation Agents）的进展与挑战：从类脑智能到进化、协作和安全系统</a>：大语言模型（LLM）的出现催化了人工智能的变革性转变，为能够进行复杂推理、稳健表现的高级智能体铺平了道路...</li><li><a href="https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md">llmbenchmark/raytracer/Readme.md at master · cpldcpu/llmbenchmark</a>：各种 LLM 基准测试。通过在 GitHub 上创建账号来为 cpldcpu/llmbenchmark 的开发做出贡献。</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述</li><li><a href="https://www.llama.com/llama4-reasoning-is-coming/">Llama 4 推理能力</a>：即将推出</li><li><a href="https://github.com/meta-llama/llama-models/tree/main/models/llama4">llama-models/models/llama4 at main · meta-llama/llama-models</a>：旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账号来为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://www.trae.ai/">Trae - 使用 Trae 更快交付</a>：Trae 是一款自适应 AI IDE，它改变了你的工作方式，通过协作助你提升效率。</li><li><a href="https://github.com/13o-bbr-bbq/Broken_LLM_Integration_App">GitHub - 13o-bbr-bbq/Broken_LLM_Integration_App: 这是一个包含漏洞的 LLM 集成应用；请使用它来验证 LLM 集成应用的漏洞。</a>：这是一个包含漏洞的 LLM 集成应用；请使用它来验证 LLM 集成应用的漏洞。 - 13o-bbr-bbq/Broken_LLM_Integration_App</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12791">llama : 由 ngxson 支持 Llama 4 纯文本模式 · Pull Request #12791 · ggml-org/llama.cpp</a>：解决 #12774。此 PR 针对 Llama-4-Scout-17B-16E-Instruct。我（目前还）没有足够强大的系统来处理更大的模型。但是孩子，你缺乏 GPU 资源，你怎么能测试一个...</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct">meta-llama/Llama-4-Maverick-17B-128E-Instruct · Hugging Face</a>：未找到描述</li><li><a href="ht">

<li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E">meta-llama/Llama-4-Scout-17B-16E · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E">meta-llama/Llama-4-Maverick-17B-128E · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1357947681011667077)** (27 messages🔥): 

> `Claude Think Tool, Local LLM for 300 Pages of Text, Nous Capybara 34B Model, DeepHermes, BatchNorm and LayerNorm Implementations` 


- **Claude Think Tool：聪明的头脑风暴工具**：[Claude Think Tool](https://www.anthropic.com/engineering/claude-think-tool) 是一种将关键任务从小型本地模型卸载到大型模型的配置。
   - 它有助于创建多个思维线程，每个线程都将注意力集中在具有明确范围的特定领域和问题上，从大脑的角度来看，其功能类似于一个 **multi-agent 系统**。
- **思考适用于 300 页文本摄取的完美本地 LLM**：一位成员询问如何在拥有 **12GB** GPU 和 **32GB** 普通内存的情况下，运行一个 **40B** 或更小的本地 LLM，并能够理解约 **300 页** 的纯文本。
   - 建议包括 **DeepHermes**、**Cohere Command R 7B** 和 **Qwen 7B 1M**，并警告 CPU 推理对于如此庞大的文档可能并不可行。
- **Nous Capybara 34B：上下文巨兽**：[Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B) 是在 **Yi-34B** 模型基础上，使用 Capybara 数据集以 **200K** 上下文长度训练了 **3 epochs**。
   - 它利用了一种名为 **Amplify-instruct** 的新型数据合成技术，结合了用于 Airoboros、Evol-Instruct、Orca、Vicuna 等 SOTA 模型的顶级现有数据合成技术和分布。
- **BatchNorm 反向传播：数值涅槃**：一位成员分享了使用 NumPy 实现 **BatchNorm** 的原始代码，强调反向传播是最令人畏惧的部分，因为需要根据多元链式法则计算预归一化输入的梯度，详情见[此处](https://cdn.discordapp.com/attachments/1154120232051408927/1358871065119686746/image.png)。
   - 随后他们实现了 **LayerNorm**，[强调了关键区别](https://cdn.discordapp.com/attachments/1154120232051408927/1358919204270641342/image.png)在于统计数据是按样本（per sample）而不是按批次（per batch）计算的。



**提到的链接**：<a href="https://huggingface.co/NousResearch/Nous-Capybara-34B">NousResearch/Nous-Capybara-34B · Hugging Face</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 messages): 

> `Reinforcement Learning for LLMs, Reward Modeling Improvements, Self-Principled Critique Tuning` 


- **Deepseek 发布强化学习论文**：Deepseek 发布了一篇关于 **Reinforcement Learning (RL)** 在大语言模型 (**LLMs**) 大规模训练后阶段被广泛采用的新论文；论文可以在[此处](https://arxiv.org/abs/2504.02495)找到。
   - 该论文研究了如何通过更多的推理计算来改进通用查询的奖励建模 (**RM**)，即 *通用 RM 的推理时间可扩展性*，并进一步探讨如何通过适当的学习方法提高性能-计算缩放的有效性。
- **提出 Self-Principled Critique Tuning**：Deepseek 采用逐点生成式奖励建模 (**GRM**)，以实现不同输入类型的灵活性和推理时间扩展的潜力。
   - 论文提出了 **Self-Principled Critique Tuning (SPCT)** 以促进可扩展性。



**提到的链接**：<a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement learning (RL) 已在大语言模型 (LLMs) 的大规模训练后阶段被广泛采用。最近，RL 对 LLMs 推理能力的激励表明 $...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1357794753265271069)** (9 条消息🔥): 

> `Claude Squad, Heterogeneous Recursive Planning, Panthalia Decentralized Compute, TextPulse Library` 


- **Claude Squad 管理多个 Agent**: [Claude Squad](https://github.com/smtg-ai/claude-squad) 是一个免费且开源的 **Claude Code & Aider 任务**管理器，可以在一个地方监督多个具有隔离 git 工作区的 Agent。
   - 它允许用户并行运行 **10 个 Claude Code**。
- **用于创意 AI 的 Heterogeneous Recursive Planning**: 一种名为 Heterogeneous Recursive Planning 的新方法使 **AI** 能够像专家一样编写创意故事和深刻的深度研究报告 ([论文](http://arxiv.org/abs/2503.08275), [演示](http://writehere.site))。
   - 它利用自适应子目标和动态执行，允许 Agent 在流程中动态重新规划并交织检索、推理和创作，该方法基于[之前的研究](https://www.google.com/url?sa=D&q=https://people.idsia.ch/~juergen/recursiveplanning.html)。
- **Panthalia 验证低成本分布式计算**: [Panthalia](https://x.com/panthaliaxyz/status/1909342585505669228) 是一个使用去中心化计算原语在点对点计算上安全、轻松地训练 **ML 模型**的平台，目前采用候补名单制。
   - 该平台使用了一种深受 [Nous DeMo 论文](https://docs.panthalia.com/gradient-compression-algorithm)和[相关代码库](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py)启发的压缩算法。
- **用于文本处理的 TextPulse 库**: 一位成员分享了他们用于文本处理的库 [TextPulse](https://github.com/jfinst1/TextPulse)，并正在寻求反馈。
   - 目前，他们转售低成本供应商的资源，目标是相同的可中断价格（**H100 约为 $0.60/小时，4090 约为 $0.13/小时**）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/moofeez/status/1907893901077196861?s=46">来自 mufeez (@moofeez) 的推文</a>: 既然可以并行运行十个，为什么要满足于一个 Claude Code？我们构建了 Claude Squad —— 一个 Claude Code & Aider 任务管理器：• 在一个地方监督多个 Agent • 隔离的 git 工作区 免费 + ...</li><li><a href="https://x.com/SchmidhuberAI/status/1908172744409403793">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>: 如果 AI 能像专家一样编写创意故事和深刻的 #DeepResearch 报告会怎样？我们的 Heterogeneous Recursive Planning [1] 通过自适应子目标 [2] 和动态执行实现了这一点。Agent 能够动态地...</li><li><a href="https://x.com/panthaliaxyz/status/1909342585505669228">来自 Panthalia (@panthaliaxyz) 的推文</a>: Panthalia: 去中心化计算原语。在点对点计算上安全、轻松地训练 ML 模型的平台。候补名单现已开放。</li><li><a href="https://docs.panthalia.com/gradient-compression-algorithm">Panthalia 梯度压缩算法 | Panthalia</a>: 本文档详细描述了 Panthalia 中使用的基于 DCT 的梯度压缩算法。该算法旨在高效压缩从节点发送到中心...</li><li><a href="https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py">panthalia-worker/spl/util/demo.py at main · ritser-labs/panthalia-worker</a>: 通过在 GitHub 上创建账号为 ritser-labs/panthalia-worker 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1357794815206752408)** (2 条消息): 

> `Deepseek, Reinforcement Learning, Large Language Models, Reward Modeling, Self-Principled Critique Tuning` 


- **Deepseek 关于 LLM RL 的新论文**: Deepseek 发布了一篇新论文，可在 [arXiv](https://arxiv.org/abs/2504.02495) 上查阅，内容涉及在大规模 **Large Language Models (LLMs)** 的训练后阶段采用 **Reinforcement Learning (RL)**。
   - 该论文研究了如何通过增加通用查询的推理计算来改进 **Reward Modeling (RM)**，以及通过适当的学习方法实现性能-计算缩放的有效性，并提出了 **Self-Principled Critique Tuning (SPCT)**。
- **SPCT 改进 Reward Modeling**: 论文介绍了 **Self-Principled Critique Tuning (SPCT)**，这是一种增强 LLM Reward Modeling 中性能-计算缩放有效性的方法。
   - 该方法旨在通过改进通用查询（超越可验证问题或人工规则）的 Reward Model 推理计算来促进可扩展性。



**提到的链接**: <a href="https://arxiv.org/abs/2504.02495">Inference-Time Scaling for Generalist Reward Modeling</a>: Reinforcement Learning (RL) 已被广泛应用于大规模 Large Language Models (LLMs) 的训练后阶段。最近，RL 对 LLM 推理能力的激励表明 $...

  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1358810800252391476)** (6 messages): 

> `推理基准测试，Open Reasoning Tasks` 


- **研究员转型至 LLM 领域**：一位从事逻辑与推理研究的研究员正考虑进入 LLM 领域，并希望为推理分类和基准测试做出贡献。
   - 一位成员建议查看 [reasoning tasks repo](https://github.com/NousResearch/Open-Reasoning-Tasks)。
- **关于 Open Reasoning Tasks 的讨论**：一位成员正在探索推理任务列表以对某个 LLM 进行基准测试，并询问其背后的分类法 (Taxonomy)、背景以及相关文献。
   - 他们特别询问了该分类法的制定者及其历史。



**提到的链接**：<a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks: A comprehensive repository of reasoning tasks for LLMs (and beyond)</a>：一个面向 LLM（及更多领域）的全面推理任务仓库 - NousResearch/Open-Reasoning-Tasks

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1357795938168602715)** (293 messages🔥🔥): 

> `MCP 治理 SDK，MCP 协议 2025 修订版，MCP 桌面工作流集成，初始化前 Ping MCP 服务器，适用于 Microsoft Loop 的 MCP 服务器` 


- **使用 MCP 治理 SDK 进行 Auth0 令牌验证**：一份指南专注于使用治理 SDK 进行服务端实现，以验证令牌（例如来自 **Auth0** 的令牌），并在 MCP 操作上强制执行用户角色和权限，从而决定对工具或资源的访问权限。
   - 该指南从客户端发送令牌后的步骤开始，详细说明了服务器如何验证令牌并获取用户角色，利用 SDK 的 RBAC 系统来强制执行权限。
- **MCP 的可流式 HTTP 传输**：[Model Context Protocol (MCP) 规范](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http) 使用 **JSON-RPC** 对消息进行编码，强制要求 UTF-8 编码，并定义了两种传输机制：stdio 和可流式 HTTP。
   - 客户端*应当*支持 stdio，但如规范所述，自定义传输也是可能的，规范中还包括 stdio 消息的换行符分隔等要求。
- **Llama 4 发布，但仍不了解 MCP**：**Llama 4** 已发布，拥有 [17B 参数](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，性能超越了 deepseekv3，但尽管其能力令人印象深刻，它仍然不知道 MCP 是什么。
   - 根据公告，这是一个 **17B MoE**，总参数量为 **109B**。
- **MCP 工具安装应标准化**：成员们讨论了对 **MCP server** 安装进行更多标准化的必要性，类似于 *scoop* 或 VS Code 扩展，以提高非技术用户的易用性。
   - 讨论强调了当前流程中的摩擦，涉及命令行参数、环境变量以及不同的安装方法（Python、Node.js、Docker），并建议使其像 *python-mcp install web-search* 一样简单。
- **圣战？支持 OAuth 的 API MCP 是关键**：成员们就 MCP 的安全性展开了辩论，一些人认为需要一个带有监管机制的*应用商店*来检查被黑的服务器和支持 OAuth 的 API，而另一些人则声称这已经可以实现。
   - 一项提议是让像 PayPal 这样的提供商托管他们自己的支持 OAuth 的 API，而不需要安装外部服务器。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@studymyvisualsco/unlock-effortless-ai-automation-best-way-to-self-host-n8n-is-railway-integrate-firecrawl-mcp-7964019c6c28">开启轻松的 AI 自动化：自托管 n8n 的最佳方式是 Railway 并集成 Firecrawl MCP…</a>: 无需编写任何代码即可释放 AI 驱动的 Web 自动化力量！</li><li><a href="https://x.com/MCP_Community">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://gitmcp.io/docs">GitMCP</a>: 为任何 GitHub 项目即时创建 MCP 服务器</li><li><a href="https://developers.cloudflare.com/agents/guides/remote-mcp-server/">构建远程 MCP 服务器 · Cloudflare Agents 文档</a>: 本指南将引导你如何将示例 MCP 服务器部署到你的 Cloudflare 账户。随后你将根据需求自定义此示例。</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#streamable-http>?">传输协议</a>: ℹ️ 协议修订版本：2025-03-26。MCP 使用 JSON-RPC 对消息进行编码。JSON-RPC 消息必须采用 UTF-8 编码。该协议目前定义了两种标准传输机制...</li><li><a href="https://glama.ai/mcp/servers/@611711Dark/mcp_calculate_server">MCP Calculate Server</a>: 一个数学计算服务，使用户能够通过 MCP 协议执行符号计算，包括基础算术、代数、微积分、方程求解和矩阵运算。</li><li><a href="https://learn.microsoft.com/en-us/microsoft-copilot-studio/agent-extend-action-mcp">使用 Model Context Protocol (预览版) 扩展你的 Agent - Microsoft Copilot Studio</a>: 通过连接到 Model Context Protocol (MCP) 服务器的操作来扩展 Agent 的功能。</li><li><a href="https://www.reddit.com/r/mcp/comments/1jtgug1/dis">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://github.com/jaw9c/awesome-remote-mcp-servers">GitHub - jaw9c/awesome-remote-mcp-servers: 一个精选的、有见解的高质量远程 Model Context Protocol (MCP) 服务器列表。</a>: 一个精选的、有见解的高质量远程 Model Context Protocol (MCP) 服务器列表。 - GitHub - jaw9c/awesome-remote-mcp-servers...</li><li><a href="https://github.com/EnactProtocol">Enact Protocol</a>: Enact Protocol 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/modelcontextprotocol/specification/blob/main/docs/specification/2025-03-26/basic/lifecycle.md">specification/docs/specification/2025-03-26/basic/lifecycle.md at main · modelcontextprotocol/specification</a>: Model Context Protocol 的规范。通过在 GitHub 上创建账号为 modelcontextprotocol/specification 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/mcp/comments/1jtgug1/discussion_unified_tool_registry_for_ai_agents/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://aistudio.google.com/app/prompts/new_chat">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/EnactProtocol/enact-mcp/blob/0e155b5d52c340b14de0a3f7804aec0c2456ff36/src/index.ts#L93">enact-mcp/src/index.ts at 0e155b5d52c340b14de0a3f7804aec0c2456ff36 · EnactProtocol/enact-mcp</a>: 用于 Enact Protocol 的 MCP 服务器。通过在 GitHub 上创建账号为 EnactProtocol/enact-mcp 的开发做出贡献。</li><li><a href="https://github.com/semgrep/mcp#hosted-server">GitHub - semgrep/mcp: 一个用于使用 Semgrep 扫描代码安全漏洞的 MCP 服务器。</a>: 一个用于使用 Semgrep 扫描代码安全漏洞的 MCP 服务器。 - semgrep/mcp</li><li><a href="https://glama.ai/mcp/reference">MCP API 参考</a>: Glama Gateway 的 API 参考</li><li><a href="https://www.pulsemcp.com/api">REST API 文档 | PulseMCP</a>: 以编程方式访问每日更新的所有 MCP 服务器元数据的 JSON，这些数据经过抓取以保证全面，并经过过滤以确保实用。</li><li><a href="https://www.reddit.com/r/mcp/comments/1jrq4o8/how_do_we_improve_the_distribution_of_mcp_servers/?rdt=35027">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1357830147528593513)** (23 条消息🔥): 

> `MCP-k8s Docker 镜像, 支持 MCP 的 chat.md, 用于远程 MCP 服务器的 Cloudflare, WhatsMCP OAuth 支持, Semgrep MCP 重写` 


- ****MCP-k8s** Docker 镜像发布**：**mcp-k8s server** 的首个可用 [**Docker 镜像**](https://hub.docker.com/r/mcpk8s/server) 已发布，且发布流水线已完全在 CI 上运行。
   - 这些镜像是**多架构 (multiarch)** 的，因此可以在搭载 **ARM** 的 **Mac** 上原生运行（无需 **Rosetta**），也可以在 **Raspberry Pi** 上运行。
- ****Chat.md**：支持 MCP 的全可编辑对话界面**：一个在任何 **LLM** 上都支持 **MCP** 的全可编辑对话界面已发布。该项目采用 **MIT 协议**开源，并通过其 **VS Code 扩展** ([chat.md](https://github.com/rusiaaman/chat.md)) 将 Markdown 文件转换为可编辑的 AI 对话。
   - 显著特性包括编辑历史消息、**与 LLM 无关的 MCP 支持**、通过 **shift+enter** 实现流式响应以及工具调用检测。
- **Cloudflare 支持远程 MCP 服务器**：现在可以[在 **Cloudflare** 上构建和部署远程 **MCP 服务器**](https://developers.cloudflare.com/agents/guides/remote-mcp-server/)，并增加了通过 **workers-oauth-provider** 实现的 **OAuth** 支持以及内置的 **McpAgent** 类。
   - 通过处理授权和其他复杂环节，这简化了构建远程 **MCP 服务器** 的过程。
- **WhatsApp MCP 客户端上线**：一位用户构建了 WhatsApp MCP 并让 **Claude** 处理所有 WhatsApp 消息，在大约 **50 秒**内回复了 8 个人。
   - 该机器人立即检测到了正确的语言（**英语 / 匈牙利语**），使用了完整的对话上下文，并发送了合适的回复，包括*给妻子的 ❤️ 以及给领事的正式语气*。
- **Semgrep MCP 服务器重写**：[Semgrep MCP server](https://github.com/semgrep/mcp) 是一个用于扫描代码安全漏洞的**开源**工具，现已完全重写，并发布了展示其在 **Cursor** 和 **Claude** 中使用的演示视频。
   - 它使用 **SSE** (Server-Sent Events) 进行通信，尽管 Python SDK 可能尚未完全支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/vargastartup/status/1907904839448715657">来自 Alex Varga (@vargastartup) 的推文</a>：我让 Claude 处理我所有的 WhatsApp 消息。只需一个提示词。就是这样。1. 它在大约 50 秒内回复了 8 个人。2. 立即检测到正确的语言（英语 🇺🇸 / 匈牙利语 🇭🇺）。3. 使用了完整的...</li><li><a href="https://wassist.app/mcp">WhatsApp MCP 客户端 | 连接你的 AI 技术栈</a>：通过 WhatsApp 连接你的 MCP 服务器以驱动你的 AI 技术栈。安全、私密且易于使用。</li><li><a href="https://blog.cloudflare.com/remote-model-context-protocol-servers-mcp/">在 Cloudflare 上构建和部署远程模型上下文协议 (MCP) 服务器</a>：你现在可以在 Cloudflare 上构建和部署远程 MCP 服务器，我们会为你处理构建远程 MCP 服务器的难点。与你之前可能使用的本地 MCP 服务器不同，远程 MCP 服...</li><li><a href="https://github.com/semgrep/mcp">GitHub - semgrep/mcp: 用于使用 Semgrep 扫描代码安全漏洞的 MCP 服务器。</a>：一个用于使用 Semgrep 扫描代码安全漏洞的 MCP 服务器。 - semgrep/mcp</li><li><a href="https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df">Semgrep MCP 演示</a>：使用 Loom 快速录制屏幕和摄像头视频。清晰轻松地解释任何事情——并跳过会议。混合办公场所的必备工具。</li><li><a href="https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411">Claude Desktop 使用 Semgrep MCP 资源</a>：使用 Loom 快速录制屏幕和摄像头视频。清晰轻松地解释任何事情——并跳过会议。混合办公场所的必备工具。</li><li><a href="https://github.com/SDCalvo/MCP-to-Langchain-addapter">GitHub - SDCalvo/MCP-to-Langchain-addapter: 将 MCP 服务器工具转换为 LangChain 可用工具的适配器</a>：将 MCP 服务器工具转换为 LangChain 可用工具的适配器 - SDCalvo/MCP-to-Langchain-addapter</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/416">由 josh-newman 提交的流式 HTTP 客户端传输 · Pull Request #416 · modelcontextprotocol/python-sdk</a>：规范版本 2025-03-26 新传输协议的客户端实现。动力与背景：2025-03-26 规范引入了一种新的 HTTP 传输机制（并可回退到前一个版本）。我做了...</li><li><a href="https://github.com/rusiaaman/chat.md">GitHub - rusiaaman/chat.md</a>：通过在 GitHub 上创建账号来为 rusiaaman/chat.md 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1357847563901993020)** (39 条消息🔥): 

> `使用 lm-evaluation-harness 进行 RAG 评估，the_alt_man 的 RoR-Bench 论文，Llama 4 发布，使用贝叶斯更新对齐 AGI` 


- **使用 LLM Harness 进行 RAG 评估？**: 一名成员建议将 **RAG 输出包装为补全任务 (completion tasks)**，并在本地使用 **llm-harness** 配合自定义提示词 + 响应文件进行评估。
   - 另一名成员承认*完全不知道那些是什么*。
- **LLM 表现出背诵行为？**: 一名成员分享了 [RoR-Bench 论文](https://arxiv.org/abs/2504.00509)的链接，该论文提出了一个新颖的多模态基准测试，用于检测 LLM 的背诵行为，发现顶尖模型在更改条件中的一个短语后，可能会遭受 **60% 的性能损失**。
   - 该成员对这些论文表示怀疑，因为他们发现某些在推理任务中被评估为 0% 的模型，实际上可以通过 one-shot 完成任务。
- **Llama 4 发布**: 分享了 **Llama 4 发布**的链接 ([https://www.llama.com/llama4/](https://www.llama.com/llama4/))，展示了同类中最智能的多模态 OSS 模型，其中 Llama4 Maverick > Gemma3 且 Llama4 Maverick > DeepSeek V3。
   - 另一名成员注意到了其训练过程、架构以及推理时的温度缩放 (temperature scaling)。
- **使用道德权重对齐 AGI**: 一名成员分享了一份 [Google Doc](https://docs.google.com/document/d/1j11OUXWtS6yLAzXsbSo4lrzoqtiZ5RM0ykilLEvpYwM/edit)，关于使用其**道德权重**的**贝叶斯更新 (Bayesian Updating)** 和**意识建模**来对齐 **AGI**。
   - 另一名成员分享了一个 Arweave 链接，讨论了 AI 在保护人类意识方面的作用。 ([https://arweave.net/q6CszfPrxFZfm-BiVsvtiOXWuDkcYo8Pf9viDqv-Nhg](https://arweave.net/q6CszfPrxFZfm-BiVsvtiOXWuDkcYo8Pf9viDqv-Nhg))


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2504.00509">Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems?</a>: 近年来 LLM 基准测试难度从小学水平到前沿问题的迅速升级，为研究人员创造了一个奇迹，即我们距离超越...仅一步之遥</li><li><a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2305.19466">The Impact of Positional Encoding on Length Generalization in Transformers</a>: 长度泛化，即从较小的训练上下文尺寸泛化到较大尺寸的能力，是 Transformer 语言模型开发中的一个关键挑战。位置编码...</li><li><a href="https://docs.google.com/document/d/1j11OUXWtS6yLAzXsbSo4lrzoqtiZ5RM0ykilLEvpYwM/edit">ALBUM-WMC: Aligning AGI Using Bayesian Updating of its Moral Weights &amp; Modelling Consciousness</a>: (欢迎在此留言“我来过！”) ALBUM-WMC：使用道德权重的贝叶斯更新与意识建模来对齐 AGI。本文档概述了一系列相关的想法...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1357808468794277966)** (204 条消息🔥🔥): 

> `Mixture of Experts, Large Language Models, 无梯度学习方法, 作为残差连接替代方案的超连接 (Hyper-connections), LLM 中的注意力汇 (Attention Sinks)`

- ****MoE++ 框架实现专家吞吐量提升****：根据[这篇研究论文](https://openreview.net/forum?id=t7P5BUKcYv)，全新的 **MoE++** 框架集成了 **Feed-Forward Network (FFN)** 和零计算专家（**zero expert、copy expert 和 constant expert**）以增强有效性和效率，与原生 **MoE** 模型相比，实现了 **1.1$\sim$2.1 倍的专家前向吞吐量**。
   - **MoE++** 的设计具有*低计算开销*等优势，通过启用动态 Token 参与，区别于原生 MoE 中的均匀混合。
- ****NoProp 提供无梯度学习****：[这篇论文](https://arxiv.org/abs/2503.24322)描述了一种名为 **NoProp** 的新学习方法，它不依赖于前向或反向传播，并从 Diffusion 和 Flow Matching 方法中汲取灵感，学习在每一层独立地对噪声目标进行去噪。
   - 存在一个由 [lucidrains 开发的 GitHub 实现](https://github.com/lucidrains/hyper-connections)，并且还有讨论指出*论文末尾的伪代码显示他们正在使用基于梯度的方法执行实际更新。*
- ****Meta 发布 Llama 4****：Meta 宣布了 **Llama 4** 系列模型，包括 **Llama 4 Scout**，这是一个拥有 **170 亿**参数、**16 个专家**和 **10M Token 上下文窗口**的模型，在其同类产品中表现优于 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1**，详见[这篇博客文章](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。
   - Llama 4 Scout 的 **10M 上下文**是在公开数据、授权数据以及来自 Meta 产品和服务的信息（包括 **Instagram** 和 **Facebook** 的帖子以及人们与 **Meta AI** 的互动）的混合数据集上训练的。
- ****Hyper-Connections 提供残差连接的替代方案****：如[这篇论文](https://arxiv.org/abs/2409.19606)所述，**Hyper-connections** 作为 Residual Connections 的替代方案，解决了梯度消失（gradient vanishing）与表示崩溃（representation collapse）之间的跷跷板效应。
   - 该架构像展开的 Diffusion 模型一样简单，其*奥秘更多在于每一层相对于彼此的独立性*。
- ****LLM 中的 Attention Sinks 防止过度混合****：最近的一篇论文认为，**Attention Sinks**（即 LLM 严重关注序列中的第一个 Token）是一种使 LLM 能够避免过度混合的机制，详见[这篇论文](https://arxiv.org/abs/2504.02732)。
   - 较早的一篇论文 ([https://arxiv.org/abs/2502.00919](https://arxiv.org/abs/2502.00919)) 表明，*Attention Sinks 利用离群特征（outlier features）来：捕获 Token 序列，通过应用共同的扰动为捕获的 Token 打上标签，然后将 Token 释放回残差流（residual stream），打上标签的 Token 最终在那里被检索*。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.19606">Hyper-Connections</a>: 我们提出了 Hyper-connections，这是一种简单且有效的方法，可以作为残差连接 (residual connections) 的替代方案。该方法专门解决了在残差连接中观察到的常见缺点...</li><li><a href="https://arxiv.org/abs/2503.24322">NoProp: Training Neural Networks without Back-propagation or Forward-propagation</a>: 传统的深度学习学习方法需要通过从输出向每个可学习参数反向传播误差信号，来计算每一层的梯度项。考虑到堆叠...</li><li><a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: 我们提出了一种新型神经网络架构，即在超球面上进行表示学习的归一化 Transformer (nGPT)。在 nGPT 中，所有构成 Embeddings、MLP、注意力矩阵的向量...</li><li><a href="https://openreview.net/forum?id=t7P5BUKcYv">MoE++: Accelerating Mixture-of-Experts Methods with...</a>: 在这项工作中，我们旨在同时增强 Mixture-of-Experts (MoE) 方法的有效性和效率。为了实现这一目标，我们提出了 MoE++，一个通用且异构的 MoE 框架...</li><li><a href="https://arxiv.org/abs/2504.02732">Why do LLMs attend to the first token?</a>: 大语言模型 (LLM) 倾向于高度关注序列中的第一个 Token——产生所谓的注意力汇 (attention sink)。许多研究详细探讨了这一现象，并提出了各种方法...</li><li><a href="https://arxiv.org/abs/2503.05453">Soft Policy Optimization: Online Off-Policy RL for Sequence Models</a>: 基于 RL 的语言模型后训练几乎完全使用 PPO 等在线策略 (on-policy) 方法。这些方法无法从训练早期产生的任意序列中学习...</li><li><a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: 未找到描述</li><li><a href="https://x.com/BlinkDL_AI/status/1909280712567787947">Tweet from BlinkDL (@BlinkDL_AI)</a>: https://arxiv.org/abs/2503.24322 我认为 NoProp 方法可能也适用于 LLM 训练，因为每个 LLM 块都在对下一个 Token 分布进行去噪。因此我们可以尝试并行训练所有块...</li><li><a href="https://arxiv.org/abs/2502.00919">Attention Sinks and Outlier Features: A &#39;Catch, Tag, and Release&#39; Mechanism for Embeddings</a>: 大语言模型 (LLM) 的两个显著特征是大范数（离群）特征的存在，以及 Token 倾向于强烈关注极少数 Token。尽管通常具有...</li><li><a href="https://arxiv.org/abs/2504.01990">Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems</a>: 大语言模型 (LLM) 的出现催化了人工智能的变革性转变，为能够进行复杂推理、稳健表现的高级智能 Agent 铺平了道路...</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">no title found</a>: 未找到描述</li><li><a href="https://github.com/lucidrains/hyper-connections">GitHub - lucidrains/hyper-connections: Attempt to make multiple residual streams from Bytedance&#39;s Hyper-Connections paper accessible to the public</a>: 尝试让字节跳动 Hyper-Connections 论文中的多残差流 (multiple residual streams) 可供公众使用 - lucidrains/hyper-connections
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1358672878937178192)** (17 条消息🔥): 

> `神经网络的 Polytope 视角，ReLU 网络几何学，Machine Unlearning Workshop，神经网络的 Origami 视图，深度网络的 Expressivity` 


- **“Polytope 视角”助力神经网络思考**：一位成员分享了一篇[博文](https://addxorrol.blogspot.com/2025/04/some-experiments-to-help-me-understand.html)，讨论了神经网络的几何方法，主张将 **polytope lens** 作为正确的视角，并链接到了之前关于“神经网络的 **origami view**”的[帖子](https://addxorrol.blogspot.com/2024/07/some-experiments-to-help-me-understand.html)。
- **“ReLU 网络区域”揭示原理**：一位成员分享了 [Boris Hanin 的论文](https://arxiv.org/abs/1906.00904)，该论文展示了 **ReLU networks** 的数学特性，特别是研究了它们常数区域（constant regions）的几何结构。
   - 他们强调了论文中的一张图是他们“喜爱这篇论文的主要原因”，并提到了深度网络的 **expressivity** 以及 **activation patterns** 的数量。
- **“超平面和谐”：神经网络的自然细微差别**：一位成员认为神经网络（尤其是 **ReLUs**）具有防止过拟合的隐式偏置（implicit bias），因为它们沿 **hyperplanes** 切割输入空间，这在高维空间中变得更加有效。
   - 他们认为优化器更倾向于高效利用超平面的简单配置，这与受维度灾难（curse of dimensionality）困扰的 **spline bases** 等学习方案形成对比。
- **“Unlearning 的紧迫性”：机器思维管理**：一位成员链接到了 [ICML Machine Unlearning Workshop](https://mugenworkshop.github.io/)，该研讨会专注于从在互联网规模数据集上训练的 **Generative AI models** 中删除敏感数据的挑战。
   - 该研讨会旨在推进稳健、可验证的 **unlearning** 方法，以解决隐私、安全和法律问题，如欧盟的 **GDPR**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1906.00904">Deep ReLU Networks Have Surprisingly Few Activation Patterns</a>: 深度网络的成功部分归功于它们的 expressivity：在相同参数量下，深度网络比浅层网络能逼近更丰富的函数类。在 ReLU 网络中，...</li><li><a href="https://mugenworkshop.github.io/">MUGen @ ICML 2025 - Workshop on Machine Unlearning for Generative AI</a>: 未找到描述</li><li><a href="https://addxorrol.blogspot.com/2025/04/some-experiments-to-help-me-understand.html">ADD / XOR / ROL: Some experiments to help me understand Neural Nets better, post 2 of N</a>: 未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1357837368907923466)** (19 messages🔥): 

> `lm-eval-harness EOS token, Llama 2 vs Llama 3 IFEval Score, Huggingface tokenization` 


- **EOS token 准确率异常出现**：一名成员尝试在 **lm-eval-harness** 的 *social_iqa* 任务中为数据实例添加 EOS token，结果评估准确率下降了 **18 个百分点**。
   - 建议仅针对续写（continuations）而非上下文（context）在 [此处](https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364) 的 `continuation_enc` 中添加 `self.eot_token_id`。
- **IFEval 分数：Llama 2 奇怪的统治地位**：一名成员对比了 **Llama 2** 与 **Llama 3.1** 及 **3.2** 模型，发现 **Llama 2** 拥有更高的 **IFEval Score**，查看 [HF leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?pinned=meta-llama%2FLlama-2-7b-hf_float16_01c7f73d771dfac7d292323805ebc428287df4f9_False%2Cmeta-llama%2FLlama-3.1-8B_float16_d04e592bb4f6aa9cfee91e2e20afa771667e1d4b_False%2Cmeta-llama%2FLlama-3.2-1B_bfloat16_a7c18587d7f473bfea02aa5639aa349403307b54_False%2Cmeta-llama%2FLlama-3.2-3B_bfloat16_95c102307f55fbd6d18ddf28bfbcb537ffdc2806_False) 时，这对于基础模型（base model）来说显得很奇怪。
   - 结果发现，这似乎只是因为该指标不适合基础模型，因为*模型只是简单地继续提问，而不知为何这被判定为正确*。
- **Huggingface Tokenization 故障排除**：成员们讨论了 Huggingface 的 tokenization，以及它是如何在 **HFLM.tok_encode** 中实现的。
   - 有人指出，对于 BOS，你可以将 `add_bos_token` 传递给模型参数（model args）。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?pinned=meta-llama%2FLlama-2-7b-hf_float16_01c7f73d771dfac7d292323805ebc428287df4f9_False%2Cmeta-llama%2FLlama-3.1-8B_float16_d04e592bb4f6aa9cfee91e2e20afa771667e1d4b_False%2Cmeta-llama%2FLlama-3.2-1B_bfloat16_a7c18587d7f473bfea02aa5639aa349403307b54_False%2Cmeta-llama%2FLlama-3.2-3B_bfloat16_95c102307f55fbd6d18ddf28bfbcb537ffdc2806_False">Open LLM Leaderboard - 由 open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/11ac352d5f670fa14bbce00e423cff6ff63ff048/lm_eval/api/model.py#L364)">lm-evaluation-harness/lm_eval/api/model.py at 11ac352d5f670fa14bbce00e423cff6ff63ff048 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1358759336612200508)** (1 messages): 

> `huggingface_hub v0.30.0, monoELECTRA reranker models, YourBench Custom Evals, Jetson Robot, Accelerate v1.6.0` 


- **Huggingface Hub 迎来史上最大更新！**：[huggingface_hub v0.30.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0) 版本引入了下一代 **Git LFS 替代方案**和新的**推理提供商（inference providers）**。
   - 此版本是*两年来最大的更新！*
- **MonoELECTRA 重排序模型移植至 Sentence Transformers**：来自 @fschlatt1 和 Webis Group 研究网络的 **monoELECTRA-{base, large} 重排序模型**现已在 [Sentence Transformers](https://x.com/tomaarsen/status/1906652865675862125) 中可用。
   - 正如 **Rank-DistiLLM 论文**中所述，这些模型是从 **RankZephyr** 和 **RankGPT4** 等 **LLM** 蒸馏而来的。
- **YourBench 瞬间构建自定义评估**：**YourBench** 允许用户使用其**私有文档**构建**自定义评估（custom evals）**，以评估微调模型在独特任务上的表现（[公告](https://x.com/nathanhabib1011/status/1907728631167902067)）。
   - 该工具对 **LLM 评估**具有变革性意义。
- **Gradio 开发者突破 100 万！**：**Gradio** 是一个用于构建 AI Web 应用的 Python 库，目前每月有超过 **100 万开发者**使用（[公告](https://x.com/abidlabs/status/1907886482150580381)）。
   - 该库已被 **Automatic1111**、**Fooocus** 和 **LLaMA-Factory** 等热门开源项目采用。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/huggingface_hub/releases/tag/v0.30.0">Release Xet is here! (+ many cool Inference-related things!) · huggingface/huggingface_hub</a>: 🚀 准备就绪。Xet 出发！这可能是我们过去两年中最大的更新！Xet 是一个用于在 Git 仓库中存储大对象的新协议，旨在取代 Git LFS。不同于...</li><li><a href="https://x.com/tomaarsen/status/1906652865675862125">来自 tomaarsen (@tomaarsen) 的推文</a>: 我刚刚将来自 @fschlatt1 和 Webis Group 研究网络的优秀 monoELECTRA-{base, large} reranker 模型移植到了 Sentence Transformers！这些模型是在 Rank-DistiL 中引入的...</li><li><a href="https://x.com/nathanhabib1011/status/1907728631167902067">来自 Nathan (@nathanhabib1011) 的推文</a>: 🚀 隆重推出 ✨ YourBench ✨！使用您的私有文档立即构建自定义 evals，并查看您的自定义 fine-tuned 模型在独特任务上的表现。恭喜 @sumukx @clefourrier 和 @ailozovsk...</li><li><a href="https://x.com/RemiCadene/status/1907689862930833545">来自 Remi Cadene (@RemiCadene) 的推文</a>: Jetson @nvidia 版本的机器人现已推出！计算现在像拥有 FSD 的 @Tesla 汽车一样集成在车载系统中 🚗。重要的是，我们重新构思了控制界面，以便您可以查看视频流并...</li><li><a href="https://x.com/_marcsun/status/1907070902455685298">来自 Marc Sun (@_marcsun) 的推文</a>: accelerate v1.6.0 发布了，包含许多优秀功能！- 由我们出色的实习生 @m_sirovatka 支持的 FSDPv2！- DeepSpeed 团队支持的 DeepSpeed + tensor parallel - 用于...的 XCCL 分布式后端。</li><li><a href="https://x.com/hmellor_/status/1906665949530366169">来自 Harry Mellor (@hmellor_) 的推文</a>: @vllm_project 现在有了用户论坛，访问地址：https://discuss.vllm.ai/。这个初创社区仍在成长，但我鼓励所有用户去那里进行以使用为中心的 Q&A！</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1joy1g9/you_can_now_check_if_your_laptop_rig_can_run_a/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://x.com/orr_zohar/status/1907526778278859205">来自 Orr Zohar (@orr_zohar) 的推文</a>: 很高兴看到 SmolVLM 在最新的 BIOMEDICA 更新中为 BMC-SmolVLM 提供动力！仅 2.2B 参数，性能即可媲美 7-13B 的生物医学 VLMs。查看完整发布：@huggingface #smolvlm 引用 Alejandro Lo...</li><li><a href="https://x.com/UnslothAI/status/1906726176556712318">来自 Unsloth AI (@UnslothAI) 的推文</a>: 我们与 @HuggingFace 合作，教你如何使用 GRPO 对 LLMs 进行 fine-tune！学习内容：• Reward functions + 创建它们 • GRPO 数学 + Colab 中的 Free Reasoning 训练 • 将 RL 应用于现实场景...</li><li><a href="https://x.com/_akhaliq/status/1907502083231670300">来自 AK (@_akhaliq) 的推文</a>: 免费 vibe coding AI 应用从未如此简单。100% 开源应用，Hugging Face 上的 DeepSite</li><li><a href="https://x.com/ben_burtenshaw/status/1907798840410808518">来自 Ben Burtenshaw (@ben_burtenshaw) 的推文</a>: 欢迎来到 LLM 课程！教育一直是 Hugging Face 民主化 AI 使命的核心，我们正通过对 http://hf.co/learn 进行重大升级来加大投入！</li><li><a href="https://x.com/SergioPaniego/status/1907095475292897765">来自 Sergio Paniego (@SergioPaniego) 的推文</a>: 🆕 Hugging Face Agents 课程新单元。我们刚刚发布了关于 Agentic RAG 的第一个用例——并排比较了三个框架：🤏 smolagents, 🦙 @llama_index, 🦜 LangGraph (@LangChainAI) ⬇...</li><li><a href="https://x.com/abidlabs/status/1907886482150580381">来自 Abubakar Abid (@abidlabs) 的推文</a>: 迈向 100 万开发者之路。5 年前，我们推出了 @Gradio，作为一个简单的 Python 库，让斯坦福大学的研究人员能够通过 Web 界面轻松演示 computer vision 模型。今天，Gradio 已被...</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1357802672844963912)** (169 条消息🔥🔥): 

> `Llama-4-Scout vs Mistral Small 3.1, AI 工程师面试, Deepmind 内部已实现 AGI？, Fine Tuning 量化模型, Huggingchat 500 错误`

- **Llama-4-Scout 还是 Mistral Small 3.1 更好？**：**Mistral Small 3.1** *增加了视觉理解能力*，并将上下文增强至 **128k tokens**。
   - 一位成员建议 **Llama-4-Scout** 更好，但它需要 **16*17B VRAM**。
- **AI 工程师面试代码环节**：一位社区成员询问 **AI 工程师面试** 的代码部分是什么样的。
   - 另一位成员提到了 **scikit-learn 库**。
- **Deepmind 内部创建 AGI 的传闻**：另一个 Discord 频道的一位成员表示，**Google** 将在下周发布另一个强大的模型，*它甚至比 gemini 2.5 pro exp 还要好*。
   - 他们还声称 **Deepmind** 在内部创建了 **AGI**；然而，这位成员后来表示他不再信任那个人了。
- **微调量化模型是否具有挑战性？**：一位成员询问了关于微调量化模型的问题，社区给出了不同的建议，一些人指出 **QLoRA, Unsloth, bitsandbytes** 是潜在的解决方案。查看 [Unsloth 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-guide)。
   - 而另一位成员表示，只能使用 **LoRA** 来完成。*GGUF 是一种推理优化格式，并非为训练工作流设计的*。
- **Huggingchat 出现 500 错误**：用户报告 **Huggingchat** 遇到了 **500 错误**。
   - 一位成员表示已经提出了该问题，并指出了在 [discord](https://discord.com/channels/879548962464493619/1355513801554006084) 上讨论的变通方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/VIDraft/Gemma-3-R1984-12B">VIDraft/Gemma-3-R1984-12B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.llama.com/llama4/use-policy/">Llama 4 可接受使用政策</a>: Llama 4 可接受使用政策</li><li><a href="https://huggingface.co/posts/Reality123b/155118307932581">Hugging Face 上的 @Reality123b: &quot;好吧，肯定出问题了。HF 对 3 次推理请求收了我 0.12 美元...&quot;</a>: 未找到描述</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI 兼容性 · Ollama 博客</a>: Ollama 现在初步兼容 OpenAI Chat Completions API，使得通过 Ollama 在本地模型上使用为 OpenAI 构建的现有工具成为可能。</li><li><a href="https://huggingface.co/spaces/Remiscus/Customer_Support_Agent">Customer_Support_Agent - Remiscus 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/mindspore-ai/LeNet">mindspore-ai/LeNet · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/consuming_tgi#python">使用 Text Generation Inference</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/text-generation-inference/backends/llamacpp">Llamacpp 后端</a>: 未找到描述</li><li><a href="https://forums.docker.com/t/error-docker-buildx-build-requires-exactly-1-argument-with-vs-code/136577">错误：&quot;docker buildx build&quot; 在 VS code 中需要恰好 1 个参数</a>: 你好，在阅读了大量文档并在互联网上搜索后，我找不到这个问题的原因。当我右键点击并选择“Build image”来构建镜像时，出现了这个错误...</li><li><a href="https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF">bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503">mistralai/Mistral-Small-3.1-24B-Instruct-2503 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/models?dataset=dataset:ylecun/mnist">模型 - Hugging Face</a>: 未找到描述</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-18-27-31.html">剪贴板</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/smolagents/reference/models">模型</a>: 未找到描述</li><li><a href="https://pypi.org/project/audioop-lts/#description">audioop-lts</a>: Python audioop 的 LTS 移植版本</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/mistral3">Mistral3</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/learnpython/comments/144kxze/installed_module_was_told_module_couldnt_be_found/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide>">Unsloth 文档</a>: 未找到描述</li><li><a href="https://youtu.be/9KMxNZ2CvUg">DeepSeek 新品：基于 DeepSeek-GRM-27B 的 SPCT</a>: DeepSeek 发布了一种新的学习方法和一种用于下一代推理模型的新模型，称为 DeepSeek-GRM-27B。在这段视频中，我解释了...</li><li><a href="https://huggingface.co/spaces/Remiscus/Customer_Support_Agent/blob/main/README.md">README.md · Remiscus/Customer_Support_Agent (main 分支)</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-config-reference">Spaces 配置参考</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/36894">由 lddabhi-semron 为 Moshi 启用追踪 · Pull Request #36894 · huggingface/transformers</a>: 此 PR 做了什么？为 MoshiForConditionalGeneration 启用了追踪。将 forward 内部使用的 kwargs 替换为 args。解析 forward 签名以为音频编码器、解码器创建 kwargs...</li><li><a href="https://huggingface.co/blog/how-to-train">如何使用 Transformers 和 Tokenizers 从头开始训练一个新的语言模型</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/mlabonne/llm-course">大语言模型课程</a>: 未找到描述</li><li><a href="https://youtu.be/UU1WVnMk4E8?si=juwuA26e4N_FTaD9">使用 Python 从头开始创建一个大语言模型 – 教程</a>: 学习如何从头开始构建你自己的大语言模型。本课程深入探讨了大语言模型背后的数据处理、数学和 Transformer 架构...</li><li><a href="https://github.com/aashishjhaa/eq-for-youtube">GitHub - aashishjhaa/eq-for-youtube: 使用 6 频段实时处理 YouTube 视频音频</a>: 使用 6 频段实时处理 YouTube 视频音频 - aashishjhaa/eq-for-youtube</li><li><a href="https://aashishjhaa.github.io/eq-for-youtube/">YouTube 均衡器</a>: 未找到描述</li><li><a href="https://github.com/huggingface/text-generation-inference/issues/2890">make in</a>

<li><a href="https://github.com/huggingface/text-generation-inference/issues/2890">install-server 缺少 Apple MacOS Metal Framework · Issue #2890 · huggingface/text-generation-inference</a>: 系统信息显示 `make install-server` 缺少 Apple MacOS Metal Framework。请从 readme 信息中完全移除关于 brew/macOS 的内容，以免误导用户。或者增加对 Apple MPS 的支持...</li><li><a href="https://stackoverflow.com/questions/75593929/torch-circular-import-attributeerror">torch 循环导入 AttributeError</a>: 我尝试运行一个使用 torch 的脚本，但一直遇到这个 AttributeError:&#xA;AttributeError: partially initialized module &#x27;torch&#x27; has no attribute &#x27;Tensor&#x27; (很可能是...</li><li><a href="https://github.com/deepspeedai/DeepSpeed/issues/6005">[BUG] PyTorch nightly 版本的循环导入错误 · Issue #6005 · deepspeedai/DeepSpeed</a>: 描述该 bug：PyTorch nightly 版本的循环导入错误。如果我卸载 DeepSpeed，它就能正常工作。Traceback (most recent call last): File &quot;/test/oss.py&quot;, line 322, in &lt;module&gt; mp.sp...</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: C/C++ 环境下的 LLM 推理</a>: C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号来为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/huggingface/text-generation-inference.git">GitHub - huggingface/text-generation-inference: 大语言模型文本生成推理</a>: 大语言模型文本生成推理 (Large Language Model Text Generation Inference)。通过在 GitHub 上创建账号来为 huggingface/text-generation-inference 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/issues/388">何时使用 SFTTrainer 而非 Trainer？ · Issue #388 · huggingface/trl</a>: 在最近的 QLoRA 博客文章中，Colab 笔记本使用了标准的 Trainer 类，但文章末尾简要提到了 SFTTrainer。为什么在相关的 Colab 笔记本中没有使用它...</li><li><a href="https://stackoverflow.com/questions/76461859/lmm-fine-tuning-supervised-fine-tuning-trainer-sfttrainer-vs-transformers-tr">LMM 微调 - Supervised Fine Tuning Trainer (SFTTrainer) 对比 Transformers Trainer</a>: 在对语言模型 (LLMs) 进行指令微调时，什么时候应该选择 Supervised Fine Tuning Trainer (SFTTrainer) 而不是常规的 Transformers Trainer？据我所知...</li><li><a href="https://opensource.org/blog/metas-llama-2-license-is-not-open-source">Meta 的 LLaMa 许可证不是 Open Source</a>: Meta 正在降低获取强大 AI 系统的门槛，但不幸的是，Meta 制造了 LLaMa 2 是“开源”的误解——事实并非如此。</li><li><a href="https://gnu.support/gnu-emacs/emacs-lisp/Gemma-License-danger-is-not-Free-Software-and-is-not-Open-Source.html">Gemma 许可证（危险）不是 Free Software，也不是 Open Source</a>: **Gemma 使用条款**和**禁止使用政策**规定了 Google Gemma 机器学习模型及其衍生品的使用、修改和分发。虽然 Gemma 可供个人使用...</li><li><a href="https://opensource.org/osd">Open Source 定义</a>: 简介：Open Source 不仅仅意味着可以访问源代码。开源软件的分发条款必须符合以下标准：1. 自由再分发 许可证...</li><li><a href="https://www.gnu.org/philosophy/free-sw.html">什么是 Free Software？
- GNU 项目 - Free Software Foundation</a>: 未找到描述</li><li><a href="https://app.foundershub.ai/user/blogs/d019a1f3-02c3-4388-8d00-5d3d9afcea9a">Hugging Face 如何通过 n8n 工作流增强 AI Agents</a>: 探索 Hugging Face 的 NLP 模型如何与 n8n 集成以构建更智能的 AI Agents。了解由开源语言模型驱动的聊天机器人和数据查询工具等实际用例。</li><li><a href="https://github.com/huggingface/transformers/commit/e959530b8f0011098246572e1777cac06e4bfe73">添加 Mistral3 (#36790) · huggingface/transformers@e959530</a>: * 初始启动 * 样式和占位符 * 创建 convert_mistral3_weights_to_hf.py * 更新 * 拼写错误 * 拼写错误 * 更新 convert_mistral3_weights_to_hf.py * 更新 convert_mistral3_weights_to_hf.py *...
</li>
</ul>

</div>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1357862826504360088)** (16 条消息🔥): 

> `LLM Development, Sebastian Raschka Book, Andrej Karpathy Video, NLP course chapter 3` 


- **社区成员寻求 LLM 开发指导**：一位具有 Data Science 和 ML 背景的社区成员询问从何处开始开发一个 **100M 参数的 LLM**。
   - 建议包括从 **NLP** 或 **DL** 开始，或者找一个特定的课程来学习。
- **推荐使用 Sebastian Raschka 的书来构建 LLM**：推荐使用 **Sebastian Raschka** 编写的 *Build a Large Language Model (From Scratch)* 一书来学习从零开始构建 LLM。
   - 一位成员分享说他们的公司围绕这本书成立了一个读书会，另一位成员提到也订购了同一本书。
- **Andrej Karpathy 的 GPT 复现视频引发讨论**：**Andrej Karpathy** 的视频 [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=2TN0KIfMR8_NxC29) 被推荐为很好的资源。
   - 然而，原帖作者表示*他开始复制粘贴代码且没有解释太多*，所以他们停止了观看。
- **辅助预训练与共享 Embeddings**：一位成员建议*初始化权重并使用来自另一个模型的相同 tokenizer，有点像一种“辅助”预训练*。
   - 他们还提议*共享 embeddings 甚至可能是 linear layer*，以潜在地加速 LLM 的开发过程。



**提到的链接**：<a href="https://youtu.be/l8pRSuU81PU?si=2TN0KIfMR8_NxC29">Let&#39;s reproduce GPT-2 (124M)</a>：我们从零复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先我们构建 GPT-2 网络，然后优化其训练以使其真正……

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1357984983716859996)** (2 条消息): 

> `Windows CLI, Virtual Environment Reset, LocalAI, Dify` 


- **用于虚拟环境重置的 CLI 技巧**：一个用于重置虚拟环境的快速 **Windows CLI** 命令是 `pip freeze | Select-String -Pattern "^(?!pip)" | ForEach-Object { pip uninstall -y $_.ToString().Trim() }`。
   - 根据[一篇博客文章](https://app.foundershub.ai/user/blogs/cf808968-49be-41b9-81e6-9833b2bf2498)，这段代码通过卸载除 **pip** 本身以外的包来帮助清理环境，从而简化了重新开始的过程。
- **[占位符]**：[占位符]
   - [占位符]



**提到的链接**：<a href="https://app.foundershub.ai/user/blogs/cf808968-49be-41b9-81e6-9833b2bf2498">The Complete Roadmap to Mastering Agentic AI in 2025 | Girish Kotte</a>：发现 2025 年掌握 Agentic AI 的全面 12 步路线图。学习从基础概念到高级部署技术的所有内容，并提供每个阶段的资源链接。非常适合开发人员……

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1357965193673703575)** (8 条消息🔥): 

> `MCP Server 和 RAG 应用，Osyllabi AI 课程，DocQuery AI 文档搜索，市政法律数据集，搭载 Llama-4 的 LlamaResearcher` 


- ****MCP Server + RAG 应用亮相****：一名成员构建了一个 **MCP Server** 和客户端，通过 **ngrok** 连接，并开发了一个简单的 **RAG 应用**，用于对 GitHub 仓库中的 Markdown 文档进行问答，并在 [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_mcp-llm-enterprise-activity-7314281410712735744-yXJI) 上进行了展示。
   - 该 RAG 应用名为 **DocQuery**，可在 [docquery-ten.vercel.app](https://docquery-ten.vercel.app/) 获取并提供反馈。
- ****Osyllabi：AI 课程制定工具上线 GitHub****：一名成员分享了 **Osyllabi**，这是一个 Python 应用，利用网页爬取和数据集成，通过 **Ollama**、**HuggingFace**、**Langchain** 和 **Llama-Index** 驱动，生成 AI 驱动的个性化课程，代码托管在 [GitHub](https://github.com/Ollama-Agent-Roll-Cage/oarc-osyllabi)。
   - 它具有 AI 驱动的课程生成、高级网页爬取、与教育平台的无缝集成、可定制的学习路径以及灵活的导出选项。
- ****DocQuery 将文档转换为知识库****：一名成员分享了 **DocQuery**，它可以将文档 Markdown 转换为知识库，代码托管在 [GitHub](https://github.com/md-abid-hussain/docquery)。
   - DocQuery 为开发团队提供了改进的搜索能力、智能问答系统以及流式化的知识管理。
- ****美国市政法律数据集发布****：一名成员在 [Hugging Face Datasets](https://huggingface.co/datasets/the-ride-never-ends/american_municipal_law) 上分享了 **American Municipal Law** 数据集，包含来自全美各地的市级和县级法律，采用 Parquet 格式，按地理位置的 **GNIS id** 组织。
   - 访问该数据集需要同意分享联系信息。
- ****LlamaResearcher：Llama-4 驱动深度研究****：一名成员介绍了 **LlamaResearcher** ([llamaresearcher.com](https://llamaresearcher.com))，这是一个由 **Llama 4** 和 **Groq** 驱动的深度研究 AI 助手，它可以将查询扩展为子查询，搜索网页，并生成带有来源引用的论文。
   - 该项目是开源的且支持 Docker，代码托管在 [GitHub](https://github.com/AstraBert/llama-4-researcher)，并使用了 **LlamaIndex**、**Groq**、**Linkup**、**FastAPI**、**Redis** 和 **Gradio**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/md-abid-hussain/docquery">GitHub - md-abid-hussain/docquery: DocQuery: Turn your documentation markdown to knowledgebase</a>: DocQuery: 将你的文档 Markdown 转换为知识库 - md-abid-hussain/docquery</li><li><a href="https://github.com/p3nGu1nZz/osyllabi">GitHub - Ollama-Agent-Roll-Cage/oarc-osyllabi: Osyllabi: A streamlined Python app for designing personalized curriculums using AI, web crawling, and data integration.</a>: Osyllabi: 一个精简的 Python 应用，用于利用 AI、网页爬取和数据集成设计个性化课程。 - Ollama-Agent-Roll-Cage/oarc-osyllabi</li><li><a href="https://github.com/the-ride-never-ends/municipal_law_search">GitHub - the-ride-never-ends/municipal_law_search</a>: 通过在 GitHub 上创建账号来为 the-ride-never-ends/municipal_law_search 做出贡献。</li><li><a href="https://huggingface.co/datasets/the-ride-never-ends/american_municipal_law">the-ride-never-ends/american_municipal_law · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://llamaresearcher.com),">未找到标题</a>: 未找到描述</li><li><a href="https://llamaresearcher.com">LlamaResearcher - 数秒内将主题转化为论文！</a>: AI 驱动的研究助手，可深度搜索网页、验证信息，并在数秒内生成关于任何主题的论文。</li><li><a href="https://github.com/AstraBert/llama-4-researcher">GitHub - AstraBert/llama-4-researcher: Turn topics into essays in seconds!</a>: 数秒内将主题转化为论文！通过在 GitHub 上创建账号来为 AstraBert/llama-4-researcher 做出贡献。</li><li><a href="https://docquery-ten.vercel.app/">DocQuery</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1358136363932848199)** (5 messages): 

> `OCR 数据标注、手写文本的 VLM 微调、将 OCR 技术与 VLM 结合、使用 Roboflow 管理图像和标签、MS-Swift 和 PEFT/Unsloth 方法` 


- **VLM 模型辅助手写文本 OCR**：一名成员正在寻求数据标注方法，以便在手写文本图像上微调 VLM 模型，旨在摆脱传统的 OCR 模型，并需要真实的文本标签进行训练。
   - 他们正在考虑使用各种工具和方法从图像中生成或纠正文本标签，以用于微调目的。
- **经典 OCR 与开源 VLM 结合进行标注**：一名成员将经典 OCR 技术与 **InternVL2_5** 和 **Qwen2.5** 等开源 VLM 相结合，生成初始标注，以从巴西文档中提取结构化数据。
   - 在使用 OCR/VLM 后进行了人工复核以纠正错误，并指出像 **Gemini** 这样的闭源模型可能会提供更高质量的预标注。
- **Roboflow 有效管理图像和标签**：一名成员使用 **Roboflow** 管理和存储原始图像及纠正后的标签，标注了 **510** 张图像，并增强至 **1218** 个样本。
   - 尽管发现其交互体验并不理想，他们仍使用 **Roboflow** 来管理数据集。
- **MS-Swift 和 PEFT/Unsloth 增强微调效果**：一名成员使用 **MS-Swift** 微调了多个模型，并尝试了 **PEFT** 和 **Unsloth** 方法，在模型从 1B 调整至 7B 的过程中，实现了优于 **Gemini** 和传统 OCR 方法的性能。
   - 该成员成功微调了模型，强调了这些框架的有效性。
- **Tesseract OCR 与 Label Studio 联手**：一名成员正在考虑使用 **Tesseract OCR** 配合 **Label Studio** 来完善标注。
   - 他们还测试了 **Gemma 3** 并发现其效果显著，这意味着数据标注可以采用自动化与人工相结合的方法。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1357937615596421371)** (5 messages): 

> `从 PDF 中提取文本、Docling、SmolDocling、RolmOCR、Sci-BERT` 


- **寻求 PDF 文本提取建议**：一名成员正在寻求改进从 PDF（特别是研究论文）中提取文本的建议，因为他们目前的结果不尽如人意。
   - 他们一直在使用正则表达式（regex）进行章节大纲提取，但在字体、页眉和页脚方面面临挑战，由于 Token 限制，这影响了提取内容在 **Sci-BERT** 嵌入中的可用性。
- **推荐使用 Docling 和 SmolDocling 进行文本提取**：一名成员推荐使用 **Docling** ([GitHub](https://github.com/docling-project/docling)) 和 **SmolDocling** ([HuggingFace](https://huggingface.co/ds4sd/SmolDocling-256M-preview)) 以改进 PDF 文本提取。
   - 他们指出，虽然这些工具仍会出现错误（尤其是在处理图像时），但已取得了不错的效果。**SmolDocling** 是一款超紧凑的视觉语言模型，用于端到端的多模态文档转换，正如[其论文](https://huggingface.co/papers/2503.11576)中所强调的那样。
- **基于 Qwen 2.5 VL 的 RolmOCR 模型发布**：一名成员提到发布了 **RolmOCR** ([HuggingFace](https://huggingface.co/reducto/RolmOCR))，这是一个基于 **Qwen 2.5 VL** 的新模型，用于 OCR 任务。
   - 尽管他们尚未亲自测试，但建议将其作为文本提取的潜在工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/docling-project/docling">GitHub - docling-project/docling: 为生成式 AI 准备好你的文档</a>: 为生成式 AI 准备好你的文档。通过在 GitHub 上创建账号，为 docling-project/docling 的开发做出贡献。</li><li><a href="https://huggingface.co/ds4sd/SmolDocling-256M-preview">ds4sd/SmolDocling-256M-preview · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1357795572920225884)** (24 messages🔥): 

> `OpenWeatherMap API, ISO 3166-1 alpha-2 code, Qwen/Qwen2.5-Coder-32B-Instruct Alternatives, Hugging Face Token for Agent Creation, llm-course Channel` 


- **Geolocation API vs 静态国家代码字典**: 一位成员正在构建一个使用 **OpenWeatherMap API** 获取天气状况的工具，并正在讨论是使用 **GeoCoding API** 和另一个用于 **ISO 3166-1 alpha-2 代码**的 API，还是使用静态字典。
- **Qwen/Qwen2.5-Coder-32B-Instruct 的免费替代方案？**: 一位成员询问 **Qwen/Qwen2.5-Coder-32B-Instruct** 的免费替代方案。
   - 另一位成员指出，该模型本身在 Apache 2.0 许可证下是免费的（[Hugging Face 链接](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)），但建议使用 **Together AI** 或 **Groq** 获取免费 API 访问，并指出其速率限制约为 60 RPM。
- **关于 Agent 创建的 Hugging Face Token 指南**: 一位成员请求关于在课程 Unit 1 中获取用于 Agent 创建的 **Hugging Face token** 的指导。
- **llm-course 频道请求**: 一位成员询问是否可以为 **LLM 课程**开设专门的频道。
- **AI agents 课程设置寻求帮助**: 一位成员请求协助解决在 **AI agents 课程** Unit 1 中遇到的代码问题，特别是与 Colab 中的 **HF token 设置**相关的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-17-33-44.html">clipboard</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1357936147308875839)** (36 messages🔥): 

> `MCP in Agent Course, Inference Usage Costs, Gemini Models, Course Feedback, Hallucination in Agents` 


- **Agent 课程中几乎未提及 MCP**: 一位用户询问在 Agent 课程中学习 **MCP** 的事宜，但被告知没有专门的章节，尽管在 [unit 2.1 (smolagents)](https://huggingface.co/learn/agents-course/unit2/smolagents/tools#importing-a-tool-collection-from-any-mcp-server) 和 [unit 2.2 (llamaindex)](https://huggingface.co/learn/agents-course/unit2/llama-index/tools#model-context-protocol-mcp-in-llamaindex) 中简要提到了 **MCP** 服务器。
- **产生了推理费用！**: 一位用户不小心超出了他们的 **Inference Usage Due Balance**（推理使用欠费余额）并询问如何支付。
   - 建议查看问题频道的 **FAQ**，或者使用本地或更便宜的托管替代方案。
- **Gemini 模型可能是你的救星**: 一位用户在第 2 章中因付费要求而遇到 **Code_agents** notebook 问题，被建议尝试使用 **Gemini 模型**。
   - 有人指出 **Gemini 模型**在许多国家可以免费使用，并提供了包含说明的[课程笔记](https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face)链接。
- **课程体验：内容很好但 Bug 较多**: 一位用户总结该课程充满了优秀的材料，但指出许多 notebook 和代码片段无法运行，包括 Unit 2 中一个臭名昭著的代码测试，且没有讲师参与。
   - 建议以怀疑的态度对待课程，专注于理解代码部分，并获取必要的账号和 API token。
- **解释幻觉（hallucinations）！**: 用户寻求关于 Agent 中**幻觉**示例的澄清。
   - 提供的解释是，由于 Agent 无法访问天气数据，它编造了答案，解决方案是为 Agent 配备一个检索天气信息的工具。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face))">前端开发关于 Hugging Face "Agents Course" 的笔记</a>: 前端开发关于 Hugging Face "Agents Course" 的笔记 - 01_context.md</li><li><a href="https://huggingface.co/learn/agents-course/unit2/smolagents/tools#importing-a-tool-collection-from-any-mcp-server).">Tools - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit2/llama-index/tools#model-context-protocol-mcp-in-llamaindex)">在 LlamaIndex 中使用工具 - Hugging Face Agents Course</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1357805087711432905)** (177 messages🔥🔥):

> `Grok 3, Turing Machines, Raw Binary AI training, LLama 4, Quantization Techniques` 


- **Grok 3 流形类比出现**：一位成员分享了一个描述 **NLP** 不同方法的类比，对比了 **0D Manifolds (tokens)**、**1D Manifolds (embeddings)** 以及一种**动态信号方法 (dynamic signal approach)**，后者将语言视为一条没有固定边界的“奔腾旋涡”之河。
- **探讨原始二进制 AI 训练 (Raw Binary AI Training)**：成员们讨论了在**原始二进制数据 (raw binary data)** 上训练 AI，以直接输出 **mp3** 或 **wav** 等文件格式，一位成员指出这种方法基于 **Turing machines** 等离散数学原理。
   - 另一位成员认为目前的 AI 模型远未达到 **Turing-complete**，而原帖作者解释说，AI 不需要具备 **Turing-complete** 特性也能输出合适的 **tokens** 作为响应。
- **新 Llama 4 模型发布**：**Llama 4 Scout** 拥有 **10 million context window**、**17B active parameters** 和 **109B total parameters**；**Llama 4 Maverick** 提供 **1m context length**、**17B active parameters** 和 **400B total parameters**；而 **Llama 4 Behemoth** 则具备 **2 trillion parameters**。
   - 成员们对 **10M context window** 的说法及新的 **license** 表示怀疑，并质疑最近的模型是经过 **RL** 还是仅仅是 **base + SFT** 模型，同时指出了性能问题和参差不齐的 **benchmarks**。
- **探索自原则批判微调 (Self-Principled Critique Tuning)**：来自 DeepSeek 的 **Self-Principled Critique Tuning (SPCT)** 是一种新的奖励模型系统，其中 **LLM** 在自动生成的推理原则提示下，根据这些原则对 **CoT** 输出生成**批判 (critiques)**。
   - 该系统旨在训练模型自动开发推理原则，并以更趋向 **system 2** 的方式评估自身输出，而不是依赖人工设计的奖励，详见 [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)。
- **研究量化技术 (Quantization Techniques)**：成员们讨论了针对大语言模型的新型量化技术，并引用了一篇包含该[文件](https://proceedings.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf)的论文。
   - 有观点认为，量化可以作为维持超长上下文长度与模型服务能力之间的折中方案，但代价是从这些长上下文中实际获取的价值会有所衰减。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ficlive/status/1908911992686931989?t=N1BGmubwXQQ-ZYLSKfmXqw&s=19">来自 Fiction.live (@ficlive) 的推文</a>：更新了 Llama 4 的长上下文基准测试</li><li><a href="https://arxiv.org/abs/2504.01990">Foundation Agents 的进展与挑战：从类脑智能到进化、协作与安全系统</a>：大语言模型 (LLMs) 的出现催化了人工智能领域的变革性转变，为具备复杂推理、稳健性能的先进智能 Agent 铺平了道路...</li><li><a href="https://arxiv.org/abs/2504.01002">Token 嵌入违反了流形假设</a>：要全面理解大语言模型 (LLM) 的行为，需要我们理解其输入空间。如果这个输入空间与我们的假设不同，我们对其的理解和结论...</li><li><a href="https://x.com/_arohan_/status/1909018336060747976">来自 rohan anil (@_arohan_) 的推文</a>：@ficlive 他们似乎没有启用 attn 配置。我会尝试联系他们。同时，https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/build_with_llama_4.ipynb 包含了...</li><li><a href="https://arxiv.org/abs/2410.10714">SeedLM：将 LLM 权重压缩为伪随机生成器的种子</a>：大语言模型 (LLMs) 改变了自然语言处理，但由于其高昂的运行成本，在广泛部署方面面临重大挑战。在本文中，我们介绍了 SeedLM，...</li><li><a href="https://arxiv.org/abs/2504.02495">通用奖励模型的推理时扩展</a>：强化学习 (RL) 已被广泛应用于大语言模型 (LLMs) 的大规模后训练。最近，通过 RL 激励 LLMs 的推理能力表明 $...</li><li><a href="https://arxiv.org/abs/2504.01017">扩展无语言视觉表示学习</a>：视觉自监督学习 (SSL) 目前在视觉问答 (VQA) 等多模态场景下的表现不如对比语言-图像预训练 (CLIP)。这种多模态差距通常...</li><li><a href="https://arxiv.org/abs/2502.02631">ParetoQ：极低比特 LLM 量化中的扩展定律</a>：在量化模型大小和准确性之间实现最佳权衡的最佳比特宽度一直是持续争论的话题。虽然有些人主张 4-bit 量化，但其他人建议 1...</li><li><a href="https://arxiv.org/abs/2409.12917">通过强化学习训练语言模型进行自我纠错</a>：自我纠错是大语言模型 (LLMs) 一项非常理想的能力，但在现代 LLMs 中一直被发现很大程度上是无效的。目前的自我纠错训练方法...</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | 模型卡片与提示词格式</a>：Llama 4 Maverick 和 Llama 4 Scout 的技术细节和提示词指南
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1357864625990926478)** (28 条消息🔥): 

> `Llama 4, DeepSeek Paper, PaperBench, Text Diffusion` 


- **Llama 4 Omni 觉醒**：一位成员分享了 [Llama 4 文档](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/)，随后附上了 Meta 关于 [Llama 4 多模态智能 (Multimodal Intelligence)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 的博客文章链接。
   - **Llama 4 Scout** 模型拥有 **170 亿** 激活参数、**16 个专家 (experts)**，以及行业领先的 **10M** 上下文窗口，性能超越了 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1** 等模型。
- **PaperBench：OpenAI 的复现基准测试**：一位成员分享了关于 [OpenAI PaperBench 基准测试](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052) 的文章，该基准旨在测试 AI Agent 从零开始复现前沿机器学习研究论文的能力。
   - 该基准测试评估 Agent 复现 **ICML 2024** 整篇 **ML 论文** 的能力，并使用 **LLM judges** 和与原作者共同设计的细粒度评分标准进行自动评分。
- **DeepSeek 论文时间**：成员们计划在一小时内研读第一篇 **DeepSeek** 论文，并提供了论文链接 ([https://arxiv.org/abs/2401.02954](https://arxiv.org/abs/2401.02954))。
   - 讨论在 [Discord 活动](https://discord.gg/jvDVwtfq?event=1357475477500985605) 中进行。
- **文本扩散引导自回归 LM**：成员们计划讨论一篇论文 ([https://arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220))，该论文关于使用引导扩散模型 (guided diffusion model) 来引导自回归语言模型 (auto-regressive language model) 生成具有所需属性的文本。
   - 分享了主作者近期讨论该论文的演讲视频 ([https://www.youtube.com/watch?v=klW65MWJ1PY](https://www.youtube.com/watch?v=klW65MWJ1PY))。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.02954">DeepSeek LLM: Scaling Open-Source Language Models with Longtermism</a>：开源大语言模型 (LLMs) 的快速发展确实令人瞩目。然而，先前文献中描述的 Scaling Law 呈现出不同的结论，这给……蒙上了阴影。</li><li><a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-052">🥇本周热门 AI 论文</a>：本周热门 AI 论文 (3 月 31 日 - 4 月 6 日)</li><li><a href="https://arxiv.org/abs/2408.04220">Diffusion Guided Language Modeling</a>：目前的语言模型在文本生成方面表现出卓越的能力。然而，对于许多应用来说，控制生成文本的属性（如情感或毒性）是很有必要的……</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama4_omni/">Llama 4 | 模型卡片与提示词格式</a>：Llama 4 Maverick 和 Llama 4 Scout 的技术细节和提示词指南。</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>：未找到描述。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1357801975487402144)** (17 messages🔥): 

> `GPT-6 发布, Llama 4, Mindcraft 更新, 适配预训练文本, 利用 diffusion modeling 控制 LLM` 


- **GPT-6 即将推出（也许？）**: 一名用户昨天开玩笑地宣布了 **GPT-6** 的发布，随后几周将发布 **O0** 和 **OO**，理由是 **GPT-5** 遇到了困难。
   - 这引发了幽默的反应，另一名用户调侃道：*“‘发布’并不意味着像那些对 AI 开放的公司那样真正发布权重。”*
- **Llama 4 问世，具备 10M 上下文**: 根据 [llama.com](https://www.llama.com/llama4/) 的消息，**Llama 4 Maverick** 是*同类中最智能的多模态 OSS 模型*，拥有 **128 个专家的 170 亿参数模型**以及 **10M** 的上下文窗口。
   - 据称该模型比所有前代 Llama 模型都更强大，同时能装入单块 **NVIDIA H100 GPU**，超越了 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1**。
- **Mindcraft 更新：机器人能看见了！**: 一名成员分享了一个名为 **"Vision and Vibe Coding | Mindcraft Update"** 的 [YouTube 视频](https://www.youtube.com/watch?v=iDJ6GrHNoDs)。
   - 视频描述中包含了一个 [Tripo AI](https://www.tripo3d.ai/app?invite_code=R2XF70) 的链接，为前 300 名使用代码 **R2XF70** 注册的用户提供额外积分。
- **为数据库查询训练的 LLM**: 一名成员提到适配预训练文本以包含相关事实的数据库查询，从而训练 **LLM** 在生成过程中进行查找，并引用了[这段视频](https://youtu.be/upbz6k6IDrk)。
- **Diffusion Modeling 现可控制 LLM**: 用户讨论了使用 diffusion modeling 来控制 **LLM**，参考了论文 ["Diffusion-LM Improves Controllable Text Generation"](https://arxiv.org/pdf/2408.04220)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.llama.com/llama4/">Llama 4 is Here | Meta</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct">meta-llama/Llama-4-Scout-17B-16E-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">无标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=iDJ6GrHNoDs">Vision and Vibe Coding | Mindcraft Update</a>: 尝试 Tripo AI: https://www.tripo3d.ai/app?invite_code=R2XF70 我的代码: R2XF70 前 300 名注册用户将在 Tripo 上获得 500 点额外积分！机器人能看见了！...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1357860447231021209)** (17 messages🔥): 

> `CUDA Python 包, 向量化内存访问, Llama-4 Router 归一化, 高 RAM/VRAM SSH 访问` 


- **CUDA Python 包首次亮相**: Nvidia 发布了 [CUDA Python 包](https://developer.nvidia.com/cuda-python)，为 CUDA 驱动程序和运行时 API 提供 **Cython/Python 封装**，可通过 PIP 和 Conda 安装。
   - 它的目的是统一 Python CUDA 生态系统，提供对 Python **CUDA host API** 的全面覆盖和访问，主要惠及需要与 C++ API 交互的库开发者。
- **寻求向量化内存访问的最佳实践**: 成员们讨论了在处理动态形状（特别是具有动态维度 *m, n 和 k* 的矩阵乘法）时，**向量化内存访问**的最佳实践。
   - 讨论中提到了 [Cutlass](https://developer.nvidia.com/cutlass) 的支持和高效的向量化加载作为潜在的解决方案。
- **探讨 Llama-4 Router 归一化**: 频道讨论了 **Llama-4** 是否使用了 Router 归一化，类似于 DeepSeek V3 和 Mixtral 处理其 *topk_weights* 归一化的方式。
   - 有人指出 Llama-4 跳过了归一化，可能是因为它使用了 `top_k = 1`，并且 **DeepSeek V3** 和 **Llama 4** 都对 Router Logits 使用了 Sigmoid。
- **测试需要高 RAM/VRAM 的 SSH 访问**: 一名成员寻求访问一个至少拥有 **500GB RAM/VRAM** 的类 SSH 实例几个小时，以便在 **SGL** 中测试一个模型。
   - 他们拥有来自 Modal 的 GPU 额度，并询问了关于 SSH 访问容器的事宜。



**提到的链接**: <a href="https://developer.nvidia.com/cuda-python">CUDA Python</a>: CUDA Python 为我们的合作伙伴提供统一的 API 和绑定，以便包含在他们经过 Numba 优化的工具包和库中，从而简化 HPC、数据科学和 AI 的基于 GPU 的并行处理。

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1358241585007165562)** (18 条消息🔥): 

> `Triton Kernel Debugging, GPU Assembly Debugging, Grayscale Kernel Writing, Block Index Creation, Data Transposing` 


- ****Triton Kernel Debugging** 逐步指南**: 一位初次发言者询问了如何逐步调试 **Triton kernels**，特别是解决在 `interpret mode = 1` 时 `cdiv` 和 `fill zeros` 的问题。
   - 另一个建议涉及深入研究 **GPU assembly**，使用 `cuda gdb` 或 `roc gdb` 在 Python 文件中设置断点，并单步执行汇编文件。
- **使用 VSCode 进行 **GPU Assembly Debugging****: 一位成员询问是否可以使用 **VSCode debugger** 代替 `cuda gdb` 来调试 **GPU assembly**。
   - 讨论指出，虽然需要运行 `cuda gdb` 并传入 **Python arguments**，但用户更希望获得 **VSCode debugger** 的便利性和可读性。
- ****Grayscale Kernel Writing** 的 Block Index**: 一位成员描述了为 `(K, K, 3)` 输入编写 **grayscale kernel** 的尝试，目标是在 **Triton** 中获取 `(BLOCK_K, BLOCK_K, 3)` 的 blocks。
   - 然而，他们在处理 `tl.arange(0, 3)` 时遇到了挑战，因为 3 不是 2 的幂。
- **加载 **Nx3 Blocks****: 一位成员询问如何加载 **Nx3 block**，因为 `tl.arange` 无法工作（3 不是 2 的幂）。
   - 一个建议是分三次加载数据，并按 `image_w * image_h` 递增范围；另一位成员建议对所有索引加 1 应该可行。
- **针对连续数据的 **Data Transposing****: 一位成员考虑在比赛中使用 **Torch** 进行数据转置，但他们担心滥用 strides 来加载连续数据。
   - 建议认为在比赛中使用 **Torch** 进行转置是可接受的，因为原始 Tensor 是连续的，转置操作仅是符号上的。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1357816562878513334)** (18 条消息🔥): 

> `CUDA debugger, nvshmem + mpi, nvbench and ubuntu 24.04, Shared memory access in CUDA, cute::copy and tiled_copy behavior` 


- **CUDA Debugging 的乐趣**: 一位用户确认 **CUDA's debugger** 的工作方式与 **GDB CLI** 非常相似。
   - 另一位成员询问了今年 GTC 上宣布的 **cutile** 的发布日期。
- **nvshmem + MPI 竞态条件**: 一位成员报告了在进程数比 GPU 数量多一个的情况下运行 **nvshmem + mpi** 时（无论是否开启 **MPS**）出现的竞态条件和挂起问题。
   - 他们在拥有 **4 个 GPU** 的系统上运行 `mpirun -np 5 ./myapp`，并询问是否有人有解决方案。
- **NVBench 提高 CMake 版本要求**: **NVBench** 某种程度上放弃了对 **Ubuntu 24.04** 的支持，因为它需要最低 **3.30** 版本的 CMake，而 Ubuntu 24.04 自带的是 **3.28**。
   - 一位成员建议[在 nvbench 仓库提交 issue](https://github.com/NVIDIA/nvbench/issues/new)，并指出可以使用[之前的 tag](https://github.com/NVIDIA/nvbench)作为权宜之计。
- **CUDA 中的 Shared Memory 广播**: 针对关于 CUDA 中 Shared memory 访问的问题，确认了 Shared memory 存在 **broadcasts 和 multicasts** 机制。
   - 一位成员指向了 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)，并补充说 Warp shuffles 应该具有更高的性能。
- **Cute Copy 的异常行为**: 一位用户发现 **cute::copy** 在处理 **tiled_copy** 时存在奇怪行为：Warp 中的所有线程集体将数据从 Shared memory 复制到寄存器，而不是每个线程复制其对应的数据。
   - 附带的[图片](https://cdn.discordapp.com/attachments/1189607726595194971/1358772917415973076/image.png?ex=67f50f64&is=67f3bde4&hm=fd23fb036477608b6ded972e4d638b8ade745699033f0d26ff8ca5d08da56f2c)展示了复制操作后寄存器中意想不到的数据排列。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://apt.kitware.com">Kitware APT Repository</a>: 无描述</li><li><a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks">1. Preface — CUDA C++ Best Practices Guide 12.8 documentation</a>: 无描述</li><li><a href="https://cmake.org/download.">Download CMake</a>: 无描述</li><li><a href="https://github.com/NVIDIA/nvbench/issues/new">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1357857834632740884)** (10 messages🔥): 

> `torch compile backend, libtorch, mojo, torchscript, gelu+mul fusion` 


- ****Graphviz 后端尚未为 Torch Compile 准备就绪****：一位成员询问是否有能输出 **graphviz** 的 **torch.compile backend**，另一位成员回答说，他们正致力于使用 **torch.compile** 生成**不依赖 libtorch 的二进制文件**。
   - 他们进一步声称，目前没有巧妙的方法可以在 **torchscript** 上加载模型。
- ****Mojo 不太可能绕过 Python 的 GIL****：一位成员询问是否有人使用 **mojo** 来绕过 **Python 的 GIL**。
   - 未收到回复，因此可以推断答案是否定的。
- ****为基准测试编译 Gelu+Mul 融合****：一位成员询问如何让 **torch.compile** 在 PyTorch 2.8 版本中正确且可靠地融合 **gelu+mul**，以便进行基准测试，并与其 **Triton kernel** 进行对比。
   - 未收到回复，因此可以推断该融合确实存在困难！
- ****DDP/FSDP 与编译惯例****：一位成员询问在将模型包装到 **DDP/FSDP1/FSDP2** 之前进行编译的通用惯例。
   - 另一位成员指向了 [torchtitan 的实现](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313) 作为参考，该实现预先执行了一种奇怪的逐块编译（per-block compile），可能是为了规避某些 **torch compile bugs**。
- ****数值问题困扰 FSDP****：一位成员报告在使用 **FSDP** 时遇到了**数值问题**，并已完全禁用 **torch compile**。
   - 他们声称 *编译对他们作用不大*，但 **torchtitan** 的作者需要编译 **flex attention**，并希望能融合他们的一些序列并行 TP (sequence parallel TP) 相关内容，而块包装（block-wrapping）是一种折中方案。



**提到的链接**: <a href="https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313">torchtitan/torchtitan/models/llama3/parallelize_llama.py at main · pytorch/torchtitan</a>：一个用于大模型训练的 PyTorch 原生库。通过在 GitHub 上创建账号为 pytorch/torchtitan 的开发做出贡献。

  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1358007632308470000)** (1 messages): 

> `GPU Mode Website, Active Leaderboards, Website Feedback` 


- ****GPU Mode** 发布新网站**：感谢两位成员的辛勤工作，**GPU Mode** 推出了新[网站](https://www.gpumode.com/)。
   - 该网站包含活跃的排行榜、YouTube 上的讲座链接以及他们的 GitHub 仓库。
- **排行榜状态显示 H100 的主导地位**：网站设有 **A100, T4, H100 和 L4 GPU** 的活跃排行榜，其中多个排行榜显示了 **H100** 的结果。
   - 例如，在一个还有 21 天结束的排行榜中，*ajhinh* 在 **H100** 上以 **7574.126μs** 排名第一。
- **征求对网站功能的反馈**：团队正在征求关于[网站](https://www.gpumode.com/)增加哪些功能的反馈。
   - 目前的功能包括排行榜状态、YouTube 讲座和 GitHub 仓库；反馈可以在指定频道中提供。



**提到的链接**: <a href="https://www.gpumode.com/">Leaderboards &ndash; GPU MODE</a>：未找到描述

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1358180662217281617)** (6 messages): 

> `Llama 4, Triton Distributed, Tensara Triton Support, AMD Instinct MI325X Performance` 


- ****Llama 4** 携多模态实力登场**：Meta 推出 **Llama 4**，这是其最新迭代版本，拥有增强的个性化多模态体验，并包含 **Llama 4 Scout** —— 一个拥有 **170 亿**参数和 **16 个专家 (experts)** 的模型（[博客文章点击此处](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)）。
   - 它声称性能优于 **Gemma 3**、**Gemini 2.0 Flash-Lite** 和 **Mistral 3.1**，并能适配单块 **NVIDIA H100 GPU**，拥有行业领先的 **10M** 上下文窗口。
- **字节跳动发布用于并行系统的 **Triton-distributed****：ByteDance-Seed 发布了 **Triton-distributed**，旨在扩展 Triton 语言的可用性（[GitHub 点击此处](https://github.com/ByteDance-Seed/Triton-distributed)）。
   - 该新版本专为并行系统开发而设计。
- ****Tensara** 增加 **Triton** 支持以应对 GPU Kernel 挑战**：**Tensara** 现在支持 **Triton**，邀请用户参加 Kernel 优化挑战并冲击全球排行榜（[主页点击此处](https://tensara.org)）。
   - 最近的更新包括 **基于 PyTorch 的测试用例**、3D/4D Tensor matmul 问题，以及 Sigmoid 和 Tanh 等激活函数。
- **AMD 的 **Instinct MI325X** 在 MLPerf 推理性能中表现强劲**：**AMD Instinct™ MI325X** GPU 在 **MLPerf Inference v5.0** 中展示了强大的性能，在 GenAI、LLM 和推理模型方面表现出色（[博客点击此处](https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark)）。
   - 结果表明，需要为 AI 转型量身定制创新的 GPU 架构。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark)">AMD InstinctTM MI325X GPUs Produce Strong Performance in MLPerf Inference v5.0 &#8212; ROCm Blogs</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/ByteDance-Seed/Triton-distributed">GitHub - ByteDance-Seed/Triton-distributed: Distributed Triton for Parallel Systems</a>: 用于并行系统的分布式 Triton。通过在 GitHub 上创建账号来为 ByteDance-Seed/Triton-distributed 的开发做出贡献。</li><li><a href="https://tensara.org">Home | Tensara</a>: 一个 GPU 编程挑战平台。编写高效的 GPU Kernel 并与其他开发者的解决方案进行比较。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1357913744159805531)** (6 messages): 

> `Qualcomm AI Engineer Hiring, Suno ML roles and H100 resources, Zero latency music creation` 


- **高通寻求 AI 工程师团队负责人**：Qualcomm 正在招聘一名具有深厚深度学习背景的 **AI 工程师/团队负责人**，负责设计/部署 **SOTA 模型**，重点关注 **准确率-延迟帕累托最优 (accuracy-latency Pareto optimality)**。
   - 有意向的候选人请提供一份简短摘要以及简历或作品集。
- **Suno 的 ML 人才搜寻**：Suno 正在招聘所有 **ML 相关职位**，宣传其团队精干且资源丰富，每位研究员拥有 **数百块 H100**。
   - 他们的目标是实现 **零延迟音乐创作**，让人们可以与 **AI 实时即兴演奏**。
- **零延迟音乐创作听起来很酷**：Suno 旨在实现 **零延迟音乐创作**，支持实时 AI 即兴演奏。
   - 一位用户表示希望 **Suno** 能成为 **Ableton 中的 VSTi**。
- **Suno 实习机会丰富**：一位用户询问了 Suno 的实习机会，并对该平台表示赞赏。
   - 暂无回复。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1357818915912286422)** (19 条消息🔥): 

> `集中式 GPU 编程语言, OpenCL 与 SYCL, ROCm 与 HIP, LLM 的 CUDA 4-bit 操作, 性能 Roofline 模型与算术强度` 


- **为什么没有一种 GPU 编程语言能统治一切**：一位 GPU 编程新手询问，既然 NVIDIA 有 **CUDA**，AMD 有 **ROCm**，为什么没有像 **C** 语言那样统一的 GPU 编程语言。
   - 一位专家解释说，**OpenCL** 和 **SYCL** 确实存在，但由于 NVIDIA 等厂商的支持较差，它们并未成为主流，并指出 OpenCL 的接口陈旧且过于接近 C 语言。
- **ROCm 的双重性质：AMD 的 CUDA Toolkit 与 HIP**：**ROCm** 是 AMD 的 CUDA Toolkit，而 **HIP** 是 AMD 的 CUDA C++，它支持 NVIDIA 硬件并能编译为 PTX，但不持支 Intel 或其他平台。
   - 这提供了一定程度的跨平台能力，尽管并非通用。
- **在 CUDA 中处理 LLM 的 4-Bit 操作**：一位用户询问如何在 CUDA 中为 LLM 执行 **4-bit 操作**（如 matmul）。
   - 另一位成员建议在专门的 CUDA 频道提问，并对具体操作描述得更详细一些。
- **解读性能 Roofline 模型中的算术强度**：一位成员对在性能 Roofline 模型中通过累加矩阵大小（**MN + MK + KN**）来计算 GEMM 访问字节数以得出算术强度的常规做法提出疑问。
   - 另一位成员澄清说，这是一种建立理论最大值的简化方法，对于具有大容量 **L2 caches** 的新型 GPU 来说是现实的，因为其中一个输入矩阵可能完全装入 L2 缓存。
- **通过自定义项目快速开始 CUDA 学习**：一位用户寻求适合新手的 CUDA 项目，另一位用户建议通过你感兴趣的东西来学习。
   - 建议创建一些需要大量多线程或并行性的项目，例如在不使用库的情况下实现线性代数运算，以模拟流水线（pipelining）的概念。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1358304042102820986)** (2 条消息): 

> `Int4WeightOnlyConfig, torch.compile 加速, 编译单个子模块` 


- **使用 Int4WeightOnlyConfig 的反量化受益于 torch.compile**：一位成员正尝试集成 `Int4WeightOnlyConfig`，并询问是否需要 `torch.compile` 来加速反量化（dequant）过程。
   - 另一位成员建议，可以通过对子模块调用 `torch.compile` 来尝试编译单个子模块。
- **为了效率对子模块进行 torch.compile**：为了仅编译 int4 模块，一位成员建议遍历模型的命名模块（named modules），并在特定的子模块（如 `torch.nn.Linear`）上使用 `torch.compile`。
   - 建议的代码片段如下：
```py
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        setattr(model, n, torch.compile(getattr(model, n)))
```


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1357816398658666696)** (3 条消息): 

> `硅谷见面会, 旧金山见面会, 夏季实习生见面会` 


- **硅谷夏季见面会？**：一位在该地区的实习生询问今年夏天是否会在**硅谷**举行见面会，并表示愿意协助组织。
- **计划于今年晚些时候在旧金山举行见面会**：一位成员确认，目前正计划今年晚些时候在**旧金山**举行见面会，但未提及具体日期。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1358014169403883520)** (37 条消息🔥): 

> `带沙箱代码解释器的 RL 微调, Gemma 3 QAT vs HQQ, Wavespeed AI 推理 API, Vector Sum CUDA Kernel 优化, 使用 Transformer 生成猫和老鼠视频`

- **RL 代码微调工具集展示**：一位成员分享了一个[工具包](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm)，用于通过强化学习（RL）微调编程模型，并配备了本地、零设置的沙箱代码解释器。他们发现，与传统的监督微调（SFT）相比，使用极少的数据和训练时间就取得了非常有前景的结果，并期待将其从 Python 扩展到其他语言，例如 [HIP Script](https://lights0123.com/blog/2025/01/07/hip-script/)。
- **HQQ 量化在 Gemma 3 上优于 QAT**：一位成员评估了 **Gemma 3 12B QAT** 与 **HQQ**，发现 [HQQ](https://x.com/mobicham/status/1908477280029986933) 仅需几秒钟即可完成模型量化，且在更高的 group-size 下表现优于 QAT 版本（AWQ 格式）。凭借 **GemLite bfp16** 的支持，量化后的 Gemma 3 可以运行得更快且没有性能问题。
- **Wavespeed AI 宣传高效推理 API**：[Wavespeed AI](https://wavespeed.ai/) 的 CEO 宣传了他们平台最快、最高效的 AI 图像和视频推理 API，例如带有 **LoRA** 的 **FLUX** 和 **Wan**。他们提供具有竞争力的定制价格，并希望建立双赢模式共同成长。
- **Vector Sum Kernel 达到 SOTA**：一位成员分享了一篇[博客文章](https://veitner.bearblog.dev/making-vector-sum-really-fast/)和[代码](https://github.com/simveit/effective_reduction)，关于在 CUDA 中实现向量求和的 SOTA 性能，达到了理论带宽的 **97.94%**，优于 NVIDIA 的 **CUB**。然而，另一位成员指出，由于隐式 Warp 同步编程（implicit warp-synchronous programming），可能存在潜在的竞态条件，建议使用 `__warp_sync()` 以确保正确性，并参考了 [Independent Thread Scheduling (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#independent-thread-scheduling)。
- **使用 Diffusion Transformers 生成《猫和老鼠》动画**：一个团队完成了一个通过微调 Diffusion Transformer 创建 1 分钟长 **Tom and Jerry** 动画的项目，该项目已被 CVPR 2025 接收，代码已在 [GitHub](https://github.com/test-time-training/ttt-video-dit) 上发布。该模型利用预训练 Transformer 中的 **Test-Time Training (TTT) 层**，使其能够根据文本分镜生成连贯的视频，表现优于 **Mamba 2** 等基准模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mobicham/status/1908477280029986933">来自 mobicham (@mobicham) 的推文</a>：我对 Gemma 3 12B 的 QAT 与 HQQ 进行了评估。HQQ 只需几秒钟即可完成模型量化，并且在采用更高 group-size 的情况下性能优于 QAT 版本（AWQ 格式）。配合 GemLite bfp16 支持，...</li><li><a href="https://wavespeed.ai/">WaveSpeedAI - 加速 AI 图像和视频生成的终极 API</a>：加速 AI 图像和视频生成的终极 API</li><li><a href="https://x.com/panthaliaxyz/status/1909342585505669228">来自 Panthalia (@panthaliaxyz) 的推文</a>：Panthalia：去中心化计算原语。在点对点计算上安全、轻松地训练 ML 模型的平台。候补名单现已开放。</li><li><a href="https://docs.panthalia.com/gradient-compression-algorithm">Panthalia 梯度压缩算法 | Panthalia</a>：本文档详细介绍了 Panthalia 中使用的基于 DCT 的梯度压缩算法。该算法旨在高效压缩从节点发送到 c... 的梯度。</li><li><a href="https://github.com/pauleonix">pauleonix - 概览</a>：计算科学与工程博士生，研究用于稀疏线性问题的 GPU 加速预条件子和求解器；物理学硕士。- pauleonix</li><li><a href="https://chromewebstore.google.com/detail/MyLMArena/dcmbcmdhllblkndablelimnifmbpimae">MyLMArena - Chrome 网上应用店</a>：使用 MyLMArena 通过 ELO 评分跟踪您的个人 LLM 偏好。</li><li><a href="https://github.com/huggingface/transformers/issues/31474#issuecomment-2198023128">为 heads 和 embeddings 提供量化支持 · Issue #31474 · huggingface/transformers</a>：功能请求。你好！我最近一直在研究 LLM 量化（这篇论文），并注意到在使用 1-2 bit 量化的 LLM 时出现的一个潜在重要问题。问题描述...</li><li><a href="https://test-time-training.github.io/video-dit/">通过 Test-Time Training 生成一分钟视频</a>：一种使用 Test-Time Training (TTT) 层从文本生成连贯的一分钟视频的新方法。</li><li><a href="https://github.com/test-time-training/ttt-video-dit">GitHub - test-time-training/ttt-video-dit</a>：通过在 GitHub 上创建账号来为 test-time-training/ttt-video-dit 的开发做出贡献。</li><li><a href="https://x.com/karansdalal/status/1909312851795411093">来自 Karan Dalal (@karansdalal) 的推文</a>：今天，我们发布了一篇新论文——《通过 Test-Time Training 生成一分钟视频》。我们在预训练的 Transformer 中加入 TTT 层，并对其进行微调，以生成一分钟的《猫和老鼠》动画...</li><li><a href="https://veitner.bearblog.dev/making-vector-sum-really-fast/">让向量求和变得飞快</a>：在这篇博文中，我们想简要描述如何针对向量 reduction 任务实现 SOTA 性能，即我们的程序应该执行以下操作：...</li><li><a href="https://github.com/simveit/effective_reduction">GitHub - simveit/effective_reduction：逐步改进 reduction kernel</a>：逐步改进 reduction kernel。通过在 GitHub 上创建账号来为 simveit/effective_reduction 的开发做出贡献。</li><li><a href="https://github.com/pranjalssh/fast.cu/tree/main">GitHub - pranjalssh/fast.cu：从零开始编写的最快 kernel</a>：从零开始编写的最快 kernel。通过在 GitHub 上创建账号来为 pranjalssh/fast.cu 的开发做出贡献。
</li>
</ul>

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1357981342557536316)** (18 messages🔥): 

> `Curriculum Learning for Reasoning, Llama 3 vs Qwen 2.5, Dream 7B Diffusion Model, Llama 4 Maverick coding, Claude Think Tool` 


- **课程学习诱发推理能力 (Curriculum Learning Elicits Reasoning)**：一位成员正在尝试通过[课程学习](https://arxiv.org/html/2503.01307v1#S4)在较弱的 LLM（如 **Llama-3.2-3B**）中诱发推理行为。通过使用较简单的推理任务并逐渐增加难度，在不进行 SFT 的情况下对模型进行引导。
   - 另一位成员提到，已有用户在 RG 上进行了课程学习的相关工作，并发现其结果优于没有课程学习的相同任务，主分支的 `training/` 目录支持了这一结论。
- **Qwen 2.5 在训练中胜过 Llama 3.2**：成员们大多选择使用 **Qwen 2.5 3B** 而非 **Llama 3.2 3B**，因为 **Qwen** 在推理训练方面似乎更容易。
   - 这与“**4 Habits**”论文的发现一致，在该论文中，Llama 3.2 在没有预先进行 SFT 的情况下，在回溯（backtracking）和子目标设定（sub-goal setting）方面表现挣扎。
- **Dream 7B 扩散推理**：**Dream 7B**（[HKU 博客文章](https://hkunlp.github.io/blog/2025/dream/)）是一个基于 Diffusion 的 LLM，在频道关注的这类问题上表现出了非常好的效果，这使其成为 Gym 训练的极佳候选对象，尤其是在数独（sudoku）任务上。
   - Dream 7B 持续大幅领先现有的 Diffusion 语言模型，并在通用、数学和编程能力上达到或超过了同等规模的顶尖自回归（AR）语言模型。
- **Llama 4 Maverick Aider 分数公布**：**Llama 4 Maverick** 在 [Aider 多语言编程基准测试 (polyglot coding benchmark)](https://aider.chat/docs/leaderboards/) 中得分 **16%**。
   - 这一消息引用自 X 上的一条讨论编程基准测试的推文。
- **Claude 通过工具使用进行思考**：一位成员分享了 [Anthropic 的 Claude Think Tool](https://www.anthropic.com/engineering/claude-think-tool) 链接。
   - 目前尚未具体讨论这与 Reasoning Gym 的关联。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hkunlp.github.io/blog/2025/dream/">Dream 7B | HKU NLP Group </a>：未找到描述</li><li><a href="https://x.com/paulgauthier/status/1908976568879476843">Paul Gauthier (@paulgauthier) 的推文</a>：Llama 4 Maverick 在 aider 多语言编程基准测试中得分 16%。https://aider.chat/docs/leaderboards/</li><li><a href="https://arxiv.org/html/2503.01307v1#S4,">Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1358939166250893482)** (3 messages): 

> `Deepseek communication library, NVSHMEM and UVA, Peer-to-peer GPU communication` 


- **Deepseek 利用 NVSHMEM 库**：**Deepseek** 通信库基于 NVIDIA 的 **NVSHMEM** 库构建，可实现高性能通信。
   - 一位成员询问 **NVSHMEM** 是否利用 **Unified Virtual Addressing (UVA)** 进行节点内通信，以促进对通过 NVlink 连接的远程 GPU 中存储的数据进行点对点（P2P）加载/存储（loads/stores）。
- **NVSHMEM UVA 在 GPU 间通信中的应用**：一位成员正在咨询关于 **NVSHMEM** 及其在 GPU 间通信中使用 **Unified Virtual Addressing (UVA)** 的情况。
   - 具体而言，他们想知道 **UVA** 是否支持对存储在远程 GPU（通过 NVlink 等连接）中的数据进行点对点加载/存储。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

leikowo: 有什么办法可以实现 PTX Torch 扩展（而不是带有内联 PTX 的 CUDA）吗？
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1358719735528816732)** (24 messages🔥): 

> `matmul Leaderboard submissions, vectoradd Benchmark Submissions, Modal Runners success, grayscale Leaderboard submissions` 


- **Modal 运行器交付 Matmul 杰作**：在 **H100, A100, T4, L4** GPU 上使用 Modal 运行器进行的多次 `matmul` 基准测试排行榜提交均告成功，ID 范围从 **3440** 到 **3453**。
- **Vectoradd 在 L4 上通过 Modal 取得胜利**：在 **L4** GPU 上使用 Modal 运行器进行的多次 `vectoradd` 基准测试提交成功，包括 ID 从 **3464** 到 **3506** 的提交。
- **Grayscale 挑战获得绿灯**：使用 Modal 运行器在 **A100, H100, L4, T4** GPU 上进行的 `grayscale` 基准测试的一个测试提交（ID **3447**）和排行榜提交（ID **3503**）均告成功。


  

---

### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1358059938156642304)** (5 messages): 

> `libsanitizer-collection.so, compute-sanitizer, LD_LIBRARY_PATH` 


- **故障排除者寻求 `libsanitizer-collection.so` 解决方案**：一位成员在为 i8mm/gpu_blm 运行 `./grading test` 期间执行 `compute-sanitizer` 时，遇到了评分程序找不到 `libsanitizer-collection.so` 的问题。
   - 他们尝试根据搜索结果设置 `LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/compute-sanitizer`，但没有效果。
- **i8mm 的 Compute Sanitizer 错误**：一位成员报告了一个 `compute-sanitizer` 错误，系统提示 *Unable to find injection library libsanitizer-collection.so*。
   - 该错误发生在通过命令 `compute-sanitizer --tool memcheck` 对 `i8mm` 进行测试运行期间。
- **似曾相识的调试经历**：另一位成员回想起之前遇到过 `libsanitizer-collection.so` 的问题。
   - 他们表示不太记得当时的解决方案是什么了。


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1358880301543329923)** (2 messages): 

> `Leaderboard Units, Nanos vs Millis, Discord Cluster Manager` 


- **排行榜显示单位冲突！**：一位用户注意到排行榜的时间单位存在差异，[网页排行榜](https://gpu-mode.github.io/discord-cluster-manager/)显示的是 **nanoseconds**，而 **Discord 排行榜**显示的是 **milliseconds**。
   - 一位成员回复称，已准备好一个*新的排行榜网站*，它会*转换为最佳单位以提高清晰度*。
- **新排行榜网站即将上线**：一位成员宣布他们准备了一个*新的排行榜网站*，并且确实会*转换为最佳单位以提高清晰度*。
   - 原始排行榜网站的差异在于 [网页排行榜](https://gpu-mode.github.io/discord-cluster-manager/) 显示 **nanoseconds**，而 **Discord 排行榜**显示 **milliseconds**。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1358876526145179689)** (2 messages): 

> `Local LLM inference, Fine-tuning, GPU selection, L40 vs A100, Quantization` 


- **为组织构建本地 LLM 设备**：成员们正在考虑为组织的 LLM 任务（如**摘要、聊天机器人和文本生成**）构建小型设备，并探讨了 **L40** 或 **A100** GPU 等选项。
   - 主要关注点是优化 **4-bit 和 8-bit 模型推理以及潜在的 fine-tuning**，同时考虑了价格因素（比美国价格高出 5-10%）。
- **L40 性能不佳之谜**：尽管理论上 L40 更适合 **4-bit 量化的 Llama 3 70b**，但在通过 vLLM 进行单用户请求时仅达到 **30-35 tok/s**，表现逊于 A100 的在线基准测试。
   - 性能差距可能是由于 **A100 卓越的 DRAM 带宽和 tensor ops 性能**，其速度几乎是 L40 的两倍。
- **探索量化和优化策略**：讨论建议探索 **TensorRT** 和特定的量化格式，以提升 **L40** 的性能。
   - 尽管 **L40** 具有 **FP8** 支持和更大的 **L2 cache**，但在当前设置下，这些优势似乎并未转化为优于 **A100** 的性能。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1357973724795502592)** (14 messages🔥): 

> `Interactive voice mode, Mind maps rollout, Website URL use cases, Commercial scale version of NotebookLM` 


- **交互式语音模式带来启发！**：一位用户表示，**交互式语音模式**是一种引导他们思考创意的有趣方式。
   - 自一月份以来一直尝试建立稳固的 **NotebookLM** 基础后，他们提到现在几乎可以让每段文本都发挥作用，并有信心帮助企业建立针对其特定需求定制的笔记本。
- **思维导图终于上线！**：用户报告称**思维导图功能**已全面推出，部分用户在中间面板看到了该功能，而其他用户仍在等待。
   - 一位用户提到在右侧面板短暂看到过它，随后消失了。
- **音频概览将网站识别为书籍**：一位用户询问了使用网站 URL 的用例，并指出 **Audio Overview** 错误地将网站识别为书籍。
   - 另一位用户建议，来源类型/体裁是根据来源的内容/格式识别的，通过指定它是网站的“自定义”设置重新运行即可解决问题。
- **询问商业版 NotebookLM**：一位用户询问是否存在商业规模版本的 **NotebookLM**，其中数据不属于公共领域，并且可以输入特定的编程或提示词。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1357793377864908963)** (154 条消息🔥🔥): 

> `NotebookLM 的 Discover 功能推出、Gemini 2.5 系列、基于 generative AI 的思维导图演进、YouTube 音频 EQ Chrome 扩展程序、Google Cloud Next 和 Google I/O 活动` 


- **理论化基于图像的思维导图革命**：用户讨论了 **generative AI** 工具如何很快演进思维导图以包含图像，并从 **Tony Buzan** 的原始思维导图中汲取灵感。
   - 成员们对视觉更丰富、信息量更大的思维导图潜力表示兴奋。
- **Discover 功能推出延迟令用户沮丧**：用户对 NotebookLM 中新的 **'Discover Sources'** 功能延迟推出表示沮丧。该功能已持续推出一周多，预计需要长达两周才能完全可用（4 月 1 日宣布）。
   - 该功能承诺通过允许用户直接在 NotebookLM 中创建带有来源的笔记本，从而简化学习和数据库构建，消除在平台外搜索的需求；一位用户甚至分享了 Peter Griffin 的 *'But I want it now'* GIF。
- **NotebookLM 仍在使用 Gemini 2.0；2.5 的可调性预告**：目前，NotebookLM 使用 **Gemini 2.0 Thinking 模型**，尽管在这种情况下它与 **Flash 模型** 相比的有效性仍在评估中。
   - **Gemini 2.5** 已确认是一个模型系列，包括 **Flash** 版本，并且 **2.5 Pro** 很快将支持调优，使开发人员能够调整其“思考”强度。
- **Chrome 扩展程序利用 AI 调节 YouTube 音频**：一位成员创建了一个名为 *EQ for YouTube* 的 **AI 驱动的 Chrome 扩展程序**，允许用户使用 6 段参数均衡器实时操作 YouTube 视频的音频；该扩展程序具有实时频率可视化、内置预设和自定义预设创建功能。
   - [GitHub 仓库](https://github.com/aashishjhaa/eq-for-youtube) 已开放下载。
- **NotebookLM 语言更改说明**：要更改 NotebookLM 中的语言，请使用 URL `https://notebooklm.google.com/?hl=LANGUAGE_CODE`，将 `LANGUAGE_CODE` 替换为所需的语言代码（例如，西班牙语为 `es`）。
   - 虽然团队承认了之前发现的一个翻译 Bug（现已解决），但播客输出目前无法翻译。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1124402182171672732/1357653558140342343.">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement">为互联网之最投票</a>：我刚刚在 The Webby People's Voice Awards 中投票并检查了我的选民登记。</li><li><a href="https://tenor.com/view/peter-griffin-but-i-want-it-now-gif-26307521">Peter Griffin But I Want It Now GIF - Peter Griffin But I Want It Now - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://careers.google.com">Google Careers</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=9">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://notebooklm.google.com/?hl=LANGUAGE_CODE">未找到标题</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=7650624668661580589-EU">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://jamboard.google.com/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/aashishjhaa/eq-for-youtube">GitHub - aashishjhaa/eq-for-youtube: Manipulate the audio of YouTube Video Realtime with 6 Frequency Band</a>：使用 6 个频段实时操作 YouTube 视频音频 - aashishjhaa/eq-for-youtube</li><li><a href="https://aashishjhaa.github.io/eq-for-youtube/">EQ for YouTube</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&r">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en&ref_topic=14775295&sjid=9087543013148016209-NA">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1357941955438448650)** (28 messages🔥): 

> `Nvidia CUDA Python Support, Mojo GenAI, CuTile Programming Model, SIMD vs SIMT, Tenstorrent and Modular` 


- **Nvidia 为 CUDA 增加原生 Python 支持**：一位成员分享了一篇文章链接，[Nvidia Finally Adds Native Python Support to CUDA](https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/)，并询问这是否是“帝国反击战”。
   - 文章讨论了 **Nvidia** 使用 **CuTile** 编程模型进行 GPU 执行的方法，该模型从线程级编程中抽象出来。
- **Mojo 能应对 GenAI 吗？**：一位成员想知道 **Mojo** 是否已经具备开发 **GenAI** 或 **Inference.ai** 的能力。
   - 这引发了关于 **Mojo** 在 **Generative AI** 领域当前能力和潜力的讨论。
- **CuTile 编程模型遭到质疑**：一位成员对 **Nvidia** 的 **CuTile** 编程模型持保留意见，认为这是一种高级抽象，剥夺了编写 **GPU code** 的乐趣。
   - 他们表示：*they are taking the fun out of writing gpu code*。
- **SIMD vs SIMT**：一位成员正在开发一个概念验证（PoC）模型，并指出通过典型的线程模型来观察现代并行计算已经不太合理。
   - 讨论围绕着将 **SM** 暴露为带有掩码（masking）的大型 **SIMD core**，以及考虑到硬件灵活性和潜在限制，**SIMD** 还是 **SIMT** 更合适。
- **Tenstorrent 软件栈**：一位成员建议 **Tenstorrent** 应该使用 **Modular** 的软件栈，但另一位成员指出 **Tenstorrent** 的驱动程序非常容易适配和使用。
   - 他们表示：*他们的驱动程序非常容易适配和使用，因此虽然有效利用其架构可能需要一些微调，但仅仅让程序在上面运行似乎几乎是轻而易举的*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://thenewstack.io/nvidia-finally-adds-native-python-support-to-cuda/">NVIDIA Finally Adds Native Python Support to CUDA</a>：多年来，NVIDIA 用于 GPU 的 CUDA 软件工具包一直没有原生 Python 支持。但现在情况发生了变化。</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc25-s72449/">1,001 Ways to Write CUDA Kernels in Python | GTC 25 2025 | NVIDIA On-Demand</a>：你必须编写一个 CUDA kernel。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1357838483485364344)** (85 messages🔥🔥): 

> `Auto Lowering, MLIR Interpreter stress test, Implicit ctor hack, Mojo language spec, Mojo implicit copies` 


- **实现自动 Lowering**：一位成员发现，在添加来自不同尺度的值时可以实现自动 Lowering，并分享了[代码链接](https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106)。
   - 该成员提到：*这可能是我目前最“诅咒”的作品*。
- **MLIR Interpreter 面临压力测试**：一位成员评论说，时间间隔库可能会变成对 **MLIR interpreter** 的压力测试。
   - 另一位成员补充说，有些功能没有按预期工作，但可以通过 **implicit ctor hack**（隐式构造函数黑科技）来修复。
- **Mojo 规范辩论升温**：围绕 **Mojo** 是否会有规范（spec）展开了讨论，一些人认为规范能赋予语言责任感和成熟度，并参考了 **Carbon** 的设计原则。
   - 其他人则反驳说，**Mojo** 的设计与 **MAX** 的需求紧密耦合，制定规范会减慢开发速度。一位成员指出，Chris Lattner 将 **OpenCL** 的失败归咎于“委员会设计”（design by committee）。
- **Mojo 的复制语义得到澄清**：一位成员询问 **Mojo** 的隐式复制是否使用写时复制（CoW）。
   - 另一位成员澄清说：*从语义上讲，总是复制；从优化上讲，许多复制被转化为 move 或被完全消除（原地操作）。虽然这发生在编译时，但 CoW 是一个运行时概念*。
- **ChatGPT 的 Mojo 能力受到质疑**：一位成员询问 **ChatGPT** 或其替代品是否足以将一个大型 Python 项目重写为 **Mojo**。
   - 另一位成员回答说：*ChatGPT 并不擅长任何新语言*。



**提到的链接**：<a href="https://github.com/bgreni/ChronoFlare/blob/main/chronoflare/__init__.mojo#L90-L106">ChronoFlare/chronoflare/__init__.mojo at main · bgreni/ChronoFlare</a>：一个用 Mojo 编写的时间间隔库。可以通过在 GitHub 上创建账户来为 ChronoFlare 的开发做出贡献。

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1357800414619111616)** (54 messages🔥): 

> `Nomic Embed Text V2, GPT4All 发布节奏, Llama 4 发布, 用于多模态任务的 ComfyUI, 语义分块 (Semantic chunking)` 


- **Nomic Embed Text V2 正集成至 Llama.cpp**：一名成员分享了一个 [GitHub Pull Request 链接](https://github.com/ggml-org/llama.cpp/pull/12466)，显示 **Llama.cpp** 正在努力集成 **Nomic Embed Text V2**，该模型采用 Mixture-of-Experts (MoE) 架构，用于多语言 Embeddings。
   - 另一位成员表示 *“一切都取决于 Llama.cpp”*，并希望支持 Mistral Small 3.1 多模态。
- **GPT4All 的沉默令用户困扰**：成员们注意到核心开发者近期保持沉默，一位成员提到这 *“导致了对该应用和社区贡献的不确定性”*。
   - 该成员建议对于一个开源项目来说，这可能不是一个好的策略，但也表示 *“当他们打破沉默时，通常会大展身手”*。
- **Llama 4 已经发布，但它是最强的吗？**：Meta 于 2025 年 4 月 5 日发布了 **Llama 4**（[公告](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)），其特色是 **Llama 4 Scout**，这是一个拥有 17B 参数、16 个专家（experts）以及 **10M Token** 上下文窗口的模型。
   - 尽管一些用户对发布感到兴奋，但其他人表示 *“这有点令人失望”*，认为 **DeepSeek** 和 **Qwen** 需要迎头赶上，而另一位用户指出最大的模型拥有 **2 Trillion (2万亿) 参数**。
- **ComfyUI 不仅仅是图像生成的漂亮界面**：成员们讨论了 **ComfyUI** 的广泛功能，指出 *“如果你有足够的节点，你可以用 Comfy 做很多事情”*，包括图像和音频标注（captioning）。
   - 另一位成员提到了视频处理的可能性，并描述了使用命令行工具进行视觉模型分析的方法。
- **用于美味 RAG 的语义分块 (Semantic chunking) 服务器方案**：一位成员分享了一个使用 FastAPI 实现的 [语义分块服务器链接](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html)。
   - 该成员还分享了一个 [curl 命令示例](https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html)，用于向分块端点发送请求，展示了如何设置 `max_tokens` 和 `overlap` 等参数。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>：未找到描述</li><li><a href="https://ai.meta.com/blog/llama-4-multimodal-intelligence/">无标题</a>：未找到描述</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html">clipboard</a>：未找到描述</li><li><a href="https://gnu.support/files/tmp/clipboard-2025-04-07-22-50-50.html">clipboard</a>：未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12466">manyoso 提交的基于 Mixture-of-Experts (MoE) 架构的 Nomic Embed Text V2 · Pull Request #12466 · ggml-org/llama.cpp</a>：添加了支持多语言 Embeddings 的基于 MoE 的 Embedding 模型。根据超参数检测（MoE 层）选择架构变体。删除了不必要的子类初始化检查...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1358121947153825845)** (3 messages): 

> `MCP Servers, 全栈 Agent 应用, LlamaParse Layout Agent` 


- **MCP Servers 获得 CLI 工具支持**：由 @MarcusSchiesser 开发的一个工具可以让你通过单个 CLI 界面轻松地 **发现、安装、配置和移除新的 MCP Servers**，支持 Claude、@cursor_ai 和 @windsurf_ai，详见 [此处](https://t.co/zmqxBwNKvQ)。
   - 目前已有数百个官方 MCP Servers。
- **使用 Create Llama 构建全栈 Agent**：**create-llama CLI 工具** 允许你通过单行代码启动一个带有 **FastAPI 后端和 Next.js 前端** 的 Web 应用程序，仅创建 5 个源文件，详见 [此处](https://t.co/TuZ0O0nMfe)。
   - 这旨在快速启动如深度研究（deep research）之类的 Agent 应用开发。
- **LlamaParse 发布 Layout Agent**：**LlamaParse** 中全新的 **Layout Agent** 为你提供一流的文档解析和提取功能，并带有精确的视觉引用，使用 SOTA VLM 模型来检测页面上的所有区块并动态适应。
   - 这个新 Agent 可以动态适应，详见 [此处](https://t.co/2WRRXxIRa1)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1357807162138689607)** (46 条消息🔥): 

> `将 Workflow 作为 Tool 使用、带有 Supervisor 模式的多 Agent 系统、使用 LlamaParse 的 RAG 系统、DocumentSummaryIndex 的可扩展性问题、发生异常时 Tool 的重试机制` 


- **使用 FunctionTool 将 Workflow 包装为 Tool**：要将 **Workflow** 转换为 **Tool**，可以使用 `FunctionTool` 来包装该工作流，从而控制其名称、描述、输入注解和返回值。
   - 一位成员建议的代码片段：
```python
async def tool_fn(...):
  """Some helpful description"""
  result = await workflow.run(...)
  return str(result)

tool = FunctionTool.from_defaults(tool_fn)
```
- **Agent 移交（Handoffs）优于 Supervisor 模式**：在构建多 Agent 系统时，让 Agent 根据需要相互移交任务比使用 Supervisor 模式更健壮，后者更容易出错。
   - 分享了一个 [GitHub 仓库](https://github.com/run-llama/multi-agent-concierge) 作为 Supervisor 模式实现的示例。
- **使用 Vector Store Index 替代 Document Summary Index**：`DocumentSummaryIndex` 可能存在可扩展性问题；建议通过使用普通的 `VectorStoreIndex` 来复制其功能，具体方法是总结文档、使用参考 ID 进行索引，并在检索期间将摘要节点替换为原始文档。
   - 使用 `load_index_from_storage` 时，索引存储会被加载到内存中，随着摄入文档数量的增加，这会导致延迟。
- **Context 的 State 会前置到 user_msg**：为了避免在用户消息中前置状态内容，应避免在 context 中使用 `state` 键，而应将 Tool 之间的数据放在 context 的其他位置。
   - 建议改用 `ctx.set("some_key", "some_val")` 和 `ctx.get("some_key")`。
- **实现 Text-to-SQL 查询引擎 Tool**：在为 Agent 实现 Text-to-SQL 查询引擎 Tool 时，如果只有少量表，则没必要创建表描述索引并执行向量查询。
   - 在表数量较少的情况下，可以跳过索引和向量搜索部分以获得更好的性能。



**提到的链接**：<a href="https://github.com/run-llama/multi-agent-concierge">GitHub - run-llama/multi-agent-concierge: An example of multi-agent orchestration with llama-index</a>：使用 llama-index 进行多 Agent 编排的示例 - run-llama/multi-agent-concierge

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1357901550936391761)** (16 messages🔥): 

> `torch-geometric for tinygrad, Llama 4 10M context limitations, fast pattern matcher bounty, UOps generation, tinygrad YouTube video` 


- **Tinygraph: 适配 Tinygrad 的 Torch-geometric？**: 一位成员询问，考虑到 tinygrad 现有的 torch 接口，在 tinygrad 中创建一个类似于 **torch-geometric** 的图机器学习（Graph ML）模块是否可行。
   - 他们质疑追求这样一个模块是否 *"有用"*。
- **Llama 4 的长上下文可能没那么好**: 一位用户分享了一篇 [推文](https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA)，声称 **Llama 4** 宣称的 **10M context** 是 *"虚拟的"*，因为模型并没有在超过 **256k tokens** 的 prompt 上进行训练。
   - 该推文还指出，即使是低于 **256k tokens** 的问题也可能产生低质量的输出，因为获取高质量训练样本非常困难，而且拥有 **2T parameters** 的最大模型 *"并没有击败 SOTA 推理模型"*。
- **$2000 快速模式匹配器悬赏已发布**: 一位成员强调了一个针对 tinygrad 快速模式匹配器（fast pattern matcher）的 [$2000 悬赏](https://github.com/tinygrad/tinygrad/pull/9737)。
   - 拟议的解决方案涉及为匹配函数引入 **JIT**，以避免函数调用和 dict 复制。
- **减少 UOps 以加速重写**: 有建议指出 tinygrad 有时会生成比实际需要更多的 **UOps**，从而增加了重写的成本。
   - 一位成员询问，如果最初生成较少的 **UOps**（即使它们稍后会被优化为相同的结果）需要牺牲几行代码，这是否可以接受。
- **分享了 Tinygrad YouTube 视频**: 一位成员分享了一个 [YouTube 视频](https://youtu.be/fWiieyG2zes?si=3CzwFRfJmFQhqUZvY) 链接。
   - 未提供更多细节。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/burkov/status/1908666701362978979?s=46&t=fQTa8qEB1aBjOkD2ftKKbA">来自 Andriy Burkov (@burkov) 的推文</a>: 我将为你节省阅读关于 Llama 4 的时间。宣称的 10M 上下文是虚拟的，因为没有模型是在超过 256k tokens 的 prompt 上训练的。这意味着如果你发送超过 256k tokens 给它，...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9737">geohot 提交的快速模式匹配器初次尝试 [pr] · Pull Request #9737 · tinygrad/tinygrad</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1357958444744708307)** (24 messages🔥): 

> `Tensor and SimpleMathTrait inheritance, Mesozoic tinygrad tutorials issues, METAL sync issue, AMD and BEAM issues` 


- **关于 Tensor 继承 SimpleMathTrait 的辩论**: 讨论围绕 **Tensor** 是否应该继承自 `SimpleMathTrait` 展开，因为 **Tensor** 重新实现了 `SimpleMathTrait` 提供的每个方法，而没有使用 `.alu()` 函数。
   - 有人指出，之前关于重构 **Tensor** 以继承 `MathTrait` 的悬赏由于提交质量差而被取消，一些人建议 **Tensor** 可能不需要继承其中任何一个。
- **Colab CUDA Bug 导致 Mesozoic Tinygrad 教程出现问题**: 一位用户在 Colab 中运行 Mesozoic tinygrad 教程的代码时遇到问题，促使其他人请求错误消息以便调试。
   - 该问题被确定为与不兼容的 CUDA 和驱动程序版本相关的 Colab bug，建议的解决方法包括使用特定的 `apt` 命令删除并安装兼容版本；同时建议暂时使用 CPU 设备。
- **METAL 分片行为导致意外结果**: 一位成员在尝试重现 METAL 同步问题的最小示例时遇到了意外的分片（sharding）行为，怀疑从 **METAL:1** 到 **CPU** 的 **COPY** 可能在从 **METAL** 到 **METAL:1** 的 **XFER** 完成之前就执行了。
   - DEBUG 输出似乎显示时间线是在提交到 GPU 命令队列时添加 **XFER** 的，而不是在它结束时。
- **AMD 和 BEAM 导致 AssertionError**: 一位用户在运行 **BEAM=2** 和 **AMD=1** 时遇到了 `AssertionError`，这似乎与在 `if __name__ == "__main__"` 块之外打开设备有关。
   - 设置 **PARALLEL=0** 或确保在 `if __name__ == "__main__"` 块内打开设备解决了该问题。


  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1357996936690401351)** (19 messages🔥): 

> `MCP 与 Command-A 模型，Cohere Tool Use，Cohere Scholars Program，活动录影` 


- **探讨 MCP 与 Command-A 模型的使用**: 一位成员询问关于将 **MCP (Modular Conversational Platform)** 与 **Command-A 模型** 结合使用的问题，并建议通过 **OpenAI SDK** 应该可以实现。
   - 另一位成员表示同意，称 *没有理由不能正常工作*。
- **Cohere Tool Use 能力详情**: 一位成员分享了 [Cohere Tool Use Overview](https://docs.cohere.com/docs/tool-use-overview)，强调了其将 **Command 系列模型** 连接到外部工具（如 **搜索引擎、API 和数据库**）的能力。
   - 文档还提到 **Command-A** 支持 Tool Use，这与 **MCP** 旨在实现的目标类似。
- **分享 Cohere Scholars Program 详情**: 一位成员询问了 **Cohere Scholars Program** 的要求，特别是是否接受之前发表过的论文。
   - 一位社区成员通过链接申请表 ([https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw)) 进行了回复，并澄清虽然之前的研究经验很有帮助，但并非硬性要求。
- **关于活动录影的查询**: 一位成员询问 Cohere 的活动是否有录影，因为他们对活动很感兴趣但无法参加直播。
   - 在提供的上下文中，该问题尚未得到解答。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw">Form</a>: 未找到描述</li><li><a href="https://docs.cohere.com/docs/tool-use-overview">Basic usage of tool use (function calling) — Cohere</a>: Cohere Tool Use 能力概览，支持开发者构建 Agent 工作流 (API v2)。</li><li><a href="https://cohere.com/research/scholars-program">Cohere For AI - Scholars Program </a>: C4AI Scholars Program 提供了与知名研究员和工程师合作的机会，共同探索未知领域。 
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1358869670899351734)** (1 messages): 

> `Aya Vision, 多语言多模态模型, 开源权重模型` 


- **Aya Vision 团队举办技术讲座和 AMA**: **Aya Vision**（一个多语言多模态开源权重模型）背后的核心团队正在举办技术讲座，随后在 <t:1744383600:F> 进行 AMA。
   - 参会者可以获得关于团队如何构建其首个多模态模型以及所获经验教训的独家见解；活动由高级研究科学家 <@787403823982313533> 主持，核心研究和工程团队成员将进行闪电演讲；更多详情请见 [Discord Event](https://discord.gg/sH3SSRp2?event=1358866070315860068)。
- **多语言模型 Aya 期待社区反馈**: 团队安排了 Ask Me Anything (AMA) 环节，以便社区直接与创作者交流。
   - 问题可以涵盖从模型架构到未来路线图的任何内容。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1358314800660217906)** (5 messages): 

> `Notion 连接器, 用于 Notion 的向量数据库` 


- **Slack 应用在 Notion 集成方面遇到困难**: 一位成员就 **Slack app** 与公司 **Notion wiki 数据库** 集成的可行解决方案寻求帮助。
- **建议使用向量数据库增强 Notion**: 由于 **Notion** 的搜索 API 表现欠佳，一位成员建议使用 **向量数据库 (Vector DB)**。
   - 未给出具体推荐，但提到 Cohere 模型可以与所有向量数据库良好协作。


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1358662283135291483)** (3 messages): 

> `问候` 


- **用户互相问候**: 两位用户在 「🤖」bot-cmd 频道互相问候，使用了 "hey" 和 "sup"。
   - Cmd R Bot 对问候做出了回应。
- **机器人回应问候**: 一个机器人回应了用户的问候。
   - 机器人使用了随意的 "sup" 来确认互动。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1357861706201694339)** (22 messages🔥): 

> `超时崩溃修复, NeMo 弹性训练, RL 工作流, DeepSpeed 集成` 


- **修复 **Timeout Crash** Bug**: 一名成员修复了一个与**超时崩溃 (timeout crashes)**相关的 Bug，并在[此 Pull Request](https://github.com/pytorch/torchtune/pull/2560) 中创建了 `torchtune.utils._tensor_utils.py`，其中包含对 `torch.split` 的封装。
   - 他们建议单独合并 Tensor 工具类，然后与另一个分支同步以处理任何冲突。
- ****NeMo** 应对弹性训练**: 一名成员参加了关于弹性训练的 **NeMo** 会议，重点介绍了**容错 (fault tolerance)**、**掉队者检测 (straggler detection)**、**异步 Checkpointing**、**抢占 (preemption)**、**进程内重启 (in-process restart)**、**静默数据损坏检测 (silent data corruption detection)** 以及**本地 Checkpointing**等特性。
   - 并非所有特性都已实现，有些仅在计划中；该成员提出重新观看会议并展示 **torchtune** 与 **NeMo** 在弹性方面的详细对比。
- **RL 工作流、数据标准格式和 Prompt**: 一名成员讨论了 **RL 工作流**、数据格式和 Prompt 模板的复杂性，建议采用关注点分离的方法，将数据转换与 Prompt 创建解耦，从而允许在不同数据集之间复用相同的模板。
   - 该成员建议将其分解为一个将数据转换为标准格式的组件，以及另一个将此标准格式转换为带有 Prompt 的实际字符串的组件。
- **Torchtune 的 DeepSpeed 后端？**: 一名成员询问是否可以将 **DeepSpeed** 作为后端集成到 **torchtune** 中，并创建了[一个 Issue](https://github.com/pytorch/torchtune/issues/2569) 来讨论这种可能性。
   - 另一名成员询问了更多背景信息，并指出 **FSDP** 已经支持 **DeepSpeed** 的所有分片 (sharding) 选项。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/generated/torchtune.models.llama3.llama3_tokenizer.html#torchtune.models.llama3.llama3_tokenizer.">llama3_tokenizer &mdash; torchtune 0.6 文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/issues/2569">torchtune 中的 deepspeed 后端？ · Issue #2569 · pytorch/torchtune</a>: 如果不超出范围，拥有这个可选性会很好——我很乐意研究它</li><li><a href="https://github.com/pytorch/torchtune/pull/2560">fix: 修复由 chunked_output 长度导致的超时崩溃，由 bogdansalyp 提交 · Pull Request #2560 · pytorch/torchtune</a>: 背景：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue - closes #25...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

pjbontrager: 你觉得他们是用 AI 来编写那个滚动实时更新的图表吗？
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1358886256062500924)** (1 messages): 

> `AI4Math, 定理证明, 自动形式化, 形式化数学推理, 语言模型` 


- **Kaiyu Yang 演讲：自动形式化与定理证明**: Kaiyu Yang 将在 [PDT 时间今天下午 4 点](https://www.youtube.com/live/cLhWEyMQ4mQ)发表题为“用于自动形式化和定理证明的语言模型”的演讲。
   - 演讲将涵盖使用 **LLM** 进行形式化数学推理的基础知识，重点关注**定理证明 (theorem proving)**和**自动形式化 (autoformalization)**。
- **AI4Math 对 AI 驱动的系统设计至关重要**: **数学人工智能 (AI4Math)** 在智力上引人入胜，且对于 AI 驱动的系统设计与验证至关重要，目前已有大量努力在借鉴 NLP 中的技术。
   - 此次演讲探讨了基于**证明助手 (proof assistants)**等形式化系统的形式化数学推理，这些系统可以验证推理的正确性并提供自动反馈。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1358394025971028008)** (4 messages): 

> `LLM Agents MOOC, AgentX 竞赛, 课程测验` 


- **分享 LLM Agents MOOC 链接**: 一名成员询问 **LLM Agents MOOC** 的链接，另一名成员分享了[该链接](https://llmagents-learning.org/sp25)。
- **AgentX 竞赛报名**: 工作人员分享了 **AgentX 竞赛**的报名地址，可在[此处](https://rdi.berkeley.edu/agentx/)访问。
- **课程测验延迟**: 一名成员询问上周缺失的测验。
   - 一名工作人员为忘记发布测验道歉，并提到测验将在几分钟内上线。



**提到的链接**: <a href="https://llmagents-learning.org/sp25">高级大语言模型 Agent MOOC</a>: MOOC, 2025 春季

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1357846993761992804)** (4 messages): 

> `asyncio support, full-async fork of dspy, reasons to migrate` 


- **Asyncio 支持：DSPy 会支持异步吗？**：一位成员询问了为通用 DSPy 调用添加 **asyncio 支持**的计划。
   - 他们提到最初使用 *litelm*，随后发展到 DSPy 优化，并对原生的 DSPy **async** 能力表示出兴趣。
- **全异步分支面临弃用？**：一位成员维护了一个 [DSPy 的真正全异步分支](https://github.com/swiftdevil/dspy/tree/full_async) 数月，但目前正准备从 DSPy 迁移走。
   - 如果社区有兴趣，他们愿意继续合并上游变更，否则将放弃维护。
- **迁移原因与异步 DSPy 的益处**：成员们对迁移出 DSPy 的原因以及迁移到哪个工具表示好奇。
   - 一位成员询问了拥有 **全异步 DSPy** 的优势，并建议将相关功能合并到主仓库中。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1358004519241388083)** (3 messages): 

> `GitHub PR Review, Phi-4 Support` 


- **GitHub PR 受到关注**：一位成员提到正在评审一个 [GitHub Pull Request](https://github.com)，并在平台上留下了进一步讨论的意见。
   - 作者对评审表示感谢，认可了其中的付出，并表示需要根据反馈重新运行流程。
- **考虑支持 Phi-4 系列**：一位成员正考虑将功能扩展到 **Phi-4-mini** 和 **Phi-4**，尽管它们尚未得到官方支持。
   - 这表明正在努力扩大兼容性，超出最初预定的范围，从而可能增强该工具的吸引力。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1358918518338355270)** (1 messages): 

> `Manifold Research, Multimodal AI, Self-assembling space robotics, Robotic metacognition, Community Research Call` 


- **Manifold Research 主持第 4 次社区研究电话会议**：Manifold Research Group 将于本周六（太平洋标准时间 4/12 上午 9 点）主持 **第 4 次社区研究电话会议 (CRC #4)**，涵盖他们在 **Multimodal AI、自组装空间机器人和机器人元认知** 方面的最新工作。
   - 感兴趣的人员可以在[此处](https://lu.ma/wlne416w)注册，参加这个专注于前沿科学、开放且协作的活动。
- **CRC 是 Manifold 的基石活动**：**Community Research Calls (CRCs)** 是 Manifold 的核心活动，他们在会上展示其研究组合中的重大进展。
   - 这些互动环节提供有关正在进行的项目的全面更新，介绍新的研究方向，并强调合作机会。
- **CRC #4 议程公布**：**CRC #4** 的议程包括 **Generalist Multimodality Research**、**空间机器人进展**、**元认知研究进展** 以及 **新兴研究方向** 的更新。
   - 活动将涵盖其 **MultiNet framework** 的最新突破和技术进展、**Self-Assembling Swarm** 技术的发展、**VLM Calibration** 方法论的更新，以及一项新型机器人元认知计划的介绍。



**提及的链接**：<a href="https://lu.ma/wlne416w">Community Research Call #4 · Zoom · Luma</a>：对通用 AI 模型、自组装空间机器人或机器自我意识感兴趣吗？加入我们的第 4 次社区研究电话会议！Community Research Calls…

  

---


---


{% else %}


> 完整的逐频道详情已因邮件长度限制而截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}