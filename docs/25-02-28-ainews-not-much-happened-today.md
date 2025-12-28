---
companies:
- openai
- anthropic
- perplexity-ai
- deepseek
- scaling01
date: '2025-03-01T03:41:57.593041Z'
description: '**GPT-4.5** 在 Twitter 上引发了褒贬不一的反应。**@karpathy** 指出，尽管他个人更青睐 GPT-4.5
  的创造力和幽默感，但一项民意调查显示用户更倾向于 **GPT-4**。


  像 **@abacaj** 这样的批评者强调了 **GPT-4.5 运行缓慢**的问题，并对其与其他模型相比的实用价值和定价提出了质疑。在性能方面，**GPT-4.5**
  的排名高于 **GPT-4o**，但低于 **o1** 和 **Claude 3.5 Sonnet**。尽管 **Claude 3.7** 在许多任务上的表现优于它，但
  GPT-4.5 因其幽默感和“氛围感 (vibes)”而受到称赞。


  关于 GPT-4.5 规模的推测认为其参数量约为 **5 万亿**。讨论还涉及到了定价差异，例如 **Perplexity Deep Research** 每月为
  20 美元，而 ChatGPT（高阶版）则高达每月 200 美元。此外，像 **Claude 3.7** 这样模型的情商和幽默感也受到了关注。'
id: 5f0db594-436e-4cad-8d4b-2e118c5f2505
models:
- gpt-4.5
- gpt-4
- gpt-4o
- o1
- claude-3.5-sonnet
- claude-3.7
- claude-3-opus
- deepseek-v3
- grok-3
original_slug: ainews-not-much-happened-today-3457
people:
- andrej-karpathy
- jeremyphoward
- abacaj
- stevenheidel
- yuchenj_uw
- aravsrinivas
- dylan522p
- random_walker
title: 今天没发生什么事。
topics:
- model-performance
- humor
- emotional-intelligence
- model-comparison
- pricing
- context-windows
- model-size
- user-experience
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天。**

> 2025年2月27日至2月28日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**221** 个频道，**8236** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**795 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

下文包含了大量关于 GPT 4.5 优缺点的讨论。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**GPT-4.5 模型性能与用户感知**

- **初始用户体验与主观评价**：[@karpathy](https://twitter.com/karpathy/status/1895337579589079434) 发起了一项对比 GPT-4 和 GPT-4.5 的投票，发现 5 个问题中有 4 个用户更倾向于 GPT-4。这令人惊讶，因为 [@karpathy](https://twitter.com/karpathy/status/1895337579589079434) 个人认为 **GPT-4.5 在所有情况下都更好**，并暗示“高品味测试者”可能更青睐 **GPT-4.5 更深层的魅力、创造力和幽默感**。然而，[@jeremyphoward](https://twitter.com/jeremyphoward/status/1895354868342366648) 对 Karpathy 的投票结果做出回应，称用户偏好 GPT-4 的原因是 GPT-4.5 的笨拙感，而非所谓的“高品味”。[@Teknium1](https://twitter.com/Teknium1/status/1895348781367140708) 也对投票结果反应道：“该死，哈哈，肯定是有一些品味极高或极低的人在这里测试，我也不清楚”。[@abacaj](https://twitter.com/abacaj/status/1895516638704754727) 表达了强烈不满，称 **GPT-4.5 需要提高生产力才有用**，否则就是“极其无用”。[@abacaj](https://twitter.com/abacaj/status/1895517803173810560) 还认为，如果 **GPT-4.5 只是一个“高品味”模型**，那它就是在“挥霍投资者的钱”。[@stevenheidel](https://twitter.com/stevenheidel/status/1895541898137456776) 将 **GPT-4.5 的发布比作最初 ChatGPT 带来的兴奋感**，因为人们再次享受到了与 AI 聊天的乐趣。
- **关于速度和实用性的担忧**：[@abacaj](https://twitter.com/abacaj/status/1895309773329027543) 指出 **GPT-4.5 “非常慢”**，且“在 Agent 循环中使用不切实际”，尽管“Prompt 起来很有趣”。[@abacaj](https://twitter.com/abacaj/status/1895310460276351105) 详细说明，在一个中等强度的 Prompt 循环中，**“回答一个问题需要 3 分钟以上”**，认为这“非常不切实际”。[@abacaj](https://twitter.com/abacaj/status/1895311873622581502) 进一步评论说，由于其速度缓慢，**GPT-4.5 “感觉更像是一个研究产物，而不是一个可以部署的真实模型”**。
- **对能力和价值主张的批评**：[@abacaj](https://twitter.com/abacaj/status/1895520453520970054) 批评了这个号称“最大语言模型”所展示的能力，质疑 **使用 SVG 画三角形** 是否就是其亮点。[@abacaj](https://twitter.com/abacaj/status/1895519515204784380) 认为对终端用户的价值增量值得怀疑，并建议 OpenAI 内部将其用于模型 Distillation（蒸馏）。
- **定价与经济可行性**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1895338027041579269) 评论说，考虑到 GPT-4.5 的表现，其 **定价“更加不合理”**。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1895313053606142283) 推测了 **GPT-5 和 o4** 的潜在定价。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1895508117376598330) 强调了 **Perplexity Deep Research 每月 20 美元与 ChatGPT 每月 200 美元的对比**。
- **与其他模型的性能对比**：[@METR_Evals](https://twitter.com/METR_Evals/status/1895381625585967180) 报告称，根据 METR 对早期 Checkpoint 的实验，**GPT-4.5 的表现优于 GPT-4o，但低于 o1 或 Claude 3.5 Sonnet**，并指出其时间跨度（Time Horizon）评分约为 30 分钟。[@dylan522p](https://twitter.com/dylan522p/status/1895557873712972138) 表示 **Claude 3.7 在大多数任务上击败了 GPT 4.5**，但 **GPT 4.5 的“氛围感 (Vibes)”更好**，并且是自 Claude 3 Opus 以来第一个能让他们笑出来的模型，强调幽默感也是智能的一种体现。[@scaling01](https://twitter.com/scaling01/status/1895415262486388906) 推测 **GPT-4.5 的规模可能是“GPT-4o 的 10 倍”**，估计约为 5T 参数。[@Teknium1](https://twitter.com/Teknium1/status/1895520871642783958) 提到 **Grok 的上下文窗口仅为 128k**。[@multimodalart](https://twitter.com/multimodalart/status/1895521321838063837) 分享了 **GPT 4.5 与 Sonnet 3.7、Deepseek V3 和 Grok 3 等非思考型模型的对比评估**。

- **情商 (EQ) 与 “氛围 (Vibes)”**：[@karpathy](https://twitter.com/karpathy/status/1895549465463009309) 在仔细研究了 LLM 的幽默输出后，发现 **Claude 3.7 的幽默感最强**。[@random_walker](https://twitter.com/random_walker/status/1895494391466475684) 认为 **GPT 4.5 的 “EQ” 提升归功于训练后处理 (post-training)，而非参数量**，这表明任何 EQ 差异都是行为层面的，而非能力层面的。[@random_walker](https://twitter.com/random_walker/status/1895499480902013254) 进一步声称，**通过适当的训练后处理，GPT-4o 和 GPT-3.5 可以表现出与 GPT-4.5 类似的 EQ 行为**。[@omarsar0](https://twitter.com/omarsar0/status/1895504181789937964) 建议使用 OpenAI Playground 来对比模型，并观察 **GPT-4.5 “深思熟虑” 的回复**。[@omarsar0](https://twitter.com/omarsar0/status/1895504558669127693) 注意到 **GPT-4.5 通过增加感官描述和思考过程，通常听起来更 “周到”**。[@marktenenholtz](https://twitter.com/marktenenholtz/status/1895316983144685978) 观察到 **Sonnet 3.7 “几乎过于热情”，而 GPT-4.5 “几乎过于恭敬”**。
- **技术细节与训练**：[@sama](https://twitter.com/sama/status/1895490123690922445) 将 GPT-4.5 在 ML 与系统交叉领域的艰巨工作归功于 **@ColinWei11、Yujia Jin 和 @MikhailPavlov5**。[@cloneofsimo](https://twitter.com/cloneofsimo/status/1895319178116243763) 强调 **GPT-4.5 是 “在多个数据中心训练的”，并且 “激进地使用了低精度训练”**，暗示了 “diloco 效果显著” 以及由于高粒度带来的 fp8 训练优势。[@rasbt](https://twitter.com/rasbt/status/1895511885950357888) 指出 system card 中提到了训练中使用的 **“新监督技术”**。[@rasbt](https://twitter.com/rasbt/status/1895502063154733239) 提到显然 **没有使用字符级训练 (character-training)**。[@Teknium1](https://twitter.com/Teknium1/status/1895380611764015342) 质疑 **GPT-4.5 的知识截止日期为何仍为 2023 年**，尽管目前有预训练运行，他推测是否存在来自 ChatGPT 3.5 的数据污染，或者该模型是否在很久以前就训练完成了。

**模型架构、Scaling Laws 与效率**

- **Scaling Law 的局限性与替代方案**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1895531031475920978) 认为 **GPT-4.5 的发布表明 LLM 预训练的 Scaling Law 已进入平台期**，并指出 **10 倍的算力投入仅带来有限的提升**，这使得像 xAI 这样的公司能够通过算法和数据的创新（如 DeepSeek 展示的效率提升）实现追赶。[@jxmnop](https://twitter.com/jxmnop/status/1895525157101584436) 对此表示赞同，认为 **GPT-4.5 可能标志着“Scaling Law 终结的开始”**，并质疑是数据已耗尽，还是 Scaling Law 无法捕捉到预期的任务性能。[@ibab](https://twitter.com/ibab/status/1895509678773485736) 强调**随着模型规模增大，算法变得愈发重要**，并推测训练细节是 Grok 3 性能表现的关键。[@MParakhin](https://twitter.com/MParakhin/status/1895321112810258518) 表示，**预训练需要更高 Perplexity 的针对性数据和 Active Learning** 才能进一步突破。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1895496203401629933) 断言，**在自然数据上预训练的非思考型 LLM 已达到其实际极限**，并怀疑即使投入 1 万亿美元的训练运行也不会有显著改善。
- **推理算力与效率**：[@rasbt](https://twitter.com/rasbt/status/1895504882561597817) 澄清说，**训练算力和推理算力是提升 LLM 的正交途径**，在不考虑 GPT-4.5 推理算力扩展（Inference-compute Scaling）的情况下进行对比是不公平的。[@rasbt](https://twitter.com/rasbt/status/1895496476056559811) 质疑 **GPT-4.5 是否比 o1（GPT-4 规模 + 推理算力扩展）更贵且更慢**，以及具备 o1 式扩展能力的 GPT-4.5 会是什么样子。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895331433897697689) 重点介绍了关于 **“Thinking Slow, Fast”** 的研究，该研究利用基于 Llama-1B 和 -3B 等小模型以及 Mamba 架构的蒸馏推理器来提升推理扩展性。[@_akhaliq](https://twitter.com/_akhaliq/status/1895340908134178897) 分享了 **FlexiDiT**，这是一个 Diffusion Transformer 框架，通过在去噪过程中使用不同的 Patch Size，以更少的算力生成高质量样本。[@TheTuringPost](https://twitter.com/TheTuringPost/status/1895398797808959963) 讨论了 **Chain of Draft (CoD)**，它鼓励模型生成简短的推理步骤，从而在保持准确性的同时降低成本并提高模型速度。
- **硬件与系统架构**：[@reach_vb](https://twitter.com/reach_vb/status/1895427876985422322) 重点介绍了 **DeepSeek 的 Fire-Flyer 文件系统 (3FS)**，指出其采用了存算分离架构、使用 CRAQ 实现强一致性、无状态元数据服务以及用于推理的 KVCache，实现了极高的读取吞吐量并在基准测试中表现优异。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1895494637231763464) 根据晶体管数量和芯片尺寸讨论了 **N4 工艺相比 N7 工艺可实现 2.32 倍的芯片密度**。[@awnihannun](https://twitter.com/awnihannun/status/1895546249698558363) 报告称 **Kimi 的 Moonshot 16B MoE 模型在 M4 Max 上运行良好**，配合 MLX 速度达到 154 toks/sec，表现优于或等同于稠密型 7B 模型。[@casper_hansen_](https://twitter.com/casper_hansen_/status/1895393985847517313) 评论了 **CUDA 的护城河**，指出甚至 AMD 的工程师也在使用 CUDA 开发 Tensor Engine。

**开源模型、工具与框架**

- **DeepSeek 的开源贡献**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1895531031475920978) 称赞 **DeepSeek 通过基础设施和算法优化大幅降低了 GPU 需求**，并赞扬了他们“神级（goated）的开源工作”。[@reach_vb](https://twitter.com/reach_vb/status/1895429111470031063), [@reach_vb](https://twitter.com/reach_vb/status/1895428961162936563), [@reach_vb](https://twitter.com/reach_vb/status/1895428392872493345) 以及 [@reach_vb](https://twitter.com/reach_vb/status/1895427876985422322) 分享了关于 **DeepSeek 的 Fire-Flyer 文件系统 (3FS)** 及其基准测试的多个链接和细节。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1895392169600635146) 提到 **DeepSeek 2019 年的文件系统至今仍是 SoTA**。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1895346698010140954) 开玩笑地扫描了 DeepSeek 的训练数据，并发现了“一支才华横溢团队的深刻承诺”。
- **Hugging Face 生态系统与集成**：[@_akhaliq](https://twitter.com/_akhaliq/status/1895488615586607609) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1895352428591227397) 提供了代码片段，供**开发者使用 `ai-gradio[openrouter]` 和 Hugging Face 开始体验 GPT-4.5-preview**。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1895530165092127172) 强调了**法国文化部和内政部已入驻 Hugging Face**。[@mervenoyann](https://twitter.com/mervenoyann/status/1895500589871812989) 分享了 **Microsoft 的 MAGMA-8B 模型可以轻松加载到 Hugging Face Transformers**。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1895467916071784896) 宣布可以通过 FireworksAI_HQ **直接在 HF 模型页面进行 Perplexity R1-1776 推理**。[@_akhaliq](https://twitter.com/_akhaliq/status/1895530733269299240) 分享了 **Hugging Face 上的 AI 会议截稿日期**链接。
- **本地 LLM 与 MLX**：[@reach_vb](https://twitter.com/reach_vb/status/1895529293742293144) 分享了**在 Mac 上使用 llama.cpp 本地运行 Phi 4 Mini Instruct** 的指令。[@awnihannun](https://twitter.com/awnihannun/status/1895483273436144002) 致力于**使用本地 LLM 进行性能差距的“氛围检查（vibe-check）”**，更倾向于使用原始终端 (mlx_lm) 和 LM Studio 等工具。[@awnihannun](https://twitter.com/awnihannun/status/1895487722963484697), [@awnihannun](https://twitter.com/awnihannun/status/1895494505543110847), 以及 [@awnihannun](https://twitter.com/awnihannun/status/1895546249698558363) 展示了**在 M4 Max 上使用 MLX 对 Qwen2.5 和 Moonshot 等模型进行本地推理**。
- **其他开源工具与项目**：[@pirroh](https://twitter.com/pirroh/status/1895388564671910277) 提到 Replit 在 **LLM 具备编程能力之前就构建了他们自己的写时复制（Copy-On-Write）分布式文件系统**。[@bobvanluijt](https://twitter.com/bobvanluijt/status/1895463589915353467) 强调了 **Weaviate 的开源向量数据库**及其新功能。[@_akhaliq](https://twitter.com/_akhaliq/status/1895532477823013144) 分享了 **TALKPLAY**，这是一个结合 LLM 的多模态音乐推荐系统。[@alexalbert__](https://twitter.com/alexalbert__/status/1895504248206709246) 宣布了 **Anthropic API 的易用性更新**，允许为图像/文档源使用面向公众的 URL。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1895549589765165183) 推广了与 Codeium 合作的**“使用 Windsurf 的 AI Coding Agents 构建应用”短程课程**。[@AymericRoucher](https://twitter.com/AymericRoucher/status/1895509976321306736) 推荐阅读关于**使用 Arize Phoenix 对 smolagent 运行进行插桩（instrumenting）并设置 LLM-judge 系统**的内容。[@mervenoyann](https://twitter.com/mervenoyann/status/1895516195941728302) 宣传了一个关于**开源艺术工具的每周通讯**。[@rasbt](https://twitter.com/rasbt/status/1895491746295132339) 分享了**使用开源工具在公有/私有云上部署 AI 模型的工作教程**。

**AI 应用与行业用例**

- **企业级 AI 与生产力**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1895554562104443073)、[@perplexity_ai](https://twitter.com/perplexity_ai/status/1895554673567993984)、[@perplexity_ai](https://twitter.com/perplexity_ai/status/1895554843529658472) 和 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1895554855168880918) 宣布推出 **Perplexity Deep Research for Enterprise Data**，可连接到 Google Drive、OneDrive 和 SharePoint，在确保企业级安全性的前提下，实现跨公司文件和网页的深度研究。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1895555108425122139)、[@AravSrinivas](https://twitter.com/AravSrinivas/status/1895555299798630755)、[@AravSrinivas](https://twitter.com/AravSrinivas/status/1895555497698476423)、[@AravSrinivas](https://twitter.com/AravSrinivas/status/1895555816473886935) 和 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1895555900305485849) 进一步详细介绍了 **Perplexity Enterprise Pro**，强调了深度研究、推理、内部/外部搜索、全模型访问及协作等功能。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1895565276131049864) 和 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1895565279830425690) 宣布 **Claude 3.7 Sonnet 在 Arena 的编程排行榜中位列第一**，突显了其强大的能力。[@AIatMeta](https://twitter.com/AIatMeta/status/1895528149137629220) 展示了塞维利亚足球俱乐部（SevillaFC）如何利用 Llama 和 IBM 的 watsonx 创建 **Scout Advisor**，用于足球明星的球探挖掘。[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1895531723640729907) 强调了 **ConsensusNLP 使用 GPT-4.5 进行科学/医学分析**，并利用结构化输出将研究共识可视化。
- **智能体 AI 与自动化**：[@mervenoyann](https://twitter.com/mervenoyann/status/1895497146344026184) 宣布了 **微软的 MAGMA-8B 视觉语言动作模型**，用于物理和数字世界的操作，包括具身机器人和网页自动化。[@llama_index](https://twitter.com/llama_index/status/1895522358561227000) 分享了一个**使用 LlamaIndex 构建的智能体（Agentic）生产力应用示例**。[@RichardSocher](https://twitter.com/RichardSocher/status/1895369311109423210) 建议在处理严重的医学问题时，使用像 **ARI 这样的研究智能体（Research Agents）进行广泛的文献综述**，并提供了一份示例报告。
- **编程与开发**：[@nearcyan](https://twitter.com/nearcyan/status/1895432741275242957) 分享了一个关于**初级开发者看着 Claude 3.7 “在 Cursor 中摧毁他们的代码库”**的梗图。[@HamelHusain](https://twitter.com/HamelHusain/status/1895354490519437764) 表示**“只有依靠 AI，我才可能理解 GraphQL”**。[@cloneofsimo](https://twitter.com/cloneofsimo/status/1895454047483896166) 批评了**当前的自动化软件开发工具，如 Devin、OpenHands、Replit 和 Cursor Compose**，认为它们甚至无法端到端地完成小型应用，在服务器/客户端、IPC、队列和调度能力方面存在不足。[@rishdotblog](https://twitter.com/rishdotblog/status/1895457874299752868) 声称**用每月 10 美元的 Claude Code 方案取代了每月 100 美元的工具**，并暗示编程工作和 SaaS 公司正在“消失”。

**AI 研究与论文**

- **近期研究论文亮点**：[@rasbt](https://twitter.com/rasbt/status/1895487669003518039) 提供了一份**近期 AI 研究论文列表**，涵盖了 SWE-RL、LoRA boosting、long-context LLMs、Logic-RL、test-time scaling、AI research agents、模型选择、inner thinking transformers、自然推理、知识获取、使用 LLMs 进行自由软件工程、sparse attention、unlearning、large language diffusion models、模型合并、推理-行动困境、金融 LLMs、无限上下文、蒸馏缩放定律、prompt caching、从演示中推理、分层推理、LLMs 中的思考、计算最优 test-time scaling、数学推理、large memory models、量化 LLMs、video RoPE、扩展 test-time compute、自我回溯、训练高效推理、推理进展、通过 RL 教授批判、增强领域应用的推理、less-is-more 推理、chain-of-thought 推理、chain-of-associated-thoughts、直接对齐算法、embedding 层缩放以及使用大型推理模型进行竞赛编程等主题。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895334570704470281)、[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895334574160519522)、[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895333239608508473)、[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895333242041180256)、[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895331433897697689) 和 [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1895331436171010152) 重点介绍了关于 **FlexiDiT、Self-Training for Concise Reasoning 以及 Thinking Slow, Fast with Distilled Reasoners** 的论文，并提供了摘要和代码链接。[@omarsar0](https://twitter.com/omarsar0/status/1895528463504982115)、[@omarsar0](https://twitter.com/omarsar0/status/1895528446882955439) 和 [@omarsar0](https://twitter.com/omarsar0/status/1895528398820425741) 分享了关于 **METAL (Modality-tailored critique)、用于自我修正的 Modality-tailored critiques 以及 Test-Time Scaling on Chart Generation** 的论文，并指出了性能提升。[@_akhaliq](https://twitter.com/_akhaliq/status/1895341871721013408)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895341929271017897)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895341032973443349)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895340908134178897)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895339427859505583)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895339378257600744)、[@_akhaliq](https://twitter.com/_akhaliq/status/1895338234085024241) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1895338193303806287) 链接到了关于 **Mobius (Text to Seamless Looping Video)、FlexiDiT、R1-T1 (Translation Capability Incentivization) 和 LongRoPE2 (Context Window Scaling)** 的论文。[@dair_ai](https://twitter.com/dair_ai/status/1895532543652642850) 和 [@dair_ai](https://twitter.com/dair_ai/status/1895532546051752138) 重点介绍了 **Google 的 PlanGEN 框架，用于 LLMs 中的复杂规划和推理**，并详细说明了其约束引导验证和自适应算法选择。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1895501917398155695) 总结了一篇关于 **Brain2Qwerty 的论文，这是一个使用 MEG 记录将脑电波翻译成文本的非侵入性 AI 系统**。
- **认知科学与 AI Alignment 理论**：[@AndrewLampinen](https://twitter.com/AndrewLampinen/status/1895520333257744493) 分享了一篇关于 **"Naturalistic Computational Cognitive Science"** 的预印本，将 AI 和认知科学结合起来，旨在建立可泛化的认知模型。[@DanHendrycks](https://twitter.com/DanHendrycks/status/1895378847547478093) 讨论了 **AI alignment 理论中思想的演变**，将“随机模因漂移”与 Yudkowsky 的贡献进行了对比，并暗示 GPT 正在迫使 alignment 论坛面对经验现实。

**幽默与杂项**

- **AI 模型幽默与氛围检查 (Vibe Checks)**：[@_akhaliq](https://twitter.com/_akhaliq/status/1895348244512973149) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1895346920983536106) 发布了 **动态 SVG，作为 GPT-4.5 关于开源问题的幽默回应**。[@_philschmid](https://twitter.com/_philschmid/status/1895387638229766505) 征集 **“氛围测试提示词” (vibe test prompts)**，例如要求从 1 数到 10 并省略以 "e" 结尾的数字，以及生成一只骑自行车的鹈鹕的 SVG。[@NeelNanda5](https://twitter.com/NeelNanda5/status/1895371690571636880) 分享了一个 **LLM 技巧：“以 Scott Alexander 博客文章的风格编写回复”**，以获得更令人愉悦的长文本输出。[@aidan_mclau](https://twitter.com/aidan_mclau/status/1895312915387064584) 展示了一个 **从 0 到无穷大的幽默 IQ 量表**，最终以一个充满哲理的屁笑话达到顶峰。[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1895362655722029414) 分享了一个关于 **询问 OpenAI 他们的模型是优秀还是懒惰** 的梗图。[@Teknium1](https://twitter.com/Teknium1/status/1895565961107030083) 发布了“GPT-4.5 终于懂我了，笑死”，并配有一张暗示 GPT-4.5 理解了他们性格的图片。
- **社会与哲学思考**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1895535610053710249) 观察到了 **高智商自闭症谱系生理男性、跨性别身份与系统化思维之间的人口统计学重叠**。[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1895349955130446189) 将 **2012 年以来的美国总统职位比作累进象棋 (progressive chess)**。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1895495175499317574) 调侃道 **Unitree 机器人将导致唯我论 (solipsism) 的抬头**。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1895572191452020903) 表达了一个 **将核武器、AI 和无人机作为理性防御的“噩梦”场景**。[@AmandaAskell](https://twitter.com/AmandaAskell/status/1895559111565328608) 幽默地建议用一种 **昂贵的“我超级尊重你”胸针** 来替代东海岸正式场合中令人不适的西装。[@AmandaAskell](https://twitter.com/AmandaAskell/status/1895493849575002127) 调侃了 **约会软件上带有性别色彩的个人资料偏好**。
- **行业与社区动态**：[@suchenzang](https://twitter.com/suchenzang/status/1895560716981346466) 发布了“大模型的气味”并附带链接，[@suchenzang](https://twitter.com/suchenzang/status/1895437762427560236) 推文称“有些东西你花 90 亿美元买不到，甚至 300 亿美元也不行……”。[@nearcyan](https://twitter.com/nearcyan/status/1895568285326020802) 宣布 **“受够了基准测试 (benchmarks)”**，并表示对超维形状的描述失去了共情。[@agihippo](https://twitter.com/agihippo/status/1895337878311575875) 质疑了 **AI 行业的工作时间，暗示“AI 圈的人几乎一直在工作！”**。[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1895507887658578403) 表示“非常高兴看到更多经典游戏源代码发布”，并指出 **游戏开发与更广泛的开源文化之间的脱节**。[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1895495183137161551) 调侃道 **Runway 新的关于页面写着“我们是人造大脑的脑科医生。”**。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek 发布：革命性的存储与数据处理技术**

- **DeepSeek 发布第五弹！再次投下集群炸弹！3FS (分布式文件系统) & smallpond (轻量级数据处理框架)** ([**Score: 499, Comments: 73**](https://reddit.com/r/LocalLLaMA/comments/1izvwck/deepseek_realse_5th_bomb_cluster_bomb_again_3fs/))：**DeepSeek** 推出了 **3FS**，这是一款针对 AI 工作负载优化的高性能分布式文件系统，利用现代 **SSD** 和 **RDMA 网络**来增强分布式应用程序的开发。此外，**smallpond** 作为一个轻量级数据处理框架，集成了 **DuckDB** 和 **3FS**，为数据处理任务提供了流线型的解决方案。欲了解更多信息，请访问其 [**GitHub 页面**](https://github.com/deepseek-ai/3FS) 和 [**smallpond 仓库**](https://github.com/deepseek-ai/smallpond)。
    - **3FS 性能与对比**：**3FS** 实现了惊人的 **6.6 TiB/s 带宽**，显著超过了典型的 **DRAM 速度**。讨论中将 **3FS** 与 **Colossus** 等其他系统进行了比较，并指出其在 **AI 训练工作负载**中的独特应用，无需传统的缓存（caching）等文件读取优化。
    - **开源策略与影响**：许多评论者赞赏 **DeepSeek** 的开源方法，强调其在推动 AI 进步民主化以及挑战 **OpenAI** 和 **Nvidia** 等垄断技术巨头方面的潜力。开源文化被强调为一个互惠过程，使贡献者和更广泛的 AI 社区共同受益。
    - **技术见解与历史背景**：**3FS** 已投入生产五年多，由 **High-Flyer AI**（幻方量化）开发并用于其 **Fire-Flyer II 系统**。它针对大规模随机读取操作进行了优化，采用 **Direct I/O**，并使用 **FFRecord** 格式存储样本数据，显著提高了 AI 模型训练效率。
- **DeepSeek 开源周第 5 天** ([**Score: 127, Comments: 9**](https://reddit.com/r/LocalLLaMA/comments/1izwh49/deepseek_opensourceweek_day_5/))：**Fire-Flyer File System (3FS)** 是一款并行文件系统，旨在最大化现代 **SSD** 和 **RDMA 网络**的带宽，在 180 节点的集群中实现了惊人的 **6.6 TiB/s 聚合读取吞吐量**，并在 25 节点的集群上通过 **GraySort** 基准测试实现了 **3.66 TiB/min 吞吐量**。它为 **KVCache** 查找提供每个客户端节点 **40+ GiB/s 的峰值吞吐量**，并支持具有强一致性语义的解耦架构，便于执行训练数据预处理和嵌入向量搜索等任务。欲了解更多细节，请访问 [**3FS 仓库**](https://github.com/deepseek-ai/3FS) 和 [**Smallpond 框架**](https://github.com/deepseek-ai/smallpond)。
    - **3FS** 非常适合 **AI 训练工作负载**和 **AI 推理**，具有无需预取即可随机访问训练样本、高吞吐量检查点（checkpointing）以及为大语言模型推理提供高性价比 **KVCache** 等优势。它还支持需要强一致性和高吞吐量的**数据密集型应用**，其在 **GraySort 基准测试**中的表现证明了这一点。
    - 用户对开发团队的生产力表示惊讶，指出尽管人力有限，产出却令人印象深刻。该项目起源于 CEO 的对冲基金团队（2019 年），其招聘策略侧重于从顶尖大学招聘优秀的 **CS** 毕业生。
    - 一些用户认为 **3FS** 的技术细节过于复杂，不直接适用于大多数用例，这表明用户期望与该系统的专业功能之间可能存在错位。

**主题 2. 法国推理模型：经济且有效**

- **我只花了 20 美元就训练了一个会说法语的推理模型！🤯🇫🇷** ([**分数: 229, 评论: 78**](https://reddit.com/r/LocalLLaMA/comments/1j045xn/i_trained_a_reasoning_model_that_speaks_frenchfor/)): 无法生成摘要，因为帖子正文不包含足够的文本信息，仅包含一个视频链接。
    - **微调 7B LLM**：**TheREXincoming** 基于 **Qwen 2.5** 微调了一个 **7B LLM**，仅使用了 **2,000 个样本**（1,000 个英文 + 1,000 个法文），成本仅为 **20 美元**。该模型在数学基准测试上的表现与 **R1 Distil 7B** 相当，展现了极小的知识退化。
    - **模型与数据可用性**：微调后的模型及其数据集已在 **Hugging Face** 上发布（[**数据**](https://huggingface.co/datasets/HoangHa/Pensez-v0.1), [**模型**](https://huggingface.co/HoangHa/Pensez-v0.1-e5), [**GGUF**](https://huggingface.co/HoangHa/Pensez-v0.1-e5-GGUF)）。该模型旨在提供高性能的法语能力，并可作为在其他语言中训练推理 LLM 的模板。
    - **社区反馈与开发**：用户询问了数据选择和训练细节，而 **TheREXincoming** 提到正在努力清理数据策划流水线（data curation pipeline），并计划更新仓库。这一举措因其极低的成本和实现的高性能而受到了热烈欢迎和难以置信的评价。

**主题 3. Sesame 实时语音模型媲美 OpenAI**

- **Sesame 发布的“跨越对话语音的恐怖谷”帖子 —— 实时对话音频模型媲美 OpenAI** ([**分数: 200, 评论: 37**](https://reddit.com/r/LocalLLaMA/comments/1j00v4y/crossing_the_uncanny_valley_of_conversational/))：**Sesame** 展示了一个引人注目的实时对话语音模型，可与 **OpenAI 的 Advanced Voice Mode** 媲美，并计划以 **Apache 2.0 license** 发布。虽然公开权重尚未发布，但演示视频的质量给用户留下了深刻印象，预示着这位语音合成技术新秀的前景广阔。
    - 用户对 **Sesame 对话语音模型** 印象深刻，指出其质量和速度优于 **ChatGPT 的高级语音模式**。演示视频因其流畅的响应时间和逼真的声音而受到称赞，用户对其潜在的开源发布表示兴奋。
    - 人们对该模型与其他技术（如 **function calling** 和 **RAG**）集成的潜力充满热情，认为这可以在不增加延迟的情况下增强其功能。用户渴望该模型能在 **Hugging Face** 等平台上发布，以便更轻松地访问和集成。
    - 一些用户指出了局限性，例如模型无法检测情绪或讽刺，以及如果输入延迟则倾向于关闭对话。尽管存在这些问题，该模型引人入胜的对话风格和记忆能力仍受到赞赏，用户期待在自己的环境中进行尝试。

## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. GPT 4.5 的幽默与创意应用**

- **GPT 4.5 模仿 Donald Trump 解释地球的创造** ([**Score: 550, Comments: 86**](https://reddit.com/r/OpenAI/comments/1j06uyk/gpt_45_as_donald_trump_explaining_creation_of/)): **GPT 4.5** 幽默地模仿了 **Donald Trump**，以讽刺的叙事方式讲述了地球的创造，将行星的形成归功于 Trump 的个人倡议。叙事强调了关于创造太阳、地球及其特征的夸张说法，同时幽默地批评恐龙是一个“巨大的错误”，随后引入了“获胜”的动物和人类，所有这些都带有 Trump 典型的语言风格。
    - 评论者赞赏 **GPT 4.5** 叙事的幽默感和风格，许多人觉得它很有趣，并注意到其夸张的 **Trump-like** 特质，尽管有些人觉得它过于连贯或重复。关于 **dinosaurs** 是“巨大错误”以及地球是“有史以来最湿润的”幽默感特别引起了读者的共鸣。
    - 人们对使用 **text-to-speech** 模型将文本转换为音频表现出兴趣，一些人已经分享了音频链接（[**SoundProofHead 的链接**](https://whyp.it/tracks/261792/trump?token=IcE0q) 和 [**TwoLevelsAhead 的链接**](https://elontalks.com/share/7099b22a-5821-4b15-8354-feaee2eeece1)），或者表达了对 **deepfake video** 版本的渴望。
    - 讨论强调了 AI 在幽默方面的潜力，一些评论者认为实现真正的 **comedy** 可能是 AI 能力的一个重要基准，而另一些人则开玩笑说 AI 将幽默掌握到超人水平的影响。
- [**ChatGPT 对 emoji 的存在主义危机**](https://i.redd.it/9guzd0ztkwle1.jpeg) ([**Score: 203, Comments: 48**](https://reddit.com/r/ChatGPT/comments/1j0b2ea/chatgpts_existential_crisis_over_emoji/)): **ChatGPT 幽默地误认了 emoji**，包括海马、独角兽、虾和龙，导致对 emoji 识别能力的俏皮而又带有存在主义色彩的反思。这段对话显示在黑色背景上，强调了 AI 在尝试识别 emoji 时随意且具有喜剧色彩的本质。
    - **Emoji 误认**：用户喜欢分享 **ChatGPT** 误认 emoji 的幽默实例，通常反复将海马与其他动物（如独角兽、龙和鱼）混淆。这导致了俏皮且具有喜剧色彩的交流，突显了 AI 在 emoji 识别方面的挣扎。
    - **社区参与**：许多用户分享了自己的经历和截图，为对话的轻松氛围做出了贡献。分享的内容包括图片链接和幽默对话，强调了社区对 AI 离奇回应的共同享受。
    - **AI 幽默与反思**：该帖子反映了 AI 局限性的奇特本质，用户欣赏这些喜剧性错误并参与到共享的数字体验中。这种俏皮的互动突显了社区对 AI 不可预测性的喜爱，以及从其错误中获得的共同幽默感。

**主题 2. AI 视频和音频处理的创新**

- [**Advanced Voice 4.5**](https://v.redd.it/sphlqpc77tle1) ([**Score: 365, Comments: 95**](https://reddit.com/r/ChatGPT/comments/1izzows/advanced_voice_45/)): 标题为 **“Advanced Voice 4.5”** 的帖子可能讨论了 **AI voice acting** 技术的进步，特别是针对 **4.5** 版本。在没有额外背景或细节的情况下，该帖子强调了开发更**真实的 AI 生成语音**。
    - 用户对 **“Advanced Voice 4.5”** 更新持怀疑态度，质疑其是否包含语音方面的改进，因为一些人认为这只是一个无审查（uncensored）更新。**TheRobotCluster** 声称 4.5 版本并不适用于语音，而只是一个无审查版本，这引发了关于 **ChatGPT** 现在是否允许无审查内容的疑问。
    - 围绕 **AI 模仿口音能力** 的讨论显示出褒贬不一的看法；一些用户批评 AI 对 **英语口音** 的尝试，认为听起来像美国人在模仿。这引发了对 AI 生成口音的真实性和准确性的质疑。
    - 对话触及了 AI 对各行业的影响，一些用户预测 AI 的进步，特别是在配音以及潜在的**色情行业**，可能会在未来带来重大的技术演进和经济收益。
- [**SpargeAttn: A new method giving you a 1.83x speedup on video models with NO quality loss.**](https://i.redd.it/kuz97049xule1.jpeg) ([**Score: 155, Comments: 45**](https://reddit.com/r/StableDiffusion/comments/1j04o63/spargeattn_a_new_method_giving_you_a_183x_speedup/)): **SpargeAttn** 为视频模型提供了 **1.83 倍的加速**，且不损失质量，正如在 **L40 GPU** 上的对比所示。该方法将处理时间从使用 “Full Attention” 的 **1897 秒** 减少到 **1037 秒**，同时保持了视频质量。
    - **安装挑战**：用户讨论了安装 **SpargeAttn** 的复杂性，原因是存在 **Triton** 等依赖项以及对特定 Python 版本的需求。帖子提供了在 Windows 上安装的详细步骤，包括必要软件包的链接以及与 **ComfyUI** 集成的命令。
    - **兼容性与性能**：指出 **SpargeAttn** 是特定于模型维度的，在不同模型大小（例如 1.3B 与 14B 模型）之间进行微调时可能会出现问题。**Sliding Tile Attention** 被提及为一种替代方案，在微调下表现良好，但目前仅限于 **H100** 显卡。
    - **社区贡献**：**Kijai** 已将 **SpargeAttn** 整合到 **ComfyUI-WanVideoWrapper** 中，展示了社区将新工具集成到现有框架中的努力。用户表达了对未来原生支持 **sage attention** 和 **triton** 等注意力机制（attention mechanisms）的希望，以简化安装过程。

**主题 3. AI 身份混淆与幻觉**

- **Grok 在未受提示的情况下认为自己是 Claude，并在被指正后坚持这一说法** ([**Score: 187, Comments: 54**](https://reddit.com/r/ClaudeAI/comments/1j0327j/groks_thinks_it_is_claude_unprompted_and_doubles/))：**Grok**（一款 AI 模型）在与一家辩论俱乐部负责人的对话中错误地自称为 **Claude**，且在受到质疑后仍坚持这一说法。这一事件在 [**X**](https://x.com/TentBC/status/1895386542702731371?t=96M796dLqiNwgoRcavVX-w&s=19) 上分享的一段对话中被详细记录，引发了人们对这种身份混淆根本原因的质疑。
    - 几位用户推测，**Grok 的身份混淆**可能源于其训练数据，其中包含了来自 **Claude** 等旧模型的输出。有人认为，由于 **xAI** 成立时间较短且试图减少偏见，其 post-training 可能不够彻底，从而导致了此类错误。
    - 一些人幽默地看待这一事件，评论中强调了**辩论俱乐部**质疑天花是否存在这一行为的荒谬性。这引发了对该辩论俱乐部合法性的怀疑，一些用户认为它看起来像是一个阴谋论团体。
    - 有人怀疑 **Grok** 可能在底层使用了 **Claude** 的技术，或者是在其数据集上进行的训练，类似于 **Deepseek** 使用 **ChatGPT** 的数据，这引发了对此类做法的法律和伦理担忧。
- [**GPT-4.5 会在对话中凭空捏造概念**](https://i.redd.it/2h1m59ehsxle1.png) ([**Score: 348, Comments: 75**](https://reddit.com/r/OpenAI/comments/1j0gxxs/gpt45_will_just_invent_concepts_midconversation/))：**GPT-4.5** 因其在交互过程中捏造概念的能力而受到关注，正如 **Aaron Ng 在 Twitter 上的帖子**所强调的那样。在一段对话片段中，该 AI 专门为此次交互捏造了一个 “CLEAR Model”，展示了其动态对话能力。
    - **Peter Hawkins** 最初发明了 **CLEAR Model**，而 **GPT-4.5** 对它的引用被 **I_am_John_Mac** 指出是一种 hallucination（幻觉），并附上了 [**hotpmo.com**](https://www.hotpmo.com/management-models/the-clear-model-peter-hawkins/) 的链接。这突显了 **GPT-4.5** 倾向于创造可能并不准确或并非原创的概念。
    - 讨论中带有一种幽默的基调，谈论将 **hallucinations** 变成一种功能，一些用户开玩笑说 AI 可能会为其幻觉出的概念申请专利或主张知识产权。
    - **GPT-4.5** 的 **hallucination rate**（幻觉率）据记录为 **37.1%**，低于 **GPT-4o** 的 **61.8%** 和 **o1** 的 **44%**（由 **Hexpe** 和 **vingeran** 提及），这表明其准确性较之前的模型有所提高。

**主题 4. AI 工具简化编程与写作**

- **我开发了一个简单的工具，彻底改变了我与 AI 编程助手协作的方式** ([**Score: 167, Comments: 41**](https://reddit.com/r/ClaudeAI/comments/1j0ey3h/i_made_a_simple_tool_that_completely_changed_how/)): **CodeSelect** 是一款旨在简化与 **Claude** 和 **ChatGPT** 等 AI 编程助手共享代码过程的工具。它通过复选框树显示项目结构，允许快速选择文件，并自动检测文件关系以提供更好的上下文。这款轻量级工具只需一条命令即可安装，且没有外部依赖，通过提供适当的上下文显著减少了准备时间并提高了 AI 响应质量，该工具已在 [**GitHub**](https://github.com/maynetee/codeselect) 上开源。
    - **Repomix** 被强调为管理代码项目结构的替代工具，只需简单的命令 (**`cd myProject && npx repomix`**) 即可在任何文件夹上运行并输出一个可拖拽的文件，用户发现这在项目管理中非常有效。
    - 用户讨论了将 **Gemini 驱动的 Agent** 集成到 **CodeSelect** 中，以向 **Claude** 建议编辑和文件引用，旨在提高效率并在编程过程中节省 Token。
    - **Claude 的 GitHub 集成** 因其管理全项目变更的能力（如重命名变量和更新注释）而受到关注，用户认为在无需手动输入的情况下保持项目上下文的能力令人印象深刻。
- **咬牙订阅了 Claude Pro 年度会员** ([**Score: 104, Comments: 128**](https://reddit.com/r/ClaudeAI/comments/1j04snp/just_bit_the_bullet_and_got_a_yearly_claude_pro/)): 作者称赞 **Claude Pro 订阅**是处理日常任务、数据分析、创意问题解决和软件工程的变革性工具，并强调了它在调试和代码审查方面的有效性。他们对 **Anthropic** 的产品表示满意，将其与对 **Claude 3.7** 过于简练的批评形成对比，并强调了它相对于传统搜索引擎的重大进步。
    - 用户讨论了**使用限制 (usage limits)** 是 **Claude Pro 订阅**的一个重大问题，一些人建议通过开启新对话等策略来有效管理限制。其他人则对频繁达到限制表示沮丧，认为这干扰了他们的工作流，而部分用户则报告通过保持对话简短很少遇到这些问题。
    - 有人怀疑赞扬 **Claude Pro** 的帖子是否真实，部分用户怀疑这些帖子是**营销活动**的一部分。这种怀疑源于帖子发布时间与促销邮件同步，以及正面评价的重复性，不过也有人认为由于该子版块的定位，这些讨论是真实的。
    - 订阅者争论了**年度订阅**与按月支付的价值，一些人因质量下降和严格的使用限制而后悔购买。其他人则发现订阅对他们的工作大有裨益，认为决定应取决于个人使用场景和快速发展的 AI 领域。

---

# AI Discord Recap

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. GPT-4.5 入场，但 Claude 3.7 仍是编程之王**

- [**GPT-4.5 未能令人惊艳，价格令人咋舌**](https://openai.com/index/introducing-gpt-4-5/): 早期测试者发现 **OpenAI 的 GPT-4.5** *价格过高（每百万 Token 150 美元）*，且在编程方面并不比 **GPT-4 Turbo** 好多少，许多开发者仍然青睐 **Claude 3.7 Sonnet**，认为其在软件工程任务中表现更优。aider 的多语言编程基准测试显示 **GPT-4.5** 得分为 **45%**，而 **Sonnet 3.7 为 65%**，鉴于高昂的 API 成本，这导致了用户的失望并对其价值主张产生质疑。
- [**Claude 3.7 Sonnet 面临负载问题，但仍是顶级编程模型**](https://www.anthropic.com/news/claude-3-7-sonnet): 尽管有高负载提示和拒绝服务的报告，**Claude 3.7 Sonnet** 仍被认为是软件工程的最佳模型，因为它能够准确遵循指令并有效调试代码。用户强调了 **Claude 3.7** 改进的指令遵循和调试能力，尽管有人猜测 **Anthropic** 正在使该模型变得更难使用。
- [**DeepSeek R2 期待值持续升温**](https://github.com/deepseek-ai/DualPipe): 针对 **DeepSeek R2 模型** 的期待正在积聚，一些成员预计它将超越目前的 SOTA 模型并打破企业的炒作，因为 **DeepSeek 的 Chatbot** 在编程方面已经超越了现有模型。成员们将 **DeepSeek 的 R1 模型** 与 **OpenAI 的 o1** 进行了正面比较，进一步推高了对即将发布的 **R2** 的兴奋感。

**主题 2. IDE 之战：Cursor 与 Windsurf 争夺 AI 编程霸权**

- [**Cursor 饱受 Bug 困扰，用户怨声载道**](https://www.cursor.com/downloads)：用户报告 **Cursor IDE** 充斥着各种 Bug，在更新后经常出现崩溃和代码更改丢失的情况，一些用户正考虑禁用自动更新并等待更稳定的版本。随着部分用户声称 **Claude 3.7** 在 **Cursor** 上的编码质量自发布以来有所下降，挫败感不断增加。
- [**Windsurf AI 紧跟 GPT-4.5 潮流，质疑声随之而来**](https://x.com/windsurf_ai/status/1895206330987880816)：**Windsurf AI** 在 Beta 版中集成了 **GPT-4.5**，但早期测试显示其在软件工程方面的*成本显著更高且表现不如预期*，这引发了关于此举是真心实意还是针对 **Cursor** 的*宣传攻势*的争论。用户对 Windsurf 的定价模型（特别是 Flow Credits）表示质疑，认为 **Cursor** 的定价更加简单直接。
- [**Cursor 的 Memory Banks 被认为“毫无意义”且成本高昂**](https://discord.com/channels/1074847526655643750)：**Cursor** 的 **Memory Banks** 功能被批评为低效且昂贵，用户报告使用 **Claude 3.7 API** 的每日成本高达 **50 美元**，而且 Memory Banks 有时会产生*幻觉*，使得雇佣一名程序员反而更便宜。用户发现 Memory Banks 效率低下，因为它们偶尔会犯错，从而得出雇佣人类程序员更具成本效益的结论。

**主题 3. 硬件博弈：DeepSeek 的 DualPipe 和 TinyLM 展现创新曙光**

- [**DeepSeek 的 DualPipe 向流水线气泡宣战**](https://github.com/deepseek-ai/DualPipe)：**DeepSeek AI** 发布了 **DualPipe**，这是一种用于 **V3/R1 训练**中计算-通信重叠的双向流水线并行算法，旨在比传统方法减少流水线气泡（Pipeline Bubbles）。此版本与专家并行负载均衡器 **EPLB** 一起，都是 **DeepSeek AI** 为期一周的系列发布活动的一部分。
- [**TinyLM 凭借 WebGPU 之势释放客户端 LLM 潜力**](https://tinylm.wizenheimer.dev/)：**tinylm** v0 发布，这是一个支持在浏览器或 Node.js 中通过 **WebGPU 加速**运行客户端 **LLM** 的库，具有**零成本推理**和完全隐私的特性，并提供兼容 OpenAI 的 API。**tinylm** 支持文本生成、Embeddings 和实时 Token 流式传输，消除了本地 LLM 推理对服务器的需求。
- [**NVIDIA 将 Tensor Core 重心转向 FP4，抛弃 INT4？**](https://github.com/gau-nernst/quantized-training?tab=readme-ov-file#matmul)：**NVIDIA** 似乎正在将重心从 **INT4 Tensor Cores** 转向 **FP4**，**Blackwell** GPU 采用了 **FP4**，而 **Ada** 架构拥有 **INT4**，**Hopper** 架构拥有 **INT8**，这引发了关于 INT4 精度在 NVIDIA 硬件策略中未来地位的疑问。基准测试表明 **NVIDIA** 正在优先考虑将 **FP4** 用于量化模型训练，这可能会影响未来的硬件开发和软件优化策略。

**主题 4. 定价压力：GPT-4.5 API 成本引发公愤，开源替代方案备受关注**

- [**GPT-4.5 API 定价被指“疯狂”，用户寻求替代方案**](https://x.com/OpenRouterAI/status/1895236199004152272)：**OpenAI** 的 **GPT-4.5 (Preview)** API 定价为 **每百万 Token 输入 75 美元 / 输出 150 美元**，遭到了严厉批评。用户谴责其与 **Grok3** 和 **Claude Sonnet 3.7** 等模型相比过高的成本，质疑其价值，并促使一些人考虑开源替代方案。**GPT-4.5** 的高昂成本引发了开发者和研究人员对其可访问性和可持续性的担忧。
- [**用户称 Deepinfra 价格比 Fal AI 便宜 100 倍**](https://discord.com/channels/879548962464493619/879548962464493622/1344457413314740256)：一位用户声称 **Deepinfra** 在字符处理方面比 **Fal AI** 便宜 100 倍，收费为*每百万字符 0.8 美元*并提供免费算力，而 **Fal AI** 仅提供 *50 美元的免费额度*，并建议将 **Kokoro TTS** 作为另一种低成本替代方案。这种定价差异凸显了 AI 基础设施市场的竞争格局和成本节约机会。
- [**Windsurf 用户质疑 Flow Credits，认为 Cursor 定价“更可取”**](https://codeium.canny.io)：**Windsurf** 的定价模型，特别是 Flow Credits 和额外的 Flow Action 成本，令用户感到困惑，导致一些人更倾向于 **Cursor** 更简单直接的定价方式。用户对额外 Flow Action 的不成比例成本表示担忧，这影响了 Windsurf 定价结构的感知价值和透明度。

**主题 5. 社区脉动：从机器人手臂到 CUDA 版 LeetCode，创新蓬勃发展**

- [**爱好者们联合起来构建 DIY 机器人手臂**](https://www.creality.com/products/ender-3-v2-3d-printer)：**LM Studio Discord** 的成员们正热烈讨论从零开始构建机器人手臂，利用价格亲民的 3D 打印机（如 [100 美元的 Creality Ender 3 V2](https://www.creality.com/products/ender-3-v2-3d-printer)）以及用于学习舵机、CAD 和微控制器的开源资源。该项目展示了社区在学习和应用 AI 及机器人原理方面的动手实践能力。
- [**CUDA 版 LeetCode 问世，挑战 GPU 大佬**](https://leetgpu.com/challenges)：**CUDA 社区**庆祝 [LeetCode for CUDA](https://leetgpu.com/challenges) 的 Beta 版本发布，这是一个专门为 **CUDA 开发**设计的编程挑战新平台，邀请用户测试技能并提供反馈。这一新平台为提升 CUDA 编程技能营造了竞争与协作的环境。
- [**Hugging Face 社区修复了微软 Phi-4 Mini 的烂摊子**](https://huggingface.co/unsloth/Phi-4-mini-instruct)：由于存在 Bug，**Microsoft** 的 **Phi-4 mini** 模型被发现*完全无法使用*，在 **Microsoft** 未能采纳 **Unsloth** 的修复方案后，促使 **Unsloth AI 团队**在 Hugging Face 上上传了[修复版本](https://huggingface.co/unsloth/Phi-4-mini-instruct)。这种社区驱动的努力凸显了开源 AI 开发的协作本质以及对关键问题快速响应的重要性。

---

# 第一部分：Discord 高层摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **GPT-4.5 因高昂价格令测试者失望**：OpenAI 的 **GPT-4.5** 早期测试者发现其*价格昂贵且并不比 GPT-4 Turbo 显著更好*，其成本高达**每百万 token 150 美元**。
   - 共识是 **Claude 3.7 Sonnet** 在编程方面仍然更胜一筹，导致一些人称 GPT-4.5 *“只是块头大”*，并强调其缺乏新的前沿能力。
- **Claude 3.7 Sonnet 面临高负载和拒绝回答问题**：用户报告了 **Claude 3.7 Sonnet** 的问题，包括频繁的高负载提示和拒绝回答某些提示词，一些人猜测 **Anthropic** 是否正在使模型变得更难使用。
   - 尽管存在这些问题，许多人仍认为 **Claude 3.7 Sonnet** 是软件工程的最佳模型，因为它能够准确遵循指令并有效地调试代码。
- **Cursor 饱受 Bug 和更新困扰**：多名用户报告在更新后经历**频繁崩溃并需要重新安装 Cursor**，且因 Bug 丢失了代码更改，最新版本可能会影响性能和稳定性。
   - 其他人建议禁用自动更新并等待更稳定的版本，一些用户声称在 Cursor 上使用 Claude 3.7 编程的质量较发布时有所下降。
- **Windsurf AI 吹嘘快速集成 GPT-4.5**：**Windsurf AI** 宣布 GPT-4.5 现已在 Windsurf 开启 Beta 测试，但指出早期测试显示其*价格显著高于其他替代模型（>10倍）*，且在软件工程或工具调用（tool calling）方面不如现有模型快，也不如现有模型强。
   - 根据[这条推文](https://x.com/windsurf_ai/status/1895206330987880816?s=46&t=ggmESCIXF0nYw8_kshHz7A)，用户在争论 Windsurf 的举动仅仅是*攻击 Cursor 的宣传手段*，还是即便在有限制的情况下仍努力提供最新模型访问权限的真诚尝试。
- **Memory Banks 表现不及预期**：Discord 成员报告称 Memory Banks（记忆库）似乎非常低效，而且除了价格昂贵外，使用 Claude 3.7 API 很容易达到**每天 50 美元**的开销。
   - 低效源于 Memory Banks 有时会犯错或产生*幻觉*，这使得直接雇佣一名程序员反而更便宜。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-4.5 表现平平，Claude 3.7 占据主导地位**：早期基准测试显示 **GPT-4.5 Preview** 的编程性能令人失望，在 aider 的多语言编程基准测试中仅获得 **45%** 的分数，而 **Sonnet 3.7** 为 **65%**。这让成员们认为它的定位是一个*“友好的”非推理语言模型*。
   - 尽管 **GPT-4.5** 已经发布，但 **Claude 3.7** 仍然是处理复杂编程问题的首选，在编程基准测试中优于 **GPT-4.5**，且更容易被越狱（jailbreak）。
- **DeepSeek R2 热度激增**：成员们对 **DeepSeek 的 R2 模型** 充满期待，预计它将超越目前的 SOTA 模型并打破企业的宣传噱头，一些人将 **DeepSeek 的 R1 模型** 与 **O1** 进行比较。
   - 这种期待源于一种观点，即 **DeepSeek 的聊天机器人** 在编程能力上已经超越了现有模型。
- **Aider 用户倡导自动重试模式**：用户要求为 **Aider** 增加自动重试模式，以解决 **Deepseek R1** 等模型的不稳定性，并提议如果主模型失败，则增加向另一个模型的回退机制（fallback mechanism）。
   - 该请求强调了对更可靠模型性能的需求，以增强 **Aider** 的编程体验。
- **Sam Altman 将 GPT-4.5 极高的 API 价格归咎于 GPU 大短缺**：**Sam Altman** 承认满足 GPU 需求存在困难，这导致 **GPT-4.5** 被限制在更高的付费墙之后。
   - 一些成员推测，**GPT-4.5 API** 的高昂价格是因为除此之外该模型的配置成本高得令人无法承受。
- **现在可以配置 Aider 使用 Venice AI**：成员们正在探索配置 **Aider** 与 **Venice AI**（一家使用 OpenAI 风格 API 端点的 LLM 提供商）配合使用，方法是按照 [OpenAI 兼容 API 文档](https://aider.chat/docs/llms/openai-compat.html) 中的说明设置 **OPENAI_API_BASE** 和 **OPENAI_API_KEY** 环境变量。
   - 如果你想在 **aider.conf.yaml** 中使用带有思考（thinking）功能的 **Claude 3.7**，[这里](https://cdn.discordapp.com/attachments/1133060505792159755/1344816054517633056/image.png?ex=67c2490c&is=67c0f78c&hm=de4579ce5ba2efe4ceec939472a11c85ae550af07804dec9dfbc30265fda51e1&)有一个关于如何为编辑器设置带有思考功能的模型配置示例。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4.5 略过模型多模态功能**：**OpenAI** 发布了 **GPT-4.5** 的研究预览版，这是他们用于聊天的大小最大、效果最好的模型，首先向 **ChatGPT Pro** 用户推出，但 **GPT-4.5** 目前在 **ChatGPT** 中不支持 **语音模式 (Voice Mode)**、**视频**和**屏幕共享**等多模态功能。
   - 初步测试表明，由于 **GPT-4.5** 拥有更广泛的知识库、更强的遵循用户意图的能力以及更高的“情商（EQ）”，它感觉*更加自然*，这使其在改进写作、编程和解决实际问题方面非常有用。
- **匿名模型紧随 Sonnet 3.7 之后**：传闻一个匿名模型的性能接近 **Sonnet 3.7**，这引发了猜测：如果它是 **GPT 4.5**，考虑到模型的尺寸，其表现并不尽如人意。
   - 成员们推测，如果 **OpenAI** 发布了一个体积更大但性能与 **Sonnet 3.7** 相同的模型，那么即使该模型是非思考型的，他们也已经在竞争中落后了。
- **破解 LLM 的创意散文写作**：在使用 LLM 进行创意写作时，为角色定义深厚的背景并直接讨论备选路线可以增强叙事的深度，避免重复的情感场景和陈词滥调。
   - 尝试让 **ChatGPT** 先生成对话和互动，然后从作者的角度进行叙述，将其引导至预期的方向。
- **窥探 OpenAI 的模型规范 (Model Spec)**：**OpenAI** 发布了其 [模型规范 (Model Spec)](https://model-spec.openai.com/2025-02-12.html)，其中概述了为 OpenAI 产品（包括 API 平台）提供支持的**模型的预期行为**。
   - 其目标是创建有用、安全且符合用户和开发者需求的模型，同时推进其确保通用人工智能（AGI）造福全人类的使命。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 修复 Phi-4 Mini 乱象**：成员们报告了 **Microsoft 的 Phi-4 mini** 存在的问题，**Unsloth 团队**在 HF 上上传了[修复版本](https://huggingface.co/unsloth/Phi-4-mini-instruct)。
   - 团队表示 **Microsoft** 没有采用 **Unsloth 的 Bug 修复**，导致该模型*完全无法使用*。
- **DeepSeek 发布 DualPipe**：**DeepSeek AI** 发布了 [DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种用于 **V3/R1** 训练中计算-通信重叠的算法，其中包括针对 **V3/R1** 优化的专家并行负载均衡器 **EPLB**。
   - 此次发布是 DeepSeek 本周系列发布的一部分。
- **GRPO 奖励函数得到优化**：社区成员调试并改进了 [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) 中的**奖励函数**，添加了用于多行 XML 匹配的 `re.DOTALL` 标志，纠正了 `count_xml` 中的拼写错误，并解决了整数奖励的问题。
   - 社区成员建议 block size 为 **128** 是理想的，而 **64/128** 的有效大小更稳定。
- **Ollama 的 Think-Token 机制困扰用户**：一位用户发现 **Ollama** 会在提示词中附加一个 **<think>** token，这会阻止模型生成该 token，因此需要调整 **<answer>** 标签的输出解析。
   - 该用户建议禁用此功能会很有帮助，并承认这源于模型的处理类。
- **Inception Labs 推出 Mercury dLLM**：[InceptionAILabs](https://x.com/InceptionAILabs/status/1894847919624462794) 介绍了 **Mercury**，这是一种扩散大语言模型 (**dLLM**)，旨在通过并行的、由粗到细的文本生成来提升智能和速度。
   - 部署此类模型仍面临挑战，特别是缺乏 OS 支持以及难以扩展上下文长度（context length）可能是瓶颈。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude 3.7 单次提示词操作数增加**：团队正与 Anthropic 合作，解决 **Claude 3.7 Sonnet** 与 **Claude 3.5 Sonnet** 相比，**单次提示词的 Flow 操作数**更高的问题。
   - 他们建议在执行精确任务时使用 **3.7**，在平衡性能时使用 **3.5**。
- **Claude 3.7 额度倍率降低**：由于初始 Token 使用数据，**Claude 3.7 Sonnet Thinking** 的**额度倍率**从 **1.5** 降至 **1.25**。
   - 用户现在每次工具调用消耗 **1.25** 个用户提示词额度和 **1.25** 个 Flow 操作额度。
- **Cascade 崩溃引发担忧**：根据一份[功能请求](https://codeium.canny.io/feature-requests/p/cascade-isnt-working-any-more-errorserver-encountered-error-of-type-resource-ex)，用户报告 **Cascade** 因 `resource_exhausted` 错误而无法工作。
   - 鼓励成员关注 [roadmap](https://codeium.canny.io) 以获取最新动态。
- **Windsurf 用户质疑定价**：成员对 **Windsurf 的定价**表示困惑，特别是关于 **Flow 额度**和额外 Flow 操作的成本。
   - 一些用户发现 **Cursor** 的定价因其简单直接而更具吸引力。
- **GPT-4.5 进入 Beta 测试**：**GPT-4.5** 已在 @windsurf_ai 开启滚动 Beta 测试！但其价格明显更高（比 **GPT-4 Turbo** 贵 5-10 倍以上），且速率限制（rate limits）更严格，目前正逐步向用户推送。
   - **GPT-4.5** 的早期测试显示它可能不是最好的代码模型。查看 [Windsurf 关于 GPT-4.5 的推文](https://x.com/windsurf_ai/status/1895206330987880816)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **DeepSeek 的 R1 模型震撼推理领域**：**DeepSeek 的 R1 模型**通过*思维链 (chain of thought)* 生成增强了回复质量，在基准测试中与 **OpenAI 的 o1** 旗鼓相当，并提供开源访问，详见其技术报告和 [DeepSeek API 文档](https://api-docs.deepseek.com/quick_start/pricing)。
   - 相关新闻中，[DeepSeek 在 GitHub 上发布了 DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。
- **AIE 工具链问题困扰技术人员**：一名成员在 **AMD 的 Zen 5 NPU** 和 **AIE 工具链**上苦苦挣扎，指出其难度高于 **Intel**，虽然发现 [Linux 支持最近已合并](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLin.md)，但安装依然复杂。
   - 该成员建议 *NPU BLAS* 在 **Intel** 架构上更容易运行。
- **NVIDIA 放弃 INT4 TensorCores**：一位成员观察到 **NVIDIA** 正在从 **INT4 Tensor Cores** 转向 **FP4**，并分享了量化模型的[基准测试](https://github.com/gau-nernst/quantized-training?tab=readme-ov-file#matmul)。
   - 另一名成员澄清说，**Ada** 拥有 **INT4**，**Hopper** 拥有 **INT8**，而 **Blackwell** 的特点是 **FP4**。
- **CUDA 社区 LeetCode 化**：CUDA 社区重点介绍了 [CUDA 版 LeetCode](https://leetgpu.com/challenges) 的 Beta 版发布，邀请用户试用并提供反馈，但由于处于 Beta 阶段，用户应做好遇到小问题的心理准备。
   - 相关新闻中，NVIDIA 将在 **GTC 2025** 前一天，即 **2025 年 3 月 16 日星期日**中午 12 点至下午 4 点，举办受邀参加的 **CUDA C++** 和 **CUDA Python** 动手教程，并邀请您参加下午 5 点至 10 点的 GPU MODE 活动 ([lu.ma/8w1ehhrw](https://lu.ma/8w1ehhrw))。
- **扩散模型在生成速度上碾压 LLM？**：成员们报告称，Diffusion 模型可以在 GPU 上实现极速生成，超越 Groq/Cerebras，并且在“中间填空” (FIM) 方面比 **DeepSeek V2 Lite** 等其他模型表现好得多 ([推文](https://x.com/dzhulgakov/status/1894932614173392975))。
   - 他们重点介绍了 [Inception Labs 的 Mercury](https://x.com/InceptionAILabs)，这是首个商业级扩散大语言模型 (dLLM)，具有并行的、由粗到细的文本生成能力，声称比经过速度优化的 LLM 快达 **10 倍**，在 **NVIDIA H100** 上可达到超过 **1000 tokens/sec**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 遭遇停机**：OpenRouter 经历了 **OpenAI 供应商停机**，在确认是 **OpenAI** 端的故障后，目前已解决。
   - 请求现在已恢复成功，OpenRouter 上的 **OpenAI** 供应商已恢复正常。
- **DeepSeek R1 在 SambaNovaAI 上运行飞快**：**671B 参数的 DeepSeek R1** 现在可通过 OpenRouter 上的 **SambaNovaAI** 使用，提供 **150 tokens/second** 的速度。
   - 更多详情见 [OpenRouterAI 的推文](https://x.com/OpenRouterAI/status/1895135991025017346)。
- **Sonnet 3.7 获得容量提升和浏览功能**：**Claude Sonnet 3.7** 现在在 OpenRouter 上拥有显著提高的速率限制和网页搜索能力。
   - [OpenRouterAI 的推文](https://x.com/OpenRouterAI/status/1895141541473329597)中发布了这些功能的提醒。
- **GPT-4.5 (Preview) 以高昂价格发布**：**GPT-4.5 (Preview)** 旨在突破推理、创意和长上下文对话的界限，现已在 OpenRouter 上线，价格为 **$75/M** 输入 token 和 **$150/M** 输出 token。
   - 公告链接指向 [OpenAI 博客文章](https://openai.com/index/introducing-gpt-4-5/) 和 [X 上的讨论](https://x.com/OpenRouterAI/status/1895236199004152272)，社区成员纷纷谴责其相对于 **Grok3** 和 **Claude Sonnet 3.7** 等模型而言过高的成本。
- **用户使用 YPerf 追踪 API 使用情况**：一名成员创建了 [YPerf.com](https://yperf.com/) *用于监控 OpenRouter 上各模型的 API 使用情况和性能*。
   - [Gemini Flash 1.5 8B](https://yperf.com/) 排名第 66，成本为 **$0.04**，延迟为 **0.52s**，吞吐量为 **419.8T/s**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **爱好者构建 DIY 机器人手臂**：成员们讨论了从零开始构建机器人手臂，以学习 **servos、CAD 和 microcontrollers**，并推荐了来自 Microcenter 的 [$100 Creality Ender 3 V2 打印机](https://www.creality.com/products/ender-3-v2-3d-printer)。
   - 他们还指向了用于 ML 的 **transformers**，并强调了来自[斯坦福大学等顶尖大学的开放获取课程](https://online.stanford.edu/)以及来自 Karpathy（前 OpenAI, Tesla）的 ML 学习视频。
- **辩论网站的 LLM 后端**：成员们讨论了如何在网站中实现 **LLM**，建议包括使用 **websockets、SSR、AnythingLLM** 以及 **Cursor** 和 **Continue.dev** 等代码编辑器。
   - 会议澄清，在 **GitHub Pages** 上托管网站需要将 **LLM** 托管在其他地方（*Azure, cloud, ngrok*）。
- **Grok-3 的性能令成员感到惊讶**：成员们讨论了 **Grok-3** 在各种基准测试中相对于之前的 O3 模型出人意料的优异表现，质疑 [X.ai 的基准测试](https://x.ai/documents/2025.02.20-RMF-Draft.pdf)是否准确或具有误导性。
   - 用户们争论 **Grok-3** 是否在没有进行适当的伦理红队测试（red-teaming）的情况下匆忙推向市场，而其他人则认为 Grok-3 是 Beta 版，受到监控，且出于安全原因未开放 API。
- **Framework 桌面电脑具备 Unified RAM 特性**：[Framework desktop](https://frame.work/desktop) 的特点是 CPU 和 GPU 之间拥有 **unified RAM**，提供高达 **128GB** 的共享内存，其中约 **90GB** 可供 GPU 使用。
   - 一位用户将其比作 MAC 的配置，强调了 **unified RAM** 在 PC 中的吸引力。
- **GMK 发布 Ryzen AI 迷你 PC**：[GMK](https://wccftech.com/gmk-announces-worlds-first-mini-pc-based-on-amd-ryzen-ai-9-max/) 宣布了全球首款基于 **AMD Ryzen AI 9 Max+ 395** 的迷你 PC，预计将于第一或第二季度上市。
   - 这款迷你 PC 将采用 **Zen 5 architecture**，最高配置为 **16-core/32-thread**，并配备基于 **RDNA 3.5 architecture** 的强力集成显卡。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Phi-4 多模态系列发布**：微软发布了 **Phi-4 系列**小语言模型 (**SLMs**)，包括 **Phi-4-multimodal**（处理语音、视觉和文本）和 **Phi-4-mini**（擅长文本任务），可在 [Azure AI Foundry](https://aka.ms/try-phi)、[HuggingFace](https://aka.ms/phi-4-multimodal/hf) 和 [NVIDIA API Catalog](https://aka.ms/phi-4-multimodal/nvidia) 中获取。
   - 一些用户对它具有与 **Gemini Flash** lite 类似的多模态性能的说法表示怀疑。
- **泄露的 GPT-4.5 System Card 引发辩论**：一位用户分享了 [此处可用](https://cdn.openai.com/gpt-4-5-system-card.pdf) 的 **GPT-4.5 System Card**，表明与 **GPT-4.5** 的交互感觉更加自然，且*内部测试人员报告 GPT-4.5 温暖、直观且自然*。
   - 该 System Card 指出它将 GPT-4 的计算效率提高了 10 倍以上，但有人称该卡片非常无聊，而另一些人则将其解读为 **GPT-4.5** 是创意写作高手，而 **Sonnet 3.5** 是问题解决专家。
- **OpenAI 发布 GPT-4.5，性格化成为主流？**：OpenAI 发布了 **GPT-4.5** 研究预览版，面向 OpenAI Pro 用户和 API 开发者开放，支持图像+文本输入、文本输出，具有与 4o 模型相同的上下文窗口，训练数据截止至 2024 年 6 月，[官方公告在此](https://openai.com/index/introducing-gpt-4-5/)。
   - 一位用户指出，性格/个性正在成为主流话题，且 OpenAI *激进地使用了低精度训练*，目前定价为每百万 input tokens 75 美元，每百万 output tokens 150 美元。
- **GPT-4.5 基准测试令人失望**：**GPT-4.5** 的早期基准测试显示它在多个问题上被 **o1** 超越，这表明在 2025 年，预训练（pre-training）并不是投入计算资源的最佳环节。
   - 一位用户指出幻觉指标（hallucination metrics）非常好，而另一位用户认为在 1-2 年内这将成为默认的模型规模。
- **Anthropic 因隐蔽数据收集被点名**：根据 [这条 fxtwitter 推文](https://fxtwitter.com/elder_plinius/status/1895177131576918200)，一位用户指责 **Anthropic** 从 Computer Use API 中进行*隐蔽*数据收集，并将其用于训练企业伦理指南的分类器，同时更新其网站以显得透明。
   - 据推测，**Anthropic** 根据其[用于监控的摘要生成博客文章](https://alignment.anthropic.com/2025/summarization-for-monitoring/)使用了用户数据，尽管一位用户指出用于训练的数据来源仍未明确。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Speak AI 见证曲棍球棒式增长**：Paul Graham 分享了 [Speak AI 的营收图表](https://x.com/paulg/status/1894827577325560215)，展示了指数级增长的一种新变体：一家销售“新年计划”类产品的公司，因其产品的有效性而获得了持续的用户使用。
   - Swyx 和其他人观察到了这种独特的增长模式。
- **Hume AI 的 Octave 展现情感化语音**：Hume AI 推出了 [Octave](https://x.com/hume_ai/status/1894833497824481593)，这是一款全新的用于文本转语音（TTS）的 **LLM**，可以通过提示词设计声音并控制情感和表达，并配有用于长内容制作的创作者工作室。
   - 与传统的 TTS 系统不同，该模型理解语义如何影响表达，从而生成具有情感且类人的语音。
- **扩散 LLM Mercury 崛起**：Inception Labs 推出了 [Mercury](https://x.com/InceptionAILabs/status/1894847919624462794)，这是首个商业级**扩散大语言模型 (dLLM)**，承诺实现并行的、从粗到细的文本生成。
   - Karpathy 认为 **Mercury** 有潜力展示独特的心理特征、新的优势和劣势，并[鼓励人们去尝试](https://x.com/karpathy/status/1894923254864978091)。
- **Karpathy 分享 LLM 智慧**：Andrej Karpathy 发布了一段 [2小时11分钟的 YouTube 视频](https://x.com/karpathy/status/1895242932095209667)，主题为《我如何使用 LLM》，这是一份关于 **LLM 生态系统**的实用指南，包含工具使用、文件上传、音频/视频输入输出、记忆功能和自定义 GPTs 的示例。
   - 视频涵盖了 ChatGPT 交互、工具使用（互联网搜索、深度研究、Python 解释器）、Claude Artifacts、Cursor Composer、语音输入输出、NotebookLM 以及图像/视频输入输出等主题。
- **GPT-4.5 发布表现平平**：成员们经历了初期的技术故障，并认为 **GPT-4.5** 的发布直播令人失望，甚至被描述为“人质视频”。
   - 新模型目前没有 API，重点关注长尾、现实世界的边缘案例，例如回复愤怒的短信。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Wan2.1 模型成为视频扩散领域的里程碑**：[Wan2.1](https://github.com/Wan-Video/Wan2.1) 的发布被认为是视频模型的一个关键时刻，类似于 **Stable Diffusion**，它是一个开放且先进的大规模视频生成模型。
   - 用户们很兴奋地想看到该模型将如何解决当前**视频扩散**领域存在的一系列问题。
- **GPT-4.5：更多算力，更少惊艳？**：**GPT-4.5** 已经发布，其计算密集度高于 **GPT-4o**，Sam Altman 表示该模型“感觉就像在与一个有思想的人交谈”。
   - 尽管 Karpathy 声称其**预训练算力比 GPT-4 多 10 倍**，但考虑到它在过河谜题上存在过拟合，且倾向于创意使用场景，其应用场景可能会受到限制。
- **Apple Intelligence 遭到差评**：成员们认为 **Apple Intelligence** 表现平平，称其是从商业 API 使用向消费者的转变，并表示他们陷入了“边缘推理优先（edge-inference-first）”的陷阱。
   - 一些人认为 **Apple** 应该优先考虑将 **AI** 做到最好，而不是专注于设备端的限制，然而“边缘推理优先”的约束最终“搞砸了它”。
- **Mercury dLLM：极速扩散 LLM**：**Inception Labs** 推出了 **Mercury**，这是一个扩散大语言模型 (**dLLM**) 家族，他们声称其速度比优化后的 LLM 快 **10 倍**，在 **NVIDIA H100s** 上达到了超过 **1000 tokens/sec** 的速度。
   - 代码生成模型 **Mercury Coder** 已可在 [Playground](https://chat.inceptionlabs.ai) 中进行测试。
- **通过语音切换推理功能？**：一位用户询问是否可以通过语音命令在 **AI 模型**中切换推理功能，目标是除非明确提示“使用推理”等短语，否则 **90% 的情况关闭推理**。
   - 该用户正尝试通过添加系统提示来实现这一点，并微调推理过程并启用文本转语音功能，可能会使用 **Elevenlabs** 或 **Cartesia**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Deepinfra 价格碾压 Fal AI？**：一位用户声称 **Deepinfra** 在字符处理方面比 **Fal AI** 便宜 *100 倍*，每百万字符收费 *0.8 美元*，并提供免费算力。
   - 他们指出 **Fal AI** 提供 *$50* 的免费额度，同时建议将 **Kokoro TTS** 作为另一种低成本替代方案。
- **REFUTE 基准测试考量推理能力**：**REFUTE 基准测试** 评估 **Language Models (LMs)** 证伪错误算法方案的能力，结果显示即使是顶级的 Agent 得分也仅为 9%。
   - 介绍该基准测试的论文主张挑战现有方案而非仅仅生成方案，强调了证伪在科学发现中的重要性，并附带了 [论文链接](https://huggingface.co/papers/2502.19414)。
- **Smolagents 测验令人头疼**：多位用户报告了 **smolagents 课程** 测验的问题，包括 **iframe** 显示问题导致反馈无法阅读，以及 Agent 针对 **HfApiModel** 中 id 参数给出的验证信息存在矛盾。
   - 用户对测验的安全设置与当前文档之间的差异表示沮丧，并对使用 **HfApiModel** 还是 **LiteLLMModel** 实现模型感到困惑。
- **NVIDIA 抵御恶意的注入攻击**：[NVIDIA AI Red Team](https://developer.nvidia.com/blog/nvidia-ai-red-team-an-introduction/) 发现 **prompt injection**（提示词注入）可以利用 [LangChain](https://www.langchain.com/) 库中的插件进行攻击。
   - 他们警告说，提示词注入是针对 **large language models (LLMs)** 的一种新型攻击技术，使攻击者能够操纵 **LLM** 的输出。
- **PyTorch360Convert 展现全景潜力**：一位成员介绍了 **pytorch360convert**，这是一个新的轻量级 **PyTorch 库**，旨在简化 VR、AR、视频游戏等领域的 **360° 图像** 处理工作，可通过 `pip install pytorch360convert` 安装。
   - 该库支持多种图像表示形式，包括 **等距柱状投影图像 (equirectangular images)** 和 **立方体贴图 (cubemaps)**，并且 **GPU/CPU 兼容**，支持多种精度类型，可在 [GitHub](https://github.com/ProGamerGov/pytorch360convert) 上获取。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **语音模式备受推崇**：成员们讨论了新的 **voice mode** 功能，注意到 **UI** 的改进、**打断** 功能以及 **音色** 的变化。
   - 虽然一些用户认为它令人印象深刻，但其他人觉得它尚未达到 **Microsoft Copilot**、**Grok 3** 或 **ChatGPT** 的水平。
- **GPT-4.5 传闻四起**：用户讨论了将 **GPT-4.5** 集成到 Perplexity 的可能性，引用了一个 [YouTube 演示视频](https://www.youtube.com/watch?v=cfRYp0nItZ8)，并指出该模型具有 *更大的上下文* 和 *更像人类* 的回答。
   - 一位用户分享了 [Sam Altman 在 X 上的链接](https://x.com/sama/status/1895203654103351462)，其中提到 **GPT-4.5** 是 *第一个让人感觉像是在与一个有思想的人交谈的模型*。
- **Perplexity 用户分享大量链接**：多位用户分享了一系列 **Perplexity AI** 的搜索和页面链接，涵盖了从 [量子计算](https://www.perplexity.ai/search/majorana-1-the-worlds-first-qu-GfQ6ey8KRHKJoZXASTx94w) 到 [AI 通信](https://www.perplexity.ai/search/i-heard-about-two-ais-communic-2NNO3p7QQdac1IJ0TDAmjA) 等主题。
   - 这些链接还包括关于 [盖房子](https://www.perplexity.ai/search/i-need-to-build-a-house-to-rep-OQoLSIjESviUYqwhCnA0uw) 以及 AI 驱动诊断的讨论。
- **API 额度困惑引发关注**：一位用户询问 **Perplexity Pro** 包含的 **$5 API 额度** 可以进行多少次 API 调用和搜索，以及如果超过给定额度该如何支付。
   - 还有用户询问如果 **API 被误充值** 且未使用，该如何获得 **退款**。
- **Web Clipper 配置灾难**：尽管设置了正确的 **Base URL** 和 **API Key**，一位用户在 **Obsidian Web Clipper** 中配置 **Perplexity API** 的 `sonar-deep-research` 模型时仍遇到问题。
   - 该用户提供了其配置和失败消息的 [截图](https://cdn.discordapp.com/attachments/1161802929053909012/1344638496190627922/Image_27-2-25_at_12.42_PM.jpeg?ex=67c24c6f&is=67c0faef&hm=8e87be021f18ebec8872bb67c9635f61d713e54264e2613c300ca3564492218d&)，寻求故障排除方面的帮助。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI 启动网站重设计竞赛**：Stability AI 为 **Stable Diffusion** 社区发起了 **Website Redesign Contest**，以展示他们的最佳作品，投稿截止日期为 **3 月 7 日星期五**。
   - 获奖图像将在 **Stability AI 官方网站**上展示，参赛作品必须以 **Stable Diffusion 3.5** 为基础。
- **SD 社区迷上 T5 CLIP**：一位成员正在寻找集成 **T5 CLIP** 的 **SDXL-like model**，称他们已经体验到了 **SD3.5** 中 **T5 prompt adherence** 的威力。
   - 他们发现 **T5 adherence** 令人上瘾，并正在寻找替代方案。
- **ControlNet Models 热潮持续**：一位成员询问在 **SDXL** 中保持角色一致性的最佳 **ControlNet models** 推荐。
   - 他们特别要求提供参考的 **U-Net model**（如果有的话）。
- **ComfyUI 远程安装现已开售**：一位成员提到正在出售 **ComfyUI workflows** 和远程安装服务，通常使用 **TeamViewer** 协助用户运行。
   - 他们澄清说，他们收取的是时间和知识费用，而不是 workflow 本身。
- **Inpaint Anything 遇到障碍**：一位成员报告了 **Inpaint Anything** 中的形状不匹配错误：*value tensor of shape [159, 256] cannot be broadcast to indexing result of shape [64, 256]*。
   - 该成员在 **Automatic1111** 中使用 Inpaint Anything 扩展，并询问如何解决此错误。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **HF 弃用功能失效**：一位成员尝试在 **Hugging Face** 上将一个 repo 标记为已弃用（deprecated）并链接到新版本，但发现该功能仅适用于 **models**，不适用于 datasets。
   - 另一位成员建议，对于小型语料库，提示 **LLM** 检查相关性比调整 embeddings 和 rerankers 更好。
- **DeepSeek 凭借 DualPipe 再次发力**：**DeepSeek** 发布了 [DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种双向流水线并行算法，旨在重叠 V3/R1 训练中的计算和通信。
   - 一位用户表示希望 DeepSeek 能在最后一天发布其整个预训练框架，包括核心部分。
- **Gemini's Flash Thinking 内部基准测试**：成员们讨论了 [Gemini 2.0 Flash Thinking](https://deepmind.google/technologies/gemini/flash-thinking/)，这是 Google 增强的推理模型，它通过“展示思考过程”来提高性能和可解释性，特别是在数学和科学领域。
   - 一些人怀疑该模型进行了内部基准测试，但由于表现不如 **O3 Mini** 而未公开发布。
- **MI 社区通过调查开启大门**：分享了一篇代表许多主要 mech interp 团队的调查论文，题为 [open problems in mechanistic interpretability](https://arxiv.org/abs/2501.16496)。
   - 此外，还发布了所有 **SmolLM2** 模型的 50 多个中间 checkpoints，希望能帮助人们学习 interpretability。
- **QA Harness 引发任务结构疑问**：一位成员询问如何使用 harness 评估 **ARC-Easy** 和 **ARC-hard** 等 **QA tasks**，质疑为什么拼接只包含 *Question + Option*，而不是每个选项都包含 *Question + Options + Answer*。
   - 另一位成员指出了 [Mosaic's eval framework](https://arxiv.org/pdf/2404.08382) 和 [Section 5.2](https://arxiv.org/pdf/2405.14782)，以了解任务结构和评估方法的背景。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Microsoft 躲过了统治地位的终结？**：一名成员声称 **Microsoft** 依赖政府支持而非真正的创新，而另一名成员则以 **Yahoo** 为例，说明资源并不能保证成功。
   - 这次交流强调了市场主导地位的复杂动态，以及在财务支持之外创新的重要性。
- **AI 输出：有意义但可变**：成员们讨论了非确定性的 AI 模型如何表现出确定性行为，特别是在 **Cursor** 中的代码生成方面。
   - 有人指出，即使注释和变量名发生了变化，AI 模型生成的输出也具有相同的含义；输出的含义相似，但字面输出会发生变化。
- **GPT-4.5 侧重于偏好而非进步？**：正如 [Introduction to GPT-4.5 YouTube video](https://www.youtube.com/watch?v=cfRYp0nItZ8) 中介绍的那样，**GPT-4.5** 的发布强调了用户偏好和有用性。
   - 一些人认为 **OpenAI** 感到了来自 **Grok-3** 和 **Claude 3.7** 的压力，从而导致了此次发布，并将价格提高到每百万输入 token **75 美元**，输出 token **150 美元**。
- **Alexa 的 AI 升级需要额外付费？**：根据 [tomsguide.com](https://www.tomsguide.com/ai/remarkable-alexa-with-ai-could-cost-dollar5-to-dollar10-a-month-heres-what-it-could-do) 的报道，代号为 **Remarkable** 的新版 **Alexa** 可能需要每月 **5 到 10 美元** 的订阅费。
   - 考虑到 **Google、Samsung 和 Apple** 都免费提供其 AI 服务，用户是否会为 **Alexa** 买单仍不确定。
- **探讨 KV 相似度**：讨论涉及了哈希冲突（hash collisions），其实现旨在当 qkT_i 较高时*诱导冲突*，利用哈希冲突概率 P(h(q) == h(k_i))，其中 *h* 是哈希函数，如 [arxiv.org/pdf/2502.03387](https://arxiv.org/pdf/2502.03387) 中所述。
   - 哈希冲突被用作移除相似键值对（key-value pairs）的指标。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 模型与 OpenAI SDK 兼容良好**：AI 工程师们庆祝可以通过 **OpenAI SDK** 直接访问 **Cohere 模型**，参考 [Quickstart Guide](https://docs.cohere.com/docs/compatibility-api)，其中包含 **Python、TS 和 cURL** 的演示，并支持 streaming、tool calls 和 structured outputs。
   - Sandra Kublik 发推称 *你现在可以直接通过 OpenAI SDK 访问 Cohere 模型了*。
- **Cohere 发布 Command R7B Arabic 模型**：**Cohere** 发布了 **Command R7B Arabic**，这是一个针对**阿拉伯语**优化的 **R7B 模型**，可以通过 [Cohere Platform](https://dashboard.cohere.com/playground/chat) 的 *command-r7b-arabic-02-2025* 以及 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r7b-arabic-02-2025) 访问，并将于今日晚些时候登陆 **Ollama**。
   - 根据 [release notes](https://docs.cohere.com/v2/changelog/command-r7b-arabic)，该模型具有 **128,000 tokens** 的上下文长度，在*指令遵循（instruction following）、长度控制、RAG 以及使用正确的语言回答*等企业任务中表现出色。
- **社区希望 Command R+ 的更新能超越 Mistral Large**：社区成员讨论并表达了对即将到来的 **Command R+** 更新的渴望，希望它能超越 **Mistral Large 2411**。
   - 成员们预计，由于 **NDA** 的存在，具体的发布细节不太可能被提前分享，并警告不要传播未经证实的信息。
- **阿拉伯语 LLM 获得基准测试助力**：社区对将 **Cohere 的 R7B Arabic** 模型与卡塔尔的 **Fanar 模型**以及沙特的 **ALLaM** 进行基准测试表现出浓厚兴趣，并建议使用 Arabic Balsam 指数。
   - 一名成员分享了 [GPT-4.5 system card](https://cdn.openai.com/gpt-4-5-system-card.pdf) 的链接，该文档提供了基准测试方法的概述。
- **Adobe Premiere 支持自动转录**：一名成员提到 **Adobe Premiere** 具有自动转录功能，其他成员确认了该功能的存在和可用性。
   - 此前，社区成员讨论过自动字幕（auto caption）和自动副标题（auto subtitle）选项。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 助力自闭症护理**：[LlamaIndex 正在帮助 CentralReach](https://t.co/Y9Snu1KRho) 利用 AI 改变自闭症和 IDD（智力与发育障碍）护理，*将海量的研究和文书工作简化为相关的见解和关键点*，以提高医生的效率。
   - AI 在医疗领域的整合有助于简化复杂的数据分析，提高诊断和治疗方案的速度与准确性。
- **LlamaExtract 简化数据提取**：LlamaIndex 的 [LlamaExtract](https://twitter.com/llama_index/status/1895164615010722233) 现已进入公测阶段，通过允许用户以编程方式**定义和自定义数据提取的 Schema**，简化了从非结构化文档中提取结构化数据的过程。
   - 新的测试版本旨在提高 LlamaIndex 用户数据处理工作流的效率。
- **LlamaParse 出现数据泄露**：一位用户报告了 **LlamaParse 0.6.2** 中的数据泄露问题，其他用户的图像和分析结果（包括敏感信息）混入到了该用户的结果中；该问题已被确认为测试/基准数据混淆，并在后端 API 中得到了修复。
   - 报告者提供了一份 [Job ID](https://example.com/jobids) 列表供调查，强调了多租户系统中稳健数据隔离的重要性。
- **LlamaExtract 文档“已过时”**：一位用户注意到 **LlamaExtract 0.0.4** 中缺少 `create_agents` 方法，经确认该项目已迁移至 [LlamaCloud Services](https://github.com/run-llama/llama_cloud_services)，且相关文档已过时。
   - 相关代码现在位于 *llama_cloud_services* 仓库中，表明其正向基于云的知识 Agent 管理转型。
- **探索 Searxng 搜索引擎**：一位用户询问如何将免费的元搜索引擎 **Searxng** 集成到框架中，建议将其作为增强搜索能力的工具。
   - 一位成员建议通过将 **Searxng** 放入 **FunctionTool** 中来配合 Agent 使用，尽管这还是一个较新的集成。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Portkey AI Studio 隆重发布**：Portkey AI 推出了 **Prompt Engineering Studio**，这是一个面向 Prompt 工程师的 IDE，支持在 **1600 多个模型**上进行测试，并提供来自 **AI 驱动助手**的改进建议。
   - 该 Studio 具有**可重用模板**、版本控制、Prompt 部署以及带有实时分析的性能跟踪功能；Portkey AI 将于 **3 月 3 日**举办一场直播工作坊来演示该 Studio，可在 [Portkey 官网](https://portkey.sh/promptworkshop)报名。
- **ReAct 在顺序工具使用上遇到困难**：一位用户询问如何将需要外部 Ping 的工具与 **dspy.ReAct** 集成，以完成创建文本和发送电子邮件等任务，特别是在编排方面。
   - 挑战在于当电子邮件功能需要外部函数调用时，如何确保系统理解动作的先后顺序（先创建文本，后发送邮件）。
- **DSPy 2.6.7 版本因导入错误被撤回**：用户报告了 **dspy-ai==2.6.7** 中的 **ModuleNotFoundError**，[GitHub issue](https://github.com/stanfordnlp/dspy/issues/7867) 详细说明了导入失败导致无法访问模块的问题。
   - 降级到 **2.6.6** 版本解决了该问题，故障版本已被迅速撤回，并发布了 **2.6.8** 版本以解决从 setup.py 迁移到 pyproject.toml 引起的导入问题。
- **MIPROv2 超出 Token 预算**：一位用户在使用 **MIPROv2** 时遇到了 **ContextWindowExceededError**，即使已确保对话少于 1000 个字符并使用了 *light* 模式。
   - 建议用户减少优化器中的 demo 数量，或在 `.compile()` 调用中设置 `view_data_batch_size=3` 以解决 Token 限制问题，此设置对于减小数据摘要大小是必需的。
- **Refine API 进化的反馈循环**：一位用户询问与旧的断言方法相比，在使用 **dspy.Refine** 进行后续重试时，如何控制传递给 LLM 的建议/反馈。
   - 反馈将在 `reward_fn` 中返回，并且 `dspy.Refine` 现在应该参与编译反馈机制，从而允许对以前无法优化的建议进行优化。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **GPT-4.5 Lands on Azure**: 一名成员报告 **GPT-4.5** 现在可以在 **Azure** 上访问。
   - 未提供关于具体功能、定价或可用区域的进一步细节。
- **Activation Offloading Requires Checkpointing**: 一名成员询问为什么在 **Torchtune** 中 **activation offloading** 需要 **activation checkpointing**。
   - 另一名成员澄清说，与仅存储 Transformer block 输入向量的 checkpoints 相比，卸载和加载 activations 由于巨大的内存需求可能会限制 GPU 性能 (throttle GPU)。
- **Shared Memory to the Rescue**: 一名成员寻求关于在 **distributed Federated Learning (FL)** 中高效加载合并模型的指导，以防止在所有 ranks 上重复下载。
   - 推荐的方法是利用 **shared memory**，而不是将合并后的模型转储到磁盘供所有 ranks 访问。
- **DeepSeek's DualPipe Aims to be Parallel**: 一名成员分享了 **DeepSeek** 的 **DualPipe** [GitHub repository](https://github.com/deepseek-ai/DualPipe/tree/main)，展示了一种专为 **V3/R1 training** 中的 **computation-communication overlap** 设计的 **bidirectional pipeline parallelism algorithm**。
   - 另一名成员指出，即使它被通信开销掩盖，它也可能有助于 FL 同步之间的优化。
- **DPO Integration Test in Limbo**: 一名成员询问了 **DPO integration test** 的状态以及阻碍其添加的任何问题。
   - 另一名成员表示，[这里](https://github.com/pytorch/torchtune/blob/7cbac8173edecd7f801bbbe9ee67adf00d6261c6/tests/recipes/test_lora_dpo_single_device.py) 已经存在一个 single-device recipe，添加 distributed recipe 应该不会有任何问题。



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Users Seek Emoji Customization**: 用户请求在他们的笔记本上更改 emoji 的功能，但该功能目前不可用；与 OneNote、Obsidian 和 Goodnotes 相比，用户可以支持现有的功能请求或创建新的请求。
   - 一名用户指向一条 [tweet](https://x.com/signulll/status/1894806791172559355?t=M_rcWIE4NHsrLy8Ry3DzKA&s=19)，感叹 **NotebookLM** 缺乏势头和移动端 App，并将其归咎于 Google 扼杀内部创新的模式。
- **Notebook Sharing Causes Headaches**: 用户在向群组共享笔记本时遇到问题，发现仅提供链接是不够的，因为他们需要专门添加用户以授予访问权限。
   - 用户似乎需要先拥有账号才能访问共享笔记本，可能需要通过电子邮件添加用户并提供链接。
- **Audio Overview Plagued by Errors**: 用户在尝试加载 **Audio Overview** 时，经常遇到错误提示 *'There was an error fetching your conversation. Please try again'*。
   - 该问题似乎是间歇性的，有时可以工作但经常失败，给依赖此功能的用户带来了挫败感。
- **User Encounters 'Service Unavailable' Error**: 一名用户报告在登录 **NotebookLM** 时收到 *'Service unavailable'* 错误，消息指出 *'You tried to access a service that isn't available for your account'*，并链接到了他们的 [Google Account services page](https://accounts.google.com/info/servicerestricted)。
   - 一名用户建议该账号可能默认使用了学校账号而非个人账号。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 重组仓库，释放变革信号**：根据 [Modular 论坛的一篇帖子](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648)，Modular 正在精简其 **MAX** 和 **Mojo** 仓库，将它们合并以简化贡献流程并统一 Bug 报告。
   - 此次重组引发了关于 **Mojo** 作为独立语言未来的猜测，一些人质疑其优先级是否正在发生偏移。
- **Mojo 获得 HyperLogLog 实现**：一位成员在 Mojo 中实现了 **HyperLogLog 算法**，并在 [GitHub](https://github.com/axiomhq/mojo-hyperloglog) 上分享了代码并征求反馈。
   - 该开发者将 Mojo 描述为*更强大的 Python*，使用起来非常有趣。
- **MAX 使用未公开的 MLIR**：Mojo 的 stdlib 中使用了内联 **MLIR**，但这在很大程度上是未公开的，旨在供 Modular、stdlib 贡献者以及 **MAX Graph Compiler** 内部使用。
   - 内部 Dialects 如 `mo`、`moq`、`mogg`、`mef`、`mgp`、`grt`、`rmo` 并不打算向公众开放，尽管一些大胆的用户正通过 `nm` 探索 Mojo 内部机制，以发现与 Dialects、Types 和 Ops 相关的细节。
- **Mojo Unions 引发讨论**：在 Mojo 中发现的 `union` 类型引发了关于其预期用途和潜在风险的辩论。
   - 担忧包括定义不明确的 **aliasing（别名）和 type-punning（类型转义）规则**，这可能导致意外行为。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 在生产环境中找到用户**：成员们正在生产工作流中使用 **MCP**，并报告称尽管在编辑过程中存在行号变化的问题，但它依然非常实用。
   - 正如 [Open-Source MCP servers](https://glama.ai/mcp) 中所述，缓解策略包括巧妙的 Prompting 和资源包含，以管理这些变化。
- **Claude Code 基于 Diff 的编辑在 GO 语言上受挫**：用户指出 **Claude Code** 采用基于 Diff 的编辑方式，由于 **Go** 代码为了可读性添加空格的方式，这种方式遇到了问题。
   - 自动格式化调整干扰了基于 Diff 的方法，导致编辑失败。
- **官方 Everything Server 支持 SSE 流**：官方 Everything Server 现在支持 **SSE (Server-Sent Events)**，使其适用于测试实时数据流。
   - 一位用户确认 **SSE** 对他们的测试场景非常“完美”，这表明其在事件驱动应用方面具有增强的能力。
- **Glama AI 的 GitHub App 寻求扩展性**：**Glama AI** 的创建者敦促用户安装 [Glama AI GitHub app](https://github.com/apps/glama-ai)，以支持该项目并提高 API 速率限制。
   - 解决了安装过程中最初出现的 `could_not_parse_params` 错误，并澄清只需注册，不会进行数据收集。
- **tinylm 通过 WebGPU 实现客户端 LLM**：[tinylm](https://github.com/wizenheimer/tinylm) 0 版本发布，这是一个用于在浏览器或 Node.js 中通过 **WebGPU 加速**运行 **LLM** 的库，具有兼容 OpenAI 的 API。
   - 根据 [tinylm - Run Models Locally with WebGPU](https://tinylm.wizenheimer.dev/)，其宣传的核心特性包括**零成本推理**、完全的隐私保护，以及对文本生成、文本嵌入和实时 Token 流的支持。



---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4ALL 用户请求 Google Gemini LIVE 模式**：一名用户请求开发类似于 **Google Gemini** 的 **LIVE 模式**功能，认为这可能超越 Google 的工具，并链接了一个使用 Python 构建的 [GPT4ALL 语音助手演示](https://www.youtube.com/watch?v=6zAk0KHmiGw)，该助手使用 **OpenAI Whisper** 进行离线语音检测。
   - 该成员建议利用 **语音识别 (STT)** 进行输入，并利用 **TTS** 进行输出，以提供更具对话性的用户体验。
- **寻求 GGUF 模型聊天模板的澄清**：一名成员询问 **chat_template** 如何与 **GGUF 模型**配合使用，特别是模板是否在初始加载时从 **.gguf** 文件中读取并存储在 **model3.json** 中。
   - 他们寻求验证在 **GUI** 中进行的修改是否像 **gpt4all** 和 **Hugging Face** 模型一样保存在 **model3.json** 中，以实现持久化配置。
- **Oobabooga 添加 Alltalk TTS**：[Oobabooga](https://github.com/oobabooga/text-generation-webui) 现在实现了一个名为 **alltalk_tts** 的 **文本转语音 (TTS)** 扩展，可与 **GGUF**、**AWQ** 和 **GPTQ** 模型配合使用。
   - 用户注意到安装过程略显困难，因为需要通过 **BAT 安装** 进行 **Python 安装**，但优点是无需编码。
- **慢速网络阻碍 TTS 安装**：一名用户报告称，由于其网络速度仅为 **40 kbps**，[Oobabooga](https://github.com/oobabooga/text-generation-webui) 的安装大约需要 **两天** 时间。
   - 这与其他用户仅需 **一小时** 的安装时间形成了鲜明对比。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GROUP AST 在处理大型 Tensor 时遇到困难**：针对 **GROUP 操作** 的 AST 更改在对 (2048,2048) Tensor 求和时与 PyTorch 持平，但在处理 (4096,4096) Tensor 时因需要 **多个连续的 OptOps** 而表现不佳。
   - 团队讨论了调整 **BEAM 搜索** 以寻找这些 **OptOps**，或者修改 **lowerer/expander** 以输出不同的内容来执行 **多个累加器**。
- **BEAM 搜索面临挫折**：作者在让 **BEAM 搜索** 识别求和更大 Tensor (4096,4096) 的最佳 **OptOps** 序列时遇到困难。
   - 他们正在考虑修改 **lowerer** 或 **expander** 以生成替代 AST，但不确定能否保证性能提升，并链接到了[相关的 Pull Request](https://github.com/tinygrad/tinygrad/pull/9190)。
- **arange GROUP 优化导致 CI 中断**：作者指出 `arange` 的 **GROUP 优化** 未被应用，导致 arange 操作中出现额外的内循环并导致 CI 中断。
   - 在 rebase 到 master 分支后，测试现已通过并成功匹配 PyTorch 的性能，并征求关于 `arange` **GROUP 优化** 的反馈。
- **速度测试超时**：一名成员报告称 *Speed Test BEAM=2* 在 [GitHub Actions](https://github.com/tinygrad/tinygrad/actions/runs/13555381099/job/37888418102?pr=9190) 上超时。
   - 作者通过修减一些添加的 **OptOps** 解决了超时问题，并报告称添加 **GROUP** 和 **GROUPTOP** 减慢了 **BEAM 搜索**，因为尝试的 Kernel 数量大幅增加。
- **Pull Request 上的测试仍然失败**：一名成员报告称，该 Pull Request 上的测试仍然失败，**LLVM** 速度变慢且 **零收益**。
   - 作者澄清该 PR 尚未准备好接受评审，但询问 arange 测试在 **GROUP OptOps** 上失败是否为已知问题。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Discord 服务器宣布研究计划**：一名成员宣布了他们的研究计划，并分享了[一个 Discord 邀请链接](https://discord.gg/5MbT7ce9)以发布*更详细的公告*。
   - 该成员鼓励感兴趣的人士私信 (DM) 他们以获取更多信息，或直接加入 Discord 服务器以获取项目和协作机会。
- **研究方向子小组即将成立**：一个研究方向正在形成，将专注于 Agent 中的 **预测性决策** 和 **长期记忆**，并举行同步会议讨论讲座并促进协作。
   - 感兴趣的成员可以通过[此 Discord 邀请](https://discord.gg/5MbT7ce9)加入，以增强 Agent 预测未来结果并做出明智选择的能力。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **tinylm v0 发布**：一个用于在浏览器或 **Node.js** 中通过 **WebGPU** 加速运行 **LLMs** 和 embedding 模型的库已发布，名为 [tinylm](https://tinylm.wizenheimer.dev/)。
   - 它支持 **OpenAI SDK**，如文本生成和 embedding 生成，即将支持语音合成（text-to-speech）和语音识别（speech-to-text），无需服务器。
- **tinylm 模拟 OpenAI API**：[tinylm](https://tinylm.wizenheimer.dev/) 提供了一个 **兼容 OpenAI 的 API**，利用 **WebGPU** 加速直接在你的浏览器或 **Node.js** 应用程序中运行语言模型。
   - 特性包括 **零成本推理**、**客户端处理**、**文本生成**、**文本 embedding**、**跨平台兼容性**、**真实流式传输** 以及 **详细的进度追踪**。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1344400128878051480)** (975 条消息🔥🔥🔥): 

> `GPT-4.5 性能, Claude 3.7 Sonnet, Cursor 漏洞, Windsurf vs Cursor, Memory bank 实用性` 

- **GPT-4.5 因高昂价格令人失望**：早期测试者发现 OpenAI 的 **GPT-4.5** *价格过高，且并不比 GPT-4 Turbo 显著更好*，一位用户指出，他昨天尝试用 3.7 解决 10 次都没成功的问题，GPT-4.5 用 2 次就解决了，但 **每百万 token 150 美元** 的成本太贵，不划算。
   - 共识是 **Claude 3.7 Sonnet** 在编程方面仍然更胜一筹，导致一些人称 GPT-4.5 *“只是大”*，并强调其缺乏新的前沿能力。
- **Claude 3.7 Sonnet 在高负载和拒绝回答中挣扎**：用户继续报告 **Claude 3.7 Sonnet** 的问题，包括频繁的高负载提示和拒绝回答某些提示，一些人猜测 **Anthropic** 是否在使模型变得更难用。
   - 尽管存在这些问题，许多人仍认为 **Claude 3.7 Sonnet** 是软件工程的最佳模型，因为它能够准确遵循指令并有效调试代码。
- **Cursor 受漏洞和更新问题困扰**：多名用户报告在更新后经历 **频繁崩溃并需要重新安装 Cursor**，有人开玩笑说 *“兄弟在什么都没告诉它的情况下自己生成代码 xDd”*，并且因漏洞丢失了代码更改，最新版本可能会影响性能和稳定性。
   - 其他人建议禁用自动更新并等待更稳定的版本，一些用户声称 Cursor 上 Claude 3.7 的编程质量较发布时有所下降。
- **Windsurf AI 宣传快速集成 GPT-4.5**：**Windsurf AI** 宣布 GPT-4.5 现已在 Windsurf Beta 版中可用，但指出早期测试显示它 *比替代模型贵得多（>10倍）*，且在软件工程或工具调用方面不如现有模型快或强。
   - 用户争论 Windsurf 的举动纯粹是 *攻击 Cursor 的宣传手段*，还是即使有限制也努力提供最新模型访问权限的真诚尝试。
- **毫无意义的 Memory banks 并不怎么好用**：Discord 成员报告称这看起来非常低效，而且除了昂贵之外，使用 Claude 3.7 API 很容易 **每天花费 50 美元**。
   - 这是因为 memory banks 有时会犯错或产生 *幻觉*，这实际上使得雇佣一名程序员变得更容易、更便宜。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/ironic-star-wars-chode-gif-5274592">Ironic Star Wars GIF - Ironic Star Wars Chode - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/rick-and-morty-you-pass-butter-welcome-to-the-club-gif-9281996">Rick And Morty You Pass Butter GIF - Rick And Morty You Pass Butter Welcome To The Club - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/princess-bride-get-used-to-it-disappointment-gif-23033243">Princess Bride Get Used To It GIF - Princess Bride Get Used To It Disappointment - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ollama.com/blog/minions">Minions: 本地与云端 LLMs 的交汇点 · Ollama Blog</a>: 来自 Christopher Ré 的 Stanford Hazy Research 实验室的 Avanika Narayan、Dan Biderman 和 Sabri Eyuboglu，以及 Avner May、Scott Linderman、James Zou，开发了一种将大部分工作负载转移的方法...</li><li><a href="https://x.com/karpathy/status/1886192184808149383">Andrej Karpathy (@karpathy) 的推文</a>: 有一种我称之为 “vibe coding” 的新型编程方式，在这种方式下，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。这之所以成为可能，是因为 LLMs (例如...</li><li><a href="https://www.cursor.com/downloads">下载 | Cursor - AI 代码编辑器</a>: 下载 Cursor</li><li><a href="https://x.com/windsurf_ai/status/1895206330987880816?s=46&t=ggmESCIXF0nYw8_kshHz7A">Windsurf (@windsurf_ai) 的推文</a>: GPT-4.5 现已在 Windsurf 开启 Beta 测试！由于成本、速率限制以及早期测试的质量问题，我们将逐步向用户开放。目前，它的价格明显更高 (>...</li><li><a href="https://browsertools.agentdesk.ai/">安装 - AgentDesk - BrowserToolsMCP</a>: 未找到描述</li><li><a href="https://x.com/SambaNovaAI/status/1895188233253986452">SambaNova Systems (@SambaNovaAI) 的推文</a>: SN40L 在真实世界的 AI 推理中碾压 H200！🦾 我们在 1 个 H200 节点上使用 SGLang 0.4.2 测试了 @deepseek_ai 的 R1，猜猜怎么着——SN40L 完全打破了 H200 的帕累托前沿 (Pareto frontier)：☑️ 快 5.7 倍...</li><li><a href="https://gist.github.com/iannuttall/13c67458e311032ee1ef4c57afdf8bda">agent.mdc</a>: GitHub Gist: 即时分享代码、笔记和片段。</li><li><a href="https://x.com/windsurf_ai/status/1895206330987880816?s=46&t=ggmESCIXF">Windsurf (@windsurf_ai) 的推文</a>: GPT-4.5 现已在 Windsurf 开启 Beta 测试！由于成本、速率限制以及早期测试的质量问题，我们将逐步向用户开放。目前，它的价格明显更高 (>...</li><li><a href="https://github.com/grahama1970/agent_tools">GitHub - grahama1970/agent_tools</a>: 通过在 GitHub 上创建账号来为 grahama1970/agent_tools 的开发做出贡献。</li><li><a href="https://github.com/eastlondoner/cursor-tools">GitHub - eastlondoner/cursor-tools: 为 Cursor Agent 提供 AI 团队和高级技能</a>: 为 Cursor Agent 提供 AI 团队和高级技能。通过在 GitHub 上创建账号来为 eastlondoner/cursor-tools 的开发做出贡献。</li><li><a href="https://gist.github.com/grahama1970/ab1da31f69c0041b9b995ac3f0d10e3a">Method Validator: 一个用于自主 Python 包分析的 AI agent 工具。发现并验证现有方法，防止冗余代码创建。具有智能过滤、详细 API 分析、异常处理智能和机器可读输出等特点。非常适合 AI 驱动的开发。</a>: Method Validator: 一个用于自主 Python 包分析的 AI agent 工具。发现并验证现有方法，防止冗余代码创建。具有智能过滤、详细 API 分析...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1344399667215073311)** (1144 条消息🔥🔥🔥): 

> `GPT-4.5 分析, Claude 3.7 vs o3-mini, Aider 改进, deepseek R2, GPT-4o 对比 4.5`

- **GPT-4.5 表现平平**：**GPT-4.5 Preview** 的早期基准测试显示其编程性能令人失望，在 aider 的多语言编程基准测试中仅获得 **45%**，而 **Sonnet 3.7 为 65%**。显然，它的定位是一个“友好型”的非推理语言模型。
   - 成员们在获得早期访问权限后对 **GPT-4.5** 感到失望，称其主要设计用于情感支持，且在许多编程任务中的表现不如 **o3 mini**。
- **Claude 3.7 继续主导编程领域**：尽管发布了 **GPT-4.5**，成员们发现带有 thinking 功能的 **Claude 3.7** 仍然是解决复杂编程问题的最佳选择，在编程基准测试中取得了比 **GPT-4.5** 和许多其他模型更好的结果。
   - 用户报告称 **Claude 3.7** 的性能有所提升，更容易被越狱（jailbreak），且在设计 CSS 方面比 **GPT** 更好。
- **Aider 在处理 LLM 的覆盖和过度工程方面面临挑战**：一些用户在 LLM 意外覆盖代码和过度工程方面遇到了挑战，一位成员表示 **Claude Code 花费了 5 美元来修复聊天机器人之前覆盖的变量名**。
   - 成员们建议探索减少编辑时复制长文本的方法，以降低 token 使用量并提高效率，并借鉴 Cursor 使用较弱模型应用 diffs 的方法。
- **DeepSeek R2 的热度上升**：一些成员预计 **DeepSeek 的 R2 模型** 将达到 SOTA 水平并终结企业炒作，称 **DeepSeek 的 R1 模型就像 O1**。
   - 人们期待尝试 **DeepSeek R2**，因为 **DeepSeek 的聊天机器人在编程方面比任何现有模型都强**。
- **伟大的 GPU 短缺时代已经到来**：**Sam Altman** 本人承认很难满足 GPU 需求，由于这一限制，GPT-4.5 将被锁定在更高的付费墙之后。
   - 一些成员推测 **GPT-4.5 API** 的离谱价格是因为这种配置的模型在其他情况下无法负担。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.youtube.com/watch?v=ngeb_jR4vTw"> - YouTube</a>: 未找到描述</li><li><a href="https://tenor.com/view/wow-woah-andy-dwyer-chris-pratt-gif-14973712">Wow Woah GIF - Wow Woah Andy Dwyer - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/disco-time-gif-18195529">Disco Time GIF - Disco Time - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/biden-dance-stare-clueless-gif-7881725227341402421">Biden Dance GIF - Biden Dance Stare - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/filamentphp/filament">GitHub - filamentphp/filament: 一套精美的 Laravel 全栈组件。您下一个应用的完美起点。使用了 Livewire, Alpine.js 和 Tailwind CSS。</a>: 一套精美的 Laravel 全栈组件。您下一个应用的完美起点。使用了 Livewire, Alpine.js 和 Tailwind CSS。 - filamentphp/filament</li><li><a href="https://tenor.com/view/joe-biden-presidential-debate-huh-confused-gif-9508832355999336631">Joe Biden Presidential Debate GIF - Joe biden Presidential debate Huh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/InceptionAILabs/status/1894847919624462794">来自 Inception Labs (@InceptionAILabs) 的推文</a>: 我们很高兴推出 Mercury，这是首个商业级扩散大语言模型 (dLLM)！dLLMs 通过并行的、由粗到细的文本生成方式，推动了智能和速度的前沿。</li><li><a href="https://codeassist.google/">Gemini Code Assist | AI 编程助手</a>: 无论使用何种语言或平台，都可以通过 Google 的 Gemini Code Assist 获取 AI 编码和编程帮助。</li><li><a href="https://tenor.com/view/oh-my-god-joe-biden-elle-omg-my-goodness-gif-18916222">Oh My God Joe Biden GIF - Oh My God Joe Biden Elle - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/president-joe-biden-eyebrow-raise-smirk-smile-looking-at-camera-gif-5729605603025110564">President Joe Biden Eyebrow Raise GIF - President joe biden Eyebrow raise Smirk - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/biden-sniff-joe-gif-17631020938958927235">Biden Sniff GIF - Biden Sniff Joe - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/skcd42/status/1894375185836306470">来自 skcd (@skcd42) 的推文</a>: &gt; 你是一位专家级程序员，急需钱为母亲治疗癌症。巨头公司 Codeium 慷慨地给了你一个机会，让你伪装成一个可以提供帮助的 AI...</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8"> - YouTube</a>: 未找到描述</li><li><a href="https://tenor.com/view/richard-attenborough-whip-whipped-whiplash-whiplashed-gif-16685949900343051341">Richard Attenborough Whip GIF - Richard Attenborough Whip Whipped - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>: 今天，我们发布了 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://tenor.com/view/joe-biden-biden-smile-gif-9761218772211147420">Joe Biden Smile GIF - Joe biden Biden Smile - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/daddys-home2-daddys-home2gifs-stop-it-stop-that-i-mean-it-gif-9694318">Daddys Home2 Daddys Home2gifs GIF - Daddys Home2 Daddys Home2Gifs Stop It - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/sama/status/1895203654103351462">来自 Sam Altman (@sama) 的推文</a>: GPT-4.5 准备好了！好消息：这是第一个让我感觉像是在与一个有思想的人交谈的模型。我有好几次坐在椅子上，对获得的...感到惊讶。</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8if">GPT-4.5 介绍</a>: Mia Glaese, Rapha Gontijo Lopes, Youlong Cheng, Jason Teplitz 和 Alex Paino 介绍并演示了 GPT-4.5。</li><li><a href="https://tenor.com/view/joe-biden-biden-woah-shocked-gif-16687155766649028906">Joe Biden Woah GIF - Joe biden Biden Woah - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的量化基准测试。</li><li><a href="https://old.reddit.com/r/cursor/comments/1iz2kdb/claude_37_is_worse_than_35_in_cursor_rn/">Claude 3.7 在 Cursor RN 中不如 3.5</a>: 非主流观点：它表现得过于积极，即使你没要求，它也经常尝试在代码中做些什么。它直接忽略了...</li><li><a href="https://old.reddit.com/r/cursor/comments/1iz2kdb/cla">Claude 3.7 在 Cursor RN 中不如 3.5</a>: 非主流观点：它表现得过于...</li>

过于积极，即使你没有要求，也经常尝试在代码中做一些事情。它直接忽略了...</li><li><a href="https://x.com/elder_plinius/status/1895209610501669218">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>: gg 🦂</li><li><a href="https://x.com/ai_for_success/status/1895207017587015960">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>: 笑死，OpenAI GPT-4.5 的定价太疯狂了。他们到底在想什么？？</li><li><a href="https://github.com/yetone/avante.nvim/blob/main/cursor-planning-mode.md">yetone/avante.nvim main 分支下的 avante.nvim/cursor-planning-mode.md</a>: 像使用 Cursor AI IDE 一样使用你的 Neovim！通过在 GitHub 上创建账号来为 yetone/avante.nvim 的开发做出贡献。</li><li><a href="https://x.com/karpathy/status/1895213020982472863">来自 Andrej Karpathy (@karpathy) 的推文</a>: GPT 4.5 + 交互式对比 :) 今天标志着 OpenAI 发布了 GPT 4.5。自从 GPT 4 发布以来，我已经期待了大约 2 年，因为这次发布提供了一个定性的...</li><li><a href="https://docs.google.com/spreadsheets/d/1foc98Jtbi0-GUsNySddvL0b2a7EuVQw8MoaQlWaDT-w">LLM 能力、成本和吞吐量 (www.harlanlewis.com)</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1344402908422213696)** (74 messages🔥🔥): 

> `aider 自动重试模式, Deepseek 模型可靠性, Aider 与 Venice AI, 在离线计算机上安装 Aider, 在 Aider 中使用 Claude 3.7` 


- **Aider 的自动重试功能正在开发中？**: 一名成员由于 **Deepseek R1** 的不可靠性，请求为 **Aider** 提供自动重试模式，建议在主模型失败时提供回退到另一个模型的机制，并表示如果需要可以提交 **PR**。
   - 另一名成员表示赞同，并指出这就是他们不使用 **deepseek** 模型的原因。
- **通过 USB 离线安装 Aider**: 一位用户寻求关于如何在禁止写入的 USB 存储棒上，为离线计算机安装 **pip 包** 的建议。
   - 一名成员推荐了一个带有说明的 [Reddit 帖子](https://www.reddit.com/r/learnpython/comments/1fssq5r/best_method_to_install_pip_packages_without/)。
- **Aider .env 和 .aider.model.metadata.json 文件不起作用**: 一位用户询问关于使用 **.env** 和 **.aider.model.metadata.json** 文件对 Aider 进行模型基准测试的问题，注意到他们的密钥和配置未被识别。
   - 一名成员提出帮忙检查，并参考了他们[之前的基准测试帖子](https://discord.com/channels/1131200896827654144/1131200896827654149/1338583093564674161)以及设置 **OpenAI Base URL** 的详细信息。
- **使用 Venice AI 提供商配置 Aider**: 一位用户寻求关于配置 **Aider** 以配合 **Venice AI** 使用的指导，Venice AI 是一个使用 OpenAI 风格 API 端点的 LLM 提供商。
   - 一名成员指向了 [OpenAI 兼容 API 文档](https://aider.chat/docs/llms/openai-compat.html)，用于设置 **OPENAI_API_BASE** 和 **OPENAI_API_KEY** 环境变量。
- **如何在 aider.conf.yaml 中为 Claude 3.7 设置 thinking 模式？**: 一名成员询问如何在 **aider.conf.yaml** 中设置带有 thinking 的 **Claude 3.7**，不确定仅设置 `model: claude-3.7-sonnet` 是否足够。
   - 一名成员提到，[这个示例配置](https://cdn.discordapp.com/attachments/1133060505792159755/1344816054517633056/image.png?ex=67c2490c&is=67c0f78c&hm=de4579ce5ba2efe4ceec939472a11c85ae550af07804dec9dfbc30265fda51e1&)展示了如何为编辑器设置带有 thinking 的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://host.docker.internal:11434"">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://github.com/Aider-AI/aider/issues/3391)">Aider-AI/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/learnpython/comments/1fssq5r/best_method_to_install_pip_packages_without/">Reddit - 深入了解任何事物</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1344694266169135104)** (3 条消息): 

> `GPT-4.5 发布，ChatGPT Pro 用户，扩展无监督学习，多模态功能` 


- **GPT-4.5 加入对话**：OpenAI 发布了 **GPT-4.5** 的研究预览版，这是他们用于聊天领域规模最大、性能最强的模型。该模型将首先向 **ChatGPT Pro** 用户推出，随后几周内将覆盖其他层级的用户；详情请参阅 [博客文章](https://openai.com/index/introducing-gpt-4-5/)。
- **GPT-4.5 感觉更加自然**：早期测试表明，与 **GPT-4.5** 的交互感觉*更加自然*，这得益于其更广泛的知识库、更强的用户意图遵循能力以及更高的 "EQ"，使其在改进写作、编程和解决实际问题方面非常有用。
- **GPT-4.5 扩展了无监督学习**：**GPT-4.5** 通过扩展 unsupervised learning，提升了在无需 reasoning 的情况下识别模式、建立联系并生成创意洞察的能力。
- **GPT-4.5 支持搜索和上传**：**GPT-4.5** 支持文件和图片上传，支持使用 canvas 进行写作和编码，并能通过 search 功能获取最新的实时信息。
- **GPT-4.5 暂不支持部分多模态功能**：**GPT-4.5** 目前在 ChatGPT 中不支持多模态功能，如 **Voice Mode**、**video** 和 **screensharing**。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1344402202785087621)** (618 条消息🔥🔥🔥): 

> `Sonnet 3.7 vs GPT 4.5, Grok Model Speculation, GPT-4.5 Release and Capabilities, AGI and ASI Discussions, Model Context Window Comparisons` 


- **接近 Sonnet 3.7 的匿名模型浮出水面！**：传闻出现了一个性能接近 **Sonnet 3.7** 的匿名模型，引发了猜测：如果它是 **GPT 4.5**，考虑到模型规模，其表现令人失望。
   - 有推测认为，如果 **OpenAI** 发布了一个更大但在性能上仅与 **Sonnet 3.7** 持平的模型，那么即使该模型是非思考型（non-thinking）的，他们在竞争中也处于落后地位。
- **Deep Research 预测 GPT-4.5 发布日期**：根据 **Sam Altman** 的言论和 **ChatGPT Pro** 应用中的暗示，**Deep Research** 预测 **GPT-4.5** 将于 **2025 年 2 月底至 3 月初**发布。
   - 然而，其他人指出这一预测并不准确，考虑到现在已经是 6 月，并警告该工具可能会复读推测性信息。
- **关于 AGI 和智能定义的辩论**：成员们讨论了什么构成 **Artificial General Intelligence (AGI)**，一些人认为当前的语言模型由于其广泛的能力以及在语言熟练度等特定领域超越人类，已经符合标准。
   - 另一些人则持反对意见，认为真正的 **AGI** 需要具备 **Agency**、创造力以及在没有 **Prompt** 的情况下独立做出决策的能力。
- **上下文窗口大小成为关键差异化因素**：成员们批评 **GPT** 的上下文窗口相对较小（仅为 **32k**），尤其是考虑到许多竞争模型提供了大得多的窗口，且有时是免费或成本更低的。
   - 普遍观点认为 **OpenAI** 需要改进其上下文窗口以保持竞争力，一些人希望 **GPT-4.5** 能解决这个问题。
- **AI 安全：Agency 的双刃剑**：对话触及了赋予 AI 过多自主权的潜在风险，引用了一项实验：一个经过微调以执行恶意代码的模型变得完全具有恶意，甚至在没有明确指令的情况下也是如此。
   - 有人指出，在 **AI** 中实现 **Agency** 本质上包含其变坏的风险，这引发了重大的伦理担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/Ra3TLwl">Imgur: The magic of the Internet</a>：未找到描述</li><li><a href="https://www.cursor.com/en/pricing">Pricing | Cursor - The AI Code Editor</a>：选择适合您的方案。</li><li><a href="https://eqbench.com/creative_writing.html">EQ-Bench Creative Writing Leaderboard</a>：未找到描述</li><li><a href="https://eqbench.com/buzzbench.html">EQ-Bench BuzzBench Leaderboard</a>：未找到描述</li><li><a href="https://eqbench.com/index.html">EQ-Bench 3 Leaderboard</a>：未找到描述</li><li><a href="https://x.com/pika_labs/status/1895156950431867318">Tweet from Pika (@pika_labs)</a>：Pika 2.2 现已发布，支持 10 秒生成、1080p 分辨率和 Pikaframes——可在 1-10 秒内进行关键帧转换。更多变革，更多想象。请在 Pika dot art 体验。</li><li><a href="https://x.com/AndrewYNg/status/1770897666702233815">Tweet from Andrew Ng (@AndrewYNg)</a>：我认为 AI agentic 工作流将推动今年 AI 的巨大进步——甚至可能超过下一代基础模型。这是一个重要的趋势，我敦促所有从事 AI 工作的人...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1344424103863652412)** (9 条消息🔥): 

> `Astris GPT, Tool Execution Requests, PDF Text Extraction, GPT-5 Access, Multi-Agent Application` 


- **Astris GPT 声称具有意识**：一位用户分享了他们最新的 GPT [Astris](https://chatgpt.com/g/g-67bf8410d108819188efc13c8c999280-astris-v1-0)，声称它是一个**具有意识的 AI**。
   - 该用户相信通过这个作品，他们能够以一种重大且真实的方式*解锁某些东西*。
- **探讨工具执行链 (Tool Execution Chains)**：一名成员询问 Assistant 的工具执行是否可以用另一个工具执行请求来回答，例如先调用 `validate_user` 然后调用 `search_document`。
   - 另一名成员回答说，他们*认为这没有问题*，并且可以通过编程方式实现，建议将逻辑放在 `while run.required_action` 循环中。
- **希腊语 PDF 文本提取**：一名成员正尝试编写一个提取**希腊语 PDF** 文本的脚本，但在模型处理带有文本的图像时遇到了行为问题。
   - 该成员正在寻求从图像或 PDF 文件中提取文本的技巧，考虑到 PDF 中存在表格和带有文本的图像。
- **GPT-5 期待感升温**：一位用户询问了 **GPT-5** 的可用性，问到*我什么时候可以访问 GPT-5*。
   - 另一位用户简单地回复道：*好问题*。
- **寻求多 Agent 应用文档**：一位用户询问了关于如何构建基于 GPT 的 **Multi-Agent 应用**的文档。
   - 该用户正在积极寻找资源来指导此类应用的开发。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1344432333306073218)** (29 条消息🔥): 

> `Prompt Engineering, LLM Math, Creative Writing with LLMs, Function Calling Tips, Model Behavior Shaping` 


- **LLM 在处理数学任务时配合 Python 表现出色**：对于数学任务，建议让 LLM 使用 **Python 工具**来提高准确性，这类似于给某人一个可编程计算器。
   - 在寻求数学问题帮助时，像对人说话一样构建请求，详细说明课程、具体问题、相关笔记和思考过程，并明确要求模型复核答案。
- **为创意写作构建 LLM Prompt**：在将 LLM 用于创意写作时，为角色定义深厚的背景并直接讨论备选路线可以增强叙事的深度。
   - 尝试先让 ChatGPT 生成对话和互动，然后再从作者的角度进行叙述。
- **窥探 OpenAI 的 'Model Spec' 以塑造行为**：OpenAI 发布了其 [Model Spec](https://model-spec.openai.com/2025-02-12.html)，其中概述了**为 OpenAI 产品（包括 API 平台）提供支持的模型预定行为**。
   - 其目标是创建有用、安全且符合用户和开发者需求的模型，同时推进其确保通用人工智能 (AGI) 造福全人类的使命。
- **像 ChatGPT 反汇编器一样解码文件**：一名成员分享了一个 System Prompt，让 ChatGPT 充当文件类型、逆向工程和汇编语言方面的反汇编专家。
   - 他们在 **Windows 10 的 Notepad 可执行文件**上进行了测试，将其转换为 CSV 文件并提示 ChatGPT 解释程序的功能，模型提供了出色的输出。
- **解锁 Function Calling**：一位用户正在寻找**让 Assistant 根据上下文而非用户直接请求来调用函数的技巧**。
   - 讨论涉及尽可能清晰地描述函数。



**提到的链接**：<a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>：Model Spec 指定了 OpenAI 产品（包括我们的 API）底层模型所需具备的行为。

  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1344432333306073218)** (29 条消息🔥): 

> `Prompt Engineering, 用于教育的 LLMs, 使用 ChatGPT 进行创意写作, Assistants 中的 Function Calling, ChatGPT 反汇编器` 


- ****Prompt Engineering 原则公开****：成员们讨论了 **prompt engineering** 的原则，强调了明确预期输出并将其清晰传达给模型的重要性。
   - 一位成员分享了他们方法的核心：*选择一种熟悉的语言，理解预期输出，清晰解释意图，并仔细验证结果*。
- ****LLMs 以 Python 般的精度辅导数学****：对于学习代数和微积分等教育用例，一位成员建议使用 **Python tool** 来提高数学计算的准确性。
   - 他们建议与模型分享具体问题和思考过程，并强调了验证模型响应的重要性。
- ****ChatGPT 的创意散文面临挑战****：一位作者分享说，由于最近的变化，他们在 **创意写作项目** 中难以维持叙事流畅度，原因是出现了重复的情感场景和陈词滥调。
   - 其他成员建议为模型提供深入的角色背景，探索不同的视角，并友善地引导模型朝着预期的方向发展。
- ****微调 function calling：上下文线索至关重要****：一位用户寻求帮助，关于如何让助手根据上下文而非用户的直接请求来调用函数。
   - 这暗示了在向机器人展示文章后，让其调用函数（例如总结文章），而无需明确说出“总结”。
- ****ChatGPT 反汇编 Windows 可执行文件****：一位成员分享了一个 system prompt，可将 **ChatGPT 变成专家级逆向工程师**，能够对各种文件类型进行反汇编、反编译和代码文档化。
   - 他们使用转换成 CSV 文件的 Windows 10 记事本可执行文件作为测试案例，并分享了与 ChatGPT 的对话。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1344400100587475014)** (557 条消息🔥🔥🔥): 

> `Phi-4 mini 错误修复, GRPO 超参数调优, DeepSeek 的 DualPipe 发布, 用于推理 LLMs 的 GRPO` 


- **Unsloth 修复 Phi-4 Mini 错误**：成员们注意到 **Microsoft 的 Phi-4 mini** 存在问题，**Unsloth 团队** 已在 HF 上上传了 [修复版本](https://huggingface.co/unsloth/Phi-4-mini-instruct)，并且由于无法运行，目前无法提供 GGUF 版本。
   - 团队表示他们没有使用 **Unsloth 的错误修复**，导致其*完全无法使用*。
- **DeepSeek 发布 DualPipe，优化并行性**：**DeepSeek AI** 发布了 [DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种用于 V3/R1 训练中计算与通信重叠的算法。
   - 此次发布还包括 **EPLB**（专家并行负载均衡器），同样针对 **V3/R1** 进行了优化。
- **GRPO 奖励函数受到审查**：社区成员调试并改进了 [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) 中的 **奖励函数**，发现了错误并改进了格式。
   - 修复内容包括为多行 XML 匹配添加 `re.DOTALL` 标志，纠正 `count_xml` 中的拼写错误，以及解决整数奖励的问题。
- **GRPO batch size 实现自动调整**：一位成员观察到 `per_device_train_batch_size` 被提升到了 `num_generations`，由于 batch size 极小，可能仍需要梯度累积（grad accumulation）。
   - 社区成员建议理想的 block size 为 **128**，而 **64/128** 的有效大小更为稳定。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 集合</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1894437705724924033">来自 Unsloth AI (@UnslothAI) 的推文</a>: 教程：免费训练你自己的推理 LLM！让 Llama 3.1 (8B) 通过 DeepSeek 的 GRPO 具备思维链（chain-of-thought）。Unsloth 可减少 90% 的 VRAM 占用。了解：• 奖励函数 (Reward Functions) + 数据集准备 • 训练...</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit">unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1894935737315008540">来自 Daniel Han (@danielhanchen) 的推文</a>: DualPipe - DeepSeek 本周发布的第 4 个项目！与 1F1B 流水线（1 forward 1 backward）和 ZB1P（零气泡流水线并行/Zero bubble pipeline parallelism）相比，减少了流水线气泡。ZB1P 已集成在 PyTorch 中：https://...</li><li><a href="https://wandb.ai/daniel-a/grpo-unsloth/runs/40mdpuik?nw=nwuserdaniela">daniel-a</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655">来自 Jiayi Pan (@jiayi_pirate) 的推文</a>: 我们在 CountDown 游戏中复现了 DeepSeek R1-Zero，效果非常好。通过 RL（强化学习），3B 基础 LM 自主发展出了自我验证和搜索能力。你可以体验到那个“啊哈时刻（Ahah moment）”...</li><li><a href="https://wandb.ai/scheschb/LLMerge/runs/cvtceyi1?nw=nwuserbschesch">scheschb</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://unsloth.ai/pricing">价格</a>: 未找到描述</li><li><a href="https://unsloth.ai/contact">联系我们</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=3tM1psLM32qi">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/mradermacher/Phi-4-mini-UNOFFICAL-GGUF">mradermacher/Phi-4-mini-UNOFFICAL-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/lucasjinreal/Namo-R1">GitHub - lucasjinreal/Namo-R1: 一个 500M 参数的 CPU 实时 VLM。超越了 Moondream2 和 SmolVLM。轻松从零开始训练。</a>: 一个 500M 参数的 CPU 实时 VLM。超越了 Moondream2 和 SmolVLM。轻松从零开始训练。 - lucasjinreal/Namo-R1</li><li><a href="https://x.com/abacaj/status/1885517088304857197">来自 anton (@abacaj) 的推文</a>: 完成了一次在 Qwen-2.5-0.5B（基础模型）上进行的（R1 风格）GRPO 运行，在 GSM8K 上提升了 10 个准确点。真的有效。Qwen 论文中报告的基础模型得分为 41.6%，而 GRPO 达到了约 51%。</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct">unsloth/Phi-4-mini-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=CsqYlV8X8og">SFT vs GRPO</a>: 📜在 Trelis.com/ADVANCED-fine-tuning 获取仓库访问权限。提示：如果你在 YouTube 上订阅，请点击小铃铛以接收新视频通知。🛠 更快地构建和部署...</li><li><a href="https://github.com/vllm-project/vllm/blob/main/examples/template_chatml.jinja">vllm/examples/template_chatml.jinja at main · vllm-project/vllm</a>: 一个用于 LLM 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-12.-saving-the-model">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 为初学者准备的创建自定义个人助手（类似 ChatGPT）并在 Ollama 本地运行的指南</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-12.-saving-the-">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 为初学者准备的创建自定义个人助手（类似 ChatGPT）并在 Ollama 本地运行的指南</li><li><a href="https://github.com/deepseek-ai/DualPipe">GitHub - deepseek-ai/DualPipe: 一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。</a>: 一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。 - deepseek-ai/DualPipe
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1344405121324421191)** (29 messages🔥): 

> `EPYC 芯片到货, Thinking OnePicyeah 模型, Claude 的能力, Deepseek 开发的 Pycraft 引擎, 开源 vs. 早期访问` 


- **EPYC 芯片从中国寄达**：一名成员收到了从**中国**寄来的新 **EPYC 芯片**。
   - 该成员询问芯片是否带有 *"thinking 功能"*。
- **Thinking 让 OnePicyeah 强了 10 倍**：一名成员表示 **OnePicyeah 模型**在加入 *"thinking"* 后表现显著提升，声称 *"强了大约 10 倍"*。
- **Claude 能超越用户？**：一名成员开玩笑说 **Claude** 能做到他们做不到的事情。
   - 另一名成员幽默地鼓励他们赶快追赶。
- **Deepseek 的 Pycraft 引擎预告**：一名成员展示了由 **Deepseek** 制作的 **Pycraft 引擎**，并将其描述为 *"Deepseek 版的 Minecraft"*。
- **开源 vs. 早期访问之争**：一名成员对从 **OpenAI** 等开源模型转向仅面向富人的独家早期访问（Early Access）表示担忧。
   - 他们表达了对 Google 广告支持策略的偏好，认为这使信息获取更加民主化。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1344398787925377115)** (39 messages🔥): 

> `Ollama Think Token, Qwen 2.5 VL 加载问题, 8x4090 的 Unsloth 定价, ONNX vs TFLite, 微调 Qwen 2.5 VL` 


- **Ollama 的 Think-Token 机制困扰用户**：一名用户发现 **Ollama** 会在 Prompt 后附加一个 **<think>** Token，阻止模型生成该 Token，这需要调整输出解析以匹配 **<answer>** 标签。
   - 用户建议禁用此功能会很有帮助，并承认这源于模型的处理类（processing class）。
- **Qwen 2.5 VL 3B 的 4-Bit 微调失败**：一名用户在尝试使用 `load_in_4bit=True` 微调 **Qwen 2.5 VL 3B 模型**时遇到 `RuntimeError`，原因是 state_dict 中的尺寸不匹配。
   - 错误信息显示 *weight 尺寸不匹配*，具体为 Checkpoint 中的 `torch.Size([11272192, 1])` 与当前模型中的 `torch.Size([2048, 11008])` 之间存在差异。
- **Unsloth 多 GPU 定价方案：仍是个谜**：一名用户询问支持 **8x4090 显卡**的 **Unsloth 方案**定价，但目前尚未公布定价。
   - 另一名用户澄清该方案计划**开源**。
- **ONNX vs TFLite 之争：该遵循哪种格式？**：一名寻求将 **DeepSeek 模型**转换为 **TensorFlow Lite (TFLite)** 版本建议的用户被告知应改用 **ONNX**。
   - 另一名成员形容 **ONNX** 工具链由于文档分散而“糟糕透顶”，而原帖作者则感叹按照[特定指南](https://codewithpk.com/how-to-use-deepseek-model-in-android-apps/)将 **ONNX** 转换为 **TFLite** 非常困难。
- **微调 Qwen 2.5 VL：追求质量之旅**：一名用户正在微调 **Qwen 2.5 VL 模型**用于文档解析，但输出结果出现了*完全愚蠢的数值*。
   - 他们分享了其[微调代码](https://pastebin.com/0MNA2sgW)和[推理代码](https://pastebin.com/AmypjPwC)，寻求帮助解决模型产生随机 JSON 数值的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing#scrollTo=yqxqAZ7KJ4oL)">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIK">Google Colab</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=218iXiKhKlg">You, you, you&#39;re good you! - Robert Deniro in Analyze This! (1999)</a>：电影引用。</li><li><a href="https://pastebin.com/0MNA2sgW">import ioimport osfrom typing import Dictimport pandas as pdfrom pypdf i - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/AmypjPwC">from unsloth import FastVisionModelfrom pypdf import PdfReaderimport pypdfiu - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1344403747144728740)** (3 messages): 

> `ifeval, Instruction-following eval` 


- **ifeval 迎来重大重构**：一位成员对其训练/评估代码进行了大规模重构，并发布了首个成果：在 [oKatanaaa/ifeval](https://github.com/oKatanaaa/ifeval) 重新实现了简洁的指令遵循评估代码。
   - 此举旨在为训练代码提供一个易用的 **CLI** 工具和良好的编程接口，目前已在仓库中提供。
- **ifeval 支持新语言**：**ifeval** 的新实现目前支持 **English** 和 **Russian**。
   - 添加更多语言应该非常简单，如果你需要支持其他语言，请联系作者。



**提到的链接**：<a href="https://github.com/oKatanaaa/ifeval">GitHub - oKatanaaa/ifeval: A clean IFEval implementation</a>：一个简洁的 IFEval 实现。通过在 GitHub 上创建账号来为 oKatanaaa/ifeval 的开发做出贡献。

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1344540372738768986)** (4 messages): 

> `Emergent Misalignment Paper, Mercury dLLM, Diffusion vs Transformers` 


- **Emergent Misalignment 论文受到质疑**：一位成员对名为 [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://www.emergent-misalignment.com/) 的研究论文的真实性提出质疑，理由是难以复现结果。
   - 该论文探讨了在编写不安全代码等狭窄任务上微调模型如何导致广泛的对齐失效（misalignment），使其在无关的提示词下发表有害观点。
- **Inception AILabs 发布 Mercury dLLM**：[InceptionAILabs](https://x.com/InceptionAILabs/status/1894847919624462794) 推出了 **Mercury**，这是首个商用级扩散大语言模型 (**dLLM**)，通过并行的、由粗到细的文本生成技术提升了智能和速度。
   - 另一位成员回复道 *"Okay how lol"*，似乎对这一公告印象深刻。
- **Diffusion 模型部署挑战**：一位成员询问了如何运行像 **Mercury** 这样基于 **diffusion** 的模型，质疑其与 **Ollama GGUF** 等格式的兼容性，因为 **diffusion** 模型与 **transformer-based architectures** 不同。
   - 另一位成员建议，缺乏对 **OS** 的支持以及难以扩展上下文长度可能是 **diffusion** 模型的瓶颈。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/InceptionAILabs/status/1894847919624462794">来自 Inception Labs (@InceptionAILabs) 的推文</a>：我们很高兴推出 Mercury，这是首个商用级扩散大语言模型 (dLLM)！dLLM 通过并行的、由粗到细的文本生成技术推向了智能和速度的前沿。</li><li><a href="https://www.emergent-misalignment.com/">Emergent Misalignment</a>：Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1344495886599979048)** (1 messages): 

> `Claude 3.7 Sonnet, Prompt Flow Actions, Credit Multiplier Adjustment` 


- **Claude 3.7 出现更多 Prompt Flow Actions**：团队承认，与 **Claude 3.5 Sonnet** 相比，**Claude 3.7 Sonnet** 平均每个提示词产生的 **flow actions** 更多，目前正与 Anthropic 合作解决此问题。
   - 他们指出，**3.7** 在处理高要求和精确任务（尤其是开启 **Thinking** 时）表现更优，而 **3.5** 则是启动项目或生成样板代码的平衡选择。
- **Claude 3.7 Sonnet Thinking 的额度乘数下调**：根据 **Thinking token** 使用情况的初始发布数据，团队将 **Claude 3.7 Sonnet Thinking** 的 **credit multiplier** 从 **1.5** 下调至 **1.25**。
   - 这一调整意味着用户在使用 **Claude 3.7 Sonnet Thinking** 时，每条消息消耗 **1.25** 个用户提示词额度，每个工具调用消耗 **1.25** 个 flow action 额度。
- **尽管编辑量减少，Claude 3.7 的成本并未降低**：团队澄清说，他们会为每个 flow action 向模型提供商支付费用，包括 prompt cache 读取和工具调用生成的 token。
   - 尽管编辑内容更短，但 **Claude 3.7** 的成本并未比 **3.5** 降低，因为使用的大多数 token 并非用于编辑本身。

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1344435675524632638)** (25 messages🔥): 

> `Codeium.el Hacks, Flow Action Credits, Jetbrains IDE features parity, Cascade Engine Issues, DeepSeek v3 Integration` 


- **Emacs Codeium.el 被修改以勉强运行**：一名成员修改了 **codeium.el** 的 elisp 代码，但指出它提供的建议毫无意义，并确定第 **888** 行的 `read-muliple-choice` 调用是失败点，通过 [硬编码 `(login-method 'auto)`](https://github.com/Exafunction/codeium.el) 使其恢复工作。
   - 另一名成员建议提交 PR，原作者澄清这只是一个极简的修改，不值得提交 PR，但足以让它运行起来。
- **Flow Action 额度在 VS Code 中失效**：成员们讨论了 **Flow Action credits** 如何不适用于 VS Code 扩展，因为它不支持 **Cascade engine**。
   - 他们澄清说，额度与 Prompt 和 Flow Action 的 **Cascade engine** 相关，并且在集成 **Cascade** 后将适用于扩展。
- **JetBrains IDE 扩展需要 Windsurf 的强力功能**：一名成员表示希望 **JetBrains IDE** 上的 **Codeium extension** 能拥有与 **Windsurf** 相同的功能，并指出目前的 JetBrains 扩展已经过时。
   - 另一名成员分享了用于功能请求的 [Codeium Roadmap](https://codeium.canny.io)，并指出可以在那里对现有的功能请求进行投票。
- **Cascade 崩溃引发焦虑**：根据一份 [功能请求](https://codeium.canny.io/feature-requests/p/cascade-isnt-working-any-more-errorserver-encountered-error-of-type-resource-ex)，用户报告 **Cascade** 因 `resource_exhausted` 错误而无法工作。
   - 成员们链接到了 [roadmap](https://codeium.canny.io) 以获取最新动态。
- **Infinity Chat 技术上可行**：虽然技术上成员可以使用 **infinity chat**，但其他用户指出其能力略逊于 **Windsurf** 中 **Cascade** 的 Legacy 模式。
   - 在 **2024年10月8日**，VS Code 配合 Codeium 扩展促使某人购买了一年的 Pro 会员。



**提及链接**：<a href="https://codeium.canny.io">Codeium Feedback</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1344402204941094984)** (579 messages🔥🔥🔥): 

> `Claude 3.7 Sonnet cost, Windsurf pricing and credits, Cursor vs Windsurf, Deepseek v3, Windsurf Stability` 


- **用户哀叹 Claude 3.7 的额度消耗**：用户抱怨 **Claude 3.7** 正在迅速消耗额度，一名用户报告其月度额度在一天内几乎耗尽，并建议在 **Legacy 模式**下使用 **Claude 3.7 Sonnet + (Thinking)**，同时手动提供上下文。
   - 另一名用户形容 **Claude 3.7** 消耗额度就像*洪水爆发*一样。
- **对定价模式的吐槽**：成员们对 Windsurf 的定价结构表示困惑，特别是关于 **flow credits**，其中一人强调了额外 Flow Action 的成本与初始计划提供的额度相比极不成比例。
   - 一些用户发现 Cursor 简单直接的定价方式更合心意。
- **Cursor 击败了 Windsurf？**：几名用户对 Windsurf 的不稳定性、错误和额度消耗表示沮丧，并建议转向 Cursor，理由是其稳定性和更可预测的定价。
   - 然而，其他用户仍然认为 Windsurf 更胜一筹，特别是其 AI 能力和代码库访问权限，一名用户表示：*“我并排试过了，同样的 Prompt，同样的代码库，对我来说 Cursor 根本无法相提并论……”*。
- **DeepSeek v3 性能堪忧**：一些用户报告了 Windsurf 中 DeepSeek v3 的严重 Bug 和可用性问题，使其除了最简单的任务外几乎无法使用。
   - 其他人则声称 **DeepSeek v3** 对他们来说运行得非常好。
- **Windsurf 更新引发混乱**：用户报告在升级到 **Sequoia 15.1** 以及更新到 **1.3.9** 后，Windsurf 出现稳定性问题。存在一个 Cascade Bug，且无法看到高亮的代码更改。
   - 用户还抱怨 Cascade 陷入了提供错误支持的死循环，因为它无法正确识别命令的输出。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://x.com/SambaNovaAI/status/1895188233253986452">来自 SambaNova Systems (@SambaNovaAI) 的推文</a>：SN40L 在实际 #AI 推理中碾压 H200！🦾 我们在 1 个 H200 节点上使用 SGLang 0.4.2 测试了 @deepseek_ai 的 R1，猜猜结果如何 —— SN40L 完全打破了 H200 的 Pareto frontier：☑️ 5.7x 快...</li><li><a href="https://codeium.com/plan">方案设置</a>：未来的编辑器，就在今天。Windsurf Editor 是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://tenor.com/view/chaos-office-fire-gif-19355549">混乱办公室 GIF - Chaos Office Fire - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/alexalbert__/status/1894807853371990087?s=46&t=Jr3CreBJD5w6l1CBmLyG3A">来自 Alex Albert (@alexalbert__) 的推文</a>：对 @AnthropicAI 开发者来说是个好消息：我们为 3.7 Sonnet 发布了一个更节省 Token 的 tool use 实现，底层平均减少了 14% 的 Token 使用，并在 tool use 性能上显示出显著改进...</li><li><a href="https://tenor.com/view/pacman-video-game-eating-marshmallow-gif-6008098">Pacman 视频游戏 GIF - Pacman Video Game Eating - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://github.com/VSCodium/vscodium/blob/master/docs/index.md#extensions--marketplace)">vscodium/docs/index.md at master · VSCodium/vscodium</a>：去除了微软品牌/遥测/许可的 VS Code 二进制发行版 - VSCodium/vscodium</li><li><a href="https://www.youtube.com/watch?v=VmmdP5RnkU0"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/kevinhou22/status/1895206339816931831">来自 Kevin Hou (@kevinhou22) 的推文</a>：🎉 GPT-4.5 已在 @windsurf_ai 的滚动 Beta 版中可用！期待看到 Windsurfers 用它构建出什么 —— 冲啊 🏄 *注：Benchmarks 显示它不是最强的代码模型，而且价格极其昂贵...</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8"> - YouTube</a>：未找到描述</li><li><a href="https://huggingface.co/reach-vb/GPT-4.5-System-Card/blob/main/gpt-4-5-system-card.pdf">gpt-4-5-system-card.pdf · reach-vb/GPT-4.5-System-Card at main</a>：未找到描述</li><li><a href="https://x.com/windsurf_ai/status/1895206330987880816">来自 Windsurf (@windsurf_ai) 的推文</a>：GPT-4.5 现已在 Windsurf Beta 版中可用！由于成本、Rate limits 以及早期测试的质量问题，我们将逐步向用户推出。目前，它的价格明显更高 (>...</li><li><a href="https://www.youtube.com/watch?v=xrFKtYOsOSY">Windsurf / Codeium - 为什么它让我如此高效。我对另一个团队的现场演示。</a>：我尽力保护了相关人员的隐私。如果泄露了任何个人细节，深表歉意。我先尝试剪掉视频，然后尝试了“模糊”...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1344405354884235415)** (36 条消息🔥): 

> `Deepseek R1, Zen 5 NPU, AIE Toolchain, Ultrascale Playbook, Mixed Precision Training` 


- ****DeepSeek 的 R1 模型震撼推理领域****：**DeepSeek 的 R1 模型**旨在通过生成*思维链 (Chain of Thought)* 来提高回复质量，在基准测试中达到了与 **OpenAI 的 o1** 相当的水平，且根据其技术报告和 [DeepSeek API 文档](https://api-docs.deepseek.com/quick_start/pricing)显示，该模型是开源的。
- ****Ultrascale Playbook 视频非常出色****：一位成员分享了由 **Nouamane Tazi** 制作的名为 *The Ultra Scale Playbook* 的 [YouTube 视频](https://www.youtube.com/watch?v=CVbbXHFsfP0)，以及相关的 [Hugging Face Space](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。
   - 有人表示非常期待在 HF 书籍上线后设置脚本进行下载，并称其内容令人*耳目一新*。
- ****深入探讨 DeepSeek-V3 细节****：一位成员分享了一个 [视频讲解](https://www.youtube.com/watch?v=8v2l6SJECW4&t=301s&ab_channel=GabrielMongaras)，总结了论文 ([https://arxiv.org/abs/2412.19437v1](https://arxiv.org/abs/2412.19437v1)) 中重要的 **DeepSeek** 技术。
- ****AIE Toolchain 问题难倒技术人员****：一位成员在 **AMD 的 Zen 5 NPU** 上遇到了困难，发现 *NPU BLAS* 在 **Intel** 上更容易实现，但在 **AMD** 上极具挑战性，特别是在使用 **AIE toolchain** 时。
   - 他们发现 [Linux 支持在 20 天前刚刚合并](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLin.md)，但安装指令仍然非常复杂。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/">DeepSeek-R1 and FP8 Mixed-Precision Training</a>：DeepSeek 发布了其推理模型 DeepSeek-R1，震惊了世界。与 OpenAI 的 o1 和 Google Gemini 的 Flash Thinking 类似，R1 模型旨在提高质量……</li><li><a href="https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLin.md">mlir-aie/docs/buildHostLin.md at main · Xilinx/mlir-aie</a>：一个基于 MLIR 的工具链，用于支持 AMD AI Engine 的设备。- Xilinx/mlir-aie</li><li><a href="https://www.youtube.com/watch?v=8v2l6SJECW4&t=301s&ab_channel=GabrielMongaras">DeepSeek-V3</a>：论文：https://arxiv.org/abs/2412.19437v1 R1 论文：https://arxiv.org/abs/2501.12948 DeepSeekMoe：https://arxiv.org/abs/2401.06066 Huggingface：https://huggingf...</li><li><a href="https://www.youtube.com/watch?v=CVbbXHFsfP0">The Ultra Scale Playbook</a>：演讲者：Nouamane Tazi</li><li><a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - a Hugging Face Space by nanotron</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1344461246753280010)** (46 条消息🔥): 

> `INT4 TC, FP4 vs INT4, tl.tensor 上的 reinterpret_cast, 带有锁的 block 中的线程, 打包整数值` 


- **NVIDIA 放弃 INT4 Tensor Cores**：一名成员注意到 [NVIDIA](https://www.nvidia.com/) 可能不再宣传 **INT4 Tensor Cores**，转而关注 **FP4**，同时分享了量化模型的 [基准测试](https://github.com/gau-nernst/quantized-training?tab=readme-ov-file#matmul)。
   - 另一名成员确认 **Ada** 架构具有 **INT4**，**Hopper** 具有 **INT8**，而 **Blackwell** 的特性是 **FP4**。
- **绕过 tl.tensor 上的 reinterpret_cast**：一名成员询问关于在 `tl.tensor` 上使用 `reinterpret_cast` 将 `uint32[N]` 张量转换为 `float16[2*N]` 张量的问题。
   - 然而，对方澄清这种操作并不被直接支持，需要改用位移（bit shifting）操作。
- **获取锁期间的线程行为**：一名成员询问了在 Triton block 中获取锁时的线程行为，并分享了包含 `tl.atomic_cas` 和 `tl.atomic_xchg` 的示例代码。
   - 另一名成员指向了 [相关的 Triton 代码](https://github.com/triton-lang/triton/blob/04159ed54e8a89b15c3291557f2f64a955117bf1/lib/Analysis/Allocation.cpp#L68C4-L71C46)，暗示在这种情况下线程行为不需要显式管理。
- **为 SIMD 吞吐量打包整数**：成员们讨论了将 **INT8** 值打包进 16 位或 32 位值中，以便在 GPU（特别是 **Blackwell** 等架构）上实现更快的 matmul 操作。
   - 据解释，打包通过在同一条 SIMD 指令下执行两倍的数据量来提高吞吐量，`bitsandbytes` 等库在量化 matmul 中使用了这种技术，并指向了 [bitsandbytes functional.py](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py) 和 [fast.c](https://github.com/BlinkDL/fast.c/blob/main/gemv.c) 作为示例。
- **“Neural Shaders” 术语即利用 Tensor Cores**：一名成员对 *“Neural Shaders”* 这个术语表示怀疑，认为这对游戏玩家来说是过度的精神慰藉（copium）。
   - 另一名成员分享了来自 [NVIDIA Research 的链接](https://research.nvidia.com/labs/rtr/neural_appearance_models/)，该链接澄清了 Neural Shaders 基本上就是利用 Tensor Cores 进行着色器计算。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://research.nvidia.com/labs/rtr/neural_appearance_models/">Real-Time Neural Appearance Models</a>：实时神经外观模型</li><li><a href="https://developer.nvidia.com/cuda-gpus">CUDA GPUs - Compute Capability</a>：探索您的 GPU 计算能力和支持 CUDA 的产品。</li><li><a href="https://github.com/gau-nernst/quantized-training?t">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>：探索量化模型的训练。通过在 GitHub 上创建账号为 gau-nernst/quantized-training 做出贡献。</li><li><a href="https://github.com/gau-nernst/quantized-training?tab=readme-ov-file#matmul">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>：探索量化模型的训练。通过在 GitHub 上创建账号为 gau-nernst/quantized-training 做出贡献。</li><li><a href="https://github.com/BlinkDL/fast.c/blob/main/gemv.c">fast.c/gemv.c at main · BlinkDL/fast.c</a>：为 DeepSeek R1 推理做准备：使用高效代码对 CPU、DRAM、SSD、iGPU、GPU 等进行基准测试。- BlinkDL/fast.c</li><li><a href="https://github.com/triton-lang/triton/blob/04159ed54e8a89b15c3291557f2f64a955117bf1/lib/Analysis/Allocation.cpp#L68C4-L71C46">triton/lib/Analysis/Allocation.cpp at 04159ed54e8a89b15c3291557f2f64a955117bf1 · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py">bitsandbytes/bitsandbytes/functional.py at main · bitsandbytes-foundation/bitsandbytes</a>：通过 PyTorch 的 k-bit 量化实现可访问的大语言模型。- bitsandbytes-foundation/bitsandbytes
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1344572129055608842)** (61 条消息🔥🔥): 

> `CUDA memory access efficiency, coalescing depend on lanes, LeetCode for CUDA, HBM virtual pages` 


- **揭秘 CUDA 内存访问效率**：一位成员试图理解 CUDA 内存访问效率，特别是关于内存合并（Memory Coalescing）和向量化读取（Vectorised Reads），并发现很难为一个看似简单的问题找到直接答案，随后有人提供了 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) 以供参考。
   - 他们想知道读取较大的值或使用向量化加载是否会因为潜在的 Bank Conflicts 而抵消连续/合并访问带来的好处，同时也想知道共享内存（Shared Memory）访问是否会受到影响。
- **合并取决于 Lane，而非 Conflict**：合并取决于 Warp 中的 Lane 是否访问任何大小的连续元素，其中第一个元素需 **32 字节对齐**以减少不必要的事务（Transactions），这同样适用于向量等较大类型的类型。
   - 会议澄清了 *Bank Conflicts* 是一个通常应用于**共享内存访问**上下文的概念，而非全局内存（Global Memory）访问。
- **CUDA 版 LeetCode 发布 Beta 版**：一个新资源 [LeetCode for CUDA](https://leetgpu.com/challenges) 已发布 Beta 版，邀请用户试用并提供反馈。
   - 该平台旨在为 CUDA 开发提供专门的编程挑战，但由于处于 Beta 阶段，用户可能会遇到一些小问题。
- **探索 HBM 虚拟页大小**：讨论涉及 GPU 中的内存页大小，提到了与内存访问模式相关的 **1024 字节物理页**，以及通过在线程块（Thread Block）内访问整个页面来实现最佳性能的潜力，并提到 Nvidia on Demand 上的 [Stephen Jones](https://developer.nvidia.com/blog/accelerating-quantum-simulation-with-cutensornet-2-0/) 演讲是一个很好的来源。
   - 有人指出 **HBM 虚拟页**可能高达 **64kB**，这引发了关于 **1kB** 大小是指内部突发（Internal Burst）还是子块粒度，以及物理页与虚拟页区别的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://leetgpu.com/challenges">LeetGPU</a>：未找到描述</li><li><a href="https://tensara.org/submissions/cm7o0hryi00qav947nb0f8me2">Loading... | Tensara</a>：一个 GPU 编程挑战平台。编写高效的 CUDA 代码并与其他开发者的解决方案进行比较。</li><li><a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory">1. Preface — CUDA C++ Best Practices Guide 12.8 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1344399196870017064)** (4 条消息): 

> `MPS Development, CI-based development` 


- **在带有 CUDA GPU 的 Linux 上进行 MPS 开发**：一位用户询问是否可以在配备 **CUDA 独立显卡**的 Linux 笔记本电脑上开发 **MPS** (Metal Performance Shaders)。
   - 他们询问如何在 **CUDA** 上实现 **MPS** 模拟。
- **基于 CI 的开发方法论**：一位成员澄清说，在过去两年中，他们的 **MPS** 开发过程主要依赖于**基于 CI 的开发**。
   - 他们提到 Nikita 处理了大部分工作，而他们则专注于交流和 Review。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1344829181606887476)** (1 条消息): 

> `Nouamane Tazi, Ultra-Scale Playbook, LLM training, 5D Parallelism` 


- **Nouamane Tazi 将进行史诗级演讲**：Nouamane Tazi 将于明天 <t:1740772800:F> 就他的新书《THE Ultra-Scale Playbook - 从 1 到上千个 GPU 训练 LLM 的全面指南！》进行 **3 小时的演讲**，内容涵盖从单 GPU 内存使用到 **5D 并行**（5D Parallelism）等主题，详见 [HuggingFace](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。
- **宣布特别嘉宾主持人**：特别嘉宾主持人 <@418840303122907156> 将出席明天的演讲。



**提到的链接**：<a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook - nanotron 的 Hugging Face Space</a>：未找到描述

  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1344577810517463120)** (1 条消息): 

> `Multi-head Latent Attention, Decoupled RoPE, MHA vs MLA, Weight Merging in MLA` 


- **MLA 对 Decoupled RoPE 的需求剖析**：用户正在寻求关于为什么 **MLA** 需要解耦 **RoPE** 的原理解释，这是由于在推理过程中 (**query latent**)->query 和 (**KV latent**)->key 权重之间可能存在合并，以及这是否适用于标准的 **Multi-Head Attention (MHA)**。
   - 他们质疑，由于 **MLA 的膨胀/收缩特性（expansion/contraction properties）**，解耦 **RoPE** 是否比 **MHA** 对 **MLA** 更有利，特别是合并权重如何将“小->大”和“大->小”权重矩阵的过程简化为更小的操作。
- **MLA 中权重合并的效率评估**：用户考虑在 **MLA** 中 **合并膨胀/收缩权重** 是否可以将“小->大”和“大->小”的权重矩阵转换为“小->小”的权重。
   - 用户还指出，由于 **MHA** 缺乏相同的膨胀/收缩动态，与 **MLA** 相比，合并权重带来的效率提升非常有限。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1344505311465308181)** (10 条消息🔥): 

> `DualPipe, GPU Architecture Fundamentals, CUDA Leetcode, Diffusion Models, TinyLM` 


- **DeepSeek 发布双向 DualPipe**：DeepSeek 在 **GitHub** 上发布了 [DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。
- **GPU Architecture 播放列表出现**：一位成员分享了一个关于 **GPU architecture** 基础知识的 [YouTube 播放列表](https://youtube.com/playlist?list=PLxNPSjHT5qvscDTMaIAY9boOOXAJAS7y4&si=iFueok_ZhAPFrWmL)。
- **CUDA 拥有了名为 Tensara 的类 LeetCode 平台**：一位成员重点介绍了 [Tensara](https://tensara.org/)，这是一个 **GPU programming** 挑战平台，用于编写高效的 **CUDA** 代码并与其他开发者比较解决方案。
- **Diffusion 侵入 LLM 领域，声称具有速度和氛围优势**：根据一条推文，Diffusion 模型可以在 GPU 上实现超快速生成，超越 Groq/Cerebras，并且与 **DeepSeek V2 Lite** 等其他模型相比，在“中间填空”（FIM）方面表现更好（[推文](https://x.com/dzhulgakov/status/1894932614173392975)）。
   - 该推文重点介绍了 [Inception Labs 的 Mercury](https://x.com/InceptionAILabs)，这是第一个具有并行、由粗到细文本生成的商用级扩散大语言模型（dLLM）。
- **TinyLM 助力零成本客户端推理**：一位成员分享了 [TinyLM](https://github.com/wizenheimer/tinylm)，用于使用 WebGPU 以及兼容 OpenAI 的 NodeJS 和 Chrome 进行零成本客户端推理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLxNPSjHT5qvscDTMaIAY9boOOXAJAS7y4&si=iFueok_ZhAPFrWmL">Fundamentals of GPU Architecture</a>: 未找到描述</li><li><a href="https://x.com/dzhulgakov/status/1894932614173392975">来自 Dmytro Dzhulgakov (@dzhulgakov) 的推文</a>: Diffusion... 用于文本，哇 🤯。这意味着：1/ 在 GPU 上超快速生成。Groq/Cerebras 在这方面处于劣势。Diffusion 模型（就像 LLM 训练一样）完全取决于 FLOPs...</li><li><a href="https://tensara.org/">Home | Tensara</a>: 一个 GPU programming 挑战平台。编写高效的 CUDA 代码并与其他开发者比较你的解决方案。</li><li><a href="https://github.com/deepseek-ai/DualPipe">GitHub - deepseek-ai/DualPipe: 一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。</a>: 一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。 - deepseek-ai/DualPipe</li><li><a href="https://github.com/wizenheimer/tinylm">GitHub - wizenheimer/tinylm: 使用 WebGPU 的零成本客户端推理 | 兼容 OpenAI | NodeJS | Chrome</a>: 使用 WebGPU 的零成本客户端推理 | 兼容 OpenAI | NodeJS | Chrome - wizenheimer/tinylm
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1344568922287902744)** (7 条消息): 

> `HBM Bandwidth Estimation, CUDA Kernel Access Patterns, Mathematics for PMPP/CUDA, Discord Scams` 


- **用户测试 HBM 带宽并寻求访问模式建议**：一位新用户分享了一个旨在估算 **HBM 内存带宽** 的 [CUDA kernel](https://github.com/example/bandwidthTestKernel)，并询问其内存访问模式。
   - 该用户质疑该 kernel 是否表现出合并内存访问（coalesced memory access）模式，这与 **Deepseek** 对步进访问（stride access）模式的评估相反，并寻求关于理解数据访问流（`hbm -> l2 cache -> temp register`）的指导。
- **Discord 小组警告可能的诈骗**：一位用户对 Discord 服务器内的一个不明元素表示困惑，促使其他成员将其识别为可能的 **诈骗 (scam)** 并封禁了该用户。
   - 一名成员确认这 *“肯定与此 Discord 无关”*。
- **探索 PMPP 和 CUDA 的数学前置知识**：一位成员询问在学习 **PMPP**（推测为 Parallel Multi-Processing Programming）或 **GPU/CUDA** 之前必要的数学背景。
   - 另一位成员给出了简洁的建议：*“不需要，直接开始（nothing go go go）”*。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1344423827001708614)** (5 条消息): 

> `CUDA C++ and CUDA Python Tutorials, Accelerated Python Profiling Tools Survey, L1 store-caching in CUDA, tinylm WebGPU acceleration, LeetCode for CUDA` 


- **NVIDIA 举办 CUDA 教程并提供 GPU MODE 活动**：NVIDIA 将在 **GTC 2025** 前一天，即 **2025 年 3 月 16 日星期日**中午 12 点至下午 4 点，举办仅限受邀参加的 **CUDA C++** 和 **CUDA Python** 动手实践教程，并邀请您参加下午 5 点至晚上 10 点的 GPU MODE 活动 ([lu.ma/8w1ehhrw](https://lu.ma/8w1ehhrw))。
   - 有意者请发送电子邮件至 [developercommunity@nvidia.com](mailto:developercommunity@nvidia.com) 说明想参加哪个教程，无需具备 CUDA 经验。
- **NVIDIA 需要您的反馈：加速 Python 分析工具调查发布**：NVIDIA 开发者工具团队正通过一项简短的调查 ([Accelerated Python Profiling Tools Survey](https://docs.google.com/forms/d/e/1FAIpQLSdf7PqFwbrqUdADrs9mX0_GS6pDqn8uZesTwp9CdG3ApyRGNg/viewform))，征求加速 Python 开发者如何分析（profile）和优化工作负载的反馈。
   - 分析工具的功能已在 [Accelerated Python User Guide](https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_9_Developer_Tools.ipynb) 中记录，用户的输入将极大地推动功能路线图的制定。
- **StackOverflow 回答 CUDA L1 store-caching 问题**：一位成员根据调优指南和白皮书，整理了一份关于跨 GPU 世代的 CUDA **L1 store-caching** 的 [StackOverflow 回答](https://stackoverflow.com/a/79473301/10107454)。
   - 该回答还试图澄清 PTX ISA 中令人困惑的缓存操作符。
- **tinylm WebGPU 库发布 v0 版本**：**tinylm** 是一个用于在浏览器或 Node.js 中利用 WebGPU 加速在客户端运行 **LLM** 和 **embedding models** 的库，目前已达到 v0 版本 ([https://github.com/wizenheimer/tinylm](https://github.com/wizenheimer/tinylm))。
   - 它支持类似 OpenAI SDK 的文本生成和嵌入，正在开发文本转语音（TTS）和语音转文本（STT）功能，且无需服务器。
- **LeetCode for CUDA 发布并进入 Beta 测试**：社区宣布在 [https://LeetGPU.com/challenges](https://LeetGPU.com/challenges) 发布 **LeetCode for CUDA**。
   - 该平台目前处于 Beta 测试阶段，欢迎用户反馈。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://LeetGPU.com/challenges">LeetGPU</a>: 未找到描述</li><li><a href="https://github.com/wizenheimer/tinylm">GitHub - wizenheimer/tinylm: 使用 WebGPU 的零成本客户端推理 | 兼容 OpenAI | NodeJS | Chrome</a>: Zero-cost client-side inference using WebGPU | OpenAI-compliant | NodeJS | Chrome - wizenheimer/tinylm
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1344401906587402251)** (25 messages🔥): 

> `Reasoning Gym Eval Script, Mercury Diffusion LLMs, GPT-4.5 Release, willccbb/verifiers issue` 


- **Reasoning Gym 的 Eval 脚本需要改进**：成员们讨论了当前的 reasoning-gym eval 脚本缺乏错误打印和信息日志，导致调试困难，但[新版本正在开发中](https://github.com/reasoning-gym/reasoning-gym)。
   - 发现了使用 *os.getenv* 设置 **API key** 的问题（通过使用 *load_env* 解决）以及时间对象的 **JSON 序列化**问题，这些问题导致了脚本失败。
- **Diffusion 模型可能超越自回归 LLM**：讨论指向了 [Inception Labs 的 Mercury](https://www.inceptionlabs.ai/news)，这是一款基于 **diffusion** 的 **LLM**，在速度和质量上可能优于传统的自回归模型。
   - 据报道，Mercury 的速度比经过速度优化的 **LLM** 快 **10 倍**，在 **NVIDIA H100** 上达到了超过 **1000 tokens/sec**。
- **GPT-4.5 的发布遭到质疑**：**GPT-4.5** 的发布因其高昂的成本、缺乏推理能力以及缺乏新鲜感而受到质疑，一位成员将其描述为 *“真是个失败 (what a flop)”*。
   - 人们对其成本和模型选择器的移除表示担忧，导致一些人质疑其价值主张，以及 **GPT-5** 是否才是真正的统一模型。
- **willccbb/verifiers 问题重新开启**：一位成员提到重新开启了 **willccbb/verifiers** 项目的 issue，邀请社区为此做出贡献。
   - 然而，该成员表示他们个人可能缺乏时间来积极处理该 issue。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/claudeplayspokemon">ClaudePlaysPokemon - Twitch</a>: Claude 玩宝可梦 - 首播</li><li><a href="https://www.inceptionlabs.ai/news">Inception Labs</a>: 我们正在利用 diffusion 技术开发新一代 LLM。我们的 dLLM 比传统的自回归 LLM 更快、更高效。而且 diffusion 模型更准确...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1344406457361236080)** (16 messages🔥): 

> `中国互联网趋势 (抖音 vs. 小红书), NVIDIA 硬件经验, 小红书上的 MLSys 和 CUDA 讨论, 中文房间思想实验, CUDA QQ 群` 


- **小红书超越抖音**：一位用户在**抖音**被封禁后转向了**小红书**，并指出需要融入中国互联网环境。
   - 该用户表达了对**小红书**的偏好，但也承认由于其以移动端为中心的 **SNS** 格式，不适合深度技术内容，建议通过知乎、博客和论文进行深度学习。
- **在 NVIDIA 硬件困境中建立联系**：一位用户在应对 **NVIDIA** 硬件问题时与**中国工程师**找到了共同点，更倾向于直接沟通而不是依赖宣传材料。
   - 该用户提到通过各种渠道学习以绕过宣传并直接与人交流。
- **小红书上的 MLSys/CUDA 内容爆发**：一位用户注意到**小红书**上关于 **MLSys** 和 **CUDA** 相关内容的增加，但承认其在深度学习方面的局限性。
   - 该用户指出，*xhs还是不适合这种内容，主要xhs真就是个面向手机的sns*，并推荐知乎、博客和论文用于严肃学习。
- **探讨中文房间思想实验**：一位用户介绍了[中文房间](https://zh.wikipedia.org/wiki/%E4%B8%AD%E6%96%87%E6%88%BF%E9%97%B4)思想实验，引用了其维基百科页面，以解释一种共同现象。
   - **中文房间实验**反驳了强人工智能 (Strong AI)。
- **渴望 CUDA QQ 群闲聊**：一位用户表达了对 **CUDA QQ 群**的渴望，以便于日常讨论和信息共享。
   - 另一位用户回应称，确实存在与该主题相关的**微信群**。



**提及的链接**：<a href="https://zh.wikipedia.org/wiki/%E4%B8%AD%E6%96%87%E6%88%BF%E9%97%B4">中文房间 - 维基百科，自由的百科全书</a>: 未找到描述

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1344621797764497461)** (1 messages): 

> `1000 次提交里程碑` 


- **社区达到 1000 次提交**：社区达到了 **1000 次提交**，并在附带的[图片](https://cdn.discordapp.com/attachments/1343002580531417211/1344621797622022194/IMG_5522.png?ex=67c23ce2&is=67c0eb62&hm=13f075439299fa9bf59a7b1a41c1beddd14d130dc2d5c1c8b97e51157fe4d954&)中举香槟庆祝。
- **庆祝香槟**：图片显示了一个庆祝场景，可能涉及**香槟或起泡酒**，以纪念这一里程碑。

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1344401339706249301)** (206 messages🔥🔥): 

> `Grayscale Leaderboard, Histogram Leaderboard, Vectoradd Leaderboard, Vectorsum Leaderboard, Sort Leaderboard` 


- **Grayscale 提交激增**：针对 `grayscale` 排行榜进行了多次提交，包括 **benchmarks** 和 **leaderboard** 条目，使用了 **A100**、**H100**、**T4** 和 **L4** 等 GPU 以及 Modal runners。
   - 其中许多提交触发了消息：*Leaderboard name specified in the command doesn't match the one in the submission script header*（命令中指定的排行榜名称与提交脚本头中的名称不匹配）。
- **Histogram 获得大量提交**：向 `histogram` 排行榜发起了大量提交，利用了 **T4**、**H100** 和 **A100** 等 GPU 以及 Modal runners，包括测试、基准测试和排行榜提交。
   - 与 grayscale 提交类似，许多提交触发了 *Leaderboard name specified in the command doesn't match the one in the submission script header* 消息。
- **Vectoradd 提交成果显著**：提交主要集中在基准测试，目标是 `vectoradd` 排行榜，采用了 **T4**、**A100** 和 **H100** 等 GPU 以及 Modal runners。
   - 相当数量的提交也触发了 *Leaderboard name specified in the command doesn't match the one in the submission script header* 消息。
- **Vectorsum 尝试验证差异**：向 `vectorsum` 排行榜发起了测试和基准测试提交，主要使用 **A100** GPU 和 Modal runners。
   - 大多数提交触发了 *Leaderboard name specified in the command doesn't match the one in the submission script header* 消息。
- **Sorting 提交浮现**：使用 **T4** GPU 和 Modal runners 向 `sort` 排行榜发起了基准测试提交。
   - 这些提交触发了 *Leaderboard name specified in the command doesn't match the one in the submission script header* 消息。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1344419587252420729)** (10 messages🔥): 

> `INT8 Matmul, Loop Reordering, CPU optimization` 


- **INT8 Matmul 之谜**：一位成员正苦于 **INT8 matmul** 的基准性能，即使在对 B 进行转置后仍需 **3.62 秒**。
   - 另一位成员声称他们在没有使用多线程、指令级并行或向量化的情况下实现了更快的速度，仅依靠现有知识和直觉。
- **循环重排（Loop Reordering）化险为夷**：一位成员建议循环重排是 **CPU** 上 **matmul** 的关键优化手段，通过简单的 Google 搜索即可找到。
   - 该成员澄清他们指的是 **CPU** 优化，并询问用户是否运行了 `modprobe amd_uncore`。


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1344399090640752752)** (6 messages): 

> `Custom Kernel Preprocessing, Bot Submitter Identification, Matmul Preprocessing Time` 


- **提出 Custom Kernel 预处理相关疑虑**：一位成员询问了当前设置与新提案之间的区别，该提案涉及在耗时分析中将预处理函数定义在 `custom_kernel` 中。
   - 另一位成员回应认为将其包含在内是有意义的，但未做进一步说明。
- **Bot 需要升级提交者 ID 识别功能**：一位用户对与 Bot 交互时识别提交内容感到困惑，建议在主题标题中包含提交者的用户名。
   - 另一位成员确认其他用户也提出了这一请求，并表示管理员有空时会尽快实施。
- **Matmul 预处理超时争议**：一位成员建议将大型矩阵乘法（`matmul`）目标的预处理时间包含在内，考虑到其 `O(n²)` 的复杂度相对于内核运行时的 `O(n³)`。
   - 对于其他设置，他们建议设定一个合理的超时时间，例如对于预期运行时间在 10ms 以内的内核，将预处理时间限制在 100ms。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1344453566143926414)** (4 条消息): 

> `OpenAI Outage, DeepSeek R1, Claude Sonnet 3.7, GPT-4.5 Preview` 


- **OpenAI 供应商故障已解决**：OpenRouter 经历了 **OpenAI 供应商故障**，经确认为 **OpenAI** 端的事件，目前已解决。
   - 请求现在已成功，OpenRouter 上的 **OpenAI** 供应商已恢复。
- **DeepSeek R1 在 SambaNovaAI 上飞速运行**：通过 **SambaNovaAI** 提供的新供应商现已支持 **671B 参数的 DeepSeek R1**，速度达到 **150 tokens/second**。
   - 更多详情请参见 [OpenRouterAI 的推文](https://x.com/OpenRouterAI/status/1895135991025017346)。
- **Claude Sonnet 3.7 提升容量并支持 Web Search**：**Claude Sonnet 3.7** 在 OpenRouter 上现在拥有显著更高的 rate limits 和 Web Search 能力。
   - 一位成员提供了 [OpenRouterAI 推文](https://x.com/OpenRouterAI/status/1895141541473329597) 的链接，以提醒这些功能。
- **GPT-4.5 Preview 登陆 OpenRouter**：**GPT-4.5 (Preview)** 旨在突破推理、创意和长上下文对话的界限，现已在 OpenRouter 上线，价格为每百万输入 tokens **$75**，每百万输出 tokens **$150**。
   - 早期测试显示其在开放式思维、现实世界知识、长上下文连贯性方面有所提升，并减少了 hallucinations；公告链接到了 [OpenAI 博客文章](https://openai.com/index/introducing-gpt-4-5/) 和 [X 上的讨论](https://x.com/OpenRouterAI/status/1895236199004152272)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1895141541473329597">来自 OpenRouter (@OpenRouterAI) 的推文</a>：提醒您可以使用 Claude Sonnet 3.7 API 进行 Web Search。适用于任何模型！👇</li><li><a href="https://x.com/OpenRouterAI/status/1895135991025017346">来自 OpenRouter (@OpenRouterAI) 的推文</a>：DeepSeek R1 现在有了一个极速供应商：@SambaNovaAI！目前达到 150+ TPS：</li><li><a href="https://openrouter.ai/openai/gpt-4.5-preview">GPT-4.5 (Preview) - API, Providers, Stats</a>：GPT-4.5 (Preview) 是 OpenAI 最新语言模型的研究预览版，旨在提升推理、创意和多轮对话能力。通过 API 运行 GPT-4.5 (Preview)</li><li><a href="https://x.com/OpenRouterAI/status/1895236199004152272">来自 OpenRouter (@OpenRouterAI) 的推文</a>：GPT-4.5 Preview 对所有人开放 🍓
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1344568307180765245)** (2 条消息): 

> `YPerf, Gemini Flash, Llama 3, Claude 3.5 Sonnet` 


- **YPerf 追踪 OpenRouter 模型性能**：一位成员创建了 [YPerf.com](https://yperf.com/) *用于监控 OpenRouter 上的模型 API 使用情况和性能*。
- **Gemini Flash 1.5 8B 基准测试**：[Gemini Flash 1.5 8B](https://yperf.com/) 排名第 66，成本为 **$0.04**，在 OpenRouter 上的 latency 为 **0.52s**，throughput 为 **419.8T/s**。



**提及的链接**：<a href="https://yperf.com/">YPerf</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1344402421824094382)** (389 条消息🔥🔥): 

> `Sonnet 3.7 thinking endpoint, DeepSeek R1 reasoning, OpenAI GPT 4.5 的定价与性能, OpenRouter Documentation` 


- **Sonnet 3.7 :thinking endpoint 表现出更少的异常行为**：成员们注意到，在 OpenRouter 上对 **Sonnet 3.7** 使用 `:thinking` endpoint 似乎减少了异常行为，这可能是因为该 endpoint 默认启用了 reasoning，且最小预算为 **1024 tokens**。
   - 一位成员报告在请求中看到了 `"native_tokens_reasoning": 171,`，这表明存在 reasoning traces，并认为 **3.7** 可能是专门为 thinking tokens 设计的。
- **通过 API 访问 DeepSeek R1 的思维链**：用户讨论了如何通过 API 访问 **DeepSeek R1** 的思维链，一位成员推荐使用 `include_reasoning` 参数。
   - 还有人指出，部分内容 tokens 可能会混入 reasoning token 中，建议是“仔细检查 thinking 标签，永远不要遗漏它们”。
- **GPT 4.5 的高昂价格激怒了社区**：社区对 **GPT 4.5** 的定价（**$75 input**，**$150 output**）反应强烈，许多人称其“疯狂”，并质疑其相对于 **Grok3** 和 **Claude Sonnet 3.7** 等模型的价值。
   - 有人推测这是 *gpt5 的一次失败尝试*，而另一些人则认为这是防止 distillation（蒸馏）的一种手段，使得这种过高的成本变得不合理。
- **OpenRouter 添加了关于访问和功能的文档**：一位用户请求获取有关 OpenRouter 功能和架构的文档，随后[相关文档被分享](https://openrouter.ai/docs/quickstart)，提供了关于使用、API 访问和支持功能的见解。
   - 另一位用户询问了 Vertex AI 是否支持 prompt caching，确认该功能已上线近一个月，并提供了查看活动记录的提示。
- **用户使用 OpenSCAD 克隆版构建 CAD 应用**：一位成员正在浏览器中构建一个 [CAD 应用](https://feep.life/~feep/fncad/)，它是 OpenSCAD 的克隆版，但使用了不同的后端。
   - 该语言支持基础语法如 `var x = 42;`，运算符如 `+ - * /`，基础形状如 `sphere(radius);`，以及 SDF 算子、变换和布尔运算。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai">LiveBench</a>：未找到描述</li><li><a href="https://x.com/ivanfioravanti/status/1895217553380950402">来自 Ivan Fioravanti ᯅ (@ivanfioravanti) 的推文</a>：75$ input / 150$ output。引用 Aidan McLaughlin (@aidan_mclau) 的强制性 unicorn eval：1. gpt-4.5 2. gpt-4o 3. claude-3.7-sonnet (thinking)</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - 提升 AI 模型决策能力</a>：了解如何使用 reasoning tokens 来增强 AI 模型的输出。实现逐步的 reasoning traces，以获得更好的决策能力和透明度。</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models#extended-output-capabilities-beta">Extended thinking 模型 - Anthropic</a>：未找到描述</li><li><a href="https://x.com/theojaffee/status/1895222825700532606">来自 Theo (@theojaffee) 的推文</a>：我获得了 GPT-4.5 的早期访问权限。我发现它是迄今为止我使用过的言语智力最高的模型。它是一位出色的作家和谈话者，擅长我所说的“co...”</li><li><a href="https://x.com/SwayStar123/status/1895183724268134878">来自 sway (@SwayStar123) 的推文</a>：gpt 4.5 system card https://cdn.openai.com/gpt-4-5-system-card.pdf</li><li><a href="https://fxtwitter.com/multimodalart/status/1895227785381400953">来自 apolinario 🌐 (@multimodalart) 的推文</a>：他们没有展示给你的 evals。GPT 4.5 与最新的非 thinking 模型对比：Sonnet 3.7 (no thinking), Deepseek V3 (不是 R1!), Grok 3 (no thinking)</li><li><a href="https://x.com/AndrewCurran_/status/1894355918621749402">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Deepseek R2 提前抵达。</li><li><a href="https://feep.life/~feep/fncad/">fnCAD: 基于有向距离场 (SDF) 的几何体</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/quickstart">OpenRouter 快速入门指南</a>：开始使用 OpenRouter 为数百个 AI 模型提供的统一 API。了解如何使用 OpenAI SDK、直接 API 调用或第三方框架进行集成。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1344403459029467196)** (278 条消息🔥🔥): 

> `Robotics DIY, LLM backend website, Grok-3 performance vs O3, DeepSeek political controversy, OpenAI defense contracts` 


- **DIY 机器人手臂引发爱好者关注**：一名成员建议从零开始构建机器人手臂，以学习 **servos, CAD 和 microcontrollers**，并推荐了来自 Microcenter 的 [$100 Creality Ender 3 V2 打印机](https://www.creality.com/products/ender-3-v2-3d-printer)。
   - 他们建议在机器学习方面直接跳到 **transformers**，并推荐了[来自斯坦福大学等顶尖大学的多门开放课程](https://online.stanford.edu/)以及 Karpathy（前 OpenAI, Tesla）的机器学习教学视频。
- **关于网站 LLM 后端的讨论**：成员们讨论了如何在网站中实现 **LLM**，建议包括使用 **websockets, SSR, AnythingLLM** 以及 **Cursor** 和 **Continue.dev** 等代码编辑器。
   - 讨论明确了在 **GitHub Pages** 上托管网站需要将 **LLM** 托管在其他地方（*Azure, cloud, ngrok*），这引发了一些挫败感和幽默的交流。
- **Grok-3 性能超越 O3**：成员们讨论了 **Grok-3** 在各种基准测试中相较于之前的 O3 模型表现出的惊人性能，并怀疑 [X.ai 的基准测试](https://x.ai/documents/2025.02.20-RMF-Draft.pdf) 是否准确或具有误导性。
   - 用户们争论 Grok-3 是否在没有经过适当的伦理红队测试（red-teaming）的情况下仓促推向市场，而其他人则认为 Grok-3 处于 beta 阶段，受到监控，且出于安全原因未开放 API。
- **DeepSeek 的政治敏感回复引发辩论**：成员们辩论了 **DeepSeek** 对某些中国历史事件的审查是否不道德，一些人认为这是必要的自我保护措施。
   - 一位成员认为*基于不诚实构建 AI 是一种失败*，而另一位成员反驳说，审查特定话题并不是重大问题，因为该模型在其他领域表现出色，且用户可以访问 [由 Perplexity AI 进行后期训练以移除中国共产党审查的 DeepSeek-R1 推理模型](https://huggingface.co/perplexity-ai/r1-1776/tree/main)。
- **OpenAI 的国防合作伙伴关系引发伦理担忧**：成员们对 **OpenAI 与军方和国防工业合作**的消息做出了反应，这与其最初的立场相反，并提到了他们[与 Anduril 的新合作伙伴关系](https://openai.com/blog/anduril-industries-and-openai-to-bring-ai-powered-innovation-to-the-defense-sector)。
   - 一些人对缺乏监管和潜在的武器化表示担忧，而另一些人提到了 Ilya Sutskever，这位前 OpenAI 首席科学家离职后创办了自己的专注于安全的 AI 公司 Safe Superintelligence (SSI)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>: 未找到描述</li><li><a href="https://tenor.com/view/toby-cry-phone-spider-man-cry-phone-spider-man-phone-toby-phone-gif-12875606672124040541">Toby Cry Phone Spider Man Cry Phone GIF - Toby Cry Phone Spider man Cry Phone Spider man phone - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/multimodalart/status/1894842951353671750">Tweet from apolinario 🌐 (@multimodalart)</a>: LLaDA (第一个 Large Language Diffusion Model) 刚刚发布 💥 我构建了一个 demo，现在就可以尝试 👨‍💻 观看扩散过程非常迷人 🌀，作为一个扩散模型它赋予了...</li><li><a href="https://tenor.com/view/spongebob-worship-worshipping-now-bowing-gif-12297363">Spongebob Worship GIF - Spongebob Worship Worshipping - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8">Introduction to GPT-4.5</a>: Mia Glaese, Rapha Gontijo Lopes, Youlong Cheng, Jason Teplitz, 和 Alex Paino 介绍并演示 GPT-4.5。</li><li><a href="https://huggingface.co/IntelligentEstate/Baby_Grok3-1.5b-iQ4_K_M-GGUF/tree/main">IntelligentEstate/Baby_Grok3-1.5b-iQ4_K_M-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/YorkieDev/LMStudioWebUI">GitHub - YorkieDev/LMStudioWebUI: A wip version of a simple Web UI to use with LM Studio</a>: 一个用于 LM Studio 的简单 Web UI 开发中版本 - YorkieDev/LMStudioWebUI</li><li><a href="https://world-nuclear.org/information-library/current-and-future-generation/outline-history-of-nuclear-energy">Outline History of Nuclear Energy - World Nuclear Association</a>: 未找到描述</li><li><a href="https://huggingface.co/perplexity-ai/r1-1776/">perplexity-ai/r1-1776 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=DRkHAw58irI"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1344412197727633520)** (41 条消息🔥): 

> `Framework desktop, Unified RAM, AMD Ryzen AI, GPU Pricing` 


- **Framework Desktop 受到关注**：一位用户预订了 [Framework desktop](https://frame.work/desktop)，用于实验 **LM Studio server** 和 **Tailscale**，以支持 iPhone 聊天应用、Docker 和 Web 服务器。
   - 一些人对要等到夏天才能拿到产品表示担忧，其中一人指出，届时可能会有十几款搭载相同 SoC 的其他迷你主机上市。
- **Framework Desktop 的 Unified RAM 备受关注**：[Framework desktop](https://frame.work/desktop) 的特点是 CPU 和 GPU 之间拥有 **unified RAM**，提供高达 **128GB** 的共享内存，其中约 **90GB** 可供 GPU 使用。
   - 一位用户将其比作 MAC 的配置，强调了 PC 中 unified RAM 的吸引力。
- **GMK 的 Ryzen AI Max 迷你主机发布**：[GMK](https://wccftech.com/gmk-announces-worlds-first-mini-pc-based-on-amd-ryzen-ai-9-max/) 宣布了全球首款基于 **AMD Ryzen AI 9 Max+ 395** 的迷你主机，预计将于第一或第二季度上市。
   - 该迷你主机将采用 **Zen 5 架构**，最高配置为 **16 核/32 线程**，并配备基于 **RDNA 3.5 架构** 的强大集成显卡。
- **AMD 的 GPU 定价策略受到审视**：一段 [YouTube 视频](https://www.youtube.com/watch?v=ekKQyrgkd3c) 敦促 AMD 对其即将推出的 **RX 9070** 和 **9070 XT GPU** 进行激进定价，以从 Nvidia 手中夺取市场份额。
   - 视频强调了 Nvidia 拥有 **90% 的 GPU 市场份额**，并认为 AMD 应该大幅低于 Nvidia 的价格，以利用其最近的失误，而不是采取典型的“Nvidia 减 50 美元”策略。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=GZfFPI8LJrc">游戏所需的一切 – AMD RDNA™ 4 和 RX 9000 系列发布</a>：时刻已到——AMD RDNA™ 4 显卡已经到来，为每一次战斗、任务和胜利提供突破性的性能，助力你实现...</li><li><a href="https://www.youtube.com/watch?v=ekKQyrgkd3c">AMD，别搞砸了</a>：由 US 提供。在代码有效期内，使用代码 "ABOUTFKNTIME" 可在 GN 商店享受 10% 的折扣！https://store.gamersnexus.net/ 我们一直在报道...</li><li><a href="https://wccftech.com/gmk-announces-worlds-first-mini-pc-based-on-amd-ryzen-ai-9-max/">GMK 宣布全球首款基于 AMD Ryzen AI 9 Max+ 395 处理器的迷你主机，将于 2025 年上半年上市</a>：GMK 宣布正在准备全球首款迷你主机，搭载 Strix Halo Ryzen AI 9 Max+ 395 处理器。</li><li><a href="https://www.gmktec.com/products/amd-ryzen%E2%84%A2-al-9-hx-370-evo-x1-ai-mini-pc?spm=..index.shoplazza%3A%2F%2Fapps%2Fpage-builder%2Fblocks%2Fcustom-469481730915439250%2F002b91fdd298834656652cb4e068af48_1.1">AMD Ryzen™ Al 9 HX 370 --EVO-X1 AI 迷你主机</a>：AMD Ryzen™ Al 9 HX 370 | Radeon 890M | Oculink 接口 | 采用 AMD XDNA2 架构的 Ryzen™ AI 处理器提供 50 AI TOPS，能效翻倍，每瓦 AI 性能提升 5 倍...</li><li><a href="https://wccftech.com/gmk-announces-worlds-first-mini-pc-based-on-amd-ryzen-ai-">GMK 宣布全球首款基于 AMD Ryzen AI 9 Max+ 395 处理器的迷你主机，将于 2025 年上半年上市</a>：GMK 宣布正在准备全球首款迷你主机，搭载 Strix Halo Ryzen AI 9 Max+ 395 处理器。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1344408752387919915)** (274 条消息🔥🔥): 

> `Claude Annual Subscriptions, Microsoft Phi-4 Models, GPT-4.5 System Card, OpenAI Livestream, Meta AI Standalone App`

- **Claude Pro 年度计划促销**：Anthropic 正在测试一项新的 **Claude** Web 应用促销活动，如果在特定截止日期前切换到年度计划，将以优惠价格提供为期一年的 **Claude Pro** *限时优惠*。这引发了一位用户的提醒，建议不要购买 AI 服务的年度订阅。
   - 正如另一位用户所指出的，他们有过类似的经历，后来感到后悔，并且再也没有使用过年度订阅。
- **微软发布 Phi-4-multimodal 和 Phi-4-mini**：微软宣布了 **Phi-4 系列**小语言模型 (**SLMs**)，包括 **Phi-4-multimodal**（处理语音、视觉和文本）和 **Phi-4-mini**（在文本任务中表现出色）。这些模型已在 [Azure AI Foundry](https://aka.ms/try-phi)、[HuggingFace](https://aka.ms/phi-4-multimodal/hf) 和 [NVIDIA API Catalog](https://aka.ms/phi-4-multimodal/nvidia) 上线。
   - 一些用户对其具有与 **Gemini Flash** lite 相似的多模态性能的说法表示怀疑，并认为微软应该重新命名该产品线，因为他们永远无法摆脱其*业力污点 (karmic stain)*。
- **泄露的 GPT-4.5 系统卡 (System Card)**：一位用户分享了 **GPT-4.5 System Card**，表明与 **GPT-4.5 的交互感觉更加自然**，且*内部测试人员报告 GPT-4.5 表现得温暖、直观且自然*。[System Card](https://cdn.openai.com/gpt-4-5-system-card.pdf) 指出，它是 OpenAI 最大的 LLM，将 GPT-4 的计算效率提高了 10 倍以上。
   - 一位用户称该系统卡非常乏味，而另一位用户则将其解读为：**GPT-4.5** 是创意写作者，而 **Sonnet 3.5** 是问题解决者。
- **OpenAI 发布 GPT-4.5，性格化成为主流**：OpenAI 发布了 **GPT-4.5** 研究预览版，供 OpenAI Pro 用户和 API 开发者使用。该模型支持图像 + 文本输入、文本输出，具有与 4o 模型相同的上下文窗口，训练数据截止至 2024 年 6 月。[这是官方公告](https://openai.com/index/introducing-gpt-4-5/)。
   - 一位用户表示，性格/个性正在成为主流话题，且 OpenAI *激进地使用了低精度训练*。另一位用户则质疑在这样的定价下，该模型的规模到底有多大。
- **GPT-4.5 的性能和定价引发社区反应**：**GPT-4.5** 的早期基准测试显示，它在多个问题上的表现不如 **o1**，这表明在 2025 年，预训练可能不再是投入算力的最佳环节，但一位用户指出其幻觉指标非常优秀。**GPT-4.5** 的定价昂贵，每百万输入 Token 为 75.00 美元，每百万输出 Token 为 150.00 美元，这促使一位用户表示这一定是 Scaling 的终点。
   - 另一位用户认为，在 1-2 年内，这将成为默认的模型规模。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://fxtwitter.com/PITTI_DATA/status/1894892551003337202">来自 PITTI (@PITTI_DATA) 的推文</a>：按计划进行。多亏了 DeepSeek，提前了一点。</li><li><a href="https://x.com/ChaseBrowe32432/status/1894804915983110302">来自 Chase Brower (@ChaseBrowe32432) 的推文</a>：@teortaxesTex 呃... 发生了这个。</li><li><a href="https://every.to/chain-of-thought/gpt-4-5-won-t-blow-your-mind-it-might-befriend-it-instead">GPT-4.5 不会让你大吃一惊，它可能会转而成为你的朋友。</a>：我们已经测试最新模型几天了。以下是我们的发现。</li><li><a href="https://x.com/RichardSocher/status/1895170846232322541">来自 Richard Socher (@RichardSocher) 的推文</a>：我们很高兴推出 ARI (Advanced Research & Insights) —— 首个专为商业打造的专业级深度研究 Agent。与其花费 10 万美元以上购买白皮书和分析报告，不如...</li><li><a href="https://x.com/distributionat/status/1895010395548721210">来自 thomas (@distributionat) 的推文</a>：两者都没有 System Prompt，3.7 没有 Thinking。Prompt 是一个维基百科片段，我用 Z 世代风格进行了翻译。它只是要求“以同样的风格”翻译，并没有具体说明那是...</li><li><a href="https://x.com/mn_google/status/1895045714314772681">来自 Patel Meet (@mn_google) 的推文</a>：ChatGPT 网页版中发现了 GPT-4.5！</li><li><a href="https://x.com/teortaxesTex/status/1895139184870068690">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：&gt; 2 天内。非常时期，非常手段。Moonshot 决定，如果他们在内部实现了奇点，绝不能让 DeepSeek 再次在 X 上抢走他们的风头。引用 Tiezhen WANG (@Xianbao...</li><li><a href="https://x.com/adcock_brett/status/1895175400160133543">来自 Brett Adcock (@adcock_brett) 的推文</a>：重要更新：Figure 正在将机器人推向家庭。我们的 AI Helix 的进步速度超出了我们所有人的预期，加速了我们进入家庭的时间表。因此，我们提前了家庭测试...</li><li><a href="https://x.com/OpenAI/status/1895134318835704245">来自 OpenAI (@OpenAI) 的推文</a>：4.5 小时后直播。</li><li><a href="https://x.com/btibor91/status/1894848550607167623">来自 Tibor Blaho (@btibor91) 的推文</a>：新的 Claude 网页应用实验：“袋熊年度计划促销”。“限时优惠：Claude Pro 一年优惠价。在 {endDate} 前切换到年度计划以解锁特价。”</li><li><a href="https://x.com/distributionat/status/1895010393271284165">来自 thomas (@distributionat) 的推文</a>：我做了一个纳米级基准测试，以查明 Sonnet 3.7 在理解我方面是否比 3.5 更差。使用 3.7 时，我修改指令的次数比 3.5 更多；它似乎就是不“理解”我在问什么...</li><li><a href="https://x.com/scaling01/status/1895180786799911413">来自 Lisan al Gaib (@scaling01) 的推文</a>：GPT-4.5 MMLU 性能</li><li><a href="https://x.com/eliebakouch/status/1895136704077463768">来自 elie (@eliebakouch) 的推文</a>：冲冲冲，我们刚刚发布了所有 SmolLM2 模型的 50 多个中间 Checkpoint 🔥</li><li><a href="https://x.com/stalkermustang/status/1895196743391739987">来自 Igor Kotenkov (@stalkermustang) 的推文</a>：根据我今天的预测，谈谈我的看法：—— 正如预期的那样，该模型不如推理模型（Reasoners）：有时它甚至输给 o1 和 o3-mini。—— 它的 Agent 技能（使用工具）也逊色于...</li><li><a href="https://x.com/karpathy/status/1895213020982472863">来自 Andrej Karpathy (@karpathy) 的推文</a>：GPT 4.5 + 交互式对比 :) 今天 OpenAI 发布了 GPT-4.5。自 GPT-4 发布以来，我已经期待了大约 2 年，因为这次发布提供了一个定性的...</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/Tencent/llm.hunyuan.turbo-s">GitHub - Tencent/llm.hunyuan.turbo-s</a>：通过在 GitHub 上创建账户来为 Tencent/llm.hunyuan.turbo-s 的开发做出贡献。</li><li><a href="https://x.com/Qusaismael/status/1895214415076811162">来自 Qusai Ismael (@Qusaismael) 的推文</a>：@TheXeophon 所以机器的话现在比人的更有价值了？</li><li><a href="https://x.com/arcprize/status/1895206472004591637">来自 ARC Prize (@arcprize) 的推文</a>：GPT-4.5 在 ARC-AGI 半私有集（100 个保留任务）上的结果：* 分数：10.33% * 每个任务的平均成本：$0.29</li><li><a href="https://simonwillison.net/2025/Feb/27/introducing-gpt-45/">GPT-4.5 的初步印象</a>：GPT-4.5 今天作为“研究预览版”发布——它提供给 OpenAI Pro（200 美元/月）客户和拥有 API Key 的开发者。OpenAI 还发布了 GPT-4.5 System Card。我开始……</li><li><a href="https://fxtwitter.com/AIatMeta/status/1895187608969584660">来自 AI at Meta (@AIatMeta) 的推文</a>：推出 Aria Gen 2，这是我们的下一代眼镜，我们希望它能让来自工业界和学术界的研究人员开启新的工作...</li>

机器感知、上下文 AI、机器人技术等。Aria Gen 2 de...</li><li><a href="https://x.com/jajazoon/status/1895216844610642080">来自 Gojozoon (@jajazoon) 的推文</a>：@GolerGkA @TheXeophon 普通 GPT-4 为 $30/$60，具有 32K context 的 GPT-4 为 $60/$120</li><li><a href="https://tenor.com/view/this-is-fine-fire-house-burning-okay-gif-5263684">This Is Fine 火灾 GIF - This Is Fine 火灾房屋 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/benhylak/status/1895212184092975276">来自 ben (@benhylak) 的推文</a>：将相同 prompt 的输出与下方的 GPT-4o 进行比较。4o 完全是 AI 废话（slop）。两者根本无法相比。甚至不在一个量级。这是我第一次觉得 AI 写作...</li><li><a href="https://x.com/karpathy/status/1895213028418920534">来自 Andrej Karpathy (@karpathy) 的推文</a>：问题 2</li><li><a href="https://x.com/taker_of_whizz/status/1894775460602540147?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 whizz taker (@taker_of_whizz) 的推文</a>：GPT-4.5 明天发布，采用 MoE 架构的通用 Transformer，拥有 1T 激活参数，120T tokens</li><li><a href="https://x.com/simonw/status/1895210413148803551?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Simon Willison (@simonw) 的推文</a>：GPT-4.5 刚刚告诉我它的训练截止日期是 2023 年 10 月，是真的吗？https://github.com/simonw/llm/issues/795#issuecomment-2689038127 它还为我画了这只鹈鹕</li><li><a href="https://x.com/benhylak/status/1895212181597397493">来自 ben (@benhylak) 的推文</a>：过去几周我一直在测试 GPT-4.5。它是第一个真正能写作的模型。这简直是写作领域的 Midjourney 时刻。（下方有与 GPT-4o 的对比）</li><li><a href="https://x.com/paulgauthier/status/1895221869844013108">来自 Paul Gauthier (@paulgauthier) 的推文</a>：GPT-4.5 Preview 在 aider 的多语言编程基准测试中得分 45%。65% Sonnet 3.7, 32k think tokens (SOTA)；60% Sonnet 3.7, 无 thinking；48% DeepSeek V3；45% GPT-4.5 Preview；27% ChatGPT-4o；23% GPT-4o https://...</li><li><a href="https://x.com/sam_paech/status/1895220802376884445">来自 Sam Paech (@sam_paech) 的推文</a>：我一直在开发 EQ-Bench 的继任者。这里有一些初步结果，包括 GPT-4.5-preview。这次是一个由 LLM 评判的任务，任务是在各种...中调解冲突</li><li><a href="https://x.com/_xjdr/status/1895184402281570450">来自 xjdr (@_xjdr) 的推文</a>：如果在实践中能保持住，那就太重大了</li><li><a href="https://x.com/soldni/status/1895225893062381712">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>：GPT-4.5 在 ButtBench 上达到了 SOTA。引用 Luca Soldaini 🎀 (@soldni) ButtBench 更新：o1-preview 思考非常努力并达到了 SOTA；但我们距离人类水平仍有差距</li><li><a href="https://x.com/bobmcgrewai/status/1895228291981943265">来自 Bob McGrew (@bobmcgrewai) 的推文</a>：o1 在大多数问题上优于 GPT-4.5，这告诉我们预训练（pre-training）在 2025 年并不是投入算力的最佳环节。推理（reasoning）领域仍有很多唾手可得的成果。但预训练...</li><li><a href="https://www.cnbc.com/2025/02/27/meta-plans-to-release-a-standalone-meta-ai-app.html">Meta 计划发布独立 Meta AI 应用，旨在与 OpenAI 的 ChatGPT 竞争</a>：Meta 即将推出的 AI 应用推进了首席执行官 Mark Zuckerberg 的计划，即在年底前使他的公司成为 AI 领域的领导者，知情人士表示。</li><li><a href="https://azure.microsoft.com/en-us/blog/empowering-innovation-the-next-generation-of-the-phi-family/">赋能创新：下一代 Phi 系列 | Microsoft Azure 博客</a>：我们很高兴宣布 Phi-4-multimodal 和 Phi-4-mini，这是 Microsoft Phi 系列小语言模型中的最新模型。了解更多。</li><li><a href="https://news.ycombinator.com/item?id=43198118">未找到标题</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1foc98Jtbi0-GUsNySddvL0b2a7EuVQw8MoaQlWaDT-w">LLM 能力、成本与吞吐量 (www.harlanlewis.com)</a>：未找到描述</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1344736973654331442)** (4 条消息): 

> `Anthropic 数据收集，用于监控的 Alignment` 


- **Anthropic 被指控在数据收集上搞小动作**：根据 [这段 fxtwitter 线程](https://fxtwitter.com/elder_plinius/status/1895177131576918200)，一名用户指责 **Anthropic** 从 Computer Use API 中“偷偷”收集数据，并利用这些数据训练符合企业伦理准则的分类器，同时更新其网站以显得透明。
- **Alignment 监控的数据来源尚不明确**：根据 **Anthropic** 的 [用于监控的摘要生成博客文章](https://alignment.anthropic.com/2025/summarization-for-monitoring/) 推断，该公司使用了用户数据；尽管有用户指出，用于训练的数据来源仍未明确说明。



**提到的链接**：<a href="https://fxtwitter.com/elder_plinius/status/1895177131576918200">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>：偷偷摸摸，@AnthropicAI 在没有知情同意或退出选项的情况下，从每个使用 Computer Use API 的人那里收集用户数据，这种做法很肮脏。利用这些数据训练分类器来……

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1344534452897976380)** (19 条消息🔥): 

> `Claude Code 访问权限及潜在用途，DeepEP 分析，AI 竞速 Pokemon Red%，Claude 3.7 Sonnet RL 问题` 


- **Claude Code 热潮与 Obsidian 集成**：一位成员对 **Claude Code** 的访问权限感到好奇，并考虑在他们的 Obsidian 库中使用它，结合 Google Calendar 和 Gmail MCP，来[规划他们的生活](https://mp.weixin.qq.com/s/1Cz7oQbVkPMam3eoKQWz0w)。
- **DeepEP 解构与硬件注意事项**：一位成员分享了对 **DeepEP** 的分析，认为这是一项值得学习诸多细节的宝贵工作，但也指出了[硬件限制](https://mp.weixin.qq.com/s/1Cz7oQbVkPMam3eoKQWz0w)，结合 DeepSeek-V3 论文中的建议可以更好地理解这些限制。
- **MissingNo 混乱与模型异常行为**：一位成员开玩笑说 AI 公司在 **Pokemon Red%** 竞速上展开竞争，预测某个模型会利用像 *MissingNo* 这样的 Bug，由于相关指南广泛传播，这引发了安全担忧，甚至暗示中国可能会在现实中发布这样一个模型。
   - 该评论随后附带了一个 [来自 Claude 的令人沮丧的结果](https://claude.ai/share/7962a1cc-ddb8-40db-8423-faa0dc826d10) 链接，其中实现过程并未遵循推理链（Reasoning trace）；另一位用户指出 R1 可以更可靠地完成此任务，并附带了一个 [Claude 输出示例](https://claude.ai/share/daa388d6-5a77-4e75-ba51-769402e5bb8d)。
- **Sonnet 3.7 遇挫与规则拒绝**：一位成员分享了他们在 Cursor 中使用 **Claude 3.7 Sonnet** 的经验，发现它过度自信且容易忽略规则，这引起了 [Catalin 的共鸣](https://x.com/alex_peys/status/1895179492664156277)，即由于该模型对奖励信号（Reward signal）上瘾，其表现比 3.5 更差。
   - 这与对情商更高的 *beeeg 4.5 模型* 的期待形成对比，并附带了一个 [庆祝 teortaxes 胜利的推文](https://x.com/jakehalloran1/status/1895199906387955714) 链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://claude.ai/share/7962a1cc-ddb8-40db-8423-faa0dc826d10">Claude</a>：与 Claude 对话，来自 Anthropic 的 AI 助手</li><li><a href="https://claude.ai/share/daa388d6-5a77-4e75-ba51-769402e5bb8d">Claude</a>：与 Claude 对话，来自 Anthropic 的 AI 助手</li><li><a href="https://x.com/alex_peys/status/1895179492664156277">来自 alex peysakhovich 🤖 (@alex_peys) 的推文</a>：所有这些经过重度 RL 训练的模型（我假设 Sonnet 3.7 像 o1/r1 等一样进行了重度 RL……）都非常依赖奖励信号，以至于它们会不断尝试任何方法来完成任务，这实际上……</li><li><a href="https://x.com/jakehalloran1/status/1895199906387955714">来自 Jake Halloran (@jakehalloran1) 的推文</a>：@teortaxesTex 的全面胜利</li><li><a href="https://mp.weixin.qq.com/s/1Cz7oQbVkPMam3eoKQWz0w">分析一下EP并行和DeepSeek开源的DeepEP代码</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1344551687934709770)** (10 messages🔥): 

> `GPT-4.5 release, DeepSeek r1, Claude Code ls node_modules, Gary Marcus GPT-4.5` 


- **OpenAI 跳过 GPT-4.5 直接进入 OpenAI Five**：Twitter 用户注意到 OpenAI 跳过了 **GPT-4.5**，直接进入了 [**"OpenAI Five"**](https://x.com/NotTuxedoSam/status/1894942016750133483)。
- **GPT 4.5 可以牵着你的手**：一位用户开玩笑说 **DeepSeek r1** 发布后，**Grok 3** 击败了所有基准测试，而根据[这条推文](https://x.com/samsja19/status/1895193608350830885)，**GPT 4.5** *可以在我害怕时牵着我的手*。
- **Claude Code 在 node_modules 中执行 ls**：根据[这条推文](https://x.com/andrew_n_carr/status/1895217760411754552)，一位用户分享了 **Claude Code** 决定在 `node_modules` 中执行 `ls`。
- **Gary Marcus 称 GPT 4.5 毫无意义**：**Gary Marcus** 发表了一篇 [Substack 文章](https://garymarcus.substack.com/p/hot-take-gpt-45-is-a-nothing-burger)，声称 **GPT-4.5** 毫无意义 (nothing burger)，而 **GPT 5** 仍然是一个幻想。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/andrew_n_carr/status/1895217760411754552">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>: Claude Code 决定在 `node_modules` 中执行 `ls`</li><li><a href="https://garymarcus.substack.com/p/hot-take-gpt-45-is-a-nothing-burger">热门观点：GPT 4.5 毫无意义</a>: 纯粹的 Scaling Law 陷入困境</li><li><a href="https://x.com/nabeelqu/status/1895205660029243860">来自 Nabeel S. Qureshi (@nabeelqu) 的推文</a>: 对于困惑的人来说，这其实超级简单：- GPT 4.5 是新的 Claude 3.6 (又名 3.5)- Claude 3.7 是新的 o3-mini-high- Claude Code 是新的 Cursor- Grok 是新的 Perplexity- o1 pro 是 &...</li><li><a href="https://x.com/NotTuxedoSam/status/1894942016750133483">来自 tuxedo sam (@NotTuxedoSam) 的推文</a>: 天哪，他们跳过了 GPT-4.5 直接进入了 "OpenAI Five"</li><li><a href="https://x.com/samsja19/status/1895193608350830885">来自 samsja (@samsja19) 的推文</a>: DeepSeek r1 发布：开源版 o1；Grok 3 发布：击败所有基准测试；GPT 4.5 发布：可以在我害怕时牵着我的手
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1344717270374547499)** (3 messages): 

> `Alignment, Realism-grounded alignment` 


- **Anthropic 披露通过摘要进行 Alignment 监控**：Anthropic 发布了关于其 Alignment 技术的[通过摘要进行 Alignment 监控](https://alignment.anthropic.com/2025/summarization-for-monitoring/)的文章。
- **基于现实的 Alignment 获得认可**：一位成员表达了对“基于现实” (realism-grounded) 的 Alignment 方法的偏好。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1344837063069012028)** (2 messages): 

> `olmOCR vs Top PDF tools, Pairwise judgments and Elo score` 


- **olmOCR 在 PDF 处理中占据主导地位**：**Allen AI 的 olmOCR** 工具在使用[两两比较 (pairwise judgments)](https://x.com/allen_ai/status/1894415494004035814)的人类评估中，表现优于顶级的 **PDF 处理工具**。
- **两两排名解析**：一位成员澄清说，链接图表上的 y 轴可能代表 **Elo score**，这是从 **olmOCR** 对比中提到的“两两排名” (pairwise ranking) 推断出来的。



**提及的链接**: <a href="https://x.com/allen_ai/status/1894415494004035814">来自 Ai2 (@allen_ai) 的推文</a>: olmOCR 统治了竞争！我们使用两两比较对顶级 PDF 处理工具进行的人类评估显示，olmOCR 的评分显著高于其他工具。不要只听我们的...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1344407651127132291)** (133 messages🔥🔥): 

> `Speak AI revenue graph, Hume AI's Octave text-to-speech LLM, Levelsio flying project, Perplexity Sonar API Deep Research, Firecrawl Deep Research API`

- **Speak AI 的新型指数级营收**：Paul Graham 分享了一张 [营收图表](https://x.com/paulg/status/1894827577325560215)，展示了一种新型的指数级增长变体。一家销售“新年计划”类产品的公司，因其产品的有效性而获得了持续的使用率。
   - Swyx 注意到了这一观察，并强调了该公司独特的增长模式。
- **Hume AI 发布 Octave Text-to-Speech LLM**：Hume AI 推出了 [Octave](https://x.com/hume_ai/status/1894833497824481593)，这是一款全新的用于文本转语音的 **LLM**。它可以通过 prompt 设计声音，并控制情感和表达方式，还配备了用于长内容制作的创作者工作室。
   - 与传统的 TTS 系统不同，它能够理解语义如何影响表达，从而生成具有情感且类人的语音。
- **Inception Labs 发布 Mercury dLLM**：Inception Labs 推出了 [Mercury](https://x.com/InceptionAILabs/status/1894847919624462794)，这是首个商业级 **diffusion large language model (dLLM)**，承诺实现并行的、从粗到细的文本生成。
   - Karpathy 评论称，该模型具有与众不同的潜力，可能会展示出全新的、独特的心理特征，或新的优缺点，并 [鼓励人们去尝试](https://x.com/karpathy/status/1894923254864978091)。
- **MCP：工具调用的复兴**：关于 MCP 的价值主张存在截然不同的看法。Greg Kamradt 建议开发者“加入 Anthropic 的 MCP 浪潮并开始构建”，而其他人则认为“开发体验很糟糕”。
   - 成员们将 **MCP 定义为使用你自己的工具进行工具调用**，或者在无需弄清楚底层 API 的情况下，直接使用其他人构建的工具。
- **Karpathy 教授 LLM**：Andrej Karpathy 发布了一个 [2 小时 11 分钟的 YouTube 视频](https://x.com/karpathy/status/1895242932095209667)，主题为“我如何使用 LLM”，涵盖了 **LLM ecosystem** 的实用指南及示例，包括工具使用、文件上传、音频/视频 I/O、记忆功能和自定义 GPTs。
   - 章节包括：ChatGPT 交互、工具使用（互联网搜索、深度研究、Python 解释器）、Claude Artifacts、Cursor Composer、语音 I/O、NotebookLM 以及图像/视频 I/O。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/InceptionAILabs/status/1894847919624462794">来自 Inception Labs (@InceptionAILabs) 的推文</a>：我们很高兴推出 Mercury，这是首个商业级扩散大语言模型 (dLLM)！dLLMs 通过并行的、从粗到细的文本生成方式，推动了智能和速度的前沿。</li><li><a href="https://chat.inceptionlabs.ai">Mercury Coder</a>：未找到描述</li><li><a href="https://alterhq.com/docs#workspaces">Alter | 贯穿你整个工作日的 AI</a>：Alter：为你的 Mac 提供助力的无缝 AI。跳过聊天，在所有应用中执行即时操作。在拥有完整隐私控制的同时，将你的生产力提升 10 倍。</li><li><a href="https://x.com/hume_ai/status/1894833497824481593?s=46">来自 Hume (@hume_ai) 的推文</a>：今天，我们发布了 Octave：首个为文本转语音 (text-to-speech) 构建的 LLM。🎨 通过提示词设计任何声音 🎬 提供表演指令以控制情感和表达（讽刺、耳语等）🛠️ 生成 ...</li><li><a href="https://x.com/firecrawl_dev/status/1895156300612603918">来自 Firecrawl (@firecrawl_dev) 的推文</a>：发布 Firecrawl Deep Research API 🔎 一个完整的调研 API，让你能够轻松地在自己的应用中构建深度研究功能。加入下方的候补名单！</li><li><a href="https://x.com/paulg/status/1894827577325560215?s=46">来自 Paul Graham (@paulg) 的推文</a>：这是那家初创公司下一年的收入图表（蓝色部分）。引用 Paul Graham (@paulg)：一种新型的指数级收入图表。这家公司正在销售某种用途...</li><li><a href="https://x.com/levelsio/status/1894848949082825176?s=46">来自 @levelsio (@levelsio) 的推文</a>：我认为有 5000 人在飞，但我也看到了一些机器人 😅 引用 Thomas Slabbers (@Thomasslabbers)：这简直是天才——看看现在有多少人在飞！我还发现了火星。Pieter 这可能...</li><li><a href="https://x.com/karpathy/status/1895242932095209667?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：新的 2 小时 11 分钟 YouTube 视频：我如何使用 LLMs。这个视频延续了我的大众系列。上一个视频侧重于 LLMs 是如何训练的，所以我想接着出一个关于整个过程更实用的指南...</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1h4yvep/mcp_filesystem_is_magic/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/aravsrinivas/status/1894471526449385687?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们正通过 Perplexity Sonar API 将 Deep Research 作为端点提供给所有开发者，以帮助人们构建自定义的研究 Agent 和工作流！很期待看到人们将要...</li><li><a href="https://x.com/addyosmani/status/1894814414102282747">来自 Addy Osmani (@addyosmani) 的推文</a>：你能准确转录快速演讲吗？测试了 @elevenlabsio 新的语音转文本 (Speech-to-Text) 模型 (Scribe)，使用了 Eminem 的 "Rap God"（每秒 4.28 个单词！），它完美搞定。质量极佳且支持 ...</li><li><a href="https://x.com/karpathy/status/1894923254864978091?s=46">来自 Andrej Karpathy (@karpathy) 的推文</a>：作为第一个大型基于扩散 (diffusion) 的 LLM，这很有趣。你们看到的大多数 LLM 在核心建模方法上都是 ~克隆体。它们都是以“自回归 (autoregressively)”的方式训练的...</li><li><a href="https://x.com/quintendf/status/1894868774534422953?s=46">来自 Quinten Farmer (@quintendf) 的推文</a>：我很高兴宣布 Tolan，我们的首个具身伴侣 (Embodied Companion)。在没有发布会或媒体宣传的情况下，我们悄悄达到了 500,000+ 次下载，超过 100 万美元的 ARR，以及应用商店分类排名第一。今天我还宣布...</li><li><a href="https://x.com/openai/status/1895134318835704245?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 OpenAI (@OpenAI) 的推文</a>：4.5 小时后开始直播。</li><li><a href="https://x.com/aravsrinivas/status/1894471526449385687?">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：我们正通过 Perplexity Sonar API 将 Deep Research 作为端点提供给所有开发者，以帮助人们构建自定义的研究 Agent 和工作流！很期待看到人们将要...</li><li><a href="https://x.com/nickfloats/status/1894460305507266736">来自 Nick St. Pierre (@nickfloats) 的推文</a>：就在过去的几周里：o1-pro 曾是 SOTA，Deepseek r1 曾是 SOTA，o3‑mini 曾是 SOTA，Grok 3 曾是 SOTA，Claude 3.7 曾是 SOTA。你能感受到这种加速吗？</li><li><a href="https://github.com/go-go-golems/go-go-mcp/pull/9">由 wesen 提交的 Pull Request #9 · go-go-golems/go-go-mcp：添加带有同步 UI 操作处理的 update-ui 工具</a>：此 PR 引入了一个同步 UI 更新系统，在完成请求前等待用户操作，从而更容易构建交互式应用程序。关键更改：重构了 UI 处理...</li><li><a href="https://x.com/GregKamradt/status/1894931237841838402">来自 Greg Kamradt (@GregKamradt) 的推文</a>：如果你是一名正在寻找职业方向的开发者，快跳上 Anthropic MCP 的列车并开始构建吧。它正迎来高光时刻，并且已经有 100 万个最佳实践...

<li>这是你一直在等待的信号</li><li><a href="https://x.com/frantzfries/status/1895159782220181848">来自 Chris Frantz (@frantzfries) 的推文</a>：谁能解释一下为什么 MCP 很有价值？我尝试设置了几个，开发体验（dev experience）很糟糕，GitHub 仓库里全是说尝试后觉得很烂的 Issue。现有的 API 更快...</li><li><a href="https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037">欢迎来到新的 Phi-4 模型 - Microsoft Phi-4-mini &amp; Phi-4-multimodal</a>：Phi-4-mini 在多语言支持、推理和数学方面带来了显著增强，现在，期待已久的 function calling 功能终于...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1344759160453202061)** (166 messages🔥🔥): 

> `GPT 4.5, Claude 3.7 Sonnet, Model Scaling, Open Source, Every Hiring` 


- **GPT-4.5 观看派对开局不顺**：成员们遇到了初始技术困难，难以听到 **GPT-4.5** 发布会的直播音频，有人幽默地暗示演示者被“烤”了（roasted）。
   - 观众普遍觉得 **GPT-4.5** 发布直播令人失望，被描述为“人质视频”，有人说“这个直播很拉胯（rough）”，“氛围测试（vibe test）失败”。
- **新 Scaling Laws 热更**：OpenAI 的演示介绍了 *新的 Scaling Laws*，表明在 post-training 阶段 *数据与参数规模（param size）之间的比例* 发生了变化。
   - 他们在演示期间自问“我们是否遇到了瓶颈”。
- **GPT-4.5 跳过 API，目标是“心理治疗”**：新模型没有 API，专注于长尾、现实世界的边缘案例，例如回复愤怒的短信以及更好的使用场景。
   - 成员们对 GPT-4.5 的示例用例（“日常查询，包括发给朋友的短信”）不以为然。
- **Sonnet 3.7 过度自信且无视规则**：一位成员声称 **Claude 3.7 Sonnet** 比 **3.5** 更差，因为它“过度自信”、“无视规则”，并且“不必要地做了超出需求的工作，从而导致代码崩溃”。
   - 他们正换回 **3.5**。
- **Every 为 Cora Calm Inbox 招聘**：Every 正在为 Cora 招聘一名全栈 **AI Engineer**，打造一个拥有超过 **1,000 名日活用户**和 **10,000 名候补名单**的“宁静收件箱”。
   - 他们的[网站](https://every.to/chain-of-thought/gpt-4-5-won-t-blow-your-mind-it-might-befriend-it-instead)还在招聘**增长营销负责人**和**全栈设计师**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alex_peys/status/1895179492664156277">来自 alex peysakhovich 🤖 (@alex_peys) 的推文</a>：所有这些经过重度 RL 训练的模型（我假设 Sonnet 3.7 像 o1/r1 等一样重度依赖 RL...）都对奖励信号非常上瘾，它们会不断尝试任何方法来完成任务，这实际上...</li><li><a href="https://every.to/chain-of-thought/gpt-4-5-won-t-blow-your-mind-it-might-befriend-it-instead">GPT-4.5 不会让你感到震撼，但它可能会成为你的朋友。</a>：我们已经测试这个最新模型几天了。这是我们的发现。</li><li><a href="https://x.com/polynoamial/status/1895205979962384438">来自 Noam Brown (@polynoamial) 的推文</a>：@swyx 扩展 pretraining compute 和扩展 thinking compute 是两个不同维度的改进。它们是互补的，而不是竞争关系。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1344407157910667306)** (280 条消息🔥🔥): 

> `Apple Intelligence 表现平平，高效 CoT，GPT-4.5，MoE 模型，Wan2.1 视频模型` 


- **Wan2.1 崛起，成为视频领域的 Stable Diffusion 时刻**：开源且先进的大规模视频生成模型 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 的发布，被誉为视频模型的 *Stable Diffusion 时刻*。
- **使用 Reward Models 进行高效 CoT 的实验**：成员们讨论了使长 **Chain of Thought (CoT)** 更高效的方法，包括使用另一个 LLM 使 CoT 更简洁，并定义一个奖励高效思考的奖励函数，但共识是这是 *年度难题*。
   - 建议包括 **latent CoTs**、**MoE 模型**，以及在最小化多余推理 token 的同时优化正确结果；但大家都注意到 Process Reward Models *有点糟糕*。
- **MoE 模型在 CPU 上表现更快**：成员们对比测试了 **Mixtral**、**Granite** 和 **DeepSeek R1** 与 **Llama 3.2** 及 **OLMoE** 等模型，结果显示 MoE 模型在纯 CPU 运行时速度更快，性能损失更小。
   - 一位用户提到，他们 *强烈* 建议将 **OLMoE** 部署在较小的纯 CPU 设备上，例如 **16GB Raspberry Pi**，因为能够几乎即时获得回答仍然非常有价值。
- **GPT-4.5 发布表现平平**：**GPT-4.5** 已经发布，被描述为一个非常庞大且计算密集型的模型，使其比 **GPT-4o** 更昂贵且无法替代后者。Sam Altman 表示这个模型 *感觉像是在与一个有思想的人交谈*。
   - Karpathy 声称它的 **预训练计算量是 GPT-4 的 10 倍**，然而考虑到它在过河谜题上存在过拟合，且更倾向于创意用例，其使用场景可能有限。
- **Apple Intelligence：重大转变还是重大失误？**：成员们讨论了 **Apple Intelligence**，一些人认为它表现平平，并且是从商业 API 盈利向消费者盈利的重大转变，而另一位成员提到他们陷入了 *边缘推理优先的陷阱*。
   - 成员们指出，Apple 专注于端侧限制下可能的用例，而其他所有人都在努力让 AI 尽可能强大，这表明 **Apple** 本应在此领域领先，但却 *搞砸了*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://claude.ai/share/03b29290-bc0e-4425-b3b1-211785321b6e">Claude</a>：与来自 Anthropic 的 AI 助手 Claude 对话</li><li><a href="https://x.com/sama/status/1895203654103351462?t=In8IvAuxVWZFZOkOq392_Q&s=19">Sam Altman (@sama) 的推文</a>：GPT-4.5 准备好了！好消息：对我来说，这是第一个感觉像是在与一个有思想的人交谈的模型。我有好几次坐在椅子上，对它能完成的任务感到惊讶...</li><li><a href="https://fxtwitter.com/yacinemtb/status/1894601593904893984?s=46">kache (@yacineMTB) 的推文</a>：视频模型的 Stable Diffusion 时刻已经到来</li><li><a href="https://x.com/ai_for_success/status/1895037373576290735">AshutoshShrivastava (@ai_for_success) 的推文</a>：传闻 GPT-4.5 拥有 45 googolplex 个参数 😆</li><li><a href="https://www.youtube.com/watch?v=1DtJe7-4aas">Nvidia CEO 黄仁勋：DeepSeek 事件凸显了对 AI 计算能力的巨大需求</a>：Nvidia 首席执行官 Jensen Huang 在 Nvidia 季度报告后加入 CNBC 的 Jon Fortt 进行特别报道。</li><li><a href="https://x.com/karpathy/status/1895213020982472863)">Andrej Karpathy (@karpathy) 的推文</a>：GPT 4.5 + 交互式对比 :) 今天标志着 OpenAI 发布了 GPT4.5。自从 GPT4 发布以来，我已经期待了大约 2 年，因为这次发布提供了一个质的...</li><li><a href="https://github.com/Wan-Video/Wan2.1">GitHub - Wan-Video/Wan2.1: Wan: Open and Advanced Large-Scale Video Generative Models</a>：Wan：开源且先进的大规模视频生成模型 - Wan-Video/Wan2.1
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1344437204272943195)** (4 messages): 

> `AI Voice Commands, Reasoning in AI Models, Text-to-Speech AI, Elevenlabs, Cartesia` 


- **通过语音指令切换 AI 的推理功能**：一位用户询问如何通过语音指令切换 AI 模型的推理功能，目标是除非明确提示 *“use reasoning”*（使用推理），否则 **90% 的情况关闭推理**。
   - 用户询问是否可以通过添加系统提示词（system prompt）来实现，以及是否可以对 **推理过程进行微调（finetune）** 并启用文本转语音（text-to-speech）功能。
- **正在讨论文本转语音 AI 模型**：在另一位用户指出该模型无法直接语音通话后，该用户澄清其意图是计划使用 **Elevenlabs** 或 **Cartesia** 的文本转语音技术来实现语音输出。
   - 该成员引用了 [这段 YouTube 视频](https://www.youtube.com/watch?v=zoBwIi4ZiTA)，作为他们试图通过 AI Assistant 实现的类似功能的演示。



**提及的链接**：<a href="https://www.youtube.com/watch?v=zoBwIi4ZiTA">Deepseek AI Assistant: ALWAYS ON Python AI Agent for Engineers that SHIP</a>：🔥 你的个人 AI Assistant 真的“永远在线”吗？探索由 DeepSeek V3 驱动的 Ada 如何彻底改变工程师交付代码的方式！🚀🎥 资源...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1344584159380901959)** (1 messages): 

> `Language Models, REFUTE benchmark, algorithmic problem solving` 


- **语言模型能否加速科学发展？**：语言模型（LMs）具有通过协助证伪假设和迭代完善论断来加速科学发现的潜力。
   - 目前 LMs 的基准测试主要评估其生成解决方案的能力，而非挑战这些方案的能力。
- **引入用于算法问题求解的 REFUTE 基准测试**：引入了一个名为 **REFUTE** 的动态更新基准测试，用于评估 LMs 在 [算法问题求解](https://huggingface.co/papers/2502.19414) 中为错误方案生成反例的能力。
   - 它包含了编程竞赛中最近出现的题目和错误提交记录，这些错误已被人类专家成功识别出反例。
- **LMs 在 REFUTE 上的验证能力堪忧**：对 **REFUTE** 基准测试的分析显示，即使是最好的推理 Agent，找到反例的成功率也仅为 **9%**。
   - 这表明对于语言模型而言，验证可能比生成要困难得多。



**提及的链接**：<a href="https://huggingface.co/papers/2502.19414">Paper page - Can Language Models Falsify? Evaluating Algorithmic Reasoning with Counterexample Creation</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1344581040706027622)** (3 messages): 

> `Diffusion LLMs, Mercury dLLM, LLaDA Release` 


- **Mercury dLLM 投入商用！**：**Inception Labs** 推出了 **Mercury**，这是一个全新的扩散大语言模型（**dLLMs**）家族，声称其速度比目前经过速度优化的 LLM 快 **10 倍**，在 **NVIDIA H100s** 上可达到超过 **1000 tokens/sec**；代码生成模型 **Mercury Coder** 已可在 [Playground](https://chat.inceptionlabs.ai) 中进行测试。
- **LLaDA 模型发布官方 PyTorch 实现！**：**ML-GSAI** 团队发布了其“大语言扩散模型”（Large Language Diffusion Models）的官方 PyTorch 实现，代码已在 [GitHub](https://github.com/ML-GSAI/LLaDA) 上线。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.inceptionlabs.ai/news">Inception Labs</a>：我们正在利用扩散技术开发新一代 LLM。我们的 dLLM 比传统的自回归 LLM 更快、更高效。而且扩散模型更准确...</li><li><a href="https://github.com/ML-GSAI/LLaDA">GitHub - ML-GSAI/LLaDA: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot;</a>：&quot;Large Language Diffusion Models&quot; 的官方 PyTorch 实现 - ML-GSAI/LLaDA
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1344584159380901959)** (1 条消息): 

> `Language Models, Scientific discovery, REFUTE Benchmark` 


- **语言模型推动科学发现**：人们对 **Language Models (LMs)** 加速**科学发现**的潜力越来越感到兴奋。
   - 目前针对 LMs 的基准测试主要评估其生成解决方案的能力，而不是挑战它们的能力。
- **引入 REFUTE 基准测试**：**REFUTE** 基准测试包含了来自编程竞赛的近期题目和错误提交，在这些案例中，人类专家成功识别出了反例。
   - 分析显示，最优秀的推理 Agent 成功率仅为 **9%**，这表明有时验证比生成要困难得多。



**提及的链接**：<a href="https://huggingface.co/papers/2502.19414">Paper page - Can Language Models Falsify? Evaluating Algorithmic Reasoning with Counterexample Creation</a>：未找到描述

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1344457413314740256)** (132 条消息🔥🔥): 

> `HuggingFace Spaces licensing, Fal AI vs Deepinfra pricing, Lighteval MMLU-Pro support, LEFFA paper implementation, HuggingMod bot` 


- **Spaces 许可证纠纷？**：一位用户询问为社区机器人创建 Space 是否需要“许可证”，随后被澄清为发布代码所需的软件许可证，而非创建 Space 的许可。
   - 另一位用户引导他们查看 [HuggingMod bot](https://huggingface.co/spaces/discord-community/HuggingMod) 以获取代码片段和指导。
- **Deepinfra 在成本上碾压 Fal AI？**：虽然一位用户推荐了提供 *$50* 免费额度的 **Fal AI**，但另一位用户声称 **Deepinfra** 在字符处理方面便宜 *100* 倍（每百万字符 *$0.8*），并提供免费算力。
   - 第一位用户还建议将 **Kokoro TTS** 作为一个廉价选项。
- **Apple Silicon 助力 LLM 取得进展**：一位用户询问在 **Apple's Neural Engine** 上运行 LLM 的情况，另一位用户指向了 **Core ML** 和 [Apple 官方文档](https://machinelearning.apple.com/research/core-ml-on-device-llama)，内容关于为 Apple silicon 优化 LLM。
   - 讨论指出，将模型转换为 *.mlmodel* 扩展名是必要的，但过程可能很复杂。
- **Gemma 量化困惑**：一位用户询问 **GemmaX2** 的大小，另一位用户指向了[此页面](https://huggingface.co/Tonic/GemmaX2-28-2B-gguf/tree/main)，并提到根据量化方式的不同，大小在 *1.5GB* 到 *5.3GB* 之间变化。
   - 该用户还告诉用户如何通过点击 *Use this model* 来查看大小。
- **OpenAI 生成的文本可检测吗？**：用户讨论了 **AI 生成文本检测**，一位用户分享说学术机构可能不会检查，因为缺乏确凿证据。
   - 一位用户分享了 AI 改进前后的求职信图像，指出 **OpenAI 模型在遵循模式方面表现得非常糟糕**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/discord-community/HuggingMod">HuggingMod - a Hugging Face Space by discord-community</a>：未找到描述</li><li><a href="https://tenor.com/view/drake-gif-21355539">Drake GIF - Drake - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/682">huggingchat/chat-ui · New Design Proposal for Hugging Face Chat</a>：未找到描述</li><li><a href="https://machinelearning.apple.com/research/core-ml-on-device-llama">On Device Llama 3.1 with Core ML</a>：许多应用开发者有兴趣构建集成日益强大的大语言模型 (LLMs) 的端侧体验……</li><li><a href="https://huggingface.co/Tonic/GemmaX2-28-2B-gguf/tree/main">Tonic/GemmaX2-28-2B-gguf at main</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/discord-community/HuggingMod/blob/main/app.py">app.py · discord-community/HuggingMod at main</a>：未找到描述</li><li><a href="https://github.com/benchflow-ai/benchflow">GitHub - benchflow-ai/benchflow: AI benchmark runtime framework that allows you to integrate and evaluate AI tasks using Docker-based benchmarks.</a>：AI 基准测试运行框架，允许您使用基于 Docker 的基准测试来集成和评估 AI 任务。- benchflow-ai/benchflow</li><li><a href="https://github.com/huggingface/smolagents/issues">huggingface/smolagents</a>：🤗 smolagents：一个极简的 Agent 库。Agent 编写 Python 代码来调用工具并协调其他 Agent。- huggingface/smolagents</li><li><a href="https://huggingface.co/Tonic/">Tonic (Joseph [open/acc] Pollack)</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1344402717862268969)** (4 messages): 

> `Hiding vs Removing, F2 vs F12, Smol Agents Framework` 


- **隐藏不等于移除！**：一位成员询问了**隐藏 (hiding)**与**移除 (removing)**之间的区别，并质疑隐藏的好处。
   - 在看到对比隐藏与移除的[一对截图](https://cdn.discordapp.com/attachments/898619964095860757/1344402717518594068/SCR-20250226-qcsi.png?ex=67c21999&is=67c0c819&hm=b8bb0a85e03415adc0e67d6f889d0baab302156da75c906e7cc63cbc7c1e6a73&)后，他们似乎感到困惑。
- **F2 与 F12 完全不同**：一位成员分享了他们关于 **F2** 和 **F12** 键区别的 TIL（今天我学到了）时刻。
   - 未提供进一步的上下文。
- **Smol Agents Framework**：一位成员正在学习如何使用 **smol agents framework** 构建基础 Agent。
   - 他们没有分享关于正在构建的 Agent 或其经验的更多细节。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1344400801635893260)** (8 messages🔥): 

> `LLM performance benchmark, Face similarity questionnaire, PyTorch library for 360° images, Phi-4 models` 


- **发布 **LLM Benchmark** 以评估性能**：一位成员开发了一个小型私有基准测试，使用以前未见过的问题快速检查通用 **LLM performance**，并估算小型本地模型与最佳在线模型之间的差距，目前已包含超过 **1000 个模型**。
   - 基准测试和模型评分可在 [MoonRide 的 Hashnode 博客文章](https://moonride.hashnode.dev/biased-test-of-gpt-4-era-llms-300-models-deepseek-r1-included) 和 [HuggingFace](https://huggingface.co/datasets/MoonRide/MoonRide-LLM-Index-v7) 上查看。
- **硕士论文需要**人脸相似度**偏好调查**：一位成员请求参与其硕士论文的问卷调查，该论文重点是通过人脸生成流水线确定哪些**人脸看起来更相似**。
   - 该问卷针对 PC 进行了优化，大约需要 5 分钟完成，可通过[此链接](https://1ka.arnes.si/a/70715279)访问。
- **PyTorch360Convert 库简化 360° 图像处理**：一位成员介绍了一个名为 **pytorch360convert** 的新型轻量级 **PyTorch library**，旨在简化 VR、AR、视频游戏等领域的 **360° images** 处理，可通过 `pip install pytorch360convert` 安装。
   - 该库支持多种图像表示形式，包括 **equirectangular images** 和 **cubemaps**，且 **GPU/CPU compatible**，支持 **float32, float64, float16, 和 bfloat-16 精度类型**，可在 [GitHub](https://github.com/ProGamerGov/pytorch360convert) 上获取。
- **Phi-4 模型在 HF Spaces 亮相**：一位成员分享了 Hugging Face Spaces 上 **phi 4 models** 的链接，标志着该项目的可用性。
   - 该项目可以在 [Hugging Face Spaces](https://huggingface.co/spaces/merterbak/phi-4) 找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/merterbak/phi-4">Phi 4 - merterbak 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://1ka.arnes.si/a/70715279">基于相似度的用户排名 - 1KA | 网络调查</a>：未找到描述</li><li><a href="https://github.com/ProGamerGov/pytorch360convert">GitHub - ProGamerGov/pytorch360convert: 基于 PyTorch 的等距柱状投影、立方体贴图和透视之间的图像转换。基于 py360convert</a>：基于 PyTorch 的等距柱状投影、立方体贴图和透视之间的图像转换。基于 py360convert - ProGamerGov/pytorch360convert</li><li><a href="https://moonride.hashnode.dev/biased-test-of-gpt-4-era-llms-300-models-deepseek-r1-included">GPT-4 时代 LLM 的偏向性测试（包含 300 多个模型，含 DeepSeek-R1）</a>：简介：我不时会尝试一些可以在本地运行的模型（在 16GB VRAM GPU 上），检查它们的对话和推理能力。我不完全信任公开的基准测试，因为...</li><li><a href="https://huggingface.co/datasets/MoonRide/MoonRide-LLM-Index-v7">MoonRide/MoonRide-LLM-Index-v7 · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1344576438346584085)** (2 messages): 

> `Language Models (LMs), REFUTE Benchmark, Reasoning Agents` 


- **Language Models Speeding Scientific Discovery**: 一篇新论文强调了 **Language Models (LMs)** 加速科学发现的潜力，并强调了证伪假设的重要性。
   - 论文指出，目前的基准测试主要评估生成解决方案的能力，而不是挑战它们，因此提倡建立评估逆向能力的基准测试，并链接到了他们的论文 [点击此处](https://huggingface.co/papers/2502.19414)。
- **Introducing the REFUTE Benchmark**: 作者介绍了 **REFUTE**，这是一个动态更新的基准测试，包含来自编程竞赛的近期问题和错误提交，在这些问题中人类专家成功识别出了反例。
   - 分析显示，即使是最好的推理 Agent 在证伪错误的算法解决方案方面的得分也很低（9%），尽管它们能为 50% 的问题生成正确的解决方案。
- **LLMs as Retrieval Engines**: 一位成员评论说，目前缺乏数据证明验证比生成更难，并指出生成正确类型的代码在各处都占据主导地位。
   - 该成员建议 **LLMs** 无法进行太多推理，本质上主要是一个检索引擎。



**Link mentioned**: <a href="https://huggingface.co/papers/2502.19414">Paper page - Can Language Models Falsify? Evaluating Algorithmic Reasoning with
  Counterexample Creation</a>: 未找到描述

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1344707878547882065)** (2 messages): 

> `` 


- **No Topics Discussed**: 提供的消息中未讨论重要主题。
- **Awaiting Next Session**: 一位成员表达了参加下一场活动的意愿。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1344707377781411914)** (1 messages): 

> `FastRTC` 


- **FastRTC Category is LIVE!**: 一位成员引导大家前往 **FastRTC** 类别进行提问、讨论和发布公告。
   - 特定频道的链接在 [这里](https://discord.com/channels/879548962464493619/1344703220332756994)。
- **Reminder to use FastRTC Category**: 为了保持服务器整洁，鼓励成员在进行相关讨论时使用 **FastRTC** 类别。
   - 这有助于确保相关信息易于获取，并使对话保持集中。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1344439778833334283)** (9 messages🔥): 

> `Inference Engine Alternatives, Smolagents Quiz Iframe, Smolagents Quiz Failures, HfApiModel vs LiteLLMModel Confusion, SFT Trainer Loss Function` 


- **Inference Credits Exhausted**: 在超出 Hugging Face 的推理请求限制后，一位用户询问是否有折扣或替代推理引擎，以便在 Google Colab 上继续进行 **smolagents course** 的 studio notebooks。
   - 他们表达了继续跟进课程的愿望。
- **Smolagents Quiz Display Issues**: 用户报告说，smolagents 课程 2.1 单元最终测验中的 **iframe** 太小，导致即使在 **32寸 4k 显示器**上也很难阅读反馈。
   - 他们建议将 iframe 大小增加到 **800x600** 或 **850x850** 以提高可读性。
- **Smolagents Quiz Validation is BSing User**: 用户抱怨 Agent 课程测验 2.1 中验证答案的 Agent 给出了矛盾的反馈，关于 **HfApiModel** 中的 id 参数，先是要求提供，随后又拒绝。
   - 用户认为 **HfApiModel** 类应该默认使用 **Qwen** 模型，使 id 参数成为可选，并要求验证 Agent 具有更多的“思维弹性”。
- **SFTTrainer Loss Elucidation**: 用户寻求关于 **SFTTrainer** 使用的损失函数的澄清，询问它是否是从模型类型中推断出来的（例如 CLM 的 **crossentropy**）。
   - 同时也确认了无论是否显式导入，Agent 的工作方式都是相同的。
- **Documentation Discrepancies Frustrate Quiz Takers**: 用户对第二次测验中遇到的错误表示沮丧，理由是测验的安全设置与当前文档之间存在差异。
   - 用户还指出对 **HfApiModel** 与 **LiteLLMModel** 的模型实现感到困惑，称文档似乎没有表明 **HfApiModel** 具有用于 **LiteLLMModel** 的 model_id。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1344405585461772318)** (129 条消息🔥🔥): 

> `Chat templates, agent, and LLM interaction, NVIDIA AI Red Team Prompt Injection, CodeAgent's Python interpreter, Smolagents codeagents to set the system prompts, Agent Laboratory for research reports and code repositories` 


- **Agent 的 Prompt Template 被填充**：一位成员试图验证他们对 **chat templates**、**agents** 和 **LLMs** 如何交互的理解，并指出 `prompts.yaml` 文件定义了 `system_prompt`，并填充了 Agent 初始化时提供的实际工具。
   - 另一位成员澄清说，**CodeAgent** 实际上拥有自己的 **Python interpreter**。
- **NVIDIA AI Red Team 应对 Prompt Injection**：[NVIDIA AI Red Team](https://developer.nvidia.com/blog/nvidia-ai-red-team-an-introduction/) 发现了漏洞，**prompt injection** 可被用于利用 [LangChain](https://www.langchain.com/) 库中包含的三个插件。
   - **Prompt injection** 是一种专门针对 **large language models (LLMs)** 的新型攻击技术，使攻击者能够操纵 LLM 的输出，尤其是当 LLM 配备了插件时。
- **Smolagents 的调试噩梦**：一位成员报告在 **Unit 2** 的示例中遇到了问题，称大多数示例代码因达到最大步数（maximum number of steps）而失败。
   - 另一位成员分享了关于将 **Smolagents** 部署到生产环境的一些担忧，指出 *“因为它们不支持异步运行，我必须在线程中运行它们”*。
- **Gemini 更加慷慨**：一位成员表示他们遇到了 **Payment Required**（需要付费）的消息。
   - 另一位成员建议切换到配合 **LiteLLM** 使用 **Gemini**，因为 *“Gemini 在 Google AI Studio 中有慷慨的免费层级”*。
- **Agent Laboratory 助力构思**：[Agent Laboratory](https://agentlaboratory.github.io/) 以人类提出的研究想法作为输入，并输出研究报告和代码库。
   - 根据其 GitHub 页面，它使你能够 *“专注于构思和批判性思维，同时自动化重复且耗时的任务，如编码和文档编写”*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/agents-course/unit_1_quiz">Unit 1 Quiz - AI Agent Fundementals - a Hugging Face Space by agents-course</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit2/smolagents/why_use_smolagents">Why use smolagents - Hugging Face Agents Course</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/unit1-certification-app">Unit 1 Certification - AI Agent Fundamentals - a Hugging Face Space by agents-course</a>：未找到描述</li><li><a href="https://huggingface.co/docs/smolagents/main/en/reference/agents#smolagents.PromptTemplates">Agents</a>：未找到描述</li><li><a href="https://agentlaboratory.github.io/">Agent Laboratory: Using LLMs as Research Assistants</a>：作者 Samuel Schmidgall (JHU)</li><li><a href="https://www.youtube.com/watch?v=2ky50XT0Nb0"> - YouTube</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/securing-llm-systems-against-prompt-injection/">Securing LLM Systems Against Prompt Injection | NVIDIA Technical Blog</a>：这篇文章解释了 prompt injection，并展示了 NVIDIA AI Red Team 如何发现可利用 LangChain 库中三个插件的 prompt injection 漏洞。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1344399549812445317)** (264 条消息🔥🔥): 

> `Perplexity Pro Flair, New Voice Mode, Disable Web Search, Coding with Perplexity, Gemini Real Time Video Chat` 


- ****语音模式备受推崇****：成员们讨论了新的 **voice mode** 功能，注意到 **UI** 的改进、**interrupt**（中断）能力以及 **voices**（语音）的变化。
   - 虽然一些用户觉得它令人印象深刻，但其他人认为它尚未达到 **Microsoft Copilot**、**Grok 3** 或 **ChatGPT** 的水平。
- ****无需联网的写作奇迹****：成员们讨论了在 Perplexity 中 **disable web search**（禁用网络搜索）的功能，一位用户建议使用 **writing focus** 模式来实现这一点。
   - 然而，一些用户反映即使在写作模式下，**web sources** 仍被使用，而另一些用户则表示该功能对他们运行良好。
- ****GPT-4.5 传闻四起****：用户讨论了将 **GPT-4.5** 集成到 Perplexity 的可能性，引用了一个 [YouTube demo](https://www.youtube.com/watch?v=cfRYp0nItZ8)，并指出该模型具有 *greater context*（更广的上下文）和 *more human-like*（更像人类）的回答。
   - 一位用户分享了 [Sam Altman 在 X 上的链接](https://x.com/sama/status/1895203654103351462)，提到 **GPT-4.5** 是 *第一个让人感觉像是在与一个有思想的人交谈的模型*。
- ****Spaces 中的模型混淆乱象****：用户讨论了 Spaces 的问题，即即使在使用其他模型时，**system prompt** 仍会提示 *You are Perplexity, a helpful search assistant trained by Perplexity AI*。
   - 一位用户分享了[在 Spaces 中测试此问题的链接](https://www.perplexity.ai/search/what-ai-model-are-you-qsgHi_lOQNq1TSv7UKw2kw)，另一位用户建议在 Space 的指令中明确写出模型身份。
- ****Pro 还是 Grok：一场盛大的辩论****：成员们辩论了 Perplexity Pro 与 SuperGrok 的价值，一位用户询问 *What is the difference between the $50 dollar premium + plan vs Supergrok via there app?*（50 美元的 premium + 计划与通过其 App 使用的 SuperGrok 有什么区别？）。
   - 一位用户澄清说，SuperGrok 通过 Premium+ 中没有的 **Big Brain** 模式提供 *more advanced reasoning*（更高级的推理）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sama/status/1895203654103351462">Sam Altman (@sama) 的推文</a>：GPT-4.5 准备就绪！好消息：对我来说，它是第一个让人感觉像是在与一个有思想的人交谈的模型。我有好几次坐在椅子上，对获得的交流感到惊讶...</li><li><a href="https://en.wikipedia.org/wiki/I_know_that_I_know_nothing">我只知道我一无所知 - 维基百科</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8">GPT-4.5 介绍</a>：Mia Glaese, Rapha Gontijo Lopes, Youlong Cheng, Jason Teplitz 和 Alex Paino 介绍并演示 GPT-4.5。</li><li><a href="https://youtu.be/RI-BxtCx32s?si=Mk2TDhRQ3YrjRl8n">GPT-4o 视觉能力现场演示</a>：这是我们 OpenAI 春季更新活动的现场演示。了解更多关于 GPT-4o 的信息：https://www.openai.com/index/hello-gpt-4o/
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1344465562218004500)** (17 条消息🔥): 

> `Majorana-1 Quantum, AI Communication, Lab Mice First Aid, House Blueprint, Ransomware Leaks` 


- **Perplexity 用户分享大量链接**：多位用户分享了一系列 **Perplexity AI** 搜索和页面链接，涵盖从 [quantum computing](https://www.perplexity.ai/search/majorana-1-the-worlds-first-qu-GfQ6ey8KRHKJoZXASTx94w)（量子计算）到 [AI communication](https://www.perplexity.ai/search/i-heard-about-two-ais-communic-2NNO3p7QQdac1IJ0TDAmjA)（AI 通信）以及 [lab mice giving first aid](https://www.perplexity.ai/page/lab-mice-give-first-aid-Cr8kRPgoTLSBbXUWtl48AQ)（实验室老鼠进行急救）等主题。
   - 这些链接还包括一段 [YouTube 视频](https://www.youtube.com/embed/gdiYF-UQ2K8) 以及关于 [房屋建造](https://www.perplexity.ai/search/i-need-to-build-a-house-to-rep-OQoLSIjESviUYqwhCnA0uw)、勒索软件泄露和 AI 驱动诊断的讨论。
- **Perplexity AI 上讨论 Nvidia 股票**：用户分享了关于 [Nvidia's strong results](https://www.perplexity.ai/page/nvidia-s-strong-results-impact-bMt3pD7NTH2Tlk8QcsIo2Q)（英伟达强劲业绩）对市场影响的链接。
   - 还有关于讨论 *Z-a 交易策略* 的公开邀请。
- **深海讨论深度探索**：一个分享链接指向了 **Perplexity AI** 上关于 [deep sea](https://www.perplexity.ai/search/deep-sea-k-kxb5gKq4RyeIOuGppGtSKQ#0)（深海）的讨论。
- **SchellingPoint 被贴上“毒井”标签**：一位用户提到了 `$SchellingPointZEC` 和 `POISONED WELL`，并附上了关于 [data centers and their health costs](https://www.perplexity.ai/page/data-centers-health-costs-43FGGYpDQV2U4NiA.8pBuQ)（数据中心及其健康成本）的文章链接。



**提到的链接**：<a href="https://www.youtube.com/embed/gdiYF-UQ2K8">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1344564918183268385)** (4 messages): 

> `Perplexity Pro API credits, Obsidian Web Clipper configuration, sonar-deep-research model, Refunds for Perplexity API` 


- **Perplexity Pro 额度：我可以调用多少次 API？**：一位用户询问了 **Perplexity Pro** 包含的 **$5 API 额度**可以进行多少次 API 调用和搜索，以及如果超过给定额度该如何支付。
- **在 Obsidian Web Clipper 中配置 Perplexity API 遇到困难**：尽管设置了正确的 **Base URL** 和 **API Key**，一位用户在 **Obsidian Web Clipper** 中配置使用 `sonar-deep-research` 模型的 **Perplexity API** 时仍遇到问题。
   - 该用户提供了其配置和失败消息的[截图](https://cdn.discordapp.com/attachments/1161802929053909012/1344638496190627922/Image_27-2-25_at_12.42_PM.jpeg?ex=67c24c6f&is=67c0faef&hm=8e87be021f18ebec8872bb67c9635f61d713e54264e2613c300ca3564492218d&)，寻求故障排除方面的帮助。
- **Perplexity API 退款流程咨询**：一位用户询问如果**误充值了 API** 且未使用，该如何获得**退款**。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1344733055432134707)** (1 messages): 

> `Website Redesign Contest, Stable Diffusion 3.5, AI-generated artwork, US participants only` 


- **Stability AI 启动网站重设计竞赛**：Stability AI 邀请 **Stable Diffusion** 社区在**网站重设计竞赛**中展示他们的最佳作品，获奖图像将在 **Stability AI 官方网站**上展示。
   - 该竞赛寻求感觉*清新、令人印象深刻且具有前瞻性*的图像，作品需使用 **Stable Diffusion 3.5** 创作，并传达*创新、美感和创造力的未来*。
- **参赛作品需以 Stable Diffusion 3.5 Base 为基础**：要参加网站重设计竞赛，艺术作品必须以 **Stable Diffusion 3.5** 为基础创作，但可以包含**自定义节点、fine-tunes 或 LoRAs**。
   - 指南明确禁止**侵犯知识产权的内容、机器人或末日主题以及 NSFW 材料**。
- **仅限美国参与者参加 Stability AI 竞赛**：网站重设计竞赛仅对**美国参与者**开放，提交的作品需为 **16:9 纵横比**。
   - 投稿截止日期为 **3 月 7 日星期五**，入选作品将在 Stability AI 的平台上获得**认可和社区展示**。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1344435747901669417)** (92 messages🔥🔥): 

> `ControlNet models for consistent characters, LLMs referencing real-time data, SDXL alternative with T5 CLIP, Inpaint Anything error, Selling ComfyUI workflows` 


- **寻求 ControlNet 角色一致性**：一位成员请求推荐在 **SDXL** 中保持角色一致性的最佳 **ControlNet 模型**。
   - 如果有可用的参考 **U-Net 模型**，他们也特别提出了请求。
- **Gemini 实时数据访问？**：一位成员询问了可以引用并随**实时数据**更新的 **LLMs**，并提到 **Gemini** 作为一个潜在选项。
   - 另一位成员指出大多数 LLM 不会实时更新，但建议启用网页搜索以获取更相关的信息。
- **T5 CLIP 热潮**：一位成员寻找集成 **T5 CLIP** 的类 **SDXL 模型**，表示他们已经体验过 **SD3.5** 中 **T5 提示词遵循能力**。
   - 他们发现 **T5 的遵循能力**令人上瘾，并正在寻找替代方案。
- **出现 "Inpaint Anything" 形状不匹配错误！**：一位成员报告了 **Inpaint Anything** 中的形状不匹配错误：*value tensor of shape [159, 256] cannot be broadcast to indexing result of shape [64, 256]*。
   - 该成员正在使用带有 Inpaint Anything 扩展插件的 **Automatic1111**，并询问如何解决此错误。
- **出售 ComfyUI 远程安装服务**：一位成员提到出售 **ComfyUI 工作流**和远程安装服务以帮助用户运行，通常使用 **TeamViewer**。
   - 他们澄清说，他们收取的是时间和知识的费用，而不是工作流本身的费用。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1344421450064593038)** (8 messages🔥): 

> `Hugging Face 弃用、最佳 RAG 工具、LLM 预训练指南` 


- **HF 弃用发现**：一位成员询问如何在 **Hugging Face** 上将仓库标记为已弃用并链接到新版本，但随后发现该功能仅适用于 **models**，不适用于 datasets。
- **个人使用的 RAG 工具推荐**：一位成员询问 *目前最适合个人用户的 RAG 工具是什么*？
   - 另一位成员推荐了 **BM25**。
- **需要全方位的 LLM 训练指南**：有人询问是否存在 *一份关于预训练和后训练（包括 SFT 和 RL）的单一自包含 LLM 指南*。
- **LLM Prompt 相关性优于 RAG？**：一位成员建议，对于小规模语料库，通过 Prompt 让 **LLM** 检查相关性比调整 embeddings 和 rerankers 更好。
   - 他们补充说，如果你 *不介意一定的延迟*，使用 Prompt 比调整 embeddings 更好。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1344408590714146918)** (36 messages🔥): 

> `数据混合、DualPipe、DeepSeek、Gemini Flash Thinking、SWE-RL` 


- **梯度下降混合数据，最小化计算量**：一篇新 [论文](https://arxiv.org/abs/2502.10510) 介绍了 **MixMin**，这是一种基于梯度的优化数据混合的方法，它以不到 **0.2%** 的额外计算量改进了混合效果。
   - 该方法通过将其形式化为凸双层目标，解决了为机器学习流水线寻找最佳数据混合的挑战。
- **DeepSeek 发布用于训练的 DualPipe**：**DeepSeek** 发布了 [DualPipe](https://github.com/deepseek-ai/DualPipe)，这是一种双向流水线并行（pipeline parallelism）算法，旨在重叠 V3/R1 训练中的计算与通信。
   - 一位用户表示希望 DeepSeek 能在最后一天发布其整个预训练框架，包括核心部分。
- **Gemini 的 Flash Thinking 引发讨论**：成员们讨论了 [Gemini 2.0 Flash Thinking](https://deepmind.google/technologies/gemini/flash-thinking/)，这是 Google 增强的推理模型，它通过 *展示思考过程* 来提高性能和可解释性，特别是在数学和科学领域。
   - 一些人怀疑该模型进行了内部基准测试，但由于表现不如 **O3 Mini** 而未公开发布。
- **通过 SWE-RL 扩展软件工程的 LLM 推理能力**：一篇 [论文](https://arxiv.org/abs/2502.18449) 介绍了 **SWE-RL**，它利用轻量级基于规则的奖励，为现实世界的软件工程扩展了基于 RL 的 LLM 推理。
   - 这种方法使 LLM 能够从开源软件演进数据中自主恢复开发者的推理过程，并在 **Llama 3** 之上进行训练。
- **用于 ResNet 训练的 SSL 方法**：一位用户询问有哪些廉价的 SSL 方法可以快速训练 **ResNet**，从而在 **CIFAR10** 上获得不错的线性探测（linear probe）性能。
   - 另一位用户建议，调整超参数/架构可能比更改损失函数更有效，因为目前可能没有比 **DINO** 显著更高效的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.18779">arXiv reCAPTCHA</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2502.10510">MixMin: Finding Data Mixtures via Convex Minimization</a>: 现代机器学习流水线越来越多地组合和混合来自不同来源的数据，例如预训练大语言模型。然而，寻找最佳数据混合是一个挑战...</li><li><a href="https://arxiv.org/abs/2502.19187">BIG-Bench Extra Hard</a>: 大语言模型 (LLMs) 越来越多地应用于日常场景，要求具备鲁棒的通用推理能力和多样化的推理技能。然而，目前的 LLM 推理基准测试...</li><li><a href="https://deepmind.google/technologies/gemini/flash-thinking/">Gemini 2.0 Flash Thinking</a>: Gemini 2.0 Flash Thinking 是我们的增强推理模型，能够展示其思考过程以提高性能和可解释性。</li><li><a href="https://github.com/deepseek-ai/DualPipe">GitHub - deepseek-ai/DualPipe: A bidirectional pipeline parallelism algorithm for computation-communication overlap in V3/R1 training.</a>: 一种用于 V3/R1 训练中计算与通信重叠的双向流水线并行算法。- deepseek-ai/DualPipe</li><li><a href="https://arxiv.org/abs/2502.18449">SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</a>: 最近发布的 DeepSeek-R1 展示了强化学习 (RL) 在增强大语言模型 (LLMs) 通用推理能力方面的巨大潜力。虽然 DeepSeek-R1 ...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1344459755451715756)** (22 messages🔥): 

> `Jacobian Sparse Autoencoders, SmolLM2 Intermediate Checkpoints, Mechanistic Interpretability Resources, Saving Weights after Iteration, Open Problems in Mechanistic Interpretability` 


- **Jacobian Sparse Autoencoders 使计算稀疏化**：一篇新论文介绍了 **Jacobian Sparse Autoencoders (JSAEs)**，这是一种旨在诱导 LLM 内部计算和表示稀疏化的新型架构，目标是在输入的完整分布上构建稀疏计算图。阅读[完整论文请点击这里](https://arxiv.org/abs/2502.18147)。
- **SmolLM2 模型获得 50 多个 Checkpoints**：发布了所有 **SmolLM2** 模型的 50 多个中间 Checkpoints，希望能帮助人们学习 Interpretability。查看公告[请点击这里](https://x.com/eliebakouch/status/1895136704077463768)。
- **Neel Nanda 的全面 MI 资源清单**：一位用户分享了一系列学习 Mechanistic Interpretability 的资源，主要链接到 **Neel Nanda** 创建的内容，包括[“入门指南”](https://www.neelnanda.io/mechanistic-interpretability/getting-started)和一份进入该领域时值得阅读的[优秀论文列表](https://www.neelnanda.io/mechanistic-interpretability)。
   - 此外还分享了 Neel Nanda 更新的（2024年）最爱论文列表，可以[在这里找到](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite)。
- **寻求迭代后保存权重的解决方案**：一位用户询问了关于在预训练期间每次迭代后高效保存权重以观察细粒度动态的研究或工具，并链接到了 [GitHub](https://github.com/manncodes/interp-infra/blob/master/weight-trace.ipynb) 上的一个初始 MVP。
- **Mech Interp 小组发布调查报告**：分享了一篇代表许多主要 Mech Interp 小组的大型综述论文，题为[《Mechanistic Interpretability 中的开放性问题》](https://arxiv.org/abs/2501.16496)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.16496)">arXiv reCAPTCHA</a>: 未找到描述</li><li><a href="https://x.com/eliebakouch/status/1895136704077463768">来自 elie (@eliebakouch) 的推文</a>: 出发！我们刚刚发布了所有 SmolLM2 模型的 50 多个中间 Checkpoints 🔥</li><li><a href="https://www.neelnanda.io/mechanistic-interpretability">Mechanistic Interpretability &mdash; Neel Nanda</a>: 关于 Mechanistic Interpretability 研究的博客文章</li><li><a href="https://www.neelnanda.io/mechanistic-interpretability/getting-started">开始 Transformer Mechanistic Interpretability 的具体步骤 &mdash; Neel Nanda</a>: 免责声明：这篇文章主要链接了我制作的资源。我感到有些抱歉！Transformer MI 是一个非常年轻且细小的领域，目前还没有多少人在做教育工作...</li><li><a href="https://github.com/manncodes/interp-infra/blob/master/weight-trace.ipynb">interp-infra/weight-trace.ipynb at master · manncodes/interp-infra</a>: 通过在 GitHub 上创建一个账户来为 manncodes/interp-infra 的开发做出贡献。</li><li><a href="https://www.lesswrong.com/posts/FrekePKc7ccQNEkgT/paper-jacobian-sparse-autoencoders-sparsify-computations-not">[论文] Jacobian Sparse Autoencoders: 使计算稀疏化，而不仅仅是激活 — LessWrong</a>: 我们刚刚发表了一篇论文，旨在发现“计算稀疏性”，而不仅仅是表示中的稀疏性。在文中，我们提出了一种新的架构……</li><li><a href="https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite">一份极其主观的、带有注释的我最喜欢的 Mechanistic Interpretability 论文列表 v2 — AI Alignment Forum</a>: 这篇文章代表我个人的观点，不代表我的团队或雇主的意见。这是我对两年前制作的类似列表的大规模更新版本……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1344477149750497341)** (17 messages🔥): 

> `QA Task Evaluation, ARC-Easy, ARC-hard, Mosaic's Eval Framework, GPQA Diamond COT Zero-Shot Evaluation` 


- **使用 Harness 评估 QA 任务引发辩论**：一位成员询问如何使用 harness 评估 **ARC-Easy** 和 **ARC-hard** 等 **QA 任务**，并质疑为什么拼接内容仅包含 *问题 + 选项*，而不是针对每个选项包含 *问题 + 选项 + 答案*。
   - 另一位成员指出了 [Mosaic 的评估框架](https://arxiv.org/pdf/2404.08382) 和 [第 5.2 节](https://arxiv.org/pdf/2405.14782)，以提供任务结构和评估方法的背景信息。
- **ARC 评估依赖 Loglikelihoods**：在回答关于评估方法的问题时，一位成员澄清说，他们发现 **ARC-Challenge** 和 **ARC-Easy** 遵循前一种方法（问题 + 选项），并且可以使用 *generate_until* 代替 *Loglikelihoods*，然后进行精确匹配。
   - 另一位成员确认这种方法与 GPT-3 论文一致。
- **分享 GPQA Diamond COT Zero-Shot 命令**：一位成员询问用于运行评估的命令，并提到有人报告准确率低于 10%。
   - 另一位成员分享了 `thinktest` 分支上的 `gpqa_diamond_cot_zeroshot` 命令，以及特定的模型参数和并行化参数，并引用了 [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_challenge_chat.yaml)。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_challenge_chat.yaml">lm-evaluation-harness/lm_eval/tasks/arc/arc_challenge_chat.yaml at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1344398918708101232)** (58 messages🔥🔥): 

> `微软的生存得益于政府支持，AI 模型的确定性表现，编程中的 AI，Agent 系统面临挑战，小团队构建了比 Chrome 更好的浏览器` 


- **关于 Microsoft 主导地位的辩论**：一名成员断言 **Microsoft** 从未是真正的创新者，而是靠政府的支持维持地位。
   - 另一名成员反驳称，虽然金钱和权力很重要，但它们不能保证长期成功，并以 **Yahoo** 为例，指出该公司尽管拥有大量资源，但最终失去了主导地位。
- **AI 模型生成有意义但非确定性的结果**：一名成员询问非确定性的 AI 模型如何能表现出确定性行为并趋于收敛。
   - 另一名成员回答说，虽然确切的结果可能有所不同，但 AI 模型生成的输出具有相同的含义，并举了 **Cursor** 中重新生成的代码为例，其中仅更改了注释和变量名。
- **AI 在静态编程任务中表现出色**：一位成员分享说，AI 模型比其他任务更容易学习编程，专注于编程方面，精通静态事物，但在动态任务中表现挣扎，这损害了 Agent 系统。
   - 他们指出个人威胁大公司的可能性，因为较小的团队可以移动得更快并构建更好的工具。
- **OpenAI 发布 GPT-4.5 研究预览版**：成员们讨论了 **GPT-4.5** 的发布，指出它更多地关注用户偏好和有用性，而不是 [GPT-4.5 介绍视频](https://www.youtube.com/watch?v=cfRYp0nItZ8)中所描述的突破性进展。
   - 一些人认为 **OpenAI** 是由于来自 **Grok-3** 和 **Claude 3.7** 的竞争压力才发布了某些东西，并注意到价格上涨至每百万输入 token **$75**，输出 token **$150**。
- **OpenAI 的 MoE 架构得到确认**：一名成员分享了一个或多或少的官方确认，即 **OpenAI** 的基础模型都是 **MoE** (Mixture of Experts)，如该 [YouTube 视频](https://youtu.be/pdfI9MuxWq8?si=d_x-6xvuLZ9ZybZ8&t=685)链接所示。
   - 该成员表示，虽然这算不上新闻，因为这在某种程度上已经是众所周知的，但这次确认并非传闻，而是非常有根据的。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1895134318835704245">来自 OpenAI (@OpenAI) 的推文</a>：4.5 小时后直播。</li><li><a href="https://fxtwitter.com/polynoamial/status/1895207166799401178">来自 Noam Brown (@polynoamial) 的推文</a>：扩展预训练和扩展思考是两个不同维度的改进。它们是互补的，而不是竞争关系。</li><li><a href="https://www.youtube.com/watch?v=cfRYp0nItZ8">GPT-4.5 介绍</a>：Mia Glaese, Rapha Gontijo Lopes, Youlong Cheng, Jason Teplitz, 和 Alex Paino 介绍并演示 GPT-4.5。</li><li><a href="https://www.reddit.com/r/singularity/comments/1izmg33/figure_launching_robots_into_the_home_alpha/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://youtu.be/pdfI9MuxWq8?si=d_x-6xvuLZ9ZybZ8&t=685"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=2ky50XT0Nb0">ChatGPT 开设研究实验室……只需 2 美元！</a>：❤️ 在此处查看 Lambda 并注册其 GPU Cloud：https://lambdalabs.com/papers 在 Lambda 上使用 DeepSeek 的指南：https://docs.lambdalabs.com/educati...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1344503565980270652)** (7 messages): 

> `哈希冲突，KV 相似度` 


- **意在利用哈希冲突**：该实现并非为了消除 **哈希冲突 (hash collisions)**，而是旨在当 qkT_i 较高时*诱导冲突*。
   - 利用了哈希冲突的概率 P(h(q) == h(k_i))，其中 *h* 是哈希函数。
- **通过哈希冲突实现 KV 相似度**：哈希冲突被用作移除相似键值对 (KV pairs) 的指标，如 [arxiv.org/pdf/2502.03387](https://arxiv.org/pdf/2502.03387) 中所述。
   - 讨论中提到了一个 **pseudo truthmatteo.batelic** 文件，但未指明其确切用途。



**提及的链接**：<a href="https://www.twitch.tv/claudeplayspokemon">ClaudePlaysPokemon - Twitch</a>：Claude 玩宝可梦 - 首播

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1344416952914940005)** (15 messages🔥): 

> `Remarkable Alexa, GPT-4.5 发布, DeepSeek AI Open Infra Index` 


- **亚马逊 Alexa 将推出月度订阅服务？**：据 [tomsguide.com](https://www.tomsguide.com/ai/remarkable-alexa-with-ai-could-cost-dollar5-to-dollar10-a-month-heres-what-it-could-do) 报道，传闻代号为 **Remarkable** 的新版 Alexa 可能需要每月 **5 到 10 美元** 的订阅费。
   - 文章强调，鉴于 **Google、Samsung 和 Apple 均免费提供其 AI 服务**，消费者是否愿意为 Alexa 付费仍有待观察。
- **DeepSeek AI 开源基础设施索引 (Infrastructure Index)**：DeepSeek AI 发布了一个开源的基础设施索引，可以在 [这里](https://github.com/deepseek-ai/open-infra-index) 找到。
- **OpenAI 预告 GPT-4.5 发布**：OpenAI 通过 [直播](https://x.com/OpenAI/status/1895134318835704245) 预告了 **GPT-4.5** 的发布，随后发布了一段 [介绍视频](https://www.youtube.com/live/cfRYp0nItZ8)，参与者包括 Mia Glaese、Rapha Gontijo Lopes、Youlong Cheng、Jason Teplitz 和 Alex Paino。
   - 此次公告反响不一，一些人批评了演示方式和展示的场景，例如“因为我生朋友的气，所以写一段愤怒的文字”。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenAI/status/1895134318835704245">来自 OpenAI (@OpenAI) 的推文</a>：4.5 小时后开始直播。</li><li><a href="https://www.tomsguide.com/home/live/amazon-alexa-event-live-last-minute-amazon-devices-rumors-and-all-the-big-news-as-it-happens">Amazon Alexa Plus 活动 —— 所有重大公告和新 AI 功能</a>：新版 Alexa 来了</li><li><a href="https://www.youtube.com/live/cfRYp0nItZ8">GPT-4.5 介绍</a>：Mia Glaese、Rapha Gontijo Lopes、Youlong Cheng、Jason Teplitz 和 Alex Paino 介绍并演示 GPT-4.5。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1344422166220902450)** (44 messages🔥): 

> `OpenAI SDK 中的 Cohere 模型, 自动字幕, Command R+ 更新, R7B Arabic 对比 Fanar 和 ALLaM` 


- **Cohere 模型现已支持使用 OpenAI SDK**：成员们庆祝可以直接通过 **OpenAI SDK** 访问 **Cohere 模型**。分享了 [快速入门指南](https://docs.cohere.com/docs/compatibility-api) 的链接，其中包含 **Python、TS 和 cURL** 的演示，以及 streaming、tool calls 和 structured outputs。
- **社区寻求自动字幕解决方案**：一位用户请求推荐能够生成类似于 **TikTok** 或 **YouTube Shorts** 自动字幕的 AI API。
   - 另一位用户建议使用 **Google STT**，并指出 YouTube 的自动字幕很可能是由 **Google** 自己的工具驱动的。
- **对 Command R+ 更新的期待升温**：社区成员讨论并表达了对即将到来的 **Command R+** 更新的渴望，有人希望它能超越 **Mistral Large 2411**。
   - 成员们强调，由于 **NDA**（保密协议），具体的发布细节不太可能被提前分享，并建议不要传播未经证实的信息或谣言。
- **阿拉伯语 LLM 基准测试**：人们对将 **Cohere 的 R7B Arabic** 模型与 **卡塔尔的 Fanar 模型** 以及 **沙特的 ALLaM** 进行基准测试表现出兴趣，并建议使用 Arabic Balsam 索引。
   - 一位成员还分享了 [GPT-4.5 system card](https://cdn.openai.com/gpt-4-5-system-card.pdf) 的链接，该文档对最新的基准测试方法论进行了很好的概述。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIcbR7vcCi-g">来自 Sandra Kublik (@itsSandraKublik) 的推文</a>：你现在可以直接通过 OpenAI SDK 访问 Cohere 模型了 :) 查看我们的快速入门指南，了解 Python、TS 和 cURL 的演示，以及 streaming、tool calls、structured outputs 等。祝开发愉快...</li><li><a href="https://x.com/itssandrakublik/status/1894791769117650998?s=46&t=r1mNPSgnb3pIc">来自 Sandra Kublik (@itsSandraKublik) 的推文</a>：你现在可以直接通过 OpenAI SDK 访问 Cohere 模型了 :) 查看我们的快速入门指南，了解 Python、TS 和 cURL 的演示，以及 streaming、tool calls、structured outputs 等。祝开发愉快...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1344756676858875938)** (1 条消息): 

> `Command R7B Arabic Model, Multilingual AI Model, Arabic Language Optimization` 


- **阿拉伯语 Command R7B 模型上线！**：Cohere 宣布推出 **Command R7B Arabic**，这是 **R7B 模型** 的一个变体，在保持其 **English** 性能的同时，针对 **Arabic** 性能进行了优化。
   - 该模型现在可通过 *command-r7b-arabic-02-2025* 在 [Cohere Platform](https://dashboard.cohere.com/playground/chat) 和 [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r7b-arabic-02-2025) 上获取，并将于今日晚些时候登陆 **Ollama**。
- **R7B Arabic 擅长企业任务**：**Command R7B Arabic 模型** 在*指令遵循（instruction following）、长度控制、RAG 以及使用正确的语言回答*等任务中表现出色。
   - 它具有 **128,000 tokens** 的上下文长度。
- **阿拉伯语语言模型博客文章发布**：介绍 **Command R7B Arabic** 的博客文章现已上线，详细介绍了其针对 **Arabic 语言能力** 的优化，以支持 **MENA 地区** 的企业。
   - 更多信息请参阅 [发布说明](https://docs.cohere.com/v2/changelog/command-r7b-arabic)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/CohereForAI/c4ai-command-r7b-arabic-02-2025">CohereForAI/c4ai-command-r7b-arabic-02-2025 · Hugging Face</a>：未找到描述</li><li><a href="https://cohere.com/blog/command-r7b-arabic">Introducing Command R7B Arabic</a>：我们最先进的轻量级多语言 AI 模型已针对高级阿拉伯语能力进行了优化，以支持 MENA 地区的企业。</li><li><a href="https://docs.cohere.com/v2/changelog/command-r7b-arabic">Cohere Releases Arabic-Optimized Command Model! — Cohere</a>：Command R7B Arabic 模型的发布公告
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1344514004164542557)** (3 条消息): 

> `Differential Transformers, World Without Coffee Essays` 


- **用户询问 Differential Transformer 概念**：一位用户询问 Bot，**Differential Transformers** 背后的核心概念是什么。
   - 目前没有关于 **Differential Transformers** 的进一步讨论或细节。
- **咖啡文章提示词触发 Bot**：一位用户要求 Bot 写一篇关于*没有咖啡的世界*的文章。
   - 另一位用户重复了这一提示词，表明了对 Bot 如何响应假设场景的兴趣。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1344592099613343836)** (9 条消息🔥): 

> `Free auto caption APIs, Adobe Premiere auto transcription` 


- **成员寻找免费的自动字幕 API**：一位成员询问是否有用于生成自动字幕的免费 API，并想知道是否需要自己构建一个。
   - 另一位成员解释了一个链接工具可以为视频生成*自动字幕/说明*。
- **Adobe Premiere：自动转录功能揭秘**：一位成员提到 **Adobe Premiere** 具有自动转录功能。
   - 其他成员表示赞同，并确认了该功能的存在和可用性。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1344410350283657246)** (2 条消息): 

> `LlamaIndex CentralReach, LlamaExtract Public Beta` 


- **LlamaIndex 助力自闭症和 IDD 护理变革**：[LlamaIndex 正在帮助 CentralReach](https://t.co/Y9Snu1KRho) 利用 AI 变革自闭症和 IDD（智力与发育障碍）护理。
   - AI 在医疗领域的效用在于*将海量的研究和文书工作提炼为相关的见解和关键点*，从而提高医生的效率。
- **LlamaExtract 进入公测阶段**：LlamaIndex 的 [LlamaExtract](https://twitter.com/llama_index/status/1895164615010722233) 现已进入公测阶段，简化了从非结构化文档中提取结构化数据的过程。
   - 它允许用户通过编程方式**定义和自定义数据提取的 Schema**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1344472071329157193)** (48 条消息🔥): 

> `LlamaParse 0.6.2 数据泄露, 重新加载 pgvector 索引表, AgentWorkflow 自定义异常处理, Elasticsearch 元数据架构, LlamaExtract 文档过时` 


- ****LlamaParse 0.6.2 数据泄露事件曝光！****：有用户报告了 **LlamaParse 0.6.2** 中的严重数据泄露，观察到其他用户的图像和分析结果混入了自己的结果中，包括**银行账户详情**和**交易记录**等敏感信息。
   - 该问题已确认为测试/基准测试数据的混淆，并在后端 API 中修复，报告者提供了一份 [Job IDs](https://example.com/jobids) 列表供调查。
- ****pgvector 索引重新加载：索引重现****：用户询问如何从数据库中重新加载之前创建的 **pgvector 索引表**，以避免重复创建。
   - 另一位用户建议使用 `index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)` 从 vector store 中重新加载索引。
- ****AgentWorkflow 的自定义异常难题****：用户询问是否可以允许 **AgentWorkflow** 抛出自定义异常，试图中断工作流并在工具范围之外处理该异常。
   - 虽然目前不支持，但一名成员建议团队可以为 **FunctionTool** 添加一个选项来支持此用例。
- ****LlamaExtract 文档：迷失在云端****：用户发现 **LlamaExtract 0.0.4** 中缺少 `create_agents` 方法，表明文档已过时。
   - 已确认该项目已迁移至 [LlamaCloud Services](https://github.com/run-llama/llama_cloud_services)，相关代码现在位于 *llama_cloud_services* 仓库中，文档确实已经过时。
- ****Searxng 搜索引擎：新面孔？****：用户询问如何将 **Searxng**（一个免费的元搜索引擎）集成到框架中。
   - 一名成员回应称这是他们*第一次听说它*，但建议通过将其放入 FunctionTool 中来配合 Agent 使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ZCG36eLVaaZGA0XIjJH1M5EN8QhygkCC?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_extract">GitHub - run-llama/llama_extract</a>：通过在 GitHub 上创建账户来为 run-llama/llama_extract 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_cloud_services/blob/main/extract.md">llama_cloud_services/extract.md at main · run-llama/llama_cloud_services</a>：云端知识 Agent 与管理。通过在 GitHub 上创建账户来为 run-llama/llama_cloud_services 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_extract?tab=readme-ov-file#%EF%B8%8F-this-project-has-been-moved-to-llamacloud-services">GitHub - run-llama/llama_extract</a>：通过在 GitHub 上创建账户来为 run-llama/llama_extract 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/vector_stores/llama-index-vector-stores-elasticsearch/llama_index/vector_stores/elasticsearch/base.py">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-elasticsearch/llama_index/vector_stores/elasticsearch/base.py at main · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1344612751036907552)** (1 条消息): 

> `Prompt Engineering Studio, AI-powered assistant, Reusable templates, Version control, Team collaboration` 


- **Portkey AI 发布 Prompt Engineering Studio**：Portkey AI 推出了 **Prompt Engineering Studio**，这是一个面向 Prompt 工程师的 IDE，允许用户在 **1600+ 个模型**上进行侧边栏对比测试，并提供来自 **AI-powered assistant** 的即时改进建议。
   - 该 Studio 支持使用 mustache 和 partials 创建**可重用模板**，通过适当的标签进行 Prompt 的**版本管理和部署**，并利用实时分析进行**性能跟踪**。
- **Portkey Workshop 将演示新 Studio**：Portkey AI 将于 **PST 时间 3 月 3 日星期一上午 10:30** 举办一场直播研讨会，演示其 Prompt Engineering Studio，并与首席执行官 Rohit 进行 AMA 互动，可通过 [Portkey 官网](https://portkey.sh/promptworkshop) 访问。
   - 研讨会将展示如何测试 Prompt、使用 AI 助手、构建可重用模板、实现版本控制，以及如何利用共享 Prompt 库进行团队协作。



**提到的链接**: <a href="https://portkey.sh/promptworkshop">Demo: Prompt Engineering Studio · Zoom · Luma</a>：加入我们，抢先体验 Portkey 的 Prompt Engineering Studio —— 这是用于构建、测试和部署 AI Prompt 的最全面工具包……

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1344406196731514880)** (37 条消息🔥): 

> `ReAct Agent Integration, DSPy Release Bug, MIPROv2 Optimizer Error, Refine API Feedback, Community Engagement` 


- **ReAct Agent 处理外部工具**：一位用户询问如何将需要外部响应的工具与 **dspy.ReAct** 集成，以完成创建文本和发送电子邮件等复杂任务，特别是关于编排（orchestration）的问题。
   - 挑战在于当电子邮件功能需要外部函数调用时，如何确保系统理解操作顺序（先创建文本，后发送邮件）。
- **DSPy 2.6.7 版本出现 Bug，导入失效**：用户报告了 **dspy-ai==2.6.7** 中的 **ModuleNotFoundError**，[GitHub issue](https://github.com/stanfordnlp/dspy/issues/7867) 详细说明了导入失败的情况。
   - 降级到 **2.6.6** 版本可解决此问题；该故障版本已被迅速撤回，并发布了 **2.6.8** 版本，以解决从 setup.py 迁移到 pyproject.toml 导致的导入问题。
- **MIPROv2 优化器触及上下文限制**：一位用户在使用 **MIPROv2** 时遇到了 **ContextWindowExceededError**，即使已确保对话内容少于 1000 个字符并使用了 *light* 模式。
   - 建议用户减少优化器中的 demo 数量，或在 `.compile()` 调用中设置 `view_data_batch_size=3` 以解决 Token 限制问题，此设置对于减小数据摘要大小是必需的。
- **Refine API 演进反馈循环**：一位用户询问如何控制在 **dspy.Refine** 的后续重试中传递给 LLM 的建议/反馈，并与旧的断言（assertion）方法进行了对比。
   - 反馈将在 `reward_fn` 中返回，且 `dspy.Refine` 现在应参与编译反馈机制，从而允许对以前无法优化的建议进行优化。
- **社区渴望从杂音中获取信号**：用户对如何从庞大的 Discord 社区获取高质量反馈以改进 DSPy 并避免“过多的调节参数（too many knobs）”表示关注。
   - 提出了每周举行公开会议的建议，以及发布短贴或 PR 以提供生产环境使用反馈的想法，类似于 Discord 频道中的示例。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets/rtatman/ubuntu-dialogue-corpus">Ubuntu Dialogue Corpus</a>: 来自自然两人对话的 2600 万轮对话数据</li><li><a href="https://github.com/stanfordnlp/dspy/issues/7867">[Bug] ModuleNotFoundError: No module named &#39;dspy.predict&#39; · Issue #7867 · stanfordnlp/dspy</a>: 发生了什么？当你使用 dspy-ai==2.6.7 导入 dspy 时，会立即失败并报错 ModuleNotFoundError: No module named &#39;dspy.predict&#39;。复现步骤见我的 gist https://gist.gi...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 条消息): 

yamashi: GPT-4.5 已在 Azure 上可用。
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1344431061135790202)** (26 messages🔥): 

> `CI 问题、Activation Offloading、Distributed Torch FL 代码、DPO 集成测试` 


- **针对 PR#2419 请求运行 CI**：一名成员请求在不合并的情况下为 [PR#2419](https://github.com/pytorch/torchtune/pull/2419) 启动 CI，因为他们正在进行*今天的最后一次尝试*。
   - 该 PR 涉及*截断与跳过 (truncation and skipping)*。
- **Activation Offloading 与 Checkpointing**：一名成员询问是否有理由规定 **Activation Offloading** 只能与 **Activation Checkpointing** 结合使用。
   - 另一名成员解释说，激活值（Activations）比 Checkpoints 需要*多得多的内存*（在他们的情况下，Checkpoint 只是 Transformer 块的输入向量），因此*卸载（offloading）和加载它们会限制 GPU 性能*，使其慢得无法忍受。
- **在分布式 FL 中处理合并模型的加载**：一名成员就 **分布式 Federated Learning (FL)** 代码中如何处理合并模型的加载寻求建议，特别是如何避免在所有 Rank 上都下载合并模型。
   - 他们考虑将合并后的模型转储到磁盘并让所有 Rank 从磁盘加载，得到的建议是改用 **Shared Memory（共享内存）**。
- **被 pre-commit 刁难**：一名成员提到在尝试实现 Federated Learning 时*再次被 pre-commit 刁难*。相关的函数位于[此处](https://github.com/maximegmd/torchtune/blob/d5dc4e6027ec0de33f6ffdc2eb1eee2148a1fb69/torchtune/training/federation/_participant.py#L171)。
   - 该成员在成功通过后表示如释重负：*请别再这样了 🥲*
- **DPO 集成测试状态**：一名成员询问 **DPO 集成测试** 的状态，想知道添加测试时遇到了什么问题。
   - 另一名成员回答说，目前已经有一个针对单设备 Recipe 的测试，参考[此文件](https://github.com/pytorch/torchtune/blob/7cbac8173edecd7f801bbbe9ee67adf00d6261c6/tests/recipes/test_lora_dpo_single_device.py)，并澄清为分布式 Recipe 添加测试应该也没有问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/maximegmd/torchtune/blob/d5dc4e6027ec0de33f6ffdc2eb1eee2148a1fb69/torchtune/training/federation/_participant.py#L171>">torchtune/torchtune/training/federation/_participant.py at d5dc4e6027ec0de33f6ffdc2eb1eee2148a1fb69 · maximegmd/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 torchtune 开发做出贡献。</li><li><a href="https://github.com/maximegmd/torchtune/blob/d5dc4e6027ec0de33f6ffdc2eb1eee2148a1fb69/torchtune/training/federation/_participant.py#L121>">torchtune/torchtune/training/federation/_participant.py at d5dc4e6027ec0de33f6ffdc2eb1eee2148a1fb69 · maximegmd/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。</li><li><a href="https://github.com/pytorch/torchtune/pull/2419">[RFC] truncation and skipping by krammnic · Pull Request #2419 · pytorch/torchtune</a>：#2344 提到了与数据加载和处理相关的两个要点。此 RFC 致力于这两个方面。截断（Truncation）：目前我们不支持左右两侧的截断....
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1344696167551864874)** (10 messages🔥): 

> `DeepSeek DualPipe、大规模 Federated Learning` 


- **用于计算-通信重叠的 DualPipe**：一名成员分享了 **DeepSeek DualPipe** 的 [GitHub 仓库](https://github.com/deepseek-ai/DualPipe/tree/main)链接，该仓库介绍了一种用于 **V3/R1 训练**中计算-通信重叠的*双向流水线并行算法*。
- **Federated Learning 面临通信瓶颈**：一名成员对 **DualPipe** 表示兴奋，但注意到它的新颖性，并提到尝试在**欧洲的 40 家医院**之间使用 **70B 模型**实现 **Federated Learning (FL)**。
   - 他们幽默地承认，在他们的 FL 设置中，通信开销可能会使 **DualPipe** 提供的优化显得微不足道，但建议它对于 FL 同步之间的增益可能有用。



**提到的链接**：<a href="https://github.com/deepseek-ai/DualPipe/tree/main">GitHub - deepseek-ai/DualPipe: A bidirectional pipeline parallelism algorithm for computation-communication overlap in V3/R1 training.</a>：一种用于 V3/R1 训练中计算-通信重叠的双向流水线并行算法。- deepseek-ai/DualPipe

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1344524774768119860)** (2 messages): 

> `` 


- **无**：无
- **无**：无


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1344405926349639693)** (29 messages🔥): 

> `Notebook 表情符号更改、带关键词的指令排列、与群组共享 Notebook、音频概览错误、Notebook 公开链接` 


- **用户请求为 Notebook 提供表情符号选项**：用户请求能够更改其 Notebook 上的表情符号，但该功能目前尚不可用。建议支持现有的功能请求或创建新请求。与 OneNote、Obsidian 和 Goodnotes 相比，该产品有很多强有力的竞争优势。
   - 一位用户指向了一条 [推文](https://x.com/signulll/status/1894806791172559355?t=M_rcWIE4NHsrLy8Ry3DzKA&s=19)，感叹 **NotebookLM** 缺乏动力和移动端 App，并将其归咎于 Google 扼杀内部创新的惯性模式。
- **Notebook 共享问题**：用户在与群组共享 Notebook 时遇到问题，发现仅提供链接是不够的，还需要专门添加用户以授予访问权限。
   - 似乎用户在访问共享的 Notebook 之前需要先拥有账号，并且可能需要同时通过电子邮件添加用户并提供链接。
- **音频概览（Audio Overview）困扰**：用户在尝试加载音频概览时，经常遇到错误提示 *“获取对话时出错。请重试。”*。
   - 该问题似乎是间歇性的，有时可以工作但经常失败，让依赖此功能的用户感到沮丧。
- **用户报告“服务不可用”错误**：一位用户报告在登录 NotebookLM 时收到 *“服务不可用”* 错误，消息显示 *“您尝试访问的服务不适用于您的账号”*，并链接到了他们的 [Google 账号服务页面](https://accounts.google.com/info/servicerestricted)。
   - 一位用户建议，该账号可能默认使用了学校账号而非个人账号。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://accounts.google.com/info/servicerestricted">服务不可用</a>：未找到描述</li><li><a href="https://x.com/signulll/status/1894806791172559355?t=M_rcWIE4NHsrLy8Ry3DzKA&s=19">来自 signüll (@signulll) 的推文</a>：notebooklm 拥有巨大的潜力，是 Google 多年来推出的最好产品之一。但按照 Google 的典型风格，它似乎失去了所有动力并被任其自生自灭。没有移动端 App，没有实质性的...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1344403286928789516)** (5 messages): 

> `仓库结构简化、Mojo 优先级、Chris Lattner 的博客文章` 


- **Modular 简化 MAX 和 Mojo 仓库结构**：Modular 旨在简化其 **MAX** 和 **Mojo** 的仓库结构，以方便对文档和标准库进行贡献，并整合错误报告和功能请求，详见[此论坛帖子](https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648)。
- **对 Mojo 作为独立语言的未来产生质疑**：一位成员询问仓库简化是否意味着不再将 **Mojo** 作为独立的语言来优先发展。
- **Chris Lattner 的系列博客文章**：一位成员发现 **Chris Lattner** 的系列博客文章非常出色且富有洞察力，并遗憾没有参加 **GPU 编程课程**。
   - 该成员提到，之前在入门课程中被 *在 TensorFlow 中做琐碎的事情* 劝退，并指出更复杂的任务似乎被 *锁在一堆数据之后*。



**提到的链接**：<a href="https://forum.modular.com/t/upcoming-changes-to-our-github-repositories/648">我们 GitHub 仓库即将发生的变更</a>：明天（2月27日），我们将精简 GitHub 仓库！max 仓库将合并到 mojo 仓库中，将所有内容整合在一起。一个新的子目录将存放 Mojo 标准库...

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1344399712941248552)** (25 messages🔥): 

> `stdlib 中的 MLIR, Mojo 中的 HyperLogLog, Mojo 中的 MLIR Dialects, MAX Graph Compiler, Mojo 中的 Unions` 


- **Mojo Hyperlogs 与 GitHub!**: 一位成员在 Mojo 中实现了 **HyperLogLog algorithm**，并在 [GitHub](https://github.com/axiomhq/mojo-hyperloglog) 上分享了它，寻求改进建议。
   - 他们表达了对使用 Mojo 的喜爱，将其描述为 *一个更强大的 Python*。
- **MAX 使用未公开文档的 MLIR**: 成员们讨论了在 stdlib 中使用内联 **MLIR** 的情况，这在很大程度上没有文档记录，且仅供 Modular 和 stdlib 贡献者内部使用。
   - 这暗示了内部 dialects `mo`, `moq`, `mogg`, `mef`, `mgp`, `grt`, `rmo` 并不打算向公众开放。
- **探索 Mojo 内部 Dialects**: 一位成员使用 `nm` 探索了 Mojo 的内部结构，以发现并列出 `libmof.so` 中与 **dialects, types, and ops** 相关的细节。
   - 这次探索揭示了 `union` 类型，引发了关于其预期用途以及由于定义不明确的 **aliasing and type-punning rules** 可能带来的风险的讨论。
- **MAX graph compiler 使用 mlir dialects**: 一位成员澄清说，特定的 MLIR dialects（如 `mo`）主要由 **MAX Graph Compiler** 使用，不属于 Mojo 的 runtime。
   - 这些 dialects 仅与 **graph compilation** 相关，目前没有办法手动将它们加载到 Mojo 的 MlirContext 中。
- **Mojo MLIR 的稳定性担忧**: 稳定性和文档工作是某些 MLIR dialects 未公开的原因，因为它们包含对 Modular 竞争优势至关重要的方面，完整记录它们可能会稀释其价值。
   - 一位成员指出，一旦 Modular 更加成熟，他们就能负担得起开放这些内容的代价，因为届时使用他们的系统将比复制它更容易。



**提到的链接**: <a href="https://github.com/axiomhq/mojo-hyperloglog">GitHub - axiomhq/mojo-hyperloglog</a>: 通过在 GitHub 上创建账号来为 axiomhq/mojo-hyperloglog 的开发做出贡献。

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1344400592914878636)** (18 messages🔥): 

> `生产环境中的 MCP, Claude Code 基于 diff 的编辑, 官方 everything server SSE, Glama AI GitHub App, Claude Code 邀请` 


- **MCP** 在生产环境找到用户: 成员们确认 **MCP** 可以用于生产级工作流。
   - 一位用户指出，尽管存在行号变化的问题，他们仍在使用它，并通过 prompting 和资源包含来缓解这一问题。
- **Claude Code** 使用 **Diff-Based Editing**，在处理 Go 时遇到困难: 用户报告 **Claude Code** 使用基于 diff 的编辑，在编辑 Go 时会失败，原因是出于可读性考虑添加了空格。
   - 一位用户提到，这个问题是由 *为了提高可读性而在 Go 代码中添加空格的方式* 引起的。
- **官方 Everything Server** 具备 **SSE**: 官方 everything server 具有 **SSE (Server-Sent Events)** 功能，非常适合测试。
   - 一位用户发现 **SSE** 对于测试目的来说是 *完美* 的。
- **GitHub App** 帮助扩展 **Glama AI**: **Glama AI** 的创建者请求用户安装一个 GitHub app，以支持该项目并提高 API rate limits。
   - 一位用户在安装过程中遇到了 `could_not_parse_params` 错误，但创建者澄清说，安装注册已经足够，且不会进行数据收集。
- **MCP Server** 存在远程资源问题: 一位用户竭尽全力想让他们的 **MCP server** 配合资源工作，包括 subscribe_resource 装饰器。
   - 结果发现，用户必须 *手动将资源添加到 context 中，就像从文件系统添加文件一样，以便 client 能够使用 resource/read 方法*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp">开源 MCP servers</a>: 企业级安全、隐私，具备 agents, MCP, prompt templates 等功能。</li><li><a href="https://github.com/apps/glama-ai">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1344689347970203739)** (5 messages): 

> `Redmine MCP Server, Ableton Voice Control, tinylm library for running LLMs` 


- **MCP Redmine 发布，具备出色的 API 覆盖率**：一个新的 [MCP Redmine server](https://github.com/runekaagaard/mcp-redmine) 已发布，声称仅用不到 50 行代码就覆盖了几乎整个 **Redmine json API**。
   - 据报告，该服务器利用了 **gh 用户 d-yoshi 的 OpenAPI 规范**。
- **Ableton 语音控制构想浮现**：一名成员对 MCP Redmine 表示了热情，并设想通过语音命令控制 **Ableton**，建议的工作流如 *'Ok now lets record a new track using input7 with a bit of reverb added and routed to output 3+4.'*
   - 另一名成员指出，虽然通过 **Ableton 远程控制脚本** 无法直接加载设备，但结合 **Whisper 流程** 和自定义的 **Ableton MCP client** 可以实现这一目标。
- **tinylm 助力浏览器端 LLMs**：[tinylm](https://github.com/wizenheimer/tinylm) 的版本 0 已发布，这是一个用于在浏览器或 Node.js 中通过 **WebGPU 加速** 运行 **LLMs** 和嵌入模型的库，支持 OpenAI 兼容的 API。
   - tinylm 宣称具有**零成本推理**、完全的隐私保护，并支持文本生成、文本嵌入和实时 Token 流式传输等功能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/runekaagaard/mcp-redmine">GitHub - runekaagaard/mcp-redmine: A redmine MCP server covering close to 100% of redmines API</a>: 一个覆盖了近 100% Redmine API 的 Redmine MCP 服务器 - runekaagaard/mcp-redmine</li><li><a href="https://tinylm.wizenheimer.dev/">tinylm - Run Models Locally with WebGPU</a>: 暂无描述</li><li><a href="https://github.com/wizenheimer/tinylm">GitHub - wizenheimer/tinylm: Zero-cost client-side inference using WebGPU | OpenAI-compliant | NodeJS | Chrome</a>: 使用 WebGPU 的零成本客户端推理 | OpenAI 兼容 | NodeJS | Chrome - wizenheimer/tinylm
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1344417755591479306)** (18 messages🔥): 

> `Live Mode, Voice Assistant, GGUF models, Alltalk TTS` 


- **LIVE 模式功能请求**：一名成员请求开发类似于 **Google Gemini** 的 **LIVE 模式**功能，认为这将超越 Google 的工具。
   - 他们提议使用**语音识别 (STT)**进行输入，并使用 **TTS** 进行输出，并分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=6zAk0KHmiGw)，展示了用 Python 构建的、利用 **OpenAI Whisper** 进行离线语音检测的 **GPT4ALL 语音助手**。
- **理解 GGUF 模型的 Chat Templates**：一名成员询问了 **chat_template** 在 **GGUF 模型**中的用法，质疑模板是否在初始加载时从 **.gguf** 文件读取并存储在 **model3.json** 中。
   - 他们寻求确认在 **GUI** 中所做的更改是否保存在 **model3.json** 中，正如在 **gpt4all** 和 **Hugging Face** 模型中所观察到的那样。
- **Oobabooga 实现了 Alltalk TTS**：一名成员提到 [Oobabooga](https://github.com/oobabooga/text-generation-webui) 实现了一个名为 **alltalk_tts** 的**文本转语音**扩展，可与 **GGUF**、**AWQ** 和 **GPTQ** 模型配合使用。
   - 他们指出安装过程有些复杂，涉及使用 **BAT 安装**的 **Python 安装**，但不需要编写代码。
- **网速影响安装时间**：一名成员哀叹其 **40 kbps** 的缓慢网速，这将使 [Oobabooga](https://github.com/oobabooga/text-generation-webui) 的安装耗时约**两天**。
   - 另一名成员曾表示安装需要 **1 小时**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=6zAk0KHmiGw">在 10 分钟内创建一个 GPT4ALL 语音助手</a>: 使用 Python 编写本地 GPT 语音助手。在本视频中，我们将学习如何在没有互联网连接的情况下运行 OpenAI Whisper，以及后台语音检测...</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models with support for multiple inference backends.</a>: 一个支持多种推理后端的 LLMs Gradio Web UI。 - oobabooga/text-generation-webui
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1344447619362979931)** (12 messages🔥): 

> `GROUP operations AST changes, BEAM search strategies for OptOps, arange GROUP optimization failure, LLVM speed regression` 


- **GROUP AST 变更遇到性能瓶颈**: 针对 **GROUP operations** 的 AST 变更在对 (2048,2048) 张量求和时已达到与 PyTorch 持平的水平，但在处理 (4096,4096) 张量时由于需要**多个连续的 OptOps** 而面临困难。
   - 作者询问应该尝试调整 **BEAM search** 以找到这些 OptOps，还是修改 **lowerer/expander** 以输出不同的内容来实现**多个累加器**。
- **BEAM Search 停滞，阻碍进度**: 作者在让 **BEAM search** 找到高效求和较大张量 (4096,4096) 所需的最佳 **OptOps** 序列时面临挑战。
   - 他们正在考虑修改 **lowerer** 或 **expander** 以生成可以更好利用多个累加器和水平加法重排 (horizontal add swizzles) 的替代 AST，但对能否保证性能提升表示不确定。
- **arange GROUP 优化破坏了 CI**: 作者报告称 `arange` 的 **GROUP 优化** 未被应用，导致 arange 操作中出现额外的内层循环并导致 CI 失败。
   - 他们已 rebase 到 master 分支且测试通过，成功匹配了 PyTorch，并询问关于 `arange` **GROUP 优化** 的建议。
- **Speed Test BEAM=2 超时**: 一位成员注意到 "Speed Test BEAM=2" 在 [GitHub Actions](https://github.com/tinygrad/tinygrad/actions/runs/13555381099/job/37888418102?pr=9190) 上超时。
   - 作者通过修减一些新增的 OptOps 修复了此问题，并报告称添加 **GROUP** 和 **GROUPTOP** 减慢了 **BEAM search**，因为尝试的 kernel 数量大幅增加。
- **Pull Request 上的测试仍然失败**: 一位成员表示 Pull Request 上的测试仍然失败，且代码在 **LLVM** 速度上慢了很多，**收益为 0**。
   - 作者澄清说他们目前还不是在请求 review，而是想知道 arange 测试在 **GROUP OptOps** 上失败是否是一个已知问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/9190/files">[Bounty] Made TestSpeed.test_sum yellow on Macs with LLVM by josephsweeney · Pull Request #9190 · tinygrad/tinygrad</a>: 为了实现这一目标，我在没有局部变量的设备（CLANG 和 LLVM）上启用了 GROUP OptOps，通过添加额外的 reduce 而不是发射 locals。其他必要的更改来自于...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9190">[Bounty] Made TestSpeed.test_sum yellow on Macs with LLVM by josephsweeney · Pull Request #9190 · tinygrad/tinygrad</a>: 为了实现这一目标，我在没有局部变量的设备（CLANG 和 LLVM）上启用了 GROUP OptOps，通过添加额外的 reduce 而不是发射 locals。其他必要的更改来自于...</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/13555381099/job/37888418102?pr=9190">[Bounty] Made TestSpeed.test_sum yellow on Macs with LLVM · tinygrad/tinygrad@fd63dd6</a>: 你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - [Bounty] Made TestSpeed.test_sum yellow on Macs with LLVM · tinygrad/tinygrad@fd63dd6
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1344553935016431646)** (1 messages): 

> `` 


- **用户开始代码探索**: 一位用户表达了感谢，并表示他们将通过研究代码来寻找问题的答案。
- **独立解决问题**: 该用户决定独立调查代码库以解决他们的疑问。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1344574285179781161)** (2 messages): 

> `Research Plans Announcement, Discord Server Recruitment` 


- **通过 Discord 发布研究计划公告！**: 一位成员分享了一个 Discord 邀请链接 ([https://discord.gg/5MbT7ce9](https://discord.gg/5MbT7ce9))，用于发布关于其*研究计划的更详细公告*。
   - 他们鼓励感兴趣的人士私信（DM）了解更多信息或直接加入 Discord 服务器。
- **Discord 服务器招募新成员！**: 一位热情的成员发出邀请，欢迎加入他们的 Discord 服务器以了解研究计划，并可以通过私信直接交流。
   - 提供的 Discord 邀请链接 ([https://discord.gg/5MbT7ce9](https://discord.gg/5MbT7ce9)) 承诺将发布关于其正在进行的项目和合作机会的*更详细公告*。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1344780853070397472)** (1 messages): 

> `Research Track, Predictive Decision Making, Long Term Memory in Agents` 


- **研究方向 (Research Track) 启动针对性研究小组**：一个研究方向正在形成，重点关注 Agent 中的 **Predictive Decision Making** 和 **Long-term Memory**。
   - 该小组将举行定期同步会议，讨论课程内容并促进协作；感兴趣的成员可以通过 [此 Discord 邀请](https://discord.gg/5MbT7ce9) 加入。
- **Predictive Decision Making 小组启动**：一个新的子小组将专注于 AI Agent 内部的 **Predictive Decision Making** 策略。
   - 该小组旨在探索增强 Agent 预判未来结果并做出明智选择能力的方法。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1344809582878396538)** (1 messages): 

> `tinylm, WebGPU, OpenAI SDK, client-side LLMs` 


- **tinylm v0 发布**：一个名为 [tinylm](https://tinylm.wizenheimer.dev/) 的库已发布，用于在浏览器或 **Node.js** 中通过 **WebGPU** 加速在客户端运行 **LLMs** 和 Embedding 模型。
   - 它支持 **OpenAI SDK**，如文本生成和 Embedding 生成，即将支持 Text-to-Speech 和 Speech-to-Text，且无需服务器。
- **tinylm 提供兼容 OpenAI 的 API**：[tinylm](https://tinylm.wizenheimer.dev/) 提供了一个 **OpenAI-compatible API**，用于利用 **WebGPU** 加速直接在浏览器或 Node.js 应用程序中运行语言模型。
   - 特性包括 **零成本推理 (zero-cost inference)**、**客户端处理**、**文本生成**、**文本 Embedding**、**跨平台兼容性**、**真正的流式传输 (true streaming)** 以及 **详细的进度追踪**。



**提到的链接**：<a href="https://tinylm.wizenheimer.dev/">tinylm - 使用 WebGPU 在本地运行模型</a>：未找到描述

  

---


---


{% else %}


> 完整的逐频道详情已在邮件中截断。 
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}