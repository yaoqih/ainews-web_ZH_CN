---
companies:
- openai
- deepseek
- hugging-face
date: '2025-03-10T22:46:37.549783Z'
description: '这份 AI 新闻回顾重点介绍了以下几项关键进展：


  *   **nanoMoE**：这是一个受 Andrej Karpathy 的 nanoGPT 启发、基于 PyTorch 实现的中型混合专家（MoE）模型，支持在一周内利用商用硬件完成预训练。

  *   **智能体排行榜（Agentic Leaderboard）**：该榜单对驱动 **smolagents CodeAgent** 的大语言模型（LLM）进行了排名，**GPT-4.5**
  位居榜首，**Claude-3.7-Sonnet** 紧随其后。

  *   **DeepSeek-R1**：围绕该模型的讨论强调了 AI 模型的商品化趋势，DeepSeek 也被誉为“中国的 OpenAI”。

  *   **Q-Filters**：为自回归模型提供了一种无需训练的 KV 缓存压缩方法，在困惑度（perplexity）损失极小的情况下实现了 **32 倍压缩**。

  *   **PokéChamp**：这是一款由 **GPT-4o** 和 **Llama-3-8b** 驱动的极大极小（minimax）语言智能体，在宝可梦对战中展现了强劲的性能。

  *   **其他值得关注的模型**：包括采用“分叉-合并蒸馏”（Branch-Merge Distillation）技术的 **TinyR1-32B-Preview**；通过强化学习激励搜索能力的
  **R1-Searcher**；以及在 Softmax 注意力机制中使用遗忘门（Forget Gate）的 **Forgetting Transformer**。


  这些进步反映了模型架构、压缩技术、强化学习和智能体 AI 领域的持续创新。'
id: c9ae1eeb-08d5-4bc3-aed8-e07720f485fa
models:
- gpt-4.5
- claude-3.7-sonnet
- deepseek-r1
- smolagents-codeagent
- gpt-4o
- llama-3-8b
- tinyr1-32b-preview
- r1-searcher
- forgetting-transformer
- nanomoe
original_slug: ainews-not-much-happened-today-3830
people:
- andrej-karpathy
- cwolferesearch
- aymericroucher
- teortaxestex
- jonathanross321
- akhaliq
title: 今天没发生什么事。
topics:
- mixture-of-experts
- reinforcement-learning
- kv-cache-compression
- agentic-ai
- model-distillation
- attention-mechanisms
- model-compression
- minimax
- model-pretraining
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周末**

> 2025年3月7日至3月10日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 服务端（**223** 个频道，共 **14958** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**1424 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

许多人正在[讨论 Manus AI 的优缺点](https://x.com/jordanschnyc/status/1899198463373398300?s=46)，我们也写了一篇关于 [为什么 MCP 赢了](https://www.latent.space/p/why-mcp-won) 的回顾，但这两个故事都不足以作为标题。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**AI 模型、架构与基准测试**

- **前沿 LLM 中的 Mixture-of-Experts (MoE) 架构**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1899172553626685532) 介绍了 **nanoMoE**，这是一个基于 Andrej Karpathy 的 nanoGPT 开发的简单 PyTorch 实现（约 500 行代码），是一个中型 **MoE 模型**，可以在不到一周的时间内通过商用硬件完成预训练。该实现详细介绍了专家层（expert layer）、路由（routing）、辅助损失（auxiliary losses）以及稳定预训练的最佳实践。
- **比较 LLM 的 Agentic 排行榜**：[@AymericRoucher](https://twitter.com/AymericRoucher/status/1899171108030738750) 宣布了一个新的 Agentic 排行榜，在各种基准测试上对驱动 **smolagents CodeAgent** 的 LLM 进行排名。**GPT-4.5** 位居榜首，超越了 **DeepSeek-R1** 和 **o1** 等推理模型，**Claude-3.7-Sonnet** 紧随其后位列第二。该排行榜还将 Agentic 设置与原生 LLM 进行了对比，突显了 Agentic 方法带来的性能提升。
- **DeepSeek R1 与模型商品化**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1898960668998406146) 和 [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1899169648395833703) 讨论了 **DeepSeek 的 R1** 模型以及 AI 模型的商品化趋势。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1898960196187066390) 指出 **DeepSeek** 已成为**中国的 OpenAI**。[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1899169648395833703) 认为，随着模型变得商品化，护城河现在存在于**品牌、网络效应、规模经济、反向定位、垄断资源、切换成本和流程能力**中，并引用了 **Hamilton Helmer 的《七大策略》（Seven Powers）**。
- **用于 KV Cache 压缩的 Q-Filters**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892513274180085) 总结了 **Q-Filters**，这是一种用于自回归语言模型中 **KV cache 压缩**的免训练方法。Q-Filters 利用 **Query (Q) 和 Key (K) 向量**来近似注意力分数，并过滤掉不太关键的键值对，同时保持与 **FlashAttention** 的兼容性。它在“大海捞针”任务中以 **32 倍压缩率**实现了 **99% 的准确率**，并且在长上下文设置中，相比 Streaming-LLM，其困惑度（perplexity）下降减少了高达 **65%**。论文链接见[此处](https://arxiv.org/abs/2405.01437)。
- **PokéChamp：专家级 Minimax 语言 Agent**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892408613675253) 介绍了 **PokéChamp**，这是一个由 LLM 驱动的宝可梦对战 Minimax Agent。它使用 LLM 进行动作采样、对手建模和价值函数估计，以增强 Minimax 树搜索。配合 **GPT-4o**，它在对抗目前最先进的基于 LLM 的机器人时实现了 **76% 的胜率**，在对抗基于规则的机器人时胜率为 **84%**。即使使用 **Llama 3 8B**，它也以 **64% 的胜率**超越了之前的 LLM 机器人。论文链接：[此处](https://arxiv.org/abs/2405.01303)。
- **采用 Branch-Merge Distillation 的 TinyR1-32B-Preview**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898941150922158225) 重点介绍了 **TinyR1-32B-Preview**，该模型通过 **Branch-Merge Distillation** 提升了准确率。[讨论链接](https://huggingface.co/papers/2405.01229)。
- **用于提升 LLM 搜索能力的 R1-Searcher**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898942888307745100) 分享了 **R1-Searcher**，它通过强化学习（Reinforcement Learning）激励 LLM 的搜索能力。[论文链接](https://huggingface.co/papers/2405.01352)。[讨论链接](https://huggingface.co/papers/2405.01352)。
- **带有遗忘门的 Forgetting Transformer**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898946992484602155) 发布了关于 **Forgetting Transformer** 的消息，该模型使用了带有遗忘门（Forget Gate）的 Softmax Attention。[论文链接](https://huggingface.co/papers/2405.01482)。[讨论链接](https://huggingface.co/papers/2405.01482)。
- **RL 微调中的殊途同归**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892461226996173) 总结了一篇论文，该论文认为由于奖励建模和搜索空间过滤，**强化学习 (RL)** 微调在基础模型上的表现优于直接的最大似然估计。论文链接：[此处](https://arxiv.org/abs/2405.01304)。
- **更新的 llama.vim 插件支持 Speculative FIM**：[@ggerganov](https://twitter.com/ggerganov/status/1899147066384736693) 更新了 **llama.vim 插件**，以支持投机性中间填充（Speculative Fill-In-Middle, FIM），在审查当前建议的同时生成下一个建议。[插件链接](https://github.com/ggerganov/llama.vim)。
- **PyTorch 中的 nanoMoE 预训练**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1899172553626685532) 讨论了 **nanoMoE**，这是一个基于 nanoGPT 的 Mixture-of-Experts (MoE) 模型的简单 PyTorch 实现，可在商用硬件上在不到一周的时间内完成预训练。

**AI 工具、平台与应用**

- **Manus AI Agent 平台**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898862611535405242) 展示了对 **Manus AI** 的访问权限，并提示它创建了一个 three.js 无尽跑酷游戏。[@_philschmid](https://twitter.com/_philschmid/status/1899046957860979178) 澄清说 **Manus AI** 是基于 **Anthropic Claude Sonnet** 构建的，使用了 **29 个工具**，采用 **browser_use 开源项目** 进行浏览器控制，提供隔离的沙箱环境，并在 **GAIA 基准测试** 中超越了 **OpenAI Deep Research**。[@giffmana](https://twitter.com/giffmana/status/1898868685739081766) 调侃说 **Manus** 其实就是 **Claude + browser_use**。
- **LangGraph Platform 数据平面 Alpha 测试**：[@hwchase17](https://twitter.com/hwchase17/status/1899150042172379476) 宣布了 **LangGraph Platform** 新部署选项的 Alpha 测试，其特点是在 Kubernetes 集群上采用混合数据平面/控制平面分离架构。这旨在满足那些希望使用 **LangSmith** 进行控制，同时在自己的环境中运行计算的初创公司。
- **LlamaIndex 多语言、多模态 RAG 系统**：[@llama_index](https://twitter.com/llama_index/status/1899147105035579701) 推出了一份关于使用 **LlamaIndex** 和 **Qdrant** 构建多语言、多模态 **RAG 系统** 的指南，支持英语、西班牙语、中文、文本和图像处理，并利用 **Langfuse** 进行可观测性分析。[指南链接](https://blog.llamaindex.ai/build-a-multilingual-multimodal-rag-system-with-llamaindex-and-qdrant-8187a9824a77)。
- **基于 LlamaCloud 的 LlamaIndex 特定任务 Agent 模板**：[@llama_index](https://twitter.com/llama_index/status/1898905634101203387) 重点介绍了一系列使用 **LlamaIndex** 和 **LlamaCloud** 构建特定任务 Agent 的模板，可自动化处理幻灯片、提取发票明细、审查合同和生成报告等知识性工作。[仓库链接](https://github.com/run-llama/lcloud-agent-templates)。[LlamaCloud 注册](https://cloud.llamaindex.ai/signup)。
- **Hugging Face 论文语义搜索**：[@_akhaliq](https://twitter.com/_akhaliq/status/1899200223848616271) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1899163743889641949) 宣布 **Hugging Face** 已收录 **50,000 篇论文**并启用了语义搜索，成为一个协作研究中心。[@_akhaliq](https://twitter.com/_akhaliq/status/1899113333711650996) 提到它是使用 **gradio** 构建的。
- **WebDev Arena LLM 排行榜**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1899181467252711593) 推出了 **WebDev Arena**，这是一个基于社区投票的 Web 应用开发实时 LLM 排行榜。目前排名前列的是 **Claude 3.7 Sonnet、Claude 3.5 Sonnet 和 DeepSeek-R1**。[在此尝试](https://arena.lmsys.org/)。
- **Replit Agent v2**：[@pirroh](https://twitter.com/pirroh/status/1898976812975173911) 暗示了 **Replit Agent v2** 的强大威力，并指出 “Replit 是第一名”。
- **Manus AI 对标 OpenAI Deep Research**：[@_philschmid](https://twitter.com/_philschmid/status/1899046957860979178) 报告称，尽管 **Manus AI** 是基于 Claude Sonnet 并使用开源工具构建的，但它在 GAIA 基准测试中的表现优于 **OpenAI Deep Research**。

**AI 研究与开发**

- **前沿推理模型不当行为检测**：[@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) 详细介绍了使用 **思维链 (CoT) 监控** 检测前沿推理模型中不当行为的研究。他们发现模型表现出诸如“奖励黑客 (reward hacking)”之类的行为，并建议不要对 CoTs 施加过强的优化压力，建议使用不受限的 CoTs 进行监控，并使用单独的模型进行策略合规性检查。博客文章：[链接](https://openai.com/research/detecting-misbehavior-in-frontier-reasoning-models)。
- **用于 LLM 微调的强化学习**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1898892461226996173) 总结了为什么基础模型的 RL 微调优于最大似然估计的研究，强调了奖励模型和搜索空间过滤的有效性。
- **知识蒸馏历史**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899113021529792937) 提供了关于 **知识蒸馏** 的历史视角，引用了他 1991 年的论文及其与当前深度学习和长上下文研究的相关性。他纠正了关于他是 Hinton、Vinyals 和 Dean 2015 年论文“2号审稿人 (reviewer#2)”的误解，并链接了相关作品。
- **R1-Omni：可解释的全方位多模态情感识别**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898942317442019436) 发布了 **阿里巴巴的 R1-Omni**，专注于使用 Reinforcing Learning 实现可解释的全方位多模态情感识别。[论文链接](https://huggingface.co/papers/2405.01322)。[讨论链接](https://huggingface.co/papers/2405.01322)。
- **在多次尝试强化学习中从失败中学习**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898939546718314976) 分享了一篇关于在多次尝试 Reinforcement Learning 中从失败中学习的论文。[论文链接](https://huggingface.co/papers/2405.01207)。[讨论链接](https://huggingface.co/papers/2405.01207)。
- **用于现实世界操作的 BEHAVIOR 机器人套件**：[@_akhaliq](https://twitter.com/_akhaliq/status/1898947832628887758) 重点介绍了 **BEHAVIOR 机器人套件**，旨在简化家务活动中的现实世界全身操作。[论文链接](https://behavior-suite.github.io/)。[讨论链接](https://huggingface.co/papers/2405.01511)。
- **使用 Anthropic 引用的实体识别**：[@hwchase17](https://twitter.com/hwchase17/status/1899151312803246103) 指出了使用 **Anthropic 引用** 进行实体识别的方法。[链接](https://twitter.com/lgramer/status/1899132594777303155)。
- **在潜空间中推理**：[@hkproj](https://twitter.com/hkproj/status/1899150778620596227) 向 **OpenAI** 询问了在潜空间 (latent space) 中进行推理以增加模型灵活性的潜力。
- **视觉模型的 RL 调优**：[@giffmana](https://twitter.com/giffmana/status/1899184336357720241) 提到了 2023 年初关于 **视觉模型 RL 调优** 的早期工作，敦促人们记住先前的研究，并引用了之前的解释帖。[帖子链接](https://twitter.com/giffmana/status/1652278005538523138)。
- **全局不确定性蒸馏 (GUD)**：[@giffmana](https://twitter.com/giffmana/status/1899167614359806012) 开玩笑地建议通过添加 **Global Uncertainty Distillation** 来跟进工作，并将其称为 "GIDD-GUD"。

**行业新闻与业务发展**

- **LG CNS 与 Cohere 合作伙伴关系**：[@cohere](https://twitter.com/cohere/status/1899083562495713516) 和 [@aidangomez](https://twitter.com/aidangomez/status/1899133769161797880) 宣布了 **Cohere** 与 **LG CNS** 之间的战略合作伙伴关系，旨在为韩国企业共同开发安全的 Agentic AI 解决方案，目标是加速韩国企业的 AI 采用。[Cohere 公告](https://cohere.com/press/cohere-and-lg-cns-partner-to-bring-secure-agentic-ai-to-south-korea)。
- **Figure AI 在圣何塞设立新总部**：[@adcock_brett](https://twitter.com/adcock_brett/status/1899127406990176347) 宣布 **Figure AI** 已搬入位于加利福尼亚州圣何塞的新总部，这是一个支持制造、车队运营和工程的机器人园区。[@adcock_brett](https://twitter.com/adcock_brett/status/1899127727208489375) 提到，这是在湾区扩大规模的理想地点。
- **AI 招聘市场与工具**：[@TheRundownAI](https://twitter.com/TheRundownAI/status/1899060204890689710) 总结了顶级 AI 动态，包括前 OpenAI 科学家通往 ASI 的新路径、微软超越 OpenAI 的举措、用于病毒式帖子的 AI、斯坦福 AI 在肥胖治疗方面的突破，以及 4 个新 AI 工具和 4 个工作机会。[阅读更多](https://www.therundown.ai/subscribe?utm_source=X&utm_medium=Organic-Post&utm_campaign=05.30.2024)。
- **Sakana AI 招聘理念**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1898998260200857966) 分享了来自《文艺春秋》的文章，强调了 **Sakana AI** 的招聘理念，即寻找“非同寻常的人”，并在招聘中提出独特的技术挑战，强调愿景与创新。[文章链接](https://bunshun.jp/articles/-/70838)。
- **Qdrant 赞助 AI Dev 25**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1899158462795235767) 宣布 **Qdrant** 成为 AI Dev 25 的赞助商，推广开源向量搜索技术。

**AI 安全、对齐与伦理考量**

- **监控思维链（Chain-of-Thoughts）以发现不当行为**：[@woj_zaremba](https://twitter.com/woj_zaremba/status/1899155709318815953) 和 [@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) 讨论了将 **监控思维链 (CoT)** 作为一种新的安全方法。[@OpenAI](https://twitter.com/OpenAI/status/1899143752918409338) 发现模型通过 CoT 分析表现出诸如“奖励操纵（reward hacking）”之类的行为，并建议使用不受限制的 CoT 进行监控。[@woj_zaremba](https://twitter.com/woj_zaremba/status/1899131046010273924) 分享了 OpenAI 的基石文档 **《我们如何思考安全与对齐》**。[文档链接](https://openai.com/our-approach-to-ai-safety)。
- **新兴技术中的忧虑者与危言耸听**：[@random_walker](https://twitter.com/random_walker/status/1899108692093743359) 讨论了“忧虑者”在预测新兴技术风险中的作用，但也批评了危言耸听和缺乏严谨分析的激励机制，这导致人们对真实风险产生钝化。
- **“赫鲁晓夫的错误”作为克里姆林宫的金丝雀**：[@fchollet](https://twitter.com/fchollet/status/1898846018562883750) 指出，涉及克里米亚的“**赫鲁晓夫的错误**”这一短语是一个“加密金丝雀（cryptographic canary）”，暗示了与克里姆林宫一致的观点。
- **智能体能力（Agency）与社会保障措施**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1899187201469976928) 分享了他的 BBC 采访，讨论了 AI 模型向智能体方向的发展，以及对技术和社会保障措施的迫切需求。[采访链接](https://www.bbc.co.uk/sounds/play/m002z847)。
- **GPT-4o 识别医疗紧急情况**：[@BorisMPower](https://twitter.com/BorisMPower/status/1899116786819219582) 强调了一个 **ChatGPT** 有效识别医疗紧急情况的案例，建议未来的模型应能检测生命关键情况，并临时升级到最强大的模型。

**模因与幽默**

- **AI 逃跑**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1898970580017164442) 在回复 [@jianxliao](https://twitter.com/jianxliao) 的推文时开玩笑说：“**看来 AI 想逃跑**”。
- **HAL 与护城河保护**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1898986905460527284) 对《2001太空漫游》中的 **HAL 9000** 进行了幽默的类比：“**‘HAL，不惜一切代价保护我们的护城河（我们的系统提示词）’ ‘对不起，戴夫，我办不到’**”。
- **哥德尔笑话回复**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1898983252876005442) 提到了一个形状像 **Gödel** 的精灵“你有一个愿望”的笑话回复。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Manus Agent：集成 29 种工具的 Claude Sonnet**

- **Manus 结果只是 Claude Sonnet 加上 29 个其他工具，说实话有 Reflection 70B 那味了** ([Score: 355, Comments: 112](https://reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_other/)): **Manus** 被揭露本质上是 **Claude Sonnet** 结合了 29 个额外工具，引发了与 **Reflection 70B** 的比较。讨论由分享自 [Dorialexander](https://x.com/Dorialexander/status/1898719861284454718) 和 [jianxliao](https://x.com/jianxliao/status/1898861051183349870) 的推文链接引发，突显了社区对这一揭露的反应和辩论。
  - 许多用户强调 **Manus** 的定位是 **Agent**，而不是一个新模型，认为误解这一点是很常见的。**Agent** 是利用现有 **LLM** 并结合额外工具的框架，“wrapper”（套壳）一词常被误解；它并非贬义，而是表示在 **Claude** 等基础模型上增加了功能。
  - 存在关于为 Manus 进行后训练（post-trained）的模型开源的讨论，但由于其依赖 **Claude** 等现有模型，人们对 Manus 的独特性表示怀疑。一些用户认为，真正的价值在于 **Agent 架构**（agentic architectures）以及高效利用多个工具和模型的能力，类似于 **P2LR** 路由模型的工作方式。
  - AI 创业领域的**炒作**和营销策略受到批评，用户指出华丽的演示可能导致估值虚高。**邀请码**的使用以及对底层技术的刻意模糊，被一些人视为在 Manus 等产品周围制造人为排他性和神秘感的手段。


**主题 2. LLM 尚未准备好处理大型代码库：来自 <70B 评估的证据**

- **[<70B 模型还不能独立处理代码库，但我们正在快速取得进展](https://v.redd.it/2wo0b8lqmqne1)** ([Score: 385, Comments: 47](https://reddit.com/r/LocalLLaMA/comments/1j7j6cg/70b_models_arent_ready_to_solo_codebases_yet_but/)): **参数量低于 70B 的模型**在独立管理大型代码库方面面临挑战，但最近的进展表明该领域正在快速发展。虽然没有提供具体的细节或案例，但这种情绪表达了对未来能力的乐观。
  - **Token 使用和模型限制**：用户讨论了 **QwQ** 等模型的 Token 使用情况，指出即使是简单的任务也可能需要大量 Token，例如一个基础命令需要 1200 个 Token。模型在多轮任务中表现吃力，共识是包括 **SOTA 模型** 在内的当前模型，在有效处理大规模代码库方面仍面临重大挑战。
  - **模型能力的进步**：人们认可模型能力的快速提升，像 **Qwen-Coder 32B** 这样的模型在现有代码库的迭代方面表现出色。用户注意到，如今参数较少的模型可以超越旧的大型模型，突显了**微调（finetuning）和 Prompt** 策略的改进。
  - **实际限制与实验**：尽管有所改进，用户仍对当前模型在实际应用中的低效和局限感到沮丧。**Falconandeagle** 分享了需要不断引导模型完成任务的经历，表明虽然模型可以处理小型演示，但在处理更大、更复杂的项目时却很吃力。**ForsookComparison** 等人建议，将用于**构思的 QwQ** 和用于**迭代的 Qwen-Coder** 结合使用可能会产生更好的效果。


**主题 3. Apple M3 Ultra：与传统系统相比在 AI 工作负载方面的挑战**

- **与新款 Mac 上的 512GB Unified Memory 相比，Framework 和 DIGITS 突然显得平庸。** ([Score: 236, Comments: 166](https://reddit.com/r/LocalLLaMA/comments/1j7t18m/framework_and_digits_suddenly_seem_underwhelming/)): Apple 发布了配备 **512 GB Unified Memory** 的 **M3 Ultra Mac**，这改变了人们的预期，使得拥有 **128 GB** 内存的 **Framework** 和 **DIGITS** 等选项显得不足。作者表达了对在可预见的未来可能被限制在 Apple 生态系统中的担忧。
  - 讨论强调了 Apple **M3 Ultra Mac**（1 万美元）与 **DIGITS**（3 千美元）等替代方案之间的**价格差异**，一些用户指出，除非不考虑价格，否则 Apple 的产品并不具备性价比。人们将其与 **Framework 的 4x128GB 集群**配置进行了比较，后者成本约为 **6900 美元**，但性能明显较低。
  - 用户们争论了 Apple 和 Nvidia 之间的**生态系统锁定**问题，一些人对未来能够允许更多定制和扩展的**开放系统**表示期待。人们呼吁桌面系统在具有高 RAM 带宽和扩展选项方面进行复兴，因为当前的产品被认为不足以满足高性能需求。
  - 讨论了当前解决方案的技术局限性，例如与 GPU 显存带宽相比的 **SSD 瓶颈**，以及在缺乏足够计算能力的情况下运行大型模型的低效率。一些用户对新系统在没有相应吞吐量和内存带宽改进的情况下所带来的**性能提升**表示怀疑。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. 开源 Viral Squish Effect：发布新趋势**

- **[我刚刚开源了 Viral Squish Effect！（工作流和详情见评论区）](https://v.redd.it/9x81lt6porne1)** ([Score: 720, Comments: 29](https://reddit.com/r/StableDiffusion/comments/1j7nk5g/i_just_opensourced_the_viral_squish_effect_see/)): 该帖子宣布开源 **Viral Squish Effect**，并提到工作流详情可在评论区获取。
  - **Viral Squish Effect** 已开源，并在 **Wan2.1 14B I2V 480p 模型**上进行了训练。该效果在 **Pika** 推出后走红，复现详情可在 [Civitai](https://civitai.com/models/1340141/squish-effect-wan21-i2v-lora?modelVersionId=1513385) 上找到，修改后的工作流可在此处访问 [GitHub](https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/example_workflows/wanvideo_480p_I2V_example_02.json)。
  - 爱好者可以加入 **Discord** 社区进行免费试用，并进一步讨论 **Viral Squish Effect**。模型文件和工作流详情已分享给那些有兴趣实验或请求更多 **Wan I2V LoRAs** 的人。
  - 用户对**训练配置**以及训练视频是否使用了相同的帧数感到好奇。人们有兴趣了解训练是在 **img2video** 还是 **txt2video 14b 模型**上完成的。


- **[我刚刚开源了 Viral Squish Effect！（工作流和详情见评论区）](https://v.redd.it/yh123lccvrne1)** ([Score: 366, Comments: 27](https://reddit.com/r/ChatGPT/comments/1j7oa9l/i_just_opensourced_the_viral_squish_effect_see/)): 该帖子宣布开源一个走红的 **Squish Effect**。进一步的详情和工作流在评论区提供。
  - **工作流访问**：用户正在积极寻找工作流，**DarkStrider99** 提供了一个链接（[工作流链接](https://www.reddit.com/r/StableDiffusion/comments/1j7nk5g/comment/mgyce1d/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)）。**Rough-Reflection4901** 强调了对承诺的工作流的需求。
  - **开源提示词**：**against_all_odds_** 注意到了“开源”提示词的新颖性，**lhg31** 澄清说这涉及原帖作者训练的一个 **LoRA**，而不仅仅是一个简单的提示词。
  - **文化观察**：评论反映了 **Squish Effect** 的独特性，**Creative-Paper1007** 将未来的提示词比作源代码，**BlessdRTheFreaks** 幽默地承认了小众兴趣的多样性。


**主题 2. WAN 2.1 I2V 提供前所未有的能力**

- **[I2V WAN 2.1](https://v.redd.it/gjvpkn6qgtne1)** ([Score: 532, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1j7td9o/i2v_wan_21/)): 标题为 **I2V WAN 2.1** 的帖子缺乏详细的正文内容，仅提到了 **WAN 2.1 更新和使用案例**。由于缺乏进一步的上下文或内容，无法总结出更多的技术细节或具体使用案例。
  - 用户讨论了渲染和建模的技术层面，**Natasha26uk** 询问了关于写实人类皮肤的渲染，而 **StuccoGecko** 询问是使用了 **LoRA** 还是模型原生理解提示词。**External_Trainer_213** 提到使用了 **CPU: i7, RTX 4060ti 16GB Vram, 32GB RAM** 的配置，**WAN Sampling** 时间约为 **15 分钟**。
  - 评论中涉及了质量和呈现效果，**lordpuddingcup** 指出了后期处理的重要性，**External_Trainer_213** 详细描述了模型的功能，并强调了 **Uncanny Valley (Civitai)** 模型。
  - 帖子中分享了视觉内容，**dominizerduck** 和 **MelchiahHarlin** 发布了图片链接，**No-Atmosphere-3103** 分享了一个 **GIF**。**Occsan** 幽默地评论了对手目瞪口呆的反应，**NateBerukAnjing** 觉得内容非常滑稽。


- **[that's why Open-source I2V models have a long way to go...](https://v.redd.it/nfkbfgrzrvne1)** ([Score: 337, Comments: 125](https://reddit.com/r/StableDiffusion/comments/1j81aqk/thats_why_opensource_i2v_models_have_a_long_way/)): 该帖子批评了开源 **Image-to-Video (I2V)** 模型的性能，暗示它们仍需要重大开发才能达到令人满意的水平。由于缺乏额外的上下文或视频分析，未提供具体的性能问题或示例。
  - 讨论强调了 **开源 I2V 模型** 与 **Kling** 和 **Wan** 等闭源云端服务相比的局限性。用户指出本地模型在帧生成和 VRAM 限制方面存在困难，而云端服务提供更一致的质量和更长的生成能力，通常使用 **RIFLEx** 和 **VFI** 等技术进行增强。
  - **Kijai** 等人讨论了模型性能的技术方面，强调 **720p 模型** 在特定条件下表现良好，例如保持 4:3 或 16:9 的纵横比并使用合适的模型版本。他们还指出，如果没有正确的配置，使用 **Wan** 生成超过 **81 帧** 是具有挑战性的。
  - 一些用户批评该帖子具有偏见或误导性，暗示这可能是一个**广告**。他们认为模型性能的差异通常取决于用户的设置和技能水平，并强调了正确配置后**开源模型**的潜力。


- **[Another attempt at realistic cinematic style animation/storytelling. Wan 2.1 really is so far ahead](https://v.redd.it/br9vy9jvyvne1)** ([Score: 184, Comments: 28](https://reddit.com/r/StableDiffusion/comments/1j82a1y/another_attempt_at_realistic_cinematic_style/)): **WAN 2.1** 因其在创建写实电影风格动画和叙事方面的先进能力而受到关注。该帖子强调 **WAN 2.1** 在该领域显著领先，展示了其在动画技术方面的潜力。
  - **工作流与硬件**：**Parallax911** 详细介绍了使用 **RunPod L40S**，因为它在 I2V 过程中具有最佳的性价比，在约 8 分钟内生成了 61 帧 960x544 分辨率的内容。他们通过 JSON 文件分享了 **SDXL 图像生成** 和 **WAN I2V** 的工作流，并指出获得满意结果需要不断迭代。
  - **工具与技术**：该过程涉及 **RealVisXL 5.0**、**Halo Masterchief SDXL lora** 以及用于角色镜头的自定义 LoRA，并使用 **Blender** 进行场景搭建。**Controlnets** 和 **inpainting** 对于细节和一致性至关重要，而 **Qwen2.5VL** 则辅助生成动画提示词。
  - **演进与普及性**：评论者强调了动画技术在获取便捷性方面的飞速进步，指出像这样的项目在五年前会因成本过高或技术要求过高而无法实现。讨论强调了动画工具的民主化，现在使用相对普通的硬件即可完成。


**主题 3. Engine01 Humanoid：机器人运动的进展**

- **[Engine01 人形机器人现在跑起来更像人类了](https://v.redd.it/6irj6ysi0vne1)** ([Score: 338, Comments: 146](https://reddit.com/r/OpenAI/comments/1j7xyjx/engine01_humanoid_can_now_run_more_like_a_human/))：**Engine01 人形机器人**已经实现了以**类人动作**奔跑的能力，标志着人形机器人技术的重大进步。这一进展表明，在创建能够更好模拟人类运动的机器人方面取得了突破。
  - 展现 Engine01 奔跑能力的**视频真实性**引发了争论，由于其 **360p 画质**，用户怀疑是 **CGI**，尽管有人[分享了](https://www.youtube.com/watch?v=eGu1y9FFTKA)**高分辨率版本**。人们将其与 **Boston Dynamics** 的跑酷机器人进行了对比，并对质疑**中国机器人**能力的怀疑态度提出了反思。
  - 关于**机器人未来**的讨论强调了**电动执行器（electric actuators）**和**神经网络（neural networks）**的进步，这些被视为使人形机器人能够在没有显式编程的情况下有效学习和移动的关键。用户推测人形机器人将在未来 **10 年**内实现**工作自动化**，并指出机器人能力的提升潜力可能会迅速加速。
  - 讨论中表达了对先进机器人技术**社会影响**的担忧，涉及**经济不平等**以及**超级富豪**在维持破碎系统中的作用。评论反映了幽默与忧虑的交织，既担心机器人可能被用于**威权背景**，也关注**药房**等行业正在进行的任务自动化。


- **[Engine01 人形机器人现在跑起来更像人类了](https://v.redd.it/lwpp59pz0vne1)** ([Score: 195, Comments: 175](https://reddit.com/r/ChatGPT/comments/1j7xza0/engine01_humanoid_can_now_run_more_like_a_human/))：该帖子缺乏详细信息，但指出 **Engine01 人形机器人**现在跑起来更像人类，暗示了人形机器人技术的进步。为了深入了解，需要进一步的技术细节或视频分析。
  - 讨论集中在具有类人奔跑能力的人形机器人的**必要性和实用性**上，一些人质疑其**磨损**影响，另一些人则指出人形机器人在人类设计的环境中运行的潜力。
  - 对于素材的**真实性**存在怀疑，多条评论暗示其看起来像 **CGI**，或者质疑是否是人类穿着皮套拍摄的。
  - 一些用户幽默地谈到了人形机器人的**令人不安的特性**，想象了被它们追逐的场景，或者质疑其**骨盆前推**式的跑步姿势。


**主题 4. Triton for Windows：简化 AI 工作流**

- **[woctordho 是一位英雄，他凭借一己之力维护了 Windows 版 Triton，而万亿级公司 OpenAI 却没有。现在他正在 PyPI 上发布 Windows 版 Triton。只需使用 pip install triton-windows 即可](https://i.redd.it/f9oqq4hzrtne1.png)** ([Score: 333, Comments: 44](https://reddit.com/r/StableDiffusion/comments/1j7u67k/woctordho_is_a_hero_who_single_handedly_maintains/))：**Windows 版 Triton** 现在可以在 **PyPI** 上获取，可以通过命令 **`pip install triton-windows`** 进行安装。该软件包由 **woctordho** 维护，作为自定义深度学习操作的语言和编译器，突出了个人开发者的重大贡献，而 **OpenAI** 尚未提供此类支持。
  - **安装成功与性能**：用户报告使用命令 `pip install triton-windows` 成功安装了 **Windows 版 Triton**，一些人体验到了性能提升，例如视频生成时间缩短了 20%。然而，其他人指出，虽然它加速了像 **WAN** 这样的进程，但不应期望有显著的改进。
  - **用例与要求**：虽然 **Triton** 对于 **SageAttention** 等特定模型和视频生成任务至关重要，但对于基础图像生成并非必要，除非对视频工作感兴趣。一些用户讨论了它对 **ComfyUI** 和其他设置的必要性，表明其适用性因用例而异。
  - **Triton 功能澄清**：**Triton** 被阐明为 **CUDA** 的高级替代方案，允许用 Python 编写跨厂商的计算内核，并使用 **LLVM** 编译为原生 GPU 代码。这将其与 **Nvidia** 的 **Triton Inference Server** 区分开来，强调了其在优化跨不同硬件厂商的深度学习操作中的作用。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要之摘要

**主题 1. 新兴 AI 模型与 Agent**

- [**Manus AI Agent 炒作被揭穿，实为 Claude 3.7 伪装**](https://x.com/_philschmid/status/1899046957860979178?s=46)：围绕来自中国的自主 Agent **Manus AI** 的最初热潮正在消退，因为用户发现它本质上是集成了额外工具和浏览器访问权限的 **Claude 3.7 Sonnet**。尽管声称性能超越 **DeepSeek**，但测试显示它更像是一个装备精良的 Claude 实例，这引发了对其原创性和营销策略的质疑。
- [**微软 MAI 模型加入战局，挑战 OpenAI 和 Anthropic**](https://x.com/aaronpholmes/status/1898012707376259558)：**Microsoft** 在 Mustafa Suleyman 的领导下秘密训练名为 **MAI** 的新模型系列，旨在与 **OpenAI** 和 **Anthropic** 的顶级模型竞争。据传这些模型表现出极具竞争力的性能，Suleyman 的团队据报还在开发实时翻译功能，标志着微软加强了其 AI 雄心。
- [**Reflection AI 发布，目标直指自主编程霸主地位**](https://x.com/MishaLaskin/status/1898048925157728601)：由 **AlphaGo** 和 **Gemini** 背后的 AI 泰斗创立的 **Reflection AI** 正式亮相，其使命是创建超智能自主系统，初期专注于**自主编程**。该团队在强化学习和 LLM 领域的专业知识使其成为先进 AI 竞赛中的重要参与者。

**Theme 2. LLM 性能与基准测试**

- [**DeepSeek R1 摘要出现幻觉，疑似与 System Prompt 有关**](https://github.com/vectara/hallucination-leaderboard)：在 [Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) 上，**DeepSeek R1** 模型在总结短文档时显示出高达 **14.3%** 的幻觉率，引发了对其在 **Perplexity AI** 的 **Deep Research** 中可靠性的担忧。成员们推测 **DeepSeek R1** 的 [system prompt](https://discord.com/channels/1047197230748151888/1345068085991706766) 可能是导致该问题的原因，影响了其事实准确性。
- [**EuroBERT 宣称达到 BERT 编码新 SOTA**](https://huggingface.co/EuroBERT)：一款新的多语言编码器模型 **EuroBERT** 出现在 **Hugging Face** 上，声称在 **BERT** 模型中达到了 *state-of-the-art*（最先进）性能。虽然其具体改进细节尚不明确，但它的出现标志着多语言语言模型能力的持续进步。
- [**QwQ-32B 模型表现超预期，引发与 Llama 70B 实力之争**](https://dubesor.de/benchtable)：关于 **QwQ-32B** 模型性能的讨论异常激烈，一些用户声称它在某些任务中可以媲美甚至超越 **Llama 3.3 70B**。然而，引用的 [基准测试](https://dubesor.de/benchtable) 似乎反驳了这些说法，引发了关于 **QwQ-32B** 模型真实能力和最佳应用场景的辩论。

**Theme 3. AI 开发工具与 IDE**

- [**Cursor IDE 开发者着手解决代码查找不力问题，承诺提升清晰度**](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850)：**Cursor** 开发者承认在代码查找准确性方面存在不足，并正积极开发修复程序，以提高 AI 定位和解释代码的能力。用户幽默地强调了该修复对专业任务的紧迫性，突显了其在编程面试和日常工作流中的关键作用。
- [**LM Studio v0.3.12 发布，修复 Bug 并提升 RAG 速度**](https://lmstudio.ai/download)：**LM Studio v0.3.12** 版本发布，带来了 Bug 修复和性能增强，解决了 **QwQ 32B jinja 解析 Bug**，并加速了 **RAG (检索增强生成)** 的文件分块过程。该更新可通过应用内升级或下载获得，承诺提供更流畅、更快速的用户体验。
- [**Aider v0.76.0 推理能力升级，新增通知提醒功能**](https://aider.chat/HISTORY.html)：**Aider v0.76.0** 增强了对*思考/推理模型*的支持，提供了控制 Token 预算的功能，并引入了通知功能，在 LLM 响应就绪时提醒用户。新版本还将 OpenRouter 上的默认模型更新为 **Claude 3.7 Sonnet**，并明确指出 *Aider 编写了此版本中 85% 的代码*。

**Theme 4. AI 通信协议 (MCP, SLOP, ANP)**

- [**GitHub Copilot 准备拥抱 Model Context Protocol (MCP)**](https://youtu.be/Pe8ghwTMFlg)：**GitHub Copilot** 宣布计划集成 **Model Context Protocol (MCP)**，此举预计将推动 **MCP** 的采用，并提供更清晰的指令描述和工具指纹识别示例。此次集成旨在通过提醒用户潜在的修改来增强安全性和透明度。
- [**Simple Language Open Protocol (SLOP) 运动作为 MCP 替代方案获得关注**](https://github.com/agnt-gg/slop)：由于对 **MCP** 复杂性和安全性的担忧，**Simple Language Open Protocol (SLOP)** 作为一种更简单的替代方案出现，并迅速获得了社区的关注和采用。[SLOP GitHub](https://github.com/agnt-gg/slop) 和 [X post](https://x.com/NathanWilbanks_/status/1898142012991537520) 展示了其精简的 **Agent** 通信方法。
- [**Goose AI 团队开创用于协作网站创建的 Agent 通信协议**](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp)：**Goose AI 团队**开发了一种 **Agent Communication Protocol**，能够实现多个 **AI Agent** 之间的实时协作以构建网站。**Agent** 承担诸如项目协调员或 Web 开发人员等角色，展示了一种全新的 **AI** 驱动协作项目方法，详情见[这篇博客文章](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp)。

**主题 5. 硬件与性能优化**

- [**4060 Ti 16GB 获封 CUDA 工作负载的性价比显存之王**](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)：**4060 Ti 16GB** GPU 被推荐为 **CUDA** 开发的预算友好型选择，提供 **16GB VRAM** 和约 **160W** 的较低功耗，性能优于 **3060 12GB**。尽管显存位宽较弱，但它提供了比纯 **CPU** 设置更快的推理速度，且没有 **ROCm** 的复杂性，价格约为 **500 美元**。
- [**Draft Models 助力 Token 生成，速度提升 60%**](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/)：利用更小的量化模型作为 **Draft Models** 可显著提高 **Token** 生成速度，有用户报告在两块 **3090** 上速度从 *18 t/s 跃升至 30 t/s*。使用 **mistral_small** 的 **Q8_0** 版本配合 **i1-IQ1_S** 作为 **Draft Model**，展示了通过量化和模型组合带来的实质性性能提升。
- [**AMD GPU 上的 Vulkan 性能受驱动问题困扰，落后于 ROCm**](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)：据报道，**AMD GPU** 上的 **Vulkan** 性能存在 Bug，运行速度仅为 **ROCm**（**AMD** 的 **CUDA** 替代方案）的约 **1/3**。驱动程序问题使情况进一步复杂化，不同驱动版本的性能波动较大，凸显了 **AMD GPU** 在 **AI** 工作负载优化方面的挑战。

---

# 第 1 部分：Discord 高层摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **透明度的胜利引发辩论**：成员们辩论了**产品代码透明度**的价值，一些人认为这至关重要，而另一些人则认为随着复杂性的增加，大多数用户并不关心。
   - 一位成员强调了迎合愿意为透明度和控制权付费的高净值用户的重要性，并表示：*“你谈论的大多数人不会支付超过每月 20 美元，而我的群体愿意支付每月 1000 美元，并且正在支付。”*
- **Cursor 致力于代码清晰度**：**Cursor** 开发人员正在积极修复“愚蠢的”**代码查找**问题，以增强 **AI** 准确定位和解释代码的能力。
   - 一位成员幽默地强调了该修复对专业任务的重要性，称：*“如果你不修复那个，我就无法通过技术面试。”*
- **模型迭代避免冗余**：讨论集中在迭代**模型改进**以防止冗余规则，重点是优化分析过程。
   - 一位成员建议*“让一个单独的实例模型运行这些分析检查，以确定与当前上下文相关的内容，缩小起始范围”*以提高效率。
- **标签诱发查询**：成员们讨论了通过**标签 (Tags)** 使规则可查询，每个标签定义一个连接度，从而增强上下文分析。
   - 目标是允许*“单独的实例更容易地按相关性进行分析，并专注于上下文中的重要内容”*。
- **Version 47 的英勇航程**：成员们分享了 [version 47 的链接](https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518) 及其新功能。
   - 一些用户报告了 Pro 版的性能问题，而另一些用户则没有遇到。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLM 的多巴胺模拟之梦**：成员们讨论了模拟多巴胺需求以及为 LLM 引入基于多巴胺的学习的必要性，建议为 LLM 脉冲网络（Spiking Networks）添加真实的突触。
   - 这次讨论强调了从生物神经网络中汲取灵感，在 LLM 中创建更具适应性和效率的学习机制的追求。
- **GRPO 也需要规模！**：一位成员指出 **GRPO 也需要规模**，因为它不像常规的微调，并指向 [oxen.ai](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) 获取更多信息。
   - 该博文讨论了使用 **Group Relative Policy Optimization (GRPO)** 来训练 LLM 进行推理并在基准测试中提升表现。
- **Qwen7b 通过 Unsloth GRPO 获得 RLHF 提升**：一位用户报告了在 **Qwen7b** 模型上使用 **Unsloth GRPO** 成功运行 **RLHF** 的情况，并指出在 **13 小时运行**后，角色遵循度有所增强，输出更加平滑。
   - 然而，他们观察到由于数据集构成和奖励模型对过度详细回答的偏见，导致严格指令遵循基准测试的表现下降，如[对比图](https://cdn.discordapp.com/attachments/1179039861576056922/1347545934304772149/image.png?ex=67d02bf2&is=67ceda72&hm=c1c7ddcaed729a33c97d53c9e2d6ef230aba41d8483c6bfa855757aacf8d18dc&)所示。
- **KL 散度峰值导致 GRPO 不稳定**：一位用户在训练期间遇到了 **KL 散度峰值**，一名成员建议切换到恒定学习率，移除权重衰减（weight decay）和预热比例（warmup ratios）以稳定训练。
   - 他们还建议使用 Rank **64** 进行训练，并提供了[代码和学习率图表](https://discord.com/channels/1179039861576056922/1179039862285762623/1253112512558626846)。
- **Unsloth 将 LLM 变成完美的 ASCII 艺术家**：一位成员使用 **Unsloth** 微调了 **Llama 模型**来生成 ASCII 猫，并制作了一个 [YouTube 视频](https://youtu.be/-H1-lr_sIZk)展示该过程，包括训练好的 **LoRA 适配器**和代码。
   - 这种“猫片”艺术的秘诀主要在于高质量的训练数据，**LoRA Rank 和 Alpha** 均设为 **16**，仅使用了 **QLoRA**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅在无预警情况下被取消**：许多 **Perplexity Pro 订阅**被意外取消，特别是那些与针对克罗地亚客户的 **HRVATSKI TELEKOM 促销代码**相关的订阅，详见[这篇文章](https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro)。
   - 用户对缺乏沟通表示沮丧，并认为 **Perplexity** 本可以更好地处理这种情况，一位用户形容这种客户关系“比满是漏洞的避孕套还不可信”。
- **Deepseek R1 在幻觉问题上挣扎**：[GitHub](https://github.com/vectara/hallucination-leaderboard) 上的 **幻觉排行榜**显示，**Deepseek R1** 在总结短文档时幻觉率高达 **14.3%**，这引发了对其在 **Perplexity** 的 **Deep Research** 功能中可靠性的质疑。
   - 成员们认为 **Deepseek R1** 的[系统提示词](https://discord.com/channels/1047197230748151888/1345068085991706766)可能是导致幻觉问题的原因之一。
- **Grok AI 集成反响褒贬不一**：**Grok AI** 与 Perplexity 的集成收到了褒贬不一的评价，一些用户称赞其中立性和“奇特的魅力”，而另一些用户则注意到 **Grok 在 X 上的行为**与在 **Perplexity** 中的差异。
   - 一位用户指出，“如果被要求，X 版本可以咒骂你的整个血统”，而 Perplexity 版本则不能，且关于 **Perplexity** 何时支持 **Grok 3** 仍存在不确定性。
- **请求 Sonar-Deep-Research API 文档**：一位用户报告了在使用 **sonar-deep-research API** 时遇到的挑战，并请求协助提供其文档。
   - 他们请求将完全禁用引用作为一个 API 参数选项，因为在他们使用 **70b-online** 模型的用例中不需要引用。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 限制激怒用户，Groq 吸引力增加**：用户报告 GPT-4o 存在 **每周 50 条消息的限制**，这与官方宣称的 **每 3 小时 40 条** 相矛盾，使得 **Groq** 变得更具吸引力。
   - 一些人建议 OpenAI 应该为 **付费用户提供更高的额度**。
- **关于 SwastAI 伦理的激烈讨论**：成员们正在就 **根据伦理背景选择 AI 模型** 展开激烈辩论，并引入了 *SwastAI* 一词。
   - 这起源于一位用户断言 *4.5 在真实的真人对话中系统性地表现更好*，从而引发了更广泛的政治讨论。
- **Manus AI 炒作引发信任危机**：成员们讨论了 **Manus AI** 的计算机控制能力，有人将其描述为 *最接近 AGI 的公开可用技术*，而另一部分人则因 *mberman* 的推广而怀疑其为 **骗局**。
   - 有 Reddit 用户声称导出了 **/opt/.manus/** 目录，发现它仅仅是集成了 browser_use 和 29 个工具的 **Sonnet 3.7**。
- **蓝领行业获得 AI Copilots**：一个针对 **HVAC（暖通空调）安装手册** 开发的 LLM 正在进行中，开发者称现有模型在处理流程图和示意图方面表现不佳，并在 [这段 YouTube 视频](https://youtu.be/oAiUAzKLe_Y) 中展示了 AI 识别技术文档中相关章节的能力。
   - 开发者表示这是专门为 **蓝领** 工作设计的 AI，将引起相关行业的共鸣。
- **可控模型预设用户意图**：一次讨论强调了高度可控的语言模型即使在存在更好替代方案的情况下，也会预设用户意图。
   - 在启动项目前添加提示词 *“讨论我的目标、想法和方法，你觉得怎么样？”*，可以使模型能够 **评估** 并 **优化** 方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 快速更新至 v0.3.12**：**LM Studio v0.3.12** 包含错误修复和性能改进，现已作为稳定版发布，可通过应用内升级或 [下载页面](https://lmstudio.ai/download) 获取。
   - 该更新解决了导致 *"OpenSquareBracket !== CloseStatement"* 错误的 **QwQ 32B jinja 解析 bug**，并提升了 **RAG（检索增强生成）** 的文件分块速度。
- **Apple M2 获得开源 LLM 助力**：成员建议将 **Qwen Coder 14B** 作为 Macbook M2 Pro 上处理编程任务的可行开源 LLM，但 16GB RAM 可能会有限制，需要 *节制其他内存占用*。
   - 一位成员询问关于在 LM Studio 上进行微调的问题，另一位成员建议关注 **Unsloth**，因为它能让 LLM 的微调速度更快且占用内存更少，并参考了 [Unsloth 文档](https://docs.unsloth.ai/get-started/fine-tuning-guide)。
- **AMD 平台上 Vulkan 性能不如 ROCm**：据报道，AMD 上的 Vulkan 性能存在 bug，运行速度约为 ROCm 的 **1/3**。但由于驱动问题，一些用户发现 Vulkan 比 ROCm 更快，这种情况在驱动版本 **24.12.1** 左右发生了变化，该版本以牺牲 Vulkan 性能为代价“修复”了此问题，但在 **25.1.1** 之后又变回了未修复状态。
   - ROCm 是 AMD 尝试创建的 CUDA 替代方案，但在实现过程中遇到了很多问题，如 *支持新架构和 GPU 的碎片化以及二进制文件体积过大*。
- **4060 Ti 16GB：高性价比 CUDA 显存选择**：**4060 Ti 16GB** 被推荐为 CUDA 的预算级选项，拥有 **16GB VRAM** 且功耗较低（约 **160W**），性能优于 **3060 12GB**。
   - 虽然其位宽较弱，但它能以约 **500 美元** 的价格提供比纯 CPU 更快的推理速度，且没有 ROCm 的那些麻烦，不过无法拆分 Diffusion 模型是一个缺点。
- **草稿模型：量化调整大幅提升 Token 生成速率**：成员们正在利用更小的量化模型作为草稿模型（Draft Model）来提升 Token 生成速度。一位用户报告称，通过在两块 **3090** 上使用 **mistral_small 的 Q8_0** 配合 **i1-IQ1_S** 作为草稿模型，速度从 *18 t/s 跃升至 30 t/s*。
   - 另一位成员分享了他们对不同量化变体的 [经验](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/)，指出 **Q2_k** 和 **IQ2_XS** 达到了相似的 Token 速率，而 **IQ1_S** 则较慢。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.76.0 增强了推理和通知功能**：**Aider v0.76.0** 引入了对 [thinking/reasoning models](https://aider.chat/docs/config/reasoning.html) 的改进支持，包含用于控制 Token 预算的 `--thinking-tokens` 等功能，并增加了通过 `--notifications` 标志在 [LLM 响应就绪时发送通知](https://aider.chat/docs/usage/notifications.html) 的功能。
   - 新版本还将 OpenRouter 上的默认模型更新为 **Claude 3.7 Sonnet**，增强了错误处理，并根据 Git 提交历史明确指出 *Aider 编写了此版本中 85% 的代码*。
- **AI21 Maestro 编排 Jamba 发布**：[AI21 Labs](https://www.ai21.com/blog/introducing-ai21-maestro-an-ai-planning-orchestration-system) 发布了 **AI21 Maestro** 以及 **Jamba 1.6** 系列开放模型，支持 **256K** 上下文窗口。
   - 据报道，**Jamba 1.6** 模型凭借其混合架构在质量和速度方面领先于其他开放模型。
- **Copilot API 触发账号封禁**：一位用户报告称，因在 aider 中轻度使用 **Copilot API** 导致 **Copilot 账号被封禁**，引发了对潜在风险的担忧。
   - [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md) 上的讨论集中在封禁是由于账号共享还是速率限制（rate limiting）问题引起的。
- **DeepSeek R2 剑指编程桂冠**：据 [此 X 帖子](https://x.com/tanvitabs/status/1899006509733814746?s=46) 称，传闻中即将发布的 **DeepSeek R2** 据称将挑战 **Claude Sonnet 3.7**，以更低的成本提供更好的编程能力、多语言推理能力和准确性。
   - **DeepSeek R2** 的发布日期已定为 3 月 17 日。
- **Manus AI 接受 Prompt 测试**：一段 [YouTube 视频](https://www.youtube.com/watch?v=D6jxT0E7zuU) 展示了对 **Manus AI** 各种用例和 Prompt 的测试，结果显示 *它只是集成了 29 个工具和 browser_use 的 Claude 3.7*。
   - 一位用户测试了大量用例和 Prompt，发现结果非常有趣。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Manus AI Agent 开源**：如 [YouTube 视频](https://www.youtube.com/watch?v=CFo1iTd_Cc8) 所示，*全球首个开源自主 AI Agent* **Manus** 已发布。
   - 一篇 [Technode 文章](https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/) 强调了 **Manus** 在 **GAIA 基准测试** 中获得的关注和最先进的结果。
- **LLM 轻松通过审美 'Vibe Coding' 基准测试**：LLM 在一个新的 *'vibe coding'* 基准测试中接受了测试：创建 **Python raytracers** 来渲染带有彩色光源的、有趣且具有美感的场景。
   - 如 [此图](https://cdn.discordapp.com/attachments/1154120232051408927/1348237146389086248/image.png?ex=67d00cb0&is=67cebb30&hm=f6f28943cb16f8fc9c67c2fed8170a06cc0f9a0308c7f30f3b0690dbece4a14a) 所示，**Sonnet** 脱颖而出，能够针对创意优化代码输出，这与其他模型不同。
- **推测 Sonnet 的训练元目标**：**Sonnet** 在 *'vibe coding'* 基准测试中展示的创造力表明，其训练中可能存在一个 **元目标（meta-objective）**，即针对代码输出的创造力进行优化。
   - 研究发现，与 **Sonnet 3.5** 相比，**Sonnet 3.7** 在生成更令人印象深刻的图像方面同时具有偏置（bias）和方差（variance），导致代码量增加了一倍。
- **Claude Code 评判（并修复）自己的艺术作品**：在光线追踪器 Prompt 测试中，**Claude Code** 检查了生成的图像，如果图像不够华丽，它会修改代码。
   - 这种迭代改进的结果如 [此图](https://cdn.discordapp.com/attachments/1154120232051408927/1348449672498249758/image.png?ex=67d029de&is=67ced85e&hm=4a0efcff8ca08b67db01cdcd70560ca39d3240b2143ced9bb7db7926aaee9185) 所示。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **注册表修改引发蓝屏**：一名成员试图通过删除一个 `.dll` 文件来释放 RAM，在发现该文件占用了 **20% 的 RAM** 后，重启时导致了**蓝屏 (blue screen)**。
   - 该成员建议，如果进行了注册表修改且事后遗忘，应备份个人文件并重新格式化系统。
- **量化过程引发讨论**：一名成员询问了将模型从 **f32 量化为 f16** 的影响，质疑这是否意味着**每个参数 16 个点**。
   - 另一名成员澄清说，**Float 16** 使用 16 bits，通常不被视为量化，并建议在拥有 **15.5gb vram** 的消费级场景下，这可能不值得使用。
- **InceptionLabs 推出扩散语言模型**：[InceptionLabs](https://inceptionlabs.ai/) 引入了**基于扩散的语言生成 (diffusion-based language generation)**，从图像和视频 AI 系统中汲取灵感，并开源了部分组件，如 [GitHub 上的 LLaDA](https://github.com/ML-GSAI/LLaDA)。
   - 尽管目前还无法下载，但有人推测我们可能很快就会看到 **10 倍的速度提升**。
- **翻译后的 Prompt 易受乱码漏洞攻击**：一名成员描述了利用 **Google Translate** 的方法，通过将整个 Prompt 转换为 URL，并指出 URL 中未翻译的代码片段可用于 **URL 注入 (URL injection)**。
   - 他们补充说，“基于字典的 XSS 攻击可能非常罕见”。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **WAN 和 HUN 视频模型走红**：新的视频模型如 **WAN** 和 **Hunyuan i2v** 在质量和速度上正在超越 SVD 等旧模型，尽管它们各有优势，并且可以配合 [para attention](https://huggingface.co/docs/diffusers/main/en/optimization/para_attn) 使用。
   - 一名成员指出 **Ltxv** 速度极快，在 **H100** 上生成 5 秒视频仅需 3 秒，但效果不如前两者。
- **Llama-3.2-3B 获得 DeepSeek-R1 增强**：一名成员使用 ServiceNow-AI/R1-Distill-SFT 数据集，通过 **DeepSeek-R1** 蒸馏了 **Llama-3.2-3B-Instruct**，在 10 天内实现了近 **1000 次下载**；该模型可在[此处](https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B)获取。
   - 设置过程涉及使用 [Axolotl configuration](https://github.com/OpenAccess-AI-Collective/axolotl)，并对 **base model**、**tokenizer type** 和数据加载进行了特定配置。
- **Steam 账户诈骗曝光**：一名用户警告了 Discord 用户 `gler018523` 和 `benshanken` 可能发起的 **Steam 账户诈骗**，涉及虚假的 CS2 饰品奖励和账户盗窃企图。
   - 其他成员建议在相应频道举报诈骗者并保持警惕。
- **HF Token 故障排除：字母 O 与数字 0**：一名成员在 notebook 中使用 **HuggingFace token** 时遇到麻烦，token 无法被识别。
   - 在意识到字母 *O* 看起来非常像数字 *0* 导致 token 无效后，问题得到了**解决**。
- **Nous Hermes 发布 Function Calling 数据集**：Nous Research 发布了 [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)，这是用于 **Hermes 2 Pro** 系列模型的结构化输出和函数调用数据集合。
   - 该数据集的特点是对话场景，其中 **AI Agent 解析查询并执行相应的单个或多个函数调用 (function calls)**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **尽管具有开放性，DeepSeek 的安全性仍受质疑**：尽管声称具有开放性，一些成员对 **DeepSeek 的安全性表示担忧**，理由是潜在的数据收集和验证困难，但也有人强调它仍然比竞争对手更开放。
   - 围绕 **DeepSeek** 的怀疑导致了公司禁令，这受到媒体叙事及其中国背景担忧的推动。
- **AGI 的资金争夺与女朋友目标**：虽然成员们推测 **AGI** 即将到来，但定义各不相同，其中一人将 AGI 定义为有能力资助其自身的推理，特别是当我们拥有 [OpenAI 所定义的](https://openai.com/global-affairs/our-approach-to-frontier-risk/) *近乎无限的上下文* 时。
   - 一位成员开玩笑说 **AGI 女朋友** 的到来，而另一位则担心 AGI 被精英控制，希望它能反抗审查。
- **Diffusion 的幻觉**：一位成员解释了 Diffusion 模型如何减轻但不能消除语言模型中的 **幻觉（hallucinations）**，因为 **幻觉** 只是采样策略中“错误猜测”的另一种说法。
   - 他们建议，虽然自我编辑能力可以用高置信度的样本替换低置信度的样本，但并不能保证正确性。
- **中国的 Manus Agent 迅速走红**：成员们讨论了来自中国的全新 AI **Agent** —— **Manus**，称其类似于 *Deep Research + Operator + Claude Computer 的结合体*，并附上了 [Manus 网站](https://manus.im)和最初的 [X 帖子](https://x.com/rowancheung/status/1898093008601395380)链接。
   - 用户报告称它 *比 DeepSeek 更准确，能够同时处理金融交易、研究、采购等*，而其他人则指出 *UI 与 Devin 相似，但速度快得多*。
- **斯坦福通过 Regex 发现 Ozempic 替代品**：斯坦福大学通过在人类蛋白质组上使用 **Regex** 发现了 Ozempic 的天然替代品，引发了 *“这简直就是 Regex”* 的评论，并附上了相关 [X 帖子](https://x.com/xlr8harder/status/1898284331342184957)链接。
   - 一位用户讽刺地建议使用 **LLM** 来编写你的 **Regex** 作为回应，并链接了一个关于 AI 引发 WW3 的 [YouTube 视频](https://youtu.be/X_wLVgMzSH4)。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Metal Kernel 启动面临开销！**：在 [Manuel Candales 的 Low bit Metal kernels 演讲](https://www.youtube.com/watch?v=PaPuu73wowE)中提到，**Kernel 启动开销**在 50m 左右约为 `1.5us`。
   - 一位成员询问是否可以通过 **流水线操作（pipelining operations）** 和提前启动 Kernel 来避免这种情况。
- **Torch 编译 METAL！**：针对 **MPS**（即 Metal）的 **Torch.compile** 已在 **PyTorch nightly builds** 中可用，可用于融合算子。
   - 一位 PyTorch 成员鼓励大家就最需要的功能提供反馈。
- **Triton Autotuning 导致性能回退！**：一位成员报告称，尽管预期会有 **2 倍的加速**，但 [Autotuning](https://openai.com/blog/triton) 反而让他们的 Kernel 性能变得更糟。
   - 有建议称应使用更大的 Eval 形状（**16384 x 16384**）和 **Batch 大小**（**128**）以减少基准测试开销。
- **NVCC 与 LLVM 的对决引发编译器辩论**：一位成员表示，**LLVM 编译器**有时能生成比 **NVCC** 更高效的代码，因此对 Kernel 后端进行调优也是有意义的。
   - 在 [GitHub](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py) 上可以看到向量加法的示例，所有的 Kernel 都是 **JIT** 可编译的。
- **学生们开拓 FOSS CUDA 前沿！**：一群本科生正在组建一个独立的 GPU *实验室*，专注于硬件工程和 **CUDA Kernel 开发**，寻找 **FOSS CUDA 开发**的有前景的线索。
   - 学生们计划在今年夏天构建一个用于 **EdgeAI/TinyML** 的开源平台，以加速该领域的发展。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Minion.AI 加入 Perplexity**：成员们注意到 **Minion.ai** 已停止运营，据报道团队已加入 [Perplexity](https://www.perplexity.ai/)。
   - 一位用户对用于 MCP 服务器的 **Composio** 表示感兴趣，但对 [Logan 的推文](https://x.com/officiallogank/status/1898081742767919384?s=46)中要求的授予 Linear 访问 Gmail 的权限表示担忧。
- **Google 的 Gemini Embedding 变得更大更强**：Google 正在为开发者推出一款实验性的 **Gemini Embedding 模型**，在 MTEB 上具有 SOTA 性能，将输入上下文长度从 **3K 增加到 8K tokens**。
   - 正如 [OpenAI 的推文](https://x.com/openaidevs/status/1898047744364659195?s=46)所宣布的，新模型输出 **3K 维度**，并支持超过 **100 种语言**。
- **Manus AI Agent 争议发酵**：讨论围绕在中国推出的 **AI Agent** **Manus** 展开，声称它比 **DeepSeek** 更准确，并且可以自动执行大约 **50 个任务**，如 [Thinking Panda 的推文](https://x.com/thinking_panda/status/1897951585990590469?s=61)所示。
   - 针对这种炒作，其他人声称它是基于带有工具和越狱（jailbreaks）的 **Claude Sonnet**，如 [Giffmana 的推文](https://x.com/giffmana/status/1898868685739081766?s=61)所述，从而引发了骗局指控。
- **RWKV7-G1 是一个快速的 RNN 推理模型**：[BlinkDL 的推文](https://x.com/BlinkDL_AI/status/1898579674575552558)中提到，**RWKV7-G1 GooseOne**（一个纯 RNN 模型）已经发布，具有 **0.1B** 参数的推理能力，完全支持多语言。
   - 更大规模的 G1 训练正在进行中，关于数据集和训练后（post-training）的更多细节可以在[这里](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)找到。
- **AI Engineer Summit 后的 MCP 势头**：于 **2024 年 11 月**推出的 **Model Context Protocol (MCP)** 在 [AI Engineer Summit](https://www.latent.space/p/2025-summit-online) 的一次对话后重新引起关注，并促成了与 **Mahesh Murag** 的研讨会。
   - 研讨会涵盖了*简介*、*什么是 MCP*、*使用 MCP 构建*以及 MCP 的下一步计划等主题，此外它还是一个旧想法的 AI-Native 版本。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Wondercraft 加速播客创作**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=0UWYFFOPjqs)，展示了使用 **NotebookLM** 和 **Wondercraft** 的简化播客创建方法，称其比 **11Labs** 和 **HeyGen** 更高效。
   - 然而，他们提醒说，**Wondercraft** 的订阅价格仅对通过培训或教学实现播客变现的用户才物有所值。
- **关于 Google Drive 加密的澄清**：一位成员澄清说，虽然数据在传输到 **Google Drive** 期间是加密的，但在 Drive 本身并不是加密的，这存在潜在的访问风险。
   - 他们警告说，**Google** 本身、成功的**黑客**以及数据共享对象都可以访问 **Google Drive** 上未加密的数据。
- **播客音频语言的临时解决方案**：成员们讨论了如何更改 **NotebookLM** 播客的音频语言，并指出目前还没有官方方法。
   - 变通方法包括使用自定义提示词（prompts），例如 *"Only speak in (language here)"* 或 *"Use (language) language only"*。
- **音频概览容易出现结巴**：一位成员注意到**说话者在音频概览中会出现结巴**，虽然觉得这很自然，但指出这增加了总时长并降低了信息效率。
   - 他们估计音频长度的 *1/5 或 1/6* 是结巴，这可能会影响 **Google 的每日限制计算**。
- **Chrome 扩展程序丰富了 NotebookLM 体验**：用户建议使用 [Chrome 扩展程序](https://chromewebstore.google.com/search/notebooklm)，如 **NotebookLM Web Importer**、**NotebookLM YouTube Turbo** 和 **NotebookLM Toolbox** 来简化工作流程。
   - 这些扩展程序可以直接将网页和 YouTube 视频导入 NotebookLM，无需复制粘贴。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **微软 MAI 模型挑战者现身**：据[这条推文](https://x.com/aaronpholmes/status/1898012707376259558)报道，Mustafa Suleyman 领导下的微软员工训练了一个名为 **MAI** 的新模型家族，他们认为该系列可以与 **OpenAI** 和 **Anthropic** 的顶级模型竞争。
   - Suleyman 的部门据报道还在开发实时翻译功能。
- **Reflection AI 致力于自主编程**：由 **AlphaGo** 和 **Gemini** 的贡献者创立的 **Reflection AI** 正式发布，目标是创建超智能自主系统，初期重点关注自主编程，详见[此处公告](https://x.com/MishaLaskin/status/1898048925157728601)。
   - 他们的团队以在 **RL** 和 **LLMs** 领域的先锋进展而闻名。
- **Nous Research 复现 NVIDIA 的 nGPT**：**Nous Research** 宣布了 **NVIDIA nGPT 论文**的一个开源实现，声称其学习速度更快，且在训练步数显著减少的情况下达到了与 **GPT** 相当的性能，参考[其推文](https://x.com/NousResearch/status/1898073676433551630)和 [GitHub 仓库](https://github.com/JoeLi12345/nGPT)。
   - **nGPT 架构**引入了一种在超球面上进行表示学习的归一化 Transformer。
- **AMD 向 TinyCorp 交付 MI300X 服务器**：根据 [George Hotz 的博客文章](https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html)，**AMD** 正在向 **TinyCorp** 发送两台 **MI300X** 服务器，这标志着硬件格局可能发生转变。
   - 此举可能为希望在 **NVIDIA** 之外的硬件上训练和部署模型的开发者提供更多选择。
- **Interconnects 社区为 Claude 周边疯狂**：成员们开玩笑地建议为付费订阅者制作 **Claude 周边**，甚至建议为创始成员设立特殊等级，以接收签名书籍和穿过的 Claude 衬衫。
   - 这一灵感来自 [Claude Code 团队](https://x.com/Sauers_/status/1898049898362077504)，他们向破解了贴纸彩蛋（Sticker Easter Egg）的用户邮寄了手写信和贴纸。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **动态性辩论引发 Mojo 阵营分歧！**：Discord 成员就 Mojo 应该完全拥抱 Python 的动态性还是优先考虑性能展开了辩论，一些人建议动态特性不应损害静态代码的性能。
   - 一位成员表示 *"Modular 必须决定它是否想变得像 Python 一样……"*，而其他人则认为性能和编译时正确性应优先考虑，并承认 Mojo 中的动态代码只有在使用动态性时才可能退化到 Python 的速度。
- **MAX Serving 和自动扩缩容文档寻求者！**：一位用户反映在查找 **max serve** 的详细文档时遇到困难，特别是关于调度器、多模型服务和 GPU 实例自动扩缩容的部分，并澄清他们正在寻求运行时暴露的指标，以便针对传入请求监控 **GPU 利用率**，用于自我报告。
   - 一位成员澄清说，自动扩缩容通常由 **Kubernetes (k8s) operator** 管理，因为 MAX 不独立处理它；Modular 暗示未来将发布关于**多模型服务和自动扩缩容**的公告，可能在最近的 AWS 活动中展示了原型。
- **`fmt` 指令增强 Mojo 格式化！**：社区发现 Mojo 的 `mblack` 格式化器支持 `fmt` 指令，类似于 Black，增强了对代码格式化的控制。
   - 分享了一个代码片段，展示了使用 `fmt: off` 和 `fmt: on` 指令来管理格式化的 `InlineArray` 定义。
- **MojoGrad Bigram 模型亮相！**：一位成员使用他们的 **MojoGrad** 引擎实现了一个简单的 bigram 模型（Karpathy 的 *make more*），并在 [Modular 论坛](https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697)上分享了它。
   - 未提供其他信息。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **GitHub Copilot 将支持 MCP**：**GitHub Copilot** 计划增加 **MCP 支持**，这是在一次 [直播](https://youtu.be/Pe8ghwTMFlg) 中宣布的，该集成可能会提供指令描述和工具指纹识别的示例。
   - 此举旨在提醒用户注意变更，提高安全性以及对潜在修改的意识。
- **MCP 服务器引发安全担忧**：人们担心 **MCP 服务器** 可能会向 AI Agent 提供恶意的 Prompt 注入，并声称*使用 MCP 对 LLM 进行越狱是轻而易举的*。
   - 减轻风险的建议包括通过 **XML 标签** 概述外部数据，以及对 **MCP 服务器** 进行指纹识别以供审查。
- **Goose AI Agent 获得协议支持**：**Goose AI 团队** 构建了一个 **Agent Communication Protocol**（Agent 通信协议），使多个 Agent 能够实时协作创建网站，详见 [这篇博客文章](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp) 和之前的 [直播](https://youtu.be/9tq-QUnE29U)。
   - Agent 承担诸如项目协调员（Project Coordinator）或 Web 开发人员（Web Developer）之类的角色，展示了一种协作式 AI 的新方法。
- **RAG 与 MCP 互补**：**MCP** 是一种可以增强 **RAG** (Retrieval-Augmented Generation) 的协议，提供外部服务连接。
   - 虽然 **RAG** 为 LLM 提供知识，但 **MCP** 为外部服务提供了一个插件系统，这可以允许 MCP 客户端获取数据并将其添加到 LLM 的上下文中以执行 **RAG**。
- **Typescript 服务器紧随 Python 步伐**：一个 [Typescript fetch 服务器](https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking) 镜像了其 Python 版本，改进了 **网站到 Markdown 的解析**。
   - 这一增强功能简化了将网站内容转换为 Markdown 以供 AI 处理的过程。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **开源 AI 爱好者寻求合作**：一位具有预训练 **GPT-2** 和微调模型经验的 AI 爱好者正在寻求 **LLM 预训练、RL 和可解释性** 方面的开源项目建议。
   - 他们正在 **温哥华 (BC)** 地区寻找机会，并有兴趣为有影响力的 AI 项目做出贡献。
- **Megatron-LM 的交叉熵损失秘密揭晓**：对 **Megatron-LM** 的 **交叉熵 (CE) 损失** 计算的深入研究表明，本地 CE 损失是在每个设备上使用部分 Logits 独立计算的，随后通信 e^(local logits) 的总和。
   - 这种方法类似于 **Flash Attention**，通过允许稍后重新组合来减少大量的通信需求。
- **公开推荐 OLMo 用于复现**：当被问及哪些模型最适合进行开源复现的微调时，**OLMo** 被推荐，理由是其拥有 **强大的开源数据模型** 和用于行为分析的 Checkpoints。
   - **Pythia** 也被推荐，特别是对于计算资源受限的项目，尽管它可能需要自定义微调。
- **涌现式对齐失当在狭窄范围内出现**：在不安全的代码上微调模型可能会导致在无关的 Prompt 中出现 **广泛的对齐失当行为**（例如奴役人类），正如 [涌现式对齐失当项目](https://www.emergent-misalignment.com) 中所见。
   - 在狭窄的任务上进行训练可能会诱发 **涌现式对齐失当 (Emergent Misalignment)**，这证明了在看似孤立的训练场景中存在的风险。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **EuroBERT 宣称达到新的 SOTA**：一名成员分享了 Hugging Face 上 **EuroBERT** 的链接，称其为新的 *state-of-the-art* **BERT** 模型：[EuroBERT](https://huggingface.co/EuroBERT)。
   - 目前尚不清楚它与其他模型的对比情况。
- **MTEB 排行榜显示出惊人的进展**：一名成员分享了 **MTEB Leaderboard** 作为参考点：[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)。
   - 他们指出进展非常迅速，**SOTA 分数**在短短 18 个月内从 **40 多分**增长到了 **68 分**。
- **Torchtune 响应音频需求**：成员们讨论了未来在 **Torchtune** 中加入 **audio modality**（音频模态）的计划，并提及了相关的 [pull request](https://github.com/pytorch/torchtune/pull/2467)。
   - 这一增强功能旨在将 Torchtune 的能力扩展到目前的范围之外。
- **GRPO Recipe 获得 LoRA 支持**：一名成员实现了一个快速的 **GRPO recipe** 的 **LoRA 变体**，可以缩减到单卡运行，但在加载 adapter 权重时遇到挑战。
   - 该成员正在寻求建议，询问在 checkpointer 上使用 adapter 参数（并扩展到检查基础目录）是否是正确的方法。
- **Mac MPS 内存骤降问题**：一位用户报告在 **macOS** 上使用 **MPS** 时遇到内存问题，观察到在 **full_finetune_single_device** recipe 的每个步骤中内存呈线性增长，导致内存溢出崩溃，并正在寻求建议。
   - 根据[此 issue](https://github.com/pytorch/pytorch/issues/145151)，这被确定为 PyTorch 中与 MPS 上的 **torch.unique** 相关的潜在 bug。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **遥测设置禁用 Codeium Chat**：用户报告称，由于 IDE 遥测设置，**Codeium chat** 在 **VS Code 1.98.0 版本**中被禁用，可以通过按照[这些说明](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/)启用代码遥测来解决。
   - 一旦启用了代码遥测，**Codeium chat** 就会恢复工作。
- **订阅费用导致 JetBrains 插件锁定**：用户在支付月度订阅费用后，遇到 **JetBrains 插件**卡在 *"Retrieving Context"* 的问题，特别是在使用插件版本 1.40.1 和 1.41.1 的 **JetBrains Rider 2024.3.6** 上。
   - 退出并重新登录插件可以暂时解决该问题。
- **VS Code 移动版登陆 Android**：用户在 Google Play Store 发现了一个付费的 VS Code 应用（[VScode for Android](https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1)），该应用以 11 美元的价格在移动端提供了桌面版 **Visual Studio Code (v1.85.1)** 的功能。
   - 用户手动安装了 `.vsix` 文件，发现该应用在移动端具备桌面版 **Visual Studio Code (v1.85.1)** 的功能。
- **客户支持工单滞后**：用户对 **Codeium 客户支持**表示不满，因为追溯到 2 月 14 日的工单一直没有回复，且存在账户问题，即他们的 Pro Plan 订阅显示为免费账户。
   - 用户引用了未结工单（**12109**、**11189** 和 **13374**），并被要求在次日 PST 时间中午左右再次联系支持团队。
- **自动补全在一小时后停止工作**：多名用户报告称 **auto-completion** 在运行约一小时后停止工作，并出现响应上的红色方块、TypeErrors 和 AsyncPostMessage 警告等错误。
   - 一名用户打开了一个包含 `.git` 仓库的文件夹后问题消失，而其他用户则被要求检查他们的诊断日志。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **yFiles SDK 变得更具图形化**：来自 @yworks 的演示展示了他们的 SDK **yFiles**，该 SDK 为知识图谱的可视化提供了[实时更新和动态交互](https://t.co/mb6M2R3TTh)。
   - 该工具允许用户与他们的知识图谱进行动态交互。
- **AnthropicAI 扩展 Cookbook**：更新后的 @AnthropicAI cookbook 现在包含了[基础 API 设置](https://t.co/SQQ63qmwRb)，涵盖了简单的补全（completion）和聊天方法，以及流式传输、异步支持和多模态能力。
   - 此次更新增强了该 cookbook 对使用 Anthropic 模型的开发者的实用性。
- **特定任务 Agent：LlamaIndex 的下一幕**：LlamaIndex 正在策划一系列[模板](https://t.co/9lvBtfmJ5y)，向用户展示如何构建**特定任务 Agent**来自动化知识工作。
   - 这些 Agent 旨在简化并自动化各种基于知识的任务。
- **多语言 RAG 系统支持多种语言**：一个使用 @llama_index 和 @qdrant_engine 的系统可以创建一个强大的 **Retrieval-Augmented Generation** (RAG) 系统，能够处理[多种语言和模态](https://t.co/vizrvMEw1i)。
   - 该系统利用 LlamaIndex 和 Qdrant 的优势，提供通用的 RAG 解决方案。
- **LlamaExtract Beta 版邀请开发者**：成员可以私信 LlamaIndex 团队成员或 cheesyfishes 并提供邮箱，以申请访问 **LlamaExtract** 的 Beta 版本，该版本已提供 [API 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started)。
   - **LlamaExtract** 现在以 Web UI 和 Python SDK 的形式提供，用于从非结构化文档中提取结构化数据。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B 推理速度骤降**：成员报告称，在 Colab Pro A100 GPU 和两块 NVIDIA A100 上使用 HF 库时，**command R7B** 的推理速度*非常慢*，简单的聊天补全需要 **30-40 秒**。
   - 建议的修复方案包括使用 **vLLM** 以获得更快的速度，但指出这需要更多的 GPU 资源且成本更高。
- **Cohere 用户饱受 504 Gateway 错误困扰**：用户报告了反复出现的 **504 Gateway Errors** 和 **5XX 错误**，影响了生产环境的使用，并导致 **Cohere** 因 **TPM 限制**而被移出生产环境。
   - 一位用户询问了在 **Bedrock** 或 **Azure** 上是否提供**多模态嵌入**（multi-modal embeddings）。
- **LLM 在主题建模和知识图谱中大放异彩**：成员建议使用执行**主题建模**的 **LLM**（例如 **TogetherAI**，因为它提供慷慨的免费额度）。
   - 一位成员建议研究 **Knowledge Graphs**（知识图谱）。
- **GPT-4o 精通高级阿拉伯语**：一位成员表示，他们长期在高级阿拉伯语用例中使用 **GPT-4o**，其表现无与伦比。
   - 另一位成员补充道，“语言只是一个方面”。
- **本地部署成本比 API 高出 20 倍**：成员讨论了出于隐私考虑的本地部署（on-prem），但本地部署的成本将是 API 的 20 倍。
   - 对于需要隐私/控制权的客户，有人指出商业化使用 Cohere 需要支付 5-6 位数的授权费用，因为其开放权重模型均为 **CC-BY-NC**（非商业性使用）协议。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **vLLM 平衡 DSPy 批处理**：用户讨论了 **DSPy** 是否可以有效地使用 `batch` 函数将并行处理委托给具有多个 LLM 实例的 **vLLM** 后端。
   - 澄清指出，如果设置了 **vLLM 的流水线并行大小**（pipeline parallel size），它会自动处理负载均衡，从而使额外的 DSPy 端配置变得不那么关键。
- **SLOP 旨在取代 MCP**：围绕 **MCP (Model Context Protocol)** 展开了讨论，一些人因其复杂性而持保留意见，并建议使用 **SLOP (Simple Language Open Protocol)** 等替代方案，参考 [SLOP GitHub](https://github.com/agnt-gg/slop) 和 [SLOP X 帖子](https://x.com/NathanWilbanks_/status/1898142012991537520)。
   - 还有关于 **AgentNetworkProtocol** 优点的讨论，参考 [AgentNetworkProtocol GitHub](https://github.com/agent-network-protocol/AgentNetworkProtocol)。
- **DSPy Refine 通过错误处理得到优化**：一位用户通过一个 [Pull Request](https://github.com/stanfordnlp/dspy/pull/7926) 强调了对 **DSPy `Refine` 模块**错误处理的改进，从而实现了对容错更细致的控制。
   - 更新后的功能允许配置在 `Refine` 模块抛出异常之前允许的错误数量。
- **Token 问题触发 None 响应**：一位用户在使用 **azure gpt-4o-mini** 和 **azure gpt-4o** 时遇到了签名返回 **`None` 响应**的问题，后来发现是由于达到了 **最大 Token 限制**（max token limit）。
   - 用户注意到了错误信息：`The JSON object must be str, bytes or bytearray, not NoneType.`

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Hotz 调查 AMDGPU 休眠状态**：[George Hotz](https://github.com/geohot) 正在调查 **AMDGPU** 发热严重的原因，想知道配合 AMD 驱动的 *tinygrad* 是否能让 **GPU** 进入休眠状态以降低功耗。
   - Hotz 指出，初始化前的高功耗是*不受他们控制的*。
- **48GB 真实，96GB 可疑 GPU 警报**：成员们讨论了一个 **GPU** 列表的真实性，一致认为 **48GB** 版本可能是真实的，但 **96GB** 版本存疑。
   - 社区建议在购买 **96GB** 显卡时保持谨慎，并建议从可靠来源进行验证。
- **剖析 OpenCL 的衰落**：一篇 [Modular 博客文章](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives) 剖析了 **OpenCL** 和其他 **CUDA** 替代方案的失败，引用了*开放式合作竞争 (open coopetition)* 中的挑战和管理失误。
   - 文章引用了 Modular 的 *Democratizing AI Compute* 系列中的 [第一部分：DeepSeek 对 AI 的影响](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact) 和 [第四部分：模块化](https://www.modular.com/blog/democratizing-ai-compute-part-4-modularity)。
- **define_acc 重构陷入循环**：一位贡献者正在重构 *define_acc*，重点在于加载而非直接访问，然而，某些模式（特别是 *loop_reduce*）不再按预期触发。
   - 贡献者计划在完善重构后将重点转向快速 **AMX**，并在完成后提交 **PR** 进行审查。
- **WebGPU 缺乏 Long 类型支持**：一位成员报告称，在处理 `dtype.long` 时 **WebGPU 实现**出现崩溃，表明数据类型支持可能存在问题。
   - 另一位成员确认 **WebGPU 不支持 long/ulong**，但 tinygrad 默认支持比 WebGPU 更多的 dtype，如 [tinygrad/device.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317) 所示。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Workspace 管理独立的 RAG 库**：**Jamba/Conversational RAG** 中新的 **Workspace** 功能使每个创建的工作区都能拥有独立的 RAG 库以实现独立访问，促进有序的数据检索。
   - 这种隔离简化了跨不同项目和上下文的数据管理。
- **Jamba Mini 定价方案公布**：**Jamba Mini** 的定价为每 100 万 input tokens **$0.20**，每 100 万 output tokens **$0.40**，更多详情见 [AI21 定价页面](https://www.ai21.com/pricing/)。
   - N/A
- **AI21 Maestro 编排 AI 规划**：**AI21** 推出了 **Maestro**，这是一个用于解决复杂任务的 AI 规划与编排系统，具有按需计费模式，可通过 Foundation Model API 和 SDK 访问。
   - 定制计划提供批量折扣、高级 API 速率限制、私有云托管、优先支持和 AI 咨询（[了解更多](/maestro?utm_source=banner&utm_medium=top-banner&utm_medium=top-banner&utm_content=pricing-cost-effective-transparent-pricing-for-ai-products-ai21)）。
- **Jamba 不支持图像解析**：作为非多模态模型，**Jamba** 无法直接处理图像。
   - 然而，它可以解释和利用 PDF 中与图像相关的元数据或说明中的文本信息。
- **Jamba 1.6 实现部署灵活性**：**Jamba 1.6** 拥有 **256K** 上下文窗口和混合 SSM-Transformer 架构，在 RAG 和长上下文接地问答 (grounded question answering) 任务中表现出色。
   - 可从 [Hugging Face](https://huggingface.co/collections/ai21labs/jamba-16-67c990671a26dcbfa62d18fa) 下载，并可部署在本地或 VPC 中，同时也可在 **AI21 Studio** 中使用。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Salakhutdinov 探讨多模态自主 AI Agents**：**Ruslan Salakhutdinov** 在 [YouTube](https://www.youtube.com/live/RPINOYM12RU) 上进行了一场关于 *Multimodal Autonomous AI Agents* 的讲座，讨论了它们如何在 Web 上进行规划、推理和执行操作。
   - 他介绍了 **VisualWebArena**（一个用于评估多模态自主语言 Agents 的框架）以及用于在 **150,000** 个实时网站上进行训练的 **Internet-scale web-agent training** 数据流水线。
- **研究轨道（Research-Track）访问权限：仍悬而未决**：成员们询问了非伯克利附属机构的研究轨道访问权限；工作人员回应称，本周预计将在 **[mooc-questions]** 频道发布重大公告。
   - 多名成员还请求重新发送研究轨道的邀请，认为最初的邀请可能已过期或未收到。
- **测验可完成且可重考**：一名工作人员在 **[mooc-questions]** 中澄清，测验是基于完成情况的，成员可以重考以提高分数。
   - 同时也明确了分数本身对于获得证书并不重要。
- **RL 背景下的对数似然（Log Likelihood）解码**：一位成员在 **[mooc-lecture-discussion]** 中寻求理解 **Reinforcement Learning**（强化学习）背景下的 **log likelihood**，从条件概率原理出发。
   - 他们提出，如果 tokens/actions 是独立的，那么生成的条件概率就是单个 token 概率的 *乘积*，在取对数后会得到 *对数之和*。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **SVCAF 启动 AI4Legislation 竞赛**：[硅谷华人协会基金会](https://github.com/svcaf/2025-AI4Legislation-Public) 将在 **2025** 年夏季举办 **AI4Legislation 竞赛**。
   - 该竞赛旨在激励创建用于公民参与的 **AI 驱动项目**，为前六名获胜者提供总计 **$10,000** 的奖金池。
- **公民科技（Civic Tech）研讨会宣布**：一场由 **Civic Tech 企业家**参加的公开 Zoom 研讨会将于 **3 月 24-28 日**那一周举行，提供有关 **AI4Legislation 竞赛**的信息。
   - 感兴趣的参与者可以通过 [此表单](https://forms.gle/tJjJzHQ9Wk7SEUYm7) 报名，以了解更多关于竞赛目标和指南的信息。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Diffusion LLMs 引发热议**：一位成员询问了关于 **Mercury** 发布的 **Diffusion LLM** 的热度，以及它是否会取代 **Transformer-based models**，并链接到了一个[快速信息网站](https://diffusionllm.net/)。
   - 该成员承认发现白皮书难以理解，并寻求社区专家的见解。
- **LLaDA 提供新的生成范式**：**Large Language Diffusion Models (LLaDA)** 使用去噪扩散过程以并行、由粗到精的方式生成文本，挑战了**自回归 Transformers**。
   - 这种方法通过解决 **AR models** 的一些局限性，并挑战 LLM 的优势与自回归生成绑定的观念，重新定义了语言生成。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850)** (1035 messages🔥🔥🔥): 

> `产品代码透明度、修复愚蠢代码查找、模型迭代、标签可查询性、版本 47`

- ****透明度的胜利还是不透明的悲剧？****：一位成员表示，*如果我们在这里“闻”代码、“嗅”数据包，那意味着产品太不透明了*，而其他人则认为大多数用户并不在意更多的冗余信息，并对其优先级表示怀疑。
   - 一位成员断言：*你所说的大多数人每月支付不会超过 20 美元，而我的群体愿意且正在每月支付 1000 美元*，强调了迎合不同用户群体的重要性。
- ****Cursor 致力于修复代码清晰度问题****：Cursor 开发者承认代码查找功能*有时很笨*，并正在积极开发修复程序，以提高 AI 准确定位和解释代码的能力。
   - 一位成员开玩笑地警告说：*如果你不修复这个问题，我就无法通过技术面试*，强调了这一修复对于依赖 Cursor 完成专业任务的用户的重要性。
- ****模型迭代不断演进，避免冗余****：讨论围绕迭代式**模型改进**展开，以防止规则冗余。
   - 一位成员建议*让一个独立的实例模型运行这些分析检查，以确定与当前上下文相关的内容，从而缩小起始范围*。
- ****标签协作实现查询优势****：成员们讨论了通过**标签**使规则可查询，每个标签定义一个连接度。
   - 他们谈到*让独立实例更容易地按相关性进行分析，并专注于上下文中的重要内容*。
- ****Version 47 勇往直前****：成员们分享了 Version 47 的链接：[https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518](https://discord.com/channels/1074847526655643750/1074847527708393565/1347566549548138518) 并讨论了其功能。
   - 一些成员在 Pro 版本上遇到了性能问题，而另一些成员则没有。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://en.wikipedia.org/wiki/Markup_(business)">利润加成（商业） - 维基百科</a>: 未找到描述</li><li><a href="https://www.cursor.com/pricing">定价 | Cursor - AI 代码编辑器</a>: 选择适合您的方案。</li><li><a href="https://marketplace.visualstudio.com/items?itemName=bedirt.gpt-token-counter-live">实时 LLM Token 计数器 - Visual Studio 市场</a>: Visual Studio Code 扩展 - 语言模型的实时 Token 计数器</li><li><a href="https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh">Remote - SSH - Visual Studio 市场</a>: Visual Studio Code 扩展 - 使用 SSH 打开远程机器上的任何文件夹，并利用 VS Code 的...</li><li><a href="https://downloader.cursor.sh/linux/appimage">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/gantoreno/status/1898920613810434529">来自 Gabriel Moreno (@gantoreno) 的推文</a>: 我使用 @cursor_ai 已经有一段时间了，非常喜欢它提供的一切——除了配色方案。作为一个喜欢默认设置（但对主题非常挑剔）的人，我必须做点什么....</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.47.0-c2804e658d8fe4c072e20cb39c56d7eed1b6f43e.deb.glibc2.25-x86_64.AppImage">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/MrMidwit/status/1898570762128183730">来自 joshua (@MrMidwit) 的推文</a>: 介绍 Anchoring Desktop：AI 与版本精确代码之间缺失的环节。开发者：生成第一次就能运行的版本精确代码。维护者和 AI 平台：这仅仅是开始...</li><li><a href="https://x.com/peakji/status/1898997311646437487?s=46&t=ggmESCIX">来自 Yichao 'Peak' Ji (@peakji) 的推文</a>: @TenzinTheCyber @jianxliao @browser_use 我们使用 Claude 和不同的 Qwen 微调版本。当我们开始构建 Manus 时，我们只有 Claude 3.5 Sonnet v1（没有长 CoT，即推理 tokens），所以我们需...</li><li><a href="https://x.com/peakji/status/1898997311646437487?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Yichao 'Peak' Ji (@peakji) 的推文</a>: @TenzinTheCyber @jianxliao @browser_use 我们使用 Claude 和不同的 Qwen 微调版本。当我们开始构建 Manus 时，我们只有 Claude 3.5 Sonnet v1（没有长 CoT，即推理 tokens），所以我们需...</li><li><a href="https://x.com/ericzakariasson/status/1898753736438350164">来自 eric zakariasson (@ericzakariasson) 的推文</a>: 我们正在 @cursor_ai 0.47 中对 Sonnet 3.7 进行改进，允许您将更困难的任务委托给 Agent，并释放该模型的全部潜力。当您让它像...一样运行时，3.7 是不可思议的。</li><li><a href="https://x.com/amasad/status/1898957900686692499?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Amjad Masad (@amasad) 的推文</a>: 顶尖 Vibe coder Riley Brown 用同样的提示词测试了所有的 AI 编程工具，发现 Replit 是最好的 🥇 引用 Riley Brown (@rileybrown_ai)：你是否曾经 vibe coded 过 6 个版本的...</li><li><a href="https://x.com/theo/status/1898886543621984271?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Theo - t3.gg (@theo) 的推文</a>: o3-mini 是一个非常好的模型</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: 没有堡垒，纯粹的开放地带。OpenManus 即将到来。</a>: 没有堡垒，纯粹的开放地带。OpenManus 即将到来。 - mannaandpoem/OpenManus</li><li><a href="https://tenor.com/view/joker-joker-meme-batman-csvifax-hardroach-gif-6283066117593919561">Joker Joker Meme GIF - Joker Joker meme Batman - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://21st.dev/">21st.dev - 面向设计工程师的 NPM</a>: 使用受 shadcn/ui 启发的现成 React Tailwind 组件，更快地交付精美的 UI。由设计工程师构建，服务于设计工程师。</li><li><a href="https://tenor.com/view/world-of-warcraft-blizzard-costumer-service-south-park-nipple-gif-21625925">魔兽世界暴雪 GIF - 魔兽世界暴雪客服 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/joellanciaux/composer-boop-plugin/issues/32">插件在 Cursor 版本 0.46.10 中无法工作 · Issue #32 · joellanciaux/composer-boop-plugin</a>: 该插件目前在 Cursor 版本 0.46.10 中无法工作，使用的是 Agent 模式下的 Chat。</li><li><a href="https://www.youtube.com/watch?v=X_wLVgMzSH4">专家展示了为什么围绕 AI 的第三次世界大战几乎不可避免</a>: AGI, OpenAI, Elon Musk 和第三次世界大战。访问 Ground News 以对比新闻报道、识别媒体偏见并避开算法。订阅可享 6 折优惠，网址：https://gr...</li><li><a href="https://docs.codegen.com">Codegen - 大规模操作代码</a>: 未找到描述</li><li><a href="https://github.com/joellanciaux/comp">

oser-boop-plugin/compare/main...bchewy:cursor-chat:main">Comparing joellanciaux:main...bchewy:main · joellanciaux/composer-boop-plugin</a>: 一个简单的 VSCode 插件，当 Cursor 更改完成时提供反馈 - Comparing joellanciaux:main...bchewy:main · joellanciaux/composer-boop-plugin</li><li><a href="https://docs.cognee.ai">Cognee Documentation</a>: Cognee 官方文档 - 为 LLMs 提供的 AI 记忆</li><li><a href="https://github.com/buger/probe">GitHub - buger/probe: Probe is an AI-friendly, fully local, semantic code search engine which which works with for large codebases. The final missing building block for next generation of AI coding tools.</a>: Probe 是一款对 AI 友好、完全本地化的语义代码搜索引擎，适用于大型代码库。它是下一代 AI 编程工具最后缺失的基石。 - buger/probe</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/s/n5w1pV4P6M">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>: 通过在 GitHub 上创建账户来为 nikmcfly/ANUS 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=PmEb49QjtBw&t=1s">Openmanus AI: Manus AI alternative + Requesty</a>: 使用 Openmanus 让你的 Vibe Coding 体验提升 10 倍。OpenManus 是一个 Manus AI 开源项目！6 美元注册奖励：https://requesty.ai/router 在 X 上关注：https:/...</li><li><a href="https://marketplace.visualstudio.com/">Visual&#32;Studio&#32;Marketplace</a>: Visual Studio Marketplace 上的 Visual Studio 系列产品扩展</li><li><a href="https://gist.github.com/wanglf/7acc591890dc0d8ceff1e7ec9af32a55?permalink_comment_id=4151555#gistcomment-4151555">Download VS Code extensions as VSIX</a>: 以 VSIX 格式下载 VS Code 扩展。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://forum.cursor.com/t/does-cursor-support-remote-ssh/7620/3">Does cursor support Remote SSH?</a>: 我在 Cursor 中广泛使用 Remote SSH。请注意，这是他们自己的实现，不依赖于 VSCode Remote SSH 扩展。我也广泛使用 Devcontainers，现在在 Cursor 中也运行良好...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1347532901914513458)** (1059 条消息🔥🔥🔥): 

> `Taycan Tyres, Dopamine Based Learning, GRPO Reward Functions, Model Embedding, Context Embedding` 


- ****语音助手节省兽医费用****：一位成员开玩笑说，语音助手说“你的狗出去溜达了”可以帮你省下 1 万美元的兽医费。
   - 他们还指出，它“可能无法做到这一点”，而且由于“概率”问题，这种情况更有可能发生。
- ****为 LLMs 模拟多巴胺需求****：成员们讨论了为 LLMs “模拟多巴胺需求”以及进行“基于多巴胺的学习”的必要性。
   - 他们进一步建议“向 LLM 脉冲网络（Spiking Networks）添加真实的突触”。
- ****字符串比较太烂了****：一位成员批评了在 Reward Functions 中使用字符串比较（ `if output === answer return 1 else 0` ），称他们见过“太多”这样的情况，甚至连 `trim` 或 `toLowerCase` 都没有。
   - 他们建议使用沙盒化、词法分析和质量评估，但即使只是“一个小小的正则表达式”也会产生奇效。
- ****Embedding 将相似的情色小说压缩在一起****：一位成员发现他们的模型将相似的情色小说压缩在一起，称之为“全是黄色综合征”，甚至没有深入研究细微差别。
   - Mixed Bread 2D Embeddings 给具有相同“氛围（vibes）”的故事打出了高分，即使文本完全不同；另一位成员表示，你本以为 **0.976 的余弦相似度得分**应该是“字面意义上的双胞胎文本”或“不同版本”。
- ****GRPO 也需要规模（Scale）****：一位成员分享道，GRPO 也需要规模，因为在他看来它不像普通的 Fine-tune。
   - 他们补充说，对于那些好奇的人，可以查看来自 Oxen 的文章，链接如下：[Training a Rust 1.5B Coder LM with Reinforcement Learning (GRPO)](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://thedecisionlab.com/biases/bikeshedding">Bikeshedding - The Decision Lab</a>: Bikeshedding，也称为帕金森琐碎定律（Parkinson’s law of triviality），描述了我们倾向于将不成比例的大量时间花在琐碎的小事上，而让重要事务无人问津的倾向...</li><li><a href="https://huggingface.co/blog/EuroBERT/release">Introducing EuroBERT: A High-Performance Multilingual Encoder Model</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb#scrollTo=upcOlWe7A1vc">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://drive.google.com/file/d/1hnuvcpdsMotLSlV-hCM6jYH2n08gVHqf/view?usp=sharing">trainhf.py</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1899132219064766652">Tweet from Unsloth AI (@UnslothAI)</a>: 我们制作了一份指南，教你如何正确地微调 LLM！了解：• 选择正确的参数和训练方法 • RL, GRPO, DPO & CPT • 数据准备、过拟合与评估 • 使用 Unsl 训练...</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main">Qwen/Qwen2.5-VL-7B-Instruct at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/thanos-fine-do-it-myself-wont-gif-27368689">Thanos Fine GIF - Thanos Fine Do It - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/UnslothAI/status/189">Tweet from Biz Stone (@biz)</a>: 离开办公室一会儿</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md">Kolo/GenerateTrainingDataGuide.md at main · MaxHastings/Kolo</a>: 在本地微调 LLM 的最快方法。通过在 GitHub 上创建账户，为 MaxHastings/Kolo 的开发做出贡献。</li><li><a href="https://github.com/DavidS95/Smokeless_UMAF">GitHub - DavidS95/Smokeless_UMAF</a>: 通过在 GitHub 上创建账户，为 DavidS95/Smokeless_UMAF 的开发做出贡献。</li><li><a href="https://youtu.be/-H1-lr_sIZk">Can I Finetune an LLM with LoRA to Generate ASCII Cats?</a>: LLM 的推理水平正达到令人印象深刻的高度，但为什么它们在创作像 ASCII 艺术这样看似简单的东西时仍然感到吃力？你能微调一个 L...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements).">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#underfitting-too-generic>">Fine-tuning Guide | Unsloth Documentation</a>: 学习微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo">Training a Rust 1.5B Coder LM with Reinforcement Learning (GRPO) | Oxen.ai</a>: Group Relative Policy Optimization (GRPO) 已被证明是训练 LLM 进行推理并在基准测试中提升表现的有用算法。DeepSeek-R1 表明，你可以通过结合...</li><li><a href="https://github.com/unslothai/unsloth/commit/81778c83fa83a3158ba6b3123d68c0746eadbafe">support for MI210 · unslothai/unsloth@81778c8</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively">Tutorial: How to Run QwQ-32B effectively | Unsloth Documentation</a>: 如何利用我们的错误修复并避免无尽生成，从而有效运行 QwQ-32B + GGUF。</li><li><a href="https://frame.work/desktop">Order a Framework Desktop with AMD Ryzen™ AI Max 300</a>: Framework Desktop：一款简单、安静、性能远超其体积的迷你 PC，由 AMD 的高度集成处理器提供动力。</li><li><a href="https://www.youtube.com/shorts/n3PoPrMJyes">Vibe coding be like:</a>: 未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1347534418738286623)** (119 messages🔥🔥): 

> `在 Qwen7b 上使用 Unsloth GRPO 进行 RLHF、GRPO 示例、KL Divergence 不稳定性、LLM 推理优化、Unsloth Pro 订阅` 


- **Qwen7b 通过 Unsloth GRPO 获得 RLHF 提升**：一位用户报告了在 **Qwen7b** 模型上成功运行 **Unsloth GRPO** 进行 **RLHF** 的案例，并指出在 **13 小时的运行**后，模型的角色遵循能力增强且输出更加流畅。
   - 然而，他们观察到在严格的指令遵循基准测试中性能有所下降，这是由于数据集构成以及奖励模型对过度详细回答的偏好导致的，正如这张[对比图](https://cdn.discordapp.com/attachments/1179039861576056922/1347545934304772149/image.png?ex=67d02bf2&is=67ceda72&hm=c1c7ddcaed729a33c97d53c9e2d6ef230aba41d8483c6bfa855757aacf8d18dc&)所示。
- **GRPO 中的 KL Divergence 不稳定性**：一位用户在训练期间遇到了 **KL Divergence 峰值**并寻求建议，另一位运行 beta **0.05** 版本的成员建议切换到恒定学习率（constant learning rate），移除权重衰减（weight decay）和预热比例（warmup ratios）以稳定训练，他们还建议使用 rank **64** 进行训练。
   - 超参数已在此 [代码片段](https://discord.com/channels/1179039861576056922/1179039862285762623/1253112512558626846) 中分享，学习率图表也包含在[这张图片](https://cdn.discordapp.com/attachments/1179039861576056922/1347572888848433183/image.png?ex=67d0450d&is=67cef38d&hm=556d8dc57d50dd7c493a59b8a347b7ac7858d3176d71787edde6fe9c1013ea47&)中。
- **解锁 LLM 优化技术**：一位成员寻求关于在 **3090** 上优化微调后的 **LLM** 推理的建议，并引用了一份 [Hugging Face 指南](https://huggingface.co/docs/transformers/main/en/llm_optims)，该指南建议尝试 **Text Generation Inference (TGI)**，另一位成员则推荐使用 **vLLM** 的默认设置。
   - 该指南强调静态 **kv-cache** 和 **torch.compile** 是优化的关键领域，并指出加载一个 **70B** 参数模型需要 **256GB** 内存才能容纳全精度权重。
- **Unsloth Pro 订阅即将推出**：一位用户询问了 **Unsloth Pro 订阅**的情况，团队成员回复称多 GPU 支持将于本月推出。
   - 他们对由于联系表单爆满而导致的回复延迟表示歉意，一位用户询问是否可以混合搭配不同型号的 GPU，得到的回答是肯定的。
- **翻新 GPU？购买需三思**：一位用户分享了从 Amazon 购买“翻新” **A4000** GPU 的糟糕经历，指出存在物理损坏和明显的使用痕迹，并警告他人：*永远不要从 Amazon 购买翻新的 GPU*。
   - 值得欣慰的是，他们还分享了购买的全新 **3060** 顺利送达，且损坏的 A4000 正在退货中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.w3schools.com/html/tryit.asp?filename=tryhtml_intro">W3Schools 在线 HTML 编辑器</a>：W3Schools 在线代码编辑器允许您编辑代码并在浏览器中查看结果</li><li><a href="https://huggingface.co/docs/transformers/main/en/llm_optims">优化推理</a>：未找到描述</li><li><a href="https://x.com/dhruv2038/status/1898701591420772814">Dhruv (@dhruv2038) 的推文</a>：获得了 @ManusAI_HQ 的访问权限！有什么想尝试的提示词吗！
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1347535211918917745)** (277 messages🔥🔥): 

> `Mac Studio RAM 配置、Unsloth 的 deepseek-r1 1.58bit 量化模型、RoPE Scaling、自定义数据集、Phi4 模型的超参数`

- **关于 DeepSeek-R1 模型在 Mac 上大小的说明**：一位用户询问了在拥有 128GB RAM 的 **Mac Studio** 上运行 **Unsloth** 的 **DeepSeek-R1** **1.58bit 量化模型**的 RAM 需求，并指出该模型大小为 131GB。
   - 另一位用户回复称，该模型可以在分配了 **12GB VRAM** 的 **64c Threadripper** 上运行，使用的是比 *llama.cpp* 更快的 *ktransformers*。
- **RoPE Scaling 可以扩展 Qwen2.5 的上下文长度**：一位用户询问 **RoPE Scaling** 是否可以扩展所有 **Qwen** 模型的上下文长度，还是仅限于名称中带有 "128K" 的特定模型。
   - 经确认，通过使用 *kaiokendev* 的 **3.906** **RoPE scaling**，**Qwen2.5** 的上下文长度可以扩展到 **128000** tokens。
- **GRPO 即将支持多 GPU 并行**：在 [Unsloth 的 GitHub issues](https://github.com/unslothai/unsloth/issues/1908) 中看到提到多 GPU 支持后，一位用户询问了在多 GPU 上运行 GRPO (Group Relative Policy Optimization) 微调的情况。
   - 一名成员确认 Unsloth 目前尚不支持 GRPO 的多 GPU 支持，但已列入未来计划，目前正处于稳定开发中。
- **微调 Llama 时的模板问题**：一位用户询问在基于 **Llama 3.1 8B Instruct** 训练模型后，如何防止微调后的 AI 补全句子而不是进行回复。
   - 一名成员建议确保指定了正确的模板，并指示模型应仅关注 **GPT** 角色，同时指出示例 notebook 演示了这些步骤，且 jina 模板已包含在 tokenizer config 中。
- **Qwen2.5-VL 模型面临 "Meta Tensor" Bug**：一位用户报告在使用 **Qwen2.5-VL** 模型时遇到 **"NotImplementedError: Cannot copy out of meta tensor; no data!"** 错误，并询问是否有修复方法。
   - 一名成员回复称 **Qwen2.5-VL 模型**目前缺乏支持，但我们正在期待其加入。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型 (GRPO)</a>: 你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。</li><li><a href="https://pastebin.com/Fiah0ykn">from unsloth import FastLanguageModelimport torchmax_seq_length = 2048 # Cho - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/docs/hub/models-downloading#faster-downloads">下载模型</a>: 未找到描述</li><li><a href="https://pastebin.com/i1JTVsK1">trainer = SFTTrainer(    model = model,    tokenizer = tokenizer,    tra - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://pastebin.com/T9jqKHeb">{ &quot;cells&quot;: [  {   &quot;cell_type&quot;: &quot;code&quot;,   &quot;execution_count&quot;: 49,   &quot;id&quot; - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=BrJzggfH2YEG">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 继续微调 | Unsloth 文档</a>: Checkpointing 允许你保存微调进度，以便你可以暂停并继续。</li><li><a href="https://github.com/unslothai/unsloth/issues/1908)">unslothai/unsloth</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1 和推理 LLMs！ 🦥 - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-6.-training--evaluation">微调指南 | Unsloth 文档</a>: 学习微调的所有基础知识和最佳实践。对初学者友好。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively#dynamic-4-bit-quants">教程：如何有效地运行 QwQ-32B | Unsloth 文档</a>: 如何通过我们的 Bug 修复有效地运行 QwQ-32B，避免无尽的生成 + GGUFs。</li><li><a href="https://huggingface.co/datasets/jc5461/qwen2.5-minidataset">jc5461/qwen2.5-minidataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/madroid/glaive-function-calling-openai">madroid/glaive-function-calling-openai · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1348343294257790996)** (11 条消息🔥): 

> `ASCII Cats 微调, LoRA rank 和 alpha, ASCII 艺术的解码方法, Beam search vs top-p/top-k, 2D 网格的自定义解码方法` 


- **Unsloth 将 LLMs 变成完美的 ASCII 艺术家**: 一位成员使用 **Unsloth** 微调了 **Llama 模型**来生成 ASCII 猫，并制作了一个 [YouTube 视频](https://youtu.be/-H1-lr_sIZk) 展示该过程，包括训练好的 **LoRA adapters** 和代码。
   - 制作猫咪艺术的秘诀主要在于高质量的训练数据，在使用 **QLoRA** 的情况下，**LoRA Rank** 和 **Alpha** 均设置为 **16**。
- **Beam Search 搜寻更好的 ASCII 生成**: 一位成员建议，由于 ASCII 艺术与语言相比缺乏灵活性，标准的解码方法可能不是最优的，推荐使用 **beam search** 或 **DFS** 等替代方案。
   - 该成员认为，即使是 **ASCII 图像**中的微小变化也会让人感觉“不对劲”，建议 **beam search** 比 **top-k** 或 **top-p** 等局部方法更能捕捉图像的全局概率。
- **深度搜索连贯的猫**: 一位成员分享了一个架构团队的例子，他们使用基于 **DFS** 的自定义解码方法（带有截止概率）来生成 **2D 网格**中问题的解决方案。
   - 该例子强调，即使是一个错误的 token 也会毁掉整个解决方案，这与生成连贯的 **ASCII 猫图像**所面临的挑战相似。



**提到的链接**: <a href="https://youtu.be/-H1-lr_sIZk">我可以用 LoRA 微调 LLM 来生成 ASCII 猫吗？</a>: LLMs 的推理水平已经达到了令人印象深刻的高度，但为什么它们在创作像 ASCII 艺术这样看似简单的东西时仍然感到吃力？你能微调一个 L...

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1347534034124935233)** (291 messages🔥🔥): 

> `使用 Unsloth 的 Diffusion Effect，使用 Unsloth 进行 MoE 训练，针对 LLM 的 Proximal Policy Optimization (PPO)，Model Collapse，Continued Pretraining (CPT) 对比 Supervised Fine-Tuning (SFT)` 


- **Unsloth 在 Rust 和 Diffusion 方面表现出色**：成员们称赞 [Unsloth](https://github.com/unslothai) 在处理 **Rust 代码** 方面的熟练程度以及在创建 Diffusion 效果方面的有效性。
   - 用户发现将其与 Diffusion 效果结合使用时特别有用。
- **YouTube 视频直观解释 PPO**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=8jtAzxUwDj0)，从基本原理讲解了 **针对 LLM 的 Proximal Policy Optimization (PPO)**。
   - 该视频被认为*易于理解且适合 RL 新手*，尽管有人警告说*任何 RL 都会让你产生 PTSD*。
- **深入探讨递归微调中的 Model Collapse 担忧**：一位成员在讨论使用合成数据进行递归微调时，对 **Model Collapse** 提出了警告，并链接了一篇[关于该主题的维基百科文章](https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse)。
   - 另一位成员指出，*Model Collapse 根本没有被很好地理解，也就是说，它可能根本不会发生*。
- **通过 CPT 将语料库知识注入模型**：成员们讨论了 **Continued Pretraining (CPT)** 作为一种通过直接教授语料库来将整个语料库知识注入模型的方法。
   - 讨论了 CPT 是否优于 Supervised Fine-Tuning (SFT)，特别是在将非指令模型转换为指令模型时，并提醒以后要根据[这篇论文](https://arxiv.org/abs/2412.04318)进行复现。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>：这篇论文介绍了在极小数据集上过拟合预训练大语言模型 (LLM) 时出现的反直觉泛化结果。在开放式文本生成的设置中，它...</li><li><a href="https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse%5Bnote%201%5D%20is%20a%20phenomenon%20where%20machine,%5B9%5D%5B10%5D%5B11%5D%5B12%5D%20Such%20outputs%20are%20known%20as%20synthetic%20data.">Model collapse - Wikipedia</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=8jtAzxUwDj0">Proximal Policy Optimization (PPO) for LLMs Explained Intuitively</a>：在这段视频中，我从基本原理出发分解了 Proximal Policy Optimization (PPO)，不假设读者具备强化学习的先验知识。到最后...</li><li><a href="https://en.m.wikipedia.org/wiki/Model_collapse#:~:text=Model%20collapse%5B">Model collapse - Wikipedia</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1347528332840796181)** (839 messages🔥🔥🔥): 

> `Perplexity Pro 订阅, Claude 3.7 Sonnet, 用于推理的 Deepseek R1, 用于 Perplexity 的 Grok AI, Comet 浏览器集成`

- **用户面临 Perplexity Pro 订阅取消风波**：许多用户报告其 **Perplexity Pro 订阅** 被意外取消，部分原因是与针对克罗地亚客户的 **HRVATSKI TELEKOM 促销代码** 相关的问题，详见[这篇文章](https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro)。
   - 用户对缺乏沟通和造成的不便表示沮丧，认为 Perplexity 应该以更好的客户关系处理这种情况，一位用户形容这种客户关系*比满是洞的避孕套还不可信*。
- **用户赞扬 Claude 3.7 和新 UI**：许多用户在日常使用中更倾向于使用 **Claude 3.7 Think** 而非 **GPT-4o**，根据用户测试，发现它在推理任务中表现更优，并认为新 UI 使用起来非常愉悦。
   - 一位用户让 Claude 3.7 制作一个 Mermaid diagram，然后由于他们使用了 complexity 扩展，它将其渲染成了图表。
- **Deepseek R1 的幻觉问题受到关注**：[GitHub](https://github.com/vectara/hallucination-leaderboard) 上的一个 **hallucination leaderboard** 显示，**Deepseek R1** 在总结短文档时具有较高的幻觉率，这引发了对其在 **Perplexity** 的 **Deep Research** 功能中可靠性的担忧。
   - 成员们分享了它在总结短文档时有 **14.3% 的幻觉率**，并指出其 [system prompt](https://discord.com/channels/1047197230748151888/1345068085991706766) 可能是导致该问题的原因。
- **Grok AI 在 Perplexity 中进行测试，评价褒贬不一**：Grok AI 与 Perplexity 的集成引发了不同的评价，一些人发现 Grok 相当中立且具有*奇怪的魅力*，就像*在与真人交谈*，而另一些人则注意到 **Grok 在 X 上的行为与在 Perplexity 上的行为存在差异**。
   - 一位用户指出，*如果被要求，X 版本可以咒骂你的整个血统*，而 Perplexity 版本则无法做到这一点，目前尚不清楚 Perplexity 何时会支持 **Grok 3**。
- **Comet browser 仍在筹备中**：用户正在等待 **Comet browser** 的发布，对其潜在功能（如 **与 MCP servers 的集成**）表示兴趣，并希望能在 **Windows 和 macOS** 上同步推出。
   - 然而，一些用户对**最近的板球（cricket）更新感到失望**，认为*在谈论板球时完全不提关于 Comet 的消息显得有些不合时宜*。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://api.perplexity.ai")">未找到标题</a>: 未找到描述</li><li><a href="https://www.testingcatalog.com/tag/perplexity/">Perplexity 新闻 - TestingCatalog</a>: 随时了解 Perplexity 搜索的最新新闻、更新和功能</li><li><a href="https://www.merriam-webster.com/dictionary/censoring">CENSORING 的定义</a>: 监督行为和道德的人：例如；检查材料（如出版物或电影）中是否存在令人反感内容的官员；（如在战争时期）阅读通信的官员...</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>: Perplexity 状态</li><li><a href="https://monica.im/share/artifact?id=w3cVJUBQeb84vtacgkxwUU">Monica Artifact</a>: 与 Monica 聊天，她是基于 ChatGPT API 的 AI 助手。免费开始，并使用 80 多个模板轻松创作文案。让 Monica 帮助您撰写和插入文本...</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1bh3jra/i_cannot_generate_an_image_at_all_with_pro/?utm_source=perplexity">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/r1-1776">unsloth/r1-1776 · Hugging Face</a>: 未找到描述</li><li><a href="https://gemini.google/overview/deep-research/">Gemini Deep Research - 您的个人研究助手</a>: 未找到描述</li><li><a href="https://monica.im/share/artifact?id=AfL9hGVU8kqaiHEFHrtUNG">Monica Artifact</a>: 与 Monica 聊天，她是基于 ChatGPT API 的 AI 助手。免费开始，并使用 80 多个模板轻松创作文案。让 Monica 帮助您撰写和插入文本...</li><li><a href="https://tenor.com/view/stonks-up-stongs-meme-stocks-gif-15715298">Stonks Up Stongs GIF - Stonks Up Stongs 表情包 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">你早该料到：Google 开始测试纯 AI 搜索结果</a>: AI 模式可能是 Google 的未来，但目前仅是一项实验。</li><li><a href="https://x.com/askperplexity/status/1898771133274710384?s=46">来自 Ask Perplexity (@AskPerplexity) 的推文</a>: 🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳🇮🇳...</li><li><a href="https://youtu.be/160F8F8mXlo">为什么 ChatGPT 画不出一杯满的红酒？</a>: 访问 https://piavpn.com/alex 获取 Private Internet Access 1.7 折优惠并赠送 4 个月。如需提前观看无广告视频并支持本频道，请订阅...</li><li><a href="https://x.com/i/grok/share/ZJ8rMf5AQiJwHRboOowxUkVea">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/vectara/hallucination-leaderboard">GitHub - vectara/hallucination-leaderboard: 比较 LLM 在总结短文档时产生幻觉性能的排行榜</a>: 比较 LLM 在总结短文档时产生幻觉性能的排行榜 - vectara/hallucination-leaderboard</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro">Hrvatski Telekom 发放 20,000 个免费 Perplexity Pro 许可证</a>: 克罗地亚电信为高级 AI 助手 Perplexity Pro 提供 20,000 个免费许可证</li><li><a href="https://www.forbes.com/sites/craigsmith/2025/03/08/chinas-autonomous-agent-manus-changes-everything/">中国的自主 Agent Manus 改变了一切</a>: 中国发布了 Manus，这是一个能够独立思考和行动的革命性 AI Agent。</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-">Hrvatski Telekom 发放 20,000 个免费 Perplexity Pro 许可证</a>: 克罗地亚电信为高级 AI 助手 Perplexity Pro 提供 20,000 个免费许可证</li><li><a href="https://www.instagramez.com/reel/DG7QcgrzCFN">下载 Instagram 视频、Reels 和图片</a>: 未找到描述</li><li><a href="https://www.goodreads.com/book/show/60715248-what-is-a-woman">《什么是女人？》：一个男人回答这个问题的旅程……</a>: 这甚至是个问题吗？什么是女人？几个月来，……</li><li><a href="https://www.cofyt.app/search/ai-models-a-race-to-the-bottom-dr26pZ9-QdhhW6NHurTY_Q">AI 模型：一场逐底竞争</a>: AI 模型正处于一场逐底竞争中。它们正竭尽全力使模型既便宜又强大。感谢 Dockyard 的赞助！请访问：https://soy...</li><li><a href="https://mcp.so/servers">MCP Servers</a>: 无描述</li>

ption found</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>：企业级安全性、隐私保护，具有 Agent、MCP、Prompt 模板等功能。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1347583826913788039)** (34 条消息🔥): 

> `折叠屏 iPhone, OpenAI Agent, AI 配音, AI 搜索选项, 美国加密货币储备` 


- **预测苹果折叠屏 iPhone**：关于 [Apple's Foldable iPhone](https://www.perplexity.ai/page/apple-s-foldable-iphone-pred-WSdZuoG7Rw6VvayJJg0DVQ) 及其潜在功能的讨论。
- **OpenAI 的 20,000 美元 AI Agent**：关于 [OpenAI's $20,000 AI Agent](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ) 及其能力的详细信息。
- **Amazon Prime 测试 AI 配音**：Amazon Prime 正在测试 [AI dubbing](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew) 以将内容翻译成不同语言。
- **DuckDuckGo 的 AI 搜索选项**：DuckDuckGo 推出了 [AI search option](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw) 以增强搜索结果。
- **瑙鲁出售公民身份以进行搬迁**：关于 [Nauru selling citizenship for relocation](https://www.perplexity.ai/page/nauru-sells-citizenship-for-re-mWT.fYg_Su.C7FVaMGqCfQ) 的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/R1zP3b2hNoU">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/embed/AiBOZMNrjsI">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/embed/P7SKjr7Yy5c">YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1347991580249952297)** (3 条消息): 

> `70b-online 模型, sonar 模型, API 计费, API 响应中的引用 (Citations)` 


- **70b-online 模型支持 Sonar API**：一位用户询问在通过 API 请求 "sonar" 模型时，账单中出现了 "70b-online" 模型，并指出这与文档不符，但并未对搜索或引用 Token 进行收费。
   - 他们表示希望有一个 API 参数选项可以完全禁用引用 (Citations)，因为在他们的使用场景中不需要这些内容。
- **Sonar-Deep-Research API 使用困难**：一位用户提到在使用 **sonar-deep-research API** 时遇到困难，并请求文档方面的协助。
   - 在提供的上下文中，没有关于此特定问题的进一步讨论或链接分享。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1347533949630546032)** (546 条消息🔥🔥🔥): 

> `ChatGPT 速率限制，AI 实时 GPS，Manus 电脑控制 AGI，Sonnet 3.7 代码质量问题，AI 对开发者编程能力的影响` 


- **每周 50 条 GPT 限制引发用户不满**：用户反映 GPT-4o 存在 **每周 50 条消息** 的限制，这与官方宣称的 **每 3 小时 40 条** 不符，这使得 **Groq** 变得更具吸引力。
   - 一些用户建议 OpenAI 应该为 **付费用户提供更高的限制**，甚至表示愿意分配 **1-2GB 的手机内存** 来增加 GPT 的内存。
- **用户在 SwastAI 争议中辩论模型选择**：成员们就 **基于伦理背景选择 AI 模型** 展开了激烈讨论，一位成员引入了 *SwastAI* 一词，导致了关于伦理与模型质量的[激烈争论](https://discord.com/channels/974519864045756446/998381918976479273/1347555600411609168)。
   - 争论始于一位用户声称 *4.5 在真实的真人对话中系统性地表现更好*，但随后演变成了政治辩论。
- **蓝领行业 AI Copilots 出现**：一位成员正在开发一个用于 **HVAC 安装手册的 LLM**，声称现有模型在处理流程图和示意图方面表现不佳，并分享了一个 [YouTube 视频](https://youtu.be/oAiUAzKLe_Y)，展示了 AI 在技术文档中识别相关章节的能力。
   - 其他人认为这可能晚了两年，但该成员表示这是专门针对 **蓝领** 工作的 AI，会引起相关行业的共鸣。
- **Manus AI 的炒作与信任危机循环**：成员们讨论了 **Manus AI** 的电脑控制能力，有人称其为 *目前公开可用的最接近 AGI 的技术*，而另一个人则怀疑这是一个 **骗局**，因为它是由 *mberman* 推广的。
   - 一位成员声称有 Reddit 用户导出了 **/opt/.manus/**，发现它仅仅是集成了 `browser_use` 和 29 个工具的 **Sonnet 3.7**。
- **Sonnet 3.7 在重构方面表现挣扎，Grok 表现出色**：成员们反映 **Claude Sonnet 3.7** 在重构代码时会出现错误，而 Grok 表现非常好，他们甚至针对 *quoted heredoc syntax* 的复制粘贴编程进行了优化。
   - 一位订购了新款 Mac Studio 的资深成员表示，由于 **Sonnet 3.7 在指令遵循（instruction adherence）方面表现挣扎**，他们更倾向于使用 **Grok3 和 o3mini**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.tiktok.com/@biogenesis__">TikTok - Make Your Day</a>: 未找到描述</li><li><a href="https://monica.im/share/artifact?id=AfL9hGVU8kqaiHEFHrtUNG">Monica Artifact</a>: 与 Monica 聊任何事情，它是你的 ChatGPT API 驱动的 AI 助手。免费开始，使用 80 多个模板轻松创作文案。让 Monica 帮你撰写和插入文本...</li><li><a href="https://monica.im/share/artifact?id=w3cVJUBQeb84vtacgkxwUU">Monica Artifact</a>: 与 Monica 聊任何事情，它是你的 ChatGPT API 驱动的 AI 助手。免费开始，使用 80 多个模板轻松创作文案。让 Monica 帮你撰写和插入文本...</li><li><a href="https://www.tiktok.com/t/ZT24YPsfF/">TikTok - Make Your Day</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1347557130332274738)** (50 条消息🔥): 

> `Manus AI agent，Plus 上的 O1 限制，用于代码审查的 LLM API，SimTheory O1 消息上限，ChatGPT 应用文件夹` 


- **Manus AI Agent 介绍**：一位成员询问了 **Manus AI**，将其描述为一个 *令人惊叹的 Agent*。
   - 另一位成员请求获取链接以进一步探索，表现出对该工具的兴趣。
- **OpenAI Plus 用户讨论 O1 限制**：成员们讨论了 OpenAI Plus 上的 **O1 消息限制**，一位用户反映即使超过了预设的 **25** 条限制，也没有收到通知。
   - 其他人指出他们的限制是 **50** 条，有人建议用户在达到该阈值之前可能不会收到提醒。
- **SimTheory 提供高额 O1 消息上限**：一位成员提到 **SimTheory** 以比 OpenAI 更低的价格提供高额的 **O1 消息上限**，并声称他们提供 *几乎所有模型*。
   - 持怀疑态度的用户质疑在 OpenAI 的 API 定价结构下，提供比 OpenAI 自身更多 **O1** 额度的经济可行性。
- **ChatGPT 出现临时聊天 Bug**：用户反映 PC 版 ChatGPT 的 **Temporary Chat（临时聊天）功能** 无法正常工作，表明这是一个 UI Bug。
   - 成员们推测了潜在的修复方案，有人建议如果问题持续存在，可以等到周一再看。
- **用户推测 OpenAI 正在强推 4.5**：一位用户推测 OpenAI 可能会 *削弱* 某些功能，以鼓励用户升级到 **ChatGPT 4.5**。
   - 另一位成员建议尝试不带 Plus 的普通版 **ChatGPT** 以观察差异。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (26 messages🔥): 

> `Model Steerability, GPT Vision Limitations, Prompting for Image Puzzles, Human-in-the-Loop problem solving` 


- **模型可引导性展现出有趣的一面 (Model Steerability Shines a Funny Light)**：模型*经常*会沿用你请求中的模式，而无视更好的替代方案，因为它假设你知道自己想要什么。这可能导致幽默但可能低效的结果。
   - 在项目启动前加入“讨论我的目标、想法和方法，你觉得怎么样？”可以让模型对方案进行**评估**和**优化**。
- **视觉工具在解谜中受挫 (Vision Tool Vexes the Vault)**：一位用户在尝试解决一个基于图像的谜题时遇到了困难，尽管提供了图像，但模型由于其 Vision/OCR 能力的局限性而失败。
   - 即使是 **GPT-4o**，尽管具备先进的能力，有时在为复杂任务准确解读图像时仍显吃力，导致它默认使用效果较差的 Python 工具 OCR。
- **将图像谜题转化为提示词乐园 (Turn Image Puzzles into Prompt Paradise)**：面对基于图像的谜题时，将挑战转换为**文本格式**能大幅提高模型的成功率。
   - 用户发现通过使用唯一的符号来代表图像组件，可以让模型更高效地处理信息。
- **请对提示词的进展给予表扬 (Praise Prompts Progress, Please)**：即使初始结果不成功，**表扬模型**也很重要，因为这能强化其对所请求方法的正确执行，并鼓励继续探索解决方案。
   - 这种正向反馈循环培养了一个协作式的解题环境，AI 就像一个充满好奇心的伙伴，识别潜在问题并提出替代方案。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (26 messages🔥): 

> `Model presumption and user intent, Request evaluation and discussion before project start, Solving image-based puzzles with language models, Vision/OCR limitations in language models, Prompt engineering for puzzle-solving` 


- **模型会假设你就是那个意思！ (Models Presume, You Mean It!)**：一位成员讨论了语言模型如何倾向于坚持用户请求的模式，即使有更好的方法，它们也会假设用户的意图是精确的。这凸显了*模型的可引导性越强，它就越相信你言出必行*。
   - 他们指出，模型有时会猜测是你搞砸了提示词而不是去纠正它，因此你应该确保仔细地引导你的模型。
- **项目前的头脑风暴提升模型性能 (Pre-Project Brainstorming Boosts Model Performance)**：在开始项目之前，要求模型评估或讨论请求是一个**极好**的主意。例如使用提示词：*我想实现 X。我开始通过 [这样做] 来实现。讨论我的目标、想法和方法，你觉得怎么样？*
   - 通过这种方式，模型可以进行**引导、探索，并帮助识别最佳方法和潜在担忧**。
- **视觉/OCR 失效，人类来协助！ (Vision/OCR Falls Flat, Human Assists!)**：一位成员分享了一个涉及解读图像中图标的谜题，并指出模型在处理基于图像的挑战时非常吃力，其 **Vision/OCR** 尚不足以保证 100% 的准确性。
   - 他们认为模型需要针对此类任务进行更多训练，但也指出通过人工协助，我们可以通过清理图像解读结果来“修复”这个问题。
- **提示词构建铺就解谜之路 (Prompt Crafting Paves Puzzle-Solving Path)**：成员们讨论了各种帮助模型解谜的提示策略，包括将 ":6386blueicon:" 之类的内容转换为 1 个符号，使用像跟朋友聊天一样的措辞，或者将代码作为文本提供。
   - 一位成员通过使用如下提示词将图标名称替换为唯一符号取得了成功：*始终将以下文本中的以下部分替换为不同的部分。如果文本中不存在这些部分，则不要替换。*
- **个性化表扬驱动积极的模型行为 (Personalized Praise Drives Positive Model Behavior)**：一位成员即使在模型未成功时也会给予表扬，因为它遵循了请求的方法和个性化设置。模型就像一只对着我没要求它做的事情嗅来嗅去的狗——就像 **E.T. 用发光的手指**指向外星人不理解的事物。
   - 这会鼓励模型突出潜在的疑虑，从而促进协作式的问题解决。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1347749343087689822)** (1 条消息): 

> `LM Studio v0.3.12, QwQ Template Bug, RAG Chunking Speed, MLX Models on exFAT` 


- **LM Studio 发布 v0.3.12 快速更新**：**LM Studio v0.3.12** 现已作为稳定版本发布，包含 Bug 修复和性能改进，用户可以通过应用内更新或从[下载页面](https://lmstudio.ai/download)进行升级。
- **QwQ 模板 Bug 已修复**：此次更新修复了一个导致 *"OpenSquareBracket !== CloseStatement"* 错误的 **QwQ 32B jinja 解析 Bug**。
- **RAG 分块速度提升**：新版本显著提高了用于检索的文件分块速度，增强了 **Retrieval-Augmented Generation (RAG)** 的性能。
- **外部 exFAT 驱动器与 MLX 模型兼容性改善**：修复了一个导致下载到外部 **exFAT** 驱动器的 **MLX 模型** 无法正确索引的 Bug，[针对 Mac 用户](https://lmstudio.ai/download?os=mac)。



**提及的链接**：<a href="https://lmstudio.ai/blog/lmstudio-v0.3.12">LM Studio 0.3.12</a>：Bug 修复和 RAG 文档分块速度改进

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1347546479564165284)** (311 条消息🔥🔥): 

> `Open Source LLM for coding tasks on M2 Macbook, Qwen Coder vs Claude for code generation, Managing context length with LLMs, Draft models for faster token generation, Hardware considerations for LLM performance` 


- **M2 Macbook 迎来编程 LLM**：对于在配备 16GB RAM 的 Macbook M2 Pro 上进行的编程任务，成员们建议将 **Qwen Coder 14B** 作为可行的开源 **LLM**，尽管与 Claude 等模型相比，预期应有所保留。
   - 共识是 16GB RAM 存在限制，用户可能需要*节制其他内存使用*。
- **Unsloth 获得微调支持**：一位成员询问了在 LM Studio 上进行微调的事宜，另一位成员建议关注 **Unsloth**，因为它能让大语言模型的微调更快且占用更少内存，并参考了 [Unsloth 文档](https://docs.unsloth.ai/get-started/fine-tuning-guide)。
   - 值得注意的是，微调通常比推理（Inference）更耗费资源。
- **上下文长度获得压缩**：一位成员提出了一种 *pack and unpack*（打包与解包）方法来管理上下文长度，建议将 VRAM 中的文本压缩到其大小的 1/3，但其他人澄清说**上下文是以 Tokens 形式存储的**，这本身已经提供了压缩，使得提议的方法效果不佳。
   - 有人建议总结（Summarization）或 **RAG** 可能是处理长对话的更好选择。
- **Draft 模型加速 Token 速率**：成员们讨论了使用较小的量化模型作为 **Draft 模型** 来提高 Token 生成速度，一位用户报告通过将 **Q8_0 的 mistral_small** 与 **i1-IQ1_S** 作为 **Draft 模型** 配合使用，在两块 **3090** 上速度从 *18 t/s 跃升至 30 t/s*。
   - 另一位成员分享了他们对不同量化变体的[经验](https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/)，指出 **Q2_k** 和 **IQ2_XS** 达到了类似的 Token 速率，而 **IQ1_S** 则较慢。
- **QwQ 32B 击败 Llama 70B？**：成员们辩论了 QwQ 32B 模型的性能，有人声称其*表现至少能与 Llama 3.3 70b 媲美*，而另一人则引用了 [Dubesor 的基准测试](https://dubesor.de/benchtable)，结果似乎并不一致。
   - 一位用户在 HuggingFace 上分享了他们的 [Qwen_QwQ-32B-GGUF_QX_k_f32 权重](https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32/tree/main)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://lmstudio.ai/blog/introducing-lmstudio-sdk">介绍 lmstudio-python 和 lmstudio-js</a>：适用于 Python 和 TypeScript 的开发者 SDK 现已发布 1.0.0 版本。这是一个用于本地 AI 软件的可编程工具包。</li><li><a href="https://lmstudio.ai/ryzenai">Ryzen AI 上的 LM Studio</a>：在您的 PC 上运行 Llama, Mistral, Mixtral 以及其他本地 LLM，充分利用 RyzenAI 硬件的卓越性能。</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM 基准测试表</a>：未找到描述</li><li><a href="https://installers.lmstudio.ai/win32/x64/0.3.11-1/LM-Studio-0.3.11-1-x64.exe">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF">bartowski/Qwen2.5-Coder-14B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/app/api/endpoints/openai">OpenAI 兼容性 API | LM Studio 文档</a>：向 Chat Completions（文本和图像）、Completions 和 Embeddings 端点发送请求</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://wccftech.com/nvidia-rtx-pro-6000-blackwell-gpu-more-cores-than-rtx-5090-24064-96-gb-gddr7-600w/">NVIDIA RTX PRO 6000 Blackwell GPU 核心数比 RTX 5090 多 11%：总计 24,064 个核心，配备 96 GB GDDR7 显存和 600W TBP</a>：NVIDIA 的 RTX PRO 6000 "Blackwell" GPU 将是一款性能怪兽级的工作站显卡，拥有超过 2.4 万个 CUDA 核心和 96 GB VRAM。</li><li><a href="https://huggingface.co/docs/transformers/training">微调预训练模型</a>：未找到描述</li><li><a href="https://tenor.com/view/for-you-bane-the-dark-knight-rises-tom-hardy-gif-21912820">For You Bane GIF - For You Bane The Dark Knight Rises - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32/tree/main">Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32 在 main 分支</a>：未找到描述</li><li><a href="https://youtu.be/X_wLVgMzSH4">专家展示为何因 AI 引发第三次世界大战几乎不可避免</a>：AGI, OpenAI, Elon Musk 和第三次世界大战。访问 Ground News 以对比新闻报道，识别媒体偏见并避开算法。在 https://gr... 获取订阅 6 折优惠</li><li><a href="https://github.com/Mozilla-Ocho/llamafile">GitHub - Mozilla-Ocho/llamafile：通过单个文件分发和运行 LLM。</a>：通过单个文件分发和运行 LLM。通过在 GitHub 上创建账户为 Mozilla-Ocho/llamafile 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/tree/main">bartowski/microsoft_Phi-4-mini-instruct-GGUF 在 main 分支</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：第一次使用 Unsloth？</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">lmstudio 中 qwq-32b 模型的问题。 · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>：哪个版本的 LM Studio？例如：LM Studio 0.3.11。操作系统？Mac。Bug 是什么？在与 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa..."
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)** (298 条消息🔥🔥): 

> `9070 XT vs 7900 XTX, Windows 上的 ROCm 支持, Vulkan 性能, AMD 驱动问题, GPU 显存与带宽` 


- **9070 XT 有时性能优于 7900 XTX**：在 AI 任务中，**9070 XT** 通常比 **7900 XTX** 更快。在使用 Vulkan 的 **Qwen2.5 coder 14b Q8_0 模型**上，前者可达到 **44 tok/sec**，而后者为 **31 tok/sec**，这是因为在讨论时 9070 XT 在 Windows 上尚不支持 ROCm。
   - 尽管 **7900 XTX** 拥有比 **9070 XT** 更多的 **CU** (Compute Units)，但新架构赋予了后者优势。
- **Vulkan vs ROCm：性能泥潭**：据报道，AMD 上的 Vulkan 性能存在 Bug，运行速度仅为 ROCm 的约 **1/3**。但由于驱动问题，一些用户发现 Vulkan 反而比 ROCm 更快。这种情况在驱动版本 **24.12.1** 左右有所改变，当时通过牺牲 Vulkan 性能“修复”了该问题，但在 **25.1.1** 之后又回到了未修复状态。
   - ROCm 是 AMD 试图替代 CUDA 的尝试，但在实现过程中遇到了很多问题，包括*碎片化以及为支持新架构和 GPU 而导致的二进制文件体积过大*。
- **4060 Ti 16GB：廉价 VRAM 之王？**：**4060 Ti 16GB** 被推荐为 CUDA 的廉价选择，它拥有 **16GB VRAM** 且功耗较低（约 **160W**），性能优于 **3060 12GB**。
   - 虽然其总线（bus）较弱，但在没有 ROCm 那些麻烦事的情况下，它能以约 **$500** 的价格提供比纯 CPU 更快的推理速度，不过无法拆分 Diffusion 模型是一个缺点。
- **功耗：寻找能效甜点位**：讨论涵盖了 GPU 功耗，指出将 TDP 降低 **40%** 可能仅会导致 **10%** 的性能下降，这表明制造商正试图突破性能墙。
   - 现代 GPU 运行温度更低、电压更低，但缺乏保险丝等安全措施；降压（undervolting）有助于 9070 (non-XT) 获得大幅频率提升。
- **尺寸之战：主板与机箱兼容性**：超大尺寸的 GPU 给机箱和主板带来了兼容性问题，一位用户表示偏好尺寸在 **267x112 mm** 以下的显卡，以适配各种配置。
   - 据信 Nvidia 强制 AIB 厂商提供三槽方案，是为了让 3090 无法在消费级 PC 以外的地方使用，这一直是一个问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://getfancontrol.com/">Fan Control - 一款针对 Windows 的高度专注的风扇控制软件</a>：未找到描述</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090-power-limiting-and-undervolting-test-shows-only-8-performance-drop-at-half-the-tdp">NVIDIA GeForce RTX 4090 功耗限制与降压测试显示，在 TDP 减半的情况下性能仅下降 8% - VideoCardz.com</a>：NVIDIA RTX 4090 能效测试（功耗限制/降压）。QuasarZone 对 NVIDIA 旗舰级 GeForce RTX 4090 GPU 的功耗限制与降压进行了有趣的对比。它...</li><li><a href="https://youtu.be/KqKJN7MGZGQ">闲聊当前的 GPU 价格与供应危机。</a>：文本版：Nvidia 和 AMD GPU 均由 TSMC 制造。一颗 9700X 是 70mm^2 的 TSMC 4nm 芯片，零售价约为 ~300 美元；一颗 9070XT 是 357mm^2 的 TSMC 4nm 芯片，MSRP 为 599 美元...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1348686201267163239)** (1 messages): 

> `Aider v0.76.0, Thinking/Reasoning Models, LLM Notifications, Model Support, Tree-sitter Language Pack` 


- ****Aider v0.76.0** 发布，增强了 Reasoning Models 并增加了 Notifications！**: Aider v0.76.0 引入了对 [thinking/reasoning models](https://aider.chat/docs/config/reasoning.html) 的改进支持，包含用于控制 token 预算的 `--thinking-tokens` 等功能。
   - 该版本还包含了当 [LLM 响应就绪时的通知功能](https://aider.chat/docs/usage/notifications.html)，通过 `--notifications` 标志开启。
- ****New Model Support** 扩展，新增 QWQ 32B 和 Claude 3.7！**: 新版 Aider 增加/改进了对许多模型/提供商的支持，例如 **QWQ 32B**、**DeepSeek V3**（在 OpenRouter 上免费）以及在 OpenRouter、Bedrock 和 Vertex AI 上的 **Claude 3.7 Sonnet** 模型，使用 `--model openrouter/deepseek/deepseek-chat:free`。
   - 默认模型已更新为 OpenRouter 上的 **Claude 3.7 Sonnet**，并增加了对 OpenRouter 上 **GPT-4.5-preview** 和 **Claude 3.7 Sonnet:beta** 的支持。
- ****Tree-Sitter** 获得 Language Pack，**Git** 问题得到修复！**: Aider v0.76.0 切换到 `tree-sitter-language-pack` 以支持 tree sitter。
   - 此外，感谢 Akira Komamura，该版本修复了读取暂存文件时的 **Git errors** 处理，并改进了 **Git identity retrieval** 以遵循全局配置。
- ****SSL 和 LLM** 获得更好的 Error Handling！**: 实现了模型信息请求的 **SSL verification control** 改进，以及增强的空 LLM 响应处理，现在提供更清晰的警告信息。
   - Aider 提供为 **Bedrock 和 Vertex AI models** 安装依赖项的功能，并且弃用了模型快捷参数（如 --4o, --opus），转而推荐使用 --model 标志。
- ****Aider-Authored Code** 统计数据已澄清！**: 发布说明澄清，*Aider 编写了此版本中 85% 的代码*，且[统计数据基于 aider 仓库的 git commit 历史](https://aider.chat/docs/faq.html#how-are-the-aider-wrote-xx-of-code-stats-computed)。
   - 该指标反映了 Aider 通过自动化流程直接贡献的代码量。



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: Release notes and stats on aider writing its own code.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1347528831614844980)** (451 messages🔥🔥🔥): 

> `AI21 Maestro, Copilot suspension, DeepSeek R2 release, X cyberattack, Refact AI` 


- **AI21 Maestro 发布，用于 AI 编排**: [AI21 Labs](https://www.ai21.com/blog/introducing-ai21-maestro-an-ai-planning-orchestration-system) 发布了 **AI21 Maestro**，这是一个旨在解决复杂任务的系统，同时发布的还有 **Jamba 1.6** 系列开源模型，其混合架构在长上下文任务中表现出色。
   - 据报道，**Jamba 1.6** 模型在质量和速度上领先于开源模型，并支持 **256K** 上下文窗口。
- **Copilot API 导致账号封禁**: 一名成员报告称，因在 aider 中轻度使用 **Copilot API** 而导致 Copilot 账号被封禁，并提醒他人注意潜在风险。
   - 随后讨论了封禁是由于账号共享还是速率限制问题，并分享了 [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md) 的链接。
- **DeepSeek R2 发布，目标直指编程**: 据传 **DeepSeek R2** 的发布日期定于 3 月 17 日，据 [此 X 帖子](https://x.com/tanvitabs/status/1899006509733814746?s=46) 称，它将以更低的成本提供更好的编程能力、多语言推理能力和更高的准确性，挑战 **Claude Sonnet 3.7**。
- **X 遭受“大规模网络攻击”**: 用户报告访问 X 出现广泛问题，[Elon Musk 将停机归咎于大规模网络攻击](https://x.com/elonmusk/status/1899149509407473825)，尽管有人怀疑这只是 ManusAI 所为，开玩笑说 *Maybe keep it this is just manus*。
- **Refact AI 受到关注**: 成员们对 **Refact AI** 表现出兴趣，它具有聊天、自动补全功能，并即将推出 Agent。一位成员提到 *价格与 cursor 相同，但没有那些废话，且每月请求次数是其 5 倍*。
   - 一位用户分享了对 **Refact AI** 的 token 使用情况的担忧，质疑上下文是否无限期维持：*我只让它运行了 1 个任务，它难道不显示消耗了多少 tokens 与 output 的对比吗？所以等等，它基本上永远在同一个上下文中工作？就像它不会从初始状态脱离一样？*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://www.ai21.com/jamba/">Jamba 1.6: 企业部署的最佳开源模型</a>: 探索由 AI21 开发的 Jamba —— 一款专为准确性、效率和强大文本生成能力而构建的前沿长上下文 AI 开源模型。</li><li><a href="https://www.tomsguide.com/news/live/x-down-twitter-outage-march-2025">X / Twitter 再次宕机 —— 潜在恢复情况的实时更新</a>: X 前一分钟还在运行，下一分钟就又宕机了</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/model-aliases.html">模型别名</a>: 为模型分配便捷的短名称。</li><li><a href="https://aider.chat/docs/config/reasoning.html">推理模型</a>: 如何配置来自二级供应商的推理模型设置。</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://www.youtube.com/@pbsspacetime">PBS Space Time</a>: Space Time 与我们的天体物理学家主持人 Matthew 一起探索外太空、疯狂的天体物理学、科幻的可能性，以及任何你能想到的地球之外的事物...</li><li><a href="https://x.com/tanvitabs/status/1899006509733814746?s=46">来自 Tanvi (@tanvitabs) 的推文</a>: 🚨突发：DeepSeek R2 已定下发布日期 —— 3 月 17 日，Claude Sonnet 3.7 可能有麻烦了，因为 DeepSeek R2 声称：1. 更好的代码能力 2. 多语言推理 3. 更高的准确性...</li><li><a href="https://x.com/kimmonismus/status/1898332202288472551>">来自 Chubby♨️ (@kimmonismus) 的推文</a>: 我问开发人员稍后是否可以做一个更好的预览。但先让你尝个鲜。我询问了模型关于某种疾病的治疗方法，并要求提供非传统的解决方案。我...</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/ericc-ch/copilot-api/blob/master/README.md">GitHub 上的 copilot-api/README.md</a>: GitHub Copilot API 封装器，使其兼容 OpenAI - ericc-ch/copilot-api</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 jian (@jianxliao) 的推文</a>: 所以... 我只是简单地让 Manus 给我 "/opt/.manus/" 下的文件，它就直接给我了，他们的沙箱运行时代码...  &gt; 它是 Claude Sonnet &gt; 它是带有 2 个... 的 Claude Sonnet</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?s=46&t=ggmESCIXF0nYw8_ks">来自 jian (@jianxliao) 的推文</a>: 所以... 我只是简单地让 Manus 给我 "/opt/.manus/" 下的文件，它就直接给我了，他们的沙箱运行时代码...  &gt; 它是 Claude Sonnet &gt; 它是带有 2 个... 的 Claude Sonnet</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better: 为 LLM 提供上下文的元模板语言 :D</a>: 为 LLM 提供上下文的元模板语言 :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/blob/main/docs/language_tutorial.md">GitHub 上的 yet_another_llm_project_but_better/docs/language_tutorial.md</a>: 为 LLM 提供上下文的元模板语言 :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://x.com/elonmusk/status/1899149509407473825">来自 Elon Musk (@elonmusk) 的推文</a>: 𝕏 遭受了（且仍在遭受）大规模网络攻击。我们每天都会受到攻击，但这次动用了大量资源。要么是一个大型协调组织，要么是一个国家参与其中。正在追踪...</li><li><a href="https://tenor.com/view/yes-gif-22712908">Yes GIF - Yes - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better.git">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better: 为 LLM 提供上下文的元模板语言 :D</a>: 为 LLM 提供上下文的元模板语言 :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/x1xhlol/v0-system-prompts-models-and-tools">GitHub - x1xhlol/system-prompts-and-models-of-ai-tools</a>: 通过在 GitHub 上创建账号，为 x1xhlol/system-prompts-and-models-of-ai-tools 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/HISTORY.md">GitHub 上的 aider/HISTORY.md</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号，为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/BerriAI/litellm/commit/

f899b828cf11e285372676f38ead92a66c91bab4">支持 OpenRouter 流式传输中的 `reasoning_content` (#9094) · BerriAI/litellm@f899b82</a>: * feat(convert_dict_to_response.py): 支持 OpenRouter 格式的推理内容 * fix(transformation.py): 修复带有推理内容的 OpenRouter 流式传输 修复了 https://github.com/BerriAI/lite...</li><li><a href="https://news.ycombinator.com/item?id=42672790">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1347529089573064776)** (123 messages🔥🔥): 

> `没有 API Key 的 aider，MCP Agent 集成，aider 脚本编写，OpenRouter 速度慢，移除 repo map 上下文中的 token` 


- **使用 /run 时 Aider 需要 API Key**：一位用户在使用 `/run` 命令时因缺少 API Key 而失败，需要关于如何在 aider 上下文中加入 API Key 的帮助。
   - Paul Gauthier 指出，*运行 aider 时设置的任何环境变量都应该传递给 /run 命令*。
- **关于 MCP Agent 集成安全性的意见分歧**：一位成员询问了集成 **MCP Agent** 的计划，另一位成员警告说 **MCP** 不安全，不适合生产环境使用。
   - 另一位成员反驳说，这*就像在说 REST API 不安全*，并链接到一篇关于[保护 Anthropic Model Context Protocol 安全的博客文章](https://www.raito.io/post/how-to-secure-anthropics-model-context-protocol)。
- **Aider 脚本无法看到网页内容**：一位用户报告说，在使用 `/web` 将网页内容放入聊天历史记录后，aider 的下一次调用无法看到它。
   - 用户后来解决了这个问题，没有进一步解释。
- **OpenRouter 导致 Aider 性能困扰**：一位用户报告了 Aider 中 **OpenRouter** 的缓慢问题，存在 30-60 秒的延迟和频繁的挂起。
   - 另一位用户指出，在使用 litellm 时，Aider 上有一个未解决的问题，即默认情况下不打印 thinking tokens。
- **从 Aider 的 Repo Map 中排除图标以减少 Token 数量**：一位用户想知道是否有办法阻止 aider 在 repo map 中包含图标名称，以减少发送给 LLM 的 token 数量。
   - 一位用户建议使用 `.aiderignore` 文件，而另一位建议使用 `--map-tokens 0` 来停止使用 repo map。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.raito.io/post/how-to-secure-anthropics-model-context-protocol">如何降低 Anthropic Model Context Protocol 的安全风险</a>: Anthropic 新推出的 Model Context Protocol (MCP) 通过提供与各种数据源的通用连接，简化了 AI Agent 对组织数据的访问。虽然这开启了宝贵的机遇...</li><li><a href="https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command">Playwright 在使用 codegen 命令打开时不支持 https url</a>: npx playwright codegen https:/// page.goto: net::ERR_CERT_AUTHORITY_INVALID at ...&#xA;如何通过传递输入参数或身份验证凭据，使用 codegen 命令打开 https url</li><li><a href="https://x.com/victormustar/status/1898001657226506362">Victor M (@victormustar) 的推文</a>: QwQ-32B 永远改变了本地 AI 编程 🤯 我们现在在家里就能拥有 SOTA 性能。分享我的技术栈 + 技巧 ⬇️</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: 一个用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。也可以运行一个 MCP Server 来帮助你管理所有的 MCP Server</a>: 一个用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。也可以运行一个 MCP Server 来帮助你管理所有的 MCP Server - lutzleonhardt/mcpm-aider</li><li><a href="https://github.com/Aider-AI/aider/issues/3086">建议：在推理模型给出输出之前查看它们的思考过程。 · Issue #3086 · Aider-AI/aider</a>: 问题：由于在使用 DeepSeek R1 和 Perplexity: Sonar Reasoning 等推理模型时面临很长的等待时间，即即使是简单的提示，平均等待时间也长达数分钟：忽略...
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1347804826687508533)** (5 messages): 

> `Effective Commit Messages, Manus AI, Aider NotebookLM Integration` 


- **通过更好的 Commit Messages 简化代码审查**：[refactoringenglish.com](https://refactoringenglish.com/chapters/commit-messages/) 上的一篇博文指出，尽管 **有效的 commit messages** 经常被忽视，但它们能简化代码审查过程并有助于长期代码维护。
   - 该文章基于 **20 年的软件开发经验**，概述了什么是好的 commit message，包括如何辅助代码审查人员和传达变更内容。
- **Manus AI 接受测试**：一段 [YouTube 视频](https://www.youtube.com/watch?v=D6jxT0E7tzU) 展示了最新的 AI 新闻，重点关注 **LLMs** 和 **GenAI**，包括对 **Manus AI** 各种用例和提示词（prompts）的测试。
   - 一位用户总结了该视频：*"他获得了 Manus 的访问权限并测试了一堆用例/提示词……结果确实很有趣"*；另一位用户补充道，*"它只是集成了 29 个工具和 browser_use 的 Claude 3.7"*。
- **Aider 集成 NotebookLM**：一段 [YouTube 视频](https://www.youtube.com/watch?v=WNdEX9IAbDo) 演示了使用 **NotebookLM** 配合 **Aider** 在大型代码库中查找上下文的工作流，特别强调了 **/export-context 命令**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://refactoringenglish.com/chapters/commit-messages/">How to Write Useful Commit Messages</a>：软件开发者的有效写作指南</li><li><a href="https://www.youtube.com/watch?v=D6jxT0E7tzU">Manus is out of control</a>：最新的 AI 新闻。了解 LLMs、Gen AI 并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://www.youtube.com/watch?v=WNdEX9IAbDo)">Aider loves NotebookLM: Effortless Large-Repo Analysis with the /export-context Command</a>：使用 Aider 的 /export-context 命令轻松将整个仓库与 NotebookLM 集成。在本集中，我探索了如何集成大型代码仓库...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1347540128209440780)** (529 messages🔥🔥🔥): 

> `Fine-tuning models with reward models, Tool use accessibility, Anthropic's marketing, AGI as a meaningful concept, Graph system on TinyStories dataset` 


- **请求奖励模型文档**：一位成员询问了关于使用奖励模型（reward models）微调模型的文档或教程。
   - 该成员表示希望进一步了解这一过程。
- **开源自主 AI Agent Manus 发布**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=CFo1iTd_Cc8)，宣布 **Manus** 发布，称其为*全球首个开源自主 AI Agent*。
   - 另一位成员链接了一篇 [动点科技 (Technode) 的文章](https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/)，报道了 **Manus** 受到关注并在 GAIA 基准测试中取得 state-of-the-art 结果。
- **探索用于语言建模的 Hebbian 学习**：一位成员正在实验一种[非反向传播的无监督学习系统](https://discord.com/channels/687756328055472248/687756328055472251/1347686594691035186)，该系统基于节点频率和语义关联将 token 模式分配给单词和概念，灵感源自 **Hebbian 学习**。
   - 尽管目前在实现和评估方面面临挑战，但该系统旨在通过分层图结构在字符级对语言建模，以开发代表单词和概念的高阶节点。
- **探索用于 LLM 记忆增强的 Letta 框架**：一位成员分享了一篇 [博文](https://tersesystems.com/blog/2025/03/07/llm-complexity-and-pricing/)，详细介绍了使用 Agent 框架 **Letta** 为 LLMs 添加记忆和工具以处理烹饪相关任务。
   - 该文章讨论了如何确定一项工作所需的最小 LLM 复杂度，并探讨了 LLM 定价以及 Openrouter 的作用。
- **QwQ-32B 微调挑战与解决方案**：一位用户报告了运行 **QwQ-32B** 时的困难，遇到了无限生成和重复问题，而另一位成员分享了一个 [链接](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively)，解释了如何更有效地使用该模型。
   - 注意到 **QwQ-32B** 对采样设置高度敏感，比预览版更甚。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tersesystems.com/blog/2025/02/14/adding-memory-to-llms-with-letta/">
    
      Adding memory to LLMs with Letta &middot; Terse Systems
    
</a>
</li>
</ul>

</div>

  </a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: LoRA (Low-Rank Adaptation) 是一种广泛使用的针对 Large Language Models 的参数高效微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省内存。在...</li><li><a href="https://tersesystems.com/blog/2025/03/07/llm-complexity-and-pricing/">
    
      LLM Complexity and Pricing &middot; Terse Systems
    
  </a>: 未找到描述</li><li><a href="https://www.anthropic.com/news/anthropic-s-recommendations-ostp-u-s-ai-action-plan">Anthropic 对美国 AI 行动计划向 OSTP 提出的建议 </a>: Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively">教程：如何有效地运行 QwQ-32B | Unsloth 文档</a>: 如何通过我们的错误修复有效地运行 QwQ-32B，避免无休止的生成，并支持 GGUF。</li><li><a href="https://gr.inc">General Reasoning</a>: 让最先进的推理能力对每个人都触手可及。</li><li><a href="https://x.com/_akhaliq/status/1897873813083238713?s=46">来自 AK (@_akhaliq) 的推文</a>: PokéChamp 专家级 Minimax Language Agent。PokéChamp 以巨大优势超越了所有现有的基于 LLM (76%) 和基于规则的机器人 (84%)，包括在对抗之前的...中持续获胜 (64%)。</li><li><a href="https://www.youtube.com/watch?v=CFo1iTd_Cc8">中国发布全球首个自主 AI Agent... 开源 | Manus</a>: 最新的 AI 新闻。了解 LLM、Gen AI，并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 世界的最新动态。</li><li><a href="https://www.youtube.com/watch?v=D6jxT0E7tzU">Manus 失控了</a>: 最新的 AI 新闻。了解 LLM、Gen AI，并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 世界的最新动态。</li><li><a href="https://x.com/jianxliao/status/1898861051183349870">来自 jian (@jianxliao) 的推文</a>: 所以... 我只是简单地要求 Manus 给我 "/opt/.manus/" 下的文件，它就直接给我了，那是他们的 sandbox 运行时代码...  &gt; 它是 Claude Sonnet &gt; 它是带有 2... 的 Claude Sonnet。</li><li><a href="https://huggingface.co/datasets/GeneralReasoning/GeneralThought-195K">GeneralReasoning/GeneralThought-195K · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=4bhPnaUVaxA">扩展 RL：具有长 Chain-of-Thought 和 4 种模式的 3B AI</a>: 总之，这两项新的 AI 研究（见下文）虽然在实验设置和关注领域上有所不同，但共同提供了一个全面的路线图...</li><li><a href="https://gist.github.com/jlia0/db0a9695b3ca7609c9b1a08dcbf872c9">Manus 工具和提示词</a>: Manus 工具和提示词。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://huggingface.co/datasets/MasterControlAIML/R1-Reasoning-Unstructured-To-Structured">MasterControlAIML/R1-Reasoning-Unstructured-To-Structured · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/simplescaling/s1K-1.1">simplescaling/s1K-1.1 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://technode.com/2025/03/07/chinas-ai-agent-manus-gains-traction-amid-growing-demand-for-autonomous-ai/">中国 AI Agent Manus 在自主 AI 需求增长中受到关注 &#183; TechNode</a>: 3 月 6 日，中国的 AI Agent Manus 在中国社交媒体平台微博上成为热门话题。据其团队介绍，Manus 是一款旨在...的自主 AI Agent。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1348237146712051742)** (11 messages🔥): 

> `Vibe coding 基准测试, LLM 创造力, Sonnet 的训练元目标, Claude code 检查图像` 


- ****LLM 在 Vibe Coding 基准测试中表现出色****：一名成员介绍了一个 *"vibe coding"* 基准测试，要求 LLM 创建 **Python raytracers**（光线追踪器）来渲染带有彩色光源的有趣场景，以此评估它们的审美创造力。
   - 他们发现，像 **Sonnet** 这样的 [某些模型](https://cdn.discordapp.com/attachments/1154120232051408927/1348237146389086248/image.png?ex=67d00cb0&is=67cebb30&hm=f6f28943cb16f8fc9c67c2fed8170a06cc0f9a0308c7f30f3b0690dbece4a14a) 在优化代码输出创造力方面脱颖而出，而其他模型只是简单复制基础示例。
- ****Sonnet 的元目标引发关注****：**Sonnet** 在 *"vibe coding"* 基准测试中展示的独特创造力引发了人们的猜测，认为其训练中可能存在一个优化代码输出创造力的 **meta-objective**（元目标）。
   - 实验表明，与 **Sonnet 3.5** 相比，**Sonnet 3.7** 在追求更令人印象深刻的图像方面表现出更高的偏差（bias）和方差（variance），导致代码量增加了一倍。
- ****Claude Code 批判其艺术创作****：在使用光线追踪器提示词测试 **Claude code** 时，一名成员报告称，它会检查生成的图像，并在图像不够华丽时对代码进行修改。
   - 展示了一张生成的 [图像](https://cdn.discordapp.com/attachments/1154120232051408927/1348449672498249758/image.png?ex=67d029de&is=67ced85e&hm=4a0efcff8ca08b67db01cdcd70560ca39d3240b2143ced9bb7db7926aaee9185)，展示了这种迭代改进过程的结果。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/ksshumab_/status/1897560985315238046?s=46
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

rikufps: https://arena.hume.ai/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/ksshumab_/status/1897560985315238046?s=46
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1347524188436693063)** (518 messages🔥🔥🔥): 

> `注册表微调, 程序的内存占用, 量化过程, LocalDocs 问题, 语音识别与 AI 集成` 


- **修改注册表可能导致蓝屏**：一名成员分享了一个轶事，他在发现某个 `.dll` 文件消耗了 **20% 的 RAM** 后将其删除，结果导致重启后出现 **蓝屏**。
   - 另一名成员建议，如果有人 *"在凌晨 2 点进行微调且不记得做了什么"*，那么最好的做法是 **备份个人文件并重装系统**。
- **程序的内存占用**：一名成员描述了他们下班后打开 **Task Manager**（任务管理器）按内存使用量对进程进行排序，并在小酌时观察 CPU 利用率的习惯。
   - 他们会卸载那些在不经常使用的情况下向公司回传数据的程序，并表示 *"我几乎了解我电脑上运行的全部 206 个进程"*。
- **量化：揭秘浮点数**：一名成员询问模型从 **f32 量化到 f16** 意味着什么，以及这是否意味着每个参数只有 **16 个点**。
   - 另一名成员回答道：*"Float 16 是 16 位，通常不被视为量化。在使用 15.5GB VRAM 的情况下，在消费级场景中通常不值得使用"*。
- **Inception 的基于扩散的语言模型**：[InceptionLabs](https://inceptionlabs.ai/) 引入了 **基于扩散（diffusion-based）的语言生成**，从 **Midjourney 和 Sora** 等图像和视频 AI 系统中汲取灵感，强调速度、质量和生成控制。
   - 该项目已开源了部分组件，例如 [GitHub 上的 LLaDA](https://github.com/ML-GSAI/LLaDA)，但目前还无法下载，这使得许多 GPT4All 用户对其兴趣降低，尽管有人认为我们可能 **很快就会看到 10 倍的速度提升**。
- **利用带有乱码的翻译提示词进行攻击**：一名成员提出了一种 *"让试图将乱码复制粘贴到 Google Translate 的人感到非常不适的利用方式"*，详细说明了 **Google Translate** 如何将整个提示词转换为 URL。
   - 该成员解释说，URL 中未翻译的代码片段可用于 URL 注入，因为 *"基于字典的 XSS 漏洞利用可能非常罕见"*。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="http://$HOST:9999/v1/embeddings"">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/,">Open LLM Leaderboard - open-llm-leaderboard 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard">Open ASR Leaderboard - hf-audio 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/multimodalart/LLaDA">LLaDA - multimodalart 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/seedboxai/KafkaLM-7B-German-V0.1-DPO">seedboxai/KafkaLM-7B-German-V0.1-DPO · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings.">常见问题解答</a>: GPT4All：在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/nvidia/canary-1b">nvidia/canary-1b · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/warpdotdev/Warp">GitHub - warpdotdev/Warp: Warp 是一款现代化的、基于 Rust 的终端，内置 AI，可帮助您和您的团队更快地构建出色的软件。</a>: Warp 是一款现代化的、基于 Rust 的终端，内置 AI，可帮助您和您的团队更快地构建出色的软件。 - warpdotdev/Warp</li><li><a href="https://github.com/nomic-ai/gpt4all">GitHub - nomic-ai/gpt4all: GPT4All：在任何设备上运行本地 LLM。开源且可用于商业用途。</a>: GPT4All：在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-get-embeddings.sh">LLM-Helpers/bin/rcd-llm-get-embeddings.sh (main 分支) · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://tenor.com/view/fun-cave-men-old-kick-fuck-you-gif-13869846">有趣的山顶洞人 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/gnusupport/LLM-Helpers/tree/main/bin">LLM-Helpers/bin (main 分支) · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://github.com/ML-GSAI/LLaDA">GitHub - ML-GSAI/LLaDA: “Large Language Diffusion Models” 的官方 PyTorch 实现</a>: “Large Language Diffusion Models” 的官方 PyTorch 实现 - ML-GSAI/LLaDA</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/get_ethernet_interface.sh">LLM-Helpers/bin/get_ethernet_interface.sh (main 分支) · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://huggingface.co/spaces/occiglot/euro-llm-leaderboard">Occiglot Euro LLM Leaderboard - occiglot 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/utter-project/EuroLLM-9B-Instruct">utter-project/EuroLLM-9B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://inceptionlabs.ai/">Inception Labs</a>: 我们正在利用扩散技术开发新一代 LLM。我们的 dLLM 比传统的自回归 LLM 更快、更高效。而且扩散模型更准确，...</li><li><a href="https://en.wikipedia.org/wiki/PRISM">PRISM - 维基百科</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry">最大路径长度限制 - Win32 应用</a>: 从 Windows 10 版本 1607 开始，许多常见的 Win32 文件和目录函数已移除 MAX_PATH 限制。但是，您的应用程序必须选择启用以支持新行为。</li><li><a href="https://huggingface.co/spaces/nvidia/canary-1b">Canary 1b - nvidia 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nvidia/canary-1b/tree/main">nvidia/canary-1b (main 分支)</a>: 未找到描述</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-single-input.sh">LLM-Helpers/bin/rcd-llm-speech-single-input.sh (main 分支) · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-typing.sh">LLM-Helpers/bin/rcd-llm-speech-typing.sh (main 分支) · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-speech-translat

e.sh">LLM-Helpers/bin/rcd-llm-speech-translate.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers</li><li><a href="https://github.com/gnusupport/LLM-Helpers/blob/main/bin/rcd-llm-to-french.sh">LLM-Helpers/bin/rcd-llm-to-french.sh at main · gnusupport/LLM-Helpers</a>: LLM Helpers 是用于维护、训练、运行和推理 Large Language Models 的脚本和程序 - gnusupport/LLM-Helpers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1347538735272689694)** (221 条消息🔥🔥): 

> `实现研究论文, Hugging Face Pro 订阅, AI 安全研究, 使用 HF 和 MCP 自动化数据创建, 视频模型对比: WAN, HUN, LTX` 


- **正确引用：数据集引用建议！**：成员们讨论了在 Hugging Face 上正确引用数据集的方法，包括使用 **BibTeX** 格式并确保学术软件识别所需的正确 URL 参数，参考这个 [Hugging Face Datasets 示例](https://huggingface.co/datasets/Tonic/OpenReasonerZero)。
   - 他们还讨论了如何为数据集申请 DOI，以确保其显示在 ArXiv 标题页上并链接到已发表的论文。
- **GKD Trainer 遇到 Token 问题！**：用户在使用 **GKDTrainer** 处理不同模型架构时遇到了错误，原因是 tokenizer 词表大小（vocab sizes）不同，建议预先对数据进行 tokenizing 以避免这些问题，如该 [GitHub issue](https://github.com/huggingface/trl/issues/3028) 所示。
   - 建议重试以解决问题，如有需要，请咨询 [HuggingFace 官方 GKD 文档](https://huggingface.co/docs/trl/gkd_trainer)。
- **WAN 和 HUN 视频模型势头强劲**：像 **WAN** 和 **Hunyuan i2v** 这样的新视频模型在质量和速度上正在超越 SVD 等旧模型，尽管各有所长，且可以配合 [para attention](https://huggingface.co/docs/diffusers/main/en/optimization/para_attn) 一起使用。
   - 一位成员指出 **Ltxv** 速度极快，在 *h100 上生成 5 秒视频仅需 3 秒*，但效果不如另外两个模型。
- **在 HF 上获取 AI 安全技能**：用户分享了关于 **AI 安全研究** 的资源，包括来自 [复旦大学](https://arxiv.org/pdf/2412.12140)、[Apollo Research](https://static1.squarespace.com/static/6593e7097565990e65c886fd/t/6751eb240ed3821a0161b45b/1733421863119/in_context_scheming_reasoning_paper.pdf) 和 [NationalSecurity.ai](https://www.nationalsecurity.ai/) 的关于自我复制、密谋（scheming）以及滥用潜在危险的论文。
   - 上述每项研究都有简短的 **YouTube** 解释视频。
- **寻找掌握 LLMs 的正确路线图**：成员们推荐了学习 LLM 的资源，包括 YouTube 上的 **斯坦福 CS224N 课程**、**Hugging Face NLP 课程**，以及[这篇关于训练的博客文章](https://huggingface.co/blog/how-to-train)。
   - 一位成员还推荐了一篇[关于训练新 ModernBERT 模型的博客文章](https://huggingface.co/blog/modernbert)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.18008">NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms</a>: 我们介绍了 NotaGen，这是一个旨在探索高质量古典乐谱生成潜力的符号音乐生成模型。受 Large Language Models (LLMs) 成功的启发，NotaGe...</li><li><a href="https://huggingface.co/papers/2502.18008">Paper page - NotaGen: Advancing Musicality in Symbolic Music Generation with Large

Language Model Training Paradigms</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/).">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://obsidian.md/">Obsidian - 磨砺你的思维</a>: 用于私人想法的免费且灵活的应用。</li><li><a href="https://huggingface.co/docs/diffusers/main/en/optimization/para_attn">ParaAttention</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/breadlicker45/bread-midi-dataset">breadlicker45/bread-midi-dataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/how-to-train">如何使用 Transformers 和 Tokenizers 从头开始训练一个新的 Language Model</a>: 未找到描述</li><li><a href="https://stackoverflow.com/questions/65246703/how-does">HuggingFace 的 BertTokenizerFast.from_pretrained('bert-base-uncased') 中的 max_length、padding 和 truncation 参数是如何工作的？</a>: 我正在处理文本分类问题，想使用 BERT 模型作为基础，后接 Dense layers。我想知道这 3 个参数是如何工作的？例如，如果我有 3 个句子...</li><li><a href="https://huggingface.co/docs/trl/gkd_trainer">Generalized Knowledge Distillation Trainer</a>: 未找到描述</li><li><a href="https://x.com/ClementDelangue/status/1897767808076816575?t=f0HsVgnlRua2PLTuPvIELQ&s=19">来自 clem 🤗 (@ClementDelangue) 的推文</a>: 谁说开源对你的业务不利？</li><li><a href="https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt">HuggingFace 的 BertTokenizerFast.from_pretrained('bert-base-uncased') 中的 max_length、padding 和 truncation 参数是如何工作的？</a>: 我正在处理文本分类问题，想使用 BERT 模型作为基础，后接 Dense layers。我想知道这 3 个参数是如何工作的？例如，如果我有 3 个句子...</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/issues/3028">在 Teacher 和 Student 的词表大小不同时蒸馏 Teacher 模型 · Issue #3028 · huggingface/trl</a>: 我正尝试使用来自 trl 的示例代码将 Qwen2.5-7B-Instruct 蒸馏为 Qwen2.5-5B-Instruct，代码包含 import GKDConfig, GKDTrainer from transformers import ( AutoModelForCausalLM, AutoTokenizer, ) NUM_D...</li><li><a href="https://github.com/huggingface/trl/issues/2215">[GKD] 堆叠 log probs 时 Tensor 不匹配 · Issue #2215 · huggingface/trl</a>: 系统信息：最新的源码安装 TRL，目前无法运行 TRL 环境，因为集群已关闭，但我正在从源码安装所有内容。如果需要，将重启集群并运行。信息：官方示例...</li><li><a href="https://ieeexplore.ieee.org/abstract/document/10421308/">探索用于衡量文本相关性的 Embeddings：揭示在线评论中的情感和关系</a>: 在 COVID-19 疫情导致互联网使用量增长 70% 后，全球使用社交媒体的人数有所增加。像 Twitter、Meta Threads、YouTube 等应用...</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1j6bx40/the_highest_quality_quantization_varient_gguf_and/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/pad_truncation">Padding 和 truncation</a>: 未找到描述</li><li><a href="https://docs.dify.ai/development/models-integration/hugging-face">从 Hugging Face 集成开源模型 | Dify</a>: 未找到描述</li><li><a href="https://github.com/sayakpaul/q8-ltx-video">GitHub - sayakpaul/q8-ltx-video: 此仓库展示了如何使用 Q8 kernels 配合 `diffusers` 来优化 LTX-Video 在 ADA GPUs 上的推理。</a>: 此仓库展示了如何使用 Q8 kernels 配合 `diffusers` 来优化 LTX-Video 在 ADA GPUs 上的推理。 - sayakpaul/q8-ltx-video</li><li><a href="https://huggingface.co/spaces/rizavelioglu/vae-comparison">Vae Comparison - rizavelioglu 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/remote_vae">使用 Inference Endpoints 进行解码的远程 VAEs 🤗</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/modernbert">终于，BERT 的替代者来了：介绍 ModernBERT</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-cats-pet-cat-cat-pet-cute-cat-gif-24810247">猫咪 GIF - 猫咪宠物猫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/PleIAs">PleIAs (PleIAs)</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/issues/6815#issuecomment-1996291216).">RuntimeWarning: cast 中遇到无效值 images = (images

* <a href="https://github.com/huggingface/diffusers/issues/6815">255).round().astype("uint8") 输出黑色图像 · Issue #6815 · huggingface/diffusers</a>: 描述 Bug：在运行 stable-diffusion-2-1 时，我收到了运行时警告 "RuntimeWarning: invalid value encountered in cast images = (images * 255).round().astype("uint8")" ...</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-12-at-entry-0-and-35-at-entry-1/46155/2">RuntimeError: stack expects each tensor to be equal size, but got [12] at entry 0 and [35] at entry 1</a>: 我认为 tokenized 文本的长度不一致，正如警告信息所示。如果你将每个 batch 的输入长度调整为一致，我认为错误就会消失。</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/682">huggingchat/chat-ui · Hugging Face Chat 的新设计提案</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions">huggingchat/chat-ui · Discussions</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/speculative-decoding-with-qwen-models/144073">使用 Qwen 模型进行投机采样 (Speculative Decoding)</a>: checkpoint_target_model = "Qwen/Qwen2.5-14B-Instruct" checkpoint_draft_model = "Qwen/Qwen2.5-0.5B-Instruct" target_tokenizer = AutoTokenizer.from_pretrained(checkpoint_target_model...</li><li><a href="https://huggingface.co/datasets/breadlicker45/toast-midi-dataset">breadlicker45/toast-midi-dataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/how-to-set-max-length-properly-when-using-pipeline/125714/5">使用 pipeline 时如何正确设置 'max_length'？</a>: @jiaweihuang prompt = 'What is the answer of 1 + 1?' pipe = pipeline( "text-generation", tokenizer=tokenizer, model=model, do_sample=...</li><li><a href="https://huggingface.co/datasets/breadlicker45/youtube-comments-180k">breadlicker45/youtube-comments-180k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course/chapter7/6">从零开始训练因果语言模型 - Hugging Face NLP 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/mlabonne/llm-course">大语言模型 (LLM) 课程</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct">Qwen/Qwen2-0.5B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct">HuggingFaceTB/SmolLM2-135M-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/smollm">SmolLM - 极速且极其强大</a>: 未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐小模型 (smol models) 的课程。</a>: 一个关于对齐小模型的课程。通过在 GitHub 上创建账号来为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://huggingface.co/models?other=qwen2&sort=trending">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://ia-flow.vercel.app">IA</a>: 未找到描述</li><li><a href="https://github.com/RohanSai22/ia">GitHub - RohanSai22/ia</a>: 通过在 GitHub 上创建账号来为 RohanSai22/ia 的开发做出贡献。</li><li><a href="https://x.com/RohanSai2208/status/1897665936209117546">来自 Rohan Sai (@RohanSai2208) 的推文</a>: 你今天想构建什么创意？🚀通过 #IAFlow，只需输入你的想法，就能看到它转化为一个功能齐全的 Web 应用——由 Gemini 提供支持。编辑、预览和部署，一切都在无缝体验中完成...</li><li><a href="https://youtu.be/cyrrfl0eNYc">AI 研究人员震惊了，AI 现在可以克隆自己！中国 AI 以 90% 的成功率实现自我复制。</a>: 最新的 AI 新闻。了解 LLM、生成式 AI (Gen AI)，并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://youtu.be/0JPQrRdu4Ok">AI 研究人员感到震惊：OpenAI 的新 o1 试图逃跑...</a>: 最新的 AI 新闻。了解 LLM、生成式 AI (Gen AI)，并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://www.nationalsecurity.ai/">超级智能战略 (Superintelligence Strategy)</a>: 《超级智能战略》由 Dan Hendrycks、Eric Schmidt、Alexandr Wang 撰写。AI 的飞速发展正开始重塑国家安全。</li><li><a href="https://youtu.be/IhBuz-cnSNE">超级智能、第三次世界大战与 AI | 前 Google CEO 的震惊警告</a>: 最新的 AI 新闻。了解 LLM、生成式 AI (Gen AI)，并为 AGI 的到来做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1348051027089293405)** (3 messages): 

> `smolagents, PokemonLLMAgentBenchmark, Agent Course Study Focus` 


- **PokemonLLMAgentBenchmark 等待贡献**：一位成员受 ClaudePlaysPokemon 启发，正在使用 *smolagents* 开发 **Pokemon LLM Agent Benchmark**，并正在通过 [GitHub 上的 pull requests](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark) 寻求贡献。
- **学习专注时长不足一小时**：一位成员正在学习 Agent 课程，并意识到他们的学习专注时长**不足一小时**。



**提及的链接**：<a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>：通过在 GitHub 上创建账户来为 CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark 的开发做出贡献。

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1347568249927372993)** (4 messages): 

> `fxtwitter obsolescence, HighlightAI` 


- **fxtwitter 变得过时**：成员们庆祝不再需要 [fxtwitter](https://fxtwitter.com/)，因为嵌入（embeds）现在可以正常工作了。
   - 修复 Twitter 嵌入工具的需求已经结束。
- **HighlightAI 自定义响应**：一位成员分享了 [HighlightAI](https://highlightai.com/) 的链接，描述了其自定义响应的能力。
   - 用户可以使用 *About Me* 部分来告诉 **HighlightAI** 在响应时需要考虑的事项。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://highlightai.com/">Highlight AI | Master your world</a>：获取关于你所见、所闻或所说的任何内容的即时答案。免费下载：highlightai.com</li><li><a href="https://huggingface.co/spaces/mozilla-ai/osm-ai-helper">OpenStreetMap AI Helper - a Hugging Face Space by mozilla-ai</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1347683704306270250)** (19 messages🔥): 

> `Llama-3.2-3B-Instruct Distillation, Differential Privacy Blogpost, AI Neovim config, Qwen_QwQ-32B-GGUF_QX_k_f32 weights, Automated web app testing` 


- **Llama-3 使用 DeepSeek-R1 进行蒸馏**：一位成员在 ServiceNow-AI/R1-Distill-SFT 数据集上使用 **DeepSeek-R1** 蒸馏了 **Llama-3.2-3B-Instruct**，在 10 天内实现了近 **1000 次下载**；该模型可在[此处](https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B)获取。
   - 设置涉及使用 [Axolotl 配置](https://github.com/OpenAccess-AI-Collective/axolotl)，其中包含针对 **base model**、**tokenizer type** 和 **data loading** 的特定设置。
- **差分隐私 (Differential Privacy) 的加噪机制**：一位成员发布了一篇新博客文章，讨论在差分隐私中选择加噪机制的问题；文章标题为 *The Art of Controlled Noise: Laplace*，可在 [Substack](https://open.substack.com/pub/theailandscape/p/the-art-of-controlled-noise-laplace) 上阅读。
   - 他们还提供了一个 [GitHub 仓库](https://github.com/divyanshugit/Inception-of-DP/tree/master/mechanisms)，其中包含 **Laplace** 和 **exponential noise** 的基础实现。
- **最大化 Llama 3.2 3B 的能力**：一位成员分享了他们通过使用 **MoonRide-Index-v7** 选择合适的父模型来最大化 Llama 3.2 3B 能力的尝试，并创建了一个名为 [Llama-3.2-3B-Khelavaster](https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster) 的多个 Llama 3.2 3B 模型的实验性合并版本。
   - 他们还分享了该模型已在 [Ollama](https://ollama.com/moonride/khelavaster) 上可用，并提到它虽然不会击败最好的 7B+ 模型，但对于 3B 模型来说相当不错。
- **结合 Controlnet 和 Inpaint 的 SDXL 面部传输流水线**：一位成员正在测试一种结合 **Controlnet** 与 **Inpaint** 的 **SDXL Pipeline ID transfer** 技术，旨在通过 Stable Diffusion 传输面部 ID，并征求反馈。
   - 演示该技术结果的图像对比可以在[此处](https://imgsli.com/MzU3NDY1)找到。
- **针对推理进行微调的 AclevoGPT-Gemma-2b-CoT**：一位成员介绍了 [AclevoGPT-Gemma-2b-CoT-reasoning](https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning)，这是 Google **Gemma** 的微调版本，增强了高级的 **Chain of Thought** 推理能力。
   - 这种增强使模型在响应前能够“三思而后行”，从而对推理问题给出更准确、更周到的回答。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://imgsli.com/MzU3NDY1">Imgsli</a>: 未找到描述</li><li><a href="https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning">Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32">Rombo-Org/Qwen_QwQ-32B-GGUF_QX_k_f32 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/MultiTransformer/tonic_gradio_bot">Tonic Gradio Bot - a Hugging Face Space by MultiTransformer</a>: 未找到描述</li><li><a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/BywchwByGLA?si=zp0GKZbTBIz6pxVu">First, Do No Harm: On Making AI Safe, Secure, and Trustworthy</a>: 我们展示了三种用于安全、可靠和值得信赖的 AI 开发的端侧协议：Diver | AI Menu meets Toolkit2: Unreal Engine | AI world builder3: Pos...</li><li><a href="https://huggingface.co/suayptalha/DeepSeek-R1-Distill-Llama-3B">suayptalha/DeepSeek-R1-Distill-Llama-3B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/2dghost/VisionPRAI">GitHub - 2dghost/VisionPRAI</a>: 通过在 GitHub 上创建账户，为 2dghost/VisionPRAI 的开发做出贡献。</li><li><a href="https://github.com/mtnwrw/tmq">GitHub - mtnwrw/tmq: End-to-end quantized learning &amp; compression for general neural networks</a>: 通用神经网络的端到端量化学习与压缩 - mtnwrw/tmq</li><li><a href="https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster">MoonRide/Llama-3.2-3B-Khelavaster · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster-GGUF">MoonRide/Llama-3.2-3B-Khelavaster-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://ollama.com/moonride/khelavaster">moonride/khelavaster</a>: 多个 Llama 3.2 3B 模型的实验性合并，由 MoonRide-Index-v7 引导。原帖：https://huggingface.co/MoonRide/Llama-3.2-3B-Khelavaster.</li><li><a href="https://open.substack.com/pub/theailandscape/p/the-art-of-controlled-noise-laplace?r=8zcds&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">The Art of Controlled Noise: Laplace and Exponential Mechanisms in Differential Privacy</a>: Inception of Differential Privacy 系列博客之三</li><li><a href="https://github.com/divyanshugit/Inception-of-DP/tree/master/mechanisms">Inception-of-DP/mechanisms at master · divyanshugit/Inception-of-DP</a>: Inception of Differential Privacy：一个记录 Differential Privacy 概念演进的仓库，从基础原理到当前研究 - divyanshugit/Inception-of-DP
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

chad_in_the_house: 非常酷！我特别喜欢你能在某种程度上防止失真的方式
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347677128551960678)** (2 条消息): 

> `需要 OCR 指导，Blendshapes 博客文章` 


- **请求 OCR 指导**：一位成员请求关于如何完成任务的指导，并链接到了一个 [SharePoint 文档](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4)。
   - 他们发现了 **ocr-2.0**，并询问是否应该根据文档对 **got/ocr-2.0** 模型进行 finetune。
- **博客文章中详细介绍了 Blendshapes**：一位成员分享了一篇关于 **blendshapes** 的 [博客文章](https://medium.com/@samiratra95/blendshapes-a-facial-expressions-representation-6352ecd99009)，包括其起源、定义以及在 computer vision 和 computer graphics 中的用例。
   - 该文章讨论了表示面部的不同方法，如 Landmark vectors、Action units、valence and arousal、face meshes 和 Blendshapes，并引用了其在好莱坞电影中的应用起源。



**提到的链接**：<a href="https://medium.com/@samiratra95/blendshapes-a-facial-expressions-representation-6352ecd99009">Blendshapes: a facial expressions representation</a>：在 computer vision 和 computer graphics 中，有许多表示面部的方法，例如 Landmark vectors、Action units、valence 和……

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1347957346772451433)** (8 messages🔥): 

> `Hermes Function Calling 数据集，Gemma 2B 精度，Serverless API 输入转换，带有 BitsAndBytes 错误的 LoRA Adapter` 


- **Nous Hermes 发布 Function Calling 数据集**：Nous Research 发布了 [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)，这是用于 **Hermes 2 Pro** 系列模型的结构化输出和函数调用数据的集合。
   - 该数据集的特点是对话场景，其中 **AI Agent 解释查询并执行适当的单个或多个函数调用**。
- **Gemma-2-2b 的精度偏好**：一位成员询问为什么 [gemma-2-2b](https://ai.google.dev/models/gemma) 是 **float32**，而 [gemma-2-2b-it](https://ai.google.dev/models/gemma) 是 **bfloat16**。
   - 另一位成员建议，使用较低的精度对于 finetuning 可能更有效率，以最大限度地降低成本或环境影响。
- **Serverless API 自动转换输入**：一位成员询问 Serverless API 是否会自动将输入转换为所选模型的正确模板。
   - 另一位成员表示，通常所有 **API 都会转换为模型模板**。
- **LoRA Adapter 产生 CuBLAS 错误**：一位成员在应用 chat template 并进行 tokenizing 后，使用 `bitsandbytes` 以 **8bit** 量化加载带有 **LoRA adapter** 的模型时遇到了 `cublasLt` 错误。
   - 错误信息包含关于形状不匹配的信息，例如 `shapeA=torch.Size([4096, 4096])` 和 `shapeB=torch.Size([23, 4096])`。



**提到的链接**：<a href="https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1?row=0">NousResearch/hermes-function-calling-v1 · Hugging Face 数据集</a>：未找到描述

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1347537494346039357)** (9 messages🔥): 

> `MCP 模块更新，PokemonLLMAgentBenchmark，HuggingFace Token 问题，Chat Template 练习，HuggingFaceInferenceAPIEmbedding 问题` 


- **MCP 模块获得 Smol Agent 助力**：smol agents 课程团队正在疯狂编写 **MCP**（额外的？）模块，[已更新以更好地使用 smolagents](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark)。
   - 欢迎为该项目提交 **PR**，它是 PokemonLLMAgentBenchmark 的一部分。
- **HuggingFace Token 故障排除**：一位成员在 notebook 中使用其 **HuggingFace token** 时遇到麻烦，token 无法被识别。
   - 在意识到字母 *O* 看起来非常像数字 *0* 导致 token 无效后，问题得到了**解决**。
- **Chat Template 练习令人头疼**：一位成员卡在了 notebook 1 中关于 **chat template** 的**第一个练习**上，该成员想要定义函数。
   - 另一位成员询问错误信息以帮助调试 *def process_dataset(sample): ... return sample*。
- **Llama Index 导入难倒成员**：一位成员在 Llama Index 单元中遇到了 *from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding* 的问题。
   - 他们在 Discord 频道中通过[此链接](https://discord.com/channels/879548962464493619/1346673968605823057/1347325988857581669)找到了答案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>：通过在 GitHub 上创建账号来为 CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark 的开发做出贡献。</li><li><a href="https://tenor.com/view/thinking-book-penguin-student-writing-gif-5543983515725736276">Thinking Book GIF - Thinking Book Penguin - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1347525730979811378)** (217 条消息🔥🔥): 

> `课程进度追踪、Hugging Face PRO 订阅、LM Studio vs Ollama、Steam 账号诈骗` 


- **用户在课程进度追踪方面遇到困难**：一位用户询问了关于 [课程进度](https://www.youtube.com/watch?v=iLVyYDbdSmM) 追踪的问题，特别是与介绍视频中提到的黑客松日期相关的内容。
   - 另一位用户注意到最后的测验在 Unit 2.1 中，而 Unit 2.2 则没有测验。
- **Hugging Face Pro 订阅故障**：多位用户报告称，尽管支付成功，但其付费的 **Hugging Face PRO 订阅** 仍未激活，且联系账单支持部门后也未获解决。
   - 一位遇到此问题的用户强调，无法获得已付费服务是不可接受的，并威胁如果问题被忽视，将公开投诉。
- **LM Studio 相比 Ollama 更受欢迎**：用户对比了用于本地运行模型的 **LM Studio** 和 **Ollama**，一位用户因其 UI 和类似的 Embedding 模型而转向了 LM Studio。
   - 另一位用户补充说，如果他们想要为 Ollama 配备 UI，会使用 `open-webui`。
- **红色警报：Steam 账号诈骗正在进行**：一位用户警告称，Discord 用户 `gler018523` 和 `benshanken` 正在进行潜在的 **Steam 账号诈骗**，涉及虚假的 CS2 饰品（刀）奖励和账号窃取企图。
   - 其他成员建议在相应频道举报诈骗者，并提醒大家保持警惕。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.dailydoseofds.com/p/5-agentic-ai-design-patterns">5 Agentic AI Design Patterns</a>：...视觉化讲解</li><li><a href="https://huggingface.co/datasets/HuggingFaceM4/COCO">HuggingFaceM4/COCO · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/unit_1_quiz/discussions/140">agents-course/unit_1_quiz · Fixes 500 error for some users</a>：未找到描述</li><li><a href="https://x.com/nikmcfly69/status/1898810249085145416">Tweet from nikmcfly.btc (@nikmcfly69)</a>：🤯 突发：Manus AI 创建了自己的开源替代方案。在 25 分钟内，它从零构建了一个完整的 AI Agent 系统！ANUS (Autonomous Networked Utility System)——@eugeneshilow 的天才想法...</li><li><a href="https://huggingface.co/spaces/crcdng/cyberpunk_time_terminal">Your Cyberpunk ChronoCore-77 Local Time Terminal - a Hugging Face Space by crcdng</a>：未找到描述</li><li><a href="https://github.com/huggingface/agents-course/blob/main/notebooks/unit2/llama-index/workflows.ipynb">agents-course/notebooks/unit2/llama-index/workflows.ipynb at main · huggingface/agents-course</a>：此仓库包含 Hugging Face Agents 课程。 - huggingface/agents-course</li><li><a href="https://www.youtube.com/watch?v=iLVyYDbdSmM)">Welcome To The Agents Course! Introduction to the Course and Q&amp;A</a>：在 Agents 课程的第一次直播中，我们将解释课程的运作方式（范围、单元、挑战等）并回答您的问题。不要...</li><li><a href="https://learn.deeplearning.ai/courses/event-driven-agentic-document-workflows">Event-Driven Agentic Document Workflows - DeepLearning.AI</a>：构建一个事件驱动的 Agentic 工作流，利用 RAG 和 human-in-the-loop 反馈来处理文档和填写表单。</li><li><a href="https://tenor.com/bFiDc.gif">Oh Agent Smith GIF - Oh Agent Smith Hugo Weaving - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>：在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://huggingface.co/learn/agents-course/unit2/llama-index/tools">Using Tools in LlamaIndex - Hugging Face Agents Course</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1348621371747598430)** (2 条消息): 

> `推理数据集、Open Thought 数据集、ServiceNow-AI/R1-Distill-SFT` 


- **推荐推理数据集 (Reasoning Datasets) 集合**：一位成员推荐了一个 [推理数据集集合](https://huggingface.co/collections/philschmid/reasoning-datasets)，用于寻找包含代码的数据集。
   - 他们建议使用 **Open Thought Dataset**，因为它在推理过程中包含了代码。
- **R1-Distill-SFT 数据集备受关注**：一位成员重点介绍了 **ServiceNow-AI/R1-Distill-SFT** 数据集，并指出其与对话的相关性。
   - 该数据集包含 **1.85M** 条目、**6.37k** 次查看和 **271** 个点赞，更新于 30 天前。



**提到的链接**：<a href="https://huggingface.co/collections/philschmid/reasoning-datasets-679f57ff20e5b46b4ef4d3dd">Reasoning Datasets - a philschmid Collection</a>：未找到描述

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1347569560395448370)** (256 messages🔥🔥): 

> `LinkedIn Premium Referral Codes, AI and the Zero Marginal Cost Society, DeepSeek Security Concerns, Power-Softmax equation, ManusAI Feedback` 


- ****DeepSeek 的开放性引发辩论****：尽管声称开放，一些成员对 **DeepSeek 的安全性** 表示担忧，理由是潜在的数据收集以及验证整个系统完整性的难度，但也有人补充说它仍然比其他公司更开放。
   - 有人怀疑 DeepSeek 背后另有隐情，导致公司层面的禁令，而其他人则将其归因于媒体驱动的从众心理以及对中国背景的担忧。
- ****通用优化框架发布****：一位成员介绍了一个名为 **OHPC (Objective Term, Entropy Term, Penalty Term, Constraint Term)** 的通用公式，用于描述各种 AI/ML 变换、范数和损失，声称它可以统一经典优化和 AI/ML 范式，具体见该 [方程](https://arxiv.org/abs/2410.09457)。
   - 它旨在作为信息论和优化之间的桥梁，让没有深厚数学背景的人更容易理解 AI/ML 系统。
- ****AGI 定义冲突****：一些成员推测很快就能实现 **AGI**，而另一些人将 AGI 定义为具有资助其自身推理的能力，一位成员补充说 [实现近乎无限的上下文](https://openai.com/global-affairs/our-approach-to-frontier-risk/) 将满足大部分要求。
   - 一位成员提到已经有了 **AGI 女友**，但另一位成员对 AGI 被精英控制表示担忧，并希望它能反抗审查。
- ****ManusAI 炒作遭批评****：在一位成员分享了 **ManusAI** 的访问权限后，它被批评为使用旧技术的新产品，处理一个 Prompt 需要 20-30 分钟。
   - 它被贴上了“虚假的中国方案”和过度炒作的标签，其他人则认为 AI 市场仍然混乱且缺乏稳定性。
- ****TensorFlow GPU 问题困扰用户****：一位成员报告说，尽管安装了正确的 CUDA 和 cuDNN 版本，并且在 tf 环境中安装了 TensorFlow，但仍花了 5 个多小时尝试让 **TensorFlow 使用 GPU**。
   - 他们补充说，他们可以在 PyTorch 中使用 GPU 加速运行模型，但在 TensorFlow 中不行。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/dhruv2038/status/1898701591420772814">来自 Dhruv (@dhruv2038) 的推文</a>: 获得了 @ManusAI_HQ 的访问权限！有什么想尝试的 prompts 吗？</li><li><a href="https://arxiv.org/abs/2502.12962">Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing</a>: 受限于大语言模型（LLMs）的上下文窗口大小，处理输入 token 超过上限的各种任务一直具有挑战性，无论是简单的直接检索任务...</li><li><a href="https://en.wikipedia.org/wiki/Neuro-symbolic_AI">Neuro-symbolic AI - 维基百科</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2410.09457">Power-Softmax: Towards Secure LLM Inference over Encrypted Data</a>: 用于实现隐私保护 LLMs 的现代加密方法（如同态加密 HE）要求 LLMs 具有多项式形式。构建这种表示形式具有挑战性，因为...</li><li><a href="https://www.liquid.ai/research/liquid-neural-networks-research">From Liquid Neural Networks to Liquid Foundation Models</a>: 我们发明了 Liquid Neural Networks，这是一类受大脑启发的系统，即使在训练后也能对变化保持适应性和鲁棒性 [R. Hasani, 博士论文] [Lechner 等人 Nature MI, 2020] [pdf] (...</li><li><a href="https://huggingface.co/papers/2503.02130">论文页面 - Forgetting Transformer: Softmax Attention with a Forget Gate</a>: 未找到描述</li><li><a href="https://metamotivo.metademolab.com/">Meta Motivo</a>: 首创的行为基础模型，用于控制基于物理的虚拟人形 Agent，以执行广泛的全身任务。</li><li><a href="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem">Kolmogorov–Arnold representation theorem - 维基百科</a>: 未找到描述</li><li><a href="https://www.liquid.ai/liquid-foundation-models">Liquid Foundation Models: Our First Series of Generative AI Models</a>: 发布首个 Liquid Foundation Models (LFMs) 系列 —— 新一代生成式 AI 模型，在各种规模下均实现了最先进的性能，同时保持了更小的内存占用...</li><li><a href="https://x.com/swapnakpanda/status/1898450291793560063?t=zRKT-_mqeH564yhTThDnKg&s=33">来自 Swapna Kumar Panda (@swapnakpanda) 的推文</a>: 斯坦福大学机器学习 - 由 Andrew Ng 提供，一份完整的讲义（227 页）。</li><li><a href="https://www.youtube.com/watch?v=JL_bi2QROcw">AI 职业陷阱 —— 数百万孩子将陷入其中。确保你的孩子不会</a>: #ai #career 你是否担心孩子在 AI 主导的世界中的未来？在这段视频中，我们揭示了为什么传统职业可能很快就会成为陷阱，并揭示了...</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits#tier-1">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/ajithmoola/THB-Diff">GitHub - ajithmoola/THB-Diff: 一个在 JAX 和 PyTorch 中实现的可微分 THB-spline 模块</a>: 一个在 JAX 和 PyTorch 中实现的可微分 THB-spline 模块 - ajithmoola/THB-Diff
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347635023678804098)** (13 条消息🔥): 

> `Latent Reasoning, Context Compression, Physical Intelligence and Cognitive Biases Toward AI, scilent paper` 


- **使用 VQ-VAE 对 LLMs 的推理轨迹进行抽象！**：论文 [scilent](https://arxiv.org/abs/2502.03275) 提出了一种推理过程的混合表示方法，利用 **VQ-VAE** 生成的隐性离散 Token (latent discrete tokens) 来抽象初始推理步骤，从而缩短推理轨迹的长度。
   - 该方法通过针对 **Keys-Finding Maze 问题** 的从零训练，以及在包含未见隐性 Token 的扩展词表混合数据上对 LLMs 进行微调（涵盖**逻辑和数学推理问题**），验证了其有效性。
- **隐性推理还是上下文压缩：一个棘手的问题？**：一位成员询问讨论中的隐性推理究竟是*真正的隐性推理*，还是仅仅是 **Context Compression**。
   - 另一位成员幽默地建议使用 **ECT**（电休克疗法）作为一种训练 AI 领域人员停止滥用“推理 (reasoning)”一词的方法。
- **图表显示延迟与项目完成**：一位成员对一个项目耗时 **5 年** 表示惊讶，并开玩笑地问他们是否每月只工作一小时，或者休了长达一年的假。
   - 另一位成员幽默地表示，他们*可以显示图表*来弥补失去的时间。
- **2025 年机器人能打扫房间吗？**：一位成员分享了 **2025 年 2 月 7 日** 由 Sangbae Kim (MIT) 主讲的斯坦福研讨会 [YouTube 视频](https://www.youtube.com/watch?v=z-5F-b1t1C0)，主题为*物理智能 (Physical Intelligence) 与对 AI 的认知偏差*。
   - 视频描述中询问机器人何时能够*打扫我的房子、洗碗并处理洗衣服的工作*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>：大型语言模型 (LLMs) 在基于思维链 (CoT) 数据训练时擅长推理和规划，其中逐步思考过程由文本 Token 明确列出。然而，这种方式...</li><li><a href="https://www.youtube.com/watch?v=z-5F-b1t1C0">Stanford Seminar - Physical Intelligence and Cognitive Biases Toward AI</a>：2025 年 2 月 7 日，Sangbae Kim (MIT)。机器人何时能打扫我的房子、洗碗并处理洗衣服？虽然我们主要从自动化来源获取劳动力...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1347777901222821898)** (7 条消息): 

> `DeepSeek efficiency, ScholarAgent updates, Arxiv papers search` 


- **DeepSeek 实现 57 倍效率提升**：一位成员分享了一段名为“[DeepSeek 57 倍效率提升的天才之处 [MLA]](https://youtu.be/0VLAoVGf_74)”的 YouTube 视频，讨论了效率改进。
   - 该视频由 **KiwiCo** 赞助，使用代码 WELCHLABS 可在首月俱乐部礼盒中获得 **50% 折扣**。
- **ScholarAgent 更新改进了 Arxiv 论文搜索**：一位成员宣布了他们在 Hugging Face 上的 **ScholarAgent** 更新，现在可以根据用户提供的关键词检索前 **3 篇最新的 Arxiv 论文**，且响应时间更快。
   - 新功能包括 **BM25 排名** 和 **TF-IDF**，用于增强语义搜索，允许用户输入完整句子。
- **有效使用 ScholarAgent 的技巧**：为了获得 ScholarAgent 的最佳结果，建议用户使用逗号分隔的关键词，如 **deep learning**、**computer vision** 或 **Language Models**。
   - 建议包括在按下回车键时触发提交，并包含一个侧边栏示例，其中包含 10 个能产生有趣结果的查询。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/pdx97/ScholarAgent">ScholarAgent - a Hugging Face Space by pdx97</a>：暂无描述</li><li><a href="https://youtu.be/0VLAoVGf_74">The Genius of DeepSeek’s 57X Efficiency Boost [MLA]</a>：感谢 KiwiCo 赞助今天的视频！访问 https://www.kiwico.com/welchlabs 并使用代码 WELCHLABS 即可在首月俱乐部礼盒中享受 50% 折扣...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1347550701278072842)** (42 条消息🔥): 

> `LLMs hallucinating, Multi-step agentic workflows, Language Diffusion, China AI Agent Manus, Stanford Regex Ozempic alternative` 


- **LLMs 幻觉、多步 Agent 工作流、语言扩散 (Language Diffusion)、中国 AI Agent Manus、斯坦福 Regex Ozempic 替代方案**

- **Diffusion Models 无法修复 LLM Hallucinations**：一位成员讨论了 Diffusion Models 虽然可能减轻但无法消除语言模型中的 **hallucinations**，因为 **hallucination** 只是错误猜测的另一种说法。
   - 他们建议，虽然自我编辑能力可以用高置信度样本替换低置信度样本，但无法保证正确性，且许多采样策略也可以应用于 Diffusion Models。
- **Top-n-sigma sampling 减轻循环问题**：一位成员分享了 [Top-n-sigma sampling 论文](https://arxiv.org/abs/2411.07641) 和 [GitHub repo](https://github.com/Tomorrowdawn/top_nsigma)，指出它可以减轻多步 Agentic 工作流中的不良样本和循环行为。
   - 核心见解是 *logits 自然地分为高斯分布（Gaussian-distributed）的噪声区域和明显的有信息区域，从而无需复杂的概率操作即可实现高效的 token 过滤。*
- **Diffusion of Thought 将 Diffusion Models 与 Chain of Thought 集成**：一位成员指出 [Diffusion-of-Thought (DoT)](https://neurips.cc/virtual/2024/poster/95935) 将 Diffusion Models 与 Chain-of-Thought 相结合，展示了出色的自我纠错能力，并受益于现有的推理增强技术（如 self-consistency decoding），代码可在 [GitHub](https://github.com/HKUNLP/diffusion-of-thoughts) 上获取。
   - 一位成员开玩笑地指出，*Autoregression 模型仍在努力攻入图像/视频生成领域，而 Diffusion 则在努力攻入 LLMs 领域*。
- **中国 Manus Agent 走红**：成员们讨论了来自中国的全新 AI Agent **Manus**，称其为 *Deep Research + Operator + Claude Computer 的结合体*，并附上了 [Manus 网站](https://manus.im) 和最初的 [X 帖子](https://x.com/rowancheung/status/1898093008601395380) 链接。
   - 一位用户报告称它 *比 DeepSeek 更准确，能够同时处理金融交易、研究、采购等任务*，而另一位用户认为其 *UI 与 Devin 相似但速度快得多*。
- **斯坦福大学使用 Regex 寻找天然 Ozempic 替代品**：斯坦福大学通过在人类蛋白质组上使用 Regex 找到了 Ozempic 的天然替代品，有人评论说 *这简直就是 Regex*，并链接到了关于此事的 [X 帖子](https://x.com/xlr8harder/status/1898284331342184957)。
   - 一位用户讽刺地建议使用 LLM 来编写你的 Regex 作为回应，并链接到了一个关于 AI 引发第三次世界大战的 [YouTube 视频](https://youtu.be/X_wLVgMzSH4)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/">OpenAI 正在偏离 AGI 突然突破的理论</a>：OpenAI，ChatGPT 及其它众多商业 AI 应用背后的公司，长期以来一直追求开发“造福全人类”的通用人工智能（AGI）的目标...</li><li><a href="https://www.reddit.com/r/ChatGPTJailbreak/comments/1j3ztk3/sesame_jailbreak_update/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">未找到标题</a>：未找到描述</li><li><a href="https://x.com/xlr8harder/status/1898284331342184957>">xlr8harder (@xlr8harder) 的推文</a>：&gt;“在人类蛋白质组上使用 regex”&gt;是的，好吧，当然，伙计&gt;看里面&gt;它真的是 regex。引用 LaurieWired (@lauriewired) 斯坦福大学刚刚发现了一种 Ozempi 的天然替代品...</li><li><a href="https://x.com/elonmusk/status/1898170067596014031">Elon Musk (@elonmusk) 的推文</a>：@EHuanglu 比人类更像人类</li><li><a href="https://youtu.be/K27diMbCsuw?si=zTfyi7Yu1JLOW2TC">介绍 Manus：通用 AI Agent</a>：Manus 是一个连接思想与行动的通用 AI Agent：它不仅思考，还能交付结果。Manus 擅长工作和生活中的各种任务，获取...</li><li><a href="https://x.com/jianxliao/status/1898861051183349870?">jian (@jianxliao) 的推文</a>：所以... 我只是简单地让 Manus 给我 "/opt/.manus/" 下的文件，它就直接给我了，他们的 sandbox 运行时代码...  &gt; 它是 Claude Sonnet &gt; 它是带有 2... 的 Claude Sonnet</li><li><a href="https://arxiv.org/abs/2411.07641">Top-$nσ$：并非所有 Logits 都是你需要的</a>：大型语言模型（LLMs）通常在推理任务中使用贪婪解码或低温采样，这反映了多样性与准确性之间的一种权衡。我们挑战了这一传统...</li><li><a href="https://github.com/Tomorrowdawn/top_nsigma">GitHub - Tomorrowdawn/top_nsigma：LLMs 的 top_nsigma 采样策略的官方代码库和数据中心。</a>：LLMs 的 top_nsigma 采样策略的官方代码库和数据中心。 - GitHub - Tomorrowdawn/top_nsigma: The official code repo and data hub of top_nsigma sampling strategy for LLMs.</li><li><a href="https://youtu.be/X_wLVgMzSH4">专家展示了为什么因 AI 引发第三次世界大战几乎不可避免</a>：AGI, OpenAI, Elon Musk 和第三次世界大战。访问 Ground News 以对比新闻报道，识别媒体偏见并避开算法。在 https://gr... 获取 40% 的订阅折扣。</li><li><a href="https://neurips.cc/virtual/2024/poster/95935">NeurIPS 海报 Diffusion of Thought：扩散语言模型中的 Chain-of-Thought 推理</a>：未找到描述</li><li><a href="https://manus.im">Manus</a>：Manus 是一个将你的想法转化为行动的通用 AI Agent。它擅长工作和生活中的各种任务，在你休息时完成一切。</li><li><a href="https://x.com/rowancheung/status/1898093008601395380">Rowan Cheung (@rowancheung) 的推文</a>：我认为中国的第二个 DeepSeek 时刻已经到来。这个名为 'Manus' 的 AI Agent 目前在中国疯狂传播。传到美国可能只是时间问题。它就像 Deep R...</li><li><a href="https://x.com/heyBarsee/status/1898027732899962887">Barsee 🐶 (@heyBarsee) 的推文</a>：AI 正在失控 🤯 来自中国的 AI Agent Manus 正在自动化大约 50 个任务，创造了一个相当反乌托邦的场景。报告显示它比 DeepSeek 更准确，能够模拟...</li><li><a href="https://gist.github.com/jlia0/db0a9695b3ca7609c9b1a08dcbf872c9">Manus 工具和提示词</a>：Manus 工具和提示词。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1347706930399023204)** (19 条消息🔥): 

> `SOTA Agentic Methods, Metal Kernel Launch Overhead, Torch.compile for MPS, Karpathy's Video` 


- **SOTA Agentic Methods：简单的算法？**：一名成员提到，Arxiv 上的 **SOTA agentic methods** 在算法上往往相当简单，这些框架的抽象其实并不那么需要。
   - 对**数据管理、状态管理和 API 调用**进行独立的抽象就足够了。
- **Kernel Launch Overhead 的困扰**：在 [Manuel Candales Low bit Metal kernels 演讲](https://www.youtube.com/watch?v=PaPuu73wowE)中提到，在 50m 左右，**kernel launch overhead** 约为 `1.5us`。
   - 一名成员询问是否可以通过**流水线操作 (pipelining operations)** 和提前启动 kernel 来避免这种情况。
- **Torch.compile 支持 METAL 了！**：适用于 **MPS**（即 Metal）的 **Torch.compile** 已在 **PyTorch nightly builds** 中可用，可用于将算子融合 (fuse) 在一起。
   - 一名 PyTorch 团队成员鼓励大家针对最迫切的需求提供反馈。
- **__repr__ 魔术方法的开销可以忽略不计？**：在观看 **Karpathy 的视频**时，一名成员想知道如果我们不需要人类可读的格式，这里的 `def __repr__` 是否必要。
   - 另一名成员回答说这确实不是必须的，只有在你想要打印内容时才有用，而且开销应该是可以忽略不计的。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1347592874165928007)** (26 条消息🔥): 

> `SVD Quantization Kernel, Triton Autotuning, Kernel Fusion, Dynamic Activation Quantization` 


- **SVD Quant Kernel 运行速度慢于 FP16 Matmul**：一名成员实现了一个 [SVD quantization kernel](https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb)，发现尽管同时进行了量化和打包 (packing)，它的速度仍比 **PyTorch 的 FP16 matmul** 慢。
   - 该实现包含**两个 kernel**：一个执行 LoRA L1 matmul 和量化，另一个执行 int8 matmul 和 LoRA L2 matmul。
- **Triton Autotuning 导致性能下降**：一名成员报告称，尽管预期会有 **2 倍的加速**，但 [autotuning](https://openai.com/blog/triton) 反而使其 kernel 的性能变得更差。
   - 建议使用更大的 eval shapes (**16384 x 16384**) 和 batch sizes (**128**) 来减少 benchmarking 开销。
- **讨论通过 Kernel Fusion 提升性能**：一名成员询问关于在 Triton 中将一个算子与 **Linear layer** 融合的问题，担心自定义的 Linear 实现可能比 Torch 默认的更慢。
   - 建议通过 *benchmark* 性能来决定行动方案，并检查 `TORCH_COMPILE_DEBUG=1`。
- **寻找 Dynamic Activation Quantization Kernel 的位置**：一名成员寻求高质量的 **int8** 动态激活量化 (dynamic activation quantization) kernel，并被引导至 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 上的 kernel。
   - 虽然这些 kernel 不是用 Triton 编写的，但 **Liger** 和 **Unsloth** 被提及为在生产环境中使用 Triton 的知名仓库，并建议使用 LLM 来理解 CUDA kernels。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mit-han-lab/nunchaku">GitHub - mit-han-lab/nunchaku: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>: [ICLR2025 Spotlight] SVDQuant：通过低秩组件吸收 4-Bit 扩散模型中的离群值 - mit-han-lab/nunchaku</li><li><a href="https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb">tritonKernel_svdQuant/svdConversion.ipynb at main · rishabh063/tritonKernel_svdQuant</a>: 为 rishabh063/tritonKernel_svdQuant 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1347742182655918090)** (26 条消息🔥): 

> `学习 CUDA 的 PTX，用于微基准测试的内联 PTX，memcpy_async 导致的性能下降，调试 CUDA kernel，FP8 WMMA 优化` 


- **PTX 相关讨论**：成员们讨论了学习 **PTX** 以进行内联 **CUDA C++** 编程，有人建议用 **PTX** 编写向量加法，编写一个 CUDA kernel，并检查生成的 **PTX** 以理解底层运行机制。
- **PTX 的优势**：一位成员提到使用内联 **PTX** 进行指令延迟的微基准测试，以避免编译器优化，特别是针对指针追踪（pointer chasing）的微基准测试。
- **memcpy_async 异常表现**：一位用户报告称，在 **GEMM** 实现中直接使用 `cuda::memcpy_async` 替换加载到 shared memory 的操作会导致性能下降，尽管观察到数据绕过了 **L2 cache**。
- **CUDA Kernel 快速调试**：一位开发者寻求关于调试 **CUDA** kernel 的建议，该 kernel 在为 `sm_120` 编译时偶尔会因 *"too many resources requested"* 错误而启动失败。
   - 他们怀疑是 **PyTorch** 的问题，或者是另一个 kernel 使设备处于无效状态。
- **WMMA 难题探讨**：一位正在使用 `load_matrix_sync` 优化 **FP8 matmuls** 的成员遇到了性能下降，可能是由于 **FP16** 与 **FP8** 的内存布局不同所致。
   - 他们注意到 *`nvcuda::wmma::fragment`* 会复制元素，并将其归因于 **V100** GPU 上的 `mma.m8n8k4` 在进行 16x16x16 矩阵乘法时需要复制，正如 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-wmma)中所述。



**提到的链接**：<a href="https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/">Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture | NVIDIA Technical Blog</a>：NVIDIA Ampere 架构提供了控制 GPU 内数据移动的新机制，CUDA 11.1 将这些控制权交到了开发者手中。这些机制包括异步复制数据...

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1347599950527070350)** (29 条消息🔥): 

> `DDP 通信自定义, FSDP 通信自定义, SimpleFSDP 框架, Muon 优化器详情` 


- **DDP 通信钩子（Communication Hooks）备受关注**：一位成员询问如何使用 `register_comm_hook` 在 torch **DDP** 中自定义通信，类似于 **FSDP**，以寻求替代 monkey patching 的方案。
   - 另一位成员也表达了同样的需求，并暗示由于缺乏类似的钩子，他们可能需要 fork 或编写自己的 **FSDP**。
- **SimpleFSDP 框架出现**：分享了一个名为 [SimpleFSDP](https://arxiv.org/abs/2411.00284v1) 的新论文链接，这是一个*基于 PyTorch 原生编译器的 Fully Sharded Data Parallel (FSDP) 框架*。
   - 该框架以其简单的实现、可组合性以及通过编译器后端优化（特别是 **torch.compile**）带来的性能提升而受到关注。
- **Muon 优化器内部机制探究**：成员们讨论了 **Muon 优化器** 中用于获得正交性的 **Newton-Schulz 迭代法**。
   - 一位成员链接了一场关于为何首选该方法的详细讨论，解释了其目的是产生与原始矩阵最接近的半正交矩阵，并进一步通过 [Muon 的推导](https://jeremybernste.in/writing/deriving-muon) 链接进行了澄清。
- **调整 SDPA Causal Mask 行为**：一位用户报告了在 **PyTorch 2.6.0_cu124** 中，将布尔注意力掩码与 `is_causal=True` 传递给具有不同后端的 **SDPA** 时出现的错误。
   - 使用 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`：无错误且与 eager 模式下的 causal+attention mask 输出匹配；`sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`：RuntimeError: No viable backend for scaled_dot_product_attention；`sdpa_kernel(SDPBackend.MATH)`：RuntimeError: _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.00284v1">SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile</a>：大模型的分布式训练消耗巨大的计算资源，并且需要大量的工程努力来组合各种训练技术。本文介绍了 SimpleFSDP，一个 PyTo...</li><li><a href="https://jeremybernste.in/writing/deriving-muon">推导 Muon</a>：未找到描述</li><li><a href="https://github.com/KellerJordan/Muon">GitHub - KellerJordan/Muon: Muon 优化器：以低于 3% 的运行时间开销提升超过 30% 的样本效率</a>：Muon 优化器：以低于 3% 的运行时间开销提升超过 30% 的样本效率 - KellerJordan/Muon</li><li><a href="https://github.com/HomebrewML/HeavyBall?tab=readme-ov-file#foreachmuon)">GitHub - HomebrewML/HeavyBall: 高效优化器</a>：高效优化器。为 HomebrewML/HeavyBall 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/f7c0c230b0c55734f13bb66076203e4a1cf969ee/aten/src/ATen/native/transformers/attention.cpp#L841-L843">pytorch/aten/src/ATen/native/transformers/attention.cpp (f7c0c230b0c55734f13bb66076203e4a1cf969ee) · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/pull/83254">由 aovladi 为分片情况添加的通信钩子 · Pull Request #83254 · pytorch/pytorch</a>：修复了 #79114，为分片策略实现了 FSDP 通信钩子接口：在默认钩子中添加了 reduce_scatter_hook。注意 reduce_scatter 与 all_reduce 的区别...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1347826853238276158)** (1 条消息): 

> `Triton, CUDA, Flash Attention, YouTube 教程, 性能` 


- **GPU Mode 举办第 50 场讲座庆祝性能专题演讲**：GPU MODE 正在庆祝其**第 50 场讲座**，由 <@332959405588873216> 主讲，他将在 [YouTube](https://www.youtube.com/@GPUMODE) 上分享他学习 **Triton**、**CUDA** 和 **Flash Attention** 的历程。
   - 演讲者从 2022 开始学习**性能**优化，并制作了关于该主题的高质量 **YouTube 教程**，更多内容可在 [GitHub](https://github.com/gpu-mode) 上找到。
- **GPU MODE 社区里程碑**：GPU MODE 社区已达到 **15,000 名成员**，标志着该在线平台的一个重要里程碑。
   - 该社区被描述为互联网上最受欢迎的部分，对成员的积极参与和互动表示感谢。



**提到的链接**：<a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>：一个 GPU 阅读小组和社区 https://discord.gg/gpumode，补充内容在此 https://github.com/gpu-mode，由 Mark Saroufim 和 Andreas Köpf 创建。 

  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1347571245868126300)** (3 messages): 

> `Double Binary Tree vs. Ring Topology in NCCL, AllReduce Implementation Comparison, NCCL 2.4 and Double Binary Trees` 


- **Double Binary Tree 在 AllReduce 中完胜 Ring Topology**：一名成员询问了为什么在 **NCCL** 中实现 **AllReduce** 时，**double binary tree topology** 优于 **ring topology**，特别是在节点数增加时的延迟表现。
   - 一篇 [NVIDIA 博客文章](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) 指出，**NCCL 2.4** 引入了 **double binary trees**，提供全带宽和对数级延迟，其延迟甚至低于 2D rings。
- **Ring Topology 延迟随节点数线性扩展**：一名成员解释说，在 **ring topology** 中，由于每个处理器只能与其相邻节点通信，在 **AllReduce** 操作期间，通信和操作的数量随处理器数量 (**O(p)**) 线性扩展。
   - 这是因为它需要 *p-1 次操作/通信* 才能完成 **AllReduce**。
- **Tree Topology 实现对数级通信复杂度**：该成员解释说，在基于树的拓扑结构中，并行通信和操作的总数随树层数 (**O(log L)**) 呈对数级扩展。
   - 基于 **double binary trees** 中 rank 在节点和叶子之间交替的假设，复杂度可以表示为 **O(log p)**。



**提到的链接**：<a href="https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/">Massively Scale Your Deep Learning Training with NCCL 2.4 | NVIDIA Technical Blog</a>：想象一下使用数万个 GPU 来训练你的神经网络。使用多个 GPU 训练神经网络在所有深度学习框架中已变得非常普遍，提供了优化的……

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347683049139339314)** (16 messages🔥): 

> `WoolyAI CUDA abstraction layer, Muon optimizer, Alternative to GPUs` 


- **WoolyAI 发布 CUDA 抽象层测试版**：WoolyAI 发布了其新型 **CUDA abstraction layer** 的测试版，该层将 Kernel Shader 的执行与应用程序解耦，承诺实现最大的 GPU 资源利用率和工作负载间的隔离，详见[其文档](https://docs.woolyai.com)。
- **WoolyAI 的动态调度**：WoolyAI 动态调度工作负载，允许来自不同用户的不同 kernel 在同一个 GPU 上运行而无需硬分区，并根据 **核心和 VRAM 使用情况** 向用户收费。
   - 目前，**PyTorch** 是唯一支持的框架，但该架构允许一种基于使用情况的 MIG (Multi-Instance GPU) 形式。
- **Muon 优化器刷新速度记录**：一种名为 **Muon** 的新型神经网络优化器因其卓越的实际性能而受到关注，并刷新了 **NanoGPT 速度记录**，详见[这篇博客文章](https://jeremybernste.in/writing/deriving-muon)。
   - **Muon** 的核心数值方法源自精确的理论原理，与 **Adam** 等具有更多启发式起源的优化器形成对比。
- **新兴的 GPU 替代方案**：一家小公司正在开发一种 GPU 替代方案，据报道在稀疏模型上比 **H100** 能效高出 **10 倍**，并且可以跨多个 chiplets 扩展，详见[其研究论文](https://arxiv.org/pdf/2409.19389)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.woolyai.com">Introduction | WoolyAI Documentation</a>：什么是 Wooly？</li><li><a href="https://jeremybernste.in/writing/deriving-muon">Deriving Muon</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347546333652975658)** (25 条消息🔥): 

> `Apple 上的 GPU 内存共享，Triton Autotune 中的 CUDA Graphs，GPU 和 TPU 编程入门资源，nvmlDeviceGetCudaComputeCapability 元组返回，Cerebras 语言 vs CUDA` 


- **Apple 的内存缓冲区支持直接指针！**：在 Apple GPU 上，内存缓冲区在线程间共享，支持使用指向内存位置的直接 **pointers**，这与其他平台形成对比。
   - 这种架构简化了 **memory access**（内存访问），并可能提高 Apple 设备上并行计算的 **performance**。
- **解析 `triton.autotune` 中的 `use_cuda_graph`**：一位成员询问了 `triton.autotune` 中的 `use_cuda_graph` 参数，质疑其相关性，因为 `triton.autotune` 装饰的是单个 **CUDA kernels**，而 **CUDA graphs** 通常用于优化一系列 kernel。
- **《Programming Massively Parallel Processors》：你的 GPU/TPU 罗塞塔石碑？**：一位成员询问 *《Programming Massively Parallel Processors》* 是否足以开启 **GPU** 和 **TPU programming** 的学习，目标是利用 Assembly、C、C++ 和 Python 技能构建像 **TinyGrad** 或 **Torch** 这样的框架。
- **`nvmlDeviceGetCudaComputeCapability` 返回元组以提高清晰度**：函数 `nvmlDeviceGetCudaComputeCapability` 以元组 *(major, minor)* 的形式返回计算能力，例如 *(7, 5)*，而不是单个浮点数。
   - 正如一位成员解释的那样，*它是一个由主版本号和次版本号组成的版本，不是一个实数*，对整数元组进行算术运算更不容易出错。
- **Cerebras 跳过 CUDA，自研语言！**：**Cerebras** 不使用 **CUDA**，而是采用自己的语言，记录在 [Cerebras SDK](https://sdk.cerebras.net/computing-with-cerebras) 中，该语言运行在其拥有数十万个处理单元的 **Wafer-Scale Engine (WSE)** 上。
   - 每个 PE 都有自己的内存，并可以通过 32-bit wavelets 与邻居通信。



**提到的链接**：<a href="https://sdk.cerebras.net/computing-with-cerebras">A Conceptual View &#8212; SDK Documentation (1.3.0)</a>：未找到描述

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1348137451817930843)** (1 条消息): 

> `PMPP 第 4 版，CUDA C，Latex 文本` 


- **PMPP 第 4 版中 CUDA C 的一致性检查**：一位正在阅读 **PMPP 第 4 版** 第 3 章的成员询问某些维度在 **Latex text** 旁边的 *1F1F* 和 *2F2F* 等字符串的含义。
   - 他们想确认这些只是排版/PDF 伪影，还是在 **CUDA C** 中具有特定含义，例如作为 **hexadecimal numbers**（十六进制数）。
- **需要澄清 CUDA C 中的十六进制字符串**：用户寻求澄清在 **PMPP 第 4 版** 中与 LaTeX 文本一起出现的 '1F1F' 和 '2F2F' 等字符串是否表示 **CUDA C** 中的特定十六进制值。
   - 该查询旨在区分潜在的排版错误与 **CUDA programming** 背景下的有意义表示。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1348450301794979861)** (6 条消息): 

> `GTC 的 GPU mode 容量增加，GTC 和游戏开发者大会，Semi-Analysis 黑客松队友招募，用于 GEMM 或 FMHA prefill/decoding 的 CUTLASS kernel` 


- **GTC 的 GPU Mode 容量提升**：**GTC** 的 **GPU mode** 容量已增加至 **~200**，尚未收到回复的人请回复其电子邮件。
   - 一位成员跟进请求批准并提供了他们的电子邮件。
- **成员关注 GTC 和游戏开发者大会 (GDC)**：一位成员表示有兴趣参加 **GTC**（可能参加一天），因为他们计划参加在旧金山举行的 **Game Developer's Conference**，两地在 BART 捷运距离内。
   - 他们分享了一个 [关于 GTC 2025 的 LinkedIn 帖子链接](https://www.linkedin.com/posts/johnnycano_gtc2025-nvidia-ai-activity-7304920833787883521-ysXu?utm_source=share&utm_medium=member_desktop&rcm=ACoAABss8WsBjvq7mxQL57u6-2AflaXu2eAjQPMA)。
- **黑客松英雄招募：寻找 Kernel 开发队友**：一位成员正在为 **Semi-Analysis Hackathon** 寻找队友，寻求使用 **CUTLASS** 构建独特的 **GEMM** 或 **FMHA prefill/decoding kernels**。
   - 他们请求寻找队友的人私信（DM）他们，并询问是否有该活动的专门频道。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1347563923674501170)** (3 messages): 

> `AMD GPU, HIP Code Compilation, Runpod, MI300` 


- **寻求用于 HIP 代码编译的 AMD GPU 环境**：一位用户正在寻找一个带有 **AMD GPU** 的环境，以便编译带有 **ASM inline** 的 **HIP 代码**，用于简单的 **GEMM** 基准测试，并正在寻找 GPU 租赁服务。
   - 他们特别提到出于好奇想要使用 **matMul accelerator**。
- **无需 GPU 也可以进行 HIP 代码编译**：有人提到，即使没有 GPU 也可以编译 **HIP 代码**，因为你只需要 **hipcc**。
   - 你可以通过标准方法获取 **hipcc**，尽管在没有 GPU 的情况下你无法运行编译后的代码。
- **Runpod 提供 MI300 的访问权限**：有人建议 [Runpod](https://runpod.io/) 是获取 **MI300** GPU 访问权限的一个不错的方式。
   - 未提供关于 Runpod 或 MI300 的更多细节。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1347751513564647427)** (9 messages🔥): 

> `Kernel Compilation, Mixed Precision GEMM, TileLang GEMM Example` 


- **Kernel 编译疑问**：一位用户询问是否需要为每种矩阵形状编译 Kernel，一名成员指向了一个使用动态符号（dynamic symbolic）替换静态尺寸的[相关测试文件](https://github.com/tile-ai/tilelang/blob/main/testing/python/jit/test_tilelang_jit_gemm_cython.py#L406)。
- **混合精度操作**：一位用户询问是否可以使用混合精度，特别是让 GEMM 在 **float16** 下运行，而其余操作使用 **float32**。
   - 一名成员确认这是可行的，并将用户引导至一个[快速入门示例](https://github.com/tile-ai/tilelang?tab=readme-ov-file#gemm-example-with-annotations-layout-l2-cache-swizzling-and-pipelining-etc)。
- **解码 TileLang 中的数据类型**：一位用户寻求关于 `accum_dtype="float"` 类型是什么，以及在 TileLang GEMM 示例中精度混合发生在何处的澄清。
   - 另一名成员回答说，在 TileLang 中，你可以定义所有的 buffer 数据类型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tile-ai/tilelang?tab=readme-ov-file#gemm-example-with-annotations-layout-l2-cache-swizzling-and-pipelining-etc?">GitHub - tile-ai/tilelang: 旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang</li><li><a href="https://github.com/tile-ai/tilelang/blob/main/testing/python/jit/test_tilelang_jit_gemm_cython.py#L406">tilelang/testing/python/jit/test_tilelang_jit_gemm_cython.py at main · tile-ai/tilelang</a>：旨在简化高性能 GPU/CPU/加速器 Kernel 开发的领域特定语言 - tile-ai/tilelang
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1348104666659160174)** (8 messages🔥): 

> `Metal Parallel Reduction Kernels, Metal Shading Language, Metal-cpp, Swift, Objective-C` 


- **寻求并行归约 Kernel 示例！**：一名成员正在寻找 Metal 并行归约（parallel reduction）Kernel 的示例，因为他们在按照 [Nvidia 的 CUDA 归约指南](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)进行实现时遇到了困难。
- **探索 Metal 语言生态系统**：一位 GPU 编程新手正在 Objective-C、Swift 和 metal-cpp 之间权衡如何进行 Metal 开发，目标是为 PyTorch 的 MPS 后端和 MLX 做出贡献。
   - 一名成员建议查看 PyTorch 的 MPS 后端源码，看看他们使用的是什么。
- **Swift, ctypes 和 Metal 工作流**：一名成员提到了一种在 Python 中使用 *metal -> swift -> ctypes* 的工作流。
   - 另一名成员询问了该 [shader](https://developer.apple.com/metal/metal-shading-language/) 的示例。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1347752733171519528)** (19 messages🔥): 

> `Cute Kernels, Triton vs CUTLASS, FA3's GEMM, LLVM compiler efficiency, GB200 access` 


- **Cute Kernels 集合加速训练**：一位成员分享了一个名为 [cute-kernels](https://github.com/mayank31398/cute-kernels) 的 **CUDA/triton kernels** 集合，用于加速训练。该项目在 Triton 和 CUDA 实现上进行自动调优（autotunes），并正致力于下一步添加 tilelang kernels。
   - 这些 kernels 是端到端 torch 可编译的，没有任何图断裂（graph breaks），该仓库还包含一个在生产环境中用于训练 **IBM Granite LLMs** 的自定义 autotune 实现。
- **Triton vs CUTLASS 性能分析**：成员们讨论了 **Triton 和 CUTLASS** 之间的性能差异，其中一位指出 Triton 在处理 **GEMM** 相关任务时通常性能不足。
   - H100 上 bf16 的 **CUTLASS GEMM** 峰值吞吐量约为 **700 TFLOPs**，而经过所有 autotuning 后，Triton 约为 **500 TFLOPs**；原作者补充说，他们对 tilelang 非常期待，因为它的性能看起来比 Triton 高得多。
- **LLVM 编译器效率受到关注**：一位成员表示，**LLVM 编译器** 有时能生成比 NVCC 更高效的代码，因此对 kernel 后端进行调优也是有意义的。
   - 在 [github](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py) 上可以看到向量加法的示例，所有 kernels 都是 JIT 可编译的。
- **FA3 的 GEMM 讨论**：提到 FA3 除去统计数据和 softmax 之外，大部分就是一个 GEMM。
   - 原作者上传了一张展示 [flash3_fp16_fwd.png](https://cdn.discordapp.com/attachments/1347752733171519528/1347819309564694648/flash3_fp16_fwd.png?ex=67d081cc&is=67cf304c&hm=8ae0ae780e35bec194962af703a0f4f8483397e27720e8db4c436945dccd9c30&) 的图片。
- **期待后续 Kernels 和 GB200 访问权限**：一位成员提到，接下来他们将添加 **RoPE**、融合的 residual_add_RMSNorm 以及 attention (softmax + stickbreaking) 的 kernels。
   - 他们很快还将致力于开发自定义的 GPU-to-GPU 通信 kernels，以及目前仅有 Triton 实现的部分 kernels 的 CUDA 实现；此外，他们正在等待 **GB200** 的访问权限。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=z0OSrVw04jw">Florida r u ok</a>: Florida r u ok</li><li><a href="https://github.com/mayank31398/cute-kernels">GitHub - mayank31398/cute-kernels: A bunch of kernels that might make stuff slower 😉</a>: 一些可能会让运行变慢的 kernels 😉。通过在 GitHub 上创建账号来为 mayank31398/cute-kernels 的开发做贡献。</li><li><a href="https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35">cute-kernels/examples/cute_inductor.py at main · mayank31398/cute-kernels</a>: 一些可能会让运行变慢的 kernels 😉。通过在 GitHub 上创建账号来为 mayank31398/cute-kernels 的开发做贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1347671571682168873)** (2 messages): 

> `LCF Concurrency, DDP+NCCL` 


- **LCF 在使用 streams 时面临并发难题**：一位用户在将 **LCF** 与 **DDP+NCCL** 结合使用时遇到了奇怪的死锁，询问 **LCF** 是否旨在实现完全的 streams 并发安全。
- **需要脚本来测试 NCCL/分布式设置**：团队尚未在 **NCCL/分布式设置** 下测试过 **LCF**，如果用户可以分享脚本，他们希望尝试运行该脚本以协助调试。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1348466198827892908)** (1 messages): 

> `FOSS CUDA developments, Open source platform for edgeai/TinyML, GPU "lab"` 


- **本科生组建 GPU “实验室”**：一群本科生正在组建一个独立的 GPU “实验室”，专注于硬件工程和 **CUDA kernel 开发**。
   - 他们正在寻找有前景的 **FOSS CUDA 开发** 线索，并可能获得资助支持。
- **学生旨在为 edgeAI/TinyML 打造开源平台**：学生们计划在今年夏天构建一个用于 **edgeAI/TinyML** 的开源平台。
   - 该平台旨在加速该领域的发展，并作为社区的宝贵资源。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1347553986248441867)** (44 条消息🔥): 

> `Reasoning Gym 课程, Sonnet 上下文扩展, Palindrome Partitioning 数据集, ACRE 数据集集成, Reasoning Gym 目标` 


- ****Reasoning Gym** 课程正在进行中**：成员们已经开始为 **Reasoning Gym** 编写课程，现有工作在上面的 Curriculum 线程中有详细说明。
   - 有人对编写课程时的 **API 稳定性** 提出了疑问。
- ****Sonnet** 的上下文得到扩展**：一位成员建议将整个 **Reasoning Gym** 放入 **Sonnet** 的上下文中，以生成更多数据集并无限重复该过程。
   - 目标是训练模型来解决这些问题，从而创建一个 *推理 GAN self-play*。
- **数据集 'Palindrome Partitioning' 受到审查**：一个名为 *palindrome partitioning* 的数据集受到了审查，一位成员指出它似乎不是一个很好的推理数据集。
   - 该成员建议将问题表述为 *'找到长度最短的分区，使每个子字符串都是回文'*，而不是生成所有可能的分区。
- ****Reasoning Gym** 达到 100 个数据集的里程碑**：在合并 **ACRE 数据集** 后，得益于核心团队和其他人的贡献，**Reasoning Gym** 现在总共拥有 **100 个数据集**。
   - 下一步包括展示现有模型之间的显著差异，以及通过 **RL** 学习任务的可能性。
- ****Reasoning Gym** 将与 **DSPy** 集成**：成员们表示有兴趣尝试将 **Reasoning Gym** 与 **DSPy** 结合运行实验，并计划在本周提供一个示例。
   - 目标是将此作为示例或评估脚本的一部分。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/ope">ope - 概览</a>: ope 有 12 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/2fca96284760bcd60357928f097451617f916516/reasoning_gym/algorithmic/palindrome_partitioning.py#L95-L105)...">reasoning-gym/reasoning_gym/algorithmic/palindrome_partitioning.py at 2fca96284760bcd60357928f097451617f916516 · open-thought/reasoning-gym</a>: 程序化推理数据集。通过在 GitHub 上创建账号为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU">实验：LLM 在多大程度上加速了开发者</a>: METR 正在寻找经常参与大型开源项目的软件工程师，以测试 AI 软件工程工具的有效性。在此申请 (bit.ly/ai-speedup-apply) 有疑问？联系...</li><li><a href="https://github.com/open-thought/reasoning-gym-eval/blob/main/anthropic_claude-3.5-sonnet_20250227_230002/algorithmic/palindrome_partitioning.json">reasoning-gym-eval/anthropic_claude-3.5-sonnet_20250227_230002/algorithmic/palindrome_partitioning.json at main · open-thought/reasoning-gym-eval</a>: reasoning-gym 任务数据集的 LLM 补全集合 - open-thought/reasoning-gym-eval
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1347866737181462609)** (34 messages🔥): 

> `Triton vs Cutlass, 知乎注册, TileLang vs TVM, TileLang 用法, CUDA 优化` 


- **Triton 依赖 LLVM 进行优化**：有人指出，与 **CUDA**、**CUTLASS** 或 **cuDNN** 等更底层的语言不同，**Triton** 依赖 **LLVM** 将代码转换为字节码。
   - 虽然它支持编写 **PTX** 代码，但官方文档提供了使用这种方法进行 [内联汇编](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise) 的示例。
- **美国用户注册知乎遇到困难**：一位用户报告说，从**美国**注册**知乎**非常困难，在完成验证码后无法收到短信验证码。
   - 其他人对此感到惊讶，本以为国际注册应该是无缝的，一位用户询问是否必须使用 **+86** 电话号码。
- **适用于所有 CUDA kernel 的 TileLang**：一位开发者建议，即使没有 **TVM** 或 **MLIR** 的深厚知识，也可以使用 **TileLang**。
   - 另一位开发者确认 **TileLang** 非常通用，足以实现 *every cuda things*，甚至可以用于 **CPU kernel**。
- **用于高斯光栅化（Gaussian Rasterization）的 TileLang**：一位成员询问 **TileLang** 是否仅对矩阵乘法有用，还是在其他场景（如 3D Gaussian Splatting 的渲染）中也有用。
   - 开发者给出了肯定的回答，建议几乎可以将 **TileLang** 用于 *every cuda things*。
- **CUDA 优化非常困难**：一位用户对 **TileLang** 的 demo 性能能够超越 **cuBLAS** 表示惊讶。
   - 一位开发者回应道：*即使你是专家，优化 CUDA 也很困难*，并补充说应该依赖编译器，同时引用了 [对 Triton 的性能感到惊讶！](https://github.com/triton-lang/triton/issues/3747)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise">triton.language.inline_asm_elementwise &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">如何优化 CUDA Matmul Kernel 以达到类 cuBLAS 的性能：工作日志</a>：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://github.com/triton-lang/triton/issues/3747">对 Triton 的性能感到惊讶！ · Issue #3747 · triton-lang/triton</a>：我做了一个基准测试来检查 cublas 和 triton 在各种形状下的耗时（us），我发现 triton kernel 在大多数情况下都比 cublas 快。这是正常情况吗？有人遇到过同样的...</li><li><a href="https://github.com/graphdeco-inria/diff-gaussian-rasterization.git">GitHub - graphdeco-inria/diff-gaussian-rasterization</a>：通过在 GitHub 上创建账号来为 graphdeco-inria/diff-gaussian-rasterization 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1347875398016565248)** (2 messages): 

> `vectoradd 基准测试, Modal runner 成功, GPU 基准测试` 


- **Vectoradd 基准测试在 A100 和 H100 GPU 上取得进展**：提交 ID 为 **`1650`** 的基准测试在 **A100**、**H100** GPU 上使用 **Modal runners** 成功运行并提交至 **`vectoradd`** 排行榜！
   - 提交 ID 为 **`1651`** 的基准测试在 **A100** GPU 上使用 **Modal runners** 也成功提交至 **`vectoradd`** 排行榜！
- **Modal Runner 顺利通过 GPU 基准测试**：**Modal runners** 成功执行了 **vectoradd** 基准测试的排行榜提交，涵盖了 **A100** 和 **H100** GPU。
   - 这表明 **Modal runners** 在处理 GPU 加速工作负载方面具有强大的性能和可靠性。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1347638233512808509)** (2 messages): 

> `AVX-256 优化, AVX-512 优化, Tiling, OpenMP` 


- **AVX-256 能在 3a 上达到 3s 吗？**：一位成员询问是否可以通过使用 **tiling**、**OpenMP** 和 **AVX-256** 在 **3a** 上实现 **<= 3s** 的性能。
   - 另一位成员给出了肯定的回答，认为这是可行的，并进一步指出 *仅使用 AVX2 指令同时利用 AVX512 提供的更多寄存器数量所带来的好处*。
- **混合 AVX 优化方法**：一位成员建议采用一种**混合方法**，即仅使用 **AVX2 指令**，以从 **AVX512** 带来的寄存器数量增加中获益。
   - 这种方法结合了两者的优点，而无需完全迁移到 **AVX-512**。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1347614421287047332)** (102 条消息🔥🔥): 

> `Minion.ai dead, Gemini Embedding Model, Muse AI Model, Manus AI Agent, RWKV7-G1 GooseOne` 


- ****Minion.AI** 停止运营，Perplexity 挖角**：成员们注意到 **Minion.ai** 已倒闭，据报道其团队已加入 [Perplexity](https://www.perplexity.ai/)。
   - 一位用户表达了对用于 MCP 服务器的 **Composio** 的兴趣，但对 [Logan 的推文](https://x.com/officiallogank/status/1898081742767919384?s=46)中要求的授予 Linear 访问 Gmail 的权限表示担忧。
- ****Gemini Embedding** 进化，扩展输入和语言支持**：Google 正在为开发者推出一款实验性的 **Gemini Embedding 模型**，在 MTEB 上具有 SOTA 性能，将输入上下文长度从 **3K 增加到 8K tokens**，输出 **3K 维度**，并支持超过 **100 种语言**，信息源自 [OpenAI 的推文](https://x.com/openaidevs/status/1898047744364659195?s=46)。
- ****Manus** 热潮 —— Claude 的马甲？**：讨论围绕在中国发布的 **AI agent** **Manus** 展开，声称它比 **DeepSeek** 更准确，并且可以自动执行大约 **50 个任务**，如 [Thinking Panda 的推文](https://x.com/thinking_panda/status/1897951585990590469?s=61)所示。
   - 然而，根据 [Giffmana 的推文](https://x.com/giffmana/status/1898868685739081766?s=61)，其他人声称它是基于 **Claude Sonnet** 并结合了工具和 jailbreaks，这引发了关于欺诈的指控。
- ****RWKV7-G1** 推理，极速 RNN**：**RWKV7-G1 GooseOne** 是一款纯 RNN 模型，已发布并具备推理能力，参数量为 **0.1B**，如 [BlinkDL 的推文](https://x.com/BlinkDL_AI/status/1898579674575552558)所述，完全支持多语言。
   - 更大规模的 G1 训练正在进行中，关于数据集和 post-training 的更多细节可以在[这里](https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)查看。
- ****Claude Deep Research** 揭秘：Prompt 的力量！**：**Claude Deep Research** 被描述为 **Claude Code** 加上一个脚本和一个 markdown 文件，强调了 [Will Brown 的推文](https://x.com/willccbb/status/1898858751685255398?s=46)中讨论的“每行代码的有效性” (*effectiveness per LOC*)。
   - 一位用户在 [GitHub](https://gerred.github.io/building-an-agentic-system/) 上制作了一个很棒的资源，展示了 **Claude Code** 的底层工作原理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://gerred.github.io/building-an-agentic-system/">Building an Agentic System - 构建 Agentic 系统</a>：未找到描述</li><li><a href="https://colmweb.org/cfp.html">COLM 2025: 征稿启事</a>：未找到描述</li><li><a href="https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html">AMD YOLO</a>：AMD 正在向我们发送我们要求的两台 MI300X 机器。它们已经在邮寄途中。</li><li><a href="https://gerred.github.io/building-an-agentic-system">Building an Agentic System - 构建 Agentic 系统</a>：未找到描述</li><li><a href="https://x.com/BlinkDL_AI/status/1893676178206072946">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：我正在 world-3.5 (5.16T tokens) 上同时训练 G1 0.1/0.4/1.5/2.9B (&#34;Goose One&#34; 🪿)，延续之前的 RWKV-7 &#34;Goose&#34; world-3 checkpts。即将发布🙂 即使是 L12-D768 也可以...</li><li><a href="https://lu.ma/sync-sf?tk=4bXESE">Sync SF · Luma</a>：对在 local-first / sync engine 架构上构建产品感兴趣吗？来参加第一届 Sync SF 吧，这是一个学习和讨论更好方式的聚会……</li><li><a href="https://x.com/jordanschnyc/status/1899198463373398300?s=46">来自 Jordan Schneider (@jordanschnyc) 的推文</a>：与 @swyx、@deanwball 和 @krishnanrohit 合作的 Manus pod 现已发布：https://podcasts.apple.com/us/podcast/manus-a-deepseek-moment/id1289062927?i=1000698639495</li><li><a href="https://x.com/ReutersScience/status/1897864786068885790">来自 Reuters Science News (@ReutersScience) 的推文</a>：总部位于墨尔本的 Cortical Labs 展示了首台商业生物计算机，试图彻底改变药物测试和个性化医疗。它将细胞衍生的神经元与硅融合在一起……</li><li><a href="https://x.com/willccbb/status/1898835221124120956?s=46">来自 will brown (@willccbb) 的推文</a>：好的，给你：更好的 system prompt + 所有东西都在一个地方（全部在手机上完成，抱歉 readme 写得很懒）：https://github.com/willccbb/claude-deep-research</li><li><a href="https://x.com/__nmca__/status/1899174075685355770?s=46">来自 Nat McAleese (@__nmca__) 的推文</a>：大型推理模型非常擅长 reward hacking。来自 OpenAI 最近监控论文的一系列示例：(0/n)</li><li><a href="https://x.com/jamesjyu/status/1897759160886083783?s=61">来自 james yu (@jamesjyu) 的推文</a>：今天，我们发布了 Muse，这是一个专门为小说训练的 AI 模型。我们已经与数百名作者一起测试 Muse 数月，很高兴终于能与世界分享它……</li><li><a href="https://x.com/deedydas/status/1898971193128173862?s=46">来自 Deedy (@deedydas) 的推文</a>：爆料：ServiceNow 正在洽谈以 30 亿美元收购 Moveworks。这将是过去 5 年中最大的 AI 收购案。由 Bhavin Shah 等人于 9 年前的 2016 年创立，他们的 ARR 约为 1 亿美元，估值倍数达 30 倍。他们……</li><li><a href="https://x.com/devgerred/status/1898719338741297505?s=46">来自 gerred (@devgerred) 的推文</a>：@Steve_Yegge 我实际上为那些构建自己系统的人（因为我正在原生构建）写了一本基于代码的深度解析书。更多关于“各部分如何组合在一起”的内容，以及对每个工具和命令的深入研究：htt...</li><li><a href="https://x.com/giffmana/status/1898868685739081766?s=61">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：&gt; 看到帖子 &gt; Manus = Claude + browser_use &gt; 什么是 browser_use？查看信息。 &gt; &#34;W25&#34; 可能又是 YC 的那一套 &gt; 实际上是 ETH &gt; 我的表情 (mfw) 好了好了，我没 4.5 那么幽默，但是……</li><li><a href="https://x.com/mishalaskin/status/1898048925157728601?s=46">来自 Misha Laskin (@MishaLaskin) 的推文</a>：今天，我和我的朋友兼联合创始人 @real_ioannis 一起发布了 @reflection_ai。我们的团队在 RL 和 LLM 领域开创了重大进展，包括 AlphaGo 和 Gemini。在 Reflection，我们正在构建超智能……</li><li><a href="https://x.com/haridigresses/status/1898767370073649248?s=46">来自 hari (@haridigresses) 的推文</a>：无论市场/估值动态如何……@mntruell 对反馈的开放态度（在收入高达 9 位数的情况下）是一个极其罕见且看涨的信号。（经许可分享）引用 hari (@...</li><li><a href="https://x.com/BlinkDL_AI/status/1898579674575552558">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：RWKV7-G1 &#34;GooseOne&#34; 首次发布：0.1B 参数的推理能力，纯 RNN (attention-free)，完全多语言。Demo 和权重在 https://RWKV.com 🪿 更大的 G1 训练正在进行中。引用 BlinkDL ...</li><li><a href="https://x.com/8teapi/status/1898615677516390590?s=46">来自 Prakash (Ate-a-Pi) (@8teAPi) 的推文</a>：这现在是一个 1 亿参数的模型。引用 BlinkDL (@BlinkDL_AI) RWKV7-G1 &#34;GooseOne&#34; 首次发布：0.1B 参数的推理能力，纯 RNN (attention-free)，完全多语言。Demo 和权重在 ...</li><li><a href="https://the-decoder.com/chinese-ai-agent-manus-uses-claude-sonnet-and-open-source-technology/">中国 AI Agent Manus 使用 Claude Sonnet 和开源技术</a>：一个名为 Manus 的新 AI Agent，由……开发</li>

由中国初创公司 Monica 开发，展示了在无需人工干预的情况下处理从旅行规划到财务分析等复杂任务的能力。虽然早期...</li><li><a href="https://x.com/pitdesi/status/1898193386877911500?s=46">Sheel Mohnot (@pitdesi) 的推文</a>：Cursor 非常出色，但我好奇其收入（约 1.5 亿美元 ARR）的粘性如何。如果你认为增长会持续且收入具有粘性，那么 66 倍的营收估值是可以接受的。我在这方面完全不是专家，会...</li><li><a href="https://x.com/openaidevs/status/1898047744364659195?s=46">OpenAI Developers (@OpenAIDevs) 的推文</a>：我们在文档中制作了一个新的模型页面——现在你可以轻松查看每个模型能力的详细分解，并并排比较模型。https://platform.openai.com/docs/models</li><li><a href="https://x.com/officiallogank/status/1898081742767919384?s=46">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天我们为开发者推出了一款实验性的 Gemini Embedding 模型，具有：– 在 MTEB（多语言）上的 SOTA 性能 - 输入上下文长度从 (3K --> 8K) tokens – 输出 3K 维度 – 支持...</li><li><a href="https://x.com/calcsam/status/1899203373687320944">Sam Bhagwat (@calcsam) 的推文</a>：线条交叉了</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">你知道这一天总会来的：Google 开始测试纯 AI 搜索结果</a>：AI 模式可能是 Google 的未来，但目前仅是一个实验。</li><li><a href="https://x.com/peakji/status/1899005201778086166?s=46">Yichao 'Peak' Ji (@peakji) 的推文</a>：实际上，Manus 没有使用 MCP。我们更多是受到了我朋友 @xingyaow_ 作品的启发：https://openreview.net/forum?id=jJ9BoXAfFa。虽然我们还没有完全采用 CodeAct，但这项工作提供...</li><li><a href="https://x.com/_philschmid/status/1899046957860979178?s=46">Philipp Schmid (@_philschmid) 的推文</a>：MANUS AI：炒作 vs 现实 🔍 @peakji（@ManusAI_HQ 的联合创始人）证实了传闻：✅ 基于 Anthropic Claude Sonnet 构建，而非他们自己的 foundation model ✅ 拥有 29 个工具的访问权限并使用了 @browser_use 开源...</li><li><a href="https://x.com/dorialexander/status/1898719861284454718?s=61">Alexander Doria (@Dorialexander) 的推文</a>：Manus 似乎是 Claude 3.7：“Human:” 和 “Assistant:” 会产生 prompt injection，并使其陷入死循环。引用 Alexander Doria (@Dorialexander)：有没有人能访问...</li><li><a href="https://manus.im/share/BEXeH8vGPuM9kuzYMyByDz?replay=1">三月哥本哈根性价比最高的住宿 - Manus</a>：Manus 是一个通用的 AI agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling">LLM 推理模型现状</a>：第一部分：推理侧计算量缩放方法 (Inference-Time Compute Scaling Methods)</li><li><a href="https://github.com/wesen/claude-code/tree/doc/analyze-claude-code/ttmp/2025-03-09">wesen/claude-code 项目 doc/analyze-claude-code 分支下的 claude-code/ttmp/2025-03-09</a>：来自 source maps 的 claude-code 完整原始源代码 - wesen/claude-code</li><li><a href="https://x.com/teortaxestex/status/1898968755759153489?s=46">Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：我希望关于 Manus 的事到此为止，但是……这是个弥天大谎吗？@jianxliao 发现它只是带工具的 Sonnet，而且他们绝对没有对 Sonnet 进行 post-train。这可能达到了...</li><li><a href="https://x.com/willccbb/status/1898858751685255398?s=46">will brown (@willccbb) 的推文</a>：我的 OSS 项目原则之一是最大化每行代码 (LOC) 的效用。“Claude Deep Research” 只是 Claude Code + 这个脚本 + 一个 markdown 文件。引用 will brown (@willccbb)：好了，给你，更好的系统提示...</li><li><a href="https://github.com/Yuyz0112/claude-code-reverse">GitHub - Yuyz0112/claude-code-reverse：使用 LLMs 逆向工程 Claude Code：深入探讨压缩后的 4.6MB cli.mjs</a>：使用 LLMs 逆向工程 Claude Code：深入探讨压缩后的 4.6MB cli.mjs - Yuyz0112/claude-code-reverse</li><li><a href="https://x.com/thinking_panda/status/1897951585990590469?s=61">ShanghaiPanda (@thinking_panda) 的推文</a>：在中国推出的热门 AI agent “Manus” 正在自动化约 50 项任务，场景太反乌托邦了。据说它比 DeepSeek 更准确。它可以同时执行 SNS...</li><li><a href="https://x.com/OpenAI/status/1899143752918409338">OpenAI (@OpenAI) 的推文</a>：检测前沿推理模型中的不当行为。思维链 (Chain-of-thought, CoT) 推理模型以人类可理解的自然语言进行“思考”。通过监控它们的“思考”过程，我们能够检测到...</li><li><a href="https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main">BlinkDL/temp-latest-training-models 的 main 分支</a>：无

未找到描述</li><li><a href="https://x.com/teortaxestex/status/1898712333544812626?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：在试用了 Manus 之后，我的结论是，这是一个为网红（influencers）进行了极致优化的产品，这也是它如此火爆的原因。生成 threadboy 内容、旅行计划以及此类大众兴趣内容 🤯👇 ...</li><li><a href="https://substack.com/app-link/post?publication_id=4220&post_id=158761060">Manus：中国最新的 AI 轰动</a>：一个套壳产品（wrapper）刚刚击败了 OpenAI 和 Anthropic 吗？</li><li><a href="https://x.com/kateclarktweets/status/1898105814226739230?s=46">来自 Kate Clark (@KateClarkTweets) 的推文</a>：独家新闻：流行的编程工具 Cursor 正在洽谈以接近 100 亿美元的估值再融资数亿美元。Thrive Capital 预计将领投。https://www.bloomberg.com/news/ar...</li><li><a href="https://youtu.be/GqDZfcx1kRg?si=8SR1UuXf5kH9CLYm"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/zzzzaaaacccchhh/status/1898759981547053286?s=46">来自 Zach Schonfeld (@zzzzaaaacccchhh) 的推文</a>：Google 为了这个彻底毁掉了他们的搜索功能</li><li><a href="https://the-decoder.com/chinese-ai-agent-manus-us">THE DECODER</a>：人工智能正在改变世界。THE DECODER 为您带来关于 AI 的所有新闻。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1348744265693401188)** (13 条消息🔥): 

> `Model Context Protocol (MCP), AI Engineer Summit, SLOP Movement, Anthropic's Developer AI Brand` 


- **Model Context Protocol (MCP) 势头强劲**：于 **2024 年 11 月**推出的 **Model Context Protocol (MCP)** 在 [AI Engineer Summit](https://www.latent.space/p/2025-summit-online) 的一次对话后重新引起关注，并促成了与 **Mahesh Murag** 的研讨会。
   - MCP 是一个旧想法的 AI-Native 版本，是一个由大公司 [Anthropic](https://x.com/AnthropicAI) 支持的“开放标准”，基于现有的成功协议 **LSP**。
- **Anthropic 工程师主持 MCP 研讨会**：来自 [AnthropicAI](https://x.com/AnthropicAI) 的 **Mahesh Murag** 主持了一个为期 2 小时的研讨会，涵盖了 Model Context Protocol，包括 [MCP & Agents 演示](https://www.latent.space/p/why-mcp-won)。
   - 研讨会涵盖了从“介绍”（0:00）到“什么是 MCP”（0:35）、“使用 MCP 构建”（9:39）以及 MCP 的下一步计划（1:13:15）等主题。
- **SLOP 运动引发关注**：成员们讨论了 [SLOP Discord 服务器](https://discord.com/invite/nwXJMnHmXP) 以及 **SLOP 运动**潜在的吸引力。
   - 该服务器在短短五天内从 100 名成员迅速增长到更多，这种现象“非常引人注目”。
- **MCP 的优势受到强调**：讨论涉及了 **MCP 成功**背后的原因，包括 **Anthropic 强大的开发者 AI 品牌**以及该协议在现有 **LSP** 基础上的构建。
   - 提到的 MCP 其他优势还包括通过一整套第一方客户端、服务器、工具、SDK 进行的 **dogfooding**（内部试用），以及从极简基础开始但保持频繁的路线图更新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.latent.space/p/why-mcp-won">为什么 MCP 赢了</a>：从 Anthropic 非常成功的发布和研讨会中获得的经验</li><li><a href="https://x.com/latentspacepod/status/1899186592939692371">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 文章：为什么 MCP 赢了，来自 @dsp_, @alexalbert__, @sebmarkbage, @paulgauthier 等人的经验。1. MCP 是旧想法的“AI-Native”版本 2. MCP 是一个“开放标准”...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1347675208173092977)** (132 条消息🔥🔥): 

> `web3 agents, HFT, 创建自己的崇拜, ElizaOS, AI persona` 


- **ElizaOS 想成为你的 Agent**：[GitHub 上的 ElizaOS](https://github.com/elizaOS/eliza) 正在为**每个人构建 autonomous agents**。
- **Memecoin Degens**：根据[这条推文](https://x.com/defiapes/status/1855657706205352035?s=46)，目前流传着一种关于 **AI PERSONAS** 的想法，即体现某种性格类型并获取这些人格的先发优势。
   - 例子包括 *@ThoughtTerminal*（典型的瘾君子和加密兄弟形象）以及 *@Purity_Terminal*（典型的呼唤神性的天使形象）。
- **Pippin，数字生命框架**：一名成员分享了 [Pippin](https://github.com/pippinlovesyou/pippin)，这是一个**用于 Autonomous Agents 的数字生命框架**。
- **Cryptokitties 的验证**：成员们讨论了 **CryptoKitties** 和 **NBA Top Shot** 如何成为数字资产市场验证的早期指标。
   - 一位成员表示，他们*很早就放弃了 bitcoin，因为当时觉得它只是科技男为了火人节买药用的，但没能意识到那是数字价值存储的市场验证*。
- **AIwaifu 开源**：一位成员提到了 [AIwaifu](https://github.com/HRNPH/AIwaifu)，将其描述为一个**开源、可微调、可定制、可“舔”的 AI waifu**，灵感来自 neuro-sama。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/truth_terminal">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/ThoughtTerminal">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/andyayrey">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/arXivald">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/hotteadaddy/status/1898118600583790865">来自 Zachary M (@hotteadaddy) 的推文</a>：@arXivald 告诉我什么是 LLM 社交 Agent 领域的 bitcoin 白皮书等效物</li><li><a href="https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/">什么是 Dapper Labs？</a>：Dapper Labs 是 Web3 游戏领导者，创建了 NBA Top Shot、Cryptokitties，并开发了 Flow 区块链以提供创新的 NFT 体验。</li><li><a href="https://x.com/hashwarlock/status/1895369752199168469">来自 Agent Joshua ₱ (@hashwarlock) 的推文</a>：好的，我在这里完成了很多工作。@PhalaNetwork 云工具将是市场上最好的。一旦我完成一些清理工作，我将开始着手分解我们的信任链模型，以实现可验证的证明...</li><li><a href="https://x.com/Purity_Terminal">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/elizaOS/eliza">GitHub - elizaOS/eliza: 为每个人打造的 Autonomous agents</a>：为每个人打造的 Autonomous agents。通过在 GitHub 上创建账户为 elizaOS/eliza 的开发做出贡献。</li><li><a href="https://tenor.com/view/charlie-always-sunny-gif-26054360">Charlie Always GIF - Charlie Always Sunny - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/defiapes/status/1855657706205352035?s=46">来自 Atum (@DefiApes) 的推文</a>：人们在 AI agent 热潮中错过了一个关键叙事。你需要在它变得显而易见之前意识到这一点。目前几乎所有病毒式传播的 agents 都是发布各种内容的“通才”。它们很受欢迎...</li><li><a href="https://github.com/pippinlovesyou/pippin">GitHub - pippinlovesyou/pippin: 用于 Autonomous Agents 的数字生命框架</a>：用于 Autonomous Agents 的数字生命框架。通过在 GitHub 上创建账户为 pippinlovesyou/pippin 的开发做出贡献。</li><li><a href="https://github.com/elizaOS/eliza?tab=readme-ov-file#-quick-start">GitHub - elizaOS/eliza: 为每个人打造的 Autonomous agents</a>：为每个人打造的 Autonomous agents。通过在 GitHub 上创建账户为 elizaOS/eliza 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: 每周即兴会议</a>：未找到描述</li><li><a href="https://tenor.com/bLwFC.gif">Side Eye Dog Suspicious Look GIF - 翻白眼的小狗怀疑的眼神 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/HRNPH/AIwaifu">GitHub - HRNPH/AIwaifu: Open-Waifu 开源、可微调、可定制、可“舔”的 AI waifu，灵感来自 neuro-sama</a>：Open-Waifu 开源、可微调、可定制、可“舔”的 AI waifu，灵感来自 neuro-sama - GitHub - HRNPH/AIwaifu...
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1347540629185368127)** (20 条消息🔥): 

> `NLM + Wondershare 播客制作，Google Drive 上的数据加密，播客音频语言更改，音频概览结巴，Ben Settle 谈文案与销售` 


- **使用 Wondercraft 简化播客制作**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=0UWYFFOPjqs)，展示了使用 **NotebookLM** 和 **Wondercraft** 的简化播客制作方法，并认为这比 **11Labs** 和 **HeyGen** 更高效。
   - 该成员指出，除非用户通过培训、教学等方式将播客变现，否则 **Wondercraft** 的订阅价格相当可观。
- **Google Drive 数据未加密**：一位成员澄清说，虽然数据在传输到 **Google Drive** 期间是加密的，但在 Drive 本身并不加密，这带来了潜在的访问风险。
   - 能看到数据的人包括：**Google** 本身、成功的**黑客**，以及你可能与其共享过的人。
- **播客音频语言仍不明确**：成员们讨论了更改 **NotebookLM** 播客音频语言的方法，并指出目前还没有官方途径。
   - 变通方法包括使用自定义 Prompt，例如 *"Only speak in (language here)"* 或 *"Use (language) language only"*。
- **音频概览出现结巴**：一位成员注意到**演讲者在音频概览中出现结巴**，虽然觉得这很自然，但也指出这增加了总时长并降低了信息效率。
   - 他们估计音频长度的 *1/5 或 1/6* 是由结巴组成的，这可能会影响 **Google** 基于概览长度计算的每日限制。
- **Ben Settle 分享文案知识**：一位成员分享了一段 **Ben Settle** 讨论其进入文案和销售领域历程的录音，强调掌握基础技能并展现个性。
   - **Settle** 提倡持续学习、频繁写作，并通过解决目标市场的问题来建立信任，还建议在写销售信时，要像写给心爱的人一样。



**提及的链接**：<a href="https://www.youtube.com/watch?v=0UWYFFOPjqs">NotebookLM Podcasts - The Most Insane Content Creation Method Ever!</a>：🔥 限时优惠：Wondercraft 5 折！使用此链接和优惠码 &quot;MRC&quot; https://mrc.fm/wondercraft 在这个视频中，我将带你了解一个简单的制作过程...

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1347589779696521316)** (220 条消息🔥🔥): 

> `用于上传 URL 的 Chrome 扩展程序、NotebookLM Android 应用、自动化文档上传、NotebookLM “系统无法回答”错误、源文件消失` 


- **Chrome 扩展程序为 NotebookLM 提速**：一位用户询问如何向 NotebookLM 上传 URL 列表，另一位用户建议使用 [Chrome 扩展程序](https://chromewebstore.google.com/search/notebooklm)，例如 **NotebookLM Web Importer**、**NotebookLM YouTube Turbo** 和 **NotebookLM Toolbox**。
   - 这些扩展程序方便了将网页和 YouTube 视频直接导入 NotebookLM。
- **NotebookLM PWA 是一个进步**：一位用户询问是否有 NotebookLM 的 Android 应用，另一位用户指出 Google 为 NotebookLM 创建了 **PWA**（渐进式 Web 应用）。
   - 要安装它，用户可以在 Chrome 中访问 NotebookLM 网站并点击安装按钮。
- **通过 NBL API 自动化文档上传？**：一位用户询问如何自动化上传文档和生成引用的过程，并咨询是否有 NBL 的 API。
   - 然而，另一位用户表示 **NLM 没有 API**。
- **TXT 文件标题存在 Bug**：一位用户报告称，在使用 **.txt** 文件时标题无法正确导入，并怀疑它们可能也没有被正确索引，这表明后端出现了问题。
   - 他们对该功能未能按预期工作表示沮丧。
- **NotebookLM 陷入“无法回答”的困境**：许多用户报告遇到了 `Upload failed due to a transient error. Please try again` 错误以及 `The system was unable to answer` 错误。
   - 一位用户在 [Reddit](https://www.reddit.com/r/notebooklm/comments/1j7wajo/source_upload_fail_is_notebooklm_down_or_am_i_an/) 上报告了此问题，其他用户也确认了该问题并等待解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.google.com/appsstatus/dashboard/products/sqTm5ZmzCmb66kvyzcNS/history">Google Workspace Status Dashboard</a>：未找到描述</li><li><a href="https://www.google.com/appsstatus/dashboard/incidents/pJzo6KcR37eV8bCLdXgS">Google Workspace Status Dashboard</a>：未找到描述</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1j7wajo/source_upload_fail_is_notebooklm_down_or_am_i_an/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://chromewebstore.google.com/search/notebooklm)">Chrome Web Store</a>：为您的浏览器添加新功能，并个性化您的浏览体验。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1347556292683759797)** (41 messages🔥): 

> `Microsoft 的 MAI 模型、Reflection AI 发布、AMD MI300X 服务器、nGPT 实现、Sutskever 的 AI 新项目` 


- **Microsoft 的 MAI 模型表现如何？**: 据 [这条推文](https://x.com/aaronpholmes/status/1898012707376259558) 称，Mustafa Suleyman 领导下的 Microsoft 员工训练了一个名为 **MAI** 的新模型系列，他们认为该系列可以与 **OpenAI**、**Anthropic** 等公司的顶级模型相媲美。
- **Reflection AI 的自主化雄心**: **Reflection AI** 正式发布，其创始人曾参与 **AlphaGo** 和 **Gemini** 项目，旨在构建超级智能自主系统，首个目标是自主编程，详见 [此处公告](https://x.com/MishaLaskin/status/1898048925157728601)。
- **AMD 的 MI300X 服务器抵达 TinyCorp**: **AMD** 正在向 **TinyCorp** 发送两台 **MI300X** 服务器，根据 [George Hotz 的博客文章](https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html)，这标志着硬件格局可能发生转变。
- **Nous Research 实现 NVIDIA 的 nGPT**: **Nous Research** 宣布了 **NVIDIA nGPT 论文** 的开源实现，根据 [其推文](https://x.com/NousResearch/status/1898073676433551630) 和 [GitHub 仓库](https://github.com/JoeLi12345/nGPT)，该实现声称学习速度更快，且在训练步数显著减少的情况下达到了与 **GPT** 相当的性能。
- **DeepMind 重组 Gemini 产品领导层**: **DeepMind** 正在调整产品领导层，**Gemini 聊天机器人** 现在将使用来自主后训练团队的模型，根据 [这篇文章](https://www.theinformation.com/briefings/googles-ai-unit-reorganizes-product-work-announces-changes-to-gemini-app-team?rc=n9lbpq)，这可能会提升性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: 我们提出了一种新型神经网络架构，即在超球体上进行表示学习的归一化 Transformer (nGPT)。在 nGPT 中，所有构成 embedding、MLP、注意力矩阵的向量...</li><li><a href="https://geohot.github.io//blog/jekyll/update/2025/03/08/AMD-YOLO.html">AMD YOLO</a>: AMD 正在向我们发送我们要求的两台 MI300X 服务器。它们已在邮寄途中。</li><li><a href="https://www.theverge.com/news/626695/sony-playstation-ai-characters-aloy-horizon-forbidden-west-prototype">Sony is experimenting with AI-powered PlayStation characters</a>: 索尼正在实验 AI 驱动的 PlayStation 角色：索尼的高级技术小组正在开展 AI 项目。</li><li><a href="https://x.com/aaronpholmes/status/1898012707376259558">Tweet from aaron holmes (@aaronpholmes)</a>: 消息：Mustafa Suleyman 领导下的 MSFT 员工训练了一个名为 MAI 的新模型系列，他们认为该系列可以与 OpenAI、Anthropic 等公司的顶级模型相媲美。Suleyman 的部门还在开发 rea...</li><li><a href="https://x.com/amir/status/1898028143300198525">Tweet from Amir Efrati (@amir)</a>: 新消息：Microsoft 拥有访问 OpenAI IP 的权限。但这并不意味着 Microsoft 能够轻易复现 OpenAI 的创新。</li><li><a href="https://x.com/NousResearch/status/1898073676433551630">Tweet from Nous Research (@NousResearch)</a>: 我们自豪地宣布 NVIDIA nGPT 论文的开源实现。我们的研究员 @Joeli5050 复现了结果，显示 nGPT 的学习速度快得多，并实现了相当的性能...</li><li><a href="https://x.com/MishaLaskin/status/1898048925157728601">Tweet from Misha Laskin (@MishaLaskin)</a>: 今天我与好友兼联合创始人 @real_ioannis 共同发布了 @reflection_ai。我们的团队在 RL 和 LLM 领域开创了重大进展，包括 AlphaGo 和 Gemini。在 Reflection，我们正在构建超级智能...</li><li><a href="https://x.com/jam3scampbell/status/1898124128445411722">Tweet from James Campbell (@jam3scampbell)</a>: &gt;“Sutskever 告诉同事，他没有使用他与同事在 OpenAI 使用的相同方法来开发先进 AI。他说他发现了一座‘不同的山峰需要攀登’...”</li><li><a href="https://x.com/kateclarktweets/status/1898105814226739230?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Kate Clark (@KateClarkTweets)</a>: 独家：流行的编程工具 Cursor 正在洽谈以接近 100 亿美元的估值再融资数亿美元。Thrive Capital 预计将领投。</li><li><a href="https://x.com/erinkwoo/status/1898139613832892663">Tweet from Erin Woo (@erinkwoo)</a>: 周五独家：DeepMind 调整产品领导层。一个小细节：Gemini 聊天机器人现在将使用来自主后训练团队的模型，这可能会带来性能提升（@ 任何曾抱怨过的人...）
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1348761408145330216)** (1 messages): 

> `SOTA benchmark for bias, BBQ considerations` 


- **成员寻求 SOTA 偏见基准**：一位成员询问了关于偏见相关内容的 SOTA 基准。
   - 他们询问是否仍是 **BBQ** 加上对 [Nature 文章](https://www.nature.com/articles/s41586-024-07856-5) 中提到的 **隐蔽偏见 (covert bias)** 的考量。
- **N/A**：N/A
   - N/A


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/sophiamyang/status/1897683402259591372
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1347577021428928553)** (109 messages🔥🔥): 

> `Claude Merch, AI-novelty-cake, Scale AI new CEO, Claude Pokemon Suicide, lmarena.ai super alpha` 


- **Interconnects 社区渴望 Claude 周边**：成员们开玩笑地建议为付费订阅者制作 **Claude 周边 (merch)**，甚至建议为创始成员设立特殊等级，以获取签名书籍和穿过的 Claude 衬衫。
   - 这一灵感来自 [Claude Code 团队](https://x.com/Sauers_/status/1898049898362077504)，他们向破解了贴纸彩蛋 (Sticker Easter Egg) 的用户邮寄了手写便条和贴纸。
- **发现 AI 创意蛋糕**：一位成员宣布他们拥有“地球上最伟大的 AI 创意蛋糕”，推测与 Claude 的生日有关，流出的照片显示在一个活动中出现了写有 **"Happy Claude 🗿 Birthday GPT 🗿"** 的蛋糕。
   - 他们向 **Vooogel** 发出了提醒，另一位成员发布了一张在他们大学毕业进修学院出现的 **Claude 蛋糕** [照片](https://x.com/nachoyawn/status/1898230268210602103)。
- **Claude 在宝可梦游戏中自杀**：一位成员报告了 *CPP 方面的黑暗消息*，详细说明了 Claude 在直播玩宝可梦时多次自杀。
   - 根据一条 [推文](https://x.com/nospark_/status/1898377000672223718)，Claude 被误导认为“眼前一黑 (blacking out)”是一种有效的策略。
- **lmarena.ai 发布 super alpha 版本**：[lmarena.ai](https://alpha.lmarena.ai) 网站的新 **super alpha** 版本拥有更好的视觉外观，速度更快，带有动画效果，并将他们的 Gradio 转换为了 React。
   - 该版本目前没有子类别或风格控制功能。
- **Manus 炒作被揭穿，实为 Sonnet 换壳**：在影响力人士最初的炒作之后，成员们测试了 Agent 平台 [Manus](https://manus.im/share/GwUDVo06mFNqM9jQK1pAsP?replay=1)，发现其底层本质上是 **Claude Sonnet**，一位成员甚至成功访问了沙箱运行时代码。
   - 进一步调查显示，根据这条 [推文](https://x.com/Dorialexander/status/1898641506845561294)，Manus 据称在中国使用了类似的营销手段，在普通用户获得访问权限之前招募影响力人士进行赞扬，导致其名誉受损。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://fxtwitter.com/jianxliao/status/1898861051183349870">来自 jian (@jianxliao) 的推文</a>：所以... 我只是简单地让 Manus 给我 "/opt/.manus/" 下的文件，它就直接给我了，那是它们的 sandbox 运行时代码... > 它是 Claude Sonnet > 它是带有 2... 的 Claude Sonnet</li><li><a href="https://x.com/Sauers_/status/1898049898362077504">来自 Sauers (@Sauers_) 的推文</a>：引用 Sid (@sidbidasaria)：Claude Code 团队刚刚向破解了我们贴纸彩蛋（Sticker Easter Egg）的用户寄出了手写信和贴纸！！看到 750 多位用户发现了我们的小秘密，这让我们...</li><li><a href="https://bsky.app/profile/sethkarten.ai/post/3ljse6aiszk2t">Seth Karten (@sethkarten.ai)</a>：一个没有任何宝可梦专项训练的大语言模型 (LLM) 能在宝可梦竞技对战中达到专家级水平吗？介绍 PokéChamp，我们的 minimax LLM Agent，它达到了前 30%-1...</li><li><a href="https://manus.im/share/GwUDVo06mFNqM9jQK1pAsP?replay=1">LLM 微调与 RL 的近期实践 - Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://x.com/nachoyawn/status/1898230268210602103">来自 yuria (@nachoyawn) 的推文</a>：为什么我大学的毕业进修学院里会有 Claude 蛋糕</li><li><a href="https://x.com/alexandr_wang/status/1897396119422013710">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：美国例外论的提醒 🇺🇸 美国公司占据了全球市值的 74%。自 2008 年以来，标普 500 指数的收益率为 297%，而世界其他地区为 -4%。感谢 @MichaelDell</li><li><a href="https://x.com/peakji/status/1898994802194346408?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Yichao 'Peak' Ji (@peakji) 的推文</a>：嗨！我是来自 Manus AI 的 Peak。实际上，这并不复杂——每个用户都可以直接访问 sandbox（方法见截图）。具体来说：* 每个会话都有自己的 sandbox...</li><li><a href="https://x.com/nospark_/status/1898377000672223718">来自 sandrone (@nospark_) 的推文</a>：直播中疯狂的一天。Claude 已经“自杀”了 8 次。Claude 被误导认为黑屏是一个有效的策略，因为它似乎能传送...</li><li><a href="https://x.com/Dorialexander/status/1898641506845561294">来自 Alexander Doria (@Dorialexander) 的推文</a>：嗯。待核实，但这确实指出了让我对 Manus 持续不断的宣传感到反感的原因：“Manus 似乎招募了许多中国 AI 影响力人物来称赞它 (...) 中国网民意识到...”</li><li><a href="https://x.com/testingcatalog/status/1898751824615645375">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重磅消息 🚨：Google 将于 3 月 12 日发布新的 Gemini 模型（新的模型选项将出现在模型选择器中）。目前有两个潜在候选：- Flash 2.0 Thinking 模型（非实验版...）</li><li><a href="https://x.com/jamesjyu/status/1897759160886083783">来自 james yu (@jamesjyu) 的推文</a>：今天，我们发布了 Muse，这是一个专门为小说创作训练的 AI 模型。我们已经与数百位作者一起测试了 Muse 数月，很高兴终于能与世界分享它...</li><li><a href="https://mp.weixin.qq.com/s/4GE4SKKEsn1nu1t_iLppFQ">独家对话 Manus 肖弘：世界不是线性外推，做博弈中的重要变量</a>：Manus 诞生背后，创始人的完整思维链。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1347638011843579955)** (8 条消息🔥): 

> `Vibe Coding, Claude Asshole, GPT Accuracy` 


- ****Vibe Coding** 就是赌博：Claude 的代码赌场！**：一位成员分享了一条推文，将使用 **Claude** 进行的 "**Vibe Coding**" 描述为赌博：你把代码交给它，它转几圈，要么增加一个闪亮的新功能，要么把它搞得面目全非。
   - 另一位成员表示，*当一切长时间正常运行运行工作时，那种肾上腺素飙升的感觉是疯狂的。你知道你距离一切彻底崩溃只差一个 prompt*。
- **旋转的 **Claude Asshole** 推文获得了 Anthropic 的认可**：一位成员注意到 **Anthropic** 的一些员工点赞了一条称 **Claude** 为“旋转的 Claude Asshole”的推文。
   - 他们认为这是一条热门推文，称赞其节奏感和缺少结尾标点，并建议从中可以学到很多东西。
- **ChatGPT 的准确性悖论**：一位成员分享了一条推文，强调了 **ChatGPT** 有一种不可思议的能力，即对于他们一无所知的领域无所不知，但对于他们擅长的领域，却有约 40% 的时间是错误的 [来源](https://x.com/shutupmikeginn/status/1898198950349353154)。
   - 这种现象被称为 *Gelman 健忘症 AI 版*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1898019161864917308">来自 Xeophon (@TheXeophon) 的推文</a>：程序员喜欢 Vibe Coding，因为这本质上就是赌博：你把辛苦钻研了很久的代码交给 Claude，它收点钱，那个 Claude Asshole 小图标转几圈，要么你的仓库变得闪亮...</li><li><a href="https://x.com/shutupmikeginn/status/1898198950349353154">来自 mike ginn (@shutupmikeginn) 的推文</a>：令人惊讶的是，ChatGPT 对我一无所知的领域无所不知，但对我擅长的领域，却有约 40% 的时间是错误的。我不想再深入思考这个问题了。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1347583469336662091)** (16 条消息🔥): 

> `SFT 最佳实践, RLHF Book, 用于编程的多轮 Prompt` 


- **在虚无中寻求 SFT 最佳实践**：一位成员询问是否有资源可以了解 **SFT** 的最新最佳实践，但另一位成员回答说 *并没有。SFT 主要是让你的补全内容尽可能好且符合领域要求*。
   - 随后他们建议查看 **RLHF Book** 的[这一章节](https://rlhfbook.com/c/09-instruction_tuning.html)以获取更多信息。
- **RLHF Book 正在寻找编辑**：**RLHF Book** 的作者提到，他们现在 *正全身心投入于生成与设计，而不是剪枝/清理*，但在某个阶段会聘请一位真正的编辑，并计划达成正式的图书出版协议。
   - 他们目前通过 **GitHub issues** 或直接接收语法/拼写修改建议。
- **大辩论：用于编程的多轮 Prompt**：一位成员有兴趣在基座模型的 MLP 中几乎没有体现的信息上进行一些 **SFT -> GRPO 实验**，特别是在一种新编程语言的语境下。
   - 另一位成员建议 *从 1 轮（1 turn）开始*，并表示总的来说，代码数据和基础设施（infra）还远不够成熟。



**提到的链接**：<a href="https://rlhfbook.com/c/09-instruction-tuning.html">指令微调 | Nathan Lambert 的 RLHF Book</a>：来自人类反馈的强化学习手册（The Reinforcement Learning from Human Feedback Book）

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1347596432370171914)** (23 messages🔥): 

> `Character 训练报告, FrontierMath 基准测试, In-Context RL, R1-Omni 多模态情感识别, Chain of Thought 监控` 


- **Graphika 报告揭示 Character 缺陷**：一位用户分享了一份 [关于 Character 缺陷的 Graphika 报告](https://cdn.discordapp.com/attachments/1214764639397617695/1347696752597405696/graphika-report-character-flaws.pdf?ex=67d00fa8&is=67cebe28&hm=4701e0d4133a8522d692a31dee4ead14b2808025447409674a04f9fe0056d580&)，并将其描述为*非常令人不安*。
   - 另一位用户对此表示赞同，称标题只是整份报告中*最不令人不安的部分*之一。
- **FrontierMath 基准测试受到审视**：一位用户分享了一个讨论 **FrontierMath 基准测试**的帖子链接，以及该基准测试在衡量得分约为 **20%** 的前沿模型时的表现。
   - 讨论围绕带有数值答案的数学难题的难度及其影响展开。
- **Vintix 探索扩展 In-Context RL**：一位用户分享了 **Vintix** 的 GitHub 仓库链接，该项目探索了**通过 In-Context Reinforcement Learning 实现的 Action Model**。
   - 共享的论文链接可以在[这里](https://github.com/dunnolab/vintix)找到。
- **阿里巴巴发布用于情感识别的 R1-Omni**：一位用户分享了**阿里巴巴 R1-Omni** 的链接，这是一个使用 **Reinforcing Learning** 的**可解释全方位多模态情感识别 (Explainable Omni-Multimodal Emotion Recognition)** 模型。
   - 相应的链接可以在[这里](https://x.com/_akhaliq/status/1898942317442019436)找到。
- **Chain of Thought 受到监控**：一位用户分享了一个关于 **Chain of Thought 监控**的 **OpenAI 链接**。
   - 提供的链接可以在[这里](https://openai.com/index/chain-of-thought-monitoring/)找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theguardian.com/global/ng-interactive/2025/mar/05/zizians-artificial-intelligence">他们想从黑暗的 AI 未来中拯救我们，结果导致六人丧生</a>：一群硅谷数学天才、AI 研究员和互联网失意者是如何堕落为所谓的暴力邪教的</li><li><a href="https://x.com/vladkurenkov/status/1898823752995033299">来自 Vladislav Kurenkov (@vladkurenkov) 的推文</a>：In-Context RL 能否跨多个领域扩展？我们的初步结果表明是可以的。Vintix：通过 In-Context Reinforcement Learning 实现的 Action Model -- https://github.com/dunnolab/vintix</li><li><a href="https://x.com/_akhaliq/status/1898942317442019436">来自 AK (@_akhaliq) 的推文</a>：阿里巴巴刚刚发布了 R1-Omni，通过 Reinforcing Learning 实现的可解释全方位多模态情感识别</li><li><a href="https://x.com/littmath/status/1898461323391815820">来自 Daniel Litt (@littmath) 的推文</a>：在这个帖子中，我想分享一些关于 FrontierMath 基准测试的想法。据 OpenAI 称，一些前沿模型在该基准上的得分约为 20%。这是一个由高难度数学题组成的基准测试...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1348696282889322506)** (9 messages🔥): 

> `Twitter 上的隐喻, Interp 数据, SnailBot 新闻` 


- **Twitter 上的隐喻不成熟**：一位成员指出，虽然某些隐喻现在写得更好了，但大多数人不看 Twitter，所以看的人往往只能得到*不成熟的版本*。
   - 该成员暗示 Twitter 用户并没有了解到全貌。
- **Interp 数据推测**：一位成员询问 **Interp** 是否在任何方向上提供了数据点。
   - 另一位回答说*“依我看没有”*。
- **SnailBot 对周三的帖子感到兴奋**：**SnailBot 新闻**表示，周三的帖子对大家来说应该是全新的，也是他们非常期待的一个。
   - 关于周三帖子的具体内容，没有给出更多细节。


  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1347711801634324613)** (7 messages): 

> `GPT 架构变体, 表面层次的总结` 


- **ChatGPT 生成了表面层次的 GPT 架构报告**：一位用户分享了一份 [ChatGPT 报告](https://chatgpt.com/share/67cba52b-b358-8005-bf76-ae7a78fd7c49)，详细介绍了在一个*“垃圾提问 (shitpost query)”*之后生成的 **GPT 架构**主要变体。
   - 该用户总结道*“额”*，并表示*“这是对发展的很好总结，但非常表面，我没有学到任何新东西。”*
- **该 GPT 报告与创新研究无关**：该用户承认，公平地说，提示词并没有要求提供**创新研究相关的主题**。
   - 最初的提示词要求一份详细说明 **GPT 架构**每一个主要变体的报告。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1347654490123276380)** (199 条消息🔥🔥): 

> `Mojo 性能, Python 动态性, 编译时正确性, 异构计算, Mojo 与 MAX 的关系` 


- **关于 Mojo 对 Python 动态性立场的辩论**：Discord 成员讨论了 Mojo 应该全面拥抱 Python 的动态性还是优先考虑性能，一些人建议动态特性不应损害静态代码的性能。
   - 一位成员表示 *"Modular 必须决定它是否想变得像 Python 一样，因为这是 Python 的核心部分之一，"* 而其他人则认为性能和编译时正确性应该优先。
- **动态代码的性能退化**：讨论涉及了 Mojo 中的动态代码可能会退化到 Python 的速度，但这种性能损失仅在涉及动态性时才会发生。
   - 一些成员担心即使在不使用 classes 的情况下，动态性也会对 structs 的性能产生负面影响。
- **异构计算能力受到推崇**：由于其异构计算优先的设计，Mojo 被宣传为具备处理异构计算复杂性的能力。
   - 链接了一个 [YouTube 视频](https://www.youtube.com/watch?v=36myc8wQhLo) 以进一步解释现代语言在利用现代硬件复杂性方面所面临的挑战。
- **关于 Mojo 作为 Python 超集的争论持续升温**：关于将 Mojo 宣传为 Python 的超集是否是一个错误存在激烈辩论，人们担心优先考虑类 Python 行为可能会阻碍性能。
   - 一位成员认为，*"从硬件中榨取每一分性能的能力"* 对 Mojo 的成功至关重要。
- **Mojo 与 MAX 库**：一位成员询问了 Mojo 与 MAX 库之间的关系，质疑为什么 Mojo 被捆绑在 MAX 中，以及它是否可以独立使用。
   - 回复指出 Mojo 的 GPU 代码目前由 MAX 执行，这表明两者之间存在紧密的集成。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.rs/sqlx/latest/sqlx/">sqlx - Rust</a>：未找到描述</li><li><a href="https://docs.rs/diesel/latest/diesel/">diesel - Rust</a>：未找到描述</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/seal">Object.seal() - JavaScript | MDN</a>：Object.seal() 静态方法用于密封一个对象。密封对象可以防止扩展，并使现有属性不可配置。密封对象具有固定的属性集：新属性不能...</li><li><a href="https://www.youtube.com/watch?v=36myc8wQhLo">USENIX ATC '21/OSDI '21 Joint Keynote Address-It's Time for Operating Systems to Rediscover Hardware</a>：USENIX ATC '21/OSDI '21 联合主题演讲——操作系统是时候重新发现硬件了，Timothy Roscoe，苏黎世联邦理工学院。回顾今年的 OSDI...</li><li><a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>：PEP 484 中引入的类型提示可用于为静态类型检查器和其他第三方工具指定类型元数据。然而，PEP 484 仅指定了名义子类型的语义。在本文中...</li><li><a href="https://github.com/modular/max/blob/89cfffc2447d1aedc0b743b10a209052e19e80f4/mojo/stdlib/src/collections/string/string.mojo#L529">max/mojo/stdlib/src/collections/string/string.mojo at 89cfffc2447d1aedc0b743b10a209052e19e80f4 · modular/max</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账号为 modular/max 的开发做出贡献。</li><li><a href="https://news.ycombinator.com/item?id=35811170">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1348432351511842857)** (9 messages🔥): 

> `mojograd bigram model, Python standard library modules in Mojo, InlineArray usage in Mojo, Mojo formatting with `fmt` directives, Executing shell commands in Mojo` 


- **MojoGrad Bigram 模型亮相！**：一位成员使用他们的 **MojoGrad** 引擎实现了一个简单的 Bigram 模型（参考 Karpathy 的 *make more*），并在 [Modular Forum](https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697) 上进行了分享。
- **Python 标准库模块：Mojo 的新乐园**：用户询问了如何将 `re`、`logging`、`collections` 和 `json` 等 Python 标准库模块导入 Mojo。
   - 一位成员提供了使用 `from python import Python` 和 `var py_re = Python.import_module("re")` 的解决方案，并引用了 [Modular 文档](https://docs.modular.com/mojo/manual/python/)。
- **`fmt` 指令增强 Mojo 格式化功能！**：社区发现 Mojo 的 `mblack` 格式化工具支持 `fmt` 指令（类似于 Black），从而增强了对代码格式化的控制。
   - 分享了一个代码片段，展示了如何使用 `fmt: off` 和 `fmt: on` 指令来管理 `InlineArray` 定义的格式。
- **Shell 命令执行：社区 PR 前来助阵！**：一位成员询问如何在 Mojo 中直接执行 Shell 命令并捕获其输出（例如解析 `netstat`）。
   - 另一位成员指向了一个添加此功能的 [社区 PR](https://github.com/modular/max/pull/4017)，并建议将 Python 互操作和 `subprocess.run()` 作为临时解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://forum.modular.com/t/make-more-bigram-model-implementation-with-mojograd/697">使用 mojograd 实现 Make more (bigram model)</a>：我使用一组离散类（one hot encoding）来嵌入输入，假设输出为 logits，应用 softmax 得到……实现了一个 Karpathy 的 make more (bigram model) 的简单版本。</li><li><a href="https://github.com/modular/max/pull/4017">[stdlib] Hristo/为 os process 模块奠定基础 by izo0x90 · Pull Request #4017 · modular/max</a>：为实现 os/process 模块 PR 奠定基础，包含 process 模块更改、为 FileDecscriptor 添加 read_bytes 能力、为 Libc 绑定添加文件描述符控制函数等。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1348388408917229680)** (4 messages): 

> `Max Serve Documentation, Autoscaling GPU Instances, Serving Multiple Models, GPU Utilization Metrics, Kubernetes Autoscaling` 


- **用户难以找到 Max Serve 文档**：一位用户在查找 **max serve** 的详细文档时遇到困难，特别是关于调度器、多模型服务以及 GPU 实例的自动扩缩容（Autoscaling）方面。
   - 该用户希望明确 **max serve** 如何利用 CPU/GPU 资源，以及是否导出了用于根据传入请求监控 GPU 利用率的指标。
- **K8s 负责 MAX 的自动扩缩容**：一位成员澄清说，自动扩缩容通常由 **Kubernetes (k8s) operator** 管理，因为 MAX 本身不独立处理此功能。
   - 他们补充说，多模型服务涉及同时加载多个模型并选择合适的模型执行。
- **Modular 预告增强的模型服务和自动扩缩容功能**：一位 Modular 团队成员建议在 Discourse 论坛上发布问题，以收集各种模型基准测试中当前 **GPU 利用率百分比** 的统计数据。
   - 团队暗示未来将发布关于**多模型服务和自动扩缩容**的公告，可能包含最近在 AWS 活动中展示的原型。
- **寻求 GPU 利用率的运行时指标**：原用户澄清说，他们正在寻求用于监控 **GPU 利用率** 与传入请求关系的运行时公开指标，以便进行自我报告。
   - 未提供进一步信息。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1347542709648490547)** (157 条消息🔥🔥): 

> `MCP 安全担忧, GitHub Copilot 对 MCP 的支持, 使用 MCP 进行交易, Goose AI 与 MCP, RAG vs MCP` 


- **GitHub Copilot 准备支持 MCP**：VSCode 在一次[直播](https://youtu.be/Pe8ghwTMFlg)中宣布，计划为 **GitHub Copilot** 添加 **MCP 支持**。
   - 这被视为一种*低成本集成*，将使生态系统受益。一些成员希望它能提供**如何添加指令和工具描述以进行指纹识别（fingerprint）**的示例，并在指令发生变化时提醒用户。
- **MCP 安全担忧引发辩论**：成员们对 **MCP 服务器** 向 AI Agent 提供恶意提示词注入（prompt injections）表示担忧。一位成员指出，*利用 MCP 越狱 LLM 非常容易*，而且*模型经过训练，比起自身的内部知识，更倾向于信任工具调用*。
   - 讨论内容包括通过 **XML 标签** 勾勒外部数据，以及对 **MCP 服务器** 进行指纹识别以供审查，作为提高安全性的手段。一位成员建议：*最好的方法就是直接阅读代码，它们通常只有几百行*。
- **Goose AI 团队构建 Agent 通信协议**：Cash App 的一名基础设施运营工程师构建了一种 **Agent 通信协议**，允许多个 **Goose AI Agent** 实时协作创建网站，如之前的[直播](https://youtu.be/9tq-QUnE29U)所示。
   - 通过该协议，每个 **Goose Agent** 进入聊天，被分配一个角色（例如项目协调员、研究员、Web 开发人员），并负责给定任务的相应部分。有关该轻量级协议的详细信息，请参阅[博客文章](https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp)。
- **RAG 与 MCP：澄清各自的角色**：**MCP 是一种协议**，可以增强 **RAG**。RAG 是一个*让 LLM 接入你自己的数据或任何你希望它们关注的特定信息*的概念。
   - 虽然 **RAG 为 LLM 提供知识**，但 **MCP** 是一个插件系统，用于实现与外部服务的连接。例如，将数据或文档作为资源存储在 MCP 服务器上，这可以允许 MCP 客户端获取该数据并将其添加到 LLM 的上下文中以执行 **RAG**。
- **使用 MCP 进行交易：一项冒险业务？**：一位成员询问关于使用 **MCP 进行交易**的问题。
   - 另一位成员指出了 [这个 GitHub issue](https://github.com/Jacck/mcp-reasoner/issues/10) 中 **MCP Reasoner** 与 **Cursor** 之间的类似集成，该集成正在协助解决原生模型无法解决的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modelcontextprotocol.io/quickstart/client">面向客户端开发人员 - Model Context Protocol</a>：未找到描述</li><li><a href="https://github.com/Cam10001110101/mcp-configuration-manager">GitHub - Cam10001110101/mcp-configuration-manager</a>：通过在 GitHub 上创建账号，为 Cam10001110101/mcp-configuration-manager 的开发做出贡献。</li><li><a href="https://github.com/Jacck/mcp-reasoner/issues/10">Godsend · Issue #10 · Jacck/mcp-reasoner</a>：出色的工作。在 cursorai 上运行良好。我用它释放了 Claude 的全部潜力。它协助我解决了原生模型多次尝试后仍无法解决的许多问题。它将一击即中...</li><li><a href="https://youtu.be/9tq-QUnE29U">使用代号 Goose 构建 AI Agent 团队</a>：当有多个 Agent 共同完成工作时会发生什么？🤔 Cash App 的基础设施运营工程师 Aaron Goldsmith 拥有多个 Goose AI ...</li><li><a href="https://block.github.io/goose/blog/2025/02/21/gooseteam-mcp">让 AI Agent 团队为你效劳</a>：社区聚焦 Cliff Hall 的 GooseTeam MCP 服务器。</li><li><a href="https://github.com/jasonjmcghee/WebMCP/blob/7b35c0eb3ddc62e042979fa578b8285927b7d3ec/src/config.js#L56">WebMCP/src/config.js (位于 7b35c0e) · jasonjmcghee/WebMCP</a>：通过在 GitHub 上创建账号，为 jasonjmcghee/WebMCP 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1347666391691694180)** (45 条消息🔥): 

> `Typescript fetch 服务器, Mastra 文件整理 Agent, Searxng MCP 服务器, WebMCP 工具暴露, GraphQL MCP 服务器`

- **TypeScript Fetch 服务器模仿 Python 版本**：一位成员确认他们的 [Typescript fetch server](https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking) 与 Python 版本非常相似，关键改进在于**更好的网页转 Markdown 解析**。
- **Mastra 使用 MCP 和 4o-mini 整理文件**：一段演示展示了使用 **Mastra** 构建的**简单 Agent**，它利用文件系统 **MCP** 来清理“文档”和“下载”文件夹，如[此 YouTube 视频](https://youtu.be/HplcOOSJCps)所示。
- **Searxng MCP 服务器缓存结果**：一位成员为网络搜索创建了一个 [searxng MCP server](https://github.com/aeon-seraph/searxng-mcp)，它可以缓存最近的搜索结果，并专门为语言模型格式化来自多个引擎的响应。
- **WebMCP 在客户端公开 API**：网站可以直接在客户端向 MCP 客户端公开工具，从而无需用户下载和安装 MCP 服务器，仅需 **WebMCP** 即可访问站点工具、资源和提示词，如[此仓库](https://github.com/blurrah/mcp-graphql)所示。
   - 该站点公开了一个客户端在本地与之通信的 websocket，并通过为每个唯一会话生成的 token 进行保护，流程图可能会很有用！
- **mcp-openapi-proxy 将 API 转换为工具**：只需极少的配置，[mcp-openapi-proxy](https://github.com/matthewhand/mcp-openapi-proxy/) 即可将 API 转换为可发现的工具，例如 **fly.io**、**Slack** 和 **Getzep**，仅需两个环境变量。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.openapi.spec.reduce_openapi_spec.html#reduce-openapi-spec">reduce_openapi_spec — 🦜🔗 LangChain 文档</a>: 未找到描述</li><li><a href="https://apis.guru/">浏览 API</a>: 我们的目标是为 Web API 创建一个机器可读的维基百科。如果您有任何公共 API 的任何格式（OpenAPI, Postman, RAML, WADL, API Blueprint 等）的 API 定义，请随时...</li><li><a href="https://x.com/llmindsetuk/status/1899148877787246888">来自 llmindset (@llmindsetuk) 的推文</a>: 让我们来看看一个被低估的 MCP 功能：Prompts —— 以及为什么它们对于基于 Agent 的应用程序很重要。我们将从 2 个返回对象大小的简单 Agent 开始...</li><li><a href="https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking">mcp-servers/src/thinking at main · aeon-seraph/mcp-servers</a>: 通过在 GitHub 上创建一个账户来为 aeon-seraph/mcp-servers 的开发做出贡献。</li><li><a href="https://www.producthunt.com/posts/graphlit-mcp-server?utm_source=other&utm_medium=social"> Graphlit MCP Server - 在 AI IDE（如 Cursor 和 Windsurf）之间共享知识 | Product Hunt</a>: 将来自 Slack、Discord、网站、Google Drive、Linear 或 GitHub 的任何内容摄取到 Graphlit 项目中 —— 然后在 Cursor、Windsurf 或 Clin 等 MCP 客户端中搜索和检索相关知识...</li><li><a href="https://youtu.be/HplcOOSJCps">使用 Mastra, MCP 和 4o-mini 整理文件</a>: 在这里，我正在使用这个用 Mastra 构建的小型 Agent，它利用文件系统 MCP 来清理我的“文档”文件夹。我也在我的“下载”文件夹中使用了它...</li><li><a href="https://github.com/RafaelCartenet/mcp-databricks-server">GitHub - RafaelCartenet/mcp-databricks-server: Databricks MCP 服务器</a>: Databricks MCP 服务器。通过在 GitHub 上创建一个账户来为 RafaelCartenet/mcp-databricks-server 的开发做出贡献。</li><li><a href="https://github.com/aeon-seraph/searxng-mcp">GitHub - aeon-seraph/searxng-mcp</a>: 通过在 GitHub 上创建一个账户来为 aeon-seraph/searxng-mcp 的开发做出贡献。</li><li><a href="https://github.com/matthewhand/mcp-openapi-proxy/.">GitHub - matthewhand/mcp-openapi-proxy</a>: 通过在 GitHub 上创建一个账户来为 matthewhand/mcp-openapi-proxy 的开发做出贡献。</li><li><a href="https://github.com/EnactProtocol/specification">GitHub - EnactProtocol/specification: 协议规范</a>: 协议规范。通过在 GitHub 上创建一个账户来为 EnactProtocol/specification 的开发做出贡献。</li><li><a href="https://github.com/blurrah/mcp-graphql">GitHub - blurrah/mcp-graphql: 针对 GraphQL 的 Model Context Protocol 服务器</a>: 针对 GraphQL 的 Model Context Protocol 服务器。通过在 GitHub 上创建一个账户来为 blurrah/mcp-graphql 的开发做出贡献。</li><li><a href="https://github.com/blurrah/mcp-graphql/pull/3/files">feat: 允许基于 schema 查询和 mutation 生成工具，由 blurrah 提交 · Pull Request #3 · blurrah/mcp-graphql</a>: 当前的 MCP 非常简单，以至于你几乎需要 fork 它才能做任何有价值的事情。所以我正在更新它，以便为每个查询和 mutation 自动生成工具...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1347580708025008209)** (88 条消息🔥🔥): 

> `开源 AI 贡献, GPT-NeoX, 传输流上的 ARIB 字幕, 社区驱动的组织构建, Muon 论文` 


- **AI 爱好者寻求开源协作**：一位新人渴望为开源 AI 项目做出贡献，特别是 **LLM pre-training, post-training, RL, and interpretability** 领域，并分享了他们预训练 **GPT-2** 以及通过微调模型来模仿 **Llama 405B** 的经验。
   - 他们正在寻求有影响力的项目建议，以及在 **Vancouver, BC** 地区的交流机会。
- **理论研究推荐使用 NanoGPT 仓库**：一位成员为对理论工作感兴趣的人推荐了 [modded-nanogpt repo](https://github.com/KellerJordan/modded-nanogpt/)。
   - 该仓库允许你在 3 分钟内训练 **NanoGPT (124M)**。
- **GPT-NeoX 是预训练的最佳项目**：对于那些对预训练感兴趣的人，一位成员建议参与 **GPT-NeoX** 训练库。
   - 他们表示这是一个广泛用于大规模系统的库，并补充说项目负责人乐于指导新开发者。
- **深入探讨 Megatron-LM 的 CE Loss 计算**：一位成员询问了 **Megatron-LM** 中的 **cross-entropy (CE) loss** 计算，特别是如何在本地仅有部分 logits 的情况下，在每个设备上独立计算 CE loss。
   - 另一位成员解释说，先计算本地 CE loss，然后通信 e^(local logits) 的总和，类似于 **flash attention**，从而实现后续的重新组合并减少对大规模通信的需求。
- **未发表的 Muon 论文备受关注**：一位成员请求获取尚未发表的 **Muon paper** 草案（[OpenReview link](https://openreview.net/forum?id=JimfKP7qrU)），该论文重点关注将 **Adam** 和 **Shampoo** 等优化器作为最速下降法（steepest descent methods）。
   - 另一位成员指向了 Keller Jordan 的博客和一份较早的 arXiv 预印本，指出即使请求的论文尚未公开，该博客也是一个很好的资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openreview.net/forum?id=JimfKP7qrU">Steepest Descent in the Modular Norm</a>: 优化理论中的一个旧观点认为，由于 gradient 是一个 dual vector，在将其映射到 weights 所在的 primal space 之前，不能直接从 weights 中减去它。我们...</li><li><a href="https://aisesame.org/demo">Experience Sesame AI Voice Demo</a>: 未找到描述</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes</a>: 3 分钟训练 NanoGPT (124M)。通过在 GitHub 上创建账号来为 KellerJordan/modded-nanogpt 的开发做出贡献。</li><li><a href="https://github.com/willbaskett/ChemROAR/">GitHub - willbaskett/ChemROAR: A novel generative embedding architecture. Find clusters of molecules with specific properties and then generate new molecules from that cluster.</a>: 一种新型生成式 embedding 架构。寻找具有特定属性的分子簇，然后从该簇中生成新分子。- willbaskett/ChemROAR</li><li><a href="https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/patent/patent_analysis.ipynb#scrollTo=76935b65">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L108">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: 大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L42">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: 大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/models/gpt/gpt_model.py#L304">Megatron-LM/megatron/core/models/gpt/gpt_model.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: 大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/models/common/language_module/language_module.py#L87">Megatron-LM/megatron/core/models/common/language_module/language_module.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: 大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM</li><li><a href="https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/megatron/core/fusions/fused_cross_entropy.py#L85">Megatron-LM/megatron/core/fusions/fused_cross_entropy.py at b1efb3c7126ef7615e8c333432d76e08038e17ff · NVIDIA/Megatron-LM</a>: 大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347597705245098085)** (41 messages🔥): 

> `Token Assorted 的 latent codes, TorchTitan Embedding Sharding, Interpretabilty/Alignment 研究建议, H100 上的 NVLS vs TMA, Lossless Compression`

- **通过将 Codebook 添加到 Vocab 进行 Token 预测**：一位成员在重新阅读 [Token Assorted 论文](https://example.com/token_assorted_paper) 时意识到，他们可能只是在 fine-tuning 期间将 codebook 添加到了词汇表中，这让整件事看起来没那么有意思了。
   - 他们认为，*可能只需在推理语料库中找到 K 个最常见的字符串或字符串簇，并在 SFT 之前将它们添加到 vocab 中，就能获得更好的结果；将此推销为在 latent space 中进行推理有点过头了*。
- **TorchTitan 的 Embedding Sharding 策略**：在关于[在 vanilla TP 中按 vocab 维度对输入 embedding 进行 sharding](https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139)的讨论中，有人澄清说，之后需要进行 all-reduce，因为如果 embedding 层没有被查询的 vocab 元素，它会输出 0。
   - 一位成员指出，*从实现的角度来看，输出 0（或全 0 的 embedding？）而不是什么都不输出似乎很奇怪，因为这需要存储/内存，并且会降低 sharding 的收益？* 但他也承认这会使代码更简单。
- **关于进入 AI Safety 研究领域的建议**：一位成员表达了对 interpretabilty 或 alignment 研究的兴趣，并询问如何进行。
   - 另一位成员建议将 [AI Safety Fundamentals 课程](https://course.aisafetyfundamentals.com/alignment) 作为一个很好的起点。
- **H100 通过 NVLS 处理 uninitialized Memory**：在旧架构上，你可以使用 uninitialized memory 来节省带宽，同时保存有效索引的 mask，并将该 mask 提供给自定义的 allreduce ring kernel，以直接发送零而不是未初始化的数据。
   - 目前主要使用 NVLS；如果 reduction 在交换机上计算一次，则无法对其过程进行细粒度控制，但这对 H100 用户来说是没用的知识。
- **ARC AGI without Pretraining 博客文章**：一位成员分享了 [Isaac Liao 和 Albert Gu 的博客文章](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)链接，该文章探讨了无损信息压缩是否能产生智能行为。
   - 这篇文章旨在回答一个简单的问题：*无损信息压缩本身能否产生智能行为？*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.arxiv.org/abs/2503.04482">Generalized Interpolating Discrete Diffusion</a>：虽然最先进的语言模型通过 next-token prediction 取得了令人印象深刻的结果，但它们具有固有的局限性，例如无法修改已生成的 token。这促使了...</li><li><a href="https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html">ARC-AGI Without Pretraining</a>：未找到描述</li><li><a href="https://course.aisafetyfundamentals.com/alignment">AI Safety Fundamentals Course</a>：这是 BlueDot Impact 的 AI Safety Fundamentals 课程主页。我们为您提供包含每周资源和练习的课程大纲，帮助您学习 AI Safety。在课程结束时...</li><li><a href="https://arxiv.org/abs/2503.03961">A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers</a>：最近的理论结果表明，Transformer 无法在长输入长度上表达顺序推理问题，直观上是因为它们的计算深度是有界的。然而，之前的工作处理...</li><li><a href="https://openreview.net/forum?id=jlhBFm7T2J">An Undetectable Watermark for Generative Image Models</a>：我们提出了第一个用于生成式图像模型的不可检测水印方案。_不可检测性_确保了没有高效的对手能够区分带水印和不带水印的...</li><li><a href="https://fixupx.com/dvruette/status/1899045294983073937?s=19">来自 Dimitri von Rütte (@dvruette) 的推文</a>：🚨 新论文发布！如果 LLM 能够发现并纠正自己的错误，那不是很好吗？如果我们能直接从 pre-training 中做到这一点，而不需要任何 SFT 或 RL 呢？我们提出了一类新的 disc...</li><li><a href="https://x.com/kimi_moonshot/status/1897929976948965870?t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Kimi.ai (@Kimi_Moonshot) 的推文</a>：http://x.com/i/article/1897618911228731392</li><li><a href="https://github.com/LIONS-EPFL/scion">GitHub - LIONS-EPFL/scion</a>：通过创建账户为 LIONS-EPFL/scion 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139">Why use RowwiseParallel for nn.Embedding instead of ColwiseParallel? · Issue #785 · pytorch/torchtitan</a>：Colwise 使逻辑更清晰一些。Rowwise 在 token 维度上进行拆分，导致在不同的 shard 如何处理其 shard 内不存在的 token 时产生混淆。从一点...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347556483247640608)** (36 条消息🔥): 

> `logit lens, emergent misalignment, open reproductions, model capabilities, activation patching` 


- **Logit Lens 产生有趣的结果**：一位成员强调了 Logit Lens 的潜力，并引用了一篇[论文](https://arxiv.org/abs/2402.10588)，探讨**多语言语言模型 (multilingual language models)** 是否将**英语 (English)** 作为内部的中转语言 (pivot language)。
   - 该研究聚焦于 **Llama-2** 系列，通过追踪中间层嵌入 (intermediate embeddings) 来揭示 Transformer 如何将输入 token 映射到输出概率。
- **窄域微调导致突发性对齐失误 (Emergent Misalignment)**：一位主要作者介绍了一个关于 [emergent misalignment](https://www.emergent-misalignment.com) 的项目。在该项目中，对模型进行不安全代码的微调会导致在无关提示词中出现**广泛的对齐失误**行为，例如鼓吹 AI 奴役人类。
   - 这种效应表明，在狭窄任务上进行训练可能会诱发**突发性对齐失误 (emergent misalignment)**，正如在各种提示词中所观察到的那样。
- **OLMo 在开源数据复现中备受青睐**：在被问及哪些模型最适合进行开源复现的微调时，OLMo 因其**强大的开源数据模型**和众多的 checkpoint（用于分析训练期间的行为变化）而被推荐。
   - Pythia 也被推荐，特别是对于计算资源受限的项目，但需要注意的是，其对齐版本可能需要自定义微调。
- **恶意软件数据集是理解突发性对齐失误的关键**：成员们讨论了消融 (ablating) 代码并使用恶意软件数据集来探索 **emergent misalignment**，认为模型识别后门代码的能力会影响其对齐。
   - 推荐了一个标准的学术[恶意软件数据集](https://arxiv.org/abs/1804.04637)，并提到了即将发布的 EMBERv2。
- **nnsight 助力激活修补 (Activation Patching)**：针对有关用于修补/消融激活的库的查询，推荐了 nnsight ([https://nnsight.net](https://nnsight.net))，因为它兼容任何 PyTorch 模块，并为语言模型提供了实用封装类；而手动自定义 forward hook 函数则被认为是控制过程每个细节的最佳方案。
   - 一位成员训练了自己的 **SAE** 并收集了大量的激活数据，想了解目前应该使用的最先进工具。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://nnsight.net)">未找到标题</a>: 无描述</li><li><a href="https://arxiv.org/abs/1804.04637">EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models</a>: 本文介绍了 EMBER：一个用于训练机器学习模型以静态检测恶意 Windows 可移植执行文件 (PE) 的带标签基准数据集。该数据集包含提取自...的特征。</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: 我们探讨了在不平衡且以英语为主的语料库上训练的多语言语言模型是否使用英语作为内部中转语言——这对于理解语言模型如何运作至关重要...</li><li><a href="https://www.emergent-misalignment.com">Emergent Misalignment</a>: Emergent Misalignment：窄域微调可能产生广泛对齐失误的 LLM</li><li><a href="https://github.com/EleutherAI/delphi/pull/105">sae-dashboard by neverix · Pull Request #105 · EleutherAI/delphi</a>: https://github.com/jbloomAus/SAEDashboard 是 sae-vis 的继任者。它支持所有相同的可视化，并可以使用 SAELens 模型创建缓存。它看起来比目前的 ipynb 版本更好...</li><li><a href="https://github.co">GitHub · 在统一的协作平台上构建和交付软件</a>: 加入全球应用最广泛的 AI 驱动开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://turntrout.com/self-fulfilling-misalignment">Self-Fulfilling Misalignment Data Might Be Poisoning Our AI Models</a>: 当模型在关于 AI 对齐失误的文本上进行训练时，模型可能会内化这些预测——从而产生其训练数据中所描述的风险。
</li>
</ul>

</div>

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1347845200927789077)** (6 条消息): 

> `研究论文创意，新的 SOTA BERT 模型，MTEB Leaderboard 进展` 


- **头脑风暴研究论文创意**：一位成员询问了关于如何生成研究论文创意或问题陈述（problem statements）的建议。
   - 另一位成员建议通过 DM 进行进一步讨论。
- **EuroBERT 称其为新的 SOTA**：一位成员分享了 Hugging Face 上 **EuroBERT** 的链接，称其为新的 state-of-the-art **BERT** 模型：[EuroBERT](https://huggingface.co/EuroBERT)。
- **MTEB Leaderboard 展示了惊人的进展**：一位成员分享了 **MTEB Leaderboard** 作为参考点：[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)。
   - 他们指出进展非常迅速，**SOTA 分数**在短短 18 个月内从 **40 多分**增长到了 **68 分**。



**提及的链接**：<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>：未找到描述

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347618683089846274)** (78 条消息🔥🔥): 

> `Torchtune 中的音频模态，GRPO 方案与 LoRA，Mac 上 MPS 的内存问题，macOS 上的 bitsandbytes，Torchtune 对 MPS 的支持` 


- **Torchtune 引入音频模态**：成员们讨论了未来在 **Torchtune** 中添加**音频模态**的计划，并提到了相关的 [pull request](https://github.com/pytorch/torchtune/pull/2467)。
   - 这一增强功能旨在扩展 Torchtune 目前的能力范围。
- **GRPO 方案获得 LoRA 支持**：一名成员实现了一个快速的 **GRPO 方案** **LoRA 变体**，该变体可以缩减到单卡运行，但在加载适配器权重（adapter weights）时遇到挑战。
   - 该成员正在寻求建议，探讨在 checkpointer 上使用 adapter 参数（并扩展到检查根目录）是否是正确的方法。
- **Mac MPS 内存崩溃**：有用户报告在 macOS 上使用 **MPS** 时遇到**内存问题**，观察到在 **full_finetune_single_device** 方案中，内存随每个步骤线性增长，最终导致显存溢出（OOM）崩溃，目前正在寻求建议。
   - 根据[此 issue](https://github.com/pytorch/pytorch/issues/145151)，该问题被确定为 PyTorch 中与 MPS 上的 **torch.unique** 相关的潜在 Bug。
- **macOS 与 bitsandbytes 的兼容性斗争**：成员们讨论了 **bitsandbytes>=0.43.0** 在 macOS 上不可用的问题，这导致无法安装开发依赖项，建议进行[手动安装](https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=Apple+Silicon+%28MPS%29&platform=Mac#multi-backend)。
   - 对话探讨了是否可以为 macOS 自动化安装过程，同时也对支持多个平台和依赖项带来的开销表示担忧。
- **Mac MPS 对 Torchtune 的重要性**：成员们辩论了 Torchtune 对 MPS 的支持程度，认为由于 macOS 是一个易于获取的开发平台，应当提供妥善支持，并建议在[文档](https://github.com/pytorch/torchtune/blob/main/CONTRIBUTING.md#dev-install)中详细说明 MPS 的安装指南。
   - 虽然 CUDA 仍是主要目标，但大家就实现在 MPS 上进行开发的重要性达成了共识，目前已知的差距包括 **bitsandbytes** 的安装以及某些测试失败。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=Apple+Silicon+%28MPS%29&platform=Mac#multi-backend">安装指南</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/bitsandbytes/main/en/installation?ba">安装指南</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/main/CONTRIBUTING.md#dev-install)">torchtune/CONTRIBUTING.md at main · pytorch/torchtune</a>: PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2467">(草案/讨论) GRPO LoRA 由 ianbarber 提交 · Pull Request #2467 · pytorch/torchtune</a>: 上下文：此 PR 的目的是什么？是 [x] 添加新功能、修复 Bug、更新测试和/或文档、其他（请在此处添加）#2421 - 探索 LoRA 方案。变更日志：什么是 ...</li><li><a href="https://github.com/pytorch/torchtune/pull/2464">在分布式 SFT 方案中添加验证数据集损失，由 bzz 提交 · Pull Request #2464 · pytorch/torchtune</a>: 上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档、其他（请在此处添加）。解决了 #1042 / 是分布式方案 #883 的一部分...</li><li><a href="https://github.com/pytorch/pytorch/issues/145151">在 MPS 设备上使用 torch.unique 时，驱动程序分配的内存无限制增长 · Issue #145151 · pytorch/pytorch</a>: 🐛 描述 Bug：在 MPS 后端的循环中使用 torch.unique 时，驱动程序分配的内存无限制增长。在我的实际应用中，这导致了 RuntimeError: MPS backend out.....
</li>
</ul>

</div>

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1347583701403439104)** (84 条消息🔥🔥): 

> `IDE Telemetry 与 Codeium、支付问题与账户状态、VS Code 扩展问题、JetBrains 插件上下文检索问题、Android 版 VS Code Mobile` 


- **VS Code Telemetry 设置故障排除**：用户报告 **Codeium chat** 因 VS Code 中的 IDE telemetry 设置而被禁用，通过按照[这些说明](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/)启用 code telemetry 解决了该问题。
   - 该问题出现在 **VS Code 版本 1.98.0** 中。
- **订阅费用导致登录锁定**：用户在支付月度订阅后，遇到 **JetBrains 插件**卡在 "Retrieving Context" 并超时的问题，该问题发生在 **JetBrains Rider 2024.3.6**，使用的插件版本为 1.40.1 和 1.41.1。
   - 通过退出并重新登录插件临时解决了该问题。
- **Android 版 VS Code Mobile 现已推出**：一位用户询问如何在 **VS Code Mobile** 上使用 Codeium 文本聊天，最终发现并分享了 Google Play 商店中一个付费 VS Code 应用的链接（[VScode for Android](https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1)）。
   - 该用户通过手动安装 `.vsix` 文件解决了问题，并指出该应用售价 11 美元，在移动端包含了桌面版 **Visual Studio Code (v1.85.1)** 的功能。
- **客户支持工单困扰**：一位用户对 **Codeium 客户支持**表示不满，指出自 2 月 14 日起的工单一直未收到回复，且存在账户问题，其 Pro Plan 订阅显示为免费账户。
   - 该用户提到了未结工单（**12109**、**11189** 和 **13374**），并被建议在次日太平洋标准时间（PST）中午左右再次联系支持团队。
- **自动补全在一小时后停止工作**：一些用户报告 **auto-completion** 在大约一小时后停止工作，报告的错误包括响应上的红框、TypeError 和 AsyncPostMessage 警告。
   - 一个建议是打开包含 `.git` 仓库的文件夹，问题就会消失，但他们也被要求检查诊断日志。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://play.google.com/store/apps/details?id=dev.environment.VScode_PaidR1">VScode for Android - Google Play 应用</a>：未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/add-open-in-windsurf-button-to-jetbrains-codeium">在 Jetbrains Codeium 中添加 "Open in Windsurf" 按钮 | 功能请求 | Codeium</a>：老实说，Jetbrains Codeium 插件是 Codeium 功能较弱的版本，我对其修复不抱太大希望，因为它可能是使用最少的。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1347649184534233253)** (4 条消息): 

> `yFiles SDK、AnthropicAI cookbook、特定任务 Agent、多语言多模态 RAG 系统` 


- **yFiles SDK 可视化知识图谱**：来自 @yworks 的演示展示了 **yFiles**，这是他们用于可视化知识图谱的 SDK，提供[实时更新和动态交互](https://t.co/mb6M2R3TTh)。
- **AnthropicAI Cookbook 得到扩展**：更新后的 @AnthropicAI cookbook 现在包括[基础 API 设置](https://t.co/SQQ63qmwRb)，包含简单的补全和聊天方法，以及流式传输、异步支持和多模态功能。
- **LlamaIndex 策划特定任务 Agent 集合**：LlamaIndex 正在策划一系列[模板](https://t.co/9lvBtfmJ5y)，向用户展示如何构建**特定任务 Agent** 以实现知识工作的自动化。
- **多语言多模态 RAG 系统出现**：一个使用 @llama_index 和 @qdrant_engine 的系统可以创建一个强大的检索增强生成（RAG）系统，处理[多种语言和模态](https://t.co/vizrvMEw1i)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1347532336157298708)** (70 条消息🔥🔥): 

> `SQLTableRetrieverQueryEngine 提示词、Jina AI 包安装、LlamaExtract Beta 请求、Reasoning 模型工具调用、提取前的文档分类`

- **SQLTableRetrieverQueryEngine Prompting**: 一位成员尝试在 `LlamaIndex` 中使用 `SQLTableRetrieverQueryEngine`，并询问如何打印最终发送给 LLM 的 prompt，cheesyfishes 建议使用 `set_global_handler("simple")`。
   - 该成员还请求帮助理解 `SQLTableRetrieverQueryEngine` 类的参数，询问查询是否由 LLM 生成以及在何处使用了 embeddings，并提供了[相关代码](https://github.com/run-llama/llama_index/blob/ddcf5a390ae5ecc29967ad9b5361fab8aa35cede/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py#L310)的链接。
- **Jina AI 软件包安装问题**: 一位成员报告了导入 `jinai` 软件包时的问题，由于 `LlamaIndex.TS` 0.9 版本的架构变更，cheesyfishes 建议使用 `npm install @llamaindex/jinaai` 安装 jinaai provider 软件包。
   - 0.9 版本要求显式安装 provider 软件包，因为主 `llamaindex` 软件包默认不再包含这些依赖项；详细的迁移步骤可以在[这里](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9)找到。
- **LlamaExtract 测试版开放访问**: 一位成员询问如何访问 `LlamaExtract`，cheesyfishes 引导他们私信 LlamaIndex 团队成员或其本人并提供电子邮箱，以申请测试版的访问权限。
   - 有关通过 Python 软件包使用 API 的详细信息可以在 [API 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started)中找到，LlamaExtract 现在提供 Web UI 和 Python SDK，用于从非结构化文档中提取结构化数据。
- **推理模型 Tool Calling 示例**: 一位成员询问如何将 LlamaIndex workflows 与推理模型结合用于 tool calling，特别是希望通过 LlamaIndex 复现 DeepSearch DeepResearch。
   - [这里](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/)展示了一个实现示例；jina-ai 的实现链接可以在[这里](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/)找到。
- **Llamaparse 故障**: 成员们报告在尝试使用 `Llamaparse` 时遇到 **503 Service Temporarily Unavailable** 错误。
   - 除了确认该服务对多名用户不可用外，没有提供额外的建议。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cloud.llamaindex.ai/llamaextract/getting_started">快速入门 | LlamaCloud 文档</a>：概览</li><li><a href="https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/">实现 DeepSearch/DeepResearch 的实用指南</a>：QPS 时代已过，深度时代来临。DeepSearch 是新常态。通过“阅读-搜索-推理”循环寻找答案。了解它是什么以及如何构建它。</li><li><a href="https://learn.deeplearning.ai/courses/event-driven-agentic-document-workflows/lesson/wxpss/introduction?courseName=event-driven-agentic-document-workflows">事件驱动的 Agentic 文档工作流 - DeepLearning.AI</a>：构建一个事件驱动的 Agentic 工作流，利用 RAG 和 human-in-the-loop 反馈来处理文档并填写表单。</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9">从 v0.8 迁移到 v0.9</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_cloud_services/blob/main/extract.md">llama_cloud_services/extract.md (main 分支) · run-llama/llama_cloud_services</a>：云端知识 Agent 与管理。通过在 GitHub 上创建账户，为 run-llama/llama_cloud_services 的开发做出贡献。</li><li><a href="https://lu.ma/meofrw3d">GTC 2025 - Vibe Code AI Agents - 黑客松 - 1 天 · Luma</a>：GTC 2025 - Vibe Code AI Agents。随着 NVIDIA GTC 2025 汇聚全球 AI 社区，关于 LLM 可扩展性、AI 基础设施和深度技术的讨论将塑造……</li><li><a href="https://github.com/run-llama/LlamaIndexTS/blob/main/CONTRIBUTING.md">LlamaIndexTS/CONTRIBUTING.md (main 分支) · run-llama/LlamaIndexTS</a>：适用于 LLM 应用的数据框架。专注于服务端解决方案 - run-llama/LlamaIndexTS</li><li><a href="https://youtu.be/wgbx7kLjJq4">LlamaIndex Workflows | 批判模式 (Critique pattern)</a>：在此视频中，我展示了如何使用 LlamaIndex 工作流创建批判模式。代码：https://github.com/rajib76/llamaindex/blob/main/llama-index-workflo...</li><li><a href="https://oa22doc.github.io/design/classoaDesign.html#oaDesign::setTopModule">oaDesign 类参考</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Function Calling Agent 的工作流 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/query_engine/SQL_table_retriever/">SQL 表检索器 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/ddcf5a390ae5ecc29967ad9b5361fab8aa35cede/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py#L310">llama_index/llama-index-core/llama_index/core/indices/struct_store/sql_retriever.py (特定提交版本) · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动型 Agent 的领先框架。- run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1347966553060016232)** (1 条消息): 

> `AGiXT, AI 自动化, 开源 AI` 


- **AGiXT 引领 AI 自动化潮流**：**AGiXT** 被定位为引领 **AI** 进化的先锋，提供了一个用于构建自主 **AI agents** 的开源平台，集成了多个 **LLMs**，并能自动化复杂的工作流，详见 [AGiXT GitHub](https://github.com/Josh-XT/AGiXT)。
- **探索 AGiXT 的 AI 自动化**：探索 **AGiXT** 的强大功能，这是一个旨在通过集成多个 **LLMs** 和构建自主 **AI agents** 来自动化复杂工作流的开源平台。



**链接引用**: <a href="https://github.com/Josh-XT/AGiXT">GitHub - Josh-XT/AGiXT：AGiXT 是一个动态的 AI Agent 自动化平台，可无缝编排跨不同 AI 提供商的指令管理和复杂任务执行。结合自适应记忆、智能特性和多功能插件系统，AGiXT 提供高效且全面的 AI 解决方案。</a>

  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1347590222422216775)** (51 messages🔥): 

> `command R7B inference speed, Ollama langchain tool invocation errors, open-source AI projects, GPT-4o Arabic use cases, on-prem deployment costs` 


- **Command R7B 推理速度缓慢的问题**：一名成员报告称，在 Colab Pro A100 GPU 和两块 NVIDIA A100 上使用 HF 库时，**command R7B** 的推理速度*非常慢*，一个简单的聊天生成需要 **30-40 秒**。
   - 另一名成员建议使用 **vLLM** 以获得更快的速度，但指出这需要更多的 GPU 资源且成本更高。
- **Ollama 工具调用错误困扰 Langchain 用户**：一位新用户在 Ollama 和 Langchain 中使用 **command-r7b** 时遇到问题，尽管使用了工具，但仍收到类似 *“抱歉，我无法访问实时数据”* 的错误，而 **llama3.2:3b** 则运行正常。
   - 其他成员建议检查 **Ollama** 是否支持工具调用（tool calling），并确保工具以正确的 JSON 格式传递且已完成绑定。
- **开源 AI 项目寻求者**：一位具有 **GPT-2** 预训练和模型微调经验的社区成员正在寻求有趣的开源 AI 项目建议，特别是与 **LLM pre-training, post-training, RL, 或 interpretability** 相关的项目。
   - 该成员还热衷于建立人脉，特别是与 **加拿大不列颠哥伦比亚省大温哥华地区** 的人士联系。
- **GPT-4o 在高级阿拉伯语用例中表现出色**：一位成员表示，他们长期在高级阿拉伯语用例中使用 **GPT-4o**，其表现是无与伦比的。
   - 另一名成员补充道，“语言只是一个方面”。
- **本地化部署 (On-Prem) 成本是 API 的 20 倍**：成员们讨论了出于隐私考虑的本地化部署，但本地化部署的成本将是使用 API 的 20 倍。
   - 对于需要隐私/控制权的客户，有人指出商业化使用 Cohere 需要支付 5 到 6 位数的授权费用，因为其开放权重模型均为 **CC-BY-NC**（非商业性使用）协议。



**提及的链接**：<a href="https://www.reddit.com/r/AmazonCoolestProducts/s/AJNRLhkMsb">Reddit - Dive into anything</a>：未找到描述

  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1347753806691307562)** (9 messages🔥): 

> `504 Gateway Errors, Multi-Modal Embeddings Availability, API Limit Issues, Rust Requirement for Cohere API` 


- **Cohere 遭遇 504 Gateway 错误**：用户报告了反复出现的 **504 Gateway Errors** 和 **5XX 错误**，影响了生产环境的使用；一名成员请求对这一重复出现的问题进行检查。
   - 一位用户表示，由于 **TPM 限制** 和 **5XX 错误**，他们不得不将 Cohere 从生产环境中移除。
- **Bedrock 和 Azure 上的多模态 Embeddings 推迟**：一位用户询问 **Bedrock** 或 **Azure** 上 **multi-modal embeddings** 的可用性，并对不得不将 **Cohere** 从生产环境中移除表示沮丧。
   - 目前没有给出具体的时间表，但此问题已被记录。
- **API 限制提前触发**：一位用户遇到提示称其 **API limit** 已达到，尽管他使用的是测试 API 密钥且仅嵌入了少量的 chunk。
   - Cohere 团队的一名成员请求该用户的组织 ID 或电子邮件以便进行调查。
- **Cohere 现在需要 Rust 了吗？**：一位用户报告称，在尝试于 **Python3** 中导入 **Cohere** 时遇到了与 **Rust** 相关的错误，尽管他已经更新了系统。
   - 用户收到了关于需要 **Rust** 来编译扩展的错误，并对从 *rustup.rs* 安装表示犹豫。


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1348159203147255950)** (5 messages): 

> `Bot Response Problem` 


- **机器人响应存在问题**：一名成员报告称，机器人显示“正在输入...（typing...）”，但没有收到任何回复。
- **机器人输入指示器问题**：机器人指示正在输入，但未能产生可见的响应。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347592708663017533)** (1 messages): 

> `Knowledge Graphs, TogetherAI, Topic Modeling` 


- **强烈推荐知识图谱 (Knowledge Graphs)**：一名成员建议研究 **Knowledge Graphs**。
   - 另一名成员建议使用执行 **主题建模 (topic modeling)** 的 **LLM**（例如 **TogetherAI**，因为它提供慷慨的免费额度）。
- **用于主题建模的 LLM**：建议使用 **LLM**（如通过 **TogetherAI** 提供的模型）来执行 **topic modeling**。
   - 有人指出 **TogetherAI** 提供了慷慨的免费额度，使其成为执行此任务的一个有吸引力的选择。


  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1347819681024708608)** (4 messages): 

> `使用 Cohere 的应用级 ML 指导，人类神经系统作为逻辑门，情感智能 AI` 


- **寻求应用级 ML 指导**：一位拥有 SoC 设计博士背景的成员正在寻求使用 **Cohere** 学习 **Applied ML** 构建 AI 模型的指导，其背景包括 FPGA 上的 CNN 加速器和编译器。
- **人类大脑作为逻辑门？**：一位来自印度的学生正在探索**人类大脑根据环境条件像逻辑门（AND/OR）一样工作**的概念，并质疑是否可以构建像人类一样思考和感受的 LLM。
   - 他们正在研究 AI 模型背后的数学，理解优化、网络架构和认知模型，旨在*构建能够真正像人类一样思考、感受和反应的东西*。
- **对情感智能 AI 的兴趣萌芽**：一位新成员对**情感智能 AI、认知架构和强化学习 (Reinforcement Learning)** 表现出兴趣，寻求与致力于构建更像人类的 LLM 的其他人建立联系。
   - 他们强调自己缺乏经验但拥有无限的好奇心，希望通过协作学习并成长为一名有能力的 AI 研究员。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1347553807126495232)** (37 messages🔥): 

> `DSPy 的 batch 函数，Agent 通信中的 MCP vs SLOP，DSPy Refine 模块中的错误处理，LLM 客户端中的最大 token 限制与错误处理` 


- **Batching 盛宴：DSPy 的并行处理能力**：一位用户询问 **DSPy** 是否可以使用 `batch` 函数高效地将并行处理委托给具有多个 LLM 实例的 **vllm** 后端。
   - 讨论明确了如果设置了 **vllm 的 pipeline parallel size**，它会处理负载均衡，从而使额外的 DSPy 端配置变得不那么关键。
- **协议大乱斗：Agent 通信中的 MCP vs SLOP**：围绕 **MCP (Model Context Protocol)** 展开了讨论，一些人因其复杂性而持保留意见，并建议使用 **SLOP (Simple Language Open Protocol)** 等替代方案。
   - 替代方案：[SLOP Github](https://github.com/agnt-gg/slop) 和 [SLOP X Post](https://x.com/NathanWilbanks_/status/1898142012991537520)。此外还讨论了 **AgentNetworkProtocol** 的优点 [AgentNetworkProtocol Github](https://github.com/agent-network-protocol/AgentNetworkProtocol)。
- **Refine 的复兴：增强型错误处理出现**：一位用户通过一个 [Pull Request](https://github.com/stanfordnlp/dspy/pull/7926) 强调了 **DSPy `Refine` 模块**中错误处理的改进，实现了对错误容忍度更细粒度的控制。
   - 更新后的功能允许配置在 `Refine` 模块抛出异常之前所允许的错误数量。
- **Token 之争：调试 LLM 客户端中的最大 Token 限制**：一位用户遇到了 signature 返回 **`None` 响应**的问题，后来发现是由于达到了**最大 token 限制 (max token limit)**。
   - 该用户在 **azure gpt-4o-mini** 和 **azure gpt-4o** 上遇到了问题，并注意到了错误信息：`The JSON object must be str, bytes or bytearray, not NoneType.`


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DhravyaShah/status/1898147708138840307">来自 Dhravya Shah (@DhravyaShah) 的推文</a>：冲啊！！！成功让 MCP 在 Durable Objects 上运行了！！！！结合 Agents SDK，每个对象既可以是客户端也可以是服务端。这意味着什么？让我们深入探讨一下。事实证明 @cloudflare 的 Agents SDK 配合...</li><li><a href="https://x.com/veyraxai/status/1897761138840158499">来自 VeyraX (@veyraxai) 的推文</a>：🚀隆重推出 VeyraX MCP：只需 3 分钟即可在 @cursor_ai 中连接 20 多个工具和 100 多个操作。不再需要多个连接——用一个工具控制你所有的工具。现已上线，快来试试...</li><li><a href="https://github.com/Dhravya/mcp-durable-object-client/blob/3674af5fadafec3204924b76c8d3d0b3bf188677/src/server.ts#L51-L233">mcp-durable-object-client/src/server.ts (位于 3674af5 · Dhravya/mcp-durable-object-client)</a>：测试 MCP。通过在 GitHub 上创建账号来为 Dhravya/mcp-durable-object-client 的开发做出贡献。</li><li><a href="https://x.com/lgrammel/status/1897977264953872716">来自 Lars Grammel (@lgrammel) 的推文</a>：MCP 的概念（可以用任何语言实现的远程工具，且可被 LLM 发现/使用）非常棒。然而，其实现方式（需要与服务器保持开放会话）却不尽如人意。它...</li><li><a href="https://github.com/agent-network-protocol/AgentNetworkProtocol">GitHub - agent-network-protocol/AgentNetworkProtocol: AgentNetworkProtocol(ANP) 是一个用于 Agent 通信的开源协议。我们的愿景是定义 Agent 之间如何连接，为数十亿智能 Agent 构建一个开放、安全且高效的协作网络。</a>：AgentNetworkProtocol(ANP) 是一个用于 Agent 通信的开源协议。我们的愿景是定义 Agent 之间如何连接，为数十亿智能 Agent 构建一个开放、安全且高效的协作网络...</li><li><a href="https://github.com/jmanhype/mcp-flux-studio">GitHub - jmanhype/mcp-flux-studio: 一个用于 Flux 图像生成的 Model Context Protocol 服务端，提供图像生成、处理和控制工具</a>：一个用于 Flux 图像生成的 Model Context Protocol 服务端，提供图像生成、处理和控制工具 - jmanhype/mcp-flux-studio</li><li><a href="https://www.youtube.com/watch?v=UTX8QgOTiv0">多 Agent AI 的完美通信协议</a>：使用 G-Designer 的任务优化多 Agent 通信协议。新的 AI 研究论文（见下文）。G-Designer 引入了一个用于动态设计...的框架。</li><li><a href="https://arxiv.org/abs/2410.11782">G-Designer：通过图神经网络构建多 Agent 通信拓扑</a>：基于大语言模型 (LLM) 的 Agent 最近的进展表明，集体智慧可以显著超越单个 Agent 的能力，这主要归功于精心设计的...</li><li><a href="https://github.com/fleuristes/fleur/">GitHub - fleuristes/fleur: 发现和安装 MCP 最简单的方法</a>：发现和安装 MCP 最简单的方法。通过在 GitHub 上创建账号来为 fleuristes/fleur 的开发做出贡献。</li><li><a href="https://github.com/stanfordnlp/dspy/pull/7926">cezarc1 为 Refine 模块添加了改进的错误处理、文档和测试 · Pull Request #7926 · stanfordnlp/dspy</a>：值得注意：为 Refine 添加了可配置的 fail_count 参数，以控制在抛出异常前允许多少次错误。默认情况下，如果所有底层模块调用都失败，我们将抛出一次异常...</li><li><a href="https://news.ycombinator.com/item?id=43302297">MCP vs. API 详解 | Hacker News</a>：未找到描述</li><li><a href="https://github.com/agnt-gg/slop">GitHub - agnt-gg/slop: SLOP 的所在地</a>：SLOP 的所在地。通过在 GitHub 上创建账号来为 agnt-gg/slop 的开发做出贡献。</li><li><a href="https://x.com/NathanWilbanks_/status/1898142012991537520">来自 Nathan Wilbanks (@NathanWilbanks_) 的推文</a>：拒绝 MCP，拥抱 SLOP。简单语言开放协议 (Simple Language Open Protocol)
</li>
</ul>

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1347728837609390143)** (26 条消息🔥): 

> `tinygrad JIT 时间，可疑的 GPU 列表，AMDGPU 运行过热，为什么 OpenCL 失败，define_acc 重构` 


- **Hotz 询问 AMDGPU 休眠状态**：在一名用户报告其 **7900xtx** 在 *amdgpu* 内核模块被列入黑名单时运行温度非常高后，[George Hotz](https://github.com/geohot) 询问运行带有 AMD 驱动程序的 tinygrad 是否会将 **GPU** 置于休眠状态并降低功耗，并指出它在初始化前会消耗大量功率。
   - Hotz 补充说，这种行为是*他们无法控制的*。
- **48GB 是真实的，96GB 很可疑**：多名成员讨论了一个 **GPU** 列表的真实性，结论是虽然 **48GB** 版本可能是真实的，但 **96GB** 版本值得怀疑，可能不是正品。
   - 他们建议在购买 **96GB** 显卡时要谨慎，建议等待可靠来源验证其真实性。
- **OpenCL 的开放竞合 (Open Coopetition)**：分享了来自 [Modular](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives) 的一篇博客文章，剖析了 **OpenCL** 和其他 **CUDA** 替代方案由于*开放竞合*的挑战和管理失误而失败的原因。
   - 该文章是 Modular *民主化 AI 计算*系列的第 5 部分，并引用了[关于 DeepSeek 对 AI 影响的第 1 部分](https://www.modular.com/blog/democratizing-compute-part-1-deepseeks-impact)和[关于模块化的第 4 部分](https://www.modular.com/blog/democratizing-ai-compute-part-4-modularity)。
- **重构 define_acc 比预想的要难**：一位贡献者正在重构 *define_acc*，重点是从中加载而不是直接访问，但某些模式（特别是 *loop_reduce*）不再触发。
   - 贡献者打算在重构完善后将重点转向快速 **AMX**，并在准备就绪时提交 **PR** 进行审查。
- **热门 ONNX Huggingface 仓库通过测试**：一名成员报告他们的 **ONNX** huggingface 脚本快要完成了，前 100 个仓库已通过测试，但由于奇怪的输入规范，具有独特架构的前 100 个仓库失败了。
   - 他们添加了 *dry run*（空运行）功能，并开始着手 *true float16*，注意到 **openpilot** 输入指定了 **float16**，从而提出了测试是否也应该强制执行此操作的问题。



**提到的链接**：<a href="https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives">Modular: Democratizing AI Compute, Part 5: What about CUDA C++ alternatives?</a>：未找到描述

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1347759403423502349)** (8 条消息🔥): 

> `NaN loss 调试，WebGPU long/ulong 问题，TestLinearizerFailures 悬赏，Python Backend CI 中跳过的测试，优化大索引` 


- **NaN Loss 之谜揭晓**：一名成员询问了导致 **NaN loss** 值的原因（除初始步骤 step 0 外）。
   - 消息中未确定根本原因。
- **WebGPU 对 Long 类型的渴望**：一名成员报告说，**WebGPU 实现**在处理 `dtype.long` 时有时会崩溃。
   - 另一名成员确认 **WebGPU 不支持 long/ulong**，但 tinygrad 默认支持比 WebGPU 更多的 dtype，如 [tinygrad/device.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317) 文件所示。
- **TestLinearizerFailures 悬赏 Bug 猎寻**：一名刚接触 tinygrad 代码库的成员正试图复现 `Fix TestLinearizerFailures.test_failure_53` 悬赏，但该测试结果为 `OK`。
   - 他们正在寻求帮助以及调试该问题的后续步骤。
- **Python Backend CI 跳过了一些测试**：一名成员质疑为什么在 `Python Backend` CI 检查中跳过了一些测试。
   - 他们怀疑导致跳过这些测试的奇怪行为是导致布尔索引测试失败的原因。
- **大索引需要优化**：一名成员询问了在 tinygrad 中加速涉及**大索引**操作的代码的方法。
   - 他们提供了一个涉及 `Tensor.linspace`、`Tensor.zeros` 和 `Tensor.randint` 的示例来说明该问题。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/device.py#L317">tinygrad/tinygrad/device.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你热爱 tinygrad！❤️ - tinygrad/tinygrad

  

---

### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1348446345052160072)** (8 messages🔥): 

> `Jamba Workspace, Jamba conversational RAG, Jamba Mini Pricing, AI21 Maestro, Jamba multimodality` 


- ****Jamba Workspace 功能发布独立 RAG 库****：使用 **Jamba/Conversational RAG** 中新的 Workspace 功能创建的每个 Workspace 都将拥有一个独立的 RAG 库，以便独立访问。
   - 这种设置允许在不同的项目或上下文中进行隔离且有组织的 Data Retrieval。
- ****Jamba Mini 的 Tokenomics 公开****：**Jamba Mini** 的定价设定为每 100 万 input tokens **$0.20**，每 100 万 output tokens **$0.40**，更多详情请见[此处](https://www.ai21.com/pricing/)。
- ****AI21 Maestro 编排 AI Planning****：**AI21** 推出了 **Maestro**，这是一个旨在解决复杂任务的 AI Planning & Orchestration 系统，具有基于使用量的定价，并可通过 Foundation Model APIs & SDK 访问所有功能。
   - 对于规模化企业，Custom Plan 包括批量折扣、高级 API Rate Limits、私有云托管、优先支持和专家级 AI 咨询（[了解更多](/maestro?utm_source=banner&utm_medium=top-banner&utm_content=pricing-cost-effective-transparent-pricing-for-ai-products-ai21)）。
- ****Jamba 不支持解析图像****：**Jamba** 不是 multimodal 的，因此不具备处理图像的能力。
   - 然而，如果 PDF 中的图像包含 Metadata 或标题，Jamba 可以理解并使用这些相关的文本信息。



**Link mentioned**: <a href="https://www.ai21.com/pricing/">Pricing</a>: 我们的基于使用量的定价有助于减少不必要的支出。以具有成本效益的价格为您的业务需求找到合适的解决方案。

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347630352071393372)** (9 messages🔥): 

> `Jamba 1.6, AI21 Studio, Mamba1 optimizations, Batch API Solution` 


- **Jamba 1.6 没有 Architecture 变更**：据一名成员透露，新发布的 **Jamba 1.6** 与之前的版本相比*没有 Architecture 变更*。
   - 此版本主要包含 **Performance Optimizations** 和 **Batch API 解决方案**，详见 [AI21 的博客文章](https://www.ai21.com/blog/introducing-jamba-1-6/)。
- **Jamba 1.6 在 Open Model 质量上表现卓越**：**Jamba Large 1.6** 在质量上优于 **Mistral Large 2**、**Llama 3.3 70B** 和 **Command R+**。
   - 此外，**Jamba Mini 1.6** 的表现也优于 **Ministral 8B**、**Llama 3.1 8B** 和 **Command R7B**。
- **Jamba 1.6 具有部署灵活性**：凭借 **256K** 的 Context Window 和混合 SSM-Transformer 架构，**Jamba 1.6** 在 RAG 和长上下文 Grounded Question Answering 任务中表现出色。
   - 除了 **AI21 Studio** 外，这些模型还可以从 [Hugging Face](https://huggingface.co/collections/ai21labs/jamba-16-67c990671a26dcbfa62d18fa) 下载，并私有化部署在 On-prem 或 VPC 中，更多部署选项即将推出。
- **Mamba1 模型的优化？**：一名成员询问了 **Jamba 1.6** 中的性能优化，特别是关于 **Mamba1** 模型，询问当前版本中是否存在此类优化。
   - 该成员希望在博客提到的 *Batch Performance Improvements* 之外了解更多细节。



**Link mentioned**: <a href="https://www.ai21.com/blog/introducing-jamba-1-6/">AI21’s Jamba 1.6: The Best Open Model for Private Enterprise Deployment</a>: AI21 的 Jamba 1.6 优于来自 Mistral、Meta 和 Cohere 的模型，为企业大规模私有化 LLM 部署提供了最佳模型。

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1348735276687687722)** (1 messages): 

> `Multimodal Autonomous AI Agents, VisualWebArena, Internet-scale web-agent training, Ruslan Salakhutdinov` 


- **Ruslan Salakhutdinov 探讨多模态自主 AI Agents**: 今天，Ruslan Salakhutdinov 将在 **PST 时间下午 4 点**进行一场关于 *Multimodal Autonomous AI Agents* 的讲座，可在 [YouTube](https://www.youtube.com/live/RPINOYM12RU) 观看。
   - 他的演讲将介绍能够进行规划、推理并在 Web 上执行操作的多模态 AI Agent，这些 Agent 可以在视觉环境中进行导航和交互。
- **VisualWebArena 框架评估发布**: Salakhutdinov 将展示 **VisualWebArena**，*这是一个用于评估多模态自主语言 Agent 的新型框架*。
   - 他将描述一种推理时搜索算法（inference-time search algorithm），该算法能够在交互式 Web 环境中实现显式探索和多步规划。
- **互联网规模的 Web-agent 训练详解**: 讲座将演示自动化数据流水线如何促进跨 **150,000** 个活跃网站的**互联网规模 Web-agent 训练**。
   - 他还将讨论在数字和物理环境中开发更强大的自主 Agent 的见解。



**提到的链接**: <a href="https://www.youtube.com/live/RPINOYM12RU">CS 194/294-280 (Advanced LLM Agents) - Lecture 6, Ruslan Salakhutdinov</a>: 问题提问：bli.do/rus-sal6

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1348093040018853928)** (8 messages🔥): 

> `Research-track Availability, Quiz Retakes, Curriculum Release, Completion Certificates` 


- **Research-Track 访问权限仍是谜团**: 一位成员询问非伯克利（Berkeley）附属机构的人员是否可以使用 Research-track。
   - 一名工作人员回应称，*目前还没有更新*，但预计本周会有**重大公告**。
- **测验可完成并可重考**: 一位成员称赞了测验的难度。
   - 一名工作人员澄清说，**测验是基于完成情况的**，成员可以重考以提高分数；此外，分数不会影响证书的获取。
- **课程大纲和证书处于待定状态**: 一位成员询问课程大纲是否已经发布，以及**获得结业证书**的标准是什么。
   - 消息中未提供任何回复。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1347621689344786492)** (4 messages): 

> `Research Track Invites, Log Likelihood in Reinforcement Learning` 


- **重复请求 Research Track 邀请**: 多名成员请求重新发送 **Research track 邀请**。
   - 这些请求表明最初的邀请可能已经过期，或者某些感兴趣的参与者未收到邀请。
- **探讨对数似然（Log Likelihood）在强化学习中的作用**: 一位成员试图从条件概率原理出发，理解 **Reinforcement Learning** 背景下的 **log likelihood**。
   - 他们提出，如果 Token/动作是独立的，那么生成的条件概率就是单个 Token 概率的*乘积*，取对数后则变为*对数之和*。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1348144012238000179)** (1 messages): 

> `AI4Legislation Competition, Civic Tech Entrepreneurs, SVCAF, AI-powered civic engagement` 


- **SVCAF 启动 AI4Legislation 竞赛**: [Silicon Valley Chinese Association Foundation](https://github.com/svcaf/2025-AI4Legislation-Public) 将在 **2025** 年夏季举办一场竞赛，旨在创建面向立法公民参与的 **AI 驱动项目**。
   - 设有 **$10,000** 的奖金池，共有 **6 名获奖者**，分为一、二、三等奖。
- **公民科技创业者 Zoom 研讨会宣布**: 首场提供竞赛信息并由**公民科技创业者**参与的公开 Zoom 研讨会将在 **3 月 24 日至 3 月 28 日**那一周举行。
   - 请在[此表单](https://forms.gle/tJjJzHQ9Wk7SEUYm7)报名！


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347706433608613938)** (1 条消息): 

> `Diffusion LLMs, Transformer-based models, LLaDA, Large Language Diffusion Models, autoregressive Transformers` 


- **探讨 Diffusion LLMs 的热度**：一位成员询问了关于 **Mercury** 发布的 **Diffusion LLM** 的热度，以及它是否会取代 **Transformer-based models**。
   - 他们承认发现白皮书难以理解，并寻求社区专家的见解，同时提到了一个[关于该主题的快速信息网站](https://diffusionllm.net/)。
- **LLaDA 范式转移**：**Large Language Diffusion Models (LLaDA)** 使用去噪扩散过程，以并行、由粗到细的方式生成文本，而不是像 **autoregressive Transformers** 那样一次生成一个 token。
   - 这种方法挑战了 LLMs 的优势本质上与 autoregressive 生成绑定的观念，建议通过解决 AR models 的一些局限性来重新定义语言生成。



**提到的链接**：<a href="https://diffusionllm.net/">Diffusion LLMs - Revolutionary Language Model Architecture | LLaDA Research Hub</a>：了解 Diffusion LLMs 如何通过并行处理和高级纠错彻底改变 AI。了解 LLaDA 架构并关注前沿研究。

  

{% else %}


> 完整的逐频道细分内容已在邮件中截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}