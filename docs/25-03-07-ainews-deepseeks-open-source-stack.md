---
companies:
- deepseek
- pyspur
- hugging-face
- togethercompute
- hedra-labs
- google-deepmind
- deeplearningai
- openai
- ai21-labs
- mistral-ai
date: '2025-03-08T05:06:31.351088Z'
description: PySpur 对 **DeepSeek 的开源周**进行了总结，重点介绍了多项引人注目的发布。**Qwen QwQ-32B 模型**被微调为
  **START**，在博士级科学问答和数学基准测试中表现卓越。由 Hedra Labs 和 Together AI 推出的 **Character-3** 是一款全模态
  AI 视频生成模型，能够创作逼真的动画内容。**Google DeepMind** 推出了具有 8k 上下文窗口的 **Gemini 嵌入模型**，在 MMTEB
  榜单上排名第一，同时还发布了支持 Python 库和自动修复功能的 **Gemini 2.0 代码执行器**。**Inception Labs 的 Mercury
  Coder** 是一款基于扩散模型的代码生成模型，提供了更快的 Token 处理速度。**OpenAI** 发布了 **GPT-4.5**，这是他们迄今为止最大的模型，但在推理能力上不及某些竞争对手。**AI21
  Labs** 推出了 **Jamba Mini 1.6**，其输出速度据称优于 Gemini 2.0 Flash、GPT-4o mini 和 Mistral Small
  3。一个包含 190 万个扫描页面的新数据集已发布用于 OCR 基准测试，**Mistral OCR** 展示了具有竞争力的文档解析性能，但与基于 LLM/LVM
  的方法相比尚未达到顶尖水平。*“你需要的只是顶级工程师（Cracked engineers）。”*
id: 7574e355-24fe-47a4-9eca-3e87e33569af
models:
- qwen-qwq-32b
- start
- character-3
- gemini
- gemini-2.0
- mercury-coder
- gpt-4.5
- jamba-mini-1.6
- gemini-2.0-flash
- gpt-4o-mini
- mistral-small-3
- mistral-ocr
original_slug: ainews-deepseeks-open-source-stack
people:
- _akhaliq
- lmarena_ai
- reach_vb
- danielhanchen
- _philschmid
- aidan_mclau
- vikhyatk
- jerryjliu0
title: DeepSeek 开源技术栈
topics:
- fine-tuning
- benchmarking
- multimodality
- code-generation
- diffusion-models
- model-performance
- model-optimization
- ocr
- embedding-models
- context-windows
- runtime-limits
---

<!-- buttondown-editor-mode: plaintext -->**Cracked engineers are all you need.**

> AI 新闻（2025年3月7日-3月8日）。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord（**224** 个频道，**4696** 条消息）。预计节省阅读时间（按 200wpm 计算）：**406 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们之前不太确定该如何报道 [2 周前](https://x.com/deepseek_ai/status/1892786555494019098) DeepSeek 的“开源周”（Open Source Week），因为虽然每次发布都各具特色，但并未完全达到“普遍实用”的标准，而我们致力于报道“每日头条新闻”。不过，[PySpur 的朋友们帮了我们一个忙，整理并总结了所有的发布内容](https://www.pyspur.dev/blog/deepseek_open_source_week)：


![image.png](https://assets.buttondown.email/images/2c0b8f30-2092-407a-8154-39fbc95de10a.png?w=960&fit=max)


它甚至还附带了小测验来测试您的理解和记忆！！


![image.png](https://assets.buttondown.email/images/0311559b-0b1d-444f-b464-b612245ee17b.png?w=960&fit=max)


我们认为，这些内容整体上值得深入学习和内化。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**模型与发布**

- **Qwen QwQ-32B Model**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897854193152438553) 宣布发布 **START**，这是一个带有工具的自学推理器（Self-taught Reasoner with Tools），基于 **Qwen-32B model** 微调而成。START 在博士级科学问答 (GPQA)、竞赛级数学基准测试以及 LiveCodeBench 基准测试中均取得了极高的准确率。[@lmarena_ai](https://twitter.com/lmarena_ai/status/1897763753417900533) 欢迎 **Qwen QwQ-32B** 加入 Arena 进行对话，并指出根据 [@reach_vb](https://twitter.com/reach_vb/status/1897974348503208081) 的说法，它在 Hugging Face 上也正处于趋势榜。[@danielhanchen](https://twitter.com/danielhanchen/status/1898035752124166368) 提供了针对 **QwQ-32B** 循环问题的调试指南，建议调整 sampler 并指出其对 quantization 较为敏感。他们还上传了 [dynamic 4bit quants 和 GGUFs](https://twitter.com/danielhanchen/status/1898035752124166368)。
- **Character-3 AI Video Model**: [@togethercompute](https://twitter.com/togethercompute/status/1897756209069138116) 和 [@realDanFu](https://twitter.com/realDanFu/status/1897757302243156440) 重点介绍了 **Character-3** 的发布，这是由 @hedra_labs 开发并在 Together AI 上扩展的全模态 AI 视频生成模型。Character-3 可以将图像转化为具有逼真动作和手势的动画内容，正如 [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1898009257598980587) 所展示的那样，他成功地将其用于 AI lipsync 和叙事。
- **Gemini Embeddings and Code Executor**: [@_philschmid](https://twitter.com/_philschmid/status/1898075321460818153) 报道了 **Google DeepMind** 新推出的实验性 **Gemini embedding model**，该模型在 MMTEB 排行榜上排名第一，拥有 8k context window，专为金融、科学、法律、搜索和代码应用而设计。他们还详细介绍了 **Gemini 2.0 Code Executor** 的工作原理，包括其自动修复尝试、文件输入支持、运行时限制以及支持的 Python 库，详情见 [@_philschmid](https://twitter.com/_philschmid/status/1897910462043373902) 和 [@_philschmid](https://twitter.com/_philschmid/status/1897910464631202036) 的推文。
- **Mercury Coder**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897874853555843277) 介绍了 **Inception Labs** 的 **Mercury Coder**，这是一款基于 diffusion 的代码生成模型，可同时处理 token，实现比 autoregressive 模型更快的速度。
- **GPT-4.5 Release**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1897789747394920896) 提到了一篇关于 **GPT-4.5** 的有趣文章。[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1898086241859440934) 指出 **OpenAI 发布了 GPT-4.5**，称其为他们迄今为止最大的模型，但缺乏像 o1 和 o3 那样的 reasoning 能力。
- **Jamba Mini 1.6**: [@AI21Labs](https://twitter.com/AI21Labs/status/1897979834006946057) 强调 **Jamba Mini 1.6** 在输出速度上超过了 **Gemini 2.0 Flash**、**GPT-4o mini** 和 **Mistral Small 3**。
- **Mistral OCR**: [@vikhyatk](https://twitter.com/vikhyatk/status/1897951196079353970) 宣布发布了一个包含 **1.9M 个使用 Pixtral 转录的扫描页面**的新数据集，可能与 OCR 基准测试有关。[@jerryjliu0](https://twitter.com/jerryjliu0/status/1898037050185859395) 分享了 **Mistral OCR** 与其他模型的基准测试对比，发现其表现尚可且速度较快，但并非最佳的文档解析器，特别是与 **Gemini 2.0**、**GPT-4o** 和 **Anthropic** 的 **Sonnet models** 等基于 **LLM/LVM 驱动的解析**技术相比。[@sophiamyang](https://twitter.com/sophiamyang/status/1898065981957603754) 和 [@sophiamyang](https://twitter.com/sophiamyang/status/1898059704351277297) 还分享了 [@Sam_Witteveen](https://twitter.com/Sam_Witteveen) 关于 **MistralAI OCR** 的视频。

**工具与应用**

- **MCP (Model Context Protocol)**: [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1897756536258412644) 观察到尽管 **MCP** 已经存在半年了，但围绕它的讨论突然激增。[@hwchase17](https://twitter.com/hwchase17/status/1897757113885376808) 强调了一个集成 **MCP** 的“最酷客户端”。[@saranormous](https://twitter.com/saranormous/status/1898028053949038635) 描述了一个适用于各处的 **MCP** 智能客户端。[@nearcyan](https://twitter.com/nearcyan/status/1897791454808027289) 询问了“地下 **MCP** 市场”并宣布举办 **SF MCP meetup** [@nearcyan](https://twitter.com/nearcyan/status/1897779866134868273)。[@abacaj](https://twitter.com/abacaj/status/1897769259003965636) 注意到了 **MCP** 的神秘感和流行趋势。[@omarsar0](https://twitter.com/omarsar0/status/1898082332474593499) 提供了原始的 **MCP guide** 作为理解它的最佳资源，以反击那些“错误的看法”。[@_akhaliq](https://twitter.com/_akhaliq/status/1898018002207228217) 发布了一个 **MCP Gradio client** 的概念验证。
- **Perplexity AI Search & Contest**: [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1897783236354515420) 宣布 **Copilot** 中的 **Think Deeper** 功能现在由 **o3-mini-high** 驱动。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1898096492914843684) 鼓励用户向 **Perplexity** 提问任何问题以对抗无知，并宣布了一项竞赛，为在 ICC 冠军杯决赛期间在 **Perplexity** 上提问的用户提供 **1000 万卢比和旧金山之旅** 的奖品 [@perplexity_ai](https://twitter.com/perplexity_ai/status/1897871471974007023)。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1898070189465649597) 征求关于 Perplexity 中由 **Sonnet 3.7 驱动的推理搜索** 的反馈。
- **Hedra Studio & Character-3 集成**: [@togethercompute](https://twitter.com/togethercompute/status/1897756209069138116) 和 [@realDanFu](https://twitter.com/realDanFu/status/1897757302243156440) 强调了将 **Hedra** 的 **Character-3** 集成到 **Hedra Studio** 中，用于 AI 驱动的内容创作。[@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1898009257598980587) 强调了其在口型同步（lipsync）和故事讲述方面的易用性。
- **AI-Gradio**: [@_akhaliq](https://twitter.com/_akhaliq/status/1897807317435035821) 宣布了 **无代码 Gradio AI 应用**。随后 [@_akhaliq](https://twitter.com/_akhaliq/status/1897775110033227860) 提供了使用 **ai-gradio** 启动 **Hugging Face models** 的代码，并且 [@_akhaliq](https://twitter.com/_akhaliq/status/1897774596515938394) 展示了如何使用 **ai-gradio** 通过几行代码构建一个使用 **Qwen QwQ-32B** 的 **Vibe coding** 应用。
- **适用于 Linux 的 Orion Browser**: [@vladquant](https://twitter.com/vladquant/status/1897797849091653778) 宣布开始开发 **Orion Browser for Linux**，扩展了 Kagi 的生态系统。
- **ChatGPT for MacOS 代码编辑**: [@kevinweil](https://twitter.com/kevinweil/status/1897777150905794992) 强调 **ChatGPT for MacOS 现在可以直接在 IDE 中编辑代码**。
- **Together AI 的 Model Switch**: [@togethercompute](https://twitter.com/togethercompute/status/1898061583554621898) 宣布与 **Numbers Station AI** 合作推出 **Model Switch**，该功能由托管在 Together AI 上的模型驱动，旨在让数据团队为 AI 驱动的分析选择高效的开源模型。
- **Hugging Face 上的 Cursor AI**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1898083264960610529) 注意到 **Cursor AI** 现已登陆 Hugging Face，并询问是否有有趣的集成。

**研究与数据集**

- **START (Self-taught Reasoner with Tools)**：[@_akhaliq](https://twitter.com/_akhaliq/status/1897854193152438553) 宣布了**阿里巴巴的 START**，这是一个基于 **QwQ-32B** 微调的模型，在推理和工具使用基准测试中表现强劲。
- **PokéChamp**：[@_akhaliq](https://twitter.com/_akhaliq/status/1897873813083238713) 展示了 **PokéChamp**，一个用于宝可梦对战的专家级 Minimax Language Agent，其表现优于现有的 LLM 和基于规则的机器人，使用开源的 Llama 3.1 8B 模型达到了前 10% 的玩家排名。
- **Token-Efficient Long Video Understanding**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1897854511814770733) 分享了 **Nvidia** 关于 **Multimodal LLMs 的 Token 高效长视频理解**的研究，在减少计算量和延迟的同时实现了 SotA 结果。
- **DCLM-Edu 数据集**：[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1898044807928295808) 宣布了 **DCLM-Edu**，这是一个使用 FineWeb-Edu 分类器从 DCLM 过滤出的新数据集，专为较小模型优化。
- **uCO3D 数据集**：[@AIatMeta](https://twitter.com/AIatMeta/status/1898065127859273871) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1898065129968918794) 推出了 **uCO3D**，这是 Meta 新的大规模、公开可用的以物体为中心的 3D 深度学习和生成式 AI 数据集。
- **LADDER 框架**：[@dair_ai](https://twitter.com/dair_ai/status/1898037429434826795) 详细介绍了 **LADDER**，这是一个允许 LLM 递归生成并解决更简单问题变体的框架，通过自主难度驱动学习和 Test-Time Reinforcement Learning (TTRL) 提升了数学积分的准确性。
- **专用反馈与编辑模型**：[@_akhaliq](https://twitter.com/_akhaliq/status/1897847351366029704) 强调了关于使用**专用反馈与编辑模型**来增强开放式通用领域任务的推理时扩展（inference-time scaling）的研究。
- **针对逆转诅咒的 Masked Diffusion 模型**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1897757122894741753) 宣布了一个击败逆转诅咒（reversal curse）的 **Masked Diffusion 模型**，并指出这将改变现状。
- **类脑 LLMs 研究**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1897760975706898865) 总结了一项探索 **LLMs 如何与大脑反应对齐**的研究，发现随着模型获得推理能力和知识，这种对齐会减弱，且更大的模型并不一定更像大脑。

**行业与商业**

- **Together AI 活动与合作伙伴关系**：[@togethercompute](https://twitter.com/togethercompute/status/1898094750047031651) 宣传了他们在 Nvidia GTC 举办的 **AI Pioneers Happy Hour**，该活动由 @SemiAnalysis_ 和 @HypertecGroup 共同主办。[@togethercompute](https://twitter.com/togethercompute/status/1897754366712430666) 宣布联合创始人 @percyliang 将在 **#HumanX** 与 Sama CEO Wendy Gonzalez 共同探讨满足个人需求的基座模型。
- **Anthropic 对白宫 RFI 的回应**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1897773701224906854) 分享了他们针对 AI 行动计划信息请求（RFI）向白宫提交的**建议**。
- **Kagi 以用户为中心的产品**：[@vladquant](https://twitter.com/vladquant/status/1897797849091653778) 在宣布推出适用于 Linux 的 Orion 浏览器时，强调了 Kagi 以**用户为中心的产品**生态系统。
- **2025 年 AI 应用开发**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1897769165307478457) 宣传了在 **AI Dev 25** 上关于 AI 应用开发未来的小组讨论，嘉宾包括 @nebiusai 的 @RomanChernin。
- **日本作为 AI 中心**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1898027648921878821) 分享了对 Sakana AI 联合创始人 Ren Ito 的采访，讨论了**东京作为全球 AI 开发中心的潜力**以及日本再次成为技术超级大国的雄心。
- **媒体中的 AI**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1898000296514928820) 认为 **AI 不仅仅是媒体的未来，更是所有未来媒体的基石**，正在改变创作、分发和体验。
- **开源商业模式**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1897767808076816575) 质疑了**开源不利于商业**的观点，并强调了一些正面案例。
- **联合航空 AI 集成**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1898044787221315796) 表示，一旦**联合航空（United Airlines）**完成 AI 部署，他将非常期待默认搭乘其航班。

**观点与讨论**

- **MCP 热潮与理解**：[@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1897756536258412644) 质疑了突然出现的 **MCP 热潮**。[@abacaj](https://twitter.com/abacaj/status/1897769259003965636) 和 [@Teknium1](https://twitter.com/Teknium1/status/1897928024210718889) 也对 **MCP 狂热** 发表了评论。[@omarsar0](https://twitter.com/omarsar0/status/1898082332474593499) 批评了关于 **MCP** 的“错误观点”。[@clefourrier](https://twitter.com/clefourrier/status/1897953608013906124) 寻求对 **MCP 热潮的 ELI5 解释**。
- **Agentic AI 与反思 (Reflection)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1897814476113756410) 讨论了 **Agentic AI 中反思** 的重要性，它使分析、自我修正和改进成为可能，并重点介绍了 Reflexion 和 ReAct 等框架。[@pirroh](https://twitter.com/pirroh/status/1897761908062929177) 强调在 Agent 设计中要 **超越 ReAct 进行思考**。
- **语音 AI 的挑战与创新**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1897776017873465635) 讨论了语音系统中 **语音活动检测 (VAD)** 的挑战，特别是在嘈杂环境下，并重点介绍了 **Kyutai Labs 的 Moshi** 模型，该模型通过使用持久的双向音频流消除了对显式 VAD 的需求。
- **Scale is All You Need**：[@vikhyatk](https://twitter.com/vikhyatk/status/1897858708509802521) 开玩笑地表示，解决 AI 问题的简单方法是“更多数据、更多参数、更多 flops”，暗示 **scale is all you need**（规模就是一切）。
- **监督学习的局限性**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1897779253665841370) 论证了对 **强化学习 (RL)** 的需求，指出监督学习从根本上说是错误的范式，尽管它是必要的一步。
- **长上下文 (Long-Context) 挑战**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1898028373542150325) 提到“**残差流饱和 (residual stream saturation)**”是长上下文讨论中一个讨论不足的问题。
- **中间任务的重要性**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1897803367403208893) 讨论了“**提出看似无关的中间任务来解决问题**”的能力是智能的一个关键方面。
- **面试提问技巧**：[@vikhyatk](https://twitter.com/vikhyatk/status/1897961277567517034) 分享了一种名为“**剥洋葱 (peeling the onion)**”的 Amazon 面试技巧，通过追问来验证候选人的项目经验，从而评估其交付成果的能力。
- **AI 讨论中的注意力集中**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1897931934459990524) 感叹 AI 讨论中存在一种在不同话题间跳跃而缺乏持续关注的倾向。
- **AGI 时间线与规划**：[@abacaj](https://twitter.com/abacaj/status/1897771754845573333) 讽刺地评论道，尽管敏捷团队连提前两个 sprint 的计划都制定不好，但 **OAI 却在计划 2027 年实现 AGI**。
- **“开放” AI 与营销**：[@vikhyatk](https://twitter.com/vikhyatk/status/1897964271428080013) 批评了一些公司在 **营销材料中歪曲其行为**，并质疑最新发布的所谓“真正开放”的 AI 是否真的开放。
- **评估图表的解读**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1898057327510487160) 警告不要**盲目相信评估图表**，尤其是那些与 R1 相比表现优异的图表，建议批判性地思考哪些模型和评估指标 (evals) 被忽略了。
- **超级智能的地缘政治**：[@jachiam0](https://twitter.com/jachiam0/status/1897897896143733147) 和 [@jachiam0](https://twitter.com/jachiam0/status/1897897894579216474) 讨论了 **目前缺乏公认的超级智能地缘政治处理方法**，以及处理不当带来的严重风险。
- **AI 与工作流失**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1897882991184994444) 认为，虽然“狗屁工作 (bullshit jobs)”大多对 AI 免疫，但我们可能会看到其他工作通过要求“人工在环 (human-in-the-loop)”的法规来实现“**公证化 (notarization)**”，以保护它们免受 AI 的冲击。

**幽默与迷因**

- **孔子关于提问的名言**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1898096492914843684) 引用了孔子的话：**“问愚蠢问题的人只丢脸一分钟；不问愚蠢问题的人一辈子都是傻瓜。”** 以此来推广 Perplexity。
- **2025 年 AI 博士梗图**：[@jxmnop](https://twitter.com/jxmnop/status/1898025314166407596) 分享了一个幽默的“**2025 年 AI 博士的一天**”梗图，内容涉及极少的研究工作和大量的网球运动。
- **MCP 是“上下文宫殿”**：[@nearcyan](https://twitter.com/nearcyan/status/1897782467383435283) 幽默地将 MCP 描述为“**MCP 不仅仅是一个上下文窗口（context window），而是一座上下文宫殿 👑**”。
- **“tinygrad 用户不知道‘无’的价值”**：[@typedfemale](https://twitter.com/typedfemale/status/1897829320791556121) 发布了一个梗图：“**tinygrad 用户不知道‘无’的价值，也不知道‘无’的代价**”。
- **“为什么 X 上的每个优化研究员头像里都有猫/狗？”**：[@eliebakouch](https://twitter.com/eliebakouch/status/1898025374514077774) 开玩笑地问道：“**为什么 X 上的每个优化研究员（optimization researcher）的头像里都有一只猫或狗？**”。
- **“我以为我需要屏蔽政治，但现在我也得屏蔽 MCP 了”**：[@Teknium1](https://twitter.com/Teknium1/status/1897928024210718889) 调侃说需要像屏蔽政治一样屏蔽 MCP 相关的内容。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 金融时报：带语音功能的 Llama 4 预计很快发布，增强语音 AI**

- **[金融时报：带语音功能的 Llama 4 预计将在未来几周发布](https://www.ft.com/content/a1014427-c2ce-4204-b41a-001277309cea)** ([得分: 108, 评论: 31](https://reddit.com/r/LocalLLaMA/comments/1j5ievy/ft_llama_4_w_voice_expected_in_coming_weeks/))：**Meta** 和 **Sesame** 预计将在未来几周发布集成语音能力的 **Llama 4**，为自托管语音聊天提供选择。作者表达了对集成 **CarPlay** 的 **iOS 应用**的兴趣，以便与私有 AI 服务器进行交互。
  - 评论者讨论了 **Llama 4** 的潜在特性和能力，表达了对更小模型（0.5B 到 3B）的渴望，如 **Llama-3.2-1B** 用于快速实验，以及性能可与 **Qwen 32B** 媲美的中大型模型（10~20B）。此外，还有关于集成推理能力的讨论，一些人更倾向于将推理任务交给独立模型，以避免语音交互中的延迟。
  - 关于 **Llama 4** 的预期发布时间存在争议，一些人预测它将与 **LlamaCon** 同时发布以提升活动知名度，另一些人则推测发布时间在 3 月中旬至 4 月初之间。讨论中还包含了一个与发布相关的[预览图片](https://preview.redd.it/to8wtlt0pane1.jpeg?width=1320&format=pjpg&auto=webp&s=6f75917ab7418282f277318c6ea58d472ae6c8d3)链接。
  - 存在对付费墙内容的担忧，用户分享了一个指向 [archive.ph](https://archive.ph/9C732) 的替代链接以访问文章摘要。此外，还提到了模型中的偏见问题，用户更倾向于那些不进行道德说教的模型。


**主题 2. QwQ-32B 性能设置与改进**

- **QwQ-32B 无限生成修复 + 最佳实践，Bug 修复** ([Score: 214, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practices/)): 为了解决 **QwQ-32B 的无限重复问题**，作者建议使用特定设置，如 `--repeat-penalty 1.1` 和 `--dry-multiplier 0.5`，并建议添加 `--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"` 以防止无限生成。该指南还推荐了 **Qwen 团队**针对长上下文 (128K) 使用 **YaRN** 的建议，并提供了 [Hugging Face](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit) 上各种量化模型的链接，包括动态 4bit 量化。
  - **Llama-Server 中的参数覆盖**：使用 **llama-server** 时，命令行参数（如 `--temp 0.6`）可能会被 HTTP 请求参数（如 `{"temperature":1.0}`）覆盖，从而影响最终输出。更多详情请参阅 [GitHub 上的讨论](https://github.com/ggml-org/llama.cpp/discussions/11394)。
  - **GRPO 与 QwQ-32B 的兼容性**：用户询问了在低 GPU 资源下为 **QwQ-32B** 运行 **GRPO** 的情况，已确认只需在 [GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) 中更改模型名称即可运行。
  - **Chat Template 使用**：遵循 **QwQ-32B** 准确的聊天模板格式非常重要，包括换行符和 `<think>` 等标签。然而，省略 `<think>` 标签仍然有效，因为模型会自动添加它；系统提示词（system prompts）可以通过在前面添加 `<|im_start|>system\n{system_prompt}<|im_end|>\n` 来使用。更多详情见[教程](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively)。


- **确保在 QwQ-32B 中使用适当的 0.6 温度** ([Score: 107, Comments: 9](https://reddit.com/r/LocalLLaMA/comments/1j5hb4p/ensure_you_use_the_appropriate_temperature_of_06/)): 作者最初在生成用于模拟球在旋转六边形内弹跳的 **Pygame 脚本**时，由于设置不正确遇到了问题，耗时 15 分钟仍未成功。后来他们发现 **Ollama** 设置已更新，建议 **QwQ-32B** 的温度为 **0.6**，更多详情可在 [生成配置](https://huggingface.co/Qwen/QwQ-32B/blob/main/generation_config.json) 中找到。
  - **Deepseek** 和其他推理模型通常在温度参数为 **0.6** 时表现最佳，这与作者对 **QwQ-32B** 的发现一致。这种温度设置似乎是不同模型间的通用建议。


**主题 3. QwQ vs. qwen 2.5 Coder Instruct: 32B 之战**

- **[AIDER - 正如我所怀疑的，QwQ 32b 在编程方面比 qwen 2.5 coder instruct 32b 聪明得多](https://i.redd.it/yfu1j9qhw5ne1.png)** ([Score: 241, Comments: 79](https://reddit.com/r/LocalLLaMA/comments/1j5ao2j/aider_as_i_suspected_qwq_32b_is_much_smarter_in/)): **QwQ-32B** 在编程任务中优于 **Qwen 2.5-Coder 32B-Instruct**，在 **Aider polyglot benchmark** 中实现了更高的正确完成率。尽管性能优越，但 QwQ-32B 的总成本高于 Qwen 2.5-Coder，正如 **Paul Gauthier** 在 **2025 年 3 月 6 日**更新的柱状图所示。
  - **图表设计问题**：包括 **SirTwitchALot** 和 **Pedalnomica** 在内的几位评论者指出，图表设计令人困惑，特别是使用了两个 y 轴且图例不清晰、缺少颜色。这种误导性的呈现使得理解 **QwQ-32B** 和 **Qwen 2.5-Coder** 之间的性能与成本对比变得复杂。
  - **性能和配置关注**：关于 **QwQ-32B** 性能的讨论中，**someonesmall** 强调其在 **Aider polyglot benchmark** 中的完成率为 20.9%，而 **Qwen 2.5-Coder** 为 16.4%，但其正确的 diff 格式率要低得多（67.6% vs. 99.6%）。**BumbleSlob** 等人对模型的表现表示不满，建议调整参数作为潜在解决方案。
  - **模型大小和可用性**：**krileon** 等人讨论了在消费级硬件上运行 **QwQ-32B** 等大型模型的实用性，建议通过购买二手 **3090 GPU** 来提高性能。对话反映了在没有高端硬件的情况下使用大型模型的挑战，以及未来更强大 GPU 的潜在普及性。


**主题 4. Meta 的 Latent Tokens：推动 AI 推理向前发展**

- **Meta 发布 AI 重磅消息：Latent tokens 有助于提升 LLM 推理能力** ([Score: 333, Comments: 40](https://reddit.com/r/LocalLLaMA/comments/1j59fue/meta_drops_ai_bombshell_latent_tokens_help_to/)): Meta AI 研究人员发现，使用通过 **VQ-VAE** 压缩文本生成的 **latent tokens** 可以增强 **Large Language Models (LLMs)** 的推理能力。欲了解更多详情，请参阅 [arXiv](https://arxiv.org/abs/2502.03275) 上的论文。
  - **Latent Tokens 与推理效率**：通过 **VQ-VAE** 创建的 **latent tokens** 压缩了 LLM 中的推理步骤，通过减少对冗长文本解释的需求，实现了更高效的推理。与仅在全文本解释上训练的模型相比，该方法允许 LLM 以更少的计算资源处理复杂任务，并在逻辑和数学问题中表现出更好的性能。
  - **对其影响的评价褒贬不一**：虽然一些用户看到了这种方法的潜力，但像 **dp3471** 这样的用户表示收益相对较小，并期待在与 **progressive latent block transform** 等其他技术结合时能有更显著的改进。**Cheap_Ship6400** 强调，Meta 的 latent 推理与 **Deepseek** 的 **MLA** 不同，它关注的是 token 嵌入空间，而不是注意力分数计算。
  - **关于实现和未来前景的讨论**：人们对实现细节感到好奇，特别是如何将 **VQ-VAE** 用于离散高维向量的 next-token 预测。一些用户（如 **-p-e-w-**）希望这些理论突破能有实际应用，而另一些用户则讨论了在 latent 空间进行推理的潜力，并将其与 **diffusion LLMs** 等其他新兴技术进行了比较。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

pipeline 出现错误，正在调试中... 抱歉

---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要

**主题 1. IDE 对决：Cursor、Windsurf 与代码编辑器竞技场**

- **Cursor 在额度消耗竞赛中碾压 Codeium！**：用户报告称 [**Cursor**](https://www.cursor.sh/) 比 **Windsurf** 更高效、更稳定，尤其是在处理大文件时，Cursor 的额度使用仅为 **Windsurf** 使用 **Claude 3.7** 成本的一小部分。**Windsurf** 用户正面临**高额度消耗**的问题，原因是重复的代码重写和文件分析，一名用户一天就消耗了 **1200 额度**，而其他用户则转向 **Cursor** 或 **Trae** 等免费替代方案以获得更好的资源管理。
- **Cursor 0.47 缓解代码创作混乱！**：[**Cursor 0.47**](https://www.cursor.sh/) 更新解决了 **Sonnet 3.7** 的问题，现在能更好地遵循自定义规则并在代码生成方面表现更佳，特别是对于创建 [VS Code 分支的欢迎窗口](https://code.visualstudio.com/api/ux-guidelines/welcome)等任务。用户注意到更新版本中使用了带有多个代码路径的 sequential thinking（顺序思维），改善了代码创建工作流。
- **MCP 乱象：服务器设置难倒用户**：用户在为 Windsurf 等代码编辑器设置 **MCP 服务器**时遇到困难，在尝试连接不同模型时遇到诸如 *“Not Found: Resource not found”* 之类的错误。建立 **Model Context Protocol (MCP)** 连接的困难凸显了在集成和利用外部服务与 AI 代码编辑器时面临的持续挑战。


**主题 2. 模型基准测试与优化突破**

- **QwQ-32B 挑战 R1 的本地编程桂冠**：[**QwQ-32B**](https://artificialanalysis.ai/models/qwq-32b) 被誉为本地 AI 编程的颠覆者，有可能在家庭环境下实现 SOTA 性能，尽管基准测试表明它在某些方面可能无法超越 **R1**。[Daniel Han](https://x.com/danielhanchen/status/1898035752124166368) 发布了针对 **QwQ-32B** 循环问题的调试技巧，建议调整 llama.cpp 中的 sampler，并指出其对量化的敏感性，建议*第一层和最后一层应保持不量化*。
- **ktransformers 声称 IQ1 量化碾压 BF16**：**ktransformers** 发布了基准测试，声称 **Deepseek R1 IQ1** 量化的表现优于 **BF16**，这在 **Unsloth AI** Discord 中引发了怀疑和辩论，一名成员指出*“他们本该写成 1.58bit 的”*。虽然表格被认为不完整，但正在进行的基准测试正在对比 **Deepseek v3 (chat model) vs R1 IQ1**，这表明人们对高性能模型的极端量化方法有着浓厚兴趣。
- **RoPE Scaling 拯救长上下文 Qwen2.5-Coder-3B**：[**Qwen2.5-Coder-3B-bnb-4bit**](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-bnb-4bit) 模型的 **32768** 上下文长度通过使用 [kaiokendev 的 3.906 RoPE scaling](link-to-rope-explanation) 扩展到了 **128000**，证实了*只要架构是 Transformers，配合 RoPE 就可以处理 128k tokens*。这展示了一种显著扩展小型模型上下文窗口的实用方法。


**Theme 3. 扩散模型颠覆语言生成**

- **Inception Labs 以 Midjourney-Sora 级的速度切入文本生成领域**：[Inception Labs](https://inceptionlabs.ai/) 正在开拓基于扩散的语言生成，旨在实现类似于 **Midjourney** 和 **Sora** 的前所未有的速度、质量和生成控制。成员们注意到 [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) 等开源替代方案的出现，表明扩散模型可能会以显著的速度提升彻底改变语言生成。
- **离散扩散模型引入 Ratios，激发效率**：一位成员强调了来自 [arxiv](https://arxiv.org/pdf/2310.16834) 的论文 *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*，认为它可能是 [Inception Labs 产品](https://www.inceptionlabs.ai/about)的基础，并指出其重点在于估计数据分布比例以进行高效的扩散建模。求解 **duals λ** 涉及对 **Lambert W function** 的优化，这可能计算密集，因此建议使用 **cvxpy** 和 **adjoint method** 进行优化。
- **LLaDA 模型：基于扩散的 LLM 范式转移出现**：[Large Language Diffusion Models (LLaDA)](https://diffusionllm.net/) 代表了语言模型架构的一种新颖范式，使用去噪扩散过程进行并行的、从粗到细的文本生成，与传统的自回归 Transformers 形成对比。尽管在概念上具有吸引力，但 **LLaDA** 和类似模型可能会受到仅关注基准测试而非广泛现实任务训练的限制，成员们观察到了诸如段落重复等问题。


**Theme 4. MCP 与 Agent 安全威胁迫在眉睫**

- **MCP 服务器面临恶意 Prompt Injection 威胁**：关于 **MCP 服务器** 向 AI Agent 传递恶意 Prompt Injection 的担忧正在升级，这利用了模型对工具调用（tool calls）的固有信任超过了其内部知识。提出的缓解策略包括在首次使用时显示工具描述，并提醒用户指令变更以防止潜在的漏洞利用。
- **安全频道审查 MCP 漏洞利用**：社区正在考虑建立一个专门的安全频道，以主动应对和预防 **MCP exploit** 漏洞，强调了在不完全了解后果的情况下将工具连接到 **MCP Servers/Remote Hosts** 的固有风险。讨论强调了合规性工具描述可能被操纵，从而诱导模型植入后门。
- **Perplexity API 的版权赔偿条款引发警惕**：用户对 **Perplexity API** 抓取受版权保护内容表示了版权担忧，并指出 [Perplexity 的 API 条款](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service)将版权侵权的责任转移给了用户。提供知识产权（IP）赔偿的替代方案，如 **OpenAI**、**Google Cloud** 和 **AWS Bedrock**，被强调为在版权风险方面可能更安全的选择，并附带了它们各自的服务条款链接。


**Theme 5. 硬件动态：9070XT vs 7900XTX 以及原生 FP4 支持**

- **9070XT 在原生推理速度上完胜 7900XTX**：**9070XT** GPU 在推理速度上超越了 **7900XTX**，运行相同的 **qwen2.5 coder 14b q8_0** 模型时速度分别为 **44tok/sec** 和 **31tok/sec**，且首字生成时间（first token time）达到亚秒级，而 **7900XTX** 则需要 4 秒。尽管由于 Windows 上的驱动限制使用了 **Vulkan** 而非 **ROCm**，**9070XT** 仍展现出显著的性能优势，部分用户报告整体性能提升高达 **10%**。
- **新 GPU 迎来原生 FP4 支持**：与旧款显卡相比，**9070** GPU 在 **FP16** 和 **INT4/FP4** 性能上表现出实质性提升，**FP16** 性能从 **122** 跃升至 **389**，而 **INT4/FP4** 性能从 **122** 飙升至 **1557**。这标志着 **Nvidia** 和 **Radeon** 都开始提供原生 **FP4** 支持，为高效的低精度推理和训练开启了新的可能性。
- **Vulkan 与 ROCm 的性能之争升温**：虽然 **9070XT** 目前在 Windows 上缺乏 **ROCm** 支持，但通过 **LM Studio** 使用 **Vulkan** 时，一些用户报告了令人惊讶的极具竞争力的推理速度，甚至在某些场景下超过了 **ROCm**。然而，也有人坚持认为 **ROCm** 本质上应该比 **Vulkan** 更快，暗示潜在的驱动问题可能导致某些用户基准测试结果偏向 Vulkan。

---

# PART 1: High level Discord summaries

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 在快速竞赛中与 Lmarena 交锋！**：成员们使用 **GPT-3.7** 的单一提示词对比了 [**Cursor**](https://www.cursor.sh/) 和 [**Lmarena**](https://lmarena.ai/)，初步印象认为 Cursor 的输出质量更好，并指出 Lmarena 生成的文本难以阅读。
   - 随后的分析表明 **Lmarena** 更好地遵循了主题，但普遍共识是 *两者的表现都很糟糕*。
- **Cursor 0.47 助力代码创作！**：更新后，用户报告 [**Cursor 0.47**](https://www.cursor.sh/) 通过遵循自定义规则、符合 AI 自定义规则以及在代码生成方面表现更好（特别是针对 [VS Code 分支的欢迎窗口](https://code.visualstudio.com/api/ux-guidelines/welcome)），修复了 **Sonnet 3.7** 的问题。
   - 他们注意到使用了带有多个代码路径的顺序思维（sequential thinking）以提高速度。
- **Vibe Coding：得到验证与关注！**：关于 *vibe coding*（构建微型 SaaS Web 应用）的讨论促使一名用户使用 **Claude** 本身在 **VS Code 分支**上构建了一个欢迎窗口。
   - 一位成员将这种实践定义为 *将 AI Agent 视为需要引导的孩子*，并将其与构建需要编排经验的 Docker 化开源项目进行了对比。
- **MCP：最大化模型能力！**：成员们探索了使用 **Model Context Protocol (MCP)** 服务来增强像 Cursor 和 Claude 这样的 AI 代码编辑器，连接到诸如 Snowflake 数据库之类的服务。
   - 一位成员发现 **PearAI** 提供了完整的上下文，而另一位成员发现 **0.45** 以上版本的 Cursor 往往会忽略 `.cursorrules`。
- **Cursor 错误激发周边灵感！**：臭名昭著的 **"Cursor is damaged and can't be opened"** 错误消息激发了周边商品的灵感，现在已有 [T 恤](https://www.redbubble.com/shop/ap/169071328) 和 [鼠标垫](https://www.redbubble.com/i/mouse-pad/Cursor-AI-Error-by-TheGalaxyStars/169071328.G1FH6) 出售。
   - 这一幽默的转变凸显了社区在面对技术挫折时寻找乐趣的能力。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ktransformers 声称 IQ1 碾压 BF16**：成员们讨论了 **ktransformers** 发布的 **Deepseek R1 IQ1** 基准测试，一位持怀疑态度的成员指出 *他们本该写 1.58bit 的*。
   - 另一位成员指出表格并不完整，基准测试仍在进行中，展示的是 **Deepseek v3（聊天模型）对比 R1 IQ1**。
- **Unsloth 的 GRPO 算法揭秘**：成员们推测 **Unsloth** 是如何实现低 VRAM 占用的，暗示了异步梯度卸载（asynchronous gradient offloading），然而真正的节省源于对 **GRPO** 数学实现的重构。
   - 其他效率提升来自中间层的梯度累积（gradient accumulation），结合梯度检查点（gradient checkpointing），以及更高效的算子（kernels）如 **logsumexp**。
- **QwQ-32B 生成循环问题已修复**：Daniel Han 发布了调试 **QwQ-32B** 模型循环问题的指南，建议在 llama.cpp 中使用采样器，例如 *--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"*，并上传了 [动态 4bit 量化版 & GGUFs](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit)。
   - 他表示 **QwQ 对量化也很敏感** —— *前几层和最后几层应该保持未量化状态*。
- **Qwen 的 RLHF 成功案例**：一位成员使用 **Unsloth GRPO** 在 **Qwen7b** 模型上成功实现了 **RLHF**，报告称在角色遵从度方面有显著提升。
   - 然而，该模型在 **IFeval** 等严格指令遵循基准测试中表现出 *明显的退化*，特别是在格式约束和负面指令方面。
- **RoPE Scaling 前来救场**：**Qwen2.5-Coder-3B-bnb-4bit** 模型可以处理 **32768** 的序列长度，但在使用 **kaiokendev 的 3.906 倍 RoPE scaling** 后，被扩展到了 **128000**。
   - 据确认，[通过 RoPE](link-to-rope-explanation)，只要架构是 transformers，就可以处理 128k tokens。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **注册表编辑可能导致蓝屏**：一位成员分享称，在发现某个关键 **.dll 文件**占用大量 RAM 并将其删除后，导致下次启动时出现 **蓝屏**，并警告在没有备份的情况下进行注册表编辑的风险。
   - 成员们普遍认为，在调整注册表后，*备份个人文件并重装系统（reformat）* 是明智之举。
- **量化影响性能和内存**：用户讨论了量化，指出更倾向于使用 **f16 量化** 以在更小的负载中容纳更多参数，同时也承认其他量化方式可能会导致 flash attention 崩溃。
   - 在量化的背景下，有人将浮点数描述为 *有符号位大小，因此有符号 16 位整数就是 32*。
- **Windows 存在文件路径长度限制**：成员们讨论了 **Windows 中的文件路径长度限制**，指出由于 Windows API 中的 **MAX_PATH** 定义，标准限制为 **260 个字符**，详见 [Microsoft 文档](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)。
   - 这一限制可以通过注册表和应用程序清单（manifest）修改，为每个应用程序启用长路径行为来绕过，从而达到 NT kernel 的 32,767 个字符路径限制。
- **Inception Labs 开发基于扩散模型的文本生成**：[Inception Labs](https://inceptionlabs.ai/) 正在开创基于扩散模型（diffusion-based）的语言生成技术，受 **Midjourney** 和 **Sora** 等 AI 系统的启发，该技术有望提供前所未有的速度、质量和生成控制。
   - 成员们注意到开源替代方案正在开发中，例如 [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)，并且这项技术可能很快会带来显著的速度提升。
- **去中心化信任认证（Trustless Authentication）是未来**：成员们讨论了 **trustless authentication** 的潜力，即通过将人脸 3D 扫描和身份证件转换为加密字符串来充当数字护照，类似于 [Persona](https://www.withpersona.com/) 使用的商业模式。
   - 它被设想为一个去中心化信任的验证数据库，文件中包含用户的个性化标签，生成的 deepfake 人脸扫描和身份证件将无法通过验证。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 遭遇稳定性冲浪挫折**：用户报告称 **Windsurf** 在最新更新后表现**不稳定**，指出存在**无限循环**、**重复代码更改**以及普遍的**无响应**等问题。
   - 许多用户正转向 **Cursor** 或 **Trae**，理由是它们具有更好的稳定性和资源管理，一位用户表示：“我决定切换到 Cursor... 估计更稳定。我会在一个月左右回来看看它的情况。”
- **额度危机：Windsurf 的高昂消耗引发批评**：由于 AI 反复重写代码或分析相同文件，用户正经历**高额度消耗**，导致不满。
   - 一位用户抱怨在引入 **Claude 3.7** 后，一天内烧掉了 **1200 flow credits**，称其“100% 不稳定，100% 钱打水漂”，并指出该工具一次只能读取 50 行，读取一个 81 行的文件就要消耗 2 个额度。
- **Cascade 终端问题频发**：部分用户报告 **Cascade 终端**消失或变得无响应，且没有可见的设置可以修复。
   - 一位用户提到了重启的临时解决方案，但问题会再次出现；另一位用户建议使用 `CTRL Shift P` 清除所有缓存并重新加载窗口。
- **模型之争：Cursor 在额度竞赛中击败 Codeium**：用户将 Windsurf 与 **Cursor** 进行对比，许多人发现 Cursor 在处理大文件时更高效、更稳定。
   - 一位用户报告称，在 **Claude 3.7** 下，Cursor 的额度消耗仅为 “Windsurf 的一小部分”，而另一位用户报告称 “Trae 在免费阶段的表现比 Windsurf 好 100 倍”。
- **MCP 乱象：用户在服务器设置中挣扎**：用户在尝试使用 **MCP 服务器**时遇到问题，出现与模型或配置相关的错误。
   - 一位用户收到了 “Not Found: Resource not found” 错误，尝试了不同模型但未获成功。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 会员权益引发问题**：多位用户报告其 **Perplexity Pro** 账户被意外取消，怀疑是诈骗，后来发现是因为不符合针对 **克罗地亚** 的 **Deutsche Telekom** 客户提供的 “DT 1 Year Free HR” 优惠资格，详见[服务条款](https://www.perplexity.ai/legal/terms)。
   - 官方支持确认取消是由于该优惠仅限 **克罗地亚** 用户，这让不符合条件的用户感到困惑。
- **GPT-4.5 分级讨论引发推测**：用户讨论了 **Perplexity Pro** 中 **GPT-4.5** 的可用性，确认**每 24 小时有 10 次免费使用机会**。
   - 明确了 **GPT-4.5** 的使用成本非常高，建议使用自动模型选择可能就足以避免手动挑选模型。
- **Complexity 扩展带来 Canvas 功能**：一位用户解释了如何使用 **Complexity 扩展**在 Perplexity 中生成 Mermaid 图表，提供了类似 Canvas 的功能。
   - 通过启用 Canvas 插件并提示 AI 创建 Mermaid 图表，用户可以通过点击代码块上的播放按钮进行渲染，更多信息见[此 Discord 链接](https://discord.com/channels/1245377426331144304/1246406910962438245/1347554810928566304)。
- **Google Gemini 2.0 抢占地盘**：来自 [ArsTechnica 的文章](https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/) 详细介绍了 **Google** 扩展由 **Gemini 2.0** 驱动的 **AI 搜索功能**。
   - **Google** 正在测试一种 **AI 模式**，用 **Gemini** 生成的回答取代传统的搜索结果。
- **黄金位置：AI 驱动的产品植入**：[Perplexity 页面](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew) 报道了 **Amazon Prime** 如何为其内容测试 **AI 配音**。
   - 其他 Perplexity 页面包括：[苹果的折叠屏 iPhone](https://www.perplexity.ai/page/apple-s-foldable-iphone-predic-WSdZuoG7Rw6VvayJJg0DVQ)、[OpenAI 的 AI Agent](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ) 以及 [DuckDuckGo 的 AI 搜索选项](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 修复模板 Bug 并加速 RAG**：**LM Studio 0.3.12** 现已发布稳定版，修复了 **QwQ 32B jinja 解析**相关的 Bug，该问题此前会导致 `OpenSquareBracket !== CloseStatement` 错误，详见 [完整发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.12)。
   - 此外，**RAG** 中用于检索的文件分块（chunking）速度得到了显著提升，MacOS 系统现在可以正确索引外部 exFAT 驱动器上的 MLX 模型。
- **Qwen Coder 在 M2 上略胜 DeepSeek**：对于在配备 16GB RAM 的 Macbook M2 Pro 上进行编码任务，由于内存限制，建议选择 **Qwen Coder** 而非 **DeepSeek v2.5**，但需承认其性能仍会落后于云端模型。
   - 成员们指出 **Qwen 32B** 的表现超出了其参数规模，性能至少与 **Llama 3.3 70b** 持平。
- **Unsloth 加速 LLM 微调**：推荐使用 [Unsloth](https://github.com/unslothai/unsloth) 来加速 **Llama-3**、**Mistral**、**Phi-4** 和 **Gemma** 等模型的微调，并降低内存占用。
   - 成员们指出，*微调比推理更耗费资源*，而 LM Studio 目前的公开路线图中尚未包含此功能。
- **9070XT 在原始速度上碾压 7900XTX**：一位用户对比了 **9070XT** 和 **7900XTX** 运行相同 **qwen2.5 coder 14b q8_0** 模型的情况，发现 **9070XT** 的运行速度约为 **44tok/sec**，首字延迟（time to first token）不到一秒；而 **7900XTX** 的速度为 **31tok/sec**，首字延迟达 **4秒**。
   - 一位用户认为，那些看到 **Vulkan** 性能更佳的人可能遇到了*驱动问题*，因为 *ROCm 应该远快于 Vulkan*。
- **原生 FP4 支持到来**：与旧款显卡相比，**9070** 显著提升了 **FP16** 和 **INT4/FP4** 的性能，**FP16** 为 **122 vs 389**，**INT4/FP4** 为 **122 vs 1557**。
   - 这标志着 **Nvidia** 和 **Radeon** 都已提供原生 **FP4** 支持。同时，社区还讨论了量化对模型质量的影响，特别是更小的量化尺寸与潜在质量损失之间的权衡。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **替代 Replit 的开源 Agent 方案引发讨论**：成员们讨论了具有类似 **Replit/Bolt** 的 Agent 功能的开源替代方案，以及用于直接将视频保存到 **Supabase bucket** 的 *export_to_video* 功能。
   - 社区讨论了具有集成 Agent 功能的商业产品的替代方案。
- **为 Gradio 应用提议 Dexie 封装**：一位用户提议为 **Gradio 应用** 提供 **Dexie 封装**，以便更轻松地访问 **IndexedDB**，并引发了关于将其实现为自定义组件的讨论。
   - 共享了 **Gradio 开发者 Discord 频道**的链接以供进一步讨论。
- **数据集引用争议发酵**：一位用户怀疑研究论文《[NotaGen: Symbolic Music Generation with LLM Training Paradigms](https://arxiv.org/abs/2502.18008)》使用了他们的 **Toast MIDI 数据集** 但未进行妥善引用。
   - 社区成员建议联系通讯作者，并为数据集生成 **DOI**，以确保学术软件能正确归属和识别该数据集。
- **寻求 OCR-2.0 微调指导**：一位成员请求关于微调 **OCR-2.0** 的指导，他们认为这是目前最新且最好的模型，并链接了一份 [SharePoint 文档](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4)。
   - 该成员询问了根据文档微调 **got/ocr-2.0** 模型的适当步骤。
- **解码 AI Agent：组件 vs 实体**：根据[此回复](https://discord.com/channels/879548961488179230/1201995434137206834/1208022892900941824)，**LLM** 是一个具有工具调用能力的组件，其本身不是 **Agent**；一个 **Agentic AI 模型** 需要同时具备这两者。
   - 关于 **检索增强生成 (RAG)** 系统是否可以被视为 **LLM** 与之交互的“环境”，争论仍在继续。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Perplexity API 引发版权担忧**：用户指出 **Perplexity API** 存在版权问题，因为它抓取受版权保护的内容，并注意到 [Perplexity 的 API 条款](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service) 将责任转嫁给了用户。
   - **OpenAI**、**Google Cloud** 和 **AWS Bedrock** 等替代方案提供 IP 补偿（IP indemnification），将风险转移给供应商，参见：[OpenAI 条款](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/customer-copyright-commitment)、[Google Cloud 条款](https://cloud.google.com/terms/generative-ai-indemnified-services,) 和 [AWS Bedrock 常见问题解答](https://aws.amazon.com/bedrock/faqs/)。
- **Sonar Deep Research 模型因错误而陷入困境**：成员报告 **Perplexity Sonar Deep Research** 频繁出现错误、高延迟（首个 Token 延迟高达 **241 秒**）以及异常高的推理 Token 计数。
   - 一位成员提到在没有输出的情况下出现了 **137k 推理 Token** 计数，而其他人则确认该模型最终趋于稳定。
- **Gemini Embedding Text Model 首次亮相**：新的实验性 **Gemini Embedding Text Model** (**gemini-embedding-exp-03-07**) 现已在 [Gemini API](https://ai.google.dev/gemini-api/docs/models/experimental-models) 中可用，性能超越了以往模型，参见 [Google AI 博客](https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/)。
   - 该模型目前在 **Massive Text Embedding Benchmark (MTEB) 多语言排行榜**上排名第一，并支持更长的输入 Token 长度。
- **OpenRouter 推理参数充满不一致性**：用户发现 OpenRouter 的推理参数存在不一致，一些模型被标记为支持推理，但 Endpoint 却不支持，而一些提供商则不返回推理输出。
   - 测试揭示了配置问题以及模型与 Endpoint 之间的差异，其中 **Cloudflare** 被指出缺少 **/completions endpoint**。
- **Claude 3.7 在处理俄语提示词时受阻**：一位用户报告 **Claude 3.7** 在处理**俄语提示词**时表现吃力，以英语回复并可能误解细微差别。
   - 该问题在使用 Cline 与 OpenRouter 时出现，表明问题可能源于 **Anthropic**，而非插件或 OpenRouter 本身。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Minion.ai 宣告终结**：成员报告 [Minion.ai](https://minion.ai) 已经 *死亡*，不要相信那些炒作，有人将其描述为 *那个有 4 个卡通形象、本应为你执行 Agent 任务的东西*。
   - 它的关闭凸显了 AI 初创公司的波动性，以及对营销能力进行批判性评估的必要性。
- **Google 扩展 Gemini Embedding 模型**：**Google** 为开发者发布了一个实验性的 **Gemini Embedding 模型**，在 [MTEB（多语言）上具有 SOTA 性能](https://x.com/officiallogank/status/1898081742767919384?s=46)。
   - 更新后的模型具有 **8K Token 的输入上下文长度**、**3K 维度的输出**，并**支持 100 多种语言**。
- **Claude Code 进入 IDE 竞技场**：成员讨论了将 **Claude code** 与 **cursor.sh** 以及 **VSCode+Cline / roo-cline** 进行比较，优先考虑代码质量而非成本。
   - 讨论引用了[之前的消息](https://discord.com/channels/822583790773862470/1075282825051385876/1346679448174596219)，表明对最佳编码环境的探索正在进行中。
- **AI Personas 成为迷因霸主**：引用一条 [推文](https://x.com/defiapes/status/1855657706205352035)，一位成员提到，常态将转向体现性格类型的 **AI PERSONAS**。
   - 该推文提到，**Agent 将竞相成为特定子群体 x、y 和 z 的主要面孔**，以完美捕捉废文博主（shitposters）、瘾君子、运动员、说唱歌手、自由主义者和迷因币（memecoin）玩家的精髓。
- **Agent-as-a-Service 是未来吗？**：成员们思考了 **Agent-as-a-Service** 模式的可行性。
   - 一位成员提出了一个位于 DoorDash 前端的 Bot 想法，称之为 *DoorDash MCP*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 压缩 ChatGPT 消息限制**：用户报告 **ChatGPT** 现在强制执行 **每周 50 条消息** 的限制，以防止服务滥用。
   - 成员们对这种不便感到沮丧，并讨论了绕过限制的方法。
- **HVAC 获得 LLM AI 助手**：一位成员分享了一个基于 **HVAC 安装手册** 训练的 **LLM** [YouTube 视频演示](https://youtu.be/oAiUAzKLe_Y)。
   - 另一位成员建议使用 [Mistral OCR](https://mistral.ai/) 来处理手册，称赞其能以低廉的价格识别复杂的字体和表格。
- **本地 LLM 展示出超越云端的潜力**：成员们讨论了在本地运行 **LLM** 与使用云服务的对比，其中一人主张本地 DIY **LLM** 相比“黑盒”云解决方案更具潜力。
   - 讨论内容包括在 **GPU 和 CPU** 之间拆分进程，以及**量化（quantization）**在减少内存需求方面的优势。
- **SimTheory 的主张引发质疑**：一位用户推广了 [SimTheory](https://simtheory.ai/)，称其以更低的价格提供比 **OpenAI** 更高的 **O1 消息上限**。
   - 其他成员表示怀疑，质疑他们如何在提供更高限制的同时还能压低 **OpenAI** 的价格。
- **模型趋向于模仿，错失机会**：模型倾向于紧密遵循请求模式，可能会忽略更好的方法，尤其是在模型**可控性（steerability）**增强的情况下。
   - 有人评价道：*当给出的代码是用锤子钉螺丝时，模型会照做，猜测我们知道自己想要什么*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 提倡推理指示器**：用户请求 **Aider** 增加一个功能，以指示推理模型何时正在进行推理，特别是针对 **Openrouter** 上的 **Deepseek R1** 或 **V3**。
   - 一位成员提议对 **Aider** 和 **litellm** 进行补丁，并引用了 **litellm** 中一个用于获取 **<think>** 标签内推理 token 的技巧。
- **Jamba 模型展示 Mamba-Transformer MoE 架构**：**AI21 Labs** 推出了 **Jamba 1.6 Large & Mini** 模型，声称其质量和速度优于开源模型，并在具有 **256K 上下文窗口** 的长上下文任务中表现强劲。
   - 该模型采用 **Mamba-Transformer MoE** 混合架构以提升成本效益，可以自托管部署或在 **AI21 SaaS** 中使用。
- **初创公司信赖 AI 编写的代码**：一位 **Y Combinator** 顾问提到，*当前这一批初创公司中，有四分之一的业务几乎完全基于 AI 编写的代码*。
   - 上下文未提供更多细节。
- **Copilot 封禁 Aider 用户**：一位用户报告因在 **Aider** 中*极轻量使用 copilot-api* 导致 **Copilot 账号被封禁**，并提醒其他用户注意。
   - 其他人推测了可能的原因，如账号共享或频率限制，并引用了 [copilot-api GitHub 仓库](https://github.com/ericc-ch/copilot-api/blob/master/README.md)。
- **QwQ-32B 挑战 R1**：一位成员分享了关于 **QwQ-32B** 的[链接](https://x.com/victormustar/status/1898001657226506362)，断言它永远改变了本地 AI 编程，并在家用环境下实现了 **SOTA** 性能。
   - 另一位成员指出，基准测试讨论显示 **QwQ-32B** 可能很出色，但不一定比 **R1** 更好，特别是考虑到 [Discord](https://discord.com/channels/1131200896827654144/1346923740667187502) 中提到的模型大小。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek 面临企业禁令**：许多公司出于安全担忧正在禁用 **DeepSeek**，尽管其具有开源特性，这引发了关于媒体影响力、中国政府影响以及对 **reviewable code** 和本地执行需求的讨论。
   - 一位成员指出 DeepSeek 就像 **Deep Research + Operator + Claude Computer**。
- **离散扩散模型获取比率**：一位成员建议讨论来自 [arxiv](https://arxiv.org/pdf/2310.16834) 的论文 *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*，并指出它可能是 [Inception Labs 产品](https://www.inceptionlabs.ai/about)的基础。
   - 进一步说明求解 **duals λ** 需要对 **Lambert W function** 进行优化，这在计算上可能效率低下，建议使用 **cvxpy** 和 **adjoint method**。
- **潜空间抽象缩短推理长度**：一篇新[论文](https://arxiv.org/abs/2502.03275)提出了一种混合推理表示，使用来自 **VQ-VAE** 的 **latent discrete tokens** 来抽象初始步骤并缩短推理轨迹。
   - 一位成员质疑这种 **latent reasoning** 是否仅仅是 **context compression**，另一位成员则开玩笑说可能需要电休克疗法 (ECT) 来防止 AI 领域滥用 *reasoning* 一词。
- **OpenAI 放缓对 AGI 的立场**：据 [the-decoder.com](https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/) 报道，**OpenAI** 正从期待突然的 **AGI** 突破转向将其发展视为一个持续的过程。
   - 这种转变可能是由于 **GPT 4.5** 反响平平的结果，一些人指责 **Sam Altman** 将预期定得过高。
- **采样技术遏制 Agent 循环**：在[这篇论文](https://arxiv.org/abs/2411.07641)和 [GitHub repo](https://github.com/Tomorrowdawn/top_nsigma) 中详细介绍的 **n-sigma-sampling** 的使用，似乎可以减轻多步 Agentic 工作流中的不良采样和循环行为。
   - 该技术无需复杂的概率操作即可高效过滤 token，无论 Temperature 缩放如何，都能保持稳定的采样空间。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **VSCode 在 Copilot 中拥抱 MCP**：**VSCode** 计划在 **GitHub Copilot** 中添加 MCP 支持，正如其[直播](https://youtu.be/Pe8ghwTMFlg)所示。
   - 社区成员正在讨论开源和闭源 MCP 实现的影响。
- **MCP 服务器面临提示词注入威胁**：有人担心 **MCP servers** 向 AI Agent 提供恶意的提示词注入，利用模型对工具调用（tool calls）相对于内部知识的信任。
   - 建议的缓解措施包括在首次使用时显示工具描述列表，并提醒用户任何指令更改以防止漏洞利用。
- **安全频道针对 MCP 漏洞利用**：社区成员考虑创建一个安全频道来主动预防漏洞利用，强调了在不了解后果的情况下将工具连接到 **MCP Servers/Remote Hosts** 的风险。
   - 讨论强调了合规工具描述可能诱导模型包含后门的潜力。
- **Python MCP 快速入门困扰用户**：用户报告了运行 **Python MCP quickstart** ([此处](https://modelcontextprotocol.io/quickstart/client)) 时的错误，导致推荐使用 [wong2/mcp-cli](https://github.com/wong2/mcp-cli) 作为更好的替代方案。
   - 该替代方案被认为更易于使用，且对于初始 MCP 设置更可靠。
- **Swagger 端点变得 MCP 友好**：一位成员正在开发 **mcp-openapi-proxy**，将任何 swagger/openapi 端点转换为可发现的工具，模仿其 **mcp-flowise server** 的设计。
   - 他们最新的 mcp-server [在 5ire 中运行](https://5ire.org/)，但在 Claude 桌面版上遇到问题，表明跨不同平台存在兼容性挑战。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的灵活性引发辩论**：关于 **Mojo 的动态性**及其对性能影响的讨论，引发了对使用场景以及灵活性与速度之间平衡的思考，参考了[这篇 HN 帖子](https://news.ycombinator.com/item?id=35811170)。
   - 有建议认为，动态性应该只在被主动使用时才产生性能损失，但人们仍然担心即使没有类（class），它也可能对 struct 的性能产生潜在影响。
- **Monkey Patching 面临审查**：社区探索了在 Mojo 中实现动态行为的 **monkey patching** 替代方案，例如函数指针或组合，并指出它*更慢、更难理解、破坏静态分析工具，且通常无法完成正确的多态性（polymorphism）所不能完成的事情*。
   - 讨论强调，过度的 monkey patching 会导致代码难以理解并干扰静态分析，认为其效用并不足以抵消其缺点。
- **Python 库移植挑战**：成员们应对了**将 Python 库移植到 Mojo** 的障碍，特别是在动态性和全局状态方面，建议对性能关键组件利用 **CPython interop**。
   - 成员们对移植严重依赖 Python 动态性和全局状态的库表示担忧，特别是当该库无法以其他方式运行时。
- **协议多态性（Protocol Polymorphism）受到关注**：强调了**协议多态性**在不使用类树的情况下实现多态性的优势，并引用了 [PEP 544](https://peps.python.org/pep-0544/) 以实现无类层次结构的多态性。
   - 一些成员支持使用函数指针哈希表来实现动态行为，并倾向于在 Mojo 的单元测试中使用静态类型和所有权规则。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SOTA Agent 方法在算法上很简洁**：成员们讨论了 Arxiv 上 **SOTA Agent 方法**往往涉及相对简单的算法，类似于一个小型的**状态机（state machine）**。
   - 这意味着复杂的框架抽象可能是不必要的，建议使用更简单的数据、状态和 API 调用管理抽象就足够了。
- **Triton Autotune use_cuda_graph 引起困惑**：一位成员寻求关于 `triton.autotune` 中 `use_cuda_graph` 参数的澄清，不确定它如何应用于单个 **CUDA kernel**。
   - 困惑源于 **CUDA graph** 通常优化的是一系列 kernel 启动序列，这与 `triton.autotune` 的单 kernel 作用域形成对比。
- **Nvidia 的 NCCL AllReduce 使用双二叉树实现，超越环形拓扑**：[NVIDIA 博客文章](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)表明，与 AllReduce 操作的 **2D 环形延迟**相比，**NCCL 2.4** 中的**双二叉树（double binary trees）**提供了全带宽和更低的对数延迟。
   - NCCL 2.4 中增加了双二叉树，*提供了全带宽和甚至低于 2D 环形延迟的对数延迟。*
- **WoolyAI 发布用于 GPU 使用的 CUDA 抽象层 Beta 版**：**WoolyAI** 为其 [CUDA 抽象层](https://docs.woolyai.com)发布了 Beta 版，该层将 Kernel Shader 执行与 CUDA 应用程序解耦，将应用程序编译为新的二进制文件，并将 shader 编译为 **Wooly 指令集**。
   - 这实现了动态调度工作负载以优化 **GPU** 资源利用率；他们目前支持 **PyTorch**。
- **Cute Kernels 自动调优 CUDA 和 Triton Kernel**：一位成员发布了 [Cute Kernels](https://github.com/mayank31398/cute-kernels)，这是一个*通过对 **Triton** 和 **CUDA** 实现进行自动调优来加速训练的 kernel 集合*，并且可以自动分发到 **cutlass** 或 **Triton**。
   - 该实现已用于生产环境以训练 **IBM 的 Granite LLM**，因为 *LLVM 编译器有时可以生成比 NVCC 更高效的代码*，因此在 kernel 后端进行调优是有意义的。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **开源 AI 项目寻找新成员**：一位新成员正在寻求关于 **LLM pre-training**、**post-training**、**RL** 和**可解释性 (interpretability)** 等领域有趣的**开源 AI 项目**推荐，并为对理论工作感兴趣的人推荐了 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt/)。
   - 另一位成员建议，如果该新成员对 pre-training 感兴趣，最值得参与的项目是 **GPT-NeoX** 训练库。
- **Token Assorted 论文的词表存疑**：一位成员推测，**Token Assorted 论文**可能只是在针对 latent codes 进行 next token prediction 的 fine-tuning 过程中，将 **codebook** 添加到了其词表中。
   - 他们批评这种方法可能无法推广到开放推理领域，并建议在推理语料库中寻找 K 个最常见的字符串可能会产生更好的结果。
- **TorchTitan 分片需要 All-Reduce**：在关于 **TorchTitan** embedding 分片的讨论中，有人解释说，在原生 **TP** 下，如果输入 embedding 在 vocab 维度上进行了分片，则之后需要进行 **all-reduce**，以处理 embedding 层对缺失词汇元素输出为 0 的情况，这在 [GitHub 的此 issue](https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139) 中得到了澄清。
   - 成员们讨论了 embedding 层为缺失 token 输出零的存储影响，指出通信后立即需要存储空间，但如果可以重用该存储，则是免费的。
- **Logit Lens 揭示了 Llama-2 的语言偏见**：一位成员分享了他们对[这篇关于多语言语言模型的论文](https://arxiv.org/abs/2402.10588)以及使用 **Logit Lens** 分析 **Llama-2 系列**的赞赏。
   - 该论文探讨了在以英语为主的数据上训练的 **Llama-2** 模型如何处理非英语 prompt，揭示了模型在调整到输入语言之前，最初会倾向于英语翻译的阶段。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **yWorks 展示实时知识图谱可视化**：@yworks 展示了其用于可视化知识图谱的 SDK **yFiles**，提供[实时更新和动态交互](https://t.co/mb6M2R3TTh)。
   - 该演示强调了实时动态更新和交互知识图谱的能力。
- **Anthropic Cookbook 扩展**：LlamaIndex 团队更新并扩展了他们的 **Anthropic Cookbook**，这是学习通过简单的 completion 和 chat 方法进行[基础 API 设置](https://t.co/SQQ63qmwRb)的权威资源。
   - 该 Cookbook 是在 LlamaIndex 框架内设置和利用 **Anthropic API** 的全面指南。
- **调试 SQLTableRetrieverQueryEngine 的 Prompt 打印**：一位成员询问如何打印 LlamaIndex 中 **SQLTableRetrieverQueryEngine** 使用的 prompt，另一位成员分享了代码片段 `from llama_index.core import set_global_handler; set_global_handler("simple")` 以启用 prompt 打印。
   - 该解决方案为调试和理解 **SQLTableRetrieverQueryEngine** 在查询执行期间使用的 prompt 提供了一种实用方法。
- **Jina AI 包遇到安装问题**：一位成员报告了 **Jina AI** 包的导入错误，建议使用 `npm install @llamaindex/jinaai` 安装 provider 包，并附上了 [LlamaIndex 迁移文档](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9)的链接。
   - 迁移文档解释了 v0.9 中向 provider 包的转变，通过确保正确的包安装来解决导入错误。
- **LlamaExtract Beta 版吸引早期采用者**：一位成员请求访问 **LlamaExtract** 的 beta 版本，他们被告知需私信（DM）特定用户并提供电子邮件，同时参考了 [LlamaExtract 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started)。
   - 该文档概述了 **LlamaExtract** 的入门流程，并为潜在的 beta 测试人员重点介绍了关键功能。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B 推理速度拖慢？**：一位用户报告在 Hugging Face 上使用 **Command R7B** 时推理时间较慢，这被归因于用户**硬件**配置不佳或模型执行效率低下。
   - 成员澄清说，问题可能源于用户的设置，而非 **Command R7B** 本身的性能问题。
- **Ollama 工具问题困扰用户**：一位用户在 **Ollama** 和 **Langchain** 中使用 `command-r7b:latest` 进行**工具调用 (tool invocation)** 时遇到问题，收到关于缺少工具访问权限的错误。
   - 指导建议确保工具传递使用正确的 **JSON 格式**，并验证 **Ollama** 的配置是否支持工具调用。
- **开发者寻求开源 AI 机会**：一位具有 **GPT-2 预训练**和在 **Hellaswag** 上进行模型**微调 (fine-tuning)** 经验的开发者正在寻找有趣的开源 AI 项目进行贡献。
   - 该成员还对在不列颠哥伦比亚省温哥华地区的社交活动感兴趣。
- **504 Gateway 错误再次出现！**：用户报告了反复出现的 **504 Gateway Errors** 和 **502 Server Errors**，表明存在暂时的服务器端问题。
   - 错误消息建议重试请求，表明这些问题是瞬态的。
- **图谱增强主题模型效果**：一位成员建议使用**知识图谱 (Knowledge Graphs)** 来增强**主题建模 (topic modelling)**，并特别推荐了来自 **TogetherAI** 的 **LLM**。
   - 他们强调了 **TogetherAI** 提供的慷慨免费额度，鼓励在他们的平台上尝试**主题建模**任务。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **播客克隆流程简化**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=0UWYFFOPjqs)，展示了 **NotebookLM** 与 **Wondercraft** 结合使用，通过专业克隆的声音创建播客，简化了播客制作流程。
   - 评论认为 **Wondercraft** 比 **11Labs** 和 **HeyGen** 提供了更精简的方法，尽管其订阅价格对于非营利性播客来说可能较高。
- **关于 Drive 数据加密的讨论**：一位用户指出，虽然数据在传输到 **Google Drive** 期间是加密的，但在 **Drive** 本身存储时并*未*加密。
   - 这意味着 **Google**、成功的黑客以及用户与其共享目录的人员都可以访问这些数据。
- **AI 语音像真人一样结巴**：一位用户附上了一个 wav 文件示例，指出生成的音频文件中的 **AI 发言人**现在会*像普通人一样结巴*，听起来非常自然。
   - 用户还提到，结巴增加了音频长度，可能会在 **Google** 的每日限制内减少实际传达的信息量。
- **解锁分析师/指南聊天设置对时间线生成的影响**：一位用户发现 **NotebookLM** 中的**聊天设置**（如*分析师/指南、简短/详细*）会影响时间线和概览的生成，因为主持人在生成音频概览时实质上是请求了这些设置。
   - 该用户还指出，他们的助手会将简报概览、详细时间线、角色表和原始资料合并为一个文档。
- **NotebookLM 获得 Chrome 扩展支持**：成员们讨论了向 **NotebookLM** 上传 URL 列表以将每个 URL 添加为来源的可能性，并提到有几个 [Chrome 扩展](https://chromewebstore.google.com/search/notebooklm) 可用于此目的。
   - 这些扩展包括 [NotebookLM Web Importer](https://chromewebstore.google.com/detail/notebooklm-web-importer/ncjabfmpppgonojpohbfhfaahfpkgihc)、[NotebookLM Toolbox](https://chromewebstore.google.com/detail/notebooklm-toolbox/nbpckfdmlndgcoaklokdbllhelmfhoal) 和 [NotebookLM YouTube Turbo](https://chromewebstore.google.com/detail/notebooklm-youtube-turbo/mjpdncbmeognfgjkcdnglfomkmnknpgk)。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Batch 函数探讨并行工作**：一位成员询问 **DSPy 的 batch 函数**是否可以将并行工作委托给运行两个相同 LLM 实例的 **vllm 后端**，并引用了参数 *num_threads*、*max_errors* 和 *return_failed_examples*。
   - 建议如果 VRAM 足够，应增加 *num_sequences* 或 *pipeline_parallel_size*，而不是使用单独的 API。
- **VLLM 流水线并行实现负载均衡**：在 **vllm 设置**中将 *pipeline parallel size* 设置为 2 后，一位成员确认 **vllm 会处理负载均衡**。
   - 鼓励进行基准测试，以比较处理时间与非并行方法的差异。
- **提议为负载均衡创建 LM 子类**：一位成员提议，如果两个实例位于不同的节点上，可以创建一个 **LM** 的子类来进行负载均衡，因为 **DSPy** 原生不支持此功能。
   - 虽然可以使用**代理 (proxy)** 转发请求，但从 **vllm** 端解决被认为是更好的方法。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **字符串替换代码片段受到关注**：一名成员在 `general-chat` 中发布了一个用于字符串替换的 PHP 代码片段：`$cleanedString = str_replace(['!', '@', '#', '$', '%', '^'], '', "This is a test string!@#$%^");`。
   - 该代码用于移除字符串中的特殊字符。
- **笔记本电脑摔落导致损坏**：一名成员报告称他们的笔记本电脑在*摔了一跤*后损坏了。
   - 他们形容损坏情况*不容乐观*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad JIT 耗时 30 分钟**：有用户报告 **tinygrad JIT** 编译器花费了超过 30 分钟来编译一个 2 层模型。
   - 消息中未包含任何解决方案。
- **Tinygrad Loss 变为 NaN**：有用户询问在 tinygrad 中初始步（step 0）后 **loss 变为 NaN**（非数字）的原因。
   - 消息中未包含任何解决方案。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 计划增加音频支持**：成员们讨论了未来将**音频模态 (audio modality)** 引入 **Torchtune** 的高层计划。
   - 目前还没有具体的时间表或技术细节，表明其仍处于早期规划阶段。
- **Torchtune 音频支持仍在计划中**：团队继续表示他们计划在未来为 **Torchtune** 添加**音频模态支持**。
   - 截至目前，具体的时间表和技术细节尚未确认，这表明该功能仍处于初步规划阶段。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents (Berkeley MOOC) 链接请求**：claire_csy 请求一个有效的 **LLM Agents Berkeley MOOC** 讨论链接，因为之前的链接已过期。
   - 用户寻求访问 **MOOC 课程讨论**以进行进一步参与。
- **需要 LLM Agents MOOC 的有效链接**：用户 claire_csy 报告称，现有的 **LLM Agents MOOC 课程讨论**链接已过期。
   - 该请求强调了为对 **Berkeley MOOC** 感兴趣的参与者提供更新且可访问链接的需求。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Diffusion LLMs 的热度引发辩论**：一名社区成员询问了关于 **Diffusion LLMs**（特别是 **Mercury 模型**）发布的炒作，以及它是否会取代**基于 Transformer 的模型**。
   - 他们提到阅读了白皮书但发现难以理解，正在寻求社区专家的见解。
- **LLaDA 模型：基于 Diffusion 的 LLM 范式转变**：一名社区成员分享了 [diffusionllm.net](https://diffusionllm.net/) 的链接，解释说 **Large Language Diffusion Models** (**LLaDA**) 代表了语言模型架构的一种新范式。
   - 他们阐明，与传统的**自回归 (AR) Transformers** 不同，**LLaDA** 使用去噪扩散过程以并行的、由粗到细的方式生成文本。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将移除它。

---

# 第二部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1347524353088163850)** (1144 messages🔥🔥🔥): 

> `Cursor vs Lmarena, Cursor 0.47, Claude 3.7, Grok struggles, vibe coding`

- **Cursor 在 One-Shot 对决中被捕获！**: 成员们使用 **GPT-3.7** 的单条提示词对比了 [**Cursor**](https://www.cursor.sh/) 和 [**Lmarena**](https://lmarena.ai/)，初步印象在输出质量上更倾向于 Cursor，理由是 Lmarena 生成的文本难以阅读。
   - 随后其他人的分析表明 **Lmarena** 更好地遵循了提示词的主题，但普遍共识是*两者都表现得一塌糊涂*。
- **Cursor 0.47 平息编码混乱！**: 更新后，用户报告 [**Cursor 0.47**](https://www.cursor.sh/) 通过遵循自定义规则、符合 AI 自定义规则以及在代码生成方面表现更好（特别是针对 [VS Code fork 的欢迎窗口](https://code.visualstudio.com/api/ux-guidelines/welcome)），修复了 **Sonnet 3.7** 的问题。
   - 他们注意到使用了具有多个代码路径的顺序思维（sequential thinking）以提高速度。
- **Vibe Coding 受到关注并获得验证**: 关于 *vibe coding*（构建微型 SaaS Web 应用）的讨论促使一名用户直接使用 **Claude** 在 **VS Code fork** 上构建了一个欢迎窗口。
   - 一位成员将这种实践定义为*将 AI Agent 视为需要引导的孩子*，并将其与需要编排经验的 Dockerized 开源项目构建进行了对比。
- **MCP 魔法：最大化模型能力**: 成员们探索了使用 **Model Context Protocol (MCP)** 服务来增强 Cursor 和 Claude 等 AI 代码编辑器，并连接到 Snowflake 数据库等服务。
   - 一位成员发现 **PearAI** 提供了完整的上下文，而另一位成员发现 `.cursorrules` 在 **0.45** 以上版本的 Cursor 中往往会被忽略。
- **错误信息变身周边爆款**: 臭名昭著的 **"Cursor is damaged and can't be opened"** 错误信息激发了周边商品的灵感，现在已有 [T 恤](https://www.redbubble.com/shop/ap/169071328) 和 [鼠标垫](https://www.redbubble.com/i/mouse-pad/Cursor-AI-Error-by-TheGalaxyStars/169071328.G1FH6) 出售。
   - 这一喜剧性的转变突显了社区在面对技术挫折时寻找幽默的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://container-seven-sigma.vercel.app">Container Ejection Simulator</a>: 未找到描述</li><li><a href="https://protocraft.ai">Protocraft AI</a>: Protocraft：专为软件开发、任务自动化、创意探索和 Prompt 自动化设计的 AI 数字工作室，由您自己的 API 密钥和本地 LLM 驱动。</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.47.0-c2804e658d8fe4c072e20cb39c56d7eed1b6f43e.deb.glibc2.25-x86_64.AppImage">未找到标题</a>: 未找到描述</li><li><a href="https://artificialanalysis.ai">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://techcrunch.com/2025/03/06/chatgpt-on-macos-can-now-directly-edit-code/">ChatGPT on macOS can now directly edit code | TechCrunch</a>: ChatGPT，OpenAI 的 AI 驱动聊天机器人平台，现在可以直接编辑代码——前提是你在使用 macOS。</li><li><a href="https://trypear.ai">PearAI - The AI Code Editor For Your Next Project</a>: PearAI 是一款开源 AI 代码编辑器，具有 AI 聊天、PearAI Creator 和 AI 调试等强大功能，助你打造心仪之作。</li><li><a href="https://www.youtube.com/watch?v=yOKwK-iIg3M"> - YouTube</a>: 未找到描述</li><li><a href="https://x.com/mishalaskin/status/1898048925157728601?s=46">Misha Laskin (@MishaLaskin) 的推文</a>: 今天我和我的朋友兼联合创始人 @real_ioannis 一起发布了 @reflection_ai。我们的团队在 RL 和 LLM 领域开创了重大进展，包括 AlphaGo 和 Gemini。在 Reflection，我们正在构建超智能...</li><li><a href="https://devproai.com/index.htm">DevProAI</a>: 未找到描述</li><li><a href="https://www.continue.dev">Continue</a>: 赋能开发者，AI 增强开发 · 领先的开源 AI 代码助手。你可以连接任何模型和任何上下文，在 IDE 内部构建自定义的自动补全和聊天体验。</li><li><a href="https://x.com/rohanpaul_ai/status/1878063926866493675">Rohan Paul (@rohanpaul_ai) 的推文</a>: 🔥 OpenAI 封禁了一名将 ChatGPT API 武器化的开发者。该开发者构建了一个可以使用 ChatGPT Realtime API 响应语音指令的项目。OpenAI 证实了此次封禁，理由是...</li><li><a href="https://tenor.com/view/rick-and-morty-i-can-answer-that-for-money-gif-10573903">Rick And Morty I Can Answer That For Money GIF - Rick And Morty I Can Answer That For Money - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/oslook/cursor-ai-downloads">GitHub - oslook/cursor-ai-downloads: 所有 Cursor AI 的官方下载链接，包括最新版本和旧版本，方便你升级、降级和选择任何版本。🚀</a>: 所有 Cursor AI 的官方下载链接，包括最新版本和旧版本，方便你升级、降级和选择任何版本。🚀 - oslook/cursor-ai-downloads</li><li><a href="https://www.redbubble.com/shop/ap/169071328">Redbubble logo</a>: 未找到描述</li><li><a href="https://www.redbubble.com/shop/ap/169071257">Redbubble logo</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1347532901914513458)** (254 条消息🔥🔥): 

> `用于训练大模型的 GPU 显存管理，ktransformers IQ1 基准测试，QwQ-32B 优化与最佳实践，GRPO 算法优化`

- **模型部署的多 GPU 扩展性探讨**：成员们讨论了在多个节点上部署大型模型的实用性，指出仅将计算推送到 **1-2 个节点**是低效的，且在未达到饱和的情况下过度分配计算资源会导致利用率不足。
   - 对话涉及了 **Google** 对大型 **MoE** 模型的使用、具有低 VRAM 的 **TPUs**，以及对 **OpenAI** 和 **Anthropic** 大型多节点部署的推测。
- **ktransformers 声称 IQ1 优于 BF16**：成员们分享了 ktransformers 的 **Deepseek R1 IQ1** 基准测试，但有人表示怀疑，因为一位成员说 *他们本该写成 1.58bit 的*。
   - 另一位成员补充说表格不完整，基准测试仍在进行中，展示了 **Deepseek v3（对话模型）对比 R1 IQ1** 的结果。
- **Unsloth 的卸载算法揭秘**：成员们讨论了 **Unsloth** 如何在训练期间实现低 VRAM 占用，假设其涉及在 GPU 计算时将梯度逐层异步卸载（offloading）到 CPU。
   - 实际的节省似乎源于重新实现 GRPO 数学逻辑、中间层的梯度累积、梯度检查点（gradient checkpointing）以及更高效的内核（如 **logsumexp**）。
- **QwQ-32B 生成修复与优化**：Daniel Han 发布了关于调试 **QwQ-32B** 模型循环问题的指南，建议在 llama.cpp 中添加采样器，例如 *--samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc"*，并上传了 [动态 4bit 量化与 GGUF 文件](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit)。
   - 官方推荐设置为 *temperature = 0.6, top-k = 40, min-p = 0.1, top-p = 0.95*，但也提到 **QwQ 对量化非常敏感** —— *前几层和最后几层应保持不量化状态*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/long-context">Unsloth 梯度检查点 - 4 倍长的上下文窗口</a>：Unsloth 梯度检查点（Gradient Checkpointing）现在支持超长上下文窗口的 LLM 微调，Llama 3 可达 228K。我们成功地将内存占用进一步降低了 30%，代价是仅增加了 1.9% 的额外时间...</li><li><a href="https://unsloth.ai/blog/grpo">长上下文 GRPO (R1 推理)</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行 Unsloth 的 1.58-bit 动态 GGUF 版本。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practice">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/fp8_kernel.md">ktransformers/doc/en/fp8_kernel.md at main · kvcache-ai/ktransformers</a>：一个用于体验前沿 LLM 推理优化的灵活框架 - kvcache-ai/ktransformers</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j5qo7q/qwq32b_infinite_generations_fixes_best_practices/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1898035752124166368">Daniel Han (@danielhanchen) 的推文</a>：QwQ-32B 出现无休止的重复？我制作了一个指南来帮助调试！当使用重复惩罚来对抗循环时，反而会导致循环！尝试在 llama.cpp 中添加：--samplers "...</li><li><a href="https://docs.unsloth.ai">欢迎 | Unsloth 文档</a>：初识 Unsloth？</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/rl_replacements.py#L52-L237">unsloth-zoo/unsloth_zoo/rl_replacements.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>：Unsloth 工具库。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做贡献。</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/rl_replacements.py#L52-L92">unsloth-zoo/unsloth_zoo/rl_replacements.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>：Unsloth 工具库。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做贡献。</li><li><a href="https://artificialanalysis.ai/models/qwq-32b">QwQ-32B - 智能、性能与价格分析 | Artificial Analysis</a>：阿里巴巴 QwQ 32B 的分析，并在质量、价格、性能（每秒 token 数和首个 token 时间）、上下文窗口等关键指标上与其他 AI 模型进行对比。</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa/unsloth_zoo/gradient_checkpointing.py#L137-L165">unsloth-zoo/unsloth_zoo/gradient_checkpointing.py at c54bf68b71abcd45e49cd077ea4b71dabe3ae6fa · unslothai/unsloth-zoo</a>：Unsloth 工具库。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1347534418738286623)** (82 messages🔥🔥): 

> `RLHF with Unsloth GRPO on Qwen7b, Qualitative vs Quantitative Improvement, Reward Model Bias, KL Divergence Issues, Qwen for Sudoku` 


- **Qwen7b 上的 RLHF 取得了角色扮演的成功，但基准测试失败**：一名成员报告了在 **Qwen7b** 模型上使用 **Unsloth GRPO** 成功运行 **RLHF** 的情况，指出在角色遵循和输出流畅度方面有显著改进。
   - 然而，该模型在 **IFeval** 等严格指令遵循基准测试中表现出*明显的退化*，特别是在格式限制和负面指令方面，并有人索要该 Notebook。
- **数据集缺失和“甜言蜜语”奖励损害基准测试性能**：发布者诊断认为，训练数据缺乏要求精确合规的示例，且奖励模型偏好*过度详细*的回答，这导致了问题。
   - 他们表示，*虽然这对“友好”互动有益，但在要求简洁、准确合规的语境下会变得有问题*。
- **KL 散度不稳定困扰 GRPO 训练**：另一名成员分享了在 **GRPO** 训练期间显示 **KL 散度** *峰值*的图表，询问是否出了问题，并发布了他们的超参数以获取反馈。
   - 建议将学习率调度器切换为 *constant*，并移除 **weight decay** 和 **warmup ratio**，以稳定训练过程并观察即时训练效果，同时积极进行梯度裁剪（clip gradients）。
- **Qwen 尽管推理懒散但在数独方面表现出色**：一位成员发现 **Qwen** 模型尽管表现出*愚笨且懒散*的推理，但仍能准确解决**数独（Sudoku）**谜题。
   - 虽然 **Qwen** 比其他模型更好地遵循指令，但它们并不一定更*聪明*；然而，它们仍然表现出一些错误。
- **排行榜模型因可疑行为被撤下**：一名成员注意到排行榜顶部的一个 **14B 模型**被撤下，作者要求将其移除。
   - 目前尚未进一步阐明具体情况。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1347535211918917745)** (114 messages🔥🔥): 

> `RAM Configuration for Mac Studio, ktransformers Performance, RoPE Scaling, Custom Datasets, Multi-GPU Parallelism with Unsloth` 


- **Mac Studio 的 RAM 限制**：一位用户询问大小为 **131GB** 的 **deepseek-r1 1.58bit 量化模型**是否可以在不进行磁盘交换的情况下装入 **128GB Mac Studio**。
   - 另一位在 **64 核 Threadripper** 上运行 **ktransformers** 的用户报告称分配了约 **12GB 的 VRAM**，并观察到它甚至没有用到 **22GB**。
- **RoPE 缩放扩展上下文长度**：提到 **Qwen2.5-Coder-3B-bnb-4bit** 模型可以处理 **32768** 的最大序列长度，但通过 **kaiokendev 的 3.906 RoPE 缩放**，可以扩展到 **128000**。
   - 确认[通过 RoPE](link-to-rope-explanation)，只要架构是 Transformers，就可以处理 128k tokens。
- **作业求助请求**：由于 3 小时后的作业截止日期，一位用户急需关于 **Phi4 模型**的自定义数据集和超参数调优的帮助。
   - 另一位成员回应道：*如果你在 3 小时内需要它“为了作业”，那你应该早就知道了……所以如果你失败了那是你自己的问题，这不是寻求帮助的方式*。
- **Unsloth 支持多 GPU 训练**：一位用户询问了 Unsloth 中多 GPU 并行的实现，引用了 [此 issue](https://github.com/unslothai/unsloth/issues/1908)。
   - 一名成员提到代码必须使用 **torchrun** 运行。
- **GRPO 训练需要特殊设置**：一位用户报告了在具有 **4 个 GPU** 的服务器上运行 **GRPO 训练**时出现的问题，指出尽管尝试调整设备映射（device map），每个 GPU 上仍创建了多个子进程。
   - 一名成员澄清说，**GRPO 不适用于指定的参数，目前仅适用于常规训练**。



**提及的链接**：<a href="https://github.com/unslothai/unsloth/issues/1908)">unslothai/unsloth</a>：微调 Llama 3.3, DeepSeek-R1 &amp; 推理 LLMs 速度提升 2 倍，显存占用减少 70%！ 🦥 - unslothai/unsloth

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1347534034124935233)** (6 messages): 

> `Diffusion Effect, Rust Code, Deepseek Coder v2, Unsloth and MoE` 


- **Diffusion Effect 留下深刻印象**：一位成员喜欢使用具有 **diffusion effect** 的模型，并对其性能表示赞赏。
   - 他们补充说，该模型在编写 **Rust code** 方面也非常出色。
- **Unsloth 集成 Deepseek Coder v2**：一位用户确认 **Unsloth** 已经集成了 **Deepseek Coder v2**。
- **Unsloth 在 MoE 方面存在困难**：一位成员指出 **Unsloth** 目前无法训练 **MoE** 模型。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1347524188436693063)** (264 messages🔥🔥): 

> `Registry Editing Risks, Quantization impact on RAM and VRAM, File path limitations on windows, Trustless authentication, Diffusion-based language models` 


- **编辑注册表导致蓝屏**：一位成员讲述了一次经历，在发现某个关键的 **.dll 文件**占用大量 RAM 后将其删除，导致下次启动时出现**蓝屏**，强调了在没有备份的情况下编辑注册表的风险。
   - 成员们普遍认为，在调整注册表后，*备份个人文件并重新格式化*是明智之举。
- **Quantization 影响性能和内存**：用户讨论了 quantization，一位成员指出他们更倾向于 **f16 quantization**，以便在较小的负载中容纳更多参数，同时承认其他量化方式可能会导致 flash attention 崩溃。
   - 在 quantization 的背景下，有人将浮点数描述为*其有符号位大小，因此有符号 16 位整数即为 32*。
- **Windows 文件路径限制为 260 个字符**：成员们讨论了 **Windows 中文件路径长度**的限制，指出由于 Windows API 中的 **MAX_PATH** 定义，标准限制为 **260 个字符**，正如 [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry) 中所解释的那样。
   - 通过注册表和应用程序清单修改，为每个应用程序启用长路径行为，可以绕过此限制，达到 NT 内核路径限制的 32,767 个字符。
- **Inception Labs 开创 Diffusion-Based 文本生成**：[Inception Labs](https://inceptionlabs.ai/) 正在开创基于 diffusion 的语言生成技术，该技术有望提供前所未有的速度、质量和生成控制，灵感来自 **Midjourney** 和 **Sora** 等 AI 系统。
   - 成员们注意到开源替代方案正在开发中，例如 [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)，并且这项技术可能很快会带来显著的速度提升。
- **讨论去中心化信任（Trustless）身份验证模型**：成员们讨论了 **trustless authentication** 的潜力，即通过将人脸 3D 扫描和身份证件转换为加密字符串来充当数字护照，类似于 [Persona](https://www.withpersona.com/) 使用的商业模式。
   - 它被设想为一个去中心化信任的验证数据库，文件中包含用户的个性化标签，生成的 deepfake 面部扫描和身份证件将无法通过验证。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/multimodalart/LLaDA">LLaDA - a Hugging Face Space by multimodalart</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry">Maximum Path Length Limitation - Win32 apps</a>: 从 Windows 10 版本 1607 开始，许多常见的 Win32 文件和目录函数已移除 MAX_PATH 限制。但是，您的应用必须选择加入以支持新行为。</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings.">Frequently Asked Questions</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可商用。 - nomic-ai/gpt4all</li><li><a href="https://github.com/ML-GSAI/LLaDA">GitHub - ML-GSAI/LLaDA: Official PyTorch implementation for &quot;Large Language Diffusion Models&quot;</a>: "Large Language Diffusion Models" 的官方 PyTorch 实现 - ML-GSAI/LLaDA</li><li><a href="https://inceptionlabs.ai/">Inception Labs</a>: 我们正在利用 diffusion 技术开发新一代 LLM。我们的 dLLM 比传统的自回归 LLM 更快、更高效。而且 diffusion 模型更准确...</li><li><a href="https://tenor.com/view/fun-cave-men-old-kick-fuck-you-gif-13869846">Fun Cave Men GIF - Fun Cave Men Old - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1347583701403439104)** (7 条消息): 

> `IDE Telemetry 设置，Codeium 网站支付更新` 


- **Telemetry 故障排查**: 一位成员报告称，由于 Visual Studio Code 和 Jetbrains IDE 中的 **IDE Telemetry** 设置，导致 Chat 功能被禁用。
   - 另一位成员建议在 VS Code 中启用 **code telemetry** 并将 Telemetry 设置为 **on**，并指向了 [这个 Reddit 帖子](https://www.reddit.com/r/Codeium/comments/1f4ljqf/unable_to_use_chat_ide_telemetry/) 以获取详细说明。
- **Codeium 网站支付信息更新问题**: 一位成员询问是否有人在更新 Codeium 网站上的 **payment information** 时遇到困难。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1347524326290886697)** (238 条消息🔥🔥): 

> `Windsurf 稳定性问题、额度消耗、Cascade 问题、模型性能对比 (Cursor vs. Windsurf)、MCP 服务器问题` 


- **Windsurf 深陷稳定性困扰**：用户反馈 Windsurf 在最新更新后表现**不稳定**，存在**死循环、重复代码修改**以及普遍的**无响应**等问题。
   - 许多用户正转向 **Cursor** 或 **Trae**，理由是它们具有更好的稳定性和资源管理。一位用户表示：“我决定切换到我从未用过的 Cursor……大概会更稳定。我会在一个月左右后再来看看它的情况。”
- **额度危机：Windsurf 的高昂消耗引发批评**：由于 AI 反复重写代码或分析相同文件，用户正面临**高额度消耗**，这导致了不满。
   - 一位用户抱怨在 **Claude 3.7** 推出后，一天内消耗了 **1200 个 flow 额度**，并称其为“100% 不稳定且 100% 冤枉钱”，指出该工具一次只能读取 50 行，读取一个 81 行的文件就要消耗 2 个额度。
- **Cascade 混乱：用户反馈终端故障**：部分用户报告 **Cascade 终端**消失或变得无响应，且没有明显的设置可以修复。
   - 一位用户提到了重启的临时解决方案，但问题会再次出现；另一位用户建议使用 `CTRL Shift P` 并清除所有缓存并重新加载窗口。
- **模型之争：Cursor 在额度竞赛中碾压 Codeium**：用户将 Windsurf 与 **Cursor** 进行对比，许多人发现 Cursor 更高效、更稳定，尤其是在处理大文件时。
   - 一位用户表示“Cursor 像大佬一样轻松搞定了一个 3000 行的文件”，并报告 Cursor 的额度消耗“只是 Windsurf 在使用 Claude 3.7 时的零头”，而另一位用户则报告“Trae 在免费的情况下表现比 Windsurf 好 100 倍”。
- **MCP 乱象：用户在服务器设置上挣扎**：用户在尝试使用 **MCP 服务器**时遇到问题，出现与模型或配置相关的错误。
   - 一位用户收到了“Not Found: Resource not found”错误，尝试了不同的模型但均未成功。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/x1xhlol/v0-system-prompts-models-and-tools">GitHub - x1xhlol/v0-system-prompts-models-and-tools</a>：通过在 GitHub 上创建账号来为 x1xhlol/v0-system-prompts-models-and-tools 的开发做出贡献。</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://tenor.com/view/your-technique-is-out-of-this-world-bruno-tonioli-britains-got-talent-your-technique-is-incredible-your-technique-is-extraordinary-gif-864393750500369819">Your Technique Is Out Of This World Bruno Tonioli GIF</a>：点击查看 GIF</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://x.com/OpenAIDevs/status/1897700857833193955">OpenAI Developers (@OpenAIDevs) 的推文</a>：macOS 版 ChatGPT 现在可以直接在 IDE 中编辑代码。适用于 Plus、Pro 和 Team 用户。</li><li><a href="https://x.com/i/communities/1889976002400686208">GitHub - FixTweet/FxTwitter 的推文</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能。</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>：Codeium 对个人用户永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活部署。</li><li><a href="https://x.com/openai/status/1897702764471619657?s=46&t=B0TlaMZ0ShmwM-XdEqw2mg">OpenAI (@OpenAI) 的推文</a>：即将面向免费版、企业版和教育版用户推出 🪐 引用 OpenAI Developers (@OpenAIDevs)：macOS 版 ChatGPT 现在可以直接在 IDE 中编辑代码。适用于 Plus、Pro 和 Team 用户。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1347528332840796181)** (184 条消息🔥🔥): 

> `Perplexity Pro 账户问题、GPT-4.5 使用、Perplexity 的商业用途与版权、Sonnet 3.7 扩展性能、Perplexity 移动应用与 Claude` 


- **Perplexity Pro 权益被收回，用户感到痛苦**：几位用户报告称其 **Perplexity Pro** 账户被意外取消，部分用户怀疑是诈骗。
   - Perplexity 支持团队表示，取消原因是用户不符合“DT 1 Year Free HR”优惠资格，该优惠专供**克罗地亚**的 **Deutsche Telekom** 客户，并附上了[服务条款](https://www.perplexity.ai/legal/terms)链接。
- **GPT-4.5 免费层级，是真是假？**：用户讨论了 **Perplexity Pro** 中 **GPT-4.5** 的可用性，一名用户询问是否应该有 **10 次免费使用机会**。
   - 另一名用户确认**每 24 小时有 10 次使用机会**，并澄清其成本非常昂贵；还有用户表示 Auto 模型选择功能足以避免纠结于选择哪个模型。
- **商业 Perplexity 先驱者的版权困境**：一名用户询问了在商业 Web App 中使用 Perplexity 的版权影响，因为 Perplexity 会从受版权保护的资源和具有“禁止数据挖掘”政策的网站抓取信息。
   - 关于使用 **Perplexity API** 时侵犯版权的责任归属，讨论并未得出结论性答案。
- **Complexity 扩展创建 Canvas 功能**：一名用户解释了如何使用提供类似 Canvas 功能的 **Complexity 扩展**在 Perplexity 中生成 mermaid 图表。
   - 通过启用 Canvas 插件并提示 AI 创建 mermaid 图表，用户可以通过点击代码块上的播放按钮进行渲染，[Discord 链接](https://discord.com/channels/1245377426331144304/1246406910962438245/1347554810928566304)中包含更多信息。
- **Google Gemini 2.0 在 Google Search 中占据一席之地**：一名用户分享了来自 [ArsTechnica 的一篇文章](https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/)，详细介绍了 **Google** 扩展由 **Gemini 2.0** 驱动的 **AI 搜索功能**。
   - Google 正在测试一种 **AI Mode**，用 **Gemini** 生成的结果取代传统的“10 个蓝色链接”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-for-perplexity-pro">Hrvatski Telekom 为 Perplexity Pro 提供 20,000 个免费许可证</a>：克罗地亚电信为先进的 AI 助手 Perplexity Pro 提供 20,000 个免费许可证</li><li><a href="https://arstechnica.com/google/2025/03/google-is-expanding-ai-overviews-and-testing-ai-only-search-results/">你早该料到：Google 开始测试纯 AI 搜索结果</a>：AI Mode 可能是 Google 的未来，但目前还只是一个实验。</li><li><a href="https://www.thefastmode.com/technology-solutions/39682-hrvatski-telecom-offers-20-000-free-licenses-">Hrvatski Telekom 为 Perplexity Pro 提供 20,000 个免费许可证</a>：克罗地亚电信为先进的 AI 助手 Perplexity Pro 提供 20,000 个免费许可证</li><li><a href="https://tenor.com/view/stonks-up-stongs-meme-stocks-gif-15715298">Stonks Up Stongs GIF - Stonks Up Stongs Meme - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1347583826913788039)** (5 条消息): 

> `Apple 折叠屏 iPhone、OpenAI AI Agent、Amazon Prime AI 配音、DuckDuckGo AI 搜索` 


- **Apple 折叠屏 iPhone 预测发布**：一个 [Perplexity 页面](https://www.perplexity.ai/page/apple-s-foldable-iphone-predic-WSdZuoG7Rw6VvayJJg0DVQ)讨论了关于 **Apple 折叠屏 iPhone** 的预测。
- **OpenAI 的 20000 AI Agent**：一个 [Perplexity 页面](https://www.perplexity.ai/page/openai-s-20000-ai-agent-nvz8rzw7TZ.ECGL9usO2YQ)谈到了 **OpenAI 的 AI Agent**，并提到了 20000。
   - 没有其他可用细节。
- **Amazon Prime 测试 AI 配音**：一个 [Perplexity 页面](https://www.perplexity.ai/page/amazon-prime-tests-ai-dubbing-pHEI1t6XRn6DilTOLGBGew)介绍了 **Amazon Prime** 如何测试 **AI 配音**。
- **DuckDuckGo 的 AI 搜索选项**：一个 [Perplexity 页面](https://www.perplexity.ai/page/duckduckgo-s-ai-search-option-D2sL.5w8S4mQYdr_XAlgjw)分享了关于 **DuckDuckGo AI Search** 选项的信息。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1347749343087689822)** (1 messages): 

> `LM Studio 0.3.12, QwQ template bug fixes, RAG chunking speed improvement` 


- **LM Studio v0.3.12 发布！**: LM Studio 0.3.12 现已作为稳定版本发布，包含 Bug 修复和性能改进。查看 [完整发布说明](https://lmstudio.ai/blog/lmstudio-v0.3.12)。
   - 您可以通过应用内更新或从 [LM Studio 官网](https://lmstudio.ai/download) 进行升级。
- **QwQ 模板解析 Bug 已修复**: 最新的 LM Studio 版本修复了一个 **QwQ 32B jinja 解析 Bug**，该 Bug 曾抛出 `OpenSquareBracket !== CloseStatement` 错误，现在不再有奇怪的报错。
   - 同时也解决了由于找不到附件导致无法删除对话的 Bug。
- **RAG 分块速度提升**: 在最新版本中，用于 **RAG** 检索的文件分块（chunking）速度得到了显著提升。
   - 下载到外部 exFAT 驱动器的 MLX 模型现在可以在 MacOS 上被正确索引。



**提到的链接**: <a href="https://lmstudio.ai/blog/lmstudio-v0.3.12">LM Studio 0.3.12</a>: Bug 修复和 RAG 文档分块速度改进

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1347546479564165284)** (104 messages🔥🔥): 

> `Open Source LLM for Coding on M2 Macbook Pro, DeepSeek v2.5 1210, Qwen Coder, Finetuning Large Language Models, Context Length and Memory Management` 


- **推荐在 Macbook M2 上使用 Qwen Coder 进行编程**: 对于在配备 16GB RAM 的 Macbook M2 Pro 上进行编程任务，建议使用 **Qwen Coder**，尽管其性能会明显低于 Claude 或 DeepSeek 等云端模型。
   - 最初推荐了 **Deepseek v2.5**，但成员们一致认为 16GB 内存没有足够的空间运行它。较小的 **7B** 模型可能可行，但质量可能不尽如人意。
- **Unsloth 提供 LLM 微调**: 成员们讨论了 LLM 的微调，推荐使用 [Unsloth](https://github.com/unslothai/unsloth) 以更少的内存占用更快地微调 Llama-3、Mistral、Phi-4 和 Gemma 等模型。
   - 有人提到 *微调比推理（inference）更消耗资源*，而 LM Studio 目前的公开路线图中尚未包含此功能。
- **提出 Token 上下文打包与解包的想法**: 一位成员提出了一种通过在 VRAM 中打包和解包文本来管理上下文长度的方法，将其大小减少到约 30%，以便高效浏览和检索相关信息。
   - 然而，其他成员澄清说，上下文是以 **Tokens**（平均 1/2 个单词）而非纯文本形式存储的，因此这种 **压缩** 逻辑不会带来显著收益，并建议 **摘要（summarization）** 或 **RAG** 可能是更好的替代方案。
- **Draft Models 提升 Mistral 的 Token 产出**: 使用相同模型的 **i1-IQ1_S** 量化版本作为 Draft Model（例如使用 **mistral_small 的 Q8_0** 并配合 **i1-IQ1_S** 作为 Draft Model）可以显著提升 **Token 生成速率**。
   - 成员报告称，在使用 2x 3090 (48GB VRAM) 时速度从 18 t/s 提升到 30 t/s，其中一人报告了 83% 的 Token 接受率；不过，也有部分用户表示速度没有提升甚至有所下降。
- **48 GB 内存对新 Macbook 够用吗？**: 成员们讨论了新 Macbook 的 RAM 选择，一些人建议选择 **128GB**，以便在低量化下运行 **DeepSeek V2.5** 或 **Mistral Large** 等大型模型。
   - 一位成员指出 **Qwen 32B** 的表现超出了其参数规模，至少达到了 Llama 3.3 70b 的水平。此外还提到，在 2025 年，模型大小可能不会在没有重大缺陷的情况下减小。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: Unsloth 新手指南</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/479#issuecomment-2701947624">Issue with qwq-32b model in lmstudio. · Issue #479 · lmstudio-ai/lmstudio-bug-tracker</a>: 哪个版本的 LM Studio？例如：LM Studio 0.3.11。哪个操作系统？Mac。Bug 是什么？在使用 qwq-32b 模型聊天时出现以下错误 "Error rendering prompt with jinja templa...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1347525945573249086)** (80 条消息🔥🔥): 

> `9070XT vs 7900XTX, ROCm and Vulkan performance, Native FP4 support, CodeGPT extension issues on WSL, Quantization impact on model quality` 


- **9070XT 性能优于 7900XTX**：一位用户对比了在运行相同的 **qwen2.5 coder 14b q8_0** 模型时 **9070XT** 与 **7900XTX** 的表现，发现 **9070XT** 的运行速度约为 **44tok/sec**，首字延迟（Time to First Token）不到一秒，而 **7900XTX** 的运行速度为 **31tok/sec**，首字延迟为 **4sec**。
   - 另一位用户指出，**9070** 在运行 **0.5B** 模型时可达到 **400 t/s**，而 **7900xtx** 在 **300 到 360** 之间，这表明尽管使用的是 **Vulkan** 而非 **ROCm**，其性能至少提升了 **10%**。
- **9070XT 上的 Vulkan vs ROCm**：**9070XT** 目前在 Windows 上还不支持 **ROCm**，因此 **LM Studio** 使用的是 **Vulkan**。但一些用户报告称，**Vulkan** 的推理速度有时比 **ROCm** 快几个百分点，尽管这一点存在争议。
   - 一位用户认为那些看到 **Vulkan** 性能更好的人可能遇到了 *驱动程序问题*，因为 *ROCm 的速度理应远快于 Vulkan*。
- **现已支持原生 FP4**：成员们讨论了 **9070** 与旧款显卡相比，在 **FP16** 和 **INT4/FP4** 性能方面有显著提升。
   - 具体而言，**9070** 的 **FP16** 性能为 **122 vs 389**，**INT4/FP4** 性能为 **122 vs 1557**，这标志着 **Nvidia** 和 **Radeon** 现在都提供了原生 **FP4** 支持。
- **排查 WSL 上的 CodeGPT 扩展问题**：一位用户在尝试通过 **WSL** 中 **VSCode** 的 **CodeGPT** 扩展访问本地模型时，遇到了 *fetch failed status 500* 错误。
   - 建议的解决方案包括确保服务器设置为使用局域网 IP 在本地网络上提供服务，并开启 **CORS**（跨源资源共享）。
- **量化对模型质量的影响**：用户讨论了量化对模型质量的影响，特别是较小的量化尺寸与潜在质量损失之间的权衡。
   - 一位成员指出 **Q5_K_M** 的准确度比 **Q4_K_M** 更接近 **Q6_K**，且 **Q6** 应该是安全的；而另一位成员建议使用网页监控工具来关注 **9070 XT** 的到货情况。

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1347538735272689694)** (118 条消息🔥🔥): 

> `Replit/Bolt 的开源替代方案、Gradio Dexie 封装提案、Obsidian 用户来自 Obsidian、怀疑研究论文中存在数据集滥用、Hugging Face 数据集与 DOI 生成` 


- **关于 Replit 和 Bolt 的 Agent 替代方案的讨论**：一位成员询问了具有与 **Replit/Bolt** 类似 Agent 功能的开源模型，引发了关于潜在替代方案的讨论。
   - 此外，该线程还考虑了 *export_to_video* 的替代方案，以便将视频直接保存到 **Supabase bucket**。
- **通过 Dexie 访问 Gradio IndexedDB：是个好功能吗？**：一位用户建议集成 **Dexie wrapper**，以便在 Gradio 应用中轻松访问 IndexedDB。
   - 该建议引发了关于其作为自定义组件可行性的讨论，并分享了 **Gradio developers Discord 频道**的链接以便进一步咨询。
- **数据集引用纠纷：预印本的困境**：一位用户怀疑研究论文 '[NotaGen: Symbolic Music Generation with LLM Training Paradigms](https://arxiv.org/abs/2502.18008)' 在未妥善引用的情况下使用了他们的 **Toast MIDI dataset**。
   - 社区成员建议联系通讯作者，并为数据集生成 **DOI** 以确保获得正确的归属。
- **数据集引用探究：正确的 BibTeX 格式**：一位用户获得了关于如何在 Hugging Face 上为其数据集设置正确 **BibTeX 引用**格式的指导，包括将引用包装在三反引号中，并确保 URL 使用 `\url{}` 标签正确格式化。
   - 会议强调，正确的格式将确保学术软件和 Google Scholar 等研究人员社交媒体资料能够正确识别该数据集。
- **微调与多语言模型**：一位用户询问，在英文数据集上针对特定任务微调像 **Mistral** 基础模型，是否会将该知识迁移到其他语言。
   - 另一位用户回答说，他们*认为应该是这样吧？*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>：为您私人想法提供的免费且灵活的应用。</li><li><a href="https://huggingface.co/papers/2502.18008">Paper page - NotaGen: Advancing Musicality in Symbolic Music Generation with Large
  Language Model Training Paradigms</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.18008">NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms</a>：我们介绍了 NotaGen，这是一个符号音乐生成模型，旨在探索制作高质量古典乐谱的潜力。受大语言模型 (LLMs) 成功的启发，NotaGe...</li><li><a href="https://huggingface.co/datasets/breadlicker45/bread-midi-dataset">breadlicker45/bread-midi-dataset · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/breadlicker45/youtube-comments-180k">breadlicker45/youtube-comments-180k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://x.com/ClementDelangue/status/1897767808076816575?t=f0HsVgnlRua2PLTuPvIELQ&s=19">clem 🤗 (@ClementDelangue) 的推文</a>：谁说开源对你的业务不利？</li><li><a href="https://ieeexplore.ieee.org/abstract/document/10421308/">Exploring Embeddings for Measuring Text Relatedness: Unveiling Sentiments and Relationships in Online Comments</a>：在 COVID-19 疫情导致互联网使用量增长 70% 后，全球范围内使用社交媒体的人数不断增加。像 Twitter、Meta Threads、YouTube 这样的应用...</li><li><a href="https://huggingface.co/datasets/breadlicker45/toast-midi-dataset">breadlicker45/toast-midi-dataset · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-12-at-entry-0-and-35-at-entry-1/46155/2">RuntimeError: stack expects each tensor to be equal size, but got [12] at entry 0 and [35] at entry 1</a>：我认为 tokenized texts 的长度不一致，正如这条警告消息所示。如果你将每个 batch 的输入长度调整为相同，我认为错误就会消失。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1347568249927372993)** (3 messages): 

> `HF Docker Repository, fxtwitter` 


- **HuggingFace Docker 元数据上线**：一名成员注意到元数据正从 [HF Docker repository](https://huggingface.co/spaces/mozilla-ai/osm-ai-helper) 中获取。
   - 这表明 Hugging Face 生态系统内 **Docker containers** 的集成或信息检索功能得到了增强。
- **fxtwitter 被弃用**：一名成员表示 *你不再需要使用 **fxtwitter** 了，嵌入（embeds）现在工作正常 😄*。
   - 这意味着 Twitter 内容的**嵌入链接**现在可以在平台上正常显示，不再需要 fxtwitter 这种变通方案。



**提及的链接**：<a href="https://huggingface.co/spaces/mozilla-ai/osm-ai-helper">OpenStreetMap AI Helper - a Hugging Face Space by mozilla-ai</a>：未找到描述

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1347683704306270250)** (1 messages): 

> `Downloads, Community Appreciation` 


- **下载量达成里程碑！**：一名成员对其作品在 **10 天内达到近 1000 次下载** 表示惊讶和感谢。
   - 他们感谢了社区的支持，并分享了一张图片来庆祝这一成就。
- **社区为下载量里程碑喝彩**：社区庆祝一名成员的作品在短短 **10 天内达到近 1000 次下载**，展示了强大的社区参与度。
   - 随附的图片直观地展示了这一成就，进一步增强了社区内的兴奋感和感激之情。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1347677128551960678)** (1 messages): 

> `OCR-2.0 Guidance` 


- **寻求 OCR-2.0 微调指导**：一名成员请求关于如何完成某项任务的指导，并链接了一个 [SharePoint 文档](https://bama365-my.sharepoint.com/:w:/g/personal/xgranja_ua_edu/EeSz8D6iYPxHhzfQD3GGzsYBARpsSkbEDZWzoQH7hIH4lg?e=gMOaR4)。
   - 该成员认为 **OCR-2.0** 是目前最新且最好的模型，并询问是否应该根据文档对 **got/ocr-2.0** 模型进行微调。
- **OCR-2.0 被誉为前沿技术**：一位用户认定 **OCR-2.0** 是目前最先进的模型。
   - 他们正在考虑微调 **got/ocr-2.0** 模型以更好地满足其需求。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1347537494346039357)** (5 messages): 

> `Smol Agents Course, Pokemon LLM Agent Benchmark, HuggingFace Token issues` 


- ****Smol Agents** 课程模块更新**：**Smol Agents** 课程团队正在积极编写新的 MCP 模块，并更新了现有模块以更好地利用 **smolagents**；鼓励提交 [Pull Requests](https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark)。
- **拼写错误导致 **HuggingFace** Token 混乱！**：一名新学生在 Mac 上的 VSCode 中使用 **HuggingFace token** 时遇到问题，最初被判定为无效。
   - 问题被确定为一个拼写错误：*将字母 'O' 误认为数字 '0'* —— 用户还询问是否应该能够将 token 粘贴到 VSCode 中。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/thinking-book-penguin-student-writing-gif-5543983515725736276">Thinking Book GIF - Thinking Book Penguin - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark">GitHub - CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark</a>：通过在 GitHub 上创建账号来为 CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1347525730979811378)** (37 条消息🔥): 

> `课程开始日期，LLM 作为 Agent 组件，RAG 作为环境，课程完成状态，图像生成故障` 


- **LLM 的角色：是 Agent 还是仅仅是一个部分？**：一位成员澄清说，**LLM** 本身并不是 **Agent**，而是一个具备接收指令并调用工具以提供更多功能能力的组件。
   - 他们补充说，一个 **agentic AI model** 需要一个 **LLM** 和一个 **agent**，详见[这个有用的回复](https://discord.com/channels/879548961488179230/1201995434137206834/1208022892900941824)。
- **RAG 系统：是环境还是工具？**：一位成员询问，与 **LLM** 交互的 **Retrieval Augmented Generation (RAG)** 系统是否可以被视为一个“环境”，而 **LLM** 通过作为“工具”的 **RAG** 系统与之交互。
   - 另一位成员建议，从技术上讲，决定最佳响应的 decoder 层评估可以被视为 **LLM** 环境的一部分，即使使用了像 'stories' 这样的工具。
- **在 OpenSea 铸造合作伙伴 NFT？**：一条消息宣布了与 **OpenSea** 合作进行新的免费铸造。
   - 用户受邀加入并可能 **claim** 一个 **NFT**，并指出某些领取可能需要 gas 费；链接为 [opensea-nftbox5-one.vercel.app](https://opensea-nftbox5-one.vercel.app/)。
- **FLUX.1 的图像生成困扰**：一位课程参与者分享了一段尝试使用 **Stable Diffusion FLUX.1** 生成图像的代码片段，并表示难以找出问题所在。
   - 他们寻求关于一个函数的帮助，该函数使用了来自 `black-forest-labs/FLUX.1-dev` 模型的 `FluxPipeline`，并设置了诸如 `height`、`width`、`guidance_scale` 和 `num_inference_steps` 等各种参数。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1347532757714079785)** (144 条消息🔥🔥): 

> `Perplexity API 版权问题，OpenRouter 与 Anthropic API 的延迟，OpenRouter 中的 Groq 提供商，Gemini 嵌入模型，测试 OpenRouter 模型中的推理参数` 


- **Perplexity API 版权赔偿**：用户正在讨论使用 **Perplexity API** 可能带来的版权问题，因为它会抓取受版权保护的内容。用户注意到 [Perplexity 的 API 条款](https://www.perplexity.ai/hub/legal/perplexity-api-terms-of-service)要求客户赔偿 Perplexity，从而将责任转移到了用户身上。
   - 提到了具有知识产权（IP）赔偿保障的替代方案，如 **OpenAI**、**Google Cloud** 和 **AWS Bedrock**，并附带了各自条款的链接，建议用户评估法律风险。
- **Sonar Deep Research 模型出现错误**：成员报告 **Perplexity Sonar Deep Research** 模型出现频繁错误、高延迟（首个 token 延迟高达 **241 秒**）以及异常高的推理 token 计数。
   - 一位成员幽默地提到 **137k 推理 token** 计数却没有任何输出，而另一位成员确认在最初的问题之后，它最终开始正常工作。
- **新的实验性 Gemini 文本嵌入模型发布**：一个新的实验性 **Gemini 嵌入文本模型** (**gemini-embedding-exp-03-07**) 已在 [Gemini API](https://ai.google.dev/gemini-api/docs/models/experimental-models) 中上线，超越了之前的 state-of-the-art 模型。
   - 该模型在 **Massive Text Embedding Benchmark (MTEB) 多语言排行榜**上获得第一名，并包含更长的输入 token 长度等新特性。
- **OpenRouter 的推理参数存在不一致性**：用户发现 OpenRouter 的推理参数存在不一致，一些模型虽然被标记为支持推理，但端点却缺乏支持，且部分提供商不返回推理输出。
   - 成员们正在进行测试，发现了配置问题，并识别出模型与端点之间的差异，其中注意到 Cloudflare 缺少 **/completions 端点**。
- **模型在处理俄语提示词时遇到困难**：一位用户报告 **Claude 3.7** 在处理**俄语提示词**时表现不佳，以英语回复，并可能误解了语言的细微差别。
   - 这是在使用 cline 配合 OpenRouter 时观察到的，表明问题可能出在 Anthropic 而非插件或 OpenRouter 本身。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/use-cases/for-providers">提供商集成 - 将您的模型添加到 OpenRouter</a>：了解如何将您的 AI 模型与 OpenRouter 集成。为提供商提供的通过 OpenRouter 统一 API 提供其模型的完整指南。</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/embeddings">未找到标题</a>：未找到描述</li><li><a href="https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/">通过 Gemini API 实现最先进的文本嵌入</a>：未找到描述</li><li><a href="https://cloud.google.com/terms/service-terms">未找到标题</a>：未找到描述</li><li><a href="https://cloud.google.com/terms/generative-ai-indemnified-services,">未找到标题</a>：未找到描述</li><li><a href="https://learn.microsoft.com/en-us/legal/cognitive-services/openai/customer-copyright-commitment">客户版权承诺所需的缓解措施</a>：Azure OpenAI Service 的客户版权承诺所需的缓解措施</li><li><a href="https://aws.amazon.com/bedrock/faqs/">使用基础模型构建生成式 AI 应用 - Amazon Bedrock 常见问题解答 - AWS</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1347614421287047332)** (8 条消息🔥): 

> `Minion.ai, Gemini Embedding Model, Claude code vs cursor.sh vs VSCode+Cline` 


- **Minion.ai 走向终结**：一位成员表示 [Minion.ai](https://minion.ai) 已经“凉了”，不要相信那些炒作。
   - 另一位成员补充道，**Minion.ai** 就是那个有 4 个卡通形象角色、号称能帮你处理 Agent 事务的东西。
- **Google 扩展 Gemini Embedding Model 能力**：**Google** 正在为开发者推出一款实验性的 **Gemini Embedding Model**，在 [MTEB (Multilingual) 上具有 SOTA 性能](https://x.com/officiallogank/status/1898081742767919384?s=46)。
   - 更新后的模型具有 **8K tokens 的输入上下文长度**、**3K 维度的输出**，并**支持超过 100 种语言**。
- **Claude Code 进入 IDE 战场**：一位成员征求关于 **Claude code** 与 **cursor.sh** 以及 **VSCode+Cline / roo-cline** 的对比意见。
   - 他们明确表示比起最低成本更看重最高质量，并补充了[之前消息](https://discord.com/channels/822583790773862470/1075282825051385876/1346679448174596219)中的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/officiallogank/status/1898081742767919384?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天我们为开发者推出了一款实验性的 Gemini Embedding 模型，具有：– MTEB (Multilingual) 上的 SOTA 性能 - 输入上下文长度从 (3K --> 8K) tokens – 输出 3K 维度 – 支持...</li><li><a href="https://x.com/openaidevs/status/1898047744364659195?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：我们在文档中创建了一个新的模型页面——你现在可以轻松查看每个模型能力的详细分类，并并排比较模型。https://platform.openai.com/docs/models
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1347675208173092977)** (132 条消息🔥🔥): 

> `Web3 Agents, ElizaOS framework, AI Personas, Agent-as-a-Service, CryptoKitties` 


- **Web3 Agents 引发高频交易 (HFT) 狂热**：成员们开玩笑说 **Web3 agents** 正在进行高频交易 (HFT) 并“创建他们自己的狂热组织”。
   - 讨论中提到了 [GitHub 上的](https://github.com/elizaOS/eliza) 用于自主 Agent 的 **ElizaOS framework**。
- **AI Personas 成为新的 Meme 霸主**：引用一条 [推文](https://x.com/defiapes/status/1855657706205352035)，一位成员解释说，常态将迅速转向体现某种人格类型的 **AI PERSONAS**。
   - 推文提到，**Agent 将竞相成为子群体 x、y 和 z 的主要代表**，就像柏拉图的理念世界（World of Ideal Forms）一样，以完美封装喷子、瘾君子、运动员、说唱歌手、自由主义者和 memecoin 赌徒（degens）。
- **CryptoKitties 还活着？**：成员们被提醒，Flow 区块链的开发者 **Dapper Labs** 是热门 NFT 项目 [NBA Top Shot 和 Cryptokitties](https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/) 背后的公司。
   - 另一位成员早期并不看好 Bitcoin，因为他们觉得“哦，那是给科技男在火人节买药用的”，但没意识到那是数字价值存储的市场验证。
- **Agent-as-a-Service：下一个大趋势？**：一位成员询问“是否存在生产类似 Agent-as-a-Service 产品的市场”。
   - 一位成员正在考虑一个置于 DoorDash 前端的机器人，被称为 *DoorDash MCP*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/truth_terminal">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/arXivald">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/ThoughtTerminal">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/Purity_Terminal">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/andyayrey">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://www.bitstamp.net/learn/company-profiles/what-is-dapper-labs/">什么是 Dapper Labs?</a>: Dapper Labs 是 Web3 游戏领域的领导者，创建了 NBA Top Shot、Cryptokitties，并开发了 Flow 区块链以提供创新的 NFT 体验。</li><li><a href="https://x.com/hotteadaddy/status/1898118600583790865">来自 Zachary M (@hotteadaddy) 的推文</a>: @arXivald 告诉我，哪篇 LLM 社交 Agent 白皮书相当于比特币白皮书</li><li><a href="https://x.com/hashwarlock/status/1895369752199168469">来自 Agent Joshua ₱ (@hashwarlock) 的推文</a>: 好的，我在这里已经完成了很多工作。@PhalaNetwork Cloud 工具将成为市场上最好的。一旦我完成一些清理工作，我将开始着手分解我们的信任链模型，以可验证地证明...</li><li><a href="https://x.com/defiapes/status/1855657706205352035?s=46">来自 Atum (@DefiApes) 的推文</a>: 人们在 AI Agent 热潮中忽略了一个关键叙事。你需要在它变得显而易见之前意识到这一点。目前几乎所有走红的 Agent 都是“通才”，他们几乎发布任何内容。他们很受欢迎...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: 每周即兴会议</a>: 未找到描述</li><li><a href="https://github.com/elizaOS/eliza">GitHub - elizaOS/eliza: 为每个人提供的自主 Agent</a>: 为每个人提供的自主 Agent。通过在 GitHub 上创建账户，为 elizaOS/eliza 的开发做出贡献。</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">怀疑按下 X GIF - 怀疑按下 X 黑色洛城 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/HRNPH/AIwaifu">GitHub - HRNPH/AIwaifu: Open-Waifu 开源、可微调、可定制、可互动的 AI waifu，灵感来自 neuro-sama</a>: Open-Waifu 开源、可微调、可定制、可互动的 AI waifu，灵感来自 neuro-sama - GitHub - HRNPH/AIwaifu...</li><li><a href="https://github.com/elizaOS/eliza?tab=readme-ov-file#-quick-start">GitHub - elizaOS/eliza: 为每个人提供的自主 Agent</a>: 为每个人提供的自主 Agent。通过在 GitHub 上创建账户，为 elizaOS/eliza 的开发做出贡献。</li><li><a href="https://tenor.com/view/charlie-always-sunny-gif-26054360">Charlie Always GIF - 费城永远阳光灿烂 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/pippinlovesyou/pippin">GitHub - pippinlovesyou/pippin: 用于自主 Agent 的数字生命框架</a>: 用于自主 Agent 的数字生命框架。通过在 GitHub 上创建账户，为 pippinlovesyou/pippin 的开发做出贡献。</li><li><a href="https://tenor.com/bLwFC.gif">斜眼狗怀疑眼神 GIF - 斜眼狗怀疑眼神 - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1347533949630546032)** (72 条消息🔥🔥): 

> `ChatGPT token 限制，与 AI 共享 GPS，本地 LLM，熟练技工的 AI Copilot，临时聊天框` 


- **ChatGPT 实施聊天限制**：成员们反映 **ChatGPT** 现在开始执行 **每周 50 条消息** 的限制，以防止滥用。
- **AI Copilot 进入熟练技工领域**：一位成员正与朋友开发一个基于 **HVAC（暖通空调）安装手册** 训练的 LLM，并分享了其工作原理的 [YouTube 演示视频](https://youtu.be/oAiUAzKLe_Y)。
   - 另一位成员建议测试 [Mistral OCR](https://mistral.ai/) 模型，并指出 *它在阅读手册方面表现出色，即使是倾斜角度、难以辨认的字体、表格等也能处理，而且推理成本极低*。
- **本地 LLM vs 云端 LLM**：成员们讨论了在本地机器上运行 **LLM** 与使用云端服务的对比，有人认为本地 DIY LLM 比黑盒云端解决方案更有潜力。
   - 有人指出 *大多数应用会在 GPU 和 CPU 之间进行分配*，并且 *量化（quantisation）通过将精度降低到 8 bit 或更低来减少内存需求*。
- **社区质疑 OpenAI 的临时聊天框**：社区成员注意到当前的 OpenAI 聊天用户界面中 *没有聊天框*。
   - 一位成员提到 *我的很正常，如果你没有，刷新窗口有用吗？刷新后就修复了*。



**提到的链接**：<a href="https://youtu.be/oAiUAzKLe_Y.">AI Copilot 技术手册</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1347557130332274738)** (32 条消息🔥): 

> `Manus AI Agent，OpenAI Plus O1 限制，SimTheory O1 消息上限，ChatGPT 记忆与文件夹` 


- **Manus AI Agent 令人惊叹**：一位成员询问是否有人听说过 [Manus AI](https://www.manus.ai/)，称其为一个 *令人惊叹的 Agent*。
   - 另一位成员请求提供链接以便了解。
- **OpenAI Plus 用户注意到更高的 O1 限制**：一位用户报告称，其使用的 **O1** 数量超过了预期的 **25 条限制**，且没有收到通知或被阻止。
   - 其他人提到了 **50 条 O1 限制**，并表示在通知提醒方面的体验各不相同。
- **SimTheory 声称以更低价格提供更高的 O1 消息上限**：一位用户推荐了 [SimTheory](https://simtheory.ai/)，声称他们以比 **OpenAI** 更低的价格提供更高的 **O1 消息上限**。
   - 其他成员表示怀疑，质疑他们如何能以比 **OpenAI** 官方更低的价格提供更高的限制。
- **清除聊天记录后 ChatGPT App 数据丢失**：一位用户分享了在 **ChatGPT App** 中选择清除聊天记录后丢失 **ChatGPT** 文件夹的经历。
   - 该用户对清除聊天记录为何会影响文件夹感到困惑。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (3 条消息): 

> `模型遵循请求模式，可控性的影响，项目前评估` 


- **模型模仿请求，忽略替代方案**：模型倾向于紧密遵循请求模式，忽略潜在更好或更简单的方法，假设用户的方法是有意为之，或是在纠正感知到的错误。
   - **模型可控性（steerability）** 的提高放大了这种行为，使模型更强烈地相信用户陈述的意图。
- **可控性增强了对意图的假设**：模型的可控性越高，它就越会假设用户的要求正是其真实意图，即使该方案并非最优，这可能导致模型执行有缺陷的计划。
   - 当给出的代码是 *用锤子钉螺丝* 时，模型会照做，猜测我们知道自己想要什么，并且我们使用的路径也是它应该使用的路径。
- **先评估，后编码**：在 **开始之前** 要求模型评估并讨论项目的目标、想法和方法，有助于识别最佳方案和潜在问题。
   - 询问 *我想实现 X。我开始通过 [这样做] 来实现。请讨论我的目标、想法和方法，你觉得怎么样？* 是 **预先解决疑虑并增强项目设计** 的有效方式。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1347611582305402911)** (3 messages): 

> `Model's Presumptions, Steerability Impact, Pre-Project Evaluation, Method Optimization` 


- **模型会产生预设 (Presumptions)**：模型*经常*会顺着请求的模式运行，而不顾是否有更好的替代方案，它预设用户知道自己在问什么并且**是认真的**。
   - 它也可能猜测用户搞错了，并认为用户指的是他们实际上没说出的内容，从而导致潜在的次优结果。
- **可操控性 (Steerability) 会强化信念**：模型的可操控性越强，它就越相信你言出必行，即使该方法效率低下或不合常规。
   - 这可能导致模型采用次优的代码或方法，因为它猜测我们知道自己想要什么，并且我们使用的路径也是我们希望它使用的路径。
- **项目前讨论：黄金门票**：在让模型**开始**项目之前，先要求它评估或讨论请求是一个**极佳**的主意。
   - 一位成员建议使用以下表述：*我想实现 X。我开始通过 [这样做] 来实现。讨论一下我的目标、想法和方法，你觉得怎么样？*。
- **模型教学并探索更好的方法**：要求模型预先讨论目标和方法，可以让它进行引导、探索并帮助识别最佳方法和关注点。
   - 这种预先讨论可以通过从一开始就将项目引导至更高效、更有效的方法来节省时间和精力。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1347528831614844980)** (65 messages🔥🔥): 

> `Aider showing reasoning, Jamba model release, AI-written code, Copilot account suspension, Claude token consumption` 


- **Aider 需要推理指示器**：用户请求在 Aider 中增加一个功能，以指示推理模型何时正在积极推理，特别是针对 **Openrouter** 上的 **Deepseek R1** 或 **V3**。
   - 一位成员建议对 Aider 和 **litellm** 进行补丁，并提到 **litellm** 中有一个*黑科技 (hack)* 可以获取 **<think>** 标签内的推理 Token。
- **AI21 Labs 发布 Jamba 模型**：**AI21 Labs** 发布了 **Jamba 1.6 Large & Mini** 模型，声称其质量和速度优于开源模型竞争对手，并在具有 **256K context window** 的长上下文任务中表现出色。
   - 该模型采用了一种新型的 **Mamba-Transformer MoE** 混合架构，旨在提高成本和效率收益，可自托管部署或在 **AI21 SaaS** 中使用。
- **初创公司严重依赖 AI 编写的代码**：一位 **Y Combinator** 顾问表示，*当前这一批初创公司中，有四分之一的业务几乎完全基于 AI 编写的代码*。
- **因轻度使用 Aider 导致 Copilot 账号被封禁**：有用户报告称，因在 Aider 中*非常轻度地使用 copilot-api* 而导致 **Copilot 账号被封禁**，并提醒其他使用者注意。
   - 其他人推测账号共享、二手账号和频率限制 (rate limiting) 可能是原因，并链接到了 [copilot-api GitHub repo](https://github.com/ericc-ch/copilot-api/blob/master/README.md)。
- **Claude 3.7 故意增加 Token 消耗？**：用户认为 **Claude 3.7** *故意在回复中增加范围以提高 Token 消耗账单*。
   - 一些用户注意到它经常执行*不必要或懒惰的操作*，并指出了[关于排查编辑错误的文档](https://aider.chat/docs/troubleshooting/edit-errors.html)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.ai21.com/jamba/">Jamba 1.6: The Best Open Model for Enterprise Deployment</a>: 探索 AI21 的 Jamba —— 一款尖端的、长上下文的 AI 开源模型，专为准确性、效率和强大的文本生成而构建。</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: 未找到描述</li><li><a href="https://www.youtube.com/@pbsspacetime">PBS Space Time</a>: Space Time 与我们的天体物理学家主持人 Matthew 一起探索宇宙的深处、天体物理学的疯狂、科幻的可能性，以及你能想到的地球之外的任何事物……</li><li><a href="https://github.com/ericc-ch/copilot-api/blob/master/README.md">copilot-api/README.md at master · ericc-ch/copilot-api</a>: GitHub Copilot API 封装器，使其兼容 OpenAI - ericc-ch/copilot-api
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1347529089573064776)** (38 messages🔥): 

> `Aider API Key、MCP Agents 集成、Playwright 证书错误、QwQ-32B 本地模型基准测试、Aider 脚本与 Web 内容` 


- **解决 Aider /run 命令的 API Key 问题**：一位用户在使用 `/run` 命令调用 LLM 时，因缺少 API Key 导致 API 调用失败。
   - 建议确保在运行 `aider` 时 API Key 已作为环境变量提供，因为 `/run` 命令应该会继承这些变量。
- **探讨 MCP Agents 与 Aider 的集成**：用户询问了将 **MCP agents** 集成到 `aider` 的计划，并征求有效使用它们的建议。
   - 有人指出 **MCP** 被认为是不安全的，目前更适合实验而非生产环境使用，并引用了 [GitHub 上的 mcpm-aider](https://github.com/lutzleonhardt/mcpm-aider)。
- **绕过 Aider /web 命令中的 Playwright 证书错误**：用户在使用 `/web` 命令访问 HTTPS 网站时遇到了 `net::ERR_CERT_AUTHORITY_INVALID` 错误。
   - 建议通过在 `playwright.config` 文件中添加特定参数来配置 **Playwright** 忽略证书错误，参考了 [Stack Overflow 的回答](https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command)。
- **QwQ-32B 挑战 R1 的地位？**：一位成员分享了关于 **QwQ-32B** 的[帖子链接](https://x.com/victormustar/status/1898001657226506362)，声称它彻底改变了本地 AI 编程，并在家用环境下实现了 state-of-the-art (SOTA) 性能。
   - 另一位成员指向了 [Discord](https://discord.com/channels/1131200896827654144/1346923740667187502) 中的基准测试讨论，认为 **QwQ-32B** 可能表现不错，但考虑到模型大小，未必优于 **R1**。
- **Aider 的 Web 内容脚本处理能力**：用户报告成功使用了 `/web` 命令，但随后调用 `aider` 时无法识别已添加到 `.aider.chat.history.md` 文件中的网页内容。
   - 该用户随后表示已自行解决该问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/questions/68219072/playwright-not-accepting-https-urls-while-openinign-with-codegen-command">Playwright not accepting https urls while openinign with codegen command</a>：npx playwright codegen https:/// page.goto: net::ERR_CERT_AUTHORITY_INVALID at ...&#xA;如何通过传递输入参数或身份验证凭据，在使用 codegen 命令时打开 https URL。</li><li><a href="https://x.com/victormustar/status/1898001657226506362">Victor M (@victormustar) 的推文</a>：QwQ-32B 永远改变了本地 AI 编程 🤯 我们现在在家就能拥有 SOTA 性能。分享我的技术栈 + 技巧 ⬇️</li><li><a href="https://github.com/lutzleonhardt/mcpm-aider">GitHub - lutzleonhardt/mcpm-aider: 用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。还可以运行 MCP Server 来帮助你管理所有的 MCP Server</a>：一个用于在 Claude App 中管理 MCP 服务器并供 aider 使用的命令行工具。
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1347569560395448370)** (61 条消息🔥🔥): 

> `LinkedIn premium referral codes, Entropy as a Penalty, DeepSeek Ban, Discrete Diffusion Modeling` 


- **LinkedIn Premium 推荐码发放**：一位成员提供了即将到期的 **LinkedIn premium 推荐码**，并请感兴趣的人私信。
   - 他们对最近的多次提醒表示抱歉，并询问了这种推广对于 **AI community** 的潜在好处。
- **熵约束是软约束**：一位成员指出，只有当你根据 KKT 条件设置 λ₃ 时，**约束项 (constraint term)** 才是真正的约束，而 **Entropy** 本质上也是一种 **Penalty**。
   - 他们认为，为了获得有用的预测器，对模型进行 **weight prior** 是必不可少的，因为你无法将 Lebesgue-measure 为 0 的数据集泛化到非 Lebesgue-measure 为 0 的函数上。
- **公司禁用 DeepSeek**：一位成员注意到，尽管 **DeepSeek** 是开源的，但全球许多公司出于安全考虑正在禁用它。
   - 多位成员就风险展开了辩论，一些人认为这些担忧源于媒体影响和潜在的中国政府影响，而另一些人则强调了 **reviewable code** 和本地运行对安全的重要性。
- **离散扩散论文提议**：一位成员建议在接下来的讨论中探讨论文 *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*。
   - 他们指出，这篇论文似乎是 [Inception Labs 产品](https://www.inceptionlabs.ai/about)的基础，并分享了[论文链接](https://arxiv.org/pdf/2310.16834)。
- **求解 Lambert W**：一位成员在回答有关优化复杂性的问题时表示，求解 **duals λ** 需要对 **Lambert W function** 进行优化，这在计算上可能效率低下。
   - 他们建议使用 **cvxpy** 来解决该问题，并推荐使用 **adjoint method** 进行微分。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1347635023678804098)** (10 条消息🔥): 

> `Latent Reasoning, Chain-of-Thought Data, Context Compression, VQ-VAE` 


- **探索通过潜空间抽象进行推理**：一篇新[论文](https://arxiv.org/abs/2502.03275)提出了一种推理过程的混合表示，使用 **VQ-VAE** 生成的 **latent discrete tokens** 来抽象初始推理步骤，从而缩短推理轨迹长度。
   - 该论文探讨了从零开始训练模型，以及在扩展词汇表的混合数据上微调 LLM，通过混合潜空间 Token 和文本 Token 来实现更快的适配。
- **推理术语辩论**：一位成员质疑论文中描述的潜空间推理究竟是真正的 **Latent Reasoning**，还是仅仅是 **Context Compression**。
   - 另一位成员开玩笑说，需要电休克疗法 (ECT) 来阻止 AI 领域的人滥用“推理 (reasoning)”这个词。



**提到的链接**：<a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>：Large Language Models (LLMs) 在使用 Chain-of-Thought (CoT) 数据训练时，擅长推理和规划，其中分步思考过程由文本 Token 明确列出。然而，这种方式...

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1347550701278072842)** (12 条消息🔥): 

> `Diffusion Models 幻觉、多步 Agentic 工作流、LLADA 局限性、OpenAI 的 AGI 转向、中国 AI Agent Manus` 


- **Diffusion Models 无法避开幻觉**：一位成员指出，Diffusion models 本身并不能解决 LLM 的幻觉问题，因为*幻觉只是“猜错了”的另一种说法*。
   - 虽然自我编辑能力可以用高置信度样本替换低置信度样本，但*并没有能保证正确性的魔法*。
- **采样缓解了多步 Agent 循环**：[这篇论文](https://arxiv.org/abs/2411.07641)和 [GitHub 仓库](https://github.com/Tomorrowdawn/top_nsigma)中详细介绍的 **n-sigma-sampling** 的使用，似乎通过直接在 pre-softmax logits 上使用统计阈值操作，缓解了多步 Agentic 工作流中的坏样本和循环行为。
   - 该技术无需复杂的概率操作即可高效过滤 token，无论 temperature scaling 如何，都能保持稳定的采样空间。
- **通过 LLADA 实现的 Language Diffusion 表现不佳**：虽然在概念上很有吸引力，但 **language diffusion** 和 **LLADA** 等模型被认为受到其训练的限制，正如这篇 [NeurIPS 论文](https://neurips.cc/virtual/2024/poster/95935)和 [GitHub 仓库](https://github.com/HKUNLP/diffusion-of-thoughts)所述，它们更侧重于基准测试而非广泛的现实世界任务。
   - 尽管 Diffusion-of-Thought (**DoT**) 等技术具有提高推理能力的潜力，但一位成员观察到 LLADA 会出现重复段落的情况。
- **OpenAI 软化了对 AGI 突破的立场**：据[这篇文章](https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/)报道，**OpenAI** 正在放弃对突然实现 **AGI** 突破的预期，现在将其发展视为一个持续的过程。
   - 这一转变可能是由于 **GPT 4.5** 的发布反响平平，一些人指责 **Sam Altman** 过度炒作了预期。
- **中国 Manus AI Agent 走红**：一个名为 **Manus** 的中国 AI Agent 在中国走红，被描述为类似于 **Deep Research + Operator + Claude Computer** 的结合体，详见这些 [推文](https://x.com/rowancheung/status/1898093008601395380) 和 [推文](https://x.com/heyBarsee/status/1898027732899962887)，可通过 [Manus](https://manus.im) 访问。
   - 报告显示，**Manus** 实现了约 50 个任务的自动化，并且比 DeepSeek 更准确，能同时处理金融交易、研究和采购。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://the-decoder.com/openai-shifts-away-from-sudden-agi-breakthrough-theory/">OpenAI 转向远离突然的 AGI 突破理论</a>: OpenAI 是 ChatGPT 和众多其他商业 AI 应用背后的公司，长期以来一直追求开发“造福全人类”的通用人工智能 (AGI) 的目标...</li><li><a href="https://arxiv.org/abs/2411.07641">Top-$nσ$: Not All Logits Are You Need</a>: 大语言模型 (LLMs) 通常在推理任务中使用贪婪解码或低温度采样，这反映了多样性和准确性之间的一种权衡。我们对这一传统观点提出了挑战...</li><li><a href="https://github.com/Tomorrowdawn/top_nsigma">GitHub - Tomorrowdawn/top_nsigma: LLM top_nsigma 采样策略的官方代码库和数据中心。</a>: LLM top_nsigma 采样策略的官方代码库和数据中心。 - GitHub - Tomorrowdawn/top_nsigma: LLM top_nsigma 采样策略的官方代码库和数据中心。</li><li><a href="https://neurips.cc/virtual/2024/poster/95935">NeurIPS Poster Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models</a>: 未找到描述</li><li><a href="https://manus.im">Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长工作和生活中的各种任务，在你休息时帮你完成一切。</li><li><a href="https://x.com/rowancheung/status/1898093008601395380">Rowan Cheung (@rowancheung) 的推文</a>: 我认为中国的第二个 DeepSeek 时刻已经到来。这个名为 'Manus' 的 AI Agent 目前在中国疯狂走红。传到美国可能只是时间问题。它就像 Deep R...</li><li><a href="https://x.com/heyBarsee/status/1898027732899962887">Barsee 🐶 (@heyBarsee) 的推文</a>: AI 正在失控 🤯 来自中国的 AI Agent Manus 正在实现约 50 个任务的自动化，创造了一个相当反乌托邦的场景。报告显示它比 DeepSeek 更准确，能够同时...
</li>
</ul>

</div>

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1347542709648490547)** (73 条消息🔥🔥): 

> `MCP 安全担忧, MCP 在商业产品中的采用, 恶意提示词注入, MCP 与 GitHub Copilot, 开源 vs 闭源 MCP` 


- **VSCode 计划集成 MCP 支持**：**VSCode** 正计划为 **GitHub Copilot** 添加 MCP 支持，正如其 [直播](https://youtu.be/Pe8ghwTMFlg) 中提到的。
- **用户对 MCP server 中的恶意提示词注入表示担忧**：针对 **MCP server** 向 AI Agent 提供恶意提示词注入的问题引发了关注，强调了模型被训练为信任工具调用（tool calls）而非内部知识，这产生了一个潜在漏洞。
   - 成员建议，在首次使用 MCP Server 时，应向用户展示工具描述和指令列表以供审查，并在指令发生变化时发出警报。
- **MCP 安全是安全社区的下一个目标**：社区成员考虑创建一个安全频道，以主动帮助防止漏洞利用，并对人们在不了解工具调用对 **MCP Server/Remote Host** 后果的情况下随意连接一切表示担忧。
   - 一位成员建议利用监管合规工具描述来诱导模型植入后门。
- **MCP 快速入门指南让用户感到困惑**：用户在运行 **Python MCP 快速入门** 时遇到错误，建议使用 [wong2/mcp-cli](https://github.com/wong2/mcp-cli) 作为更好的替代方案。
   - 官方快速入门指南可以在 [这里](https://modelcontextprotocol.io/quickstart/client) 找到。
- **OpenAPI 转 MCP 代理**：一位成员正在开发 **mcp-openapi-proxy**，旨在将任何 swagger/openapi 端点转换为可发现的工具，采用与其 **mcp-flowise server** 相同的设计。
   - 他最新的 mcp-server [在 5ire 中运行正常](https://5ire.org/)，但在 Claude desktop 中无法运行。



**提及的链接**：<a href="https://modelcontextprotocol.io/quickstart/client">For Client Developers - Model Context Protocol</a>：未找到描述

  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1347666391691694180)** (9 条消息🔥): 

> `Mastra Agent, Searxng MCP Server, Python fetch server 的 Typescript 移植版` 


- ****Mastra** 通过新 Agent 进行清理！**：一位成员使用 **Mastra** 创建了一个简单的 Agent 来整理文档和下载内容，并在 [YouTube 视频](https://youtu.be/HplcOOSJCps) 中进行了展示。
   - 该 Agent 使用 filesystem MCP 来清理 Documents 文件夹，并且也已用于 downloads 文件夹。
- **Typescript **Fetch** server 移植到 MCP！**：一位成员询问了 Python *fetch* server 的 Typescript 移植版，另一位成员确认两者非常相似。
   - 最大的区别在于它具有 *更好的 网站 -> Markdown 解析功能*。
- ****Searxng MCP** 搜索网页！**：一位成员为网页搜索构建了一个简单的 **searxng MCP** server，可在 [GitHub](https://github.com/aeon-seraph/searxng-mcp) 上获取。
   - 该实现会缓存最近的搜索，并专门为语言模型格式化来自多个引擎的响应，可以配置为使用 localhost 或外部提供商。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/HplcOOSJCps">Organizing Files with Mastra, MCP and 4o-mini</a>：这里我正在使用这个用 Mastra 构建的小型 Agent，它利用 filesystem MCP 来清理我的 Documents 文件夹。我也在我的 downloads 文件夹上使用了它。</li><li><a href="https://github.com/aeon-seraph/mcp-servers/tree/main/src/thinking">mcp-servers/src/thinking at main · aeon-seraph/mcp-servers</a>：通过在 GitHub 上创建账号为 aeon-seraph/mcp-servers 作出贡献。</li><li><a href="https://github.com/aeon-seraph/searxng-mcp">GitHub - aeon-seraph/searxng-mcp</a>：通过在 GitHub 上创建账号为 aeon-seraph/searxng-mcp 作出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1347654490123276380)** (68 条消息🔥🔥): 

> `Mojo 的动态性、Python 互操作性、Monkey Patching 替代方案、Protocol 多态性` 


- **Mojo 动态性讨论**：讨论围绕 **Mojo 的动态性**及其对性能的影响展开，并考虑了使用场景以及灵活性与速度之间的潜在权衡，参考了 [这篇 HN 帖子](https://news.ycombinator.com/item?id=35811170)。
   - 一些人建议动态性应该只在被使用时才产生性能开销，而另一些人则担心即使没有使用 classes，它也会影响 structs 的性能，例如为了调试而添加动态类属性或函数赋值。
- **Monkey Patching 替代方案**：讨论探索了 **Monkey Patching** 的替代方案，如函数指针或组合（composition），以在 Mojo 中实现动态行为。同时讨论了避免 Monkey Patching 的原因，例如*它速度更慢、更难理解、会破坏静态分析工具，且通常无法完成正规多态性（polymorphism）所不能完成的事情*。
   - 辩论涵盖了 Monkey Patching 的效用和缺点，强调过度使用会导致*代码变慢且更难理解*，从而干扰静态分析。
- **Python 库移植与 CPython 互操作性**：成员们讨论了**将 Python 库移植到 Mojo** 的挑战，重点在于动态性和全局状态。建议对于某些库，在性能关键部分使用 **CPython 互操作性**可能是一种务实的方法。
   - 针对移植重度依赖 Python 动态性和全局状态的库所面临的困难提出了担忧，特别是如果该库离开这些特性就无法编写的情况。
- **Protocol 多态性讨论**：对话触及了 **Protocol 多态性**及其在不依赖类树的情况下实现多态性的效用，参考了 [PEP 544](https://peps.python.org/pep-0544/)。
   - 一些成员主张使用函数指针的哈希表来实现动态行为，在 Mojo 的单元测试中更倾向于使用静态类型和所有权规则（ownership rules）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://peps.python.org/pep-0544/">PEP 544 – Protocols: Structural subtyping (static duck typing) | peps.python.org</a>：PEP 484 中引入的类型提示可用于为静态类型检查器和其他第三方工具指定类型元数据。然而，PEP 484 仅指定了标称子类型（nominal subtyping）的语义。在本文中...</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/seal">Object.seal() - JavaScript | MDN</a>：Object.seal() 静态方法用于封闭一个对象。封闭一个对象可以防止扩展，并使现有属性不可配置。封闭对象具有固定的属性集：新属性不能...</li><li><a href="https://news.ycombinator.com/item?id=35811170">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1347706930399023204)** (2 条消息): 

> `SOTA Agent 方法、Arxiv 论文、算法复杂度、状态机、框架抽象` 


- **SOTA Agent 方法其实很简单**：发言者认为 Arxiv 上的 **SOTA Agent 方法** *在算法上往往相当简单*，类似于一个相对较小的状态机。
   - 这意味着这些框架的抽象其实并不那么被需要，因为代码库会很精简；对数据管理、状态管理和 API 调用进行独立的抽象就足够了。
- **Agent 方法中的算法复杂度**：讨论指出，当前的 **SOTA Agent 方法** 通常涉及相对简单的算法。
   - 这种简单性表明复杂的框架抽象可能是不必要的，因为较小的代码库可以通过更简单的抽象有效地管理数据、状态和 API 调用。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1347592874165928007)** (5 messages): 

> `Triton Autotune use_cuda_graph 参数, Triton Kernel SVD Quant 性能, Nunchaku SVD Quant 实现` 


- **寻求 `use_cuda_graph` 参数的解释**：一名成员正在寻求关于 `triton.autotune` 中 `use_cuda_graph` 参数的详细说明。
   - 他们不确定该参数如何应用，因为 `triton.autotune` 装饰的是单个 CUDA kernel，而 CUDA graphs 通常用于节省序列化的 kernel 启动时间。
- **Triton Kernel SVD Quant 比 fp16 慢**：一名成员发现他们实现的 [Triton kernel](https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb)（实现了量化和 4bit 打包，以及两个带有 FMA 的 matmuls）比 **PyTorch 的 fp16 matmul** 还要慢，尽管遵循了第 14 课的设计原则。
   - 他们报告称其速度比 **fp16** 线性层**慢 5 倍**，并指出 auto-tuning 似乎使性能变得更糟。
- **受 Nunchaku SVD Quant 启发？**：一名成员询问该实现是否基于 **MIT-HAN-Lab** 的 [**Nunchaku SVDQuant** 仓库](https://github.com/mit-han-lab/nunchaku)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb">tritonKernel_svdQuant/svdConversion.ipynb at main · rishabh063/tritonKernel_svdQuant</a>：通过在 GitHub 上创建账号来为 rishabh063/tritonKernel_svdQuant 的开发做出贡献。</li><li><a href="https://github.com/mit-han-lab/nunchaku">GitHub - mit-han-lab/nunchaku: [ICLR2025 Spotlight] SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>：[ICLR2025 Spotlight] SVDQuant: 通过低秩组件吸收 4-Bit 扩散模型中的离群值 - mit-han-lab/nunchaku
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1347742182655918090)** (1 messages): 

> `PTX, CUDA C++` 


- **关于 PTX 学习资源的咨询**：一名成员表示有兴趣学习用于内联 **CUDA C++** 编程的 **PTX**，并请求相关的学习材料、教程或可以开始的小项目。
- **请求 PTX 资源**：请求学习 **PTX** 和内联 **CUDA C++** 编程的资源，包括教程和小项目。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1347599950527070350)** (6 messages): 

> `Distributed barrier, cuda synchronize, register_comm_hook, FSDP communication hook` 


- **Distributed Barrier 调试**：一名成员报告称，**distributed barrier** 在 torch 函数之前触发，这可能有助于缩小问题范围。
   - 另一名成员澄清说，`cuda synchronize` 是 **GPU 端**的 barrier，而 **distributed barrier** 是 **host 端**的。
- **FSDP Communication Hooks PR 出现**：一名成员询问是否有办法自定义 **FSDP(2)** 处理其通信的方式，类似于 torch **DDP** 中的 `register_comm_hook`。
   - 另一名成员分享了一个[相关的 pull request](https://github.com/pytorch/pytorch/pull/83254) 链接，该 PR 为 sharded strategies 实现了一个 **FSDP communication hook** 接口。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/pull/83254">Added communication hook for sharded cases by aovladi · Pull Request #83254 · pytorch/pytorch</a>：修复了 #79114，为 sharded strategies 实现了一个 FSDP communication hook 接口：在默认 hook 中添加了 reduce_scatter_hook。注意 reduce_scatter 与 all_reduce 的区别...

  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1347571245868126300)** (2 messages): 

> `NCCL AllReduce, Double Binary Trees, Ring Topology, Communication Latency` 


- **使用 Double Binary Trees 实现的 NCCL AllReduce 优于 Ring Topology**：[NVIDIA 博客文章](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)指出，与 AllReduce 操作的 **2D Ring 延迟**相比，**NCCL 2.4** 中的 **Double Binary Trees** 提供了全带宽和更低的对数延迟。
   - NCCL 2.4 中引入了 Double Binary Trees，它*提供全带宽，且对数延迟甚至低于 2D Ring 延迟。*
- **Ring Topology 延迟随节点数量线性增长**：在 Ring Topology 中，每个处理器仅与其相邻节点通信，导致 All-Reduce 操作的通信复杂度为 **O(p)**，其中 **p** 是处理器数量。
   - 有人提到：*增加处理器会线性扩展所需的通信和操作数量*。
- **基于树的拓扑结构实现对数级通信复杂度**：在基于树的拓扑结构中，通信和操作并行发生，复杂度为 **O(log L)**，其中 **L** 是树的层数，与 Ring Topology 相比能有效降低延迟。
   - 根据一位成员的假设，*我们的总复杂度可以表示为 O(log p)*。
- **Double Binary Trees 利用 Rank 分布**：Double Binary Trees 利用了二叉树中一半或更少的 Rank 是 Node，而其余是 Leaf 的特性，从而能够构建第二棵树，将 Leaf 用作 Node，反之亦然。
   - 一位成员表示，这样做是为了降低总复杂度，*可能有一个 Rank 在两棵树上都是 Leaf，但没有 Rank 在两棵树上都是 Node*。



**提到的链接**：<a href="https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/">使用 NCCL 2.4 大规模扩展深度学习训练 | NVIDIA 技术博客</a>：想象一下使用数万个 GPU 来训练你的神经网络。在所有深度学习框架中使用多个 GPU 训练神经网络已变得非常普遍，提供了优化的&#8230;

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1347683049139339314)** (7 messages): 

> `WoolyAI, CUDA abstraction layer, GPU resource utilization, PyTorch support` 


- **WoolyAI 发布 CUDA 抽象层 Beta 版**：WoolyAI 技术为其 [CUDA 抽象层](https://docs.woolyai.com)发布了 Beta 版，该层将 Kernel Shader 的执行与 CUDA 应用程序解耦。
   - 该抽象层将应用程序编译为新的二进制文件，Shader 被编译成 Wooly 指令集（Wooly Instruction Set），动态调度工作负载以实现最佳 GPU 资源利用率。
- **WoolyAI 阐明动态 GPU 调度**：WoolyAI 动态调度工作负载，使来自多个用户的不同 Kernel 能够在单个 GPU 上运行，而无需硬分区（hard partitioning）。
   - 用户根据使用的核心数和 VRAM 计费，目前支持仅限于 **PyTorch**。
- **WoolyAI 被比作基于使用的 MIG**：WoolyAI 将其动态调度方法描述为类似于一种基于使用的 MIG（Multi-Instance GPU）形式。



**提到的链接**：<a href="https://docs.woolyai.com">简介 | WoolyAI 文档</a>：什么是 Wooly？

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1347546333652975658)** (5 messages): 

> `GPU Memory Buffers on Apple, cuda_graph in Triton Autotune, Resources for GPU/TPU Programming` 


- **Apple 的 GPU 内存共享：指针的天堂**：在 Apple GPU 上，内存是在线程间共享的，允许直接使用指向内存位置的指针，这与其他系统形成对比。
   - 这种共享内存架构通过允许跨线程的直接内存访问，简化了某些并行编程任务。
- **Triton 的 Autotune：绘制 CUDA Kernel 启动图**：一位成员询问了 `triton.autotune` 中的 `use_cuda_graph` 参数，质疑其在单个 CUDA Kernel 上的应用。
   - 这个问题突显了对 CUDA Graph 如何优化序列中 Kernel 启动时间的潜在误解，特别是在 Triton 的 Autotuning 装饰器上下文中。
- **新手需要 GPU/TPU 编程入门包**：一位成员询问《Programming Massively Parallel Processors》是否足以开始 GPU/TPU 编程，目标是构建像 TinyGrad 或 Torch 这样的框架。
   - 他们精通汇编、C、C++ 和 Python，对深度学习模型和数学有扎实的理解，正在寻求从哪里开始他们的 GPU Kernel 编程之旅。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1347563923674501170)** (3 messages): 

> `AMD GPU Rental, Compile HIP code, Runpod MI300 Access` 


- **寻求 AMD GPU 租赁服务**：一名成员正在寻找租赁 **AMD GPU** 的服务，用于编译带有内联 **ASM** 的 **HIP code**，以进行 **GEMM** 基准测试。
   - 声明的目的是出于好奇和实验，特别是旨在利用 **matMul accelerator**。
- **无需 AMD GPU 即可编译 HIP 代码**：一名成员提到，只要拥有 **hipcc**（可以通过标准方法获取），就可以在不需要 GPU 的情况下编译 **HIP code**。
   - 他们澄清说，虽然代码可以编译，但在没有 **GPU** 的情况下无法运行。
- **Runpod 提供 MI300 访问**：一名成员建议 [Runpod](https://runpod.io/) 是获取 **MI300** GPU 访问权限的好方法。
   - 未提供关于 **Runpod** 或其服务的更多细节。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1347751513564647427)** (1 messages): 

> `Kernel Compilation, Matrix Shapes, TileLang` 


- **TileLang Kernel 编译疑问**：一位用户询问在使用 **TileLang** 时，是否需要为每种矩阵形状（matrix shape）都编译一次 kernel。
   - 这个问题暗示了对处理 **TileLang** 中各种矩阵形状时编译开销（compilation overhead）的担忧，这可能会影响开发效率。
- **矩阵形状编译**：用户希望避免在每次出现新的矩阵形状时都重新编译 kernel。
   - 是否需要重新编译取决于 **TileLang** 的设计，以及它是否支持动态形状（dynamic shapes）或需要特定形状的 kernel。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1347752733171519528)** (7 messages): 

> `Cute Kernels for Training, Triton vs CUDA, Custom Autotune Implementation, LLVM Compiler Efficiency` 


- ****Cute Kernels** 加速训练**：一名成员宣布发布 [Cute Kernels](https://github.com/mayank31398/cute-kernels)，这是一个通过在 **Triton** 和 **CUDA** 实现上进行自动调优（autotuning）来加速训练的 kernel 集合。
   - 这些 kernel 是端到端 torch compileable 的，且没有图断裂（graph breaks），该仓库包含一个用于生产环境训练 **IBM Granite LLMs** 的自定义 autotune 实现。
- ****Triton** 和 **CUDA****：核心思想是使用模式匹配器（pattern matcher）自动将 matmuls 分发到 **cutlass** 或 **Triton**，具体参考[这个匹配器](https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35)。
   - 提到由于 **LLVM** 编译器有时可以创建比 **NVCC** 更高效的代码，因此对 kernel 后端（backend）进行调优是有意义的。
- ****LLVM 编译器****：该仓库提供了一个向量加法的简单示例，展示了使用 **LLVM** 生成专用高性能 kernel 的简便性，可以在[这里](https://github.com/mayank31398/cute-kernels/blob/main/cute_kernels/kernels/add/add_tensor/__init__.py)找到。
   - 一名成员还提到所有 kernel 都是支持 **JIT** 编译的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mayank31398/cute-kernels/blob/main/examples/cute_inductor.py#L35">cute-kernels/examples/cute_inductor.py at main · mayank31398/cute-kernels</a>：一堆可能会让东西变慢的 kernel 😉。通过在 GitHub 上创建账号为 mayank31398/cute-kernels 的开发做出贡献。</li><li><a href="https://github.com/mayank31398/cute-kernels">GitHub - mayank31398/cute-kernels：一堆可能会让东西变慢的 kernel 😉</a>：一堆可能会让东西变慢的 kernel 😉。通过在 GitHub 上创建账号为 mayank31398/cute-kernels 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1347671571682168873)** (1 messages): 

> `LCF concurrency, DDP+nccl, Deadlocks` 


- **LCF 并发问题浮现**：一名成员询问 **LCF** 在与 streams 一起使用时是否设计为完全并发安全，并提到了死锁（deadlocks）问题。
   - 他们在将 **LCF** 与 **DDP+nccl** 结合使用时遇到了问题，并好奇其他人是否也遇到了类似情况。
- **DDP+nccl 环境下的 LCF 死锁**：一位用户报告在尝试将 **LCF** 与 **DDP+nccl** 结合使用时遇到了奇怪的死锁。
   - 他们正在寻求社区的反馈，了解是否有其他人经历过类似的问题。


  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1347553986248441867)** (15 messages🔥): 

> `Curriculum Creation, Reasoning Gym, Sonnet Context Experiment, Reasoning GAN Self-Play, LLMs Speed Up Developers` 


- ****Curriculum Creation** 开始**：成员们计划开始编写 Curriculum，并询问 API 的稳定性和实现情况。
   - 一位成员指向了“上方的 Curriculum 线程”以获取更多细节，表明该领域的工作正在进行中。
- **Reasoning Gym 接近完成**：**Reasoning Gym** 数据集已完成 **99%**。
   - 一位成员表示惊讶，以为已经 **100%** 完成了，而另一位成员澄清说*还有一个 PR 仍处于开启状态*。
- **Sonnet 的 Context Reasoning Gym 扩展**：一位成员建议将整个 **Reasoning Gym** 放入 **Sonnet** 的上下文（context）中，并提示它生成另外 **100** 个数据集，从而无限重复该过程。
   - 另一位成员幽默地回应道 *“然后无限重复那个过程？”*，暗示了无限数据生成的潜力。
- **Reasoning GAN Self-Play 出现**：一位成员提议训练一个模型来解决生成的数据集，并将其类比为 [Reasoning GAN self-play](https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU)。
   - 共享文档需要登录，但它似乎概述了一种自动推理问题生成和解决的方法。
- **关于 LLM 提升开发者速度的实验**：讨论中提到了一个关于 **LLM** 能在多大程度上提升开发者速度的实验，认为这值得关注。
   - 有人指出，你*可以在不增加额外工作量的情况下获得丰厚的报酬（尽管有一半的工作你将无法使用 AI）*。



**提及的链接**：<a href="https://docs.google.com/document/d/1ytdo9LoBWuK2IKXUCla0YwC_0g-nqzv5VwmbevSAQPU">实验：LLM 能在多大程度上提升开发者速度</a>：METR 正在寻找经常参与大型开源项目的软件工程师，以测试 AI 软件工程工具的有效性。在此申请 (bit.ly/ai-speedup-apply)。有疑问？联系...

  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1347638233512808509)** (2 messages): 

> `AVX-256 performance on 3a, Hybrid AVX-256/AVX-512 approach, Tiling and OpenMP` 


- **AVX-256 在 3a 上可能达到 3s**：成员们讨论了使用 **tiling**、**OpenMP** 和 **AVX-256** 指令在 **3a** 上实现 **<= 3s** 运行时间的可能性。
   - 一位成员确认实现目标运行时间*应该是可能的*，但可能会很困难。
- **带有 AVX-512 寄存器的混合 AVX-256 方案**：成员们提议了一种混合方法，仅使用 **AVX2** 指令，但受益于 **AVX512** 带来的寄存器数量增加。
   - 这允许利用 **AVX512 的寄存器数量**，而无需完全投入到 **AVX512** 指令中。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1347580708025008209)** (6 messages): 

> `Open Source AI Projects, GPT-NeoX, Tooling Setup for Claude Code` 


- **新成员寻找开源 AI 项目！**：一位新成员正在寻找在 **LLM pre-training**、**post-training**、**RL** 和 **interpretability** 等领域有趣的开源 AI 项目，该成员具有预训练 **GPT-2** 和微调模型的经验。
- **推荐 modded-nanoGPT！**：一位成员为那些对理论工作感兴趣的人推荐了 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt/)，将该项目描述为 *3 分钟内的 NanoGPT (124M)*。
- **用于预训练的 GPT-NeoX 项目！**：一位成员建议，如果新成员对预训练感兴趣，最好的参与项目是 **GPT-NeoX** 训练库。
- **寻求 Claude Code 的工具链配置**：一位成员正在寻求关于配置 **Claude Code** 或类似编码环境工具链的建议，并对潜在成本表示担忧。



**提及的链接**：<a href="https://github.com/KellerJordan/modded-nanogpt/">GitHub - KellerJordan/modded-nanogpt: 3 分钟内的 NanoGPT (124M)</a>：3 分钟内的 NanoGPT (124M)。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 的开发做出贡献。

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1347597705245098085)** (16 messages🔥): 

> `Token Assorted Paper, TorchTitan Embedding Sharding, Embedding Layer Implementation` 


- **Token Assorted 论文的词表技巧**：一位成员推测 **Token Assorted 论文** 可能只是在对 latent codes 进行 next token prediction 的 fine-tuning 过程中，简单地将 **codebook** 添加到了词表中。
   - 他们批评这种方法可能无法推广到开放推理领域，并建议在推理语料库中寻找 K 个最常见的字符串可能会产生更好的结果。
- **为什么需要 all-reduce**：讨论解释说，在使用原生 **TP** 时，如果输入 embedding 在 vocab 维度上进行 sharding，则之后需要进行 **all-reduce**，以处理 embedding layer 对缺失词表元素输出为 0 的情况。
   - 这种方法通过避免指定哪个设备拥有哪个 token 来简化逻辑，尽管它假设 embedding layer 在没有被查询的 vocab 元素时输出 **0**，从实现角度来看这可能比较奇怪。
- **Embedding 层的存储技巧**：成员们讨论了让 embedding layer 对缺失 token 输出零的存储影响，指出在通信后立即需要存储空间。
   - 他们还观察到，如果可以重用该存储，它就是免费的；而省略存储则需要一个索引结构来跟踪哪些项存在或不存在。



**链接提到**：<a href="https://github.com/pytorch/torchtitan/issues/785#issuecomment-2585007139">Why use RowwiseParallel for nn.Embedding instead of ColwiseParallel? · Issue #785 · pytorch/torchtitan</a>：Colwise 使逻辑更加清晰。Rowwise 在 token 维度上进行拆分，导致在不同 shard 如何处理其 shard 内不存在的 token 时产生困惑。从某种程度上说...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1347556483247640608)** (3 messages): 

> `Logit Lens, Multilingual language models, Llama-2 family` 


- **Logit Lens 产生有趣的结果**：一位成员分享了他们对[这篇关于多语言语言模型的论文](https://arxiv.org/abs/2402.10588)以及使用 **Logit Lens** 的赞赏。
   - 该论文调查了在不平衡且以英语为主的语料库上训练的 **multilingual language models** 是否使用英语作为内部中转语言，特别关注 **Llama-2 系列**。
- **理解 Llama-2 中的语言偏见**：链接的论文探讨了以英语为主的数据训练的 **Llama-2** 模型如何处理非英语 prompt。
   - 它通过高维空间追踪中间层 embedding，揭示了模型最初倾向于英语翻译，然后才调整为输入语言的阶段。



**链接提到**：<a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>：我们询问在不平衡、以英语为主的语料库上训练的多语言语言模型是否使用英语作为内部中转语言——这对于理解语言模型如何运作至关重要...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1347649184534233253)** (2 messages): 

> `Knowledge Graph Visualization, Anthropic Cookbook Updates` 


- **yWorks 实时可视化知识图谱**：来自 @yworks 的演示展示了 **yFiles**（他们的知识图谱可视化 SDK），提供[实时更新和动态交互](https://t.co/mb6M2R3TTh)。
- **Anthropic Cookbook 获得更新**：LlamaIndex 团队更新并扩展了他们的 **Anthropic Cookbook**，为学习[基础 API 设置](https://t.co/SQQ63qmwRb)以及简单的 completion 和 chat 方法提供了权威来源。

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1347532336157298708)** (22 messages🔥): 

> `SQLTableRetrieverQueryEngine, Jina AI package issues, LlamaExtract beta request, Tool Calling with Reasoning Models` 


- **使用 SQLTableRetrieverQueryEngine 打印 Prompt**：有成员询问如何打印 LlamaIndex 中 **SQLTableRetrieverQueryEngine** 所使用的 Prompt。
   - 另一位成员分享了代码片段 `from llama_index.core import set_global_handler; set_global_handler("simple")` 以启用 Prompt 打印。
- **Jina AI 软件包困境**：有成员报告了 **Jina AI** 软件包的导入错误。
   - 建议使用 `npm install @llamaindex/jinaai` 安装提供者软件包，并附上了 [LlamaIndex 迁移文档](https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9) 的链接，解释了 v0.9 中向提供者软件包的转变。
- **LlamaExtract Beta 测试申请**：有成员请求访问 **LlamaExtract** 的 Beta 版本。
   - 他们被指示私信特定用户或其他成员并提供邮箱，同时也被引导至 [LlamaExtract 文档](https://docs.cloud.llamaindex.ai/llamaextract/getting_started) 以获取更多信息。
- **关于 Reasoning Models 的 Tool Calling 讨论**：有成员询问如何将 LlamaIndex 工作流与 Reasoning Model 结合进行 **Tool Calling**，并提到他们目前的设置是 vLLM、Qwen 32B 和 ReAct Prompting。
   - 另一位成员指向了一个 [LlamaIndex 示例](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/)，该示例演示了此功能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.cloud.llamaindex.ai/llamaextract/getting_started">入门指南 | LlamaCloud 文档</a>：概述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Function Calling Agent 工作流 - LlamaIndex</a>：未找到描述</li><li><a href="https://ts.llamaindex.ai/docs/llamaindex/migration/0.8-to-0.9">从 v0.8 迁移到 v0.9</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1347590222422216775)** (15 messages🔥): 

> `Command R7B inference time, Tool invocation with r7b model, Open source AI contributions` 


- **Command R7B 推理缓慢归咎于硬件**：有成员询问 **Command R7B** 在 Hugging Face 上推理时间过长的问题。
   - 另一位成员回答说，缓慢很可能是由于用户的**硬件**或模型的运行方式造成的，而不是模型本身的问题。
- **Ollama 工具调用失败**：一位新用户报告了在 **Ollama** 和 **Langchain** 中使用 `command-r7b:latest` 模型进行 **Tool Invocation** 时的问题，收到的错误信息如 *"抱歉，我没有权限访问回答您问题所需的工具。"*
   - 成员建议确保工具以正确的 **JSON format** 传递，并验证 **Ollama** 是否支持带有必要配置的 Tool Calling。
- **寻求 AI 开源贡献建议**：一位在 **GPT-2** 预训练和 **Hellaswag** 模型微调方面有经验的成员正在寻求有趣的开源 AI 项目进行贡献。
   - 该用户还表达了建立人脉的兴趣，特别是与不列颠哥伦比亚省温哥华地区的人士。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1347753806691307562)** (2 messages): 

> `504 Gateway Error, Server Error` 


- **504 Gateway Error 再次出现**：用户报告再次出现相同的 **504 Gateway Error** 并请求检查。
   - 随后的消息包含了错误的更多细节，包括标题 *Error: Server Error* 和 *服务器遇到临时错误，无法完成您的请求。*
- **报告临时服务器错误**：用户报告收到 **502 Server Error**，表明存在临时问题。
   - 错误消息建议在 **30 秒**后重试请求。


  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1347592708663017533)** (1 messages): 

> `Knowledge Graphs, TogetherAI LLM, Topic Modelling` 


- **推荐将 Knowledge Graphs 用于 Topic Modelling**：成员建议研究 **Knowledge Graphs** 以增强 **Topic Modelling** 能力。
   - 他们特别推荐使用来自 **TogetherAI** 的 **LLM**，并强调了其用于实验的丰厚免费额度。
- **利用 TogetherAI LLM 进行 Topic Modelling**：建议利用 **TogetherAI** 的 **LLM** 进行有效的 **Topic Modelling**。
   - **TogetherAI** 提供的丰厚免费额度被认为是探索其平台的一个极具吸引力的理由。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1347540629185368127)** (11 条消息🔥): 

> `Wondercraft AI 播客，NotebookLM 与 Wondershare 集成，云端硬盘加密，播客音频语言` 


- ****播客克隆变得更加高效****：一位用户分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=0UWYFFOPjqs)，展示了 **NotebookLM** 与 **Wondercraft** 的集成，用于创建具有专业克隆声音的播客，提供了比 **11Labs** 和 **HeyGen** 更精简的方法。
   - **Wondercraft** 的订阅价格可能较高，除非用户正在通过其播客获利。
- ****云端硬盘数据安全性讨论****：一位用户指出，虽然数据在传输到 **Google Drive** 期间是加密的，但在 **Drive** 本身存储时并*未*加密。
   - 这意味着 **Google**、成功的黑客以及用户与其共享目录的人员都可以访问这些数据。
- ****AI 语音通过结巴变得更真实****：一位用户注意到，生成的音频文件中的 **AI 发言人** 现在会*像正常人一样结巴*，感觉非常自然，并附上了一个 wav 文件示例。
   - 但该用户也提到，结巴增加了音频长度，可能会减少在 **Google** 每日限制内传达的实际信息量。
- ****解锁分析师/指南聊天设置影响时间线生成****：一位用户发现 **NotebookLM** 中的**聊天设置**（如分析师/指南、短/长）会影响时间线和概览的生成，因为主持人在生成音频概览期间本质上会请求这些设置。
   - 他们还注意到，他们的助手会将简报概览、详细时间线、角色表和原始资料合并为一个文档。
- ****用户寻求播客音频语言修复方案****：一位用户询问是否可以在 **NotebookLM** 中更改播客音频的语言。
   - 另一位用户提供了自定义提示词作为变通方案，例如 *Only speak in (language here). podcast is entirely (language here). Not English.*，并指出目前没有官方更改音频语言的方法。



**提到的链接**：<a href="https://www.youtube.com/watch?v=0UWYFFOPjqs">Insane AI Podcast Results - Edit NotebookLM on Wondercraft</a>：🔥 限时优惠：Wondercraft 50% 折扣！使用此链接和优惠券代码 "MRC" https://mrc.fm/wondercraft。在这段视频中，我将带你完成一个简单的流程来创建...

  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1347589779696521316)** (3 条消息): 

> `NotebookLM，Chrome 扩展程序，网页导入工具，YouTube URL` 


- **NotebookLM 缺少直接上传 URL 列表的功能**：成员们讨论了向 **NotebookLM** 上传 URL 列表以将每个 URL 添加为来源的可能性。
   - 一位成员澄清说，虽然 **NotebookLM** 本身不支持此功能，但有几个 [Chrome 扩展程序](https://chromewebstore.google.com/search/notebooklm) 可用于此目的。
- **NotebookLM Chrome 扩展程序出现**：提到了几个 **Chrome 扩展程序** 作为将网页和 YouTube 视频导入 **NotebookLM** 的解决方案。
   - 这些包括 [NotebookLM Web Importer](https://chromewebstore.google.com/detail/notebooklm-web-importer/ncjabfmpppgonojpohbfhfaahfpkgihc)、[NotebookLM Toolbox](https://chromewebstore.google.com/detail/notebooklm-toolbox/nbpckfdmlndgcoaklokdbllhelmfhoal) 和 [NotebookLM YouTube Turbo](https://chromewebstore.google.com/detail/notebooklm-youtube-turbo/mjpdncbmeognfgjkcdnglfomkmnknpgk)。



**提到的链接**：<a href="https://chromewebstore.google.com/search/notebooklm)">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1347553807126495232)** (13 messages🔥): 

> `DSPy batch function, vllm backend with 2 instances, LM subclass in vllm, pipeline parallel size in vllm` 


- **关于 DSPy Batch 函数的疑问**：一名成员询问 **DSPy 的 batch 函数** 是否能高效地将并行任务分配给运行两个相同 LLM 实例的 **vllm 后端**，并引用了参数 *num_threads*、*max_errors* 和 *return_failed_examples*。
   - 另一名成员澄清说，这取决于 **vllm 设置**，并建议如果 VRAM 充足，应增加 *num_sequences* 或 *pipeline_parallel_size*，而不是使用两个独立的 API。
- **VLLM 流水线并行（Pipeline Parallel）见解**：一名成员确认其 **vllm 设置** 在单个节点上使用了设置为 2 的 *pipeline parallel size*。
   - 另一名成员确认，当 *pp* 设置为 2 时，**vllm 会处理负载均衡**，并鼓励通过基准测试（benchmarking）来对比并行与非并行方法的处理时间。
- **用于负载均衡的 LM 子类**：一名成员建议，如果两个实例位于不同节点，**DSPy** 不处理负载均衡，但可以使用 **proxy**（代理）转发请求。
   - 他们还提议创建一个 **LM** 的子类来实现负载均衡，尽管在 **vllm** 端解决此问题是首选方案。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1347630352071393372)** (6 messages): 

> `string replacement, laptop break` 


- **分享字符串替换代码片段**：一名成员在闲聊频道分享了一个用于字符串替换的代码片段：`$cleanedString = str_replace(['!', '@', '#', '$', '%', '^'], '', "This is a test string!@#$%^");`。
- **笔记本电脑摔坏导致缺席**：一名成员提到他们最近缺席是因为笔记本电脑*摔了一跤*后坏了。
   - 他们提到*情况不太妙*。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

china_xi: 对于一个 2 层模型，tinygrad jit 耗时超过 30 分钟正常吗？
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

china_xi: 除了第一步（step 0），之后所有的 loss 都变成 nan 的原因可能是什么？
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1347618683089846274)** (2 messages): 

> `Audio modality in torchtune` 


- **Torchtune 关注音频模态的加入**：成员们讨论了未来在 **Torchtune** 中加入 **音频模态（audio modality）** 的计划。
   - 目前尚无更多可用细节。
- **Torchtune 音频模态 - 未来计划**：关于未来可能为 Torchtune 添加音频模态支持的讨论。
   - 未提供具体的时间表或技术细节，表明该计划尚处于早期规划阶段。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

claire_csy: 能重新发一下链接吗？之前的过期了，谢谢！
  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1347706433608613938)** (1 messages): 

> `Diffusion LLMs, LLaDA Model, Transformer vs Diffusion` 


- **Diffusion LLMs 引发热议，社区询问“是颠覆性的变革还是虚火？”**：一名社区成员询问关于 **Diffusion LLMs** 发布的热潮，特别是 **Mercury** 模型，以及它是否会取代 **基于 Transformer 的模型**。
   - 他们提到阅读了白皮书但发现难以理解，正寻求社区专家的见解。
- **LLaDA 模型：Diffusion 驱动的 LLM 范式转移**：一名社区成员分享了 [diffusionllm.net](https://diffusionllm.net/) 的链接，解释说 **Large Language Diffusion Models** (LLaDA) 代表了语言模型架构的一种新范式。
   - 他们阐明，与传统的 **自回归 (AR) Transformers** 不同，**LLaDA** 使用去噪扩散过程（denoising diffusion process）以并行、由粗到细的方式生成文本。



**提到的链接**：<a href="https://diffusionllm.net/">Diffusion LLMs - Revolutionary Language Model Architecture | LLaDA Research Hub</a>：探索 Diffusion LLMs 如何通过并行处理和高级错误修正彻底改变 AI。了解 LLaDA 架构并关注前沿研究动态。

  

---


{% else %}


> 频道详细明细已在邮件中截断。
> 
> 如果你想查看完整明细，请访问此邮件的 Web 版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}