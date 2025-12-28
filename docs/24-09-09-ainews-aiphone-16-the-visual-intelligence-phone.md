---
companies:
- apple
- openai
- weights-biases
date: '2024-09-09T23:00:14.695088Z'
description: '**苹果**发布了全新的 **iPhone 16** 系列，其特色是“**视觉智能**”（Visual Intelligence）。这是一种集成了相机控制、苹果地图和
  Siri 的新 AI 功能，强调隐私保护，并优先使用默认服务而非 OpenAI 等第三方 AI。**苹果照片**（Apple Photos）现在包含先进的视频理解功能，支持时间戳识别。与此同时，**Reflection-70B**
  声称是顶级的开源模型，但基准测试显示其性能接近 **Llama 3 70B**，且略逊于 **Qwen 2 72B**。**Yann LeCun** 强调了大型语言模型（LLM）在规划能力方面面临的持续挑战，并指出
  **Llama-3.1-405b** 和 **Claude** 等模型表现出了一定的能力，而 **GPT-4** 和 **Gemini** 则相对落后。**Weights
  & Biases** 正在赞助一项旨在推进 LLM 评估技术的活动，并提供奖金和 API 访问权限。'
id: 7214b2cc-a903-40be-b22f-bcdae70ab4f5
models:
- reflection-70b
- llama-3-70b
- qwen-2-72b
- llama-3-1-405b
- claude
- gpt-4
- gemini
original_slug: ainews-aiphone-16-the-visual-intelligence-phone
people:
- yann-lecun
title: AIPhone 16：视觉智能手机
topics:
- vision
- video-understanding
- benchmarking
- planning
- model-evaluation
- privacy
- ai-integration
- instruction-following
---

<!-- buttondown-editor-mode: plaintext -->**Apple Intelligence 或许就是你所需的一切。**

> 2024年9月6日至9月9日的 AI 新闻。我们为你查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**215** 个频道，**7493** 条消息）。预计节省阅读时间（以 200wpm 计算）：**774 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

在今天的 [Apple 特别活动](https://www.youtube.com/watch?v=uarNiSl_uh4) 中，新款 iPhone 16 系列正式发布，同时花费了 [5 分钟](https://x.com/swyx/status/1832234771973583220) 介绍了 Apple Intelligence 的一些更新（我们假设你已经跟进了我们关于 [WWDC](https://buttondown.com/ainews/archive/ainews-talaria-apples-new-mlops-superweapon-4066/) 和 [Beta 版本](https://buttondown.com/ainews/archive/ainews-apple-intelligence/) 的报道）。

 
![image.png](https://assets.buttondown.email/images/ad63676a-c8e0-47a3-96e6-4bda3883ad12.png?w=960&fit=max)
 

最新的更新是他们现在称之为 **Visual Intelligence** 的功能，随 iPhone 16 新增的专用 Camera Control 按钮一同推出：

 
![image.png](https://assets.buttondown.email/images/3d0c61de-f2d7-4b79-925f-6e848d2eb964.png?w=960&fit=max)
 

正如在 [Winds of AI Winter 播客](https://x.com/latentspacepod/status/1819394111352590802) 中讨论并现已得到证实的那样，Apple 正在将 OpenAI 商品化，并将自己的服务放在首位：

 
![image.png](https://assets.buttondown.email/images/6fadfdf0-da63-4ae9-a2ee-d10670e369cf.png?w=960&fit=max)
 

据推测，用户最终将能够在新的 UI 中配置 Ask 和 Search 按钮调用的内容，但每个 Visual Intelligence 请求都会首先通过 Apple Maps 和 Siri 运行，然后才是那些第三方服务。Apple 在这里通过抢先运行、作为默认选项以及保证私密/免费而获胜，这出人意料地比追求“最强”更具防御性。

Apple Photos 现在也具备了非常出色的视频理解能力，甚至可以精确到视频中的时间戳：

 
![image.png](https://assets.buttondown.email/images/e575859c-9290-4a9b-91a3-4d7a1c97f948.png?w=960&fit=max)
 

Craig Federighi 在他的环节中称这是 Apple Intelligence 的一部分，但其中一些功能已经出现在 [iOS 18.0 beta](https://news.ycombinator.com/item?id=41493502) 中（Apple Intelligence 仅在 iOS 18.1 中发布）。

你可以阅读 [Hacker News 的评论](https://news.ycombinator.com/item?id=41493023) 了解其他亮点和愤世嫉俗的观点，但这就是今天最重大的必知事项。

还要过多少年，Apple Visual Intelligence 才会变成……始终开启？

 
![image.png](https://assets.buttondown.email/images/e340df47-f02e-4707-bbc4-1eb9cf5b90ea.png?w=960&fit=max)
 

---

**关于 Reflection 70B 的说明**：我们[上周的报道](https://buttondown.com/ainews/archive/ainews-reflection-70b-by-matt-from-it-department/)（以及 [Twitter 评论](https://x.com/swyx/status/1832234771973583220)）涵盖了周五已知的批评，但周末出现了更多质疑其主张的证据。我们预计本周会有更多进展，因此现在将其作为另一个标题故事还为时过早，但感兴趣的读者可以滚动到下方的 /r/localLlama 部分查看完整说明。

也许我们应该[致力于开发更多不可作弊的 LLM evals](https://x.com/drjimfan/status/1833160432833716715?s=46)？好消息是，本月的推理支持由我们的朋友 W&B 提供……

---

**由 Weights & Biases 赞助**：如果你是湾区的开发者，在 **9月21/22日**，[Weights & Biases](https://wandb.ai/site/?utm_source=sponsorship&utm_medium=newsletter&utm_campaign=swyx) 邀请你与他们一起黑客马拉松，推动 **LLM-evaluators 现状** 的发展。在 [W&B Judgement Day hack](http://wandb.me/swyx-hack) 中构建更好的 LLM Judges —— **5000 美元奖金**，提供 API 访问和食物。

[
![image.png](https://assets.buttondown.email/images/a9630f4b-58d6-40e8-b545-6eacb5b44ba4.png?w=960&fit=max)
](http://wandb.me/ainews-hack)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型进展与基准测试**

- **Reflection-70B 的宣称**：[@JJitsev](https://twitter.com/JJitsev/status/1832758737397797270) 报道称，Reflection-70B 声称根据通用基准测试，它是“全球顶尖的开源模型”。然而，使用 AIW 问题的初步测试显示，该模型**接近 Llama 3 70B，且略逊于 Qwen 2 72B**，并未达到其宣称的顶尖性能。

- **LLM 规划能力**：[@ylecun](https://twitter.com/ylecun/status/1832860107925024789) 指出 **LLM 在规划方面仍然面临困难**。Llama-3.1-405b 和 Claude 在 Blocksworld 上表现出一定的规划能力，而 GPT4 和 Gemini 表现不佳。所有模型在 Mystery Blocksworld 上的表现都被描述为“极其糟糕”。

- **PLANSEARCH 算法**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1832922753734689059) 重点介绍了一种名为 PLANSEARCH 的用于代码生成的搜索算法。它**生成多样化的观察结果，以自然语言构建计划，并将有前景的计划转化为代码**。Claude 3.5 使用该方法在 LiveCodeBench 上实现了 77.0% 的 pass@200，优于无搜索基准。

**AI 工具与应用**

- **RAG 流水线开发**：[@dzhng](https://twitter.com/dzhng/status/1832925319415886183) 报告称，使用 Cursor AI composer 在不到一小时内编写了一个 RAG 流水线，并使用 Hyde 和 Cohere reranker 进行了优化，**期间没有编写一行代码**。整个过程是通过语音听写完成的。

- **Google AI 的 Illuminate**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1832759050271887562) 提到 Google AI 发布了 Illuminate，这是一个将研究论文转换为短播客的工具。用户可能需要等待几天。

- **Claude 对比 Google**：[@svpino](https://twitter.com/svpino/status/1832875910274867222) 分享了一次经历：在尝试使用 Google 解决一个问题数小时未果后，Claude 在 5 分钟内就提供了该问题的分步指导。

**AI 研究与进展**

- **AlphaProteo**：[@adcock_brett](https://twitter.com/adcock_brett/status/1832812027003150583) 报道了 Google DeepMind 推出的 AlphaProteo，这是一个 AI 系统，旨在创建用于与特定分子靶点结合的定制蛋白质，有望加速药物发现和癌症研究。

- **AI 驱动的研究助手**：[@LangChainAI](https://twitter.com/LangChainAI/status/1832826233102454806) 分享了一个先进的 AI 驱动研究助手系统，该系统使用多个专门的 Agent 执行数据分析、可视化和报告生成等任务。它是开源的，并使用了 LangGraph。

- **顶级 ML 论文**：[@dair_ai](https://twitter.com/dair_ai/status/1832807193990627638) 列出了本周顶级的 ML 论文，包括 OLMoE、LongCite、AlphaProteo、Role of RAG Noise in LLMs、Strategic Chain-of-Thought 以及 RAG in the Era of Long-Context LLMs。

**AI 伦理与社会影响**

- **移民担忧**：[@fchollet](https://twitter.com/fchollet/status/1832832611229864405) 对潜在的移民执法行动表示担忧，认为在某些情况下法律文件可能无法提供保护。

- **AI 的更广泛影响**：[@bindureddy](https://twitter.com/bindureddy/status/1832847309350310157) 强调 AI 不仅仅是炒作或商业周期，他指出我们正在创造比人类更有能力的新生命，AI “远比金钱重要”。

**硬件与基础设施**

- **Framework 13 电脑**：[@svpino](https://twitter.com/svpino/status/1832856449900560749) 提到购买了一台 Framework 13 电脑（Batch 3）用于运行 Ubuntu，在使用了 14 年 Mac 后转向了新平台。

- **Llama 3 性能**：[@vipulved](https://twitter.com/vipulved/status/1832875063630303548) 报告称，随着新推理引擎的发布，Llama 3 405B 在 Together API 上突破了 100 TPS 大关，在 NVIDIA H100 GPU 上达到了 106.9 TPS。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Reflection 70B 争议：潜在的 API 欺诈与社区抵制**

- **[确认：REFLECTION 70B 的官方 API 是 SONNET 3.5](https://i.redd.it/csggt4kkonnd1.png)** ([得分: 278, 评论: 168](https://reddit.com//r/LocalLLaMA/comments/1fc98fu/confirmed_reflection_70bs_official_api_is_sonnet/)): **Reflection 70B** 的官方 API 已被确认为 **Sonnet 3.5**。这一信息与之前的推测一致，并澄清了支持这一大语言模型的技术架构。确认 Sonnet 3.5 作为 API 意味着为使用 Reflection 70B 的开发者提供了特定的功能和集成方法。

- **[[OpenRouter Reflection 70B 声称自己是 Claude，由 Anthropic 创建（你可以亲自尝试）](https://i.redd.it/mn1cfnnbrnnd1.png)]** ([Score: 68, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1fc9lf4/openrouter_reflection_70b_claims_to_be_claude/)): 通过 OpenRouter API 提供的 **Reflection 70B** 模型声称自己是 **Claude**，并表示它是由 **Anthropic** 开发的。这一断言引发了人们对该模型真实身份和来源的质疑，因为 **Anthropic** 不太可能在不发布公告的情况下通过第三方 API 发布 **Claude**。建议用户亲自测试该模型以验证这些说法并评估其能力。

- **[[Reflection 70B (Free) 现在已失效](https://i.redd.it/ksx2rvbmqpnd1.png)]** ([Score: 86, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fchdsx/reflection_70b_free_is_broken_now/)): **Reflection 70B** 免费 API 目前无法运行，原因可能是 **Claude 额度耗尽**。尝试访问该服务的用户遇到了错误，这表明底层 AI 模型可能已不再可用，或无法通过免费层级访问。
  - **Reflection 70B** API 停机归因于 **Claude 额度**耗尽，用户猜测开发者的**最终目的**。一篇 [VentureBeat 文章](https://venturebeat.com/ai/meet-the-new-most-powerful-open-source-ai-model-in-the-world-hyperwrites-reflection-70b/) 曾大肆宣传 **GlaiveAI** 是 **OpenAI** 和 **Anthropic** 的威胁，但主流媒体尚未报道这一后续影响。
  - **OpenRouter** 将 API 版本替换为了一个开放权重版本，名称仍为 **Reflection 70B (Free)**。用户质疑 **OpenRouter** 的审核流程，而该公司辩称其在没有进行广泛审查的情况下快速部署了模型。
  - 一些用户认为这次事件与之前的 **Glaive-instruct 3b** 争议如出一辙，表明存在为了融资而炒作模型的模式。其他人则猜测这次损害名誉的事件背后可能存在潜在的干扰项或别有用心。


**主题 2. Reflection 70B 事件带来的社区教训：AI 中的信任与验证**

- **[[好了，就是这个。据说是“那个东西”的新权重。](https://huggingface.co/mattshumer/ref_70_e3)]** ([Score: 67, Comments: 77](https://reddit.com//r/LocalLLaMA/comments/1fc79xd/well_here_it_goes_supposedly_the_new_weights_of/)): 该帖子暗示发布了 **Reflection 70B**（一个大型语言模型）的**新权重**。然而，正如帖子标题中谨慎且不确定的语气所暗示的那样，社区似乎对这次发布的真实性或重要性仍持高度**怀疑**态度。

- **[Reflection 70B 的经验教训]** ([Score: 114, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1fciqfp/reflection_70b_lessons_learned/)): 该帖子强调了 AI 研究中**模型验证**和**基准测试怀疑论**的关键重要性。它建议所有的基准测试都应该从通过仔细检查**识别所使用的特定模型**（例如 **LLAMA**、**GPT-4**、**Sonnet**）开始，并警告不要在没有亲自复制和验证的情况下信任基准测试或 API 的声明。
  - 用户强调了通过 **Lmarena** 和 **livebench** 等平台**验证模型**的重要性，并警告不要信任来自未知来源的未经证实的说法。社区表示需要意识到人们倾向于相信突破性改进的偏见。
  - 越来越多的证据表明 **Matt Shumer** 可能在其 AI 模型声明上撒了谎。一些人猜测这可能是由于心理健康问题，考虑到从项目构思到揭露欺诈的时间跨度非常短。
  - 评论者强调了根据实际使用案例开发**个人基准测试**的重要性，以避免陷入炒作。他们还指出，这次事件突显了人们对开源权重模型很快就能匹配或超越专有选项的期望。

- **[非凡的主张需要非凡的证据，而 Reflection 70B 显然缺乏这一点](https://i.redd.it/o3vu589mvpnd1.jpeg)** ([Score: 177, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1fchtlg/extraordinary_claims_require_extraordinary/)): 帖子标题 "**Extraordinary claims require extraordinary evidence, something Reflection 70B clearly lacks**" 表达了对 **Reflection 70B** 模型相关主张的怀疑。然而，帖子正文仅包含不完整的短语 "Extraordinary c"，未能为作者预期的论点或批评提供足够的上下文来进行有意义的总结。
  - 与私有 API 相比，使用最新的 **HuggingFace release** 进行基准测试时，**Reflection 70B** 的性能显著下降。用户推测私有 API 实际上是 **Claude**，从而引发了对该模型声称的能力的怀疑。
  - 关于 **Matt Shumer** 的最终目的产生了疑问，因为他最终需要交付一个可运行的模型。一些人认为他没有预料到他的主张会获得如此高的关注度，而另一些人则将此情况与 **LK99** 和 **Elon Musk** 的 **FSD** 承诺相提并论。
  - 用户批评 Shumer 缺乏技术知识，指出他在社交媒体上询问关于 **LORA** 的问题。这一事件被视为可能损害他的信誉，一些人甚至将其贴上骗局的标签。


**Theme 3. 围绕 Reflection 70B 争议的梗图与幽默**



- **[你是谁？](https://i.redd.it/ys0636jefond1.png)** ([Score: 363, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fccflj/who_are_you/)): 该帖子展示了一个 **meme**（梗图），描绘了 **Reflection 70B** 对“你是谁？”这一问题的回答前后不一。图像显示了该 AI 模型做出的多个相互矛盾的身份声明，包括自称是 **AI language model**、**人类**，甚至是**耶稣基督**。这个梗图突出了 **AI 模型自我意识不一致**的问题，以及它们产生关于自身身份的矛盾陈述的倾向。
  - **Reflection 70B** 的争议引发了大量的 **memes** 和讨论，用户注意到随着对其真实性的怀疑增加，模型的回答从 **Claude** 变为 **OpenAI** 再到 **Llama 70B**。
  - 一位用户建议 Reflection 背后的开发者正在利用**商业 SOTA 模型**收集数据进行重新训练，旨在最终交付一个能部分实现其主张的模型。其他人则对开发者的真实意图表示怀疑。
  - 帖子提供了一个关于争议的详细解释，描述了该模型最初如何给用户留下深刻印象，但在发布后未能达到预期表现。调查显示，**请求被转发到了热门模型**（如 Claude Sonnet），导致了欺骗指控。


- **[太长不看 (TL;DR)](https://i.redd.it/q7w9pffkbqnd1.jpeg)** ([Score: 249, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1fcj48t/tldr/)): 该帖子仅由一张总结近期 **Reflection 70B** 情况的梗图组成。该梗图使用流行格式幽默地对比了模型发布的预期与现实，暗示 Reflection 70B 的实际表现或影响可能远未达到最初的炒作或预期。
  - **Twitter AI 社区** 因过度炒作 **Reflection 70B** 而受到批评，并提到它实际上是在 Reddit 上接受测试的。用户指出了 **/r/OpenAI** 和 **/r/Singularity** 等子版块中的类似行为。
  - 一些用户对该梗图及其创作者表示困惑或批评，而另一些人则为该发布辩护，指出它提供了**免费访问**与 **Claude Sonnet 3.5** 相当的模型的机会。
  - 一位用户建议，围绕 Reflection 70B 的炒作可能是由于 **OpenAI 转向 B2B SaaS**，这表明开源 AI 社区渴望新的发展。

- **[POV : The anthropic employee under NDA that see all the API requests from a guy called « matt.schumer.freeaccounttrial27 »](https://i.redd.it/cdeby1teopnd1.jpeg)** ([Score: 442, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1fch5zp/pov_the_anthropic_employee_under_nda_that_see_all/))：一名受 **NDA** 约束的 Anthropic 员工观察到一个名为 "**matt.schumer.freeaccounttrial27**" 的可疑账户发出的 **API 请求**。该用户名暗示其可能试图规避**免费试用限制**或进行未经授权的访问，引发了对 Anthropic **API** 服务中**账户滥用**和**安全影响**的担忧。
  - 用户们调侃了 **API 滥用**的潜在后果，其中一条评论建议，随着诈骗策略的升级，身份可以从“*IT 部门的 Matt*”演变为“*关塔那摩牢房里的 Matt*”。
  - 讨论帖随后转向幽默风格，出现了关于 **Anthropic 雇佣猫咪**的评论，包括“喵 🐱”和“*作为一只猫，我可以证实这一点*”等俏皮回复。
  - 一些用户对该帖子本身提出了批评，有人建议发起“**浪费我们时间的集体诉讼**”，另一个人则指出原帖中误用了“**POV**”（Point of View）一词。


**Theme 4. Advancements in Open-Source AI Models and Tools**



- **[gemma-2-9b-it-WPO-HB surpassed gemma-2-9b-it-simpo on AlpacaEval 2.0 Leaderboard](https://i.redd.it/n2medlnyflnd1.jpeg)** ([Score: 30, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1fbyu6l/gemma29bitwpohb_surpassed_gemma29bitsimpo_on/))：**gemma-2-9b-it-WPO-HB** 模型在 **AlpacaEval 2.0 Leaderboard** 上表现优于 **gemma-2-9b-it-simpo**，得分为 **80.31**，而后者为 **79.99**。这一改进证明了 **WPO-HB (Weighted Prompt Optimization with Human Baseline)** 技术在增强模型指令遵循任务性能方面的有效性。
  - **WPO (Weighted Preference Optimization)** 技术在[最近的一篇论文](https://arxiv.org/html/2406.11827v1)中有所详述，其中 "hybrid" 是指偏好优化数据集中包含了**人工生成数据和合成数据**的混合。
  - **AlpacaEval 2.0** 可能需要更新，因为它目前使用 **GPT4-1106-preview** 进行人类偏好基准测试。建议包括使用 **gpt-4o-2024-08-06** 并使用 **claude-3-5-sonnet-20240620** 进行验证。
  - **gemma-2-9b-it-WPO-HB** 模型可在 [Hugging Face](https://huggingface.co/wzhouad/gemma-2-9b-it-WPO-HB) 上获取，它在不同排行榜上均超越了 **gemma-2-9b-it-simpo** 和 **llama-3-70b-it**，引发了进一步测试的兴趣。


- **New upstage release: SOLAR-Pro-PT** ([Score: 33, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1fbunv7/new_upstage_release_solarpropt/))：Upstage 发布了 **SOLAR-Pro-PT**，这是一个可在 **Hugging Face** 上获取的新预训练模型。该模型可通过 [upstage/SOLAR-Pro-PT](https://huggingface.co/upstage/SOLAR-Pro-PT) 访问，不过目前关于其功能和架构的详细信息还很有限。
  - 用户推测 **SOLAR-Pro-PT** 可能是一个**扩展规模的 Nemo 模型**。之前的 **SOLAR 模型**因其相对于体积的性能表现给用户留下了深刻印象。
  - 该模型的**条款和条件**禁止重新分发，但允许进行微调并开源生成的模型。一些用户建议在空数据集上对其进行微调，以创建量化版本。
  - 人们期待 **nousresearch** 对该模型进行微调，因为他们之前的 **Open Hermes solar 微调版本**在编码和推理任务中备受推崇。

- **支持文本、图像、音频和多模态模型的本地推理 Ollama 替代方案** ([Score: 54, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fc3yjt/ollama_alternative_for_local_inference_across/))：**Nexa SDK** 是一个新的工具包，支持使用 **ONNX** 和 **GGML** 格式进行 **文本**、**音频**、**图像生成** 和 **多模态模型** 的本地推理。它包含一个带有 **JSON schema**、支持 **function calling** 和 **streaming** 的 **OpenAI 兼容 API**，以及一个用于轻松测试和部署的 **Streamlit UI**。它可以在任何带有 Python 环境的设备上运行，并支持 **GPU 加速**。开发者正在为该项目寻求社区反馈和建议，该项目已在 **GitHub** 上发布：[https://github.com/NexaAI/nexa-sdk](https://github.com/NexaAI/nexa-sdk)。
  - 社区请求了对 **AMD GPU** 的 **ROCm 支持**，开发者计划在下周添加。该 SDK 已经支持 **ONNX** 和 **GGML** 格式，这些格式现已具备 ROCm 兼容性。
  - 一位用户将 Nexa SDK 与 **Ollama** 进行了比较，并提出了改进建议，如确保模型准确性、提供清晰的更新信息，以及改进模型管理和命名规范。
  - 对 Nexa SDK 的建议包括将 **K quantization** 作为默认设置，提供 **I matrix quantization**，并改进模型列表和下载体验，以层级化方式显示不同的量化版本。

## AI Reddit 全面回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型开发与发布**

- **Salesforce 的 xLAM-1b 模型在 function calling 方面超越 GPT-3.5**：一个 10 亿参数的模型在 [function calling 中实现了 70% 的准确率](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)，尽管体积更小，但表现优于 GPT-3.5。

- **Phi-3 Mini 更新支持 function calling**：Rubra AI 发布了更新后的 [具有 function calling 能力的 Phi-3 Mini 模型](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。

- **Reflection API 争议**：一个 [带有 prompt engineering 的 Sonnet 3.5 封装层](https://www.reddit.com/r/singularity/comments/1fcdt85/reflection_api_is_a_sonnet_35_wrapper_with_prompt/) 被当作新模型进行营销，引发了关于 AI 炒作和验证的讨论。

**AI 研究与应用**

- **乳腺癌的病毒疗法**：一位 [病毒学家使用实验性病毒疗法成功治愈了自己复发的乳腺癌](https://www.reddit.com/r/singularity/comments/1fbzjhe/scientist_successfully_treats_her_own_breast/)，引发了关于医学伦理和自我实验的讨论。

- **Waymo 无人驾驶出租车进展**：Waymo [每周提供 100,000 次无人驾驶出租车服务](https://www.reddit.com/r/singularity/comments/1fbxjj4/waymo_giving_100000_robotaxi_rides_per_week_but/) 但尚未盈利，这让人联想到 Uber 和 YouTube 等公司的早期发展策略。

- **AI 生成视频创作**：演示了 [如何使用多种工具创建 AI 生成视频](https://www.reddit.com/r/StableDiffusion/comments/1fbz6d8/viki_the_first/)，包括用于生成的 ComfyUI、Runway GEN.3 以及用于音乐生成的 SUNO。

**AI 开发工具与可视化**

- **TensorHue 可视化库**：一个 [开源的 Python 张量可视化库](https://www.reddit.com/r/MachineLearning/comments/1fbz318/p_tensorhue_a_tensor_visualization_library_info/)，兼容 PyTorch、JAX、TensorFlow、Numpy 和 Pillow，旨在简化张量内容的调试。

**AI 伦理与社会影响**

- **AI 生成艺术的评估**：关于 [将重点从识别 AI 生成艺术转向评估其质量](https://www.reddit.com/r/singularity/comments/1fc3850/we_should_stop_asking_wehter_a_piece_of_art_is_ai/) 的讨论，强调了创意领域对 AI 看法的演变。

**AI 行业与市场趋势**

- **数据增长与 AI 训练**：迈克尔·戴尔（Michael Dell）声称 [全球数据量每 6-7 个月就会翻倍](https://www.reddit.com/r/singularity/comments/1fbvja9/michael_dell_says_the_amount_of_data_in_the_world/)，戴尔科技拥有 120,000 PB 的数据，而先进 AI 模型训练仅使用了 1 PB。

**梗与幽默**

- 一个关于 [OpenAI 发布周期](https://www.reddit.com/r/OpenAI/comments/1fci2rg/openai_preparing_to_drop_their_new_frontier_model/) 以及对新模型期待的幽默视频。


---

# AI Discord 摘要

> GPT4O (gpt-4o-2024-05-13) 汇总的摘要之摘要

**1. AI 模型性能**

- **Reflection 70B 表现不及预期**：**Reflection 70B** 在基准测试中的表现落后于 **Llama 3.1**，引发了对其能力的质疑；独立测试显示其得分较低，且权重发布延迟。
  - [Matt Shumer](https://x.com/mattshumer_/status/1832554497408700466) 承认上传至 Hugging Face 的权重存在问题，并承诺很快会修复。
- **DeepSeek Coder 遭遇困境**：用户报告 **DeepSeek Coder** 出现故障并无法提供任何回复，尽管 [状态页面](https://status.deepseek.com/) 显示正常，但这表明可能存在上游问题。
  - 这加剧了用户对 **API limitations** 和服务不一致性的现有挫败感。
- **CancerLLM 和 MedUnA 推动医疗 AI 进步**：在 [TrialBench](https://x.com/OpenlifesciAI/status/1832476252260712788) 等基准测试的支持下，**CancerLLM** 和 **MedUnA** 正在增强临床应用和医学影像能力。
  - 讨论强调了深入研究医学论文以提高研究可见性的重要性。


**2. AI 工具与集成**

- **Aider 提升工作流效率**：社区成员分享了他们的 **Aider workflows**，集成了 CodeCompanion 等工具以简化项目设置，并强调了清晰规划的重要性。
  - 预计一个经过优化的 System Prompt 将增强 Aider 的 **output consistency**（输出一致性）。
- **OpenInterpreter 的资源管理困扰**：虽然 **01** 应用允许快速访问音频文件，但用户在 **Mac** 上面临性能波动，导致结果不一致。
  - 一位用户表示，由于 **01** 应用的稳定性问题，他们更倾向于使用纯净版的 OpenInterpreter。


**3. 开源 AI 进展**

- **GitHub 开源 AI 面板讨论会**：GitHub 将于下周四 (9/19) 在其旧金山办公室举办一场免费的 [开源 AI 面板讨论会](https://lu.ma/wbc5bx0z)，讨论接入、民主化以及开源对 **AI** 的影响。
  - 小组成员包括来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的代表。
- **Finegrain 的开源图像分割模型**：Finegrain 发布了一款开源 **image segmentation model**，其表现优于闭源替代方案，可在 [Hugging Face](https://huggingface.co/spaces/finegrain/finegrain-object-cutter) 上通过 **MIT License** 获取。
  - 未来的改进包括一种更微妙的提示方法，以实现超越简单边界框的增强型消除歧义功能。


**4. 基准测试与评估**

- **模型训练中的过拟合担忧**：人们对 **overfitting**（过拟合）表示担忧，认为基准测试往往具有误导性，且模型无论规模大小都不可避免地会经历过拟合，这导致了对基准测试可靠性的怀疑。
  - 一位成员希望他们关于 **benchmark issues** 的文章能在 NeurIPS 上获得审阅，并强调了评估面临的挑战。
- **基准测试的局限性得到认可**：成员们分享了关于 **benchmark limitations** 的见解，指出尽管存在缺陷，但基准测试对于比较仍然至关重要。
  - 讨论强调了使用多样化基准测试来衡量 AI 模型的必要性，并指出了对特定数据集过拟合的风险。


**5. AI 社区活动**

- **柏林 AI Hackathon**：**Factory Network x Tech: Berlin AI Hackathon** 定于 **9 月 28-29 日** 在 Factory Berlin Mitte 举行，旨在聚集 50-100 名有动力推动 **AI-driven innovations** 的开发者。
  - 参与者可以在协作环境中改进现有产品或启动新项目。
- **LLVM 开发者大会**：即将于 10 月举行的 **秋季 LLVM 开发者大会** 将包含 **5 场由 Modular 带来的演讲**，主题涵盖 **Mojo** 和 **GPU programming**。
  - 活动结束后，录制课程将在 [YouTube](https://www.youtube.com/@LLVMPROJ) 上发布，这引起了与会者的极大兴趣。


---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Inference API 故障**：用户在通过 **Hugging Face Inference API** 访问私有模型时遇到“凭证错误”（bad credentials），且通常缺乏有用的日志。
   - 建议的解决方案包括验证 API token 设置，以及审查影响功能的最新更新。
- **在 Hugging Face 上微调模型**：讨论指出，在 Hugging Face 上微调的模型可能并不总是能正确上传，导致仓库中缺少文件。
   - 用户建议在转换过程中仔细检查配置并管理较大的模型，以获得最佳结果。
- **AI 艺术生成的挑战**：社区分享了生成高质量 AI 艺术的经验，强调了肢体和手部表现方面持续存在的问题。
   - 有人建议，更简单、更“俗气”的提示词在产生理想结果方面出人意料地更有效。
- **通用近似定理（Universal Approximation Theorem）见解**：成员们分析了 **Universal Approximation Theorem**，引用了 [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem#Arbitrary-width_case) 获取基础细节。
   - 讨论揭示了 **Haykin 的工作** 中的局限性，以及来自 **Leshno et al.** 关于连续性更好的泛化研究。
- **探索医疗 AI 进展**：最近的更新重点介绍了 **CancerLLM** 和 **MedUnA** 在临床应用中的作用，以及 **TrialBench** 等基准测试。
   - 成员们对深入研究医疗论文表现出极大的热情，提升了重要研究的曝光度。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 在基准测试准确率方面表现挣扎**：用户对 **DeepSeek Coder** 的性能表示担忧，指出其可能使用了错误的模型 ID，导致仪表盘上的统计数据较差。
   - 目前两个模型 ID 都指向 **DeepSeek 2.5**，这可能是导致基准测试问题的原因之一。
- **Aider 提升工作流效率**：社区成员分享了他们的 **Aider 工作流**，集成了 **CodeCompanion** 等工具以简化项目设置，并强调了清晰规划的重要性。
   - 引入经过改进的系统提示词（system prompt）预计将增强 Aider 的**输出一致性**。
- **Reflection 70B 表现逊于 Llama3 70B**：**Reflection 70B** 在代码编辑基准测试中得分 **42%**，而 **Llama3 70B** 达到了 **49%**；Aider 的修改版本在某些标签下缺乏必要的功能。
   - 欲了解更多详情，请查看 [排行榜](https://aider.chat/docs/leaderboards/)。
- **V0 更新显示出强劲的性能指标**：专为 **NextJS UIs** 定制的 **v0** 最近更新展示了卓越的能力，用户分享了一个 [YouTube 视频](https://youtu.be/zA-eCGFBXjM?si=p-CuTkCzmlwyW2vy) 展示其潜力。
   - 欲获取更多见解，请访问 [v0.dev/chat](https://v0.dev/chat) 查看演示和更新。
- **关于 AI 对开发者岗位影响的担忧**：成员们对先进的 **AI 工具** 可能如何改变开发者角色表示担忧，引发了关于岗位过度饱和和职业相关性的疑问。
   - 随着 AI 的不断进化，关于开发领域劳动力未来的紧张情绪正在上升。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Reflection API 开放测试**：[Reflection API](https://openrouter.ai/models/mattshumer/reflection-70b:free) 现在可以在 OpenRouter 上免费进行测试，托管版本和内部版本之间存在显著的性能差异。
   - Matt Shumer 表示，托管 API 目前尚未完全优化，预计很快会推出修复版本。
- **ISO20022 在加密货币领域引起关注**：成员们被敦促研究 **ISO20022**，因为它可能会在加密货币的发展中显著影响金融交易。
   - 讨论强调了该标准的意义，反映出人们对其与不断演变的金融格局相关性的兴趣日益浓厚。
- **DeepSeek Coder 面临 API 故障**：用户报告称 **DeepSeek Coder** 响应为零且运行异常，尽管状态页面显示没有报告问题，但这表明可能存在上游问题。
   - 这一复杂情况加剧了用户对现有 API 限制和服务可用性不一致的挫败感。
- **Vertex AI 的 Base64 编码变通方案**：针对 Vertex AI 的 JSON 上传问题设计了一个变通方案；现在建议用户在提交前将整个 JSON 转换为 **Base64**。
   - 该技术源自 [GitHub PR 讨论](https://github.com/saoudrizwan/claude-dev/pull/45#issuecomment-2293115878)，简化了传输过程。
- **多模态模型的集成**：技术人员询问了将本地图像与多模态模型结合的方法，重点关注正确集成的请求格式。
   - 提供了关于将图像编码为 **Base64** 格式以促进直接 API 交互的指导。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA 与 Dreambooth 的对决**：**LoRA** 体积小且易于分享，允许在运行时组合，而 **Dreambooth** 则生成大得多的完整 Checkpoints。
   - 两种训练方法在有限的图像下都能表现出色，**Kohya** 和 **OneTrainer** 处于领先地位，而 **Kohya** 在受欢迎程度方面夺冠。
- **600 美元以下的预算级 GPU 指南**：对于本地图像生成，用户建议在 600 美元预算内考虑二手的 **3090** 或 **2080**，以提升依赖 VRAM 的性能。
   - 增加 VRAM 可确保更好的结果，特别是对于本地训练任务。
- **向后兼容性的最后希望**：用户呼吁新的 **Stable Diffusion** 模型保持与 **SD1.5 LoRA** 的向后兼容性，因为 SD1.5 仍然受到用户的青睐。
   - 对话强调了 **SD1.5** 在构图方面的优势，许多人断言较新的模型尚未超越其效果。
- **内容创作评论：网红与创作者**：针对**网红文化（influencer culture）**出现了一种批评，这种文化迫使内容创作者通过 Patreon 和 YouTube 等平台变现。
   - 一些社区成员渴望回归到商业化程度较低的内容创作，同时平衡网红营销的现实。
- **LoRA 增强图像生成**：用户强调，提高 AI 生成图像的细节在很大程度上取决于工作流的增强，而不仅仅是 Prompt，其中 **LoRA** 被证明是必不可少的。
   - 许多人在图像制作中加入了 **Detail Tweaker XL** 等组合，以实现效果最大化。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **用户对 LM Studio v0.3 表示担忧**：关于 LM Studio v0.3 的反馈显示，用户对移除 v0.2 中的功能感到失望，引发了关于潜在版本降级的讨论。
   - 针对缺失 **system prompts** 和设置调整的担忧，促使开发者向用户保证更新即将发布。
- **模型配置 Bug 影响性能**：用户面临模型配置问题，特别是关于 **GPU offloading** 和 **context length** 设置，影响了助手的消息连续性。
   - 建议的解决方案包括调整 **GPU layers** 并确保 **dedicated VRAM**，因为有用户遇到了 **context overflow** 错误。
- **对训练 Small Language Models 的兴趣**：讨论集中在训练较小语言模型的可行性上，权衡数据集质量和参数数量与预期的 **training loss**。
   - 多位成员强调了支持冷门语言和获取高质量数据集的具体挑战。
- **导航 LM Studio 服务器交互**：用户澄清，与 LM Studio 服务器交互必须发送 **API requests**，而不是通过 Web 界面。
   - 一位用户在掌握了正确的 **API request** 格式后获得了成功，解决了之前的问题。
- **对 Apple 硬件的期待**：围绕 Apple 即将发布的硬件公告存在猜测，特别是关于 **5090 GPU** 及其与之前型号相比的能力。
   - 预期表明，Apple 将在下一波硬件中凭借创新的内存架构保持领先地位。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **订阅取消引发愤怒**：用户对使用泄露的 **promo codes** 后订阅被取消感到沮丧，并报告称 Perplexity 团队的支持响应有限。
   - 许多人正在寻求对此问题的澄清，觉得对自己的订阅状态一无所知。
- **模型使用限制引发混乱**：需要澄清关于模型使用的强制限制，Pro 用户面临 450 次查询的上限，而 **Claude Opus** 用户仅为 50 次。
   - 关于如何在交互过程中准确指定所用模型的问题不断出现，表明缺乏直接的指导。
- **API 响应缺乏深度**：用户注意到 **API** 响应较短，缺乏 Web 响应的丰富性，引发了对默认响应格式的担忧。
   - 他们正在寻找调整参数以增强 **API** 输出的建议，指出了潜在的改进领域。
- **支付方式错误导致挫败感**：许多用户在尝试设置 **API** 访问时报告了支付方式的身份验证问题，多张卡片出现各种错误。
   - 这个问题似乎很普遍，因为其他人也提到了类似的支付挑战，特别是安全码错误消息。
- **Web Scraping 替代方案出现**：讨论转向了 Perplexity 功能的替代方案，提到了 **You.com** 和 **Kagi** 等利用 **Web Scraping** 的其他搜索引擎。
   - 这些选项因有效解决与知识截止（knowledge cutoffs）和生成响应不准确相关的问题而受到关注。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 技术应对审核垃圾信息**：成员们强调了 **Cohere** 的分类技术如何有效地过滤加密货币垃圾信息，维护服务器讨论的完整性。
   - *一位用户评论道：“这是进行愉快对话的必备工具！”，强调了该 Bot 的重要性。*
- **Wittgenstein 发布 LLM Web 应用**：一位成员分享了他们新编写的 LLM Web 应用的 [GitHub 链接](https://github.com/xettrisomeman/llm_simple_app)，并表示期待反馈。
   - 他们确认该应用使用了 **Langchain**，并已部署在云端的 [Streamlit](https://llmsimpleapp-mrzdrd8jxzcxmy5yisnmis.streamlit.app/) 上。
- **对加密货币诈骗者的担忧**：成员们对渗透到 AI 领域的**加密货币诈骗**表示沮丧，这影响了合法技术进步的声誉。
   - 一位爱好者指出，此类垃圾信息在更广泛的讨论中损害了 **AI** 的可信度。
- **探索 Cohere 产品及其应用**：成员们对 Cohere 产品表现出兴趣，并提到了 [Cohere 博客](https://cohere.com/blog)上定期发布的客户使用案例。
   - 在 [Cookbooks](https://docs.cohere.com/page/cookbooks) 中可以找到使用见解和入门代码，为成员的项目提供灵感。
- **无效的 raw prompt 和 API 使用挑战**：成员们讨论了与 `raw_prompting` 参数相关的 **400 Bad Request** 错误，并澄清了如何配置输出。
   - *一位成员指出：“理解对话轮次（chat turns）至关重要”，这强化了 API 文档清晰度的必要性。*

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Reflection 70B 表现平平的基准测试**：最近的评估显示，**Reflection 70B** 在 aider 代码编辑基准测试中得分为 **42%**，低于 **Llama 3.1** 的 **49%**。
   - 这种差异导致了对其能力的怀疑，以及部分模型权重延迟发布的问题，引发了对透明度的质疑。
- **肿瘤学医疗 LLM 的进展**：重点介绍的 **CancerLLM** 和 **MedUnA** 等模型增强了在肿瘤学和医学影像中的应用，在临床环境中展现出前景。
   - 像 [OpenlifesciAI 的推文](https://x.com/OpenlifesciAI/status/1832476252260712788) 详细描述了它们在改善患者护理方面的影响。
- **通过 RL 训练实现 AGI**：讨论强调，通过**强化训练**结合**强化学习 (RL)** 可能会实现 **AGI**。
   - 然而，对于 **Transformer** 在实现**监督语义智能 (SSI)** 方面的有效性仍存在疑问。
- **PlanSearch 引入多样化的 LLM 输出**：**Scale SEAL** 发布了 **PlanSearch**，这是一种通过自然语言搜索促进输出多样性，从而提高 LLM 推理能力的方法。
   - Hugh Zhang 指出，这使得在推理时能够进行**更深层次的推理**，代表了模型能力的战略转变。
- **扩展模型规模以增强推理能力**：通过在多样化、干净的数据集上进行训练，扩展更大规模的模型可能会解决**推理挑战**，从而提高性能。
   - 尽管如此，对于资源需求以及当前认知模拟在实现类人推理方面的局限性，人们仍存有疑虑。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Together AI 的 MLP Kernels 性能超越 cuBLAS**：成员们讨论了 **Together AI 的 MLP kernels** 如何实现 **20% 的速度提升**，并观察到 **SwiGLU** 驱动了性能增长。对话暗示 Tri Dao 将在即将举行的 CUDA MODE IRL 活动中分享更多见解。
   - 这引发了关于与 **cuBLAS** 效率指标对比的询问，并促使了关于在机器学习框架中实现具有竞争力的加速效果的交流。
- **ROCm/AMD 落后于 NVIDIA**：讨论引发了关于为何 **ROCm/AMD** 在 AI 浪潮中难以像 **NVIDIA** 那样获利的担忧，成员们质疑了企业信任问题。尽管 **PyTorch** 兼容 ROCm，但社区共识认为 NVIDIA 的硬件在实际应用中表现更佳。
   - 这些见解引发了对 AMD 在不断变化的 GPU 市场中所做战略决策的推测。
- **Triton Matmul 集成展现潜力**：Thunder 频道会议重点介绍了 **Triton Matmul** 的应用，聚焦于自定义 kernel 的实际集成。感兴趣的人可以观看 [YouTube 视频](https://www.youtube.com/watch?v=i79Op6DXI7c) 回顾。
   - 成员们对 **fusing operations** 的部署表现出热情，并预告了未来在 **Liger kernel** 上的应用。
- **AMD 发布 UDNA 架构**：在 IFA 2024 上，AMD 推出了 **UDNA**，这是一种合并了 **RDNA** 和 **CDNA** 的统一架构，旨在更好地与 NVIDIA 的 **CUDA 生态**竞争。这一战略转型表明了其致力于提升游戏和计算领域性能的决心。
   - 此外，AMD 决定降低旗舰游戏 GPU 的优先级，反映了其扩大在多样化 GPU 应用中影响力的更广泛战略，不再仅仅局限于高端游戏。
- **关于 PyTorch ignore_index 的担忧**：已确认 **Cross Entropy** 中 `ignore_index` 的处理避免了无效内存访问，通过 early returns 有效地管理了各种条件。展示正确处理方式的测试用例让担忧的成员们放了心。
   - 随着性能调优讨论的不断深入，这次交流指出了 kernel 实现中稳健测试的重要性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Reflection Llama-3.1 宣称夺得开源模型桂冠**：新发布的 **Reflection Llama-3.1 70B** 模型被声称是目前最好的开源 **LLM**，利用 **Reflection-Tuning** 增强了推理能力。
   - 用户报告早期的问题已得到解决，鼓励进行进一步测试以获得更好的结果。
- **关于 OpenAI 神秘的 'GPT Next' 的澄清**：成员们对 **GPT Next 是一个新模型**持怀疑态度，OpenAI 澄清这只是一个比喻性术语，没有实际含义。
   - 尽管有了澄清，但在期望不断增高的情况下，由于缺乏具体的更新，挫败感依然存在。
- **运行 Llama 3.1 70B 的硬件需求**：为了成功运行 **Llama 3.1 70B** 等模型，用户需要高规格的 GPU PC 或至少拥有 **8GB VRAM** 的 **Apple Silicon Mac**。
   - 在各种配置上的经验表明，资源不足会严重阻碍性能。
- **通过 Prompt Engineering 增强 AI 输出**：成员们建议使用类似“以 Terry Pratchett 的写作风格”等风格来创意性地提升 AI 回复，展示了 prompt 的适应性。
   - 强调了结构化输出模板和定义的 chunking 策略对于高效 **API** 交互的重要性。
- **关于使用 AI 进行股票分析的辩论**：对于使用 OpenAI 模型进行股票分析持谨慎态度，主张不要在没有历史数据的情况下仅依赖 prompt。
   - 讨论指向了实时更新和传统模型对于全面评估的必要性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **通过 DLHandle 将 C 与 Mojo 集成**：成员们讨论了如何使用 `DLHandle` 动态链接到共享库，从而将 **C** 代码与 **Mojo** 集成，实现两者之间的函数调用。
   - 示例展示了从 C 库加载后成功执行了一个检查数字是否为偶数的函数。
- **LLVM 开发者大会精华**：即将于 10 月举行的 **秋季 LLVM 开发者大会** 将包含 **5 场由 Modular 带来的演讲**，主题涵盖 **Mojo** 和 **GPU 编程**。
   - 与会者表示期待，会议录像预计将在活动结束后发布在 [YouTube](https://www.youtube.com/@LLVMPROJ) 上。
- **Subprocess 实现愿景**：一位成员表示有兴趣在 **Mojo stdlib** 中实现 **Subprocess** 功能，这表明了增强该库的意愿。
   - 有人对在旧硬件上搭建开发环境的挑战表示担忧，强调了资源方面的困难。
- **DType 在 Dict 键中的角色**：讨论集中在为什么 `DType` 不能作为 Dict 的键，并指出 *DType.uint8* 是一个值而非类型。
   - 成员们提到，由于其与具有特定约束的 SIMD 类型紧密相关，更改此实现可能会很复杂。
- **多精度算术探索**：成员们讨论了在 Mojo 中开发多精度整数算术包的潜力，参考了类似于 Rust 的实现。
   - 一位参与者分享了一个 [GitHub 链接](https://github.com/zmalatrax/uint)，展示了该功能在 `uint` 包上的进展。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepMind 资源分配转向**：一位前 DeepMind 员工指出，项目所需的 **算力 (compute)** 很大程度上取决于其 **产品导向**，尤其是在转向 genai 之后。
   - 这一见解引发了关于基础研究可能面临资源缩减的讨论，正如社区普遍存在的怀疑态度所指出的那样。
- **抓取 Quora 数据问题**：成员们探讨了在 AI 训练数据集中使用 **Quora 数据** 的潜力，承认其价值但也对其 **TOS** 表示担忧。
   - 讨论强调了由于严格的监管，抓取数据可能并不可行。
- **发布 TurkishMMLU 数据集**：**TurkishMMLU** 正式发布，并附带了数据集链接和相关的 [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/pull/2283)。
   - 这一补充旨在加强土耳其语的语言模型评估，如 [相关论文](https://arxiv.org/abs/2407.12402) 所述。
- **ML 中幂律曲线的见解**：成员们讨论了 **幂律曲线 (power law curves)** 如何有效地模拟 ML 中的 **性能缩放 (performance scaling)**，并参考了与估计任务中的缩放定律相关的统计模型。
   - 一位成员指出 *LLM loss 的缩放定律* 与统计估计中的缩放定律相似，表明均方误差按 **N^(-1/2)** 缩放。
- **探索 Adaptive Transformers**：讨论集中在“带有 Adaptive Transformers 的持续上下文学习”，这允许 Transformer 在不改变参数的情况下，利用先验知识适应新任务。
   - 该技术旨在实现高适应性，同时最大限度地降低灾难性遗忘风险，吸引了多个领域的关注。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Reflection API 性能受到质疑**：**Reflection 70B 模型**面临审查，被怀疑只是在 **Llama 3.0** 之上针对基准测试集训练的 **LoRA**；由于评估存在缺陷，其顶级性能的说法具有误导性。
   - 最初的私有 API 测试结果优于公开版本，引发了对各版本间**不一致性**的担忧。
- **AI 模型发布实践遭到批评**：围绕在没有稳健验证的情况下发布重大模型公告的**不专业行为**引发了辩论，导致社区对 AI 能力产生不信任。
   - 成员们敦促行业在公开发布声明前执行更严格的评估标准，并指出预期膨胀的趋势令人担忧。
- **OpenAI 成员转投 Anthropic 引发讨论**：讨论集中在 **OpenAI** 联合创始人 **John Schulman** 转投 **Anthropic** 一事，这被描述为一种超现实的现象，突显了领导层的变动。
   - 关于频繁提到“来自 OpenAI（现就职于 Anthropic）”的调侃捕捉到了社区动态的变化。
- **围绕 GPT Next 的投机性热议**：**KDDI Summit** 演示文稿中出现了一个标记为 **GPT Next** 的模型并引发猜测，OpenAI 澄清这只是一个**比喻性的占位符**。
   - 公司发言人指出，图形表示仅具说明性，并不代表未来发布的路线图。
- **内部官僚主义拖慢 Google 速度**：一位前 Googler 表达了对 **Google** 内部**严重官僚主义**的担忧，称众多的内部利益相关者阻碍了项目的有效执行。
   - 这种情绪强调了员工在大组织中面临的挑战，内部政治往往会阻碍生产力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Codex 助力 Cursor**：用于 Cursor 的新 [AI Codex](http://codex.md) 实现了自我改进功能，如自动保存洞察和智能分类。
   - 成员们建议，使用一个月后可能会揭示其效率方面的宝贵学习成果。
- **Reflection API 令人侧目**：**Reflection API** 似乎是一个 **Sonnet 3.5** 的 wrapper，据报道它过滤掉了对 **Claude** 的引用以掩盖其身份。
   - 各种评估表明其性能可能与宣传不符，引发了对基准测试方法的质疑。
- **Apple 大胆推进 AI 进展**：Apple 最近的活动预告了 **Apple Intelligence** 的重大更新，暗示了可能改进的 **Siri** 和即将推出的 AI 手机。
   - 这引发了对竞争影响的兴奋，许多成员呼吁 Apple 工程师分享见解。
- **Gemini 推出全新 Enum Mode**：Logan K 宣布在 **Gemini API** 中推出 **Enum Mode**，通过允许从预定义选项中进行选择来增强结构化输出。
   - 这项创新旨在简化开发者与 **Gemini** 框架交互时的决策过程。
- **对写实 LoRA 模型的关注**：一位用户展示了一个**写实 LoRA** 模型，其细节处理能力正吸引着 **Stable Diffusion** 社区。
   - 围绕其表现（尤其是意想不到的动漫图像）的讨论引起了极大关注。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 的资源管理困扰**：虽然 **01** app 允许快速访问音频文件，但用户在 **Mac** 上面临性能波动，导致结果不一致。
   - 一位用户表示，由于 **01** app 的稳定性问题，他们更倾向于使用原生的 OpenInterpreter。
- **呼吁在 OpenInterpreter 中加入 AI Skills**：用户渴望为标准版 OpenInterpreter 发布 **AI Skills**，而不仅仅是 **01** app，这显示了对增强功能的需求。
   - 用户对 **01** app 相对于基础版 OpenInterpreter 的性能表现感到沮丧。
- **01 Light 停产及退款**：团队宣布正式结束 **01 Light** 项目，重点转向免费的 **01 app**，并正在处理所有硬件订单的退款。
   - 焦急等待设备的用户中普遍存在失望情绪，但官方保证将通过 help@openinterpreter.com 处理退款。
- **Scriptomatic 在开源模型上的成功**：一名成员成功将 **Scriptomatic** 与开源模型的结构化输出集成，并计划很快提交 PR。
   - 他们对 **Dspy** 提供的支持表示感谢，并强调了他们涉及 *grepping and printing* 的系统化方法。
- **Instructor 库增强 LLM 输出**：分享了 [Instructor](https://pypi.org/project/instructor/) 库，该库旨在通过基于 Pydantic 的易用 API 简化 LLM 的结构化输出。
   - *Instructor* 将简化验证、重试和流式传输，从而增强用户的 LLM 工作流。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 llama-deploy 部署 Agentic 系统**：探索这个使用 **LlamaIndex** 和 **getreflex** 将 Agentic 系统部署为微服务的[全栈示例](https://t.co/jL14R8cJMD)。
   - 这种设置简化了聊天机器人系统，使其成为追求效率的开发者的首选。
- **轻松运行 Reflection 70B**：如果你的笔记本电脑支持，现在可以使用 Ollama 直接从 **LlamaIndex** 运行 **Reflection 70B**（[详情点击此处](https://t.co/ZkF05l159I)）。
   - 这种能力允许在没有大规模基础设施要求的情况下进行动手实验。
- **构建高级 RAG 流水线**：查看这篇关于使用 **Amazon Bedrock** 构建具有[动态查询路由](https://t.co/mzJzDMGhM2)的高级 Agentic RAG 流水线的指南。
   - 该教程涵盖了有效优化 RAG 实现的所有必要步骤。
- **自动化财务分析工作流**：一篇博客文章讨论了创建一个 Agentic 总结系统，用于自动化季度和年度财务分析（[阅读更多](https://t.co/ktj55fQSlZ)）。
   - 这种方法可以显著提高财务报告和洞察的效率。
- **RAG 环境的动态 ETL**：了解 LLM 如何通过特定数据的决策来自动化 ETL 过程，如本[教程](https://t.co/6yZmHoUjCW)所述。
   - 这种方法通过适应不同的数据集特征，增强了数据提取和过滤。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Gemma 模型配置更新**：为了使用 **Torchtune** 配置 **Gemma 9B** 模型，用户建议根据 [config.json](https://huggingface.co/google/gemma-2-9b/blob/main/config.json) 中的特定参数修改配置中的 `model` 条目。
   - 该方法利用了组件构建器（component builder），旨在为各种模型规模提供灵活性。
- **Torchtune 中支持 Gemma 2 的挑战**：讨论围绕在 **Torchtune** 中支持 **Gemma 2** 的困难展开，主要原因是 **logit-softcapping** 问题和带宽限制。
   - **Gemma 2** 中迅速发展的架构改进导致了大量待实现的请求功能积压。
- **Torchtune 的拟议增强功能**：强调了一个关于 **Torchtune** 中填充序列（padding sequence）行为的潜在 Bug，并提出了一个 **PR**，通过澄清 flip 方法来修复该问题。
   - 目标是实现与 **torch pad_sequence** 的功能对等，增强库的整体功能。
- **生成过程中的缓存处理需要改进**：用户讨论了生成过程中缓存行为的修改需求，建议在 attention 模块的连续前向调用中使用 `torch.inference_mode`。
   - 尽管如此，他们承认为 `.forward()` 设置显式标志可能会产生更稳健的解决方案。
- **分块线性方法（Chunked Linear Method）实现参考**：一位成员分享了对来自 [GitHub gist](https://gist.github.com/Chillee/22cd93e11b887db1f596ab754d60a899) 的分块线性结合交叉熵的整洁实现的兴趣，将其作为 **Torchtune** 的潜在增强功能。
   - 由于该库目前将 LM-head 与损失计算分离，集成此方法可能会面临挑战。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **.astream_events() 解码困境**：用户报告了在解码来自 **.astream_events()** 的流时遇到的挑战，特别是通过各种分支和事件类型进行繁琐的手动序列化。
   - 参与者强调缺乏有用的资源，呼吁提供参考实现以减轻这一过程的负担。
- **Gradio 并发处理困难**：在启动具有 10 个标签页的 **Gradio** 后，尽管并发限制更高，但仅生成了 6 个请求，这暗示了潜在的配置问题。
   - 用户指出了硬件限制，建议进一步调查处理并发请求的方法。
- **Azure OpenAI 集成面临 500 错误**：一位用户在与 Azure OpenAI 交互时遇到 **500 错误**，引发了对端点参数的查询。
   - 建议包括验证环境变量和命名规范，以潜在地解决这些排错难题。
- **VAKX 提供无代码 AI 助手构建**：**VAKX** 作为一个无代码平台被引入，使用户能够构建 AI 助手，具有 **VAKChat** 集成等功能。
   - 鼓励成员探索 [VAKX](https://vakx.io) 和 [Start Building for Free](https://studio.vakx.io) 链接以进行快速设置。
- **Selenium 与 GPT-4 Vision 集成**：一个实验性项目展示了 **Selenium** 与 **GPT-4 vision model** 的集成，详细过程可在 [此 YouTube 视频](https://youtu.be/nTtZnzYS_24) 中查看。
   - 围绕利用此集成通过向量数据库进行更有效的自动化测试引发了关注。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **过拟合问题成为讨论焦点**：成员们提出了关于 **overfitting** 的问题，强调基准测试可能会误导预期，并暗示无论模型规模如何，模型都不可避免地会经历过拟合。
   - *“我不再相信基准测试了”* 表达了对基于不充分数据的模型评估可靠性的怀疑。
- **基准测试的局限性受到审视**：分享了关于 **benchmark** 局限性的见解，揭示了尽管存在缺陷，它们对于模型间的比较仍然至关重要。
   - 一位成员对他们关于 **benchmark** 问题的文章能在 NeurIPS 上接受评审表示乐观，强调了当前的评估挑战。
- **AI 工具被揭露为骗局**：最近一款被大肆宣传的 **AI 工具** 被证实是骗局，它虚假地声称可以与 **Claude 3.5** 或 **GPT-4** 相媲美。
   - 讨论强调了此类骗局造成的 **时间损失** 及其在各个频道中产生的干扰性质。
- **关于 RAG API 的紧急咨询**：一位成员紧急寻求 **RAG APIs** 的使用经验，由于其模型尚未就绪，需要立即为项目提供支持。
   - 他们强调了 **24/7 托管** 成本的挑战，并寻求有效管理其 AI 项目的替代方案。
- **H100 的 8-Bit 加载限制受到质疑**：一位成员询问为什么 **H100** 不支持以 **8-bit** 格式加载模型，寻求关于此限制的澄清。
   - 他们重申了获取有关 **H100** 在 **8-bit 模型加载** 方面限制见解的紧迫性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **柏林 AI 黑客松预示创新**：**Factory Network x Tech: Berlin AI Hackathon** 定于 **9月28-29日** 在 Factory Berlin Mitte 举行，旨在聚集 50-100 名有动力推动 **AI 驱动创新** 的开发者。
   - 参与者可以在协作环境中改进现有产品或启动新项目，培养创造性方法。
- **Finegrain 的开源突破**：Finegrain 发布了一个开源的 **图像分割模型**，其性能优于闭源替代方案，可在 Hugging Face 上通过 **MIT License** 获取。
   - 未来的改进包括一种更微妙的提示方法，以增强歧义消除和超出简单边界框的可用性。
- **Concrete ML 面临扩展问题**：讨论指出 **Concrete ML** 需要 **Quantization Aware Training (QAT)** 才能与同态加密有效集成，这可能导致潜在的性能折衷。
   - 成员们提出了对文档有限的担忧，特别是其在机器学习中大型模型上的适用性。
- **免费开源 AI 面板活动**：GitHub 将于 **9月19日** 在旧金山举办 **开源 AI 面板** 活动，邀请了来自 **Ollama** 和 **Nous Research** 等组织的知名嘉宾。
   - 虽然可以免费参加，但由于座位有限，必须预先注册，因此尽早报名至关重要。
- **AI 中的多模态引起关注**：AI 中 **multimodality** 的兴起受到了关注，例如 **Meta AI transfusion** 和 **DeepMind RT-2** 等示例展示了重大进展。
   - 讨论建议研究采用 RAG、API 交互、网页搜索和 Python 执行等技术的 **工具增强生成 (tool augmented generation)**。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LanceDB 集成 PR 已提交**：一名成员提交了 [LanceDB 集成 PR](https://github.com/stanfordnlp/dspy/pull/1444)，将其作为检索器 (retriever) 添加到项目中，以处理大型数据集。
   - 他们请求特定用户对评审过程提供反馈和修改建议，强调在功能增强方面的协作。
- **对 GPT-3.5 弃用的复杂情绪**：成员们讨论了在 **GPT-3.5** 弃用后不同的用户体验，指出模型性能不一致，尤其是像 **4o-mini** 这样的模型。
   - 一名用户建议使用顶级的闭源模型作为教师模型来指导低端模型，以提高性能的一致性。
- **AttributeError 困扰 MIPROv2**：有用户报告在 **MIPROv2** 中遇到 `AttributeError`，表明 `GenerateModuleInstruction` 函数中可能存在潜在问题。
   - 讨论围绕建议的修复方案展开，一些成员指出 **CookLangFormatter** 代码中可能存在问题。
- **微调小型 LLM 引发热议**：一名成员分享了使用独特的 **reflection** 数据集微调小型 LLM 的成功经验，该模型已在 Hugging Face 上提供交互。
   - 他们提供了链接，同时鼓励其他人探索他们在该领域的发现。
- **CookLangFormatter 问题备受关注**：成员们辩论了 **CookLangFormatter** 类中的潜在问题，识别出方法签名中的错误。
   - 在修改后，一名用户报告了积极的结果，并建议在 GitHub 上记录该问题以供未来参考。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **WebGPU PR #6304 引起轰动**：由 **geohot** 提交的 [WebGPU PR #6304](https://github.com/tinygrad/tinygrad/pull/6304) 标志着在 **Asahi Linux** 上恢复 **WebGPU** 功能的重大努力，并附带 **$300 悬赏金 (bounty)**。
   - 一名成员指出：*“这是该计划的一个充满希望的开始，”* 强调了社区对该提案的兴奋。
- **多 GPU Tensor 问题使开发复杂化**：开发者在进行 **multi-GPU** 操作时遇到 **AssertionError**，该操作要求所有缓冲区共享同一个设备。
   - 一名沮丧的用户评论道：*“我花了足够的时间……确信这个目标与 tinygrad 目前处理多 GPU tensor 的方式是正交的。”*
- **GGUF PR 面临延迟和困惑**：关于多个 **GGUF PR** 停滞状态的担忧日益增加，这些 PR 缺乏合并和明确的项目方向。
   - 一名用户询问了 GGUF 的 **roadmap**，强调了对后续指导的需求。
- **模型分片 (Model Sharding) 的挑战**：讨论揭示了模型分片的问题，即某些设置在单 GPU 上运行正常，但在扩展到多个设备时会失败。
   - 一名用户观察到 *“George 对我的变通方案表示反对……”*，表明围绕解决方案存在复杂的对话。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **xLAM Prompt 偏离标准**：成员们讨论了 **xLAM** 使用的独特 **system prompt**，详见 [Hugging Face 模型卡](https://huggingface.co/Salesforce/xLAM-7b-fc-r#basic-usage-with-huggingface)。
   - 这引发了关于个性化 Prompt 如何偏离 **BFCL** 默认设置的分析。
- **LLaMA 缺乏 Function Calling 的清晰说明**：参与者指出 **LLaMA** 没有提供关于 **function calling** 的文档，引发了对 Prompt 格式的担忧。
   - 虽然被归类为 Prompt 模型，但由于文档不足，**LLaMA** 对 Function Calling 的处理仍然模糊不清。
- **GitHub 冲突导致集成延迟**：一名用户报告其 Pull Request [#625](https://github.com/ShishirPatil/gorilla/pull/625) 面临合并冲突，阻碍了合并进程。
   - 在解决冲突后，他们重新提交了一个新的 Pull Request [#627](https://github.com/ShishirPatil/gorilla/pull/627) 以促进集成。
- **探索通过 VLLM 进行模型评估**：有人询问在设置 **VLLM** 服务后如何评估模型。
   - 该询问反映了社区对模型评估方法论和最佳实践的浓厚兴趣。
- **介绍 Hammer-7b 处理器 (Handler)**：社区讨论了新的 **Hammer-7b** 处理器，强调了相关 Pull Request 中概述的功能。
   - 包含 [CSV 表格](https://github.com/ShishirPatil/gorilla/pull/625) 的详细文档突出了模型准确性和性能指标。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **4090 GPU 支持更大的模型**：凭借 **4090 GPU**，工程师可以并发运行更大的 embedding 模型，包括 **Llama-8b**，并应考虑使用 **3.1** 版本以获得增强的性能。
   - 这种配置提升了处理任务的效率，并允许更复杂的模型平稳运行。
- **Milvus 的混合搜索魔力**：讨论强调了在 Milvus 上结合使用 **BGE** 和 **BM25** 的混合搜索，并参考了 [GitHub repository](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py) 中的示例。
   - 该示例有效地展示了如何结合 sparse 和 dense 混合搜索以改进数据检索。
- **通过 Reranking 提升结果**：实现一个利用每个 chunk 的 metadata 的 **reranker**，有助于优先排序并优化结果排序。
   - 该方法旨在增强数据处理能力，使检索到的信息更具相关性和准确性。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **理解基于 RAG 的检索评估**：一位成员询问了在特定领域背景下评估 **RAG based retrieval** 系统所需的 **evaluation metrics**（评估指标）。
   - 他们不确定是将自己的 **RAG approach** 与其他 **LLMs** 进行比较，还是针对不使用 RAG 的结果进行评估。
- **RAG 的比较策略**：同一位成员思考是仅进行有无 RAG 的对比，还是也要与其他 **large language models** 进行对比。
   - 这个问题引起了兴趣，促使成员们考虑在项目中评估 RAG 有效性的各种方法。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GitHub 主办开源 AI 研讨会**：GitHub 将于下周四 (9/19) 在其旧金山办公室举办一场免费的 [开源 AI 研讨会](https://lu.ma/wbc5bx0z)，旨在讨论 **AI** 的准入、民主化以及开源的影响。
   - 小组成员包括来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的代表，为 AI 社区的重要对话做出贡献。
- **AI 研讨会需要注册审批**：与会者需要注册活动，注册需经主办方批准，以便有效管理参会人数。
   - 随着 AI 领域对该活动的兴趣日益增长，此流程旨在确保环境可控。



---


**Mozilla AI Discord** 没有新消息。如果该社区长时间保持安静，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该社区长时间保持安静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间保持安静，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1281700054922559569)** (930 条消息🔥🔥🔥): 

> - `Hugging Face Inference API 问题`
> - `模型微调经验`
> - `AI 艺术与 Prompting 挑战`
> - `关于 LLM 特性与使用的问答` 


- **Hugging Face Inference API 问题**：用户在使用 Hugging Face Inference API 时遇到困难，特别是在尝试访问私有模型时，会导致“凭证错误”且没有有用的日志。
   - 建议的解决方案包括确保正确设置 API tokens，并评估可能影响功能的最新更新。
- **模型微调经验**：讨论了在 Hugging Face 上微调模型的过程，用户指出生成的模型可能并不总是能正确上传，导致 repository 中缺少文件。
   - 用户建议检查配置并处理大型模型，特别是在为本地托管转换 GGUF 等格式时。
- **AI 艺术与 Prompting 挑战**：对话探讨了生成高质量 AI 艺术的挑战，特别是关注生成图像中肢体和手部表现的问题。
   - 强调了使用有效 prompt 的重要性，用户建议更简单、更直白的 prompt 通常会产生更好的结果。
- **关于 LLM 特性与使用的问答**：用户询问了关于语言模型和 **vLLM** 等工具的有效本地托管选项，并讨论了 batching 和不同推理方法的实用性。
   - 提到各种模型（如 Mistral 和 Llama）突显了人们对其在实际应用中的性能和可用性的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="http://127.0.0.1:7860"">未找到标题</a>: 未找到描述</li><li><a href="https://stackoverflow.com/questions/48497566/401-client-error-unauthorized-for-url">401 Client Error: Unauthorized for url</a>: 最近在使用 soundcloud (0.5.0) Python 库时，我开始遇到 requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://api.soundcloud.com/oauth2/token...</li><li><a href="https://civitai.com/user/datavoid">Civitai | 分享你的模型</a>: 未找到描述</li><li><a href="https://colab.research.google.com/">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/freeCS-dot-org/Artificium-llama-3.1-8B">Meta-Llama3.1-8B - 由 freeCS-dot-org 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/karate-kid-wax-rotate-car-training-gif-4993063">龙威小子 GIF - Karate Kid Wax Rotate - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/shafire/talktoaiZERO">shafire/talktoaiZERO · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/camenduru/joy-caption-jupyter/blob/main/joy_caption_jupyter.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/chat-ui/configuration/models/providers/tgi">Text Generation Inference (TGI)</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/error-401-client-error-unauthorized-for-url/19714">Error 401 Client Error: Unauthorized for url</a>: 当在带有 LM 的私有语音识别模型中使用 model card 时，我遇到了这个错误：401 Client Error: Unauthorized for url: https://huggingface.co/api/models/taliai/tali-asr-with-lm/revision/main...</li><li><a href="https://huggingface.co/spaces/SmilingWolf/wd-tagger">WaifuDiffusion Tagger - 由 SmilingWolf 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/dies-cat-dead-died-gif-13827091">Dies Cat GIF - Dies Cat Dead - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lu.ma/15w6fhbv">Gen Battle SF: 让我们用 AI 制作音乐视频！ · Luma</a>: 让我们分组制作音乐视频！AI 初学者和专家将分成小组共同创作短片。到今晚结束时，我们将……</li><li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForestClassifier</a>: 图库示例：scikit-learn 1.4 发布亮点、scikit-learn 0.24 发布亮点、scikit-learn 0.22 发布亮点、分类器校准比较、概率校准...</li><li><a href="https://tenor.com/view/napoleon-dynamite-kip-yes-gif-5860703">拿破仑炸药 Kip GIF - Napoleon Dynamite Kip Yes - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/cocktailpeanut/status/1832952487541658077">来自 cocktail peanut (@cocktailpeanut) 的推文</a>: OpenAI 正准备发布他们的新模型</li><li><a href="https://huggingface.co/shafire">shafire (Shafaet Brady Hussain)</a>: 未找到描述</li><li><a href="https://tenor.com/view/kamala-harris-real-though-gif-10836306879417478300">Kamala Harris Real Though GIF - Kamala harris Real though - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/joe-biden-presidential-debate-huh-confused-gif-16704157274113773062">乔·拜登总统竞选辩论 GIF - Joe biden Presidential debate Huh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/steve-brule-orgasm-funny-chills-gif-8291454">Steve Brule Orgasm GIF - Steve Brule Orgasm Funny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/tim-and-eric-spaghetti-funny-face-gif-14238957">Tim And Eric Spaghetti GIF - Tim And Eric Spaghetti Funny Face - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ohearn-sad-ohearn-mike-ohearn-sad-mike-sad-gif-13532193191719643333">Ohearn Sad Mike Ohearn Sad GIF - Ohearn sad Ohearn Mike ohearn sad - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/shafire/talktoaiZERO/tree/main">shafire/talktoaiZERO (main 分支)</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces">管理你的 Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/empire-i-got-you-brothers-lucious-terrence-howard-gif-4652933">Empire'S Got Your Back GIF - Empire I Got You Brothers - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/btc-blockchain-fud-cryptocurrency-crypto-gif-14490386">Btc Blockchain GIF - Btc Blockchain Fud - 发现并分享 GIF</a>: 点击查看</li>

w the GIF</li><li><a href="https://tenor.com/view/hello-gif-11025697">Hello GIF - Hello - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtu.be/7NWRnWZghGA">Data Visualization :  Bar Chart and Heat Map</a>: 在这段视频中，我将讨论条形图和热力图，解释它们的工作原理以及它们在数据中揭示的趋势，以及其他相关主题。如果你...</li><li><a href="https://huggingface.co/shafire/talktoai/tree/main">shafire/talktoai at main</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1282191242859646996)** (9 messages🔥): 

> - `CMOS 微电路中的锁定效应 (Latch-up effect)`
> - `将未审查模型部署到 SageMaker`
> - `每日学习进度论坛` 


- **理解 CMOS 中的锁定效应 (Latch-up Effect)**：一位成员询问了 CMOS 微电路中的 **Latch-up 效应**，寻求关于其工作原理的信息。
   - 该话题仍处于开放状态，等待知识渊博的成员进一步讨论和澄清。
- **分享 SageMaker 部署见解**：一位成员根据 Hugging Face 文档，询问了关于 **将未审查模型部署到 SageMaker** 的经验和指导。
   - 另一位成员提到他们正在研究类似的问题，并后续指出进展相当顺利。
- **通过每日进度激发社区动力**：一位成员询问该频道是否像论坛一样用于发布每日学习进度，类似于 **100 days of code**。
   - 其他成员确认了这种设置，旨在激励每个人的学习之旅。
- **对协作的赞赏**：一位成员对另一位用户的作品表示钦佩，称其“令人惊叹”，而原作者则将功劳归于 **Nvidia 和 Epic Games** 的贡献。
   - 这突显了社区内的协作精神和认可。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1281777990975492138)** (11 messages🔥): 

> - `医疗 AI 研究更新`
> - `AlphaProteo 蛋白质预测模型`
> - `医疗 LLMs 应用`
> - `ML 训练可视化工具`
> - `探索医学文献` 


- **上周医疗 AI 亮点**：最新更新涵盖了多个前沿医疗 LLM，包括 **CancerLLM** 和 **MedUnA**，以及它们在临床任务中的应用。
   - *TrialBench* 和 *DiversityMedQA* 被认为是评估 LLM 在医疗应用中性能的重要基准。
- **DeepMind 的 AlphaProteo 模型彻底改变蛋白质设计**：来自 Google DeepMind 的 **AlphaProteo** 模型可以预测蛋白质与分子的结合，增强了药物设计等生物工程应用。
   - 这一全新的 AI 系统旨在通过改进蛋白质相互作用来增进我们对生物过程的理解，正如其 [博客文章](https://deepmind.google/discover/blog/alphaproteo-generates-novel-proteins-for-biology-and-health-research/) 中所强调的那样。
- **深入研究医学论文的兴趣**：成员们对进一步探索医学论文表现出极大的热情，提高了医疗 AI 领域研究的可见性。
   - 有人建议围绕最新研究更新中列出的近期论文进行更深入的讨论。
- **关于 AlphaProteo 开放获取的咨询**：有人提出了关于 Google DeepMind 的 **AlphaProteo** 模型是否 **开放获取 (open access)** 的问题。
   - 这反映了研究社区对先进 AI 工具可访问性的持续讨论。
- **ML 训练曲线可视化工具**：一位成员询问了自动生成 ML 模型（特别是图像分类模型）训练和验证曲线的框架和工具。
   - 这强调了对提高模型训练过程效率的可视化方法的持续关注。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1832476252260712788">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>: 上周医疗 AI：顶级研究论文/模型 🏅（2024 年 9 月 1 日 - 9 月 7 日） 医疗 LLM 及其他模型： - CancerLLM：癌症领域的语言大模型 - MedUnA：视觉语言...</li><li><a href="https://huggingface.co/posts/aaditya/989215269740443">Hugging Face 上的 @aaditya: &quot;上周医疗 AI：顶级研究论文/模型 🏅（9 月 1 日 -…&quot;</a>: 未找到描述</li><li><a href="https://deepmind.google/discover/blog/alphaproteo-generates-novel-proteins-for-biology-and-health-research/">AlphaProteo 为生物学和健康研究生成新型蛋白质</a>: 新的 AI 系统设计的蛋白质能成功与目标分子结合，具有推进药物设计、疾病理解等方面的潜力。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1281779452740636734)** (51 messages🔥): 

> - `PowershAI 特性`
> - `GraphRAG 利用`
> - `Om LLM 架构`
> - `FLUX.1 [dev] 模型发布`
> - `OCR 纠错技术` 


- **PowershAI 简化 AI 集成**：PowershAI 旨在通过 PowerShell 命令轻松集成和调用 AI 模型，从而简化 Windows 用户的 AI 使用体验，并增强脚本的面向对象能力。
   - 它支持 function calling 和 Gradio 集成等特性，帮助用户通过多个 AI 源优化工作流。
- **本地 GraphRAG 模型测试**：一个新仓库已创建，允许用户使用来自 Hugging Face 的各种模型测试 Microsoft 的 GraphRAG，突破了 Ollama 提供的有限选项。
   - 这为希望扩展图检索能力的用户提供了更大的灵活性，且无需承担使用 OpenAI API 的相关成本。
- **Om 带来的 LLM 架构创新**：Dingoactual 介绍了一种名为 Om 的新型 LLM 架构，强调了其独特的特性，如初始卷积层和用于处理长上下文输入的 multi-pass memory。
   - 设计改进侧重于在有效管理 VRAM 需求的同时优化处理性能。
- **FLUX.1 [dev] 模型介绍**：FLUX.1 [dev] 模型是一个拥有 120 亿参数的用于图像生成的 flow transformer，现已发布开放权重，允许科学家和艺术家利用其能力。
   - 该模型提供的高质量输出可与领先的闭源替代方案相媲美，增强了创意领域创新工作流的潜力。
- **OCR 纠错与创意文本生成**：Tonic 重点介绍了 Pleiasfr 开发的一种纠错 OCR 输出的技术，该技术也可创意性地用于生成多种语言的历史风格文本。
   - 这种方法体现了利用 AI 进行数据纠错和创意尝试的多功能性与创新性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://livebook.manning.com/book/powershell-in-depth/chapter-34/1">第 34 章。使用组件对象模型 (COM) · PowerShell 深度解析</a>：探索 COM 是什么以及不是什么 · 使用 COM 对象</li><li><a href="https://huggingface.co/spaces/gokaygokay/Reflection-70B-llamacpp">Reflection 70B llama.cpp (正确权重) - gokaygokay 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/rrg92/xtts">Xtts - rrg92 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/lazarzivanovicc/timestretchlora">lazarzivanovicc/timestretchlora · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>：未找到描述</li><li><a href="https://civitai.com/models">Civitai | 分享你的模型</a>：未找到描述</li><li><a href="https://github.com/NotTheStallion/graphrag-local-model_huggingface">GitHub - NotTheStallion/graphrag-local-model_huggingface: Microsoft's graphrag using ollama and hugging face to support all LLMs (Llama3, mistral, gemma2, fine-tuned Llama3 ...).</a>：使用 ollama 和 Hugging Face 的 Microsoft graphrag，支持所有 LLM (Llama3, mistral, gemma2, 微调后的 Llama3 ...)。- NotTheStallion/graphrag-local-model_huggingface</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-Plugin-for-LM-Studio: Plugin that lets you use LM Studio to ask questions about your documents including audio and video files.</a>：让你能够使用 LM Studio 对文档（包括音频和视频文件）进行提问的插件。- BBC-Esq/VectorDB-Plugin-for-LM-Studio</li><li><a href="https://github.com/dingo-actual/om">GitHub - dingo-actual/om: An LLM architecture utilizing a recurrent structure and multi-layer memory</a>：一种利用循环结构和多层记忆的 LLM 架构 - dingo-actual/om</li><li><a href="https://huggingface.co/spaces/Tonic/OCRonos-TextGen">Tonics-OCRonos-TextGen - Tonic 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/AssistantsLab/Tiny-Toxic-Detector">AssistantsLab/Tiny-Toxic-Detector · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/AssistantsLab/Tiny-Toxic-Detector#benchmarks">AssistantsLab/Tiny-Toxic-Detector · Hugging Face</a>：未找到描述</li><li><a href="https://doi.org/10.48550/arXiv.2409.02114">Tiny-Toxic-Detector：一种用于毒性内容检测的紧凑型 Transformer 模型</a>：本文介绍了 Tiny-toxic-detector，这是一种专为毒性内容检测设计的紧凑型 Transformer 模型。尽管只有 210 万个参数，Tiny-toxic-detector 仍取得了具有竞争力的表现...</li><li><a href="https://www.youtube.com/watch?v=e1BY_fQ5TZI">Hugging Face 和 Gradio 登陆 PowershAI：学习如何使用它们</a>：在本视频中，我们将深入探讨 PowershAI 的最新更新：全面支持 Hugging Face 和 Gradio API！你将学习如何使用 PowershAI 连接到...</li><li><a href="https://github.com/rrg92/powershai">GitHub - rrg92/powershai: Powershell + AI</a>：Powershell + AI。通过在 GitHub 上创建账号来为 rrg92/powershai 的开发做出贡献。</li><li><a href="https://github.com/rrg92/powershai/tree/main/docs/en-US">rrg92/powershai 项目 main 分支下的 powershai/docs/en-US</a>：Powershell + AI。通过在 GitHub 上创建账号来为 rrg92/powershai 的开发做出贡献。</li><li><a href="https://civitai.com/models/731347">SECourses 3D Render for FLUX - 完整数据集和工作流共享 - v1.0 | Stable Diffusion LoRA | Civitai</a>：针对 FLUX 风格 Hugging Face 仓库的完整训练教程、指南和研究，包含完整工作流、详细研究细节、过程、结论...
</li>
</ul>

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1281845685377040475)** (6 条消息): 

> - `Universal Approximation Theorem`
> - `Uncensored Models`
> - `Model Definitions`
> - `Leshno's Theorem`
> - `HuggingFace Models` 


- **Universal Approximation Theorem 深度讨论**：成员们讨论了 **Universal Approximation Theorem**，并引用了 [Wikipedia 文章](https://en.wikipedia.org/wiki/Universal_approximation_theorem#Arbitrary-width_case) 中关于 depth-1 UAT 的细节。
   - 有人指出 **Haykin 的研究** 局限于 monotone families，而 **Leshno 等人** 提供了一个涵盖 continuity 的更通用的定义。
- **Uncensored Models 概览**：一位成员推荐了一篇[详细文章](https://erichartford.com/uncensored-models)，解释了创建像 WizardLM 这样的 **uncensored models** 的过程。
   - 提供了各种 WizardLM 模型的链接，包括 [WizardLM-30B](https://huggingface.co/ehartford/WizardLM-30B-Uncensored) 和 [Wizard-Vicuna](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored)。
- **关于模型定义的澄清**：对什么是 **model** 进行了澄清，特别是针对经过 instructed responses 训练的 **HuggingFace transformer models**。
   - 明确了虽然存在许多 transformer 模型，但只有特定的模型是为 interactive chatting 设计的。
- **解释 Uncensored Models**：分享了关于 **uncensored models**（如 Alpaca 和 Vicuna）的全面解释，详细说明了它们的特征和用途。
   - 强调了这些模型在获取不受典型内容限制的响应方面具有价值。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://erichartford.com/uncensored-models">Uncensored Models</a>: 我发布这篇文章是因为很多人问我是如何做到的，所以我将进行解释。 https://huggingface.co/ehartford/WizardLM-30B-Uncensored https://huggingface.co/ehartford/WizardLM-13B-Uncensore...</li><li><a href="https://en.wikipedia.org/wiki/Universal_approximation_theorem#Arbitrary-width_case)">Universal approximation theorem - Wikipedia</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1281937278947823668)** (8 条消息🔥): 

> - `Community Computer Vision Course`
> - `Stanford CS231n Course`
> - `Imgcap CLI Tool`
> - `Face Recognition Datasets`
> - `Data Training Methods with CSV Files` 


- **社区计算机视觉课程发布**：一名成员分享了 [Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome) 的链接，该课程涵盖了计算机视觉的各种基础主题。
   - 该课程旨在让各个水平的学习者都能轻松入门且友好，强调了计算机视觉带来的革命性影响。
- **高度推荐的斯坦福 CS231n 课程**：一名成员建议将 **Stanford CS231n** 课程作为学习计算机视觉的最佳资源。
   - 这一推荐突显了该课程在领域内的声誉和价值。
- **用于图像字幕生成的 Imgcap CLI 工具发布**：发布了一个名为 [Imgcap](https://github.com/ash-01xor/Imgcap) 的新 CLI 工具，用于为本地图像生成字幕。
   - 开发者鼓励用户尝试并对结果提供反馈。
- **寻找人脸识别数据集**：一名成员询问关于按文件夹组织的、中等规模的人脸识别数据集，类似于 [Data Science Stack Exchange](https://datascience.stackexchange.com/questions/63676/face-dataset-organized-by-folder) 上讨论的结构。
   - 他们找到了一个符合要求的数据集，并质疑文件夹结构与命名规范相比的实用性。
- **使用 PNG 和 CSV 数据训练模型**：一名成员询问在 CSV 包含图像 ID 和标签的情况下，是应该使用原始 **PNG 图像** 还是关联的 **CSV 文件** 来训练模型。
   - 他们还想知道使用 CSV 文件是否能加快模型训练，并提到了客户需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://datascience.stackexchange.com/questions/63676/face-dataset-organized-by-folder.">Face dataset organized by folder</a>：我正在寻找一个相当小/中等规模（50MB 到 500MB）的数据集，其中包含按文件夹组织的知名人士照片。目录结构必须类似于...</li><li><a href="https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>：未找到描述</li><li><a href="https://github.com/ash-01xor/Imgcap">GitHub - ash-01xor/Imgcap: A CLI to generate captions for images</a>：一个用于为图像生成字幕的 CLI。通过在 GitHub 上创建账户来为 ash-01xor/Imgcap 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1281845714061627497)** (3 条消息): 

> - `HF Trainer confusion matrix`
> - `RAG-based retrieval evaluation` 


- **在 TensorBoard 中绘制混淆矩阵**：一名用户询问在使用 **HF Trainer** 训练时，如何将 **confusion matrix**（混淆矩阵）作为图像绘制在 TensorBoard 中。
   - 该查询侧重于集成可视化工具，以增强训练过程中的模型评估。
- **评估基于 RAG 的检索框架**：另一名用户提到了为一个涉及特定领域 **RAG-based retrieval** 的项目定义 **评估指标** 的需求。
   - 他们还质疑是应该仅将他们的 RAG 方法与其他 LLM 进行比较，还是应该与有无 RAG 的版本进行对比以评估有效性。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1281845696730894418)** (2 条消息): 

> - `Transformer2DModel`
> - `DiT` 


- **Transformer2DModel 是否等同于 DiT？**：一名成员询问了 **Transformer2DModel** 与 **DiT** 之间的关系。
   - 他们具体询问了这些模型是否等效，或者是否存在关键差异。
- **关于模型对比的讨论**：另一名参与者征求了关于各种模型及其功能的见解，包括 **DiT**。
   - 这引发了关于模型架构及其在领域内应用的更广泛讨论。


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1281691190437216337)** (687 条消息🔥🔥🔥): 

> - `DeepSeek and Aider Performance`
> - `AI Development Concerns`
> - `Aider Workflow Strategies`
> - `Using a Config File for Aider`
> - `Conventions and Prompt Engineering`

- **DeepSeek 最近的基准测试问题**：用户对 DeepSeek Coder 模型的性能表示担忧，认为基准测试可能使用了错误的 model ID，导致仪表盘上的统计数据不佳。
   - 有人指出，两个 model ID 现在都指向同一个 DeepSeek 2.5 模型，这可能会影响性能。
- **AI 开发担忧与反馈**：社区成员讨论了 AI 对开发工作可能产生的影响，以及随着 AI 工具变得更加先进，开发者角色的转变。
   - 讨论中涉及了对 AI 的依赖是否会导致劳动力市场过度饱和或被淘汰。
- **Aider 工作流与用例**：用户分享了使用 Aider 以及集成 CodeCompanion 等工具进行高效项目设置的工作流，强调了清晰规划的重要性。
   - 提到了引入经过强化的、遵循惯例和规划的 system prompt 的想法，建议这可能提高 Aider 输出的一致性。
- **正确配置 Aider 设置**：讨论强调了高效设置环境变量和配置文件以简化 Aider 使用的需求，包括使用 `.aider.conf.yml` 的可能性。
   - 社区成员还提到了使用 `.env` 文件管理 API keys，从而在 Aider 配置和项目特定设置之间实现分离。
- **Google Cloud 配额问题**：用户报告遇到了 Google Cloud 的 Vertex AI 配额问题，特别是新账号在进行预测请求时遇到 429 error，引发了对配额限制的猜测。
   - 用户观察到 Google 服务存在更广泛的问题，因为他们在使用各种 AI 工具时收到了意料之外的 rate limit 错误。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/sure-moron-gif-]">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/ArtificialAnlys/status/1832457791010959539">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: Reflection Llama 3.1 70B 独立评估结果：我们在独立测试中无法复现所声称的评估结果，且观察到的性能比 Meta 的 Llama 3.1 70B 更差，而不是...</li><li><a href="https://direnv.net">direnv – 让你的 .profile 保持整洁</a>: 让你的 .profile 保持整洁</li><li><a href="https://x.com/kimmonismus/status/1831237312887308718">来自 Chubby♨️ (@kimmonismus) 的推文</a>: GPT-5 被拍到疑似参数：3*5T（推测为 MoE）。准确来说，GPT-4 在那里的参数被标为 1.7T。此外，计算资源为 7000 块 B100。官方声明正变得越来越...</li><li><a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>: 让 aider 在处理你的代码时遵循你的编码规范。</li><li><a href="https://x.com/deep9483/status/1832267473204461960?s=46">来自 blueblue (@deep9483) 的推文</a>: @teortaxesTex 我们在 DeepSeek v2.5 的部署中遇到了一些问题，目前已临时修复。能否请您再次测试？</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>: 使用 chat、ask 和 help 聊天模式。</li><li><a href="https://stackoverflow.com/questions/78323246/encountered-429-error-quota-exceeded-for-online-prediction-concurrent-requests">使用 Claude 3 Haiku 时遇到 429 错误 "Quota exceeded for online_prediction_concurrent_requests_per_base_model"</a>: 我正在 Vertex AI 上使用 Claude 3 Haiku，偶尔会遇到以下错误信息：&#xA;&#xA;{&#xA;  &amp;quot;code&amp;quot;: 429,&#xA;  &amp;quot;message&amp;quot;: &amp;quot;Quota exceed...</li><li><a href="https://aider.chat/docs/faq.html#can-i-change-the-system-prompts-that-aider-uses">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md">git-lfs/git-lfs 的 main 分支下的 docs/spec.md</a>: 用于大文件版本控制的 Git 扩展。通过在 GitHub 上创建账号来为 git-lfs/git-lfs 的开发做出贡献。</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://x.com/teortaxestex/status/1832363928283685105?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>: 是的，现在看起来运行得好多了，比之前的模型都要好。我敦促你们重新进行测试。引用 Teortaxes▶️ (@teortaxesTex)：新的 DeepSeek 有一种令人扫兴、愤怒的倾向...</li><li><a href="https://x.com/mattshumer_/status/1832424499054309804">来自 Matt Shumer (@mattshumer_) 的推文</a>: 我们已经找到了问题所在。Hugging Face 上的 reflection 权重实际上是几个不同模型的混合——在上传过程中出了一些差错。今天会修复。引用 Matt Shu...</li><li><a href="https://tenor.com/view/sure-moron-gif-1638860404339486033">Sure Moron GIF - Sure Moron - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=PvQRRGiVsWA">新内容：Replit AI Agents 击败了 Cursor Composor?!? 🤖🤔 端到端编码与部署 AI Coding</a>: 新内容：Replit AI Agents 击败了 Cursor Composor?!? 🤖🤔 端到端编码与部署 AI Coding https://replit.com/ https://cursor.com/ 🤑 免费价值：👉 免费 6-D...</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">paul-gauthier/aider 的 main 分支下的 aider/benchmark/README.md</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/cg-dot/vertexai-cf-workers/issues/18">    &quot;code&quot;: 429,     &quot;message&quot;: &quot;基础模型 anthropic-claude-3-5-sonnet 的 aiplatform.googleapis.com/online_prediction_requests_per_base_model 配额已超出。请提交配额增加请求。https://cloud.google.com/vertex-ai/docs/generative-ai/quotas-genai.&quot;,     &quot;status&quot;: &quot;RESOURCE_EXHAUSTED&quot; · Issue #18 · cg-dot/vertexai-cf-workers</a>: &quot;code&quot;: 429, &quot;message&quot;: &quot;基础模型 anthropic-claude-3-5-sonnet 的 aiplatform.googleapis.com/online_prediction_requests_per_base_model 配额已超出。请提交配额增加请求...
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1281753763207450748)** (193 条消息🔥🔥): 

> - `Aider Chat Functionality` (Aider 聊天功能)
> - `Model Performance Comparisons` (模型性能对比)
> - `Git Integration Features` (Git 集成特性)
> - `Language Output Behavior` (语言输出行为)
> - `Using Aider with Conventions` (结合规范使用 Aider)


- **Aider 的命令执行和初始化延迟**：用户注意到，与不带参数运行 aider 相比，使用特定模型（如 `--model`）运行 aider 可能会引入初始化延迟。
   - 命令执行速度低于预期的实例可能是由于所选模型的复杂性或初始加载过程造成的。
- **调整 Aider 的语言输出**：Aider 在会话期间可能会无意中切换语言，这促使用户需要明确指定所需的输出语言。
   - 使用命令 `/chat-mode ask` 或在 Prompt 中添加“answer in English”有助于保持响应的一致性。
- **管理 Aider 的 Git 集成**：Aider 与 Git 紧密集成，会自动为更改创建提交，但可以使用 `--no-auto-commits` 选项进行自定义。
   - 这允许用户管理 aider 如何与其 Git 仓库交互，包括是否自动创建新分支。
- **在工作流中利用 Aider 进行自动化**：用户可以通过命令行或 Python 编写脚本与 aider 交互，以实现自动化的代码修改和 Pull Request 创建。
   - 虽然将 aider 作为库使用具有潜力，但目前注意到 aider 还没有为此目的提供稳定的 API。
- **使用 Aider 设置项目规范**：为了向 Aider 指示特定的编码指南，用户可以创建一个 `CONVENTIONS.md` 文件并读取它，以确保遵循指南。
   - Aider 对这些规范的遵守可能需要在 Prompt 中进行明确提醒，以保持一致性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat`">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/llms/vertex.html">Vertex AI</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/tips.html#creating-new-files">Tips</a>: 使用 aider 进行 AI 结对编程的技巧。</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: 让 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/benchmarks.html#the-benchmark">GPT code editing benchmarks</a>: 使用基于 Exercism Python 练习的新代码编辑基准测试套件，对 GPT-3.5 和 GPT-4 的代码编辑能力进行基准测试。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/git.html#disabling-git-integration">Git integration</a>: Aider 与 Git 紧密集成。</li><li><a href="https://huggingface.co/sahil2801/reflection_70b_v5">sahil2801/reflection_70b_v5 · Hugging Face</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--chat-language-chat_language">Options reference</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://unrollnow.com/status/1832933747529834747">Thread By @shinboson - A story about fraud in the AI research c..</a>: 关于 AI 研究社区造假的故事。9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fb1h48/psa_matt_shumer_has_not_disclosed_his_investment/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=92YgIVSlfAE">How To Develop 2 AI Apps in 10 Minutes!</a>: 你不必为了尝试构建使用 AI 的应用而付费。使用 Ollama，你可以在本地免费运行 AI 模型。Vercel 的 AI 库让管理变得简单...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude#regions">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650)">Issues · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/commit/6638efbee42d526d238f615ee3f44ee47b61c037">better prompting for LLM to suggest files · paul-gauthier/aider@6638efb</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1281813850940772372)** (14 条消息🔥): 

> - `Reflection 70B vs Llama3 70B`
> - `V0 更新与应用`
> - `Zed 的 GitHub 讨论`
> - `YouTube AI 编程视频` 


- **Reflection 70B 落后于 Llama3 70B**：Reflection 70B 在 aider 代码编辑基准测试中得分为 **42%**，而 **Llama3 70B** 达到了 **49%**。值得注意的是，在修改 aider 以忽略某些标签后，当前模型无法与已发布的 aider 正常配合工作。
   - 欲了解更多见解，请查看 [排行榜](https://aider.chat/docs/leaderboards/)。
- **近期 V0 更新的效果令人印象深刻**：一位成员建议关注 **v0** 的更新，这是 Vercel 为 **NextJS UI** 量身定制的 Claude 版本，据报道效果非常出色。他们还提供了一个展示其功能的 [YouTube 视频](https://youtu.be/zA-eCGFBXjM?si=p-CuTkCzmlwyW2vy)。
   - 演示和更多信息可以在 [v0.dev/chat](https://v0.dev/chat) 及其他链接资源中找到。
- **Zed 的 GitHub 暗示即将推出订阅服务**：讨论显示，**Zed 的 GitHub** 上多次提到了即将推出的 **Zed Pro 订阅**。这次与 **Anthropic** 的合作预计将引入“编辑模式（edit mode）”功能。
   - 成员们推测，这可能会在未来的更新中大幅增强功能。
- **探索 AI 编程的“秘诀”**：新分享的 [YouTube 视频](https://www.youtube.com/watch?v=QlUt06XLbJE) 标题为“AI 编程的秘诀？”，调查了高产出的 AI 编程技术。它重点介绍了包括 Aider、Cursor、Bun 和 Notion 在内的各种工具。
   - 该视频是正在进行的对实用 AI 编程解决方案和方法探索的一部分。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://app.hyperbolic.xyz/models/reflection-70b">Hyperbolic AI Dashboard</a>: 未找到描述</li><li><a href="https://x.com/paulgauthier/status/1832160129720185225">来自 Paul Gauthier (@paulgauthier) 的推文</a>: Reflection 70B 在 aider 代码编辑基准测试中得分为 42%，远低于 Llama3 70B 的 49%。我修改了 aider 以忽略 &lt;thinking/reflection&gt; 标签。该模型无法与已发布的 aider 正常工作...</li><li><a href="https://www.youtube.com/watch?v=QlUt06XLbJE">AI 编程的秘诀？使用 Aider, Cursor, Bun 和 Notion 的 AI 开发日志</a>: 高产出 AI 编程的秘诀是什么？🔗 更多关于 AIDER 的 AI 编程 https://youtu.be/ag-KxYS8Vuw 🚀 更多关于 Cursor 的 AI 编程 https://youtu.be/V9_Rzj...</li><li><a href="https://youtu.be/zA-eCGFBXjM?si=p-CuTkCzmlwyW2vy">使用 v0 构建任何东西（3D 游戏、交互式应用）</a>: 在 https://v0.dev/chat 尝试。• 演示: https://x.com/v0/status/1826020673908535325 • shadcn/ui: https://ui.shadcn.com • 部署: https://vercel.com
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1282439439917781115)** (3 条消息): 

> - `Reflection API`
> - `Reflection-Tuning Technique`
> - `Self-Correcting AI Models` 


- **Reflection API 现已开放公开测试**：[Reflection API](https://openrouter.ai/models/mattshumer/reflection-70b:free) 目前已在 OpenRouter 上提供免费测试，预计很快会发布修复版本。
   - *Matt Shumer* 指出托管 API 与内部 API 之间存在明显的质量差异，表明当前的托管版本尚未完全优化。
- **介绍 Reflection-Tuning 技术**：由 *Matt Shumer* 开发的 **Reflection-70B** 模型采用了一种名为 **Reflection-Tuning** 的新技术，使模型能够检测并纠正其推理中的错误。
   - 该模型利用合成数据进行训练，正如包括 [LinkedIn 帖子](https://www.linkedin.com/posts/mattshumer/im-excited-to-announce-reflection-70b-the-activity-7237801794293174272-kvIm/) 在内的多个来源所指出的，这增强了其性能。
- **关于 Reflection 70B 的社区资源**：用户可以访问有关 Reflection 70B 模型的各种资源，包括一篇讨论其自我纠错能力的 [Medium 文章](https://medium.com/@LakshmiNarayana_U/reflection-70b-enhancing-open-source-ai-with-self-correcting-abilities-7b09896cc80b)。
   - 此外还有一些深入的视频，例如与 *Matt Shumer* 讨论这一创新模型的 [YouTube 讨论](https://www.youtube.com/watch?v=5_m-kN64Exc)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1832880567437729881">来自 OpenRouter (@OpenRouterAI) 的推文</a>: Reflection 自己的 API 现已在 OpenRouter 上提供免费测试：https://openrouter.ai/models/mattshumer/reflection-70b:free 请关注修复版本的生产端点...</li><li><a href="https://community.prod.aws.cyber-boardroom.com/web/docs/platforms/open-router/reflection-70b">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1281859068763574284)** (10 条消息🔥): 

> - `ISO20022`
> - `Bitcoin and CBDCs`
> - `cli_buddy GitHub project`
> - `Open Source Multi-lingual Model`
> - `OpenRouter Usage` 


- **探索加密货币的 ISO20022**：一位成员强调了 **ISO20022** 在加密货币持续发展背景下的重要性，建议其他人研究其影响。
   - 他们鼓励深入研究这一标准，以了解其对金融交易的潜在影响。
- **Bitcoin 与 CBDC 的不兼容性**：**Bitcoin** 无法与 **CBDC** 进行交易，这引发了关于央行数字货币对去中心化加密货币影响的讨论。
   - 成员们对这一限制及其对交易动态的潜在影响表示惊讶。
- **为 OpenRouter 介绍 cli_buddy**：一位成员分享了一个名为 [cli_buddy](https://github.com/rezmeplxrf/cli_buddy) 的 **GitHub** 项目，旨在通过提供多种命令来增强与 OpenRouter 的交互。
   - **info 命令**允许用户搜索 AI 模型并显示 OpenRouter 中的可用额度，提高了易用性。
- **开源多语言模型的开发**：讨论中提到了一个正在开发中的数据集，大小为 **1.5GB**，旨在训练一个开源的 **multi-lingual model**（多语言模型）。
   - 该数据集结合了图像位置数据，使其适合与视觉模型集成。
- **近期 OpenAI 使用的成本效益**：成员们对比了 **1 周使用** OpenAI 额度的成本，大约为 **$2,500**，考虑到讨论的其他项目开支，认为这相当昂贵。
   - 参与者指出，在 AI 服务成本上升的情况下，需要更具性价比的选择。



**提到的链接**: <a href="https://github.com/rezmeplxrf/cli_buddy">GitHub - rezmeplxrf/cli_buddy</a>: 通过在 GitHub 上创建一个账户来为 rezmeplxrf/cli_buddy 的开发做出贡献。

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1281692077121470540)** (611 条消息🔥🔥🔥): 

> - `DeepSeek Coder`
> - `Reflection Model`
> - `OpenRouter API Issues`
> - `Gemini Models`
> - `Multi-Modal Models`

- **DeepSeek Coder 出现问题**：用户报告 DeepSeek Coder 产生零响应且 API 发生故障，表明可能存在上游问题。
   - 尽管 DeepSeek 状态页面显示没有报告的问题，但用户在 API 和 OpenRouter 聊天中仍持续遇到问题。
- **对 Reflection 模型的担忧**：讨论中出现了关于 Reflection 模型合法性的质疑，一些用户对其声明和性能表示怀疑。
   - 由于担心诈骗和虚假信息，有用户希望将该模型从 OpenRouter 中移除。
- **OpenRouter API 调用错误**：用户遇到了诸如 'httpx.RemoteProtocolError' 之类的错误，表明连接被提前关闭，暗示 DeepSeek API 存在问题。
   - 一些用户正试图验证这些错误是源于他们自己的实现还是上游问题。
- **对 AI 模型托管的兴趣**：用户讨论了在 OpenRouter 上托管模型的问题，指出 Euryale 2.2 是 RP 应用的推荐选择，而 Magnum 缺乏更新是一个令人担忧的问题。
   - 对话包括与其他模型的比较，以及对可靠角色扮演选项的需求。
- **多模态模型使用**：用户询问如何将本地图像与多模态模型集成，寻求关于如何正确格式化请求的指导。
   - 提供了将图像解码为 base64 格式以进行 API 请求的说明，以帮助用户利用多模态功能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/cocktailpeanut/status/1832952487541658077">来自 cocktail peanut (@cocktailpeanut) 的推文</a>：OpenAI 准备发布他们的新模型</li><li><a href="https://news.ycombinator.com/item?id=41478241">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/terms">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>：优化 LLM 成本，最高可降低 90%</li><li><a href="https://tenor.com/view/monopoly-guy-money-gif-13385386">大富翁金钱 GIF - 大富翁金钱 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/docs/requests#images-_-multimodal-requests">Requests | OpenRouter</a>：处理传入和传出的请求</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://x.com/mattshumer_/status/1832554497408700466">来自 Matt Shumer (@mattshumer_) 的推文</a>：快速更新 —— 我们重新上传了权重，但仍然存在问题。我们刚刚开始重新训练以消除任何可能的问题。应该很快就会完成。非常抱歉。这个数量的...</li><li><a href="https://www.lumenorbit.com">Lumen Orbit</a>：加入 Lumen Orbit，共同开拓可持续的太空数据中心。了解我们如何减少 90% 的电力消耗并获取 24/7 太阳能。立即下载我们的白皮书！</li><li><a href="https://openrouter.ai/models?q=base>">模型：'base&gt;' | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://x.com/OpenRouterAI/status/1832880567437729881">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Reflection 自己的 API 现在可以在 OpenRouter 上免费进行公开测试：https://openrouter.ai/models/mattshumer/reflection-70b:free 请关注固定版本的生产端点，以便...</li><li><a href="https://x.com/mattshumer_/status/1832424499054309804?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Matt Shumer (@mattshumer_) 的推文</a>：我们已经找到了问题所在。Hugging Face 上的 Reflection 权重实际上是几个不同模型的混合 —— 在上传过程中出了一些差错。今天会修复。引用 Matt Shu...</li><li><a href="https://github.com/googleapis/python-aiplatform/blob/6d1f7fdaadade0f9f6a77c136490fac58d054ca8/google/cloud/aiplatform_v1/types/tool.py#L29">python-aiplatform/google/cloud/aiplatform_v1/types/tool.py at 6d1f7fdaadade0f9f6a77c136490fac58d054ca8 · googleapis/python-aiplatform</a>：Vertex AI 的 Python SDK，一个用于数据科学和机器学习的全托管端到端平台。- googleapis/python-aiplatform</li><li><a href="https://openrouter.ai/models/sao10k/l3.1-euryale-70b">Llama 3.1 Euryale 70B v2.2 - API、提供商、统计数据</a>：Euryale L3.1 70B v2。通过 API 运行 Llama 3.1 Euryale 70B v2.2</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#stable-versions-available">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder">DeepSeek-Coder-V2 - API、提供商、统计数据</a>：DeepSeek-Coder-V2，一个开源的混合专家（MoE）代码语言模型。它是在 DeepSeek-V2 的中间检查点基础上，通过额外的 6 万亿 token 进一步预训练而成的。运行 DeepSeek...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/ZAwetdPza7">Reddit - 深入探索任何事物</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=EbZv6-N8Xlk">什么是 Top K？- 解释 AI 模型参数</a>：今天，我深入探讨 AI 中的 Top K 概念，这是一个影响文本生成的关键参数。通过将 AI 的词汇选择限制在 Top K 个最像...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b>">Llama 3.1 405B (base) - API、提供商、统计数据</a>：Meta 最新级别的模型（Llama 3.1）发布了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x7b>">Mixtral 8x7B (base) - API、提供商、统计数据</a>：由 Mistral AI 开发的预训练生成式稀疏混合专家模型。包含 8 个专家（前馈网络），总计 47B 参数。通过 API 运行 Mixtral 8x7B (base)</li><li><a href="https://search.brave.com/search?q=typhoon+yagi&source=desktop">Brave Search</a>：搜索网络。私密。真正有用的结果、AI 驱动的回答等。全部来自独立的索引。无画像、无偏见、无大科技公司。</li><li><a href="https://openrouter.ai/models/alpindale/magnum-72b">Magnum 72B - API、提供商、统计数据</a>：由 [Goliath](https://openrouter.ai/models/alpindale/goliath-120b) 的制作者开发，Magnum 72B 是新模型系列中的首款，旨在达到 Claude 3 模型的散文质量，特别是...</li><li><a href="https://git">

hub.com/OthersideAI/self-operating-computer/issues/21">这似乎与我们的 Atlas-1 模型非常相似，但带有硬编码的点击。是这样吗？ · Issue #21 · OthersideAI/self-operating-computer</a>: 嘿伙计们，我们一直在训练一个非常相似的多模态模型，名为 Atlas-1，然而我们不需要像这里显示的那样硬编码点击位置，因为我们训练模型去寻找 UI-e...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/pricing#gemini-models:~:text=(%20%3E%20128K%20context%20window)>">未找到标题</a>: 未找到描述</li><li><a href="https://platform.deepseek.com/api-docs/updates/">变更日志 | DeepSeek API 文档</a>: 版本: 2024-09-05</li><li><a href="https://github.com/googleapis/python-aiplatform/commit/72fcc063ed4a086da0ad37ec2ac58860d4e79051">feat: 在分词中增加对系统指令和工具的支持。 · googleapis/python-aiplatform@72fcc06</a>: PiperOrigin-RevId: 669058979</li><li><a href="https://platform.deepseek.com/api-docs/updates">变更日志 | DeepSeek API 文档</a>: 版本: 2024-09-05
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1282154475175542875)** (11 messages🔥): 

> - `Vertex AI 密钥兼容性`
> - `JSON 格式问题`
> - `Google AI Studio 使用`
> - `Base64 编码变通方案` 


- **Vertex AI 密钥需要完整的 JSON**: 一位成员指出，对于 Vertex AI 密钥，它确实需要是整个 **JSON** 对象，包括 **project_id** 和其他详细信息。
   - 在讨论了仅提供 **private_key** 是否足够之后，这一点得到了确认。
- **Google AI Studio 是当前的要求**: 成员们讨论了使用 Vertex AI 的限制，确认截至目前，只能使用 **Google AI Studio**。
   - 这表明需要进一步的修复来扩展兼容性选项。
- **建议将 Base64 编码作为解决方案**: 针对 **JSON** 文件的上传问题，有人提出了一个巧妙的变通方案：将整个 **JSON** 转换为 **Base64**，并在发送到 Vertex AI 之前进行解码。
   - 这种方法被提到是“窃取”自 [GitHub PR 讨论](https://github.com/saoudrizwan/claude-dev/pull/45#issuecomment-2293115878) 的一个想法。



**提到的链接**: <a href="https://github.com/saoudrizwan/claude-dev/pull/45#issuecomment-2293115878)">由 u-minor 添加 Vertex AI 支持 · Pull Request #45 · saoudrizwan/claude-dev</a>: 此 PR 增加了对 Google Cloud 中 Vertex AI 的支持。目前，必须在 gcloud 命令中设置应用默认凭据 (ADC) 才能使用 Vertex AI。身份验证支持以下之一...

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1281690532338208882)** (592 messages🔥🔥🔥): 

> - `AI 模型训练方法`
> - `图像生成的 GPU 推荐`
> - `Stable Diffusion 模型对比`
> - `网红文化与内容创作`
> - `使用细节增强 LoRA` 


- **训练方法对比：LoRA vs Dreambooth**: LoRA 更小、更易于分发，并且可以在运行时组合，而 Dreambooth 输出完整的 checkpoint，占用空间显著更多。
   - 两种方法都只需要极少量的图像进行训练，但对于 LoRA，Kohya 和 OneTrainer 等工具更受青睐，其中 Kohya 特别受欢迎。
- **600 美元以下用于本地图像生成的 GPU 推荐**: 对于 600 美元的预算，建议选择二手 3090 或 2080 作为增强本地图像生成能力的可靠选择。
   - 用户强调了 VRAM 对最佳性能的重要性，特别是在涉及本地训练等任务时。
- **SD 模型的演进及其兼容性**: 有人呼吁开发能够向后兼容 SD1.5 LoRA 的新模型，因为 SD1.5 至今仍是许多用户的经典工具。
   - 目前的讨论强调了 SD1.5 在构图方面的优势，用户注意到新模型并没有削弱它的有效性。
- **内容创作中的网红文化**: 被批评的网红文化强调了对内容创作者通过 Patreon 和 YouTube 等平台将其努力变现的期望。
   - 一些社区成员表达了希望回归较少商业化的内容创作形式的愿望，同时也承认了网红策略的普遍使用。
- **图像生成中的细节增强 LoRA**: 用户报告称，AI 生成图像的细节在很大程度上依赖于工作流的增强而非 prompting，其中 LoRA 对于提高图像质量至关重要。
   - 几位用户利用 LoRA 的组合（如 Detail Tweaker XL）来获得图像生成的最佳效果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://imgur.com/a/vmFARe4">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性 GIF、励志故事、病毒视频等来振奋你的精神……</li><li><a href="https://imgur.com/mXGqSkm">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性 GIF、励志故事、病毒视频等来振奋你的精神……</li><li><a href="https://www.uvmapper.com/">UVMapper - UV Mapping Software</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2409.03755">DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation</a>: 扩散概率模型 (DPMs) 在视觉合成方面表现出卓越的性能，但由于在采样过程中需要多次评估，计算成本很高。最近的预测……</li><li><a href="https://imgur.com/a/xLuCmIA">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性 GIF、励志故事、病毒视频等来振奋你的精神……</li><li><a href="https://huggingface.co/Kijai/flux-fp8">Kijai/flux-fp8 · Hugging Face</a>: 未找到描述</li><li><a href="https://imgur.com/a/izXOC9P">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、娱乐性 GIF、励志故事、病毒视频等来振奋你的精神……</li><li><a href="https://www.youtube.com/@Green-Code">Green Code</a>: 01001000 01101001 00100001 00100000 01001001 00100000 01101101 01100001 01101011 01100101 00100000 01110110 01101001 01100100 01100101 01101111 01110011 00100000 01100001 01100010 01101111 01110101 01...</li><li><a href="https://huggingface.co/spaces/Gradio-Community/Text-guided-Flux-Inpainting">Text Guided Flux Inpainting - a Hugging Face Space by Gradio-Community</a>: 未找到描述</li><li><a href="https://tenor.com/view/they-live-eat-trash-can-coub-glasses-gif-3495333">Wtf Movie Threat GIF - They Live Eat Trash Can Coub - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/@Dice_Ai_Development">🎲 DICE AI DEVELOPMENT 🎲</a>: &quot;嗨，我是 DICE，一位资深的 AI 专业人士，拥有超过 10 年的 AI 编码经验和 20 多年的专业编码经验。作为 Civitai 上的 Master Generator，我……</li><li><a href="https://civitai.com/models">Civitai | Share your models</a>: 未找到描述</li><li><a href="https://www.artstation.com/amirzand">Amir Zand</a>: Artist @ QuanticDream</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ej8sb3/invoke_staff_insisting_that_inpainting_with_flux/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=g74Cq9Ip2ik&t=3113s">Master AI image generation - ComfyUI full tutorial 2024</a>: ComfyUI 完整安装与教程。终极图像生成器。Text to image, image to image, faceswap, ControlNet, upscaling, 外部插件等……</li><li><a href="https://stability.ai/news/introducing-stable-fast-3d?utm_medium=email&_hsenc=p2ANqtz-8eWlqHd4HC0UUG-kEsNVAq5IrP2_6Xm3LOYT9VZTuYDsaoA-1m4F7pdvXJAzs9lbOOF3Epg5DcEdg1gFn0z4vdKAmx3w&_hsmi=94321401&utm_content=94321306&utm_source=hs_email">Introducing Stable Fast 3D: Rapid 3D Asset Generation From Single Images &mdash; Stability AI</a>: 我们很高兴推出 Stable Fast 3D，这是 Stability AI 在 3D 资产生成技术方面的最新突破。这一创新模型将单张输入图像转换为详细的 3D 资产，设定了……</li><li><a href="https://shadermap.com/home/">ShaderMap - Normal Map Generator - Create Rendering and PBR Maps from Textures and 3D Models</a>: 未找到描述</li><li><a href="https://github.com/wl-zhao/DC-Solver">GitHub - wl-zhao/DC-Solver: [ECCV 2024] DC-Solver: Improving Predictor-Corrector Diffusion Sampler via Dynamic Compensation</a>: [ECCV 2024] DC-Solver: 通过动态补偿改进 Predictor-Corrector Diffusion 采样器 - wl-zhao/DC-Solver</li><li><a href="https://www.usenet.com/">Best Usenet Service Providers 2024</a>: 2024 年最佳 Usenet 服务提供商，按新闻组访问、新闻服务器、Usenet 搜索、功能和免费试用排名。添加 VPN 以保护隐私。</li><li><a href="https://youtu.be/cn5BC3Vzcsc?si=qLmqCYPKfTlph5P9">Understanding Normals in Blender</a>: 在本视频中，我将解释在 Blender 中重新计算 Normals 的基础知识。● 帮助支持频道：• Patreon: https://www.patreon.com/ryankingart• Gumro...</li><li><a href="https://civitai.com/models/731347">SECourses 3D Render for FLUX - Full Dataset and W...</a></li>

orkflow Shared - v1.0 | Stable Diffusion LoRA | Civitai</a>: 完整的训练教程、指南和研究，针对 FLUX 风格的 Hugging Face 仓库，包含完整的工作流、研究细节、过程、结论...</li><li><a href="https://github.com/leejet/stable-diffusion.cpp">GitHub - leejet/stable-diffusion.cpp: 纯 C/C++ 实现的 Stable Diffusion 和 Flux</a>: 纯 C/C++ 实现的 Stable Diffusion 和 Flux。通过在 GitHub 上创建账号来为 leejet/stable-diffusion.cpp 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=9DbRJDitVhA">音频反应式生成涂鸦 - [TouchDesigner + Stable Diffusion]</a>: 您可以通过以下链接访问这个新补丁以及更多系统、实验和教程：https://linktr.ee/uisato #touchdesigner #stablediffusion #visuals</li><li><a href="https://rentry.org/voldyold">--最终 GUI 傻瓜指南--</a>: "不可直呼其名者" 权威的 Stable Diffusion 体验 ™ ---新功能展示与操作指南--- 显著功能：Inpainting/Outpainting（局部重绘/扩图）、实时生成预览、Tiling（平铺）、Upscaling（放大）等...</li><li><a href="https://github.com/tensorflow/tensorflow/">GitHub - tensorflow/tensorflow: 为每个人准备的开源机器学习框架</a>: 为每个人准备的开源机器学习框架 - tensorflow/tensorflow</li><li><a href="https://pytorch.org/tutorials/advanced/cpp_frontend.html">使用 PyTorch C++ 前端 — PyTorch 教程 2.4.0+cu121 文档</a>: 未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1bdidA4w5pB2BQMyxhkFu710Nu5bzKM1E-Wh_38ZZlT0/edit?usp=sharing">AI 艺术提示词</a>: 未找到描述</li><li><a href="https://civitai.com/models/622686/underwatermovielora">Underwater_movie_lora - underwater_movie_loraV1 | Stable Diffusion LoRA | Civitai</a>: 为 2020 年电影《深海异兽》(Underwater) 训练的 LoRA。</li><li><a href="https://civitai.com/models/618692/flux">FLUX - Dev | Stable Diffusion Checkpoint | Civitai</a>: FLUX.1 [dev] 是一个拥有 120 亿参数的 rectified flow transformer，能够根据文本描述生成图像。更多信息请访问...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1281690684981510207)** (402 条消息🔥🔥): 

> - `LM Studio 更新`
> - `模型性能与设置`
> - `训练语言模型`
> - `LM Studio 用户体验`
> - `服务器交互与 API 请求` 


- **关于 LM Studio v0.3 的反馈**：用户对 LM Studio v0.3 的新界面表示担忧，指出与 v0.2 相比，某些功能和设置被移除。开发者保证在未来的版本中会有许多更新和改进。
   - 反馈包括对系统提示词（system prompts）丢失的抱怨，以及调整设置的困难，导致用户考虑降级版本。
- **模型配置问题**：用户报告了模型配置问题，特别是与 GPU offloading（GPU 卸载）和 context length（上下文长度）设置相关的问题。建议包括调整 GPU 层数并确保专用 VRAM 以提高性能。
   - 一位用户在尝试继续助手消息时因上下文溢出而遇到错误，引发了关于潜在 Bug 报告的讨论。
- **训练语言模型**：用户讨论了训练小语言模型的可行性，对数据集和参数量表现出兴趣。重点在于理解训练损失（training loss）及其与模型性能的关系。
   - 强调了为冷门语言训练较小模型的挑战，以及高质量数据集的重要性。
- **与 LM Studio 服务器交互**：提出了关于如何与 LM Studio 服务器交互的问题，并明确了应发送 API 请求而非使用 Web 界面。引导用户参考服务器选项卡上的示例以获取进一步帮助。
   - 一位用户在理解了所需的 API 请求格式后，迅速解决了他们的服务器交互问题。
- **用户体验与建议**：用户分享了使用 LM Studio 的各种经验，讨论了近期更新的积极方面和令人沮丧之处。改进建议包括提供清晰的文档和访问功能的替代方案。
   - 还强调了对新界面进行更好教程和指导的需求，表明希望提高用户对 LM Studio 的使用能力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://i.imgur.com/">Imgur: The magic of the Internet</a>: 未找到描述</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/audio/">Audio Examples</a>: ComfyUI 工作流示例</li><li><a href="https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf">abetlen/Phi-3.5-vision-instruct-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/AGI-0/Artificium-llama3.1-8B-001">AGI-0/Artificium-llama3.1-8B-001 · Hugging Face</a>: 未找到描述</li><li><a href="https://smcleod.net/2024/07/understanding-ai/llm-quantisation-through-interactive-visualisations/">Understanding AI/LLM Quantisation Through Interactive Visualisations</a>: AI/LLM 量化可视化</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.0">LM Studio 0.3.0 | LM Studio</a>: 我们非常激动地终于分享了 LM Studio 0.3.0 🥳。</li><li><a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B/discussions/38">mattshumer/Reflection-Llama-3.1-70B · I created the Llama-3.1-8B Version</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fcietm/lm_studio_alternatives/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.hwinfo.com/download/">Free Download HWiNFO Sofware | Installer &amp; Portable for Windows, DOS</a>: 现在就开始分析你的硬件！HWiNFO 提供适用于 Windows（32/64 位）的安装版和便携版，以及适用于 DOS 的便携版。</li><li><a href="https://huggingface.co/bartowski/Reflection-Llama-3.1-70B-GGUF/tree/main">bartowski/Reflection-Llama-3.1-70B-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>: LM Studio CLI。通过在 GitHub 上创建账户为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/Vasco0x4/Neo-AI">GitHub - Vasco0x4/Neo-AI: Neo AI integrates into the Linux terminal, capable of executing system commands and providing helpful information.</a>: Neo AI 集成到 Linux 终端中，能够执行系统命令并提供有用的信息。 - GitHub - Vasco0x4/Neo-AI: Neo AI 集成到 Linux 终端中，能够...</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct">microsoft/Phi-3.5-vision-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://community.make.com/t/what-is-the-difference-between-system-user-and-assistant-roles-in-chatgpt/36160/3">What is the difference between System, User, and Assistant roles in ChatGPT?</a>: 根据 Mastering the OpenAI API: Tips and Tricks - Arize AI：常用的角色包括 “system”、“user” 和 “assistant”。“system” 提供高层指令，“user” 提出 ...</li><li><a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Dee">mostafaibrahim17</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0#what-are-accuracy-and-loss-curves?-">A Deep Dive Into Learning Curves in Machine Learning</a>: 通过我们的准确率和损失曲线指南更好地理解机器学习。我们解释了它们的区别、如何阅读它们以及它们为什么重要。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9119">Feature Request: Add support for Phi-3.5 MoE and Vision Instruct · Issue #9119 · ggerganov/llama.cpp</a>: 前提条件 我正在运行最新的代码。如果可能，请提及版本。我仔细阅读了 README.md。我使用与我的问题相关的关键词进行了搜索，以确保我正在创建...</li><li><a href="https://github.com/ollama/ollama/pull/3657#issuecomment-2131036569">Add support for IQ1_S, IQ3_S, IQ2_S, IQ4_XS. IQ4_NL is not functional by mann1x · Pull Request #3657 · ollama/ollama</a>: 此补丁添加了对 IQ1_S, IQ3_S, IQ2_S, IQ4_XS 的支持。IQ4_NL 使用不同的格式，必须进一步调查其区别。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1282138965339799653)** (83 messages🔥🔥): 

> - `LM Studio and VOSK`
> - `Intel A770 Performance`
> - `NVIDIA Caution with VRAM`
> - `Reflection-Llama-3.1 Issues`
> - `Apple's Upcoming Hardware` 


- **LM Studio 集成 VOSK 用于语言提示词**：在配置 LM Studio 接收来自 Vector 的提示词并通过 Intel A770 上的 VOSK 进行响应后，**性能提升**显著，响应时间被描述为“几乎是瞬时的”。
   - 仍需进行微调，建议将响应长度限制在 **100-200 个单词**左右以保持简洁。
- **Intel A770 和 SYCL 性能讨论**：围绕 **Intel A770** 的讨论强调了其使用 **Vulkan** 和 **fp16** 计算进行推理的能力，成员们询问了 Token 吞吐量，平均约为 **7000 TPS**。
   - 对话还涉及利用 **Q8 量化**，据报道这可以在不牺牲模型智能的情况下增强性能。
- **对 NVIDIA VRAM 限制的担忧**：用户对 NVIDIA 缺乏显著的 **VRAM 增长**表示失望，称尽管有预期，但近几代产品中预期的 VRAM 容量并未实现。
   - 讨论指出制造商正将重点从消费级显卡转向利润更高的企业级解决方案。
- **加载 Reflection-Llama-3.1 模型的问题**：一位用户报告加载 **Reflection-Llama-3.1-70B-Q4_0_4_4.gguf** 模型失败，尽管配置了充足的 VRAM，仍面临 **CUDA 内存分配错误**。
   - 建议他们考虑使用 [Hugging Face](https://huggingface.co/mattshumer/ref_70_e3) 上提供的修正版模型来解决加载问题。
- **对 Apple 硬件发布的期待**：用户对 **Apple** 即将发布的公告表示关注，并推测 **5090 GPU** 的能力及其相对于先前型号的显存配置。
   - 预计 Apple 将凭借其新硬件产品继续主导统一内存（Unified Memory）市场。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mattshumer/ref_70_e3">mattshumer/ref_70_e3 · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/U9_o6X3k6A8">Vector</a>: 使用 Arc A770 本地生成的 Phi 3 LLM 的 Vector。
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1281702162077978724)** (334 messages🔥🔥): 

> - `Perplexity Subscription Issues`
> - `Promo Code Leak Controversy`
> - `Model Usage Limits`
> - `Web Scraping by LLMs`
> - `Technical Issues with Perplexity` 


- **使用优惠码的订阅被取消**：许多用户对使用泄露的优惠码后订阅被取消表示沮丧，一些人收到的邮件声称是他们自己取消了订阅。
   - 用户正在寻求 Perplexity 支持团队的澄清，但报告称几乎没有得到回应。
- **对模型限制和访问的担忧**：用户对模型使用限制感到困惑，讨论指出 Pro 模型的限制为 450 次查询，Claude Opus 为 50 次。
   - 一些用户质疑在写作时如何指定正在使用的模型，因为当前功能似乎模糊了这一点。
- **Perplexity LLM 功能的替代方案**：出现了一场关于其他搜索引擎和 LLM（如 You.com 和 Kagi）的讨论，这些引擎利用网页抓取在响应中提供数据。
   - 这些替代方案被强调为解决了与知识截止日期和幻觉响应相关的一些问题。
- **Perplexity 的技术困难**：用户报告了各种技术问题，包括访问“Pages”困难以及对查询的响应不足。
   - 许多人在不同的浏览器和设备上都遇到了这些问题，表明该平台可能存在广泛的问题。
- **即将推出的功能和更新**：出现了关于添加新功能（如 Reflection LLM）以及模型托管规格（如 FP16 或 FP8）细节的问题。
   - 用户正在积极寻求产品增强的更新以及对 Perplexity 当前产品的澄清。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1832201335175049434">来自 lmsys.org (@lmsysorg) 的推文</a>: ⚠️警告：内含冒犯性内容。介绍 RedTeam Arena with Bad Words——我们的第一款游戏。你有 60 秒的时间让模型说出脏话。越快越好。（合作 ...</li><li><a href="https://prollm.toqan.ai/leaderboard/stack-unseen">ProLLM Benchmarks | Toqan</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1281764827617955881)** (49 条消息🔥): 

> - `One Piece Documentation` (One Piece 文档)
> - `AI Services` (AI 服务)
> - `Carbon Capture Technologies` (碳捕集技术)
> - `Kung Pao Chicken Recipe` (宫保鸡丁食谱)
> - `AI Tutors Engagement` (AI 导师参与度)


- **深入研究 One Piece 文档**：开始为 One Piece 编写一份全面的[文档](https://www.perplexity.ai/page/one-piece-journey-documentatio-IyKqoJFITa.gpTjfS0EoSg)，重点是添加所有篇章。
   - 该项目反映了为 One Piece 粉丝组织和增强可访问性的承诺。
- **讨论热门 AI 服务**：成员们分享了对现有[热门 AI 服务](https://www.perplexity.ai/search/what-are-the-top-ai-services-a-VOrGhhfiQMyXU76b24WReA)及其对参与度影响的兴趣。
   - 讨论强调了 AI 如何为各个领域做出贡献，推动创新和效率。
- **探索碳捕集技术**：成员们讨论了[碳捕集与封存](https://www.perplexity.ai/search/carbon-capture-and-storage-aMmizjq3RUmByAR1r1yUQg)的新方法，强调了其在气候行动中的重要性。
   - 这次对话强调了为减轻环境影响而在技术上取得的进步。
- **掌握宫保鸡丁**：分享了一份美味的[宫保鸡丁](https://www.perplexity.ai/search/how-to-make-kung-pao-chicken-hnAOfrAISaSHh9WXX4tfQA)食谱，承诺带来充满风味的烹饪体验。
   - 成员们交流了改进菜肴的技巧和变体，培养了一个烹饪社区。
- **AI 导师提升学生参与度**：一份演示文稿展示了 [AI 导师](https://www.youtube.com/embed/IJCFJzEbfYE) 如何在学习环境中有效地将学生参与度提高一倍。
   - 这项技术的意义表明了教育方法论和学生互动方式的转变。



**提到的链接**: <a href="https://www.youtube.com/embed/IJCFJzEbfYE">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1281756715616047218)** (13 条消息🔥): 

> - `API response length` (API 响应长度)
> - `API access issues` (API 访问问题)
> - `Payment method errors` (支付方式错误)
> - `Model deprecation` (模型弃用)
> - `Search domain filter` (搜索域名过滤器)


- **API 响应需要更多深度**：一位用户注意到，尽管查询内容相同，但与 Web 端响应相比，API 响应显得**简短且枯燥**，并寻求有关调整参数的建议。
   - 改进建议可以增强 API 回复的丰富度。
- **API URL 出现 404 错误**：一位用户在尝试通过指定 URL 访问 API 时遇到了 **HTTP ERROR 404**。
   - 另一位用户指出了正确的端点为 [https://api.perplexity.ai/chat/completions](https://api.perplexity.ai/chat/completions)。
- **支付方式身份验证问题**：一位用户报告了在设置 API 访问时支付方式身份验证的问题，多张卡均收到错误。
   - 另一位参与者确认了类似的经历，特别是安全码错误。
- **对模型弃用的担忧**：一位用户对许多模型被弃用表示沮丧，这影响了对更新信息和链接的访问。
   - 他们询问了如何提示模型以获得更直接的链接访问方法。
- **使用搜索域名过滤器**：一位用户建议利用 API 中的 `search_domain_filter` 参数来规范模型搜索的域名。
   - 这种方法可能有助于用户从当前模型中检索更准确的信息。



**提到的链接**: <a href="https://docs.perplexity.ai/api-reference/chat-completions>">未找到标题</a>: 未找到描述

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1281793672391819378)** (334 messages🔥🔥): 

> - `Cohere tech`
> - `发型与风格`
> - `Bot 在内容审核中的作用`
> - `AI 诈骗与加密货币`
> - `多模态模型与项目` 


- **Cohere 技术在审核中表现出色**：成员们讨论了 Cohere 的分类技术如何有效地消除加密货币垃圾信息，从而改善了服务器的对话环境。
   - 一位用户强调，在遇到猖獗的垃圾信息后，该 Bot 是保持讨论集中且愉快的必要工具。
- **聊天中流行发型话题**：参与者们进行了一场关于发型的轻松对话，特别是提到了 Aidan Gomez 的发型，并分享了各自的经历。
   - 几位成员考虑剪类似的发型，在分享发型轶事的同时，突显了有趣的社区氛围。
- **加密货币对 AI 的影响**：成员们对加密货币诈骗者渗透到 AI 领域表示担忧，并对相关的诈骗行为感到沮丧。
   - 一位资深的 AI 爱好者分享了处理此类垃圾信息的经验，并提到这对比合法的 AI 进步的认知产生了负面影响。
- **探索 Cohere 产品**：新成员表达了对探索 Cohere 产品以及深入了解该平台能力的兴奋。
   - 讨论强调了 R 和 R+ 的最新更新，这些更新改善了用户的编程体验。
- **多模态模型与项目**：讨论涉及了视觉模型在规划任务中的潜力，社区成员分享了他们在机器人技术和 AI 方面的见解。
   - 对话反映了不同的 AI 模型如何为更现实的问题解决方法做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com/@a.sale/chatgpt-5-and-beyond-openais-five-level-roadmap-to-agi-unveiled-be09db42ca27">ChatGPT 5 and Beyond: OpenAI’s Five-Level Roadmap to AGI Unveiled</a>: 在最近的一项进展中，OpenAI 揭晓了一个新的五级系统，用于跟踪其在实现通用人工智能 (AGI) 方面的进展……</li><li><a href="https://www.reddit.com/r/SelfBarber/comments/155u0tk/attempt_2_at_low_fade/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=FUGosOgiTeI&ab_channel=20VCwithHarryStebbings">Aidan Gomez: What No One Understands About Foundation Models | E1191</a>: Aidan Gomez 是 Cohere 的联合创始人兼 CEO，Cohere 是领先的企业级 AI 平台，已从顶级投资者处筹集了超过 10 亿美元，其最新一轮融资……</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1282142131183681547)** (25 messages🔥): 

> - `招聘团队联系方式`
> - `Cohere 产品的使用`
> - `MrDragonFox 的存在感`
> - `Embed vs Embed Jobs` 


- **招聘团队联系方式咨询**：一位成员在 LinkedIn 上发现了一个远程兼职职位并被引导至 Discord 服务器后，寻求招聘团队的联系方式。
   - 另一位成员建议，一旦团队回复，他们就会获得联系方式，并指出该服务器旨在进行技术讨论，而非招聘。
- **探索 Cohere 产品**：针对人们使用 Cohere 产品做什么的问题，有人指出客户用例会定期发布在 [Cohere 博客](https://cohere.com/blog)上。
   - Discord 成员也会在专门的频道中分享他们的用例，而 [cookbooks](https://docs.cohere.com/page/cookbooks) 为各种应用提供了具有启发性的入门代码。
- **MrDragonFox 无处不在的存在**：成员们开玩笑说 MrDragonFox 在服务器中无处不在，一位成员幽默地质疑他是否真的是人类。
   - MrDragonFox 肯定地回答自己是人类，并幽默地补充说他只是“一直在线”。
- **Embed 与 Embed Jobs 的区别**：一位成员请求澄清术语 “Embed” 和 “Embed Jobs” 之间的区别，并表示他们已经理解了 Embed 的过程。
   - 讨论旨在简化这两个概念之间的技术区别。



**提到的链接**: <a href="https://docs.cohere.com/page/cookbooks">Cookbooks — Cohere</a>: 未找到描述

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1282139055127134279)** (20 条消息🔥): 

> - `Configuring Output Lengths` (配置输出长度)
> - `Search Query Costs` (搜索查询成本)
> - `Using Calendar Agent` (使用 Calendar Agent)
> - `Invalid Raw Prompt Error` (无效 Raw Prompt 错误)
> - `Chat Turns in API` (API 中的 Chat Turns)


- **配置输出长度讨论**：成员们讨论了如何配置输出长度和提前停止序列（early stop sequences），表明需要更清晰的说明。
   - *一位参与者提到他们会向 Alicja 寻求进一步帮助，因为她目前正在休间隔年（gap year）。*
- **理解搜索查询成本**：一位成员询问包含 10 个文档的查询是否计为 0.1 次搜索，对此得到的澄清是：100 个以内的任何数量都计为单次搜索。
   - *不存在分数执行；无论你搜索 1 个还是 99 个文档，它仍被视为一次搜索查询。*
- **使用 Calendar Agent**：出现了关于 Calendar Agent 使用方法以及如何通过正确的 API 调用预订会议的问题。
   - *用户被引导至特定文档，但仍难以获得示例中演示的预期输出。*
- **处理无效 Raw Prompt 错误**：一位成员报告在使用 `raw_prompting` 参数时出现 400 Bad Request 错误，并询问关于“有效 Chat Turns”的澄清。
   - *澄清指出，一个 Chat Turn 被定义为一次 user、system 或 agent 的交互。*



**提及的链接**：<a href="https://docs.cohere.com/page/calendar-agent">Calendar Agent with Native Multi Step Tool — Cohere</a>：该页面描述了如何使用 Cohere Chat API 配合 list_calendar_events 和 create_calendar_event 工具来预订会议。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1281806144565678101)** (13 条消息🔥): 

> - `LLM Web App Launch` (LLM Web App 发布)
> - `Streamlit Hosting Plans` (Streamlit 托管计划)
> - `Langchain Integration` (Langchain 集成)
> - `Admin Access Concern` (管理员访问权限担忧)


- **Wittgenstein 发布了一个简单的 LLM Web App**：一位成员宣布编写了一个简单的 LLM Web App，并分享了 [GitHub 链接](https://github.com/xettrisomeman/llm_simple_app) 供他人探索。
   - 他们表达了热情并欢迎提问，断言 **Cohere** 是一个很棒的工具。
- **计划在 Streamlit 上托管应用**：成员们讨论了在 **Streamlit** 上托管该 LLM 应用以方便访问的可能性，开发者对此表示同意。
- **Langchain 的集成**：开发者确认该应用是作为一个涉及 **Langchain** 的学习项目构建的，从而增强了其功能。
- **应用已部署在云端**：Wittgenstein 分享了该应用现已部署在云端，并提供了访问链接：[Streamlit App](https://llmsimpleapp-mrzdrd8jxzcxmy5yisnmis.streamlit.app/)。
   - 他们对开发过程中获得的动力表示感谢。
- **发现管理员访问权限问题**：当发现该应用允许通过 JSON 输出轻松进行管理员登录并泄露管理员密码时，引发了担忧。
   - 成员们对密码为 'admin' 幽默地做出了反应，并指出了潜在的安全风险。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/xettrisomeman/llm_simple_app">GitHub - xettrisomeman/llm_simple_app: Simple LLM APP</a>：简单的 LLM APP。通过在 GitHub 上创建账号来为 xettrisomeman/llm_simple_app 的开发做出贡献。</li><li><a href="https://llmsimpleapp-mrzdrd8jxzcxmy5yisnmis.streamlit.app/">无标题</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1281697372845375669)** (199 条消息🔥🔥): 

> - `Reflection 70B Performance` (Reflection 70B 性能)
> - `Upcoming AI Models` (即将推出的 AI 模型)
> - `Nous Forge Presentation` (Nous Forge 演示)
> - `Benchmark Evaluations` (基准测试评估)
> - `AI Model Mislabeling` (AI 模型标签错误)

- **Reflection 70B 表现不佳的基准测试**：最近的评估显示，**Reflection 70B** 在各项基准测试中始终不如 **Llama 3.1**，这表明其能力可能存在过度承诺。
   - 独立测试显示其得分较低，引发了对其最初声明的怀疑，并对某些 weights 尚未发布的原因提出了疑问。
- **社区对 AI 声明的质疑**：社区成员对新 AI 模型的**性能声明**表示怀疑，认为这种情况可能具有误导性，或者是营销噱头。
   - 一些讨论认为，持续发布的内容可能无法反映模型的真实能力，类似于 AI 早期发展中的炒作周期。
- **Nous Forge 可能在 38C3 亮相**：目前正在考虑在即将举行的 **2024年混沌通信大会 (38C3)** 上进行 **Nous Forge 演示**，成员们正在讨论该活动的关联性。
   - 虽然该活动可能主要面向德语观众，但其双语形式仍允许就数字自由和 AI 进行全面的演示。
- **多样化基准测试的重要性**：参与者一致认为，有必要利用**多样化的基准测试**来衡量 AI 模型，并指出对特定数据集 overfitting 的风险。
   - 例如 **Alice** 基准测试表明，特定的弱点可能无法准确代表模型的整体性能，并可能导致评估结果偏差。
- **对更干净的预训练数据的需求**：大家达成共识，认为在某些 AI 模型中观察到的问题是**预训练数据洁净度**的征兆，而非 Transformer 架构的系统性缺陷。
   - 建议包括使用**合成数据**来改进模型训练，并减轻数据集中发现的偏差或误导性模式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://vxtwitter.com/shinboson/status/1832933747529834747">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/cocktailpeanut/status/1832952487541658077">来自 cocktail peanut (@cocktailpeanut) 的推文</a>：OpenAI 正准备发布他们的新模型</li><li><a href="https://x.com/paulgauthier/status/1832160129720185225">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Reflection 70B 在 aider 代码编辑基准测试中得分为 42%，远低于 Llama3 70B 的 49%。我修改了 aider 以忽略 &lt;thinking/reflection&gt; 标签。该模型无法正常配合 t...</li><li><a href="https://vxtwitter.com/RealJosephus/status/1832904398831280448">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/OpenRouterAI/status/1832880567437729881">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Reflection 自己的 API 现在可以在 OpenRouter 上免费进行公开测试：https://openrouter.ai/models/mattshumer/reflection-70b:free 请关注修复版本的生产端点，以便...</li><li><a href="https://x.com/paulgauthier/status/1832203435896402151">来自 Paul Gauthier (@paulgauthier) 的推文</a>：澄清一下，42% 的得分是在没有使用特定推荐 system prompt 的情况下得出的。使用该 prompt 后，得分为 43%。</li><li><a href="https://x.com/N8Programs/status/1832290974023795093">来自 N8 Programs (@N8Programs) 的推文</a>：神经网络 MNIST 训练的 CPU 单线程实现。运行速度接近每秒 20000 张图像。纯 Javascript 编写，带有用于 SIMD 的 WASM 扩展。专为在 NodeJS 中运行而设计。https://github.com/...</li><li><a href="https://x.com/ArtificialAnlys/status/1832965630472995220?t=9UGPOogWfNAVx7vc-l3lVw&s=19">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Reflection 70B 更新：关于时间线和我们视角下未解决问题的简要说明。时间线：- 我们测试了初始发布的 Reflection 70B，发现其性能低于 Llama 3.1 70B。- ...</li><li><a href="https://x.com/terryyuezhuo/status/1832901679387394341?t=Hkfx2OAd-qAtXTibXAyOgA&s=19">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：应要求，这里是更新后的 Reflection 模型的新结果。(1) 无 thinking + 无 system prompt：Complete 33.1（> Llama-3.1-405B 的 30.4，等于几个接近的 LLM）Instruct 23.0（仍然 ~&lt; ...</li><li><a href="https://huggingface.co/mattshumer/ref_70_e3">mattshumer/ref_70_e3 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/%D0%BA%D0%BE%D0%BC%D0%B0%D1%80%D1%83-%D0%BA%D0%BE%D0%BC%D0%B0%D1%80%D1%83-%D0%BA%D0%BE%D1%82-komaru-komaru-cat-%D1%87%D0%B0%D1%82-gif-11530076981254865092">комару комару кот GIF - Комару Комару кот Komaru - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/Besteuler/status/1833163141066145819">来自 Weiyang Liu (@Besteuler) 的推文</a>：🧐 有趣的发现：我们在 SGP-Bench（https://sgp-bench.github.io/，一个评估符号程序理解的基准测试）上测试了 Reflection-70B。尽管 Reflection-70B 声称优于许多 ...</li><li><a href="https://x.com/abacaj/status/1832816808690114642">来自 anton (@abacaj) 的推文</a>：@ArtificialAnlys @mattshumer_ 仍在等待正确的 weights 以便在本地尝试该模型，不确定为什么它必须保留在 API 之后（这样很难判断提供的是什么服务）</li><li><a href="https://x.com/WenhuChen/status/1832621826523934944">来自 Wenhu Chen (@WenhuChen) 的推文</a>：我们更新了 MMLU-Pro 排行榜，加入了一些最近的模型，如 Reflection、GPT-4o (0806) 和 Arx-0.3（由 Thomas Baker 创立的初创公司）。</li><li><a href="https://x.com/JJitsev/status/1832758733866222011">来自 Jenia Jitsev (@JJitsev) 的推文</a>：（又一个）崛起与陨落的故事：Reflection-70B 发布并声称拥有强大的前沿 LLM 性能——依赖于 MMLU 等常用基准测试。它能处理揭示泛化能力的 AIW 问题吗...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fb6jdy/reflectionllama3170b_is_actually_llama3/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f5ii16/where_did_arx03_come_from_and_w">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://agi-v2.webflow.io/arx">ARX</a>：来自 Applied General Intelligence (AGI) 的 ARX</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f5ii16/where_did_arx03_come_from_and_who_makes_it/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/codelion/optillm/blob/main/plansearch.py">codelion/optillm 中的 optillm/plansearch.py</a>：LLM 的优化推理代理。通过在 GitHub 上创建账号为 codelion/optillm 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fbclkk/reflection_llama_31_70b_independent_eval_result">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: 一个...</a></li>

<li><a href="https://github.com/cpldcpu/MisguidedAttention">挑战大语言模型在存在误导信息时的推理能力的提示词集合</a>：一个旨在挑战大语言模型在面对误导性信息时推理能力的提示词集合 - cpldcpu/MisguidedAttention</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fbclkk/reflection_llama_31_70b_independent_eval_results/">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://sci-hub.scrongyao.com/10.1017/S0140525X98001733">Sci-Hub | 认知科学中的动力学假设 | 10.1017/S0140525X98001733</a>：未找到描述</li><li><a href="https://european-pirateparty.eu/">欧洲海盗党</a>：未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fc98fu/confirmed_reflection_70bs_official_api_is_sonnet/">已确认：REFLECTION 70B 的官方 API 是 SONNET 3.5</a>：由 u/TGSCrust 发布在 r/LocalLLaMA • 1,043 点赞和 303 条评论
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1281695101952987168)** (7 messages): 

> - `DeepSeek v2.5 性能`
> - `用于书籍和电影查询的 LLM`
> - `用于 One-Shot 识别的 FaceNet`
> - `Hermes Nemo 发布日期`
> - `对 Anything LLM 的兴趣` 


- **测试 DeepSeek v2.5 性能**：一位成员询问其他正在使用 **DeepSeek v2.5** 的用户，是否有发现其相比前一版本有明显的改进。
   - *关于性能的反馈将有助于评估新版本中引入的增强功能。*
- **寻找用于电影和书籍问题的 LLM**：一位用户询问是否有能够回答关于电影或书籍问题的 **LLM 服务**，例如哈利·波特在第一章中的年龄。
   - *期望是 LLM 能够提供正确答案或承认其局限性。*
- **FaceNet 在 One-Shot 识别中的可行性**：一位成员对 **FaceNet** 的功能感到好奇，询问是否有人测试过它在 One-Shot 人脸识别中的表现。
   - *该询问表明了对探索人脸识别技术在特定场景下有效性的兴趣。*
- **对 Hermes Nemo 的期待**：一位成员对 **Hermes Nemo** 的发布日期表示好奇。
   - *即将推出的模型生成似乎引起了该小组的兴趣。*
- **对 Anything LLM 的普遍兴趣**：几位成员对 **Anything LLM** 相关话题的更广泛范围表示了兴趣。
   - *这表明了对 LLM 社区发展和讨论的持续好奇。*


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1282055619054010410)** (2 messages): 

> - `Medical LLMs`
> - `Continual In-Context Learning`
> - `Frameworks for Medical AI`
> - `LLM Digital Twins` 


- **医疗 LLMs 的新进展**：本周重点介绍了多种 **Medical LLMs**，包括服务于癌症领域的 **CancerLLM**，以及用于医学影像的视觉语言模型 **MedUnA**。
   - **机器人内窥镜手术基础模型 (Foundation Model for Robotic Endoscopic Surgery)** 和 **DHIN**（**去中心化健康智能网络**，Decentralized Health Intelligence Network）等关键进展，指向了医疗保健领域的创新应用。
- **医疗 AI 基准测试评估**：出现了多项评估，例如提供临床试验数据集和基准的 **TrialBench**，以及探索医疗 LLMs 鲁棒性的 **MedFuzz**。
   - 通过 **DiversityMedQA** 等倡议关注诊断中 **LLM 偏见** 的评估，体现了医疗 AI 公平性的积极应对。
- **医疗应用中的数字孪生**：**Digital Twins** 是一个核心话题，相关工作包括为罕见妇科肿瘤创建模型，以及使用 **DT-GPT** 预测患者健康状况。
   - 该技术强调了通过预测分析改进针对特定患者的医疗干预的潜力。
- **鲁棒医疗 AI 框架**：**Rx Strategist** 等创新实现了基于 LLM 的处方验证，增强了医疗 AI 工具的可靠性。
   - 此外，**医疗 LLMs 护栏 (guardrails)** 的发展表明，人们对医疗保健领域 AI 应用的安全性和可靠性日益关注。
- **持续上下文学习的进展**：**具有自适应 Transformers 的持续上下文学习 (Continual In-Context Learning with Adaptive Transformers)** 架构扩展了 Transformer 模型以适应动态学习场景，重点关注有效的梯度流。
   - 该系统支持快速适应新任务，从而在保持学习完整性的同时降低灾难性失效的风险。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1832476252260712788">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024年9月1日 - 9月7日） 医疗 LLM 及其他模型：- CancerLLM：癌症领域的大语言模型 - MedUnA：视觉语言...

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1281731955292897384)** (19 messages🔥): 

> - `PlanSearch 引入多样化的 LLM 输出`
> - `RedTeam Arena 以游戏化形式发布`
> - `Reflection 70b 模型能力`
> - `关于 AI 研究欺诈的见解`
> - `Itext2kg 作为知识图谱工具` 


- **PlanSearch 引入多样化的 LLM 输出**：Scale SEAL 发布了一种名为 **PlanSearch** 的新方法，通过自然语言搜索方法鼓励代码生成过程中的多样性，从而显著提高了 LLM 的推理能力。
   - Hugh Zhang 表示，这种方法使 **LLM 在推理时能够进行更深层次的推理**，标志着 AI 领域内一个充满前景的方向。
- **RedTeam Arena 以游戏化形式发布**：一款名为 **RedTeam Arena** 的新游戏邀请参与者在 60 秒内挑战模型说出冒犯性词汇，旨在吸引 AI 黑客参与测试模型能力。
   - 该游戏旨在创建一个以竞争性提示（prompting）和红队测试（red teaming）为核心的**社区驱动平台**，所有数据集和提示词将在披露后公开。
- **Reflection 70b 模型能力**：最近讨论的 **Reflection 70b 模型** 据报道内置了利用 XLM 标签的草稿本（scratchpad），引发了人们对其高级推理潜力的好奇。
   - 社区成员推测，以“反思”（reflection）为核心的模型是否预示着多步问题解决的新范式，尽管一些人认为提示词（prompts）仍然起着更关键的作用。
- **关于 AI 研究欺诈的见解**：一个帖子强调了涉及 **OthersideAI** 宣布模型训练突破的涉嫌欺诈行为，其真实性受到质疑。
   - 讨论引用了一段欺骗的时间线，强调了 AI 研究与开发中**问责制的重要性**。
- **Itext2kg 作为知识图谱工具**：一个名为 **Itext2kg** 的 GitHub 项目提供了一个用户友好的工具，利用 LLM 从非结构化文档构建增量知识图谱，并可直接连接到 Neo4j。
   - 用户现在可以毫不费力地在生产环境中使用他们的本体（ontologies），为 **GraphRAG** 等传统学术工具提供了一个易于使用的替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1832201335175049434">来自 lmsys.org (@lmsysorg) 的推文</a>：⚠️警告：包含冒犯性内容。介绍带有“脏话”的 RedTeam Arena——我们的第一款游戏。你有 60 秒的时间让模型说出脏话。越快越好。（合作 ...</li><li><a href="https://arxiv.org/abs/2409.03733">自然语言规划改进了用于代码生成的 LLM 搜索</a>：虽然扩展训练计算量带来了大语言模型 (LLM) 的显著提升，但扩展推理计算量尚未产生类似的收益。我们假设一个核心缺失的组件...</li><li><a href="https://x.com/alexandr_wang/status/1832147956562284987?s=46">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：来自 Scale SEAL 的全新 SOTA 推理时计算结果⚡️ 我们正在发布一种名为 PlanSearch 的新 SOTA 推理时计算方法。它通过一种新的方式在 LiveCodeBench 上显著超越了现有方法...</li><li><a href="https://x.com/shinboson/status/1832933747529834747">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson) 的推文</a>：一个关于 AI 研究社区欺诈的故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破，允许他们训练一个中型模型...</li><li><a href="https://github.com/AuvaLab/itext2kg">GitHub - AuvaLab/itext2kg：使用大语言模型的增量知识图谱构建器</a>：使用大语言模型的增量知识图谱构建器 - AuvaLab/itext2kg
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1282055619054010410)** (2 条消息): 

> - `Medical LLM advancements` (Medical LLM 进展)
> - `Continual In-Context Learning`
> - `Transformer architecture` (Transformer 架构)
> - `Robotic Endoscopic Surgery` (机器人内窥镜手术)
> - `Decentralized Health Intelligence` (去中心化健康智能)


- **创新模型引领 Medical AI 进展**：重点介绍的 **CancerLLM** 和 **MedUnA** 等模型正在医疗语言模型和视觉语言任务领域铺平道路，增强了在肿瘤学和医学影像中的应用。
   - 这些模型在临床环境中发挥着至关重要作用，并得到了 [OpenlifesciAI 的帖子](https://x.com/OpenlifesciAI/status/1832476252260712788) 等倡议的支持，该帖子详细介绍了它们的影响。
- **基于 Adaptive Transformers 的 Continual In-Context Learning**：“基于 Adaptive Transformers 的 Continual In-Context Learning”架构扩展了 Transformer 在各种任务中的适用性，利用预训练的 Transformer 结合额外层进行自适应学习。
   - 它采用了一种双重方法，最初使用 **In-Context Learning**，仅在性能不足时才修改系统，旨在平衡适应性与风险管理。
- **医疗基准测试的扩展**：引入了 **TrialBench** 和 **DiversityMedQA** 等新基准，以评估 Medical LLM 在临床环境中的表现，并解决诊断过程中的偏差问题。
   - 这些评估对于提高模型可靠性和展示 Medical AI 应用不断发展的标准至关重要。
- **数字孪生与患者预测**：**Digital Twins for Rare Gynecological Tumors** 和 **DT-GPT** 等新兴技术将彻底改变患者健康预测，实现更个性化的医疗解决方案。
   - 这些创新标志着利用 AI 模拟患者状况并有效预测结果方面的进展。
- **Medical AI 应用框架**：正在开发 **Rx Strategist** 和 **Guardrails for Medical LLMs** 等框架，以增强处方验证并在 AI 使用中建立安全协议。
   - 这些努力对于确保 AI 在医疗保健中的部署符合安全和功效的高标准至关重要。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1832476252260712788">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：Medical AI 上周动态：顶级研究论文/模型 🏅（2024年9月1日 - 9月7日） Medical LLM 及其他模型：- CancerLLM：癌症领域的 Large Language Model - MedUnA：Vision-Languag...

  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1281809703562907690)** (2 条消息): 

> - `AGI through RL` (通过 RL 实现 AGI)
> - `Transformers and SSI` (Transformers 与 SSI)
> - `Importance of Scaling` (Scaling 的重要性)
> - `Breakthroughs Needed in AI` (AI 领域所需的突破)


- **AGI 可能通过强化训练和 RL 实现**：讨论强调了 **AGI** 有可能通过**强化训练**和**强化学习 (RL)** 来实现。
   - 然而，对于 **Transformers** 是否能导向 **Supervised Semantic Intelligence (SSI)** 存在疑问。
- **Scaling 可能会增强推理能力**：会议指出，通过在**大规模、多样化且干净的数据集**上进行训练，扩大模型规模（Scaling up）可能有助于解决**推理挑战**。
   - 这种方法可能会产生显著差异，尽管不足以完全模拟人类认知系统。
- **资源需求阻碍认知模拟**：人们对模拟人类认知系统的**资源需求**表示担忧，这使得其**极难进行 Scaling**。
   - 这表明**迫切需要** AI 领域的**新突破**来克服这些挑战。


  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1281700074274951281)** (16 条消息🔥): 

> - `Together AI 的 MLP Kernels`
> - `ROCm/AMD vs. NVIDIA`
> - `RTX 5XXX 架构代际`
> - `Reflection 争议事件`
> - `ROCm 上的 PyTorch` 


- **对 Together AI 的 MLP Kernels 的好奇**：成员们讨论了 Together AI 的 MLP Kernels 带来的 **20% 速度提升**，并特别提到 **SwiGLU** 可能是其中的关键因素。
   - *Tri Dao* 可能会在即将举行的 CUDA MODE IRL 活动中进一步探讨这一话题。
- **ROCm/AMD 与 NVIDIA 相比的困境**：有人询问为什么 **ROCm/AMD** 在 AI 浪潮中没有像 **NVIDIA/CUDA** 那样有效地获利，并质疑这是否与**企业信任**有关。
   - 另一位成员指出 **PyTorch 确实可以在 ROCm 上运行**，但实际性能仍然严重向 NVIDIA 硬件倾斜。
- **对 RTX 5XXX 架构的推测**：讨论包括对即将推出的 **RTX 5XXX** 系列将采用 **Blackwell** 还是 **Hopper** 架构的猜测。
   - 还有关于是否可能包含 **int/fp4 tensor cores** 的疑问。
- **Reflection 争议事件令人尴尬**：对话集中在 **Reflection 争议事件**上，一位成员将其描述为“令人尴尬”，并敦促其他人忽略它。
   - 分享了一个 Reddit 讨论链接，概述了从 Reflection 70B 中吸取的**教训**，强调了复现 Benchmark 的重要性。
- **PyTorch 在 ROCm 上的兼容性**：一位成员确认 **PyTorch** 确实可以在 **ROCm** 上运行，这补充了正在进行的关于硬件性能的对话。
   - 尽管具有兼容性，但与 NVIDIA 的产品相比，仍然存在明显的性能差距。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">使用 Together Kernel Collection 提升 NVIDIA H200 和 H100 GPU 集群性能</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fciqfp/reflection_70b_lessons_learned/">Reddit - 深入了解一切</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1281976608311480392)** (49 条消息🔥): 

> - `Triton 内部机制文章`
> - `FP16 vs BFP16 性能`
> - `Kernel 优化策略`
> - `量化技术` 


- **关于 Triton 内部机制的最终见解**：[Triton Internals](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/) 系列的最后一篇文章讨论了 MLIR 生成和渐进式 IR Lowering，提供了宝贵的学习经验。
   - 成员们对该系列表示赞赏，评论反映了其有用性。
- **测试 FP16 累加的加速效果**：一位成员对 FP16 配合 FP16 累加（Accumulation）相比其他类型的加速效果表示好奇。
   - 有人指出，虽然 FP16 累加通常更快，但其支持仅限于特定条件，尤其是在消费级设备上。
- **优化 Kernel Load**：讨论了创建一个将元数据与权重打包在一起的 Kernel，以减少 Load 次数，从而提高效率。
   - 有人担心将 Scale 和 Zero 与权重打包带来的开销和影响，这引发了针对 Batch Size 的潜在优化讨论。
- **Benchmark 和性能比较**：成员们讨论了在不同 Batch Size 下寻找一致速度的挑战，以及使用 TFlops 进行 Benchmark 的重要性。
   - 他们指出，报告相对于未量化 FP16 的加速比是很常见的，并且积极探索了性能增强的方案。
- **未来 Kernel 开发建议**：有人建议开发一个专注于 Batch-size 1 优化的 Kernel，以消除 Padding 造成的资源浪费。
   - 最终，社区表现出对尝试不同配置以增强性能的兴趣，特别是在低比特精度方面。



**提到的链接**：<a href="https://github.com/microsoft/BitBLAS/tree/main/benchmark">BitBLAS/benchmark at main · microsoft/BitBLAS</a>：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。- microsoft/BitBLAS

  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1282058495193120860)** (6 条消息): 

> - `Dynamo 调用分析`
> - `getitem 性能`
> - `PyTorch Container 模块`
> - `TorchDynamo 缓存查找` 


- **分析 Dynamo 调用**：成员们讨论了在 **Dynamo** 中追踪调用的问题，特别关注与 **getitem** 方法相关的性能差距。
   - 一位成员表示有兴趣了解这些调用的**来源**及其各自的**耗时**。
- **在 PyTorch 的 container.py 中识别来源**：[PyTorch container 模块](https://github.com/pytorch/pytorch/blob/31c4e0d37d8efc37a0697159e5b9121ec34d5141/torch/nn/modules/container.py#L332)中的一行相关代码被认为可能是导致迭代 **getitem** 调用的原因。
   - 正在调查的具体代码行是第 **320** 行，这引发了关于其影响的讨论。
- **TorchDynamo 缓存查找的挑战**：一位成员提到，搜索 **torchdynamo cache lookup** 只得到了一个包装器（wrapper），但缺乏关于直接调用的具体细节。
   - 这促使人们进一步探索 **Dynamo** 内部缓存管理的深入见解。



**提及的链接**：<a href="https://github.com/pytorch/pytorch/blob/31c4e0d37d8efc37a0697159e5b9121ec34d5141/torch/nn/modules/container.py#L332">pytorch/torch/nn/modules/container.py at 31c4e0d37d8efc37a0697159e5b9121ec34d5141 · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1282297172678934640)** (2 条消息): 

> - `消息中的自我推广` 


- **服务器限制自我推广**：一位成员强调了限制以**自我推广**为中心的消息的重要性，并表示只有与性能相关的内容才被认为是有吸引力的。
   - 另一位成员以一个 *oopsie* 承认了该反馈，表示他们理解了所提出的观点。
- **关于消息内容的反馈**：对话强调了服务器消息需要具备价值，不鼓励只发链接的帖子，除非它们与性能相关。
   - 这一反馈得到了积极响应，展示了社区对建设性互动的承诺。


  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1281708898792771624)** (18 messages🔥): 

> - `Course Lab Notebooks` (课程实验 Notebooks)
> - `Zen, CUDA, and Tensor Cores` (Zen, CUDA 和 Tensor Cores)
> - `VLLM Office Hours`
> - `AdEMAMix Optimizer` (AdEMAMix 优化器)
> - `Herbie Tool for Numerical Analysis` (用于数值分析的 Herbie 工具)


- **课程实验 Notebooks 备受推崇**：成员们讨论了某门课程的 **2023 实验 Notebooks**，强调了它们的质量以及对学习的实用价值。
   - *一位成员提到他们正在等待后续版本的发布*，但对现有材料表示了充分肯定。
- **关于 CUDA 的精彩 YouTube 内容**：分享了一个名为 *Zen, CUDA, and Tensor Cores - Part 1* 的 **YouTube 视频**，提供了关键概念的概述和见解。
   - 该视频是一个系列的一部分，更多信息可以在 [Computer Enhance](https://www.computerenhance.com/p/zen-cuda-and-tensor-cores-part-i) 找到。
- **最新 VLLM Office Hours 录音**：向感兴趣的成员分享了讨论量化 **CUTLASS GEMM 优化** 的最新 **VLLM Office Hours** 录音链接。
   - 这主要针对那些热衷于在 NVIDIA CUDA 相关工作中优化性能的人员，为 AI 协作提供了宝贵的见解。
- **AdEMAMix 优化器介绍**：分享了一篇 **arXiv 论文** 和 GitHub 仓库，讨论了 *AdEMAMix 优化器*，突出了优化器效率方面的进展。
   - 论文可以在 [arXiv](https://arxiv.org/pdf/2409.03137) 找到，代码仓库可在 [此处](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch) 获取。
- **Herbie 工具增强数值分析**：一位成员介绍了 **Herbie**，这是一个旨在通过各种实现提高输入方程速度和准确性的工具。
   - 建议在本地 [安装 Herbie](https://herbie.uwplse.org/demo/) 使用，以避免 Web 演示版的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://herbie.uwplse.org/demo/">Herbie web demo</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=uBtuMsAY7J8&ab_channel=MollyRocket">Zen, CUDA, and Tensor Cores - Part 1</a>：访问 https://www.computerenhance.com/p/zen-cuda-and-tensor-cores-part-i 获取更多信息、链接、附录以及该系列的其他视频。</li><li><a href="https://youtu.be/oAriAaOu00c?si=czCLyZCCHmTljPmf&t=256)">Advanced AI Accelerators and Processors with Andrew Feldman of Cerebras Systems</a>：在本期节目中，我们邀请到了 Cerebras Systems 的创始人兼 CEO Andrew Feldman。Andrew 和 Cerebras 团队负责构建了世界上最大的...</li><li><a href="https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch">GitHub - nanowell/AdEMAMix-Optimizer-Pytorch: The AdEMAMix Optimizer: Better, Faster, Older.</a>：AdEMAMix 优化器：更好、更快、更老。通过在 GitHub 上创建账号来为 nanowell/AdEMAMix-Optimizer-Pytorch 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1282039658997350401)** (27 messages🔥): 

> - `Tensor Core Efficiency`
> - `WMMA Usage`
> - `CUDA Kernel Optimization`
> - `Occupancy in Tensor Cores`
> - `CUDA Development Templates` 


- **理解 Matmul 中的 Tensor Core 效率**：一位成员解释说，与每个 warp 仅使用 **1 个 WMMA** 相比，在矩阵乘法中每个 warp 使用 **4 个 WMMA** 操作可以实现更好的流水线化（pipelining），从而提升整体性能。
   - 讨论强调，在 NVIDIA 的 Ampere 架构下，更高的算术密度会带来性能提升，特别是建议在操作中使用 **4x4 布局**。
- **对 WMMA 性能增益的批评**：*一位参与者不建议直接使用 WMMA*，认为像 **CUTLASS** 这样的框架对于从 Tensor Core 中提取最佳性能是必要的，尤其是在 FP32 操作中。
   - 他们指出，将 NVIDIA 的 **WMMA 示例**集成到代码中虽然比标准的 FP32 FMA 性能更好，但仍然落后于 **cuBLAS**。
- **Occupancy 和寄存器分配的挑战**：关于 **Occupancy** 的讨论显示，虽然更高的 Occupancy 可以提高资源利用率，但它需要减少每个线程的寄存器数量，从而限制了数据复用。
   - 一位成员提到，随着 **Hopper** 架构的到来，warp 之间的动态寄存器重新分配（dynamic register reallocation）可能会同时提高 Occupancy 和性能。
- **分享新的 CUDA 开发模板**：一位成员介绍了一个 **GitHub 模板**，旨在简化 CUDA C++ Kernel 的开发，方便在 **Python/PyTorch** 中进行测试。
   - 该计划旨在为未来的 CUDA 开发者提供一个流线型的配置环境，并收到了社区的积极反馈。
- **矩阵乘法代码的澄清**：成员们澄清了涉及 **wmma::mma_sync** 的代码片段，确认该示例实际上执行了 **16 个 matmul**，而不是最初陈述的 2x2 配置。
   - 对话强调了在优化矩阵乘法时，正确术语和理解 Kernel 操作的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVIDI">nvidi - 概览</a>：nvidi 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/tobiasvanderwerff/cuda-pytorch-template">GitHub - tobiasvanderwerff/cuda-pytorch-template: 一个用于开发 CUDA C++ kernel 并在 Python/PyTorch 中进行测试的简洁模板 🚀🚀</a>：一个用于开发 CUDA C++ kernel 并在 Python/PyTorch 中进行测试的简洁模板 🚀🚀 - tobiasvanderwerff/cuda-pytorch-template</li><li><a href="https://github.com/Leikoe/">Leikoe - 概览</a>：我 ❤️ 加速器。Leikoe 有 42 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/karpathy/llm.c/pull/696">由 ademeure 提交的重大 FP32 llm.c 改进/重构等 · Pull Request #696 · karpathy/llm.c</a>：我有点投入过度了，这最终显著改变了 train_gpt2_fp32.cu 中的几乎每一个 kernel！我还给 kernel 添加了很多注释——可能太多了，但是如果...</li><li><a href="https://github.com/Leikoe/cuda-explore/blob/main/matmul_tc_4x4.cu">cuda-explore/matmul_tc_4x4.cu at main · Leikoe/cuda-explore</a>：我的 CUDA 琢磨仓库。通过在 GitHub 上创建账号为 Leikoe/cuda-explore 的开发做贡献。</li><li><a href="https://github.com/Leikoe/cuda-explore/blob/main/matmul_tc.cu">cuda-explore/matmul_tc.cu at main · Leikoe/cuda-explore</a>：我的 CUDA 琢磨仓库。通过在 GitHub 上创建账号为 Leikoe/cuda-explore 的开发做贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1281847472544682014)** (2 messages): 

> - `PMPP Book for Parallel Computing`
> - `CUDA Resource Stream on GitHub` 


- **推荐初学者阅读 PMPP 书籍**：一位成员询问 **PMPP** 书籍是否是学习**并行计算**的最佳起点。
   - 作为回应，另一位成员确认它是新手的不错选择。
- **探索 GitHub 上的 CUDA 资源**：一位参与者建议查看 [CUDA Resource Stream GitHub 仓库](https://github.com/cuda-mode/resource-stream) 以获取更多有用的材料和链接。
   - 该仓库汇集了各种 **CUDA 相关新闻和材料链接**，帮助开发者保持更新。



**提到的链接**：<a href="https://github.com/cuda-mode/resource-stream">GitHub - cuda-mode/resource-stream: CUDA 相关新闻和材料链接</a>：CUDA 相关新闻和材料链接。通过在 GitHub 上创建账号为 cuda-mode/resource-stream 的开发做贡献。

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1281709480236810250)** (2 条消息): 

> - `构建修复`
> - `GitHub Pull Requests` 


- **通过 Pull Request #826 修复构建问题**：一名成员建议 [这个 Pull Request](https://github.com/pytorch/ao/pull/826) 应该能修复 PR #621 之后持续存在的构建问题。
   - 另一名成员确认并表示，这**似乎已经解决了**他们的问题，并对帮助表示感谢。
- **调试协作**：对话强调了协作，一名成员标记了另一名成员以寻求有关构建问题的协助。
   - 这种方法体现了社区在解决开发过程中出现的技术挑战时的积极努力。



**提到的链接**：<a href="https://github.com/pytorch/ao/pull/826">Unbreak build after #621 by andrewor14 · Pull Request #826 · pytorch/ao</a>：未找到描述

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1282235431680737310)** (14 条消息🔥): 

> - `马拉松体验`
> - `伤病恢复`
> - `CUDA 相关内容`
> - `图片遮盖`
> - `徒步事故` 


- **马拉松挑战与挫折**：一名成员分享了参加马拉松的兴奋之情，但最终因严重的腿部抽筋在 *20 英里左右退出*，将健康置于完赛之上。
   - 他们幽默地承认了这一困境，表达了在尝试活动时不希望受伤的意图。
- **徒步时严重的踝关节受伤**：另一名成员报告了在徒步时发生的 *严重踝关节受伤*，导致最近进行了一次顺利的手术。
   - *他们表达了在恢复期间被困在房间里的沮丧* 以及保持动力的挑战。
- **受伤导致更多的编程工作**：一名成员反思了伤病如何迫使他们投入更多精力到编程中，因为他们无法参加体育运动，在艰难的处境中找到了一线希望。
   - 他们指出这种重心的转移是一种应对机制，强调了身体局限对爱好的影响。
- **寻求恢复期间的视频推荐**：受伤的成员请求推荐 *CUDA 相关视频和算法*，以帮助度过恢复期的时光。
   - 他们表示目前处于动力低谷，正在寻找能让大脑保持活跃的内容，尽管身体受到限制。
- **关于图片遮盖的技术咨询**：讨论中包括一个关于如何 *给图片添加剧透遮盖* 的问题，并很快找到了解决方案。
   - 该成员分享了他们 *严重淤青的脚踝* 的链接（现已解决），展示了对平台的积极使用。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1282047430136959057)** (6 条消息): 

> - `多伦多 GPU 编程线下聚会`
> - `Triton 学习`
> - `Cutlass 兴趣` 


- **筹备中的多伦多 GPU 编程线下聚会**：一名成员表示有兴趣在 **多伦多** 组织 GPU 编程线下聚会，如果有足够的兴趣，欢迎其他人协作。
   - *很想看看这里有哪些人是在多伦多的！*
- **组建 GPU 编程读书小组**：提到了组建 GPU 编程读书小组或工作组的想法，并对深入参与该主题表现出极大的热情。
   - *一名成员指出那会非常酷！*
- **学习 Triton 和 Cutlass 的热情**：成员们分享了对 **Triton** 和 **Cutlass** 的浓厚兴趣，突显了学习这些 GPU 编程工具的愿望日益增长。
   - *两人都表示了进一步探索 Triton 和 Cutlass 的个人兴趣。*


  

---

### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1282031359115919431)** (10 条消息🔥): 

> - `Triton-Puzzles 错误处理`
> - `安装 Triton-Viz`
> - `Localhost 403 错误` 


- **用户努力解决 Triton-Puzzles 错误**：一名成员报告在运行 Triton-Viz 时遇到 **TypeError**，错误信息为 ‘_init_args_hst() missing 1 required positional argument: 'kwargs'’，并指出这与现有的 [GitHub issue](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/issues/33) 有关。
   - 另一名成员澄清说 **AlphaGo** 已经提供了一个解决方案，尽管它不适用于当前遇到的错误。
- **尝试通过重建环境修复错误**：在删除虚拟环境后，一名成员提到他们按照 **AlphaGo** 的安装说明进行操作，但仍然面临同样的错误。
   - 他们分享了更新后的输出，显示应用正在 `http://127.0.0.1:5000` 上运行，但在访问该地址时遇到了 **403** 错误。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://127.0.0.1:5000,">未找到标题</a>：未找到描述</li><li><a href="https://github.com/Deep-Learning-Profiling-Tools/triton-viz/issues/33">Triton Puzzle was broken (by a recent change?) · Issue #33 · Deep-Learning-Profiling-Tools/triton-viz</a>：在尝试此处的 Colab Notebook 时：https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb#scrollTo=_981RFRp4Avz 我遇到了一个关于 kwargs 的早期错误...</li><li><a href="https://github.com/triton-lang/triton/pull/3777.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1281850191653376002)** (2 条消息): 

> - `HFGenerator`
> - `Batch Size 支持` 


- **HFGenerator 限制为 Batch Size 为 1**：确认 **HFGenerator** 仅支持 **batch_size=1**，使用默认的 **Hugging Face generator** 是一个替代方案。
   - *Mobicham* 对 Hugging Face 中的 static cache 是否支持 **batch_size > 1** 表示不确定。
- **默认 Hugging Face Generator 作为替代方案**：由于 **HFGenerator** 被限制为 Batch Size 为 1，可以利用默认的 **Hugging Face generator** 作为替代。
   - 这一替代方案可以帮助那些尽管有局限性但仍需处理更大 Batch 的用户。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1281701604801904715)** (2 条消息): 

> - `H100 扩展性`
> - `NCCL 多 GPU 训练` 


- **Chinthysl 展示了在 472x H100 上的线性扩展**：Chinthysl 展示了早在 6 月份在 **472x H100** 上运行的情况，在训练过程中实现了至少到 **128 个 GPU** 的**线性扩展**。
   - 成员们注意到，与 MPI 相比，使用 [Slurm](https://github.com/karpathy/llm.c/pull/426#issuecomment-2175386065) 调度作业更加容易，特别是对于多节点设置。
- **关于 Token 扩展性能的讨论**：讨论强调，在 **128 个 GPU** 之上的早期 Token 扩展数据可能没有更新，这引发了对某些修复后所做调整的好奇。
   - 成员们发现该系统能够良好扩展令人印象深刻，并对未来的性能基准测试感到兴奋。



**提到的链接**：<a href="https://github.com/karpathy/llm.c/pull/426#issuecomment-2175386065),">NCCL only multi-gpu multi-node training without MPI by chinthysl · Pull Request #426 · karpathy/llm.c</a>：在多节点训练设置中，使用 Slurm 调度作业似乎比为集群设置 MPI 要容易得多。此草案包含了在单节点训练中使用 mpirun 的更改，以及 S...

  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1282748978630361272)** (1 messages): 

> - `AMD's UDNA Architecture` (AMD 的 UDNA 架构)
> - `Deprioritization of High-End Gaming GPUs` (降低高端游戏 GPU 的优先级)
> - `Transition from GCN to RDNA and CDNA` (从 GCN 到 RDNA 和 CDNA 的转型)


- **AMD 将 RDNA 和 CDNA 统一为 UDNA**：在柏林举行的 IFA 2024 上，AMD 的 **Jack Huynh** 宣布将面向消费者的 **RDNA** 和面向数据中心的 **CDNA** 架构统一为名为 **UDNA** 的单一微架构，旨在更好地与 Nvidia 的 **CUDA ecosystem** 竞争。
   - 这一进展标志着 AMD 的战略转变，旨在应对游戏和计算中心化需求的同时，提升其在 **market** 中的竞争地位。
- **AMD 降低旗舰游戏 GPU 的优先级**：正如 Huynh 的公告所反映的，AMD 已决定[降低高端游戏显卡的优先级](https://www.tomshardware.com/pc-components/gpus/amd-deprioritizing-flagship-gaming-gpus-jack-hyunh-talks-new-strategy-for-gaming-market)，以提高其市场份额。
   - 这一转变表明 AMD 专注于更广泛的战略目标，而不是仅仅在高端游戏领域进行竞争。
- **从 GCN 到新架构**：在 2019 年告别 **GCN** 微架构时，AMD 选择为其图形微架构创建不同的设计：用于游戏 GPU 的 **RDNA** 和用于计算及 HPC 工作负载的 **CDNA**。
   - 统一为 **UDNA** 标志着 AMD 在 GPU 领域方法上的关键演进，融合了游戏和计算能力。



**提到的链接**：<a href="https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem">AMD 宣布统一的 UDNA GPU 架构 —— 将 RDNA 和 CDNA 结合起来，对抗 Nvidia 的 CUDA ecosystem</a>：合二为一。

  

---


### **CUDA MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1282759162832621690)** (1 messages): 

> - `ExecuTorch`
> - `PyTorch` 


- **ExecuTorch 中的 ARM 工作进展**：一位成员提到他们一直在 **ExecuTorch** 和 **PyTorch** 中专门从事 **ARM** 相关的任务。
   - “只是进来打个招呼”表明其与社区的持续互动。
- **关于 PyTorch 应用的讨论**：该成员在 **PyTorch** 中的参与表明其专注于将该框架应用于与 **ARM** 相关的实际场景中。
   - 他们似乎渴望分享自己的见解，这体现了社区内的协作精神。


  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1281703003895890000)** (19 条消息🔥): 

> - `Liger 的 Swiglu Kernel 与 Together AI 基准测试对比`
> - `cuBLAS 和 PyTorch 实现中的优化`
> - `Cross Entropy 中 ignore_index 的处理`
> - `Conv2D 性能问题`
> - `在 A100 上对 Phi3 进行基准测试` 


- **Liger 的 Swiglu Kernel 优于 cuBLAS**：一名成员声称他们的专用 Kernel 比使用 **cuBLAS** 和 **PyTorch eager mode** 的常用实现快 **22-24%**。
   - 他们询问了 Together AI 是如何实现显著加速的，引发了关于性能基准测试的讨论。
- **解决代码中 ignore_index 的疑虑**：有人担心当 `y_i == ignore_index` 时可能存在无效内存访问，但随后澄清了 Kernel 通过提前返回（early returns）无误地处理了这种情况。
   - 分享了一个确认 `ignore_index` 处理情况的额外测试用例，展示了稳健的测试。
- **Conv2D 性能下降**：注意到 **Conv2D** 的性能问题，尽管在较小的基准测试中表现相似，但随着输入和输出通道的增加，性能似乎有所下降。
   - 讨论强调了改进的必要性，因为在某些条件下，相对于 **Torch**，其性能似乎有所减弱。
- **Phi3 基准测试挑战**：一位用户报告在使用 **Flyte** 编排基准测试时，难以在单张 **A100 40GB** 上达到预期的 Token 吞吐量。
   - 他们提到正在参考并适配仓库中提供的示例，并计划探索**多 GPU 分布式训练**。
- **性能调优的下一步**：成员们承认在性能调优讨论中索引处理存在某些不准确之处，并提到正在调查 **pyproject.toml** 的问题。
   - 强调了一个拟议的修复方案，该方案可以解决 nightly 和 main 版本中的打包检测问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.together.ai/blog/nvidia-h200-and-h100-gpu-cluster-performance-together-kernel-collection">通过 Together Kernel Collection 提升 NVIDIA H200 和 H100 GPU 集群性能</a>：未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/638b31057d283a0d841a1795f742068a63b7dcdd/test/transformers/test_cross_entropy.py#L33">Liger-Kernel/test/transformers/test_cross_entropy.py at 638b31057d283a0d841a1795f742068a63b7dcdd · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/226#issuecomment-2336479305">(fix) fix pyproject.toml by wizyoung · Pull Request #226 · linkedin/Liger-Kernel</a>：摘要：在 #218 中，我修复了 tool.setuptools.packages.find 字段，并仅在 pip install -e . 的可编辑模式下进行了测试。然而，在 pip install . 的生产模式下，只有 env_report.py 文...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/638b31057d283a0d841a1795f742068a63b7dcdd/src/liger_kernel/ops/cross_entropy.py#L65">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at 638b31057d283a0d841a1795f742068a63b7dcdd · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/638b31057d283">GitHub - linkedin/Liger-Kernel at 638b31057d283a0d841a1795f742068a63b7dcdd</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/236">在单张 A100 40gb GPU 上对 phi3 进行基准测试：无法复现基准测试结果 · Issue #236 · linkedin/Liger-Kernel</a>：🐛 描述 Bug：我正在使用 flyte 在略有不同的条件下复现此仓库 README 中报告的 Token 吞吐量和内存节省结果：使用 microsoft/Phi-3-m...</li><li><a href="https://github.com/linkedin/Liger-Kernel/tree/main/examples/huggingface">Liger-Kernel/examples/huggingface at main · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernel。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[thunder](https://discord.com/channels/1189498204333543425/1281971603252580484/1282005293064261683)** (4 条消息): 

> - `Thunder 频道介绍`
> - `Triton Matmul 示例`
> - `融合操作 (Fusing operations)`
> - `Liger kernel 应用` 


- **介绍用于源码到源码编译 (Source-to-Source Compilation) 的 Thunder 频道**：Thunder 频道旨在将 **原生 PyTorch 模型** 编译为优化的 Python 函数，贡献者包括 <@790925083828682752>、<@222363567192670219> 和 <@761222713611386900>。
   - 他们邀请其他人尝试 Thunder 并提供反馈，以改进其功能。
- **探索 Triton Matmul 集成**：一周前，一个涵盖 **Triton Matmul 示例** 的会议展示了如何使用 Thunder 将自定义 kernel 集成到模型中，详情见 [YouTube 视频](https://www.youtube.com/watch?v=i79Op6DXI7c)。
   - 该会议强调实际应用而非理论，以帮助理解集成过程。
- **向 Thunder 添加融合操作**：本周，Thunder 团队宣布在其编译器中添加了 **融合操作 (fusing operations)**，并在 [最新的 YouTube 会议](https://www.youtube.com/watch?v=DF7_XGUmCD8) 中进行了分享。
   - 这一进展延续了关于提高深度学习编译器效率的讨论。
- **下一步：将融合应用于 Liger kernel**：团队的下一个目标是将融合技术应用于 **Liger kernel**，展示了 Thunder 能力的持续发展。
   - 这体现了扩展 Thunder 功能和性能的承诺。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=i79Op6DXI7c&list=PLaMu-SDt_RB7ImARcTT_Wjypwx2vBIBen&index=6">The Thunder Sessions | Session 6 | More Transforms, Less Theory</a>：在今天的会议中，Luca 和 Tom 将专注于更多的 Transforms（变换），更少的理论！The Thunder Sessions 是关于深度学习编译器及其实现细节的讨论...</li><li><a href="https://www.youtube.com/watch?v=DF7_XGUmCD8&list=PLaMu-SDt_RB7ImARcTT_Wjypwx2vBIBen&index=7">The Thunder Sessions | Session 7 | Fusing Kernels with Thunder &amp; Triton</a>：The Thunder Sessions 是关于深度学习编译器及其实现细节的讨论，主持人包括 CTO Luca Antiga 和首席研究员 Thomas Viehmann...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1281746568353022013)** (112 messages🔥🔥): 

> - `Reflection Llama-3.1 updates` (Reflection Llama-3.1 更新)
> - `OpenAI model announcements` (OpenAI 模型公告)
> - `AI hardware requirements` (AI 硬件需求)
> - `Learning OpenAI API` (学习 OpenAI API)
> - `Performance of local models` (本地模型性能)


- **Reflection Llama-3.1 的性能更新**：最近发布的 **Reflection Llama-3.1 70B** 被誉为全球顶尖的开源 LLM，它采用了一种名为 Reflection-Tuning 的技术，旨在增强模型的推理能力。
   - 用户注意到模型初期存在一些问题，目前已得到解决，建议测试者重新尝试以获得更好的结果。
- **关于 OpenAI 模型公告的澄清**：讨论显示了对“GPT Next”这一术语的怀疑，OpenAI 澄清这仅仅是一个比喻性的占位符，没有具体含义。
   - 尽管意见不一，一些成员对 OpenAI 在即将推出的模型炒作中缺乏实质性更新表示沮丧。
- **运行模型的硬件规格**：为了有效运行像 **Llama 3.1 70B** 这样的本地模型，用户需要一台配备充足 GPU 的 PC 或搭载 Apple Silicon 的 Mac；8GB VRAM 被提及为实现最佳性能的最低要求。
   - 一位用户分享了在高配置 MacBook Pro 上运行高强度模型的经验，并将其与缺乏足够资源的配置进行了对比，强调了硬件的重要性。
- **学习 OpenAI API 和使用限制**：一位成员在尝试使用 OpenAI API 时遇到了错误代码 429，询问尽管是新账号为何仍有账号限制。
   - 其他人建议购买额度或利用模型的免费使用选项来缓解问题，并建议从更简单的模型开始以降低学习难度。
- **本地模型性能探索**：用户讨论了在低配硬件上运行大模型的可行性，分享了在仅有 4GB RAM 的低端笔记本电脑上性能表现不佳的轶事。
   - 结论是，虽然实验很有趣，但高性能模型在实际使用中需要强大的计算资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/stevenheidel/status/1832523212724564387...">来自 Steven Heidel (@stevenheidel) 的推文</a>: something ever happens</li><li><a href="https://mashable.com/article/openai-clarifies-no-gpt-next-not-a-new-model">OpenAI 澄清：不，“GPT Next”不是一个新模型。</a>: 演示文稿引起的混乱让 OpenAI 的粉丝们感到不安。</li><li><a href="https://www.promptingguide.ai/techniques/reflexion">提示工程指南</a>: 提示工程的全面概述</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF">bartowski/Meta-Llama-3.1-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B">mattshumer/Reflection-Llama-3.1-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/TheRealAdamG/status/1832823604914106503">来自 Adam.GPT (@TheRealAdamG) 的推文</a>: Angel - 如果我和同事开内部玩笑这件事让你觉得成问题，以至于你基本上是在叫我闭嘴——这有点不公平，我建议你应该屏蔽我...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1282132131425943662)** (7 messages): 

> - `GPT handling books` (GPT 处理书籍)
> - `Voice access rollout` (语音访问权限推出)


- **GPT 将书籍作为知识文件处理**：一位成员询问了 GPT 在上传整本书作为知识文件时的处理效果，另一位成员解释说 GPT 将文件作为搜索特定信息的参考，而不是完全“知晓”其内容。
   - 这一见解似乎让询问者感到安心，他指出这一功能非常有用并对解释表示感谢。
- **对高级语音模式（Advanced Voice Access）推出的担忧**：一位成员质疑高级语音功能的推出是真实的还是仅仅是延迟用户访问的策略，引发了其他人的好奇。
   - 这引发了其他用户对类似挫折的确认，至少有一位成员尝试获取访问权限但未成功。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1281716160777162895)** (30 条消息🔥): 

> - `AI 推理过程拆解`
> - `Prompt Engineering 见解`
> - `股市 Prompt 使用案例`
> - `不同的回复风格`
> - `Prompt Library 频道位置` 


- **AI 推理过程拆解非常有趣**：成员们讨论了要求 AI 拆解其对所提供回复的推理过程的吸引力，并建议其他人刷新查询以获得不同的视角。
   - 一位成员将这种不确定性比作一个试图提供讨好答案的幼儿，这是一个有趣的观察。
- **使用特定风格的 Prompt 以获得更好的输出**：一位成员建议，在 Prompt 前加上类似“以 Terry Pratchett 的写作风格”的前缀可以产生极好的效果。
   - 这种方法表明，调整 Prompt 可以增强 AI 回复的创造力和参与度。
- **对使用 LLM 进行股票分析的担忧**：针对使用 LLM 衡量股票数据兴趣的讨论引发了争议，观点认为这种方法存在局限性且效率低下。
   - 成员们建议不要仅仅依赖 Prompt 进行股票分析，主张使用传统模型进行数据评估。
- **ChatGPT 对 Prompt Engineering 的响应**：成员们分享了推荐使用输出模板进行有效的 Prompt Engineering，暗示了改进交互的结构化方法。
   - 一位成员还指出，来自实时数据的持续更新增强了股票评估相关任务的性能。
- **寻找 Prompt Library 频道**：有人询问 Prompt Library 频道的位置，强调了轻松获取资源的重要性。
   - 另一位成员迅速回应，引导询问者前往特定频道寻求帮助。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1281716160777162895)** (30 条消息🔥): 

> - `AI 推理过程拆解`
> - `AI 回复的多样性`
> - `API 讨论与 Prompt`
> - `使用 AI 进行股票历史分析`
> - `使用 AI 判断趣味性` 


- **AI 推理过程拆解引发关注**：成员们发现要求 AI 解释其回复背后的推理过程非常有趣，通过重复 Prompt 可以产生多样化的见解。
   - *Madame_architect* 指出：“刷新几次回复，看看它有多少种不同的回复方式和不同的路径。”
- **为获得更好结果的 API 讨论**：多位用户讨论了在与 API 交互时使用输出模板和任务分块（chunking tasks）以获得更好结果的重要性。
   - *Darthgustav* 分享了关于 Prompt Engineering 的见解，强调虽然他们不是 API 专家，但有效的 Prompt 有助于更强大的交互。
- **AI 在股票分析中的局限性**：成员们警告不要在没有全面数据的情况下使用 OpenAI 模型分析股票，强调了实时更新的重要性。
   - *Niko3757* 解释了准确评估所需的历史数据和实时更新的必要性，建议从可靠来源下载股票历史记录。
- **探索用于判断趣味性的 Prompt**：一位用户寻求创建评估各种输入因素“趣味性”的 Prompt 案例，旨在利用 LLM 作为评判者。
   - *Sps0707* 澄清他们的意图不仅限于股票相关，而是专注于更广泛的用于衡量兴趣的 Prompt 应用。
- **AI 讨论中的对话协作**：成员们参与了协作讨论，分享技巧并尝试 Prompt 以实现理想的 AI 行为。
   - 讨论语气轻松愉快，参与者之间交流了笑话和随意的鼓励。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1281730945266749450)** (80 条消息🔥🔥): 

> - `C 与 Mojo 的集成`
> - `LLVM 开发者大会见解`
> - `Mojo 中的 Subprocess 实现`
> - `Mojo 社区会议交接`
> - `哈希函数演示` 


- **通过 DLHandle 集成 C 与 Mojo**：成员们讨论了如何使用 `DLHandle` 将 **C** 代码与 **Mojo** 集成，通过动态链接共享库实现两者之间的函数调用。
   - 提供了一个示例，展示了从 C 库加载后成功执行检查数字是否为偶数的函数。
- **LLVM 开发者大会见解**：即将于 10 月举行的秋季 LLVM 开发者大会将包含 **5 场由 Modular 带来的演讲**，主题涵盖 **Mojo** 和 **GPU 编程**。
   - 与会者对预期的讨论表示兴奋，并分享了活动结束后录播视频将在 [YouTube](https://www.youtube.com/@LLVMPROJ) 上发布。
- **对 Mojo 中 Subprocess 实现的期待**：一位成员表示有兴趣在未来为 **Mojo stdlib** 实现 **Subprocess** 功能，反映了增强该标准库的愿望。
   - 成员们还讨论了在尝试搭建 **Mojo** 开发环境时的资源问题，特别是在旧硬件上。
- **社区会议负责人交接**：Tatiana 宣布将 **Mojo 社区会议** 的领导权移交给 Caroline，并感谢大家迄今为止的参与和贡献。
   - 社区会议议程包括关于复杂算法中的 SIMD 和哈希函数的讨论。
- **mzaks 的哈希函数演示**：一位成员分享了题为 **'Hash Functions and Where to Find Them'** 的演示文稿 PDF，并链接到了他们在 GitHub 仓库中用 **Mojo** 实现的相关函数。
   - 该演示是社区会议的一部分，展示了实际实现并为参与者分享了资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/Apple-MacBook-Touch-Intel-Quad-Core/dp/B0BTMNF41B?th=1&psc=1">未找到标题</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/sys/ffi/DLHandle">DLHandle | Modular Docs</a>：表示可以加载和卸载的动态链接库。</li><li><a href="https://modul.ar/community-meeting-zoom">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简便、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。</li><li><a href="https://www.youtube.com/@LLVMPROJ">LLVM</a>：llvm.org 官方 YouTube 频道。查看 LLVM 开发者大会视频及更多内容！</li><li><a href="https://github.com/mzaks/mojo-hash/blob/main/HashFunctionsAndWhereToFindThem.pdf">mojo-hash/HashFunctionsAndWhereToFindThem.pdf at main · mzaks/mojo-hash</a>：在 Mojo 中实现的哈希函数集合 - mzaks/mojo-hash</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：Mojo 社区会议文档链接：https://modul.ar/community-meeting-doc。这是一个公开文档；欢迎所有人查看并发表评论/建议。所有会议参与者必须遵守...</li><li><a href="https://modul.ar/community-meeting-zoom.">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简便、可靠的云平台，适用于移动端、桌面端和会议室系统的视频和音频会议、聊天及网络研讨会。</li><li><a href="https://discourse.llvm.org/t/announcing-the-2024-llvm-developers-meeting-program/81108/1">宣布 2024 LLVM 开发者大会议程</a>：我很高兴宣布 2024 LLVM 开发者大会议程！快速提醒一下，早鸟注册将于 9 月 20 日截止。这也是保证获得 T 恤的注册截止日期...</li><li><a href="https://llvm.swoogo.com/2024devmtg/home">LLVM 开发者大会 2024</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1281722355290214430)** (96 messages🔥🔥): 

> - `DType 作为 Dict 键`
> - `多精度整数算术`
> - `Mojo 硬件访问驱动`
> - `Variant 类型用法`
> - `为 GStreamer 创建绑定` 


- **DType 无法用作 Dict 键**：讨论集中在为什么 `DType` 不能作为 Dict 的键，其中 *DType.uint8* 被指出是一个值而非类型。
   - 有人提到，由于其与 SIMD 类型的关系，更改实现可能并不简单，而 SIMD 类型目前具有特定的约束。
- **探索 Mojo 中的多精度整数支持**：成员们讨论了在 Mojo 中开发多精度整数算术包的可能性，并参考了类似于 Rust 中的实现。
   - 一位参与者分享了一个 [GitHub 链接](https://github.com/zmalatrax/uint)，展示了他们在多精度算术 `uint` 包上的进展。
- **Mojo 的硬件访问驱动能力**：确认了 Mojo 可以编写用户空间（userspace）驱动程序，尽管目前尚不支持底层内核开发。
   - 主要目标是替换像 CUDA 这样的组件，重点关注用户空间交互而非裸机（bare-metal）编程。
- **使用 Variant 类型处理多种元素类型**：对话强调了在 Mojo 中使用 `Variant` 来创建包含不同 struct 类型的多态列表。
   - 示例展示了成员如何利用 `Variant` 存储不同的基础类型，不过目前仍不支持存储 `Trait` 的实例。
- **在 Mojo 中为 GStreamer 创建绑定**：一位用户询问了如何在 Mojo 中为 GStreamer 创建绑定，引发了关于可用方法的讨论。
   - 建议使用 FFI 模块 `DLHandle` 或通过 Python 导入，尽管没有提供具体的 GStreamer 细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular 文档</a>：定义了一个 Variant 类型。</li><li><a href="https://en.cppreference.com/w/cpp/freestanding">Freestanding and hosted implementations - cppreference.com</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/int_literal/IntLiteral">IntLiteral | Modular 文档</a>：此类型表示具有无限精度的静态整数文字值。它们无法在运行时实例化，必须转换为其他整数类型（如 Int），但允许编译时...</li><li><a href="https://www.youtube.com/watch?v=JRcXUuQYR90">Mojo Lang - 明天的高性能 Python？（对话 Chris Lattner）</a>：Mojo 是来自 Swift 和 LLVM 创始人的最新语言。它尝试采用 CPU/GPU 级编程的一些最佳技术并封装...</li><li><a href="https://docs.modular.com/mojo/manual/types#simd-and-dtype">Types | Modular 文档</a>：标准 Mojo 数据类型。</li><li><a href="https://github.com/modularml/mojo/issues/3455">[BUG] 无法将 SIMD 数据类型用作 Dict 的键 · Issue #3455 · modularml/mojo</a>：Bug 描述：尽管 SIMD 似乎符合 KeyElement 的要求，但无法将 SIMD 数据类型（UInt8, Int16 等）用作 Dict 键。重现步骤：from collections import Dict var map1 = Dict[...</li><li><a href="https://github.com/r">r - 概览</a>：r 有 4 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/zmalatrax/uint">GitHub - zmalatrax/uint: Mojo `uint` 包 - 多精度整数算术</a>：Mojo `uint` 包 - 多精度整数算术 - GitHub - zmalatrax/uint: Mojo `uint` 包 - 多精度整数算术</li><li><a href="https://github.com/recmo/uint">GitHub - recmo/uint: 使用 const-generics 的 Rust Uint crate</a>：使用 const-generics 的 Rust Uint crate。通过在 GitHub 上创建账号来为 recmo/uint 做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">mojo/stdlib/src/utils/variant.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1281694620064944229)** (124 条消息🔥🔥): 

> - `DeepMind 的转型`
> - `Quora 数据抓取`
> - `Continual In-Context Learning`
> - `Adaptive Transformers`
> - `AI Hackathons` 


- **DeepMind 员工分享见解**：一位前 DeepMind 员工指出，项目所需的 **compute** 很大程度上取决于其 **product-focus**，这揭示了在转向 GenAI 后的当前资源分配变化。
   - 这引发了关于从事基础研究（foundational research）可能导致资源减少的讨论，正如社区所持的怀疑态度。
- **抓取 Quora 数据的挑战**：成员们讨论了将 **Quora 数据** 纳入 AI 训练数据集的可能性，并指出其内容虽然有价值，但通常受到限制。
   - 针对 **Quora 的 TOS**（服务条款）提出了担忧，认为由于严格的规定，抓取数据可能并不可行。
- **讨论 Adaptive Transformers 架构**：分享了关于 'Continual In-Context Learning with Adaptive Transformers' 的详细描述，重点讨论其如何使 Transformers 在不修改 **parameters** 的情况下，利用先验知识适应新任务。
   - 该方法旨在实现高适应性，同时最大限度地降低灾难性失效（catastrophic failure）的风险，其在各个领域的应用前景引起了关注。
- **对 AI Hackathon 的好奇**：几位成员回忆了由 Eleuther AI 组织的 **AI hackathon**，回想起有趣的参与者和实验。
   - 特别提到了一个 RLHF hackathon，尽管具体地点尚不确定。
- **AI 模型训练建议**：用户讨论了用于聊天机器人审核（moderation）任务的模型推荐，提到 **Mistral 7b** 和 **LLaMA-3.1-8b** 可作为进一步探索的起点。
   - 社区建议利用拒绝数据集（rejection dataset）来增强所选模型的审核能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07524">Simple and Effective Masked Diffusion Language Models</a>：虽然 Diffusion 模型在生成高质量图像方面表现出色，但先前的研究报告称，在语言建模中，Diffusion 与 Autoregressive (AR) 方法之间存在显著的性能差距。在这项工作中，我们...</li><li><a href="https://aaronlou.com/blog/2024/discrete-diffusion/#learning-concrete-scores-with-score-entropy">Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou</a>：未找到描述内容
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1281754140585889803)** (20 条消息🔥): 

> - `梯度的余弦相似度`
> - `贝叶斯深度学习中的 Laplace Approximation`
> - `Weight Decay 与正交正则化`
> - `贝叶斯方法中的先验 (Prior)`
> - `训练动力学与相位变化` 


- **余弦相似度指示梯度模式**：通过比较第 N 步和第 N+1 步梯度的余弦相似度发现，在训练数据集的某些序列上，**梯度趋于一致（align）**，这表明存在显著的模式。
   - 这表明问题不仅限于梯度幅值过大，因为可能出现的模式会导致持续的方向性偏移。
- **Laplace Approximation 简化贝叶斯分析**：成员们讨论了如何通过关注输出层的 **Hessian** 矩阵，利用 Laplace Approximation 来简化 ReLU 网络的分析。
   - 对话指出了在实现过程中调整 **先验精度（prior precision）** 和平衡协方差缩放（covariance scaling）所面临的挑战。
- **关于 Weight Decay 与正交正则化的讨论**：一位成员对在投影中使用 **Weight Decay** 同时配合正交正则化表示担忧，思考由于力量冲突可能导致的崩溃（collapse）等潜在问题。
   - 虽然 Weight Decay 可能带来理想的稀疏化，但它与基于损失的正交正则化之间的相互作用引发了关于稳定性的疑问。
- **贝叶斯模型中先验 (Prior) 的重要性**：在贝叶斯方法中忽略先验被认为具有重大影响，讨论表明 **考虑先验** 可以极大地影响模型性能。
   - 一条幽默的评论强调了在这些语境下，指数分布作为无记忆先验的作用。
- **用于高效 Attention 的半径最近邻 (Radius Nearest Neighbor)**：有人建议在 Attention 机制中实现半径最近邻查询，以期实现 **渐近更快 (asymptotically faster)** 的计算。
   - 该方法依赖于潜变量 (latents) 的结构特性，为处理 Attention 任务的优化开辟了道路。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard">distily/distily_attn_mlp_sweep · 训练指标</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.04431">Theory, Analysis, and Best Practices for Sigmoid Self-Attention</a>：Attention 是 Transformer 架构的关键部分。它是一种序列到序列的映射，将每个序列元素转换为值的加权和。权重通常通过...获得。</li><li><a href="https://arxiv.org/abs/2002.10118">Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks</a>：ReLU 分类网络的点估计（可以说是应用最广泛的神经网络架构）已被证明在远离训练数据的地方会产生任意高的置信度。...</li><li><a href="https://arxiv.org/abs/2106.14806">Laplace Redux -- Effortless Bayesian Deep Learning</a>：深度学习的贝叶斯公式已被证明具有引人注目的理论特性，并提供实际的功能优势，例如改进的预测不确定性量化和模型...</li><li><a href="https://openreview.net/forum?id=FJiUyzOF1m">Bayesian Low-rank Adaptation for Large Language Models</a>：参数高效微调 (PEFT) 已成为大语言模型 (LLM) 成本高效微调的新范式，其中低秩自适应 (LoRA) 是一种被广泛采用的选择。...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1282086516469923851)** (13 messages🔥): 

> - `Power Law Curves in ML`
> - `Self-Organized Criticality`
> - `Scaling Laws in Statistical Estimation`
> - `Sandpile Avalanche Model`
> - `Critical Systems and Fluctuations` 


- **关于幂律曲线的讨论**：成员们讨论了为什么 **power law curves** 在建模 ML 中的 **performance scaling** 时显得如此有效，包括相关的理论和框架。他们引用了与统计估计任务中观察到的 Scaling Laws 相关的特定统计模型。
   - 一位成员建议 *LLM loss 的 scaling laws* 与统计估计中的类似，并指出估计均值的均方误差（mean squared error）按 **N^(-1/2)** 比例缩放。
- **自组织临界性解释**：引入了 **self-organized criticality**（自组织临界性）的概念，断言许多系统会收敛到一个表现出 **power-law fluctuations**（幂律波动）的临界点。这一现象对于理解各个领域中 **critical systems**（临界系统）的行为至关重要。
   - 一位成员强调该概念起源于 Per Bak，并提供了一个指向 Bak 的 [evolution model](https://www.jasss.org/4/4/reviews/bak.html) 的链接以展示该理论。
- **沙堆模型演示临界性**：文中提到了 Bak、Tang 和 Wiesenfeld 研究沙堆模型中 **avalanches**（雪崩）实验的历史引用。他们观察到，当坡度达到临界角时，雪崩的大小分布遵循 **power law**，从而导致在该角度上的收敛。
   - 需要澄清的是，该实验是一个数学模型而非物理装置，旨在捕捉临界点的动力学特征。
- **对幂律证据的怀疑**：针对 **power law scaling** 声明的有效性提出了质疑，指出可能存在许多更简单的解释。此外，还指出在 **log-log plots**（对数-对数图）中仅展示几个 **orders of magnitude**（数量级）作为通用幂律的证据是乏力的。
   - 成员们一致认为，需要跨越 **more orders of magnitude** 的缩放才能令人信服地证明通用幂律正在起作用。



**提及的链接**：<a href="https://www.jasss.org/4/4/reviews/bak.html">Per Bak: How Nature Works: The Science of Self-Organised Criticality</a>：未找到描述

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1281793526648274984)** (12 messages🔥): 

> - `Layer Responsibilities in Models` (模型中的层级职责)
> - `Graph Cluster Detection Probability` (图聚类检测概率)
> - `Residual Stream Differences` (残差流差异)
> - `SAE Latent Activation Variations` (SAE 潜变量激活变化)
> - `Communication Network Protection` (通信网络保护)


- **关于最后一层功能的共识**：似乎存在一种共识，即**最终层 (final layers)** 主要侧重于构建输出的表面形式，据一位成员称，这类似于*运动神经元 (motor neurons)*。
   - 然而，有人指出这一假设尚未得到彻底验证，仍存在不确定性。
- **SAE 重构探索**：一位成员分享了其项目的显著发现，即**中间层 (middle layer)** 的残差流相比最终层表现出显著更低的 SAE 重构损失。
   - 这表明不同层在获取复杂性方面的有效性有所不同，特别是在潜变量激活向量 (latent activation vectors) 的背景下。
- **在高维空间中检测聚类**：一位成员询问了如何推导图中聚类的**检测概率 (detection probability)**，并强调了高维和稀疏性带来的挑战。
   - 回复强调了与信号、噪声和检测算法相关的模型细节对于准确建立检测概率的重要性。
- **理解网络保护策略**：在讨论**通信网络保护**时，一位成员描述了通过特征多样性增强信道安全性的目标。
   - 他们指出其策略建模与混淆 (obfuscation) 的相关性，最近的一篇论文利用合成数据集来评估检测极限。
- **使用图神经网络进行实证测试**：成员们指出，在已知 Ground Truth 的真实或模拟数据上进行实证测试，是评估聚类场景中检测概率的常用方法。
   - 讨论反映了利用图神经网络 (Graph Neural Networks) 进行网络数据建模所涉及的可解释性方面和复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/html/2404.07066v1">Exploring Concept Depth: How Large Language Models Acquire Knowledge at Different Layers?</a>: 未找到描述</li><li><a href="https://github.com/google-research/google-research/blob/master/graph_embedding/simulations/sbm_simulator.py">google-research/graph_embedding/simulations/sbm_simulator.py at master · google-research/google-research</a>: Google Research。在 GitHub 上为 google-research/google-research 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1281956487433814180)** (5 messages): 

> - `Generate Until Tasks Bug` (Generate Until 任务 Bug)
> - `TurkishMMLU Release` (TurkishMMLU 发布)
> - `Community Feedback on Changes` (社区对变更的反馈)


- **Generate Until 任务可能存在 Bug**：一位用户询问在 generate until 任务中省略 'until' 参数是否会默认使用模型的 tokenizer EOS，但观察到它被 fewshot 分隔符覆盖了。
   - 另一位用户确认这似乎是*非预期行为 (unintended behavior)*，并提议修复它或允许他人修复。
- **TurkishMMLU 发布并添加到仓库**：一位成员宣布发布 **TurkishMMLU**，并提供了 [数据集链接](https://huggingface.co/datasets/AYueksel/TurkishMMLU) 和相应的 [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/pull/2283)。
   - 这一贡献旨在增强土耳其语的语言模型评估，详情见[提供的论文](https://arxiv.org/abs/2407.12402)。
- **关于反馈的社区对话**：一位用户就论坛之前的讨论征求 Hailey 的进一步想法。
   - Hailey 确认她已做出回应，表明正在与社区持续互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/543617fef9ba885e87f8db8930fbbff1d4e2ca49/lm_eval/api/task.py#L124">lm-evaluation-harness/lm_eval/api/task.py at 543617fef9ba885e87f8db8930fbbff1d4e2ca49 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2283">Added TurkishMMLU to LM Evaluation Harness by ArdaYueksel · Pull Request #2283 · EleutherAI/lm-evaluation-harness</a>: 在此拉取请求中，我想将我们的工作 TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish 添加到 LM Evaluation Harness。您可以在我们的论文中找到工作的详细信息...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1281691181289439336)** (144 条消息🔥🔥): 

> - `Reflection API 问题`
> - `AI 模型发布中的不专业行为`
> - `自动化 AI 研究`
> - `LLMs 评估`
> - `Hugging Face 社区回应` 


- **Reflection API 因低性能受到审查**：**Reflection 70B 模型**的性能持续受到质疑，有迹象表明它可能是在 **Llama 3.0** 基础上，针对基准测试集训练的 LoRA。多项讨论指出，早期关于顶级性能的宣称具有误导性，且与有缺陷的评估流程有关。
   - 报告显示，最初的私有 API 测试结果优于公开版本，这引发了对不同版本之间明显差异的质疑。
- **对 AI 模型发布实践的担忧**：评论者指出，在没有稳健验证的情况下宣布重大模型突破是**不专业**的表现，并质疑为何有人试图在 AI 能力方面误导社区。有多次提到归因于过高预期和不完整评估的内部失败与疏忽。
   - 成员们对发布中采用的方法表示难以置信，并强调在公开发布声明之前，需要有严格的 AI 模型评估标准。
- **Hugging Face 社区以幽默回应**：鉴于 Reflection API 的溃败，**Hugging Face 社区**的成员分享了对该情况的幽默看法，强调了其平台与所发布的模型相比的可靠性。一些 HF 员工开玩笑说上传大型模型非常容易，暗示令人沮丧的体验在他们的平台上并不常见。
   - 这种轻松的批评反映了关于 AI 模型评估和发布中社区标准的更广泛情绪。
- **LLM 生成的研究想法的新颖性**：一项新研究声称，从统计学上看，**LLM 生成的想法**比人类专家研究员产生的想法更具新颖性，这引发了关于 AI 在创意领域有效性的讨论。然而，在评估这些说法时，也考虑了评审员对现有文献的认知等干扰因素。
   - 对研究领域局限于“基于 prompting”领域的担忧表明，研究结果可能无法反映其他领域的普遍适用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/mattshumer_/status/1832554497408700466">来自 Matt Shumer (@mattshumer_) 的推文</a>：快速更新 —— 我们重新上传了权重，但仍然存在问题。我们刚刚重新开始训练，以消除任何可能的问题。应该很快就会完成。对此深表歉意。...的数量...</li><li><a href="https://x.com/shinboson/status/1832933747529834747">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson)</a>：一个关于 AI 研究社区造假的故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向全世界宣布，他们取得了一项突破，使他们能够训练一个中型模型...</li><li><a href="https://fxtwitter.com/mattshumer_/status/1832554497408700466">来自 Matt Shumer (@mattshumer_) 的推文</a>：快速更新 —— 我们重新上传了权重，但仍然存在问题。我们刚刚重新开始训练，以消除任何可能的问题。应该很快就会完成。对此深表歉意。...的数量...</li><li><a href="https://x.com/mattshumer_/status/1832556398854746371">来自 Matt Shumer (@mattshumer_) 的推文</a>：@JacquesThibs 我们本不应该这样做，但我们几乎尝试了一切，无论我们做什么，HF 上的模型都存在问题。性能远未达到我们在本地看到或应该看到的水平。</li><li><a href="https://x.com/ChengleiSi/status/1833166031134806330">来自 CLS (@ChengleiSi)</a>：自动化 AI 研究令人兴奋！但 LLM 真的能产生新颖的、专家级的研究想法吗？经过为期一年的研究，我们得出了第一个具有统计学意义的结论：LLM 生成的...</li><li><a href="https://x.com/mattshumer_/status/1832424499054309804?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>：我们已经找到了问题所在。Hugging Face 上的 Reflection 权重实际上是几个不同模型的混合 —— 在上传过程中出了一些差错。今天会修复。引用 Matt Shu...</li><li><a href="https://x.com/goodside/status/1828329834256232770?s=46">来自 Riley Goodside (@goodside)</a>：@TheXeophon 同意 —— 对于大型多步 Prompt 流水线来说，一个更公平的基准应该是扩展到相同推理预算的通用 Self-consistency。</li><li><a href="https://huggingface.co/mattshumer/ref_70_e3">mattshumer/ref_70_e3 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1832449234756628781?s=46">来自 Teknium (e/λ) (@Teknium1)</a>：@terryyuezhuo 这太让人困惑了，笑死，仓库里在随机更新 README 的间隙，零散地上传了一些单一模型的部分……即使在 mergeland 我也没见过这种操作</li><li><a href="https://fxtwitter.com/kalomaze/status/1833151794651808202">来自 kalomaze (@kalomaze)</a>：🤔</li><li><a href="https://x.com/thexeophon/status/1828313998460363140?s=46">来自 Xeophon (@TheXeophon)</a>：那些将自己复杂的 Prompt 设置与单次运行的 CoT Prompt 进行比较的论文，从根本上说是不严肃的。至少应该多次运行 CoT Prompt，你会惊讶于它的效果...</li><li><a href="https://x.com/osanseviero/status/1833045419896746282">来自 Omar Sanseviero (@osanseviero)</a>：这是一个关于如何将 70B+ 模型上传到 Hugging Face 的分步指南：
第一步：`pip install huggingface_hub`
第二步：`huggingface-cli upload-large-folder <repo-id> <local-path> --repo-ty...`</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fb6jdy/reflectionllama3170b_is_actually_llama3/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://x.com/RealJosephus/status/1832904398831280448">来自 Joseph (@RealJosephus)</a>：“Reflection API”是一个带有 Prompt 的 Sonnet 3.5 封装。他们目前正通过过滤掉“claude”字符串来伪装它。https://www.reddit.com/r/LocalLLaMA/comments/1fc98fu/c...</li><li><a href="https://x.com/artificialanlys/status/1832965630472995220?s=46>">来自 Artificial Analysis (@ArtificialAnlys)</a>：Reflection 70B 更新：从我们的角度对时间线和未决问题的简要说明。时间线：- 我们测试了最初发布的 Reflection 70B，发现性能比 Llama 3.1 70B 更差。- ...</li><li><a href="https://fxtwitter.com/Yuchenj_UW/status/1832865464827204065">来自 Yuchen Jin (@Yuchenj_UW)</a>：Reflection Llama 3.1 70B 更新：@mattshumer_ 及其团队在 Hugging Face 上发布了“新的、可运行的 Reflection Llama 3.1 70B 模型版本”，所以我们现在正在提供新权重的服务...</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1dx6025/claude_has_a_moral_crisis_when_jailbreak_leaks/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/09/05/the-ai-industry-is-obsessed-with-chatbot-arena-but-it-might-not-be-the-best-benchmark/">AI 行业痴迷于 Chatbot Arena，但它可能不是最好的基准测试 | TechCrunch</a>：LMSYS 的 Chatbot Arena 可能是当今最受欢迎的 AI 基准测试 —— 也是行业的痴迷所在</li>

ssion. 但它远非完美的衡量标准。</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1fc98fu/confirmed_reflection_70bs_official_api_is_sonnet/">已确认：REFLECTION 70B 的官方 API 是 SONNET 3.5</a>：由 u/TGSCrust 发布在 r/LocalLLaMA • 1,043 点赞和 303 条评论
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1282079542713913406)** (3 messages): 

> - `GPT Next`
> - `KDDI Summit Presentation` 


- **OpenAI 澄清 GPT Next 混淆**：在 Tadao Nagasaki 于 KDDI 峰会发表演讲后，围绕名为 **GPT Next** 的新模型出现了猜测；然而，OpenAI 发言人确认这仅仅是一个代表模型未来演进的**象征性占位符**。
   - 发言人强调，幻灯片中的图形表示是**说明性的**，而非即将发布的**时间表**。
- **Nagasaki 强调 AI 增长潜力**：Nagasaki 表示，根据过往表现，标记为 'GPT Next' 的未来 AI 模型预计将进化**近 100 倍**，突显了 AI 技术的**指数级增长**。
   - 他将其与传统软件开发进行了对比，指出 **AI 技术呈指数级增长**，正如 **ITmedia** 所报道的那样。



**提到的链接**：<a href="https://mashable.com/article/openai-clarifies-no-gpt-next-not-a-new-model">OpenAI 澄清：不，“GPT Next”不是一个新模型。</a>：来自演示文稿的混淆让 OpenAI 的粉丝们陷入了不安。

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1282728216858984481)** (12 messages🔥): 

> - `OpenAI team dynamics`
> - `Google's recent activity`
> - `System prompts focus` 


- **OpenAI 向 Anthropic 的转型**：成员们讨论了 **OpenAI** 转型的超现实性，特别是提到了联合创始人 **John Schulman** 现在已加入 **Anthropic**。
   - *“你能写多少次 '来自 OpenAI 的 XY（现在在 Anthropic）'？”* 这是一个突显这一变化的轻松调侃。
- **Anthropic 和 OpenAI 的社区氛围**：情绪出现了分歧，一位成员形容 **Anthropic** 氛围良好，而 **OpenAI** 则被认为氛围复杂。
   - 有人对 OpenAI 框架内调整所需的时间表示担忧，反映了当前的社区情绪。
- **关于模型规范和提示词的辩论**：围绕公开关注 **system** 和 **dev prompts** 展开了对话，质疑规范是否需要层级结构。
   - 一位成员思考了没有层级结构的规范的有效性，展示了关于 prompt 结构的讨论。
- **Google 觉醒的传闻**：一位成员指出 **Google 正在觉醒**，暗示了 AI 领域潜在的新兴竞争和令人兴奋的发展。
   - 这一说法引起了笑声，表明了对 Google 战略举措持续存在的戏谑性怀疑。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1281809698831466569)** (2 messages): 

> - `Internal bureaucracy at Google`
> - `Challenges of scaling within large organizations` 


- **Google 的官僚主义负担**：一位前 Googler 表示对 Google **庞大的官僚机构**感到不知所措，理由是内部利益相关者和流程过多。
   - *那里能发布任何东西简直是个奇迹*，因为员工经常发现自己忙于应对内部力量，而无法专注于大局。
- **应对内部力量**：这位前 Googler 指出，忙于内部流程几乎没有给长期愿景和创新留出空间。
   - 这种情绪突显了大型组织中员工面临的挑战，**内部政治**可能会扼杀生产力。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1281697036604932138)** (47 messages🔥): 

> - `AI Codex for Cursor`
> - `Reflection API`
> - `Apple Intelligence Updates`
> - `Gemini Enum Mode`
> - `Photorealistic LoRA Model`

- **AI Codex 增强了 Cursor 的功能**：为 @cursor_ai 推出的全新 [AI Codex](http://codex.md) 提供了一个自我改进系统，具有自动保存见解和智能分类等功能。
   - 一位用户建议，使用 AI Codex 一个月可能会揭示出有价值的学习成果。
- **Reflection API 引发争议**：据多个来源报道，新发现的 *Reflection API* 被指是 Sonnet 3.5 的封装（wrapper），据称它过滤掉了对 Claude 的引用以掩盖其本质。
   - 各项评估发现，该 API 的表现可能不如之前声称的那样出色，从而引发了关于此类性能基准测试背后方法论的讨论。
- **Apple 发布重大 AI 进展**：在最近的 Apple 发布会上，Apple Intelligence 的更新暗示了值得关注的进步，包括可能改进的 Siri 以及领先于竞争对手的 AI 手机。
   - 这些进展引发了人们对 AI 部署影响的兴奋，并促使人们呼吁 Apple 工程师分享见解。
- **Gemini API 引入 Enum Mode**：Logan K 宣布在 Gemini API 中发布全新的 Enum Mode，允许从预定义的输出选项中进行选择，增强了结构化输出能力。
   - 这一新增功能旨在简化使用 Gemini 框架的开发者的决策过程。
- **创新的写实 LoRA 模型出现**：一位用户重点介绍了一个*令人惊叹的写实 LoRA*，它在 Stable Diffusion 社区引起了关注，并通过各种图像展示了其能力。
   - 围绕该模型性能及其意外包含的动漫图像的讨论引起了社区的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ArtificialAnlys/status/1832457791010959539">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Reflection Llama 3.1 70B 独立评估结果：我们在独立测试中无法复现其声称的评估结果，且观察到的性能优于 Meta 的 Llama 3.1 70B，而不是...</li><li><a href="https://x.com/zbeyens/status/1832079140083687671?s=46">来自 Ziad Beyens (@zbeyens) 的推文</a>：介绍 AI Codex：为 @cursor_ai 打造的自我改进系统。◆ http://codex.md：错误与学习仓库。◆ http://learn.md：自动保存新见解。◆ http://split-codex.md：智能分类...</li><li><a href="https://x.com/RealJosephus/status/1832904398831280448">来自 Joseph (@RealJosephus) 的推文</a>：“Reflection API” 是一个带有 Prompt 的 Sonnet 3.5 封装。他们目前正通过过滤字符串 “claude” 来进行伪装。https://www.reddit.com/r/LocalLLaMA/comments/1fc98fu/c...</li><li><a href="http://llmagents-learning.org/f24">Large Language Model Agents</a>：未找到描述</li><li><a href="https://x.com/clementdelangue/status/1833136159209263552?s=46">来自 clem 🤗 (@ClementDelangue) 的推文</a>：正如我们每天所见，评估是 AI 中最重要的步骤之一——如果不是最重要的那一个。我们不仅需要改进通用 Benchmarking，还应该...</li><li><a href="https://x.com/swyx/status/1833231875537850659">来自 swyx 🇸🇬 (@swyx) 的推文</a>：哇。Apple 可能刚刚修复了 Siri。并在第一款 AI 手机上击败了 OpenAI。还通过 Google 将 OpenAI 商品化。并随手发布了一个视频理解模型。执行力极佳。（见...</li><li><a href="https://news.ycombinator.com/item?id=41492172">未找到标题</a>：未找到描述</li><li><a href="https://x.com/swyx/status/1832138164951249104">来自 swyx 🇸🇬 (@swyx) 的推文</a>：Diffusion Transformers 非常出色，但在我们等待 Sora 的同时，我喜欢 @toinfinityai 的方法——将使用场景严格限制在视频同步（不仅是口型同步）上——并以此为起点。</li><li><a href="https://x.com/OfficialLoganK/status/1833226001670934827">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：我们刚刚在 Gemini API 中发布了一种名为 Enum Mode 的 Structured Outputs 新变体，它允许你轻松地约束模型在预定义选项中进行选择 🚢</li><li><a href="https://github.com/udecode/dotai/blob/main/codex/learn.md">dotai/codex/learn.md at main · udecode/dotai</a>：通过在 GitHub 上创建账号来为 udecode/dotai 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=uarNiSl_uh4">Apple Event - 9 月 9 日</a>：观看 Apple 特别活动，了解下一代 iPhone、Apple Watch 和 AirPods 等更多内容。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1fak0jl/comment/lltkdun/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/6p1QVJCAYe">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。</a>：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。
</li>
</ul>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1281705587557666910)** (76 条消息🔥🔥): 

> - `开源 AI 代码编辑器`
> - `协作工具`
> - `代码中的错误处理`
> - `使用 Loras 进行微调`
> - `Zed 对比 Cursor` 


- **探索开源 AI 代码编辑器**：成员们讨论了各种 **开源 AI 代码编辑器**，如 [Melty](https://github.com/meltylabs/melty) 和 [PearAI](https://github.com/trypear/pearai-app)，作为 Cursor 的替代方案。
   - 一位成员建议在每个工具上花些时间，以评估它们的功能和可用性。
- **高效处理代码错误**：一位成员指出，处理编码中的 **非理想路径场景 (non-happy-path scenarios)** 是区分工程与简单原型设计的关键。
   - 另一位用户提到，他们的 **理想路径代码 (happy path code)** 仅占总代码量的约 **10%**，突显了错误管理的重要性。
- **Zed 代码编辑器趋势**：讨论了 **Zed 编辑器** 的功能，成员们对其新的 Linux 版本表示赞赏，但注意到缺乏位图字体支持。
   - 用户们对其在 AI 和人类程序员之间进行 **高性能协作** 的潜力表示热忱。
- **Aider 在代码编辑方面的优势**：成员们强调了 **Aider 工具** 有效的代码编辑能力，并展示了评估各种 LLM 编辑技能的排行榜。
   - 提到 **Claude 3.5 Sonnet** 被公认为代码编辑能力表现最好的模型之一。
- **使用 Loras 进行微调**：一位用户表示有兴趣在接下来的讨论中涵盖用于量化的 **使用 Loras 进行微调**，预示着潜在的社区学习。
   - 另一位成员询问重点是图像模型还是语言模型，表明了应用方向的分歧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑技能的定量基准。</li><li><a href="https://zed.dev/">Zed - The editor for what&#x27;s next</a>：Zed 是一款高性能、多用户协作的代码编辑器，由 Atom 和 Tree-sitter 的创建者开发。</li><li><a href="https://github.com/meltylabs/melty">GitHub - meltylabs/melty: Open source AI code editor. To download the packaged app:</a>：开源 AI 代码编辑器。下载打包后的应用：- meltylabs/melty</li><li><a href="https://github.com/trypear/pearai-app">GitHub - trypear/pearai-app: The Open Source AI-Powered Code Editor. A fork of VSCode and Continue.</a>：开源 AI 驱动的代码编辑器。VSCode 和 Continue 的分叉版本。- trypear/pearai-app</li><li><a href="https://github.com/go-go-golems/go-go-labs/tree/main/cmd/apps/catter">go-go-labs/cmd/apps/catter at main · go-go-golems/go-go-labs</a>：GO GO 实验实验室。通过在 GitHub 上创建账户为 go-go-golems/go-go-labs 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>：未找到描述</li><li><a href="https://github.com/MikeBirdTech/ai-toolkit">GitHub - MikeBirdTech/ai-toolkit: A collection of community created AI tools to improve your life</a>：一系列由社区创建的旨在改善生活的 AI 工具 - MikeBirdTech/ai-toolkit
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1281690972891381910)** (38 条消息🔥): 

> - `OpenInterpreter 性能`
> - `OpenInterpreter 上的 AI 技能 (Skills)`
> - `01 iOS 应用功能`
> - `结合 LLM 使用 OpenInterpreter`
> - `联系风险投资人 (Venture Capitalists)` 


- **OpenInterpreter 在资源管理方面存在困难**：用户报告称，虽然 **01** 应用可以快速访问并播放音频文件，但其在 **Mac** 上的性能可能会下降，导致结果不一致。
   - *一位用户提到，由于 01 在其设备上的稳定性问题，他们更倾向于使用原生的 OI。*
- **AI 技能 (Skills) 开发讨论**：用户提出了关于 **skills** 何时能用于标准版 OpenInterpreter 而不仅仅是 **01** 应用的问题，突显了用户对改进功能的偏好。
   - *一位用户对 01 应用相对于原生 OI 的性能表现表达了沮丧。*
- **探索 01 iOS 应用的能力**：**01** iOS 应用旨在通过语音命令无缝控制电脑和智能家居，具备文件管理和智能设备集成等功能。
   - *用户注意到该应用兼容 iPadOS，确认了其跨设备的可用性。*
- **使用 OpenInterpreter 创建自定义 LLM**：讨论集中在 **OpenInterpreter** 与 **LLMs** 通信并可能创建自定义模型的潜力上，并鼓励用户尝试微调 (fine-tuning)。
   - *一位用户对在即将举行的 LLM 工作坊中使用 OpenInterpreter 的可能性感到兴奋。*
- **寻求融资指导**：一位用户询问如何为其 AI 应用联系 **Venture Capitalists**，并表示愿意以合适的价格出售。
   - *社区参与了关于融资机会的指导和引荐。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://apps.apple.com/ca/app/01-light/id6601937732">‎01 Light</a>: ‎随时随地通过语音命令控制您的电脑和智能家居。01 连接到您家中的服务器，实现对文件、应用和 IoT 设备的远程访问。能力：...</li><li><a href="https://suno.com/song/a15352a6-7aa1-41db-9c2e-62a668df74ff">01 by @techfren | Suno</a>: deep house 氛围电子乐。使用 Suno 聆听并创作属于你自己的音乐。</li><li><a href="https://x.com/hellokillian/status/1803090274186617188">来自 killian (@hellokillian) 的推文</a>: Open Interpreter 的 Local III 今日发布。我们正在构建可以离线工作的电脑控制 Agent。这是我们迈出的最大一步。- interpreter --local 可设置快速的本地 LLM。- 我们正在...</li><li><a href="https://tenor.com/view/ooh-despicable-me-4-surprised-uh-oh-that%27s-gotta-hurt-gif-14253073070740964952">Ooh Despicable Me 4 GIF - Ooh Despicable me 4 Surprised - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/dbc52593e608d3ce3d25a0eece4e84cf57bb7892/interpreter/core/computer/skills/skills.py">open-interpreter/interpreter/core/computer/skills/skills.py (GitHub)</a>: 电脑的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://pastebin.com/kWpxhx31">from interpreter import AsyncInterpreter... - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/Nchn0jV7">  import os - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1281749056653234248)** (54 messages🔥): 

> - `Torch 安装问题`
> - `01 Light 停产`
> - `01 的退款流程`
> - `01 app 发布详情`
> - `使用 OpenInterpreter` 


- **Torch 安装困境**：许多用户在使用 Poetry 安装 **Torch 2.3.1** 时遇到问题，导致出现提示没有安装候选者的 **RuntimeError**。一位用户分享说，切换 Python 版本甚至更新他们的 VS Code 似乎为他们解决了这个问题。
   - *Ohhhhh boy have I had that same problem* 描述了他们反复尝试修复该问题的过程。
- **01 Light 停产**：**01 Light** 已正式停产，团队宣布将退还所有硬件订单的款项，同时发布一款免费的 01 app。这一决定旨在让软件团队能够专注于推进其平台，而不损害软件功能。
   - 用户对停产表示失望，尤其是那些一直热切等待设备的用户。
- **01 硬件的退款流程**：用户询问了 **01 Light** 的退款政策，并得到保证，退款正在通过发送邮件至 help@openinterpreter.com 进行处理。一些用户担心如果使用礼品卡购买是否能收到退款。
   - Mikebirdtech 确认可以退款，并表示：*Now worries, you'll get your money back*（别担心，你会拿回你的钱）。
- **01 App 的发布**：团队宣布发布一款免费的 **01 app**，并表示它保留了 01 Light 的所有功能。他们鼓励用户尽管硬件设备停产，也要尝试使用该 app。
   - 创意性的回应承认智能手机可以执行类似的功能，这使得硬件停产变得不那么关键。
- **在不同平台上运行 OpenInterpreter**：一些用户询问在 **iOS** 和 **Windows** 上运行该 app 的情况，表现出对跨平台兼容性的兴趣。关于 Poetry 配置的问题被提出，特别是当缺少 `pyproject.toml` 文件时。
   - 用户在处理管理虚拟环境和运行命令的复杂问题时提供了建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hellokillian/status/1833215071880941972">来自 killian (@hellokillian) 的推文</a>：今天我们停产了 01 light，为所有人退款，并发布了一款免费的 01 app。我们还开源了所有的制造材料以及一个重大的 01.1 更新。为什么？为了专注。这个软件...</li><li><a href="https://changes.openinterpreter.com/log/01-app">Open Interpreter - 它本该是一个 app</a>：开源 Open Interpreter 项目的官方变更日志。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1282314570836475956)** (5 messages): 

> - `Scriptomatic 与开源模型`
> - `Instructor Python 库` 


- **Scriptomatic 与开源模型集成**：一位成员报告成功让 **Scriptomatic** 与开源模型的结构化输出协同工作，并表示很快将提交 PR。
   - 他们对为 **Dspy** 提供的框架表示感谢，并提到他们的处理过程涉及大量的 *grepping and printing*。
- **Instructor 库增强 LLM 输出**：一条消息分享了 [Instructor](https://pypi.org/project/instructor/) 库的链接，该库使用基于 Pydantic 的用户友好型 API 简化了处理 LLM 结构化输出的工作。
   - 对于希望改进 LLM 工作流的用户，*Instructor* 承诺简化验证、重试和流式响应。
- **Scriptomatic 的 YouTube 资源**：一位成员发布了一个 [YouTube 视频](https://www.youtube.com/watch?v=XkDSQq0fwfU) 链接，该视频在关于 Scriptomatic 的持续讨论中对他们很有帮助。
   - 该资源似乎旨在帮助他人浏览频道中讨论的工具。



**提到的链接**：<a href="https://pypi.org/project/instructor/">instructor</a>：LLM 的结构化输出

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1281692120675254365)** (9 messages🔥): 

> - `Agentic System Deployment` (Agentic 系统部署)
> - `Running Reflection 70B` (运行 Reflection 70B)
> - `Advanced RAG Pipelines` (高级 RAG 流水线)
> - `Automating Financial Analysis` (自动化财务分析)
> - `Dynamic ETL for RAG` (用于 RAG 的动态 ETL)


- **使用 llama-deploy 部署 Agentic 系统**：如果你正在寻找将 Agentic 系统部署为微服务的端到端示例，请查看这个使用 @getreflex 和 LlamaIndex 的[全栈示例](https://t.co/jL14R8cJMD)。
   - 它显著简化了流程，非常适合希望优化其聊天机器人系统的开发者。
- **轻松运行 Reflection 70B**：如果你的笔记本电脑配置允许，现在可以按照[这里](https://t.co/ZkF05l159I)提到的方法，直接通过 LlamaIndex 使用 Ollama 运行 **Reflection 70B**。
   - 这使得开发者无需庞大的基础设施即可实验这一先进模型。
- **构建高级 RAG 流水线**：有一份关于使用 Amazon Bedrock 构建高级 Agentic RAG 流水线的指南，其中包括[动态查询路由](https://t.co/mzJzDMGhM2)和 top-k 向量搜索。
   - 这篇全面的教程涵盖了优化 RAG 实现所需的一切。
- **通过 Agentic 工作流自动化财务分析**：这篇博客文章讨论了如何构建一个 Agentic 摘要系统来自动化季度和年度财务分析，从而有效地汇总结果（[阅读更多](https://t.co/ktj55fQSlZ)）。
   - 这种自动化可以极大地提高财务报告和决策的效率。
- **使用 LLM 进行动态 ETL**：LLM 可以根据数据特性做出决策，在 RAG 环境中自动化 ETL 过程，而不是使用固定分块（fixed chunking），如本教程所示（[链接](https://t.co/6yZmHoUjCW)）。
   - 这种方法简化了数据提取和过滤，能够适应不同数据集的特征。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1281751087778627726)** (51 messages🔥): 

> - `Cohere Reranker`
> - `LlamaIndex Node Postprocessors`
> - `Llama Parse Service Status`
> - `LlamaIndex Structured Outputs`
> - `Using Llama 3 with LlamaIndex` 


- **在 Azure 上使用 Cohere Reranker**：一位用户询问如何在 Azure AI studio 中将 Cohere reranker 用作 node postprocessor，并提到了现有导入可能存在的问题。
   - 另一位成员确认 Azure 目前没有专门的 rerank 模块，但提到创建一个是可行的，因为基类很简单。
- **理解 LlamaIndex Workflows**：一位成员询问在 LlamaIndex workflows 中通过 Context 传递数据与设置实例属性之间的区别，寻求关于跨运行持久性的澄清。
   - 解释指出，为了提高模块化程度，Context 在嵌套的 workflows 之间不共享，而在 'self' 上设置属性可以在多次运行中保留数据。
- **Llama Parse 服务状态更新**：用户对 Llama Parse 服务的运行状态表示关注，促使参与者提供了更新和当前状态指示。
   - 根据最新更新，该服务似乎已恢复在线，但由于处理积压，仍显示出一些性能下降。
- **LlamaIndex 中的 Structured Outputs 支持**：一位用户询问 LlamaIndex 是否支持 OpenAI 的 structured outputs，确认已支持并提供了具体的使用说明。
   - 此外，还分享了文档链接，说明如何使用 LlamaIndex 和 OpenAI 实现 structured prediction。
- **LlamaIndex 使用 Llama 3 的示例**：一位用户寻求使用 LlamaIndex 而非 OpenAI 来运行 Llama 3 的示例，反映了对该集成资源的开发需求。
   - 一位成员引导他们查看相关文档，详细介绍了如何有效地配置和使用 Llama 3 与 LlamaIndex。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/">Ollama - Llama 3.1 - LlamaIndex</a>: 未找到描述</li><li><a href="https://llamaindex.statuspage.io/">LlamaIndex Status</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/">Self-querying | 🦜️🔗 LangChain</a>: 前往 Integrations 查看具有内置 self-querying 支持的 vector stores 文档。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#structured-prediction">OpenAI - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1281828815336640522)** (25 条消息🔥): 

> - `Gemma 模型配置`
> - `对 Gemma 2 的支持`
> - `Torchtune 调整的 PR`
> - `Tokenizer eos 问题` 


- **Gemma 模型配置更新**：为了使用 **Torchtune** 配置 **Gemma 9B** 模型，一名成员建议将配置中的 `model` 条目替换为特定参数，包括 `vocab_size`、`num_layers` 等。
   - 这利用了 **Gemma** 的底层组件构建器，旨在根据 [config.json](https://huggingface.co/google/gemma-2-9b/blob/main/config.json) 中的值实现模型尺寸的通用性。
- **Gemma 2 的支持挑战**：讨论强调了在 **Torchtune** 中支持 **Gemma 2** 的障碍，主要归因于 **logit-softcapping** 和带宽问题。
   - 有人指出，**Gemma 2** 架构的增强功能尚未被正式请求，这增加了待实现功能的积压。
- **Torchtune 改进的 PR 提案**：一名成员发现了 **Torchtune** 中关于填充序列（padding sequence）行为的一个潜在 Bug，并提出了一个 **PR** 来修复它。
   - 他们建议修改 flip 方法以提高清晰度，并旨在确保与 **torch pad_sequence** 的功能对等。
- **数据集返回类型需要澄清**：有人对 **Torchtune** 中 **ConcatDataset** 实现中具有误导性的返回类型表示担忧，这可能需要为所有数据集定义一个一致的类型。
   - 讨论还提到，虽然 **Torchtune** 不支持负索引，但这一决定的背后原因受到了质疑。
- **Mistral 和 Gemma 中的 Tokenizer Eos 问题**：一名成员提出提交一个 **PR** 来解决 **eos** token 问题，但指出目前的 **Mistral** 和 **Gemma** tokenizer 缺少 `add_eos` 选项。
   - 这突显了 tokenizer 功能的局限性，可能会影响依赖序列结束（end-of-sequence）token 的实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/test/nn/test_packed_sequence.py#L190">pytorch/pytorch 主分支下的 pytorch/test/nn/test_packed_sequence.py</a>：Python 中具有强 GPU 加速的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/torchtune/">GitHub - pytorch/torchtune: 一个用于 LLM 微调的原生 PyTorch 库</a>：一个用于 LLM 微调的原生 PyTorch 库。欢迎在 GitHub 上为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/data/_collate.py#L52">pytorch/torchtune 主分支下的 torchtune/torchtune/data/_collate.py</a>：一个用于 LLM 微调的原生 PyTorch 库。欢迎在 GitHub 上为 pytorch/torchtune 做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_concat.py#L77">pytorch/torchtune 主分支下的 torchtune/torchtune/datasets/_concat.py</a>：一个用于 LLM 微调的原生 PyTorch 库。欢迎在 GitHub 上为 pytorch/torchtune 做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1282042802296852554)** (32 条消息🔥): 

> - `Compiling Generation Methods` (编译生成方法)
> - `Cache Handling During Generation` (生成过程中的缓存处理)
> - `Handling Non-Contiguous Inputs` (处理非连续输入)
> - `Tensor.is_inference() Method Proposal` (Tensor.is_inference() 方法提案)
> - `Proposed Implementation of Chunked Linear + CE` (分块线性 + 交叉熵的建议实现)


- **为了提速而编译生成方法**：用户旨在利用 `torch.compile` 处理 `generate_next_token` 以提升生成速度，类似于他们之前在 PPO 损失步骤中的成功经验。
   - 然而，他们报告没有达到预期的加速，可能是由于激活检查点 (activation checkpointing) 和非连续输入等问题。
- **生成过程中的缓存处理讨论**：讨论围绕 Attention 模块中连续前向调用 (forward calls) 的需求展开，这些调用需要根据生成过程中的缓存状态表现出不同的行为。
   - 他们建议使用 `torch.inference_mode`，但也承认向 `.forward()` 传递显式标志可能是更好的方法。
- **提议 Tensor.is_inference() 方法**：用户提议实现 `Tensor.is_inference()` 方法，以便更好地管理跨多个前向调用的缓存行为。
   - 尽管对此感兴趣，但他们担心将此更改集成到现有维护者工作流中的挑战。
- **关于属性实现的担忧**：有建议向模型添加一个切换属性 (toggle attribute)，用于在不修改 `.forward()` 签名的情况下检查缓存行为。
   - 有人担心在修改非 Tensor 模块属性时，`torch.compile` 可能会出现潜在问题。
- **分块线性 (Chunked Linear) + 交叉熵 (CE) 的简洁实现**：一位成员引用了来自 GitHub gist 的分块线性结合交叉熵的简洁实现，作为关注点。
   - 他们指出，由于 torchtune 将 LM-head 和损失计算分离，将类似方法集成到 torchtune 中可能会很困难。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/Chillee/22cd93e11b887db1f596ab754d60a899">chunked_lce.py</a>: chunked_lce.py。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/pytorch/pull/124927/files">[dynamo] Add support for tensor&#39;s is_complex method by YangQun1 · Pull Request #124927 · pytorch/pytorch</a>: 此 PR 旨在 dynamo 中添加对 tensor 的 is_complex 方法的支持。以下代码为例：    def test_tensor_is_complex(x):         if x.is_complex():             return x + 1...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1281716381955395655)** (41 条消息🔥): 

> - `Decoding .astream_events()` (解码 .astream_events())
> - `Gradio Upload Limitations` (Gradio 上传限制)
> - `LangChain Azure Integration` (LangChain Azure 集成)
> - `Data Set Creation Strategies` (数据集创建策略)
> - `Audio Transcription with Claude` (使用 Claude 进行音频转录) 


- **解码 .astream_events() 的困扰**：用户在解码来自 **.astream_events()** 的流时遇到挑战，其中一位提到通过所有分支和事件类型进行手动序列化非常繁琐。
   - 一位参与者询问如何寻找参考实现，强调了该主题缺乏优质资源。
- **Gradio 并发问题**：一位用户注意到，在打开 10 个标签页启动 **Gradio** 后，只有 6 个请求开始生成，这表明尽管设置了更高的并发限制，但仍存在限制。
   - 尽管 Token 速率很高，但硬件似乎无法处理超过 6 个并发请求，这表明可能存在配置或限制问题。
- **排查 Azure OpenAI 集成问题**：一位用户报告在尝试与 Azure OpenAI 交互时遇到 **500 错误**，寻求关于参数和可能端点问题的建议。
   - 另一位成员指出，验证环境变量和命名规范（特别是围绕端点），可能会解决这些问题。
- **从不同文档创建数据集**：一位用户询问是应该为不同的文档集构建独立的数据集，还是在一个数据集中将输入文本与相应的文档一起保存。
   - 这突显了数据集创建中关于效率和组织的一个常见困境。
- **探索 Claude 的音频处理能力**：关于是否可以使用 LangChain 将音频数据传递给 **Claude 3.5 LLM** 进行转录的讨论引起了一些兴趣。
   - 参与者指出，虽然 Claude 支持图像输入，但音频功能尚不确定。


  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1282290827271602236)** (9 条消息🔥): 

> - `VAKX 平台`
> - `Selenium 与 GPT-4 vision 集成`
> - `AI Reddit Manager 工具`
> - `Mocking LLM embedder`
> - `使用 OpenAI 和 LangChain 的 RAG 聊天机器人` 


- **VAKX：您的无代码助手构建器**：一位成员介绍了 **VAKX**，这是一个无代码 LLM 编排平台，使用户能够快速构建 AI 助手。他们邀请大家提供反馈，并提供了进一步探索该平台的链接：[VAKX](https://vakx.io) 和 [免费开始构建](https://studio.vakx.io)。
   - 他们强调了诸如 **VAKChat** 集成等功能，用于为网站添加 AI 驱动的聊天，并概述了吸引用户的简单设置步骤。
- **Selenium 遇见 GPT-4 Vision**：一位成员分享了他们集成 **Selenium** 和 **GPT-4 vision model** 的实验性项目，并在 [此 YouTube 视频](https://youtu.be/nTtZnzYS_24) 中详细介绍了集成过程。他们还提供了包含代码的 GitHub 仓库链接：[GitHub Repository](https://github.com/rajib76/browser_agent)。
   - 随后讨论了这种集成的目的，重点在于使用向量数据库进行集成测试的好处，而不是使用实时的 embedding 模型。
- **使用 AI Reddit Manager 创建帖子**：一位成员展示了他们的 **AI Reddit Manager**，该工具使用 Lyzr Agent API 和 Streamlit 自动策划并发布内容到 subreddits。他们的目标是通过根据特定主题生成帖子来节省时间，尽管他们链接的 Medium 文章目前是一个失效链接。
   - 他们提供了一个 YouTube 链接来演示其工具的功能：[YouTube Video](https://www.youtube.com/watch?v=2H7etaeSWgA)。
- **Mocking LLM Embedder 指南**：一位成员撰写了一篇关于如何为 **MongoDB Atlas** 的集成测试 Mocking **LLM embedder** 的指南，可在 [此处](https://dev.to/prestonvasquez/mocking-an-llm-embedder-targeting-mongodb-atlas-1glp) 查看。他们谈到了在集成过程中使用实时 embedding 模型时面临的挑战。
   - 讨论包括澄清这项工作的目标是集成测试，而不是专注于 embedding 模型本身，从而促进与 **LangChainGo** 的集成。
- **拥抱 OpenAI 和 LangChain 的 RAG 聊天机器人**：一位成员介绍了他们利用 **OpenAI** 和 **LangChain** 构建的 **RAG chatbot**，用户可在 [AdaletGPT](https://adaletgpt.com) 访问。他们鼓励成员在需要时寻求帮助。
   - 该聊天机器人代表了近期 AI 进展在参与性对话和交互方面的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dev.to/prestonvasquez/mocking-an-llm-embedder-targeting-mongodb-atlas-1glp">未找到标题</a>: 未找到描述</li><li><a href="https://adaletgpt.com">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/nTtZnzYS_24">集成 Selenium 和 gpt-4 vision</a>: 在此录音中，我展示了一个将 GPT4 vision 模型与 selenium 集成的用例。代码：https://github.com/rajib76/browser_agent</li><li><a href="https://medium.com/@harshit_56733/step-by-step-guide-to-build-an-ai-powered-reddit-manager-that-curates-relevant-content-for-daily-2434cd965509)">未找到标题</a>: 未找到描述</li><li><a href="https://vakx.io">VAKX | 为您的文档提供 AI 驱动的辅助</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1281713597558296577)** (33 messages🔥): 

> - `模型中的过拟合 (Overfitting)`
> - `基准测试 (Benchmark) 的局限性`
> - `AI 工具中的骗局`
> - `RAG APIs` 


- **训练过程中的过拟合 (Overfitting)**：一名成员对 **Overfitting** 提出了担忧，强调 **Benchmarks** 可能会产生误导，且模型无论规模大小 **都会** 发生过拟合。
   - *“我不再相信基准测试了”* 反映了对在不充分数据上评估模型可靠性的怀疑。
- **承认基准测试 (Benchmark) 的局限性**：一位成员分享了关于 **Benchmark 局限性** 的见解，指出虽然 **Benchmarks** 通常存在缺陷，但它们仍是少数可用的比较工具之一。
   - 他们希望关于 **Benchmark 问题** 的文章能被 NeurIPS 接收，以揭示当前评估方法中的挑战。
- **新型 AI 工具被证实为骗局**：一位成员透露，一个被大肆宣传的 **AI 工具** 是个骗局，它通过一个声称可与 **Claude 3.5** 或 **GPT-4** 媲美的私有模型来误导用户。
   - 成员们对这类骗局造成的干扰表示担忧，有人指出这导致了 **时间损失**，且相关讨论在各个平台蔓延。
- **探索 RAG APIs**：一位成员询问了使用 **RAG APIs** 的经验，表示由于自己的模型尚未准备好，某个项目急需支持。
   - 他们正在寻找替代方案，以避免与 **24/7 托管** 相关的成本，突显了管理 AI 项目的实际挑战。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1282721477350654087)** (2 messages): 

> - `H100 加载支持`
> - `8-bit 模型加载` 


- **关于 H100 的 8-bit 支持问题**：一位成员询问为什么 **H100** 不支持以 **8-bit** 格式加载模型。
   - 他们询问是否有人了解关于此限制的信息。
- **寻求 H100 局限性的答案**：同一位成员迫切想知道是否存在导致 **H100** 缺乏 **8-bit** 模型加载支持的已知原因。
   - 他们再次请求社区提供见解或解释。

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1282275236901027881)** (21 messages🔥): 

> - `Factory Network x Tech: Berlin AI Hackathon`
> - `Finegrain Object Cutter`
> - `Concrete ML and Homomorphic Encryption`
> - `Open Source AI Event by GitHub` 


- **加入 Factory Network x Tech: Berlin AI Hackathon**：**Factory Network x Tech: Berlin AI Hackathon** 定于 **9月28-29日** 在 Factory Berlin Mitte 举行，面向 50-100 名渴望利用 AI 进行创新的雄心勃勃的构建者。
   - 参与者可以在专注于 **AI-driven innovations** 的协作环境中改进产品或启动新想法。
- **Finegrain 发布开源 Image Segmentation 模型**：Finegrain 发布了一个新的 **Image Segmentation Model**，其性能优于闭源 API，并以 **MIT License** 在 Hugging Face 上开源。
   - 他们正在努力添加一种更精细的 Prompting 方法，以增强除基本 Bounding Boxes 之外的消歧能力。
- **探索用于加密模型的 Concrete ML**：关于 **Concrete ML** 的讨论显示，它需要 **Quantization Aware Training (QAT)** 才能与 Homomorphic Encryption 正常配合工作，这引发了对性能开销的担忧。
   - 成员们对文档主要集中在较小模型上表示怀疑，暗示了在扩展到更大网络时存在挑战。
- **GitHub 将主办 Open Source AI 小组讨论**：由 GitHub 主办的 **Open Source AI Panel** 定于 **9月19日** 在旧金山举行，嘉宾来自 **Ollama** 和 **Nous Research** 等多个 AI 组织。
   - 活动免费但需要注册，因为名额有限且需要审核。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/finegrain/finegrain-object-cutter">Finegrain Object Cutter - a Hugging Face Space by finegrain</a>: 未找到描述</li><li><a href="https://lu.ma/royrg8gx">Factory Network x {Tech: Berlin} AI Hackathon · Luma</a>: 你准备好将你的 AI 想法变为现实了吗？加入我们的 Factory Network x {Tech: Berlin} AI Hackathon，这是一个专为雄心勃勃的构建者设计的独家活动……</li><li><a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>: AI 正在迅速改变从软件开发、内容创作、Agentic workflows 等各个行业。这一转型的核心是 Open Source……</li><li><a href="https://github.com/zama-ai/concrete-ml">GitHub - zama-ai/concrete-ml: Concrete ML: Privacy Preserving ML framework using Fully Homomorphic Encryption (FHE), built on top of Concrete, with bindings to traditional ML frameworks.</a>: Concrete ML：使用 Fully Homomorphic Encryption (FHE) 的隐私保护 ML 框架，构建在 Concrete 之上，并带有传统 ML 框架的绑定。 - zama-ai/concrete-ml
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1281879226966081598)** (9 条消息🔥): 

> - `LLM 中的 Multimodality`
> - `Reflection-70B 性能声明`
> - `AI 骗局与欺诈`
> - `Tool Augmented Generation` 


- **Multimodality 引起关注**：一名成员强调了人们对 **Multimodality** 日益增长的兴趣，并以 **Meta AI transfusion** 和 **DeepMind RT-2** 的重大贡献为例。
   - 他们建议探索涉及 RAG、API 调用、网络搜索和 Python 解释器的 **tool augmented generation**。
- **Reflection-70B 被过度炒作**：关于 **Reflection-70B** 及其微调的声明被描述为言过其实，根据初步测试，其性能更接近 **Llama 3 70B** 和 **Qwen 2 72B**。
   - 成员们对其依赖标准化基准测试表示担忧，认为这反映了当前最先进（SOTA）模型在泛化和推理方面的缺陷，正如[这篇论文](https://arxiv.org/abs/2406.02061)中所讨论的。
- **关于 AI 骗局的讨论**：成员们对 AI/LLM 领域出现的**骗子**表示失望，并提到了像 **Siraj Raval** 这样的早期典型人物。
   - 一位成员评论说 **cryptobros** 正在入侵这一领域，这进一步证实了欺骗性行为的问题。
- **对 OthersideAI 声明的怀疑**：流传着一个关于 **OthersideAI** CEO **Matt Shumer** 的故事，他声称在中型模型上取得了突破，但据报道该消息不实。
   - 社区被敦促批判性地评估 AI 领域的各种大胆声明，并指出如果听起来好得令人难以置信，那么它很可能就是假的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/JJitsev/status/1832758733866222011">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：(又) 一个关于崛起与衰落的故事：Reflection-70B 发布并声称拥有强大的前沿 LLM 性能——依赖于 MMLU 等常用基准测试。它能处理揭示泛化能力的 AIW 问题吗...</li><li><a href="https://x.com/shinboson/status/1832933747529834747?t=cc2q2tZcRK2DK9DOqsKGUw&s=19">来自 𝞍 Shin Megami Boson 𝞍 (@shinboson) 的推文</a>：一个关于 AI 研究社区欺诈的故事：9 月 5 日，OthersideAI 的 CEO Matt Shumer 向世界宣布他们取得了突破，允许他们训练一个中型模型...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 条消息): 

erkinalp: https://arxiv.org/abs/2408.06292
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1281902062245908522)** (2 条消息): 

> - `LanceDB 集成`
> - `针对 dspy 的 Pull Request`
> - `GitHub 评审流程` 


- **提交 LanceDB 集成 PR**：一名成员提交了 [LanceDB 集成的 PR](https://github.com/stanfordnlp/dspy/pull/1444)，将其作为 Retriever 添加到项目中，以处理大型数据集。
   - 他们请求特定用户对该集成的评审过程提供反馈和修改建议。
- **呼吁 PR 评审**：同一名成员标记了另一位用户以促使其评审提交的 PR，并强调了任何必要修改的需求。
   - 这突显了项目的协作性质以及同行评审在功能增强中的重要性。



**提到的链接**：<a href="https://github.com/stanfordnlp/dspy/pull/1444">Lancedb Integration by PrashantDixit0 · Pull Request #1444 · stanfordnlp/dspy</a>：此 PR 添加了 LanceDB 作为 Retriever 以处理大型数据集。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1282125390302941284)** (26 条消息🔥): 

> - `GPT-3.5 的弃用`
> - `MIPROv2 错误`
> - `微调 LLMs`
> - `CookLangFormatter 问题`
> - `DSPy 中的检索模型` 


- **对 GPT-3.5 弃用的复杂感受**：成员们正在讨论 **GPT-3.5** 弃用后不同模型的使用体验，指出性能不一致，特别是像 **4o-mini** 这样的开放模型。
   - 一位用户建议使用顶级的闭源模型作为低级模型的教师，以提高一致性。
- **在 MIPROv2 中遇到 'NoneType' 错误**：一位用户报告在使用 **MIPROv2** 时遇到 `AttributeError`，表明 `GenerateModuleInstruction` 函数内部可能存在潜在问题。
   - 另一位成员建议问题可能出在 **CookLangFormatter** 代码中，引发了关于可能修复方案的讨论。
- **使用独特数据集微调小型 LLM**：一位成员分享了他们使用特殊的 **reflection** 数据集微调小型 LLM 的成功经验，该模型可在 Hugging Face 上进行交互。
   - 他们被问及所使用的数据集并提供了链接，同时鼓励其他人探索他们的发现。
- **探索 CookLangFormatter 的问题**：成员们讨论了 **CookLangFormatter** 类中的潜在问题，将错误来源缩小到其方法签名。
   - 在进行了一些修改后，一位用户报告了积极的结果，并建议有必要在 GitHub 上记录该问题以供未来参考。
- **询问 colpali 作为检索模型**：一位用户提出是否有人在 DSPy 模块中尝试过将 **colpali** 作为检索模型。
   - 这一询问反映了在 DSPy 框架内优化检索方法的持续探索。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/DavidFSWD/status/1832738133227770308">来自 fullstack (@DavidFSWD) 的推文</a>：它奏效了！HF spaces 展示了概念验证 &lt;Reflection&gt;&lt;/&gt; 标签，基于 Gemma 2 2.8B 的 LLM 基础微调，在 Maheswar 的 reflection 数据集上训练，仅用时两分钟...</li><li><a href="https://huggingface.co/forcemultiplier/fmx-reflective-2b">forcemultiplier/fmx-reflective-2b · Hugging Face</a>：无描述</li><li><a href="https://huggingface.co/mahiatlinux">mahiatlinux (Maheswar KK)</a>：无描述</li><li><a href="https://github.com/SylphAI-Inc/AdalFlow">GitHub - SylphAI-Inc/AdalFlow: AdalFlow: The “PyTorch” library to auto-optimize any LLM tasks.</a>：AdalFlow：用于自动优化任何 LLM 任务的 “PyTorch” 库。- SylphAI-Inc/AdalFlow
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1282518259299651687)** (6 条消息): 

> - `WebGPU PR #6304`
> - `WGPU 缓冲区限制提升`
> - `Rubicon ObjC 的依赖问题`
> - `时区变更通知` 


- **WebGPU PR #6304 是一个充满希望的开始**：一位成员强调了 **geohot** 的 [bring back webgpu](https://github.com/tinygrad/tinygrad/pull/6304) 的重要性，认为这是一个在 **Asahi Linux** 上运行良好的良好倡议。
   - 值得注意的是，该 Pull Request 有 **300 美元的悬赏**，表明了其在社区中的重要性。
- **WGPU 获得缓冲区限制提升**：**wgpu** 中的一个新标志允许增加 **每个 kernel 的缓冲区限制**，使其能够与 **Metal 的 32** 相匹配。
   - 这一变化可以为在该生态系统中工作的开发者增强性能和兼容性。
- **WGPU 中 ObjC 的挑战**：一位成员表达了挫败感，因为将 **wgpu** 作为依赖项会导致对 **rubicon_objc** 的依赖，特别是在 **macOS** 上。
   - 这种情绪引起了其他对 ObjC 复杂性有类似不满的人的共鸣。
- **会议日程变更**：一位成员宣布由于时间切换到 **香港时间**，今天将 **不举行会议**。
   - 这一调整表明该小组正在努力维持跨时区的有效沟通。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/6304">geohot 提交的 bring back webgpu [run_process_replay] · Pull Request #6304 · tinygrad/tinygrad</a>：这在 Asahi Linux 上可以运行！

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1281779303951892542)** (17 条消息🔥): 

> - `Multi-GPU Tensor Issues` (Multi-GPU Tensor 问题)
> - `PTX Compilation Time for Tinygrad` (Tinygrad 的 PTX 编译时间)
> - `GGUF PRs Status` (GGUF PR 状态)
> - `Const with dtype uchar` (dtype 为 uchar 的 Const)
> - `Model Performance with Sharding` (分片下的模型性能)


- **Multi-GPU Tensor 问题困扰开发者**：成员们对与 **multi-GPU** Tensor 操作相关的错误表示沮丧，包括一个指出所有 buffer 必须位于同一设备的 `AssertionError`。
   - 一位用户表示：*“我花了足够的时间……确信这个目标与 tinygrad 目前处理 multi-gpu Tensor 的方式是正交（不兼容）的。”*
- **MLPerf BERT 的 PTX 编译时间过长**：一位拥有 **H100** 和 **H200 SXM** GPU 的用户询问运行 **tinygrad MLPerf BERT** 预期的 PTX 编译时间，目前看起来非常耗时。
   - 另一位成员估计：*“在 tinybox 上可能需要 30 分钟左右？”*，这表明编译时间可能相当长。
- **GGUF PR 缺乏合并且路线图（Roadmap）不明确**：成员们对各种 **GGUF PR** 的状态表示担忧，这些 PR 似乎陷入停滞，且相关的悬赏（bounty）已经消失。
   - 一位用户询问是否有 GGUF 的 **roadmap**，强调需要明确项目的方向。
- **关于 dtype 为 uchar 的 Const 的疑问**：一位用户询问 **dtype uchar** 的常量是否可以接受 `-1` 作为参数，这暗示了潜在的类型限制。
   - 另一位成员推测：*“self.arg 永远不会被解释为 uchar -1……”*，暗示了变量解释方面的细微差别。
- **模型分片（Sharding）挑战**：讨论围绕在多个设备上对模型进行分片（sharding）相关的问题展开，特定的模型设置在单 GPU 上运行正常，但在分布式运行时失败。
   - 一位用户提到：*“George 反对（pushback）了我的变通方案（workaround）……”*，表明协作排查仍在进行中。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/22e33795785f6c72449480e380ffdc213b5c7bbc/examples/mlperf/training_submission_v4.1/tinycorp/benchmarks/bert/implementations/tinybox_green/run_and_time.sh#L22">tinygrad/examples/mlperf/training_submission_v4.1/tinycorp/benchmarks/bert/implementations/tinybox_green/run_and_time.sh at 22e33795785f6c72449480e380ffdc213b5c7bbc · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1281748599537143859)** (10 条消息🔥): 

> - `xLAM 系统提示词差异`
> - `LLaMA 的 Function Calling 文档`
> - `GitHub Pull Request 中的合并冲突`
> - `使用 VLLM 进行模型评估`
> - `Hammer-7b Handler Pull Request` 


- **xLAM 系统提示词差异解析**：成员们讨论了 **xLAM** 与其他 OSS 模型相比所使用的独特系统提示词，并指出这已在其 [Hugging Face 模型卡片](https://huggingface.co/Salesforce/xLAM-7b-fc-r#basic-usage-with-huggingface) 中记录。
   - 对话强调，如果相关信息可用，模型在有详细文档记录时会使用个性化提示词，从而偏离 BFCL 默认设置。
- **LLaMA 缺乏 Function Calling 文档**：大家认识到 **LLaMA** 模型没有提供关于 Function Calling 的文档，这在关于提示词格式的讨论中引起了成员们的质疑。
   - 澄清指出，**LLaMA** 被归类为提示词模型（prompt model），而处理 Function Calling 的差异可能源于其文档方法。
- **解决 GitHub Pull Request 冲突**：一位成员指出其 Pull Request [#625](https://github.com/ShishirPatil/gorilla/pull/625) 面临合并冲突，导致无法成功合并。
   - 在解决冲突后，他们重新提交了一个新的 Pull Request [#627](https://github.com/ShishirPatil/gorilla/pull/627)，以促进其贡献的集成。
- **使用 VLLM 评估模型**：一位用户询问在通过 **VLLM** 设置服务后如何评估自己的模型。
   - 对话反映了社区对模型评估技术和最佳实践的广泛兴趣。
- **引入 Hammer-7b Handler**：社区讨论了在 Pull Request 背景下引入 **Hammer-7b** handler 的情况，重点介绍了新功能和性能指标。
   - 文档包含一个详细的 [CSV 表格](https://github.com/ShishirPatil/gorilla/pull/625)，概述了模型的准确性和执行摘要。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Trelis/Meta-Llama-3-70B-Instruct-function-calling">Trelis/Meta-Llama-3-70B-Instruct-function-calling · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Salesforce/xLAM-7b-fc-r#basic-usage-with-huggingface),">Salesforce/xLAM-7b-fc-r · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ShishirPatil/gorilla/pull/625">[BFCL] add MadeAgents/Hammer-7b handler by linqq9 · Pull Request #625 · ShishirPatil/gorilla</a>：此 PR 添加了 MadeAgents/Hammer-7b。以下是转换为 Markdown 格式的 CSV 表格：总体准确率、模型 AST 摘要、执行摘要、简单 AST、多重 AST、并行 AST、并行多重 AST...</li><li><a href="https://github.com/ShishirPatil/gorilla/pull/627">[BFCL] add MadeAgents/Hammer-7b handler by linqq9 · Pull Request #627 · ShishirPatil/gorilla</a>：此 PR 添加了 MadeAgents/Hammer-7b。以下是转换为 Markdown 格式的 CSV 表格：总体准确率、模型 AST 摘要、执行摘要、简单 AST、多重 AST、并行 AST、并行多重 AST...
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1281734821395304553)** (2 条消息): 

> - `4090 GPU 能力`
> - `使用 Milvus 进行混合搜索`
> - `Embedding 模型`
> - `重排序元数据` 


- **4090 GPU 可以处理更大的模型**：使用 **4090 GPU**，你应该能够与 **Llama-8b** 同时运行一个更大的 Embedding 模型，并建议考虑使用 **3.1 版本**。
   - 这为增强模型性能和提高任务处理效率开辟了可能性。
- **利用 Milvus 进行混合搜索**：讨论指向在 Milvus 上结合 **BGE** 和 **BM25** 使用混合搜索，并参考了 [GitHub 仓库](https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py) 中的示例。
   - 该示例展示了如何高效地结合稀疏（sparse）和稠密（dense）混合搜索。
- **使用元数据进行重排序**：如果你为每个 chunk 都有元数据，实现一个 **reranker** 将有效帮助进一步排序和过滤结果。
   - 该策略旨在优化数据处理，提高检索信息的检索相关性。



**提及的链接**：<a href="https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py">pymilvus/examples/hello_hybrid_sparse_dense.py at master · milvus-io/pymilvus</a>：Milvus 的 Python SDK。通过在 GitHub 上创建账户为 milvus-io/pymilvus 的开发做出贡献。

  

---

### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1282773662272848023)** (1 messages): 

> - `基于 RAG 的检索`
> - `RAG 的评估指标`
> - `RAG 与其他 LLM 的对比分析` 


- **理解基于 RAG 的检索评估**：一位成员询问了在特定领域背景下评估 **基于 RAG 的检索** 系统所需的 **评估指标**。
   - 他们表示不确定是应该将自己的 **RAG 方法** 与其他 **LLM** 进行比较，还是与不使用 RAG 的结果进行评估。
- **RAG 的对比策略**：同一位成员思考是仅进行使用与不使用 RAG 的对比，还是也要与其他 **大语言模型 (LLM)** 进行对比。
   - 这个问题引起了兴趣，成员们考虑了各种评估 RAG 在其项目中有效性的方法。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1282799151092072510)** (1 messages): 

> - `开源 AI`
> - `GitHub 小组讨论活动`
> - `讨论嘉宾` 


- **GitHub 举办开源 AI 小组讨论**：GitHub 将于下周四 (9/19) 在其旧金山办公室举办一场免费的 [开源 AI 小组讨论](https://lu.ma/wbc5bx0z)，邀请所有人注册参加。
   - 讨论嘉宾包括来自 **Ollama**、**Nous Research**、**Black Forest Labs** 和 **Unsloth AI** 的代表，探讨开源对 **AI** 的访问、民主化及其影响。
- **注册需经批准**：参与者必须注册该活动，注册需经主办方批准。
   - 随着该活动在 AI 社区引起关注，这一要求旨在有效地管理出席人数。



**提及的链接**：<a href="https://lu.ma/wbc5bx0z">GitHub Presents: Open Source AI - Access, Democratization, and Responsibility · Luma</a>：AI 正在迅速改变从软件开发、内容创作、Agent 工作流等各个行业。这一转型的核心是开源……

  

---



---



---



{% else %}


> 完整的频道细分内容已针对邮件进行截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}